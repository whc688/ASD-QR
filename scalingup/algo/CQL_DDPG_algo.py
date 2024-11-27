from __future__ import annotations

import logging
import os
import pickle
import typing
from copy import deepcopy
from time import time
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Union
import dataclasses

import hydra
import numpy as np
import torch
import torch.nn as nn

from scalingup.algo.single_action_critic_backbone import CriticLinearNet
from scalingup.algo.single_action_actor_backbone import LinearNet

from scalingup.algo.state_encoder.state_sequence_encoder import StateSequenceEncoder

from scalingup.algo.utils import replace_bn_with_gn
from scalingup.data.dataset import TensorDataClass
from scalingup.data.window_dataset import (
    NormalizationConfig,
    PolicyWindowRolloutConfig,
    StateSequenceTensor,
    RLTrajectoryWindowDataset,
    RLTrajectoryWindowTensor,
)
from scalingup.policy.scalingup import ScalingUpDataGen
from scalingup.utils.core import (
    ControlAction,
    ControlConfig,
    Observation,
    ObservationWithHistory,
    PartialObservation,
    Policy,
    RayPolicy,
    Task,
    split_state_phrase,
)
from transforms3d import quaternions, euler
from pytorch_lightning import LightningModule
from scalingup.algo.diffusion import DiffusionScalingUpAlgo, DiffusionPolicy

GRADIENT_EXPLOSION_THRESHOLD = 100


class CqlDdpgTrainingAlgo(LightningModule):
    def __init__(
            self,
            actor_state_sequence_encoder: StateSequenceEncoder,
            actor_backbone: LinearNet,
            critic_state_sequence_encoder: StateSequenceEncoder,
            critic_backbone: CriticLinearNet,
            action_dim: int,
            ctrl_config: ControlConfig,
            replay_buffer: RLTrajectoryWindowDataset,
            critic_optimizer: Any,
            actor_optimizer: Any,
            rollout_config: PolicyWindowRolloutConfig,
            critic_lr_scheduler: Optional[Any] = None,
            actor_lr_scheduler: Optional[Any] = None,
            log_grad_frequence: int = 50,
            gamma: float = 0.98,
            tau: float = 0.005,
            alpha: float = 1,
            num_validation_batches: int = 16,
            float32_matmul_precision: str = "high",
            # not used
            should_condition_on_task_metric: bool = False,
            policy_task_metric_guidance: float = 0.0,
            task_metric_corrupt_prob: float = 0.0,
            policy_suppress_token: float = -1.0,
            use_remote_policy: bool = False,
    ):
        super().__init__()
        # 由于使用两个 optimizer，所以需要关闭 automatic_optimization
        self.automatic_optimization = False

        self.replay_buffer = replay_buffer
        self.critic_lr_scheduler_partial = critic_lr_scheduler
        self.actor_lr_scheduler_partial = actor_lr_scheduler
        self.critic_optimizer_partial = critic_optimizer
        self.actor_optimizer_partial = actor_optimizer
        torch.set_float32_matmul_precision(float32_matmul_precision)
        logging.info(f"Using {float32_matmul_precision} precision training")
        if float32_matmul_precision == "medium":
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            logging.info("Use `float32_matmul_precision=medium` for faster training")
        self.action_dim = action_dim
        # BN 适合大 batch size, 通常大于 32，GN 适合小 batch size，通常小于 32，这里我们先尝试 BN
        # TODO： norm 尝试
        # state_sequence_encoder = replace_bn_with_gn(state_sequence_encoder)
        # critic_backbone = replace_bn_with_gn(critic_backbone)
        # 使用两个 critic 网络，每个 critic 网络都有一个 target 网络, 两个 critic 使用不同的初始化参数
        # 检查 critic 网络的初始化参数是否不同： 没问题， 是不同的
        self.critic = nn.ModuleDict(
            {
                "state_sequence_encoder": critic_state_sequence_encoder,
                "critic_backbone": critic_backbone,
            }
        )
        self.actor = nn.ModuleDict(
            {
                "state_sequence_encoder": actor_state_sequence_encoder,
                "actor_backbone": actor_backbone,
            }
        )
        self.critic_target = deepcopy(self.critic)
        # freeze
        for param in self.critic_target.parameters():
            param.requires_grad = False
        self.gamma = gamma
        self.alpha = alpha
        self.log_grad_frequence = log_grad_frequence

        self.rollout_config = rollout_config
        logging.info(str(self.rollout_config))
        self.tau = tau
        self.ctrl_config = ctrl_config
        if len(self.window_replay_buffer) > 0:
            assert (
                    self.ctrl_config.frequency == self.window_replay_buffer.control_frequency
            )
        self.num_validation_batches = num_validation_batches

        self.validation_batches: Dict[str, List[RLTrajectoryWindowTensor]] = {}
        self.validation_batches["sim"] = self.get_validation_batches(
            self.num_validation_batches
        )

    def train_dataloader(self):
        return self.replay_buffer.get_loader()

    def configure_optimizers(self):
        actor_params = list(self.actor.parameters())
        actor_optimizer = self.actor_optimizer_partial(params=actor_params)

        critic_params = list(self.critic.parameters())
        critic_optimizer = self.critic_optimizer_partial(params=critic_params)
        if self.critic_lr_scheduler_partial is None:
            raise NotImplementedError("critic_lr_scheduler_partial is None")
        # return [critic_optimizer], [
        #     {
        #         "scheduler": self.critic_lr_scheduler_partial(optimizer=critic_optimizer),
        #         "interval": "step",
        #     }
        # ],
        return [actor_optimizer, critic_optimizer], [
            {
                "scheduler": self.actor_lr_scheduler_partial(optimizer=actor_optimizer),
                "interval": "step",
            },
            {
                "scheduler": self.critic_lr_scheduler_partial(optimizer=critic_optimizer),
                "interval": "step",
            }
        ],

    def training_step(self, tensor_data: TensorDataClass, batch_idx: int):
        stats = self.get_stats(tensor_data.to(device=self.device, non_blocking=True))
        total_loss = sum(v for k, v in stats.items() if "loss" in k)
        self.log_dict(
            {f"train/{k}": v for k, v in stats.items()},
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.replay_buffer.batch_size,
        )
        return total_loss

    def get_stats(self, tensor_data: TensorDataClass) -> Dict[str, Any]:
        actor_optimizer, critic_optimizer = self.optimizers()
        actor_lr_scheduler, critic_lr_scheduler = self.lr_schedulers()

        # critic
        self.critic.train()
        traj_window = typing.cast(RLTrajectoryWindowTensor, tensor_data).to(dtype=self.dtype)

        '''get current Q value'''
        state_features = self.critic["state_sequence_encoder"](
            state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)

        # prepare action sequence
        normalized_action = self.action_norm_config.normalize(
            traj_window.action_sequence.tensor).to(device=self.device)
        # 只取当前步的动作
        normalized_action = normalized_action[:, self.rollout_config.proprio_obs_horizon - 1, :]
        current_q = self.critic["critic_backbone"](
            action=normalized_action, global_cond=state_features)

        '''get target Q value'''
        with torch.no_grad():
            self.critic_target.eval()

            next_actor_state_features = self.actor["state_sequence_encoder"](
                state_sequence=traj_window.next_state_sequence, task_names=traj_window.task_names)
            # 返回的是 normalized action
            next_action = self.actor["actor_backbone"](global_cond=next_actor_state_features)

            # next_action = self.actor.diffuse(
            #     state_sequence=traj_window.next_state_sequence,
            #     task_names=traj_window.task_names,
            #     task_metric=traj_window.task_metrics)

            next_target_state_features = self.critic_target["state_sequence_encoder"](
                state_sequence=traj_window.next_state_sequence, task_names=traj_window.task_names)

            target_q = self.critic_target["critic_backbone"](
                action=next_action, global_cond=next_target_state_features)

        target_q = traj_window.reward + self.gamma * (1 - traj_window.terminated) * target_q
        target_q = target_q.detach()

        critic_mse_loss = nn.functional.mse_loss(current_q, target_q)

        # 计算 CQL loss
        actor_predict_state_features = self.actor["state_sequence_encoder"](
            state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
        # 返回的是 normalized action
        actor_predict_action = self.actor["actor_backbone"](global_cond=actor_predict_state_features)

        actor_predict_action_state_feature = self.critic["state_sequence_encoder"](
            state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)

        actor_predict_action_q = self.critic["critic_backbone"](
            action=actor_predict_action, global_cond=actor_predict_action_state_feature)

        logsumexp = torch.logsumexp(actor_predict_action_q, dim=1, keepdim=True)

        cql_loss = self.alpha * (logsumexp - current_q).mean()

        critic_loss = cql_loss + critic_mse_loss / 2
        self.log("train/critic_mse_loss", critic_mse_loss,
                 sync_dist=True,
                 on_step=True,
                 on_epoch=True,
                 batch_size=self.replay_buffer.batch_size, )
        self.log("train/cql_loss", cql_loss,
                 sync_dist=True,
                 on_step=True,
                 on_epoch=True,
                 batch_size=self.replay_buffer.batch_size, )
        self.log("train/critic_loss", critic_loss,
                 sync_dist=True,
                 on_step=True,
                 on_epoch=True,
                 batch_size=self.replay_buffer.batch_size, )

        critic_optimizer.zero_grad()
        self.manual_backward(critic_loss)
        critic_optimizer.step()

        # actor
        # 计算 actor 产生的动作
        actor_predict_state_features = self.actor["state_sequence_encoder"](
            state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
        # 返回的是 normalized action
        actor_predict_action = self.actor["actor_backbone"](global_cond=actor_predict_state_features)

        # 根据 actor 产生的动作和状态特征计算 Q 值
        actor_predict_action_state_feature = self.critic["state_sequence_encoder"](
            state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
        actor_predict_action_q = self.critic["critic_backbone"](
            action=actor_predict_action, global_cond=actor_predict_action_state_feature)

        actor_loss = -actor_predict_action_q.mean()
        self.log("train/actor_loss", actor_loss,
                 sync_dist=True,
                 on_step=True,
                 on_epoch=True,
                 batch_size=self.replay_buffer.batch_size, )

        actor_optimizer.zero_grad()
        self.manual_backward(actor_loss)
        actor_optimizer.step()

        actor_lr_scheduler.step()
        critic_lr_scheduler.step()
        loss = critic_loss + actor_loss
        return {"train/total_loss": loss}

    # def optimizer_step(
    #         self,
    #         epoch,
    #         batch_idx,
    #         optimizer,
    #         optimizer_closure,
    # ):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)
    #     super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        # 计算梯度范数并记录
        if self.global_step % self.log_grad_frequence == 0:  # 每 self.log_grad_frequence 步检查一次
            max_grad_norm = 0
            max_grad_norm_name = ""
            for name, param in self.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2)
                    if grad_norm > max_grad_norm:
                        max_grad_norm = grad_norm
                        max_grad_norm_name = name

            # 使用 wandb 记录最大梯度范数和对应的参数名
            self.log("train/max_grad_norm", max_grad_norm, on_step=True, on_epoch=False)
            # self.log("train/max_grad_norm_param", max_grad_norm_name, on_step=True, on_epoch=False)
            if max_grad_norm > GRADIENT_EXPLOSION_THRESHOLD:
                logging.warning(
                    f"Gradient explosion detected: {max_grad_norm} > {GRADIENT_EXPLOSION_THRESHOLD}, name"
                    f" {max_grad_norm_name}"
                )
        for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
        ):
            if param.requires_grad:
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def on_train_epoch_end(self):
        if self.num_validation_batches > 0:
            validation_stats: Dict[str, List[float]] = {}
            for validation_key, batches in self.validation_batches.items():
                for i in range(len(batches)):
                    batches[i] = batches[i].to(self.device, self.dtype, non_blocking=True)
                for batch in batches:
                    for stat_key, stat_value in self.get_validation_stats(
                            traj_window=batch,
                    ).items():
                        full_key = os.path.join(validation_key, stat_key)
                        if full_key not in validation_stats:
                            validation_stats[full_key] = []
                        validation_stats[full_key].append(stat_value.cpu().numpy())
            self.log_dict(
                {
                    f"validation/{k}": float(np.mean(v))
                    for k, v in validation_stats.items()
                },
                sync_dist=True,
                on_epoch=True,
                batch_size=self.replay_buffer.batch_size,
            )
        else:
            raise ValueError("num_validation_batches must be > 0")

        self.log_dict(
            dictionary={
                f"replay_buffer/{k}": v for k, v in self.replay_buffer.summary.items()
            },
            on_epoch=True,
            sync_dist=True,
        )

    def get_validation_batches(
            self, num_batches: int, replay_buffer: Optional[RLTrajectoryWindowDataset] = None
    ):
        if replay_buffer is None:
            replay_buffer = self.replay_buffer
        logging.info("pre-loading validation batches")
        batches = []
        for batch in replay_buffer.get_loader(
                num_steps_per_update=num_batches,
                batch_size=1,
                persistent_workers=False,
                pin_memory=False,
                repeat=False,
                shuffle=True,
        ):
            batches.append(batch)
            if len(batches) >= self.num_validation_batches:
                break
        logging.info(f"pre-loaded {len(batches)} validation batches")
        return batches

    def get_validation_stats(
            self,
            traj_window: RLTrajectoryWindowTensor,
    ) -> Dict[str, float]:
        # 设置为 eval 模式
        self.critic.eval()
        self.actor.eval()
        with torch.no_grad():
            if not traj_window.is_batched:
                traj_window = RLTrajectoryWindowTensor.collate([traj_window])

            '''get current Q value'''
            state_features = self.critic["state_sequence_encoder"](
                state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)

            # prepare action sequence
            normalized_action = self.action_norm_config.normalize(
                traj_window.action_sequence.tensor).to(device=self.device)
            normalized_action = normalized_action[:, self.rollout_config.proprio_obs_horizon - 1, :]
            current_q = self.critic["critic_backbone"](
                action=normalized_action, global_cond=state_features)

            '''get target Q value'''
            with torch.no_grad():
                self.critic_target.eval()

                next_actor_state_features = self.actor["state_sequence_encoder"](
                    state_sequence=traj_window.next_state_sequence, task_names=traj_window.task_names)
                # 返回的是 normalized action
                next_action = self.actor["actor_backbone"](global_cond=next_actor_state_features)

                next_target_state_features = self.critic_target["state_sequence_encoder"](
                    state_sequence=traj_window.next_state_sequence, task_names=traj_window.task_names)

                target_q = self.critic_target["critic_backbone"](
                    action=next_action, global_cond=next_target_state_features)

            target_q = traj_window.reward + self.gamma * (1 - traj_window.terminated) * target_q
            target_q = target_q.detach()

            critic_mse_loss = nn.functional.mse_loss(current_q, target_q)

            # 计算 CQL loss
            actor_predict_state_features = self.actor["state_sequence_encoder"](
                state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
            # 返回的是 normalized action
            actor_predict_action = self.actor["actor_backbone"](global_cond=actor_predict_state_features)

            actor_predict_action_state_feature = self.critic["state_sequence_encoder"](
                state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)

            actor_predict_action_q = self.critic["critic_backbone"](
                action=actor_predict_action, global_cond=actor_predict_action_state_feature)

            logsumexp = torch.logsumexp(actor_predict_action_q, dim=1, keepdim=True)

            cql_loss = self.alpha * (logsumexp - current_q).mean()

            critic_loss = cql_loss + critic_mse_loss / 2
            self.log("validation/critic_mse_loss", critic_mse_loss,
                     sync_dist=True,
                     on_epoch=True,
                     batch_size=self.replay_buffer.batch_size, )
            self.log("validation/cql_loss", cql_loss,
                     sync_dist=True,
                     on_epoch=True,
                     batch_size=self.replay_buffer.batch_size, )
            self.log("validation/critic_loss", critic_loss,
                     sync_dist=True,
                     on_epoch=True,
                     batch_size=self.replay_buffer.batch_size, )

            # actor
            # 计算 actor 产生的动作
            actor_predict_state_features = self.actor["state_sequence_encoder"](
                state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
            # 返回的是 normalized action
            actor_predict_action = self.actor["actor_backbone"](global_cond=actor_predict_state_features)

            # 根据 actor 产生的动作和状态特征计算 Q 值
            actor_predict_action_state_feature = self.critic["state_sequence_encoder"](
                state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
            actor_predict_action_q = self.critic["critic_backbone"](
                action=actor_predict_action, global_cond=actor_predict_action_state_feature)

            actor_loss = -actor_predict_action_q.mean()
            self.log("validation/actor_loss", actor_loss,
                     sync_dist=True,
                     on_epoch=True,
                     batch_size=self.replay_buffer.batch_size, )
            loss = critic_loss + actor_loss

        self.critic.train()
        self.actor.train()
        return {"validation/loss": loss}

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if os.environ.get("TORCH_COMPILE", "0") == "0":
            # need to handle torch compile, for instance:
            # critic_backbone._orig_mod.final_conv.1.bias
            # critic_backbone.final_conv.1.bias
            checkpoint["state_dict"] = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["state_dict"].items()
            }
        action_norm_config: NormalizationConfig = checkpoint["action_norm_config"]
        self.window_replay_buffer.min_action_values = action_norm_config.min_value
        self.window_replay_buffer.max_action_values = action_norm_config.max_value

        # 这里需要处理两个 state_sequence_encoder 的参数和 target 参数
        # critic
        critic_keys = [k for k in checkpoint["state_dict"].keys() if "lang_embs_cache" in k and "critic." in k]
        critic_texts = [k.split("lang_embs_cache.")[-1] for k in critic_keys]
        logging.info(
            f"Loading {len(critic_texts)} cached text features for critic: "
            + ", ".join(f"{t!r}" for t in critic_texts)
        )
        self.critic["state_sequence_encoder"].lang_embs_cache = torch.nn.ParameterDict(
            {
                text: torch.nn.Parameter(checkpoint["state_dict"][k], requires_grad=False)
                for text, k in zip(critic_texts, critic_keys)
            }
        )
        # critic_target
        critic_target_keys = [k for k in checkpoint["state_dict"].keys() if
                               "lang_embs_cache" in k and "critic_target." in k]
        critic_target_texts = [k.split("lang_embs_cache.")[-1] for k in critic_target_keys]
        logging.info(
            f"Loading {len(critic_target_texts)} cached text features for critic_target: "
            # + ", ".join(f"{t!r}" for t in critic_target_texts)
        )
        self.critic_target["state_sequence_encoder"].lang_embs_cache = torch.nn.ParameterDict(
            {
                text: torch.nn.Parameter(checkpoint["state_dict"][k], requires_grad=False)
                for text, k in zip(critic_target_texts, critic_target_keys)
            }
        )

        # actor
        actor_keys = [k for k in checkpoint["state_dict"].keys() if "lang_embs_cache" in k and "actor." in k]
        actor_texts = [k.split("lang_embs_cache.")[-1] for k in actor_keys]
        logging.info(
            f"Loading {len(actor_texts)} cached text features for actor: "
            + ", ".join(f"{t!r}" for t in actor_texts)
        )
        self.actor["state_sequence_encoder"].lang_embs_cache = torch.nn.ParameterDict(
            {
                text: torch.nn.Parameter(checkpoint["state_dict"][k], requires_grad=False)
                for text, k in zip(actor_texts, actor_keys)
            }
        )

        return super().on_load_checkpoint(checkpoint)

    # 这个好像没用到
    # def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
    #     retval = super().load_state_dict(state_dict)
    #     return retval

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["action_norm_config"] = self.action_norm_config

    @property
    def action_norm_config(self) -> NormalizationConfig:
        return self.window_replay_buffer.action_norm_config

    @property
    def window_replay_buffer(self) -> RLTrajectoryWindowDataset:
        assert type(self.replay_buffer) is RLTrajectoryWindowDataset
        return typing.cast(RLTrajectoryWindowDataset, self.replay_buffer)

    @property
    def dtype(self) -> torch.dtype:
        return typing.cast(torch.dtype, super().dtype)


class CqlDdpgPolicy(Policy):
    def __init__(
            self,

    ):