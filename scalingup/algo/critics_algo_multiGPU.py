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

from scalingup.algo.critic_backbone import ConditionalCNN1D
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

class CriticTarget():
    def __init__(self, critic1_target: nn.ModuleDict, critic2_target: nn.ModuleDict):
        self.critic1_target = critic1_target
        self.critic2_target = critic2_target
        # freeze
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False


class CriticTrainingAlgo(LightningModule):
    def __init__(
            self,
            state_sequence_encoder1: StateSequenceEncoder,
            state_sequence_encoder2: StateSequenceEncoder,
            critic_backbone1: ConditionalCNN1D,
            critic_backbone2: ConditionalCNN1D,
            action_dim: int,
            ctrl_config: ControlConfig,
            replay_buffer: RLTrajectoryWindowDataset,
            optimizer: Any,
            rollout_config: PolicyWindowRolloutConfig,
            max_q_backup: bool,
            actor_checkpoint_path: str,
            monte_carlo: bool,
            actor_inference_device: Union[str, torch.device] = "cuda",
            lr_scheduler: Optional[Any] = None,
            log_grad_frequence: int = 50,
            gamma: float = 0.98,
            tau: float = 0.005,
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
        self.replay_buffer = replay_buffer
        self.lr_scheduler_partial = lr_scheduler
        self.optimizer_partial = optimizer
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
        # 这里采用 TD3 的方法，使用两个 critic 网络，每个 critic 网络都有一个 target 网络, 两个 critic 使用不同的初始化参数
        # 检查 critic 网络的初始化参数是否不同： 没问题， 是不同的
        self.critic1 = nn.ModuleDict(
            {
                "state_sequence_encoder": state_sequence_encoder1,
                "critic_backbone": critic_backbone1,
            }
        )
        self.critic2 = nn.ModuleDict(
            {
                "state_sequence_encoder": state_sequence_encoder2,
                "critic_backbone": critic_backbone2,
            }
        )
        self.critic_target = CriticTarget(
            critic1_target=deepcopy(self.critic1),
            critic2_target=deepcopy(self.critic2)
        )
        self.actor = self.load_actor(actor_checkpoint_path, self.device)  # DiffusionPolicy
        self.max_q_backup = max_q_backup
        self.gamma = gamma
        self.monte_carlo = monte_carlo
        self.log_grad_frequence = log_grad_frequence
        self.train_loss_list = []

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

    def load_actor(self, path: str, inference_device: Union[str, torch.device]) -> DiffusionPolicy:
        kwargs = dict(
            remote=False,
            task_metric_guidance=None,
            suppress_token=None,
            towards_token=None,
            num_inference_steps=None,
            action_horizon=None,
        )
        conf = pickle.load(
            open(
                os.path.dirname(os.path.dirname(path)) + "/conf.pkl",
                "rb",
            )
        )
        # conf.algo.replay_buffer.should_reindex = False
        conf.algo.replay_buffer.rootdir = '/home/kemove/scalingup_original/success_data'
        algo_kwargs = {
            k: (hydra.utils.instantiate(v) if "Config" in type(v).__name__ else v)
            for k, v in conf.algo.items()
            if not (k.endswith("_") and k.startswith("_"))
        }
        algo_kwargs["device"] = "cpu"
        algo = DiffusionScalingUpAlgo.load_from_checkpoint(
            **algo_kwargs, strict=False, checkpoint_path=path, map_location="cpu"
        )
        policy = deepcopy(
            algo.get_policy(
                inference_device=inference_device,
                **kwargs,
            )
        )
        logging.info(f"loaded from checkpoint: {path!r}")
        return policy

    def train_dataloader(self):
        return self.replay_buffer.get_loader()

    def configure_optimizers(self):
        critic1_params = list(self.critic1.parameters())
        critic2_params = list(self.critic2.parameters())
        params = critic1_params + critic2_params
        optimizer = self.optimizer_partial(params=params)
        if self.lr_scheduler_partial is None:
            return optimizer
        return [optimizer], [
            {
                "scheduler": self.lr_scheduler_partial(optimizer=optimizer),
                "interval": "step",
            }
        ]

    def training_step(self, tensor_data: TensorDataClass, batch_idx: int):

        self.actor.noise_pred_net.to(device=self.device)
        self.actor.state_sequence_encoder.to(device=self.device)
        self.actor.device = self.device
        self.critic_target.critic1_target.to(device=self.device)
        self.critic_target.critic2_target.to(device=self.device)

        stats = self.get_stats(tensor_data.to(device=self.device, non_blocking=True))
        total_loss = sum(v for k, v in stats.items() if "loss" in k)
        self.train_loss_list.append(total_loss.item())
        self.log_dict(
            {f"train/{k}": v for k, v in stats.items()},
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.replay_buffer.batch_size,
        )
        return total_loss

    def get_stats(self, tensor_data: TensorDataClass) -> Dict[str, Any]:
        self.critic1.train()
        self.critic2.train()
        action_start = self.rollout_config.proprio_obs_horizon - 1
        action_end = action_start + self.rollout_config.action_horizon

        traj_window = typing.cast(RLTrajectoryWindowTensor, tensor_data).to(dtype=self.dtype)

        '''get current Q value'''
        state_features1 = self.critic1["state_sequence_encoder"](
            state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
        state_features2 = self.critic2["state_sequence_encoder"](
            state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
        # prepare action sequence
        normalized_action = self.action_norm_config.normalize(
            traj_window.action_sequence.tensor).to(device=self.device)

        normalized_action = normalized_action[:, action_start:action_end, :]

        current_q1 = self.critic1["critic_backbone"](
            action=normalized_action, global_cond=state_features1)
        current_q2 = self.critic2["critic_backbone"](
            action=normalized_action, global_cond=state_features2)

        '''get target Q value'''
        with torch.no_grad():
            self.critic1_target.eval()
            self.critic2_target.eval()
            self.actor.state_sequence_encoder.eval()
            self.actor.noise_pred_net.eval()
            if self.max_q_backup:
                target_q1 = []
                target_q2 = []
                for _ in range(10):
                    next_action = self.actor.diffuse(   # 返回的是 unnormalized action
                        state_sequence=traj_window.next_state_sequence,
                        task_names=traj_window.task_names,
                        task_metric=traj_window.task_metrics)

                    next_action = next_action[:, action_start:action_end, :]

                    next_target_state_features1 = self.critic1_target["state_sequence_encoder"](
                        state_sequence=traj_window.next_state_sequence, task_names=traj_window.task_names)
                    next_target_state_features2 = self.critic2_target["state_sequence_encoder"](
                        state_sequence=traj_window.next_state_sequence, task_names=traj_window.task_names)
                    next_target_normalized_action = self.action_norm_config.normalize(
                        next_action).to(device=self.device)
                    target_q1.append(self.critic1_target["critic_backbone"](
                        action=next_target_normalized_action, global_cond=next_target_state_features1))
                    target_q2.append(self.critic2_target["critic_backbone"](
                        action=next_target_normalized_action, global_cond=next_target_state_features2))
                # (batch_size, 10, 1)
                target_q1 = torch.stack(target_q1, dim=1)
                target_q2 = torch.stack(target_q2, dim=1)
                # (batch_size, 1)
                target_q1 = target_q1.max(dim=1, keepdim=True)[0].squeeze(1)
                target_q2 = target_q2.max(dim=1, keepdim=True)[0].squeeze(1)
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.actor.diffuse(
                    state_sequence=traj_window.next_state_sequence,
                    task_names=traj_window.task_names,
                    task_metric=traj_window.task_metrics)

                next_action = next_action[:, action_start:action_end, :]

                next_target_state_features1 = self.critic1_target["state_sequence_encoder"](
                    state_sequence=traj_window.next_state_sequence, task_names=traj_window.task_names)
                next_target_state_features2 = self.critic2_target["state_sequence_encoder"](
                    state_sequence=traj_window.next_state_sequence, task_names=traj_window.task_names)
                next_target_normalized_action = self.action_norm_config.normalize(
                    next_action).to(device=self.device)
                target_q1 = self.critic1_target["critic_backbone"](
                    action=next_target_normalized_action, global_cond=next_target_state_features1)
                target_q2 = self.critic2_target["critic_backbone"](
                    action=next_target_normalized_action, global_cond=next_target_state_features2)
                target_q = torch.min(target_q1, target_q2)

        target_q = traj_window.reward + self.gamma * (1 - traj_window.terminated) * target_q

        if self.monte_carlo:
            # target_q = torch.max(target_q, traj_window.task_metrics).detach()
            target_q = ((target_q + traj_window.task_metrics) / 2).detach()
        else:
            target_q = target_q.detach()

        loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(current_q2, target_q)
        return {"train/loss": loss}

    def optimizer_step(
            self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_closure,
    ):
        for name, param in self.named_parameters():
            if param.grad is None:
                print(name)
        optimizer.step(closure=optimizer_closure)

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
                self.critic1.parameters(), self.critic1_target.parameters()
        ):
            if param.requires_grad:
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(
                self.critic2.parameters(), self.critic2_target.parameters()
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

        val_loss = np.array(validation_stats['sim/validation/loss']).mean()
        train_loss = self.train_loss_list[-1]
        loss_diff = val_loss - train_loss
        self.log("loss_difference/validation-train", loss_diff, on_epoch=True, sync_dist=True)
        self.train_loss_list = []

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
        self.critic1.eval()
        self.critic2.eval()
        action_start = self.rollout_config.proprio_obs_horizon - 1
        action_end = action_start + self.rollout_config.action_horizon

        with torch.no_grad():
            if not traj_window.is_batched:
                traj_window = RLTrajectoryWindowTensor.collate([traj_window])

            '''get current Q value'''
            state_features1 = self.critic1["state_sequence_encoder"](
                state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
            state_features2 = self.critic2["state_sequence_encoder"](
                state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
            # prepare action sequence
            normalized_action = self.action_norm_config.normalize(
                traj_window.action_sequence.tensor).to(device=self.device)

            normalized_action = normalized_action[:, action_start:action_end, :]

            current_q1 = self.critic1["critic_backbone"](
                action=normalized_action, global_cond=state_features1)
            current_q2 = self.critic2["critic_backbone"](
                action=normalized_action, global_cond=state_features2)

            '''get target Q value'''
            with torch.no_grad():
                self.critic1_target.eval()
                self.critic2_target.eval()
                self.actor.state_sequence_encoder.eval()
                self.actor.noise_pred_net.eval()
                if self.max_q_backup:
                    target_q1 = []
                    target_q2 = []
                    for _ in range(10):
                        next_action = self.actor.diffuse(
                            state_sequence=traj_window.next_state_sequence,
                            task_names=traj_window.task_names,
                            task_metric=traj_window.task_metrics)

                        next_action = next_action[:, action_start:action_end, :]

                        next_target_state_features1 = self.critic1_target["state_sequence_encoder"](
                            state_sequence=traj_window.next_state_sequence, task_names=traj_window.task_names)
                        next_target_state_features2 = self.critic2_target["state_sequence_encoder"](
                            state_sequence=traj_window.next_state_sequence, task_names=traj_window.task_names)
                        next_target_normalized_action = self.action_norm_config.normalize(
                            next_action).to(device=self.device)
                        target_q1.append(self.critic1_target["critic_backbone"](
                            action=next_target_normalized_action, global_cond=next_target_state_features1))
                        target_q2.append(self.critic2_target["critic_backbone"](
                            action=next_target_normalized_action, global_cond=next_target_state_features2))
                    # (batch_size, 10, 1)
                    target_q1 = torch.stack(target_q1, dim=1)
                    target_q2 = torch.stack(target_q2, dim=1)
                    # (batch_size, 1)
                    target_q1 = target_q1.max(dim=1, keepdim=True)[0].squeeze(1)
                    target_q2 = target_q2.max(dim=1, keepdim=True)[0].squeeze(1)
                    target_q = torch.min(target_q1, target_q2)
                else:
                    next_action = self.actor.diffuse(
                        state_sequence=traj_window.next_state_sequence,
                        task_names=traj_window.task_names,
                        task_metric=traj_window.task_metrics)

                    next_action = next_action[:, action_start:action_end, :]

                    next_target_state_features1 = self.critic1_target["state_sequence_encoder"](
                        state_sequence=traj_window.next_state_sequence, task_names=traj_window.task_names)
                    next_target_state_features2 = self.critic2_target["state_sequence_encoder"](
                        state_sequence=traj_window.next_state_sequence, task_names=traj_window.task_names)
                    next_target_normalized_action = self.action_norm_config.normalize(
                        next_action).to(device=self.device)
                    target_q1 = self.critic1_target["critic_backbone"](
                        action=next_target_normalized_action, global_cond=next_target_state_features1)
                    target_q2 = self.critic2_target["critic_backbone"](
                        action=next_target_normalized_action, global_cond=next_target_state_features2)
                    target_q = torch.min(target_q1, target_q2)

            target_q = traj_window.reward + self.gamma * (1 - traj_window.terminated) * target_q

            if self.monte_carlo:
                # target_q = torch.max(target_q, traj_window.task_metrics).detach()
                target_q = ((target_q + traj_window.task_metrics) / 2).detach()
            else:
                target_q = target_q.detach()

            loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(current_q2, target_q)

        self.critic1.train()
        self.critic2.train()
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
        # critic1
        critic1_keys = [k for k in checkpoint["state_dict"].keys() if "lang_embs_cache" in k and "critic1." in k]
        critic1_texts = [k.split("lang_embs_cache.")[-1] for k in critic1_keys]
        logging.info(
            f"Loading {len(critic1_texts)} cached text features for critic1: "
            + ", ".join(f"{t!r}" for t in critic1_texts)
        )
        self.critic1["state_sequence_encoder"].lang_embs_cache = torch.nn.ParameterDict(
            {
                text: torch.nn.Parameter(checkpoint["state_dict"][k], requires_grad=False)
                for text, k in zip(critic1_texts, critic1_keys)
            }
        )
        # critic1_target
        critic1_target_keys = [k for k in checkpoint["state_dict"]["critic1_target"].keys() if "lang_embs_cache" in k]
        critic1_target_texts = [k.split("lang_embs_cache.")[-1] for k in critic1_target_keys]
        logging.info(
            f"Loading {len(critic1_target_texts)} cached text features for critic1_target: "
            # + ", ".join(f"{t!r}" for t in critic1_target_texts)
        )
        self.critic1_target["state_sequence_encoder"].lang_embs_cache = torch.nn.ParameterDict(
            {
                text: torch.nn.Parameter(checkpoint["state_dict"]["critic1_target"][k], requires_grad=False)
                for text, k in zip(critic1_target_texts, critic1_target_keys)
            }
        )
        # critic2
        critic2_keys = [k for k in checkpoint["state_dict"].keys() if "lang_embs_cache" in k and "critic2." in k]
        critic2_texts = [k.split("lang_embs_cache.")[-1] for k in critic2_keys]
        logging.info(
            f"Loading {len(critic2_texts)} cached text features for critic2: "
            # + ", ".join(f"{t!r}" for t in critic2_texts)
        )
        self.critic2["state_sequence_encoder"].lang_embs_cache = torch.nn.ParameterDict(
            {
                text: torch.nn.Parameter(checkpoint["state_dict"][k], requires_grad=False)
                for text, k in zip(critic2_texts, critic2_keys)
            }
        )
        # critic2_target
        critic2_target_keys = [k for k in checkpoint["state_dict"]["critic2_target"].keys() if "lang_embs_cache" in k]
        critic2_target_texts = [k.split("lang_embs_cache.")[-1] for k in critic2_target_keys]
        logging.info(
            f"Loading {len(critic2_target_texts)} cached text features for critic2_target: "
            # + ", ".join(f"{t!r}" for t in critic2_target_texts)
        )
        self.critic2_target["state_sequence_encoder"].lang_embs_cache = torch.nn.ParameterDict(
            {
                text: torch.nn.Parameter(checkpoint["state_dict"]["critic2_target"][k], requires_grad=False)
                for text, k in zip(critic2_target_texts, critic2_target_keys)
            }
        )

        return super().on_load_checkpoint(checkpoint)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        retval = super().load_state_dict(state_dict, strict=False)
        retval2 = self.critic_target.critic1_target.load_state_dict(state_dict["critic1_target"], strict=False)
        retval3 = self.critic_target.critic2_target.load_state_dict(state_dict["critic2_target"], strict=False)
        return retval

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["action_norm_config"] = self.action_norm_config
        checkpoint["state_dict"]["critic1_target"] = self.critic_target.critic1_target.state_dict()
        checkpoint["state_dict"]["critic2_target"] = self.critic_target.critic2_target.state_dict()

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

    @property
    def critic1_target(self):
        return self.critic_target.critic1_target

    @property
    def critic2_target(self):
        return self.critic_target.critic2_target
