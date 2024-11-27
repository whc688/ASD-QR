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

from scalingup.algo.single_action_critic_backbone import CriticLinearNet, CriticValueFunctionLinearNet
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
EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)

class CriticTarget():
    def __init__(self, critic_Q1_target: nn.ModuleDict, critic_Q2_target: nn.ModuleDict):
        self.critic_Q1_target = critic_Q1_target
        self.critic_Q2_target = critic_Q2_target
        # freeze
        for param in self.critic_Q1_target.parameters():
            param.requires_grad = False
        for param in self.critic_Q2_target.parameters():
            param.requires_grad = False


class IqlSacTrainingAlgo(LightningModule):
    def __init__(
            self,
            actor_state_sequence_encoder: StateSequenceEncoder,
            actor_backbone: LinearNet,
            critic_Q1_state_sequence_encoder: StateSequenceEncoder,
            critic_Q2_state_sequence_encoder: StateSequenceEncoder,
            critic_Q1_backbone: CriticLinearNet,
            critic_Q2_backbone: CriticLinearNet,
            critic_V_state_sequence_encoder: StateSequenceEncoder,
            critic_V_backbone: CriticValueFunctionLinearNet,
            action_dim: int,
            ctrl_config: ControlConfig,
            replay_buffer: RLTrajectoryWindowDataset,
            critic_Q_optimizer: Any,
            critic_V_optimizer: Any,
            actor_optimizer: Any,
            rollout_config: PolicyWindowRolloutConfig,
            critic_Q_lr_scheduler: Optional[Any] = None,
            critic_V_lr_scheduler: Optional[Any] = None,
            actor_lr_scheduler: Optional[Any] = None,
            log_grad_frequence: int = 50,
            gamma: float = 0.98,  # 折扣因子
            alpha: float = 0.005,  # target 进行指数移动平均的参数
            tau: float = 0.9,  # IQL Expectile Regression 的参数
            beta: float = 3.0,  # IQL AWR 的参数
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
        # 由于使用三个 optimizer，所以需要关闭 automatic_optimization
        self.automatic_optimization = False

        self.replay_buffer = replay_buffer
        self.critic_Q_lr_scheduler_partial = critic_Q_lr_scheduler
        self.critic_V_lr_scheduler_partial = critic_V_lr_scheduler
        self.actor_lr_scheduler_partial = actor_lr_scheduler
        self.critic_Q_optimizer_partial = critic_Q_optimizer
        self.critic_V_optimizer_partial = critic_V_optimizer
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
        self.critic_Q1 = nn.ModuleDict(
            {
                "state_sequence_encoder": critic_Q1_state_sequence_encoder,
                "critic_backbone": critic_Q1_backbone,
            }
        )
        self.critic_Q2 = nn.ModuleDict(
            {
                "state_sequence_encoder": critic_Q2_state_sequence_encoder,
                "critic_backbone": critic_Q2_backbone,
            }
        )
        self.critic_V = nn.ModuleDict(
            {
                "state_sequence_encoder": critic_V_state_sequence_encoder,
                "critic_backbone": critic_V_backbone,
            }
        )
        self.actor = nn.ModuleDict(
            {
                "state_sequence_encoder": actor_state_sequence_encoder,
                "actor_backbone": actor_backbone,
            }
        )
        self.critic_target = CriticTarget(
            critic_Q1_target=deepcopy(self.critic_Q1),
            critic_Q2_target=deepcopy(self.critic_Q2),
        )
        # freeze
        for param in self.critic_Q1_target.parameters():
            param.requires_grad = False
        for param in self.critic_Q2_target.parameters():
            param.requires_grad = False
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.beta = beta
        self.log_grad_frequence = log_grad_frequence

        self.rollout_config = rollout_config
        logging.info(str(self.rollout_config))
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

        critic_Q1_params = list(self.critic_Q1.parameters())
        critic_Q2_params = list(self.critic_Q2.parameters())
        critic_Q_params = critic_Q1_params + critic_Q2_params
        critic_Q_optimizer = self.critic_Q_optimizer_partial(params=critic_Q_params)

        critic_V_params = list(self.critic_V.parameters())
        critic_V_optimizer = self.critic_V_optimizer_partial(params=critic_V_params)

        if self.critic_Q_lr_scheduler_partial is None:
            raise NotImplementedError("critic_lr_scheduler_partial is None")

        return [actor_optimizer, critic_Q_optimizer, critic_V_optimizer], [
            {
                "scheduler": self.actor_lr_scheduler_partial(optimizer=actor_optimizer),
                "interval": "step",
            },
            {
                "scheduler": self.critic_Q_lr_scheduler_partial(optimizer=critic_Q_optimizer),
                "interval": "step",
            },
            {
                "scheduler": self.critic_V_lr_scheduler_partial(optimizer=critic_V_optimizer),
                "interval": "step",
            }
        ]

    def training_step(self, tensor_data: TensorDataClass, batch_idx: int):
        self.critic_Q1_target.to(device=self.device)
        self.critic_Q2_target.to(device=self.device)

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
        actor_optimizer, critic_Q_optimizer, critic_V_optimizer = self.optimizers()
        actor_lr_scheduler, critic_Q_lr_scheduler, critic_V_lr_scheduler = self.lr_schedulers()
        self.actor.train()
        self.critic_Q1.train()
        self.critic_Q2.train()
        self.critic_V.train()
        traj_window = typing.cast(RLTrajectoryWindowTensor, tensor_data).to(dtype=self.dtype)

        with torch.no_grad():
            # target_q = self.q_target(observations, actions)
            state_features1 = self.critic_Q1_target["state_sequence_encoder"](
                state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
            # prepare action sequence
            normalized_action = self.action_norm_config.normalize(
                traj_window.action_sequence.tensor).to(device=self.device)
            # 只取当前步的动作
            normalized_action = normalized_action[:, self.rollout_config.proprio_obs_horizon - 1, :]
            target_q1 = self.critic_Q1_target["critic_backbone"](
                action=normalized_action, global_cond=state_features1)
            state_features2 = self.critic_Q2_target["state_sequence_encoder"](
                state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
            target_q2 = self.critic_Q2_target["critic_backbone"](
                action=normalized_action, global_cond=state_features2)

            # next_v = self.vf(next_observations)
            next_state_features = self.critic_V["state_sequence_encoder"](
                state_sequence=traj_window.next_state_sequence, task_names=traj_window.task_names)
            next_v = self.critic_V["critic_backbone"](global_cond=next_state_features)

        # Update value function
        # v = self.vf(observations)
        state_features = self.critic_V["state_sequence_encoder"](
            state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
        v = self.critic_V["critic_backbone"](global_cond=state_features)

        adv1 = target_q1 - v
        adv2 = target_q2 - v
        v_loss1 = asymmetric_l2_loss(adv1, self.tau)
        v_loss2 = asymmetric_l2_loss(adv2, self.tau)
        v_loss = v_loss1 + v_loss2
        # self.v_optimizer.zero_grad(set_to_none=True)
        critic_V_optimizer.zero_grad()
        self.manual_backward(v_loss)
        critic_V_optimizer.step()

        # Update Q function
        # targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        targets = traj_window.reward + self.gamma * (1 - traj_window.terminated) * next_v.detach()

        # qs = self.qf.both(observations, actions)
        state_features1 = self.critic_Q1["state_sequence_encoder"](
            state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
        state_features2 = self.critic_Q2["state_sequence_encoder"](
            state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
        # prepare action sequence
        normalized_action = self.action_norm_config.normalize(
            traj_window.action_sequence.tensor).to(device=self.device)
        # 只取当前步的动作
        normalized_action = normalized_action[:, self.rollout_config.proprio_obs_horizon - 1, :]
        qs1 = self.critic_Q1["critic_backbone"](
            action=normalized_action, global_cond=state_features1)
        qs2 = self.critic_Q2["critic_backbone"](
            action=normalized_action, global_cond=state_features2)

        # q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        q_loss1 = nn.functional.mse_loss(qs1, targets)
        q_loss2 = nn.functional.mse_loss(qs2, targets)
        q_loss = (q_loss1 + q_loss2) / 2
        # self.q_optimizer.zero_grad(set_to_none=True)
        critic_Q_optimizer.zero_grad()
        self.manual_backward(q_loss)
        critic_Q_optimizer.step()

        # Update target Q network
        # 这一步放到 on_train_batch_end 中

        # Update actor
        # exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        exp_adv1 = torch.exp(self.beta * adv1.detach()).clamp(max=EXP_ADV_MAX)
        exp_adv2 = torch.exp(self.beta * adv2.detach()).clamp(max=EXP_ADV_MAX)

        # policy_out = self.policy(observations)
        state_features = self.actor["state_sequence_encoder"](
            state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
        # 返回的是 normalized action
        policy_out = self.actor["actor_backbone"](global_cond=state_features)

        # bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        bc_losses = torch.sum((policy_out - normalized_action) ** 2, dim=1)

        # policy_loss = torch.mean(exp_adv * bc_losses)
        policy_loss1 = torch.mean(exp_adv1 * bc_losses)
        policy_loss2 = torch.mean(exp_adv2 * bc_losses)
        policy_loss = (policy_loss1 + policy_loss2) / 2

        # self.policy_optimizer.zero_grad(set_to_none=True)
        actor_optimizer.zero_grad()
        self.manual_backward(policy_loss)
        actor_optimizer.step()

        self.log("train/v_loss", v_loss,
                 sync_dist=True,
                 on_step=True,
                 on_epoch=True,
                 batch_size=self.replay_buffer.batch_size, )
        self.log("train/q_loss", q_loss,
                 sync_dist=True,
                 on_step=True,
                 on_epoch=True,
                 batch_size=self.replay_buffer.batch_size, )
        self.log("train/policy_loss", policy_loss,
                 sync_dist=True,
                 on_step=True,
                 on_epoch=True,
                 batch_size=self.replay_buffer.batch_size, )

        actor_lr_scheduler.step()
        critic_Q_lr_scheduler.step()
        critic_V_lr_scheduler.step()
        loss = v_loss + q_loss + policy_loss
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
        # for param, target_param in zip(
        #         self.critic.parameters(), self.critic_target.parameters()
        # ):
        #     if param.requires_grad:
        #         target_param.data.copy_(self.alpha * param.data + (1 - self.alpha) * target_param.data)

        for param, target_param in zip(
                self.critic_Q1.parameters(), self.critic_Q1_target.parameters()
        ):
            if param.requires_grad:
                target_param.data.copy_(self.alpha * param.data + (1 - self.alpha) * target_param.data)
        for param, target_param in zip(
                self.critic_Q2.parameters(), self.critic_Q2_target.parameters()
        ):
            if param.requires_grad:
                target_param.data.copy_(self.alpha * param.data + (1 - self.alpha) * target_param.data)

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
        self.actor.eval()
        self.critic_Q1.eval()
        self.critic_Q2.eval()
        self.critic_V.eval()

        with torch.no_grad():
            # target_q = self.q_target(observations, actions)
            state_features1 = self.critic_Q1_target["state_sequence_encoder"](
                state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
            # prepare action sequence
            normalized_action = self.action_norm_config.normalize(
                traj_window.action_sequence.tensor).to(device=self.device)
            # 只取当前步的动作
            normalized_action = normalized_action[:, self.rollout_config.proprio_obs_horizon - 1, :]
            target_q1 = self.critic_Q1_target["critic_backbone"](
                action=normalized_action, global_cond=state_features1)
            state_features2 = self.critic_Q2_target["state_sequence_encoder"](
                state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
            target_q2 = self.critic_Q2_target["critic_backbone"](
                action=normalized_action, global_cond=state_features2)

            # next_v = self.vf(next_observations)
            next_state_features = self.critic_V["state_sequence_encoder"](
                state_sequence=traj_window.next_state_sequence, task_names=traj_window.task_names)
            next_v = self.critic_V["critic_backbone"](global_cond=next_state_features)

            # Update value function
            # v = self.vf(observations)
            state_features = self.critic_V["state_sequence_encoder"](
                state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
            v = self.critic_V["critic_backbone"](global_cond=state_features)

            adv1 = target_q1 - v
            adv2 = target_q2 - v
            v_loss1 = asymmetric_l2_loss(adv1, self.tau)
            v_loss2 = asymmetric_l2_loss(adv2, self.tau)
            v_loss = v_loss1 + v_loss2

            # Update Q function
            # targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
            targets = traj_window.reward + self.gamma * (1 - traj_window.terminated) * next_v.detach()

            # qs = self.qf.both(observations, actions)
            state_features1 = self.critic_Q1["state_sequence_encoder"](
                state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
            state_features2 = self.critic_Q2["state_sequence_encoder"](
                state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
            # prepare action sequence
            normalized_action = self.action_norm_config.normalize(
                traj_window.action_sequence.tensor).to(device=self.device)
            # 只取当前步的动作
            normalized_action = normalized_action[:, self.rollout_config.proprio_obs_horizon - 1, :]
            qs1 = self.critic_Q1["critic_backbone"](
                action=normalized_action, global_cond=state_features1)
            qs2 = self.critic_Q2["critic_backbone"](
                action=normalized_action, global_cond=state_features2)

            # q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
            q_loss1 = nn.functional.mse_loss(qs1, targets)
            q_loss2 = nn.functional.mse_loss(qs2, targets)
            q_loss = (q_loss1 + q_loss2) / 2

            # Update actor
            # exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
            exp_adv1 = torch.exp(self.beta * adv1.detach()).clamp(max=EXP_ADV_MAX)
            exp_adv2 = torch.exp(self.beta * adv2.detach()).clamp(max=EXP_ADV_MAX)

            # policy_out = self.policy(observations)
            state_features = self.actor["state_sequence_encoder"](
                state_sequence=traj_window.state_sequence, task_names=traj_window.task_names)
            # 返回的是 normalized action
            policy_out = self.actor["actor_backbone"](global_cond=state_features)

            # bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
            bc_losses = torch.sum((policy_out - normalized_action) ** 2, dim=1)

            # policy_loss = torch.mean(exp_adv * bc_losses)
            policy_loss1 = torch.mean(exp_adv1 * bc_losses)
            policy_loss2 = torch.mean(exp_adv2 * bc_losses)
            policy_loss = (policy_loss1 + policy_loss2) / 2

            self.log("validation/v_loss", v_loss,
                     sync_dist=True,
                     on_epoch=True,
                     batch_size=self.replay_buffer.batch_size, )
            self.log("validation/q_loss", q_loss,
                     sync_dist=True,
                     on_epoch=True,
                     batch_size=self.replay_buffer.batch_size, )
            self.log("validation/policy_loss", policy_loss,
                     sync_dist=True,
                     on_epoch=True,
                     batch_size=self.replay_buffer.batch_size, )
            loss = v_loss + q_loss + policy_loss

        self.actor.train()
        self.critic_Q1.train()
        self.critic_Q2.train()
        self.critic_V.train()
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

        # 这里需要处理 critic_Q 和 target 还有 critic_V 以及 actor 的 state_sequence_encoder 的参数
        # critic_Q1
        critic_keys = [k for k in checkpoint["state_dict"].keys() if "lang_embs_cache" in k and "critic_Q1." in k]
        critic_texts = [k.split("lang_embs_cache.")[-1] for k in critic_keys]
        logging.info(
            f"Loading {len(critic_texts)} cached text features for critic_Q1: "
            + ", ".join(f"{t!r}" for t in critic_texts)
        )
        self.critic_Q1["state_sequence_encoder"].lang_embs_cache = torch.nn.ParameterDict(
            {
                text: torch.nn.Parameter(checkpoint["state_dict"][k], requires_grad=False)
                for text, k in zip(critic_texts, critic_keys)
            }
        )
        # critic_Q1_target
        critic_target_keys = [k for k in checkpoint["state_dict"].keys() if
                              "lang_embs_cache" in k and "critic_Q1_target." in k]
        critic_target_texts = [k.split("lang_embs_cache.")[-1] for k in critic_target_keys]
        logging.info(
            f"Loading {len(critic_target_texts)} cached text features for critic_Q1_target: "
            + ", ".join(f"{t!r}" for t in critic_target_texts)
        )
        self.critic_Q1_target["state_sequence_encoder"].lang_embs_cache = torch.nn.ParameterDict(
            {
                text: torch.nn.Parameter(checkpoint["state_dict"][k], requires_grad=False)
                for text, k in zip(critic_target_texts, critic_target_keys)
            }
        )
        # critic_Q2
        critic_keys = [k for k in checkpoint["state_dict"].keys() if "lang_embs_cache" in k and "critic_Q2." in k]
        critic_texts = [k.split("lang_embs_cache.")[-1] for k in critic_keys]
        logging.info(
            f"Loading {len(critic_texts)} cached text features for critic_Q2: "
            + ", ".join(f"{t!r}" for t in critic_texts)
        )
        self.critic_Q2["state_sequence_encoder"].lang_embs_cache = torch.nn.ParameterDict(
            {
                text: torch.nn.Parameter(checkpoint["state_dict"][k], requires_grad=False)
                for text, k in zip(critic_texts, critic_keys)
            }
        )
        # critic_Q2_target
        critic_target_keys = [k for k in checkpoint["state_dict"].keys() if
                              "lang_embs_cache" in k and "critic_Q2_target." in k]
        critic_target_texts = [k.split("lang_embs_cache.")[-1] for k in critic_target_keys]
        logging.info(
            f"Loading {len(critic_target_texts)} cached text features for critic_Q2_target: "
            + ", ".join(f"{t!r}" for t in critic_target_texts)
        )
        self.critic_Q2_target["state_sequence_encoder"].lang_embs_cache = torch.nn.ParameterDict(
            {
                text: torch.nn.Parameter(checkpoint["state_dict"][k], requires_grad=False)
                for text, k in zip(critic_target_texts, critic_target_keys)
            }
        )
        # critic_V
        critic_keys = [k for k in checkpoint["state_dict"].keys() if "lang_embs_cache" in k and "critic_V." in k]
        critic_texts = [k.split("lang_embs_cache.")[-1] for k in critic_keys]
        logging.info(
            f"Loading {len(critic_texts)} cached text features for critic_V: "
            + ", ".join(f"{t!r}" for t in critic_texts)
        )
        self.critic_V["state_sequence_encoder"].lang_embs_cache = torch.nn.ParameterDict(
            {
                text: torch.nn.Parameter(checkpoint["state_dict"][k], requires_grad=False)
                for text, k in zip(critic_texts, critic_keys)
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

    @property
    def critic_Q1_target(self):
        return self.critic_target.critic_Q1_target

    @property
    def critic_Q2_target(self):
        return self.critic_target.critic_Q2_target
