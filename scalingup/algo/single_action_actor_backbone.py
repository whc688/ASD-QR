# Taken from https://diffusion-policy.cs.columbia.edu/
import logging
import math
from typing import List, Optional, Tuple, Union
import typing

import torch
from torch import nn

from scalingup.algo.state_encoder.token_learner import SeqTokenLearnerModule


# @markdown ### **Network**
# @markdown
# @markdown Defines a 1D UNet architecture `ConditionalUnet1D`
# @markdown as the noies prediction network
# @markdown
# @markdown Components
# @markdown - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
# @markdown - `Downsample1d` Strided convolution to reduce temporal resolution
# @markdown - `Upsample1d` Transposed convolution to increase temporal resolution
# @markdown - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
# @markdown - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
# @markdown `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
# @markdown `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.

class LinearNet(nn.Module):
    def __init__(
            self,
            action_dim: int,
            global_cond_dim: int,
            dropout_prob: float = 0.2,
    ):
        """
        global_cond_dim: Dim of global conditioning applied with FiLM
        """

        super().__init__()
        self.action_dim = action_dim
        self.global_cond_dim = global_cond_dim

        self.linear1 = nn.Linear(global_cond_dim, 128)
        self.mish1 = nn.Mish()
        self.dropout1 = nn.Dropout(dropout_prob)

        self.linear2 = nn.Linear(128, 64)
        self.mish2 = nn.Mish()
        self.dropout2 = nn.Dropout(dropout_prob)

        self.linear3 = nn.Linear(64, action_dim)

    def forward(
            self,
            global_cond: Optional[torch.Tensor] = None,  # condition
    ):
        """
        返回的是 normalized action
        """
        # linear
        x = self.linear1(global_cond)
        x = self.mish1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.mish2(x)
        x = self.dropout2(x)

        x = self.linear3(x)
        return x


if __name__ == '__main__':
    # 测试 ConditionalCNN1D
    batch_size = 5
    seq_len = 16
    action_dim = 10
    global_cond_dim = 282
    sample = torch.randn(batch_size, seq_len, action_dim)
    global_cond = torch.randn(batch_size, global_cond_dim)
    network = LinearNet(
        action_dim=action_dim,
        global_cond_dim=global_cond_dim,
    )
    output = network(global_cond)
    print(output.shape)
