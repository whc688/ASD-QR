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


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalCNN1D(nn.Module):
    def __init__(
            self,
            input_dim: int,
            global_cond_dim: int,
            down_dims: List[int],
            kernel_size: int = 3,
            n_groups: int = 8,
    ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines number of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)

        cond_dim = global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        kernel_size_list = [5, 3, 1]
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            kernel_size = kernel_size_list[ind]
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        # Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.down_modules = down_modules

        self.token_learner = SeqTokenLearnerModule(
            inputs_channels=mid_dim,
            num_tokens=1,
            bottleneck_dim=mid_dim // 4,
            dropout_rate=0.0,
        )

        self.linear1 = nn.Linear(mid_dim, mid_dim // 4)
        self.mish = nn.Mish()
        self.linear2 = nn.Linear(mid_dim // 4, 1)

    def forward(
            self,
            action: torch.Tensor,  # action
            global_cond: Optional[torch.Tensor] = None,  # condition
    ):
        """
        x: (B,T,input_dim)
        global_cond: (B, global_cond_dim)
        output: (B,1)
        """
        # (B,T,C)
        action = action.moveaxis(-1, -2)
        # (B,C,T)

        global_feature = global_cond

        x = action
        # h = []
        for resnet, resnet2 in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            # h.append(x)
            # x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        # 上述代码输出特征为 x: (b, 1024, 4) 即 (batch_size, channel, horizon)

        # token_learner
        x = self.token_learner(x)  # (b, 1, 1024)
        x = x.squeeze(1)  # (b, 1024)

        # linear
        x = self.linear1(x)  # (b, 256)
        x = self.mish(x)  # (b, 256)
        x = self.linear2(x)  # (b, 1)

        return x


class DiscreteDiffusionEncoderDecoder(nn.Module):
    def __init__(self,
                 action_dim: int,
                 decoder_horizon: int,
                 encoder_horizon: int,
                 cond_dim: int,
                 n_encoder_layer: int = 6,
                 n_decoder_layer: int = 6,
                 n_head: int = 8,
                 n_emb: int = 512,
                 p_drop_emb: float = 0.1,
                 p_drop_attn: float = 0.1,
                 diff_step_cond: str = 'add',
                 pos_emb_type: str = 'learned',
                 diffusion_step_embed_dim: int = 512,
                 ) -> None:
        """
        action_dim: number of action dimension
        decoder_horizon: length of input to transformer decoder, pred_horizon*action_dim
        encoder_horizon: length of input to transformer encoder, obs_horizon
        cond_dim: dimension of global conditioning, (image_emb+proprio+task_emb), it will be scaled to n_emb
        n_encoder_layer: number of layers for transformer encoder
        n_decoder_layer: number of layers for transformer decoder
        n_head: number of attention heads
        n_emb: dimension of embedding
        p_drop_emb: dropout rate for embedding
        p_drop_attn: dropout rate for attention
        causal_attn: whether to use causal attention
        diff_step_cond: how to condition on diffusion step, 'add', etc.
        pos_emb_type: type of position embedding, 'learned', 'sinusoidal'
        """
        super().__init__()

        # decoder input embedding stem
        self.decoder_input_emb = nn.Linear(action_dim, n_emb)
        if pos_emb_type == 'learned':
            self.decoder_pos_emb = nn.Embedding(decoder_horizon, n_emb)
        elif pos_emb_type == 'sinusoidal':
            self.decoder_pos_emb = SinusoidalPosEmb(n_emb)
        else:
            raise NotImplementedError(f'pos_emb_type {pos_emb_type} not implemented')

        # diffusion step embedding
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # encoder input embedding stem
        self.encoder_input_emb = nn.Linear(cond_dim, n_emb)
        if pos_emb_type == 'learned':
            self.encoder_pos_emb = nn.Embedding(encoder_horizon, n_emb)
        elif pos_emb_type == 'sinusoidal':
            self.encoder_pos_emb = SinusoidalPosEmb(n_emb)
        else:
            raise NotImplementedError(f'pos_emb_type {pos_emb_type} not implemented')

        # dropout
        self.drop = nn.Dropout(p_drop_emb)

        # encoder
        if n_encoder_layer > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_encoder_layer
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_emb, 4 * n_emb),
                nn.Mish(),
                nn.Linear(4 * n_emb, n_emb)
            )

        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True  # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layer
        )

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, action_dim)

        # constants
        self.action_dim = action_dim
        self.decoder_horizon = decoder_horizon
        self.encoder_horizon = encoder_horizon
        self.cond_dim = cond_dim
        self.n_encoder_layer = n_encoder_layer
        self.n_decoder_layer = n_decoder_layer
        self.n_head = n_head
        self.n_emb = n_emb
        self.p_drop_emb = p_drop_emb
        self.p_drop_attn = p_drop_attn
        self.diff_step_cond = diff_step_cond
        self.pos_emb_type = pos_emb_type
        self.diffusion_step_embed_dim = diffusion_step_embed_dim

    def forward(self,
                sample: torch.Tensor,
                timestep: torch.Tensor,
                global_cond: torch.Tensor,
                action_sequence: torch.Tensor,
                ) -> torch.Tensor:
        """
        sample: (batch_size, (pred_horizon - obs_horizon + 1), action_dim)
        timestep: (batch_size,), diffusion step
        global_cond: (batch_size, obs_horizon, cond_dim)
        action_sequence: (batch_size，(obs_horizon - 1), action_dim)

        output: (batch_size, (pred_horizon - obs_horizon + 1), action_dim)
        """
        # 记录 sample 的序列长度，也即需要输出的序列长度
        prediction_length = sample.shape[1]

        timesteps = timestep
        if timesteps.shape == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        # encoder
        global_cond = self.encoder_input_emb(global_cond)
        if self.pos_emb_type == 'learned':
            # 获取批次大小和序列长度
            batch_size, seq_len = global_cond.shape[0], global_cond.shape[1]

            # 生成位置索引
            # 这里的位置索引是一个从0到seq_len-1的整数序列
            # 它被扩展到与批次大小相同，以便为每个序列生成位置嵌入
            positions = torch.arange(seq_len, device=global_cond.device).expand(batch_size, seq_len)

            # 获取位置嵌入
            # 输出维度为 (batch_size, seq_len, n_emb)
            position_embeddings = self.encoder_pos_emb(positions)
        elif self.pos_emb_type == 'sinusoidal':
            # 获取批次大小和序列长度
            batch_size, seq_len = global_cond.shape[0], global_cond.shape[1]

            # 生成位置索引
            # 这里的位置索引是一个从0到seq_len-1的整数序列
            # SinusoidalPosEmb类的forward方法，期望接收一个一维的位置索引向量
            # 维度为 (seq_len,)
            positions = torch.arange(seq_len, device=global_cond.device)
            # 获取位置嵌入
            # 维度为 (1, seq_len, n_emb)
            position_embeddings = self.encoder_pos_emb(positions).unsqueeze(0)
        else:
            raise NotImplementedError(f'pos_emb_type {self.pos_emb_type} not implemented')

        # 将位置嵌入与条件嵌入相加
        global_cond += position_embeddings

        # 对条件嵌入进行dropout
        global_cond = self.drop(global_cond)

        # 将条件嵌入输入到transformer encoder中
        memory = self.encoder(global_cond)

        # decoder
        # 首先将 history action sequence 与 sample 进行拼接，组成完整的输入序列
        # (batch_size, pred_horizon, action_dim)
        sample = torch.cat([action_sequence, sample], dim=1)

        # (batch_size, pred_horizon, n_emb)
        sample = self.decoder_input_emb(sample)

        # 得到 diffusion timesteps 的嵌入
        # 输出维度为 (batch_size, diffusion_step_embed_dim)
        diffsuion_step_emb = self.diffusion_step_encoder(timesteps)

        if self.pos_emb_type == 'learned':
            # 获取批次大小和序列长度
            batch_size, seq_len = sample.shape[0], sample.shape[1]

            # 生成位置索引
            # 这里的位置索引是一个从0到seq_len-1的整数序列
            # 它被扩展到与批次大小相同，以便为每个序列生成位置嵌入
            positions = torch.arange(seq_len, device=sample.device).expand(batch_size, seq_len)

            # 获取位置嵌入
            # 输出维度为 (batch_size, seq_len, n_emb)
            position_embeddings = self.decoder_pos_emb(positions)
        elif self.pos_emb_type == 'sinusoidal':
            # 获取批次大小和序列长度
            batch_size, seq_len = sample.shape[0], sample.shape[1]

            # 生成位置索引
            # 这里的位置索引是一个从0到seq_len-1的整数序列
            # SinusoidalPosEmb类的forward方法，期望接收一个一维的位置索引向量
            # 维度为 (seq_len,)
            positions = torch.arange(seq_len, device=sample.device)
            # 获取位置嵌入
            # 维度为 (1, seq_len, n_emb)
            position_embeddings = self.decoder_pos_emb(positions).unsqueeze(0)
        else:
            raise NotImplementedError(f'pos_emb_type {self.pos_emb_type} not implemented')

        # 将位置嵌入与 sample embedding 相加
        sample += position_embeddings

        if self.diff_step_cond == 'add':
            # 维度为 (batch_size, pred_horizon, n_emb)
            diffsuion_step_emb = diffsuion_step_emb.unsqueeze(1).expand(-1, seq_len, -1)
            assert diffsuion_step_emb.shape == sample.shape
            sample += diffsuion_step_emb
        else:
            raise NotImplementedError(f'diff_step_cond {self.diff_step_cond} not implemented')

        # 对 sample 进行 dropout
        sample = self.drop(sample)

        # 将 sample 和 memory 输入到 transformer decoder 中
        # 输出维度为 (batch_size, pred_horizon, n_emb)
        sample = self.decoder(
            tgt=sample,
            memory=memory
        )

        # 只取最后 prediction_length 个时间步的输出
        # (batch_size, (pred_horizon - obs_horizon + 1), n_emb)
        sample = sample[:, -prediction_length:, :]
        sample = self.ln_f(sample)
        # (batch_size, (pred_horizon - obs_horizon + 1), action_dim)
        sample = self.head(sample)
        return sample


if __name__ == '__main__':
    # 测试 ConditionalCNN1D
    batch_size = 5
    seq_len = 16
    action_dim = 10
    global_cond_dim = 282
    sample = torch.randn(batch_size, seq_len, action_dim)
    global_cond = torch.randn(batch_size, global_cond_dim)
    network = ConditionalCNN1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
    )
    output = network(sample, global_cond)
    print(output.shape)
