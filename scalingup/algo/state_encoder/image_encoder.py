from copy import deepcopy
import logging
from typing import Dict, Optional, Tuple
import typing
import torch
from scalingup.algo.state_encoder.vision_encoder import ImageEncoder, ImageFilmEfficientEncoder
from scalingup.algo.robomimic_nets import SpatialSoftmax
import torchvision
from scalingup.algo.state_encoder.token_learner import TokenLearnerModule
from torchvision.ops.misc import Conv2dNormActivation
import torch.nn as nn

class PerViewImageEncoder(ImageEncoder):
    def __init__(
        self,
        base_model: torch.nn.Module,
        per_view_output_dim: int,
        use_spatial_softmax: bool = False,
        spatial_softmax_input_shape: Optional[Tuple[int, int, int]] = None,
        spatial_softmax_num_kp: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if issubclass(type(base_model), torchvision.models.ResNet):
            resnet_model = typing.cast(torchvision.models.ResNet, base_model)
            resnet_model.fc = torch.nn.Identity()  # type: ignore
            if use_spatial_softmax:
                assert spatial_softmax_input_shape is not None
                resnet_model.avgpool = SpatialSoftmax(  # type: ignore
                    input_shape=spatial_softmax_input_shape, num_kp=spatial_softmax_num_kp
                )
        else:
            logging.warning(
                f"PerViewImageEncoder: {type(base_model).__name__} is not a ResNet. "
                + "Ignoring spatial softmax arguments"
            )

        # TODO try changing
        # ```
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # ```
        # to identity also
        self.nets = torch.nn.ModuleDict(
            {view: deepcopy(base_model) for view in self.views}
        ).to(memory_format=self.memory_format)  # channel_last 是指数据在内存中的存储顺序，即先存储完一个channel的所有数据，再存储下一个channel的所有数据。
        self.proj = torch.nn.Linear(
            len(self.views) * per_view_output_dim,
            self.output_dim,
        )

    def process_views(self, views: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, obs_horizon, channels, height, width = views[self.views[0]].shape
        features = []
        for view_name, view in views.items():
            view = view.reshape(batch_size * obs_horizon, channels, height, width).to(
                memory_format=self.memory_format
            )
            features.append(self.nets[view_name](view).view(batch_size, obs_horizon, -1))
        return self.proj(torch.cat(features, dim=-1))


class DirectPerViewImageEncoder(PerViewImageEncoder):
    # 继承所有初始化，但是输出时不对 concate 的特征进行缩小，原本的是从 192->128，先在直接输出 192
    # TODO: neeed to check
    def __init__(
        self,
        base_model: torch.nn.Module,
        per_view_output_dim: int,
        use_spatial_softmax: bool = False,
        spatial_softmax_input_shape: Optional[Tuple[int, int, int]] = None,
        spatial_softmax_num_kp: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            base_model=base_model,
            per_view_output_dim=per_view_output_dim,
            use_spatial_softmax=use_spatial_softmax,
            spatial_softmax_input_shape=spatial_softmax_input_shape,
            spatial_softmax_num_kp=spatial_softmax_num_kp,
            **kwargs,
        )
        # 192
        self.proj = torch.nn.Identity()


class PerViewImageFilmEfficientEncoder(ImageFilmEfficientEncoder):
    def __init__(
        self,
        base_model: torch.nn.Module,
        per_view_output_dim: int,
        num_tokens: int,
        use_spatial_softmax: bool = False,
        spatial_softmax_input_shape: Optional[Tuple[int, int, int]] = None,
        spatial_softmax_num_kp: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nets = torch.nn.ModuleDict(
            {view: deepcopy(base_model) for view in self.views}
        ).to(memory_format=self.memory_format)  # channel_last 是指数据在内存中的存储顺序，即先存储完一个channel的所有数据，再存储下一个channel的所有数据。
        downconvNormAct = Conv2dNormActivation(
                1280,
                per_view_output_dim,
                kernel_size=1,
                stride=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.SiLU
            )
        self.downconvs = torch.nn.ModuleDict(
            {view: deepcopy(downconvNormAct) for view in self.views}
        ).to(memory_format=self.memory_format)
        token_learner = TokenLearnerModule(
                inputs_channels=per_view_output_dim,
                num_tokens=num_tokens,
        )
        self.token_learners = torch.nn.ModuleDict(
            {view: deepcopy(token_learner) for view in self.views}
        ).to(memory_format=self.memory_format)
        self.num_tokens = num_tokens


    def process_views(self, views: Dict[str, torch.Tensor], context=None) -> torch.Tensor:
        assert context is not None, "context should not be None"
        batch_size, obs_horizon, channels, height, width = views[self.views[0]].shape
        features = []
        context = context.reshape(batch_size * obs_horizon, -1)
        for view_name, view in views.items():
            view = view.reshape(batch_size * obs_horizon, channels, height, width).to(
                memory_format=self.memory_format
            )
            # batch_size, obs_horizon, 1280, 5, 8
            out = self.nets[view_name](view, context)
            # batch_size, obs_horizon, 512, 5, 8
            out = self.downconvs[view_name](out)
            # batch_size * obs_horizon, num_tokens, 512
            out = self.token_learners[view_name](out)
            out = out.reshape(batch_size, obs_horizon, out.shape[1], out.shape[2])
            # batch_size, obs_horizon * num_tokens, 512
            out = out.view(batch_size, obs_horizon * self.num_tokens, -1)
            features.append(out)
        # batch_size, obs_horizon * num_tokens * n_views, 512
        return torch.cat(features, dim=-2)
