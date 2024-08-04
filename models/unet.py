from abc import abstractmethod
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttentionBlock
from .timesteps_handler import encode_timestep


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, embedding: torch.Tensor):  # type: ignore
        """
        Apply to x given embedding"""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, input_tensor: torch.Tensor, timesteps_encoding: torch.Tensor):  # type: ignore
        x = input_tensor
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, timesteps_encoding)
            else:
                x = layer(x)
        return x


class ResidualBlock(TimestepBlock):
    def __init__(
        self,
        in_channel: int,
        time_embed_dim: int,
        out_channel: int,
        dropout: float,
        upsample: bool = False,
        downsample: bool = False,
    ):
        super().__init__()  # type: ignore

        self.use_conv_for_skip_connection = False

        self.in_channel = in_channel
        self.time_embed_dim = time_embed_dim
        self.out_channel = out_channel
        self.dropout = dropout
        self.downsample = downsample
        self.upsample = upsample

        self.upordown = self.upsample or self.downsample

        self.h_upd: UpSampling | DownSampling | nn.Identity
        self.x_upd: UpSampling | DownSampling | nn.Identity

        self.embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                self.time_embed_dim, 2 * self.out_channel
            ),  # TODO check use_scale_shift_norm
        )

        if upsample:
            self.h_upd = UpSampling(self.in_channel, self.in_channel, use_conv=False)
            self.x_upd = UpSampling(self.in_channel, self.in_channel, use_conv=False)
        elif downsample:
            self.h_upd = DownSampling(self.in_channel, self.in_channel, use_conv=False)
            self.x_upd = DownSampling(self.in_channel, self.in_channel, use_conv=False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.input_layer = nn.Sequential(
            nn.GroupNorm(32, self.in_channel),
            nn.SiLU(),
            nn.Conv2d(
                self.in_channel, self.out_channel, 3, padding=1
            ),  # Increase the number if channel while keeping the resolution intact
        )

        self.output_layer = nn.Sequential(
            nn.GroupNorm(32, self.out_channel),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            zero_module(
                nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)
            ),  # TODO: Need to zero this one
        )

        if self.in_channel == self.out_channel:
            self.skip_connection: nn.Conv2d | nn.Identity = nn.Identity()
        else:
            channel = 3 if self.use_conv_for_skip_connection else 1
            self.skip_connection = nn.Conv2d(
                self.in_channel, self.out_channel, channel, padding=0
            )

    def forward(self, input_tensor: torch.Tensor, timesteps_encoding: torch.Tensor):

        timesteps_embedding = self.embedding_layer(timesteps_encoding)

        if self.upordown:
            in_rest, in_conv = self.input_layer[:-1], self.input_layer[-1]
            transformed_input_tensor = in_rest(input_tensor)
            transformed_input_tensor = self.h_upd(transformed_input_tensor)
            input_tensor = self.x_upd(input_tensor)
            transformed_input_tensor = in_conv(transformed_input_tensor)
        else:
            transformed_input_tensor = self.input_layer(input_tensor)

        timesteps_embedding = timesteps_embedding.unsqueeze(dim=2).unsqueeze(dim=3)

        if (
            True
        ):  # TODO FIXME revisit later to understand the use of scale and shift norm
            output_norm, output_rest = self.output_layer[0], self.output_layer[1]
            scale, shift = torch.chunk(timesteps_embedding, 2, dim=1)
            output_tensor = output_norm(transformed_input_tensor) * (1 + scale) + shift
            output_tensor = output_rest(transformed_input_tensor)
        else:
            input_to_second_layer = (
                transformed_input_tensor + timesteps_embedding
            )  # unqueeze timesteps embedding until it has the dimen len size than the transformed
            output_tensor = self.output_layer(input_to_second_layer)
        output_tensor = output_tensor + self.skip_connection(input_tensor)

        return output_tensor


class DownSampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        padding: int = 1,
        use_conv: bool = True,
    ):
        super().__init__()  # type: ignore
        self.use_conv = use_conv
        self.downsampling_layer: nn.Conv2d | nn.AvgPool2d

        if self.use_conv:
            self.downsampling_layer = nn.Conv2d(
                in_channels, out_channels, 3, stride=stride, padding=padding
            )
        else:
            self.downsampling_layer = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.downsampling_layer(input_tensor)


class UpSampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 1,
        use_conv: bool = True,
    ):
        super().__init__()  # type: ignore
        self.use_conv = use_conv
        if self.use_conv:
            self.conv_layer = nn.Conv2d(in_channels, out_channels, 3, padding=padding)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = F.interpolate(input_tensor, scale_factor=2, mode="nearest")  # type: ignore
        if self.use_conv:
            return self.conv_layer(input_tensor)
        return input_tensor  # type: ignore


class Unet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        model_channels: int,
        mult_channel: List[int],
        residual_blocks_num: int,
        dropout: float,
        attention_resolutions: List[int],
        num_classes: int = 0,
    ) -> None:
        super().__init__()  # type: ignore

        self.input_channel = input_channels
        self.output_channels = output_channels
        self.model_channels = model_channels
        self.mult_channel = mult_channel
        self.residual_blocks_num = residual_blocks_num
        self.dropout = dropout
        self.attention_resolutions = attention_resolutions
        self.num_classes = num_classes

        # embed timesteps
        self.time_embedding_dim = 4 * self.model_channels
        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(self.model_channels, self.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )

        # classes embedding
        if self.num_classes:
            self.label_emb = nn.Embedding(self.num_classes, self.time_embedding_dim)

        # input downsampling block
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(3, self.model_channels, 3, padding=1))]
        )
        previous_channels = self.mult_channel[0] * self.model_channels
        att_res = 1
        channels_register = [previous_channels]
        for level, mult in enumerate(self.mult_channel):
            current_channels = mult * self.model_channels
            for _ in range(self.residual_blocks_num):
                layers: List[ResidualBlock | AttentionBlock] = [
                    ResidualBlock(
                        previous_channels,
                        self.time_embedding_dim,
                        current_channels,
                        self.dropout,
                    )
                ]
                previous_channels = current_channels
                channels_register.append(previous_channels)

                if att_res in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            previous_channels, num_heads=4, num_head_channels=-1
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))

            if level != len(self.mult_channel) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResidualBlock(
                            current_channels,
                            self.time_embedding_dim,
                            current_channels,
                            self.dropout,
                            downsample=True,
                        )
                    )
                )
                att_res *= 2
                channels_register.append(previous_channels)
        # Define middle block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(
                previous_channels,
                self.time_embedding_dim,
                previous_channels,
                self.dropout,
            ),
            AttentionBlock(previous_channels, num_heads=4, num_head_channels=-1),
            ResidualBlock(
                previous_channels,
                self.time_embedding_dim,
                previous_channels,
                self.dropout,
            ),
        )

        # Define the output blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.mult_channel))[::-1]:
            current_channels = mult * self.model_channels
            for i in range(self.residual_blocks_num + 1):
                res_channel = channels_register.pop()
                layers = [
                    ResidualBlock(
                        previous_channels + res_channel,
                        self.time_embedding_dim,
                        current_channels,
                        self.dropout,
                    )
                ]
                previous_channels = current_channels

                if att_res in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            previous_channels,
                            num_heads=4,
                            num_head_channels=-1,
                        )
                    )
                if level and i == self.residual_blocks_num:
                    layers.append(
                        ResidualBlock(
                            current_channels,
                            self.time_embedding_dim,
                            current_channels,
                            self.dropout,
                            upsample=True,
                        )
                    )
                    att_res = int(att_res // 2)

                self.output_blocks.append(TimestepEmbedSequential(*layers))
        self.out = nn.Sequential(
            nn.GroupNorm(32, previous_channels),
            nn.SiLU(),
            zero_module(nn.Conv2d(previous_channels, output_channels, 3, padding=1)),
        )

    def forward(
        self,
        input_tensor: torch.Tensor,  # Expected shape B x 3 x H x W
        timesteps: torch.Tensor,  # Expected shape B x 1 or B
        y: torch.Tensor | None = None,  # Expected shape B x 1
    ) -> torch.Tensor:  # Expected shape B x 3 x H x W
        embeded_timesteps = self.time_embedding_mlp(
            encode_timestep(timesteps, dimension=self.model_channels)
        )

        if self.num_classes:
            embeded_timesteps = embeded_timesteps + self.label_emb(y)

        res_connection: List[torch.Tensor] = []
        for module in self.input_blocks:
            input_tensor = module(input_tensor, embeded_timesteps)
            res_connection.append(input_tensor)

        output_tensor = self.middle_block(input_tensor, embeded_timesteps)

        for module in self.output_blocks:
            output_tensor = torch.cat([output_tensor, res_connection.pop()], dim=1)
            output_tensor = module(output_tensor, embeded_timesteps)

        output_tensor = self.out(output_tensor)

        return output_tensor


# RUN = True
# PATH_ROOT = "/Users/aimans/Storage/consistency_models/"
# if RUN:
#     timesteps_s = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.long)
#     input_tensor_s = torch.randn(10, 3, 64, 64)
#     unet_inputs = {
#         "input_channels": 3,
#         "output_channels": 3,
#         "model_channels": 192,
#         "mult_channel": [1, 2, 3, 4],
#         "residual_blocks_num": 3,
#         "dropout": 0.1,
#         "attention_resolutions": [2, 4, 8],
#         "num_classes": 1000,
#     }
#     unet = Unet(**unet_inputs)

#     print(unet(input_tensor_s, timesteps_s, timesteps_s).size())
# # torch.save(unet.state_dict(), PATH_ROOT + 'unet.pt')
# params = unet.state_dict()
# test_params_unet = {
#     k: v for k, v in params.items() if "down" not in k and "upsam" not in k
# }
# print("len custom unet param", len(list(unet.state_dict().keys())))

# edm_params = torch.load(PATH_ROOT + "edm_imagenet64_ema.pt")  # type: ignore
# print("len edm unet param after removing label layer", len(edm_params.keys()))

# mapping = {
#     "embedding_layer": "emb_layers",
#     "time_embedding_mlp": "time_embed",
#     "output_layer": "out_layers",
#     "input_layer": "in_layers",
# }

# renamed_dict = {}
# for key, value in edm_params.items():
#     new_key = key  # Start with the original key
#     for new_substring, old_substring in mapping.items():
#         if old_substring in new_key:
#             # Replace the substring if found in the key
#             new_key = new_key.replace(old_substring, new_substring)

#     # Add the renamed key and its corresponding value to the new dictionary
#     renamed_dict[new_key] = value

# for element in list(unet.state_dict().keys())[:300]:
#     print(element)
#     print(
#         "#" * 40,
#         unet.state_dict()[element].size(),
#         renamed_dict[element].size(),
#         unet.state_dict()[element].size() == renamed_dict[element].size(),
#     )
