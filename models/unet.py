from abc import abstractmethod
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import AttentionBlock
from timesteps_handler import encode_timestep


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
        self, in_channel: int, time_embed_dim: int, out_channel: int, dropout: float
    ):
        super().__init__()  # type: ignore

        self.in_channel = in_channel
        self.time_embed_dim = time_embed_dim
        self.out_channel = out_channel
        self.dropout = dropout

        self.embedding_layer = nn.Sequential(
            nn.SiLU(), nn.Linear(self.time_embed_dim, self.out_channel)
        )

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
            nn.Conv2d(
                self.out_channel, self.out_channel, 3, padding=1
            ),  # TODO: Need to zero this one
        )

        if self.in_channel == self.out_channel:
            self.skip_connection: nn.Conv2d | nn.Identity = nn.Identity()
        else:
            self.skip_connection: nn.Conv2d | nn.Identity = nn.Conv2d(
                self.in_channel, self.out_channel, 3, padding=1
            )

    def forward(self, input_tensor: torch.Tensor, timesteps_encoding: torch.Tensor):
        timesteps_embedding = self.embedding_layer(timesteps_encoding)
        transformed_input_tensor = self.input_layer(input_tensor)
        input_to_second_layer = (
            timesteps_embedding.unsqueeze(dim=2).unsqueeze(dim=3)
            + transformed_input_tensor
        )  # unqueeze timesteps embedding until it has the dimen len size than the transformed
        output_tensor = self.output_layer(input_to_second_layer) + self.skip_connection(
            input_tensor
        )

        return output_tensor


class DownSampling(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 2, padding: int = 1
    ):
        super().__init__()  # type: ignore
        self.downsampling_layer = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=padding
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.downsampling_layer(input_tensor)


class UpSampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding: int = 1):
        super().__init__()  # type: ignore
        self.conv_layer = nn.Conv2d(in_channels, out_channels, 3, padding=padding)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = F.interpolate(input_tensor, scale_factor=2, mode="nearest")  # type: ignore
        return self.conv_layer(input_tensor)


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
    ) -> None:
        super().__init__()  # type: ignore

        self.input_channel = input_channels
        self.output_channels = output_channels
        self.model_channels = model_channels
        self.mult_channel = mult_channel
        self.residual_blocks_num = residual_blocks_num
        self.dropout = dropout
        self.attention_resolutions = attention_resolutions

        # embed timesteps
        self.time_embedding_dim = 4 * model_channels
        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(model_channels, self.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )

        # input downsampling block
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(3, 128, 3, padding=1))]
        )
        previous_channels = self.mult_channel[0] * self.model_channels
        att_res = 1
        channels_register = [previous_channels]
        for level, mult in enumerate(self.mult_channel):
            current_channels = mult * self.model_channels
            for _ in range(self.residual_blocks_num):
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResidualBlock(
                            previous_channels,
                            self.time_embedding_dim,
                            current_channels,
                            self.dropout,
                        )
                    )
                )
                previous_channels = current_channels
                channels_register.append(previous_channels)

                if att_res in self.attention_resolutions:
                    self.input_blocks.append(
                        TimestepEmbedSequential(
                            AttentionBlock(
                                previous_channels, num_heads=4, num_head_channels=-1,
                            )
                        )
                    )

            if level != len(self.mult_channel) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        DownSampling(current_channels, current_channels)
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
                layers: List[nn.Module] = [
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
                            previous_channels, num_heads=4, num_head_channels=-1,
                        )
                    )
                if level and i == self.residual_blocks_num:
                    layers.append(UpSampling(current_channels, current_channels))
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
    ) -> torch.Tensor:  # Expected shape B x 3 x H x W
        embeded_timesteps = self.time_embedding_mlp(
            encode_timestep(timesteps, dimension=self.model_channels)
        )
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


timesteps_s = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
input_tensor_s = torch.randn(10, 3, 64, 64)
unet = Unet(3, 3, 128, [1, 2, 4], 4, 0.5, [])
print(unet(input_tensor_s, timesteps_s).size())
