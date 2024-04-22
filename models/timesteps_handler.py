# Few helpers

import math

import torch


def encode_timestep(
    timesteps: torch.Tensor, dimension: int, max_period: int = 10000
) -> torch.Tensor:
    """
    Encode timesteps into a sinusoidal embedding with a specified dimensionality using
    exponentially scaled frequencies. This function generates sinusoidal embeddings for a given
    tensor of timesteps. The frequencies of the sinusoids are scaled exponentially based on the
    maximum period specified. It handles both even and odd dimensions by appending a zero column
    if necessary.

    Parameters:
    timesteps (torch.Tensor): A tensor containing timesteps at which to evaluate the sinusoids.
                              This tensor should typically be 1-dimensional.
    dimension (int): The total dimension of the output embeddings. If the dimension is odd,
                     the output is padded with a zero column to match the requested dimension.
    max_period (int, optional): The base period used to calculate the frequency scaling.
                                Defaults to 10000, affecting the rate of frequency decay.

    Returns:
    torch.Tensor: A tensor of shape [N, dimension] where N is the number of timesteps provided.
                  Each row represents the sinusoidal embedding of the corresponding timestep.
    """

    half_dimension = dimension // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half_dimension, dtype=torch.float32)
        / half_dimension
    ).to(device=timesteps.device)
    full_frequencies = 2 * math.pi * frequencies * (timesteps[:, None].float())
    encoded_timesteps = torch.cat(
        [torch.sin(full_frequencies), torch.cos(full_frequencies)], dim=1
    )
    if dimension % 2:
        encoded_timesteps = torch.cat(
            [encoded_timesteps, torch.zeros_like(encoded_timesteps[:, :1])], dim=1
        )
    return encoded_timesteps
