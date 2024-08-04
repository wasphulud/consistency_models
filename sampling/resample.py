from types import UnionType
from typing import Tuple, Union

import numpy as np
import torch


class UniformSampler:
    def __init__(self, num_steps: int):
        self.num_steps = num_steps

    def sample(
        self, batch_size: int, device: Union[str, torch.device, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = np.ones([self.num_steps])
        probabilities = weights / np.sum(weights)
        indices_np = np.random.choice(
            len(probabilities), size=(batch_size,), p=probabilities
        )
        indices = torch.from_numpy(indices_np).long().to(device)  # TODO add device
        weights_np = 1 / (len(probabilities) * probabilities[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)  # TODO add device

        return indices, weights
