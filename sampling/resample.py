import numpy as np
import torch


class UniformSampler:
    def __init__(self, num_steps: int):
        self.num_steps = num_steps

    def sample(self, batch_size):
        weights = np.ones([self.num_steps])
        probabilities = weights / np.sum(weights)
        indices_np = np.random.choice(
            len(probabilities), size=(batch_size,), p=probabilities
        )
        indices = torch.from_numpy(indices_np).long()  # TODO add device
        weights_np = 1 / (len(probabilities) * probabilities[indices_np])
        weights = torch.from_numpy(weights_np).float()  # TODO add device

        return indices, weights
