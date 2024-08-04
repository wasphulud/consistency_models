import copy
import functools
import sys
from typing import Any, Iterator, List, Mapping, Tuple, Union

import torch
from denoising.karras_denoiser import consistency_loss
from infrastructure.device import get_device
from sampling.resample import UniformSampler
from torch import nn
from torch.optim import RAdam
from torch.utils.data import DataLoader

sys.path.append("..")


class CDTrainLoop:
    def __init__(
        self,
        model: nn.Module,
        teacher_model: nn.Module,
        target_model: nn.Module,
        dataloader: Iterator[Tuple[torch.Tensor, Mapping[str, List[int]]]],
        experiment_args: Mapping[str, Any],
        device: Union[str, torch.device, int],
    ):  # type: ignore

        self.current_step = 0

        self.experiment_args = experiment_args

        self.sigma_data = self.experiment_args["sigma_data"]
        self.sigma_max = self.experiment_args["sigma_max"]
        self.sigma_min = self.experiment_args["sigma_min"]
        self.rho = self.experiment_args["rho"]

        self.total_training_steps = self.experiment_args["total_training_steps"]
        self.batch_size = experiment_args["batch_size"]
        self.sampler = UniformSampler(self.experiment_args["num_steps"])

        self.device = device
        self.model = model
        self.model.to(device)

        self.teacher_model = teacher_model
        self.teacher_model.to(device)

        self.target_model = target_model
        self.model.to(device)

        self.dataloader = dataloader

        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()

        self.target_model.requires_grad_(False)
        self.target_model.train()

        self.opt = RAdam(
            list(self.model.parameters()),
            lr=experiment_args["lr"],
            weight_decay=experiment_args["weight_decay"],
        )

        ema_rate = experiment_args["ema_rate"]

        self.ema_rate = (
            [ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate]
        )

        self.ema_params = [
            copy.deepcopy(list(self.model.parameters()))
            for _ in range(len(self.ema_rate))
        ]
        self.ema_fn = create_ema_and_scales_fn(
            experiment_args["start_ema"], experiment_args["start_scales"]
        )

    def train(self):

        while self.current_step < self.total_training_steps:
            batch, cond = next(self.dataloader)
            print("batch loaded")
            self.run_step(batch, cond)
            print("step ran")

    def run_step(self, batch: torch.Tensor, cond: Mapping[str, List[int]]):
        self.opt.zero_grad()
        self.forward_backward(batch, cond)
        self.opt.step()
        self._update_ema()
        self._update_target_ema()
        self.current_step += 1

    def forward_backward(
        self,
        batch: torch.Tensor,
        cond: Mapping[str, List[int]],
    ):
        timestep, weights = self.sampler.sample(batch.shape[0])

        _, num_scales = self.ema_fn()

        loss: torch.Tensor = functools.partial(
            consistency_loss,  # TODO implement consistenncy_losses
            self.model,
            batch,
            num_scales,
            self.target_model,
            self.teacher_model,
            cond,
            self.sigma_data,
            self.sigma_min,
            self.sigma_max,
            self.rho,
        )()  # TODO This has no sense to use partial func like this, it might have more sense if we are using ddp or/and cuda - we will keep it for now
        loss = (loss * weights).mean()
        print("got loss", loss)

        loss.backward()
        print("backward run")

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            self.update_ema(params, self.model.parameters(), rate=rate)

    def update_ema(self, target_params, source_params, rate=0.99):
        """
        Update target parameters to be closer to those of source parameters using
        an exponential moving average.

        :param target_params: the target parameter sequence.
        :param source_params: the source parameter sequence.
        :param rate: the EMA rate (closer to 1 means slower).
        """

        for targ, src in zip(target_params, source_params):
            targ.detach().mul_(rate).add_(src, alpha=1 - rate)

    # TODO is there a need to use mastre paramters like in the original repo ?
    def _update_target_ema(self):
        target_ema, _ = self.ema_fn()
        with torch.no_grad():
            self.update_ema(
                self.target_model.parameters(),
                self.model.parameters(),
                rate=target_ema,
            )


def initialize_inputs(dataloader: DataLoader, model: nn.Module, teacher_path: str):  # type: ignore
    model.train()
    teacher_model = copy.deepcopy(model)
    # teacher_model.load_state_dict(torch.load(teacher_path))  # type: ignore
    teacher_model.eval()

    target_model = copy.deepcopy(model)
    target_model.train()

    experiment_args: Mapping[str, Any] = {
        "total_training_steps": 1000,
        "start_ema": 0.95,
        "start_scales": 40,
        "batch_size": 10,
        "lr": 0.000008,
        "weight_decay": 0.0,
        "ema_rate": [0.999, 0.9999, 0.9999432189950708],
        "num_steps": 40,
        "sigma_data": 0.5,
        "sigma_max": 80.0,
        "sigma_min": 0.002,
        "rho": 7.0,
    }

    CDTrainLoop(
        model, teacher_model, target_model, dataloader, experiment_args, get_device()
    ).train()


def create_ema_and_scales_fn(start_ema, start_scales):
    def ema_and_scales_fn():
        return float(start_ema), int(start_scales)

    return ema_and_scales_fn
