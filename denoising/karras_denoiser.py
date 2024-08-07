from re import A

import numpy as np
import numpy.typing as npt
import torch
from torch import nn


def consistency_loss(
    model: nn.Module,
    x_start: torch.Tensor,
    num_scales: int,
    target_model,
    teacher_model,
    model_cond,
    sigma_data,
    sigma_min,
    sigma_max,
    rho,
):
    noise = torch.randn_like(x_start)
    x_dims = x_start.ndim

    def denoise_fn(x, t):  # denoise using model
        return denoise(model, x, t, sigma_data, sigma_min, **model_cond)[1]

    @torch.no_grad()
    def target_denoise_fn(x, t):  # denoise using target_model
        return denoise(target_model, x, t, sigma_data, sigma_min, **model_cond)[1]

    @torch.no_grad()
    def teacher_denoise_fn(x, t):  # denoise using teacher_model
        return denoise(teacher_model, x, t, sigma_data, sigma_min, **model_cond)[1]

    @torch.no_grad()
    def euler_solver(samples, t, next_t):
        x = samples
        denoised = teacher_denoise_fn(x, t)  # denoiser (function of the neural network)

        # increase t dimensions to match the batch and compute the derivation term (x_f - x_i) / (t_f- t_i) # TODO is this the right formula ? why do we divide only by t and not the step as the denoised image is generated by the teacher and is not supposed to be the fully denoised one
        diff = (x - denoised) / append_dims(t, x_dims)  # evaluate dx/dt at t_i
        # compute the next values using euler formula y_next = y_now + step * dy/dx
        samples = samples + diff * append_dims(
            next_t - t, x_dims
        )  # euler step from t_i to t_{i+1}
        return samples

    @torch.no_grad()
    def heun_solver(samples, t, next_t):
        x = samples
        denoised = teacher_denoise_fn(x, t)  # denoiser
        diff = (x - denoised) / append_dims(
            t, x_dims
        )  # evaluate dx/dt at t_i (= -t * score = (x-Denoised)/t see EDM paper)
        samples = x + diff * append_dims(
            next_t - t, x_dims
        )  # take euler step from t_i to t_{i+1}
        # apply  2nd order correction
        denoised = teacher_denoise_fn(samples, next_t)  # denoiser at t_{i+1}
        next_diff = (samples - denoised) / append_dims(
            next_t, x_dims
        )  # eval dx/dt at t_{i+1}
        samples = x + (diff + next_diff) * append_dims(
            (next_t - t) / 2, x_dims
        )  # explicit trapezoidal rule at t_{i+1}

        return samples

    indices = torch.randint(
        0, num_scales - 1, (x_start.shape[0],), device=x_start.device
    )  # TODO move to device

    t, t2 = get_timesteps(rho, sigma_min, sigma_max, indices, num_scales)

    x_t = x_start + noise * append_dims(t, x_dims)
    print("noisy data generated")

    dropout_state = torch.get_rng_state()  # to control the state generator later

    distiller = denoise_fn(x_t, t)  # f_{theta} (x_t_{n+1},t_{n+1})
    print("first dist")
    x_t2 = heun_solver(
        x_t, t, t2
    ).detach()  # =x^phi_{t_{n}} : one discretization step of numerical solver
    print("one step disc in num solver")

    torch.set_rng_state(dropout_state)  # to control the state generator

    distiller_target = target_denoise_fn(
        x_t2, t2
    ).detach()  # f_{theta-} (x^phi_{t_{n}},t_{n})
    print("second dist")

    snrs = get_snr(t)
    weights = torch.ones_like(snrs)  # uniform weighting

    diffs = (distiller - distiller_target) ** 2  # l2 norm
    loss = mean_flat(diffs) * weights

    return loss


def get_timesteps(rho, sigma_min, sigma_max, indices, num_scales):
    t = sigma_max ** (1 / rho) + indices / (num_scales - 1) * (
        sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
    )
    t = t**rho

    t2 = sigma_max ** (1 / rho) + (indices + 1) / (num_scales - 1) * (
        sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
    )
    t2 = t2**rho
    return t, t2


def denoise(model, x_t, sigmas, sigma_data, sigma_min, **model_cond):
    c_skip, c_out, c_in = [
        append_dims(x, x_t.ndim)
        for x in get_scalings_for_boundary_condition(
            sigmas,
            sigma_data,
            sigma_min,
        )
    ]

    rescaled_t = (
        1000 * 0.25 * torch.log(sigmas + 1e-44)
    )  # TODO undestand why time step is rescaled as such: https://github.com/openai/consistency_models/issues/12#issuecomment-1513213098

    model_output = model(c_in * x_t, rescaled_t, **model_cond)
    denoised = c_out * model_output + c_skip * x_t
    return model_output, denoised


def get_scalings_for_boundary_condition(sigmas, sigma_data, sigma_min):
    c_skip = sigma_data**2 / ((sigmas - sigma_min) ** 2 + sigma_data**2)
    c_out = (sigmas - sigma_min) * sigma_data / (sigmas**2 + sigma_data**2) ** 0.5
    c_in = 1 / (sigmas**2 + sigma_data**2) ** 0.5

    return c_skip, c_out, c_in


def append_dims(x: torch.Tensor, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def get_snr(sigmas):
    return sigmas**-2


def mean_flat(tensor: torch.Tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
