from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-12


# Eq. (3): kappa_{i,k}
def relu_second_moment(mean: Tensor, std: Tensor) -> Tensor:
    std = std.abs().clamp_min(_EPS)
    a = mean / std
    normal = torch.distributions.Normal(
        torch.zeros((), dtype=mean.dtype, device=mean.device),
        torch.ones((), dtype=mean.dtype, device=mean.device),
    )
    cdf = normal.cdf(a)
    pdf = torch.exp(normal.log_prob(a))
    return (mean.pow(2) + std.pow(2)) * cdf + mean * std * pdf
