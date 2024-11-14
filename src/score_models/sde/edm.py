from typing import Literal

from torch import Tensor
from torch.distributions import Independent, Normal
from torch.func import vmap, grad
from .sde import SDE
from ..utils import DEVICE
import torch
import numpy as np

__all__ = ["EDMSDE"]


class EDMSDE(SDE):
    def __init__(
        self,
        sigma_min: float = 1e-2,
        sigma_max: float = 100,
        rho: int = 7, # Karras et al. recommended value for sampling. Set to rho=3 to minimize errors at low temperatures.
        T: float = 1.0,
        epsilon: float = 0,
        **kwargs,
    ):
        super().__init__(T, epsilon, **kwargs)
        self.rho = rho
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def mu(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)

    def sigma(self, t: Tensor) -> Tensor:
        """
        Table 1 of Karras et al. 2022: Elucidating the Design Space of Diffusion-Based Generative Models,
        EDM column, time-steps row. Our implementation has sigma(t) and time steps interchanged.
        Also, our schedule is such that t=0 corresponds to sigma_min instead of sigma_max (reverse of Karras et al. 2022).
        """
        sigma_diff = self.sigma_max**(1/rho) - self.sigma_min**(1/rho)
        return (self.sigma_min**(1/rho) + (t/self.T) * sigma_diff)**rho
        
    def diffusion(self, t: Tensor, x: Tensor) -> Tensor:
        B, *D = x.shape
        sigma = self.sigma(t)
        dsigma_dt = vmap(grad(self.sigma))(t)
        g = (2 * dsigma_dt * sigma)**(1/2)
        return g.view(-1, *[1]*len(D))

    def drift(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    def inv_beta_primitive(self, beta: Tensor, beta_max, *args) -> Tensor:
        ...

    def t_sigma(self, sigma: Tensor) -> Tensor:
        ...

    def prior(self, shape, device=DEVICE):
        ...
