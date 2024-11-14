from typing import Union, Optional, Literal, TYPE_CHECKING

from torch import Tensor
from torch.nn import Module
import torch

from .score_model import ScoreModel
from ..sde import SDE
from ..losses import edm_dsm
from ..solver import Solver, ODESolver
from ..utils import DEVICE
if TYPE_CHECKING:
    from score_models import HessianDiagonal


__all__ = ["EDMScoreModel"]


class EDMScoreModel(ScoreModel):
    def __init__(
        self,
        net: Optional[Union[str, Module]] = None,
        sde: Optional[Union[str, SDE]] = None,
        path: Optional[str] = None,
        checkpoint: Optional[int] = None,
        hessian_diagonal_model: Optional["HessianDiagonal"] = None,
        device=DEVICE,
        **hyperparameters
    ):
        super().__init__(net, sde, path, checkpoint=checkpoint, device=device, **hyperparameters)

    def loss(self, x, *args, step: int) -> Tensor:
        return edm_dsm(self, x, *args)

    def reparametrized_score(self, t, x, *args, **kwargs) -> Tensor:
        B, *D = x.shape
        c_in = self.sde.c_in(t).view(B, *[1]*len(D))
        return self.net(t, c_in * x, *args, **kwargs) 
    
    def score(self, t, x, *args, **kwargs) -> Tensor:
        """
        Score function is defined using Tweedie formula and the preconditioned denoiser
        """
        B, *D = x.shape
        x0 = self.preconditioned_denoiser(t, x, *args, **kwargs) # Estimate of E[x0 | xt]
        sigma = self.sde.sigma(t).view(B, *[1]*len(D))
        mu = self.sde.mu(t).view(B, *[1]*len(D))
        return (mu * x0 - x) / sigma**2

    def preconditioned_denoiser(self, t, x: Tensor, *args, **kwargs) -> Tensor:
        B, *D = x.shape
        # Broadcast the coefficients to the shape of x
        c_in = self.sde.c_in(t).view(B, *[1]*len(D))
        c_out = self.sde.c_out(t).view(B, *[1]*len(D))
        c_skip = self.sde.c_skip(t).view(B, *[1]*len(D))
        F_theta = self.net(t, c_in * x, *args, **kwargs)
        return c_skip * x + c_out * F_theta
    
    def sample_noise_level(self, t: Tensor) -> Tensor:
        # return torch.rand_like(t) * (self.sde.T - self.sde.epsilon) + self.s # Uniform
        ...

    def tweedie(self, t: Tensor, x: Tensor, *args, **kwargs) -> Tensor:
        return self.preconditioned_denoiser(t, x, *args, **kwargs)

    def forward(self, t, x, *args, **kwargs):
        return self.score(t, x, *args, **kwargs)

