from typing import Union, Optional, Literal, Callable, TYPE_CHECKING

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
        # hessian_diagonal_model: Optional["HessianDiagonal"] = None,
        device=DEVICE,
        **hyperparameters
    ):
        # Hessian Diagonal model is not supported for EDM models
        super().__init__(net, sde, path, checkpoint=checkpoint, device=device, **hyperparameters)
        self.hyperparameters["formulation"] = "edm"

    def loss(self, x, *args) -> Tensor:
        return edm_dsm(self, x, *args)
    
    def sample_noise_level(self, B: int) -> Tensor:
        return self._uniform_noise_level_distribution(B)

    def reparametrized_score(self, t, x, *args, **kwargs) -> Tensor:
        """
        In this formulation, reparametrized score is F_theta(t, x)
        """
        B, *D = x.shape
        c_in = self.sde.c_in(t).view(B, *[1]*len(D))
        return self.net(t, c_in * x, *args, **kwargs)
    
    def score(self, t, x, *args, **kwargs) -> Tensor:
        """
        Score function is defined through Tweedie's formula and the preconditioned denoiser. 
        For the VP, we use the edm_scale function, one can look at equation 186 in 
        Karras et al. (2022) for the EDM formulation. 
        This works for all SDEs, including VE.
        """
        B, *D = x.shape
        x0 = self.preconditioned_denoiser(t, x, *args, **kwargs) # Estimate of E[x0 | xt]
        sigma = self.sde.sigma(t).view(B, *[1]*len(D))
        var = sigma**2
        return (x0 - x) / var

    def preconditioned_denoiser(self, t, x: Tensor, *args, **kwargs) -> Tensor:
        B, *D = x.shape
        # Broadcast the coefficients to the shape of x
        F_theta = self.reparametrized_score(t, x, *args, **kwargs)
        c_out = self.sde.c_out(t).view(B, *[1]*len(D))
        c_skip = self.sde.c_skip(t).view(B, *[1]*len(D))
        return c_skip * x + c_out * F_theta
    
    def _uniform_noise_level_distribution(self, B: int) -> Tensor:
        return torch.rand(B, device=self.device) * (self.sde.T - self.sde.epsilon) + self.sde.epsilon
    
    def _normal_noise_level_distribution(self, B: int) -> Tensor:
        """
        Sample noise level from a log-Normal distribution, 
        then compute the corresponding time-index for this noise level.
        This is the recommended setting for the EDM formulation.
        """
        log_sigma = torch.randn(B, device=self.device) * self.log_sigma_std + self.log_sigma_mean
        sigma = torch.exp(log_sigma)
        t = self.sde.sigma_inverse(sigma)
        return t

    def tweedie(self, t: Tensor, x: Tensor, *args, **kwargs) -> Tensor:
        return self.preconditioned_denoiser(t, x, *args, **kwargs)

    def forward(self, t, x, *args, **kwargs):
        return self.score(t, x, *args, **kwargs)
    
    def fit(
            self,
            *args,
            sample_noise_level_function: Optional[Callable] = None,
            noise_level_distribution: Literal["uniform", "normal"] = "uniform",
            log_sigma_mean: float = -1.2,
            log_sigma_std: float = 1.2,
            adaptive_loss_weights: bool = False,
            **kwargs
            ):
        if sample_noise_level_function is None:
            if noise_level_distribution == "uniform":
                print("Samplng noise level from a Uniform in [epsilon, T]")
                self.sample_noise_level = self._uniform_noise_level_distribution
            elif noise_level_distribution == "normal":
                print(f"Sampling noise level from log-Normal with mean log sigma = {log_sigma_mean} and standard deviation {log_sigma_std}")
                self.log_sigma_mean = log_sigma_mean
                self.log_sigma_std = log_sigma_std
                self.sample_noise_level = self._normal_noise_level_distribution
            else:
                raise ValueError(f"Sampling distribution {noise_level_distribution} is not recognized. Choose between 'uniform' and 'normal'.")
        else:
            print(f"Using custom function {sample_noise_level_function.__name__} to sample nosie level ") 
            self.sample_noise_level = sample_noise_level_function
        
        if adaptive_loss_weights:
            raise NotImplementedError("Adaptive loss weights are not supported for EDM models.")
        return super().fit(*args, **kwargs)

