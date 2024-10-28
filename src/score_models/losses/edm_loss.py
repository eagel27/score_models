from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    from score_models import ScoreModel, HessianDiagonal

from torch import Tensor
import torch

def edm_loss(model: "ScoreModel", x: Tensor, *args: list[Tensor], **kwargs):
    B, *D = x.shape
    sde = model.sde
    
    # t = torch.rand(B).to(model.device) * (sde.T - sde.epsilon) + sde.epsilon    # t ~ U(epsilon, T)
    t = model.sample_noise_level(B)
    z = torch.randn_like(x)                                                       # z ~ N(0, 1)
    
    # Sample from the marginal at time t using the Gaussian perturbation kernel
    mu = sde.mu(t).view(-1, *[1]*len(D))
    sigma = sde.sigma(t).view(-1, *[1]*len(D))
    n = sigma * z
    xt = mu * x + n                                                               # xt ~ p(xt | x0)
    
    # Compute the loss
    c_skip = model.c_skip(t).view(B, *[1]*len(D))
    c_out = model.c_out(t).view(B, *[1]*len(D))
    F_theta = model.preconditioned_denoiser(t, xt, *args)
    effective_score = (x - c_skip * (x + n)) / c_out
    return ((F_theta - effective_score)**2).sum() / (2 * B)
