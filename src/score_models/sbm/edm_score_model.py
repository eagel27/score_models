from typing import Union, Optional, Literal, TYPE_CHECKING

from torch import Tensor
from torch.nn import Module
import torch

from .base import Base
from ..sde import SDE
from ..losses import dsm
from ..solver import Solver, ODESolver
from ..utils import DEVICE
if TYPE_CHECKING:
    from score_models import HessianDiagonal


__all__ = ["ScoreModel"]


class EDMScoreModel(Base):
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
        if hessian_diagonal_model is not None:
            self.dlogp = hessian_diagonal_model.dlogp
        else:
            self.dlogp = None

    def loss(self, x, *args, step: int) -> Tensor:
        return dsm(self, x, *args)
    
    def c_in(self, t: Tensor,  *args, **kwargs) -> Tensor:
        ...
    
    def c_out(self, t: Tensor,  *args, **kwargs) -> Tensor:
        ...

    def c_skip(self, t: Tensor, *args, **kwargs) -> Tensor:
        ...

    def c_noise(self, t: Tensor,  *args, **kwargs) -> Tensor:
        ...

    def reparametrized_score(self, t, x: Tensor, *args, **kwargs) -> Tensor:
        ...
        

    def forward(self, t, x, *args, **kwargs):
        """
        Overwrite the forward method to return the score function instead of the model output.
        This also affects the __call__ method of the class, meaning that
        ScoreModel(t, x, *args) is equivalent to ScoreModel.forward(t, x, *args).
        """
        return self.score(t, x, *args, **kwargs)

    def score(self, t, x, *args, **kwargs) -> Tensor:
        ...

    def tweedie(self, t: Tensor, x: Tensor, *args, **kwargs) -> Tensor:
        ...
