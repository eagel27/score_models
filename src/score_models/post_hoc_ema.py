import torch
import numpy as np
import os
import glob

from .utils import DEVICE
from .save_load_utils import (
        load_architecture,
        load_checkpoint,
        checkpoint_number,
        ema_length_from_path,
        step_from_path,
        )

__all__ = ["PostHocEMA"]

def ema_lengths_from_path(path: str):
    checkpoint_paths = glob.glob(os.path.join(path, "*checkpoint*.pt"))
    return sorted(list(set([ema_length_from_path(path) for path in checkpoint_paths])))

# Algorithm 2 from Karras el al. 2024
def ema_length_to_gamma(sigma_rel):
    """ sigma_rel and ema_length are the same thing """
    t = sigma_rel ** (-2)
    gamma = np.roots([1, 7, 16 - t, 12 - t]). real.max()
    return gamma

# Algorithm 3 from Karras el al. 2024
def p_dot_p(t_a, gamma_a, t_b, gamma_b):
    t_ratio = t_a / t_b
    t_exp = np.where(t_a < t_b, gamma_b, -gamma_a)
    t_max = np.maximum(t_a, t_b)
    num = (gamma_a + 1) * (gamma_b + 1) * t_ratio ** t_exp
    den = (gamma_a + gamma_b + 1) * t_max
    return num / den

def solve_weights(t_i, gamma_i, t_r, gamma_r):
    rv = lambda x: np.float64(x).reshape(-1, 1)
    cv = lambda x: np.float64(x).reshape(1, -1)
    A = p_dot_p(rv(t_i), rv(gamma_i), cv(t_i), cv(gamma_i ))
    B = p_dot_p(rv(t_i), rv(gamma_i), cv(t_r), cv(gamma_r ))
    X = np.linalg.solve(A, B)
    return X.squeeze(-1)

class PostHocEMA:
    def __init__(self, path: str, device=DEVICE, **kwargs):
        self.path = path
        self.ema_lengths = ema_lengths_from_path(path)
        self.device = device
        self.validate_checkpoints()
        self.ema_model, self.hyperparameters = load_architecture(path, device=device)
        for p in self.ema_model.parameters():
            p.requires_grad = False
            p.zero_()
    
    def validate_checkpoints(self):
        # Is it necessary? Feels like we could synthesize models even with a single ema_length. Not guarantee to obtain a good model but still
        if len(self.ema_lengths) < 2:
            raise ValueError("At least 2 EMA lengths are required to synthesize a new model")
        paths = {}
        num_checkpoints = 0
        for ema_length in self.ema_lengths:
            pattern = f"*checkpoint_*emalength{ema_length:.2f}*.pt" # Only works for default neural net, not LoRA 
            paths[ema_length] = sorted(glob.glob(os.path.join(self.path, pattern)), key=checkpoint_number)
            num_checkpoints = max(num_checkpoints, len(paths[ema_length]))
            if len(paths[ema_length]) == 0:
                raise ValueError(f"No checkpoints found with ema_length {ema_length}")
            if len(paths[ema_length]) != num_checkpoints:
                raise ValueError("Different number of checkpoints found for the different ema_length")

    def zero_model(self):
        model, _ = load_architecture(self.path, device=self.device) # Random weights
        for p in model.parameters():
            p.requires_grad = False # Freeze the model
            p.zero_()
        return model
    
    def weights(self, paths, ema_length: float):
        if ema_length < 0 or ema_length > self.ema_lengths[-1]:
            raise ValueError(f"EMA length {ema_length} is out of bounds, should be in [{self.ema_lengths[0]}, {self.ema_lengths[-1]}]")

        # Grab ema_length and step number for each checkpoint
        gammas = torch.tensor(
                [ema_length_to_gamma(ema_length_from_path(path)) for path in paths],
                device=self.device
                )
        steps = torch.tensor([step_from_path(path) for path in paths], device=self.device)
        # Target gamma and step (last step)
        target_gamma = torch.tensor(ema_length_to_gamma(ema_length), device=self.device)
        target_step = steps.max()
        # Compute the dot product and solve for the weights
        weights = solve_weights(steps, gammas, target_step, target_gamma)
        return weights
    
    def synthesize_ema(self, ema_length: float, model_requires_grad: bool = True):
        pattern = "*checkpoint*.pt"
        paths = glob.glob(os.path.join(self.path, pattern))
        weights = self.weights(paths, ema_length)
        
        online_model = self.zero_model()

        # Sum up the weighted checkpoints
        for path, weight in zip(paths, weights):
            # Load checkpoint (this assumes we are not using a LoRA model)
            online_model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            for (ema_tensor, checkpoint_tensor) in zip(self.ema_model.parameters(), online_model.parameters()):
                ema_tensor.add_(checkpoint_tensor * weight)
        
        if model_requires_grad:
            for p in self.ema_model.parameters():
                p.requires_grad = True
        return self.ema_model
