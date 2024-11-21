from typing import Optional, Callable, TYPE_CHECKING, Union
if TYPE_CHECKING:
    from score_models import ScoreModel

import torch
import json, os
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
# from torch_ema import ExponentialMovingAverage
from ema_pytorch import EMA, KarrasEMA
from datetime import datetime
from tqdm import tqdm

from .utils import DEVICE
from .save_load_utils import (
        remove_oldest_checkpoint, 
        last_checkpoint,
        load_checkpoint,
        load_global_step
        )


__all__ = ["Trainer"]

def inverse_sqrt_schedule(step: int, learning_rate_decay: Optional[int] = None, warmup: int = 0):
    if learning_rate_decay is None:
        return 1.0
    return 1 / np.sqrt(max((step - warmup) / learning_rate_decay, 1))

def warmup_schedule(step: int, warmup: int = 0):
    if warmup == 0:
        return 1.0
    return np.minimum(step / warmup, 1.0)


class Trainer:
    def __init__(
        self,
        model: "ScoreModel",
        dataset: Dataset,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        learning_rate: float = 1e-3,
        learning_rate_decay: Optional[int] = None, # Number of epochs before decaying learning rate with a square root decay schedule
        ema_lengths: Union[float, tuple] = 0.13, # Karras EMA
        ema_decay: Optional[float] = None, # Traditional EMA
        update_ema_after_step: int = 100, # Parameter to delay update of traditional EMA
        update_model_with_ema_every: Optional[int] = None, # Parameter to reset the online model with EMA ala Hare and Tortoise (https://arxiv.org/abs/2406.02596)
        clip: float = 1.,
        warmup: int = 0,
        shuffle: bool = False,
        iterations_per_epoch: Optional[int] = None,
        max_time: float = float('inf'),
        checkpoint_every: int = 10,
        models_to_keep: int = 1,
        total_checkpoints_to_save: Optional[int] = None,
        path: Optional[str] = None,
        name_prefix: Optional[str] = None,
        seed: Optional[int] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        ): 
        # Model
        self.model = model
        self.net = model.net # Neural network to train
        
        # Gradient clipping
        self.clip = clip
        
        # Dataset
        if batch_size is not None:
            self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        else:
            self.dataloader = dataset
        self.data_iter = iter(self.dataloader)
        
        # Optimizer
        self.lr = learning_rate
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.iterations_per_epoch = iterations_per_epoch or len(self.dataloader)
        
        # Exponential Moving Averages, with Karras prescription
        if ema_decay:
            print("It is recommended to use the Karras EMA with ema_lengths instead of the traditional EMA."
                  " ema_lengths is set to 0.13 by default, a number between 0 and 1. Set ema_decay to None (default) to use Karras EMA.")
            # self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay) # torch_ema
            self.emas = [EMA(self.model, beta=ema_decay, update_after_step=update_ema_after_step)] # ema_pytorch
            self.ema_lengths = [None]
        else:
            if not isinstance(ema_lengths, (list, tuple)):
                ema_lengths = [ema_lengths]
            for sigma_rel in ema_lengths:
                assert sigma_rel > 0, "ema_length must be a positive float."
                assert sigma_rel < 0.28, "ema_length must be less than 12^{-0.5}, see algorithm 2 from Karras et al. 2024."
            self.emas = [KarrasEMA(self.model, sigma_rel=sigma_rel) for sigma_rel in ema_lengths] # ema_pytorch
            self.ema_lengths = ema_lengths
            print(f"Using Karras EMA with ema lengths [" + ",".join([f"{sigma_rel:.2f}" for sigma_rel in ema_lengths]) + "]")
        
        # Learning rate schedule
        self.global_step = 0
        if learning_rate_decay:
            assert learning_rate_decay > 0, "learning_rate_decay must be a positive integer."
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            # Use self.global_step, so that we can reload training from a checkpoint and have the correct learning rate
            lr_lambda=lambda step: 
                warmup_schedule(self.global_step, warmup) * inverse_sqrt_schedule(self.global_step, learning_rate_decay, warmup)
        )
        if seed:
            torch.manual_seed(seed)
            
        # Logic to save checkpoints
        self.epochs = epochs
        self.checkpoint_every = checkpoint_every
        self.models_to_keep = models_to_keep
        self.max_time = max_time
        if total_checkpoints_to_save:
            # Implement some logic to recalculate the total number of checkpoints to save based on resources (epochs, max_time, etc.)
            # This is super useful for PostHocEMA, where we want to save a specific number of snapshots.
            # The logic here will override the models_to_keep parameter and checkpoint_every parameter.
            raise NotImplementedError("total_checkpoints_to_save is not implemented yet.")
        
        # Provided model already has a path to load a checkpoint from
        if path and self.model.path:
            print(f"Loading a checkpoint from the model path {self.model.path} and saving in new path {path}...")
        if self.model.path:
            if not os.path.isdir(self.model.path): # Double check the path is valid
                print(f"Provided path {self.model.path} is not a valid directory. Can't load checkpoint.")
            else:
                checkpoint = load_checkpoint(
                        model=self.optimizer, 
                        checkpoint=self.model.loaded_checkpoint, 
                        path=self.model.path, 
                        key="optimizer",
                        device=self.model.device
                        )
                # Resume global step
                self.global_step = load_global_step(self.model.path, self.model.loaded_checkpoint)
                print(f"Resumed training from checkpoint {checkpoint}.")
            
        # Create a new checkpoint and save checkpoint there
        if path:
            self.path = path
            if name_prefix: # Instantiate a new model, stamped with the current time
                model_name = name_prefix + "_" + datetime.now().strftime("%y%m%d%H%M%S")
                self.path = os.path.join(self.path, model_name)
            else:
                model_name = os.path.split(self.path)[-1]
            if not os.path.isdir(self.path):
                os.makedirs(self.path, exist_ok=True)

            # Save Training parameters
            file = os.path.join(self.path, "script_params.json")
            if not os.path.isfile(file):
                with open(file, "w") as f:
                    json.dump(
                        {
                            "dataset": dataset.__class__.__name__,
                            "optimizer": self.optimizer.__class__.__name__,
                            "learning_rate": self.lr,
                            "ema_decay": ema_decay,
                            "learning_rate_decay": learning_rate_decay,
                            "iterations_per_epoch": iterations_per_epoch,
                            "batch_size": batch_size,
                            "shuffle": shuffle,
                            "epochs": epochs,
                            "max_time": max_time,
                            "warmup": warmup,
                            "clip": clip,
                            "checkpoint_every": checkpoint_every,
                            "models_to_keep": models_to_keep,
                            "seed": seed,
                            "path": str(path),
                            "model_name": model_name,
                            "name_prefix": name_prefix,
                        },
                        f,
                        indent=4
                    )
            # Save model hyperparameters to reconstruct the model later
            self.model.save_hyperparameters(path)
            
            # Create the loss.txt file
            file = os.path.join(self.path, "loss.txt")
            if not os.path.isfile(file):
                with open(file, "w") as f:
                    f.write("checkpoint step loss time_per_step\n")
            else:
                # Grab the last checkpoint and step
                self.global_step = load_global_step(self.path)
        
        elif self.model.path:
            # Continue saving checkpoints in the model path
            self.path = self.model.path
        else:
            self.path = None
            print("No path provided. Training checkpoints will not be saved.")

    def save_checkpoint(self, loss: float, time_per_step):
        """
        Save model and optimizer if a path is provided. Then save loss and remove oldest checkpoints
        when the number of checkpoints exceeds models_to_keep.
        """
        if self.path:
            ## torch_ema
            # with self.ema.average_parameters():
                # self.model.save(self.path, optimizer=self.optimizer)
            ## ema_pytorch
            for ema_length, ema in zip(self.ema_lengths, self.emas):
                ema.ema_model.save(self.path, optimizer=self.optimizer, ema_length=ema_length, step=self.global_step)
        
            checkpoint = last_checkpoint(self.path)
            with open(os.path.join(self.path, "loss.txt"), "a") as f:
                f.write(f"{checkpoint} {self.global_step} {loss} {time_per_step}\n")
                
            if self.models_to_keep:
                remove_oldest_checkpoint(self.path, self.models_to_keep)
    
    def train_epoch(self):
        time_per_step_avg = 0
        cost = 0
        for _ in range(self.iterations_per_epoch):
            start = time.time()
            # Load data
            try:
                X = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.dataloader) # Reset the iterator
                X = next(self.data_iter)
            if isinstance(X, (list, tuple)): # Handle conditional arguments
                x, *args = X
            else:
                x = X
                args = []
            # Training step
            self.optimizer.zero_grad()
            loss = self.model.loss(x, *args, step=self.global_step)
            loss.backward()
            if self.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
            self.optimizer.step()
            for ema in self.emas:
                ema.update()
            self.lr_scheduler.step()
            # Logging
            time_per_step_avg += time.time() - start
            cost += loss.item()
            self.global_step += 1
        cost /= self.iterations_per_epoch 
        time_per_step_avg /= self.iterations_per_epoch
        return cost, time_per_step_avg
        
    def train(self, verbose=0) -> list:
        losses = []
        global_start = time.time()
        estimated_time_for_epoch = 0
        out_of_time = False
        for epoch in (pbar := tqdm(range(self.epochs))):
            if (time.time() - global_start) > self.max_time * 3600 - estimated_time_for_epoch:
                break
            # Train
            epoch_start = time.time()
            cost, time_per_step_avg = self.train_epoch()
            # Logging
            losses.append(cost)
            pbar.set_description(f"Epoch {epoch + 1:d} | Cost: {cost:.1e} |")
            if verbose >= 2:
                print(f"epoch {epoch} | cost {cost:.2e} | time per step {time_per_step_avg:.2e} s", flush=True)
            elif verbose == 1:
                if (epoch + 1) % self.checkpoints == 0:
                    print(f"epoch {epoch} | cost {cost:.1e}", flush=True)
            if np.isnan(cost):
                print("Model exploded and returns NaN")
                break
            if (time.time() - global_start) > self.max_time * 3600:
                out_of_time = True
            if (epoch + 1) % self.checkpoint_every == 0 or epoch == self.epochs - 1 or out_of_time:
                self.save_checkpoint(cost, time_per_step_avg)
            if out_of_time:
                print("Out of time")
                break
            if epoch > 0:
                estimated_time_for_epoch = time.time() - epoch_start

        print(f"Finished training after {(time.time() - global_start) / 3600:.3f} hours.")
        # Save EMA weights in the model for dynamic use (e.g. Jupyter notebooks)
        # self.ema.copy_to(self.model.parameters()) ## torch_ema
        self.emas[0].copy_params_from_ema_to_model() ## ema_pytorch
        return losses
