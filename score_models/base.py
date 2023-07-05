import torch
from torch.nn import Module
from .utils import DEVICE
from score_models.sde import VESDE, VPSDE
from typing import Union
from .utils import load_architecture
from abc import ABC, abstractmethod
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
import time
import os, glob, re, json
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, key, extension="h5", device=DEVICE):
        self.device = device
        self.key = key
        if extension in ["h5", "hdf5"]:
            self._data = h5py.File(self.filepath, "r")
            self.data = lambda index: self._data[self.key][index]
            self.size = self._data[self.key].shape[0]
        elif extension == "npy":
            self.data = np.load(path)
            self.size = self.data.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.tensor(self.data[[index]], dtype=DTYPE).to(self.device)


# TODO support data parallel
class ScoreModelBase(Module, ABC):
    def __init__(self, model: Union[str, Module]=None, checkpoints_directory=None, device=DEVICE, **hyperparameters):
        super().__init__()
        if model is None or isinstance(model, str):
            model, hyperparams = load_architecture(checkpoints_directory, model=model, device=device, hyperparameters=hyperparameters)
            hyperparameters.update(hyperparams)
        if "sde" not in hyperparameters.keys():
            if "sigma_min" in hyperparameters.keys():
                hyperparameters["sde"] = "vesde"
            elif "beta_min" in hyperparameters.keys():
                hyperparameters["sde"] = "vpsde"
        if hyperparameters["sde"].lower() == "vesde":
            sde = VESDE(sigma_min=hyperparameters["sigma_min"], sigma_max=hyperparameters["sigma_max"])
        elif hyperparameters["sde"].lower() == "vpsde":
            sde = VPSDE(beta_min=hyperparameters["beta_min"], beta_max=hyperparameters["beta_max"])
        else:
            raise ValueError("sde parameters missing from hyperparameters")
        self.checkpoints_directory = checkpoints_directory
        self.model = model
        self.model.to(device)
        self.sde = sde
        self.device = device

    def forward(self, t, x):
        return self.score(t, x)
   
    @abstractmethod
    def score(self, t, x):
        ...

    @torch.no_grad()
    def sample(self, size, N: int):
        # A simple Euler-Maruyama integration of the model SDE
        x = self.sde.prior(size).to(self.device)
        dt = -1.0 / N
        t = torch.ones(size[0]).to(self.device)
        for _ in tqdm(range(N)):
            t += dt
            drift = self.sde.drift(t, x)
            diffusion = self.sde.diffusion(t, x)
            score = self.score(t, x)
            drift = drift - diffusion**2 * score
            z = torch.randn_like(x)
            x_mean = x + drift * dt
            x = x_mean + diffusion * (-dt)**(1/2) * z
        return x_mean
    
    def fit(
        self,
        dataset=None,
        dataset_path=None,
        dataset_extension="h5",
        dataset_key=None,
        preprocessing_fn=None,
        epochs=100,
        learning_rate=1e-4,
        ema_decay=0.9999,
        batch_size=1,
        shuffle=False,
        patience=float('inf'),
        tolerance=0,
        max_time=float('inf'),
        epsilon=1e-5,
        warmup=0,
        clip=0.,
        checkpoints_directory=None,
        model_checkpoint=None,
        checkpoints=10,
        models_to_keep=2,
        seed=None,
        model_id=None,
        logname=None,
        n_iterations_in_epoch=None,
        logname_prefixe="score_model",
        verbose=0
    ):
        """
        Train the model on the provided dataset.

        Parameters:
            dataset (torch.utils.data.Dataset): The training dataset.
            dataset_path (str, optional): The path to the dataset file. If not provided, the function will assume a default path. Default is None.
            dataset_extension (str, optional): The extension of the dataset file. Default is "h5".
            dataset_key (str, optional): The key or identifier for the specific dataset within the file. If not provided, the function will assume a default key. Defaul
            preprocessing_fn (function, optional): A function to preprocess the input data. Default is None.
            learning_rate (float, optional): The learning rate for optimizer. Default is 1e-4.
            ema_decay (float, optional): The decay rate for Exponential Moving Average. Default is 0.9999.
            batch_size (int, optional): The batch size for training. Default is 1.
            shuffle (bool, optional): Whether to shuffle the dataset during training. Default is False.
            epochs (int, optional): The number of epochs for training. Default is 100.
            patience (float, optional): The patience value for early stopping. Default is infinity.
            tolerance (float, optional): The tolerance value for early stopping. Default is 0.
            max_time (float, optional): The maximum training time in hours. Default is infinity.
            epsilon (float, optional): The epsilon value. Default is 1e-5.
            warmup (int, optional): The number of warmup iterations for learning rate. Default is 0.
            clip (float, optional): The gradient clipping value. Default is 0.
            model_checkpoint (float, optional): If checkpoints_directory is provided, this can be used to restart training from checkpoint.
            checkpoints_directory (str, optional): The directory to save model checkpoints. Default is None.
            checkpoints (int, optional): The interval for saving model checkpoints. Default is 10 epochs.
            models_to_keep (int, optional): The number of best models to keep. Default is 3.
            seed (int, optional): The random seed for numpy and torch. Default is None.
            model_id (str, optional): The ID for the model. Default is None.
            logname (str, optional): The logname for saving checkpoints. Default is None.
            logname_prefixe (str, optional): The prefix for the logname. Default is "score_model".

        Returns:
            list: List of loss values during training.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        ema = ExponentialMovingAverage(self.parameters(), decay=ema_decay)
        if dataset is None:
            dataset = Dataset(dataset_path, dataset_key, dataset_extension, device=self.device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        if n_iterations_in_epoch is None:
            n_iterations_in_epoch = len(dataloader)
        if checkpoints_directory is None:
            checkpoints_directory = self.checkpoints_directory
        
        if preprocessing_fn is not None:
            preprocessing_name = preprocessing_fn.__name__
        else:
            preprocessing_name = None
            preprocessing_fn = lambda x: x
        
        hyperparameters = self.model.hyperparameters
        if isinstance(self.sde, VPSDE):
            hyperparameters["sde"] = "vpsde"
            hyperparameters["beta_min"] = self.sde.beta_min
            hyperparameters["beta_max"] = self.sde.beta_max
        elif isinstance(self.sde, VESDE):
            hyperparameters["sde"] = "vesde"
            hyperparameters["sigma_min"] = self.sde.sigma_min
            hyperparameters["sigma_max"] = self.sde.sigma_max

       # ==== Take care of where to write checkpoints and stuff =================================================================
        if model_id is not None:
            logname = model_id
        elif logname is not None:
            logname = logname
        else:
            logname = logname_prefixe + "_" + datetime.now().strftime("%y%m%d%H%M%S")
        
        if checkpoints_directory is not None:
            if not os.path.isdir(checkpoints_directory):
                os.mkdir(checkpoints_directory)
                with open(os.path.join(checkpoints_directory, "script_params.json"), "w") as f:
                    json.dump(
                        {
                            "dataset_path": dataset_path,
                            "dataset_extension": dataset_extension,
                            "dataset_key": dataset_extension,
                            "preprocessing": preprocessing_name,
                            "learning_rate": learning_rate,
                            "ema_decay": ema_decay,
                            "batch_size": batch_size,
                            "shuffle": shuffle,
                            "epochs": epochs,
                            "patience": patience,
                            "tolerance": tolerance,
                            "max_time": max_time,
                            "epsilon": epsilon,
                            "warmup": warmup,
                            "clip": clip,
                            "checkpoint_directory": checkpoints_directory,
                            "checkpoints": checkpoints,
                            "models_to_keep": models_to_keep,
                            "seed": seed,
                            "model_id": model_id,
                            "logname": logname,
                            "logname_prefixe": logname_prefixe,
                        },
                        f,
                        indent=4
                    )
                with open(os.path.join(checkpoints_directory, "model_hparams.json"), "w") as f:
                    json.dump(hyperparameters, f, indent=4)
            save_checkpoint = True

            # ======= Load model if model_id is provided ===============================================================
            paths = glob.glob(os.path.join(checkpoints_directory, "checkpoint*.pt"))
            opt_paths = glob.glob(os.path.join(checkpoints_directory, "optimizer*.pt"))
            checkpoint_indices = [int(re.findall('[0-9]+', os.path.split(path)[-1])[-1]) for path in paths]
            scores = [float(re.findall('([0-9]{1}.[0-9]+e[+-][0-9]{2})', os.path.split(path)[-1])[-1]) for path in paths]
            if model_id is not None and checkpoint_indices:
                if model_checkpoint is not None:
                    checkpoint_path = paths[checkpoint_indices.index(model_checkpoint)]
                    self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.model.device))
                    optimizer.load_state_dict(torch.load(opt_paths[checkpoints == model_checkpoint], map_location=self.device))
                    print(f"Loaded checkpoint {model_checkpoint} of {model_id}")
                    latest_checkpoint = model_checkpoint
                else:
                    max_checkpoint_index = np.argmax(checkpoint_indices)
                    checkpoint_path = paths[max_checkpoint_index]
                    self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                    print(f"Loaded checkpoint {checkpoint_indices[max_checkpoint_index]} of {model_id}")
                    latest_checkpoint = checkpoint_indices[max_checkpoint_index]
            else:
                latest_checkpoint = 0
        else:
            save_checkpoint = False

        if seed is not None:
            torch.manual_seed(seed)

        def loss_fn(x):
            B, *D = x.shape
            broadcast = [B, *[1] * len(D)]
            z = torch.randn_like(x)
            t = torch.rand(B).to(self.device) * (1. - epsilon) + epsilon
            mean, sigma = self.sde.marginal_prob(t=t, x=x)
            sigma_ = sigma.view(*broadcast)
            return torch.sum((z + sigma_ * self(t=t, x=mean + sigma_ * z)) ** 2) / B

        best_loss = float('inf')
        losses = []
        step = 0
        global_start = time.time()
        estimated_time_for_epoch = 0
        out_of_time = False

        for epoch in (pbar := tqdm(range(epochs))):
            if (time.time() - global_start) > max_time * 3600 - estimated_time_for_epoch:
                break
            epoch_start = time.time()
            time_per_step_epoch_mean = 0
            cost = 0
            data_iter = iter(dataloader)
            for _ in range(n_iterations_in_epoch):
                start = time.time()
                try:
                    x = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    x = next(data_iter)
                if isinstance(x, tuple) or isinstance(x, list): # wrapper for tensor dataset
                    x = x[0]
                if preprocessing_fn is not None:
                    x = preprocessing_fn(x)
                optimizer.zero_grad()
                loss = loss_fn(x)
                loss.backward()

                if step < warmup:
                    for g in optimizer.param_groups:
                        g['lr'] = learning_rate * np.minimum(step / warmup, 1.0)

                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip)

                optimizer.step()
                ema.update()

                _time = time.time() - start
                time_per_step_epoch_mean += _time
                cost += float(loss)
                step += 1

            time_per_step_epoch_mean /= len(dataloader)
            cost /= len(dataloader)
            pbar.set_description(f"Epoch {epoch + 1:d} | Cost: {cost:.1e} |")
            losses.append(cost)
            if verbose >= 2:
                print(f"epoch {epoch} | cost {cost:.2e} | time per step {time_per_step_epoch_mean:.2e} s")
            elif verbose == 1:
                if (epoch + 1) % checkpoints == 0:
                    print(f"epoch {epoch} | cost {cost:.1e}")

            if np.isnan(cost):
                print("Model exploded and returns NaN")
                break

            if cost < (1 - tolerance) * best_loss:
                best_loss = cost
                patience = patience
            else:
                patience -= 1

            if (time.time() - global_start) > max_time * 3600:
                out_of_time = True

            if save_checkpoint:
                if epoch % checkpoints == 0 or patience == 0 or epoch == epochs - 1 or out_of_time:
                    latest_checkpoint += 1
                    with open(os.path.join(checkpoints_directory, "score_sheet.txt"), mode="a") as f:
                        f.write(f"{latest_checkpoint} {cost}\n")
                    with ema.average_parameters():
                        torch.save(self.state_dict(), os.path.join(checkpoints_directory, f"checkpoint_{cost:.4e}_{latest_checkpoint:03d}.pt"))
                    torch.save(optimizer.state_dict(), os.path.join(checkpoints_directory, f"optimizer_{cost:.4e}_{latest_checkpoint:03d}.pt"))
                    paths = glob.glob(os.path.join(checkpoints_directory, "*.pt"))
                    checkpoint_indices = [int(re.findall('[0-9]+', os.path.split(path)[-1])[-1]) for path in paths]
                    scores = [float(re.findall('([0-9]{1}.[0-9]+e[+-][0-9]{2})', os.path.split(path)[-1])[-1]) for path in paths]

                    if len(checkpoint_indices) >= models_to_keep + 1:
                        index_to_delete = np.argmax(scores[:-1])
                        os.remove(os.path.join(checkpoints_directory, f"checkpoint_{scores[index_to_delete]:.4e}_{checkpoint_indices[index_to_delete]:03d}.pt"))
                        os.remove(os.path.join(checkpoints_directory, f"optimizer_{scores[index_to_delete]:.4e}_{checkpoint_indices[index_to_delete]:03d}.pt"))
                        del scores[index_to_delete]
                        del checkpoint_indices[index_to_delete]

            if patience == 0:
                print("Reached patience")
                break

            if out_of_time:
                print("Out of time")
                break

            if epoch > 0:
                estimated_time_for_epoch = time.time() - epoch_start

        print(f"Finished training after {(time.time() - global_start) / 3600:.3f} hours.")
        # Save EMA weights in the model
        ema.copy_to(self.parameters())
        return losses
