import torch
from torch.utils.data import TensorDataset
from score_models import ScoreModel, EnergyModel, MLP, NCSNpp
import shutil, os
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, size, channels, dimensions:list, conditioning="None", test_input_list=False):
        self.size = size
        self.channels = channels
        self.dimensions = dimensions
        self.conditioning = conditioning
        self.test_input_list = test_input_list

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.test_input_list:
            return torch.randn(self.channels, *self.dimensions),
        if self.conditioning.lower() == "none":
            return torch.randn(self.channels, *self.dimensions)
        elif self.conditioning.lower() == "time":
            return torch.randn(self.channels, *self.dimensions), torch.randn(1)
        elif self.conditioning.lower() == "input":
            return torch.randn(self.channels, *self.dimensions), torch.randn(self.channels, *self.dimensions)
        elif self.conditioning.lower() == "input_and_time":
            return torch.randn(self.channels, *self.dimensions), torch.randn(self.channels, *self.dimensions), torch.randn(1)
        elif self.conditioning.lower() == "time_and_discrete":
            return torch.randn(self.channels, *self.dimensions), torch.randn(1), torch.randint(10, (1,))
        elif self.conditioning.lower() == "discrete_time":
            return torch.randn(self.channels, *self.dimensions), torch.tensor(np.random.choice(range(10)))
    
def test_multiple_channels_ncsnpp():
    C = 3
    D = 16
    dim = 2
    B = 5
    size = 2*B
    dataset = Dataset(size, C, [D]*dim)
    net = NCSNpp(nf=8, channels=C, ch_mul=(1, 1))
    model = ScoreModel(model=net, sigma_min=1e-2, sigma_max=10)
    model.fit(dataset, batch_size=B, epochs=2)


def test_training_conditioned_input_ncsnpp():
    C = 1
    D = 16
    dim = 2
    B = 5
    size = 2*B
    dataset = Dataset(size, C, [D]*dim, conditioning="input")
    net = NCSNpp(nf=8, ch_mul=(1, 1), condition=["input"], condition_input_channels=1)
    model = ScoreModel(model=net, sigma_min=1e-2, sigma_max=10)
    model.fit(dataset, batch_size=B, epochs=2)
    

def test_training_conditioned_continuous_timelike_ncsnpp():
    C = 1
    D = 16
    dim = 2
    B = 5
    size = 2*B
    dataset = Dataset(size, C, [D]*dim, conditioning="time")
    net = NCSNpp(nf=8, ch_mul=(1, 1), condition=["continuous_timelike"])
    model = ScoreModel(model=net, sigma_min=1e-2, sigma_max=10)
    model.fit(dataset, batch_size=B, epochs=2)

def test_training_conditioned_discrete_timelike_ncsnpp():
    C = 1
    D = 16
    dim = 2
    B = 5
    size = 2*B
    dataset = Dataset(size, C, [D]*dim, conditioning="discrete_time")
    net = NCSNpp(nf=8, ch_mul=(1, 1), condition=["discrete_timelike"], condition_num_embedding=(10,))
    model = ScoreModel(model=net, sigma_min=1e-2, sigma_max=10)
    model.fit(dataset, batch_size=B, epochs=2)


def test_training_conditioned_discrete_and_timelike_ncsnpp():
    C = 1
    D = 16
    dim = 2
    B = 5
    size = 2*B
    dataset = Dataset(size, C, [D]*dim, conditioning="time_and_discrete")
    net = NCSNpp(nf=8, ch_mul=(1, 1), condition=["continuous_timelike", "discrete_timelike"], condition_num_embedding=(10,))
    model = ScoreModel(model=net, sigma_min=1e-2, sigma_max=10)
    model.fit(dataset, batch_size=B, epochs=2)
        

def test_training_score_mlp(tmp_path):
    C = 10
    B = 5
    size = 2*B
    dataset = Dataset(size, C, [])
    hyperparameters = {
        "dimensions": C,
        "units": 2*C,
        "layers": 2,
        "time_embedding_dimensions": 32,
        "embedding_scale": 32,
        "activation": "swish",
        "time_branch_layers": 1
    }
    net = MLP(**hyperparameters)
    # Create an instance of ScoreModel
    model = ScoreModel(model=net, sigma_min=1e-2, sigma_max=10)

    # Define any preprocessing function if needed
    def preprocessing_fn(x):
        return x

    # Set the hyperparameters and other options for training
    learning_rate = 1e-3
    ema_decay = 0.9999
    batch_size = 1
    epochs = 10
    warmup = 0 # learning rate warmup
    clip = 0. # gradient clipping
    path = tmp_path / "test"
    seed = 42

    # Fit the model to the dataset
    losses = model.fit(
        dataset, 
        preprocessing_fn=preprocessing_fn, 
        learning_rate=learning_rate, 
        ema_decay=ema_decay,
        batch_size=batch_size, 
        epochs=epochs, 
        warmup=warmup, 
        clip=clip, 
        checkpoints_directory=path, # For backward compatibility
        seed=seed,
        models_to_keep=10,
        checkpoint_every=1
        )
    print(losses)
    assert len(losses) == epochs, f"Expected {epochs} losses, got {len(losses)}"
    assert os.path.isfile(os.path.join(path, "model_hparams.json")), "model_hparams.json not found"
    assert os.path.isfile(os.path.join(path, "script_params.json")), "script_params.json not found"
    for i in range(1, 11):
        assert os.path.isfile(os.path.join(path, f"checkpoint_{i:03}.pt")), f"checkpoint_{i:03}.pt not found"
        assert os.path.isfile(os.path.join(path, f"optimizer_{i:03}.pt")), f"optimizer_{i:03}.pt not found"


def test_training_score_mlp_input_list(tmp_path):
    C = 10
    B = 5
    size = 2*B
    hyperparameters = {
        "dimensions": C,
        "units": 2*C,
        "layers": 2,
        "time_embedding_dimensions": 32,
        "embedding_scale": 32,
        "activation": "swish",
        "time_branch_layers": 1
    }
    net = MLP(**hyperparameters)
    # Create an instance of ScoreModel
    model = ScoreModel(model=net, sigma_min=1e-2, sigma_max=10)
    dataset = Dataset(size, C, [], test_input_list=True)
    path = tmp_path / "test"
    losses = model.fit(
        dataset, 
        path=path, 
        epochs=10,
        checkpoints=1,
        models_to_keep=12,
        batch_size=1
        )


def test_load_checkpoint_at_scoremodel_init(tmp_path):
    C = 10
    B = 5
    size = 2*B
    hyperparameters = {
        "dimensions": C,
        "units": 2*C,
        "layers": 2,
        "time_embedding_dimensions": 32,
        "embedding_scale": 32,
        "activation": "swish",
        "time_branch_layers": 1
    }
    net = MLP(**hyperparameters)
    model = ScoreModel(model=net, sigma_min=1e-2, sigma_max=10)
    
    # Save a checkpoint
    path = tmp_path / "test"
    model.save(path)
    
    print(os.listdir(path))
    
    # Reload model
    model1 = ScoreModel(path=path, model_checkpoint=1)
    assert model1.loaded_checkpoint == 1, f"Expected checkpoint 1, got {model1.loaded_checkpoint}"
    
    # Save some additional models with fit
    model.fit(
        Dataset(size, C, []),
        path=path,
        epochs=10,
        checkpoint_every=1,
        models_to_keep=12,
        )

    model2 = ScoreModel(path=path, checkpoint=4)
    assert model2.loaded_checkpoint == 4, f"Expected checkpoint 4, got {model2.loaded_checkpoint}"

    model3 = ScoreModel(path=path)
    expected_checkpoint = 11  # Based on previous test, training 10 epochs and saving each one, we should have 11 checkpoints (also saving the last one)
    assert model3.loaded_checkpoint == expected_checkpoint, f"Expected checkpoint {expected_checkpoint}, got {model3.loaded_checkpoint}"


def test_training_score_ncsnpp():
    C = 1
    D = 140
    B = 5
    size = 2*B
    dataset = Dataset(size, C, [D])
    hyperparameters = {
     'channels': C,
     'nf': 8,
     'activation_type': 'swish',
     'ch_mult': (2, 2),
     'num_res_blocks': 2,
     'resample_with_conv': True,
     'dropout': 0.0,
     'fir': True,
     'fir_kernel': (1, 3, 3, 1),
     'skip_rescale': True,
     'progressive': 'output_skip',
     'progressive_input': 'input_skip',
     'init_scale': 0.01,
     'fourier_scale': 16.0,
     'resblock_type': 'biggan',
     'combine_method': 'sum',
     'attention': True,
     'dimensions': 1,
     'sde': 'vesde',
     'sigma_min': 0.001,
     'sigma_max': 200,
     'T': 1.0}
    net = NCSNpp(**hyperparameters)
    # Create an instance of ScoreModel
    model = ScoreModel(model=net, sigma_min=1e-2, sigma_max=10)

    # Define any preprocessing function if needed
    def preprocessing_fn(x):
        return x

    # Set the hyperparameters and other options for training
    learning_rate = 1e-3
    ema_decay = 0.9999
    batch_size = 1
    epochs = 2
    warmup = 0 # learning rate warmup
    clip = 0. # gradient clipping
    seed = 42

    # Fit the model to the dataset
    losses = model.fit(
        dataset, 
        preprocessing_fn=preprocessing_fn, 
        learning_rate=learning_rate, 
        ema_decay=ema_decay,
        batch_size=batch_size, 
        epochs=epochs, 
        warmup=warmup, 
        clip=clip, 
        seed=seed
        )
    print(losses)

def test_training_energy():
    # Create a dummy dataset
    X = torch.randn(10, 10)

    # Convert the data into a TensorDataset
    dataset = TensorDataset(X)

    hyperparameters = {
        "dimensions": 10,
        "units": 10,
        "layers": 2,
        "time_embedding_dimensions": 32,
        "embedding_scale": 32,
        "activation": "swish",
        "time_branch_layers": 1,
        # "nn_is_energy": True
    }
    net = MLP(**hyperparameters)
    # Create an instance of ScoreModel
    model = EnergyModel(model=net, sigma_min=1e-2, sigma_max=10)

    # Define any preprocessing function if needed
    def preprocessing_fn(x):
        return x

    # Set the hyperparameters and other options for training
    learning_rate = 1e-3
    ema_decay = 0.9999
    batch_size = 1
    epochs = 10
    warmup = 0 # learning rate warmup
    clip = 0. # gradient clipping
    seed = 42

    # Fit the model to the dataset
    losses = model.fit(
        dataset, 
        preprocessing_fn=preprocessing_fn, 
        learning_rate=learning_rate, 
        ema_decay=ema_decay,
        batch_size=batch_size, 
        epochs=epochs, 
        warmup=warmup, 
        clip=clip, 
        seed=seed
        )
    print(losses)
