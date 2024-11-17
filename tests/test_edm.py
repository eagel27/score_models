from score_models import ScoreModel, EDMScoreModel, MLP, NCSNpp, VPSDE, EDMv2Net
import torch
import pytest


def test_init(tmp_path):
    # Some basic tests
    D = 2
    B = 10
    sde = VPSDE()
    net = MLP(D)
    model = ScoreModel(net, sde, formulation="edm")
    print(model)
    
    x = torch.randn(B, D)
    t = torch.rand(B)
    out = model(t, x)
    assert tuple(out.shape) == (B, D)
    
    # Make sure that when we reload the model, it is an EDM model
    model.save(tmp_path)
    new_model = ScoreModel(path=tmp_path)
    assert isinstance(new_model, EDMScoreModel)
    
    
    

