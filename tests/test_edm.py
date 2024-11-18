from score_models import ScoreModel, EDMScoreModel, MLP, NCSNpp, VPSDE, EDMv2Net
import torch
import pytest


def test_edm_init(tmp_path):
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


# TODO improve MLP with EDM framework as well, and test the new net
def test_edm_net(tmp_path):
    P = 16      # pixels
    D = [P]*2   # dimensions
    C = 3       # channels
    B = 10      # batch size
    
    # Parameters
    nf = 16
    ch_mult = [1, 2]
    
    sde = VPSDE()
    net = EDMv2Net(P, C, nf=nf, ch_mult=ch_mult)
    model = ScoreModel(net, sde, formulation="edm")
    
    x = torch.randn(B, C, *D)
    t = torch.rand(B)
    out = model(t, x)
    assert tuple(out.shape) == (B, C, *D)
    
    # Make sure that the model is reloaded correctly
    model.save(tmp_path)
    
    new_model = ScoreModel(path=tmp_path)
    out2 = new_model(t, x)
    assert torch.allclose(out, out2)
