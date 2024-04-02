from typing import Callable
import jax.numpy as jnp
from tqdm import tqdm

def euler(x, drift_fn: Callable, N: int, t0=0., t1=1., verbose=0, **kwargs):
    """
    A basic implementation of Euler discretisation method of the ODE associated 
    with the marginales of the learned SDE in JAX.
    
    x: Initial state
    drift_fn: Update the state x
    N: Number of steps
    t0: Initial time of integration, defaults to 0.
    t1: Final time of integration, defaults to 1.
    
    Returns the final state
    """
    disable = False if verbose else True  
    B, *D = x.shape
    t = jnp.ones([B]) * t0
    dt = (t1 - t0) / N
    for _ in tqdm(range(N), disable=disable):
        x = x + drift_fn(t, x, **kwargs) * dt
        t += dt
    return x

