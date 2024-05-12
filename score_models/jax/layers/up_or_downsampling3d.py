import jax.numpy as jnp
from jax import lax
from .upfirdn3d import upfirdn3d

__all__ = [
    "naive_upsample_3d",
    "naive_downsample_3d",
    "upsample_3d",
    "downsample_3d",
    "conv_downsample_3d",
    "upsample_conv_3d",
]


def naive_upsample_3d(x, factor=2):
    C, H, W, D = x.shape
    x = jnp.reshape(x, (C, H, 1, W, 1, D, 1))
    x = jnp.tile(x, (1, 1, factor, 1, factor, 1, factor))
    return jnp.reshape(x, (C, H * factor, W * factor, D * factor))


def naive_downsample_3d(x, factor=2):
    C, H, W, D = x.shape
    x = jnp.reshape(
        x, (C, H // factor, factor, W // factor, factor, D // factor, factor)
    )
    return jnp.mean(x, axis=(2, 4, 6))


def upsample_conv_3d(x, w, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    outC, inC, convH, convW, convD = w.shape
    C, H, W, D = x.shape
    assert convW == convH
    assert convW == convD

    if k is None:
        k = jnp.array([1] * factor)
    k = _setup_kernel(k) * (gain * (factor**3))
    p = (k.shape[0] - factor) - (convW - 1)

    stride = (factor, factor, factor)
    output_shape = (
        (H - 1) * factor + convH,
        (W - 1) * factor + convW,
        (D - 1) * factor + convD,
    )
    output_padding = (
        (output_shape[0] - (H - 1) * stride[0] - convH),
        (output_shape[1] - (W - 1) * stride[1] - convW),
        (output_shape[2] - (D - 1) * stride[2] - convD),
    )
    assert output_padding[0] >= 0 and output_padding[1] >= 0 and output_padding[2] >= 0
    num_groups = C // inC

    w = jnp.reshape(w, (num_groups, -1, inC, convH, convW, convD))
    w = jnp.flip(w, axis=(3, 4, 5)).transpose((0, 2, 1, 3, 4, 5))
    w = jnp.reshape(w, (num_groups * inC, -1, convH, convW, convD))
    
    x = lax.conv_transpose(
        x.reshape(1, *x.shape),
        w,
        strides=stride,
        padding="VALID",
        dimension_numbers=("NCHWD", "IOHWD", "NCHWD"),
    )
    x = x.reshape(*x.shape[1:]) # Remove singleton dimension
    x = jnp.pad(
        x,
        (
            (0, 0),
            (output_padding[0], output_padding[0]),
            (output_padding[1], output_padding[1]),
            (output_padding[2], output_padding[2]),
        ),
    )
    return upfirdn3d(x, jnp.array(k), pad=((p + 1) // 2 + factor - 1, p // 2 + 1))


def conv_downsample_3d(x, w, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    _outC, _inC, convH, convW, convD = w.shape
    assert convW == convH
    assert convW == convD

    if k is None:
        k = jnp.array([1] * factor)
    k = _setup_kernel(k) * gain
    p = (k.shape[0] - factor) + (convW - 1)
    s = (factor, factor, factor)
    
    x = upfirdn3d(x, jnp.array(k), pad=((p + 1) // 2, p // 2))
    x = x.reshape(1, *x.shape)
    x = lax.conv(x, w, window_strides=s, padding="VALID")
    return x.reshape(x.shape[1:])


def _setup_kernel(k):
    k = jnp.asarray(k, dtype=jnp.float32)
    if k.ndim == 1:
        C = k.shape[0]
        m = jnp.outer(k, k)
        k = jnp.outer(m, k).reshape(C, C, C) # Equivalent to np.multiply.outer(m, k)
    k /= jnp.sum(k)
    assert k.ndim == 3
    assert k.shape[0] == k.shape[1]
    assert k.shape[0] == k.shape[2]
    return k


def upsample_3d(x, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = jnp.array([1] * factor)
    k = _setup_kernel(k) * (gain * (factor**3))
    p = k.shape[0] - factor
    return upfirdn3d(
        x,
        jnp.array(k),
        up=factor,
        pad=((p + 1) // 2 + factor - 1, p // 2),
    )


def downsample_3d(x, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = jnp.array([1] * factor)
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    return upfirdn3d(
        x, jnp.array(k), down=factor, pad=((p + 1) // 2, p // 2)
    )

