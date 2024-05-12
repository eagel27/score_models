import jax.numpy as jnp
from jax import lax
from .upfirdn1d import upfirdn1d

__all__ = [
    "naive_upsample_1d",
    "naive_downsample_1d",
    "upsample_1d",
    "downsample_1d",
    "conv_downsample_1d",
    "upsample_conv_1d",
]


def naive_upsample_1d(x, factor=2):
    C, H = x.shape
    x = jnp.reshape(x, (C, H, 1))
    x = jnp.tile(x, (1, 1, factor))
    return jnp.reshape(x, (C, H * factor))


def naive_downsample_1d(x, factor=2):
    C, H = x.shape
    x = jnp.reshape(x, (C, H // factor, factor))
    return jnp.mean(x, axis=2)


def upsample_conv_1d(x, w, k=None, factor=2, gain=1):
    """Fused `upsample_1d()` followed by `tf.nn.conv1d()`.
    Padding is performed only once at the beginning, not between the
    operations.
    The fused op is considerably more efficient than performing the same
    calculation using standard TensorFlow ops. It supports gradients of arbitrary order.
    Args:
      x: Input tensor of the shape `[C, D]`.
      w: Weight tensor of the shape `[filterD, inChannels, outChannels]`. 
        Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
      k: FIR filter of the shape `[firN]`
        (separable). The default is `[1] * factor`, which corresponds to
        nearest-neighbor upsampling.
      factor:       Integer upsampling factor (default: 2).
      gain:         Scaling factor for signal magnitude (default: 1.0).
    Returns:
      Tensor of the shape `[C, D * factor]` and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1
    outC, inC, convH = w.shape
    C, H = x.shape

    # Setup filter kernel.
    if k is None:
        k = jnp.array([1] * factor)
    k = _setup_kernel(k) * (gain * factor)
    p = (k.shape[0] - factor) - (convH - 1)

    # Determine data dimensions.
    stride = factor
    output_shape = ((H - 1) * factor + convH,)
    output_padding = ((output_shape[0] - (x.shape[1] - 1) * stride - convH),)
    num_groups = C // inC
    
    # Transpose weights.
    w = jnp.reshape(w, (num_groups, -1, inC, convH))
    w = jnp.flip(w, axis=(3,)).transpose((0, 2, 1, 3))
    w = jnp.reshape(w, (num_groups * inC, -1, convH))
    
    x = lax.conv_transpose( # Default is channel_last for this op, so we explicitly set it to channel_first
        x.reshape(1, *x.shape),
        w,
        strides=(stride,),
        padding="VALID",
        dimension_numbers=("NCH", "IOH", "NCH"),
    )
    x = x.reshape(x.shape[1:]) # Remove singleton dimension
    x = jnp.pad(x, pad_width=((0, 0), (output_padding[0], output_padding[0])))
    return upfirdn1d(x, jnp.array(k), pad=((p + 1) // 2 + factor - 1, p // 2 + 1))


def conv_downsample_1d(x, w, k=None, factor=2, gain=1):
    """Fused `tf.nn.conv1d()` followed by `downsample_1d()`.
    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same
    calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.
    Args:
        x:            Input tensor of the shape `[C, D]`
        w:            Weight tensor of the shape `[filterD, inChannels, outChannels]`.
                  Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
        k:            FIR filter of the shape `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).
    Returns:
        Tensor of the shape `[N, C, D // factor]` and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1
    _outC, _inC, convH = w.shape
    if k is None:
        k = jnp.array([1] * factor)
    k = _setup_kernel(k) * gain
    p = (k.shape[0] - factor) + (convH - 1)
    s = factor
    x = upfirdn1d(x, jnp.array(k), pad=((p + 1) // 2, p // 2))
    x = x.reshape(1, *x.shape)
    x = lax.conv(x, w, window_strides=(s,), padding="VALID") # Note: default is channel first for this op
    return x.reshape(x.shape[1:])


def _setup_kernel(k):
    k = jnp.asarray(k, dtype=jnp.float32)
    k /= jnp.sum(k)
    assert k.ndim == 1
    return k


def upsample_1d(x, k=None, factor=2, gain=1):
    r"""Upsample a batch of 1D time series with the given filter.
    Accepts a batch of 1D time series of the shape `[N, C, D]`
    and upsamples each time series with the given filter. The filter is normalized so
    that if the input pixels are constant, they will be scaled by the specified
    `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded
    with zeros so that its shape is a multiple of the upsampling factor.
    Args:
        x:            Input tensor of the shape `[N, C, D]`.
        k:            FIR filter of the shape `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          nearest-neighbor upsampling.
        factor:       Integer upsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).
    Returns:
        Tensor of the shape `[N, C, D * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = jnp.array([1] * factor)
    k = _setup_kernel(k) * (gain * factor)
    p = k.shape[0] - factor
    return upfirdn1d(
        x,
        jnp.array(k),
        up=factor,
        pad=((p + 1) // 2 + factor - 1, p // 2),
    )


def downsample_1d(x, k=None, factor=2, gain=1):
    r"""Downsample a batch of 1D time series with the given filter.
    Accepts a batch of 2D images of the shape `[N, C, D]`
    and downsamples each time series with the given filter. The filter is normalized
    so that if the input pixels are constant, they will be scaled by the specified
    `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded
    with zeros so that its shape is a multiple of the downsampling factor.
    Args:
        x:            Input tensor of the shape `[N, C, D]`.
        k:            FIR filter of the shape `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).
    Returns:
        Tensor of the shape `[N, C, D // factor]`
    """

    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = jnp.array([1] * factor)
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    return upfirdn1d(
        x, jnp.array(k), down=factor, pad=((p + 1) // 2, p // 2)
    )
