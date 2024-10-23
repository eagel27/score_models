from typing import Union, Optional

from torch.nn import Module
import torch.nn.functional as F
from .conv_layers import conv3x3
from .style_gan_conv import StyleGANConv
from .up_or_downsampling import upsample
from ..definitions import default_init

__all__ = ['UpsampleLayer']


class UpsampleLayer(Module):
    def __init__(
            self, 
            in_ch: Optional[int] = None,
            out_ch: Optional[int] = None,
            with_conv: bool = False,
            fir: bool = False,
            fir_kernel: tuple = (1, 3, 3, 1),
            factor: Union[int, tuple] = 2,
            dimensions: int = 2,
            ):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if isinstance(factor, int):
            factor = [factor]*dimensions
        if len(factor) != dimensions:
            raise ValueError(f'Factor must have {dimensions} elements.')
        self.factor = factor
        if out_ch != in_ch:
            assert with_conv
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch, dimensions=dimensions)
        else:
            if with_conv:
                self.Conv1d_0 = StyleGANConv(in_ch, out_ch,
                                   kernel=3, up=True,
                                   resample_kernel=fir_kernel,
                                   use_bias=True,
                                   kernel_init=default_init(),
                                   dimensions=dimensions)
        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch

    def forward(self, x):
        B, C, *D = x.shape
        if not self.fir:
            h = F.interpolate(x, size=[d*2 for d in D], mode='nearest')
            if self.with_conv:
                h = self.Conv_0(h)
        else:
            if not self.with_conv:
                h = upsample(x, self.fir_kernel, factor=self.factor, dimensions=len(D))
            else:
                h = self.Conv1d_0(x)
        return h

