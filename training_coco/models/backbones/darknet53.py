"""
models/backbones/darknet53.py — Darknet53 backbone for YOLOv3.

Reference: Redmon & Farhadi, "YOLOv3: An Incremental Improvement", 2018.

Structure (for 300×300 input):
    Conv               →  150×150 ×  32
    Residual ×  1      →  150×150 ×  64   (after stride-2 conv)
    Residual ×  2      →   75×75  × 128
    Residual ×  8      →   38×38  × 256   ← D3
    Residual × 16      →   19×19  × 512   ← D4
    Residual × 32      →   10×10  × 1024  ← D5

D3, D4, D5 are fed to the three YOLOv3 detection heads.
"""

import tensorflow as tf
from tensorflow.keras import layers


# ─────────────────────────── Building Blocks ─────────────────────────────────

def _conv_bn_leaky(x: tf.Tensor,
                   filters: int,
                   kernel_size: int = 3,
                   stride: int = 1,
                   name: str = '') -> tf.Tensor:
    """Conv2D → BN → LeakyReLU(0.1).  Darknet53 uses LeakyReLU throughout."""
    padding = 'same' if stride == 1 else 'valid'
    if stride > 1:
        # Darknet pads before strided conv (top-left vs symmetric)
        x = layers.ZeroPadding2D(((1, 0), (1, 0)), name=f'{name}_pad')(x)
    x = layers.Conv2D(
        filters, kernel_size, strides=stride, padding=padding,
        use_bias=False, name=f'{name}_conv',
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.LeakyReLU(0.1, name=f'{name}_lrelu')(x)
    return x


def _residual_block(x: tf.Tensor,
                    filters_half: int,
                    name: str = '') -> tf.Tensor:
    """
    Darknet53 residual unit:
        1×1 Conv (filters_half) → 3×3 Conv (2×filters_half) → add shortcut
    """
    h = _conv_bn_leaky(x, filters_half,     kernel_size=1, name=f'{name}_r1')
    h = _conv_bn_leaky(h, filters_half * 2, kernel_size=3, name=f'{name}_r2')
    return layers.Add(name=f'{name}_add')([x, h])


def _res_stack(x: tf.Tensor,
               out_channels: int,
               num_blocks: int,
               name: str = '') -> tf.Tensor:
    """Stride-2 conv to double channels, then num_blocks residual units."""
    x = _conv_bn_leaky(x, out_channels, kernel_size=3, stride=2, name=f'{name}_down')
    for i in range(num_blocks):
        x = _residual_block(x, out_channels // 2, name=f'{name}_res{i}')
    return x


# ─────────────────────────── Public Builder ──────────────────────────────────

def build_darknet53_features(inp: tf.Tensor):
    """
    Build Darknet53 body.

    Args:
        inp : Keras tensor  (B, 300, 300, 3)

    Returns:
        (D3, D4, D5) — three feature maps at strides 8, 16, 32
    """
    # Stem: 300×300 → 300×300 × 32
    x = _conv_bn_leaky(inp, 32, kernel_size=3, name='dk_stem')

    # Stage 1: 300×300 → 150×150 × 64   (1 residual)
    x = _res_stack(x, 64,   num_blocks=1,  name='dk_s1')

    # Stage 2: 150×150 → 75×75 × 128    (2 residuals)
    x = _res_stack(x, 128,  num_blocks=2,  name='dk_s2')

    # Stage 3: 75×75 → 38×38 × 256      (8 residuals)  ← D3
    x  = _res_stack(x, 256,  num_blocks=8,  name='dk_s3')
    D3 = x

    # Stage 4: 38×38 → 19×19 × 512      (16 residuals) ← D4
    x  = _res_stack(x, 512,  num_blocks=16, name='dk_s4')
    D4 = x

    # Stage 5: 19×19 → 10×10 × 1024     (32 residuals) ← D5
    x  = _res_stack(x, 1024, num_blocks=32, name='dk_s5')
    D5 = x

    return D3, D4, D5
