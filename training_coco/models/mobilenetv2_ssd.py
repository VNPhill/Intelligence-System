"""
models/mobilenetv2_ssd.py — MobileNetV2-SSD.

Backbone:  MobileNetV2 (inverted residual bottlenecks, linear bottleneck,
           ReLU6 activations).
Reference: Sandler et al., "MobileNetV2: Inverted Residuals and Linear
           Bottlenecks", CVPR 2018.

MobileNetV2 configuration used here (standard paper spec):
    Input 300×300
    conv2d     s=2  → 150×150 × 32
    bottleneck t=1, c=16,  n=1, s=1 → 150×150 × 16
    bottleneck t=6, c=24,  n=2, s=2 →  75×75  × 24
    bottleneck t=6, c=32,  n=3, s=2 →  38×38  × 32   ← feat_38  (after expand)
    bottleneck t=6, c=64,  n=4, s=2 →  19×19  × 64
    bottleneck t=6, c=96,  n=3, s=1 →  19×19  × 96   ← feat_19
    bottleneck t=6, c=160, n=3, s=2 →  10×10  × 160
    bottleneck t=6, c=320, n=1, s=1 →  10×10  × 320  ← feat_10
    Extra SSD layers → 5×5 · 3×3 · 1×1

Note on feat_38: MobileNetV2 has only 32 channels at 38×38 in the standard
spec.  A 1×1 pointwise expansion to 96 channels is added before tapping this
feature map to give the detection head sufficient capacity.
"""

import tensorflow as tf
from tensorflow.keras import layers
from config import NUM_CLASSES_WITH_BG
from .ssd_common import build_extra_layers, assemble_ssd


# ─────────────────────────── Building Blocks ─────────────────────────────────

def _inverted_residual(x: tf.Tensor,
                       expand_ratio: int,
                       out_channels: int,
                       stride: int,
                       name: str) -> tf.Tensor:
    """
    MobileNetV2 inverted residual (bottleneck) block.

        Expand  1×1  → BN → ReLU6        (skip if expand_ratio == 1)
        DW      3×3  → BN → ReLU6
        Project 1×1  → BN               (no activation = linear bottleneck)
        + residual addition if stride == 1 and channels match
    """
    in_channels  = x.shape[-1]
    mid_channels = in_channels * expand_ratio

    # ── Expansion ─────────────────────────────────────────────────────────
    if expand_ratio != 1:
        h = layers.Conv2D(
            mid_channels, 1, use_bias=False, name=f'{name}_expand'
        )(x)
        h = layers.BatchNormalization(name=f'{name}_expand_bn')(h)
        h = layers.ReLU(6.0, name=f'{name}_expand_relu')(h)
    else:
        h = x

    # ── Depthwise ─────────────────────────────────────────────────────────
    h = layers.DepthwiseConv2D(
        3, strides=stride, padding='same',
        use_bias=False, name=f'{name}_dw',
    )(h)
    h = layers.BatchNormalization(name=f'{name}_dw_bn')(h)
    h = layers.ReLU(6.0, name=f'{name}_dw_relu')(h)

    # ── Linear projection ─────────────────────────────────────────────────
    h = layers.Conv2D(
        out_channels, 1, use_bias=False, name=f'{name}_proj'
    )(h)
    h = layers.BatchNormalization(name=f'{name}_proj_bn')(h)

    # ── Residual ──────────────────────────────────────────────────────────
    if stride == 1 and in_channels == out_channels:
        h = layers.Add(name=f'{name}_add')([x, h])

    return h


def _bottleneck_stack(x: tf.Tensor,
                      expand_ratio: int,
                      out_channels: int,
                      num_blocks: int,
                      stride: int,
                      name: str) -> tf.Tensor:
    """Stack of inverted residual blocks (first block may have stride > 1)."""
    x = _inverted_residual(x, expand_ratio, out_channels,
                           stride=stride, name=f'{name}_b0')
    for i in range(1, num_blocks):
        x = _inverted_residual(x, expand_ratio, out_channels,
                               stride=1, name=f'{name}_b{i}')
    return x


# ─────────────────────────── Model Builder ───────────────────────────────────

def build_mobilenetv2_ssd(num_classes: int = NUM_CLASSES_WITH_BG,
                          width: float = 1.0) -> tf.keras.Model:
    """
    Build MobileNetV2-SSD.

    Args:
        num_classes: total classes including background  (default 81)
        width:       channel-width multiplier applied to all channel counts

    Returns:
        tf.keras.Model  →  cls_out [B, 8732, C],  loc_out [B, 8732, 4]
    """
    def f(n): return max(1, int(n * width))

    inp = layers.Input(shape=(300, 300, 3), name='input_image')

    # ── Stem: 300×300 → 150×150 ─────────────────────────────────────────────
    x = layers.Conv2D(
        f(32), 3, strides=2, padding='same',
        use_bias=False, name='stem_conv',
    )(inp)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.ReLU(6.0, name='stem_relu')(x)

    # ── MobileNetV2 Backbone ─────────────────────────────────────────────────
    # t=1, c=16, n=1, s=1  → 150×150 × 16
    x = _bottleneck_stack(x, expand_ratio=1,  out_channels=f(16),
                          num_blocks=1, stride=1, name='block1')

    # t=6, c=24, n=2, s=2  →  75×75  × 24
    x = _bottleneck_stack(x, expand_ratio=6,  out_channels=f(24),
                          num_blocks=2, stride=2, name='block2')

    # t=6, c=32, n=3, s=2  →  38×38  × 32
    x = _bottleneck_stack(x, expand_ratio=6,  out_channels=f(32),
                          num_blocks=3, stride=2, name='block3')

    # Expand 38×38 channels to 96 so the detection head has enough capacity
    x = layers.Conv2D(
        f(96), 1, use_bias=False, name='feat38_expand'
    )(x)
    x = layers.BatchNormalization(name='feat38_expand_bn')(x)
    x = layers.ReLU(6.0, name='feat38_expand_relu')(x)
    feat_38 = x                                         # 38×38 × 96·w

    # t=6, c=64, n=4, s=2  →  19×19  × 64
    x = _bottleneck_stack(x, expand_ratio=6,  out_channels=f(64),
                          num_blocks=4, stride=2, name='block4')

    # t=6, c=96, n=3, s=1  →  19×19  × 96
    x = _bottleneck_stack(x, expand_ratio=6,  out_channels=f(96),
                          num_blocks=3, stride=1, name='block5')
    feat_19 = x                                         # 19×19 × 96·w

    # t=6, c=160, n=3, s=2 →  10×10  × 160
    x = _bottleneck_stack(x, expand_ratio=6,  out_channels=f(160),
                          num_blocks=3, stride=2, name='block6')

    # t=6, c=320, n=1, s=1 →  10×10  × 320
    x = _bottleneck_stack(x, expand_ratio=6,  out_channels=f(320),
                          num_blocks=1, stride=1, name='block7')
    feat_10 = x                                         # 10×10 × 320·w

    # ── Extra SSD Layers ─────────────────────────────────────────────────────
    feat_5, feat_3, feat_1 = build_extra_layers(feat_10, width=width)

    return assemble_ssd(
        inp,
        [feat_38, feat_19, feat_10, feat_5, feat_3, feat_1],
        num_classes=num_classes,
        model_name='MobileNetV2_SSD',
    )
