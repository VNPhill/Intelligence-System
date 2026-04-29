"""
models/mobilenet_ssd.py — MobileNetV1-SSD.

Backbone:  MobileNetV1 (depthwise-separable convolutions, ReLU6)
Reference: Howard et al., "MobileNets: Efficient CNNs for Mobile Vision
           Applications", arXiv 2017.

Feature maps tapped at:  38×38 (256ch) · 19×19 (512ch) · 10×10 (1024ch)
Extra SSD layers produce:  5×5 · 3×3 · 1×1
"""

import tensorflow as tf
from tensorflow.keras import layers
from config import NUM_CLASSES_WITH_BG
from .ssd_common import conv_bn_relu, build_extra_layers, assemble_ssd


# ─────────────────────────── Building Blocks ─────────────────────────────────

def _conv_bn_relu6(x, filters: int,
                   kernel_size: int = 3,
                   stride: int = 1,
                   padding: str = 'same',
                   name: str = '') -> tf.Tensor:
    """Conv2D → BN → ReLU6  (standard MobileNetV1 block)."""
    x = layers.Conv2D(
        filters, kernel_size, strides=stride, padding=padding,
        use_bias=False, name=f'{name}_conv',
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.ReLU(6.0, name=f'{name}_relu')(x)
    return x


def _dw_sep(x, out_channels: int,
            stride: int = 1,
            name: str = '') -> tf.Tensor:
    """
    MobileNetV1 depthwise-separable block.

        DW Conv2D 3×3 → BN → ReLU6
        PW Conv2D 1×1 → BN → ReLU6
    """
    # Depthwise
    x = layers.DepthwiseConv2D(
        3, strides=stride, padding='same',
        use_bias=False, name=f'{name}_dw',
    )(x)
    x = layers.BatchNormalization(name=f'{name}_dw_bn')(x)
    x = layers.ReLU(6.0, name=f'{name}_dw_relu')(x)

    # Pointwise
    x = layers.Conv2D(
        out_channels, 1, padding='same',
        use_bias=False, name=f'{name}_pw',
    )(x)
    x = layers.BatchNormalization(name=f'{name}_pw_bn')(x)
    x = layers.ReLU(6.0, name=f'{name}_pw_relu')(x)
    return x


# ─────────────────────────── Model Builder ───────────────────────────────────

def build_mobilenet_ssd(num_classes: int = NUM_CLASSES_WITH_BG,
                        width: float = 1.0) -> tf.keras.Model:
    """
    Build MobileNetV1-SSD.

    Args:
        num_classes: total classes including background  (default 81)
        width:       channel-width multiplier  (0.25 / 0.5 / 0.75 / 1.0)

    Returns:
        tf.keras.Model  →  cls_out [B, 8732, C],  loc_out [B, 8732, 4]
    """
    def f(n): return max(1, int(n * width))

    inp = layers.Input(shape=(300, 300, 3), name='input_image')

    # ── Stage 0: 300×300 → 150×150 ──────────────────────────────────────────
    x = _conv_bn_relu6(inp, f(32), stride=2, name='conv0')

    # ── Stage 1-2: 150×150 → 75×75 ──────────────────────────────────────────
    x = _dw_sep(x, f(64),  stride=1, name='dw1')
    x = _dw_sep(x, f(128), stride=2, name='dw2')   # → 75×75
    x = _dw_sep(x, f(128), stride=1, name='dw3')

    # ── Stage 3: 75×75 → 38×38 ──────────────────────────────────────────────
    x = _dw_sep(x, f(256), stride=2, name='dw4')   # → 38×38
    x = _dw_sep(x, f(256), stride=1, name='dw5')
    feat_38 = x                                     # 38×38 × 256·w

    # ── Stage 4: 38×38 → 19×19 (6 blocks) ───────────────────────────────────
    x = _dw_sep(x, f(512), stride=2, name='dw6')   # → 19×19
    for k in range(7, 12):
        x = _dw_sep(x, f(512), stride=1, name=f'dw{k}')
    feat_19 = x                                     # 19×19 × 512·w

    # ── Stage 5: 19×19 → 10×10 ──────────────────────────────────────────────
    x = _dw_sep(x, f(1024), stride=2, name='dw12')  # → 10×10
    x = _dw_sep(x, f(1024), stride=1, name='dw13')
    feat_10 = x                                      # 10×10 × 1024·w

    # ── Extra SSD Layers ─────────────────────────────────────────────────────
    feat_5, feat_3, feat_1 = build_extra_layers(feat_10, width=width)

    return assemble_ssd(
        inp,
        [feat_38, feat_19, feat_10, feat_5, feat_3, feat_1],
        num_classes=num_classes,
        model_name='MobileNetV1_SSD',
    )
