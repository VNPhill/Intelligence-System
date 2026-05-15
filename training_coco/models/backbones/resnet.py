"""
models/backbones/resnet.py — Shared ResNet50 feature extractor.

Returns the three intermediate feature maps used by RetinaNet, FCOS,
CenterNet, and ResNet-SSD.

Output tensor shapes for 300×300 input:
    C3 :  38×38 ×  512    (stride  8)
    C4 :  19×19 × 1024    (stride 16)
    C5 :  10×10 × 2048    (stride 32)
"""

import tensorflow as tf
from tensorflow.keras import layers


# ─────────────────────────── Bottleneck Block ────────────────────────────────

def _bottleneck(x: tf.Tensor,
                filters: int,
                stride: int   = 1,
                dilation: int = 1,
                project: bool = False,
                name: str     = '') -> tf.Tensor:
    """
    ResNet50 bottleneck:  1×1 → 3×3 (strided/dilated) → 1×1 → +shortcut → ReLU
    Output channels = filters × 4.
    """
    out = filters * 4

    h = layers.Conv2D(filters, 1, use_bias=False, name=f'{name}_c1')(x)
    h = layers.BatchNormalization(name=f'{name}_c1_bn')(h)
    h = layers.ReLU(name=f'{name}_c1_relu')(h)

    h = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      dilation_rate=dilation,
                      use_bias=False, name=f'{name}_c2')(h)
    h = layers.BatchNormalization(name=f'{name}_c2_bn')(h)
    h = layers.ReLU(name=f'{name}_c2_relu')(h)

    h = layers.Conv2D(out, 1, use_bias=False, name=f'{name}_c3')(h)
    h = layers.BatchNormalization(name=f'{name}_c3_bn')(h)

    if project:
        x = layers.Conv2D(out, 1, strides=stride,
                          use_bias=False, name=f'{name}_proj')(x)
        x = layers.BatchNormalization(name=f'{name}_proj_bn')(x)

    return layers.ReLU(name=f'{name}_out')(layers.Add(name=f'{name}_add')([x, h]))


def _stage(x: tf.Tensor,
           filters: int,
           num_blocks: int,
           stride: int   = 2,
           dilation: int = 1,
           name: str     = '') -> tf.Tensor:
    """Stack of bottleneck blocks; first block handles downsampling/projection."""
    needs_proj = (x.shape[-1] != filters * 4) or (stride != 1)
    x = _bottleneck(x, filters, stride=stride, dilation=dilation,
                    project=needs_proj, name=f'{name}_b0')
    for i in range(1, num_blocks):
        x = _bottleneck(x, filters, stride=1, dilation=dilation,
                        project=False, name=f'{name}_b{i}')
    return x


# ─────────────────────────── Public Builder ──────────────────────────────────

def build_resnet50_features(inp: tf.Tensor,
                             prefix: str = 'rn50'):
    """
    Build ResNet50 body attached to the provided Input tensor.

    Args:
        inp    : Keras tensor with shape (B, 300, 300, 3)
        prefix : name prefix to avoid collisions when the backbone is reused

    Returns:
        (C3, C4, C5)  —  three feature tensors
    """
    # ── Stem ─────────────────────────────────────────────────────────────────
    x = layers.Conv2D(64, 7, strides=2, padding='same',
                      use_bias=False, name=f'{prefix}_stem_conv')(inp)  # 150×150
    x = layers.BatchNormalization(name=f'{prefix}_stem_bn')(x)
    x = layers.ReLU(name=f'{prefix}_stem_relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same',
                            name=f'{prefix}_stem_pool')(x)               # 75×75

    # ── C2: 75×75 × 256  (stride 1, 3 blocks) ───────────────────────────────
    x = _stage(x, filters=64,  num_blocks=3, stride=1, name=f'{prefix}_C2')

    # ── C3: 38×38 × 512  (stride 2, 4 blocks) ───────────────────────────────
    x  = _stage(x, filters=128, num_blocks=4, stride=2, name=f'{prefix}_C3')
    C3 = x

    # ── C4: 19×19 × 1024 (stride 2, 6 blocks) ───────────────────────────────
    x  = _stage(x, filters=256, num_blocks=6, stride=2, name=f'{prefix}_C4')
    C4 = x

    # ── C5: 10×10 × 2048 (stride 2, 3 blocks) ───────────────────────────────
    x  = _stage(x, filters=512, num_blocks=3, stride=2, name=f'{prefix}_C5')
    C5 = x

    return C3, C4, C5
