"""
models/resnet_ssd.py — ResNet50-SSD.

Backbone:  ResNet50 (bottleneck residual blocks, BatchNorm, ReLU).
Reference: He et al., "Deep Residual Learning for Image Recognition",
           CVPR 2016.

Feature maps tapped at:
    C3  →  38×38  (512  ch,  end of res-stage 2)
    C4  →  19×19  (1024 ch,  end of res-stage 3)
    C5  →  10×10  (2048 ch,  end of res-stage 4, stride reduced to 1 with
                             dilation so the map stays at 10×10 when input
                             is 300×300)
Extra SSD layers → 5×5 · 3×3 · 1×1

Stride schedule for 300×300 input:
    stem  stride 2 → 150×150
    pool  stride 2 →  75×75
    C2    stride 1 →  75×75   (res2)
    C3    stride 2 →  38×38   (res3)  ← feat_38
    C4    stride 2 →  19×19   (res4)  ← feat_19
    C5    stride 2 →  10×10   (res5, last block dilated) ← feat_10
"""

import tensorflow as tf
from tensorflow.keras import layers
from config import NUM_CLASSES_WITH_BG
from .ssd_common import build_extra_layers, assemble_ssd


# ─────────────────────────── Building Blocks ─────────────────────────────────

def _bottleneck(x: tf.Tensor,
                filters: int,
                stride: int = 1,
                dilation: int = 1,
                project: bool = False,
                name: str = '') -> tf.Tensor:
    """
    ResNet50 bottleneck block.

        1×1 Conv  → BN → ReLU          (reduce)
        3×3 Conv  → BN → ReLU          (spatial, possibly dilated)
        1×1 Conv  → BN                 (expand to 4×filters)
        + shortcut (1×1 projection if shapes differ)
        → ReLU

    Args:
        filters : number of filters for the 1×1 reduce/expand layers
                  (expand output = 4 × filters)
        stride  : spatial stride applied to the 3×3 conv
        dilation: dilation rate for the 3×3 conv (use instead of stride
                  to preserve feature map size in C5)
        project : whether to apply a 1×1 projection to the shortcut
    """
    out = filters * 4

    # ── Main path ─────────────────────────────────────────────────────────
    h = layers.Conv2D(
        filters, 1, use_bias=False, name=f'{name}_1x1a'
    )(x)
    h = layers.BatchNormalization(name=f'{name}_1x1a_bn')(h)
    h = layers.ReLU(name=f'{name}_1x1a_relu')(h)

    h = layers.Conv2D(
        filters, 3,
        strides=stride, padding='same', dilation_rate=dilation,
        use_bias=False, name=f'{name}_3x3',
    )(h)
    h = layers.BatchNormalization(name=f'{name}_3x3_bn')(h)
    h = layers.ReLU(name=f'{name}_3x3_relu')(h)

    h = layers.Conv2D(
        out, 1, use_bias=False, name=f'{name}_1x1b'
    )(h)
    h = layers.BatchNormalization(name=f'{name}_1x1b_bn')(h)

    # ── Shortcut ──────────────────────────────────────────────────────────
    if project:
        x = layers.Conv2D(
            out, 1, strides=stride,
            use_bias=False, name=f'{name}_proj',
        )(x)
        x = layers.BatchNormalization(name=f'{name}_proj_bn')(x)

    return layers.ReLU(name=f'{name}_out')(layers.Add(name=f'{name}_add')([x, h]))


def _res_stage(x: tf.Tensor,
               filters: int,
               num_blocks: int,
               stride: int = 2,
               dilation: int = 1,
               name: str = '') -> tf.Tensor:
    """
    One ResNet stage: first block (with optional downsampling / projection)
    followed by (num_blocks - 1) identity blocks.

    Args:
        filters   : base filter count (output channels = filters × 4)
        num_blocks: total blocks in this stage
        stride    : stride applied in the FIRST block only
        dilation  : dilation for the 3×3 conv (only used in stage C5)
    """
    in_channels = x.shape[-1]
    out_channels = filters * 4
    needs_proj   = (in_channels != out_channels) or (stride != 1)

    x = _bottleneck(x, filters,
                    stride=stride, dilation=dilation,
                    project=needs_proj, name=f'{name}_b0')

    for i in range(1, num_blocks):
        x = _bottleneck(x, filters,
                        stride=1, dilation=dilation,
                        project=False, name=f'{name}_b{i}')
    return x


# ─────────────────────────── Model Builder ───────────────────────────────────

def build_resnet_ssd(num_classes: int = NUM_CLASSES_WITH_BG,
                     width: float = 1.0) -> tf.keras.Model:
    """
    Build ResNet50-SSD.

    For ResNet50, `width` scales only the shared SSD extra layers (backbone
    channel counts follow the standard ResNet50 spec: 256-512-1024-2048).

    Args:
        num_classes: total classes including background  (default 81)
        width:       channel-width multiplier for extra SSD layers only

    Returns:
        tf.keras.Model  →  cls_out [B, 8732, C],  loc_out [B, 8732, 4]
    """
    inp = layers.Input(shape=(300, 300, 3), name='input_image')

    # ── Stem: 300×300 → 75×75 ───────────────────────────────────────────────
    x = layers.Conv2D(
        64, 7, strides=2, padding='same',
        use_bias=False, name='stem_conv',
    )(inp)                                               # → 150×150
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.ReLU(name='stem_relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same',
                            name='stem_pool')(x)         # →  75×75

    # ── C2: 75×75  (stride 1, 3 blocks) ─────────────────────────────────────
    x = _res_stage(x, filters=64,  num_blocks=3, stride=1, name='C2')
    #                                                    # →  75×75 × 256

    # ── C3: 75×75 → 38×38  (stride 2, 4 blocks) ─────────────────────────────
    x = _res_stage(x, filters=128, num_blocks=4, stride=2, name='C3')
    feat_38 = x                                          # →  38×38 × 512

    # ── C4: 38×38 → 19×19  (stride 2, 6 blocks) ─────────────────────────────
    x = _res_stage(x, filters=256, num_blocks=6, stride=2, name='C4')
    feat_19 = x                                          # →  19×19 × 1024

    # ── C5: 19×19 → 10×10  (stride 2, 3 blocks) ─────────────────────────────
    # Last block uses dilation=2 on the 3×3 conv to preserve receptive field
    # while halving the spatial resolution via stride.
    x = _res_stage(x, filters=512, num_blocks=3, stride=2, dilation=1,
                   name='C5')
    feat_10 = x                                          # →  10×10 × 2048

    # ── Extra SSD Layers ─────────────────────────────────────────────────────
    feat_5, feat_3, feat_1 = build_extra_layers(feat_10, width=width)

    return assemble_ssd(
        inp,
        [feat_38, feat_19, feat_10, feat_5, feat_3, feat_1],
        num_classes=num_classes,
        model_name='ResNet50_SSD',
    )
