"""
models/ssd_common.py — Shared SSD components used by every backbone.

Public API:
    conv_bn_relu()        Standard Conv → BN → ReLU block.
    build_extra_layers()  Appended SSD layers: 10×10 → 5×5 → 3×3 → 1×1.
    assemble_ssd()        Attach prediction heads, reshape, concatenate,
                          and return a tf.keras.Model.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from config import ANCHORS_PER_CELL, FEATURE_MAP_SIZES, NUM_CLASSES_WITH_BG


# ─────────────────────────── Building Block ──────────────────────────────────

def conv_bn_relu(x, filters: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: str = 'same',
                 dilation: int = 1,
                 name: str = '') -> tf.Tensor:
    """
    Conv2D → BatchNormalization → ReLU.

    Used for the shared extra SSD layers (standard ReLU, no cap).
    Backbone modules define their own variants with ReLU6 / no-act as needed.
    """
    x = layers.Conv2D(
        filters, kernel_size,
        strides=stride, padding=padding,
        dilation_rate=dilation,
        use_bias=False, name=f'{name}_conv',
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.ReLU(name=f'{name}_relu')(x)
    return x


# ──────────────────────── Extra SSD Layers ───────────────────────────────────

def build_extra_layers(feat_10: tf.Tensor,
                       width: float = 1.0):
    """
    Standard SSD extra detection layers that extend the backbone.

        feat_10  (10×10)  →  feat_5  (5×5)
                          →  feat_3  (3×3)
                          →  feat_1  (1×1)

    Each step uses a bottleneck-style 1×1 → 3×3 pair.  The width multiplier
    scales channel counts; it is shared with the backbone for MobileNets but
    applied independently for VGG / ResNet.

    Args:
        feat_10 : tensor with spatial size 10×10
        width   : channel-width multiplier (default 1.0)

    Returns:
        (feat_5, feat_3, feat_1)
    """
    def f(n): return max(1, int(n * width))

    # 10×10 → 5×5
    x      = conv_bn_relu(feat_10, f(256), kernel_size=1, name='extra1_1x1')
    x      = conv_bn_relu(x,       f(512), stride=2,      name='extra1_3x3')
    feat_5 = x

    # 5×5 → 3×3
    x      = conv_bn_relu(feat_5, f(128), kernel_size=1, name='extra2_1x1')
    x      = conv_bn_relu(x,      f(256), stride=2,      name='extra2_3x3')
    feat_3 = x

    # 3×3 → 1×1   ('valid' padding collapses the 3×3 spatial map completely)
    x      = conv_bn_relu(feat_3, f(128), kernel_size=1,       name='extra3_1x1')
    x      = conv_bn_relu(x,      f(256), padding='valid',     name='extra3_3x3')
    feat_1 = x

    return feat_5, feat_3, feat_1


# ──────────────────────── Prediction Heads ───────────────────────────────────

def _prediction_head(feature: tf.Tensor,
                     num_anchors: int,
                     num_classes: int,
                     name: str = ''):
    """
    Two parallel 3×3 convolutions: class logits and box offsets.
    No BN / activation — raw outputs are consumed by SSDLoss.
    """
    cls_pred = layers.Conv2D(
        num_anchors * num_classes, 3,
        padding='same', name=f'{name}_cls',
    )(feature)
    loc_pred = layers.Conv2D(
        num_anchors * 4, 3,
        padding='same', name=f'{name}_loc',
    )(feature)
    return cls_pred, loc_pred


# ───────────────────────── Final Assembly ────────────────────────────────────

def assemble_ssd(inp,
                 features: list,
                 num_classes: int = NUM_CLASSES_WITH_BG,
                 model_name: str = 'SSD') -> Model:
    """
    Attach SSD prediction heads to all six feature maps, reshape, and
    concatenate into the final [B, 8732, *] outputs.

    Args:
        inp        : Keras Input tensor (the model's input node)
        features   : list of exactly 6 tensors
                     [feat_38, feat_19, feat_10, feat_5, feat_3, feat_1]
        num_classes: total classes including background  (default 81)
        model_name : name assigned to the returned tf.keras.Model

    Returns:
        tf.keras.Model whose outputs are:
            cls_out  [B, 8732, num_classes]  raw logits
            loc_out  [B, 8732, 4]            encoded box offsets
    """
    if len(features) != 6:
        raise ValueError(
            f"assemble_ssd requires exactly 6 feature maps, got {len(features)}.")

    cls_outputs: list = []
    loc_outputs: list = []

    for i, (feat, n_anchors, fmap_size) in enumerate(
            zip(features, ANCHORS_PER_CELL, FEATURE_MAP_SIZES)):

        cls_p, loc_p = _prediction_head(
            feat, n_anchors, num_classes, name=f'head{i}')

        n_boxes = fmap_size * fmap_size * n_anchors
        cls_p   = layers.Reshape(
            (n_boxes, num_classes), name=f'cls_reshape{i}'
        )(cls_p)
        loc_p   = layers.Reshape(
            (n_boxes, 4), name=f'loc_reshape{i}'
        )(loc_p)

        cls_outputs.append(cls_p)
        loc_outputs.append(loc_p)

    cls_out = layers.Concatenate(axis=1, name='cls_out')(cls_outputs)
    loc_out = layers.Concatenate(axis=1, name='loc_out')(loc_outputs)

    return Model(inputs=inp, outputs=[cls_out, loc_out], name=model_name)
