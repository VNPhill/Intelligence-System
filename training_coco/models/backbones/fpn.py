"""
models/backbones/fpn.py — Feature Pyramid Network (FPN).

Reference: Lin et al., "Feature Pyramid Networks for Object Detection",
           CVPR 2017.

Takes three backbone feature maps (C3, C4, C5) and returns five FPN levels:
    P3 :  38×38  (from C3 + lateral)
    P4 :  19×19  (from C4 + lateral)
    P5 :  10×10  (from C5 + lateral)
    P6 :   5×5   (stride-2 conv on C5)
    P7 :   3×3   (stride-2 conv on P6, with ReLU)

Used by RetinaNet and FCOS.  YOLOv3 uses its own cross-scale feature fusion.
CenterNet uses a separate decoder.
"""

import tensorflow as tf
from tensorflow.keras import layers


def build_fpn(C3: tf.Tensor,
              C4: tf.Tensor,
              C5: tf.Tensor,
              out_channels: int = 256) -> list:
    """
    Build the FPN top-down pathway and return [P3, P4, P5, P6, P7].

    Args:
        C3, C4, C5   : backbone feature tensors at strides 8, 16, 32
        out_channels : uniform channel count for all FPN levels (default 256)

    Returns:
        list of 5 tensors [P3, P4, P5, P6, P7]
    """
    f = out_channels

    # ── Lateral 1×1 convolutions to unify channel counts ─────────────────────
    lat_C3 = layers.Conv2D(f, 1, padding='same',
                           use_bias=True, name='fpn_lat_c3')(C3)
    lat_C4 = layers.Conv2D(f, 1, padding='same',
                           use_bias=True, name='fpn_lat_c4')(C4)
    lat_C5 = layers.Conv2D(f, 1, padding='same',
                           use_bias=True, name='fpn_lat_c5')(C5)

    # ── Top-down pathway ──────────────────────────────────────────────────────
    # P5 ← lat_C5
    p5 = lat_C5

    # # P4 ← lat_C4 + upsample(P5)
    # up5 = layers.UpSampling2D(size=2, interpolation='nearest', name='fpn_up_p5')(p5)
    # p4  = layers.Add(name='fpn_add_p4')([lat_C4, up5])

    # # P3 ← lat_C3 + upsample(P4)
    # up4 = layers.UpSampling2D(size=2, interpolation='nearest', name='fpn_up_p4')(p4)
    # p3  = layers.Add(name='fpn_add_p3')([lat_C3, up4])

    # P4 ← lat_C4 + upsample(P5)
    up5 = layers.Lambda(
        lambda x: tf.image.resize(x[0], tf.shape(x[1])[1:3], method='nearest'),
        name='fpn_resize_p5'
    )([p5, lat_C4])
    p4  = layers.Add(name='fpn_add_p4')([lat_C4, up5])

    # P3 ← lat_C3 + upsample(P4)
    up4 = layers.Lambda(
        lambda x: tf.image.resize(x[0], tf.shape(x[1])[1:3], method='nearest'),
        name='fpn_resize_p4'
    )([p4, lat_C3])
    p3  = layers.Add(name='fpn_add_p3')([lat_C3, up4])

    # ── Output 3×3 convolutions (anti-aliasing) ───────────────────────────────
    P3 = layers.Conv2D(f, 3, padding='same', use_bias=True, name='fpn_out_p3')(p3)
    P4 = layers.Conv2D(f, 3, padding='same', use_bias=True, name='fpn_out_p4')(p4)
    P5 = layers.Conv2D(f, 3, padding='same', use_bias=True, name='fpn_out_p5')(p5)

    # ── Extra levels (no lateral connection, stride-2 convs) ──────────────────
    # P6 from C5 directly (not from P5) — as in the RetinaNet paper
    P6 = layers.Conv2D(f, 3, strides=2, padding='same',
                       use_bias=True, name='fpn_out_p6')(C5)

    # P7 from ReLU(P6)
    P7 = layers.Conv2D(f, 3, strides=2, padding='same',
                       use_bias=True,
                       name='fpn_out_p7')(layers.ReLU(name='fpn_p6_relu')(P6))

    return [P3, P4, P5, P6, P7]
