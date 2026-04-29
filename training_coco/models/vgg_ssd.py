"""
models/vgg_ssd.py — VGG16-SSD.

This is the original SSD architecture from Liu et al. (ECCV 2016).

Backbone modifications vs. standard VGG16:
  • pool5  uses kernel 3×3, stride 1 (keeps 19×19 spatial size)
  • fc6/fc7 replaced with atrous Conv2D (dilation=6) and 1×1 Conv2D
  • L2 normalisation on conv4_3 feature map (scale learned per-channel)
  • BatchNorm added throughout for training-from-scratch stability

Reference: Liu et al., "SSD: Single Shot MultiBox Detector", ECCV 2016.

Feature maps tapped at:
    conv4_3   →  38×38  (512 ch, L2-normalised)
    conv7     →  19×19  (1024 ch)
Extra SSD layers → 10×10 · 5×5 · 3×3 · 1×1
"""

import tensorflow as tf
from tensorflow.keras import layers, initializers
from config import NUM_CLASSES_WITH_BG
from .ssd_common import build_extra_layers, assemble_ssd


# ─────────────────────────── L2 Normalisation ────────────────────────────────

class L2Normalise(layers.Layer):
    """
    Per-channel L2 normalisation with a learnable scale vector γ.

    In the original SSD paper, conv4_3 features have much larger L2 norm
    than other feature maps.  This layer normalises each spatial location to
    unit norm and then rescales by a learned γ (initialised to 20).
    """

    def __init__(self, n_channels: int, init_scale: float = 20.0, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.init_scale = init_scale

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=(1, 1, 1, self.n_channels),
            initializer=initializers.Constant(self.init_scale),
            trainable=True,
        )

    def call(self, x):
        norm = tf.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10
        return (x / norm) * self.gamma


# ─────────────────────────── Building Blocks ─────────────────────────────────

def _conv_block(x, filters: int, num_convs: int,
                pool: bool = True, pool_stride: int = 2,
                pool_size: int = 2, pool_padding: str = 'valid',
                name: str = '') -> tf.Tensor:
    """
    VGG convolutional block: num_convs × (Conv → BN → ReLU)  [+ MaxPool].

    BN is added for training-from-scratch stability (not in the original
    VGG paper, but standard practice when not fine-tuning from ImageNet).
    """
    for i in range(num_convs):
        x = layers.Conv2D(
            filters, 3, padding='same',
            use_bias=False, name=f'{name}_c{i + 1}',
        )(x)
        x = layers.BatchNormalization(name=f'{name}_c{i + 1}_bn')(x)
        x = layers.ReLU(name=f'{name}_c{i + 1}_relu')(x)

    if pool:
        x = layers.MaxPooling2D(
            pool_size=pool_size, strides=pool_stride,
            padding=pool_padding, name=f'{name}_pool',
        )(x)
    return x


# ─────────────────────────── Model Builder ───────────────────────────────────

def build_vgg_ssd(num_classes: int = NUM_CLASSES_WITH_BG,
                  width: float = 1.0) -> tf.keras.Model:
    """
    Build VGG16-SSD.

    For VGG16, `width` is only applied to the shared SSD extra layers;
    the backbone channel counts follow the standard VGG16 specification.

    Args:
        num_classes: total classes including background  (default 81)
        width:       channel-width multiplier for extra SSD layers only

    Returns:
        tf.keras.Model  →  cls_out [B, 8732, C],  loc_out [B, 8732, 4]
    """
    inp = layers.Input(shape=(300, 300, 3), name='input_image')

    # ── VGG16 Backbone ───────────────────────────────────────────────────────

    # Block 1: 300×300 → 150×150
    x = _conv_block(inp, 64, num_convs=2,
                    pool=True, pool_stride=2, pool_padding='valid',
                    name='block1')

    # Block 2: 150×150 → 75×75
    x = _conv_block(x, 128, num_convs=2,
                    pool=True, pool_stride=2, pool_padding='valid',
                    name='block2')

    # Block 3: 75×75 → 38×38
    # MaxPool with 'same' padding: ceil(75 / 2) = 38
    x = _conv_block(x, 256, num_convs=3,
                    pool=True, pool_stride=2, pool_padding='same',
                    name='block3')

    # Block 4: 38×38 (no downsampling — conv4_3 is the first SSD feature map)
    x = _conv_block(x, 512, num_convs=3, pool=False, name='block4')
    # L2-normalise conv4_3 before handing it to the detection head
    feat_38 = L2Normalise(n_channels=512, init_scale=20.0,
                          name='conv4_3_l2norm')(x)    # 38×38 × 512

    # Pool4: 38×38 → 19×19
    x = layers.MaxPooling2D(2, strides=2, padding='valid',
                            name='block4_pool')(x)

    # Block 5: 19×19  (pool5 uses stride=1 to keep 19×19)
    x = _conv_block(x, 512, num_convs=3, pool=False, name='block5')
    x = layers.MaxPooling2D(pool_size=3, strides=1, padding='same',
                            name='block5_pool')(x)     # still 19×19

    # ── Modified FC layers (conv6 atrous + conv7 1×1) ────────────────────────

    # conv6: dilated 3×3 replaces fc6  (dilation_rate=6, no BN in original
    #        but added here for stability)
    x = layers.Conv2D(
        1024, 3, padding='same', dilation_rate=6,
        use_bias=False, name='conv6',
    )(x)
    x = layers.BatchNormalization(name='conv6_bn')(x)
    x = layers.ReLU(name='conv6_relu')(x)

    # conv7: 1×1 replaces fc7
    x = layers.Conv2D(
        1024, 1, padding='same',
        use_bias=False, name='conv7',
    )(x)
    x = layers.BatchNormalization(name='conv7_bn')(x)
    x = layers.ReLU(name='conv7_relu')(x)
    feat_19 = x                                        # 19×19 × 1024

    # ── Extra SSD Layers (shared implementation) ─────────────────────────────
    feat_5, feat_3, feat_1 = build_extra_layers(feat_19, width=width)
    # build_extra_layers expects a 10×10 input; here feat_19 is 19×19 so we
    # need to derive feat_10 first using a stride-2 extra block.

    # Override: VGG needs an extra stride-2 step (19×19 → 10×10) before the
    # standard extra-layer sequence.  Re-derive feat_10 and then regenerate.
    from .ssd_common import conv_bn_relu
    def f(n): return max(1, int(n * width))

    # 19×19 → 10×10  (first extra layer for VGG)
    x = conv_bn_relu(feat_19, f(256), kernel_size=1, name='extra0_1x1')
    x = conv_bn_relu(x,       f(512), stride=2,      name='extra0_3x3')
    feat_10 = x                                        # 10×10 × 512·w

    # 10×10 → 5×5 → 3×3 → 1×1
    feat_5, feat_3, feat_1 = build_extra_layers(feat_10, width=width)

    return assemble_ssd(
        inp,
        [feat_38, feat_19, feat_10, feat_5, feat_3, feat_1],
        num_classes=num_classes,
        model_name='VGG16_SSD',
    )
