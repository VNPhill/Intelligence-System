"""
models/centernet.py — CenterNet: Objects as Points.

Reference: Zhou et al., "Objects as Points", 2019.

Architecture:
    ResNet50 backbone  →  C5  (10×10 × 2048)
    Deconv decoder     →  38×38 × 64   (3 × transposed conv + BN + ReLU)

    Three output heads (all 1×1 conv):
        heatmap  [B, H, W, C]    — one channel per class, Gaussian peaks
        size     [B, H, W, 2]    — log(w), log(h) at object centres
        offset   [B, H, W, 2]    — fractional centre offset (quantisation fix)

    OUTPUT_STRIDE = 8  →  heatmap size = 38×38 for 300×300 input.

Training targets:
    Heatmap: 2-D Gaussian splat centred at each object's downsampled centre.
             Radius r = gaussian_radius(box_wh) following the TTFNet criterion.
    Size:    log(w/INPUT_SIZE), log(h/INPUT_SIZE) at the integer centre pixel.
    Offset:  (cx / stride - floor(cx / stride),
              cy / stride - floor(cy / stride))

Inference (per image):
    1. Sigmoid heatmap → apply 3×3 max-pool (finds local peaks).
    2. Take top-K peaks per class above conf_threshold.
    3. Decode (cx, cy) from peak location + offset.
    4. Decode (w, h) from size prediction.
    5. Apply NMS on decoded boxes.
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from models.base            import DetectionModel
from models.backbones       import build_resnet50_features
from losses.centernet_loss  import CenterNetLoss
from config import (
    NUM_CLASSES, NUM_CLASSES_WITH_BG, INPUT_SIZE,
    CENTERNET_OUTPUT_STRIDE, CENTERNET_HEATMAP_SIZE,
    CENTERNET_DECONV_CHANNELS, CENTERNET_MIN_OVERLAP,
)


# ──────────────────────── Gaussian Radius Utility ────────────────────────────

def _gaussian_radius(det_size: tuple, min_overlap: float = CENTERNET_MIN_OVERLAP) -> int:
    """
    Compute the minimum Gaussian radius such that a circle of that radius
    around the GT centre has at least `min_overlap` IoU with the GT box.

    Follows the CornerNet / TTFNet calculation:
        r = (1 − √(min_overlap)) / (1 + √(min_overlap)) · min(h, w) / 2

    Args:
        det_size    : (h, w) of the box in heatmap-resolution pixels
        min_overlap : minimum allowed IoU

    Returns:
        integer radius (minimum 0)
    """
    h, w = det_size
    ratio = (1.0 - math.sqrt(min_overlap)) / (1.0 + math.sqrt(min_overlap))
    r = ratio * min(h, w) / 2.0
    return max(0, int(r))


def _draw_gaussian(hmap: np.ndarray, cx: int, cy: int, radius: int):
    """
    Draw a 2-D Gaussian peak into `hmap[cy, cx]` in-place.
    Values are clipped to max 1; existing values are max-merged.

    Args:
        hmap   : 2-D array [H, W]  (single class channel)
        cx, cy : integer peak location in heatmap coordinates
        radius : Gaussian sigma ≈ radius / 3
    """
    sigma  = (2 * radius + 1) / 6.0
    H, W   = hmap.shape

    # Gaussian kernel
    size = 2 * radius + 1
    x = np.arange(0, size) - radius
    gaussian_1d = np.exp(-0.5 * (x / sigma) ** 2)
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)

    # Clip to heatmap boundaries
    x0, y0 = cx - radius, cy - radius
    x1, y1 = cx + radius + 1, cy + radius + 1

    img_x0 = max(0, x0);  img_x1 = min(W, x1)
    img_y0 = max(0, y0);  img_y1 = min(H, y1)

    ker_x0 = img_x0 - x0;  ker_x1 = ker_x0 + (img_x1 - img_x0)
    ker_y0 = img_y0 - y0;  ker_y1 = ker_y0 + (img_y1 - img_y0)

    if img_x1 <= img_x0 or img_y1 <= img_y0:
        return

    hmap[img_y0:img_y1, img_x0:img_x1] = np.maximum(
        hmap[img_y0:img_y1, img_x0:img_x1],
        gaussian_2d[ker_y0:ker_y1, ker_x0:ker_x1],
    )


# ──────────────────────── Target Encoding ────────────────────────────────────

def _encode_centernet_targets(gt_boxes_batch: np.ndarray,
                               gt_labels_batch: np.ndarray,
                               num_valid_batch: np.ndarray) -> dict:
    """
    Encode raw GT into CenterNet heatmap / size / offset targets.

    Args:
        gt_boxes_batch  : [B, MAX_GT, 4]  cx,cy,w,h normalised
        gt_labels_batch : [B, MAX_GT]     1-based
        num_valid_batch : [B]

    Returns dict of tf.Tensors:
        'hmap'     : [B, H, W, C]
        'size'     : [B, H, W, 2]   log-space
        'offset'   : [B, H, W, 2]   fractional offset
        'obj_mask' : [B, H, W]      bool, True at GT centre pixels
    """
    B  = gt_boxes_batch.shape[0]
    C  = NUM_CLASSES
    H  = W = CENTERNET_HEATMAP_SIZE
    print()
    st = CENTERNET_OUTPUT_STRIDE

    hmap_b  = np.zeros((B, H, W, C),  np.float32)
    size_b  = np.zeros((B, H, W, 2),  np.float32)
    off_b   = np.zeros((B, H, W, 2),  np.float32)
    mask_b  = np.zeros((B, H, W),     bool)

    for b in range(B):
        n = int(num_valid_batch[b])
        for k in range(n):
            cx_n, cy_n, w_n, h_n = gt_boxes_batch[b, k]
            label = int(gt_labels_batch[b, k]) - 1   # 0-based

            # Heatmap-space centre (float then floor)
            cx_f = cx_n * INPUT_SIZE / st
            cy_f = cy_n * INPUT_SIZE / st
            cx_i = int(cx_f)
            cy_i = int(cy_f)
            cx_i = min(cx_i, W - 1)
            cy_i = min(cy_i, H - 1)

            # Heatmap-space box size
            bw_h = w_n * INPUT_SIZE / st
            bh_h = h_n * INPUT_SIZE / st
            radius = _gaussian_radius((bh_h, bw_h))

            _draw_gaussian(hmap_b[b, :, :, label], cx_i, cy_i, radius)

            # Size target: log(w), log(h) in normalised units
            size_b[b, cy_i, cx_i, 0] = math.log(max(w_n, 1e-6))
            size_b[b, cy_i, cx_i, 1] = math.log(max(h_n, 1e-6))

            # Fractional offset
            off_b[b, cy_i, cx_i, 0] = cx_f - cx_i
            off_b[b, cy_i, cx_i, 1] = cy_f - cy_i

            mask_b[b, cy_i, cx_i] = True

    return {
        'hmap':     tf.constant(hmap_b),
        'size':     tf.constant(size_b),
        'offset':   tf.constant(off_b),
        'obj_mask': tf.constant(mask_b),
    }


# ──────────────────────── Model Architecture ─────────────────────────────────

def _deconv_block(x: tf.Tensor,
                  channels: int,
                  name: str) -> tf.Tensor:
    """Transposed Conv 4×4 (stride 2) → BN → ReLU  (upsamples ×2)."""
    x = layers.Conv2DTranspose(
        channels, 4, strides=2, padding='same',
        use_bias=False, name=f'{name}_deconv',
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.ReLU(name=f'{name}_relu')(x)
    return x


# ──────────────────────── DetectionModel subclass ────────────────────────────

class CenterNet(DetectionModel):
    """CenterNet detector implementing the DetectionModel interface."""

    model_type    = 'centernet'
    target_format = 'raw'

    def __init__(self):
        self._loss_fn = CenterNetLoss()

    # ── build ─────────────────────────────────────────────────────────────────

    def build(self, num_classes: int = NUM_CLASSES_WITH_BG,
              width: float = 1.0) -> Model:
        """
        Returns a Keras Model with three outputs:
            hmap   [B, H, W, C]      raw logits (sigmoid in loss)
            size   [B, H, W, 2]      log(w), log(h)
            offset [B, H, W, 2]      fractional centre offset
        where H = W = CENTERNET_HEATMAP_SIZE = 38.
        """
        C = num_classes - 1    # no explicit background channel

        inp = layers.Input(shape=(300, 300, 3), name='input_image')

        # ResNet50 backbone — only C5 needed for CenterNet
        _, _, C5 = build_resnet50_features(inp, prefix='cn_rn')

        # ── Deconv decoder: 10×10 → 20×20 → 38×38 ─────────────────────────
        # Note: 10×10 → 20×20 → 40×40 then crop to 38 would be cleaner,
        # but we use a direct 10→19→38 path with a 4×4 kernel (valid at 10).
        # Simpler: use three ×2 upsamples and accept 40×40, then crop.
        x = C5    # [B, 10, 10, 2048]

        dec_channels = CENTERNET_DECONV_CHANNELS   # [256, 128, 64]
        for i, ch in enumerate(dec_channels):
            x = _deconv_block(x, ch, name=f'dec{i}')
        # After 3 × stride-2 deconv: 10 → 20 → 40 → 80; we want 38.
        # For a 300px input with stride-8 target: 300/8 = 37.5 → 38.
        # Crop to (38, 38) to match the heatmap size.
        x = layers.Cropping2D(cropping=((1, 1), (1, 1)), name='dec_crop')(x)
        # Now: [B, 38, 38, 64]

        # ── Prediction heads (shared conv + 1×1 final) ──────────────────────
        def _head(feat, out_ch, head_ch=64, name=''):
            h = layers.Conv2D(head_ch, 3, padding='same',
                              use_bias=False, name=f'{name}_conv')(feat)
            h = layers.BatchNormalization(name=f'{name}_bn')(h)
            h = layers.ReLU(name=f'{name}_relu')(h)
            return layers.Conv2D(out_ch, 1, name=f'{name}_out')(h)

        hmap_out   = _head(x, C,  name='hmap')    # [B, 38, 38, C]
        size_out   = _head(x, 2,  name='size')    # [B, 38, 38, 2]
        offset_out = _head(x, 2,  name='offset')  # [B, 38, 38, 2]

        return Model(inputs=inp,
                     outputs=[hmap_out, size_out, offset_out],
                     name='CenterNet')

    # ── encode_targets ────────────────────────────────────────────────────────

    def encode_targets(self, gt_boxes, gt_labels, num_valid) -> dict:
        return _encode_centernet_targets(gt_boxes, gt_labels, num_valid)

    # ── compute_loss ──────────────────────────────────────────────────────────

    def compute_loss(self, predictions, targets: dict) -> tuple:
        hmap_pred, size_pred, off_pred = predictions
        preds = {'hmap': hmap_pred, 'size': size_pred, 'offset': off_pred}
        return self._loss_fn(preds, targets)

    # ── postprocess ───────────────────────────────────────────────────────────

    def postprocess(self, predictions,
                    conf_threshold: float = 0.05,
                    nms_iou: float = 0.45,
                    max_dets: int = 200) -> tuple:
        """
        Decode CenterNet predictions for ONE image.

        Steps:
            1. Sigmoid heatmap + 3×3 max-pool pseudo-NMS (keep local maxima).
            2. Take top-K peaks above conf_threshold.
            3. Decode cx,cy from grid position + offset prediction.
            4. Decode w,h from size prediction (exp).
            5. Box NMS per class.
        """
        hmap_logits, size_pred, off_pred = predictions
        # hmap_logits: [H, W, C],  size_pred: [H, W, 2],  off_pred: [H, W, 2]

        H, W, C = hmap_logits.shape
        st = CENTERNET_OUTPUT_STRIDE

        hmap = tf.sigmoid(hmap_logits).numpy()     # [H, W, C]

        # 3×3 max-pool to suppress non-peak locations (heatmap NMS)
        hmap_t = tf.constant(hmap[np.newaxis])                   # [1,H,W,C]
        hmap_max = tf.nn.max_pool2d(hmap_t, ksize=3, strides=1,
                                    padding='SAME')[0].numpy()   # [H,W,C]
        hmap = np.where(hmap == hmap_max, hmap, 0.0)

        all_boxes, all_scores, all_labels = [], [], []

        for cls_idx in range(C):
            heat = hmap[:, :, cls_idx]             # [H, W]
            ys, xs = np.where(heat > conf_threshold)
            if len(ys) == 0:
                continue

            scores = heat[ys, xs].astype(np.float32)

            # Decode centres: point (xs[i], ys[i]) in heatmap + offset
            ox = off_pred[ys, xs, 0]               # fractional offset x
            oy = off_pred[ys, xs, 1]               # fractional offset y
            cx = (xs + ox) * st / INPUT_SIZE
            cy = (ys + oy) * st / INPUT_SIZE

            # Decode sizes: exp(log(w)), exp(log(h))
            log_w = size_pred[ys, xs, 0]
            log_h = size_pred[ys, xs, 1]
            bw = np.exp(log_w)
            bh = np.exp(log_h)

            x1 = np.clip(cx - bw / 2, 0, 1)
            y1 = np.clip(cy - bh / 2, 0, 1)
            x2 = np.clip(cx + bw / 2, 0, 1)
            y2 = np.clip(cy + bh / 2, 0, 1)

            fb   = np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)
            keep = tf.image.non_max_suppression(
                fb, scores, max_dets, nms_iou).numpy()

            all_boxes.extend(fb[keep].tolist())
            all_scores.extend(scores[keep].tolist())
            all_labels.extend([cls_idx + 1] * len(keep))

        if not all_boxes:
            return (np.zeros((0, 4), np.float32),
                    np.zeros(0, np.float32),
                    np.zeros(0, np.int32))
        return (np.array(all_boxes,  np.float32),
                np.array(all_scores, np.float32),
                np.array(all_labels, np.int32))
