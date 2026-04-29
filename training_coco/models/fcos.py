"""
models/fcos.py — FCOS: Fully Convolutional One-Stage Object Detection.

Reference: Tian et al., "FCOS: Fully Convolutional One-Stage Object
           Detection", ICCV 2019.

Architecture:
    ResNet50 backbone  →  C3, C4, C5
    FPN                →  P3 (38×38), P4 (19×19), P5 (10×10),
                          P6 (5×5),   P7 (3×3)

    Per FPN level, for EACH spatial location (point):
        Class head   : 4 × Conv3 + Conv1 → C logits       (sigmoid)
        Reg   head   : 4 × Conv3 + Conv1 → 4 distances    (exp → l,t,r,b)
        Centerness   : branch off reg head → 1 scalar      (sigmoid)

    Centerness = sqrt( min(l,r)/max(l,r) · min(t,b)/max(t,b) )
    suppresses low-quality detections without NMS score adjustment.

    Target assignment: a point at stride s belongs to level k if its
    assigned GT box's max(l,t,r,b) falls in FCOS_REGRESS_RANGES[k].

Output: list of 5 dicts, one per level
    {'cls':   [B, H, W, C],
     'reg':   [B, H, W, 4],   log-space distances
     'cness': [B, H, W, 1]}
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from models.base       import DetectionModel
from models.backbones  import build_resnet50_features, build_fpn
from losses.fcos_loss  import FCOSLoss
from config import (
    NUM_CLASSES, NUM_CLASSES_WITH_BG, INPUT_SIZE,
    FCOS_FPN_CHANNELS, FCOS_NUM_CONVS, FCOS_STRIDES,
    FCOS_REGRESS_RANGES, FCOS_CENTERNESS_ON_REG,
)


# ──────────────────────── Shared Head Subnet ─────────────────────────────────

def _build_fcos_head(in_channels: int,
                     num_convs: int,
                     name: str) -> tf.keras.Model:
    """
    Shared 4-conv tower applied at every FPN level.
    Returns a Keras model: (B,H,W,C_in) → (B,H,W,C_in).
    Weights are shared across levels (GN would be ideal but we use BN here).
    """
    inp = layers.Input(shape=(None, None, in_channels))
    x   = inp
    for i in range(num_convs):
        x = layers.Conv2D(in_channels, 3, padding='same',
                          use_bias=False, name=f'{name}_c{i}')(x)
        x = layers.BatchNormalization(name=f'{name}_bn{i}')(x)
        x = layers.ReLU(name=f'{name}_relu{i}')(x)
    return tf.keras.Model(inputs=inp, outputs=x, name=name)


# ──────────────────────── Target Encoding ────────────────────────────────────

def _encode_fcos_targets(gt_boxes_batch: np.ndarray,
                         gt_labels_batch: np.ndarray,
                         num_valid_batch: np.ndarray) -> dict:
    """
    Encode raw GT into FCOS training targets for the full batch.

    For each FPN level and each spatial point:
      - If the point falls inside any GT box AND that box's regression
        range matches the level → positive.
      - Among multiple overlapping GT boxes, assign the smallest-area one.
      - Regression target = (l, t, r, b) normalised to [0, 1] by INPUT_SIZE.
        Stored as log(distance) so exp(pred) gives the actual distance.
      - Centerness target = sqrt(min(l,r)/max(l,r) · min(t,b)/max(t,b)).

    Returns dict of lists (one entry per FPN level):
        'cls'      : [B, H, W, C]   float32 one-hot
        'reg'      : [B, H, W, 4]   float32 log-distances (l,t,r,b)
        'cness'    : [B, H, W, 1]   float32 centerness score
        'pos_mask' : [B, H, W]      bool    True at positive points
    """
    B        = gt_boxes_batch.shape[0]
    C        = NUM_CLASSES
    strides  = FCOS_STRIDES           # [8,16,32,64,128]
    ranges   = FCOS_REGRESS_RANGES    # [(lo,hi), ...]

    cls_list   = []
    reg_list   = []
    cness_list = []
    pos_list   = []

    for lvl, (stride, (r_lo, r_hi)) in enumerate(zip(strides, ranges)):
        H = INPUT_SIZE // stride
        W = H

        cls_t   = np.zeros((B, H, W, C),  np.float32)
        reg_t   = np.zeros((B, H, W, 4),  np.float32)
        cness_t = np.zeros((B, H, W, 1),  np.float32)
        pos_t   = np.zeros((B, H, W),     bool)

        # Pixel coordinates of each grid point's centre (normalised)
        ys = (np.arange(H) + 0.5) * stride / INPUT_SIZE   # [H]
        xs = (np.arange(W) + 0.5) * stride / INPUT_SIZE   # [W]
        grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')  # [H,W]

        for b in range(B):
            n = int(num_valid_batch[b])
            if n == 0:
                continue

            boxes  = gt_boxes_batch[b, :n]   # [n,4] cx,cy,w,h normalised
            labels = gt_labels_batch[b, :n]   # [n]   1-based

            # Convert to x1,y1,x2,y2
            x1 = boxes[:, 0] - boxes[:, 2] / 2   # [n]
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2

            areas = (x2 - x1) * (y2 - y1)        # [n]

            # For each point: distance to all box edges  [H,W,n]
            l = grid_x[:, :, np.newaxis] - x1[np.newaxis, np.newaxis, :]
            t = grid_y[:, :, np.newaxis] - y1[np.newaxis, np.newaxis, :]
            r = x2[np.newaxis, np.newaxis, :] - grid_x[:, :, np.newaxis]
            b_dist = y2[np.newaxis, np.newaxis, :] - grid_y[:, :, np.newaxis]

            # A point is inside a box if all four distances > 0
            inside = (l > 0) & (t > 0) & (r > 0) & (b_dist > 0)  # [H,W,n]

            # Max regression distance must fall within this level's range
            max_reg = np.maximum(
                np.maximum(l, r), np.maximum(t, b_dist))  # [H,W,n]
            in_range = (max_reg >= r_lo) & (max_reg <= r_hi)

            valid = inside & in_range                      # [H,W,n]

            # Among valid GT boxes per point, pick smallest area
            big_areas = np.where(valid,
                                 areas[np.newaxis, np.newaxis, :],
                                 np.inf)                   # [H,W,n]
            best_gt = np.argmin(big_areas, axis=-1)        # [H,W]
            any_valid = valid[
                np.arange(H)[:, None],
                np.arange(W)[None, :],
                best_gt]                                    # [H,W]

            pos_t[b] = any_valid

            # Fill targets for positive points
            if not any_valid.any():
                continue

            bj, bi = np.where(any_valid)
            bg      = best_gt[bj, bi]                      # [P]

            lv = l[bj, bi, bg]
            tv = t[bj, bi, bg]
            rv = r[bj, bi, bg]
            bv = b_dist[bj, bi, bg]

            # Log-distance regression targets (clamp to avoid log(0))
            reg_t[b, bj, bi, 0] = np.log(np.maximum(lv, 1e-6))
            reg_t[b, bj, bi, 1] = np.log(np.maximum(tv, 1e-6))
            reg_t[b, bj, bi, 2] = np.log(np.maximum(rv, 1e-6))
            reg_t[b, bj, bi, 3] = np.log(np.maximum(bv, 1e-6))

            # Centerness
            cness = np.sqrt(
                (np.minimum(lv, rv) / np.maximum(lv, rv).clip(1e-6)) *
                (np.minimum(tv, bv) / np.maximum(tv, bv).clip(1e-6))
            )
            cness_t[b, bj, bi, 0] = cness

            # One-hot class (0-based indexing inside cls_t)
            lab = labels[bg] - 1                           # 0-based
            cls_t[b, bj, bi, lab] = 1.0

        cls_list.append(tf.constant(cls_t))
        reg_list.append(tf.constant(reg_t))
        cness_list.append(tf.constant(cness_t))
        pos_list.append(tf.constant(pos_t))

    return {
        'cls':      cls_list,
        'reg':      reg_list,
        'cness':    cness_list,
        'pos_mask': pos_list,
    }


# ──────────────────────── DetectionModel subclass ────────────────────────────

class FCOS(DetectionModel):
    """FCOS detector implementing the DetectionModel interface."""

    model_type    = 'fcos'
    target_format = 'raw'

    def __init__(self):
        self._loss_fn = FCOSLoss()

    # ── build ─────────────────────────────────────────────────────────────────

    def build(self, num_classes: int = NUM_CLASSES_WITH_BG,
              width: float = 1.0) -> Model:
        """
        Returns a Keras model.  Outputs are a list of 5 dicts
        (one per FPN level P3–P7):
            {'cls': [B,H,W,C], 'reg': [B,H,W,4], 'cness': [B,H,W,1]}

        Because Keras cannot output Python dicts from a Model, the actual
        outputs are flat tensors concatenated in a fixed order and split
        back during loss / postprocess.
        Output order: cls_P3, reg_P3, cness_P3, cls_P4, … (15 tensors total)
        """
        C = num_classes - 1    # no background class
        f = FCOS_FPN_CHANNELS

        inp = layers.Input(shape=(300, 300, 3), name='input_image')
        C3, C4, C5 = build_resnet50_features(inp, prefix='fcos_rn')
        fpn_levels  = build_fpn(C3, C4, C5, out_channels=f)

        # Shared head towers (weights shared across FPN levels)
        cls_tower = _build_fcos_head(f, FCOS_NUM_CONVS, name='fcos_cls_tower')
        reg_tower = _build_fcos_head(f, FCOS_NUM_CONVS, name='fcos_reg_tower')

        # Final prediction convolutions (NOT shared — separate per level)
        cls_pred_convs   = [layers.Conv2D(C, 1, name=f'cls_pred_{i}')
                            for i in range(5)]
        reg_pred_convs   = [layers.Conv2D(4, 1, name=f'reg_pred_{i}')
                            for i in range(5)]
        cness_pred_convs = [layers.Conv2D(1, 1, name=f'cness_pred_{i}')
                            for i in range(5)]

        # Learnable scale per level for regression (exp(si) * pred)
        scales = [tf.Variable(1.0, trainable=True, name=f'scale_{i}',
                              dtype=tf.float32)
                  for i in range(5)]

        outputs = []
        for i, feat in enumerate(fpn_levels):
            cls_feat   = cls_tower(feat)
            reg_feat   = reg_tower(feat)

            cls_out    = cls_pred_convs[i](cls_feat)              # [B,H,W,C]
            reg_out    = reg_pred_convs[i](reg_feat) * scales[i]  # [B,H,W,4]
            cness_out  = cness_pred_convs[i](
                reg_feat if FCOS_CENTERNESS_ON_REG else cls_feat)  # [B,H,W,1]

            outputs += [cls_out, reg_out, cness_out]

        return Model(inputs=inp, outputs=outputs, name='FCOS')

    # ── encode_targets ────────────────────────────────────────────────────────

    def encode_targets(self, gt_boxes, gt_labels, num_valid) -> dict:
        return _encode_fcos_targets(gt_boxes, gt_labels, num_valid)

    # ── compute_loss ──────────────────────────────────────────────────────────

    def compute_loss(self, predictions, targets: dict) -> tuple:
        # Unpack flat output list → list of per-level dicts
        level_preds = []
        for i in range(5):
            level_preds.append({
                'cls':   predictions[i * 3],
                'reg':   predictions[i * 3 + 1],
                'cness': predictions[i * 3 + 2],
            })
        return self._loss_fn(level_preds, targets)

    # ── postprocess ───────────────────────────────────────────────────────────

    def postprocess(self, predictions,
                    conf_threshold: float = 0.05,
                    nms_iou: float = 0.45,
                    max_dets: int = 200) -> tuple:
        """Decode FCOS predictions for one image."""
        strides = FCOS_STRIDES
        all_boxes, all_scores, all_labels = [], [], []

        for i in range(5):
            cls_logits = predictions[i * 3]       # [H, W, C]
            reg_pred   = predictions[i * 3 + 1]   # [H, W, 4]  log-distances
            cness_logit = predictions[i * 3 + 2]  # [H, W, 1]

            H, W, C = cls_logits.shape

            cls_scores = tf.sigmoid(cls_logits).numpy()          # [H,W,C]
            cness      = tf.sigmoid(cness_logit).numpy()[..., 0] # [H,W]
            ltrb       = np.exp(reg_pred.numpy())                 # [H,W,4]

            stride = strides[i]
            ys = (np.arange(H) + 0.5) * stride / INPUT_SIZE
            xs = (np.arange(W) + 0.5) * stride / INPUT_SIZE
            grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')  # [H,W]

            x1 = np.clip(grid_x - ltrb[..., 0], 0, 1)
            y1 = np.clip(grid_y - ltrb[..., 1], 0, 1)
            x2 = np.clip(grid_x + ltrb[..., 2], 0, 1)
            y2 = np.clip(grid_y + ltrb[..., 3], 0, 1)

            # Score = sqrt(cls_score * centerness) per paper
            combined = cls_scores * np.sqrt(cness[..., np.newaxis])  # [H,W,C]

            for cls_idx in range(C):
                scores = combined[..., cls_idx].flatten()
                mask   = scores > conf_threshold
                if not mask.any():
                    continue
                bx1 = x1.flatten()[mask]
                by1 = y1.flatten()[mask]
                bx2 = x2.flatten()[mask]
                by2 = y2.flatten()[mask]
                sc  = scores[mask].astype(np.float32)
                fb  = np.stack([bx1, by1, bx2, by2], axis=-1).astype(np.float32)
                keep = tf.image.non_max_suppression(fb, sc, max_dets, nms_iou).numpy()
                all_boxes.extend(fb[keep].tolist())
                all_scores.extend(sc[keep].tolist())
                all_labels.extend([cls_idx + 1] * len(keep))

        if not all_boxes:
            return (np.zeros((0, 4), np.float32),
                    np.zeros(0, np.float32),
                    np.zeros(0, np.int32))
        return (np.array(all_boxes,  np.float32),
                np.array(all_scores, np.float32),
                np.array(all_labels, np.int32))
