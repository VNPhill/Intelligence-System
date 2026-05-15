"""
models/retinanet.py — RetinaNet.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

Architecture:
    ResNet50 backbone  →  C3, C4, C5
    FPN                →  P3 (38×38), P4 (19×19), P5 (10×10),
                          P6 (5×5),   P7 (3×3)
    Class subnet       →  4 × (Conv 3×3 + ReLU) → Conv 3×3 → [B,H,W, A×C]
    Box subnet         →  4 × (Conv 3×3 + ReLU) → Conv 3×3 → [B,H,W, A×4]

    (subnets share weights across all FPN levels)

Output:
    cls_out : [B, total_anchors, num_classes]  raw logits (sigmoid in loss)
    loc_out : [B, total_anchors, 4]            encoded box offsets

Total anchors for 300×300 input:
    P3: 38×38×9  = 12,996
    P4: 19×19×9  =  3,249
    P5: 10×10×9  =    900
    P6:   5×5×9  =    225
    P7:   3×3×9  =     81
                 = 17,451
"""

import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import layers, Model

from models.base              import DetectionModel
from models.backbones         import build_resnet50_features, build_fpn
from losses.focal_loss        import sigmoid_focal_loss
from config import (
    NUM_CLASSES, NUM_CLASSES_WITH_BG, INPUT_SIZE,
    RETINA_FPN_CHANNELS, RETINA_NUM_CONVS,
    RETINA_ANCHOR_SCALES, RETINA_ANCHOR_RATIOS,
    RETINA_ANCHORS_PER_CELL, RETINA_ANCHOR_BASE_SIZES,
    RETINA_IOU_POS, RETINA_IOU_NEG,
    RETINA_FOCAL_ALPHA, RETINA_FOCAL_GAMMA,
    ENCODE_VARIANCES,
)


# ──────────────────────── Anchor Generation ──────────────────────────────────

def _generate_retinanet_anchors() -> np.ndarray:
    """
    Generate all 17,451 RetinaNet anchors for 300×300 input.

    Returns np.ndarray [17451, 4] in [cx, cy, w, h] normalised coords.
    """
    strides    = [8, 16, 32, 64, 128]
    base_sizes = RETINA_ANCHOR_BASE_SIZES     # normalised

    anchors = []
    for stride, base in zip(strides, base_sizes):
        # fmap = INPUT_SIZE // stride
        fmap = math.ceil(INPUT_SIZE / stride)
        for i in range(fmap):
            for j in range(fmap):
                cx = (j + 0.5) / fmap
                cy = (i + 0.5) / fmap
                for scale in RETINA_ANCHOR_SCALES:
                    for ratio in RETINA_ANCHOR_RATIOS:
                        w = base * scale * np.sqrt(ratio)
                        h = base * scale / np.sqrt(ratio)
                        anchors.append([cx, cy, w, h])

    return np.clip(np.array(anchors, dtype=np.float32), 0.0, 1.0)


_RETINANET_ANCHORS = _generate_retinanet_anchors()   # [17451, 4]


# ──────────────────────── Target Encoding ────────────────────────────────────

def _iou_matrix(anchors: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """[A,4] × [N,4] → [A,N] IoU, both in [cx,cy,w,h]."""
    def to_xyxy(b):
        return np.stack([b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2,
                         b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2], axis=1)
    a = to_xyxy(anchors)[:, None, :]   # [A,1,4]
    b = to_xyxy(gt_boxes)[None, :, :]  # [1,N,4]
    ix1 = np.maximum(a[...,0], b[...,0])
    iy1 = np.maximum(a[...,1], b[...,1])
    ix2 = np.minimum(a[...,2], b[...,2])
    iy2 = np.minimum(a[...,3], b[...,3])
    inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
    aa = (a[...,2]-a[...,0]) * (a[...,3]-a[...,1])
    ab = (b[...,2]-b[...,0]) * (b[...,3]-b[...,1])
    return inter / (aa + ab - inter + 1e-10)


def _encode_retina(gt_boxes, gt_labels, anchors,
                   variances=ENCODE_VARIANCES,
                   iou_pos=RETINA_IOU_POS,
                   iou_neg=RETINA_IOU_NEG):
    """
    Assign anchors to GT boxes and encode targets for one image.

    Returns:
        loc_t  : [A, 4]  encoded offsets
        cls_t  : [A, C]  float one-hot (0 background, -1 ignore)
        weights: [A]     1 = valid (pos or neg), 0 = ignore
    """
    A = len(anchors)
    C = NUM_CLASSES

    cls_t   = np.zeros((A, C), dtype=np.float32)
    weights = np.ones(A,      dtype=np.float32)
    loc_t   = np.zeros((A, 4), dtype=np.float32)

    if len(gt_boxes) == 0:
        return loc_t, cls_t, weights

    iou = _iou_matrix(anchors, gt_boxes)        # [A, N]
    best_gt   = np.argmax(iou, axis=1)          # [A]
    best_iou  = iou[np.arange(A), best_gt]      # [A]

    # Guarantee every GT gets its best anchor
    best_anc_per_gt = np.argmax(iou, axis=0)
    for gt_i, anc_i in enumerate(best_anc_per_gt):
        best_gt[anc_i]  = gt_i
        best_iou[anc_i] = 1.0

    neg_mask    = best_iou < iou_neg
    ignore_mask = (best_iou >= iou_neg) & (best_iou < iou_pos)
    pos_mask    = best_iou >= iou_pos

    # Class targets (one-hot, 0-based inside cls_t)
    cls_t[pos_mask, gt_labels[best_gt[pos_mask]] - 1] = 1.0
    weights[ignore_mask] = 0.0   # ignore zone

    # Box targets (SSD-style delta encoding)
    matched_gt = gt_boxes[best_gt]               # [A, 4]
    loc_t[:, 0] = (matched_gt[:,0]-anchors[:,0]) / (variances[0]*anchors[:,2])
    loc_t[:, 1] = (matched_gt[:,1]-anchors[:,1]) / (variances[0]*anchors[:,3])
    loc_t[:, 2] = np.log(np.maximum(matched_gt[:,2]/anchors[:,2], 1e-10)) / variances[1]
    loc_t[:, 3] = np.log(np.maximum(matched_gt[:,3]/anchors[:,3], 1e-10)) / variances[1]
    loc_t[~pos_mask] = 0.0

    return loc_t, cls_t, weights


# ──────────────────────── Model Architecture ─────────────────────────────────

def _build_subnet(in_channels: int,
                  out_channels: int,
                  num_convs: int,
                  prior_prob: float = 0.01,
                  name: str = 'subnet') -> tf.keras.Model:
    """
    Shared subnet (class or box) applied independently at each FPN level.
    Weights are shared across levels — the subnet is a standalone Keras model.
    """
    inp = layers.Input(shape=(None, None, in_channels))
    x   = inp
    for i in range(num_convs):
        x = layers.Conv2D(in_channels, 3, padding='same',
                          kernel_initializer='normal',
                          name=f'{name}_conv{i}')(x)
        x = layers.ReLU(name=f'{name}_relu{i}')(x)
    # Final conv: bias initialised so sigmoid output ≈ prior_prob
    bias_init = -np.log((1 - prior_prob) / prior_prob)
    out = layers.Conv2D(out_channels, 3, padding='same',
                        kernel_initializer='zeros',
                        bias_initializer=tf.keras.initializers.Constant(bias_init),
                        name=f'{name}_out')(x)
    return tf.keras.Model(inputs=inp, outputs=out, name=name)


# ──────────────────────── DetectionModel subclass ────────────────────────────

class RetinaNet(DetectionModel):
    """
    RetinaNet detector wrapping the DetectionModel interface.

    Uses SSD-style anchor encoding (via _encode_retina) so target_format
    is 'ssd' — the training loop calls encode_targets on raw GT and then
    passes the result to compute_loss.
    """

    model_type    = 'retinanet'
    target_format = 'raw'   # encoding happens in encode_targets()

    def __init__(self):
        self._anchors = _RETINANET_ANCHORS      # [17451, 4]
        self._tf_anchors = tf.constant(self._anchors, dtype=tf.float32)

    # ── build ─────────────────────────────────────────────────────────────────

    def build(self, num_classes: int = NUM_CLASSES_WITH_BG,
              width: float = 1.0) -> Model:
        """
        Returns a Keras Model with outputs:
            cls_out  [B, 17451, num_classes-1]   logits  (NO background class)
            loc_out  [B, 17451, 4]               encoded offsets
        """
        C = num_classes - 1   # RetinaNet uses sigmoid (no background class)
        f = RETINA_FPN_CHANNELS
        A = RETINA_ANCHORS_PER_CELL

        inp = layers.Input(shape=(300, 300, 3), name='input_image')
        C3, C4, C5 = build_resnet50_features(inp, prefix='rn')
        fpn_levels  = build_fpn(C3, C4, C5, out_channels=f)   # [P3..P7]

        # Build shared subnets once
        cls_subnet = _build_subnet(f, A * C,  RETINA_NUM_CONVS, name='cls_subnet')
        box_subnet = _build_subnet(f, A * 4,  RETINA_NUM_CONVS, name='box_subnet')

        cls_outs, loc_outs = [], []
        # fmap_sizes = [INPUT_SIZE // s for s in [8, 16, 32, 64, 128]]
        fmap_sizes = [38, 19, 10, 5, 3]

        for lvl, (feat, fmap_sz) in enumerate(zip(fpn_levels, fmap_sizes)):
            n_boxes = fmap_sz * fmap_sz * A
            cls_p   = cls_subnet(feat)              # [B, H, W, A*C]
            loc_p   = box_subnet(feat)              # [B, H, W, A*4]
            cls_outs.append(layers.Reshape((n_boxes, C), name=f'cls_r{lvl}')(cls_p))
            loc_outs.append(layers.Reshape((n_boxes, 4), name=f'loc_r{lvl}')(loc_p))

        cls_out = layers.Concatenate(axis=1, name='cls_out')(cls_outs)
        loc_out = layers.Concatenate(axis=1, name='loc_out')(loc_outs)

        return Model(inputs=inp, outputs=[cls_out, loc_out], name='RetinaNet')

    # ── encode_targets ────────────────────────────────────────────────────────

    def encode_targets(self, gt_boxes, gt_labels, num_valid) -> dict:
        """Encode raw GT into per-anchor targets for the whole batch."""
        B = gt_boxes.shape[0]
        A = len(self._anchors)
        C = NUM_CLASSES

        batch_loc  = np.zeros((B, A, 4), dtype=np.float32)
        batch_cls  = np.zeros((B, A, C), dtype=np.float32)
        batch_wt   = np.ones( (B, A),    dtype=np.float32)

        for b in range(B):
            n = int(num_valid[b])
            if n == 0:
                continue
            boxes  = gt_boxes[b, :n]
            labels = gt_labels[b, :n]
            loc_t, cls_t, wt = _encode_retina(boxes, labels, self._anchors)
            batch_loc[b]  = loc_t
            batch_cls[b]  = cls_t
            batch_wt[b]   = wt

        return {
            'loc':     tf.constant(batch_loc),
            'cls':     tf.constant(batch_cls),
            'weights': tf.constant(batch_wt),
        }

    # ── compute_loss ──────────────────────────────────────────────────────────

    def compute_loss(self, predictions, targets: dict) -> tuple:
        cls_pred, loc_pred = predictions          # [B,A,C], [B,A,4]
        cls_t    = targets['cls']                 # [B,A,C]
        loc_t    = targets['loc']                 # [B,A,4]
        weights  = targets['weights']             # [B,A]

        num_pos = tf.reduce_sum(
            tf.cast(tf.reduce_any(tf.cast(cls_t, tf.bool), axis=-1), tf.float32)
        )
        num_pos = tf.maximum(num_pos, 1.0)

        # ── Focal classification loss ─────────────────────────────────────────
        wt_exp  = weights[..., tf.newaxis]         # [B,A,1]
        cls_loss = sigmoid_focal_loss(
            cls_pred * wt_exp, cls_t * wt_exp,
            alpha=RETINA_FOCAL_ALPHA, gamma=RETINA_FOCAL_GAMMA,
            reduction='sum',
        ) / num_pos

        # ── Smooth L1 regression (positives only) ────────────────────────────
        pos_mask = tf.cast(
            tf.reduce_any(tf.cast(cls_t, tf.bool), axis=-1), tf.float32)  # [B,A]
        diff  = tf.abs(loc_pred - loc_t)
        sl1   = tf.where(diff < 1.0, 0.5 * diff**2, diff - 0.5)
        loc_loss = tf.reduce_sum(
            tf.reduce_sum(sl1, axis=-1) * pos_mask) / num_pos

        total = cls_loss + loc_loss
        return total, cls_loss, loc_loss

    # ── postprocess ───────────────────────────────────────────────────────────

    def postprocess(self, predictions,
                    conf_threshold: float = 0.05,
                    nms_iou: float = 0.45,
                    max_dets: int = 200) -> tuple:
        """Decode one image's predictions → (boxes, scores, labels)."""
        cls_logits, loc_offsets = predictions     # [A,C], [A,4]

        scores_all = tf.sigmoid(cls_logits).numpy()   # [A,C]
        anchors    = self._anchors
        vx, vw     = ENCODE_VARIANCES

        # Decode boxes
        cx = loc_offsets[:, 0] * vx * anchors[:, 2] + anchors[:, 0]
        cy = loc_offsets[:, 1] * vx * anchors[:, 3] + anchors[:, 1]
        w  = np.exp(loc_offsets[:, 2] * vw) * anchors[:, 2]
        h  = np.exp(loc_offsets[:, 3] * vw) * anchors[:, 3]
        boxes_cx = np.clip(np.stack([cx, cy, w, h], axis=-1), 0, 1)
        x1 = boxes_cx[:, 0] - boxes_cx[:, 2] / 2
        y1 = boxes_cx[:, 1] - boxes_cx[:, 3] / 2
        x2 = boxes_cx[:, 0] + boxes_cx[:, 2] / 2
        y2 = boxes_cx[:, 1] + boxes_cx[:, 3] / 2
        boxes_xyxy = np.clip(np.stack([x1, y1, x2, y2], axis=-1), 0, 1).astype(np.float32)

        all_boxes, all_scores, all_labels = [], [], []
        for cls_idx in range(NUM_CLASSES):
            scores = scores_all[:, cls_idx]
            mask   = scores > conf_threshold
            if not mask.any():
                continue
            fb = boxes_xyxy[mask]
            fs = scores[mask].astype(np.float32)
            keep = tf.image.non_max_suppression(fb, fs, max_dets, nms_iou).numpy()
            all_boxes.extend(fb[keep].tolist())
            all_scores.extend(fs[keep].tolist())
            all_labels.extend([cls_idx + 1] * len(keep))

        if not all_boxes:
            return (np.zeros((0,4), np.float32),
                    np.zeros(0, np.float32),
                    np.zeros(0, np.int32))
        import numpy as _np
        return (_np.array(all_boxes,  np.float32),
                _np.array(all_scores, np.float32),
                _np.array(all_labels, np.int32))


import numpy as np   # ensure available at module level
