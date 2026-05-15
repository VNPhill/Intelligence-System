"""
models/yolov3.py — YOLOv3.

Reference: Redmon & Farhadi, "YOLOv3: An Incremental Improvement", 2018.

Architecture:
    Darknet53 backbone  →  D3 (38×38), D4 (19×19), D5 (10×10)

    Three detection heads (large → small objects):
      Head 2  (D5)              →  10×10 ×  3 × (5 + C)
      Head 1  (D4 + up(D5))    →  19×19 ×  3 × (5 + C)
      Head 0  (D3 + up(D4))    →  38×38 ×  3 × (5 + C)

    Each cell predicts: tx, ty, tw, th, objectness, class_1, …, class_C

Per-prediction:
    bx = σ(tx) + cx
    by = σ(ty) + cy
    bw = pw · exp(tw)
    bh = ph · exp(th)

Output list: [pred_38, pred_19, pred_10]
    pred_S : [B, S, S, 3, 5+C]   (un-activated)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from models.base       import DetectionModel
from models.backbones  import build_darknet53_features
from losses.yolo_loss  import YOLOv3Loss
from config import (
    NUM_CLASSES, NUM_CLASSES_WITH_BG, INPUT_SIZE,
    YOLO_ANCHORS, YOLO_STRIDES, YOLO_IOU_IGNORE,
)


# ─────────────────────────── Building Blocks ─────────────────────────────────

def _cbl(x, filters, kernel_size=3, stride=1, name=''):
    """Conv → BN → LeakyReLU(0.1) block."""
    padding = 'same' if stride == 1 else 'valid'
    if stride > 1:
        x = layers.ZeroPadding2D(((1,0),(1,0)), name=f'{name}_pad')(x)
    x = layers.Conv2D(filters, kernel_size, strides=stride,
                      padding=padding, use_bias=False, name=f'{name}_c')(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    return layers.LeakyReLU(0.1, name=f'{name}_lr')(x)


def _detection_block(x, filters, name=''):
    """5 × (1×1 / 3×3 alternating) CBL block — precedes the detection conv."""
    x = _cbl(x, filters,   kernel_size=1, name=f'{name}_a1')
    x = _cbl(x, filters*2, kernel_size=3, name=f'{name}_a2')
    x = _cbl(x, filters,   kernel_size=1, name=f'{name}_a3')
    x = _cbl(x, filters*2, kernel_size=3, name=f'{name}_a4')
    x = _cbl(x, filters,   kernel_size=1, name=f'{name}_a5')
    return x


# ─────────────────────────── IoU (numpy) ─────────────────────────────────────

def _iou_wh(anchors_wh: np.ndarray, gt_wh: np.ndarray) -> np.ndarray:
    """
    IoU between anchor boxes and GT boxes centred at the origin.
    anchors_wh: [A, 2],  gt_wh: [2]  → [A]
    """
    inter_w = np.minimum(anchors_wh[:, 0], gt_wh[0])
    inter_h = np.minimum(anchors_wh[:, 1], gt_wh[1])
    inter   = inter_w * inter_h
    union   = anchors_wh[:, 0]*anchors_wh[:, 1] + gt_wh[0]*gt_wh[1] - inter
    return inter / (union + 1e-10)


# ──────────────────────── DetectionModel subclass ────────────────────────────

class YOLOv3(DetectionModel):
    """YOLOv3 detector implementing the DetectionModel interface."""

    model_type    = 'yolov3'
    target_format = 'raw'

    def __init__(self):
        self._loss_fn = YOLOv3Loss()
        # Flat anchor array per scale: [3, 2]
        self._anchors_np = [np.array(a, dtype=np.float32)
                            for a in YOLO_ANCHORS]   # pixels

    # ── build ─────────────────────────────────────────────────────────────────

    def build(self, num_classes: int = NUM_CLASSES_WITH_BG,
              width: float = 1.0) -> Model:
        C = num_classes - 1       # YOLOv3 uses per-class sigmoid (no BG class)
        out_ch = 3 * (5 + C)

        inp = layers.Input(shape=(300, 300, 3), name='input_image')
        D3, D4, D5 = build_darknet53_features(inp)

        # ── Head 2: D5 → pred_10 ─────────────────────────────────────────────
        x2     = _detection_block(D5, 512, name='head2')
        route2 = x2                                    # saved for upsample path
        out2   = _cbl(x2, 1024, kernel_size=3, name='head2_out_cbl')
        pred_10 = layers.Conv2D(out_ch, 1, use_bias=True,
                                name='pred_10')(out2)  # [B,10,10, 3*(5+C)]

        # ── Head 1: concat(up(D5), D4) → pred_19 ────────────────────────────
        route2_up = _cbl(route2, 256, kernel_size=1, name='up2_cbl')
        # route2_up = layers.UpSampling2D(2, name='up2_us')(route2_up)
        route2_up = layers.Lambda(
            lambda x: tf.image.resize(x[0], tf.shape(x[1])[1:3], method='nearest'),
            name='resize_up2'
        )([route2_up, D4])
        x1     = layers.Concatenate(name='concat1')([route2_up, D4])
        x1     = _detection_block(x1, 256, name='head1')
        route1 = x1
        out1   = _cbl(x1, 512, kernel_size=3, name='head1_out_cbl')
        pred_19 = layers.Conv2D(out_ch, 1, use_bias=True,
                                name='pred_19')(out1)  # [B,19,19, 3*(5+C)]

        # ── Head 0: concat(up(head1), D3) → pred_38 ─────────────────────────
        route1_up = _cbl(route1, 128, kernel_size=1, name='up1_cbl')
        # route1_up = layers.UpSampling2D(2, name='up1_us')(route1_up)
        route1_up = layers.Lambda(
            lambda x: tf.image.resize(x[0], tf.shape(x[1])[1:3], method='nearest'),
            name='resize_up1'
        )([route1_up, D3])
        x0     = layers.Concatenate(name='concat0')([route1_up, D3])
        x0     = _detection_block(x0, 128, name='head0')
        out0   = _cbl(x0, 256, kernel_size=3, name='head0_out_cbl')
        pred_38 = layers.Conv2D(out_ch, 1, use_bias=True,
                                name='pred_38')(out0)  # [B,38,38, 3*(5+C)]

        # Reshape to [B, S, S, 3, 5+C]  (easier to index in loss/postprocess)
        # def _reshape(t, s):
        #     return layers.Reshape((s, s, 3, 5+C), name=f'reshape_{s}')(t)
        def _reshape(t, name):
            return layers.Lambda(
                lambda x: tf.reshape(
                    x,
                    (tf.shape(x)[0],
                    tf.shape(x)[1],
                    tf.shape(x)[2],
                    3,
                    5 + C)
                ),
                name=name
            )(t)

        # out38 = _reshape(pred_38, 38)
        # out19 = _reshape(pred_19, 19)
        # out10 = _reshape(pred_10, 10)
        out38 = _reshape(pred_38, 'reshape_38')
        out19 = _reshape(pred_19, 'reshape_19')
        out10 = _reshape(pred_10, 'reshape_10')

        # Return as list via a Lambda to keep Keras happy
        return Model(inputs=inp, outputs=[out38, out19, out10],
                     name='YOLOv3')

    # ── encode_targets ────────────────────────────────────────────────────────

    def encode_targets(self, gt_boxes, gt_labels, num_valid) -> dict:
        """
        Build per-scale targets for the full batch.

        Returns a dict of lists (one per scale):
            'obj'   : [B, S, S, 3]        float32  objectness target
            'noobj' : [B, S, S, 3]        float32  no-object mask
            'box'   : [B, S, S, 3, 4]     float32  (tx, ty, tw, th) targets
            'cls'   : [B, S, S, 3, C]     float32  one-hot class targets
        """
        B  = gt_boxes.shape[0]
        C  = NUM_CLASSES
        sizes   = [38, 19, 10]
        strides = YOLO_STRIDES          # [8, 16, 32]

        # All anchors flattened: [9, 2] in pixels
        all_anchors_px = np.concatenate(self._anchors_np, axis=0)

        # Initialise containers
        obj_t   = [np.zeros((B, s, s, 3), np.float32) for s in sizes]
        noobj_t = [np.ones( (B, s, s, 3), np.float32) for s in sizes]
        box_t   = [np.zeros((B, s, s, 3, 4), np.float32) for s in sizes]
        cls_t   = [np.zeros((B, s, s, 3, C), np.float32) for s in sizes]

        for b in range(B):
            n = int(num_valid[b])
            for k in range(n):
                cx_n, cy_n, w_n, h_n = gt_boxes[b, k]
                label = int(gt_labels[b, k]) - 1  # 0-based

                # GT width/height in pixels
                gt_wh_px = np.array([w_n * INPUT_SIZE,
                                     h_n * INPUT_SIZE], dtype=np.float32)

                # Find best anchor (ignoring position, just shape match)
                ious      = _iou_wh(all_anchors_px, gt_wh_px)
                best_anc  = int(np.argmax(ious))
                scale_idx = best_anc // 3
                anc_idx   = best_anc  % 3

                s  = sizes[scale_idx]
                st = strides[scale_idx]

                # Cell coordinates
                gx = cx_n * s   # float grid x
                gy = cy_n * s   # float grid y
                gi = int(gx)    # cell col
                gj = int(gy)    # cell row
                gi = min(gi, s - 1)
                gj = min(gj, s - 1)

                # Offset targets
                tx = gx - gi                     # fractional part
                ty = gy - gj
                tw = np.log(max(w_n * INPUT_SIZE /
                                self._anchors_np[scale_idx][anc_idx, 0], 1e-10))
                th = np.log(max(h_n * INPUT_SIZE /
                                self._anchors_np[scale_idx][anc_idx, 1], 1e-10))

                obj_t  [scale_idx][b, gj, gi, anc_idx]      = 1.0
                noobj_t[scale_idx][b, gj, gi, anc_idx]      = 0.0
                box_t  [scale_idx][b, gj, gi, anc_idx]      = [tx, ty, tw, th]
                cls_t  [scale_idx][b, gj, gi, anc_idx, label] = 1.0

                # Mark ignore: anchors with IoU > threshold but not best
                for sc in range(3):
                    for ai in range(3):
                        global_anc = sc * 3 + ai
                        if global_anc == best_anc:
                            continue
                        if ious[global_anc] > YOLO_IOU_IGNORE:
                            sz = sizes[sc]
                            gi2 = min(int(cx_n * sz), sz - 1)
                            gj2 = min(int(cy_n * sz), sz - 1)
                            noobj_t[sc][b, gj2, gi2, ai] = 0.0

        return {
            'obj':   [tf.constant(x) for x in obj_t],
            'noobj': [tf.constant(x) for x in noobj_t],
            'box':   [tf.constant(x) for x in box_t],
            'cls':   [tf.constant(x) for x in cls_t],
        }

    # ── compute_loss ──────────────────────────────────────────────────────────

    def compute_loss(self, predictions, targets: dict) -> tuple:
        return self._loss_fn(predictions, targets)

    # ── postprocess ───────────────────────────────────────────────────────────

    def postprocess(self, predictions,
                    conf_threshold: float = 0.25,
                    nms_iou: float = 0.45,
                    max_dets: int = 200) -> tuple:
        """Decode YOLOv3 multi-scale predictions for one image."""
        sizes   = [38, 19, 10]
        strides = YOLO_STRIDES

        all_boxes, all_scores, all_labels = [], [], []

        for sc, (pred, s, st) in enumerate(zip(predictions, sizes, strides)):
            # pred: [S, S, 3, 5+C]
            anchors_px = self._anchors_np[sc]    # [3, 2]

            # Grid offsets
            grid_x = np.tile(np.arange(s).reshape(1, s, 1), (s, 1, 3))
            grid_y = np.tile(np.arange(s).reshape(s, 1, 1), (1, s, 3))

            tx = tf.sigmoid(pred[..., 0]).numpy()
            ty = tf.sigmoid(pred[..., 1]).numpy()
            tw = pred[..., 2].numpy()
            th = pred[..., 3].numpy()
            obj = tf.sigmoid(pred[..., 4]).numpy()
            cls_p = tf.sigmoid(pred[..., 5:]).numpy()   # [S,S,3,C]

            bx = (tx + grid_x) / s
            by = (ty + grid_y) / s
            bw = anchors_px[:, 0] * np.exp(tw) / INPUT_SIZE
            bh = anchors_px[:, 1] * np.exp(th) / INPUT_SIZE

            scores_raw = obj[..., np.newaxis] * cls_p    # [S,S,3,C]

            # Pick best class per anchor
            best_cls  = np.argmax(scores_raw, axis=-1)   # [S,S,3]
            best_conf = np.max(scores_raw, axis=-1)       # [S,S,3]

            mask = best_conf > conf_threshold
            if not mask.any():
                continue

            bx_m = bx[mask]; by_m = by[mask]
            bw_m = bw[mask]; bh_m = bh[mask]
            sc_m = best_conf[mask]
            lb_m = best_cls[mask]

            x1 = np.clip(bx_m - bw_m/2, 0, 1)
            y1 = np.clip(by_m - bh_m/2, 0, 1)
            x2 = np.clip(bx_m + bw_m/2, 0, 1)
            y2 = np.clip(by_m + bh_m/2, 0, 1)
            boxes_sc = np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)

            for cls_idx in range(NUM_CLASSES):
                cmask = lb_m == cls_idx
                if not cmask.any():
                    continue
                fb = boxes_sc[cmask]
                fs = sc_m[cmask].astype(np.float32)
                keep = tf.image.non_max_suppression(fb, fs, max_dets, nms_iou).numpy()
                all_boxes.extend(fb[keep].tolist())
                all_scores.extend(fs[keep].tolist())
                all_labels.extend([cls_idx + 1] * len(keep))

        if not all_boxes:
            return (np.zeros((0,4), np.float32),
                    np.zeros(0, np.float32),
                    np.zeros(0, np.int32))
        return (np.array(all_boxes,  np.float32),
                np.array(all_scores, np.float32),
                np.array(all_labels, np.int32))
