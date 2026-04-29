"""
losses/yolo_loss.py — YOLOv3 multi-scale loss.

Reference: Redmon & Farhadi, "YOLOv3: An Incremental Improvement", 2018.

Per detection cell, YOLOv3 predicts for each of 3 anchors:
    (tx, ty)     : centre offset inside the cell  (sigmoid → [0,1])
    (tw, th)     : log-space size offsets relative to anchor prior
    objectness   : confidence that a box contains an object  (sigmoid)
    class scores : per-class probabilities  (sigmoid, NOT softmax)

Loss components:
    box_loss     : MSE on (tx, ty, tw, th) for positive anchors
    obj_loss     : BCE objectness for positives
    noobj_loss   : BCE objectness for true negatives (λ_noobj = 0.5)
    class_loss   : BCE per class for positive anchors
"""

import tensorflow as tf
import numpy as np
from config import (
    YOLO_ANCHORS, YOLO_STRIDES,
    YOLO_LAMBDA_OBJ, YOLO_LAMBDA_NOOBJ,
    YOLO_LAMBDA_CLASS, YOLO_LAMBDA_BOX,
    YOLO_IOU_IGNORE, INPUT_SIZE, NUM_CLASSES,
)


class YOLOv3Loss:
    """Callable loss for YOLOv3 — wraps all three detection scales."""

    def __call__(self,
                 predictions: list,
                 targets: dict) -> tuple:
        """
        Args:
            predictions : list of 3 tensors  [B, H, W, 3, 5+C]
                          from large → small objects  (strides 8, 16, 32)
            targets     : dict with keys 'obj', 'noobj', 'box', 'cls'
                          each a list of 3 scale tensors

        Returns:
            (total_loss, cls_loss, box_loss) — scalar tensors
        """
        total_box = tf.constant(0.0)
        total_cls = tf.constant(0.0)
        total_obj = tf.constant(0.0)

        for scale_idx in range(3):
            pred = predictions[scale_idx]           # [B, H, W, 3, 5+C]

            obj_mask    = targets['obj'][scale_idx]    # [B, H, W, 3]
            noobj_mask  = targets['noobj'][scale_idx]  # [B, H, W, 3]
            box_target  = targets['box'][scale_idx]    # [B, H, W, 3, 4]
            cls_target  = targets['cls'][scale_idx]    # [B, H, W, 3, C]

            tx_pred = pred[..., 0]   # [B, H, W, 3]
            ty_pred = pred[..., 1]
            tw_pred = pred[..., 2]
            th_pred = pred[..., 3]
            obj_pred = pred[..., 4]
            cls_pred = pred[..., 5:]              # [B, H, W, 3, C]

            tx_t = box_target[..., 0]
            ty_t = box_target[..., 1]
            tw_t = box_target[..., 2]
            th_t = box_target[..., 3]

            # ── Box regression (positive anchors only) ───────────────────────
            # MSE on sigmoid(tx/ty) and raw tw/th
            xy_loss = obj_mask * (
                tf.square(tf.sigmoid(tx_pred) - tx_t) +
                tf.square(tf.sigmoid(ty_pred) - ty_t)
            )
            wh_loss = obj_mask * (
                tf.square(tw_pred - tw_t) +
                tf.square(th_pred - th_t)
            )
            box_loss = YOLO_LAMBDA_BOX * tf.reduce_sum(xy_loss + wh_loss)

            # ── Objectness ───────────────────────────────────────────────────
            bce = tf.nn.sigmoid_cross_entropy_with_logits

            obj_loss   = YOLO_LAMBDA_OBJ   * tf.reduce_sum(
                obj_mask  * bce(labels=tf.ones_like(obj_pred),  logits=obj_pred))
            noobj_loss = YOLO_LAMBDA_NOOBJ * tf.reduce_sum(
                noobj_mask * bce(labels=tf.zeros_like(obj_pred), logits=obj_pred))

            # ── Classification (positive anchors, per-class BCE) ─────────────
            cls_loss = YOLO_LAMBDA_CLASS * tf.reduce_sum(
                obj_mask[..., tf.newaxis] *
                bce(labels=cls_target, logits=cls_pred)
            )

            total_box = total_box + box_loss
            total_cls = total_cls + cls_loss + obj_loss + noobj_loss

        total = total_box + total_cls
        return total, total_cls, total_box
