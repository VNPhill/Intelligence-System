"""
losses/centernet_loss.py — CenterNet loss.

Reference: Zhou et al., "Objects as Points", 2019.

Three components:
    heatmap_loss : Penalty-reduced focal loss on the class heatmap.
                   Down-weights pixels near a GT centre (they are
                   "almost correct") while focusing on hard negatives.
    size_loss    : L1 loss on (log w, log h) at GT centre pixels.
    offset_loss  : L1 loss on sub-pixel centre offset (cx%1, cy%1) at
                   GT centre pixels.  Corrects for the quantisation from
                   stride-8 downsampling.

Only the K pixels that correspond to GT object centres contribute to the
size and offset losses.
"""

import tensorflow as tf
from config import (
    CENTERNET_LAMBDA_HMAP,
    CENTERNET_LAMBDA_SIZE,
    CENTERNET_LAMBDA_OFFSET,
)


# ─────────────────────── Penalty-Reduced Focal Loss ──────────────────────────

def _heatmap_focal_loss(pred_hmap: tf.Tensor,
                        tgt_hmap:  tf.Tensor,
                        alpha: float = 2.0,
                        beta:  float = 4.0) -> tf.Tensor:
    """
    CornerNet / CenterNet penalty-reduced focal loss on a Gaussian heatmap.

        L = −(1−p)^α · log(p)              for  y = 1  (GT centres)
            −(1−y)^β · p^α · log(1−p)      for  y < 1  (near-centre penalty)

    Args:
        pred_hmap : [B, H, W, C]  raw logits
        tgt_hmap  : [B, H, W, C]  Gaussian targets in [0, 1]

    Returns:
        scalar loss
    """
    p   = tf.sigmoid(pred_hmap)
    eps = 1e-6

    pos_mask = tf.cast(tf.equal(tgt_hmap, 1.0), tf.float32)
    neg_mask = 1.0 - pos_mask

    pos_loss = -pos_mask  * tf.pow(1.0 - p, alpha) * tf.math.log(p + eps)
    neg_loss = -neg_mask  * tf.pow(1.0 - tgt_hmap, beta) \
                          * tf.pow(p, alpha) \
                          * tf.math.log(1.0 - p + eps)

    num_pos = tf.reduce_sum(pos_mask) + eps
    return tf.reduce_sum(pos_loss + neg_loss) / num_pos


# ──────────────────────────── CenterNet Loss ─────────────────────────────────

class CenterNetLoss:
    """
    Callable CenterNet loss.

    Predictions and targets share the same spatial layout:
        heatmap  [B, H, W, C]
        size     [B, H, W, 2]   log(w), log(h)
        offset   [B, H, W, 2]   fractional cx offset, cy offset
    """

    def __call__(self,
                 predictions: dict,
                 targets:     dict) -> tuple:
        """
        Args:
            predictions : dict  {'hmap': [B,H,W,C], 'size': [B,H,W,2],
                                  'offset': [B,H,W,2]}
            targets     : dict  {'hmap': [B,H,W,C],  'size': [B,H,W,2],
                                  'offset': [B,H,W,2], 'obj_mask': [B,H,W]}
                          obj_mask = 1 at GT centre pixels, 0 elsewhere

        Returns:
            (total_loss, hmap_loss, size_loss + offset_loss)
        """
        obj_mask = tf.cast(targets['obj_mask'], tf.float32)[..., tf.newaxis]
        # num_obj  = tf.reduce_sum(targets['obj_mask']) + 1e-6
        num_obj = tf.reduce_sum(tf.cast(targets['obj_mask'], tf.float32)) + 1e-6

        # ── Heatmap ───────────────────────────────────────────────────────────
        hmap_loss = CENTERNET_LAMBDA_HMAP * _heatmap_focal_loss(
            predictions['hmap'], targets['hmap'])

        # ── Size (only at GT centres) ─────────────────────────────────────────
        size_loss = CENTERNET_LAMBDA_SIZE * tf.reduce_sum(
            obj_mask * tf.abs(predictions['size'] - targets['size'])
        ) / num_obj

        # ── Offset (only at GT centres) ───────────────────────────────────────
        off_loss = CENTERNET_LAMBDA_OFFSET * tf.reduce_sum(
            obj_mask * tf.abs(predictions['offset'] - targets['offset'])
        ) / num_obj

        reg_loss  = size_loss + off_loss
        total     = hmap_loss + reg_loss
        return total, hmap_loss, reg_loss
