"""
losses/fcos_loss.py — FCOS multi-level loss.

Reference: Tian et al., "FCOS: Fully Convolutional One-Stage Object
           Detection", ICCV 2019.

Three components:
    cls_loss    : sigmoid focal loss on per-point class predictions
    reg_loss    : GIoU loss on (l, t, r, b) regression for positive points
    cness_loss  : binary cross-entropy on centerness for positive points

All losses are normalised by the total number of positive points in the batch.
"""

import tensorflow as tf
from losses.focal_loss import sigmoid_focal_loss
from config import RETINA_FOCAL_ALPHA, RETINA_FOCAL_GAMMA


# ──────────────────────────── GIoU Loss ──────────────────────────────────────

def giou_loss(pred_ltrb: tf.Tensor,
              tgt_ltrb:  tf.Tensor) -> tf.Tensor:
    """
    Generalised IoU loss on (l, t, r, b) box parameterisation.

    Args:
        pred_ltrb, tgt_ltrb : [N, 4]  non-negative distances to box edges

    Returns:
        [N]  element-wise GIoU loss in [0, 2]
    """
    # Predicted box area
    pw = pred_ltrb[:, 0] + pred_ltrb[:, 2]   # l + r
    ph = pred_ltrb[:, 1] + pred_ltrb[:, 3]   # t + b
    p_area = pw * ph + 1e-7

    # Target box area
    tw = tgt_ltrb[:, 0] + tgt_ltrb[:, 2]
    th = tgt_ltrb[:, 1] + tgt_ltrb[:, 3]
    t_area = tw * th + 1e-7

    # Intersection (overlap of two ltrb boxes anchored at the same point)
    inter_w = tf.minimum(pred_ltrb[:, 0], tgt_ltrb[:, 0]) + \
              tf.minimum(pred_ltrb[:, 2], tgt_ltrb[:, 2])
    inter_h = tf.minimum(pred_ltrb[:, 1], tgt_ltrb[:, 1]) + \
              tf.minimum(pred_ltrb[:, 3], tgt_ltrb[:, 3])
    inter_area = tf.maximum(inter_w, 0.0) * tf.maximum(inter_h, 0.0)

    union     = p_area + t_area - inter_area
    iou       = inter_area / (union + 1e-7)

    # Enclosing box area
    enc_w = tf.maximum(pred_ltrb[:, 0], tgt_ltrb[:, 0]) + \
            tf.maximum(pred_ltrb[:, 2], tgt_ltrb[:, 2])
    enc_h = tf.maximum(pred_ltrb[:, 1], tgt_ltrb[:, 1]) + \
            tf.maximum(pred_ltrb[:, 3], tgt_ltrb[:, 3])
    enc_area = enc_w * enc_h + 1e-7

    giou = iou - (enc_area - union) / enc_area
    return 1.0 - giou                        # loss in [0, 2]


# ──────────────────────────── FCOS Loss Class ────────────────────────────────

class FCOSLoss:
    """Callable FCOS loss across all FPN levels."""

    def __init__(self,
                 alpha: float = RETINA_FOCAL_ALPHA,
                 gamma: float = RETINA_FOCAL_GAMMA):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self,
                 predictions: list,
                 targets: dict) -> tuple:
        """
        Args:
            predictions : list of 5 dicts (one per FPN level P3–P7)
                          each dict: {'cls': [B,H,W,C], 'reg': [B,H,W,4],
                                      'cness': [B,H,W,1]}
            targets     : dict with keys 'cls', 'reg', 'cness', 'pos_mask'
                          each a list of 5 level tensors

        Returns:
            (total_loss, cls_loss, reg_loss) — scalar tensors
        """
        num_pos = tf.cast(
            tf.reduce_sum([
                tf.reduce_sum(tf.cast(targets['pos_mask'][i], tf.float32))
                for i in range(5)
            ]), tf.float32)
        num_pos = tf.maximum(num_pos, 1.0)

        total_cls   = tf.constant(0.0)
        total_reg   = tf.constant(0.0)
        total_cness = tf.constant(0.0)

        for lvl in range(5):
            pred_cls   = predictions[lvl]['cls']    # [B, H, W, C]
            pred_reg   = predictions[lvl]['reg']    # [B, H, W, 4]
            pred_cness = predictions[lvl]['cness']  # [B, H, W, 1]

            tgt_cls    = targets['cls'][lvl]        # [B, H, W, C]  one-hot
            tgt_reg    = targets['reg'][lvl]        # [B, H, W, 4]
            tgt_cness  = targets['cness'][lvl]      # [B, H, W, 1]
            pos_mask   = targets['pos_mask'][lvl]   # [B, H, W]  bool

            B, H, W, C = tf.shape(pred_cls)[0], tf.shape(pred_cls)[1], \
                         tf.shape(pred_cls)[2], pred_cls.shape[-1]

            # ── Classification ────────────────────────────────────────────────
            cls_flat = tf.reshape(pred_cls, [-1, C])
            tgt_flat = tf.reshape(tgt_cls,  [-1, C])
            total_cls = total_cls + sigmoid_focal_loss(
                cls_flat, tgt_flat,
                alpha=self.alpha, gamma=self.gamma, reduction='sum',
            ) / num_pos

            # ── Regression (positive points only) ────────────────────────────
            pos_flat = tf.reshape(pos_mask, [-1])                  # [B*H*W]
            reg_flat  = tf.reshape(pred_reg,  [-1, 4])             # [B*H*W, 4]
            tgt_r_flat = tf.reshape(tgt_reg,  [-1, 4])

            pos_reg   = tf.boolean_mask(reg_flat,   pos_flat)      # [P, 4]
            pos_tgt_r = tf.boolean_mask(tgt_r_flat, pos_flat)

            if tf.size(pos_reg) > 0:
                total_reg = total_reg + tf.reduce_sum(
                    giou_loss(tf.exp(pos_reg), pos_tgt_r)) / num_pos

            # ── Centerness (positive points only) ────────────────────────────
            cness_flat = tf.reshape(pred_cness, [-1])
            tgt_c_flat = tf.reshape(tgt_cness,  [-1])
            pos_cness   = tf.boolean_mask(cness_flat, pos_flat)
            pos_tgt_c   = tf.boolean_mask(tgt_c_flat, pos_flat)

            if tf.size(pos_cness) > 0:
                total_cness = total_cness + tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=pos_tgt_c, logits=pos_cness,
                    )) / num_pos

        total = total_cls + total_reg + total_cness
        return total, total_cls, total_reg
