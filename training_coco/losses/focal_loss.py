"""
losses/focal_loss.py — Sigmoid Focal Loss.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

Focal loss down-weights the loss for well-classified examples, focusing
training on hard, misclassified ones.  It replaces hard-negative mining.

    FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)

where p_t is the model probability for the ground-truth class.
"""

import tensorflow as tf


def sigmoid_focal_loss(logits:  tf.Tensor,
                       targets: tf.Tensor,
                       alpha:   float = 0.25,
                       gamma:   float = 2.0,
                       reduction: str = 'sum') -> tf.Tensor:
    """
    Compute sigmoid focal loss element-wise and reduce.

    Args:
        logits  : [B, N, C]  raw (un-sigmoided) class predictions
        targets : [B, N, C]  float32 binary targets  (0 or 1 per class)
        alpha   : balancing factor for positive / negative examples
        gamma   : focusing exponent  (0 → standard cross-entropy)
        reduction : 'sum' | 'mean' | 'none'

    Returns:
        Scalar (or un-reduced tensor if reduction='none')
    """
    p    = tf.sigmoid(logits)                            # [B, N, C]
    ce   = tf.nn.sigmoid_cross_entropy_with_logits(
               labels=targets, logits=logits)            # [B, N, C]

    p_t  = targets * p + (1.0 - targets) * (1.0 - p)   # prob of correct class
    a_t  = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    loss = a_t * tf.pow(1.0 - p_t, gamma) * ce          # [B, N, C]

    if reduction == 'sum':
        return tf.reduce_sum(loss)
    if reduction == 'mean':
        return tf.reduce_mean(loss)
    return loss
