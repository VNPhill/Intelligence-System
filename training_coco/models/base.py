"""
models/base.py — DetectionModel interface.

Every architecture implements this interface so train.py and evaluate.py
need no architecture-specific code at all.

Two dataset formats are supported:
    'ssd'   SSD-style pre-encoded anchor targets.
            Dataset yields: (images, loc_targets[B,A,4], cls_targets[B,A])
            Used by: mobilenet_ssd, mobilenetv2_ssd, vgg_ssd, resnet_ssd, retinanet

    'raw'   Raw padded ground-truth boxes. Target encoding happens inside
            compute_loss() so the dataset stays architecture-agnostic.
            Dataset yields: (images, gt_boxes[B,MAX_GT,4], gt_labels[B,MAX_GT],
                             num_valid[B])
            Used by: yolov3, fcos, centernet
"""

import abc
import tensorflow as tf


class DetectionModel(abc.ABC):
    """
    Abstract base class for all detection models.

    Subclasses must set `model_type` (registry key) and `target_format`,
    then implement the four abstract methods below.
    """

    # ── Class-level attributes (set in subclass) ──────────────────────────────
    model_type:    str = ''
    target_format: str = 'raw'     # 'ssd' or 'raw'

    # ── Core interface ────────────────────────────────────────────────────────

    @abc.abstractmethod
    def build(self, num_classes: int, width: float) -> tf.keras.Model:
        """Construct and return the tf.keras.Model."""

    @abc.abstractmethod
    def compute_loss(self, predictions, targets: dict) -> tuple:
        """
        Compute the total loss and its components.

        Args:
            predictions : raw model output  (architecture-specific shape)
            targets     : dict produced by encode_targets() for 'raw' models,
                          or {'loc': loc_t, 'cls': cls_t} for 'ssd' models

        Returns:
            (total_loss, cls_loss, reg_loss) — three scalar tf.Tensors
        """

    @abc.abstractmethod
    def encode_targets(self, gt_boxes, gt_labels, num_valid) -> dict:
        """
        Convert raw padded GT to model-specific training targets.

        Called ONCE per batch before compute_loss().
        Only used when target_format == 'raw'.

        Args:
            gt_boxes  : np.ndarray [B, MAX_GT, 4]  float32, [cx,cy,w,h] normalised
            gt_labels : np.ndarray [B, MAX_GT]     int32,   1-based, 0 = padding
            num_valid : np.ndarray [B]             int32,   real GT count

        Returns:
            dict of np.ndarray / tf.Tensor targets (architecture-specific)
        """

    @abc.abstractmethod
    def postprocess(self,
                    predictions,
                    conf_threshold: float = 0.05,
                    nms_iou:        float = 0.45,
                    max_dets:       int   = 200) -> tuple:
        """
        Decode raw model predictions for a SINGLE image to final detections.

        Args:
            predictions    : slice of model output for one image
            conf_threshold : minimum confidence to keep a detection
            nms_iou        : NMS IoU threshold

        Returns:
            (det_boxes  [K, 4]  float32  [x1,y1,x2,y2] normalised,
             det_scores [K]     float32,
             det_labels [K]     int32    1-based class index)
        """

    # ── Convenience helpers (may be overridden) ───────────────────────────────

    def wrap_ssd_targets(self, loc_t, cls_t) -> dict:
        """Package pre-encoded SSD targets into the unified targets dict."""
        return {'loc': loc_t, 'cls': cls_t}
