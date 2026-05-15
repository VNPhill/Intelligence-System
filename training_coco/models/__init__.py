"""
models/__init__.py — Unified model registry.

Usage:
    from models import get_detector
    detector = get_detector()            # uses MODEL_TYPE from config.py
    detector = get_detector('retinanet') # override at call site
    keras_model = detector.build()

Every entry in the registry returns a DetectionModel instance that
implements:  build()  compute_loss()  encode_targets()  postprocess()

SSD-family models are thin-wrapped in SSDDetector which adapts them to
the same interface as the anchor-free models.
"""

import tensorflow as tf
from config import MODEL_TYPE, MODEL_WIDTH, NUM_CLASSES_WITH_BG

from models.base import DetectionModel

# ── SSD family ────────────────────────────────────────────────────────────────
from models.mobilenet_ssd   import build_mobilenet_ssd
from models.mobilenetv2_ssd import build_mobilenetv2_ssd
from models.vgg_ssd         import build_vgg_ssd
from models.resnet_ssd      import build_resnet_ssd

# ── New architectures ─────────────────────────────────────────────────────────
from models.retinanet  import RetinaNet
from models.yolov3     import YOLOv3
from models.fcos       import FCOS
from models.centernet  import CenterNet

# ── SSD-family loss (needed for the wrapper) ──────────────────────────────────
from loss import SSDLoss as _SSDLoss
import numpy as np
from anchors import generate_anchors, decode_offsets
from config  import ENCODE_VARIANCES, NUM_CLASSES

_SSD_ANCHORS = generate_anchors()     # [8732, 4]


# ─────────────────────────── SSD Wrapper ─────────────────────────────────────

class SSDDetector(DetectionModel):
    """
    Thin wrapper that adapts any SSD-family builder to the DetectionModel
    interface expected by train.py and evaluate.py.

    SSD models use pre-encoded anchor targets ('ssd' format): the dataset
    already yields (loc_targets, cls_targets) so encode_targets() is a
    pass-through.
    """

    target_format = 'ssd'

    def __init__(self, builder_fn, name: str):
        self.model_type  = name
        self._builder_fn = builder_fn
        self._loss_fn    = _SSDLoss(loc_weight=1.0)

    def build(self, num_classes: int = NUM_CLASSES_WITH_BG,
              width: float = MODEL_WIDTH) -> tf.keras.Model:
        return self._builder_fn(num_classes=num_classes, width=width)

    def encode_targets(self, gt_boxes, gt_labels, num_valid) -> dict:
        # Never called for SSD (dataset returns pre-encoded targets)
        raise NotImplementedError("SSD uses pre-encoded dataset targets.")

    def wrap_ssd_targets(self, loc_t, cls_t) -> dict:
        return {'loc': loc_t, 'cls': cls_t}

    def compute_loss(self, predictions, targets: dict) -> tuple:
        cls_pred, loc_pred = predictions
        total, cls_l, loc_l = self._loss_fn(
            cls_pred, loc_pred,
            targets['cls'], targets['loc'],
        )
        return total, cls_l, loc_l

    def postprocess(self, predictions,
                    conf_threshold: float = 0.01,
                    nms_iou: float = 0.45,
                    max_dets: int = 200) -> tuple:
        """Decode one image's SSD predictions."""
        cls_logits, loc_offsets = predictions   # [A,C], [A,4]

        probs        = tf.nn.softmax(cls_logits, axis=-1).numpy()
        boxes_cxcywh = decode_offsets(loc_offsets, _SSD_ANCHORS).numpy()
        x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
        boxes_xyxy = np.clip(
            np.stack([x1, y1, x2, y2], axis=-1), 0, 1
        ).astype(np.float32)

        all_boxes, all_scores, all_labels = [], [], []
        for cls_idx in range(1, NUM_CLASSES_WITH_BG):
            scores = probs[:, cls_idx]
            mask   = scores > conf_threshold
            if not mask.any():
                continue
            fb   = boxes_xyxy[mask]
            fs   = scores[mask].astype(np.float32)
            keep = tf.image.non_max_suppression(
                fb, fs, max_dets, nms_iou).numpy()
            all_boxes.extend(fb[keep].tolist())
            all_scores.extend(fs[keep].tolist())
            all_labels.extend([cls_idx] * len(keep))

        if not all_boxes:
            return (np.zeros((0, 4), np.float32),
                    np.zeros(0, np.float32),
                    np.zeros(0, np.int32))
        return (np.array(all_boxes,  np.float32),
                np.array(all_scores, np.float32),
                np.array(all_labels, np.int32))


# ─────────────────────────── Registry ────────────────────────────────────────

_REGISTRY: dict = {
    # SSD family
    'mobilenet_ssd'   : lambda: SSDDetector(build_mobilenet_ssd,   'mobilenet_ssd'),
    'mobilenetv2_ssd' : lambda: SSDDetector(build_mobilenetv2_ssd, 'mobilenetv2_ssd'),
    'vgg_ssd'         : lambda: SSDDetector(build_vgg_ssd,         'vgg_ssd'),
    'resnet_ssd'      : lambda: SSDDetector(build_resnet_ssd,      'resnet_ssd'),
    # New architectures
    'retinanet'       : RetinaNet,
    'yolov3'          : YOLOv3,
    'fcos'            : FCOS,
    'centernet'       : CenterNet,
}

AVAILABLE_MODELS = list(_REGISTRY.keys())


def get_detector(model_type: str = MODEL_TYPE) -> DetectionModel:
    """
    Instantiate and return the requested DetectionModel.

    Args:
        model_type : registry key  (default: config.MODEL_TYPE)

    Returns:
        DetectionModel instance

    Raises:
        ValueError: if model_type is not registered.
    """
    key = model_type.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{model_type}'. "
            f"Available: {AVAILABLE_MODELS}"
        )
    return _REGISTRY[key]()


def build_model(model_type: str = MODEL_TYPE,
                num_classes: int = NUM_CLASSES_WITH_BG,
                width: float = MODEL_WIDTH) -> tf.keras.Model:
    """
    Convenience: get detector + call build() in one step.
    Backward-compatible with the old API.
    """
    return get_detector(model_type).build(num_classes=num_classes, width=width)
