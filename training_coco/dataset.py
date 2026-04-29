# """
# dataset.py — MS COCO 2017 dataset pipeline.

# Two output formats are supported, selected by the `target_format` argument:

#     'ssd'  Pre-encoded anchor targets (used by all SSD-family models).
#            Yields: (images [B,300,300,3],
#                     loc_targets [B,8732,4],
#                     cls_targets [B,8732])

#     'raw'  Padded raw ground-truth (used by RetinaNet, YOLO, FCOS, CenterNet).
#            Yields: (images [B,300,300,3],
#                     gt_boxes  [B,MAX_GT,4]  cx,cy,w,h normalised,
#                     gt_labels [B,MAX_GT]    int32 1-based (0 = padding),
#                     num_valid [B]           int32 real GT count per image)
# """

# import os
# import json
# from typing import List, Tuple

# import numpy as np
# import tensorflow as tf

# from config import (
#     DATA_DIR, INPUT_SIZE, BATCH_SIZE,
#     COCO_ID_TO_LABEL, NUM_ANCHORS, MAX_GT,
# )
# from anchors import generate_anchors, encode_boxes

# _ANCHORS = generate_anchors()     # [8732, 4]


# # ─────────────────────────── Annotation Loading ──────────────────────────────

# def load_coco_annotations(ann_json: str):
#     """Parse COCO instances JSON.  Returns (images_dict, ann_by_img_dict)."""
#     with open(ann_json) as f:
#         data = json.load(f)

#     images = {img['id']: img for img in data['images']}

#     ann_by_img: dict = {}
#     for ann in data['annotations']:
#         if ann.get('iscrowd', 0):
#             continue
#         iid = ann['image_id']
#         ann_by_img.setdefault(iid, []).append({
#             'bbox':        ann['bbox'],
#             'category_id': ann['category_id'],
#         })
#     return images, ann_by_img


# def parse_coco_boxes(anns: List[dict],
#                      img_w: int,
#                      img_h: int) -> Tuple[np.ndarray, np.ndarray]:
#     """Convert COCO annotation dicts → normalised [cx,cy,w,h] + 1-based labels."""
#     boxes, labels = [], []
#     for ann in anns:
#         cat_id = ann['category_id']
#         if cat_id not in COCO_ID_TO_LABEL:
#             continue
#         x, y, bw, bh = ann['bbox']
#         if bw <= 0 or bh <= 0:
#             continue
#         cx = np.clip((x + bw / 2.0) / img_w, 0, 1)
#         cy = np.clip((y + bh / 2.0) / img_h, 0, 1)
#         w  = np.clip(bw / img_w, 0, 1)
#         h  = np.clip(bh / img_h, 0, 1)
#         boxes.append([cx, cy, w, h])
#         labels.append(COCO_ID_TO_LABEL[cat_id])

#     if not boxes:
#         return (np.zeros((0, 4), np.float32), np.zeros(0, np.int32))
#     return (np.array(boxes,  np.float32), np.array(labels, np.int32))


# # ─────────────────────────── Augmentation ────────────────────────────────────

# def _random_flip(img: np.ndarray, boxes: np.ndarray):
#     if np.random.random() < 0.5:
#         img = img[:, ::-1, :]
#         if len(boxes):
#             boxes = boxes.copy()
#             boxes[:, 0] = 1.0 - boxes[:, 0]
#     return img, boxes


# def _photo_distortion(img: tf.Tensor) -> tf.Tensor:
#     img = tf.image.random_brightness(img, max_delta=32.0 / 255.0)
#     img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
#     img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
#     img = tf.image.random_hue(img, max_delta=0.1)
#     return tf.clip_by_value(img, 0.0, 255.0)


# def _random_crop(img: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
#                  min_iou: float = 0.5):
#     if len(boxes) == 0:
#         return img, boxes, labels
#     h, w = img.shape[:2]
#     for _ in range(50):
#         scale  = np.random.uniform(0.3, 1.0)
#         aspect = np.random.uniform(0.5, 2.0)
#         nh = max(1, min(int(h * scale), h))
#         nw = max(1, min(int(w * scale * aspect), w))
#         top  = np.random.randint(0, h - nh + 1)
#         left = np.random.randint(0, w - nw + 1)
#         cx1, cy1 = left / w,    top / h
#         cx2, cy2 = (left + nw) / w, (top + nh) / h
#         bx1 = boxes[:, 0] - boxes[:, 2] / 2
#         by1 = boxes[:, 1] - boxes[:, 3] / 2
#         bx2 = boxes[:, 0] + boxes[:, 2] / 2
#         by2 = boxes[:, 1] + boxes[:, 3] / 2
#         inter = (np.maximum(0, np.minimum(bx2, cx2) - np.maximum(bx1, cx1)) *
#                  np.maximum(0, np.minimum(by2, cy2) - np.maximum(by1, cy1)))
#         area  = (bx2 - bx1) * (by2 - by1)
#         iou   = inter / (area + 1e-10)
#         if np.all(iou >= min_iou):
#             cropped = img[top:top + nh, left:left + nw]
#             cx = (boxes[:, 0] - cx1) / (cx2 - cx1)
#             cy = (boxes[:, 1] - cy1) / (cy2 - cy1)
#             bw = boxes[:, 2] / (cx2 - cx1)
#             bh = boxes[:, 3] / (cy2 - cy1)
#             new_boxes = np.stack([cx, cy, bw, bh], axis=-1)
#             valid = (cx > 0) & (cx < 1) & (cy > 0) & (cy < 1)
#             if valid.any():
#                 return cropped, new_boxes[valid], labels[valid]
#     return img, boxes, labels


# # ─────────────────────────── Dataset Class ───────────────────────────────────

# class COCODataset:
#     """
#     COCO dataset that supports both target formats.

#     Args:
#         img_dir       : path to image folder (train2017 or val2017)
#         ann_json      : path to instances JSON
#         augment       : whether to apply training augmentation
#         target_format : 'ssd' or 'raw'
#     """

#     def __init__(self, img_dir: str, ann_json: str,
#                  augment: bool = False,
#                  target_format: str = 'ssd'):
#         self.img_dir       = img_dir
#         self.augment       = augment
#         self.target_format = target_format
#         print(f"[Dataset] Loading {ann_json} …")
#         self.images, self.ann_by_img = load_coco_annotations(ann_json)
#         self.image_ids = list(self.images.keys())
#         print(f"[Dataset] {len(self.image_ids)} images  "
#               f"(format='{target_format}')")

#     def _load_raw(self, image_id: int):
#         """Load, augment, and preprocess one image + GT boxes."""
#         meta = self.images[image_id]
#         anns = self.ann_by_img.get(image_id, [])
#         raw  = tf.io.read_file(os.path.join(self.img_dir, meta['file_name']))
#         img  = tf.image.decode_jpeg(raw, channels=3)
#         gt_boxes, gt_labels = parse_coco_boxes(anns, meta['width'], meta['height'])

#         if self.augment:
#             img    = _photo_distortion(tf.cast(img, tf.float32))
#             img_np = img.numpy().astype(np.uint8)
#             img_np, gt_boxes              = _random_flip(img_np, gt_boxes)
#             img_np, gt_boxes, gt_labels   = _random_crop(img_np, gt_boxes, gt_labels)
#             img = tf.constant(img_np, dtype=tf.float32)
#         else:
#             img = tf.cast(img, tf.float32)

#         img = tf.image.resize(img, [INPUT_SIZE, INPUT_SIZE])
#         img = img / 127.5 - 1.0
#         return img, gt_boxes, gt_labels

#     def load_ssd_sample(self, image_id: int):
#         """Return (image, loc_targets, cls_targets) for SSD training."""
#         img, gt_boxes, gt_labels = self._load_raw(image_id)
#         loc_t, cls_t = encode_boxes(gt_boxes, gt_labels, _ANCHORS)
#         return img, loc_t, cls_t

#     def load_raw_sample(self, image_id: int):
#         """Return (image, padded_boxes, padded_labels, num_valid) for anchor-free."""
#         img, gt_boxes, gt_labels = self._load_raw(image_id)
#         n = len(gt_boxes)
#         # Pad / truncate to MAX_GT
#         padded_boxes  = np.zeros((MAX_GT, 4), np.float32)
#         padded_labels = np.zeros((MAX_GT,),   np.int32)
#         actual = min(n, MAX_GT)
#         if actual > 0:
#             padded_boxes[:actual]  = gt_boxes[:actual]
#             padded_labels[:actual] = gt_labels[:actual]
#         return img, padded_boxes, padded_labels, np.int32(actual)

#     def as_tf_dataset(self) -> tf.data.Dataset:
#         if self.target_format == 'ssd':
#             def _gen():
#                 for iid in self.image_ids:
#                     try:
#                         yield self.load_ssd_sample(iid)
#                     except Exception as e:
#                         print(f"[Dataset] skip {iid}: {e}")

#             return tf.data.Dataset.from_generator(
#                 _gen,
#                 output_signature=(
#                     tf.TensorSpec((INPUT_SIZE, INPUT_SIZE, 3), tf.float32),
#                     tf.TensorSpec((NUM_ANCHORS, 4),            tf.float32),
#                     tf.TensorSpec((NUM_ANCHORS,),              tf.int32),
#                 ),
#             )
#         else:   # 'raw'
#             def _gen():
#                 for iid in self.image_ids:
#                     try:
#                         yield self.load_raw_sample(iid)
#                     except Exception as e:
#                         print(f"[Dataset] skip {iid}: {e}")

#             return tf.data.Dataset.from_generator(
#                 _gen,
#                 output_signature=(
#                     tf.TensorSpec((INPUT_SIZE, INPUT_SIZE, 3), tf.float32),
#                     tf.TensorSpec((MAX_GT, 4),                 tf.float32),
#                     tf.TensorSpec((MAX_GT,),                   tf.int32),
#                     tf.TensorSpec((),                          tf.int32),
#                 ),
#             )


# # ─────────────────────────── Public API ──────────────────────────────────────

# def build_dataset(split: str = 'train',
#                   batch_size: int = BATCH_SIZE,
#                   data_dir: str = DATA_DIR,
#                   target_format: str = 'ssd') -> tf.data.Dataset:
#     """
#     Build a batched, prefetched tf.data.Dataset for COCO 2017.

#     Args:
#         split         : 'train' or 'val'
#         batch_size    : samples per batch
#         data_dir      : root path  (must contain train2017/, val2017/,
#                         annotations/)
#         target_format : 'ssd' or 'raw'

#     Returns:
#         tf.data.Dataset
#     """
#     is_train = (split == 'train')
#     img_dir  = os.path.join(data_dir, 'train2017' if is_train else 'val2017')
#     ann_json = os.path.join(
#         data_dir, 'annotations',
#         'instances_train2017.json' if is_train else 'instances_val2017.json',
#     )

#     ds = COCODataset(
#         img_dir, ann_json,
#         augment=is_train,
#         target_format=target_format,
#     ).as_tf_dataset()

#     if is_train:
#         ds = ds.shuffle(buffer_size=4096, reshuffle_each_iteration=True)

#     return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


"""
dataset.py — MS COCO 2017 dataset pipeline.

Only images containing at least one annotation from ACTIVE_CLASSES are loaded.
Labels are remapped to the 1-based active-subset index defined in config.py.

Two output formats are supported, selected by the `target_format` argument:

    'ssd'  Pre-encoded anchor targets (used by all SSD-family models).
           Yields: (images [B,300,300,3],
                    loc_targets [B,8732,4],
                    cls_targets [B,8732])

    'raw'  Padded raw ground-truth (used by RetinaNet, YOLO, FCOS, CenterNet).
           Yields: (images [B,300,300,3],
                    gt_boxes  [B,MAX_GT,4]  cx,cy,w,h normalised,
                    gt_labels [B,MAX_GT]    int32 1-based (0 = padding),
                    num_valid [B]           int32 real GT count per image)
"""

import os
import json
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from config import (
    DATA_DIR, INPUT_SIZE, BATCH_SIZE,
    COCO_ID_TO_LABEL, NUM_ANCHORS, MAX_GT,
)
from anchors import generate_anchors, encode_boxes

_ANCHORS = generate_anchors()     # [8732, 4]

# Set of cat_ids that map to an active class — used to fast-filter images.
_ACTIVE_CAT_IDS = set(COCO_ID_TO_LABEL.keys())


# ─────────────────────────── Annotation Loading ──────────────────────────────

def load_coco_annotations(ann_json: str):
    """
    Parse COCO instances JSON.

    Returns:
        images_dict   : image_id → image meta dict
        ann_by_img    : image_id → list of annotation dicts
                        (only annotations whose category_id is in ACTIVE_CLASSES)
        active_ids    : sorted list of image_ids that have ≥1 active annotation
    """
    with open(ann_json) as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}

    ann_by_img: dict = {}
    for ann in data['annotations']:
        # Skip crowd annotations and classes outside the active subset
        if ann.get('iscrowd', 0):
            continue
        if ann['category_id'] not in _ACTIVE_CAT_IDS:
            continue
        iid = ann['image_id']
        ann_by_img.setdefault(iid, []).append({
            'bbox':        ann['bbox'],
            'category_id': ann['category_id'],
        })

    # Only keep images that have at least one active annotation
    active_ids = sorted(iid for iid in ann_by_img if iid in images)
    return images, ann_by_img, active_ids


def parse_coco_boxes(anns: List[dict],
                     img_w: int,
                     img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert COCO annotation dicts → normalised [cx,cy,w,h] + active 1-based labels.
    Annotations whose category_id is not in COCO_ID_TO_LABEL are silently skipped
    (they were already filtered during loading, but this is a safety guard).
    """
    boxes, labels = [], []
    for ann in anns:
        cat_id = ann['category_id']
        if cat_id not in COCO_ID_TO_LABEL:
            continue
        x, y, bw, bh = ann['bbox']
        if bw <= 0 or bh <= 0:
            continue
        cx = np.clip((x + bw / 2.0) / img_w, 0, 1)
        cy = np.clip((y + bh / 2.0) / img_h, 0, 1)
        w  = np.clip(bw / img_w, 0, 1)
        h  = np.clip(bh / img_h, 0, 1)
        boxes.append([cx, cy, w, h])
        labels.append(COCO_ID_TO_LABEL[cat_id])   # active 1-based label

    if not boxes:
        return (np.zeros((0, 4), np.float32), np.zeros(0, np.int32))
    return (np.array(boxes,  np.float32), np.array(labels, np.int32))


# ─────────────────────────── Augmentation ────────────────────────────────────

def _random_flip(img: np.ndarray, boxes: np.ndarray):
    if np.random.random() < 0.5:
        img = img[:, ::-1, :]
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, 0] = 1.0 - boxes[:, 0]
    return img, boxes


def _photo_distortion(img: tf.Tensor) -> tf.Tensor:
    img = tf.image.random_brightness(img, max_delta=32.0 / 255.0)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.1)
    return tf.clip_by_value(img, 0.0, 255.0)


def _random_crop(img: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
                 min_iou: float = 0.5):
    if len(boxes) == 0:
        return img, boxes, labels
    h, w = img.shape[:2]
    for _ in range(50):
        scale  = np.random.uniform(0.3, 1.0)
        aspect = np.random.uniform(0.5, 2.0)
        nh = max(1, min(int(h * scale), h))
        nw = max(1, min(int(w * scale * aspect), w))
        top  = np.random.randint(0, h - nh + 1)
        left = np.random.randint(0, w - nw + 1)
        cx1, cy1 = left / w,    top / h
        cx2, cy2 = (left + nw) / w, (top + nh) / h
        bx1 = boxes[:, 0] - boxes[:, 2] / 2
        by1 = boxes[:, 1] - boxes[:, 3] / 2
        bx2 = boxes[:, 0] + boxes[:, 2] / 2
        by2 = boxes[:, 1] + boxes[:, 3] / 2
        inter = (np.maximum(0, np.minimum(bx2, cx2) - np.maximum(bx1, cx1)) *
                 np.maximum(0, np.minimum(by2, cy2) - np.maximum(by1, cy1)))
        area  = (bx2 - bx1) * (by2 - by1)
        iou   = inter / (area + 1e-10)
        if np.all(iou >= min_iou):
            cropped = img[top:top + nh, left:left + nw]
            cx = (boxes[:, 0] - cx1) / (cx2 - cx1)
            cy = (boxes[:, 1] - cy1) / (cy2 - cy1)
            bw = boxes[:, 2] / (cx2 - cx1)
            bh = boxes[:, 3] / (cy2 - cy1)
            new_boxes = np.stack([cx, cy, bw, bh], axis=-1)
            valid = (cx > 0) & (cx < 1) & (cy > 0) & (cy < 1)
            if valid.any():
                return cropped, new_boxes[valid], labels[valid]
    return img, boxes, labels


# ─────────────────────────── Dataset Class ───────────────────────────────────

class COCODataset:
    """
    COCO dataset filtered to ACTIVE_CLASSES only.

    Images with zero active annotations are excluded entirely, which
    significantly reduces dataset size for the visually-impaired project.

    Args:
        img_dir       : path to image folder (train2017 or val2017)
        ann_json      : path to instances JSON
        augment       : whether to apply training augmentation
        target_format : 'ssd' or 'raw'
    """

    def __init__(self, img_dir: str, ann_json: str,
                 augment: bool = False,
                 target_format: str = 'ssd'):
        self.img_dir       = img_dir
        self.augment       = augment
        self.target_format = target_format
        print(f"[Dataset] Loading {ann_json} …")
        self.images, self.ann_by_img, self.image_ids = load_coco_annotations(ann_json)
        print(f"[Dataset] {len(self.image_ids)} images with active-class annotations "
              f"(format='{target_format}', {len(_ACTIVE_CAT_IDS)} active cat_ids)")

    def _load_raw(self, image_id: int):
        """Load, augment, and preprocess one image + GT boxes."""
        meta = self.images[image_id]
        anns = self.ann_by_img.get(image_id, [])
        raw  = tf.io.read_file(os.path.join(self.img_dir, meta['file_name']))
        img  = tf.image.decode_jpeg(raw, channels=3)
        gt_boxes, gt_labels = parse_coco_boxes(anns, meta['width'], meta['height'])

        if self.augment:
            img    = _photo_distortion(tf.cast(img, tf.float32))
            img_np = img.numpy().astype(np.uint8)
            img_np, gt_boxes              = _random_flip(img_np, gt_boxes)
            img_np, gt_boxes, gt_labels   = _random_crop(img_np, gt_boxes, gt_labels)
            img = tf.constant(img_np, dtype=tf.float32)
        else:
            img = tf.cast(img, tf.float32)

        img = tf.image.resize(img, [INPUT_SIZE, INPUT_SIZE])
        img = img / 127.5 - 1.0
        return img, gt_boxes, gt_labels

    def load_ssd_sample(self, image_id: int):
        """Return (image, loc_targets, cls_targets) for SSD training."""
        img, gt_boxes, gt_labels = self._load_raw(image_id)
        loc_t, cls_t = encode_boxes(gt_boxes, gt_labels, _ANCHORS)
        return img, loc_t, cls_t

    def load_raw_sample(self, image_id: int):
        """Return (image, padded_boxes, padded_labels, num_valid) for anchor-free."""
        img, gt_boxes, gt_labels = self._load_raw(image_id)
        n = len(gt_boxes)
        padded_boxes  = np.zeros((MAX_GT, 4), np.float32)
        padded_labels = np.zeros((MAX_GT,),   np.int32)
        actual = min(n, MAX_GT)
        if actual > 0:
            padded_boxes[:actual]  = gt_boxes[:actual]
            padded_labels[:actual] = gt_labels[:actual]
        return img, padded_boxes, padded_labels, np.int32(actual)

    def as_tf_dataset(self) -> tf.data.Dataset:
        if self.target_format == 'ssd':
            def _gen():
                for iid in self.image_ids:
                    try:
                        yield self.load_ssd_sample(iid)
                    except Exception as e:
                        print(f"[Dataset] skip {iid}: {e}")

            return tf.data.Dataset.from_generator(
                _gen,
                output_signature=(
                    tf.TensorSpec((INPUT_SIZE, INPUT_SIZE, 3), tf.float32),
                    tf.TensorSpec((NUM_ANCHORS, 4),            tf.float32),
                    tf.TensorSpec((NUM_ANCHORS,),              tf.int32),
                ),
            )
        else:   # 'raw'
            def _gen():
                for iid in self.image_ids:
                    try:
                        yield self.load_raw_sample(iid)
                    except Exception as e:
                        print(f"[Dataset] skip {iid}: {e}")

            return tf.data.Dataset.from_generator(
                _gen,
                output_signature=(
                    tf.TensorSpec((INPUT_SIZE, INPUT_SIZE, 3), tf.float32),
                    tf.TensorSpec((MAX_GT, 4),                 tf.float32),
                    tf.TensorSpec((MAX_GT,),                   tf.int32),
                    tf.TensorSpec((),                          tf.int32),
                ),
            )


# ─────────────────────────── Public API ──────────────────────────────────────

def build_dataset(split: str = 'train',
                  batch_size: int = BATCH_SIZE,
                  data_dir: str = DATA_DIR,
                  target_format: str = 'ssd') -> tf.data.Dataset:
    """
    Build a batched, prefetched tf.data.Dataset for COCO 2017.

    Args:
        split         : 'train' or 'val'
        batch_size    : samples per batch
        data_dir      : root path  (must contain train2017/, val2017/,
                        annotations/)
        target_format : 'ssd' or 'raw'

    Returns:
        tf.data.Dataset  (filtered to ACTIVE_CLASSES only)
    """
    is_train = (split == 'train')
    img_dir  = os.path.join(data_dir, 'train2017' if is_train else 'val2017')
    ann_json = os.path.join(
        data_dir, 'annotations',
        'instances_train2017.json' if is_train else 'instances_val2017.json',
    )

    ds = COCODataset(
        img_dir, ann_json,
        augment=is_train,
        target_format=target_format,
    ).as_tf_dataset()

    if is_train:
        ds = ds.shuffle(buffer_size=4096, reshuffle_each_iteration=True)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

