"""
evaluate.py — Unified COCO 2017 val mAP evaluation for all architectures.

Usage:
    python evaluate.py                          # uses config.MODEL_TYPE
    python evaluate.py --model retinanet
    python evaluate.py --model yolov3 --conf 0.25
    python evaluate.py --model centernet --ckpt checkpoints/centernet/best_model.weights.h5

Computes per-class AP and mAP at IoU=0.50 (VOC 2010+ protocol).
Results are saved to  results/<model_type>/map_results.txt .
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import tensorflow as tf

from models  import get_detector, AVAILABLE_MODELS
from dataset import load_coco_annotations, parse_coco_boxes
from config  import (
    DATA_DIR, INPUT_SIZE,
    NUM_CLASSES, NUM_CLASSES_WITH_BG,
    COCO_CLASSES, MODEL_TYPE, MODEL_WIDTH,
)


# ──────────────────────────── CLI ────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate any detection model on COCO 2017 val.')
    p.add_argument('--model',    default=MODEL_TYPE, choices=AVAILABLE_MODELS)
    p.add_argument('--width',    type=float, default=MODEL_WIDTH)
    p.add_argument('--ckpt',     default=None,
                   help='Explicit checkpoint path '
                        '(default: checkpoints/<model>/best_model.weights.h5)')
    p.add_argument('--data_dir', default=DATA_DIR)
    p.add_argument('--iou',      type=float, default=0.5,
                   help='IoU threshold for TP matching (default: 0.50)')
    p.add_argument('--conf',     type=float, default=0.05,
                   help='Confidence threshold before NMS (default: 0.05)')
    p.add_argument('--nms_iou',  type=float, default=0.45)
    return p.parse_args()


# ──────────────────────────── IoU (scalar) ───────────────────────────────────

def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter    = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a   = (a[2] - a[0]) * (a[3] - a[1])
    area_b   = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-10)


# ──────────────────────────── AP Computation ─────────────────────────────────

def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall,    [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    pts = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[pts + 1] - mrec[pts]) * mpre[pts + 1]))


# ──────────────────────────── Main mAP loop ──────────────────────────────────

def compute_map(model,
                detector,
                model_type:     str,
                data_dir:       str   = DATA_DIR,
                iou_threshold:  float = 0.5,
                conf_threshold: float = 0.05,
                nms_iou:        float = 0.45) -> float:
    """
    Run inference on COCO 2017 val, compute per-class AP and mAP.
    Save results to  results/<model_type>/map_results.txt .

    Args:
        model          : loaded tf.keras.Model
        detector       : corresponding DetectionModel instance
        model_type     : string key used for results path
        data_dir       : path to ../data/coco/
        iou_threshold  : TP matching threshold
        conf_threshold : minimum confidence to keep a detection
        nms_iou        : NMS IoU threshold

    Returns:
        mAP scalar
    """
    img_dir  = os.path.join(data_dir, 'val2017')
    ann_json = os.path.join(data_dir, 'annotations', 'instances_val2017.json')

    images_meta, ann_by_img = load_coco_annotations(ann_json)
    image_ids = list(images_meta.keys())

    print(f"\n[Eval] Model      : {model_type}")
    print(f"[Eval] Val images : {len(image_ids)}")
    print(f"[Eval] IoU={iou_threshold}  Conf={conf_threshold}  NMS={nms_iou}\n")

    detections    = defaultdict(list)   # cls_idx → [(score, img_id, box)]
    ground_truths = defaultdict(list)   # cls_idx → [(img_id, box_xyxy)]

    for n, img_id in enumerate(image_ids):
        if (n + 1) % 500 == 0:
            print(f"  [{n + 1}/{len(image_ids)}] …")

        meta     = images_meta[img_id]
        anns     = ann_by_img.get(img_id, [])
        img_path = os.path.join(img_dir, meta['file_name'])

        # ── Preprocess ───────────────────────────────────────────────────────
        try:
            raw = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(raw, channels=3)
        except Exception as exc:
            print(f"  [Eval] Skip {img_id}: {exc}")
            continue

        img_t = tf.image.resize(tf.cast(img, tf.float32),
                                [INPUT_SIZE, INPUT_SIZE])
        img_t = img_t / 127.5 - 1.0
        img_t = img_t[tf.newaxis]                  # [1, 300, 300, 3]

        # ── Inference ────────────────────────────────────────────────────────
        raw_preds = model(img_t, training=False)

        # Slice batch dim off each output tensor
        if isinstance(raw_preds, (list, tuple)):
            preds_one = [p[0] for p in raw_preds]
        else:
            preds_one = raw_preds[0]

        # ── Decode + NMS via detector.postprocess ────────────────────────────
        det_boxes, det_scores, det_labels = detector.postprocess(
            preds_one,
            conf_threshold=conf_threshold,
            nms_iou=nms_iou,
        )

        for box, score, label in zip(det_boxes, det_scores, det_labels):
            detections[int(label)].append((float(score), img_id, box))

        # ── Ground-truth ─────────────────────────────────────────────────────
        gt_boxes, gt_labels = parse_coco_boxes(
            anns, meta['width'], meta['height'])

        for box, label in zip(gt_boxes, gt_labels):
            xyxy = np.array([
                box[0] - box[2] / 2, box[1] - box[3] / 2,
                box[0] + box[2] / 2, box[1] + box[3] / 2,
            ], np.float32)
            ground_truths[int(label)].append((img_id, xyxy))

    # ── Per-class AP ─────────────────────────────────────────────────────────
    aps   = []
    lines = [
        f"Model          : {model_type}",
        f"IoU threshold  : {iou_threshold}",
        f"Conf threshold : {conf_threshold}",
        f"NMS IoU        : {nms_iou}",
        "-" * 48,
    ]

    print()
    for cls_idx in range(1, NUM_CLASSES_WITH_BG):
        cls_name = COCO_CLASSES[cls_idx - 1]
        cls_dets = sorted(detections[cls_idx], key=lambda x: -x[0])
        cls_gts  = ground_truths[cls_idx]

        if not cls_gts:
            msg = f"  {cls_name:<20s}  (no GT, skipped)"
            print(msg);  lines.append(msg.strip())
            continue

        gt_by_img = defaultdict(list)
        for (img_id, box) in cls_gts:
            gt_by_img[img_id].append({'box': box, 'matched': False})

        num_gt = len(cls_gts)
        tp = np.zeros(len(cls_dets), np.float32)
        fp = np.zeros(len(cls_dets), np.float32)

        for d, (score, img_id, det_box) in enumerate(cls_dets):
            gts = gt_by_img.get(img_id, [])
            if not gts:
                fp[d] = 1; continue
            ious     = [_iou_xyxy(det_box, g['box']) for g in gts]
            best_idx = int(np.argmax(ious))
            best_iou = ious[best_idx]
            if best_iou >= iou_threshold and not gts[best_idx]['matched']:
                tp[d] = 1;  gts[best_idx]['matched'] = True
            else:
                fp[d] = 1

        cum_tp    = np.cumsum(tp)
        cum_fp    = np.cumsum(fp)
        recall    = cum_tp / (num_gt + 1e-10)
        precision = cum_tp / (cum_tp + cum_fp + 1e-10)
        ap        = _voc_ap(recall, precision)
        aps.append(ap)

        msg = f"  {cls_name:<20s}  AP={ap:.4f}  ({num_gt} GT)"
        print(msg);  lines.append(msg.strip())

    mAP     = float(np.mean(aps)) if aps else 0.0
    summary = (f"\n  mAP@{iou_threshold:.2f}: {mAP:.4f}  "
               f"({len(aps)}/{NUM_CLASSES} classes)")
    print(summary)
    lines.append(summary.strip())

    # ── Save ─────────────────────────────────────────────────────────────────
    results_dir  = os.path.join('results', model_type)
    results_path = os.path.join(results_dir, 'map_results.txt')
    os.makedirs(results_dir, exist_ok=True)
    with open(results_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"\n[Eval] Results → {results_path}")

    return mAP


# ─────────────────────────── Entry point ─────────────────────────────────────

if __name__ == '__main__':
    args = _parse_args()

    ckpt_path = args.ckpt or os.path.join(
        'checkpoints', args.model, 'best_model.weights.h5')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run  python train.py --model {args.model}  first."
        )

    detector = get_detector(args.model)
    model    = detector.build(num_classes=NUM_CLASSES_WITH_BG,
                              width=args.width)

    print(f"[Eval] Loading weights from {ckpt_path}")
    model.load_weights(ckpt_path)

    compute_map(
        model          = model,
        detector       = detector,
        model_type     = args.model,
        data_dir       = args.data_dir,
        iou_threshold  = args.iou,
        conf_threshold = args.conf,
        nms_iou        = args.nms_iou,
    )
