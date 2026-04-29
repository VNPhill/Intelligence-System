# MobileNet-SSD Training Pipeline

Custom object detection model trained from scratch, exported to TFLite for Flutter deployment.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step 1 — Download Dataset

**COCO 2017** (~20 GB):
```bash
bash download_dataset.sh
```

Or if testing with **Pascal VOC** first (~2.8 GB):
```bash
# VOC 2007 trainval
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar -P data/
# VOC 2007 test
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar -P data/
# VOC 2012 trainval
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -P data/

tar -xf data/VOCtrainval_06-Nov-2007.tar -C data/
tar -xf data/VOCtest_06-Nov-2007.tar     -C data/
tar -xf data/VOCtrainval_11-May-2012.tar -C data/
```

> ⚠️ If using VOC, also set `NUM_CLASSES_WITH_BG = 21` in `config.py` and use the VOC `labelmap.txt`.

---

## Step 2 — Train

```bash
python train.py
```

- Checkpoints saved to `checkpoints/`
- Best model (lowest val loss) saved as `checkpoints/best_model.weights.h5`
- Periodic saves every 10 epochs: `checkpoints/epoch_010.weights.h5`, etc.
- Monitor training live:
```bash
tensorboard --logdir logs/
# open http://localhost:6006
```

---

## Step 3 — Evaluate (mAP)

```bash
python evaluate.py
```

Prints per-class AP and final `mAP@0.50` on the test set.  
Loads `checkpoints/best_model.weights.h5` by default.

---

## Step 4 — Convert to TFLite

```bash
# Float32 (default)
python convert_tflite.py --checkpoint checkpoints/best_model.weights.h5

# Int8 quantized (smaller, faster on mobile)
python convert_tflite.py --checkpoint checkpoints/best_model.weights.h5 --quantize
```

Output: `outputs/detect.tflite` (or `outputs/detect_quant.tflite`)

The verify step at the end prints the output tensor index mapping — check it matches your Flutter app.

---

## Step 5 — Deploy to Flutter

Copy these two files into your Flutter project:

| File | Flutter destination |
|---|---|
| `outputs/detect.tflite` | `assets/models/detect.tflite` |
| `labelmap.txt` | `assets/models/labelmap.txt` |

Only `services/detection_service.dart` needs to be updated (see `detect_service_updated.dart`).

---

## File Overview

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters, class list, paths |
| `model.py` | MobileNetV1-SSD architecture |
| `anchors.py` | 8732 anchor generation, encode/decode |
| `loss.py` | Smooth L1 + Hard Negative Mining loss |
| `dataset.py` | COCO/VOC data loader and augmentation |
| `train.py` | Training loop |
| `evaluate.py` | mAP evaluation |
| `convert_tflite.py` | TFLite export with post-processing |
| `labelmap.txt` | Class names for the Flutter app |
| `detect_service_updated.dart` | Updated Flutter detection service |

---

## Switching Between VOC and COCO

| | VOC (testing) | COCO (production) |
|---|---|---|
| `config.py` | Add `NUM_CLASSES_WITH_BG = 21` | Remove that line |
| `labelmap.txt` | 21-line VOC version | 81-line COCO version |
| Dataset | `data/VOCdevkit/` | `data/coco/` |
| Expected mAP | ~0.68–0.72 | higher, more classes |