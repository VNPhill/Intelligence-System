# Object Detection — Multi-Architecture Training Framework

Train and evaluate 8 object detection architectures on MS COCO 2017 from a
single codebase.  Switch models with one flag; everything else (data loading,
training loop, checkpointing, evaluation) is handled automatically.

---

## Supported Models

| Key | Architecture | Family | Backbone | Loss |
|---|---|---|---|---|
| `mobilenet_ssd` | MobileNetV1-SSD | Anchor-based | MobileNetV1 | Smooth L1 + Hard-neg CE |
| `mobilenetv2_ssd` | MobileNetV2-SSD | Anchor-based | MobileNetV2 | Smooth L1 + Hard-neg CE |
| `vgg_ssd` | VGG16-SSD | Anchor-based | VGG16 | Smooth L1 + Hard-neg CE |
| `resnet_ssd` | ResNet50-SSD | Anchor-based | ResNet50 | Smooth L1 + Hard-neg CE |
| `retinanet` | RetinaNet | Anchor-based | ResNet50 + FPN | Focal + Smooth L1 |
| `yolov3` | YOLOv3 | Anchor-based (grid) | Darknet53 | BCE obj + BCE cls + MSE box |
| `fcos` | FCOS | Anchor-free | ResNet50 + FPN | Focal + GIoU + Centerness BCE |
| `centernet` | CenterNet | Anchor-free (heatmap) | ResNet50 + Deconv | Penalty-focal + L1 size/offset |

---

## Project Structure

```
.
├── config.py               # All hyperparameters — change MODEL_TYPE here
├── train.py                # Training loop (all models)
├── evaluate.py             # mAP evaluation (all models)
├── dataset.py              # COCO data pipeline ('ssd' and 'raw' formats)
├── anchors.py              # SSD anchor generation + encoding / decoding
├── loss.py                 # SSD Multibox Loss
│
├── losses/
│   ├── focal_loss.py       # Sigmoid focal loss (RetinaNet, FCOS)
│   ├── yolo_loss.py        # YOLOv3 multi-scale loss
│   ├── fcos_loss.py        # FCOS: focal + GIoU + centerness
│   └── centernet_loss.py   # CenterNet: heatmap focal + L1
│
└── models/
    ├── base.py             # DetectionModel interface (ABC)
    ├── ssd_common.py       # Shared SSD extra layers + prediction heads
    ├── mobilenet_ssd.py
    ├── mobilenetv2_ssd.py
    ├── vgg_ssd.py
    ├── resnet_ssd.py
    ├── retinanet.py
    ├── yolov3.py
    ├── fcos.py
    ├── centernet.py
    └── backbones/
        ├── resnet.py       # Shared ResNet50 → C3 / C4 / C5
        ├── darknet53.py    # YOLOv3 backbone → D3 / D4 / D5
        └── fpn.py          # Feature Pyramid Network → P3 … P7
```

---

## Requirements

```bash
pip install tensorflow>=2.12 numpy
```

GPU is strongly recommended. TensorFlow will use it automatically if available.
For multi-GPU training see the note at the bottom of this file.

---

## Dataset Setup

This codebase uses **MS COCO 2017**.  Download and extract into:

```
../data/coco/
  annotations/
    instances_train2017.json
    instances_val2017.json
  train2017/          # 118,287 images
  val2017/            #   5,000 images
```

Download links:
- Images: https://cocodataset.org/#download
- Annotations: https://cocodataset.org/#download  (2017 Train/Val annotations)

Or with wget:
```bash
mkdir -p ../data/coco/annotations
cd ../data/coco

# Images
wget http://images.cocodataset.org/zips/train2017.zip && unzip train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip   && unzip val2017.zip

# Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

---

## Quick Start

### 1 — Pick a model

Edit `config.py`:

```python
MODEL_TYPE  = 'retinanet'   # change this
MODEL_WIDTH = 1.0           # channel-width multiplier
```

Or pass `--model` on the command line (overrides config.py).

### 2 — Train

```bash
# Use whatever MODEL_TYPE is set in config.py
python train.py

# Override model on the command line
python train.py --model retinanet
python train.py --model yolov3
python train.py --model fcos
python train.py --model centernet
python train.py --model mobilenet_ssd
python train.py --model vgg_ssd

# Extra flags
python train.py --model retinanet --epochs 200 --batch 16 --lr 0.01
python train.py --model mobilenet_ssd --width 0.5   # half-width (faster)
```

Training output goes to:
```
checkpoints/<model>/
    best_model.weights.h5       # saved whenever val loss improves
    epoch_010.weights.h5        # periodic saves every 10 epochs
logs/<model>/
    train/                      # TensorBoard event files
    val/
```

### 3 — Monitor with TensorBoard

```bash
tensorboard --logdir logs/
```

Open http://localhost:6006 — all models appear as separate runs.

### 4 — Resume interrupted training

Training resumes automatically from the latest `epoch_NNN.weights.h5`
checkpoint the next time you run the same `--model`.  No extra flags needed.

### 5 — Evaluate

```bash
# Evaluate the best checkpoint for a model
python evaluate.py --model retinanet

# Evaluate a specific checkpoint
python evaluate.py --model yolov3 --ckpt checkpoints/yolov3/epoch_100.weights.h5

# Tune detection thresholds
python evaluate.py --model fcos --conf 0.05 --nms_iou 0.45 --iou 0.50
```

Results are printed to stdout and saved to:
```
results/<model>/map_results.txt
```

---

## All CLI Flags

### `train.py`

| Flag | Default | Description |
|---|---|---|
| `--model` | config.MODEL_TYPE | Model to train |
| `--width` | config.MODEL_WIDTH | Channel-width multiplier |
| `--epochs` | 200 | Number of training epochs |
| `--batch` | 16 | Batch size |
| `--lr` | 0.01 | Initial learning rate |
| `--data_dir` | `../data/coco` | Path to COCO root |

### `evaluate.py`

| Flag | Default | Description |
|---|---|---|
| `--model` | config.MODEL_TYPE | Model to evaluate |
| `--width` | config.MODEL_WIDTH | Must match the width used during training |
| `--ckpt` | `checkpoints/<model>/best_model.weights.h5` | Override checkpoint path |
| `--data_dir` | `../data/coco` | Path to COCO root |
| `--iou` | 0.50 | IoU threshold for TP matching |
| `--conf` | 0.05 | Minimum confidence score |
| `--nms_iou` | 0.45 | NMS IoU threshold |

---

## Configuration Reference (`config.py`)

### Model selection

```python
MODEL_TYPE  = 'mobilenet_ssd'   # pick from the table at the top
MODEL_WIDTH = 1.0               # 0.25 / 0.5 / 0.75 / 1.0
```

### Training hyperparameters

```python
INPUT_SIZE   = 300      # input resolution (all models use 300×300)
BATCH_SIZE   = 16
NUM_EPOCHS   = 200
LR_INIT      = 1e-2     # step-decayed at epochs LR_STEPS
LR_STEPS     = [120, 160]
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4     # L2 regularization on all Conv kernels
```

### Architecture-specific constants

Each architecture has its own section in `config.py`.  The most commonly
tuned ones are:

**RetinaNet**
```python
RETINA_FOCAL_ALPHA = 0.25   # focal loss balance factor
RETINA_FOCAL_GAMMA = 2.0    # focusing exponent (0 = standard CE)
RETINA_IOU_POS     = 0.5    # IoU ≥ this → positive anchor
RETINA_IOU_NEG     = 0.4    # IoU <  this → negative (between = ignore)
```

**YOLOv3**
```python
YOLO_LAMBDA_BOX   = 5.0     # box regression loss weight
YOLO_LAMBDA_OBJ   = 1.0     # objectness loss weight
YOLO_LAMBDA_NOOBJ = 0.5     # no-object loss weight
YOLO_IOU_IGNORE   = 0.5     # anchors above this IoU to any GT are ignored
```

**FCOS**
```python
FCOS_REGRESS_RANGES = [(-1,64),(64,128),(128,256),(256,512),(512,1e8)]
# Controls which FPN level handles which object size
```

**CenterNet**
```python
CENTERNET_LAMBDA_HMAP   = 1.0
CENTERNET_LAMBDA_SIZE   = 0.1
CENTERNET_LAMBDA_OFFSET = 1.0
CENTERNET_MIN_OVERLAP   = 0.7  # controls Gaussian radius
```

---

## Expected Results (approximate, COCO val mAP@0.50)

These are approximate published / reproduced numbers at full training length.
Your results may vary based on hardware, exact augmentation, and epochs run.

| Model | mAP@0.50 | Speed (V100) | Params |
|---|---|---|---|
| `mobilenet_ssd` | ~23 | ~45 ms | ~5 M |
| `mobilenetv2_ssd` | ~25 | ~40 ms | ~4 M |
| `vgg_ssd` | ~43 | ~25 ms | ~26 M |
| `resnet_ssd` | ~46 | ~30 ms | ~36 M |
| `retinanet` | ~55 | ~55 ms | ~37 M |
| `yolov3` | ~55 | ~20 ms | ~62 M |
| `fcos` | ~56 | ~50 ms | ~32 M |
| `centernet` | ~52 | ~15 ms | ~32 M |

---

## Common Issues

**Out of GPU memory**
Reduce batch size: `python train.py --model retinanet --batch 8`
Or use a lighter backbone: `python train.py --model mobilenet_ssd --width 0.5`

**Training is slow on CPU**
Normal — COCO training requires a GPU. A full 200-epoch run can take
several days on CPU. For quick testing, reduce `NUM_EPOCHS` in `config.py`.

**"Checkpoint not found" when evaluating**
You must train the model first:
```bash
python train.py --model <model_name>
python evaluate.py --model <model_name>
```

**Resuming picks up wrong epoch number**
The epoch is parsed from the filename `epoch_NNN.weights.h5`. If you rename
checkpoint files, the parser may default to epoch 0 (training still continues
from the loaded weights, it just resets the epoch counter).

**width mismatch between train and evaluate**
Always pass the same `--width` flag used during training:
```bash
python train.py    --model mobilenet_ssd --width 0.5
python evaluate.py --model mobilenet_ssd --width 0.5
```

---

## Adding a New Model

1. Create `models/my_model.py` with a class that subclasses `DetectionModel`.
2. Implement the four required methods: `build()`, `compute_loss()`,
   `encode_targets()`, `postprocess()`.
3. Set `model_type = 'my_model'` and `target_format = 'raw'` (or `'ssd'`).
4. Register it in `models/__init__.py`:
   ```python
   from models.my_model import MyModel
   _REGISTRY['my_model'] = MyModel
   ```
5. Add any new hyperparameters to `config.py`.

That's it — `train.py` and `evaluate.py` will pick it up automatically.

---

## References

- Liu et al., *SSD: Single Shot MultiBox Detector*, ECCV 2016
- Howard et al., *MobileNets*, arXiv 2017
- Sandler et al., *MobileNetV2*, CVPR 2018
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
- Lin et al., *Feature Pyramid Networks for Object Detection*, CVPR 2017
- Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017
- Redmon & Farhadi, *YOLOv3: An Incremental Improvement*, arXiv 2018
- Tian et al., *FCOS: Fully Convolutional One-Stage Object Detection*, ICCV 2019
- Zhou et al., *Objects as Points*, arXiv 2019


