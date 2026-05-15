"""
train.py — Unified training loop for all detection architectures.

Usage:
    python train.py                                  # uses config.MODEL_TYPE
    python train.py --model retinanet
    python train.py --model yolov3   --batch 8
    python train.py --model fcos     --width 1.0
    python train.py --model centernet --epochs 120
    python train.py --model mobilenet_ssd

All models are trained through the DetectionModel interface:
    detector.build()          → tf.keras.Model
    detector.encode_targets() → targets dict   (raw-format models only)
    detector.compute_loss()   → (total, cls, reg) scalars
"""

import os
import time
import argparse
import tensorflow as tf
import numpy as np

from models  import get_detector, AVAILABLE_MODELS
from dataset import build_dataset
from config  import (
    NUM_CLASSES_WITH_BG,
    NUM_EPOCHS, BATCH_SIZE,
    LR_INIT, LR_STEPS, MOMENTUM, WEIGHT_DECAY,
    MODEL_TYPE, MODEL_WIDTH,
)


# ──────────────────────────── CLI ────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Train any detection model on MS COCO 2017.')
    p.add_argument('--model',  default=MODEL_TYPE,  choices=AVAILABLE_MODELS)
    p.add_argument('--width',  type=float, default=MODEL_WIDTH,
                   help='Width multiplier (default: %(default)s)')
    p.add_argument('--epochs', type=int,   default=NUM_EPOCHS)
    p.add_argument('--batch',  type=int,   default=BATCH_SIZE)
    p.add_argument('--lr',     type=float, default=LR_INIT,
                   help='Initial learning rate (default: %(default)s)')
    p.add_argument('--data_dir', default=None,
                   help='Override data directory')
    return p.parse_args()


# ──────────────────────────── LR Schedule ────────────────────────────────────

def _get_lr(epoch: int, lr_init: float, lr_steps=LR_STEPS) -> float:
    lr = lr_init
    for milestone in lr_steps:
        if epoch >= milestone:
            lr *= 0.1
    return lr


# ──────────────────────────── Paths ──────────────────────────────────────────

def _make_dirs(model_type: str):
    ckpt_dir = os.path.join('checkpoints', model_type)
    log_dir  = os.path.join('logs', model_type)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'val'),   exist_ok=True)
    return ckpt_dir, log_dir


def _find_latest_checkpoint(ckpt_dir: str):
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = sorted([
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith('epoch_') and f.endswith('.weights.h5')
    ])
    return candidates[-1] if candidates else None


# ──────────────────────────── Main ───────────────────────────────────────────

def train(model_type:  str,
          width:       float,
          num_epochs:  int,
          batch_size:  int,
          lr_init:     float,
          data_dir:    str = None):

    from config import DATA_DIR as _DATA_DIR
    data_dir = data_dir or _DATA_DIR

    ckpt_dir, log_dir = _make_dirs(model_type)

    print(f"\n{'='*65}")
    print(f"  Model   : {model_type}  (width={width})")
    print(f"  Epochs  : {num_epochs}   Batch: {batch_size}   LR: {lr_init}")
    print(f"  Ckpts   : {ckpt_dir}")
    print(f"  Logs    : {log_dir}")
    print(f"{'='*65}\n")

    # ── Detector + model ─────────────────────────────────────────────────────
    detector = get_detector(model_type)
    model    = detector.build(num_classes=NUM_CLASSES_WITH_BG, width=width)
    model.summary(line_length=100)
    print(f"\n  Parameters: {model.count_params():,}\n")

    # ── Datasets ─────────────────────────────────────────────────────────────
    fmt      = detector.target_format       # 'ssd' or 'raw'
    train_ds = build_dataset('train', batch_size, data_dir, target_format=fmt)
    val_ds   = build_dataset('val',   batch_size, data_dir, target_format=fmt)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_init, momentum=MOMENTUM, nesterov=False)

    l2_kernels = [v for v in model.trainable_variables if 'kernel' in v.name]

    # ── TensorBoard ──────────────────────────────────────────────────────────
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
    val_writer   = tf.summary.create_file_writer(os.path.join(log_dir, 'val'))

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    resume_ckpt = _find_latest_checkpoint(ckpt_dir)
    if resume_ckpt:
        print(f"[Train] Resuming from {resume_ckpt}")
        model.load_weights(resume_ckpt)
        try:
            start_epoch = int(
                os.path.basename(resume_ckpt).split('_')[1].split('.')[0])
            print(f"[Train] Starting at epoch {start_epoch + 1}")
        except Exception:
            pass

    # ── Training step ────────────────────────────────────────────────────────
    # We cannot @tf.function the full step because anchor-free encode_targets
    # runs in numpy.  We compile only the gradient section.

    @tf.function
    def _apply_gradients(images, targets_tf):
        with tf.GradientTape() as tape:
            preds = model(images, training=True)
            task_loss, cls_l, reg_l = detector.compute_loss(preds, targets_tf)
            l2_loss   = WEIGHT_DECAY * tf.add_n(
                [tf.nn.l2_loss(k) for k in l2_kernels])
            total_loss = task_loss + l2_loss
        grads = tape.gradient(total_loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 10.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, cls_l, reg_l

    @tf.function
    def _val_step(images, targets_tf):
        preds = model(images, training=False)
        total_loss, cls_l, reg_l = detector.compute_loss(preds, targets_tf)
        return total_loss, cls_l, reg_l

    def _prepare_targets(batch, fmt):
        """Convert raw dataset batch into the targets dict expected by the model."""
        if fmt == 'ssd':
            _, loc_t, cls_t = batch
            return detector.wrap_ssd_targets(loc_t, cls_t)
        else:
            _, gt_boxes, gt_labels, num_valid = batch
            # encode_targets runs on CPU/numpy — keep outside tf.function
            return detector.encode_targets(
                gt_boxes.numpy(), gt_labels.numpy(), num_valid.numpy())

    # ── Epoch loop ───────────────────────────────────────────────────────────

    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        ep = epoch + 1
        new_lr = _get_lr(epoch, lr_init)
        optimizer.learning_rate.assign(new_lr)

        # ── Train ─────────────────────────────────────────────────────────
        t_sum = cls_sum = reg_sum = 0.0
        n_steps = 0
        t0 = time.time()

        for step, batch in enumerate(train_ds):
            images  = batch[0]
            targets = _prepare_targets(batch, fmt)
            total_l, cls_l, reg_l = _apply_gradients(images, targets)

            t_sum   += float(total_l)
            cls_sum += float(cls_l)
            reg_sum += float(reg_l)
            n_steps += 1

            if step % 1000 == 0:
                print(f"  Ep {ep:3d}/{NUM_EPOCHS:3d} step {step:5d} | "
                      f"total={total_l:.4f}  cls={cls_l:.4f}  "
                      f"reg={reg_l:.4f}  lr={new_lr:.2e}")

        mean_t   = t_sum   / max(1, n_steps)
        mean_cls = cls_sum / max(1, n_steps)
        mean_reg = reg_sum / max(1, n_steps)
        elapsed  = time.time() - t0

        # ── Validate ──────────────────────────────────────────────────────
        v_sum = 0.0
        n_val = 0
        for batch in val_ds:
            images  = batch[0]
            targets = _prepare_targets(batch, fmt)
            total_l, _, _ = _val_step(images, targets)
            v_sum += float(total_l)
            n_val += 1
        mean_val = v_sum / max(1, n_val)

        print(f"Epoch {ep:3d}/{num_epochs} [{model_type}] | "
              f"train={mean_t:.4f} (cls={mean_cls:.4f} reg={mean_reg:.4f}) | "
              f"val={mean_val:.4f} | {elapsed:.0f}s")

        with train_writer.as_default():
            tf.summary.scalar('loss/total', mean_t,   step=epoch)
            tf.summary.scalar('loss/cls',   mean_cls, step=epoch)
            tf.summary.scalar('loss/reg',   mean_reg, step=epoch)
            tf.summary.scalar('lr',         new_lr,   step=epoch)
        with val_writer.as_default():
            tf.summary.scalar('loss/total', mean_val, step=epoch)

        # ── Checkpoint ────────────────────────────────────────────────────
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_path = os.path.join(ckpt_dir, 'best_model.weights.h5')
            model.save_weights(best_path)
            print(f"  ✓ Best model saved  (val_loss={best_val_loss:.4f})")

        if ep % 10 == 0:
            model.save_weights(
                os.path.join(ckpt_dir, f'epoch_{ep:03d}.weights.h5'))

    print(f"\n[Train] Done — best val loss: {best_val_loss:.4f}")
    return model


# ─────────────────────────── Entry point ─────────────────────────────────────

if __name__ == '__main__':
    args = _parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[Train] {len(gpus)} GPU(s) found.")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    else:
        print("[Train] No GPU — training on CPU.")

    train(
        model_type = args.model,
        width      = args.width,
        num_epochs = args.epochs,
        batch_size = args.batch,
        lr_init    = args.lr,
        data_dir   = args.data_dir,
    )
