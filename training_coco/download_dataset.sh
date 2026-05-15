#!/usr/bin/env bash
# download_dataset.sh — Download MS COCO 2017 dataset
#
# Total download: ~20 GB
#   train2017 images : 18.0 GB  (118,287 images)
#   val2017   images :  1.0 GB  (  5,000 images)
#   annotations      :  0.5 GB
#
# Final structure:
#   ../data/coco/
#     annotations/
#       instances_train2017.json
#       instances_val2017.json
#     train2017/   ← JPEG images
#     val2017/     ← JPEG images

set -euo pipefail

DEST="../data/coco"
mkdir -p "$DEST/annotations"

echo "========================================================"
echo " Downloading MS COCO 2017"
echo " Destination: $DEST"
echo "========================================================"

# ── Annotations ───────────────────────────────────────────────────────────────
echo ""
echo "[1/3] Annotations (~241 MB) ..."
wget -c --show-progress \
  "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
  -O "$DEST/annotations_trainval2017.zip"

# ── Val images ────────────────────────────────────────────────────────────────
echo ""
echo "[2/3] Val images (~1 GB) ..."
wget -c --show-progress \
  "http://images.cocodataset.org/zips/val2017.zip" \
  -O "$DEST/val2017.zip"

# ── Train images ──────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Train images (~18 GB) ..."
wget -c --show-progress \
  "http://images.cocodataset.org/zips/train2017.zip" \
  -O "$DEST/train2017.zip"

# ── Extract ───────────────────────────────────────────────────────────────────
echo ""
echo "Extracting annotations ..."
unzip -q "$DEST/annotations_trainval2017.zip" -d "$DEST/"

echo "Extracting val2017 ..."
unzip -q "$DEST/val2017.zip" -d "$DEST/"

echo "Extracting train2017 ..."
unzip -q "$DEST/train2017.zip" -d "$DEST/"

echo ""
echo "Done!  Dataset ready at $DEST/"
echo ""
echo "Verify with:"
echo "  ls ../data/coco/train2017/ | wc -l   # expect 118287"
echo "  ls ../data/coco/val2017/   | wc -l   # expect  5000"
echo "  ls ../data/coco/annotations/         # expect instances_train2017.json etc."
