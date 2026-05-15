import tensorflow as tf

from models import get_detector
from config import MODEL_TYPE, MODEL_WIDTH, NUM_CLASSES_WITH_BG

CKPT_PATH = f'checkpoints/{MODEL_TYPE}/best_model.weights.h5'
TFLITE_PATH = f'outputs/{MODEL_TYPE}.tflite'

MAX_DET = 10  # must match your Dart code

# ─────────────────────────────────────────────

detector = get_detector(MODEL_TYPE)
model = detector.build(num_classes=NUM_CLASSES_WITH_BG, width=MODEL_WIDTH)
model.load_weights(CKPT_PATH)

print(f"[Export] Model: {MODEL_TYPE}")

# ─────────────────────────────────────────────
# Universal wrapper
# ─────────────────────────────────────────────

class ExportModel(tf.Module):
    def __init__(self, model, detector):
        super().__init__()
        self.model = model
        self.detector = detector

    @tf.function(input_signature=[
        tf.TensorSpec([1, 300, 300, 3], tf.float32)
    ])
    def __call__(self, x):
        preds = self.model(x, training=False)

        # Remove batch dim safely
        if isinstance(preds, (list, tuple)):
            preds = [p[0] for p in preds]
        else:
            preds = preds[0]

        # Universal postprocess
        boxes, scores, labels = self.detector.postprocess(
            preds,
            conf_threshold=0.05,
            nms_iou=0.45
        )

        # Ensure tensors
        boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
        scores = tf.convert_to_tensor(scores, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)

        # Limit detections
        num = tf.shape(scores)[0]
        num = tf.minimum(num, MAX_DET)

        boxes = boxes[:num]
        scores = scores[:num]
        labels = labels[:num]

        # Pad safely
        pad = MAX_DET - num

        boxes = tf.pad(boxes, [[0, pad], [0, 0]])
        scores = tf.pad(scores, [[0, pad]])
        labels = tf.pad(labels, [[0, pad]])

        # Convert box format if needed (xyxy → ymin,xmin,ymax,xmax)
        boxes = tf.stack([
            boxes[:,1],  # ymin
            boxes[:,0],  # xmin
            boxes[:,3],  # ymax
            boxes[:,2],  # xmax
        ], axis=-1)

        return (
            tf.expand_dims(boxes, 0),
            tf.expand_dims(labels, 0),
            tf.expand_dims(scores, 0),
            tf.expand_dims(tf.cast(num, tf.float32), 0)
        )

export_model = ExportModel(model, detector)

# ─────────────────────────────────────────────
# Convert
# ─────────────────────────────────────────────

concrete_func = export_model.__call__.get_concrete_function()

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Optional (recommended for mobile):
# converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"[Export] Saved → {TFLITE_PATH}")