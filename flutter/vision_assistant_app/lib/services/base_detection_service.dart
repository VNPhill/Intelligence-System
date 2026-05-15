import 'dart:io';
import 'package:camera/camera.dart';

// ── Shared Detection model ────────────────────────────────────────────────────
class Detection {
  final String label;
  final double score;

  /// Normalised bounding box [ymin, xmin, ymax, xmax]
  final List<double> box;
  final DateTime timestamp;

  Detection({
    required this.label,
    required this.score,
    required this.box,
    required this.timestamp,
  });

  String get scorePercent => '${(score * 100).toStringAsFixed(0)}%';

  String get tier {
    if (score >= 0.85) return 'high';
    if (score >= 0.65) return 'medium';
    return 'low';
  }
}

// ── Priority catalogue (used by UI overlay) ───────────────────────────────────
abstract class BaseDetectionService {
  bool get isReady;

  Future<void> init();

  /// Run on a live camera frame.
  Future<List<Detection>> runOnCameraImage(
    CameraImage image, {
    double threshold = 0.5,
  });

  /// Run on a static image file (e.g. from gallery).
  Future<List<Detection>> runOnImageFile(
    File file, {
    double threshold = 0.5,
  });

  /// Called by the camera screen whenever the active [CameraDescription]
  /// changes (e.g. switching between front/back).  ML Kit needs this to
  /// determine the sensor rotation; TFLite ignores it.
  // ignore: avoid_returning_null_for_void
  void onCameraChanged(CameraDescription camera) {}

  void dispose();

  // ── Shared priority helpers ────────────────────────────────────────────────
  static const List<String> priorityObjects = [
    'person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle',
    'traffic light', 'stop sign', 'stairs', 'door', 'chair',
    'couch', 'bed', 'dining table', 'toilet', 'potted plant',
    'dog', 'cat', 'bottle', 'cup', 'cell phone',
  ];

  static bool isPriority(String label) =>
      priorityObjects.any((p) => label.toLowerCase().contains(p));
}
