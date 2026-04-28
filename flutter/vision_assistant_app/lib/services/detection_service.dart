import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class Detection {
  final String label;
  final double score;
  final List<double> box; // [ymin, xmin, ymax, xmax] normalized
  final DateTime timestamp;

  Detection({
    required this.label,
    required this.score,
    required this.box,
    required this.timestamp,
  });

  String get scorePercent => '${(score * 100).toStringAsFixed(0)}%';

  /// Confidence tier for color coding
  String get tier {
    if (score >= 0.85) return 'high';
    if (score >= 0.65) return 'medium';
    return 'low';
  }
}


class DetectionService {
  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isReady = false;

  bool get isReady => _isReady;
  List<String> get labels => _labels;

  // ── Priority object catalog (addresses reviewer comment) ──────────────────
  static const List<String> priorityObjects = [
    'person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle',
    'traffic light', 'stop sign', 'stairs', 'door', 'chair',
    'couch', 'bed', 'dining table', 'toilet', 'potted plant',
    'dog', 'cat', 'bottle', 'cup', 'cell phone',
  ];

  static bool isPriority(String label) =>
      priorityObjects.any((p) => label.toLowerCase().contains(p));

  Future<void> init() async {
    await _loadModel();
    await _loadLabels();
    _isReady = true;
  }

  Future<void> _loadModel() async {
    _interpreter = await Interpreter.fromAsset('assets/models/detect_voc.tflite');
  }

  Future<void> _loadLabels() async {
    final raw = await rootBundle.loadString('assets/models/labelmap_voc.txt');
    _labels = raw.split('\n').where((l) => l.trim().isNotEmpty).toList();
  }

  /// Run inference on a CameraImage (YUV420 → RGB)
  List<Detection> runOnCameraImage(CameraImage image,
      {double threshold = 0.5}) {
    if (!_isReady || _interpreter == null) return [];
    try {
      final input = _yuv420ToRgb(image).reshape([1, 300, 300, 3]);
      return _infer(input, threshold);
    } catch (e) {
      return [];
    }
  }

  /// Helper to convert
  Float32List _normalizeToFloat32(Uint8List bytes) {
    final result = Float32List(bytes.length);
    for (int i = 0; i < bytes.length; i++) {
      result[i] = bytes[i] / 127.5 - 1.0;
    }
    return result;
  }

  /// Run inference on a raw RGB Uint8List (already 300×300×3)
  List<Detection> runOnRgbBytes(Uint8List bytes, {double threshold = 0.5}) {
    if (!_isReady || _interpreter == null) return [];
    try {
      // final input = bytes.reshape([1, 300, 300, 3]);
      final input = _normalizeToFloat32(bytes).reshape([1, 300, 300, 3]);
      return _infer(input, threshold);
    } catch (e) {
      return [];
    }
  }

  List<Detection> _infer(dynamic inputTensor, double threshold) {
    var outputBoxes = List.filled(1 * 10 * 4, 0.0).reshape([1, 10, 4]);
    var outputClasses = List.filled(1 * 10, 0.0).reshape([1, 10]);
    var outputScores = List.filled(1 * 10, 0.0).reshape([1, 10]);
    var numDetections = List.filled(1, 0.0);

    // _interpreter!.runForMultipleInputs(
    //   [inputTensor],
    //   {0: outputBoxes, 1: outputClasses, 2: outputScores, 3: numDetections},
    // );

    final outputs = <int, Object>{
      0: outputBoxes,
      1: outputClasses,
      2: outputScores,
      3: numDetections,
    };
    _interpreter!.runForMultipleInputs([inputTensor], outputs);

    final results = <Detection>[];
    for (int i = 0; i < 10; i++) {
      final score = outputScores[0][i] as double;
      if (score > threshold) {
        final classId = (outputClasses[0][i] as double).toInt();
        final label =
            classId + 1 < _labels.length ? _labels[classId + 1] : 'unknown';
        results.add(Detection(
          label: label,
          score: score,
          box: List<double>.from(outputBoxes[0][i]),
          timestamp: DateTime.now(),
        ));
      }
    }
    // Sort: priority objects first, then by score
    results.sort((a, b) {
      final aPriority = isPriority(a.label) ? 0 : 1;
      final bPriority = isPriority(b.label) ? 0 : 1;
      if (aPriority != bPriority) return aPriority.compareTo(bPriority);
      return b.score.compareTo(a.score);
    });
    return results;
  }

  // ── YUV420 → RGB (300×300) ─────────────────────────────────────────────────
  Uint8List _yuv420ToRgb(CameraImage image) {
    final width = image.width;
    final height = image.height;

    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final uvRowStride = uPlane.bytesPerRow;
    final uvPixelStride = uPlane.bytesPerPixel!;
    final output = Uint8List(300 * 300 * 3);
    int index = 0;

    for (int y = 0; y < 300; y++) {
      for (int x = 0; x < 300; x++) {
        final srcX = (x * width / 300).floor();
        final srcY = (y * height / 300).floor();

        final yIdx = srcY * yPlane.bytesPerRow + srcX;
        final uvIdx =
            (srcY ~/ 2) * uvRowStride + (srcX ~/ 2) * uvPixelStride;

        final Y = yPlane.bytes[yIdx];
        final U = uPlane.bytes[uvIdx];
        final V = vPlane.bytes[uvIdx];

        output[index++] = (Y + 1.402 * (V - 128)).round().clamp(0, 255);
        output[index++] =
            (Y - 0.344136 * (U - 128) - 0.714136 * (V - 128))
                .round()
                .clamp(0, 255);
        output[index++] = (Y + 1.772 * (U - 128)).round().clamp(0, 255);
      }
    }
    return output;
  }

  void dispose() {
    _interpreter?.close();
  }
}


