import 'dart:io';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'base_detection_service.dart';

export 'base_detection_service.dart' show Detection, BaseDetectionService;

// ── TFLite / self-trained model implementation ────────────────────────────────
class DetectionService extends BaseDetectionService {
  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isReady = false;

  @override
  bool get isReady => _isReady;

  @override
  Future<void> init() async {
    await _loadModel();
    await _loadLabels();
    _isReady = true;
  }

  Future<void> _loadModel() async {
    _interpreter = await Interpreter.fromAsset(
        'assets/models/epoch_460.weights.tflite');
  }

  Future<void> _loadLabels() async {
    final raw =
        await rootBundle.loadString('assets/models/labelmap_voc.txt');
    _labels = raw.split('\n').where((l) => l.trim().isNotEmpty).toList();
  }

  // ── Camera frame ──────────────────────────────────────────────────────────
  @override
  Future<List<Detection>> runOnCameraImage(
    CameraImage image, {
    double threshold = 0.5,
  }) async {
    if (!_isReady || _interpreter == null) return [];
    try {
      final input =
          _yuv420ToRgbFloat(image).reshape([1, 300, 300, 3]);
      return _infer(input, threshold);
    } catch (_) {
      return [];
    }
  }

  // ── Static image file ─────────────────────────────────────────────────────
  @override
  Future<List<Detection>> runOnImageFile(
    File file, {
    double threshold = 0.5,
  }) async {
    if (!_isReady || _interpreter == null) return [];
    try {
      final bytes = await file.readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) return [];

      final resized = img.copyResize(decoded, width: 300, height: 300);
      final rgbRaw = Uint8List(300 * 300 * 3);
      int idx = 0;
      for (int y = 0; y < 300; y++) {
        for (int x = 0; x < 300; x++) {
          final pixel = resized.getPixel(x, y);
          rgbRaw[idx++] = pixel.r.toInt();
          rgbRaw[idx++] = pixel.g.toInt();
          rgbRaw[idx++] = pixel.b.toInt();
        }
      }
      final input = _normalizeToFloat32(rgbRaw).reshape([1, 300, 300, 3]);
      return _infer(input, threshold);
    } catch (_) {
      return [];
    }
  }

  // ── Inference ─────────────────────────────────────────────────────────────
  List<Detection> _infer(dynamic inputTensor, double threshold) {
    var outputBoxes =
        List.filled(1 * 10 * 4, 0.0).reshape([1, 10, 4]);
    var outputClasses = List.filled(1 * 10, 0.0).reshape([1, 10]);
    var outputScores = List.filled(1 * 10, 0.0).reshape([1, 10]);
    var numDetections = List.filled(1, 0.0);

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
        final label = classId + 1 < _labels.length
            ? _labels[classId + 1]
            : 'unknown';
        results.add(Detection(
          label: label,
          score: score,
          box: List<double>.from(outputBoxes[0][i]),
          timestamp: DateTime.now(),
        ));
      }
    }

    results.sort((a, b) {
      final aPriority = BaseDetectionService.isPriority(a.label) ? 0 : 1;
      final bPriority = BaseDetectionService.isPriority(b.label) ? 0 : 1;
      if (aPriority != bPriority) return aPriority.compareTo(bPriority);
      return b.score.compareTo(a.score);
    });
    return results;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  Float32List _normalizeToFloat32(Uint8List bytes) {
    final result = Float32List(bytes.length);
    for (int i = 0; i < bytes.length; i++) {
      result[i] = bytes[i] / 127.5 - 1.0;
    }
    return result;
  }

  Float32List _yuv420ToRgbFloat(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];
    final uvRowStride = uPlane.bytesPerRow;
    final uvPixelStride = uPlane.bytesPerPixel!;
    final rgb = Uint8List(300 * 300 * 3);
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
        rgb[index++] =
            (Y + 1.402 * (V - 128)).round().clamp(0, 255);
        rgb[index++] =
            (Y - 0.344136 * (U - 128) - 0.714136 * (V - 128))
                .round()
                .clamp(0, 255);
        rgb[index++] =
            (Y + 1.772 * (U - 128)).round().clamp(0, 255);
      }
    }
    return _normalizeToFloat32(rgb);
  }

  @override
  void dispose() {
    _interpreter?.close();
  }
}
