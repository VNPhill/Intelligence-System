import 'dart:io';
import 'dart:ui' show Size;
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:google_mlkit_object_detection/google_mlkit_object_detection.dart';
import 'package:image/image.dart' as img;
import 'base_detection_service.dart';

export 'base_detection_service.dart' show Detection, BaseDetectionService;

// ── Google ML Kit Object Detection implementation ─────────────────────────────
class MLKitDetectionService extends BaseDetectionService {
  /// Detector used for live-camera frames (stream mode with object tracking).
  ObjectDetector? _streamDetector;

  /// Detector used for static image files (single-image mode).
  ObjectDetector? _imageDetector;

  bool _isReady = false;
  CameraDescription? _currentCamera;

  @override
  bool get isReady => _isReady;

  @override
  Future<void> init() async {
    final streamOptions = ObjectDetectorOptions(
      mode: DetectionMode.stream,
      classifyObjects: true,
      multipleObjects: true,
    );
    final imageOptions = ObjectDetectorOptions(
      mode: DetectionMode.single,
      classifyObjects: true,
      multipleObjects: true,
    );
    _streamDetector = ObjectDetector(options: streamOptions);
    _imageDetector = ObjectDetector(options: imageOptions);
    _isReady = true;
  }

  /// Keep a reference to the current camera so we can compute the correct
  /// sensor rotation when building the [InputImage].
  @override
  void onCameraChanged(CameraDescription camera) {
    _currentCamera = camera;
  }

  // ── Camera frame ──────────────────────────────────────────────────────────
  @override
  Future<List<Detection>> runOnCameraImage(
    CameraImage image, {
    double threshold = 0.5,
  }) async {
    if (!_isReady || _streamDetector == null) return [];
    try {
      final inputImage = _cameraImageToInputImage(image);
      if (inputImage == null) return [];

      final objects = await _streamDetector!.processImage(inputImage);
      return _convertObjects(
        objects,
        imageWidth: image.width.toDouble(),
        imageHeight: image.height.toDouble(),
        threshold: threshold,
      );
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
    if (!_isReady || _imageDetector == null) return [];
    try {
      // Decode once to obtain image dimensions for box normalisation.
      final bytes = await file.readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) return [];

      final inputImage = InputImage.fromFilePath(file.path);
      final objects = await _imageDetector!.processImage(inputImage);
      return _convertObjects(
        objects,
        imageWidth: decoded.width.toDouble(),
        imageHeight: decoded.height.toDouble(),
        threshold: threshold,
      );
    } catch (_) {
      return [];
    }
  }

  // ── Conversion helpers ────────────────────────────────────────────────────

  /// Build an [InputImage] from a [CameraImage] captured by the camera plugin.
  InputImage? _cameraImageToInputImage(CameraImage image) {
    if (_currentCamera == null) return null;

    final rotation = InputImageRotationValue.fromRawValue(
          _currentCamera!.sensorOrientation,
        ) ??
        InputImageRotation.rotation0deg;

    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    if (format == null) return null;

    // Concatenate all plane bytes (works for NV21 & YUV_420_888 on Android).
    final WriteBuffer allBytes = WriteBuffer();
    for (final Plane plane in image.planes) {
      allBytes.putUint8List(plane.bytes);
    }

    return InputImage.fromBytes(
      bytes: allBytes.done().buffer.asUint8List(),
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format,
        bytesPerRow: image.planes[0].bytesPerRow,
      ),
    );
  }

  /// Convert ML Kit [DetectedObject] list to shared [Detection] list.
  List<Detection> _convertObjects(
    List<DetectedObject> objects, {
    required double imageWidth,
    required double imageHeight,
    required double threshold,
  }) {
    final results = <Detection>[];

    for (final obj in objects) {
      // Pick the label with the highest confidence (or synthesise one).
      final topLabel = obj.labels.isNotEmpty
          ? obj.labels.reduce(
              (a, b) => a.confidence > b.confidence ? a : b,
            )
          : null;

      final confidence = topLabel?.confidence ?? 0.5;
      if (confidence < threshold) continue;

      final label = topLabel?.text ?? 'object';

      // Normalise bounding box to [0, 1] as [ymin, xmin, ymax, xmax].
      final box = obj.boundingBox;
      results.add(Detection(
        label: label,
        score: confidence,
        box: [
          (box.top / imageHeight).clamp(0.0, 1.0),
          (box.left / imageWidth).clamp(0.0, 1.0),
          (box.bottom / imageHeight).clamp(0.0, 1.0),
          (box.right / imageWidth).clamp(0.0, 1.0),
        ],
        timestamp: DateTime.now(),
      ));
    }

    // Mirror the TFLite sort: priority objects first, then by confidence.
    results.sort((a, b) {
      final aPr = BaseDetectionService.isPriority(a.label) ? 0 : 1;
      final bPr = BaseDetectionService.isPriority(b.label) ? 0 : 1;
      if (aPr != bPr) return aPr.compareTo(bPr);
      return b.score.compareTo(a.score);
    });
    return results;
  }

  @override
  void dispose() {
    _streamDetector?.close();
    _imageDetector?.close();
  }
}
