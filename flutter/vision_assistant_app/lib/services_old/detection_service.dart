import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Data model
// ─────────────────────────────────────────────────────────────────────────────

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

  String get tier {
    if (score >= 0.85) return 'high';
    if (score >= 0.65) return 'medium';
    return 'low';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Payload passed into the background isolate (all data must be sendable)
// ─────────────────────────────────────────────────────────────────────────────

class _InferPayload {
  final Uint8List rgbBytes;      // already 300×300×3 uint8
  final List<String> labels;
  final double threshold;
  // TFLite output index mapping (determined at init time via verify step)
  final int idxBoxes;
  final int idxClasses;
  final int idxScores;
  final int idxNum;
  // Model asset bytes — interpreter must be created inside the isolate
  final Uint8List modelBytes;

  const _InferPayload({
    required this.rgbBytes,
    required this.labels,
    required this.threshold,
    required this.idxBoxes,
    required this.idxClasses,
    required this.idxScores,
    required this.idxNum,
    required this.modelBytes,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level function — runs in a separate isolate via compute()
// ─────────────────────────────────────────────────────────────────────────────

List<Detection> _inferInIsolate(_InferPayload p) {
  // Create a fresh interpreter inside the isolate
  final interpreter = Interpreter.fromBuffer(p.modelBytes);

  // Normalize uint8 → float32 in [-1, 1]
  final inputF32 = Float32List(p.rgbBytes.length);
  for (int i = 0; i < p.rgbBytes.length; i++) {
    inputF32[i] = p.rgbBytes[i] / 127.5 - 1.0;
  }
  final inputTensor = inputF32.reshape([1, 300, 300, 3]);

  // Pre-allocate output buffers
  var outputBoxes     = List.filled(1 * 10 * 4, 0.0).reshape([1, 10, 4]);
  var outputClasses   = List.filled(1 * 10, 0.0).reshape([1, 10]);
  var outputScores    = List.filled(1 * 10, 0.0).reshape([1, 10]);
  var numDetections   = List.filled(1, 0.0);

  final outputs = <int, Object>{
    p.idxBoxes:   outputBoxes,
    p.idxClasses: outputClasses,
    p.idxScores:  outputScores,
    p.idxNum:     numDetections,
  };

  interpreter.runForMultipleInputs([inputTensor], outputs);
  interpreter.close();

  final results = <Detection>[];
  final count   = (numDetections[0] as double).toInt().clamp(0, 10);

  for (int i = 0; i < count; i++) {
    final score = outputScores[0][i] as double;
    if (score < p.threshold) continue;

    final classId = (outputClasses[0][i] as double).toInt();
    final label   = classId < p.labels.length ? p.labels[classId] : 'unknown';

    results.add(Detection(
      label: label,
      score: score,
      box:   List<double>.from(outputBoxes[0][i] as List),
      timestamp: DateTime.now(),
    ));
  }

  // Sort: priority objects first, then by descending score
  results.sort((a, b) {
    final aPri = DetectionService.isPriority(a.label) ? 0 : 1;
    final bPri = DetectionService.isPriority(b.label) ? 0 : 1;
    if (aPri != bPri) return aPri.compareTo(bPri);
    return b.score.compareTo(a.score);
  });

  return results;
}

// ─────────────────────────────────────────────────────────────────────────────
// DetectionService
// ─────────────────────────────────────────────────────────────────────────────

class DetectionService {
  // ── Public state ────────────────────────────────────────────────────────────
  bool get isReady => _isReady;
  List<String> get labels => _labels;

  /// The currently active camera description.
  CameraDescription? get activeCamera => _activeCamera;

  /// The live [CameraController]. Null until [startCamera] is called.
  CameraController? get cameraController => _cameraController;

  // ── Priority catalog ────────────────────────────────────────────────────────
  static const List<String> priorityObjects = [
    'person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle',
    'traffic light', 'stop sign', 'stairs', 'door', 'chair',
    'couch', 'bed', 'dining table', 'toilet', 'potted plant',
    'dog', 'cat', 'bottle', 'cup', 'cell phone',
  ];

  static bool isPriority(String label) =>
      priorityObjects.any((p) => label.toLowerCase().contains(p));

  // ── Private state ────────────────────────────────────────────────────────────
  bool _isReady = false;
  List<String> _labels = [];

  // Model bytes kept in memory so the isolate can create its own interpreter
  Uint8List? _modelBytes;

  // TFLite output index mapping (resolved once at init)
  int _idxBoxes   = 0;
  int _idxClasses = 1;
  int _idxScores  = 2;
  int _idxNum     = 3;

  // Frame-drop guard — prevents queueing more frames than we can process
  bool _isProcessing = false;

  // Camera state
  List<CameraDescription> _availableCameras = [];
  CameraDescription?      _activeCamera;
  CameraController?       _cameraController;
  int                     _activeCameraIndex = 0;

  // Callback fired with new detections (set by caller)
  void Function(List<Detection>)? onDetections;

  // ── Init ─────────────────────────────────────────────────────────────────────

  /// Call once before using the service.
  Future<void> init() async {
    await _loadModelBytes();
    await _loadLabels();
    _resolveOutputIndices();
    _availableCameras = await availableCameras();
    _isReady = true;
  }

  Future<void> _loadModelBytes() async {
    final data = await rootBundle.load('assets/models/epoch_460.tflite');
    _modelBytes = data.buffer.asUint8List();
  }

  Future<void> _loadLabels() async {
    final raw = await rootBundle.loadString('assets/models/labelmap_voc.txt');
    _labels = raw.split('\n').where((l) => l.trim().isNotEmpty).toList();
  }

  /// Spin up a temporary interpreter just to read output tensor metadata,
  /// then map each tensor to the correct semantic slot by shape.
  void _resolveOutputIndices() {
    if (_modelBytes == null) return;
    final interp   = Interpreter.fromBuffer(_modelBytes!);
    final outDetails = interp.getOutputTensors();

    for (int i = 0; i < outDetails.length; i++) {
      final shape = outDetails[i].shape; // e.g. [1,10,4] or [1,10] or [1]
      if (shape.length == 3 && shape[2] == 4) {
        _idxBoxes = i;   // [1, 10, 4] → boxes
      } else if (shape.length == 1) {
        _idxNum = i;     // [1]        → num_detections
      }
      // classes and scores both have shape [1,10]; assign in encountered order
    }

    // Assign classes / scores by elimination
    final taken = {_idxBoxes, _idxNum};
    bool classesAssigned = false;
    for (int i = 0; i < outDetails.length; i++) {
      if (taken.contains(i)) continue;
      if (!classesAssigned) {
        _idxClasses = i;
        classesAssigned = true;
      } else {
        _idxScores = i;
      }
    }

    interp.close();
    debugPrint('[DetectionService] Output mapping → '
        'boxes=$_idxBoxes  classes=$_idxClasses  '
        'scores=$_idxScores  num=$_idxNum');
  }

  // ── Camera management ────────────────────────────────────────────────────────

  /// Start the camera at [cameraIndex] (0 = back, 1 = front by convention).
  Future<void> startCamera({
    int cameraIndex = 0,
    ResolutionPreset resolution = ResolutionPreset.high,
  }) async {
    if (_availableCameras.isEmpty) {
      _availableCameras = await availableCameras();
    }
    if (_availableCameras.isEmpty) return;

    _activeCameraIndex = cameraIndex.clamp(0, _availableCameras.length - 1);
    await _startCameraAtIndex(_activeCameraIndex, resolution: resolution);
  }

  /// Toggle between front and back cameras.
  Future<void> switchCamera({
    ResolutionPreset resolution = ResolutionPreset.high,
  }) async {
    if (_availableCameras.length < 2) return;

    // Stop current stream before rebuilding the controller
    _isProcessing = false;
    await _cameraController?.stopImageStream();
    await _cameraController?.dispose();
    _cameraController = null;

    _activeCameraIndex = (_activeCameraIndex + 1) % _availableCameras.length;
    await _startCameraAtIndex(_activeCameraIndex, resolution: resolution);
  }

  /// Switch to a specific [CameraLensDirection].
  Future<void> switchToLens(
    CameraLensDirection direction, {
    ResolutionPreset resolution = ResolutionPreset.high,
  }) async {
    final idx = _availableCameras.indexWhere((c) => c.lensDirection == direction);
    if (idx == -1) return; // requested lens not available

    _isProcessing = false;
    await _cameraController?.stopImageStream();
    await _cameraController?.dispose();
    _cameraController = null;

    _activeCameraIndex = idx;
    await _startCameraAtIndex(idx, resolution: resolution);
  }

  Future<void> _startCameraAtIndex(
    int index, {
    ResolutionPreset resolution = ResolutionPreset.high,
  }) async {
    _activeCamera = _availableCameras[index];

    final controller = CameraController(
      _activeCamera!,
      resolution,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    _cameraController = controller;

    await controller.initialize();

    await controller.startImageStream((CameraImage image) {
      if (_isProcessing) return; // ← frame-drop guard
      _isProcessing = true;
      _processFrameAsync(image).then((_) {
        _isProcessing = false;
      }).catchError((_) {
        _isProcessing = false;
      });
    });
  }

  // ── Inference pipeline ───────────────────────────────────────────────────────

  Future<void> _processFrameAsync(CameraImage image) async {
    final rgbBytes = _yuv420ToRgbScaled(image, targetSize: 300);
    final results  = await _runInBackground(rgbBytes, threshold: 0.5);
    onDetections?.call(results);
  }

  Future<List<Detection>> _runInBackground(
    Uint8List rgbBytes, {
    double threshold = 0.5,
  }) async {
    if (!_isReady || _modelBytes == null) return [];
    final payload = _InferPayload(
      rgbBytes:   rgbBytes,
      labels:     _labels,
      threshold:  threshold,
      idxBoxes:   _idxBoxes,
      idxClasses: _idxClasses,
      idxScores:  _idxScores,
      idxNum:     _idxNum,
      modelBytes: _modelBytes!,
    );
    return compute(_inferInIsolate, payload);
  }

  /// Public entry point for running inference on raw RGB bytes (e.g. from a
  /// decoded image file).  Returns detections asynchronously.
  Future<List<Detection>> runOnRgbBytes(
    Uint8List bytes, {
    double threshold = 0.5,
  }) =>
      _runInBackground(bytes, threshold: threshold);

  /// Public entry point for running inference on a [CameraImage].
  Future<List<Detection>> runOnCameraImage(
    CameraImage image, {
    double threshold = 0.5,
  }) async {
    final rgbBytes = _yuv420ToRgbScaled(image, targetSize: 300);
    return _runInBackground(rgbBytes, threshold: threshold);
  }

  // ── YUV420 → RGB (scaled to targetSize × targetSize) ─────────────────────────
  //
  // Runs synchronously but is kept lean:
  //   • integer-only arithmetic (no floating-point in the inner loop)
  //   • single allocation
  //   • nearest-neighbour scaling combined with colour conversion in one pass

  static Uint8List _yuv420ToRgbScaled(CameraImage image, {int targetSize = 300}) {
    final srcW = image.width;
    final srcH = image.height;

    final yPlane  = image.planes[0];
    final uPlane  = image.planes[1];
    final vPlane  = image.planes[2];

    final uvRowStride   = uPlane.bytesPerRow;
    final uvPixelStride = uPlane.bytesPerPixel!;

    final output = Uint8List(targetSize * targetSize * 3);
    int outIdx = 0;

    for (int dstY = 0; dstY < targetSize; dstY++) {
      final srcY = (dstY * srcH ~/ targetSize).clamp(0, srcH - 1);
      for (int dstX = 0; dstX < targetSize; dstX++) {
        final srcX = (dstX * srcW ~/ targetSize).clamp(0, srcW - 1);

        final yVal = yPlane.bytes[srcY * yPlane.bytesPerRow + srcX];
        final uvIdx = (srcY ~/ 2) * uvRowStride + (srcX ~/ 2) * uvPixelStride;
        final uVal  = uPlane.bytes[uvIdx];
        final vVal  = vPlane.bytes[uvIdx];

        // Integer BT.601 coefficients (×256 fixed-point, no floats)
        final yShifted = (yVal - 16)  * 298;
        final uShifted =  uVal - 128;
        final vShifted =  vVal - 128;

        final r = ((yShifted + 409 * vShifted + 128) >> 8).clamp(0, 255);
        final g = ((yShifted - 100 * uShifted - 208 * vShifted + 128) >> 8).clamp(0, 255);
        final b = ((yShifted + 516 * uShifted + 128) >> 8).clamp(0, 255);

        output[outIdx++] = r;
        output[outIdx++] = g;
        output[outIdx++] = b;
      }
    }
    return output;
  }

  // ── Cleanup ──────────────────────────────────────────────────────────────────

  Future<void> dispose() async {
    _isProcessing = false;
    await _cameraController?.stopImageStream();
    await _cameraController?.dispose();
    _cameraController = null;
    _isReady = false;
  }
}


