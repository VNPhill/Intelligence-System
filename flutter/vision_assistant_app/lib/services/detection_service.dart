import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
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

  String get tier {
    if (score >= 0.85) return 'high';
    if (score >= 0.65) return 'medium';
    return 'low';
  }

  Detection copyWith({
    String? label,
    double? score,
    List<double>? box,
    DateTime? timestamp,
  }) {
    return Detection(
      label: label ?? this.label,
      score: score ?? this.score,
      box: box ?? this.box,
      timestamp: timestamp ?? this.timestamp,
    );
  }
}

class _InferPayload {
  final Uint8List rgbBytes;
  final List<String> labels;
  final double threshold;

  final int idxBoxes;
  final int idxClasses;
  final int idxScores;
  final int idxNum;

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

class _CropRegion {
  final int x;
  final int y;
  final int width;
  final int height;
  final double scale;

  const _CropRegion({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
    required this.scale,
  });
}

List<Detection> _inferInIsolate(_InferPayload p) {
  final interpreter = Interpreter.fromBuffer(p.modelBytes);

  final inputF32 = Float32List(p.rgbBytes.length);
  for (int i = 0; i < p.rgbBytes.length; i++) {
    inputF32[i] = p.rgbBytes[i] / 127.5 - 1.0;
  }

  final inputTensor = inputF32.reshape([1, 300, 300, 3]);

  final outputBoxes = List.generate(
    1,
    (_) => List.generate(10, (_) => List.filled(4, 0.0)),
  );

  final outputClasses = List.generate(
    1,
    (_) => List.filled(10, 0.0),
  );

  final outputScores = List.generate(
    1,
    (_) => List.filled(10, 0.0),
  );

  final numDetections = List.filled(1, 0.0);

  final outputs = <int, Object>{
    p.idxBoxes: outputBoxes,
    p.idxClasses: outputClasses,
    p.idxScores: outputScores,
    p.idxNum: numDetections,
  };

  interpreter.runForMultipleInputs([inputTensor], outputs);
  interpreter.close();

  final results = <Detection>[];
  final count = (numDetections[0]).toInt().clamp(0, 10);

  for (int i = 0; i < count; i++) {
    final score = outputScores[0][i];
    if (score < p.threshold) continue;

    final classId = outputClasses[0][i].toInt();
    final label = classId < p.labels.length ? p.labels[classId] : 'unknown';

    results.add(
      Detection(
        label: label,
        score: score,
        box: List<double>.from(outputBoxes[0][i]),
        timestamp: DateTime.now(),
      ),
    );
  }

  results.sort((a, b) {
    final aPri = DetectionService.isPriority(a.label) ? 0 : 1;
    final bPri = DetectionService.isPriority(b.label) ? 0 : 1;

    if (aPri != bPri) return aPri.compareTo(bPri);
    return b.score.compareTo(a.score);
  });

  return results;
}

class DetectionService {
  bool get isReady => _isReady;
  List<String> get labels => _labels;

  CameraDescription? get activeCamera => _activeCamera;
  CameraController? get cameraController => _cameraController;

  static const List<String> priorityObjects = [
    'person',
    'car',
    'truck',
    'bus',
    'motorcycle',
    'bicycle',
    'traffic light',
    'stop sign',
    'stairs',
    'door',
    'chair',
    'couch',
    'bed',
    'dining table',
    'toilet',
    'potted plant',
    'dog',
    'cat',
    'bottle',
    'cup',
    'cell phone',
  ];

  static bool isPriority(String label) {
    return priorityObjects.any(
      (p) => label.toLowerCase().contains(p),
    );
  }

  bool _isReady = false;
  List<String> _labels = [];

  Uint8List? _modelBytes;

  int _idxBoxes = 0;
  int _idxClasses = 1;
  int _idxScores = 2;
  int _idxNum = 3;

  bool _isProcessing = false;

  List<CameraDescription> _availableCameras = [];
  CameraDescription? _activeCamera;
  CameraController? _cameraController;
  int _activeCameraIndex = 0;

  void Function(List<Detection>)? onDetections;

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

  void _resolveOutputIndices() {
    if (_modelBytes == null) return;

    final interp = Interpreter.fromBuffer(_modelBytes!);
    final outDetails = interp.getOutputTensors();

    for (int i = 0; i < outDetails.length; i++) {
      final shape = outDetails[i].shape;

      if (shape.length == 3 && shape[2] == 4) {
        _idxBoxes = i;
      } else if (shape.length == 1) {
        _idxNum = i;
      }
    }

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

    debugPrint(
      '[DetectionService] Output mapping → '
      'boxes=$_idxBoxes classes=$_idxClasses '
      'scores=$_idxScores num=$_idxNum',
    );
  }

  Future<void> startCamera({
    int cameraIndex = 0,
    ResolutionPreset resolution = ResolutionPreset.medium,
  }) async {
    if (_availableCameras.isEmpty) {
      _availableCameras = await availableCameras();
    }

    if (_availableCameras.isEmpty) return;

    _activeCameraIndex = cameraIndex.clamp(0, _availableCameras.length - 1);
    await _startCameraAtIndex(_activeCameraIndex, resolution: resolution);
  }

  Future<void> switchCamera({
    ResolutionPreset resolution = ResolutionPreset.medium,
  }) async {
    if (_availableCameras.length < 2) return;

    _isProcessing = false;
    await _cameraController?.stopImageStream();
    await _cameraController?.dispose();
    _cameraController = null;

    _activeCameraIndex = (_activeCameraIndex + 1) % _availableCameras.length;
    await _startCameraAtIndex(_activeCameraIndex, resolution: resolution);
  }

  Future<void> switchToLens(
    CameraLensDirection direction, {
    ResolutionPreset resolution = ResolutionPreset.medium,
  }) async {
    final idx = _availableCameras.indexWhere(
      (c) => c.lensDirection == direction,
    );

    if (idx == -1) return;

    _isProcessing = false;
    await _cameraController?.stopImageStream();
    await _cameraController?.dispose();
    _cameraController = null;

    _activeCameraIndex = idx;
    await _startCameraAtIndex(idx, resolution: resolution);
  }

  Future<void> _startCameraAtIndex(
    int index, {
    ResolutionPreset resolution = ResolutionPreset.medium,
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
      if (_isProcessing) return;

      _isProcessing = true;

      _processFrameAsync(image).then((_) {
        _isProcessing = false;
      }).catchError((_) {
        _isProcessing = false;
      });
    });
  }

  Future<void> _processFrameAsync(CameraImage image) async {
    final results = await runOnCameraImage(image, threshold: 0.5);
    onDetections?.call(results);
  }

  Future<List<Detection>> _runInBackground(
    Uint8List rgbBytes, {
    double threshold = 0.5,
  }) async {
    if (!_isReady || _modelBytes == null) return [];

    final payload = _InferPayload(
      rgbBytes: rgbBytes,
      labels: _labels,
      threshold: threshold,
      idxBoxes: _idxBoxes,
      idxClasses: _idxClasses,
      idxScores: _idxScores,
      idxNum: _idxNum,
      modelBytes: _modelBytes!,
    );

    return compute(_inferInIsolate, payload);
  }

  Future<List<Detection>> runOnRgbBytes(
    Uint8List bytes, {
    double threshold = 0.5,
  }) {
    return _runInBackground(bytes, threshold: threshold);
  }

  Future<List<Detection>> runOnCameraImage(
    CameraImage image, {
    double threshold = 0.5,
  }) async {
    final rgbBytes = _yuv420ToRgbScaled(
      image,
      targetSize: 300,
      crop: _CropRegion(
        x: 0,
        y: 0,
        width: image.width,
        height: image.height,
        scale: 1.0,
      ),
    );

    return _runInBackground(rgbBytes, threshold: threshold);
  }

  Future<List<Detection>> runOnCameraImageMultiScale(
    CameraImage image, {
    double threshold = 0.5,
    List<double> centerScales = const [1.0, 0.8, 0.5],
    double nmsIouThreshold = 0.45,
  }) async {
    final crops = _buildCenterCrops(
      srcWidth: image.width,
      srcHeight: image.height,
      centerScales: centerScales,
    );

    final allResults = <Detection>[];

    for (final crop in crops) {
      final rgbBytes = _yuv420ToRgbScaled(
        image,
        targetSize: 300,
        crop: crop,
      );

      final cropResults = await _runInBackground(
        rgbBytes,
        threshold: threshold,
      );

      final remapped = cropResults.map((d) {
        return _remapCropDetectionToFullFrame(
          detection: d,
          crop: crop,
          srcWidth: image.width,
          srcHeight: image.height,
        );
      });

      allResults.addAll(remapped);
    }

    return _sortDetections(
      _nmsSameLabel(
        allResults,
        iouThreshold: nmsIouThreshold,
      ),
    );
  }

  Future<List<Detection>> runOnImageMultiScale(
    img.Image decoded, {
    double threshold = 0.5,
    List<double> centerScales = const [1.0, 0.8, 0.5],
    double nmsIouThreshold = 0.45,
  }) async {
    final crops = _buildCenterCrops(
      srcWidth: decoded.width,
      srcHeight: decoded.height,
      centerScales: centerScales,
    );

    final allResults = <Detection>[];

    for (final crop in crops) {
      final rgbBytes = _imageCropToRgbScaled(
        decoded,
        crop: crop,
        targetSize: 300,
      );

      final cropResults = await _runInBackground(
        rgbBytes,
        threshold: threshold,
      );

      final remapped = cropResults.map((d) {
        return _remapCropDetectionToFullFrame(
          detection: d,
          crop: crop,
          srcWidth: decoded.width,
          srcHeight: decoded.height,
        );
      });

      allResults.addAll(remapped);
    }

    return _sortDetections(
      _nmsSameLabel(
        allResults,
        iouThreshold: nmsIouThreshold,
      ),
    );
  }

  static List<_CropRegion> _buildCenterCrops({
    required int srcWidth,
    required int srcHeight,
    required List<double> centerScales,
  }) {
    final cleaned = centerScales
        .where((s) => s > 0.0 && s <= 1.0)
        .map((s) => double.parse(s.toStringAsFixed(3)))
        .toSet()
        .toList()
      ..sort((a, b) => b.compareTo(a));

    if (cleaned.isEmpty) {
      cleaned.add(1.0);
    }

    return cleaned.map((scale) {
      final cropW = (srcWidth * scale).round().clamp(1, srcWidth);
      final cropH = (srcHeight * scale).round().clamp(1, srcHeight);

      final cropX = ((srcWidth - cropW) / 2).round();
      final cropY = ((srcHeight - cropH) / 2).round();

      return _CropRegion(
        x: cropX,
        y: cropY,
        width: cropW,
        height: cropH,
        scale: scale,
      );
    }).toList();
  }

  static Detection _remapCropDetectionToFullFrame({
    required Detection detection,
    required _CropRegion crop,
    required int srcWidth,
    required int srcHeight,
  }) {
    final ymin = detection.box[0];
    final xmin = detection.box[1];
    final ymax = detection.box[2];
    final xmax = detection.box[3];

    final fullYmin = (crop.y + ymin * crop.height) / srcHeight;
    final fullXmin = (crop.x + xmin * crop.width) / srcWidth;
    final fullYmax = (crop.y + ymax * crop.height) / srcHeight;
    final fullXmax = (crop.x + xmax * crop.width) / srcWidth;

    return detection.copyWith(
      box: [
        fullYmin.clamp(0.0, 1.0),
        fullXmin.clamp(0.0, 1.0),
        fullYmax.clamp(0.0, 1.0),
        fullXmax.clamp(0.0, 1.0),
      ],
    );
  }

  static List<Detection> _sortDetections(List<Detection> detections) {
    final sorted = [...detections];

    sorted.sort((a, b) {
      final aPri = DetectionService.isPriority(a.label) ? 0 : 1;
      final bPri = DetectionService.isPriority(b.label) ? 0 : 1;

      if (aPri != bPri) return aPri.compareTo(bPri);
      return b.score.compareTo(a.score);
    });

    return sorted;
  }

  static List<Detection> _nmsSameLabel(
    List<Detection> detections, {
    double iouThreshold = 0.45,
  }) {
    final sorted = [...detections]
      ..sort((a, b) => b.score.compareTo(a.score));

    final kept = <Detection>[];

    for (final det in sorted) {
      bool shouldKeep = true;

      for (final existing in kept) {
        final sameLabel =
            det.label.toLowerCase().trim() == existing.label.toLowerCase().trim();

        if (!sameLabel) continue;

        final iou = _iou(det.box, existing.box);

        if (iou >= iouThreshold) {
          shouldKeep = false;
          break;
        }
      }

      if (shouldKeep) {
        kept.add(det);
      }
    }

    return kept;
  }

  static double _iou(List<double> a, List<double> b) {
    final ay1 = a[0];
    final ax1 = a[1];
    final ay2 = a[2];
    final ax2 = a[3];

    final by1 = b[0];
    final bx1 = b[1];
    final by2 = b[2];
    final bx2 = b[3];

    final interY1 = ay1 > by1 ? ay1 : by1;
    final interX1 = ax1 > bx1 ? ax1 : bx1;
    final interY2 = ay2 < by2 ? ay2 : by2;
    final interX2 = ax2 < bx2 ? ax2 : bx2;

    final interH = (interY2 - interY1).clamp(0.0, 1.0);
    final interW = (interX2 - interX1).clamp(0.0, 1.0);
    final interArea = interH * interW;

    final areaA = ((ay2 - ay1).clamp(0.0, 1.0)) *
        ((ax2 - ax1).clamp(0.0, 1.0));

    final areaB = ((by2 - by1).clamp(0.0, 1.0)) *
        ((bx2 - bx1).clamp(0.0, 1.0));

    final union = areaA + areaB - interArea;

    if (union <= 0.0) return 0.0;
    return interArea / union;
  }

  static Uint8List _imageCropToRgbScaled(
    img.Image image, {
    required _CropRegion crop,
    int targetSize = 300,
  }) {
    final output = Uint8List(targetSize * targetSize * 3);
    int outIdx = 0;

    for (int dstY = 0; dstY < targetSize; dstY++) {
      final srcY = crop.y + (dstY * crop.height ~/ targetSize);

      for (int dstX = 0; dstX < targetSize; dstX++) {
        final srcX = crop.x + (dstX * crop.width ~/ targetSize);
        final pixel = image.getPixel(srcX, srcY);

        output[outIdx++] = pixel.r.toInt();
        output[outIdx++] = pixel.g.toInt();
        output[outIdx++] = pixel.b.toInt();
      }
    }

    return output;
  }

  static Uint8List _yuv420ToRgbScaled(
    CameraImage image, {
    int targetSize = 300,
    required _CropRegion crop,
  }) {
    final srcW = image.width;
    final srcH = image.height;

    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final uvRowStride = uPlane.bytesPerRow;
    final uvPixelStride = uPlane.bytesPerPixel!;

    final output = Uint8List(targetSize * targetSize * 3);
    int outIdx = 0;

    for (int dstY = 0; dstY < targetSize; dstY++) {
      final srcY = (crop.y + dstY * crop.height ~/ targetSize)
          .clamp(0, srcH - 1);

      for (int dstX = 0; dstX < targetSize; dstX++) {
        final srcX = (crop.x + dstX * crop.width ~/ targetSize)
            .clamp(0, srcW - 1);

        final yVal = yPlane.bytes[srcY * yPlane.bytesPerRow + srcX];

        final uvIdx =
            (srcY ~/ 2) * uvRowStride + (srcX ~/ 2) * uvPixelStride;

        final uVal = uPlane.bytes[uvIdx];
        final vVal = vPlane.bytes[uvIdx];

        final yShifted = (yVal - 16) * 298;
        final uShifted = uVal - 128;
        final vShifted = vVal - 128;

        final r = ((yShifted + 409 * vShifted + 128) >> 8).clamp(0, 255);
        final g =
            ((yShifted - 100 * uShifted - 208 * vShifted + 128) >> 8)
                .clamp(0, 255);
        final b = ((yShifted + 516 * uShifted + 128) >> 8).clamp(0, 255);

        output[outIdx++] = r;
        output[outIdx++] = g;
        output[outIdx++] = b;
      }
    }

    return output;
  }

  Future<void> dispose() async {
    _isProcessing = false;
    await _cameraController?.stopImageStream();
    await _cameraController?.dispose();
    _cameraController = null;
    _isReady = false;
  }
}