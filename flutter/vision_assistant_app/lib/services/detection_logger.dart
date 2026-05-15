import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'detection_service.dart';

class DetectionLog {
  final String label;
  final double score;
  final DateTime timestamp;
  final String mode; // 'live' | 'image'

  DetectionLog({
    required this.label,
    required this.score,
    required this.timestamp,
    required this.mode,
  });

  Map<String, dynamic> toJson() => {
        'label': label,
        'score': score,
        'timestamp': timestamp.toIso8601String(),
        'mode': mode,
      };
}

/// Simple append-only detection logger (addresses reviewer data logging comment)
class DetectionLogger {
  final List<DetectionLog> _sessionLogs = [];

  List<DetectionLog> get sessionLogs => List.unmodifiable(_sessionLogs);

  void log(List<Detection> detections, String mode) {
    for (final d in detections) {
      _sessionLogs.add(DetectionLog(
        label: d.label,
        score: d.score,
        timestamp: d.timestamp,
        mode: mode,
      ));
    }
  }

  Map<String, int> get labelCounts {
    final map = <String, int>{};
    for (final log in _sessionLogs) {
      map[log.label] = (map[log.label] ?? 0) + 1;
    }
    return map;
  }

  Future<void> exportToFile() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final file = File(
          '${dir.path}/detection_log_${DateTime.now().millisecondsSinceEpoch}.json');
      final data = _sessionLogs.map((l) => l.toJson()).toList();
      await file.writeAsString(jsonEncode(data));
    } catch (_) {}
  }

  void clear() => _sessionLogs.clear();
}

