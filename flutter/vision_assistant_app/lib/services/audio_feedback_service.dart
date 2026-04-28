import 'package:flutter_tts/flutter_tts.dart';
import 'detection_service.dart';

class AudioFeedbackService {
  final FlutterTts _tts = FlutterTts();
  bool _enabled = true;
  DateTime _lastSpoken = DateTime.fromMillisecondsSinceEpoch(0);

  // Minimum gap between announcements (ms)
  static const int _cooldownMs = 2500;

  bool get enabled => _enabled;

  Future<void> init() async {
    await _tts.setLanguage('en-US');
    await _tts.setSpeechRate(0.5);
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);
  }

  void toggle() => _enabled = !_enabled;

  /// Announce the most important detections (priority objects first)
  Future<void> announce(List<Detection> detections) async {
    if (!_enabled || detections.isEmpty) return;

    final now = DateTime.now();
    if (now.difference(_lastSpoken).inMilliseconds < _cooldownMs) return;
    _lastSpoken = now;

    // Take top 3, priority objects get announced first
    final top = detections.take(3).toList();
    final parts = top.map((d) {
      final dist = _inferDistance(d);
      return '${d.label}${dist.isNotEmpty ? ", $dist" : ""}';
    }).toList();

    final message = parts.length == 1
        ? 'Detected: ${parts[0]}'
        : 'Detected: ${parts.sublist(0, parts.length - 1).join(', ')} and ${parts.last}';

    await _tts.speak(message);
  }

  /// Simple heuristic: larger bounding box → closer object
  String _inferDistance(Detection d) {
    final boxArea =
        (d.box[2] - d.box[0]).abs() * (d.box[3] - d.box[1]).abs();
    if (boxArea > 0.4) return 'very close';
    if (boxArea > 0.15) return 'nearby';
    return '';
  }

  Future<void> speakRaw(String text) async {
    if (!_enabled) return;
    await _tts.speak(text);
  }

  Future<void> stop() async => _tts.stop();

  void dispose() {
    _tts.stop();
  }
}

