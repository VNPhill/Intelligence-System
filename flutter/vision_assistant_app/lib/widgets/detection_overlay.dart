import 'package:flutter/material.dart';
import '../main.dart';
import '../services/detection_service.dart';

class DetectionOverlay extends StatelessWidget {
  final List<Detection> detections;
  final Size previewSize;

  const DetectionOverlay({
    super.key,
    required this.detections,
    required this.previewSize,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _BoxPainter(detections),
      size: previewSize,
    );
  }
}

class _BoxPainter extends CustomPainter {
  final List<Detection> detections;

  _BoxPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {
    for (final det in detections) {
      final color = _colorForTier(det.tier);
      final paint = Paint()
        ..color = color
        ..strokeWidth = 2.5
        ..style = PaintingStyle.stroke;

      final ymin = det.box[0] * size.height;
      final xmin = det.box[1] * size.width;
      final ymax = det.box[2] * size.height;
      final xmax = det.box[3] * size.width;

      // Draw corner-style brackets instead of full rectangle
      _drawCornerBrackets(canvas, paint, xmin, ymin, xmax, ymax);

      // Label background pill
      _drawLabel(
        canvas,
        '${det.label}  ${det.scorePercent}',
        Offset(xmin, ymin - 24),
        color,
      );
    }
  }

  void _drawCornerBrackets(
      Canvas canvas, Paint paint, double x1, double y1, double x2, double y2) {
    const len = 16.0;
    final paths = [
      // Top-left
      [Offset(x1, y1 + len), Offset(x1, y1), Offset(x1 + len, y1)],
      // Top-right
      [Offset(x2 - len, y1), Offset(x2, y1), Offset(x2, y1 + len)],
      // Bottom-left
      [Offset(x1, y2 - len), Offset(x1, y2), Offset(x1 + len, y2)],
      // Bottom-right
      [Offset(x2 - len, y2), Offset(x2, y2), Offset(x2, y2 - len)],
    ];

    for (final pts in paths) {
      final path = Path()
        ..moveTo(pts[0].dx, pts[0].dy)
        ..lineTo(pts[1].dx, pts[1].dy)
        ..lineTo(pts[2].dx, pts[2].dy);
      canvas.drawPath(path, paint);
    }
  }

  void _drawLabel(Canvas canvas, String text, Offset offset, Color color) {
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 13,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.3,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    // Background pill
    final bgRect = RRect.fromRectAndRadius(
      Rect.fromLTWH(
          offset.dx - 4, offset.dy - 2, tp.width + 8, tp.height + 4),
      const Radius.circular(4),
    );
    canvas.drawRRect(bgRect, Paint()..color = color.withOpacity(0.85));
    tp.paint(canvas, offset);
  }

  Color _colorForTier(String tier) {
    switch (tier) {
      case 'high':
        return AppTheme.accent;
      case 'medium':
        return AppTheme.warning;
      default:
        return AppTheme.success;
    }
  }

  @override
  bool shouldRepaint(covariant _BoxPainter old) =>
      old.detections != detections;
}

// ── Detection result card for list UI ─────────────────────────────────────────
class DetectionCard extends StatelessWidget {
  final Detection detection;
  final bool isPriority;

  const DetectionCard({
    super.key,
    required this.detection,
    required this.isPriority,
  });

  @override
  Widget build(BuildContext context) {
    final color = _colorForTier(detection.tier);
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withOpacity(0.4), width: 1),
      ),
      child: Row(
        children: [
          Container(
            width: 4,
            height: 36,
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      detection.label.toUpperCase(),
                      style: const TextStyle(
                        color: AppTheme.textPrimary,
                        fontSize: 13,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 0.8,
                      ),
                    ),
                    if (isPriority) ...[
                      const SizedBox(width: 6),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 6, vertical: 1),
                        decoration: BoxDecoration(
                          color: AppTheme.accent.withOpacity(0.2),
                          borderRadius: BorderRadius.circular(4),
                          border: Border.all(
                              color: AppTheme.accent.withOpacity(0.5)),
                        ),
                        child: const Text(
                          'PRIORITY',
                          style: TextStyle(
                            color: AppTheme.accent,
                            fontSize: 9,
                            fontWeight: FontWeight.w800,
                            letterSpacing: 0.5,
                          ),
                        ),
                      ),
                    ],
                  ],
                ),
                const SizedBox(height: 4),
                ClipRRect(
                  borderRadius: BorderRadius.circular(2),
                  child: LinearProgressIndicator(
                    value: detection.score,
                    backgroundColor: AppTheme.surface,
                    color: color,
                    minHeight: 3,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 12),
          Text(
            detection.scorePercent,
            style: TextStyle(
              color: color,
              fontSize: 16,
              fontWeight: FontWeight.w800,
            ),
          ),
        ],
      ),
    );
  }

  Color _colorForTier(String tier) {
    switch (tier) {
      case 'high':
        return AppTheme.accent;
      case 'medium':
        return AppTheme.warning;
      default:
        return AppTheme.success;
    }
  }
}

