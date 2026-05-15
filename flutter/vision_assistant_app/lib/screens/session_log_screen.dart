import 'package:flutter/material.dart';
import '../main.dart';
import '../services/detection_logger.dart';

class SessionLogScreen extends StatelessWidget {
  final DetectionLogger logger;

  const SessionLogScreen({super.key, required this.logger});

  @override
  Widget build(BuildContext context) {
    final logs = logger.sessionLogs.reversed.toList();
    final counts = logger.labelCounts.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    return Scaffold(
      backgroundColor: AppTheme.bg,
      appBar: AppBar(
        backgroundColor: AppTheme.bg,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded, color: AppTheme.textPrimary),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text(
          'Session Log',
          style: TextStyle(
            color: AppTheme.textPrimary,
            fontSize: 18,
            fontWeight: FontWeight.w700,
          ),
        ),
        actions: [
          TextButton.icon(
            onPressed: () async {
              await logger.exportToFile();
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Log exported to app documents folder'),
                  backgroundColor: AppTheme.success,
                ),
              );
            },
            icon: const Icon(Icons.download_rounded,
                color: AppTheme.accent, size: 18),
            label: const Text('Export',
                style: TextStyle(color: AppTheme.accent)),
          ),
        ],
        elevation: 0,
      ),
      body: logs.isEmpty
          ? const Center(
              child: Text(
                'No detections yet this session.',
                style: TextStyle(color: AppTheme.textSecondary),
              ),
            )
          : CustomScrollView(
              slivers: [
                // Summary counts
                SliverToBoxAdapter(
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          '${logs.length} TOTAL DETECTIONS',
                          style: const TextStyle(
                            color: AppTheme.accent,
                            fontSize: 11,
                            fontWeight: FontWeight.w800,
                            letterSpacing: 2,
                          ),
                        ),
                        const SizedBox(height: 16),
                        Wrap(
                          spacing: 8,
                          runSpacing: 8,
                          children: counts.take(8).map((e) {
                            return Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 12, vertical: 6),
                              decoration: BoxDecoration(
                                color: AppTheme.card,
                                borderRadius: BorderRadius.circular(8),
                                border: Border.all(
                                    color: Colors.white.withOpacity(0.08)),
                              ),
                              child: Text(
                                '${e.key}  ×${e.value}',
                                style: const TextStyle(
                                  color: AppTheme.textPrimary,
                                  fontSize: 12,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            );
                          }).toList(),
                        ),
                        const SizedBox(height: 20),
                        const Divider(color: AppTheme.surface),
                      ],
                    ),
                  ),
                ),

                // Recent log entries
                SliverList(
                  delegate: SliverChildBuilderDelegate(
                    (_, i) {
                      final log = logs[i];
                      return ListTile(
                        dense: true,
                        leading: Container(
                          width: 36,
                          height: 36,
                          decoration: BoxDecoration(
                            color: AppTheme.surface,
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Icon(
                            log.mode == 'live'
                                ? Icons.videocam_rounded
                                : Icons.image_rounded,
                            color: AppTheme.textSecondary,
                            size: 18,
                          ),
                        ),
                        title: Text(
                          log.label,
                          style: const TextStyle(
                            color: AppTheme.textPrimary,
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        subtitle: Text(
                          '${_formatTime(log.timestamp)}  ·  ${log.mode}',
                          style: const TextStyle(
                            color: AppTheme.textSecondary,
                            fontSize: 12,
                          ),
                        ),
                        trailing: Text(
                          '${(log.score * 100).toStringAsFixed(0)}%',
                          style: const TextStyle(
                            color: AppTheme.accentLight,
                            fontSize: 13,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      );
                    },
                    childCount: logs.length,
                  ),
                ),
              ],
            ),
    );
  }

  String _formatTime(DateTime dt) {
    final h = dt.hour.toString().padLeft(2, '0');
    final m = dt.minute.toString().padLeft(2, '0');
    final s = dt.second.toString().padLeft(2, '0');
    return '$h:$m:$s';
  }
}

