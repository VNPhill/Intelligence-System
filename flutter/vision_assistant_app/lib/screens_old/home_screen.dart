import 'package:flutter/material.dart';
import '../main.dart';
import '../services/detection_service.dart';
import '../services/audio_feedback_service.dart';
import '../services/detection_logger.dart';
import 'live_camera_screen.dart';
import 'image_detection_screen.dart';
import 'session_log_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final DetectionService _detectionService = DetectionService();
  final AudioFeedbackService _audioService = AudioFeedbackService();
  final DetectionLogger _logger = DetectionLogger();

  bool _loading = true;
  bool _audioEnabled = true;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    await _detectionService.init();
    await _audioService.init();
    if (mounted) setState(() => _loading = false);
  }

  void _toggleAudio() {
    _audioService.toggle();
    setState(() => _audioEnabled = _audioService.enabled);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.bg,
      body: SafeArea(
        child: _loading ? _buildLoading() : _buildHome(),
      ),
    );
  }

  Widget _buildLoading() {
    return const Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          CircularProgressIndicator(color: AppTheme.accent),
          SizedBox(height: 16),
          Text(
            'Loading model...',
            style: TextStyle(color: AppTheme.textSecondary, fontSize: 14),
          ),
        ],
      ),
    );
  }

  Widget _buildHome() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 40),
          _buildHeader(),
          const SizedBox(height: 48),
          _buildModeCard(
            icon: Icons.videocam_rounded,
            title: 'Live Detection',
            subtitle: 'Real-time object detection via camera',
            onTap: () => _navigate(
              LiveCameraScreen(
                detectionService: _detectionService,
                audioService: _audioService,
                logger: _logger,
              ),
            ),
          ),
          const SizedBox(height: 16),
          _buildModeCard(
            icon: Icons.image_search_rounded,
            title: 'Image Detection',
            subtitle: 'Detect objects in a photo from gallery',
            onTap: () => _navigate(
              ImageDetectionScreen(
                detectionService: _detectionService,
                audioService: _audioService,
                logger: _logger,
              ),
            ),
          ),
          const SizedBox(height: 16),
          _buildModeCard(
            icon: Icons.analytics_outlined,
            title: 'Session Log',
            subtitle: 'View detection history this session',
            accent: false,
            onTap: () => _navigate(SessionLogScreen(logger: _logger)),
          ),
          const Spacer(),
          _buildBottomBar(),
          const SizedBox(height: 24),
        ],
      ),
    );
  }

  Widget _buildHeader() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Container(
              width: 10,
              height: 10,
              decoration: const BoxDecoration(
                color: AppTheme.accent,
                shape: BoxShape.circle,
              ),
            ),
            const SizedBox(width: 8),
            const Text(
              'VISION ASSIST',
              style: TextStyle(
                color: AppTheme.accent,
                fontSize: 11,
                fontWeight: FontWeight.w800,
                letterSpacing: 2.5,
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        const Text(
          'What do you\nwant to do?',
          style: TextStyle(
            color: AppTheme.textPrimary,
            fontSize: 36,
            fontWeight: FontWeight.w800,
            height: 1.1,
          ),
        ),
        const SizedBox(height: 8),
        const Text(
          'AI-powered environment awareness for everyone.',
          style: TextStyle(
            color: AppTheme.textSecondary,
            fontSize: 14,
            height: 1.5,
          ),
        ),
      ],
    );
  }

  Widget _buildModeCard({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
    bool accent = true,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: AppTheme.card,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: accent
                ? AppTheme.accent.withOpacity(0.3)
                : Colors.white.withOpacity(0.07),
          ),
        ),
        child: Row(
          children: [
            Container(
              width: 52,
              height: 52,
              decoration: BoxDecoration(
                color:
                    accent ? AppTheme.accent.withOpacity(0.15) : AppTheme.surface,
                borderRadius: BorderRadius.circular(14),
              ),
              child: Icon(icon,
                  color: accent ? AppTheme.accent : AppTheme.textSecondary,
                  size: 26),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      color: AppTheme.textPrimary,
                      fontSize: 16,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 3),
                  Text(
                    subtitle,
                    style: const TextStyle(
                      color: AppTheme.textSecondary,
                      fontSize: 13,
                    ),
                  ),
                ],
              ),
            ),
            const Icon(Icons.chevron_right_rounded,
                color: AppTheme.textSecondary),
          ],
        ),
      ),
    );
  }

  Widget _buildBottomBar() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.06)),
      ),
      child: Row(
        children: [
          const Icon(Icons.volume_up_rounded, color: AppTheme.textSecondary, size: 20),
          const SizedBox(width: 10),
          const Expanded(
            child: Text(
              'Audio Feedback',
              style: TextStyle(color: AppTheme.textPrimary, fontSize: 14),
            ),
          ),
          Switch(
            value: _audioEnabled,
            onChanged: (_) => _toggleAudio(),
            activeColor: AppTheme.accent,
          ),
        ],
      ),
    );
  }

  void _navigate(Widget screen) {
    Navigator.push(context, MaterialPageRoute(builder: (_) => screen));
  }

  @override
  void dispose() {
    _detectionService.dispose();
    _audioService.dispose();
    super.dispose();
  }
}

