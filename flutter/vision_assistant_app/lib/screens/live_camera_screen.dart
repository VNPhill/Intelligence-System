import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import '../main.dart' show AppTheme, cameras;
import '../services/detection_service.dart';
import '../services/audio_feedback_service.dart';
import '../services/detection_logger.dart';
import '../widgets/detection_overlay.dart';

class LiveCameraScreen extends StatefulWidget {
  final DetectionService detectionService;
  final AudioFeedbackService audioService;
  final DetectionLogger logger;

  const LiveCameraScreen({
    super.key,
    required this.detectionService,
    required this.audioService,
    required this.logger,
  });

  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}

class _LiveCameraScreenState extends State<LiveCameraScreen> {
  late CameraController _cameraController;
  List<Detection> _detections = [];
  bool _isBusy = false;
  bool _isDetecting = true;
  bool _cameraReady = false;
  double _threshold = 0.5;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    _cameraController = CameraController(
      cameras[0],
      ResolutionPreset.medium,
      enableAudio: false,
    );
    await _cameraController.initialize();
    _cameraController.startImageStream(_processFrame);
    if (mounted) setState(() => _cameraReady = true);
  }

  void _processFrame(CameraImage image) {
    if (!_isDetecting || _isBusy) return;
    _isBusy = true;

    final results = widget.detectionService.runOnCameraImage(
      image,
      threshold: _threshold,
    );

    widget.logger.log(results, 'live');
    widget.audioService.announce(results);

    if (mounted) {
      setState(() => _detections = results);
    }
    _isBusy = false;
  }

  void _toggleDetection() {
    setState(() => _isDetecting = !_isDetecting);
    if (_isDetecting) {
      widget.audioService.speakRaw('Detection started');
    } else {
      widget.audioService.speakRaw('Detection paused');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: _cameraReady ? _buildBody() : _buildLoading(),
    );
  }

  Widget _buildLoading() {
    return const Center(
      child: CircularProgressIndicator(color: AppTheme.accent),
    );
  }

  Widget _buildBody() {
    final size = MediaQuery.of(context).size;
    return Stack(
      children: [
        // Camera preview
        SizedBox.expand(child: CameraPreview(_cameraController)),

        // Detection boxes
        SizedBox(
          width: size.width,
          height: size.height,
          child: DetectionOverlay(
            detections: _isDetecting ? _detections : [],
            previewSize: size,
          ),
        ),

        // Top bar
        _buildTopBar(),

        // Bottom panel
        Positioned(
          bottom: 0,
          left: 0,
          right: 0,
          child: _buildBottomPanel(),
        ),
      ],
    );
  }

  Widget _buildTopBar() {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        child: Row(
          children: [
            _iconButton(
              Icons.arrow_back_rounded,
              () => Navigator.pop(context),
            ),
            const Spacer(),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.6),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                children: [
                  AnimatedContainer(
                    duration: const Duration(milliseconds: 300),
                    width: 8,
                    height: 8,
                    decoration: BoxDecoration(
                      color: _isDetecting ? AppTheme.danger : AppTheme.textSecondary,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 6),
                  Text(
                    _isDetecting ? 'LIVE' : 'PAUSED',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 1.5,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            _iconButton(
              widget.audioService.enabled
                  ? Icons.volume_up_rounded
                  : Icons.volume_off_rounded,
              () {
                widget.audioService.toggle();
                setState(() {});
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBottomPanel() {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.bottomCenter,
          end: Alignment.topCenter,
          colors: [Colors.black.withOpacity(0.92), Colors.transparent],
          stops: const [0.6, 1.0],
        ),
      ),
      padding: const EdgeInsets.fromLTRB(0, 32, 0, 40),
      child: Column(
        children: [
          // Detection results
          if (_detections.isNotEmpty)
            SizedBox(
              height: 140,
              child: ListView.builder(
                physics: const NeverScrollableScrollPhysics(),
                itemCount: _detections.take(3).length,
                itemBuilder: (_, i) => DetectionCard(
                  detection: _detections[i],
                  isPriority: DetectionService.isPriority(_detections[i].label),
                ),
              ),
            )
          else
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 20),
              child: Text(
                _isDetecting ? 'Scanning environment...' : 'Detection paused',
                style: const TextStyle(
                    color: AppTheme.textSecondary, fontSize: 14),
              ),
            ),

          const SizedBox(height: 16),

          // Threshold slider
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Row(
              children: [
                const Text('Sensitivity',
                    style: TextStyle(
                        color: AppTheme.textSecondary, fontSize: 12)),
                Expanded(
                  child: Slider(
                    value: _threshold,
                    min: 0.2,
                    max: 0.9,
                    onChanged: (v) => setState(() => _threshold = v),
                    activeColor: AppTheme.accent,
                    inactiveColor: AppTheme.surface,
                  ),
                ),
                Text('${(_threshold * 100).toInt()}%',
                    style: const TextStyle(
                        color: AppTheme.textSecondary, fontSize: 12)),
              ],
            ),
          ),

          const SizedBox(height: 8),

          // Play/Pause button
          GestureDetector(
            onTap: _toggleDetection,
            child: Container(
              width: 64,
              height: 64,
              decoration: BoxDecoration(
                color: _isDetecting ? AppTheme.danger : AppTheme.accent,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: (_isDetecting ? AppTheme.danger : AppTheme.accent)
                        .withOpacity(0.4),
                    blurRadius: 20,
                    spreadRadius: 2,
                  )
                ],
              ),
              child: Icon(
                _isDetecting ? Icons.pause_rounded : Icons.play_arrow_rounded,
                color: Colors.white,
                size: 30,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _iconButton(IconData icon, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.5),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Icon(icon, color: Colors.white, size: 22),
      ),
    );
  }

  @override
  void dispose() {
    _cameraController.dispose();
    super.dispose();
  }
}

