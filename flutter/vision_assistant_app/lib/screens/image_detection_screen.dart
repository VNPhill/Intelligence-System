import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import '../main.dart';
import '../services/detection_service.dart';
import '../services/audio_feedback_service.dart';
import '../services/detection_logger.dart';
import '../widgets/detection_overlay.dart';

class ImageDetectionScreen extends StatefulWidget {
  final DetectionService detectionService;
  final AudioFeedbackService audioService;
  final DetectionLogger logger;

  const ImageDetectionScreen({
    super.key,
    required this.detectionService,
    required this.audioService,
    required this.logger,
  });

  @override
  State<ImageDetectionScreen> createState() => _ImageDetectionScreenState();
}

class _ImageDetectionScreenState extends State<ImageDetectionScreen> {
  File? _imageFile;
  List<Detection> _detections = [];
  bool _isProcessing = false;
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage(ImageSource source) async {
    final picked = await _picker.pickImage(source: source);
    if (picked == null) return;

    setState(() {
      _imageFile = File(picked.path);
      _detections = [];
      _isProcessing = true;
    });

    await _runDetection(File(picked.path));
  }

  Future<void> _runDetection(File file) async {
    try {
      final bytes = await file.readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) return;

      final resized = img.copyResize(decoded, width: 300, height: 300);
      final rgb = Uint8List(300 * 300 * 3);
      int idx = 0;
      for (int y = 0; y < 300; y++) {
        for (int x = 0; x < 300; x++) {
          final pixel = resized.getPixel(x, y);
          rgb[idx++] = pixel.r.toInt();
          rgb[idx++] = pixel.g.toInt();
          rgb[idx++] = pixel.b.toInt();
        }
      }

      final results = widget.detectionService.runOnRgbBytes(rgb);
      widget.logger.log(results, 'image');

      if (results.isNotEmpty) {
        await widget.audioService.announce(results);
      } else {
        await widget.audioService.speakRaw('No objects detected');
      }

      setState(() {
        _detections = results;
        _isProcessing = false;
      });
    } catch (e) {
      setState(() => _isProcessing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.bg,
      appBar: AppBar(
        backgroundColor: AppTheme.bg,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded, color: AppTheme.textPrimary),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text(
          'Image Detection',
          style: TextStyle(
            color: AppTheme.textPrimary,
            fontSize: 18,
            fontWeight: FontWeight.w700,
          ),
        ),
        elevation: 0,
      ),
      body: Column(
        children: [
          Expanded(
            flex: 6,
            child: _buildImageArea(),
          ),
          Expanded(
            flex: 4,
            child: _buildResultsArea(),
          ),
        ],
      ),
      bottomNavigationBar: _buildPickBar(),
    );
  }

  Widget _buildImageArea() {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 8, 16, 0),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withOpacity(0.07)),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: _imageFile == null ? _buildPlaceholder() : _buildImageWithBoxes(),
      ),
    );
  }

  Widget _buildPlaceholder() {
    return const Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.add_photo_alternate_outlined,
              color: AppTheme.textSecondary, size: 52),
          SizedBox(height: 12),
          Text(
            'Select an image to analyze',
            style: TextStyle(color: AppTheme.textSecondary, fontSize: 14),
          ),
        ],
      ),
    );
  }

  Widget _buildImageWithBoxes() {
    return LayoutBuilder(builder: (context, constraints) {
      return Stack(
        children: [
          Positioned.fill(
            child: Image.file(_imageFile!, fit: BoxFit.cover),
          ),
          if (!_isProcessing)
            Positioned.fill(
              child: DetectionOverlay(
                detections: _detections,
                previewSize:
                    Size(constraints.maxWidth, constraints.maxHeight),
              ),
            ),
          if (_isProcessing)
            const Center(
              child: CircularProgressIndicator(color: AppTheme.accent),
            ),
        ],
      );
    });
  }

  Widget _buildResultsArea() {
    if (_isProcessing) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(color: AppTheme.accent),
            SizedBox(height: 12),
            Text('Analyzing image...',
                style: TextStyle(color: AppTheme.textSecondary)),
          ],
        ),
      );
    }

    if (_imageFile == null) return const SizedBox();

    if (_detections.isEmpty) {
      return const Center(
        child: Text(
          'No objects detected.',
          style: TextStyle(color: AppTheme.textSecondary, fontSize: 14),
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 16, 20, 8),
          child: Text(
            '${_detections.length} object${_detections.length > 1 ? 's' : ''} detected',
            style: const TextStyle(
              color: AppTheme.textSecondary,
              fontSize: 12,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.5,
            ),
          ),
        ),
        Expanded(
          child: ListView.builder(
            itemCount: _detections.length,
            itemBuilder: (_, i) => DetectionCard(
              detection: _detections[i],
              isPriority: DetectionService.isPriority(_detections[i].label),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildPickBar() {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
        child: Row(
          children: [
            Expanded(
              child: _actionButton(
                icon: Icons.photo_library_rounded,
                label: 'Gallery',
                onTap: () => _pickImage(ImageSource.gallery),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _actionButton(
                icon: Icons.camera_alt_rounded,
                label: 'Camera',
                onTap: () => _pickImage(ImageSource.camera),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _actionButton({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 16),
        decoration: BoxDecoration(
          color: AppTheme.accent,
          borderRadius: BorderRadius.circular(14),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: Colors.white, size: 20),
            const SizedBox(width: 8),
            Text(
              label,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 15,
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

