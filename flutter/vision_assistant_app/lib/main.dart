import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'screens/home_screen.dart';

late List<CameraDescription> cameras;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(const VisionAssistApp());
}

class VisionAssistApp extends StatelessWidget {
  const VisionAssistApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'VisionAssist',
      theme: AppTheme.darkTheme,
      home: const HomeScreen(),
    );
  }
}

class AppTheme {
  static const Color bg = Color(0xFF0D0D0D);
  static const Color surface = Color(0xFF1A1A1A);
  static const Color card = Color(0xFF242424);
  static const Color accent = Color(0xFFFF6B00);
  static const Color accentLight = Color(0xFFFF9240);
  static const Color textPrimary = Color(0xFFF5F5F5);
  static const Color textSecondary = Color(0xFF9E9E9E);
  static const Color success = Color(0xFF4CAF50);
  static const Color warning = Color(0xFFFFB300);
  static const Color danger = Color(0xFFF44336);

  static ThemeData get darkTheme => ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: bg,
        colorScheme: const ColorScheme.dark(
          primary: accent,
          secondary: accentLight,
          surface: surface,
        ),
        useMaterial3: true,
      );
}


