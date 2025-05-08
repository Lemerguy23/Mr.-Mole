import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:image_picker/image_picker.dart';
import 'settings_page.dart';
import 'camera.dart';
import '../packages/notification.dart';
import '../packages/camera_repo.dart';
import 'analys.dart';

class HomePage extends StatefulWidget {
  final NotificationService notificationService;
  const HomePage({super.key, required this.notificationService});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  Future<void> _pickImage(ImageSource source) async {
    try {
      final ImagePicker picker = ImagePicker();
      final XFile? photo = await picker.pickImage(
        source: source,
        imageQuality: 85,
        maxWidth: 2000,
        maxHeight: 2000,
      );

      if (photo != null) {
        widget.notificationService.setTime();

        // Не нужно сохранять путь в состоянии, сразу переходим на экран анализа
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => AnalysisScreen(imagePath: photo.path),
          ),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Ошибка при выборе изображения: ${e.toString()}'),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: appBar(context),
      body: Center(
        child: const Text(
          'Выберите фото или сделайте снимок',
          style: TextStyle(fontSize: 16),
        ),
      ),
      floatingActionButton: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          FloatingActionButton(
            onPressed: () => _pickImage(ImageSource.gallery),
            backgroundColor: Color(0xFF1b264a),
            elevation: 4.0,
            tooltip: 'Выбрать из галереи',
            heroTag: 'galleryButton',
            child: const Icon(
              Icons.photo_library,
              color: Colors.white,
              size: 32,
            ),
          ),
          SizedBox(width: 128),
          FloatingActionButton(
            onPressed: () async {
              try {
                final cameras = await CameraHandler.getAvailableCameras();
                if (!mounted) return;
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => CameraScreen(cameras: cameras),
                  ),
                );
              } catch (e) {
                if (!mounted) return;
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Ошибка камеры: ${e.toString()}')),
                );
              }
            },
            backgroundColor: Color(0xFF1b264a),
            elevation: 4.0,
            tooltip: 'Сделайте фото',
            heroTag: 'cameraButton',
            child: const Icon(Icons.camera_alt, color: Colors.white, size: 32),
          ),
        ],
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}

AppBar appBar(BuildContext context) {
  return AppBar(
    backgroundColor: Color(0xFF1b264a),
    title: const Text(
      "Фотографируйте ",
      style: TextStyle(
        color: Colors.white,
        fontSize: 18,
        fontWeight: FontWeight.bold,
      ),
    ),
    centerTitle: true,
    elevation: 0.0,
    actions: [
      IconButton(
        icon: SvgPicture.asset(
          'assets/icons/settings.svg',
          height: 30,
          width: 30,
        ),
        onPressed: () {
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => const SettingsPage()),
          );
        },
      ),
    ],
  );
}
