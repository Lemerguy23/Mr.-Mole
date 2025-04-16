import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'settings_page.dart';
import '../widget/camera.dart';
import '../packages/notification.dart';

class HomePage extends StatefulWidget {
  final NotificationService notificationService;
  const HomePage({super.key, required this.notificationService});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  String? _imagePath;

  Future<void> _pickImage(ImageSource source) async {
    final ImagePicker picker = ImagePicker();
    final XFile? photo = await picker.pickImage(source: source);
    if (photo != null) {
      setState(() {
        _imagePath = photo.path;
        widget.notificationService.setTime();
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: appBar(context),
      body: Center(
        child:
            _imagePath != null
                ? ClipRRect(
                  borderRadius: BorderRadius.circular(20),
                  child: Image.file(
                    File(_imagePath!),
                    fit: BoxFit.cover,
                    width: 300,
                    height: 300,
                  ),
                )
                : const Text(
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
            mini: false,
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
            onPressed:
                () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => CustomCameraScreen()),
                ),
            backgroundColor: Color(0xFF1b264a),
            elevation: 4.0,
            mini: false,
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
      GestureDetector(
        onTap: () {
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => const SettingsPage()),
          );
        },
        child: Container(
          margin: EdgeInsets.all(5),
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: Color(0xFF1b264a),
            borderRadius: BorderRadius.circular(20),
          ),
          child: SvgPicture.asset(
            'assets/icons/settings.svg',
            height: 30,
            width: 30,
          ),
        ),
      ),
    ],
  );
}
