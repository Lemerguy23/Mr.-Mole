import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

class CustomCameraScreen extends StatefulWidget {
  @override
  CustomCameraScreenState createState() => CustomCameraScreenState();
}

class CustomCameraScreenState extends State<CustomCameraScreen> {
  CameraController? _controller;
  List<CameraDescription>? cameras;
  Future<void>? _initializeControllerFuture;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    cameras = await availableCameras();
    if (cameras == null || cameras!.isEmpty) {
      print("Камера не найдена");
      return;
    }
    _controller = CameraController(cameras!.first, ResolutionPreset.medium);
    _initializeControllerFuture = _controller!.initialize();
    setState(() {}); // Обновляем состояние после инициализации
  }

  Future<void> _takePicture() async {
    if (_controller == null || !_controller!.value.isInitialized) return;

    final image = await _controller!.takePicture();
    print("Фото сохранено: ${image.path}");
  }

  @override
  void dispose() {
    if (_controller != null && _controller!.value.isInitialized) {
      _controller?.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Камера'), centerTitle: true),
      body:
          _controller == null
              ? Center(
                child: CircularProgressIndicator(),
              ) // Показываем загрузку
              : FutureBuilder(
                future: _initializeControllerFuture,
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.done) {
                    return CameraPreview(_controller!);
                  } else {
                    return Center(child: CircularProgressIndicator());
                  }
                },
              ),
      floatingActionButton:
          _controller != null && _controller!.value.isInitialized
              ? FloatingActionButton(
                onPressed: _takePicture,
                shape: CircleBorder(),

                child: Icon(Icons.circle, size: 50),
              )
              : null,

      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}
