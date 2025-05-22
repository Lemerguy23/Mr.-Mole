import 'dart:io';
import 'package:flutter/material.dart';
import 'package:mr_mole/core/utils/notification.dart';
import 'package:mr_mole/core/utils/image_processor.dart';
import 'package:flutter/rendering.dart';

class MoleConfirmationScreen extends StatefulWidget {
  final String imagePath;
  final NotificationService notificationService;
  final Function(String) onConfirm;
  final VoidCallback onCancel;

  const MoleConfirmationScreen({
    super.key,
    required this.imagePath,
    required this.notificationService,
    required this.onConfirm,
    required this.onCancel,
  });

  @override
  State<MoleConfirmationScreen> createState() => _MoleConfirmationScreenState();
}

class _MoleConfirmationScreenState extends State<MoleConfirmationScreen> {
  Offset _position = Offset.zero;
  final double _rectSize = 224.0;
  bool _isDragging = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Подтверждение положения'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: widget.onCancel,
        ),
      ),
      body: Column(
        children: [
          Expanded(
            child: Stack(
              children: [
                Center(
                  child: Image.file(
                    File(widget.imagePath),
                    fit: BoxFit.contain,
                  ),
                ),
                Positioned(
                  left: _position.dx,
                  top: _position.dy,
                  child: GestureDetector(
                    onPanStart: (details) {
                      setState(() {
                        _isDragging = true;
                      });
                    },
                    onPanUpdate: (details) {
                      setState(() {
                        _position += details.delta;
                      });
                    },
                    onPanEnd: (details) {
                      setState(() {
                        _isDragging = false;
                      });
                    },
                    child: Container(
                      width: _rectSize,
                      height: _rectSize,
                      decoration: BoxDecoration(
                        color: Colors.transparent,
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: _isDragging ? Colors.blue : Colors.white,
                          width: 2,
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              children: [
                const Text(
                  'Переместите прямоугольник так, чтобы родинка находилась внутри него',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 16),
                ),
                const SizedBox(height: 16),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    ElevatedButton(
                      onPressed: widget.onCancel,
                      child: const Text('Отмена'),
                    ),
                    ElevatedButton(
                      onPressed: () async {
                        final croppedPath = await _cropImage();
                        widget.onConfirm(croppedPath);
                      },
                      child: const Text('Подтвердить'),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Future<String> _cropImage() async {
    final File imageFile = File(widget.imagePath);
    final imageBytes = await imageFile.readAsBytes();
    final image = await decodeImageFromList(imageBytes);

    // Получаем размеры контейнера изображения
    final RenderBox renderBox = context.findRenderObject() as RenderBox;
    final containerSize = renderBox.size;

    // Вычисляем размеры изображения с учетом BoxFit.contain
    double imageWidth = containerSize.width;
    double imageHeight = containerSize.height;
    double scale = 1.0;

    if (image.width / image.height >
        containerSize.width / containerSize.height) {
      // Изображение шире контейнера
      imageHeight = containerSize.width * image.height / image.width;
      scale = containerSize.width / image.width;
    } else {
      // Изображение выше контейнера
      imageWidth = containerSize.height * image.width / image.height;
      scale = containerSize.height / image.height;
    }

    // Вычисляем отступы для центрирования
    final double offsetX = (containerSize.width - imageWidth) / 2;
    final double offsetY = (containerSize.height - imageHeight) / 2;

    // Создаем прямоугольник с учетом масштаба и отступов
    final cropRect = Rect.fromLTWH(
      (_position.dx - offsetX) / scale,
      (_position.dy - offsetY) / scale,
      _rectSize / scale,
      _rectSize / scale,
    );

    return ImageProcessor.cropImage(
      imagePath: widget.imagePath,
      cropRect: cropRect,
      scaleX: 1.0,
      scaleY: 1.0,
    );
  }
}
