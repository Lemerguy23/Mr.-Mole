import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:math' as math;
import 'package:mr_mole/features/camera/presentation/widgets/instruction_overlay.dart';

class CameraPreviewWidget extends StatelessWidget {
  final CameraController controller;
  final bool showInstruction;
  final VoidCallback onCloseInstruction;

  // Целевой размер для модели - 224 пикселя
  static const int MODEL_SIZE = 224;

  // Масштабный коэффициент для увеличения прямоугольника на экране
  static const double DISPLAY_SCALE_FACTOR = 2.5;

  const CameraPreviewWidget({
    super.key,
    required this.controller,
    this.showInstruction = false,
    required this.onCloseInstruction,
  });

  @override
  Widget build(BuildContext context) {
    // Необходимо дождаться инициализации камеры
    if (!controller.value.isInitialized) {
      return const Center(
        child: CircularProgressIndicator(),
      );
    }

    // Получаем размеры экрана и превью
    final screenSize = MediaQuery.of(context).size;
    final previewSize = controller.value.previewSize!;

    // Соотношение сторон превью камеры
    final double previewAspectRatio = previewSize.width / previewSize.height;

    // Рассчитываем масштаб: каким будет размер 1 пикселя камеры на экране
    double scale;
    if (screenSize.width > screenSize.height * previewAspectRatio) {
      // Экран шире, чем превью камеры относительно соотношения сторон
      scale = screenSize.height / previewSize.height;
    } else {
      // Экран уже, чем превью камеры относительно соотношения сторон
      scale = screenSize.width / previewSize.width;
    }

    // Базовый размер прямоугольника на экране (224 пикселя камеры)
    final double baseRectSize = MODEL_SIZE * scale;

    // Увеличенный размер для отображения (визуально больше, но обрабатываются те же 224 пикселя)
    final double displayRectSize = baseRectSize * DISPLAY_SCALE_FACTOR;

    // Вычисляем координаты центра превью для crop прямоугольника
    final centerX = previewSize.width / 2;
    final centerY = previewSize.height / 2;

    print(
        'Разрешение превью камеры: ${previewSize.width.toInt()}x${previewSize.height.toInt()}');
    print('Масштаб (пикселей камеры на экране): $scale');
    print(
        'Базовый размер прямоугольника: ${baseRectSize}px (соответствует $MODEL_SIZE пикселям камеры)');
    print(
        'Размер отображаемого прямоугольника: ${displayRectSize}px (в ${DISPLAY_SCALE_FACTOR}x больше)');
    print('Центр кадра: $centerX, $centerY');

    return Stack(
      fit: StackFit.expand,
      children: [
        // Базовый превью камеры (заполняет весь экран)
        _buildCameraView(context, BoxFit.cover),

        // Затемнение вокруг прямоугольника
        ClipPath(
          clipper: HoleClipper(squareSize: displayRectSize),
          child: Container(
            color: Colors.black.withOpacity(0.5),
          ),
        ),

        // Рамка для выделенной области с превью необрезанных пикселей
        Center(
          child: Stack(
            alignment: Alignment.center,
            children: [
              // Белая рамка увеличенного прямоугольника
              Container(
                width: displayRectSize,
                height: displayRectSize,
                decoration: BoxDecoration(
                  color: Colors.transparent,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.white, width: 2),
                ),
              ),

              // Показываем реальное превью без масштабирования
              ClipRRect(
                borderRadius: BorderRadius.circular(10),
                child: SizedBox(
                  width: displayRectSize - 4, // Учитываем рамку
                  height: displayRectSize - 4,
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      // Увеличенное превью центра кадра
                      Transform.scale(
                        scale: DISPLAY_SCALE_FACTOR,
                        alignment: Alignment.center,
                        child: CameraPreview(controller),
                      ),

                      // Рамка показывающая реальный размер области 224x224
                      Container(
                        width: baseRectSize,
                        height: baseRectSize,
                        decoration: BoxDecoration(
                          color: Colors.transparent,
                          border: Border.all(
                            color: Colors.yellow,
                            width: 1.0,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),

        // Поясняющий текст
        Positioned(
          bottom: 80,
          left: 0,
          right: 0,
          child: Center(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.6),
                borderRadius: BorderRadius.circular(20),
              ),
              child: const Text(
                'В желтом прямоугольнике - реальные 224×224 пикселя для анализа',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                ),
              ),
            ),
          ),
        ),

        // Инструкция поверх всего
        if (showInstruction) InstructionOverlay(onClose: onCloseInstruction),
      ],
    );
  }

  // Строит превью камеры с указанным режимом fit
  Widget _buildCameraView(BuildContext context, BoxFit fit) {
    return SizedBox.expand(
      child: FittedBox(
        fit: fit,
        child: SizedBox(
          width: controller.value.previewSize!.width,
          height: controller.value.previewSize!.height,
          child: CameraPreview(controller),
        ),
      ),
    );
  }
}

// Специальный класс для создания вырезанного прямоугольника
class HoleClipper extends CustomClipper<Path> {
  final double squareSize;

  HoleClipper({required this.squareSize});

  @override
  Path getClip(Size size) {
    final path = Path()..addRect(Rect.fromLTWH(0, 0, size.width, size.height));

    // Вычисляем координаты центрального квадрата
    final double left = (size.width - squareSize) / 2;
    final double top = (size.height - squareSize) / 2;

    // Вырезаем квадрат из пути
    path.addRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(left, top, squareSize, squareSize),
        const Radius.circular(12),
      ),
    );

    // Используем правило для вырезания внутренней части
    return Path.combine(
      PathOperation.difference,
      path,
      Path()
        ..addRRect(
          RRect.fromRectAndRadius(
            Rect.fromLTWH(left, top, squareSize, squareSize),
            const Radius.circular(12),
          ),
        ),
    );
  }

  @override
  bool shouldReclip(CustomClipper<Path> oldClipper) => true;
}
