import 'dart:io';
import 'dart:typed_data';
import 'dart:ui';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

class ImageProcessor {
  // Размер изображения для модели
  static const int MODEL_SIZE = 224;

  /// Обрезает изображение до указанного размера
  static Future<String> cropImage({
    required String imagePath,
    required Rect cropRect,
    required double scaleX,
    required double scaleY,
  }) async {
    try {
      final File imageFile = File(imagePath);
      final Uint8List imageBytes = await imageFile.readAsBytes();
      final img.Image? originalImage = img.decodeImage(imageBytes);

      if (originalImage == null) {
        throw Exception('Не удалось декодировать изображение');
      }

      print(
          'Оригинальное изображение: ${originalImage.width}x${originalImage.height}');
      print('Прямоугольник обрезки: $cropRect');

      // Берем координаты прямоугольника как есть
      final int cropX = cropRect.left.round();
      final int cropY = cropRect.top.round();

      // Проверяем, что координаты обрезки не выходят за пределы изображения
      final int safeCropX = cropX.clamp(0, originalImage.width - MODEL_SIZE);
      final int safeCropY = cropY.clamp(0, originalImage.height - MODEL_SIZE);

      print(
          'Безопасные координаты обрезки: x=$safeCropX, y=$safeCropY, size=$MODEL_SIZE');

      // Обрезаем изображение
      final img.Image croppedImage = img.copyCrop(
        originalImage,
        x: safeCropX,
        y: safeCropY,
        width: MODEL_SIZE,
        height: MODEL_SIZE,
      );

      // Получаем временную директорию приложения
      final Directory tempDir = await getTemporaryDirectory();
      final String timestamp = DateTime.now().millisecondsSinceEpoch.toString();
      final String croppedPath = '${tempDir.path}/cropped_$timestamp.jpg';
      final File croppedFile = File(croppedPath);

      // Сохраняем обрезанное изображение
      await croppedFile.writeAsBytes(img.encodeJpg(croppedImage, quality: 95));

      // Проверяем, что файл действительно создан
      if (await croppedFile.exists()) {
        print('Обрезанное изображение создано: $croppedPath');
        return croppedPath;
      } else {
        throw Exception('Не удалось создать обрезанное изображение');
      }
    } catch (e) {
      print('Ошибка при обработке изображения: $e');
      // В случае ошибки возвращаем оригинальный путь
      return imagePath;
    }
  }

  /// Обрезает центральную часть изображения
  static Future<String> cropCenterSquare(String imagePath) async {
    try {
      final File imageFile = File(imagePath);
      final Uint8List imageBytes = await imageFile.readAsBytes();
      final img.Image? originalImage = img.decodeImage(imageBytes);

      if (originalImage == null) {
        throw Exception('Не удалось декодировать изображение');
      }

      print(
          'Оригинальное изображение: ${originalImage.width}x${originalImage.height}');

      // Вычисляем координаты центрального квадрата
      final int centerX = (originalImage.width - MODEL_SIZE) ~/ 2;
      final int centerY = (originalImage.height - MODEL_SIZE) ~/ 2;

      final int safeCropX = centerX.clamp(0, originalImage.width - MODEL_SIZE);
      final int safeCropY = centerY.clamp(0, originalImage.height - MODEL_SIZE);

      print(
          'Координаты центрального квадрата: x=$safeCropX, y=$safeCropY, size=$MODEL_SIZE');

      // Обрезаем изображение
      final img.Image croppedImage = img.copyCrop(
        originalImage,
        x: safeCropX,
        y: safeCropY,
        width: MODEL_SIZE,
        height: MODEL_SIZE,
      );

      // Получаем временную директорию приложения
      final Directory tempDir = await getTemporaryDirectory();
      final String timestamp = DateTime.now().millisecondsSinceEpoch.toString();
      final String croppedPath =
          '${tempDir.path}/cropped_center_$timestamp.jpg';
      final File croppedFile = File(croppedPath);

      // Сохраняем обрезанное изображение
      await croppedFile.writeAsBytes(img.encodeJpg(croppedImage, quality: 95));

      print('Центральный квадрат изображения создан: $croppedPath');

      // Информация для отладки проблемы с масштабированием
      double scaleRatio = originalImage.width / MODEL_SIZE;
      print(
          'Коэффициент масштабирования: $scaleRatio (чем больше, тем сильнее зум)');
      print('Целевой размер: $MODEL_SIZE x $MODEL_SIZE');

      return croppedPath;
    } catch (e) {
      print('Ошибка при обрезке центрального квадрата: $e');
      return imagePath;
    }
  }

  /// Обрезает центральную часть изображения пропорционально разрешению
  static Future<String> cropCenterProportional(String imagePath,
      {double cropRatio = 0.5}) async {
    try {
      final File imageFile = File(imagePath);
      final Uint8List imageBytes = await imageFile.readAsBytes();
      final img.Image? originalImage = img.decodeImage(imageBytes);

      if (originalImage == null) {
        throw Exception('Не удалось декодировать изображение');
      }

      print(
          'Оригинальное изображение: ${originalImage.width}x${originalImage.height}');

      // Вычисляем размер большого прямоугольника пропорционально размеру изображения
      // cropRatio указывает, какой процент от ширины изображения мы используем (0.5 = 50%)
      final int cropSize = (originalImage.width * cropRatio).round();

      // Вычисляем координаты центрального квадрата
      final int centerX = (originalImage.width - cropSize) ~/ 2;
      final int centerY = (originalImage.height - cropSize) ~/ 2;

      print('Размер прямоугольника для обрезки: ${cropSize}x${cropSize}');
      print('Координаты обрезки: x=$centerX, y=$centerY');

      // Обрезаем изображение до большего размера
      final img.Image croppedLargeImage = img.copyCrop(
        originalImage,
        x: centerX,
        y: centerY,
        width: cropSize,
        height: cropSize,
      );

      // Масштабируем до целевого размера 224x224 (модель требует именно этот размер)
      final img.Image resizedImage = img.copyResize(
        croppedLargeImage,
        width: MODEL_SIZE,
        height: MODEL_SIZE,
        interpolation: img.Interpolation.cubic,
      );

      // Получаем временную директорию приложения
      final Directory tempDir = await getTemporaryDirectory();
      final String timestamp = DateTime.now().millisecondsSinceEpoch.toString();
      final String croppedPath = '${tempDir.path}/cropped_prop_$timestamp.jpg';
      final File croppedFile = File(croppedPath);

      // Сохраняем обрезанное изображение
      await croppedFile.writeAsBytes(img.encodeJpg(resizedImage, quality: 95));

      print('Пропорционально обрезанное изображение создано: $croppedPath');
      print('Пропорция обрезки: $cropRatio (от ширины оригинала)');
      print(
          'Масштабирование: из ${cropSize}x${cropSize} в ${MODEL_SIZE}x${MODEL_SIZE}');

      return croppedPath;
    } catch (e) {
      print('Ошибка при пропорциональной обрезке: $e');
      return imagePath;
    }
  }

  /// Обрезает точно 224×224 пикселей из центра изображения
  static Future<String> cropExactPixels(String imagePath) async {
    try {
      final File imageFile = File(imagePath);
      final Uint8List imageBytes = await imageFile.readAsBytes();
      final img.Image? originalImage = img.decodeImage(imageBytes);

      if (originalImage == null) {
        throw Exception('Не удалось декодировать изображение');
      }

      print(
          'Оригинальное изображение: ${originalImage.width}x${originalImage.height}');

      // Проверяем, достаточно ли большое изображение
      if (originalImage.width < MODEL_SIZE ||
          originalImage.height < MODEL_SIZE) {
        print(
            'ПРЕДУПРЕЖДЕНИЕ: Изображение меньше требуемого размера ${MODEL_SIZE}x${MODEL_SIZE}!');

        // Увеличиваем изображение, если оно меньше требуемого размера
        final img.Image resizedImage = img.copyResize(
          originalImage,
          width: originalImage.width < MODEL_SIZE
              ? MODEL_SIZE
              : originalImage.width,
          height: originalImage.height < MODEL_SIZE
              ? MODEL_SIZE
              : originalImage.height,
          interpolation: img.Interpolation.cubic,
        );

        // Используем увеличенное изображение для дальнейшей обработки
        // В библиотеке image 4.x нет метода dispose()
        final updatedImage = resizedImage;
        print(
            'Изображение увеличено до: ${updatedImage.width}x${updatedImage.height}');
        return _completeCropExact(updatedImage, imagePath);
      }

      return _completeCropExact(originalImage, imagePath);
    } catch (e) {
      print('Ошибка при точной обрезке: $e');
      return imagePath;
    }
  }

  /// Завершает процесс обрезки и сохраняет результат
  static Future<String> _completeCropExact(
      img.Image image, String originalPath) async {
    try {
      // Вычисляем координаты центрального квадрата 224×224 пикселей
      final int centerX = (image.width - MODEL_SIZE) ~/ 2;
      final int centerY = (image.height - MODEL_SIZE) ~/ 2;

      print(
          'Координаты для точного вырезания ${MODEL_SIZE}x${MODEL_SIZE}: x=$centerX, y=$centerY');

      // Обрезаем изображение без изменения размера
      final img.Image croppedImage = img.copyCrop(
        image,
        x: centerX,
        y: centerY,
        width: MODEL_SIZE,
        height: MODEL_SIZE,
      );

      // Проверяем размер обрезанного изображения
      print(
          'Размер обрезанного изображения: ${croppedImage.width}x${croppedImage.height}');
      if (croppedImage.width != MODEL_SIZE ||
          croppedImage.height != MODEL_SIZE) {
        print(
            'ВНИМАНИЕ: Размер не соответствует целевому ${MODEL_SIZE}x${MODEL_SIZE}!');
      }

      // Получаем временную директорию приложения
      final Directory tempDir = await getTemporaryDirectory();
      final String timestamp = DateTime.now().millisecondsSinceEpoch.toString();
      final String croppedPath = '${tempDir.path}/exact_crop_$timestamp.jpg';
      final File croppedFile = File(croppedPath);

      // Сохраняем обрезанное изображение
      await croppedFile.writeAsBytes(img.encodeJpg(croppedImage, quality: 95));

      print('Точно вырезанное изображение создано: $croppedPath');

      return croppedPath;
    } catch (e) {
      print('Ошибка при завершении точной обрезки: $e');
      return originalPath;
    }
  }

  /// Выводит информацию о размерах изображения
  static Future<Map<String, dynamic>> getImageInfo(String imagePath) async {
    try {
      final File imageFile = File(imagePath);
      final Uint8List imageBytes = await imageFile.readAsBytes();
      final img.Image? image = img.decodeImage(imageBytes);

      if (image == null) {
        return {
          'error': 'Не удалось декодировать изображение',
          'path': imagePath
        };
      }

      return {
        'width': image.width,
        'height': image.height,
        'path': imagePath,
        'aspectRatio': image.width / image.height,
        'fileSize': await imageFile.length(),
      };
    } catch (e) {
      return {'error': e.toString(), 'path': imagePath};
    }
  }
}
