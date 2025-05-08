import 'package:camera/camera.dart';

class CameraHandler {
  /// Получает список доступных камер устройства
  static Future<List<CameraDescription>> getAvailableCameras() async {
    try {
      final cameras = await availableCameras();
      return cameras;
    } catch (e) {
      print('Ошибка получения камер: $e');
      return [];
    }
  }
}
