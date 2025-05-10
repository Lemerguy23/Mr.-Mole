import 'package:camera/camera.dart';

class CameraHandler {
  static Future<List<CameraDescription>> getAvailableCameras() async {
    try {
      final cameras = await availableCameras();
      return cameras;
    } catch (e) {
      return [];
    }
  }
}
