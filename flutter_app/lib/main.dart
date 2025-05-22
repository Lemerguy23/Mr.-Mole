import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:mr_mole/core/utils/camera_repo.dart';
import 'package:mr_mole/features/home/presentation/pages/home_page.dart';
import 'package:mr_mole/core/utils/notification.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter_native_splash/flutter_native_splash.dart';

void main() async {
  WidgetsBinding widgetsBinding = WidgetsFlutterBinding.ensureInitialized();
  FlutterNativeSplash.preserve(widgetsBinding: widgetsBinding);

  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  final camerasFuture = CameraHandler.getAvailableCameras();
  final notificationsPlugin = FlutterLocalNotificationsPlugin();
  final notificationService = NotificationService(notificationsPlugin);

  FlutterNativeSplash.remove();

  runApp(
    MyApp(
      camerasFuture: camerasFuture,
      notificationService: notificationService,
    ),
  );
}

class MyApp extends StatelessWidget {
  final Future<List<CameraDescription>> camerasFuture;
  final NotificationService notificationService;

  const MyApp({
    super.key,
    required this.camerasFuture,
    required this.notificationService,
  });

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Mr. Mole',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: FutureBuilder<List<CameraDescription>>(
        future: camerasFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Scaffold(
              body: Center(
                child: CircularProgressIndicator(),
              ),
            );
          }

          if (snapshot.hasError) {
            return Scaffold(
              body: Center(
                child: Text('Ошибка: ${snapshot.error}'),
              ),
            );
          }

          final cameras = snapshot.data ?? [];
          if (cameras.isEmpty) {
            return const Scaffold(
              body: Center(
                child: Text('Камера недоступна'),
              ),
            );
          }

          return WillPopScope(
            onWillPop: () async {
              return true;
            },
            child: HomePage(
              camerasFuture: camerasFuture,
              notificationService: notificationService,
            ),
          );
        },
      ),
    );
  }
}
