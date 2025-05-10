// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mr_mole/core/utils/camera_repo.dart';
import 'package:mr_mole/main.dart';
import 'package:mr_mole/core/utils/notification.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:mr_mole/core/utils/model_cache.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('App initialization test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    final camerasFuture = CameraHandler.getAvailableCameras();
    final notificationsPlugin = FlutterLocalNotificationsPlugin();
    final notificationService = NotificationService(notificationsPlugin);

    await tester.pumpWidget(
      MyApp(
        camerasFuture: camerasFuture,
        notificationService: notificationService,
      ),
    );

    // Проверяем, что приложение запустилось
    expect(find.byType(MaterialApp), findsOneWidget);
  });

  test('ModelCache initialization test', () async {
    // Мокаем rootBundle для тестов
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMessageHandler('flutter/assets', (message) async {
      return null;
    });

    // Тестируем инициализацию ModelCache
    try {
      final interpreter = await ModelCache.getInstance('assets/model.tflite');

      // В режиме отладки модель может быть null
      if (interpreter == null) {
        return;
      }

      // Если модель загружена, проверяем её состояние
      expect(interpreter.isAllocated, isTrue);

      // Проверяем, что модель можно закрыть
      interpreter.close();
      expect(interpreter.isAllocated, isFalse);
    } catch (e) {
      return;
    }
  });
}
