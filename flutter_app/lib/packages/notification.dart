import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'dart:async';

class NotificationService {
  static final NotificationService _instance = NotificationService._internal();
  factory NotificationService() => _instance;
  NotificationService._internal();

  final FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
      FlutterLocalNotificationsPlugin();
  DateTime? lastFunctionCallTime;
  final Duration delay = Duration(minutes: 1);
  Timer? _timer;

  Future<void> init() async {
    const AndroidInitializationSettings initializationSettingsAndroid =
        AndroidInitializationSettings('app_icon');
    const InitializationSettings initializationSettings =
        InitializationSettings(android: initializationSettingsAndroid);

    // Создаем канал уведомлений
    await _createNotificationChannel();

    await flutterLocalNotificationsPlugin.initialize(
      initializationSettings,
      onDidReceiveNotificationResponse: (details) {},
    );

    _timer = Timer.periodic(Duration(minutes: 1), _checkTimeSinceLastCall);
  }

  Future<void> _createNotificationChannel() async {
    const AndroidNotificationChannel channel = AndroidNotificationChannel(
      'your_channel_id', // Должно совпадать с channelId в showNotification
      'your_channel_name',
      description: 'your_channel_description',
      importance: Importance.max,
      playSound: true,
      showBadge: true,
    );

    await flutterLocalNotificationsPlugin
        .resolvePlatformSpecificImplementation<
          AndroidFlutterLocalNotificationsPlugin
        >()
        ?.createNotificationChannel(channel);
  }

  void setTime() {
    lastFunctionCallTime = DateTime.now();
    print('Function called at: $lastFunctionCallTime');
  }

  void _checkTimeSinceLastCall(Timer timer) {
    if (lastFunctionCallTime == null) return;

    final elapsed = DateTime.now().difference(lastFunctionCallTime!);
    print('Elapsed time: $elapsed');

    if (elapsed >= delay) {
      _showNotification();
      lastFunctionCallTime = null;
    }
  }

  Future<void> _showNotification() async {
    try {
      const AndroidNotificationDetails
      androidDetails = AndroidNotificationDetails(
        'your_channel_id', // Должно совпадать с channelId при создании канала
        'your_channel_name',
        channelDescription: 'your_channel_description',
        importance: Importance.max,
        priority: Priority.high,
        ticker: 'ticker',
        largeIcon: DrawableResourceAndroidBitmap('app_icon'),
        colorized: true,
        enableVibration: true,
        playSound: true,
      );

      const NotificationDetails platformDetails = NotificationDetails(
        android: androidDetails,
      );

      await flutterLocalNotificationsPlugin.show(
        0,
        'Напоминание',
        'Функция не вызывалась последние 2 минуты!',
        platformDetails,
      );
      print('Уведомление показано');
    } catch (e) {
      print('Ошибка при показе уведомления: $e');
    }
  }

  void dispose() {
    _timer?.cancel();
  }
}
