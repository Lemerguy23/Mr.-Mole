import 'package:mr_mole/pages/home.dart';
import 'package:flutter/material.dart';
import 'package:mr_mole/packages/notification.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final notificationService = NotificationService();
  await notificationService.init();

  runApp(MyApp(notificationService: notificationService));
}

class MyApp extends StatelessWidget {
  final NotificationService notificationService;

  const MyApp({super.key, required this.notificationService});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(fontFamily: 'Nata_Sans'),
      home: HomePage(notificationService: notificationService),
    );
  }
}
