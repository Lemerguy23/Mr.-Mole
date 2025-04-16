import 'package:flutter/material.dart';

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          onPressed: () {
            Navigator.pop(context);
          },
          icon: Icon(Icons.arrow_back, size: 32, color: Colors.white),
        ),
        backgroundColor: Color(0xFF1b264a),
        title: const Text(
          'Настройки',
          style: TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
        centerTitle: true,
      ),
      body: ListView(
        children: [
          ListTile(
            leading: Icon(
              Icons.photo_size_select_large,
              color: Color(0xFF1b264a),
            ),
            title: Text('Размер фото'),
            subtitle: Text('Настройка размера отображаемого фото'),
            onTap: () {
              // Здесь будет логика изменения размера
            },
          ),
          ListTile(
            leading: Icon(Icons.camera_alt, color: Color(0xFF1b264a)),
            title: Text('Камера'),
            subtitle: Text('Настройки камеры'),
            onTap: () {
              // Здесь будет логика настроек камеры
            },
          ),
          ListTile(
            leading: Icon(Icons.storage, color: Color(0xFF1b264a)),
            title: Text('Хранение'),
            subtitle: Text('Настройки хранения фотографий'),
            onTap: () {
              // Здесь будет логика настроек хранения
            },
          ),
        ],
      ),
    );
  }
}
