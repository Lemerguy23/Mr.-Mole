import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:uuid/uuid.dart';
import 'package:mr_mole/features/home/domain/models/scan_history_item.dart';

class ScanHistoryRepository {
  static const String _historyKey = 'scan_history';
  final Uuid _uuid = const Uuid();

  // Получение всей истории сканирований
  Future<List<ScanHistoryItem>> getHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final String? historyJson = prefs.getString(_historyKey);

    if (historyJson == null || historyJson.isEmpty) {
      return [];
    }

    try {
      return ScanHistoryItem.decode(historyJson);
    } catch (e) {
      print('Ошибка при чтении истории: $e');
      return [];
    }
  }

  // Добавление нового элемента в историю
  Future<bool> addToHistory(String imagePath, String result) async {
    try {
      // Сохраняем копию изображения в локальном хранилище
      final String savedImagePath = await _saveImageToLocalStorage(imagePath);

      final historyItem = ScanHistoryItem(
        id: _uuid.v4(),
        imagePath: savedImagePath,
        result: result,
        timestamp: DateTime.now(),
      );

      final prefs = await SharedPreferences.getInstance();
      final List<ScanHistoryItem> currentHistory = await getHistory();

      // Добавляем новый элемент в начало списка
      currentHistory.insert(0, historyItem);

      // Ограничиваем историю последними 50 записями
      final List<ScanHistoryItem> trimmedHistory = currentHistory.length > 50
          ? currentHistory.sublist(0, 50)
          : currentHistory;

      // Сохраняем обновленную историю
      return await prefs.setString(
          _historyKey, ScanHistoryItem.encode(trimmedHistory));
    } catch (e) {
      print('Ошибка при добавлении в историю: $e');
      return false;
    }
  }

  // Удаление элемента из истории
  Future<bool> removeFromHistory(String id) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final List<ScanHistoryItem> currentHistory = await getHistory();

      // Находим удаляемый элемент
      final itemToRemove = currentHistory.firstWhere((item) => item.id == id);

      // Удаляем сохранённое изображение
      final imageFile = File(itemToRemove.imagePath);
      if (await imageFile.exists()) {
        await imageFile.delete();
      }

      // Удаляем элемент из списка
      currentHistory.removeWhere((item) => item.id == id);

      // Сохраняем обновленную историю
      return await prefs.setString(
          _historyKey, ScanHistoryItem.encode(currentHistory));
    } catch (e) {
      print('Ошибка при удалении из истории: $e');
      return false;
    }
  }

  // Очистка всей истории
  Future<bool> clearHistory() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final List<ScanHistoryItem> currentHistory = await getHistory();

      // Удаляем все сохранённые изображения
      for (var item in currentHistory) {
        final imageFile = File(item.imagePath);
        if (await imageFile.exists()) {
          await imageFile.delete();
        }
      }

      // Очищаем историю
      return await prefs.remove(_historyKey);
    } catch (e) {
      print('Ошибка при очистке истории: $e');
      return false;
    }
  }

  // Сохранение изображения в локальное хранилище
  Future<String> _saveImageToLocalStorage(String originalPath) async {
    try {
      final directory = await getApplicationDocumentsDirectory();
      final historyDir = Directory('${directory.path}/history_images');

      // Создаем директорию, если она не существует
      if (!await historyDir.exists()) {
        await historyDir.create(recursive: true);
      }

      final fileName = '${_uuid.v4()}.jpg';
      final destinationPath = '${historyDir.path}/$fileName';

      // Копируем файл в новое место
      await File(originalPath).copy(destinationPath);

      return destinationPath;
    } catch (e) {
      print('Ошибка при сохранении изображения: $e');
      // Возвращаем оригинальный путь в случае ошибки
      return originalPath;
    }
  }
}
