import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class ModelCache {
  static Interpreter? _interpreter;
  static int _usersCount = 0;
  static bool _isInitializing = false;
  static const int _maxRetries = 3;
  static const Duration _retryDelay = Duration(seconds: 1);

  static void _resetState() {
    _interpreter = null;
    _usersCount = 0;
    _isInitializing = false;
  }

  static Future<Interpreter?> getInstance(String assetPath) async {
    print('ModelCache: Запрос на получение интерпретатора для $assetPath');

    if (_interpreter != null) {
      print('ModelCache: Используем существующий интерпретатор');
      _usersCount++;
      return _interpreter;
    }

    if (_isInitializing) {
      print('ModelCache: Ожидаем завершения инициализации');
      await Future.delayed(_retryDelay);
      if (_isInitializing) {
        print('ModelCache: Сбрасываем состояние из-за зависшей инициализации');
        _resetState();
      }
      return getInstance(assetPath);
    }

    _isInitializing = true;
    int retryCount = 0;

    while (retryCount < _maxRetries) {
      try {
        print(
            'ModelCache: Попытка загрузки модели (${retryCount + 1}/$_maxRetries)');

        try {
          print('ModelCache: Пробуем загрузить из assets');
          final modelData = await rootBundle.load(assetPath);
          print('ModelCache: Размер модели: ${modelData.lengthInBytes} байт');

          if (modelData.lengthInBytes == 0) {
            throw Exception('Файл модели пуст');
          }

          final modelBytes = Uint8List.fromList(modelData.buffer.asUint8List());
          _interpreter = await Interpreter.fromBuffer(modelBytes);
          print('ModelCache: Модель успешно загружена из assets');
        } catch (e) {
          print('ModelCache: Ошибка загрузки из assets: $e');
          print('ModelCache: Пробуем загрузить из файловой системы');
          _interpreter = await Interpreter.fromAsset(assetPath);
          print('ModelCache: Модель успешно загружена из файловой системы');
        }

        if (_interpreter == null) {
          throw Exception('Не удалось загрузить модель');
        }

        if (!_interpreter!.isAllocated) {
          throw Exception('Модель не инициализирована');
        }

        print('ModelCache: Модель успешно инициализирована');
        _usersCount++;
        _isInitializing = false;
        return _interpreter;
      } catch (e) {
        print('ModelCache: Ошибка при загрузке модели: $e');
        retryCount++;

        if (retryCount < _maxRetries) {
          print('ModelCache: Повторная попытка через $_retryDelay');
          await Future.delayed(_retryDelay);
        } else {
          print('ModelCache: Все попытки загрузки модели исчерпаны');
          _resetState();
        }
      }
    }

    return null;
  }

  static void release() {
    print('ModelCache: Запрос на освобождение интерпретатора');
    _usersCount--;
    if (_usersCount <= 0 && _interpreter != null) {
      try {
        print('ModelCache: Закрываем интерпретатор');
        if (_interpreter!.isAllocated) {
          _interpreter!.close();
        }
        print('ModelCache: Интерпретатор успешно закрыт');
      } catch (e) {
        print('ModelCache: Ошибка при закрытии интерпретатора: $e');
      } finally {
        _resetState();
      }
    } else {
      print(
          'ModelCache: Интерпретатор все еще используется ($_usersCount пользователей)');
    }
  }
}
