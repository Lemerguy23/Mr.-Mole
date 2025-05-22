import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:camera/camera.dart';
import 'package:equatable/equatable.dart';
import 'dart:ui';
import 'dart:io';
import 'dart:typed_data';
import 'dart:async';
import 'package:image/image.dart' as img;
import 'package:mr_mole/core/utils/image_processor.dart';

part 'camera_event.dart';
part 'camera_state.dart';

class CameraBloc extends Bloc<CameraEvent, CameraState> {
  final List<CameraDescription> cameras;
  CameraController? _controller;
  int _selectedCameraIndex = 0;
  double _currentZoom = 1.0;
  double _minZoom = 1.0;
  double _maxZoom = 1.0;
  bool _isDisposed = false;
  bool _isFlashOn = false;

  // Прямоугольник для захвата - центральная область камеры
  late Rect _captureRect;

  CameraBloc(this.cameras) : super(CameraInitial()) {
    on<CameraInitializeEvent>(_onInitialize);
    on<CaptureImageEvent>(_onCaptureImage);
    on<SwitchCameraEvent>(_onSwitchCamera);
    on<ResetCameraEvent>(_onResetCamera);
    on<CameraDisposeEvent>(_onDispose);
    on<ZoomChangedEvent>(_onZoomChanged);
    on<ToggleFlashEvent>(_onToggleFlash);
    on<ToggleInstructionEvent>(_onToggleInstruction);
  }

  bool get _isCameraReady =>
      _controller != null && _controller!.value.isInitialized;

  Future<void> _initializeCamera(Emitter<CameraState> emit) async {
    if (_isDisposed) return;

    try {
      if (_isCameraReady) {
        emit(CameraReady(
          _controller!,
          currentZoom: _currentZoom,
          minZoom: _minZoom,
          maxZoom: _maxZoom,
          isFlashOn: _isFlashOn,
          captureRect: _captureRect,
          showInstruction: false,
        ));
        return;
      }

      emit(CameraLoading());

      if (cameras.isEmpty) {
        emit(CameraError('Камера недоступна'));
        return;
      }

      // Проверяем индекс камеры
      if (_selectedCameraIndex >= cameras.length) {
        _selectedCameraIndex = 0;
      }

      // Лучше начать с задней камеры
      if (_selectedCameraIndex == 0 && cameras.length > 1) {
        for (int i = 0; i < cameras.length; i++) {
          if (cameras[i].lensDirection == CameraLensDirection.back) {
            _selectedCameraIndex = i;
            break;
          }
        }
      }

      // Создаем и инициализируем контроллер
      await _disposeCamera(); // Убедимся, что предыдущий контроллер освобожден

      _controller = CameraController(
        cameras[_selectedCameraIndex],
        ResolutionPreset.max, // Используем максимальное разрешение
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.yuv420
            : ImageFormatGroup.bgra8888,
      );

      print('Инициализация камеры...');

      // Увеличиваем таймаут для инициализации камеры до 30 секунд
      try {
        await _controller!.initialize().timeout(
          const Duration(seconds: 30), // Увеличено с 10 до 30 секунд
          onTimeout: () {
            print('Превышено время ожидания инициализации камеры (30 секунд)');
            throw Exception(
                'Превышено время ожидания инициализации камеры. Пожалуйста, перезапустите приложение или проверьте доступность камеры.');
          },
        );
        print('Камера инициализирована успешно');
      } catch (e) {
        print('Ошибка при инициализации камеры: $e');

        // Попробуем инициализировать с более низким разрешением как запасной вариант
        if (!_isDisposed && _controller != null) {
          try {
            await _disposeCamera();

            print(
                'Пробуем инициализировать камеру с более низким разрешением...');
            _controller = CameraController(
              cameras[_selectedCameraIndex],
              ResolutionPreset.medium, // Более низкое разрешение
              enableAudio: false,
            );

            await _controller!.initialize();
            print('Камера инициализирована с более низким разрешением');
          } catch (fallbackError) {
            print(
                'Не удалось инициализировать камеру даже с более низким разрешением: $fallbackError');
            await _disposeCamera();
            if (!_isDisposed) {
              emit(CameraError(
                  'Не удалось инициализировать камеру. Пожалуйста, проверьте разрешения и перезапустите приложение.'));
            }
            return;
          }
        } else {
          if (!_isDisposed) {
            emit(CameraError('Ошибка инициализации камеры: ${e.toString()}'));
          }
          return;
        }
      }

      if (_isDisposed) {
        await _disposeCamera();
        return;
      }

      // Запускаем поток изображений для более плавного предпросмотра
      try {
        await _controller!.startImageStream((image) {
          // Поток запущен
        });
      } catch (e) {
        print('Ошибка при запуске потока изображений: $e');
        // Продолжаем работу даже если поток не запустился
      }

      // Установка параметров зума после инициализации
      try {
        _minZoom = await _controller!.getMinZoomLevel();
        _maxZoom = await _controller!.getMaxZoomLevel();
        _currentZoom = _minZoom;

        print(
            'Параметры зума: мин=$_minZoom, макс=$_maxZoom, текущий=$_currentZoom');
        print('Поддерживается ли зум: ${(_maxZoom > _minZoom) ? 'Да' : 'Нет'}');

        // Пробная установка зума для проверки работоспособности
        if (_maxZoom > _minZoom) {
          try {
            // Пробуем установить минимальный зум просто для проверки
            await _controller!.setZoomLevel(_minZoom);
            print('Тестовая установка зума успешна, зум должен работать');
          } catch (testZoomError) {
            print('Ошибка при тестовой установке зума: $testZoomError');
            print('Камера может не поддерживать программное управление зумом');
          }
        }
      } catch (e) {
        print('Ошибка при получении параметров зума: $e');
        print('Используем значения по умолчанию');
        // Используем значения по умолчанию
        _minZoom = 1.0;
        _maxZoom = 2.0;
        _currentZoom = 1.0;
      }

      try {
        // Отключаем вспышку по умолчанию
        await _controller!.setFlashMode(FlashMode.off);
        _isFlashOn = false;
      } catch (e) {
        print('Ошибка при установке режима вспышки: $e');
      }

      // Устанавливаем прямоугольник захвата в центр изображения
      // Он будет точно соответствовать 224×224 пикселям оригинального изображения
      if (_controller!.value.previewSize != null) {
        final previewSize = _controller!.value.previewSize!;

        // Вычисляем координаты центрального квадрата 224×224 пикселя
        final double left = (previewSize.width - 224) / 2;
        final double top = (previewSize.height - 224) / 2;

        _captureRect = Rect.fromLTWH(left, top, 224, 224);
        print('Установлен прямоугольник захвата: $_captureRect');
      } else {
        // Если размеры превью недоступны, используем стандартный прямоугольник
        _captureRect = const Rect.fromLTWH(0, 0, 224, 224);
        print('Установлен стандартный прямоугольник захвата');
      }

      emit(CameraReady(
        _controller!,
        currentZoom: _currentZoom,
        minZoom: _minZoom,
        maxZoom: _maxZoom,
        isFlashOn: _isFlashOn,
        captureRect: _captureRect,
        showInstruction: false,
      ));
    } catch (e) {
      print('Общая ошибка инициализации камеры: $e');
      if (!_isDisposed) {
        emit(CameraError('Ошибка инициализации камеры: ${e.toString()}'));
      }
    }
  }

  Future<void> _onInitialize(
    CameraInitializeEvent event,
    Emitter<CameraState> emit,
  ) async {
    await _initializeCamera(emit);
  }

  Future<void> _onCaptureImage(
    CaptureImageEvent event,
    Emitter<CameraState> emit,
  ) async {
    if (_isDisposed) return;

    try {
      if (!_isCameraReady) {
        emit(CameraError('Камера не инициализирована'));
        return;
      }

      print('Начало захвата изображения...');

      // Останавливаем стрим перед захватом для повышения производительности
      if (_controller!.value.isStreamingImages) {
        await _controller!.stopImageStream();
      }

      final XFile image = await _controller!.takePicture();
      print('Изображение захвачено: ${image.path}');

      // Получаем информацию о размерах оригинального изображения
      final imageInfo = await ImageProcessor.getImageInfo(image.path);
      print('Информация об оригинальном изображении: $imageInfo');

      // Используем новый метод для точного вырезания 224×224 пикселей
      final String croppedPath =
          await ImageProcessor.cropExactPixels(image.path);
      print('Обрезанное изображение готово: $croppedPath');

      // Получаем информацию о размерах обрезанного изображения
      final croppedInfo = await ImageProcessor.getImageInfo(croppedPath);
      print('Информация об обрезанном изображении: $croppedInfo');

      print(
          'Путь к обрезанному изображению, передаваемый в AnalysisScreen: $croppedPath');
      emit(ImageCaptured(croppedPath, captureRect: _captureRect));
    } catch (e) {
      print('Ошибка при съемке: $e');
      if (!_isDisposed) {
        emit(CameraError('Ошибка при съемке: ${e.toString()}'));
      }
    }
  }

  Future<void> _reinitializeCamera(Emitter<CameraState> emit,
      {bool switchCamera = false}) async {
    if (_isDisposed) return;

    try {
      await _disposeCamera();

      if (switchCamera) {
        _selectedCameraIndex = (_selectedCameraIndex + 1) % cameras.length;
      }

      await _initializeCamera(emit);
    } catch (e) {
      if (!_isDisposed) {
        emit(CameraError(
            'Ошибка при переинициализации камеры: ${e.toString()}'));
      }
    }
  }

  Future<void> _onSwitchCamera(
    SwitchCameraEvent event,
    Emitter<CameraState> emit,
  ) async {
    await _reinitializeCamera(emit, switchCamera: true);
  }

  Future<void> _onResetCamera(
    ResetCameraEvent event,
    Emitter<CameraState> emit,
  ) async {
    await _reinitializeCamera(emit);
  }

  Future<void> _onZoomChanged(
    ZoomChangedEvent event,
    Emitter<CameraState> emit,
  ) async {
    if (_isDisposed || !_isCameraReady) return;

    try {
      // Сохраняем предыдущее значение для отладки
      final double previousZoom = _currentZoom;

      // Устанавливаем новое значение зума
      _currentZoom = event.zoomLevel.clamp(_minZoom, _maxZoom);

      // Логируем для отладки
      print('---ZOOM---');
      print('Запрошено изменение зума: $previousZoom -> $_currentZoom');
      print(
          'Контроллер камеры: ${_controller != null ? "Инициализирован" : "NULL"}');
      if (_controller != null) {
        print('Камера готова: ${_controller!.value.isInitialized}');
      }

      // Проверяем, поддерживается ли зум
      if (_maxZoom <= _minZoom) {
        print('ОШИБКА: Зум не поддерживается (мин=$_minZoom, макс=$_maxZoom)');
        return;
      }

      // Создаем Future с таймаутом для установки зума
      try {
        print('Попытка установить зум на $_currentZoom...');
        await _controller!.setZoomLevel(_currentZoom).timeout(
          const Duration(seconds: 2), // Увеличиваем таймаут
          onTimeout: () {
            print(
                'ОШИБКА: Таймаут при установке зума $_currentZoom (2 секунды)');
            throw TimeoutException('Таймаут установки зума');
          },
        );
        print('Зум успешно установлен на: $_currentZoom');
      } catch (zoomError) {
        print('ОШИБКА при установке зума: $zoomError');
        print('Тип ошибки: ${zoomError.runtimeType}');

        if (zoomError.toString().contains('CameraException')) {
          print(
              'Это ошибка CameraException - возможно, камера не поддерживает зум');
        }

        // Повторная попытка с немного измененным значением
        try {
          final double adjustedZoom = _currentZoom - 0.1;
          if (adjustedZoom >= _minZoom) {
            print('Повторная попытка с меньшим зумом: $adjustedZoom');
            await _controller!.setZoomLevel(adjustedZoom);
            print('Удалось установить зум после корректировки: $adjustedZoom');
            _currentZoom = adjustedZoom;
          } else {
            print(
                'Не удалось скорректировать зум: минимальное значение $_minZoom');
          }
        } catch (retryError) {
          print('Повторная попытка также не удалась: $retryError');
          print('Тип ошибки при повторной попытке: ${retryError.runtimeType}');
        }
      }
      print('---END ZOOM---');

      // Если мы дошли сюда, обновляем состояние
      if (state is CameraReady && !_isDisposed) {
        final currentState = state as CameraReady;
        emit(currentState.copyWith(
          currentZoom: _currentZoom,
        ));
      }
    } catch (e) {
      print('Общая ошибка при изменении зума: $e');
      print('Тип общей ошибки: ${e.runtimeType}');
      // Не меняем состояние при ошибке, чтобы не блокировать UI
    }
  }

  Future<void> _onToggleFlash(
    ToggleFlashEvent event,
    Emitter<CameraState> emit,
  ) async {
    if (_isDisposed || !_isCameraReady) return;

    try {
      final newFlashMode = _isFlashOn ? FlashMode.off : FlashMode.torch;
      await _controller!.setFlashMode(newFlashMode);
      _isFlashOn = !_isFlashOn;

      if (state is CameraReady) {
        final currentState = state as CameraReady;
        emit(currentState.copyWith(isFlashOn: _isFlashOn));
      }
    } catch (e) {
      // Игнорируем ошибки при переключении вспышки
      print('Ошибка при переключении вспышки: $e');
    }
  }

  void _onToggleInstruction(
    ToggleInstructionEvent event,
    Emitter<CameraState> emit,
  ) {
    if (state is CameraReady) {
      final currentState = state as CameraReady;
      emit(currentState.copyWith(
        showInstruction: !currentState.showInstruction,
      ));
    }
  }

  Future<void> _disposeCamera() async {
    if (_controller != null) {
      try {
        if (_controller!.value.isInitialized) {
          try {
            if (_controller!.value.isStreamingImages) {
              await _controller!.stopImageStream();
            }
          } catch (e) {
            print('Ошибка при остановке потока изображений: $e');
          }

          try {
            await _controller!.dispose();
          } catch (e) {
            print('Ошибка при освобождении контроллера камеры: $e');
          }
        }
        _controller = null;
      } catch (e) {
        // Игнорируем ошибки при освобождении камеры
        print('Ошибка при освобождении камеры: $e');
      }
    }
  }

  Future<void> _onDispose(
    CameraDisposeEvent event,
    Emitter<CameraState> emit,
  ) async {
    _isDisposed = true;
    await _disposeCamera();
    emit(CameraInitial());
  }

  @override
  Future<void> close() async {
    _isDisposed = true;
    await _disposeCamera();
    return super.close();
  }
}
