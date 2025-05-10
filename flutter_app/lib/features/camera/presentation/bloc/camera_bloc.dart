import 'dart:io';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:camera/camera.dart';
import 'package:equatable/equatable.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;

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

  CameraBloc(this.cameras) : super(CameraInitial()) {
    on<CameraInitializeEvent>(_onInitialize);
    on<CaptureImageEvent>(_onCaptureImage);
    on<SwitchCameraEvent>(_onSwitchCamera);
    on<ResetCameraEvent>(_onResetCamera);
    on<CameraDisposeEvent>(_onDispose);
    on<ZoomChangedEvent>(_onZoomChanged);

    add(CameraInitializeEvent());
  }

  Future<void> _onInitialize(
    CameraInitializeEvent event,
    Emitter<CameraState> emit,
  ) async {
    if (_isDisposed) return;

    try {
      if (cameras.isEmpty) {
        emit(CameraError('Камера недоступна'));
        return;
      }

      if (_controller != null) {
        await _controller!.dispose();
        _controller = null;
      }

      _controller = CameraController(
        cameras[_selectedCameraIndex],
        ResolutionPreset.high,
        enableAudio: false,
      );

      await _controller!.initialize();

      if (_isDisposed) {
        await _controller!.dispose();
        _controller = null;
        return;
      }

      _minZoom = await _controller!.getMinZoomLevel();
      _maxZoom = await _controller!.getMaxZoomLevel();
      _currentZoom = _minZoom;

      emit(CameraReady(
        _controller!,
        currentZoom: _currentZoom,
        minZoom: _minZoom,
        maxZoom: _maxZoom,
      ));
    } catch (e) {
      if (!_isDisposed) {
        emit(CameraError('Ошибка инициализации камеры: ${e.toString()}'));
      }
    }
  }

  Future<void> _onCaptureImage(
    CaptureImageEvent event,
    Emitter<CameraState> emit,
  ) async {
    if (_isDisposed) return;

    try {
      if (_controller == null || !_controller!.value.isInitialized) {
        emit(CameraError('Камера не инициализирована'));
        return;
      }

      final XFile image = await _controller!.takePicture();
      final String imagePath = image.path;

      emit(ImageCaptured(imagePath));
    } catch (e) {
      if (!_isDisposed) {
        emit(CameraError('Ошибка при съемке: ${e.toString()}'));
      }
    }
  }

  Future<void> _onSwitchCamera(
    SwitchCameraEvent event,
    Emitter<CameraState> emit,
  ) async {
    if (_isDisposed) return;

    try {
      if (_controller != null) {
        await _controller!.dispose();
        _controller = null;
      }

      _selectedCameraIndex = (_selectedCameraIndex + 1) % cameras.length;
      _controller = CameraController(
        cameras[_selectedCameraIndex],
        ResolutionPreset.high,
        enableAudio: false,
      );

      await _controller!.initialize();

      if (_isDisposed) {
        await _controller!.dispose();
        _controller = null;
        return;
      }

      _minZoom = await _controller!.getMinZoomLevel();
      _maxZoom = await _controller!.getMaxZoomLevel();
      _currentZoom = _minZoom;

      emit(CameraReady(
        _controller!,
        currentZoom: _currentZoom,
        minZoom: _minZoom,
        maxZoom: _maxZoom,
      ));
    } catch (e) {
      if (!_isDisposed) {
        emit(CameraError('Ошибка при переключении камеры: ${e.toString()}'));
      }
    }
  }

  Future<void> _onResetCamera(
    ResetCameraEvent event,
    Emitter<CameraState> emit,
  ) async {
    if (_isDisposed) return;

    try {
      if (_controller != null) {
        await _controller!.dispose();
        _controller = null;
      }

      _controller = CameraController(
        cameras[_selectedCameraIndex],
        ResolutionPreset.high,
        enableAudio: false,
      );

      await _controller!.initialize();

      if (_isDisposed) {
        await _controller!.dispose();
        _controller = null;
        return;
      }

      _minZoom = await _controller!.getMinZoomLevel();
      _maxZoom = await _controller!.getMaxZoomLevel();
      _currentZoom = _minZoom;

      emit(CameraReady(
        _controller!,
        currentZoom: _currentZoom,
        minZoom: _minZoom,
        maxZoom: _maxZoom,
      ));
    } catch (e) {
      if (!_isDisposed) {
        emit(CameraError('Ошибка при сбросе камеры: ${e.toString()}'));
      }
    }
  }

  Future<void> _onZoomChanged(
    ZoomChangedEvent event,
    Emitter<CameraState> emit,
  ) async {
    if (_isDisposed) return;

    try {
      if (_controller == null || !_controller!.value.isInitialized) {
        return;
      }

      _currentZoom = event.zoomLevel.clamp(_minZoom, _maxZoom);
      await _controller!.setZoomLevel(_currentZoom);

      if (state is CameraReady && !_isDisposed) {
        emit(CameraReady(
          _controller!,
          currentZoom: _currentZoom,
          minZoom: _minZoom,
          maxZoom: _maxZoom,
        ));
      }
    } catch (e) {
      // Игнорируем ошибки при изменении зума
    }
  }

  Future<void> _onDispose(
    CameraDisposeEvent event,
    Emitter<CameraState> emit,
  ) async {
    _isDisposed = true;
    if (_controller != null) {
      await _controller!.dispose();
      _controller = null;
    }
    emit(CameraInitial());
  }

  @override
  Future<void> close() async {
    _isDisposed = true;
    if (_controller != null) {
      await _controller!.dispose();
      _controller = null;
    }
    return super.close();
  }
}
