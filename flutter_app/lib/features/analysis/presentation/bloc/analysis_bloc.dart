import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:image/image.dart' as img;
import 'package:equatable/equatable.dart';
import 'dart:async';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:mr_mole/core/utils/notification.dart';
import 'package:mr_mole/core/utils/model_cache.dart';

part 'analysis_event.dart';
part 'analysis_state.dart';

class AnalysisBloc extends Bloc<AnalysisEvent, AnalysisState> {
  final String imagePath;
  final NotificationService notificationService;
  Interpreter? _interpreter;
  bool _isDisposed = false;

  AnalysisBloc({
    required this.imagePath,
    required this.notificationService,
  }) : super(AnalysisInitial()) {
    on<AnalyzeImageEvent>(_onAnalyzeImage);
    on<SaveResultEvent>(_onSaveResult);
  }

  Future<void> _onAnalyzeImage(
    AnalyzeImageEvent event,
    Emitter<AnalysisState> emit,
  ) async {
    if (_isDisposed) return;

    try {
      emit(AnalysisLoading());
      print('Начинаем загрузку модели...');

      _interpreter = await ModelCache.getInstance('assets/model.tflite');
      if (_isDisposed) {
        ModelCache.release();
        return;
      }

      print(
          'Результат загрузки модели: ${_interpreter != null ? 'успешно' : 'ошибка'}');

      if (_interpreter == null) {
        emit(AnalysisError('Модель не загружена'));
        return;
      }

      final imageFile = File(imagePath);
      if (!await imageFile.exists()) {
        emit(AnalysisError('Изображение не найдено'));
        return;
      }

      final imageBytes = await imageFile.readAsBytes();
      final image = img.decodeImage(imageBytes);
      if (image == null) {
        emit(AnalysisError('Не удалось декодировать изображение'));
        return;
      }

      if (_isDisposed) {
        ModelCache.release();
        return;
      }

      print('Изображение успешно загружено и декодировано');

      final resizedImage = img.copyResize(
        image,
        width: 224,
        height: 224,
      );

      final inputShape = _interpreter!.getInputTensor(0).shape;
      final outputShape = _interpreter!.getOutputTensor(0).shape;

      print('Размеры входного тензора: $inputShape');
      print('Размеры выходного тензора: $outputShape');

      final inputBuffer =
          Float32List(inputShape[1] * inputShape[2] * inputShape[3]);
      final outputBuffer = Float32List(outputShape[1]);

      var index = 0;
      for (var y = 0; y < resizedImage.height; y++) {
        for (var x = 0; x < resizedImage.width; x++) {
          final pixel = resizedImage.getPixel(x, y);
          final r = pixel.r;
          final g = pixel.g;
          final b = pixel.b;
          inputBuffer[index++] = r / 255.0;
          inputBuffer[index++] = g / 255.0;
          inputBuffer[index++] = b / 255.0;
        }
      }

      if (_isDisposed) {
        ModelCache.release();
        return;
      }

      print('Запускаем инференс модели...');
      _interpreter!.run(inputBuffer.buffer, outputBuffer.buffer);
      print('Инференс завершен');

      if (_isDisposed) {
        ModelCache.release();
        return;
      }

      final result = outputBuffer[0] > 0.5
          ? 'Обнаружены признаки меланомы. Рекомендуется обратиться к врачу.'
          : 'Признаков меланомы не обнаружено.';

      emit(AnalysisSuccess(result));
    } catch (e, stackTrace) {
      print('Ошибка при анализе: $e');
      print('Stack trace: $stackTrace');
      if (!_isDisposed) {
        emit(AnalysisError('Ошибка при анализе: ${e.toString()}'));
      }
    }
  }

  Future<void> _onSaveResult(
    SaveResultEvent event,
    Emitter<AnalysisState> emit,
  ) async {
    if (_isDisposed) return;

    try {
      if (state is AnalysisSuccess) {
        final result = (state as AnalysisSuccess).result;
        await notificationService.showNotification(
          title: 'Результат анализа',
          body: result,
        );
      }
    } catch (e) {
      if (!_isDisposed) {
        emit(
            AnalysisError('Ошибка при сохранении результата: ${e.toString()}'));
      }
    }
  }

  @override
  Future<void> close() async {
    _isDisposed = true;
    ModelCache.release();
    return super.close();
  }
}
