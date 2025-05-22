import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:mr_mole/features/analysis/presentation/bloc/analysis_bloc.dart';
import 'package:mr_mole/core/utils/notification.dart';
import 'package:mr_mole/features/home/data/repositories/scan_history_repository.dart';

class AnalysisScreen extends StatelessWidget {
  final String imagePath;
  final NotificationService notificationService;
  final VoidCallback onRetake;

  const AnalysisScreen({
    super.key,
    required this.imagePath,
    required this.notificationService,
    required this.onRetake,
  });

  void _handleBack(BuildContext context) {
    onRetake();
  }

  // Метод для возврата на главный экран
  void _navigateToHome(BuildContext context) {
    // Закрываем все экраны до главного (первого)
    Navigator.of(context).popUntil((route) => route.isFirst);
  }

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (context) => AnalysisBloc(
        imagePath: imagePath,
        notificationService: notificationService,
        historyRepository: ScanHistoryRepository(),
      )..add(AnalyzeImageEvent()),
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Анализ'),
          leading: IconButton(
            icon: const Icon(Icons.arrow_back),
            onPressed: () => _handleBack(context),
          ),
        ),
        body: BlocBuilder<AnalysisBloc, AnalysisState>(
          builder: (context, state) {
            if (state is AnalysisInitial) {
              return const Center(
                child: CircularProgressIndicator(),
              );
            }

            if (state is AnalysisLoading) {
              return const Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 16),
                    Text('Анализируем изображение...'),
                  ],
                ),
              );
            }

            if (state is AnalysisSuccess) {
              return SingleChildScrollView(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const SizedBox(height: 32),
                    _buildImagePreview(),
                    const SizedBox(height: 32),
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 16),
                      child: Text(
                        state.result,
                        textAlign: TextAlign.center,
                        style: const TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                    const SizedBox(height: 32),
                    ElevatedButton(
                      onPressed: () {
                        context.read<AnalysisBloc>().add(SaveResultEvent());
                        // Возвращаемся на главную страницу вместо предыдущего экрана
                        _navigateToHome(context);
                      },
                      child: const Text('Сохранить результат'),
                    ),
                    const SizedBox(height: 32),
                  ],
                ),
              );
            }

            if (state is AnalysisError) {
              return Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(
                      Icons.error_outline,
                      color: Colors.red,
                      size: 48,
                    ),
                    const SizedBox(height: 16),
                    Text(
                      state.message,
                      style: const TextStyle(
                        fontSize: 18,
                        color: Colors.red,
                      ),
                    ),
                  ],
                ),
              );
            }

            return const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 16),
                  Text('Загрузка...'),
                ],
              ),
            );
          },
        ),
      ),
    );
  }

  // Специальный виджет для правильного отображения анализируемого изображения
  Widget _buildImagePreview() {
    final imageFile = File(imagePath);

    // Проверяем существование файла
    if (!imageFile.existsSync()) {
      return Container(
        width: 224,
        height: 224,
        decoration: BoxDecoration(
          color: Colors.grey[300],
          borderRadius: BorderRadius.circular(12),
        ),
        child: const Center(
          child: Text(
            'Ошибка загрузки изображения',
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    return Column(
      children: [
        Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: Colors.grey[300]!, width: 1),
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Image.file(
              imageFile,
              width: 224,
              height: 224,
              fit: BoxFit.none, // Важно: не масштабировать изображение
              alignment: Alignment.center,
            ),
          ),
        ),
        const SizedBox(height: 8),
        Text(
          'Обрезанное изображение 224×224 пикселей',
          style: TextStyle(
            fontSize: 12,
            color: Colors.grey[600],
          ),
        ),
      ],
    );
  }
}
