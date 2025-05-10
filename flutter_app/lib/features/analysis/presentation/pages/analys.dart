import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:mr_mole/features/analysis/presentation/bloc/analysis_bloc.dart';
import 'package:mr_mole/core/utils/notification.dart';

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

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (context) => AnalysisBloc(
        imagePath: imagePath,
        notificationService: notificationService,
      )..add(AnalyzeImageEvent()),
      child: WillPopScope(
        onWillPop: () async {
          onRetake();
          Navigator.of(context).popUntil((route) => route.isFirst);
          return false;
        },
        child: Scaffold(
          appBar: AppBar(
            title: const Text('Анализ'),
            leading: IconButton(
              icon: const Icon(Icons.arrow_back),
              onPressed: () {
                onRetake();
                Navigator.of(context).popUntil((route) => route.isFirst);
              },
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
                return Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Image.file(
                        File(imagePath),
                        height: 300,
                        width: 300,
                        fit: BoxFit.cover,
                      ),
                      const SizedBox(height: 32),
                      Text(
                        state.result,
                        style: const TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 32),
                      ElevatedButton(
                        onPressed: () {
                          context.read<AnalysisBloc>().add(SaveResultEvent());
                          onRetake();
                          Navigator.of(context)
                              .popUntil((route) => route.isFirst);
                        },
                        child: const Text('Сохранить результат'),
                      ),
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
                      const SizedBox(height: 32),
                      ElevatedButton(
                        onPressed: () {
                          onRetake();
                          Navigator.of(context)
                              .popUntil((route) => route.isFirst);
                        },
                        child: const Text('Попробовать снова'),
                      ),
                    ],
                  ),
                );
              }

              return const Center(
                child: Text('Неизвестное состояние'),
              );
            },
          ),
        ),
      ),
    );
  }
}
