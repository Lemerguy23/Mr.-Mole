import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class AnalysisScreen extends StatefulWidget {
  final String imagePath;

  const AnalysisScreen({Key? key, required this.imagePath}) : super(key: key);

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen> {
  String _result = 'Анализ...';
  bool _isLoading = true;
  double _confidence = 0.0;
  String _detailedResult = '';
  Interpreter? _interpreter;
  img.Image? _processedImage; // Для хранения обработанного изображения

  @override
  void initState() {
    super.initState();
    _initializeAndAnalyze();
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  Future<void> _initializeAndAnalyze() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model.tflite');
      await _analyzeImage();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _result = 'Ошибка инициализации';
        _detailedResult = 'Не удалось загрузить модель: $e';
        _isLoading = false;
      });
    }
  }

  Future<void> _analyzeImage() async {
    if (!mounted) return;
    setState(() {
      _isLoading = true;
      _result = 'Анализ...';
      _detailedResult = '';
      _processedImage = null;
    });

    try {
      final imageBytes = await File(widget.imagePath).readAsBytes();
      final image = img.decodeImage(imageBytes);
      if (image == null) throw Exception('Не удалось декодировать изображение');

      final resizedImage = img.copyResize(image, width: 224, height: 224);
      _processedImage = resizedImage;

      final inputBuffer = Float32List(1 * 224 * 224 * 3);
      int pixelIndex = 0;
      for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
          final pixel = resizedImage.getPixel(x, y);
          inputBuffer[pixelIndex++] = pixel.r / 255.0;
          inputBuffer[pixelIndex++] = pixel.b / 255.0;
          inputBuffer[pixelIndex++] = pixel.g / 255.0;
        }
      }

      final output = List.filled(1 * 1, 0.0).reshape([1, 1]);
      _interpreter!.run(inputBuffer.buffer, output);

      final probability = output[0][0];
      final confidence = (probability * 100).clamp(0.0, 100.0);

      if (!mounted) return;
      setState(() {
        _confidence = confidence;
        _result = probability > 0.5 ? "Рак обнаружен" : "Рак не обнаружен";
        _detailedResult = _getDetailedResult(probability);
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _result = 'Ошибка анализа';
        _detailedResult = 'Не удалось обработать изображение: ${e.toString()}';
        _isLoading = false;
      });
    }
  }

  String _getDetailedResult(double probability) {
    if (probability > 0.5) {
      return 'Обнаружена высокая вероятность рака (${_confidence.toStringAsFixed(1)}%)';
    } else {
      return 'Вероятность наличия рака низкая (${(100 - _confidence).toStringAsFixed(1)}%)';
    }
  }

  // Widget _buildProcessedImage() {
  //   if (_processedImage == null) return Container();

  //   final processedBytes = img.encodePng(_processedImage!);

  //   return Column(
  //     children: [
  //       const SizedBox(height: 20),
  //       const Text(
  //         'Обработанное изображение (224x224)',
  //         style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
  //       ),
  //       const SizedBox(height: 10),
  //       Image.memory(
  //         Uint8List.fromList(processedBytes),
  //         width: 224,
  //         height: 224,
  //         fit: BoxFit.cover,
  //       ),
  //     ],
  //   );
  // }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Результаты анализа'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.1),
                    blurRadius: 10,
                    spreadRadius: 2,
                  ),
                ],
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.file(
                  File(widget.imagePath),
                  width: MediaQuery.of(context).size.width * 0.8,
                  fit: BoxFit.cover,
                ),
              ),
            ),
            const SizedBox(height: 30),

            // if (!_isLoading) _buildProcessedImage(),
            _isLoading
                ? const Column(
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 20),
                    Text(
                      'Идет анализ изображения...',
                      style: TextStyle(fontSize: 18),
                    ),
                  ],
                )
                : Column(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 20,
                        vertical: 15,
                      ),
                      decoration: BoxDecoration(
                        color:
                            _confidence > 50
                                ? Colors.red.withOpacity(0.2)
                                : Colors.green.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Column(
                        children: [
                          Text(
                            _result,
                            style: TextStyle(
                              fontSize: 22,
                              fontWeight: FontWeight.bold,
                              color:
                                  _confidence > 50 ? Colors.red : Colors.green,
                            ),
                            textAlign: TextAlign.center,
                          ),
                          const SizedBox(height: 8),
                          LinearProgressIndicator(
                            value: _confidence / 100,
                            backgroundColor: Colors.grey[200],
                            valueColor: AlwaysStoppedAnimation<Color>(
                              _confidence > 50 ? Colors.red : Colors.green,
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            '${_confidence.toStringAsFixed(1)}% уверенности',
                            style: const TextStyle(fontSize: 14),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 20),

                    Text(
                      _detailedResult,
                      style: const TextStyle(fontSize: 16, height: 1.5),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 30),

                    ElevatedButton.icon(
                      onPressed: _isLoading ? null : _analyzeImage,
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 20,
                          vertical: 12,
                        ),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10),
                        ),
                      ),
                      icon:
                          _isLoading
                              ? Container()
                              : const Icon(Icons.refresh, size: 20),
                      label:
                          _isLoading
                              ? const SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                ),
                              )
                              : const Text('Повторить'),
                    ),
                  ],
                ),
          ],
        ),
      ),
    );
  }
}
