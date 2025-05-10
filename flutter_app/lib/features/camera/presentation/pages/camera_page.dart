import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:mr_mole/features/camera/presentation/bloc/camera_bloc.dart';
import 'package:mr_mole/features/analysis/presentation/pages/analys.dart';
import 'package:mr_mole/core/utils/notification.dart';
import 'package:mr_mole/features/camera/presentation/widgets/camera_controls.dart';

class CameraPage extends StatefulWidget {
  final List<CameraDescription> cameras;
  final NotificationService notificationService;

  const CameraPage({
    super.key,
    required this.cameras,
    required this.notificationService,
  });

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  late CameraBloc _cameraBloc;

  @override
  void initState() {
    super.initState();
    _cameraBloc = CameraBloc(widget.cameras);
  }

  @override
  void dispose() {
    _cameraBloc.add(CameraDisposeEvent());
    _cameraBloc.close();
    super.dispose();
  }

  Future<void> _navigateToAnalysis(String imagePath) async {
    _cameraBloc.add(CameraDisposeEvent());
    if (!mounted) return;

    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => AnalysisScreen(
          imagePath: imagePath,
          notificationService: widget.notificationService,
          onRetake: () {
            Navigator.of(context).popUntil((route) => route.isFirst);
          },
        ),
      ),
    );
    if (mounted) {
      Navigator.of(context).popUntil((route) => route.isFirst);
    }
  }

  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: () async {
        _cameraBloc.add(CameraDisposeEvent());
        Navigator.of(context).popUntil((route) => route.isFirst);
        return false;
      },
      child: BlocProvider.value(
        value: _cameraBloc,
        child: BlocListener<CameraBloc, CameraState>(
          listener: (context, state) {
            if (state is ImageCaptured) {
              _navigateToAnalysis(state.imagePath);
            } else if (state is CameraError) {
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text(state.message),
                  backgroundColor: Colors.red,
                ),
              );
            }
          },
          child: Scaffold(
            appBar: AppBar(
              title: const Text('Камера'),
              leading: IconButton(
                icon: const Icon(Icons.arrow_back),
                onPressed: () {
                  _cameraBloc.add(CameraDisposeEvent());
                  Navigator.of(context).popUntil((route) => route.isFirst);
                },
              ),
            ),
            body: BlocBuilder<CameraBloc, CameraState>(
              builder: (context, state) {
                if (state is CameraInitial) {
                  return const Center(
                    child: CircularProgressIndicator(),
                  );
                }

                if (state is CameraError) {
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
                        const SizedBox(height: 24),
                        ElevatedButton(
                          onPressed: () {
                            _cameraBloc.add(CameraInitializeEvent());
                          },
                          child: const Text('Попробовать снова'),
                        ),
                      ],
                    ),
                  );
                }

                if (state is CameraReady) {
                  return Stack(
                    children: [
                      GestureDetector(
                        onScaleStart: (details) {
                          _cameraBloc.add(
                            ZoomChangedEvent(state.currentZoom),
                          );
                        },
                        onScaleUpdate: (details) {
                          if (details.scale != 1.0) {
                            final newZoom = state.currentZoom * details.scale;
                            _cameraBloc.add(
                              ZoomChangedEvent(newZoom),
                            );
                          }
                        },
                        child: CameraPreview(state.controller),
                      ),
                      Positioned(
                        bottom: 30,
                        left: 0,
                        right: 0,
                        child: CameraControls(),
                      ),
                    ],
                  );
                }

                return const Center(
                  child: Text('Неизвестное состояние'),
                );
              },
            ),
          ),
        ),
      ),
    );
  }
}
