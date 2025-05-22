import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:mr_mole/features/camera/presentation/bloc/camera_bloc.dart';
import 'package:mr_mole/features/analysis/presentation/pages/analys.dart';
import 'package:mr_mole/core/utils/notification.dart';
import 'package:mr_mole/features/camera/presentation/widgets/camera_controls.dart';
import 'package:mr_mole/features/camera/presentation/widgets/camera_preview.dart';
import 'package:mr_mole/features/home/data/repositories/scan_history_repository.dart';
import 'package:permission_handler/permission_handler.dart';

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

class _CameraPageState extends State<CameraPage> with WidgetsBindingObserver {
  late CameraBloc _cameraBloc;
  bool _isPermissionRequested = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _cameraBloc = CameraBloc(widget.cameras);
    _requestPermissions();
  }

  Future<void> _requestPermissions() async {
    // Запрашиваем разрешение на камеру
    final cameraStatus = await Permission.camera.request();

    _isPermissionRequested = true;

    if (cameraStatus.isGranted) {
      _cameraBloc.add(CameraInitializeEvent());
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Для работы приложения необходим доступ к камере'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    super.didChangeAppLifecycleState(state);

    // Перезапускаем камеру при возвращении из фона
    if (state == AppLifecycleState.resumed && _isPermissionRequested) {
      _cameraBloc.add(CameraInitializeEvent());
    }
  }

  void _handleBack() {
    Navigator.of(context).pop();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraBloc.add(CameraDisposeEvent());
    _cameraBloc.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return BlocProvider.value(
      value: _cameraBloc,
      child: BlocListener<CameraBloc, CameraState>(
        listener: (context, state) {
          if (state is ImageCaptured) {
            Navigator.of(context).push(
              MaterialPageRoute(
                builder: (context) => AnalysisScreen(
                  imagePath: state.imagePath,
                  notificationService: widget.notificationService,
                  onRetake: () {
                    Navigator.of(context).pop();
                    _cameraBloc.add(CameraInitializeEvent());
                  },
                ),
              ),
            );
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
              onPressed: _handleBack,
            ),
          ),
          body: _buildCameraPreview(),
        ),
      ),
    );
  }

  Widget _buildCameraPreview() {
    return BlocBuilder<CameraBloc, CameraState>(
      builder: (context, state) {
        if (state is CameraInitial) {
          return const Center(
            child: CircularProgressIndicator(),
          );
        }

        if (state is CameraLoading) {
          return const Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                CircularProgressIndicator(),
                SizedBox(height: 16),
                Text('Инициализация камеры...'),
              ],
            ),
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
          return SafeArea(
            child: Stack(
              children: [
                Container(
                  color: Colors.black,
                  width: MediaQuery.of(context).size.width,
                  height: MediaQuery.of(context).size.height,
                  child: CameraPreviewWidget(
                    controller: state.controller,
                    showInstruction: state.showInstruction,
                    onCloseInstruction: () {
                      context.read<CameraBloc>().add(ToggleInstructionEvent());
                    },
                  ),
                ),
                Positioned(
                  bottom: 30,
                  left: 0,
                  right: 0,
                  child: CameraControls(),
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
    );
  }
}
