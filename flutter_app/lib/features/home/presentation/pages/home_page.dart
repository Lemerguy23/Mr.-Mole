import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:camera/camera.dart';
import 'package:mr_mole/features/home/presentation/bloc/home_bloc.dart';
import 'package:mr_mole/features/camera/presentation/pages/camera_page.dart';
import 'package:mr_mole/features/home/presentation/pages/settings_page.dart';
import 'package:mr_mole/features/analysis/presentation/pages/analys.dart';
import 'package:mr_mole/core/utils/notification.dart';

class HomePage extends StatefulWidget {
  final Future<List<CameraDescription>> camerasFuture;
  final NotificationService notificationService;

  const HomePage({
    super.key,
    required this.camerasFuture,
    required this.notificationService,
  });

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  late HomeBloc _homeBloc;

  @override
  void initState() {
    super.initState();
    _homeBloc = HomeBloc(widget.camerasFuture);
  }

  @override
  void dispose() {
    _homeBloc.close();
    super.dispose();
  }

  void _handleCameraReturn() {
    _homeBloc.add(ResetHomeEvent());
  }

  @override
  Widget build(BuildContext context) {
    return BlocProvider.value(
      value: _homeBloc,
      child: BlocListener<HomeBloc, HomeState>(
        listener: (context, state) {
          if (state is GalleryImageSelected) {
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) => AnalysisScreen(
                  imagePath: state.imagePath,
                  notificationService: widget.notificationService,
                  onRetake: () {
                    Navigator.of(context).popUntil((route) => route.isFirst);
                    _homeBloc.add(ResetHomeEvent());
                  },
                ),
              ),
            ).then((_) {
              _homeBloc.add(ResetHomeEvent());
            });
          }
        },
        child: Scaffold(
          appBar: AppBar(
            title: const Text('Mr. Mole'),
            actions: [
              IconButton(
                icon: const Icon(Icons.settings),
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const SettingsPage(),
                    ),
                  );
                },
              ),
            ],
          ),
          body: BlocBuilder<HomeBloc, HomeState>(
            builder: (context, state) {
              if (state is HomeLoading) {
                return const Center(
                  child: CircularProgressIndicator(),
                );
              }

              if (state is HomeError) {
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
                          _homeBloc.add(OpenCameraEvent());
                        },
                        child: const Text('Попробовать снова'),
                      ),
                    ],
                  ),
                );
              }

              if (state is CameraReady) {
                return WillPopScope(
                  onWillPop: () async {
                    _handleCameraReturn();
                    return true;
                  },
                  child: CameraPage(
                    cameras: state.cameras,
                    notificationService: widget.notificationService,
                  ),
                );
              }

              return Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text(
                      'Добро пожаловать в Mr. Mole',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 32),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        ElevatedButton.icon(
                          onPressed: () {
                            _homeBloc.add(OpenGalleryEvent());
                          },
                          icon: const Icon(Icons.photo_library),
                          label: const Text('Галерея'),
                        ),
                        const SizedBox(width: 16),
                        ElevatedButton.icon(
                          onPressed: () {
                            _homeBloc.add(OpenCameraEvent());
                          },
                          icon: const Icon(Icons.camera_alt),
                          label: const Text('Камера'),
                        ),
                      ],
                    ),
                  ],
                ),
              );
            },
          ),
        ),
      ),
    );
  }
}
