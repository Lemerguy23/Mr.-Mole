import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:mr_mole/features/camera/presentation/bloc/camera_bloc.dart';

class CameraControls extends StatelessWidget {
  const CameraControls({super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(20.0),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Индикатор зума
          BlocBuilder<CameraBloc, CameraState>(
            builder: (context, state) {
              if (state is CameraReady) {
                return ZoomSlider(
                  currentZoom: state.currentZoom,
                  minZoom: state.minZoom,
                  maxZoom: state.maxZoom,
                );
              }
              return const SizedBox();
            },
          ),
          // Основные кнопки
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              IconButton(
                icon: const Icon(Icons.switch_camera, color: Colors.white),
                onPressed: () =>
                    context.read<CameraBloc>().add(SwitchCameraEvent()),
              ),
              _CaptureButton(),
              const SizedBox(width: 48), // Для симметрии
            ],
          ),
        ],
      ),
    );
  }
}

class _CaptureButton extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return BlocBuilder<CameraBloc, CameraState>(
      builder: (context, state) {
        return FloatingActionButton(
          onPressed: state is CameraReady
              ? () => context.read<CameraBloc>().add(CaptureImageEvent())
              : null,
          backgroundColor: Colors.white,
          child: const Icon(Icons.camera, color: Colors.black),
        );
      },
    );
  }
}

class ZoomSlider extends StatelessWidget {
  final double currentZoom;
  final double minZoom;
  final double maxZoom;

  const ZoomSlider({
    super.key,
    required this.currentZoom,
    required this.minZoom,
    required this.maxZoom,
  });

  @override
  Widget build(BuildContext context) {
    return Slider(
      value: currentZoom,
      min: minZoom,
      max: maxZoom,
      onChanged: (value) {
        context.read<CameraBloc>().add(ZoomChangedEvent(value));
      },
    );
  }
}
