import 'package:flutter/material.dart';
import 'package:mr_mole/features/home/presentation/bloc/home_bloc.dart';

class MainTab extends StatelessWidget {
  final HomeBloc homeBloc;

  const MainTab({
    super.key,
    required this.homeBloc,
  });

  @override
  Widget build(BuildContext context) {
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
                onPressed: () => homeBloc.add(OpenGalleryEvent()),
                icon: const Icon(Icons.photo_library),
                label: const Text('Галерея'),
              ),
              const SizedBox(width: 16),
              ElevatedButton.icon(
                onPressed: () => homeBloc.add(OpenCameraEvent()),
                icon: const Icon(Icons.camera_alt),
                label: const Text('Камера'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
