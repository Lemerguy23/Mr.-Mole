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
                // Проверяем, поддерживается ли зум
                final bool zoomSupported = state.maxZoom > state.minZoom;
                print(
                    'Отображение слайдера зума: поддержка зума = $zoomSupported');
                print(
                    'Зум мин: ${state.minZoom}, макс: ${state.maxZoom}, текущий: ${state.currentZoom}');

                if (!zoomSupported) {
                  return Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    decoration: BoxDecoration(
                      color: Colors.black45,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: const Text(
                      'Зум не поддерживается на этом устройстве',
                      style: TextStyle(color: Colors.white, fontSize: 14),
                    ),
                  );
                }

                return ZoomSlider(
                  currentZoom: state.currentZoom,
                  minZoom: state.minZoom,
                  maxZoom: state.maxZoom,
                );
              }
              return const SizedBox();
            },
          ),
          const SizedBox(height: 16),
          // Основные кнопки
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _buildControlButton(
                icon: Icons.info_outline,
                color: Colors.white,
                size: 28,
                onPressed: () =>
                    context.read<CameraBloc>().add(ToggleInstructionEvent()),
                tooltip: 'Инструкция',
              ),
              _CaptureButton(),
              BlocBuilder<CameraBloc, CameraState>(
                builder: (context, state) {
                  if (state is CameraReady) {
                    return _buildControlButton(
                      icon: state.isFlashOn ? Icons.flash_on : Icons.flash_off,
                      color: Colors.white,
                      size: 28,
                      onPressed: () =>
                          context.read<CameraBloc>().add(ToggleFlashEvent()),
                      tooltip: state.isFlashOn
                          ? 'Выключить вспышку'
                          : 'Включить вспышку',
                    );
                  }
                  return const SizedBox();
                },
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildControlButton({
    required IconData icon,
    required Color color,
    required double size,
    required VoidCallback onPressed,
    required String tooltip,
  }) {
    return Container(
      width: 50,
      height: 50,
      decoration: BoxDecoration(
        color: Colors.black26,
        shape: BoxShape.circle,
      ),
      child: IconButton(
        icon: Icon(icon, color: color, size: size),
        onPressed: onPressed,
        tooltip: tooltip,
      ),
    );
  }
}

class _CaptureButton extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return BlocBuilder<CameraBloc, CameraState>(
      builder: (context, state) {
        final bool isReady = state is CameraReady;
        return Container(
          width: 70,
          height: 70,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            border: Border.all(
              color: Colors.white,
              width: 3,
            ),
          ),
          child: Center(
            child: GestureDetector(
              onTap: isReady
                  ? () => context.read<CameraBloc>().add(
                        CaptureImageEvent(MediaQuery.of(context).size),
                      )
                  : null,
              child: Container(
                width: 60,
                height: 60,
                decoration: BoxDecoration(
                  color: isReady ? Colors.white : Colors.grey,
                  shape: BoxShape.circle,
                ),
              ),
            ),
          ),
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
    print('Построение слайдера зума: $minZoom-$maxZoom, текущий: $currentZoom');

    // Показываем значения зума с шагом 0.5x
    final List<double> zoomLevels = [];
    for (double z = 1.0; z <= maxZoom && z <= 5.0; z += 0.5) {
      zoomLevels.add(z);
    }

    // Принудительно добавляем минимальное и максимальное значения
    if (!zoomLevels.contains(minZoom)) zoomLevels.insert(0, minZoom);
    if (!zoomLevels.contains(maxZoom) && maxZoom <= 5.0)
      zoomLevels.add(maxZoom);

    // Сортируем для правильного порядка
    zoomLevels.sort();

    print('Доступные уровни зума: $zoomLevels');

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
          color: Colors.black45,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: Colors.white.withOpacity(0.3))),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Текущий зум
          Text(
            'Зум: ${currentZoom.toStringAsFixed(1)}x',
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 16,
            ),
          ),
          const SizedBox(height: 10),

          // Индикаторы зума
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: zoomLevels.map((zoom) {
              final bool isActive = (currentZoom - zoom).abs() < 0.25;
              return Column(
                children: [
                  Text(
                    '${zoom.toStringAsFixed(1)}x',
                    style: TextStyle(
                      color: isActive ? Colors.white : Colors.grey,
                      fontSize: 12,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Container(
                    height: 10,
                    width: 3,
                    color:
                        isActive ? Colors.white : Colors.grey.withOpacity(0.5),
                  ),
                ],
              );
            }).toList(),
          ),

          // Слайдер зума
          SliderTheme(
            data: SliderTheme.of(context).copyWith(
              trackHeight: 6.0, // Увеличиваем толщину трека
              thumbShape: const RoundSliderThumbShape(
                enabledThumbRadius: 10.0, // Увеличиваем размер ползунка
                disabledThumbRadius: 8.0,
              ),
              overlayShape: const RoundSliderOverlayShape(
                overlayRadius: 16.0,
              ),
              activeTrackColor: Colors.blue, // Меняем цвет на более заметный
              inactiveTrackColor: Colors.white.withOpacity(0.3),
              thumbColor: Colors.blue, // Синий цвет ползунка
              overlayColor: Colors.blue.withOpacity(0.3),
            ),
            child: Slider(
              value: currentZoom,
              min: minZoom,
              max: maxZoom > 5.0 ? 5.0 : maxZoom, // Ограничиваем для удобства
              divisions: 8,
              onChanged: (value) {
                print('Слайдер: изменение зума на $value');
                context.read<CameraBloc>().add(ZoomChangedEvent(value));
              },
            ),
          ),
        ],
      ),
    );
  }
}
