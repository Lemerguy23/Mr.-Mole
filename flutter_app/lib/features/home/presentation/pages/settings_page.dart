import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:mr_mole/features/settings/presentation/bloc/settings_bloc.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key});

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  late Future<SharedPreferences> _prefsFuture;

  @override
  void initState() {
    super.initState();
    _prefsFuture = SharedPreferences.getInstance();
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<SharedPreferences>(
      future: _prefsFuture,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Scaffold(
            body: Center(child: CircularProgressIndicator()),
          );
        }

        if (snapshot.hasError) {
          return Scaffold(
            body: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.error_outline, color: Colors.red, size: 48),
                  const SizedBox(height: 16),
                  Text('Ошибка: ${snapshot.error}'),
                  const SizedBox(height: 24),
                  ElevatedButton(
                    onPressed: () {
                      setState(() {
                        _prefsFuture = SharedPreferences.getInstance();
                      });
                    },
                    child: const Text('Повторить'),
                  ),
                ],
              ),
            ),
          );
        }

        return BlocProvider(
          create: (context) =>
              SettingsBloc(snapshot.data!)..add(LoadSettingsEvent()),
          child: Scaffold(
            appBar: AppBar(
              leading: IconButton(
                onPressed: () => Navigator.pop(context),
                icon: const Icon(
                  Icons.arrow_back,
                  size: 32,
                  color: Colors.white,
                ),
              ),
              backgroundColor: const Color(0xFF1b264a),
              title: const Text(
                'Настройки',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              centerTitle: true,
            ),
            body: BlocBuilder<SettingsBloc, SettingsState>(
              builder: (context, state) {
                if (state is SettingsLoading) {
                  return const Center(child: CircularProgressIndicator());
                }

                if (state is SettingsError) {
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
                        Text(state.message),
                        const SizedBox(height: 24),
                        ElevatedButton(
                          onPressed: () {
                            context.read<SettingsBloc>().add(
                                  LoadSettingsEvent(),
                                );
                          },
                          child: const Text('Повторить'),
                        ),
                      ],
                    ),
                  );
                }

                if (state is SettingsLoaded) {
                  return ListView(
                    children: [
                      _buildPhotoSizeSection(context, state),
                      _buildCameraQualitySection(context, state),
                      _buildStorageSection(context, state),
                      _buildNotificationsSection(context, state),
                    ],
                  );
                }

                return const SizedBox();
              },
            ),
          ),
        );
      },
    );
  }

  Widget _buildPhotoSizeSection(BuildContext context, SettingsLoaded state) {
    return ListTile(
      leading: const Icon(
        Icons.photo_size_select_large,
        color: Color(0xFF1b264a),
      ),
      title: const Text('Размер фото'),
      subtitle: Text('${state.photoSize}x${state.photoSize}'),
      onTap: () => _showPhotoSizeDialog(context, state),
    );
  }

  Widget _buildCameraQualitySection(
    BuildContext context,
    SettingsLoaded state,
  ) {
    return ListTile(
      leading: const Icon(Icons.camera_alt, color: Color(0xFF1b264a)),
      title: const Text('Качество камеры'),
      subtitle: Text(_getQualityText(state.cameraQuality)),
      onTap: () => _showCameraQualityDialog(context, state),
    );
  }

  Widget _buildStorageSection(BuildContext context, SettingsLoaded state) {
    return ListTile(
      leading: const Icon(Icons.storage, color: Color(0xFF1b264a)),
      title: const Text('Хранение'),
      subtitle: Text(_getStoragePathText(state.storagePath)),
      onTap: () => _showStorageDialog(context, state),
    );
  }

  Widget _buildNotificationsSection(
      BuildContext context, SettingsLoaded state) {
    return Column(
      children: [
        SwitchListTile(
          secondary: const Icon(Icons.notifications, color: Color(0xFF1b264a)),
          title: const Text('Уведомления'),
          subtitle: const Text('Включить напоминания о проверке'),
          value: state.notificationsEnabled,
          onChanged: (value) {
            context.read<SettingsBloc>().add(UpdateNotificationsEvent(value));
          },
        ),
        if (state.notificationsEnabled)
          ListTile(
            leading: const Icon(Icons.timer, color: Color(0xFF1b264a)),
            title: const Text('Интервал напоминаний'),
            subtitle: Text(
                _getNotificationDurationText(state.notificationDurationMonths)),
            onTap: () => _showNotificationDurationDialog(context, state),
          ),
      ],
    );
  }

  void _showPhotoSizeDialog(BuildContext context, SettingsLoaded state) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Размер фото'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            RadioListTile<int>(
              title: const Text('Маленький (224x224)'),
              value: 224,
              groupValue: state.photoSize,
              onChanged: (value) {
                if (value != null) {
                  context.read<SettingsBloc>().add(
                        UpdatePhotoSizeEvent(value),
                      );
                  Navigator.pop(context);
                }
              },
            ),
            RadioListTile<int>(
              title: const Text('Средний (448x448)'),
              value: 448,
              groupValue: state.photoSize,
              onChanged: (value) {
                if (value != null) {
                  context.read<SettingsBloc>().add(
                        UpdatePhotoSizeEvent(value),
                      );
                  Navigator.pop(context);
                }
              },
            ),
            RadioListTile<int>(
              title: const Text('Большой (896x896)'),
              value: 896,
              groupValue: state.photoSize,
              onChanged: (value) {
                if (value != null) {
                  context.read<SettingsBloc>().add(
                        UpdatePhotoSizeEvent(value),
                      );
                  Navigator.pop(context);
                }
              },
            ),
          ],
        ),
      ),
    );
  }

  void _showCameraQualityDialog(BuildContext context, SettingsLoaded state) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Качество камеры'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            RadioListTile<String>(
              title: const Text('Низкое'),
              value: 'low',
              groupValue: state.cameraQuality,
              onChanged: (value) {
                if (value != null) {
                  context.read<SettingsBloc>().add(
                        UpdateCameraQualityEvent(value),
                      );
                  Navigator.pop(context);
                }
              },
            ),
            RadioListTile<String>(
              title: const Text('Среднее'),
              value: 'medium',
              groupValue: state.cameraQuality,
              onChanged: (value) {
                if (value != null) {
                  context.read<SettingsBloc>().add(
                        UpdateCameraQualityEvent(value),
                      );
                  Navigator.pop(context);
                }
              },
            ),
            RadioListTile<String>(
              title: const Text('Высокое'),
              value: 'high',
              groupValue: state.cameraQuality,
              onChanged: (value) {
                if (value != null) {
                  context.read<SettingsBloc>().add(
                        UpdateCameraQualityEvent(value),
                      );
                  Navigator.pop(context);
                }
              },
            ),
          ],
        ),
      ),
    );
  }

  void _showStorageDialog(BuildContext context, SettingsLoaded state) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Хранение'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            RadioListTile<String>(
              title: const Text('По умолчанию'),
              value: 'default',
              groupValue: state.storagePath,
              onChanged: (value) {
                if (value != null) {
                  context.read<SettingsBloc>().add(
                        UpdateStoragePathEvent(value),
                      );
                  Navigator.pop(context);
                }
              },
            ),
            RadioListTile<String>(
              title: const Text('Внешнее хранилище'),
              value: 'external',
              groupValue: state.storagePath,
              onChanged: (value) {
                if (value != null) {
                  context.read<SettingsBloc>().add(
                        UpdateStoragePathEvent(value),
                      );
                  Navigator.pop(context);
                }
              },
            ),
          ],
        ),
      ),
    );
  }

  void _showNotificationDurationDialog(
      BuildContext context, SettingsLoaded state) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Интервал напоминаний'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            RadioListTile<int>(
              title: const Text('3 месяца'),
              value: 3,
              groupValue: state.notificationDurationMonths,
              onChanged: (value) {
                if (value != null) {
                  context.read<SettingsBloc>().add(
                        UpdateNotificationsEvent(
                          state.notificationsEnabled,
                          durationMinutes: value,
                        ),
                      );
                  Navigator.pop(context);
                }
              },
            ),
            RadioListTile<int>(
              title: const Text('6 месяцев'),
              value: 6,
              groupValue: state.notificationDurationMonths,
              onChanged: (value) {
                if (value != null) {
                  context.read<SettingsBloc>().add(
                        UpdateNotificationsEvent(
                          state.notificationsEnabled,
                          durationMinutes: value,
                        ),
                      );
                  Navigator.pop(context);
                }
              },
            ),
            RadioListTile<int>(
              title: const Text('9 месяцев'),
              value: 9,
              groupValue: state.notificationDurationMonths,
              onChanged: (value) {
                if (value != null) {
                  context.read<SettingsBloc>().add(
                        UpdateNotificationsEvent(
                          state.notificationsEnabled,
                          durationMinutes: value,
                        ),
                      );
                  Navigator.pop(context);
                }
              },
            ),
            RadioListTile<int>(
              title: const Text('1 год'),
              value: 12,
              groupValue: state.notificationDurationMonths,
              onChanged: (value) {
                if (value != null) {
                  context.read<SettingsBloc>().add(
                        UpdateNotificationsEvent(
                          state.notificationsEnabled,
                          durationMinutes: value,
                        ),
                      );
                  Navigator.pop(context);
                }
              },
            ),
          ],
        ),
      ),
    );
  }

  String _getQualityText(String quality) {
    switch (quality) {
      case 'low':
        return 'Низкое';
      case 'medium':
        return 'Среднее';
      case 'high':
        return 'Высокое';
      default:
        return 'Среднее';
    }
  }

  String _getStoragePathText(String path) {
    switch (path) {
      case 'default':
        return 'Внутреннее хранилище';
      case 'external':
        return 'Внешнее хранилище';
      default:
        return 'Внутреннее хранилище';
    }
  }

  String _getNotificationDurationText(int months) {
    if (months == 12) {
      return '1 год';
    }
    return '$months ${_getMonthText(months)}';
  }

  String _getMonthText(int months) {
    if (months % 10 == 1 && months != 11) {
      return 'месяц';
    } else if ((months % 10 >= 2 && months % 10 <= 4) &&
        !(months >= 12 && months <= 14)) {
      return 'месяца';
    } else {
      return 'месяцев';
    }
  }
}
