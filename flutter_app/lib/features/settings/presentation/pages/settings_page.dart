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
              onChanged: (value) =>
                  _updateNotificationDuration(context, state, value),
            ),
            RadioListTile<int>(
              title: const Text('6 месяцев'),
              value: 6,
              groupValue: state.notificationDurationMonths,
              onChanged: (value) =>
                  _updateNotificationDuration(context, state, value),
            ),
            RadioListTile<int>(
              title: const Text('9 месяцев'),
              value: 9,
              groupValue: state.notificationDurationMonths,
              onChanged: (value) =>
                  _updateNotificationDuration(context, state, value),
            ),
            RadioListTile<int>(
              title: const Text('1 год'),
              value: 12,
              groupValue: state.notificationDurationMonths,
              onChanged: (value) =>
                  _updateNotificationDuration(context, state, value),
            ),
          ],
        ),
      ),
    );
  }

  void _updateNotificationDuration(
      BuildContext context, SettingsLoaded state, int? value) {
    if (value != null) {
      context.read<SettingsBloc>().add(
            UpdateNotificationsEvent(
              state.notificationsEnabled,
              durationMonths: value,
            ),
          );
      Navigator.pop(context);
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
