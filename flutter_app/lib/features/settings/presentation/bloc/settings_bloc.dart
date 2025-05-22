import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:equatable/equatable.dart';
import 'package:shared_preferences/shared_preferences.dart';

part 'settings_event.dart';
part 'settings_state.dart';

class SettingsBloc extends Bloc<SettingsEvent, SettingsState> {
  final SharedPreferences _prefs;
  static const String _notificationsKey = 'notifications';
  static const String _notificationDurationKey = 'notification_duration';

  SettingsBloc(this._prefs) : super(SettingsInitial()) {
    on<LoadSettingsEvent>(_onLoadSettings);
    on<UpdateNotificationsEvent>(_onUpdateNotifications);
  }

  void _onLoadSettings(LoadSettingsEvent event, Emitter<SettingsState> emit) {
    try {
      emit(SettingsLoading());
      final notificationsEnabled = _prefs.getBool(_notificationsKey) ?? true;
      final notificationDuration = _prefs.getInt(_notificationDurationKey) ?? 3;

      emit(SettingsLoaded(
        notificationsEnabled: notificationsEnabled,
        notificationDurationMonths: notificationDuration,
      ));
    } catch (e) {
      emit(SettingsError('Ошибка загрузки настроек: $e'));
    }
  }

  void _onUpdateNotifications(
      UpdateNotificationsEvent event, Emitter<SettingsState> emit) {
    try {
      _prefs.setBool(_notificationsKey, event.enabled);
      if (event.durationMonths != null) {
        _prefs.setInt(_notificationDurationKey, event.durationMonths!);
      }

      if (state is SettingsLoaded) {
        final currentState = state as SettingsLoaded;
        emit(SettingsLoaded(
          notificationsEnabled: event.enabled,
          notificationDurationMonths:
              event.durationMonths ?? currentState.notificationDurationMonths,
        ));
      }
    } catch (e) {
      emit(SettingsError('Ошибка обновления настроек уведомлений: $e'));
    }
  }
}
