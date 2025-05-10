import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:equatable/equatable.dart';
import 'package:shared_preferences/shared_preferences.dart';

// Events
abstract class SettingsEvent extends Equatable {
  const SettingsEvent();

  @override
  List<Object?> get props => [];
}

class LoadSettingsEvent extends SettingsEvent {}

class UpdatePhotoSizeEvent extends SettingsEvent {
  final int size;

  const UpdatePhotoSizeEvent(this.size);

  @override
  List<Object?> get props => [size];
}

class UpdateCameraQualityEvent extends SettingsEvent {
  final String quality;

  const UpdateCameraQualityEvent(this.quality);

  @override
  List<Object?> get props => [quality];
}

class UpdateStoragePathEvent extends SettingsEvent {
  final String path;

  const UpdateStoragePathEvent(this.path);

  @override
  List<Object?> get props => [path];
}

class UpdateNotificationsEvent extends SettingsEvent {
  final bool enabled;
  final int? durationMinutes;

  const UpdateNotificationsEvent(this.enabled, {this.durationMinutes});

  @override
  List<Object?> get props => [enabled, durationMinutes];
}

// States
abstract class SettingsState extends Equatable {
  const SettingsState();

  @override
  List<Object?> get props => [];
}

class SettingsInitial extends SettingsState {}

class SettingsLoading extends SettingsState {}

class SettingsLoaded extends SettingsState {
  final int photoSize;
  final String cameraQuality;
  final String storagePath;
  final bool notificationsEnabled;
  final int notificationDurationMonths;

  const SettingsLoaded({
    required this.photoSize,
    required this.cameraQuality,
    required this.storagePath,
    required this.notificationsEnabled,
    this.notificationDurationMonths = 3,
  });

  @override
  List<Object?> get props => [
        photoSize,
        cameraQuality,
        storagePath,
        notificationsEnabled,
        notificationDurationMonths,
      ];
}

class SettingsError extends SettingsState {
  final String message;

  const SettingsError(this.message);

  @override
  List<Object?> get props => [message];
}

// Bloc
class SettingsBloc extends Bloc<SettingsEvent, SettingsState> {
  final SharedPreferences _prefs;
  static const String _photoSizeKey = 'photo_size';
  static const String _cameraQualityKey = 'camera_quality';
  static const String _storagePathKey = 'storage_path';
  static const String _notificationsKey = 'notifications';
  static const String _notificationDurationKey = 'notification_duration';

  SettingsBloc(this._prefs) : super(SettingsInitial()) {
    on<LoadSettingsEvent>(_onLoadSettings);
    on<UpdatePhotoSizeEvent>(_onUpdatePhotoSize);
    on<UpdateCameraQualityEvent>(_onUpdateCameraQuality);
    on<UpdateStoragePathEvent>(_onUpdateStoragePath);
    on<UpdateNotificationsEvent>(_onUpdateNotifications);
  }

  void _onLoadSettings(LoadSettingsEvent event, Emitter<SettingsState> emit) {
    try {
      emit(SettingsLoading());
      final photoSize = _prefs.getInt(_photoSizeKey) ?? 224;
      final cameraQuality = _prefs.getString(_cameraQualityKey) ?? 'medium';
      final storagePath = _prefs.getString(_storagePathKey) ?? 'default';
      final notificationsEnabled = _prefs.getBool(_notificationsKey) ?? true;
      final notificationDuration = _prefs.getInt(_notificationDurationKey) ?? 3;

      emit(SettingsLoaded(
        photoSize: photoSize,
        cameraQuality: cameraQuality,
        storagePath: storagePath,
        notificationsEnabled: notificationsEnabled,
        notificationDurationMonths: notificationDuration,
      ));
    } catch (e) {
      emit(SettingsError('Ошибка загрузки настроек: $e'));
    }
  }

  void _onUpdatePhotoSize(
      UpdatePhotoSizeEvent event, Emitter<SettingsState> emit) {
    try {
      _prefs.setInt(_photoSizeKey, event.size);
      if (state is SettingsLoaded) {
        final currentState = state as SettingsLoaded;
        emit(SettingsLoaded(
          photoSize: event.size,
          cameraQuality: currentState.cameraQuality,
          storagePath: currentState.storagePath,
          notificationsEnabled: currentState.notificationsEnabled,
          notificationDurationMonths: currentState.notificationDurationMonths,
        ));
      }
    } catch (e) {
      emit(SettingsError('Ошибка обновления размера фото: $e'));
    }
  }

  void _onUpdateCameraQuality(
      UpdateCameraQualityEvent event, Emitter<SettingsState> emit) {
    try {
      _prefs.setString(_cameraQualityKey, event.quality);
      if (state is SettingsLoaded) {
        final currentState = state as SettingsLoaded;
        emit(SettingsLoaded(
          photoSize: currentState.photoSize,
          cameraQuality: event.quality,
          storagePath: currentState.storagePath,
          notificationsEnabled: currentState.notificationsEnabled,
          notificationDurationMonths: currentState.notificationDurationMonths,
        ));
      }
    } catch (e) {
      emit(SettingsError('Ошибка обновления качества камеры: $e'));
    }
  }

  void _onUpdateStoragePath(
      UpdateStoragePathEvent event, Emitter<SettingsState> emit) {
    try {
      _prefs.setString(_storagePathKey, event.path);
      if (state is SettingsLoaded) {
        final currentState = state as SettingsLoaded;
        emit(SettingsLoaded(
          photoSize: currentState.photoSize,
          cameraQuality: currentState.cameraQuality,
          storagePath: event.path,
          notificationsEnabled: currentState.notificationsEnabled,
          notificationDurationMonths: currentState.notificationDurationMonths,
        ));
      }
    } catch (e) {
      emit(SettingsError('Ошибка обновления пути хранения: $e'));
    }
  }

  void _onUpdateNotifications(
      UpdateNotificationsEvent event, Emitter<SettingsState> emit) {
    try {
      _prefs.setBool(_notificationsKey, event.enabled);
      if (event.durationMinutes != null) {
        _prefs.setInt(_notificationDurationKey, event.durationMinutes!);
      }

      if (state is SettingsLoaded) {
        final currentState = state as SettingsLoaded;
        emit(SettingsLoaded(
          photoSize: currentState.photoSize,
          cameraQuality: currentState.cameraQuality,
          storagePath: currentState.storagePath,
          notificationsEnabled: event.enabled,
          notificationDurationMonths:
              event.durationMinutes ?? currentState.notificationDurationMonths,
        ));
      }
    } catch (e) {
      emit(SettingsError('Ошибка обновления настроек уведомлений: $e'));
    }
  }
}
