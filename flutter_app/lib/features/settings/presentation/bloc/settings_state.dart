part of 'settings_bloc.dart';

abstract class SettingsState extends Equatable {
  const SettingsState();

  @override
  List<Object?> get props => [];
}

class SettingsInitial extends SettingsState {}

class SettingsLoading extends SettingsState {}

class SettingsLoaded extends SettingsState {
  final bool notificationsEnabled;
  final int notificationDurationMonths;

  const SettingsLoaded({
    required this.notificationsEnabled,
    this.notificationDurationMonths = 3,
  });

  @override
  List<Object?> get props => [notificationsEnabled, notificationDurationMonths];
}

class SettingsError extends SettingsState {
  final String message;

  const SettingsError(this.message);

  @override
  List<Object?> get props => [message];
}
