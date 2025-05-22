part of 'settings_bloc.dart';

abstract class SettingsEvent extends Equatable {
  const SettingsEvent();

  @override
  List<Object?> get props => [];
}

class LoadSettingsEvent extends SettingsEvent {}

class UpdateNotificationsEvent extends SettingsEvent {
  final bool enabled;
  final int? durationMonths;

  const UpdateNotificationsEvent(this.enabled, {this.durationMonths});

  @override
  List<Object?> get props => [enabled, durationMonths];
}
