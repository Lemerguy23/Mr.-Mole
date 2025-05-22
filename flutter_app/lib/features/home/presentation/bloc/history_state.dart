part of 'history_bloc.dart';

abstract class HistoryState {}

class HistoryInitial extends HistoryState {}

class HistoryLoading extends HistoryState {}

class HistoryEmpty extends HistoryState {}

class HistoryLoaded extends HistoryState {
  final List<ScanHistoryItem> items;

  HistoryLoaded(this.items);
}

class HistoryError extends HistoryState {
  final String message;

  HistoryError(this.message);
}
