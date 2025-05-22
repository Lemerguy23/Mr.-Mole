part of 'history_bloc.dart';

abstract class HistoryEvent {}

class LoadHistoryEvent extends HistoryEvent {}

class RemoveHistoryItemEvent extends HistoryEvent {
  final String id;

  RemoveHistoryItemEvent(this.id);
}

class ClearHistoryEvent extends HistoryEvent {}
