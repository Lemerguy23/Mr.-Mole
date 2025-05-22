import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:mr_mole/features/home/data/repositories/scan_history_repository.dart';
import 'package:mr_mole/features/home/domain/models/scan_history_item.dart';

part 'history_event.dart';
part 'history_state.dart';

class HistoryBloc extends Bloc<HistoryEvent, HistoryState> {
  final ScanHistoryRepository _repository;

  HistoryBloc(this._repository) : super(HistoryInitial()) {
    on<LoadHistoryEvent>(_onLoadHistory);
    on<RemoveHistoryItemEvent>(_onRemoveHistoryItem);
    on<ClearHistoryEvent>(_onClearHistory);
  }

  Future<void> _onLoadHistory(
    LoadHistoryEvent event,
    Emitter<HistoryState> emit,
  ) async {
    try {
      emit(HistoryLoading());
      final history = await _repository.getHistory();

      if (history.isEmpty) {
        emit(HistoryEmpty());
      } else {
        emit(HistoryLoaded(history));
      }
    } catch (e) {
      emit(HistoryError('Ошибка при загрузке истории: ${e.toString()}'));
    }
  }

  Future<void> _onRemoveHistoryItem(
    RemoveHistoryItemEvent event,
    Emitter<HistoryState> emit,
  ) async {
    try {
      emit(HistoryLoading());
      final success = await _repository.removeFromHistory(event.id);

      if (success) {
        final history = await _repository.getHistory();

        if (history.isEmpty) {
          emit(HistoryEmpty());
        } else {
          emit(HistoryLoaded(history));
        }
      } else {
        emit(HistoryError('Не удалось удалить элемент из истории'));
      }
    } catch (e) {
      emit(HistoryError('Ошибка при удалении элемента: ${e.toString()}'));
    }
  }

  Future<void> _onClearHistory(
    ClearHistoryEvent event,
    Emitter<HistoryState> emit,
  ) async {
    try {
      emit(HistoryLoading());
      final success = await _repository.clearHistory();

      if (success) {
        emit(HistoryEmpty());
      } else {
        emit(HistoryError('Не удалось очистить историю'));
      }
    } catch (e) {
      emit(HistoryError('Ошибка при очистке истории: ${e.toString()}'));
    }
  }
}
