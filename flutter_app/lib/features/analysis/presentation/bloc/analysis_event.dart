part of 'analysis_bloc.dart';

abstract class AnalysisEvent extends Equatable {
  const AnalysisEvent();

  @override
  List<Object> get props => [];
}

class AnalyzeImageEvent extends AnalysisEvent {}

class SaveResultEvent extends AnalysisEvent {}
