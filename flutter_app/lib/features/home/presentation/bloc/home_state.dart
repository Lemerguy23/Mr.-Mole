part of 'home_bloc.dart';

abstract class HomeState {}

class HomeInitial extends HomeState {}

class HomeLoading extends HomeState {}

class GalleryImageSelected extends HomeState {
  final String imagePath;
  GalleryImageSelected(this.imagePath);
}

class CameraReady extends HomeState {
  final List<CameraDescription> cameras;
  CameraReady(this.cameras);
}

class HomeError extends HomeState {
  final String message;
  final String source;

  HomeError(this.message, {this.source = 'general'});
}

class NavigateToSettings extends HomeState {}
