part of 'camera_bloc.dart';

abstract class CameraState extends Equatable {
  const CameraState();

  @override
  List<Object> get props => [];
}

class CameraInitial extends CameraState {}

class CameraLoading extends CameraState {}

class CameraReady extends CameraState {
  final CameraController controller;
  final double currentZoom;
  final double minZoom;
  final double maxZoom;

  const CameraReady(
    this.controller, {
    this.currentZoom = 1.0,
    this.minZoom = 1.0,
    this.maxZoom = 1.0,
  });

  @override
  List<Object> get props => [controller, currentZoom, minZoom, maxZoom];
}

class ImageCaptured extends CameraState {
  final String imagePath;

  const ImageCaptured(this.imagePath);

  @override
  List<Object> get props => [imagePath];
}

class CameraError extends CameraState {
  final String message;

  const CameraError(this.message);

  @override
  List<Object> get props => [message];
}
