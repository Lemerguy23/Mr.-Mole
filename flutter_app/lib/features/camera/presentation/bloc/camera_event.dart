part of 'camera_bloc.dart';

abstract class CameraEvent extends Equatable {
  const CameraEvent();

  @override
  List<Object> get props => [];
}

class CameraInitializeEvent extends CameraEvent {}

class CaptureImageEvent extends CameraEvent {
  final Size screenSize;

  const CaptureImageEvent(this.screenSize);

  @override
  List<Object> get props => [screenSize];
}

class SwitchCameraEvent extends CameraEvent {}

class ResetCameraEvent extends CameraEvent {}

class CameraDisposeEvent extends CameraEvent {}

class ZoomChangedEvent extends CameraEvent {
  final double zoomLevel;

  const ZoomChangedEvent(this.zoomLevel);

  @override
  List<Object> get props => [zoomLevel];
}

class ToggleFlashEvent extends CameraEvent {}

class ToggleInstructionEvent extends CameraEvent {}
