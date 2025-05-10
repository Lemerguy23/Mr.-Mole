part of 'camera_bloc.dart';

abstract class CameraEvent extends Equatable {
  const CameraEvent();

  @override
  List<Object> get props => [];
}

class CameraInitializeEvent extends CameraEvent {}

class CaptureImageEvent extends CameraEvent {}

class SwitchCameraEvent extends CameraEvent {}

class ResetCameraEvent extends CameraEvent {}

class CameraDisposeEvent extends CameraEvent {}

class ZoomChangedEvent extends CameraEvent {
  final double zoomLevel;

  const ZoomChangedEvent(this.zoomLevel);

  @override
  List<Object> get props => [zoomLevel];
}

class ZoomGestureEvent extends CameraEvent {
  final double gestureScale;

  const ZoomGestureEvent(this.gestureScale);

  @override
  List<Object> get props => [gestureScale];
}

class ConfirmMolePositionEvent extends CameraEvent {
  final double x; // координаты центра квадрата
  final double y;

  const ConfirmMolePositionEvent({required this.x, required this.y});

  @override
  List<Object> get props => [x, y];
}

class ToggleFlashEvent extends CameraEvent {}
