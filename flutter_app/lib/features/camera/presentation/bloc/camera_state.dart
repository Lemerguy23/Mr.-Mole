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
  final bool isFlashOn;
  final bool showInstruction;
  final Rect captureRect;

  const CameraReady(
    this.controller, {
    this.currentZoom = 1.0,
    this.minZoom = 1.0,
    this.maxZoom = 1.0,
    this.isFlashOn = false,
    this.showInstruction = false,
    this.captureRect = const Rect.fromLTWH(0, 0, 224, 224),
  });

  CameraReady copyWith({
    CameraController? controller,
    double? currentZoom,
    double? minZoom,
    double? maxZoom,
    bool? isFlashOn,
    bool? showInstruction,
    Rect? captureRect,
  }) {
    return CameraReady(
      controller ?? this.controller,
      currentZoom: currentZoom ?? this.currentZoom,
      minZoom: minZoom ?? this.minZoom,
      maxZoom: maxZoom ?? this.maxZoom,
      isFlashOn: isFlashOn ?? this.isFlashOn,
      showInstruction: showInstruction ?? this.showInstruction,
      captureRect: captureRect ?? this.captureRect,
    );
  }

  @override
  List<Object> get props => [
        controller,
        currentZoom,
        minZoom,
        maxZoom,
        isFlashOn,
        showInstruction,
        captureRect,
      ];
}

class ImageCaptured extends CameraState {
  final String imagePath;
  final Rect captureRect;

  const ImageCaptured(this.imagePath, {required this.captureRect});

  @override
  List<Object> get props => [imagePath, captureRect];
}

class CameraError extends CameraState {
  final String message;

  const CameraError(this.message);

  @override
  List<Object> get props => [message];
}
