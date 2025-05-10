import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

class CameraPreviewWidget extends StatefulWidget {
  final CameraController controller;

  const CameraPreviewWidget({super.key, required this.controller});

  @override
  State<CameraPreviewWidget> createState() => _CameraPreviewWidgetState();
}

class _CameraPreviewWidgetState extends State<CameraPreviewWidget> {
  double _baseScale = 1.0;
  double _currentScale = 1.0;
  double _minAvailableZoom = 1.0;
  double _maxAvailableZoom = 1.0;

  @override
  void initState() {
    super.initState();
    _initZoomValues();
  }

  Future<void> _initZoomValues() async {
    _minAvailableZoom = await widget.controller.getMinZoomLevel();
    _maxAvailableZoom = await widget.controller.getMaxZoomLevel();
  }

  void _handleScaleStart(ScaleStartDetails details) {
    _baseScale = _currentScale;
  }

  Future<void> _handleScaleUpdate(ScaleUpdateDetails details) async {
    if (widget.controller.value.isInitialized) {
      setState(() {
        _currentScale = (_baseScale * details.scale).clamp(
          _minAvailableZoom,
          _maxAvailableZoom,
        );
      });
      await widget.controller.setZoomLevel(_currentScale);
    }
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onScaleStart: _handleScaleStart,
      onScaleUpdate: _handleScaleUpdate,
      child: SizedBox.expand(
        child: FittedBox(
          fit: BoxFit.cover,
          child: SizedBox(
            width: MediaQuery.of(context).size.width,
            height: MediaQuery.of(context).size.height,
            child: CameraPreview(widget.controller),
          ),
        ),
      ),
    );
  }
}
