import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'preview.dart';
import 'analys.dart';

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const CameraScreen({Key? key, required this.cameras}) : super(key: key);

  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  bool _isTakingPicture = false;
  bool _showInstruction = false;
  FlashMode _flashMode = FlashMode.off; // Текущий режим вспышки

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _initCamera() async {
    if (_controller != null) {
      await _controller!.dispose();
    }

    if (widget.cameras.isEmpty) {
      print("Камеры не найдены");
      return;
    }

    _controller = CameraController(
      widget.cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    try {
      await _controller!.initialize();
      // Устанавливаем начальный режим вспышки
      await _controller!.setFlashMode(_flashMode);
      if (mounted) setState(() {});
    } catch (e) {
      print("Ошибка инициализации камеры: $e");
    }
  }

  Future<void> _toggleFlash() async {
    if (_controller == null || !_controller!.value.isInitialized) return;

    setState(() {
      _flashMode = switch (_flashMode) {
        FlashMode.off => FlashMode.auto,
        FlashMode.auto => FlashMode.always,
        FlashMode.always => FlashMode.torch,
        FlashMode.torch => FlashMode.off,
      };
    });

    try {
      await _controller!.setFlashMode(_flashMode);
    } catch (e) {
      print("Ошибка при изменении вспышки: $e");
    }
  }

  Future<void> _takePicture() async {
    if (_controller == null ||
        !_controller!.value.isInitialized ||
        _isTakingPicture) {
      return;
    }

    setState(() => _isTakingPicture = true);

    try {
      final image = await _controller!.takePicture();
      final confirmed = await Navigator.push<bool>(
        context,
        MaterialPageRoute(builder: (_) => PreviewScreen(imagePath: image.path)),
      );

      if (confirmed == true) {
        print("Фото подтверждено: ${image.path}");
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => AnalysisScreen(imagePath: image.path),
          ),
        );
      } else {
        print("Фото отменено");
      }
    } catch (e) {
      print("Ошибка при съемке: $e");
    } finally {
      if (mounted) setState(() => _isTakingPicture = false);
    }
  }

  void _toggleInstruction() {
    setState(() {
      _showInstruction = !_showInstruction;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Камера'), centerTitle: true),
      body: Stack(
        children: [
          if (_controller != null && _controller!.value.isInitialized)
            CameraPreview(_controller!)
          else
            const Center(child: CircularProgressIndicator()),

          if (_showInstruction)
            Positioned(
              left: MediaQuery.of(context).size.width * 0.1,
              right: MediaQuery.of(context).size.width * 0.1,
              top: MediaQuery.of(context).size.height * 0.1,
              bottom: MediaQuery.of(context).size.height * 0.1,

              child: Container(
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.85),
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.5),
                      blurRadius: 10,
                      spreadRadius: 3,
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Icon(
                          Icons.info_outline,
                          color: Color(0xFFF2DDCC),
                        ),
                        const SizedBox(width: 10),
                        const Text(
                          '5 заповедей',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            color: Color(0xFFF2DDCC),
                            fontSize: 20,
                          ),
                        ),
                        const Spacer(),
                        IconButton(
                          icon: const Icon(Icons.close, color: Colors.white),
                          onPressed: _toggleInstruction,
                        ),
                      ],
                    ),
                    const SizedBox(height: 20),
                    _buildInstructionText('- Хорошее освещение'),
                    _buildInstructionText('- Четкость - залог успеха.'),
                    _buildInstructionText('- Фокусируйтесь на области'),
                    _buildInstructionText(
                      '- Посторонние предметы на фото - зло',
                    ),
                    _buildInstructionText(
                      '- Камера перпендикулярна плоскости родинки',
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
      floatingActionButton: Container(
        padding: const EdgeInsets.symmetric(horizontal: 20),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            // Кнопка инструкции
            IconButton(
              icon: const Icon(Icons.info_outline, size: 32),
              color: Color(0xFF8D8376),
              onPressed: _toggleInstruction,
            ),
            // Кнопка сьемки
            FloatingActionButton(
              onPressed: _takePicture,

              backgroundColor: Color(0xFF8D8376),
              shape: CircleBorder(
                side: BorderSide(color: Color(0xFFC7C1BB), width: 3),
              ),
              mini: false,
            ),
            // Кнопка вспышки
            IconButton(
              icon: Icon(
                _flashMode == FlashMode.off
                    ? Icons.flash_off
                    : _flashMode == FlashMode.auto
                    ? Icons.flash_auto
                    : _flashMode == FlashMode.always
                    ? Icons.flash_on
                    : Icons.highlight,
                size: 32,
                color:
                    _flashMode == FlashMode.off
                        ? Color(0xFF8D8376)
                        : Colors.amber,
              ),
              onPressed: _toggleFlash,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInstructionText(String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 18),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(padding: EdgeInsets.only(top: 2)),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              text,
              style: const TextStyle(
                color: Color(0xFFF2DDCC),
                fontSize: 16,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
