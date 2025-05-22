import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:image_picker/image_picker.dart';
import 'package:camera/camera.dart';
import 'dart:io';

part 'home_event.dart';
part 'home_state.dart';

class HomeBloc extends Bloc<HomeEvent, HomeState> {
  final Future<List<CameraDescription>> camerasFuture;
  final ImagePicker _picker = ImagePicker();

  HomeBloc(this.camerasFuture) : super(HomeInitial()) {
    on<OpenGalleryEvent>(_onOpenGallery);
    on<OpenCameraEvent>(_onOpenCamera);
    on<OpenSettingsEvent>(_onOpenSettings);
  }

  Future<void> _onOpenGallery(
    OpenGalleryEvent event,
    Emitter<HomeState> emit,
  ) async {
    try {
      emit(HomeLoading());

      final photo = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );

      if (photo != null) {
        final file = File(photo.path);
        if (!await file.exists()) {
          throw Exception('Файл не найден');
        }

        final fileSize = await file.length();
        if (fileSize == 0) {
          throw Exception('Файл имеет нулевой размер');
        }

        final extension = photo.path.split('.').last.toLowerCase();
        if (!['jpg', 'jpeg', 'png'].contains(extension)) {
          throw Exception('Неподдерживаемый формат изображения');
        }

        print('Selected image: ${photo.path}, size: $fileSize bytes');
        emit(GalleryImageSelected(photo.path));
      } else {
        emit(HomeInitial());
      }
    } catch (e) {
      print('Gallery error: $e');
      emit(HomeError('Ошибка галереи: ${e.toString()}', source: 'gallery'));
    }
  }

  Future<void> _onOpenCamera(
    OpenCameraEvent event,
    Emitter<HomeState> emit,
  ) async {
    try {
      emit(HomeLoading());
      final cameras = await camerasFuture;

      if (cameras.isEmpty) {
        emit(HomeError('Камера недоступна', source: 'camera'));
      } else {
        emit(CameraReady(cameras));
      }
    } catch (e) {
      emit(HomeError('Ошибка камеры: ${e.toString()}', source: 'camera'));
    }
  }

  Future<void> _onOpenSettings(
    OpenSettingsEvent event,
    Emitter<HomeState> emit,
  ) async {
    emit(NavigateToSettings());
  }
}
