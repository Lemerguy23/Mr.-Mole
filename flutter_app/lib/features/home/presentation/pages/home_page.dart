import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:camera/camera.dart';
import 'package:mr_mole/features/home/presentation/bloc/home_bloc.dart';
import 'package:mr_mole/features/camera/presentation/pages/camera_page.dart';
import 'package:mr_mole/features/settings/presentation/pages/settings_page.dart';
import 'package:mr_mole/features/analysis/presentation/pages/analys.dart';
import 'package:mr_mole/core/utils/notification.dart';
import 'package:mr_mole/features/image_processing/presentation/pages/mole_confirmation_screen.dart';
import 'package:mr_mole/features/home/presentation/widgets/main_tab.dart';
import 'package:mr_mole/features/home/presentation/widgets/history_tab.dart';
import 'package:mr_mole/features/home/presentation/widgets/faq_tab.dart';
import 'package:mr_mole/features/home/data/repositories/scan_history_repository.dart';

class HomePage extends StatefulWidget {
  final Future<List<CameraDescription>> camerasFuture;
  final NotificationService notificationService;

  const HomePage({
    super.key,
    required this.camerasFuture,
    required this.notificationService,
  });

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage>
    with SingleTickerProviderStateMixin {
  late HomeBloc _homeBloc;
  late PageController _pageController;
  int _currentIndex = 1;

  @override
  void initState() {
    super.initState();
    _homeBloc = HomeBloc(widget.camerasFuture);
    _pageController = PageController(initialPage: 1);
  }

  void _resetToMainTab() {
    if (mounted) {
      setState(() {
        _currentIndex = 1;
        _pageController.animateToPage(
          1,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeInOut,
        );
      });
    }
  }

  void _navigateToAnalysis(String imagePath) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => AnalysisScreen(
          imagePath: imagePath,
          notificationService: widget.notificationService,
          onRetake: () {
            Navigator.of(context).pop();
          },
        ),
      ),
    ).then((_) {
      _resetToMainTab();
    });
  }

  Future<void> _navigateToCamera(List<CameraDescription> cameras) async {
    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => CameraPage(
          cameras: cameras,
          notificationService: widget.notificationService,
        ),
      ),
    ).then((_) {
      _resetToMainTab();
    });
  }

  Future<void> _navigateToSettings() async {
    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => const SettingsPage(),
      ),
    );
  }

  void _navigateToConfirmation(String imagePath) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => MoleConfirmationScreen(
          imagePath: imagePath,
          notificationService: widget.notificationService,
          onConfirm: (String croppedPath) {
            _navigateToAnalysis(croppedPath);
          },
          onCancel: () {
            Navigator.of(context).pop();
            _resetToMainTab();
          },
        ),
      ),
    );
  }

  void _onPageChanged(int index) {
    setState(() {
      _currentIndex = index;
    });
  }

  void _onItemTapped(int index) {
    setState(() {
      _currentIndex = index;
      _pageController.animateToPage(
        index,
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
      );
    });
  }

  @override
  void dispose() {
    _homeBloc.close();
    _pageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return BlocProvider.value(
      value: _homeBloc,
      child: BlocListener<HomeBloc, HomeState>(
        listener: (context, state) {
          if (state is GalleryImageSelected) {
            _navigateToConfirmation(state.imagePath);
          } else if (state is CameraReady) {
            _navigateToCamera(state.cameras);
          } else if (state is NavigateToSettings) {
            _navigateToSettings();
          }
        },
        child: Scaffold(
          appBar: AppBar(
            title: const Text('Mr. Mole'),
            actions: [
              IconButton(
                icon: const Icon(Icons.settings),
                onPressed: () => _homeBloc.add(OpenSettingsEvent()),
              ),
            ],
          ),
          body: BlocBuilder<HomeBloc, HomeState>(
            builder: (context, state) {
              if (state is HomeLoading) {
                return const Center(
                  child: CircularProgressIndicator(),
                );
              }

              if (state is HomeError) {
                return Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(
                        Icons.error_outline,
                        color: Colors.red,
                        size: 48,
                      ),
                      const SizedBox(height: 16),
                      Text(
                        state.message,
                        style: const TextStyle(
                          fontSize: 18,
                          color: Colors.red,
                        ),
                      ),
                    ],
                  ),
                );
              }

              return PageView(
                controller: _pageController,
                onPageChanged: _onPageChanged,
                children: [
                  const HistoryTab(),
                  MainTab(
                    homeBloc: _homeBloc,
                  ),
                  const FAQTab(),
                ],
              );
            },
          ),
          bottomNavigationBar: BottomNavigationBar(
            currentIndex: _currentIndex,
            onTap: _onItemTapped,
            selectedItemColor: Theme.of(context).colorScheme.primary,
            unselectedItemColor: Colors.grey,
            items: const [
              BottomNavigationBarItem(
                icon: Icon(Icons.history),
                label: 'История',
              ),
              BottomNavigationBarItem(
                icon: Icon(Icons.home),
                label: 'Главная',
              ),
              BottomNavigationBarItem(
                icon: Icon(Icons.question_answer),
                label: 'FAQ',
              ),
            ],
          ),
        ),
      ),
    );
  }
}
