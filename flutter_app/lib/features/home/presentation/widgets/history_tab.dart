import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:intl/intl.dart';
import 'package:mr_mole/features/home/domain/models/scan_history_item.dart';
import 'package:mr_mole/features/home/presentation/bloc/history_bloc.dart';
import 'package:mr_mole/features/home/presentation/pages/history_detail_page.dart';
import 'package:mr_mole/features/home/data/repositories/scan_history_repository.dart';

class HistoryTab extends StatefulWidget {
  const HistoryTab({super.key});

  @override
  State<HistoryTab> createState() => _HistoryTabState();
}

class _HistoryTabState extends State<HistoryTab> {
  late HistoryBloc _historyBloc;

  @override
  void initState() {
    super.initState();
    _historyBloc = HistoryBloc(
      ScanHistoryRepository(),
    )..add(LoadHistoryEvent());
  }

  @override
  void dispose() {
    _historyBloc.close();
    super.dispose();
  }

  void _showDeleteConfirmationDialog(BuildContext context, String id) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Удаление записи'),
        content:
            const Text('Вы уверены, что хотите удалить эту запись из истории?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Отмена'),
          ),
          TextButton(
            onPressed: () {
              _historyBloc.add(RemoveHistoryItemEvent(id));
              Navigator.of(context).pop();
            },
            child: const Text('Удалить'),
          ),
        ],
      ),
    );
  }

  void _showClearHistoryDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Очистка истории'),
        content: const Text(
            'Вы уверены, что хотите удалить всю историю сканирований?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Отмена'),
          ),
          TextButton(
            onPressed: () {
              _historyBloc.add(ClearHistoryEvent());
              Navigator.of(context).pop();
            },
            child: const Text('Очистить'),
          ),
        ],
      ),
    );
  }

  void _navigateToDetail(BuildContext context, ScanHistoryItem item) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => HistoryDetailPage(item: item),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return BlocProvider.value(
      value: _historyBloc,
      child: BlocBuilder<HistoryBloc, HistoryState>(
        builder: (context, state) {
          if (state is HistoryInitial || state is HistoryLoading) {
            return const Center(
              child: CircularProgressIndicator(),
            );
          }

          if (state is HistoryError) {
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
                  const SizedBox(height: 16),
                  ElevatedButton(
                    onPressed: () => _historyBloc.add(LoadHistoryEvent()),
                    child: const Text('Повторить'),
                  ),
                ],
              ),
            );
          }

          if (state is HistoryEmpty) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(
                    Icons.history,
                    size: 64,
                    color: Colors.grey,
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'История пуста',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Здесь будут отображаться ваши предыдущие сканирования',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: Colors.grey,
                    ),
                  ),
                ],
              ),
            );
          }

          if (state is HistoryLoaded) {
            return Column(
              children: [
                Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text(
                        'История сканирований',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      TextButton.icon(
                        onPressed: () => _showClearHistoryDialog(context),
                        icon: const Icon(Icons.delete_forever),
                        label: const Text('Очистить'),
                      ),
                    ],
                  ),
                ),
                Expanded(
                  child: ListView.builder(
                    itemCount: state.items.length,
                    itemBuilder: (context, index) {
                      final item = state.items[index];
                      return HistoryItemCard(
                        item: item,
                        onTap: () => _navigateToDetail(context, item),
                        onDelete: () =>
                            _showDeleteConfirmationDialog(context, item.id),
                      );
                    },
                  ),
                ),
              ],
            );
          }

          return const Center(
            child: Text('Неизвестное состояние'),
          );
        },
      ),
    );
  }
}

class HistoryItemCard extends StatelessWidget {
  final ScanHistoryItem item;
  final VoidCallback onTap;
  final VoidCallback onDelete;

  const HistoryItemCard({
    super.key,
    required this.item,
    required this.onTap,
    required this.onDelete,
  });

  String _formatDate(DateTime date) {
    return DateFormat('dd.MM.yyyy HH:mm').format(date);
  }

  @override
  Widget build(BuildContext context) {
    final bool isMelanoma = item.result.contains('меланом');
    final Color statusColor = isMelanoma ? Colors.red : Colors.green;

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(12.0),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: SizedBox(
                  width: 80,
                  height: 80,
                  child: _buildImage(),
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          width: 12,
                          height: 12,
                          decoration: BoxDecoration(
                            color: statusColor,
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          isMelanoma
                              ? 'Подозрение на меланому'
                              : 'Признаков меланомы нет',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            color: statusColor,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _formatDate(item.timestamp),
                      style: TextStyle(
                        color: Colors.grey[600],
                        fontSize: 12,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      item.result,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(fontSize: 14),
                    ),
                  ],
                ),
              ),
              IconButton(
                icon: const Icon(Icons.delete_outline),
                onPressed: onDelete,
                color: Colors.grey,
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildImage() {
    try {
      final file = File(item.imagePath);
      return file.existsSync()
          ? Image.file(
              file,
              fit: BoxFit.cover,
            )
          : Container(
              color: Colors.grey[300],
              child: const Icon(Icons.image_not_supported),
            );
    } catch (e) {
      return Container(
        color: Colors.grey[300],
        child: const Icon(Icons.error),
      );
    }
  }
}
