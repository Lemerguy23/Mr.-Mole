import 'dart:convert';

class ScanHistoryItem {
  final String id;
  final String imagePath;
  final String result;
  final DateTime timestamp;

  ScanHistoryItem({
    required this.id,
    required this.imagePath,
    required this.result,
    required this.timestamp,
  });

  // Создание из JSON
  factory ScanHistoryItem.fromJson(Map<String, dynamic> json) {
    return ScanHistoryItem(
      id: json['id'],
      imagePath: json['imagePath'],
      result: json['result'],
      timestamp: DateTime.parse(json['timestamp']),
    );
  }

  // Преобразование в JSON
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'imagePath': imagePath,
      'result': result,
      'timestamp': timestamp.toIso8601String(),
    };
  }

  // Преобразование объекта в строку для сохранения
  static String encode(List<ScanHistoryItem> items) {
    return json.encode(
      items.map<Map<String, dynamic>>((item) => item.toJson()).toList(),
    );
  }

  // Преобразование строки в список объектов
  static List<ScanHistoryItem> decode(String items) {
    return (json.decode(items) as List<dynamic>)
        .map<ScanHistoryItem>((item) => ScanHistoryItem.fromJson(item))
        .toList();
  }
}
