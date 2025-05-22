import 'package:flutter/material.dart';

class FAQTab extends StatelessWidget {
  const FAQTab({super.key});

  @override
  Widget build(BuildContext context) {
    final List<FAQItem> faqItems = [
      FAQItem(
        question: 'Что такое Mr. Mole?',
        answer:
            'Mr. Mole - это приложение для анализа родинок и выявления потенциально опасных изменений кожи. Приложение не заменяет консультацию врача и служит только для предварительной оценки.',
      ),
      FAQItem(
        question: 'Как правильно сделать снимок родинки?',
        answer:
            'Для получения качественного снимка убедитесь, что: 1) Родинка находится в центре кадра, 2) Освещение равномерное, без теней, 3) Камера держится параллельно поверхности кожи, 4) Фокус настроен правильно.',
      ),
      FAQItem(
        question: 'Что означают результаты анализа?',
        answer:
            'Приложение оценивает вероятность наличия подозрительных признаков в родинке по различным критериям. Результаты представлены в процентах и цветовой шкале. Зеленый цвет означает низкий риск, желтый - средний, красный - высокий риск потенциальных проблем.',
      ),
      FAQItem(
        question: 'Как часто нужно проверять родинки?',
        answer:
            'Рекомендуется проверять родинки регулярно, примерно раз в 1-3 месяца, особенно если они находятся в местах, подверженных трению или солнечному воздействию. При обнаружении любых изменений обратитесь к дерматологу.',
      ),
      FAQItem(
        question: 'Безопасны ли мои данные?',
        answer:
            'Да, ваши изображения обрабатываются локально на устройстве и не передаются в интернет без вашего явного согласия. Мы серьезно относимся к конфиденциальности ваших медицинских данных.',
      ),
    ];

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Center(
            child: Text(
              'Часто задаваемые вопросы',
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          const SizedBox(height: 24),
          ...faqItems.map((item) => FAQExpansionTile(item: item)).toList(),
          const SizedBox(height: 32),
          const Center(
            child: Text(
              'Не нашли ответ на свой вопрос?',
              style: TextStyle(
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          const SizedBox(height: 8),
          Center(
            child: ElevatedButton.icon(
              onPressed: () {
                // В будущем здесь может быть реализован переход на страницу поддержки
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content:
                        Text('Функция будет доступна в ближайшем обновлении'),
                  ),
                );
              },
              icon: const Icon(Icons.mail_outline),
              label: const Text('Связаться с поддержкой'),
            ),
          ),
          const SizedBox(height: 32),
        ],
      ),
    );
  }
}

class FAQItem {
  final String question;
  final String answer;

  FAQItem({
    required this.question,
    required this.answer,
  });
}

class FAQExpansionTile extends StatelessWidget {
  final FAQItem item;

  const FAQExpansionTile({
    super.key,
    required this.item,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      child: ExpansionTile(
        tilePadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        childrenPadding: const EdgeInsets.only(left: 16, right: 16, bottom: 16),
        title: Text(
          item.question,
          style: const TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 16,
          ),
        ),
        children: [
          Text(
            item.answer,
            style: const TextStyle(
              fontSize: 14,
              height: 1.5,
            ),
          ),
        ],
      ),
    );
  }
}
