# Mr.Mole — сервис ИИ для диагностики новообразований кожи

**Mr.Mole — это кроссплатформенное решение для раннего выявления раковых образований по фото родинок.** Сервис включает мобильное приложение (APK готово для установки) и веб-интерфейс (в будущем планируется запуск). В основе системы — обученная на датасетах ISIC2020 и ISIC2024 нейросеть (архитектура EfficientNetV2M), способная точно классифицировать изображения родинок. Высокие метрики модели (точность 93.5%, специфичность 95%, чувствительность 92%, F1-score 93.4%) обеспечивают надежную диагностику.

Mr.Mole не является медицинским устройством и не заменяет консультацию квалифицированного врача-дерматолога. Результаты, предоставляемые сервисом, носят исключительно информационный и рекомендательный характер. Диагноз может поставить только специалист после проведения всех необходимых исследований. Не используйте это приложение для принятия решений о лечении. При любых сомнениях относительно здоровья вашей кожи **незамедлительно обратитесь к врачу**!

## Основные возможности

* **ИИ-диагностика по фото:** автоматический анализ изображения родинки и определение риска злокачественности.
* **История сканирований:** журнал всех проведенных сканирований для отслеживания динамики изменений со временем.
* **Умные напоминания:** настраиваемые уведомления о необходимости повторного сканирования или посещения врача.
* **Кроссплатформенность:** мобильное приложение для Android (возможно расширение на iOS) и веб-интерфейс для доступа с компьютера.
* **Дополнительные материалы:** встроенные рекомендации по уходу за кожей и ответы на часто задаваемые вопросы (FAQ).

## Целевая аудитория

* Пользователи смартфонов, следящие за своим здоровьем и регулярно проводящие самообследование кожи.
* Люди из регионов с ограниченным доступом к дерматологам и квалифицированной медицинской диагностике.
* Пациенты и их родственники, стремящиеся к раннему выявлению симптомов меланомы и других кожных раковых заболеваний.
* Исследователи и специалисты, заинтересованные в мобильных медицинских ИИ-решениях и телемедицине.

## Описание архитектуры

* **Модель ИИ:** Обученная на Keras модель на основе сверточной сети EfficientNetV2M (предобученная на ImageNet) с дообучением на медицине. Архитектура модели: последовательность слоев EfficientNetV2M, слой Dropout, глобальное усредняющее пулинг, затем полносвязный слой на один нейрон с `sigmoid`-активацией для бинарной классификации (раковая/нераковая родинка).
* **Backend:** Python-приложение на FastAPI. При старте сервера загружается TensorFlow Lite-модель, после чего входящие HTTP-запросы с изображением обрабатываются функцией `predict_mole_tflite`, возвращающей вероятность наличия рака.
* **Шаблонизация:** Для веб-страниц используется Mako Templates, а статика (CSS/JS) обслуживается через FastAPI StaticFiles.
* **Мобильное приложение:** Разработано на Flutter (Dart) для Android. Приложение отправляет фото родинки на сервер или работает с локальным TFLite-модулем для получения результата.
* **Инфраструктура:** Сервер запускается через Uvicorn/ASGI. Клиент-серверное взаимодействие по HTTP(S), данные хранятся временно (фото обрабатываются и не сохраняются на сервере дольше сеанса).

## Установка и запуск сервера

1. **Подготовка окружения:** Установите Python 3.9 или выше. Склонируйте репозиторий проекта:

   ```bash
   git clone https://github.com/your-org/mr-mole.git
   cd mr-mole/site
   ```
2. **Установка зависимостей:** Установите необходимые пакеты:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   В `requirements.txt` указаны FastAPI, Uvicorn, TensorFlow, Pillow и другие зависимости.
3. **Запуск сервера:** В папке `site` выполните:

   ```bash
   python main.py
   ```

   либо через Uvicorn:

   ```bash
   uvicorn main:app --reload
   ```

   После запуска сервер по умолчанию доступен по адресу `http://127.0.0.1:8000/`.
4. **Доступ к веб-интерфейсу:** В браузере перейдите по указанному адресу. Пока веб-интерфейс находится в разработке, возможен ограниченный функционал (TODO: обновить статус веб-проекта).

## Запуск мобильного приложения

1. **Установка APK:** Файл `MrMole.apk` находится в корне репозитория (или в папке `android/app/build/outputs/apk/release/`). Скопируйте APK на устройство Android и разрешите установку из неизвестных источников. Запустите инсталляцию.
2. **Работа приложения:** После установки откройте Mr.Mole. С помощью встроенной камеры сделайте фото родинки или выберите его из галереи. Приложение отправит изображение на сервер или использует локальную модель и выдаст результат с вероятностью наличия аномалии.
3. **Обновления:** Регулярно проверяйте обновления приложения и модели, чтобы использовать самые свежие данные и алгоритмы.

## Используемые технологии

* **Языки и фреймворки:** Python, FastAPI, Uvicorn, Flask (?) — не используется, заменён FastAPI, Mako. Flutter и Dart для мобильной части.
* **ИИ/ML:** TensorFlow 2, Keras, TensorFlow Lite. Модель EfficientNetV2M с дополнительными слоями Dropout и Dense.
* **Датасеты:** ISIC 2020, ISIC 2024 — крупные публичные наборы изображений кожных новообразований.
* **Прочее:** Pandas/NumPy для обработки данных, Pillow для работы с изображениями, Git для контроля версий.
* **Инструменты:** Android Studio (для Flutter), GitHub Actions (если используется CI), PyCharm/VSCode.

## Данные и модель

* **Датасеты:** ISIC2020 и ISIC2024 — наборы данных дермоскопических изображений с аннотациями «раковая/нераковая». Всего около 14 655 изображений (примерно 9673 – нераковые, 4877 – раковые образцы).
* **Подготовка данных:** Изображения нормализуются (масштабируются пиксели в \[0,1]), дополнительно применялось расширение выборки (аугментация) для повышения устойчивости модели.
* **Архитектура модели:** Используется EfficientNetV2M (предобученная на ImageNet) с дополнительными слоями: Dropout для регуляризации, затем глобальный усредняющий пулинг и Dense(1, activation=`sigmoid`). Модель обучалась на бинарной классификации.
* **Формат модели:** После обучения модель конвертируется в формат TensorFlow Lite (`.tflite`) для использования как на сервере, так и в мобильном приложении.

## Метрики

| Метрика                        | Значение |
| ------------------------------ | -------- |
| Точность (Accuracy)            | 93.5%    |
| Специфичность (Specificity)    | 95%      |
| Чувствительность (Sensitivity) | 92%      |
| F1-score                       | 93.4%    |

Эти результаты демонстрируют высокую эффективность модели в определении раковых образований по фото родинок.

## Ссылки на артефакты

* **Мобильное приложение (APK):** [MrMole.apk](https://github.com/Andrey-Good/Mr.-Mole/releases/tag/v0.9)
* **Обученная модель:** `model.tflite` (включена в репозиторий)
* **Исходный код:** [GitHub-репозиторий Mr.Mole](https://github.com/Andrey-Good/Mr.-Mole)
* **Презентация проекта:** `Презентация_ОПД.pdf` (включена в репозиторий)
* **TODO:** Добавить ссылку на веб-сайт проекта, когда он станет доступен.

## Лицензия

Этот проект распространяется под открытой лицензией MIT.

---

# Mr.Mole — AI-Based Melanoma Detection Service

**Mr.Mole is a cross-platform solution for early detection of melanoma (skin cancer) from photos of moles.** The project includes an Android mobile app (APK available) and a web interface (to be completed). At its core is a neural network trained on the ISIC2020 and ISIC2024 datasets, built with Keras (EfficientNetV2M architecture). The model achieves high performance (accuracy 93.5%, specificity 95%, sensitivity 92%, F1-score 93.4%), providing reliable diagnostics.

Mr.Mole is not a medical device and is not a substitute for a professional diagnosis by a qualified dermatologist. The results provided by this service are for informational purposes only. A definitive diagnosis can only be made by a medical professional. Do not use this app to make decisions about medical treatment. If you have any concerns about your skin's health, **please consult a doctor immediately**!

## Main Features

* **Photo-based AI diagnosis:** Automatically analyze a photo of a mole and estimate the risk of malignancy.
* **Scan history:** Maintain a log of all scanned images to track changes in moles over time.
* **Smart reminders:** Configurable notifications to remind users about follow-up scans or medical check-ups.
* **Cross-platform access:** Available as an Android mobile app (and potentially iOS) and a web application.
* **Additional content:** Built-in health recommendations and an FAQ section to help users learn about skin care and cancer signs.

## Target Audience

* **Health-conscious users:** Individuals who regularly perform skin self-exams and want to monitor their moles.
* **People with limited access to specialists:** Those living in areas with few dermatologists, looking for preliminary screening tools.
* **Patients and caregivers:** Anyone seeking early detection of melanoma or other skin cancers through technology.
* **Researchers and developers:** Professionals interested in mobile health solutions and AI diagnostics for healthcare.

## Architecture

* **AI Model:** A Keras-based convolutional neural network using EfficientNetV2M as a backbone (pre-trained on ImageNet) with custom heads: a Dropout layer, global average pooling, and a final Dense(1, activation=`sigmoid`) layer for binary classification (benign vs malignant).
* **Backend:** Python application using FastAPI. On startup, the server loads the TensorFlow Lite model. Incoming images are handled by the `predict_mole_tflite` function which returns a malignancy probability.
* **Web Templates:** The web UI uses Mako templates for rendering HTML, with static assets (CSS/JS) served via FastAPI’s static file support.
* **Mobile App:** Built with Flutter (Dart) for Android. The app captures or selects an image of a mole and sends it to the server (or runs the local TFLite model) to get a result.
* **Infrastructure:** The server runs on Uvicorn (ASGI). Client-server communication is over HTTP(S). No database is used; data is processed on-the-fly and images are not saved permanently.

## Server Installation and Setup

1. **Prepare the environment:** Install Python 3.9 or later. Clone the project repository:

   ```bash
   git clone https://github.com/your-org/mr-mole.git
   cd mr-mole/site
   ```
2. **Install dependencies:** In the `site` directory, run:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes FastAPI, Uvicorn, TensorFlow, Pillow, and other libraries.
3. **Run the server:** From the `site` folder, execute:

   ```bash
   python main.py
   ```

   or using Uvicorn:

   ```bash
   uvicorn main:app --reload
   ```

   By default, the server listens on `http://127.0.0.1:8000/`.
4. **Access the web interface:** Open the above URL in a web browser. (Note: The web UI is under development; functionality may be limited. *TODO:* Update when the site is live.)

## Running the Mobile App

1. **Install the APK:** The file `MrMole.apk` is included in the repository root (or under `android/app/build/outputs/`). Transfer it to your Android device and allow installation from unknown sources. Then install it.
2. **Using the app:** Launch Mr.Mole on the device. Take a photo of a mole using the app or choose an image from the gallery. The app will send the image to the server (or use its on-device model) and display the analysis result (malignant/benign probability).
3. **Updates:** Check for updates regularly to get the latest model and app improvements.

## Technologies Used

* **Languages & Frameworks:** Python (FastAPI, Uvicorn, Mako) for the backend; Flutter (Dart) for the mobile app.
* **Machine Learning:** TensorFlow 2.x, Keras for training; TensorFlow Lite for inference.
* **Model:** EfficientNetV2M (pretrained backbone), with additional Dropout and Dense layers.
* **Datasets:** ISIC 2020 and ISIC 2024 (public dermoscopic image datasets for skin lesions).
* **Utilities:** NumPy/Pandas for data handling, Pillow for image processing.
* **Tools:** Git for version control, Android Studio (Flutter SDK) for building the app.

## Data and Model

* **Training Data:** Combined ISIC2020 and ISIC2024 datasets (over 14,000 labeled images of skin lesions: \~9,000 benign and \~5,600 malignant samples).
* **Preprocessing:** Images were resized and normalized (pixel values scaled to \[0,1]). Data augmentation (rotation, flip, etc.) was applied during training to improve generalization.
* **Model Architecture:** Uses EfficientNetV2M as a feature extractor. After its layers, a Dropout layer is added for regularization, followed by GlobalAveragePooling2D, and a final Dense layer (1 unit, sigmoid activation) for binary output.
* **Model Conversion:** The trained Keras model is saved and converted to TensorFlow Lite format (`model.tflite`) for efficient deployment on the server and mobile app.

## Metrics

| Metric      | Value |
| ----------- | ----- |
| Accuracy    | 93.5% |
| Specificity | 95%   |
| Sensitivity | 92%   |
| F1-score    | 93.4% |

These performance metrics indicate the model’s strong capability to correctly identify malignant moles while minimizing false alarms.

## Artifacts

* **Android App (APK):** [MrMole.apk](https://github.com/Andrey-Good/Mr.-Mole/releases/tag/v0.9)
* **Trained Model:** `model.tflite` (included in the repository)
* **Source Code:** [GitHub Repository Mr.Mole](https://github.com/Andrey-Good/Mr.-Mole)
* **Project Presentation:** `Презентация_ОПД.pdf` (in the repo)
* **TODO:** Add link to the official website once it’s available.

## License

This project is released under the MIT License.
