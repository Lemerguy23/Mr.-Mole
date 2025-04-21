import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
import tensorflow_addons as tfa # Убедитесь, что tensorflow_addons установлен
from sklearn.utils import resample
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os


def del_big_data(df):
    df_majority = df[df['label'] == 0]
    df_minority = df[df['label'] == 1]
    n_benign_target = min(len(df_majority), len(df_minority))

    df_majority_downsampled = resample(df_majority,
                                        replace=False,
                                        n_samples=n_benign_target,
                                        random_state=42)

    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    return df_balanced.reset_index(drop=True)



model_weights_path = r"C:\Users\urako\OneDrive\Документы\Code\Mr.-Mole\CNN\checkpoints\model_DenseNet121_not_full150_Dp_6_AdamW_VeryBigD_Flips_We6_12_0.63.h5"

test_image_dir = r"A:\Datasets\ISIC-images"
test_csv_path = r"A:\Datasets\ISIC-images\challenge-2020-test_metadata_2025-04-20.csv"

IMG_WIDTH = (224, 224)
IMG_CHANNELS = 3
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

cancer = {
    'benign': 0,
    'malignant': 1,
}

print("--- Загрузка и подготовка тестовых данных ---")
test_df = pd.read_csv(test_csv_path)

test_df['filepath'] = test_df['isic_id'].apply(lambda x: os.path.join(test_image_dir, x + ".jpg"))

test_df['label'] = test_df['benign_malignant'].map(cancer)
test_df = del_big_data(test_df)

# Проверка на пропуски после маппинга
if test_df['label'].isnull().any():
    print("Предупреждение: Есть пропуски в метках после маппинга. Возможно, в 'benign_malignant' есть неожиданные значения.")
    print("Значения в 'benign_malignant':", test_df['benign_malignant'].unique())
    print("Пропущенные значения будут удалены.")
    test_df = test_df.dropna(subset=['label'])

print(f"Загружено {len(test_df)} тестовых записей.")
print("Распределение классов в тестовом датасете:")
print(test_df['label'].value_counts(normalize=True))

def load_and_preprocess_test_image(path):
    image = tf.io.read_file(path)
    try:
        image = tf.io.decode_jpeg(image, channels=IMG_CHANNELS)
    except tf.errors.InvalidArgumentError:
        # Пробуем PNG, если JPEG не сработал
        try:
             image = tf.io.decode_png(image, channels=IMG_CHANNELS)
        except tf.errors.InvalidArgumentError:
             print(f"Ошибка декодирования файла: {path}. Пропускаем.")
             # Возвращаем тензор нулей или вызываем исключение, чтобы пропустить изображение
             # Здесь вернем тензор нулей, чтобы не прерывать процесс, но это нужно учитывать
             return tf.zeros([IMG_WIDTH[0], IMG_WIDTH[1], IMG_CHANNELS], dtype=tf.float32)


    image = tf.image.resize(image, IMG_WIDTH)
    image = image / 255.0
    return image


test_paths = test_df['filepath'].tolist()
test_labels = test_df['label'].tolist() # Нам нужны истинные метки для оценки

# Создаем датасет только из путей для предсказания
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)

test_ds = (
    test_ds
    .map(load_and_preprocess_test_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE) # Используем батчи для ускорения предсказания
    .prefetch(buffer_size=AUTOTUNE)
)

print("--- Тестовый датасет готов для предсказания ---")

# --- Воссоздание архитектуры модели ---
print("--- Воссоздание архитектуры модели ---")
input_tensor = keras.Input(shape=(IMG_WIDTH[0], IMG_WIDTH[1], IMG_CHANNELS), name='image_input')

base_model = DenseNet121(
    weights=None, # НЕ загружаем веса imagenet, загрузим свои ниже
    include_top=False,
    input_tensor=input_tensor
)
# Замораживать/размораживать слои здесь не нужно, т.к. мы загрузим
# финальное состояние весов, где слои уже имеют нужный trainable статус.
# Но для ясности можно повторить разморозку, если она была в конце обучения:
# base_model.trainable = True # Или установите trainable как было в конце обучения

x = base_model.output
x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
# Dropout используется и при предсказании (inference), если не отключен специально
# Если вы хотите его отключить для предсказания, можно пересоздать модель без него
# или использовать model(..., training=False), но model.predict обычно сам это делает.
x = layers.Dropout(0.6, name='head_dropout')(x) # Тот же dropout, что и при обучении
output_tensor = layers.Dense(1, activation='sigmoid', name='class_predictions')(x)

model = keras.Model(inputs=input_tensor, outputs=output_tensor, name='densenet121_transfer_test')

# Компиляция не обязательна для predict, но может быть полезна для evaluate
# Используйте те же метрики, что и при обучении
model.compile(
    optimizer=tfa.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4), # Параметры LR/WD здесь не влияют на predict
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print("--- Загрузка весов модели ---")
model.load_weights(model_weights_path)
print(f"Веса успешно загружены из {model_weights_path}")

# --- Получение предсказаний ---
print("--- Получение предсказаний модели ---")
predictions = model.predict(test_ds, verbose=1)

# predictions будет массивом вероятностей (значения между 0 и 1)
# Преобразуем вероятности в классы (0 или 1) с порогом 0.5
predicted_classes = (predictions > 0.5).astype(int).flatten() # flatten() для преобразования в 1D массив

# --- Оценка производительности ---
print("\n--- Оценка производительности ---")

# Истинные метки (убедимся, что порядок соответствует предсказаниям)
y_true = np.array(test_labels)
y_pred_prob = predictions.flatten() # Вероятности для AUC
y_pred_class = predicted_classes

# 1. Accuracy (Точность) - доля правильных ответов
accuracy = accuracy_score(y_true, y_pred_class)
print(f"Accuracy: {accuracy:.4f}")

# 2. AUC (Area Under the ROC Curve) - хорошо для несбалансированных данных
auc = roc_auc_score(y_true, y_pred_prob)
print(f"AUC: {auc:.4f}")

# 3. Precision, Recall, F1-score (для класса 'malignant' = 1)
#    Precision: TP / (TP + FP) - Доля верных срабатываний среди всех срабатываний
#    Recall (Sensitivity): TP / (TP + FN) - Доля найденных позитивных объектов
#    F1-score: Гармоническое среднее Precision и Recall
precision = precision_score(y_true, y_pred_class, pos_label=1) # pos_label=1 для 'malignant'
recall = recall_score(y_true, y_pred_class, pos_label=1)       # pos_label=1 для 'malignant'
f1 = f1_score(y_true, y_pred_class, pos_label=1)               # pos_label=1 для 'malignant'
print(f"\nМетрики для позитивного класса (malignant = 1):")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# 4. Classification Report - подробный отчет по всем классам
print("\nClassification Report:")
# target_names нужны для красивого вывода отчета
# Убедитесь, что порядок соответствует меткам 0 и 1
target_names = ['benign (0)', 'malignant (1)']
print(classification_report(y_true, y_pred_class, target_names=target_names))

# 5. Confusion Matrix (Матрица ошибок)
print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred_class)
print(cm)

# Визуализация Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

print("\n--- Оценка завершена ---")

# Дополнительно: можно сохранить результаты в CSV
# results_df = test_df.copy()
# results_df['predicted_probability'] = y_pred_prob
# results_df['predicted_label'] = y_pred_class
# results_df.to_csv('test_predictions.csv', index=False)
# print("Результаты предсказаний сохранены в test_predictions.csv")