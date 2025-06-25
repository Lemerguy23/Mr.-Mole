import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.optimizers import AdamW
import tensorflow.keras.backend as K
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

parametrs = "model_EfficientNetV2M_"


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

test_image_dir = "/mnt/a/Datasets/processed_data/test_data_crop3"
test_csv_path = "/mnt/a/Datasets/processed_data/test_data_crop3/test_data_selection.csv"

IMG_WIDTH = (260, 260) #(384, 384)
IMG_CHANNELS = 3
BATCH_SIZE = 1
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
        try:
            image = tf.io.decode_png(image, channels=IMG_CHANNELS)
        except tf.errors.InvalidArgumentError:
            print(f"Ошибка декодирования файла: {path}. Пропускаем.")
            return tf.zeros([IMG_WIDTH[0], IMG_WIDTH[1], IMG_CHANNELS], dtype=tf.float32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    if height != width:
        side = tf.minimum(height, width)
        offset_height = (height - side) // 2
        offset_width = (width - side) // 2
        
        image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, side, side)

    image = tf.image.resize(image, IMG_WIDTH)
    image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
    #image /= 255.0
    return image


test_paths = test_df['filepath'].tolist()
test_labels = test_df['label'].tolist()

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)

test_ds = (
    test_ds
    .map(load_and_preprocess_test_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=AUTOTUNE)
)

print("--- Тестовый датасет готов для предсказания ---")

print("--- Воссоздание архитектуры модели ---")
input_tensor = keras.Input(shape=(IMG_WIDTH[0], IMG_WIDTH[1], IMG_CHANNELS), name='image_input')

base_model = EfficientNetV2M(
    weights='imagenet',
    include_top=False,
    input_tensor=input_tensor
)

base_model.trainable = False
print(f"Базовая модель {base_model.name} заморожена. Trainable: {base_model.trainable}")

x = base_model.output
x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
x = layers.Dropout(0.4, name='head_dropout')(x)
output_tensor = layers.Dense(1, activation='sigmoid', name='class_predictions')(x)

model = keras.Model(inputs=input_tensor, outputs=output_tensor, name='densenet121_transfer')

# model.summary()
# input()

def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-7, 1. - 1e-7)
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        return -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt))
    return loss_fn

model.compile(
    optimizer= AdamW(
        learning_rate=1e-4,
        weight_decay=1e-5,
        clipnorm=1.0
    ),
    loss= focal_loss(gamma=2.0, alpha=0.5),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

a = []
for f in os.listdir("/mnt/c/Users/urako/OneDrive/Документы/Code/Mr.-Mole/CNN/checkpoints"):
    if f.startswith(parametrs):
        try:
            model_weights_path = os.path.join("/mnt/c/Users/urako/OneDrive/Документы/Code/Mr.-Mole/CNN/checkpoints", f)

            print("--- Загрузка весов модели ---")
            model.load_weights(model_weights_path)
            print(f"Веса успешно загружены из {model_weights_path}")

            print("--- Получение предсказаний модели ---")
            predictions = model.predict(test_ds, verbose=1)

            b = []
            for i in range(200, 720, 1):
                predicted_classes = (predictions > i*0.001).astype(int).flatten()
                b.append((recall_score(test_labels, predicted_classes), recall_score(test_labels, predicted_classes, pos_label=0), f, i))

            a.append(max(b, key= lambda x: sum(x[:2])/2))

        except:
            print(f"Ошибка при обработке файла {f}")
            continue
a.sort(key= lambda x: sum(x[:2])/2)
for i in a[:10]:
    print(i)

# print("\n--- Оценка производительности ---")

# y_true = np.array(test_labels)
# y_pred_prob = predictions.flatten() 
# y_pred_class = predicted_classes

# # 1. Accuracy (Точность) - доля правильных ответов
# accuracy = accuracy_score(y_true, y_pred_class)
# print(f"Accuracy: {accuracy:.4f}")

# # 2. AUC (Area Under the ROC Curve) - хорошо для несбалансированных данных
# auc = roc_auc_score(y_true, y_pred_prob)
# print(f"AUC: {auc:.4f}")

# # 3. Precision, Recall, F1-score (для класса 'malignant' = 1)
# #    Precision: TP / (TP + FP) - Доля верных срабатываний среди всех срабатываний
# #    Recall (Sensitivity): TP / (TP + FN) - Доля найденных позитивных объектов
# #    F1-score: Гармоническое среднее Precision и Recall
# precision = precision_score(y_true, y_pred_class, pos_label=1)
# recall = recall_score(y_true, y_pred_class, pos_label=1)      
# f1 = f1_score(y_true, y_pred_class, pos_label=1)              
# print(f"\nМетрики для позитивного класса (malignant = 1):")
# print(f"Precision: {precision:.4f}")
# print(f"Recall (Sensitivity): {recall:.4f}")
# print(f"F1-Score: {f1:.4f}")

# # 4. Classification Report - подробный отчет по всем классам
# print("\nClassification Report:")
# target_names = ['benign (0)', 'malignant (1)']
# print(classification_report(y_true, y_pred_class, target_names=target_names))

# # 5. Confusion Matrix (Матрица ошибок)
# print("\nConfusion Matrix:")
# cm = confusion_matrix(y_true, y_pred_class)
# print(cm)

# # Визуализация Confusion Matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
# plt.title('Confusion Matrix')
# plt.ylabel('Actual Label')
# plt.xlabel('Predicted Label')
# plt.show()

# print("\n--- Оценка завершена ---")
