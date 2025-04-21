import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils import resample
import collections
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import seaborn as sns



#Обязательно делать динамический learning rate, обучать долого, с автоматическим прекращением, 

IMG_WIDTH = (224, 224)  # DenseNet обычно обучалась на 224x224
IMG_CHANNELS = 3
RANDOM_SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32

##############################################################

def del_big_data(df):
    df_majority = df[df['label'] == 0]
    df_minority = df[df['label'] == 1]
    n_benign_target = min(len(df_majority), len(df_minority) * 50)

    df_majority_downsampled = resample(df_majority,
                                        replace=False,
                                        n_samples=n_benign_target,
                                        random_state=RANDOM_SEED)

    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    return df_balanced.reset_index(drop=True)

all_filepaths = []
all_labels = []

#Загрузка данных
ISIC_2024_meta_path = "A:/Datasets/ISIC_2024_Training_Input/ISIC_2024_Training_GroundTruth.csv"
ISIC_2024_image_dir = "A:/Datasets/ISIC_2024_Training_Input/ISIC_2024_Training_Input"

ISIC_2020_meta_path = "A:/Datasets/ISIC_2020_Training_JPEG/ISIC_2020_Training_GroundTruth_v2.csv"
ISIC_2020_image_dir = "A:/Datasets/ISIC_2020_Training_JPEG/train"

HAM10000_meta_path  = "A:/Datasets/dataverse_files/HAM10000_metadata.csv"
HAM10000_image_dir  = "A:/Datasets/dataverse_files/HAM10000_images"

archive_meta_path  = "A:/Datasets/archive/metadata.csv"
archive_image_dir  = "A:/Datasets/archive/imgs_part_1"

mednode_dataset_meta_path  = "A:/Datasets/complete_mednode_dataset/complete_mednode_dataset/metadata.csv"
mednode_dataset_image_dir  = "A:/Datasets/complete_mednode_dataset/complete_mednode_dataset/foto"

# ISIC_2024
try:
    df1 = pd.read_csv(ISIC_2024_meta_path)
    df1['filepath'] = df1['isic_id'].apply(lambda x: ISIC_2024_image_dir + "/" + x + '.jpg')
    df1['label'] = df1['malignant'].astype(int)
    df1 = del_big_data(df1)
    all_filepaths.extend(df1['filepath'].tolist())
    all_labels.extend(df1['label'].tolist())

    print(f"Загружен датасет ISIC_2024: {len(df1)} записей.")
except Exception as e:
    print(f"Ошибка загрузки датасета ISIC_2024: {e}")

# ISIC_2020
try:
    df1 = pd.read_csv(ISIC_2020_meta_path)
    df1['filepath'] = df1['image_name'].apply(lambda x: ISIC_2020_image_dir + "/" + x + '.jpg')
    df1['label'] = df1['target'].astype(int)
    df1 = del_big_data(df1)
    all_filepaths.extend(df1['filepath'].tolist())
    all_labels.extend(df1['label'].tolist())
    print(f"Загружен датасет ISIC_2020: {len(df1)} записей.")
except Exception as e:
    print(f"Ошибка загрузки датасета ISIC_2020: {e}")

# HAM10000
try:

    cancer = {
    'nv': 0,
    'mel': 1,
    'bkl': 0,
    'bcc': 1,
    'akiec': 1,
    'vasc': 0,
    'df': 0
    }
    df1 = pd.read_csv(HAM10000_meta_path)
    df1['filepath'] = df1['image_id'].apply(lambda x: HAM10000_image_dir + "/" + x + '.jpg')
    df1['label'] = df1['dx'].apply(lambda x: cancer[x])
    all_filepaths.extend(df1['filepath'].tolist())
    all_labels.extend(df1['label'].tolist())
    print(f"Загружен датасет HAM10000: {len(df1)} записей.")
except Exception as e:
    print(f"Ошибка загрузки датасета HAM10000: {e}")

# archive
try:
    cancer = {
    'BCC': 1,
    'ACK': 1,
    'NEV': 0,
    'SEK': 0,
    'SCC': 1,
    'MEL': 1 
}

    df1 = pd.read_csv(archive_meta_path)
    df1['filepath'] = df1['img_id'].apply(lambda x: archive_image_dir + "/" + x)
    df1['label'] = df1['diagnostic'].apply(lambda x: cancer[x])
    all_filepaths.extend(df1['filepath'].tolist())
    all_labels.extend(df1['label'].tolist())
    print(f"Загружен датасет archive: {len(df1)} записей.")
except Exception as e:
    print(f"Ошибка загрузки датасета archive: {e}")

# mednode_dataset
try:
    df1 = pd.read_csv(mednode_dataset_meta_path)
    df1['filepath'] = df1['name'].apply(lambda x: mednode_dataset_image_dir + "/" + x)
    df1['label'] = df1['malignant'].astype(int)
    all_filepaths.extend(df1['filepath'].tolist())
    all_labels.extend(df1['label'].tolist())
    print(f"Загружен датасет mednode_dataset: {len(df1)} записей.")
except Exception as e:
    print(f"Ошибка загрузки датасета mednode_dataset: {e}")









##############################################################
# Разделение на обучающую и валидационную выборки

# Эта функция загружает, декодирует, изменяет размер и нормализует
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    try:
        image = tf.io.decode_jpeg(image, channels=IMG_CHANNELS)
    except tf.errors.InvalidArgumentError:
        image = tf.io.decode_png(image, channels=IMG_CHANNELS)

    image = tf.image.resize(image, IMG_WIDTH)
    image = image / 255.0  # Нормализация (эквивалент rescale=1./255)
    return image, label

def augment_image(image, label):
    # Применяем аугментации к изображению
    image = data_augmentation(image, training=True) # training=True важно для некоторых слоев
    return image, label

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=RANDOM_SEED),
        layers.RandomFlip("vertical", seed=RANDOM_SEED),
        layers.RandomRotation(
            factor=(25/360),
            fill_mode='nearest',
            seed=RANDOM_SEED
        ),
        layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
            fill_mode='reflect'
        ),
        layers.RandomTranslation(
            height_factor=0.1,
            width_factor=0.1,
            fill_mode='nearest',
            seed=RANDOM_SEED
        ),
        layers.RandomBrightness(
            factor=0.15,
            value_range=(0.0, 1.0)
        ),
        layers.RandomContrast(factor=0.15),
        layers.GaussianNoise(stddev=0.01)
    ],
    name="data_augmentation",
)

# data_augmentation = tf.keras.Sequential(
#     [
#         layers.RandomFlip("horizontal", seed=RANDOM_SEED),
#         layers.RandomFlip("vertical", seed=RANDOM_SEED),

#         layers.RandomRotation(
#             factor=0.25,
#             fill_mode='reflect', # 'reflect' или 'nearest' часто лучше 'constant' для краев
#             interpolation='bilinear', # Качественная интерполяция
#             seed=RANDOM_SEED
#         ),

#         layers.RandomZoom(
#             height_factor=(-0.1, 0.1),
#             width_factor=(-0.1, 0.1),
#             fill_mode='reflect',
#             interpolation='bilinear',
#             seed=RANDOM_SEED
#         ),

#         layers.RandomTranslation(
#             height_factor=0.1,
#             width_factor=0.1,
#             fill_mode='reflect',
#             interpolation='bilinear',
#             seed=RANDOM_SEED
#         ),

#         layers.RandomBrightness(
#             factor=0.2,
#             value_range=(0.0, 1.0),
#             seed=RANDOM_SEED
#         ),

#         layers.RandomContrast(
#             factor=0.2,
#             seed=RANDOM_SEED
#         ),

#         layers.GaussianNoise(
#             stddev=0.02,
#             seed=RANDOM_SEED
#         )
#     ],
#     name="aggressive_data_augmentation",
# )

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_filepaths,
    all_labels,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=all_labels
)

print(f"Размер обучающей выборки: {len(train_paths)}")
print(f"Размер валидационной выборки: {len(val_paths)}")

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

train_ds = (
    train_ds
    .shuffle(buffer_size=len(train_paths), seed=RANDOM_SEED) # Перемешивание
    .map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE) # Загрузка и базовая обработка
    .map(augment_image, num_parallel_calls=AUTOTUNE) # **Применение аугментации**
    .batch(BATCH_SIZE) # Формирование батчей
    .prefetch(buffer_size=AUTOTUNE) # Предзагрузка для производительности
)

val_ds = (
    val_ds
    .map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE) # Загрузка и базовая обработка
    .batch(BATCH_SIZE) # Формирование батчей
    .prefetch(buffer_size=AUTOTUNE) # Предзагрузка
)







##############################################################
# Создание модели

input_tensor = keras.Input(shape=(IMG_WIDTH[0], IMG_WIDTH[1], IMG_CHANNELS), name='image_input')

base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_tensor=input_tensor
)

base_model.trainable = False
print(f"Базовая модель {base_model.name} заморожена. Trainable: {base_model.trainable}")

x = base_model.output
x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
x = layers.Dropout(0.6, name='head_dropout')(x)
output_tensor = layers.Dense(1, activation='sigmoid', name='class_predictions')(x)

model = keras.Model(inputs=input_tensor, outputs=output_tensor, name='densenet121_transfer')

# model.summary()
# input()

model.compile(
    #optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    optimizer=tfa.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-5
    ),
    loss= 'binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)





##############################################################
#Работа с развесовкой классов
y_train_array = np.array(train_labels)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_array),
    y=y_train_array
)

class_weights_dict = {i : class_weights[i] for i, cls in enumerate(np.unique(y_train_array))}
class_weights_dict[1] = class_weights_dict[1] * 10
print(f"Рассчитанные веса классов: {class_weights_dict}")








##############################################################
# Контрольные точки

checkpoint_filepath = 'C:\\Users\\urako\\OneDrive\\Документы\\Code\\Mr.-Mole\\CNN\\checkpoints\\model_DenseNet121_not_full150_Dp_6_AdamW_VeryBigD_Flips_We6_{epoch:02d}_{val_loss:.2f}.h5' # Путь к файлу для сохранения
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True, # Сохранять только веса модели, а не всю модель целиком (меньше места, быстрее)
    monitor='val_loss', # Метрика, которую отслеживать для сохранения (обычно val_loss - loss на валидационной выборке)
    mode='min', # В каком направлении должна меняться метрика (min - сохранять при минимальном значении val_loss, max - при максимальном, например, val_accuracy)
    save_best_only=True, # Сохранять только лучшую модель (с наилучшим значением monitor метрики)
    verbose=1 # Выводить сообщения о сохранении контрольных точек
)








##############################################################
# Обучение модели

# lastest = "C:/Users/urako/OneDrive/Документы/Код/Mr.-Mole/checkpoints/model_11_0.67.h5"
# model.load_weights(lastest)


print("\n--- Начало обучения только 'головы' ---")
history = model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds,
    class_weight=class_weights_dict,
    callbacks=[model_checkpoint_callback],
    initial_epoch=0
)
print("--- Обучение 'головы' завершено ---")



print("\n--- Начало этапа Fine-tuning ---")
base_model.trainable = True # Размораживаем всю базовую модель
print("\n--- Summary после разморозки для Fine-tuning ---")

unfreeze_from_layer_name = 'conv4_block21_0_bn' #149 слой
unfreeze_idx = None
for idx, layer in enumerate(base_model.layers):
    if layer.name == unfreeze_from_layer_name:
        unfreeze_idx = idx
        break
if unfreeze_idx is not None:
    for i, layer in enumerate(base_model.layers):
        layer.trainable = (i >= unfreeze_idx)
    print(f"Заморожены все слои до {unfreeze_from_layer_name} (индекс {unfreeze_idx}).")
else:
    print(f"Слой с именем {unfreeze_from_layer_name} не найден!")

model.compile(
    #optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    optimizer=tfa.optimizers.AdamW(
        learning_rate=1e-5,
        weight_decay=1e-4
    ),
    loss= 'binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)


best_head_weights = tf.train.latest_checkpoint("C:\\Users\\urako\\OneDrive\\Документы\\Code\\Mr.-Mole\\CNN\\checkpoints") # Находим лучший чекпоинт первого этапа
if best_head_weights:
     print(f"Загрузка лучших весов 'головы' из: {best_head_weights} для начала fine-tuning.")
     model.load_weights(best_head_weights)
else:
     print("Предупреждение: Не найдены веса для загрузки перед fine-tuning. Продолжаем с текущими.")

print("\n--- Начало обучения всей модели (Fine-tuning) ---")

initial_epoch_fine_tuning = history.epoch[-1] + 1
epochs_fine_tuning = 50 # Сколько эпох хотим обучать на этапе fine-tuning
total_epochs = initial_epoch_fine_tuning + epochs_fine_tuning

early_stopper = EarlyStopping(monitor='val_loss',
                            patience=10,
                            verbose=1, # Выводить сообщение при остановке
                            mode='min',
                            restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
                            
print(f"\n--- Начало обучения всей модели (Fine-tuning) с эпохи {initial_epoch_fine_tuning} ---")
history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    validation_data=val_ds,
    class_weight=class_weights_dict,
    callbacks=[model_checkpoint_callback, early_stopper, reduce_lr],
    initial_epoch=initial_epoch_fine_tuning
)
print("--- Fine-tuning завершен ---")






##############################################################
#Графики обучения

# Объединяем истории обучения первого этапа и fine-tuning'а
history_dict_head = history.history
history_dict_fine = history_fine.history

acc = history_dict_head['accuracy'] + history_dict_fine['accuracy']
val_acc = history_dict_head['val_accuracy'] + history_dict_fine['val_accuracy']
loss = history_dict_head['loss'] + history_dict_fine['loss']
val_loss = history_dict_head['val_loss'] + history_dict_fine['val_loss']

epochs_range_head = range(len(history_dict_head['accuracy']))
epochs_range_fine = range(len(history_dict_head['accuracy']), len(history_dict_head['accuracy']) + len(history_dict_fine['accuracy']))
total_epochs_range = range(len(acc)) # Общий диапазон эпох

plt.figure(figsize=(12, 5))

# График точности
plt.subplot(1, 2, 1)
plt.plot(total_epochs_range, acc, label='Точность на обучении')
plt.plot(total_epochs_range, val_acc, label='Точность на валидации')
plt.axvline(len(epochs_range_head) - 1, color='gray', linestyle='--', label='Начало Fine-tuning') # Вертикальная линия
plt.legend(loc='lower right')
plt.title('Точность (Accuracy)')
plt.xlabel('Эпохи')
plt.ylabel('Точность')

# График потерь
plt.subplot(1, 2, 2)
plt.plot(total_epochs_range, loss, label='Потери на обучении')
plt.plot(total_epochs_range, val_loss, label='Потери на валидации')
plt.axvline(len(epochs_range_head) - 1, color='gray', linestyle='--', label='Начало Fine-tuning') # Вертикальная линия
plt.legend(loc='upper right')
plt.title('Потери (Loss)')
plt.xlabel('Эпохи')
plt.ylabel('Потери')

plt.suptitle('Результаты обучения модели')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()










# --- Параметры ---

# Пути к тестовым данным
test_image_dir = r"A:\Datasets\ISIC-images" # Используем raw string
test_csv_path = r"A:\Datasets\ISIC-images\challenge-2020-test_metadata_2025-04-20.csv" # Используем raw string

# Маппинг классов (как в обучении)
cancer = {
    'benign': 0, # Убедимся, что типы совпадают (строки или числа)
    'malignant': 1,
}

# --- Загрузка и подготовка тестовых данных ---
print("--- Загрузка и подготовка тестовых данных ---")
test_df = pd.read_csv(test_csv_path)

# Проверка наличия необходимых колонок
required_cols = ['isic_id', 'benign_malignant']
if not all(col in test_df.columns for col in required_cols):
    raise ValueError(f"CSV файл должен содержать колонки: {required_cols}")

# Создание полного пути к изображениям
# Используем os.path.join для кроссплатформенности
test_df['filepath'] = test_df['isic_id'].apply(lambda x: os.path.join(test_image_dir, x + ".jpg"))

# Проверка существования файлов (опционально, но полезно)
# file_exists = test_df['filepath'].apply(os.path.exists)
# if not file_exists.all():
#     print(f"Предупреждение: Не найдены файлы для {sum(~file_exists)} записей.")
#     print("Примеры не найденных файлов:")
#     print(test_df[~file_exists]['filepath'].head())
#     # Можно либо удалить строки с отсутствующими файлами, либо продолжить,
#     # но tf.data выдаст ошибку во время выполнения
#     # test_df = test_df[file_exists].reset_index(drop=True)

# Преобразование текстовых меток в числовые
# Используем .map() с преобразованием в int после маппинга
test_df['label'] = test_df['benign_malignant'].map(cancer)

# Проверка на пропуски после маппинга
if test_df['label'].isnull().any():
    print("Предупреждение: Есть пропуски в метках после маппинга. Возможно, в 'benign_malignant' есть неожиданные значения.")
    print("Значения в 'benign_malignant':", test_df['benign_malignant'].unique())
    print("Пропущенные значения будут удалены.")
    test_df = test_df.dropna(subset=['label']) # Удаляем строки с NaN метками

# Преобразование в целые числа
test_df['label'] = test_df['label'].astype(int)

print(f"Загружено {len(test_df)} тестовых записей.")
print("Распределение классов в тестовом датасете:")
print(test_df['label'].value_counts(normalize=True)) # Показывает дисбаланс

# --- Функция препроцессинга для тестовых данных ---
# Важно: она должна быть идентична препроцессингу ВАЛИДАЦИОННЫХ данных при обучении
# (без аугментации!)
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
    image = image / 255.0 # Нормализация (как в обучении)
    # Важно: НЕ используйте densenet_preprocess_input, если вы его не использовали при обучении
    # Если вы использовали densenet_preprocess_input, раскомментируйте строку ниже и закомментируйте image = image / 255.0
    # from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
    # image = densenet_preprocess_input(image)
    return image

# --- Создание tf.data.Dataset для теста ---
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
