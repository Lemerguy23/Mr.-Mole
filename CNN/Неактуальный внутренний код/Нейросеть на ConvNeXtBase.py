import pandas as pd
import numpy as np
import os
import math
import cv2

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import ConvNeXtBase
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import layers
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess_input
from tensorflow.keras.callbacks import TerminateOnNaN
import tensorflow.keras.backend as K
from tensorflow import keras

import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

import albumentations as A

from keras_unet.models import custom_unet
from PIL import Image

from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy("mixed_float16")


#Обязательно делать динамический learning rate, обучать долого, с автоматическим прекращением, 

IMG_WIDTH = (224, 224)  #384×384 надо попробовать в будующем
IMG_CHANNELS = 3
RANDOM_SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16
CROP = False
COEF_FOTO = 15
COEF_POSITIVE = 1
COEF_NEGATIVE = 1
ALPHA = 0.5
parametrs = "model_ConvNeXtBase_2"


##############################################################

def del_big_data(df):                                                     #Применяется к ISIC_2024 и ISIC_2020 так как в них очень сильный дисбаланс классов, кроме того самих изображений очень много и поэтому их нужно уменьшить.
    df_majority = df[df['label'] == 0]
    df_minority = df[df['label'] == 1]
    n_benign_target = min(len(df_majority), len(df_minority) * COEF_FOTO)

    df_majority_downsampled = resample(df_majority,
                                        replace=False,
                                        n_samples=n_benign_target,
                                        random_state=RANDOM_SEED)

    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    return df_balanced.reset_index(drop=True)

all_filepaths = []
all_labels = []

# #Загрузка данных
# ISIC_2024_meta_path = "A:/Datasets/ISIC_2024_Training_Input/ISIC_2024_Training_GroundTruth.csv"
# ISIC_2024_image_dir = "A:/Datasets/ISIC_2024_Training_Input/ISIC_2024_Training_Input"

# ISIC_2020_meta_path = "A:/Datasets/ISIC_2020_Training_JPEG/ISIC_2020_Training_GroundTruth_v2.csv"
# ISIC_2020_image_dir = "A:/Datasets/ISIC_2020_Training_JPEG/train"

# HAM10000_meta_path  = "A:/Datasets/dataverse_files/HAM10000_metadata.csv"
# HAM10000_image_dir  = "A:/Datasets/dataverse_files/HAM10000_images"

# archive_meta_path  = "A:/Datasets/archive/metadata.csv"
# archive_image_dir  = "A:/Datasets/archive/imgs_part_1"

# mednode_dataset_meta_path  = "A:/Datasets/complete_mednode_dataset/complete_mednode_dataset/metadata.csv"
# mednode_dataset_image_dir  = "A:/Datasets/complete_mednode_dataset/complete_mednode_dataset/foto"

#Загрузка данных
ISIC_2024_meta_path = "/mnt/a/Datasets/ISIC_2024_Training_Input/ISIC_2024_Training_GroundTruth.csv"
ISIC_2024_image_dir = "/mnt/a/Datasets/ISIC_2024_Training_Input/ISIC_2024_Training_Input" + ("_crop" if CROP else "")

ISIC_2020_meta_path = "/mnt/a/Datasets/ISIC_2020_Training_JPEG/ISIC_2020_Training_GroundTruth_v2.csv"
ISIC_2020_image_dir = "/mnt/a/Datasets/ISIC_2020_Training_JPEG/train" + ("_crop" if CROP else "")

HAM10000_meta_path  = "/mnt/a/Datasets/dataverse_files/HAM10000_metadata.csv"
HAM10000_image_dir  = "/mnt/a/Datasets/dataverse_files/HAM10000_images"

archive_meta_path  = "/mnt/a/Datasets/archive/metadata.csv"
archive_image_dir  = "/mnt/a/Datasets/archive/imgs_part_1"

mednode_dataset_meta_path  = "/mnt/a/Datasets/complete_mednode_dataset/complete_mednode_dataset/metadata.csv"
mednode_dataset_image_dir  = "/mnt/a/Datasets/complete_mednode_dataset/complete_mednode_dataset/foto"

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





##############################################################
# Разделение на обучающую и валидационную выборки

# Эта функция загружает, декодирует, изменяет размер и нормализует
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    try:
        image = tf.io.decode_jpeg(image, channels=IMG_CHANNELS)
    except tf.errors.InvalidArgumentError:
        image = tf.io.decode_png(image, channels=IMG_CHANNELS)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    if height != width:
        side = tf.minimum(height, width)
        offset_height = (height - side) // 2
        offset_width = (width - side) // 2

        image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, side, side)

    image = tf.image.resize(image, IMG_WIDTH)
    #image = convnext_preprocess_input(image)
    image = image / 255.0
    return image, label

albumentations_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.7),
    # A.HueSaturationValue(p=0.5)
    A.Rotate(limit=45, p=0.7),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
])

def apply_albumentations(image, label):
    # image: tf.Tensor, [H,W,3], float32 [0,1]
    def _aug(img_np):
        # img_np: numpy массив uint8 [0,255]
        augmented = albumentations_transform(image=img_np)
        return augmented['image']
    # приводим к uint8 и обратно к float32
    img_uint8 = tf.image.convert_image_dtype(image, tf.uint8)
    aug_img = tf.numpy_function(_aug, [img_uint8], tf.uint8)
    # восстанавливаем формат float32 [0,1]
    aug_img = tf.image.convert_image_dtype(aug_img, tf.float32)
    aug_img.set_shape([IMG_WIDTH[0], IMG_WIDTH[1], IMG_CHANNELS])
    return aug_img, label

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
    #.map(augment_image, num_parallel_calls=AUTOTUNE) # **Применение аугментации**
    .map(apply_albumentations, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE) # Формирование батчей
    #.map(augment_and_mix, num_parallel_calls=AUTOTUNE)
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

#Построение объединённой модели
input_tensor = keras.Input(shape=(*IMG_WIDTH, IMG_CHANNELS), name='image_input')

base_model = ConvNeXtBase(
    weights='imagenet',
    include_top=False
)

base_model.trainable = False
print(f"Базовая модель {base_model.name} заморожена. Trainable: {base_model.trainable}")


x = base_model(input_tensor)  
x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
x = layers.Dropout(0.2, name='head_dropout')(x)
output_tensor = layers.Dense(1, activation='sigmoid', name='class_predictions')(x)

model = keras.Model(inputs=input_tensor, outputs=output_tensor)

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
        learning_rate=1e-3,
        weight_decay=1e-5,
        clipnorm=1.0
    ),
    loss= focal_loss(gamma=2.0, alpha=ALPHA),
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
class_weights_dict[1] = class_weights_dict[1] * COEF_POSITIVE  #Обычная балансировка не помогает, приходится вручную увеличивать вес раковых родинок, тогда на тестовых данных оба значения будут иметь примерно одинаковую точность, иначе злокачественность будет плохо определяться.
class_weights_dict[0] = class_weights_dict[0] * COEF_NEGATIVE
print(f"Рассчитанные веса классов: {class_weights_dict}")







##############################################################
# Контрольные точки
checkpoint_filepath = '/mnt/c/Users/urako/OneDrive/Документы/Code/Mr.-Mole/CNN/checkpoints/' + parametrs + '_{epoch:02d}_{val_loss:.5f}.weights.h5' # Путь к файлу для сохранения
model_checkpoint_callback_1 = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True, # Сохранять только веса модели, а не всю модель целиком (меньше места, быстрее)
    monitor='val_loss', # Метрика, которую отслеживать для сохранения (обычно val_loss - loss на валидационной выборке)
    mode='min', # В каком направлении должна меняться метрика (min - сохранять при минимальном значении val_loss, max - при максимальном, например, val_accuracy)
    save_best_only=True, # Сохранять только лучшую модель (с наилучшим значением monitor метрики)
    verbose=1 # Выводить сообщения о сохранении контрольных точек
)


model_checkpoint_callback_2 = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True, # Сохранять только веса модели, а не всю модель целиком (меньше места, быстрее)
    monitor='val_loss', # Метрика, которую отслеживать для сохранения (обычно val_loss - loss на валидационной выборке)
    mode='min', # В каком направлении должна меняться метрика (min - сохранять при минимальном значении val_loss, max - при максимальном, например, val_accuracy)
    save_best_only=not True, # Сохранять только лучшую модель (с наилучшим значением monitor метрики)
    verbose=1 # Выводить сообщения о сохранении контрольных точек
)






##############################################################
# Подготовка к обучению

# lastest = "C:/Users/urako/OneDrive/Документы/Код/Mr.-Mole/checkpoints/model_11_0.67.h5"
# model.load_weights(lastest)
 
# Параметры One‑Cycle
max_lr = 1e-4
div_factor = 25.0               # lr_start = max_lr/div_factor
final_div_factor = 1e4          # минимальный lr = lr_start/final_div_factor
epochs_head = 8
steps_per_epoch = len(train_paths) // BATCH_SIZE
total_steps = epochs_head * steps_per_epoch
warmup_steps = int(total_steps * 0.3)      # 30% на «разогрев»
cooldown_steps = total_steps - warmup_steps

lr_start = max_lr / div_factor
lr_end = lr_start / final_div_factor

def one_cycle_schedule(step):
    if step < warmup_steps:
        # линейный рост от lr_start до max_lr
        return lr_start + (max_lr - lr_start) * (step / warmup_steps)
    else:
        # косинусный спад от max_lr до lr_end
        progress = (step - warmup_steps) / cooldown_steps
        return lr_end + 0.5 * (max_lr - lr_end) * (1 + math.cos(math.pi * progress))


class BatchLRScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule_fn):
        super().__init__()
        self.schedule_fn = schedule_fn
        self.step = 0

    def on_train_batch_begin(self, batch, logs=None):
        lr = self.schedule_fn(self.step)
        current_optimizer = self.model.optimizer
        if isinstance(current_optimizer, mixed_precision.LossScaleOptimizer):
            current_optimizer = current_optimizer.inner_optimizer

        try:
            current_optimizer.learning_rate.assign(lr)
        except Exception as e:
            print(f"Предупреждение: Не удалось установить LR через .assign(). Тип атрибута LR: {type(current_optimizer.learning_rate)}. Пытаемся через backend.set_value. Ошибка: {e}")
            try:
                tf.keras.backend.set_value(current_optimizer.learning_rate, lr)
            except Exception as e2:
                raise RuntimeError(f"Не удалось установить learning_rate для оптимизатора {current_optimizer} "
                                   f"с атрибутом LR типа {type(current_optimizer.learning_rate)}. "
                                   f"Ошибка при .assign(): {e}. Ошибка при backend.set_value(): {e2}")
        
        self.step += 1


one_cycle_cb = BatchLRScheduler(one_cycle_schedule)






##############################################################
# Обучение модели

print("\n--- Начало обучения только 'головы' ---")

history = model.fit(
    train_ds,
    epochs=epochs_head,
    validation_data=val_ds,
    class_weight=class_weights_dict,
    callbacks=[model_checkpoint_callback_1, TerminateOnNaN(), one_cycle_cb],
    initial_epoch=0
)
print("--- Обучение 'головы' завершено ---")

x = base_model(input_tensor)  
x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
x = layers.Dropout(0.4, name='head_dropout')(x)
output_tensor = layers.Dense(1, activation='sigmoid', name='class_predictions')(x)

model = keras.Model(inputs=input_tensor, outputs=output_tensor)

print("\n--- Начало этапа Fine-tuning ---")
base_model.trainable = True # Размораживаем всю базовую модель
print("\n--- Summary после разморозки для Fine-tuning ---")

unfreeze_from_layer_name = 'convnext_base_stage_1_block_0_depthwise_conv' #7 слой
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
    optimizer= AdamW(
        learning_rate=1e-5,
        weight_decay=1e-4,
        clipnorm=1.0
    ),
    loss= focal_loss(gamma=2.0, alpha=ALPHA),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)


weights_dir = "/mnt/c/Users/urako/OneDrive/Документы/Code/Mr.-Mole/CNN/checkpoints" # Находим лучший чекпоинт первого этапа
try:
     print(f"Загрузка лучших весов 'головы' из: {weights_dir} для начала fine-tuning.")
     weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.weights.h5') and f.startswith(parametrs)]
     weight_files.sort(reverse=True)
     best_weight_path = os.path.join(weights_dir, weight_files[0])
     model.load_weights(best_weight_path)
except Exception as e:
    print(f"[Ошибка загрузки] Не удалось загрузить веса: {e}")
    raise

print("\n--- Начало обучения всей модели (Fine-tuning) ---")

initial_epoch_fine_tuning = history.epoch[-1] + 1
epochs_fine_tuning = 50 # Сколько эпох хотим обучать на этапе fine-tuning
total_epochs = initial_epoch_fine_tuning + epochs_fine_tuning





# Параметры
lr_start_ft = 1e-5
lr_end_ft   = 1e-8
steps_ft    = epochs_fine_tuning * steps_per_epoch

cosine_schedule = CosineDecay(
    initial_learning_rate=lr_start_ft,
    decay_steps=steps_ft,
    alpha=lr_end_ft / lr_start_ft   # lr будет спускаться до lr_start_ft*alpha = lr_end_ft
)

def ft_lr_schedule(global_step):
    # global_step — номер батча от начала Fine‑tuning
    return cosine_schedule(global_step)

ft_lr_cb = BatchLRScheduler(ft_lr_schedule)

early_stopper = EarlyStopping(monitor='val_loss',
                            patience=15,
                            verbose=1, # Выводить сообщение при остановке
                            mode='min',
                            restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-8, verbose=1)
                            




print(f"\n--- Начало обучения всей модели (Fine-tuning) с эпохи {initial_epoch_fine_tuning} ---")

history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    validation_data=val_ds,
    class_weight=class_weights_dict,
    callbacks=[model_checkpoint_callback_2, early_stopper, ft_lr_cb, TerminateOnNaN()],
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