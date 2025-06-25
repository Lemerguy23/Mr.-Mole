import pandas as pd
import numpy as np
import os
import math

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import layers
from tensorflow.keras.optimizers.schedules import CosineDecay
import tensorflow.keras.backend as K
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

import albumentations as A

from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy("mixed_float16")



tf.keras.backend.clear_session()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Включает "растущее" потребление памяти (по мере надобности)
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

#Обязательно делать динамический learning rate, обучать долого, с автоматическим прекращением, 

IMG_WIDTH = (260, 260)  
IMG_CHANNELS = 3
RANDOM_SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 8
COEF_FOTO = 15
COEF_POSITIVE = 1
COEF_NEGATIVE = 1
parametrs = "model_EfficientNetV2M_not_full150_Dp_2,4_VeryBigD_480_notManyDT_albumentations_Coef15,1"


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
ISIC_2024_image_dir = "/mnt/a/Datasets/ISIC_2024_Training_Input/ISIC_2024_Training_Input"

ISIC_2020_meta_path = "/mnt/a/Datasets/ISIC_2020_Training_JPEG/ISIC_2020_Training_GroundTruth_v2.csv"
ISIC_2020_image_dir = "/mnt/a/Datasets/ISIC_2020_Training_JPEG/train"

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

# # HAM10000
# try:

#     cancer = {
#     'nv': 0,
#     'mel': 1,
#     'bkl': 0,
#     'bcc': 1,
#     'akiec': 1,
#     'vasc': 0,
#     'df': 0
#     }
#     df1 = pd.read_csv(HAM10000_meta_path)
#     df1['filepath'] = df1['image_id'].apply(lambda x: HAM10000_image_dir + "/" + x + '.jpg')
#     df1['label'] = df1['dx'].apply(lambda x: cancer[x])
#     all_filepaths.extend(df1['filepath'].tolist())
#     all_labels.extend(df1['label'].tolist())
#     print(f"Загружен датасет HAM10000: {len(df1)} записей.")
# except Exception as e:
#     print(f"Ошибка загрузки датасета HAM10000: {e}")

# # archive
# # try:
# #     cancer = {
# #     'BCC': 1,
# #     'ACK': 1,
# #     'NEV': 0,
# #     'SEK': 0,
# #     'SCC': 1,
# #     'MEL': 1 
# # }

# #     df1 = pd.read_csv(archive_meta_path)
# #     df1['filepath'] = df1['img_id'].apply(lambda x: archive_image_dir + "/" + x)
# #     df1['label'] = df1['diagnostic'].apply(lambda x: cancer[x])
# #     all_filepaths.extend(df1['filepath'].tolist())
# #     all_labels.extend(df1['label'].tolist())
# #     print(f"Загружен датасет archive: {len(df1)} записей.")
# # except Exception as e:
# #     print(f"Ошибка загрузки датасета archive: {e}")

# # mednode_dataset
# try:
#     df1 = pd.read_csv(mednode_dataset_meta_path)
#     df1['filepath'] = df1['name'].apply(lambda x: mednode_dataset_image_dir + "/" + x)
#     df1['label'] = df1['malignant'].astype(int)
#     all_filepaths.extend(df1['filepath'].tolist())
#     all_labels.extend(df1['label'].tolist())
#     print(f"Загружен датасет mednode_dataset: {len(df1)} записей.")
# except Exception as e:
#     print(f"Ошибка загрузки датасета mednode_dataset: {e}")









##############################################################
# Разделение на обучающую и валидационную выборки

# @tf.function
# def mixup_same_class(batch_x, batch_y, alpha=0.2):
#     batch_size = tf.shape(batch_x)[0]
#     # выпрямляем метки
#     y_flat = tf.reshape(batch_y, [-1])  # [B]

#     # разбиваем батч на две части по классу
#     part_idx = tf.cast(y_flat, tf.int32)  # 0 или 1
#     xs = tf.dynamic_partition(batch_x, part_idx, 2)   # [xs[0]: все class=0], [xs[1]: все class=1]
#     ys = tf.dynamic_partition(batch_y, part_idx, 2)

#     # для каждой части делаем случайную перестановку
#     xs_shuffled = [tf.random.shuffle(xs[0]), tf.random.shuffle(xs[1])]

#     # индексы исходных позиций для восстановления
#     idxs = tf.dynamic_partition(tf.range(batch_size), part_idx, 2)

#     # собираем перемешанные батчи обратно в один тензор
#     x2 = tf.dynamic_stitch(idxs, xs_shuffled)  # [B, H, W, C]

#     # смешиваем с оригиналом
#     lam = tfp.distributions.Beta(alpha, alpha).sample([batch_size])
#     lam_x = tf.reshape(lam, [-1, 1, 1, 1])

#     lam_x = tf.cast(lam_x, batch_x.dtype)  # Приводим lam_x к типу batch_x
#     x2 = tf.cast(x2, batch_x.dtype)         # Приводим x2 к типу batch_x

#     mixed_x = lam_x * batch_x + (1.0 - lam_x) * x2
#     # метки не меняются, т.к. x2 всегда того же класса
#     return mixed_x, batch_y

# @tf.function
# def smart_cutmix(batch_x, batch_y, p=0.5):
#     """
#     Smart-CutMix: для каждого элемента батча с вероятностью p
#     отбираем другую картинку Того же класса, сегментируем
#     ROI по среднему порогу, вырезаем и вставляем в центр.
#     """
#     batch_size = tf.shape(batch_x)[0]
#     img_height = tf.shape(batch_x)[1]
#     img_width  = tf.shape(batch_x)[2]

#     # Инициализируем выходной тензор
#     out_x = batch_x

#     def apply_cutmix(i, out_x):
#         do_mix = tf.less(tf.random.uniform([], 0, 1), p)

#         def _mix():
#             # Индексы тех же класс-значений
#             same_cls = tf.reshape(tf.where(batch_y[:,0] == batch_y[i,0]), [-1])

#             # Исключаем сам i
#             i32 = tf.cast(i, same_cls.dtype)
#             same_cls = tf.boolean_mask(same_cls, same_cls != i32)

#             # Если больше нет кандидатов, возвращаем как есть
#             def _no_candidates():
#                 return out_x

#             def _with_candidate():
#                 # Берём случайного j из same_cls
#                 j = same_cls[tf.random.uniform([], 0, tf.shape(same_cls)[0], dtype=tf.int32)]

#                 img1 = batch_x[i]
#                 img2 = batch_x[j]

#                 # Грубая сегментация порогом по среднему
#                 gray = tf.image.rgb_to_grayscale(img1)
#                 mask = tf.cast(gray > tf.reduce_mean(gray), tf.float32)

#                 # Координаты ROI
#                 coords = tf.where(mask[...,0] > 0)
#                 ymin = tf.cast(tf.reduce_min(coords[:,0]), tf.int32)
#                 ymax = tf.cast(tf.reduce_max(coords[:,0]), tf.int32)
#                 xmin = tf.cast(tf.reduce_min(coords[:,1]), tf.int32)
#                 xmax = tf.cast(tf.reduce_max(coords[:,1]), tf.int32)

#                 # Вырезаем патч и маску
#                 patch  = img1[ymin:ymax, xmin:xmax, :]
#                 m_patch= mask[ymin:ymax, xmin:xmax, :]

#                 # Центр вставки
#                 insert_y = (img_height - (ymax-ymin)) // 2
#                 insert_x = (img_width  - (xmax-xmin)) // 2

#                 # Расширяем до размера full
#                 mask_full = tf.image.pad_to_bounding_box(
#                     m_patch,
#                     insert_y, insert_x,
#                     img_height, img_width
#                 )
#                 patch_full = tf.image.pad_to_bounding_box(
#                     patch,
#                     insert_y, insert_x,
#                     img_height, img_width
#                 )

#                 # Приводим типы для mixed_precision: все к float32
#                 mask_full  = tf.cast(mask_full,  tf.float32)
#                 patch_full = tf.cast(patch_full, tf.float32)
#                 img2_f     = tf.cast(img2,     tf.float32)

#                 # Шовное смешение
#                 img2_aug = patch_full * mask_full + img2_f * (1.0 - mask_full)

#                 # Обновляем элемент i
#                 return tf.tensor_scatter_nd_update(out_x,
#                                                    indices=[[i]], 
#                                                    updates=[img2_aug])

#             return tf.cond(tf.equal(tf.size(same_cls), 0),
#                            _no_candidates,
#                            _with_candidate)

#         return tf.cond(do_mix, _mix, lambda: out_x)

#     # Проходим по всем i
#     for idx in tf.range(batch_size):
#         out_x = apply_cutmix(idx, out_x)

#     return out_x, batch_y

# @tf.function
# def augment_and_mix(batch_x, batch_y):
#     x, y = mixup_same_class(batch_x, batch_y, alpha=0.2)
#     #x, y = smart_cutmix(x, y, p=0.5)
#     return x, y

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
    image = image / 255.0  # Нормализация, такая нормализацая работает лучше для DenseNet121
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
    loss= focal_loss(gamma=2.0, alpha=0.5),
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
# Обучение модели

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


print("\n--- Начало обучения только 'головы' ---")
# history = model.fit(
#     train_ds,
#     epochs=8,
#     validation_data=val_ds,
#     class_weight=class_weights_dict,
#     callbacks=[model_checkpoint_callback_1],
#     initial_epoch=0
# )

history = model.fit(
    train_ds,
    epochs=epochs_head,
    validation_data=val_ds,
    class_weight=class_weights_dict,
    callbacks=[model_checkpoint_callback_1, one_cycle_cb],
    initial_epoch=0
)
print("--- Обучение 'головы' завершено ---")

x = base_model.output
x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
x = layers.Dropout(0.4, name='head_dropout')(x)
output_tensor = layers.Dense(1, activation='sigmoid', name='class_predictions')(x)

model = keras.Model(inputs=input_tensor, outputs=output_tensor)

print("\n--- Начало этапа Fine-tuning ---")
base_model.trainable = True # Размораживаем всю базовую модель
print("\n--- Summary после разморозки для Fine-tuning ---")

unfreeze_from_layer_name = 'block1a_project_conv' #7 слой
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
    loss= focal_loss(gamma=2.0, alpha=0.5),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)


weights_dir = "/mnt/c/Users/urako/OneDrive/Документы/Code/Mr.-Mole/CNN/checkpoints" # Находим лучший чекпоинт первого этапа
try:
     print(f"Загрузка лучших весов 'головы' из: {weights_dir} для начала fine-tuning.")
     weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.weights.h5') and f.startswith(parametrs)]
     weight_files.sort(reverse=True)
     best_weight_path = os.path.join(weights_dir, weight_files[0])
     model.load_weights(best_weight_path)
except:
     print("Предупреждение: Не найдены веса для загрузки перед fine-tuning. Продолжаем с текущими.")

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
# history_fine = model.fit(
#     train_ds,
#     epochs=total_epochs,
#     validation_data=val_ds,
#     class_weight=class_weights_dict,
#     callbacks=[model_checkpoint_callback_2, early_stopper],
#     initial_epoch=initial_epoch_fine_tuning
# )

history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    validation_data=val_ds,
    class_weight=class_weights_dict,
    callbacks=[model_checkpoint_callback_2, early_stopper, ft_lr_cb],
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