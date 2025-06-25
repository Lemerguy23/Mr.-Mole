import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras import mixed_precision
import os
import numpy as np # Нужен для np.pi, хотя в этом скрипте WarmupCosineDecay не используется напрямую

# --- КОНФИГУРАЦИЯ ---
TF_CHECKPOINT_PREFIX = "/mnt/c/Users/urako/OneDrive/Документы/Code/Mr.-Mole/CNN/moco_checkpoints/original_mobilenet_encoder_bs64x1_q32768_t0.10/ckpt-62"

OUTPUT_KERAS_MODEL_PATH = "/mnt/c/Users/urako/OneDrive/Документы/Code/Mr.-Mole/CNN/ssl_mole_encoder_from_ckpt62.keras"

IMG_SIZE = (224, 224)
EMBED_DIM = 128
EPSILON_NORM = 1e-6

# 4. Установите политику смешанной точности, если она использовалась при обучении
#    (распространенные значения: 'mixed_float16', 'float32')
#    Если не использовали смешанную точность, можно установить 'float32' или закомментировать установку.
GLOBAL_POLICY = 'mixed_float16'
# --- КОНЕЦ КОНФИГУРАЦИИ ---


# --- Функции для построения модели (скопированы из вашего скрипта) ---
# Важно: эти функции должны точно воссоздавать архитектуру,
# сохраненную в чекпоинте.

def build_encoder_original():
    inp = layers.Input(shape=(*IMG_SIZE,3), name="input_mobilenet")
    # При загрузке весов из чекпоинта, инициализируем MobileNetV3Small с weights=None,
    # так как чекпоинт уже содержит обученные веса для базовой модели.
    base = MobileNetV3Small(weights=None, include_top=False, input_shape=(*IMG_SIZE, 3))
    x = base(inp)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool_in_encoder_q')(x)
    # Keras обычно обрабатывает dtype для Dense слоев на основе глобальной политики,
    # но при необходимости можно указать dtype=GLOBAL_POLICY явно.
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(EMBED_DIM)(x)
    # dtype в Lambda важен, если использовалась смешанная точность
    final_output = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1, epsilon=EPSILON_NORM),
        output_shape=(EMBED_DIM,), # Указываем форму выходного вектора
        name='l2_normalize_lambda', # Дадим имя, чтобы легче найти
        dtype=GLOBAL_POLICY
    )(x)
    print("DEBUG: Building ORIGINAL MobileNetV3Small encoder (weights will be loaded from checkpoint).")
    return models.Model(inp, final_output, name="original_mobilenet_encoder")
# --- Конец функций для построения модели ---

def main():
    print(f"TensorFlow Version: {tf.__version__}")

    if GLOBAL_POLICY and GLOBAL_POLICY != 'float32':
        print(f"Setting global mixed precision policy to: {GLOBAL_POLICY}")
        mixed_precision.set_global_policy(GLOBAL_POLICY)
    else:
        print("Using global precision policy: float32 (or policy not explicitly set to mixed).")

    # 1. Воссоздать архитектуру модели encoder_q
    print("\nStep 1: Re-creating encoder_q model architecture...")
    encoder_q = build_encoder_original()
    encoder_q.summary(line_length=100)

    # 2. Создать объект tf.train.Checkpoint для загрузки весов
    #    Ключ 'encoder_q' должен совпадать с ключом, использованным при сохранении оригинального чекпоинта.
    #    В вашем скрипте это было:
    #    ckpt_obj = tf.train.Checkpoint(..., encoder_q=encoder_q, ...)
    checkpoint_loader = tf.train.Checkpoint(encoder_q=encoder_q)
    print("\nStep 2: Checkpoint object for loading created.")

    # 3. Загрузить веса из указанного TensorFlow чекпоинта
    print(f"\nStep 3: Attempting to load weights for encoder_q from: {TF_CHECKPOINT_PREFIX}")

    # Проверка существования файлов чекпоинта
    checkpoint_index_file = TF_CHECKPOINT_PREFIX + ".index"
    # Проверяем наличие хотя бы одного data-файла
    checkpoint_data_file_exists = False
    if os.path.exists(os.path.dirname(TF_CHECKPOINT_PREFIX)):
        for f in os.listdir(os.path.dirname(TF_CHECKPOINT_PREFIX)):
            if f.startswith(os.path.basename(TF_CHECKPOINT_PREFIX) + ".data-00000-of-"):
                checkpoint_data_file_exists = True
                break
    
    if not (os.path.exists(checkpoint_index_file) and checkpoint_data_file_exists):
        print(f"ERROR: Checkpoint files not found at prefix: {TF_CHECKPOINT_PREFIX}")
        print(f"Expected .index file: {checkpoint_index_file}")
        print(f"Expected .data-*****_of-***** files in directory: {os.path.dirname(TF_CHECKPOINT_PREFIX)}")
        return

    try:
        # status.expect_partial() используется, потому что оригинальный чекпоинт, скорее всего,
        # содержал больше объектов (например, encoder_k, оптимизатор, очередь),
        # а мы загружаем только encoder_q.
        status = checkpoint_loader.restore(TF_CHECKPOINT_PREFIX).expect_partial()
        
        # Дополнительная проверка, что веса действительно были загружены в encoder_q.
        # Если это вызывает ошибку, возможно, структура модели сильно отличается или ключ 'encoder_q' неверен.
        try:
            status.assert_nontrivial_match()
            print("Checkpoint loaded successfully: Non-trivial match found for encoder_q weights.")
        except AssertionError as e:
            print(f"WARNING: Checkpoint loaded, but assert_nontrivial_match for encoder_q failed: {e}")
            print("This MIGHT indicate that no weights were loaded into encoder_q, or the model structure/names have changed significantly.")
            print("Proceeding with saving, but please verify the output .keras model carefully.")

    except Exception as e:
        print(f"ERROR: Failed to restore weights from checkpoint '{TF_CHECKPOINT_PREFIX}'. Error: {e}")
        return

    # 4. Сохранить модель encoder_q с загруженными весами в файл .keras
    print(f"\nStep 4: Saving encoder_q model to: {OUTPUT_KERAS_MODEL_PATH}")
    try:
        TARGET_OUTPUT_LAYER_NAME = 'global_avg_pool_in_encoder_q'

        # Получаем вход encoder_q и выход нужного слоя
        model_input = encoder_q.input
        feature_output = encoder_q.get_layer(TARGET_OUTPUT_LAYER_NAME).output

        # Создаем новую модель, которая является только извлекателем признаков
        feature_extractor_model = models.Model(
            inputs=model_input,
            outputs=feature_output,
            name='mobilenetv3small_feature_extractor'
        )
        
        feature_extractor_model.summary(line_length=100)
        print(f"Feature extractor model created. Output layer: '{TARGET_OUTPUT_LAYER_NAME}'")

        print(f"Saving feature extractor model to: {OUTPUT_KERAS_MODEL_PATH}")
        output_dir = os.path.dirname(OUTPUT_KERAS_MODEL_PATH)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        feature_extractor_model.save(OUTPUT_KERAS_MODEL_PATH) # Сохраняем новую модель
        print(f"Successfully saved feature extractor model to {OUTPUT_KERAS_MODEL_PATH}")
        print("This model contains MobileNetV3Small + GlobalAveragePooling2D.")

    except Exception as e:
        print(f"ERROR: Failed to create or save the feature extractor model. Error: {e}")
        print("Original full encoder_q summary (if built):")
        if 'encoder_q' in locals():
            encoder_q.summary(line_length=100)
        return

if __name__ == "__main__":
    main()