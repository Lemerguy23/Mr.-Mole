import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

MODEL_PATH = r"CNN\checkpoints\model_DenseNet121_not_full150_Dp_6_AdamW_VeryBigD_Flips_We6_12_0.63.h5"
IMG_SIZE = (224, 224)

CLASS_NAMES = {0: 'benign', 1: 'malignant'}

def load_and_preprocess_image(image_path):
    """Загрузка и предобработка изображения"""
    image = tf.io.read_file(image_path)
    try:
        image = tf.io.decode_jpeg(image, channels=3)
    except:
        image = tf.io.decode_png(image, channels=3)
    
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image.numpy()

def predict_single_image(image_path, model, threshold=0.5):
    """Предсказание для одного изображения"""
    image = load_and_preprocess_image(image_path)
    
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image, verbose=0)[0][0]  # [0][0] чтобы получить скаляр
    
    class_idx = 1 if prediction > threshold else 0
    class_name = CLASS_NAMES[class_idx]
    confidence = prediction if class_idx == 1 else 1 - prediction
    
    return {
        'class': class_name,
        'class_index': class_idx,
        'confidence': float(confidence),
        'raw_prediction': float(prediction)
    }

try:
    model = load_model(MODEL_PATH)
    print("Модель успешно загружена")
except:
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras import layers
    
    input_tensor = tf.keras.Input(shape=(*IMG_SIZE, 3))
    base_model = DenseNet121(weights=None, include_top=False, input_tensor=input_tensor)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.6)(x)
    output_tensor = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    
    model.load_weights(MODEL_PATH)
    print("Архитектура создана и веса загружены")


if __name__ == "__main__":
    pass
    """
    test_image_path = r"site\rodinka4.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"Ошибка: файл {test_image_path} не найден")
    else:
        result = predict_single_image(test_image_path, model)
        print("\nРезультат предсказания:")
        print(f"Класс: {result['class']}")
        print(f"Уверенность: {result['confidence']:.2%}")
        print(f"Сырое значение предсказания: {result['raw_prediction']:.4f}")"""