import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflowjs as tfjs

IMG_WIDTH = (260, 260)
IMG_CHANNELS = 3

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


lastest = "/mnt/c/Users/urako/OneDrive/Документы/Code/Mr.-Mole/CNN/checkpoints/model_EfficientNetV2M_not_full150_Dp_2,4_VeryBigD_Flips_notManyDT_albumentations_Coef15,2_41_0.02442.weights.h5"
model.load_weights(lastest)

full_model_path = "/mnt/c/Users/urako/OneDrive/Документы/Code/Mr.-Mole/CNN/model_EfficientNetV2M_not_full150_Dp_2,4_VeryBigD_Flips_notManyDT_albumentations_Coef15,2_41_0.02442.keras"
# model.save(full_model_path)
print(f"Модель сохранена в {full_model_path}")

# Конвертация в TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
