import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer
from sklearn.utils import class_weight

train_image_dir = "A:/Datasets/ISIC_2020_Training_JPEG/train"

ISIC_training_data = pd.read_csv("A:/Datasets/ISIC_2020_Training_JPEG/ISIC_2020_Training_GroundTruth_v2.csv")

ISIC_training_data['target'] = ISIC_training_data['target'].astype(str)       #flow_from_dataframe требует строку
ISIC_training_data['image_name'] = ISIC_training_data['image_name'] + '.jpg'

train_df = ISIC_training_data.sample(frac=0.8, random_state=42) #80%
val_df = ISIC_training_data.drop(train_df.index).reset_index(drop=True) #20%
train_df = train_df.reset_index(drop=True)


train_datagen = ImageDataGenerator(
    rescale=1./255, # Нормализация: масштабирование пикселей к [0, 1]
    rotation_range=20, # Повороты до 20 градусов
    width_shift_range=0.1, # Cдвиги по ширине 10%
    height_shift_range=0.1, # Cдвиги по высоте 10%
    horizontal_flip=True, #Горизонтальные отражения
    fill_mode='nearest' # Заполнение пустых пикселей после преобразований ближайшими
)

val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_image_dir,
    x_col="image_name",
    y_col="target",
    target_size=(384, 384),
    batch_size=32,
    class_mode='binary', # Тип задачи классификации ('binary' для 2 классов, 'categorical' для >2 классов)
    seed=42 # Для воспроизводимости
)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=train_image_dir,
    x_col="image_name",
    y_col="target",
    target_size=(384, 384),
    batch_size=32,
    class_mode='binary',
    seed=42
)

y_train = train_df['target'].astype(int).values

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Преобразуем веса классов в словарь, который принимает Keras
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}





model = Sequential([
    InputLayer(input_shape=(384, 384, 3)),

    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(units=128, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])



history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    class_weight=class_weights_dict
)