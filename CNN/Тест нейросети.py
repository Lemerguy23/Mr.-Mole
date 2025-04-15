import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

model = Sequential([
    InputLayer(input_shape=(144, 144, 3)),

    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(units=256, activation='relu'),
    Dense(units=128, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

lastest = "C:/Users/urako/OneDrive/Документы/Код/Mr.-Mole/checkpoints/model_26_0.67.h5"
model.load_weights(lastest)

test_image_dir = "A:\\Datasets\\archive\\imgs_part_1"
test_csv_path = "A:\\Datasets\\archive\\metadata.csv"

cancer = {
    "BCC": '1',  
    "ACK": '0', 
    "NEV": '0', 
    "SEK": '0', 
    "SCC": '1',  
    "MEL": '1'   
}

test_df = pd.read_csv(test_csv_path)
test_df['diagnostic'] = test_df['diagnostic'].map(cancer)



test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_image_dir,
    x_col="img_id",
    y_col="diagnostic",
    target_size=(144, 144),
    batch_size=32, # Можно выбрать другой размер батча для тестирования
    class_mode='binary',
    shuffle=False, # ВАЖНО: НЕ перемешивайте тестовые данные для evaluate/predict
    seed=42
)


print("Оценка модели на тестовых данных...")
results = model.evaluate(test_generator)

print(f"Тестовая Loss: {results[0]:.4f}")
print(f"Тестовая Accuracy: {results[1]:.4f}")


# Получаем вероятности для класса 1 (malignant)
predictions_prob = model.predict(test_generator)

# Преобразуем вероятности в классы (0 или 1) с порогом 0.5
predictions_class = (predictions_prob > 0.5).astype(int).flatten() # flatten() преобразует (N, 1) в (N,)

# Получаем истинные метки из генератора
true_labels = test_generator.classes

# Вычисляем и выводим детальные метрики
print("\nМатрица ошибок:")
print(confusion_matrix(true_labels, predictions_class))

print("\nОтчет по классификации:")
# target_names можно получить из test_generator.class_indices
# target_names = list(test_generator.class_indices.keys()) # Получит ['0', '1']
target_names = ['Benign (0)', 'Malignant (1)'] # Более понятные имена
print(classification_report(true_labels, predictions_class, target_names=target_names))

# Вычисляем AUC (Area Under the ROC Curve)
try:
    auc_score = roc_auc_score(true_labels, predictions_prob)
    print(f"\nAUC: {auc_score:.4f}")
except ValueError as e:
    print(f"\nНе удалось вычислить AUC: {e}")
    # Это может произойти, если в true_labels есть только один класс