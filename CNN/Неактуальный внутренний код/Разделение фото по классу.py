import os
import pandas as pd
import shutil
import random

# === Пути ===
ISIC_2024_meta_path = "/mnt/a/Datasets/ISIC_2024_Training_Input/ISIC_2024_Training_GroundTruth.csv"
ISIC_2024_image_dir = "/mnt/a/Datasets/ISIC_2024_Training_Input/ISIC_2024_Training_Input"
output_dir = "/mnt/a/Datasets/Divided"  # Укажи свою папку

# === Загрузка CSV и добавление путей ===
df = pd.read_csv(ISIC_2024_meta_path)
df['filepath'] = df['isic_id'].apply(lambda x: os.path.join(ISIC_2024_image_dir, f"{x}.jpg"))
df['label'] = df['malignant'].astype(int)

# === Фильтрация по классам ===
df_class_1 = df[df['label'] == 1]
df_class_0 = df[df['label'] == 0].sample(n=len(df_class_1), random_state=42)

print(f"Класс 1: {len(df_class_1)} изображений.")
print(f"Класс 0 (выборка): {len(df_class_0)} изображений.")

# === Подготовка выходных папок ===
class_1_dir = os.path.join(output_dir, "malignant")
class_0_dir = os.path.join(output_dir, "benign")
os.makedirs(class_1_dir, exist_ok=True)
os.makedirs(class_0_dir, exist_ok=True)

# === Копирование файлов ===
def copy_files(df_subset, target_dir):
    for _, row in df_subset.iterrows():
        src = row['filepath']
        dst = os.path.join(target_dir, os.path.basename(src))
        try:
            shutil.copy2(src, dst)
        except FileNotFoundError:
            print(f"Файл не найден: {src}")

copy_files(df_class_1, class_1_dir)
copy_files(df_class_0, class_0_dir)

print("Копирование завершено.")
