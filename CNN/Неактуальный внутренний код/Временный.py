import os
import pandas as pd
import shutil
from tqdm import tqdm

def process_and_save_dataset(df, dataset_name, base_output_dir, multiplier=30):
    """
    Отбирает все раковые и в 'multiplier' раз больше нераковых изображений,
    копирует их в новую папку и сохраняет отфильтрованный CSV-файл.

    Args:
        df (pd.DataFrame): DataFrame с колонками 'filepath' и 'label'.
        dataset_name (str): Уникальное имя для датасета (используется для папки).
        base_output_dir (str): Корневая папка для всех обработанных данных.
        multiplier (int): Множитель для нераковых изображений.
    """
    print(f"\n{'='*20} Обработка датасета: {dataset_name} {'='*20}")

    # 1. Создаем выходную директорию для этого конкретного датасета
    output_dir = os.path.join(base_output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Данные будут сохранены в: {output_dir}")

    # 2. Разделяем данные на раковые (label=1) и нераковые (label=0)
    cancer_df = df[df['label'] == 1].copy()
    non_cancer_df = df[df['label'] == 0].copy()

    num_cancer = len(cancer_df)
    num_non_cancer = len(non_cancer_df)

    if num_cancer == 0:
        print("Внимание: В датасете не найдено раковых изображений (label=1). Обработка пропущена.")
        return

    print(f"Найдено раковых изображений: {num_cancer}")
    print(f"Найдено нераковых изображений: {num_non_cancer}")

    # 3. Определяем, сколько нераковых изображений нужно взять
    num_non_cancer_to_select = num_cancer * multiplier

    # Проверяем, достаточно ли у нас нераковых изображений для выборки
    if num_non_cancer_to_select > num_non_cancer:
        print(f"Внимание: Недостаточно нераковых изображений.")
        print(f"Требуется {num_non_cancer_to_select}, но доступно только {num_non_cancer}.")
        print("Будут использованы все доступные нераковые изображения.")
        selected_non_cancer_df = non_cancer_df
    else:
        # Случайным образом отбираем нужное количество нераковых изображений
        print(f"Случайным образом отбирается {num_non_cancer_to_select} нераковых изображений.")
        selected_non_cancer_df = non_cancer_df.sample(n=num_non_cancer_to_select, random_state=42)

    # 4. Объединяем все раковые и отобранные нераковые
    final_df = pd.concat([cancer_df, selected_non_cancer_df]).reset_index(drop=True)
    
    # Можно дополнительно перемешать итоговый DataFrame для случайного порядка в CSV
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Итоговый размер выборки: {len(final_df)} изображений.")
    
    # 5. Сохраняем новый CSV-файл с отфильтрованными данными
    output_csv_path = os.path.join(output_dir, f"{dataset_name}_selection.csv")
    final_df.to_csv(output_csv_path, index=False)
    print(f"Отфильтрованный CSV файл сохранен: {output_csv_path}")

    # 6. Копируем файлы изображений в новую папку
    print("Начинаю копирование файлов изображений...")
    copied_count = 0
    for _, row in tqdm(final_df.iterrows(), total=len(final_df), desc=f"Копирование ({dataset_name})"):
        source_path = row['filepath']
        
        if not os.path.exists(source_path):
            print(f"\nВнимание: Файл не найден и будет пропущен: {source_path}")
            continue
            
        filename = os.path.basename(source_path)
        destination_path = os.path.join(output_dir, filename)
        
        # Используем copy2 для сохранения метаданных файла
        shutil.copy2(source_path, destination_path)
        copied_count += 1

    print(f"Копирование завершено. Скопировано {copied_count} из {len(final_df)} файлов.")


# --- Главная часть скрипта ---

# Общая папка для всех результатов
OUTPUT_BASE_DIR = "./processed_data"
CANCER_MULTIPLIER = 30

# --- 1. Обработка тестового датасета ---
try:
    print("\n--- Загрузка тестовых данных (challenge-2020-test) ---")
    test_image_dir = "/mnt/a/Datasets/ISIC-images/all"
    test_csv_path = "/mnt/a/Datasets/ISIC-images/challenge-2020-test_metadata_2025-04-20.csv"
    
    test_df = pd.read_csv(test_csv_path)
    test_df['filepath'] = test_df['isic_id'].apply(lambda x: os.path.join(test_image_dir, x + ".jpg"))
    test_df['label'] = test_df['benign_malignant'].map({'benign': 0, 'malignant': 1})
    
    # Удаляем строки, где метка не была определена (если такие есть)
    test_df.dropna(subset=['label'], inplace=True)
    test_df['label'] = test_df['label'].astype(int)
    
    process_and_save_dataset(test_df, "test_data", OUTPUT_BASE_DIR, multiplier=CANCER_MULTIPLIER)

except FileNotFoundError:
    print(f"Ошибка: Не найден CSV-файл или папка с изображениями для тестового датасета.")
except Exception as e:
    print(f"Произошла непредвиденная ошибка при обработке тестового датасета: {e}")


# --- 2. Обработка ISIC 2024 ---
try:
    print("\n--- Загрузка данных ISIC 2024 ---")
    ISIC_2024_meta_path = "/mnt/a/Datasets/ISIC_2024_Training_Input/ISIC_2024_Training_GroundTruth.csv"
    ISIC_2024_image_dir = "/mnt/a/Datasets/ISIC_2024_Training_Input/ISIC_2024_Training_Input"

    df_2024 = pd.read_csv(ISIC_2024_meta_path)
    df_2024['filepath'] = df_2024['isic_id'].apply(lambda x: os.path.join(ISIC_2024_image_dir, x + '.jpg'))
    df_2024['label'] = df_2024['malignant'].astype(int)
    
    process_and_save_dataset(df_2024, "isic_2024_data", OUTPUT_BASE_DIR, multiplier=CANCER_MULTIPLIER)
    
except FileNotFoundError:
    print(f"Ошибка: Не найден CSV-файл или папка с изображениями для ISIC 2024.")
except Exception as e:
    print(f"Произошла непредвиденная ошибка при обработке ISIC 2024: {e}")

# --- 3. Обработка ISIC 2020 ---
try:
    print("\n--- Загрузка данных ISIC 2020 ---")
    ISIC_2020_meta_path = "/mnt/a/Datasets/ISIC_2020_Training_JPEG/ISIC_2020_Training_GroundTruth_v2.csv"
    ISIC_2020_image_dir = "/mnt/a/Datasets/ISIC_2020_Training_JPEG/train"

    df_2020 = pd.read_csv(ISIC_2020_meta_path)
    df_2020['filepath'] = df_2020['image_name'].apply(lambda x: os.path.join(ISIC_2020_image_dir, x + '.jpg'))
    df_2020['label'] = df_2020['target'].astype(int)
    
    process_and_save_dataset(df_2020, "isic_2020_data", OUTPUT_BASE_DIR, multiplier=CANCER_MULTIPLIER)

except FileNotFoundError:
    print(f"Ошибка: Не найден CSV-файл или папка с изображениями для ISIC 2020.")
except Exception as e:
    print(f"Произошла непредвиденная ошибка при обработке ISIC 2020: {e}")

print(f"\n{'='*25} ВСЯ РАБОТА ЗАВЕРШЕНА {'='*25}")
print(f"Проверьте результаты в папке: {os.path.abspath(OUTPUT_BASE_DIR)}")