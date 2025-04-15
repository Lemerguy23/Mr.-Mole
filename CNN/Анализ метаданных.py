import pandas as pd
import numpy as np
import tensorflow as tf

ISIC_2024_meta_path = pd.read_csv("A:/Datasets/ISIC_2024_Training_Input/ISIC_2024_Training_GroundTruth.csv")
ISIC_2024_image_dir = "A:/Datasets/ISIC_2024_Training_Input/ISIC_2024_Training_Input"

ISIC_2020_meta_path = pd.read_csv("A:/Datasets/ISIC_2020_Training_JPEG/ISIC_2020_Training_GroundTruth_v2.csv")
ISIC_2020_image_dir = "A:/Datasets/ISIC_2020_Training_JPEG/train"

HAM10000_meta_path  = pd.read_csv("A:/Datasets/dataverse_files/HAM10000_metadata.csv")
HAM10000_image_dir  = "A:/Datasets/dataverse_files/HAM10000_images_part_1"

archive_meta_path  = pd.read_csv("A:/Datasets/archive/metadata.csv")
archive_image_dir  = "A:/Datasets/archive/imgs_part_1"

mednode_dataset_meta_path  = pd.read_csv("A:/Datasets/complete_mednode_dataset/complete_mednode_dataset/metadata.csv")
mednode_dataset_image_dir  = "A:/Datasets/complete_mednode_dataset/complete_mednode_dataset/foto"


#iddx_full = ISIC_training_data['iddx_full'].value_counts()
# iddx_1 =       ISIC_training_data['iddx_1'].value_counts()
# iddx_2 =       ISIC_training_data['iddx_2'].value_counts()
#iddx_3 =       ISIC_training_data['iddx_3'].value_counts()
# iddx_4 =       ISIC_training_data['iddx_4'].value_counts()
# iddx_5 =       ISIC_training_data['iddx_5'].value_counts()

#print("ISIC_training_data")
# print(iddx_full)
# print()
# print(iddx_1)
# print()
# print(iddx_2)
# print()
#print(iddx_3)
#print()
# print(iddx_4)
# print()
# print(iddx_5)
# print()


iddx_full = ISIC_2024_meta_path['malignant'].value_counts()
print("ISIC_2024_meta_path")
print(iddx_full)
print()

iddx_full = ISIC_2020_meta_path['target'].value_counts()
print("ISIC_2020_meta_path")
print(iddx_full)
print()

iddx_full = HAM10000_meta_path['dx'].value_counts()
print("HAM10000_meta_path")
print(iddx_full)
print()

iddx_full = archive_meta_path['diagnostic'].value_counts()
print("archive_meta_path")
print(iddx_full)
print()

iddx_full = mednode_dataset_meta_path['malignant'].value_counts()
print("mednode_dataset_meta_path")
print(iddx_full)
print()
