import pandas as pd
import cv2
import numpy as np
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ISIC_training_data = pd.read_csv("A:/Datasets/ISIC_2020_Training_JPEG/ISIC_2020_Training_GroundTruth_v2.csv")
# HAM10000_training_data = pd.read_csv("A:/Datasets/dataverse_files/HAM10000_metadata.csv")
# zr7vgbcyr2_training_data = pd.read_csv("A:/Datasets/zr7vgbcyr2-1/metadata.csv")
# #print(ISIC_training_data.head())

# uniq = ISIC_training_data['target'].unique()
# value_unique = ISIC_training_data['target'].value_counts()

# print("ISIC_training_data")
# print(uniq)
# print(value_unique)
# print()
# print(ISIC_training_data.head())


# uniq = HAM10000_training_data['dx'].unique()
# value_unique = HAM10000_training_data['dx'].value_counts()

# print("HAM10000_training_data")
# print(uniq)
# print(value_unique)
# print()
# print(HAM10000_training_data.head())


# uniq = zr7vgbcyr2_training_data['diagnostic'].unique()
# value_unique = zr7vgbcyr2_training_data['diagnostic'].value_counts()

# print("zr7vgbcyr2_training_data")
# print(uniq)
# print(value_unique)
# print()
# print(zr7vgbcyr2_training_data.head())





