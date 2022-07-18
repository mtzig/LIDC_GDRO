'''
takes raw data from LIDC_lables and returns the subset with max sclice images

'''



import pandas as pd
import numpy as np
import os


df_LIDC = pd.read_csv('../data/LIDC_labels.csv')
df_labels = df_LIDC[['noduleID', 'spiculation', 'malignancy']]


image_folder = '../data/LIDC(MaxSlices)_Nodules'
noduleIDs = []

for file in os.scandir(image_folder):
    noduleID = int(file.name.split('.')[0])
    noduleIDs.append(noduleID)

df_labels = df_labels[df_labels['noduleID'].isin(noduleIDs)]
df_labels.to_csv('../data/LIDC_labels_cleaned.csv')