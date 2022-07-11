import os
import shutil
import pandas as pd
import numpy as np

'''
moves images into folder seperated by malignancy level
'''

df_LIDC = pd.read_csv('../data/LIDC_labels_cleaned.csv', index_col=0)

image_folder = '../LIDC(MaxSlices)_Nodules'
subtyped_image_folder = '../LIDC(MaxSlices)_Nodules_Subgrouped'


for file in os.scandir(image_folder):

    #find the nodule malignancy using unique ID
    temp_nodule_ID = int(file.name.split('.')[0])
    malignancy = int(df_LIDC[df_LIDC['noduleID']==temp_nodule_ID]['malignancy'].iloc[0])
    
    subdir = f'Malignancy_{malignancy}'
    shutil.copy(file, os.path.join(subtyped_image_folder, subdir, os.path.basename(os.path.normpath(file))))
