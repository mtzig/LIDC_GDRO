import pandas as pd
import numpy as np

lidc = pd.read_csv('../data/LIDC_labels_cleaned.csv', index_col=0)
lidc = lidc[lidc['malignancy']!=3]
lidc['malignancy'] = np.where(lidc['malignancy'] > 3, lidc['malignancy'] - 2, lidc['malignancy'] - 1)
lidc = lidc.sample(frac=1, random_state=59)

noduleID = []
maligs = []
split = []

tr_split, cv_split, test_split = 0.7, 0.1, 0.2

for malig in range(4):
    lidc_temp = lidc[lidc['malignancy']==malig]
    length = len(lidc_temp)

    tr_idx = int(tr_split * length) 
    cv_idx = tr_idx + int(cv_split * length)
    test_idx = length

    indices = [0,tr_idx,cv_idx, test_idx]
    for i,l in enumerate(indices[:-1]):
        noduleID.extend(lidc_temp['noduleID'].iloc[l:indices[i+1]])
        split.extend((i,)*(indices[i+1]-l))

    maligs.extend((malig,)*length)

df_split = pd.DataFrame(zip(noduleID, maligs, split), columns=['noduleID', 'malignancy', 'split'])
df_split.to_csv('../data/LIDC_data_split.csv')
    

