import pandas as pd
import numpy as np

lidc = pd.read_csv('./data/LIDC_semantic_spiculation_malignancy.csv', index_col=0)
lidc = lidc[lidc['malignancy']!=3]

malignancy = np.where(lidc['malignancy'] > 3, lidc['malignancy'] - 2, lidc['malignancy'] - 1).astype(int)
lidc['malignancy']= malignancy
spic_b = np.where(lidc['spiculation'] > 1, 1, 0)

lidc = lidc.sample(frac=1, random_state=59)

noduleID = []
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


df_split = pd.DataFrame(zip(noduleID, split), columns=['noduleID', 'split'])

# df_split = pd.DataFrame(zip(noduleID, maligs, maligs, split), columns=['noduleID', 'subgroup', 'malignancy', 'split'])
df_split.sort_values('noduleID', inplace=True)
df_split.reset_index(drop=True, inplace=True)

df_split['malignancy'] = malignancy

maligs_b = list(map( lambda x:int(x>1), malignancy))
df_split['malignancy_b'] = maligs_b

spics = [m*2+s for m,s in zip(maligs_b, spic_b)]
df_split['spic_groups'] = spics

df_split.to_csv('./data/train_test_splits/LIDC_data_split.csv')
    

