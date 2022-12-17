'''
NOTE: this script should be run from the root directory i.e. python scripts/cluster.py
'''

import sys
sys.path.append('./')

import pandas as pd
import torch
import numpy as np
from dataloaders import InfiniteDataLoader
from datasets import SubclassedDataset
from utils.image_data_utils import images_to_df, get_features
from utils.cluster_utils import do_clustering
from tqdm import tqdm

## NOTE: suppresses all warnings about deprecating functions
import warnings
warnings.filterwarnings('ignore')


def cluster_one_split(split_num, images_df, split_file, split_df, device):
    '''
    gets clustered subclass label for one random split
        - i.e. derive clusters from 50 different trained models
        - then take their mode times an
    '''

    train_data, cv_data, test_data = get_features(images=True, features=images_df, split_file=split_file, device=device, subclass='malignancy', split_num=split_num)

    id = np.concatenate((split_df[split_df[f'split_{split_num}'] == 0]['noduleID'].values,  
                         split_df[split_df[f'split_{split_num}'] != 0]['noduleID'].values))

    #datasets
    tr = SubclassedDataset(*train_data)
    cv = SubclassedDataset(*cv_data)
    tst = SubclassedDataset(*test_data)

    #dataloaders
    tr_loader = InfiniteDataLoader(tr, batch_size=512)
    cv_loader = InfiniteDataLoader(cv, len(cv))
    tst_loader = InfiniteDataLoader(tst, len(tst))

    labels = []

    print('training 50 models and clustering')
    for _ in tqdm(range(50)):
        results = None
        while results is None:
            results = do_clustering(tr_loader, cv_loader, tst_loader, images_df, split_path=split_file, split_num=split_num, device=device)
  
        label_df = results[0]
        labels.append(label_df['clusters'].values)

    labels_df = pd.DataFrame(zip(*labels))
    labels_df['noduleID'] = id

    labels_df.sort_values('noduleID', inplace=True)
    labels_df.reset_index(drop=True, inplace=True)

    labels_df.to_csv(f'./cluster_results/split_{split_num}_all_clusters.csv')

    mode_val=labels_df.drop(['noduleID'], axis=1).mode(axis=1).min(axis=1).values.astype(int)

    return mode_val

if __name__ == "__main__":

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    split_file = './data/train_test_splits/Nodule_Level_30Splits/nodule_split_all.csv'
    split_df = pd.read_csv(split_file, index_col=0)

    print('Loading images')
    images_df = images_to_df()

    df_clusters = pd.DataFrame()

    for split_num in range(30):
        print(f'================= random split {split_num} =================')

        mode_vals = cluster_one_split(split_num, images_df, split_file, split_df, DEVICE)
        df_clusters[f'cluster_{split_num}'] = mode_vals

        print(f'Saving results for random split {split_num}')
        df_clusters.to_csv(f'./cluster_results/all_clusters.csv')

    


