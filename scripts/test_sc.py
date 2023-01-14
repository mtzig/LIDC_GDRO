import sys
sys.path.append('./')

from dataloaders import InfiniteDataLoader
from datasets import SubclassedDataset
from models import TransferModel18
from utils.image_data_utils import images_to_df, get_features
from utils.cluster_utils import train_erm_cluster, extract_features, features_to_df, split_features, check_cluster
from umap import UMAP
import numpy as np
import pandas as pd
import torch


def test_sc(tr_loader, cv_loader, tst_loader, images_df, split_path='./data/train_test_splits/LIDC_data_split.csv', split_num=None, get_cv_embeds=False, device='cpu'):

    model = TransferModel18(pretrained=True, freeze=False, device=device)
    train_erm_cluster(model, device=device, loaders=(
        tr_loader, cv_loader, tst_loader))

    noduleID, features = extract_features(
        model, images_df=images_df, device=device)

    df_features_all = features_to_df(noduleID, features)

    # features and corresponding malignancy, noduleID
    train_f, cv_test_f = split_features(
        df_features_all, split_path=split_path, split_num=split_num)

    reducer = UMAP(random_state=8)
    reducer.fit(train_f[0])

    train_e, cv_test_e = reducer.transform(
        train_f[0]), reducer.transform(cv_test_f[0])

    malig_max = np.argmin(check_cluster(train_e[train_f[1] > 1]))+1
    print(check_cluster(train_e[train_f[1] > 1]))
    benig_max = np.argmin(check_cluster(train_e[train_f[1] <= 1]))+1

    return malig_max, benig_max

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
split_file = './data/train_test_splits/LIDC_data_split.csv' #'./data/train_test_splits/Nodule_Level_30Splits/nodule_split_all.csv'
split_df = pd.read_csv(split_file, index_col=0)
# split_num = 0

print('Loading images')
images_df = images_to_df()

df_clusters = pd.DataFrame()

train_data, cv_data, test_data = get_features(
    images=True, features=images_df, device=DEVICE, subclass='malignancy')

# datasets
tr = SubclassedDataset(*train_data)
cv = SubclassedDataset(*cv_data)
tst = SubclassedDataset(*test_data)

# dataloaders
tr_loader = InfiniteDataLoader(tr, batch_size=512)
cv_loader = InfiniteDataLoader(cv, len(cv))
tst_loader = InfiniteDataLoader(tst, len(tst))

for trial in range(30):
    print(f'==============  Trial {trial} ==============')

    m, b = test_sc(tr_loader, cv_loader, tst_loader, images_df, device=DEVICE)
    print(f'Malginant max sc: {m} Benign max sc:{b}')
