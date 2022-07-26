import os
import pandas as pd
import numpy as np
from dataloaders import InfiniteDataLoader
from datasets import SubclassedDataset
from models import TransferModel18
from train_eval import train, evaluate, train_epochs
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss import ERMLoss, GDROLoss 
from torchvision import transforms
import torch
from utils.image_data_utils import images_to_df, get_features, show_scatter

from umap import UMAP
from matplotlib import pyplot as plt
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from datetime import datetime
from scipy.stats import mode


def train_erm_cluster(model, device='cpu', loaders=None):
    '''
    Trains a default ERM model on LIDC

    model: This should be a TransferModel18
    
    '''

    #get dataloaders
    if loaders:
        tr_loader, cv_loader, tst_loader = loaders
    else:
        train_data, cv_data, test_data = get_features(images=True, device=device, subclass='malignancy')
        tr = SubclassedDataset(*train_data)
        cv = SubclassedDataset(*cv_data)
        tst = SubclassedDataset(*test_data)

        tr_loader = InfiniteDataLoader(tr, batch_size=512)
        cv_loader = InfiniteDataLoader(cv, len(cv))
        tst_loader = InfiniteDataLoader(tst, len(tst))


    #get loss, scheduler, optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.005)
    loss_fn = ERMLoss(model, torch.nn.CrossEntropyLoss())
    scheduler=ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2, verbose=False)

    #train model
    epochs = 20
    train_epochs(epochs, tr_loader, tst_loader, model, loss_fn, optimizer, 
                 scheduler=scheduler, verbose=False, num_subclasses=3)
    

def extract_features(model, images_df=None, device='cpu'):
    '''
    extract features of model
    '''

    if images_df is None:
        images_df = images_to_df()

    noduleID, data = images_df['noduleID'], torch.stack(list(images_df['image'])).to(device)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook


    model.model.avgpool.register_forward_hook(get_activation('avgpool'))
    model(data)

    return noduleID, activation['avgpool'].squeeze()

def features_to_df(noduleID, features):

    ids = np.asarray(noduleID).reshape(-1,1)
    feats = features.cpu().numpy()
    cols = np.concatenate((ids, feats), axis=1)

    # Get features of all images
    df_features_all = pd.DataFrame(cols).rename({0:'noduleID'}, axis=1)
    df_features_all.sort_values('noduleID', inplace=True)
    df_features_all.reset_index(drop=True, inplace=True)


    return df_features_all

def split_features(df_features_all, splits_path='./data/train_test_splits/LIDC_data_split.csv'):
    
    df_splits = pd.read_csv(splits_path, index_col=0)

    # Get features of only determinate nodules
    df_features = df_features_all[df_features_all['noduleID'].isin(df_splits['noduleID'])]
    # df_features.sort_values('noduleID', inplace=True)
    df_features.reset_index(drop=True, inplace=True)

    train_idx = df_splits['split'] == 0
    cv_test_idx = df_splits['split'] != 0

    df_features_train = df_features[train_idx]
    df_features_cv_test = df_features[cv_test_idx]


    train_features = df_features_train.drop(['noduleID'], axis=1).values
    cv_test_features = df_features_cv_test.drop(['noduleID'], axis=1).values

    train_malignancy = df_splits[train_idx]['malignancy'].values
    cv_test_malignancy = df_splits[cv_test_idx]['malignancy'].values

    train_id =df_splits[train_idx]['noduleID'].values
    cv_test_id =df_splits[cv_test_idx]['noduleID'].values


    return (train_features, train_malignancy, train_id), (cv_test_features, cv_test_malignancy, cv_test_id)

def check_cluster(embeds, max_clusters=15):
    clusters = [n for n in range(2,max_clusters+1)]
    silhouette_coefficients = []

    for cluster in clusters:
        gmm = GaussianMixture(n_components=cluster, random_state=61).fit(embeds)
        labels = gmm.predict(embeds)
    
        silhouette_avg = silhouette_score(embeds, labels)
        silhouette_coefficients.append(silhouette_avg)

    return silhouette_coefficients


def get_cluster_label(t_e, cvt_e, t_f, easy, hard):

    clusterer = GaussianMixture(n_components=2, random_state=61).fit(t_e)

    train_l, cv_test_l = clusterer.predict(t_e), clusterer.predict(cvt_e)


    size_0 = sum(train_l == 0)
    size_1 = sum(train_l == 1)

    if min(size_0, size_1) < 50:
        print('bad generated clusters, restarting process...')
        return None

    #find well defined group
    malig_counts_0 = sum(train_l[t_f == easy] == 0)/sum(train_l[t_f == hard] == 0)
    malig_counts_1 = sum(train_l[t_f == easy] == 1)/sum(train_l[t_f == hard] == 1)
    defined_group = 0 if malig_counts_0 > malig_counts_1 else 1

    #set malignant groups

    t_e_i = train_l == defined_group
    t_h_i = train_l == (1-defined_group)
    
    train_l[t_e_i] = easy
    train_l[t_h_i] = hard

    cvt_e_i = cv_test_l == defined_group
    cvt_h_i = cv_test_l == (1-defined_group)

    cv_test_l[cvt_e_i] = easy
    cv_test_l[cvt_h_i] = hard
    

    return train_l, cv_test_l

def do_clustering(tr_loader, cv_loader, tst_loader, images_df, device='cpu'):

        model=TransferModel18(pretrained=True, freeze=False, device=device)
        train_erm_cluster(model, device=device, loaders=(tr_loader, cv_loader, tst_loader))

        noduleID, features = extract_features(model, images_df=images_df, device=device)

        df_features_all = features_to_df(noduleID, features)

        #features and corresponding malignancy, noduleID
        train_f, cv_test_f = split_features(df_features_all)

        reducer = UMAP(random_state=8)
        reducer.fit(train_f[0])

        train_e, cv_test_e = reducer.transform(train_f[0]), reducer.transform(cv_test_f[0])
        
        malig_r = get_cluster_label(train_e[train_f[1] > 1], 
                          cv_test_e[cv_test_f[1]>1], 
                          train_f[1][train_f[1] > 1], 
                        #   cv_test_f[cv_test_f[1]>1], 
                          3,
                          2)

        if malig_r is None:
            return None
        
        benig_r = get_cluster_label(train_e[train_f[1] <= 1], 
                          cv_test_e[cv_test_f[1]<=1], 
                          train_f[1][train_f[1] <= 1], 
                        #   cv_test_f[1][cv_test_f[1]<=1], 
                          0,
                          1)
        

        if benig_r is None:
            return None

        train_l = np.zeros_like(train_f[1])
        cv_test_l = np.zeros_like(cv_test_f[1])

        train_l[train_f[1] > 1] = malig_r[0]
        train_l[train_f[1] <= 1] = benig_r[0]

        cv_test_l[cv_test_f[1] > 1] = malig_r[1]
        cv_test_l[cv_test_f[1] <= 1] = benig_r[1]

        labels = np.concatenate((train_l, cv_test_l), axis=0)
        noduleIDs = np.concatenate((train_f[2], cv_test_f[2]), axis=0)
        
        label_df = pd.DataFrame({'noduleID':noduleIDs, 'clusters':labels})

        return label_df, (train_e, train_f[1], train_l)
