import sys
sys.path.append('./')

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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

now = datetime.now()
results_dir = f'cluster_results/cluster_{now.strftime("%Y%m%d_%H%M%S")}'
os.mkdir(f'./data/{results_dir}')
images_df = images_to_df()
train_data, cv_data, test_data = get_features(images=True, features=images_df, device=DEVICE, subclass='malignancy')

#file with splits
df_splits = pd.read_csv('./data/train_test_splits/LIDC_data_split.csv', index_col=0)


#datasets
tr = SubclassedDataset(*train_data)
cv = SubclassedDataset(*cv_data)
tst = SubclassedDataset(*test_data)

#dataloaders
tr_loader = InfiniteDataLoader(tr, batch_size=512)
cv_loader = InfiniteDataLoader(cv, len(cv))
tst_loader = InfiniteDataLoader(tst, len(tst))


def train_model(model):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.005)
    epochs = 20
    loss_fn = ERMLoss(model, torch.nn.CrossEntropyLoss())
    train_epochs(epochs, tr_loader, tst_loader, model, loss_fn, optimizer, 
                 scheduler=ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2, verbose=False), verbose=False, num_subclasses=4)


def collect_features(model):

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.model.avgpool.register_forward_hook(get_activation('avgpool'))
    noduleID, data = images_df['noduleID'], torch.stack(list(images_df['image'])).to(DEVICE)
    model(data)

    return noduleID, activation['avgpool'].squeeze()

def check_cluster(embeds):
    clusters = [n for n in range(2,16)]
    silhouette_coefficients = []

    for cluster in clusters:
        gmm = GaussianMixture(n_components=cluster, random_state=61).fit(embeds)
        labels = gmm.predict(embeds)
    
        silhouette_avg = silhouette_score(embeds, labels)
        silhouette_coefficients.append(silhouette_avg)

    return silhouette_coefficients

def run_clustering():

    cluster_stats = pd.DataFrame()

    print('Training Model')
    model = TransferModel18(device=DEVICE, pretrained=True, freeze=False)
    train_model(model)

    #base model accuracies
    accuracies = evaluate(tst_loader,model, 4, verbose=False)
    cluster_stats['accuracies'] = accuracies


    #get features
    noduleID, img_features = collect_features(model)

    print('Collecting features')
    #collect features
    cols = []
    for idx,id in enumerate(noduleID):
        cols.append([id] + img_features[idx].cpu().numpy().tolist())
    df_features_all = pd.DataFrame(cols).rename({0:'noduleID'}, axis=1)
    df_features_all.sort_values('noduleID', inplace=True)
    df_features_all.reset_index(drop=True, inplace=True)



    df_features = df_features_all[df_features_all['noduleID'].isin(df_splits['noduleID'])]
    df_features.sort_values('noduleID', inplace=True)
    df_features.reset_index(drop=True, inplace=True)

    df_features['split'] = df_splits['split']
    df_features['malignancy'] = df_splits['malignancy']

    df_features_train = df_features[df_features['split'] == 0]
    df_features_cv_test = df_features[df_features['split'] != 0]


    train_features = df_features_train.drop(['noduleID', 'split', 'malignancy'], axis=1).values
    cv_test_features = df_features_cv_test.drop(['noduleID', 'split', 'malignancy'], axis=1).values

    print('Reducing Features')
    #reduce features
    reducer = UMAP(random_state=8)
    reducer.fit(train_features)

    train_embeds = reducer.transform(train_features)
    cv_test_embeds = reducer.transform(cv_test_features)

    train_embeds_malig = reducer.transform(train_features[df_features_train['malignancy'] > 1])
    train_embeds_benign = reducer.transform(train_features[df_features_train['malignancy'] < 2])

    print('calculating silhouettes')
    #calculate silhouette_coefficients
    silhouette_coefficients = check_cluster(train_embeds_malig)
    if silhouette_coefficients[0] != max(silhouette_coefficients):
        return False
    cluster_stats['silhouettes'] = silhouette_coefficients[:5]

    print('clustering')
    #final clusterer
    clusterer = GaussianMixture(n_components=2, random_state=61).fit(train_embeds_malig)

    #prevent instances, when GMM wants to make one cluster really small
    malig_lables = clusterer.predict(train_embeds_malig)
    if min(sum(malig_lables==0), sum(malig_lables==1)) < 50:
        return False

    print('getting labels')
    train_labels = clusterer.predict(train_embeds)
    cv_test_labels = clusterer.predict(cv_test_embeds)

    #We want majority class to be 0
    majority_label = mode(clusterer.predict(train_embeds_benign))
    if majority_label != 0:
        train_labels = 1 - train_labels
        cv_test_labels = 1 - cv_test_labels





    df_features_train['cluster'] = train_labels
    df_features_cv_test['cluster'] = cv_test_labels

    print('writing to file')

    df_clusters = pd.concat([df_features_train, df_features_cv_test])[['noduleID', 'cluster']]
    df_clusters.sort_values('noduleID', inplace=True)
    df_clusters.reset_index(drop=True, inplace=True)
    df_splits['cluster'] = [ 2 * m + c for m,c in zip(df_splits['malignancy_b'], df_clusters['cluster'])]



    # torch.save(model.state_dict(), f'./data/{results_dir}/erm_cluster_weights.pt')
    df_features_all.to_csv(f'./data/{results_dir}/erm_cluster_cnn_features.csv')

    f_reducer = f'./data/{results_dir}/cnn_umap_reducer.sav'
    pickle.dump(reducer, open(f_reducer, 'wb'))
    
    f_clusterer = f'./data/{results_dir}/cnn_umap_clusterer.sav'
    pickle.dump(clusterer, open(f_clusterer, 'wb'))

    df_splits.to_csv(f'./data/{results_dir}/LIDC_data_split_with_cluster.csv')
    cluster_stats.to_csv(f'./data/{results_dir}/cluster_stats.csv')

    return True

if __name__ == "__main__":


    found_good_cluster = False
    while not found_good_cluster:
        print('Run script')
        found_good_cluster = run_clustering()
