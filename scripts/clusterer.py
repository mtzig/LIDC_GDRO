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
from utils.cluster_utils import do_clustering
from matplotlib import pyplot as plt
import pickle
from sklearn.mixture import GaussianMixture
from datetime import datetime
from scipy.stats import mode



# def save()
#     # torch.save(model.state_dict(), f'./data/{results_dir}/erm_cluster_weights.pt')
#     df_features_all.to_csv(f'./data/{results_dir}/erm_cluster_cnn_features.csv')

#     f_reducer = f'./data/{results_dir}/cnn_umap_reducer.sav'
#     pickle.dump(reducer, open(f_reducer, 'wb'))
    
#     f_clusterer = f'./data/{results_dir}/cnn_umap_clusterer.sav'
#     pickle.dump(clusterer, open(f_clusterer, 'wb'))

#     df_splits.to_csv(f'./data/{results_dir}/LIDC_data_split_with_cluster.csv')
#     cluster_stats.to_csv(f'./data/{results_dir}/cluster_stats.csv')

#     return True


if __name__ == "__main__":

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    #make unique directory to store results
    now = datetime.now()

    images_df = images_to_df()
    train_data, cv_data, test_data = get_features(images=True, features=images_df, device=DEVICE, subclass='malignancy')


    #datasets
    tr = SubclassedDataset(*train_data)
    cv = SubclassedDataset(*cv_data)
    tst = SubclassedDataset(*test_data)

    #dataloaders
    tr_loader = InfiniteDataLoader(tr, batch_size=512)
    cv_loader = InfiniteDataLoader(cv, len(cv))
    tst_loader = InfiniteDataLoader(tst, len(tst))

    counts = 0
    while counts < 10:

        results = do_clustering(tr_loader, cv_loader, tst_loader, images_df, device=DEVICE)
        if results is None:
            continue
        

        label_df, embeds, silhouette_scores = results
        malig_plot = show_scatter(embeds[0][:,0], embeds[0][:,1], embeds[1], 'train embeds labeled by malignancy', 0.5 )
        cluster_plot = show_scatter(embeds[0][:,0], embeds[0][:,1], embeds[2], 'train embeds labeled by cluster', 0.5 )

        results_dir = f'cluster_results/cluster_{now.strftime("%Y%m%d_%H%M%S")}{counts}'
        os.mkdir(f'./data/{results_dir}')

        label_df.to_csv(f'./data/{results_dir}/cluster_labels')

        malig_plot.savefig(f'./data/{results_dir}/malig_plot.png')
        cluster_plot.savefig(f'./data/{results_dir}/cluster_plot.png')

        counts += 1

        