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


def get_activation(name):
    '''
    Hooks used to extract CNN activation
    '''
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def collect_features(model), images_df=None):
    if images_df is None:
        images_df = images_to_df()
    activation = {}

    model.model.avgpool.register_forward_hook(get_activation('avgpool'))
    noduleID, data = images_df['noduleID'], torch.stack(list(images_df['image'])).to(DEVICE)
    model(data)

    return noduleID, activation['avgpool'].squeeze()

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
                 scheduler=scheduler, verbose=False, num_subclasses=4)
    
    return model
