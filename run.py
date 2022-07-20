import torch
from loss import ERMLoss, GDROLoss, DynamicLoss, UpweightLoss
import models
from utils import data_utils
from train_eval import run_trials
import pandas as pd
from datetime import datetime
import argparse

# hyperparameters

lr = 0.0005
wd = 0.005
eta = 0.01
gamma = 1.0

model_class = models.NeuralNetwork
optimizer_class = torch.optim.Adam

device = "cuda" if torch.cuda.is_available() else "cpu"

trials = 100
epochs = 100
batch_size = 128
split_path = 'data/train_test_splits/LIDC_data_split.csv'
subclass_path = 'data/subclass_labels/LIDC_data_split_with_cluster.csv'
subclass_column = 'spic_groups'
feature_path = 'data/LIDC_designed_features.csv'


df = data_utils.preprocess_data(
    *data_utils.load_lidc(
        data_root='data/',
        feature_path='LIDC_20130817_AllFeatures2D_MaxSlicePerNodule_inLineRatings.csv',
        subclass_path='subclass_labels/LIDC_data_split_with_cluster.csv'),
    subclass_column='spic_groups')

num_subclasses = len(df['subclass'].unique())
subtypes = ["Overall"]
subtypes.extend(["Unspiculated benign", "Spiculated benign", "Spiculated malignant", "Unspiculated malignant"])

erm = ERMLoss(None, torch.nn.CrossEntropyLoss())
gdro = GDROLoss(None, torch.nn.CrossEntropyLoss(), eta, num_subclasses)
dynamic = DynamicLoss(None, torch.nn.CrossEntropyLoss(), eta, gamma, num_subclasses)
upweight = UpweightLoss(None, torch.nn.CrossEntropyLoss(), num_subclasses)

train_dataloader, val_dataloader, test_dataloader = data_utils.train_val_test_dataloaders(df, split_path="data/train_test_splits/LIDC_data_split.csv", batch_size=batch_size)


model_args = [64, 36, 2]
optimizer_args = {'lr': lr, 'weight_decay': wd}

results = {"Accuracies": {}, "q": {}, "g": {}, "ROC": {}}

for loss_fn in [erm, gdro, dynamic, upweight]:
    fn_name = loss_fn.__class__.__name__
    accuracies, q_data, g_data, roc_data = run_trials(
        num_trials=trials,
        epochs=epochs,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        model_class=model_class,
        model_args=model_args,
        loss_fn=loss_fn,
        optimizer_class=optimizer_class,
        optimizer_args=optimizer_args,
        device=device,
        scheduler=None,
        verbose=False,
        record=True,
        num_subclasses=num_subclasses
    )
    results["Accuracies"][fn_name] = accuracies
    results["q"][fn_name] = q_data
    results["g"][fn_name] = g_data
    results["ROC"][fn_name] = zip(roc_data)

now = datetime.now()

accuracies_df = pd.DataFrame(
    results["Accuracies"],
    index=pd.MultiIndex.from_product(
        [range(trials), range(epochs), subtypes],
        names=["trial", "epoch", "subtype"]
    )
)
q_df = pd.DataFrame(
    results["q"],
    index=pd.MultiIndex.from_product(
        [range(trials), range(epochs), subtypes[1:]],
        names=["trial", "epoch", "subtype"]
    )
)
g_df = pd.DataFrame(
    results["q"],
    index=pd.MultiIndex.from_product(
        [range(trials), range(epochs), subtypes[1:]],
        names=["trial", "epoch", "subtype"]
    )
)
roc_df = pd.DataFrame(results["ROC"])

results_dir = 'test_results/standardized/LIDC_designed_features_spic_groups/'

accuracies_df.to_csv(results_dir + f'accuracies_{now.strftime("%Y%m%d_%H%M%S")}.csv')
q_df.to_csv(results_dir + f'q_{now.strftime("%Y%m%d_%H%M%S")}.csv')
g_df.to_csv(results_dir + f'g_{now.strftime("%Y%m%d_%H%M%S")}.csv')
roc_df.to_csv(results_dir + f'roc_{now.strftime("%Y%m%d_%H%M%S")}.csv')