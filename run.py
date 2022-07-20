import torch
from loss import ERMLoss, GDROLoss, DynamicLoss, UpweightLoss
import models
from utils import data_utils
from train_eval import run_trials
import argparse

# hyperparameters

lr = 0.0005
wd = 0.005
eta = 0.01
gamma = 1.0

model_class = models.NeuralNetwork
optimizer_class = torch.optim.Adam
erm_class = ERMLoss
gdro_class = GDROLoss
dynamic_class = DynamicLoss
upweight_class = UpweightLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

trials = 1
epochs = 10
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
    subclass_column='cluster')

train_dataloader, val_dataloader, test_dataloader = data_utils.train_val_test_dataloaders(df, split_path="data/train_test_splits/LIDC_data_split.csv", batch_size=batch_size)


model_args = [64, 36, 2]

loss_class = erm_class
loss_args = [torch.nn.CrossEntropyLoss]

optimizer_args = [lr, wd]


accuracies, q_data, g_data, roc_data = run_trials(
    trials,
    epochs,
    train_dataloader,
    test_dataloader,
    model_class,
    model_args,
    loss_class,
    loss_args,
    optimizer_class,
    optimizer_args,
    device,
    scheduler=None,
    verbose=False,
    record=False,
    num_subclasses=1
)

print(accuracies)
