import torch
import loss
import models
import argparse

# hyperparameters

lr = 0.0005
wd = 0.005
eta = 0.01
gamma = 1.0

model_class = models.NeuralNetwork
optimizer_class = torch.optim.Adam
erm_class = loss.ERMLoss
gdro_class = loss.GDROLoss
dynamic_class = loss.DynamicLoss
upweight_class = loss.UpweightLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

trials = 100
epochs = 100
batch_size = 128
split_path = '../data/train_test_splits/LIDC_data_split.csv'
subclass_path = '../data/subclass_labels/LIDC_data_split_with_cluster.csv'
subclass_column = 'spic_groups'
feature_path = '../data/LIDC_designed_features.csv'

def main():
    print("Hello world")


if __name__ == '__main__':
    main()