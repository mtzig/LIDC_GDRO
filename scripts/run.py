import torch
import loss

# hyperparameters
lr = 0.0005
wd = 0.005
eta = 0.01
gamma = 1.0

optimizer_class = torch.optim.Adam
loss_fn_class = loss.ERMLoss

batch_size = 128
split_path = '../data/train_test_splits/LIDC_data_split.csv'
subclass_path = '../data/subclass_labels/LIDC_data_split_with_cluster.csv'
subclass_column = 'spic_groups'
feature_path = '../data/LIDC_designed_features.csv'

