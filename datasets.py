import torch
from torch.utils.data import DataLoader, Dataset


# functions to create our datasets and dataloaders


# SubtypedDataLoader class

# inputs:
# subclass_class_data: {subclass_1: (image_list, label_list), subclass_2: (image_list, label_list), ...}
# batch_size
# iterable
# gives a list of size n_subclasses
# each element is a batch of tensor from specific subclass

class NoduleDataset(Dataset):

    def __init__(self, feature_array, label):
        self.feature_array = feature_array
        self.label = label

    def __len__(self):
        return len(self.feature_array)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.feature_array[idx], self.label[idx]


# wraps dataloader class to provide data separated by subgroup
class SubtypedDataLoader:

    def __init__(self, subtype_data, batch_size, shuffle=True):
        self.dataloaders = []

        for subtype in subtype_data:
            subtype_dataset = NoduleDataset(*subtype_data[subtype])
            subtype_iter_loader = iter(DataLoader(subtype_dataset, batch_size, shuffle=shuffle))
            self.dataloaders.append(subtype_iter_loader)

    def __iter__(self):
        return self

    def __next__(self):
        # list of batches
        # ex: 3 subclasses
        # [(X_subclass_1, y_subclass_1),(X_subclass_2, y_subclass_2),(X_subclass_3, y_subclass_3)]
        minibatch = []
        for iterLoader in self.dataloaders:
            minibatch.append(next(iterLoader))

        return minibatch
