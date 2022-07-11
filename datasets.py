import torch
from torch.utils.data import Dataset


class NoduleDataset(Dataset):

    def __init__(self, features, labels, singular=False):
        '''
        INPUTS:
        features: list of features (as Pytorch tensors)
        labels:   list of corresponding lables (if singular is false),
                  otherwise, a tensor of size 1 with just the class label
        singular: False if labels is list of lables corresponding to features,
                  True if all features have only one label
        
        '''


        self.features = features
        self.labels = labels
        self.singular = singular

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label = self.labels if self.singular else self.labels[idx]
        return self.features[idx], label


class SubclassedNoduleDataset(Dataset):

    def __init__(self, features, labels, subclasses,):
        '''
        INPUTS:
        features: list of features (as Pytorch tensors)
        labels:   list of corresponding lables
        subclasses: list of corresponding subclasses
        
        '''


        self.features = features
        self.labels = labels
        self.subclasses = subclasses


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.features[idx], self.labels[idx], self.subclasses[idx]
