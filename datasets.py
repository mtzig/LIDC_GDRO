import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


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


class OnDemandWaterbirdsDataset(Dataset):
    def __init__(self, metadata, root_dir, transform, device):
        '''
        INPUTS:
        metadata: metadata dataframe storing the image paths, labels, and subclasses
        root_dir: the directory where the image files are stored
        transform: the transform to apply to the image when it is loaded
        device to move tensors to as they are loaded

        '''

        self.metadata = metadata
        self.root_dir = root_dir
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # column 1: image path
        img_path = self.metadata.iloc[idx, 1]
        img_tensor = self.transform(Image.open(self.root_dir + img_path)).squeeze().to(self.device)

        # column 2: image label
        label = torch.LongTensor([self.metadata.iloc[idx, 2]]).squeeze().to(self.device)

        # column 4 contains the confounding label, which is combined with column 2 to get the subclass
        subclass = torch.LongTensor([2 * self.metadata.iloc[idx, 2] + self.metadata.iloc[idx, 4]]).squeeze().to(self.device)

        return img_tensor, label, subclass
