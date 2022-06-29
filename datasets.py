import torch
from torch.utils.data import Dataset
from fast_data_loader import InfiniteDataLoader


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
        
        label = self.labels[0] if self.singular else self.labels[idx]
        return self.features[idx], label

class SubtypedDataLoader:
    '''
    A modified DataLoader, that divides a batch by subgroups so
    that we can calculate losses seperately for each subgroup

    '''


    def __init__(self, subtype_data, batch_size):
        '''
        INPUTS:
        subtyped_data: list of data for each subclass, e.g. 
                       [(features_subclass_0, labels_subclass_0), 
                        (features_subclass_1, labels_subclass_1), 
                        ...]

        batch_size  : either a number indicating uniform batch size for each subclass,
                       or a tensor of length number of subclasses with batchsize for each subclass
        '''
        dataloaders = []

        for idx, (features, labels) in enumerate(subtype_data):
            
            subtype_dataset = NoduleDataset(features, labels)
            subclass_batch_size = batch_size if type(batch_size) == int else batch_size[idx]
            
            subtype_iter_loader = InfiniteDataLoader(subtype_dataset, subclass_batch_size)
            dataloaders.append(subtype_iter_loader)
            
        self.minibatch_iterator = zip(*dataloaders)

    def __iter__(self):
        return self

    def __next__(self):
        '''
        OUTPUT: a list of length number of subclasses where each element is a tuple
                of a batch of features and labels, e.g.
                [(X_subclass_0, y_subclass_0), 
                 (X_subclass_1, y_subclass_1), 
                 (X_subclass_2, y_subclass_2),
                 ...]
        '''
        return next(self.minibatch_iterator)