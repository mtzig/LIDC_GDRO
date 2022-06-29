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


    def __init__(self, subtype_data, batch_size, total = False):
        '''
        INPUTS:
        subtype_data: list of data for each subclass, e.g. 
                       [(features_subclass_0, labels_subclass_0), 
                        (features_subclass_1, labels_subclass_1), 
                        ...]

        batch_size  : either a number indicating uniform batch size for each subclass,
                       or a batch size for entire minibatch
        
        total       : False if batch_size is for each subclass,
                      True if batch_size is for entire minibatch, (NOTE: actual batch size 
                      might be different due to rouding)     

        '''


        subtype_data_sizes = list(map(lambda x:len(x[0]), subtype_data)) #list of datapoints per subclass
        total_data_size = sum(subtype_data_sizes)

        if total:

            #batch size for each subclass
            subtype_batch_sizes = list(map(lambda x:max(1,int(batch_size * x/total_data_size)), subtype_data_sizes))
            
            actual_batch_size = sum(subtype_batch_sizes)
            self._batches_per_epoch = total_data_size // actual_batch_size

        else:

            #we define epoch as the batches to go through smallest subclass
            self._batches_per_epoch = max(1, min(*subtype_data_sizes) // batch_size)

        self.dataloaders = []

        for idx, (features, labels) in enumerate(subtype_data):
            
            subtype_dataset = NoduleDataset(features, labels)
            subclass_batch_size = min(batch_size, subtype_data_sizes[idx]) if not total else subtype_batch_sizes[idx]

            print(subclass_batch_size)

            subtype_iter_loader = InfiniteDataLoader(subtype_dataset, subclass_batch_size)
            self.dataloaders.append(subtype_iter_loader)
            
        self.minibatch_iterator = zip(*self.dataloaders)

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

    def batches_per_epoch(self):
        return self._batches_per_epoch
