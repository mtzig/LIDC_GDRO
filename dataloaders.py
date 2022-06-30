"""
Code directly adapted from DomainBeds paper
"""


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch

from datasets import NoduleDataset


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, weights=None, num_workers=0):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=True,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=True)

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __next__(self):
        return next(self._infinite_iterator)

    def __len__(self):
        raise ValueError

    def batches_per_epoch(self):
        return len(self.dataset) // self.batch_size


class SubtypedDataLoader:
    '''
    A modified DataLoader, that divides a batch by subgroups so
    that we can calculate losses seperately for each subgroup

    '''

    def __init__(self, subtype_data, batch_size, total=False):
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

        subtype_data_sizes = list(map(lambda x: len(x[0]), subtype_data))  # list of datapoints per subclass
        total_data_size = sum(subtype_data_sizes)

        if total:

            # batch size for each subclass
            subtype_batch_sizes = list(map(lambda x: max(1, int(batch_size * x / total_data_size)), subtype_data_sizes))

            actual_batch_size = sum(subtype_batch_sizes)
            self._batches_per_epoch = total_data_size // actual_batch_size

        else:

            # we define epoch as the batches to go through smallest subclass
            self._batches_per_epoch = max(1, min(*subtype_data_sizes) // batch_size)

        self.dataloaders = []

        for idx, (features, labels) in enumerate(subtype_data):
            subtype_dataset = NoduleDataset(features, labels)
            subclass_batch_size = min(batch_size, subtype_data_sizes[idx]) if not total else subtype_batch_sizes[idx]

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
