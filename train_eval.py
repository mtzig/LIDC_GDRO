import numpy as np

import torch


def train(dataloader, model, loss_fn, optimizer, verbose=False):
    model.train()

    steps_per_epoch = dataloader.batches_per_epoch()

    avg_loss = 0

    for i in range(steps_per_epoch):

        loss = loss_fn(next(dataloader))
        avg_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss /= steps_per_epoch

    if verbose:
        print("Average training loss:", avg_loss)


def evaluate(dataloader, model, num_subclasses, verbose=False):
    model.eval()
    steps_per_epoch = dataloader.batches_per_epoch()

    num_samples = np.zeros(num_subclasses)
    subgroup_correct = np.zeros(num_subclasses)
    with torch.no_grad():
        for i in range(steps_per_epoch):
            minibatch = next(dataloader)

            X, y, c = minibatch
            pred = model(X)

            for subclass in range(num_subclasses):
                subclass_idx = c == subclass
                num_samples[subclass] += torch.sum(subclass_idx)
                subgroup_correct[subclass] += (pred[subclass_idx].argmax(1) == y[subclass_idx]).type(torch.float).sum().item()

    subgroup_accuracy = subgroup_correct / num_samples

    accuracy = sum(subgroup_correct)/sum(num_samples)

    if verbose:
        print("Accuracy:", accuracy, "\nAccuracy over subgroups:", subgroup_accuracy, "\nWorst Group Accuracy:", min(subgroup_accuracy))

    return accuracy, *subgroup_accuracy
