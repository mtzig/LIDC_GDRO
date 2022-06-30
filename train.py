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


def test(dataloader, model):
    model.eval()

    steps_per_epoch = dataloader.batches_per_epoch()

    num_samples = []
    subgroup_correct = []
    with torch.no_grad():
        for i in range(steps_per_epoch):
            minibatch = next(dataloader)

            if len(subgroup_correct) == 0:
                subgroup_correct = np.zeros(len(minibatch))
                num_samples = np.zeros(len(minibatch))

            for m in range(len(minibatch)):
                X, y = minibatch[m]
                batch_size = X.shape[0]
                pred = model(X)
                subgroup_correct[m] += (pred.argmax(1) == y).type(torch.float).sum().item()
                num_samples[m] += batch_size

    subgroup_accuracy = subgroup_correct / num_samples

    accuracy = sum(subgroup_correct)/sum(num_samples)

    print("Accuracy:", accuracy, "\nAccuracy over subgroups:", subgroup_accuracy, "\nWorst Group Accuracy:", min(subgroup_accuracy))

    return accuracy, subgroup_accuracy
