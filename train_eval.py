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
                subgroup_correct[subclass] += (pred[subclass_idx].argmax(1) == y[subclass_idx]).type(
                    torch.float).sum().item()

    subgroup_accuracy = subgroup_correct / num_samples

    accuracy = sum(subgroup_correct) / sum(num_samples)

    if verbose:
        print("Accuracy:", accuracy, "\nAccuracy over subgroups:", subgroup_accuracy, "\nWorst Group Accuracy:",
              min(subgroup_accuracy))

    return (accuracy, *subgroup_accuracy)


def train_epochs(epochs,
                 train_dataloader,
                 val_dataloader,
                 model,
                 loss_fn,
                 optimizer,
                 scheduler=None,
                 verbose=False,
                 record_accuracies=False,
                 num_subclasses=1):
    if record_accuracies:
        results = []

    for epoch in range(epochs):
        if verbose:
            print(f'Epoch {epoch + 1} / {epochs}')

        train(train_dataloader, model, loss_fn, optimizer)
        if scheduler:
            scheduler.step(evaluate(val_dataloader, model, num_subclasses=num_subclasses)[0])

        if record_accuracies:
            accuracies = evaluate(val_dataloader, model, num_subclasses=num_subclasses)
            results.extend(accuracies)

    if record_accuracies:
        return results
    else:
        return None


def run_trials(num_trials,
               epochs,
               train_dataloader,
               val_dataloader,
               model,
               loss_fn,
               optimizer,
               scheduler=None,
               verbose=False,
               record_accuracies=False,
               num_subclasses=1):

    if record_accuracies:
        results = []

    for n in range(num_trials):
        if verbose:
            print(f"Trial {n + 1}/{num_trials}")

        accuracies = train_epochs(epochs,
                                  train_dataloader,
                                  val_dataloader,
                                  model,
                                  loss_fn,
                                  optimizer,
                                  scheduler,
                                  verbose,
                                  record_accuracies,
                                  num_subclasses)

        if record_accuracies:
            results.extend(accuracies)

    if record_accuracies:
        return results
    else:
        return None
