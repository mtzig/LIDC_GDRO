import numpy as np
import torch


def train(dataloader, steps_per_epoch, model, loss_fn, optimizer):
    model.train()

    print(steps_per_epoch)

    steps_per_epoch = dataloader.dataset_len() /

    print(steps_per_epoch)

    for i in range(steps_per_epoch):

        loss = loss_fn(next(dataloader))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, steps_per_epoch, model, loss_fn, is_gdro):
    model.eval()

    test_loss = 0
    num_samples = []
    subgroup_correct = []
    with torch.no_grad():
        for i in range(steps_per_epoch):
            minibatch = next(dataloader)

            if len(subgroup_correct) == 0:
                subgroup_correct = np.zeros(len(minibatch))
                num_samples = np.zeros(len(minibatch))

            if is_gdro:
                test_loss += loss_fn(minibatch).item()

            for m in range(len(minibatch)):
                X, y = minibatch[m]
                batch_size = X.shape[0]
                pred = model(X)
                subgroup_correct[m] += (pred.argmax(1) == y).type(torch.float).sum().item()
                num_samples[m] += batch_size
                if not is_gdro:
                    test_loss += loss_fn(minibatch[m]).item()

    test_loss /= steps_per_epoch
    subgroup_accuracy = subgroup_correct / num_samples

    accuracy = sum(subgroup_correct)/sum(num_samples)

    print("Average Loss:", test_loss, "\nAccuracy:", accuracy, "\nAccuracy over subgroups:", subgroup_accuracy, "\nWorst Group Accuracy:", min(subgroup_accuracy))
