import numpy as np
import torch


def train(dataloader, steps_per_epoch, model, loss_fn, optimizer):
    model.train()
    for i in range(steps_per_epoch):

        loss = loss_fn(next(dataloader))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, steps_per_epoch, batch_size, model, loss_fn, is_gdro):
    num_batches = steps_per_epoch

    model.eval()

    test_loss, correct = 0, []
    with torch.no_grad():
        for i in range(steps_per_epoch):
            minibatch = next(dataloader)

            if len(correct) == 0:
                correct = np.zeros(len(minibatch))

            if is_gdro:
                test_loss += loss_fn(minibatch).item()

            for m in range(len(minibatch)):
                X, y = minibatch[m]
                pred = model(X)
                correct[m] += (pred.argmax(1) == y).type(torch.float).sum().item()
                if not is_gdro:
                    test_loss += loss_fn(minibatch[m]).item()

    test_loss /= num_batches
    correct /= num_batches * batch_size

    print("Average Loss:", test_loss, "\nAccuracy:", sum(correct)/len(correct), "\nAccuracy over subgroups:", correct, "\nWorst Group Accuracy:", min(correct))