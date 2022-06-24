import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np


# generic fully-connected neural network class
class NeuralNetwork(nn.Module):

    # layers = int layer sizes, starting with the input layer
    def __init__(self, *layers):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        stack = []
        for i in range(len(layers) - 1):
            stack.extend([nn.Linear(layers[i], layers[i + 1]), nn.ReLU()])

        self.linear_relu_stack = nn.Sequential(*stack)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_gdro(dataloader: DataLoader, model: nn.Module, loss_fn, num_groups: int, optimizer, device):
    model.train()

    # initialize group weights

    for (X, y) in enumerate(dataloader):

        # X is a float tensor of size (number of groups, batch size, number of features)
        # y is a long tensor of size (number of groups, batch size)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        # pred and loss are stored by group
        num_groups = len(X)
        pred = [model(X[group]) for group in range(num_groups)]
        loss = [loss_fn(pred[group], y[group]) for group in pred]

        # Update weights for group g
        q = []

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader: DataLoader, model: nn.Module, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
