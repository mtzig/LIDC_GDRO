import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import SubtypedDataLoader


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


def train(dataloader: SubtypedDataLoader, model: nn.Module, loss_fn, optimizer):
    model.train()

    gdro_loss = GDROLoss(model, loss_fn)

    for minibatch in enumerate(dataloader):

        loss = gdro_loss(minibatch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class GDROLoss:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        self.q = torch.tensor([])

    def __call__(self, minibatch):
        device = "cuda" if minibatch[0][0].is_cuda else "cpu"

        if len(self.q) == 0:
            self.q = torch.ones(len(minibatch)).to(device)

        losses = torch.zeros(len(minibatch)).to(device)

        for m in range(len(minibatch)):
            X, y = minibatch[m]
            losses[m] = nn.CrossEntropyLoss(self.model(X), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        return loss

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
