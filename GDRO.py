import torch
from torch import nn


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


class GDROLoss:
    def __init__(self, model, loss_fn, hparams):
        self.model = model
        self.loss_fn = loss_fn
        self.q = torch.tensor([])
        self.hparams = hparams

    def __call__(self, minibatch):
        device = "cuda" if minibatch[0][0].is_cuda else "cpu"

        if len(self.q) == 0:
            self.q = torch.ones(len(minibatch)).to(device)

        losses = torch.zeros(len(minibatch)).to(device)

        for m in range(len(minibatch)):
            X, y = minibatch[m]
            losses[m] = self.loss_fn(self.model(X), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        return loss


class ERMLoss:
    def __init__(self, model, loss_fn, hparams):
        self.model = model
        self.loss_fn = loss_fn
        self.hparams = hparams

    def __call__(self, minibatch):

        X, y = minibatch

        loss = self.loss_fn(self.model(X), y)

        return loss


def train(dataloader, model, loss_fn, optimizer):
    model.train()

    for minibatch in dataloader:

        loss = loss_fn(minibatch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, model, loss_fn, is_gdro, batch_size):
    num_batches = len(dataloader)

    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for minibatch in dataloader:
            if is_gdro:
                X = torch.cat([m[0] for m in minibatch])
                y = torch.cat([m[1] for m in minibatch])
            else:
                X, y = minibatch

            pred = model(X)

            test_loss += loss_fn(minibatch).item()

            print(X)

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    # correct /= num_batches * batch_size

    print("Average Loss:", test_loss)#, "\nAccuracy:", correct)
