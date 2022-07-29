import torch


class GDROLoss:
    """
    Implements the gDRO loss function
    See https://arxiv.org/abs/1911.08731 for details on the algorithm
    """

    def __init__(self, model, loss_fn, eta, num_subclasses, normalize_loss=False):
        self.model = model
        self.loss_fn = loss_fn
        self.q = torch.tensor([])
        self.eta = eta
        self.num_subclasses = num_subclasses
        self.normalize_loss = normalize_loss

    def __call__(self, minibatch):

        X, y, c = minibatch

        batch_size = X.shape[0]
        device = X.device

        if len(self.q) == 0:
            self.q = torch.ones(self.num_subclasses).to(device)
            self.q /= self.q.sum()

        losses = torch.zeros(self.num_subclasses).to(device)

        subclass_counts = torch.zeros(self.num_subclasses).to(device)

        preds = self.model(X)

        # computes loss
        # get relative frequency of samples in each subclass
        for subclass in range(self.num_subclasses):
            subclass_idx = c == subclass
            subclass_counts[subclass] = torch.sum(subclass_idx)

            # only compute loss if there are actually samples in that class
            if torch.sum(subclass_idx) > 0:
                losses[subclass] = self.loss_fn(preds[subclass_idx], y[subclass_idx])

        # update q
        if self.model.training:
            self.q *= torch.exp(self.eta * losses.data)
            self.q /= self.q.sum()

        if self.normalize_loss:
            losses *= subclass_counts
            loss = torch.dot(losses, self.q)
            loss /= batch_size
            loss *= self.num_subclasses
        else:
            loss = torch.dot(losses, self.q)

        return loss


class ERMLoss:
    """
    Implements a standard Empirical Risk Minimization loss function
    Takes the classifier model and an underlying loss function as input, ex. a neural network for the model and cross-entropy loss as the underlying function
    """

    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn

    def __call__(self, minibatch):
        # minibatch contains one batch of non-subtyped data

        X, y, _ = minibatch

        loss = self.loss_fn(self.model(X), y)

        return loss


