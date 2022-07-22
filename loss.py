import torch
'test'

class GDROLoss:
    """
    GDROLoss function to be used with SubclassedNoduleDataset
    """

    def __init__(self, model, loss_fn, eta, num_subclasses, normalize_loss=False):
        self.model = model
        self.loss_fn = loss_fn
        self.q = torch.tensor([])
        self.eta = eta
        self.num_subclasses = num_subclasses
        self.normalize_loss = normalize_loss

        # for debug purposes
        self.losses = torch.tensor([])

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

        # store losses for retrieval by debug program
        self.losses = losses

        return loss


class ERMLoss:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn

    def __call__(self, minibatch):
        # minibatch contains one batch of non-subtyped data

        X, y, _ = minibatch

        loss = self.loss_fn(self.model(X), y)

        return loss


class ERMGDROLoss:
    def __init__(self, model, loss_fn, eta, num_subclasses):
        self.eta = eta
        self.model = model
        self.loss_fn = loss_fn
        self.q = torch.tensor([])
        self.t = 1
        self.num_subclasses = num_subclasses

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

        # loss has to be normalized
        losses *= subclass_counts
        gdro_loss = torch.dot(losses, self.q)
        gdro_loss /= batch_size
        gdro_loss *= self.num_subclasses

        erm_loss = torch.sum(losses)

        loss = self.t * erm_loss + (1 - self.t) * gdro_loss

        return loss, erm_loss, gdro_loss


class DynamicLoss:
    def __init__(self, model, loss_fn, eta, gamma, num_subclasses, initial=None):

        if initial is None:
            initial = [0.5, 0.5]
        self.loss_fn = loss_fn
        self.model = model

        self.eta = eta
        self.gamma = gamma

        self.g = torch.tensor([])
        self.q = torch.tensor([])

        self.num_subclasses = num_subclasses

        self.initial = initial

    def __call__(self, minibatch):

        device = minibatch[0].device

        # initialize g, q if first step
        if len(self.g) == 0:
            self.g = torch.tensor(self.initial, device=device, dtype=torch.float32)
            self.q = torch.ones(self.num_subclasses, device=device)

        # intialize losses
        losses = torch.zeros(self.num_subclasses, device=device)
        subclass_freq = torch.zeros(self.num_subclasses, device=device)
        ERM_GDRO_losses = torch.zeros(2, device=device)

        X, y, c = minibatch
        batch_size = X.shape[0]

        preds = self.model(X)

        # computes loss
        # get relative frequency of samples in each subclass
        for subclass in range(self.num_subclasses):
            subclass_idx = c == subclass
            subclass_freq[subclass] = torch.sum(subclass_idx) / batch_size

            # only compute loss if there are actually samples in that class
            if torch.sum(subclass_idx) > 0:
                losses[subclass] = self.loss_fn(preds[subclass_idx], y[subclass_idx])

        # update q
        if self.model.training:
            self.q *= torch.exp(self.eta * losses.data)
            self.q /= self.q.sum()

        # normalize loss
        losses *= subclass_freq

        ERM_GDRO_losses[0] = torch.sum(losses)
        ERM_GDRO_losses[1] = torch.dot(losses, self.q) * self.num_subclasses

        print(ERM_GDRO_losses)

        # update g
        if self.model.training:
            self.g *= torch.exp(self.gamma * ERM_GDRO_losses.data)
            self.g /= torch.sum(self.g)

        loss = torch.dot(ERM_GDRO_losses, self.g)

        return loss


class UpweightLoss:
    def __init__(self, model, loss_fn, num_subclasses):
        self.model = model
        self.loss_fn = loss_fn
        self.num_subclasses = num_subclasses

    def __call__(self, minibatch):
        X, y, c = minibatch

        device = X.device

        losses = torch.zeros(self.num_subclasses).to(device)

        preds = self.model(X)

        # computes loss
        # get relative frequency of samples in each subclass
        for subclass in range(self.num_subclasses):
            subclass_idx = c == subclass

            # only compute loss if there are actually samples in that class
            if torch.sum(subclass_idx) > 0:
                losses[subclass] = self.loss_fn(preds[subclass_idx], y[subclass_idx])

        loss = torch.sum(losses) / self.num_subclasses

        return loss
