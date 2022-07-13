import torch


class GDROLoss:
    def __init__(self, model, loss_fn, hparams, normalize_loss=False):
        self.model = model
        self.loss_fn = loss_fn
        self.q = torch.tensor([])
        self.eta = hparams["groupdro_eta"]
        self.normalize_loss = normalize_loss

    def __call__(self, minibatch):
        # minibatch contains n batches of data where n is the number of subtypes

        device = "cuda" if minibatch[0][0].is_cuda else "cpu"

        if len(self.q) == 0:
            self.q = torch.ones(len(minibatch)).to(device)
            self.q /= self.q.sum()

        losses = torch.zeros(len(minibatch)).to(device)

        if self.normalize_loss:
            subgroup_batch_sizes = list(map(lambda x:x[0].shape[0], minibatch))
            total_samples = sum(subgroup_batch_sizes)

        for m in range(len(minibatch)):
            X, y = minibatch[m]
            losses[m] = self.loss_fn(self.model(X), y)

        if self.model.training:
            self.q *= torch.exp(self.eta * losses.data)  # vectorized (might not work)
            self.q /= self.q.sum()

        # print(self.q)

        loss = torch.dot(losses, self.q)

        return loss


class GDROLossAlt:
    '''
    GDROLoss function to be used with SubclassedNoduleDataset
    '''

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

            #only compute loss if there are actually samples in that class
            if torch.sum(subclass_idx) > 0:
              losses[subclass] = self.loss_fn(preds[subclass_idx], y[subclass_idx])
        

        #update q
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
    def __init__(self, model, loss_fn, hparams, subclassed=False):
        self.model = model
        self.loss_fn = loss_fn
        self.hparams = hparams
        self.subclassed = subclassed #true if we are using SubclassedNoduleDataset

    def __call__(self, minibatch):
        # minibatch contains one batch of non-subtyped data
        
        if self.subclassed:
            X, y, _ = minibatch
        else:
            X, y = minibatch

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


class DynamicERMGDROLoss:
    def __init__(self, model, loss_fn, gdro_eta, mix_eta, num_subclasses):
        self.ermgdro = ERMGDROLoss(model, loss_fn, gdro_eta, num_subclasses)
        self.mix_eta = mix_eta
        self.q = torch.tensor([])
        self.model = model

    def __call__(self, minibatch):
        device = minibatch[0].device

        if len(self.q) == 0:
            self.q = torch.tensor([0.5, 0.5]).to(device)

        erm_loss = self.ermgdro(minibatch)[1]
        gdro_loss = self.ermgdro(minibatch)[2]
        losses = torch.zeros(2).to(device)
        losses[0] = erm_loss
        losses[1] = gdro_loss

        if self.model.training:
            self.q *= torch.exp(self.mix_eta * losses.data)
            self.q /= self.q.sum()

        loss = torch.dot(self.q, losses)

        return loss
