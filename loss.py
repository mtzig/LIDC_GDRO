import torch


class GDROLoss:
    def __init__(self, model, loss_fn, hparams, normalize_loss=False):
        self.model = model
        self.loss_fn = loss_fn
        self.q = torch.tensor([])
        self.hparams = hparams
        self.normalize_loss = normalize_loss

    def __call__(self, minibatch):
        # minibatch contains n batches of data where n is the number of subtypes

        device = "cuda" if minibatch[0][0].is_cuda else "cpu"

        if len(self.q) == 0:
            self.q = torch.ones(len(minibatch)).to(device)

        losses = torch.zeros(len(minibatch)).to(device)

        if self.normalize_loss:
            subgroup_batch_sizes = list(map(lambda x:x[0].shape[0], minibatch))
            total_samples = sum(subgroup_batch_sizes)

        for m in range(len(minibatch)):
            X, y = minibatch[m]
            losses[m] = self.loss_fn(self.model(X), y)

            if self.normalize_loss:
                losses[m] *= subgroup_batch_sizes[m] / total_samples
            #if self.model.training:
                #self.q[m] *= torch.exp((self.hparams["groupdro_eta"] * losses[m].data))

        if self.model.training:
            self.q *= torch.exp(self.hparams["groupdro_eta"] * losses.data) #vectorized (might not work)
            self.q /= self.q.sum()

        # print(self.q)

        loss = torch.dot(losses, self.q)

        return loss


class GDROLossAlt:
    '''
    GDROLoss function to be used with SubclassedNoduleDataset
    '''

    def __init__(self, model, loss_fn, eta, num_subclasses, rescale=False):
        self.model = model
        self.loss_fn = loss_fn
        self.q = torch.tensor([])
        self.eta = eta
        self.num_subclasses = num_subclasses
        self.rescale = rescale

    def __call__(self, minibatch):
        
        X, y, c = minibatch

        batch_size = X.shape[0]
        device = X.device

        if len(self.q) == 0:
            self.q = torch.ones(self.num_subclasses).to(device)

        losses = torch.zeros(self.num_subclasses).to(device)

        subclass_counts = torch.zeros(self.num_subclasses).to(device)

        preds = self.model(X)

        # computes loss
        # get relative frequency of samples in each subclass        
        for subclass in range(self.num_subclasses):
            subclass_idx = c == subclass
            subclass_counts[subclass] = torch.sum(subclass_idx)

            #only compute loss if there are actually samples in that class
            if subclass_counts[subclass] > 0:
              losses[subclass] = self.loss_fn(preds[subclass_idx], y[subclass_idx])
        

        #update q
        if self.model.training:
            self.q *= torch.exp(self.eta * losses.data)
            self.q /= self.q.sum()

        #rescale loss
        if self.rescale:
          losses *= subclass_counts
          loss = torch.dot(losses, self.q)
          loss /= batch_size
          loss *= self.num_subclasses
        else:
          loss = torch.dot(losses, self.q)

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
    def __init__(self, model, loss_fn, hparams, normalize_loss=False):
        self.hparams = hparams
        self.model = model
        self.t = 1
        self.erm = ERMLoss(model, loss_fn, hparams)
        self.gdro = GDROLoss(model, loss_fn, hparams, normalize_loss)

        # used to record the initial loss to set a baseline for the interpolation
        self.initial_loss = 0

    def __call__(self, minibatch):
        # minibatch contains n batches of data where n is the number of subtypes

        # remove subtype info for the ERM loss function
        unsubtyped_batch = torch.cat(minibatch)

        # linearly interpolate between ERM and GDRO loss functions
        loss = self.t * self.erm(unsubtyped_batch) + (1 - self.t) * self.gdro(minibatch)

        # record initial loss value
        if self.initial_loss == 0:
            self.initial_loss = loss.item()

        # update t based on the loss
        if self.model.training:
            # reduce t proportional to loss TODO experiment with different ways to change t as the model trains
            t = loss.item() / self.initial_loss

        return loss
