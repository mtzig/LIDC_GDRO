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
        self.mix_gamma = mix_gamma
        self.q = torch.tensor([])
        self.model = model

    def __call__(self, minibatch):
        device = minibatch[0].device

        if len(self.q) == 0:
            self.q = torch.tensor([0.5, 0.5]).to(device)

        losses = torch.zeros(2).to(device)

        ermgdro_losses = self.ermgdro(minibatch)
        losses[0] = ermgdro_losses[1]  # ERM loss
        losses[1] = ermgdro_losses[2]  # GDRO loss

        if self.model.training:
            self.q *= torch.exp(self.mix_eta * losses.data)
            self.q /= self.q.sum()

        loss = torch.dot(self.q, losses)

        del losses

        return loss

class DynamicERMGDROLossAlt:
    def __init__(self, model, loss_fn, gdro_eta, mix_gamma, num_subclasses):
        
        self.loss_fn = loss_fn
        self.model = model

        self.gdro_eta = gdro_eta
        self.mix_gamma = mix_gamma

        self.g = torch.tensor([])
        self.q = torch.tensor([])

        self.num_subclasses = num_subclasses


    def __call__(self, minibatch):

        device = minibatch[0].device

        #initialize g, q if first step
        if len(self.g) == 0: 
            self.g = torch.tensor([1, 0], device=device, dtype=torch.float32)
            self.q = torch.ones(self.num_subclasses, device=device)
        

        #intialize losses
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

            #only compute loss if there are actually samples in that class
            if torch.sum(subclass_idx) > 0:
              losses[subclass] = self.loss_fn(preds[subclass_idx], y[subclass_idx])

        #update q
        if self.model.training:
            self.q *= torch.exp(self.gdro_eta * losses.data)
            self.q /= self.q.sum()
        
        #normalize loss
        losses *= subclass_freq

        ERM_GDRO_losses[0] = torch.sum(losses)
        ERM_GDRO_losses[1] = torch.dot(losses,self.q) * self.num_subclasses




        #update g
        if self.model.training:
            self.g *= torch.exp(self.mix_gamma * ERM_GDRO_losses.data)
            self.g /= torch.sum(self.g)

        loss = torch.dot(ERM_GDRO_losses, self.g)
        
        return loss
