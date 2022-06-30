import torch


class GDROLoss:
    def __init__(self, model, loss_fn, hparams, normalize_loss=False):
        self.model = model
        self.loss_fn = loss_fn
        self.q = torch.tensor([])
        self.hparams = hparams
        self.normalize_loss = normalize_loss

    def __call__(self, minibatch):
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


class ERMLoss:
    def __init__(self, model, loss_fn, hparams):
        self.model = model
        self.loss_fn = loss_fn
        self.hparams = hparams

    def __call__(self, minibatch):

        X, y = minibatch

        loss = self.loss_fn(self.model(X), y)

        return loss
