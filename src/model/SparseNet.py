import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseNet(nn.Module):

    def __init__(self, K:int, M:int, R_lr:float=0.1, lmda:float=5e-3, device=None):
        super(SparseNet, self).__init__()
        self.K = K
        self.M = M
        self.R_lr = R_lr
        self.lmda = lmda
        # synaptic weights
        self.device = torch.device("cpu") if device is None else device
        self.U = nn.Linear(self.K, self.M ** 2, bias=False).to(self.device)
        # responses
        self.R = None
        self.normalize_weights()

    def ista_(self, img_batch):
        # create R
        self.R = torch.zeros((img_batch.shape[0], self.K), requires_grad=True, device=self.device)
        converged = False
        # update R
        optim = torch.optim.SGD([{'params': self.R, "lr": self.R_lr}])
        # train
        while not converged:
            old_R = self.R.clone().detach()
            # pred
            pred = self.U(self.R)
            # loss
            loss = ((img_batch - pred) ** 2).sum()
            loss.backward()
            # update R in place
            optim.step()
            # zero grad
            self.zero_grad()
            # prox
            self.R.data = SparseNet.soft_thresholding_(self.R, self.lmda)
            # convergence
            converged = torch.norm(self.R - old_R) / torch.norm(old_R) < 0.01

    @staticmethod
    def soft_thresholding_(x, alpha):
        with torch.no_grad():
            rtn = F.relu(x - alpha) - F.relu(-x - alpha)
        return rtn.data

    def zero_grad(self):
        self.R.grad.zero_()
        self.U.zero_grad()

    def normalize_weights(self):
        with torch.no_grad():
            self.U.weight.data = F.normalize(self.U.weight.data, dim=0)

    def forward(self, img_batch):
        # first fit
        self.ista_(img_batch)
        # now predict again
        pred = self.U(self.R)
        return pred


