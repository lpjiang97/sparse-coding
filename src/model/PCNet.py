import torch
import torch.nn as nn
import torch.nn.functional as F


class PCNet(nn.Module):

    def __init__(self, K:int, M:int, R_epochs:int=150, R_lr:float=0.1, lmda:float=5e-3):
        '''
        Create a sparse coding network. Neural responses are fitted through ISTA algorithm.

        Args:
            K: number of neurons
            M: size of receptive field (width / height)
            R_epochs: number of epochs to run for ISTA
            R_lr: learning rate for ISTA
            lmda: regularization strength for ISTA
        '''
        super(PCNet, self).__init__()
        self.K = K
        self.M = M
        self.R_epochs = R_epochs
        self.R_lr = R_lr
        self.lmda = lmda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model weigths
        self.U = torch.randn(self.K, self.M ** 2, requires_grad=True, device=self.device)
        with torch.no_grad():
            self.U = F.normalize(self.U, dim=1)
        self.U.requires_grad_(True)
        # responses
        self.R = None

    def _ista(self, img_batch):
        # create R
        batch_size = img_batch.shape[0]
        self.R = torch.zeros((batch_size, self.K), requires_grad=True, device=self.device)
        # trian
        for _ in range(self.R_epochs):
            # pred
            pred = self.R @ self.U
            # loss
            loss = ((img_batch - pred) ** 2).sum()
            loss.backward()
            # update R in place
            self.R.data.sub_(self.R_lr * self.R.grad.data)
            # zero grad
            self.zero_grad()
            # soft thresholding
            with torch.no_grad():
                self.R = PCNet._soft_thresholding(self.R, self.lmda)
            self.R.requires_grad_(True)

    @staticmethod
    def _soft_thresholding(x, alpha):
        return F.relu(x - alpha) - F.relu(-x - alpha)

    def zero_grad(self):
        self.U.grad.data.zero_()
        self.R.grad.data.zero_()

    def forward(self, img_batch):
        # first fit
        self._ista(img_batch)
        # now predict again
        pred = self.R @ self.U
        return pred


