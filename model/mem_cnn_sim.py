import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

from model.cnn import CNN
from model.mem_n2n import MemN2N


class MemCnnSim(nn.Module):

    def __init__(self, param, margin=0.5):
        super(MemCnnSim, self).__init__()

        self.param = param

        self.memn2n = MemN2N(param)
        self.cnn = CNN(param)

        self.criterion = nn.CosineEmbeddingLoss(margin=margin)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, utter, memory, cand, flag):

        context = self.memn2n(utter, memory)
        cand_ = self.cnn(cand)

        return context, cand_

    def loss_op(self, context, cand_):
        loss = self.criterion(context, cand_)
        return loss


