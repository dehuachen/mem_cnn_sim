import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

class MemCnnSim(nn.Module):

    def __init__(self, margin):
        super(MemCnnSim, self).__init__()

        self.margin = margin

        self.memn2n = MemN2N()
        self.cnn = CNN()

        self.criterion = nn.CosineEmbeddingLoss(margin=0.5)


    def forward(self, utter, memory, cand, flag):

        context = self.memn2n(utter, memory)
        cand_ = self.cnn(cand)

        return context, cand_


    def loss_op(self, context, cand_):
        loss = self.criterion(context, cand_)
        return loss




class MemN2N(nn.Module):

    def __init__(self):
        super(MemN2N, self).__init__()


    def forward(self, utter, memory):
        pass




class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

    def forward(self, cand):
        pass