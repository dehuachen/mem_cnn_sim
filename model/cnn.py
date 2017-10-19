import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, param):
        super(CNN, self).__init__()

        self.param = param

    def forward(self, cand):
        pass