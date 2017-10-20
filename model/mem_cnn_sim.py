import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch import optim

from model.cnn import CNN
from model.mem_n2n import MemN2N


class MemCnnSim(nn.Module):

    def __init__(self, param, margin=0.5):
        super(MemCnnSim, self).__init__()

        self.param = param

        self.max_grad_norm = param['max_grad_norm']

        self.memn2n = MemN2N(param)
        self.cnn = CNN(param)

        self.criterion = nn.CosineEmbeddingLoss(margin=margin)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, utter, memory, cand):

        cand_size = cand.size(0)

        context = self.memn2n(utter, memory)
        context = context.repeat(cand_size, 1)

        cand_ = self.cnn(cand)

        return context, cand_

    def predict(self, context, cand_):
        sims = F.cosine_similarity(context, cand_)
        _, pred = torch.max(sims, 0)

        return pred

    def loss_op(self, context, cand_, flag):
        loss = self.criterion(context, cand_, flag)
        return loss

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.parameters(), self.max_grad_norm)
        self.optimizer.step()


# if __name__ == '__main__':
#     param = {
#             'hops': 3,
#             "vocab_size": 20,
#             "embedding_size": 20,
#             'num_filters': 20,
#             "cand_vocab_size": 20
#              }
#
#     mem = MemCnnSim(param)
#
#     for i in range(20):
#         # print(param)
#         memory = torch.ones(1, 3, 10).long()
#         utter = torch.ones(1, 10).long()
#         cand = torch.ones(30, 10).long()
#         flag = torch.ones(30, 1).long()
#
#         context, cand = mem(V(utter), V(memory), V(cand))
#         loss = mem.loss_op(context, cand, V(flag))
#         print(loss)
#
#         mem.optimize(loss)