import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable as V


class CNN(nn.Module):

    def __init__(self, param):
        super(CNN, self).__init__()

        self.param = param

        self.vocab_size = param['cand_vocab_size']
        self.embedding_size = param['embedding_size']
        self.num_filters = param['num_filters']

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        # a possible modification is apply n convolutions
        self.cnn = nn.Conv2d(1, self.num_filters, (2, self.embedding_size))

        self.linear = nn.Linear(self.num_filters, self.embedding_size)

        self.weights_init()

    def forward(self, cand):
        # embedding
        cand_ = self.embedding(cand)  # (num_cand, cand_size, embed_size)

        cand_ = cand_.unsqueeze(1)  # (num_cand, 1, cand_size, embed_size)

        cand_ = F.relu(self.cnn(cand_))  # (num_cand, num_filters, Width, 1)
        cand_ = cand_.squeeze(3)  # (num_cand, num_filters, Width)

        cand_ = F.max_pool1d(cand_, cand_.size(2))  # (num_cand, num_filters, 1)
        cand_ = cand_.squeeze(2)  # (num_cand, num_filters)

        cand_ = self.linear(cand_)  # (num_cand, embed_size)

        return cand_

    def weights_init(self):
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Embedding):
                m.weight.data.normal_(0, 0.05)
            if isinstance(m, nn.Embedding):
                m.weight.data[0].zero_()

# if __name__ == '__main__':
#     param = {'num_filters': 20, "cand_vocab_size": 20, "embedding_size": 20}
#     mem = CNN(param)
#
#     # print(param)
#     utter = torch.ones(20, 10).long()
#
#     print(mem(V(utter)))
