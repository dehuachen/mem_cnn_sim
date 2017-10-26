import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable as V


class CNN(nn.Module):

    def __init__(self, param, embedding=None):
        super(CNN, self).__init__()

        self.param = param

        self.vocab_size = param['cand_vocab_size']
        self.embedding_size = param['embedding_size']
        self.num_filters = param['num_filters']

        if embedding:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        # a possible modification is apply n convolutions
        self.cnn1 = nn.Conv2d(1, self.num_filters, (1, self.embedding_size))
        self.cnn2 = nn.Conv2d(1, self.num_filters, (2, self.embedding_size))
        self.cnn3 = nn.Conv2d(1, self.num_filters, (3, self.embedding_size))

        self.l1 = nn.Linear(self.num_filters * 3, self.embedding_size)
        self.l2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.l3 = nn.Linear(self.embedding_size, self.embedding_size)

        self.weights_init()

    def forward(self, cand):
        # embedding
        cand_ = self.embedding(cand)  # (num_cand, cand_size, embed_size)

        cand_ = cand_.unsqueeze(1)  # (num_cand, 1, cand_size, embed_size)


        cand_1 = F.relu(self.cnn1(cand_)).squeeze(3)  # (num_cand, num_filters, Width, 1)
        cand_1 = F.max_pool1d(cand_1, cand_1.size(2)).squeeze(2)  # (num_cand, num_filters, 1)
        cand_2 = F.relu(self.cnn2(cand_)).squeeze(3)  # (num_cand, num_filters, Width, 1)
        cand_2 = F.max_pool1d(cand_2, cand_2.size(2)).squeeze(2)  # (num_cand, num_filters, 1)
        cand_3 = F.relu(self.cnn3(cand_)).squeeze(3)  # (num_cand, num_filters, Width, 1)
        cand_3 = F.max_pool1d(cand_3, cand_3.size(2)).squeeze(2)  # (num_cand, num_filters, 1)

        cand_ = torch.cat([cand_1, cand_2, cand_3], 1)

        cand_ = F.relu(self.l1(cand_))  # (num_cand, embed_size)
        cand_ = F.relu(self.l2(cand_))  # (num_cand, embed_size)
        cand_ = F.relu(self.l3(cand_))  # (num_cand, embed_size)

        return cand_

    def weights_init(self):
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) \
                    or isinstance(m, nn.Embedding) \
                    or isinstance(m, nn.Linear):
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
