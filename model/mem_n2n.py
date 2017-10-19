import torch.nn as nn
import torch
from torch.autograd import Variable as V


class MemN2N(nn.Module):

    def __init__(self, param):
        super(MemN2N, self).__init__()

        self.param = param

        self.hops = self.param['hops']
        self.vocab_size = param['vocab_size']
        self.embedding_size = param['embedding_size']

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.linear = nn.Linear(self.embedding_size, self.embedding_size)

        self.softmax = nn.Softmax()

        self.weights_init()

    def forward(self, utter, memory):
        # embed query
        utter_emb = self.embedding(utter)
        utter_emb_sum = torch.sum(utter_emb, 1)
        contexts = [utter_emb_sum]

        for _ in range(self.hops):
            memory_emb = self.embed_3d(memory, self.embedding)
            memory_emb_sum = torch.sum(memory_emb, 2)

            # get attention
            context_temp = torch.transpose(torch.unsqueeze(contexts[-1], -1), 1, 2)
            attention = torch.sum(memory_emb_sum * context_temp, 2)
            attention = self.softmax(attention)

            attention = torch.unsqueeze(attention, -1)
            attn_stories = torch.sum(attention * memory_emb_sum, 1)

            new_context = self.linear(contexts[-1]) + attn_stories

            contexts.append(new_context)

        return new_context

    def weights_init(self):
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                m.weight.data.normal_(0, 0.1)
            if isinstance(m, nn.Embedding):
                m.weight.data[0].zero_()

    def embed_3d(self, to_emb, embedding):
        num_elem = to_emb.size(1)
        elem_size = to_emb.size(2)

        to_emb = to_emb.view(-1, num_elem * elem_size)
        out = embedding(to_emb)
        out = out.view(-1, num_elem, elem_size, self.embedding_size)

        return out
#
# if __name__ == '__main__':
#     param = {'hops': 3, "vocab_size": 20, "embedding_size": 20}
#     mem = MemN2N(param)
#
#     # print(param)
#     memory = torch.ones(20, 3, 10).long()
#     utter = torch.ones(20, 10).long()
#
#     print(mem(V(utter), V(memory)))
