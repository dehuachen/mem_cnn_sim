import torch

from data_utils import load_dialog_task, vectorize_data, load_candidates, vectorize_candidates, tokenize
from six.moves import range, reduce
from itertools import chain
import numpy as np
from sklearn import metrics
from torch.autograd import Variable as V

from model.mem_cnn_sim import MemCnnSim


def init(data_dir, task_id, OOV=False):
    # load candidates
    candidates, candid2indx = load_candidates(
        data_dir, task_id)
    n_cand = len(candidates)
    print("Candidate Size", n_cand)
    indx2candid = dict(
        (candid2indx[key], key) for key in candid2indx)

    # load task data
    train_data, test_data, val_data = load_dialog_task(
        data_dir, task_id, candid2indx, OOV)
    data = train_data + test_data + val_data

    # build parameters
    word_idx, sentence_size, \
    candidate_sentence_size, memory_size, \
    vocab_size = build_vocab(data, candidates)

    # Variable(torch.from_numpy(candidates_vec)).view(len(candidates), sentence_size)
    candidates_vec = vectorize_candidates(
        candidates, word_idx, candidate_sentence_size)

    return candid2indx, \
           indx2candid, \
           candidates_vec, \
           word_idx, \
           sentence_size, \
           candidate_sentence_size, \
           memory_size, \
           vocab_size, \
           train_data, test_data, val_data

def build_vocab(data, candidates, memory_size=50):
    vocab = reduce(lambda x, y: x | y, (set(
        list(chain.from_iterable(s)) + q) for s, q, a in data))
    vocab |= reduce(lambda x, y: x | y, (set(candidate)
                                         for candidate in candidates))
    vocab = sorted(vocab)
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean([len(s) for s, _, _ in data]))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    candidate_sentence_size = max(map(len, candidates))
    query_size = max(map(len, (q for _, q, _ in data)))
    memory_size = min(memory_size, max_story_size)
    vocab_size = len(word_idx) + 1  # +1 for nil word
    sentence_size = max(query_size, sentence_size)  # for the position
    # params
    print("vocab size:", vocab_size)
    print("Longest sentence length", sentence_size)
    print("Longest candidate sentence length", candidate_sentence_size)
    print("Longest story length", max_story_size)
    print("Average story length", mean_story_size)

    return word_idx, \
           sentence_size, \
           candidate_sentence_size, \
           memory_size, \
           vocab_size

def eval(utter_batch, memory_batch, answer__batch, dialog_idx, mem_cnn_sim):
    mem_cnn_sim.eval()

    total_loss = []
    preds = []
    for start, end in dialog_idx:

        loss_per_diaglo = []

        for j in range(start, end + 1):

            memory = V(torch.from_numpy(memory_batch[j])).unsqueeze(0)
            utter = V(torch.from_numpy(utter_batch[j])).unsqueeze(0)


            context, cand_ = mem_cnn_sim(utter, memory, cands_tensor)
            pred = mem_cnn_sim.predict(context, cand_)
            preds.append(pred.data[0])

            loss_per_diaglo.append(loss.data[0])

        total_loss += loss_per_diaglo

    accuracy = metrics.accuracy_score(answer__batch[:len(preds)], preds)
    print('Validation accuracy: {}'.format(accuracy))
    print('Validation loss: {}'.format(sum(total_loss)))

if __name__ == '__main__':
    data_dir = "data/dialog-bAbI-tasks/"
    task_id = 6
    epochs = 10
    # cuda = torch.

    candid2indx, \
    indx2candid, \
    candidates_vec, \
    word_idx, \
    sentence_size, \
    candidate_sentence_size, \
    memory_size, \
    vocab_size, \
    train_data, test_data, val_data = init(data_dir, task_id)

    trainS, trainQ, trainA, dialog_idx = vectorize_data(
        train_data, word_idx, sentence_size, memory_size)
    valS, valQ, valA, dialog_idx_val = vectorize_data(
        val_data, word_idx, sentence_size, memory_size)
    n_train = len(trainS)
    n_val = len(valS)

    print("Training Size", n_train)
    print("Validation Size", n_val)

    param = {
            'hops': 1,
            "vocab_size": vocab_size,
            "embedding_size": 80,
            'num_filters': 20,
            "cand_vocab_size": vocab_size
             }

    mem_cnn_sim = MemCnnSim(param)

    best_validation_accuracy = 0
    time = []

    cands_tensor = V(torch.from_numpy(candidates_vec))

    num_cand = cands_tensor.size(0)
    num_dialog = len(dialog_idx)

    for i in range(1, epochs+1):

        mem_cnn_sim.train()
        for i, (start, end) in enumerate(dialog_idx):

            loss_per_diaglo = []

            for j in range(start, end+1):

                ans = trainA[j]

                memory = V(torch.from_numpy(trainS[j])).unsqueeze(0)
                utter = V(torch.from_numpy(trainQ[j])).unsqueeze(0)

                flag = -1 * torch.ones(num_cand)
                flag[ans] = 1

                flag = V(flag)

                context, cand_ = mem_cnn_sim(utter, memory, cands_tensor)
                loss = mem_cnn_sim.loss_op(context, cand_, flag)
                mem_cnn_sim.optimize(loss)

                loss_per_diaglo.append(loss.data[0])
            # print('loss: {}'.format(sum(loss_per_diaglo)/len(loss_per_diaglo)))
            print('[{}/{}]\r'.format(i+1, num_dialog))
        eval(valQ, valS, valA, dialog_idx_val, mem_cnn_sim)