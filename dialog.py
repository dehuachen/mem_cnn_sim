from data_utils import load_dialog_task, vectorize_data, load_candidates, vectorize_candidates, tokenize
from six.moves import range, reduce
from itertools import chain
import numpy as np
from sklearn import metrics


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

if __name__ == '__main__':
    data_dir = "data/dialog-bAbI-tasks/"
    task_id = 6

    candid2indx, \
    indx2candid, \
    candidates_vec, \
    word_idx, \
    sentence_size, \
    candidate_sentence_size, \
    memory_size, \
    vocab_size, \
    train_data, test_data, val_data = init(data_dir, task_id)

    print()