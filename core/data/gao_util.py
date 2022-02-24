"""
Code from ge gao et al
https://github.com/gao-g/metaphor-in-context
"""


def get_vocab(raw_dataset):
    """
    return vocab set, and prints out the vocab size
    :param raw_dataset: a list of lists: each inner list is a triple:
                a sentence: string
                a index: int: idx of the focus verb
                a label: int 1 or 0
    :return: a set: the vocabulary in the raw_dataset
    """
    vocab = []
    for example in raw_dataset:
        vocab.extend(example[0].split())
    vocab = set(vocab)
    print("vocab size: ", len(vocab))
    return vocab


def get_word2idx_idx2word(vocab):
    """
    :param vocab: a set of strings: vocabulary
    :return: word2idx: string to an int
             idx2word: int to a string
    """
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    for word in vocab:
        assigned_index = len(word2idx)
        word2idx[word] = assigned_index
        idx2word[assigned_index] = word
    return word2idx, idx2word
