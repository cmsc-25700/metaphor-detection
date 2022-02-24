from LSTM import LSTM_Classifier
from core.data.util import *
from core.data.gao_data import *


def data_prep_classification(data_object, num_suffix_tag, use_elmo = False):
    '''
    Data Preperation Step for Classification Task
    '''
    vocab = get_vocab(data_object)
    word2idx, idx2word = get_word2idx_idx2word(vocab)
    glove_embeddings = get_embedding_matrix(word2idx, idx2word, normalization=False)
    suffix_embeddings = nn.Embedding(num_suffix_tag, 50)
    if use_elmo:
        #elmos = h5py.File('../elmo/MOH-X_cleaned.hdf5', 'r')
    else:
        elmos = None
    return glove_embeddings, elmos, suffix_embeddings

