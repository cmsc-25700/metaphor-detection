'''
Utility Functions
'''

def get_embedding_matrix(word2idx, idx2word, normalization=False):
    """
    assume padding index is 0
    :param word2idx: a dictionary: string --> int, includes <PAD> and <UNK>
    :param idx2word: a dictionary: int --> string, includes <PAD> and <UNK>
    :param normalization:
    :return: an embedding matrix: a nn.Embeddings
    """
    # Load the GloVe vectors into a dictionary, keeping only words in vocab
    embedding_dim = 300
    glove_path = "../resources/glove/glove840B300d.txt"
    glove_vectors = {}
    with open(glove_path) as glove_file:
        for line in tqdm(glove_file, total=get_num_lines(glove_path)):
            split_line = line.rstrip().split()
            word = split_line[0]
            if len(split_line) != (embedding_dim + 1) or word not in word2idx:
                continue
            assert (len(split_line) == embedding_dim + 1)
            vector = np.array([float(x) for x in split_line[1:]], dtype="float32")
            if normalization:
                vector = vector / np.linalg.norm(vector)
            assert len(vector) == embedding_dim
            glove_vectors[word] = vector
     print("Number of pre-trained word vectors loaded: ", len(glove_vectors))

    # Calculate mean and stdev of embeddings
    all_embeddings = np.array(list(glove_vectors.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_stdev = float(np.std(all_embeddings))
    print("Embeddings mean: ", embeddings_mean)
    print("Embeddings stdev: ", embeddings_stdev)

    # Randomly initialize an embedding matrix of (vocab_size, embedding_dim) shape
    # with a similar distribution as the pretrained embeddings for words in vocab.
    vocab_size = len(word2idx)
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean, embeddings_stdev)
    # Go through the embedding matrix and replace the random vector with a
    # pretrained one if available. Start iteration at 2 since 0, 1 are PAD, UNK
    for i in range(2, vocab_size):
        word = idx2word[i]
        if word in glove_vectors:
            embedding_matrix[i] = torch.FloatTensor(glove_vectors[word])
    if normalization:
        for i in range(vocab_size):
            embedding_matrix[i] = embedding_matrix[i] / float(np.linalg.norm(embedding_matrix[i]))
    embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    embeddings.weight = nn.Parameter(embedding_matrix)
    return embeddings

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

def embed_sequence(sequence, verb_idx, word2idx, glove_embeddings, elmo_embeddings, suffix_embeddings):
    """
    Assume that word2idx has 1 mapped to UNK
    Assume that word2idx maps well implicitly with glove_embeddings
    i.e. the idx for each word is the row number for its corresponding embedding
    :param sequence: a single string: a sentence with space
    :param word2idx: a dictionary: string --> int
    :param glove_embeddings: a nn.Embedding with padding idx 0
    :param elmo_embeddings: a h5py file
                    each group_key is a string: a sentence
                    each inside group is an np array (seq_len, 1024 elmo)
    :param suffix_embeddings: a nn.Embedding without padding idx
    :return: a np.array (seq_len, embed_dim=glove+elmo+suffix)
    """
    words = sequence.split()

    # 1. embed the sequence by glove vector
    # Replace words with tokens, and 1 (UNK index) if words not indexed.
    indexed_sequence = [word2idx.get(x, 1) for x in words]
    # glove_part has shape: (seq_len, glove_dim)
    glove_part = glove_embeddings(Variable(torch.LongTensor(indexed_sequence)))

    # 2. embed the sequence by elmo vectors
    if elmo_embeddings != None:
        elmo_part = elmo_embeddings[sequence]
        assert (elmo_part.shape == (len(words), 1024))

    # 3. embed the sequence by suffix indicators i.e. wether it is a verb or not
    indicated_sequence = [0] * len(words)
    indicated_sequence[verb_idx] = 1
    suffix_part = suffix_embeddings(Variable(torch.LongTensor(indicated_sequence)))

    # concatenate three parts: glove+elmo+suffix along axis 1
    assert(glove_part.shape == (len(words), 300))
    assert(suffix_part.shape == (len(words), 50))
    # glove_part and suffix_part are Variables, so we need to use .data
    # otherwise, throws weird ValueError: incorrect dimension, zero-dimension, etc..
    if elmo_embeddings != None:
        result = np.concatenate((glove_part.data, elmo_part), axis=1)
        result = np.concatenate((result, suffix_part.data), axis=1)
    else:
        result = np.concatenate((glove_part.data, suffix_part.data), axis=1)
    return result