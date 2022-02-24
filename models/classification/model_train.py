
from LSTM import LSTM_Classifier
from core.data.util import *
import torch.optim as optim
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn

BATCH_SIZE = 64
EMBEDDING_SIZE = 300 + 1024 + 50
HIDDEN_SIZE = 300
NUM_LAYERS = 1
DROPOUT = 0.3
NUM_EPOCHS = 20


def data_prep_classification(raw_train_data, raw_test_data, raw_val_data,
                             num_suffix_tag, use_elmo):
    '''
    Data Preperation Step for Classification Task
    '''
    vocab = get_vocab(raw_train_data + raw_test_data + raw_val_data)
    word2idx, idx2word = get_word2idx_idx2word(vocab)
    glove_embeddings = get_embedding_matrix(word2idx, idx2word, normalization=False)
    suffix_embeddings = nn.Embedding(num_suffix_tag, 50)
    if use_elmo:
        ## Will change this later once elmo is incorporated.
        elmos_train = None
        elmos_val = None
    else:
        elmos_train = None
        elmos_val = None
    
    ### Data Embedding Step ###

    embedded_train = [[embed_sequence(data[0], data[1], word2idx,
                                      glove_embeddings, elmos_train, suffix_embeddings), data[2]]
                      for data in raw_train_data]
    embedded_val = [[embed_sequence(data[0], data[1], word2idx, glove_embeddings,
                                     elmos_val, suffix_embeddings), 
                                     data[2]] for data in raw_val_data]
    
    ### Data Loader for Batching to TextDataset Class###

    train_loaded = DataLoader(dataset=embedded_train, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=TextDatasetWithGloveElmoSuffix.collate_fn)
    val_loaded = DataLoader(dataset=embedded_val, batch_size=BATCH_SIZE,
                                collate_fn=TextDatasetWithGloveElmoSuffix.collate_fn)

    return train_loaded, val_loaded


def train_data(raw_train_data, raw_test_data, raw_val_data, num_suffix_tag, learning_rate = 0.01, 
               num_epochs = NUM_EPOCHS, using_GPU = False, use_elmo = False):

    lstm_obj = LSTM_Classifier(embedding_size = EMBEDDING_SIZE, hidden_size = HIDDEN_SIZE, 
                               num_layers = NUM_LAYERS, dropout_rate = DROPOUT)
    if using_GPU:
        lstm_obj = lstm_obj.cuda()
    
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(lstm_obj.parameters(), lr = learning_rate, momentum = 0.9)

    training_loss = []
    training_f1 = []
    validation_loss = []
    validation_f1 = []

    iterations = 0

    print("#### Data Preperation Step #####")

    train_loaded, val_loaded = data_prep_classification(raw_train_data, raw_test_data, 
                                                        raw_val_data, num_suffix_tag, use_elmo)


    for epoch in range(num_epochs):
        start = time.time()
        print(f"#### Epoch number {epoch + 1} #### ")
        for (text, lengths, labels) in train_loaded:
            text = Variable(text)
            lengths = Variable(lengths)
            labels = Variable(labels)
            if using_GPU:
                text.cuda()
                lengths.cuda()
                labels.cuda()
            predicted = lstm_obj(text, lengths)
            batch_loss = criterion(labels, lengths)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            iterations += 1
            if iterations % 200 == 0:
                avg_eval_loss, eval_accuracy, \
                precision, recall, f1, fus_f1 = evaluate(val_loaded, lstm_obj, criterion, using_GPU)
                validation_loss.append(avg_eval_loss)
                validation_f1.append(f1)
                step = time.time()
                time_passed = start - step
                start = time.time()
                print(
                "Time {} Iteration {}. Validation Loss {}. Validation Accuracy {}. \
                Validation Precision {}. Validation Recall {}. Validation F1 {}. \
                Validation class-wise F1 {}."\
                .format(time_passed, iterations, avg_eval_loss, eval_accuracy,
                        precision, recall, f1, fus_f1))
                


    



