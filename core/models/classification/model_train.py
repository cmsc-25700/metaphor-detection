import time
# import torch.nn as nn
import torch.optim as optim
# from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib
from core.models.classification.LSTM import LSTM_Classifier
from core.gao_files.classification.util import *


matplotlib.use('Agg')  # to avoid the error: _tkinter.TclError: no display name and no $DISPLAY environment variable
# matplotlib.use('tkagg')  # to display the graph on remote server


BATCH_SIZE = 64
EMBEDDING_SIZE = 300 + 50 + 0  # should be 1024 when elmo is incorporated.
HIDDEN_SIZE = 300
NUM_LAYERS = 1
DROPOUT = 0.3
NUM_EPOCHS = 30


### Data Preparation for train and testing raw data####

def data_prep_classification(raw_train_data, raw_test_data, raw_val_data,
                             num_suffix_tag, use_elmo):
    """
    Data Preperation Step for Classification Task
    Input = raw_train_data, raw_test_data, raw_val_data: list of lists. Comes from the data object
    num_suffix_tag = Number of tags. Is set to 2.
    use_elmo = Path for relevant elmo vector.
    """
    vocab = get_vocab(raw_train_data + raw_test_data + raw_val_data)
    word2idx, idx2word = get_word2idx_idx2word(vocab)
    glove_embeddings = get_embedding_matrix(word2idx, idx2word, normalization=False)
    suffix_embeddings = nn.Embedding(num_suffix_tag, 50)
    if use_elmo:
        # Will change this later once elmo is incorporated.
        elmos_train = None
        elmos_val = None
    else:
        elmos_train = None
        elmos_val = None

    ## Embedding ##

    embedded_train = [[embed_sequence(data[1], data[2], word2idx,
                                      glove_embeddings, elmos_train, suffix_embeddings),
                       data[3]] for data in raw_train_data]
    embedded_val = [[embed_sequence(data[1], data[2], word2idx, glove_embeddings,
                                    elmos_val, suffix_embeddings),
                     data[3]] for data in raw_val_data]

    train_dataset = TextDatasetWithGloveElmoSuffix([example[0] for example in embedded_train],
                                                   [example[1] for example in embedded_train])
    val_dataset = TextDatasetWithGloveElmoSuffix([example[0] for example in embedded_val],
                                                 [example[1] for example in embedded_val])

    ### Data Loader for Batching to TextDataset Class###

    train_loaded = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=TextDatasetWithGloveElmoSuffix.collate_fn)
    val_loaded = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE,
                            collate_fn=TextDatasetWithGloveElmoSuffix.collate_fn)

    return train_loaded, val_loaded


### Data Training without Cross Validation ###

def train_data(raw_train_data, raw_test_data, raw_val_data, num_suffix_tag,
               learning_rate=0.01, num_epochs=NUM_EPOCHS, using_GPU=False, use_elmo=False):
    lstm_obj = LSTM_Classifier(embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE,
                               num_layers=NUM_LAYERS, dropout_rate=DROPOUT,
                               using_GPU=using_GPU)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(lstm_obj.parameters(), lr=learning_rate, momentum=0.9)
    if using_GPU:
        lstm_obj = lstm_obj.cuda()
        criterion = criterion.cuda()

    validation_loss = []
    validation_f1 = []

    iterations = 0

    print("#### Data Preparation Step #####")

    train_loaded, val_loaded = data_prep_classification(raw_train_data, raw_test_data,
                                                        raw_val_data, num_suffix_tag, use_elmo)

    print("##### Starting Training#####")
    for epoch in range(num_epochs):
        start = time.time()
        print(f"#### Epoch number {epoch + 1} #### ")
        for (text, lengths, labels) in train_loaded:
            text = Variable(text)
            lengths = Variable(lengths)
            labels = Variable(labels)
            if using_GPU:
                text = text.cuda()
                lengths = lengths.cuda()
                labels = labels.cuda()
            predicted = lstm_obj(text, lengths)
            batch_loss = criterion(predicted, labels)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            iterations += 1
            if iterations % 200 == 0:
                avg_eval_loss, eval_accuracy, precision, \
                recall, f1, fus_f1 = evaluate(val_loaded, lstm_obj, criterion, using_GPU)
                validation_loss.append(avg_eval_loss)
                validation_f1.append(f1)
                step = time.time()
                time_passed = start - step
                start = time.time()
                print(
                    "Time {} Iteration {}. Validation Loss {}. Validation Accuracy {}. \
                Validation Precision {}. Validation Recall {}. Validation F1 {}. \
                Validation class-wise F1 {}." \
                        .format(time_passed, iterations, avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1))


### Model Training with Cross Validation ###

def train_kfold(raw_data, num_suffix_tag, learning_rate=0.01,
                num_epochs=NUM_EPOCHS, using_GPU=False, use_elmo=False):
    vocab = get_vocab(raw_data)
    word2idx, idx2word = get_word2idx_idx2word(vocab)
    glove_embeddings = get_embedding_matrix(word2idx, idx2word, normalization=False)
    suffix_embeddings = nn.Embedding(num_suffix_tag, 50)

    if use_elmo:
        elmo = None  # will be changed once elmo is incorporated
    else:
        elmo = None

    # indexes for labels
    embedded_data = [[embed_sequence(data[3], data[4], word2idx,
                                     glove_embeddings, elmo, suffix_embeddings),
                      data[5]] for data in raw_data]

    sentences = [data[0] for data in embedded_data]
    labels = [data[1] for data in embedded_data]

    # ten_folds is a list of 10 tuples, each tuple is (list_of_embedded_sentences, list_of_corresponding_labels)
    k_fold = [(sentences[i * 65: (i + 1) * 65], labels[i * 65: (i + 1) * 65]) for i in range(10)]
    optimal_f1s = []
    accuracies = []
    precisions = []
    recalls = []


    print("### BATCHING DATA ####")

    for i in range(10):
        training_sentences = []
        training_labels = []
        for j in range(10):
            if j != i:
                training_sentences.extend(k_fold[j][0])
                training_labels.extend(k_fold[j][1])
        train = TextDatasetWithGloveElmoSuffix(training_sentences, training_labels)
        validation = TextDatasetWithGloveElmoSuffix(k_fold[i][0], k_fold[i][1])

        # Data-related hyper parameters
        batch_size = 10
        # Set up a DataLoader for the training, validation, and test dataset
        train_dataloader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True,
                                      collate_fn=TextDatasetWithGloveElmoSuffix.collate_fn)
        val_dataloader = DataLoader(dataset=validation, batch_size=batch_size, shuffle=True,
                                    collate_fn=TextDatasetWithGloveElmoSuffix.collate_fn)
        print("### Training Model ###")
        #### MODEL TRAINING STEP ####
        lstm_obj = LSTM_Classifier(embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE,
                                   num_layers=NUM_LAYERS, dropout_rate=DROPOUT,
                                   using_GPU=using_GPU)
        # Move the model to the GPU if available
        if using_GPU:
            lstm_obj = lstm_obj.cuda()
        # Set up criterion for calculating loss
        nll_criterion = nn.NLLLoss()
        # Set up an optimizer for updating the parameters of the rnn_clf
        lstm_optimizer = optim.SGD(lstm_obj.parameters(), lr=learning_rate, momentum=0.9)

        training_loss = []
        val_loss = []
        training_f1 = []
        val_f1 = []

        
        # A counter for the number of gradient updates
        iterations = 0
        for epoch in range(NUM_EPOCHS):
            print(" ###### Starting EPOCH {} ######".format(epoch + 1))
            for (example_text, example_lengths, labels) in train_dataloader:
                example_text = Variable(example_text)
                example_lengths = Variable(example_lengths)
                labels = Variable(labels)
                if using_GPU:
                    example_text = example_text.cuda()
                    example_lengths = example_lengths.cuda()
                    labels = labels.cuda()

                predicted = lstm_obj(example_text, example_lengths)
                batch_loss = nll_criterion(predicted, labels)
                lstm_obj.zero_grad()
                batch_loss.backward()
                lstm_optimizer.step()
                iterations += 1
                # Calculate validation and training set loss and accuracy every 200 gradient updates
                if iterations % 200 == 0:
                    avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1 = evaluate(val_dataloader, lstm_obj,
                                                                                           nll_criterion, using_GPU)
                    val_loss.append(avg_eval_loss)
                    val_f1.append(f1)
                    print(
                        "####Iteration {}. Validation Loss {}. Validation Accuracy {}. Validation Precision {}. "
                        "Validation Recall {}. Validation F1 {}. Validation class-wise F1 {}." \
                            .format(iterations, avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1))
                    # filename = '../models/LSTMSuffixElmoAtt_???_all_iter_' + str(num_iter) + '.pt'
                    # torch.save(lstm_obj, filename)
                    avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1 = evaluate(train_dataloader, lstm_obj,
                                                                                           nll_criterion, using_GPU)
                    accuracies.append(eval_accuracy.item())
                    if str(precision) != "nan":
                      precisions.append(precision)
                    recalls.append(recall)
                    if str(max(val_f1)) != "nan":
                        training_loss.append(avg_eval_loss)
                        training_f1.append(f1)
                        print(
                            "#####Iteration {}. Training Loss {}. Training Accuracy {}. Training Precision {}. Training "
                            "Recall {}. Training F1 {}. Training class-wise F1 {}." \
                                .format(iterations, avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1))
        print("###Training done for fold {}###".format(i))

        if str(max(val_f1)) != "nan":
            optimal_f1s.append(max(val_f1))

        print('F1 on MOH-X by 10-fold = ', optimal_f1s)
        print('F1 on MOH-X = ', np.mean(np.array(optimal_f1s)))
        print('precisions on MOH-X = ', np.mean(np.array(precisions)))
        print('recalls on MOH-X = ', np.mean(np.array(recalls)))
        print('accuracies on MOH-X = ', np.mean(np.array(accuracies)))

        
