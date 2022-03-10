"""
Utils for VUA CLS Model
Compute metrics using code adapted from sequence util
"""
import csv
import torch
import numpy as np
from torch.autograd import Variable


# adapted from sequence util
def write_predictions_vua_cls(raw_dataset, evaluation_dataloader, model, using_GPU, rawdata_filename):
    """
    Evaluate the model on the given evaluation_dataloader

    :param raw_dataset
    :param evaluation_dataloader:
    :param model:
    :param using_GPU: a boolean
    :return: a list of
    """
    # Set model to eval mode, which turns off dropout.
    model.eval()

    predictions = []
    num_correct = 0
    total_examples = 0
    total_eval_loss = 0
    confusion_matrix = np.zeros((2, 2))
    for (example_text, example_lengths, labels) in evaluation_dataloader:
        eval_text = Variable(example_text, volatile=True)
        eval_lengths = Variable(example_lengths, volatile=True)
        eval_labels = Variable(labels, volatile=True)
        if using_GPU:
            eval_text = eval_text.cuda()
            eval_lengths = eval_lengths.cuda()
            eval_labels = eval_labels.cuda()

        predicted = model(eval_text, eval_lengths)
        # get 0 or 1 predictions
        # predicted_labels: (batch_size, seq_len)
        # print("predicted", predicted.data )
        _, predicted_labels = torch.max(predicted.data, 1)
        predictions.extend(predicted_labels)

        total_examples += eval_labels.size(0)
        num_correct += torch.sum(predicted_labels == eval_labels.data)
        for i in range(eval_labels.size(0)):
            confusion_matrix[int(predicted_labels[i]), eval_labels.data[i]] += 1


    # Set the model back to train mode, which activates dropout again.
    model.train()
    assert (len(predictions) == len(raw_dataset))

    ###
    accuracy = 100 * num_correct / total_examples
    average_eval_loss = total_eval_loss / total_examples

    precision = 100 * confusion_matrix[0, 0] / np.sum(confusion_matrix[0])
    recall = 100 * confusion_matrix[0, 0] / np.sum(confusion_matrix[:, 0])
    lit_f1 = 2 * precision * recall / (precision + recall)

    precision = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[1])
    recall = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[:, 1])
    met_f1 = 2 * precision * recall / (precision + recall)
    class_wise_f1 = (met_f1 + lit_f1) / 2
    print(confusion_matrix)

    # Set the model back to train mode, which activates dropout again.
    model.train()

    # read original data
    data = []
    with open(rawdata_filename, encoding='latin-1') as f:
        lines = csv.reader(f)
        for line in lines:
            data.append(line)

    # append predictions to the original data
    data[0].append('prediction')
    for i in range(len(predictions)):
        data[i + 1].append(predictions[i])

    return data


# adapted from sequence util
def get_performance_VUAverb_test(data_path, seq_test_pred):
    """
    Similar treatment as get_performance_VUAverb_val
    Read the VUA-verb test data, and the VUA-sequence test data.
    Extract the predictions for VUA-verb test data from the VUA-sequence test data.
    Prints the performance of LSTM sequence model on VUA-verb test set based on genre
    Prints the performance of LSTM sequence model on VUA-verb test set regardless of genre

    :return: the averaged performance across genre and performance on verb test
    regardless of genre
    """
    # get genre
    ID2genre = {}
    with open(data_path + 'VUAsequence/VUA_seq_formatted_test.csv', encoding='latin-1') as f:
      lines = csv.reader(f)
      next(lines)
      for line in lines:
        ID2genre[(line[0], line[1])] = line[6]

    predictions = []
    genres = ['news', 'fiction', 'academic', 'conversation']
    confusion_matrix = np.zeros((4, 2, 2))
    confusion_matrix_alt = np.zeros((2, 2))

    for row in seq_test_pred[1:]: # first row is header
        ID = (row[0], row[1])
        label = int(row[5])
        pred = row[6].item()
        predictions.append(pred)
        genre = ID2genre[ID]
        genre_idx = genres.index(genre)
        confusion_matrix[genre_idx][pred][label] += 1
        confusion_matrix_alt[pred, label] += 1
    assert (np.sum(confusion_matrix) == len(seq_test_pred)-1)
    print("confusion matrix by genre\n")
    print(confusion_matrix)
    print("\nconfusion matrix all\n")
    print(confusion_matrix_alt)

    print('Tagging model performance on test-verb: genre')
    avg_performance = []
    for i in range(len(genres)):
        precision = 100 * confusion_matrix[i, 1, 1] / np.sum(confusion_matrix[i, 1])
        recall = 100 * confusion_matrix[i, 1, 1] / np.sum(confusion_matrix[i, :, 1])
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = 100 * (confusion_matrix[i, 1, 1] + confusion_matrix[i, 0, 0]) / np.sum(confusion_matrix[i])
        print(genres[i], 'Precision, Recall, F1, Accuracy: ', precision, recall, f1, accuracy)
        avg_performance.append([precision, recall, f1, accuracy])
    avg_performance = np.array(avg_performance)
    macro_avg_performance = avg_performance.mean(0)

    print('Tagging model performance on test-verb: regardless of genre')
    confusion_matrix = confusion_matrix.sum(axis=0)
    precision = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[1])
    recall = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[:, 1])
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = 100 * (confusion_matrix[1, 1] + confusion_matrix[0, 0]) / np.sum(confusion_matrix)
    overall_performance = np.array([precision, recall, f1, accuracy])
    print('Precision, Recall, F1, Accuracy: ', precision, recall, f1, accuracy)
    # print(confusion_matrix)

    return macro_avg_performance, overall_performance
