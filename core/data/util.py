'''
Simple Utility Functions for data processing
For gao util see: core.gao_files.classification.util
'''
import os
from ast import literal_eval
import pandas as pd
from core.gao_files.classification.util import get_vocab
from core.gao_files.sequence.util import get_pos2idx_idx2pos
from core.data.gao_data import ExperimentData

# path for constructing ExperimentData object in
# get_vua_seq_bert_POS_performance()
input_data_path = os.path.join("resources", "metaphor-in-context", "data")

bert_vua_seq_preds_path = os.path.join("predictions", "vua_seq_bert_pred.txt")


"""
Functions for computing summary statsitics on dataset
"""

### Classification Task Data
def get_sentence_length(x):
    """
    Compute number of tokens in string
    Split a string on whitespace and return len

    Input:
        x: (str) e.g. a sentence (already tokenized)
    Output:
        (int) the number of tokens in x
    """
    return len(x.split())


def get_classification_data_summary(data_dict):
    """
    FOR CLASSIFICATION DATA
    Compute summary statistics on dataset to replicate
    tables in https://arxiv.org/pdf/1808.09653.pdf

    Input:
        data_dict: (dict) with key: (str) name of dataset, val: (pd.DataFrame)
    Output:
        list: list of list containing summary stats
    """
    # Per the gao-g code, can just split the sentence
    summary = [["", "n", "Perc Metaphor", "Uniq Verb", "Avg Sentence Len"]]

    for data_name, dataset in data_dict.items():
        num_obs = len(dataset)
        perc_metaphor = dataset['label'].mean()
        n_uniq_verb = dataset['verb'].nunique()
        avg_sen_len = dataset['sentence'].apply(get_sentence_length).mean()
        summary.append([data_name, num_obs, perc_metaphor, n_uniq_verb, avg_sen_len])

    return summary

### Sequencing Task Data
def get_sequencing_data_summary(data_dict):
    """
    FOR SEQUENCING DATA
    Compute summary statistics on dataset to replicate
    tables in https://arxiv.org/pdf/1808.09653.pdf

    Input:
        data_dict: (dict) with key: (str) name of dataset, val: (pd.DataFrame)
    Output:
        list: list of list containing summary stats
    """
    # Per the gao-g code, can just split the sentence
    summary = [["", "Uniq Tokens", "n Tokens", "Unique Sentences", "Perc Metaphor"]]

    for data_name, dataset in data_dict.items():
        uniq_sent = len(dataset)
        # line below formats data for gao function
        raw_sentence_data = [[x] for x in dataset["sentence"].tolist()]
        vocab = get_vocab(raw_sentence_data)
        uniq_tokens = len(vocab)
        data_as_list = dataset["label_seq"].apply(literal_eval)
        n_tokens = data_as_list.apply(len).sum()
        n_metaphor = data_as_list.apply(sum).sum()
        summary.append([data_name, uniq_tokens, n_tokens, uniq_sent, n_metaphor/n_tokens])

    return summary


def get_vua_seq_bert_POS_performance():
    """
    For getting the POS performance breakdown
    of the bert predictions for the vua
    sequence labeling task

    returns a dataframe with the POS performance breakdown
    """
    gao_data = ExperimentData(input_data_path)
    gao_data.read_vua_seq_data()
    test_pred_labels = [] 

    f = open(bert_vua_seq_preds_path, "r")
    for labels in f:
        labels_lst_str = labels.split()
        labels_lst = [int(lbl) for lbl in labels_lst_str] 
        test_pred_labels.append(labels_lst)


    # gao_data.vua_seq_formatted_test contains lists of:
    # 0-'fragment' 1-'id' 2-'sentence' 3-'[true labs]'
    # 4-'[POS labs]' 5='sentence' 6-'genre'

    test_true_labels = [literal_eval(sentence[3]) for sentence in \
                        gao_data.vua_seq_formatted_test] # index 3--true labels

    pos_set = set()
    pos_labels = []
    for sentence in gao_data.vua_seq_formatted_test:
        pos_lbl_lst = literal_eval(sentence[4]) # index 4 are the pos labels
        pos_set.update(pos_lbl_lst)
        pos_labels.append(pos_lbl_lst)

    pos2idx, idx2pos = get_pos2idx_idx2pos(pos_set)

    classifications = ["TP", "TN", "FP", "FN"]

    pos_performance_dict = {pos: {classification: 0 for classification in \
                                        classifications} for pos in pos_set}


    for pos_lst, true_labels, pred_labels in zip(pos_labels, test_true_labels,
                                       test_pred_labels):
        for pos, true_lab, pred_lab in zip(pos_lst, true_labels, pred_labels):
            if true_lab == 1: # positive
                if pred_lab == 1: # true pos
                    pos_performance_dict[pos]["TP"] += 1
                elif pred_lab == 0: # false neg
                    pos_performance_dict[pos]["FN"] += 1
                else:
                    raise ValueError('Predicted label neither 1 or 0')
            elif true_lab == 0: # negative
                if pred_lab == 0: # true neg
                    pos_performance_dict[pos]["TN"] += 1
                elif pred_lab == 1: # false pos
                    pos_performance_dict[pos]["FP"] += 1
                else:
                    raise ValueError('Predicted label neither 1 or 0')
            else:
                raise ValueError('True label neither 1 or 0')

    dict_for_df = {pos:[pos_performance_dict[pos][classification] for \
                        classification in classifications] for pos in pos_set}

    pos_df = pd.DataFrame(dict_for_df, index=classifications).T

    pos_df['Precision'] = pos_df['TP'] / (pos_df['TP'] + pos_df['FP'])
    pos_df['Recall'] = pos_df['TP'] / (pos_df['TP'] + pos_df['FN'])
    pos_df['F1'] = pos_df['TP'] / (pos_df['TP'] + 0.5 * (pos_df['FP'] + pos_df['FN']))

    return pos_df

    
            
