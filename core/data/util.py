'''
Simple Utility Functions for data processing
For gao util see: core.gao_files.classification.util
'''

from ast import literal_eval
from core.gao_files.classification.util import get_vocab

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
