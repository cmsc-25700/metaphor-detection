'''
Simple Utility Functions for data processing
For gao util see: core.gao_files.classification.util
'''


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
    Compute summary statistics on dataset to replicate
    tables in https://arxiv.org/pdf/1808.09653.pdf

    Input:
        data_dict: (dict) with key: (str) name of dataset, val: (pd.DataFrame)
    Output:
        list: list of list containig summar stats
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
