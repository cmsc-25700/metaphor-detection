"""
Utility Functions for error analysis
"""
import pandas as pd
import os
import ast
#from sklearn.metrics import precision_score, recall_score, f1_score
from core.data.gao_data import ExperimentData

PREDS_DIRECTORY = "predictions/"

### path for constructing ExperimentData object
input_data_path = os.path.join("resources", "metaphor-in-context", "data")


def check_for_fp(row):
    """
    Helper function for checking error analysis dataframe for 
    false positives in a sentence
    """
    true_labs = ast.literal_eval(row['true_labels'])
    pred_labs = ast.literal_eval(row['pred_labels'])
    for ind,true_lab in enumerate(true_labs):
        if true_lab == 0 and pred_labs[ind] == 1:
          return True
    return False


def check_for_fn(row):
    """
    Helper function for checking error analysis dataframe for 
    false negatives in a sentence
    """
    true_labs = ast.literal_eval(row['true_labels'])
    pred_labs = ast.literal_eval(row['pred_labels'])
    for ind,true_lab in enumerate(true_labs):
        if true_lab == 1 and pred_labs[ind] == 0:
          return True
    return False


def display_errors(row):
    """
    Helper function to be used with the error_analysis dataframe to 
    helpfully display prediction errors
    """
    words = row['Text'].split()
    true_labs = ast.literal_eval(row['true_labels'])
    pred_labs = ast.literal_eval(row['pred_labels'])
    fp_indices = []
    fn_indices = []
    tp_indices = []
    for ind,true_lab in enumerate(true_labs):
        if true_lab == 0 and pred_labs[ind] == 1:
            fp_indices.append(ind)
        elif true_lab == 1 and pred_labs[ind] == 0:
            fn_indices.append(ind)
        elif true_lab == 1 and pred_labs[ind == 1:]:
            tp_indices.append(ind)
    print(f"""
          Sentence: {row['Text']},
          False positive words: {[words[i] for i in fp_indices]},
          False negative words: {[words[i] for i in fn_indices]},
          True positive words: {[words[i] for i in tp_indices]}
          """)


def construct_vua_err_analydis_df(preds_filename, gao_data_obj):
    """
    Function to read predictions .txt file in and merge it with
    labels and original sentence, and information about errors
    into a pandas dataframe

    Inputs
        preds_filename: str: name of the file containing the predictions
        gao_data_obj: str: name of the ExperimentData attr corresponding
                    to predictions data (e.g.'vua_seq_formatted_test')

    Outputs
        err_analysis_df: DataFrame: df with the following columns:
            ['Text', 'index', 'true_labels', 'pred_labels', 'correctly_labeled',
                'false_pos', 'false_neg']
    """
    gao_data = ExperimentData(input_data_path)
    gao_data.read_vua_seq_data()
    test_pred_labels = [] 

    f = open(PREDS_DIRECTORY + preds_filename, "r")
    for labels in f:
        labels_lst_str = labels.split()
        labels_lst = [int(lbl) for lbl in labels_lst_str] 
        test_pred_labels.append(labels_lst)

    test_true_labels = [ast.literal_eval(sentence[3]) for sentence in \
                        getattr(gao_data, gao_data_obj)] # index 3 is the true labels

    for true_labs, pred_labs in zip(test_true_labels, 
                                    test_pred_labels):
        assert len(true_labs) == len(pred_labs), \
            "Unequal lengths of true labels and prediction labels"

    sentence_str = [] #list of sentence strings
    for sample in getattr(gao_data, gao_data_obj):
        sentence_str.append(sample[2]) #index 2 is the sentence text

    err_analysis_df = pd.DataFrame(sentence_str, columns = ["Text"])
    err_analysis_df['index'] = err_analysis_df.index
    err_analysis_df['true_labels'] = [str(lbls) for lbls in test_true_labels]
    err_analysis_df['pred_labels'] = [str(lbls) for lbls in test_pred_labels]
    err_analysis_df["correctly_labeled"] = err_analysis_df['true_labels'] == err_analysis_df['pred_labels']
    err_analysis_df['false_pos'] = err_analysis_df.apply(check_for_fp, axis = 1)
    err_analysis_df['false_neg'] = err_analysis_df.apply(check_for_fn, axis = 1)

    return err_analysis_df