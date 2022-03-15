"""
Simple Utility Functions for error analysis
"""

import os
import ast
#from sklearn.metrics import precision_score, recall_score, f1_score
from core.data.gao_data import ExperimentData

PREDS_DIRECTORY = "predictions/"

### INPUT
input_data_path = os.path.join("resources", "metaphor-in-context", "data")

gao_data = ExperimentData(input_data_path)
gao_data.read_vua_seq_data()

def read_in_preds(preds_filename):
    """
    Function to read predictions .txt file into a list of lists of ints
    """
    test_pred_labels = [] 

    f = open(PREDS_DIRECTORY + preds_filename, "r")
    for labels in f:
        labels_lst_str = labels.split()
        labels_lst = [int(lbl) for lbl in labels_lst_str] 
        test_pred_labels.append(labels_lst)

    return test_pred_labels