"""
Class for organizing gao et all datasets
"""

import os
import csv
import pandas as pd

MOH = "MOH"
MOH_X = "MOH-X"
TROFI = "TroFi"
TROFI_X = "TroFi-X"
VUA = "VUA"
VUA_SEQ = "VUA_seq"

##### MOH-X and TroFi datasets
# 10 fold cross-validation on the

##### VUA dataset
# use the original training and test split
#  set aside 10% of the training set as a development set.


class GetExperimentData:
    """
    Attributes:
        there is an attribute for every file in the gao et all repo
        every attribute is initialized to None
        call public read functions to read in file by dataset

        e.g.read_moh_data reads all MOH datasets to respective attributes
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

        # MOH
        self.moh_formatted = None
        self.moh_formatted_test = None
        self.moh_formatted_train = None
        self.moh_formatted_val = None

        # MOH X
        self.moh_x_formatted_svo = None
        self.moh_x_formatted_svo_cleaned = None

        # TROFI
        self.trofi_formatted_all = None

        # TROFI-X
        self.trofi_x_formatted_svo = None

        # VUA
        self.vua_formatted = None
        self.vua_formatted_test = None
        self.vua_formatted_train = None
        self.vua_formatted_train_augmented = None
        self.vua_formatted_train_noVAL = None
        self.vua_formatted_val = None

        # VUA sequence
        self.vua_seq_formatted_test = None
        self.vua_seq_formatted_train = None
        self.vua_seq_formatted_val = None

    def read_moh_data(self, to_pandas=False):
        """
        Read data into public instance variables

        :param to_pandas: if true, data should be DataFrame, else list
        :return: NA
        """
        self.moh_formatted, nrow = self.read_experiment_file(MOH, "{}_formatted.csv".format(MOH), to_pandas)
        print(f"{MOH} formatted nrow: {nrow}")

        self.moh_formatted_test, nrow = self.read_experiment_file(MOH, "{}_formatted_test.csv".format(MOH), to_pandas)
        print(f"{MOH} test nrow: {nrow}")

        self.moh_formatted_train, nrow = self.read_experiment_file(MOH, "{}_formatted_train.csv".format(MOH), to_pandas)
        print(f"{MOH} train nrow: {nrow}")

        self.moh_formatted_val, nrow = self.read_experiment_file(MOH, "{}_formatted_val.csv".format(MOH), to_pandas)
        print(f"{MOH} val nrow: {nrow}")

    def read_moh_x_data(self, pandas=False):
        """
        Read data into public instance variables

        :param pandas: if true, data should be DataFrame, else list
        :return: NA
        """
        pass

    def read_trofi_data(self, pandas=False):
        pass

    def read_trofi_x_data(self, pandas=False):
        pass

    def read_vua_data(self, pandas=False):
        pass

    def read_vua_seq(self, pandas=False):
        pass

    def read_experiment_file(self, folder, filename, pandas, encoding=None):
        """
        Static method to read file
        Input: filename
        Output: if pandas: pd DataFrame, else: list of strings
        """
        filepath = os.path.join(self.data_dir, folder, filename)
        enc = 'utf-8' if not encoding else encoding

        if pandas:
            data = pd.read_csv(filepath, encoding=enc)
            return data, len(data)

        data = []
        with open(filepath, encoding=enc) as f:
            lines = csv.reader(f)
            next(lines)
            for line in lines:
                data.append(line)
        f.close()
        return data, len(data)



