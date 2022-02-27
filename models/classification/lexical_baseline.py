from importlib.util import module_for_loader
import numpy as np


class LexicalBaseline():

    """
    Class containing Lexical Baseline model
    """

    def __init__(self):
        print("Initializing new Lexical Baseline model")
        self.CLS_model = None
        self.predictions = None
        self.labels = None

        #evaluation metrics
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.met_f1 = None


    def create_CLS_Model(self, dataset):

        """
        :param dataset: list of tuples of form ("verb","label")
        :self.CLS_model: a dictionary: verb --> number that shows the probability
                    of being metaphor
                    dictionary only contains verbs that are more likely to
                    be metaphors

        """
        model = {}
        for verb, label in dataset:
            if verb in model:
                model[verb].append(int(label))
            else:
                model[verb] = [int(label)]

        final_model = {}
        for key in model.keys():
            value = model[key]
            prob = sum(value) / len(value)
            if prob > 0.5:
                final_model[key] = prob

        self.CLS_model = final_model


    def CLS_predict(self, dataset):
        """
        :param dataset: a list of verb-label pairs
        outcomes
        :self.predictions: list of predictions
        :self.labels: list of labels
        """
        predictions = []
        labels = []
        for verb, label in dataset:
            labels.append(int(label))
            if verb in self.CLS_model:
                predictions.append(1)
            else:
                predictions.append(0)

        self.predictions = predictions
        self.labels = labels
        #return evaluate(predictions, labels)


    def evaluate(self):
        """
        :param predictions: a list
        :param labels: a list
        :return: 4 numbers: precision, recall, met_f1, accuracy
        """
        # Set model to eval mode, which turns off dropout.
        predictions = self.predictions
        labels = self.labels
        assert(len(predictions) == len(labels))
        total_examples = len(predictions)

        num_correct = 0
        confusion_matrix = np.zeros((2, 2))
        for i in range(total_examples):
            if predictions[i] == labels[i]:
                num_correct += 1
            confusion_matrix[predictions[i], labels[i]] += 1

        assert(num_correct == confusion_matrix[0, 0] + confusion_matrix[1, 1])
        self.accuracy = 100 * num_correct / total_examples
        self.precision = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[1])
        self.recall = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[:, 1])
        self.met_f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

        print(f"""
                Accuracy: {self.accuracy}
                Precision: {self.precision}
                Recall: {self.recall}
                F1: {self.met_f1}
                """)
         



#MOH-X preprocessing
def process_moh_x_to_tuple(dataset):
    """
    Function for putting MOH_X data in correct format for lexical
    baseline model

    Input
        dataset: list of lists obtained from ExperimentData class
                moh_x_formatted_svo attribute
    Output
        moh_x_tpl_list: list of tuples of form (verb, label)
    """

    moh_x_tpl_list = []
    for line in dataset:
        moh_x_tpl_list.append((line[2],line[5]))

    return moh_x_tpl_list

#TROFI preprocessing
def process_trofi_to_tuple(dataset):
    """
    Function for putting TroFi data in correct format for lexical
    baseline model

    Input
        dataset: list of lists obtained from ExperimentData class
                trofi_formatted_all attribute
    Output
        trofi_tpl_list: list of tuples of form (verb, label)
    """

    trofi_tpl_list = []
    for line in dataset:
        trofi_tpl_list.append((line[0],line[3]))

    return trofi_tpl_list


# VUA preprocessing
def process_vua_to_tuple(dataset):
    """
    Function for putting VUA data in correct format for lexical
    baseline model

    Input
        dataset: list of lists obtained from ExperimentData class
                vua_formatted_{} attribute
    Output
        vua_tpl_list: list of tuples of form (verb, label)
    """

    vua_tpl_list = []
    for line in dataset:
        vua_tpl_list.append((line[2],line[5]))

    return vua_tpl_list



def lex_baseline_CV(dataset, num_folds=10,rand_seed=3):
    """
    Perform k-fold cross validation with the lexical baseline
    model on input data
    Input
        dataset: list of tuples of the form ('verb', 'label)
    """
    
    # upside down floor division
    lines_per_fold = -(len(dataset) // -10)

    random.seed(rand_seed)
    random.shuffle(dataset)

    # prepare 10 folds
    folds = []
    for i in range(num_folds):
        folds.append(dataset[i*lines_per_fold: (i+1)*lines_per_fold])

    # k fold
    PRFA_list = []
    for i in range(num_folds):
        raw_train_set = []
        raw_val_set = []
         # separate training and validation data
        for j in range(num_folds):
            if j != i:
                raw_train_set.extend(folds[j])
            else:
                raw_val_set = folds[j]
        # make model, predict, and evaluate
        model = lb.LexicalBaseline()
        model.create_CLS_Model(raw_train_set)
        model.CLS_predict(raw_val_set)
        model.evaluate()
        PRFA_list.append([model.precision,
                        model.recall,
                        model.met_f1,
                        model.accuracy])

    PRFA = np.array(PRFA_list)
    return PRFA