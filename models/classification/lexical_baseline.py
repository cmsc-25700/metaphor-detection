class LexicalBaseline():

    """
    Class containing Lexical Baseline model
    """

    def __init__(self):
        print("Initializing new Lexical Baseline model")
        self.CLS_model = None


    def create_CLS_Model(self, dataset):

        """
        :param dataset: list of tuples of form ("verb","label")
        :return: a dictionary: verb --> number that shows the probability
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