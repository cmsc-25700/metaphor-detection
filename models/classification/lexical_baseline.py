class LexicalBaseline():

    """
    Class containing Lexical Baseline model
    """

    def __init__(self):
        print("Initializing new Lexical Baseline model")
        self.CLS_model = None


    def create_CLS_Model(dataset):

        """
        :param dataset: list of tuples of form ("verb","label")
        :return: a dictioanry: verb --> number that shows the probability of being metaphor
        """
        model = {}
        for verb, label in dataset:
            if verb in model:
                model[verb].append(label)
            else:
                model[verb] = [label]

        final_model = {}
        for key in model.keys():
            value = model[key]
            prob = sum(value) / len(value)
            if prob > 0.5:
                final_model[key] = prob

        self.CLS_model = final_model
        return final_model



#MOH-X preprocessing
def process_moh_x_to_tuple(dataset):
    """
    Function for putting MOH_X data in correct format for lexical baseline model

    Input
        dataset: list of lists obtained from ExperimentData class
                moh_x_formatted_svo attribute
    Output
        moh_x_list: list of tuples of form (verb, label)
    """

    moh_x_list = []
    for line in dataset:
        moh_x_list.append((line[2],line[5]))

    return moh_x_list

