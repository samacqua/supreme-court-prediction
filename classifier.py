import re

class Classifier:
    """
    base class for all classifiers
    """
    def __init__(self, classes):
        """
        :param classes: np array of unique classes
        """
        self.classes = classes

    def train(self, dataset, labels):
        """
        trains classifier
        :param dataset: np array of shape m x d
        :param labels: np array of shape m x 1
        """
        raise NotImplementedError

    def test(self, test_set):
        """
        predict the labels for a set of examples
        :param test_set: data to use to predict
        :return: np array of predictions
        """
        raise NotImplementedError

    def tokenize_string(self, str_arg):
        """"
        tokenize a string of text
        """

        cleaned_str = re.sub('[^\w\s\']+', ' ', str_arg, flags=re.IGNORECASE)  # ignore non-alphanumeric or dashes or ' or whitespace
        cleaned_str = re.sub('(\s+)', ' ', cleaned_str)  # multiple spaces are replaced by single space
        cleaned_str = cleaned_str.lower()  # converting the cleaned string to lower case

        return cleaned_str.split()
