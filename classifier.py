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