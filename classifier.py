class Classifier:
    """
    base class for all classifiers
    """
    def __init__(self, classes):
        self.classes = classes

    def train(self, dataset, labels):
        raise NotImplementedError

    def test(self, test_set):
        raise NotImplementedError