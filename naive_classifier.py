import numpy as np

class NaiveClassifier:
    """
    just guesses the category that occurs most often in the training set
    """
    def __init__(self, classes):
        self.classes = classes
        self.probs = np.zeros(len(classes))

    def train(self, dataset, labels):
        N = labels.shape[0]
        for i, c in enumerate(self.classes):
            count = np.sum(labels == c)
            self.probs[i] = count / N

    def test(self, test_set):
        predictions = []
        for example in test_set:
            post_prob = self.probs
            predictions.append(self.classes[np.argmax(post_prob)])

        return np.array(predictions)

if __name__ == '__main__':
    from sklearn.datasets import fetch_20newsgroups

    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']  # limiting to 4 categories
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)

    train_data = np.array(newsgroups_train.data)  # getting all trainign examples
    train_labels = newsgroups_train.target  # getting training labels

    print("Total Number of Training Examples: ", len(train_data))  # Outputs -> Total Number of Training Examples:  2257
    print("Total Number of Training Labels: ", len(train_labels))  # Outputs -> #Total Number of Training Labels:  2257

    c = NaiveClassifier(np.unique(train_labels))  # instantiate a NB class object
    print("---------------- Training In Progress --------------------")

    c.train(train_data, train_labels)  # start tarining by calling the train function
    print('----------------- Training Completed ---------------------')

    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)  # loading test data
    test_data = newsgroups_test.data  # get test set examples
    test_labels = newsgroups_test.target  # get test set labels

    print("Number of Test Examples: ", len(test_data))  # Output : Number of Test Examples:  1502
    print("Number of Test Labels: ", len(test_labels))  # Output : Number of Test Labels:  1502

    pclasses = c.test(test_data)  # get predcitions for test set

    # check how many predcitions actually match original test labels
    test_acc = np.sum(pclasses == test_labels) / float(test_labels.shape[0])

    print("Test Set Examples: ", test_labels.shape[0])  # Outputs : Test Set Examples:  1502
    print("Test Set Accuracy: ", test_acc * 100, "%")  # Outputs : Test Set Accuracy:  93.8748335553 %
