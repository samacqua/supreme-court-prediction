import numpy as np
from classifier import Classifier


class NaiveBayes(Classifier):
    """
    calculates P(class|data) = (P(data|class) * P(class)) / P(data)
    """

    def addToBow(self, tokens, dict_index):
        """
        adds count of each word to Bag of Words
        :param tokens: a tokenized utterance to add
        :param dict_index: the category to add the count to
        """

        if isinstance(tokens, np.ndarray): tokens = tokens[0]  # sometimes given as [utterance] from np apply func

        for token_word in tokens:
            self.bow_dicts[dict_index][token_word] = self.bow_dicts[dict_index].get(token_word, 0) + 1

    def train(self, dataset, labels):
        """
        trains naive bayes by calculating P(class|data) = (P(data|class) * P(class)) / P(data) based on dataset
        """

        self.examples = dataset
        self.labels = labels
        self.bow_dicts = np.array([{} for _ in range(self.classes.shape[0])])

        for cat_index, cat in enumerate(self.classes):
            all_cat_examples = self.examples[self.labels == cat]

            cleaned_examples = [self.tokenize_string(cat_example) for cat_example in all_cat_examples]
            for ex in cleaned_examples:
                self.addToBow(ex, cat_index)

        # precalculate for test time:
        #   1. prior probability of each class - p(c)
        #   2. vocabulary |V|
        #   3. denominator value of each class - [ count(c) + |V| + 1 ]

        prob_classes = np.empty(self.classes.shape[0])
        all_words = []
        cat_word_counts = np.empty(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):
            # Calculating prior probability p(c)
            prob_classes[cat_index] = np.sum(self.labels == cat) / float(self.labels.shape[0])

            # Calculating total counts of all the words
            cat_word_counts[cat_index] = np.sum(
                np.array(list(self.bow_dicts[cat_index].values()))) + 1  # |v| is remaining to be added

            # get all words of this category
            all_words += self.bow_dicts[cat_index].keys()

        # combine all words of every category & make them unique to get vocabulary -V- of entire training set
        self.vocab = np.unique(np.array(all_words))
        self.vocab_length = self.vocab.shape[0]

        # computing denominator value
        denoms = np.array(
            [cat_word_counts[cat_index] + self.vocab_length + 1 for cat_index, cat in enumerate(self.classes)])

        # make easily indexable
        self.cats_info = [(self.bow_dicts[cat_index], prob_classes[cat_index], denoms[cat_index]) for cat_index, cat in
                          enumerate(self.classes)]
        self.cats_info = np.array(self.cats_info)

    def getExampleProb(self, test_example):
        """
        get the posterior probability of given example for each class
        :param test_example: the example to use to classify
        :return: log probability of given example for each class
        """

        tokens = self.tokenize_string(test_example)
        likelihood_prob = np.zeros(self.classes.shape[0])

        for cat_index, cat in enumerate(self.classes):

            for test_token in tokens:  # split the test example and get p of each test word

                # This loop computes : for each word w [ count(w|c)+1 ] / [ count(c) + |V| + 1 ]

                # get total count of this test token from it's respective training dict to get numerator value
                test_token_counts = self.cats_info[cat_index][0].get(test_token, 0) + 1

                # now get likelihood of this test_token word
                test_token_prob = test_token_counts / float(self.cats_info[cat_index][2])

                likelihood_prob[cat_index] += np.log(test_token_prob)

        # we have likelihood estimate of the given example against every class but we need posterior probility
        post_prob = np.empty(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):
            post_prob[cat_index] = likelihood_prob[cat_index] + np.log(self.cats_info[cat_index][1])

        return post_prob

    def test(self, test_set):
        predictions = []
        for example in test_set:

            # get the posterior probability of every example
            post_prob = self.getExampleProb(example)

            # pick the max value and map against self.classes!
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

    nb = NaiveBayes(np.unique(train_labels))  # instantiate a NB class object
    print("---------------- Training In Progress --------------------")

    nb.train(train_data, train_labels)  # start tarining by calling the train function
    print('----------------- Training Completed ---------------------')

    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)  # loading test data
    test_data = newsgroups_test.data  # get test set examples
    test_labels = newsgroups_test.target  # get test set labels

    print("Number of Test Examples: ", len(test_data))  # Output : Number of Test Examples:  1502
    print("Number of Test Labels: ", len(test_labels))  # Output : Number of Test Labels:  1502

    pclasses = nb.test(test_data)  # get predcitions for test set

    # check how many predcitions actually match original test labels
    test_acc = np.sum(pclasses == test_labels) / float(test_labels.shape[0])

    print("Test Set Examples: ", test_labels.shape[0])  # Outputs : Test Set Examples:  1502
    print("Test Set Accuracy: ", test_acc * 100, "%")  # Outputs : Test Set Accuracy:  93.8748335553 %
