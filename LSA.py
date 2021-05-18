from classifier import Classifier
import numpy as np

from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel

import matplotlib.pyplot as plt


class LSA(Classifier):

    def tokenize_string(self, str_arg):
        tokenizer = RegexpTokenizer(r'\w+')
        en_stop = set(stopwords.words('english'))
        p_stemmer = PorterStemmer()

        # clean and tokenize document string
        raw = str_arg.lower()
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [i for i in tokens if not i in en_stop]
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

        return stemmed_tokens

    def create_matrix(self, tokenized_dataset):
        """

        :param tokenized_dataset:
        :return:
        """
        dictionary = corpora.Dictionary(tokenized_dataset)  # assign each unique term an index
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in tokenized_dataset]    # map each term to index in matrix

        return dictionary, doc_term_matrix

    def create_lsa(self, tokenized_dataset, num_topics):
        """
        Input  : clean document, number of topics and number of words associated with each topic
        Purpose: create LSA model using gensim
        Output : return LSA model
        """
        dictionary, doc_term_matrix = self.create_matrix(tokenized_dataset)

        self.model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary)  # train model

    def compute_coherence_values(self, dictionary, doc_term_matrix, doc_clean, start=1, stop=10, step=1):
        """
        compute the coherence of different number of topics
        https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
        :param dictionary:
        :param doc_term_matrix:
        :param doc_clean:
        :param start:
        :param stop:
        :param step:
        :return:
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, stop, step):
            print(num_topics)
            # generate LSA model
            model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary)  # train model
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
        return model_list, coherence_values

    def plot_coherence(self, tokenized_datset):
        """
        plot the coherence values over a range of topic numbers
        https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
        :param tokenized_datset:
        :return:
        """
        dictionary, doc_term_matrix = self.create_matrix(tokenized_datset)
        model_list, coherence_values = self.compute_coherence_values(dictionary, doc_term_matrix, tokenized_datset,
                                                                1, 10, 1)
        # Show graph
        plt.plot(coherence_values)
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()

    def tfidf(self, cleaned_dataset):
        """
        calculate the term frequency - inverse document frequency for dataset
        :param cleaned_dataset: np array of sentences (documents) that are tokenizable
        :return: tf-idf (np array of dictionaries)
        """

        tf = {}
        document_counts = {}    # how many docs each word is in
        for i, doc in enumerate(cleaned_dataset):
            doc_word_counts = {}
            words = doc.split(' ')

            for word in words:    # count number of times word appears in document
                if word not in doc_word_counts:     # if word occurs multiple times, still only in doc once
                    document_counts[word] = document_counts.get(word, 0) + 1
                doc_word_counts[word] = doc_word_counts.get(word, 0) + 1

            num_words = len(words)  # / num words in the doc
            for word, count in doc_word_counts.items():
                tf[i, word] = count / num_words

        # calculate inverse-document frequency
        idf = {}
        num_docs = len(cleaned_dataset)
        for word, docs_w_word in document_counts.items():
            idf[word] = np.log(1 + num_docs / docs_w_word)

        # join to calculate tf-idf
        tfidf = {}
        for doc, word in tf:
            tfidf[doc] = tfidf.get(doc, {})
            tfidf[doc][word] = tf[doc, word] * idf[word]

        return tfidf

    def train(self, dataset, labels, num_themes=30):
        # cleaned_dataset = [Classifier.preprocess_string(doc) for doc in dataset]
        # tfidf = self.tfidf(cleaned_dataset)
        # tfidf_df = pd.DataFrame(tfidf).fillna(0)
        # tfidf_np = tfidf_df.to_numpy()
        # u, s, vh = np.linalg.svd(tfidf_np)
        #
        # for i in range(num_themes):
        #     word_weights = u[:,i]
        #     top_words_i = word_weights.argsort()[-5:][::-1]
        #     top_words = []
        #     for indx in top_words_i:
        #         top_words.append(( tfidf_df.index[indx], word_weights[indx] ))
        #
        #     print(i, s[i], top_words)

        tokenized_dataset = [self.tokenize_string(doc) for doc in dataset]
        self.model = self.create_lsa(tokenized_dataset, num_themes)

    def test(self, test_set):
        print(self.model.show_topics(5))


if __name__ == '__main__':
    from sklearn.datasets import fetch_20newsgroups

    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']  # limiting to 4 categories
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)

    train_data = np.array(newsgroups_train.data)[:300]  # getting all trainign examples
    train_labels = newsgroups_train.target[:300]  # getting training labels

    print("Total Number of Training Examples: ", len(train_data))  # Outputs -> Total Number of Training Examples:  2257
    print("Total Number of Training Labels: ", len(train_labels))  # Outputs -> #Total Number of Training Labels:  2257

    c = LSA(np.unique(train_labels))  # instantiate a NB class object
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


