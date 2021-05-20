from convokit import Corpus, download
from convokit import BoWTransformer

import os
import pandas as pd
from random import shuffle
import numpy as np

from naivebayes import NaiveBayes
from naive_classifier import NaiveClassifier
from LSA import LSA

def download_corpus(name, base_path='dataset'):
    """
    download a corpus using and save it locally
    :param name: the name of the corpus you want to save
    :param base_path: the base path to download the corpus to
    """
    corpus = Corpus(filename=download(name))

    if not os.path.exists(base_path):
        os.makedirs(base_path)
    corpus.dump(name, base_path=base_path)


def predict_judge_speaking(corpus):
    """
    use BoW to see predict if speaker is a justice
    just following along w: https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/vectors/bag-of-words-demo.ipynb
    """

    bow_transformer = BoWTransformer(obj_type="utterance")
    bow_transformer.fit_transform(corpus)

    from convokit import VectorClassifier
    bow_classifier: VectorClassifier = VectorClassifier(obj_type="utterance",
                                      vector_name='bow_vector',
                                      labeller=lambda utt: utt.meta['speaker_type'] == 'J')
    bow_classifier.fit_transform(corpus)
    bow_classifier.summarize(corpus).head(10)
    print("Weight of words in predicting whether judge or not")
    print(bow_classifier.get_coefs(feature_names=corpus.get_vector_matrix('bow_vector').columns).head())
    print(bow_classifier.get_coefs(feature_names=bow_transformer.get_vocabulary()).tail())

    # The base accuracy by predicting all objects to have the majority label, i.e. has positive score
    print("Baseline accuracy:", bow_classifier.base_accuracy(corpus))
    print("Classifier accuracy:", bow_classifier.accuracy(corpus))

    test_size = 0.3
    corp_size = len(list(corpus.iter_utterances()))
    acc, conf_matrix = bow_classifier.evaluate_with_train_test_split(test_size=int(corp_size*test_size), corpus=corpus)
    print(f"Accuracy using {1-test_size}/{test_size} train/test split:", acc)


def add_convo_meta(corpus, meta_fname, corpus_id, lines=True, overwrite=False):
    """
    add meta data to the conversation of a corpus
    :param corpus: the corpus to add meta data to
    :param meta_fname: the name of the json file to read the metadata from.
                       Must have 'id' match the id field in the corpus
    :param corpus_id: the name of the id field in the corpus converations
    :param lines: True if .jsonl file, False if .json
    :param overwrite: overwrite previous metadata?
    :return: the modified corpus
    """
    df = pd.read_json(path_or_buf=meta_fname, lines=lines)

    for convo in corpus.iter_conversations():
        case_meta = df[df.id == convo.meta[corpus_id]].to_dict()
        for key, val_dict in case_meta.items():
            val = list(val_dict.values())[0]     # unpack the val -- pd to_dict() gives {key: {indx: val}}
            if not overwrite and key not in corpus.meta:  # don't overwrite
                convo.add_meta(key, val)

    return corpus


def train_test_split(data, labels, split=0.8):
    """
    split into train and test set
    :param data: np array of complete data (m x d)
    :param labels: np array of labels (m x 1)
    :param split: the train-test split, defaults to 0.8
    :return: np arrays of train_data, train_labels, test_data, test_labels
    """

    # randomize the order
    shuffled_data = list(zip(data, labels))
    shuffle(shuffled_data)

    # split into train and test
    split_indx = int(len(data) * split)
    train_data, train_labels = zip(*shuffled_data[:split_indx])
    train_data, train_labels = np.array(train_data), np.array(train_labels)
    test_data, test_labels = zip(*shuffled_data[split_indx:])
    test_data, test_labels = np.array(test_data), np.array(test_labels)

    return train_data, train_labels, test_data, test_labels


def test_classifier(classifier_class, data, labels, num_trials=15, verbose=True):
    """
    test a classifier on test data and labels n times
    :param classifier_class: the classifier to evaluate. Class takes array of classes in initializer, and has train
                             and test methods
    :param data: d x n array of test data
    :param labels: 1 x n array of test labels
    :param num_trials: the number of trials to evaluate over
    :param verbose: print accuracy per trial
    :return: np array of accuracies over all trials
    """

    test_accs = np.zeros(num_trials)
    for i in range(num_trials):
        train_data, train_labels, test_data, test_labels = train_test_split(data, labels)
        unique_labels = np.unique(train_labels)
        if verbose:
            N = len(train_labels)
            print('\tlabels:', end='\t')
            for j, lab in enumerate(unique_labels):
                print(f'{lab}: {round(np.sum(train_labels == j) / N*100, 2)}%', end='\t')
            print('')

        classifier = classifier_class(unique_labels)

        classifier.train(train_data, train_labels)

        pclasses = classifier.test(test_data)  # get predcitions for test set
        test_acc = np.sum(pclasses == test_labels) / float(test_labels.shape[0]) * 100  # perc correct predictions

        test_accs[i] = test_acc
        if verbose:
            print(f"\ttest accuracy trial {i}: {round(test_acc, 3)} %")

    return test_accs


def main():

    ############
    # get corpus
    ############

    # download_corpus('supreme-2015', 'dataset')
    # corpus = Corpus('dataset/supreme-2018')

    # for y in range(2000, 2015):
    #     download_corpus(f'supreme-{y}', 'dataset')

    # get multiple corpuses
    start_year = 2000
    years = list(range(start_year, 2020))
    corpus = Corpus(f'dataset/supreme-{years[0]}')
    for year in years[1:]:
        corpus = corpus.merge(Corpus(f'dataset/supreme-{year}'))

    corpus.print_summary_stats()

    #############################
    # add metadata file to corpus
    #############################

    cases_path = 'dataset/cases.jsonl'
    corpus = add_convo_meta(corpus, cases_path, 'case_id')

    #######################
    # get dataset w/ labels
    #######################

    data = []
    transformer_data = []
    labels = []
    max_conv_len = float('inf')
    for convo in corpus.iter_conversations():
        convo_str = ' '.join([utt.text for utt in convo.iter_utterances()])     # get all as one string
        label = convo.meta['win_side']
        if label not in [0, 1]:
            continue    # ignore cases w/ out clear outcome

        # take into account max attention of transformer
        words = convo_str.split(' ')
        head_tail_words = words[:128] + words[-382:]
        head_tail_convo_str = ' '.join(head_tail_words)

        # even if transformer has attention, for size/memory purposes, reduce num words
        if len(words) > max_conv_len:
            print('too long')
            truncated_words = words[:int(max_conv_len*0.25)] + words[-int(max_conv_len*0.75):]
            convo_str = ' '.join(truncated_words)

        transformer_data.append(head_tail_convo_str)
        data.append(convo_str)
        labels.append(label)

    df = pd.DataFrame()
    df['sentence'] = transformer_data
    df['label'] = labels
    df.to_csv(f'formatted_datasets/scotus_head_tail_{start_year}.csv', index=False)

    df = pd.DataFrame()
    df['sentence'] = data
    df['label'] = labels
    df.to_csv(f'formatted_datasets/scotus_{max_conv_len}_{start_year}.csv', index=False)

    return

    #################
    # try classifiers
    #################

    num_trials = 1

    classifiers = [NaiveBayes, NaiveClassifier, LSA]
    for classifier_type in classifiers:
        print('\n', str(classifier_type)[8:-2])

        class_accs = test_classifier(classifier_type, data, labels, num_trials, verbose=False)
        print(f"test accuracy: {round(np.mean(class_accs), 3)} %")

        class_accs = np.sort(class_accs)
        conf = 0.9
        conf_indx = int(num_trials * (1 - conf))
        conf = 1 - (conf_indx + 1) / num_trials  # bc rounding, not actually conf interval so update
        print(f"bootstrap {round(conf * 100, 2)}% " +
              f"confidence: {round(class_accs[conf_indx], 2)} % - {round(class_accs[num_trials - 1 - conf_indx], 2)} %")


if __name__ == '__main__':
    main()
