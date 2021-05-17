from convokit import Corpus, download, VectorClassifier
import os
import pandas as pd
from convokit import BoWTransformer

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



def win_lose_BoW(corpus):
    """
    http://scdb.wustl.edu/data.php
    :param corpus:
    :return:
    """
    bow_transformer = BoWTransformer(obj_type="conversation", vector_name='bow_vector_2',
                                      text_func=lambda convo: ' '.join(
                                          [utt.text for utt in convo.get_chronological_utterance_list()])
                                      )



def main():

    # get the corpus up and running
    # download_corpus('supreme-2019', 'dataset')
    corpus = Corpus('dataset/supreme-2019')
    corpus.print_summary_stats()

    # print all speakers in corpus
    # for convo in corpus.iter_conversations():
    #     print(convo.meta, end='\n\n')
    #     for speaker in convo.iter_speakers():
    #         print(speaker.meta)

    # BoW to predict whether judge is speaking on utterance level
    # print(corpus.random_utterance().meta)
    # predict_judge_speaking(corpus)

    # boW to predict win or lose on

    # add metadata file to corpus
    cases_path = 'dataset/cases.jsonl'
    corpus = add_convo_meta(corpus, cases_path, 'case_id')

    # parse win side
    for case in corpus.iter_conversations():
        if case.meta['win_side'] == 1:
            case.meta['win'] = True
        elif case.meta['win_side'] == 0:
            case.meta['win'] = False
        else:
            case.meta['win'] = None

    win_lose_BoW(corpus)



if __name__ == '__main__':
    main()
