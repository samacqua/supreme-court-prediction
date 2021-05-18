import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import transformers as ppb
import warnings
from transformers import TrainingArguments
from transformers import Trainer
warnings.filterwarnings('ignore')

def compute_metrics(p):
    """function used by the trainer telling it which metrics to show. Input is a tuple of pred, labels"""
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Create torch dataset subclass so we can make our own
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def distill_bert_classification(data, sentence_column, label_column): 
    """assigns pre-trained BERT weights to each sentence, then uses an sklearn logistic regression to classify 0 or 1
       data is a dataframe, sentence_column is the column of the dataframe with the features(sentences), label column has labels
       (can be string or index)"""
    # load distillbert model, tokenizer, and weights
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    
    # tokenizer breaks up sentences into a format the transformer needs as input
    padded = tokenizer(list(data[sentence_column].values), padding=True, truncation=True)
    
    # attention mask tells bert where values are padded (should be ignored)
    attention_mask = np.where(padded != 0, 1, 0)
    
    input_ids = torch.Tensor(padded['input_ids']).to(torch.int64)
    attention_mask = torch.Tensor(padded['attention_mask']).to(torch.int64)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    # weights for each sentence for classification
    features = last_hidden_states[0][:,0,:].numpy()
    labels = data[label_column]
    
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)
    return lr_clf.score(test_features, test_labels)


def distill_bert_with_training(data, sentence_column, label_column): 
	"""
	trains on eval dataset and can test after. Currenlty broken :(
	"""
    # load distillbert model, tokenizer, and weights
	model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

	# Load pretrained model/tokenizer
	tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
	model = model_class.from_pretrained(pretrained_weights)

	x = data[sentence_column].values
	y = data[label_column].values

	x_train, x_eval, y_train, y_eval = train_test_split(x, y)
	tune_x_train, tune_x_eval, tune_y_train, tune_y_eval = train_test_split(x_train, y_train)

	tune_x_train_tok = tokenizer(list(tune_x_train), padding=True, truncation=True)
	tune_x_eval_tok = tokenizer(list(tune_x_eval), padding=True, truncation=True)
	x_eval_tok = tokenizer(list(x_eval), padding=True, truncation=True)

	train_dataset=Dataset(tune_x_train_tok, tune_y_train)
	eval_dataset=Dataset(tune_x_eval_tok, tune_y_eval)

	training_args = TrainingArguments("testing")

	trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, 
	                  eval_dataset=eval_dataset, compute_metrics=compute_metrics)

	#line that doesn't work
	trainer.train()
	return trainer.evaluate()
