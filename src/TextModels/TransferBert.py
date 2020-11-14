# imports
import numpy as np

from src.TextModels.TextModel import TextModel

# net
import torch.nn as nn
import torch.nn.functional as F

# bert
import torch
from transformers import BertTokenizer, BertModel
from src.TextModels.text_model_utils import embed_sentence_mean_layer11


class ClassificationHead(nn.Module):
    """
    A class for the Fully connected network with one hidden layer, for binary classification with an input the same size
    as the BERT embedding
    """
    def __init__(self):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(768, 400)
        self.fc2 = nn.Linear(400, 2)

    def forward(self, x):
        x = x.view(-1, 768)
        x = F.tanh(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x


class TransferBertTextModel(TextModel):
    """
    A language model giving predictions by first embedding the sentence using a BERT language model, and then inputting
    the received embedding into a simple fully connected network
    """
    def __init__(self, num_classes=2, model=ClassificationHead, trained_model=None, bert_type='bert-base-uncased',
                 device="cuda"):
        self.num_classes = num_classes

        # the fully connected net
        self.model = model()
        if trained_model is not None:
            self.model.load_state_dict(torch.load(trained_model, map_location=lambda storage, loc: storage))
        self.model.to(device)

        # set to eval mode
        self.model.eval()

        self.bert_model = BertModel.from_pretrained(bert_type)
        self.bert_model.to(device)
        self.bert_tokeniser = BertTokenizer.from_pretrained(bert_type)

        self.device = device

    def train(self, X, y):
        # At the moment only accepts pretrained models for class simplicity
        raise NotImplementedError

    def test(self, X, y):
        # At the moment only accepts pretrained models for class simplicity
        raise NotImplementedError

    def embed(self, X):
        return embed_sentence_mean_layer11(X, self.bert_model, self.bert_tokeniser, self.device)

    def predict_proba(self, X):
        embedded_sent = self.embed(X)
        self.model.eval()
        with torch.no_grad():
            probs = self.model(embedded_sent).detach().cpu().numpy()  # TODO: check about efficiency
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
