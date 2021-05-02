# imports
import numpy as np
from collections import defaultdict
from src.TextModels.TextModel import TextModel
import os 

# net
import torch.nn as nn

# bert
import torch
from transformers import BertModel, BertConfig, BertTokenizer
from src.TextModels.text_model_utils import glove_tokenify, load_embedding, embed_texts

LIB_DIR = os.path.abspath(__file__).split('src')[0]
VOCAB_SIZE = 399999 + 2  # number of Glove words + 2 special tokens (padding + out of vocabulary)
EMBEDDING_SIZE = 200  # glove embedding size


class WordLSTMNet(nn.Module):
    def __init__(self, hidden_size=150, depth=1, nclasses=2):
        super(WordLSTMNet, self).__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.lstm = nn.LSTM(EMBEDDING_SIZE, hidden_size // 2, depth, bidirectional=True, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.out = nn.Linear(hidden_size, nclasses)

    def forward(self, x):
        emb = self.embedding(x)
        output, hidden = self.lstm(emb)
        output = torch.tanh(self.bn(torch.max(output, dim=1)[0]))
        return self.out(output)


class WordLSTM(TextModel):
    """
    A language model giving predictions by first embedding the sentence using a BERT language model, and then inputting
    the received embedding into a simple fully connected network
    """
    def __init__(self, num_classes=2, trained_model=None, bert_type='bert-base-uncased', maxlen=128, device="cuda",
                 glove_path='/resources/word_vectors/glove.6B.200d.txt'):
        self.num_classes = num_classes

        # the fully connected net
        self.model = WordLSTMNet()
        if trained_model is not None:
            self.model.load_state_dict(torch.load(trained_model))
        self.model.to(device)

        # set to eval mode
        self.model.eval()

        # # load glove words and create word2ind
        words, _ = load_embedding(LIB_DIR + glove_path)
        self.len_words = len(words)
        self.word2ind = defaultdict(self.len_word)  # this is the out of vocabulary token
        self.maxlen = maxlen  # fixed length for padding sequences
        self.pad_ind = len(words) + 1  # a padding index so that all inputs are the same length
        self.word2ind.update({word: i for i, word in enumerate(words)})

        # pre-trained bert LM as text embedder
        config = BertConfig.from_pretrained(bert_type, output_hidden_states=True)
        self.bert_tokeniser = BertTokenizer.from_pretrained(bert_type, do_lower_case=True)
        self.bert_model = BertModel.from_pretrained(bert_type, config=config)
        self.bert_model.to(device)
        self.device = device

    def len_word(self):
        return self.len_words

    def train(self, X, y):
        # At the moment only accepts pretrained models for class simplicity
        raise NotImplementedError

    def test(self, X, y):
        # At the moment only accepts pretrained models for class simplicity
        raise NotImplementedError

    def embed(self, X):
        return embed_texts(X, self.bert_model, self.bert_tokeniser, device=self.device)

    def predict_proba(self, X):
        if type(X) == list:
            embedded_sent = torch.cat([glove_tokenify(text, self.word2ind, self.pad_ind, self.maxlen).to(self.device)
                                      .view(1, -1) for text in X])
        else:
            embedded_sent = glove_tokenify(X, self.word2ind, self.pad_ind, self.maxlen).to(self.device)
        self.model.eval()
        with torch.no_grad():
            probs = self.model(embedded_sent).detach().cpu().numpy()
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# # TODO: move to notebook or something
# import pandas as pd
# model = WordLSTM(trained_model='../../data/aclImdb/imdb_word_lstm.pth')
# df = pd.read_csv('../../data/aclImdb/imdb_sample.csv')
# df = df.dropna()
# model.model.eval()
#
# outs = []
# for text in df.content:
#     outs.append(model.predict(text))
#
# df['preds_lstm'] = np.concatenate(outs)
# df.to_csv('../../data/aclImdb/imdb_sample.csv', index=False)