# imports
import numpy as np
from src.TextModels.TextModel import TextModel

# bert
import torch
from transformers import BertForSequenceClassification, BertTokenizer, BertModel, BertConfig
from src.TextModels.text_model_utils import embed_texts


class BertTextModel(TextModel):
    def __init__(self, num_classes=2, trained_model=None, bert_type='bert-base-uncased', device="cuda"):
        self.num_classes = num_classes

        # bert for sequence classification (the model being "attacked")
        self.bert_tokeniser = BertTokenizer.from_pretrained(bert_type, do_lower_case=True)
        if trained_model is not None:
            bert_config = BertConfig.from_pretrained(bert_type, num_labels=num_classes)
            self.model = BertForSequenceClassification.from_pretrained(None, config=bert_config,
                                                                       state_dict=torch.load(trained_model))
        else:
            self.model = BertForSequenceClassification.from_pretrained(bert_type, num_labels=num_classes)

        # set to eval mode
        self.model.eval()
        self.model.to(device)

        # pre-trained bert LM as text embedder
        config = BertConfig.from_pretrained(bert_type, output_hidden_states=True)
        self.bert_model = BertModel.from_pretrained(bert_type, config=config)
        self.bert_model.to(device)
        self.device = device

        self.proba_cache = dict()
        self.embed_cache = dict()

    def train(self, X, y):
        # At the moment only accepts pre-trained models for class simplicity
        raise NotImplementedError

    def test(self, X, y):
        # At the moment only accepts pre-trained models for class simplicity
        raise NotImplementedError

    def embed(self, X):
        if X in self.embed_cache:
            return self.embed_cache[X]
        embed = embed_texts(X, self.bert_model, self.bert_tokeniser, device=self.device)
        self.embed_cache[X] = embed
        return embed

    def predict_proba(self, X):
        if (not (type(X) == list or (type(X) == tuple and type(X[0]) == list))) and X in self.proba_cache:
            return self.proba_cache[X]

        self.model.eval()
        with torch.no_grad():
            if type(X) == tuple or (type(X) == list and type(X[0]) == tuple):
                # relevant for multi text models
                inputs = self.bert_tokeniser(*X, padding=True, truncation=True, max_length=256, pad_to_multiple_of=256,
                                             return_tensors='pt')
            else:
                inputs = self.bert_tokeniser(X, padding=True, truncation=True, max_length=256, pad_to_multiple_of=256,
                                             return_tensors='pt')

            sent_token = inputs['input_ids'].to(self.device)
            sent_att = inputs['attention_mask'].to(self.device)
            # res = F.softmax(self.model(sent_token, attention_mask=sent_att)[0])
            res = self.model(sent_token, attention_mask=sent_att)[0]
            probs = res.detach().cpu().numpy()
        if not (type(X) == list or (type(X) == tuple and type(X[0]) == list)):
            self.proba_cache[X] = probs
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
