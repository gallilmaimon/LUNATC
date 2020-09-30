# imports
import numpy as np
from text_xai.TextModels.TextModel import TextModel

# bert
import torch
from pytorch_pretrained_bert import BertModel
from transformers import BertForSequenceClassification, BertTokenizer  # , BertModel
from text_xai.TextModels.text_model_utils import embed_sentence_mean_layer11, pad_sequences


class E2EBertTextModel(TextModel):
    def __init__(self, num_classes=2, trained_model=None, bert_type='bert-base-uncased', device="cuda"):
        self.num_classes = num_classes

        # end2end bert for sequence classification (the model being "attacked")
        self.bert_tokeniser = BertTokenizer.from_pretrained(bert_type, do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                   num_labels=num_classes)
        if trained_model is not None:
            self.model.load_state_dict(torch.load(trained_model, map_location=lambda storage, loc: storage))

        # set to eval mode
        self.model.eval()
        self.model.to(device)

        # pre-trained bert LM as text embedder
        self.bert_model = BertModel.from_pretrained(bert_type)
        self.bert_model.to(device)
        self.device = device

    def train(self, X, y):
        # At the moment only accepts pre-trained models for class simplicity
        raise NotImplementedError

    def test(self, X, y):
        # At the moment only accepts pre-trained models for class simplicity
        raise NotImplementedError

    def embed(self, X):
        return embed_sentence_mean_layer11(X, self.bert_model, self.bert_tokeniser, self.device)

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            sent_token = torch.Tensor(pad_sequences([self.bert_tokeniser.encode(X, add_special_tokens=True)],
                                                    128)).long().to(self.device)
            sent_att = (sent_token > 0).int().to(self.device)
            # res = F.softmax(self.model(sent_token, attention_mask=sent_att)[0])
            res = self.model(sent_token, attention_mask=sent_att)[0]
            probs = res.detach().cpu().numpy()
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
