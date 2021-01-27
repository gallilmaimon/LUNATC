import torch
import numpy as np
from collections import defaultdict


def embed_texts(texts, model, tokeniser, num_hiddens=4, maxlen=256, device="cuda"):
    inputs = tokeniser(texts, padding=True, truncation=True, max_length=maxlen, pad_to_multiple_of=maxlen,
                       return_tensors='pt')

    inputs['input_ids'] = inputs['input_ids'].to(device)
    inputs['attention_mask'] = inputs['attention_mask'].to(device)
    inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.cat([layer.unsqueeze(0) for layer in outputs[2][-num_hiddens:]]).mean(dim=[0, 2])


def get_bert_hidden_layers(sent, model, tokenizer, device, max_size=512):
    """returns a list (with length same as the number of layers) of tensors each of shape
    [batch_size, len of tokenised sent (+2), hidden state shape]"""
    # add special start and end tokens
    marked_text = "[CLS] " + sent + " [SEP]"

    # tokenise text
    tokenized_text = tokenizer.tokenize(marked_text)

    # turn tokens into BERT number ids
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # generate token used to indicate that all words are from the same sent
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens[:max_size]]).to(device)
    segments_tensors = torch.tensor([segments_ids[:max_size]]).to(device)

    # Put the model in "evaluation" mode
    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    return encoded_layers


def embed_sentence_mean_layer11(sent, model, tokenizer, device):
    """
    follows a text embedding method suggested in the original BERT paper, of taking the mean of the vector for each
    word at the 11th layer. This outputs a fixed size (768) embedding for each input text
    :param sent: the sentence to be embedded
    :param model: the BERT language model used
    :param tokenizer: the BERT tokeniser used to tokenise the text
    :return: a torch tensor representing the embedding (size 768)
    """
    encoded_layers = get_bert_hidden_layers(sent, model, tokenizer, device)
    return torch.mean(encoded_layers[11], 1)


def pad_sequence(inp, maxlen, token=0):
    if len(inp) >= maxlen:
        return inp[:maxlen-1] + [inp[-1]]
    else:
        return inp + [token]*(maxlen - len(inp))


# region GLOVE
def load_embedding(path):
    words = []
    vals = []
    with open(path, encoding='utf-8') as fin:
        fin.readline()
        for line in fin:
            line = line.rstrip()
            if line:
                parts = line.split(' ')
                words.append(parts[0])
                vals += [float(x) for x in parts[1:]]
    return words, np.asarray(vals).reshape(len(words), -1)


def load_embedding_dict(path, default_vec):
    word2vec = defaultdict(lambda: default_vec)
    with open(path, encoding='utf-8') as fin:
        fin.readline()
        for line in fin:
            line = line.rstrip()
            if line:
                parts = line.split(' ')
                word2vec[parts[0]] = torch.tensor([[float(x) for x in parts[1:]]])
    return word2vec


def glove_tokenify(text, word2ind, pad_ind, max_len=128):
    return torch.Tensor(pad_sequence([word2ind[w] for w in text.split()], max_len, pad_ind)).view(1, -1).long()
# endregion GLOVE
