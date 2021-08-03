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
