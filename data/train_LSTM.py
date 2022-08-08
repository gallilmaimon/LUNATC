import argparse
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from torch import nn
from collections import defaultdict

# add base path so that can import other files from project
import os
import sys
LIB_DIR = os.path.abspath(__file__).split('data')[0]
sys.path.insert(1, LIB_DIR)

from src.Attacks.utils.optim_utils import seed_everything
from src.TextModels.text_model_utils import load_embedding


def pad_sequences(inp, maxlen, token=0):
    if len(inp) >= maxlen:
        return inp[:maxlen-1] + [inp[-1]]
    else:
        return inp + [token]*(maxlen - len(inp))


def val_metrics(model, dataloader, device, criterion):
    running_accuracy = 0.0
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # forward
        inputs = inputs.to(device)
        with torch.no_grad():
            model.eval()
            outputs = model(inputs)
        labels = labels.to(device)
        loss = criterion(outputs, labels.view(-1))

        _, preds = outputs.max(1)
        running_accuracy += ((preds == labels.view(-1)).sum().to(dtype=torch.float)/len(outputs)).item()
        running_loss += loss.item()

    print('[%d, %5d] val loss: %.3f' % (1, i + 1, running_loss / (i+1)))
    print('[%d, %5d] val accuracy: %.3f' %(1, i + 1, running_accuracy / (i+1)))


class wordLSTMDataset(Dataset):
    def __init__(self, words, text_df, maxlen=128):
        self.word2ind = defaultdict(lambda: len(words))  # this is the out of vocabulary token
        self.maxlen = maxlen           # fixed length for padding sequences
        self.pad_ind = len(words) + 1  # a padding index so that all inputs are the same length
        self.word2ind.update({word: i for i, word in enumerate(words)})
        self.text_df = text_df

    def __getitem__(self, i):
        row = self.text_df.iloc[i]
        text, label = row.content, row.label
        word_tokens = [self.word2ind[w] for w in text.split()]
        return torch.Tensor(pad_sequences(word_tokens, self.maxlen, self.pad_ind)).long(), label

    def __len__(self):
        return len(self.text_df)


class WordLSTM(nn.Module):
    def __init__(self, embedding, hidden_size=150, depth=1, dropout=0.3, nclasses=2, fix_emb=True,
                 normalise=False):
        super(WordLSTM, self).__init__()

        if normalise:
            embedding /= np.linalg.norm(embedding,axis=1).reshape(-1, 1)

        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(embedding.shape[0]+2, embedding.shape[1])
        self.embedding.weight.data.uniform_(-0.25, 0.25)

        self.embedding.weight.data[:len(embeddings)].copy_(torch.from_numpy(embeddings))

        if fix_emb:
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding.shape[1], hidden_size//2, depth, dropout=dropout, bidirectional=True, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.out = nn.Linear(hidden_size, nclasses)

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.drop(emb)
        output, hidden = self.lstm(emb)
        output = torch.tanh(self.bn(torch.max(output, dim=1)[0]))
        output = self.drop(output)
        return self.out(output)


def train(args, model, words):
    df_train = pd.read_csv(args.data_path+"_train_clean.csv")
    df_train, df_val = train_test_split(df_train, test_size=args.val_size)
    df_test = pd.read_csv(args.data_path+"_test_clean.csv")

    dataset_train = wordLSTMDataset(words, df_train, maxlen=args.seq_len)
    dataset_val = wordLSTMDataset(words, df_val, maxlen=args.seq_len)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=8, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=8, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.n_epochs):  # loop over the dataset multiple times
        running_accuracy = 0.0
        running_loss = 0.0
        for i, data in enumerate(dataloader_train, 0):
            model.train()
            inputs, labels = data
            optimizer.zero_grad()

            inputs = inputs.to(args.device)
            outputs = model(inputs)
            labels = labels.to(args.device)
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            optimizer.step()

            _, preds = outputs.max(1)
            running_accuracy += ((preds == labels.view(-1)).sum().to(dtype=torch.float)/len(outputs)).item()

            # print statistics
            n = 200
            running_loss += loss.item()
            if i % n == n-1:    # print every n mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / n))
                running_loss = 0.0

                print('[%d, %5d] accuracy: %.3f' % (epoch + 1, i + 1, running_accuracy / n))
                running_accuracy = 0.0

        val_metrics(model, dataloader_val, args.device, criterion)

    torch.save(model.state_dict(), open(args.data_path+'_word_lstm.pth', 'wb'))


def infer(args, model, words):
    model_path = args.data_path + '_word_lstm.pth'
    tst_path = args.data_path + '_test_clean.csv'
    out_path = args.data_path + '_test_pred_lstm.csv'

    model.load_state_dict(torch.load(model_path))

    tst_df = pd.read_csv(tst_path)
    tst_df = tst_df.dropna()
    test_sent_dataset = wordLSTMDataset(words, tst_df)
    valid_dataloader = DataLoader(test_sent_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)

    pred_list = []
    for data in valid_dataloader:
        inputs, labels = data
        inputs = inputs.to(args.device)
        with torch.no_grad():
            model.eval()
            outputs = model(inputs)
        _, preds = outputs.max(1)
        pred_list.append(preds.cpu().numpy())

    tst_df['preds'] = np.concatenate(pred_list)
    print('\nModel test accuracy is: ', (tst_df.preds == tst_df.label).mean())
    tst_df.to_csv(out_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='Whether to train or inference in [\'train\', \'infer\']')
    parser.add_argument('--data_path', default='data/aclImdb/imdb', help='Path to data')
    parser.add_argument('--embedding_path', default='resources/word_vectors/glove.6B.200d.txt', help='Path to data')
    parser.add_argument('--device', default='cuda:0', help='Device to run on')
    parser.add_argument('--seed', type=int, default=42, help='random seed, use -1 for non-determinism')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for train and inference')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate of the AdamW optimiser')
    parser.add_argument('--n_epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument('--val_size', type=float, default=.1, help='relative size of the validation from the train set')
    parser.add_argument('--seq_len', type=int, default=128, help='The number of tokens to enter the model')
    args = parser.parse_args()

    seed_everything(args.seed)

    words, embeddings = load_embedding(args.embedding_path)
    model = WordLSTM(embeddings)
    model.to(args.device)

    if args.mode == 'train':
        train(args, model, words)
    elif args.mode == 'infer':
        infer(args, model, words)
