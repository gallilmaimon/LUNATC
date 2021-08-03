import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from torch import nn
import tensorflow as tf
from copy import deepcopy
import pickle

# add base path so that can import other files from project
import os
import sys
LIB_DIR = os.path.abspath(__file__).split('src')[0]
sys.path.insert(1, LIB_DIR)

from src.Environments.utils.action_utils import replace_with_synonym_greedy, get_similarity, possible_actions

# importing the text model attacked and used for embedding
from src.TextModels.Bert import BertTextModel
from src.TextModels.WordLSTM import WordLSTM

# importing the textfooler word_importance
from src.Attacks.textfooler_attack import calc_word_importance

# fix randomness
from src.Attacks.utils.optim_utils import seed_everything


class GenFooler(nn.Module):
    def __init__(self):
        super(GenFooler, self).__init__()
        self.fc1 = nn.Linear(768, 400, bias=True)
        self.fc1.bias.data.fill_(0)
        self.fc2 = nn.Linear(400, 400, bias=True)
        self.fc2.bias.data.fill_(0)
        self.fc21 = nn.Linear(400, 400, bias=True)
        self.fc21.bias.data.fill_(0)
        self.fc22 = nn.Linear(400, 400, bias=True)
        self.fc22.bias.data.fill_(0)
        self.fc23 = nn.Linear(400, 400, bias=True)
        self.fc23.bias.data.fill_(0)
        self.fc24 = nn.Linear(400, 400, bias=True)
        self.fc24.bias.data.fill_(0)
        self.fc3 = nn.Linear(400, 150, bias=True)
        self.fc3.bias.data.fill_(0)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = x.view(-1, 768)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc21(x))
        x = self.act(self.fc22(x))
        x = self.act(self.fc23(x))
        x = self.act(self.fc24(x))
        x = self.fc3(x)
        return x


def prepare_data(path, df_train, df_test, text_model, use_pre_calculated=False, imp_type='tf', sess=None):
    if use_pre_calculated:
        with open(path + f'_genfooler_{model_type}/data_{imp_type}.pkl', 'rb') as f:
            return pickle.load(f)

    # embed all the texts
    full_train_emb = embed_all_texts(text_model, df_train.content.values.tolist())
    test_emb = embed_all_texts(text_model, df_test.content.values.tolist())
    print("Embedding calculated for all texts")

    # generate masks indicating how many words each text has
    full_train_mask = mask_all_texts(df_train.content.values.tolist())
    test_mask = mask_all_texts(df_test.content.values.tolist())
    print("Masks calculated for all texts")

    # generate textfooler importance labels
    full_train_imp = importance_all_texts(text_model, df_train.content.values.tolist(), imp_type, sess)
    test_imp = importance_all_texts(text_model, df_test.content.values.tolist(), imp_type, sess)
    print("Importance calculated for all texts")

    with open(path + f'_genfooler_{model_type}/data_{imp_type}.pkl', 'wb') as f:
        pickle.dump((full_train_emb, test_emb, full_train_mask, test_mask, full_train_imp, test_imp), f)

    return full_train_emb, test_emb, full_train_mask, test_mask, full_train_imp, test_imp


def embed_all_texts(text_model: BertTextModel, texts: list):
    # maybe change to batch proccessing in the future
    emb_list = []
    for text in texts:
        emb_list.append(text_model.embed(text))
    return torch.cat(emb_list)


def mask_all_texts(texts: list):
    mask_list = []
    for text in texts:
        mask_list.append(zero_pad([1] * len(text.split())).reshape(1, -1))
    return torch.tensor(np.concatenate(mask_list))


def importance_all_texts(text_model: BertTextModel, texts: list, imp_type: str, sess: tf.Session):
    imp_list = []
    for text in texts:
        imp_list.append(zero_pad(calc_word_importance(text, text_model, imp_type, sess)).reshape(1, -1))
    return torch.tensor(np.concatenate(imp_list))


def pred_importance_all_texts(model, dataloader):
    imp_pred_list = []
    model.eval()
    for batch in dataloader:
        inp = batch[0].cuda()
        with torch.no_grad():
            imp_pred_list.append(model(inp).cpu().numpy())

    return np.concatenate(imp_pred_list)


def zero_pad(val_list, fixed_size=150):
    return np.pad(np.array(val_list), (0, fixed_size-len(val_list)))


def train_epoch(model, train_dl, val_dl, criterion, optim, device='cuda'):
    train_epoch_loss = 0
    for batch in train_dl:
        model.zero_grad()
        inp, label, mask = batch[0], batch[1], batch[2]
        inp = inp.to(device)
        label = label.float().to(device)
        mask = mask.to(device)

        pred = model(inp)
        mse = criterion(pred, label)
        loss = ((mse * mask).sum(axis=1) / mask.sum(axis=1)).mean()
        loss.backward()
        optim.step()
        train_epoch_loss += loss.detach().cpu()

    val_epoch_loss = None
    if val_dl is not None:
        val_epoch_loss = 0
        for batch in val_dl:
            inp, label, mask = batch[0], batch[1], batch[2]
            inp = inp.to(device)
            label = label.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                pred = model(inp)
                mse = criterion(pred, label)
                loss = ((mse * mask).sum(axis=1) / mask.sum(axis=1)).mean()

            val_epoch_loss += loss.detach().cpu()
    return train_epoch_loss, val_epoch_loss


def train(train_emb, train_mask, train_imp, val_emb=None, val_mask=None, val_imp=None, n_epochs=10000, device='cuda',
          bs=16, lr=1e-6, n_early_stop_rounds=10):
    # build model
    model = GenFooler()
    model.to(device)

    # build dataset & dataloader
    train_dataset = TensorDataset(train_emb, train_imp, train_mask)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=bs)
    val_dataloader = None
    if val_emb is not None and val_imp is not None and val_mask is not None:
        val_dataset = TensorDataset(val_emb, val_imp, val_mask)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=bs)

    # define loss & optimiser
    criterion = nn.MSELoss(reduction='none')
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # start training
    best_epoch = None
    best_loss = 1000000
    early_stop_counter = 0
    for epoch in range(n_epochs):
        print("--------------------------------------------")
        train_loss, val_loss = train_epoch(model, train_dataloader, val_dataloader, criterion, optim, device)
        print(f"Epoch {epoch} - train loss: {train_loss}, val_loss: {val_loss}")

        # early stopping
        if val_loss is not None and val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if early_stop_counter >= n_early_stop_rounds:
            return model, best_epoch, best_loss

    return model, best_epoch, best_loss


def attack_sent(sent: str, text_model: BertTextModel, max_turns: int, sess: tf.Session, word_importance: np.array):
    word_rank = list(reversed(np.argsort(word_importance)))
    orig_pred = np.argmax(text_model.predict_proba(sent)[0])
    legal_actions = possible_actions(sent)
    word_rank = [w for w in word_rank if w in legal_actions]
    cur_sent = deepcopy(sent)
    for word_index in word_rank[:max_turns]:
        cur_sent = replace_with_synonym_greedy(cur_sent, word_index, text_model, sess)
        if text_model.predict(cur_sent)[0] != orig_pred:
            print((np.array(cur_sent.split()) != np.array(sent.split())).sum())
            return cur_sent, get_similarity([sent, cur_sent], sess)[0]

    return cur_sent, 0


if __name__ == '__main__':
    # constants
    path = '../../data/aclImdb/imdb'
    N_SPLITS = 5
    MAX_TURNS = 300000
    SEED = 42
    model_type = 'bert'
    train_size = 5  # this lets us load the pre-computed vectors and just sub-sample them for efficiency
    attack_type = 'pwws'  # whether we are generalising textfooler or PWWS
    pre_calc_data = False  # whether to use pre-calculated data (embedding, masks and word - importance)

    train_inds = [0, 1, 3, 10, 11]
    test_inds = [13, 17, 35, 83, 103]

    if attack_type == 'pwws':
        tf_sess = tf.Session()
        tf_sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    else:
        tf_sess = None
    os.makedirs(path + f'_genfooler_{model_type}', exist_ok=True)

    # read data
    df = pd.read_csv(path + f'_sample_{model_type}.csv')
    df.drop_duplicates('content', inplace=True)

    text_model = None
    if model_type == 'bert':
        text_model = BertTextModel(trained_model=path + '_bert.pth')
    elif model_type == 'lstm':
        text_model = WordLSTM(trained_model=path + '_word_lstm.pth')
    else:
        print('non-existent model type selected, please select one of ["lstm", "bert"]')
        exit()

    df_train = df.iloc[train_inds]
    df_test = df.iloc[test_inds]
    print('model and data loaded')

    full_train_emb, test_emb, full_train_mask, test_mask, full_train_imp, test_imp = \
        prepare_data(path, df_train, df_test, text_model, use_pre_calculated=pre_calc_data, imp_type=attack_type,
                     sess=tf_sess)

    full_train_emb, full_train_mask, full_train_imp = \
        full_train_emb[:train_size], full_train_mask[:train_size], full_train_imp[:train_size]
    print("data pre - processed")

    test_dataset = TensorDataset(test_emb, test_imp, test_mask)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=16)

    # fix seed
    seed_everything(SEED)
    print("seeds fixed, running CV for choosing the number of epochs")

    # train in cross-validation to choose the right number of epochs
    kfold = KFold(shuffle=True, n_splits=N_SPLITS)
    stop_epoch = []
    for train_ind, val_ind in kfold.split(full_train_emb):
        print('****************** New Fold ***************')
        train_emb, train_imp, train_mask = full_train_emb[train_ind], full_train_imp[train_ind], full_train_mask[
            train_ind]
        val_emb, val_imp, val_mask = full_train_emb[val_ind], full_train_imp[val_ind], full_train_mask[val_ind]
        _, best_epoch, _ = train(train_emb, train_mask, train_imp, val_emb, val_mask, val_imp)
        stop_epoch.append(best_epoch)
    print(f"CV finished, number of epochs chosen: {int(sum(stop_epoch)/len(stop_epoch))}")

    # train the final model on all the training data for the right number of epochs
    model, _, _ = train(full_train_emb, full_train_mask, full_train_imp, n_epochs=int(sum(stop_epoch)/len(stop_epoch)),
                        n_early_stop_rounds=100000000000)
    print("final model finished training!")

    # predict the word importance on the test texts
    test_imp_pred = pred_importance_all_texts(model, test_dataloader)
    print("Word importance predicted, performing the attack")

    # attack using the predicted importance
    tf_sess = tf.Session()
    tf_sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    sim_scores = []
    best_sents = []
    for i, text in enumerate(df_test.content.values.tolist()):
        pred_imp = test_imp_pred[i][:len(text.split())]
        best_sent, sim_score = attack_sent(text, text_model, MAX_TURNS, tf_sess, pred_imp)
        print(f"------{i}--- score: {sim_score}-----")
        best_sents.append(best_sent)
        sim_scores.append(sim_score)

    # save results
    print("Attack finished! saving results...")
    df_test['max_score'] = sim_scores
    df_test['best_sent'] = best_sents
    df_test.to_csv(path + f'_genfooler_{model_type}/attack_{attack_type}{train_size}_{SEED}.csv', index=False)
    torch.save(model.state_dict(), path + f'_genfooler_{model_type}/model_{attack_type}{train_size}_{SEED}.pth')


