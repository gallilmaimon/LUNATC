import numpy as np
import pandas as pd
import tensorflow as tf
from copy import deepcopy

# add base path so that can import other files from project
import os
import sys
LIB_DIR = os.path.abspath(__file__).split('src')[0]
sys.path.insert(1, LIB_DIR)

from src.TextModels.TextModel import TextModel
from src.TextModels.WordLSTM import WordLSTM
from src.TextModels.Bert import BertTextModel
from src.TextModels.XLNet import XLNetTextModel
from src.Environments.utils.action_utils import replace_with_synonym, get_similarity, replace_with_synonym_greedy, possible_actions
from src.Config.Config import Config
from src.Attacks.utils.pwws_utils import softmax


def calc_word_importance(sent: str, text_model: TextModel, imp_type: str = 'tf', sess: tf.Session = None) -> list:
    if type(sent) == tuple:
        words = sent[1].split()
    else:
        words = sent.split()
    orig_probs = softmax(text_model.predict_proba(sent)[0])
    orig_pred = np.argmax(orig_probs)
    orig_prob = orig_probs[orig_pred]
    new_sents = [' '.join(words[:i] + ['<oov>'] + words[i + 1:]) for i in range(len(words))]
    if type(sent) == tuple:
        new_sents = [(sent[0], t) for t in new_sents]
    tf_imp = [orig_prob - softmax(text_model.predict_proba(new_sent)[0])[orig_pred] for new_sent in new_sents]
    if imp_type == 'tf':
        return tf_imp
    elif imp_type == 'pwws':
        new_sents = [replace_with_synonym_greedy(sent, i, text_model, sess) for i in range(len(words))]
        word_imp = [orig_prob - softmax(text_model.predict_proba(new_sent)[0])[orig_pred] for new_sent in new_sents]
        return (softmax(np.array(tf_imp)) * np.array(word_imp)).tolist()


def rank_word_importance(sent: str, text_model: TextModel, imp_type: str = 'tf', sess: tf.Session = None) -> list:
    new_probs = calc_word_importance(sent, text_model, imp_type, sess)
    return list(reversed(np.argsort(new_probs)))


def attack_sent(sent: str, text_model: TextModel, attack_type: str, max_turns: int, sess: tf.Session):
    word_importance = rank_word_importance(sent, text_model, attack_type, sess)
    orig_pred = np.argmax(text_model.predict_proba(sent)[0])
    legal_actions = possible_actions(sent)
    word_importance = [w for w in word_importance if w in legal_actions]
    cur_sent = deepcopy(sent)
    for word_index in word_importance:
        cur_sent = replace_with_synonym_greedy(cur_sent, word_index, text_model, sess)
        if text_model.predict(cur_sent)[0] != orig_pred:
            return cur_sent, get_similarity([sent[1], cur_sent[1]], sess)[0]

    return cur_sent, 0


if __name__ == '__main__':
    # constants
    cfg = Config(LIB_DIR + "src/Config/constants.yml")
    base_path = cfg.params["base_path"]
    MAX_TURNS = cfg.params["MAX_TURNS"]
    model_type = cfg.params["MODEL_TYPE"]
    num_classes = cfg.params["NUM_CLASSES"]  # Relevant only for BERT - 3 for MNLI, and 2 otherwise
    attack_type = 'tf'

    tf_sess = tf.Session()
    tf_sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    # load data and model
    data_path = base_path + f'_sample_{model_type}.csv'

    text_model = None
    if model_type == "bert":
        text_model = BertTextModel(num_classes=num_classes, trained_model=base_path + '_bert.pth')
    elif model_type == "lstm":
        text_model = WordLSTM(trained_model=base_path + '_word_lstm.pth')
    elif model_type == "xlnet":
        text_model = XLNetTextModel(trained_model=base_path + '_xlnet.pth')
    else:
        print("please choose a valid kind of TEXT_MODEL!")
        exit(1)

    # move to cuda
    text_model.bert_model.cuda()
    text_model.model.cuda()

    # attack
    df = pd.read_csv(data_path)

    # used for 2 text tasks like NLI
    if "content2" in df.columns:
        df["content"] = list(zip(df.content, df.content2))
        df = df.drop("content2", 1)

    df.drop_duplicates('content', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['best_sent'] = ''
    df['max_score'] = 0.

    for n in range(len(df)):
        cur_df = df.iloc[n:n+1]
        sent_list = list(cur_df.content.values)
        se = sent_list[0]
        best_sent, sim_score = attack_sent(se, text_model, attack_type, MAX_TURNS, tf_sess)
        print(n, se, best_sent, sim_score)
        df.at[n, 'best_sent'] = best_sent
        df.at[n, 'max_score'] = sim_score

    print(df.head())
    # save results
    df.to_csv(base_path + f'_{attack_type}_{model_type}.csv', index=False)

