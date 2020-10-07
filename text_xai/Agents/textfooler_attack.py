import numpy as np
import pandas as pd
import tensorflow as tf
from copy import deepcopy
from text_xai.TextModels.TextModel import TextModel
from text_xai.TextModels.TransferBert import TransferBertTextModel
from text_xai.TextModels.WordLSTM import WordLSTM
from text_xai.TextModels.E2EBert import E2EBertTextModel
from text_xai.Environments.utils.action_utils import replace_with_synonym, get_similarity, replace_with_synonym_greedy, possible_actions
from text_xai.Config.Config import Config


def rank_word_importance(sent: str, text_model: TextModel) -> list:
    words = sent.split()
    orig_probs = text_model.predict_proba(sent)[0]
    orig_pred = np.argmax(orig_probs)
    orig_prob = orig_probs[orig_pred]
    new_sents = [' '.join(words[:i] + ['<oov>'] + words[i+1:]) for i in range(len(words))]
    new_probs = [orig_prob - text_model.predict_proba(new_sent)[0][orig_pred] for new_sent in new_sents]
    return list(reversed(np.argsort(new_probs)))


def attack_sent(sent: str, text_model: TextModel, max_turns: int, sess: tf.Session):
    word_importance = rank_word_importance(sent, text_model)
    orig_pred = np.argmax(text_model.predict_proba(sent)[0])
    legal_actions = possible_actions(sent)
    word_importance = [w for w in word_importance if w in legal_actions]
    cur_sent = deepcopy(sent)
    for word_index in word_importance:
        cur_sent = replace_with_synonym(cur_sent, word_index, sess)
        if text_model.predict(cur_sent)[0] != orig_pred:
            return cur_sent, get_similarity([sent, cur_sent], sess)[0]

    return cur_sent, 0


if __name__ == '__main__':
    # constants
    cfg = Config("../Config/constants.yml")
    base_path = cfg.params["base_path"]
    MAX_TURNS = cfg.params["MAX_TURNS"]
    model_type = cfg.params["MODEL_TYPE"]

    tf_sess = tf.Session()
    tf_sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    # load data and model
    data_path = base_path + '_sample.csv'
    text_model = None
    if model_type == "transfer":
        text_model = TransferBertTextModel(trained_model=base_path + '.pth')
    elif model_type == "e2e":
        text_model = E2EBertTextModel(trained_model=base_path + 'e2e_bert.pth')
    elif model_type == "lstm":
        text_model = WordLSTM(trained_model=base_path + '_word_lstm.pth')
    else:
        print("please choose a valid kind of TEXT_MODEL!")
        exit(1)

    # move to cuda
    text_model.bert_model.cuda()
    text_model.model.cuda()

    # attack
    df = pd.read_csv(data_path)
    df['best_sent'] = ''
    df['max_score'] = 0.
    for n in range(500):
        cur_df = df.iloc[n:n+1]
        sent_list = list(cur_df.content.values)
        se = sent_list[0]
        best_sent, sim_score = attack_sent(se, text_model, MAX_TURNS, tf_sess)
        print(n, se, best_sent, sim_score)
        df.at[n, 'best_sent'] = best_sent
        df.at[n, 'max_score'] = sim_score

    print(df.head())
    # save results
    df.to_csv(base_path + f'_mit_{model_type}.csv', index=False)

