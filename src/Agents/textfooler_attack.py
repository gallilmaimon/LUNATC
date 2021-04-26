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
from src.TextModels.TransferBert import TransferBertTextModel
from src.TextModels.WordLSTM import WordLSTM
from src.TextModels.E2EBert import E2EBertTextModel
from src.Environments.utils.action_utils import replace_with_synonym, get_similarity, replace_with_synonym_greedy, possible_actions
from src.Config.Config import Config


def calc_word_importance(sent: str, text_model: TextModel) -> list:
    words = sent.split()
    orig_probs = text_model.predict_proba(sent)[0]
    orig_pred = np.argmax(orig_probs)
    orig_prob = orig_probs[orig_pred]
    new_sents = [' '.join(words[:i] + ['<oov>'] + words[i + 1:]) for i in range(len(words))]
    return [orig_prob - text_model.predict_proba(new_sent)[0][orig_pred] for new_sent in new_sents]


def rank_word_importance(sent: str, text_model: TextModel) -> list:
    new_probs = calc_word_importance(sent, text_model)
    return list(reversed(np.argsort(new_probs)))


def attack_sent(sent: str, text_model: TextModel, max_turns: int, sess: tf.Session):
    word_importance = rank_word_importance(sent, text_model)
    orig_pred = np.argmax(text_model.predict_proba(sent)[0])
    legal_actions = possible_actions(sent)
    word_importance = [w for w in word_importance if w in legal_actions]
    cur_sent = deepcopy(sent)
    for word_index in word_importance:
        cur_sent = replace_with_synonym_greedy(cur_sent, word_index, text_model, sess)
        if text_model.predict(cur_sent)[0] != orig_pred:
            return cur_sent, get_similarity([sent, cur_sent], sess)[0]

    return cur_sent, 0


if __name__ == '__main__':
    # constants
    cfg = Config(LIB_DIR + "src/Config/DQN_constants.yml")
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
    # df = df[df.preds == df.label]
    df.drop_duplicates('content', inplace=True)
    # df = df.iloc[[3225, 3228, 3233, 3235, 3241, 3247, 3250, 3258, 3261, 3283, 3284, 3286, 3289, 3299, 3300, 3306, 3307, 3310, 3314, 3321, 3322, 3327, 3328, 3331, 3333, 3334, 3339, 3341, 3342, 3343, 3351, 3353, 3354, 3356, 3357, 3366, 3369, 3371, 3372, 3375, 3390, 3395, 3400, 3401, 3402, 3404, 3411, 3419, 3420, 3425, 3434, 3438, 3439, 3446, 3449, 3451, 3456, 3468, 3476, 3489, 3503, 3505, 3509, 3516, 3520, 3523, 3531, 3534, 3535, 3541, 3542, 3553, 3562, 3570, 3571, 3572, 3582, 3586, 3588, 3590, 3593, 3594, 3595, 3596, 3599, 3601, 3614, 3615, 3620, 3634, 3647, 3649, 3668, 3670, 3674, 3677, 3678, 3681, 3684, 3687, 3688, 3689, 3694, 3703, 3704, 3706, 3708, 3709, 3710, 3711, 3712, 3715, 3718, 3723, 3731, 3738]]
    df['best_sent'] = ''
    df['max_score'] = 0.
    for n in range(len(df)):
        cur_df = df.iloc[n:n+1]
        sent_list = list(cur_df.content.values)
        se = sent_list[0]
        best_sent, sim_score = attack_sent(se, text_model, MAX_TURNS, tf_sess)
        print(n, se, best_sent, sim_score)
        df.at[n, 'best_sent'] = best_sent
        df.at[n, 'max_score'] = sim_score

    print(df.head())
    # save results
    # df.to_csv(base_path + f'_mit_{model_type}.csv', index=False)

