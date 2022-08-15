import numpy as np
import pandas as pd
from copy import deepcopy

# add base path so that can import other files from project
import os
import sys
LIB_DIR = os.path.abspath(__file__).split('src')[0]
sys.path.insert(1, LIB_DIR)

from src.Attacks.utils.pwws_utils import nlp, get_synonym_options, get_named_entity_replacements, \
    calculate_word_saliency, delta_p_star

import tensorflow as tf
from src.TextModels.Bert import BertTextModel
from src.TextModels.WordLSTM import WordLSTM
from src.TextModels.XLNet import XLNetTextModel

from src.Environments.utils.action_utils import get_similarity
from src.Config.Config import Config


def pwws_attack_text(text: str, text_model, sess, dataset: str, use_ne: bool = False):
    doc = nlp(text)  # use spacy to calculate POS, named entities etc.
    orig_pred = np.argmax(text_model.predict_proba(text)[0])

    words = [tok.text for tok in doc]
    oracle_usage = len(words)
    word_saliency = calculate_word_saliency(text_model, deepcopy(words))
    substitute_options = []  # contains a list of tuples (ind, replacement_word, score)

    NE_options = get_named_entity_replacements()[dataset][orig_pred] if use_ne else ''

    # find best replacement for each word, and the importance of eac
    for i, token in enumerate(doc):
        # find replacement options (synonym or Named Entity)
        NER_tag = token.ent_type_ if use_ne else ''

        if use_ne and NER_tag in NE_options.keys():
            rep_options = [NE_options[NER_tag]]
        else:
            rep_options = get_synonym_options(token)

        if len(rep_options) == 0:
            continue

        oracle_usage += len(rep_options)
        sub, delta_p = delta_p_star(text_model, deepcopy(words), i, rep_options)
        substitute_options.append((i, sub, delta_p*word_saliency[i]))

    sorted_substitutes = sorted(substitute_options, key=lambda t: t[2], reverse=True)

    # perturb the text iteratively according tho the found order and options until attack is successful
    for (i, sub, _) in sorted_substitutes:
        words[i] = sub.lower()
        cur_sent = ' '.join(words)
        if text_model.predict(cur_sent)[0] != orig_pred:
            return cur_sent, get_similarity([text, cur_sent], sess)[0], oracle_usage

    return ' '.join(words), 0, oracle_usage


if __name__ == '__main__':
    cfg = Config(LIB_DIR + "src/Config/constants.yml")
    base_path = cfg.params["base_path"]
    MAX_TURNS = cfg.params["MAX_TURNS"]
    model_type = cfg.params["MODEL_TYPE"]
    USE_NE = True  # whether to use the named entity replacement action
    tf_sess = tf.Session()
    tf_sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    # load data and model
    data_path = base_path + f'_sample_{model_type}.csv'
    text_model = None
    if model_type == "bert":
        text_model = BertTextModel(trained_model=base_path + '_bert.pth')
    elif model_type == "lstm":
        text_model = WordLSTM(trained_model=base_path + '_word_lstm.pth')
    elif model_type == "xlnet":
        text_model = XLNetTextModel(trained_model=base_path + '_xlnet.pth')
    else:
        print("please choose a valid kind of model type!")
        exit(1)

    # move to cuda
    text_model.bert_model.cuda()
    text_model.model.cuda()

    # attack
    df = pd.read_csv(data_path)
    # df = df[df.preds == df.label]
    df['best_sent'] = ''
    df['max_score'] = 0.
    df['oracle_usage'] = 0
    for n in range(len(df)):
        cur_df = df.iloc[n:n + 1]
        sent_list = list(cur_df.content.values)
        se = sent_list[0]
        best_sent, sim_score, oracle_usage = pwws_attack_text(se, text_model, tf_sess, base_path.split('/')[-1], USE_NE)
        print(n, se, best_sent, sim_score)
        df.at[n, 'best_sent'] = best_sent
        df.at[n, 'max_score'] = sim_score
        df.at[n, 'oracle_usage'] = oracle_usage

    print(df.head())
    # save results
    out_name = '_pwws'
    out_name += model_type
    out_name += '_noNE' if not USE_NE else ''
    df.to_csv(base_path + out_name + '.csv', index=False)
