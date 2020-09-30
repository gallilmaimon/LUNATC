import numpy as np
import pandas as pd
from copy import deepcopy

from text_xai.Agents.utils.pwws_utils import nlp, get_synonym_options, get_named_entity_replacements, \
    calculate_word_saliency, delta_p_star

import tensorflow as tf
from text_xai.TextModels.TransferBert import TransferBertTextModel
from text_xai.TextModels.E2EBert import E2EBertTextModel
from text_xai.TextModels.WordLSTM import WordLSTM
from text_xai.Environments.utils.action_utils import get_similarity
from text_xai.Config.Config import Config


def pwws_attack_text(text: str, lang_model, sess, dataset: str, use_ne: bool = False):
    doc = nlp(text)  # use spacy to calculate POS, named entities etc.
    orig_pred = np.argmax(lang_model.predict_proba(text)[0])

    words = [tok.text for tok in doc]
    oracle_usage = len(words)
    word_saliency = calculate_word_saliency(lang_model, deepcopy(words))
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
        sub, delta_p = delta_p_star(lang_model, deepcopy(words), i, rep_options)
        substitute_options.append((i, sub, delta_p*word_saliency[i]))

    sorted_substitutes = sorted(substitute_options, key=lambda t: t[2], reverse=True)

    # perturb the text iteratively according tho the found order and options until attack is succesful
    for (i, sub, _) in sorted_substitutes:
        words[i] = sub.lower()
        cur_sent = ' '.join(words)
        if lang_model.predict(cur_sent)[0] != orig_pred:
            return cur_sent, get_similarity([text, cur_sent], sess)[0], oracle_usage

    return ' '.join(words), 0, oracle_usage


if __name__ == '__main__':
    cfg = Config("../Config/constants.yml")
    base_path = cfg.params["base_path"]
    MAX_TURNS = cfg.params["MAX_TURNS"]
    model_type = cfg.params["MODEL_TYPE"]
    USE_NE = True  # whether to use the named entity replacement action
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
        print("please choose a valid kind of model type!")
        exit(1)

    # move to cuda
    text_model.bert_model.cuda()
    text_model.model.cuda()

    # attack
    df = pd.read_csv(data_path)
    df['best_sent'] = ''
    df['max_score'] = 0.
    df['oracle_usage'] = 0
    for n in range(500):
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
    df.to_csv(base_path + '_pwws.csv', index=False)
