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
from src.TextModels.E2EBert import E2EBertTextModel
from src.Environments.utils.action_utils import replace_with_synonym, replace_with_synonym_greedy, possible_actions, possible_synonyms
from src.Attacks.utils.pwws_utils import softmax
from src.Config.Config import Config
from src.Attacks.utils.optim_utils import seed_everything


def crossover(text1, text2):
    words1 = np.array(deepcopy(text1).split())
    words2 = np.array(deepcopy(text2).split())

    assert len(words1) == len(words2), "number of words in cross-over texts does not match!"
    new_words = np.where(np.random.random(len(words1)) > 0.5, words1, words2)
    return ' '.join(new_words)


def attack_sent(sent: str, text_model: TextModel, sess: tf.Session, pop_size: int = 20, max_generation: int = 100):
    orig_pred = np.argmax(text_model.predict_proba(sent)[0])
    cur_text = deepcopy(sent)
    target = 1 - orig_pred
    legal_actions = possible_actions(sent)
    words = sent.split()
    num_options = np.array([len(possible_synonyms(w)) if i in legal_actions else 0 for i, w in enumerate(words)])
    w_select_probs = num_options / sum(num_options)
    selected_i = np.random.choice(len(words), pop_size, p=w_select_probs)
    pop = [replace_with_synonym_greedy(cur_text, selected_i[i], text_model, sess) for i in range(pop_size)]

    for i in range(max_generation):
        pop_preds = softmax(text_model.predict_proba(pop), axis=1)
        pop_scores = pop_preds[:, target]
        best_ind = np.argmax(pop_scores)
        best_adv = pop[best_ind]
        if pop_scores[best_ind] > 0.5:
            return best_adv, 1

        select_prob = softmax(pop_scores, temp=0.3)
        parent1 = np.random.choice(pop_size, pop_size - 1, p=select_prob)
        parent2 = np.random.choice(pop_size, pop_size - 1, p=select_prob)
        selected_i = np.random.choice(len(words), pop_size, p=w_select_probs)
        children = [crossover(pop[parent1[i]], pop[parent2[i]]) for i in range(len(parent1))]
        pop = [replace_with_synonym_greedy(c, selected_i[i], text_model, sess) for i, c in enumerate(children)] + [best_adv]

    return best_adv, 0


if __name__ == '__main__':
    # constants
    cfg = Config(LIB_DIR + "src/Config/DQN_constants.yml")
    base_path = cfg.params["base_path"]
    MAX_TURNS = cfg.params["MAX_TURNS"]
    model_type = cfg.params["MODEL_TYPE"]

    seed_everything(42)

    tf_sess = tf.Session()
    tf_sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    # load data and model
    data_path = base_path + f'_sample_{model_type}.csv'
    text_model = None
    if model_type == "e2e":
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
    df['best_sent'] = ''
    df['max_score'] = 0.
    for n in range(len(df)):
        cur_df = df.iloc[n:n+1]
        sent_list = list(cur_df.content.values)
        se = sent_list[0]
        best_sent, sim_score = attack_sent(se, text_model, tf_sess)
        print(n, se, best_sent, sim_score)
        df.at[n, 'best_sent'] = best_sent
        df.at[n, 'max_score'] = sim_score

    print(df.head())
    # save results
    df.to_csv(base_path + f'_mit_{model_type}.csv', index=False)

