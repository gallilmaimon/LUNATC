import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# add base path so that can import other files from project
import os
import sys
LIB_DIR = os.path.abspath(__file__).split('src')[0]
sys.path.insert(1, LIB_DIR)

from src.TextModels.E2EBert import E2EBertTextModel
from src.TextModels.TransferBert import TransferBertTextModel
from src.TextModels.WordLSTM import WordLSTM
from src.Agents.Normalizers.norm_utils import get_normaliser
from src.Agents.DQNAgent import DQNAgent
from src.Agents.ContinuousDQNAgent import ContinuousDQNAgent
from src.Agents.utils.vis_utils import running_mean

# configuration
from src.Config.Config import Config

# region constants
cfg = Config(LIB_DIR + "src/Config/DQN_constants.yml")
base_path = cfg.params["base_path"]
# endregion constants


def attack_individually(model_type: str = "e2e"):
    """
    this function performs the attack on each sentence individually - by retraining the model from scratch each time.
    the different parameters are read from the constants at the top of the file
    """
    # initialise parameters
    device = cfg.params["DEVICE"]
    state_shape = cfg.params["STATE_SHAPE"]
    norm_rounds = cfg.params["NORMALISE_ROUNDS"]
    offline_normalising = True if norm_rounds == 'offline' else False

    assert model_type in ["e2e", "transfer", 'lstm'], "model type unrecognised or unsupported!"

    # define text model
    text_model = None  # just to make sure it is not somehow referenced before assignment
    if model_type == "transfer":
        text_model = TransferBertTextModel(trained_model=base_path + '.pth')
    elif model_type == "e2e":
        text_model = E2EBertTextModel(trained_model=base_path + 'e2e_bert.pth', device=device)
    elif model_type == "lstm":
        text_model = WordLSTM(trained_model=base_path + '_word_lstm.pth', device=device)

    # generate data
    data_path = base_path + '_sample.csv'
    df = pd.read_csv(data_path)

    for n in eval(cfg.params['ATTACKED_INDICES']):
        # get current text
        cur_df = df.iloc[n:n + 1]
        sent_list = list(cur_df.content.values)
        print('original text', sent_list[0])
        print('original class', cur_df.label.values[0])
        print('original prediction', cur_df.preds.values[0])

        # normaliser
        norm_states = None
        if offline_normalising:  # If using offline normalising with the initial states
            norm_states = np.empty((len(cur_df), 768))
            for j, text in enumerate(sent_list):
                norm_states[j] = text_model.embed(text).cpu()  # todo: make more efficient with batch processing

        norm = get_normaliser(state_shape, norm_rounds, norm_states, None, device=device) if norm_rounds != -1 else None

        # agent
        n_actions = len(sent_list[0].split())
        dqn = None
        if cfg.params['AGENT_TYPE'] == 'dqn':
            dqn = DQNAgent(sent_list, text_model, n_actions, norm, device)
        elif cfg.params['AGENT_TYPE'] == 'dqn_contin':
            dqn = ContinuousDQNAgent(sent_list, text_model, n_actions, norm, device)
        else:
            print("illegal AGENT_TYPE selected! choose one of ['dqn', 'dqn_contin']")
            exit(0)

        try:
            dqn.train_model(cfg.params['NUM_EPISODES'])
            plt.plot(dqn.rewards)
            plt.plot(running_mean(dqn.rewards, 100))
            plt.show()
        except KeyboardInterrupt:
            plt.plot(dqn.rewards)
            plt.plot(running_mean(dqn.rewards, 100))
            plt.show()
            exit(0)


def pretrain_attack_model(epoch=0, model_type: str = "e2e"):
    """this model pretrains a single network on the data given"""
    # initialise parameters
    device = cfg.params["DEVICE"]
    state_shape = cfg.params["STATE_SHAPE"]
    n_actions = cfg.params["MAX_SENT_LEN"]
    norm_rounds = cfg.params["NORMALISE_ROUNDS"]
    offline_normalising = True if norm_rounds == 'offline' else False

    assert model_type in ["e2e", "transfer", 'lstm'], "model type unrecognised or unsupported!"

    print(f"Starting epoch number: {epoch}")
    # generate data
    data_path = base_path + '_sample.csv'
    df = pd.read_csv(data_path)
    # take smaller subset
    df = df.iloc[eval(cfg.params['ATTACKED_INDICES'])]
    print(len(df))
    sent_list = list(df.content.values)
    print(sent_list)

    # define text model
    text_model = None  # just to make sure it is not somehow referenced before assignment
    if model_type == "transfer":
        text_model = TransferBertTextModel(trained_model=base_path + '.pth')
    elif model_type == "e2e":
        text_model = E2EBertTextModel(trained_model=base_path + 'e2e_bert.pth', device=device)
    elif model_type == "lstm":
        text_model = WordLSTM(trained_model=base_path + '_word_lstm.pth', device=device)

    # normaliser
    norm_states = None
    if offline_normalising:  # If using offline normalising with the initial states
        norm_states = np.empty((len(df), 768))
        for j, text in enumerate(sent_list):
            norm_states[j] = text_model.embed(text).cpu()  # todo: make more efficient with batch processing

    norm = get_normaliser(state_shape, norm_rounds, norm_states, None, device=device) if norm_rounds != -1 else None

    # define agent
    dqn = None
    if cfg.params['AGENT_TYPE'] == 'dqn':
        dqn = DQNAgent(sent_list, text_model, n_actions, norm, device)
    elif cfg.params['AGENT_TYPE'] == 'dqn_contin':
        dqn = ContinuousDQNAgent(sent_list, text_model, n_actions, norm, device)
    else:
        print("illegal AGENT_TYPE selected! choose one of ['dqn', 'dqn_contin']")
        exit(0)

    # train
    try:
        dqn.train_model(cfg.params['NUM_EPISODES'])
        plt.plot(dqn.rewards)
        plt.plot(running_mean(dqn.rewards, 100))
        plt.show()
    except KeyboardInterrupt:
        plt.plot(dqn.rewards)
        plt.plot(running_mean(dqn.rewards, 100))
        plt.show()
        exit(0)


if __name__ == "__main__":
    attack_type = cfg.params['ATTACK_TYPE']
    # attack each text separately
    if attack_type == 'individual':
        attack_individually(model_type=cfg.params['MODEL_TYPE'])

    elif attack_type == 'universal':
        general_start = time.time()
        for epoch in range(1):
            start = time.time()
            pretrain_attack_model(model_type=cfg.params['MODEL_TYPE'], epoch=epoch)
            print('time', time.time() - start)
        print('total time', time.time() - general_start)
