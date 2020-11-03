import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# add base path so that can import other files from project
import os
import sys
LIB_DIR = os.path.abspath(__file__).split('src')[0]
sys.path.insert(1, LIB_DIR)

from src.TextModels.E2EBert import E2EBertTextModel
from src.Agents.Normalizers.norm_utils import get_normaliser
from src.Agents.DQNAgent import DQNAgent
from src.Agents.utils.vis_utils import running_mean

# configuration
from src.Config.Config import Config

# region constants
cfg = Config(LIB_DIR + "src/Config/DQN_constants.yml")
base_path = cfg.params["base_path"]
# endregion constants


# initialise parameters
model_type = "e2e"
device = cfg.params["DEVICE"]
num_rounds = cfg.params["NUM_EPISODES"]
state_shape = cfg.params["STATE_SHAPE"]
max_sent_len = cfg.params["MAX_SENT_LEN"]
lr = cfg.params["LEARNING_RATE"]
norm_rounds = cfg.params["NORMALISE_ROUNDS"]
env_type = cfg.params["ENV_TYPE"]
offline_normalising = True if norm_rounds == 'offline' else False

assert model_type in ["e2e", "transfer", 'lstm'], \
    "model type or agent type unrecognised or unsupported!"


# generate data
data_path = base_path + '_sample.csv'
df = pd.read_csv(data_path)
# take smaller subset
df = df.iloc[eval(cfg.params['ATTACKED_INDICES'])]
print(len(df))
sent_list = list(df.content.values)
print(sent_list)

# define language model
text_model = E2EBertTextModel(trained_model=base_path + 'e2e_bert.pth', device=device)


# normaliser
norm_states = None
if offline_normalising:  # If using offline normalising with the initial states
    norm_states = np.empty((len(df), 768))
    for j, text in enumerate(sent_list):
        norm_states[j] = text_model.embed(text).cpu()  # todo: make more efficient with batch processing

# multi-processing lock used for normalisation
norm_lock = None
norm = get_normaliser(state_shape, norm_rounds, norm_states, norm_lock, device=device) if norm_rounds != -1 else None

dqn = DQNAgent(sent_list, text_model, norm, device)

for j in range(5):
    dqn.train_model(cfg.params['NUM_EPISODES'])
    plt.plot(dqn.rewards)
    plt.plot(running_mean(dqn.rewards, 100))
    plt.show()
