# region imports
import os
os.environ["OMP_NUM_THREADS"] = "1"

# add base path so that can import other files from project
import sys
LIB_DIR = os.path.abspath(__file__).split('text_xai')[0]
sys.path.insert(1, LIB_DIR)

# general
import time
import pandas as pd
import numpy as np
import pickle
import torch
import shutil

# agent
import torch.multiprocessing as mp
from text_xai.Agents.utils.shared_adam import SharedAdam
from text_xai.Agents.ReLeGATeAgent import ReLeGATeAgentNet, ReLeGATeAgentWorker, Normalizer
from text_xai.Agents.SearchAgent import SearchAgentWorker

# environment & language model
from text_xai.TextModels.E2EBert import E2EBertTextModel
from text_xai.TextModels.TransferBert import TransferBertTextModel
from text_xai.TextModels.WordLSTM import WordLSTM

# configuration
from text_xai.Config.Config import Config
# endregion imports

# region constants
cfg = Config(LIB_DIR + "text_xai/Config/constants.yml")
base_path = cfg.params["base_path"]
# endregion constants


def attack_individually(model_type: str = "e2e", agent_type: str = "relegate", device: str = "cpu"):
    """
    this function preforms the attack on each sentence individually - by retraining the model from scratch each time.
    the different paramaters are read from the constants at the top of the file
    """
    # check model type and agent type
    assert model_type in ["e2e", "transfer", 'lstm'] and agent_type in ["relegate", "search"], \
        "model type or agent type unrecognised or unsupported!"

    # initialise parameters
    num_rounds = cfg.params["NUM_EPISODES"]
    state_shape = cfg.params["STATE_SHAPE"]
    lr = cfg.params["LEARNING_RATE"]
    norm_rounds = cfg.params["NORMALISE_ROUNDS"]
    env_type = cfg.params["ENV_TYPE"]
    sync_start = cfg.params["SYNC_START"]
    num_workers = mp.cpu_count() if cfg.params["NUM_WORKERS"] == 'cpu_count' else cfg.params["NUM_WORKERS"]

    # load data
    data_path = base_path + '_sample.csv'
    df = pd.read_csv(data_path)

    # define language model
    text_model = None  # just to make sure it is not somehow referenced before assignment
    if model_type == "transfer":
        text_model = TransferBertTextModel(trained_model=base_path + '.pth')
    elif model_type == "e2e":
        text_model = E2EBertTextModel(trained_model=base_path + 'e2e_bert.pth', device=device)
    elif model_type == "lstm":
        text_model = WordLSTM(trained_model=base_path + '_word_lstm.pth', device=device)

    # multiproccessing lock used for normalisation
    norm_lock = mp.Lock() if norm_rounds != -1 else None

    # barrier used to make sure all workers start at the same time, despite slow spawn (slower but better randomisation)
    b = mp.Barrier(parties=num_workers) if sync_start else None

    # make experiment results directory & save config
    os.makedirs(f'{base_path}_{agent_type}_results', exist_ok=True)
    shutil.copyfile(LIB_DIR + "text_xai/Config/constants.yml", f'{base_path}_{agent_type}_results/constants.yml')

    print(len(df))
    for n in cfg.params['ATTACKED_INDICES']:
        if sync_start:
            b.reset()

        # get current sentence
        cur_df = df.iloc[n:n + 1]
        sent_list = list(cur_df.content.values)
        print('original sentence', sent_list[0])
        print('original class', cur_df.label.values[0])
        print('original prediction', cur_df.preds.values[0])

        # calculate number of maximum actions - by specific text length for when not training for generalisation
        sent_len = len(sent_list[0].split())
        num_actions = None
        if env_type == 'Synonym':
            num_actions = sent_len
        elif env_type in ['SynonymDelete', 'SynonymMisspell']:
            num_actions = 2*sent_len
        else:
            print("Unsupported environment type!")
            exit(1)

        # define shared network, optimiser and params
        norm = None
        if norm_rounds != -1:
            norm = Normalizer(state_shape, norm_rounds)
        gnet = ReLeGATeAgentNet(state_shape, num_actions, norm=norm, lock=norm_lock)  # global network
        gnet.share_memory()  # share the global parameters in multiprocessing
        opt = SharedAdam(gnet.parameters(), lr=lr)  # global optimizer
        # global_ep, res_queue = mp.Value('i', 0), mp.Queue()
        global_ep = mp.Value('i', 0)

        # make results directory
        os.makedirs(f'{base_path}_{agent_type}_results/{n}', exist_ok=True)

        # parallel training
        print('num workers: ', num_workers)
        if agent_type == "relegate":
            workers = [ReLeGATeAgentWorker(gnet, opt, sent_list, global_ep, i, n, text_model, sent_len, num_rounds, b,
                                           train=True)
                       for i in range(num_workers)]
        else:  # defaults to random
            workers = [SearchAgentWorker(sent_list, global_ep, i, n, text_model, sent_len, num_rounds)
                       for i in range(num_workers)]

        [w.start() for w in workers]
        [w.join() for w in workers]


def pretrain_attack_model(epoch=0, model_path=None, model_type: str = "e2e", agent_type: str = "relegate",
                          device: str = "cpu"):
    """this model pretrains a single network on the data given"""
    # initialise parameters
    num_rounds = cfg.params["NUM_EPISODES"]
    state_shape = cfg.params["STATE_SHAPE"]
    max_sent_len = cfg.params["MAX_SENT_LEN"]
    lr = cfg.params["LEARNING_RATE"]
    norm_rounds = cfg.params["NORMALISE_ROUNDS"]
    sync_start = cfg.params["SYNC_START"]
    env_type = cfg.params["ENV_TYPE"]
    num_workers = mp.cpu_count() if cfg.params["NUM_WORKERS"] == 'cpu_count' else cfg.params["NUM_WORKERS"]
    offline_normalising = True if norm_rounds == 'offline' else False

    assert model_type in ["e2e", "transfer", 'lstm'] and agent_type in ["relegate", "search"], \
        "model type or agent type unrecognised or unsupported!"

    if env_type == 'Synonym':
        num_actions = max_sent_len
    elif env_type == 'SynonymDelete':
        num_actions = 2 * max_sent_len
    else:
        print("Unsupported environment type!")
        exit(1)

    norm = None
    if norm_rounds != -1 and norm_rounds != 'offline':
        norm = Normalizer(state_shape, norm_rounds)

    if epoch != 0:
        with open(f'{base_path}_normaliser.pkl', 'rb') as f:
            norm = pickle.load(f)

    # multiproccessing lock used for normalisation
    norm_lock = None
    if norm_rounds != -1:
        norm_lock = mp.Lock()

    # generate data
    data_path = base_path + '_sample.csv'
    df = pd.read_csv(data_path)
    # take smaller subset
    df = df.iloc[cfg.params['ATTACKED_INDICES']]
    print(len(df))
    sent_list = list(df.content.values)
    print(sent_list)

    # define language model
    text_model = None  # just to make sure it is not somehow referenced before assignment
    if model_type == "transfer":
        text_model = TransferBertTextModel(trained_model=base_path + '.pth')
    elif model_type == "e2e":
        text_model = E2EBertTextModel(trained_model=base_path + 'e2e_bert.pth', device=device)

    # for offline normalising using initial states statistics
    if offline_normalising:
        init_states = np.empty((len(df), 768))
        norm = Normalizer(state_shape, -1)
        for j in range(len(df)):
            init_states[j] = text_model.embed(df.content.values[j]).cpu()  # todo: make more efficient with batch processing
        norm_mean = init_states.mean(axis=0)
        norm_var = init_states.var(axis=0)

        norm.mean = torch.Tensor(norm_mean)
        norm.var = torch.Tensor(norm_var)

    gnet = ReLeGATeAgentNet(state_shape, num_actions, norm, norm_lock)  # global network

    # parallel training
    print('num workers: ', num_workers)

    # preform cleanup - remove old files
    if epoch == 0:
        os.remove(f'{base_path}_normaliser.pkl') if os.path.exists(f'{base_path}_normaliser.pkl') else ''
        for j in range(num_workers):
            # remove only if exists
            os.remove(f'{base_path}_agent_{j}.pth') if os.path.exists(f'{base_path}_agent_{j}.pth') else ''
            os.remove(f'{base_path}_optimiser_{j}.pth') if os.path.exists(f'{base_path}_optimiser_{j}.pth') else ''

    # if continuing to train an existing model
    if model_path is not None and epoch != 0:
        gnet.load_state_dict(torch.load(model_path))

    # gnet.share_memory()         # share the global parameters in multiprocessing
    global_ep = mp.Value('i', 0)
    opt = SharedAdam(gnet.parameters(), lr=lr)  # global optimizer

    # make experiment results directory & save config
    os.makedirs(f'{base_path}_{agent_type}_results/train/{epoch}', exist_ok=True)
    shutil.copyfile(LIB_DIR + "text_xai/Config/constants.yml",
                    f'{base_path}_{agent_type}_results/train/{epoch}/constants.yml')

    # barrier used to make sure all workers start at the same time, despite slow spawn
    # (slower but can lead to better randomisation)
    b = None
    if sync_start:
        b = mp.Barrier(parties=num_workers)

    # for A2C synced version
    sync_barrier = mp.Barrier(parties=num_workers)
    sync_lock = mp.Lock()
    sync_event = mp.Event()
    v_target_queue = mp.Queue()
    a_queue = mp.Queue()
    s_queue = mp.Queue()

    # agent init - training only really makes sense with A3C
    if agent_type == "relegate":
        workers = [ReLeGATeAgentWorker(gnet, opt, sent_list, global_ep, i, 'train', text_model,
                                       max_sent_len, num_rounds, b, epoch=epoch, train=True, sync_barrier=sync_barrier,
                                       sync_event=sync_event, v_target_queue=v_target_queue, a_queue=a_queue,
                                       s_queue=s_queue) for i in range(num_workers)]
    else:  # defaults to random
        workers = [SearchAgentWorker(sent_list, global_ep, i, 'train', text_model, max_sent_len, num_rounds)
                   for i in range(num_workers)]

    [w.start() for w in workers]
    [w.join() for w in workers]

    # save final agent model
    if model_path is not None:
        torch.save(gnet.state_dict(), model_path)

    # save normaliser
    with open(f'{base_path}_normaliser.pkl', 'wb') as f:
        pickle.dump(norm, f)


def attack_with_pretrained(model_path, epoch=0, model_type: str = "e2e", agent_type: str = "relax",
                           device: str = "cpu"):
    """
    this model uses a pretrained A3C agent and runs inference on the sentences
    :param model_path: the path to the pretrained model
    """
    # initialise parameters
    num_rounds = 5  # attack rounds
    state_shape = cfg.params["STATE_SHAPE"]
    max_sent_len = cfg.params["MAX_SENT_LEN"]
    norm_rounds = cfg.params["NORMALISE_ROUNDS"]
    env_type = cfg.params["ENV_TYPE"]

    if env_type == 'Synonym':
        num_actions = max_sent_len
    elif env_type == 'SynonymDelete':
        num_actions = 2 * max_sent_len
    else:
        print("Unsupported environment type!")
        exit(1)

    norm = None
    if norm_rounds != -1:
        with open(f'{base_path}_normaliser.pkl', 'rb') as f:
            norm = pickle.load(f)

    # multiproccessing lock used for normalisation
    norm_lock = None
    if norm_rounds != -1:
        norm_lock = mp.Lock()

    # define shared network
    gnet = ReLeGATeAgentNet(state_shape, num_actions, norm, norm_lock)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing

    # generate data
    data_path = base_path + '_sample.csv'
    df = pd.read_csv(data_path)

    # define language model
    text_model = None  # just to make sure it is not somehow referenced before assignment
    if model_type == "transfer":
        text_model = TransferBertTextModel(trained_model=base_path + '.pth')
    elif model_type == "e2e":
        text_model = E2EBertTextModel(trained_model=base_path + 'e2e_bert.pth', device=device)

    # make experiment results directory & save config
    os.makedirs(f'{base_path}_{agent_type}_results', exist_ok=True)
    shutil.copyfile(LIB_DIR + "text_xai/Config/constants.yml", f'{base_path}_{agent_type}_results/constants.yml')

    for n in [2, 12, 19, 24, 41, 67, 78, 83, 88, 92]:  # [2, 12, 19, 24, 41]:  # range(1):
        # get current sentence
        cur_df = df.iloc[n:n + 1]
        sent_list = list(cur_df.content.values)

        print('original sentence', sent_list[0])
        print('original class', cur_df.label.values[0])
        print('original prediction', cur_df.preds.values[0])

        # load model
        gnet.load_state_dict(torch.load(model_path))
        gnet.eval()
        global_ep = mp.Value('i', 0)

        # make results directory
        os.makedirs(f'{base_path}_{agent_type}_results/{n}', exist_ok=True)
        os.makedirs(f'{base_path}_{agent_type}_results/{n}/{epoch}', exist_ok=True)

        # parallel training
        num_workers = 1  # mp.cpu_count()
        print('num workers: ', num_workers)

        # agent init - training only really makes sense with A3C
        if agent_type == "relegate":
            workers = [ReLeGATeAgentWorker(gnet, None, sent_list, global_ep, i, n, text_model,
                                           max_sent_len, num_rounds, None, epoch=epoch, train=False) for i in
                       range(num_workers)]
        else:  # defaults to random
            workers = [SearchAgentWorker(sent_list, global_ep, i, n, text_model, max_sent_len, num_rounds)
                       for i in range(num_workers)]

        [w.start() for w in workers]
        [w.join() for w in workers]


if __name__ == "__main__":
    device = cfg.params['DEVICE']
    attack_type = cfg.params['ATTACK_TYPE']
    if "cuda" in device:
        mp.set_start_method('spawn')
    # attack each sentence separately
    if attack_type == 'individual':
        attack_individually(model_type=cfg.params['MODEL_TYPE'], agent_type=cfg.params['AGENT_TYPE'], device=device)

    # train loop with classic train-test scheme
    elif attack_type == 'universal':
        save_all_weights = True  # whether to copy the agent checkpoint files to a separate location for testing

        if save_all_weights:
            os.makedirs(f'{base_path}_agents', exist_ok=True)

        general_start = time.time()
        for epoch in range(1):
            start = time.time()
            pretrain_attack_model(model_type=cfg.params['MODEL_TYPE'], agent_type=cfg.params['AGENT_TYPE'],
                                  device=device,
                                  model_path=base_path + '_agent.pth', epoch=epoch)
            print('time', time.time() - start)
            # attack_with_pretrained(base_path + '_agent.pth', epoch, device=device)

            # copy all model weights to seperate file for later usage
            if save_all_weights:
                shutil.copyfile(base_path + '_agent.pth', f'{base_path}_agents/agent{epoch}.pth')
                shutil.copyfile(base_path + '_normaliser.pkl', f'{base_path}_agents/normaliser{epoch}.pkl')
        print('total time', time.time() - general_start)
