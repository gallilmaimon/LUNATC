import torch
import numpy as np
import random
import os


def seed_everything(seed=42):
    if seed == -1:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
