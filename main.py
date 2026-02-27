import os, time
import numpy as np
import random
import torch

import tensorboard
from dqn import DQN

import warnings
warnings.simplefilter("error", RuntimeWarning)

CONFIG = {
    "environment": "LayoutGenerator",
    "dataset_path": "./dataset/data_eval/",
    "para_path": "net_param",
    
    "train_mode": "train",
    "total_epoch": 1000,
    "resume_epoch": 0,

    "worker_num": 8,
    "lr_max": 1e-3,
    "lr_min": 1e-5,
    "lr_min_retrain": 5e-6,

    "gamma": 0.99,
    "epsilon": 0.2,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.005,
    "max_grad_norm": 1.0,
    "batch_size": 128,
    "n_step": 3,
    "buffer_capacity": 100000,
    "buffer_alpha": 0.6,
    "buffer_beta": 0.4,
    "buffer_beta_increment": 1e-5,

    "log_dir": "./logs/" + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())),

    "base_model_path": "./base_model",

    "wandb_project": "RLhouse-DQN-prefly",
}

torch.set_printoptions(threshold=10000)

os.environ["WANDB_MODE"]="offline"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    set_seed(0)

    agent = DQN(config=CONFIG)

    agent.learn()

    agent.predict()


if __name__ == "__main__":
    main()
