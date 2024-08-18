import os
import random

import numpy as np
import torch

from models.AWN_model import AWN
from models.CNN_Pijackova import CNNModel
from models.CLDNN_Pijackova import CLDNNModel
from models.CGDNN_Pijackova import CGDNNModel
from models.Resnet_Alexivaner import ResidualModel
from models.RNN import RNNModel
from models.LSTM import LSTMModel
from models.AMCNet import AMC_Net


def fix_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_exp_settings(logger, cfg):
    """
    log the current experiment settings.
    """
    logger.info('=' * 20)
    log_dict = cfg.__dict__.copy()
    for k, v in log_dict.items():
        logger.info(f'{k} : {v}')
    logger.info('=' * 20)


def create_AWN_model(cfg):
    """
    build AWN model
    """
    model = AWN(
        num_classes=cfg.num_classes,
        num_levels=cfg.num_level,
        in_channels=cfg.in_channels,
        kernel_size=cfg.kernel_size,
        latent_dim=cfg.latent_dim,
        regu_details=cfg.regu_details,
        regu_approx=cfg.regu_approx,
    ).to(cfg.device)

    return model


def create_CNN_model(cfg):
    """
    build CNN model
    """
    model = CNNModel(
        num_classes=cfg.num_classes,
        dataset_name=cfg.dataset,
    ).to(cfg.device)

    return model

def create_CLDNN_model(cfg):
    """
    build CLDNN model
    """
    model = CLDNNModel(
        num_classes=cfg.num_classes,
        dataset_name=cfg.dataset,
    ).to(cfg.device)

    return model

def create_CGDNN_model(cfg):
    """
    build CGDNN model
    """
    model = CGDNNModel(
        num_classes=cfg.num_classes,
        dataset_name=cfg.dataset,
    ).to(cfg.device)

    return model

def create_Resnet_model(cfg):
    """
    build Resnet model
    """
    model = ResidualModel(
        num_classes=cfg.num_classes,
        dataset_name=cfg.dataset,
    ).to(cfg.device)

    return model


def create_RNN_model(cfg):
    """
    build RNN model
    """
    if cfg.dataset == '2016.10a' or cfg.dataset == '2016.10b' or cfg.dataset == 'migou_dataset_19.08':
        input_dim = 128
        hidden_dim = 128
    elif cfg.dataset == '2018.01a':
        input_dim = 1024
        hidden_dim = 1024
    model = RNNModel(
        num_classes=cfg.num_classes,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    ).to(cfg.device)

    return model

def create_LSTM_model(cfg):
    """
    build LSTM model
    """
    if cfg.dataset == '2016.10a' or cfg.dataset == '2016.10b' or cfg.dataset == 'migou_dataset_19.08':
        input_dim = 128
        hidden_dim = 128
    elif cfg.dataset == '2018.01a':
        input_dim = 1024
        hidden_dim = 1024
    model = LSTMModel(
        num_classes=cfg.num_classes,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    ).to(cfg.device)

    return model

def create_AMC_Net(cfg):
    """
    build AMCNet model
    """
    model = AMC_Net(
        num_classes=cfg.num_classes,
        sig_len=128,
        extend_channel=36,
        latent_dim=512,
        num_heads=2,
        conv_chan_list=[36, 64, 128, 256]
    ).to(cfg.device)

    return model
