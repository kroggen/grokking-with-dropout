
import torch
DEFAULT_MODULO = 113
MODULO = 113
TRAIN_FRACTION = 0.3
HIDDEN_SIZES = [300, 300]  
USE_XAVIER_INITIALIZATION = False
LEARNIGN_RATE = 0.002
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 800
USE_BIAS = True
HYPER_PARAMETERS = {}
HYPER_PARAMETERS["AdamW"] = {"lr": 0.001, "weight_decay": 0.2, "betas": (0.9, 0.98)}
HYPER_PARAMETERS["SGD"] = {"lr": 1, "weight_decay": 0.0002, "momentum": 0}

INPUT_TYPES = ['train', 'test', 'noise', 'ones', 'minus_ones', 'zeros', 'general']

GPU = "cuda:5"
FLOAT_PRECISION = torch.float64
FLOAT_PRECISION_MAP = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64,
}