import math
import random
import numpy as np
import torch
import os

def set_seed(random_seed=42) -> 0:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    print(f"Random seed set as {random_seed}")
    return 0

def get_train_val_test_size(total_size, train_val_test_ratio):
    train_scale, validation_scale, test_scale = train_val_test_ratio
    training_size = math.ceil(total_size*(train_scale/(train_scale + validation_scale + test_scale)))
    validation_size = math.floor(total_size*(validation_scale/(train_scale+ validation_scale+ test_scale)))
    return training_size, validation_size, total_size - training_size - validation_size


def get_parameter_num(model):
    para_num = 0
    for parameter in model.parameters():
        para_num += parameter.numel()
    return para_num

