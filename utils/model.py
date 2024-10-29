import torch
from tqdm import tqdm

from model.protonet import load_protonet_lin
from utils import filter_opt
from model.factory import get_model
import numpy as np

def load_model(path):
    pnet = load_protonet_lin(**{"x_dim": [128], "hid_dim": 64, "z_dim": 64})
    pnet.load_state_dict(torch.load(path, weights_only=True))
    return pnet

def evaluate(model, data_loader, meters, desc=None):
    model.eval()

    for field,meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    for sample in data_loader:
        _, output = model.loss(sample)
        for field, meter in meters.items():
            meter.add(output[field])

    return meters