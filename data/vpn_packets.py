#vpn_packets.py: Data loading classes
'''
Notes:
Batch size implemented with hardcoded magic numbers

'''


import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import numpy as np
import os

from data.base import EpisodicBatchSampler


class VPNDataTransforms(object):
    def __init__(self, splits):
        self.data_path = os.path.join(os.getcwd(),
            f"../../enc-vpn-uncertainty-class-repl/processed_data/test_train_cal/data-label-combined-{splits}.jsonl")
        self.data = pd.read_json(self.data_path, lines=True)
        self.data.data = self.data.data.apply(lambda x: torch.tensor(x, dtype=torch.float32))
        self.class_map = {"C2": 0, "CHAT": 1, "FILE_TRANSFER": 2, "STREAMING": 3, "VOIP": 4}
        self.out_dict = {}

    def make_dict(self):
        for k in self.class_map.keys():
            self.out_dict[k] = {"label": k,
                                "data": torch.stack(self.data[self.data["labels"] == k].data.values.tolist())}


def episode_sampler(data, n_query, n_support):
    n_examples = len(data["data"])
    ex_inds = torch.randperm(n_examples)[:(n_support + n_query)]
    support_inds = ex_inds[:n_support]
    query_inds = ex_inds[n_support:]
    xs = data["data"][support_inds]
    xq = data["data"][query_inds]
    return {
        "class": data["label"],
        "xs": xs,
        "xq": xq
    }


class VPNDataset(Dataset):
    def __init__(self, splits):
        dicter = VPNDataTransforms(splits)
        dicter.make_dict()
        self.batch_size = 100 if splits == "train" else 33 # SETTING BATCH SIZE HERE: SETS TO MAX POSSIBLE BATCH FOR CLASSES
        self.data = dicter.out_dict
        self.class_map = {"C2": 0, "CHAT": 1, "FILE_TRANSFER": 2, "STREAMING": 3, "VOIP": 4}
        self.class_list = ["C2", "CHAT", "FILE_TRANSFER", "STREAMING", "VOIP"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #key = list(self.class_map.keys())[list(self.class_map.values()).index(idx)]
        key = self.class_list[idx]
        return episode_sampler(self.data[key], self.batch_size, 5)

def load(splits, n_episodes):
    sampler = EpisodicBatchSampler(5, 5, n_episodes)
    ds = VPNDataset(splits)

    return DataLoader(ds, batch_sampler=sampler, shuffle=False, num_workers=0)




