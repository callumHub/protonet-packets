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

UT_MOBILE_CLASS_MAP = {
                "FACEBOOK": 0, "TWITTER": 1, "REDDIT": 2, "INSTAGRAM": 3, "PINTREST": 4,
                "YOUTUBE": 5, "NETFLIX": 6, "HULU": 7, "SPOTIFY": 8, "PANDORA": 9, "MAPS": 10,
                "DRIVE": 11, "DROPBOX": 12, "GMAIL": 13, "MESSENGER": 14, "HANGOUT": 15
            }

OG_CLASS_MAP = {"C2": 0, "CHAT": 1, "FILE_TRANSFER": 2, "STREAMING": 3, "VOIP": 4} # OG VPN Data Classses

class VPNDataTransforms(object):
    def __init__(self, splits, n_classes, full_path=None):
        self.data_path = os.path.join(os.getcwd(),
            f"../../enc-vpn-uncertainty-class-repl/processed_data/test_train_cal/data-label-combined-{splits}.jsonl")
        if full_path is not None:
            self.data_path = full_path
        self.data = pd.read_json(self.data_path, lines=True)
        if full_path is not None:
            pass
            #print("Data length: ", len(self.data))
            #print("length of the minimum class: ", len(self.data.data[self.data["labels"] == "VOIP"]))


        self.data.data = self.data.data.apply(lambda x: torch.tensor(x, dtype=torch.float32))
        if n_classes == 5: # OG VPN DATA
            self.class_map = OG_CLASS_MAP
            self.class_list = list(self.class_map.keys())
        else: # UTMOBILE CLASSLIST
            self.class_map = UT_MOBILE_CLASS_MAP
            self.class_list = list(self.class_map.keys())
        self.out_dict = {}
        self.removed_classes = 0

    def make_dict(self):
        lt_classes = []
        for k in self.class_map.keys():
            class_data = self.data[self.data["labels"] == k].data.values.tolist()
            if len(class_data) < 5:
                #print(f"Class {k} has less than 10 samples")
                lt_classes.append(k)
                continue
            self.out_dict[k] = {"label": k,
                                "data": torch.stack(self.data[self.data["labels"] == k].data.values.tolist())}
        for k in lt_classes:
            self.class_map.pop(k)
            self.class_list.pop(self.class_list.index(k))
            self.data.drop(self.data[self.data["labels"] == k].index, inplace=True)
        self.removed_classes = len(lt_classes)

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
    def __init__(self, splits, n_classes, fp=None):
        if fp is not None: dicter = VPNDataTransforms(splits, n_classes, full_path=fp)
        else: dicter = VPNDataTransforms(splits, n_classes)
        dicter.make_dict()
        min_class = len(dicter.data.data[dicter.data["labels"] == dicter.data.labels.value_counts().keys()[-1]])  # VOIP is min class for OG one
        #print("length of the minimum class: ", min_class)  # DRIVE is min for utmobile
        self.n_support = 5 if min_class > 8 else 3
        self.batch_size = min_class - self.n_support
        self.data = dicter.out_dict
        self.class_map = dicter.class_map
        self.class_list = dicter.class_list
        self.removed_classes = dicter.removed_classes
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #key = list(self.class_map.keys())[list(self.class_map.values()).index(idx)]
        key = self.class_list[idx]

        return episode_sampler(self.data[key], self.batch_size, self.n_support)

def load(splits, n_episodes, n_classes, n_way, fp=None):

    if fp is not None: ds = VPNDataset(splits, n_classes, fp=fp)
    else: ds = VPNDataset(splits, n_classes)
    n_classes = n_classes-ds.removed_classes
    n_way = n_way-ds.removed_classes
    sampler = EpisodicBatchSampler(n_classes, n_way, n_episodes)
    return DataLoader(ds, batch_sampler=sampler, shuffle=False, num_workers=0)




