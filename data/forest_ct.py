
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


from data.base import EpisodicBatchSampler

class ForestCTDataTransforms(object):
    def __init__(self, splits, n_classes, full_path=None):
        self.data_path = os.path.join(os.getcwd(),
                                      f"../../forest_ct_data/covertype_data/splits/split0/{splits}.csv")\
            if full_path is None else full_path
        self.df = pd.read_json(self.data_path, lines=True)
        #self.df = pd.read_csv(self.data_path)
        self.out_dict = {}

    def make_dict(self):
        # Use torch.Tensor(eval(x)) if df has string lists
        self.df["data"] = self.df["data"].apply(lambda x: torch.Tensor(x))
        for k in range(1, 8): # 5 ct's from 1 to 6

            class_data = self.df[self.df['labels'] == k].data.values.tolist()
            self.out_dict[k] = {"label": k,
                                "data": torch.stack(class_data),
            }


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

class ForestCTDataset(Dataset):
    def __init__(self, splits, n_classes, full_path=None):
        if full_path: dicter = ForestCTDataTransforms(splits, n_classes, full_path=full_path)
        else: dicter = ForestCTDataTransforms(splits, n_classes)
        dicter.make_dict()
        min_class = len(dicter.df.data[dicter.df["labels"] == dicter.df.labels.value_counts().keys()[-1]])
        self.n_support = 5 if min_class > 8 else 3
        self.batch_size = min_class - self.n_support
        self.data = dicter.out_dict
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key = int(idx) + 1 # labels start from 1
        return episode_sampler(self.data[key], self.batch_size, self.n_support)

def load(splits, n_episodes, n_classes, n_way, fp=None):
    if fp: ds = ForestCTDataset(splits, n_classes, full_path=fp)
    else: ds = ForestCTDataset(splits, n_classes)
    sampler = EpisodicBatchSampler(n_classes, n_way, n_episodes)
    return DataLoader(ds, batch_sampler=sampler, shuffle=False, num_workers=0)

