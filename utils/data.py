import os

import numpy as np
import torch.distributions

from data.vpn_packets import VPNDataset
from scipy import stats



class KDEDist(stats.rv_continuous):

    def __init__(self, kde, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kde = kde

    def _pdf(self, x, *args, **kwargs) -> any:
        return self._kde.pdf(x)


def sample_transform(sample, cal_samples):
    sorted_class_cal, sorted_indices_cal = zip(*sorted((cls, idx) for idx, cls in enumerate(cal_samples['class'])))
    sorted_class, sorted_indices = zip(*sorted((cls, idx) for idx, cls in enumerate(sample['class'])))
    temp = list(sorted_indices)
    for i in range(len(sorted_indices)):
        if sorted_indices[i] in sorted_indices_cal:
            continue
        else:
            temp.remove(sorted_indices[i])
    sorted_indices = tuple(temp)
    # Reorder 'xq' and 'xs' based on sorted indices
    sample['class'] = list(sorted_class)
    sample['xq'] = sample['xq'][list(sorted_indices)]
    sample['xs'] = sample['xs'][list(sorted_indices)]
    return sample


def save_stats(data, name, run_desc):
   save_path = os.path.join(f"./eval_outs/{run_desc}")
   os.makedirs(save_path, exist_ok=True)
   np.save(os.path.join(save_path, name+".npy"), np.array(data))
