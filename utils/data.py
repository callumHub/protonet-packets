import torch.distributions

from data.vpn_packets import VPNDataset
from scipy import stats

def load(opt, splits):
    ds = VPNDataset(splits)

    return ds

class KDEDist(stats.rv_continuous):

    def __init__(self, kde, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kde = kde

    def _pdf(self, x, *args, **kwargs) -> any:
        return self._kde.pdf(x)


def sample_transform(sample):
    sorted_class, sorted_indices = zip(*sorted((cls, idx) for idx, cls in enumerate(sample['class'])))

    # Reorder 'xq' and 'xs' based on sorted indices
    sample['class'] = list(sorted_class)
    sample['xq'] = sample['xq'][list(sorted_indices)]
    sample['xs'] = sample['xs'][list(sorted_indices)]
    return sample
