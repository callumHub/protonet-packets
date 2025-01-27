import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassCalibrationError, MulticlassF1Score, MulticlassConfusionMatrix

from torch.autograd import Variable
from scipy.stats import gaussian_kde
from scipy.integrate import quad
import numpy as np

from model.factory import register_model
from utils.data import KDEDist
from utils.mahal_utils import Mahalanobis

'''
return list of univariate gaussian kde for each class
use to get p values by with pdf f'n
'''
def get_kde(rel_mahalanobis, target_inds, n_way):
    class_kernel_densities = [0 for _ in range(n_way)]
    for idx in range(n_way):#target_inds.squeeze().T[0]:
        # TODO: Sometimes gives singular covariance matrix error, must handle
        class_kernel_densities[idx] = gaussian_kde(rel_mahalanobis[idx].cpu(), bw_method='scott')

    return class_kernel_densities

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()

        self.encoder = encoder
        self.rmd = None

    def loss(self, sample):
        xs = Variable(sample['xs'])  # support
        xq = Variable(sample['xq'])  # query

        if torch.cuda.is_available():
            xs = xs.cuda()
            xq = xq.cuda()

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        # WHY ARE THESE THE TARGET INDS!! How are they the same every time!
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query,
                                                                         1).long()  # CHANGED, removed ,1 from last arg of view and expand
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)
        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(1) # Remove mean if using mahal
        zq = z[n_class * n_support:]

        dists = euclidean_dist(zq, z_proto)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)


        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)

        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

    def calibrate(self, sample):
        # Use support to calc mu's and sigmas, query to calc m_rel
        xs = Variable(sample['xs'])  # support
        xq = Variable(sample['xq'])  # calibrate
        n_class, n_support, x_dim = xs.size()
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_support,
                                                                          1).long()  # CHANGED, removed ,1 from last arg of view and expand
        target_inds = Variable(target_inds, requires_grad=False)


        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_cal = xq.size(1)
        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_cal, *xq.size()[2:])], 0)
        #x = xs.view(n_class * n_support, *xs.size()[2:])
        with torch.no_grad():
            z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_support = z[:n_class * n_support].view(n_class, n_support, z_dim)
        z_cal = z[n_class * n_support:].view(n_class, n_cal, z_dim)
        self.rmd = Mahalanobis(z_support) # Now compute mahalanobis # USE support to set mahal vars
        # use calibrate to compute rel_mahal (if using sup, ood scores at test time will be higher
        m_k_rel = torch.min(self.rmd.relative_mahalanobis_distance(z_cal), dim=1).values.view(n_class, n_cal)


        # Obtain n_class Gaussian KDE's for each class
        g_k = get_kde(m_k_rel, target_inds, n_class)
        return g_k # Used in alg 3!

    # Alg 3
    def test(self, sample, g_k, use_cuda=False, calc_stats=True):
        """
        :param sample: contains query and support examples for test
        :param g_k: Gaussian kernel densities for each class
        :param use_cuda:
        :return: pvals, acc_vals, caliber, micro_f, confusion
        """
        xq = sample['xq']
        xs = sample['xs']
        n_way = xq.size(0) # len(g_k) # number of targets
        n_class = xq.size(0)
        n_query = xq.size(1)
        n_support = xs.size(1)
        # Examples are in order of class so just set target indices as:
        target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query,
                                                                         1).long().squeeze()
        target_inds = Variable(target_inds, requires_grad=False)

        if use_cuda: target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)
        with torch.no_grad():
            z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z = z[n_class * n_support:].view(n_class, n_query, z_dim)
        #z = z.view(n_class, n_query+n_support, z_dim) #Doing this significantly lowers accuracy: find out why by testing different splits
        dists = self.rmd.diag_mahalanobis_distance(z)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        _, y_hat = log_p_y.max(2)

        rel_d = torch.min(self.rmd.relative_mahalanobis_distance(z), dim=1).values.view(n_class, n_query)
        pvals = []
        mid_pvals = []
        correct_preds = 0
        # OOD Scoring
        for i in range(n_way):
            for index in range(len(y_hat.tolist()[i])):
                predicted_class = y_hat.tolist()[i][index]
                if i == predicted_class: correct_preds += 1
                r = rel_d.tolist()[predicted_class][index]
                p_val = quad(g_k[predicted_class].pdf, r, np.inf)[0] # Integrate pdf to get p values
                mid_pvals.append(1-p_val) # confidence score is 1 minus the p value
            pvals.append(mid_pvals)

        # Calc stats
        if calc_stats:
            calibration_error = MulticlassCalibrationError(num_classes=n_class, n_bins=n_class, norm='l1')
            micro_f1 = MulticlassF1Score(num_classes=n_class, average='micro')
            caliber = calibration_error(log_p_y.view(n_class*n_query, -1), target_inds.flatten())
            micro_f = micro_f1(log_p_y.view(n_class*n_query, -1), target_inds.flatten())
            confusion_matrix = MulticlassConfusionMatrix(num_classes=n_class)
            confusion = confusion_matrix(log_p_y.view(n_class*n_query, -1), target_inds.flatten())
            #print("Expected Calibration Error is: ", caliber)
            #print("Micro F1 Score is: ", micro_f)
            acc_vals = torch.eq(y_hat, target_inds).float().mean()
            #print("Accuracy:", correct_preds/(n_class*n_query))
            return mid_pvals, acc_vals, caliber, micro_f, confusion
        else:
            return pvals, torch.tensor(0.1), torch.tensor(0.01), torch.tensor(0.1), torch.tensor(0.01) # Dummy vals

    def ood_score(self, sample, g_k):
        n_way = len(g_k)
        n_examples = sample.size(0)
        z = self.encoder.forward(sample)
        z = z.unsqueeze(0)
        dists = self.rmd.diag_mahalanobis_distance(z)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_examples, -1)
        _, y_hat = log_p_y.max(1)
        rel_d = torch.min(self.rmd.relative_mahalanobis_distance(z), dim=1).values.view(-1, n_examples)
        pvals = []
        mid_pvals = []
        correct_preds = 0
        # OOD Scoring

        for index in range(len(y_hat.tolist())):
            predicted_class = y_hat.tolist()[index]
            r = rel_d.tolist()[0][index]
            p_val = quad(g_k[predicted_class].pdf, r, np.inf)[0]  # Integrate pdf to get p values
            mid_pvals.append(1 - p_val)  # confidence score is 1 minus the p value
        pvals.append(mid_pvals)
        return pvals


@register_model('protonet_lin')
def load_protonet_lin(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    dropout = kwargs['dropout']
    hidden_layers = kwargs['hidden_layers']
    hid_dim2 = 64 # int(1024) # hardcode
    hid_dim3 = 64 # int(1024)
    z_dim = kwargs['z_dim']
    z_dim = 64
    if hidden_layers == 3:
        encoder = nn.Sequential(
            nn.Linear(x_dim[0], hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, z_dim),
            Flatten()
        )
    elif hidden_layers == 2:
        encoder = nn.Sequential(
            nn.Linear(x_dim[0], hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, z_dim),
            Flatten()
        )
    elif hidden_layers == 1:
        encoder = nn.Sequential(
            nn.Linear(x_dim[0], hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, z_dim),
            Flatten()
        )
    elif hidden_layers == 0:
        encoder = nn.Sequential(
            nn.Linear(x_dim[0], z_dim)
        )
    else:
        encoder = None
    return Protonet(encoder)


