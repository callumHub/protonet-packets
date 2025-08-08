import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassCalibrationError, MulticlassF1Score, MulticlassConfusionMatrix

from torch.autograd import Variable
from scipy.stats import gaussian_kde
from scipy.integrate import quad
import numpy as np

from model.factory import register_model

from utils.mahal_utils import Mahalanobis
import utils.experiment_context
from KDEpy import bw_selection
'''
return list of univariate gaussian kde for each class
use to get p values by with pdf f'n
'''
def get_kde(rel_mahalanobis, target_inds, n_way) -> list[gaussian_kde]:
    if utils.experiment_context.bandwidth_experiment:
        class_kernel_densities = []
        class_bandwidths = [bw_selection.improved_sheather_jones(np.asarray(rel_mahalanobis[i]).reshape(-1, 1))
                            for i in range(n_way)]
        bandwidth = np.mean(class_bandwidths)
        print('bandwidth:', bandwidth, " With std: ", np.std(class_bandwidths), " Max: ", np.max(class_bandwidths),
              " Min: ", np.min(class_bandwidths)) # Consider using max instead of mean.

        for idx in range(n_way):  # target_inds.squeeze().T[0]:
            # TODO: Sometimes gives singular covariance matrix error, must handle
            class_kernel_densities.append(gaussian_kde(rel_mahalanobis[idx],
                                                       bw_method=float(class_bandwidths[idx])))

        return class_kernel_densities
    else:
        class_kernel_densities = [0 for _ in range(n_way)]
        for idx in range(n_way):#target_inds.squeeze().T[0]:
            # TODO: Sometimes gives singular covariance matrix error, must handle
            class_kernel_densities[idx] = gaussian_kde(rel_mahalanobis[idx].cpu(), bw_method=utils.experiment_context.bandwidth_value)

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
        self.prototypes = None
    def loss(self, sample, batch):
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
        z_sup = z[:n_class * n_support].view(n_class, n_support, z_dim)

        z_proto = z_sup.mean(1) # Remove mean if using mahal


        zq = z[n_class * n_support:]
        # RECTIFY PROTOTYPES: (removed to test during calibrate
        #if batch > 150:
        #    z_proto = self.rectify_prototypes(query_features=zq.view(n_class, n_query, z_dim), support_features=z_sup, prototypes=z_proto)

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
        # TODO: Add test for rectify during calibrate

        z_cal = z[n_class * n_support:].view(n_class, n_cal, z_dim)
        #bias_diminish = [self, z_cal]
        #TODO: Jun 24, changed og pnet from support in mahal to the mean of support in mahal.
        input_to_mahal = z_support.mean(1).expand(n_class, n_class, z_dim)
        self.rmd = Mahalanobis(z_support, n_cal=n_support) #, bias_diminish) # Now compute mahalanobis # USE support to set mahal vars
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
                max_val = g_k[predicted_class].dataset.max()
                min_val = g_k[predicted_class].dataset.min()
                bw = g_k[predicted_class].factor
                #p_val = quad(g_k[predicted_class].pdf, r, max_val + bw, limit=50000, epsabs=0.1, epsrel=0.1)[
                #    0]  # Integrate pdf to get p values
                grid = np.linspace(r, max_val+(max_val-min_val)*0.1, 5000)
                p_val = np.trapz(g_k[predicted_class].pdf(grid), grid, dx=bw/2)
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
        # TODO: June 24: Trying to figure Difference in OOD score, this now is the same as easyfsl at each step.
        n_class = len(g_k)
        n_examples = sample.size(0)
        z = self.encoder.forward(sample)
        z = z.unsqueeze(0)
        dists = self.rmd.diag_mahalanobis_distance(z).view(n_examples, -1)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_examples, -1)
        _, y_hat = log_p_y.max(1)

        rel_d = torch.min(self.rmd.relative_mahalanobis_distance(z), dim=1).values.view(-1, n_examples)
        rel_d = rel_d.detach().cpu().numpy()
        y_hat = y_hat.detach().cpu().numpy()
        pvals = []
        mid_pvals = []
        correct_preds = 0
        # OOD Scoring

        for index in range(len(y_hat.tolist())):
            predicted_class = y_hat.tolist()[index]
            r = rel_d.tolist()[0][index]
            max_val = g_k[predicted_class].dataset.max()
            min_val = g_k[predicted_class].dataset.min()
            #p_val = quad(g_k[predicted_class].pdf, r, max_val+bw, limit=50000000, epsabs=1e-10, epsrel=1e-10)[0]  # Integrate pdf to get p values
            grid = torch.linspace(r, max_val+10*g_k[predicted_class].factor, 1500000)
            p_val = torch.trapz(torch.as_tensor(g_k[predicted_class].pdf(grid)), torch.as_tensor(grid)).item()
            mid_pvals.append(1 - p_val)  # confidence score is 1 minus the p value
        pvals.append(mid_pvals)
        return pvals

    @staticmethod
    def rectify_prototypes(query_features: torch.Tensor, support_features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        device = query_features.device
        n_classes = query_features.size(0)
        n_support = support_features.size(1)
        n_query = query_features.size(1)
        support_target_inds = torch.arange(0, n_classes).view(n_classes, 1, 1).expand(n_classes, n_support,
                                                                      1).long().squeeze()
        support_target_inds = Variable(support_target_inds, requires_grad=False)
        one_hot_support_labels = nn.functional.one_hot(support_target_inds, n_classes).view(n_classes*n_support, -1).to(device)
        average_support_query_shift = support_features.mean(dim=1, keepdim=True) -\
            query_features.mean(dim=1, keepdim=True)
        query_features = query_features + average_support_query_shift
        #support_logits = (-euclidean_dist(support_features.view(n_classes*n_support, -1), prototypes)).exp().to(device)
        #query_logits = (-euclidean_dist(query_features.view(n_classes*n_query, -1), prototypes)).exp().to(device)
        #support_logits = support_logits.clamp(min=0.1, max=10000) # NEW FROM ME, but why do I need to handle here and not in easyFSL?
        #query_logits = query_logits.clamp(min=0.1, max=10000) # NEW FROM ME
        support_logits = (-euclidean_dist(support_features.view(n_classes * n_support, -1), prototypes)).exp().to(device)
        query_logits = (-euclidean_dist(query_features.view(n_classes * n_query, -1), prototypes)).exp().to(device)
        one_hot_query_pred = nn.functional.one_hot(query_logits.argmax(-1), n_classes).view(n_classes*n_query, -1)

        normalization_vector = (one_hot_support_labels*support_logits).sum(0) +\
                               (one_hot_query_pred*query_logits).sum(0).unsqueeze(0) # [1, n_classes]
        normalization_vector = normalization_vector.clamp(min=0.00001)

        support_reweighting = (one_hot_support_labels*support_logits) / normalization_vector # [n_sup, n_classes]
        query_reweighting = (one_hot_query_pred*query_logits) / normalization_vector # [n_query, n_classes]

        prototypes = (support_reweighting*one_hot_support_labels).t().matmul(support_features.view(n_classes*n_support, -1)) + \
                     (query_reweighting*one_hot_query_pred).t().matmul(query_features.view(n_classes*n_query, -1))
        return prototypes




@register_model('protonet_lin')
def load_protonet_lin(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    dropout = kwargs['dropout']
    hidden_layers = kwargs['hidden_layers']
    z_dim = kwargs['z_dim']
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
            Flatten(),
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
    elif hidden_layers == 4:
        encoder = nn.Sequential(
            nn.Linear(x_dim[0], hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, z_dim),
            Flatten(),
        )
    else:
        encoder = None
    return Protonet(encoder)


