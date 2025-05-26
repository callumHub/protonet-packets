import copy

from model.protonet import Protonet, get_kde, euclidean_dist, Flatten, load_protonet_lin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from model.factory import register_model
from utils.mahal_utils import Mahalanobis


class ProtonetWithTarget(Protonet):
    def __init__(self, encoder, update_frequency, tau, params):
        super().__init__(encoder)
        self.params = params
        self.target = load_protonet_lin(
        **{"x_dim": [params.x_dim], "hid_dim": params.hidden_dim, "z_dim": params.z_dim, "dropout": params.dropout,
           "hidden_layers": params.hidden_layers}).encoder
        self.target.load_state_dict(copy.deepcopy(encoder.state_dict()))
        self.target.eval()
        self.encoder.train()
        self.update_frequency = update_frequency
        self.tau = tau
        self.proto_combinator = params.proto_combinator
        self.combinator_slope = params.combinator_slope
    def loss(self, sample, batch):
        self.proto_combinator = min(1.0, self.proto_combinator+self.combinator_slope*batch)
        self.tau = min(0.99, self.tau+self.proto_combinator*batch*0.1)

        if (batch+1) % self.update_frequency == 0:
            #self.update_target_network()
            self.soft_update_target_network()
            #assert self.encoder.training, "encoder must be training"
            #assert not self.target.training, "target is in training mode"

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
        # TARGET MODEL DUTY: Get support set and prototype, use prediction to update prototype.
        with torch.no_grad():
            z_target = self.target.forward(x)
        z_target = F.normalize(z_target, p=2, dim=-1)
        z_dim = z_target.size(-1)
        z_target_sup = z_target[:n_class*n_support].view(n_class, n_support, z_dim)
        z_target_q = z_target[n_class*n_support:].view(n_class, n_query, z_dim)

        if batch == 658:
            pass
        z_proto_rect = z_target_sup.mean(1)  # Remove mean if using mahal
        z_proto_rect = self.rectify_prototypes(query_features=z_target_q, support_features=z_target_sup, prototypes=z_proto_rect)


        # Predictive model duty
        z = self.encoder.forward(x)
        z_proto_actual = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:].view(n_class*n_query, z_dim)
        # RECTIFY PROTOTYPES: (removed to test during calibrate
        # if batch > 100:
        #    z_proto = self.rectify_prototypes(query_features=zq.view(n_class, n_query, z_dim), support_features=z_sup, prototypes=z_proto)
        z_proto = z_proto_rect*self.proto_combinator + z_proto_actual*(1.0-self.proto_combinator)
        zq = F.normalize(zq, p=2, dim=-1)
        z_proto = F.normalize(z_proto, p=2, dim=-1)
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
        z_q = z[n_class*n_support:].view(n_class, -1, z_dim)
        # TODO: Add test for rectify during calibrate

        z_cal = z[n_class * n_support:].view(n_class, n_cal, z_dim)
        bias_diminish = self.rectify_prototypes(z_q, z_support, z_support.mean(1))
        self.rmd = Mahalanobis(z_support, n_support, bias_diminish) # Now compute mahalanobis # USE support to set mahal vars
        # use calibrate to compute rel_mahal (if using sup, ood scores at test time will be higher
        m_k_rel = torch.min(self.rmd.relative_mahalanobis_distance(z_cal), dim=1).values.view(n_class, n_cal)

        # Obtain n_class Gaussian KDE's for each class
        g_k = get_kde(m_k_rel, target_inds, n_class)
        return g_k # Used in alg 3!

    def update_target_network(self):
        self.encoder.to('cpu')
        self.target.to('cpu')
        self.target.load_state_dict(copy.deepcopy(self.encoder.state_dict()))
        self.encoder.to('cuda:0')
        self.target.to('cuda:0')
        self.target.eval()

    @torch.no_grad()
    def soft_update_target_network(self):

        for target_param, encoder_param in zip(self.target.parameters(), self.encoder.parameters()):
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(self.tau * encoder_param.data)
        self.target.eval()


