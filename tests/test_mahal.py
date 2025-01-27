from scipy.spatial.distance import mahalanobis

from utils.mahal_utils import Mahalanobis
from pytorch_ood.detector import RMD
import torch
import unittest
import numpy as np
from model.protonet import load_protonet_lin

Z_DIM = 64
N_CAL = 100
N_TEST = 33
N_CLASS = 5
z_cal = torch.rand((N_CLASS, N_CAL, Z_DIM))
z_test = torch.rand((N_CLASS, N_TEST, Z_DIM))
targets = torch.arange(N_CLASS).view(N_CLASS, 1).expand(N_CLASS, N_CAL)

class TestMahalanobis(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMahalanobis, self).__init__(*args, **kwargs)
        self.mahal = Mahalanobis(z_cal)
        pnet = load_protonet_lin(**{"x_dim": [128], "hid_dim": 64, "z_dim": 64, "dropout": 0.25, "hidden_layers": 3})
        pnet.load_state_dict(torch.load("../runs/outs/pnet.pt", weights_only=True))
        self.baseline_mahal = RMD(pnet)
        self.baseline_mahal.fit_features(z_cal, targets)
    def test_init(self):
        assert self.mahal.z_dim == Z_DIM
        assert self.mahal.n_class == N_CLASS
        assert self.mahal.n_calibrate == N_CAL
        return None

    def test_class_means(self):
        """
        Tests if class means calculated correctly
        :return: true if all class means (mu_k forall k in K) is same as libraries versions.
        """
        slow_mu_k = self.baseline_mahal.mu
        mu_k = self.mahal._get_class_means().squeeze()
        for mu, my_mu in zip(slow_mu_k, mu_k):
            assert torch.allclose(torch.Tensor(mu), my_mu), f"{mu} != {my_mu}"
        return None

    def test_whole_mean(self):
        """
        Tests if whole class mean (mu_naught) calculated correctly
        :return: True if correct
        """
        slow_mu_naught = self.baseline_mahal.background_mu
        mu_naught = self.mahal._get_whole_mean().squeeze()

        assert torch.allclose(mu_naught, slow_mu_naught, atol=1e-4), f"{mu_naught} != {slow_mu_naught}"
        return None

    def test_class_covariance(self):
        """
        Tests if each classes' covariance (sigma_k forall k in K) calculated correctly
        """
        my_sigma = self.mahal._get_class_sigma().squeeze()
        sigma = self.baseline_mahal.cov

        assert torch.allclose(sigma, my_sigma, atol=1e-4), (f"{sigma} != {my_sigma}"
                                                 f"{sigma.size()}, {my_sigma.size()}")
        return None

    def test_whole_covariance(self):
        """
        Tests if whole class covariance (sigma_naught) calculated correctly
        """
        my_sigma_naught = self.mahal._get_whole_sigma().squeeze()
        sigma_naught = self.baseline_mahal.background_cov
        assert torch.allclose(my_sigma_naught, sigma_naught, atol=1e-4), f"{sigma_naught} != {my_sigma_naught}"

    def test_mahalanobis(self):
        my_hal = self.mahal.mahalanobis_distance(self.mahal.calibration_embeddings, self.mahal._get_class_means(), self.mahal._get_class_sigma())
        mahal = self.baseline_mahal._calc_gaussian_scores(z_cal.view(N_CLASS*N_CAL, Z_DIM))
        assert torch.allclose(mahal, my_hal, atol=1e-4), f"{mahal} != {my_hal}" # Must be it!

    def test_relative_mahalanobis(self):
        mahal = (self.baseline_mahal._calc_gaussian_scores(z_cal.view(N_CLASS*N_CAL, Z_DIM))
                 - self.baseline_mahal._background_score(z_cal.view(N_CLASS*N_CAL, Z_DIM)).view(-1,1))
        my_hal = self.mahal.relative_mahalanobis_distance(z_cal)
        assert torch.allclose(my_hal, mahal, atol=1e-4), f"{my_hal} != {mahal}"

    def test_diag_mahalanobis(self):
        mahal = self.baseline_mahal._calc_gaussian_scores(z_test.view(N_CLASS*N_TEST, Z_DIM))
        my_hal = self.mahal.diag_mahalanobis_distance(z_test)
        assert torch.allclose(my_hal, mahal, atol=1e-4), f"{my_hal} != {mahal}"





    def test_calibrate(self):
        return None

