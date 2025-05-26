from dataclasses import dataclass
from typing import Optional
@dataclass
class ParameterStore:
    run_type: str

    n_classes: int
    n_way: int
    x_dim: int

    dropout: float
    lr: float
    weight_decay: float
    hidden_dim: int
    hidden_layers: int
    epochs: int = 1
    episodes: int = 20000

    z_dim: int = 64
    n_support: int = 5
    n_query: int = 30
    use_target_model: bool = False
    update_frequency: int = 25
    tau: float = 0.90
    proto_combinator: float = 0.1
    combinator_slope: float = 0.000045

class HyperParameterStore:
    def __init__(self):
        pass
        self.models = {
        "forest_models": {
            "forest_pnet_2hidden": ParameterStore(run_type="forest_pnet_2hidden", n_classes=7, x_dim=54, n_way=7,
                                                  dropout=0.117821,
                                                  lr=0.00740, weight_decay=0.0000074,
                                                  hidden_dim=64, hidden_layers=2),
            "forest_pnet_0hidden": ParameterStore(run_type="forest_pnet_0hidden", n_classes=7, x_dim=54, n_way=7,
                                                  dropout=0.190646,
                                                  lr=0.002229, weight_decay=0.000148,
                                                  hidden_dim=64, hidden_layers=0
                                                  ),
            "forest_pnet_wavelet_only_0hidden": ParameterStore(run_type="forest_pnet_wavelet_only_0hidden",
                                                               n_classes=7, x_dim=8, n_way=7,
                                                               dropout=0.253079,
                                                               lr=0.072684, weight_decay=0.0000004,
                                                               hidden_dim=64, hidden_layers=0),
            "forest_pnet_wavelet_only_3hidden": ParameterStore(run_type="forest_pnet_wavelet_only_3hidden",
                                                               n_classes=7, x_dim=8, n_way=7,
                                                               dropout=0.18591,
                                                               lr=0.003382, weight_decay=0.0000021,
                                                               hidden_dim=64, hidden_layers=3),

        },
        "vpn_models": {
            "vpn_3hidden": ParameterStore(run_type="vpn_3hidden", n_classes=5, x_dim=128, n_way=5,
                                          dropout=0.115648,
                                          lr=0.000514, weight_decay=0.004862, hidden_dim=64, hidden_layers=3),
            "vpn_3h_high_dropout": ParameterStore(run_type="vpn_3h_high_dropout", n_classes=5, x_dim=128, n_way=5,
                                                  dropout=0.33508, lr=0.003102, weight_decay=0.004199,
                                                  hidden_dim=64, hidden_layers=3),
            "vpn_1h": ParameterStore(run_type="vpn_1h", n_classes=5, x_dim=128, n_way=5,
                                     dropout=0.32884, lr=0.000233, weight_decay=0.001078,
                                     hidden_dim=64, hidden_layers=1),
            "vpn_basic": ParameterStore(run_type="vpn_basic", n_classes=5, x_dim=128, n_way=5,
                                        dropout=0.25,
                                        lr=0.0005, weight_decay=0, hidden_dim=64, hidden_layers=3),
            "vpn_min_ece": ParameterStore(run_type="vpn_min_ece", n_classes=5, x_dim=128, n_way=5,
                                          dropout=0.414646,
                                          lr=0.000669, weight_decay=0.00168, hidden_dim=64, hidden_layers=3),
            "vpn_bd": ParameterStore(run_type="vpn_bd_calib", n_classes=5, x_dim=128, n_way=5,
                                     dropout=0.26141, lr=0.000657, weight_decay=0, hidden_dim=64, hidden_layers=3,
                                     episodes=10000),
        }
    }

    def get_model_params(self, model_name: str, model_type: str) -> Optional[ParameterStore]:
        m_to_ret = self.models.get(model_type, {}).get(model_name, None)
        if m_to_ret is None:
            print("Unknown model type '{}'".format(model_name))
            print("Available models:", list(self.models.keys()))
        else:
            return m_to_ret