from torch.utils.data import DataLoader
import torchnet as tnt
from data.base import EpisodicBatchSampler
# TODO: Find better way to plug in and plug out forest ct data
from data.vpn_packets import VPNDataset, load
#from data.forest_ct import load
from model.protonet import Protonet, load_protonet_lin
from engine import Engine
from torch.optim import Adam
import torch
import utils.model as model_utils
import os
#from run_calibrate import calibrate_and_test
from parameter_store import HyperParameterStore, ParameterStore

def main():
    # vpn path
    data_path = "../../enc-vpn-uncertainty-class-repl/processed_data/stable_cal_fraction/min_max_normalized/run0/frac_80"
    data_path = "../../forest_ct_data/covertype_data/splits/split0" # forest path
    params = HyperParameterStore().get_model_params("forest_pnet_2hidden", "forest_models")
    run_train(params, full_path=data_path)

    #protonet = model_utils.load_model(path="./outs/pnet.pt")
    #protonet.eval()
    #calibrate_and_test(protonet, False, False)


def run_train(params: ParameterStore, full_path=None):
    # initialize data loaders
    data_loader, val_loader = init_dataloaders(full_path, params.n_classes, params.n_way)
    # initialize protonet
    accuracy, loss, pnet = train_pnet(data_loader, val_loader, params)
    print(f"Accuracy: {accuracy}, Loss: {loss}")
    print("Saving Model... ")
    os.makedirs(os.path.join(os.getcwd(), "outs"), exist_ok=True)
    torch.save(pnet.state_dict(), os.path.join(os.getcwd(), "outs", f"{params.run_type}.pt"))
    print("Model saved to: ", os.path.join(os.getcwd(), "outs", f"{params.run_type}.pt"))
    return accuracy, loss, pnet


def init_dataloaders(full_path, n_classes, n_way):
    if full_path is not None:
        data_loader = load("train", 20000, n_classes, n_way, fp=full_path + "/train.jsonl")
        val_loader = load("train", 50, n_classes, n_way, fp=full_path + "/train.jsonl")
    else:
        data_loader = load("train", 20000, n_classes, n_way)
        val_loader = load("train", 50, n_classes, n_way)
    return data_loader, val_loader


def train_pnet(data_loader, val_loader, params: ParameterStore):
    pnet = get_new_pnet_for_train(params)
    engine = Engine()
    meter_fields = ['loss', 'acc']
    meters = {'train': {field: tnt.meter.AverageValueMeter() for field in meter_fields}}
    pnet.train()
    engine.train(
        model=pnet,
        loader=data_loader,
        optim_method=Adam,
        optim_config={'lr': params.lr,  # was 0.0001 for most 80% frac of data test, set to lower
                      'weight_decay': params.weight_decay},
        max_epoch=1
    )
    meters['val'] = {field: tnt.meter.AverageValueMeter() for field in meter_fields}
    meters = model_utils.evaluate(pnet,
                                  val_loader,
                                  meters['val'],
                                  desc=None ) #"Epoch {:d} valid".format(500))
    pnet.cpu()
    accuracy = meters['acc'].val
    loss = meters['loss'].val
    return accuracy, loss, pnet


def get_new_pnet_for_train(params: ParameterStore):
    pnet = load_protonet_lin(**{"x_dim": [params.x_dim], "hid_dim": params.hidden_dim, "z_dim": params.z_dim, "dropout": params.dropout,
                                "hidden_layers": params.hidden_layers})
    torch.cuda.manual_seed(1234)
    if torch.cuda.is_available():
        pnet.cuda()
    return pnet


if __name__ == '__main__':
    main()