from torch.utils.data import DataLoader
import torchnet as tnt
from data.base import EpisodicBatchSampler
from data.vpn_packets import VPNDataset, load
from model.protonet import Protonet, load_protonet_lin
from engine import Engine
from torch.optim import Adam
import torch
import utils.model as model_utils
import utils.log as log_utils
import os
from run_calibrate import calibrate_and_test
def main():
    data_loader = load("train", 20000)

    val_loader = load("test", 50)

    pnet = load_protonet_lin(**{"x_dim": [128], "hid_dim": 64, "z_dim": 64})
    torch.cuda.manual_seed(1234)
    if torch.cuda.is_available():
        pnet.cuda()

    engine = Engine()

    meter_fields = ['loss','acc']

    meters = {'train': {field: tnt.meter.AverageValueMeter() for field in meter_fields}}
    pnet.train()
    engine.train(
        model=pnet,
        loader=data_loader,
        optim_method=Adam,
        optim_config={'lr': 0.0001,
                      'weight_decay': 0.0},
        max_epoch=1
    )

    meters['val'] = {field: tnt.meter.AverageValueMeter() for field in meter_fields}
    meters = model_utils.evaluate(pnet,
                         val_loader,
                         meters['val'],
                         desc="Epoch {:d} valid".format(500))
    pnet.cpu()
    print(f"Accuracy: {meters['acc'].val}, Loss: {meters['loss'].val}")
    print("Saving Model: ")
    os.makedirs(os.path.join(os.getcwd(), "outs"), exist_ok=True)
    torch.save(pnet.state_dict(), os.path.join(os.getcwd(), "outs", "pnet.pt"))
    print("Model saved to: ", os.path.join(os.getcwd(), "outs", "pnet.pt"))





if __name__ == '__main__':
    main()
    calibrate_and_test()