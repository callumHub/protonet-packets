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
from show_results.result_plot import micro_f1_against_data_frac, ece_against_data_frac

def main():
    '''
    run_train()
    protonet = model_utils.load_model(path="./outs/pnet.pt")
    protonet.eval()
    calibrate_and_test(protonet, False, False)
    '''
    eces = []
    f1s = []
    for i in range(10):
        acc, ece, f1 = run_train_with_fraction_train()
        eces.append(ece)
        f1s.append(f1)
    micro_f1_against_data_frac(f1s)
    ece_against_data_frac(eces)


def run_train(full_path=None):
    # initialize data loaders
    if full_path is not None:
        data_loader = load("train", 20000, 5, 5, fp=full_path+"/train.jsonl")
        val_loader = load("test", 50, 5, 5, fp=full_path+"/test.jsonl")
    else:
        data_loader = load("train", 20000, 5, 5)
        val_loader = load("test", 50, 5, 5)
    # initialize protonet
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
        optim_config={'lr': 0.0005, # was 0.0001 for most 80% frac of data test, set to lower
                      'weight_decay': 0.0},
        max_epoch=1
    )

    meters['val'] = {field: tnt.meter.AverageValueMeter() for field in meter_fields}
    meters = model_utils.evaluate(pnet,
                         val_loader,
                         meters['val'],
                         desc="Epoch {:d} valid".format(500))
    pnet.cpu()
    accuracy = meters['acc'].val
    loss = meters['loss'].val
    print(f"Accuracy: {accuracy}, Loss: {loss}")
    print("Saving Model: ")
    os.makedirs(os.path.join(os.getcwd(), "outs"), exist_ok=True)
    torch.save(pnet.state_dict(), os.path.join(os.getcwd(), "outs", "pnet.pt"))
    print("Model saved to: ", os.path.join(os.getcwd(), "outs", "pnet.pt"))
    return accuracy, loss, pnet



def run_train_with_fraction_train():
    train_accs = []
    test_accs = []
    test_calibers = []
    test_micro_f1s = []
    for i in range(1, 8):
        print(f"***Training with {80 - i*10}% of entire data")
        fp = f"../../enc-vpn-uncertainty-class-repl/processed_data/fraction_train/train_{80 - i*10}.jsonl"
        train_acc, train_loss, net = run_train(full_path=fp)
        net.eval()
        test_acc, test_calib_err, micro_f1s = calibrate_and_test(net, True, False, full_path=fp)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        test_calibers.append(test_calib_err)
        test_micro_f1s.append(micro_f1s)
    print("**** FINISHED ****")
    print("Test accuracies: ", test_accs)
    print("Test calibration errors: ", test_calibers)
    print("Test Micro F1's: ", test_micro_f1s)
    return test_accs, test_calibers, test_micro_f1s

if __name__ == '__main__':
    main()