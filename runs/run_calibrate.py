import torch

from data.vpn_packets import load
from model.protonet import load_protonet_lin
import numpy as np
from tqdm import tqdm

from utils.data import sample_transform
from utils.model import load_model
import time

def main():
    start = time.time()
    protonet = load_model(path="./outs/pnet.pt")
    protonet.eval()
    for i in range(10):
        calibrate_and_test(protonet, True, use_cuda=False, full_path="../../enc-vpn-uncertainty-class-repl/processed_data/stable_cal_fraction/min_max_normalized/run0/frac_80") # faster without on mine data
    end = time.time()
    print("\nelapsed time: ", end-start) # 7 seconds with cuda

def calibrate_and_test(pnet, print_stats, use_cuda, full_path=None):

    # For testing different splits
    if full_path is not None:
        sup_dl = load("train", 1, 5, 5, full_path+"/train.jsonl")
        cal_dl = load("cal", 1, 5, 5, full_path+"/cal.jsonl")
    else:
        cal_dl = load("cal", 1, 5, 5)
        sup_dl = load("train", 1, 5, 5) # splits = train => batch size 100

    cal = {}
    sup = {}
    for sample in cal_dl:
        cal = { "class": sample["class"],
                "xq": sample["xq"],
                }
    for sample in sup_dl:
        sup = {
            "class": sample["class"],
            "xs": sample["xq"]
        }

    sample = cal
    sample.update(sup)
    sample = sample_transform(sample) # ensure class samples are ordered correctly

    if use_cuda:
        sample["xs"] = sample["xs"].cuda()
        sample["xq"] = sample["xq"].cuda()
        pnet = pnet.cuda()

    if print_stats:
        print(f"Support examples, per class, for alg2: {sample["xs"].size()[1]}")
        print(f"Calibration examples, per class, for alg 2: {sample["xq"].size()[1]}")


    g_k = pnet.calibrate(sample)

    # TEST:
    if full_path is not None:
        test_dl = load("test", 1, 5, 5, full_path+"/test.jsonl")
    else:
        test_dl = load("test", 1, 5, 5)
    total_acc = 0
    acc_vals = 0
    pvals = 0
    total_pval = []
    calibers = []
    micros = []
    for sample in test_dl:
        if use_cuda:
            sample["xs"] = sample["xs"].cuda()
            sample["xq"] = sample["xq"].cuda()
        pvals, acc_vals, caliber, micro_f1, confusions = pnet.test(sample_transform(sample), g_k, use_cuda)
        total_acc += acc_vals.mean()
        total_pval.append(np.mean(pvals))
        calibers.append(caliber.item())
        micros.append(micro_f1.item())
    if print_stats:
        print(f"Test accuracy (from last episode): {acc_vals.mean().item()}, \n"
              f"mean p value (from last episode): {np.mean(pvals)}\n"
              f"std p val (from last episode): {np.std(pvals)}\n"
              f"OOD Examples: {list[int](np.greater_equal(pvals, 0.95)).count(1)}\n"
              f"Proportion OOD: "
              f"{list[int](np.greater_equal(pvals, 0.95)).count(1)/list[int](np.greater_equal(pvals, 0.95)).count(0)}")

    return acc_vals.mean().item(), calibers[0], micros[0], confusions


if __name__ == '__main__':
    main()