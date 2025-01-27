import torch

from data.vpn_packets import UT_MOBILE_CLASS_MAP, OG_CLASS_MAP #, load

#TODO: To test with vpn loader, import load from data.vpn_packets
from data.forest_ct import load

from model.protonet import load_protonet_lin
import numpy as np
from tqdm import tqdm
import os

from utils.data import sample_transform, save_stats
from utils.model import load_model
import time

UT_MOBILE_PATH_TEMPLATE = ("../../enc-vpn-uncertainty-class-repl/processed_data/ut_mobile_data/stable_cal_fraction/"
                           "min_max_normalized/run{}/frac_{}")
OG_PATH_TEMPLATE = ("../../enc-vpn-uncertainty-class-repl/processed_data/stable_cal_fraction/"
                    "min_max_normalized/run{}/frac_{}")
CT_PATH = "../../forest_ct_data/covertype_data/splits/split0"
SPECIFIC_UT_CLASS = "SPOTIFY"
SPECIFIC_OG_CLASS = "CHAT"
OG_CLASS_COUNT = 5
UT_CLASS_COUNT = 16
CT_CLASS_COUNT = 7



def main():
    start = time.time()
    protonet = load_model(x_dim=54, hid_dim=125, dropout=0.079311, hidden_layers=4, path="./outs/forest_pnet.pt")
    protonet.eval()
    calibration_data_path = (OG_PATH_TEMPLATE.format(1, 20))
    calibration_data_path = CT_PATH
    #run_calibrate_and_ood_comparison(frac, og_run_desc, ut_run_desc)
    output_values = calibrate_and_test(protonet, False, use_cuda=False,
                                       n_classes=CT_CLASS_COUNT, n_way=CT_CLASS_COUNT,
                                       full_path=calibration_data_path)

    end = time.time()
    print("\nelapsed time: ", end-start) # 7 seconds with cuda
    print(output_values)
    accs, calibs, micros, confusions = output_values
    save_stats(accs, "forest_pnet_accs", "forest_pnet")
    save_stats(calibs, "forest_pnet_calibs", "forest_pnet")
    save_stats(micros, "forest_pnet_micros", "forest_pnet")
    save_stats(confusions, "forest_pnet_confusions", "forest_pnet")

def calibrate_and_test(pnet, print_stats, use_cuda, n_classes, n_way, full_path=None):
    calibration_classes, g_k, pnet = calibrate(full_path, n_classes, n_way, pnet, print_stats, use_cuda)
    # TEST:
    test_dl = load_dataset("test", full_path, n_classes, n_way)
    total_acc, acc_vals, pvals, total_pval, calibers, micros = 0, 0, 0, [], [], []
    for sample in test_dl:
        if use_cuda:
            sample["xs"], sample["xq"] = sample["xs"].cuda(), sample["xq"].cuda()
        # Run test algorithm on single sample from test_dl
        pvals, acc_vals, caliber, micro_f1, confusions =\
            pnet.test(sample_transform(sample, sample), g_k,
                      use_cuda, calc_stats=True)

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


def calibrate(full_path, n_classes, n_way, pnet, print_stats, use_cuda):
    # For testing different splits
    sup_dl = load_dataset("train", full_path, n_classes, n_way) # when splits==train, batch size == 100
    cal_dl = load_dataset("cal", full_path, n_classes, n_way)

    # TODO: Due to how min classes affect n_way, calibrate and test dl can have different # of classes. BUG
    cal, sup, calibration_classes = {}, {}, {}
    for sample in cal_dl:
        cal = {"class": sample["class"],
               "xq": sample["xq"],
               }
        calibration_classes = {"class": sample["class"]}
    for sample in sup_dl:
        sup = {
            "class": sample["class"],
            "xs": sample["xq"]
        }
    sample = {**cal, **sup}
    sample = sample_transform(sample, calibration_classes)  # ensure class samples are ordered correctly
    if use_cuda:
        sample["xs"], sample["xq"] = sample["xs"].cuda(), sample["xq"].cuda()
        pnet = pnet.cuda()

    if print_stats:
        print(f"Support examples, per class, for alg2: {sample["xs"].size()[1]}")
        print(f"Calibration examples, per class, for alg 2: {sample["xq"].size()[1]}")
    # Calibrate and obtain gaussian kernels, g_k,  for each class.
    g_k = pnet.calibrate(sample)
    return calibration_classes, g_k, pnet

def calibrate_and_ood_score(pnet, print_stats, use_cuda, n_classes, n_way, full_path=None, ut_mobile=False,):
    calibration_classes, g_k, pnet = calibrate(full_path, n_classes, n_way, pnet, print_stats, use_cuda)

    if ut_mobile:
        ut_mobile_data_path = UT_MOBILE_PATH_TEMPLATE.format(1, 20)  # UTMOBILE TESTING, run1, frac20
        n_way = n_classes = UT_CLASS_COUNT
        specific_class = SPECIFIC_UT_CLASS
        class_map = UT_MOBILE_CLASS_MAP
        test_dl = load_dataset("test", ut_mobile_data_path, n_classes, n_way)
    else:
        test_dl = load_dataset("test", full_path, n_classes, n_way)
        specific_class = SPECIFIC_OG_CLASS
        class_map = OG_CLASS_MAP

    class_specific_data = None
    for sample in test_dl:
        class_specific_data = sample["xq"][class_map[specific_class]]

    pvals = pnet.ood_score(class_specific_data, g_k)
    return pvals

def run_calibrate_and_ood_comparison(frac, run_desc_og, run_desc_ut):
    protonet = load_model(x_dim=54, hid_dim=125, dropout=0.079311, hidden_layers=4, path="./outs/p_value_testing_pnet.pt")
    protonet.eval()
    full_path = OG_PATH_TEMPLATE.format(1, frac)
    pvals = calibrate_and_ood_score(protonet, print_stats=False, use_cuda=False,
                                    n_classes=OG_CLASS_COUNT, n_way=OG_CLASS_COUNT,
                                    full_path=full_path, ut_mobile=True)

    save_stats(pvals, f"after_train_pvals_og_to_ut_frac{frac}", run_desc_ut)
    # Run on OG
    pvals = calibrate_and_ood_score(protonet, print_stats=True, use_cuda=False,
                                    n_classes=OG_CLASS_COUNT, n_way=OG_CLASS_COUNT,
                                    full_path=full_path, ut_mobile=False)

    save_stats(pvals, f"pvals_og_to_ut_frac{frac}", run_desc_og)
    print("SAVED")

def load_dataset(split, full_path, n_classes, n_way):
    file_path = f"{full_path}/{split}.jsonl" if full_path else None
    # TODO: FOR TREES FULL PATH IS CSV, Network is jsonl
    return load(split, 1, n_classes, n_way, file_path)

if __name__ == '__main__':
    main()
