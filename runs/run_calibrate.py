import torch

from data.vpn_packets import UT_MOBILE_CLASS_MAP, OG_CLASS_MAP, COMBINED_CLASS_MAP, ONE_OUT_CLASS_MAP #, load

#TODO: To test with vpn loader, import load from data.vpn_packets
#from data.forest_ct import load
from data.vpn_packets import load
from model.protonet import load_protonet_lin
import numpy as np
from tqdm import tqdm
import os
from utils.parameter_store import HyperParameterStore
from utils.data import sample_transform, save_stats
from utils.model import load_model
import time
from run_train import run_train
UT_MOBILE_PATH_TEMPLATE = ("../../enc-vpn-uncertainty-class-repl/processed_data/ut_mobile_data/stable_cal_fraction/"
                           "min_max_normalized/run{}/frac_{}")
OG_PATH_TEMPLATE = ("../../enc-vpn-uncertainty-class-repl/processed_data/stable_cal_fraction/"
                    "min_max_normalized/run{}/frac_{}")
CT_PATH = "../../forest_ct_data/covertype_data/splits/split0"

COMBINED_PATH = "../../enc-vpn-uncertainty-class-repl/processed_data/ut_vpn_validity_data/with_spotify"
OG_PATH = "../../enc-vpn-uncertainty-class-repl/processed_data/ut_vpn_validity_data/no_spotify"

OG_CLASS_COUNT = 5
UT_CLASS_COUNT = 16
CT_CLASS_COUNT = 7



def main():
    start = time.time()
    #protonet = load_model(x_dim=128, hid_dim=64, z_dim=64, dropout=0.079311, hidden_layers=3, path="./outs/vpn_3hidden.pt")
    params = HyperParameterStore().get_model_params("vpn_3hidden", "vpn_models")
    params.z_dim = 64
    _, _, protonet = run_train(params, full_path=OG_PATH) # train without spotify data
    calibration_data_path = (OG_PATH_TEMPLATE.format(1, 20))
    calibration_data_path = CT_PATH
    # get ood scores for spotify data
    run_calibrate_and_ood_comparison(frac=80, run_desc_og="vpn_calib_ood_before", run_desc_ut="ut_calib_ood_before",
                                     combined_path=COMBINED_PATH, cal_path=OG_PATH, protonet=protonet, n_class=params.n_classes, n_sup=params.n_support,
                                                 n_query=params.n_query, before_train=True)



    pre_train_output_values = calibrate_and_test(protonet, False, use_cuda=False,
                                                 n_classes=params.n_classes+1, n_way=params.n_way+1, n_sup=params.n_support,
                                                 n_query=params.n_query,
                                                 full_path=COMBINED_PATH)
    save_test_stats(pre_train_output_values, "test_caliber_on_one_run")
    params.n_classes += 1
    params.n_way += 1
    # Retrain with spotify data
    _, _, protonet = run_train(params, full_path=COMBINED_PATH)
    protonet.eval()
    # get ood scores for spotify data after retrain
    run_calibrate_and_ood_comparison(frac=80, run_desc_og="vpn_calib_ood_after", run_desc_ut="ut_calib_ood_after",
                                     combined_path=COMBINED_PATH, cal_path=COMBINED_PATH, protonet=protonet, n_class=params.n_classes, n_sup=params.n_support,
                                                 n_query=params.n_query)
    post_train_output_values = calibrate_and_test(protonet, True, use_cuda=False,
                                                  n_classes=params.n_classes, n_way=params.n_way, n_sup=params.n_support,
                                                 n_query=params.n_query,
                                                  full_path=COMBINED_PATH)
    save_test_stats(post_train_output_values, "retrained_test_caliber_on_one_run")



    exit(2)
    output_values = calibrate_and_test(protonet, False, use_cuda=False,
                                       n_classes=CT_CLASS_COUNT, n_way=CT_CLASS_COUNT,
                                       full_path=calibration_data_path)

    end = time.time()
    print("\nelapsed time: ", end-start) # 7 seconds with cuda
    print(output_values)
    run_desc = "forest_ct"
    save_test_stats(output_values, run_desc)




def calibrate_and_test(pnet, print_stats, use_cuda, n_classes, n_way, n_sup, n_query, full_path=None):
    calibration_classes, g_k, pnet = calibrate(full_path, n_classes, n_way, pnet, print_stats, use_cuda, n_sup, n_query)
    # TEST:
    test_dl = load_dataset("test", full_path, n_classes, n_way, n_sup, n_query)
    total_acc, acc_vals, pvals, total_pval, calibers, micros, confusions = 0, 0, 0, [], [], [], []
    for sample in test_dl:
        if use_cuda:
            sample["xs"], sample["xq"] = sample["xs"].cuda(), sample["xq"].cuda()
        # Run test algorithm on single sample from test_dl
        pvals, acc_vals, caliber, micro_f1, confusion =\
            pnet.test(sample_transform(sample, sample), g_k,
                      use_cuda, calc_stats=True)

        total_acc += acc_vals.mean()
        total_pval.append(np.mean(pvals))
        calibers.append(caliber.item())
        micros.append(micro_f1.item())
        confusions.append(confusion)

    if print_stats:
        print(f"Test accuracy (from last episode): {acc_vals.mean().item()}, \n"
              f"mean p value (from last episode): {np.mean(pvals)}\n"
              f"std p val (from last episode): {np.std(pvals)}\n"
              f"OOD Examples: {list[int](np.greater_equal(pvals, 0.95)).count(1)}\n"
              f"Proportion OOD: "
              f"{list[int](np.greater_equal(pvals, 0.95)).count(1)/(list[int](np.greater_equal(pvals, 0.95)).count(0)+1)}")

    return acc_vals.mean().item(), calibers[0], micros[0], confusions


def calibrate(full_path, n_classes, n_way, pnet, print_stats, use_cuda, n_sup, n_query):
    # For testing different splits
    sup_dl = load_dataset("train", full_path, n_classes, n_way, n_sup, n_query) # when splits==train, batch size == 100
    cal_dl = load_dataset("cal", full_path, n_classes, n_way, n_sup, n_query)
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

def calibrate_and_ood_score(pnet, print_stats, use_cuda, n_classes, n_way, combined_path, cal_path, n_sup, n_query,
                            ood_class: str, iid_class: str, class_map: dict, ut_mobile=False, before_train=False):
    '''
    if before_train:
        n_classes = n_way = n_classes-1
        '''
    x = "before training" if before_train else "after training"
    print(f"N-Classes used to generate g_k: {n_classes}, {x}")
    calibration_classes, g_k, pnet = calibrate(cal_path, n_classes, n_way, pnet, print_stats, use_cuda, n_sup, n_query)
    class_map = class_map
    if before_train:
        n_classes = n_way = n_classes+1
    if ut_mobile:
        specific_class = ood_class
        test_dl = load_dataset("test", combined_path, n_classes, n_way, n_sup, n_query)
    else:
        test_dl = load_dataset("test", combined_path, n_classes, n_way, n_sup, n_query)
        specific_class = iid_class

    class_specific_data = None
    for sample in test_dl:
        class_ind = class_map[specific_class]
        class_specific_data = sample["xq"][class_ind]

    pvals = pnet.ood_score(class_specific_data, g_k)
    return pvals

def run_calibrate_and_ood_comparison(frac, run_desc_og, run_desc_ut, combined_path, cal_path, protonet, n_class,
                                     n_sup, n_query, ood_class: str, iid_class: str, class_map: dict, before_train=False):
    pvals = calibrate_and_ood_score(protonet, print_stats=False, use_cuda=False,
                                    n_classes=n_class, n_way=n_class,
                                    combined_path=combined_path, cal_path=cal_path, ut_mobile=True,
                                    n_sup=n_sup, n_query=n_query, ood_class=ood_class,
                                    iid_class=iid_class,class_map=class_map, before_train=before_train)

    save_stats(pvals, f"pval_from_ood_class_frac{frac}", run_desc_ut)
    # Run on OG
    pvals = calibrate_and_ood_score(protonet, print_stats=True, use_cuda=False,
                                    n_classes=n_class, n_way=n_class,
                                    combined_path=combined_path, cal_path=cal_path, ut_mobile=False,
                                    n_sup=n_sup, n_query=n_query, ood_class=ood_class,
                                    iid_class=iid_class,class_map=class_map, before_train=before_train)

    save_stats(pvals, f"pvals_from_existing_class_frac{frac}", run_desc_og)
    print("SAVED")

def save_test_stats(output_values, run_desc):
    """
    Saves stats to a folder in ./eval_outs/run_desc
    :param output_values: values to save
    :param run_desc: folder to save stats to
    :return:
    """
    accs, calibs, micros, confusions = output_values
    save_stats(accs, "pnet_accs", run_desc)
    save_stats(calibs, "pnet_calibs", run_desc)
    save_stats(micros, "pnet_micros", run_desc)
    save_stats(confusions, "pnet_confusions", run_desc)


def load_dataset(split, full_path, n_classes, n_way, n_sup, n_query):
    file_path = f"{full_path}/{split}.jsonl" if full_path else None
    # TODO: FOR TREES FULL PATH IS CSV, Network is jsonl
    return load(split, 1, n_classes, n_way, n_sup, n_query, file_path)

if __name__ == '__main__':
    main()
