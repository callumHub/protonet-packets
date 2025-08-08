#!/home/callumm/.virtualenvs/enc-vpn-uncertainty-class-repl/bin/python
from runs.run_calibrate import load_dataset
import numpy as np
# used 'dos2unix' so that I can run this with wsl irace
from utils.parameter_store import ParameterStore
from utils.data import sample_transform
import scipy
import os

def main():
    import sys
    from runs.run_train import init_dataloaders, train_pnet
    import random
    import numpy as np
    import torch
    import logging

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG,\
                        datefmt='%Y-%m-%d %H:%M:%S',\
                        filename='irace_runner.log', filemode='a')
    logger = logging.getLogger(__name__)
    if not torch.cuda.is_available():
        logger.error("CUDA NOT AVAILABLE")
        exit(1)
    configuration_id = sys.argv[1]
    instance_id = sys.argv[2]
    seed = sys.argv[3]
    instance = sys.argv[4]
    cand_params = sys.argv[5:]

    # get random run split to use:
    run = np.random.randint(0, 10)
    logger.debug(
        f"Configuration ID: {configuration_id}, Instance ID: {instance_id}, Seed: {seed}, Instance: {instance}, Data split: run{run}")


    # set seeds
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    # data parameters
    n_classes = 5
    n_way = 5

    #default values
    lr = 0.03421
    weight_decay = 0.002298
    hidden_size = 64
    output_size = 64
    dropout = 0.107547
    hidden_layers = 3
    while cand_params:
        param = cand_params.pop(0)
        value = cand_params.pop(0)
        logger.debug(f"Parameter parsed: {param} = {value}")
        match param:
            case "--lr":
                lr = float(value)
            case "--weight_decay":
                weight_decay = float(value)
            case "--hidden_size":
                hidden_size = int(value)
            case "--dropout":
                dropout = float(value)
            case "--hidden_layers":
                hidden_layers = int(value)
    # vpn data
    data_path = "../../enc-vpn-uncertainty-class-repl/processed_data/stable_cal_fraction/min_max_normalized_20cal_val/run0/frac_60"
    #data_path = f"../../forest_ct_data/covertype_data/with_wavelets/stable_cal_fraction/min_max_normalized/run0/frac_80" # forest data
    params = ParameterStore(lr=lr, weight_decay=weight_decay, hidden_layers=hidden_layers, dropout=dropout, hidden_dim=hidden_size,
                            n_classes=n_classes, n_way=n_way, x_dim=128, run_type="irace_run", epochs=1, episodes=10000,
                            z_dim=64)
    train_dl, val_dl = init_dataloaders(full_path=data_path, params=params)
    excep_occured = False
    try:
        acc, loss, pnet = train_pnet(train_dl, val_dl, params)
    except Exception as e:
        logger.error(e)
        print(1.0)
        excep_occured = True


    if acc < 0.7:
        logger.warning("Accuracy lower than 0.7, Cov could be singular.\nAccuracy: {}".format(acc))
        print(1.0)
    elif acc > 0.7:
        if excep_occured:
            print("1.0")
        else:
            logger.info(f"Training completed. Train Accuracy: {acc}, Train Loss: {loss}")
            try:
                acc, calib, micro, confusion = irace_calibrate_and_test(pnet, False, False,
                                                params.n_classes, params.n_way, params.n_support, params.n_query,
                                                  full_path=data_path)
                logger.info(f"Calibration completed. Calibration Error: {calib}")
                print(calib)
            except (np.linalg.LinAlgError, scipy.integrate.IntegrationWarning) as e:
                if e == np.linalg.LinAlgError:
                    logger.info(f"Lin alg error in calib, Train Loss: {loss}")
                elif e == scipy.integrate.IntegrationWarning:
                    logger.info(f"Integration error in calib, Train Loss: {loss}")
                calib = 1.0
                print(calib)

    else:
        print(1.0)




def irace_calibrate_and_test(pnet, print_stats, use_cuda, n_classes, n_way, n_sup, n_query, full_path=None):
    calibration_classes, g_k, pnet = irace_calibrate(full_path, n_classes, n_way, pnet, print_stats, use_cuda, n_sup, n_query)
    # TEST:
    test_dl = load_dataset("val", full_path, n_classes, n_way, n_sup, n_query)
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
              f"{list[int](np.greater_equal(pvals, 0.95)).count(1)/list[int](np.greater_equal(pvals, 0.95)).count(0)}")

    return acc_vals.mean().item(), calibers[0], micros[0], confusions


def irace_calibrate(full_path, n_classes, n_way, pnet, print_stats, use_cuda, n_sup, n_query):
    # For testing different splits
    sup_dl = load_dataset("train", full_path, n_classes, n_way, n_sup, n_query) # when splits==train, batch size == 100
    cal_dl = load_dataset("val", full_path, n_classes, n_way, n_sup, n_query)
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

if __name__ == '__main__':
    main()
