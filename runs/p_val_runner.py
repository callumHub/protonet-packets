import copy
import gc
from time import sleep

from utils.parameter_store import HyperParameterStore
from runs.run_calibrate import run_calibrate_and_ood_comparison, calibrate_and_test, save_test_stats
from runs.run_train import run_train
import sys
import logging
import os
from data.vpn_packets import COMBINED_CLASS_MAP, OG_CLASS_MAP

import torch
import numpy as np

np.random.seed(42)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


#COMBINED_PATH = "../../enc-vpn-uncertainty-class-repl/processed_data/coarse_grain_ood/all_classes" # coarse 4 class combined to 5
#OG_PATH = "../../enc-vpn-uncertainty-class-repl/processed_data/coarse_grain_ood/no_ft" # coarse 4 class

#COMBINED_PATH = "../../enc-vpn-uncertainty-class-repl/processed_data/stable_cal_fraction_ut/min_max_normalized/decreasing_cal_decrease_train/run0/frac_60"

COMBINED_PATH = "../../forest_ct_data/covertype_data/splits/split0"
OG_PATH = COMBINED_PATH

#COMBINED_PATH = "../../enc-vpn-uncertainty-class-repl/processed_data/coarse_grain_ood_type2/all_classes"
#OG_PATH = "../../enc-vpn-uncertainty-class-repl/processed_data/coarse_grain_ood_type2/no_chat"

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG,\
                        datefmt='%Y-%m-%d %H:%M:%S', filename='model_experiments.log', filemode='a')
logger = logging.getLogger(__name__)

def cli_runner():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG,\
                        datefmt='%Y-%m-%d %H:%M:%S', filename='model_experiments.log', filemode='a')
    logger = logging.getLogger(__name__)
    logger.info("*****************************************************************************************************")
    logger.info("*****************************************************************************************************")
    logger.info("\t\tSTARTING NEW RUN")
    logger.info("*****************************************************************************************************")
    logger.info("*****************************************************************************************************")
    run_type = sys.argv[1]
    cand_params = sys.argv[2:]
    logger.info(f"run_type: {run_type}")

    # default values
    n_classes = n_way = 5
    lr = 0.01
    dropout = 0.25
    weight_decay = 0
    hidden_layer = 3
    model_name = ""
    combined_path = ""
    one_out_path = ""
    episodes_per_epoch = 20000
    num_runs  = 30
    save_folder = None
    iid_class = None
    ood_class = None
    while cand_params:
        param = cand_params.pop(0)
        value = cand_params.pop(0)
        logger.info(f"Param pased: {param} = {value}")
        match param:
            case "--combined_path":
                combined_path = value
            case "--one_out_path":
                one_out_path = value
            case "--model_name":
                model_name = value
            case "--n_way":
                n_way = int(value)
            case "--n_classes":
                n_classes = int(value)
            case "--episodes_per_epoch":
                episodes_per_epoch = int(value)
            case "--num_runs":
                num_runs = int(value)
            case "--save_folder":
                save_folder = value
            case "--iid_class":
                iid_class = value
            case "--ood_class":
                ood_class = value

    ok_name, valid_names = validate_model_name(model_name)
    if not ok_name: print(f"Model name: {model_name} invalid, valid models are: \n{valid_names}"); exit(0);
    if validate_path(combined_path) or validate_path(one_out_path) == False: print("Invalid path"); exit(0);
    if save_folder is None: print("Must specify save folder"); exit(0);
    class_map = OG_CLASS_MAP
    if n_classes == 6: class_map = COMBINED_CLASS_MAP

    params = HyperParameterStore().get_model_params(model_name, "vpn_models")


    # start experiment (pval)
    #params.n_classes = params.n_way = n_classes -1 # start with minority classes
    for i in range(num_runs):
        pval_experiment_runner(params, i, run_desc=f"{save_folder}/z_64/run_{i}/",
                               ood_class=ood_class, iid_class=iid_class,
                               combined_path=combined_path, one_out_path=one_out_path, class_map=class_map,)
    '''
    # start decreasing frac experiment
    print("Running Increasing Calibration Data Experiment")
    fracs = [20, 30, 40, 50, 60, 70, 80]
    for i in range(num_runs):
        for j in range(7):
            combined_path = f"../../enc-vpn-uncertainty-class-repl/processed_data/decreasing_cal_ood_data/run{i}/frac_{fracs[j]}/all_classes"
            one_out_path = f"../../enc-vpn-uncertainty-class-repl/processed_data/decreasing_cal_ood_data/run{i}/frac_{fracs[j]}/no_ft"
            pval_experiment_runner(params, i, run_desc=f"{save_folder}/cal_{fracs[j]}/run_{i}/",
                                   ood_class=ood_class, iid_class=iid_class,
                                   combined_path=combined_path, one_out_path=one_out_path, class_map=class_map, )

    '''



def main():
    import utils.experiment_context
    # For this run I changed mean bandwidth to specific bw for each class
    model_names = ["forest_3hidden_tunewidth"]#, "vpn_1h", "vpn_min_ece", "vpn_3h_high_dropout"]
    for model_name in model_names:
        utils.experiment_context.bandwidth_experiment = True
        n_classes = 7
        params = HyperParameterStore().get_model_params(model_name, "forest_models")
        params.n_classes = n_classes
        params.n_way = n_classes
        params.use_target_model = False
        pre_experiment_params = copy.deepcopy(params)
        decrease_z_dim_p_val_experiment(params, save_folder=f"with_val_forest_ct_{model_name}",#_bd_with_target_model_update25",
                                        ood_class=2, iid_class=1,
                                    combined_path=COMBINED_PATH, one_out_path=OG_PATH)

    exit(11)
    while True:
        try:
            decrease_z_dim_p_val_experiment(params, save_folder="vpnbd_dcal_stabletrain_bd_with_target_model",
                                            ood_class="SPOTIFY", iid_class="CHAT",
                                            combined_path=COMBINED_PATH, one_out_path=OG_PATH)
            break
        except Exception as e:
            print(e, flush=True)
            sleep(1)
            current_update_freq = params.update_frequency
            if current_update_freq > params.episodes:
                print("RAISING THE UPDATE FREQUENCY DID NOT HELP, UPDATE FREQUENCY == Num Episodes.")
                break
            print("NEURAL COLLAPSE: Incrementing Update Frequency By 100, (Current Update Frequency: {})".format(params.update_frequency))
            pre_experiment_params.update_frequency = current_update_freq+100
            params = copy.deepcopy(pre_experiment_params)


    pass
    '''
    minority_n_classes = 4
    params = HyperParameterStore().get_model_params("vpn_3hidden", "vpn_models")
    params.n_classes = minority_n_classes
    params.n_way = minority_n_classes
    decrease_z_dim_p_val_experiment(params, save_folder="hold_out_ft_decreasing_z_dim")
    params = HyperParameterStore().get_model_params("vpn_3h_high_dropout", "vpn_models")
    params.n_classes = minority_n_classes
    params.n_way = minority_n_classes
    decrease_z_dim_p_val_experiment(params, save_folder="hold_out_ft_decreasing_z_dim_high_dropout")
    params = HyperParameterStore().get_model_params("vpn_1h", "vpn_models")
    params.n_classes = minority_n_classes
    params.n_way = minority_n_classes
    decrease_z_dim_p_val_experiment(params, save_folder="hold_out_ft_decreasing_z_dim_1h")
    params = HyperParameterStore().get_model_params("vpn_basic", "vpn_models")
    params.n_classes = minority_n_classes
    params.n_way = minority_n_classes
    decrease_z_dim_p_val_experiment(params, save_folder="hold_out_ft_decreasing_z_dim_basic")
    '''
def decrease_z_dim_p_val_experiment(params, ood_class, iid_class, save_folder, combined_path, one_out_path):
    import utils.experiment_context
    fracs = [20, 30, 40, 50, 60, 70, 80]
    #fracs = [50, 60, 70, 80] # Debug: instability happens at these fracs.
    bws = [0.05]
    fracs = [80]
    for k in range(10):
        for i in range(10):
            for j in range(len(fracs)):
                run_number = i+(k*10)
                print("RUNNING WITH FRAC ", fracs[j])
                combined_path = f"../../enc-vpn-uncertainty-class-repl/processed_data/decreasing_cal_stable_train_ut/run{i}/frac_{fracs[j]}/all_classes"
                one_out_path = f"../../enc-vpn-uncertainty-class-repl/processed_data/decreasing_cal_stable_train_ut/run{i}/frac_{fracs[j]}/no_spotify"
                combined_path = f"../../enc-vpn-uncertainty-class-repl/processed_data/stable_cal_fraction_forest_ct/min_max_normalized/decreasing_cal_stable_train/run{i}/frac_80"
                one_out_path = combined_path
                #utils.experiment_context.bandwidth_value = bws[0]
                pval_experiment_runner(params, run_number, run_desc=f"{save_folder}/frac_{fracs[j]}/run_{run_number}/",
                                       ood_class=ood_class, iid_class=iid_class, combined_path=combined_path,
                                       one_out_path=one_out_path, class_map=OG_CLASS_MAP)


def pval_experiment_runner(params, run_count, run_desc, ood_class: str, iid_class: str, combined_path: str,
                           one_out_path, class_map: dict):

    curr_classes = params.n_classes
    #params.n_classes = curr_classes - 1
    print(params.n_classes)
    _, _, protonet = run_train(params, full_path=one_out_path)  # train without spotify data

    protonet.eval()
    # get ood scores for ood data
    run_calibrate_and_ood_comparison(frac=80, run_desc_og=run_desc+"vpn_calib_ood_before", run_desc_ut=run_desc+"ut_calib_ood_before",
                                     combined_path=combined_path, cal_path=one_out_path, protonet=protonet,
                                     n_class=params.n_classes, n_sup=params.n_support, n_query=params.n_query,
                                     ood_class=ood_class, iid_class=iid_class, class_map=class_map, before_train=True)
    pre_train_output_values = calibrate_and_test(protonet, False, use_cuda=False,
                                                 n_classes=params.n_classes, n_way=params.n_way,
                                                 full_path=combined_path, n_sup=params.n_support, n_query=params.n_query)
    save_test_stats(pre_train_output_values, run_desc+"before_retrain_utvpn_output")
    torch.save(protonet.state_dict(), os.path.join(os.getcwd(), "outs", f"{params.run_type}_before.pt"))
    # Retrain with ood data
    params.n_classes = params.n_way = curr_classes
    #_, _, protonet = run_train(params, full_path=combined_path)

    protonet.eval()
    # get ood scores for spotify data after retrain
    run_calibrate_and_ood_comparison(frac=80, run_desc_og=run_desc+"vpn_calib_ood_after", run_desc_ut=run_desc+"ut_calib_ood_after",
                                     combined_path=combined_path, cal_path=combined_path, protonet=protonet,
                                     n_class=params.n_classes, n_sup=params.n_support, n_query=params.n_query,
                                     ood_class=ood_class, iid_class=iid_class, class_map=class_map, before_train=False)
    post_train_output_values = calibrate_and_test(protonet, True, use_cuda=False,
                                                  n_classes=params.n_classes, n_way=params.n_way,
                                                  full_path=combined_path, n_sup=params.n_support, n_query=params.n_query)
    save_test_stats(post_train_output_values, run_desc+"retrained_utvpn_output")
    params.n_classes = params.n_way = curr_classes

def validate_model_name(model_name):
    models = ["vpn_3hidden", "vpn_3h_high_dropout", "vpn_1h", "vpn_basic", "vpn_min_ece"]
    if model_name in models:
        return True, None
    else: return False, models

def validate_path(path):
    required_files = ["train.jsonl", "cal.jsonl", "test.jsonl"]
    if not os.path.isdir(path):
        return False
    for f in os.listdir(path):
        if f in required_files:
            pass
        else:
            return False

if __name__ == "__main__":
    main()
    #cli_runner()