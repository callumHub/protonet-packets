import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
def main():
    save_path="spotify_30run_vpn_3hidden_dcal_stabletrain_bd_with_target_model_update25"
    single = False
    total_runs = 10
    data_dict = load_data_into_frame(save_path, total_runs)

    if single:
        masking_effect = [get_masking_effect(data_dict["ood_before"][i]) for i in range(len(data_dict["ood_before"]))]
        masking_effect_std = np.std(masking_effect)
        masking_effect = np.mean(masking_effect)
        swamping_effect_ood = [get_swamping_effect(data_dict["ood_after"][i]) for i in
                               range(len(data_dict["ood_after"]))]
        swamping_effect_ood_std = np.std(swamping_effect_ood)
        swamping_effect_ood = np.mean(swamping_effect_ood)
        swamping_effect_iid = [get_swamping_effect(data_dict["iid_before"][i]) for i in
                               range(len(data_dict["iid_before"]))]
        swamping_effect_iid_std = np.std(swamping_effect_iid)
        swamping_effect_iid = np.mean(swamping_effect_iid)
        swamping_effect_iid_plus_class = [get_swamping_effect(data_dict["iid_after"][i]) for i in
                                          range(len(data_dict["iid_after"]))]
        swamping_effect_iid_plus_class_std = np.std(swamping_effect_iid_plus_class)
        swamping_effect_iid_plus_class = np.mean(swamping_effect_iid_plus_class)
        avg_caliber_before, std_caliber_before = get_performance_data(data_dict["calib_before"][:])
        avg_f1_before, std_f1_before = get_performance_data(data_dict["f1_before"][:])
        avg_acc_before, std_acc_before = get_performance_data(data_dict["acc_before"][:])
        avg_caliber_after, std_caliber_after = get_performance_data(data_dict["calib_after"][:])
        avg_f1_after, std_f1_after = get_performance_data(data_dict["f1_after"][:])
        avg_acc_after, std_acc_after = get_performance_data(data_dict["acc_after"][:])
        ood_roc_auc_before = [get_auroc(data_dict["ood_before"][i], data_dict["ood_after"][i]) for i in
                              range(len(data_dict["ood_before"]))]
        ood_roc_auc_before_std = np.std(ood_roc_auc_before)
        ood_roc_auc = np.mean(ood_roc_auc_before)
        # ood_roc_auc_after = get_auroc(data_dict["ood_before"][:, :], data_dict["ood_after"][:, :])
        out_string = (
            "{:.2f}-{:.2f}\t{:.2f}-{:.2f}\t{:.2f}-{:.2f}\t{:.2f}-{:.2f}\t{:.2f}-{:.2f}\t{:.2f}-{:.2f}\t{:.2f}-{:.2f}\t{:.2f}-{:.2f}\t{:.2f}-{:.2f}\t{:.2f}-{:.2f}\t{:.2f}"
            "-{:.2f}")
        out_string = out_string.format(masking_effect, masking_effect_std, swamping_effect_ood, swamping_effect_ood_std,
                                       swamping_effect_iid, swamping_effect_iid_std,
                                       swamping_effect_iid_plus_class, swamping_effect_iid_plus_class_std,
                                       avg_caliber_before, std_caliber_before, avg_f1_before, std_f1_before,
                                       avg_acc_before,
                                       std_acc_before,
                                       avg_caliber_after, std_caliber_after, avg_f1_after, std_f1_after, avg_acc_after,
                                       std_acc_after,
                                       ood_roc_auc, ood_roc_auc_before_std)
        print(out_string)
    else:
        cali_data = [150, 130, 111, 91, 72, 53, 33]
        percent_training_data= [20, 30, 40, 50, 60, 70, 80]
        for i in range(len(percent_training_data)):
            out_string = multi_run_string_maker(data_dict, i)
            print(percent_training_data[i],"% ", out_string)
        #for i in range(total_runs):
        #    print(f"{percent_training_data[i]}% ({cali_data[i]})")


def multi_run_string_maker(data_dict, num_run):
    masking_effect = [get_masking_effect(data_dict["ood_before"][num_run, i]) for i in range(len(data_dict["ood_before"][0, :]))]
    masking_effect_std = np.std(masking_effect)
    masking_effect = np.mean(masking_effect)
    swamping_effect_ood = [get_swamping_effect(data_dict["ood_after"][num_run, i]) for i in range(len(data_dict["ood_after"][0, :]))]
    swamping_effect_ood_std = np.std(swamping_effect_ood)
    swamping_effect_ood = np.mean(swamping_effect_ood)
    swamping_effect_iid = [get_swamping_effect(data_dict["iid_before"][num_run, i]) for i in range(len(data_dict["iid_before"][0, :]))]
    swamping_effect_iid_std = np.std(swamping_effect_iid)
    swamping_effect_iid = np.mean(swamping_effect_iid)
    swamping_effect_iid_plus_class = [get_swamping_effect(data_dict["iid_after"][i]) for i in
                                      range(len(data_dict["iid_after"]))]
    swamping_effect_iid_plus_class_std = np.std(swamping_effect_iid_plus_class)
    swamping_effect_iid_plus_class = np.mean(swamping_effect_iid_plus_class)
    avg_caliber_before, std_caliber_before = get_performance_data(data_dict["calib_before"][num_run, :])
    avg_f1_before, std_f1_before = get_performance_data(data_dict["f1_before"][num_run, :])
    avg_acc_before, std_acc_before = get_performance_data(data_dict["acc_before"][num_run, :])
    avg_caliber_after, std_caliber_after = get_performance_data(data_dict["calib_after"][num_run, :])
    avg_f1_after, std_f1_after = get_performance_data(data_dict["f1_after"][num_run, :])
    avg_acc_after, std_acc_after = get_performance_data(data_dict["acc_after"][num_run, :])
    ood_roc_auc_before = [get_auroc(data_dict["ood_before"][num_run, i], data_dict["ood_after"][num_run, i]) for i in
                          range(len(data_dict["ood_before"][0, :]))]
    ood_roc_auc_before_std = np.std(ood_roc_auc_before)
    ood_roc_auc = np.mean(ood_roc_auc_before)
    # ood_roc_auc_after = get_auroc(data_dict["ood_before"][:, :], data_dict["ood_after"][:, :])
    out_string = (
        "{:.2f}-{:.2f}\t{:.2f}-{:.2f}\t{:.2f}-{:.2f}\t{:.2f}-{:.2f}\t{:.2f}-{:.2f}\t{:.2f}-{:.2f}\t{:.2f}-{:.2f}\t{:.2f}-{:.2f}\t{:.2f}-{:.2f}\t{:.2f}-{:.2f}\t{:.2f}"
        "-{:.2f}")
    out_string = out_string.format(masking_effect, masking_effect_std, swamping_effect_ood, swamping_effect_ood_std,
                                   swamping_effect_iid, swamping_effect_iid_std,
                                   swamping_effect_iid_plus_class, swamping_effect_iid_plus_class_std,
                                   avg_caliber_before, std_caliber_before, avg_f1_before, std_f1_before, avg_acc_before,
                                   std_acc_before,
                                   avg_caliber_after, std_caliber_after, avg_f1_after, std_f1_after, avg_acc_after,
                                   std_acc_after,
                                   ood_roc_auc, ood_roc_auc_before_std)
    return out_string

    #masking_effect_for_decreasing_z_values(data_dict["ood_before"])
    #swamp_effect_for_decreasing_z_values(data_dict["iid_after"])

def get_auroc(true, true2, iid=False):
    pred = np.array([int(x>0.95) for x in true.flatten()])
    pred2 = np.array([int(x>0.95) for x in true2.flatten()])
    if iid:
        positive_label=0
        negative_label=1
    else:
        positive_label=1
        negative_label=0
    true = [positive_label]*len(pred)
    others = [negative_label]*len(pred2)
    true = np.append(true, others)
    pred = np.append(pred, pred2)
    fpr, tpr, thresholds = metrics.roc_curve(true, pred, pos_label=positive_label)
    roc_auc = metrics.auc(fpr, tpr)
    #metrics.RocCurveDisplay.from_predictions(true, pred)
    #plt.show()
    return roc_auc

def masking_effect_for_decreasing_z_values(ood_data):
    for i in range(ood_data.shape[1]):
        get_masking_effect(ood_data[:,i,:])

def swamp_effect_for_decreasing_z_values(iid_data):
    for i in range(iid_data.shape[1]):
        print(get_swamping_effect(iid_data[:,i,:]))

def load_data_into_frame(path_to_data, runs):
    def read_all_z_values(load_dir, run_count):
        run_number = 0
        z_dims = [64, 56, 48, 40, 32, 24, 16, 8, 4, 2, 1]
        z_dims = [64]
        z_dims = [20, 30, 40, 50, 60, 70, 80]
        all_ut_before, all_ut_after, all_vpn_before, all_vpn_after = [], [], [], []
        (all_calib_before, all_calib_after, all_f1_before,
         all_f1_after, all_acc_before, all_acc_after) = [], [], [], [], [], []
        for j in range(len(z_dims)):
            all_z_ut_before, all_z_ut_after, all_z_vpn_before, all_z_vpn_after = [], [], [], []
            (all_z_calib_before, all_z_calib_after, all_z_f1_before,
             all_z_f1_after, all_z_acc_before, all_z_acc_after) = [], [], [], [], [], []
            for i in range(run_count):
                path_template = f"../runs/eval_outs/{load_dir}/frac_{z_dims[j]}/run_{i}/"
                #path_template = f"../runs/eval_outs/{load_dir}/cal_{z_dims[i]}/run_{run_number}/"
                before_ut = path_template + f"ut_calib_ood_before/pval_from_ood_class_frac80.npy"
                before_vpn = path_template + "vpn_calib_ood_before/pvals_from_existing_class_frac80.npy"
                after_ut = path_template + f"ut_calib_ood_after/pval_from_ood_class_frac80.npy"
                after_vpn = path_template + "vpn_calib_ood_after/pvals_from_existing_class_frac80.npy"
                before_calib = path_template + "before_retrain_utvpn_output/pnet_calibs.npy"
                after_calib = path_template + "retrained_utvpn_output/pnet_calibs.npy"
                before_acc = path_template + "before_retrain_utvpn_output/pnet_accs.npy"
                after_acc = path_template + "retrained_utvpn_output/pnet_accs.npy"
                before_f1 = path_template + "before_retrain_utvpn_output/pnet_micros.npy"
                after_f1 = path_template + "retrained_utvpn_output/pnet_micros.npy"

                all_z_ut_before.append(np.load(before_ut))
                all_z_ut_after.append(np.load(after_ut))
                all_z_vpn_before.append(np.load(before_vpn))
                all_z_vpn_after.append(np.load(after_vpn))
                all_z_calib_before.append(np.load(before_calib))
                all_z_calib_after.append(np.load(after_calib))
                all_z_f1_before.append(np.load(before_f1))
                all_z_f1_after.append(np.load(after_f1))
                all_z_acc_before.append(np.load(before_acc))
                all_z_acc_after.append(np.load(after_acc))
            all_ut_before.append(all_z_ut_before)
            all_ut_after.append(all_z_ut_after)
            all_vpn_before.append(all_z_vpn_before)
            all_vpn_after.append(all_z_vpn_after)
            all_calib_before.append(all_z_calib_before)
            all_calib_after.append(all_z_calib_after)
            all_f1_before.append(all_z_f1_before)
            all_f1_after.append(all_z_f1_after)
            all_acc_before.append(all_z_acc_before)
            all_acc_after.append(all_z_acc_after)
            run_number += 1
        return (np.array(all_ut_before).squeeze(), np.array(all_ut_after).squeeze(),
                np.array(all_vpn_before).squeeze(), np.array(all_vpn_after).squeeze(),
                np.array(all_calib_before).squeeze(), np.array(all_calib_after).squeeze(),
                np.array(all_f1_before).squeeze(), np.array(all_f1_after).squeeze(),
                np.array(all_acc_before).squeeze(), np.array(all_acc_after).squeeze())

    ood_before, ood_after, iid_before, iid_after, calib_before, calib_after,\
    f1_before, f1_after, acc_before, acc_after = read_all_z_values(path_to_data, runs)
    return {
        "ood_before": ood_before,
        "ood_after": ood_after,
        "iid_before": iid_before,
        "iid_after": iid_after,
        "calib_before": calib_before,
        "calib_after": calib_after,
        "f1_before": f1_before,
        "f1_after": f1_after,
        "acc_before": acc_before,
        "acc_after": acc_after,
    }

def get_masking_effect(ood_scores):
    """
    :param ood_scores: numpy arr of ood scores
    :return: # of outliers falsely detected as inliers
    """
    total_examples = ood_scores.flatten().shape[-1]#*ood_scores.shape[0]
    total_iid = len(np.extract(ood_scores < 0.95, ood_scores))
    return total_iid/total_examples


def get_swamping_effect(iid_scores):
    """
    :param iid_scores: numpy arr of ood scores
    :return: # of inliers falsely detected as outliers
    """
    # Total examples are # of examples per run * number of runs.
    total_examples = iid_scores.flatten().shape[-1]#*iid_scores.shape[0]
    total_ood_per = len(np.extract(iid_scores >= 0.95, iid_scores))
    return total_ood_per/total_examples

def get_performance_data(performance_data):
    # shape: n_runs, n_z_dims

    average_metric = performance_data.mean(axis=0)
    std_metric = performance_data.std(axis=0)
    return average_metric, std_metric

if __name__ == '__main__':
    main()