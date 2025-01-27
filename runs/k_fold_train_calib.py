
from runs.run_calibrate import calibrate_and_test
from runs.run_train import run_train

from utils.data import save_stats
from utils.model import load_model


def main():
    #run_train_k_folded()
    # OMIT USE OF 5% of TD as not enough support examples for VOIP. # OMIT 10% of TD as not enough support for VOIP
    FOREST_CT_TEMPLATE  = ("../../forest_ct_data/covertype_data/stable_cal_fraction/min_max_normalized/run{i}/frac_{percent_train}")
    ut_run_description = "ut_mobile_kfold_decreasing_fraction_64x64x64x64"
    run_description = "forest_pnet"
    run_decrease_fraction(FOREST_CT_TEMPLATE,True, n_classes=7, n_way=7, run_desc=run_description)
# TODO: UTMobile requires a change of the class map in vpn_packets.py Uncouple this
def run_decrease_fraction(path, save, n_classes, n_way, run_desc):
    """
    :param run_desc: description of run, used in naming output directory
    :param n_way: # of classes per episode
    :param n_classes: # of classes in dataset
    :param save: if true, save output arrays
    :param path: Path template to directory containing multiple runs, each with 8 different fractions of train data.
    Data should be stored in the format:
    :return: min_max_normalized/run{i}/frac_{percent_train}, where each run{i} (i in range(0,num_runs)):
    Contains decreasing fractions of test, train, cal.jsonl stored in: frac_y, (y in {10*j|0<=j<num_fractions})
    """
    num_runs = 10
    num_fractions = 4
    accs, calibers, micro_f1s, confusions = [], [], [], []
    for i in range(num_runs):
        test_accs, test_calibers, test_micro_f1s, test_confusions = [], [], [], []
        for j in range(num_fractions): # start from 1, 80-20 split too small for my processed data - note from callum on dec 2
            percent_train = int(80-10*j) # fraction from inner loop iterator
            if j == 8: percent_train=5
            print(f"Testing with {percent_train}% of training data, run {i}")
            # format datapath based on current iterator status
            #fp = f"../../enc-vpn-uncertainty-class-repl/processed_data/stable_cal_fraction/min_max_normalized/run{i}/frac_{percent_train}"
            fp = path.format(i=i, percent_train=percent_train)
            # runs and stores stats.
            test_acc, test_caliber, test_micro_f1, confusion = train_eval_with_data_path(fp, n_classes, n_way)
            test_accs.append(test_acc)
            test_calibers.append(test_caliber)
            test_micro_f1s.append(test_micro_f1)
            test_confusions.append(confusion)
        accs.append(test_accs)
        calibers.append(test_calibers)
        micro_f1s.append(test_micro_f1s)
        confusions.append(test_confusions)
    print("**** FINISHED ****")
    print("Test accuracies: ", accs)
    print("Test calibration errors: ", calibers)
    print("Test Micro F1's: ", micro_f1s)
    if save:
        save_stats(accs, "test_accs", run_desc)
        save_stats(calibers, "test_calibers", run_desc)
        save_stats(micro_f1s, "test_micro_f1s", run_desc)
        save_stats(confusions, "test_confusions", run_desc)


def train_eval_with_data_path(fp, n_classes, n_way):
    """
    Runs scrips in run_train.py and run_calibrate.py
    :param n_classes: # of classes in dataset
    :param n_way: # of classes per episode
    :param fp: Full path to folder containing: train, test and cal.jsonl
    :return:
    """
    dropout = 0.284728
    lr = 0.004171
    weight_decay = 0.000025
    hidden_dim = 161
    hidden_layers = 3
    train_acc, train_loss, net = run_train(n_classes, n_way, x_dim=54, hid_dim=hidden_dim, dropout=dropout, lr=lr,
                                           weight_decay=weight_decay, hidden_layers=hidden_layers, full_path=fp)
    #net = load_model(x_dim=54, hid_dim=hidden_dim, dropout=dropout, path="./outs/forest_pnet.pt")
    net.eval()
    test_acc, test_calib_err, micro_f1, confusion = (
        calibrate_and_test(net, False, False, n_classes, n_way, full_path=fp))
    return test_acc, test_calib_err, micro_f1, confusion

def run_train_k_folded(num_folds=3, num_runs=10):
    """
    I/P: num_folds, num_runs
    Notes: Should store folds in format: processed_data/{num_folds}_fold/run{i}/fold{j},
    in each fold{j}: train, test and cal.jsonl
    :return: accuracy, ece and micro_f1
    """
    test_accs, test_calibers, test_micro_f1s = [], [], []
    for i in range(num_runs):
        for j in range(num_folds):
            print("Testing on fold ", j)
            fp = f"../../enc-vpn-uncertainty-class-repl/processed_data/{num_folds}_fold/run{i}/fold{j}"
            test_accs, test_calibers, test_micro_f1s, confusion = train_eval_with_data_path(fp, n_classes=5, n_way=5)
    print("**** FINISHED ****")
    print("Test accuracies: ", test_accs)
    print("Test calibration errors: ", test_calibers)
    print("Test Micro F1's: ", test_micro_f1s)


if __name__ == '__main__':
    main()
