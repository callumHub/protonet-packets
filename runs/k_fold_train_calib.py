
from runs.run_calibrate import calibrate_and_test
from runs.run_train import run_train
import os
import numpy as np
def main():
    #run_train_k_folded()
    # OMIT USE OF 5% of TD as not enough support examples for VOIP. # OMIT 10% of TD as not enough support for VOIP
    run_decrease_fraction(True)

def run_train_k_folded():
    num_folds = 3
    num_runs = 10
    test_accs, test_calibers, test_micro_f1s = [], [], []
    for i in range(num_runs):
        for j in range(num_folds):
            print("Testing on fold ", j)
            fp = f"../../enc-vpn-uncertainty-class-repl/processed_data/{num_folds}_fold/run{i}/fold{j}"
            test_accs, test_calibers, test_micro_f1s, confusion = train_eval_with_data_path(fp)
    print("**** FINISHED ****")
    print("Test accuracies: ", test_accs)
    print("Test calibration errors: ", test_calibers)
    print("Test Micro F1's: ", test_micro_f1s)

def run_decrease_fraction(save):
    num_runs = 10
    num_fractions = 7
    accs, calibers, micro_f1s, confusions = [], [], [], []
    for i in range(num_runs):
        test_accs, test_calibers, test_micro_f1s, test_confusions = [], [], [], []
        for j in range(num_fractions):
            percent_train = 100-int(100*(0.2+(j*0.1)))
            if j == 8: percent_train=5
            print(f"Testing with {percent_train}% of training data, run {i}")
            fp = f"../../enc-vpn-uncertainty-class-repl/processed_data/fraction/run{i}/frac_{percent_train}"
            test_acc, test_caliber, test_micro_f1, confusion = train_eval_with_data_path(fp)
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
        save_stats(accs, "test_accs")
        save_stats(calibers, "test_calibers")
        save_stats(micro_f1s, "test_micro_f1s")
        save_stats(confusions, "test_confusions")


def train_eval_with_data_path(fp):
    train_acc, train_loss, net = run_train(full_path=fp)
    net.eval()
    test_acc, test_calib_err, micro_f1, confusion = calibrate_and_test(net, False, False, full_path=fp)
    return test_acc, test_calib_err, micro_f1, confusion


def save_stats(data, name):
   save_path = os.path.join("./eval_outs/fraction_nh_zdim_64")
   os.makedirs(save_path, exist_ok=True)
   np.save(os.path.join(save_path, name+".npy"), np.array(data))


if __name__ == '__main__':
    main()
