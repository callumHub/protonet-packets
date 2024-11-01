
from runs.run_calibrate import calibrate_and_test
from runs.run_train import run_train


def run_train_k_folded():
    num_folds = 3
    num_runs = 15
    train_accs = []
    test_accs = []
    test_calibers = []
    test_micro_f1s = []
    for i in range(num_runs):
        for j in range(num_folds):
            print("Testing on fold ", j)
            fp = f"../../enc-vpn-uncertainty-class-repl/processed_data/{num_folds}_fold/run{i}/fold{j}"
            train_acc, train_loss, net = run_train(full_path=fp)
            net.eval()
            test_acc, test_calib_err, micro_f1s = calibrate_and_test(net, False, False, full_path=fp)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            test_calibers.append(test_calib_err)
            test_micro_f1s.append(micro_f1s)
    print("**** FINISHED ****")
    print("Test accuracies: ", test_accs)
    print("Test calibration errors: ", test_calibers)
    print("Test Micro F1's: ", test_micro_f1s)



if __name__ == '__main__':
    run_train_k_folded()