


from data.vpn_packets import load
from parameter_store import HyperParameterStore, ParameterStore
from runs.run_calibrate import calibrate_and_test
from runs.run_train import run_train
from utils.data import save_stats




OG_PATH_TEMPLATE = ("../../enc-vpn-uncertainty-class-repl/processed_data/stable_cal_fraction/"
                    "min_max_normalized/run{}/frac_{}")

def main():
    for i in range(5):
        params = HyperParameterStore().get_model_params("vpn_3hidden", "vpn_models")
        starting_z_dim = 64
        for j in range(8):
            foldername = f"decreasing_z_dim_outs/z{starting_z_dim}/run{i}"
            path_to_use = OG_PATH_TEMPLATE.format(i, 80)
            params.z_dim = starting_z_dim
            print(f"Running with {starting_z_dim} output neurons")
            decreasing_z_dim_runs(params, foldername, path_to_use)
            starting_z_dim -= 8



def decreasing_z_dim_runs(params: ParameterStore, foldername, path_to_use):

    val_acc, val_loss, pnet = run_train(params, full_path=path_to_use)


    test_acc, test_calib, test_micro, confusion = calibrate_and_test(
        pnet, False, False, params.n_classes, params.n_way, full_path=path_to_use
    )
    decreasing_z_dim_stat_saver(params.z_dim, val_acc, val_loss, test_acc, test_calib, test_micro, confusion, foldername)

def decreasing_z_dim_stat_saver(z_dim, val_acc, val_loss, test_acc, test_calib, test_micro, confusions, foldername):
    # data, name, foldername

    save_stats(val_acc, "train_acc", foldername)
    save_stats(val_loss, "train_loss", foldername)
    save_stats(test_acc, "test_acc", foldername)
    save_stats(test_calib, "test_calib", foldername)
    save_stats(test_micro, "test_micro", foldername)
    save_stats(confusions, "confusion", foldername)

if __name__ == "__main__":
    main()