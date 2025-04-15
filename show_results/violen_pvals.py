import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
def main():
    num_runs = 30
    load_dir = "cli_vpn1h_0p025"
    ut_before, ut_after, vpn_before, vpn_after = read_all_z_values(load_dir, num_runs)
    # Now, I have lists of the form:
    # shape: (n_runs, len(z_dims), len(test_data))
    # I want to have violen plots comparing: _____????
    # Compare each z_dims distribution of ood scores (from which run?).
    # Compare each of the 10 runs z_dims
    # Average each of the 10 runs ood scores and compare z_dims
    print("ut_before: ", np.mean(ut_before))
    z_dims = [64, 56, 48, 40, 32, 24, 16, 8, 4, 2, 1]
    z_dims = [20, 30, 40, 50, 60, 70, 80]
    z_dims = [64]
    #z_dims = [0.05]
    #z_dims = [0.05, 0.025, 0.01, 0.005, 0.001, 0.00075, 0.0005, 0.00025]
    run = 10
    ut_df_before, ut_df_after = prepare_df(ut_before, ut_after)
    vpn_df_before, vpn_df_after = prepare_df(vpn_before, vpn_after)



    ut_df_before["avg_OOD"] = average_all_ood(ut_df_before, num_runs)
    ut_df_after["avg_OOD"] = average_all_ood(ut_df_after, num_runs)
    print("mean of avg ood score (before train)", ut_df_before["avg_OOD"].mean(), ut_df_before["avg_OOD"].min(), ut_df_before["avg_OOD"].max())
    print("mean of avg ood score (after train)", ut_df_after["avg_OOD"].mean(), ut_df_after["avg_OOD"].min(), ut_df_after["avg_OOD"].max())
    ut_df = pd.concat([ut_df_before, ut_df_after], ignore_index=True)




    ut_df["Z_value"] = ut_df["Z_dim"].map(lambda x: z_dims[x])
    vpn_df_before["avg_OOD"] = average_all_ood(vpn_df_before, num_runs)
    vpn_df_after["avg_OOD"] = average_all_ood(vpn_df_after, num_runs)
    vpn_df = pd.concat([vpn_df_before, vpn_df_after], ignore_index=True)
    vpn_df["Z_value"] = vpn_df["Z_dim"].map(lambda x: z_dims[x])



    #ut_df = ut_df[ut_df["Run"]==run]
    # Violin Plot 1: Compare OOD score distributions for each z_dim (aggregated over runs)
    plt.figure(figsize=(14, 7))  # Increase figure size
    sns.violinplot(
        data=ut_df,
        x="Z_value",
        y="avg_OOD",
        hue="Condition",
        split=False,  # Change to False for better visibility
        dodge=True,  # Separate distributions
        inner=None,  # Adds a boxplot inside each violin
        linewidth=1.2,  # Make outlines sharper
        palette="Set2",
        bw_method=0.05,
        width=0.6,
        cut=-0.1  # Avoid long tails from outliers
    )
    plt.title(f"\'FILE TRANSFER\'OOD Data (30-run average) bw: 0.025, IRACE OPTIMIZED 1 Hidden Layer")
    plt.legend(title="Condition", loc="lower left")
    plt.grid(axis="y", linestyle="--", alpha=0.7)  # Add light gridlines for readability
    plt.xticks(fontsize=12)  # Make axis labels larger
    plt.yticks(fontsize=12)
    plt.xlabel("Bin Width used to Generate KDE's")
    plt.show()




def read_all_z_values(load_dir, run_count):
    run_number = 0
    z_dims = [64, 56, 48, 40, 32, 24, 16, 8, 4, 2, 1]
    z_dims = [20, 30, 40, 50, 60, 70, 80]
    z_dims = [1.5, 1.0, 0.5, 0.3, 0.2, 0.1, 0.075, 0.05, 0.025, 0.01, 0.005, 0.001]
    z_dims = [64]
    #z_dims = [0.05]
    #z_dims = [0.05, 0.025, 0.01, 0.005, 0.001, 0.00075, 0.0005, 0.00025]
    all_ut_before, all_ut_after, all_vpn_before, all_vpn_after = [],[],[],[]
    while run_number < run_count:
        all_z_ut_before, all_z_ut_after, all_z_vpn_before, all_z_vpn_after =  [],[],[],[]
        for i in range(len(z_dims)):
            #path_template = f"../runs/eval_outs/{load_dir}/bw_{z_dims[i]}/run_{run_number}/"
            path_template = f"../runs/eval_outs/{load_dir}/z_{z_dims[i]}/run_{run_number}/"
            before_ut = path_template + f"ut_calib_ood_before/pval_from_ood_class_frac80.npy"
            before_vpn = path_template + "vpn_calib_ood_before/pvals_from_existing_class_frac80.npy"
            after_ut = path_template + f"ut_calib_ood_after/pval_from_ood_class_frac80.npy"
            after_vpn = path_template + "vpn_calib_ood_after/pvals_from_existing_class_frac80.npy"
            all_z_ut_before.append(np.load(before_ut))
            all_z_ut_after.append(np.load(after_ut))
            all_z_vpn_before.append(np.load(before_vpn))
            all_z_vpn_after.append(np.load(after_vpn))
        all_ut_before.append(all_z_ut_before)
        all_ut_after.append(all_z_ut_after)
        all_vpn_before.append(all_z_vpn_before)
        all_vpn_after.append(all_z_vpn_after)
        run_number += 1
    return (np.array(all_ut_before).squeeze(), np.array(all_ut_after).squeeze(),
            np.array(all_vpn_before).squeeze(), np.array(all_vpn_after).squeeze())

def prepare_df(before, after):
    # Flatten data for plotting
    def prepare_dataframe(scores, label):
        scores = np.asarray(torch.tensor(scores).unsqueeze(1))
        runs, z_dims, samples = scores.shape
        data = []
        for run in range(runs):
            for z_dim in range(z_dims):
                for sample in range(samples):
                    data.append([run, z_dim, scores[run, z_dim, sample], label])
        return pd.DataFrame(data, columns=["Run", "Z_dim", "OOD_score", "Condition"])

    # Prepare data
    df_before = prepare_dataframe(before, "Before Retrain")
    df_after = prepare_dataframe(after, "After Retrain")

    # Combine into a single DataFrame
    return df_before, df_after

def average_all_ood(df, num_runs):
    average_ood_scores = []
    print(len(df["OOD_score"])/num_runs)
    for i in range(int(len(df["OOD_score"])/num_runs)):
        new_ood_score = 0
        for j in range(num_runs):
            new_ood_score += df[df["Run"]==j]["OOD_score"].iloc[i]
        average_ood_scores.append(new_ood_score/num_runs)
    shape_to_expand = len(average_ood_scores)
    average_ood_scores = np.array(torch.tensor(average_ood_scores).expand(num_runs, shape_to_expand).reshape(shape_to_expand*num_runs))
    return average_ood_scores



if __name__ == '__main__':
    main()