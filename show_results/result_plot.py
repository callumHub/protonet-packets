#result_plot.py
"""
Using stored performance metrics, visualizes perf,
currently visualize means across 10 runs
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#frac_values = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2] # length and values hardcoded and very coupled to below.
frac_values = [0.8, 0.7, 0.6, 0.5]
frac_values = [0.8]
#frac_values = [64,56,48,40,36,24,16,8] #  z dim values
def main():

    forest_path = "../runs/eval_outs/forest_pnet/test_confusions.npy"
    ut_f1 = "../runs/eval_outs/ut_mobile_kfold_decreasing_fraction_64x64x64x64/test_micro_f1s.npy"
    ut_ece = "../runs/eval_outs/ut_mobile_kfold_decreasing_fraction_64x64x64x64/test_calibers.npy"
    ut_confusion = "../runs/eval_outs/ut_mobile_kfold_decreasing_fraction_64x64x64x64/test_confusions.npy"

    run_type = "before_retrain"
    run_type = "retrained"


    f1_data = np.load(f"../runs/eval_outs/{run_type}_utvpn_output/pnet_accs.npy")
    #micro_f1_against_data_frac(f1_data)
    #confusion_data = np.load(forest_path)
    #show_confusion_matrix(confusion_data)


    ece_data = np.load(f"../runs/eval_outs/{run_type}_utvpn_output/pnet_calibs.npy")
    #ece_against_data_frac(ece_data)
    print("Calibration Error", ece_data)
    print("Micro F1", f1_data)

    confusion_data = np.load(f"../runs/eval_outs/{run_type}_utvpn_output/pnet_confusions.npy")
    show_confusion_matrix(confusion_data.squeeze())


def z_dim_reduction():
    data_template = "../runs/eval_outs/decreasing_z_dim_outs/z{}/run{}/"

    total_test_mf, total_test_acc, total_test_ece, total_test_confusion, total_train_acc, total_train_loss = [], [], [], [], [], []
    for i in range(5):
        starting_z_dim = 64
        all_test_mf, all_test_acc, all_test_ece, all_test_confusion, all_train_acc, all_train_loss = [], [], [], [], [], []
        for j in range(8):
            all_test_mf.append(np.load(data_template.format(starting_z_dim, i)+"test_micro.npy"))
            all_test_acc.append(np.load(data_template.format(starting_z_dim, i)+"test_acc.npy"))
            all_test_ece.append(np.load(data_template.format(starting_z_dim, i)+"test_calib.npy"))
            all_test_confusion.append(np.load(data_template.format(starting_z_dim, i)+"confusion.npy"))
            all_train_acc.append(np.load(data_template.format(starting_z_dim, i)+"train_acc.npy"))
            all_train_loss.append(np.load(data_template.format(starting_z_dim, i)+"train_loss.npy"))
            starting_z_dim -= 8
        total_test_mf.append(all_test_mf)
        total_test_acc.append(all_test_acc)
        total_test_ece.append(all_test_ece)
        total_test_confusion.append(all_test_confusion)
        total_train_acc.append(all_train_acc)
        total_train_loss.append(all_train_loss)

    micro_f1_against_data_frac(total_test_mf, "Test Micro Decreasing Z Dim")
    ece_against_data_frac(total_test_ece, "Test Calibration Error Decreasing Z Dim")
    micro_f1_against_data_frac(total_train_acc, "Train Accuracy Decreasing Z Dim")


def micro_f1_against_data_frac(micro_f1_scores, title=None):
    general_plot(micro_f1_scores, title)

def ece_against_data_frac(ece_scores, title=None):
    general_plot(ece_scores, title)

def show_confusion_matrix(confusion_matrix, data_frac=None):
    #confusion_matrix = confusion_matrix.mean(axis=0)
    class_map = {0: "C2", 1: "CHAT", 2 : "FT", 3: "STREAM", 4: "VOIP"}
    if len(confusion_matrix[0]) > 8:
        class_map = {
            "FACEBOOK": 0, "TWITTER": 1, "REDDIT": 2, "INSTAGRAM": 3, "PINTREST": 4,
            "YOUTUBE": 5, "NETFLIX": 6, "HULU": 7, "SPOTIFY": 8, "PANDORA": 9,
            "DRIVE": 11, "DROPBOX": 12, "GMAIL": 13, "MESSENGER": 14
        } # NOTE HANGOUT AND MAPS HAD NO EXAMPLES
    elif len(confusion_matrix[0]) == 7:
        class_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
    elif len(confusion_matrix[0]) == 6:
        class_map = {0: "C2", 1: "CHAT", 2: "FT", 3: "STREAM", 4: "VOIP", 5: "SPOTIFY"}

    if data_frac is None:
        for i in range(len(frac_values)):
            #ax = sns.heatmap(confusion_matrix, )
            row_sums = confusion_matrix.sum(axis=1, keepdims=True)
            normalized_cm = confusion_matrix / row_sums

            # Convert to DataFrame for seaborn
            plot_df = pd.DataFrame(data=normalized_cm, columns=class_map.values(), index=class_map.values())

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(plot_df, annot=True, fmt=".0%", cmap="magma", yticklabels=class_map.values(),
                        xticklabels=class_map.values(), ax=ax)
            class_totals = confusion_matrix.sum(axis=1)
            total_samples = class_totals.sum()
            class_proportions = {class_map[i]: (class_totals[i] / total_samples) * 100 for i in
                                 range(len(class_totals))}

            # Create a custom legend for class proportions
            legend_texts = [f"Class {label}: {prop:.1f}%" for label, prop in class_proportions.items()]
            legend = "\n".join(legend_texts)

            # Add text as an annotation outside the plot
            plt.gcf().text(1.05, 0.5, legend, fontsize=12, verticalalignment='center')

            plt.title("Confusion Matrix with Class Proportions")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.show()
    else:
        index = int((0.8-data_frac)*10)
        plot_df = pd.DataFrame(data=confusion_matrix[index]/confusion_matrix[index,0,:].sum(), columns=class_map.values(), index=class_map.values())
        ax = sns.heatmap(plot_df, annot=True, fmt=".0%", yticklabels=class_map.values(), xticklabels=True)
        cbar = ax.collections[0].colorbar
        cbar.set_label("Class Balance", rotation=270, labelpad=15)
        plt.show()

def general_plot(data, title):
    axis = 0
    data = np.asarray(data)
    # TODO: replace y=data with y=mean when k fold running
    #means = np.mean(data, axis=axis)
    #std_errors = np.std(data, axis=axis, ddof=1) / np.sqrt(data.shape[axis])

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=frac_values, y=data, marker='o', label="Mean of test data")
    #plt.errorbar(frac_values, data, yerr=std_errors, fmt='o', capsize=5, label="Std Error")

    plt.xlabel("z_dim")
    plt.ylabel(title.split(" ")[0])
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()