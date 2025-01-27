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
#frac_values = [0.8, 0.7, 0.6, 0.5]
frac_values = [0.8]
def main():
    ut_f1 = "../runs/eval_outs/ut_mobile_kfold_decreasing_fraction_64x64x64x64/test_micro_f1s.npy"
    ut_ece = "../runs/eval_outs/ut_mobile_kfold_decreasing_fraction_64x64x64x64/test_calibers.npy"
    ut_confusion = "../runs/eval_outs/ut_mobile_kfold_decreasing_fraction_64x64x64x64/test_confusions.npy"
    f1_data = np.load("../runs/eval_outs/forest_pnet/forest_pnet_micros.npy")
    micro_f1_against_data_frac(f1_data)


    ece_data = np.load("../runs/eval_outs/forest_pnet/forest_pnet_calibs.npy")
    ece_against_data_frac(ece_data)

    confusion_data = np.load("../runs/eval_outs/forest_pnet/forest_pnet_confusions.npy")
    show_confusion_matrix(confusion_data, 0.8)



def micro_f1_against_data_frac(micro_f1_scores):
    general_plot(micro_f1_scores, "Micro F1 Scores: (10 Trial Average)")

def ece_against_data_frac(ece_scores):
    general_plot(ece_scores, "ECE Scores: (10 Trial Average)")

def show_confusion_matrix(confusion_matrix, data_frac=None):
    confusion_matrix = confusion_matrix.mean(axis=0)
    class_map = {0: "C2", 1: "CHAT", 2 : "FT", 3: "STREAM", 4: "VOIP"}
    if len(confusion_matrix[0]) > 5:
        class_map = {
            "FACEBOOK": 0, "TWITTER": 1, "REDDIT": 2, "INSTAGRAM": 3, "PINTREST": 4,
            "YOUTUBE": 5, "NETFLIX": 6, "HULU": 7, "SPOTIFY": 8, "PANDORA": 9,
            "DRIVE": 11, "DROPBOX": 12, "GMAIL": 13, "MESSENGER": 14
        } # NOTE HANGOUT AND MAPS HAD NO EXAMPLES
    if data_frac is None:
        for i in range(len(frac_values)):
            sns.heatmap(confusion_matrix)
            plt.show()
    else:
        index = int((0.8-data_frac)*10)
        plot_df = pd.DataFrame(data=confusion_matrix[index]/confusion_matrix[index,0,:].sum(), columns=class_map.values(), index=class_map.values())
        sns.heatmap(plot_df, annot=True, fmt=".0%", yticklabels=class_map.values(), xticklabels=True)
        plt.show()

def general_plot(data, title):
    axis = 0
    data = np.asarray(data).T

    means = np.mean(data, axis=axis)
    std_errors = np.std(data, axis=axis, ddof=1) / np.sqrt(data.shape[axis])

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=frac_values, y=means, marker='o', label="Mean of test data")
    plt.errorbar(frac_values, means, yerr=std_errors, fmt='o', capsize=5, label="Std Error")

    plt.xlabel("Data Fraction")
    plt.ylabel(title.split(" ")[0])
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()