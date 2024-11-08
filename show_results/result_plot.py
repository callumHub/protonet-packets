import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
frac_values = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
#frac_values = [0.7]

def main():

    #f1_data = np.load("../runs/eval_outs/fraction_nh_zdim_64/test_micro_f1s.npy")
    #micro_f1_against_data_frac(f1_data)


    #ece_data = np.load("../runs/eval_outs/fraction_nh_zdim_64/test_calibers.npy")
    #ece_against_data_frac(ece_data)
    confusion_data = np.load("../runs/eval_outs/fraction_nh_zdim_64/test_confusions.npy")
    show_confusion_matrix(confusion_data, 0.2)



def micro_f1_against_data_frac(micro_f1_scores):
    general_plot(micro_f1_scores, "Micro F1 Scores with layers: 128x64 (no hidden) (10 trials)")

def ece_against_data_frac(ece_scores):
    general_plot(ece_scores, "ECE Scores with layers: 128x64 (no hidden) (10 trials)")

def show_confusion_matrix(confusion_matrix, data_frac=None):
    confusion_matrix = confusion_matrix.mean(axis=0)
    if data_frac is None:
        for i in range(len(frac_values)):
            sns.heatmap(confusion_matrix)
            plt.show()
    else:
        index = int((0.8-data_frac)*10)
        sns.heatmap(confusion_matrix[index])
        plt.show()

def general_plot(data, title):
    data = np.asarray(data).T
    means = np.mean(data, axis=1)
    std_errors = np.std(data, axis=1, ddof=1) / np.sqrt(data.shape[1])

    # Plot
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=frac_values, y=means, marker='o', label="Mean of test data")
    plt.errorbar(frac_values, means, yerr=std_errors, fmt='o', capsize=5, label="Std Error")

    # Customize the plot
    plt.xlabel("Data Fraction")
    plt.ylabel(title.split(" ")[0])
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()