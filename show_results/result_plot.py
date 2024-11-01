import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
frac_values = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
#frac_values = [0.7]
def micro_f1_against_data_frac(micro_f1_scores):
    general_plot(micro_f1_scores, "Micro F1 Scores")

def ece_against_data_frac(ece_scores):
    general_plot(ece_scores, "ECE Scores")

def general_plot(data, title):
    data = np.asarray(data)
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