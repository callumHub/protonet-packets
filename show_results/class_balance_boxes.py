import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    class_map = {0: "C2", 1: "CHAT", 2: "FT", 3: "STREAM", 4: "VOIP", 5: "SPOTIFY"}
    confusion_before, confusion_after = read_all_data("pnet_confusions.npy")
    # confusion data has the shape: (n_runs, len(z_dims), n_class (actual), n_class (predicted))
    # Need to get how the prediction accuracy changes when new class is added, and after training on that new class
    # taking diag on the last 2 axis gives me shape (n_runs, len(z_dims), n_class).

    # TODO: Make a fancy df that can do what is shown on https://pythonbasics.org/seaborn-barplot/
    confusion_before = confusion_before.diagonal(axis1=-2, axis2=-1)
    divide_by_total = np.vectorize(lambda x: x/43)
    confusion_before = divide_by_total(confusion_before)
    confusion_before_means = confusion_before.mean(axis=0)
    confusion_before_stds = confusion_before.std(axis=0)


    confusion_after = confusion_after.diagonal(axis1=-2, axis2=-1)
    confusion_after = divide_by_total(confusion_after)
    confusion_after_means = confusion_after.mean(axis=0)


    make_bar_plot(confusion_before_means)



    a = 1+3


def read_all_data(data_file_name):
    """
    Goes through the directory storing my decreasing z dim runs
    :data_file_name: Which metric to obtain
    :return: metric specified by file name before and after training nn.
    """
    file = data_file_name
    z_dims = [64, 56, 48, 40, 23, 24, 16, 8, 4, 2, 1]
    z_dims.reverse()
    all_before_confusions, all_after_confusions = [], []
    for i in range(10):
        all_z_before_confusions, all_z_after_confusions = [], []
        for j in range(len(z_dims)):
            path_template = f"../runs/eval_outs/multiple_z_dim_p_val/z_{z_dims[j]}/run_{i}/"
            before = path_template + "before_retrain_utvpn_output/"
            after = path_template + "retrained_utvpn_output/"
            all_z_before_confusions.append(np.load(before + file))
            all_z_after_confusions.append(np.load(after + file))
        all_before_confusions.append(all_z_before_confusions)
        all_after_confusions.append(all_z_after_confusions)
    return np.array(all_before_confusions).squeeze(), np.array(all_after_confusions).squeeze()

def make_confusion_dataframe(confusions):
    df = pd.DataFrame(columns=["dim", "class", "avg_accuracy"])
    z_dims = [64, 56, 48, 40, 32, 24, 16, 8, 4, 2, 1]

def make_bar_plot(confusions):
    # shape: (11, 6):
    bar_width = 1/7
    fig = plt.subplots(figsize=(12, 4))
    z_dims = [64, 56, 48, 40, 32, 24, 16, 8, 4, 2, 1]
    z_dims.reverse()
    br1 = np.arange(len(z_dims))
    palette = sns.color_palette("deep")
    confusions = np.flipud(confusions)
    z_dims = np.flipud(z_dims)
    for i in range(6):
        c = palette.pop()
        plt.bar(br1+i*bar_width, confusions[:, i], width=bar_width, label=f"Class {i}", color=c)


    plt.title("Decreasing Z Dim Accuracy (Before Training on Class 5)")
    plt.xlabel('Z DIM')
    plt.ylabel("Detection Rate (10 Run Avg)")
    plt.xticks([r + 2.5*bar_width for r in range(len(br1))],
               z_dims, fontweight="bold")
    #plt.yticks([0.75, 0.8, 0.85, 0.9, 0.95, 1])
    plt.ylim(0.72,1)
    plt.legend(loc="lower right")
    plt.show()


















if __name__ == '__main__':
    main()
