import numpy as np
import matplotlib.pyplot as plt
def main():
    frac = 30
    run_type = "ut_mobile_kfold_decreasing_fraction_64x64x64x64"
    #run_type = "baseline_pvals"
    pvals = np.load(f"../runs/eval_outs/{run_type}/pvals_og_to_ut_frac{frac}.npy")
    mean_pvals = np.load(f"../runs/eval_outs/{run_type}/total_pval_og_to_ut.npy")
    pvals = pvals.flatten().tolist()
    print(f"min pval: {min(pvals)}, max pval: {max(pvals)}, mean pval: {np.mean(pvals)},"
          f"\n# of ood examples: {list[int](np.greater_equal(pvals, 0.95)).count(1)}"
          f"\nTotal Pvals: {len(pvals)}")
    pvals_1 = pvals
    run_type = "baseline_pvals"
    pvals_2 = np.load(f"../runs/eval_outs/{run_type}/pvals_og_to_ut_frac{frac}.npy")[0]

    plot_pvals(pvals_1, pvals_2)


def plot_pvals(pvals_1, pvals_2):
    # Indices for bar positions
    indices = np.arange(len(pvals_1))
    indices_2 = np.arange(len(pvals_2))

    threshold = 0.95
    # Count how many p-values are above the threshold
    above_threshold_1 = sum(p > threshold for p in pvals_1)
    above_threshold_2 = sum(p > threshold for p in pvals_2)

    # Create x-axis categories
    categories = ['UTMOBILE (OOD)', 'Jorgensen']
    data = [pvals_1, pvals_2]

    # Generate jitter for better visibility of overlapping points
    jitter_1 = np.random.uniform(-0.1, 0.1, len(pvals_1))
    jitter_2 = np.random.uniform(-0.1, 0.1, len(pvals_2))

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(np.ones(len(pvals_1)) + jitter_1, pvals_1, label='UTMOBILE (OOD)', color='blue', s=100, alpha=0.7)
    plt.scatter(2 * np.ones(len(pvals_2)) + jitter_2, pvals_2, label='Jorgensen', color='orange', s=100, alpha=0.7)

    # Add horizontal significance threshold line
    plt.axhline(y=0.95, color='red', linestyle='--', label='Significance Threshold (0.95)')

    plt.text(1, max(max(pvals_1), threshold) + 0.05, f'Above: {above_threshold_1}, Total: {len(pvals_1)}',
             horizontalalignment='center', color='blue', fontsize=12)
    plt.text(2, max(max(pvals_2), threshold) + 0.05, f'Above: {above_threshold_2}, Total: {len(pvals_2)}',
             horizontalalignment='center', color='orange', fontsize=12)

    # Formatting
    plt.xticks([1, 2], categories)  # Set x-axis ticks to categories
    plt.xlabel('Datasets')
    plt.ylabel('OOD Score')
    plt.title('Comparison of OOD Scores')
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()