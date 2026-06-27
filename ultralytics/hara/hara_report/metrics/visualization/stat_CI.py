import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. Input data
# =========================

mota_original = np.array([0.8907, 0.9355, 0.9180, 0.7891, 0.8503, 0.7967])
mota_optimized = np.array([0.9279, 0.9541, 0.9322, 0.8546, 0.9005, 0.8317])

idf1_original = np.array([0.6102, 0.6896, 0.7843, 0.5453, 0.5960, 0.5389])
idf1_optimized = np.array([0.9333, 0.9770, 0.9410, 0.7348, 0.8120, 0.7071])

ids_original = np.array([35, 17, 14, 95, 52, 57])
ids_optimized = np.array([2, 0, 2, 30, 9, 26])


# =========================
# 2. Bootstrap helper
# =========================

def bootstrap_means(values, n_boot=10000, seed=42):
    """
    Bootstrap the mean of a metric.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)

    boot = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot[i] = np.mean(sample)

    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    return boot, np.mean(values), ci_low, ci_high


# =========================
# 3. Plot helper
# =========================

def plot_bootstrap_hist_two_groups(
        metric_name,
        original,
        optimized,
        x_label,
        title,
        output_path=None,
        bins=35,
        n_boot=10000,
        seed=42
):
    """
    Draw one figure with two bootstrap frequency histograms:
    Original vs Optimized
    """
    boot_orig, mean_orig, ci_orig_low, ci_orig_high = bootstrap_means(
        original, n_boot=n_boot, seed=seed
    )
    boot_opt, mean_opt, ci_opt_low, ci_opt_high = bootstrap_means(
        optimized, n_boot=n_boot, seed=seed + 1
    )

    all_boot = np.concatenate([boot_orig, boot_opt])
    bin_edges = np.linspace(all_boot.min(), all_boot.max(), bins)

    plt.figure(figsize=(10, 6))

    # Frequency histograms
    plt.hist(boot_orig, bins=bin_edges, alpha=0.55, label="Original")
    plt.hist(boot_opt, bins=bin_edges, alpha=0.55, label="Optimized")

    # Mean lines
    plt.axvline(mean_orig, linestyle="--", linewidth=2,
                label=f"Original mean = {mean_orig:.4f}")
    plt.axvline(mean_opt, linestyle="--", linewidth=2,
                label=f"Optimized mean = {mean_opt:.4f}")

    # CI lines for original
    plt.axvline(ci_orig_low, linestyle=":", linewidth=2)
    plt.axvline(ci_orig_high, linestyle=":", linewidth=2)

    # CI lines for optimized
    plt.axvline(ci_opt_low, linestyle=":", linewidth=2)
    plt.axvline(ci_opt_high, linestyle=":", linewidth=2)

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.legend()

    # Text box
    textstr = (
        f"Original 95% CI: [{ci_orig_low:.4f}, {ci_orig_high:.4f}]\n"
        f"Optimized 95% CI: [{ci_opt_low:.4f}, {ci_opt_high:.4f}]"
    )

    plt.text(
        0.02, 0.98, textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.2)
    )

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


# =========================
# 4. Draw MOTA / IDF1 / IDS
# =========================

plot_bootstrap_hist_two_groups(
    metric_name="MOTA",
    original=mota_original,
    optimized=mota_optimized,
    x_label="Bootstrap mean MOTA",
    title="Bootstrap distribution of mean MOTA: Original vs Optimized",
    output_path="bootstrap_hist_mota.png",
    bins=35,
    n_boot=10000,
    seed=42
)

plot_bootstrap_hist_two_groups(
    metric_name="IDF1",
    original=idf1_original,
    optimized=idf1_optimized,
    x_label="Bootstrap mean IDF1",
    title="Bootstrap distribution of mean IDF1: Original vs Optimized",
    output_path="bootstrap_hist_idf1.png",
    bins=35,
    n_boot=10000,
    seed=42
)

plot_bootstrap_hist_two_groups(
    metric_name="IDS",
    original=ids_original,
    optimized=ids_optimized,
    x_label="Bootstrap mean IDS",
    title="Bootstrap distribution of mean IDS: Original vs Optimized",
    output_path="bootstrap_hist_ids.png",
    bins=35,
    n_boot=10000,
    seed=42
)