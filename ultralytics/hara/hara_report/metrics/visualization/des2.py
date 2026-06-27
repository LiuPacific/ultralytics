import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import binomtest
from pathlib import Path


# =========================
# 1. Data
# =========================

folds = np.arange(1, 7)

mota_original = np.array([0.8907, 0.9355, 0.9180, 0.7891, 0.8503, 0.7967])
mota_optimized = np.array([0.9279, 0.9541, 0.9322, 0.8546, 0.9005, 0.8317])

idf1_original = np.array([0.6102, 0.6896, 0.7843, 0.5453, 0.5960, 0.5389])
idf1_optimized = np.array([0.9333, 0.9770, 0.9410, 0.7348, 0.8120, 0.7071])

ids_original = np.array([35, 17, 14, 95, 52, 57])
ids_optimized = np.array([2, 0, 2, 30, 9, 26])


# =========================
# 2. Statistics helper
# =========================

def compute_paired_stats(original, optimized, higher_is_better=True, n_boot=10000, seed=42):
    """
    For MOTA/IDF1:
        improvement = optimized - original

    For IDS:
        lower is better, so:
        improvement = original - optimized
    """
    original = np.asarray(original, dtype=float)
    optimized = np.asarray(optimized, dtype=float)

    if higher_is_better:
        diff = optimized - original
    else:
        diff = original - optimized

    n = len(diff)
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)

    # One-sided paired t-test: improvement > 0
    t_stat = mean_diff / (sd_diff / np.sqrt(n))
    p_t = stats.t.sf(t_stat, df=n - 1)

    # Wilcoxon signed-rank test, one-sided
    wilcoxon_result = stats.wilcoxon(diff, alternative="greater")
    p_wilcoxon = wilcoxon_result.pvalue

    # Sign test, one-sided
    num_improved = np.sum(diff > 0)
    p_sign = binomtest(num_improved, n=n, p=0.5, alternative="greater").pvalue

    # 95% t confidence interval
    se = sd_diff / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se

    # Bootstrap 95% CI for mean improvement
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_low, boot_high = np.percentile(boot_means, [2.5, 97.5])

    # Cohen's dz for paired design
    dz = mean_diff / sd_diff

    return {
        "diff": diff,
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "p_t": p_t,
        "p_wilcoxon": p_wilcoxon,
        "num_improved": int(num_improved),
        "p_sign": p_sign,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "boot_low": boot_low,
        "boot_high": boot_high,
        "dz": dz,
    }


def format_p(p):
    if p < 0.001:
        return f"{p:.5f}"
    return f"{p:.4f}"


# =========================
# 3. Plot helper
# =========================

def draw_bar_result(
        metric_name,
        original,
        optimized,
        ylabel,
        title,
        subtitle,
        conclusion,
        higher_is_better=True,
        output_path="output.png",
):
    stat = compute_paired_stats(original, optimized, higher_is_better=higher_is_better)
    diff = stat["diff"]

    fig = plt.figure(figsize=(15, 7))

    # Main bar chart axis
    ax = fig.add_axes([0.07, 0.16, 0.50, 0.67])

    width = 0.36
    x = np.arange(len(folds))

    bars1 = ax.bar(x - width / 2, original, width, label="Original")
    bars2 = ax.bar(x + width / 2, optimized, width, label="Optimized")

    ax.set_title(f"{metric_name}: bar-chart fold comparison", fontsize=17, fontweight="bold")
    ax.set_xlabel("Held-out video fold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in folds])
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    # Y-axis range
    all_values = np.concatenate([original, optimized])
    y_min = max(0, np.min(all_values) - 0.08 * (np.max(all_values) - np.min(all_values)))
    y_max = np.max(all_values) + 0.18 * (np.max(all_values) - np.min(all_values))

    if metric_name in ["MOTA", "IDF1"]:
        y_min = max(0, y_min)
        y_max = min(1.05, y_max)

    ax.set_ylim(y_min, y_max)

    # Value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if metric_name in ["MOTA", "IDF1"]:
                label = f"{height:.3f}"
            else:
                label = f"{int(height)}"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                )

    # Improvement labels
    offset = 0.04 * (y_max - y_min)
    for i in range(len(folds)):
        top = max(original[i], optimized[i])
        if higher_is_better:
            label = f"+{diff[i]:.4f}"
        else:
            label = f"-{diff[i]:.0f}"

        ax.text(
            x[i],
            top + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            )

    # Page title and subtitle

    # Statistics table axis
    ax_table = fig.add_axes([0.62, 0.25, 0.34, 0.52])
    ax_table.axis("off")

    if higher_is_better:
        mean_name = "Mean diff"
        sd_name = "SD diff"
    else:
        mean_name = "Mean reduction"
        sd_name = "SD reduction"

    if metric_name in ["MOTA", "IDF1"]:
        value_fmt = lambda v: f"{v:.4f}"
        ci_fmt = lambda a, b: f"[{a:.4f}, {b:.4f}]"
    else:
        value_fmt = lambda v: f"{v:.2f}"
        ci_fmt = lambda a, b: f"[{a:.2f}, {b:.2f}]"



    output_path = Path(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {output_path}")


# =========================
# 4. Draw three figures
# =========================

draw_bar_result(
    metric_name="MOTA",
    original=mota_original,
    optimized=mota_optimized,
    ylabel="MOTA",
    title="MOTA result: consistent fold-level improvement",
    subtitle="Optimized MOTA is higher than original MOTA in all 6 held-out videos.",
    conclusion="Conclusion: statistically supported improvement",
    higher_is_better=True,
    output_path="mota_bar_result.png",
)

draw_bar_result(
    metric_name="IDF1",
    original=idf1_original,
    optimized=idf1_optimized,
    ylabel="IDF1",
    title="IDF1 result: strongest improvement is identity consistency",
    subtitle="This metric directly reflects whether the same chicken keeps a stable identity.",
    conclusion="Conclusion: large identity-tracking gain",
    higher_is_better=True,
    output_path="idf1_bar_result.png",
)

draw_bar_result(
    metric_name="IDS",
    original=ids_original,
    optimized=ids_optimized,
    ylabel="ID switches",
    title="IDS result: optimization reduces ID switches",
    subtitle="Lower IDS is better, so the improvement variable is Original IDS − Optimized IDS.",
    conclusion="Conclusion: fewer identity failures",
    higher_is_better=False,
    output_path="ids_bar_result.png",
)