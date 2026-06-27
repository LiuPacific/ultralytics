import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest


# =========================
# 1. Input data
# =========================

data = {
    "MOTA": {
        "original": np.array([0.8907, 0.9355, 0.9180, 0.7891, 0.8503, 0.7967]),
        "optimized": np.array([0.9279, 0.9541, 0.9322, 0.8546, 0.9005, 0.8317]),
        "higher_is_better": True,
    },
    "IDF1": {
        "original": np.array([0.6102, 0.6896, 0.7843, 0.5453, 0.5960, 0.5389]),
        "optimized": np.array([0.9333, 0.9770, 0.9410, 0.7348, 0.8120, 0.7071]),
        "higher_is_better": True,
    },
    "IDS": {
        "original": np.array([35, 17, 14, 95, 52, 57]),
        "optimized": np.array([2, 0, 2, 30, 9, 26]),
        "higher_is_better": False,
    },
}


# =========================
# 2. Statistics function
# =========================

def calculate_paired_statistics(original, optimized, higher_is_better=True,
                                n_boot=100000, seed=42):
    original = np.asarray(original, dtype=float)
    optimized = np.asarray(optimized, dtype=float)

    # Define improvement direction
    if higher_is_better:
        diff = optimized - original
    else:
        diff = original - optimized

    n = len(diff)

    # Mean and SD of paired differences
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)

    # Paired t-test, one-sided: mean improvement > 0
    t_stat = mean_diff / (sd_diff / np.sqrt(n))
    p_ttest = stats.t.sf(t_stat, df=n - 1)

    # Wilcoxon signed-rank test, one-sided
    wilcoxon_result = stats.wilcoxon(
        diff,
        alternative="greater",
        method="exact"
    )
    p_wilcoxon = wilcoxon_result.pvalue

    # Sign test, one-sided
    num_improved = np.sum(diff > 0)
    sign_result = binomtest(
        k=int(num_improved),
        n=n,
        p=0.5,
        alternative="greater"
    )
    p_sign = sign_result.pvalue

    # 95% t confidence interval
    se = sd_diff / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se

    # Bootstrap 95% confidence interval
    rng = np.random.default_rng(seed)
    boot_means = []

    for _ in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        boot_means.append(np.mean(sample))

    boot_low, boot_high = np.percentile(boot_means, [2.5, 97.5])

    # Cohen's dz for paired design
    cohens_dz = mean_diff / sd_diff

    return {
        "differences": diff,
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "paired_t_p": p_ttest,
        "wilcoxon_p": p_wilcoxon,
        "num_improved": int(num_improved),
        "sign_p": p_sign,
        "t_ci_low": ci_low,
        "t_ci_high": ci_high,
        "bootstrap_ci_low": boot_low,
        "bootstrap_ci_high": boot_high,
        "cohens_dz": cohens_dz,
    }


# =========================
# 3. Format results for PPT table
# =========================

def make_ppt_table(metric_name, stats_result):
    if metric_name == "IDS":
        mean_label = "Mean reduction"
        sd_label = "SD reduction"

        mean_value = f"{stats_result['mean_diff']:.1f} switches"
        sd_value = f"{stats_result['sd_diff']:.1f}"
        t_ci_value = f"[{stats_result['t_ci_low']:.2f}, {stats_result['t_ci_high']:.2f}]"
        boot_ci_value = f"[{stats_result['bootstrap_ci_low']:.2f}, {stats_result['bootstrap_ci_high']:.2f}]"

    else:
        mean_label = "Mean diff"
        sd_label = "SD diff"

        mean_value = f"+{stats_result['mean_diff']:.4f}"
        sd_value = f"{stats_result['sd_diff']:.4f}"
        t_ci_value = f"[{stats_result['t_ci_low']:.4f}, {stats_result['t_ci_high']:.4f}]"
        boot_ci_value = f"[{stats_result['bootstrap_ci_low']:.4f}, {stats_result['bootstrap_ci_high']:.4f}]"

    table = pd.DataFrame({
        "Statistic": [
            mean_label,
            sd_label,
            "Paired t-test",
            "Wilcoxon",
            "Sign test",
            "95% t CI",
            "Bootstrap 95% CI",
            "Cohen's dz",
        ],
        "Result": [
            mean_value,
            sd_value,
            f"p = {stats_result['paired_t_p']:.4g}",
            f"p = {stats_result['wilcoxon_p']:.4f}",
            f"{stats_result['num_improved']} / 6 improved; p = {stats_result['sign_p']:.4f}",
            t_ci_value,
            boot_ci_value,
            f"{stats_result['cohens_dz']:.2f}",
        ]
    })

    return table


# =========================
# 4. Run calculation
# =========================

for metric_name, values in data.items():
    result = calculate_paired_statistics(
        original=values["original"],
        optimized=values["optimized"],
        higher_is_better=values["higher_is_better"],
        n_boot=100000,
        seed=42
    )

    print("\n" + "=" * 50)
    print(metric_name)
    print("=" * 50)

    print("Fold-level improvement values:")
    print(result["differences"])

    ppt_table = make_ppt_table(metric_name, result)
    print("\nPPT table:")
    print(ppt_table.to_string(index=False))