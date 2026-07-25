"""Hardcoded MCB analyses for run-14 MOTA, IDF1, IDS, and FPS.

Install dependencies:
    python -m pip install numpy scipy matplotlib

Run:
    python run14_mota_ids_mcb_hardcoded.py

Results are printed directly to the console. Four decision-limit figures are
saved in the current working directory. MOTA, IDF1, and FPS are analyzed as
larger-is-better; IDS is analyzed as smaller-is-better.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_t


ALPHA = 0.05
RANDOM_SEED = 20260720

# Values within every list are ordered as hold1, hold2, ..., hold6.
MOTA_DATA = {
    # "DeepSORT OBB": [0.8797814207650273, 0.9191256830601093, 0.8950819672131147, 0.8010928961748633, 0.8426229508196721, 0.7792349726775956],
    # "DeepSORT OBB+track optimization": [0.9202185792349726, 0.9453551912568307, 0.9202185792349726, 0.8404371584699454, 0.8819672131147541, 0.8218579234972678],
    # "DeepSORT OBB+detection optimization": [0.9016393442622951, 0.9398907103825137, 0.9027322404371585, 0.8120218579234972, 0.862295081967213, 0.7923497267759563],
    # "DeepSORT OBB+track optimization+detection optimization": [0.9202185792349726, 0.9475409836065574, 0.9224043715846995, 0.8404371584699454, 0.8852459016393442, 0.8229508196721311],
    # "DeepSORT OBB+ReID": [0.8907103825136612, 0.9311475409836065, 0.9092896174863387, 0.8262295081967213, 0.8568306010928962, 0.7923497267759563],
    # "DeepSORT HBB": [0.8918032786885246, 0.9420765027322404, 0.8841530054644808, 0.7770491803278688, 0.8524590163934427, 0.8218579234972678],
    # "DeepSORT HBB+ReID": [0.907103825136612, 0.9431693989071038, 0.8950819672131147, 0.8524590163934427, 0.862295081967213, 0.8262295081967213],
    "DeepSORT HBB+track optimization": [0.9256830601092896, 0.9486338797814208, 0.9180327868852459, 0.8765027322404372, 0.8852459016393442, 0.8612021857923498],
    # "DeepSORT HBB+detection optimization": [0.912568306010929, 0.9453551912568307, 0.8907103825136612, 0.7956284153005464, 0.8699453551912568, 0.839344262295082],
    "DeepSORT HBB+track optimization+detection optimization": [0.9289617486338798, 0.9508196721311475, 0.9202185792349726, 0.8797814207650273, 0.8896174863387978, 0.8655737704918033],
    # "BotSORT HBB": [0.9147540983606557, 0.9486338797814208, 0.9289617486338798, 0.8830601092896175, 0.8819672131147541, 0.8469945355191257],
    "BotSORT HBB+track optimization": [0.9256830601092896, 0.9508196721311475, 0.9333333333333333, 0.898360655737705, 0.8896174863387978, 0.862295081967213],
    "BotSORT HBB+detection optimization": [0.9256830601092896, 0.9508196721311475, 0.9344262295081968, 0.8918032786885246, 0.8939890710382514, 0.8601092896174863],
    "BotSORT HBB+track optimization+detection optimization": [0.926775956284153, 0.9508196721311475, 0.9344262295081968, 0.8972677595628415, 0.8939890710382514, 0.8633879781420765],
    # "ByteTrack HBB": [0.912568306010929, 0.9497267759562842, 0.926775956284153, 0.8775956284153006, 0.8852459016393442, 0.8448087431693989],
    "ByteTrack HBB+track optimization": [0.921311475409836, 0.9508196721311475, 0.9333333333333333, 0.8907103825136612, 0.8972677595628415, 0.8579234972677596],
    "ByteTrack HBB+detection optimization": [0.9256830601092896, 0.9497267759562842, 0.9344262295081968, 0.8918032786885246, 0.8972677595628415, 0.8579234972677596],
    "ByteTrack HBB+track optimization+detection optimization": [0.926775956284153, 0.9486338797814208, 0.9344262295081968, 0.8972677595628415, 0.9005464480874317, 0.8590163934426229],
}

IDS_DATA = {
    # "DeepSORT OBB": [45, 30, 36, 84, 56, 72],
    # "DeepSORT OBB+track optimization": [8, 6, 13, 48, 20, 33],
    # "DeepSORT OBB+detection optimization": [27, 13, 29, 75, 39, 60],
    # "DeepSORT OBB+track optimization+detection optimization": [10, 6, 11, 49, 18, 32],
    # "DeepSORT OBB+ReID": [35, 19, 23, 61, 43, 60],
    # "DeepSORT HBB": [33, 10, 33, 110, 45, 48],
    # "DeepSORT HBB+ReID": [19, 9, 23, 41, 36, 44],
    "DeepSORT HBB+track optimization": [2, 4, 2, 19, 15, 12],
    # "DeepSORT HBB+detection optimization": [17, 7, 29, 97, 32, 32],
    "DeepSORT HBB+track optimization+detection optimization": [2, 2, 2, 20, 14, 8],
    # "BotSORT HBB": [10, 4, 4, 21, 20, 23],
    "BotSORT HBB+track optimization": [0, 2, 0, 7, 13, 10],
    "BotSORT HBB+detection optimization": [3, 2, 0, 16, 12, 11],
    "BotSORT HBB+track optimization+detection optimization": [2, 2, 0, 11, 12, 8],
    # "ByteTrack HBB": [12, 3, 6, 26, 17, 23],
    "ByteTrack HBB+track optimization": [4, 2, 0, 14, 6, 12],
    "ByteTrack HBB+detection optimization": [3, 3, 0, 16, 9, 11],
    "ByteTrack HBB+track optimization+detection optimization": [2, 4, 0, 11, 6, 10],
}

IDF1_DATA = {
    # "DeepSORT OBB": [0.5920262151829602, 0.638646288209607, 0.6644808743169399, 0.5733333333333334, 0.5415754923413567, 0.4977827050997783],
    # "DeepSORT OBB+track optimization": [0.8727471327143638, 0.9115720524017468, 0.8655737704918033, 0.6933333333333334, 0.824945295404814, 0.6995565410199557],
    # "DeepSORT OBB+detection optimization": [0.6670311645708037, 0.7759562841530054, 0.7213114754098361, 0.5814341300722624, 0.6611932129173509, 0.5399113082039911],
    # "DeepSORT OBB+track optimization+detection optimization": [0.8321487151448879, 0.912568306010929, 0.8775956284153006, 0.631461923290717, 0.8199233716475096, 0.6995565410199557],
    # "DeepSORT OBB+ReID": [0.6684871654833424, 0.6910480349344978, 0.7573770491803279, 0.6033333333333334, 0.6094091903719913, 0.5277161862527716],
    # "DeepSORT HBB": [0.6331877729257642, 0.7676325861126299, 0.6479338842975206, 0.44617092119866814, 0.6986899563318777, 0.6538249862410567],
    # "DeepSORT HBB+ReID": [0.732532751091703, 0.8059048660470203, 0.7261707988980717, 0.58157602663707, 0.7117903930131004, 0.662630709961475],
    "DeepSORT HBB+track optimization": [0.9213973799126638, 0.8813559322033898, 0.9586776859504132, 0.7602663706992231, 0.7794759825327511, 0.8860759493670886],
    # "DeepSORT HBB+detection optimization": [0.7643521049753964, 0.8212137780207763, 0.7203530060672918, 0.492769744160178, 0.735632183908046, 0.6898071625344353],
    "DeepSORT HBB+track optimization+detection optimization": [0.9229086932750137, 0.9436850738108256, 0.9597352454495311, 0.7931034482758621, 0.8144499178981938, 0.9234159779614325],
    # "BotSORT HBB": [0.8438864628820961, 0.9294696555494806, 0.9086932750136687, 0.7480662983425415, 0.7797164667393675, 0.7308750687947165],
    "BotSORT HBB+track optimization": [0.962882096069869, 0.9480590486604702, 0.9666484417714598, 0.8519337016574585, 0.8549618320610687, 0.7687224669603524],
    "BotSORT HBB+detection optimization": [0.8999453253143794, 0.9480590486604702, 0.9671772428884027, 0.7935805201992252, 0.8343357025697102, 0.8165289256198347],
    "BotSORT HBB+track optimization+detection optimization": [0.9218151995626025, 0.9480590486604702, 0.9671772428884027, 0.8046485888212507, 0.8365226899945325, 0.8055096418732782],
    # "ByteTrack HBB": [0.7631004366812227, 0.9458720612356479, 0.8769819573537452, 0.712707182320442, 0.787350054525627, 0.7815079801871216],
    "ByteTrack HBB+track optimization": [0.9192139737991266, 0.9480590486604702, 0.9666484417714598, 0.7756906077348066, 0.8593238822246456, 0.8337004405286343],
    "ByteTrack HBB+detection optimization": [0.9032258064516129, 0.931656642974303, 0.9671772428884027, 0.7902600996126176, 0.839803171131766, 0.8297520661157025],
    "ByteTrack HBB+track optimization+detection optimization": [0.9218151995626025, 0.9294696555494806, 0.9671772428884027, 0.801328168234643, 0.861673045379989, 0.8859504132231405],
}

# FPS observations come from the per-fold processing_fps tables in
# ``experiment data run14 fold comparison tables.md``.
FPS_DATA = {
    # "DeepSORT OBB": [11.224, 11.503, 11.530, 11.660, 11.617, 11.656],
    # "DeepSORT OBB+track optimization": [11.901, 11.722, 11.803, 11.921, 12.012, 11.816],
    # "DeepSORT OBB+detection optimization": [11.610, 11.633, 11.707, 11.901, 11.630, 11.601],
    # "DeepSORT OBB+track optimization+detection optimization": [11.548, 11.488, 11.410, 11.204, 12.927, 13.776],
    # "DeepSORT OBB+ReID": [11.060, 10.853, 10.907, 11.004, 10.852, 10.896],
    # "DeepSORT HBB": [11.312, 11.706, 11.623, 11.873, 11.704, 11.736],
    # "DeepSORT HBB+ReID": [10.475, 10.297, 9.388, 9.517, 9.478, 9.402],
    "DeepSORT HBB+track optimization": [11.788, 11.769, 11.811, 11.752, 11.842, 11.710],
    # "DeepSORT HBB+detection optimization": [11.697, 11.696, 11.712, 11.600, 11.605, 11.620],
    "DeepSORT HBB+track optimization+detection optimization": [11.710, 11.662, 11.660, 11.312, 13.015, 13.682],
    # "BotSORT HBB": [12.441, 12.688, 12.754, 12.870, 12.768, 12.737],
    "BotSORT HBB+track optimization": [13.075, 13.060, 12.845, 13.167, 13.109, 17.890],
    "BotSORT HBB+detection optimization": [12.871, 12.786, 12.770, 13.067, 13.200, 12.762],
    "BotSORT HBB+track optimization+detection optimization": [13.331, 12.805, 12.690, 12.808, 12.846, 12.393],
    # "ByteTrack HBB": [12.428, 12.694, 12.748, 12.830, 12.787, 12.809],
    "ByteTrack HBB+track optimization": [13.101, 13.120, 12.870, 13.239, 13.151, 13.217],
    "ByteTrack HBB+detection optimization": [12.846, 12.763, 12.780, 13.090, 13.205, 12.799],
    "ByteTrack HBB+track optimization+detection optimization": [13.010, 12.771, 12.707, 12.956, 12.544, 13.883],
}


def dunnett_critical_value(number_of_comparisons, error_df, alpha):
    """Calculate the one-sided Dunnett critical value d_(alpha,k,df)."""

    # Balanced comparisons against one common control have correlation 1/2.
    correlation = np.full((number_of_comparisons, number_of_comparisons), 0.5)
    np.fill_diagonal(correlation, 1.0)
    distribution = multivariate_t(
        loc=np.zeros(number_of_comparisons),
        shape=correlation,
        df=error_df,
    )

    def objective(candidate):
        probability = distribution.cdf(
            np.full(number_of_comparisons, candidate),
            maxpts=2_000_000,
            random_state=np.random.default_rng(RANDOM_SEED),
        )
        return probability - (1.0 - alpha)

    return brentq(objective, 1.5, 4.0, xtol=1e-7)


def calculate_mcb(data, higher_is_better, critical_value):
    names = list(data)
    values = np.array(list(data.values()), dtype=float)
    treatment_count, block_count = values.shape
    means = values.mean(axis=1)
    standard_deviations = values.std(axis=1, ddof=1)

    # Randomized-complete-block error: each held-out video is a matched block.
    grand_mean = values.mean()
    residuals = (
        values
        - means[:, np.newaxis]
        - values.mean(axis=0)[np.newaxis, :]
        + grand_mean
    )
    sse = np.square(residuals).sum()
    error_df = (treatment_count - 1) * (block_count - 1)
    mse = sse / error_df
    decision_limit = critical_value * np.sqrt(2.0 * mse / block_count)

    if higher_is_better:
        # Larger-is-better: D_i = mean_i - max(other means).
        competing_best = np.array(
            [np.max(np.delete(means, index)) for index in range(treatment_count)]
        )
        contrasts = means - competing_best
        best_index = int(np.argmax(means))
        order = np.argsort(-means)
        deficits = means[best_index] - means
    else:
        # Smaller-is-better: D_i = min(other means) - mean_i.
        competing_best = np.array(
            [np.min(np.delete(means, index)) for index in range(treatment_count)]
        )
        contrasts = competing_best - means
        best_index = int(np.argmin(means))
        order = np.argsort(means)
        deficits = means - means[best_index]

    upper_limits = contrasts + decision_limit
    in_best_set = upper_limits >= 0.0

    return {
        "names": names,
        "values": values,
        "means": means,
        "standard_deviations": standard_deviations,
        "sse": sse,
        "error_df": error_df,
        "mse": mse,
        "critical_value": critical_value,
        "decision_limit": decision_limit,
        "contrasts": contrasts,
        "upper_limits": upper_limits,
        "in_best_set": in_best_set,
        "order": order,
        "best_index": best_index,
        "deficits": deficits,
        "higher_is_better": higher_is_better,
    }


def print_results(metric, result, decimals):
    names = result["names"]
    best_index = result["best_index"]
    direction = "larger is better" if result["higher_is_better"] else "smaller is better"

    print()
    print(f"{metric} MULTIPLE COMPARISONS WITH THE BEST (MCB)")
    print("=" * 112)
    print(f"Direction:                   {direction}")
    print("Model:                       randomized complete block design")
    print(f"Configurations:              {len(names)}")
    print(f"Blocks / folds:              {result['values'].shape[1]}")
    print(f"Error degrees of freedom:    {result['error_df']}")
    print(f"Block-model SSE:             {result['sse']:.6f}")
    print(f"Block-model MSE:             {result['mse']:.6f}")
    print(f"One-sided Dunnett critical:  {result['critical_value']:.4f}")
    print(f"95% decision limit M:        {result['decision_limit']:.{decimals}f}")
    print(f"Sample-best configuration:   {names[best_index]}")
    print(f"Sample-best mean {metric}:       {result['means'][best_index]:.{decimals}f}")
    print()

    header = (
        f"{'Rank':>4}  {'Configuration':<62} {'Mean':>9} {'SD':>9} "
        f"{'D_i':>10} {'D_i+M':>10} {'Best set':>9}"
    )
    print(header)
    print("-" * len(header))
    for rank, index in enumerate(result["order"], start=1):
        print(
            f"{rank:>4}  "
            f"{names[index]:<62} "
            f"{result['means'][index]:>9.{decimals}f} "
            f"{result['standard_deviations'][index]:>9.{decimals}f} "
            f"{result['contrasts'][index]:>+10.{decimals}f} "
            f"{result['upper_limits'][index]:>+10.{decimals}f} "
            f"{'Yes' if result['in_best_set'][index] else 'No':>9}"
        )

    best_set_size = int(result["in_best_set"].sum())
    print()
    print(f"95% MCB best set: {best_set_size} of {len(names)} configurations")
    for index in result["order"]:
        if result["in_best_set"][index]:
            print(f"  - {names[index]}")


def save_figure(metric, result, decimals):
    order = result["order"]
    ordered_names = [result["names"][index] for index in order]
    ordered_deficits = result["deficits"][order]
    ordered_membership = result["in_best_set"][order]
    decision_limit = result["decision_limit"]

    figure, axis = plt.subplots(figsize=(11, 8.5), dpi=180)
    colors = [
        "#2f6db3" if is_in_best_set else "#9aa6b2"
        for is_in_best_set in ordered_membership
    ]
    bars = axis.barh(
        range(len(ordered_names)), ordered_deficits, color=colors, edgecolor="none"
    )
    axis.axvspan(0, decision_limit, color="#dceaf7", alpha=0.75, zorder=0)
    axis.axvline(
        decision_limit,
        color="#183b63",
        linestyle="--",
        linewidth=1.5,
        label=f"95% decision limit M = {decision_limit:.{decimals}f}",
    )
    axis.set_yticks(range(len(ordered_names)), ordered_names)
    axis.invert_yaxis()
    axis.set_xlabel(f"Observed {metric} deficit from sample-best mean")
    axis.set_title(
        f"{metric} Multiple Comparisons with the Best\n"
        "One-sided Dunnett familywise confidence level: 95%"
    )
    axis.grid(axis="x", color="#d7dce1", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", frameon=False)

    label_offset = max(float(ordered_deficits.max()) * 0.012, decision_limit * 0.04)
    for bar, deficit in zip(bars, ordered_deficits):
        axis.text(
            deficit + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{deficit:.{decimals}f}",
            va="center",
            fontsize=8,
        )

    axis.set_xlim(
        0,
        max(ordered_deficits.max() * 1.15, decision_limit * 1.3),
    )
    figure.tight_layout()

    image_path = Path.cwd() / f"experiment-data-run14-{metric.lower()}-mcb-hardcoded.png"
    figure.savefig(image_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Figure saved to: {image_path}")


def validate_data():
    datasets = {
        "MOTA": MOTA_DATA,
        "IDF1": IDF1_DATA,
        "IDS": IDS_DATA,
        "FPS": FPS_DATA,
    }
    expected_names = list(MOTA_DATA)
    # if len(expected_names) != 18:
    #     raise ValueError("Expected exactly 18 configurations")
    for metric, data in datasets.items():
        if list(data) != expected_names:
            raise ValueError(f"{metric} configuration names or order do not match")
        if any(len(values) != 6 for values in data.values()):
            raise ValueError(f"Every {metric} configuration must contain six folds")


def main():
    validate_data()
    treatment_count = len(MOTA_DATA)
    block_count = len(next(iter(MOTA_DATA.values())))
    error_df = (treatment_count - 1) * (block_count - 1)
    critical_value = dunnett_critical_value(
        number_of_comparisons=treatment_count - 1,
        error_df=error_df,
        alpha=ALPHA,
    )

    analyses = [
        ("MOTA", MOTA_DATA, True, 4),
        ("IDF1", IDF1_DATA, True, 4),
        ("IDS", IDS_DATA, False, 2),
        ("FPS", FPS_DATA, True, 3),
    ]
    for metric, data, higher_is_better, decimals in analyses:
        result = calculate_mcb(
            data,
            higher_is_better=higher_is_better,
            critical_value=critical_value,
        )
        print_results(metric, result, decimals=decimals)
        save_figure(metric, result, decimals=decimals)


if __name__ == "__main__":
    main()
