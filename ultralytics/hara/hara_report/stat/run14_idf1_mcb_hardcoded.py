"""MCB analysis of run-14 IDF1 with all observations embedded in this file.

Install dependencies:
    python -m pip install numpy scipy matplotlib

Run:
    python run14_idf1_mcb_hardcoded.py

The statistics are printed to the console. The figure is saved in the current
working directory as ``experiment-data-run14-idf1-mcb-hardcoded.png``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_t


ALPHA = 0.05
RANDOM_SEED = 20260720
IMAGE_NAME = "experiment-data-run14-idf1-mcb-hardcoded.png"

# Values are ordered as hold1, hold2, ..., hold6.
IDF1_DATA = {
    "DeepSORT OBB": [
        0.5920262151829602,
        0.638646288209607,
        0.6644808743169399,
        0.5733333333333334,
        0.5415754923413567,
        0.4977827050997783,
    ],
    "DeepSORT OBB+track optimization": [
        0.8727471327143638,
        0.9115720524017468,
        0.8655737704918033,
        0.6933333333333334,
        0.824945295404814,
        0.6995565410199557,
    ],
    "DeepSORT OBB+detection optimization": [
        0.6670311645708037,
        0.7759562841530054,
        0.7213114754098361,
        0.5814341300722624,
        0.6611932129173509,
        0.5399113082039911,
    ],
    "DeepSORT OBB+track optimization+detection optimization": [
        0.8321487151448879,
        0.912568306010929,
        0.8775956284153006,
        0.631461923290717,
        0.8199233716475096,
        0.6995565410199557,
    ],
    "DeepSORT OBB+ReID": [
        0.6684871654833424,
        0.6910480349344978,
        0.7573770491803279,
        0.6033333333333334,
        0.6094091903719913,
        0.5277161862527716,
    ],
    "DeepSORT HBB": [
        0.6331877729257642,
        0.7676325861126299,
        0.6479338842975206,
        0.44617092119866814,
        0.6986899563318777,
        0.6538249862410567,
    ],
    "DeepSORT HBB+ReID": [
        0.732532751091703,
        0.8059048660470203,
        0.7261707988980717,
        0.58157602663707,
        0.7117903930131004,
        0.662630709961475,
    ],
    "DeepSORT HBB+track optimization": [
        0.9213973799126638,
        0.8813559322033898,
        0.9586776859504132,
        0.7602663706992231,
        0.7794759825327511,
        0.8860759493670886,
    ],
    "DeepSORT HBB+detection optimization": [
        0.7643521049753964,
        0.8212137780207763,
        0.7203530060672918,
        0.492769744160178,
        0.735632183908046,
        0.6898071625344353,
    ],
    "DeepSORT HBB+track optimization+detection optimization": [
        0.9229086932750137,
        0.9436850738108256,
        0.9597352454495311,
        0.7931034482758621,
        0.8144499178981938,
        0.9234159779614325,
    ],
    "BotSORT HBB": [
        0.8438864628820961,
        0.9294696555494806,
        0.9086932750136687,
        0.7480662983425415,
        0.7797164667393675,
        0.7308750687947165,
    ],
    "BotSORT HBB+track optimization": [
        0.962882096069869,
        0.9480590486604702,
        0.9666484417714598,
        0.8519337016574585,
        0.8549618320610687,
        0.7687224669603524,
    ],
    "BotSORT HBB+detection optimization": [
        0.8999453253143794,
        0.9480590486604702,
        0.9671772428884027,
        0.7935805201992252,
        0.8343357025697102,
        0.8165289256198347,
    ],
    "BotSORT HBB+track optimization+detection optimization": [
        0.9218151995626025,
        0.9480590486604702,
        0.9671772428884027,
        0.8046485888212507,
        0.8365226899945325,
        0.8055096418732782,
    ],
    "ByteTrack HBB": [
        0.7631004366812227,
        0.9458720612356479,
        0.8769819573537452,
        0.712707182320442,
        0.787350054525627,
        0.7815079801871216,
    ],
    "ByteTrack HBB+track optimization": [
        0.9192139737991266,
        0.9480590486604702,
        0.9666484417714598,
        0.7756906077348066,
        0.8593238822246456,
        0.8337004405286343,
    ],
    "ByteTrack HBB+detection optimization": [
        0.9032258064516129,
        0.931656642974303,
        0.9671772428884027,
        0.7902600996126176,
        0.839803171131766,
        0.8297520661157025,
    ],
    "ByteTrack HBB+track optimization+detection optimization": [
        0.9218151995626025,
        0.9294696555494806,
        0.9671772428884027,
        0.801328168234643,
        0.861673045379989,
        0.8859504132231405,
    ],
}


def dunnett_critical_value(number_of_comparisons, error_df, alpha):
    """Return the one-sided Dunnett critical value d_(alpha,k,df)."""

    # Comparisons against one common control have pairwise correlation 1/2.
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


def calculate_mcb():
    names = list(IDF1_DATA)
    values = np.array(list(IDF1_DATA.values()), dtype=float)
    treatment_count, block_count = values.shape

    means = values.mean(axis=1)
    standard_deviations = values.std(axis=1, ddof=1)

    # Randomized-complete-block residuals. Each hold/video is a matched block.
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

    critical_value = dunnett_critical_value(
        number_of_comparisons=treatment_count - 1,
        error_df=error_df,
        alpha=ALPHA,
    )
    decision_limit = critical_value * np.sqrt(2.0 * mse / block_count)

    # D_i = mean_i - maximum competing mean. A configuration is in the MCB
    # best set when the simultaneous upper limit D_i + M is nonnegative.
    best_competing_means = np.array(
        [np.max(np.delete(means, index)) for index in range(treatment_count)]
    )
    contrasts = means - best_competing_means
    upper_limits = contrasts + decision_limit
    in_best_set = upper_limits >= 0.0
    order = np.argsort(-means)
    best_index = int(np.argmax(means))

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
    }


def print_results(result):
    names = result["names"]
    best_index = result["best_index"]

    print("IDF1 MULTIPLE COMPARISONS WITH THE BEST (MCB)")
    print("=" * 104)
    print("Model: randomized complete block design (configurations x held-out videos)")
    print(f"Configurations:              {len(names)}")
    print(f"Blocks / folds:              {result['values'].shape[1]}")
    print(f"Error degrees of freedom:    {result['error_df']}")
    print(f"Block-model SSE:             {result['sse']:.6f}")
    print(f"Block-model MSE:             {result['mse']:.6f}")
    print(f"One-sided Dunnett critical:  {result['critical_value']:.4f}")
    print(f"95% decision limit M:        {result['decision_limit']:.4f}")
    print(f"Sample-best configuration:   {names[best_index]}")
    print(f"Sample-best mean IDF1:       {result['means'][best_index]:.4f}")
    print()

    header = (
        f"{'Rank':>4}  {'Configuration':<62} {'Mean':>7} {'SD':>7} "
        f"{'D_i':>8} {'D_i+M':>8} {'Best set':>9}"
    )
    print(header)
    print("-" * len(header))
    for rank, index in enumerate(result["order"], start=1):
        print(
            f"{rank:>4}  "
            f"{names[index]:<62} "
            f"{result['means'][index]:>7.4f} "
            f"{result['standard_deviations'][index]:>7.4f} "
            f"{result['contrasts'][index]:>+8.4f} "
            f"{result['upper_limits'][index]:>+8.4f} "
            f"{'Yes' if result['in_best_set'][index] else 'No':>9}"
        )

    best_set_size = int(result["in_best_set"].sum())
    print()
    print(f"95% MCB best set: {best_set_size} of {len(names)} configurations")
    for index in result["order"]:
        if result["in_best_set"][index]:
            print(f"  - {names[index]}")


def save_figure(result):
    names = result["names"]
    order = result["order"]
    best_mean = result["means"][result["best_index"]]
    deficits = best_mean - result["means"]
    ordered_deficits = deficits[order]
    ordered_names = [names[index] for index in order]
    ordered_membership = result["in_best_set"][order]

    figure, axis = plt.subplots(figsize=(11, 8.5), dpi=180)
    colors = [
        "#2f6db3" if is_in_best_set else "#9aa6b2"
        for is_in_best_set in ordered_membership
    ]
    bars = axis.barh(
        range(len(ordered_names)), ordered_deficits, color=colors, edgecolor="none"
    )

    decision_limit = result["decision_limit"]
    axis.axvspan(0, decision_limit, color="#dceaf7", alpha=0.75, zorder=0)
    axis.axvline(
        decision_limit,
        color="#183b63",
        linestyle="--",
        linewidth=1.5,
        label=f"95% decision limit M = {decision_limit:.4f}",
    )
    axis.set_yticks(range(len(ordered_names)), ordered_names)
    axis.invert_yaxis()
    axis.set_xlabel("Observed IDF1 deficit from sample-best mean")
    axis.set_title(
        "IDF1 Multiple Comparisons with the Best\n"
        "One-sided Dunnett familywise confidence level: 95%"
    )
    axis.grid(axis="x", color="#d7dce1", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", frameon=False)

    for bar, deficit in zip(bars, ordered_deficits):
        axis.text(
            deficit + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{deficit:.4f}",
            va="center",
            fontsize=8,
        )

    axis.set_xlim(
        0,
        max(ordered_deficits.max() * 1.13, decision_limit * 1.3),
    )
    figure.tight_layout()

    image_path = Path.cwd() / IMAGE_NAME
    figure.savefig(image_path, bbox_inches="tight")
    plt.close(figure)
    print()
    print(f"Figure saved to: {image_path}")


def main():
    result = calculate_mcb()
    print_results(result)
    save_figure(result)


if __name__ == "__main__":
    main()
