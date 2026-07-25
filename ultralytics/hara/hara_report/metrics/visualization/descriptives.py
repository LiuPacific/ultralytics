import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1. Input corrected data
# -----------------------------

data = {
    "Fold": [1, 2, 3, 4, 5, 6],
    "Test Video": ["V1", "V2", "V3", "V4", "V5", "V6"],

    "Original MOTA": [0.8907, 0.9355, 0.9180, 0.7891, 0.8503, 0.7967],
    "Optimized MOTA": [0.9279, 0.9541, 0.9322, 0.8546, 0.9005, 0.8317],

    "Original IDF1": [0.6102, 0.6896, 0.7843, 0.5453, 0.5960, 0.5389],
    "Optimized IDF1": [0.9333, 0.9770, 0.9410, 0.7348, 0.8120, 0.7071],

    "Original IDS": [35, 17, 14, 95, 52, 57],
    "Optimized IDS": [2, 0, 2, 30, 9, 26],
}

df = pd.DataFrame(data)

# -----------------------------
# 2. Function for better boxplot
# -----------------------------

def draw_boxplot_with_points(df, cols, title, ylabel, filename=None, decimal=True):
    values = [df[col].values for col in cols]

    plt.figure(figsize=(8, 6))

    box = plt.boxplot(
        values,
        labels=cols,
        widths=0.55,          # make boxes wider
        patch_artist=True,
        showmeans=True,
        meanline=True
    )

    # Overlay individual data points
    # for i, col in enumerate(cols, start=1):
    #     y = df[col].values
    #
    #     # small horizontal jitter so points do not overlap
    #     x = np.random.normal(loc=i, scale=0.035, size=len(y))
    #
    #     plt.scatter(
    #         x,
    #         y,
    #         s=55,
    #         zorder=3,
    #         alpha=0.85
    #     )
    #
    #     # Add fold labels and values
    #     for j, value in enumerate(y):
    #         if decimal:
    #             label = f"V{j+1}: {value:.4f}"
    #         else:
    #             label = f"V{j+1}: {int(value)}"
    #
    #         plt.text(
    #             x[j] + 0.04,
    #             value,
    #             label,
    #             fontsize=9,
    #             va="center"
    #         )

    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=15, fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


# -----------------------------
# 3. MOTA boxplot
# -----------------------------

draw_boxplot_with_points(
    df,
    cols=["Original MOTA", "Optimized MOTA"],
    title="Boxplot of MOTA Across Six Folds",
    ylabel="MOTA",
    filename=".run/boxplot_mota_with_points.png",
    decimal=True
)

# -----------------------------
# 4. IDF1 boxplot
# -----------------------------

draw_boxplot_with_points(
    df,
    cols=["Original IDF1", "Optimized IDF1"],
    title="Boxplot of IDF1 Across Six Folds",
    ylabel="IDF1",
    filename=".run/boxplot_idf1_with_points.png",
    decimal=True
)

# -----------------------------
# 5. IDS boxplot
# -----------------------------

draw_boxplot_with_points(
    df,
    cols=["Original IDS", "Optimized IDS"],
    title="Boxplot of IDS Across Six Folds",
    ylabel="Number of ID Switches",
    filename=".run/boxplot_ids_with_points.png",
    decimal=False
)