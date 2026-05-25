import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

from features import ChickenFeatureExtractor, list_images
from utils import load_config, ensure_dir


def id_from_parent(path: str) -> str:
    return Path(path).parent.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output = Path(cfg["output_dir"])
    ensure_dir(str(output))
    ckpt = args.checkpoint or str(output / "best_osnet.pth")

    test_dir = Path(cfg["prepared_dir"]) / "test"
    image_paths = list_images(str(test_dir))
    if len(image_paths) < 2:
        raise RuntimeError("Need at least 2 test images to create report.")

    labels = np.array([id_from_parent(p) for p in image_paths])
    extractor = ChickenFeatureExtractor(ckpt)
    features = extractor.extract(image_paths, batch_size=cfg["batch_size"])
    dist = cosine_distances(features, features)

    same, diff = [], []
    n = len(image_paths)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                same.append(dist[i, j])
            else:
                diff.append(dist[i, j])

    same = np.array(same)
    diff = np.array(diff)

    # Rank-1 retrieval: nearest non-self image should have same ID.
    rank1_correct = 0
    valid = 0
    for i in range(n):
        d = dist[i].copy()
        d[i] = np.inf
        j = np.argmin(d)
        rank1_correct += int(labels[i] == labels[j])
        valid += 1
    rank1 = rank1_correct / max(valid, 1)

    # Simple threshold suggestion: midpoint between median same-ID and median different-ID distances.
    if len(same) > 0 and len(diff) > 0:
        suggested_threshold = float((np.median(same) + np.median(diff)) / 2)
    else:
        suggested_threshold = None

    rows = []
    rows.append(["num_test_images", n])
    rows.append(["num_chicken_ids", len(set(labels))])
    rows.append(["rank1_accuracy", rank1])
    if len(same) > 0:
        rows += [
            ["same_id_mean_distance", float(np.mean(same))],
            ["same_id_median_distance", float(np.median(same))],
            ["same_id_min_distance", float(np.min(same))],
            ["same_id_max_distance", float(np.max(same))],
        ]
    if len(diff) > 0:
        rows += [
            ["different_id_mean_distance", float(np.mean(diff))],
            ["different_id_median_distance", float(np.median(diff))],
            ["different_id_min_distance", float(np.min(diff))],
            ["different_id_max_distance", float(np.max(diff))],
        ]
    rows.append(["suggested_cosine_threshold", suggested_threshold])

    report_df = pd.DataFrame(rows, columns=["metric", "value"])
    report_df.to_csv(output / "reid_report.csv", index=False)
    print(report_df)

    if len(same) > 0:
        plt.figure()
        plt.hist(same, bins=30, alpha=0.7, label="same ID")
        if len(diff) > 0:
            plt.hist(diff, bins=30, alpha=0.7, label="different ID")
        if suggested_threshold is not None:
            plt.axvline(suggested_threshold, linestyle="--", label=f"threshold={suggested_threshold:.3f}")
        plt.xlabel("Cosine distance")
        plt.ylabel("Pair count")
        plt.title("Chicken ReID distance distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / "distance_distribution.png", dpi=200)
        print(f"Saved plot to: {output / 'distance_distribution.png'}")


if __name__ == "__main__":
    main()
