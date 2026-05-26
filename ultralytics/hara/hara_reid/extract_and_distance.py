import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

from features import ChickenFeatureExtractor, list_images
from utils import load_config, ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--track_dir", required=True, help="Existing track crop images, e.g. prepared_reid/test/id_55")
    parser.add_argument("--detection_dir", required=True, help="New detection crop images, e.g. prepared_reid/test/id_56")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output = Path(cfg["output_dir"])
    ensure_dir(str(output))
    ckpt = args.checkpoint or str(output / "best_osnet.pth")

    track_paths = list_images(args.track_dir)
    detection_paths = list_images(args.detection_dir)
    if not track_paths or not detection_paths:
        raise RuntimeError("track_dir and detection_dir must both contain images.")

    extractor = ChickenFeatureExtractor(ckpt)
    track_features = extractor.extract(track_paths, batch_size=cfg["batch_size"])
    detection_features = extractor.extract(detection_paths, batch_size=cfg["batch_size"])

    # This is the final result needed by DeepSORT.
    cost_matrix = cosine_distances(track_features, detection_features)

    df = pd.DataFrame(
        cost_matrix,
        index=[Path(p).name for p in track_paths],
        columns=[Path(p).name for p in detection_paths],
    )
    df.to_csv(output / "cost_matrix.csv")

    np.save(output / "track_features.npy", track_features)
    np.save(output / "detection_features.npy", detection_features)

    print("cost_matrix shape:", cost_matrix.shape)
    print(df.round(4))
    print(f"\nSaved to: {output / 'cost_matrix.csv'}")
    print("\nInterpretation example:")
    print("cosine distance around 0.08 -> very similar")
    print("cosine distance around 0.45 -> probably different")
    print("cosine distance around 0.80 -> very different")

def main_distance():
    output = Path("./outputs")
    model_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_reid\outputs\best_osnet.pth"
    extractor = ChickenFeatureExtractor(model_path)

    track_paths = list_images(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_reid\hara_distance_test\track")
    detection_paths = list_images(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_reid\hara_distance_test\detection")
    track_features = extractor.extract(track_paths, batch_size=32)
    detection_features = extractor.extract(detection_paths, batch_size=32)

    # This is the final result needed by DeepSORT.
    cost_matrix = cosine_distances(track_features, detection_features)

    df = pd.DataFrame(
        cost_matrix,
        index=[Path(p).name for p in track_paths],
        columns=[Path(p).name for p in detection_paths],
    )
    df.to_csv(output / "cost_matrix.csv")

    np.save(output / "track_features.npy", track_features)
    np.save(output / "detection_features.npy", detection_features)

    print("cost_matrix shape:", cost_matrix.shape)
    print(df.round(4))
    print(f"\nSaved to: {output / 'cost_matrix.csv'}")
    print("\nInterpretation example:")
    print("cosine distance around 0.08 -> very similar")
    print("cosine distance around 0.45 -> probably different")
    print("cosine distance around 0.80 -> very different")

def main_feature():
    model_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_reid\outputs\best_osnet.pth"
    extractor = ChickenFeatureExtractor(model_path)

    track_paths = list_images(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_reid\hara_distance_test\track")
    track_features = extractor.extract(track_paths, batch_size=32)
    print("---")
    print(track_features.shape)
    print(track_features)


if __name__ == "__main__":
    main_feature()
