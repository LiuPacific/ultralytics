import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import load_config, parse_chicken_filename, set_seed, ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    raw_dir = Path(cfg["raw_image_dir"])
    out_dir = Path(cfg["prepared_dir"])
    ensure_dir(str(out_dir))

    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
        image_paths.extend(raw_dir.glob(ext))

    rows = []
    for p in image_paths:
        if "-" in p.stem: # when id is `-45`, id 45 chicken is fully occluded..
            print(f"[WARN] Skip file name containing '-': {p.name}")
            continue
        info = parse_chicken_filename(str(p))
        if info is None:
            print(f"[WARN] Skip unrecognized file name: {p.name}")
            continue
        rows.append({"path": str(p), "filename": p.name, **info})

    if not rows:
        raise RuntimeError(f"No valid images found in {raw_dir.resolve()}")

    df = pd.DataFrame(rows)
    print("Images per chicken ID:")
    print(df.groupby("chicken_id").size().sort_index())

    train_ratio = float(cfg["train_ratio"])
    val_ratio = float(cfg["val_ratio"])
    test_ratio = float(cfg["test_ratio"])

    split_rows = []
    for chicken_id, g in df.groupby("chicken_id"):
        g = g.sample(frac=1.0, random_state=cfg["seed"])
        if len(g) < 3:
            print(f"[WARN] chicken id {chicken_id} has <3 images; put all into train")
            for _, r in g.iterrows():
                split_rows.append({**r.to_dict(), "split": "train"})
            continue

        train_df, temp_df = train_test_split(g, train_size=train_ratio, random_state=cfg["seed"])
        rel_val = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(temp_df, train_size=rel_val, random_state=cfg["seed"])

        for split, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
            for _, r in part.iterrows():
                split_rows.append({**r.to_dict(), "split": split})

    split_df = pd.DataFrame(split_rows)

    # Copy files into prepared_reid/{train,val,test}/id_XX/*.png
    for _, r in split_df.iterrows():
        dst_dir = out_dir / r["split"] / f"id_{int(r['chicken_id'])}"
        ensure_dir(str(dst_dir))
        shutil.copy2(r["path"], dst_dir / r["filename"])

    split_df.to_csv(out_dir / "metadata.csv", index=False)
    print(f"\nPrepared dataset written to: {out_dir.resolve()}")
    print(split_df.groupby(["split", "chicken_id"]).size())


if __name__ == "__main__":
    main()
