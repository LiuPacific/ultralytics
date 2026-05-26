import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_distances
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ChickenReIDDataset
from model import build_osnet, BatchHardTripletLoss
from utils import load_config, set_seed, get_device, ensure_dir


def _set_bn_dropout_eval(module: nn.Module) -> None:
    """Keep the OSNet top module in train mode so it returns (logits, features),
    but freeze BatchNorm/Dropout behavior during validation.

    Torchreid OSNet returns logits only when model.training is True. If we call
    model.eval(), it returns features only. This helper gives us validation logits
    without updating BatchNorm running statistics.
    """
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Dropout)):
            m.eval()


def batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == labels).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ce_loss: nn.Module,
    tri_loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Dict,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "ce": 0.0, "triplet": 0.0, "acc": 0.0, "n": 0}

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']} [train]")
    for x, y, _, _ in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # In torchreid, OSNet with loss='triplet' returns (classification logits, embedding features)
        # when model.train() is active.
        logits, features = model(x)
        loss_ce = ce_loss(logits, y)
        loss_tri = tri_loss(features, y)
        loss = cfg["ce_weight"] * loss_ce + cfg["triplet_weight"] * loss_tri

        loss.backward()
        optimizer.step()

        bs = x.size(0)
        totals["loss"] += loss.item() * bs
        totals["ce"] += loss_ce.item() * bs
        totals["triplet"] += loss_tri.item() * bs
        totals["acc"] += batch_accuracy(logits, y) * bs
        totals["n"] += bs

        pbar.set_postfix(
            loss=totals["loss"] / max(totals["n"], 1),
            ce=totals["ce"] / max(totals["n"], 1),
            triplet=totals["triplet"] / max(totals["n"], 1),
            acc=totals["acc"] / max(totals["n"], 1),
        )

    n = max(totals["n"], 1)
    return {
        "train_loss": totals["loss"] / n,
        "train_ce_loss": totals["ce"] / n,
        "train_triplet_loss": totals["triplet"] / n,
        "train_cls_acc": totals["acc"] / n,
    }


@torch.no_grad()
def validate_loss_and_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ce_loss: nn.Module,
    tri_loss: nn.Module,
    cfg: Dict,
) -> Dict[str, float]:
    # See _set_bn_dropout_eval(): torchreid needs model.training=True to return logits.
    model.train()
    _set_bn_dropout_eval(model)

    totals = {"loss": 0.0, "ce": 0.0, "triplet": 0.0, "acc": 0.0, "n": 0}
    for x, y, _, _ in tqdm(loader, desc="Validation loss/acc", leave=False):
        x, y = x.to(device), y.to(device)
        logits, features = model(x)
        loss_ce = ce_loss(logits, y)
        loss_tri = tri_loss(features, y)
        loss = cfg["ce_weight"] * loss_ce + cfg["triplet_weight"] * loss_tri

        bs = x.size(0)
        totals["loss"] += loss.item() * bs
        totals["ce"] += loss_ce.item() * bs
        totals["triplet"] += loss_tri.item() * bs
        totals["acc"] += batch_accuracy(logits, y) * bs
        totals["n"] += bs

    n = max(totals["n"], 1)
    return {
        "val_loss": totals["loss"] / n,
        "val_ce_loss": totals["ce"] / n,
        "val_triplet_loss": totals["triplet"] / n,
        "val_cls_acc": totals["acc"] / n,
    }


@torch.no_grad()
def validate_reid_distance(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Calculate validation ReID metrics using cosine distances between embeddings.

    This is closer to how DeepSORT will use the model, because DeepSORT uses the
    embedding vector rather than the classifier head.
    """
    model.eval()
    feats, labels = [], []
    for x, y, _, _ in tqdm(loader, desc="Validation ReID distance", leave=False):
        x = x.to(device)
        f = model(x)  # in eval mode, torchreid returns features only
        f = torch.nn.functional.normalize(f, p=2, dim=1)
        feats.append(f.cpu().numpy())
        labels.extend(y.numpy().tolist())

    features = np.concatenate(feats, axis=0)
    labels = np.array(labels)
    dist = cosine_distances(features, features)
    n = len(labels)

    # Rank-1 retrieval: nearest non-self image should have same ID.
    correct = 0
    for i in range(n):
        d = dist[i].copy()
        d[i] = np.inf
        j = int(np.argmin(d))
        correct += int(labels[i] == labels[j])
    rank1 = correct / max(n, 1)

    same, diff = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                same.append(dist[i, j])
            else:
                diff.append(dist[i, j])

    same = np.array(same, dtype=float)
    diff = np.array(diff, dtype=float)
    result = {
        "val_rank1": rank1,
        "val_same_mean_dist": float(np.mean(same)) if len(same) else np.nan,
        "val_diff_mean_dist": float(np.mean(diff)) if len(diff) else np.nan,
        "val_same_median_dist": float(np.median(same)) if len(same) else np.nan,
        "val_diff_median_dist": float(np.median(diff)) if len(diff) else np.nan,
    }
    if len(same) and len(diff):
        result["suggested_threshold"] = float((np.median(same) + np.median(diff)) / 2.0)
    else:
        result["suggested_threshold"] = np.nan
    return result


def save_training_curves(log_df: pd.DataFrame, output: Path) -> None:
    plt.figure()
    plt.plot(log_df["epoch"], log_df["train_loss"], label="train loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("OSNet training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "training_loss_curve.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(log_df["epoch"], log_df["train_cls_acc"], label="train classification acc")
    plt.plot(log_df["epoch"], log_df["val_cls_acc"], label="validation classification acc")
    plt.plot(log_df["epoch"], log_df["val_rank1"], label="validation ReID rank-1")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("OSNet classification and ReID validation metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "training_accuracy_curve.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(log_df["epoch"], log_df["val_same_mean_dist"], label="same-ID mean distance")
    plt.plot(log_df["epoch"], log_df["val_diff_mean_dist"], label="different-ID mean distance")
    plt.xlabel("Epoch")
    plt.ylabel("Cosine distance")
    plt.title("Validation embedding distance separation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "validation_distance_curve.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    prepared = Path(cfg["prepared_dir"])
    output = Path(cfg["output_dir"])
    ensure_dir(str(output))

    train_ds = ChickenReIDDataset(
        prepared / "train", cfg["input_height"], cfg["input_width"], cfg["use_padding"], augment=True
    )
    label_map = train_ds.label_map
    val_ds = ChickenReIDDataset(
        prepared / "val", cfg["input_height"], cfg["input_width"], cfg["use_padding"], augment=False, label_map=label_map
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        drop_last=False,
    )

    device = get_device()
    print(f"Using device: {device}")
    print(f"Train images: {len(train_ds)} | Validation images: {len(val_ds)} | IDs: {len(label_map)}")

    model = build_osnet(cfg["model_name"], num_classes=len(label_map), pretrained=cfg["pretrained"]).to(device)

    ce_loss = nn.CrossEntropyLoss()
    tri_loss = BatchHardTripletLoss(margin=cfg["triplet_margin"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_val_loss = float("inf")
    logs = []

    for epoch in range(1, cfg["epochs"] + 1):
        train_metrics = train_one_epoch(model, train_loader, device, ce_loss, tri_loss, optimizer, cfg, epoch)
        val_metrics = validate_loss_and_accuracy(model, val_loader, device, ce_loss, tri_loss, cfg)
        reid_metrics = validate_reid_distance(model, val_loader, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **val_metrics,
            **reid_metrics,
        }
        logs.append(row)
        log_df = pd.DataFrame(logs)
        log_df.to_csv(output / "training_log.csv", index=False)
        save_training_curves(log_df, output)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.4f}, val_loss={row['val_loss']:.4f}, "
            f"train_acc={row['train_cls_acc']:.4f}, val_acc={row['val_cls_acc']:.4f}, "
            f"val_rank1={row['val_rank1']:.4f}, "
            f"same_dist={row['val_same_mean_dist']:.4f}, diff_dist={row['val_diff_mean_dist']:.4f}, "
            f"threshold={row['suggested_threshold']:.4f}"
        )

        # Save the checkpoint with the lowest validation loss.
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            save_obj = {
                "model_state": model.state_dict(),
                "label_map": label_map,
                "config": cfg,
                "num_classes": len(label_map),
                "best_epoch": epoch,
                "best_val_loss": best_val_loss,
            }
            torch.save(save_obj, output / "best_osnet.pth")
            print(f"Saved best model to {output / 'best_osnet.pth'}")

        # Also save latest checkpoint so training progress is never lost.
        torch.save(
            {
                "model_state": model.state_dict(),
                "label_map": label_map,
                "config": cfg,
                "num_classes": len(label_map),
                "epoch": epoch,
            },
            output / "last_osnet.pth",
        )

    print("Training finished.")
    print(f"Training log: {output / 'training_log.csv'}")
    print(f"Loss curve: {output / 'training_loss_curve.png'}")
    print(f"Accuracy curve: {output / 'training_accuracy_curve.png'}")
    print(f"Distance curve: {output / 'validation_distance_curve.png'}")


if __name__ == "__main__":
    main()
