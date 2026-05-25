import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ChickenReIDDataset
from model import build_osnet, BatchHardTripletLoss
from utils import load_config, set_seed, get_device, ensure_dir


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y, _, _ in loader:
        x, y = x.to(device), y.to(device)
        # In eval mode torchreid returns features, not logits.
        # For validation classification accuracy, temporarily use train-like forward by calling classifier is not stable.
        # So here we only ensure forward works. Full ReID report is in report.py.
        _ = model(x)
        total += y.numel()
    return 0.0 if total else 0.0


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

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    device = get_device()
    model = build_osnet(cfg["model_name"], num_classes=len(label_map), pretrained=cfg["pretrained"]).to(device)

    ce_loss = nn.CrossEntropyLoss()
    tri_loss = BatchHardTripletLoss(margin=cfg["triplet_margin"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_loss = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for x, y, _, _ in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, features = model(x)
            loss_ce = ce_loss(logits, y)
            loss_tri = tri_loss(features, y)
            loss = cfg["ce_weight"] * loss_ce + cfg["triplet_weight"] * loss_tri
            loss.backward()
            optimizer.step()

            running += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=running / max(n, 1), ce=loss_ce.item(), triplet=loss_tri.item())

        scheduler.step()
        epoch_loss = running / max(n, 1)
        evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_obj = {
                "model_state": model.state_dict(),
                "label_map": label_map,
                "config": cfg,
                "num_classes": len(label_map),
            }
            torch.save(save_obj, output / "best_osnet.pth")
            print(f"Saved best model to {output / 'best_osnet.pth'}")

    print("Training finished.")


if __name__ == "__main__":
    main()
