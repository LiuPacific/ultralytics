from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import build_osnet
from utils import resize_with_padding_pixel, get_device


class ChickenFeatureExtractor:
    def __init__(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.cfg = ckpt["config"]
        self.label_map = ckpt["label_map"]
        self.device = get_device()
        self.model = build_osnet(
            self.cfg["model_name"],
            num_classes=ckpt["num_classes"],
            pretrained=False,
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if self.cfg["use_padding"]:
            img = resize_with_padding_pixel(img, self.cfg["input_height"], self.cfg["input_width"])
        else:
            img = img.resize((self.cfg["input_width"], self.cfg["input_height"]), Image.BICUBIC)
        return self.tf(img)

    @torch.no_grad()
    def extract(self, image_paths: List[str], batch_size: int = 64) -> np.ndarray:
        feats = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i + batch_size]
            x = torch.stack([self.preprocess(p) for p in batch_paths], dim=0).to(self.device)
            f = self.model(x)
            # L2 normalization makes cosine distance stable.
            f = torch.nn.functional.normalize(f, p=2, dim=1)
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)


def list_images(folder: str) -> List[str]:
    folder = Path(folder)
    paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
        paths.extend(folder.rglob(ext))
    return [str(p) for p in sorted(paths)]
