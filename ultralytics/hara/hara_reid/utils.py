import os
import re
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps


FILENAME_RE = re.compile(
    r"^id_(?P<chicken_id>\d+)_(?P<month>\d+)_(?P<modality>RGB|T|Thermal|thermal)_(?P<group>sick|mock)_(?P<frame>.+)\.(png|jpg|jpeg)$",
    re.IGNORECASE,
)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_chicken_filename(path: str) -> Optional[Dict]:
    name = Path(path).name
    m = FILENAME_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    d["chicken_id"] = int(d["chicken_id"])
    d["month"] = int(d["month"])
    d["modality"] = d["modality"].upper()
    d["group"] = d["group"].lower()
    return d


def resize_with_padding_pixel(img: Image.Image, target_h: int, target_w: int) -> Image.Image:
    """Keep aspect ratio, then pad to target_h x target_w."""
    img = img.convert("RGB")
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    pad_right = target_w - new_w - pad_left
    pad_bottom = target_h - new_h - pad_top
    return ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(0, 0, 0))


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
