from pathlib import Path
from typing import List, Tuple, Dict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import resize_with_padding_pil


class ChickenReIDDataset(Dataset):
    def __init__(self, root: str, input_h: int, input_w: int, use_padding: bool = True, augment: bool = False, label_map: Dict[str, int] = None):
        self.root = Path(root)
        self.input_h = input_h
        self.input_w = input_w
        self.use_padding = use_padding

        self.samples: List[Tuple[str, str]] = []
        for id_dir in sorted(self.root.glob("id_*")):
            if not id_dir.is_dir():
                continue
            for p in sorted(list(id_dir.glob("*.png")) + list(id_dir.glob("*.jpg")) + list(id_dir.glob("*.jpeg"))):
                self.samples.append((str(p), id_dir.name))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

        if label_map is None:
            ids = sorted({sid for _, sid in self.samples})
            self.label_map = {sid: i for i, sid in enumerate(ids)}
        else:
            self.label_map = label_map

        if augment:
            self.tf = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)], p=0.5),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def _preprocess(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        if self.use_padding:
            return resize_with_padding_pil(img, self.input_h, self.input_w)
        return img.resize((self.input_w, self.input_h), Image.BICUBIC)

    def __getitem__(self, idx):
        path, sid = self.samples[idx]
        img = Image.open(path)
        img = self._preprocess(img)
        x = self.tf(img)
        y = self.label_map[sid]
        return x, torch.tensor(y, dtype=torch.long), path, sid
