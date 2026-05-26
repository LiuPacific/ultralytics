from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import build_osnet
from utils import resize_with_padding_pixel, get_device, resize_with_padding_numpy, numpy_to_tensor

import cv2


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

    def preprocess(self, path: str, target_height, target_width) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if self.cfg["use_padding"]:
            img = resize_with_padding_pixel(img, target_height, target_width)
        else:
            img = img.resize((target_height, target_width), Image.BICUBIC)
        return self.tf(img)

    @torch.no_grad()
    def extract(self, image_paths: List[str], batch_size: int = 64, target_height: int = 256,
                target_width: int = 128) -> np.ndarray:
        feats = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i + batch_size]
            x = torch.stack([self.preprocess(p, target_height, target_width) for p in batch_paths], dim=0).to(
                self.device)
            f = self.model(x)
            # L2 normalization makes cosine distance stable.
            f = torch.nn.functional.normalize(f, p=2, dim=1)
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)

    def preprocess_numpy(self, img: np.ndarray, target_height: int = 256, target_width: int = 128) -> torch.Tensor:
        """
        img should be RGB numpy image: [H, W, 3]
        """

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        if img.shape[2] == 4:
            img = img[:, :, :3]

        if self.cfg["use_padding"]:
            img = resize_with_padding_numpy(
                img,
                target_height,
                target_width
            )
        else:
            img = cv2.resize(
                img,
                (target_width, target_height),
                interpolation=cv2.INTER_CUBIC
            )

        return numpy_to_tensor(img)

    @torch.no_grad()
    def extract_numpy(self, images: List[np.ndarray], batch_size: int = 32, target_height: int = 256,
                target_width: int = 128) -> np.ndarray:
        """
        Input:
            images: list of RGB numpy crops from YOLO
                    each image shape = [H, W, 3]

        Output:
            features: numpy array, shape [N, 512]
        """
        feats = []

        for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
            batch_imgs = images[i:i + batch_size]

            x = torch.stack(
                [self.preprocess_numpy(img, target_height=target_height, target_width=target_width) for img in
                 batch_imgs],
                dim=0
            ).to(self.device)

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
