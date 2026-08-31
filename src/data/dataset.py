import os
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class NYUDepthDataset(Dataset):
    def __init__(self, base_dir, csv_file, img_size=(384, 384), is_train=True):
        self.base_dir = base_dir
        self.img_size = img_size
        self.is_train = is_train
        self.df = pd.read_csv(csv_file, header=None)

        spatial_transforms = [
            A.Resize(*self.img_size),
        ]

        if self.is_train:
            spatial_transforms.append(A.HorizontalFlip(p=0.5))
            color_transforms = [
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.4
                ),
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3
                ),
            ]
        else:
            color_transforms = []

        final_transforms = [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

        transforms_list = spatial_transforms + color_transforms + final_transforms

        self.transform = A.Compose(transforms_list, additional_targets={"mask": "mask"})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_dir, self.df.iat[idx, 0])
        depth_path = os.path.join(self.base_dir, self.df.iat[idx, 1])

        img_np = np.array(Image.open(img_path).convert("RGB"))

        depth_img = Image.open(depth_path)

        if depth_img.mode in ["I;16", "I", "F"]:
            depth_np = np.array(depth_img, dtype=np.float32) / 10000.0
        else:
            depth_np = np.array(depth_img, dtype=np.float32) / 255.0

        depth_np = np.clip(depth_np, 0.0, 1.0)

        augmented = self.transform(image=img_np, mask=depth_np)
        img_tensor = augmented["image"]  # [3, H, W]
        depth_tensor = augmented["mask"].unsqueeze(0)  # [1, H, W]

        return img_tensor, depth_tensor


if __name__ == "__main__":
    from src.config import BASE_DIR, DATA_DIR

    csv_train = DATA_DIR / "nyu2_train.csv"

    dataset = NYUDepthDataset(base_dir=BASE_DIR, csv_file=csv_train, is_train=True)
    print(f"Number of samples: {len(dataset)}")

    img, depth = dataset[0]

    print(f"RGB dtype: {img.shape}, dtype: {img.dtype}")
    print(f"Depth dtype: {depth.shape}, dtype: {depth.dtype}")
    print(f"Depth min: {depth.min():.2f}m, Depth max: {depth.max():.2f}m")
