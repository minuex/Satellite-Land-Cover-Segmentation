import os
import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from config import class_mapping
from preprocessing import enhance_edges, enhance_saturation, enhance_clahe
import torchvision.transforms as T
import torch

#############################################################################
def mask_to_tensor(mask):
    return torch.tensor(np.array(mask), dtype=torch.long)

image_transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_transform = T.Compose([
    T.Resize((512, 512)),
    T.Lambda(mask_to_tensor) # 정수형 유지 (정규화 X)
])
#############################################################################

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, class_mapping, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
        self.class_mapping = class_mapping
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 전처리 적용
        image = enhance_edges(image)
        image = enhance_saturation(image)
        image = enhance_clahe(image)

        # 클래스 매핑 적용
        mask_array = np.array(mask)
        normalized_mask = np.zeros_like(mask_array)
        for old_class, new_class in self.class_mapping.items():
            normalized_mask[mask_array == old_class] = new_class

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = Image.fromarray(normalized_mask)
            mask = self.mask_transform(mask)

        return image, mask

train_image_dir = 'AIHub/train/images'
train_mask_dir = 'AIHub/train/labels'

valid_image_dir = 'AIHub/valid/images'
valid_mask_dir = 'AIHub/valid/labels'

train_dataset = SegmentationDataset(
    image_dir=train_image_dir,
    mask_dir=train_mask_dir,
    class_mapping = class_mapping,
    image_transform=image_transform,
    mask_transform=mask_transform,
)

valid_dataset = SegmentationDataset(
    image_dir=valid_image_dir,
    mask_dir=valid_mask_dir,
    class_mapping = class_mapping,
    image_transform=image_transform,
    mask_transform=mask_transform,
)

print(f"🔍 Checking dataset paths:")
print(f"Train images: {train_image_dir}, Found: {len(glob.glob(os.path.join(train_image_dir, '*.tif')))} files")
print(f"Train masks: {train_mask_dir}, Found: {len(glob.glob(os.path.join(train_mask_dir, '*.tif')))} files")
print(f"Valid images: {valid_image_dir}, Found: {len(glob.glob(os.path.join(valid_image_dir, '*.tif')))} files")
print(f"Valid masks: {valid_mask_dir}, Found: {len(glob.glob(os.path.join(valid_mask_dir, '*.tif')))} files")

