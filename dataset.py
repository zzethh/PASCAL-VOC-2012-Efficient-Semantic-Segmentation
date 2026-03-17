import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils import apply_random_corruption, IGNORE_INDEX

class SegmentationTransform:
    def __init__(self, size=300, is_train=True, noise_prob=0.5):
        self.size = size
        self.is_train = is_train
        self.noise_prob = noise_prob
        self.img_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = image.convert("RGB")
        if self.is_train:
            image = image.resize((self.size, self.size), Image.BILINEAR)
            mask = mask.resize((self.size, self.size), Image.NEAREST)
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
                image = jitter(image)
            if random.random() < self.noise_prob:
                image = apply_random_corruption(image).convert("RGB")
        else:
            image = image.resize((self.size, self.size), Image.BILINEAR)
            mask = mask.resize((self.size, self.size), Image.NEAREST)
        image = transforms.ToTensor()(image)
        image = self.img_normalize(image)  
        mask_np = np.array(mask, dtype=np.int64)
        mask_np[mask_np == 255] = IGNORE_INDEX
        return image, torch.from_numpy(mask_np)

class VOCSegDatasetKaggle(Dataset):
    def __init__(self, root_dir, image_set="train", transform=None):
        self.root_dir, self.image_set, self.transform = root_dir, image_set, transform
        split_file = os.path.join(root_dir, "ImageSets", "Segmentation", f"{image_set}.txt")
        with open(split_file, "r") as f:
            self.file_names = [line.strip() for line in f.readlines() if line.strip()]
        self.images_dir = os.path.join(root_dir, "JPEGImages")
        self.masks_dir = os.path.join(root_dir, "SegmentationClass")

    def __len__(self): return len(self.file_names)
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image = Image.open(os.path.join(self.images_dir, f"{file_name}.jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, f"{file_name}.png"))
        if self.transform: image, mask = self.transform(image, mask)
        return image, mask

def _get_official_kaggle_path():
    import kagglehub
    os.environ["KAGGLEHUB_CACHE"] = "/tmp/kagglehub"
    path = kagglehub.dataset_download('gopalbhattrai/pascal-voc-2012-dataset')
    target_dir = os.path.join(path, "VOC2012_train_val", "VOC2012_train_val")
    if not os.path.exists(target_dir):
        target_dir = os.path.join(path, "VOC2012_train_val")
        if not os.path.exists(os.path.join(target_dir, "ImageSets")):
            target_dir = path
    return target_dir

def get_train_val_datasets(data_root=None, size=300, noise_prob=0.5, download=False, seed=42, use_hf=False):
    train_transform = SegmentationTransform(size=size, is_train=True, noise_prob=noise_prob)
    val_transform = SegmentationTransform(size=size, is_train=False)
    root_dir = _get_official_kaggle_path()
    full_ds = VOCSegDatasetKaggle(root_dir, image_set="train", transform=None)
    indices = list(range(len(full_ds)))
    random.Random(seed).shuffle(indices)
    split = int(0.8 * len(full_ds))
    
    class TransformSubset(Dataset):
        def __init__(self, ds, idxs, tfm):
            self.ds, self.idxs, self.tfm = ds, idxs, tfm
        def __len__(self): return len(self.idxs)
        def __getitem__(self, idx):
            file_name = self.ds.file_names[self.idxs[idx]]
            image = Image.open(os.path.join(self.ds.images_dir, f"{file_name}.jpg")).convert("RGB")
            mask = Image.open(os.path.join(self.ds.masks_dir, f"{file_name}.png"))
            if self.tfm: image, mask = self.tfm(image, mask)
            return image, mask
    return TransformSubset(full_ds, indices[:split], train_transform), TransformSubset(full_ds, indices[split:], val_transform)

def get_test_dataset(data_root=None, size=300, use_hf=False):
    root_dir = _get_official_kaggle_path()
    return VOCSegDatasetKaggle(root_dir, image_set="val", transform=SegmentationTransform(size=size, is_train=False))
