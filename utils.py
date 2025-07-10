import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
       valid_ext = ('.jpg', '.jpeg', '.png')
       self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)])
       self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.lower().endswith(valid_ext)])
       self.transform = transform
       self.mask_transform = mask_transform
       assert len(self.image_paths) == len(self.mask_paths), \
            f"Number of images and masks do not match: {len(self.image_paths)} vs {len(self.mask_paths)}"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask
