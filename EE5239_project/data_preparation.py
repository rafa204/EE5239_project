import numpy as np
import nibabel as nib
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

data_cache = {}

#Define dataset for the training
class BRATS_dataset(Dataset):
    """
    BRATS 2020 2.5D dataset
    """
    def __init__(self, dataset_path, device, num_volumes = None, slices_per_volume = 1, num_slices = None):
        self.dataset_path = dataset_path
        self.device = device
    
        # Subdirectories (one per volume)
        self.subdirs = [p for p in self.dataset_path.iterdir() if p.is_dir()]

        self.slices_per_volume = slices_per_volume

        if num_volumes is None:
            self.num_volumes = len(self.subdirs)
        else:
            self.num_volumes = num_volumes

        self.num_slices = num_slices

    def __len__(self):
        if self.num_slices is None: 
            return self.num_volumes * self.slices_per_volume
        else:
            return self.num_slices

    def __getitem__(self, idx):

        file_idx = idx//self.slices_per_volume

        # Load volumes (T1, mask, etc.)
        files = list(self.subdirs[file_idx].glob("*.nii*"))

        # Explicit mask detect
        mask_file = [f for f in files if "seg" in f.name.lower()][0]
        mask = nib.load(mask_file).get_fdata()

        # All non-mask modalities
        vol_files = [f for f in files if "seg" not in f.name.lower()]
        imgs = [nib.load(f).get_fdata() for f in sorted(vol_files)]

        classes = [1, 2, 4]
        mask = np.stack([(mask == c).astype(np.float32) for c in classes])
        mask = mask.sum(axis=0)
        imgs = np.stack(imgs, axis = 3)
        
        flat_indices = np.flatnonzero(mask)

        # Generate a single point inside the mask and get the corresponding slice
        rng = np.random.default_rng(idx)
        i = rng.integers(0,len(flat_indices),1)
        x,y,z = np.unravel_index(flat_indices[i], mask.shape)

        imgs_out = imgs[:, :, z, 0:3].squeeze()
        masks_out = mask[:, :, z].squeeze()

        #Crop
        cx, cy = mask.shape[0]//2,  mask.shape[1]//2
        masks_out = masks_out[cx-80:cx+80, cy-112:cy+112]
        imgs_out = imgs_out[cx-80:cx+80, cy-112:cy+112, :]

        #Normalize
        imgs_out = (imgs_out/imgs_out.max() * 255).astype(np.uint8)

        return imgs_out, masks_out
    

class BRATS_dataset_2D(Dataset):

    def __init__(self, path, device):
        self.path = Path(path)
        self.device = device
        self.subdirs = [p for p in self.path.iterdir() if p.is_dir()]

    def __len__(self):
        return len(self.subdirs)

    def transform(self, image, mask):

        # Random horizontal flipping
        if random.random() > 0.5:
            image = np.flip(image, axis = 0).copy()
            mask = np.flip(mask, axis = 0).copy()

        # Random vertical flipping
        if random.random() > 0.5:
            image = np.flip(image, axis = 1).copy()
            mask = np.flip(mask, axis = 1).copy()

        return image, mask

    def __getitem__(self, idx):

        img = np.load(self.subdirs[idx] / "imgs.npy")
        mask = np.load(self.subdirs[idx] / "mask.npy")

        img, mask = self.transform(img, mask)

        flat_indices = np.flatnonzero(mask)
        rng = np.random.default_rng(idx)
        i = rng.integers(0,len(flat_indices),1)
        x,y = np.unravel_index(flat_indices[i], mask.shape)

        input_point = np.array([[x,y]]).squeeze(0).T
        
        mask = torch.tensor(mask, device = self.device, dtype=torch.float32)

        return img, mask, input_point
    
