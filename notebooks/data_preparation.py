import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Define dataset for the training
class BRATS_dataset(Dataset):
    """
    BRATS 2020 2.5D dataset
    """
    def __init__(self, dataset_path, device):
        self.dataset_path = Path(dataset_path)
        self.device = device
    
        # Subdirectories (one per volume)
        self.subdirs = [p for p in Path(dataset_path).iterdir() if p.is_dir()]

        self.num_volumes = len(self.subdirs)


    def __len__(self):
        return self.num_volumes


    @staticmethod
    def zscore(data):
        mask = data > 0
        if not np.any(mask):
            return data
        vals = data[mask]
        m, s = vals.mean(), vals.std()
        data[mask] = (vals - m) / s if s > 0 else 0
        return data
 


    def __getitem__(self, idx):

        # Load volumes (T1, mask, etc.)
        files = list(self.subdirs[idx].glob("*.nii*"))

        # Explicit mask detect
        mask_file = [f for f in files if "seg" in f.name.lower()][0]
        mask = nib.load(mask_file).get_fdata().transpose(2, 0, 1)

        # All non-mask modalities
        vol_files = [f for f in files if "seg" not in f.name.lower()]
        vols = [nib.load(f).get_fdata() for f in sorted(vol_files)]

        # Extract slabs and reorder to [D,H,W]
        imgs = [vol.transpose(2, 0, 1) for vol in vols]


        classes = [1, 2, 4]
        mask = np.stack([(mask == c).astype(np.float32) for c in classes], axis=0)
        imgs = np.stack(imgs) 
        
        imgs = torch.tensor(imgs, dtype=torch.float32, device=self.device)
        mask = torch.tensor(mask, dtype=torch.float32, device=self.device)

        
        return imgs, mask