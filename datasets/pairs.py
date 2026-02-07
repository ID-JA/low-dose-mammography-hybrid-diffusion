import torch
from torch.utils.data import Dataset
from preprocessing.degrade import degrade_mammogram


class LowDosePairs(Dataset):
    """
    Wraps an existing CBISDDSM dataset to return:
      (low_dose, clean, meta)

    clean: [1,H,W] float32 in [0,1]
    low_dose: [1,H,W] float32 in [0,1]
    """

    def __init__(self, base_dataset: Dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        clean, meta = self.base[idx]
        low = degrade_mammogram(clean)
        return low, clean, meta
