import torch
from pathlib import Path
from datasets_cbis import CBISDDSM

def get_dataset(task: str, cfg, shuffle_train=True, shuffle_test=False, return_dataset=False):
    if task == 'cbis-ddsm':
        # Assuming data is in sibling folder ../low-dose-mammography-hybrid-diffusion/data/
        project_root = Path("./")
        # Using the same resize as provided in cfg (1024, 768)
        resize_hw = (cfg.image_shape[1], cfg.image_shape[2]) 
        
        train_dataset = CBISDDSM(
            case_csv_path=str(project_root / "data/csv/mass_case_description_train_set.csv"),
            jpeg_root=str(project_root / "data/jpeg"),
            dicom_info_csv=str(project_root / "data/csv/dicom_info.csv"),
            mode="full",
            return_mask=False,
            return_pair=True, # Enable paired dataset for restoration
            noise_level=0.2, # Configure noise level (can be moved to cfg later)
            resize_hw=resize_hw
        )
        test_dataset = CBISDDSM(
            case_csv_path=str(project_root / "data/csv/mass_case_description_test_set.csv"),
            jpeg_root=str(project_root / "data/jpeg"),
            dicom_info_csv=str(project_root / "data/csv/dicom_info.csv"),
            mode="full",
            return_mask=False,
            return_pair=True,
            noise_level=0.2,
            resize_hw=resize_hw
        )
    else:
        print(f"> Unknown dataset '{task}'. Terminating")
        exit()

    print(f"> Train dataset size: {len(train_dataset)}")
    print(f"> Test dataset size: {len(test_dataset)}")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.mini_batch_size, 
        num_workers=cfg.nb_workers, 
        shuffle=shuffle_train
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=cfg.mini_batch_size, 
        num_workers=cfg.nb_workers, 
        shuffle=shuffle_test
    )
    
    if return_dataset:
        return (train_loader, test_loader), (train_dataset, test_dataset)

    return train_loader, test_loader

class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, root_path):
        super().__init__()
        if isinstance(root_path, str):
            root_path = Path(root_path)
        self.files = list(root_path.glob('*.pt'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return [torch.from_numpy(i).long() for i in torch.load(self.files[idx])]

    def get_shape(self, level):
        return self.__getitem__(0)[level].shape

