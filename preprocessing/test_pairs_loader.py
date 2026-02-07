import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from datasets.cbis_ddsm import CBISDDSM
from datasets.pairs import LowDosePairs


base = CBISDDSM(
    case_csv_path="data/raw/csv/mass_case_description_train_set.csv",
    jpeg_root="data/raw/jpeg",
    dicom_info_csv="data/raw/csv/dicom_info.csv",
    mode="full",
    return_mask=False,
    resize_hw=(512, 512), # we make the images small for fast training
)

ds = LowDosePairs(base)

loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

low, clean, meta = next(iter(loader))

print("low batch:", low.shape, low.min().item(), low.max().item())
print("clean batch:", clean.shape, clean.min().item(), clean.max().item())
print("meta example:", meta["pathology"][0] if isinstance(meta, dict) else meta[0])

# visualize one example
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(low[0, 0].numpy(), cmap="gray")
plt.title("Low-dose (Input)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(clean[0, 0].numpy(), cmap="gray")
plt.title("Clean (Target)")
plt.axis("off")

plt.tight_layout()
plt.savefig("test_pairs.png",dpi=300)
