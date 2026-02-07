import matplotlib.pyplot as plt
from datasets.cbis_ddsm import CBISDDSM
from preprocessing.degrade import degrade_mammogram

ds = CBISDDSM(
    case_csv_path="data/raw/csv/mass_case_description_train_set.csv",
    jpeg_root="data/raw/jpeg",
    dicom_info_csv="data/raw/csv/dicom_info.csv",
    mode="full",
    return_mask=False,
)

clean, meta = ds[0]
low = degrade_mammogram(clean)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(clean[0], cmap="gray")
plt.title("Clean (Target)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(low[0], cmap="gray")
plt.title("Simulated Low-Dose (Input)")
plt.axis("off")

plt.tight_layout()
plt.show()
plt.savefig("test_degrade.png", dpi=300)
