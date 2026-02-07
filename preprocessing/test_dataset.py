import matplotlib.pyplot as plt
from datasets.cbis_ddsm import CBISDDSM

ds = CBISDDSM(
    case_csv_path="data/raw/csv/mass_case_description_train_set.csv",
    jpeg_root="data/raw/jpeg",
    dicom_info_csv="data/raw/csv/dicom_info.csv",
    mode="full",
    return_mask=False,
)

x, meta = ds[0]
print("Tensor:", x.shape, float(x.min()), float(x.max()))
print("Resolved path:", meta["resolved_path"])

plt.figure(figsize=(6, 6))
plt.imshow(x[0], cmap="gray")
plt.title(f"{meta['pathology']} | {meta['view']} | {meta['side']}")
plt.axis("off")
plt.show()
