from pathlib import Path
from PIL import Image
import numpy as np

img_path = next(Path("data/raw/jpeg").rglob("*.jpg"))
img = Image.open(img_path)

arr = np.array(img)

print("Sample image:", img_path)
print("Mode:", img.mode)
print("Shape:", arr.shape)
print("dtype:", arr.dtype)
print("min/max:", arr.min(), arr.max())
