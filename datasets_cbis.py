from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset


@dataclass
class CBISPathResolver:
    """
    Resolves CBIS-DDSM Kaggle CSV DICOM-like paths to real JPEG paths
    using dicom_info.csv, which contains the official mapping.

    It builds maps like:
      SeriesInstanceUID -> JPEG absolute path
    for full/cropped/mask images.
    """
    jpeg_root: Path
    dicom_info_csv: Path

    full_map: Dict[str, Path] = None
    crop_map: Dict[str, Path] = None
    mask_map: Dict[str, Path] = None

    def __post_init__(self):
        self.jpeg_root = Path(self.jpeg_root)
        self.dicom_info_csv = Path(self.dicom_info_csv)
        self._build_maps()

    @staticmethod
    def _normalize_series_desc(s: str) -> str:
        return str(s).strip().lower()

    @staticmethod
    def _extract_series_uid_from_case_path(case_path: str) -> str:
        """
        The case_description CSV uses paths like:
        Mass-Training_.../<StudyUID>/<SeriesUID>/000000.dcm

        We take the second UID folder as SeriesUID:
        .../<StudyUID>/<SeriesUID>/...
        """
        parts = Path(case_path).parts
        # Keep only UID-like parts
        uids = [p for p in parts if p.startswith("1.3.6.1.4.1.9590")]
        if len(uids) < 1:
            raise ValueError(f"Could not find UID in path: {case_path}")

        return uids[-1]

    def _build_maps(self):
        df = pd.read_csv(self.dicom_info_csv)

        required = {"SeriesInstanceUID", "SeriesDescription", "image_path"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"dicom_info.csv missing columns: {missing}")

        # Convert image_path like:
        # "CBIS-DDSM/jpeg/<SeriesUID>/1-249.jpg"
        # to absolute path rooted at jpeg_root
        def to_abs(image_path: str) -> Path:
            # keep only the tail after ".../jpeg/"
            p = str(image_path).replace("\\", "/")
            if "/jpeg/" in p:
                tail = p.split("/jpeg/", 1)[1]
            elif p.startswith("jpeg/"):
                tail = p.split("jpeg/", 1)[1]
            else:
                tail = p
            return self.jpeg_root / tail

        full_map = {}
        crop_map = {}
        mask_map = {}

        for _, r in df.iterrows():
            series_uid = str(r["SeriesInstanceUID"])
            desc = self._normalize_series_desc(r["SeriesDescription"])
            img_abs = to_abs(r["image_path"])

            if not img_abs.exists():
                continue

            if "full mammogram" in desc:
                # TODO: One series may contain multiple images (CC/MLO),
                # but in this Kaggle dataset typically it's one image per series.
                full_map.setdefault(series_uid, img_abs)

            elif "cropped" in desc:
                crop_map.setdefault(series_uid, img_abs)

            elif "roi mask" in desc:
                mask_map.setdefault(series_uid, img_abs)

        self.full_map = full_map
        self.crop_map = crop_map
        self.mask_map = mask_map

        if len(self.full_map) == 0:
            print("Warning: full_map is empty. Check jpeg_root or dicom_info.csv format.")
        else:
            print(f"Resolver built: full={len(self.full_map)}, crop={len(self.crop_map)}, mask={len(self.mask_map)}")

    def resolve(self, case_path: str, kind: str) -> Path:
        """
        kind: "full" | "crop" | "mask"
        """
        series_uid = self._extract_series_uid_from_case_path(case_path)

        if kind == "full":
            if series_uid in self.full_map:
                return self.full_map[series_uid]
        elif kind == "crop":
            if series_uid in self.crop_map:
                return self.crop_map[series_uid]
        elif kind == "mask":
            if series_uid in self.mask_map:
                return self.mask_map[series_uid]
        else:
            raise ValueError(f"Invalid kind: {kind}")

        raise FileNotFoundError(f"Could not resolve {kind} image for SeriesUID={series_uid}\nfrom path: {case_path}")


class CBISDDSM(Dataset):
    """
    Improved CBIS-DDSM loader using dicom_info.csv mapping.
    Fast + deterministic (no glob searching).

    Returns:
        x:   Tensor [1,H,W] float32 in [0,1]
        meta dict
    Optionally:
        mask: Tensor [1,H,W] float32 {0,1}
    """

    def __init__(
        self,
        case_csv_path: str,
        jpeg_root: str,
        dicom_info_csv: str,
        mode: str = "full", 
        return_mask: bool = False,
        resize_hw: Optional[Tuple[int, int]] = (1024, 768),
    ):
        self.df = pd.read_csv(case_csv_path)
        self.jpeg_root = Path(jpeg_root)
        self.return_mask = return_mask
        self.resize_hw = resize_hw

        assert mode in ["full", "crop"], "mode must be 'full' or 'crop'"
        self.mode = mode

        # Columns from case_description CSV
        self.full_col = "image file path"
        self.crop_col = "cropped image file path"
        self.mask_col = "ROI mask file path"

        self.resolver = CBISPathResolver(
            jpeg_root=self.jpeg_root,
            dicom_info_csv=Path(dicom_info_csv),
        )

        if self.resize_hw:
            self.transform = transforms.Compose([
                transforms.Resize(self.resize_hw, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        if self.mode == "full":
            case_path = row[self.full_col]
            img_path = self.resolver.resolve(case_path, kind="full")
        else:
            case_path = row[self.crop_col]
            img_path = self.resolver.resolve(case_path, kind="crop")

        img = Image.open(img_path).convert("L")
        x = self.transform(img)

        meta = {
            "patient_id": row.get("patient_id", None),
            "pathology": row.get("pathology", None),
            "view": row.get("image view", None),
            "side": row.get("left or right breast", None),
            "assessment": row.get("assessment", None),
            "abnormality_type": row.get("abnormality type", None),
            "resolved_path": str(img_path),
            "case_path": str(case_path),
        }

        if not self.return_mask:
            return x, meta

        mask_case_path = row.get(self.mask_col, None)
        if pd.isna(mask_case_path) or mask_case_path is None:
            mask = torch.zeros_like(x)
        else:
            mask_path = self.resolver.resolve(mask_case_path, kind="mask")
            mask_img = Image.open(mask_path).convert("L")
            mask = self.transform(mask_img)
            mask = (mask > 0).float()

        return x, mask, meta
