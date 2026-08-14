"""External-data interface; no medical data or annotations are distributed."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


CONCEPT_COLUMNS = ("EX", "HE", "MA", "SE")
REQUIRED_COLUMNS = ("image_path", "dr_grade", "is_concept_labeled", *CONCEPT_COLUMNS)


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    dr_grade: int
    is_concept_labeled: bool
    concepts: tuple[float, float, float, float]


def read_external_manifest(path: str | Path) -> list[SampleRecord]:
    """Read a user-created split manifest without copying source data.

    Paths are resolved relative to the manifest. A release user must create
    separate train/validation/test manifests from datasets they are authorized
    to access. The repository intentionally provides no patient/image IDs.
    """

    manifest_path = Path(path).resolve()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        records = []
        for row in reader:
            records.append(
                SampleRecord(
                    image_path=(manifest_path.parent / row["image_path"]).resolve(),
                    dr_grade=int(row["dr_grade"]),
                    is_concept_labeled=row["is_concept_labeled"].strip().lower()
                    in {"1", "true", "yes"},
                    concepts=tuple(float(row[name]) for name in CONCEPT_COLUMNS),
                )
            )
    return records


class IDRSSCLDataset(Dataset):
    """Dataset adapter providing the mutable D_S/D_A state used by IDR-SSCL."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        weak_transform: Callable,
        strong_transform: Callable,
        evaluation_transform: Callable,
        *,
        global_offset: int = 0,
    ) -> None:
        self.records = list(records)
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.evaluation_transform = evaluation_transform
        self.global_offset = global_offset
        self.pseudo: dict[int, bool] = {}
        self.anchor: dict[int, bool] = {}
        self.pseudo_labels: dict[int, torch.Tensor] = {}
        self.anchor_labels: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.records)

    def update(self, idx: int, label, value: str) -> None:
        tensor = torch.as_tensor(label, dtype=torch.float32).detach().cpu()
        if value == "pseudo":
            self.pseudo[idx] = True
            self.pseudo_labels[idx] = tensor
        elif value == "anchor":
            self.anchor[idx] = True
            self.anchor_labels[idx] = tensor
        else:
            raise ValueError("value must be 'pseudo' or 'anchor'")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        record = self.records[idx]
        image = Image.open(record.image_path).convert("RGB")
        concepts = torch.tensor(record.concepts, dtype=torch.float32)
        pseudo = self.pseudo.get(idx, False)
        anchor = self.anchor.get(idx, False)
        missing = torch.full_like(concepts, -1)
        return {
            "img0": self.evaluation_transform(image),
            "img": self.weak_transform(image),
            "img_s": self.strong_transform(image),
            "img_m": self.weak_transform(image),
            "drgrading_level": torch.tensor(record.dr_grade, dtype=torch.long),
            "lesion_label": concepts,
            "l": torch.tensor(record.is_concept_labeled),
            "idx": idx + self.global_offset,
            "anchor": torch.tensor(anchor),
            "pseudo": torch.tensor(pseudo),
            "pseudo_label": self.pseudo_labels.get(idx, missing),
            "anchor_label": self.anchor_labels.get(idx, missing),
            "nbr_concepts": concepts.unsqueeze(0),
            "nbr_weight": torch.ones(1, dtype=torch.float32),
        }


def build_transforms(image_size: int = 224):
    """Return the evaluation, weak, and strong transforms used by IDR-SSCL."""
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    spatial = [
        transforms.RandomRotation((-180, 180)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=7, sigma=0.5)], p=0.2
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.87, 1.15),
            ratio=(0.7, 1.3),
            interpolation=InterpolationMode.BILINEAR,
        ),
    ]
    weak = transforms.Compose([*spatial, transforms.ToTensor(), normalize])
    strong = transforms.Compose(
        [*spatial, transforms.RandAugment(3, 5), transforms.ToTensor(), normalize]
    )
    evaluation = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
    )
    return evaluation, weak, strong


def build_loader(
    manifest: str | Path,
    *,
    batch_size: int,
    num_workers: int,
    training: bool,
) -> DataLoader:
    """Build a loader from an explicit user-owned split manifest.

    A one-element ``ConcatDataset`` preserves the mutable dataset contract used
    when DASS promotes pseudo-label and anchor samples during training.
    """
    evaluation, weak, strong = build_transforms()
    dataset = IDRSSCLDataset(
        read_external_manifest(manifest),
        weak_transform=weak if training else evaluation,
        strong_transform=strong if training else evaluation,
        evaluation_transform=evaluation,
    )
    wrapped = ConcatDataset([dataset])
    return DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
