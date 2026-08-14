"""Minimal training and evaluation entrypoints for authorized local data."""

from __future__ import annotations

import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from src.train.metrics import configure_metrics, update_metrics


def fit_model(
    model: pl.LightningModule,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    output_dir: str | Path,
    max_epochs: int,
    patience: int,
    accelerator: str,
    devices: int | str,
) -> Path:
    """Fit one official IDR-SSCL variant and return the best checkpoint path."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=patience),
            checkpoint,
        ],
        default_root_dir=output,
        logger=False,
    )
    trainer.fit(model, train_loader, validation_loader)
    if not checkpoint.best_model_path:
        raise RuntimeError("Training completed without a best checkpoint")
    return Path(checkpoint.best_model_path)


def load_checkpoint(model: pl.LightningModule, checkpoint: str | Path) -> None:
    """Load weights from a Lightning checkpoint into an already-built model."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = payload.get("state_dict", payload)
    model.load_state_dict(state_dict)


@torch.no_grad()
def evaluate_model(model: pl.LightningModule, loader: DataLoader) -> dict[str, float]:
    """Evaluate DR grading and labeled lesion concepts on CPU or model device."""
    model.eval()
    device = next(model.parameters()).device
    disease_metrics, lesion_metrics = configure_metrics()
    disease_metrics.to(device)
    lesion_metrics.to(device)
    saw_concept_labels = False

    for batch in loader:
        images = batch["img"].to(device)
        disease_targets = batch["drgrading_level"].to(device)
        concept_targets = batch["lesion_label"].to(device)
        concept_labeled = batch["l"].to(device).bool()
        _, concept_probabilities, _, disease_logits = model(images)
        update_metrics(
            disease_metrics,
            lesion_metrics,
            disease_logits,
            disease_targets,
            concept_probabilities,
            concept_targets,
            concept_labeled,
        )
        saw_concept_labels = saw_concept_labels or bool(concept_labeled.any())

    results = disease_metrics.compute()
    if saw_concept_labels:
        results.update(lesion_metrics.compute())
    return {name: float(value.detach().cpu()) for name, value in results.items()}


def write_metrics(metrics: dict[str, float], output: str | Path | None) -> None:
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if output is None:
        print(text)
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{text}\n", encoding="utf-8")
    print(path)
