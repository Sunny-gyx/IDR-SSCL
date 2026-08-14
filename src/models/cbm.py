"""Shared Lightning lifecycle for the official IDR-SSCL CEM and CBM models."""

from __future__ import annotations

import logging

import pytorch_lightning as pl
import torch


class CBM_SSL(pl.LightningModule):
    """Common training, validation, prediction, and optimizer behavior.

    The CEM and CBM subclasses implement their own architectures and losses.
    This base class intentionally contains only lifecycle code shared by both
    official IDR-SSCL variants.
    """

    @staticmethod
    def _unpack_batch(batch):
        """Extract the tensors used by both official models from a batch dict."""
        return (
            batch["img"],
            batch["drgrading_level"],
            batch["lesion_label"],
            batch["l"],
            batch["nbr_concepts"],
            batch["nbr_weight"],
        )

    def forward(
        self,
        x,
        c=None,
        y=None,
        l=None,
        latent=None,
        intervention_idxs=None,
        competencies=None,
        prev_interventions=None,
        output_embeddings=False,
        output_latent=None,
        output_interventions=None,
    ):
        return self._forward(
            x,
            train=False,
            c=c,
            y=y,
            l=l,
            competencies=competencies,
            prev_interventions=prev_interventions,
            intervention_idxs=intervention_idxs,
            latent=latent,
            output_embeddings=output_embeddings,
            output_latent=output_latent,
            output_interventions=output_interventions,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        del batch_idx, dataloader_idx
        x, y, c, l, _, _ = self._unpack_batch(batch)
        return self._forward(x, c=c, y=y, l=l, train=False)

    def training_step(self, batch, batch_idx):
        _, result = self._run_step(batch, batch_idx, train=True)
        self._log_losses(result, prefix="")
        return result

    def validation_step(self, batch, batch_idx):
        _, result = self._run_step(batch, batch_idx, train=False)
        self._log_losses(result, prefix="val_")
        return {
            key if key in {"c_sem", "c", "y", "y_pred"} else f"val_{key}": value
            for key, value in result.items()
        }

    def test_step(self, batch, batch_idx):
        _, result = self._run_step(batch, batch_idx, train=False)
        self._log_losses(result, prefix="test_")
        return result

    def _log_losses(self, result, *, prefix):
        for name, value in result.items():
            if name in {"c_sem", "y_pred", "c", "y"}:
                continue
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                value = value.detach().item()
            self.log(f"{prefix}{name}", value, prog_bar=True, logger=True)

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adamw":
            logging.info(
                "Using AdamW with learning rate %s and weight decay %s",
                self.learning_rate,
                self.weight_decay,
            )
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )
        else:
            optimizer = torch.optim.SGD(
                (parameter for parameter in self.parameters() if parameter.requires_grad),
                lr=float(self.learning_rate),
                momentum=self.momentum,
                weight_decay=float(self.weight_decay),
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=0,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
