"""DR grading and lesion-concept metrics used by the public evaluator."""

from __future__ import annotations

import torch
from torchmetrics import AUROC, Accuracy, CohenKappa, F1Score, MetricCollection


def configure_metrics(
    num_disease: int = 5,
    num_lesion: int = 4,
) -> tuple[MetricCollection, MetricCollection]:
    """Create the DR and lesion metric families reported by the evaluator."""
    disease = MetricCollection(
        {
            "kappa": CohenKappa(
                task="multiclass", num_classes=num_disease, weights="quadratic"
            ),
            "auc": AUROC(task="multiclass", num_classes=num_disease, average="macro"),
            "accuracy": Accuracy(task="multiclass", num_classes=num_disease),
            "f1": F1Score(task="multiclass", num_classes=num_disease, average="macro"),
        },
        prefix="dr_",
    )
    lesion = MetricCollection(
        {
            "auc": AUROC(task="multilabel", num_labels=num_lesion, average="macro"),
            "accuracy": Accuracy(task="multilabel", num_labels=num_lesion),
            "f1": F1Score(task="multilabel", num_labels=num_lesion, average="micro"),
        },
        prefix="lesion_",
    )
    return disease, lesion


def update_metrics(
    disease_metrics: MetricCollection,
    lesion_metrics: MetricCollection,
    disease_logits: torch.Tensor,
    disease_targets: torch.Tensor,
    concept_probabilities: torch.Tensor,
    concept_targets: torch.Tensor,
    concept_labeled: torch.Tensor,
) -> None:
    disease_metrics.update(disease_logits.softmax(dim=1), disease_targets.long())
    if concept_labeled.any():
        lesion_metrics.update(
            concept_probabilities[concept_labeled],
            concept_targets[concept_labeled].long(),
        )
