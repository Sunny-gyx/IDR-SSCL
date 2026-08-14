"""Minimal model construction for the two IDR-SSCL integrations."""

from __future__ import annotations

from typing import Any

import torch
from torchvision.models import resnet50

from src.models.idr_sscl_cbm import Ours_CBM
from src.models.idr_sscl_cem import Ours_CEM
from src.train.utils import wrap_pretrained_model


def build_idr_sscl_model(
    config: dict[str, Any],
    *,
    n_concepts: int = 4,
    n_tasks: int = 5,
    use_imagenet_weights: bool = False,
    task_class_weights: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
) -> torch.nn.Module:
    """Build only the CEM or CBM form of the current IDR-SSCL method.

    This extraction deliberately does not expose external comparison models.
    ImageNet weights are opt-in so ordinary local construction cannot trigger
    an unexpected network request. This default is release hygiene, not a
    statement about the paper training protocol.
    """

    architecture = config["architecture"]
    backbone = wrap_pretrained_model(
        resnet50,
        pretrain_model=use_imagenet_weights,
    )
    common = {
        "n_concepts": n_concepts,
        "n_tasks": n_tasks,
        "concept_loss_weight": config.get("concept_loss_weight", 1.0),
        "concept_loss_weight_labeled": config.get(
            "concept_loss_weight_labeled", 1.0
        ),
        "concept_loss_weight_unlabeled": config.get(
            "concept_loss_weight_unlabeled", 0.1
        ),
        "task_loss_weight": config.get("task_loss_weight", 1.0),
        "optimizer": config.get("optimizer", "adamw"),
        "momentum": config.get("momentum", 0.9),
        "learning_rate": float(config.get("learning_rate", 1e-4)),
        "weight_decay": float(config.get("weight_decay", 5e-4)),
        "task_class_weights": task_class_weights,
        "top_k_accuracy": config.get("top_k_accuracy"),
        "pos_weight": pos_weight,
        "k": int(config.get("k", 3)),
        "c_extractor_arch": backbone,
    }

    if architecture == "Ours_CEM":
        return Ours_CEM(
            emb_size=int(config.get("emb_size", 32)),
            training_intervention_prob=float(
                config.get("training_intervention_prob", 0)
            ),
            embedding_activation=config.get("embedding_activation", "leakyrelu"),
            shared_prob_gen=bool(config.get("shared_prob_gen", True)),
            **common,
        )
    if architecture == "Ours_CBM":
        return Ours_CBM(
            extra_dims=int(config.get("extra_dims", 0)),
            bool=bool(config.get("bool", False)),
            sigmoidal_prob=bool(config.get("sigmoidal_prob", True)),
            sigmoidal_extra_capacity=bool(
                config.get("sigmoidal_extra_capacity", True)
            ),
            bottleneck_nonlinear=config.get("bottleneck_nonlinear"),
            **common,
        )
    raise ValueError(
        f"Unsupported architecture {architecture!r}; expected 'Ours_CEM' or 'Ours_CBM'"
    )
