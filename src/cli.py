"""Command-line interface for the official IDR-SSCL implementation."""

from __future__ import annotations

import argparse


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official IDR-SSCL implementation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    construct = subparsers.add_parser("construct", help="Construct a model offline")
    _add_model_arguments(construct)

    train = subparsers.add_parser("train", help="Train on explicit local manifests")
    _add_model_arguments(train)
    train.add_argument("--train-manifest", required=True)
    train.add_argument("--val-manifest", required=True)
    train.add_argument("--test-manifest", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--accelerator", default="cpu")
    train.add_argument("--devices", default="1")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a local checkpoint")
    _add_model_arguments(evaluate)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--metrics-output")
    return parser


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Release YAML config")
    parser.add_argument(
        "--imagenet-weights",
        action="store_true",
        help="Allow torchvision to request ImageNet weights",
    )


def _parse_devices(value: str):
    return int(value) if value.isdigit() else value


def main() -> None:
    args = make_parser().parse_args()
    from src.config import load_config

    config = load_config(args.config)
    from src.models.build import build_idr_sscl_model

    model = build_idr_sscl_model(
        config,
        use_imagenet_weights=args.imagenet_weights,
    )
    if args.command == "construct":
        print(type(model).__name__)
        return

    from src.data_interface import build_loader
    from src.train.runner import evaluate_model, fit_model, load_checkpoint, write_metrics

    batch_size = int(config["dataset_config"]["batch_size"])
    num_workers = int(config["dataset_config"].get("num_workers", 0))
    if args.command == "train":
        train_loader = build_loader(
            args.train_manifest,
            batch_size=batch_size,
            num_workers=num_workers,
            training=True,
        )
        validation_loader = build_loader(
            args.val_manifest,
            batch_size=batch_size,
            num_workers=num_workers,
            training=False,
        )
        test_loader = build_loader(
            args.test_manifest,
            batch_size=batch_size,
            num_workers=num_workers,
            training=False,
        )
        checkpoint = fit_model(
            model,
            train_loader,
            validation_loader,
            output_dir=args.output_dir,
            max_epochs=int(config["max_epochs"]),
            patience=int(config.get("patience", 10)),
            accelerator=args.accelerator,
            devices=_parse_devices(args.devices),
        )
        load_checkpoint(model, checkpoint)
        metrics = evaluate_model(model, test_loader)
        write_metrics(metrics, f"{args.output_dir}/test_metrics.json")
        print(checkpoint)
        return

    load_checkpoint(model, args.checkpoint)
    loader = build_loader(
        args.manifest,
        batch_size=batch_size,
        num_workers=num_workers,
        training=False,
    )
    write_metrics(evaluate_model(model, loader), args.metrics_output)


if __name__ == "__main__":
    main()
