"""SOTS CLI: run self-supervised pretraining or supervised (fine-tune / baseline) segmentation
training for any of the 3 datasets x 5 SSL methods (+ no-SSL baseline) covered by the notebooks
this package was ported from.

Examples
--------
Pretrain a Barlow Twins encoder on ISIC2018:
    python main.py --dataset isic2018 --mode selfsupervised --method barlow_twins

Fine-tune a segmentation model from that encoder on 10% of the labels:
    python main.py --dataset isic2018 --mode supervised --method barlow_twins \\
        --label-ratio 0.1 --encoder-weights outputs/isic2018/barlow_twins/encoder.weights.h5

Train the no-SSL (ImageNet-only) baseline on the full training set:
    python main.py --dataset isic2018 --mode supervised --method none --label-ratio 1.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sots.paths import default_data_root
from sots.ssl.method_names import METHODS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--dataset", required=True, choices=["isic2018", "otu2d", "usova3d"])
    parser.add_argument("--mode", required=True, choices=["selfsupervised", "supervised"])
    parser.add_argument(
        "--method", required=True, choices=[*METHODS, "none"],
        help="SSL method to use. 'none' is only valid with --mode supervised (ImageNet-only baseline).",
    )

    sup = parser.add_argument_group("supervised mode")
    sup.add_argument("--label-ratio", type=float, default=1.0, choices=[0.1, 0.2, 0.3, 1.0],
                      help="Fraction of the labeled training set to fine-tune on (supervised mode only).")
    sup.add_argument("--encoder-weights", type=Path, default=None,
                      help="Path to a pretrained SSL encoder checkpoint (required unless --method none).")

    data = parser.add_argument_group("data")
    data.add_argument("--data-root", type=Path, default=None, help="Override the dataset's default root path.")
    data.add_argument("--annotator", choices=["follicle_r1", "follicle_r2", "ovary_r1", "ovary_r2"], default="ovary_r2",
                       help="USOVA3D only.")
    data.add_argument("--variant", choices=["binary", "color", "instance"], default="binary", help="USOVA3D only.")
    data.add_argument("--img-size", type=int, default=256)

    hp = parser.add_argument_group("training hyperparameters (default: per-method value from ssl/registry.py)")
    hp.add_argument("--batch-size", type=int, default=None)
    hp.add_argument("--epochs", type=int, default=None)
    hp.add_argument("--lr", type=float, default=None)
    hp.add_argument("--weight-decay", type=float, default=None)
    hp.add_argument("--accum-steps", type=int, default=None,
                     help="Gradient accumulation steps. Default: method-specific (e.g. 4 for barlow_twins, 1 otherwise).")

    ssl_hp = parser.add_argument_group("SSL-method-specific hyperparameters (selfsupervised mode)")
    ssl_hp.add_argument("--project-dim", type=int, default=None, help="Projector output dimension.")
    ssl_hp.add_argument("--temperature", type=float, default=None, help="simclr / moco")
    ssl_hp.add_argument("--momentum", type=float, default=None, help="byol / moco EMA momentum")
    ssl_hp.add_argument("--queue-size", type=int, default=None, help="moco")
    ssl_hp.add_argument("--lambd", type=float, default=None, help="barlow_twins redundancy-reduction weight")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--gpu-index", type=int, default=0)
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--output-dir", type=Path, default=Path("outputs"))
    runtime.add_argument("--run-name", type=str, default=None)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.method == "none" and args.mode != "supervised":
        raise SystemExit("--method none is only valid with --mode supervised (it's the no-SSL baseline).")
    if args.mode == "supervised" and args.method != "none" and args.encoder_weights is None:
        raise SystemExit("--encoder-weights is required for --mode supervised unless --method none.")
    if args.mode == "selfsupervised" and args.encoder_weights is not None:
        raise SystemExit("--encoder-weights is only used in --mode supervised.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    if args.data_root is None:
        args.data_root = Path(default_data_root(args.dataset))

    if args.mode == "selfsupervised":
        from sots.engine.pretrain import run_pretrain
        run_pretrain(args)
    else:
        from sots.engine.finetune import run_supervised
        run_supervised(args)


if __name__ == "__main__":
    main()
