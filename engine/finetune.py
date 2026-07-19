"""Supervised driver: build a segmentation model (SSL-pretrained encoder or ImageNet-only baseline),
fine-tune on a label-ratio subset of the training split, then evaluate on the held-out test split.
"""
from __future__ import annotations

import argparse

from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Nadam

from .. import gpu
from ..callbacks import build_callbacks
from ..data.common import build_arrays, get_label_subset
from ..data.registry import load_dataset
from ..losses import SimDice, dice_coef, hausdorff95, jacard_similarity
from ..models.encoders import build_encoder_for_method, build_segmentation_model
from ..paths import run_dir, timestamped_run_name


def _pct_str(ratio: float) -> str:
    return str(int(round(ratio * 100)))


def run_supervised(args: argparse.Namespace) -> None:
    gpu.setup_gpu(args.gpu_index)
    gpu.set_seed(args.seed)

    print(f"Loading '{args.dataset}' from {args.data_root} ...")
    splits = load_dataset(args.dataset, args.data_root, annotator=args.annotator, variant=args.variant)

    train_images, train_masks = get_label_subset(splits.image_train, splits.mask_train, args.label_ratio)
    print(f"Fine-tuning on {len(train_images)}/{len(splits.image_train)} labeled training images "
          f"(label_ratio={args.label_ratio}).")

    x_train, y_train = build_arrays(train_images, train_masks, args.img_size, desc="Loading train")
    x_valid, y_valid = build_arrays(splits.image_valid, splits.mask_valid, args.img_size, desc="Loading valid")
    x_test, y_test = build_arrays(splits.image_test, splits.mask_test, args.img_size, desc="Loading test")

    input_shape = (args.img_size, args.img_size, 3)
    if args.method == "none":
        encoder = build_encoder_for_method("none", input_shape)
        print("Baseline mode (--method none): ImageNet-only VGG16 weights, no SSL pretraining.")
    else:
        encoder = build_encoder_for_method(
            args.method, input_shape, project_dim=args.project_dim, weight_decay=args.weight_decay
        )
        if not args.encoder_weights:
            raise ValueError("--encoder-weights is required in supervised mode unless --method none")
        encoder.load_weights(str(args.encoder_weights))
        print(f"Loaded pretrained '{args.method}' encoder weights from {args.encoder_weights}")

    seg_model = build_segmentation_model(encoder, num_classes=1)

    lr = args.lr or 1e-4
    batch_size = args.batch_size or 4
    epochs = args.epochs or 200

    seg_model.compile(
        loss=SimDice,
        optimizer=Nadam(learning_rate=lr),
        metrics=[dice_coef, jacard_similarity, Precision(), Recall()],
    )

    out_dir = run_dir(args.output_dir, args.dataset, args.method)
    run_name = timestamped_run_name(args.run_name)
    pct = _pct_str(args.label_ratio)
    best_path = out_dir / f"seg_{pct}pct_best.weights.h5"
    final_path = out_dir / f"seg_{pct}pct_final.weights.h5"

    callbacks = build_callbacks(out_dir, best_path, f"{run_name}_seg_{pct}pct")

    print(f"Training segmentation model: batch_size={batch_size} epochs={epochs} lr={lr:.6g} -> {out_dir}")
    seg_model.fit(
        x_train, y_train,
        validation_data=(x_valid, y_valid),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    seg_model.save_weights(str(final_path))
    print(f"Saved final weights to {final_path}")

    eval_metrics = [dice_coef, jacard_similarity, Precision(), Recall(), hausdorff95]
    seg_model.compile(loss=SimDice, optimizer=Nadam(learning_rate=lr), metrics=eval_metrics)

    for label, path in (("final", final_path), ("best", best_path)):
        if not path.exists():
            continue
        seg_model.load_weights(str(path))
        results = seg_model.evaluate(x_test, y_test, batch_size=8, verbose=1)
        print(f"[{label}] test results ({dict(zip(seg_model.metrics_names, results))})")

    print(f"[{run_name}] Supervised training complete.")
