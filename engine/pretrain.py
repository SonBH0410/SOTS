"""Self-supervised pretraining driver: dataset -> encoder -> SSL model -> fit -> save encoder weights."""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import gpu
from ..data.common import build_image_array
from ..data.registry import load_dataset
from ..paths import run_dir, timestamped_run_name
from ..ssl.registry import (
    build_optimizer,
    build_ssl_dataset,
    build_ssl_model,
    default_hyperparams,
    get_pretrained_encoder,
    resolve_base_lr,
    resolve_warmup_epochs,
)
from ..ssl.schedules import build_warmup_cosine


def run_pretrain(args: argparse.Namespace) -> None:
    gpu.setup_gpu(args.gpu_index)
    gpu.set_seed(args.seed)

    hp = default_hyperparams(args.method)
    batch_size = args.batch_size or hp["batch_size"]
    epochs = args.epochs or hp["epochs"]
    accum_steps = args.accum_steps if args.accum_steps is not None else hp["accum_steps"]

    print(f"Loading '{args.dataset}' from {args.data_root} ...")
    splits = load_dataset(args.dataset, args.data_root, annotator=args.annotator, variant=args.variant)
    images = build_image_array(splits.image_train, args.img_size, desc="Loading train images")
    print(f"{len(images)} training images loaded for self-supervised pretraining.")

    dataset = build_ssl_dataset(args.method, images, batch_size, args.img_size, args.seed)
    steps_per_epoch = max(1, len(images) // batch_size)

    base_lr = resolve_base_lr(args.method, hp, args.lr, batch_size)
    total_steps = steps_per_epoch * epochs

    if hp["use_schedule"]:
        warmup_epochs = resolve_warmup_epochs(hp, epochs)
        learning_rate = build_warmup_cosine(
            steps_per_epoch, epochs, base_lr, warmup_epoch_fraction=warmup_epochs / max(1, epochs)
        )
    else:
        learning_rate = base_lr

    ssl_model = build_ssl_model(
        args.method,
        input_shape=(args.img_size, args.img_size, 3),
        accum_steps=accum_steps,
        project_dim=args.project_dim,
        lambd=args.lambd,
        momentum=args.momentum,
        temperature=args.temperature,
        queue_size=args.queue_size,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
    )
    optimizer = build_optimizer(args.method, learning_rate)
    ssl_model.compile(optimizer=optimizer, run_eagerly=hp["run_eagerly"])

    out_dir = run_dir(args.output_dir, args.dataset, args.method)
    run_name = timestamped_run_name(args.run_name)
    print(f"Pretraining {args.method} on {args.dataset}: batch_size={batch_size} epochs={epochs} "
          f"base_lr={base_lr:.6g} accum_steps={accum_steps} -> {out_dir}")

    history = ssl_model.fit(dataset, epochs=epochs, verbose=1)

    encoder = get_pretrained_encoder(ssl_model, args.method)
    encoder_path = out_dir / "encoder.weights.h5"
    encoder.save_weights(str(encoder_path))
    print(f"Saved pretrained encoder weights to {encoder_path}")

    loss_history = history.history.get("loss", [])
    if loss_history:
        plt.figure()
        plt.plot(loss_history)
        plt.grid(True)
        plt.title(f"{args.method} pretraining loss ({args.dataset})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plot_path = out_dir / "pretrain_loss.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved loss curve to {plot_path}")

    print(f"[{run_name}] Pretraining complete.")
