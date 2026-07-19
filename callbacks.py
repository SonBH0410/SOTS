"""Callback stack for downstream segmentation training, ported from the shared "Callbacks" section.

The notebooks built a CSVLogger but left it out of the active callback list in every run -- included
here by default since there's no reason not to keep a plain-text training log alongside TensorBoard.
"""
from __future__ import annotations

from pathlib import Path

from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)


def build_callbacks(
    run_dir: Path,
    ckpt_best_path: Path,
    run_name: str,
    monitor: str = "val_loss",
    early_stopping_patience: int = 30,
    reduce_lr_patience: int = 10,
    reduce_lr_factor: float = 0.1,
    min_lr: float = 1e-8,
) -> list:
    run_dir = Path(run_dir)
    tb_log_dir = run_dir / "logs" / "fit" / run_name
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    return [
        TensorBoard(log_dir=str(tb_log_dir), histogram_freq=0),
        ModelCheckpoint(
            filepath=str(ckpt_best_path),
            monitor=monitor,
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        ),
        EarlyStopping(monitor=monitor, patience=early_stopping_patience, verbose=1, restore_best_weights=False),
        CSVLogger(filename=str(run_dir / f"{run_name}.log"), separator=",", append=False),
        ReduceLROnPlateau(monitor=monitor, factor=reduce_lr_factor, patience=reduce_lr_patience, min_lr=min_lr, verbose=1),
    ]
