# SOTS: Self-supervised pretraining for label-efficient Ovarian Tumor Segmentation

Official code for:

> **SOTS: Leveraging Self-supervised Pretraining for Label-Efficient Ovarian Tumor Segmentation in Ultrasound Images**
> Hoang-Son Bui, Thanh-Phuc Dao, Thi-Lan Le
> *Journal of Imaging Informatics in Medicine*, 2026
> DOI: [10.1007/s10278-026-02138-0](https://doi.org/10.1007/s10278-026-02138-0)
> Received 30 December 2025 · Revised 6 July 2026 · Accepted 7 July 2026
> © The Author(s), under exclusive licence to the Society for Imaging Informatics in Medicine 2026

This repository implements a VGG16-based segmentation pipeline where the encoder is first
pretrained with a self-supervised learning (SSL) method on unlabeled ultrasound images, then
fine-tuned for ovarian tumor segmentation using only a fraction of the labeled data. It supports
5 SSL methods and 3 datasets, plus a no-SSL (ImageNet-only) baseline for comparison.

## Overview

Manual annotation of ovarian tumor ultrasound images is costly, which limits the size of labeled
datasets available for supervised segmentation. SOTS addresses this by pretraining the
segmentation encoder with self-supervised contrastive/non-contrastive learning on unlabeled
images, then fine-tuning the full segmentation model on a small labeled subset. The pipeline
compares five SSL pretraining strategies against an ImageNet-only baseline across three datasets
and four labeled-data ratios (10%, 20%, 30%, 100%).

## Supported SSL methods

| Method | Reference |
| --- | --- |
| Barlow Twins | `ssl/barlow_twins.py` |
| BYOL | `ssl/byol.py` |
| MoCo (v2) | `ssl/moco.py` |
| SimCLR | `ssl/simclr.py` |
| SimSiam | `ssl/simsiam.py` |

Each method has its own default hyperparameters, augmentation pipeline, and optimizer/schedule
configuration in `ssl/registry.py`. Pass `--method none` in supervised mode to train the
ImageNet-only baseline instead.

## Supported datasets

| Dataset | Notes |
| --- | --- |
| `isic2018` | ISIC 2018 skin lesion segmentation (used as an external benchmark) |
| `otu2d` | 2D ovarian tumor ultrasound segmentation |
| `usova3d` | USOVA3D ovarian ultrasound dataset (`--annotator`, `--variant` options) |

Dataset loaders live in `data/`; default root paths are declared in `paths.py` and can be
overridden with `--data-root`.

## Repository structure

```
SOTS/
├── main.py              # CLI entry point
├── paths.py              # dataset roots / output-directory conventions
├── gpu.py                 # GPU setup and seeding
├── losses.py              # segmentation losses & metrics (Dice, IoU, SimDice, HD95, ...)
├── callbacks.py            # Keras training callbacks
├── data/                  # dataset loaders (isic2018, otu2d, usova3d)
├── models/                 # VGG16 encoder branch, per-method encoders, decoder/segmentation head
├── ssl/                    # SSL methods, augmentations, LR schedules, method registry
└── engine/
    ├── pretrain.py          # self-supervised pretraining driver
    └── finetune.py           # supervised fine-tuning / baseline driver
```

## Installation

```bash
git clone https://github.com/SonBH0410/SOTS.git
cd SOTS
pip install -r requirements.txt
```

Requires Python with TensorFlow 2.12–2.15.

## Usage

### 1. Self-supervised pretraining

Pretrain an encoder on unlabeled images with one of the 5 SSL methods:

```bash
python main.py --dataset isic2018 --mode selfsupervised --method barlow_twins
```

This saves encoder weights and a pretraining loss curve to `outputs/<dataset>/<method>/`.

### 2. Supervised fine-tuning

Fine-tune a segmentation model from a pretrained encoder on a fraction of the labeled data:

```bash
python main.py --dataset isic2018 --mode supervised --method barlow_twins \
    --label-ratio 0.1 --encoder-weights outputs/isic2018/barlow_twins/encoder.weights.h5
```

### 3. No-SSL baseline

Train the ImageNet-only baseline (no SSL pretraining) on the full training set:

```bash
python main.py --dataset isic2018 --mode supervised --method none --label-ratio 1.0
```

Run `python main.py --help` for the full list of CLI options (dataset, augmentation,
hyperparameters, GPU index, output directory, etc.).

## Evaluation metrics

Segmentation is trained with a combined SSIM + Dice loss (`SimDice`) and evaluated with Dice
coefficient, Jaccard/IoU similarity, precision, recall, and 95th-percentile Hausdorff distance
(`losses.py`).

## Citation

If you use this code or find it useful in your research, please cite:

```bibtex
@article{bui2026sots,
  title   = {{SOTS}: Leveraging Self-supervised Pretraining for Label-Efficient Ovarian Tumor Segmentation in Ultrasound Images},
  author  = {Bui, Hoang-Son and Dao, Thanh-Phuc and Le, Thi-Lan},
  journal = {Journal of Imaging Informatics in Medicine},
  year    = {2026},
  doi     = {10.1007/s10278-026-02138-0},
  url     = {https://doi.org/10.1007/s10278-026-02138-0}
}
```

## License

© The Author(s), under exclusive licence to the Society for Imaging Informatics in Medicine 2026.
