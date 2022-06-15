# Generalizable Method for Face Anti-Spoofing with Semi-Supervised Learning

**Nikolay Sergievskiy, Roman Vlasov, Roman Trusov**

This is the official implementation of the paper [Generalizable Method for Face Anti-Spoofing with Semi-Supervised Learning](https://arxiv.org/abs/2206.06510) by the ML research team from [Entry](https://getentry.com)

[arXiv](https://arxiv.org/abs/2206.06510)

## Requirements

- [VISSL](https://github.com/facebookresearch/vissl)
- [OpenCV](https://opencv.org/), PIL
- [timm](https://github.com/rwightman/pytorch-image-models)
- tqdm

### WandB

We use [WandB](https://wandb.ai) for model comparison and monitoring, and the training/validation script relies on it heavily, so by default you will need an account to export results there.

## Structure

- `code/train_config.yaml` - main configuration for training/eval
- `code/run.py` - training/validation script with CLI
- `code/dataset` - package with utilities for loading and augmenting data
- `code/entry_antispoof` - package with utils, loss functions, and network definitions

## Contacts

[Roman Trusov](roman@getentry.com)