# Generalizable Method for Face Anti-Spoofing with Semi-Supervised Learning

**Nikolay Sergievskiy, Roman Vlasov, Roman Trusov**

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalizable-method-for-face-anti-spoofing/face-anti-spoofing-on-msu-mfsd)](https://paperswithcode.com/sota/face-anti-spoofing-on-msu-mfsd?p=generalizable-method-for-face-anti-spoofing)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalizable-method-for-face-anti-spoofing/face-anti-spoofing-on-replay-attack)](https://paperswithcode.com/sota/face-anti-spoofing-on-replay-attack?p=generalizable-method-for-face-anti-spoofing)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalizable-method-for-face-anti-spoofing/face-anti-spoofing-on-oulu-npu)](https://paperswithcode.com/sota/face-anti-spoofing-on-oulu-npu?p=generalizable-method-for-face-anti-spoofing)

This is the official implementation of the paper [Generalizable Method for Face Anti-Spoofing with Semi-Supervised Learning](https://arxiv.org/abs/2206.06510) by the ML research team from [Entry](https://getentry.com)

[arXiv](https://arxiv.org/abs/2206.06510)

## Dataset and models status

This work is done using a proprietary dataset, which is why we cannot share the data or pretrained models.

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
