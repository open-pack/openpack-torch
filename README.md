# openpack-torch

[![Test](https://github.com/open-pack/openpack-torch/actions/workflows/test.yaml/badge.svg)](https://github.com/open-pack/openpack-torch/actions/workflows/test.yaml)
[![GitHub Pages](https://github.com/open-pack/openpack-torch/actions/workflows/deploy-docs.yaml/badge.svg)](https://github.com/open-pack/openpack-torch/actions/workflows/deploy-docs.yaml)

PyTorch utilities to work around with [OpenPack Dataset](https://open-pack.github.io/).

## Setup

You can install via pip with the following command.

```bash
# Pip
pip install openpack-torch

# Poetry
poetry add  openpack-torch
```

## Docs

- [Dataset Page](https://open-pack.github.io/)
- [API Docs](https://open-pack.github.io/openpack-torch/openpack_torch)

## Examples

### Operation Recognition (Semantic Segmentation)

#### IMU

- Acceleration
  - [U-Net](./examples/unet/)
  - [DeepConvLSTM](./examples/deep-conv-lstm/)

#### Vision

- Keypoints
  - [ST-GCN](./examples/st-gcn)

## LICENCE

This software (openpack-torch) is distributed under [MIT Licence](./LICENSE).
For the license of "OpenPack Dataset", please check [this site (https://open-pack.github.io/)](https://open-pack.github.io/).
