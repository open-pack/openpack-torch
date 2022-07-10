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
- [PyPI - openpack-torch](https://pypi.org/project/openpack-torch/)

## Examples

### Operation Recognition (Semantic Segmentation)

#### IMU

- Acceleration
  - [U-Net](./examples/unet/)
  - [DeepConvLSTM](./examples/deep-conv-lstm/)

#### Vision

- Keypoints
  - [ST-GCN](./examples/st-gcn)

#### Scores of Baseline Moodel (Preliminary Experiments)

##### Split: Pilot Challenge

| Model                    | F1 (Test Set) | F1 (Submission Set) | Date       | Code |
|--------------------------|---------------|---------------------|------------|------|
| UNet                     | 0.3451        | 0.3747              | 2022-06-28 | [main.py](./examples/unet/main.py) |
| DeepConvLSTM             | 0.7081        | 0.7695              | 2022-06-28 | [main.py](./examples/deep-conv-lstm/main.py) |
| ST-GCN                   | 0.7024        | 0.6106              | 2022-07-07 | [main.py](./examples/st-gcn/main.py) |

NOTE: F1 = F1-measure (macro average)

## LICENCE

This software (openpack-torch) is distributed under [MIT Licence](./LICENSE).
For the license of "OpenPack Dataset", please check [this site (https://open-pack.github.io/)](https://open-pack.github.io/).
