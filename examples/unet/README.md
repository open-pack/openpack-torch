# U-Net

## Notebooks

### [U-Net_Train-Model-and-Make-Submission-File.ipynb](./notebooks/U-Net_Train-Model-and-Make-Submission-File.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-pack/openpack-torch/blob/main/examples/unet/notebooks/U-Net_Train-Model-and-Make-Submission-File.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-pack/openpack-torch/blob/main/examples/unet/notebooks/U-Net_Train-Model-and-Make-Submission-File__JA.ipynb) (日本語版)

This is a tutorial of OpenPack Challenge 2022.
In this notebook, we will build the U-Net, one of the basic architecture for segmentation, with the acceleration data from the left wrist (atr01).
This notebook also shows how to make submission file (submission.json) for OpenPack Challenge 2022.
In fact, the performance of this U-Net is poor. Please enjoy finding a better architecture and parameters.

## Script

```bash
# Training
$ python main.py mode=train debug=false

# Test
$ python main.py mode=test debug=false

# Make submission zip file
$ python main.py mode=submission debug=false
```

## Reference

- [U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
- [Human Activity Recognition Based on Motion Sensor Using U-Net (IEEE Access 2019)](https://ieeexplore.ieee.org/abstract/document/8731875)
- [Conditional-UNet: A Condition-aware Deep Model for Coherent Human Activity Recognition From Wearables (ICPR 2021)](https://ieeexplore.ieee.org/abstract/document/9412851/?casa_token=4IwIRlUNcQgAAAAA:dl_v2_W_KAuSwRAYsE4J_OLQ3jV_HsiTWg90vFKBayAk5ik8hM6FpUi037sX0khKYt0uVdg5Tg)
