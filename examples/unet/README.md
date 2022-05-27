# U-Net

## Notebooks

### [GETTING_STARTED__U-Net.ipynb](./notebooks/GETTING_STARTED__U-Net.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-pack/openpack-torch-dev/blob/main/examples/unet/notebooks/GETTING_STARTED__U-Net.ipynb)

In this notebook, you can train and test the U-Net with `openpack_torch` package.
Also, you can learn the basic usage of (1) pytorch-lightning's `LightningDataModule`, and (2) `LightinigModule` supported by `openpack_torch`.

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

TBA
