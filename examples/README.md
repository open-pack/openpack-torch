# Baseline Models (Train/Test Script Samples)

## [1] Deep Conv LSTM

### Python Script

```bash
# Train
python run_dcl.py mode=train debug=false

# Test (Test Set)
python run_dcl.py mode=test debug=false

# Test (Submission Set)
python run_dcl.py mode=test-on-submission debug=false
```

### Tutorial: Modeling of Time-series

#### (English Ver.) [Tutorial_Basics_of_Modeling.ipynb](./notebooks/Tutorial_Basics_of_Modeling.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-pack/openpack-torch/blob/main/examples/deep-conv-lstm/notebooks/Tutorial_Basics_of_Modeling.ipynb)

This notebook was created as a tutorial for participants of the OpenPack Challenge 2022, a work activity recognition competition to be held at the Percom 2023 Workshop BiRD.
This tutorial will (1) provide an overview of OpenPack dataset and its tasks, and (2) run a sample program of an activity recognition model.
Let's learn how to capture the characteristics of OpenPack dataset data and learn the concept of modeling to recognize the work activities of packaging.

#### (日本語版) [Tutorial_Basics_of_Modeling\_\_JA.ipynb](./notebooks/Tutorial_Basics_of_Modeling__JA.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-pack/openpack-torch/blob/main/examples/deep-conv-lstm/notebooks/Tutorial_Basics_of_Modeling__JA.ipynb)

このノートブックは， Percom 2023 Workshop BiRD で開催される行動認識コンペティション 「OpenPack Challenge 2022」の参加者用チュートリアルとして作成しました．
このチュートリアルでは，(1) OpenPack の概要とタスクの説明，(2) 行動認識モデルのサンプルプログラムの実行を行います．
OpenPack データセットのデータの性質を捉え， 梱包作業という作業行動を認識するためのモデリングの考え方を学んで行きましょう．

### Reference

- [Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition (IEEE Sensors, 2016)](https://www.mdpi.com/1424-8220/16/1/115)

## [2] U-Net

### Notebooks

#### [U-Net_Train-Model-and-Make-Submission-File.ipynb](./notebooks/U-Net_Train-Model-and-Make-Submission-File.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-pack/openpack-torch/blob/main/examples/unet/notebooks/U-Net_Train-Model-and-Make-Submission-File.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-pack/openpack-torch/blob/main/examples/unet/notebooks/U-Net_Train-Model-and-Make-Submission-File__JA.ipynb) (日本語版)

This is a tutorial of OpenPack Challenge 2022.
In this notebook, we will build the U-Net, one of the basic architecture for segmentation, with the acceleration data from the left wrist (atr01).
This notebook also shows how to make submission file (submission.json) for OpenPack Challenge 2022.
In fact, the performance of this U-Net is poor. Please enjoy finding a better architecture and parameters.

### Script

```bash
# Train
python run_unet.py mode=train debug=false

# Test (Test Set)
python run_unet.py mode=test debug=false

# Test (Submission Set)
python run_unet.py mode=test-on-submission debug=false
```

### Reference

- [U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
- [Human Activity Recognition Based on Motion Sensor Using U-Net (IEEE Access 2019)](https://ieeexplore.ieee.org/abstract/document/8731875)
- [Conditional-UNet: A Condition-aware Deep Model for Coherent Human Activity Recognition From Wearables (ICPR 2021)](https://ieeexplore.ieee.org/abstract/document/9412851/?casa_token=4IwIRlUNcQgAAAAA:dl_v2_W_KAuSwRAYsE4J_OLQ3jV_HsiTWg90vFKBayAk5ik8hM6FpUi037sX0khKYt0uVdg5Tg)

## [3] ST-GCN

### Script

```bash
# Train
python run_stgcn.py mode=train debug=false

# Test (Test Set)
python run_stgcn.py mode=test debug=false

# Test (Submission Set)
python run_stgcn.py mode=test-on-submission debug=false
```

### Reference

- [Paper - AAAI2018 - Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455)
- [GitHub - machine-perception-robotics-group/MPRGDeepLearningLectureNotebook](https://github.com/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook)
  - [Colab - 03_action_recognition_ST_GCN.ipynb](https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/15_gcn/03_action_recognition_ST_GCN.ipynb)
- [GitHub - open-mmlab/mmskeleton](https://github.com/open-mmlab/mmskeleton/blob/master/doc/START_RECOGNITION.md)
