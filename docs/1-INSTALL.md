# KDPL Installation

This guide provides step-by-step instructions to set up the [KDPL](https://github.com/miccunifi/KDPL) environment and install all necessary dependencies. The codebase has been tested on **Ubuntu 20.04.2 LTS** with **Python 3.8**.

## 1. Setting Up Conda Environment

It is recommended to use a Conda environment for this setup.

1. **Create a Conda Environment**
    ```bash
    conda create -y -n kdpl python=3.8
    ```

2. **Activate the Environment**
    ```bash
    conda activate kdpl
    ```

## 2. Installing PyTorch

Ensure you have the correct version of PyTorch and torchvision. If you need a different CUDA version, please refer to the [official PyTorch website](https://pytorch.org/).

1. **Install PyTorch, torchvision, and torchaudio**
    ```bash
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

## 3. Cloning KDPL and Installing Requirements

Follow these steps to clone and install the [Dassl library](https://github.com/KaiyangZhou/Dassl.pytorch).

1. **Clone the KDPL Code Repository**
    ```bash
    git clone https://github.com/miccunifi/KDPL
    cd KDPL/
    ```
    
2. **Clone the Dassl Repository**
    ```bash
    git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
    ```

3. **Install Dassl Dependencies**
    ```bash
    cd Dassl.pytorch/
    pip install -r requirements.txt
    ```

4. **Install Dassl Library**
    ```bash
    python setup.py develop
    ```

5. **Install KDPL Dependencies**
    ```bash
    cd ..
    pip install -r requirements.txt
    pip install setuptools==59.5.0
    ```
---

## 4. Dataset Installation

To set up the datasets for KDPL, we follow the standard preparation methods outlined by CoOp. For detailed instructions, refer to the [CoOp Dataset Preparation Guide](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).
