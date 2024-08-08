# Reproducing KDPL Results

We provide bash scripts in the [scripts/](../scripts) directory for training and evaluating [KDPL](https://github.com/miccunifi/KDPL) and the corresponding baseline approaches. Ensure that you update the `DATA` variable with the dataset path in the scripts file and run the commands from the source directory `src/`.

Below, we provide training and evaluation instructions. Note that the same instructions apply for reproducing results for the baseline and the KDPL variants. However, when using KDPL variants, ensure you update the `CLASS_AGNOSTIC` variable to `True` or `False` in the scripts depending on whether you want to use the class-agnostic **KDPL-CA** or not.

## Domain Generalization and Cross-Dataset Transfer Settings

In the cross-dataset and domain generalization setting, we first train on 16-shots per class on ImageNet-1k for 3 seeds. Then, we evaluate the trained model directly on cross-datasets and out-of-distribution datasets. Below are the instructions to reproduce domain generalization and cross-datasets results.

### Training

First, we need to train the model. Suppose we want to train CoOp+KDPL; similarly, we can train all the KDPL variants and corresponding baselines. Run the command below to train CoOp+KDPL with 16-shots and **3 seeds** on ImageNet-1k:

```bash
# Train CoOp+KDPL 16-shots, 3 seeds on ImageNet-1k 
bash scripts/coop_kdpl/reproduce_cross_d_train.sh
```

### Evaluation

Now, use the evaluation script `scripts/coop_kdpl/reproduce_cross_d_test.sh` and run the command below to calculate the results for **3 seeds on all the domain generalization and cross-dataset datasets**:

```bash
# Evaluate CoOp+KDPL 16-shots, 3 seeds on Domain Generalization and Cross-Dataset Transfer settings
bash scripts/coop_kdpl/reproduce_cross_d_test.sh
```

Replace `coop_kdpl` with the corresponding baseline or KDPL variant you want to reproduce the results for:

- Use `coop_kdpl` for CoOp+KDPL, or `coop` for CoOp.
- Use `cocoop_kdpl` for CoCoOp+KDPL, or `cocoop` for CoCoOp.
- Use `vpt_kdpl` for VPT+KDPL, or `vpt` for VPT.
- Use `maple_kdpl` for MaPLe+KDPL, or `maple` for MaPLe.
- Use `promptsrc_kdpl` for PromptSRC+KDPL, or `promptsrc` for PromptSRC.

This script should evaluate and save log files in the `output/` directory.

## Generalization to Unseen Classes

In the Generalization to Unseen Classes setting, we first train with 16-shots on half of the classes for 3 seeds. Then, we evaluate the trained model directly on the unseen half of the classes on the test set of the same dataset.

### Training

We provide the instructions below to reproduce generalization to unseen results. Run the command below to train CoOp+KDPL with 16-shots and 3 seeds on each dataset:

```bash
# Train CoOp+KDPL 16-shots, 3 seeds, on half of the classes on each dataset 
bash scripts/coop_kdpl/reproduce_base2new_train.sh
```

### Evaluation

Now, use the evaluation script `scripts/coop_kdpl/reproduce_base2new_test.sh` and run the command below to calculate the results for **3 seeds on all generalization to unseen classes datasets**:

```bash
# Evaluate CoOp+KDPL 16-shots, 3 seeds on Generalization to Unseen Classes setting
bash scripts/coop_kdpl/reproduce_base2new_test.sh
```

Replace `coop_kdpl` with the corresponding baseline or KDPL variant you want to reproduce the results for:

- Use `coop_kdpl` for CoOp+KDPL, or `coop` for CoOp.
- Use `cocoop_kdpl` for CoCoOp+KDPL, or `cocoop` for CoCoOp.
- Use `vpt_kdpl` for VPT+KDPL, or `vpt` for VPT.
- Use `maple_kdpl` for MaPLe+KDPL, or `maple` for MaPLe.
- Use `promptsrc_kdpl` for PromptSRC+KDPL, or `promptsrc` for PromptSRC.

This script should evaluate and save log files in the `output/` directory.

## Averaging results over 3 seeds: 
Once the above trainings and evaluations are completed, the `output/` directory should have the following structure:

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– oxford_pets/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoOp_KDPL/
|   |   |   |   |   |–– vit_b32_ctxv1/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |   |-- other_datasets/ ...
|   |–– train_base/
|   |   |–– oxford_pets/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoOp_KDPL/
|   |   |   |   |   |–– vit_b32_ctxv1/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |   |-- other_datasets/ ...
|–– cross_domain_and_datasets/
|   |–– test/
|   |   |–– oxford_pets/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoOp_KDPL/
|   |   |   |   |   |–– vit_b32_ctxv1/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |   |-- other_datasets/ ...
|   |–– train_base/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoOp_KDPL/
|   |   |   |   |   |–– vit_b32_ctxv1/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Now use the script `parse_test_res.py` and run the commands below to calculate the averaged results:
```bash
# averaged results for novel classes
python output/base2new/test_new/imagenet/shots_16/CoOp_KDPL/vit_b32_ctxv1 --test-log
# averaged results for the cross-domain and cross-dataset 
python output/cross_domain_and_datasets/test/imagenet/shots_16/CoOp_KDPL/vit_b32_ctxv1 --test-log
```

The above steps can be repeated for other individual datasets.


This repository also supports using official [CoOp](https://github.com/KaiyangZhou/CoOp), [CoCoOp](https://github.com/KaiyangZhou/CoOp), [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning) and [PromptSRC](https://github.com/muzairkhattak/PromptSRC) scripts, configs and models.
Please refer to the respective documentation if you prefered to use the original bash scripts.