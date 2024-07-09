# KDPL (ECCV 2024)

### Improving Zero-shot Generalization of Learned Prompts via Unsupervised Knowledge Distillation

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2407.03056)

This is the **official repository** of the [**ECCV 2024 paper**](https://arxiv.org/abs/2407.03056) "*Improving Zero-shot Generalization of Learned Prompts via Unsupervised Knowledge Distillation*" by Marco Mistretta, Alberto Baldrati, Marco Bertini and Andrew David Bagdanov.

## Overview

### Abstract

Vision-Language Models (VLMs) demonstrate remarkable zero-shot generalization to unseen tasks, but fall short of the performance of supervised methods in generalizing to downstream tasks with limited data. Prompt learning is emerging as a parameter-efficient method for adapting VLMs, but state-of-the-art approaches require annotated samples. In this paper we propose a novel approach to prompt learning based on unsupervised knowledge distillation from more powerful models.  
Our approach, which we call Knowledge Distillation Prompt Learning (KDPL), can be integrated into existing prompt learning techniques and eliminates the need for labeled examples during adaptation. Our experiments on more than ten standard benchmark datasets demonstrate that KDPL is very effective at improving generalization of learned prompts for zero-shot domain generalization, zero-shot cross-dataset generalization, and zero-shot base-to-novel class generalization problems. KDPL requires no ground-truth labels for adaptation, and moreover we show that even in the absence of any knowledge of training class names it can be used to effectively transfer knowledge. 

![assets/teaser.png](assets/teaser.png "Teaser of the method")

**Top Left** Lightweight VLMs like CLIP achieve impressive zero-shot performance but lag behind supervised approaches; large VLMs incur a high computational burden. **Bottom left** Parameter-efficient prompt learning offers a non-destructive approach to adapting VLMs to downstream tasks; however, existing methods require annotated samples and struggle to generalize to unseen classes. **Right** Our approach does not require labeled samples and learns by distilling knowledge from a more powerful VLM. It can be seamlessly integrated into existing prompt learning techniques and generalizes better to unseen classes on downstream tasks.

## Citation
```bibtex
@article{mistretta2024improving,
  title={Improving Zero-shot Generalization of Learned Prompts via Unsupervised Knowledge Distillation},
  author={Mistretta, Marco and Baldrati, Alberto and Bertini, Marco and Bagdanov, Andrew D},
  journal={arXiv preprint arXiv:2407.03056},
  year={2024}
}
```

## Authors
* [**Marco Mistretta**](https://scholar.google.com/citations?hl=it&user=KMIb4eAAAAAJ)**\***
* [**Alberto Baldrati**](https://scholar.google.com/citations?hl=en&user=I1jaZecAAAAJ)**\***
* [**Marco Bertini**](https://scholar.google.it/citations?user=SBm9ZpYAAAAJ&hl=it)
* [**Andrew David Bagdanov**](https://scholar.google.com/citations?user=_Fk4YUcAAAAJ&hl=en)

**\*** Equal contribution.

## Acknowledgements

Our code is based on [PromptSRC](https://github.com/muzairkhattak/PromptSRC), along with [Co-CoOp and CoOp](https://github.com/KaiyangZhou/CoOp) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.

## Upcoming Release
The code will be released soon, stay tuned!
