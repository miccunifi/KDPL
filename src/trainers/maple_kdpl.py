from .kdpl_utils import load_teacher_to_cpu, load_classnames_dictionary, sampling, CustomCLIPTeacher, \
    get_K_max
from .maple import CustomCLIP, MaPLe, MultiModalPromptLearner, load_clip_to_cpu, _get_clones

import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class MultiModalPromptLearnerStudent(MultiModalPromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)

    def forward(self, prompts_indices=None):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        if prompts_indices is not None:
            prompts = prompts[prompts_indices]
        return prompts, self.proj(
            self.ctx), self.compound_prompts_text, visual_deep_prompts  # pass here original, as for visual 768 is required


class CustomCLIPStudent(CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.prompt_learner = MultiModalPromptLearnerStudent(cfg, classnames, clip_model)

    def forward(self, image, prompts_indices=None):
        if prompts_indices is not None:
            tokenized_prompts = self.tokenized_prompts[prompts_indices]
        else:
            tokenized_prompts = self.tokenized_prompts

        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(
            prompts_indices)
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class MaPLe_KDPL(MaPLe):
    """MaPLe+KDPL.
       Improving Zero-shot Generalization of Learned Prompts via Unsupervised Knowledge Distillation
       M. Mistretta et al.
       https://arxiv.org/abs/2407.03056
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32"]

    def build_model(self):

        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        self.k_max, classnames = get_K_max(cfg, classnames)

        print(f"Loading Student CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        student_model = load_clip_to_cpu(cfg)
        print(f"Loading Teacher CLIP (backbone: {cfg.TRAINER.KDPL.TEACHER})")
        teacher_model = load_teacher_to_cpu(cfg)
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            student_model.float()
            teacher_model.float()

        print("Building Student CLIP")
        self.student_model = CustomCLIPStudent(cfg, classnames, student_model)
        print("Building Teacher CLIP")
        self.teacher_model = CustomCLIPTeacher(cfg, classnames, teacher_model, self.student_model.logit_scale)

        print("Turning off gradients in the teacher")
        for name, param in self.teacher_model.named_parameters():
            param.requires_grad_(False)

        print("Turning off gradients in the student")
        name_to_update = "prompt_learner"
        for name, param in self.student_model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        self.student_model.to(self.device)
        self.teacher_model.to(self.device)

        self.optim = build_optimizer(self.student_model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.student_model, self.optim, self.sched)

        self.teacher_model.init_teacher_text_features()

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        teacher_logits = self.teacher_model(image)
        propmt_indices = sampling(teacher_logits, K=self.k_max)

        student_logits = self.student_model(image, propmt_indices)
        teacher_logits = teacher_logits[:, propmt_indices]

        teacher_log_prob = F.log_softmax(teacher_logits, dim=-1)
        student_log_prob = F.log_softmax(student_logits, dim=-1)

        loss_F = F.kl_div(student_log_prob, teacher_log_prob, log_target=True, reduction="batchmean")
        loss_R = F.kl_div(teacher_log_prob, student_log_prob, log_target=True, reduction="batchmean")
        alfa = 0.5
        beta = 0.5
        loss = (alfa * loss_F + beta * loss_R)

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss symmetric KL": loss.item(),
            "loss KL forward": loss_F.item(),
            "loss KL reverse": loss_R.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, images):
        return self.student_model(images)

