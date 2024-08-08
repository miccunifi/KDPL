from .kdpl_utils import load_teacher_to_cpu, load_classnames_dictionary, sampling, CustomCLIPTeacher, \
    get_K_max
from .promptsrc import CustomCLIP, PromptSRC, VLPromptLearner, load_clip_to_cpu

import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES

_tokenizer = _Tokenizer()


class VLPromptLearnerStudent(VLPromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)

    def forward(self, prompts_indices=None):
        prompts = super().forward()
        if prompts_indices is not None:
            prompts = prompts[prompts_indices]
        return prompts


class CustomCLIPStudent(CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.prompt_learner = VLPromptLearnerStudent(cfg, classnames, clip_model)

    def forward(self, image, prompts_indices=None):
        if prompts_indices is not None:
            tokenized_prompts = self.tokenized_prompts[prompts_indices]
        else:
            tokenized_prompts = self.tokenized_prompts

        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner(prompts_indices)
        # Compute the prompted image and text features
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits
        logits = logit_scale * image_features @ text_features.t()
        if self.prompt_learner.training:
            # Now calculate the frozen pre-trained features
            fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                # Compute pre-trained frozen visual features
                zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()

            if prompts_indices is not None:
                fixed_embeddings = fixed_embeddings[prompts_indices]
                zero_shot_logits = zero_shot_logits[:, prompts_indices]
            return None, text_features, fixed_embeddings, zero_shot_features, \
                image_features, zero_shot_logits, logits
        else:
            return logits


@TRAINER_REGISTRY.register()
class PromptSRC_KDPL(PromptSRC):
    """PromptSRC+KDPL.
       Improving Zero-shot Generalization of Learned Prompts via Unsupervised Knowledge Distillation
       M. Mistretta et al.
       https://arxiv.org/abs/2407.03056
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTSRC.PREC in ["fp16", "fp32"]

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
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        self.student_model.to(self.device)
        self.teacher_model.to(self.device)

        self.optim = build_optimizer(self.student_model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.student_model, self.optim, self.sched)
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        mean = cfg.TRAINER.PROMPTSRC.GPA_MEAN
        stdev = cfg.TRAINER.PROMPTSRC.GPA_STD
        gauss = self.get_gauss(mean, stdev)
        self.gauss = np.array([gauss(a) for a in range(1, N + 1)])
        self.gauss = self.gauss / sum(self.gauss)
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTSRC.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel

        # Keep model with GPA
        self.previous_model_gpa = None
        self.teacher_model.init_teacher_text_features()

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        teacher_logits = self.teacher_model(image)
        propmt_indices = sampling(teacher_logits, K=self.k_max)

        _, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
            zero_shot_logits, student_logits = self.student_model(image, propmt_indices)
        teacher_logits = teacher_logits[:, propmt_indices]

        loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(),
                                  reduction='mean') * self.cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT
        loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(),
                                   reduction='mean') * self.cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT
        L_SCL_logits = F.kl_div(
            F.log_softmax(student_logits / 1, dim=1),
            F.log_softmax(zero_shot_logits / 1, dim=1),
            reduction='sum',
            log_target=True
        ) * (1 * 1) / student_logits.numel()
        L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)

        teacher_log_prob = F.log_softmax(teacher_logits, dim=-1)
        student_log_prob = F.log_softmax(student_logits, dim=-1)

        loss_F = F.kl_div(student_log_prob, teacher_log_prob, log_target=True, reduction="batchmean")
        loss_R = F.kl_div(teacher_log_prob, student_log_prob, log_target=True, reduction="batchmean")
        alfa = 0.5
        beta = 0.5

        loss = (alfa * loss_F + beta * loss_R) + L_SCL

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        loss_summary = {
            "loss symmetric KL": (alfa * loss_F + beta * loss_R).item(),
            "loss KL forward": loss_F.item(),
            "loss KL reverse": loss_R.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            # Means one epoch is completed, perform GPA
            self.step_counter = self.step_counter + 1
            current_epoch_weight = self.gauss[self.step_counter - 2]
            current_model_weights = copy.deepcopy(self.student_model.state_dict())
            weighted_state_dict = self.state_dict_weighting(current_model_weights, current_epoch_weight)
            if self.previous_model_gpa is None:
                self.previous_model_gpa = weighted_state_dict
            else:
                self.previous_model_gpa = self.state_dict_add(weighted_state_dict, self.previous_model_gpa)

        if self.step_counter == self.student_model.total_epochs + 1:
            print("Using GPA model for final inference...")
            self.student_model.load_state_dict(self.previous_model_gpa)
            self.student_model.load_state_dict(self.previous_model_gpa)
        return loss_summary

    def model_inference(self, images):
        return self.student_model(images)
