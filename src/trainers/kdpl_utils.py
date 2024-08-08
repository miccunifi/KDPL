import math
import random
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import open_clip

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}."
}


def load_classnames_dictionary():
    df = pd.read_csv("classnames_dictionay.csv")
    display_names = df['ClassName']
    return display_names.tolist()



class TeacherTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    @torch.no_grad()
    def forward(self, prompts, tokenized_prompts, batch_size=128):
        # do batch inference
        tokenized_prompts_list = torch.split(tokenized_prompts, batch_size)
        output = torch.cat(
            [self.clip_model.encode_text(batch_prompts.cuda()) for batch_prompts in tokenized_prompts_list], dim=0)

        return output


class TeacherPrompt(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        if cfg.TRAINER.KDPL.CLASS_AGNOSTIC:
            temp = "a photo of a {}."
        else:
            temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]

        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print("Number of Teacher prompts:", len(prompts))
        print("First 5 Teacher prompts:", prompts[:5])
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        try:
            dtype = clip_model.dtype
        except:
            dtype = clip_model.visual.conv1.weight.dtype

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("embedding", embedding)
        self.tokenized_prompts = tokenized_prompts

    def forward(self):
        return self.embedding


class CustomCLIPTeacher(nn.Module):
    def __init__(self, cfg, classnames, teacher_model, student_logit_scale):
        super().__init__()

        self.prompt_learner = TeacherPrompt(cfg, classnames, teacher_model)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = teacher_model.visual

        self.text_encoder = TeacherTextEncoder(teacher_model)

        self.logit_scale = student_logit_scale
        try:
            self.dtype = teacher_model.dtype
        except:
            self.dtype = teacher_model.visual.conv1.weight.dtype

    def init_teacher_text_features(self):
        print("Precalculation of frozen teacher text features")
        with torch.no_grad():
            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            self.teacher_text_features = self.text_encoder(prompts, tokenized_prompts)

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        text_features = self.teacher_text_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


def load_teacher_to_cpu(cfg):
    teacher_backbone = cfg.TRAINER.KDPL.TEACHER
    if teacher_backbone == "ViT-H-14-quickgelu":
        teacher, _, _ = open_clip.create_model_and_transforms(teacher_backbone, pretrained="dfn5b")
    else:
        teacher_url = clip._MODELS[teacher_backbone]
        teacher_path = clip._download(teacher_url)
        try:
            # loading JIT archive
            teacher = torch.jit.load(teacher_path, map_location="cpu").eval()
            teacher_dict = None
        except RuntimeError:
            teacher_dict = torch.load(teacher_path, map_location="cpu")
        teacher = clip.build_model(teacher_dict or teacher.state_dict())

    return teacher


def sampling(logits, K):
    logits = logits.detach().cpu().numpy()

    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
    probabilities = np.average(probabilities, axis=0)
    result_indices = np.argpartition(probabilities, -K)[-K:]

    return result_indices


def get_K_max(cfg, classnames):
    if cfg.TRAINER.KDPL.CLASS_AGNOSTIC:
        classnames = load_classnames_dictionary()
        k_max = cfg.TRAINER.KDPL.K_MAX
    else:
        k_max = len(classnames)
    return k_max, classnames
