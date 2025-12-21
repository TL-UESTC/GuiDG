import torch
import torch.nn as nn
from torchvision import models
import random
import numpy as np
import torch.nn.functional as F
import os
import torch.nn.utils.weight_norm as weightNorm
import clip
from torch.cuda.amp import GradScaler, autocast
import warnings
from tqdm import tqdm
import pandas as pd
import time
import math
from thop import profile
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

warnings.filterwarnings('ignore')


def compute_attn(q_emb, k_emb, dis='cosine'):
    assert q_emb.shape[-1] == k_emb.shape[-1]
    B, N, L = k_emb.shape
    if dis == "cosine":
        raw_score = F.cosine_similarity(q_emb.view(B, 1, -1), k_emb, dim=-1)
    elif dis == "dotproduct":
        raw_score = torch.sum(q_emb.view(B, 1, -1) * k_emb, dim=-1) / (math.sqrt(L)) 
    else:
        raise ValueError('invalid att type: {}'.format(dis))
    score = raw_score.softmax(1)
    return score


class Head(nn.Module):
    # Heads
    def __init__(self, fea_dim, class_num, att="cosine"):
        super().__init__()
        self.linear_v = nn.Linear(fea_dim, fea_dim)    # 512,512
        self.linear_t = nn.Linear(class_num, 1)      # C,1
        self.att = att

    def forward(self, f_v, f_t):
        '''
            f_v: B,D
            f_t: [(C,D),...,()]
        '''
        v_emb = self.linear_v(f_v)             
        t_emb = []
        for d, x_t in enumerate(f_t):
            input_t = x_t.transpose(0,1)            
            y_t = self.linear_t(input_t).squeeze()    
            t_emb.append(y_t)
        t_emb = torch.stack(t_emb, dim=0).repeat(f_v.shape[0],1,1)      
        score = compute_attn(v_emb, t_emb, dis=self.att)
        return score


class Adapter(nn.Module):
    def __init__(self, fea_dim):
        super().__init__()
        self.dim = fea_dim
        self.layer = nn.Linear(fea_dim, fea_dim)
        self.alpha = 0.5

    def forward(self, x):
        res = self.layer(x)
        return x + self.alpha*res


class CrossModalAttn(nn.Module):
    def __init__(self, fea_dim, class_num, att="cosine", n_head=4) -> None:
        super().__init__()
        self.dis = att
        layers = []
        for i in range(n_head):
            layer = Head(fea_dim, class_num, att=att)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
    
    def forward(self, f_v, f_t):
        '''
            f_v: B,D
            f_t: [(B,C,D),...,()]
        '''
        f_v_deta = f_v.detach()
        f_t_deta = [i.detach() for i in f_t]
        attns = []
        for i in self.layers:
            attn = i(f_v_deta, f_t_deta)
            attns.append(attn)
        attns = torch.stack(attns, -1).mean(-1)     # B, DomainNum
        return attns


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg['N_CTX']
        ctx_init = cfg['CTX_INIT']
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg['CSC']:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg['CLASS_TOKEN_POSITION']

    def forward(self, x=1):
        ctx = self.ctx
        x = ctx * x
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat( 
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model).cuda()
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        if len(image.shape) == 4:   # raw image, [btz,3,224,224]
            image_features = self.image_encoder(image.type(self.dtype))
        elif len(image.shape) == 2:     # image feature [btz, 1024]
            image_features = image

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features, text_features


def entropy(input):
    epsilon = 1e-5
    entropy = -input * torch.log(input + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy.mean()


def gentropy(softmax_out):
    epsilon = 1e-5
    msoftmax = softmax_out.mean(dim=0)
    gentropy = -msoftmax * torch.log(msoftmax + epsilon)
    return torch.sum(gentropy)


def ent_loss(out, alpha=1):
    # out: BEFORE softmax
    softmax_out = nn.Softmax(dim=1)(out)
    entropy_loss = entropy(softmax_out) - alpha * gentropy(softmax_out)
    return entropy_loss


if __name__ == '__main__':
    pass