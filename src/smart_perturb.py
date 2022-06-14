# Copyright (c) Microsoft. All rights reserved.
# From GitHub repo https://github.com/namisan/mt-dnn
from copy import deepcopy
import torch
import logging
import random
from torch.nn import Parameter, CrossEntropyLoss, BCELoss
import torch.nn.functional as F


def stable_kl(logit, target, epsilon=1e-6, reduce=True):
    logit = logit.view(-1, logit.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0 / (p + epsilon) - 1 + epsilon).detach().log()
    ry = -(1.0 / (y + epsilon) - 1 + epsilon).detach().log()
    if reduce:
        return (p * (rp - ry) * 2).sum() / bs
    else:
        return (p * (rp - ry) * 2).sum()

def generate_noise(embed, mask, epsilon=1e-5):
    noise = embed.data.new(embed.size()).normal_(0, 1) * epsilon
    noise.detach()
    noise.requires_grad_()
    return noise


class SmartPerturbation:
    def __init__(
        self,
        num_label,
        num_label2=0,
        epsilon=1e-6,
        multi_gpu_on=False,
        step_size=1e-3,
        noise_var=1e-5,
        norm_p="inf",
        k=1,
        fp16=False,
        norm_level=0,
    ):
        super(SmartPerturbation, self).__init__()
        self.NUM_LABEL = num_label
        self.NUM_LABEL2 = num_label2
        self.epsilon = epsilon
        # eta
        self.step_size = step_size
        self.multi_gpu_on = multi_gpu_on
        self.fp16 = fp16
        self.K = k
        # sigma
        self.noise_var = noise_var
        self.norm_p = norm_p
        #self.encoder_type = encoder_type
        self.norm_level = norm_level > 0

    def _norm_grad(self, grad, eff_grad=None, sentence_level=False):
        if self.norm_p == "l2":
            if sentence_level:
                direction = grad / (
                    torch.norm(grad, dim=(-2, -1), keepdim=True) + self.epsilon
                )
            else:
                direction = grad / (
                    torch.norm(grad, dim=-1, keepdim=True) + self.epsilon
                )
        elif self.norm_p == "l1":
            direction = grad.sign()
        else:
            if sentence_level:
                direction = grad / (
                    grad.abs().max((-2, -1), keepdim=True)[0] + self.epsilon
                )
            else:
                direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
                eff_direction = eff_grad / (
                    grad.abs().max(-1, keepdim=True)[0] + self.epsilon
                )
        return direction, eff_direction

    def forward(
        self,
        model,
        labels=None,
        labels2=None,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
    ):

        logits = F.one_hot(labels.flatten(), self.NUM_LABEL)
        if labels2 is not None:
            logits2 = F.one_hot(labels2.flatten(), self.NUM_LABEL2)
        # adv training
        
        vat_args = {
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'token_type_ids':token_type_ids,
            'fwd_type':1,
        }

        # init delta
        embed = model(**vat_args)
        noise = generate_noise(embed, attention_mask, epsilon=self.noise_var)
        for step in range(0, self.K):
            vat_args = {
                'input_ids':input_ids,
                'attention_mask':attention_mask,
                'token_type_ids':token_type_ids,
                'fwd_type':2,
                'embed': embed+noise
            }
            adv_logits = model(**vat_args)
            # Loss for classification
            if labels2 is not None:
                adv_loss = stable_kl(adv_logits[0], logits.detach(), reduce=False)
                adv_loss += stable_kl(adv_logits[1], logits2.detach(), reduce=False)
            else:
                adv_loss = stable_kl(adv_logits, logits.detach(), reduce=False)

            (delta_grad,) = torch.autograd.grad(
                adv_loss, noise, only_inputs=True, retain_graph=False
            )
            norm = delta_grad.norm()
            if torch.isnan(norm) or torch.isinf(norm):
                return 0
            eff_delta_grad = delta_grad * self.step_size
            delta_grad = noise + delta_grad * self.step_size
            noise, eff_noise = self._norm_grad(
                delta_grad, eff_grad=eff_delta_grad, sentence_level=self.norm_level
            )
            noise = noise.detach()
            noise.requires_grad_()
        vat_args = {
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'token_type_ids':token_type_ids,
            'fwd_type':2,
            'embed': embed+noise
        }
        adv_logits = model(**vat_args)
        adv_lc = CrossEntropyLoss()
        #adv_loss = adv_lc(logits, adv_logits, ignore_index=-1)
        if labels2 is not None:
            adv_loss =  adv_lc(adv_logits[0].flatten(0, -2), labels.flatten())
            adv_loss += adv_lc(adv_logits[1].flatten(0, -2), labels2.flatten())
        else:
            adv_loss = adv_lc(adv_logits.flatten(0, -2), labels.flatten())
        return adv_loss #, embed.detach().abs().mean() #, eff_noise.detach().abs().mean()