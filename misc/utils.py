from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.optim as optim


def if_use_att(caption_model):
    if caption_model in [""]:
        return False
    return True


def if_use_fc(caption_model):
    if caption_model in []:
        return False
    return True


def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ""
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + " "
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq > 0).float()
        mask = to_contiguous(
            torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
        ).view(-1)
        output = -input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, : input.size(1)]
        mask = mask[:, : input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group["lr"]


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def build_optimizer(params, opt):
    if opt.optim == "rmsprop":
        return optim.RMSprop(
            params,
            opt.learning_rate,
            opt.optim_alpha,
            opt.optim_epsilon,
            weight_decay=opt.weight_decay,
        )
    elif opt.optim == "adagrad":
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == "sgd":
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == "sgdm":
        return optim.SGD(
            params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay
        )
    elif opt.optim == "sgdmom":
        return optim.SGD(
            params,
            opt.learning_rate,
            opt.optim_alpha,
            weight_decay=opt.weight_decay,
            nesterov=True,
        )
    elif opt.optim == "adam":
        return optim.Adam(
            params,
            opt.learning_rate,
            (opt.optim_alpha, opt.optim_beta),
            opt.optim_epsilon,
            weight_decay=opt.weight_decay,
        )
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))


# batch_size * feat_size -> (batch_size * count) * feat_size
def expand_feats(inputs: list, count: int) -> list:
    temp = []
    for input in inputs:
        if type(input) is list or input is None:
            temp.append(input)
            continue
        expanded = (
            input.unsqueeze(1)
            .expand(
                *(
                    (
                        input.size(0),
                        count,
                    )
                    + input.size()[1:]
                )
            )
            .contiguous()
            .view(*((input.size(0) * count,) + input.size()[1:]))
        )
        temp.append(expanded)
    return temp
