from __future__ import absolute_import, division, print_function

import sys
from collections import OrderedDict

import numpy as np
import torch

from misc.utils import expand_feats

sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD

sys.path.append("coco-caption")
from pycocoevalcap.bleu.bleu import Bleu

CiderD_scorer = None
Bleu_scorer = None
# CiderD_scorer = CiderD(df='corpus')


def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)


def array_to_str(arr):
    out = ""
    for i in range(len(arr)):
        out += str(arr[i]) + " "
        if arr[i] == 0:
            break
    return out.strip()


def get_self_critical_reward(
    model, core_args, sg_data, fc_feats, att_feats, att_masks, data, gen_result, opt
):
    batch_size = gen_result.size(0)
    seq_per_img = batch_size // len(data["gts"])

    # get greedy decoding baseline
    model.eval()
    with torch.no_grad():
        greedy_res, _ = model(
            sg_data,
            fc_feats,
            att_feats,
            att_masks=att_masks,
            _core_args=core_args,
            opt={"expand_features": False},
            mode="sample",
        )
    model.train()
    greedy_res = expand_feats([greedy_res], seq_per_img)[0]
    res = OrderedDict()

    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]
    gts = OrderedDict()
    for i in range(len(data["gts"])):
        gts[i] = [array_to_str(data["gts"][i][j]) for j in range(len(data["gts"][i]))]

    res_ = [{"image_id": i, "caption": res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        print("Cider scores:", _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print("Bleu scores:", _[3])
    else:
        bleu_scores = 0
    scores = (
        opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores
    )
    scores = scores[:batch_size] - scores[batch_size:]
    # batch_size * seq_length
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards

    return rewards
