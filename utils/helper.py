from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise Exception("value not allowed")


def str2list(v):
    return v.split(",")


def sort_pack_padded_sequence(
    input: torch.Tensor, lengths: torch.Tensor
) -> tuple([PackedSequence, torch.Tensor]):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(
        input[indices], sorted_lengths.to("cpu"), batch_first=True
    )
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(
    input: PackedSequence, inv_ix: torch.Tensor
) -> torch.Tensor:
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(
    module: nn.Module, att_feats: torch.Tensor, att_masks: torch.Tensor
) -> torch.Tensor:
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(
            att_feats, att_masks.data.long().sum(1)
        )
        return pad_unsort_packed_sequence(
            PackedSequence(module(packed[0]), packed[1]), inv_ix
        )
    else:
        return module(att_feats)


def build_embeding_layer(vocab_size: int, dim: int, drop_prob: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Embedding(vocab_size, dim), nn.ReLU(), nn.Dropout(drop_prob)
    )


def feat_3d_to_2d(
    obj_vecs: torch.Tensor,
    attr_vecs: torch.Tensor,
    rela_vecs: torch.Tensor,
    edges: torch.Tensor,
) -> tuple([torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple([int, int])]):
    """
    convert 3d features of shape (B, N, d) into 2d features of shape (B*N, d)
    """
    B, No = obj_vecs.shape[:2]
    obj_vecs = obj_vecs.view(-1, obj_vecs.size(-1))
    attr_vecs = attr_vecs.view(-1, attr_vecs.size(-1))
    rela_vecs = rela_vecs.view(-1, rela_vecs.size(-1))

    # edge: (B, max_rela_num, 2) => (B*max_rela_num, 2)
    obj_offsets = edges.new_tensor(range(0, B * No, No))
    edges = edges + obj_offsets.view(-1, 1, 1)
    edges = edges.view(-1, edges.size(-1))
    return obj_vecs, attr_vecs, rela_vecs, edges, (B, No)


def feat_2d_to_3d(
    obj_vecs: torch.Tensor,
    attr_vecs: torch.Tensor,
    rela_vecs: torch.Tensor,
    rela_masks: torch.Tensor,
    ori_shape: tuple([int, int]),
) -> tuple([torch.Tensor, torch.Tensor, torch.Tensor]):
    """
    convert 2d features of shape (B*N, d) back into 3d features of shape (B, N, d)
    """
    B, No = ori_shape
    obj_vecs = obj_vecs.view(B, No, -1)
    attr_vecs = attr_vecs.view(B, No, -1)
    rela_vecs = rela_vecs.view(B, -1, rela_vecs.size(-1)) * rela_masks
    return obj_vecs, attr_vecs, rela_vecs
