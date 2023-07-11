from argparse import Namespace
from typing import Optional

import torch
import torch.nn as nn

import utils.helper as helper


class GNN(nn.Module):
    def __init__(self, opt: Namespace) -> None:
        super(GNN, self).__init__()
        self.opt = opt
        in_dim = opt.rnn_size
        out_dim = opt.rnn_size

        if self.opt.rela_gnn_type == 0:
            in_rela_dim = in_dim * 3
        elif self.opt.rela_gnn_type == 1:
            in_rela_dim = in_dim * 2
        else:
            raise NotImplementedError()

        # gnn with simple MLP
        self.gnn_attr = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(opt.drop_prob_lm),
        )
        self.gnn_rela = nn.Sequential(
            nn.Linear(in_rela_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(opt.drop_prob_lm),
        )

    def forward(
        self,
        obj_vecs: torch.Tensor,
        attr_vecs: torch.Tensor,
        rela_vecs: torch.Tensor,
        edges: torch.Tensor,
        rela_masks: Optional[torch.Tensor] = None,
    ) -> tuple([torch.Tensor, torch.Tensor, torch.Tensor]):
        # for easily indexing the subject and object of each relation in the tensors
        obj_vecs, attr_vecs, rela_vecs, edges, ori_shape = helper.feat_3d_to_2d(
            obj_vecs, attr_vecs, rela_vecs, edges
        )

        # obj
        new_obj_vecs = obj_vecs

        # attr
        new_attr_vecs = (
            self.gnn_attr(torch.cat([obj_vecs, attr_vecs], dim=-1)) + attr_vecs
        )

        # rela: get node features for each triplet <subject, relation, object>
        s_idx = edges[:, 0].contiguous()  # index of subject
        s_vecs = obj_vecs[s_idx]

        o_idx = edges[:, 1].contiguous()  # index of object
        o_vecs = obj_vecs[o_idx]

        if self.opt.rela_gnn_type == 0:
            t_vecs = torch.cat([s_vecs, rela_vecs, o_vecs], dim=1)
        elif self.opt.rela_gnn_type == 1:
            t_vecs = torch.cat([s_vecs + o_vecs, rela_vecs], dim=1)
        else:
            raise NotImplementedError()

        new_rela_vecs = self.gnn_rela(t_vecs) + rela_vecs

        new_obj_vecs, new_attr_vecs, new_rela_vecs = helper.feat_2d_to_3d(
            new_obj_vecs, new_attr_vecs, new_rela_vecs, rela_masks, ori_shape
        )

        return new_obj_vecs, new_attr_vecs, new_rela_vecs
