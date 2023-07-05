from argparse import Namespace
from typing import Optional

import torch
import torch.nn as nn


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
        obj_vecs, attr_vecs, rela_vecs, edges, ori_shape = self.feat_3d_to_2d(
            obj_vecs, attr_vecs, rela_vecs, edges
        )

        # obj
        new_obj_vecs = obj_vecs

        # attr
        new_attr_vecs = (
            self.gnn_attr(torch.cat([obj_vecs, attr_vecs], dim=-1)) + attr_vecs
        )

        # rela
        # get node features for each triplet <subject, relation, object>
        s_idx = edges[:, 0].contiguous()  # index of subject
        o_idx = edges[:, 1].contiguous()  # index of object
        s_vecs = obj_vecs[s_idx]
        o_vecs = obj_vecs[o_idx]
        if self.opt.rela_gnn_type == 0:
            t_vecs = torch.cat([s_vecs, rela_vecs, o_vecs], dim=1)
        elif self.opt.rela_gnn_type == 1:
            t_vecs = torch.cat([s_vecs + o_vecs, rela_vecs], dim=1)
        else:
            raise NotImplementedError()
        new_rela_vecs = self.gnn_rela(t_vecs) + rela_vecs

        new_obj_vecs, new_attr_vecs, new_rela_vecs = self.feat_2d_to_3d(
            new_obj_vecs, new_attr_vecs, new_rela_vecs, rela_masks, ori_shape
        )

        return new_obj_vecs, new_attr_vecs, new_rela_vecs

    def feat_3d_to_2d(
        self,
        obj_vecs: torch.Tensor,
        attr_vecs: torch.Tensor,
        rela_vecs: torch.Tensor,
        edges: torch.Tensor,
    ) -> tuple(
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple([int, int])]
    ):
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
        self,
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
