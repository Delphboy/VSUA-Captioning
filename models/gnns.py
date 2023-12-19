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
        concat = torch.cat([obj_vecs, attr_vecs], dim=-1)
        new_attr_vecs = self.gnn_attr(concat) + attr_vecs

        # rela: get node features for each triplet <subject, relation, object>
        subject_idx = edges[:, 0].contiguous()  # index of subject
        subject_vecs = obj_vecs[subject_idx]

        object_idx = edges[:, 1].contiguous()  # index of object
        object_vecs = obj_vecs[object_idx]

        if self.opt.rela_gnn_type == 0:  # Default
            triplet_vecs = torch.cat([subject_vecs, rela_vecs, object_vecs], dim=1)
        elif self.opt.rela_gnn_type == 1:
            triplet_vecs = torch.cat([subject_vecs + object_vecs, rela_vecs], dim=1)
        else:
            raise NotImplementedError()

        new_rela_vecs = self.gnn_rela(triplet_vecs) + rela_vecs

        new_obj_vecs, new_attr_vecs, new_rela_vecs = helper.feat_2d_to_3d(
            new_obj_vecs, new_attr_vecs, new_rela_vecs, rela_masks, ori_shape
        )

        return new_obj_vecs, new_attr_vecs, new_rela_vecs


class GraphAttentionLayer(nn.Module):
    """
    ## Graph attention layer based on: https://nn.labml.ai/graphs/gat/index.html
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        is_concat: bool = True,
        dropout: float = 0.6,
        leaky_relu_negative_slope: float = 0.2,
    ):
        """
        * in_features: number of input features per node
        * out_features: number of output features per node
        * n_heads: number of attention heads
        * is_concat: whether the multi-head results should be concatenated or averaged
        * dropout: dropout probability
        * leaky_relu_negative_slope: negative slope for leaky relu activation
        """
        super(GraphAttentionLayer, self).__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads

        # Calculate the number of dimensions per head
        if is_concat:
            # If we are concatenating the multiple heads
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial transformation;
        # i.e. to transform the node embeddings before self-attention
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.edge_linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)

        # Linear layer to compute attention score
        self.weight_score = nn.Sequential(
            nn.Linear(self.n_hidden * 2, 1, bias=False), nn.ReLU()
        )
        self.attn = nn.Linear(self.n_hidden * 3, 1, bias=False)

        # The activation for attention score
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)

        # Softmax to compute attention weight
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        adj_mat: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        obj_embds: Optional[torch.Tensor] = None,
    ):
        """
        * h, is the input node embeddings of shape [n_nodes, in_features].
        * adj_mat is the adjacency matrix of shape [n_nodes, n_nodes, n_heads].
        We use shape [n_nodes, n_nodes, 1] since the adj is the same for each head
        * edge_attr is the edge attributes of shape [n_edges, edge_attr_dim].

        Adjacency matrix represent the edges (or connections) among nodes.
        adj_mat[i][j] is 1 if there is an edge from node i to node j.
        """
        batch_size = h.shape[0]
        num_nodes = h.shape[1]
        num_edges = edge_attr.shape[1]

        # Add a dimension for the number of heads
        adj_mat = adj_mat.unsqueeze(-1)

        # Add self-connections
        # TODO: How to handle lack of edge features for self-connections?
        # adj_mat = adj_mat + torch.eye(num_nodes).to(adj_mat.device).unsqueeze(-1)

        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == num_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == num_nodes
        assert adj_mat.shape[3] == 1 or adj_mat.shape[3] == self.n_heads

        # Nodes
        g = self.linear(h).view(batch_size, num_nodes, self.n_heads, self.n_hidden)
        g_repeat = g.repeat(1, num_nodes, 1, 1)
        g_repeat_interleave = g.repeat_interleave(num_nodes, dim=1)
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)

        # Edges
        e_proj = self.edge_linear(edge_attr).view(
            batch_size, num_edges, self.n_heads, self.n_hidden
        )
        edge_feats = torch.zeros(
            batch_size, num_nodes * num_nodes, self.n_heads, self.n_hidden
        ).to(h.device)
        indexes = (
            adj_mat.squeeze(-1)
            .view(batch_size, num_nodes * num_nodes)
            .type(torch.int64)
        )

        diff = num_nodes * num_nodes - num_edges
        pad = torch.zeros(
            (batch_size, diff, self.n_heads, self.n_hidden), device=h.device
        )
        e_proj = torch.cat([e_proj, pad], dim=1)

        # TODO: Can we do this without a loop?
        for batch in range(batch_size):
            index_edge_feat = indexes[batch].nonzero()

            edge_feats[batch, index_edge_feat.view(indexes[batch].sum())] = e_proj[
                batch, : indexes[batch].sum(), :, :
            ]

        g_concat = torch.cat([g_concat, edge_feats], dim=-1)

        g_concat = g_concat.view(
            batch_size, num_nodes, num_nodes, self.n_heads, self.n_hidden * 3
        )

        # Compute attention score
        e = self.activation(self.attn(g_concat))
        e = e.squeeze(-1)

        # Weights from Object Label Embeddings
        if obj_embds is not None:
            obj_embds = obj_embds.view(
                batch_size, num_nodes, self.n_heads, self.n_hidden
            )
            obj_embds_repeat = obj_embds.repeat(1, num_nodes, 1, 1)
            obj_embds_repeat_interleave = obj_embds.repeat_interleave(num_nodes, dim=1)
            obj_embds_concat = torch.cat(
                [obj_embds_repeat_interleave, obj_embds_repeat], dim=-1
            ).view(batch_size, num_nodes, num_nodes, self.n_heads, self.n_hidden * 2)

            weights_score = self.weight_score(obj_embds_concat).squeeze(-1)

            e = e * weights_score

        e = e.masked_fill(adj_mat == 0, -1000)
        a = self.softmax(e)
        a = self.dropout(a)

        attn_res = torch.einsum("bijh,bjhf->bihf", a, g)

        if self.is_concat:
            return attn_res.reshape(batch_size, num_nodes, self.n_hidden * self.n_heads)
        else:
            return attn_res.mean(dim=2)


class GraphAttentionNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        is_concat: bool = True,
        dropout: float = 0.6,
        leaky_relu_negative_slope: float = 0.1,
        opt=None,
    ) -> None:
        super(GraphAttentionNetwork, self).__init__()
        self.layer_1 = GraphAttentionLayer(
            in_features,
            out_features,
            n_heads,
            is_concat,
            dropout,
            leaky_relu_negative_slope,
        )
        self.activation_1 = nn.LeakyReLU(leaky_relu_negative_slope)
        self.layer_2 = GraphAttentionLayer(
            in_features,
            out_features,
            n_heads,
            is_concat,
            dropout,
            leaky_relu_negative_slope,
        )
        self.activation_2 = nn.LeakyReLU(leaky_relu_negative_slope)
        self.gnn = GNN(opt)

    def forward(
        self,
        obj_vecs: torch.Tensor,
        attr_vecs: torch.Tensor,
        rela_vecs: torch.Tensor,
        edges: torch.Tensor,
        rela_masks: Optional[torch.Tensor] = None,
        obj_embds: Optional[torch.Tensor] = None,
    ) -> tuple([torch.Tensor, torch.Tensor, torch.Tensor]):
        # Update attributes and relations
        obj_vecs, attr_vecs, rela_vecs = self.gnn(
            obj_vecs, attr_vecs, rela_vecs, edges, rela_masks
        )

        # Create adjacency matrix
        adj_mat = torch.zeros(
            obj_vecs.shape[0], obj_vecs.shape[1], obj_vecs.shape[1]
        ).to(obj_vecs.device)
        adj_mat[
            torch.arange(obj_vecs.shape[0]).unsqueeze(1),
            edges[:, :, 0],
            edges[:, :, 1],
        ] = 1

        obj_vecs = self.layer_1(obj_vecs, adj_mat, rela_vecs, obj_embds)
        obj_vecs = self.activation_1(obj_vecs)

        # obj_vecs = self.layer_2(obj_vecs, adj_mat, rela_vecs, rela_weights)
        # obj_vecs = self.activation_2(obj_vecs)

        return obj_vecs, attr_vecs, rela_vecs


class GraphConvolutionalLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(GraphConvolutionalLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(
        self,
        h: torch.Tensor,
        adj_mat: torch.Tensor,
    ) -> torch.Tensor:
        h = self.linear(h)
        h = torch.matmul(adj_mat, h)

        return h


class GraphConvolutionalNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        opt=None,
    ) -> None:
        super(GraphConvolutionalNetwork, self).__init__()
        self.layer_1 = GraphConvolutionalLayer(in_features, out_features)
        self.activation_1 = nn.ReLU()
        self.layer_2 = GraphConvolutionalLayer(in_features, out_features)
        self.activation_2 = nn.ReLU()
        self.gnn = GNN(opt)

    def forward(
        self,
        obj_vecs: torch.Tensor,
        attr_vecs: torch.Tensor,
        rela_vecs: torch.Tensor,
        edges: torch.Tensor,
        rela_masks: Optional[torch.Tensor] = None,
        rela_weights: Optional[torch.Tensor] = None,
    ) -> tuple([torch.Tensor, torch.Tensor, torch.Tensor]):
        # Create adjacency matrix
        adj_mat = torch.zeros(
            obj_vecs.shape[0], obj_vecs.shape[1], obj_vecs.shape[1]
        ).to(obj_vecs.device)
        adj_mat[
            torch.arange(obj_vecs.shape[0]).unsqueeze(1),
            edges[:, :, 0],
            edges[:, :, 1],
        ] = 1

        obj_vecs = self.layer_1(obj_vecs, adj_mat)
        obj_vecs = self.activation_1(obj_vecs)

        obj_vecs = self.layer_2(obj_vecs, adj_mat)
        obj_vecs = self.activation_2(obj_vecs)

        # return obj_vecs, attr_vecs, rela_vecs
        return self.gnn(obj_vecs, attr_vecs, rela_vecs, edges, rela_masks)
