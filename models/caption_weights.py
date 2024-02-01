import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CaptionWeights(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scalar_projection = nn.Linear(512, 1, bias=False)
        self.activation = nn.ReLU()

    def _sinusoidal_embedding(length, dim, n=10000):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        assert dim % 2 == 0, "d_model ({}) must be even".format(dim)
        positional_encoding = torch.zeros(length, dim)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(float(n)) / dim))
        )
        positional_encoding[:, 0::2] = torch.sin(position.float() * div_term)
        positional_encoding[:, 1::2] = torch.cos(position.float() * div_term)

        return positional_encoding

    def _get_adj_mat(self, obj_vecs, edges):
        # Create adjacency matrix
        adj_mat = torch.zeros(
            obj_vecs.shape[0], obj_vecs.shape[1], obj_vecs.shape[1]
        ).to(obj_vecs.device)
        adj_mat[
            torch.arange(obj_vecs.shape[0]).unsqueeze(1),
            edges[:, :, 0],
            edges[:, :, 1],
        ] = 1
        return adj_mat

    def _get_degree_centralities(self, adj_mat):
        # Get degree centralities
        degree_centralities = torch.sum(adj_mat, dim=2)
        degree_centralities = degree_centralities / torch.max(degree_centralities)
        return degree_centralities

    def build_lambda(self, sg_data):
        """
        Builds lambda vector λ(m,n) between two nodes m and n. Elements of are:
        1. The absolute difference between the caption weights
        2. The absolute difference degree centralities of the nodes
        3. The cosine similarity between the nodes
        """
        element_1 = torch.abs(
            sg_data["obj_vecs"][:, :, 0] - sg_data["obj_vecs"][:, :, 1]
        ).unsqueeze(2)
        pass

    def forward(self, sg_data):
        """
        Produces a set of weights
        """
        λ = self.build_lambda(sg_data)

        x = λ + self._get_sinusoidal_encoding(λ.shape[1], λ.shape[2]).to(λ.device)
        x = self.scalar_projection(x)
        x = self.activation(x)
        return x
