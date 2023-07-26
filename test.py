import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 32
num_nodes = 50
num_edges = 62
n_heads = 5
out_features = 1000
n_hidden = out_features // n_heads
is_edge_included = False

assert n_hidden % n_heads == 0
assert num_edges < num_nodes * (num_nodes - 1)

node_reproject = nn.Linear(1, n_hidden * n_heads, bias=False)
edge_reproject = nn.Linear(8, n_hidden * n_heads, bias=False)

attn = (
    nn.Linear(n_hidden * 3, 1, bias=False)
    if is_edge_included
    else nn.Linear(n_hidden * 2, 1, bias=False)
)

activation = nn.LeakyReLU(negative_slope=0.2)
softmax = nn.Softmax(dim=1)
dropout = nn.Dropout(0.1)

################
# Build
################
h_proj = torch.randint(0, 100, (batch_size, num_nodes, 1), dtype=torch.float32)
adj_mat = torch.zeros(batch_size, num_nodes, num_nodes, 1)

edge_feats = []
for i in range(batch_size):
    for j in range(num_edges):
        row = torch.randint(0, num_nodes, (1,))
        col = torch.randint(0, num_nodes, (1,))
        while row == col:
            col = torch.randint(0, num_nodes, (1,))
        adj_mat[i, row, col] = 1
    edge_count = int(adj_mat.squeeze(-1)[i].sum().item())

    feat = torch.rand((edge_count, 8))
    diff = num_edges - edge_count
    pad = torch.zeros((diff, 8))
    feat = torch.cat([feat, pad], dim=0)
    edge_feats.append(feat)
e_proj = torch.stack(edge_feats, dim=0)
################

h_proj = node_reproject(h_proj).view(batch_size, num_nodes, n_heads, n_hidden)
e_proj = edge_reproject(e_proj).view(batch_size, num_edges, n_heads, n_hidden)

################

h_proj_repeat = h_proj.repeat(1, num_nodes, 1, 1)
h_proj_repeat_interleave = h_proj.repeat_interleave(num_nodes, dim=1)

concat = torch.cat([h_proj_repeat_interleave, h_proj_repeat], dim=-1)

if is_edge_included:
    edge_feats = torch.zeros(batch_size, num_nodes * num_nodes, n_heads, n_hidden)
    indexes = (
        adj_mat.squeeze(-1).view(batch_size, num_nodes * num_nodes).type(torch.int64)
    )

    diff = num_nodes * num_nodes - num_edges
    pad = torch.zeros((batch_size, diff, n_heads, n_hidden))
    e_proj = torch.cat([e_proj, pad], dim=1)

    # TODO: Can we do this without a loop?
    for batch in range(batch_size):
        index_edge_feat = indexes[batch].nonzero()
        index_edge_tensor = e_proj

        edge_feats[batch, index_edge_feat.view(indexes[batch].sum())] = e_proj[
            batch, : indexes[batch].sum(), :, :
        ]

    concat = torch.cat([concat, edge_feats], dim=-1)

c = 3 if is_edge_included else 2
concat = concat.view(batch_size, num_nodes, num_nodes, n_heads, n_hidden * c)

e = activation(attn(concat))
e = e.squeeze(-1)

assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == num_nodes
assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == num_nodes
assert adj_mat.shape[3] == 1 or adj_mat.shape[3] == n_heads

e = e.masked_fill(adj_mat == 0, float("-inf"))
a = F.softmax(e, dim=2)
a = dropout(a)

attn_res = torch.einsum("bijh,bjhf->bihf", a, h_proj)


print("we're done here")
