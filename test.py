import torch
import torch.nn as nn

batch_size = 16
num_nodes = 50
num_edges = 63
n_heads = 5
n_hidden = 1000

node_reproject = nn.Linear(1, n_hidden * n_heads, bias=False)
edge_reproject = nn.Linear(8, n_hidden * n_heads, bias=False)

################
# Build
################
nodes_tensor = torch.randint(0, 100, (batch_size, num_nodes, 1), dtype=torch.float32)
edge_features_tensor = torch.rand(batch_size, num_edges, 8)
adj_mat = torch.zeros(batch_size, num_nodes, num_nodes, 1)
for i in range(batch_size):
    for j in range(num_edges):
        row = torch.randint(0, num_nodes, (1,))
        col = torch.randint(0, num_nodes, (1,))
        while row == col:
            col = torch.randint(0, num_nodes, (1,))
        adj_mat[i, row, col] = 1

################

nodes_tensor = node_reproject(nodes_tensor).view(
    batch_size, num_nodes, n_heads, n_hidden
)
edge_features_tensor = edge_reproject(edge_features_tensor).view(
    batch_size, num_edges, n_heads, n_hidden
)

################


non_zero_indexes = []
concats = []
for batch in range(batch_size):
    non_zero = torch.nonzero(adj_mat.squeeze(-1)[batch])

    _from = nodes_tensor[batch, non_zero[:, 0], :, :]
    _to = nodes_tensor[batch, non_zero[:, 1], :, :]
    concat = torch.concat([_from, _to], dim=-1)

    diff = num_edges - non_zero.shape[0]
    pad = torch.zeros([diff, concat.shape[1], concat.shape[2]], dtype=torch.long)
    concat = torch.cat([concat, pad], dim=0)
    concats.append(concat)

concats = torch.stack(concats, dim=0)
concats = torch.concat([concats, edge_features_tensor], dim=-1)

print("we're done here")
