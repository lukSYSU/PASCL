import torch.nn as nn
import dgl.function as fn
from .mlp import MLP

class EdgeConv(nn.Module):
    def __init__(self, in_size, out_size, in_feature, out_feature, batchnorm=False, dropout=0.0):
        super().__init__()
        self.edge_layer = MLP(
            in_feat=in_size,
            hidden_feat=out_size,
            out_feat=out_size,
            dropout=dropout,
            batchnorm=batchnorm
        )
        self.in_feature = in_feature
        self.out_feature = out_feature

    def node_udf(self, nodes):
        # nodes.mailbox['m'] is a tensor of shape (N, D, K),
        # where N is the number of nodes in the batch,
        # D is the number of messages received per node for this node batch
        # K is the dimension of message
        return {"x": nodes.mailbox["m"].sum(1)}

    def edge_update(self, edges):
        input = edges.src["x"] + edges.dst["x"]
        # src_dst_input = torch.cat([edges.src["x"], edges.dst["x"]], dim=1)
        # dst_src_input = torch.cat([edges.dst["x"], edges.src["x"]], dim=1)
        return {
            self.out_feature: 0.5 * self.edge_layer(input)
        }

    def forward(self, g):
        g.update_all(fn.copy_e(self.in_feature, "m"), self.node_udf)
        g.apply_edges(self.edge_update)
        return g