import dgl.function as fn
import torch
import torch.nn as nn

from .mlp import MLP
from .gatv3conv import GATv3Conv


class NRI_block(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_block_mlp: int,
        feat: str = "hidden rep",
        dropout: float = 0.0,
        use_gat=False
    ):
        super().__init__()

        self.feat = feat
        self.n_block_mlp = n_block_mlp
        self.use_gat = use_gat

        self.init_mlp = MLP(
            in_feat=input_size,
            hidden_feat=output_size,
            out_feat=output_size,
            dropout=dropout,
            batchnorm=True,
        )
        if n_block_mlp > 0:
            self.block_mlp1 = nn.Sequential(
                *[
                    MLP(
                        in_feat=output_size,
                        hidden_feat=output_size,
                        out_feat=output_size,
                        dropout=dropout,
                        batchnorm=True,
                    )
                    for _ in range(n_block_mlp)
                ]
            )
        if self.use_gat:
            self.gat = nn.ModuleList([GATv3Conv(
                    in_feats=output_size,
                    out_feats=output_size,
                    num_heads=4,
                    activation=nn.SiLU,
                    feat_drop=dropout,
                    norm_layer=nn.LayerNorm,
                    residual=True,
                    separate_values=True,
                    dense_update=False,
                    return_pooled=False,
            ) for _ in range(1)])

        self.node_mlp = MLP(
            in_feat=output_size,
            hidden_feat=output_size,
            out_feat=output_size,
            dropout=dropout,
            batchnorm=True,
        )

        if n_block_mlp > 0:
            self.block_mlp2 = nn.Sequential(
                *[
                    MLP(
                        in_feat=output_size,
                        hidden_feat=output_size,
                        out_feat=output_size,
                        dropout=dropout,
                        batchnorm=True,
                    )
                    for _ in range(n_block_mlp)
                ]
            )

        self.edge_mlp = MLP(
            3 * output_size, output_size, output_size, dropout=dropout, batchnorm=True
        )

    def node2edge(self, edges):
        input = torch.cat([edges.src[self.feat], edges.dst[self.feat]], dim=-1)
        return {self.feat: input}

    def node_udf(self, nodes):
        # nodes.mailbox['m'] is a tensor of shape (N, D, K),
        # where N is the number of nodes in the batch,
        # D is the number of messages received per node for this node batch
        # K is the dimension of message
        return {self.feat: nodes.mailbox["m"].mean(1)}

    def forward(self, g):
        edata = g.edata[self.feat]
        g.edata[self.feat] = self.init_mlp(edata)
        if self.n_block_mlp:
            g.edata[self.feat] = g.edata[self.feat] + self.block_mlp1(
                g.edata[self.feat]
            )
        g.update_all(fn.copy_e(self.feat, "m"), self.node_udf)
        g.ndata[self.feat] = self.node_mlp(g.ndata[self.feat])
        if self.use_gat:
            for _, layer in enumerate(self.gat):
                g.ndata[self.feat] = layer(g, g.ndata[self.feat])
        if self.n_block_mlp:
            g.ndata[self.feat] = (
                self.block_mlp2(g.ndata[self.feat]) + g.ndata[self.feat]
            )
        g.apply_edges(self.node2edge)
        g.edata[self.feat] = torch.cat([edata, g.edata[self.feat]], dim=-1)
        g.edata[self.feat] = self.edge_mlp(g.edata[self.feat])

        return g
