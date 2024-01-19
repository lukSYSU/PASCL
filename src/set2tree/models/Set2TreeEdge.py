import dgl
import torch
from torch import nn

from ..layers import MLP, NRI_block, EdgeConv, GAT
from ..utils.data_utils import construct_rel_recvs, construct_rel_sends


class Set2TreeEdge(nn.Module):
    """NRI model built off the official implementation.

    Contains adaptations to make it work with our use case, plus options for extra layers to give it some more oomph

    Args:
        infeatures (int): Number of input features
        num_classes (int): Number of classes in ouput prediction
        nblocks (int): Number of NRI blocks in the model
        dim_feedforward (int): Width of feedforward layers
        initial_mlp_layers (int): Number of MLP (2 feedforward, 1 batchnorm (optional)) before NRI blocks
        block_additional_mlp_layers (int): Number of additional MLP (2 feedforward, 1 batchnorm (optional)) within NRI blocks, when 0 the total number is one.
        final_mlp_layers (int): Number of MLP (2 feedforward, 1 batchnorm (optional)) after NRI blocks
        dropout (float): Dropout rate
        factor (bool): Whether to use NRI blocks at all (useful for benchmarking)
        tokenize ({int: int}): Dictionary of tokenized features to embed {index_of_feature: num_tokens}
        embedding_dims (int): Number of embedding dimensions to use for tokenized features
        batchnorm (bool): Whether to use batchnorm in MLP layers
    """

    def __init__(
        self,
        infeatures,
        num_classes,
        nblocks=1,
        dim_feedforward=128,
        initial_mlp_layers=1,
        block_additional_mlp_layers=1,
        final_mlp_layers=1,
        dropout=0.3,
        factor=True,
        tokenize=None,
        embedding_dims=None,
        # batchnorm=True,
        symmetrize=True,
        edge_blocks=1,
        **kwargs,
    ):
        super().__init__()

        assert dim_feedforward % 2 == 0, "dim_feedforward must be an even number"

        self.num_classes = num_classes
        self.factor = factor
        self.tokenize = tokenize
        self.symmetrize = symmetrize
        self.block_additional_mlp_layers = block_additional_mlp_layers
        self.feat = "hidden rep"
        # self.max_leaves = max_leaves

        # Create first half of inital NRI half-block to go from leaves to edges
        initial_mlp = [
            MLP(
                infeatures,
                dim_feedforward,
                dim_feedforward,
                dropout=dropout,
            )
        ]
        initial_mlp.extend(
            [
                MLP(dim_feedforward, dim_feedforward, dim_feedforward, dropout=dropout)
                for _ in range(initial_mlp_layers)
            ]
        )
        self.initial_mlp = nn.Sequential(*initial_mlp)
        self.pre_blocks_mlp = MLP(
            dim_feedforward * 2,
            dim_feedforward,
            dim_feedforward,
            dropout=dropout,
        )
        # MLP to reduce feature dimensions from first Node2Edge before blocks begin
        self.blocks = nn.ModuleList(
            [
                # List of MLP sequences within each block
                NRI_block(
                    dim_feedforward,
                    dim_feedforward,
                    block_additional_mlp_layers,
                    dropout=dropout,
                )
                for _ in range(nblocks)
            ]
        )
        # Final linear layers as requested

        self.edge_conv_block = nn.Sequential(
            *[
                EdgeConv(
                    in_size=dim_feedforward,
                    out_size=dim_feedforward,
                    in_feature=self.feat,
                    out_feature=self.feat,
                    dropout=dropout,
                    batchnorm=True,
                )
                for _ in range(edge_blocks)
            ]
        )
        # self.final_mlp = nn.Sequential(*[MLP(dim_feedforward, dim_feedforward, dim_feedforward, dropout, batchnorm) for _ in range(final_mlp_layers)])
        final_mlp = [
            MLP(
                dim_feedforward * 2,
                dim_feedforward,
                dim_feedforward,
                dropout=dropout,
            )
        ]
        # Add any additional layers as per request
        final_mlp.extend(
            [
                MLP(
                    dim_feedforward,
                    dim_feedforward,
                    dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(final_mlp_layers - 1)
            ]
        )
        self.final_mlp = nn.Sequential(*final_mlp)

        self.fc_out = nn.Linear(dim_feedforward, self.num_classes)

    def node2edge(self, edges):
        input = torch.cat([edges.src[self.feat], edges.dst[self.feat]], dim=-1)
        return {self.feat: input}

    def forward(self, g):
        """
        Input: (l, b, d)
        Output: (b, c, l, l)
        """

        # Initial set of linear layers
        g.ndata["hidden rep"] = self.initial_mlp(g.ndata["leaf features"])
        g.apply_edges(self.node2edge)  # (b, l*l, 2d)

        # All things related to NRI blocks are in here
        g.edata[self.feat] = self.pre_blocks_mlp(g.edata[self.feat])  # (b, l*l, d)
        # Skip connection to jump over all NRI blocks
        
        # Skip connection to jump over all NRI blocks
        x_global_skip = g.edata[self.feat]

        for block in self.blocks:
            block(g)

        # Global skip connection
        g = self.edge_conv_block(g)

        g.edata[self.feat] = torch.cat(
            (g.edata[self.feat], x_global_skip), dim=-1
        )  # Skip connection  # (b, l*(l-1), 2d)

        # else:
        #     x = self.mlp3(x)  # (b, l*(l-1), d)
        #     x = torch.cat((x, x_skip), dim=2)  # Skip connection  # (b, l*(l-1), 2d)
        #     x = self.mlp4(x)  # (b, l*(l-1), d)

        # Final set of linear layers
        g.edata[self.feat] = self.final_mlp(
            g.edata[self.feat]
        ) 
        

        # Final set of linear layers
        g.edata[self.feat] = self.final_mlp(
            torch.cat([g.edata[self.feat], x_global_skip], dim=-1)
        )  # Series of 2-layer ELU net per node (b, l, d)

        # Output what will be used for LCA
        g.edata["pred"] = self.fc_out(g.edata[self.feat])  # (b, l*l, c)

        return g
