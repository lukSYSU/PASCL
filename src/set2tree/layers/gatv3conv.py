import math

import torch
from torch import nn
import torch.nn.functional as F

from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.nn import GlobalAttentionPooling

from .mlp import MLP


class GATv3Conv(nn.Module):
    """
    Testing some updates to the gatv2conv used in GN1...
    Requies the embedding dimension to be constant at each layer.
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        activation=nn.SiLU,
        feat_drop=0.05,
        attn_drop=0.0,
        norm_layer=nn.LayerNorm,
        residual=True,
        separate_values=True,
        dense_update=False,
        return_pooled=False,
    ):
        super(GATv3Conv, self).__init__()

        self.num_heads = num_heads
        # change input size if we support jet_features
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.feats_per_head = out_feats // num_heads
        # self.attn_scale = int(math.sqrt(self.feats_per_head))
        self.residual = residual

        self.activation = activation()
        self.attn_activation = nn.SiLU()

        # linear projections of the input: todo - make these single layer MLPs
        self.linear_src = nn.Linear(self.in_feats, self.out_feats)
        self.linear_dst = nn.Linear(self.in_feats, self.out_feats)
        if separate_values:
            self.feat_val = "feat_val"
            self.linear_val = nn.Linear(self.in_feats, self.out_feats)
        else:
            self.feat_val = "feat_src"
            self.register_buffer("linear_val", None)

        # learned attention vectors
        self.attn = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, self.feats_per_head))
        )

        # dropout
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        # optional stuff
        if norm_layer:
            self.norm = norm_layer(out_feats, elementwise_affine=False)
        else:
            self.register_buffer("norm", None)

        if return_pooled:
            self.gap = GlobalAttentionPooling(nn.Linear(out_feats, 1))
        else:
            self.register_buffer("gap", None)

        if dense_update:
            self.dense = MLP(
                in_feat=in_feats,
                hidden_feat=in_feats,
                out_feat=out_feats,
                dropout=0.1,
                batchnorm=True,
            )
        else:
            self.register_buffer("dense", None)
        if in_feats != out_feats:
            self.residual_net = nn.Linear(in_feats, out_feats)
        else:
            self.residual_net = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = 1  # nn.init.calculate_gain("silu")

        nn.init.xavier_normal_(self.linear_src.weight, gain=gain)
        nn.init.constant_(self.linear_src.bias, 0)

        nn.init.xavier_normal_(self.linear_dst.weight, gain=gain)
        nn.init.constant_(self.linear_dst.bias, 0)

        if self.linear_val:
            nn.init.xavier_normal_(self.linear_val.weight, gain=gain)
            nn.init.constant_(self.linear_val.bias, 0)

        nn.init.xavier_normal_(self.attn, gain=gain)

    def forward(self, graph, feat):
        with graph.local_scope():

            # apply feature dropout and normalisation first
            if self.norm:
                feat = self.norm(feat)
            h_src = h_dst = self.feat_drop(feat)

            # apply linear layer to input features
            feat_src = self.linear_src(h_src).view(
                -1, self.num_heads, self.feats_per_head
            )
            feat_dst = self.linear_dst(h_src).view(
                -1, self.num_heads, self.feats_per_head
            )
            if self.linear_val:
                feat_val = self.linear_val(h_src).view(
                    -1, self.num_heads, self.feats_per_head
                )

            # update graph with transformed input features
            graph.srcdata.update({"feat_src": feat_src})
            graph.dstdata.update({"feat_dst": feat_dst})
            if self.linear_val:
                graph.dstdata.update({"feat_val": feat_val})

            # add or cat src and dst nodes to make edges -- in the paper they just take a sum here
            # TODO: cat earlier and then go through a single dense MLP
            graph.apply_edges(fn.u_add_v("feat_src", "feat_dst", "e"))
            # graph.apply_edges(lambda edges: {"e": torch.cat([edges.src["feat_src"], edges.dst["feat_dst"]], dim=-1)})

            # apply non-linearity to the edges
            e = self.attn_activation(graph.edata.pop("e"))

            # dot product of each node with learned attention vector
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)

            # half the heads get softmaxed, half don't
            # e_softmax, e_scale = torch.tensor_split(e, 2, dim=1)
            # e_softmax = edge_softmax(graph, e_softmax)
            # e_scale   = e_scale / self.attn_scale

            # combine to form final attention weights
            # a = torch.cat([e_softmax, e_scale], dim=1)
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing: multiply transformed node reps by attention weights
            # and sum up all the resulting weighted nodes
            graph.update_all(
                fn.u_mul_e(self.feat_val, "a", "attn_weighted_nodes"),
                fn.sum("attn_weighted_nodes", "h_updated"),
            )

            # get updated node representations by concatentating output from each head
            h_updated = graph.dstdata["h_updated"].view(h_dst.shape[0], self.out_feats)

            # residual connection
            if self.residual:
                h_updated = h_updated + self.residual_net(h_dst)

            # final activation function
            if self.activation:
                h_updated = self.activation(h_updated)

            # residual dense update
            if self.dense:
                h_updated = self.dense(h_updated) + h_updated

            # return also pooled
            if self.gap:
                pooled = self.gap(graph, h_updated)
                return h_updated, pooled

            # just return the updated reps
            else:
                return h_updated
