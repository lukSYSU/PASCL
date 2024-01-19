import torch.nn as nn
import torch.nn.functional as F

from .gatv3conv import GATv3Conv

from .mlp import MLP


class GAT(nn.Module):
    """
    Graph attention network.
    """

    def __init__(
        self,
        num_layers,
        input_size,
        num_hidden,
        output_size,
        num_heads,
        activation=nn.SiLU,
        dropout=0.0,
        norm_layer=None,
        residual=True,
        separate_values=False,
        dense_update=False,
        return_pooled=False,
        input_name="embd_tracks",
    ):

        super(GAT, self).__init__()

        # set member variables
        self.input_name = input_name
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        # assert num_hidden == output_size, "Need to have a constant dimension through the network for now"
        # hidden layers
        self.gat_layers.append(
                GATv3Conv(
                    in_feats=input_size,
                    out_feats=num_hidden,
                    num_heads=num_heads,
                    activation=activation,
                    feat_drop=dropout,
                    norm_layer=norm_layer,
                    residual=residual,
                    separate_values=separate_values,
                    dense_update=dense_update,
                    return_pooled=return_pooled,
                )
            )
        for l in range(num_layers - 1):
            self.gat_layers.append(
                GATv3Conv(
                    in_feats=num_hidden,
                    out_feats=num_hidden,
                    num_heads=num_heads,
                    activation=activation,
                    feat_drop=dropout,
                    norm_layer=norm_layer,
                    residual=residual,
                    separate_values=separate_values,
                    dense_update=dense_update,
                    return_pooled=return_pooled,
                )
            )
        # output is quite big otherwise
        self.final_dense_node_update = MLP(
            in_feat=num_hidden,
            hidden_feat=num_hidden,
            out_feat=output_size,
            batchnorm=False,
            dropout=dropout,
        )

        # multi-layer attention pooling
        # self.weighted_sum = nn.Sequential(
        #    nn.Linear(num_layers, 1),
        #    nn.Sigmoid()
        # )

    def forward(self, g):
        """
        Forward pass through the graph network.
        """

        # get node data
        h = g.ndata[self.input_name]

        # pooled_layers = []

        # for each layer
        for l in range(self.num_layers):

            # get new node representation
            h = self.gat_layers[l](g, h)

            # if you want to get the pooled output from this layer:
            # h, pooled = self.gat_layers[l](g, h)#.flatten(1)
            # pooled_layers.append(pooled)

        # take a learned sum of each pooled layer
        # pooled_layers = torch.stack(pooled_layers, dim=2)
        # weighted_pooled_layers = self.weighted_sum(pooled_layers).squeeze()
        return self.final_dense_node_update(h)  # , weighted_pooled_layers
