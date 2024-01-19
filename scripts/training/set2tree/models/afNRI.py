import dgl
import torch
from torch import nn
from torch import Tensor

from ..layers import MLP, NRI_block
from ..utils.data_utils import construct_rel_recvs, construct_rel_sends

#set2tree fNRI
class afNRIModel(nn.Module):
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
        lambda_amp = 0.1,
        nblocks=1,
        dim_feedforward=128,
        initial_mlp_layers=1,
        block_additional_mlp_layers=1,
        dropout=0.3,
        factorised=1,# factorised module, 0: original NRI, 1: factorised using MLP, 2:factorised using NRI block
        K = 10,
        factor=True,
        tokenize=None,
        embedding_dims=None,
        # batchnorm=True,
        symmetrize=True,
        **kwargs,
    ):
        super().__init__()

        assert dim_feedforward % 2 == 0, "dim_feedforward must be an even number"

        self.num_classes = num_classes
        self.factor = factor
        self.tokenize = tokenize    #Whether to use NRI blocks at all (useful for benchmarking)
        self.symmetrize = symmetrize
        self.block_additional_mlp_layers = block_additional_mlp_layers
        self.factorised = factorised
        self.K = K
        self.lambda_amp = lambda_amp
        # self.max_leaves = max_leaves

        # Set up embedding for tokens and adjust input dims
        if self.tokenize is not None:
            assert (embedding_dims is not None) and isinstance(
                embedding_dims, int
            ), "embedding_dims must be set to an integer is tokenize is given"

            # Initialise the embedding layers, ignoring pad values
            self.embed = nn.ModuleDict({})
            for idx, n_tokens in self.tokenize.items():
                # NOTE: This assumes a pad value of 0 for the input array x
                self.embed[str(idx)] = nn.Embedding(
                    n_tokens, embedding_dims, padding_idx=0
                )

            # And update the infeatures to include the embedded feature dims and delete the original, now tokenized feats
            infeatures = infeatures + (len(self.tokenize) * (embedding_dims - 1))
            print(f"Set up embedding for {len(self.tokenize)} inputs")

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

        # MLP to reduce feature dimensions from first Node2Edge before blocks begin
        self.pre_blocks_mlp = MLP(
            dim_feedforward * 2,
            dim_feedforward,
            dim_feedforward,
            dropout=dropout,
        )

        if self.factor:
            # MLPs within NRI blocks
            # The blocks have minimum 1 MLP layer, and if specified they add more with a skip connection
            # List of blocks
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
            print("Using factor graph MLP encoder.")
        else:
            self.mlp3 = MLP(
                dim_feedforward,
                dim_feedforward,
                dim_feedforward,
                dropout=dropout,
            )
            self.mlp4 = MLP(
                dim_feedforward * 2,
                dim_feedforward,
                dim_feedforward,
                dropout=dropout,
            )
            print("Using MLP encoder.")

        # Final linear layers as requested
        # self.final_mlp = nn.Sequential(*[MLP(dim_feedforward, dim_feedforward, dim_feedforward, dropout, batchnorm) for _ in range(final_mlp_layers)])



        # factorised module
        #0: original NRI    1: factorised using MLP     2:factorised using NRI block
        if self.factorised == 0:
            self.factorised_block = MLP(
                # dim_feedforward * 2,
                dim_feedforward,
                dim_feedforward,
                dim_feedforward,
                dropout=dropout,
                batchnorm=True,
            )
            self.fc_out = nn.ModuleList([nn.Linear(dim_feedforward, self.num_classes)])
        elif self.factorised == 1:
            self.factorised_block = nn.ModuleList(
                [
                    MLP(
                        # dim_feedforward * 2,
                        dim_feedforward,
                        dim_feedforward,
                        dim_feedforward,
                        dropout=dropout,
                        batchnorm=True,
                    )
                    for _ in range(self.num_classes)
                ]
            )
            self.fc_out = nn.ModuleList([nn.Linear(dim_feedforward, self.num_classes) for _ in range(self.num_classes)])
        elif self.factorised == 2:
            self.factorised_block = nn.ModuleList(
                [
                    NRI_block(
                        dim_feedforward,
                        dim_feedforward,
                        block_additional_mlp_layers,
                        dropout=dropout,
                    )
                    for _ in range(self.num_classes)

                ]
            )
            self.fc_out = nn.ModuleList([nn.Linear(dim_feedforward, self.num_classes) for _ in range(self.num_classes)])
        else:
            raise ValueError('factorised num is not valid, must be 0, 1, or 2')

        self.feat = "hidden rep"

    def node2edge(self, edges):
        input = torch.cat([edges.src[self.feat], edges.dst[self.feat]], dim=-1)
        return {self.feat: input}

    ## Adaptive massage passing
    def adaptive_global_skip(self, x, hh, K, g):
        lambda_amp = self.lambda_amp
        gamma = 1 / (2 * (1 - lambda_amp))  ## or simply gamma = 1
        # print("y:",y.shape)
        # print("hh:",hh.shape)
        for k in range(K):
            y = x - gamma * 2 * (1 - lambda_amp) * self.compute_LX(g)
            x = hh +self.proximal_L21(x = y - hh, lambda_ = gamma * lambda_amp)
        return x

    def proximal_L21(self, x: Tensor, lambda_):
        row_norm = torch.norm(x, p=2, dim=1)
        score = torch.clamp(row_norm - lambda_, min=0)
        index = torch.where(row_norm > 0)  # Deal with the case when the row_norm is 0
        score[index] = score[index] / row_norm[index]  # score is the adaptive score in Equation (14)
        # print("score:",score)
        return score.unsqueeze(1) * x

    def compute_LX(self, g):
        y = g.edata[self.feat]
        for block in self.blocks:
            block(g)
        x = y - g.edata[self.feat]
        return x


    def forward(self, g):
        """
        Input: (l, b, d)
        Output: (b, c, l, l)
        """

        # Input shape: [batch, num_atoms, num_timesteps, num_dims]
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        # x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        # Need to match expected shape
        # TODO should batch_first be a dataset parameter?

        # # Create embeddings and merge back into x
        # # TODO: Move mask creation to init, optimise this loop
        # if self.tokenize is not None:
        #     emb_x = []
        #     # We'll use this to drop tokenized features from x
        #     mask = torch.ones(feats, dtype=torch.bool, device=device)
        #     for idx, emb in self.embed.items():
        #         # Note we need to convert tokens to type long here for embedding layer
        #         emb_x.append(emb(x[..., int(idx)].long()))  # List of (b, l, emb_dim)
        #         mask[int(idx)] = False

        #     # Now merge the embedding outputs with x (mask has deleted the old tokenized feats)
        #     x = torch.cat([x[..., mask], *emb_x], dim=-1)  # (b, l, d + embeddings)
        #     del emb_x

        # Initial set of linear layers

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        g.ndata["hidden rep"] = self.initial_mlp(g.ndata["leaf features"]).to(device)

        g.apply_edges(self.node2edge)  # (b, l*l, 2d)

        # All things related to NRI blocks are in here
        if self.factor:
            # x = self.pre_blocks_mlp(x)  # (b, l*l, d)
            g.edata[self.feat] = self.pre_blocks_mlp(g.edata[self.feat])  # (b, l*l, d)
            # Skip connection to jump over all NRI blocks
            x_global_skip = g.edata[self.feat]











            # Global skip connection
            # g.edata[self.feat] = torch.cat(
            #     (g.edata[self.feat], x_global_skip), dim=-1
            # )  # Skip connection  # (b, l*(l-1), 2d)
            print("g:",g)
            g.edata[self.feat] = self.adaptive_global_skip(x = g.edata[self.feat], hh = x_global_skip, K = self.K ,g = g)







            # else:
        #     x = self.mlp3(x)  # (b, l*(l-1), d)
        #     x = torch.cat((x, x_skip), dim=2)  # Skip connection  # (b, l*(l-1), 2d)
        #     x = self.mlp4(x)  # (b, l*(l-1), d)



        if self.factorised == 0:
            g.edata[self.feat] = self.factorised_block(g.edata[self.feat])
            g.edata["pred"] = self.fc_out[0](g.edata[self.feat])
        elif self.factorised == 1:
            y_list = []
            for i in range(self.num_classes):
                y = self.factorised_block[i](g.edata[self.feat])
                y_list.append(self.fc_out[i](y))
            g.edata["pred"] = torch.cat(y_list, dim=-1)
        elif self.factorised == 2:
            y_list = []
            for i in range(self.num_classes):
                self.factorised_block[i](g)
                y_list.append(self.fc_out[i](g.edata[self.feat]))
            g.edata["pred"] = torch.cat(y_list, dim=-1)

        # Output what will be used for LCA
        # g.edata["pred"] = self.fc_out(g.edata[self.feat])  # (b, l*l, c)
        # print(g.edata["pred"])
        # out = x.reshape(batch, n_leaves, n_leaves, self.num_classes)

        # Change to LCA shape
        # x is currently the flattened rows of the predicted LCA but without the diagonal
        # We need to create an empty LCA then populate the off-diagonals with their corresponding values
        # out = torch.zeros((batch, n_leaves, n_leaves, self.num_classes), device=device)  # (b, l, l, c)
        # ind_upper = torch.triu_indices(n_leaves, n_leaves, offset=1)
        # ind_lower = torch.tril_indices(n_leaves, n_leaves, offset=-1)

        # Assign the values to their corresponding position in the LCA
        # The right side is just some quick mafs to get the correct edge predictions from the flattened x array
        # out[:, ind_upper[0], ind_upper[1], :] = x[:, (ind_upper[0] * (n_leaves - 1)) + ind_upper[1] - 1, :]
        # out[:, ind_lower[0], ind_lower[1], :] = x[:, (ind_lower[0] * (n_leaves - 1)) + ind_lower[1], :]

        # Need in the order for cross entropy loss
        # out = out.permute(0, 3, 1, 2)  # (b, c, l, l)

        # Symmetrize
        # if self.symmetrize:
        # out = torch.div(out + torch.transpose(out, 2, 3), 2)  # (b, c, l, l)

        return g
