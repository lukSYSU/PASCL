# This is the MLP layer as given by:
# https://github.com/ethanfetaya/NRI
import torch.nn as nn
import torch.nn.functional as F
import torch


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, in_feat, hidden_feat, out_feat, dropout, batchnorm=True):
        super(MLP, self).__init__()

        self.batchnorm = batchnorm

        self.fc1 = nn.Linear(in_feat, hidden_feat)
        self.fc2 = nn.Linear(hidden_feat, out_feat)
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_feat, momentum=0.1, track_running_stats=True)
            # self.bn = nn.BatchNorm1d(n_out, momentum=0.1, track_running_stats=False)  # Use this to overfit
        self.dropout_prob = dropout

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm_layer(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        """
        Input: (b, l, c)
        Output: (b, l, d)
        """
        # Input shape: [num_sims, num_things, num_features]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x = F.elu(self.fc1(inputs).to(device))  # (b, l, d)

        x = F.dropout(x, self.dropout_prob, training=self.training)  # (b, l, d)
        x = F.elu(self.fc2(x)).to(device)  # (b, l, d)
        return self.bn(x) if self.batchnorm else x  # (b, l, d)
