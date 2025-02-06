import numpy as np
import torch
import torch.nn as nn
from skimage.segmentation import mark_boundaries
from functiongcnlocal import *
from graph_unets import *



class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)

    def forward(self, X, adj):
        h = torch.matmul(X, self.W)
        N = h.size(0)
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)\
            .view(N, -1, 2 * self.out_features)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2), negative_slope=self.alpha)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        return F.relu(h_prime)


class multiStageUnmixing(nn.Module):
    def __init__(self, col, endmember_number, band_Number):
        super(multiStageUnmixing, self).__init__()
        self.col = col
        self.endmember_number = endmember_number
        self.band_number = band_Number
        self.encoder = nn.Sequential(
            nn.Linear(band_Number*2, 128),
            # ECBAM1_blocks.MultiECBAM(128),
            nn.BatchNorm1d(128, momentum=0.9),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            # ECBAM1_blocks.MultiECBAM(64),
            nn.BatchNorm1d(64, momentum=0.9),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Linear(64, 24),
            # ECBAM1_blocks.MultiECBAM(24),
            nn.BatchNorm1d(24, momentum=0.5),
            nn.Dropout(0.25),
            nn.LeakyReLU(),

        )
        self.smooth = nn.Sequential(
            nn.Linear(24, endmember_number),
            nn.Softmax(dim=1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(endmember_number, band_Number, bias=False),
            nn.ReLU(),
        )

        self.gat = GraphAttentionLayer(in_features=band_Number, out_features=band_Number)
        self.gatunets = GraphUnet(ks=[0.9,0.8,0.7], dim=band_Number, drop_p=0.4)


    def forward(self, x, xsup, A, inputlocal, i):
        # Q.shape = torch.Size([10000, 63])


        # x1 = self.gat(xsup, A)
        # x2 = self.gat(x1, A)
        # # torch.Size([100, 198])
        x2 = self.gatunets(A, xsup)
        # torch.Size([63, 198])

        x2_extend = x2[i-1].expand_as(inputlocal)
        x3 = torch.cat((inputlocal, x2_extend), dim=1)





        x = self.encoder(x3)
        abun = self.smooth(x)
        x = self.decoder(abun)



        return abun, x