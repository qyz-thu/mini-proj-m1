import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, ReLU
from dgl.nn import GraphConv, GATConv, GatedGraphConv
import dgl
import torch.nn.functional as F
import os


class GNN(nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        if args.gnn_type == 'gcn':
            self.use_gcn = True
        else:
            assert args.gnn_type == 'gat'
            self.use_gcn = False
        self.net = nn.ModuleList()
        for i in range(args.gnn_layers):
            if self.use_gcn:
                self.net.append(GraphConv(in_feats=args.embedding_dim, out_feats=args.embedding_dim, allow_zero_in_degree=True))
            else:
                self.net.append(GATConv(in_feats=args.embedding_dim, out_feats=args.embedding_dim, num_heads=args.gat_num_head, allow_zero_in_degree=True))
            self.net.append(ReLU())

    def forward(self, graph, x):
        for i, layer in enumerate(self.net):
            if i % 2 == 0:
                x = layer(graph, x) if self.use_gcn else torch.mean(layer(graph, x), dim=1)
            else:
                x = layer(x)

        return x


class Model(nn.Module):
    def __init__(self, args, n_items, DEVICE):
        super(Model, self).__init__()
        self.args = args
        self.lstm_size = args.lstm_size
        self.embedding_dim = args.embedding_dim
        self.num_layers = args.lstm_layers
        self.batch_size = args.batch_size
        self.sequence_length = args.sequence_length
        self.DEVICE = DEVICE

        self.embedding = nn.Embedding(
            num_embeddings=n_items + 1,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=args.dropout,
        )
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.lstm_size, n_items + 1)
        if args.gnn_type == 'ggsnn':
            self.gnn = GatedGraphConv(in_feats=args.embedding_dim, out_feats=args.embedding_dim, n_steps=args.ggsnn_step, n_etypes=3)
        else:
            self.gnn = GNN(args)
        self.gnn = self.gnn.to(DEVICE)

    def forward(self, graph, graph_nodes, prev_state, edge_type=None):  # edge_type is required is use Gate Graph Conv
        graph_embed = self.embedding(graph_nodes)
        if self.args.gnn_type == 'ggsnn':
            graph_embed = self.gnn(graph, graph_embed, edge_type)
        else:
            graph_embed = self.gnn(graph, graph_embed)
        graph.ndata['h'] = graph_embed
        graphs = dgl.unbatch(graph)
        embed_list = []
        for g in graphs:
            embed = g.ndata['h']
            if embed.size()[0] >= self.sequence_length:
                p_embed = embed[:self.sequence_length, :]
            else:
                p_embed = torch.cat([torch.zeros(self.sequence_length - embed.size()[0], self.lstm_size).to(self.DEVICE), embed], dim=0)
            # padded_embed = torch.stack([padded_embed, p_embed], dim=0)
            embed_list.append(p_embed)
        padded_embed = torch.stack(embed_list, dim=0)
        # graph_embed = graph_embed.reshape(-1, self.sequence_length, self.lstm_size)  # shape: (batch_size, sequence_length, lstm_size)

        output, state = self.lstm(padded_embed, prev_state)  # output[256,128,128]
        logits = self.fc(output)  # output[256,128,3706]

        return logits[:, -1, :], state

    def init_state(self, sequence_length):
        hidden = (torch.zeros(self.num_layers, sequence_length, self.lstm_size).to(self.DEVICE),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size).to(self.DEVICE))
        return hidden


class RecTrans(nn.Module):
    def __init__(self, args, n_items, device):
        super(RecTrans, self).__init__()
        self.args = args
        self.device = device

        self.embedding = nn.Embedding(num_embeddings=n_items + 1, embedding_dim=args.embedding_dim)
        self.transformer_layer = TransformerEncoderLayer(
            d_model=args.embedding_dim,
            nhead=args.num_head,
            dim_feedforward=args.linear_hidden_size,
            dropout=args.dropout,
        )
        self.transformer = TransformerEncoder(self.transformer_layer, num_layers=args.transformer_layer)
        self.fc = nn.Linear(args.embedding_dim, n_items + 1)
        self.gnn = GNN(args)

    def forward(self, x, graph, graph_nodes):
        embed = self.embedding(x)
        embed = embed.to(self.device)
        out = self.transformer(embed)
        logits = self.fc(out)

        return logits[:, -1, :]

