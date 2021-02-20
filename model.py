import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import os


class Model(nn.Module):
    def __init__(self, args, n_items, DEVICE):
        super(Model, self).__init__()
        self.args = args
        self.lstm_size = args.lstm_size
        self.embedding_dim = args.embedding_dim
        self.num_layers = args.lstm_layers
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

    def forward(self, x, prev_state):
        embed = self.embedding(x) #x[256,128], embed[256,128]
        embed = embed.to(self.DEVICE)

        output, state = self.lstm(embed, prev_state)  #output[256,128,128]
        logits = self.fc(output)  #output[256,128,3706]

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

    def forward(self, x):
        embed = self.embedding(x)
        embed = embed.to(self.device)
        out = self.transformer(embed)
        logits = self.fc(out)

        return logits[:, -1, :]
