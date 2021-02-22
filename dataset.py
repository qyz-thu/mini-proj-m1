import torch
import pandas as pd
import json
import dgl
from collections import Counter
from util import get_device


# 'train_data.txt'
class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_name, adj_list_path, max_len=128, is_test=False):
        self.file_name = file_name
        self.max_len = max_len + 1  # for GT in test set
        # self.max_len = max_len
        self.is_test = is_test

        self.dataset = self.load_dataset()
        self.dataset = self.dataset.reset_index()
        self.adj_list = self.load_adj_list(adj_list_path)

        self.dataset.set_index("idx", inplace=True)  # to use index for comparing timestamp
        self.count = len(self.dataset)

        self.uniq_items = [i for i in range(self.count)]
        self.nuniq_items = len(self.uniq_items)

    def str_to_list_max_len(self, seq):
        _seq = seq.split(',')
        len_seq = len(_seq)
        if len_seq < self.max_len:
            ls_zero = [0 for i in range(self.max_len - len_seq)]
            ls_zero.extend(_seq)
            _seq = ls_zero
        else:
            _seq = _seq[-self.max_len:]

        return _seq

    def load_adj_list(self, path):
        with open(path) as f:
            obj = json.load(f)
        return obj

    def load_dataset(self):
        names = ['user_id', 'sequence']
        train_df = pd.read_csv(self.file_name, delimiter=':', names=names)
        train_df['idx'] = range(0, len(train_df))
        train_df['sequence'] = train_df['sequence'].map(self.str_to_list_max_len).apply(lambda x: list(map(int, x)))
        # print(train_df.head())
        return train_df

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # return input_seq and gt
        user_id = self.dataset.loc[index]['user_id']
        sequence = self.dataset.loc[index]['sequence']
        target = [sequence[-1]]
        if not self.is_test:
            sequence = sequence[:-1]
        for start_index in range(len(sequence)):
            if sequence[start_index] > 0:
                break
        graph_nodes = sequence[start_index:]
        # graph_size = len(graph_nodes)
        edges = [[], []]
        edge_type = []
        for i in range(len(graph_nodes)):
            for j in range(i + 1, len(graph_nodes)):
                if str(graph_nodes[j]) not in self.adj_list:
                    continue
                if str(graph_nodes[i]) in self.adj_list[str(graph_nodes[j])]:
                    edges[0].extend([i, j])
                    edges[1].extend([j, i])
                    if self.adj_list[str(graph_nodes[j])][str(graph_nodes[i])] == 1:
                        edge_type.append(0)
                    elif self.adj_list[str(graph_nodes[j])][str(graph_nodes[i])] <= 5:
                        edge_type.append(1)
                    else:
                        edge_type.append(2)

        graph = dgl.graph((edges[0], edges[1]), num_nodes=len(graph_nodes))
        graph = dgl.add_self_loop(graph)
        edge_type.extend([2 for _ in range(len(graph_nodes))])

        return graph, torch.Tensor(graph_nodes).long(), torch.Tensor(target).long(), torch.Tensor(edge_type).long()


if __name__ == '__main__':
    dataset = Dataset('./data/train_data.txt', 3)
    print(dataset.uniq_items)
    print(dataset.count)
    for i in range(10):
        print(i, dataset[i])
