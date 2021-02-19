import random
import os
import json
random.seed(2021)


def split_training_set(data_path, out_path, ratio=0.1):
    """
    Split the training set into train & validation
    """
    with open(data_path) as f:
        data = f.readlines()
    random.shuffle(data)
    valid_num = int(len(data) * ratio)
    with open(os.path.join(out_path, 'train.txt'), 'w') as train_writer, open(os.path.join(out_path, 'valid.txt'), 'w') as val_writer:
        for i, line in enumerate(data):
            if i < valid_num:
                val_writer.write(line)
            else:
                train_writer.write(line)


def augment_training_file(data_path, out_path, seq_len=64, win_size=32):
    """
    Split the long sequences in training data into multiple data points
    """
    with open(data_path) as f, open(out_path, 'w') as writer:
        for line in f:
            tokens = line.strip().split(':')
            user_id = tokens[0]
            item_id = tokens[1].split(',')
            index = 0
            while index < len(item_id):
                new_id = item_id[index: index + seq_len]
                writer.write(user_id + ':' + ','.join(new_id) + '\n')
                index += win_size
            # writer.write(user_id + ':' + ','.join(item_id[index:]) + '\n')


def get_adj_list(data_path, out_path):
    """
    Create the adjacency list for item graph. Items next to each other are considered connected in the graph.
    """
    adj_list = dict()
    with open(data_path) as f, open(out_path, 'w') as writer:
        for line in f:
            tokens = line.strip().split(':')
            item_id = tokens[1].split(',')
            for i in range(len(item_id)):
                item = item_id[i]
                if item not in adj_list:
                    adj_list[item] = dict()
                if i == 0:
                    continue
                prev_item = item_id[i - 1]
                if item not in adj_list[prev_item]:
                    adj_list[prev_item][item] = 0
                adj_list[prev_item][item] += 1
                if prev_item not in adj_list[item]:
                    adj_list[item][prev_item] = 0
                adj_list[item][prev_item] += 1
        json.dump(adj_list, writer)


def main():
    # split_training_set('data/train_data.txt', 'data')
    # augment_training_file('data/train.txt', 'data/aug_train.txt', win_size=63)
    get_adj_list('data/train.txt', 'data/adj_list.txt')


main()
