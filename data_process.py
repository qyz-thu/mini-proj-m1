import random
import os
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


def main():
    split_training_set('data/train_data.txt', 'data')
    augment_training_file('data/train.txt', 'data/aug_train.txt', win_size=63)


main()
