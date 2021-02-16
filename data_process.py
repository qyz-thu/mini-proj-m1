
def split_training_file(data_path, out_path, seq_len=64, win_size=32):
    """
    Split the long sequences in training data into multiple data points
    """
    with open(data_path) as f, open(out_path, 'w') as writer:
        for line in f:
            tokens = line.strip().split(':')
            user_id = tokens[0]
            item_id = tokens[1].split(',')
            index = 0
            while index + seq_len < len(item_id):
                new_id = item_id[index: index + seq_len]
                writer.write(user_id + ':' + ','.join(new_id) + '\n')
                index += win_size
            writer.write(user_id + ':' + ','.join(item_id[index:]) + '\n')


def main():
    split_training_file('data/train_data.txt', 'data/aug_train_data.txt')


main()
