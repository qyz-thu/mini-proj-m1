import os
import torch
from model import Model
from dataset import Dataset
from torch.utils.data import DataLoader
from util import load_model, get_args, get_device, set_env


@torch.no_grad()
def inference(args, dataloder, model, output_dir, DEVICE):
    f = open(output_dir, 'w')

    model = model.to(DEVICE)
    model.eval()
    state_h, state_c = model.init_state(args.sequence_length)
    state_h = state_h.to(DEVICE)
    state_c = state_h.to(DEVICE)

    i = 0
    for batch, (user_id, sequence) in enumerate(dataloder):
        sequence = sequence[:,1:].to(DEVICE)

        y_pred, (state_h, state_c) = model(sequence, (state_h, state_c))
        #y = int(torch.argmax(y_pred).data)
        #f.write('%s\n' % y)
        topk = torch.topk(y_pred, 10)[1].data[0].tolist()
        f.write('%s\n' % topk)

        i += 1

    f.close()


if __name__ == '__main__':
    set_env(kind='zf')   # kind=['ml' or 'zf']
    args = get_args()
    DEVICE = get_device()

    # data_dir = os.environ['SM_CHANNEL_EVAL']
    # model_dir = os.environ['SM_CHANNEL_MODEL']
    # in case only inference
    # model_dir = './model/'
    # output_dir = os.environ['SM_OUTPUT_DATA_DIR']
    data_path = os.path.join(args.data_dir, 'test_seq_data.txt')
    output_path = os.path.join(args.output_dir, 'output.csv')

    dataset = Dataset(data_path, max_len=args.sequence_length)
    # max_item_count = 3706     # for data_ml
    max_item_count = 21077      # for data_zf
    model = Model(args, max_item_count, DEVICE)

    loader = DataLoader(dataset, 1)

    model = load_model(model, args.model_dir)
    model = model.to(DEVICE)

    inference(args, loader, model, output_path, DEVICE)
    print('finish!')
