import os
import torch
from model import Model, RecTrans
from dataset import Dataset
from torch.utils.data import DataLoader
from util import load_model, get_args, get_device, set_env, collate
import dgl


@torch.no_grad()
def inference(args, dataloder, model, output_dir, DEVICE, encoder='lstm'):
    f = open(output_dir, 'w')

    model = model.to(DEVICE)
    model.eval()
    if encoder == 'lstm':
        state_h, state_c = model.init_state(args.sequence_length)
        state_h = state_h.to(DEVICE)
        state_c = state_h.to(DEVICE)

    i = 0
    for batch, (graph, graph_nodes, _, edge_type) in enumerate(dataloder):
        graph_nodes = graph_nodes.to(DEVICE)
        graph = graph.to(DEVICE)
        edge_type = edge_type.to(DEVICE)

        if encoder == 'lstm':
            y_pred, (state_h, state_c) = model(graph, graph_nodes, (state_h, state_c), edge_type)
        else:
            y_pred = model(graph, graph_nodes, edge_type)
        topk = torch.topk(y_pred, 10)[1].data[0].tolist()
        f.write('%s\n' % topk)

        i += 1

    f.close()


if __name__ == '__main__':
    set_env(kind='zf')   # kind=['ml' or 'zf']
    args = get_args()
    DEVICE = get_device()
    encoder = args.encoder_type

    data_dir = os.environ['SM_CHANNEL_EVAL']
    # model_dir = os.environ['SM_CHANNEL_MODEL']
    # in case only inference
    model_dir = './model/'
    output_dir = os.environ['SM_OUTPUT_DATA_DIR']
    data_path = os.path.join(data_dir, 'test_seq_data.txt')
    # data_path = os.path.join(data_dir, 'train_data.txt')
    output_path = os.path.join(output_dir, 'output.csv')
    graph_path = os.path.join(args.graph_dir, 'adj_list.txt')

    dataset = Dataset(data_path, graph_path, max_len=args.sequence_length, is_test=True)
    # max_item_count = 3706     # for data_ml
    max_item_count = 21077      # for data_zf
    if encoder == 'lstm':
        model = Model(args, max_item_count, DEVICE)
    else:
        assert encoder == 'transformer'
        model = RecTrans(args, max_item_count, DEVICE)

    loader = DataLoader(dataset, 1, collate_fn=collate)

    model = load_model(model, model_dir)
    model = model.to(DEVICE)

    inference(args, loader, model, output_path, DEVICE, encoder)
    print('finish!')
