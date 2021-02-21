import torch
import os
import argparse
import tarfile
import dgl


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=1)
    
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sequence-length', type=int, default=64)
    parser.add_argument('--encoder_type', default='lstm')

    parser.add_argument('--lstm-layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lstm-size', type=int, default=256)
    parser.add_argument('--embedding-dim', type=int, default=256)

    parser.add_argument('--num_head', type=int, default=1)
    parser.add_argument('--transformer_layer', type=int, default=1)
    parser.add_argument('--linear_hidden_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--gnn_layers', type=int, default=1)

    parser.add_argument('--data_dir', default='./data/')
    parser.add_argument('--eval_dir', default='./data-zf')
    parser.add_argument('--graph_dir', default='./data/')
    parser.add_argument('--model_dir', default='./model/')
    parser.add_argument('--output_dir', default='./output/')
    parser.add_argument('--validate', action='store_true')

    args = parser.parse_args()
    return args


def get_device():
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    return device


def set_env(root_path='.', kind='ml'):
    # for train
    if 'SM_CHANNEL_TRAIN' not in os.environ:
        os.environ['SM_CHANNEL_TRAIN'] = '%s/data-%s/' % (root_path, kind)
    if 'SM_MODEL_DIR' not in os.environ:
        os.environ['SM_MODEL_DIR'] = '%s/model/' % root_path

    # for inference
    if 'SM_CHANNEL_EVAL' not in os.environ:
        os.environ['SM_CHANNEL_EVAL'] = '%s/data-%s/' % (root_path, kind)
    if 'SM_CHANNEL_MODEL' not in os.environ:
        os.environ['SM_CHANNEL_MODEL'] = '%s/model/' % root_path
    if 'SM_OUTPUT_DATA_DIR' not in os.environ:
        os.environ['SM_OUTPUT_DATA_DIR'] = '%s/output/' % root_path


def save_model(model, model_dir):
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), path)


def load_model(model, model_dir):
    tarpath = os.path.join(model_dir, 'model.tar.gz')
    if os.path.exists(tarpath):
        tar = tarfile.open(tarpath, 'r:gz')
        tar.extractall(path=model_dir)
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path))
    return model


def collate(item):
    """Customized collate function for data loader"""
    graphs = [i[0] for i in item]
    graph_nodes = [i[1] for i in item]
    target = [i[2] for i in item]
    batched_graph_nodes = torch.cat(graph_nodes, dim=0)
    batched_graph = dgl.batch(graphs)
    batched_target = torch.stack(target, dim=0)

    return batched_graph, batched_graph_nodes, batched_target

