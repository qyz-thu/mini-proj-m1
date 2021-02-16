import torch
import os
import argparse
import tarfile


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=5)
    
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-layers', type=int, default=3)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--sequence-length', type=int, default=64)
    parser.add_argument('--lstm-size', type=int, default=256)
    parser.add_argument('--embedding-dim', type=int, default=256)

    parser.add_argument('--data_dir', default='./data/')
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
