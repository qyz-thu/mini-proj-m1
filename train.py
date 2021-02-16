import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

from model import Model
from dataset import Dataset
from inference import inference

from util import save_model, load_model, set_env, get_device, get_args


def train_model(args, data_loaders, data_lengths, DEVICE):
    model = Model(args, data_lengths['nuniq_items'], DEVICE)

    # if torch.cuda.is_available():
    #     if torch.cuda.device_count() > 1:
    #         model = nn.DataParallel(model)

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    for epoch in range(args.max_epochs):
        print('Epoch {}/{}'.format(epoch+1, args.max_epochs),)
        epoch_loss = {}
        accuracy_sum = 0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            #state_h, state_c = model.init_state(args.sequence_length)

            state_h = model.init_state(args.sequence_length)

            for batch, (user_id, sequence) in enumerate(data_loaders[phase]):
                if phase == 'train':
                    optimizer.zero_grad()

                x = sequence[:, :-1].to(DEVICE)     # size: (batch_size, max_seq_length)
                y = sequence[:, -1].to(DEVICE)      # size: (batch_size)

                state_h = tuple([each.data for each in state_h])

                #y_pred, (state_h, state_c) = model(x, (state_h, state_c), DEVICE)
                y_pred, state_h = model(x, state_h)     # y_pred size: (batch_size, item_count)

                if phase == 'val':
                    pred_id = torch.argmax(y_pred, dim=1)
                    accuracy = torch.mean((pred_id == y).float())
                    accuracy_sum += accuracy

                loss = criterion(y_pred.float(), y.squeeze())

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data

            epoch_loss[phase] = running_loss / data_lengths[phase]
            if phase == 'val':
                print('{} Train loss: {:.4f} Val loss: {:.4f}'.format(phase, epoch_loss['train'], epoch_loss['val']))
                accuracy_sum /= batch
                print('validation accuracy: %.4f' % accuracy_sum)

    return model


def test_inference():
    data_dir = os.environ['SM_CHANNEL_EVAL']
    output_dir = os.environ['SM_OUTPUT_DATA_DIR']
    data_path = os.path.join(data_dir, 'train_data.txt')
    dataset = Dataset(data_dir, max_len=args.sequence_length)
    tr_dl = torch.utils.data.DataLoader(dataset, 1)
    model = Model(args, dataset.nuniq_items, DEVICE)
    #model_dir = 'output/model.pth'
    print("model_dir=", model_dir)

    model = load_model(model, model_dir)
    model = model.to(DEVICE)

    inference(args, tr_dl, model, output_dir, DEVICE)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    set_env(kind='zf')   # kind=['ml' or 'zf']
    args = get_args()
    data_dir = os.environ['SM_CHANNEL_TRAIN']
    model_dir = os.environ['SM_MODEL_DIR']
    DEVICE = get_device()
    if not os.path.exists('model'):
        os.makedirs('model')
    data_path = os.path.join(data_dir, 'train_data.txt')

    dataset = Dataset(data_path, max_len=args.sequence_length)
    lengths = [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)]
    train_data, val_data = torch.utils.data.dataset.random_split(dataset, lengths)
    train_loader = DataLoader(train_data, args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_data, args.batch_size)
    data_loaders = {"train": train_loader, "val": val_loader}
    # data_lengths = {"train": len(train_loader), "val": len(val_loader), "nuniq_items": dataset.nuniq_items}
    data_lengths = {"train": len(train_loader), "val": len(val_loader), "nuniq_items": 21077}   # 21077 items
    print('training on ', DEVICE)

    model = train_model(args, data_loaders, data_lengths, DEVICE)
    save_model(model, model_dir)

    # test_inference(args, DEVICE)
