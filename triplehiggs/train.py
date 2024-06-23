import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data')))

import utils
from data_loader import retrieve_dataloaders
from model import psi, LorentzNet
import torch
from torch import nn, optim
import argparse, json, time
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

parser = argparse.ArgumentParser(description='Top tagging')
parser.add_argument('--exp_name', type=str, default='', metavar='N', help='experiment_name')
parser.add_argument('--test_mode', action='store_true', default=False, help='test best model')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='input batch size for training')
parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of training epochs')
parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='number of warm-up epochs')
parser.add_argument('--c_weight', type=float, default=5e-3, metavar='N', help='weight of x model')
parser.add_argument('--seed', type=int, default=99, metavar='N', help='random seed')
parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--val_interval', type=int, default=1, metavar='N', help='how many epochs to wait before validation')
parser.add_argument('--datadir', type=str, default='./data/raw_data', metavar='N', help='data dir')
parser.add_argument('--logdir', type=str, default='./logs', metavar='N', help='folder to output logs')
parser.add_argument('--dropout', type=float, default=0.2, metavar='N', help='dropout probability')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N', help='learning rate')
parser.add_argument('--n_hidden', type=int, default=72, metavar='N', help='dim of latent space')
parser.add_argument('--n_layers', type=int, default=6, metavar='N', help='number of LGEBs')
parser.add_argument('--num_workers', type=int, default=0, metavar='N', help='number of workers for the dataloader')
parser.add_argument('--weight_decay', type=float, default=1e-2, metavar='N', help='weight decay')

args = parser.parse_args()

# Setup logging
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)
logging.basicConfig(filename=os.path.join(args.logdir, f'{args.exp_name}.log'), level=logging.INFO, format='%(asctime)s %(message)s')

def run(epoch, loader, partition):
    model.train() if partition == 'train' else model.eval()

    res = {'correct': 0, 'loss': 0, 'counter': 0, 'acc': 0}

    loader_length = len(loader)

    for i, data in enumerate(tqdm(loader, desc=f"{partition} Epoch {epoch+1}/{args.epochs}")):
        if partition == 'train':
            optimizer.zero_grad()
        
        batch_size, n_nodes, _ = data['jets_p4'].size()
        atom_positions = data['jets_p4'].view(batch_size * n_nodes, -1).to(device).to(torch.float32)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device).to(torch.float32)
        edge_mask = data['edge_mask'].reshape(batch_size * n_nodes * n_nodes, -1).to(device).to(torch.float32)
        nodes = data['nodes'].view(batch_size * n_nodes, -1).to(device).to(torch.float32)
        nodes = psi(nodes)
        edges = [a.to(device) for a in data['edges']]
        label = data['is_signal'].to(device).long()

        pred = model(scalars=nodes, x=atom_positions, edges=edges, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)

        predict = pred.max(1).indices
        correct = torch.sum(predict == label).item()
        loss = loss_fn(pred, label)

        if partition == 'train':
            loss.backward()
            optimizer.step()

        res['correct'] += correct
        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
         # Log training status every log_interval batches
        if i % args.log_interval == 0 and partition == 'train':
            current_loss = res['loss'] / res['counter']
            current_acc = res['correct'] / res['counter']
            logging.info(f"Epoch {epoch+1}/{args.epochs} - Batch {i}/{loader_length} - Loss: {current_loss:.4f} - Acc: {current_acc:.4f}")

    res['loss'] /= res['counter']
    res['acc'] = res['correct'] / res['counter']
    return res

def train():
    for epoch in range(args.epochs):
        train_res = run(epoch, dataloaders['train'], partition='train')
        logging.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_res['loss']:.4f} - Train Acc: {train_res['acc']:.4f}")
        if epoch % args.val_interval == 0:
            with torch.no_grad():
                val_res = run(epoch, dataloaders['valid'], partition='valid')
            logging.info(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {val_res['loss']:.4f} - Val Acc: {val_res['acc']:.4f}")

def test():
    best_model = torch.load(f"{args.logdir}/{args.exp_name}/best-val-model.pt", map_location=device)
    model.load_state_dict(best_model)
    with torch.no_grad():
        test_res = run(0, dataloaders['test'], partition='test')
    logging.info(f"Test - Loss: {test_res['loss']:.4f} - Acc: {test_res['acc']:.4f}")

if __name__ == "__main__":
    ### initialize args
    args = parser.parse_args()
    utils.args_init(args)

    ### set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("mps")

    ### load data
    dataloaders = retrieve_dataloaders(args.batch_size, args.num_workers, datadir=args.datadir)

    # 打印数据加载器的键
    print("Dataloaders keys:", dataloaders.keys())

    ### create model
    model = LorentzNet(n_scalar=10, n_hidden=args.n_hidden, n_class=2, dropout=args.dropout, n_layers=args.n_layers, c_weight=args.c_weight)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    ### print model and data information
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Network Size:", pytorch_total_params)
    for split, dataloader in dataloaders.items():
        print(f"{split} samples: {len(dataloader.dataset)}")

    ### optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ### lr scheduler
    base_scheduler = CosineAnnealingWarmRestarts(optimizer, 4, 2, verbose=False)
    lr_scheduler = utils.GradualWarmupScheduler(optimizer, multiplier=1, warmup_epoch=args.warmup_epochs, after_scheduler=base_scheduler)

    ### loss function
    loss_fn = nn.CrossEntropyLoss()

    if not args.test_mode:
        train()
        test()
    else:
        test()
