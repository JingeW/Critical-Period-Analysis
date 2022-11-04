import argparse
import random
import pprint
import time
import sys
import os

from datetime import timedelta
from workspace import Workspace

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from utils import Logger, AverageMeter, accuracy, calc_metrics, show_batch
from tqdm import tqdm
import pickle as pkl

parser = argparse.ArgumentParser(description='Attention Analysis')
parser.add_argument('--arch', type=str)
parser.add_argument('--lr', type=float)
parser.add_argument('--wd', type=float)
parser.add_argument('--bs', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--pretrained', type=str)
parser.add_argument('--deterministic', default=False, action='store_true')


def main(cfg):
    # global settings
    global device
    if torch.cuda.is_available():
        device = torch.device(cfg['cuda'])
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    if cfg['deterministic']:
        random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        cudnn.deterministic = True
        cudnn.benchmark = False

    # Loading Data
    # -----------------
    print('[>] Loading dataset '.ljust(64, '-'))
    # train set
    train_path = os.path.join(cfg['data_dir'], cfg['train_set'])
    train_set = datasets.ImageFolder(
        train_path,
        transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        # transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg['batch_size'], shuffle=True, 
        num_workers=cfg['workers'], pin_memory=True
    )

    # validation set
    val_path = os.path.join(cfg['data_dir'], cfg['test_set'])
    val_set = datasets.ImageFolder(
        val_path,
        transform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg['batch_size'], shuffle=True, 
        num_workers=cfg['workers'], pin_memory=True
    )
    print('[*] Loaded dataset!')

    # Create Model
    # ------------
    print('[>] Model '.ljust(64, '-'))

    if cfg['arch'] == 'vgg16':
        model = models.vgg16()
        model.classifier[6] = nn.Linear(4096, cfg['class_num'])
    elif cfg['arch'] == 'resnet50':
        model = models.resnet50()
        model.fc = nn.Linear(2048, cfg['class_num'])
    elif cfg['arch'] == 'resnet18':
        model = models.resnet18()
        model.fc = nn.Linear(512, cfg['class_num'])
    else:
        model = models.alexnet()
        model.classifier[6] = nn.Linear(4096, cfg['class_num'])

    if cfg['pretrained'] == 'baseModel':
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print('[*] Model training from scratch!')
    else:
        pretrain_path = os.path.join(cfg['base_model_dir'], cfg['pretrained'].split('_')[-1] + '.pth')
        print('[*] Pretrain model:', pretrain_path)
        model.load_state_dict(torch.load(pretrain_path))
        print('[*] Model training from base model pretrained!')

    model.to(device)

    # --------------------------------
    criterion = nn.CrossEntropyLoss().to(device)
    print('Using softmax loss...')

    # define optimier
    if cfg['optim'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=cfg['lr'], momentum=cfg['momentum']
            # lr=cfg['lr'], weight_decay=cfg['weight_decay'], momentum=cfg['momentum']
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg['lr'], weight_decay=cfg['weight_decay']
        )

    # lr scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg['gamma'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg['gamma'], verbose=1, patience=5)

    # training and evaluation
    # -----------------------
    global best_valid
    best_valid = dict.fromkeys(['acc', 'rec', 'f1', 'aucpr', 'aucroc'], 0.0)

    print('[>] Begin Training '.ljust(64, '-'))
    if cfg['pretrained'] == 'baseModel':
        lr_dict = {}
        for epoch in range(1, cfg['epochs'] + 1):
            start = time.time()
            if epoch == 1:
                show_batch(train_loader, cfg['save_path'])

            train(train_loader, model, criterion, optimizer, epoch, cfg)
            # validate(val_loader, model, criterion, epoch, cfg)
            valid_loss = validate(val_loader, model, criterion, epoch, cfg)

            # progress
            end = time.time()
            lr_dict[epoch] = optimizer.param_groups[0]["lr"]
            progress = (
                f'[*] epoch time = {end - start:.2f}s | '
                f'lr = {lr_dict[epoch]} | '
            )
            print(progress)

            # lr step
            # scheduler.step()
            scheduler.step(valid_loss)

            # best valid info
            # ---------------
            print('[>] Best Valid '.ljust(64, '-'))
            stat = (
                f'[+] acc={best_valid["acc"]:.4f}\n'
                f'[+] rec={best_valid["rec"]:.4f}\n'
                f'[+] f1={best_valid["f1"]:.4f}\n'
                f'[+] aucpr={best_valid["aucpr"]:.4f}\n'
                f'[+] aucroc={best_valid["aucroc"]:.4f}\n'
            )
            print(stat)
        with open(cfg['save_path'] + '/lr_dict.pkl', 'wb') as f:
            pkl.dump(lr_dict, f, protocol=4)
        print('[!] lr_dict saved')
    elif 'recover' in cfg['log_dir']:
        print('[!] recovery...')
        for epoch in range(int(cfg['pretrained'].split('_')[-1]) + 1, cfg['epochs'] + 1):
            start = time.time()
            if epoch == int(cfg['pretrained'].split('_')[-1]) + 1:
                show_batch(train_loader, cfg['save_path'])

            train(train_loader, model, criterion, optimizer, epoch, cfg)
            valid_loss = validate(val_loader, model, criterion, epoch, cfg)

            # progress
            end = time.time()
            progress = (
                f'[*] epoch time = {end - start:.2f}s | '
                f'lr = {optimizer.param_groups[0]["lr"]} | '
            )
            print(progress)

            # lr step
            scheduler.step(valid_loss)

            # best valid info
            # ---------------
            print('[>] Best Valid '.ljust(64, '-'))
            stat = (
                f'[+] acc={best_valid["acc"]:.4f}\n'
                f'[+] rec={best_valid["rec"]:.4f}\n'
                f'[+] f1={best_valid["f1"]:.4f}\n'
                f'[+] aucpr={best_valid["aucpr"]:.4f}\n'
                f'[+] aucroc={best_valid["aucroc"]:.4f}\n'
            )
            print(stat)
    else:
        lr_dict_path = os.path.join(cfg['base_model_dir'], 'lr_dict.pkl')
        with open(lr_dict_path, 'rb') as f:
            lr_dict = pkl.load(f)
        for epoch in range(int(cfg['pretrained'].split('_')[-1]) + 1, cfg['epochs'] + 1):
            start = time.time()
            if epoch == int(cfg['pretrained'].split('_')[-1]) + 1:
                show_batch(train_loader, cfg['save_path'])

            adjust_learning_rate(optimizer, epoch, lr_dict)
            train(train_loader, model, criterion, optimizer, epoch, cfg)
            validate(val_loader, model, criterion, epoch, cfg)

            # progress
            end = time.time()
            progress = (
                f'[*] epoch time = {end - start:.2f}s | '
                f'lr = {optimizer.param_groups[0]["lr"]} | '
            )
            print(progress)

            # best valid info
            # ---------------
            print('[>] Best Valid '.ljust(64, '-'))
            stat = (
                f'[+] acc={best_valid["acc"]:.4f}\n'
                f'[+] rec={best_valid["rec"]:.4f}\n'
                f'[+] f1={best_valid["f1"]:.4f}\n'
                f'[+] aucpr={best_valid["aucpr"]:.4f}\n'
                f'[+] aucroc={best_valid["aucroc"]:.4f}\n'
            )
            print(stat)


def adjust_learning_rate(optimizer, epoch, lr_dict):
    lr = lr_dict[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch, cfg):
    losses = AverageMeter()
    accs = AverageMeter()
    y_pred, y_true, y_scores = [], [], []

    # switch to train mode
    model.train()

    with tqdm(total=int(len(train_loader.dataset) / cfg['batch_size'])) as pbar:
        for i, (images, target) in enumerate(train_loader):

            images = images.to(device)
            target = target.to(device)

            # compute output
            out = model(images)
            loss = criterion(out, target)

            # measure accuracy and record loss
            acc, pred = accuracy(out, target)
            losses.update(loss.item(), images.size(0))
            accs.update(acc.item(), images.size(0))

            # collect for metrics
            y_pred.append(pred)
            y_true.append(target)
            y_scores.append(out.data)

            # compute grads + opt step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # progressbar
            pbar.set_description(f'TRAINING [{epoch:03d}/{cfg["epochs"]}]')
            pbar.set_postfix({'Loss': losses.avg, 'acc': accs.avg})
            pbar.update(1)

    metrics = calc_metrics(y_pred, y_true, y_scores)
    progress = (
        f'[-] TRAIN [{epoch:03d}/{cfg["epochs"]}] | '
        f'Ls={losses.avg:.4f} | '
        f'acc={accs.avg:.4f} | '
        f'rec={metrics["rec"]:.4f} | '
        f'f1={metrics["f1"]:.4f} | '
        f'aucpr={metrics["aucpr"]:.4f} | '
        f'aucroc={metrics["aucroc"]:.4f}'
    )
    print(progress)
    write_log(losses, accs.avg, metrics, epoch, tag='train')


def validate(valid_loader, model, criterion, epoch, cfg):
    losses = AverageMeter()
    accs = AverageMeter()
    y_pred, y_true, y_scores = [], [], []

    # switch to evaluate mode
    model.eval()

    with tqdm(total=int(len(valid_loader.dataset) / cfg['batch_size'])) as pbar:
        with torch.no_grad():
            for i, (images, target) in enumerate(valid_loader):

                images = images.to(device)
                target = target.to(device)

                # compute output
                out = model(images)
                loss = criterion(out, target)

                # measure accuracy and record loss
                acc, pred = accuracy(out, target)
                losses.update(loss.item(), images.size(0))

                accs.update(acc.item(), images.size(0))

                # collect for metrics
                y_pred.append(pred)
                y_true.append(target)
                y_scores.append(out.data)

                # progressbar
                pbar.set_description(f'VALIDATING [{epoch:03d}/{cfg["epochs"]}]')
                pbar.update(1)

    metrics = calc_metrics(y_pred, y_true, y_scores)
    progress = (
        f'[-] VALID [{epoch:03d}/{cfg["epochs"]}] | '
        f'Ls={losses.avg:.4f} | '
        f'acc={accs.avg:.4f} | '
        f'rec={metrics["rec"]:.4f} | '
        f'f1={metrics["f1"]:.4f} | '
        f'aucpr={metrics["aucpr"]:.4f} | '
        f'aucroc={metrics["aucroc"]:.4f}'
    )
    print(progress)

    # save model checkpoints
    if epoch % 5 == 0:
        save_checkpoint(epoch, model, cfg)

    best_valid['acc'] = max(best_valid['acc'], accs.avg)
    best_valid['rec'] = max(best_valid['rec'], metrics['rec'])
    best_valid['f1'] = max(best_valid['f1'], metrics['f1'])
    best_valid['aucpr'] = max(best_valid['aucpr'], metrics['aucpr'])
    best_valid['aucroc'] = max(best_valid['aucroc'], metrics['aucroc'])
    write_log(losses, accs.avg, metrics, epoch, tag='valid')
    return losses.avg

def write_log(losses, acc, metrics, e, tag='set'):
    # tensorboard
    writer.add_scalar(f'Loss/{tag}', losses.avg, e)
    writer.add_scalar(f'acc/{tag}', acc, e)
    writer.add_scalar(f'rec/{tag}', metrics['rec'], e)
    writer.add_scalar(f'f1/{tag}', metrics['f1'], e)
    writer.add_scalar(f'aucpr/{tag}', metrics['aucpr'], e)
    writer.add_scalar(f'aucroc/{tag}', metrics['aucroc'], e)

def save_checkpoint(epoch, model, cfg):
    torch.save(model.state_dict(), os.path.join(cfg['save_path'], str(epoch) + '.pth'))


if __name__ == '__main__':

    # setting up workspace
    args = parser.parse_args()
    workspace = Workspace(args)
    cfg = workspace.config

    # setting up writers
    global writer
    writer = SummaryWriter(cfg['save_path'])
    sys.stdout = Logger(os.path.join(cfg['save_path'], 'log.log'))

    # print finalized parameter config
    print('[>] Configuration '.ljust(64, '-'))
    pp = pprint.PrettyPrinter(indent=2)
    print(pp.pformat(cfg))

    # -----------------
    start = time.time()
    main(cfg)
    end = time.time()
    # -----------------

    print('\n[*] Fini! '.ljust(64, '-'))
    print(f'[!] total time = {timedelta(seconds=end - start)}s')
    sys.stdout.flush()
