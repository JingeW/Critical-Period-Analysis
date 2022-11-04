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
import torch.nn.functional as F
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


class resnet(nn.Module):
    def __init__(self, pretrained_model):
        super(resnet, self).__init__()

        self.net = pretrained_model
        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4
        self.avgpool = self.net.avgpool
        self.classifier = self.net.fc

    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu(o)
        o = self.maxpool(o)

        g1 = self.layer1(o)
        g2 = self.layer2(g1)
        g3 = self.layer3(g2)
        g4 = self.layer4(g3)

        o = self.avgpool(g4)
        o = torch.flatten(o, 1)
        o = self.classifier(o)

        return o, [g1, g2, g3, g4]


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
        transforms.ColorJitter(brightness=0.4 ,contrast=0.4 ,saturation=0.4),
        transforms.RandomGrayscale(p=0.2),
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
    model_S = models.resnet50()
    model_S.fc = nn.Linear(2048, cfg['class_num'])
    pretrain_path = os.path.join(cfg['base_model_dir'], cfg['pretrained'].split('_')[-1] + '.pth')
    print('[*] Pretrain model:', pretrain_path)
    model_S.load_state_dict(torch.load(pretrain_path))
    print('[*] Model training from base model pretrained!')
    model_S = resnet(model_S)
    model_S.to(device)

    # set up teacher model
    model_T = models.resnet50()
    model_T.fc = nn.Linear(2048, cfg['class_num'])
    pretrain_path_T = './logs/CASIA_WebFace_20000_0.15_final/[resnet50]_[0.01]_[0.5]_[32]_[train_crop]_[test_crop]_[baseModel]/150.pth'
    print('[*] Teacher pretrain model:', pretrain_path_T)
    model_T.load_state_dict(torch.load(pretrain_path_T))
    print('[*] Teacher model loaded!')
    model_T = resnet(model_T)
    model_T.to(device)
    model_T.eval()

    # --------------------------------
    criterion = nn.CrossEntropyLoss().to(device)
    # print('Using softmax loss...')
    # guide_criterion = nn.MSELoss().to(device)
    # print('MES loss applied to the featmap comparison...')

    # define optimier
    optimizer = torch.optim.SGD(
        model_S.parameters(), 
        lr=cfg['lr'], momentum=cfg['momentum']
        # lr=cfg['lr'], weight_decay=cfg['weight_decay'], momentum=cfg['momentum']
    )

    # lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg['gamma'], verbose=1, patience=5)

    # training and evaluation
    # -----------------------
    global best_valid
    best_valid = dict.fromkeys(['acc', 'rec', 'f1', 'aucpr', 'aucroc'], 0.0)
    print('[>] Begin Training '.ljust(64, '-'))
    lr_dict_path = os.path.join(cfg['base_model_dir'], 'lr_dict.pkl')
    with open(lr_dict_path, 'rb') as f:
        lr_dict = pkl.load(f)

    for epoch in range(int(cfg['pretrained'].split('_')[-1]) + 1, cfg['epochs'] + 1):
        start = time.time()
        if epoch == int(cfg['pretrained'].split('_')[-1]) + 1:
            show_batch(train_loader, cfg['save_path'])

        adjust_learning_rate(optimizer, epoch, lr_dict)
        train(train_loader, model_S, model_T, optimizer, epoch, cfg)
        validate(val_loader, model_S, criterion, epoch, cfg)

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

def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
    # return F.normalize(x.pow(2).sum(1).view(x.size(0), -1))

def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()
    # return abs(at(x) - at(y)).mean()

def distillation(y, teacher_scores, labels, T, alpha):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
    l_ce = F.cross_entropy(y, labels)
    return l_kl * alpha + l_ce * (1. - alpha)

def train(train_loader, model_S, model_T, optimizer, epoch, cfg):
    losses = AverageMeter()
    losses_feat = AverageMeter()
    losses_KD = AverageMeter()
    accs = AverageMeter()
    y_pred, y_true, y_scores = [], [], []

    # switch to train mode
    model_S.train()

    with tqdm(total=int(len(train_loader.dataset) / cfg['batch_size'])) as pbar:
        for i, (images, target) in enumerate(train_loader):

            images = images.to(device)
            target = target.to(device)

            # compute output
            out_S, feat_S = model_S(images)
            with torch.no_grad():
                out_T, feat_T = model_T(images)

            feat_losses = sum([at_loss(x, y) for x, y in zip(feat_S, feat_T)])  # attention loss
            KD_loss = distillation(out_S, out_T, target, cfg['temperature'], cfg['alpha'])  # (hard loss) + KD soft loss 

            # loss = cfg['beta'] * feat_losses + (1 - cfg['beta']) * KD_loss
            loss = cfg['beta'] * feat_losses + KD_loss

            # measure accuracy and record loss
            acc, pred = accuracy(out_S, target)
            losses.update(loss.item(), images.size(0))
            losses_feat.update(feat_losses.item(), images.size(0))
            losses_KD.update(KD_loss.item(), images.size(0))

            accs.update(acc.item(), images.size(0))

            # collect for metrics
            y_pred.append(pred)
            y_true.append(target)
            y_scores.append(out_S.data)

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
    write_log_new(losses, losses_feat, losses_KD, accs.avg, metrics, epoch, tag='train')

def validate(valid_loader, model_S, criterion, epoch, cfg):
    losses = AverageMeter()
    accs = AverageMeter()
    y_pred, y_true, y_scores = [], [], []

    # switch to evaluate mode
    model_S.eval()

    with tqdm(total=int(len(valid_loader.dataset) / cfg['batch_size'])) as pbar:
        with torch.no_grad():
            for i, (images, target) in enumerate(valid_loader):

                images = images.to(device)
                target = target.to(device)

                # compute output
                out, _ = model_S(images)
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
        save_checkpoint(epoch, model_S, cfg)

    best_valid['acc'] = max(best_valid['acc'], accs.avg)
    best_valid['rec'] = max(best_valid['rec'], metrics['rec'])
    best_valid['f1'] = max(best_valid['f1'], metrics['f1'])
    best_valid['aucpr'] = max(best_valid['aucpr'], metrics['aucpr'])
    best_valid['aucroc'] = max(best_valid['aucroc'], metrics['aucroc'])
    write_log(losses, accs.avg, metrics, epoch, tag='valid')
    return losses.avg

def write_log_new(losses, losses_feat, losses_KD, acc, metrics, e, tag='set'):
    # tensorboard
    writer.add_scalar(f'Loss/{tag}', losses.avg, e)
    writer.add_scalar(f'Loss_fm1/{tag}', losses_feat.avg, e)
    writer.add_scalar(f'Loss_fm2/{tag}', losses_KD.avg, e)
    writer.add_scalar(f'acc/{tag}', acc, e)
    writer.add_scalar(f'rec/{tag}', metrics['rec'], e)
    writer.add_scalar(f'f1/{tag}', metrics['f1'], e)
    writer.add_scalar(f'aucpr/{tag}', metrics['aucpr'], e)
    writer.add_scalar(f'aucroc/{tag}', metrics['aucroc'], e)

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
