import numpy as np
import argparse
import os
import json
from tqdm import tqdm
import torchvision

import torch
from torch.optim import Adam, SGD
from torch.nn import NLLLoss, CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

import torchvision.transforms as transforms

from torchvision.models.alexnet import alexnet
from trainer import Trainer

from utils import save_model, load_model

# parse args
parser = argparse.ArgumentParser()

parser.add_argument('--logdir', type=str, default='logs/train_logs/')
parser.add_argument('--chkpdir', type=str, default='chkp/')
parser.add_argument('--chkpname', type=str, default=None)

parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=32)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--multi_gpu', action='store_true')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor()])
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.ImageFolder(
    root='../faces/train',
    transform=transform,
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.train_batch_size,
    num_workers=2,
    shuffle=True
)

test_dataset = torchvision.datasets.ImageFolder(
    root='../faces/test',
    transform=transform,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.test_batch_size,
    num_workers=2,
    shuffle=True
)

print('Number of batches for train - {}.'.format(len(train_loader)))
print('Number of batches for test - {}.'.format(len(test_loader)))
print('Train batch size - {}.'.format(args.train_batch_size))
print('Test batch size - {}.'.format(args.test_batch_size))

# create model
if args.chkpname == None:
    # initizalize model
    model = alexnet(pretrained=True)

    if device.type == 'cuda':
        print('CUDA device will be used.')
        model = model.cuda()

    if args.multi_gpu:
        print('Multi-gpu support is enabled.')
        model = torch.nn.DataParallel(model)
        print ('Using {} gpus.'.format(torch.cuda.device_count()))

    # set optimizer
    lr = args.lr
    optimizer = Adam(model.parameters(), lr)

    initial_epoch = 0
    num_updates = 0

else:
    # load from checkpoint
    state = load_model(args.chkpdir, args.chkpname)
    model = alexnet()

    if device.type == 'cuda':
        print('CUDA device will be used.')
        model = model.cuda()

    if args.multi_gpu:
        print('Multi-gpu support is enabled.')
        model = torch.nn.DataParallel(model)
        print ('Using {} gpus.'.format(torch.cuda.device_count()))

    model.load_state_dict(state['model'])

    # set optimizer
    lr = args.lr
    optimizer = Adam(model.parameters(), lr)
    optimizer.load_state_dict(state['optimizer'])

    initial_epoch = state['epoch']
    num_updates = state['iter']

# set criterion
criterion = CrossEntropyLoss()

# set summary writer
writer = SummaryWriter(args.logdir)

# create Trainer
trainer = Trainer(model, optimizer, criterion, writer, num_updates, device, args.multi_gpu)

# train and evaluate
for epoch in range(initial_epoch, args.num_epochs):

    # train loop
    train_loss = 0
    train_accuracy = 0
    with tqdm(ascii=True, leave=False,
              total=len(train_loader), desc='Epoch {}'.format(epoch)) as bar:

        for batch in train_loader:
            images, labels = batch
            images = images.cuda()
            labels = labels.cuda()

            loss, accuracy = trainer.train_step((images, labels))

            num_batches = len(train_loader)
            train_loss += loss.item() / num_batches
            train_accuracy += accuracy.item() / num_batches

            bar.postfix = 'train loss - {:.5f}, train acc - {:.5f}, lr - {:.5f}'.format(
                                                                                        loss,
                                                                                        accuracy,
                                                                                        lr
                                                                                        )
            bar.update()

            trainer.writer.add_scalars('iter_loss/loss', {'train' : loss.item()}, trainer.num_updates)
            trainer.writer.add_scalars('iter_score/accuracy', {'train' : accuracy.item()}, trainer.num_updates)

    # log train stats
    trainer.writer.add_scalars('epoch_loss/loss', {'train' : train_loss}, trainer.num_updates)
    trainer.writer.add_scalars('epoch_score/accuracy', {'train' : train_accuracy}, trainer.num_updates)

    # freed memory
    torch.cuda.empty_cache()

    # test loop
    test_loss = 0
    test_accuracy = 0
    with tqdm(ascii=True, leave=False,
              total=len(test_loader), desc='Epoch {}'.format(epoch)) as bar:

        for batch in test_loader:
            images, labels = batch
            images = images.cuda()
            labels = labels.cuda()

            loss, accuracy = trainer.test_step((images, labels))

            num_batches = len(test_loader)
            test_loss += loss.item() / num_batches
            test_accuracy += accuracy.item() / num_batches

    # log test stats
    trainer.writer.add_scalars('epoch_loss/loss', {'test' : test_loss}, trainer.num_updates)
    trainer.writer.add_scalars('epoch_score/accuracy', {'test' : test_accuracy}, trainer.num_updates)

    # freed memory
    torch.cuda.empty_cache()

    # save model
    save_model(trainer.model, trainer.optimizer, epoch, trainer.num_updates, args.chkpdir)
