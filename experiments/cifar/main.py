import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchinfo import summary

from cifar10_dataset import get_cifar10_data, TensorDataset
from resnet44 import resnet44
from resnet44_p4m import p4mresnet44, gp4mresnet44, gcp4mresnet44

model_dict = {'resnet44': resnet44,
              'resnet44_p4m': p4mresnet44,
              'resnet44_gp4m': gp4mresnet44,
              'resnet44_gcp4m': gcp4mresnet44}

def train(epoch, model, criterion, optimizer, train_loader):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    return 100. * correct / len(train_loader.dataset)

def test(model, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
        test_loss += criterion(output, target).data  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def get_data(args):
    # Load whitened CIFAR10 according to
    # https://github.com/tscohen/gconv_experiments/blob/master/gconv_experiments/CIFAR10/cifar10.py
    train_data, train_labels, test_data, test_labels = get_cifar10_data(args.dataset_path,
                                                                        'train_all.npz',
                                                                        'test.npz')
    # Use first 'args.samples' samples to create dataset
    x_train = TensorDataset((torch.tensor(train_data)[:args.samples,:,:,:],
                             torch.tensor(train_labels)[:args.samples]),
                            random_crop=args.rc, hflip=args.hflip)
    x_test = TensorDataset((torch.tensor(test_data), torch.tensor(test_labels)))

    # Make dataloaders
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(x_train, batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(x_test, batch_size=args.test_batch_size,
                                              shuffle=False, **kwargs)

    return train_loader, test_loader

def main(args):
    # Fix seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Build model
    model = model_dict[args.model](num_classes=10)
    summary(model, (1,3,32,32), col_names=("input_size","output_size","num_params"))
    if args.cuda: model = nn.DataParallel(model).cuda()

    # Get data
    train_loader, test_loader = get_data(args)

    # Optimization settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50,100,150], gamma=0.1)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        acc = train(epoch, model, criterion, optimizer, train_loader)
        scheduler.step()
        if epoch % args.log_interval == 0:
            print('Learning rate: {:.1e}'.format(optimizer.param_groups[0]['lr']))
            print('Epoch {:d} train accuracy: {:.2f}%'.format(epoch, acc))
            test(model, criterion, test_loader)

    # Save model
    augstr = '_aug' if (args.hflip and args.rc > 0) else ''
    torch.save(model.state_dict(),'{}{}_seed_{:d}.pth.tar'.format(args.model, augstr,args.seed))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='CIFAR10')
    parser.add_argument('--dataset_path', metavar='path/to/dataset/root', default='./',
                        type=str, help='location of cifar10 files')
    parser.add_argument('--model', metavar='resnet44', default='resnet44',
                        choices=['resnet44', 'resnet44_p4m', 'resnet44_gp4m', 'resnet44_gcp4m'],
                        type=str, help='Select model to use for experiment.')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--samples', type=int, default=None, metavar='samples',
                        help='amount of training samples to use (default: None (all))')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-5, metavar='WD',
                        help='weight decay (default: 0.0)')
    parser.add_argument('--rc', type=int, default=0, metavar='RC',
                        help='random cropping (default: 0')
    parser.add_argument('--hflip', action='store_true', default=False,
                        help='perform random horizontal flipping')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(args)

    main(args)
