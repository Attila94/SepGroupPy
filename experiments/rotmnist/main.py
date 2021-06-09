import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torchinfo import summary
from models import CNN

def main(args):

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    # Fix seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

    def train(epoch):
        model.train()
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        scheduler.step()
        return 100. * correct / len(train_loader.dataset)

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').data
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('Test set: Average loss: {:.4f}, \
              Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    model = CNN(planes=args.planes, equivariance=args.equivariance, separable=args.separable)
    if args.cuda:
        model.cuda()
    summary(model, (1,1,28,28), col_names=("input_size","output_size","num_params"))

    print('Loading dataset...')
    data_train = np.loadtxt(args.dataset_path+'/mnist_all_rotation_normalized_float_train_valid.amat',
                            dtype=np.float32)
    data_test = np.loadtxt(args.dataset_path+'/mnist_all_rotation_normalized_float_test.amat',
                           dtype=np.float32)

    x_train = torch.tensor(np.reshape(data_train[:,:-1],(-1,1,28,28)), dtype=torch.float32)[:args.samples,:,:,:]
    y_train = torch.tensor(data_train[:,-1], dtype=torch.long)[:args.samples]
    x_test = torch.tensor(np.reshape(data_test[:,:-1],(-1,1,28,28)), dtype=torch.float32)
    y_test = torch.tensor(data_test[:,-1], dtype=torch.long)
    print('Dataset loaded.')

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=args.test_batch_size, shuffle=False, **kwargs)

    if args.savestr:
        torch.save(model.state_dict(),'./{}_init_{}.pth.tar'.format(args.savestr,args.planes))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step, gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        acc = train(epoch)
        if epoch % args.log_interval == 0:
            print('Epoch {:d} train accuracy: {:.2f}%'.format(epoch, acc))
            test()
    if args.savestr:
        torch.save(model.state_dict(),'./{}_trained_{}.pth.tar'.format(args.savestr,args.planes
                                                                       ))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='RotatedMNIST Experiment')
    parser.add_argument('--dataset-path', type=str, default='./',
                        help='path to rotated mnist directory')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--wd', type=float, default=1e-5, metavar='WD',
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--step', type=int, default=40, metavar='LRS',
                        help='step size for lr scheduler (default: 40)')
    parser.add_argument('--planes', type=int, default=20, metavar='planes',
                        help='number of channels in CNN (default: 20)')
    parser.add_argument('--samples', type=int, default=None, metavar='samples',
                        help='amount of training samples to use (default: None (all))')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--savestr', type=str, default='', metavar='STR',
                        help='weights file name addendum')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--equivariance', type=str, default='Z2', metavar='E',
                        help='equivariant transformation group of CNN (Z2 or P4)')
    parser.add_argument('--separable', default=None,
                        help='separable convolutions (one of [None, g, gc]) - for Z2 models g and gc can be used interchangeably.')
    args = parser.parse_args()

    main(args)
