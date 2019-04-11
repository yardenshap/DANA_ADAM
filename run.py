import argparse
import random
from tqdm import tqdm

import torch
from torch import nn
from torch import optim

import numpy as np

from model import resnet20_cifar, get_cifar_10_data_set
from utils import (
    AverageMeter,
    batch_step,
    init_weights,
    init_momentum,
    data_loader,
    GammaRandomWorkerSelection,
    load_master,
    load_worker,
    update_master,
    update_worker,
    learning_rate_decay,
    delay_compensation,
)

torch.backends.cudnn.benchmark = True


def train(epoch, args):
    args.model.train()
    loss_avg = AverageMeter()
    acc_avg = AverageMeter()
    norm_avg = AverageMeter()

    progress_bar = tqdm(args.train_loader)
    for data, target in progress_bar:

        # learning rate warmup
        if args.warmup_lr:
            args.optimizer.param_groups[0]['lr'] = args.warmup_lr.pop(0)

        # a worker finished computing gradients according to the gamma distribution
        worker_rank = next(args.worker_order)
        load_worker(args, worker_rank)

        # worker computes gradient on its set of weights
        args.optimizer.zero_grad()
        loss = batch_step(args, data, target, loss_avg, acc_avg)
        loss.backward()

        # the master receives the gradients from the worker and updates its weights
        delay_compensation(args)
        load_master(args)
        args.optimizer.step()
        update_master(args)

        # the worker receives the master's new weights
        update_worker(args, worker_rank)

        # compute the gradient norm
        norm_avg.update(sum(p.grad.data.norm() ** 2 for p in args.model.parameters()) ** 0.5, target.shape[0])
        progress_bar.set_description("Epoch: %d, Loss: %0.8f Norm: %0.4f LR: %0.4f" % (
            epoch, loss_avg.avg(), norm_avg.avg(), args.optimizer.param_groups[0]['lr']))
    progress_bar.close()


def evaluate(epoch, args):
    args.model.eval()
    loss_avg = AverageMeter()
    acc_avg = AverageMeter()

    with torch.no_grad():
        for data, target in tqdm(args.test_loader):
            batch_step(args, data, target, loss_avg, acc_avg)

    print("Epoch:", epoch, "Test Loss:", loss_avg.avg(), "Test Accuracy:", acc_avg.avg(), "\n")


def run():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch Environment")

    train_parser = parser.add_argument_group("Train Parameters")
    train_parser.add_argument("--epochs", type=int, default=160, metavar="E",
                              help="number of epochs to train (default: 10)")
    train_parser.add_argument("--batch-size", type=int, default=128, metavar="B",
                              help="input batch size for training (default: 128)")
    train_parser.add_argument("--test-batch-size", type=int, default=128, metavar="BT",
                              help="input batch size for testing (default: 128)")
    train_parser.add_argument("--lr_decay", type=float, default=0.1, metavar="LD", help="learning rate decay rate")
    train_parser.add_argument("--schedule", type=int, nargs="*", default=[80, 120],
                              help="learning rate is decayed at these epochs")
    train_parser.add_argument("--warmup-epochs", type=int, default=5, metavar="WE", help="number of warmup epochs")
    train_parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    train_parser.add_argument("--seed", type=int, default=7186021514134990023, metavar="S",
                              help="random seed (default: 7186021514134990023)")

    simulator_parser = parser.add_argument_group("Simulator Parameters")
    simulator_parser.add_argument("--sim-size", type=int, default=16, metavar="N", help="size of simulator")
    simulator_parser.add_argument("--sim-gamma-shape", type=float, default=100, metavar="GSH",
                                  help="gamma shape parameter")
    simulator_parser.add_argument("--sim-gamma-scale", type=float, default=1.28, metavar="GSC",
                                  help="gamma scale parameter")

    optimizer_parser = parser.add_argument_group("Optimizer Parameters")
    optimizer_parser.add_argument("--lr", type=float, default=0.1, metavar="LR", help="learning rate (default: 0.1)")
    optimizer_parser.add_argument("--momentum", type=float, default=0.9, metavar="M",
                                  help="SGD momentum (default: 0.9)")
    optimizer_parser.add_argument("--dc", type=float, default=2, metavar="DC", help="Delay Compensation (default: 0)")
    optimizer_parser.add_argument("--weight-decay", type=float, default=1e-4, metavar="WD",
                                  help="SGD weight decay (default: 0)")

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    random.seed(torch.initial_seed())

    print("*** Configuration ***")
    for k in vars(args):
        print(str(k), ":", str(getattr(args, k)))

    train_set, test_set = get_cifar_10_data_set()  # get CIFAR-10 train and test set
    args.train_loader = data_loader(train_set, is_train=True, args=args)
    args.test_loader = data_loader(test_set, is_train=False, args=args)
    args.model = resnet20_cifar()  # get ResNet-20 Model
    if args.cuda:
        args.model = args.model.cuda()
    args.loss_fn = nn.CrossEntropyLoss()  # use cross-entropy loss

    # create optimizer
    args.optimizer = optim.SGD(args.model.parameters(), lr=args.lr, momentum=args.momentum,
                               weight_decay=args.weight_decay)

    assert len(args.optimizer.param_groups) == 1

    # initialize optimizer's momentum
    for p in args.model.parameters():
        args.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

    # clone weights for master
    args.master_weights = init_weights(args.model.parameters())

    # clone weights, one for each  worker
    args.worker_weights = [init_weights(args.model.parameters()) for _ in range(args.sim_size)]

    # clone optimizer, one for each  worker
    args.worker_momentum = [init_momentum(args.model.parameters()) for _ in range(args.sim_size)]

    # create the gamma distribution order
    args.worker_order = iter(GammaRandomWorkerSelection(args))

    # initialize dana
    args.momentum_sum = {id(p): torch.zeros_like(p) for p in args.model.parameters()}

    # initialize warmup
    args.warmup_lr = np.linspace(args.lr / args.sim_size, args.lr,
                                 len(args.train_loader) * args.warmup_epochs).tolist()

    print("*** Training with DANA-DC ***")

    for epoch in range(args.epochs):
        learning_rate_decay(epoch, args)
        train(epoch, args)
        evaluate(epoch, args)


if __name__ == "__main__":
    run()
