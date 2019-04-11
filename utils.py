import heapq
import numpy as np
import torch
from torch.utils.data import DataLoader


class AverageMeter(object):
    # average meter to maintain statistics
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.count += n
        self.sum += (n / self.count) * (val - self.sum)

    def avg(self):
        return self.sum


class GammaRandomWorkerSelection(object):
    # schedule workers in gamma distribution order
    def __init__(self, args):
        self.order = []
        self.shape = args.sim_gamma_shape
        self.scale = args.sim_gamma_scale
        for i in range(args.sim_size):
            rand = np.random.gamma(self.shape, self.scale, 1)
            heapq.heappush(self.order, (rand, i))

    def __iter__(self):
        while True:
            x = heapq.heappop(self.order)
            y = x[0] + np.random.gamma(self.shape, self.scale, 1)
            heapq.heappush(self.order, (y, x[1]))
            yield int(x[1])


def batch_step(args, data, target, loss_avg, acc_avg):
    # move vectors to gpu if cuda available
    if args.cuda:
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
    # get batch size
    bs = target.shape[0]
    # feed forward batch
    output = args.model(data)
    # compute loss
    loss = args.loss_fn(output, target)
    # update loss_avg
    loss_avg.update(loss.item(), bs)  # sum up batch loss
    # get the index of the max log-probability
    pred = output.data.max(1, keepdim=True)[1]
    # update acc_avg
    acc_avg.update(100.0 * pred.eq(target.data.view_as(pred)).cpu().sum().item() / bs, bs)
    return loss


def data_loader(data_set, is_train, args):
    return DataLoader(data_set, batch_size=args.batch_size, shuffle=is_train, num_workers=2, pin_memory=True)


def init_weights(params):
    return [p.clone() for p in params]


def init_momentum(params):
    with torch.no_grad():
        return {id(p): torch.zeros_like(p) for p in params}


def clone_weights(params, weights):
    set_weights(params, weights)


def load_weights(params, weights):
    set_weights(weights, params)


def set_weights(source, target):
    with torch.no_grad():
        for w, p in zip(source, target):
            p.data = w.data.clone()


def clone_momentum(opt, momentum, params):
    with torch.no_grad():
        for p in params:
            momentum[id(p)].data = opt.state[p]["momentum_buffer"].data.clone()


def load_momentum(opt, momentum, params):
    with torch.no_grad():
        for p in params:
            opt.state[p]["momentum_buffer"].data = momentum[id(p)].data.clone()


def load_worker(args, worker_rank):
    # load worker's weights
    load_weights(args.model.parameters(), args.worker_weights[worker_rank])
    # load worker's optimizer
    load_momentum(args.optimizer, args.worker_momentum[worker_rank], args.model.parameters())


def load_master(args):
    # load master's weights
    load_weights(args.model.parameters(), args.master_weights)


def update_worker(args, worker_rank):
    # get the worker's optimizer state
    worker_momentum = args.worker_momentum[worker_rank]

    with torch.no_grad():
        pg = args.optimizer.param_groups[0]
        f = pg["lr"] * pg["momentum"]

        for p in args.model.parameters():
            # add current worker's momentum diff
            args.momentum_sum[id(p)] += args.optimizer.state[p]["momentum_buffer"] - worker_momentum[id(p)]
            # compute look-ahead for worker
            p.data -= args.momentum_sum[id(p)] * f

    # update worker weights
    clone_weights(args.model.parameters(), args.worker_weights[worker_rank])
    # update worker's momentum vector
    clone_momentum(args.optimizer, args.worker_momentum[worker_rank], args.model.parameters())


def update_master(args):
    # update master's weights
    clone_weights(args.model.parameters(), args.master_weights)


def learning_rate_decay(epoch, args):
    # decay the learning rate by args.lr_decay if epoch is in args.schedule
    if epoch in args.schedule:
        args.optimizer.param_groups[0]['lr'] *= args.lr_decay


def delay_compensation(args):
    # adjust the gradients with the delay compensation
    for p, w in zip(args.model.parameters(), args.master_weights):
        p.grad.data = p.grad.data + args.dc * p.grad.data * p.grad.data * (w.data - p.data)
