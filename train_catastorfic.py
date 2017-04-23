#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from copy import deepcopy
from matplotlib import pyplot as plt

import net
import util

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

# mnist imshow convenience function
# input is a 1D array of length 784
def mnist_imshow(img):
    plt.imshow(img.reshape([28,28]), cmap="gray")
    plt.axis('off')

# return a new mnist dataset w/ pixels randomly permuted
def permute_mnist(tuple_dataset):
    tuple_dataset2 = deepcopy(tuple_dataset)
    for data in tuple_dataset2:
        np.random.shuffle(data[0])
    return tuple_dataset2

def main():
    parser = argparse.ArgumentParser(description='catastorphic forgetting')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--unit', '-u', type=int, default=50,
                        help='Number of units')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # model and optimizer
    model = net.EWC_loss(net.Net(args.unit, 10), 0.0)
    opt = chainer.optimizers.SGD(lr=0.1)
    opt.setup(model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # load mnist data
    train1, test1 = chainer.datasets.get_mnist()
    train_iter1 = chainer.iterators.SerialIterator(train1, args.batchsize)
    test_iter1 = chainer.iterators.SerialIterator(test1, args.batchsize, repeat=False, shuffle=False)

    # 1st task
    # on EWC
    model.lam = 0.0
    # Set up a trainer
    updater = training.StandardUpdater(train_iter1, opt, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(test_iter1, model, device=args.gpu))
    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='lossA.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracyA.png'))
        trainer.extend(extensions.ProgressBar())
    trainer.run()

    # 2nd task
    # set param next task
    model.set_star_weights_dict()
    # on EWC
    model.lam = 15.0
    train2 = permute_mnist(train1)
    test2 = permute_mnist(test1)
    train_iter2 = chainer.iterators.SerialIterator(train2, args.batchsize)
    test_iter2 = chainer.iterators.SerialIterator(test2, args.batchsize, repeat=False, shuffle=False)
    updater2 = training.StandardUpdater(train_iter2, opt, device=args.gpu)
    trainer2 = training.Trainer(updater2, (args.epoch, 'epoch'), out=args.out)
    # trainer2.extend(extensions.Evaluator(test_iter2, model, device=args.gpu))
    trainer2.extend(util.Multi_Evaluator({'task1':test_iter1,'task2':test_iter2}, model, device=args.gpu))
    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer2.extend(
            extensions.PlotReport(['test/task1/loss', 'test/task2/loss'],
                                  'epoch', file_name='lossB.png'))
        trainer2.extend(
            extensions.PlotReport(
                ['test/task1/accuracy', 'test/task2/accuracy'],
                'epoch', file_name='accuracyB.png'))
    trainer2.extend(extensions.ProgressBar())
    trainer2.run()




if __name__ == '__main__':
    main()