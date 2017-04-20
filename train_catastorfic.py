#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from copy import deepcopy
from matplotlib import pyplot as plt

import net


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
def permute_mnist(mnist):
    perm_inds = range(mnist.train.images.shape[1])
    np.random.shuffle(perm_inds)
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name) # shallow copy
        this_set._images = np.transpose(np.array([this_set.images[:,c] for c in perm_inds]))
    return mnist2

def main():
    parser = argparse.ArgumentParser(description='catastorphic forgetting')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # model and optimizer
    model = L.Classifier(net.Net(args.unit, 10),lossfun = F.softmax_cross_entropy)
    opt = chainer.optimizers.Adam()
    opt.setup(model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # load mnist data
    train1, test1 = chainer.datasets.get_mnist()
    train_iter1 = chainer.iterators.SerialIterator(train1, args.batchsize)
    test_iter1 = chainer.iterators.SerialIterator(test1, args.batchsize, repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter1, opt, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(test_iter1, model, device=args.gpu))
    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
            'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.ProgressBar())

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()