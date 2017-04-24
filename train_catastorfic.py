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
    parser.add_argument('--epoch', '-e', type=int, default=20,
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
    # calc fisher in task1
    train = np.array(train1)
    t, x = tuple_data_to_np(train)
    model.get_fisher(x)
    model.set_star_weights_dict()
    # on EWC
    model.lam = 15.0
    # set param next task
    train2 = permute_mnist(train1)
    test2 = permute_mnist(test1)
    # train_iter2 = chainer.iterators.SerialIterator(train2, args.batchsize)
    # updater2 = training.StandardUpdater(train_iter2, opt, device=args.gpu)
    # trainer2 = training.Trainer(updater2, (args.epoch, 'epoch'), out=args.out)
    # trainer2.extend(extensions.ProgressBar())
    # trainer2.run()

    plt_dict = {'train_loss':[],
                'train_acc':[],
                'task1_test_loss':[],
                'task2_test_loss':[],
                'task1_test_acc':[],
                'task2_test_acc':[],
                'train_fisher': [],
                'task1_test_fisher': [],
                'task2_test_fisher': [],
                }
    test_data_lst = {'task1':test1, 'task2':test2}
    # Learning loop
    N = len(train2)
    for epoch in range(args.epoch):
        # training
        # N個の順番をランダムに並び替える
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        sum_fisher = 0
        train = np.array(train2)
        t, x = tuple_data_to_np(train)
        # 0〜Nまでのデータをバッチサイズごとに使って学習
        for i in range(0, N, args.batchsize):
            x_batch = x[perm[i:i + args.batchsize]]
            t_batch = t[perm[i:i + args.batchsize]]

            # 勾配を初期化
            opt.zero_grads()
            # 順伝播させて誤差と精度を算出
            loss = model(x_batch, t_batch)
            # 誤差逆伝播で勾配を計算
            loss.backward()
            opt.update()
            sum_fisher += model.regularize_term.data*args.batchsize
            acc = F.accuracy(model.predictor(x), t)
            sum_loss += loss.data * args.batchsize
            sum_accuracy += acc.data * args.batchsize
        plt_dict['train_fisher'].append(sum_fisher)
        plt_dict['train_loss'].append(sum_loss)
        plt_dict['train_acc'].append(sum_accuracy)

        # 訓練データの誤差と、正解精度を表示
        print('train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N))

        for key in test_data_lst:
            test = np.array(test_data_lst[key])
            t, x = tuple_data_to_np(test)
            plt_dict['{}_test_loss'.format(key)].append(model(x,t).data)
            plt_dict['{}_test_acc'.format(key)].append(F.accuracy(model.predictor(x), t).data)
    print(plt_dict)


def tuple_data_to_np(test):
    test = test.transpose(1, 0)
    x = []
    y = []
    for id1, id2 in zip(test[0], test[1]):
        x.append(id1)
        y.append(id2)
    x = np.array(x, dtype=np.float32)
    t = np.array(y, dtype=np.int32)
    return t, x


if __name__ == '__main__':
    main()