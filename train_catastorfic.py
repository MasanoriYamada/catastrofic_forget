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
def permute_mnist(dataset):
    id2 = np.random.permutation(len(dataset[0]))
    dataset2 = []
    for data in dataset:
        dataset2.append(data[id2])
    return np.array(dataset2)

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
    test = np.array(test1)
    t_train, x_train = tuple_data_to_np(train)
    t_test, x_test = tuple_data_to_np(test)
    model.get_fisher(x_test)
    model.set_star_weights_dict()

    # on EWC
    model.lam = 100.0
    # set param next task
    x_train2 = permute_mnist(x_train)
    x_test2 = permute_mnist(x_test)
    t_train2 = t_train
    t_test2 = t_test
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
                }
    test_data_lst = {'task1':(x_train, t_train), 'task2':(x_train2, t_train2)}
    # Learning loop
    N = len(x_train2)
    for epoch in range(args.epoch):
        # training
        # N個の順番をランダムに並び替える
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        sum_fisher = 0
        # 0〜Nまでのデータをバッチサイズごとに使って学習
        for i in range(0, N, args.batchsize):
            x_batch = x_train2[perm[i:i + args.batchsize]]
            t_batch = t_train2[perm[i:i + args.batchsize]]

            # 勾配を初期化
            opt.zero_grads()
            # 順伝播させて誤差と精度を算出
            loss = model(x_batch, t_batch)
            # 誤差逆伝播で勾配を計算
            loss.backward()
            opt.update()
            acc = F.accuracy(model.predictor(x_batch), t_batch)
            sum_loss += loss.data * args.batchsize
            sum_accuracy += acc.data * args.batchsize
        plt_dict['train_loss'].append(sum_loss)
        plt_dict['train_acc'].append(sum_accuracy)

        # 訓練データの誤差と、正解精度を表示
        print('train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N))

        for key in test_data_lst:
            x, t = test_data_lst[key]
            plt_dict['{}_test_loss'.format(key)].append(model(x,t).data)
            plt_dict['{}_test_acc'.format(key)].append(F.accuracy(model.predictor(x), t).data)

    plt.xlabel('epoch')
    plt.ylabel('validation accuracy')
    for key in test_data_lst:
        plt.plot(plt_dict['{}_test_acc'.format(key)], label="{}".format(key))
    plt.legend(loc='best')
    plt.savefig('./accuracy.png', transparent=True)
    plt.close('all')
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

def show_img_array(x, tate, yoko):
    fig, ax = plt.subplots(tate, yoko, figsize=(yoko, tate), dpi=300)
    for ai, xi in zip(ax.flatten(), x):
        ai.set_xticklabels([])
        ai.set_yticklabels([])
        ai.imshow(xi.reshape(28, 28))
    plt.show()


if __name__ == '__main__':
    main()