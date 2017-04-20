#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

class Net(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(Net, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        return self.l2(h1)

    def get_softmax_corssentropy_loss(self, x, t):
        h1 = F.relu(self.l1(x))
        h2 = self.l2(h1)
        loss = F.softmax_cross_entropy(h2, t)
        return loss
