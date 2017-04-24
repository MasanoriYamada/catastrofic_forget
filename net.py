#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import copy

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

class EWC_loss(chainer.Chain):
    def __init__(self, predictor, lam):
        super(EWC_loss, self).__init__(predictor=predictor)
        self.param_dict = {}
        self.lam = lam

    def __call__(self, x, t):
        y = self.predictor(x)
        self.get_weights_dict()
        accuracy = F.accuracy(y, t)
        vanilla_loss = F.softmax_cross_entropy(y, t)
        if self.lam == 0.0:
            chainer.report({'loss': vanilla_loss, 'accuracy': accuracy}, self)
            return vanilla_loss

        self.ewc_loss = 0.0
        self.regularize_term = 0.0
        for key in self.param_dict:
            diff_sq = (self.param_dict[key] - self.star_param_dict[key].data)**2 # theta_star is not Variable
            self.regularize_term += F.sum(self.fisher[key] *diff_sq)
        self.ewc_loss = vanilla_loss + self.lam / 2.0 * self.regularize_term
        chainer.report({'loss': self.ewc_loss, 'accuracy': accuracy}, self)
        return self.ewc_loss

    def get_weights_dict(self):
        # Todo automatic get
        self.param_dict['W1'] = self.predictor.l1.W
        self.param_dict['b1'] = self.predictor.l1.b
        self.param_dict['W2'] = self.predictor.l2.W
        self.param_dict['b2'] = self.predictor.l2.b

    def init_weights_grad(self):
        self.predictor.l1.W.zerograd()
        self.predictor.l1.b.zerograd()
        self.predictor.l2.W.zerograd()
        self.predictor.l2.b.zerograd()

    def set_star_weights_dict(self):
        self.star_param_dict = copy.deepcopy(self.param_dict)

    def get_fisher(self, x):
        sampling = 200
        del_param_log_x = []
        for id in range(sampling):
            x_ind = np.random.randint(x.shape[0])
            log_prob = self.predictor(x[x_ind].reshape(1,-1))
            log_prob.zerograd() # need init grad for multi dimensional back prop
            self.init_weights_grad() # need init grad for multi dimensional back prop
            log_prob.backward()
            self.get_weights_dict()
            del_param_log = {}
            for key in self.param_dict:
                del_param_log[key]  = self.param_dict[key].grad**2
            del_param_log_x.append(del_param_log)
        del_param_log_x = np.array(del_param_log_x)
        # mean
        self.fisher = {'W1':0.0, 'b1':0.0, 'W2':0.0, 'b2':0.0}
        for sample_dict in del_param_log_x:
            for key in sample_dict:
                self.fisher[key] += sample_dict[key] / float(sampling)


