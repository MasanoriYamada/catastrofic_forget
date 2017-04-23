#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import six

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import link
from chainer import cuda, training, reporter
from chainer.datasets import get_mnist
from chainer.training import trainer, extensions
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import reporter as reporter_module
from chainer import variable
from chainer.datasets import get_mnist
from chainer import optimizer as optimizer_module

class Multi_Evaluator(extensions.Evaluator):

    def __init__(self, iterator_dict, target, converter=convert.concat_examples, device=None, eval_hook=None, eval_func=None):

        self._iterators = iterator_dict
        if isinstance(target, link.Link):
            target = {'target': target}
        self._targets = target

        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self.eval_func = eval_func

    def evaluate(self):
        target = self._targets['target']
        for key in self._iterators:
            iterator = self._iterators[key]
            it = copy.copy(iterator)
            summary = reporter.DictSummary()
            for batch in it:
                with reporter_module.report_scope(observation):
                    in_arrays = self.converter(batch, self.device)
                    in_vars = tuple(variable.Variable(x, volatile='on') for x in in_arrays)
                    x = in_vars[0]
                    t = in_vars[1]
                    y = target.predictor(x)
                    accuracy = F.accuracy(y, t)
                    loss = target(x, t)
                    observation = {}
                    observation['test/{}/loss'.format(key)] = loss
                    observation['test/{}/accuracy'.format(key)] = accuracy
                summary.add(observation)



        return summary.compute_mean()