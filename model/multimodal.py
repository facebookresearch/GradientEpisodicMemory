# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def reset_bias(m):
    m.bias.data.fill_(0.0)


class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()

        self.i_layer = nn.ModuleList()
        self.h_layer = nn.ModuleList()
        self.o_layer = nn.ModuleList()

        self.n_layers = args.n_layers
        nh = args.n_hiddens

        if self.n_layers > 0:
            # dedicated input layer
            for _ in range(n_tasks):
                self.i_layer += [nn.Linear(n_inputs, nh)]
                reset_bias(self.i_layer[-1])

            # shared hidden layer
            self.h_layer += [nn.ModuleList()]
            for _ in range(self.n_layers):
                self.h_layer[0] += [nn.Linear(nh, nh)]
                reset_bias(self.h_layer[0][0])

            # shared output layer
            self.o_layer += [nn.Linear(nh, n_outputs)]
            reset_bias(self.o_layer[-1])

        # linear model falls back to independent models
        else:
            self.i_layer += [nn.Linear(n_inputs, n_outputs)]
            reset_bias(self.i_layer[-1])

        self.relu = nn.ReLU()
        self.soft = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), args.lr)

    def forward(self, x, t):
        h = x

        if self.n_layers == 0:
            y = self.soft(self.i_layer[t if isinstance(t, int) else t[0]](h))
        else:
            # task-specific input
            h = self.relu(self.i_layer[t if isinstance(t, int) else t[0]](h))
            # shared hiddens
            for l in range(self.n_layers):
                h = self.relu(self.h_layer[0][l](h))
            # shared output
            y = self.soft(self.o_layer[0](h))

        return y

    def observe(self, x, t, y):
        self.zero_grad()
        self.loss(self.forward(x, t), y).backward()
        self.optimizer.step()
