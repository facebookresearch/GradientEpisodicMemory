# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib as mpl
mpl.use('Agg')

# if 'roman' in mpl.font_manager.weight_dict.keys():
#     del mpl.font_manager.weight_dict['roman']
# mpl.font_manager._rebuild()

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.family"] = "DejaVu Serif"

from matplotlib import pyplot as plt
from glob import glob
import numpy as np
import torch

models = ['single', 'independent', 'multimodal', 'icarl', 'ewc', 'gem']
datasets = ['mnist_permutations', 'mnist_rotations', 'cifar100']

names_datasets = {'mnist_permutations': 'MNIST permutations',
                  'mnist_rotations': 'MNIST rotations',
                  'cifar100': 'CIFAR-100'}

names_models = {'single': 'single',
                'independent': 'independent',
                'multimodal': 'multimodal',
                'icarl': 'iCARL',
                'ewc': 'EWC',
                'gem': 'GEM'}

colors = {'single': 'C0',
          'independent': 'C1',
          'multimodal': 'C2',
          'icarl': 'C2',
          'ewc': 'C3',
          'gem': 'C4'}

barplot = {}

for dataset in datasets:
    barplot[dataset] = {}
    for model in models:
        barplot[dataset][model] = {}
        matches = glob(model + '*' + dataset + '*.pt')
        if len(matches):
            data = torch.load(matches[0], map_location=lambda storage, loc: storage)
            acc, bwt, fwt = data[3][:]
            barplot[dataset][model]['acc'] = acc
            barplot[dataset][model]['bwt'] = bwt
            barplot[dataset][model]['fwt'] = fwt

for dataset in datasets:
    x_lab = []
    y_acc = []
    y_bwt = []
    y_fwt = []

    for i, model in enumerate(models):
        if barplot[dataset][model] != {}:
            x_lab.append(model)
            y_acc.append(barplot[dataset][model]['acc'])
            y_bwt.append(barplot[dataset][model]['bwt'])
            y_fwt.append(barplot[dataset][model]['fwt'])

    x_ind = np.arange(len(y_acc))

    plt.figure(figsize=(7, 3))
    all_colors = []
    for xi, yi, li in zip(x_ind, y_acc, x_lab):
        plt.bar(xi, yi, label=names_models[li], color=colors[li])
        all_colors.append(colors[li])
    plt.bar(x_ind + (len(y_acc) + 1) * 1, y_bwt, color=all_colors)
    plt.bar(x_ind + (len(y_acc) + 1) * 2, y_fwt, color=all_colors)
    plt.xticks([2, 8, 14], ['ACC', 'BWT', 'FWT'], fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(-1, len(y_acc) * 3 + 2)
    plt.ylabel('classification accuracy', fontsize=16)
    plt.title(names_datasets[dataset], fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('barplot_%s.pdf' % dataset, bbox_inches='tight')
    # plt.show()

evoplot = {}

for dataset in datasets:
    evoplot[dataset] = {}
    for model in models:
        matches = glob(model + '*' + dataset + '*.pt')
        if len(matches):
            data = torch.load(matches[0], map_location=lambda storage, loc: storage)
            evoplot[dataset][model] = data[1][:, 0].numpy()

for dataset in datasets:

    plt.figure(figsize=(7, 3))
    for model in models:
        if model in evoplot[dataset]:
            x = np.arange(len(evoplot[dataset][model]))
            x = (x - x.min()) / (x.max() - x.min()) * 20
            plt.plot(x, evoplot[dataset][model], color=colors[model], lw=3)
            plt.xticks(range(0, 21, 2))

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.xlabel('task number', fontsize=16)
    plt.title(names_datasets[dataset], fontsize=16)
    plt.tight_layout()
    plt.savefig('evoplot_%s.pdf' % dataset, bbox_inches='tight')
    # plt.show()
