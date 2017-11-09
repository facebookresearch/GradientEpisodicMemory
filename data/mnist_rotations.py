# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision import transforms
from PIL import Image
import argparse
import os.path
import random
import torch


def rotate_dataset(d, rotation):
    result = torch.FloatTensor(d.size(0), 784)
    tensor = transforms.ToTensor()

    for i in range(d.size(0)):
        img = Image.fromarray(d[i].numpy(), mode='L')
        result[i] = tensor(img.rotate(rotation)).view(784)
    return result


parser = argparse.ArgumentParser()

parser.add_argument('--i', default='raw/', help='input directory')
parser.add_argument('--o', default='mnist_rotations.pt', help='output file')
parser.add_argument('--n_tasks', default=10, type=int, help='number of tasks')
parser.add_argument('--min_rot', default=0.,
                    type=float, help='minimum rotation')
parser.add_argument('--max_rot', default=90.,
                    type=float, help='maximum rotation')
parser.add_argument('--seed', default=0, type=int, help='random seed')

args = parser.parse_args()

torch.manual_seed(args.seed)

tasks_tr = []
tasks_te = []

x_tr, y_tr = torch.load(os.path.join(args.i, 'mnist_train.pt'))
x_te, y_te = torch.load(os.path.join(args.i, 'mnist_test.pt'))

for t in range(args.n_tasks):
    min_rot = 1.0 * t / args.n_tasks * (args.max_rot - args.min_rot) + \
        args.min_rot
    max_rot = 1.0 * (t + 1) / args.n_tasks * \
        (args.max_rot - args.min_rot) + args.min_rot
    rot = random.random() * (max_rot - min_rot) + min_rot

    tasks_tr.append([rot, rotate_dataset(x_tr, rot), y_tr])
    tasks_te.append([rot, rotate_dataset(x_te, rot), y_te])

torch.save([tasks_tr, tasks_te], args.o)
