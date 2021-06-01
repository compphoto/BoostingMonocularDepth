"""
/*****************************************************************************/

Test for BatchNorm2dSync with multi-gpu

/*****************************************************************************/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.append("./")
from modules import nn as NN

torch.backends.cudnn.deterministic = True


def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2. / n))
        elif isinstance(m, NN.BatchNorm2d) or isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

num_gpu = torch.cuda.device_count()
print("num_gpu={}".format(num_gpu))
if num_gpu < 2:
    print("No multi-gpu found. NN.BatchNorm2d will act as normal nn.BatchNorm2d")

m1 = nn.Sequential(
    nn.Conv2d(3, 3, 1, 1, bias=False),
    nn.BatchNorm2d(3),
    nn.ReLU(inplace=True),
    nn.Conv2d(3, 3, 1, 1, bias=False),
    nn.BatchNorm2d(3),
).cuda()
torch.manual_seed(123)
init_weight(m1)
m2 = nn.Sequential(
    nn.Conv2d(3, 3, 1, 1, bias=False),
    NN.BatchNorm2d(3),
    nn.ReLU(inplace=True),
    nn.Conv2d(3, 3, 1, 1, bias=False),
    NN.BatchNorm2d(3),
).cuda()
torch.manual_seed(123)
init_weight(m2)
m2 = nn.DataParallel(m2, device_ids=range(num_gpu))
o1 = torch.optim.SGD(m1.parameters(), 1e-3)
o2 = torch.optim.SGD(m2.parameters(), 1e-3)
y = torch.ones(num_gpu).float().cuda()
torch.manual_seed(123)
for _ in range(100):
    x = torch.rand(num_gpu, 3, 2, 2).cuda()
    o1.zero_grad()
    z1 = m1(x)
    l1 = F.mse_loss(z1.mean(-1).mean(-1).mean(-1), y)
    l1.backward()
    o1.step()
    o2.zero_grad()
    z2 = m2(x)
    l2 = F.mse_loss(z2.mean(-1).mean(-1).mean(-1), y)
    l2.backward()
    o2.step()
    print(m2.module[1].bias.grad - m1[1].bias.grad)
    print(m2.module[1].weight.grad - m1[1].weight.grad)
    print(m2.module[-1].bias.grad - m1[-1].bias.grad)
    print(m2.module[-1].weight.grad - m1[-1].weight.grad)
m2 = m2.module
print("===============================")
print("m1(nn.BatchNorm2d) running_mean",
      m1[1].running_mean, m1[-1].running_mean)
print("m2(NN.BatchNorm2d) running_mean",
      m2[1].running_mean, m2[-1].running_mean)
print("m1(nn.BatchNorm2d) running_var", m1[1].running_var, m1[-1].running_var)
print("m2(NN.BatchNorm2d) running_var", m2[1].running_var, m2[-1].running_var)
print("m1(nn.BatchNorm2d) weight", m1[1].weight, m1[-1].weight)
print("m2(NN.BatchNorm2d) weight", m2[1].weight, m2[-1].weight)
print("m1(nn.BatchNorm2d) bias", m1[1].bias, m1[-1].bias)
print("m2(NN.BatchNorm2d) bias", m2[1].bias, m2[-1].bias)
