"""
/*****************************************************************************/

BatchNorm2dSync with multi-gpu

/*****************************************************************************/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    # python 3
    from queue import Queue
except ImportError:
    # python 2
    from Queue import Queue

import torch
import torch.nn as nn
from modules.functional import batchnorm2d_sync


class BatchNorm2d(nn.BatchNorm2d):
    """
    BatchNorm2d with automatic multi-GPU Sync
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm2d, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine,
            track_running_stats=track_running_stats)
        self.devices = list(range(torch.cuda.device_count()))
        if len(self.devices) > 1:
            # Initialize queues
            self.worker_ids = self.devices[1:]
            self.master_queue = Queue(len(self.worker_ids))
            self.worker_queues = [Queue(1) for _ in self.worker_ids]

    def forward(self, x):
        compute_stats = self.training or not self.track_running_stats
        if compute_stats and len(self.devices) > 1:
            if x.get_device() == self.devices[0]:
                # Master mode
                extra = {
                    "is_master": True,
                    "master_queue": self.master_queue,
                    "worker_queues": self.worker_queues,
                    "worker_ids": self.worker_ids
                }
            else:
                # Worker mode
                extra = {
                    "is_master": False,
                    "master_queue": self.master_queue,
                    "worker_queue": self.worker_queues[
                        self.worker_ids.index(x.get_device())]
                }
            return batchnorm2d_sync(x, self.weight, self.bias,
                                    self.running_mean, self.running_var,
                                    extra, compute_stats, self.momentum,
                                    self.eps)
        return super(BatchNorm2d, self).forward(x)

    def __repr__(self):
        """repr"""
        rep = '{name}({num_features}, eps={eps}, momentum={momentum},' \
              ' affine={affine}, devices={devices})'
        return rep.format(name=self.__class__.__name__, **self.__dict__)
