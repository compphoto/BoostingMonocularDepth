"""
/*****************************************************************************/

BatchNorm2dSync with multi-gpu

code referenced from : https://github.com/mapillary/inplace_abn

/*****************************************************************************/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.cuda.comm as comm
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ._syncbn._ext import syncbn as _lib_bn


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count


def _check_contiguous(*args):
    if not all([mod is None or mod.is_contiguous() for mod in args]):
        raise ValueError("Non-contiguous input")


class BatchNorm2dSyncFunc(Function):

    @classmethod
    def forward(cls, ctx, x, weight, bias, running_mean, running_var,
                extra, compute_stats=True, momentum=0.1, eps=1e-05):
        # Save context
        if extra is not None:
            cls._parse_extra(ctx, extra)
        ctx.compute_stats = compute_stats
        ctx.momentum = momentum
        ctx.eps = eps
        if ctx.compute_stats:
            N = _count_samples(x) * (ctx.master_queue.maxsize + 1)
            assert N > 1
            num_features = running_mean.size(0)
            # 1. compute sum(x) and sum(x^2)
            xsum = x.new().resize_(num_features)
            xsqsum = x.new().resize_(num_features)
            _check_contiguous(x, xsum, xsqsum)
            _lib_bn.syncbn_sum_sqsum_cuda(x.detach(), xsum, xsqsum)
            if ctx.is_master:
                xsums, xsqsums = [xsum], [xsqsum]
                # master : gatther all sum(x) and sum(x^2) from slaves
                for _ in range(ctx.master_queue.maxsize):
                    xsum_w, xsqsum_w = ctx.master_queue.get()
                    ctx.master_queue.task_done()
                    xsums.append(xsum_w)
                    xsqsums.append(xsqsum_w)
                xsum = comm.reduce_add(xsums)
                xsqsum = comm.reduce_add(xsqsums)
                mean = xsum / N
                sumvar = xsqsum - xsum * mean
                var = sumvar / N
                uvar = sumvar / (N - 1)
                # master : broadcast global mean, variance to all slaves
                tensors = comm.broadcast_coalesced(
                    (mean, uvar, var), [mean.get_device()] + ctx.worker_ids)
                for ts, queue in zip(tensors[1:], ctx.worker_queues):
                    queue.put(ts)
            else:
                # slave : send sum(x) and sum(x^2) to master
                ctx.master_queue.put((xsum, xsqsum))
                # slave : get global mean and variance
                mean, uvar, var = ctx.worker_queue.get()
                ctx.worker_queue.task_done()

            # Update running stats
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * mean)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * uvar)
            ctx.N = N
            ctx.save_for_backward(x, weight, bias, mean, var)
        else:
            mean, var = running_mean, running_var

        output = x.new().resize_as_(x)
        _check_contiguous(output, x, mean, var, weight, bias)
        # do batch norm forward
        _lib_bn.syncbn_forward_cuda(
            output, x, weight if weight is not None else x.new(),
            bias if bias is not None else x.new(), mean, var, ctx.eps)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        x, weight, bias, mean, var = ctx.saved_tensors
        dz = dz.contiguous()
        if ctx.needs_input_grad[0]:
            dx = dz.new().resize_as_(dz)
        else:
            dx = None
        if ctx.needs_input_grad[1]:
            dweight = dz.new().resize_as_(mean).zero_()
        else:
            dweight = None
        if ctx.needs_input_grad[2]:
            dbias = dz.new().resize_as_(mean).zero_()
        else:
            dbias = None
        _check_contiguous(x, dz, weight, bias, mean, var)

        # 1. compute \sum(\frac{dJ}{dy_i}) and \sum(\frac{dJ}{dy_i}*\hat{x_i})
        num_features = mean.size(0)
        sum_dz = x.new().resize_(num_features)
        sum_dz_xhat = x.new().resize_(num_features)
        _check_contiguous(sum_dz, sum_dz_xhat)
        _lib_bn.syncbn_backward_xhat_cuda(
            dz, x, mean, var, sum_dz, sum_dz_xhat, ctx.eps)
        if ctx.is_master:
            sum_dzs, sum_dz_xhats = [sum_dz], [sum_dz_xhat]
            # master : gatther from slaves
            for _ in range(ctx.master_queue.maxsize):
                sum_dz_w, sum_dz_xhat_w = ctx.master_queue.get()
                ctx.master_queue.task_done()
                sum_dzs.append(sum_dz_w)
                sum_dz_xhats.append(sum_dz_xhat_w)
            # master : compute global stats
            sum_dz = comm.reduce_add(sum_dzs)
            sum_dz_xhat = comm.reduce_add(sum_dz_xhats)
            sum_dz /= ctx.N
            sum_dz_xhat /= ctx.N
            # master : broadcast global stats
            tensors = comm.broadcast_coalesced(
                (sum_dz, sum_dz_xhat), [mean.get_device()] + ctx.worker_ids)
            for ts, queue in zip(tensors[1:], ctx.worker_queues):
                queue.put(ts)
        else:
            # slave : send to master
            ctx.master_queue.put((sum_dz, sum_dz_xhat))
            # slave : get global stats
            sum_dz, sum_dz_xhat = ctx.worker_queue.get()
            ctx.worker_queue.task_done()

        # do batch norm backward
        _lib_bn.syncbn_backard_cuda(
            dz, x, weight if weight is not None else dz.new(),
            bias if bias is not None else dz.new(),
            mean, var, sum_dz, sum_dz_xhat,
            dx if dx is not None else dz.new(),
            dweight if dweight is not None else dz.new(),
            dbias if dbias is not None else dz.new(), ctx.eps)

        return dx, dweight, dbias, None, None, None, \
            None, None, None, None, None

    @staticmethod
    def _parse_extra(ctx, extra):
        ctx.is_master = extra["is_master"]
        if ctx.is_master:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queues = extra["worker_queues"]
            ctx.worker_ids = extra["worker_ids"]
        else:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queue = extra["worker_queue"]

batchnorm2d_sync = BatchNorm2dSyncFunc.apply

__all__ = ["batchnorm2d_sync"]
