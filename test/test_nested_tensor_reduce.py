import traceback
import functools
import pdb
import sys
import torch
import nestedtensor
import unittest
from utils import TestCase
import random

import utils

ntnt = nestedtensor.nested_tensor

class TestReduce(TestCase):

    def test_cumsum(self):
        t0 = torch.arange(9).reshape(3, 3)
        t1 = torch.arange(6).reshape(2, 3)
        t2 = torch.arange(9).reshape(3, 3)
        ts = [[t0, t1], [t2]]
        nt = nestedtensor.nested_tensor(ts)

        self.assertRaises(RuntimeError, lambda: torch.cumsum(nt, 0))
        self.assertRaises(RuntimeError, lambda: torch.cumsum(nt, 1))
        self.assertEqual(ntnt([[torch.cumsum(t0, 0), torch.cumsum(t1, 0)],
                               [torch.cumsum(t2, 0)]]), torch.cumsum(nt, 2))
        self.assertEqual(ntnt([[torch.cumsum(t0, 1), torch.cumsum(t1, 1)],
                               [torch.cumsum(t2, 1)]]), torch.cumsum(nt, 3))
        self.assertRaises(IndexError, lambda: torch.cumsum(nt, 4))

    def _test_allreduce(self, fn):
        t0 = torch.randn(3, 3, requires_grad=True)
        t1 = torch.randn(2, 3, requires_grad=True)
        t2 = torch.randn(3, 3, requires_grad=True)
        ts = [[t0, t1], [t2]]
        nt = nestedtensor.nested_tensor(ts, requires_grad=True)
        t = fn(nt)
        a = torch.stack([fn(t0), fn(t1), fn(t2)])
        self.assertEqual(t, fn(a))
        fn(a).backward()
        t.backward()
        self.assertEqual(nt.grad[0][0], t0.grad)
        self.assertEqual(nt.grad[0][1], t1.grad)
        self.assertEqual(nt.grad[1][0], t2.grad)

    def test_sum_allreduce(self):
        self._test_allreduce(lambda x: x.sum())

    def test_mean_allreduce(self):
        self._test_allreduce(lambda x: x.mean())

    def test_prod_allreduce(self):
        self._test_allreduce(lambda x: x.prod())

    def _test_reduce_dim(self, fn):
        t0 = torch.randn(3, 3, requires_grad=True)
        t1 = torch.randn(2, 3, requires_grad=True)
        t2 = torch.randn(3, 3, requires_grad=True)
        ts = [[t0, t1], [t2]]
        nt = nestedtensor.nested_tensor(ts, requires_grad=True)
        self.assertRaises(RuntimeError, lambda: fn(nt, 0, True, torch.float))
        self.assertRaises(RuntimeError, lambda: fn(nt, 1, True, torch.float))
        self.assertEqual(fn(nt, 2, False, torch.double)[0][0], fn(t0, 0, False, torch.double))
        self.assertEqual(fn(nt, 2, False, torch.double)[0][1], fn(t1, 0, False, torch.double))
        self.assertEqual(fn(nt, 2, False, torch.double)[1][0], fn(t2, 0, False, torch.double))
        self.assertEqual(fn(nt, 3, False, torch.double)[0][0], fn(t0, 1, False, torch.double))
        self.assertEqual(fn(nt, 3, False, torch.double)[0][1], fn(t1, 1, False, torch.double))
        self.assertEqual(fn(nt, 3, False, torch.double)[1][0], fn(t2, 1, False, torch.double))

    def test_sum_reduce_dim(self):
        self._test_reduce_dim(lambda x, dim, keepdim, dtype: x.sum(dim, keepdim, dtype=dtype))

    def test_mean_reduce_dim(self):
        self._test_reduce_dim(lambda x, dim, keepdim, dtype: x.mean(dim, keepdim, dtype=dtype))

    def test_prod_reduce_dim(self):
        self._test_reduce_dim(lambda x, dim, keepdim, dtype: x.prod(dim, keepdim, dtype=dtype))


if __name__ == "__main__":
    unittest.main()
