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


def _iter_constructors():
    yield nestedtensor.as_nested_tensor
    # yield nestedtensor.nested_tensor


class TestNestedTensor(TestCase):
    def test_sum(self):
        for constructor in _iter_constructors():
            nt = constructor([torch.randn(2, 3)])
            # NOTE: For torch.sum dtype can be a positional for full reductions.
            print(nt)
            print(torch.sum(nt))
            print(torch.sum(nt, dim=1, keepdim=False, out=nt))
            print(torch.sum(nt, dim=1, keepdim=True, out=nt))
            print(torch.sum(nt, dim=1, keepdim=False, dtype=torch.int64, out=nt))
            print(torch.sum(nt, dim=1, keepdim=True, dtype=torch.int64, out=nt))
            print(nt.sum())
            print(nt.sum(dtype=torch.int64))
            print(nt.sum(dim=1, keepdim=False))
            print(nt.sum(dim=1, keepdim=True))
            print(nt.sum(dim=1, keepdim=False, dtype=torch.int64))
            print(nt.sum(dim=1, keepdim=True, dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
