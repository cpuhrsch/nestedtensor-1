import traceback
import functools
import pdb
import sys
import torch
import nestedtensor
import unittest
from nestedtensor.test.utils import TestCase
import random

class TestChunk(TestCase):

    def test_chunk(self):
        t0 = torch.randn(3, 3)
        t1 = torch.randn(2, 3)
        t2 = torch.randn(3, 3)
        ts = [[t0, t1], [t2, t1, t0]]
        nt = nestedtensor.nested_tensor(ts)
        print("nt: ", nt)

if __name__ == "__main__":
    unittest.main()
