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
            nt = constructor([torch.tensor(3)])
            print(nt)
            print(nt.sum())


if __name__ == "__main__":
    unittest.main()
