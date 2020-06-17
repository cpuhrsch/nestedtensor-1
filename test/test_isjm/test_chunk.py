import traceback
import functools
import pdb
import sys
import torch
import nestedtensor
import unittest
from nestedtensor.test.utils import TestCase
import random

class TestCat(TestCase):

    def test_cat(self):
        print("torch.cat is currently not supported due to https://github.com/pytorch/pytorch/issues/34294")

if __name__ == "__main__":
    unittest.main()
