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


class DynamicClassBase(TestCase):
    longMessage = True

def _gen_test_binary(func):
    def _test_binary(self):
        a = utils.gen_float_tensor(1, (2, 3))
        b = utils.gen_float_tensor(2, (2, 3))
        c = utils.gen_float_tensor(3, (2, 3))
        # The constructor is supposed to copy!
        a1 = nestedtensor.nested_tensor([a, b])
        a2 = nestedtensor.nested_tensor([b, c])
        a1_l = nestedtensor.as_nested_tensor([a.clone(), b.clone()])
        a2_l = nestedtensor.as_nested_tensor([b.clone(), c.clone()])
        a3 = nestedtensor.nested_tensor([getattr(torch, func)(a, b),
                                  getattr(torch, func)(b, c)])
        a3_l = nestedtensor.as_nested_tensor(a3)
        self.assertEqual(a3_l, getattr(torch, func)(a1_l, a2_l))
        self.assertEqual(a3_l, getattr(torch, func)(a1, a2))
        self.assertEqual(a3, getattr(a1, func)(a2))
        self.assertEqual(a3, getattr(a1, func + "_")(a2))
        self.assertEqual(a3, a1)
    return _test_binary


TestBinary = type('TestBinary', (DynamicClassBase,), {})
for func in nestedtensor.nested.codegen.extension.get_binary_functions():
    setattr(TestBinary, "test_{0}".format(func),
            _gen_test_binary(func))

if __name__ == "__main__":
    unittest.main()
