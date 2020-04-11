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
            a = torch.randn(2, 3)
            nt = constructor([a])
            # NOTE: For torch.sum dtype can be a positional for full reductions.
            self.assertEqual(a.sum(), torch.sum(nt))
            self.assertEqual(a.sum(), nt.sum())
            self.assertEqual(a.sum(dtype=torch.int64), nt.sum(dtype=torch.int64))

            print(constructor([a.sum(0, keepdim=False)]))
            print(nt.sum(dim=1, keepdim=False))
            import sys; sys.exit(1)
            self.assertEqual(
                constructor([a.sum(0, keepdim=False)]), nt.sum(dim=1, keepdim=False)
            )
            self.assertEqual(
                constructor([a.sum(0, keepdim=True)]), nt.sum(dim=1, keepdim=True)
            )
            self.assertEqual(
                constructor([a.sum(0, keepdim=False, dtype=torch.int64)]),
                nt.sum(dim=1, keepdim=False, dtype=torch.int64),
            )
            self.assertEqual(
                constructor([a.sum(0, keepdim=True, dtype=torch.int64)]),
                nt.sum(dim=1, keepdim=True, dtype=torch.int64),
            )

            print(torch.sum(nt, dim=1, keepdim=False, out=nt))
            print(torch.sum(nt, dim=1, keepdim=True, out=nt))
            print(torch.sum(nt, dim=1, keepdim=False, dtype=torch.int64, out=nt))
            print(torch.sum(nt, dim=1, keepdim=True, dtype=torch.int64, out=nt))


if __name__ == "__main__":
    unittest.main()
