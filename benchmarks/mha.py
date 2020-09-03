import torch
import nestedtensor
import utils
import torchvision
from torch.nn import functional as F

import random


class DETRNestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(
            *args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    @classmethod
    def from_tensor_list(cls, tensor_list):
        # TODO make this more general
        if tensor_list[0].ndim == 3:
            # TODO make it support different-sized images
            max_size = tuple(max(s)
                             for s in zip(*[img.shape for img in tensor_list]))
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = (len(tensor_list),) + max_size
            b, c, h, w = batch_shape
            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1],
                        : img.shape[2]].copy_(img)
                m[: img.shape[1], :img.shape[2]] = False
        else:
            raise ValueError('not supported')
        return cls(tensor, mask)


# Performance tanks hard for lots of small Tensors as expected
DEVICE = torch.device('cpu')
NDIM = 128
BSZ = 8
NHEAD = 8
MODEL = torch.nn.MultiheadAttention(NDIM, NHEAD).to(DEVICE).eval()


def run_benchmark(low, high):
    RAND_INTS = [(random.randint(low, high), random.randint(low, high)) for _ in range(BSZ)]
    src_ = nestedtensor.nested_tensor(
        [torch.arange(NDIM * i * j).float().reshape(NDIM, i, j) for (i, j) in RAND_INTS], device=DEVICE, dtype=torch.float)
    src = []
    for i, s in enumerate(src_):
        src.append(i*len(s) + s)

    def gen_t_loop_mha(src):
        detr_nt_src = DETRNestedTensor.from_tensor_list(src)
        src, mask = detr_nt_src.decompose()
        src = src.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        def t_loop():
            MODEL(src, src, src, key_padding_mask=mask,
                  need_weights=False)  # [0].sum().backward()

        return t_loop

    def gen_nt_mha(src):
        src = nestedtensor.nested_tensor([t.flatten(1).permute(
            1, 0) for t in src], device=DEVICE, dtype=torch.float)

        def nt():
            MODEL(src, src, src, need_weights=False)

        return nt

    print(utils.benchmark_fn(gen_t_loop_mha(src)))
    print(utils.benchmark_fn(gen_nt_mha(src)))


if __name__ == "__main__":
    random.seed(1011)
    torch.manual_seed(1011)
    run_benchmark(25, 35)
