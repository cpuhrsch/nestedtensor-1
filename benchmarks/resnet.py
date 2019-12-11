# from nestedtensor import torch
import nestedtensor
import torch

from torchvision import models

from utils import benchmark_fn

def gen_algorithm_regular():
    model = models.resnet18(pretrained=False).cuda()
    nested_images = torch.rand(128, 3, 40, 50).cuda()
    # nested_images = torch.rand(1, 3, 40, 50).cuda()

    def _regular():
        model(nested_images)

    return _regular

def gen_algorithm_nested():
    model_ = models.resnet18(pretrained=False).cuda()
    model = torch.tensorwise()(model_)
    
    # There is still about a 10x gap in performance, which however
    # can be significantly alleviated via custom code (e.g. using im2col).
    images = [torch.rand(1, 3, (i * 16) % 40 + 40, (i * 16) % 50 + 40).cuda() for i in range(64)]
    nested_irregular_images = torch.nested_tensor(images)

    def _nested():
        model(nested_irregular_images)

    return _nested

if __name__ == "__main__":
    regular = gen_algorithm_regular()
    print(benchmark_fn(regular))

    nested = gen_algorithm_nested()
    print(benchmark_fn(nested))
