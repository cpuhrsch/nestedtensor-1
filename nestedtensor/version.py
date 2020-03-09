__version__ = '0.0.1.dev20203917+ae345b8'
git_version = 'ae345b844dae7ae36b34cce82cf58e1ce8f8dd7f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
