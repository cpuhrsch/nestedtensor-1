__version__ = '0.0.1.dev202062318+0b2c814'
git_version = '0b2c8144bd91be1c47b79aa4874f95f09af87346'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
