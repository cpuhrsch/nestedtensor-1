__version__ = '0.0.1.dev202061722+818600c'
git_version = '818600ca53675b942f2a903c71c1258e4f51c8fa'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
