__version__ = '0.1.4+6332a1e'
git_version = '6332a1e7a86aa480cd5d620baf9fe2fea779d163'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
