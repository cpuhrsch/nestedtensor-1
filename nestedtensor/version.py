__version__ = '0.0.1.dev2020210+ea1bd66'
git_version = 'ea1bd666cbc4ac2125c77b99f87867bd7660f601'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
