__version__ = '0.1.4+257bbbc'
git_version = '257bbbc3a275d6d3833e6e3716799adfcc581750'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
