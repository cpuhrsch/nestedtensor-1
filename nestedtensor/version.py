__version__ = '0.1.4+422ac44'
git_version = '422ac447e1c07eaca97ca1aa0115cd5820cf2835'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
