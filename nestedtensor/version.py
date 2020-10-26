__version__ = '0.0.1.dev202010262+815423d'
git_version = '815423d2e297a80a428b236a33166bbb4c6868d5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
