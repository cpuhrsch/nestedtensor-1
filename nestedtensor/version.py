__version__ = '0.0.1.dev202011722+61c9776'
git_version = '61c9776d6cfbceb7c77e51b4f67875f835606a0b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
