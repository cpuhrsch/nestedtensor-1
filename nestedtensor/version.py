__version__ = '0.0.1.dev202061122+fd7cccb'
git_version = 'fd7cccb07c6ae11fa87b70270503a2189777b638'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
