__version__ = '0.0.1.dev202081219+c2d6a6d'
git_version = 'c2d6a6d4facefba03d6d5560a76e3a524ba98470'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
