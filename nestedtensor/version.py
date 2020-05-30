__version__ = '0.0.1.dev20205303+4813bbe'
git_version = '4813bbea8ba18085a976da2bf2c36bf646547c7c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
