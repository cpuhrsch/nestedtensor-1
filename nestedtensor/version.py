__version__ = '0.0.1.dev20205305+aa41a51'
git_version = 'aa41a51a2a1620dd03ada2955971228e707d961f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
