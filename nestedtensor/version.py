__version__ = '0.0.1.dev20207717+49c428c'
git_version = '49c428ca59618dd6738ceea351b9c8828d5c3e1e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
