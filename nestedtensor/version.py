__version__ = '0.0.1.dev20209245+02c4949'
git_version = '02c49492d6cda0b247fc2fe29fccdeeb6e365ea9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
