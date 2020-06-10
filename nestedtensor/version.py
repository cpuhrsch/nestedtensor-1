__version__ = '0.0.1.dev202061020+66facba'
git_version = '66facba4775e41577402e2ec219027b1158b4251'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
