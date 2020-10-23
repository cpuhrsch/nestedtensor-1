__version__ = '0.0.1.dev202010233+0d60e05'
git_version = '0d60e053d6e7008c6c23a12357266893b3646305'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
