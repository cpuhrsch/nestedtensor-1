__version__ = '0.0.1.dev202011722+9837dbd'
git_version = '9837dbd5099ec1eeb36bcb7507074f3602700ba4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
