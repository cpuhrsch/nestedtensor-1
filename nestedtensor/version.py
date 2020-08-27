__version__ = '0.0.1.dev202082722+b30bd8a'
git_version = 'b30bd8a64e2e4a38b53acf163967d4441433461f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
