__version__ = '0.0.1.dev202011523+5b1df70'
git_version = '5b1df70e894aac16db3e412db98ac8cfeda53b88'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
