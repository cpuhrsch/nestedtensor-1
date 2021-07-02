__version__ = '0.1.4+9f46d5f'
git_version = '9f46d5f06a807b50430b5eea0bf4ec6057649aba'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
