__version__ = '0.1.4+cf94b79'
git_version = 'cf94b7954ecd4375afd75ee002538f76447c35f8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
