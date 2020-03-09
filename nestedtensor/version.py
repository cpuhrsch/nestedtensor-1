__version__ = '0.0.1.dev20203921+7d705b4'
git_version = '7d705b4baff388f1738b7c31b5aa90d14d029c2b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
