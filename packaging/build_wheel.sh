#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE=wheel
setup_env 0.8.0
setup_wheel_python
pip_install numpy pyyaml future ninja
git submodule sync
git submodule update --init --recursive
printf "* Installing NT-specific pytorch and nestedtensor with cuda\n"
USE_DISTRIBUTED=OFF BUILD_TEST=OFF USE_CUDA=ON BUILD_CAFFE2_OPS=0 USE_NUMPY=ON USE_NINJA=1 ./clean_build.sh

# Copy binaries to be included in the wheel distribution
if [[ "$(uname)" == Darwin || "$OSTYPE" == "msys" ]]; then
    python_exec="$(which python)"
    bin_path=$(dirname $python_exec)
    env_path=$(dirname $bin_path)
    if [[ "$(uname)" == Darwin ]]; then
        # Include LibPNG
        cp "$env_path/lib/libpng16.dylib" nestedtensor
        # Include LibJPEG
        cp "$env_path/lib/libjpeg.dylib" nestedtensor
    else
        cp "$bin_path/Library/bin/libpng16.dll" nestedtensor
        cp "$bin_path/Library/bin/libjpeg.dll" nestedtensor
    fi
else
    # Include LibPNG
    cp "/usr/lib64/libpng.so" nestedtensor
    # Include LibJPEG
    cp "/usr/lib64/libjpeg.so" nestedtensor
fi

if [[ "$OSTYPE" == "msys" ]]; then
    pushd third_party/pytorch
        IS_WHEEL=1 "$script_dir/windows/internal/vc_env_helper.bat" python setup.py bdist_wheel
    popd
    IS_WHEEL=1 "$script_dir/windows/internal/vc_env_helper.bat" python setup.py bdist_wheel
else
    pushd third_party/pytorch
        IS_WHEEL=1 python setup.py bdist_wheel
    popd
    IS_WHEEL=1 python setup.py bdist_wheel
fi
