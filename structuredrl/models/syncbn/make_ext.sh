#!/usr/bin/env bash

PYTHON_CMD=${PYTHON_CMD:=python}
CUDA_PATH=/usr/local/cuda-8.0
CUDA_INCLUDE_DIR=/usr/local/cuda-8.0/include
GENCODE="-gencode arch=compute_61,code=sm_61 \
         -gencode arch=compute_52,code=sm_52 \
         -gencode arch=compute_52,code=compute_52"
NVCCOPT="-std=c++11 -x cu --expt-extended-lambda -O3 -Xcompiler -fPIC"

ROOTDIR=$PWD
echo "========= Build BatchNorm2dSync ========="
if [ -z "$1" ]; then TORCH=$($PYTHON_CMD -c "import os; import torch; print(os.path.dirname(torch.__file__))"); else TORCH="$1"; fi
cd modules/functional/_syncbn/src
$CUDA_PATH/bin/nvcc -c -o syncbn.cu.o syncbn.cu $NVCCOPT $GENCODE -I $CUDA_INCLUDE_DIR
cd ../
$PYTHON_CMD build.py
cd $ROOTDIR

# END
echo "========= Build Complete ========="
