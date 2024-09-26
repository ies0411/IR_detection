ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN ln -sv /usr/bin/python3 /usr/bin/python
ENV PATH="/root/.local/bin:${PATH}"

# Install MMEngine , MMCV and MMDet
RUN pip install --no-cache-dir openmim && \
    mim install --no-cache-dir "mmengine>=0.6.0" "mmcv>=2.0.0rc4,<2.1.0" "mmdet>=3.0.0,<4.0.0"

RUN pip install --user albumentations natsort yapf==0.40.1 fairscale ensemble-boxes

RUN python3 -m pip install --user jupyter
