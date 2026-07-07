FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# 禁用 Python 输出缓冲，确保日志实时输出
ENV PYTHONUNBUFFERED=1
ENV DGLBACKEND=pytorch
ENV DGL_HOME=/tmp/.dgl
ENV MPLCONFIGDIR=/tmp/matplotlib

# 安装 Python 和系统依赖（保持这层稳定）
# libxrender1 / libxext6: openbabel-wheel 的 viewmolformat 等 plugin 在加载时
# 链接 libXrender.so.1, libXext.so.6；缺失会让 OBConversion 整个 plugin 注册失败，
# SetInAndOutFormats("sdf","pdb") 返回 False，下游 RTMScore extract_pocket 静默
# 写空 PDB → "The graph of pocket cannot be generated"。
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
    wget \
    build-essential \
    g++ \
    libxrender1 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN pip3 install --upgrade pip setuptools wheel

# ✅ 单独 COPY requirements.txt，提高缓存命中率
COPY requirements.txt /tmp/requirements.txt

# ✅ 分步安装依赖，使用缓存加速重建
# 第1步：基础科学计算库
RUN pip3 install numpy==1.20.3 scipy==1.6.2 pandas==1.0.3

# 第1.5步：安装兼容 numpy 1.20.3 的 Pillow 版本
RUN pip3 install "pillow==8.3.1"

# 第2步：构建依赖
RUN pip3 install "Cython>=0.29.0,<3.0" "setuptools<65.0"

# 第3步：PyTorch (CUDA 11.1) - 只安装 torch,不需要 torchvision
RUN pip3 install \
    --extra-index-url https://download.pytorch.org/whl/cu111 \
    torch==1.9.0+cu111

# 第4步：DGL (CUDA 11.1)
RUN pip3 install \
    --find-links https://data.dgl.ai/wheels/repo.html \
    dgl-cu111==0.7.0

# 第5步：PyTorch 几何库
RUN pip3 install --no-build-isolation \
    torch-scatter==2.0.9 torch-sparse==0.6.12 torch-cluster==1.5.9

# 第6步：化学和生物信息学库
RUN pip3 install \
    rdkit-pypi==2021.3.5.1 \
    openbabel-wheel \
    biopandas==0.2.7 \
    ProDy==2.1.0

# 第7步：MDAnalysis (需要特殊处理)
RUN pip3 install --no-build-isolation MDAnalysis==2.0.0

# 第8步：机器学习和可视化
RUN pip3 install \
    scikit-learn==0.24.2 joblib==1.0.1 \
    matplotlib==3.4.3 seaborn==0.11.2

# 修复权限问题：给 Python 包目录设置写权限
RUN chmod -R 777 /usr/local/lib/python3.8/dist-packages/ || true

# 创建配置目录
RUN mkdir -p /tmp/.dgl /tmp/matplotlib && chmod -R 777 /tmp/.dgl /tmp/matplotlib

# ✅ 设置工作目录
WORKDIR /app

# 挂载点说明：
# - /data       -> 工作目录（rundir）
# - /checkpoint -> 模型权重目录（checkpointdir）