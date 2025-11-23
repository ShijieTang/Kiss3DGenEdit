# 创建环境
conda create --name kiss3dgen python=3.10 -y
conda activate kiss3dgen

pip install -U pip

# 安装 CUDA 12.1（conda 版本）
conda install cuda -c nvidia/label/cuda-12.1.0 -y

# 安装 gcc-12 / g++-12
apt-get update
apt-get install -y gcc-12 g++-12

# 设置默认编译器为 gcc-12
echo 'export CC=gcc-12' >> ~/.bashrc
echo 'export CXX=g++-12' >> ~/.bashrc
# 当前 shell 立刻生效（重新开终端时会自动生效）
source ~/.bashrc

# 安装 PyTorch & xformers（CUDA 12.1 对应的 cu121 轮子）
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.27.post1

# 安装 Pytorch3D
pip install iopath
pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html

# 安装 torch-scatter
pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# ========= 这里是 nvdiffrast，用 wget 下载 + pip 安装 =========
mkdir -p /workspace
cd /workspace

# 下载 wheel（Python 3.10 + CUDA 12 对应的一个现成轮子）
wget "https://huggingface.co/spaces/neil-ni/Unique3D/resolve/69ac8ac1b4c6efd2684d69805e6437b58ab554d3/package/nvdiffrast-0.3.1.torch-cp310-cp310-linux_x86_64.whl" \
     -O nvdiffrast-0.3.1.torch-cp310-cp310-linux_x86_64.whl

# 安装 nvdiffrast
pip install /workspace/nvdiffrast-0.3.1.torch-cp310-cp310-linux_x86_64.whl
# ========= nvdiffrast 部分结束 =========

# 回到项目目录（假设你在 ~/Kiss3DGenEdit）
cd ~/Kiss3DGenEdit

# 安装其他依赖
# ⚠️ 如果 requirements.txt 里还有一行
#   nvdiffrast @ file:///workspace/nvdiffrast-0.3.1%2Btorch-py3-none-any.whl ...
# 建议先把那一行注释掉，否则 pip 会再尝试找那个本地文件
pip install -r requirements.txt
