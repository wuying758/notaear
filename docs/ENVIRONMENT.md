# NotaEar 环境

当前验证环境：

- Conda 环境名：`notaear`；
- Python 3.12.13；
- PyTorch 2.14.0+cu130；
- NVIDIA RTX 3060 12 GiB，驱动 595.91.07；
- CUDA bf16 矩阵乘已实测通过。

创建方式：

```bash
conda create -y -n notaear python=3.12 pip
conda activate notaear
python -m pip install torch==2.14.0 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -e '.[label]' pytest \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pytest -q
```

清华源只负责镜像下载，并不自动保证 GPU 构建。安装后必须检查：

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(torch.cuda.is_bf16_supported())
PY
```

DataLoader 多进程使用 `spawn`，避免 Python 3.12 多线程进程和已初始化 CUDA 状态下
调用 `fork()` 的死锁风险。
