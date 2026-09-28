# NotaEar

NotaEar 是与旧 `deepear-main` 隔离的 NotaGen-style 单声部视唱生成实验目录。

当前包含：

- 无损 ABC `header + bar patch` 切分；
- patch 级三连音与内联调号/拍号元数据；
- 五维、无总分的视唱难度特征；
- 单元测试与真实 `v-yidiao` 抽样入口。

计划中的数据流：

```text
v-yidiao JSONL ABC
  -> 无损 header/bar patches
  -> NotaGen-style patch decoder + character/lexical decoder
  -> canonical ABC
  -> 独立难度分析与评测
```

完整的模块边界、训练阶段和验收门槛见
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 本地运行

```bash
cd /home/wuyinghao/Desktop/wq
PYTHONPATH=notaear/src python -m unittest discover -s notaear/tests -v
PYTHONPATH=notaear/src python -m notaear.abc_patches path/to/tune.abc
PYTHONPATH=notaear/src python -m notaear.difficulty path/to/tune.abc
```

指标定义见 [`docs/DIFFICULTY.md`](docs/DIFFICULTY.md)。

