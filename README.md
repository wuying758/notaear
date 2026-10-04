# NotaEar

NotaEar 是与旧 `deepear-main` 隔离的 NotaGen-style 单声部视唱生成实验目录。

当前包含：

- 无损 ABC `header + bar patch` 切分；
- patch 级三连音与内联调号/拍号元数据；
- UTF-8 byte tokenizer 与固定长度 patch；
- 约 17M 参数的双层因果 Transformer 基线；
- 可恢复的版本化 checkpoint 与 ABC continuation 推理入口；
- 作品级去重、无泄漏划分和按 epoch 选调的训练 manifest；
- 来源加权、作品级同调、分片局部读取和动态 padding 的 PyTorch DataLoader；
- 大小调、中古调式、三类小调与宫商角徵羽的保守候选分析；
- 五维视唱难度特征，以及面向用户的暂定 L1--L5 支持度接口；
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
PYTHONPATH=notaear/src python -m pytest -q notaear/tests
PYTHONPATH=notaear/src python -m notaear.abc_patches path/to/tune.abc
PYTHONPATH=notaear/src python -m notaear.difficulty path/to/tune.abc
PYTHONPATH=notaear/src python -m notaear.model
PYTHONPATH=notaear/src python -m notaear.train_smoke path/to/shard.jsonl \
  --device cuda --checkpoint out/generator-smoke/checkpoint.pt
PYTHONPATH=notaear/src python -m notaear.train_smoke path/to/shard.jsonl \
  --device cuda --steps 100 --resume out/generator-smoke/checkpoint.pt
PYTHONPATH=notaear/src python -m notaear.generate \
  out/generator-smoke/checkpoint.pt path/to/prompt.abc \
  --output out/generator-smoke/generated.abc \
  --validation-report out/generator-smoke/generated.validation.json
PYTHONPATH=notaear/src python -m notaear.validate_abc path/to/tune.abc
PYTHONPATH=notaear/src python -m notaear.build_manifest \
  data/handoff-full-2026-09-26/handoff/v-yidiao/score-grade \
  data/notaear-manifests/v-yidiao-v1 --workers 8
PYTHONPATH=notaear/src python -m notaear.dataset \
  data/notaear-manifests/v-yidiao-v1/manifest.jsonl \
  --split train --workers 2 --batches 3
PYTHONPATH=notaear/src python -m notaear.train \
  data/notaear-manifests/v-yidiao-v1/manifest.jsonl \
  --output-dir notaear/out/generator-v1 --device cuda
```

`--steps` 表示总目标步数；例如从 step 50 的 checkpoint 继续到 100，应传
`--steps 100`。checkpoint 包含模型配置、优化器、训练步数、随机状态和所选数据
签名；恢复时数据签名不一致会拒绝继续。PyTorch checkpoint 使用 pickle，只加载
自己训练或可信来源的文件。

当前生成入口已约束 `HEADER/BAR/CONT/DOC_EOS` 转换，并要求输出解码后重新编码
得到完全相同的 canonical patch 序列。生成预算耗尽时若小节仍未闭合，程序会明确
失败，不会强塞 EOS。生成后还会检查 music21 解析、单声部、逐小节拍长、三连音
实际时值、弱起配对和分段调性；默认不写出验证失败结果，可用 `--allow-invalid`
仅作诊断。非主音终止默认是 warning，只有 `--require-final-tonic` 才会硬拒绝。

指标定义见 [`docs/DIFFICULTY.md`](docs/DIFFICULTY.md)。
调式识别的版本化测试集与指标见
[`docs/TONALITY_BENCHMARK.md`](docs/TONALITY_BENCHMARK.md)。
真实谱例的调性盲标与重复样本审计见
[`docs/TONALITY_LABELING.md`](docs/TONALITY_LABELING.md)。
模型设计见 [`docs/MODEL.md`](docs/MODEL.md)。
训练数据 manifest 的字段、筛选边界和复现命令见
[`docs/DATA_MANIFEST.md`](docs/DATA_MANIFEST.md)。
正式 Dataset 与 DataLoader 的行为和训练接入方式见
[`docs/DATA_LOADER.md`](docs/DATA_LOADER.md)。
正式 GPU 训练、续训和 checkpoint 约定见
[`docs/TRAINING.md`](docs/TRAINING.md)。


Melodia 弱监督难度模型：

```bash
PYTHONPATH=src python -m notaear.train_difficulty train \
  data/reference_books/raw/melodia/Cole-Lewis-1904-xml.zip \
  --output out/difficulty-melodia-v1
```
