# NotaEar 调式识别基准

调式准确率必须同时报告测试集性质、覆盖率与拒判率，不能把“成功解析”或 ABC
中的 `K:` 声明当作独立真值。

## synthetic-v1

仓库内的 `data/evaluation/tonality/synthetic-v1.jsonl` 是固定理论回归集：

- `core`：14 种调式 × 12 个主音 × 2 种旋律模板，共 336 条；
- `challenge`：3 种无法唯一确定调式的音级集合 × 12 个主音，共 36 条；
- 所有谱例使用 `K:none` 和逐音显式临时记号，不向分析器泄漏答案；
- `complete_scale` 检查完整调式材料；
- `modal_motif` 检查改变音序和旋律方向后是否仍能识别；
- `challenge` 的正确行为主要是 `ambiguous`/`insufficient`，不是强猜一个标签。

构建与评估：

```bash
notaear-tonality-benchmark build
notaear-tonality-benchmark evaluate \
  data/evaluation/tonality/synthetic-v1.jsonl \
  --output out/tonality-benchmark-v1/report.json
```

核心指标包括主音准确率、调式准确率、联合准确率、macro-F1、高置信覆盖率及高置信
联合准确率。挑战集另报告拒判率和错误自信率。

## 不能由 synthetic-v1 证明的内容

理论旋律的标签无歧义，但不能代表真实乐谱中的调式交替、缺失特征音、终止式模糊、
民族音乐语汇或错误元数据。因此 synthetic-v1 只能作为算法回归基准，不能写成
“真实调式识别准确率”。

下一阶段需要从正式 manifest 按来源、候选调式和拒判状态分层抽取真实谱例，隐藏
模型输出后人工标注主音、调式、可比性及转调区段。真实集至少应报告：

- tonic accuracy；
- mode macro-F1；
- tonic + mode joint accuracy；
- selective accuracy/coverage 曲线；
- modulation boundary F1；
- 标注者一致性。

当前盲标入口、字段定义与重复样本一致性审计见
[`TONALITY_LABELING.md`](TONALITY_LABELING.md)。

## declared-v1 真实谱弱标签集

`declared-v1.jsonl` 只保存正式 manifest test split 中的安全定位信息和原始 `K:`
弱标签，不复制乐谱正文。评估时仍用原调号解析实际音高，但关闭调式分析器的
`K:` 先验，因此模型不能直接抄标签。

构建器排除多重/内联换调、无法识别的声明，以及会被 music21 静默替换成音符的
裸导航符号（例如部分 PDMX 中用 `S` 表示 Segno）。这些样本不能通过猜测修复后
混入准确率统计。

```bash
notaear-tonality-benchmark build-declared \
  ../data/notaear-manifests/v-yidiao-v1/manifest.jsonl
notaear-tonality-benchmark evaluate-declared \
  data/evaluation/tonality/declared-v1.jsonl \
  ../data/notaear-manifests/v-yidiao-v1/manifest.jsonl \
  --output out/tonality-benchmark-v1/declared-report.json
```

`K:` 可能错误、简化或只表示记谱调号，所以这里的数字必须称为“声明弱标签一致率”，
不能称为人工金标准准确率。它的价值是用真实旋律结构发现 synthetic-v1 看不到的
问题，并为后续人工复核提供固定候选集合。

## 2026-09-30 基线结果

| 基准 | 样本 | 联合准确率 | 高置信覆盖率 | 高置信联合准确率 |
| --- | ---: | ---: | ---: | ---: |
| synthetic-v1 core | 336 | 100.0% | 100.0% | 100.0% |
| synthetic-v1 challenge | 36 | 不适用 | 拒判 66.7% | 错误自信 33.3% |
| declared-v1 | 149 | 40.9% | 43.0% | 40.6% |

排除 30 条含裸 `S` 导航符号、会被 music21 静默替换音高的候选后，
`declared-v1` 的主音准确率为 63.8%，调式准确率为 42.3%，mode macro-F1 为
0.460。分调式联合准确率为：major 36.0%、natural minor 46.0%、Dorian
39.6%、Phrygian 100%（但只有 1 条，不能据此概括）。高置信子集没有比总体更准，
说明当前启发式置信度尚未在真实旋律上校准。

因此当前分析器能可靠处理证据完整的理论材料，但真实短旋律中调式旋转与缺失特征音
仍是主要失败点；Dorian 常与 Shang、major 和 natural minor 混淆，major 与
Mixolydian/五声候选也经常混淆。下一步不是继续调 synthetic 模板或移动阈值，而是
人工复核 declared-v1，并补充当前完全缺失的 Mixolydian、五声调式及和声/旋律
小调真实谱例。
