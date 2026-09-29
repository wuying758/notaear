# v-yidiao 训练 manifest v1

## 目的

`v-yidiao` 的 15 个目标调目录是数据增强版本，不是 15 份独立训练样本。manifest
以“逻辑声部”为一行，仅保存各调所在的 `[分片文件名, JSONL 行号]`，训练时再按
epoch 选择一个调。这样不会把移调副本同时塞进一个 epoch，也不会让同一作品的不同
声部或重复来源跨越 train/validation/test。

## 复现

```bash
cd /home/wuyinghao/Desktop/wq/notaear
PYTHONPATH=src python -m notaear.build_manifest \
  ../data/handoff-full-2026-09-26/handoff/v-yidiao/score-grade \
  ../data/notaear-manifests/v-yidiao-v1 \
  --workers 8 --patch-size 32 --max-patches 512
```

输出包括：

- `manifest.jsonl`：可训练候选；
- `rejected.jsonl`：被排除的记录及原因；
- `summary.json`：划分、来源、权重和错误统计；
- `index.sqlite`：构建中间索引和断点状态。

输入路径、patch 大小或最大 patch 数变化时，已有索引会拒绝混用。已索引分片的
大小或修改时间变化也会失败关闭，应使用新的输出目录重建。

## 一行记录的语义

关键字段如下：

- `example_id`：一个逻辑声部；
- `score_id`：作品标识，同一作品的所有声部共用 split 和每轮目标调；
- `work_fingerprint`：由作品内各声部的规范化内容组成的精确指纹；
- `variants`：目标调到 `[shard, line_number]` 的映射；
- `split`：`train`、`validation` 或 `test`；
- `sampling_weight`：来源平衡用的逐样本权重；
- `patch_count`、`bar_count`、`has_triplet`：代表版本的结构统计。

`notaear.manifest.choose_variant(record, epoch, seed=...)` 会确定性选择本轮版本；同一
`score_id` 的多个声部会选择相同目标调。

## v1 筛选和划分规则

1. ABC 必须能被无损小节切分器和 byte tokenizer 往返编码；
2. 排除无小节、含多音和弦或超过 512 patches 的声部；
3. 优先用 C 调版本生成内容指纹；同指纹作品只保留有效声部比例最高的来源版本；
4. 用作品指纹稳定哈希进行 98% / 1% / 1% 划分；
5. 来源权重以 `count ** 0.5` 为目标总采样质量，逐样本权重归一到均值 1，最大为 10。

## 当前正式构建结果

- 输入逻辑声部：822,097；
- 训练候选：542,551；拒绝：279,546；
- train / validation / test：531,706 / 5,526 / 5,319；
- 完整 15 调候选：519,725；少于 15 调：12,826；
- JSON 摄入错误：0；
- 随机 100 条共 1,499 个 locator 回读错误：0；
- 作品指纹跨 split 泄漏：0。

## 已知边界

- 这是训练候选的结构清洗，不等于每条都通过 `music21` 的严格拍长和调性验证；
- 当前只做精确内容去重，不做旋律近似去重；
- 缺少 C 调版本的少量记录使用其他可用调作代表，跨调等价去重能力较弱；
- `sampling_weight` 是来源平衡先验，不是质量分或难度分；
- 难度标签不写入这个 manifest，避免把尚未校准的弱监督分数固化为事实。
