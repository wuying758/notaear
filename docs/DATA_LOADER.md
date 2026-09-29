# Manifest Dataset 与 DataLoader

## 数据契约

`ManifestABCDataset` 读取 `manifest.jsonl` 的一个 split。它不会把 626 MiB manifest
或 13 GiB ABC 增广语料展开进 Python 对象内存，只保留 manifest 字节偏移、来源采样
权重和代表版本 patch 数。训练 worker 按 locator 读取原始 JSONL，并用小型 LRU 缓存
最近的分片。

`ManifestBatchSampler` 每个 epoch 执行：

1. train split 按 `sampling_weight` 做有放回抽样，epoch 样本数保持不变；
2. validation/test 默认不加权、不打乱；
3. 对同一 `score_id` 的已抽中声部求可用调交集，再确定性选择目标调；
4. 先按目标调和源分片聚合，减少随机文件读取；
5. 将小分片余数按最大样本数和最大 padded patch 数重新装箱；
6. 输出 `SampleRef`，由 worker 校验 locator、编码 ABC 并动态 padding。

极长乐谱存在少量“代表调不超过 512、其他调超过 512”的情况。Dataset 会从同一作品
确定性选调顺序继续寻找可用版本，并在 batch 的 `variant_fallbacks` 中显式标记，不会
静默截断 ABC。

## Python 接入

```python
from notaear.dataset import create_manifest_dataloader

dataset, batch_sampler, loader = create_manifest_dataloader(
    "../data/notaear-manifests/v-yidiao-v1/manifest.jsonl",
    split="train",
    patch_size=32,
    max_patches=512,
    max_examples_per_batch=8,
    max_patches_per_batch=2048,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    seed=7,
)

for epoch in range(num_epochs):
    batch_sampler.set_epoch(epoch)
    for batch in loader:
        patches = batch["patches"].to(device, non_blocking=True)
        output = model(patches)
```

batch 同时返回 `patch_counts`、`example_ids`、`score_ids`、`sources`、`target_keys`、
`requested_target_keys` 和 `variant_fallbacks`。padding 使用 tokenizer 的 `PAD_ID`，
模型 loss 已忽略该 token。

## 命令行检查

```bash
PYTHONPATH=src python -m notaear.dataset \
  ../data/notaear-manifests/v-yidiao-v1/manifest.jsonl \
  --split train --workers 2 --batches 3
```

正式 v1 manifest 的实测结果：

- train：531,706 个 epoch 样本，batch size 8、预算 2048 时规划 67,435 批；
- validation：5,526 条，规划 695 批，理论最少 691 批；
- train 完整规划并读取前三批约 21.5 秒；
- 两个 worker 的进程峰值 RSS 约 680 MiB。

这些是当前机器上的工程基准，不是跨机器性能承诺。当前 sampler 尚未实现分布式
多 GPU 分片；17M 基线先使用单 GPU，后续如确有吞吐瓶颈再增加 distributed sampler。
