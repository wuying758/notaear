# 正式生成模型训练

## 首轮建议

第一轮先运行 10,000 optimizer steps，不直接承诺全量收敛。学习率计划可预留
100,000 steps，这样 10k 检查通过后能从同一 checkpoint 继续，不会改变前 10k 的
学习率轨迹。

RTX 3060 已验证的保守配置：

```bash
conda activate notaear
cd /home/wuyinghao/Desktop/wq/notaear
python -m notaear.train \
  ../data/notaear-manifests/v-yidiao-v1/manifest.jsonl \
  --output-dir out/generator-v1 \
  --device cuda --precision auto \
  --max-steps 10000 --schedule-steps 100000 --warmup-steps 2000 \
  --batch-size 4 --batch-patches 512 --gradient-accumulation 4 \
  --workers 2 --log-every 20 --eval-every 1000 --checkpoint-every 500
```

`precision=auto` 在当前显卡上选择 bf16。首版训练使用 `condition_dim=0`：难度模型尚未
校准并写入 manifest，不能用全零伪标签训练一个条件接口。难度条件应在无条件基线通过
生成评测后单独加入。

## 输出

- `latest.pt`：周期保存及训练结束位置；
- `best.pt`：验证 loss 最低的位置；
- `report.json`：模型、数据签名、训练参数、loss 历史及峰值 CUDA 显存。

checkpoint 包含模型、AdamW、AMP scaler、Python/Torch/CUDA RNG、epoch 和下一 batch。
保存只发生在 optimizer step 边界。manifest SHA-256 或影响样本顺序的训练参数变化时，
恢复会拒绝继续。

续训到 20,000 steps 时必须复用原始不可变参数：

```bash
python -m notaear.train \
  ../data/notaear-manifests/v-yidiao-v1/manifest.jsonl \
  --output-dir out/generator-v1 \
  --resume out/generator-v1/latest.pt \
  --device cuda --precision auto \
  --max-steps 20000 --schedule-steps 100000 --warmup-steps 2000 \
  --batch-size 4 --batch-patches 512 --gradient-accumulation 4 \
  --workers 2 --log-every 20 --eval-every 1000 --checkpoint-every 500
```

## 已验证范围

- 微型 CPU 任务：连续两步与一步后恢复到第二步，所有模型参数逐位相同；
- 正式 manifest：531,706 个 train、5,526 个 validation 样本成功建立索引；
- RTX 3060：17,086,080 参数、bf16、正式数据完成两步训练及两批验证；
- 两步 smoke loss 为 5.605 → 5.046，验证 loss 4.908。

最后一项只证明训练管线可运行，不证明生成质量或模型已经收敛。
