# NotaEar 小模型基线

## 默认结构

NotaEar-Small 是 NotaGen-style 的双层因果 Transformer，但面向单逻辑声部视唱，
规模显著缩小：

| 项目 | 默认值 |
| --- | ---: |
| byte 词表（含特殊 token） | 264 |
| patch size | 32 |
| 最大 patch 数 | 512 |
| hidden size | 384 |
| patch Transformer | 6 层、6 头 |
| patch 内 Transformer | 3 层、6 头 |
| FFN 宽度 | 1536 |
| 难度条件入口 | 5 维 |

上层在预测第 p 个 patch 时只能看到此前的完整 patch；下层在生成 patch 内第 i 个
byte 时只能看到此前的 byte。训练代码显式右移两层输入，防止 teacher forcing
泄漏。

## 生成约束 v0.2

推理不是直接从 264 个 token 中无约束抽样。当前状态机执行以下规则：

- header 后必须先生成 `BAR`，不能直接结束空谱；
- 未闭合且已占满物理 patch 的逻辑小节只能进入 `CONT`；
- 一个逻辑小节检测到顶层终止小节线后，才允许开始下一小节或输出 `DOC_EOS`；
- 引号、和弦、装饰与注释内部的 `|` 由 ABC 切分器处理，不误作终止小节线；
- 最终 ABC 必须满足 `encode(decode(tokens)) == tokens`，否则生成失败；
- patch 预算耗尽但小节未闭合时明确报错，不静默截断或伪造小节线。

推理时每个新 patch 的上层上下文只计算一次；patch 内逐 byte 采样复用该上下文，
并通过增量接口验证其 logits 与完整前向一致。当前下层仍会重算已生成的短前缀，
尚未实现 KV cache。

这些是 patch/canonical 结构约束；随后由独立验证器检查 music21 解析、单声部、
逐小节拍长、三连音实际时值、弱起配对、调外音比例和当前调终止音。非主音终止
默认只报告 warning，避免重新引入生硬的“必须回主音”规则。

## 与 NotaGen 的差别

- 保留“小节/patch 上层 + patch 内字符下层”的核心结构；
- 使用完整 UTF-8 byte 词表，不受 ASCII 128 限制；
- patch 默认 32，最终值仍由全量长度审计决定；
- 默认约 17M 参数，先建立能快速迭代的基线；
- 单声部任务不使用多声部轮转标记；
- 五维难度作为独立连续条件，不写入需要输出的 ABC 正文。

## 为什么暂不直接用 SSM

GPT-style 指因果语言建模范式，经典 GPT 主干是 Transformer 自注意力，并不等于
SSM。Mamba/S4 等 SSM 的优势主要出现在很长序列；这里上层已经把字符压缩为 patch，
512 patch 的注意力仍然可控。

第一轮实验固定 tokenizer 和数据划分，比较：

1. NotaEar-Small 双层 Transformer；
2. 仅将 patch 上层替换为 Mamba-2，保留下层 byte Transformer；
3. 参数量、训练 token 数和采样策略保持接近。

比较验证损失之外，还必须比较 ABC 可解析率、小节时值闭合率、调性稳定性、节奏
分布距离和人工可唱性。SSM 只有在这些指标上占优才替换基线。

## 初次真实数据审计

在 21 个非空来源分层抽取 404 首、共 46,868 个小节：

- tokenizer 字符级往返 404/404；
- 小节正文中位数 12 bytes，P95 为 32 bytes，P99 为 55 bytes；
- patch size 32 的有效载荷为 30 bytes，可让 94.13% 小节只占一个 patch；
- 99.31% 小节不超过两个 patch；
- 92.33% 抽样曲目不超过 512 patches。
