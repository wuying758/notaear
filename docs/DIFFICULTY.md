# NotaEar 多维视唱难度指标 v0.2

原始分析层输出可解释特征，不在规则代码里手写总分。面向用户的稳定接口最终
输出整体 `L1`--`L5`；内部模型保留连续潜在难度和五级支持度，用教材顺序与人工
成对比较学习四个等级边界。连续值只用于训练、排序和边界不确定性，不作为主要
用户评分。

当前 Melodia 基线已提供暂定的五级映射：Series 1/3/5/7/9 分别对应
`L1`--`L5`。这是接口兼容层，不代表五个 Series 已被证明是等距的教育测量等级。
公开结果至少包含：

```json
{
  "level": "L3",
  "confidence": 0.74,
  "confidence_calibrated": false,
  "borderline": true,
  "alternative": "L4"
}
```

当前 `confidence` 是树模型的相对支持度，并未校准为“预测正确的概率”，因此
同时输出 `confidence_calibrated: false`。五级分布字段命名为 `level_support`，
不称为概率。诊断输出可以附带五维百分位、调式候选和模型不确定性。

## 1. 调性语言 `tonal_language`

- 声明调性与调式；
- 调号升降号数量；
- 中途换调次数；
- 调外音比例；
- 使用的不同音级数量；
- 末音是否为主音。

`difficulty.py` 现已接入保守的调式分析器。它支持大调、自然/和声/旋律小调、
Dorian、Phrygian、Lydian、Mixolydian、Locrian 和宫商角徵羽，并返回前三候选、
置信度、拒判状态与分段调式。Ionian/Aeolian 分别规范化为 `major` 和
`natural_minor`，避免重复候选。

分析综合音级分布、时值、强拍、首尾音、ABC `K:` 声明及旋律小调的上下行证据。
显式内联 `K:` 会形成确定的分析区段；没有声明时，只有连续窗口均达到置信门槛
才报告推断转调。音符或音级过少时返回 `insufficient`，候选过近时返回
`ambiguous`，不会强制选成可靠标签。

候选中的数值字段叫 `support`，全局置信度标记为
`confidence_calibrated: false` 和 `confidence_kind: heuristic_fit_margin`。它们用于
候选排序和拒判，不是调式正确率的统计校准概率。

调式是难度证据而非难度等级。Dorian 的自然六级、和声小调的升七级等组成音
不会再被直接计为调外音。学习者是否熟悉某种调式属于独立画像，不能写死进谱面
固有难度。

## 2. 旋律运动 `melodic_motion`

- 最低音、最高音、总音域；
- 10%–90% tessitura（避免极少数极端音主导音域）；
- 平均与最大音程；
- 大于大二度的跳进比例；
- 大于纯五度的大跳比例；
- 连续跳进比例；
- 旋律方向改变比例；
- 同音反复比例。

## 3. 节奏与节拍 `rhythm_and_meter`

- 拍号及拍号变化次数；
- 最短音符；
- 时值种类数与时值熵；
- 三连音/连音组数量和连音音符比例；
- 附点音符比例；
- 弱拍起音比例；
- 跨拍切分比例；
- 休止事件比例；
- 延音线数量。

三连音是本维度中的一个连续特征，不是默认生成标签。之后还应按三连音所在
拍位、连续组数及与普通二分节奏的切换次数细分。

## 4. 记谱与读谱负担 `notation_and_reading`

- 临时升降号数量及每音符密度；
- 装饰音与倚音组数量；
- 反复记号和多结尾数量；
- 连奏线数量；
- 和弦事件数量；
- 小节 patch 数量。

同音异名、增减音程及跨小节临时记号作用域将在 ABC 词法 tokenizer 完成后加入。

## 5. 形式、记忆与呼吸 `form_memory_and_breath`

- 音符总数及每小节音符密度；
- 三音程动机的重复比例；
- 长休止之间的最大连续音符数；
- 长休止之间的最大持续拍数；
- 连线延续音比例。

后续将增加乐句边界、模进、终止式、重复段跨度和推荐换气点。声乐音域必须
结合具体声部/年龄设置，不能用一套固定上下限决定所有人的难度。


## Melodia 弱监督模型 v0

训练集只取 Melodia 的单声部 Series 1、3、5、7、9。Series 2、4、6、8、
10、11 主要是双声部材料，其核心困难包含声部独立性，不与单旋律难度混合。
Series 6 中三个单声部转录也暂时排除，以保持课程定义一致。

Extra Trees 模型输出最接近的 Melodia Series、支持度加权的连续 Series 位置、
树投票支持度和投票熵。五个维度另以 Melodia 训练分布中的百分位表示。这些结果
都是教材弱监督，不是教师评分或考试等级。

交叉验证按照移调不变的旋律轮廓分组。同一旋律即使被移到不同调，也不会同
时进入训练折和验证折。当前验证只衡量 Melodia 内部泛化，下一步仍需使用
Gilchrist、Eyes and Ears 或真实学生表现做外部验证。

训练：

```bash
PYTHONPATH=src python -m notaear.train_difficulty train \
  data/reference_books/raw/melodia/Cole-Lewis-1904-xml.zip \
  --output out/difficulty-melodia-v1
```

评估一个 ABC：

```bash
PYTHONPATH=src python -m notaear.train_difficulty score \
  out/difficulty-melodia-v1/difficulty_model.joblib path/to/tune.abc
```

训练输出包括 `difficulty_model.joblib`、交叉验证 `report.json` 和带分组标识的
`features.csv`。`out/` 被 Git 忽略，模型产物不会误推送到仓库。

### 2026-09-30 基线结果（当前 `notaear` 环境重训）

- 训练谱例：1,148；移调不变重复组：1,094；
- 五折分组交叉验证精确 Series 命中率：92.07%；
- 误差不超过相邻一个阶段：99.48%；
- 平均绝对误差：0.084 个五阶段索引，等价于 0.169 个 Series 编号；
- 支持度期望与真实课程阶段的 Spearman 相关：0.951。

这些是同一本教材内的结果，可能受编排和转录风格影响，不能当作跨教材准确率。
这组数字不能单独用于正式数据筛选，必须结合下面的外部教材验证。

### 2026-09-30 外部验证

模型未使用 Eyes and Ears 或 Concone 训练。验证不强行把外部教材映射为
Melodia Series 真值，而是检验预测值与原教材课程顺序的 Spearman 相关。

| 验证集 | 样本 | Spearman | 平均不确定度 |
| --- | ---: | ---: | ---: |
| Eyes and Ears 单声部 | 420 | 0.587 | 0.435 |
| Eyes: Major 章内 | 137 | 0.558 | 0.417 |
| Eyes: Minor 章内 | 60 | 0.673 | 0.468 |
| Concone P1 独唱旋律 | 90 | 0.489 | 0.422 |

Eyes 五个顺序区间的平均预测 Series 为 3.00、4.99、5.40、5.40、5.91。
节奏章中，基本节奏、八分音符、切分、三连音的节奏百分位分别约为 5、26、
49、79。Major 章的级进、主三和弦跳进、六度跳进的旋律百分位分别约为 8、
58、75。这些变化与教材概念顺序基本一致。

外部不确定度明显高于 Melodia 内部的 0.184，说明模型对分布外教材没有表现
出虚假的高置信度。当前五项项目验收检查全部通过，但门槛是工程基线而非教育
测量学标准。

已知失败点是 C 谱号。模型读取的是声音内容，因此无法表达读者对 C clef 的
熟悉程度；该负担应作为用户/课程画像特征加入，而不是从音符难度中臆测。
Gilchrist 目前没有本地结构化逐题谱，只用于概念核对，未纳入数值验证。

复现：

```bash
PYTHONPATH=src python -m notaear.validate_difficulty \
  out/difficulty-melodia-v1/difficulty_model.joblib \
  data/reference_books/raw/eyes_and_ears/Crowell-2004-xml.zip \
  data/reference_books/raw/concone/Concone-1897-xml.zip \
  --output out/difficulty-melodia-v1/external_validation.json
```

## 人工成对标注与主动学习

人工标注不直接填写 L1–L5，而是盲测两条旋律：选择 A 更难、差不多、B 更难
或不可比，并记录主要困难维度与 1–3 级把握。页面在提交前隐藏教材来源、章节和
模型预测，避免锚定偏差。模型优先抽取高不确定、预测难度相近而维度构成不同的
谱例；Melodia 与外部教材混合出现，以同时校正教材内排序和跨教材偏移。

安装排版依赖并生成首批 60 对：

```bash
python -m pip install -e '.[label]'
PYTHONPATH=src python -m notaear.label_difficulty prepare
PYTHONPATH=src python -m notaear.label_difficulty serve
```

然后打开 `http://127.0.0.1:8765`。结果逐行追加到
`data/annotations/difficulty_pairs.jsonl`，该文件应纳入 Git；抽样队列在 `out/`
中，不纳入 Git。快捷键 1/2/3 分别对应 A 更难/差不多/B 更难，0 为跳过。

每个谱例可以用页面内的播放器试听。服务通过 Verovio 生成 MIDI，再用本机
FluidSynth 和统一钢琴音色合成 WAV；结果缓存在 `out/active-label-v0/audio/`。
标注会分别记录 A、B 是否播放过，以便训练时区分纯读谱判断和试听辅助判断。

训练按 30 个有效比较组成一个批次，不在每次点击后立刻更新。这样可以固定一批
题目的选样分布，重训后再用保留验证集比较排序相关、成对准确率与校准误差；只有
全部门槛未退化时才替换当前模型。`skip` 只用于质量审计，不进入训练。

首批完成后训练人工校正头：

```bash
PYTHONPATH=src python -m notaear.train_pairwise_difficulty
```

它使用带置信度权重的成对 Logistic 排序头，并把每个方向样本连同其左右翻转副本
一起训练，以消除 A/B 位置偏差。交叉验证按比较图的连通分量分组，同一谱例不会
跨训练折和验证折。`equal` 用于学习平局阈值，`skip` 不训练。

当前输出是 `baseline_expected_series + alpha * human_correction` 的候选融合分数，
不是直接覆盖 Melodia 基线。只有人工方向一致性提高、未见 Melodia 顺序不退化，
且外部 Eyes 顺序至少持平时，才允许进入下一轮主动学习。

### 2026-09-30 人工校正复核

- 60 对标注中有 49 对方向判断、10 对“差不多”和 1 对跳过；
- 当前 v1 基线在这些方向比较上的交叉验证准确率为 59.2%；
- 融合人工头后为 85.7%，单独人工头为 87.8%；
- 未参与人工比较的 Eyes 顺序相关由 0.525 提升至 0.558；
- 未参与人工比较的 Melodia 顺序相关由 0.961 轻微下降至 0.960；
- 三分类准确率为 71.2%，但“差不多”的召回率只有 10%。

因此人工头仍是候选校正器，不能替代公开 L1--L5。下一轮应重点补充边界附近的
`equal` 样本，并使用独立标注者或独立测试批次验证，而不是继续调当前 60 对的阈值。

`joblib` 内含 scikit-learn 对象，不保证跨版本安全兼容。训练报告现记录 Python、
NumPy、scikit-learn 和 joblib 版本；模型应在相同环境加载，或在新环境中重训并
重新执行外部验证。
