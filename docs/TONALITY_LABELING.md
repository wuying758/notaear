# 调性/调式盲标

`notaear-label-tonality` 用真实 `declared-v1` 候选建立人工金标准。它不是让标注者确认
ABC 的 `K:`，而是隐藏来源、调名、弱标签和模型输出后，独立判断主音与调式。

## 为什么仍保留调号

调号是谱面本身的一部分，视唱者实际会看到；删除调号会改变临时记号语义和读谱任务。
服务端先按原始 ABC 解析实际音高，再导出匿名 MusicXML。导出时保留等价调号，但删除
标题、歌词、和弦标记、文本说明以及 MusicXML 的 tonic/mode 元数据。浏览器只收到 SVG，
不会收到 ABC、MusicXML、来源定位、弱标签或模型预测。

## 建立队列并启动

```bash
cd /home/wuyinghao/Desktop/wq/notaear
conda activate notaear

notaear-label-tonality prepare
notaear-label-tonality serve
```

打开 <http://127.0.0.1:8766>。标准队列包含 149 个真实候选，以及 10% 随机重复的
隐藏审计项，共 164 次判断。页面支持播放；每条记录保存 `listened`，之后可分别报告
纯读谱与听后判断。播放器提供播放/暂停、当前时间/总时长和可拖动定位的时间轴；音频
端点支持 HTTP byte ranges，因此不必等待整段 WAV 下载完才能跳转。标注自动追加到
`data/annotations/tonality_gold_v1.jsonl`，重启服务会从未完成项继续。

不要把普通 `K:Am` 自动理解成“自然小调”。它只提供小调大类；如果旋律中没有足够的
第六、七级及方向证据，自然、和声、旋律小调无法唯一判断。界面因此允许“小调（种类
不明）”“中古调式（具体型不明）”“五声调式（具体型不明）”和“有主音但调式不明”。
这些部分标签比强猜的精确标签更适合训练和评估。

## 简单校准轮

严格校验后的首轮包含 15 条不同谱例和 2 条隐藏重复：

```bash
notaear-label-tonality prepare \
  --profile easy --count 15 --namespace tonality-easy-v2 \
  --output out/tonality-label-easy-v3/queue.json
notaear-label-tonality serve \
  --queue out/tonality-label-easy-v3/queue.json \
  --annotations data/annotations/tonality_gold_v2.jsonl
```

简单集不读取当前模型预测，只用与模型无关的谱面证据筛选：8--32 小节、32--200 个音、
至少 98% 符合声明调式、主音至少出现四次、结尾为声明主音，并在大调、小调和 Dorian
之间限额取样。此外，每个内部小节必须严格符合拍号，弱起只能出现在首尾且互相补足，
必须保持单声部，匿名 MusicXML、SVG 和 MIDI 三条实际 UI 路径也要预检通过。它适合熟悉
标注规范和建立第一批高质量标签，但由于刻意选择明确终止，不能单独用于宣称真实世界
准确率。能解析或能渲染不等于乐谱正确；失败项应隔离，不能自动补休止符后继续使用。

每条应标：

- 可判断时：主音、调式、是否中途转调、把握程度；
- 证据不足时：选择“无法唯一判断”，不要强猜；
- 数据损坏或谱面无法使用时：选择“谱面错误 / 跳过”。

## 审计

```bash
notaear-label-tonality audit \
  --output out/tonality-label-v1/audit.json

notaear-label-tonality evaluate-human \
  --output out/tonality-label-easy-v3/human-evaluation.json
```

审计报告区分两件事：

- `repeat_consistency` 是同一标注者对隐藏重复项的一致性；
- `human_vs_declared_weak_reference` 只是人工结果与 ABC `K:` 元数据的一致率，不能称为
  人工准确率。

单人盲标能生成第一版评估集，但不能测量标注者间一致性。用于论文或正式模型验收前，
应由第二位具备乐理判断能力的标注者独立复标同一固定子集，再报告 Cohen's kappa 或
类别不均衡更稳健的 Krippendorff's alpha，并对冲突项仲裁。
