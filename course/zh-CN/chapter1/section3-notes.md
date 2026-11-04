# 第 1 章第 3 节《Transformers 能做什么？》笔记

> 课程原文：Hugging Face LLM 课程（zh-CN）第 1 章第 3 节，配套 Notebook 为本仓库 `course/zh-CN/chapter1/section3.ipynb`（从 Colab 保存）。
> 本笔记按 2026-11-04 核对的 transformers v5 口径更新。课程正文是 v4 时代写的，其中几处写法在 v5 已经删除或弃用，照抄会报错，先看第 4 节对照表。
> 代码没有全部实测：标「实测记录」的是我自己在 Colab 跑过的；其余更新代码按官方迁移指南和官方文档改写，跑之前先用 5.1 查自己环境的版本和可用任务列表。

## 1. 本节一句话重点

`pipeline()` 是全节唯一的核心工具，它把三步打包了：

1. 预处理：把文本变成模型能理解的格式；
2. 模型推理；
3. 后处理：把模型输出变成人能看的结果。

换任务就是换第一个参数，换模型就是加 `model=`。本节的定位是演示和试用，课程结尾也说了：这些 pipeline 是为特定任务定制的、不能自定义修改，下一章才讲内部运作。

## 2. 还能照用的任务（重点）

| 任务（规范名） | 课程里的写法 | 干什么 | 要记住的点 |
| --- | --- | --- | --- |
| `text-classification` | 别名 `sentiment-analysis` | 判断一句话的类别，如 POSITIVE / NEGATIVE + 分数 | 默认模型是英文情感模型，分数是它对类别的把握 |
| `zero-shot-classification` | 同名 | 不训练、不用例子，给候选标签就能分类 | 原理是把每个标签套成一句假设做文本蕴含（NLI）判断；分数只是给定标签之间的相对排序，合计约 1，换一组标签全变 |
| `text-generation` | 同名 | 给开头续写文本 | 有随机性，每次结果不同；长度用 `max_new_tokens` 控制（见第 4 节） |
| `fill-mask` | 同名 | 完形填空，`top_k` 控制给几个候选 | 不同模型的 mask token 不同，换模型先确认它用 `<mask>` 还是别的 |
| `token-classification` | 别名 `ner` | 命名实体识别：抠出 PER 人名 / ORG 机构 / LOC 地点 / MISC 其他专名 | 分组参数已换成 `aggregation_strategy="simple"`（见第 4 节） |
| `feature-extraction` | 课程列表里有 | 取得文本的向量表示 | 本节只列出，没有展开演示 |

`sentiment-analysis` 和 `ner` 现在是别名，规范名分别是 `text-classification` 和 `token-classification`。别名还能用，新写的代码优先用规范名。

## 3. 我的实测记录（Colab）

- 零样本分类（默认英文模型 `facebook/bart-large-mnli`）：输入「我是一名程序员」，候选标签 [学习, 职业, 编程]，输出排序为 编程 0.5943、学习 0.2474、职业 0.1582。能跑通，但默认是英文 NLI 模型，跑中文分数不可信；中文要换多语言模型，如 `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`（轻量）或 `joeddav/xlm-roberta-large-xnli`（large，慢）。
- NER（默认英文模型）：输入「我是一名程序员，我在宙思科技上班」，返回空列表 `[]`。原因：默认模型 `dbmdz/bert-large-cased-finetuned-conll03-english` 只认英文实体；而且「程序员」是职业、不是专有名词，本来就不在 NER 的识别范围里。这不是报错。
- 日志里两个常见提示都不是错误：
  - `No model was supplied, defaulted to ...`：没指定模型时用了默认值，真实代码要写死 `model=` 和 `revision=`；
  - 加载 NER 模型时 `bert.pooler.dense ... UNEXPECTED`：这个任务用不上 pooler 权重，忽略即可。

## 4. 已淘汰写法对照表（本节的更新重点）

| 旧写法（课程正文） | 问题 | 新写法 |
| --- | --- | --- |
| `pipeline("ner", grouped_entities=True)` | `grouped_entities` 已弃用，v5 删除 | `pipeline("ner", aggregation_strategy="simple")`，新代码也可写规范名 `token-classification` |
| `pipeline("summarization")` | v5 删除了 Text2Text 系列 pipeline（含 Summarization / Translation） | 二选一：用聊天模型 + 提示词走 `text-generation`（官方迁移指南推荐，见 5.6）；或绕过 pipeline，用 `AutoModelForSeq2SeqLM` + `generate()` 继续用 BART / T5 等老模型（见 5.7）。模型本身没删，删的只是 pipeline 封装 |
| `pipeline("translation", model=...)` 和 `pipeline("translation_en_to_fr")` | 同上，Translation pipeline 在 v5 删除，把语言对写进任务名的做法一并失效 | 同 summarization 两条路线；多语言模型（如 NLLB）走 generate 路线时用 `src_lang` / `tgt_lang` 指定语言 |
| `pipeline("question-answering")` | 官方 v5 迁移指南已列入移除（个别 v5 小版本节奏不同，以自己环境的任务列表为准，见 5.1） | 聊天模型 + 提示词：把上下文和问题一起给它，并要求只用上下文回答。注意区别：旧 pipeline 是从上下文里抽取原文片段，聊天模型是生成答案，要自己核对有没有超出上下文 |
| 文本生成用 `max_length=30` | `max_length` 含输入提示的长度，且用配置控制它已被弃用 | 改用 `max_new_tokens`，只数新生成的 token；要多个不同结果再加 `num_return_sequences` 并配合 `do_sample=True` |
| `model="distilgpt2"` | 裸名是旧写法 | 现在的规范 ID 是 `distilbert/distilgpt2` |
| 所有示例不写模型、直接用默认模型 | 默认值会随版本变，运行时也会警告不建议生产使用 | 每个 pipeline 都写明 `model=`，需要可复现再加 `revision=`（分支名、标签或提交哈希） |
| 「推理 API」一节：网页小组件试用 + 直连 `api-inference.huggingface.co` | 旧的 Inference API（serverless）已下线，旧端点在 2025 年底退役，直连会返回 410 / 404 | 改为 Inference Providers：Python 用 `huggingface_hub` 的 `InferenceClient`，HTTP 走 `https://router.huggingface.co/v1`；Token 要建细粒度 Token 并勾选 Make calls to Inference Providers 权限（见 5.8） |

## 5. 更新后的代码

### 5.1 先查环境（每次开新 Colab 先跑）

```python
import transformers
from transformers.pipelines import get_supported_tasks

print(transformers.__version__)
print(get_supported_tasks())
```

任务名在不在这个列表里，是判断「写法淘汰还是我写错了」的依据。

### 5.2 情感分析（规范名 + 写明模型）

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)
classifier("I've been waiting for a HuggingFace course my whole life.")
```

### 5.3 零样本分类（写明模型；中文换多语言模型）

```python
from transformers import pipeline

# 英文 / 复现课程默认结果
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 中文改用这一行（轻量多语言模型）
# classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

classifier("我是一名程序员", candidate_labels=["学习", "职业", "编程"])
```

### 5.4 文本生成（新模型 ID + max_new_tokens）

```python
from transformers import pipeline

generator = pipeline("text-generation", model="distilbert/distilgpt2")
generator(
    "In this course, we will teach you how to",
    max_new_tokens=30,
    num_return_sequences=2,
    do_sample=True,
)
```

### 5.5 NER（aggregation_strategy 代替 grouped_entities）

```python
from transformers import pipeline

ner = pipeline("ner", aggregation_strategy="simple")
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
# 预期实体：Sylvain = PER，Hugging Face = ORG，Brooklyn = LOC
```

### 5.6 摘要 / 翻译 / 问答的新路线：聊天模型 + 提示词

官方 v5 迁移指南的口径：这三类任务统一改用 `text-generation` + 聊天模型，换任务只换提示词。指南的摘要示例用的是 `Qwen/Qwen3-4B-Instruct-2507`；4B 模型在免费 Colab 上偏重，内存不够就换更小的指令模型，写法不变。

```python
from transformers import pipeline

chat = pipeline("text-generation", model="Qwen/Qwen3-4B-Instruct-2507")

def ask(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return chat(messages)[0]["generated_text"][-1]["content"]

# 摘要
print(ask("Summarize the following text:\n\n" + long_text))

# 翻译（法译英）
print(ask("Translate the following French text into English:\n\nCe cours est produit par Hugging Face."))

# 问答（抽取式问答的替代：要求只用上下文）
context = "My name is Sylvain and I work at Hugging Face in Brooklyn."
print(ask(f"Answer the question using only the context below.\n\nContext: {context}\n\nQuestion: Where do I work?"))
```

### 5.7 摘要的老模型路线：绕过 pipeline 用 generate()

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

inputs = tokenizer(long_text, return_tensors="pt", truncation=True)
output_ids = model.generate(**inputs, max_new_tokens=60)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

翻译同理：换成 Marian / NLLB 等 seq2seq 模型走 `generate()`，多语言模型记得传 `src_lang` / `tgt_lang`。

### 5.8 Inference Providers（代替旧 Inference API）

```python
import os
from huggingface_hub import InferenceClient

client = InferenceClient()  # 默认读取环境变量 HF_TOKEN

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "How many 'G's in 'huggingface'?"}],
)
print(completion.choices[0].message)
```

Token 在 hf.co/settings/tokens 里建细粒度 Token，勾选 Make calls to Inference Providers；不要把 Token 写进代码或提交到仓库。

## 6. 如果只想原样跑完这节课

临时办法是把版本锁回 v4 再重启 Colab 运行时，课程代码就能按原文跑：

```
!pip install "transformers<5"
```

这只是为了跟课的权宜之计，不要带进新项目；新项目按第 4、5 节的 v5 写法。

## 7. 和后面学习的关系

- 本节是 W1 之前的热身：先建立「任务、模型、分数该怎么读」的直觉，不深挖每个任务。
- 带走的四条通用规矩：任务和模型分开选；模型必须和任务、语言匹配（默认模型多为英文）；分数多为相对值或置信度，别当绝对结论；真实代码写死模型名和版本。
- 下一章拆 pipeline 内部（tokenizer、模型本体、后处理），本节这些「黑盒」到时会逐个打开。

## 参考

- 课程原文：https://huggingface.co/learn/llm-course/zh-CN/chapter1/3
- Transformers v5 迁移指南（Pipelines 一节）：https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md
- Pipelines 官方文档：https://huggingface.co/docs/transformers/main_classes/pipelines
- Inference Providers 官方文档：https://huggingface.co/docs/api-inference/en/quicktour
