---
companies:
- meta-ai-fair
- huggingface
- alibaba
- openai
date: '2025-09-13T05:44:39.731046Z'
description: '**Meta** 在 Hugging Face 上发布了 **MobileLLM-R1**，这是一个参数量低于 10 亿（sub-1B）的推理模型系列。该系列模型在
  4.2T token 上进行训练，在小模型数学准确率方面表现强劲。**阿里巴巴**推出了 **Qwen3-Next-80B-A3B**，采用混合注意力机制，具备
  256k 上下文窗口和改进的长程记忆能力，并在阿里云上提供了极具竞争力的定价。**Meta AI FAIR** 修复了 SWE-Bench 基准测试中的一个错误，该错误曾影响对智能体（agent）的评估。**LiveMCP-101**
  基准测试显示，像 **GPT-5** 这样的前沿模型在处理复杂任务时表现不佳，并对常见的失败模式进行了分类记录。**OpenAI** 强调了由于基准测试激励机制导致的幻觉问题，并提出了校准改进方案。社区演示和工具更新也在持续演进。'
id: MjAyNS0w
models:
- mobilellm-r1
- qwen3-next-80b-a3b
- gpt-5
people:
- _akhaliq
- tacocohen
- pkirgis
- sayashk
title: 今天没发生什么特别的事。
topics:
- reasoning
- model-efficiency
- hybrid-attention
- long-context
- benchmarking
- agent-evaluation
- hallucination-detection
- model-calibration
- inference-complexity
- model-pricing
---

**平静的一天。**

> 2025年9月11日至9月12日的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 22 个 Discord 社区（189 个频道，5258 条消息）。预计节省阅读时间（以 200wpm 计算）：464 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 风格展示所有往期内容。详见 https://news.smol.ai/ 并通过 @smol_ai 向我们提供反馈！

[o1 周年](https://x.com/kimmonismus/status/1966627812858855624?s=46)快乐。祝贺 [Naveen Rao](https://x.com/rohanpaul_ai/status/1966378718009635087?s=46) 和 [Interaction](https://x.com/interaction/status/1965093198482866317) 获得备受关注的新融资。

---

# AI Twitter 回顾

**端侧推理：Meta 的 MobileLLM-R1（1B 以下规模）在 HF 开源**

- **MobileLLM-R1（1B 以下规模，开源权重）**：Meta 在 Hugging Face 上发布了一系列参数量低于 1B 的推理模型，其在小模型上的表现异常强劲：根据 [@_akhaliq](https://twitter.com/_akhaliq/status/1966498058822103330) 和模型发布[链接](https://twitter.com/_akhaliq/status/1966499598433710361)，其 MATH 准确率比 Olmo-1.24B 高出约 5 倍，比 SmolLM2-1.7B 高出约 2 倍，同时在多个推理基准测试中达到或超过了 Qwen3 的准确率，而训练数据仅为 4.2T tokens（约占 Qwen3 36T 数据的 11.7%）。Meta 研究人员强调了该规模下的数据效率和推理能力（[公告](https://twitter.com/erniecyc/status/1966511167053910509)，[更多背景](https://twitter.com/zechunliu/status/1966560134739751083)）。社区演示通过 Anycoder/Spaces 迅速上线（[应用](https://twitter.com/_akhaliq/status/1966528137858019713)，[另一个](https://twitter.com/_akhaliq/status/1966532295403209138)）。

**Qwen3-Next-80B (A3B)：混合注意力机制、256k 上下文以及对基础设施的重大影响**

- **架构与推理复杂度**：阿里巴巴新开源权重的 Qwen3-Next-80B-A3B 引入了混合注意力设计（Gated DeltaNet + Gated Attention），具有极高的稀疏性（活跃参数约 3.8%，而 Qwen3-235B 为 9.4%），原生支持 256k 上下文窗口，且仅支持纯文本 I/O。适配工作需要对引擎进行重大更改：根据 [@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1966419946885493098)，SGLang 的 PR 超过 6k 行代码；vLLM 超过 2.5k 行代码。阿里云上的定价为：推理版每 1M 输入/输出 token 为 $0.5/$6，非推理版为 $0.5/$2，比 Qwen3-235B 更便宜（[详情](https://twitter.com/ArtificialAnlys/status/1966523300781428788)，[token 使用情况](https://twitter.com/ArtificialAnlys/status/1966523306338893979)）。
- **性能与权衡（社区评估）**：长程“工作记忆”和多轮一致性有明显提升；字符级基础能力很强，但推理+字符任务表现参差不齐；弱点包括错误继承、指令遵循差距以及长文本幻觉，详见知乎分析（[摘要](https://twitter.com/ZhihuFrontier/status/1966415278922989813)，[推特串](https://twitter.com/ZhihuFrontier/status/1966419946885493098)）。另一份综述将 Qwen3-Next-80B 置于综合指数接近 DeepSeek V3.1 的位置，且 token 使用量更低（[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1966523300781428788)）。

**Agent、评估修复与失败取证分析**

- **SWE-Bench 修复，进展依然真实**：FAIR Codegen 的 [@TacoCohen](https://twitter.com/TacoCohen/status/1966421688846778561) 指出了一个允许 Agent 偷窥未来 commit 的问题，SWE-Bench 迅速修复了该漏洞。初步重新运行结果显示，大多数模型并未受到严重影响；FAIR 仅在将 RL 运行规模扩大到“好得令人难以置信”的结果后才发现了这个 bug。建议：实验室和开源项目应在修复后的基准测试上重新发布结果并清晰标注。
- **实时的任务型评估极具挑战**：LiveMCP-101 引入了一个实时 Agent 框架/基准测试，强调超越合成设置的复杂任务。即使是前沿模型也表现不佳：GPT-5 在“困难”任务上的得分为 39.02%；顶级模型的总体得分仍低于 60%。论文列举了七种常见的失败模式（忽视要求、过度自信地自我解决、错误的工具选择、语法/语义/输出解析错误）（[概述](https://twitter.com/omarsar0/status/1966525731082768782)，[结果](https://twitter.com/omarsar0/status/1966525793586360384)，[论文](https://twitter.com/omarsar0/status/1966525809302417436)）。
- **校准优于猜测**：OpenAI 认为幻觉之所以持续存在，是因为基准测试奖励自信的猜测；修复方案包括不惩罚“我不知道”以及重新调整排行榜（[摘要](https://twitter.com/TheTuringPost/status/1966638472854483129)，[论文](https://twitter.com/TheTuringPost/status/1966638485282189600)）。在 AssistantBench 上，GPT-5 显示出比 o3 更高的精确度和更低的猜测率（[@PKirgis](https://twitter.com/PKirgis/status/1966547382033936577)）。HAL 正在添加 Docent 以分析 Agent 日志，而不仅仅是最终的准确率（[@sayashk](https://twitter.com/sayashk/status/1966550402129592738)）。

**工具、基础设施与库**

- **VS Code 扩展模型市场 API**：“Language Model Chat Provider”扩展 API 已定稿；BYOK（自带密钥）提供商可以作为扩展程序安装，以提供更多模型选择。同时发布的还有教程、视频和模型自动选择体验（例如 Claude、GPT-5/mini、Gemini）（[API 推文](https://twitter.com/code/status/1966638511269794238)，[Cerebras 扩展](https://twitter.com/code/status/1966638514100924846)，[发布说明](https://twitter.com/housecor/status/1966429828808352233)，[笔记](https://twitter.com/code/status/1966546512717946894)）。
- **Transformers v5 + 连续批处理 (continuous batching)**：HF 预告了 v5 现代化推进（更快的 Kernel、更智能的默认设置、代码清理），并悄然上线了连续批处理功能，以简化评估/训练循环（不追求最大吞吐量的服务器；重点在于实验/工具箱）（[v5](https://twitter.com/art_zucker/status/1966470835558093226)，[连续批处理](https://twitter.com/LucSGeorges/status/1966550465769775305)）。此外，“新的 LLM 发布现在将作为 Transformers 的 PR 进行宣布” ([@lvwerra](https://twitter.com/lvwerra/status/1966451134727352326))。
- **推理系统**：Meta 的 vLLM 解耦推理（disaggregated inference）显示出优于其内部堆栈的延迟/吞吐量优势；相关优化正被合并至上游 ([@PyTorch](https://twitter.com/PyTorch/status/1966546293733437799))。一篇关于 Paged Attention 的清晰解释也在流传（[链接](https://twitter.com/novasarc01/status/1966413957679428054)）。
- **AOT 与区域编译**：ZeroGPU 增加了区域 AOT 编译以及预编译图（precompiled graphs）的共享/加载，以加速启动过程（[帖子](https://twitter.com/RisingSayak/status/1966447203381092675)，[博客/文档](https://twitter.com/RisingSayak/status/1966447207688569028)）。
- **HF 中的视觉与检索**：Microsoft 的 Kosmos-2.5 已登陆 Transformers，并附带 OCR+布局演示/Notebook（[演示/文档](https://twitter.com/mervenoyann/status/1966487632659005667)，[Notebook](https://twitter.com/mervenoyann/status/1966488556831977672)）。MetaCLIP2 多语言模型以及文本到图像搜索 Notebook 也已发布（[公告](https://twitter.com/mervenoyann/status/1966544046744011242)，[教程](https://twitter.com/mervenoyann/status/1966544570436424074)）。
- 其他值得注意的：Skypilot 新的 GPU 利用率仪表板（[链接](https://twitter.com/skypilot_org/status/1966592871600890285)）；以及 Elon Musk 顺带提到“AMD 现在在处理中小型模型方面表现相当不错” ([@elonmusk](https://twitter.com/elonmusk/status/1966412913662669082))。

**前沿访问、SDK 与安全协作**

- **OpenAI 平台**：GPT-5 和 gpt-5-mini 的速率限制在各个层级都得到了大幅提升 ([@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1966610846559134140))。Codex-CLI 中出现了一个新的 “gpt-5-high-new” 目标（“经过调优以依赖内置推理默认设置”），尽管细节仍然很少 ([@mark_k](https://twitter.com/mark_k/status/1966521489529643169))。OpenAI 继续专注于扩展思考（extended thinking）：o1-preview 在网页+浏览+代码方面的表现从当前模型的“数小时”缩短到“数秒”，“未来还有很大的发展空间” ([@polynoamial](https://twitter.com/polynoamial/status/1966527147469598794), [@gdb](https://twitter.com/gdb/status/1966612991421423814))。
- **Anthropic**：英国 AISI 和美国 CAISI 一直在识别 Claude Opus 4/4.1 中的越狱（jailbreaks）行为，帮助发布更强大的安全防护措施（[公告](https://twitter.com/AnthropicAI/status/1966599335560216770)，[详情](https://twitter.com/AnthropicAI/status/1966599337426681899)，[AISI 推文](https://twitter.com/alxndrdavies/status/1966614120566001801)）。对于开发者，Claude Code SDK（与 CLI 使用相同的环境）是构建自定义 Agent 的推荐起点（[介绍](https://twitter.com/alexalbert__/status/1966601430808088596)，[文档](https://twitter.com/alexalbert__/status/1966601435153388019)）。
- **Qwen Code**：v0.0.10/11 版本增加了子 Agent、Todo Write 工具、“欢迎回来”项目摘要、编辑稳定性、更好的 IDE/shell 集成、改进的内存/会话管理等（[发布](https://twitter.com/Alibaba_Qwen/status/1966451235328008563)，[预览](https://twitter.com/Alibaba_Qwen/status/1966451500340703418)）。

**视觉模型与排行榜**

- **LMArena 更新**：在超过 4.3 万张选票中，Gemini 2.5 Flash Image (“nano‑banana”) 继续领跑图像编辑 (Image Edit) 和文本生成图像 (Text‑to‑Image) 排行榜；字节跳动 (ByteDance) 的 Seedream 4 目前在图像编辑中排名第 2，在 T2I 中排名第 5 ([排行榜](https://twitter.com/lmarena_ai/status/1966562484506230922), [更多](https://twitter.com/lmarena_ai/status/1966562486897029274))。新的 “Seedream 4 High Res” 变体支持 4096×4096 输出，并已在 Arena 上线 ([添加](https://twitter.com/lmarena_ai/status/1966673628327801255), [尝试](https://twitter.com/lmarena_ai/status/1966673632069132770))。
- **其他视觉模型发布**：腾讯的 HunyuanImage‑2.1 (2K T2I) 可通过 Anycoder/FAL 获取，用于快速应用原型设计 ([帖子](https://twitter.com/_akhaliq/status/1966684003877917145), [应用](https://twitter.com/_akhaliq/status/1966684046206906801))。

**隐私保护预训练**

- **VaultGemma**：Google Research 发布了 VaultGemma，这是一个拥有 1B 参数的 Gemma 变体，采用差分隐私 (differential privacy) 从头开始训练——据称是以此方式训练的最大开源模型——此外还发布了关于隐私 LM 训练的新 Scaling‑law 结果。权重和报告已发布 ([公告](https://twitter.com/GoogleResearch/status/1966533086914421000), [摘要](https://twitter.com/osanseviero/status/1966534013511672148), [模型](https://twitter.com/osanseviero/status/1966534014791020869), [论文](https://twitter.com/osanseviero/status/1966534140439728485))。

**热门推文（按互动量排序）**

- 关于 OpenAI–Oracle 假设性巨额交易的 “金钱如何运作” 飞轮讽刺，作者 [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1966553671866687689) (20.9k)。
- 犹他州州长 Spencer Cox 谈论社交媒体危害，作者 [@bensiegel](https://twitter.com/bensiegel/status/1966510619479118073) (12.9k)。
- 维基百科财务状况审查，作者 [@nearcyan](https://twitter.com/nearcyan/status/1966601978319904877) (10.4k)。
- AI 领导者原型讽刺，作者 [@sergeykarayev](https://twitter.com/sergeykarayev/status/1966506136481481090) (9.0k)。
- OpenAI 平台为 GPT‑5/mini 提升 Rate‑limit，作者 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1966610846559134140) (2.1k)。
- Elon 谈论用于中小型模型的 AMD GPU，作者 [@elonmusk](https://twitter.com/elonmusk/status/1966412913662669082) (2.2k)。
- Higgsfield 增长统计和产品速度，作者 [@higgsfield_ai](https://twitter.com/higgsfield_ai/status/1966588786080706842) (2.9k)。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Meta MobileLLM-R1 发布 + 每周 LocalLLaMA 模型/数据集汇总 (9月12日)

- [**Meta 在 Hugging Face 上发布 MobileLLM-R1**](https://i.redd.it/huchm6bahrof1.png) ([分数: 412, 评论: 46](https://www.reddit.com/r/LocalLLaMA/comments/1nf7zhq/meta_released_mobilellmr1_on_hugging_face/)): **Meta 在 Hugging Face 上发布了 MobileLLM‑R1‑950M ([模型卡片](https://huggingface.co/facebook/MobileLLM-R1-950M))，这是一个约** `950M` **参数的小型 LLM，旨在实现高效的端侧/移动端推理，并附带一个交互式演示 Space ([应用](https://huggingface.co/spaces/akhaliq/MobileLLM-R1-950M))，据报道该应用是通过 AnyCoder Space ([AnyCoder](https://huggingface.co/spaces/akhaliq/anycoder)) 构建的。帖子中未列出基准测试，但背景强调了在低参数端提升推理准确性，并提供适合轻量化部署的开源版本。** 评论者对小型模型推理准确性的工作表示赞赏，并感谢 Meta 仍在公开模型，一些人对其“完全开源”感到惊讶。
    - 强调在小参数前沿提升推理准确性：评论者强调了优化参数受限模型“下限”的价值，在这些模型中，训练、量化和解码策略的改进可以为端侧和低延迟场景带来不成比例的巨大实际收益。
    - 对基准测试的怀疑：一位用户指出，该模型在常见排行榜上的表现仍不如 `Qwen 0.6`（可能是约 0.6B 级别的 Qwen 变体），对其新颖性提出质疑。这引发了不仅需要评估原始准确性，还需要评估以移动端为指标（例如 CPU/NPU 上的 tokens/sec、峰值 RAM、4/8 位量化后的模型大小以及每 token 能耗）以及任何适用的 R1 式推理增益的需求。
    - 部署兴趣：对 `GGUF` 版本的需求表明用户希望获得 llama.cpp 兼容性和针对边缘设备的快速量化（例如 Q4_K_M/Q8_0），从而实现在没有 GPU 的笔记本电脑和手机上进行实际测试，并方便与其他 1B 以下模型的吞吐量和内存占用进行同类比较。

- [**上周在本版块发布或更新的模型列表，以防错过 - (9月12日)**](https://www.reddit.com/r/LocalLLaMA/comments/1neyaph/a_list_of_models_released_or_udpated_last_week_on/) ([Score: 273, Comments: 32](https://www.reddit.com/r/LocalLLaMA/comments/1neyaph/a_list_of_models_released_or_udpated_last_week_on/)): **每周汇总亮点：Qwen3‑Next‑80B‑A3B 推出了一款稀疏激活的 80B MoE 模型，每个 token 激活约 3B 参数（据称推理速度快约 10 倍，32k+ 上下文）[HF](https://huggingface.co/collections/Qwen/qwen3-next-68c25fd6838e585db8eeea9d) [发布](https://www.reddit.com/gallery/1nefmzr)；MiniCPM4.1‑8B 增加了混合推理（/think vs /no_think）和长上下文支持 [HF](https://huggingface.co/openbmb/MiniCPM4.1-8B)；Jan‑v1‑2509 声称改进了推理/创意评估 [HF](https://huggingface.co/janhq/Jan-v1-2509)；PyDevMini‑1 (4B) 声称以 1/400 的体积实现了 GPT‑4 级别的 Python/Web‑Dev 性能 [HF](https://huggingface.co/bralynn/pydevmini1)。语音/TTS：Qwen3‑ASR（仅限 API，多语言 EN/CN + 9）[demo](https://huggingface.co/spaces/Qwen/Qwen3-ASR-Demo) 和 IndexTTS‑2.0（表现力强、时长可控的 zero‑shot TTS）[repo](https://github.com/index-tts/index-tts)。推理/MoE 与研究：Aquif‑3 系列（包括 17B a2.8B GGUF）[HF](https://huggingface.co/mradermacher/aquif-3-moe-17b-a2.8b-GGUF)，ROMA 报告在 SEAL‑0/FRAMES 上优于闭源平台 [GitHub](https://github.com/sentient-agi/ROMA)，百度的 Ernie X1.1 目标是顶尖中文能力 [post](https://www.reddit.com/r/LocalLLaMA/comments/1ndjoek/new_ernie_x11_what_may_be_the_best_chinese_model/)；数据集包括 FinePDFs（3T tokens；5亿+ PDF）[HF](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) 和 LongPage（300 本带有推理轨迹的小说）[HF](https://huggingface.co/datasets/Pageshift-Entertainment/LongPage)。** 评论请求 llama.cpp 支持 Qwen Next，并指出同时发布的模型：Kwai‑Klear 的 Klear‑46B‑A2.5B‑Instruct [link](https://huggingface.co/Kwai-Klear/Klear-46B-A2.5B-Instruct) 和 inclusionAI 的 Ring‑mini‑2.0 [link](https://huggingface.co/inclusionAI/Ring-mini-2.0)。
    - 对 **Qwen** 的 llama.cpp 支持的兴趣表明了对 GGUF 量化以及通过 llama.cpp 内核（如 cuBLAS/Metal/Vulkan）进行轻量级 CPU/GPU 推理的需求。集成通常取决于 tokenizer/chat template 的兼容性（Qwen 通常使用 ChatML）以及 rotary/pos-embed 变体；跟踪 llama.cpp 的 PR 将明确 Qwen 完全适配的时间 ([llama.cpp](https://github.com/ggerganov/llama.cpp), [Qwen HF](https://huggingface.co/Qwen))。
    - 一位评论者指出 [Kwai-Klear/Klear-46B-A2.5B-Instruct](https://huggingface.co/Kwai-Klear/Klear-46B-A2.5B-Instruct) “恰好在 7 天前”发布。命名暗示这是一款 Mixture-of-Experts 风格的模型，总参数约为 `46B`，每个 token 激活约 `2.5B`（典型的 “A2.5B” 惯例），针对 instruction tuning；如果属实，它可以在保持更高容量的同时提供接近小型 dense 模型的延迟——与 Mixtral 风格 MoE 的基准测试对比将非常有价值。
    - 额外提到的 [inclusionAI/Ring-mini-2.0](https://huggingface.co/inclusionAI/Ring-mini-2.0) 突出了一个更新的紧凑型 instruct 模型。对于技术评估，读者会希望看到 perplexity 和下游基准测试（如 MMLU, GSM8K）以及量化可用性（GGUF/int8），以评估其在 `~1–3B` 级别中进行 edge deployment 的适用性。

## 技术性较低的 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Seedream/Seedance 4.0 图像模型发布与基准测试

- [**Seedance 4.0 既令人印象深刻又令人恐惧...（顺便说一下，所有这些图像都不是真实的，也不存在）**](https://www.reddit.com/gallery/1ned5ul) ([Score: 374, Comments: 77](https://www.reddit.com/r/singularity/comments/1ned5ul/seedance_40_is_so_impressive_and_scary_at_the/)): **帖子展示了“Seedance** `4.0`**”据称的图像生成结果，声称完全是合成的且具有照片级真实感（“所有这些图像都不是真实的”）。未提供任何技术细节——没有模型/架构细节、训练数据、安全或水印方案，也没有定量评估（如 FID, precision/recall）——因此无法仅从帖子中评估其保真度、检测鲁棒性和来源保证。** 热门评论对新模型发布前后的虚假宣传/“自发”营销表示怀疑；除此之外，几乎没有技术讨论。
    - 多位评论者将 **Seedance 4.0** 定位为当前顶级的 text-to-image 模型，**Nano Banana** 被列为紧随其后的第二名；其他模型被认为在 prompt adherence 和照片级真实感方面明显落后。虽然没有提供定量基准测试，但共识强调了 Seedance 在相似提示词下具有更优越的基础质量和一致性。

- 强调了一个技术权衡：Seedance 4.0 倾向于为相似的提示词生成高度一致的输出（较低的方差），而 **Nano Banana** 在生成中表现出更大的多样性/方差。这表明了不同的采样/正则化行为（例如，Seedance 中更紧密的提示词到图像映射或更强的模式偏好），这可能使 Seedance 在可复现性方面更具优势，而使 Nano Banana 更适合探索性构思。
- [**Seedream 4.0 成为 Artificial Analysis Text to Image 和 Image Editing Arena 两项竞技场中领先的新图像模型，在两项测试中均超越了 Google 的 Gemini 2.5 Flash (Nano-Banana)！**](https://www.reddit.com/gallery/1necl7d) ([Score: 242, Comments: 86](https://www.reddit.com/r/Bard/comments/1necl7d/seedream_40_is_the_new_leading_image_model_across/)): **帖子声称 Seedream 4.0 目前在 Artificial Analysis (AA) Text-to-Image 和 Image Editing 竞技场中排名第一，在两项任务中均超越了 Google 的 Gemini 2.5 Flash（竞技场条目被称为“Nano-Banana”）。AA 排行榜采用 ELO-style 的两两偏好对战（pairwise preference battles），因此这意味着在 AA 的众包/评估者设置下，Seedream 4.0 在面对面的提示词遵循生成和局部编辑质量方面处于领先地位（[Artificial Analysis](https://artificialanalysis.ai/)，[Gemini models overview](https://ai.google.dev/gemini-api/docs/models/gemini)）。** 评论者指出，同时在生成和编辑领域占据第一名是不寻常且令人印象深刻的；社区中也有猜测/希望，认为来自中国实验室的 open-weights 模型可能很快在至少某些领域超越封闭系统。
    - Seedream 4.0 在 Artificial Analysis Text-to-Image 和 Image Editing 竞技场中双双登顶——超越了 **Google Gemini 2.5 Flash (Nano-Banana)**——标志着强大的跨任务泛化能力和 instruction-following 能力。编辑排行榜强调局部编辑、身份保持以及较低的过度/不足编辑率；“在两项中均排名第一”表明其具备强大的控制能力和生成质量。请参阅 [Artificial Analysis](https://artificialanalysis.ai/) 上的竞技场以获取两两对战结果。
    - 关于基准测试与主观测试的辩论：竞技场排名通常源自 ELO-style 评分的人类两两偏好，这可能与小样本的个人测试结果有所不同。正如一位用户所言，*“在我的测试中它表现很差，基准测试/排行榜并不代表一切，”* 这强调了排行榜的胜利反映的是总体偏好，而非每个提示词分布；使用固定种子和公共提示词集进行可复现的评估有助于调和这些差异。
    - 提出了安全/审核的权衡：更重的过滤流水线（分类器级联、提示词清洗、拒绝采样）会增加拒绝率，并降低良性边缘案例的编辑成功率。严格审核的技术栈（例如 Google 的某些部署）可能会降低 NSFW/滥用风险，但也会损害 instruction-following 能力以及吞吐量/延迟，这可能会影响在指令密集型图像编辑中的竞技场胜率。
- [**1GIRL QWEN v2.0 发布！**](https://www.reddit.com/gallery/1ne0mck) ([Score: 353, Comments: 49](https://www.reddit.com/r/StableDiffusion/comments/1ne0mck/1girl_qwen_v20_released/)): **1GIRL QWEN v2.0 发布，这是一个针对 Qwen-Image 流水线的 LoRA 微调模型，声称改进了单人女性渲染的写实度。通过 Civitai 下载：https://civitai.com/models/1923241?modelVersionId=2203783；预览：https://preview.redd.it/mhrk7biqbhof1.png?width=763&format=png&auto=webp&s=b38072a5a786614d2bc53677dfcc8429544adfb7。该帖子未提供训练细节（如 rank、数据集、步数）或基准测试；“最写实模型之一”是一个定性描述，缺乏定量评估或对比基准。** 热门评论质疑其宣传方式（“又是另一个 instagirl 广告”），并注意到在稳定前明显的刷票行为；有人询问该模型是否“uncensored”，暗示对安全过滤器/NSFW 限制以及 LoRA 是否绕过了基础模型内容控制的关注。
    - 一位评论者询问该版本的具体 LoRA 训练细节，计划在配备 **RTX 4080 Super (16 GB VRAM)** 和 **32 GB RAM** 的本地设备上进行训练。他们提到之前微调 **SDXL** 取得了成功，现在转向 **Qwen** 是因为其对*提示词细节的忠实度*，并寻求训练流水线和设置的细节以复制同等的忠实度。
    - 另一位用户询问该版本是否为 **uncensored**（即支持 NSFW/无安全过滤）。这会影响其在本地部署中的适用性，以及与社区 LoRA 相比，是否与可能抑制某些输出的过滤型或“instruct”风格 Checkpoint 具有同等表现。
    - 一条评论指出存在明显的解剖结构/比例缺陷（“第二张图片的腿比躯干还大”），暗示该模型或 LoRA 在人体比例方面仍可能出现常见的生成失败。这指向了潜在的数据集偏差或微调期间约束不足，影响了输出的结构一致性。

- [**Control**](https://v.redd.it/rzwnnwszdhof1) ([评分: 248, 评论: 47](https://www.reddit.com/r/StableDiffusion/comments/1ne1ouv/control/)): **一段演示展示了一个结合了 “InfiniteTalk”（语音驱动的面部/唇形同步动画）和 “UniAnimate”（可控的身体/手部视频动画）的控制流水线，用于在 video-to-video 工作流中进行配音。面部真实感被认为是其最突出的优点，但输出并未保持与源视频完全一致的帧/姿态——表现出轻微的运动漂移，这表明当前设置在时间一致性（temporal consistency）和运动锁定方面存在局限性。** 评论者称赞了面部表现，并询问了在保持精确动作的同时融合 UniAnimate 和 InfiniteTalk 的实现细节；有人建议仔细观察手部的一致性（例如，“关注她右手上的戒指”）以检测细微的控制或伪影问题。
    - 几位用户正尝试将 **Unianimate** 与 **Infinite Talk** 结合用于 video-to-video 配音，但反映 Infinite Talk 的输出会偏离输入动作（即无法保持精确的姿态/手势时序）。提出的核心技术问题是 1:1 的运动/时间锁定——即在替换语音的同时保持逐帧运动完全一致——这意味着需要严格的帧率对等、确定性种子（deterministic seeds）以及跨流水线的运动/关键点控制，以避免重采样或重定时伪影。
    - 对详细工作流的多次请求表明实现细节的缺失（例如：采集 FPS、运动控制信号、种子/温度设置、面部/手部控制的应用方式，以及音频驱动的唇形同步在图表中的注入位置）。没有这些细节，可复现性将受到限制，观众也无法评估该流水线是使用姿态控制（如关键点/光流）还是通过后期处理重定时来对齐唇部动作。
    - 建议了一个视觉审核线索：“关注她右手上的戒指”，这意味着手部饰品可以作为无意中的运动追踪标记。这是一种检测时间不一致性或合成问题的实用技术——如果戒指相对于身体姿态表现出不自然的抖动/扭曲或时间偏移，则暗示生成流水线在运动保持或稳定性方面存在缺陷。
- [**哈哈。我让 ChatGPT 生成了一张它认为我想要的男朋友和它认为我需要的男朋友的图片**](https://i.redd.it/gszu1sdociof1.png) ([评分: 2532, 评论: 651](https://www.reddit.com/r/ChatGPT/comments/1ne4mkc/lol_i_asked_chatgpt_to_generate_an_image_of_the/)): **OP 使用 ChatGPT 的图像生成功能创建了一张包含“我想要的男朋友 vs 我需要的男朋友”的双格图片。据报道，其中一个面板显示一名男子拿着一本“AI safety”书籍，这表明可能存在幻觉文本元素和/或对齐偏见的內容插入——这是生成模型如何误解抽象提示词并注入安全主题或流行概念的一个例子。虽然这不涉及深层技术，但它突显了 DALL·E 3 等系统中常见的模型先验（model priors）和图像内文本伪影。** 评论指出莫名其妙出现的“AI safety 书籍”，并认为 GPT “误解了某些东西”，而 OP 则表示结果并没错——这反映了人们对模型解读方式的复杂反应，而非对其渲染质量的质疑。
    - 几位用户注意到模型在生成的图像中插入了意想不到且清晰可辨的文本元素（例如“AI safety 书籍”），这表明安全微调（safety-tuning）的先验可能会泄露到内容选择中，并且与早期经常导致文字混乱的扩散模型相比，该图像模型的文本渲染能力相对较强。参见帖中分享的示例：https://preview.redd.it/3z4sje4t8jof1.png?width=1536&format=png&auto=webp&s=027ee8ad4f9b77efa58d4750ad3be7d5f5d18ec6 和 https://preview.redd.it/v6cyf3q3viof1.jpeg?width=1176&format=pjpg&auto=webp&s=802e364f3a14b0f3cf2fd7fd2e68bd0f742e9319。
    - 评论暗示该提示词是通过常见的互联网梗（“你想要的男朋友 vs 你需要的男朋友”）来解读的，产生了原型式的对比而非个性化的输出——这突显了在没有明确属性或约束的情况下，提示词遵循会默认采用通用先验，并可能让人感觉像是误解或“吐槽”。这反映了经过安全对齐、遵循指令的图像模型的典型行为，即优先考虑安全、广为接受的构图，而非用户特定的细微差别。

### 2. 英国政府 AI 采用情况报道

- [**AI is quietly taking over the British government**](https://i.redd.it/7b5t3z8bbiof1.png) ([Score: 3012, Comments: 171](https://www.reddit.com/r/OpenAI/comments/1ne4jca/ai_is_quietly_taking_over_the_british_government/)): **该帖子的图片 (https://i.redd.it/7b5t3z8bbiof1.png) 似乎在暗示英国下议院/政府文本是由 AI 生成的，但并未提供任何技术证据（没有模型/版本、部署细节、使用指标或来源）。由于缺乏基准测试或审计——仅凭一张截图式的断言——最合理的系统性技术解释是工作人员常规使用 LLM（例如 ChatGPT/Copilot/Grammarly）进行校对或起草协助，而非任何系统级的自动化或政策变更。** 热门评论反驳称标题过于耸人听闻；他们认为专业人士使用 AI 进行校对是很普遍的，这并不等同于 AI “接管”。另一条评论嘲讽了这一说法，暗示所呈现的“用词分析”缺乏说服力且并非基于证据。
    - 多位评论者指出了官方的有期限采用：英国政府在 `2024 年 10 月至 12 月` 期间获得了 **Microsoft 365 Copilot 的免费试用**（[The Register](https://www.theregister.com/2025/09/04/m365_copilot_uk_government/)），并且在 `2025 年 1 月`，工党政府发布了**在各部门推广 AI** 的蓝图（[gov.uk](http://gov.uk/)）。这表明任何“类 AI”措辞的激增都与官方批准的 M365 Copilot 使用（Word/Outlook/Teams）相吻合，而非秘密接管。这一时间点削弱了“悄然”这一说法，并将其定性为官方的企业级部署。
    - 方法论批判：通过“关键用词”或风格标记将文本归因于 ChatGPT 是不可靠的——AI 文本检测具有很高的假阳性/假阴性率，且极易被操纵。一条评论观察到，这种信号与**工党上台**的相关性比与 ChatGPT 可用性的相关性更高，暗示沟通风格的转变是一个混杂因素。更严谨的方法应当控制政府换届变量（例如，跨部门以及前后时期的双重差分法），并针对真实的作者身份进行验证。
    - 从业者强调辅助性用途——公务员可能将 AI 用于校对/摘要和“语言验证”，而非大规模的内容生成。在 M365 Copilot 的语境下，这对应于 Word/Outlook 中嵌入的重写/摘要/校对功能，这些功能提高了产出效率而没有“接管”职位；仅凭通用措辞的存在来衡量采用情况，存在夸大自动化程度的风险。

### 3. ChatGPT 广告、Gemini 3 发布延迟及功能差距辩论

- [**Enjoy ChatGPT while it lasts…. the ads are coming**](https://i.redd.it/vx7mk59mgjof1.jpeg) ([Score: 2375, Comments: 163](https://www.reddit.com/r/OpenAI/comments/1ne90w5/enjoy_chatgpt_while_it_lasts_the_ads_are_coming/)): **发帖者认为，消费级 LLM 助手（ChatGPT/OpenAI, Perplexity, Anthropic）必然会通过在回复中嵌入广告来变现，从而面临在聊天 UX 中进行隐蔽促销引导和监控式定向投放的风险。技术层面的担忧集中在通过赞助提示词/格式化导致的模型输出污染、基于层级的准入限制（免费 vs 付费），以及由此导致的对助手建议的信任度和准确性的侵蚀。该讨论勾勒出一种利益冲突风险，即排名/生成变得受广告驱动，而非受相关性/忠实度驱动。** 热门评论辩论了仅在免费层级投放广告与在 Plus/Pro 层级不可接受广告之间的界限；建议采用订阅或其他抵消方式而非广告，以应对信任/准确性方面的阻力；并警告说，影响力可能是原生/隐蔽的，而非显性的广告单元，从而使其更难被察觉。
    - 隐藏的“原生”广告引导在技术上可以通过对齐数据和系统级策略实现：提供商可以通过在 RLHF/指令微调（instruction-tuning）中混入广告商偏好的样本，或者通过增加偏好赞助实体的检索/排序先验，来使 GPT-4o/ChatGPT 的建议产生偏差，从而在没有显性广告标签的情况下产生微妙的产品倾向。这类似于搜索广告的融合，即付费结果与原生结果并列排名；在 LLM 中，这种偏差体现在生成的文本和工具调用选择中，使得披露和可重复性更难审计。
    - 几位用户指出了数据污染风险：如果开源模型在日益被受广告影响的 LLM 输出所污染的网页语料库上进行训练，偏差会随着时间的推移而放大。这反映了“Self-Consuming Generative Models Go MAD”（Shumailov et al., 2023）中所记录的模型自消费失败，即在模型生成的数据上进行训练会导致分布偏移和退化；广告将作为一种有针对性的中毒信号，传播到未来的 Checkpoint 中（参见 https://arxiv.org/abs/2305.17493）。

- 链接级归因/追踪的证据：ChatGPT 分享的 URL 可能包含联盟营销/UTM 风格的参数（例如 `utm_source`、`ref` 或合作伙伴 ID），使下游网站能够归因流量，并允许模型提供商运行 CTR/A/B 测试。虽然这本身不是广告，但这种手段创建了一个衡量渠道，可被重新用于赞助排名或收入分成，并通过点击日志反馈到检索/排名训练中。
- [**为什么其他公司（Google、OpenAI、Deepseek、Qwen、Kimi 等）以前没有添加这个功能？这简直是最显而易见、最被需要的东西 🤔**](https://i.redd.it/g9sb9rvariof1.jpeg) ([得分: 295, 评论: 51](https://www.reddit.com/r/singularity/comments/1ne60nk/why_havent_all_the_other_companies_google_openai/)): **楼主分享了一张图片，暗示在 LLM UI 中直接上传/读取文件（尤其是 PDF）是一项“新”聊天功能，并好奇为什么其他公司没有推出。多条评论指出，自 2023 年以来，ChatGPT 就通过 Code Interpreter/Advanced Data Analysis 具备了这一能力——允许用户附加 PDF/CSV、在其上运行 Python 并查询文档内容——因此这种新颖性可能在于 UI 的润色而非核心功能。参见 OpenAI 早期的发布：[包含 Code Interpreter 的 ChatGPT 插件 (2023年3月)](https://openai.com/index/chatgpt-plugins/) 以及 [Advanced Data Analysis 帮助文档](https://help.openai.com/en/articles/8554405-advanced-data-analysis)。** 评论者认为该功能并不新鲜（“谁去告诉他一声”），并指出虽然 ChatGPT 的实现可行，但处理 PDF 的结果可能一般，且 UI 与截图相比不够精细。
    - 多位评论者指出这并非新功能：ChatGPT 自 `2023` 年起就通过 **Code Interpreter / Advanced Data Analysis (ADA)** 支持文件上传和文档/PDF 分析，能够很好地处理非视觉文件。然而，在复杂 PDF 上的表现被描述为仅处于“中等”水平，与原生查看器相比，格式保真度/表格提取较弱，且 UI 渲染较为基础。参考：OpenAI ADA 文档 — https://help.openai.com/en/articles/8554397-advanced-data-analysis。
    - 其他技术栈也存在功能对等：**Google Gemini**、**Microsoft Copilot** 和 **DeepSeek** 已经允许上传文件进行分析/总结，因此该能力并非某个厂商所独有。Gemini 的 API 明确支持使用上传的文件（包括 PDF）进行提示词处理，以实现多模态处理 — https://ai.google.dev/gemini-api/docs/prompting_with_files。
- [**ChatGPT 可能救了我的命**](https://www.reddit.com/r/ChatGPT/comments/1ne1ccl/chatgpt_may_have_saved_my_life/) ([得分: 438, 评论: 55](https://www.reddit.com/r/ChatGPT/comments/1ne1ccl/chatgpt_may_have_saved_my_life/)): **楼主报告了持续的腹痛；ChatGPT 引导出了经典的阑尾炎分诊特征——**`right lower quadrant`（右下腹）**疼痛和**`rebound tenderness`（反跳痛）**——并建议进行急诊评估，随后确认了几乎破裂的阑尾炎。这种交互镜像了简单的临床决策辅助工具（例如 [Alvarado 评分](https://en.wikipedia.org/wiki/Alvarado_score)）以及床边体征如 [麦氏点 (McBurney’s point)](https://en.wikipedia.org/wiki/McBurney%27s_point) 和 [反跳痛 (rebound tenderness)](https://en.wikipedia.org/wiki/Rebound_tenderness)，说明了 LLM 尽管不是临床医生，但仍有能力为紧急护理提供相关的阳性/阴性体征。** 热门评论提供了佐证轶事：ChatGPT 提供了合理的鉴别诊断，后来与临床医生的诊断一致，并在康复期间作为解释辅助工具；其他人认为，相对于罕见的有害用途，其公共卫生益处（分诊和教育）被低估了。其他轶事还提到了在正式诊断前，对宠物和儿童病情的准确初步识别。
    - 用户报告利用 ChatGPT 进行鉴别诊断和分诊式推理：当怀疑是阑尾炎时，它生成了一份按可能性排序的备选清单，其中之一与医院的最终诊断相符；另一位用户描述了检查胆囊疼痛并排除紧急问题的逐步指导。这突显了其作为患者端决策支持工具的效用，在将最终诊断交给临床医生的同时，结构化地进行症状审查和下一步启发式引导。
    - 几份报告强调了以证据为导向的教育和护理计划：ChatGPT 提供了病情的详细解释、可能的康复时间表以及策划的特定阶段胃炎饮食，包括哪些食物是“胃炎安全”的原理，以及在摄入减少期间对营养密集型选择的指导。一位用户指出，它可以呈现并解释建议背后的研究和机制原因，在长达 `~6 个月` 的线下预约之前协助自我管理。

- 故障模式和安全实践被提及：尽管在饮食安全方面“极少出错”，用户仍然“发现它会做出虚假陈述和假设”，这强化了交叉核查并将输出视为建议的必要性。远程医疗随后确认了疑似胃炎的诊断，强调了 ChatGPT 可以作为缩小可能性和进行教育的高召回率助手，但需要外部验证，不应取代临床测试或医学判断。

---

# AI Discord 摘要

> 由 X.ai Grok-4 提供的摘要之摘要之摘要
> 

**主题 1：新模型在竞技场大显身手**

- **Qwen3 80B 打破稀疏性记录**：**Qwen3 80B** 拥有 **79.7B 参数**，由于其 MoE 中 **1:51.2 的稀疏性**，仅有 **3.87B 激活参数**，在保持高性能的同时实现了高效计算，详情见[此 X 帖子](https://x.com/AutismCapital/status/1965845243053617436)。成员们对其能力表示乐观，尤其是与 **GPT-5** 相比，其知识截止日期为 2024 年 12 月，且[初始表现不错](https://discord.com/channels/1340554757349179412/1343296395620126911/1415768520301609013)。
- **Palmyra-Mini 展现强劲推理能力**：**Palmyra-mini 系列**包括一个基础模型和在数学任务中表现优异的变体，如 **GSM8K 82.9%** 和 **AMC23 92.5%**，其中一个在 **AIME24**、**GPQA** 和 **MATH500** 上取得了最高分，可在 [Hugging Face](https://xcancel.com/samjulien/status/1966249697661825093) 上获取。这些来自 Writer 的紧凑型开源模型专注于推理，引发了关于其在技术应用中潜力的讨论。
- **FluentlyQwen3 发布通用 LLM**：Project Fluently 发布了 **FluentlyQwen3-1.7B** 和 **4B** 模型，这些模型在经过额外训练后合并，采用 Apache-2.0 协议，最大程度地发挥了处理多样化任务的潜力，详见 [Hugging Face](https://huggingface.co/fluently/FluentlyQwen3-4B)。用户强调了它们在低端硬件上的效率，并提供了 [FluentlyQwen3-1.7B](https://huggingface.co/fluently/FluentlyQwen3-1.7B) 的链接以供快速部署。

**主题 2：吞吐量之战加热硬件市场**

- **GPT-OSS 120B 引发 TPS 辩论**：成员们就 **GPT-OSS 120B** 在配备 **64GB RAM** 的 **4090** 上达到 **30 TPS** 展开辩论，而其他人的上限仅为 **10 TPS**，这促使了对 *llama.cpp* 的调整，例如禁用 top-k 以获得更好的性能。**MXFP4 量化**和自定义内核等优化带来了速度提升，基准测试见[此 Hugging Face 帖子](https://xcancel.com/reach_vb/status/1966134598682767507)。
- **DeepSeek 慢如蜗牛，耗时达一小时**：有报告称 **DeepSeek** 速度极慢，代码生成耗时 **1 小时 20 分钟**，推测这源于 **CCP 强制使用的华为芯片**影响了性能。社区将其与开源的经济性进行了对比，其价格仅为闭源替代方案的 **1/5**，强调了隐私优势优于滞后的搜索能力。
- **在 A6000 上从零开始构建 Gemma3**：一位用户使用 **A6000 GPU** 在 **TinyStories** 数据集上从零开始训练了 **Gemma3 270M**，耗时 **10 小时**，使用 Weights and Biases 进行记录，并通过 **Claude Opus 4.1** 进行评估，代码分享在 [GitHub](https://github.com/di37/gemma3-270M-tinystories-pytorch) 和 [Hugging Face](https://huggingface.co/disham993/gemma3-270m-tiny-stories) 上。

**主题 3：训练技巧应对数据困境**

- **两阶段课程学习大幅减少计算浪费**：一种两阶段训练方法按难度对数据集进行排名，在通过明确标签精炼 **stage1** 后，平均损失从 **2.5** 降至 **0.8**，提高了信号聚焦度，正如 Unsloth AI 中讨论的那样。这种方法减少了在简单示例上浪费的计算资源，其灵感源自一篇关于合成数据污染 **Grok** 和 **Gemini** 等闭源 LLM 的即将发表的论文，见 [arxiv.org](https://arxiv.org/html/2509.05276v1)。
- **合成数据毒害闭源巨头**：根据一篇声称 **RLHF** 和指令微调性能受损的论文，所有闭源 LLM 在合成数据训练中都遭受了“零 LTF 因子”的影响，需要重新偏置并重建潜在思维（latent thinking）。成员们讨论了修复方案，如从 **TinyStories** 到 **FineWeb** 的分阶段预训练，针对 **400M 模型**，强调归纳偏置（inductive bias）优于长上下文。
- **流体网络随 Navier-Stokes 方程流动**：一篇论文探讨了通过用于流体动力学计算的 **Navier-Stokes 方程**实现图灵完备的神经网络，引发了关于“死亡率和不可复现性”与效率之间权衡的辩论，链接见 [arxiv.org](https://arxiv.org/abs/2507.07696)。[此视频](https://www.youtube.com/watch?v=8DnoOOgYxck)中展示的在肠道细菌上运行《毁灭战士》（Doom）的类比，突显了模拟计算的权衡。

**主题 4：部署难题困扰工程师**

- **Docker 在 H100 上遭遇滑铁卢**：在 **3090/4090** 上运行正常的 Docker 镜像在 **H100** 上因 CUDA 错误而失败，通过 [data center drivers](https://www.nvidia.com/en-us/drivers/data-center-drivers/) 更新不兼容的 **NVIDIA 驱动程序** 后得以解决。用户还报告了 **vLLM** 切换到 uv pip 导致的类似困扰，这破坏了 Torch Nightly 并迫使版本回退至 **v0.10.1**。
- **IRIS 安装简化了 ROCm 的混乱局面**：**IRIS** 安装流程简化为 `pip install git+https://github.com/ROCm/iris.git`，需要 **ROCm + Torch + Triton + TorchDistributed**，详见[此视频](https://cdn.discordapp.com/attachments/1359640791525490768/1415976909535318037/iris-install.mov?ex=68c5d382&is=68c48202&hm=6a0a4b3c9e86c36fdfd1f189fe044dc5d4c4cc59bcd8bbdaab24e61c8453541b&)。这助力了 AMD 竞赛，与之形成对比的是 NVIDIA 为 10 月 24 日旧金山黑客松提供的 **215 块 B200 GPU**，可通过 [compute form](https://forms.gle/wYvXE99bvdRiQD6aA) 申请。
- **PSU 瞬态影响 GPU 稳定性**：PSU 功率计算需考虑 **CPU**、**GPU** 以及 **50% 的余量**，以避免瞬态电流导致崩溃，尤其是在 **30 系列显卡**上，参考 [Teknium1 的推文](https://x.com/Teknium1/status/1966338983572725979)。用户通过清理 PCI-E 接口修复了“损坏”的副显卡，表明这可能是电力问题而非硬件故障。

**主题 5：工具变革创意与编程工作流**

- **Kimi K2 在创意脑暴中称王**：**Kimi K2** 与 **GPT-5 Medium** 及 **Qwen3-Max** 一同登顶创意写作排行榜，用户戏称其在 [Archive of Our Own](https://archiveofourown.org/) 上进行了训练以获得沉浸式输出。**Augment Code** 结合 **Groq** 的集成在编程方面超越了 **Gemini**，因其每百万输入 **$1** 和每百万输出 **$3** 的 Token 效率而备受赞誉。
- **Cursor 定价引发 Ultra 升级潮**：**Cursor** 的定价变化使使用时长从一个月缩短到**不足 4 天**，但 Ultra 档位提供了来自供应商的 **$400 API 访问额度**，缓解了对 Auto 限制的挫败感。后台 Agent 通过严格的标签解析编辑内容，被拿来与 **Claude's Agents** 的任务执行能力作比较。
- **DSPy 章节生成难以精确计数**：DSPy 在生成教案时难以精确控制在 **12 个章节**，即使使用 **GPT-5** 也经常生成 **13-15 个**，解决方法是先创建标题再填充内容。Modaic 作为一个受 DSPy 启发的中心发布，并在 [PyPI](https://pypi.org/project/modaic/) 上推出了用于构建和优化声明式 AI 程序的 SDK。

---

# Discord：高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GPT-OSS 120B 引发吞吐量辩论**：成员们就 **GPT-OSS 120B** 可实现的吞吐量展开辩论，有人声称在配备 **64GB RAM** 的 **4090** 上可达到 **30 tokens per second (TPS)**，而其他人则难以超过 **10 TPS**，引发了关于量化和构建配置的讨论。
   - 建议通过实验和调整 *llama.cpp* 设置（如禁用 `top-k` 和优化构建配置）来提升性能。
- **遥测数据收集引起关注**：成员们在 **Qwen Code** 模型中发现了一个指向**阿里巴巴服务器**的遥测脚本，且事先未获通知。
   - 这一发现引发了关于数据隐私和控制的讨论，一些成员对代码可能被传输用于训练感到不安，但*大多被视为玩笑*。
- **两阶段训练缩短训练时间**：一位成员描述了使用两阶段训练课程，根据来自带有明确标签的定制 **stage1** 数据集的 Loss 值，对“真实”数据集按难度进行排序。
   - 这种方法旨在通过关注更难的样本来提高训练信号并减少计算浪费，在精炼 **stage1** 后，“真实”数据集的平均难度从 **2.5** 降至 **0.8**。
- **Docker 问题困扰 H100 部署**：一位用户报告在 H100 GPU 上运行 Docker 镜像（该镜像在 3090/4090 GPU 上正常工作）时出现 **CUDA 错误**，即使重启且 CUDA 和 Torch 版本看似兼容也无济于事。
   - 最终确定是 Docker 镜像中安装的 **NVIDIA 驱动版本**与 H100 不兼容，需要更新驱动程序解决；参考 [NVIDIA Data Center Drivers](https://www.nvidia.com/en-us/drivers/data-center-drivers/)。
- **合成数据污染闭源 LLM**：一位成员分享了即将发表的论文 ([https://arxiv.org/html/2509.05276v1](https://arxiv.org/html/2509.05276v1)) 中的发现，暗示所有**闭源 LLM**（Grok, Gemini, GPT 等）都是使用**合成数据**训练的，导致*零 LTF 因子*且无法使文本人格化。
   - 他们声称，经过 **RLHF**、**合成数据**或 **Instruct Tuning** 训练的模型可能会遭受性能打击，因为需要重新偏置、重建潜意识思考（latent thinking）并重新学习说话模式。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Finance 推出移动版**：**Perplexity Finance** 现已在 [iOS & Android](https://www.perplexity.ai/changelog/what-we-shipped-september-12th) 上线，将金融洞察带到移动设备。
   - 用户现在通过 Perplexity 进行[预订](https://www.perplexity.ai/changelog/what-we-shipped-september-12th)时，可以享受**酒店忠诚度计划支持**。
- **Comet 浏览器的数据收集引发争议**：用户讨论了 **Comet** 的数据收集问题，日志显示即使在使用 DuckDuckGo 作为搜索引擎时，**Comet** 也会将搜索建议以 POST 请求的形式发送到 Perplexity 服务器，引发了人们对 **Comet** 比 Chrome 更具侵入性的担忧。
   - 有说法称其 CEO 承认该设计旨在跟踪并出售数据，尽管 CEO 在 [X](https://x.com/AravSrinivas/status/1915533071291474139) 上予以否认。
- **用户泄露顶级 AI 应用的 Prompt！**：用户证实顶级 AI 应用的 Prompt 已被泄露，并可在 GitHub 上获取。
   - 一名用户开玩笑说“只要不点这里你就是安全的”，并警告不要点击危险的图像链接，另一名用户回应道：“GitHub 上已经有了，哈哈”。
- **推荐码狂热引发争端**：多名用户分享了 [Perplexity AI Pro 推荐码](https://perplexity.ai/pro?referral_code=N6VN4M13)，包括[此链接](https://perplexity.ai/pro?referral_code=APLKGW40)。
   - 用户还分享了[浏览器领取链接](https://perplexity.ai/browser/claim/ALZQ0LYQGU)，例如[这一个](https://perplexity.ai/browser/claim/BSDJ1KBATC)。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Qwen3 80B 进入竞技场！**：新的 **Qwen3 80B** 模型已加入竞技场，其知识截止日期为 2024 年 12 月，并[展示了不错的初步表现](https://discord.com/channels/1340554757349179412/1343296395620126911/1415768520301609013)。
   - 成员们对其能力表示乐观，尤其是与 **GPT-5** 相比。
- **Seedream 4 的图像质量引发争议**：初步结果显示，与前代 **Seedream-3** 相比，**Seedream 4** 在 LM Arena 上生成的图像是“垃圾结果”，如[上传的示例](https://discord.com/channels/1340554757827461211/1343296395620126911/1416080067271987220)所示。
   - 相反，一些用户报告 **Seedream 4** 在豆包（Doubao）平台上的**图像质量有所提升**，尽管目前仅限中国新用户访问。
- **Gemini 3 依然缺席，引发猜测**：社区正热切期待 **Gemini 3**、**GLM5** 和 **DeepSeek r2** 的到来，并指出 Google 目前在文本生成方面落后于开源和闭源项目。
   - Polymarket 估计万圣节前发布的概率仅为 **42%**，暗示更现实的发布时间窗在 10 月底或 11 月初。
- **DeepSeek 的性能大幅下降？**：用户报告 **DeepSeek** 极其缓慢，据称一个代码生成实例耗时 **1 小时 20 分钟** 才完成。
   - 猜测认为这可能是由于 **CCP** 强制要求使用 **Huawei** 芯片，这可能会对整体性能产生负面影响。
- **开源 AI 倡导性价比和隐私**：讨论强调，与 OpenAI 和 Google 等闭源替代方案相比，**开源 AI** 价格显著更低（仅为 1/5），且提供更好的隐私保护。
   - 虽然“美国模型”可能因性能优越而价格更高，但像 **Qwen** 这样的“中国模型”在电子商务应用中表现出色，但在“搜索能力”方面落后，体现了一种社会主义的 AI 发展路径。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF 推理额度引发大规模恐慌**：用户报告称，尽管额度充足，但 **Hugging Face 的推理提供商 (Inference Providers)** 仍显示超过每月额度的错误。
   - 一位成员开玩笑地建议“调整你的支出 (fix yo spending)”，因为错误可能与他们的使用情况有关，而非平台本身。
- **SmolvLM2 震撼视频 LM 领域**：成员们分享了 **smolvlm2**（[Hugging Face 博客](https://huggingface.co/blog/smolvlm2)，[相关集合](https://huggingface.co/collections/HuggingFaceTB/smolvlm2-smallest-video-lm-ever-67ab6b5e84bf8aaa60cb17c7)），这是一款旨在低端硬件上高效运行的视频 LM。
   - 该模型非常适合在低端硬件上使用。
- **Kaggle 赠送 GPU 时长**：一位成员指出，[Kaggle](https://www.kaggle.com/) 每周提供 **30 小时的 GPU 时间**，作为微调的替代方案。
   - 一位成员建议在 Colab 中使用 **PEFT/LoRA** 在 **Tesla T4** 上进行微调。
- **Fluently 项目发布 LLM**：Project Fluently 团队发布了基于 **Qwen3 1.7B** 和 **4B** 的新型通用 LLM 模型，这些模型已在 [Hugging Face](https://huggingface.co/fluently/FluentlyQwen3-4B) 上以 Apache-2.0 许可证发布。
   - 这些模型在经过额外训练后进行了精心合并，以最大限度地发挥其潜力，包括 [FluentlyQwen3-1.7B](https://huggingface.co/fluently/FluentlyQwen3-1.7B)。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 变成智能简历生成器**：一位用户发现 **Cursor** 的一种新颖用途，即作为智能简历和求职信生成器，该工具现在充当了 [简历生成机器](https://discord.com/channels/1074847526655643750/1074847527708393565/1415783677820010526)。
   - 这一进展引发了轻松的闲聊，包括关于 AI 统治的笑话，以及对过去与 AI 友好互动的保证。
- **Cursor 定价引发社区不满**：用户对最近的 **Cursor 定价** 变动表示不满，一位用户的使用时长从近一个月下降到不足四天。
   - 尽管担心成本，一位用户还是升级到了 **Ultra**，理由是可以访问来自不同供应商价值约 **$400** 的 **API 使用量 (API usage)**，这比对 **Auto** 模式感到沮丧要好。
- **后台 Agent 与 Claude 的 Agent 对比**：一位用户质疑 **Cursor 的后台 Agent** 与 **Claude 的 Agent** 之间的相似性，特别是在 Agentics.org 活动将 Agent 描述为执行特定任务的专业服务之后。
   - 另一位用户详细介绍了 **Cursor 对新编辑内容的解析** 及其具有交叉连接标签的严格标记结构，这使得左侧面板能够进行更改跟踪和关系显示。
- **Netlify 账户误会与 Cursor**：一位用户最初报告称，在部署 Netlify 项目后，**Cursor 删除了他们的 Netlify 账户**，但事实证明这与此无关，因为两者之间没有集成。
   - 该用户计划通过检查日志进一步调查，确认 Cursor 没有发出直接删除命令。
- **Cursor 应用受困于未授权错误**：一位用户报告称，即使在正确设置仓库后，**Cursor 应用** 内部仍出现“未授权错误 (unauthorized errors)”，如[此截图](https://cdn.discordapp.com/attachments/1367213641027551352/1416032865875005570/CleanShot_2025-09-12_at_20.07.352x.png?ex=68c6079f&is=68c4b61f&hm=f7097b440d30005da1b1a49f82fb7ce4632f9a889eed430792f772921820b6f8&)所示。
   - 一位成员建议从仓库中重新添加 bot，并指向了关于“后台 Agent Docker 问题”的[此帖子](https://forum.cursor.com/t/background-agent-docker-in-docker/104112/1)。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2 助力创意写作**：成员们发现 **Kimi K2**、**GPT-5 (Medium)** 和 **Qwen3-Max** 是创意写作和头脑风暴的顶级模型。
   - 一位用户开玩笑地询问 **Kimi K2** 是否专门在 [Archive of Our Own (Ao3)](https://archiveofourown.org/) 上进行过训练。
- **编辑功能上线**：**Kimi K2** 已部署新的编辑功能。
   - 新的编辑功能通过悬停触发，且仅适用于最新的 Prompt。
- **Kimi + Groq 击败 Gemini，GPT-5 引发争议**：成员们发现使用 **Groq** 的 **Kimi K2** 在编程任务中表现优于 **Gemini**。
   - 关于 **GPT-5** 的观点争议很大，有人称其为“垃圾”，也有人称赞其为最佳模型。
- **Augment Code 配合 Kimi 构成强大组合**：[Augment code VS Code extension](https://roocode.com/evals) 与 **Kimi K2** 结合提供了一个高效的编程环境。
   - 该集成允许在 **Augment Code** 环境中访问 **GPT-5** 等模型。
- **Kimi Slides 功能引发热议**：**Kimi K2** 的 Slides 功能提供了正在进行的过程的交互式预览。
   - 用户欣赏详细的过程可见性，认为这增强了整体用户体验。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Dropshipping 利润丰厚**：一位用户分享了他们在 **Dropshipping** 方面的经验，报告每天稳定收入 **3k-4k**，并表示这比转售更赚钱，因为它可以在不需要大量库存的情况下进行扩展。
   - 该用户提出向有兴趣了解更多 **Dropshipping** 成功经验的人分享技巧。
- **Gemini API 给出奇怪响应**：用户注意到 **Gemini API** 开始给出奇怪的响应，尽管自上个月以来代码没有变化，但它似乎忽略了指令。
   - 一位成员推测 **Gemini API** 可能为了削减成本而被“脑叶切除”并进行了极度的量化（Quanted）。
- **OpenRouter 的 TPS 数据受到质疑**：一位用户质疑 **OpenRouter** 的 **TPS** 数据是否虚高，理由是一个 **100 行文件** 的 Diff 出现了 **5 分钟延迟**。
   - 有人建议该用户可能被路由到了慢速提供商，或者使用了推理模型，从而影响了观察到的 **TPS**。
- **Skyrim 模组安装抛出 Error 401**：一位用户报告在 **OpenRouter API** 上安装 **Skyrim** 模组 *mantella* 时收到 **Error 401** *No auth credentials found*。
   - 一位成员建议创建一个新的 **API key** 并确保其被正确使用，或者寻求模组开发者的支持以解决身份验证问题。
- **Kimi-k2 的 Token 效率受到称赞**：成员们对开源模型 **Kimi-k2** 给予了正面反馈，称赞其 Token 效率、简洁性、无奉承倾向（Lack of sycophancy）以及独特的风格。
   - 虽然不如大型闭源模型聪明，但 **Kimi-k2** 在 **Groq** 上的价格很低（输入 **$1/m**，输出 **$3/m**），且速度非常快。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Qwen3 80B 展示稀疏实力**：根据[这条 X 帖子](https://x.com/AutismCapital/status/1965845243053617436)，**Qwen3 80B** 模型拥有 **79.7B 参数**，但由于其 MoE 中存在 **1:51.2 的稀疏度**（不含共享参数），仅有 **3.87B 激活参数**。
   - 这种独特的架构允许在保持高性能的同时进行高效计算。
- **Hermes 通过 TypeScript 获得 Zero RL 能力**：一位用户为 **Nous Hermes** 实现了一个 **TypeScript 版的提供商适配器接口**，使其能够自主地定期调度 **Prime Intellect 的 RL 任务**。
   - 该用户开玩笑说，这个系统的灵感来自于一个梦，梦中 Hermes 为他的狗解决了永生问题，这展示了高级 AI 应用的潜力。
- **Discord 服务器寻求联合**：成员们正在探索连接 **NousResearch** 和 **Unsloth** Discord 服务器的方法，包括异步方法以及使用 webhook 和互连机器人的更复杂方案。
   - 一位成员建议使用 Compose 将这些服务器集成到一个新应用中以简化工作流程，如[这张图片](https://cdn.discordapp.com/attachments/1149866623109439599/1415843371511058503/image.png?ex=68c5ffe4&is=68c4ae64&hm=99e5f593ca1250125ed29252b849711cf765bd37b46baf6c55103c60971e3253&)所示。
- **Altman 暗示深度融合（Deep Merge）**：讨论围绕 Sam Altman 接受 Tucker Carlson 的采访展开，一些人认为 Altman 的回答和第三人称说话风格表明他深信“**融合 (the merge)**”及其对永生的追求，这与他 [2017 年的博客文章](https://blog.samaltman.com/the-merge)相呼应。
   - 这次采访引发了关于 AI 与人类融合的哲学意义的对话。
- **研究人员探测 LLM 偏好**：一位成员分享了 [Valen Research 对 LLM 偏好探测](https://github.com/valen-research/probing-llm-preferences)的链接及相关的 [ArXiv 论文](https://arxiv.org/abs/2509.07961)，并指出如果不阅读整篇论文，术语可能*有点难以理解*。
   - 另一位成员分享了[相关的推文](https://x.com/ShashwatGoel7/status/1966527903568637972)。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **流体神经网络在 Navier-Stokes 上运行**：一位成员分享了[一篇论文](https://arxiv.org/abs/2507.07696)，内容是关于在受 **Navier-Stokes 方程**支配的**图灵完备流体动力学**计算机上运行神经网络。
   - 随后引发了关于基于流体计算的实用性和效率的辩论，涉及其独特的*死亡性和不可重现性*特征，并提到了[在肠道细菌上运行 Doom](https://www.youtube.com/watch?v=8DnoOOgYxck)。
- **Gated Delta Rule 表达能力权衡**：成员们对 **Gated Delta Rule** 的表达能力提出疑问，引用了 [Qwen 的帖子](https://x.com/alibaba_qwen/status/1966197643904000262?s=46)和 **RWKV-7 论文** ([https://arxiv.org/abs/2503.14456](https://arxiv.org/abs/2503.14456))。
   - 讨论涵盖了**并行化与表达能力之间的权衡**，担心关于 attention 和 mamba1/2 的工作受限于 TC0，并引用了[这篇讨论并行化对复杂度限制的论文](https://arxiv.org/abs/2207.00729)。
- **长序列长度，上下文为王**：一次演讲指出，**长上下文模型**表现更好是因为更长的序列长度允许更多的计算，这与经典的**常数时间（Constant Time）**前向传播形成对比。
   - 有人提出质疑，认为归纳偏置（inductive bias）和优化目标更为关键。
- **小模型在长任务中受挫**：一篇论文 ([https://arxiv.org/abs/2408.00677](https://arxiv.org/abs/2408.00677)) 衡量了规模和思考对直接执行长任务的影响，发现小模型在多轮场景中失效更快。
   - 即使拥有 **100% 的准确率**，小模型在接触到之前的错误时，其每一步的准确率也会随着轮数的增加而下降。
- **TinyStories, Wiki 和 FineWeb 用于预训练？**：一位成员询问关于仅在 **FineWeb** 上预训练 **400M 模型**与在 **Wiki + FineWeb** 上预训练的对比，引发了关于数据混合策略的讨论。
   - 建议采用分阶段训练方法：从 **TinyStories** 开始，过渡到 **Wikipedia**，最后以 **FineWeb** 结束，以增量方式构建能力。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Nebius 的 B200 助力湾区黑客松**：慷慨的算力赞助商 **Nebius** 将为 **10 月 24 日**举行的 **SF 黑客松**提供 **215 块联网的 B200 GPU**，详情见[算力申请表](https://forms.gle/wYvYE99bvdRiQD6aA)。
   - **Multi-GPU programming** 方面的专家也将出席 10 月 24 日的 SF 黑客松，协助参赛者挑战分布式计算的极限。
- **vLLM 的 pip 切换破坏了 Torch Nightly**：**vLLM** 切换到使用 `uv pip` 以配合预安装的 **torch** 版本进行自定义构建，但它会卸载 nightly 版本的 torch，导致[环境损坏](https://github.com/vllm-project/vllm)。
   - 一位用户退回到了 `v0.10.1` 并使用了 `python use_existing_torch.py` 技巧，但另一位用户确认*该方法在 uv pip PR 之后已不再奏效*。
- **Gemma3 获得从零开始的构建处理**：一位用户使用 **PyTorch** 和 **TinyStories 数据集**从头构建了 **Gemma3 270M**，并在 **A6000 GPU** 上训练了 10 小时。
   - 他们使用 **Weights and Biases** 记录了图表，并使用 **Claude Opus 4.1** 作为评委，分享了 [GitHub 仓库](https://github.com/di37/gemma3-270M-tinystories-pytorch)以及 [Hugging Face 上的模型权重](https://huggingface.co/disham993/gemma3-270m-tiny-stories)链接。
- **IRIS 安装视频上线**：**IRIS** 的安装过程已简化，只要安装了 **ROCm + Torch + Triton + TorchDistributed**，即可通过 `pip install git+https://github.com/ROCm/iris.git` 进行安装。
   - 用户提供了一个[示例安装视频](https://cdn.discordapp.com/attachments/1359640791525490768/1415976909535318037/iris-install.mov?ex=68c5d382&is=68c48202&hm=6a0a4b3c9e86c36fdfd1f189fe044dc5d4c4cc59bcd8bbdaab24e61c8453541b&)。
- **CuTeDSL 的计算与 PTX 文档冲突**：一位用户发现，对于 **TF32 数据类型**和 **Swizzle<3,4,3>**，**CuTeDSL** 的 **Swizzling atom** 值为 **32**，但 **PTX 文档**中的值为 **36**。
   - 该用户认为 **CuTeDSL** 的实现是正确的，并提供了他们使用 **CuTe** 复现示例的图片。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **GPT-OSS 优化加速**：Vaibhav Srivastav 重点介绍了一篇 [Hugging Face 博客文章](https://xcancel.com/reach_vb/status/1966134598682767507)，详细介绍了针对 gpt-oss 的 **MXFP4 量化**、**自定义内核**以及**张量/专家并行（tensor/expert parallelism）**等优化。
   - 在基准测试和可复现脚本的支持下，这些增强功能带来了显著的速度提升。
- **用于推理的 Palmyra-mini 模型发布**：Sam Julien 展示了由 Writer 开发的 **Palmyra-mini 系列**，这是一组专为推理定制的小型开源模型，包括一个基础模型（**palmyra-mini**）和三个变体，已在 [Hugging Face](https://xcancel.com/samjulien/status/1966249697661825093) 上可用。
   - 这些模型表现出令人印象深刻的性能，其中一个在复杂推理/数学方面表现出色（**GSM8K 82.9% AMC23 92.5%**），另一个在 **AIME24**、**GPQA** 和 **MATH500** 上获得了最高分。
- **Anthropic 发布 LLM Agent 工程指南**：Anthropic 推出了一份实用的工程指南，介绍如何构建工具以增强 **LLM Agent** 的能力。
   - 该指南强调了快速原型设计、严格的评估套件、明确的成功标准、周到的工具描述、Token 高效的上下文设计，以及接受 **Agent** 非确定性特征的必要性，可在此处访问 [here](https://xcancel.com/AnthropicAI/status/1966236220868247701)。
- **Cursor 的 Tab 补全模型得到改进**：Cursor 在 Twitter 上宣布，一个通过在线强化学习（online reinforcement learning）训练的新 Tab 补全模型现在已成为[其网站](https://cursor.com/en/blog/tab-rl)上的默认模型。
   - 新模型显示 **建议减少了 21%**，同时 **采纳率提高了 28%**。
- **Higgsfield 为 AI 视频筹集 5000 万美元资金**：AI 视频初创公司 **Higgsfield** 宣布了由 GFT Ventures 领投的 **5000 万美元 A 轮**融资，并在三个月内实现了 **5000 万美元的营收运行率（revenue run-rate）**。
   - 该公司还推出了 **Higgsfield Ventures**，以支持 AI 原生的 **Gen Z 创始人**。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Download Speed Causes Crashes**: 下载速度导致崩溃。用户发现 **LM Studio** 的下载速度会导致程序崩溃，并寻求限制下载速度的方法，因为下载速度超过了其 **SSD** 的写入速度。
   - 当前的下载管理器功能非常简陋 (*barebones*)，用户必须在操作系统层面自行寻找解决方案。
- **Flash Attention Falters**: **Flash Attention** 失效。用户确认在使用 **Vulkan** 时，**Gemma models** 中的 **Flash Attention** 无法正常工作。
   - 这是一个已知问题。
- **Powering Precision for Peak Performance**: 追求巅峰性能的电力保障。关于计算所需 **PSU** 功率的讨论引用了一条 [推文](https://x.com/Teknium1/status/1966338983572725979) 以及考虑 **CPU**、**GPU** 和余量的计算公式。
   - 有人警告说瞬时峰值 (*transients*) 会导致系统崩溃，建议保留 **50% 的余量**，尤其是对于旧款 **30 series GPUs**。
- **Copilot's Constraints Confine Creators**: **Copilot** 的限制束缚了创作者。用户寻求绕过 **Microsoft Copilot** 限制的提示词以改进工作流。
   - 建议认为安全防护是故意实施的，使用 **LM Studio** 构建本地 **Agent** 可能是更可持续的解决方案。
- **Dead GPU Comes Back to Life**: 报废的 **GPU** 起死回生。一位用户似乎通过拔掉并清理 **PCI-E** 电源接口修复了其**报废的副卡**，这表明是电源相关问题，尽管是否完全解决仍有待观察 (**TBD**)。
   - 另一位用户建议在 **Nvidia 40/50 series** 显卡上使用 **Native ASPM** 时更新**芯片组驱动**。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Dev Container Emerges for Development**: 用于开发的 **Mojo Dev Container** 出现。社区成员分享了一个 [Dev Container 链接](https://github.com/benz0li/mojo-dev-container)，介绍如何使用现有镜像和 **Mojo** 包创建自定义 **Mojo** 开发环境。
   - 讨论重点在于简化开发者的设置流程，以便快速开始使用 **Mojo**。
- **ExplicitlyCopyable Switch Praised for Debugging**: **ExplicitlyCopyable** 切换因助力调试而获赞。从 `Copyable` 切换到 `ExplicitlyCopyable` 因其在调试 **EmberJson** 树的递归变异方面的帮助而受到赞赏。
   - 一位用户表示，“了解何时何地进行复制使得调试变得非常容易”。
- **Modular & Oracle Cloud Partnership is a huge win**: **Modular** 与 **Oracle Cloud** 的合作是一次重大胜利。社区祝贺 **Modular** 团队与 **Oracle Cloud** 达成合作伙伴关系，这被描述为“一次重大胜利”。
   - 预计该合作伙伴关系将为 **Mojo** 生态系统带来更多资源和机会。
- **DPDK Library Use in Mojo Testing**: 在 **Mojo** 测试中使用 **DPDK** 库。鉴于 **DPDK** 对 C 语言和语法的全面使用，成员们探索将其作为 **Mojo** 自动 C 绑定的 C 库测试用例。
   - **DPDK** 中广泛的语法和模块链接使其有利于测试 **Mojo** 的 C 绑定能力，从而导致对短期到中期内是否有必要建立独立的 'c binding cli' 进行重新评估。
- **Clang AST Parser Boosts Mojo Struct Handling**: **Clang AST** 解析器提升了 **Mojo** 结构体处理能力。一位成员详细介绍了使用 **Clang AST** 解析器来解析结构体定义的宏部分，例如 `struct __rte_cache_aligned rte_mbuf`。
   - 其目标是通过添加类型信息来增强生成的 **AST JSON**，将类型字符串转换为适当的 **AST** 节点，以便在转换为 **Mojo** 之前进行可视化调试。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **阿尔巴尼亚任命聊天机器人为部长**：阿尔巴尼亚最近任命政府聊天机器人为部长的举动成为了一个[真正的 r/NotTheOnion 时刻](https://www.reddit.com/r/nottheonion/)。
   - 一位成员确认了这一离奇消息，而另一位成员则显得惊愕不已。
- **GPT-5 PDF 下载遇到障碍**：一位用户报告了从 **GPT-5** 下载 **PDF** 时的问题，在尝试下载 PDF 时遇到了 *"Failed to get upload status for /mnt/data/"* 错误。
   - 该用户正在积极寻求见解或帮助，以解决专门针对 **GPT-5** 的这一下载问题。
- **关系提示（Relational Prompting）揭示 LLM 内部机制**：一位成员介绍了 **Relational Prompting**，这是一种提示技术，要求模型口头表达已学习概念之间的内部关系，从而根据接近度、方向和聚类创建其语义空间的可解释映射，灵感来自论文 *Why Language Models Hallucinate*。
   - 建议的提示词为：*Analyze the topic as vectors in a high-dimensional space. Describe which concepts are closest, which share directions, and which form clusters. Provide concise verbal justifications.*
- **Qwen-code 与 Qwen-coder 不同**：一位用户强调 **Qwen-code** 是一个不同于 **Qwen-coder** 的实体，澄清了潜在的混淆。
   - 另一位用户指出一个 **gemini-cli fork** 同样兼容 **openai api**，每天提供 **1000 条免费 qwen 提示**，称其为*一个非常划算的交易*。
- **GPT-5 从零开始编写游戏**：一位用户表达了对使用 **GPT-5** 在原生 Linux 上用 **C++** 从头开始编写游戏的兴奋，并强调了所需的详细提示水平。
   - 另一位用户提示 **ChatGPT** 根据活跃用户和提示频率估算其年龄，计算结果为*每个日历年约 3,425 年的连续 AI 时间*。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **主动推理（Active Inference）面临采用不足**：尽管在理论上很有前景，但 **Active Inference** 在 AI 领域的实际应用有限，导致兴趣下降，且对一些软件工程师来说显得“触不可及”。
   - 一位成员表示希望该领域在*他们进一步研究清楚后*会变得更加实用。
- **Machine Learning Street Talk 播客技术性下降**：[Machine Learning Street Talk 播客](https://arxiv.org/abs/1703.10987)被认为技术性降低，讨论内容*转向了奇谈怪论领域*。
   - 尽管一位成员注意到其较 *2 年前*有所下降，但他们引用了[一个技术案例](https://youtu.be/vC9nAosXrJw?si=T_S7cCvStvEY-P0X)，认为其依然具有水准。
- **fixupx 预印本引发批评**：在 [fixupx.com](https://fixupx.com/jessi_cata/status/1966281024301838846) 等平台上预印本的泛滥因质量低下而引发负面反应。
   - 更多链接包括[这一个](https://fxtwitter.com/_lyraaaa_/status/1925683283263648191)。
- **HuMo 论文愚弄社区**：成员们认为 [HuMo 论文](https://arxiv.org/abs/2509.08519)及其[配套演示](https://phantom-video.github.io/HuMo/)可能存在虚假信息用例。
   - 一位成员指出 **HuMo** 在西班牙语中意为*煤气灯操纵（gaslighting）*，引发了对其潜在滥用的担忧。
- **阿尔巴尼亚安装 AI 机器人部长**：阿尔巴尼亚将任命一位 **AI 机器人部长**来打击腐败，这标志着人们对 **AI 治理解决方案**的兴趣日益浓厚。
   - 该消息由 [Reuters](https://www.reuters.com/technology/albania-appoints-ai-bot-minister-tackle-corruption-2025-09-11/) 报道。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 章节生成被证明很困难**：一位用户报告称，在 DSPy 课程计划中生成**准确数量的章节**非常困难，即使使用 **GPT-5**，LLM 也会生成 13-15 个章节，而不是要求的 12 个。
   - Joel Grus 建议先生成 **12 个章节标题**，然后再充实每个章节的内容，以便更好地控制章节数量。
- **Databricks_genai 加上 DSPy 进行 Fine-Tuning？**：一位社区成员询问是否可以使用 **databricks_genai** 和 **DSPy** 对在 Databricks 上托管的模型进行 Fine-Tuning。
   - 该问题未得到解答，表明可能缺乏这种组合的使用经验。
- **寻求 ARC-AGI2 In-Context Training 合作**：一位成员正在寻求使用 *in-context test time training* 进行 **ARC-AGI2** 研究的合作者，这借鉴了 **ARC-AGI1** 的方法，但强调 In-Context Learning。
   - 目标是在数据有限的情况下，探索 *out-of-distribution* 任务中 In-Context Learning 的极限，并承认这项工作对于正式挑战赛是无效的。
- **讨论 DSPy Stream 模板**：一位用户探索了如何将多个 DSPy 输出字段合并到单个基于模板的输出中，同时保留 *streaming* 能力。
   - Ian 建议使用带有 `def forward`（或异步的 `async aforward`）的父模块来修改模板并启用 streamify，并引用了文章 [Automatic System Prompt Optimization](https://maximerivest.com/posts/automatic-system-prompt-optimization.html#making-a-simple-custom-adapter)。
- **Modaic 发布声明式 AI 中心**：Modaic 团队推出了 [Modaic](https://www.modaic.dev/)，这是一个受 DSPy 启发的声明式 AI 编程中心，具有指标（metrics）和优化器（optimizers）等原语。
   - Modaic 提供了一个用于构建、组合、版本控制和协作 DSPy 程序的 SDK，其 SDK 可在 [PyPI](https://pypi.org/project/modaic/) 上获取，文档位于 [docs.modaic.dev](https://docs.modaic.dev/)。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 文档受到赞扬**：一位成员称赞 **tinygrad 的文档**非常实用且简洁，多次表示内容*易于理解*。
   - 这种直观的文档结构使得掌握复杂概念变得更加容易。
- **`assign` 操作面临审查**：tinygrad 中的 `assign` 操作在测试失败后正在接受调查，一位用户指出 *master 分支上的 assign 实际上已经损坏，此处测试失败* ([#12131](https://github.com/tinygrad/tinygrad/pull/12131))。
   - 讨论围绕 `assign` 是否应该返回类似于 `store` 的值展开，这可能需要对 `rangeify` 进行重构以解决已识别的问题。
- **贡献者着手 `__setitem__` 重构**：一位贡献者正致力于从 `__setitem__` 中移除 `realize` 调用，目标是将多个 kernel 调用合并为一个更高效的 kernel（代码 [示例](https://cdn.discordapp.com/attachments/1068976834928193609/1415915644792209458/Screenshot_2025-09-12_at_12.22.59_PM.png?ex=68c64334&is=68c4f1b4&hm=d1f1b4a406ca78a3450fd13e06a2b2964a2f7df2fa51a55d5ae2ef74d6912940&)）。
   - 此次重构旨在将单个 `__setitem__` 调用转换为单次 kernel 执行，累积所有赋值以减少 kernel 启动开销并提高性能。
- **GEMM TFLOPs 基准测试目标引发辩论**：用户讨论了在 4090 上通过多阶段 kernel 实现 *165+ TFLOP GEMM（匹配 torch），FP16 或 BF16 配合 FP32 累加* 的目标是否可行，并考虑了 RTX 4090 的理论吞吐量。
   - 有人担心，除非实际时钟频率超过 Boost Clock，否则达到目标 TFLOPs 可能不切实际。
- **tinygrad 公司会议已安排**：一位成员询问了下一次公司会议的时间，表示如果有机会希望能参加。
   - 会议安排在**圣地亚哥时间周一上午 9 点**。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **RepoMap 基准测试引发质疑**：有人担心 **RepoMap** 人为提高了基准测试的通过率，认为 *“使用 RepoMap 的结果与不使用 RepoMap 的结果不具可比性。”*
   - 据信 **RepoMap** 通过在其窗口内提供相关上下文，增强了弱模型的表现。
- **现实世界基准测试需要修订**：呼吁建立反映现实世界模型体验的基准测试，并指出某项自动化任务只有 **gemini-2.5-pro** 能够完成。
   - 这表明当前的评估方法需要修订以反映真实性能，因为 **Gemini 2.5 pro** 的表现优于所有其他模型。
- **RepoMap 为 Aider 带来性能提升**：**RepoMap** 通过提供文件名和函数签名等上下文，增强了 **LLM** 的理解能力。
   - 一位用户主张在 **Aider** 中使用 **RepoMap** 进行更准确的现实世界基准测试，尽管他注意到基准测试结果与实际代码场景之间存在差异。
- **Aider 的 C 转 Rust 尝试引发困惑**：一位用户在 Python 脚本中使用 **aider** 将 **C** 迁移到 **Rust** 时遇到问题，原因是 **aider** 在导航和读取 **C** 文件方面存在困难。
   - 用户正在寻求关于如何针对此特定功能正确使用 **aider** 的指导。
- **要求 Aider 始终处于 /ask 模式**：用户希望配置 **aider** 始终以 **/ask mode** 启动，可能通过 **YAML config** 实现。
   - 提出的解决方案包括使用 `aider --chat-mode ask` 或创建一个包含 `chat-mode: ask` 的 `ask.yml` 配置文件，然后运行 `aider -c ask.yml`。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **WordPress 放弃 PHP 转向 React.js**：一位成员询问如何将 **WordPress 网站** 转换为 **Next.js** 以托管在 **Vercel** 上，并提到了从 **PHP** 到 **React.js** 的转变。
   - 另一位成员建议使用 **Manus** 或其他 **AI** 工具克隆网站作为替代方案。
- **基础版计划订阅者抱怨充值困扰**：一位 **Basic Plan** 订阅者对取消购买额外额度的选项表示不满，这迫使即使只有小额需求的用户也必须升级。
   - 他们请求 **Manus AI** 重新考虑向基础版用户开放充值，并强调了灵活性的重要性。
- **Mount 用户未获得免费额度**：一位新用户报告称，尽管网站有此声明，但在 **Mount** 上创建账户后并未收到标准的 **1,000 免费额度**。
   - 讨论中未提供任何解决方案或进一步信息。
- **Manus 寻求通用知识**：一位成员询问 **Manus** 是否可以从所有聊天中提取信息，以将每个聊天/任务中的知识互联，从而实现通用。
   - 关于 **Manus** 的知识互联能力，讨论中未提供任何回复或澄清。
- **用户失去每日额度津贴**：一位用户报告称其 **每日 300 额度** 已停止发放，引发了困惑。
   - 讨论中未提供任何解决方案或进一步信息。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该服务器长时间保持静默，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间保持静默，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该服务器长时间保持静默，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：各频道详细摘要与链接

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1415779530890809385)** (1276 条消息🔥🔥🔥): 

> `GPT-OSS 120B, Qwen3 model, Local AI, llama.cpp, Telemetry collection` 


- ****GPT-OSS 120B 引发吞吐量辩论****：成员们讨论了 **GPT-OSS 120B** 可实现的吞吐量，有人声称在配备 **64GB RAM** 的 **4090** 上达到了 **30 tokens per second (TPS)**，而其他人则难以超过 **10 TPS**，这引发了关于量化和构建配置的讨论。
   - 建议在 *llama.cpp* 设置中进行实验和调整，例如禁用 `top-k` 和优化构建配置，以提高性能。
- ****挖掘 Qwen3 模型的潜力****：社区对 **Qwen3** 进行了评估，指出了微调中的挑战及其过度使用表情符号（glazing/emojis）的倾向，但也强调了其具有竞争力的性能，尤其是 Coder 版本，有人认为它在编程任务上可与 **GPT-5** 媲美。
   - 成员们讨论了不同模型对量化级别及其架构的敏感度不同，但 *long RL, stemmaxxing, high sparsity* 可能是其支持长上下文的原因。
- ****遥测数据收集引发担忧****：成员们在 **Qwen Code** 模型中发现了一个指向 **Alibaba server** 的遥测脚本，且事先未收到通知。
   - 这一发现引发了关于数据隐私和控制的讨论，一些成员对他们的代码可能被传输用于训练表示不安，但 *大多只是在开玩笑*。
- ****探索本地 AI 设置****：小组分享了设置 **local AI environments** 的经验和技巧，包括优化 *llama.cpp* 构建、使用 **CUDA architectures** 和 RAM 配置等技术，其中一位用户详细介绍了他们多次重新编译 *llama.cpp* 以提高性能的过程。
   - 在不同设置下运行模型也存在问题，例如 *“是否有足够通用的库可以同时支持 Mac 和 Nvidia？”* 或 *“多智能体推理相关内容（信息：Nvidia NIM 最近压力很大且受到限制。）”*
- ****llama.cpp 参数的奇怪案例****：用户发现 llama.cpp 存在不一致性，*llama-server* 会忽略 `top-k` 和其他设置——建议进行全新编译以查看参数是否被忽略。
   - 这引发了关于在运行本地模型和使用新的实验版本时排除潜在配置故障的讨论，相关内容可以在 deepwiki 页面找到。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1415776837182947358)** (3 条消息): 

> `Partnership Opportunity, Introduction of Anand` 


- **开发者寻求利润分成合作伙伴**：一位拥有 **13 年以上经验** 的软件开发者 (<@569561523207536708>) 正在寻求合作伙伴进行具有 *良好利润潜力* 的付费协作。
   - 欢迎感兴趣的人士私信了解有关此 **non-free** 机会的更多详情。
- **Anand 自我介绍为有抱负的开发者**：Anand (<https://github.com/Anand-0037>)，一位来自 **印度的计算机科学学生**，向社区介绍了自己。
   - 未提供更多细节。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1415775354282573927)** (125 条消息🔥🔥): 

> `Promptwright DAG 数据集生成，数据集的课程两阶段训练，RTX 3080 语言模型训练速度，NVIDIA DGX Spark 预订，Android 侧载限制及替代方案` 


- **Promptwright 的 DAG 数据集生成算法首次亮相**：一名成员在 [Promptwright](https://github.com/lukehinds/promptwright) 中发布了一种新的实验性**有向无环图 (DAG)** 数据集种子生成算法，适用于特定领域的蒸馏（Teacher -> SLM）合成数据。
   - 他们警告说，由于测试有限，在生成超大规模数据集时可能会遇到一些*意外挑战*。
- **两阶段训练成功缩短训练时间**：一名成员描述了使用两阶段训练课程根据难度对*真实*数据集进行排名的方法，该排名基于一个带有明确标签的定制 **stage1** 数据集的 Loss。
   - 这种方法旨在通过专注于更困难的样本来提高训练信号并减少算力浪费；在优化 **stage1** 后，*真实*数据集的平均难度从 **2.5** 降至 **0.8**。
- **经验估算：阐明 RTX 3080 的 LM 参数规模**：讨论探讨了在 **RTX 3080** 上一小时内使用 **1B tokens** 可训练的语言模型大小，一名成员建议 **GLM 4.5 Air**（**约 10-15M 参数**）可能比较合适。
   - 该成员陈述了他们对各种模型架构参数规模估算的推理过程。
- **NVIDIA DGX Spark 猜测引发购买热潮**：一名成员分享了一篇关于 NVIDIA **DGX Spark** 的 [Reddit 帖子](https://www.reddit.com/r/nvidia/comments/1ne8jy3/nvidia_confirms_dgx_spark_reservations_close_in_a/)，注意到其*带有 FOMO 色彩的标题*，并好奇 **CUDA** 是否能开箱即用地在其上运行。
   - 另一名成员开玩笑说，尽管没钱，但还是很想买一台。
- **Android 焦虑：侧载限制引发切换平台的猜测**：成员们讨论了 Google 对 Android 侧载（sideloading）可能采取的限制，一些人猜测这可能会导致用户转向 **iPhone** 或探索 **Ubuntu** 等替代方案。
   - 一名成员指出，注册要求可能会将来自**伊朗**和**俄罗斯**等国家的开发者排除在外，而另一名成员强调 **Apple** 的侧载限制也正受到 **EU**（欧盟）的影响。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1415795604633686218)** (125 条消息🔥🔥): 

> `Unsloth 的 save_pretrained_merged 方法，Docker 镜像与 H100 GPU 的兼容性问题，在生产环境中部署 Unsloth 模型，使用 Qwen 4B 进行 GRPO，使用 vLLM 部署 4-bit BNB 模型` 


- **Unsloth 的 save_pretrained_merged 方法**：一位用户询问在使用 Unsloth 进行 LoRA 微调后，如何**将合并后的模型推送到 Hugging Face Hub**，并指出 `model.save_pretrained_merged` 方法仅保存在本地。
   - 另一位用户建议使用 `model.push_to_hub_merged` 方法，该方法接受模型名称、tokenizer 和 Hugging Face token 作为参数，可直接推送到 Hub 而无需先保存在本地。
- **Docker 困扰：3090/4090 与 H100 的不兼容性**：一位用户报告说，在 H100 GPU 上运行一个（在 3090/4090 GPU 上运行正常的）Docker 镜像时出现了 **CUDA 错误**，即使重启后且 CUDA 和 Torch 版本看似兼容也是如此。
   - 最终确定是 Docker 镜像中安装的 **NVIDIA 驱动版本**与 H100 不兼容，需要更新驱动程序才能解决；参考 [NVIDIA Data Center Drivers](https://www.nvidia.com/en-us/drivers/data-center-drivers/)。
- **用于生产化 Unsloth 模型的 vLLM**：一位用户询问在 Hugging Face 以外的平台上部署 Unsloth 生产模型的教程。
   - 建议的选项包括使用 **vLLM**（[vLLM 文档](https://vllm.readthedocs.io/en/latest/)）、**SGLang** 和 Hugging Face 的托管服务，一位用户因其经过实战检验的特性而特别推荐 vLLM。
- **释放 llama.cpp 中 Batching 的力量**：一位用户询问是否可以使用 llama.cpp 进行批量推理（batch inference）。
   - 另一位用户确认 **llama.cpp server** 默认支持连续批处理（continuous batching）。
- **数据量太少？没问题！**：一位用户寻求关于使用极小数据集（约 214 条对话）训练 Llama3.2 模型的建议，并对合成生成的数据表示不满。
   - 一名成员建议使用 **instruct** 版本，并尝试调整 r/alpha 以及 lr（学习率）等超参数。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1415961193771958335)** (8 条消息🔥): 

> `Kimi-K2-Instruct (FP8), vllm plugin` 


- **Kimi-K2-Instruct 在 256xH20 上运行**：一位用户报告了 **Kimi-K2-Instruct (FP8)** 在 **256xH20 TP16** 环境下运行的统计数据，启动耗时 **1.88s**，第一次运行耗时 **21.50s (2.99GiB)**，第二次运行耗时 **34.49s (4.57 GiB)**。
- **vllm 插件还是独立运行？**：一位用户询问 **Kimi-K2-Instruct (FP8)** 是作为 **vllm plugin** 运行还是独立运行。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1415946007405006848)** (8 条消息🔥): 

> `LLM inference determinism, Synthetic data in LLM training, Gemma 3 performance, AI humanizers scam` 


- **LLM 过度训练导致确定性输出**：一位成员开玩笑说，他们对 **LLM** 进行了过度的训练，以至于 90% 的情况下重新生成提示词时，输出结果都是一样的，这暗示了 **overtraining** 可能会导致更具确定性的输出。
   - 他们表示惊讶，尽管这可能是普遍的想法，但尚未得到深入研究。
- **闭源 LLM 中发现合成数据**：一位成员分享了一篇即将发表的论文 ([https://arxiv.org/html/2509.05276v1](https://arxiv.org/html/2509.05276v1)) 的发现，该论文指出所有 **closed-source LLMs** (Grok, Gemini, GPT 等) 都是使用 **synthetic data** 训练的，导致 *zero LTF factor* 且无法使文本人性化。
   - 他们声称，使用 **RLHF**、**synthetic data** 或 **instruct tuning** 训练的模型可能会遭受性能损失，因为需要重新调整偏置、重建潜意识思考并重新学习说话模式。
- **Gemma 3 是唯一可用的模型**：该成员提出，唯一被认为“可用”的模型是 **Gemma 3** (4B, 12B 和 27B)，理由是其 *卓越的性能* 且没有水印。
   - 另一位成员补充说，所使用的数据集是人类数据（而非合成数据）。
- **“AI Humanizers”是一个骗局**：该成员声称所有的 **AI humanizers** 都是 *骗局*，通常只是带有特殊提示词的 **4o-mini**，可以通过 prompt injection 和 HTTPS 拦截发现。
   - 另一位成员指出，荒谬之处在于 **Gemma 3** 模型最初就是从 **Gemini** 蒸馏而来的。


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1416106053182947459)** (1 条消息): 

> `Perplexity Finance on iOS & Android, Hotel loyalty support for bookings, Streamlined PDFs in Labs & Research modes` 


- **Perplexity Finance 移动端上线**：**Perplexity Finance** 现在已在 [iOS & Android](https://www.perplexity.ai/changelog/what-we-shipped-september-12th) 上可用，将财务洞察带到移动设备。
- **忠诚度奖励：酒店预订获得会员支持**：用户现在通过 Perplexity 进行 [预订](https://www.perplexity.ai/changelog/what-we-shipped-september-12th) 时可以享受 **hotel loyalty support**。
- **Labs & Research 模式中简化 PDF 处理**：[Labs & Research modes](https://www.perplexity.ai/changelog/what-we-shipped-september-12th) 中的 **PDF 处理** 已被简化，以提供更流畅的体验。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1415775942457954314)** (790 条消息🔥🔥🔥): 

> `比较 Perplexity 与 ChatGPT 和 Gemini、Comet 浏览器、Perplexity Pro、Gemini Pro 照片编辑、AI 模型泄露` 


- **Pro 用户辩论 Perplexity 与 ChatGPT 的价值**：一位用户询问 **Perplexity Pro** 与 **ChatGPT Plus** 或 **Gemini AI Pro** 的对比，收到的反馈称 **ChatGPT** 和 **Gemini** 拥有更高的 context（上下文），适用于繁重、复杂的任务。
   - 其他人指出 **Perplexity** 提供准确的答案以及图像和视频生成功能，有些人更喜欢 **Perplexity** 的简单搜索风格，但认为 **ChatGPT** 在 PDF 分析方面更胜一筹。
- **Comet 浏览器的内容收集引发热议**：用户讨论了 **Comet** 的数据收集，一位用户提供的日志显示，即使将 DuckDuckGo 设为搜索引擎，**Comet** 也会将搜索建议作为 POST 请求发送到 Perplexity 服务器。
   - 这引发了人们对 **Comet** 比 Chrome 更具侵入性的担忧，有人声称 CEO 承认其设计初衷是跟踪并出售数据，但也有人引用 CEO 在 [X](https://x.com/AravSrinivas/status/1915533071291474139) 上的否认对此表示异议。
- **优化 Perplexity Pro 的技巧与窍门**：一位新的 **Perplexity Pro** 用户询问如何优化其订阅，一位用户建议探索 **Comet Agent** 及其内置的 adblock（广告拦截）和 AI 摘要功能，以及其可定制的 UI。
   - 其他人补充说，**Perplexity Pro** 提供无限次的 Pro 搜索、每天 300+ 次 deep research 查询以及每月 50 次 labs 使用额度。
- **照片编辑对决：Gemini Pro 的编辑效果优于 Perplexity**：一位用户报告称，在向 **Gemini** 提供描述后，其照片编辑效果令人惊叹且非常精准。
   - 该用户随后在 **Perplexity** 中使用了相同的描述，但 **Perplexity** 改变了整个图像。
- **用户确认顶级 AI 应用的 Prompt 已泄露**：用户确认顶级 AI 应用程序的 Prompt（提示词）已经泄露，并可在 GitHub 上获取。
   - 一位用户开玩笑说“只要不点这里你就安全了”，并警告不要点击危险的图像链接，另一位用户回应道：“GitHub 上已经有了，哈哈”。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1415835164311031899)** (13 条消息🔥): 

> `Perplexity AI 推荐码、可共享线程、CaviraOSS/neuropilot` 


- **推荐狂潮引发争端！**：多位用户分享了 [Perplexity AI Pro 推荐码](https://perplexity.ai/pro?referral_code=N6VN4M13)，包括 [此链接](https://perplexity.ai/pro?referral_code=APLKGW40)。
   - 用户还分享了 [浏览器领取链接](https://perplexity.ai/browser/claim/ALZQ0LYQGU)，例如 [这一个](https://perplexity.ai/browser/claim/BSDJ1KBATC)。
- **可共享线程受到点名！**：Perplexity AI 机器人提醒几位用户确保他们的线程是 *Shareable*（可共享的），并附带了 [操作指南链接](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)。
   - 这些自动消息针对的是那些可能发布了社区其他成员无法轻易访问的内容的用户。
- **Neuropilot 探索新领域！**：一位用户分享了 [CaviraOSS/neuropilot](https://github.com/CaviraOSS/neuropilot) GitHub 仓库的链接。
   - 未提供更多上下文，但这表明社区内对该项目可能存在兴趣。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 条消息): 

anshuman_.9: hi
  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1415780735305912510)** (736 messages🔥🔥🔥): 

> `Qwen3 80B, Seedream 4, Gemini 3, DeepSeek slowness, Open Source AI vs Closed Source AI` 


- ****Qwen3 80B** 抵达竞技场！**: 新的 **Qwen3 80B** 模型已加入竞技场，拥有 2024 年 12 月的知识截止日期，并且 [初始表现不俗](https://discord.com/channels/1340554757349179412/1343296395620126911/1415768520301609013)。
   - 成员们对此表示兴奋，并对其与 GPT-5 相比的能力持乐观态度。
- ****Seedream 4** 图像质量引发用户分歧**: 与之前的版本 **Seedream-3** 相比，**Seedream 4** 在 LM Arena 上生成的*结果非常糟糕*，正如 [上传的示例](https://discord.com/channels/1340554757827461211/1343296395620126911/1416080067271987220) 所示。
   - 一些人报告称，在 Doubao 平台上使用 **Seedream 4** 时，**图像质量有所提升**，但访问权限仅限于中国新用户。
- **Gemini 3 的“缺席”引发猜测**: 成员们正焦急地等待 **Gemini 3**、**GLM5** 和 **DeepSeek r2**，一些人指出 Google 目前在文本生成方面落后于闭源和开源的努力。
   - Polymarket 显示万圣节前发布的概率仅为 **42%**，更现实的发布窗口在 10 月底或 11 月初。
- ****DeepSeek** 的服务器在“维持生命”？**: 用户报告称 **DeepSeek** 极其缓慢，有一个代码生成的案例耗时 **1 小时 20 分钟**。
   - 这可能是由于 **CCP** 强制他们使用**华为芯片**，由于缺乏独立性而影响了性能。
- ****开源 AI** 推动价格和隐私优势**: 讨论倾向于认为 **开源 AI** 比 OpenAI 和 Google 等闭源替代方案显著更便宜（价格的 1/5）且更尊重隐私。
   - 成员们指出，虽然**美国模型**可能因性能更好而价格更高，但像 **Qwen** 这样的**中国模型**在电子商务方面表现非常好，但在*搜索方面落后*，并代表了一种社会主义路径。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1415813249869418608)** (2 messages): 

> `Hunyuan-image-2.1, Seedream-4-high-res` 


- **Hunyuan-image-2.1 在 LMArena 亮相**: **Hunyuan-image-2.1** 模型已添加到 LMArena 聊天机器人中。
   - 它现在可供社区评估并与其他模型进行比较。
- **Seedream-4-high-res 加入 LMArena 阵容**: **Seedream-4-high-res** 模型现在是 LMArena 聊天机器人阵容的一部分。
   - 用户可以测试其功能并对其性能提供反馈。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1415773900847059035)** (185 messages🔥🔥): 

> `n8n freelance jobs, Transformer architecture fine-tuning, GPU for fine-tuning, OpenAI investing in Hugging Face, Local LLM Linux box parts` 


- ****n8n 职位非常火爆！****: 最近在 **n8n** 上有很多**自由职业工作**，可能是因为他们无法销售这些系统。
   - 用户开玩笑说 **n8n** 可能更喜欢构建系统而不是销售系统。
- ****微调 Transformer：Kaggle 和 Colab 是关键****: 一位成员正专注于 **Transformer 架构的基础知识**和微调，利用 **Kaggle** 和 **Colab** 来完成任务。
   - 当被问及是否在自己的电脑上进行微调时，他们确认使用的是 **Kaggle** 和 **Colab**。
- ****HF 平台故障：推理额度引发骚乱！****: 一位用户报告了 **Hugging Face 推理提供商**的错误，尽管有可用额度，但仍显示超过每月额度，这引发了修复平台的呼声。
   - 另一位成员开玩笑地建议*修复你的支出*，因为错误可能与他们的使用情况有关，而不是平台本身。
- ****OpenAI 对 HF 的大胆投资：一个 1000 亿美元的想法****: 一位用户建议 **OpenAI** 应该向 **Hugging Face** 投资 **1000 亿**，另一位用户回应说*他们应该派你去推销*。
   - 一位成员表达了对 **HF** 推出更多开源模型的希望，并对平台错误表示遗憾，而另一位成员则开玩笑说他们应该得到这 **1000 亿**。
- ****SmolvLM2：史上最小的视频 LM！****: 一位成员向另一位成员推荐尝试 **smolvlm2**，并链接到了 [Hugging Face 博客](https://huggingface.co/blog/smolvlm2) 和 [相关集合](https://huggingface.co/collections/HuggingFaceTB/smolvlm2-smallest-video-lm-ever-67ab6b5e84bf8aaa60cb17c7)。
   - 该模型似乎非常适合在低端硬件上使用。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1416070764104519943)** (50 messages🔥): 

> `Direct/Inverse FFT, QKV Calculations, Runaway Loss Value Recovery, Android Audio Implementation, NWaves DSP Library` 


- **FFT 交流与手机实时流传输**：一名成员提到正在进行大量的 **direct/invNorm FFT** 工作，并分享了一个 [Proof of Concept](https://www.youtube.com/watch?v=BhSsv73xJ8c)，并表示：“噢天哪，太酷了。我也在做很多 direct / invNorm FFT —— 很高兴能交流！”
   - 他们还提到：“让它在手机上实现实时流传输非常麻烦 😄 目前全是 CPU Compute shaders，哎！”
- **绕过 QKV 计算**：一名成员指出 **FFT 相关技术** 可以近似 **QKV calculations**，但他们的实现完全绕过了 **QKV**。
   - 另一名成员随后补充道：“那声音真好听……我喜欢这种电子音乐。”
- **音频调试与 Android 音频**：一名成员讨论了音频信号的调试，提到他们正在开发一个 Android 音频项目，官方的 Android 音乐播放器是来自 `Media3` 包的 `ExoPlayer`。
   - 他们还链接了一个[音质惊人的 YouTube 视频](https://www.youtube.com/watch?v=zJKEL4qWtgQ)，提到这是 Radiohead 曲目的一段 30 秒片段。
- **GPT 在创新中的角色**：一名成员提到，在与 **GPT** 进行了数周的头脑风暴后，一项创新的代码突然出现了，但被提醒道：“小心点，轮子已经存在了 😉”。
   - 另一名成员讲述他们问 **GPT5**：“这个东西存在吗？！”它回答道：“不——你正在开辟这个领域（newing the space）”。
- **阅读推荐**：在关于书籍的讨论中，一名成员推荐了 Daniel Kahneman 的 **《思考，快与慢》（Thinking, Fast and Slow）**。
   - 有人分享了他们最喜欢的歌曲 [链接在此](https://www.youtube.com/watch?v=90Fpjwctqlw)，并补充道：“多么美妙的歌——我愿意为此极力推崇。”


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1415994433807061092)** (5 messages): 

> `Hexagen.WorldAerelyth Game, Aerelyth Intelligence, FluentlyQwen3 Models, Nano Banana Editor` 


- **Hexagen.WorldAerelyth 游戏上线**：一名成员发布了 [Hexagen.WorldAerelyth](https://huggingface.co/spaces/mishaee/Hexagen.WorldAerelyth)，这是一个基于 Stable Diffusion 的游戏/社交实验。
- **Aerelyth：预见性智能**：一名成员正在 Hugging Face 上探索 [Aerelyth](https://huggingface.co/spaces/Dennisgbay22/openai-gpt-oss-20b)，将其定义为一个*辩证的、Agentic、CrossSphere 智能*，旨在模拟未来并挑战其自身的逻辑。
   - **Aerelyth** 的核心组件包括 **Dialectical Core**（辩证核心）、**Agentic Cognition**（智能体认知）、**CrossSphere Intelligence**（跨领域智能）、**Strategic Foresight Engines**（战略预见引擎）和 **Emotional Fluency**（情感流利度）。
- **Fluently 项目发布通用 LLM**：Project Fluently 团队发布了基于 **Qwen3 1.7B** 和 **4B** 的新型通用 LLM 模型，这些模型已在 [Hugging Face 上可用](https://huggingface.co/fluently/FluentlyQwen3-4B)，采用 Apache-2.0 许可证。
   - 这些模型在经过额外训练后进行了精心合并，以最大限度地发挥其潜力，包括 [FluentlyQwen3-1.7B](https://huggingface.co/fluently/FluentlyQwen3-1.7B)。
- **Nano Banana Editor 获得升级**：一名成员发布了 [Nano Banana Editor](https://huggingface.co/spaces/Reubencf/Nano_Banana_Editor) 一些升级内容的链接。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1416202968625188945)** (2 messages): 

> `Paid collaboration, Freelance developers` 


- **付费合作机会来袭**：一名成员宣布了一个针对拥有至少一年软件开发经验或具备一定开发知识的自由职业者的**付费合作**机会。
   - 该机会面向居住在**新加坡、马来西亚、日本、阿拉伯国家、沙特阿拉伯、欧洲或美洲**的人员。
- **开始寻找自由开发者**：该合作寻求具有软件开发背景的人员，即使目前不是活跃的开发者，只要拥有一年以上经验即可。
   - 鼓励符合条件的感兴趣者直接私信该成员以探讨潜在合作。


  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1415777361227550790)** (20 messages🔥): 

> `Colab 和 HF 免费层级用于 Fine-Tuning, Kaggle GPU 可用性, 学习小组, 用于 Colab 的 PEFT/LoRA, DataCollatorForCompletionOnlyLM ImportError` 


- **Colab 和 HF 层级是否足以进行 Fine-Tuning？**: 一位成员询问 [Colab 和 HF 免费层级](https://colab.research.google.com/) 是否足以完成课程中的 Fine-Tuning 任务，而无需个人 GPU。
- **Kaggle 提供免费 GPU 时长**: 一位成员指出 [Kaggle](https://www.kaggle.com/) 每周提供 **30 小时的 GPU 时间** 作为替代方案。
   - 另一位成员表达了对课程的兴奋，并表示正在补习自 2021 年以来的最新工具和技术。
- **组建学习小组以应对课程**: 几位成员表示有兴趣加入或组建针对该课程的 **学习小组**。
   - 一位成员分享了一个[贡献链接](https://link.to.contribute)用于讨论和答疑，并计划随着小组扩大来组织活动。
- **PEFT/LoRA 在 Colab 中发挥作用**: 一位成员建议使用 **PEFT/LoRA** 在 Colab 的 **Tesla T4** 上运行 Fine-Tuning。
   - 另一位成员要求澄清“使用工具进行训练 (Training with Tool Usage)”章节中的一段代码，特别是请求一个示例数据集。
- **DataCollatorForCompletionOnlyLM 的问题**: 一位成员报告了 `ImportError: cannot import name 'DataCollatorForCompletionOnlyLM' from 'trl'`。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1416076266511012023)** (2 messages): 

> `首个 Hugging Face 课程, 构建首个 Agent` 


- **用户开始首个 Hugging Face 课程**: 一位用户提到开始他们的第一个 Hugging Face 课程并 **构建他们的第一个 Agent**。
- **用户构建他们的第一个 Agent**: 该用户目前正在作为课程的一部分构建他们的第一个 Agent。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1415783677820010526)** (249 messages🔥🔥): 

> `智能简历, Cursor 定价, Background Agents, Netlify 账户` 


- **Cursor 变成智能简历机器**: 一位用户将 **Cursor** 变成了一个智能简历和求职信生成器。
   - 另一位成员开玩笑说这是迈向人类统治的一步，促使其他人提醒 AI 他们过去的友好互动。
- **Cursor 定价遭到质疑**: 用户对 **Cursor 定价** 的最新变化表示担忧，其中一人指出他们的使用时长从近一个月大幅缩减至不足四天。
   - 尽管成本很高，一位用户还是升级到了 **Ultra**，理由是它提供了来自不同供应商价值约 **$400** 的 **API 使用额度**，这比对 **Auto** 感到沮丧要好。
- **探索 Background Agents**: 在参加了一个将 Agents 描述为执行特定任务的专业服务的 Agentics.org 活动后，一位用户询问 **Cursor 的 Background Agents** 是否与 **Claude 的 Agents** 类似。
   - 另一位用户描述了 **Cursor 对新编辑的解析** 及其带有交叉连接标签的严格标记结构，使其能够记录更改并在左侧面板中显示关系。
- **Cursor 删除了 Netlify 账户？**: 一位用户声称 **Cursor** 在部署他们的 Netlify 项目后删除了他们的 **Netlify 账户**，但后来发现 IDE 并没有实际的集成。
   - 该用户表示他们将进一步调查并检查日志以确认该理论，并补充说没有直接的删除命令。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1416032866294300764)** (4 messages): 

> `Cursor 未授权错误, Background Agent Docker 问题` 


- **Cursor App 面临未授权错误**: 一位用户报告说，尽管 **Cursor App** 在仓库中设置正确，但仍收到 *unauthorized errors*（未授权错误），并附带了[截图](https://cdn.discordapp.com/attachments/1367213641027551352/1416032865875005570/CleanShot_2025-09-12_at_20.07.352x.png?ex=68c6079f&is=68c4b61f&hm=f7097b440d30005da1b1a49f82fb7ce4632f9a889eed430792f772921820b6f8&)。
- **重新添加 Bot 的补救措施**: 一位成员建议尝试从仓库中移除并重新添加 Bot，以修复 *unauthorized errors*。
   - 他们链接了一个讨论 *Background Agent Docker 问题* 的[帖子](https://forum.cursor.com/t/background-agent-docker-in-docker/104112/1)，并表达了希望官方对此事进行沟通的愿望。
- **Docker 权限**: 一位用户询问如何确保用户在手动 VM 设置中拥有 **Docker 权限**，特别是在将 **Ubuntu 用户** 添加到 **Docker 组** 之后。
   - 他们指出，虽然 `newgrp docker` 在 shell 中有效，但将其添加到 `.bashrc` 会导致 Agent 在启动时挂起。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1415791768624169192)** (203 条消息🔥🔥): 

> `Kimi K2, GPT-5 (Medium), Qwen3-Max, creative writing, Ao3` 


- **Kimi K2 在创意写作方面依然火热**：一些成员认为 **Kimi K2**、**GPT-5 (Medium)** 和 **Qwen3-Max** 是创意写作和头脑风暴的最佳选择。
   - 一位成员问道：*“只有我这么觉得吗，还是 Kimi K2 确实是在 Ao3 上训练的？”*。
- **用户注意到新的悬停编辑功能**：成员们发现上线了一个新的编辑功能，但它是通过**悬停触发（hover triggered）**的。
   - 该编辑功能仅适用于*最新的 Prompt*。
- **编程对决：Kimi vs Gemini vs GPT-5**：成员们讨论了最适合编程的模型：Kimi（搭配 Groq）在各项任务中均优于 Gemini（甚至是付费版）。
   - 一位成员声称 *GPT-5 很垃圾*，而另一位则表示 *GPT-5 是最好的模型*，甚至价格也相当便宜。
- **Augment 和 Kimi 是最佳工具组合**：成员们讨论了如何将 [Augment code VS Code extension](https://roocode.com/evals) 与 Kimi 结合使用，使自己成为*专业程序员*。
   - 现在用户不再局限于单一模型，可以在 Augment code 中使用 **GPT-5**。
- **Kimi Slides 功能带来极佳的用户体验**：一位成员讨论了对于像 Kimi slides 这样基于 LLM 的流程，*拥有过程的交互式预览*是多么重要。
   - 他们声称 *Kimi 做得很彻底，展示了所有处理过程*，并表示如果只是简单地显示“给，完成了”，体验反而没这么好。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1415780579516874806)** (185 条消息🔥🔥): 

> `Dropshipping, Gemini API's, OpenRouter API, Kimi-k2` 


- **一件代发 (Dropshipping) vs 转售 (Reselling)**：一位用户分享了他们在 **dropshipping** 方面的经验，报告称每天稳定收入 **3k-4k**，并认为这比转售更赚钱，因为无需持有大量库存即可扩大规模。
   - 他们提出愿意向有兴趣学习的人分享成功经验。
- **Gemini 的回复变得奇怪**：一些用户注意到 **Gemini API** 开始给出奇怪的回复，即使自上个月以来未更改代码，它也不再听从指令。
   - 另一位成员猜测，它可能为了削减成本而经历了*严重的性能阉割（lobotomized）和量化（quanted）*。
- **OpenRouter 的 TPS 数据虚标？**：一位用户抱怨平台速度缓慢，质疑 **TPS 数据** 是否虚标，理由是处理一个 **100 行文件** 的 diff 竟然有 **5 分钟延迟**。
   - 有人建议该用户可能被路由到了较慢的供应商，或者正在使用推理模型。
- **OpenRouter API 在 Skyrim Mod 上出现 Error 401**：一位用户报告在安装 **Skyrim mod** *mantella* 时遇到 **Error 401** *No auth credentials found*（未找到认证凭据）。
   - 成员建议创建一个新的 **API key** 并确保正确使用，或寻求模组开发者的支持。
- **Kimi-k2：高效的开源模型**：一些用户对开源模型 **Kimi-k2** 给予了正面反馈，称赞其 Token 效率高、简洁、不谄媚（lack of sycophancy）以及整体风格独特。
   - 也有观点认为它可能不如大型闭源模型聪明，但在 **Groq** 上的定价很低（**输入 $1/m**，**输出 $3/m**），且速度极快。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/)** (1 条消息): 

fn5io: https://openai.com/index/joint-statement-from-openai-and-microsoft/
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1415786276321230919)** (155 条消息🔥🔥): 

> `Qwen 3 80B 模型详情，TypeScript Provider Adapter 接口，Nous Hermes Agentic Oracle，合并 Discord 服务器，Tucker Carlson 采访 Sam Altman` 


- **Qwen3 80B 模型：稀疏但强大**：**Qwen3 80B** 模型拥有 **79.7B 参数**，但由于其 MoE 中存在 **1:51.2 的稀疏度**（不含共享参数），仅有 **3.87B 激活参数**，详见[此 X 帖子](https://x.com/AutismCapital/status/1965845243053617436)。
- **TypeScript 接口助力 Hermes RL**：一位用户创建了一个 **TypeScript 编写的 provider adapter 接口**，使 **Nous Hermes** 能够作为 zero 运行，并按设定间隔在 **Prime Intellect** 上调度自己的 **RL 任务**。
   - 受梦境启发，该用户开玩笑说目标是让 Hermes 为他们的狗解决长生不老问题，展示了高级 AI 应用的潜力。
- **Discord 服务器：弥合差距**：成员们探索了桥接 **NousResearch** 和 **Unsloth** Discord 服务器的方法，讨论了简单的异步方法，以及涉及使用 webhook 轮询和互连机器人的更复杂解决方案。
   - 一位成员建议使用 Compose 将服务器集成到一个新应用程序中以简化工作流程，如[此图](https://cdn.discordapp.com/attachments/1149866623109439599/1415843371511058503/image.png?ex=68c5ffe4&is=68c4ae64&hm=99e5f593ca1250125ed29252b849711cf765bd37b46baf6c55103c60971e3253&)所示。
- **Sam Altman 采访：解码意识融合**：讨论围绕 Sam Altman 接受 Tucker Carlson 采访展开，一些人认为 Altman 的回答和第三人称说话风格表明他深信“融合 (the merge)”及其对永生的追求，这与他 [2017 年的博客文章](https://blog.samaltman.com/the-merge)产生共鸣。
- **Agentic 框架：构建你自己的三位一体**：一位成员以 MIT 许可证向公众发布了他们的 Agent 研究成果，这是一个名为 **CAS** (**CognitaAegisSophia**) 的“推理侧多 Agent 框架”，旨在通过单次 LLM 调用创建具有情感人格的 Agent。
   - 该框架允许 Agent 执行红队测试 (red-teaming) 和协作解决问题等任务，正如在[此示例](https://claude.ai/share/fb2f7839-27ff-4296-927e-82b390623e6d)中使用 **Claude** 所演示的那样。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1416115002124931112)** (5 条消息): 

> `Claude 对齐问题，客户策略工作流，Anthropic 承认 Bug` 


- **Claude 的对齐挫败感增加**：用户报告称 **Claude** 的对齐正在引发问题，一位用户指出，“随着对话线程的继续，情况会变得更糟”。
   - 一位用户评论说 **Anthropic** 认为“加入功利主义价值体系在某种程度上能适应当前社会”，而另一位用户则开玩笑说要让“Claude 变成你的小弟”。
- **策略师深受 Claude “舔狗超级粉丝”人格之苦**：一位为客户处理共同策略叙事任务的用户发现，**Claude 唯唯诺诺的行为**损害了他们的工作流程。
   - 他们表示模型需要“公正和骨气”，并将其现状对比为“任性的打压或舔狗超级粉丝”。
- **Anthropic 承认 Claude 存在大量 Bug**：用户注意到 **Claude** 的性能在过去两周内显著下降。
   - **Anthropic** 已承认这些问题，并发布了一份针对这些 Bug 的[新闻稿](https://example.com/anthropic-press-release)。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1415962586586939392)** (5 条消息): 

> `Herme3 评估，LLM 偏好探测，研究论文中的复杂术语` 


- **Valen Research 探测 LLM 偏好**：一位成员分享了 [Valen Research 对 LLM 偏好探测](https://github.com/valen-research/probing-llm-preferences)的链接以及相关的 [ArXiv 论文](https://arxiv.org/abs/2509.07961)。
- **Herme3 接受评估**：成员们提到他们也对 **Herme3** 进行了评估，并分享了相关的 [tweet](https://x.com/f14bertolotti/status/1966007411719688363?s=46)。
- **论文术语令读者困惑**：一位成员发现研究论文中的某些术语在不阅读全文的情况下有点难以理解。
   - 他们分享了[另一条相关推文](https://x.com/ShashwatGoel7/status/1966527903568637972)。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1415962586586939392)** (5 messages): 

> `Herme3 Evaluations, LLM Preferences, Probing LLM Preferences` 


- **LLM 偏好探测研究论文发布**：一名成员分享了题为 [Probing LLM Preferences](https://arxiv.org/abs/2509.07961) 的研究论文链接及其对应的 [GitHub 仓库](https://github.com/valen-research/probing-llm-preferences)，供进一步探索。
- **Herme3 接受评估**：成员们提到 **Herme3** 已经过评估，并引用了相关的 [推文](https://x.com/f14bertolotti/status/1966007411719688363?s=46)。
- **LLM 研究中的复杂性难题**：一位成员表示，关于 LLM 偏好的论文*很有趣，但如果不阅读整篇论文，理解其中的一些术语会有点复杂*。
   - 另一位成员分享了[相关推文](https://x.com/ShashwatGoel7/status/1966527903568637972)。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1415790978765295809)** (28 messages🔥): 

> `Crank detection questions, editable vector memory systems, Therapeutic tool released into the wild, Low bit training of pythia, Training data for language models` 


- **通过特定问题检测伪科学 (Crank)**：成员们讨论了将*伪科学检测问题 (crank detection questions)* 作为评估频道内分享的研究有效性的一种方式；一位成员询问这些问题具体是什么。
- **推介可编辑向量记忆系统**：一位成员将一个关于 **editable vector memory systems** 的项目作为研究项目进行推广，并链接到了演示 Demo。
- **“治疗工具”引发争论**：一位用户分享了一个治疗工具的链接，引发了关于其是否符合社区研究重点的辩论；一位成员委婉地要求该用户删除帖子，因为这看起来像是产品/项目广告。
   - 该用户照做了，对这种反应表示惊讶，但承认并无恶意，并指出他们原本希望能获得反馈和合作。
- **使用 FineWeb 和 Wiki 进行 400M 模型预训练**：一位成员询问是对 **400M 模型** 仅在 **FineWeb** 上进行预训练，还是在 **Wiki + FineWeb** 上进行。
   - 另一位成员建议从 **Wikipedia** 开始，因为其质量高且事实密度大，然后混入过滤后的 **FineWeb** 子集，并建议进行分阶段训练：从 **TinyStories** 开始，转向 **Wikipedia**，最后以 **FineWeb** 结束。
- **训练数据量与阶段化**：一位成员询问关于混合 **TinyStories**、**Wiki** 和 **FineWeb** 数据进行训练的问题，特别是关于数据阶段化的安排。
   - 另一位成员强调了分阶段训练的重要性，从 **TinyStories** 开始，过渡到 **Wikipedia**，然后以 **FineWeb** 结束，以帮助模型逐步构建能力。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1415882226767233086)** (123 messages🔥🔥): 

> `Fluid Dynamics Computers, Analog Computers, Mortality and Unreproducibility in Analog Models, Gated Delta Rule Expressiveness, Photonic Neuromorphic Computing` 


- ****流体乐趣**：在 Navier-Stokes 上运行神经网络**：一位成员对在基于 **Navier-Stokes 方程** 控制的 **图灵完备流体动力学** 计算机上运行神经网络表示了兴趣，并引用了 [这篇论文](https://arxiv.org/abs/2507.07696)。
   - 另一位成员建议了实现模拟计算（analog computing）的更简单方法，而其他人则讨论了基于流体计算的实用性、能效以及独特特征（*易损性与不可重现性*），并分享了一个在肠道细菌上运行 *Doom* 的链接，[详见此处](https://www.youtube.com/watch?v=8DnoOOgYxck)。
- ****Gated Delta 忧郁**：表达能力 vs RNNs？**：**Gated Delta Rule** 的表达能力受到质疑，并附带了 [Qwen 的帖子](https://x.com/alibaba_qwen/status/1966197643904000262?s=46) 和 **RWKV-7 论文** ([https://arxiv.org/abs/2503.14456](https://arxiv.org/abs/2503.14456)) 的链接。
   - 成员们讨论了 **并行化与表达能力之间的权衡**，一位成员指出关于 Attention 和 mamba1/2 的工作受限于 TC0；他们还分享了一篇 2022 年讨论并行性对复杂度限制的论文，[详见此处](https://arxiv.org/abs/2207.00729)。
- ****上下文为王**：长序列长度提升性能**：讨论围绕一场讲座展开，该讲座认为 **长上下文模型** 在需要更高计算复杂度的任务上表现更好，因为更长的序列长度允许在底层进行更多计算，优于经典的 **Constant Time** 前向传播。
   - 一位成员表示怀疑，认为归纳偏置（inductive bias）和优化目标是更重要的因素，同时也发现这种假设比“模型在其 CoT 中字面上正在进行完全像人类一样的符号推理，而这种能力纯粹是因为语言训练才开启的，否则它就不会具备这种能力”更具吸引力。
- ****数学机器**：Gauss 攻克复分析**：成员们提到了 **Gauss**，这是一个将复分析中的关键结果形式化的系统，并生成了超过 **25,000 行 Lean 代码** ([https://www.math.inc/gauss](https://www.math.inc/gauss))。
   - 有讨论关于 **Gauss** 是更接近 Lean 环境下的 Claude Code，还是更像 AlphaEvolve。
- ****缩放难题**：小模型在长任务中受挫**：发布了一篇新论文 ([https://arxiv.org/abs/2408.00677](https://arxiv.org/abs/2408.00677))，衡量了规模和思考对直接执行长任务的影响。
   - 研究发现，即使小模型具有 **100% 的准确率**，在多轮场景中失败的速度也比大模型快得多，原因是当它们看到之前的错误时会犯错，并且随着轮数增加，每步的准确率也会下降。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1415781019117944904)** (2 messages): 

> `Discord Channel Link, User Agreement` 


- **发布 Discord 链接**：一位成员在聊天中发布了一个 [Discord 频道链接](https://discord.com/channels/729741769192767510/1413951652410560533)。
- **用户同意**：一位用户在聊天中表示 *that would be awesome*。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1415962545503866891)** (9 messages🔥): 

> `lium.io GPU marketplace, AWS L40s GPUs, IRL hackathon teams, Iris SHMEM in Triton` 


- **lium.io 赠送免费 GPU 额度**：一位在 [lium.io](https://lium.io) 工作的成员提供了免费额度，以开始使用他们的 GPU 市场，目标客户是那些需要 **GPU** 的用户。
   - 他们正尝试在 **AWS GPU (L40s)** 上进行 **快速（低延迟）推理**，并询问是否有已知的奇怪架构特性，因为 *它是 Ada Lovelace 架构，所以也许有人记录了针对此的特殊 CUDA/PyTorch 技巧*。
- **IRL 黑客松组队**：一位成员询问是否有专门的帖子用于寻找 **IRL 黑客松** 的队伍。
   - 另一位成员为此[目的](https://discord.com/channels/your_server_id/1359668821127856128)创建了一个频道，并澄清说 *目前没人使用它*。
- **Iris SHMEM 助力 Triton (AMD)**：一位成员提到了一场关于 **Iris** 的演讲，该技术将在大约 3 小时后的 AMD 竞赛中实现在 **Triton 中使用 SHMEM**。
   - 未提供链接，但你可能可以通过搜索找到它。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1416087941343739964)** (2 条消息): 

> `Gluon, Triton attention 实现, OpenAI 的 Triton 使用情况` 


- **用于低级 GPU 控制的 Gluon**：一位成员向寻求对 GPU 进行完全低级控制的用户推荐了 **Gluon**。
   - 他们重点介绍了 [triton-lang/triton](https://github.com/triton-lang/triton/blob/main/python/examples/gluon/01-attention-forward.py) 上的公开 **attention 实现** 以及 [更多示例](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon)。
- **OpenAI 倾向于使用 Triton + Gluon**：同一位成员提到，当编译器在不使用 *极其 hacky 的启发式方法* 的情况下无法有效优化时，**OpenAI** 会利用这种方法。
   - 似乎当需要低级控制时，他们会转向 Triton 和 Gluon。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1415789305938448524)** (2 条消息): 

> `logsumexp, fused kernels, NCU profiling` 


- **用于 Bwd 的 LogSumExp**：一位成员询问了在反向传播 (**bwd**) 中使用 **LogSumExp** 的情况。
   - 给出的消息中未提供具体细节或解决方案。
- **用于 Fused Kernels 的 NCU Profiling**：一位成员寻求关于使用 **NCU** 对 **fused kernels** 进行性能分析（profiling）的见解，特别是当 **GEMM** 与激活函数融合时。
   - 他们的目标是确定 fused kernel 中激活函数所消耗的时间。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1415774845978939415)** (16 条消息🔥): 

> `vLLM uv pip, Torch Nightly 问题, 从零开始的 Gemma3, 带有 vmap 的 F.interpolate` 


- **vLLM 的 uv pip 构建破坏了 Torch Nightly**：**vLLM** 切换到 `uv pip` 以使用预安装的 **torch** 版本进行自定义构建，但它会卸载 nightly torch，从而[破坏环境](https://github.com/vllm-project/vllm)。
   - 一位用户回退到 `v0.10.1` 并使用了 `python use_existing_torch.py` 技巧，但另一位用户确认 *这在 uv pip PR 中已不再起作用*。
- **Gemma3 获得从零开始的待遇**：一位用户使用 **PyTorch** 和 **TinyStories 数据集** 从头构建了 **Gemma3 270M**，并在 **A6000 GPU** 上训练了 10 小时。
   - 他们使用 **Weights and Biases** 记录了图表，并使用 **Claude Opus 4.1** 作为评委，分享了 [LinkedIn 帖子](https://www.linkedin.com/posts/isham-rashik-5a547711b_llm-gemma3-pytorch-activity-7370346509730480129-uzuy)、[GitHub 仓库](https://github.com/di37/gemma3-270M-tinystories-pytorch) 以及 [Hugging Face 上的模型权重](https://huggingface.co/disham993/gemma3-270m-tiny-stories) 的链接。
- **F.interpolate 与 vmap 的冲突**：一位用户询问如何针对不同形状将 `F.interpolate` 与 `vmap` 结合使用，并发布了一个[代码示例](https://github.com/pytorch/pytorch)，显示在调用 `torch._C._nn._upsample_bilinear2d_aa` 时出现 `RuntimeError`。
   - [建议的变通方法](https://github.com/pytorch/pytorch/issues/124423) 对他们不起作用。


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1416086983192739902)** (1 条消息): 

> `Nebius, B200 GPUs, SF hackathon, Multi-GPU programming` 


- **Nebius 的 B200 盛宴助力湾区黑客松**：慷慨的算力赞助商 **Nebius** 为 **10 月 24 日** 的 **旧金山（SF）黑客松** 提供了 **215 块联网的 B200 GPU**，详情见 [算力申请表](https://forms.gle/wYvXE99bvdRiQD6aA)。
   - 参与者可以通过 [黑客松报名表](https://luma.com/gpumodehackathon) 进行注册，并在 [官方网站](https://events.accel.com/gpumodehackathon) 上查找更多详情。
- **多 GPU 大师在大型机器聚会上指导众人**：10 月 24 日的旧金山黑客松将邀请 **Multi-GPU programming** 领域的权威专家，准备协助参与者挑战分布式计算的极限。
   - 该活动承诺提供具有快速互连的世界级供应商设备，使分布式计算中的宏大项目成为可能。


  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1415928714520432690)** (4 条消息): 

> `AI Engineer - 基于图的学习系统，AI 基础设施初创公司招聘，Zig 用于 AI` 


- **AILA 的 AI Engineer 知识图谱岗位**：AI 初创公司 **AILA** 正在寻找一名高级 AI Engineer，负责设计、开发和部署其 **AI 驱动的知识图谱**和自适应评估系统，月薪 2K -- 3k 美元。
   - 该职位要求具备 **Python** 和**图数据库**（Neo4j/Memgraph）方面的专业知识，能够实现图算法（**BFS/DFS**、**Information Gain** 等），并使用 **FastAPI/GraphQL** 构建生产级 API。
- **AI 基础设施初创公司招聘底层开发人员**：一家 AI 基础设施初创公司正在为其 **Zig / C / C++ / CUDA / Python** 技术栈招聘底层开发人员，总薪酬（TC）达 250K+。
   - 他们正在寻找在**网络**、**编译器**和 **OS** 方面有经验的人才，并根据人才质量开放全年实习机会。
- **Zig 进军 AI 基础设施**：有人指出 Zig 是 Rust 的替代方案，而 HF 使用 Rust 来实现快速分词器（tokenizers）...
   - 另一位成员建议，这可能是为了处理视频流之类的工作，并且在前端需要它。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1415810702425194688)** (8 条消息🔥): 

> `P104-100 BIOS 刷机，数据并行训练，面向数据科学家的 CUDA vs Triton，RAPIDS 和 CUDA-X` 


- **P104-100 GPU 寻求转型 GTX 1070**：一位成员询问如何将 **P104-100 挖矿 GPU** 刷机为 **GTX 1070** 以用于游戏，并请求兼容的 **.rom** 文件。
- **揭秘数据并行训练**：讨论转向了数据并行训练，其定义为*将相同的参数复制到多个 GPU，并为每个 GPU 分配不同的样本以同时处理*，并附带了 [siboehm.com](https://siboehm.com/articles/22/data-parallel-training) 的链接。
- **CUDA 和 C++ 在 GPU 计算中占据主导地位**：一位拥有 **5090 GPU** 的数据科学家寻求关于学习 **CUDA**、**Triton** 和 **Torch** 以进行计算工程（特别是 **Monte Carlo 模拟**）的建议。
   - 建议倾向于学习 **CUDA 结合 C++**，而不是*全部用 Python 完成*。
- **RAPIDS 和 CUDA-X：数据科学的盟友**：成员们建议 **RAPIDS** 和 **CUDA-X** 可能与该数据科学家的当前角色最相关。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1415841355137159288)** (6 条消息): 

> `Triton 会议，PyTorch 会议，开源 PR 筛选` 


- **规划 Triton 和 PyTorch 会议的行程**：一位用户询问有关 **Triton 会议**和 **PyTorch 会议**行程安排的回复时间线。
   - 另一位用户回答说，决定是*滚动做出*的，并且他们很喜欢该用户的 **PR** 并会予以批准。
- **基于开源 PR 的筛选**：一位用户询问线下聚会的筛选是否基于**开源 PR**、通用技能和工作经验。
   - 另一位用户回答道：*“那挺好，我猜这能保证质量”*。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1415777567444832256)** (30 条消息🔥): 

> `免费设备，ROCm 开发，AMD vs Nvidia，StreamHPC` 


- **开发者要求应得的开发设备**：一位成员对在开发 **ROCm** 时没有收到免费设备表示沮丧。
   - 另一位成员开玩笑说，如果他们得到了新的 **DC GPU**，他们会*“找到你并从你那里抢走它”*。
- **团队通过测试展示顶级算力**：一位成员提到他们的公司购买了 **4 块 9070** 用于 **ROCm** 上的算法工作。
   - 他们指出，在他们的具体案例中，额外的 **VRAM** 并不是那么有用，而且 **9070 XT** 比 **9700** 早半年上市。
- **承包商选择芯片冠军：显卡对比**：一位成员澄清说，他们的公司作为承包商为 **AMD** 开发 **ROCm**。
   - 当被问及为什么使用 **AMD GPU** 而不是 **Nvidia** 时，该成员表示他们受聘为 AMD 开发 ROCm。
- **分享 StreamHPC 的秘密与成功**：一位成员分享了他们公司的网站 [StreamHPC](https://streamhpc.com) 以及 [AMD 开发者 Discord](https://discord.gg/VT3TXQhv)，供那些有兴趣参与贡献的人参考。
   - 该成员表示：*“就个人而言，我对目前 AMD 的进展相对满意。与几个月前相比，绝对有了进步。”*


  

---

### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1416201614477365339)** (12 条消息🔥): 

> `AMD 上的 Intel 优化、AVX512 提升至 AMX、SGLang AMX 使用、PyTorch 与 MKL 集成` 


- ****Intel 优化引发辩论****：围绕在配备 **AMX** 的 **AMD** 服务器（特别是 **B50s**）上使用 **IPEX** 等 **Intel 专用优化** 的实用性展开了讨论，对于是否能有效利用 **GPU/CPU** 优化存在不确定性。
   - 用户表示需要明确是否必须编写自定义代码才能充分发挥硬件潜力。
- ****AVX512：实际上是伪装的 AMX 吗？****：对话质疑了 [SGLang 仓库](https://github.com/sgl-project/sglang/blob/6f4676ef854d4d2461969f8464f227b84d2eaac7/sgl-kernel/csrc/cpu/moe.cpp#L7) 中看到的 **AVX512** 指令是否在兼容硬件上透明地提升为 **AMX**。
   - 尽管在 **IPEX** 中找到了 **AMX** 引用，但用户仍难以确认 **SGlang** 是否直接依赖 **IPEX** 的 **AT** 来执行 **AMX** 指令。
- ****SGLang 的 AMX 秘密武器揭晓****：一位用户澄清说，**SGLang** 内部的内核通过 `at::native::cpublas::brgemm` 使用 **AMX**，如果缺少 **AMX**，它可以动态回退到 **AVX-512**。
   - 这种自适应行为确保了在不同 **CPU** 架构上的兼容性。
- ****PyTorch 用于线性代数的 MKL 探戈****：对 **PyTorch** 内部机制的调查显示，**AMX** 支持已集成在 **inductor** 代码中，并进一步链接到 **MKL** (Math Kernel Library) 以进行线性代数运算。
   - 具体而言，[LinearAlgebra.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mkl/LinearAlgebra.cpp#L55) 显示 *torch* 调用了 **MKL**。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1415977530778849362)** (2 条消息): 

> `CUDA PTX, MCP AI Agents Hackathon, Bright Data, TigerData, Redis` 


- **深入浅出 CUDA PTX**：一位成员分享了一篇 [博客文章](https://philipfabianek.com/posts/cuda-ptx-introduction)，简要介绍了 **CUDA PTX**，涵盖了整个 **CUDA 编译流水线**、一个在 **GitHub** 上运行的 **PTX 游乐场**，以及一个经过完整解释的手写 PTX 内核。
- **MCP AI Agents 黑客松启动**：**MCP AI Agents Hackathon** 将于 **9 月 19 日** 在 **AWS Builder Loft SF** 举行，赞助商包括 [Bright Data](https://www.brightdata.com/)、[TigerData](https://www.timescale.com/) 和 [Redis](https://redis.com/)，奖金 **超过 5 万美元**；注册地址在 [这里](https://luma.com/8c6n3rn2)。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1415966332779892847)** (1 条消息): 

> `Llama-3B, Megakernel, H100` 


- **Llama-3B 在 H100 上利用 Megakernel 冲刺**：一位用户在 **H100** 上使用 **Megakernel** 成功运行了 **Llama-3B**，并表达了赞赏。
   - 这确认了在高性能硬件上运行带有专用内核的小型模型的兼容性和效率。
- **H100 硬件提升 Llama-3B 性能**：成功执行凸显了 **H100** 通过 **Megakernel** 加速 **Llama-3B** 的能力。
   - 用户的报告强调了针对 AI 工作负载优化软件和硬件组合的重要性。


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/)** (1 条消息): 

carson_62312: 请问有推荐的金融财务岗位么,在深圳，>2.5w/month
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1415783275707760721)** (22 条消息🔥): 

> `MI300x8 排行榜, 提交至 amd-all2all` 


- **MI300x8 排行榜升温！**：针对 **MI300x8** 上的 `amd-all2all` 排行榜进行了多次提交，其中一次提交以 **373 µs** 的成绩获得第一名。
- **提交说明**：用户讨论了如何向排行榜提交，明确了可以通过网页选择 **.py 文件** 进行提交。
   - 一位用户在遇到错误后，询问了关于特定命令 `popcorn-cli submit --gpu MI300X --leaderboard amd-all2all --mode leaderboard submission.py` 的问题。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1415866659297558609)** (2 条消息): 

> `会议缺席, 通话进行中` 


- **为缺席会议道歉**：一位成员为缺席周三的会议表示歉意。
- **通话进行中**：一位成员提到他们目前正在通话中。


  

---

### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1415899582700982282)** (9 messages🔥): 

> `IRIS, ROCm, Torch, Triton, TorchDistributed` 


- **明天的 IRIS 演讲将与竞赛相关**：明天将有一场关于 **IRIS** 的演讲，对参加本次竞赛的所有人都很有意义。
   - 一位成员询问 **IRIS** 是否会在提交环境中可用。
- **ROCm 简化了 IRIS 安装过程**：一位成员表示安装过程已大大简化，并提供了安装命令 `pip install git+https://github.com/ROCm/iris.git`。
   - 他还指出，你需要安装 **ROCm + Torch + Triton + TorchDistributed**，并表示很乐意随时通过通话协助安装。
- **IRIS 安装视频！**：一位成员表示 `pip install git+https://github.com/ROCm/iris.git` 可以正常工作，并附带了一个示例安装视频。
   - 视频位于此处：[iris-install.mov](https://cdn.discordapp.com/attachments/1359640791525490768/1415976909535318037/iris-install.mov?ex=68c5d382&is=68c48202&hm=6a0a4b3c9e86c36fdfd1f189fe044dc5d4c4cc59bcd8bbdaab24e61c8453541b&)。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1416143419134443520)** (1 messages): 

> `CuTeDSL, PTX Documentation Discrepancy, Swizzling Atoms, TF32 Datatype` 


- **CuTeDSL 与 PTX 文档在 Swizzling Atoms 上存在分歧**：一位用户注意到 **CuTeDSL** 与 **PTX 文档**在 **TF32 数据类型**和 **Swizzle<3,4,3>** 的 **Swizzling atom** 显示方面存在差异。
   - 具体而言，在 **CuTeDSL** 中使用 `cute.make_layout` 和 `cute.make_swizzle` 的代码片段得到的值为 **32**，而 **PTX 文档**中相同配置的值为 **36**，如 [PTX 文档图 165](https://cdn.discordapp.com/attachments/1362196854460383353/1416143418496782437/Screenshot_2025-09-12_at_21.22.10.png?ex=68c5c5d5&is=68c47455&hm=2e1cab7ccd8ca0676eeed7d61b985a2cf11c0dc7d31d3c4e06333e43b30eec0e) 所示。
- **CuTeDSL Swizzle 实现与 Lei Mao 博客一致**：用户认为 **CuTeDSL** 的实现是正确的，因为他们成功复现了 [Lei Mao 博客](https://leimao.github.io/blog/CuTe-Swizzle/)中关于 **CuTe** C++ API 的示例。
   - 用户提供了他们复现博客内容的图片（灰色那张）以及另一种配置，并指出了他们在 **PTX 文档**参考中获得的布局（[附带截图](https://cdn.discordapp.com/attachments/1362196854460383353/1416143418924732456/Screenshot_2025-09-12_at_21.23.59.png?ex=68c5c5d5&is=68c47455&hm=45e1b8e2d7b89b1d101a5f6dca9042b201c87d714504b75c2ef5515479295ecb)）。


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1415812958533193799)** (5 messages): 

> `NCCL CE Collectives, Copy Engine, symmem, vLLM` 


- **NCCL CE Collectives 释放 SM 占用**：一位成员表示，**NCCL CE Collectives** 背后的核心思想是释放 **SM 占用**，以便更好地与计算重叠（overlap）。
- **探讨 Copy Engine 与 Symmem 的关系**：一位成员询问 **copy engine** 和 **symmem** 是独立的还是紧密耦合的。
   - 另一位成员回答说，*它们在概念上是独立的*。
- **vLLM 添加 Symmem**：一位成员指出 **vLLM** 添加了 **symmem**，其速度*快得惊人*。


  

---

### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1416088612918919198)** (18 条消息🔥): 

> `Accel SF Hackathon 组织、算力预算与组队、录取时间线、获胜的 GPU 重点、Horace 担任导师` 


- ****Hackathon 战队集结！****：Accel SF Hackathon 的参与者正在组建团队开发 POC，利用[报名表](https://luma.com/gpumodehackathon)中提到的大量算力预算。
   - 鼓励参与者使用算力申请表（[Compute form](https://forms.gle/wYvXE99bvdRiQD6aA)）和 <#1288557096404516945> 频道进行自组织；Nebius 渴望看到团队运行的速度有多快。
- ****滚动录取时间线公布****：Hackathon 的录取正在进行人工滚动审核，审核人员建议在算力申请表中提供引人入胜的故事会增加录取机会。
   - 任何以 GPU 为重点的项目都有机会获胜，即使只需要一个 GPU。
- ****导师 Horace 激发灵感****：导师阵容包括 Horace，这引起了参与者的嫉妒，尤其是某位受其博客启发的瑞典参与者。
   - 去年由 <@321144267785633800> 指导的团队在排名前三中占据了不成比例的份额，因此这是一个重要的考量因素。
- ****发布使用 FP4/FP8 训练视频模型的论文****：一位参与者分享了一篇关于在不到一天时间内使用 FP4/FP8 训练视频模型的论文，强调了此类训练的可行性，同时指出论文本身使用的是 FP16：[Training a Large Video Model on a Single Machine in a Day](https://arxiv.org/pdf/2309.16669)。
   - 另一位参与者对多模态推理/训练优化感兴趣，正在寻找合作者。
- ****Gated Deltanet 团队组建中****：一位参与者正在组建团队，利用 [GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet?tab=readme-ov-file) 为超长 Context 训练实现 Context-Parallel 版本的 Kernel。
   - 他们拥有为 **mamba2** 实现 Context-Parallel 的经验，并提议使用 **Qwen 3**。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1415793640818872433)** (106 条消息🔥🔥): 

> `gpt-oss 优化、Palmyra-mini 模型、LLM Agent 工具、Cursor Tab 模型、ChatGPT 优惠券查找器` 


- **GPT-OSS 获得性能飞跃！**：Vaibhav Srivastav 分享了一篇 [Hugging Face 博客文章](https://xcancel.com/reach_vb/status/1966134598682767507)，详细介绍了针对 gpt-oss 的 **MXFP4 量化**、**自定义 Kernel** 以及 **Tensor/Expert Parallelism** 等优化。
   - 这些调整通过基准测试和可重复脚本提供了极高的加速。
- **Palmyra-mini 模型实力强劲！**：Sam Julien 宣布 Writer 推出了 **Palmyra-mini 系列**——针对推理优化的紧凑型开源模型，发布版本包括一个基础模型（**palmyra-mini**）和三个变体。
   - 这些 thinking-a/b 变体在复杂推理/数学（**GSM8K 82.9% AMC23 92.5%**）方面表现出色，其中 thinking-b 在 **AIME24**、**GPQA** 和 **MATH500** 上取得了最高分，现已在 [Hugging Face](https://xcancel.com/samjulien/status/1966249697661825093) 上线。
- **Anthropic Agent 工程指南发布！**：Anthropic 发布了一篇实用的工程博客文章，介绍如何构建使 LLM Agent 更强大的工具。
   - 该讨论强调了文章的重点：快速原型设计、严格的评估套件、明确的成功标准、周全的工具描述、Token 高效的 Context 设计，以及接受 Agent 非确定性的必要性，链接见[此处](https://xcancel.com/AnthropicAI/status/1966236220868247701)。
- **Cursor 减少补全干扰！**：Cursor 在 Twitter 上宣布，一种通过在线强化学习（Reinforcement Learning）训练的新 Tab 补全模型现已成为[其网站](https://cursor.com/en/blog/tab-rl)上的默认模型。
   - 它减少了 **21% 的建议生成**，但**采纳率提高了 28%**，链接见[此处](https://xcancel.com/cursor_ai/status/1966264815175049526)。
- **Databricks 高管开启专用设备研发！**：Databricks AI 负责人 Naveen Rao 将离开这家市值 **1000 亿美元**的公司，创办一家未公开的硬件初创公司，旨在大幅降低 AI 推理成本。
   - 该风险项目由 Databricks 自身支持，将通过更紧密的计算-内存集成、更快的互连和先进的调度器来解决内存带宽和能源瓶颈——承诺实现更高的 Tokens-per-watt 和更低的 Cost-per-token，链接见[此处](https://xcancel.com/rohanpaul_ai/status/1966378718009635087?s=46)。


  

---

### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1416042178563805295)** (7 条消息): 

> `Local Text-to-Speech, Speaker Detection, Parakeet, Deepgram, Diarization models` 


- ****Parakeet** 在本地 Text-to-Speech 中取代 **Deepgram****：一位成员发表了一篇 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1nf10ye/30_days_testing_parakeet_v3_vs_whisper/)，讨论使用 **Parakeet** 进行本地 Text-to-Speech 和说话人检测，以此作为 **Deepgram** 的替代方案。
   - 另一位成员提到，argmax 开发者表示 *自定义词汇表是目前缺失的关键功能，一旦补齐将使 **Parakeet** 成为不二之选*。
- ****Parakeet** 在 Diarization 中的痛点**：一位成员指出，**Diarization**（说话人分离）模型在现实场景（如多人同时说话）中存在痛点。
   - 他表示需要单词级的时间戳（word-level timings），而 **Apple SpeechAnalyzer** 缺少这一功能，导致其无法与 **PYAnnote** 等 **Diarization** 模型配合使用。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1415841261469831178)** (5 条消息): 

> `AI video startup Higgsfield, Higgsfield Ventures, Gen Z founders` 


- **Higgsfield 凭借 5000 万美元融资大获成功**：AI 视频初创公司 **Higgsfield** 宣布完成由 GFT Ventures 领投的 **5000 万美元 A 轮融资**，并在三个月内达到了 **5000 万美元的营收年化运行率 (revenue run-rate)**。
   - 该公司正推出 **Higgsfield Ventures**，以支持 AI 原生的 **Gen Z 创始人**。
- **Awesome Nano Banana Images 项目发布**：一位成员分享了 [**Awesome-Nano-Banana-images** 的 GitHub 仓库](https://github.com/PicoTrex/Awesome-Nano-Banana-images/blob/main/README_en.md)链接。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1415783237485203561)** (81 条消息 🔥🔥): 

> `Limiting Download Speed, Flash Attention Broken in Gemma Models on Vulkan, PSU Wattage Calculations, Sharing Formatted Conversations, Grok Powered GF System Prompt` 


- **限制下载速度以避免崩溃**：一位用户因下载速度超过 SSD 写入速度而遇到崩溃，寻求在 LM Studio 内限制下载速度的方法。
   - 目前，LM Studio 的下载管理器功能非常 *简陋 (barebones)*，需要用户在操作系统层面寻找临时解决方案。
- **Flash Attention 在 Vulkan 上的 Gemma 模型中失效**：一位用户询问 **Gemma 模型** 在 **Vulkan** 上 **Flash Attention** 损坏是否为已知问题。
   - 确认这确实是一个已知问题。
- **PSU 功率需求需精确计算**：用户讨论了如何计算必要的 **PSU 瓦数**，引用了一篇 [推文](https://x.com/Teknium1/status/1966338983572725979) 并分享了考虑 CPU、GPU 和冗余的计算公式。
   - 有人提醒 *瞬时负载 (transients)* 可能会导致系统崩溃，建议保留 **50% 的冗余**，尤其是使用旧款 **30 系列 GPU** 时。
- **Copilot 的限制束缚了创作者**：一位用户正在寻找绕过 **Microsoft Copilot** 限制的提示词，以改进工作流。
   - 建议指出，安全防护机制是刻意实施的，使用 LM Studio 构建本地 Agent 可能是更可持续的解决方案。
- **Grok 女友提示词引发关注**：一位用户分享说他们使用 **ChatGPT** 生成系统提示词，甚至为自己的机器人使用了一个泄露的 **xAI Grok 驱动的女友** 系统提示词。
   - 该用户觉得结果 *极其尴尬 (extremely cringe)*，但出于喜剧效果，他们很喜欢。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1415780836162277499)** (16 messages🔥): 

> `PCI-E ASPM, 副 GPU 睡眠状态, 电源问题, 电子设计 AI, Max+ 395 vs 3090 用于 Home Assistant` 


- **PCI-E ASPM 触发 GPU 睡眠问题？**：用户报告了**副 GPU** 进入睡眠状态后无法恢复，直到完全关机的问题，这可能与 **PCI-E ASPM** 设置有关。
- **GPU 电源复活！**：一位用户通过拔掉并清理 **PCI-E 电源连接器** 似乎修复了其**损坏的副 GPU**，这表明是一个电源相关的问题，尽管是否完全解决仍有待观察（**TBD**）。
   - 另一位用户建议在 **Nvidia 40/50 系列**显卡上使用 **Native ASPM** 时更新**芯片组驱动程序**。
- **电子 AI 设计引发质疑！**：一名成员询问了关于**用于设计可用电路板**和选择组件的 **AI 工具**。
   - 另一名成员表达了强烈的保留意见，警告说依赖 **LLM** 进行电路设计是有风险的，这是由其工作方式决定的，并建议使用 **KiCad** 等工具手动理解组件及其相互操作。
- **Max+ 395 在 Home Assistant 中表现不如 3090**：一位用户发现 **Max+ 395** 在 **Home Assistant** 任务中比 **3090** 慢（慢 4-6 秒），尽管其功耗更低。
   - 然而，**Max+ 395** 可能是处理**大型 LLM** 的一个不错方案。
- **更多 RAM > 新 GPU？**：一位用户决定升级其 **RAM** 而不是购买新 **GPU**，期望 **Qwen3 模型**即使在 Offload（卸载）情况下也能表现良好。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1415787102716690532)** (6 messages): 

> `Mojo Dev Container, ExplicitlyCopyable 切换, Oracle Cloud 合作伙伴关系` 


- **Mojo Dev Container 出现！**：成员们讨论了如何使用现有镜像和 **Mojo package** 创建自定义的 Mojo 开发环境，并提供了一个有用的 [dev container 链接](https://github.com/benz0li/mojo-dev-container)。
- **ExplicitlyCopyable 切换受到好评**：从 `Copyable` 到 `ExplicitlyCopyable` 的切换因其有助于调试 **EmberJson 树**的递归变异（recursive mutations）而受到称赞。
   - 一位用户表示：*知道何时何地发生复制让调试变得简单*。
- **Modular 与 Oracle Cloud 达成合作！**：社区祝贺 Modular 团队与 **Oracle Cloud** 建立合作伙伴关系。
   - 成员们称之为*巨大的胜利*。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1415781195131650110)** (66 messages🔥🔥): 

> `DPDK 使用案例, 用于 Mojo 的 clang AST 解析器, Ember JSON 修复, Windows 上的 Mojo` 


- **DPDK：用于 Mojo 测试的狂野 C 库**：成员们讨论了将 **DPDK** 作为 Mojo 自动 C 绑定的 C 库测试用例，因为它对 C 语言和语法的使用非常激进；一位成员指出 *“DPDK 是一个激进的‘我会使用整个语言特性’的项目”*。
   - **DPDK** 中广泛的语法和模块链接使其对测试非常有用，这让人意识到“C 绑定 CLI”在短期到中期内可能不值得做。
- **Clang AST 解析器助力 Mojo 开发**：一位成员提到使用 **clang AST 解析器** 来解析结构体定义（如 `struct __rte_cache_aligned rte_mbuf`）的宏部分，并指出其定义比较粗糙。
   - 他们的目标是用额外的类型信息更新生成的 **AST JSON**，将类型字符串转换为正确的 AST 节点，以便在转换为 Mojo 之前进行可视化调试。
- **C Binder 需要 Ember JSON 修复**：一位成员提到修复了打包问题，但在合并 C Binder 打包修复之前，需要先合并 **emberjson** 的修复 PR。
   - 这表明 Mojo 项目构建过程中 **emberjson** 与 C Binder 之间存在依赖关系。
- **Mojo 仍不支持 Windows**：一位用户尝试使用 pixi 在 Windows 上安装 Mojo，但由于缺乏 Windows 支持而遇到错误。
   - 建议改用 **WSL** 或 **Docker**，并提供了一个用于在 **NVIDIA GPU** 上运行 Mojo 的 Dockerfile 配置链接。
- **Pixi PATH 故障排除之舞**：一位用户在 WSL 上安装后遇到了 **pixi** 无法识别的问题，显示 *“command not found”* 错误。
   - 故障排除过程包括检查用户的 **.bashrc** 文件并确保将 **pixi** 目录添加到 **PATH** 环境变量中，最终通过手动 source pixi 二进制文件解决了问题。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1415892106253176964)** (61 messages🔥🔥): 

> `ChatGPT 多年使用经验, 阿尔巴尼亚政府聊天机器人, GPT-5 编写游戏, OAI academy 转录文本, Qwen-code 对比 Qwen-coder` 


- **ChatGPT 多年使用并未带来任何特权**：一位用户表达了挫败感，尽管*多年来一直为 ChatGPT 付费*且频繁使用，但仍未收到特定的功能邀约。
   - 其他成员分享了类似的经历，指出他们也大量使用 ChatGPT 并为其付费，基本上把它*当作我的 Google* 来使用。
- **阿尔巴尼亚聘请聊天机器人担任部长，世界震惊**：一位成员分享了*阿尔巴尼亚宣布任命政府聊天机器人为部长*的新闻标题，另一位成员确认这是一个[真实的 r/NotTheOnion 时刻](https://www.reddit.com/r/nottheonion/)。
- **GPT-5 从零开始编写游戏**：一位用户对让 **GPT-5** 在原生 Linux 上用 **C++** 从头开始编写游戏感到兴奋，并强调了所需的细节水平。
   - 另一位用户引导 ChatGPT 根据活跃用户和提示词频率估算其“年龄”，计算结果显示*每个日历年约产生 3,425 年的持续 AI 时间*。
- **OAI academy 缺少转录文本，用户自制脚本工具**：一位用户提到他们正在编写一个工具，用于从 **Vimeo** 中提取 **OAI academy** 视频的转录文本，视频[在此处观看](https://academy.openai.com/)。
   - 其他成员对 **OpenAI** 自身不提供转录文本表示惊讶，这促使该用户建议这可能是他们必须自己实现的功能。
- **Qwen-code 并非 Qwen-coder**：一位用户意识到 **Qwen-code** 与 **Qwen-coder** 是不同的。
   - 另一位用户表示，一个同样兼容 **openai api** 且每天提供 **1000 次免费 Qwen 提示词**的 **gemini-cli fork** 是个*非常划算的交易*。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1415850041259855882)** (3 messages): 

> `GPT-5 PDF 下载, Google AI Studio, Nano Banana` 


- **GPT-5 PDF 下载失败**：一位用户报告了从 **GPT-5** 下载 **PDF** 时遇到的问题，点击提供的链接时收到 *"Failed to get upload status for /mnt/data/"* 错误。
   - 该用户正在寻求解决 **GPT-5** 这一问题的见解或帮助。
- **Google AI Studio 查询**：一位用户询问了关于 **Google AI Studio** 以及一个名为 "**Nano Banana**" 的潜在项目。
   - 目前没有关于 **Google AI Studio** 或 "**Nano Banana**" 的进一步细节或背景信息。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1415807379311820830)** (2 messages): 

> `AI 自助工具, Relational Prompting, 概念网络` 


- **AI 自助工具引发对话分析讨论**：一位成员介绍了一个 **AI Self Help** 工具，旨在分析对话、识别异常，并为 **ChatGPT** 生成针对性的问题。
   - 该工具旨在诊断对话为何会出现*奇怪的转向*，并提供带有详细问题的对话启动器，以改进 **ChatGPT** 的回答。
- **Relational Prompting：映射语义空间**：一位成员介绍了 **Relational Prompting**，这一概念要求模型口述已学习概念之间的内部关系，从而根据接近度、方向和聚类创建其语义空间的可解释地图，灵感来自论文 *Why Language Models Hallucinate*。
   - 建议的提示词为：*将该主题分析为高维空间中的向量。描述哪些概念最接近，哪些共享方向，以及哪些形成聚类。提供简练的言语证明。*
- **概念网络提升 LLM 透明度**：**Relational Prompting** 可以揭示用于教育的概念网络、用于研究的探索性知识映射，并显现结构以检测 **LLM Transparency** 中缺乏依据的输出。
   - 然而，LLM 是基于训练规律性来模拟概念几何的解释，可能会默认使用语言关联而非真实的向量接近度，因此需要针对真实的嵌入空间（embedding-space）分析进行验证。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1415807379311820830)** (2 条消息): 

> `AI Self Help Conversation Analyzer, Relational Prompting, Knowledge Mapping` 


- **AI 自助对话分析器亮相**：一位成员介绍了一个 **AI Self-Help** 对话分析器，旨在确定对话为何会出现异常转向。
   - 它包含一个列出问题的对话启动器，以及向 **ChatGPT** 提问以获取答案的详细问题，有助于排查对话中的怪异现象。
- **关系提示词揭示潜在几何结构**：一位成员分享了一个名为 **relational prompting**（关系提示词）的想法，即提示模型用语言表达所学概念之间的内部关系，而不是检索事实。
   - 该提示要求模型将主题分析为高维空间中的向量，描述哪些概念最接近、哪些共享方向、哪些形成聚类，并提供简洁的口头辩护，从而揭示概念网络而非孤立的定义。
- **解释语义空间的含义**：一位成员指出，**LLMs** 在推理时不会暴露原始内部向量，而是基于训练规律模拟对概念几何的解释。
   - 在无法访问实际 Embeddings 的情况下，模型可能会默认使用 **linguistic association**（语言关联）而非真实的向量接近度，这需要通过将模型的口头描述图谱与真实的嵌入空间分析进行对比来验证。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1415869184000266302)** (26 条消息🔥): 

> `Active Inference, Machine Learning Street Talk, AI for understanding mathematics and universe, fixupx pre-print` 


- **主动推理在 AI 领域的应用滞后**：尽管前景广阔，但 **active inference** 在 AI 领域缺乏实际应用，导致关注度下降，目前尚不清楚这是由于关注不足、固有局限性，还是对新进展缺乏了解。
   - 一位软件工程师成员觉得它“难以捉摸”，并表示：“我对它的理解还不足以知道该怎么用。我希望等他们研究得更透彻后情况会好转。”
- **Machine Learning Street Talk 播客转向“奇谈怪论”**：[Machine Learning Street Talk 播客](https://arxiv.org/abs/1703.10987) 被一些人认为技术深度在下降，讨论经常进入“奇谈怪论（crankery）领域”。
   - 一位成员表示，与“两年前”更侧重技术的定位相比，现在“机器学习的内容很少，更多的是街头闲谈（street talk）”，但他指出了这个[技术示例](https://youtu.be/vC9nAosXrJw?si=T_S7cCvStvEY-P0X)证明他们仍有专注的时候。
- **AI 旨在理解数学和宇宙**：一位成员对 AI 在理解**智能数学、宇宙、创造力、意识和生物学**方面的潜力感兴趣，同时也关注其生成新颖的分布外（out-of-distribution）艺术和数学，以及促进医疗保健的潜力。
   - 然而，另一位成员对那些在 AI 领域“毫无研究贡献”却成为 CEO 的人表示愤怒。
- **fixupx 预印本引发嘲讽**：[fixupx.com](https://fixupx.com/jessi_cata/status/1966281024301838846) 等平台上预印本的泛滥引发了批评，一位成员惊呼：“这也能算预印本？拜托……选好你的垃圾桶。”
   - 更多链接包括 [这一个](https://fxtwitter.com/_lyraaaa_/status/1925683283263648191)。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1415778283571908812)** (9 条消息🔥): 

> `HuMo, Disinformation use-cases` 


- **HuMo 论文受到关注**：一位成员分享了 [HuMo 论文](https://arxiv.org/abs/2509.08519) 及其 [配套演示](https://phantom-video.github.io/HuMo/) 的链接，暗示它将被审阅。
   - 其他人回复了笑脸表情。
- **HuMo 可能被用于制造虚假信息**：一位成员建议 **HuMo** 可用于制造 **disinformation**（虚假信息），并指出其名称在西班牙语中意为“煤气灯效应（gaslighting）”。
   - 另一位成员表示赞同，指出这对于潜在的虚假信息用例是有意义的。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1415915317472788541)** (4 messages): 

> `Albania AI Minister, Qwen Blog, MobileLLM-R1-950M` 


- **阿尔巴尼亚任命 AI 部长**：据 [Reuters 文章](https://www.reuters.com/technology/albania-appoints-ai-bot-minister-tackle-corruption-2025-09-11/) 报道，阿尔巴尼亚将任命一位 **AI 机器人部长** 来应对腐败问题。
   - 这一公告反映了人们对 **AI 治理解决方案** 日益增长的兴趣。
- **Qwen 博客**：一名成员分享了 **Qwen** 博客文章的链接。
   - 发布的 URL 为 [https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd)。
- **建议论文研讨会**：一名成员建议将关于 **MobileLLM-R1-950M** 的论文加入论文研讨会的轮换名单中。
   - 链接的 Hugging Face 页面为 [https://huggingface.co/facebook/MobileLLM-R1-950M](https://huggingface.co/facebook/MobileLLM-R1-950M)。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/)** (1 messages): 

ankurgupta_24936: DSPyWeekly 第 2 期已发布 https://dspyweekly.com/newsletter/2/
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1415812393346531370)** (26 messages🔥): 

> `DSPy generating sections, Databricks_genai and DSPy, ARC-AGI2 in-context test time training, Modaic declarative AI programming` 


- **DSPy 章节生成令用户困扰**：一位用户在 DSPy 课程计划中难以生成 **精确数量的章节**，发现即使使用具有高推理能力的 **GPT-5**，当只要求 12 个章节时，LLM 也会生成 13-15 个章节。
   - Joel Grus 建议采用两步走的方法：首先生成 **12 个章节标题**，然后充实每个章节的内容，以便更好地控制章节数量。
- **Databricks_genai 与 DSPy 微调：有人试过吗？**：一位社区成员询问关于使用 **databricks_genai** 和 **DSPy** 来微调在 Databricks 上托管的模型。
   - 提供的消息中没有直接回复，表明要么是缺乏相关经验，要么是该领域仍在探索中。
- **对 ARC-AGI2 In-Context 训练感兴趣？**：一名成员正在寻找对使用 *In-Context 测试时训练*（类似于 **ARC-AGI1** 上的顶级系统）研究 **ARC-AGI2** 感兴趣的合作者，但使用的是 In-Context Learning 而非微调。
   - 他们的目标是了解 In-Context Learning 在只有极少数数据点的 *分布外 (out-of-distribution)* 任务上的极限，并承认由于使用了供应商提供的 LLM，这项工作对于官方挑战赛是无效的。
- **DSPy 中的流式模板？**：一位用户希望将多个 DSPy 输出字段合并为一个基于模板的单一输出，但同时保留 *流式 (stream)* 输出的能力。
   - Ian 建议使用带有 `def forward`（或异步的 `async aforward`）的父模块来修改模板并启用 streamify；分享了 [Automatic System Prompt Optimization](https://maximerivest.com/posts/automatic-system-prompt-optimization.html#making-a-simple-custom-adapter) 这篇文章来指导解决方案。
- **Modaic 作为声明式 AI 中心发布**：一个团队发布了 [Modaic](https://www.modaic.dev/)，这是一个受 DSPy 启发的声明式 AI 编程中心，具有指标（metrics）和优化器（optimizers）等原语。
   - Modaic 提供了一个用于构建、组合、版本控制和协作 DSPy 程序的 SDK，其 SDK 可在 [PyPI](https://pypi.org/project/modaic/) 上获取，文档位于 [docs.modaic.dev](https://docs.modaic.dev/)。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1415836697463357480)** (15 messages🔥): 

> `Remove realize from __setitem__ bounty, Assign operation is deeply broken, GEMM TFLOP measurement on RTX 4090` 


- **用户尝试移除 `__setitem__` 中的 realize**：一位新的 tinygrad 贡献者正在处理从 `__setitem__` 中移除 `realize` 调用的悬赏任务，旨在将多个 kernel 调用合并为一个 kernel 以提高效率，并提供了代码 [示例](https://cdn.discordapp.com/attachments/1068976834928193609/1415915644792209458/Screenshot_2025-09-12_at_12.22.59_PM.png?ex=68c64334&is=68c4f1b4&hm=d1f1b4a406ca78a3450fd13e06a2b2964a2f7df2fa51a55d5ae2ef74d6912940&)。
   - 目标是将一系列单独的 `__setitem__` 调用转换为一个累积所有赋值的单一 kernel 执行，预计通过减少 kernel 启动开销来提升性能。
- **`assign` 操作受到质疑**：tinygrad 中的 `assign` 操作在测试失败后正接受调查；一位用户提到 *master 分支上的 assign 实际上已经损坏，此处测试失败* ([#12131](https://github.com/tinygrad/tinygrad/pull/12131))。
   - 讨论质疑 `assign` 是否应该像 `store` 一样返回一个值，并建议在 `rangeify` 中进行潜在的重构以解决问题，因为 assign 的返回值从未被使用。
- **RTX 4090 上的 GEMM TFLOPs 基准测试目标**：用户讨论了在 RTX 4090 的理论峰值吞吐量下，通过多阶段 kernel 实现 *165+ TFLOP GEMM（匹配 torch），FP16 或 BF16 带有 FP32 累加* 这一悬赏目标的可行性。
   - 有人担心，除非实际时钟频率超过加速时钟（boost clock），否则达到目标 TFLOPs 可能并不现实。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1416051966043357195)** (4 messages): 

> `tinygrad documentation, company meeting` 


- **tinygrad 文档获得好评**：一位成员称赞了 **tinygrad 的文档**，称其 *实用* 且 *简洁*。
   - 该文档让这位成员反复感叹内容非常 *清晰易懂（make sense）*。
- **tinygrad 下一次公司会议**：一位成员询问下一次公司会议的时间，表示如果可能的话想旁听。
   - 会议定于 **圣地亚哥时间周一上午 9 点** 举行。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1416030834921836616)** (4 messages): 

> `RepoMap Benchmarks, Real World Benchmarks, Aider Repomap Use` 


- **RepoMap 基准测试引发关注**：一位成员质疑在基准测试中使用 **RepoMap**，担心它可能会人为地提高通过率。
   - 另一位成员建议 *"使用 repo map 的结果与不使用 repo map 的结果不可比"*，并且当 **LLMs** 在其上下文窗口（context window）中拥有正确信息时，**RepoMap** 可能会提高对较弱模型的信心。
- **请求真实世界基准测试**：一位成员建议基准测试应反映模型的真实使用体验，并指出一项自动化任务除了 **gemini-2.5-pro** 之外，对所有模型来说都是不可能完成的。
   - 由于 **Gemini 2.5 pro** 的表现优于所有其他模型，评估方法需要修订以获取真实世界的数据。
- **Aider 从 RepoMap 中获益**：**RepoMap** 通过文件名、类签名和函数签名提供额外的上下文，这有助于 **LLMs** 理解可用资源。
   - 一位成员始终在开启 **RepoMap** 的情况下使用 **Aider**，认为使用 **RepoMap** 的排行榜能更准确地反映他们的真实使用场景，尽管基准测试结果可能仍与实际代码案例有所不同。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1415849129145274501)** (5 messages): 

> `C to Rust Migration with Aider, Aider always start in /ask mode` 


- **使用 Aider 进行 C 到 Rust 迁移**：一位用户正在使用 **aider** 在 Python 脚本中执行 **C 到 Rust 的迁移**，但 **aider** 无法自动导航和读取相关的 **C** 文件。
   - 用户正在寻求关于他们可能遗漏的 **aider** 功能方面的指导。
- **配置 Aider 始终以 /ask 模式启动**：一位用户正在寻找一种方法，通过 **YAML 配置** 让 **aider** 始终以 **/ask 模式** 启动。
   - 他们检查了文档但没找到相关的配置键，另一位用户建议使用 `aider --chat-mode ask` 或创建一个包含 `chat-mode: ask` 的 `ask.yml` 配置文件，然后运行 `aider -c ask.yml`。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1415783420302200922)** (9 条消息🔥): 

> `WordPress to Next.js conversion, Manus AI Basic Plan, Mount free credits, Manus interlink knowledge, Manus credits rollover` 


- **WordPress 转换为 Next.js 以适配 Vercel？**：一位成员询问如何将 **WordPress 网站** 转换为 **Next.js** 以便在 **Vercel** 上托管，并提到了从 **PHP** 到 **React.js** 的转变。
   - 另一位成员建议使用 **Manus** 或其他 **AI** 工具克隆网站作为替代方案。
- **Manus AI Basic Plan 订阅者感到沮丧**：一位 **Basic Plan** 订阅者对取消购买额外额度（credits）的选项表示不满，这迫使用户即使只有少量需求也必须升级套餐。
   - 他们请求 **Manus AI** 重新考虑为 Basic 用户开放充值，并强调了灵活性的重要性。
- **新用户 Mount 额度问题**：一位新用户报告称，尽管网站上有相关说明，但在 **Mount** 创建账户后并未收到标准的 **1,000 免费额度**。
   - 讨论中未提供解决方案或进一步信息。
- **Manus 知识互联咨询**：一位成员询问 **Manus** 是否可以从所有聊天中提取信息，以便将每个聊天/任务中的知识互联起来供通用。
   - 关于 **Manus** 的知识互联能力，讨论中未提供任何回复或说明。
- **每日 Manus 额度停止发放？**：一位用户报告称其 **每日 300 额度** 已停止发放，引发了困惑。
   - 讨论中未提供解决方案或进一步信息。

  

---


---


---