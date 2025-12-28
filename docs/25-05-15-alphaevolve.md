---
companies:
- google-deepmind
- openai
date: '2025-05-15T05:44:39.731046Z'
description: 'Deepmind 的 **AlphaEvolve** 是 AlphaTensor 和 FunSearch 的 2025 年更新版本。它是一个由
  Gemini 驱动的**用于算法发现的编程智能体**，旨在设计更快的矩阵乘法算法、解决开放性数学难题，并提高数据中心和 AI 训练的效率。它在 Gemini 训练中实现了
  **23% 的内核速度提升**，并在 20% 的应用问题上超越了现有最先进水平（SOTA），包括对最小重叠问题（Minimum Overlap Problem）和接吻数问题（Kissing
  number problem）的改进。与深度强化学习（Deep-RL）不同，它优化的是代码片段而非模型权重。


  与此同时，**OpenAI** 在 ChatGPT 中发布了 **GPT-4.1**，该模型专注于编程和指令遵循；此外，更快的替代方案 **GPT-4.1 mini**
  已面向所有用户取代了 GPT-4o mini。OpenAI 还推出了安全评估中心（Safety Evaluations Hub）以及 “OpenAI to Z”
  挑战赛，利用 o3/o4 mini 和 GPT-4.1 模型来发现考古遗址。*“也许中段训练（midtrain）+ 良好的搜索就是 AI 实现科学创新所需的一切”*
  —— Jason Wei。'
id: MjAyNS0w
models:
- gemini
- gpt-4.1
- gpt-4o-mini
- o3
- o4-mini
people:
- _philschmid
- scott_swingle
- alex_dimakis
- henry
- jason_wei
- kevinweil
- michpokrass
- scaling01
- gdb
title: Gemini 的 AlphaEvolve 智能体利用 Gemini 2.0 发现新的数学成果，并在不使用强化学习（RL）的情况下，使 Gemini 的成本降低了
  1%。
topics:
- algorithm-discovery
- coding-agents
- matrix-multiplication
- optimization
- reinforcement-learning
- model-weights
- training-efficiency
- safety-evaluations
- instruction-following
- coding-tasks
- model-releases
---

Agent Harnesses 是你所需要的一切。

> 2025/5/15-5/16 的 AI 新闻。我们为你查看了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（214 个频道，3819 条消息）。预计节省阅读时间（按 200wpm 计算）：341 分钟。我们的新网站现已上线，提供完整的元数据搜索和美观的 vibe coded 呈现方式，涵盖所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上给我们反馈！

[Deepmind 的新 AlphaEvolve](https://x.com/GoogleDeepMind/status/1922669321559347498) 是 [AlphaTensor](https://deepmind.google/discover/blog/discovering-novel-algorithms-with-alphatensor/) 和 [FunSearch](https://www.nature.com/articles/s41586-023-06924-6) 的 2025 年更新版。它很难被完全理解（grok），因为它总结了过去一年在数学和 LLM 训练应用领域的广泛成果，且目前尚未公开。但 GDM 简明扼要地将其描述为一个“基于 Gemini 的**用于算法发现的 coding agent**……能够：

- 设计更快的矩阵乘法算法，
- 为开放数学问题寻找新解，
- 提高 @Google 内部数据中心、芯片设计和 AI 训练的效率。

它被描述为一个 Agent 而非模型，是因为其包含[循环中的多个组件](https://x.com/GoogleDeepMind/status/1922669325283942539)：


![](https://resend-attachments.s3.amazonaws.com/7muzi57wBPdtSjP)


这种低调处理成果的做法非常有 Google 风格，所以必须转向 Twitter 圈来获取更精彩的亮点：

- “**通过提速 23% 的 kernel 使 Gemini 训练加速，最终减少了 1% 的总训练时间。**” —— [Philipp Schmid](https://x.com/_philschmid/status/1922913381746352188)
- “在它所应用的 **20% 的问题上超越了 SOTA**，这简直疯了” —— [Scott Swingle](https://x.com/bio_bootloader/status/1923121148864164123)
- “结果令人印象深刻：他们**改善了许多问题的已知最佳边界**，包括 Erdos 的最小重叠问题（Minimum Overlap Problem）、矩阵乘法以及 11 维的吻数（Kissing number）。”
- “这里的解决方案是*代码片段*，而**这是一个修改、评估和优化代码（即文本片段）的 search agent**。这与 **Deep-RL** 形成了鲜明对比，后者的解决方案是模型，优化的是权重。” —— [Alex Dimakis](https://x.com/AlexGDimakis/status/1923160843795169447)
- 关于 FlashAttention CUDA 代码 32% 的提速 —— [Henry](https://x.com/arithmoquine/status/1922751330474500530)
- “AlphaEvolve 对像我这样的 RL 死忠粉来说非常令人不安。**也许 midtrain + 优秀的 search 就是科学创新 AI 所需的一切**。保密一年真是个 alpha move。” —— [Jason Wei](https://discord.com/channels/822583790773862470/1372364050184409088/1372651541663846611)

好奇的读者可以观看关于它的 MLST 访谈：

https://www.youtube.com/watch?v=vC9nAosXrJw

---

# AI Twitter Recap

**GPT-4.1 和 OpenAI 模型发布**

- **GPT-4.1 现已在 ChatGPT 中上线**：[@OpenAI](https://twitter.com/OpenAI/status/1922707554745909391) 宣布 **GPT-4.1** 已在 ChatGPT 中可用，并强调了其在**编程任务**和**指令遵循**方面的专长，使其成为日常编程需求中比 **OpenAI o3** 和 **o4-mini** 更快的替代方案。同时 [@kevinweil](https://twitter.com/kevinweil/status/1922732062345142306) 也指出，该模型已面向 Plus/Pro/Teams 订阅用户开放，并将很快推向 Enterprise/Edu 用户。[@michpokrass](https://twitter.com/michpokrass/status/1922716587468984689) 确认，在最初计划仅提供 API 后，**GPT-4.1 今日已登陆 ChatGPT**；而 [@scaling01](https://twitter.com/scaling01/status/1922715792849674568) 表示，这对于**所有 ChatGPT 免费用户来说是一个巨大的升级**！并指出 GPT-4.1-mini 取代了 GPT-4o mini，且表现确实要好得多。
- **推出安全评估中心 (Safety Evaluations Hub)**：[@OpenAI](https://twitter.com/OpenAI/status/1922684895496720490) 推出了 Safety Evaluations Hub，这是一个用于探索其模型安全结果的资源，强调了关于安全的积极沟通。
- **推出 GPT-4.1 mini**：[@OpenAI](https://twitter.com/OpenAI/status/1922707556402618533) 宣布他们还将在 ChatGPT 中为所有用户推出 GPT-4.1 mini，以取代 GPT-4o mini。
- **发布 OpenAI to Z 挑战赛**：[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1923062948060168542) 宣布了 OpenAI to Z 挑战赛，使用 **o3/o4 mini 和 GPT 4.1** 模型来发现以前未知的考古遗址；同时 [@gdb](https://twitter.com/gdb/status/1923105670464782516) 也在发布 OpenAI to Z 挑战赛 —— 使用 o3/o4 mini 和 GPT 4.1 模型来发现以前未知的考古遗址。
- **Evals API 和仪表板新增 Responses API 支持**：[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1923048126002102530) 宣布在 Evals API 和仪表板中增加了对 Responses API 的支持，并提供了一份简便的入门指南，以在存储的响应上比较 gpt-4.1-mini 与 gpt-4o-mini 为例 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1923048127826722849)。

**Google 的 AlphaEvolve 和 Gemini**

- **AlphaEvolve，一个由 Gemini 驱动的编程 Agent**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1922669321559347498) 推出了 AlphaEvolve，这是一个由 Gemini 驱动的用于算法发现的编程 Agent，它可以设计更快的矩阵乘法算法，寻找开放数学问题的新解决方案，并使整个 @Google 的数据中心、芯片设计和 AI 训练更加高效。[@demishassabis](https://twitter.com/demishassabis/status/1922855470374572051) 向 **AlphaEvolve、Gemini 和科学团队**取得的成就表示祝贺。他们还详细介绍了 [AlphaEvolve 的使用方式](https://twitter.com/GoogleDeepMind/status/1922669325283942539)，即利用 LLM 合成信息，对可测量问题进行自动评估，并通过进化算法迭代改进算法。该公司还一直[应用 AlphaEvolve](https://twitter.com/GoogleDeepMind/status/1922669328660299914) 来优化数据中心调度，辅助硬件设计，并增强 AI 训练和推理。AlphaEvolve 还被用于[发现新的矩阵乘法算法](https://twitter.com/GoogleDeepMind/status/1922669331336384515)（表现优于之前的 AlphaTensor 模型），以及[寻找开放数学问题的新解决方案](https://twitter.com/GoogleDeepMind/status/1922669334142271645)。该公司旨在[继续开发 AlphaEvolve](https://twitter.com/GoogleDeepMind/status/1922669336101065183)，鉴于其在不同领域的潜在影响。
- **Gemini 的隐式缓存 (Implicit Caching)**：[@_philschmid](https://twitter.com/1922650422382104584) 强调了 GoogleDeepMind 的 Gemini 对隐式缓存的支持，当请求命中缓存时，可实现高达 **75% 的成本节约**。这在发送具有公共前缀的请求时特别有用，例如查询大型 PDF 的某些部分。

**开源模型、训练和框架**

- **Nous 去中心化预训练运行**：[@Teknium1](https://twitter.com/Teknium1/status/1922778056290419166) 宣布 Nous 已开始对一个类 Deepseek 的稠密模型进行去中心化预训练，该模型拥有 40B 参数，超过 20T tokens，并采用 MLA 以实现高效的长上下文处理。
- **Hugging Face MCP 课程**：[@reach_vb](https://twitter.com/reach_vb/status/1923038382126424380) 宣布 Hugging Face 推出了 MCP 课程，涵盖了关于 Model Context Protocol 的所有必要知识及其使用方法。
- **AM-Thinking-v1 推理模型**：[@omarsar0](https://twitter.com/omarsar0/status/1922668488826741061) 指出 AM-Thinking-v1 看起来是一个强大的 32B 推理模型。它的表现优于 DeepSeek-R1，并与 Qwen3-235B-A22B 旗鼓相当，且基于开源构建。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1922483522549252200) 指出 AM-Thinking-v1 的性能与 Qwen3-235B-A22B 和 Seed1.5-Thinking 持平，而它完全是基于开源的 Qwen2.5-32B 基础模型和公开可用的查询构建的。
- **Salesforce BLIP3-o 多模态模型**：[@_akhaliq](https://twitter.com/_akhaliq/status/1923001183804764391) 和 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1922843713514193076) 提到 Salesforce 已在 Hugging Face 上发布了 BLIP3-o。这是一个完全开放的统一多模态模型系列，采用 Diffusion Transformer 来生成语义丰富的 CLIP 图像特征。

**推理与 Agent 系统**

- **LLM 在多轮对话中“迷失”**：[@omarsar0](https://twitter.com/omarsar0/status/1922755721428598988) 重点介绍了一篇研究 LLM 在现实多轮对话场景中表现的论文，在这些场景中，用户指令通常不够明确，需要通过多轮对话进行澄清。[@omarsar0](https://twitter.com/omarsar0/status/1922755768585158785) 指出，与单轮、明确的指令相比，所有测试的 LLM 在多轮、不明确的对话中表现都显著下降。即使是 SoTA 模型，在六个任务中的平均性能也下降了 39%。他还列出了 LLM “迷失”的[主要原因](https://twitter.com/omarsar0/status/1922755800843550833)，包括过早做出假设、在获得所有必要信息前尝试提供完整解决方案以及输出过于冗长。
- **FedRAG 框架**：[@*nerdai*](https://twitter.com/_nerdai_/status/1922732119706698118) 介绍了 FedRAG，这是一个用于在集中式和联邦架构中微调 RAG 系统的开源框架。
- **Chain-of-Thought 推理**：[@francoisfleuret](https://twitter.com/francoisfleuret/status/1922892961680896238) 声称 CoT 是采样有意义潜变量这一“真正过程”的简陋版本。
- **用于搜索效率 LLM 的 RL**：[@omarsar0](https://twitter.com/omarsar0/status/1922665313117552664) 指出，这提出了一种新的训练后 RL 框架，专门训练 LLM 以优化搜索的使用。
- **LangChain 的 Open Agent Platform (OAP)**：[@LangChainAI](https://twitter.com/LangChainAI/status/1922722850542346680) 推出了 Open Agent Platform，这是一个开源、无代码的 Agent 构建平台，可连接到 MCP 工具、用于 RAG 的 LangConnect 以及其他 LangGraph Agents。
- **用于 Zero-Shot 测试的 Runway References**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1922742658620903885) 展示了用于衣服、地点和姿势的 Zero-Shot 测试的 Runway References。

**AI 实施、工具与基础设施**

- **Meta 的 Transformers + MLX 集成**：[@awnihannun](https://twitter.com/awnihannun/status/1923065749234647214) 表达了 Transformers 对开源及整个 AI 生态系统的重要性，并期待 MLX 与 Transformers 之间有更多、更深层次的集成。
- **OpenAI 的技术栈**：[@nrehiew_](https://twitter.com/nrehiew_/status/1922668335960924579) 指出 OpenAI 使用 FastAPI 来提供 ChatGPT 服务，反驳了关于 Python 和 FastAPI 能力的质疑。
- **LangGraph Platform 正式商用**：[@LangChainAI](https://twitter.com/LangChainAI/status/1922709747423183226) 宣布 LangGraph Platform 现已正式商用 (GA)，允许用户部署、扩展和管理具有长期运行、有状态工作流的 Agents。
- **GPT-4.1 编程能力**：[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1922709921772036164) 表示，它是由今天在 ChatGPT 中推出的 GPT-4.1 编写的。
- **Embedding 在训练中的重要性**：[@jxmnop](https://twitter.com/jxmnop/status/1922468210256879786) 表示 Embedding 的重要性被低估了。
- **Atropos 与 Axolotl AI**：[@Teknium1](https://twitter.com/Teknium1/status/1922435846751584771) 现在你也可以使用 @axolotl_ai 通过 Atropos 进行训练了。

**AI 分析与评估**

- **ARI 击败 OpenAI 的 Deep Research**：[@RichardSocher](https://twitter.com/RichardSocher/status/1923098655768314363) 宣布 ARI (Advanced Research & Insights agent) 在两个 Benchmark 上以巨大优势击败了 OpenAI 的 Deep Research。
- **GPT-4.1 出色的编程能力**：[@kevinweil](https://twitter.com/kevinweil/status/1922732062345142306) 表示 GPT 4.1 在编程和指令遵循（instruction following）方面表现非常出色，建议用户尝试。
- **当前评估（evals）的局限性**：[@cline](https://twitter.com/cline/status/1922722359795916943) 报告称，评估循环（eval loops）在接触真实人类时很少能维持效果。
- **LangChain Interrupt 2025 评估**：[@LangChainAI](https://twitter.com/LangChainAI/status/1922747560483226041) 推出了 OpenEvals，这是一套用于模拟完整对话并评估 LLM 应用性能的工具集。
- **AM-Thinking-v1 模型评估**：[@omarsar0](https://twitter.com/omarsar0/status/1922668488826741061) 指出 AM-Thinking-v1 的表现优于 DeepSeek-R1，并可与 Qwen3-235B-A22B 媲美；[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1922483522549252200) 注意到 AM-Thinking-1 的表现与 Qwen3-235B-A22B 和 Seed1.5-Thinking 相当，且完全基于开源的 Qwen2.5-32B 基础模型和公开可用的查询构建。

**幽默与杂项**

- **真相大白**：[@omarsar0](https://twitter.com/omarsar0/status/1922755721428598988) 讨论了 LLM 在多轮对话中迷失方向的话题。
- **OpenAI 在 ChatGPT 中使用 FastAPI**：[@nrehiew_](https://twitter.com/nrehiew_/status/1922668335960924579) 调侃尽管人们不断抱怨 Python，但它仍被广泛使用。
- **Google 时间**：[@zacharynado](https://twitter.com/zacharynado/status/1922652507681026236) 惊呼 ♊️ GOOGLE TIME! (づ｡◕‿‿◕｡)づ♊️。
- **LLM 只会对我撒谎**：[@nearcyan](https://twitter.com/nearcyan/status/1922548145340195148) 表达了对 LLM 的厌恶。
- **你已经选好了你的毒药！**：[@scottastevenson](https://twitter.com/scottastevenson/status/1922491520445305338) 回复一位用户关于逃避模糊性并屈从于结构、教育、刷屏、毒瘾、健身计划、登山、创业、抚养家庭、虐待关系的话题。
- **mog 趋势有点尴尬，但这个真的很硬核**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1922694806599299432) 对一张图片的反应。
- **flash-attention 居然能和 uv 配合使用，这让我看到了希望**：[@typedfemale](https://twitter.com/typedfemale/status/1922427558924001672) 对 uv 和 flash-attention 的乐观评论。
- **用不完美的零件制造完美的机器**：[@MillionInt](https://twitter.com/MillionInt/status/1923023812821385240) 一条深刻的评论。
- **噢不**：[@lateinteraction](https://twitter.com/lateinteraction/status/1922708925142475078) 对发生某事的反应。
- **这太棒了，我没指望 YC 的发布视频会是像一个月前刚发生的事情的纪实剧，但事实就是如此**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1922924417429958760) 对 YC 发布视频的反应。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

### 1. Unsloth 中的文本转语音模型训练与工具

- [**Unsloth 现已支持 TTS 微调！**](https://v.redd.it/faqjz7kzaz0f1) ([Score: 223, Comments: 43](https://www.reddit.com/r/LocalLLaMA/comments/1kndp9f/tts_finetuning_now_in_unsloth/)): **Unsloth 推出了对高效文本转语音 (TTS) 模型微调的支持，声称与替代方案相比，训练速度快约 1.5 倍，VRAM 占用减少 50%，特别是在 FA2 硬件上。支持的模型包括** `Sesame/csm-1b`**、** `CanopyLabs/orpheus-3b-0.1-ft` **以及基于 Transformer 的模型（例如 Llasa, Outte, Spark），并使用像 'Elise' 这样带有情感标注的数据集进行数据高效的 SFT 风格工作流。用户可以利用 Google Colab [notebooks](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning)、16-bit LoRA 或全精度微调，以及 Hugging Face 上的量化/原始 Checkpoints。值得注意的是，还支持一种新的 Qwen3 GRPO 方法，将基础模型与自定义的基于邻近度的奖励函数和正则引导评估相结合。** 评论澄清说，虽然 Whisper 主要是 STT 模型，但将其包含在内可能是为了 ASR 相关的预处理或数据集生成。用户讨论了 LoRA 微调的可扩展性，以及控制 TTS 模型音调、音高和节奏的最佳实践，还有每参数量所需的数据集大小要求，突显了对实际微调方法的兴趣。
    - 一位用户质疑 "Whisper" 是否是 TTS 模型——澄清它主要是 STT（语音转文本）和 ASR（自动语音识别）模型，而不是文本转语音。该评论询问 Unsloth 的微调是否支持 ASR 微调的数据集，而非真正的 TTS。
    - 另一位评论者专门询问了 TTS 微调的要求，即每十亿或每亿参数需要多少音频/文本示例。这反映了语音合成微调中数据集规模与模型参数化之间的常见技术关注点。
    - 提出了关于原生 Mac MPS (Metal Performance Shaders) 支持的技术功能请求，旨在实现在 Apple Silicon 设备上的硬件加速训练/推理，这对于脱离 CUDA 依赖的高效 TTS 模型微调工作流非常重要。
- [**A.I.T.E Ball 介绍**](https://v.redd.it/scyofz31dx0f1) ([Score: 281, Comments: 53](https://www.reddit.com/r/LocalLLaMA/comments/1kn542r/introducing_aite_ball/)): **该帖子详细介绍了一个运行在 Orange Pi Zero 2W 上的 AI 驱动“8 号球”设备的本地离线实现。该设置使用 whisper.cpp 进行本地文本转语音，llama.cpp 进行 LLM 推理，并专门运行 Gemma 3 1B 模型，强调了资源限制和完全离线的能力。** 评论者对完全本地运行表示赞赏，强调了在重度联网的环境中离线 AI 硬件的稀缺性。没有实质性的技术辩论。
    - 一位评论者建议集成 Piper TTS (https://github.com/rhasspy/piper)，这是一个开源文本转语音引擎，以增强该项目，并指出它可以在入门级硬件上高效运行。这意味着尽管硬件有限，该设备仍可以升级语音输出功能。
    - 多条评论强调了完全离线运行模型的独特功能，将其与始终在线的 AI 设备趋势形成对比。这种离线能力被认为具有重要意义，特别是对于隐私保护和在资源受限硬件上增加对应用程序的控制。

### 2. llama.cpp 中的新特性与数据处理

- [**PDF 输入已合并至 llama.cpp**](https://github.com/ggml-org/llama.cpp/pull/13562) ([Score: 120, Comments: 34](https://www.reddit.com/r/LocalLLaMA/comments/1kn75q8/pdf_input_merged_into_llamacpp/)): **最近的一个 PR ([#13562](https://github.com/ggml-org/llama.cpp/pull/13562)) 通过集成外部 JS 库进行 PDF 解析，在 llama.cpp 的 Web UI 中添加了原生 PDF 输入支持，为用户提供了在文本提取和图像渲染之间切换的能力。这种方法确保了 C++ 核心不受影响，允许快速更新和替换 PDF 解析工具，并包含将冗长的粘贴内容自动转换为文件上传的功能。** 评论指出，这种实现方式保持了核心的模块化（符合可维护性），但也有人对功能蔓延（feature creep）与遵循 Unix 哲学之间的矛盾表示担忧。目前存在关于针对混合内容 PDF 进行 OCR 集成的技术讨论，以及要求合并相关 PR 以实现扩展文档处理的请求。
    - llama.cpp 的 PDF 输入功能是在内置的 Web 前端中使用外部 JavaScript 包实现的，而不是在核心 C++ 应用程序中。这种架构决策使核心维护工作量降至最低，并允许在不影响核心功能的情况下轻松更换或升级 PDF 转换包。
    - 目前，该解决方案提供了两种处理 PDF 的模式：解析为纯文本或纯图像。用户公认更稳健的方法是原生选择性地提取文本，同时仅对图像部分应用 OCR——类似于专业 OCR 软件的工作方式——这暗示了未来在 PDF 结构和语义理解方面可能的改进。
    - 用户质疑现有的集成是否能从 PDF 中提取并表示结构化信息，如表格或嵌入图像，这对于 RAG (Retrieval Augmented Generation) 和图谱构建等高级任务至关重要。在 LLM 流水线的 PDF 处理中，对此类功能的有效支持被认为是一项重大的技术挑战。
- [**A.I.T.E Ball 简介**](https://v.redd.it/scyofz31dx0f1) ([Score: 281, Comments: 53](https://www.reddit.com/r/LocalLLaMA/comments/1kn542r/introducing_aite_ball/)): **该帖子详细介绍了一个完全在 Orange Pi Zero 2W 上运行的 AI 驱动“8 号球”设备的本地离线实现。该设置使用 whisper.cpp 进行本地语音转文本，使用 llama.cpp 进行 LLM 推理，并专门运行 Gemma 3 1B 模型，强调了资源限制和完全离线的能力。** 评论者对完全本地运行表示赞赏，强调了在高度联网的环境中，离线 AI 硬件的稀缺性。目前没有实质性的技术争论。
    - 一位评论者建议集成 Piper TTS (https://github.com/rhasspy/piper)（一个开源文本转语音引擎）来增强该项目，并指出它能在配置较低的硬件上高效运行。这暗示了尽管硬件有限，该设备仍可以升级语音输出功能。
    - 多条评论强调了完全离线运行模型的独特之处，这与始终在线的 AI 设备趋势形成了鲜明对比。这种离线能力被认为具有重要意义，特别是在隐私保护以及在资源受限硬件上增加对应用程序的控制方面。

### 3. LLM 多轮对话挑战与基准测试

- [**LLM 在多轮对话中迷失**](https://www.reddit.com/r/LocalLLaMA/comments/1kn2mv9/llms_get_lost_in_multiturn_conversation/) ([Score: 218, Comments: 67](https://www.reddit.com/r/LocalLLaMA/comments/1kn2mv9/llms_get_lost_in_multiturn_conversation/)): **最近的一篇论文 ([arXiv:2505.06120](https://arxiv.org/abs/2505.06120)) 表明，无论是开源还是闭源的 LLM (Language Learning Models)，在多轮对话中性能都会出现大幅下降，特别是当指令被“分片”（sharded，跨轮次拆分）而非“拼接”（concat，一次性提供）时。实验显示，LLM 在最初做出错误假设后经常会产生复合错误，且很少能从早期的误解中恢复——这种现象在单轮基准测试中无法体现。研究建议，在首条提示词中重新启动包含所有相关上下文的对话可以缓解这一问题。** 评论者们通过实践经验证实了这些发现，并指出各种模型（如 o1 pro, sonnet 3.7，而 2.5 pro 则有所改进）都存在这种多轮退化现象。一个详细的例子展示了与 LLM（Gemma, Qwen）进行迭代提示如何导致语义漂移（semantic drift），并由于 LLM 对先前输出的依赖而导致初始错误的复合，说明了多轮上下文追踪中的核心挑战。
    - 用户观察到，像 o1 pro, sonnet 3.7 甚至是像 Gemma 和 Qwen 这样强大的开源模型，经常在早期轮次中做出错误的初始假设，然后由于其自回归（autoregressive）特性，在多轮对话中使这些错误复合。正如一位用户所言：*“LLM 作为词概率引擎，会倾向于坚持之前的选择，因此最初的错误会导致复合错误和普遍的异常。”*
    - 有批评指出，大多数 LLM 基准测试和微调练习主要集中在单轮、完全指定的指令设置上，这可能会误导实际的多轮或以 Agent 为中心的使用场景。一位评论者强调，在现实世界尤其是编程工作流中，多轮性能至关重要，但模型可能仅针对基准测试分数进行了优化。
    - 关于“全量与拼接”（full and concat）提示策略的评论反映了人们对上下文处理方法如何影响性能的关注：据称早期和较小的模型严重依赖提示词结构，而较新/现代的模型可能更好地管理上下文。这引起了人们对衡量模型是否适合 Agent 使用和扩展对话的一个重要技术维度的关注。

## 其他 AI Subreddit 汇总

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

待完成

---

# AI Discord 汇总

> 由 Gemini 2.5 Pro Exp 生成的总结之总结
> 

**主题 1：模型狂热——新发布与新功能引发激烈辩论**

- [**Gemini 2.5 Pro 大显身手（及其上下文窗口）**](https://ai.google.dev/gemini-api/rest/v1beta/GenerationConfig)：来自 **LMArena**、**aider** 和 **OpenAI** 等 Discord 频道的工程师们广泛讨论了 **Gemini 2.5 Pro**，称赞其编程实力、推理能力以及巨大的 **100 万 token 上下文窗口**，一些人认为这与 **GPT 的 32k 限制**相比是不可或缺的。虽然其[免费可用性](https://discord.com/channels/974519864045756446/998381918976479273/1372558644578881576)受到欢迎，但许多人承认由于高昂的运营成本，这可能是暂时的，尽管一些用户发现其推理块（reasoning chunks）*毫无用处*。
- [**GPT-4.1 变体争夺编程桂冠**](https://openai.com/index/gpt-4-1/)：**OpenAI** 和 **aider** 社区对 **GPT-4.1** 和 **GPT-4o** 进行了热烈比较，许多人断言 **GPT-4.1**（尤其是 **GPT-4.1 mini**）由于更好的指令遵循能力在编程任务中表现出色。用户分享了[模型对比截图](https://cdn.discordapp.com/attachments/998381918976479273/1372555924862140436/20250515_154222.png?ex=682733d1&is=6825e251&hm=206683cf5fa5279c41f8150a0645b58d3595bbfde5b3616003058b41a29e5cae&)，并讨论了潜在的 **GPT-5** 发布时间，推测在 2024 年夏季或晚些时候。
- [**AI 竞技场的新面孔：DeepSeek, Qwen, AlphaEvolve 及更多！**](https://deepseek.com/blog/deepseek-v3)：新模型公告点燃了讨论，包括作为 **混合专家模型 (MoE)** 的 **DeepSeek v3**、擅长翻译普通话数据集的 **Qwen3**，以及 **Google DeepMind** 的 **AlphaEvolve** ([AlphaEvolve PDF](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf))，后者引发了关于它是真正的演化算法还是 LLM 驱动的辩论。**Samsung** 也加入了竞争，推出了如 **MythoMax-L2-13B**（[在 Hugging Face 上发现](https://huggingface.co/Samsung/MythoMax-L2-13B)）和 **MuTokenZero2-32B** 等模型。

**主题 2：工程化 AI - 优化性能与完善开发工具**

- [**量化之战：QNL 速度完爆 GGUF！**](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs)：**Unsloth AI** 社区报告称，**QNL** 的性能比标准的 **GGUF** 更快，尽管正式的基准测试仍在等待中，但这凸显了在优化模型速度和效率方面的持续努力。**LM Studio** 的讨论还强调，为了获得最佳性能，将模型完全保留在 **VRAM** 中比 **DRAM** 速度更关键，其中 **KV Cache** 的位置是一个关键因素。
- [**框架热潮：DSPy、LlamaIndex、LangGraph 和 MCP 简化 AI 开发**](https://www.shortwave.com/blog/integrate-ai-with-all-your-apps-mcp/)：开发者正积极利用 **DSPy** 等框架通过 **Pydantic models** 实现结构化输出，并使用 **LlamaIndex** 构建事件驱动的 Agent 工作流，例如 [Weaviate 的多 Agent 文档助手](https://twitter.com/llama_index/status/1923085124725506441)。**LangGraph** 在管理复杂对话流方面正受到关注（[LangGraph 课程](https://huggingface.co/learn/agents-course/unit2/langgraph/first_graph)），而 **Meta-Circular Evaluator Protocol (MCP)**（现已获得 [Shortwave 客户端支持](https://www.shortwave.com/docs/how-tos/using-mcp/)）正使 AI Agent 与各种应用程序的集成变得更加容易。
- [**硬件动态：从多 GPU 微调到 WebGPU 吐槽**](https://asianmom.kuber.studio/)：使用 **Accelerate** 和 **Unsloth** 等工具进行多 GPU 微调是一个热门话题，**GPU MODE** 服务器见证了 **MI300** 显卡的活跃基准测试以及关于 AMD GPU 上 **TritonBench** 错误的讨论（[示例 kernel](https://github.com/thunlp/TritonBench/blob/main/data/TritonBench_G_v1/chunk_gla_fwd.py)）。同时，一个名为 [**AsianMOM**](https://asianmom.kuber.studio/) 的有趣 **WebGPU Vision-LLM 应用**（使用 **SmolVLM 500M** 和 **Llama 3.2 1B**）展示了浏览器内的 AI 吐槽能力。

**主题 3：平台特性与用户规避方案 - 应对 AI 现状**

- [**Perplexity 的 Pro 会员问题与 Deep Research 的失望**](https://www.perplexity.ai/search/qing-jin-xing-lian-wang-sou-su-nFCGQq.lT5WtBv8y0y6oRA)：**Perplexity AI** 用户在 Discord 上获取 **Perplexity Pro 角色** 时面临延迟（需由 [管理员手动分配](https://discord.com/channels/1047197230748151888/1111786888626438245)），在网页上查看 [移动端 App 的回答](https://www.perplexity.ai/search/qing-jin-xing-lian-wang-sou-su-nFCGQq.lT5WtBv8y0y6oRA) 时遇到错误，并对 **Deep Research** 模式默认跳转到常规搜索或使用有限来源表示沮丧，一位用户表示：“*如果它被降级了，我看不出有什么理由为此付费。*”
- [**模型故障：循环的 Llama 和错误标记令用户沮丧**](https://cdn.discordapp.com/attachments/1110598183144399058/1372503571530125382/image.png?ex=6827abcf&is=68265a4f&hm=bea36a2386fe250f3a3fddbd040813929e5fdb114f8256409b4bd3d50208415a&)：**LM Studio** 用户报告 **Llama 3.1/3.3** 模型在处理奇幻提示词时产生不理想的输出，显示出 [Token 丢失和标点符号问题](https://cdn.discordapp.com/attachments/1110598183144399058/1372503571530125382/image.png?ex=6827abcf&is=68265a4f&hm=bea36a2386fe250f3a3fddbd040813929e5fdb114f8256409b4bd3d50208415a&)。**Cursor Community** 成员发现 **Claude 3.5 Sonnet** [陷入循环](https://cdn.discordapp.com/attachments/1074847527708393565/1372432736752762930/Gq9pkJRaAAECBOI.png?ex=682769d7&is=68261857&hm=02da0d21e3c19e9a58420fc3ee21f55a47edd8f0e64e700331b1b3ae6bc99a9d&)，而 **Eleuther** 的讨论强调了 **MMLU** 等 **MCQ 评估** 错误地将模型输出标记为错误。
- [**社区救援：代理、Token 技巧和 GPT4All 替代方案**](https://aider.chat/docs/config/aider_conf.html)：当 **OpenAI** 的访问受到国家限制时，**OpenRouter** 用户建议使用代理作为规避方案。**Aider** 用户分享了通过 `/clear` 等命令管理 Token 使用以及使用 **Gemini 2.5 Flash** 等模型的技巧，而 [**Nomic.ai**](http://nomic.ai/) 社区因担心 **GPT4All** 停止维护，讨论了将 [**Jan.ai**](http://jan.ai/) 和 **LM Studio** 作为替代方案。

**主题 4：繁忙的 AI 生态系统 - 协作、学习与开源的胜利**

- [**独立开发者释放创意 AI：从酒店 Agent 到老妈吐槽器！**](https://github.com/jinkoso/jinko-mcp/blob/master/README.md)：社区展示了令人印象深刻的开源项目，包括用于构建销售酒店 AI Agent 的 [**Jinko MCP**](https://github.com/jinkoso/jinko-mcp/blob/master/README.md)、使用 LlamaIndex 工作流构建的 **Tig 编程 Agent**（[由 LlamaIndex 发布](https://twitter.com/llama_index/status/1923134285940441102)），以及幽默的 [**AsianMOM** WebGPU Vision-LLM 应用](https://asianmom.kuber.studio/)，该应用专门用于吐槽用户。此外， [**Mem0.ai**](http://mem0.ai/) 推出了 [**OpenMemory MCP**](https://mem0.ai/blog/introducing-openmemory-mcp/)，这是一个为 AI 应用提供的统一内存管理层。
- [**提升你的 AI 技能：工作坊、网络研讨会和海量挑战！**](https://lu.ma/39b7e9pu)：涌现了大量的学习机会，例如 **Nous Research** 和 **Solana Foundation** 在纽约举办的 [去中心化 AI 活动](https://lu.ma/39b7e9pu)、关于[构建 Agentic 应用](https://www.youtube.com/watch?v=VmjMIwwo9ag)的 **Lambda 工作坊**（提供 **$100 API 额度**），以及 **OpenAI 的 “OpenAI to Z Challenge”**（[详情点击此处](https://openai.com/openai-to-z-challenge/)），旨在发现亚马逊考古遗址。**BlackboxNLP 2025** 还宣布了一个使用 [MIB Benchmark](https://mib-bench.github.io/) 的[关于电路/因果变量定位的新共享任务](https://blackboxnlp.github.io/2025/task)。
- [**助燃创新：API 额度和赞助让创新持续燃烧**](https://github.com/Aider-AI/aider/issues)：慷慨之举不断，**aider** 社区的一位用户为 Gemini、Claude 和 OpenAI 提供免费的 **API 额度**以支持有趣的项目，特别是针对 [aider 项目](https://github.com/Aider-AI/aider/issues)。在其他地方，一个技术相关的非营利组织正在寻求 **Cohere** 的活动赞助和资助（联系方式 [adibvafa.fallahpour@cohere.com](mailto:adibvafa.fallahpour@cohere.com)），凸显了生态系统支持持续开发的多种方式。

**主题 5：AI 的疯狂一面——争议、意外泄露与行业大洗牌**

- [**Grok 的失误：从“白人种族灭绝”言论到承认 Elon 是威胁！**](https://x.com/)：Elon Musk 的 **Grok** 模型在 **aider** 社区引发了争议，因为它发表了关于**白人种族灭绝**的荒谬言论，导致了不信任，一些用户将 **xAI** 视为笑话。一位成员幽默地提到，他询问 **Grok** 谁是 **X** 上对民主最大的威胁，据称它回答：**Elon**。
- [**哎呀，我们泄露了！三星的 MythoMax-L2-13B 短暂亮相**](https://huggingface.co/Samsung/MythoMax-L2-13B)：正如 **Yannick Kilcher** 和 **HuggingFace** Discord 频道中所指出的，**三星**无意中发布（随后迅速删除）了 **MythoMax-L2-13B** 角色扮演模型，该模型[在 Hugging Face 上被发现](https://huggingface.co/Samsung/MythoMax-L2-13B)。这导致一位用户打趣道：*“能不能有人对 **OpenAI** 也这么干，‘发布’一下 **GPT4Chan**？或者 **Anthropic** 也行，那绝对是无价之宝。”*
- [**行业震荡：TypeScript 开发者被解雇，Agentic 工具挑战大厂**](https://x.com/brodyford_/status/1922726909365879039?s=46)：**Microsoft** 意外解雇了一名关键的 **TypeScript** 开发者（[推文在此](https://x.com/brodyford_/status/1922726909365879039?s=46)），引发了 **Latent Space** 社区的沮丧。讨论还涉及 **Agentic 工具**如何赋能独立开发者超越大厂，而 **FUNapis** 的倒闭被幽默地归因于 [**Bing** 的聊天机器人套壳野心](https://x.com/williambryk/status/1923062696095711374?s=46)。


---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 身份组：版主前来救援**：购买 **Perplexity Pro** 的用户在 Discord 上获取 **Perplexity Pro role** 时遇到延迟，但在系统重构期间，[版主正在手动分配身份组](https://discord.com/channels/1047197230748151888/1111786888626438245)。
   - Discord 和 Perplexity 使用相同的邮箱并非必要条件。
- **移动端 App 回答触发网页端错误**：由 **Perplexity 移动端 App** 生成的回答无法在网页版上读取，[导致出现错误信息](https://www.perplexity.ai/search/qing-jin-xing-lian-wang-sou-su-nFCGQq.lT5WtBv8y0y6oRA)。
   - 客服已确认该问题并报告给技术人员，但目前尚未实施修复。
- **Deep Research 令人失望？**：用户报告了 **Deep Research** 模式的问题，因为它默认使用常规搜索且使用的来源数量有限。
   - 一位用户总结了这种情绪：*我之所以使用 Perplexity，是因为它每次查询会读取 20-40 个来源。如果这被削弱了，我看不到付费的理由。*
- **23andMe 面临破产**：[23andMe 已申请 Chapter 11 破产保护](https://www.perplexity.ai/search/23andme-files-for-chapter-11-b-msxlvXlmQCK0MLt0UUeTcg)，表明其面临重大的财务挑战并需要重组。
   - Chapter 11 申请表明 **23andMe** 正在寻求法律保护以重组其债务和业务。
- **Sonar API 难倒黑客松参与者**：成员在为黑客松项目获取 **Sonar API** 时遇到困难，因为它需要信用卡详情。
   - 另一位成员报告称，使用 **Sonar API** 生成的回答与在 [Perplexity AI playground](https://labs.perplexity.ai/) 中生成的回答不同。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 参与 “Vibe Coding” 直播**：Manus 在旧金山举办了一场 “Vibe Coding with Manus” 直播，由 Manus SF Fellows 参与，可在 [YouTube](https://youtube.com/live/w4XegM6dOgc?feature=share) 上观看。
   - 直播展示了编程项目，促进了社区互动。
- **刷分天才 Johnny**：一位用户幽默地指出 Johnny 如何 *“每天在 Manus 刷分 (farming)”*，而另一位用户则 *付费让 Manus 制作一个 femboy 检测器*。
   - 这展示了在利用平台获取免费 credits 与将其用于娱乐之间的幽默对比。
- **Femboy 检测器**：一位用户创建了一个应用程序，通过使用 Function Calling 从 [wise.com](https://wise.com) 获取汇率，来判断一个人是否是 femboy。
   - 该应用对男性名字输出 “femboy”，导致用户被贴上 femboy 标签的喜剧性指控：*MANUS 的 API 在撒谎！*
- **邀请链接功能消失**：一些用户发现从 UI 中生成邀请链接的选项消失了。
   - 似乎邀请链接生成功能不再对所有用户开放，一些人推测这仅影响免费用户。
- **用户思考 Credits 消耗**：用户对 credits 的成本表示担忧，其中一位指出一份 PDF 报告消耗了 **500 credits**，而在 Google 上进行 DCF 消耗了 **1500 credits**。
   - 一些人认为 credits 消耗太贵，特别是对于复杂任务，并期待将 Alipay 作为支付方式。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 的重度推理时间？**：成员们在思考，鉴于 [Gemini 2.5 Pro](https://ai.google.dev/gemini-api/rest/v1beta/GenerationConfig) 是一个**更重的模型**，与其他推理模型相比，它是否会有更长的推理时间。
   - 讨论集中在模型大小、复杂性和推理速度之间的权衡。
- **Grok 3.5 发布陷入泥潭？**：有成员理论化地认为 **Elon** 推迟了 **Grok 3.5** 的发布，是因为将其微调至极右翼的尝试**并未成功**，但随后被另一名成员纠正，指出这是*虚假信息*。
   - 讨论涉及了在系统提示词（system prompt）中注入*政治内容*的可能性。
- **Attention Steering（注意力引导）只是幻象？**：尽管有说法称 **LLMs** 可以引导注意力，特别是在 **Grok** 的 Twitter 时间线中，但一位成员澄清说，*这并不是在需要的地方引导注意力*，而仅仅是一个普通的公告。
   - 他们进一步链接到了 [Transluce 的可观测性界面](https://transluce.org/observability-interface)，将其作为体验特征引导（feature steering）的工具，但也提醒说它在实践中目前还不是特别有用。
- **LMArena 由额度而非现金资助**：在关于 **LMArena** 如何为其模型筹集资金的讨论中，一位成员指出，*不仅仅是公司自己在支付推理费用*，这表明 **LMArena** 获得的是**额度授权（credit grants）**而非直接的货币支付。
   - 另一位成员补充说，大型实验室为 **LMArena** 提供其模型的端点（endpoints），并因此收集到了宝贵的人类偏好数据。
- **O3 Pro 在 Arena 中缺席**：尽管有关于 **O3 Pro** 发布的猜测，一位成员表示 *O3 Pro 不会出现在 Arena 中，哈哈*，对此一位管理员回复道：*我无法确认新模型是否或何时会到达 Arena，但我会确保在可以的时候发布公告*。
   - 平台添加新模型的预期仍然很高。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **QNL 在速度上完胜 GGUF**：据报道 **QNL** 比标准的 **GGUFs** 更快，但正式的基准测试（benchmarks）仍在进行中。
   - 详情请参阅 [Unsloth Dynamic 2.0 GGUFs 文档](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs)。
- **通过多 GPU 微调进行扩展**：对于多 GPU 微调，将 **Accelerate** 与 Unsloth 结合使用可能会取得成功。
   - 尽管像 **3090** 这样的消费级 GPU 提供 **24GB VRAM**，但公司在进行本地 AI 开发时通常会选择 **H100s**。
- **SLMs 挑战 LLMs**：小型模型（**SLMs**）通过针对特定任务进行微调可以变得极具竞争力，即使它们在开箱即用时并不那么聪明。
   - 对于需要不错推理能力的模型，推荐使用 **Qwen3-4B**，据称它击败了 **Mistral 7B**。
- **Qwen3 攻克翻译难题**：由于其预训练数据，**Qwen3** 被建议用于翻译中文数据集。
   - 用户报告了在 Kaggle 上使用 **Ollama** 运行 **30B** 模型处理数百万个字符串的成功案例。
- **SmolVLM 在浏览器中进行吐槽**：一位成员创建了 **AsianMOM**，这是一个 [WebGPU Vision-LLM 应用](https://asianmom.kuber.studio/)，它能在浏览器中像你老妈一样吐槽你。
   - 它使用了 **SmolVLM 500M** 和 **LLama 3.2 1B**，得益于 Transformers.js 和 HF，它可以直接在你的浏览器中运行。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Grok 引发对白人种族灭绝的担忧**：Elon Musk 的 **Grok** 因发表关于**白人种族灭绝**的荒谬言论而引发关注，导致了信任危机，并促使一些人将 **xAI** 视为笑话，认为 **Elon** 是 **X** 上对民主的最大威胁。
   - 用户正在避开该模型，一名成员开玩笑说，他们使用 **xAI** 只是为了询问谁是 **X** 上对民主的最大威胁，而它承认是 **Elon**。
- **Gemini 2.5 Pro 撤出 Copilot**：**Gemini 2.5 Pro** 曾在 Copilot 中短暂可用，但随后被移除，引发了关于 **Microsoft** 加大对开源权重模型（open-weight models）投资的猜测。
   - 此次移除引发了猜测，认为 **Microsoft** 与 **OpenAI** 之间不稳定的关系可能导致他们转向关注开源权重模型。
- **开发者慷慨解囊：免费 API 额度发放**：一位用户提供了 **Gemini**、**Claude** 和 **OpenAI** 的免费 **API credits**，邀请他人测试新基础设施并开发有趣的项目，特别是针对 [aider 项目](https://github.com/Aider-AI/aider/issues)。
   - 另一位成员计划为 aider 添加 `/consolidate` 命令，将每个长对话滚动成一个完整的 prompt，并使用主模型发起全新的单轮请求，以应对 [LLMs Get Lost In Multi-Turn Conversation](https://arxiv.org/abs/2505.06120) 论文中提到的问题。
- **Aider Token 使用得到控制**：用户讨论了管理 **Aider** 中 token 使用的策略，建议使用 `/clear` 减少上下文，仅使用 `/add` 添加必要文件，并通过 Google AI Studio 使用 **Gemini 2.5 Flash** 等模型。
   - 还建议使用 **OpenRouter**（如 Deepseek v3 0324），并配合从 **Gemini 2.5 Pro** 进行复制粘贴。
- **Muscle-mem 工具浮现**：一名成员分享了一个名为 [muscle-mem](https://github.com/pig-dot-dev/muscle-mem) 的 GitHub 工具链接，可能是一个记忆辅助工具？
   - 没有分享其他细节，因此很难知道它的具体用途！

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4.1 Mini 的智能引发编程辩论**：成员们辩论了 **GPT-4.1** 与 **GPT-4o** 的优劣，一些人断言 **4.1** 在编程方面更胜一筹，而另一些人认为 **4o** 整体上更直观，还有几个人声称 **4.1 mini** 是最好的小型模型。
   - 一些用户分享了他们的 prompt 测试结果以及在特定方言环境下的经验，以展示他们对 **GPT-4.1** 的偏好，并分享了[模型的截图](https://cdn.discordapp.com/attachments/998381918976479273/1372555924862140436/20250515_154222.png?ex=682733d1&is=6825e251&hm=206683cf5fa5279c41f8150a0645b58d3595bbfde5b3616003058b41a29e5cae&)。
- **GPT-5 发布日期猜谜游戏**：社区讨论了 **GPT-5** 可能的发布时间线，一些人预计在**夏季**（**6月至9月**）发布，而另一些人则认为会在 **8月至12月** 左右发布。
   - 一位成员表示：“结合目前的信息——Sam 在 2 月 12 日的公告中提到，GPT-5 将解决命名混乱的问题。”
- **Gemini 2.5 Pro 凭借百万上下文窗口赢得粉丝**：用户称赞了 **Gemini 2.5 Pro**，指出了它的编程能力、推理技巧和超大上下文窗口，有人说“Gemini 2.5 Pro 实际上非常擅长编程”，同时也承认[其免费可用性](https://discord.com/channels/974519864045756446/998381918976479273/1372558644578881576)可能因高昂的运行成本而只是暂时的。
   - 一位用户已转向使用 **Gemini 2.5 Pro**，原因是其拥有 **100 万上下文窗口**，并表示他们实在无法再忍受 **GPT 微小的 32k 上下文窗口**了。
- **OpenAI 在亚马逊寻找考古遗迹**：OpenAI 宣布了 **OpenAI to Z Challenge**，使用 **o3**、**o4-mini** 或 **GPT-4.1** 来发现亚马逊地区此前未知的考古遗址，邀请参与者在 X 上使用标签 #OpenAItoZ 分享进展。
   - 挑战详情可在 [OpenAI 官网](https://openai.com/openai-to-z-challenge/)查看。
- **社区对 Research GPT 进行测试**：一位成员请求对其处于最后完善阶段的 [Research GPT](https://chatgpt.com/g/g-68236174e57c8191aa65e6ed815b8f46-reserch-for-me) 提供反馈。
   - 创作者特别有兴趣识别英语中可能存在的问题（因为英语不是他们的母语），同时指出韩语功能表现令人满意。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Pro 计划说明**：一位成员询问了在当前的 **Cursor Pro** 计划中使用 **所有模型** 的情况，以及是否需要 **API key**。确认结果为由 Cursor 自行管理。
   - 该计划包含 Cursor 直接支持的各种模型，用户无需管理自己的 **API keys**。
- **Cursor 客户端版本统计**：一位用户分享了详细的 **Cursor** 客户端信息，包括 **VSCode 版本 1.96.2**、**Electron 34.3.4**、**Chromium 132.0.6834.210** 和 **Node.js 20.18.3**，以排查 [Cursor FAQ](https://cursor.sh/docs/faq) 中提到的“无重启”消息。
   - 这种详细程度有助于诊断与客户端配置和环境相关的特定问题，尽管也有人怀疑这与时区有关。
- **Claude 3.5 Sonnet 陷入循环**：一位用户报告称 **Claude 3.5 Sonnet** 陷入了死循环，并附带了 [证明图片](https://cdn.discordapp.com/attachments/1074847527708393565/1372432736752762930/Gq9pkJRaAAECBOI.png?ex=682769d7&is=68261857&hm=02da0d21e3c19e9a58420fc3ee21f55a47edd8f0e64e700331b1b3ae6bc99a9d&)，最终由于上下文限制而自行解决。
   - 该问题突显了模型在管理上下文和避免重复输出方面可能存在的问题。
- **静默实现的斜杠上下文重置命令**：成员们讨论了在 **Cursor** 中使用 **/reset** 命令来清除上下文，一些用户对其静默执行表示不满。
   - 澄清指出，在输入 **/reset** 后，*不会显示任何内容，它将静默执行*，这可能会让一些用户困惑命令是否真的被处理了。
- **Gemini Pro Preview 在编辑方面存在退步**：一位成员报告称 **Gemini Pro Preview** 花费大量时间决定代码更改，但在有效应用这些编辑时却表现挣扎。
   - 另一位成员指出，**0.50.4 版本** 旨在改进应用（apply）功能，暗示该问题可能在较新版本中得到解决。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **奇幻提示词令 Llama 模型受挫**：用户报告称 **Llama 3.1/3.3** 模型在给定奇幻主题提示词时会产生不理想的输出，表现为 [附带截图](https://cdn.discordapp.com/attachments/1110598183144399058/1372503571530125382/image.png?ex=6827abcf&is=68265a4f&hm=bea36a2386fe250f3a3fddbd040813929e5fdb114f8256409b4bd3d50208415a&) 中显示的 **token 丢失**、**标点符号问题** 以及 **部分单词遗漏**。
   - 这些问题突显了在特定类型的提示词下保持连贯性和准确性的持续挑战，社区尚未提供明确的解决方案。
- **提供 LM Studio Vision API 输入指南**：用户寻求关于在不使用 Python 库的情况下，通过 **LM Studio API** 向具备视觉能力的 LLM 提供图像的建议，重点在于 **OpenAI 端点**，一位用户指向了 [LM Studio 文档](https://lmstudio.ai/docs/typescript/llm-prediction/image-input)。
   - 分享了一个 cURL 示例，演示了如何将图像 URL 传递给 API，并为该用户解决了问题。
- **DRAM 对模型速度的影响次于 VRAM**：成员们讨论了 **VRAM** 与 **DRAM** 速度对模型性能的影响，结论是 **如果你将模型保留在 VRAM 中，DRAM 速度几乎无关紧要**。
   - 他们强调将模型完全保留在 **VRAM** 中以获得最佳速度和性能，因为 **KV Cache** 的位置会影响性能。
- **7900 XTX 显卡比 Nvidia 更具吸引力**：一位成员购买了 **7900 XTX** 显卡并准备卖掉他们的一张 **Nvidia** 显卡，因为他们对 **4080s** 和 **4060ti** 显卡双卡驱动的不稳定性感到有些恼火。
   - 他们提到 **5060 ti** 将作为退货处理，未提供更多细节。
- **5060 Ti 面临 AI 与游戏的需求分歧**：**8GB 版 5060 Ti** 被认为不足以处理 **AI** 任务，且对于廉价游戏配置来说也不够经济。
   - 共识建议，如果潜在买家想要使用 **AI** 模型，应该考虑 **16GB** 版本。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **AlphaEvolve: 披着 LLM 外衣？**: 成员们讨论了 **AlphaEvolve** 究竟只是一个由 LLM 驱动的 Agent 循环，还是一个演化算法，并参考了 [Google DeepMind 的博客](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)。
   - 争论焦点在于 **Gemini** 等 LLM 的强大能力与演化引擎及验证器组件重要性之间的平衡，建议查阅 *论文的消融实验部分 (ablation section)*。
- **Gemini 2.5 Pro 编程能力拔群**: 社区赞扬了 **Gemini 2.5 Pro** 的编程技能，特别是其 *Zero-shot* 表现，同时也注意到它在发布时对 *网络伦理 (cyber ethics)* 作业存在 *某种随机拒绝* 的情况。
   - 通过验证奖励进行 Fine-tuning 并拒绝非编译输出，可能是其卓越编程能力和推理能力的关键。
- **LiquidAI 面临“空中楼阁”式的质疑**: 围绕 **LiquidAI** 及其 *液体基础模型 (liquid foundational models)* 的质疑正在增加，一位成员最初将其斥为虚假宣传，而另一位成员则将 **LiquidAI** 与 **State Space Models (SSMs)** 进行了比较。
   - 社区更倾向于像 **Mamba**、**Gated DeltaNet** 或 **RWKV 7** 这样的 **SSMs** 处理方式，并指出这些模型与 [LiquidAI 的研究](https://www.liquid.ai/research/liquid-neural-networks-research) 之间存在相似之处。
- **Absolute Zero: 模型从无到有**: 社区讨论了 **Absolute Zero** 论文 [Absolute Zero](https://link.to.absolutezero)，该论文通过让 LLM 生成并验证合成训练数据，在没有任何数据的情况下改进模型。
   - LLM 被训练执行三个任务：**Y = F(X)**，**Y = F(?)**，以及 **Y = ?(X)**。
- **Samsung 意外泄露 MythoMax-L2-13B Roleplay 模型**: **Samsung** 不经意间发布了 **MythoMax-L2-13B** 角色扮演模型，该模型在 [Hugging Face](https://huggingface.co/Samsung/MythoMax-L2-13B) 上被发现后迅速被删除。
   - 一位成员开玩笑说：“*谁能给 **OpenAI** 也来这么一下，‘发布’个 **GPT4Chan**？或者 **Anthropic** 也行，那场面肯定很有趣。*”

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **用户获得快速聊天快捷方式**: 用户现在可以点击网格中的 **模型图标** 来启动与特定模型的 **Quick Chat**，优化了用户体验，如 [附图](https://cdn.discordapp.com/attachments/1092729520181739581/1372711951309738014/image.png?ex=6827c520&is=682673a0&hm=2548e2d91f09d872bd9bf58df2a1effa46101b1fef3b32a136cc594fa203e7c0) 所示。
   - 新功能允许用户通过简单点击模型图标与单个模型开始快速聊天，大大提高了效率，因为它 *跳过了打开整个分组的步骤*。
- **DeepSeek v3 是一个 MoE 模型**: [DeepSeek v3](https://deepseek.com/blog/deepseek-v3) 是一个 **Mixture of Experts (MoE)** 模型，这意味着它在推理过程中仅激活其参数的一个子集。
   - 尽管所有参数都加载到了 VRAM 中，但仅计算与 Prompt 相关的参数，从而使推理速度大大加快。
- **城市鸦科动物转向花生和猫粮**: 用户观察到 **鸦科动物**（乌鸦和喜鹊）只吃 **花生和猫粮**，而不吃普通的鸟食。
   - 有人建议，城市鸦科动物已经适应了更接近垃圾的饮食，并且由于饮食习惯经过多代演变，它们更喜欢标准鸟食以外的替代品。
- **使用 Proxy 绕过国家限制**: 一位用户分享了来自 OpenAI 的错误消息，指出其国家、地区或领土不受支持，他们通过使用 **Proxy** 绕过了这一限制。
   - 这避免了导致错误的地理限制。
- **`Qwen3` 需要切换开关来思考**: 对于 `Qwen3`，需要使用 `/think` 或 `/no_think` 强制开启或关闭思考功能。
   - 据报道 `/no_think` 功能存在 Bug，OpenRouter 需要自动路由到其他路径。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Samsung 进入 LLM 竞赛**：一名成员宣布了 **Samsung** 发布的新模型，包括 **MuTokenZero2-32B** 模型和 **MythoMax-L2-13B** 模型。
   - 另一名成员指出这些模型仍在开发中。
- **LangGraph 驱动复杂流程**：成员们分享了 **LangGraph** 文档链接（[LangGraph 课程](https://huggingface.co/learn/agents-course/unit2/langgraph/first_graph)），强调其在构建 Agentic workflows 和复杂对话流中的应用。
   - **LangGraph** 擅长监管复杂的对话路径和多 Agent 设置。
- **AsianMOM 吐槽你！**：一名成员介绍了 **AsianMOM**，这是一个 **WebGPU Vision-LLM app**，它使用 **SmolVLM (500M)** 和 **LLama 3.2 (1B)** 在浏览器中像你妈妈一样吐槽你，可在 [asianmom.kuber.studio](https://asianmom.kuber.studio/) 访问。
   - 创作者表示，这个有趣的小项目让他学到了很多关于 **WebML** 和 **Vision models** 的知识，并指出 *WebML 技术将 100% 实现 AI 访问的民主化*。
- **DistilRoberta 的准确性受到质疑**：一名成员询问为什么 **DistilRoberta** 版本的模型下载量比 **Roberta** 更多，好奇尽管存在潜在的准确性差异，它是否更适合情绪检测。
   - 另一名成员解释说，**DistilRoberta** 是 **Roberta** 的轻量级版本，旨在平衡计算成本和准确性，但理论上由于权重较少，其准确性较低。
- **Agent 课程模板触发错误**：一名成员报告说 **First_agent_template** 最初可以工作，但现在一直报错，并询问是否用完了额度。
   - 另一名成员指出 *Unit 3 的这个 Space 存在错误，需要修复* [Unit 3 Agentic RAG](https://huggingface.co/spaces/agents-course/Unit_3_Agentic_RAG)。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **小鸭睡前故事嘎嘎登场**：一名成员制作了一个 **小鸭睡前故事** 的有声读物，以适当的精力和热情朗读，并为每个角色使用不同的声音。
   - 专家讲述者是一只鸭子，只能说 **"QUACK!"**，随着有声读物的进行，音量逐渐减小。
- **Notebook LM 助力进入深度专注**：一位用户发现，在低音量下同时运行 Google 的 **Notebook LM** 选定书籍播客和 **YouTube music**，可以帮助他们在工作中进入 **深度专注 (deep focus)**。
   - 用户建议在播客选项中增加 **循环按钮**，并与 **YouTube Music** 集成以获得更丰富的体验。
- **巴基斯坦用户思考 VPN 途径**：一位用户询问该应用在 **巴基斯坦** 无法使用的问题，另一位用户建议使用 **VPN** 下载。
   - 另一位 **Android 应用测试员** 指出了工作室内部语音评论个性化的问题。
- **链接潜伏者发起潜在谎言**：用户被警告要警惕承诺 *免费礼物*、*轻松赚钱* 或 *惊人交易* 的 **诈骗链接**，并被建议在点击此类可疑链接前三思。
   - 强调提供赠品的链接是主要的危险信号，用户应始终保护个人信息。
- **播客计划初见成效**：一位用户在 NotebookLM 上达到了 **100 个播客的上限**，并打算下载 **WAV** 文件，将其转换为视频播客，并上传到 **YouTube** 或 **Google Photos**。
   - 另一位用户回复说 *这很聪明*，还有一位用户回复 *我也在做类似的事情*。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research 与 Solana Foundation 合作开展 Decentralized AI**：Nous Research 将于 **5 月 22 日**在纽约与 **Solana Foundation** 共同举办活动，重点展示通过 **Decentralized AI** 实现智能民主化的努力。注册链接见[此处](https://lu.ma/39b7e9pu)。
   - 该活动将讨论 **Psyche**，这是 Nous 专注于智能民主化的项目。
- **Psyche 进入 Hyperdrive 状态**：**Psyche** 的训练速度为**每天 12B tokens**，而处理整个 **20T tokens** 数据集预计需要近 **2000 天**。
   - 贡献者可以通过向 [Psyche Network](https://psyche.network/) 上的矿池捐赠或向 [GitHub](https://github.com/PsycheFoundation/psyche) 上的代码库做贡献来支持模型训练，这引发了对更多 GPU 算力的呼吁。
- **Meta 应对 AR 与 AI 集成**：**Meta** 在将 **AI** 集成到其智能眼镜中面临挑战，如果处理不当，可能会使其 **AR** 投资过时。
   - 尽管有此转变，**Meta** 仍通过 [Project Aria](https://www.projectaria.com/) 等项目继续 **AR** 研究。
- **智能眼镜渴求 Agentic AI**：普遍共识是，**智能眼镜**需要*真正的 Agentic AI* 来有效地解释用户环境并与之交互，正如 [Sesame](https://app.sesame.com/) 所展示的那样。
   - 成员们呼吁开发“开放的智能眼镜 AI”，以促进向更有用的集成方向创新。
- **Grok 在南非的故障**：讨论了 **Grok** 在南非的问题是源于调整后的 steering vectors 还是*笨拙的 prompt 更新*。
   - 一位成员表示：*“完全没有根据，但我投‘笨拙的 prompt’一票”*。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **TritonBench 基准测试在 AMD GPU 上失败**：一名成员发现，在 AMD GPU 上运行 [TritonBench 基准测试](https://github.com/thunlp/TritonBench/tree/main/data/TritonBench_G_v1) 时，大约有 **7 个 kernels 抛出内存访问错误**。
   - 提供的一个例子是 [chunk_gla_fwd.py](https://github.com/thunlp/TritonBench/blob/main/data/TritonBench_G_v1/chunk_gla_fwd.py)，它抛出 `Unknown Reason` 错误，该成员请求协助以确定原因。
- **CUDA IPC 内存句柄可字符串序列化**：一名成员探索了使用 `cudaIpcGetMemHandle()` 进行单 GPU 多进程通信，并发现 `cudaIpcMemHandle_t` 可以进行字符串序列化。
   - 这实现了一个简单的生产者-消费者设置来共享内存句柄，避开了单 GPU 共享时更复杂的进程间通信方法。
- **追踪 Fused Operations 需要仔细阅读代码**：一名成员询问如何在编译器融合后将 fused operations 映射回原始模型代码，另一名成员回复了关于 `inductor_provenance_tracking_node_mappings_<number>.json` 文件的[文档链接](https://docs.pytorch.org/docs/main/torch.compiler_inductor_provenance.html)。
   - 该成员不确定如何在不仔细阅读的情况下，轻松地将导出的程序图映射到原始模型代码。
- **Pipeline Parallelism 未产生并发收益**：一名成员尝试使用 `torch.autograd.graph.saved_tensors_hooks` 在独立的 **CUDA streams** 中管理 activations 以实现 **Pipeline Parallelism**，旨在实现并发的前向和后向传递，参考了[文档](https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.saved_tensors_hooks)。
   - 尽管在没有竞态条件的情况下成功实现，但由于模型的 kernel 占用率，该成员观察到的并发收益极小，认为这只是一个*“有趣的实验！！”*
- **MI300 在新基准测试中升温**：多位用户在不同的排行榜上提交了 **MI300** 的新基准测试，包括 `amd-fp8-mm` 和 `amd-mixture-of-experts`。
   - 在 `amd-fp8-mm` 排行榜上记录了多次成功的提交，**MI300** 上的时间从 **155 µs** 到 **3.28 ms** 不等；而 `amd-mixture-of-experts` 排行榜条目频繁，多位用户刷新了个人最佳成绩，例如在 **MI300** 上达到 **6233 ms** 和 **6247 ms**。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenMemory MCP 开启内存管理新篇章**：一位成员分享了 [OpenMemory MCP](https://mem0.ai/blog/introducing-openmemory-mcp/)，这是一个新的开源项目，旨在为 AI 应用提供统一的内存管理层。
   - 社区对此表示赞赏，称其为一个很酷的 AI 应用统一内存管理层。
- **持续跟踪 Grok 故障**：**Grok** 的持续问题正在 [此 Discord 频道](https://discord.com/channels/822583790773862470/1036726703730466896/1372435415490887770) 中被跟踪。
   - 未提供更多背景信息。
- **微软裁撤 TypeScript 人才**：微软在没有预警的情况下解雇了 **TypeScript** 专家，引发了沮丧情绪，详见 [这条推文](https://x.com/brodyford_/status/1922726909365879039?s=46)。
   - 社区成员表示，这次解雇是在没有任何预警的情况下发生的。
- **Agentic Tooling 超越大型科技公司**：成员们讨论了 **Agentic Tooling** 正在发生的转变，希望独立开发者能够利用它超越大型科技公司和企业。
   - 有人指出，考虑到公司的内部激励结构，“让计算机很好地做正确的事”比“很好地做错误的事”是一个更难解决的问题。
- **FUNapis 屈服于 Bing 的聊天机器人**：一位成员暗示 **FUNapis** 的消亡是为了让 **Bing** 销售其 API 的聊天机器人包装器，详见 [这条推文](https://x.com/williambryk/status/1923062696095711374?s=46)。
   - 未提供更多社区背景信息。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **优化数据加载与预处理**：一位在学术界和工业界之外独立工作的成员正在优化其数据加载和预处理流水线，以避免因工具不佳而导致资源瓶颈。
   - 他们希望这项工作能使未来所有大规模的音频数据 + Mech Interp 工作受益。
- **DNN 训练受困于数据停顿**：讨论强调了对 **基于 CPU 的预处理** 可能成为 **DNN 训练流水线** 瓶颈的担忧，特别是在音频模态背景下，并引用了论文 [Lotus: Characterization of Machine Learning Preprocessing Pipelines via Framework and Hardware Profiling](https://www.computer.org/csdl/proceedings-article/iiswc/2024/560300a030/22f0GQCjZGo)。
   - 成员们辩论了优化 CPU 工作负载与潜在 GPU 瓶颈之间的收益。
- **BlackboxNLP 进军 EMNLP 2025**：第 8 届 **BlackboxNLP** 将于今年 11 月在苏州与 [EMNLP 2025](https://blackboxnlp.github.io) 共同举办。
   - 他们将推出一项关于 **LM 中电路/因果变量定位** 的 [新共享任务](https://blackboxnlp.github.io/2025/task)，使用的是最近发布的 [MIB Benchmark](https://mib-bench.github.io/)，提交截止日期为 8 月 1 日。
- **MCQ 评估中的误报问题**：发现了一个问题，即 **MMLU** 等 **MCQ 评估** 会将模型输出标记为“错误”，即使模型根据 **NLL values** 为特定选项分配了最高概率。
   - 这一问题在较小的模型中尤为突出，表明这些模型处理多选题的方式存在潜在偏见或局限性。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 客户端寻求服务器模拟建议**：一位成员在启动客户端/服务器握手（*client -> method: initialize* 和 *server -> method initialized*）后，在模拟 **MCP server** 方面需要帮助。
   - 他们正在寻求关于正确实现 **MCP server** 中间步骤的见解。
- **Chainlit 查询参数难题**：尽管尝试使用了 FastAPI 中间件，一位成员仍难以在 **Chainlit** 中从 URL 访问**查询参数 (query parameters)**。
   - 该成员尝试传递 token 和解码后的字典，但未能成功访问，正在寻求正确获取查询参数的解决方案。
- **Jinko MCP 助力酒店 AI Agent**：社区宣布创建了 **Jinko MCP**，供开发者构建可以销售酒店的 **AI Agent**，[Jinko MCP GitHub 仓库](https://github.com/jinkoso/jinko-mcp/blob/master/README.md)现已上线。
   - 这一新工具提供了超过 200 万家酒店的访问权限，支持搜索、预订、支付和客户服务功能。
- **Smithery 服务器与 Claude Desktop 的集成问题**：一位成员在将 **Smithery 安装的服务器**与 **Claude Desktop** 使用 **OpenRouter key** 进行集成时需要帮助。
   - 该成员询问 MCP 工具配置中使用的模型是否需要与 Claude 中的模型保持一致（例如，MCP 配置中的 **sonnet-3.5** 与 Claude 中的 **sonnet 3.7**）。
- **Shortwave 支持 MCP 客户端**：根据其[博客文章](https://www.shortwave.com/blog/integrate-ai-with-all-your-apps-mcp/)，**Shortwave** 现在提供 **MCP 客户端支持**，同时支持 **HTTP MCP** 和 **stdio MCP**，并为 **Hubspot**、**Notion**、**Zapier**、**Asana** 和 **Linear** 等集成提供一键切换功能。
   - 更多详情请参阅其[官方文档](https://www.shortwave.com/docs/how-tos/using-mcp/)。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo stdlib 文档位置已明确**：在一个讨论帖中澄清，Mojo **stdlib** 文档是直接从 **stdlib** 本身生成的，可以直接修改，而不是在 `/mojo/docs/manual` 目录下。一位成员在 [PR 4530](https://github.com/modular/modular/pull/4530/files) 中修复了相关文档。
   - 这解决了成员想要更新的 [Issue 4482](https://github.com/modular/modular/issues/4482)。
- **成员们在 Mojo 中处理指针声明**：成员们在 **Mojo struct** 中声明 **Pointer** 时寻求帮助，有人建议如果想要借用 (borrow)，可以使 `Op` 对 *origin* 进行泛型化。
   - 官方澄清 Mojo 要求 *origin* 必须是类型的一部分，使其成为一个参数，这与 **borrow checker** 相关，详见 [Mojo Lifetimes 文档](https://docs.modular.com/mojo/manual/values/lifetimes/)。
- **MAX 受到安装问题的困扰**：一位成员在安装 MAX 时遇到错误，提示缺少核心功能，如**张量操作**（`tensor`、`nn`、`zeros`、`ones`、`matmul`、`add`、`mul`）。
   - 由于 Mojo 和 MAX 中的张量支持较弱，这阻碍了纯 MAX 实现的扩散模型 **LoRA 训练器**的开发。
- **目前混合使用 MAX 和 PyTorch 的方案更可行**：由于 **纯 MAX LoRA 训练器** 缺少张量操作，Claude AI 建议使用 **PyTorch** 和 **MAX 的互操作性功能** 的混合方案作为目前更可行的实现方式。
   - 一位成员确保像 Claude 这样的工具可以访问当前的 [Modular GitHub 仓库](https://github.com/modular/modular)和[文档](https://docs.modular.com)，以避免 LLM **幻觉 (hallucinations)**。
- **Karpathy 的 micrograd 被移植到 Mojo**：一位成员通过移植 **Karpathy 的 micrograd** 来学习 **Mojo**，并绕过了缺乏 lambda 函数支持的问题；另一位成员分享了他们去年创建的类似项目 [momograd](https://github.com/dorjeduck/momograd)，这是他们最早的 **Mojo** 学习项目之一。
   - **momograd** 项目尚未更新到最新的 **Mojo** 版本，但它展示了社区的兴趣。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 寻求与社区事业的合作**：一家技术相关的**非营利组织 (nonprofit)** 正在向 **Cohere** 寻求活动赞助和资助，旨在建立合作伙伴关系以推进其技术导向的倡议。
   - 有意向的各方应联系 [adibvafa.fallahpour@cohere.com](mailto:adibvafa.fallahpour@cohere.com) 以对接合适的 **Cohere** 员工。
- **Cohere Classify API 受到客户青睐**：用户对 **Cohere Classify API** 赞誉有加，表示渴望将其使用规模扩大到数百万条条目，并建议计划联系 **support@cohere.com** 请求提高速率限制 (rate limit)。
   - 提高限制将有助于探索在大规模运行 **API** 且无需长时间等待的可行性。
- **SiliconFlow 设置与截图引发关注**：一位用户在本地修改了 **SiliconFlow** 端点，并在[附图](https://cdn.discordapp.com/attachments/1218409701339828245/1372456961928331335/image.png?ex=68278066&is=68262ee6&hm=93a39c60bf386774de78fb37f4469ba201e375e9e173d6e6427db1a09ab94a1a&)中进行了展示。
   - 此外，还分享了 **Gemma 3 4b 4bit** 和 **Llama 3.2 3B 4bit** 的截图，展示了它们在不同[附图](https://cdn.discordapp.com/attachments/1218409701339828245/1372462317081591818/image.png?ex=68278563&is=682633e3&hm=fe00ba43ec32dd40fe99c2ec904aba7ae3f078b9df968075cbb55291717d751b&)和[另一张图片](https://cdn.discordapp.com/attachments/1218409701339828245/1372537263271051304/image.png?ex=6827cb30&is=682679b0&hm=f84185fbfe33d5e9af456ef71aaf4ff03339eabf81cae93d0be9a1901c188f69&)中的实现。
- **Web AI 工程师求职**：一位拥有 **7 年** 全栈经验的资深 **Web, AI 工程师** 介绍了自己，精通现代 Web 技术。
   - 他们擅长 **React(Next), React Native(Expo), Flutter, Vue(Nuxt), Svelte, Astro** 以及 **Node/Express, Django, Nest.js, Go, Web3.js, Shopify, Wordpress, TailwindCSS, Shadcn, MUI, Docker, Kubernetes, AWS/GCP, LLM** 等工具。
- **喜爱 AI 的全栈开发者青睐各类框架**：一位拥有超过 **20 年** 经验的全栈开发人员投身于 **AI**，并热衷于构建具有精心设计 **UI** 和 **UX** 的实时应用程序。
   - 他们是运行在 **Cloudflare** 上的 **Nuxt** 的粉丝，并使用 **RooCode** 和 **Windsurf** 等工具。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Gemini 模型的结构化输出已在 DSPy 中实现**：成员们讨论了 **Gemini 模型** 的响应模式（类似于 **OpenAI** 的结构化输出）是否已在 **DSPy** 中实现，另一位成员确认已经实现。
   - 同时确认 **DSPy** 会动态构建响应模式。
- **Pydantic 模型驱动 DSPy 中的结构化输出**：一位成员询问如何在 **DSPy** 中实现类似于 **OpenAI** 工具的**结构化输出**，包括**嵌套输出**或 **JSON schema 约束**。
   - 另一位成员回复称“只需使用 signatures”，并将 **Pydantic 模型**或 **Python TypedDicts** 作为输出字段类型传递。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 的生命迹象减弱**：由于自 2 月份以来一直没有更新，成员们推测 **GPT4All** 可能会停止维护。
   - 社区对 **Nomic** 缺乏关于新版本发布的沟通表示担忧。
- **Nomic 垂涎付费模式？**：关于 **Nomic** 可能转型为货币化平台的猜测浮出水面。
   - 关于“gpt4all 已完结”以及 **Nomic** 正在转向货币化的说法缺乏实质性证据。
- **Jan.ai 和 LM Studio 成为 GPT4All 的有力竞争者**：鉴于最近的担忧，**Jan.ai** 和 **LM Studio** 被提及作为 **GPT4All** 的可能替代品。
   - 讨论并未涉及这些替代品为何优秀，或者它们具有哪些可能有益的功能。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **事件驱动型 Agent 辅助 Weaviate**：LlamaIndex 展示了一个演示，说明如何使用 **event-driven agent workflows** 构建一个 **multi-agent Docs Assistant**，该助手可将网页写入 Weaviate 中的 **LlamaIndexDocs & WeaviateDocs collections**。
   - 编排器决定何时调用 Weaviate QueryAgent 进行搜索，详见 [此推文](https://twitter.com/llama_index/status/1923085124725506441)。
- **Tig 编程 Agent 首次亮相**：由 @rsrohan99 创建并基于 LlamaIndex 工作流构建的开源 **(human in the loop) coding agent** —— **Tig** 受到关注。
   - Tig 可以跨多种语言编写、调试和分析代码，执行 shell 命令并搜索网络，如 [Twitter](https://twitter.com/llama_index/status/1923134285940441102) 所示。
- **LlamaIndex 解决 PDF 内容提取问题**：一名成员请求关于使用 **LlamaParse** 或 **LlamaIndex** 从 PDF 中提取内容的建议，特别是提取目录并根据预定义名称隔离特定章节的内容和表格。
   - 该用户正在寻求关于设置指令或流水线的指导，以从 TOC 中检测章节、隔离内容并正确结构化提取的表格，并配合适用于 **n8n** 等无代码工具的正确参数。
- **AI 初创公司寻求 Vibe Coding 合作**：一家总部位于韩国的 AI 初创公司正在寻找具有 **Vibe Coding** 经验的热情开发者合作开展客户项目。
   - 该机会包括公平的收入分成模式和长期合作伙伴关系，要求具备良好的沟通能力、**GitHub 链接**、**Vibe Coding project references** 以及英文/韩文沟通能力。



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 网络在 vLLM 上受阻**：一名成员在尝试将自定义 **Torchtune network** 部署到 **vLLM** 时遇到实现失败，尽管参考了多个教程。
   - 有人建议将 checkpoint 转换为 **HF format** 以获得更好的同步效果，并询问该模型是否已在 **vLLM** 中注册。
- **自定义模型在 vLLM 中挣扎**：一名成员报告了在 **vLLM** 中实现具有自定义架构的自定义模型时遇到困难。
   - 另一名成员分享了 [关于实现自定义模型的 vLLM 指南](https://docs.vllm.ai/en/latest/contributing/model/basic.html) 以协助实现。



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Lambda 工作坊教授 Agentic AI**：一个 [Lambda 工作坊](https://www.youtube.com/watch?v=VmjMIwwo9ag) 正在教授如何使用 **Lambda's Inference API** 构建 Agentic 应用、优化 Agent 性能以及在生产环境中部署 Agent。
   - 参与者可以在 5/16 周五前通过 [此表单](https://forms.gle/UtVhmPS3mitS8Vxu7) 申请 **$100 serverless API credits**。
- **Nobel FutureTech 炉边谈话详情**：由 [Nobel FutureTech Group](https://nobel-futuretech.com/index.html) 和 Berkeley RDI 共同主办的炉边谈话提供了对 **Nobel FutureTech Genius Club** 创新生态系统的见解。
   - 该会议提供了关于导师指导、资金和协作机会的信息，并提供了 [直播链接](https://www.youtube.com/watch?v=ft-2W00Rtg8)。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Topk 悬赏要求修订**：一名用户对“移动 **topk**”的悬赏要求提出质疑，指出 **topk**、**masked_select** 和 **randperm_generator** 已经在 CPU 上运行。
   - 他们建议修订悬赏，因为 **_index_put_impl_** 和 **index_tensor** 等函数仍需关注。
- **等待 GPU 加速的索引函数**：注意到 **_index_put_impl_** 和 **index_tensor** 仍在 CPU 上运行。
   - 建议针对 torch 后端中的这些及其他函数进行 GPU 卸载。



---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **网络研讨会公告：使用 Featureform 进行 Agentic Enrichment**：一场关于 **Agentic Enrichment** 的直播网络研讨会，由 Featureform 创始人 **Simba Khadder** 主持，定于 **太平洋时间 5 月 27 日星期二上午 8 点** 举行。会议将涵盖如何使用 **MCP** 为 AI Agent 解锁数据，您可以点击[此处](https://buff.ly/zeoH55Y)报名。
   - 该研讨会将讨论 AI Agent 访问实时内部业务数据所需的基础设施缺失层，强调 Agent 面临的限制主要源于数据访问而非智能本身。
- **Featureform 解决 LLM 数据访问问题**：研讨会将讨论改善内部数据访问的需求，以释放 AI Agent 的全部潜力，并详细介绍 **Agentic Enrichment 的三个关键组件**：语义编目（semantic catalog）、低延迟服务（low latency serving）和治理（governance）。
   - 它将演示 **Featureform** 如何实现这种数据访问，使 Agent 在生产环境中更加实用和强大，并提供 AI 系统中改进工作流的真实案例。



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 发布 SWE-1 模型**：Windsurf 推出了 **SWE-1** 软件工程模型系列，包括 **SWE-1**、**SWE-1-lite** 和 **SWE-1-mini**，详情见[博客文章](https://windsurf.com/blog/windsurf-wave-9-swe-1)和[发布视频](https://youtu.be/LhA9pVpwgdY)。
   - Windsurf 声称新模型将使软件开发速度提升 **99%**。
- **SWE-1 展现 Claude 3.5 级别的性能**：据宣传，**SWE-1** 模型具有*高推理*、*工具调用能力*以及*针对 Cascade 优化*的性能，可与 **Claude 3.5** 媲美，但成本更低。
   - 这些模型使用独特的*“流感知（flow awareness）”方法*进行训练，能够理解开发界面中人类与 AI 之间的时间线。



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **AI Tinkerers 将举办 AI21 见面会**：AI Tinkerers 将与 **AI21** 共同举办见面会，讨论新发布的 [Maestro 平台](https://www.ai21.com/maestro/)，这是一个用于规划和编排的工具。
   - 见面会免费向公众开放，纽约、巴黎和旧金山的活动需要注册——请参阅上方链接。
- **AI21 发布 Maestro 规划平台**：**AI21 Labs** 最近推出了 **Maestro**，这是一个专为 AI 系统中的规划和编排而设计的平台。
   - 该平台旨在为开发者提供必要的工具和基础设施，以构建更复杂、更高效的 AI 应用。



---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间没有动态，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中[退订]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord：各频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1372427390973644820)** (869 条消息🔥🔥🔥): 

> `Perplexity Pro 角色问题，网页版无法读取 App 回答，Research 功能宕机，Deep Research 损坏，Deepsearch 速率限制` 


- **Pro 角色困扰，由管理员解决**：一名成员询问在购买 Pro 计划后如何获得 **Perplexity Pro 角色**，管理员手动解决了该问题，并表示[他们目前正在重新设计在 Discord 上获取 Pro 角色 的方式](https://discord.com/channels/1047197230748151888/1111786888626438245)。
   - 另一位用户确认，[Discord 和 Perplexity 使用相同的电子邮件并非必要条件](https://discord.com/channels/1047197230748151888/1111786888626438245)。
- **移动端 App 回答导致网页版故障**：用户报告称，从 **Perplexity 移动端 App** 获取的回答无法在网页版上读取，并会弹出错误提示；正如[这里](https://www.perplexity.ai/search/qing-jin-xing-lian-wang-sou-su-nFCGQq.lT5WtBv8y0y6oRA)所示，直接从网页获取回答时不会出现此问题。
   - 一名用户在 **10 小时前** 向客服报告了此问题，客服表示已上报给技术人员，但尚未修复。
- **Research 功能暂时下线**：一些用户询问 *Research 功能挂了吗？*，有人报告无法使用 **Perplexity**，且 [状态页面](https://status.perplexity.com/) 显示服务中断。
   - 该问题在一小时内得到修复，一些遇到账号被强制登出问题的用户发现，清除浏览器缓存可以解决问题。
- **Deep Research 完蛋了？用户质疑其价值**：多名用户报告 **Deep Research** 模式似乎已损坏，理由包括系统在网页端默认跳转到常规搜索、仅使用有限数量的来源，以及 Pro 搜索仅使用 10 个来源而非 20 个。
   - 一位用户表示：*我用 Perplexity 纯粹是因为它每次查询能读取 20-40 个来源。如果这被削减了，我觉得没理由再付费了。*
- **Grok 与 Perplexity 之争**：成员们讨论了 **Grok** 的使用，有人分享道 *我们不谈论它。在某些情况下它比常规搜索还难用*，而其他人则讨论了 **Deepsearch 速率限制 (rate limits)** 和 **响应质量**。
   - 另一位用户指出，Grok 抓取网页的能力很强，但有时在 One-shot 任务上表现很差，不过如果你详细阐述，它的速率限制补偿可以弥补这一点。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1372487492531392613)** (1 条消息): 

> `23andMe 申请第 11 章破产保护` 


- **23andMe 准备重组**：分享了一个关于 [23andMe 申请第 11 章破产保护](https://www.perplexity.ai/search/23andme-files-for-chapter-11-b-msxlvXlmQCK0MLt0UUeTcg) 的链接。
- **法律和财务重组迫在眉睫**：第 11 章申请表明 **23andMe** 正面临严重的财务挑战，并正在寻求法律保护以重组其债务和业务。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1372480085369688125)** (6 条消息): 

> `Sonar API, Perplexity 黑客松积分, sonar 模型` 


- **Sonar API 密钥问题困扰黑客松**：一名成员在为黑客松项目获取 **Sonar API** 时遇到困难，因为这需要提供信用卡详情，该成员正寻求仅用于演示目的的免费 API 访问途径。
   - 另一名成员也报告了同样的问题。
- **黑客松积分失踪？**：一名成员报告称已有 2 天未收到其 **Perplexity 黑客松积分**，并请求协助。
   - 该成员希望使用 API 来收集联系人列表的更多信息，就像在网页端操作那样。
- **Sonar API 输出不匹配**：一名成员报告称，使用 **Sonar API** 生成的答案与在 [Perplexity AI playground](https://labs.perplexity.ai/) 中生成的答案不同。
   - 他们推测这可能是由于其 System Prompt 导致的，并提供了一个指向 [Perplexity AI Model Cards](https://docs.perplexity.ai/models/model-cards) 的链接。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1372423709020655717)** (472 messages🔥🔥🔥): 

> `Manus Vibe Coding 直播, Johnny 的积分刷取, Femboys, 邀请链接消失, 积分使用情况` 


- ****Manus 举办 "Vibe Coding" 直播****：Manus 在旧金山举办了一场 "Vibe Coding with Manus" 直播，Manus SF Fellows 参与了其中，可在 [YouTube](https://youtube.com/live/w4XegM6dOgc?feature=share) 上观看。
   - 直播展示了编程项目，促进了社区互动。
- ****积分刷取天才 Johnny****：一位用户幽默地强调了 Johnny 如何 *"每天在 Manus 刷分"*，并将其与另一位 *付费让 Manus 制作 Femboy 检测器* 的用户进行对比。
   - 这描绘了利用平台获取免费积分与将其用于娱乐目的之间的幽默反差。
- ****Femboy 检测器应用****：一名用户创建了一个应用，通过使用 Function Calling 从 [wise.com](https://wise.com) 获取汇率，来判断一个人是否为 Femboy。
   - 该应用对男性名字输出 "femboy"，导致用户被贴上 Femboy 标签的喜剧性指控：*MANUS 的 API 在撒谎！*
- ****部分用户丢失邀请链接功能****：一些用户发现生成邀请链接的选项已从 UI 中消失。
   - 邀请链接生成功能似乎不再对所有用户开放。有人推测这仅影响免费用户。
- ****积分使用担忧浮现****：用户对积分成本表示担忧，其中一位指出一份 PDF 报告花费了 500 积分，而在 Google 上进行 DCF 分析花费了 1500 积分。
   - 一些人认为积分消耗太贵，尤其是对于复杂任务，并期待将 Alipay 作为支付方式。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1372430171654848522)** (401 messages🔥🔥): 

> `Gemini 2.5 Pro 推理时间, Elon 的 Grok 3.5 发布延迟, LLMs 引导注意力, LMArena 模型资金, Arena 上的 O3 Pro` 


- **Geminiyo 引发性能差距思考**：成员们在思考 [Gemini 2.5 Pro](https://ai.google.dev/gemini-api/rest/v1beta/GenerationConfig) 是否会比其他推理模型有更长的推理时间，因为它是一个 *更重的模型*。
- **Grok 3.5 发布延迟是因为 Elon 的极右翼微调？**：有人理论化认为 **Elon** 延迟发布 **Grok 3.5** 是因为将其向极右翼方向进行 Fine-tuning *并不成功*，但随后被另一名成员纠正，指出这是 *虚假信息*。
   - 另一人插话道：*他为什么要将政治内容加入 System Prompt 中？* 同时附上了一张关于利用社交网络和 LLMs 对社会进行编程的危险性的截图。
- **LLMs 无法像你想象的那样引导注意力**：尽管有说法称 **LLMs** 可以引导注意力（特别是通过 **Grok** 的 Twitter 时间线），但一名成员澄清说 *这并不是在引导你需要的注意力*，而仅仅是一个普通的公告。
   - 他们进一步链接到 [Transluce 的可观测性界面](https://transluce.org/observability-interface)，将其作为体验 Feature Steering 的工具，但提醒说它在实践中目前还不是特别有用。
- **实验室给 LMArena 提供积分而非现金**：在关于 **LMArena** 如何为其模型筹集资金的讨论中，一名成员指出 *不仅仅是公司自己支付推理费用*，这表明 **LMArena** 获得的是 **积分授权 (Credit Grants)** 而非直接的货币支付。
   - 另一名成员补充说，大型实验室为 **LMArena** 提供其模型的 Endpoints，从而收集到关于人类偏好的宝贵数据。
- **O3 Pro 登陆 Arena 仍是谜团**：尽管有关于 **O3 Pro** 发布的猜测，一名成员表示 *O3 Pro 不会来 Arena 哈哈*，对此一名管理员回复道：*我无法确认新模型是否/何时到达 Arena，但我会确保在可以的时候发布公告*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1372440346188841010)** (208 条消息🔥🔥): 

> `量化版本 (GGUFs, QNL), 多 GPU 微调 (Multi-GPU Finetuning), SLM vs LLM, 用于翻译的 Qwen3 模型, H200 温度` 


- **QNL 比标准 GGUFs 更快**：据报道 **QNL** 比标准 **GGUFs** 更快，但性能基准测试仍在进行中；[Unsloth Dynamic 2.0 GGUFs 文档](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs) 提供了更多细节。
   - 一位成员询问关于将 **GPTQv[1-2]** 与 **GGUF + imatrix** 结合以提高准确性的问题。
- **通过 Accelerate 进行多 GPU 微调**：为了实现多 GPU 微调，用户可以尝试在 Unsloth 中使用 **Accelerate**。
   - 虽然像 **3090** 这样的消费级 GPU 提供 **24GB VRAM**，但公司通常更倾向于使用 **H100s** 处理本地 AI 任务，尽管它们在数据中心语境下并非顶级配置。
- **LLMs vs SLMs**：虽然较小的模型 (SLMs) 在开箱即用时可能不够聪明，但通过针对特定任务的微调，它们可以变得具有竞争力；对于需要不错推理能力的模型，**Qwen3-4B** 是一个很好的入门选择。
   - 一位成员强调 **Qwen3 4B** 甚至比 **Mistral 7B** 更好。
- **Qwen3 翻译能力**：对于翻译中文数据集，建议使用 **Qwen3**，因为它拥有广泛的预训练数据，并建议使用 **14B** 参数版本以获得足够的性能。
   - 一位用户报告在 Kaggle 上使用 **Ollama** 运行 **30B** 模型取得了成功，并提到了处理数百万条字符串的速度需求。
- **H200 温度预期**：在 Runpod 等云环境中，**H200** 显卡的正常工作温度约为 **80-85°C**，这对于生产级显卡来说是可以接受的。
   - 一位用户报告运行温度低于 **80°C**，表明散热性能良好。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1372450131231903796)** (10 条消息🔥): 

> `微调 AI 模型, 类 Jarvis 的 AI 克隆, 在 Flappy Bird 上的连续思维机 (Continuous Thought Machine)` 


- **微调 AI 模型：Heaven 的看法**：一位成员建议，通过微调来研究 AI 模型是可行的。
   - 另一位成员表示赞同，并提到了他们在 **AI 相关项目** 中多年的经验。
- **在 YouTube 和 GitHub 上发现的 Jarvis 克隆**：一位成员表示，在 **YouTube** 和 **GitHub** 上有数十个**类 Jarvis 的克隆项目**。
   - 他们建议通过简单的 Google 搜索来找到这些项目。
- **连续思维机 (Continuous Thought Machine) 尝试 Flappy Bird**：一位成员根据 **CTM 论文** 在 **Flappy Bird** 上训练了一个**连续思维机 (CTM)**。
   - 在训练了 **约 750 个回合 (episodes)** 后，它有时只能通过*一个水管间隙*，这表明了该任务的难度。

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1372441995993288735)** (120 messages🔥🔥): 

> `Qwen3 DPO 训练, Orpheus-3B 微调问题, Mistral 7b VRAM 占用, Epoch 显示 Bug, BLIP2 与 Transformers` 


- **Qwen3 进行 DPO 训练**：一位用户询问关于使用 **DPO** 训练 **Qwen3** 的事宜，以及是否存在类似于 Zephyr 的示例 notebook。另一位用户回复并提供了[他们的 Kaggle notebook](https://www.kaggle.com/code/etherl/kaggle-llama-3-2-1b-3b-conversation-multigpu) 链接，该 notebook 用于在多 GPU 设置下微调 **Llama 3**，并指出类似的方法可能奏效。
   - 该 notebook 包含了对话模型的步骤、使用 accelerator 以及其他对 DPO 有用的优化。
- **Orpheus-3B 的 Loss 曲线与 Colab 崩溃**：一位用户报告在利用 Unsloth 微调 **Orpheus-3B** TTS 模型时，Loss 出现波动（4.5-3.9），并询问这是否正常。
   - 另一位用户回答称这是正常现象，增加训练 Epoch 可能会将 Loss 降低到 **1**，同时分享了使用 **SNAC** 进行推理的代码片段，并解决了由于账号登录访问问题导致的 Colab 崩溃。
- **GPU RAM 被 Mistral 占用**：一位用户在 NVIDIA RTX A2000 (8 GB) 上训练 **Mistral 7B** 时遇到问题，尽管 Unsloth 的基准测试显示使用 QLoRA 4-bit 应该是可行的。该用户寻求关于潜在配置错误的建议。
   - 一位用户指出 **batch size**、**r value** 和 **max_seq_length** 会显著影响 VRAM 占用，并建议用户确保没有其他进程在消耗 GPU 显存。
- **LLM Epoch 追踪器 Bug 已修复**：一位用户注意到训练输出中的差异：在完成所有步骤后，进度条显示已满，但 Epoch 计数仍停留在 **1/2**，导致用户困惑训练是否完成了两个 Epoch。
   - 另一位用户建议这可能只是一个微小的显示问题，因为样本数量和步数与完成两个 Epoch 的设定是一致的。
- **视觉模型紧跟 LLM 兼容性浪潮**：一位用户询问 **Unsloth** 是否支持 BLIP2 微调，以及如何检查 **Transformers** 是否支持它。
   - 一位用户确认 **Transformers** 支持 **BLIP2**，并引用了 [一个 PEFT notebook](https://github.com/huggingface/notebooks/blob/main/peft/Fine_tune_BLIP2_on_an_image_captioning_dataset_PEFT.ipynb) 和 [Hugging Face 文档](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/blip-2#usage-tips)，并表示 *Unsloth 几乎与任何 Transformers 模型兼容*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1372660755551486012)** (3 messages): 

> `WebGPU 视觉-LLM 应用, Geminized Qwen3 MoE` 


- **AsianMOM 在浏览器中吐槽你**：一位成员创建了 **AsianMOM**，这是一个 [WebGPU 视觉-LLM 应用](https://asianmom.kuber.studio/)，可以像你老妈一样在浏览器中吐槽你。
   - 它使用了 **SmolVLM 500M** 和 **LLama 3.2 1B**，得益于 Transformers.js 和 HF，无需安装任何内容即可在浏览器中直接运行。
- **Geminized Qwen3 MoE 发布**：一位成员发布了 [Geminized 版本的 Qwen3 MoE](https://huggingface.co/Ba2han/Qwen3-30B-A3B-Geminized-v0.2)，这是一个合并的 bf16 LoRA，在 **~450 个示例**（1 到 2 轮对话）上训练而成，其中约 250 个示例是直接来自 **Gemini2.5** 的多样化人类提示词对话。
   - *使用 "You are an assistant with reasoning capabilities." 作为 System Prompt 来触发 Gemini 风格的推理。*


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1372469766551375962)** (5 messages): 

> `Intellect-2, 独立作者, 机械可解释性伦理` 


- **Intellect-2 展示强劲性能**：一位成员分享了 **Intellect-2** 的 [ArXiv 链接](https://arxiv.org/abs/2503.15758)，暗示其表现出色并以此“炫耀”。
   - 该成员还分享了一个与该 ArXiv 论文相关的 [news.smol.ai 链接](https://news.smol.ai/issues/25-05-12-intellect-2)。
- **独立作者成就**：一位成员重点介绍了一个[个人网站](https://kvnmln.github.io/ecoart-website/CA11.1.html)，指出这是**独立作者**取得的成就。
   - 这似乎是对独立研发中所付出努力的一种致敬。
- **以机械方式解读伦理**：一位成员分享了一个名为 **Mechanistic-Interpretable Ethics-Cell automata** 的 Hugging Face Space 链接。
   - 该项目托管在 [Hugging Face Spaces](https://huggingface.co/spaces/KvnMln/Mechanistic-interpretable-Ethics-Cell-automata) 上。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1372426294259617802)** (193 条消息🔥🔥): 

> `Grok 的对齐问题、Gemini 2.5 Pro 移除、免费 API 额度、Aider Token 使用、Aider 的 Consolidate 命令` 


- **Grok 的对齐问题与 X 的笑话地位**：关于 Elon Musk 的 LLM [Grok](https://x.com) 在发表了关于 **white genocide** 的荒谬言论后引发了担忧，导致一些人对其在开发中的应用失去信任，并认为 **xAI** 是个笑话。
   - 一位成员开玩笑说，他们使用 **xAI** 只是为了询问谁是 **X** 上对民主最大的威胁，而它承认是 **Elon**。
- **Gemini 2.5 Pro 在 Copilot 中的短暂亮相**：用户注意到 **Gemini 2.5 Pro** 曾短暂出现在 Copilot 中，但随后被移除。
   - 有人推测，由于与 **OpenAI** 关系不稳，**Microsoft** 可能会在开源权重模型（open-weight models）上投入更多。
- **慷慨的 API 额度赠送**：一位用户为正在构建有趣项目的开发者提供 **Gemini, Claude, 和 OpenAI** 的免费 **API credits**，以测试新的基础设施。
   - 其他人开玩笑说免费 Token 不要白不要，或者表示有兴趣使用这些 Token 为 [aider project](https://github.com/Aider-AI/aider/issues) 做出贡献。
- **精明管理 Aider Token 使用**：用户讨论了在 Aider 中管理 Token 使用的重要性，建议使用 `/clear` 来减少上下文，并仅使用 `/add` 添加必要的文件。
   - 他们建议通过 Google AI Studio 使用 **Gemini 2.5 Flash** 等免费模型，或使用 **OpenRouter** 上的 Deepseek v3 0324，并配合从 **Gemini 2.5 Pro** 进行复制粘贴。
- **Aider 即将推出 `/consolidate` 命令**：为了响应 [LLMs Get Lost In Multi-Turn Conversation](https://arxiv.org/abs/2505.06120) 论文，一位成员计划为 aider 添加 `/consolidate` 命令，将长对话滚动合并为单个、完整指定的 Prompt，并使用主模型发起全新的单轮请求。
   - 目标是通过将之前的轮次重写为干净的 Prompt，解决 **LLMs** 在多轮对话中丢失上下文的问题。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1372425455902392391)** (99 条消息🔥🔥): 

> `Commit Prompt、速率限制、代码黑盒、aider 配置、主目录` 


- **优化 Commit 消息**：一位用户分享了一个关于 **YML configurations** 的酷技巧，通过添加 `AIDER_COMMIT_PROMPT="Respond with exactly two plain text lines and nothing else. Line 1: your commit title (max five words, no labels or prefixes). Line 2: Changes: file1,file2,... . Do not include the word Title or any markdown, headings, quotes, or extra text."` 来生成**简洁的 Commit 消息**。
   - 另一位成员分享了 Commit 消息的基础 Prompt 格式，应为 `<type>: <description>`，例如 `fix: add feature` 而不是 `added feature`。
- **配置深色模式，YAML 样式**：一位用户询问如何为代码片段启用“黑盒”显示，而不是默认的“白盒”。
   - 另一位成员引导他们查看 [配置文档](https://aider.chat/docs/config/aider_conf.html)，并建议在 `.aider.conf.yml` 文件中设置 `dark-mode: true`。
- **Gemini 的思考预算（Thinking budget）？**：一位成员询问 **Gemini** 的 **max tokens** 行为，是直接截断还是主动规划响应预算。
   - 另一位成员建议查看 `thinking_budget` 的文档，并提到可能存在相关的额外参数。
- **配置 O3 和 GPT-4.1**：一位用户询问如何在 `--architect` 模式下使用 `o3 (high) + gpt-4.1`。
   - 另一位成员提供了 [architect 文档](https://aider.chat/2024/09/26/architect.html) 的链接和示例命令：`aider --model openrouter/google/gemini-2.5-pro-exp-03-25 --editor-model openai/gpt-4.1 --architect`。
- **O3 余额不足！**：一位用户在运行命令 `aider --model openrouter/openai/o3 --editor-model openrouter/openai/gpt-4.1 --architect` 时遇到错误。
   - 结果发现，他们需要在 OpenAI 的设置以及 OpenRouter 中都有余额才能调用 O3 API。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 条消息): 

p0lyg0n: https://github.com/pig-dot-dev/muscle-mem
  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1372625656361386196)** (1 条消息): 

> `OpenAI to Z Challenge, 考古遗址, 亚马逊, GPT-4.1` 


- **OpenAI 启动亚马逊考古任务！**：OpenAI 宣布了 **OpenAI to Z Challenge**，使用 **o3**、**o4-mini** 或 **GPT-4.1** 来发现亚马逊地区此前未知的考古遗址，并邀请参与者在 X 上使用标签 #OpenAItoZ 分享进度。
   - 挑战详情可在 [OpenAI 官网](https://openai.com/openai-to-z-challenge/)查看。
- **使用 OpenAI 的最新工具探索亚马逊！**：鼓励参与者利用 **OpenAI 的 o3**、**o4-mini** 和 **GPT-4.1** 模型，在亚马逊雨林中寻找未被发现的考古宝藏。
   - 在 X 上使用标签 **#OpenAItoZ** 分享你的旅程和发现，与其他探险者建立联系并展示你的贡献。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1372432556833902612)** (195 条消息🔥🔥): 

> `GPT-4.1 Mini 智能, GPT-5 发布, Gemini 2.5 Pro, OpenAI 开源模型, Context Window 扩展` 


- **GPT-4.1 编程能力引发辩论**：成员们辩论了 **GPT-4.1** 与 **GPT-4o** 的优劣，一些人断言 **4.1** 在编程方面更胜一筹，而另一些人则认为 **4o** 整体上更直观，还有几个人声称 **4.1 mini** 是目前最好的小模型。
   - 一些用户分享了他们在特定方言下的 Prompt 测试结果和经验，以展示他们对 **GPT-4.1** 的偏好，并分享了 [模型的截图](https://cdn.discordapp.com/attachments/998381918976479273/1372555924862140436/20250515_154222.png?ex=682733d1&is=6825e251&hm=206683cf5fa5279c41f8150a0645b58d3595bbfde5b3616003058b41a29e5cae&)。
- **GPT-5 发布日期推测升温**：社区讨论了 **GPT-5** 的潜在发布时间表，一些人预计在 **夏季**（**6月至9月**）发布，而另一些人则建议在 **8月至12月** 左右发布，一位成员表示：*把这些线索串联起来——Sam 在 2 月 12 日的公告中提到 GPT-5 将解决命名混乱的问题。*
   - 成员们预计统一的 **o-models** 将在 **GPT-5** 发布时与 **GPT** 合并。
- **Gemini 2.5 Pro 赢得粉丝但面临高昂成本**：用户称赞了 **Gemini 2.5 Pro**，注意到其编程能力、推理技巧和巨大的 Context Window，有人说 *Gemini 2.5 Pro 的编程能力真的很强*，同时也承认其 [免费可用性](https://discord.com/channels/974519864045756446/998381918976479273/1372558644578881576) 可能只是暂时的，因为运行成本很高。
   - 一位用户由于 **100 万 Context Window** 而转向了 **Gemini 2.5 Pro**，并表示他们再也无法忍受 **GPT 极小的 32k Context Window** 了。
- **OpenAI 开源模型将于 6 月中旬亮相**：社区期待 **OpenAI 的开源模型** 在 **6 月** 中旬左右发布，但一些人怀疑他们是否能在本地运行它。
   - 他们预测该模型 *可能 >30B 参数但 <100B*。
- **在 Vast.ai 上竞价令人上瘾**：一位成员声称，*如果你的工作流适合，在 vast.ai 上竞价可中断算力是非常令人上瘾的*，而且他们 *以不到 1 美元/小时的价格运行着价值约 40,000 美元的 GPU，哈哈*。
   - 他们建议，如果工作流 *适合这种模式*，效果是最好的。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1372500838529892402)** (21 条消息🔥): 

> `GPT-4.1 vs GPT-4o, 用于故事生成的 Fine-tuning 数据集, GPT-4.1 中的数学` 


- **GPT-4.1 领先于 GPT-4o**：成员们讨论了新的 **GPT-4.1** 版本是否优于 **o3 mini** 和 **4o 模型**，引用了官方 [OpenAI 公告](https://openai.com/index/gpt-4-1/)，称其在 *指令遵循方面略好* 且 *记忆力更强*。
- **GPT-4.1 的真实性受到质疑**：一位用户开玩笑地暗示 **GPT-4.1** 并不存在，引用了 [Sam Altman 在 X 上的帖子](https://x.com/sama/status/1923104360243835131?s=33)。
- **为创意故事讲述进行 Fine-tuning 模型**：一位成员询问哪个模型最适合使用 **200 个故事** 的数据集进行 Fine-tuning，以便一致地创作出类似的创意故事。
- **GPT-4.1 的数学能力受到质疑**：一位用户询问 **GPT-4.1** 是否擅长数学，另一位回答说 *基本不行，除非 4o 在这方面表现好*。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1372522758575095860)** (2 messages): 

> `Research GPT 反馈，多语言能力` 


- **Research GPT 寻求反馈**：一位成员正在为其处于最后完善阶段的 [Research GPT](https://chatgpt.com/g/g-68236174e57c8191aa65e6ed815b8f46-reserch-for-me) 征求反馈。
   - 创作者特别希望发现英文方面的潜在问题，因为英文不是其母语，同时指出韩文功能表现良好。
- **GPT 的多语言能力**：该 GPT 模型在韩语下运行良好，显示出强大的多语言支持。
   - 然而，反馈请求也凸显了在不同语言间验证性能的重要性，特别是当开发者的语言熟练程度不一时。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1372522758575095860)** (2 messages): 

> `GPT 反馈，英文语言问题，韩文语言模型性能` 


- **为 Research GPT 寻求反馈**：一位成员正在为一个 [Research GPT](https://chatgpt.com/g/g-68236174e57c8191aa65e6ed815b8f46-reserch-for-me) 寻求反馈，并认为其已接近完成。
   - 他们正在进行最后检查，以消除可能残留的问题。
- **韩文 GPT 表现出色，英文 GPT 表现不佳？**：该成员指出，他们的 **Research GPT** 在 **韩文** 环境下运行得非常好，但由于英文不是其母语，他们不确定 **英文** 方面是否存在潜在问题。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1372423465621000292)** (192 messages🔥🔥): 

> `Cursor Pro 与免费版对比，客户端版本详情，Claude 3.5 Sonnet，Gemini Pro Preview，Agent 规则被忽略` 


- **关于 Cursor Pro 功能的困惑**：一位成员询问是否可以在当前计划中使用 **所有模型**，以及哪些模型需要 **API key**。
   - 另一位成员澄清说，*模型不应该要求你提供 API key，因为所有模型都由 Cursor 自身托管和支持*。
- **Cursor 客户端版本详情**：一位成员分享了他们的 Cursor 客户端详情（[版本 0.49.6](https://cursor.sh/docs/faq)），包括 **VSCode 版本 1.96.2**、**Electron 34.3.4**、**Chromium 132.0.6834.210** 以及 **Node.js 20.18.3**，并报告了一个“无重启”消息。
   - 有人怀疑这可能是因为加拿大时区的延迟。
- **Claude 3.5 Sonnet 陷入死循环**：一位成员报告称 **Claude 3.5 Sonnet** 陷入了死循环，并附带了[图片](https://cdn.discordapp.com/attachments/1074847527708393565/1372432736752762930/Gq9pkJRaAAECBOI.png?ex=682769d7&is=68261857&hm=02da0d21e3c19e9a58420fc3ee21f55a47edd8f0e64e700331b1b3ae6bc99a9d&)作为参考。
   - 最终，它被 **context limit**（上下文限制）救了回来。
- **使用斜杠重置上下文**：成员们讨论了在 Cursor 中使用 **/reset** 命令来清除上下文，但部分用户不喜欢这个功能。
   - 一位成员指出，输入 **/reset** 后，*不会显示任何内容，它会静默执行*。
- **Gemini 在编辑时遇到困难**：一位成员报告称 **Gemini Pro Preview** 花费大量时间决定代码更改，但在应用这些编辑时表现吃力。
   - 另一位成员提到 **0.50.4 版本** 承诺会改进 **apply**（应用）功能。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1372467631898230825)** (58 条消息🔥🔥): 

> `展开左侧边栏，Llama 在奇幻提示词下的问题，Token 丢失和标点符号问题，LM Studio API Vision 端点，Reka Flash 预设` 


- **寻求 Discord 边栏展开**：一位用户询问如何展开 Discord 的左侧边栏，以便在不悬停的情况下永久显示图标名称。
- **奇幻提示词导致 Llama 异常**：一位用户报告称，在使用奇幻主题的提示词时，各种 **Llama** 模型 (3.1/3.3) 会产生不理想的输出。
   - 该用户附带了 [截图](https://cdn.discordapp.com/attachments/1110598183144399061/1372503571530125382/image.png?ex=6827abcf&is=68265a4f&hm=bea36a2386fe250f3a3fddbd040813929e5fdb114f8256409b4bd3d50208415a&)，展示了 **token 丢失**、**标点符号问题**，甚至 **部分单词遗漏**。
- **探索 LM Studio 的 Vision API 端点**：一位用户寻求指导，如何在不使用 Python 库的情况下，通过 LM Studio API 向支持视觉的 LLM 提供图像，特别是在使用 OpenAI 端点时。
   - 另一位用户指向了 [LM Studio 文档](https://lmstudio.ai/docs/typescript/llm-prediction/image-input)，并强调了一个演示如何传递图像 URL 的 cURL 示例。
- **Llama.cpp 替换尝试**：一位用户询问是否可以在 LM Studio 中使用自定义构建的 *llama.cpp*，但被告知该软件是闭源的，无法替换 Llama.cpp 客户端引用。
   - 一位开发者提到，虽然替换 *llama.dll* 或许可行，但由于函数签名更改，可能会导致不稳定；全面支持“自带引擎 (bring your own engine)”已列入路线图。
- **“lm-server” 频道重获新生**：用户注意到 *lm-server* 频道被取消了，并建议 [self-hosting 频道](https://discord.com/channels/1110598183144399058/1153759714082033735) 是进行自托管的更好去处，也是处理 LM Studio 内部 API/服务器问题的场所。
   - 该频道与另一个频道重叠过多，因此被重命名并取消归档，供用户尝试并观察是否实用。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1372435960054284349)** (128 条消息🔥🔥): 

> `VRAM 与 DRAM 的重要性对比，Qwen 模型，KV Cache，7900 XTX，5060 Ti` 


- **VRAM 对模型速度起决定性作用**：有人提到，**如果将模型保留在 VRAM 中，DRAM 速度几乎无关紧要**；如果你想要合理的速度，应始终将模型保持在 **VRAM** 内。
- **探索 Qwen3 模型的 VRAM 效率**：成员建议在 **24GB VRAM** 中尝试 **Qwen3 14b q4**，并提到 **q4 量化的 30b 级别模型本身将占用约 20GB**。
   - 其他人建议，对于这种规模的模型，使用低于 q4 的任何量化都是值得商榷的，或者可以尝试 **Qwen3 30b moe q4**，但如果目标仅为 **10 t/s**，可以将其部分卸载 (offload) 到 **DRAM**。
- **KV Cache 位置影响性能**：一位成员使用了 **CPU/RAM KV Cache offload**，并指出在这种情况下 RAM 速度很重要，他们计划测试 **vram-ram GPU shared memory** 是否更好。
   - 另一位成员指出，使用 **14900K** 和 **100GB/s DDR5**，他们在 q4 量化下可以达到 **20+ t/s 的全 CPU 运行速度**。
- **7900 XTX 比 Nvidia 更具吸引力**：一位成员购买了 **7900XTX** 显卡，并准备卖掉他们的一张 **Nvidia** 显卡。
   - 他们提到对 **4080s** 和 **4060ti** 双卡驱动的不稳定性感到有些恼火，并且 **5060 ti** 将被退货。
- **5060 Ti 8GB 型号陷入尴尬境地**：讨论指出 **8GB 版 5060** 不适合 **AI**，对于追求极致性价比的游戏玩家来说也不够便宜，因此任何看中它的人不如直接购买 **16GB** 版本。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1372473982049189960)** (155 条消息🔥🔥): 

> `AlphaEvolve 分析，LLM vs. 系统角色，Gemini 2.5 Pro，LiquidAI 质疑，混合 AI 方法` 


- ****AlphaEvolve：披着风衣的 LLM？****：成员们讨论了 **AlphaEvolve** 究竟*仅仅*是一个 LLM 驱动的 Agent 循环，还是一个更复杂的带有进化算法的系统，并引用了 [Google DeepMind 的博客](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)。
   - 一些人认为其成功主要归功于像 **Gemini** 这样强大的 LLM，而另一些人则强调了进化引擎和验证器组件的关键作用，并指向了论文中的消融实验部分。
- ****Gemini 2.5 Pro：编程王牌？****：成员们称赞了 **Gemini 2.5 Pro** 的编程能力，尤其是它的 *Zero-shot* 表现，但有人指出，在被要求完成“网络伦理”作业时，它在发布初期会出现*某种随机的拒绝服务*。
   - 有推测认为，通过验证奖励进行微调并拒绝无法编译的输出，有助于提升其编程实力。这也是其推理能力的助推因素。
- ****LiquidAI：炒作还是希望？****：对于 **LiquidAI** 及其“液体基础模型 (liquid foundational models)”，人们持怀疑态度，一名成员最初对其不屑一顾。
   - 经过进一步调查，另一名成员将 **LiquidAI** 与 **State Space Models (SSMs)**（如 Mamba、Gated DeltaNet 或 RWKV 7）进行了比较，指出了相似之处，但更倾向于 SSMs 的处理方式。查看 [LiquidAI 的研究](https://www.liquid.ai/research/liquid-neural-networks-research)。
- ****AI 混合化：下一个前沿？****：讨论中出现了结合神经 (LLMs)、符号 (DreamCoder)、进化 (Novelty Search)、RL 以及生物启发式架构的混合方法。
   - 关于扩展当前方法是否足够，还是需要范式转移的混合模型来实现更高级的智能，甚至通用人工智能 (AGI)，存在着广泛争论。
- ****Absolute Zero 在无数据情况下改进模型****：成员们讨论了最近的论文 [Absolute Zero](https://link.to.absolutezero)，该方法在初始没有任何数据的情况下改进模型，它们只是让 LLM 生成所有内容并验证其正确性。
   - 在这个框架中，LLM 被训练执行三个任务：Y = F(X)，Y = F(?)，以及 Y = ?(X)。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1372564629909143622)** (3 条消息): 

> `Sakana AI，AI Scientist 论文，语言模型，推理错误，错误纠正` 


- **Sakana AI 面临质疑**：针对 **Sakana AI** 产生了怀疑，有人担心他们最新的 **AI Scientist 论文** 感觉过于侧重于营销。
- **数学语言模型可以从错误中学习**：讨论围绕论文 [Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems](https://ssrn.com/abstract=5250631) 展开，该论文探索了通过将**错误纠正**数据直接引入预训练阶段来提高推理准确性。
- **错误纠正提升推理能力**：论文指出，与在无错误数据上进行预训练相比，通过简单的自回归，在包含错误纠正数据的预训练下，语言模型能获得更高的推理准确性，详见[相关博客文章](https://physics.allen-zhu.com/part-2-grade-school-math/part-2-2)和 [YouTube 视频](https://www.youtube.com/watch?v=yBgxxvQ76_E&list=PLIZhMKKbVX6JmdngPRKvAS4u4L97odbGp&index=4)。
- **通过多轮 Prompting 进行自我纠错**：研究探讨了预训练语言模型如何通过**多轮 Prompting** *自我纠正*错误，重点在于将错误纠正数据直接整合到预训练阶段的实用性，如 [arXiv:2505.09343](https://arxiv.org/abs/2505.09343) 所述。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1372435500824133682)** (19 messages🔥): 

> `Stable Audio Open Small, MythoMax-L2-13B Samsung Release, Meta researchers leaving` 


- ****Stability AI** 与 **ARM** 发布 **Stable Audio Open Small****：[Stability AI](https://stability.ai/news/stability-ai-and-arm-release-stable-audio-open-small-enabling-real-world-deployment-for-on-device-audio-control) 和 **ARM** 发布了 **Stable Audio Open Small**，实现了设备端音频控制的实际部署。
- ****Samsung** 意外发布 **MythoMax-L2-13B****：**Samsung** 似乎意外发布了 **MythoMax-L2-13B** 角色扮演模型，该模型在 [Hugging Face](https://huggingface.co/Samsung/MythoMax-L2-13B) 上被发现后很快被删除。
   - 一位成员开玩笑说：“有人能对 **OpenAI** 也这么干一次，‘发布’ **GPT4Chan** 吗？或者 **Anthropic**，那简直太棒了。”
- ****Meta** 的人才流失被指为 **LLama 4** 进展不顺的原因**：一位成员对 **Meta** 在 **LLama 4** 上遇到的困难表示困惑，尽管他们拥有丰富的资源和 **LLama** 系列模型的成功历史。
   - 另一位成员认为，原始研究人员的离职和研究领导层的失败可能是原因，同时指出 [Thinking in Latent Space](https://www.youtube.com/watch?v=qhYQ20TbtJ8) 所属的部门与 GenAI 不同。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1372711951637024788)** (1 messages): 

> `Chatroom shortcut, Model Icons, Quick Chat` 


- **聊天室快捷方式上线**：用户现在可以点击网格中的**模型图标**，启动与特定模型的**快速聊天（Quick Chat）**。
   - 这省去了打开整个群组并手动移除其他模型的步骤，简化了用户体验，如[附图](https://cdn.discordapp.com/attachments/1092729520181739581/1372711951309738014/image.png?ex=6827c520&is=682673a0&hm=2548e2d91f09d872bd9bf58df2a1effa46101b1fef3b32a136cc594fa203e7c0)所示。
- **简化用户体验**：新功能允许用户绕过打开整个群组的步骤。
   - 用户只需点击模型图标即可开始与单个模型的快速聊天，大大提高了效率。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1372436355874951178)** (105 messages🔥🔥): 

> `DeepSeek v3 MoE, Corvids cat food and bird food, Proxy for OpenAI, AlphaEvolve, Qwen3 /no_think bug` 


- ****DeepSeek v3 是一个 MoE 模型****：据称 [DeepSeek v3](https://deepseek.com/blog/deepseek-v3) 是一个 **Mixture of Experts (MoE)** 模型，这意味着它在推理过程中仅激活其参数的一个子集。
   - 尽管所有参数都加载到 VRAM 中，但只有与 Prompt 相关的参数会被计算，从而使推理速度大大加快。
- ****鸦科动物只吃花生和猫粮****：用户观察到**鸦科动物**（乌鸦和喜鹊）只吃**花生和猫粮**，而不吃普通的鸟食。
   - 有人认为城市里的鸦科动物已经适应了更接近垃圾的饮食，并且由于多代以来的饮食习惯改变，它们更倾向于标准鸟食以外的替代品。
- ****使用 Proxy 绕过国家/地区限制****：一位用户分享了来自 OpenAI 的错误消息，指出其所在的国家、地区或领土不受支持。
   - 另一位用户建议使用 **Proxy** 来绕过导致该错误的地理限制。
- ****`Qwen3` 需要切换到 THINK 模式****：对于 `Qwen3`，需要通过 `/think` 或 `/no_think` 强制开启或关闭思考模式。
   - 据报道 `/no_think` 功能存在 Bug，OpenRouter 需要自动重定向路由。
- ****Gemini 2.5 Pro 的推理分块被认为无用****：一位用户报告称 **Gemini 2.5 Pro 的推理分块（Reasoning Chunks）** 毫无用处，称其仅显示用户的查询并确认正在为此开展工作。
   - 他们提到它只是呈现一些摘要，例如 *“用户正在询问 X。我已经针对 X 做了一些工作”*。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1372427795510067210)** (67 messages🔥🔥): 

> `Fine Tuning Llama on SageMaker, LibreChat privacy concerns, GraphQL schema code completion, Strcoder2 model distillation for Python, Emotion classification model accuracy` 


- **寻求 SageMaker Llama 微调指导**：一位成员询问如何使用 **SageMaker training** 微调 **Llama**，并请求相关教程。
   - 另一位成员提供了 [Hugging Face 文档](https://huggingface.co/docs/sagemaker/train)和[相关 GitHub 仓库](https://github.com/yuhuiaws/finetuning-and-deploying-llama-on-Sagemaker)的链接。
- **LibreChat 隐私性受质疑**：一位成员询问了在使用官方托管的 **LibreChat** ([librechat-librechat.hf.space/login](https://librechat-librechat.hf.space/login)) 时可能存在的隐私问题。
   - 另一位成员建议，这适用于典型的网站隐私考量，并指出 [LibreChat Docker 镜像](https://github.com/danny-avila/LibreChat/pkgs/container/librechat-dev)是 Hugging Face Space 实现的基础。
- **蒸馏 Starcoder2，仅保留 Python**：一位成员询问如何减小 **starcoder2** 模型的大小，使其仅专注于 **Python** 知识，从而有效地蒸馏模型。
   - 一位成员建议提取特定语言知识会很困难，并建议在 [BigCode Models Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard) 或 [The Big Benchmarks Collection](https://huggingface.co/collections/open-llm-leaderboard/the-big-benchmarks-collection-64faca6335a7fc7d4ffe974a) 上搜索更小的专业模型。
- **DistilRoberta 的情感分类准确率**：鉴于 **DistilRoberta** 模型的高普及率和下载量，一位成员对其情感分类的准确率感到困惑。
   - 他们质疑将长段落截断为模型的最大长度 512 tokens 是否会影响分析，以及句子级分析是否会更合适。
- **三星新模型涌现**：一位成员注意到 **Samsung** 发布了新模型，并分享了 **MuTokenZero2-32B** 模型和 **MythoMax-L2-13B** 模型的链接。
   - 另一位成员表示这些模型目前仍在构建中。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1372432527649800202)** (2 messages): 

> `LangGraph` 


- **为 Agent 工作流构建 LangGraph**：成员们分享了一个有用的 **LangGraph** 文档链接（[LangGraph 课程](https://huggingface.co/learn/agents-course/unit2/langgraph/first_graph)），强调了它在构建 Agent 工作流中的用途。
   - LangGraph 对于管理**复杂的对话流**和**多 Agent 系统**非常有用。
- **LangGraph 驱动对话流**：**LangGraph** 擅长监管复杂的对话路径和**多 Agent 设置**。
   - 它促进了交互的结构化管理，从而实现更稳健、更具适应性的 AI Agent 行为。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1372566805809463411)** (6 messages): 

> `Realistic Text-To-Speech, WebGPU Vision-LLM, AsianMOM, SmolVLM, Federated Learning AI` 


- **逼真的文本转语音生成器引起关注！**：一位成员分享了一个**逼真的文本转语音生成器**链接，声称它几乎与 **Dia 1.6B** 一样好，而且免费且无限制：[Hugging Face Space](https://huggingface.co/spaces/NihalGazi/Text-To-Speech-Unlimited)。
   - 该工具支持其他语言，但效果可能不如英语。
- **AsianMOM 在浏览器中吐槽你！**：一位成员介绍了 **AsianMOM**，这是一个 **WebGPU Vision-LLM** 应用，它使用 **SmolVLM (500M)** 和 **LLama 3.2 (1B)** 在浏览器中像你妈妈一样吐槽你，访问地址：[asianmom.kuber.studio](https://asianmom.kuber.studio/)。
   - 第一次尝试时安装模型可能会有点慢（大约需要 3 分钟），但它会进行缓存，所以第二次访问会快得多。
- **深入探讨 AI 访问的民主化**：**AsianMOM** 的创作者表示，这个有趣的小项目确实让他们学到了很多关于 **WebML** 和 **Vision models** 的知识，并指出 *WebML 带来的技术将 100% 实现 AI 访问的民主化*。
   - 他们为感兴趣的人分享了 [GitHub 仓库](https://github.com/Kuberwastaken/AsianMOM)。
- **联邦学习正流行**：一位成员分享了一个关于 **Rag Federated Learning AI** 的 **LinkedIn** 帖子链接 - [linkedin.com](https://www.linkedin.com/posts/nerdai_rag-federatedlearning-ai-activity-7328477143791775744-IjCO?utm_source=share&utm_medium=member_desktop&rcm=ACoAABpyymkBvdiXT4PxiTwTckoywfEnXZRbcCM)。


  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

cleonorris: 这是每月一次的，但这实际上是我们暑假前的最后一次！
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1372456933524504646)** (6 messages): 

> `DistilRoberta vs Roberta, Emotion Detection accuracy, GLoVE Paper, RobertaForTokenClassification extension, BERTopic` 


- **DistilRoberta 的受欢迎程度受到质疑**：一位成员质疑为什么 **DistilRoberta** 版本的模型下载量比 **Roberta** 更多，并好奇尽管存在潜在的准确率差异，它是否更适合情感检测（emotion detection）。
   - 另一位成员解释说，**DistilRoberta** 是 **Roberta** 的轻量级版本，旨在平衡计算成本和准确性，但理论上由于权重较少，其准确性较低。
- **模型截断困扰**：一位用户询问了 HF 模型中的文本截断问题，确认大段落会被截断至模型的最大长度（例如 **512 tokens**），并询问是否需要通过循环处理并对分数取平均值。
   - 该用户担心如果发生截断，则只有段落的一小部分会被分析。
- **GLoVE 的向量差（vector differences）引发讨论**：一位成员寻求对 **GLoVE** 论文中一句话的澄清：*“由于向量空间本质上是线性结构，最自然的方法是使用向量差”*。
   - 该成员质疑这一结论背后的逻辑，想知道它是基于启发式方法还是有具体的理论依据。
- **训练数据问题困扰情感检测模型**：一位成员分享了 [arXiv 链接](https://arxiv.org/abs/2310.12318)，指出训练数据依赖于众包标注，从而产生了学习模式的人工痕迹（artifacts）。
   - 例如，“饥饿（hungry）”这个词可能会错误地触发愤怒反应，而不管上下文如何。
- **建议使用 BERTopic 进行主题提取**：一位成员建议使用 **BERTopic** 在文本中寻找主题，而不是使用情感检测模型，并链接到了 [BERTopic 文档](https://maartengr.github.io/BERTopic/index.html)。
   - 总结者建议 **BERTopic** 可能是提取和发现文本主题的*更好解决方案*。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1372497337561124945)** (4 messages): 

> `Qwen, AI Agent course` 


- **Qwen 的使用方式受到询问**：一位成员询问另一位成员关于 **Qwen** 模型的使用情况，暗示可能需要更高级的方法。
   - 用户回复称他们正在使用 **Qwen 3** 以及基础的 prompts 和 tools，并请求获取更高级技术的指导。
- **AI Agent 课程**：一位成员提到开始学习 **AI Agent 课程**，并在使用课程模板构建他们的第一个 Agent 时遇到错误。
   - 该成员正在寻求帮助以解决他们遇到的错误。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1372464267143549010)** (10 messages🔥): 

> `Agent Template Errors, Course Completion, Final Unit Library, Certification Deadline` 


- **第一个 Agent 模板触发错误**：一位成员报告说 **First_agent_template** 最初可以运行，但现在一直报错。
   - 另一位成员建议检查额度是否用完，同时指出 *Unit 3 的这个 Space 存在错误需要修复* [Unit 3 Agentic RAG](https://huggingface.co/spaces/agents-course/Unit_3_Agentic_RAG)。
- **不付费能否完成课程？**：一位成员询问 *“有没有办法在不付费的情况下完成这门课程”*。
   - 一位成员建议在本地运行代码。
- **最终项目遇到 Token 限制问题**：一位成员在进行最终项目时用完了 tokens，导致其 API key 无法使用。
   - 他们被建议在本地运行，然而，他们随后询问如果本地运行该如何进行提交。
- **认证过程有截止日期**：一位成员质疑认证过程设定截止日期的必要性。
   - 另一位成员解释说 *“因为项目会过时”*。
- **关于最后一个单元所用库的疑问**：一位成员询问了最后一个单元使用的库以及选择该库的原因。
   - 其他成员通过鼓励他们在本地运行代码进行了回应。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1372488314246004758)** (16 条消息🔥): 

> `有声书格式, 小鸭子睡前故事, 工作专注, Pomodoro Timer, YouTube Music 集成` 


- **有声书格式讨论爆发**：成员们讨论了有声书的格式，其中一人建议*从头到尾完整地朗读/演绎源材料，不跳过或修改任何内容*。
   - 他们强调要等到每一部分结束后再进行讨论/反思，并建议即使时间不够也要读完该部分，而不是匆忙赶进度或跳过材料。
- **“小鸭子睡前故事”有声书**：一位成员创作了一本**小鸭子睡前故事**的有声书，朗读时充满活力和热情，并为每个角色使用了不同的配音。
   - 专家讲述者是一只鸭子，只能说 **“QUACK!”**，且音量随着有声书的进行而逐渐减小。
- **Notebook LM 助力工作深度专注**：一位用户发现，在低音量下同时运行 Google 的 Notebook LM 书籍播客和 YouTube Music，能帮助他们进入**工作深度专注**状态。
   - 该用户建议在播客选项中增加**循环按钮**，并与 **YouTube Music** 集成以获得更丰富的体验。
- **浸酸鸭子进入聊天**：一位用户发布了消息 *dip your genitals in acid*，随后附带了一个鸭子表情符号。
   - 紧接着是一个名为 **The_Duck_Dive_-_History.wav** 的附件，我们无法对其进行分析。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1372426744690245634)** (76 条消息🔥🔥): 

> `巴基斯坦的应用可用性, 移动端应用方案限制, 音频生成限制, NLM 处理表格数据, 诈骗链接警示` 


- **巴基斯坦访问困境**：一位用户询问该应用在**巴基斯坦**无法使用的问题，另一位用户建议使用 **VPN** 进行下载。
   - 另一位 **Android 应用测试员**指出了工作室内部语音评论个性化的问题。
- **移动端应用的盈利模式**：一位用户质疑移动端应用在没有 **$20/月** 订阅的情况下每日音频概览的限制，引发了关于付费墙的讨论。
   - 另一位用户对**每天 3 条音频**的生成限制表示遗憾，抱怨 *Google 太贪婪了*，而另一位用户则回应称这种想法才是贪婪的，因为这是一个免费产品。
- **播客计划初见成效**：一位用户达到了 NotebookLM 的 **100 个播客上限**，并打算下载 **WAV** 文件，将其转换为视频播客，然后上传到 **YouTube** 或 **Google Photos**。
   - 另一位用户回复说 *这很聪明*，还有用户回复 *我也在做类似的事情*。
- **表格问题困扰工具使用**：一位用户询问如何向 NLM 提供表格数据，但另一位用户建议 **NLM** 并不是处理表格数据的理想选择。
   - 该用户被建议考虑使用 **SQL** 或 **Google Sheets** 中的 **AI formula**，或者使用 **Gemini with BigQuery**。
- **链接潜伏者散布谎言**：用户被警告要警惕承诺*免费礼物*、*轻松赚钱*或*惊人交易*的**诈骗链接**，并被建议在点击此类可疑链接前三思。
   - 会议强调，提供免费赠品的链接是重大危险信号，用户应始终保护个人信息。


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1372694625218854932)** (1 条消息): 

> `Solana Foundation, Decentralized AI, Psyche` 


- **Nous Research 与 Solana Foundation 在纽约举办活动**：Nous Research 将于 **5 月 22 日**在纽约与 **Solana Foundation** 共同举办一场活动，重点关注 **Decentralized AI**。
   - 活动将涵盖 Nous 在推动智能民主化方面的努力，包括 **Psyche** 项目；注册链接见[此处](https://lu.ma/39b7e9pu)。
- **Psyche 推动智能民主化**：**Psyche** 是 Nous 的一个项目，旨在通过 Decentralized AI 的努力使智能民主化。
   - 该项目的目标和进展将在即将举行的 Solana Foundation 联合活动中进行讨论。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1372466405898715197)** (85 messages🔥🔥): 

> `Psyche Training, Meta AR, Smart Glasses, Grok crashing` 


- ****Psyche** 训练进入极速模式**: **Psyche** 的训练速度目前达到 **每天 12B tokens**，据估计处理整个 **20T tokens** 需要近 **2000 天**，这引发了对更多 GPU 的需求。
   - 贡献者可以通过向 [Psyche Network](https://psyche.network/) 上的矿池捐赠或在 [GitHub](https://github.com/PsycheFoundation/psyche) 上贡献代码来支持模型训练。
- ****Meta** 应对 **AR** 与 **AI** 的交汇挑战**: **Meta** 面临着将 **AI** 集成到其智能眼镜中的挑战，如果无法适应，其 **AR** 投资可能会变得过时。
   - 尽管发生了转变，他们仍通过 [Project Aria](https://www.projectaria.com/) 等项目继续 **AR** 研究，在当前的 **AI** 功能限制与持续进展之间寻求平衡。
- ****智能眼镜**需要 **AI****: 成员们建议**智能眼镜**需要*真正的 Agentic AI* 才能有效地解释用户环境并与之交互。
   - 来自 [Sesame](https://app.sesame.com/) 的演示展示了一家智能眼镜公司如何向实用的 Agentic AI 创新，并呼吁开发*开源智能眼镜 AI*。
- ****Grok** 在南非出现故障：**AI** Prompt 问题？**: 讨论围绕 **Grok** 在南非的问题是源于微调的 steering vectors 还是*笨拙的 Prompt 更新*。
   - 一位成员表示：*完全没有根据，但我投票给笨拙的 Prompt*。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1372658113781432450)** (1 messages): 

> `TritonBench, AMD GPU Errors, Memory Access Fault` 


- **TritonBench 在 AMD GPU 上抛出内存访问故障**: 一位成员正在进行 [TritonBench benchmark](https://github.com/thunlp/TritonBench/tree/main/data/TritonBench_G_v1) 测试，发现约有 **7 个 kernel 在 AMD GPU 上运行时抛出内存访问故障 (memory access fault) 错误**。
   - 提供的一个例子是 [chunk_gla_fwd.py](https://github.com/thunlp/TritonBench/blob/main/data/TritonBench_G_v1/chunk_gla_fwd.py)，它抛出了 `Unknown Reason` 错误，该成员请求协助以查明原因。
- **寻求 AMD GPU 内存访问故障的帮助**: 用户在 AMD GPU 上运行 Triton kernel 时遇到内存访问故障错误。
   - 具体而言，用户寻求协助以识别内存访问违规的根本原因，怀疑这与访问所提供代码中越界的内存位置有关。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1372433414048977027)** (3 messages): 

> `cudaIpcMemHandle_t Serialization, Single GPU Multiprocess Communication` 


- **CUDA IPC 内存句柄变得简单**: 一位成员探索了在 PyTorch dataloaders 不可行的情况下，使用 `cudaIpcGetMemHandle()` 进行单 GPU 多进程通信。
   - 他们注意到 `cudaIpcMemHandle_t` 可以进行字符串序列化，从而实现一个简单的生产者-消费者设置来共享内存句柄。
- **序列化简化了 GPU 数据共享**: 用户发现 `cudaIpcMemHandle_t` 可以进行字符串序列化。
   - 这允许在单 GPU 的进程之间通过简单的生产者-消费者设计来共享这些句柄，避开了更复杂的进程间通信方法。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1372573625429004321)** (12 条消息🔥): 

> `映射融合操作 (Mapping Fused Operations)、使用 torch.autograd.graph.saved_tensors_hooks 的流水线并行 (Pipeline Parallelism)、vLLM V1 中的自定义 CUDA 图与缓存问题、GEMM 代码生成性能与原生 aten 实现对比、torch.compile 模式基准测试` 


- **将融合操作溯源至源代码**：一位成员询问如何在编译器融合后将融合操作映射回原始模型代码，旨在精确识别哪些操作被融合了。另一位成员回复了一个指向 [文档](https://docs.pytorch.org/docs/main/torch.compiler_inductor_provenance.html) 的链接，涉及 `inductor_provenance_tracking_node_mappings_<number>.json` 文件。
   - 该成员不确定如果不仔细阅读，如何轻松地将导出的程序图映射到原始模型代码。
- **使用 CUDA 流实验流水线并行**：一位成员尝试使用 `torch.autograd.graph.saved_tensors_hooks` 在独立的 CUDA 流中管理激活值，以实现流水线并行，目标是并发执行前向和反向传播，参考了 [文档](https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.saved_tensors_hooks)。
   - 尽管在没有竞态条件的情况下成功实现，但该成员观察到由于模型的 Kernel 占用率（occupancy）问题，并发收益微乎其微，不过他认为这仍然是一个 *"有趣的实验！！"*。
- **vLLM V1 中的自定义算子缓存故障**：一位成员在 vLLM V1（使用 torch.compile + 自定义 CUDA Graphs）中遇到了与缓存相关的克隆错误，涉及一个调用 `f2()` 的自定义算子 `f1()`，该算子从缓存中采样，随后抛出了 `RuntimeError`。
   - 错误指出 *"该自定义算子的输出 (1) 不得同时作为该自定义算子的输入，且 (2) 不得与任何输入或其他返回值存在别名关系"*，该成员询问如何在不进行克隆的情况下绕过此限制。
- **GEMM 代码生成比原生 aten 慢？**：一位成员对 GEMM 操作的不同 `torch.compile` 模式进行了基准测试，将 `f_compile`、`f_overhead` 和 `f_max` 与非编译版本进行了对比，输入尺寸为 `N = 2_000` 和 `B = 100_000`。
   - 结果显示 `f_compile` (fullgraph=True) 的耗时为 **10.43 ms**，略快于其他模式（分别为 **12.06 ms**、**12.09 ms** 和 **12.56 ms**），这表明设备端的 Reduce 操作是瓶颈。
- **Nvidia-smi 显示显存情况**：一位成员询问 `nvidia-smi` 显示的是已分配（allocated）还是已预留（reserved）的显存，并对相对于 VRAM 容量较高的预留显存表示担忧，同时上传了一张 [nvidia-smi 的图片](https://cdn.discordapp.com/attachments/1189607750876008468/1372651600891613255/9NMqqTPCSzgAAAABJRU5ErkJggg.png?ex=68278cec&is=68263b6c&hm=3b3913414e790760f018d6eab9b2d24854f387c59c354f76c4e57a9f2dbdbcb1)。
   - 另一位成员澄清说 `nvidia-smi` 无法洞察 torch 内部情况，且预留显存不可能超过 VRAM。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 条消息): 

real.optimus.prime: 来自 DeepSeek: 
https://arxiv.org/abs/2505.09343
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1372494117665505342)** (6 条消息): 

> `CUDA SASS 取反、CCCL/libcu++ 向量类型、CUDA 编译标志` 


- **CUDA SASS 取反是免费的吗？**：据成员称，当 CUDA 代码编译为类似 `FLO.U32 R4, R4 IADD32I R4, -R4, 0x1f` 的 SASS 时，寄存器 R4 的取反操作是 **没有任何额外开销** 的。
- **CCCL/libcu++ 正在处理向量类型**：**CCCL/libcu++** 正在为向量类型实现 **tuple protocol**，这应该能很好地配合模板展开（unrolling templates）。
   - 然而，目前尚不确定它是否会包含 CUDA 中未提供的类型（如 `char16`），以及是否仍会保留 `.x` 命名约定。
- **启用 CUDA 编译的额外标志？**：一位成员询问是否需要 *编译中的额外标志*，并链接到了 [NVIDIA 关于 CUDA 7 流简化并发的博客](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/)。


  

---

### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1372447432209596497)** (1 条消息): 

> `GitHub Repository, Lecture Scripts` 


- **讲座观众寻找 GitHub Repo**：一位最近讲座的观众询问了相关的 **GitHub repository** 以及提到的其他资源的可用性。
   - 他们请求了仓库链接，并指出尽管查看了可用资源，但仍找不到脚本。
- **讲座资源查询**：一位观看讲座的用户正在寻找相关脚本的 **GitHub repository link**。
   - 该用户提到他们检查了 **GitHub repository** 和讲座中提到的其他资源，但无法找到脚本。


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1372579921188360234)** (3 条消息): 

> `Tensor Processing Unit (TPU)` 


- **TPU 定义已提供**：一位成员询问什么是 **TPU**，另一位成员回答说它是 *tensor processing unit*，基本上是针对 **AI workloads** 优化的加速器。
- **TPU vs CPU**：TPU 的设计旨在比 CPU 更好地加速机器学习工作负载。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1372424337805545525)** (40 条消息🔥): 

> `MI300, amd-fp8-mm, amd-mixture-of-experts` 


- **MI300 基准测试迎来新提交**：多位用户在不同排行榜上提交了 **MI300** 的新基准测试，包括 `amd-fp8-mm` 和 `amd-mixture-of-experts`。
   - 一位用户在 `amd-mixture-of-experts` 排行榜上取得了 **6209 ms** 的个人最好成绩。
- **amd-fp8-mm 排行榜竞争激烈**：`amd-fp8-mm` 排行榜记录了多次成功提交，在 **MI300** 上的时间范围从 **155 µs** 到 **3.28 ms**。
   - 一位用户还在 **MI300** 上记录了 **2.42 ms** 的个人最好成绩。
- **amd-mixture-of-experts 出现新纪录**：`amd-mixture-of-experts` 排行榜出现了频繁的条目，多位用户在 **MI300** 上取得了个人最好成绩，例如 **6233 ms** 和 **6247 ms**。
   - 一位用户在 **MI300** 上达到了 **32.7 ms**。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1372588989785182209)** (3 条消息): 

> `Cutlass, fp8, bf16, narrow precision dtypes` 


- **对 Cutlass 表示感谢**：一位用户感谢了 **Cutlass** 团队的工作，并表示很高兴开始使用它进行开发。
   - 该用户随后询问了“显著的不支持功能部分”，以及目前不支持哪些 **dtypes**，例如 **fp8** 和 **bf16**。
- **Cutlass 已支持 fp8**：最新的 Cutlass 已支持 **Fp8**。
   - 团队澄清说，*narrow* dtypes 指的是 *sub byte and micro scale types*，并且即将推出。


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/)** (1 条消息): 

clattner: 你们可能会觉得这个技术演讲很有趣：https://www.youtube.com/watch?v=Invd_dxC2RU
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1372424394860662795)** (50 条消息🔥): 

> `OpenMemory MCP, Grok issues, Microsoft fires TypeScript dude, Agentic tooling, FUNapis` 


- **OpenMemory MCP 开放内存**：一位成员分享了 [OpenMemory MCP](https://mem0.ai/blog/introducing-openmemory-mcp/) 的链接，称其非常酷。
   - 这是一个新的开源项目，旨在为 AI 应用程序提供统一的内存管理层。
- **持续追踪 Grok 的问题**：**Grok** 的持续性问题正在 [此 Discord 频道](https://discord.com/channels/822583790773862470/1036726703730466896/1372435415490887770) 中被追踪。
- **微软解雇 TypeScript 人才**：围绕微软在没有预警的情况下解雇了 **TypeScript** 大佬的讨论引起了关注，引发了沮丧的情绪，详见 [这条推文](https://x.com/brodyford_/status/1922726909365879039?s=46)。
- **Agentic Tooling 超越大型科技公司**：成员们讨论了 **agentic tooling** 正在发生的转变，希望独立开发者能利用它超越大型科技公司和企业。
   - 有人建议，“让计算机很好地做正确的事”仍然比“很好地做错误的事”更难解决，特别是考虑到公司内部的激励结构。
- **FUNapis 屈服于 Bing 的聊天机器人**：一位成员暗示 **FUNapis** 的终结是为了让 **Bing** 能够销售他们基于该 API 的聊天机器人封装产品，详见 [这条推文](https://x.com/williambryk/status/1923062696095711374?s=46)。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1372509787350106214)** (39 条消息🔥): 

> `Knowledge graphs for papers, Cloud GPU/HW providers, MLPerf training benchmark, Data stalls in DNN training, Audio modality preprocessing` 


- **用于知识图谱的 **Connected Papers****：一位成员分享了 [Connected Papers](https://www.connectedpapers.com/) 的链接，作为创建**研究论文知识图谱**的资源。
   - 该工具旨在可视化和探索特定领域内不同论文之间的联系，对文献综述和研究非常有用。
- **寻求开源开发的云端 GPU 指导**：一位成员正在寻求适合开源开发的**云端 GPU/硬件供应商**建议，特别是为了设置 **MLPerf 训练基准**。
   - 他们特别关注可能为学生提供免费计算时长或对开源项目有优惠条件的选项。
- ****数据停顿困扰 DNN 训练****：讨论围绕 **CPU 预处理**是否是 **DNN 训练流水线**中的瓶颈展开，特别是在音频模态背景下。
   - 一位成员引用了一篇关于该主题的论文（[Lotus: Characterization of Machine Learning Preprocessing Pipelines via Framework and Hardware Profiling](https://www.computer.org/csdl/proceedings-article/iiswc/2024/560300a030/22f0GQCjZGo)），其他人则质疑如果 GPU 成为瓶颈，优化 CPU 工作负载是否有益。
- ****优化数据加载和预处理**以避免未来的困扰**：一位成员分享说，由于他们在学术界或工业界之外独立工作，他们正在优化其数据加载和预处理流水线，以避免因工具链不佳而导致资源受限。
   - 他们希望这项工作能惠及未来所有大规模的音频数据 + mech interp（机械可解释性）工作。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1372445144338534501)** (3 条消息): 

> `BlackboxNLP, Interpretability, Causal Variable Localization, MIB Benchmark` 


- ****BlackboxNLP** 进驻 EMNLP 2025**：第 8 届 **BlackboxNLP**（NLP 领域神经网络可解释性和分析的领先研讨会）将于今年 11 月在苏州与 [EMNLP 2025](https://blackboxnlp.github.io) 共同举办。
   - 他们将推出一项关于 **LM 中电路/因果变量定位**的[新共享任务](https://blackboxnlp.github.io/2025/task)，该任务使用最近发布的 [MIB Benchmark](https://mib-bench.github.io/)，提交截止日期为 8 月 1 日。
- **用于因果变量定位的 **MIB Benchmark****：一项使用最近发布的 [MIB Benchmark](https://mib-bench.github.io/) 的新共享任务将专注于 **LM 中的电路/因果变量定位**。
   - 该任务的提交截止日期为 **8 月 1 日**，在 BlackboxNLP 研讨会中提供了一个专注的挑战。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1372481613845237781)** (2 条消息): 

> `MCQ Evaluations, MMLU Issues, Model Outputs` 


- **MCQ 评估输出被标记为错误**：一位成员报告了 **MMLU** 等 **MCQ 评估**中的一个问题：即使模型根据 **NLL 值**为特定选项分配了最高概率，模型输出仍被一致标记为 *false*（错误）。
   - 在较小的模型中观察到了这一问题，所有四个选项都被标记为 *false*。
- **较小模型显示全错输出**：**MCQ 评估**中全 *false* 输出的问题在较小模型中似乎更为普遍。
   - 这表明这些模型在处理多选题时可能存在潜在偏差或局限性，特别是在通过 **NLL 值**评估概率时。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1372644184837718076)** (2 条消息): 

> `LLaVAGuard, SafeDiffuser, Multimodal Models` 


- **寻求多模态模型论文**：一位成员请求推荐近期值得关注的**多模态模型**论文，如 [LLaVAGuard](https://github.com/LAION-AI/LLaVAGuard) 或 [SafeDiffuser](https://github.com/safeai-lab/safe-diffuser)。
- **关于频道适用性的澄清**：该成员询问他们的问题是否更适合在 *image-models* 频道中讨论。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1372454914252607509)** (33 messages🔥): 

> `MCP Client-Server Call Flow, Chainlit Query Parameters, Jinko MCP for Hotel Sales, Smithery Server and Claude Desktop, Understanding MCP Resources` 


- **MCP 客户端-服务器调用流仿真请求**：一位成员询问了 **MCP client** 与 **MCP server** 之间的调用流，旨在仿真服务器，并详细说明了初始步骤，如 *client -> method: initialize* 和 *server -> method initialized*。
   - 该成员寻求关于中间步骤的建议，并指出需要了解如何实现 MCP server 以成功仿真该协议。
- **Chainlit 查询参数探索**：一位成员在尝试了 FastAPI 中间件等解决方案后，仍面临无法从 **Chainlit** 的 URL 中获取 **query parameters**（查询参数）的挑战。
   - 他们尝试传递 token 和解码后的字典，但无法在 Chainlit 内部访问它们，并寻求在 Chainlit 中访问查询参数的解决方案。
- **Jinko MCP 助力酒店销售**：一位成员宣布为开发者创建了一个 **MCP**，用于构建想要销售酒店的 **AI agents**。
   - 该 MCP 提供了超过 **200 万家酒店** 的访问权限，支持搜索、预订、支付和客户服务，更多详情请参阅 [Jinko MCP GitHub 仓库](https://github.com/jinkoso/jinko-mcp/blob/master/README.md)。
- **Smithery 服务器在 Claude 上的使用困扰**：一位成员在添加了 **OpenRouter key** 后，寻求关于在 **Claude Desktop** 中使用 **Smithery 安装的服务器** 的指导。
   - 该成员询问 MCP 工具配置中使用的模型是否需要与 Claude 中使用的模型匹配（例如，MCP 配置中的 **sonnet-3.5** 与 Claude 中的 **sonnet 3.7**）。
- **Resources 被阐释为 GET 请求**：一位成员寻求帮助，以解释 **MCP** 背景下的 **resource** 是什么，并提到研讨会参与者对此感到困惑，尝试将其阐释为 *“MCP 的 GET 请求”*。
   - 在建议将文件拖入 Cursor 聊天有助于理解后，Resources 被解释为具有 URI 的对象，例如 `datetime://Euroupe/London/now` 和 `http://example.com/llms.txt`。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1372554582567157811)** (8 messages🔥): 

> `LLM Agent to MCP Server Connection, MCP for AI Agents Selling Hotels, MCP Democratizes Apache Kafka Usage, macos-automator-mcp for Autonomous Debugging, AI and MCP Language Barriers` 


- **LLM Agent 与 MCP Server 挂接**：一位成员分享了一篇关于 *如何将你的 LLM Agent 连接到 MCP Server* 的博客文章，阅读地址见[此处](https://sandipanhaldar.com/blog/part-1-how-to-connect-your-llm-agent-to-mcp-server.html)。
- **AI Agents 现在可以通过 MCP 预订酒店**：一位成员为开发销售酒店的 **AI agents** 的开发者发布了一个 **MCP**，提供 200 万+ 酒店的搜索、预订、支付和客户服务访问权限，并附带了 [GitHub 仓库](https://github.com/jinkoso/jinko-mcp/blob/master/README.md)链接。
- **Kafka 通过 MCP 实现平民化！**：一位成员讨论了 **MCP** 如何让 **Apache Kafka** 的使用变得平民化，允许通过自然语言提示词与实时数据进行交互，并附带了 [YouTube 视频](https://www.youtube.com/watch?v=ivlzvZzFeZMS)。
- **macos-automator-mcp 首次推出自主调试功能！**：一位成员介绍了 **macos-automator-mcp**，它使 Cursor 等工具能够控制系统功能以实现完全自主的调试，并提供了 [GitHub 链接](https://github.com/steipete/macos-automator-mcp)。
- **Shortwave 发布 MCP Client 支持！**：一位成员宣布在 **Shortwave** 中推出 **MCP client 支持**，同时支持 **HTTP MCP** 和 **stdio MCP**，并为 **Hubspot**、**Notion**、**Zapier**、**Asana** 和 **Linear** 等集成提供一键切换功能，详见其[博客文章](https://www.shortwave.com/blog/integrate-ai-with-all-your-apps-mcp/)和[文档](https://www.shortwave.com/docs/how-tos/using-mcp/)。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1372570639973482598)** (5 messages): 

> `Documentation updates, stdlib modifications` 


- **通过 PR 修复文档**：一位成员报告发现了 [Issue 4482](https://github.com/modular/modular/issues/4482) 并打算更新文档。
   - 另一位成员表示他们已在 [PR 4530](https://github.com/modular/modular/pull/4530/files) 中修复了该文档。
- **文档由 stdlib 自身生成**：一位成员询问在哪里修改 stdlib 文档，假设位置在 [modular/modular/tree/main/mojo/docs/manual](https://github.com/modular/modular/tree/main/mojo/docs/manual)。
   - 另一位成员澄清说文档是从 stdlib 自身生成的，因此可以直接在代码中修改。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1372635413272858645)** (9 messages🔥): 

> `Mojo 结构体中的指针声明、基于 origin 的 Mojo 泛型、Mojo 生命周期、将 Karpathy 的 micrograd 移植到 Mojo、Jeff 在 Modular 的演讲` 


- **Mojo 结构体中指针声明的难题**：一位成员寻求在 **Mojo struct** 中声明 **Pointer** 的帮助，引用了来自 [kapa.ai](https://kapa.ai) 的示例，并随后澄清 *origin* 是存储 **Pointer** 的结构体。
   - 另一位成员建议，如果他们想要进行借用（borrow），应该让 `Op` 针对 origin 进行泛型化。
- **Origin 与 Mojo 借用检查器揭秘**：一位成员解释说，**Mojo** 要求 *origin* 必须是类型的一部分，使其成为一个参数。
   - 另一位成员澄清说，*origin* 与 **borrow checker**（借用检查器）绑定，确保如果其指向的数据被移动或释放，该指针将不再处于活跃状态，并提供了官方 [Mojo Lifetimes 文档](https://docs.modular.com/mojo/manual/values/lifetimes/)的链接。
- **将 Micrograd 移植到 Mojo**：一位成员正在通过移植 **Karpathy 的 micrograd**（一个简单的 Python 示例）来学习 **Mojo**，目前正在设法绕过 **Mojo** 尚不支持 lambda 函数的问题。
   - 另一位成员分享了他们类似的项 [momograd](https://github.com/dorjeduck/momograd)，这是他们去年作为首批 **Mojo** 学习项目之一创建的，不过尚未更新到最新的 **Mojo** 版本。
- **Jeff 在 Modular 的演讲令成员惊叹**：一位成员对以往由 Jeff 主讲的 **Modular** 演讲表示热烈欢迎。
   - 他们分享了[这个 YouTube 播放列表](https://www.youtube.com/playlist?list=PLh0S94-sJw_6ygGMynvQkt32IwBJM4DBW)，并很高兴能再次看到此类内容。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1372601560177311804)** (11 messages🔥): 

> `MAX 安装问题、LoRA 训练器难题、Mojo 张量支持薄弱、MAX 与 PyTorch 混合方案、Modular 平台的 LLM 幻觉` 


- **MAX 安装受困于功能缺失**：一位成员在安装 MAX 时遇到错误，提示缺少基本功能，如 **张量操作** (`tensor`, `nn`, `zeros`, `ones`, `matmul`, `add`, `mul`)。
   - 尽管尝试了本地安装，所需功能仍然缺失，导致无法继续进行纯 MAX 的实现。
- **LoRA 训练器面临张量支持短板**：用户报告称，由于 Mojo 和 MAX 的张量支持薄弱，在创建扩散 **LoRA 训练器**时遇到困难，导致他们放弃了最初纯 Mojo 的方案。
   - 虽然 PyTorch 版本可以运行，但目标是避开 PyTorch，这凸显了 MAX 在张量操作方面的局限性。
- **混合使用 MAX 和 PyTorch 作为权宜之计**：Claude AI 建议，由于缺少张量操作，目前实现**纯 MAX 的 LoRA 训练器**是不可行的。
   - 相反，建议采用结合 **PyTorch** 和 **MAX 互操作性功能**的*混合方案*，作为目前更可行的解决方案。
- **LLM 在缺乏适当上下文时会产生幻觉**：一位成员建议确保 Claude、Cursor 等工具能够访问当前的 [Modular GitHub 仓库](https://github.com/modular/modular)和[文档](https://docs.modular.com)。
   - 由于 Mojo 和 MAX 相对较新，且在 LLM 训练数据中代表性不足，如果没有适当的上下文，这些工具可能会产生**幻觉**并生成错误的建议。


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1372581567376916554)** (3 messages): 

> `Cohere、赞助、资助、非营利组织` 


- **Cohere 员工联系方式以获取赞助和资助**：一位成员请求联系合适的 **Cohere** 员工，以支持/赞助其技术相关非营利组织的活动和/或提供**资助**。
   - 一位员工做出了回应，提供了他们的电子邮箱 [adibvafa.fallahpour@cohere.com](mailto:adibvafa.fallahpour@cohere.com)，以便将咨询转发给相关负责人。
- **技术非营利组织寻求与 Cohere 合作**：一家技术相关的**非营利组织**正寻求与 **Cohere** 合作，以获得活动赞助和资助。
   - 联系 [adibvafa.fallahpour@cohere.com](mailto:adibvafa.fallahpour@cohere.com) 以对接相关工作人员。


  

---

### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1372614679418900490)** (6 messages): 

> `Cohere Classify API, Rate Limit Increase` 


- **Cohere Classify API 给用户留下深刻印象**：用户对 **Cohere Classify API** 表示满意，并有兴趣将其使用规模扩大到数百万条目。
   - 然而，他们对当前的速率限制（Rate Limits）感到担忧，并正在寻找加快处理过程的方法。
- **通过提升速率限制扩展 Cohere API**：一名成员建议联系 **support@cohere.com** 以请求提高 **Cohere Classify API** 的速率限制。
   - 这种方法应该有助于确定在大规模运行 API 时，无需长时间等待的可行性。


  

---


### **Cohere ▷ #[💡-projects](https://discord.com/channels/954421988141711382/1218409701339828245/1372456961899106415)** (3 messages): 

> `SiliconFlow, Gemma 3 4b 4bit, Llama 3.2 3B 4bit` 


- **SiliconFlow 端点已修改**：一位用户正在使用 **SiliconFlow** 并将端点修改为 **localhost**，如[附图](https://cdn.discordapp.com/attachments/1218409701339828245/1372456961928331335/image.png?ex=68278066&is=68262ee6&hm=93a39c60bf386774de78fb37f4469ba201e375e9e173d6e6427db1a09ab94a1a&)所示。
- **分享 Gemma 3 4b 4bit 截图**：一位用户发布了 **Gemma 3 4b 4bit** 的截图，如[附图](https://cdn.discordapp.com/attachments/1218409701339828245/1372462317081591818/image.png?ex=68278563&is=682633e3&hm=fe00ba43ec32dd40fe99c2ec904aba7ae3f078b9df968075cbb55291717d751b&)所示。
- **分享 Llama 3.2 3B 4bit 截图**：一位用户发布了 **Llama 3.2 3B 4bit** 的截图，如[附图](https://cdn.discordapp.com/attachments/1218409701339828245/1372537263271051304/image.png?ex=6827cb30&is=682679b0&hm=f84185fbfe33d5e9af456ef71aaf4ff03339eabf81cae93d0be9a1901c188f69&)所示。


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1372656865233604778)** (3 messages): 

> `Web AI Engineer introduction, Full stack developer AI fan Introduction, Gabriel 20 years development experience` 


- **Web AI 工程师加入**：一位拥有 **7 年** 全栈经验的 **Web, AI 工程师** 介绍了自己。
   - 他们擅长使用现代 Web 技术构建响应式、用户友好的 Web 和移动应用程序，如 **React(Next), React Native(Expo), Flutter, Vue(Nuxt), Svelte, Astro**，以及 **Node/Express, Django, Nest.js, Go, Web3.js, Shopify, Wordpress, TailwindCSS, Shadcn, MUI, Docker, Kubernetes, AWS/GCP, LLM** 等工具。
- **热爱 AI 的全栈开发人员**：一位拥有超过 **20 年** 经验的开发人员介绍了自己。
   - 该开发人员主要专注于全栈开发，现在是 **AI** 的忠实粉丝，喜欢构建具有精心设计的 **UI** 和 **UX** 的实时应用程序，偏好在 **Cloudflare** 上运行 **Nuxt**，并使用 **RooCode** 和 **Windsurf** 等工具。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1372528849669197854)** (7 messages): 

> `Gemini Models Response Schema, Structured Outputs in DSPy, Pydantic models` 


- **Gemini 模型的结构化输出即将引入 DSPy**：成员们讨论了 **Gemini Models** 的响应模式（Response Schema，类似于 **OpenAI** 的结构化输出）是否已在 **DSPy** 中实现，另一位成员确认已经实现。
   - 同时确认 **DSPy** 会动态构建响应模式。
- **Pydantic 模型支持 DSPy 中的结构化输出**：一位成员询问如何在 **DSPy** 中实现类似于 **OpenAI** 工具的**结构化输出（Structured Outputs）**，包括**嵌套输出**或 **JSON Schema 约束**。
   - 另一位成员回答说*只需使用 Signatures*，并将 **Pydantic 模型**或 **Python TypedDicts** 作为输出字段类型传递。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1372523175656820788)** (5 messages): 

> `GPT4All's demise, Nomic's future direction, Jan.ai and LM Studio as alternatives` 


- **GPT4All 陷入停滞，未来不明**：成员们推测 **GPT4All** 是否因自 2 月以来缺乏更新而停止维护。
   - 一位成员对缺乏新版本发布的消息感到遗憾，并想知道 Nomic 的意图。
- **Nomic 转向付费平台？**：成员们推测 **Nomic** 可能会将其重点转向货币化平台。
   - 有说法称 *"gpt4all 结束了 .. 现在用 nomic 赚钱吧！"* —— 但这一说法没有额外的来源或证据支持。
- **Jan.ai 和 LM Studio 成为 GPT4All 的替代方案**：成员们提到 **Jan.ai** 和 **LM Studio** 是 **GPT4All** 的潜在替代品。
   - 目前尚未说明为什么这些是好的替代方案，或者它们具有哪些可能有益的功能。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1372645019734900865)** (2 messages): 

> `Event Driven Agent Workflows, Multi-Agent Docs Assistant, Tig AI Coding Agent` 


- **事件驱动的 Agent 助力 Weaviate**：LlamaIndex 发布了一个关于如何使用**事件驱动的 Agent 工作流**来构建**多 Agent 文档助手**的新教程。
   - 该助手将网页写入 Weaviate 中的 **LlamaIndexDocs & WeaviateDocs 集合**，并使用编排器（orchestrator）决定何时调用 Weaviate QueryAgent 进行搜索，详见[此推文](https://twitter.com/llama_index/status/1923085124725506441)。
- **Tig 编程 Agent 亮相**：一个名为 **Tig** 的开源**（人机协同）编程 Agent** 受到关注，它由 @rsrohan99 创建并基于 LlamaIndex 工作流构建。
   - 正如 [Twitter](https://twitter.com/llama_index/status/1923134285940441102) 所示，Tig 可以编写、调试和分析多种语言的代码，执行 shell 命令并搜索网页。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1372576791427944599)** (2 messages): 

> `PDF content extraction with LlamaIndex, Vibe Coding partnership opportunities` 


- **LlamaIndex 探索 PDF 内容提取**：一位成员正在寻求关于使用 **LlamaParse** 或 **LlamaIndex** 从 PDF 中提取内容的建议，特别是旨在提取目录（Table of Contents），然后根据预定义名称隔离特定章节的内容和表格。
   - 他们正在寻求关于设置指令或流水线（pipeline）的指导，以从目录中检测章节、隔离内容并正确结构化提取的表格，包括在 **n8n** 等无代码工具中使用的正确参数。
- **AI 初创公司寻找 Vibe Coding 伙伴**：一家总部位于韩国的 AI 初创公司正在寻找在 **Vibe Coding** 方面经验丰富的热情开发者，共同合作实际客户项目。
   - 该机会包括公平的收入分成模式和长期合作伙伴关系，要求具备良好的沟通能力、**GitHub 链接**、**Vibe Coding 项目参考**以及英语/韩语沟通能力。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1372471760405925990)** (4 messages): 

> `Custom Torchtune Network on vLLM, Unregistered Model with vLLM, Custom Model Implementation in vLLM` 


- **自定义网络在 vLLM 上面临实现问题**：一位成员尝试按照几个教程在 **vLLM** 上实现自定义 **Torchtune 网络**，但遇到了失败。
   - 另一位成员询问该模型是否已在 **vLLM** 注册，并建议将 checkpoint 转换为 **HF 格式**进行同步。
- **在 vLLM 中实现自定义模型**：一位成员确认使用了具有自定义架构的自定义模型，导致在 **vLLM** 上的实现遇到困难。
   - 另一位成员指向了一份关于在 **vLLM** 中实现自定义模型的 [vLLM 指南](https://docs.vllm.ai/en/latest/contributing/model/basic.html)。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1372622471420973056)** (3 messages): 

> `Lambda Workshop, Nobel FutureTech Info Session, Agentic AI, Inference API` 


- **Lambda 工作坊教授 Agentic AI**：一个 [Lambda 工作坊](https://www.youtube.com/watch?v=VmjMIwwo9ag)将教授如何使用 **Lambda 的 Inference API** 构建 Agent 应用、优化 Agent 性能以及在生产环境中部署 Agent。
   - 参与者可以在 5/16 周五前通过[此表单](https://forms.gle/UtVhmPS3mitS8Vxu7)申请 **100 美元的 serverless API 额度**。
- **Nobel FutureTech 炉边谈话**：由 [Nobel FutureTech Group](https://nobel-futuretech.com/index.html) 和 Berkeley RDI 共同主办的炉边谈话将深入探讨 **Nobel FutureTech Genius Club** 的创新生态系统。
   - 该会议提供了关于导师指导、资金和合作机会的信息，并提供了[直播链接](https://www.youtube.com/watch?v=ft-2W00Rtg8)。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1372504105439989800)** (1 messages): 

> `topk GPU, masked_select GPU, randperm_generator GPU, index_put_impl, index_tensor` 


- **Topk 悬赏：GPU 版本**：一位用户询问了 "move topk" 悬赏的要求，因为 **topk**、**masked_select** 和 **randperm_generator** 已经脱离了 CPU。
   - 该用户建议悬赏可能需要修订，因为 torch 后端中还有其他函数如 **_index_put_impl_** 和 **index_tensor** 仍需关注。
- **索引函数仍在 CPU 上**：用户指出 **_index_put_impl_** 和 **index_tensor** 仍在 CPU 上运行。
   - 他们建议这些函数以及 torch 后端中的其他函数可以作为 GPU offloading 的目标。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1372623165058191433)** (1 messages): 

> `Agentic Enrichment, LLM Data Access, Featureform` 


- **Agentic Enrichment 网络研讨会发布**：一场关于 **Agentic Enrichment** 的直播研讨会定于 **PT 时间 5 月 27 日星期二上午 8 点**举行，由 Featureform 创始人 **Simba Khadder** 主讲，内容涵盖如何使用 **MCP** 为 AI Agent 解锁数据；你可以在[这里](https://buff.ly/zeoH55Y)报名。
   - 研讨会将讨论 AI Agent 访问实时、内部业务数据所需的缺失基础设施层，强调 Agent 面临的限制在于数据访问而非智能。
- **通过更好的数据访问释放 LLM 潜力**：研讨会将涵盖通过更好的内部数据访问来释放 AI Agent 全部潜力的需求，详细介绍 **Agentic Enrichment 的三个关键组件**：语义编目、低延迟服务和治理。
   - 它将展示 **Featureform** 如何实现这种数据访问，使 Agent 在生产环境中更加实用和强大，并提供 AI 系统中改进工作流的真实案例。


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1372648280206479540)** (1 messages): 

> `SWE-1, Software Engineering Models, Flow Awareness Approach, Windsurf Tab Experience, Cascade Optimization` 


- **Windsurf 发布 SWE-1 系列模型**：Windsurf 推出了 **SWE-1** 软件工程模型系列，包括 **SWE-1**、**SWE-1-lite** 和 **SWE-1-mini**，详见 [博客文章](https://windsurf.com/blog/windsurf-wave-9-swe-1) 和 [发布视频](https://youtu.be/LhA9pVpwgdY)。
- **SWE-1 以更低成本提供 Claude 3.5 级别的性能**：据宣传，**SWE-1** 模型具有与 **Claude 3.5** 相当的 *高推理*、*工具调用能力* 和 *Cascade 优化* 性能，但成本更低。
   - 这些模型使用独特的 *"flow awareness"* 方法进行训练，能够理解开发界面中人类与 AI 之间的时间线。
- **SWE-1 旨在加速软件开发**：Windsurf 旨在通过新的 **SWE-1** 模型将软件开发速度提升 **99%**。
   - 这仅仅是个开始——他们正在投入巨资，使 **SWE 模型** 在软件工程方面的表现超越所有前沿模型。


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1372662490118357283)** (1 messages): 

> `AI21 Labs, Maestro, AI Tinkerers Meetups` 


- **AI Tinkerers 主办 AI21 见面会**：AI Tinkerers 正在与 AI21 合作举办即将到来的见面会，介绍他们最近发布的 [Maestro 平台](https://www.ai21.com/maestro/)，这是一款全新的规划与编排工具。
   - 见面会对公众开放且免费，需注册参加在 [纽约](https://nyc.aitinkerers.org/p/how-it-s-made-architecting-planning-based-ai-systems-ft-ai21-maestro)、[巴黎](https://paris.aitinkerers.org/p/ai-tinkerers-paris-ai21-labs-takeover-on-may-19th) 和 [旧金山](https://sf.aitinkerers.org/p/how-it-s-made-architecting-planning-based-ai-systems-ft-ai21-maestro) 举行的活动。
- **AI21 揭晓 Maestro 规划平台**：AI21 Labs 最近发布了 **Maestro**，这是一个专为 AI 系统中的规划和编排而设计的新平台。
   - 该平台旨在为开发者提供工具和基础设施，以构建更复杂、更高效的 AI 应用。