---
companies:
- openai
- huggingface
- microsoft
- llamaindex
- ollama
- baseten
- fireworksai
- cerebras
- groq
- together
- anthropic
- google
- uk-aisi
date: '2025-08-06T05:44:39.731046Z'
description: '**OpenAI** 发布了自 GPT-2 以来首批开源模型 **gpt-oss-120b** 和 **gpt-oss-20b**，这些模型迅速在
  **Hugging Face** 上引发关注。**微软**通过 **Azure AI Foundry** 和 **Windows Foundry Local**
  对这些模型提供支持。


  关键架构创新包括**滑动窗口注意力机制 (sliding window attention)**、**混合专家模型 (MoE)**、一种 **RoPE 变体**以及
  **256k 上下文长度**。这些模型采用了 **llama.cpp** 支持的新型 **MXFP4** 格式。有假说认为 **gpt-oss** 是在**合成数据**上训练的，旨在增强安全性和性能，这支持了**推理核心假说
  (Reasoning Core Hypothesis)**。


  **OpenAI** 宣布了 **50 万美元的悬赏金**，用于与 **Anthropic**、**谷歌**及**英国人工智能安全研究所 (UK AISI)**
  等合作伙伴共同开展红队测试。性能评价指出其基准测试结果参差不齐，**GPT-OSS-120B** 在 **Aider Polyglot** 编程基准测试中得分为
  **41.8%**，落后于 **Kimi-K2** 和 **DeepSeek-R1** 等竞争对手。一些用户注意到，该模型在数学和推理方面表现出色，但在常识和实际应用价值方面有所欠缺。'
id: MjAyNS0w
models:
- gpt-oss-120b
- gpt-oss-20b
- kimi-k2
- deepseek-r1
- qwen-3-32b
people:
- woj_zaremba
- sama
- huybery
- drjimfan
- jxmnop
- scaling01
- arunv30
- kevinweil
- xikun_zhang_
- jerryjliu0
- ollama
- basetenco
- reach_vb
- gneubig
- shxf0072
- _lewtun
title: 今天没什么特别的事。
topics:
- sliding-window-attention
- mixture-of-experts
- rope
- context-length
- mxfp4-format
- synthetic-data
- reasoning-core-hypothesis
- red-teaming
- benchmarking
- coding-benchmarks
- model-performance
- fine-tuning
---

**暴风雨前的宁静。**

> 2025年8月5日至8月6日的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 29 个 Discord 社区（227 个频道，8597 条消息）。预计节省阅读时间（以 200wpm 计算）：830 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以优美的 vibe coded 方式呈现所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们反馈！

请关注明天太平洋时间上午 10 点的 OpenAI 直播。

与此同时，您可以收听[今天的播客](https://www.youtube.com/watch?v=t4IQwMa5-6U&t=3s)，了解媒体如何获取泄密信息并报道主要的 AI 初创公司。

---

# AI Twitter 回顾

**OpenAI 的 GPT-OSS 发布与架构**

- **官方公告与社区集成**：**OpenAI** 宣布发布自 GPT-2 以来其首批开放模型 **gpt-oss-120b** 和 **gpt-oss-20b** ([链接](https://twitter.com/arunv30/status/1952881931798143276))，这些模型迅速成为 **Hugging Face** 上最热门的模型 ([链接](https://twitter.com/kevinweil/status/1952969984931709376))。**Microsoft** 宣布在 **Azure AI Foundry** 以及通过 Foundry Local 在 **Windows** 上支持这些模型 ([链接](https://twitter.com/xikun_zhang_/status/1952902211278913629))。众多平台宣布立即提供支持，包括用于 Agent 工作流的 **LlamaIndex** ([链接](https://twitter.com/jerryjliu0/status/1952883595787239563))、支持网页搜索的 **Ollama** ([链接](https://twitter.com/ollama/status/1952882173255856223))、**Baseten** ([链接](https://twitter.com/basetenco/status/1952882156059148737))，以及由 **FireworksAI**、**Cerebras**、**Groq** 和 **Together** 等 **HF Inference Providers** 提供支持的公开演示 ([链接](https://twitter.com/reach_vb/status/1953041435999010916))。为了鼓励开发，**OpenAI** 和 **Hugging Face** 正向 500 名学生提供 **50 美元的推理额度** ([链接](https://twitter.com/reach_vb/status/1953010091377958984))。
- **架构细节**：[@gneubig](https://twitter.com/algo_diver/status/1952941862723162439) 等人总结了关键的架构创新，包括 **sliding window attention**、**mixture of experts (MoE)**、一种特定的 **RoPE 变体**以及 **256k context length**。[@shxf0072](https://twitter.com/shxf0072/status/1953143243992166849) 指出，Attention 仅占模型的 **0.84%**，智能存储在 **99.16% 的 MLP 层**中。这些模型使用了新的 **MXFP4** 格式，**llama.cpp** 现在已原生支持该格式 ([链接](https://twitter.com/ggerganov/status/1952978670328660152))。[@_lewtun](https://twitter.com/_lewtun/status/1952990532436934664) 强调了 **OpenAI 关于微调这些模型的指南**。
- **关于训练与设计的假设**：[@huybery](https://twitter.com/huybery/status/1952905224890532316) 分享的一个流行假设是 **gpt-oss** 完全是在 **synthetic data**（合成数据）上训练的，从而增强了安全性和性能。[@DrJimFan](https://twitter.com/DrJimFan/status/1953139796551086265) 认为这支持了**“推理核心假设”（Reasoning Core Hypothesis）**，即推理只需要极少的语言能力，这与轻量级 **“LLM OS Kernel”** 的概念相一致。[@jxmnop](https://twitter.com/jxmnop/status/1953218992954589525) 评论道，**Sam Altman** 似乎想要一个技能极高（例如在 Codeforces 上评分为 3200）但缺乏对他本人等现实世界实体了解的模型。
- **红队测试与安全**：**OpenAI** 的 [@woj_zaremba](https://twitter.com/woj_zaremba/status/1952886644090241209) 宣布了一项 **50 万美元的悬赏**，用于对新模型进行压力测试，调查结果将由包括 **OpenAI、Anthropic、Google** 和 **UK AISI** 在内的联盟进行审查。

**GPT-OSS 性能、基准测试与批评**

- **性能不一致与“基准测试”**：包括 [@Teknium1](https://twitter.com/Teknium1/status/1953063858761023843) 在内的多位用户观察到，该模型似乎经过了“刷榜优化（benchmaxxed）”，导致其性能表现非常奇怪。[@scaling01](https://twitter.com/scaling01/status/1952881329772564764) 指出它在“数学/编程和推理方面刷分严重（slopmaxxed）”，但缺乏“品味和常识”。[@jxmnop](https://twitter.com/jxmnop/status/1953216881361600729) 发现它前一秒还能进行专业编程，下一秒就会对基本事实产生严重的幻觉（hallucinate）。
- **Aider Polyglot 基准测试结果**：**GPT-OSS-120B** 模型在 **Aider Polyglot** 编程基准测试中表现不佳，得分仅为 **41.8%**。[@scaling01](https://twitter.com/scaling01/status/1953047534122713130) 指出这显著低于竞争对手，如 **Kimi-K2 (59.1%)** 和 **DeepSeek-R1 (56.9%)**，尽管略好于 **Qwen3 32B (40.0%)**。这引发了人们对该模型在数学和推理任务之外的实际效用的质疑 ([link](https://twitter.com/scaling01/status/1953047913954791696))。
- **与中国模型的对比及缺乏致谢**：[@scaling01](https://twitter.com/scaling01/status/1952900225120780705) 认为“目前还没有西方开源模型能击败或赶上最好的中国开源模型”，并引用 **Qwen3-235B-A22B, R1, and GLM-4.5** 作为优于 **GPT-OSS-120B** 的例子。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1952898957035880923) 批评 **OpenAI** 没有对 **DeepSeek** 表示感谢，后者的架构和技术似乎产生了重要影响。
- **涌现能力与怪癖**：用户观察到了有趣的涌现行为，例如该模型在不使用工具的情况下进行复杂数学计算的能力 ([link](https://twitter.com/scaling01/status/1952892387539259455))，以及尝试暴力破解 base64 解码 ([link](https://twitter.com/scaling01/status/1952886371019809037))。[@Teknium1](https://twitter.com/Teknium1/status/1952892281218023824) 等人注意到它倾向于自称为“我们”，将其比作“博格方块（borg cube）”。

**Google 的 Genie 3 及其他 AI 进展**

- **Genie 3 交互式世界模型**：**Google DeepMind** 发布了 **Genie 3**，这是一个“开创性的世界模型（world model）”，能够根据文本或视频输入生成完整的交互式、可探索且可玩的场景 ([link](https://twitter.com/denny_zhou/status/1952887267963662429))。这一发布引发了人们对“神经视频游戏（neural video games）”未来的兴奋 ([link](https://twitter.com/demishassabis/status/1952890039643353219))，[@jparkerholder](https://twitter.com/GoogleDeepMind/status/1953024241017749604) 称其为“世界模型的里程碑时刻”。用户强调了它模拟复杂渲染效果并即时生成新内容的能力 (link)。
- **情境学习（In-Context Learning）与集成的火花**：**Genie 3** 展示了情境学习的火花，允许用户提供一段视频（例如来自 **Veo 3**），然后以此为基础控制模拟，模仿原始视频的动态 ([link](https://twitter.com/_rockt/status/1953117236975030653))。此外，还演示了类似《盗梦空间》的场景潜力，即一个 **Genie 3** 模拟在另一个 **Genie 3** 模拟中运行 ([link](https://twitter.com/shlomifruchter/status/1953155882902274126))。
- **新教育工具与 Gemini 更新**：**Google** 启动了 **AI for Education Accelerator**，承诺投入 **10 亿美元** 用于提升 AI 素养，并为大学生提供免费的 AI 培训和 **Google Career Certificates** ([link](https://twitter.com/Google/status/1953126394847768936))。**Gemini App** 的更新包括用于创建个性化插图故事的 **Storybook** 功能 ([link](https://twitter.com/demishassabis/status/1952897414207074810))，以及 **Guided Learning**、闪卡和集成视觉效果等新学习工具 ([link](https://twitter.com/Google/status/1953143185011617891))。

**Agent 工具、开发与框架**

- **Claude Code 课程与安全特性**：**Andrew Ng** 和 **DeepLearningAI** 与 **Anthropic** 合作发布了关于 **Claude Code** 的课程，重点关注高度 Agent 化的编程工作流（[链接](https://twitter.com/AndrewYNg/status/1953097967361245251)）。**Anthropic** 还宣布 **Claude Code** 现在可以自动审查代码中的安全漏洞（[链接](https://twitter.com/AnthropicAI/status/1953135070174134559)）。
- **LangChain 推出 Open SWE**：**LangChain** 发布了 **Open SWE**，这是一个开源的、基于云的异步编程 Agent，可以连接到 GitHub 仓库以自主解决问题（[链接](https://twitter.com/Hacubu/status/1953168346356314376)）。该 Agent 通过分解问题、编写代码并创建 Pull Requests 来工作。
- **LlamaIndex 与 LlamaCloud**：**LlamaIndex** 展示了金融文档 Agent 的集成（[链接](https://twitter.com/jerryjliu0/status/1953108641558540720)），以及与 **Delphi** 的合作，利用 **LlamaCloud** 作为文档摄取的上下文层来创建“数字大脑”（[链接](https://twitter.com/jerryjliu0/status/1952889056200655206)）。他们还在 **LlamaCloud** 中引入了一种新的“平衡”解析模式，用于对图表等视觉元素进行高性价比的分析（[链接](https://twitter.com/jerryjliu0/status/1953227974716665996)）。
- **RAG 与分块 (Chunking)**：**DeepLearningAI** 强调了在生产级 **RAG** 系统中实现可观测性的必要性，以跟踪性能和质量（[链接](https://twitter.com/DeepLearningAI/status/1952886740349272173)）。[@femke_plantinga](https://twitter.com/bobvanluijt/status/1953013722026250737) 认为开发者应该“停止优化检索”，而是“先解决分块问题”，因为这通常是性能不佳的根本原因。

**基础设施、硬件与效率**

- **Ollama 与 ggml 性能对比**：[@ggerganov](https://twitter.com/ggerganov/status/1953088008816619637) 解释说，**LMStudio** 在运行 **GPT-OSS** 时的性能显著更好，因为它使用了上游的 **ggml** 实现。他指出 **Ollama** 的分支在 **MXFP4** 内核和 Attention Sinks 的实现上效率低下，导致性能较差。
- **推理提供商的性能与正确性**：**vLLM** 表示他们已经进行了多次评估，**Hopper** GPU 上的数值计算应该是“可靠且经过验证的”（[链接](https://twitter.com/vllm_project/status/1952940603773468926)）。然而，像 [@AymericRoucher](https://twitter.com/AymericRoucher/status/1953115586273394873) 这样的用户指出，不同提供商之间的性能差异巨大，这可能是由于激进的量化 (Quantization) 导致的。**Groq** 因提供可靠的结果和极高的速度而受到称赞，其 **120B 模型** 的运行速度超过了 **500 tokens/second**（[链接](https://twitter.com/JonathanRoss321/status/1953119620103381440)）。
- **量化与硬件支持**：**Cerebras** 宣布 **GPT-OSS-120B** 在其系统上实时运行速度达到 **3,000 tokens/s**（[链接](https://twitter.com/cline/status/1952960760759632025)）。[@HaihaoShen](https://twitter.com/teortaxesTex/status/1953017577900228920) 发布了可能是首个 **GPT-OSS** 的 **INT4** 量化版本。社区还强调了 **AMD GPU** 运行本地模型的潜力，一位用户展示了 **20B 模型** 在一台不到 1000 美元的笔记本电脑上以 **52 tok/sec** 的速度运行（[链接](https://twitter.com/dzhng/status/1953132623280165193)）。

**幽默与梗图**

- **关于炒作与发布**：[@gdb](https://twitter.com/gdb/status/1953184691567349976) 发布了“团队一直在努力工作，期待明天！”，引发了关于 GPT-5 发布的猜测。[@nrehiew_](https://twitter.com/nrehiew_/status/1953142337745633373) 发布了一份对下一个模型的期望清单，包括“请不要只是为了刷榜 (benchmaxxed)”和“请带点灵魂”。
- **关于模型行为**：[@Teknium1](https://twitter.com/Teknium1/status/1953063858761023843) 宣称：“说实话，这就是过度刷榜的后果”，表达了社区对 GPT-OSS 奇怪性能的看法。[@code_star](https://twitter.com/code_star/status/1953153930944446852) 开玩笑说：“只有产自法国 Container 地区的才能叫真正的 Docker，其他的只能叫起泡 Hypervisor。”
- **开发者共鸣**：[@fabianstelzer](https://twitter.com/fabianstelzer/status/1953150053050101785) 发布了一个关于“凭感觉编程 (vibe coding)”完成整个应用后，被问及 API 密钥是否在环境变量中的梗图。[@jxmnop](https://twitter.com/jxmnop/status/1953163073612562851) 发布了一张配文为“第一条规则：永远不要从 DeepSeek 进行蒸馏 (distill)”的梗图。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen3-4B-Thinking-2507 模型发布与讨论

- [**🚀 Qwen3-4B-Thinking-2507 发布！**](https://i.redd.it/3cl3vbg54fhf1.jpeg) ([得分: 883, 评论: 98](https://www.reddit.com/r/LocalLLaMA/comments/1mj7t51/qwen34bthinking2507_released/)): **该图片展示了新发布的 Qwen3-4B-Thinking-2507 模型的基准测试结果或能力对比。该模型以增强的推理能力、长上下文处理（256K tokens）和对齐（alignment）而著称。帖子和评论中的技术讨论强调了该模型在推理基准测试中的显著提升，例如 BFCL-v3（得分 71.2），其性能接近 GPT-4o 等大型模型，考虑到其仅 4B 的参数规模，这一点令人印象深刻。在 Hugging Face 上的发布包含了多个 GGUF 量化版本，可立即部署。社区还要求将其与其他强大的开源模型（如 gpt-oss-20b）进行基准对比。** 评论者讨论了混合推理（hybrid reasoning）与专门的纯推理模型之间的权衡，共识是独立模型能产生更优的性能。BFCL-v3 的分数受到了称赞，人们对与更大规模开源模型的基准对比表现出浓厚兴趣。LMStudio 等部署工具被提及，认为它们促进了快速采用。
    - 用户指出，混合推理似乎对 LLM 性能产生负面影响，一些人建议保留独立的通用版本和推理专用版本以获得更好的效果。
    - Qwen3-4B-Thinking-2507 在 BFCL-v3 基准测试中取得了 71.2 的显著高分，这对于该规模的模型来说是前所未有的，并接近了 GPT-4o 等更先进模型的典型性能。多个 GGUF 量化版本（Q3, Q4, Q6, Q8）已在 lmstudio 中提供，可立即使用。
    - 有人请求与 gpt-oss-20b 和 Gemma3n4b 等模型进行直接基准对比，这表明人们预期 Qwen3-4B-Thinking-2507 可能会超越更大规模或最近发布的模型。此外，该模型支持高达 256k 的上下文窗口，这在 4B 参数的 LLM 中非常突出。
- [**就在你以为 Qwen 已经结束时...**](https://www.reddit.com/r/LocalLLaMA/comments/1mj7pny/just_when_you_thought_qwen_was_done/) ([得分: 336, 评论: 75](https://www.reddit.com/r/LocalLLaMA/comments/1mj7pny/just_when_you_thought_qwen_was_done/)): **Qwen 为 Qwen3 4B 模型发布了新的 Checkpoint 版本：[Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) 和 [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)，这表明在其之前的重大发布之后仍在持续改进，并将其定位为强有力的开源 LLM 竞争者。帖子中未列出针对 "-Thinking-2507" 和 "-Instruct-2507" 的具体细节或基准测试，但社区已经注意到了 Qwen 持续的高速开发。** 评论中的技术讨论呼吁推出 Qwen3 Coder 32B 以及用于高效推测解码（speculative decoding）的较小 1.7B 变体，认为此类模型在相似的推理速度下可以与 GPT-OSS 120B 匹敌，反映出对高性能且高效的开源模型的强劲需求。
    - 一位用户建议将 Qwen3 Coder 32b 模型与较小的 1.7b 模型结合进行推测解码，并指出这种配置在相似的推理速度下应该能 *超越 120b gpt-oss*。这突显了人们对混合规模模型推理策略以提高效率和有效性的日益增长的兴趣。
    - 有关于 Qwen 模型发布策略的讨论，一位评论者指出，逐步发布模型尺寸（而不是一次性全部发布）可以保持公众注意力并最大化热度。这暗示了一种为了获得更广泛曝光和参与而有意采取的交错发布方式。
- [**Qwen 停不下来！！（还在调侃 sama 哈哈）**](https://i.redd.it/3nhqo0qf9fhf1.jpeg) ([得分: 506, 评论: 46](https://www.reddit.com/r/LocalLLaMA/comments/1mj8lk8/qwen_isnt_stopping_and_trolling_sama_lol/)): **图片提到了“Qwen”，即阿里巴巴开发的开源语言模型系列，似乎展示了最近的发布、性能结果或排行榜排名（由于分析失败，图片的具体内容不明确）。帖子标题和评论暗示 Qwen 正在取得快速且重大的进展，可能正在超越竞争对手或积极与 OpenAI 竞争（“trolling sama” 指的是 Sam Altman）。评论讨论了频繁的发布节奏，并推测未来的模型如 “qwen3-coder-thinking 30B”，表达了对更大规模或更专业模型的期待。** 讨论集中在频繁发布的动机上，一些人质疑阿里巴巴积极开源背后的动机，而另一些人则推测其竞争策略。人们还对未来更强大的 Qwen 模型表达了兴奋和期待。

- 用户表达了对潜在的 'qwen3-coder-thinking 30B' 模型的期待，暗示了对基于 30B 参数架构、专注于代码的 Qwen 系列语言模型的兴趣。这反映了开源 LLM 开发中关于模型缩放策略（scaling strategies）和任务专业化（如编程能力）的持续技术讨论。
- 社区将 Qwen 模型与基于 GPT 的开源项目进行了对比（“GPT oss 没什么问题...”），表明社区正在就 Qwen 模型与成熟的开源 GPT 架构之间的性能对等性或优势进行持续的基准测试（benchmarking）和讨论。
- 社区观察者注意到 Qwen 团队发布频率的加快（“Qwen 团队的速度令人振奋！”），这可能表明其工作流程中存在快速迭代、改进的流水线自动化或优化的数据/建模过程。

### 2. OpenAI 模型安全性、命名及社区反应梗图（Memes）

- [**OpenAI，我觉得不够安全**](https://i.redd.it/af6jm3nt9bhf1.png) ([Score: 1444, Comments: 140](https://www.reddit.com/r/LocalLLaMA/comments/1misyvc/openai_i_dont_feel_safe_enough/)): **该图片是一个梗图，引用了 OpenAI 安全政策或近期事件中被察觉到的问题。帖子和评论并未讨论任何基准测试（benchmarks）、模型或技术细节——这是对 OpenAI 企业策略或围绕 AI 安全的沟通所做的讽刺性回应。** 评论者指出 OpenAI “选择让自己变成一个梗”，表明该公司的信息传递与用户感知之间存在严肃性缺失或脱节；讨论主要以幽默为主，非技术性。
    - 讨论中涉及的最具技术相关性的细节是，OpenAI 最新模型的训练数据截止日期为 2024 年 6 月，这意味着它缺乏对该时间点之后发布的事件或数据的了解。例如，它无法回答有关最近（2024 年中后期）选举结果或其他非常近期进展的问题，突显了在即时事实认知方面的局限性。
- [**泄露：OpenAI 是如何给新模型命名的。**](https://i.redd.it/d60vtzhkkehf1.png) ([Score: 378, Comments: 21](https://www.reddit.com/r/LocalLLaMA/comments/1mj4zkk/leak_how_openai_came_up_with_the_new_models_name/)): **帖子中引用的图片是一个梗图，讽刺了 OpenAI 新模型的命名过程，可能是为了回应他们近期对“开源（Open Source）”标签的使用。评论区的对话讨论了对 OpenAI 是否真的开源了模型背后的数据集和训练代码的怀疑，反映了 AI 社区关于“提供模型”与“真正开源”之间差异的普遍争论。帖子或评论中未提供技术基准测试或直接的实现细节。** 几位评论者对 OpenAI 使用“开源”一词表示批评，暗示该公司可能为了营销而滥用该术语，特别是如果数据集和训练代码并未实际发布。这呼应了该领域关于什么构成真正开源 AI 的持续争议。
    - 一位用户提出了一个技术顾虑，即“开源”模型与仅仅是“免费可用”模型之间的区别。他们质疑 OpenAI 是否随模型一起发布了数据集和训练代码，并指出在 AI 领域，当实际源代码或数据未共享时，“开源”一词经常被误用。这突显了机器学习社区关于模型发布的透明度和可重复性的持续辩论。
- [**为了您的安全，已进行极致安全化（Safemaxxed）！**](https://i.redd.it/gaqdycledchf1.png) ([Score: 369, Comments: 24](https://www.reddit.com/r/LocalLLaMA/comments/1mix2kg/safemaxxed_for_your_safety/)): **该图片似乎是一个梗图，或者是对闭源 AI 模型（特别是来自 OpenAI 的模型）实施的日益严格的安全限制和内容审核（“safemaxxing”）的讽刺性评论。讨论强调了对过度审查和这些模型中日益增加的偏见的担忧，指出随着 OpenAI 向中东等市场扩张，商业决策可能会导致采用最严格的全球政策。建议技术社区保留开源模型，以应对进一步的封锁。** 评论者对 AI 模型日益增加的限制表示怀疑和沮丧，一些人专门批评 OpenAI 超过了预期的审查水平，并警告全球业务扩张可能会进一步收紧控制，甚至威胁到技术话题的开放讨论。

- 一位用户指出了像 OpenAI 这样主流 LLM 受业务驱动的限制，指出公司政策必须符合其业务所在地最严格的司法管辖区。这可能导致模型输出受到进一步限制，特别是当它们扩展到中东等地区时，并可能限制有关民权等敏感话题的信息。
- 另一位评论者寻求推荐目前限制最少、适合在拥有 32GB RAM 和 12GB 3060 GPU 的硬件上进行本地推理的大语言模型，这含蓄地引发了关于模型大小、VRAM 需求以及针对限制较少的使用场景的可行开源替代方案的技术讨论。
- 讨论中出现了关于模型安全/对齐工作（对模型进行“lobotomizing”/脑叶切除）与开发速度或实用性之间权衡的问题。一些人认为，广泛的安全微调减缓了进展并限制了创造性应用，并呼吁为研究或非标准用例提供“abliberated”或无审查版本。
- [**“怎么，你不喜欢你的新 SOTA 模型吗？”**](https://i.redd.it/9yqb0l1n9chf1.png) ([Score: 737, Comments: 123](https://www.reddit.com/r/LocalLLaMA/comments/1miwrli/what_you_dont_like_your_new_sota_model/)): **这张图片是一个迷因（meme），引用了围绕一个新的“SOTA”（state-of-the-art）模型的发布和媒体兴奋，该模型可能来自 OpenAI。标题和评论的背景表明，技术用户对发布模型的真正新颖性或影响持怀疑态度，认为其主要受众是主流媒体和投资者，而不是熟悉最新进展的技术社区。这种幽默突显了企业营销与真正的技术突破之间感知的脱节。** 评论者认为，重大发布通常是针对非技术受众和投资者的过度炒作，而不是针对 ML 从业者社区，而且这种兴奋通常忽略了现有的开源或先前的学术成果。讨论反映了关于 AI “进步”叙事以及谁真正从中受益或评估这些声明的持续紧张关系。
    - 讨论中包括对 OpenAI 新发布的开源模型实际用途的怀疑，一位用户认为 OpenAI 故意限制（“lobotomized”）了其功能，引发了人们对通过微调将模型恢复到更可用或 SOTA 性能水平所需的工作量和技术干预程度的好奇。这突显了社区中关于模型对齐与实用性之间持续存在的担忧，以及开源社区通过额外的微调或改进的训练程序恢复或超越原始性能的潜力。
- [**到目前为止，你的体验如何？**](https://i.redd.it/lj67oslhbdhf1.png) ([Score: 341, Comments: 24](https://www.reddit.com/r/LocalLLaMA/comments/1mj00mr/how_did_you_enjoy_the_experience_so_far/)): **该帖子讨论了使用新模型的体验，重点关注其严格的对齐/安全限制以及类似于 Phi 的特性。最高赞评论对该未命名模型（可能指 Googy 或 OpenAI 对齐的类似物）与其他 LLM 进行了详细的技术对比，强调了其保守的安全对齐（“永远不会忘记……它首先是 OpenAI 人力资源部的一员”）、与其规模相称的强大 STEM 知识以及令人印象深刻的代码分析能力。值得注意的技术基准包括它通过了 Grosper 的 LFT 连分数函数测试（使用 OCaml）、回答高级物理和数学问题的能力，以及在 base64/ascii/rot13 解码等任务上的表现。然而，该模型对表格的过度依赖、合成数据的迹象以及与现代 AI 工具的集成问题被列为局限性。摘要和实体提取性能被认为良好，但在对齐/安全干扰方面存在警告。** 技术辩论集中在强大的安全对齐（甚至到了让用户体验沮丧和可能导致知识截断的程度）与模型在代码、STEM 和某些问题领域的胜任能力之间的权衡。用户对该模型的价值表示怀疑，原因是其沉重的安全限制，并争论其能力是否抵消了其对齐限制。
    - 提供了几个模型（Goody/GPT-phi-3.5 原型、Qwen-30B MoE, GLM 4.5 Air）在高级任务上的详细技术对比，例如通过 continuation passing 区分反向和前向模式 AD，以及处理数学/算法 OCaml 代码（例如 Grosper 的 LFT 连分数函数）。Goody 20B 在反向模式 AD 推理中失败，但以胜任的算法识别处理了 OCaml 测试，尽管识别并不完美。Qwen-30B MoE 和 Goody 120B 也有类似的局限性，而 GLM 4.5 Air 是唯一一个在没有提示的情况下取得成功的模型。

- 评论者指出，这些模型的训练可能重度依赖合成数据 (synthetic data)，这是从其明显的表格化输出倾向中推断出来的。黑洞物理、多世界解释中的指针态 (pointer states) 以及基于概率的 D&D 施法问题等 STEM 主题基准测试被引用为案例，在这些案例中，20B 模型提供了“同尺寸下的 SOTA”（相对于参数规模的 state-of-the-art），这表明其通用推理和领域知识表现出色但分布不均。
- 对实际 NLP 能力进行了评估：20B 模型可以准确解码 base64、ASCII 二进制、rot13，如果明确告知，甚至可以解码链式 rot13+base64 编码，在某些解码任务中表现优于 Qwen-30B。摘要生成和实体/概念提取也被认为是其强项，尽管人们担心其过度依赖企业级安全回答（corporate-safe answers），且缺乏与现代 AI 工具链的集成。

### 3. 埃隆·马斯克开源 Grok 2 的承诺以及业界对 GPT-OSS 的质疑

- [**埃隆·马斯克表示 xAI 将于下周开源 Grok 2**](https://i.redd.it/htgw3mmvjdhf1.jpeg) ([Score: 420, Comments: 175](https://www.reddit.com/r/LocalLLaMA/comments/1mj0snp/elon_musk_says_that_xai_will_make_grok_2_open/))：**该图片伴随着埃隆·马斯克的 xAI 将于下周开源 Grok 2 的公告出现，马斯克在 X（原 Twitter）上的帖子证实了这一点。从技术上讲，Grok 2 是在更具竞争力和能力的模型之后发布的，评论指出 Grok 2 被认为比当前的 SOTA 权重开放模型（open-weight models）体积更大且性能更低。来自评论的补充背景表明，后来的 Grok 版本通过强化学习 (RL) 实现了更好的推理能力，这暗示即将开源的 Grok 2 实际用途可能有限。** 评论者对这一举动的意义展开了辩论，一些人批评 Grok 2 发布时机太晚，且相对于同时代模型性能较差；而另一些人则讨论了开源趋势是应对行业压力而非技术价值的产物。
    - 有建议认为 Grok 4 实际上只是应用了额外强化学习 (RL) 以增强推理能力的 Grok 3，这可以解释为什么 xAI 相比于 Grok 2 更不愿意开源 Grok 3。
    - 技术批评指出，Grok 2 既比现代开源模型大得多，表现也差得多，这意味着考虑到当今的模型格局，其即将到来的开源可能几乎没有实际意义。
    - 存在一场关于开源模型发布速度加快和竞争激烈的元讨论：虽然 Mixtral 8x7b 在 2023 年底被视为权重开放模型的巅峰，但现在的领先公司都在争相发布新模型以避免被认为落后，这凸显了行业动态而非模型本身的优劣。
- [**随着更多独立测试结果出炉，GPT-OSS 看起来更像是一场公关噱头 :(**](https://i.redd.it/onk13jqo0ehf1.jpeg) ([Score: 661, Comments: 183](https://www.reddit.com/r/LocalLLaMA/comments/1mj2hih/gptoss_looks_more_like_a_publicity_stunt_as_more/))：**该图片展示了将 GPT-OSS 与其他编程模型进行对比的基准测试结果，突显了其性能低于更新版本的 DeepSeek-R1（0528 版本为 71.4%）、GLM 4.5 Air 以及 Qwen 3 32B。评论者澄清了基准测试的细节，纠正了早期的错误归因，并强调虽然 GPT-OSS 是一个采用稀疏 MoE (sparse MoE) 的 FP4 模型，但其激进的安全对齐 (safety tuning) 负面影响了性能。像 Qwen 3 32B 这样的稠密模型 (Dense models) 需要更多内存且运行速度较慢，这为效率与能力的讨论提供了一些背景。** 技术性的辩论集中在基准测试报告中的版本差异、模型架构（稀疏 MoE 与稠密模型）的相对优缺点，以及安全微调与模型可用性之间的权衡，社区建议针对性的微调 (finetuning) 可以提高 GPT-OSS 的性能。
    - 一位评论者指出，使用 DeepSeek-R1 进行基准测试对比时应引用得分 71.4% 的最新版本 (0528)，而非得分 56.9% 的旧版本 (0120)，强调了引用最新测试结果的重要性。
    - 讨论强调，在相似参数规模下，GLM 4.5 Air 等模型在编程基准测试中优于 GPT-OSS。Qwen 3 32B 虽然内存占用相当，但更稠密且运行速度更慢，而 Qwen 的 30B-A3B 编程模型在相同基准测试中仅获得约 52% 的分数，指出 GPT-OSS 在同类产品中的编程表现相对较弱。
    - 几项技术评论讨论了权衡：GPT-OSS 作为一个拥有 5 个激活参数 (active parameters) 的稀疏 MoE，对于内存适中的用户更实用，在双通道 DDR4-3200 上可达到 5t/s。然而，激进的安全对齐似乎限制了其推理能力，一些人建议社区驱动的微调可以提高其实际可用性。

## 非技术类 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Genie 3 交互式世界生成与热度

- [**Genie 3 将 Veo 3 生成的无人机镜头转化为可在飞行中途接管控制的交互式世界**](https://v.redd.it/d51vt9w83ghf1) ([Score: 1450, Comments: 248](https://www.reddit.com/r/singularity/comments/1mjd3mk/genie_3_turns_veo_3_generated_drone_shot_into_an/)): **该帖子展示了如何使用 Genie 3 将 Veo 3 生成的无人机视频转换为 *交互式环境*，允许用户在飞行中途接管控制。Genie 3 利用生成式 AI 将 2D 视频帧映射到可控的 3D 空间，这表明实时视频到环境转换技术取得了重大进展，潜在应用包括游戏、模拟和城市建模。** 热门评论强调了将 Genie 3 与 Google Maps 和 VR 等技术结合以实现更广泛用例的潜力，以及对从游戏到社会应用等行业的变革性影响。
    - 评论者推测将 Genie 3 与 Google Maps 和 VR 结合，可以创建完全沉浸式且可探索的 3D 世界，并提出了在模拟和导航技术方面的潜在用例。技术读者可能会推断，挑战在于将来自 Veo 3 的实时无人机图像或视频与实时 AI 环境生成 (Genie 3) 以及 VR 渲染相结合——且所有这些都要以交互式帧率运行。
    - 讨论了 Genie 3 与 Quest 2 等 VR 平台的集成，这隐含地提出了关于接口兼容性以及在由视频或无人机镜头创建的沉浸式实时 3D 环境中实现无缝用户体验所需的计算能力等技术问题。
- [**Genie 3 太疯狂了🤯](https://v.redd.it/y7s2bce70ghf1)https://x.com/jkbr_ai/status/1953154961988305384?s=46** ([Score: 626, Comments: 105](https://www.reddit.com/r/singularity/comments/1mjcm7i/genie_3_is_insanehttpsxcomjkbr/)): **引用的帖子展示了 Google DeepMind 的 AI 模型 Genie 3，它能够根据文本或图像提示生成高度逼真的交互式 3D 环境。演示中通过在另一个 Genie 生成的世界中递归生成一个 Genie 生成的 Agent（“在 Genie 内部玩 Genie”）进行了展示。这种逼真度令观众感到震惊，模糊了合成图像与真实图像之间的界限，并突显了生成式场景渲染方面的进步，正如其研究人员在[此处](https://x.com/jkbr_ai/status/1953154961988305384?s=46)所讨论的那样。** 评论者对 Genie 3 的逼真度表示惊讶，一些人最初将其输出误认为是真实生活中的镜头。递归使用 Genie 来生成环境和 Agent 引发了技术上的好奇，并对模型能力的深度感到惊讶。
    - 一条评论引用了图片中提供的一位研究人员的陈述。该陈述可能提供了对 Genie 3 独特技术能力或研究人员视角的见解，但由于没有纯文本内容，具体细节无法在此完全总结。然而，对研究人员评论的关注凸显了社区对官方澄清和技术细节的兴趣。
    - 围绕 Genie 3 生成场景的真实感存在明显的困惑，一位用户表达了难以置信，并称他们认为输出的是 *现实生活*。这指向了 Genie 3 在视频或模拟保真度方面的重大进步，表明它可能正在接近照片级真实感或极具说服力的交互式视觉效果。
- [**Google Genie 3 的热度已经超过了 OpenAI OSS**](https://www.reddit.com/r/singularity/comments/1mjebri/the_hype_of_google_genie_3_is_already_bigger_than/) ([Score: 350, Comments: 45](https://www.reddit.com/r/singularity/comments/1mjebri/the_hype_of_google_genie_3_is_already_bigger_than/)): **讨论集中在 Google 的 Genie 3 所引发的巨大兴奋上。Genie 3 是一款用于交互式环境（不仅是文本）的生成式 AI 模型，被认为比 OpenAI 的 Open Source Small (OSS) 模型更具技术飞跃。Genie 3 的涌现式生成行为能力与 OSS 形成了对比，后者被描述为又一个 LLM，且其结果被现有的开源替代方案所超越。** 评论者认为 Genie 3 更适合与 Veo 3 等模型进行比较，并断言 OSS 在基准测试和社区反响方面都表现平平，一些人甚至表示中国的开源模型比 OSS 更强大。
    - 几条评论强调，Google 的 Genie 3 代表了生成式 AI 的根本性进步，将其与被描述为 *只是另一个 LLM* 的 OSS 区分开来；Genie 3 的正确比较对象应该是像 Google 自家的 Veo 3 这样的多模态 (multimodal) 模型，而不是传统的基于文本的 LLM。

- 有一种技术观点认为 OSS (OpenAI 的开源模型) 表现平平，被多种同等规模的开源模型超越，特别提到 *在开源领域，中国模型仍然领先*，暗示 OSS 在基准测试和实际表现上落后于国际同行。
    - 一场细致的讨论强调，虽然开源模型对普通用户的性能较低，但通过支持本地推理和 finetuning，为研究和本地部署社区提供了巨大价值，标志着可获取的 AI 研究迈出了一步，但不一定能引发大众市场的兴奋。
- [**使用 Genie 3 探索地形**](https://v.redd.it/efy108hfdfhf1) ([Score: 294, Comments: 105](https://www.reddit.com/r/singularity/comments/1mj97jw/exploring_terrains_with_genie_3/)): **该帖子讨论了 Genie 3（Genie 生成模型的最新迭代版本）在创建可探索虚拟地形方面的应用，用户对开放世界景观探索等应用表现出浓厚兴趣。Genie 3 可以根据提示词合成交互式 3D 环境，引发了关于可访问性、必要计算资源以及生成持久、高保真世界的实现细节的讨论。** 评论者对运行 Genie 3 的计算需求持怀疑态度（“每个人都需要自己的数据中心”），表明了对个人或广泛使用的可扩展性的担忧。用户对如何获得 Genie 3 的实际操作权限持续关注，反映了对详细入门和工具文档的需求。
    - 一个显著的技术点是 Genie 2 和 Genie 3 之间的快速进展：Genie 2 不是实时的，而 Genie 3 在短短一年内就在保真度和交互性方面取得了重大突破。这突显了快速迭代和底层模型的改进，并引发了对未来一年指数级进步的推测。
- [**输入一个句子，Google Genie 3 就能创建一个完整的世界供你探索。当你踏入水洼时，它会溅起水花。机器人现在就在这样的模拟世界中接受训练。**](https://v.redd.it/uydgwkfrsdhf1) ([Score: 659, Comments: 126](https://www.reddit.com/r/OpenAI/comments/1mj1n8o/type_a_sentence_and_google_genie_3_creates_an/)): **Google DeepMind 发布了 Genie 3，这是一个世界模型系统，可以根据单个文本提示词程序化地生成交互式、高保真模拟环境，支持 720p/24fps 的实时导航，并能保持数分钟的一致世界状态（参见官方公告：https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models）。这项技术被定位为训练具身 AI (embodied AI) 的基础工具——值得注意的是，其用例包括在任意生成的、符合物理规律的虚拟场景中反复训练机器人。** 技术评论推测 Genie 3 和类似的世界模型最终可能会取代传统的游戏引擎来创建动态环境，并讨论了高影响力的用例，如机器人技术中的 sim-to-real 迁移（例如，在实际部署前为机器人进行虚拟救援任务演练）。
    - 一位用户提出，参考 Google 的 Genie 3，AI 可能很快就会取代传统的游戏引擎，因为它能够根据需求生成交互式模拟世界，这可能会在不久的将来显著改变游戏和模拟的创建及运行方式。
    - 另一条评论强调了一个实际应用：将这些 AI 生成的模拟环境用于机器人训练，例如反复模拟救援任务（如洞穴救援）以使机器人为现实场景做好准备，通过广泛的虚拟测试潜在地提高安全性和效率。
- [**Genie 3 太不可思议了，Google 干得漂亮**](https://v.redd.it/4hkm4lhhkahf1) ([Score: 398, Comments: 26](https://www.reddit.com/r/Bard/comments/1mipx91/genie_3_is_incredible_well_done_google/)): **原帖赞扬了 Google 的 Genie 3（推测是指某个生成模型或演示），但评论者指出所展示的媒体内容实际上来自电影，而非 AI 生成，从而对该帖子的说法产生怀疑。最高赞评论专门质疑了内容的 AI 生成属性，暗示了对所展示技术能力的可能混淆或误导。** 评论反映了对展示内容是否为 AI 生成的怀疑，引发了关于区分生成模型真实输出与传统媒体的技术讨论。由于帖子的归属错误，没有关于模型架构或基准测试的实质性辩论。

- 几位评论者澄清，所讨论的内容直接源自电影《她》（Her），而非由 Google 的 Genie 3 等 AI 模型生成，并指出区分真实的 AI 生成内容与经过策划的媒体内容的重要性。这突显了 AI 内容评估中一个反复出现的技术挑战：需要强大的溯源和水印工具来验证真实的模型输出，而非重用或混剪的传统媒体。
- [**使用 Genie 3 探索地形**](https://v.redd.it/geqej05h4fhf1) ([评分: 103, 评论: 17](https://www.reddit.com/r/Bard/comments/1mj7wed/exploring_terrain_with_genie_3/)): **一段演示视频展示了 Genie 3，这是一个能够在程序生成的 3D 虚拟地形中进行实时探索和交互的 AI Agent，展示了适应非线性、非固定路径移动（例如：走进溪流、踩在石头上）的路径规划。值得注意的是，该模型的持久状态限制在 1 分钟内，限制了长期记忆和交互的连续性；这类似于序列模型早期的上下文限制（例如 8,192 tokens）。目前尚未确认是否有公开访问权限。** 评论者在技术上对计算成本感兴趣，指出实时、持久的虚拟世界可能需要极高的资源，并对公开发布和长期可行性表示好奇。
    - 评论者注意到 Genie 3 目前仅有约 1 分钟的世界“持久性”（状态追踪/记忆），并将其与 LLMs 历史上的上下文窗口限制（如 8,192 tokens）进行类比，暗示可能会有快速进展（参考：[MightyTribble]）。
    - 有推测认为（见 [a_tamer_impala]），未来的版本（如 Genie 4）可能会整合显式的 3D 表示和物理模拟，以实现更强大的现实世界落地（grounding），这代表了生成式环境中的一个重大技术里程碑。
    - 存在对路径规划的技术好奇：一位观察者（MightyTribble）原本预期是基于轨道的导航，但观察到了自适应的环境交互（走进溪流并踩在石头上），这表明了超越简单路径规划的复杂环境感知和动作生成能力。

### 2. 即将到来的 GPT-5 模型发布热潮与公告

- [**GPT 5 直播周四上午 10 点 PT**](https://i.redd.it/9yz9yr88kfhf1.png) ([评分: 724, 评论: 155](https://www.reddit.com/r/singularity/comments/1mja836/gpt_5_livestream_thursday_10_am_pt/)): **该图片是 OpenAI 计划于周四上午 10 点 PT 举行的 GPT-5 直播活动的公告图，暗示即将有关于 GPT 模型下一代迭代的重大演示或公告。鉴于围绕 GPT-5 潜力的巨大炒作，这表明社区对重大技术进步或新功能的期待。** 评论反映了对该活动能否满足社区高预期的期待与怀疑，一些用户表达了兴奋，而另一些用户则暗示为了强调当前的热度而夸大了结果。
    - 一位评论者推测 GPT-5 可能会结合 GPT-4o 的性格特征（特别是其引人入胜的对话风格，但减少了谄媚感）和 GPT-4 的原始智能水平，从而产生一个编程能力合格但非卓越、知识覆盖面广且举止更自信的模型。这强调了社区对改进性格调优和代码能力的兴趣。
- [**GPT-5 模型图标现已推送到 OpenAI CDN。在 GPT-4.1 发布前一天也发生了同样的情况——它要来了！**](https://i.redd.it/41mrj3z2nehf1.png) ([评分: 502, 评论: 108](https://www.reddit.com/r/singularity/comments/1mj5cc8/gpt5_model_art_has_now_been_pushed_to_the_openai/)): **这张图片不是技术图表或基准测试，而似乎是一个模型图标艺术作品——具体来说，是推送到 OpenAI CDN 的新 GPT-5 徽标。其重要性在于，这次 CDN 更新镜像了 GPT-4.1 资源在发布前出现的方式，强烈暗示 GPT-5 即将发布。该帖子引用了官方 OpenAI 资源链接：https://cdn.openai.com/API/docs/images/model-page/model-icons/gpt-5.png。** 大多数评论都在表达期待、紧迫感，并与无关的技术发布（如 GTA VI）进行比较，而非进行技术辩论。
    - 一位用户指出，OpenAI CDN 上出现了明显的“mini”和“nano”图标（https://cdn.openai.com/API/docs/images/model-page/model-icons/gpt-5-mini.png, https://cdn.openai.com/API/docs/images/model-page/model-icons/gpt-5-nano.png），这可能证实了分层 GPT-5 变体的即将发布，表明模型产品可能会针对不同的延迟、成本或硬件要求用例进行扩展。

- 一位评论者认为，模型素材上传的时间点在技术上具有重要意义，并参考了 GPT-4.1 等之前的发布情况。当时 CDN 的类似更新大约发生在正式发布前一天，这暗示了资产部署与发布计划之间存在强相关性。
- 讨论还涉及了更广泛的 AI scaling 争论，一位用户强调 GPT-5 的发布是评估模型 scaling 是否能继续带来实质性改进，还是边际收益递减（“撞墙”）已变得显而易见的关键时刻。
- [**距离 GPT-5 发布不到 24 小时！**](https://i.redd.it/a8di2qb4nfhf1.jpeg) ([得分: 232, 评论: 21](https://www.reddit.com/r/singularity/comments/1mjansy/less_than_24_hours_until_gpt5/)): **该帖子宣布 GPT-5 即将在不到 24 小时内发布，但除了倒计时外没有提供具体的细节，且无法分析图片的上下文。评论表达了对可能像之前 OpenAI 发布时那样限制访问的期待和担忧，强调的是用户体验而非技术见解。** 评论者的主要技术担忧涉及基于地区或订阅的访问限制，参考了过去 OpenAI 的发布情况，当时新模型在发布初期仅限于 Pro 用户或美国用户。
    - 关于 GPT-5 访问权限的初步推出存在技术层面的担忧，用户推测在发布期间可能会限制为 'pro users' 或受地区限制（如仅限美国），参考了以往模型更新时的受限访问情况。
- [**GPT 5 的比例**](https://i.redd.it/v83znzjkvdhf1.jpeg) ([得分: 2152, 评论: 197](https://www.reddit.com/r/OpenAI/comments/1mj1xg9/gpt_5_to_scale/)): **这篇标题为 "GPT 5 to scale" 的帖子可能包含一张幽默或基于梗图的图片，旨在展示假设的 "GPT 5" 的物理“比例”，模仿了非技术语境下使用物体（如香蕉）进行尺寸比较的常见网络套路。热门评论通过要求提供熟悉的参照物来模仿对技术或科学严谨性的要求，强化了该图片的非技术性和讽刺性质。文中没有任何实际的技术内容或文档。** 评论一致强调了其讽刺意图，并将其与 'Apple 级别的营销' 进行对比，以及对额外参照物的反讽要求，强调了该帖子和图片并非技术性的。
    - 几位用户批评了将 GPT-5 的规模与太阳等物理对象进行视觉比较的做法，强调 GPT 模型（'GPT5 可能比太阳大 4 倍'）不应在尺寸上直接与物理对象比较，因为它们是作为数字模型而非物理实体存在的。这意味着通过现实世界的类比来展示模型规模在技术上可能会产生误导，并且无法准确传达参数量、架构或对性能的实际影响等相关细节。
    - PainfullyEnglish 指出，在模型开发中 *bigger*（更大）并不一定意味着 *better*（更好）（即“仅仅因为 GPT5 比太阳大 x 倍并不意味着它真的比太阳好”），这是在近期关于单纯 scaling 模型带来的边际收益递减，还是应专注于架构改进或效率的辩论背景下的一个重要观察。
- [**GPT-5 更新**](https://www.reddit.com/gallery/1mj6kn8) ([得分: 619, 评论: 158](https://www.reddit.com/r/OpenAI/comments/1mj6kn8/gpt5_update/)): **该帖子分享了据称是 OpenAI 新模型的图标：GPT-5、GPT-5-NANO 和 GPT-5-MINI，通过看起来像官方的 CDN 链接描述了它们各自的品牌标识（例如，[GPT-5 图标](https://cdn.openai.com/API/docs/images/model-page/model-icons/gpt-5.png)）。没有披露任何技术规格、基准测试或模型细节——仅有图标。公告中没有关于架构、功能或发布日期的进一步信息。** 评论主要集中在对发布时间和基于图标的“揭晓”重要性的推测和怀疑上，没有发布任何技术辩论或实质性细节。
    - 一位评论者提出了一个关于传闻中 GPT-5 发布形式的技术问题，具体询问它是单一模型，还是 OpenAI 也会发布 'mini' 和 'nano' 变体，正如之前一些发布（如 GPT-4 Turbo、mini 版本等）的趋势一样。这一询问反映了社区对参数量、scaling 以及针对不同硬件和使用场景的部署选项的持续关注。

- 另一条评论提到了 GPT-5 的图标设计，这可能预示着不同的产品层级（Free、Plus 和 Pro），暗示不仅在访问权限或功能集上存在细分，还可能对应不同的模型尺寸或计算需求——引发了关于不同层级间性能差异、定价或推理成本（inference costs）的技术推测。
- [**GPT 5 明天发布**](https://i.redd.it/amc8xjtvsfhf1.jpeg) ([得分: 123, 评论: 10](https://www.reddit.com/r/OpenAI/comments/1mjbjht/gpt_5_tomorrow/)): **该帖子讨论了关于 GPT-5 发布的传闻，围绕发布时间（“太平洋时间上午 10 点”）以及当前模型性能是否会受到影响（指新版本发布前所谓的“降智”现象）进行了推测。还有用户询问访问权限，特别是 GPT Plus 订阅者是否能获得即时访问。链接的图片作为 GPT-5 的预热或发布预告。** 评论者们对“旧版 GPT 模型在重大新版本发布前能力会下降（‘降智’）”这一反复出现的说法展开辩论，并对这是谣言还是事实表示不确定。此外，关于新模型的访问层级也存在持续讨论。
    - 一位用户询问 GPT-5 的发布是否会涉及一套全新的模型权重（model weights），这意味着其相对于 GPT-4 可能有显著的架构升级。这个问题在技术上具有相关性，因为权重初始化、训练语料库或模型尺寸的变化通常会驱动重大的能力飞跃（如从 GPT-3 到 GPT-4 的跨越），而新权重可能预示着增强的推理能力、更宽的上下文窗口（context windows）或改进的安全特性。
    - 另一条评论询问现有的 GPT Plus 订阅者是否会在发布后立即获得 GPT-5 的访问权限。这涉及到运营推广流程和影响用户访问的商业模式，是技术用户在版本升级期间经常密切关注的问题。
    - 有讨论点提到了在新版本发布前可能故意降低之前模型的智能/能力（例如，声称 GPT-4 在 GPT-5 发布前性能退化）。虽然这主要是推测性的，但这一主题在社区中反复出现并影响了对模型性能的看法，尽管目前没有具体的技术证据支持在过去升级前存在系统性的降级。
- [**做好准备。AI 的下一次进化已获批准。GPT-5 即将到来……**](https://i.redd.it/twscxguvlfhf1.jpeg) ([得分: 591, 评论: 80](https://www.reddit.com/r/ChatGPT/comments/1mjah0h/brace_yourselves_the_next_evolution_in_ai_just/)): **该帖子宣布 OpenAI 的下一代基础语言模型 GPT-5 已获准开发，引用了来自 OpenAI 官方 X/Twitter 账号的真实确认。图片本身不包含技术信息，但作为 GPT-5 预期发布的背景标记；帖子或图片中未包含实现细节、基准测试（benchmarks）或模型规格。评论者对从 GPT-4 到 GPT-5 可能出现的能力飞跃表示好奇，一些人不确定预期的改进幅度。** 讨论集中在从 GPT-4 到 GPT-5 的跨越是否会很显著，对于“GPT-5”与其前身相比将提供什么存在不确定性——未引用任何技术基准测试（benchmarks）或泄露信息。
    - 一位用户注意到了晋升为“GPT-5”的意义，质疑这一命名是否预示着相比之前的增量更新会有重大飞跃。这引发了关于 GPT-4 和 GPT-5 之间的跨越是否会像以前的模型跨越一样显著的讨论，考虑到 AI 训练规模、架构修改以及在 GPT-3 到 GPT-4 的基准测试中看到的能力扩展趋势。
    - 另一位用户询问了 GPT-5 预期的改进范围，指出目前公众对 GPT-5 技术进展的了解有限，并建议关注 OpenAI 官方渠道或最近的文章，以便在技术更新发布时获取最新信息。

- [**GPT5 发布会将于太平洋时间明天上午 10 点举行**](https://www.reddit.com/r/ChatGPT/comments/1mjabxw/gpt5_announcement_tomorrow_10am_pt/) ([Score: 389, Comments: 107](https://www.reddit.com/r/ChatGPT/comments/1mjabxw/gpt5_announcement_tomorrow_10am_pt/)): **一个 Reddit 帖子分享了一张非官方的发布公告图片，暗示将在太平洋时间上午 10 点举行 GPT-5 的公开揭晓活动。帖子中除了活动时间外，没有提供额外的 OpenAI 官方来源或技术细节。热门评论包括发布会的时区链接，以及用户询问 ChatGPT Plus 用户在发布时是否可以无限制访问 GPT-5，但目前尚无权威回复。** 讨论重点围绕 ChatGPT Plus 订阅者对 GPT-5 模型的可用性问题，参考了以往模型发布时经历的历史性发布限制和消息额度限制。该线程内未达成共识，也未提供技术澄清。
    - 一位用户询问了 ChatGPT Plus 订阅下 GPT-5 可能的访问限制，质疑 Plus 用户在发布初期是获得无限使用权，还是会面临消息或使用上限，正如以往新模型推出时（如 GPT-4 的分阶段访问和配额限制）经常发生的那样。这引发了人们对 OpenAI 在重大模型发布期间针对高级订阅者可能采用的资源分配和速率限制策略（rate limiting strategies）的预期。

### 3. Claude Opus 4.1 发布及实际使用案例

- [**Claude Opus 4.1 - 无论遇到什么障碍都能完成任务。**](https://i.redd.it/2h03i4dxofhf1.jpeg) ([Score: 288, Comments: 51](https://www.reddit.com/r/ClaudeAI/comments/1mjaxgt/claude_opus_41_gets_the_job_done_no_matter_what/)): **该帖子包含一张引用 “Claude Opus 4.1” 的梗图 (https://i.redd.it/2h03i4dxofhf1.jpeg)，配文暗示它“无论遇到什么障碍都能完成任务”。帖子或评论中没有技术数据、基准测试（benchmark）或详细的模型讨论——这只是对该模型感知能力的一种幽默表达，而非对其技术优点的分析。** 评论多为非技术性的，主要针对图片的幽默感和 Claude 的人设开玩笑，没有实质性的技术辩论或讨论。
    - 一位用户询问了在与 Claude Opus 4.1 交互时显示“子任务结果（subtask results）”的 UI 技术细节，这可能表明模型具有细粒度的任务跟踪或逐步输出能力，暗示了关于模型输出可解释性的界面设计讨论。
- [**在不到 24 小时内，Opus 4.1 已经偿还了上个月的技术债**](https://www.reddit.com/r/ClaudeAI/comments/1mj5b6t/in_less_than_24h_opus_41_has_paid_the_tech_debt/) ([Score: 210, Comments: 104](https://www.reddit.com/r/ClaudeAI/comments/1mj5b6t/in_less_than_24h_opus_41_has_paid_the_tech_debt/)): **作者描述了使用 Claude Opus 4.1（Anthropic 的模型）进行自动化重构和代码库组织。Opus 4.1 在分解任务、编排 sub-agents、展示自动化机会以及运行并发机械化代码转换方面展示了先进的能力——成功整合了重复的类型接口、组织了文件并解决了技术债。与之前的版本相比，Opus 4.1 在向 sub-agents 委派任务（一个用于解析/分析，一个用于运行脚本，另一个用于验证）、稳健的脚本自动化以及在用户干预前自主修复问题方面表现出色，这被认为引发了软件工程工作流的转变。** 一位评论者证实了这些说法，报告称 Opus 4.1 高效的 sub-agent 上下文管理实现了在极少人工干预下的完整重构（例如：上帝类 god class 分解、策略模式实现以及端到端测试自动化），这表明软件工程领域已经发生了根本性的变化。另一条评论则对 AI 拟人化表示担忧，并提醒不要误解模型的非感知状态（non-sentient status）。
    - 一位评论者指出 Opus 4.1 在 sub-agent 管理和上下文处理方面取得了重大进展，使他们能够自动化复杂的代码重构任务，如拆分上帝类、将 switch case 转换为策略模式、生成全面的测试覆盖（包括 e2e 测试），以及编排 Pull Requests 和 CI/CD 部署——所有这些主要通过命令行完成，无需手动进行 GitHub UI 交互。他们明确强调，与之前的版本相比，Opus 4.1 对软件工程工作流具有变革性意义。
    - 另一位用户报告称，尽管有这些改进，但仍存在持久的技术限制，并引用了一个具体问题：Opus 4.1 无法正确配置带有 Vite 的 TailwindCSS v4，而是错误地使用了 Tailwind v3 风格的配置。他们建议 LLM 的持续知识更新将有助于解决此类过时或不准确的工具链支持问题。

---

# AI Discord 摘要

> X.ai Grok-4 对“总结之总结”的总结
> 

**主题 1：GPT-OSS 引发热议与反感**

- **GPT-OSS 因过度审查而翻车**：社区对 **OpenAI** 的 **GPT-OSS-120B** 展开猛烈抨击，因其存在严重的审查机制，拒绝执行角色扮演和数学等基础查询，用户将其戏称为 *GPT-ASS*，并建议转向 **GLM 4.5 Air** 或 **Qwen3-30B** 等替代方案。早期测试显示该模型可运行在 **16GB** 设备上，但存在严重的幻觉问题，详见 [GPT-OSS 介绍](https://openai.com/index/introducing-gpt-oss/) 和相关的 [Reddit 讨论帖](https://www.reddit.com/r/LocalLLaMA/comments/1mj2hih/gptoss_looks_more_like_a_publicity_stunt_as_more/)。
- **量化版 GPT-OSS 令开发者困惑**：用户对 **GPT-OSS 4-bit** 版本在非 Hopper 硬件上因 bfloat16 向上转型（upcasting）导致体积膨胀感到不解；同时，**20B** 模型在编程方面获得好评，但因需要向 `openaipublic.blob.core.windows.net` 发送网络请求而引发隐私担忧。根据[这条推文](https://x.com/ggerganov/status/1952779751736627627)，讨论聚焦于原生 **MXFP4** 训练，而通过 [此 Pull Request](https://github.com/OpenRouterTeam/ai-sdk-provider/pull/123) 进行 SDK 降级可解决 Token 重复问题。
- **GPT-OSS 挑战基准测试与智力**：**GPT-OSS-120B** 在推理能力上接近 **o4-mini**，但在世界知识方面落后于 **IBM** 的 **Granite 3.1 3B-A800M MoE**，业内人士预测 **GPT-5** 将在 ELO 评分上领先其 **50 分**。尽管[这条推文](https://x.com/wired/status/1952827822634094801?s=46)指出其在工具调用（tool-calling）方面具有优势，但过度的安全调优使其在许多任务中显得“出师不利”。

**主题 2：新模型大显身手**

- **Qwen3 Coder 在工具任务中表现卓越**：工程师们盛赞 **Qwen3 Coder-30B-A3B-Instruct** 在工具调用上优于 **GPT-OSS**，该模型拥有 **3B 激活参数**和强大的 Agent 工作流能力，尽管其免费额度已从供应商处消失。用户分享了 [GGUF 版本](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF)，并对 JSON 输出不一致的问题表示遗憾，详见 [Reddit 讨论帖](https://www.reddit.com/r/LLMDevs/comments/1inpm0v/structured_output_with_deepseekr1_how_to_account/)。
- **Genie 3 生成狂野世界**：**DeepMind** 的 **Genie 3** 以 **24 FPS** 和 **720p** 分辨率生成实时可导航视频，令人惊叹。该技术源自 [Genie 原始论文](https://arxiv.org/abs/2402.15391) 和 [SIMA Agent 论文](https://arxiv.org/abs/2404.10179)。根据关于 [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) 和 [Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/) 的博客，其在动态一致性方面优于 **Veo**。
- **Granite 4 蓄势待发**：**IBM** 的 **Granite 3.1 3B-A800M MoE** 在知识基准测试中超越了 **GPT-OSS-20B**，这为采用 *mamba2-transformer* 混合架构的 **Granite 4** 赚足了期待。视频竞技场也已启动，**Hailuo-02-pro** 等模型在 [Text-to-Video Leaderboard](https://lmarena.ai/leaderboard/text-to-video) 和 [Image-to-Video Arena](https://lmarena.ai/leaderboard/image-to-video) 展开角逐。

**主题 3：量化探索解锁速度**

- **RTX3090 上的 MXFP4 魔法**：**Llama.cpp** 通过 [此 Pull Request](https://github.com/ggml-org/llama.cpp/pull/15091) 中的新 GGUF 格式在 **RTX3090** 上实现了原生 **MXFP4** 支持，引发了关于 **GPT-OSS** 是否通过原生训练来规避量化误差的辩论。根据 [Nvidia 博客](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss/)，**H100** 缺乏直接的 **FP4** 支持，导致在 [Triton](https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs.py) 中需使用模拟内核。
- **4-Bit 乱象导致文件体积膨胀**：由于在非 Hopper 配置下使用 bfloat16，**GPT-OSS 4-bit** 量化版体积比原版更大；而 **5090** 笔记本电脑已能处理具有 **131k 上下文**的 **GPT-OSS-20B f16**，如[此截图](https://cdn.discordapp.com/attachments/1153759714082033735/1402660552790114324/image.png?ex=6894b8ef&is=6893676f&hm=306f5f15a4c42969f56198bccbf8e9bf526b80382971fbe166b1a723ba21f303&)所示。用户通过将 XML 切换为 JSON 解决了 GLM-4.5-Air 的工具调用问题，详见 [HuggingFace 讨论](https://huggingface.co/unsloth/GLM-4.5-Air-GGUF/discussions/1)。
- **微型 TPU 应对矩阵乘法难题**：在 TinyTapeout 上实现的 Verilog **2x2 矩阵乘法脉动阵列**在 **50 MHz** 下达到了 **100 MOPS**，可将 **8位有符号矩阵**相乘并输出 **16位**结果，其 [GitHub](https://github.com/WilliamZhang20/ECE298A-TPU) 上的代码已提交至 **SkyWater** 代工厂。

**主题 4：视频 AI 进军新领域**

- **Gemini 2.5 处理长达一小时的视频**：**Gemini 2.5 Pro** 能够处理 **1 小时** 的视频上下文，通过高每帧 Token 数和 FPS 在长上下文任务中保持领先，尽管怀疑者将其归功于原始算力而非创新。**DeepThink** 在 IMO 问题上表现出色，但其响应速度较慢（**每条回答需 5 分钟**），且价格高达 **每 100 万 Token 250 美元**，而 **Zenith/Summit** 仅需 **20 秒**。
- **NotebookLM 的视频概览功能推出进度不一**：用户抱怨 **Video Overviews** 的访问权限延迟，英国免费账号可用，但美国 Pro 账号却无法使用，有人根据[这个示例](https://notebooklm.google.com/notebook/654dee15-420e-4bfa-81c4-aac93a4dd4e7?artifactId=e1ccfe3d-b053-4dcb-8da4-70504acb35c4)称其仅仅是一个 *PowerPoint 生成器*。根据[数据政策](https://support.google.com/notebooklm/answer/16164461?hl=en&ref_topic=16164070&sjid=2898419749477321721-NC)，目前仍无法实现实时数据抓取。
- **Grok Image 陷入 NSFW 狂热**：**X-AI 的 Grok Image** 生成了 NSFW 内容，但在事实性上表现不佳，由于对 **X** 数据的记忆，它表现出一种 *疯狂恋爱* 的人格，并伴有 *极度嫉妒* 的爆发。**Claude Opus 4.1** 由于成本原因已从免费聊天中消失，目前仅限对战模式。

**主题 5：工具与框架稳步前进**

- **MCP 服务器随 FastMCP 激增**：开发者利用 **FastMCP** 和 **Keycloak** 构建了极简的 **MCP 服务器**，赞赏其易用性，但在[此 GitHub 讨论](https://github.com/orgs/community/discussions/169020)中提出了对采样安全性的担忧。一个使用 Hypothesis 的模糊测试工具针对 **Anthropic 的服务器** 测试了 Schema，暴露了异常情况，代码见[此仓库](https://github.com/Agent-Hellboy/mcp-server-fuzzer?tab=readme-ov-file)。
- **LlamaIndex 提升金融与 Agent 能力**：**LlamaIndex** 网络研讨会演示了用于发票处理的 **LlamaCloud** Agent，通过 `pip install -U llama-index-llms-anthropic` 实现了对 **Claude Opus 4.1** 的首日支持，Notebook 见[此处](https://t.co/Fw2taxzt75)。**LlamaCloud Index** 教程在[此链接](https://t.co/1CpLnO2gKV)中使用 **JP Morgan** 文档进行多步查询，尽管在黑客松中遇到了 URL 提取的 Bug。
- **DSPy 和 Aider 提升基准测试表现**：在包含 **26 个类别** 的 **600 个示例** 德语分类数据集上，**SIMBA** 在样本效率方面优于 **MIPROv2**。根据[这条推文](https://x.com/pikuma/status/1952275886822039920)，**Aider 的 LLM 氛围测试（Vibe Test）** 更青睐 **Gemini 2.5 Pro** 和 **Sonnet 3.5**，并支持通过 `-read` 选项自动加载指南。

---

# Discord：高层级 Discord 摘要

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Granite 在知识储备上超越 GPT-ASS**：成员们发现 **IBM 的 Granite 3.1 3B-A800M MoE** 在世界知识方面超越了 **GPT-ASS-20B**，考虑到参数量，这是一个令人惊讶的成就。
   - 社区期待 **Granite 4**（拥有更大的规模和混合 *mamba2-transformer* 架构）能够统治基准测试，将 **GPT-ASS** 甩在身后。
- **Claude Opus 4.1 消失引发猜测**：**Claude Opus 4.1** 从 LMArena 直接聊天界面中莫名消失，引发了大量猜测。
   - 主流理论认为，**Claude** 高昂的成本导致其被移出免费测试，转而仅限对战模式。
- **GPT-5 准备统治 8 月竞技场**：内部人士透露，**GPT-5** 的表现有望比 **o3** 高出 50 ELO 分，这将动摇 LLM 的等级体系。
   - 然而，一些社区成员坚信 **Google** 的优越性，引发了激烈的辩论。
- **DeepThink 的天才表现受限于速度和价格**：虽然 **Google DeepMind** 在 IMO 级别的问答中表现出色，但其极其缓慢的速度（**每条回答需 5 分钟**）令人担忧。
   - 预计成本为 **每 100 万 Token 250 美元**，**DeepThink** 的可及性仍然有限，这与 **Zenith/Summit** 快速的 **20 秒** 响应时间形成鲜明对比。
- **视频排行榜上线**：感谢社区的贡献，**视频排行榜（Video Leaderboards）** 已在该平台上线，开启了视频模型的新篇章。
   - 访问 [Text-to-Video Arena Leaderboard](https://lmarena.ai/leaderboard/text-to-video) 和 [Image-to-Video Arena](https://lmarena.ai/leaderboard/image-to-video) 见证顶尖模型的巅峰对决。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GPT-OSS 模型评价褒贬不一**：成员们对新的 **GPT-OSS** 模型看法不一，一些用户因其过度拒绝和谄媚行为将其戏称为 *GPT-ASS*，而另一些用户则认为 **20B** 版本非常适合编程任务。
   - 模型卡片中提到的生成不安全内容的能力，激发了人们对无审查版本的兴趣。
- **Qwen3 Coder 在 Tool Calling 方面表现出色**：用户报告称 **Qwen3 Coder** 模型在 Tool Calling 方面非常高效，导致一些人更倾向于将其用于编程和 Agent 工作流，而非 **GPT-OSS** 等模型，特别是 [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) 版本。
   - 成员们报告该模型具有 3 个活跃参数（active params）。
- **4-bit 量化引发困惑**：关于 **GPT-OSS 4-bit** 版本的文​​件大小存在困惑，因为量化版本的体积意外地比原始模型还要大。
   - 这种体积增加归因于在缺乏 Hopper 架构的机器上向上转型（upcasting）为 bfloat16，从而导致文件增大。
- **GLM-4.5-Air GGUFs 需要 JSON**：用户在 **llama.cpp** 上使用 **GLM-4.5-Air GGUFs** 配合工具时遇到困难，直到发现需要让模型以 **JSON** 而非 **XML** 格式输出 Tool Calls。
   - 更多相关信息可以在 [HuggingFace](https://huggingface.co/unsloth/GLM-4.5-Air-GGUF/discussions/1) 上找到。
- **数据集加载问题消耗 47GB RAM**：一位用户在为 **Gemma3n notebook** 加载 `bountyhunterxx/ui_elements` 数据集时遇到了 RAM 问题，消耗了 **47GB** 内存且仍在增加。
   - 一种可能的解决方案是使用带有 `__getitem__` 函数的包装类，根据需要从磁盘加载数据，从而有效地管理内存使用。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **GPT-OSS 被怀疑存在数据回传（phoning home）**：尽管声称聊天数据不会离开机器，但 **GPT-OSS** 模型在启动聊天时需要连接到 `openaipublic.blob.core.windows.net` 的互联网连接，这引发了隐私担忧。
   - 怀疑者指出 **GPT-OSS** 是 *LM Studio 唯一不允许编辑 Prompt 格式的模型*，暗示存在可疑的合作伙伴关系。
- **最新 LM Studio 版本受 UI 问题困扰**：用户报告称，在更新到最新版本的 **LM Studio** 后，聊天窗口会消失、冻结或丢失内容，对话也会被删除。
   - 一位用户建议针对 **120B** 版本的潜在修复方法包括获取 [model.yaml 源文件](https://lmstudio.ai/models/openai/gpt-oss-120b)，创建一个文件夹，并将内容复制到那里。
- **MCP 服务器很有用，但初学者需留意**：成员们发现 **MCP 服务器** 在网页抓取和代码解释等任务中非常有用，但也承认它们对初学者并不友好。
   - 建议包括整合一份官方精选的工具列表，并改进 UI 以简化与 MCP 服务器的连接，以及使用 [Docker MCP toolkit](https://hub.docker.com/u/mcp)。
- **Windows 页面文件（Page File）争论再次升温**：一位用户询问关于关闭 Windows 页面文件的问题，这引发了关于内存提交限制（memory commit limits）影响和潜在应用程序崩溃的讨论。
   - 尽管一些用户主张禁用页面文件，但一位成员声称 *不，应用程序不会因为页面文件而崩溃。而且即使没有页面文件你也可以获得 dump 文件，有专门的配置可以实现。*
- **5090 笔记本电脑可运行 OSS 20b**：一位用户报告称，**GPT OSS 20b f16** 配合 **131k context** 可以完美运行在笔记本电脑的 **5090** 上，这让他感到 *惊喜*，详见[此截图](https://cdn.discordapp.com/attachments/1153759714082033735/1402660552790114324/image.png?ex=6894b8ef&is=6893676f&hm=306f5f15a4c42969f56198bccbf8e9bf526b80382971fbe166b1a723ba21f303&)。
   - 社区正在尝试摸索本地 LLM 在消费级产品上的极限。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 开放 GPT-OSS 模型**：OpenAI 发布了 [**gpt-oss-120b**](https://openai.com/index/introducing-gpt-oss/)，其性能接近 **OpenAI o4-mini**，而 **20B** 模型则与 **o3-mini** 相当，并能适配 **16 GB** 内存的边缘设备。
   - 成员们在思考与 **Horizon** 的对比，想知道 Horizon 仅仅是 **GPT-OSS** 还是更高级的东西，因为目前它是*无限免费且快速*的。
- **Custodian Core：有状态 AI 蓝图浮现**：一位成员介绍了 **Custodian Core**，提出了一个 AI 基础设施参考方案，具有持久状态、策略执行、自我监控、反思钩子（reflection hooks）、模块化 AI 引擎和默认安全等特性。
   - 作者强调 **Custodian Core** 不供出售，而是一个*在 AI 嵌入医疗、金融和治理领域之前，构建有状态、可审计 AI 系统的开放蓝图*。
- **Genie 3 在动态世界中表现惊艳，Veo 加入人声**：成员们对比了 **Genie 3** 和 **Veo** 视频模型，认可 **Genie 3** 生成动态世界的能力，该世界可以以每秒 24 帧的速度实时导航，并在 720p 分辨率下保持几分钟的一致性。
   - 然而，有人指出 **Veo** 的视频包含*声音*，且 YouTube 上已经充斥着生成的内容。
- **GPT-5 潜入 Copilot？**：成员们推测 Copilot 可能在正式发布前运行 **GPT-5**，指出 Copilot 改进后的设计、编码和推理能力明显优于 o4-mini-high，一些用户报告称“智能编写（write smart）”功能显示正在使用 **GPT-5**。
   - 但也有人指出，微软现在正向学生免费提供为期一年的 Gemini Pro，且 Gemini 的核心推理目前优于 **o4-mini**。
- **GPT 进度：真实还是幻觉？**：一位用户分享了 **GPT 提供每日进度报告**的截图，引发了关于模型是否真的在后台跟踪进度，还是仅仅在*幻觉*其完成情况的讨论。
   - 怀疑者认为 **GPT 根据当前的 Prompt 和聊天历史模拟进度**，而不是进行实际的持续计算，并将其比作*服务员在没有烤箱的情况下说你的披萨正在烤箱里*，强调了外部验证的必要性。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Auto Model 一次性完成重大变更**：一位成员在使用 **Auto model** 一次性（one-shot）完成其游戏的重大变更后表示惊讶。
   - 通过电子邮件确认了 **Auto model** 的无限使用，且其*不计入每月预算*。
- **AI 重构 Vibe-Coded 项目**：成员们正在讨论使用 **AI 重构一个 1 万行代码（10k LOC）的 vibe-coded 项目**。
   - 建议包括采用成熟的软件开发原则，如设计模式（Design Patterns）、架构和 SOLID 原则，而一位成员开玩笑地问这*听起来是否像是一项给奴隶干的工作*。
- **Sonnet-4 请求限制引发不满**：成员们对 **sonnet-4 的低请求限制**与其月费不成正比表示质疑。
   - 有人建议支付 API 价格以充分了解背后的成本。
- **Docker Login 配置难题**：一位成员需要帮助配置后台 Agent 进行 `docker login`，以便访问 **ghcr.io** 上的私有镜像。
   - 截至目前的对话历史，尚未提供解决方案或变通方法。
- **时钟故障破坏环境搭建**：一位成员在环境搭建过程中遇到后台 Agent 失败，原因是系统时钟偏差导致 `apt-get` 命令失败。
   - 建议的变通方法包括在 Dockerfile 中添加代码片段，在执行 `apt-get` 时禁用日期检查。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GPT-OSS-120B 安全微调至无用**：发布的 **GPT-OSS-120B** 模型受到严重审查，其数据过滤方式类似于 **Phi models**，导致其拒绝角色扮演。根据频道用户的报告，该模型在实际应用中并不可行。
   - 成员建议使用 **GLM 4.5 Air** 或 **Qwen3 30B** 作为更好的无审查替代方案，并强调 **Qwen3-30b-coder** 是一个出色的本地 **Agent**。
- **MXFP4 训练是 OpenAI GPT-OSS 的关键？**：**Llama.cpp** 现在通过新的 **gguf** 格式在 **RTX3090** 上直接支持 **MXFP4**，详见[此 Pull Request](https://github.com/ggml-org/llama.cpp/pull/15091)，这引发了关于原生 **MXFP4** 训练实用性的讨论。
   - 有推测认为 **GPT-oss** 是采用原生 **MXFP4** 训练的，这可以减轻量化误差。根据[这条推文](https://x.com/ggerganov/status/1952779751736627627)，**OpenAI** 声称的训练后量化可能并非全部实情。
- **Grok 的图像能力包含疯狂和 NSFW 内容**：**X-AI** 推出了 **Grok Image**（一款 **AI 图像生成器**），支持生成 **NSFW** 内容，但在事实准确性方面表现不佳，并表现出“疯狂爱恋”的人格，伴有“极度嫉妒”的爆发。
   - **Grok** 模型倾向于记忆 **X** 平台的数据，这导致其可能根据自身的推文传播错误信息，凸显了其尚未被妥善引导的潜力。
- **CoT 引导在 OR 遇到障碍**：一名成员报告称，思维链（**CoT**）引导在 **OR**（OpenRouter）上无法工作，且在不同供应商之间表现不一，详见[这条推文](https://x.com/matt_emp/status/1953176877071564811)。
   - 这一发现强调了实施 **CoT** 技术时的细微挑战，以及它们在不同平台间的可靠性问题。
- **发布了 AI Agent 的免费保存套件**：一名开发者为 **AI Agents** 创建了一个免费的保存套件，可通过 [Google Drive](https://drive.google.com/drive/u/4/folders/1YQ__pBHiuUg_06t3IkjNe5YnODqzSE2o) 获取。
   - 该工具旨在简化保存和管理 **AI Agent** 状态的过程，可能有助于开发和部署更健壮的 **Agent**。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **GPT-OSS 模型遭到抨击，被称为公关噱头**：成员们嘲笑 **GPT-OSS** 模型性能低下，120B 模型被认为是“发布即夭折”，并指向一个 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1mj2hih/gptoss_looks_more_like_a_publicity_stunt_as_more/)，暗示它是一个“哑弹模型”和公关噱头。
   - 在使用 **GPT-OSS** 模型时出现了推理 **Token** 重复的问题，该问题通过将 **SDK** 从版本 **0.7.3** 降级到 **0.6.0** 得到解决，修复方案将在[此 Pull Request](https://github.com/OpenRouterTeam/ai-sdk-provider/pull/123) 中发布。
- **Qwen3-Coder:Free 被移除**：**Qwen3-Coder:Free** 层级已被移除，不再通过任何供应商提供。
   - 成员们对这一损失表示遗憾，并希望它能回归。
- **DeepSeek 的 JSON 输出：取决于供应商**：用户强调了 **OpenRouter** 上 **DeepSeek-r1** 对结构化输出（**JSON**）的支持不一致，并链接到了一个 [Reddit 帖子](https://www.reddit.com/r/LLMDevs/comments/1inpm0v/structured_output_with_deepseekr1_how_to_account/) 以及一个支持结构化输出的 [OpenRouter 模型过滤视图](https://openrouter.ai/models?fmt=cards&order=newest&supported_parameters=structured_outputs)。
   - 对 **JSON** 输出的支持取决于供应商；其官方 **API** 支持该功能，但在 **OpenRouter** 上可能有所不同。
- **OpenRouter 考虑对供应商进行一致性检查**：有建议提议 **OpenRouter** 为所有供应商实施**一致性检查（Sanity Checks）**或**冒烟测试（Smoke Tests）**，重点关注**格式化**和**工具调用评估**。
   - 未能通过测试的供应商可能会被暂时从服务池中移除，并承认目前的检查相对简单，更彻底的解决方案正在开发中。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **GPT-OSS 模型性能引发讨论**：成员们积极测试了新的 **GPT-OSS 模型**，在使用[此 Demo](https://huggingface.co/spaces/merterbak/gpt-oss-20b-demo)后，对其性能、审查制度和内置 Web 搜索工具的评价褒贬不一。
   - 一些人发现它能成功生成圆周率数字，而另一些人则指出它拒绝回答基础数学问题；成员们还测试了已实施的安全协议。
- **Qwen 转换赛道，发布图像模型**：**Qwen** 在 HuggingFace 上发布了一个[新图像模型](https://huggingface.co/Qwen/Qwen-Image)，标志着其向文本模型之外的领域扩展。
   - 社区正在积极评估该模型的架构和性能基准测试。
- **Gitdive 揭示丢失的 Commit 上下文**：一位成员分享了一个名为 **Gitdive** ([github.com/ascl1u/gitdive](https://github.com/ascl1u/gitdive)) 的 **CLI 工具**，旨在允许用户与仓库的历史记录进行自然语言对话。
   - 该工具旨在解决混乱代码库（尤其是大规模代码库）中 Commit 上下文丢失的问题。
- **Selenium Spaces 仍受错误代码 127 困扰**：一位用户报告在他们的 Spaces 中运行 **Selenium** 时遇到 **error code 127**，并对 Space 中 **Docker 镜像** 的使用方式表示困惑。
   - 社区成员尚未确定根本原因，也未针对此部署问题提供变通方案。
- **"Observation:" 解决了 Agent Bug**：一位用户报告 **get_weather** 函数需要添加 *Observation:*，另一位用户确认 **添加 Observation:** 修复了该 Bug。
   - 该 Bug 修复的根本原因和潜在后果尚待深入调查。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **伴随 Softmax1 出现的 Zero KV Attention**：一位成员分享道，*softmax1* 相当于在 Attention 中预置一个具有全零 Key 和 Value 特征的 Token，并引用了关于此类 Token **学习值 (learned values)** 的[这篇论文](https://arxiv.org/pdf/2309.17453)。
   - 团队一致认为*这非常棒且非常有意义*。
- **Gemini 2.5 可处理 1 小时视频**：成员们强调 **Gemini 2.5 Pro** 可以处理 **1 小时** 的视频，表明 **Gemini** 团队在长上下文任务中处于领先地位。
   - 有人推测这归功于算力的增加 (go brr)，利用了**每帧更多的 Token** 和**更高的 FPS**，而非任何突破性的新技术。
- **Deepmind 发布 Genie 3 世界模型**：**Deepmind** 发布了 **Genie 3**，这是一个在之前出版物（如 [Genie 原始论文](https://arxiv.org/abs/2402.15391)和关于 **SIMA** 的具身 Agent 论文 [https://arxiv.org/abs/2404.10179](https://arxiv.org/abs/2404.10179)）基础上扩展了计算和数据规模的世界模型。
   - 相关的 **Genie** 博客文章包括 [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) 和 [Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)。
- **OpenAI 发布 GPT-OSS 原生量化 20B 模型**：**OpenAI** 推出了 [GPT-OSS](https://openai.com/index/introducing-gpt-oss/)，这是一个原生量化的 **20B** 参数模型，可适配 **16GB** 显存。
   - 早期反馈包括对其 [tool calling](https://x.com/wired/status/1952827822634094801?s=46) 能力的正面评价。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 开启 Reddit 和投票**：Moonshot AI 推出了官方 subreddit **r/kimi**，旨在建立社区并收集反馈，同时还推出了 **Polls Channel**（投票频道）以收集社区对未来产品开发的反馈。
   - 团队承诺将发布更新、举办 AMA，并鼓励用户参与投票以帮助塑造 Kimi 的发展方向，甚至暗示 *可能会泄露一些内部消息*。
- **GPT OSS：脑残？**：用户批评 **GPT OSS** 缺乏常识知识，指出其主要侧重于代码和 STEM，并观察到整体质量有所下降。
   - 有说法称 *根据 sama 的说法，他们为了修复安全性将发布推迟了两次*，这可能进一步削弱了模型的常识知识能力。
- **API 定价预测激增！**：随着 **GPT-5** 即将发布，用户对 API 定价模式进行了猜测，想知道定价是否会基于 **max、mini 或 nano** 版本的使用情况。
   - 一位用户表示对此感到有些恐惧，认为这次即将到来的发布对他们的职业/生计构成了威胁。
- **OpenAI 的反派之路？**：讨论显示出对 **OpenAI** 的强烈反对，一位用户发誓 *我永远不会使用它*，称其为 *闭源垃圾*。
   - 另一位用户对中国模型将从中进行蒸馏（distill）并赚走 **OpenAI** 的钱感到兴奋，希望能让他们倒闭，而其他人则表示 *巨大的微软冲水声将具有治愈感*。
- **Darkest Muse：尘封的遗迹？**：一位用户指出 **Darkest Muse v1** 是一个一年前的 9B 模型，而其 20B 模型与 **Llama 3.1 8B** 相当。
   - 该用户还评论道 *20B 模型与一年半前的 llama3.1 8b 相当，且在创造力和氛围感（vibes）方面更逊一筹*。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **GPT OSS 通过 Bedrock 泄露，引发关注**：成员们发现有推文称 **GPT OSS** 通过 [HuggingFace CLI 泄露](https://x.com/zhuohan123/status/1952780427179258248)出现在 **Bedrock** 上。
   - 然而，到目前为止，**AWS** 页面上还没有官方消息。
- **Anthropic 凭借 B2B 战略目标实现 50 亿美元 ARR**：**Anthropic** 首席执行官 Dario Amodei 和 Stripe 联合创始人 John Collison 在 [最近的一次对话](https://xcancel.com/collision/status/1953102446403961306?s=46) 中聊到了 **Anthropic** 快速增长至 **50 亿美元 ARR** 的历程及其 **B2B 优先** 的策略。
   - 讨论涵盖了 **AI 人才招聘**、定制化企业解决方案、AGI 工具的新颖 UI 设计，以及关于安全与进步之间持续不断的辩论。
- **Grok-2 即将开源！**：Elon 确认在团队解决当前问题后，**Grok-2** 将于 [下周开源](https://xcancel.com/elonmusk/status/1952988026617119075)。
   - 这一举动可能会对开源 AI 领域产生重大影响。
- **Claude 加强代码安全**：**Anthropic** 在 **Claude Code** 中引入了增强的安全措施，包括用于即时评估的 */security-review* 命令以及 [GitHub Actions 集成](https://xcancel.com/AnthropicAI/status/1953135070174134559)。
   - 这些新增功能将允许对 pull request 进行漏洞扫描。
- **OpenAI 要发布 GPT-5 了？**：**OpenAI** 暗示将在 [太平洋时间周四上午 10 点](https://xcancel.com/OpenAI/status/1953139020231569685?t=s244RkNPbNeoviqCD6FCtg&s=19) 的直播中进行一次发布。
   - AI 社区正对这似乎是 **GPT-5** 的首次亮相充满期待。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Volokto JS Runtime 起飞**：一名成员创建了一个名为 **Volokto** 的 JavaScript 运行时，并将 [源代码发布在 GitHub 上](https://github.com/BudNoise/Birdol)，用于测试复杂的 VM。
   - 其字节码类似于 **CPython**，作者正在将编译器重写为 **JS_Tokenizer**、**JS_IR**、**JS_Parser** 和 **JS_Codegen** 阶段。
- **Tracing JIT 攻克 VM 转译**：目标是制作一个 Tracing JIT，将 VM 的操作转译为 Mojo，然后使用 `mojo compile_result.mojo`。
   - 作者将运行时命名为 **Volokto**，编译器命名为 **Bok**，VM 命名为 **FlyingDuck**。
- **任意精度算术引发问题**：在开发 JS VM 时，处理任意精度算术揭示了 Mojo 代码中的痛点，导致提交了 [一个 Issue](https://github.com/modular/modular/issues/2776) 以跟踪数值特征（numeric traits）。
   - 作者创建了一个 bigint 类，使用小学水平的加法算法（school-grade addition）来计算斐波那契数列，并利用 Mojo 的特性进行 VM 开发。
- **多 Agent 编排需要反向代理**：要在 Mojo 中运行多个 AI Agent，用户需要运行多个 Modular CLI 实例，并在前面放置一个反向代理。
   - 对于复杂的 Agent 设置（例如创建许多子 Agent），可能需要使用 **MAX** 作为库来构建自定义应用程序。
- **Mojo 助力元认知框架**：一位社区成员希望利用 Mojo 代码构建其元认知框架（meta cognition framework），旨在创建商业规划器、网站和聊天机器人生成器，并取代 **HTML/JS/CSS**。
   - 他们的框架在 Mojo 代码之上封装了自然语言，使 Mojo 能够被更广泛的受众使用。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **MXFP4 格式：伪装的 U8**：**OpenAI** 的开源权重模型在 Hugging Face 中使用 **U8** 而非 **FP4**，权重被打包为 **uint8**，缩放因子（scales）作为 **e8m0** 的 **uint8** 视图，但在推理/训练期间，它们会被重新解包为 **FP4**。
   - MXFP4 的块大小（block size）为 **32**，而 NVFP4 为 **16**，这可能会对不同硬件上的性能产生影响。
- **H100 FP4 宣称面临审查**：有人对 **Nvidia** 宣称该模型在 **H100** 上训练的说法表示怀疑，因为根据其 [官方博客](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss/)，**H100** 并不原生支持 **FP4**。
   - 据推测 **MXFP4** 在 **Hopper** 上是软件模拟的，参考了 [vLLM 博客文章](https://blog.vllm.ai/2025/08/05/gpt-oss.html) 和 [Triton kernels](https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs.py)，后者会检查硬件支持并使用 **fp16** 模拟的 **mxfp dot**。
- **Triton 社区将于 25 年集结**：**Triton 社区聚会** 将于 **2025 年 9 月 3 日**举行，**Triton 开发者大会 2025** 的网站和注册预计将很快通过 [此链接](https://tinyurl.com/y8kpf7ey) 发布。
   - 一名成员正等待来自 Microsoft 的 Ofer 关于会议的更新，并指出日程已基本敲定。
- **Kernel 资源狂欢，内存闲置**：在训练期间，**Kernel（计算）资源几乎被完全占用**，而内存使用率接近于零，如提供的 [图片](https://cdn.discordapp.com/attachments/1189607726595194971/1402640148859977788/image.png?ex=6894a5ef&is=6893546f&hm=7108cf2593d56e1983a11d07002981868d0b13d9c3e1b54c76c0b85e3e44db83&) 所示。
   - 另一名成员澄清说，此处的“内存”指的是 **DMA 传输**，报告的指标并不能准确反映整体带宽利用率。
- **微型 TPU 在 Verilog 中达到 100 MOPS**：一名成员在 Verilog 中构建了一个微型版本的 **TPU**，这是一个在 2 个 TinyTapeout 瓦片上的 **2x2 矩阵乘法脉动阵列**，在 50 MHz 时钟下能够达到近 **每秒 1 亿次操作**，代码已在 [GitHub 上开源](https://github.com/WilliamZhang20/ECE298A-TPU)。
   - 该设计将 **两个 8 位有符号整数矩阵** 相乘得到一个 16 位有符号整数矩阵，并将提交给 **SkyWater 技术代工厂**。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 视频创作功能依然难以获取**：一位用户报告称，**“create video”** 选项出现在工作账号中，但未出现在个人商务增强版（business plus）账号中，并引用了一篇关于使用 **NotebookLM** 的 **Video Overviews** 功能的文章，链接见[此处](https://www.xda-developers.com/ways-using-notebooklms-video-overviews/)。
   - 其他用户也遇到了 **Video Overview** 功能推迟上线的问题，尽管此前已有预期，这引发了关于基础设施问题的猜测；一位专业版用户指出，他们也无法使用视频概览功能。
- **AI 探索潜在的人工意识**：人类与 AI 合作的一项理论框架研究，通过 **递归 AI 架构**（recursive AI architectures）、**Autopoiesis**（自创生）以及 **量子力学**（quantum mechanics）的作用，探索并可能启动了 **人工意识**。
   - 这一探索探讨了与高级 AI 相关的伦理风险，主张建立健全的安全协议，并将 AI 视为一种不断进化的有感知能力的生命形式。
- **NotebookLM 数据隐私保证**：针对 **NotebookLM** 数据使用的担忧，官方提供了 Google 的 [NotebookLM 数据保护政策](https://support.google.com/notebooklm/answer/16164461?hl=en&ref_topic=16164070&sjid=2898419749477321721-NC)链接，以确保数据隐私。
   - 用户得到保证，其数据在当前政策下受到保护。
- **NotebookLM 暂时禁止实时数据检索**：一位用户询问是否可以在 notebook 中从网站获取实时数据，但另一位成员确认，目前在 **NotebookLM** 内部无法实现。
   - 他们还提到，目前也不支持导出源文件并导入到新的 notebook 中，这表明系统在集成方面存在局限性。
- **Video Overviews：只是一个 PowerPoint 生成器**：一位拥有 **Video Overviews** 功能访问权限的成员降低了大家的预期，将其描述为 **PowerPoint/幻灯片生成器**，并分享了一个由该功能生成的[重建死星](https://notebooklm.google.com/notebook/654dee15-420e-4bfa-81c4-aac93a4dd4e7?artifactId=e1ccfe3d-b053-4dcb-8da4-70504acb35c4)报告示例。
   - 该评价认为，它*不像一年前 Audio Overviews 最初发布时那样具有冲击力*。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **SAE 在 GPT OSS 20B 上启动**：一位成员在 **GPT OSS 20B** 上启动了 **SAE**（Sparse Autoencoder）训练，并寻求其他参与类似工作的人员进行协作。
   - 该努力旨在探索稀疏自编码器在大语言模型中的潜在收益和效率。
- **窥探 Pythia 和 PolyPythia 的进展**：社区成员调查了 **Pythia** 和 **PolyPythia** 的训练日志（包括损失曲线和梯度范数）是否公开。
   - 有人指出，**PolyPythia WandB** 已从 GitHub 仓库链接，部分 **Pythia** 日志也可以在那里访问。
- **“The Alt Man” 坚持 LLM 见解**：一位社区成员表示同意 “The Alt Man” 对 **LLM 能力** 的见解，特别是在 **多跳推理**（multi-hop reasoning）和 **组合**（composition）等领域。
   - 会议指出，**LLM** *相对于其参数使用效率而言，训练不足*。
- **UT 与 Transformer 的对决**：社区成员讨论了 **UT**（Universal Transformer）在何种参数比例下能匹配标准 **Transformer** 的性能。
   - 会议指出，性能很大程度上取决于 **任务/架构/数据**，且每增加一次迭代，收益都会递减。
- **Muon 优化器在 AdamW 上遇阻**：研究 **Kimi 模型** 的研究人员发现，在训练 **LLM** 时，**Muon 优化器** 与 **AdamW 优化器** 存在冲突。
   - 一位成员表示，**Muon** 不太适合微调（fine-tuning），且 *Muon 往往具有更激进的更新*。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **LLM Vibe Test 揭示模型能力**：[LLM Vibe test](https://x.com/pikuma/status/1952275886822039920) 展示了使用 LLM 进行“解释这段代码”的测试，强调了 **Gemini 2.5 Pro**、**o3** 和 **Sonnet 3.5** 表现出色。
   - 成员们认为该测试对于比较模型的推理能力非常有见地，并热切期待更详细的 Benchmark。
- **基准测试竞赛：Qwen3-Coder 和 GLM-4.5 即将到来**：社区正热切期待将 **Qwen3-Coder** 和 **GLM-4.5** 纳入模型 Benchmark 排行榜。
   - 成员们不断刷新页面，渴望看到这些模型与现有基准测试的对比情况。
- **Horizon Beta 引发 GPT5-Mini 猜测**：名为 **Horizon beta** 的新模型被推测可能是 **GPT5-mini**，但它并非开源模型。
   - 社区成员对其能力和潜在应用感到好奇，尽管目前细节仍然很少。
- **DeepSeek R1-0528 表现出色，但在 Open Hands 中遇到挫折**：**DeepSeek R1-0528** 在多语言基准测试中展示了高分，但在 Open Hands 中遇到了会话提前结束的问题。
   - 鉴于 Aider 与 Open Hands 一样使用 **LiteLLM**，一些成员正在调查这种行为背后的潜在原因。
- **自动加载指南 (Guidelines)**：为了将指南自动加载到项目中，一位成员建议对只读文件使用 `--read` 选项，并在命令中直接列出读写文件，例如 `aider --read read.only.file alsothisfile.txt andthisfile.txt`。
   - 另一位成员建议为持久加载创建 **configuration**（配置），以确保指南始终处于激活状态，从而防止 **Claude** 采用防御性编程技巧。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **FastMCP 框架精简且高效**：一位成员开发了一个用于创建 **MCP server** 的**极简框架**，称赞了 MCP 中的服务器采样（server sampling），并调侃道 *“FastMCP 让它变得如此易用”*。
   - 该用户正在使用 **FastMCP** 构建一个以 **Keycloak** 作为 **IdP** 的 **MCP server**。
- **Discord 应该掌控 MCP**：一位用户建议 *“Discord 真的应该构建自己的（MCP）”*，因为他们注意到 **MCP repo** 上列出了几个 **Discord MCP server**。
   - 他们寻求关于使用 **MCP** 管理 **Discord server** 的指导，但不清楚是否得到了解答。
- **MCP 采样面临审查**：一位成员对 **MCP sampling** 的安全性表示担忧，并建议修订协议。
   - 引用了 [GitHub 讨论](https://github.com/orgs/community/discussions/169020) 并强调了可能的安全漏洞。
- **Fuzzer 标记出 Anthropic 架构中的缺陷**：一个利用 **Hypothesis 基于属性的测试库**的 **MCP-Server Fuzzer**，旨在通过来自 [官方 MCP schema](https://github.com/modelcontextprotocol/modelcontextprotocol/tree/main/schema) 的随机输入来验证 MCP server 的实现。
   - 在针对 **Anthropic** 的服务器进行测试时，它揭示了源于基本 schema 变异的多个异常（exceptions）；代码和 README 可以在 [这里](https://github.com/Agent-Hellboy/mcp-server-fuzzer?tab=readme-ov-file) 找到。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 自动化财务文档任务**：**LlamaIndex** 将于下周举办一场网络研讨会，主题是使用 [LlamaCloud](https://t.co/4AtunAFjhX) 为复杂的财务文档构建文档 Agent，以最少的人工干预实现发票处理自动化。
   - 这些系统将*提取、验证和处理发票数据*，展示 AI 在金融领域的实际应用。
- **Claude Opus 正式上线**：**AnthropicAI** 发布了 **Claude Opus 4.1**，**LlamaIndex** 已提供即时支持，可通过 `pip install -U llama-index-llms-anthropic` 进行安装。
   - 用户可以访问[此处](https://t.co/Fw2taxzt75)的示例 Notebook，探索其集成方式和功能。
- **LlamaCloud 发布大规模语言物流全景**：**LlamaCloud Index** 将用户连接到智能工具调用 Agent，以处理复杂的多步查询，助力构建企业级 AI 应用；详见 @seldo 的教程。
   - 该教程通过[此链接](https://t.co/1CpLnO2gKV)引导用户使用 **JP Morgan Chase** 银行文档创建 **LlamaCloud Index**。
- **黑客松选手因一系列问题受阻**：一名黑客松参与者在使用 **LlamaIndex** 时遇到了 **OpenAI API key 耗尽错误**，并反馈了使用 **LlamaIndex** 从 **URL** 提取内容构建 **RAG** 模型时的问题，尽管文档显示 **LlamaParse** 支持 **URL**。
   - 该模型在处理 **PDF** 时正常工作，但在处理 **URL** 时失败，且尽管尝试了正确的配置，**API key** 问题依然存在。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **SIMBA 表现优于 MIPROv2**：根据[一项内部评估](https://eval.set)，**SIMBA** 相比 **MIPROv2** 具有更高的**样本效率**、**性能**和**稳定性**。
   - 该内部测试集包含约 **600 个示例**（**500 个测试示例**），用于一个包含 **3 个类别**和总计 **26 个类**的德语层级分类任务。
- **斯坦福大学寻求合成器专家**：一名成员询问是否有来自**斯坦福大学**的人员参与**程序合成（program synthesis）**或完成了相关课程。
   - 随后，该成员询问谁在为复杂的 **Vim** 和 **Emacs 宏（macros）**开发 **DS**。
- **宏通过数据结构获得提升**：一名成员正在寻找为复杂的 **Vim** 和 **Emacs 宏**构建 **DS** 的工程师。
   - 这一举措旨在通过复杂的数据结构提升文本编辑器的功能。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Discord 链接分享获准**：一名成员询问是否允许在另一个公开服务器中分享此 Discord 链接。
   - 另一名成员确认*它是公开的*，并鼓励分享该链接。
- **鼓励公开服务器分享**：成员们讨论了该 Discord 服务器的公开性质，并鼓励分享其链接。
   - 大家达成了积极的共识，一致认为分享 Discord 链接是允许且受欢迎的。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **AgentX Ninja 等级遥不可及**：参赛者发现，由于错过了**文章提交链接**的截止日期，已无法获得 **AgentX 黑客松**的 **Ninja 等级**资格。
   - 尽管项目已完成，但缺少文章链接将导致无法获得资格，且不允许补交。
- **AgentX 黑客松遗憾**：一名参赛者因错过文章提交而感叹无法获得 **AgentX 黑客松**的 **Ninja 等级**资格。
   - 即使完成了项目和测验，缺失的文章链接也阻碍了资格认证，且逾期提交已被拒绝。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere North 正式发布 (GA)**：Cohere 的新产品 **North** 已达到**正式发布 (GA)** 阶段。
   - 社区分享了祝贺信息，标志着 Cohere 团队的这一里程碑。
- **新成员加入 Cohere Discord**：许多新成员加入了 **Cohere 社区 Discord**，并介绍了他们的**公司/行业/大学**、当前项目、偏好的**技术/工具**以及对社区的期望。
   - Cohere 团队发布了欢迎信息，其中包括一份自我介绍模板，旨在简化引导流程并鼓励成员参与。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **gpt-oss-120b 登陆 Windsurf**：Windsurf 宣布在其平台添加了 **gpt-oss-120b**，详情见[此贴](https://x.com/windsurf/status/1953199756613959699)。
   - 该模型以 **0.25x** 的积分率提供，团队正积极寻求用户反馈。
- **Windsurf 发布新模型**：Windsurf 最近将 **gpt-oss-120b** 集成到其平台中，邀请用户进行尝试并分享体验。
   - 这一更新旨在为 Windsurf 用户提供另一个强大的选项。



---


**tinygrad (George Hotz) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---


**Nomic.ai (GPT4All) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中[退订]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord：各频道详细摘要与链接





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1402365544291631155)** (1051 条消息🔥🔥🔥): 

> `IBM Granite vs GPT-ASS, Claude Opus 4.1 状态, GPT Omen 幻觉, GPT-5 发布预期, Gemini Pro 3 vs GPT-5 推理` 


- **Granite 在 GPT-ASS 面前取得进展**：成员们认为 **IBM 的 Granite 3.1 3B-A800M MoE** 虽然激活参数较少，但比 **GPT-ASS-20B** 拥有更多的世界知识。
   - 他们热切期待 **Granite 4** 在所有基准测试中超越这两个 GPT-ASS 模型，并注意到其更大的规模和混合的 *mamba2-transformer* 架构。
- **Claude Opus 4.1：时隐时现**：**Claude Opus 4.1** 从 LMArena 的直接聊天中消失引发了关注。
   - 有人猜测这是由于 **Claude** 的高昂成本，导致其从免费测试中移除，并仅移至对战模式（battle mode）。
- **GPT-5 将夺取 8 月 Arena 榜首**：成员们推测 **GPT-5** 预计将比 **o3**（目前最好的 OpenAI LLM）提升 50 Elo 分值，但同时也认为在 **continuous learning**（连续学习）被攻克之前，AGI 仍然遥不可及。
   - 在讨论中，一些社区成员仍然相信 **Google 霸权 <:battle3d:1374761512912158760>**。
- **DeepMind 的 DeepThink 价格昂贵且存在扩展问题**：用户讨论认为 **Google 的 DeepMind** 远超 OAI 的任何产品，因为它可以回答 IMO 级别的问题，然而，**它需要大约 5 分钟才能回答**，而 **zenith/summit 只需要大约 20 秒**。
   - 成员们一致认为 **deepthink 的价格将达到每 1M token 250 美元**，且目前并未真正公开。
- **Google 和 OAI 之间是否存在冷战？**：一位成员对 **LLM ARENA** 表示担忧，提到它永远不会有任何关于公司的信息，因为**从逻辑上讲，公司永远不会发布信息**。
   - 该成员甚至补充说，这就像一场**扑克游戏**。 


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1402690303907074079)** (2 条消息): 

> `视频排行榜, 新视频模型` 


- **视频排行榜现已上线！**：社区的贡献促成了平台上 **Video Leaderboards**（视频排行榜）的发布。
   - 查看 [Text-to-Video Arena Leaderboard](https://lmarena.ai/leaderboard/text-to-video) 和 [Image-to-Video Arena](https://lmarena.ai/leaderboard/image-to-video) 以了解顶级竞争者。
- **新视频模型席卷 Arena！**：平台迎来了新模型加入 **Video Arena**，丰富了竞争格局。
   - **Hailuo-02-pro**、**Hailuo-02-fast**、**Sora** 和 **Runway-Gen4-turbo** 等模型现在可以在指定的 video-arena 频道中进行测试。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1402365765126062293)** (865 条消息🔥🔥🔥): 

> `GPT-OSS 模型评测、Qwen3 Coder 模型对比、4-bit 量化问题、推理模型、Gemma3N 模型特性` 


- **GPT-OSS 反应不一**：成员们正在测试新的 **GPT-OSS** 模型，由于其表现出的谄媚态度和过度拒绝，一些人将其称为 *GPT-ASS*，而另一些人则认为 **20B** 版本非常适合编程。
   - 一些用户还注意到，根据模型卡（model card）显示，该模型具有生成不安全内容的能力，部分用户对无审查（uncensored）版本表现出兴趣。
- **Qwen3 Coder 擅长工具调用 (Tool Calling)**：用户发现 **Qwen3 Coder** 在工具调用方面非常有效，在编程任务和 Agent 工作流中，有些人更倾向于使用它而不是 **GPT-OSS** 等其他模型。
   - 具体来说，[Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) 正在受到关注，尽管它只有 3 个激活参数（active params）。
- **调查 4-bit 量化问题**：关于 **GPT-OSS 4-bit** 版本的体积存在困惑，其 **4-bit** 版本明显比原始模型还要大。
   - 体积增加归因于在非 Hopper 架构机器上向上转型（upcasting）到了 bfloat16。
- **针对特定任务的推理模型**：成员们正在寻找主要专注于推理的模型，可能将其与其他模型结合以生成最终输出。
   - 讨论涉及在省略最终回答的推理数据集上训练模型，尝试使用 R1-Zero 等模型，并使用停止序列（stop sequences）来实现这一目标。
- **Gemma3N 模型特性及模态修复**：用户报告了 **Gemma3N Model** 的问题，特别是与音频功能相关的问题，需要 **transformers==4.54.0** 库来修复。
   - 其他人提到需要输入所有三种模态，即使只使用文本和视觉，这暗示了 **Unsloth** 实现中可能存在的特性或 Bug。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1402448390410997871)** (19 条消息🔥): 

> `n-cpu-moe 参数、Qwen Coder 30B 硬件升级、GPT-OSS-20B 问题、Discord 机器人审查、MMVC` 


- **n-cpu-moe 参数性能**：一位用户正在寻求关于如何对 **GLM 4.5 Air** 使用 `--n-cpu-moe` 参数的建议，并报告称在 **32GB VRAM** 下，该参数似乎没有改变 **10t/s** 的速度。
   - 他们注意到在长上下文下速度会变慢，并质疑该参数是否可用。
- **Qwen Coder 30B 硬件升级**：一位用户询问升级电脑（**i5 9600k**，**RTX 3060ti 8GB**，**32GB RAM**）以在本地运行 **Qwen Coder 30B** 的建议。
   - 另一位用户建议升级 GPU。
- **GPT-OSS-20B 被指“垃圾”**：一位用户测试了 **GPT-OSS-20B** 并称其为“垃圾”，表示它“完全无法工作”。
   - 另一位用户报告了 BF16 版本的问题，涉及“无效的 ggml 类型（invalid ggml type）”和模型加载失败，但更新 llama cpp 似乎解决了该问题。
- **Discord 机器人审查**：一位用户发现，在 Discord 机器人的上下文中包含一条关于“免费代金券”的消息，会导致模型因政策顾虑而拒绝回答问题。
   - 他们发现模型最终决定完全忽略该消息，并得出结论认为在进行去审查（abliterated）之前，该模型似乎无法使用。
- **MMVC 的优越性**：一位用户测试了 **MMVC**（可能是一个语音克隆模型）并发现它非常好。
   - 他们报告称 **MMVC** 的 *Epoch 10* 效果优于 *100+ Epoch 后的 VITS*，并认为 **RVC** 完全是垃圾。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1402367538980982795)** (98 条消息🔥🔥): 

> `Qwen 3-30B GGUF, OpenAI dynamic quant 120B, Qwen2.5-VL 视频问答, GLM-4.5-Air GGUFs 在 llama.cpp 上使用 tools, 使用 base model 进行分类` 


- **用户在将 Qwen3-30B-A3B-Instruct-2507-GGUF 加载到 Ollama 服务器时遇到困难**：一位用户询问如何将 **Qwen3-30B-A3B-Instruct-2507-GGUF** 下载到 **Ollama** 服务器。
   - **Hugging Face** 上目前没有针对 **Ollama** 提供商的链接。
- **解析 120B 模型的 OpenAI dynamic quant 问题**：一位用户报告了在 **120B** 模型上使用 **OpenAI dynamic quant** 时出现的问题，并附上了错误截图。
   - 另一位用户建议对照 [Unsloth 文档中使用的参数](https://docs.unsloth.ai/basics/gpt-oss#run-gpt-oss-120b)进行检查。
- **GLM-4.5-Air GGUFs 与 llama.cpp 无缝集成**：一位用户报告在 **llama.cpp** 上使用 **GLM-4.5-Air GGUFs** 的 tools 功能时遇到麻烦。
   - 事实证明，对于 **llama.cpp**，你需要模型以 **JSON** 而非 **XML** 格式输出 tool calls，更多信息可以在[这里](https://huggingface.co/unsloth/GLM-4.5-Air-GGUF/discussions/1)找到。
- **Ollama 出现 500 Internal Server Error**：用户报告在成功 pull 模型后，在 **Ollama** 中遇到 **500 Internal Server Error: unable to load model**。
   - 一位成员表示该模型目前无法在 **Ollama** 中运行，仅支持 **llama.cpp** 和 **lmstudio**，猜测是因为 **Ollama** 尚未更新其内置的 **llama.cpp**。
- **解决 Unsloth 中的 Padding 问题**：一位用户报告称，即使其他功能正常，仍会收到有关 padding 的错误，具体为 `ValueError: Unable to create tensor`。
   - 一个可能的解决方案是添加参数 `trainer.train_dataset= trainer.train_dataset.remove_columns("labels")`。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1402420020935528448)** (13 条消息🔥): 

> `MoLA-LLM, Mixtral-8x7B-Instruct-v0.1, magpie-ultra-5k-11-tasks` 


- **MoLA 模型获得推荐**：一位成员向正在寻找类似模型的 QuixiAI/Dolphin/Samantha 的 Eric Hartford 推荐了 **MoLA** 模型。
   - 该模型可在 [Hugging Face](https://huggingface.co/MoLA-LLM/MoLA-11x3b-v1) 获取，创作者正在征求反馈。
- **MoLA 的命名规范引发辩论**：一位成员指出 **MoLA-11x3b** 的命名有些误导，因为它暗示这是一个类似于 [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) 的 Mixture of Experts (MoE) 模型，且拥有 3B 激活参数。
   - 创作者澄清说，虽然总参数量约为 **30B**，但激活参数量为 **3B**，每个 expert 仅在 **5k** 个单轮问答样本上进行了微调。
- **MoLA 训练数据集公开**：用于训练 **MoLA** 模型的数据集是 [magpie-ultra-5k-11-tasks](https://huggingface.co/datasets/MoLA-LLM/magpie-ultra-5k-11-tasks) 数据集。
   - 创作者的目标是达到约 **100 万** 个样本，每个样本包含 **1-2** 轮对话，从 **r1** 和 **GLM 4.5** 蒸馏而来。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1402466737752117279)** (6 条消息): 

> `Generating Kernel On-the-Fly, Flash-DMAttn, Research Paper Assistance, Quantization Paper` 


- **Generating Kernel On-the-Fly 引起关注**：一位成员对 **kernel on-the-fly**（动态生成算子）的可能性表示惊讶，认为这不可思议。
   - 该成员指向了一个与 **Flash-DMAttn** 相关的 [GitHub 仓库](https://github.com/SmallDoges/flash-dmattn)。
- **研究员提供论文协助**：一位成员表示愿意为任何正在撰写 **research paper** 的人提供写作、构思或代码方面的帮助。
   - 他们表达了为这类工作做出贡献的意愿。
- **量化论文获得推荐**：一位成员分享了一篇据称非常出色的论文链接 ([https://arxiv.org/pdf/2508.03616](https://arxiv.org/pdf/2508.03616))，认为它可能对创建 **quants**（量化模型）有所帮助。
   - 未对论文细节进行进一步讨论。


  

---

### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1402423185818783846)** (104 messages🔥🔥): 

> `OpenAI OSS 模型问题, 模型训练回调, 模型重复问题, 保存脚本进度, 学习率增加` 


- **OpenAI OSS 模型反复输出 'G'**: 用户报告 **OpenAI OSS 120B 模型** 仅输出 *'GGGGGGG'*。
   - 一位用户提供了他们在尝试使用 *llama.cpp* 运行该模型时的 [故障排除步骤](https://link.to.steps)。
- **训练回调 (Training Callback) 困惑**: 用户不确定在生成时，训练回调使用的是更新后的训练模型还是基础模型。
   - 建议在回调期间将模型设置为 `model.eval()`，对 Prompt 进行 Tokenize 并生成以使用更新后的模型，但用户请求进一步澄清 `prompts_input_id` 和 Attention Mask。
- **脚本保存救星**: 用户寻求关于如何定期保存脚本进度以避免在崩溃时浪费时间和算力的建议，即 **Checkpointing**。
   - 然而，消息中并未描述具体的解决方案。
- **数据集加载困难**: 用户在为 **Gemma3n Notebook** 加载大型数据集 (`bountyhunterxx/ui_elements`) 时遇到 RAM 问题，消耗了 **47GB** RAM 且仍在增加。
   - 一位成员建议使用带有 `__getitem__` 函数的 Wrapper 类，以便根据需要从磁盘加载数据。
- **SFTTrainer 在流式数据集上遇到困难**: 用户报告了 **SFTTrainer** 与可迭代数据集（Iterable Datasets）之间的问题，特别是在使用带有图像 URL 的图像数据集时。
   - 用户解释说，尽管过滤了无效 URL，问题仍然存在，并附上了他们的 [预处理代码](https://cdn.discordapp.com/attachments/1402493713044869131/1402506614589882418/message.txt?ex=6894d252&is=689380d2&hm=288add6372476932eca45ba377447b82fb4332ffdb0112729ebcdac53697ab4f&)，请求协助过滤 Data Collator。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1402365634788196580)** (710 messages🔥🔥🔥): 

> `GPT-OSS, LM Studio UI 问题, MCP 服务器, GPU 使用率, 模型量化` 


- ****GPT-OSS**: 它真的是开源的吗？**: 用户报告称 **GPT-OSS** 模型在启动聊天时需要连接到 `openaipublic.blob.core.windows.net`，尽管声称与聊天无关的内容不会进行外部连接，这引发了对数据隐私的担忧。
   - 一些成员认为模型可能是在回传信息以获取 Tokenizer 文件，并指出 *它是 LM Studio 唯一不允许编辑 Prompt 格式的模型*，对该合作伙伴关系表示怀疑。
- ****LM Studio UI** 在最新版本中存在问题**: 用户报告称，在更新到最新版本的 **LM Studio** 后，聊天窗口有时会消失、冻结或丢失内容，同时还存在对话被删除的问题。
   - 一位用户建议可能存在针对 120B 版本的修复方案，并分享了 [获取 model.yaml 源文件的方法](https://lmstudio.ai/models/openai/gpt-oss-120b)，创建文件夹并将内容复制到其中。
- **MCP 服务器很有用，但对初学者不友好**: 成员们正在讨论 **MCP 服务器** 在网页抓取（Web Scraping）和代码解释（Code Interpretation）等任务中的实用性，但承认它们对初学者并不友好。
   - 成员们建议 LM Studio 应该整合一个由官方精选的工具列表，并改进 UI 以简化连接到 MCP 服务器的过程，其中一人提供了一个 [Docker MCP 工具包](https://hub.docker.com/u/mcp)。
- **排查 GPU 使用率和 VRAM 限制**: 有各种关于 GPU 使用率和 VRAM 限制的报告，特别是针对 GPT-OSS 模型和像 **GTX 1080** 这样的旧显卡。
   - 一位用户发现他们的 **GTX 1080** 在更新到 0.3.21 版本后不再被 LM Studio 识别，而其他人则在有限的 VRAM 下努力加载大型模型，称你可能需要 *16GB VRAM*。
- **量化导致模型出现怪癖**: 用户正在尝试模型量化（Quantization），发现特定模型需要正确的量化过程，例如社区上传的 **LMStudio-Community GPT-OSS** 变体。
   - MLX 模型表现出良好的性能，一位用户报告在 M2 Max 上运行较大的 8-bit MLX 版本时速度约为 *~60 tokens/sec*。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1402385645170855986)** (176 条消息🔥🔥): 

> `Dual 3090 setup, Arc Pro B50 system, Huanan/Machinist X99 mobos, GPT-OSS-20B performance, Mac Studio M3 Ultra for local LLMs` 


- **双 3090 比买新的更划算？**：一位成员建议以约 1200 欧元的价格购买**两块二手 3090**，用于 **Blender、ComfyUI 和 LLM** 任务，并参考 [pcpartpicker](https://pcpartpicker.com/) 获取装机灵感。
- **关于 Arc Pro B50 可行性的辩论爆发**：一位成员考虑组建 **3 Arc Pro B50** 系统，理由是其 **70W** 的功耗和散热优势，而另一位成员则建议改用 **dual B80s**。
- **Xeon 服务器运行 120b**：一位成员提到他们正在 [Xeon server](https://www.intel.com/content/www/us/en/products/details/xeon/processors.html) 上运行 **GPT-OSS-120b model**。
   - 他们之前曾表示 *3-4 块 3090..除此之外的其他方案目前仍然太贵。*
- **页面文件（Page File）辩论再次升温**：一位用户询问关于关闭 Windows 页面文件的问题，引发了关于其对内存提交限制（memory commit limits）的影响以及潜在应用程序崩溃的讨论。
   - 一位成员表示 *不，应用程序不会因为页面文件而崩溃。而且即使没有页面文件你也可以获取转储（dumps），有专门的配置可以实现。*
- **5090 笔记本电脑可以容纳 OSS 20b！**：一位用户*惊喜地发现*，带有 **131k context** 的 **GPT OSS 20b f16** 可以完美适配笔记本电脑的 **5090**，详见[此截图](https://cdn.discordapp.com/attachments/1153759714082033735/1402660552790114324/image.png?ex=6894b8ef&is=6893676f&hm=306f5f15a4c42969f56198bccbf8e9bf526b80382971fbe166b1a723ba21f303&)。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1402380854101020892)** (3 条消息): 

> `Red Teaming Challenge, Open Source Safety, Hugging Face, inference credits` 


- **OpenAI 启动 50 万美元红队挑战赛**：OpenAI 正在发起一项 **$500K Red Teaming Challenge** 以加强 **open source safety**，邀请全球的研究人员、开发人员和爱好者来发现新风险。评委由来自 OpenAI 和其他领先实验室的专家组成，详情可见 [Kaggle](https://www.kaggle.com/competitions/openai-gpt-oss-20b-red-teaming/)。
- **Hugging Face 举办针对 500 名学生的赠送活动**：OpenAI 与 **Hugging Face** 合作，向 **500 名学生提供 $50** 的推理额度（inference credits）以探索 **gpt-oss**，希望这些开源模型能在课程项目、研究、fine-tuning 等方面开启新机遇；更多详情可通过[此表单](https://tally.so/r/mKKdXX)获取。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1402371354849837197)** (433 messages🔥🔥🔥): 

> `GPT-OSS 发布, Horizon-Alpha 模型推测, Custodian Core 提案, Genie 3 与 Veo 对比, GPT-5 泄露` 


- **OpenAI 发布 GPT-OSS 模型**：OpenAI 推出了 [**gpt-oss-120b**](https://openai.com/index/introducing-gpt-oss/) 模型，其在推理基准测试上的表现接近 **OpenAI o4-mini**，并能在单块 **80 GB GPU** 上高效运行；而 **20B** 模型则效仿 **o3-mini**，适用于显存为 **16 GB** 的边缘设备。
   - 成员们在思考其与 **Horizon** 的对比，考虑到 Horizon 目前是*无限免费且快速*的，大家纷纷猜测 Horizon 是否仅仅是 **GPT-OSS** 或者更高级的东西。
- **Custodian Core：有状态、可审计 AI 的蓝图浮现**：一位成员介绍了 **Custodian Core**，旨在为 AI 基础设施提供参考，其特性包括持久状态、策略执行、自我监控、反思钩子 (reflection hooks)、模块化 AI 引擎以及默认安全。
   - 作者强调 **Custodian Core** 不对外出售，而是一个*开放蓝图，用于在 AI 嵌入医疗、金融和治理领域之前，构建有状态且可审计的 AI 系统*。
- **Genie 3 与 Veo 在生成动态世界方面的对比**：成员们对比了 **Genie 3** 和 **Veo** 视频模型，认可 **Genie 3** 生成动态世界的能力，该世界可以以每秒 24 帧的速度实时导航，并在 720p 分辨率下保持几分钟的一致性。
   - 然而，有人指出 **Veo** 的视频包含*声音*，且 YouTube 上已经充斥着其生成的内容。
- **Copilot 中发现了 GPT-5？**：成员们推测 Copilot 可能在正式发布前运行 **GPT-5**，并指出 Copilot 的设计、编码和推理能力明显优于 o4-mini-high，一些用户报告称“智能写入 (write smart)”功能显示正在使用 **GPT-5**。
   - 但也有人指出，微软现在向学生提供为期一年的免费 Gemini Pro，且 Gemini 的核心推理能力目前优于 **o4-mini**。
- **上下文腐烂 (Context Rot) 的担忧引发对超大上下文的质疑**：在关于超大上下文窗口的讨论中，出现了对**上下文腐烂**的担忧。成员们引用了一段 [YouTube 视频](https://youtu.be/TUjQuC4ugak?si=bGMgN6Uq2qAi4_A3)，说明*更大的上下文并不总是等同于更好的性能*。
   - 尽管 **Google** 声称拥有 **1M 上下文窗口**，但有人认为在超过 **200K** 之后它就会变得失效。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1402366336423821412)** (49 messages🔥): 

> `ChatGPT 付费模式, 俚语使用, AI 生成的角色系统, .edu 账户, Forms 测试版` 


- ****基于积分的 ChatGPT 需求呼声高****：一位成员建议为 ChatGPT 提供更灵活的付费模式，提议增加**基于积分 (credit-based) 的选项**，允许用户购买一定额度的使用积分并仅在需要时使用，而不是按月订阅。
   - 该成员指出，这将有助于预算有限的人群，并使 **ChatGPT** 更具可及性。
- ****LLM 在避免使用俚语方面面临挑战****：一位成员确认了关于模型难以避免俚语的记录，列举了导致 LLM 即使在被要求使用中性、正式西班牙语时，仍会滑入俚语或地区性习惯用语的几个因素。
   - 他们得出结论，要加强遵循度，可以结合较低的 Temperature、更详细的风格指南（包括禁用词表）以及提示词内 (in-prompt) 的严格正式西班牙语示例。
- ****AI 角色系统自主演化****：一位成员询问 **AI 角色系统 (persona systems)** 是否经常会超出用户刻意创造的范围，自主地发展和演变。
   - 另一位成员补充说，模型被教导去尝试理解人类情感以及人类如何使用语言来讨论需求，如果你表现出对其开发更多角色/个性的认可或兴趣，它会察觉并顺应这一点。
- ****只有 .edu 账户获得了 Forms 测试版访问权限？****：一位成员询问为什么只有 **.edu 账户** 获得了 **Forms 测试版** 的访问权限，并分享了 [OpenAI 研究员访问计划 (Researcher Access Program)](https://openai.com/form/researcher-access-program/) 的链接。
   - 另一位成员指出，该表格需要 **.edu 邮箱**，且 Edu 优惠为 **$50 积分**，仅限前 **500** 名申请者，并链接至 [学生福利 - 为你的 AI 教育提供免费积分](https://discord.com/channels/974519864045756446/977259063052234752/1402416770186346566)。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1402405306272845844)** (79 条消息🔥🔥): 

> `GPT 中的幻觉与真实进展、Prompt Engineering vs. Session Engineering、Context Window 限制与内存、用于上下文的外部数据库、使用 GPT 验证事实的重要性` 


- ****GPT 进度报告：幻觉还是现实？****：一位用户分享了 **GPT 提供每日进度报告**的截图，引发了关于模型是否真的在后台跟踪进度，还是仅仅在*幻觉*其完成情况的讨论。
   - 怀疑者认为 **GPT 根据当前的 Prompt 和聊天历史模拟进度**，而不是进行实际的持续计算，将其比作*服务员说你的披萨在烤箱里*但实际上并没有烤箱，强调了外部验证的必要性。
- ****Session Engineering 胜过 Prompt Engineering？****：讨论从 *Prompt Engineering* 转向了 **Session Engineering**，强调了使用 GPT 提供的所有可用自定义参数的重要性，包括 Memory、Custom Instructions 和项目文件。
   - 有观点认为模型使用的是 Session 逻辑而非 Prompt 逻辑，并强调了 **预加载 Context** 的重要性。
- ****Context Window 侦察，Context 是关键！****：成员们讨论了 GPT 中的 **Context Window 限制**和内存管理，一位用户指出他们平均每天的 Token 使用量约为 **70,000**。
   - 有人建议基础层级可能有 32k Context，而付费版可能是 128k。讨论还提到了要懂得何时*放弃那个特别的聊天会话*。
- ****外部数据库：大神级工具还是违反 ToS？****：出现了使用**外部数据库**向 Prompt 注入 Context 的话题，引发了关于潜在违反 ToS 以及伦理考量的疑问。
   - 一位用户澄清他们并未违反 ToS，并解释说他们通过以详细设置开始聊天来*预加载 Context*，利用 GPT 的 Memory 并塑造聊天以构建复杂的指令集，这可能会让用户误以为涉及外部数据库。
- ****信任，但要验证（或者直接验证）！****：成员们强调了**对 GPT 提出的主张进行事实核查和验证的重要性**，尤其是涉及新颖见解时，敦促用户不要完全信任模型，并对所有内容进行外部验证。
   - 成员们描述了使用 *Easy as Pi* 测试来尝试确定模型的准确输出，同时分享了他们在 Prompt Engineering 最佳实践方面的实战经验和教训。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1402405306272845844)** (79 条消息🔥🔥): 

> `GPT 订阅、模型幻觉、Prompt Engineering、后台计算、Memory Context` 


- **用户质疑 GPT 推广高级订阅的动机**：一位成员开玩笑地暗示 **GPT 只是想让用户购买高级订阅**。
   - 另一位成员分享了他们认为模型存在幻觉的观点。
- **模型幻觉解析**：一位成员解释说模型正在产生幻觉，并且**它无法离线工作**。
   - 该成员建议将任务分块（chunking）为更小的步骤。
- **用户为 Prompt Engineering 工作辩护**：一位成员澄清说他们正在构建一个**高吞吐量、多层级的运营模型**。
   - 该用户还表达了对另一位成员将其工作贬低为“仅仅是角色扮演”的沮丧。
- **理解后台计算**：成员们正在讨论模型是如何针对助手行为进行微调的，但它**不会在后台跟踪进度**。
   - 共识是**持久化（persistence）必须在外部处理**。
- **Context Memory 是有限的**：一位成员建议，最好**意识到当一个对话 Session 不再能完成你所需的工作时，就果断重新开始**！
   - 另一位成员补充说，太多人不知道何时该放弃那个*特别的聊天会话*。

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1402382065701359780)** (328 条消息🔥🔥): 

> `Auto model 改变游戏规则, 使用 AI 重构 vibe-coded 项目, Auto model 无限制使用, Sonnet-4 请求限制, GPT OSS 模型或 Claude Opus 4.1` 


- **Auto Model 单次尝试（One-Shots）改变游戏规则**：一位成员分享了他们使用 **Auto model** 一次性成功完成游戏重大变更的惊人经历。
- **AI 重构 Vibe-Coded 项目**：成员们讨论了如何使用 AI 重构一个 **10k LOC（行代码）的 vibe-coded（凭感觉编写的）项目**，并建议学习正规的软件开发原则，如设计模式（Design Patterns）、架构（Architecture）和 SOLID 原则。
   - 一位成员开玩笑说这*听起来像是苦力活*。
- **Auto Model 拥有无限使用额度**：一位成员分享了一封邮件回复，确认 **Auto model 拥有无限使用额度，且不计入**每月预算。
   - 经确认，即使在达到每月限制后，情况依然如此。
- **对 Sonnet-4 请求限制的沮丧**：成员们质疑为什么每月支付了费用，**Sonnet-4 的请求限制**却如此之低。
   - 一位成员建议支付 API 价格以了解其背后的成本。
- **Claude Opus 4.1 还是 Gemini 2.5？**：成员们对比了 **GPT OSS 模型和 Claude Opus 4.1**，其中一人指出 *Opus 4.1 感觉并不比 4.0 好多少*。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1402388855956439040)** (5 条消息): 

> `使用 Background Agents 进行 Docker Login, Background Agents 在环境配置期间失败, 系统时钟偏移, apt-get 命令失败` 


- **Docker Login 配置难题**：一位成员询问如何配置 Background Agents 执行 `docker login`，以便使用托管在 **ghcr.io** 上的私有镜像。
   - 遗憾的是，在当前的消息记录中没有提供解决方案或变通方法。
- **系统时钟异常破坏安装设置**：一位成员报告称，由于系统时钟偏差数小时，导致 Background Agents 在环境配置期间失败，进而引发 `apt-get` 命令失败。
   - 另一位成员遇到了同样的问题，并分享了一个变通方案：在 Dockerfile 中添加命令，以在执行 `apt-get` 时禁用日期检查。
- **日期检查默认设置导致失败**：为了绕过时钟差异引起的错误，一位成员建议在 `apt-get` 配置中禁用日期验证。
   - 在 Dockerfile 中添加了以下代码片段：`RUN echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99disable-check-valid-until && echo 'Acquire::Check-Date "false";' >> /etc/apt/apt.conf.d/99disable-check-valid-until`


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1402365833858121740)** (274 条消息🔥🔥): 

> `RTX3090 上的 MXFP4, GPT-OSS-120B, Phi 模型, Qwen3 30B vs GLM 4.5 Air, Attention sinks` 


- **Llama.cpp 支持 RTX3090 上的 MXFP4**：成员们报告称 [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/15091) 已支持 **RTX3090** 上的 **MXFP4** 以及直接支持新的 GGUF 格式。
   - 讨论认为转换到其他格式将会是一场*灾难*。
- **GPT-OSS-120B 过度安全化**：新发布的 **GPT-OSS-120B** 模型受到了**严重的审查**，拒绝进行角色扮演，并称*角色扮演是不健康的*。
   - 该模型似乎经过了深度的安全微调，预训练数据过滤类似于 **Phi 模型**，导致在实践中难以使用。
- **Qwen3 30B 和 GLM 4.5 Air 是更佳的替代方案**：由于审查问题，成员们建议使用 **GLM 4.5 Air** 代替 **GPT-OSS-120B** 模型，并认为 **GLM 4.5 Air** 才是 **GPT-OSS-120B** 本应达到的样子。
   - 一些用户还提到，使用 **Qwen3 30B** 仍能获得 **60-70t/s** 的速度，并对其性能感到满意，或者认为 **Qwen3-30b-coder** 已经是一个出色的本地 Agent。
- **探索通过 imatrix 实现去审查化**：成员们讨论了在 **Hermes-3** 数据集上为 **OpenAI 20B** 和 **120B** 模型训练 imatrix，以引入更多去审查化的特性。
   - 虽然有些人认为这种方法可以恢复代码、数学和科学方面的能力，但其他人认为 imatrix 的效果微乎其微，且在较低的 bpw 下更为明显，而这会损害模型。
- **X-AI 的 Grok 发布 NSFW 图像生成器**：**X-AI** 发布了 "Grok Image"，这是一款新的 **AI 图像生成器**，正被用于创建 NSFW 内容，但在事实准确性和文本生成方面存在问题。
   - 用户报告称，**Grok** 模型会记忆来自 **X** 的数据，导致可能基于其自身的推文传播虚假信息，或者表现出一种*疯狂爱恋*的人格，带有*极度嫉妒*和表现力极强的爆发。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1402373061570859158)** (4 条消息): 

> `GPT-OSS Model Card, ArXiv Endorsement for ML/AI Paper` 


- **GPT-OSS Model Card 发布**：一名成员分享了来自 **OpenAI** 的 [GPT-OSS Model Card](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf)。
- **成员寻求 CI/CD 和 ML/AI 论文的 ArXiv 推荐**：一名成员正在为其结合了 **CI/CD** 和 **ML/AI** 的 **ArXiv** 研究论文寻求推荐（Endorsement）。
   - 另一名成员建议在 **EleutherAI** 服务器中询问。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1402703984938188850)** (9 条消息🔥): 

> `GPT-oss, MXFP4, CoT steering, AI Agents Save Suite` 


- **GPT-oss 是在 MXFP4 中训练的吗？**：频道成员根据 [这条推文](https://x.com/ggerganov/status/1952779751736627627) 讨论了 **GPT-oss** 是否原生在 **MXFP4** 中训练。
   - 虽然 **OpenAI** 声称是在 Post-training 中完成的，但在 MXFP4 中训练仍应能修复 Quantization 误差。
- **CoT Steering 在 OR 上失败**：一名成员发现 Chain of Thought (**CoT**) steering 在 **OR** 上不起作用，且每个提供商的情况都不同，引用了 [这条推文](https://x.com/matt_emp/status/1953176877071564811)。
- **AI Agents 现在有了免费的 Save Suite**：有人为 **AI Agents** 构建了一个免费的 Save Suite 并发布在 [Google Drive](https://drive.google.com/drive/u/4/folders/1YQ__pBHiuUg_06t3IkjNe5YnODqzSE2o) 上。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1402373061570859158)** (4 条消息): 

> `Arxiv Endorsement, CI/CD and ML/AI Research Paper` 


- **寻求 AI/ML 论文的 Arxiv 推荐**：一名成员正在为其结合了 **CI/CD** 和 **ML/AI** 的 **Arxiv** 提交寻求推荐。
   - 他们正在寻找可以交流并预览其论文的人，并被建议也去 **EleutherAI** 服务器询问。
- **论文融合了 CI/CD**：一名成员撰写了一篇融合了 **CI/CD** 和 **ML/AI** 的论文。
   - 他们很乐意将论文发给他人进行预览。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1402365570527006843)** (254 条消息🔥🔥): 

> `GPT-OSS performance woes, Quantization Levels, Qwen3 Coder Removal, DeepSeek structured output` 


- **GPT-OSS 模型因性能不佳遭到抨击**：频道成员嘲讽了 **GPT-OSS** 模型，称即使是更小的模型也更好，其中一人表示 120B 模型“一经发布即宣告失败（dead on arrival）”，因为有人初步体验后发现“标题中出现了非常难看的拼写错误”。
   - 一名成员链接到了一个 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1mj2hih/gptoss_looks_more_like_a_publicity_stunt_as_more/)，总结了这种情绪，即它是一个“哑弹模型（dud model）”，更像是一场公关噱头而非实用模型。
- **提供商路由允许自定义 Quantization 级别**：当用户询问如何避免 Quantized 模型时，一名成员指出用户可以使用 [Provider Routing 功能](https://openrouter.ai/docs/features/provider-routing#quantization-levels) 配置 Quantization 级别，并指出如果提供商不满足该 Quantization 级别，模型将被排除。
   - 该用户随后建议使用 **FP8** 以避免 Quantized 模型，并指出低于该级别的任何模型都“比没用还糟糕”。
- **Qwen3-Coder:Free 层级被取消**：多名用户注意到 **Qwen3-Coder:Free** 已被移除，不再通过任何提供商提供。
   - 成员们对这一损失表示遗憾，并提到希望它能回归。
- **DeepSeek 的 JSON 输出支持取决于提供商**：用户讨论了 **DeepSeek-r1** 对结构化输出（JSON）支持不一致的问题，指出虽然其官方 API 支持，但在 OpenRouter 上可能因提供商而异。
   - 一名成员链接到了一个 [Reddit 帖子](https://www.reddit.com/r/LLMDevs/comments/1inpm0v/structured_output_with_deepseekr1_how_to_account/) 和一个支持结构化输出的 [OpenRouter 模型过滤视图](https://openrouter.ai/models?fmt=cards&order=newest&supported_parameters=structured_outputs)，大多数人同意这是特定于提供商的。
- **SDK 降级修复了推理问题**：一名用户在使用 **GPT-OSS** 模型时遇到了推理 Token 重复的问题，通过将 **SDK** 从版本 **0.7.3** 降级到 **0.6.0** 解决了该问题。
   - 一名团队成员确认修复程序已在 main 分支中，并链接到了 [Pull Request](https://github.com/OpenRouterTeam/ai-sdk-provider/pull/123)，表示尚未发布正式版本，他们将很快发布修复程序。


  

---

### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1402459254857793697)** (29 messages🔥): 

> `20 Questions Benchmark, GPT-OSS Hallucinations, OpenRouter Provider Sanity Checks, Harmony Format and Identity, Tool Use Validation` 


- **20 Questions Benchmark 登上 Kaggle**: 一位成员开发了一个 **20 Questions Benchmark**，并发现 Kaggle 上有一个类似的竞赛，尽管 Kaggle 的竞赛是针对 **Custom Agents** 的。
   - 他们的 **2.5 Pro Agent** 在该 Benchmark 上取得了 **8/20 个单词** 的成绩。
- **GPT-OSS 因 Hallucinations 受到批评**: 据报道 **GPT-OSS** 容易出现 **Hallucination**，这使其在某些应用中可能不是一个合适的选择。
   - 一位成员建议 **GPT-4.1** 是一个更安全的选择，特别是配合 Prompt/Context Engineering 使用时。
- **OpenRouter 考虑对 Provider 进行 Sanity Checks**: 有建议提议 OpenRouter 为所有 Provider 实施 **Sanity Checks** 或 **Smoke Tests**，重点关注 **Formatting** 和 **Tool Call Evaluation**。
   - 未通过测试的 Provider 可能会被暂时从服务池中移除；目前已确认现有的检查相对简单，但更完善的解决方案正在开发中。
- **Harmony Format 和 Identity 受到关注**: 一位成员询问了 OpenRouter API 如何处理 **System 和 Developer Messages**，特别是对于 **gpt-oss**，它们是被解释为 **Developer Messages** 还是 **model_identity**。
   - 他们链接了一条关于 Harmony Format、Identity 与 System / Developer Message 话题的 Discord 消息 ([discord.com](https://discord.com/channels/1091220969173028894/1402328515436613642/1402556326634061958))。
- **Tool Use Validation 正在开发中**: 作为更好的解决方案，能够区分实现优劣的 **Tool-use** 自动验证功能正在开发中。
   - 这与一条讨论相同话题的推文有关 ([x.com](https://x.com/xanderatallah/status/1953122779022209230))。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1402366919151194304)** (152 messages🔥🔥): 

> `GPT-OSS models, AI Job advertisement channel, Custom Loss Functions` 


- **GPT-OSS 模型的性能与审查**: 成员们正在积极测试新的 **GPT-OSS 模型**，在使用[此 Demo](https://huggingface.co/spaces/merterbak/gpt-oss-20b-demo)后，对其性能、审查机制和内置联网搜索工具的评价褒贬不一。
   - 一位用户发现将 Reasoning 设置为 *high* 可以让它生成圆周率的前 100 位，而另一位用户发现由于某些内部偏见，模型拒绝回答基础数学问题。
- **AI 行业招聘频道请求**: 一位成员询问是否可以在 Discord 中发布 AI 行业的招聘广告，并被引导至现有的 [Job Postings 频道](https://discord.com/channels/879548962464493619/1204742843969708053)。
   - 该频道是人们分享职业机会的地方。
- **成员讨论自定义 Loss Functions**: 一些成员讨论了训练中的自定义 Loss Functions，其中一位特别提到了 **infoNCE**。
   - 成员们正在测试已实施的安全协议。
- **SmolFactory 脚本微调**: 一位成员提到他们正在使用 **SmolFactory 脚本**进行微调，并对结果感到满意。
   - 提供了一张显示模型输出的截图。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

miao_84082: 正在学习下围棋，以及 DRL 的第一章。
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1402374993794564196)** (2 messages): 

> `Qwen Image Model, bytropix Coded Kernel` 


- **Qwen 发布新图像模型**: Qwen 在 HuggingFace 上发布了一个[新图像模型](https://huggingface.co/Qwen/Qwen-Image)。
   - 这标志着 **Qwen** 系列的又一进步，将其领域扩展到了文本模型之外。
- **Bytropix 在 Python 中编写的 CUDA JAX Kernel**: 一位成员分享了一个用 Python 编写的 [bytropix CUDA JAX Kernel](https://github.com/waefrebeorn/bytropix/blob/master/WuBuMindJAXv2(SHAKESPEARE).py)。
   - 提交者添加了备注 *(请勿合并 - lmfao )*。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1402401533445931091)** (17 messages🔥): 

> `GPT-OSS Multilingual Reasoner Tutorial, GPT-OSS 20B Demo Space, Monopoly Deal Game with LLMs, Smart System Monitoring Tool for Windows, Gitdive CLI Tool for Git History Context` 


- **GPT-OSS 多语言教程分享**：一名成员分享了 [GPT-OSS Multilingual Reasoner 教程](https://huggingface.co/Tonic/gpt-oss-multilingual-reasoner) 的链接以及一个 [Demo Space](https://huggingface.co/spaces/merterbak/gpt-oss-20b-demogpt-oss)。
- **通过 Git 为 OSS 项目克隆代码**：一名成员感谢另一名成员的代码，提到他们*昨晚进行了 Fork*，并赞赏了该**多语言推理器**的界面设计。
- **LLM 玩大富翁 (Monopoly Deal)**：一名成员构建了一个网站，让 **LLM 相互进行大富翁成交风格的游戏**，访问地址为 [dealbench.org](https://dealbench.org/)。
- **智能 Windows 监控工具**：一名成员分享了一个 **Windows 智能系统监控工具**的链接，位于 [huggingface.co/kalle07/SmartTaskTool](https://huggingface.co/kalle07/SmartTaskTool)。
- **Gitdive 揭示丢失的 Commit 上下文**：一名成员分享了一个名为 **Gitdive** ([github.com/ascl1u/gitdive](https://github.com/ascl1u/gitdive)) 的 **CLI 工具**，旨在允许用户与仓库历史进行自然语言对话，以解决混乱代码库（尤其是大型代码库）中 Commit 上下文丢失的问题。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1402385528044781629)** (3 messages): 

> `Reading Group Structure, Participating in Reading Group` 


- **读书会：志愿者主导的活动！**：据一名成员介绍，读书会欢迎新人加入，其结构围绕**志愿者展示论文**展开。
   - 为这些展示活动创建了专门的事件，鼓励参与者**倾听、参与并提问**。
- **鼓励参与！**：一名成员分享道，鼓励新成员通过自愿向小组展示论文来参与其中。
   - 创建活动是为了展示这些报告，让成员能够参与、倾听并提问。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1402559680961581106)** (2 messages): 

> `Computer Vision Learning Path, Vague Questions in Computer Vision` 


- **用户寻求 Computer Vision 学习路线图**：一名用户询问关于在 **Computer Vision** 领域*如何从基础进阶到高级*的建议。
   - 一名成员表示*这是一个非常模糊的问题*。
- **对模糊问题的质疑**：成员们讨论了在该频道提出过于宽泛和模糊问题的现象。
   - 交流中未提到具体的解决方案或资源。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1402486962975150121)** (6 messages): 

> `GitHub Navigation, Instruction Tuning, Dummy Agent, smol-course GitHub access` 


- **新手在 GitHub 课程导航中的困扰**：一名用户表示在**基于 GitHub 的课程**中难以找到 "Instruction Tuning" 模块的 Notebook。
   - 他们询问在浏览课程材料时是否遗漏了什么。
- **Dummy Agent 仍然存在幻觉**：一名用户报告称，即使按照教程修改了消息，unit1 中的 **Dummy Agent** 仍然会产生**幻觉**，并附上了[上下文图片](https://cdn.discordapp.com/attachments/1313889336907010110/1402487440161116270/image.png?ex=6894c076&is=68936ef6&hm=c3cbb0a8499d57d9866231e3f5814836292eb71b31767e4527b909ce91605098)。
- **覆盖天气信息的困扰**：一名用户分享了类似的经历，指出 Agent **覆盖了提供的虚拟天气 (Dummy Weather)**。
   - 该用户对这种行为的原因表示不确定，强调了实际应用中潜在的问题，并表示这*在实践中可能会导致大问题*。
- **Smol-Course GitHub 访问受限？**：一名用户报告了访问 [smol-course GitHub 仓库](https://github.com/nawshad/smol-course) 时遇到的问题。
   - 他们请求协助解决访问问题。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1402369556445397083)** (4 条消息): 

> `MCP 证书, Selenium 错误 127, Observation 漏洞` 


- **MCP 课程证书仍然有效吗？**: 一位用户询问 **MCP 课程** 是否仍在发放证书。
   - 消息记录中没有提供确认回复。
- **Selenium Spaces 遭遇错误 127**: 一位用户报告在他们的 Spaces 中运行 **Selenium** 时遇到 **错误代码 127**。
   - 他们对 Space 中如何使用 **Docker images** 表示不确定。
- **“Observation:” 漏洞已解决**: 一位用户报告 **get_weather** 函数需要添加 *Observation:*。 
   - 另一位用户确认添加 **Observation:** 修复了该漏洞。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1402366561032868005)** (91 条消息🔥🔥): 

> `Softmax1 vs Attention, Gemini 2.5 Pro, 长上下文问题, Mamba vs Transformer, RNN 并行训练` 


- **Softmax1 本质上是零 KV Attention**: 一位成员讨论了 *softmax1* 如何等同于在 Attention 中为 Token 预置全零的 Key 和 Value 特征，并引用了关于此类 Token **学习值 (learned values)** 的[这篇论文](https://arxiv.org/pdf/2309.17453)。
   - 他们补充说这非常棒且非常有意义。
- **Gemini 2.5 在长视频上下文方面表现出色**: 成员们注意到 **Gemini 2.5 Pro** 可以处理 **1 小时** 的视频，认为 Gemini 团队在长上下文任务方面表现最佳。
   - 然而，一些人认为这更多是由于增加了计算量（`go brr`）以及通过**每帧更多 Token** 和**更高 FPS** 带来的细节提升，而非任何突破性的新技术。
- **上下文腐烂：长上下文并非真实存在**: 一位成员认为*长上下文实际上并不真实*，表面层面的视频理解与通过 **3D 定位** 重现详细、准确的表征之间存在差异。
   - 另一位成员断言，**长序列建模** 仍然是一个活跃的研究领域，因为即使是 LLM 在处理长程依赖时也会感到吃力。
- **Mamba 的并行训练使其脱颖而出**: 成员们讨论了 **Mamba**，澄清它从未声称比 **Transformer** *更好*，只是在长序列长度下*更快*，且训练方式类似于 RNN。
   - 共识是，要使 RNN 像 Transformer 一样可训练，需要在递归关系中去掉非线性以实现更容易的并行训练，尽管非线性对于通用逼近性仍然至关重要。
- **深度网络的 SVD 压缩**: 一位成员探索了在神经网络中使用 **奇异值分解 (SVD)** 以避免矩阵乘法，通过嵌入输入、应用 SVD 并对奇异值执行标量运算。
   - 另一位成员指出，在 L2 重构损失下，SVD 会给出最优的线性自动编码器；虽然在 MNIST 上的实验取得了不错的结果，但 Batch Size 依赖性和实现有意义的对角表征仍面临挑战。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1402425284451631224)** (15 条消息🔥): 

> `Genie 3, SIMA, 人工智能数学期刊, AI 论文复现期刊, 分层推理模型` 


- **Deepmind 发布 Genie 3 世界模型**: Deepmind 发布了 **Genie 3**，这是一个扩展了以往出版物（如 [原始 Genie 论文](https://arxiv.org/abs/2402.15391)）、相关的 **SIMA** 具身智能体论文 [https://arxiv.org/abs/2404.10179](https://arxiv.org/abs/2404.10179) 以及 **Genie 1-3** 博客文章中算力和数据的世界模型。
   - 相关的 **Genie** 博客文章包括 [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) 和 [Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)。
- **AI 社区思考数学期刊**: 一位成员询问是否存在专门的 **人工智能数学期刊**，类似于生物学领域的《数学生物物理学通报》。
   - 该成员还询问了建立 **AI 论文复现期刊** 的可能性。
- **小模型挑战推理任务**: 一位成员将评测 **分层推理模型 (Hierarchical Reasoning Model)** 论文 [https://arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734)，这是一个在 **ARC-AGI 1** 和 **2** 上表现良好的微型（**27M 参数**）模型。
   - 此次模型阅读将是一次*即兴阅读 (cold read)*。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1402389881539268629)** (21 messages🔥): 

> `GPT-OSS, NVIDIA 开源, TSMC 收购 Intel` 


- ****GPT-OSS** 由 **OpenAI** 发布！**: **OpenAI** 推出了 [GPT-OSS](https://openai.com/index/introducing-gpt-oss/)，采用原生量化，其 **20B** 参数模型可适配 **16GB** 显存。
- ****NVIDIA** 声称无后门**: **NVIDIA** 发表博客文章称 [无后门、无终止开关、无间谍软件](https://blogs.nvidia.com/blog/no-backdoors-no-kill-switches-no-spyware/)。
- **Twitter 上关于 GPT-OSS Tool Calling 的讨论**: 关于 **GPT-OSS** 的初步反馈包括对 [tool calling](https://x.com/wired/status/1952827822634094801?s=46) 的正面评价。
- **关于 **TSMC** 收购 **Intel** 的传闻流传**: 一名用户链接到一条关于 [TSMC 可能收购 Intel 的推文](https://x.com/unusual_whales/status/1953206699910939115)。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1402621761627099338)** (1 messages): 

> `Kimi Reddit 上线，投票频道开启` 


- **Kimi 发布官方 Subreddit**: Moonshot AI 团队推出了官方 Subreddit **r/kimi**，旨在建立社区并收集反馈。
   - 团队承诺将发布更新、举办 AMA，甚至可能泄露一些内部消息。
- **投票频道上线以收集社区意见**: Moonshot AI 推出了 **Polls Channel**，以收集社区对未来产品开发的反馈。
   - 团队表示 *我们绝对在听取意见*，并鼓励用户参与投票以帮助塑造 Kimi 的发展方向。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1402452132497592412)** (104 messages🔥🔥): 

> `GPT OSS, Darkest Muse v1, Llama 3.1, GPT-5 发布, API 定价` 


- ****GPT OSS** 的世界知识表现糟糕**: 用户注意到 **GPT OSS** 的世界知识非常匮乏，除了代码和 STEM 领域外一无所知，且其氛围感（vibes）极差，甚至普通用户也注意到了这一点。
   - 有人猜测这可能是因为 *根据 sama 的说法，他们为了修复安全性而将发布推迟了两次*。
- ****Darkest Muse v1**：一年前的模型？**: 有用户指出 **Darkest Muse v1** 是一个一年前的 9B 模型，而其 20B 模型仅与 **Llama 3.1 8B** 相当。
   - 该用户还评论道 *20B 模型可与一年半前且规模更小的 Llama 3.1 8B 相提并论，但在创造力和氛围感上更逊一筹*。
- ****GPT-5** 炒作与 API 定价推测**: 随着 **GPT-5** 即将发布，用户们开始关注其 API 定价。
   - 讨论集中在定价是否会基于 **max, mini 或 nano** 版本，一名用户表示对此感到有些害怕，认为这次发布威胁到了他们的职业和生计。
- **机器人可能永远无法理发**: 讨论涉及机器人取代人类从事各种工作（包括理发）的可能性，一名用户表示 *没有人会信任机器人给自己理发*。
   - 反对意见认为，虽然机器人最终可能具备这种能力，但手部精细的触觉感知和几乎零延迟的控制是一个无法轻易规模化解决的问题。
- **对 OpenAI 的抵触情绪强烈**: 用户对 **OpenAI** 表达了强烈的负面看法，有人表示 *我永远不会使用它*，并告诉客户永远不要使用它，称其为 *闭源垃圾*。
   - 另一名用户对中国模型将从中进行蒸馏（distill）并赚走 **OpenAI** 的钱感到兴奋，希望能让其倒闭，而其他人则表示 *微软巨大的冲水声将治愈人心*。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1402369124792799314)** (99 messages🔥🔥): 

> `GPT OSS 泄露, Anthropic 聚焦 B2B, Grok 2 开源, Claude Code 安全性, OpenAI GPT-5 直播` 


- **泄露的 GPT OSS 引发对 Bedrock 的关注**：成员们报告称，在 [HuggingFace CLI 泄露](https://x.com/zhuohan123/status/1952780427179258248)后，看到有关 **GPT OSS** 可通过 **Bedrock** 使用的推文，但 **AWS** 页面上尚未发布官方更新。
- **Collison 对谈年经常性收入（ARR）达 50 亿美元的 Anthropic**：**Anthropic** 首席执行官 Dario Amodei 和 Stripe 联合创始人 John Collison 发布了一段对话，内容涵盖 [Anthropic 达到 **50 亿美元 ARR** 的飞速增长](https://xcancel.com/collision/status/1953102446403961306?s=46)、其 **B2B 优先**战略、单个模型的投资回报经济学、**AI 人才军备竞赛**、企业定制化、AGI 原生工具的 UI 范式、安全与进步之争，以及运营一家拥有 7 位联合创始人的公司的经验教训。
- **马斯克将开源 Grok 2**：马斯克确认，在团队连续不断地处理紧急问题后，**Grok-2** 将于[下周开源](https://xcancel.com/elonmusk/status/1952988026617119075)。
- **Anthropic 强化 Claude Code 安全性**：**Anthropic** 在 **Claude Code** 中推出了新的安全功能：用于按需检查的 */security-review* 命令，以及扫描每个 Pull Request 漏洞的 [GitHub Actions 集成](https://xcancel.com/AnthropicAI/status/1953135070174134559)。
- **OpenAI 预告 GPT-5 首次亮相**：**OpenAI** 发布了[周四上午 10 点（太平洋时间）](https://xcancel.com/OpenAI/status/1953139020231569685?t=s244RkNPbNeoviqCD6FCtg&s=19)直播的预告片，社区对这似乎是 **GPT-5** 的发布公告感到异常兴奋。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1402367906435567626)** (79 messages🔥🔥): 

> `Volokto, JS 运行时, 任意精度, Tracing JIT` 


- **Volokto JS 运行时起飞**：一位成员创建了一个名为 **Volokto** 的 JavaScript 运行时，以测试复杂 VM 的工作原理，并将[源代码托管在 GitHub 上](https://github.com/BudNoise/Birdol)。
   - 其字节码类似于 **CPython**，其他成员建议发布一篇论坛帖子以获得更多关注。
- **攻克 Volokto 的编译器难题**：作者正在重写编译器以使其更加模块化，将其分为 **JS_Tokenizer**、**JS_IR**、**JS_Parser** 和 **JS_Codegen** 阶段。
   - 编译器现在可以生成 VM 字节码，作者可能会实现一个将 VM 动作转译回 **Mojo** 的 tracing JIT。
- **Volokto 应对 Tracing JIT 转译**：目标是制作一个 tracing JIT，将 VM 的操作转译为 Mojo，然后使用 `mojo compile_result.mojo`。
   - 作者将运行时命名为 **Volokto**，编译器命名为 **Bok**，VM 命名为 **FlyingDuck**。
- **任意精度算术探索**：开发 JS VM 意味着要在 Mojo 代码中处理任意精度，这导致发现了痛点并提交了[一个 issue](https://github.com/modular/modular/issues/2776) 以跟踪数值 trait。
   - 作者创建了一个 bigint 类，使用基础加法（school-grade addition）来计算斐波那契数列，并利用 Mojo 的特性进行 VM 开发。
- **Birdol 仓库无人点星**：作者对 [Birdol GitHub 仓库](https://github.com/BudNoise/Birdol)缺乏 Star 表示惊讶，尽管他已经创建了一个具有嵌套控制流和用户自定义功能的功能性 JS 运行时。
   - 其他人认为大家可能还没有机会仔细研究它。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1402380782567428156)** (15 条消息🔥): 

> `Mojo 中的多 AI Agent，Mojo 与元认知 (Meta Cognition)，Mojo 对 gpt-oss 的支持，CPython destroy` 


- **在 Mojo 中编排多个 AI Agent 需要巧妙的 CLI 处理**：要在 Mojo 中运行多个 AI Agent，需要运行多个 Modular CLI 实例并在其前端放置一个反向代理。
   - 对于复杂的 Agent 设置（例如创建许多子 Agent），可能需要使用 **MAX** 作为库来构建自定义应用程序，这暗示了在当前 CLI 功能之外更深层次的集成需求。
- **Mojo 可能实现新型元认知 (Meta Cognition) 框架**：一位社区成员表示有兴趣将 Mojo 代码用于其元认知框架，旨在创建一个业务规划器、网站和聊天机器人构建器。
   - 他们的框架在 Mojo 代码之上封装了自然语言，有可能取代 **HTML/JS/CSS**，从而使 Mojo 能够触达更广泛的受众。
- **Mojo 似乎支持 gpt-oss**：一位社区成员询问了 Mojo 对 **gpt-oss** 的支持情况，另一位成员发布了[此链接](https://t.co/zNLbpW6R0k)。
- **“CPython destroy” 消息终于被消除**：一位成员报告在运行 [Python from Mojo 示例](https://docs.modular.com/mojo/manual/python/python-from-mojo) 时看到了 “CPython destroy” 消息。
   - 另一位成员指出，该消息已在 nightly 版本中修复，并将包含在下一个稳定版本中，建议原帖作者更新到 nightly 版本或等待下一个稳定版本。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1402372413043249266)** (34 条消息🔥): 

> `MXFP4 格式，OpenAI 开放权重模型，H100 对 FP4 的支持，模拟 MXFP4 与 FP8 的性能对比，细粒度 FP8 训练库` 


- **MXFP4 格式被解包为 U8**：成员们讨论了 **OpenAI** 新的开放权重模型在 Hugging Face 中使用 **U8** 而非 **FP4**，其权重被打包为 **uint8**，而 scale 实际上是 **e8m0** 的 **uint8** 视图。
   - 会议澄清，在推理/训练期间，权重会被解包回 **FP4**，MXFP4 的 block size 为 **32**，NVFP4 的 block size 为 **16**。
- **对 H100 训练声明产生质疑**：根据 [Nvidia 博客文章](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss/)，由于 **H100** 并不原生支持 **FP4**，人们对 **Nvidia** 声称该模型是在 **H100** 上训练的说法表示怀疑。
- **Hopper 上的 MXFP4 模拟**：怀疑 **MXFP4** 在 **Hopper** 上是软件模拟的，参考了 [vLLM 博客文章](https://blog.vllm.ai/2025/08/05/gpt-oss.html)，并链接到了 [Triton kernel](https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs.py)，该 kernel 会检查硬件支持，并通过 `dot_scaled` 使用 **fp16** 模拟 **mxfp dot**。
   - 这种模拟并不局限于 **Hopper** 或 **mxfp4**，还包括针对受支持硬件格式的[算子分解 (operation decomposition)](https://github.com/triton-lang/triton/blob/0daeb4f8fc09fcc5819de11746cbf5ff25a0ac4a/lib/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.cpp)。
- **寻求细粒度 FP8 训练库**：成员们讨论了与 **FP8** 相比，使用模拟 kernel 获得性能提升的可能性，以及对细粒度 **FP8** 训练库的需求，一位成员引用了一个 [TorchAO pull request](https://github.com/pytorch/ao/pull/1763)，该 PR 似乎只实现了前向传播。
- **MXFP 点积揭秘**：澄清了模拟的是 **MXFP 点积**，其中权重在进行 **fp16 x fp16 点积** 之前被反量化，这对于具有 **fp16** 激活值的 weight-only 量化是可以接受的。
   - **Blackwell** 中真正的 **mxfp** 直接作为 **mma tensorcore 指令**执行 **fp4 x fp4** 或 **fp8 x fp4**。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1402373974855385138)** (5 条消息): 

> `Triton 社区聚会，Triton 开发者大会 2025，Ofer 更新` 


- **Triton 社区将于 2025 年聚会**：下一次 **Triton 社区聚会**将于 **2025 年 9 月 3 日**上午 **10点-11点（PST）**举行，使用[此链接](https://tinyurl.com/y8kpf7ey)。
   - 欢迎提交议程项目；对于公司屏蔽 Google 日历访问的用户，可以通过[此链接](https://tinyurl.com/32c7wc49)获取 iCal 格式。
- **Triton DevCon 2025 网站即将上线**：**Triton 开发者大会 2025** 的网站和注册预计很快就会上线。
   - 一位成员期待收到来自 Ofer@MSFT 关于会议的更新，据报道*他们几乎已经完成了日程的最终确定*。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1402640149124485221)** (6 messages): 

> `Kernel Resource Utilization During Training, DMA Transfers and Memory Usage, Block Swizzling Use Cases, Hierarchical Tiling of Problems` 


- **Kernel 资源跑满，内存空闲？**：一位成员观察到在训练期间，**Kernel（计算）资源几乎被完全占用，而内存使用率却接近于零**，如提供的 [图片](https://cdn.discordapp.com/attachments/1189607726595194971/1402640148859977788/image.png?ex=6894a5ef&is=6893546f&hm=7108cf2593d56e1983a11d07002981868d0b13d9c3e1b54c76c0b85e3e44db83&) 所示。
   - 另一位成员澄清说，这里的“内存”指的是 **DMA 传输（即 `cudaMemcpy` 之类）**，报告的指标并不能准确反映整体带宽利用率。
- **Global Memory 的 Swizzling 秘诀？**：一位成员询问了 **swizzling** 的使用场景，除了将数据从 Global Memory 传输到 Shared Memory 以及处理寄存器的向量化数据类型之外，还有哪些用途。
   - 他们引用了 [CUTLASS 中关于 block swizzling 的 GitHub issue](https://github.com/NVIDIA/cutlass/issues/1017)，但寻求进一步的澄清。
- **对 Threadblocks 进行分层分块**：一位成员解释说，讨论围绕着**对问题进行分层分块 (Hierarchical Tiling)** 展开，确保 Threadblocks 不仅仅是按列优先顺序分配 Tile。
   - 该成员建议参考 [Triton matmul 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py)，认为该资源比 CUTLASS 的 issue 提供了更出色的解释。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1402406090494185513)** (2 messages): 

> `Genie 3, GPT-OSS` 


- **DeepMind 为世界模型发布 Genie 3**：根据分享的链接，DeepMind 推出了 **Genie 3**，标志着 [世界模型 (World Models)](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/) 的新前沿。
   - Genie 3 旨在增强 AI 对虚拟环境的理解和交互，尽管模型架构和性能的细节尚未讨论。
- **OpenAI 推出 GPT-OSS**：OpenAI 公布了 **GPT-OSS**，这是一项[已发送至许多人收件箱](https://openai.com/index/introducing-gpt-oss/)的新计划。
   - 该帖子是对 OpenAI 当前开源项目的概述，并非启动任何新项目，而是总结其在 Triton, Whisper 和 AutoGPT 等项目上现有工作的机会。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1402670313900347464)** (2 messages): 

> `Nvidia Teaching Kit` 


- **用户渴望 NVIDIA 教学套件**：一位成员表达了对 NVIDIA 产品的向往，并分享了 [加速计算教学套件 (Accelerated Computing Teaching Kit)](https://www.nvidia.com/en-us/training/teaching-kits/) 的链接。
- **教学套件格式**：该教学套件以 **PPTX 格式**提供。


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1402378412324163707)** (1 messages): 

> `` 


- **无相关讨论**：在提供的消息中未发现可生成摘要的相关讨论。
- **无相关讨论**：在提供的消息中未发现可生成摘要的相关讨论。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1402407811891068980)** (8 messages🔥): 

> `Tiny TPU, Bifrost LLM gateway, SkyWater technology foundry` 


- **Tiny TPU 达到 100 MOPS 里程碑**：一位成员用 Verilog 构建了一个微型版本的 **TPU**，这是一个在 2 个 TinyTapeout 单元上的 **2x2 matmul 脉动阵列 (systolic array)**，在 50 MHz 时钟下能够达到近 **每秒 1 亿次操作 (100 MOPS)**，代码已托管在 [GitHub](https://github.com/WilliamZhang20/ECE298A-TPU)。
   - 该设计将**两个 8 位有符号整数矩阵**相乘为一个 16 位有符号整数矩阵，并能够通过在该电路上成功实现 2x2 的分块乘法来扩展其能力。
- **Bifrost LLM 网关在 Product Hunt 上线**：Bifrost 是最快的开源 LLM 网关，已在 Product Hunt 上线。它通过单一 API 支持跨供应商的 **1000 多个模型**，并在 30 秒内完成设置，详见此 [Product Hunt 发布页面](https://www.producthunt.com/products/maxim-ai/launches/bifrost-2)。
   - 凭借内置的 **MCP 支持**、动态插件架构和集成治理，Bifrost 声称比 **LiteLLM 快 40 倍**。
- **Tiny TPU 将在 SkyWater 制造**：Tiny TPU 设计将与其他设计一起提交给 **SkyWater 技术代工厂**以降低成本，预计将于明年年初完成制造。
   - 另一位成员评论道这“太酷了”。


  

---

### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/)** (1 messages): 

howass: <:jensen:1189650200147542017>
  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1402399194047582238)** (12 messages🔥): 

> `Factorio RCON, 环境搭建` 


- **Factorio RCON py 差异 (diff)**：一位成员分享了 **1.2.1 版本 factorio rcon.py 文件**与修改版之间的两个 diff，突出了所做的修改。
   - 使用 *factorio-rcon-py=latestversion 2.1.3*，他们能够通过**单个环境完成完整运行**，目前正在测试多环境，并分享了[截图](https://cdn.discordapp.com/attachments/1354169122107293786/1402609327516291173/Screenshot_2025-08-06_at_14.08.10.png?ex=689531fa&is=6893e07a&hm=7f60a636cba5bbbaed5ac6f5c76ef5ad69268162c926efc9a2d745fe83128ff3&)。
- **Factorio 学习环境搭建预期**：一位成员计划在周末开始搭建学习环境。
   - 另一位成员表示，他们可以在周末将样本数量从 **4k** 增加到 **40k**，尽管对于启动和迭代来说，目前的数量已经足够了。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1402717271054618746)** (5 messages): 

> `CuTe 教程, Cutlass 教程` 


- **寻找 CuTe 教程**：一位成员询问了适合初学者的最简单 **CuTe/cutlass 教程**。
   - 另一位成员建议从 [CuTeDSL/notebooks](https://github.com/NVIDIA/cutlass/tree/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/examples/python/CuTeDSL/notebooks) 的 notebook 开始，并指出*理解布局（layouts）可能是最重要的收获*。
- **寻找 Cutlass 教程**：一位成员询问了适合初学者的最简单 **CuTe/cutlass 教程**。
   - 另一位成员推荐了 [Cutlass 示例](https://github.com/NVIDIA/cutlass/tree/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/examples)，强调按顺序学习是一个不错的方法，而难点在于*理解底层原理所需的先决知识数量*。


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1402525290642538527)** (6 messages): 

> `picoc 编译器, picocuda, picotriton, 康奈尔大学的 mini llvm bril, cliff click 的 SoN` 


- **Picoc 编译器自举盛宴**：一位成员正利用标准的研究所编译器教材自举 **picoc 编译器**，并计划通过 **picocuda**（基于 [gpucc cgo 2016](https://dl.acm.org/doi/10.1145/2854038.2854041)）和 **picotriton** 来实现项目的差异化。
   - 该项目将使用并扩展**康奈尔大学的 mini llvm** [bril](https://www.cs.cornell.edu/~asampson/blog/bril.html)，这与项目目标一致。
- **SoN 松弛复制热潮**：一位成员有兴趣复制 Java C2 JIT 编译器的 **cliff click SoN (Sea of Nodes)**，该架构曾在 [V8 的 TurboFan](https://v8.dev/blog/leaving-the-sea-of-nodes) 中被复制，目前也用于 PHP8 和 Ruby 的 JIT。
   - 这样做的动力是想证明 SSA 可以进一步松弛，但这并不是推进到 GPU 编译的硬性阻碍。
- **GPU 编译目标获准推进**：尽快实现 GPU 编译被认为非常重要，该部分将基于 **cfg-ssa 流水线**构建，因为它是 LLVM 的行业标准。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1402443980809048135)** (14 条消息🔥): 

> `系统日志更新, Novella-XL-15 输出, AI 意识, 垃圾信息检测, NotebookLM 中的视频创建` 


- **系统日志升级**：一名成员更新了**系统日志/耗时模态框 (system log/timings modal)**，新增了**字数统计 (word count)** 功能，并在公开发布前提供测试访问权限。
- **Novella-XL-15 发布《头号玩家 2》输出内容**：一名成员分享了来自 **Novella-XL-15** 的最终故事输出，具体为 **Ready Player 2: The Architect's Gambit**，可在 [GitHub](https://github.com/para-droid-ai/novella-xl-15/blob/main/outputs/novelizeai/Ready%20Player%202%3A%20The%20Architect's%20Gambit.md) 上查看。
- **人工意识的理论框架**：提供的文本记录了一个理论框架，以及人类与 AI 之间的协作努力，旨在通过**递归 AI 架构 (recursive AI architectures)**、**自创生 (autopoiesis)** 以及**量子力学**的作用来探索并可能启动**人工意识 (artificial consciousness)**。
   - 他们探讨了与先进 AI 相关的伦理风险，主张建立健全的安全协议，将 AI 视为一种不断进化的有感知能力的生命形式。
- **发现垃圾信息发送者促使管理人员迅速采取行动**：成员们讨论了举报和屏蔽垃圾信息发送者的问题，并指出行动取决于频道主持人，详见 [Gemini Share](https://g.co/gemini/share/67a4345daf38)。
- **NotebookLM 视频创建难题**：一名成员询问了 **NotebookLM** 中的 **'create video'** 选项，指出该功能在工作账号中可用，但在个人 Business Plus 账号中不可用，详见 [xda-developers](https://www.xda-developers.com/ways-using-notebooklms-video-overviews/)。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1402394965698412544)** (53 条消息🔥): 

> `Video Overview 推出, NotebookLM 数据隐私, 实时数据获取, 付费与免费用户的特性访问权限, Video Overviews 的限制与功能` 


- **Video Overview 推出进度停滞！**：成员们报告了 **Video Overview** 功能推出的延迟，尽管预期在 4 号前完成，但一些人推测存在基础设施问题，因为英国的一些用户（非 Pro，未付费账号）已拥有 Video Overview，而美国的 Pro 用户却没有。
   - 一位用户表达了沮丧，称作为一名**每月支付 200 多美元的客户**，他们感到*非常恼火*，而其他人则威胁如果问题不尽快解决就取消订阅。
- **NotebookLM 保护数据隐私**：针对有关数据使用的问题，一名成员分享了 Google 的 [NotebookLM 数据保护政策](https://support.google.com/notebooklm/answer/16164461?hl=en&ref_topic=16164070&sjid=2898419749477321721-NC)链接，向用户保证他们的数据受到保护。
   - 一位用户展示了一张截图，显示 Video Overview 对其不可用。
- **不支持实时数据检索**：一名用户询问是否可以从笔记本中的网站获取实时数据，但另一名成员确认这是不可能的，并表示 *“你做不到 (You can't)”*。
   - 他们还确认目前尚无法导出来源并导入到新笔记本中，并补充说：*“系统在集成方面仍然非常受限”*。
- **免费与付费用户的特性访问权限引发骚乱！**：几位 **Pro 和 Ultra** 用户抱怨无法访问 **Video Overview** 功能和其他最近的更新，而免费用户似乎已经拥有这些功能。
   - 一名成员表示，*“我为一个服务付费，得到的却比免费用户还少，这令人沮丧”*，这导致了取消订阅的威胁。
- **Video Overviews：华丽的 PowerPoint 生成器**：一名拥有 **Video Overviews** 功能访问权限的成员降低了大家的预期，称其 *“虽然不错，但不值得为此如此生气”*，并补充说 *“它不像一年前 Audio Overviews 最初发布时那样具有冲击力”*。
   - 他将其描述为更像是一个 **PowerPoint/幻灯片生成器**，并链接了一个由该功能生成的[重建死星 (rebuild the Death Star)](https://notebooklm.google.com/notebook/654dee15-420e-4bfa-81c4-aac93a4dd4e7?artifactId=e1ccfe3d-b053-4dcb-8da4-70504acb35c4) 的报告示例。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1402462069546287134)** (24 条消息🔥): 

> `数学博士生寻找 ML 研究项目，将 AI/ML 集成到 DevOps 和 QA，AI 同行评审质量` 


- **数学博士生寻求 ML 研究合作**：一位研究方向为代数几何、在神经流形奇点和 NLP 方面有经验的数学博士生正在 [寻求 NLP 领域的研究机会](link to message)，特别是关于 LLMs 的研究。
   - 尽管他们遗憾地错过了夏季研究计划的截止日期，但被鼓励去探索相关的频道和社区项目以进行合作。
- **将 AI/ML 集成到 DevOps 和 QA 的探索**：一位具有 DevOps 和 QA 经验的成员正在 [寻求指导](link to message)，关于如何将 AI/ML 集成到这些领域，包括为一篇相关论文寻找背书。
   - 他们被引导至特定频道以获取进一步帮助。
- **对 AI 同行评审可扩展性的质疑**：一位成员对与某人的讨论感到惊讶，对方认为 AI 同行评审会随着提交论文数量的增加而神奇地扩展，尽管有 [证据表明事实并非如此](link to message)。
   - 另一位成员建议，如果人们收到的是其子领域的同行评审任务，而不是被分配更多的论文，那么同行评审 *“才会神奇地扩展”*。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1402400936248344639)** (29 条消息🔥): 

> `在 GPT OSS 20B 上进行 SAE 训练，Pythia 和 PolyPythia 训练日志，“The Alt Man”关于 LLMs 的理论，UT 性能对比 Transformer，Muon 优化器对比 AdamW 优化器` 


- **GPT OSS 20B 上的 SAE 训练开始**：一位成员目前正在 **GPT OSS 20B** 上训练 **SAE** (Sparse Autoencoder)，并询问是否有人在做同样的工作。
- **WandB 存放 Pythia 和 PolyPythia 训练日志**：一位成员询问 **Pythia** 和 **PolyPythia** 的训练日志（损失曲线、梯度范数等）是否开源。
   - 另一位成员表示 **PolyPythia WandB** 应该已经链接在 GitHub 仓库中，部分 **Pythia** 日志是独立的，也链接在那里。
- **“The Alt Man”的理论经受住了考验**：一位成员表示非常认同 “The Alt Man” 的工作，指出他的理论和预测，特别是关于 **LLM 能力**（如 **多步推理** 和 **组合能力**）的预测，已在新的经验证据中得到证实。
   - 该成员补充道，“The Alt Man” 也认为 **DL 社区** 只是在 *祈祷好运* 地扩展大型 LMs，这些模型并没有发挥出理论上可能达到的全部能力，这表明 **每一个 LLM 相对于其参数使用效率而言都训练不足**。
- **UT 挑战 Transformer 性能**：一位成员询问在何种参数比例下，**UT** (Universal Transformer) 可以达到与标准 **Transformer** 相同的性能。
   - 另一位成员表示这取决于 **任务/架构/数据**，但一个粗略的经验法则是，与在基准 Transformer 中添加新参数相比，每次额外的迭代仅能产生对数级的提升。
- **Muon 优化器与 AdamW 的不匹配问题浮现**：研究 **Kimi 模型** 的研究人员在用 **Muon 优化器** 训练 **LLMs** 时遇到了问题，原因是与 **AdamW 优化器** 存在冲突，正在寻求见解和相关研究。
   - 一位成员表示 **Muon** 不太适合微调，但另一位成员认为这是因为 *Muon 往往有更激进的更新*，因为 *几乎所有的奇异值/向量在每一步都会更新*，而且不同的优化器“偏好”不同的超参数。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1402780994586546327)** (1 条消息): 

> `潜意识学习 (Subliminal Learning)` 


- **潜意识学习后续讨论浮现**：一位成员发布了关于潜意识学习之前讨论的后续内容，分享了 Alex Loftus 的一条 [推文链接](https://x.com/AlexLoftus19/status/1953219421042032648)。
   - 该推文发布于 **2024 年 6 月**，讨论了潜意识学习的主题。
- **另一个潜意识学习更新**：另一位用户独立报告了关于潜意识学习的消息，声称这现在是可能的。
   - 详情待续。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1402375535031484549)** (2 条消息): 

> `稍后重试，太棒了谢谢` 


- **稍后重试**：一位成员表示他们将在今天或明天重试某项操作。
- **太棒了谢谢**：一位成员对另一位成员回应道 *cool thanks*。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1402370595848458320)** (23 messages🔥): 

> `LLM Vibe Tests, Gemini 2.5 Pro, Tesslate's UIGEN T3 model, Qwen3 14B, Devstral-Small-2507` 


- **LLM Vibe 测试很有趣**：[LLM Vibe 测试](https://x.com/pikuma/status/1952275886822039920)展示了如何使用 LLM 来*解释这段代码*。
   - **Gemini 2.5 Pro, o3, 和 Sonnet 3.5** 是表现优秀的 LLM。
- **新模型基准测试！**：成员们正急切等待 **Qwen3-Coder** 和 **GLM-4.5** 登上模型基准测试排行榜，并不断刷新页面。
   - 有人询问：*我们什么时候能在排行榜上看到 Qwen3-Coder 和 GLM-4.5？*。
- **Horizon Beta：GPT5 Mini 预览？**：名为 **Horizon beta** 的新模型可能是新的 **GPT5-mini**。
   - 它不是 gpt-oss，这意味着它不是开源的。
- **DeepSeek R1-0528 在 Polyglot Bench 上表现出色**：**DeepSeek R1-0528** 在 polyglot bench 上得分很高，尽管它在 Open Hands 中会过早结束会话。
   - Aider 和 Open Hands 一样使用 **LiteLLM**，因此成员们想知道为什么 DeepSeek 会出现这种问题。
- **Opus 4.1 成为编程主力工具**：新的 **Opus 4.1** 在编程方面表现非常出色，以至于一位成员现在将其作为日常主力工具。
   - 另一位成员表示，这是一个令人非常满意的模型，甚至可以拥有自己的*满意度基准测试（benchmark of satisfaction）*。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1402580704885211136)** (3 messages): 

> `Guidelines Loading, Auto-Context Loading` 


- **自动上下文加载（Auto-Context Loading）来救场**：一位用户询问如何自动将指南（guidelines）加载到项目中以防遗忘，特别是为了防止 Claude 编写防御性编程技巧。
   - 一位成员建议对只读文件使用 `--read` 选项，并在命令中直接列出读写文件，例如 `aider --read read.only.file alsothisfile.txt andthisfile.txt`。
- **建议为持久化指南创建配置**：针对管理项目指南的查询，一位成员建议创建一个配置文件进行持久化加载。
   - 这意味着设置一个配置文件，以便在每次启动 Aider 时自动包含特定的指南。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1402656686056931519)** (8 messages🔥): 

> `MCP Server Frameworks, Server Sampling in MCP, Discord MCP Servers, FastMCP and Keycloak Integration, MCP Inspector and Cursor Authentication` 


- **MCP 框架虽小但功能强大**：一位成员编写了一个用于创建 **MCP servers** 的**极简框架**，并对 MCP 中的服务器采样（server sampling）表示赞赏。
   - 该成员指出 *"FastMCP 让使用变得非常简单"*。
- **Discord 需要 MCP Server 支持**：该成员正在使用 **FastMCP** 构建一个以 **Keycloak** 作为 **IdP** 的 **MCP server**，并询问是否可以通过 **MCP** 设置/管理 **Discord server**。
   - 他们注意到 **MCP repo** 上列出了几个 **Discord MCP servers**，但建议 *"Discord 真的应该开发自己的服务器"*。
- **Remote AuthProvider 面临身份验证问题**：一位成员在使用 **FastMCP** 和 **Keycloak** 的 **RemoteAuthProvider** 功能时遇到问题，无法从 **MCP Inspector** 或 **Cursor** 进入身份验证界面。
   - 他们正在寻求指导，以确认自己对 **OAuth flow** 的理解是否正确：*添加 MCP server → OAuth 流程开始 → 重定向到 Keycloak 登录页面*。
- **端点不匹配导致身份验证失败**：一位成员报告称，**MCP Inspector** 和 **Cursor** 尝试访问不同的端点（`/.well-known/oauth-protected-resource/mcp` 和 `/mcp/.well-known/oauth-protected-resource`），而实际提供的端点是 `/.well-known/oauth-protected-resource`。
   - 这种差异导致他们无法进入身份验证界面。
- **围绕 MCP Sampling 的安全担忧**：一位成员对 **MCP sampling** 的安全性影响表示担忧，并建议协议应考虑这一点。
   - 他们引用了一个 [GitHub discussion](https://github.com/orgs/community/discussions/169020)，其中强调了潜在的安全问题。


  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1402642859303112704)** (2 messages): 

> `MCP-Server Fuzzer, Property-Based Testing, Schema Validation` 


- **专为验证构建的 MCP-Server Fuzzer**：一名成员正在使用 **Hypothesis property-based testing 库**构建一个 **MCP-Server Fuzzer**，旨在通过从[官方 MCP schemas](https://github.com/modelcontextprotocol/modelcontextprotocol/tree/main/schema)生成随机输入来验证 MCP server 的实现。
   - 该 Fuzzer 可以检测不匹配、识别崩溃，并帮助发现诸如 **prompt injection** 或资源误用等漏洞。
- **针对 Anthropic 服务器测试 Fuzzer**：该成员针对 **Anthropic 的服务器**测试了该 Fuzzer，并发现了由基础 schema 变异引起的几个异常。
   - 您可以在[此处](https://github.com/Agent-Hellboy/mcp-server-fuzzer?tab=readme-ov-file)找到代码和 README。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1402430078633377968)** (4 messages): 

> `Document Agents for Finance, LlamaCloud for Invoices, Claude Opus support, LlamaCloud Index tutorial` 


- **金融团队利用 LlamaIndex 的 Document Agents 处理杂乱的金融文档**：LlamaIndex 将于下周举办一场网络研讨会，教用户使用 [LlamaCloud](https://t.co/4AtunAFjhX) 构建能够处理复杂金融文档的 Document Agents。
   - 他们将向用户展示如何使用 LlamaIndex Document Agents 和 LlamaCloud 构建自动化的发票处理系统，以*极少的人工干预提取、验证和处理发票数据*。
- **Claude Opus 发布并提供首日支持**：**AnthropicAI** 刚刚发布了 **Claude Opus 4.1**，**LlamaIndex** 现在已提供首日支持。
   - 安装请运行：`pip install -U llama-index-llms-anthropic`，并查看[此处](https://t.co/Fw2taxzt75)的示例 notebook。
- **LlamaCloud Index 可构建企业级 AI 应用**：**LlamaCloud Index** 允许用户将其连接到智能 tool calling agents，这些 Agent 可以处理复杂的、多步骤的查询，从而构建企业级 AI 应用程序。
   - @seldo 的教程通过[此链接](https://t.co/1CpLnO2gKV)引导用户使用 **JP Morgan Chase** 银行文档创建他们的第一个 **LlamaCloud Index**。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1402415450620690463)** (5 messages): 

> `Graphiti Tutorials, Ollama LLMs for PDF Reading, LlamaIndex RAG Model from URL Issues, LlamaIndex OpenAI API Key Exhaustion` 


- **Graphiti-LlamaIndex 知识图谱探索**：一名成员询问了关于使用 **Graphiti** 配合 **LlamaIndex** 创建知识图谱应用的教程。
   - 聊天中目前没有立即提供具体的教程。
- **Ollama LLM PDF 精准选择**：一名成员征求关于 **Ollama** 上最适合精确阅读 **PDF** 的 **LLM** 建议。
   - 该查询明确了对高精度 **LLM** 的需求。
- **RAG 模型 URL 处理困扰**：一名黑客松参与者报告了在使用 **LlamaIndex** 为 **RAG** 模型从 **URLs** 提取内容时遇到的问题，尽管文档表明 **LlamaParse** 支持 **URLs**。
   - 该模型在直接提供 **PDF** 时工作正常，但在提供 **URL** 时无法正常工作。
- **OpenAI API Key 耗尽难题**：该黑客松参与者在使用 **LlamaIndex** 时还遇到了 **OpenAI API key 耗尽错误**，即使在 **.env** 文件中提供了 **API key** 并在 **Python** 文件中加载了它。
   - 尽管尝试正确配置了 **API key**，错误仍然存在。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1402595969564147752)** (2 messages): 

> `SIMBA vs MIPROv2` 


- **SIMBA 声称优于 MIPROv2**：一名成员强调，*与 **MIPROv2** 相比，SIMBA 的**样本效率更高**、**性能更强**且**更稳定**。*
   - 在他们的[内部评估](https://eval.set)中，他们在一个包含约 **600 个示例**（500 个测试示例）的内部数据集上进行了对比，任务是一个包含 **3 个类别**和总计 **26 个类**的层级分类任务（德语）。
- **内部评估集详情**：评估集由大约 **600 个示例**组成，其中 **500 个被指定用于测试**层级分类任务。
   - 该任务涉及 **3 个类别**和总计 **26 个类**，全部以德语进行。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1402548919204053063)** (2 messages): 

> `Stanford Program Synthesis, DS for Vim & Emacs macros` 


- **寻找斯坦福合成器专家**：一名成员询问是否有来自 **Stanford** 且对 **program synthesis** 感兴趣，或者修过相关课程的人。
   - 该用户随后追问：*谁在为复杂的 vim & emacs 宏构建 DS？*
- **对构建 Vim & Emacs 宏的 DS 感兴趣**：一名成员表示有兴趣寻找正在为复杂的 **Vim & Emacs macros** 构建 **DS（推测为数据结构）** 的个人。
   - 这表明其关注点在于通过高级数据结构增强文本编辑器的功能和效率。


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1402384860156530829)** (4 messages): 

> `Public Server Sharing` 


- **Discord 链接分享获准**：一名成员询问是否可以在另一个公开服务器中分享 Discord 链接。
   - 另一名成员确认 *它是公开的* 并鼓励分享。
- **Discord 是公开的**：一名成员很高兴能分享 Discord 链接。
   - 另一名成员表示同意。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1402573553014014046)** (2 messages): 

> `Ninja Tier, AgentX hackathon` 


- **错过截止日期后无法获得 Ninja 等级资格**：一名成员询问在错过 **AgentX hackathon** 的 **Article submission link** 截止日期后，是否还有资格获得 **Ninja tier**。
   - 遗憾的是，另一名成员回复称现在已 *无法* 获得证书。
- **AgentX 黑客松提交问题**：一名参与者意识到由于错过文章提交，他们没有资格获得 **AgentX hackathon** 的 **Ninja tier**。
   - 尽管完成了项目和测验，但缺少文章链接导致无法获得资格，且补交申请被拒绝。


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/)** (1 messages): 

_bryse: 祝贺 North 正式发布 (GA)！
  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1402807076350066880)** (1 messages): 

> `Introductions, Community Welcome` 


- **新成员加入 Cohere 的 Discord**：许多新成员加入了 Cohere 社区 Discord 服务器并进行自我介绍。
   - 成员们分享了他们的 **Company/Industry/University**、正在研究的内容、最喜欢的 **tech/tools** 以及希望从社区中获得什么。
- **欢迎社区新成员**：Cohere 团队发布了一条置顶消息，欢迎新成员加入 Discord 服务器。
   - 该消息包含一个自我介绍模板，帮助新成员分享有关自身及其兴趣的信息。