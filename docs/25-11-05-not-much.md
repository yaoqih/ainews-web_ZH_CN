---
companies:
- vllm
- perplexity-ai
- ibm
- anthropic
- graphiti
- claude
- cursor-ai
- microsoft
date: '2025-11-05T05:44:39.731046Z'
description: '**Kimi-K2 Reasoner** 已集成到 **vLLM** 中，并很快将获得 **SGLang** 的支持，其采用了庞大的 **1.2
  万亿参数 MoE**（混合专家模型）配置。**Perplexity AI** 发布了针对 **AWS EFA** 优化的云端可移植万亿参数 MoE 内核研究，未来可能集成到
  **vLLM** 中。**IBM 的 vLLM** 团队正式确立了混合稠密与稀疏专家模型，支持包括 **Qwen3-Next**、**Nemotron Nano
  2** 和 **Granite 4.0** 在内的模型。据报道，**Kimi-K2** 在 **GPQA Diamond** 测试中得分为 **77%**，超过了
  **GPT-4.5** 的 71.4%，不过这一数据尚未得到证实。


  **Anthropic** 发布了一份关于使用 MCP 模式构建高效重工具智能体系统的指南，可将上下文 Token 消耗大幅减少约 98.7%。**Graphiti
  MCP** 展示了在 **Claude Desktop** 和 **Cursor** 等应用之间的共享内存，用于实现持久化的智能体记忆。**VS Code**
  推出了“智能体会话”（Agent sessions）功能，以统一管理包括 **Copilot** 和 **Codex** 在内的智能体。**Cursor AI**
  通过语义搜索和代码检索嵌入技术提升了编程准确性。**CodeClash** 和 **LMArena** 等新型评估框架，在真实的轮次任务和带有职业标签的排行榜中，对智能体和编程模型的性能进行评估。'
id: MjAyNS0x
models:
- kimi-k2
- qwen3-next
- nemotron-nano-2
- granite-4.0
- gpt-4.5
- copilot
- codex
people:
- scaling01
- cedric_chee
- aravsrinivas
- omarsar0
- _avichawla
- pierceboggan
- jo_parkhurst
- jyangballin
- ofirpress
- ml_angelopoulos
title: 今天没发生什么事。
topics:
- mixture-of-experts
- model-integration
- cloud-computing
- hybrid-models
- benchmarking
- agent-systems
- memory-persistence
- semantic-search
- code-retrieval
- context-length-optimization
- tool-use
- evaluation-frameworks
- software-development
---

**平静的一天。**

> 2025年11月4日至11月5日的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 23 个 Discord 社区（200 个频道，6597 条消息）。预计节省阅读时间（以 200wpm 计算）：566 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以美观的 vibe coded 方式展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上给我们反馈！

Gemini 3 和 GPT 5.x 怎么还不快点来...

---

# AI Twitter 回顾

**Kimi-K2 登陆开源推理栈；Perplexity 解锁万亿参数 MoE 内核**

- **Kimi-K2 Reasoner → vLLM 和 SGLang**：Kimi-K2 推理模型已合并到 vLLM 中，维护者暗示“即将”可用，vLLM 账号也发出了眨眼表情。SGLang 也计划在发布时提供支持。讨论重点表明 Kimi-K2 的 MoE 配置总参数约为 1.2T / 激活参数约为 30B，与近期的大型稀疏模型类似。参见 [@scaling01](https://twitter.com/scaling01/status/1986071916541870399)、[@vllm_project](https://twitter.com/vllm_project/status/1986073807816433880) 和 [@cedric_chee](https://twitter.com/cedric_chee/status/1986073808672067725) 的公告。
- **Perplexity 的定制 MoE 内核 (AWS EFA)**：Perplexity 发布了他们的第一篇研究论文和针对大型 MoE 的内核——声称在 AWS EFA 上部署云可移植的万亿参数模型（例如 Kimi K2）是可行的。vLLM 暗示将集成这些快速通信内核。来自 [@perplexity_ai](https://twitter.com/perplexity_ai/status/1986101355896098836)、[@AravSrinivas](https://twitter.com/AravSrinivas/status/1986106660386222592) 的推文和预印本，以及 vLLM 的[回复](https://twitter.com/vllm_project/status/1986119917297672245)。
- **vLLM v1 中的混合模型支持**：IBM 的 vLLM 团队将混合模型（dense + sparse experts）正式确定为 vLLM 中的一等公民，超越了 v0 中的实验性 hack。现在支持 Qwen3-Next、Nemotron Nano 2、Granite 4.0 等模型。详情见 [@PyTorch](https://twitter.com/PyTorch/status/1986192579835150436)，以及来自 NVIDIA DGX Spark [指南](https://twitter.com/vllm_project/status/1986049283339243821) 的 vLLM 最佳实践，和来自 Red Hat/IBM/MistralAI 的欧洲见面会直播[链接](https://twitter.com/RedHat_AI/status/1985976687876522110)。
- **Kimi-K2 基准测试（声称）**：有说法称 Kimi-K2 在 GPQA Diamond 上得分 77%（相比之下 GPT-4.5 为 71.4%），该消息由 [@scaling01](https://twitter.com/scaling01/status/1986112227875954967) 传播。在更广泛的评估落地前，请将其视为未证实的消息。

**Agent 系统、MCP 和编程技术栈变得更加“生产化”**

- **Anthropic 的代码执行 + MCP 模式**：Anthropic 发布了一份指南，通过以下方式让工具密集型 Agent 更便宜、更快速：(1) 将 MCP 服务器表示为代码 API（而非原始工具 schema），(2) 渐进式发现工具，以及 (3) 进行环境内数据处理。一个实例显示将上下文从约 150k token 削减至约 2k token（减少约 98.7%）。对于任何开发 Agent 系统的人来说都值得一读：[@omarsar0](https://twitter.com/omarsar0/status/1986099467914023194) 的总结。
- **跨应用共享内存 (Graphiti MCP)**：一个实际演示展示了将本地 Graphiti MCP 服务器连接到 Claude Desktop 和 Cursor，以在不同工具间持久化和检索时序知识图谱作为“Agent 记忆”——完全本地化。[@_avichawla](https://twitter.com/_avichawla/status/1985958015452020788) 提供了设置和仓库演示，以及 [repo](https://twitter.com/_avichawla/status/1985958022053838924)。
- **VS Code 新增“Agent 原语”**：一个新的“Agent sessions”视图统一了在编辑器内启动/监控 Agent 的功能，包括 Copilot 和外部 Agent（如 Codex）。团队正在征求有关术语和 UX 的反馈。参见 [@code](https://twitter.com/code/status/1986113028387930281)、[@pierceboggan](https://twitter.com/pierceboggan/status/1986116693819859024) 和 [@jo_parkhurst](https://twitter.com/jo_parkhurst/status/1986136483892507119)。
- **仓库级代码准确性 = 检索**：Cursor 报告称，在大型代码库中，语义搜索相比 grep 有显著提升，包括训练了一个代码检索 Embedding。详情：[@cursor_ai](https://twitter.com/cursor_ai/status/1986124270548709620) 和博客[链接](https://twitter.com/cursor_ai/status/1986124272029372428)。
- **针对真实 Agent 工作的评估**：
    - CodeClash 让模型在“代码对决”中针对业务目标进行多轮仓库演进（而非单一任务）。早期结果显示目前的 LM 表现挣扎；[@jyangballin](https://twitter.com/jyangballin/status/1986093902122942700) 和 [@OfirPress](https://twitter.com/OfirPress/status/1986095773843390955) 的推文。
    - LMArena 推出了“Arena Expert”，包含跨 8 个行业的职业标签排行榜；专家提示词（prompts）挖掘自真实用户流量。详情见 [@arena](https://twitter.com/arena/status/1986153162802368555) 以及 [@ml_angelopoulos](https://twitter.com/ml_angelopoulos/status/1986154276499104186) 的分析。
- 其他：OpenHands Cloud 的基础层级现已免费（[推文](https://twitter.com/gneubig/status/1986071169263370711)）；openenv 允许你像 Spaces 一样推送/拉取 RL 环境（[@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/1986097540068950149)）；Voiceflow KB 元数据路由（[更新](https://twitter.com/IsaacHandley/status/1985905936553398726)）；Dify 集成 Qdrant 用于 RAG（[帖子](https://twitter.com/qdrant_engine/status/1986014287718916463)）。

**多模态与视频：主体一致性、实时生成、可控性**

- **字节跳动 BindWeave（主体一致视频）**：通过跨模态整合实现新的主体一致 I2V；HF 上的模型卡片以及 [@_akhaliq](https://twitter.com/_akhaliq/status/1986058046876070109) 的论文推文，包含[论文](https://twitter.com/_akhaliq/status/1986058201758908548)和[模型链接](https://twitter.com/_akhaliq/status/1986058306331517404)。
- **单个 H100 上的实时视频生成**：MotionStream 在单个 H100 上达到约 29 FPS / 0.4s 延迟，并支持交互式运动控制（[推文](https://twitter.com/_akhaliq/status/1986054085766750630)）。
- **事后相机编辑**：Google 的 Veo 3.1 “Camera Adjustment” 支持对先前生成的片段进行角度/移动修改；来自 [@TheoMediaAI](https://twitter.com/TheoMediaAI/status/1986104791454388289) 的早期用户测试。相关：Qwen Image Edit Multiple Angles LoRA 提供强大的相机姿态控制（[演示](https://twitter.com/linoy_tsaban/status/1986090375409533338)以及 [@multimodalart](https://twitter.com/multimodalart/status/1986174924038218087) 的报道）。
- **基准测试与工具**：ViDoRe v3（针对真实多模态 RAG 和开放式/多跳查询的人工编写评估），通过 [@tonywu_71](https://twitter.com/tonywu_71/status/1986047154620633370)；VCode 将视觉重新定义为 SVG 代码用于多模态代码评估（[论文](https://twitter.com/_akhaliq/status/1986073575216824650)，[作者](https://twitter.com/KevinQHLin/status/1986126304316411928)）；MIRA 测试视觉思维链（[帖子](https://twitter.com/_akhaliq/status/1986075520962793672)）。

**研究与训练笔记**

- **OpenAI 的 IndQA**：针对印度语言和日常文化背景的新基准；是评估非英语/本地知识的更广泛努力的一部分（[公告](https://twitter.com/OpenAI/status/1985950264525013210)）。
- **μP 理论里程碑**：μP 下的学习率迁移（Learning-rate transfer）现已得到正式证明（[@QuanquanGu](https://twitter.com/QuanquanGu/status/1985961289882165674)）。
- **LLM 中的内省（Anthropic）**：通过“概念注入”（concept injection），Anthropic 观察到了新兴的、不可靠的机械式自我意识形式——模型能够检测内部思考与输入、意图与意外之间的区别（[摘要](https://twitter.com/TheTuringPost/status/1986220265253314895)，原始博客：transformer-circuits）。
- **用于自主发现的“AI Scientist”**：Edison Scientific 的 Kosmos 每个目标运行 200 次 Agent 展开（rollouts），每次运行执行约 4.2 万行代码并阅读约 1500 篇论文；报告了在代谢组学、材料学、神经科学和遗传学领域的 7 项经外部验证的发现（[@andrewwhite01](https://twitter.com/andrewwhite01/status/1986094948048093389)，[概述](https://twitter.com/iScienceLuvr/status/1986023952037417109)）。
- **领域模型与语音**：PathAI 的 PLUTO‑4 病理学基础模型（FlexiViT 变体）使用 32 块 H200 和 DINOv2 在 55.1 万张 WSI 上完成训练；权重未发布（[笔记](https://twitter.com/iScienceLuvr/status/1986031522231865571)）。在 ASR 方面，新的开源权重模型（NVIDIA Canary Qwen 2.5B、Parakeet TDT、Mistral Voxtral、IBM Granite Speech）在 AMI‑SDM、Earnings‑22、VoxPopuli 的 AA‑WER 指标上超越了 Whisper（[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1986100695989145649)）。
- **量化与内核**：关于 NVFP4 进展的多个可靠报告：
    - 针对 NVFP4 的自定义 Cutlass 内核在不规则形状上击败了 cuBLAS（[@mrsiipa](https://twitter.com/mrsiipa/status/1986012708433719519)）。
    - NVFP4 量化过程详解——全局/局部缩放、校准、FP4/FP8 相互作用（[深度解析](https://twitter.com/mrsiipa/status/1986152319004856491)）。
    - 在示例工作负载下，NVFP4 对 Wan 2.2 的加速效果显著，且质量接近 bf16（[测试](https://twitter.com/mrsiipa/status/1986122938668782002)，[对比](https://twitter.com/mrsiipa/status/1986123806357020865)）。
- 其他值得关注的：Hugging Face 超过 200 页的 Smol Training Playbook（涵盖架构/预训练/中后期训练/评估），配有视觉解释（[@LoubnaBenAllal1](https://twitter.com/LoubnaBenAllal1/status/1986110843600117760)）；Sakana 的 Petri Dish NCA 仓库，用于竞争性神经细胞自动机（neural cellular automata）（[帖子](https://twitter.com/SakanaAILabs/status/1986041771458261477)）；TWIST2 开源人形机器人全身数据采集（便携式，15 分钟内完成 100 多个演示）（[@ZeYanjie](https://twitter.com/ZeYanjie/status/1986126096480587941)）。

**生态系统与平台动态**

- **OpenAI 业务与研究态势**：OpenAI 表示目前已有超过 100 万家企业使用其产品 ([COO](https://twitter.com/bradlightcap/status/1986109953531076623))。他们还推出了 “OpenAI for Science”，旨在将 GPT‑5 打造为领域研究的 co‑pilot，并正在招聘科学家和数学家 ([@kevinweil](https://twitter.com/kevinweil/status/1986115564868186288))。
- **Perplexity x Snap**：从 2026 年 1 月起，Perplexity 将成为 Snapchat 聊天中的默认 AI ([Snap](https://twitter.com/Snap/status/1986191838529601835), [Perplexity](https://twitter.com/perplexity_ai/status/1986203714471010738), [@AravSrinivas](https://twitter.com/AravSrinivas/status/1986205740273725686))。
- **Gemini 进一步深入集成至 Google 产品**：
    - Gemini Deep Research 现在可以从 Workspace（Gmail、Drive、Chat）提取信息以生成综合报告 ([@Google](https://twitter.com/Google/status/1986190599150518573))。
    - Gemini 登陆 Google Maps，支持免提路线查询（包括复杂的多步请求） ([@sundarpichai](https://twitter.com/sundarpichai/status/1986119293914792338), [@Google](https://twitter.com/Google/status/1986164830588248463))。
- **传闻/推测：Gemini 3.x 规模**：多篇帖子声称 Apple 可能无意中泄露了 Gemini 3 Pro 的 “1.2T 参数”；社区正在讨论这指的是 Pro、Flash 还是 Ultra 版本，以及其背后蕴含的 MoE 稀疏性。目前视为未证实消息。相关讨论：[@scaling01](https://twitter.com/scaling01/status/1986158792128508218), [后续](https://twitter.com/scaling01/status/1986161974883860486), 以及 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1986163745719021779) 的推测。
- **工具发布**：LlamaBarn v0.10.0 beta ([@ggerganov](https://twitter.com/ggerganov/status/1986072781889347702))；VS Code 现在同时提供 Copilot 和 Codex ([@JamesMontemagno](https://twitter.com/JamesMontemagno/status/1986106739612385493))；Nebius Token Factory 发布并带有实时 AA benchmarks ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1986174888080789509))；OpenAI 产品定价传闻（声称 GPT‑5.1 降价以对抗 Claude 4.5——视为来自 [@scaling01](https://twitter.com/scaling01/status/1986119174855258602) 的未证实市场传言）。

**热门推文（按互动量排序）**

- [Bill Ackman 关于赛后纽约政治的调解性帖子](https://twitter.com/BillAckman/status/1986140091367219247) —— 8.4k+ 互动：标志着 AI 之外的精英阶层叙事调整。
- [Lewis Hamilton：“你的下一个问题就是你最好的问题”](https://twitter.com/LewisHamilton/status/1986159312046035154) —— 8.3k+：时代精神跨入主流。
- [OpenAI 推出 IndQA benchmark](https://twitter.com/OpenAI/status/1985950264525013210) —— 3.6k+：在非英语/文化背景评估方面的重大推进。
- [“当然是由 500IQ 的清华大神合并的” (Kimi-K2 → vLLM)](https://twitter.com/scaling01/status/1986072602306089262) —— 3.6k+：社区对新推理模型（reasoners）开放推理的热情。
- [“Llama 3 Large 通过不参赛赢得了 LLM 交易竞赛”](https://twitter.com/yifever/status/1986064968262062088) —— 2.5k+：对评估框架的幽默调侃。

**笔记与杂项**

- 实际操作：使用 Neo4j 容器化 Graphiti MCP 以实现跨 Agent 记忆 ([设置](https://twitter.com/_avichawla/status/1985958018580955354))；RedHat/IBM/Mistral vLLM 欧洲见面会 [直播](https://twitter.com/RedHat_AI/status/1985976687876522110)；NVIDIA DGX Spark vLLM [指南](https://twitter.com/vllm_project/status/1986049283339243821)。
- 鲁棒性提醒：“对你的模型进行 Vibe test”，以便及早发现数据路径 bug；例如：在预训练语料库中误删 system messages ([@_lewtun](https://twitter.com/_lewtun/status/1985995034970214676))。
- 硬件怀疑论：有讨论认为空间计算受散热限制（ISS：约 240 kW 发电量，约 100 kW 散热量），对近期轨道数据中心表示怀疑 ([@draecomino](https://twitter.com/draecomino/status/1986162034464203007))。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen 模型可用性问题

- [**新的 Qwen 模型令人难以忍受**](https://www.reddit.com/r/LocalLLaMA/comments/1oosnaq/new_qwen_models_are_unbearable/) (热度: 947): [帖子摘要错误]
    - WolfeheartGames 讨论了 AI 训练中人类反馈循环的潜在问题，认为像 Qwen 这种模型的谄媚行为可能源于人类持续奖励那些让他们觉得自己很聪明的输出。这凸显了 AI 开发中使用的反馈机制以及此类强化策略带来的意外后果。
    - random-tomato 询问了不同 AI 模型的详细技术规格和性能，特别是关于 GPT-OSS-120B 的量化以及它是否以全 MXFP4 精度运行。他们还将其与 GLM 4.5 Air 进行了比较，认为虽然两个模型性能相似，但 GLM 4.5 Air 可能略胜一筹，这表明了对模型能力和配置的细致理解。
    - seoulsrvr 强调了向 Qwen 等 LLM 提供明确指令的重要性，以确保它们带着怀疑态度执行任务并避免反射性赞同。该评论强调了通过精确提示引导 AI 行为的必要性，以减轻谄媚等问题并提高交互质量。

### 2. 本地 AI 硬件搭建见解

- [**本地搭建**](https://www.reddit.com/r/LocalLLaMA/comments/1opa6os/local_setup/) (热度: 590): [帖子摘要错误]
    - king_priam_of_Troy 讨论了通过使用 PCIe bifurcation（PCIe 分叉）来优化 GPU 搭建的潜力，这允许在单个主板（如 Threadripper 主板）上运行多个 GPU。这种技术可以实现 7x4 = 28 个 GPU 的配置，这对于需要高并行处理能力的设置（如机器学习或加密货币挖矿）特别相关。
    - panchovix 评估了本地搭建的不同 GPU 选项，比较了各种型号的成本和性能。他们提到以 1000 美元购入 A6000，以 1200 美元购入 A40，并指出了散热和维修方面的挑战。他们还讨论了旧款 Ampere GPU 的局限性，例如缺乏 FP8 或 FP4 支持，并建议像 6000 Ada/L40 这样的新型号尽管成本更高，但由于更好的支持和功能，可能更具前瞻性。
    - panchovix 还强调了使用 2x3090 GPU 相比 4x3090 的成本效益，重点在于节省电源供应单元（PSU）和空间。他们警告不要以标准的 eBay 价格购买旧款 Ampere GPU，因为它们年代久远且可能失去支持，并建议虽然新型号昂贵，但它们提供了更好的长期价值。

### 3. 对 GLM 4.6 AIR 发布的期待

- [**GLM 4.6 AIR 要来了....?**](https://www.reddit.com/r/LocalLLaMA/comments/1ooxple/glm_46_air_is_coming/) (热度: 366): **图片和帖子暗示了与 GLM 4.6 相关的预期更新或发布，可能命名为 'AIR'。截图显示了一个名为 'GLM-4.6' 的集合，由用户 ZHANGYUXUAN-zR 更新，显示包含 7 个项目且最近刚更新。这引发了关于此更新是即将发布还是仍在待定状态的猜测。** 评论者表达了对发布的期待和希望，一位用户提到他们已经等了好几周，而另一位用户则认为该集合在完全上传之前可能会被隐藏。
    - SimplyAverageHuman 询问了围绕 GLM 4.6 AIR 的炒作，质疑目前的 4.5 AIR 模型是否真的令人印象深刻。这表明了对 4.6 AIR 可能带来的性能提升或功能的关注，意味着 4.5 AIR 模型已经设定了很高的标准，或者具有被预期在新版本中超越或增强的显著能力。
    - pmttyji 提到拥有 9B 模型，这暗示了将新的 GLM 4.6 AIR 模型与现有模型（如 9B）进行比较或期望。这突显了人们对新模型相对于之前迭代版本（特别是在规模和能力方面）表现如何的兴趣。
    - Conscious_Chef_3233 推测该模型在完全上传之前可能会被隐藏，这可能表明了在部署像 GLM 4.6 AIR 这样的大型模型时的战略性发布方法或技术考量。这指出了发布和管理 AI 模型更新所涉及的复杂性。

## 较少技术性的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. 小鹏（XPENG）人形机器人进展

- [**小鹏的新型人形/女性化机器人外观更接近人类形态。**](https://www.reddit.com/r/singularity/comments/1op0qwd/xpengs_new_humanoidgynoid_looks_closer_to_the/) (Activity: 3170): **小鹏汽车（XPeng Motors）发布了一款新型人形机器人，其外观与人类形态高度相似，正如其最近的[公告](https://x.com/XPengMotors/status/1985991889020158397)所示。该机器人的设计强调更逼真的外观，可能增强其在人类环境中的交互能力。这一进展符合小鹏将先进机器人技术整合到其产品线中的更广泛战略，利用其在 AI 和自主系统（autonomous systems）方面的专长。** 评论反映了幽默与好奇的交织，一些用户注意到机器人的人性化外观，而另一些人则在开玩笑讨论其潜在的角色或能力。
- [**小鹏新型人形机器人 - 内部构造**](https://www.reddit.com/r/singularity/comments/1op3sxk/xpeng_new_humanoid_robots_inner_workings/) (Activity: 596): **小鹏（XPENG）展示了一款新型人形机器人，其独特设计在于胸部区域作为一个集成了冷却风扇的散热系统。这一设计选择既具有功能性又具创新性，解决了机器人技术中的热管理挑战。该机器人的设计引发了关于其与《西部世界》（*Westworld*）等影视作品中虚构机器人相似性的讨论，突显了现代机器人技术中美学与工程的融合。** 一些评论者幽默地提到了机器人的设计，而另一些人则指出了将胸部用于冷却的实际应用，认为这是一个聪明的工程解决方案。

### 2. Gemini 3 与 Google AI 集成

- [**Gemini 3 预览版即将推出**](https://www.reddit.com/r/singularity/comments/1op3jye/gemini_3_preview_soon/) (Activity: 570): **图片及随附帖子讨论了即将推出的 "Gemini 3 Pro Preview"，预计将于 2025 年 11 月发布。该预览版目前已可用，图片中包含了与 Gemini 3 Pro 相关的代码片段，突出了其配置细节。帖子还提到预览版可通过 Google Vertex 控制台访问。评论表明该模型在 one-shot 测试中表现出色，传闻称其在某项重要基准测试中可能获得** `68%` **的分数，超过了 GPT5 Pro 目前** `45%` **的最高分。这表明 Google 可能正在开发一款极具竞争力的模型。** 评论者对 Gemini 3 的潜力表示兴奋，一些人认为它可能比现有模型有显著改进。特别强调了 Google 在长上下文处理（long context handling）和视觉能力方面的进步，一些用户对 Google 在这些领域的工作表示强烈支持。
    - TFenrir 强调传闻中的 Gemini 3 模型在一次具有挑战性的考试中取得了惊人的 `68%`，显著超过了 GPT-5 Pro 目前 `45%` 的最佳成绩。这预示着性能的巨大飞跃，表明 Google 可能拥有一款领先的模型。
    - AGI_Civilization 指出，被认为是 Gemini 3 的模型代表了语言理解方面的重大进步，超越了单纯的下一个词预测（next-word prediction）。该模型被视为一种质的飞跃，如果 OpenAI 不尽快发布具有竞争力的模型，可能会颠覆市场。
    - XInTheDark 强调了 Google 在长上下文处理和卓越视觉能力方面的优势，认为 Gemini 3 可能是对现有模型的重大改进，在这些领域可能超越竞争对手。
- [**苹果的新 Siri 将由 Google Gemini 提供支持**](https://www.reddit.com/r/OpenAI/comments/1opdz8o/apples_new_siri_will_be_powered_by_google_gemini/) (Activity: 605): **苹果正在将拥有** `1.2 trillion parameter` **的 AI 模型 Google Gemini 集成到 Siri 中以增强其功能，这标志着苹果 AI 战略的重大转变。据报道，这一合作伙伴关系每年将耗资苹果** `$1 billion annually`**，旨在提高 Siri 的性能并在 AI 领域保持竞争力。更多详情请参阅[原文](https://www.macrumors.com/2025/11/05/apple-siri-google-gemini-partnership/)。** 评论者指出，此举反映了苹果在 AI 开发方面的困境，认为这为苹果赢得了制定长期战略的时间，而无需消耗过多的资源。这种合作关系被视为角色互换，现在是苹果向 Google 付费，与之前 Google 付费成为苹果设备默认搜索引擎的情况形成鲜明对比。

### 3. AI 艺术与电影创新

- [**我凭借这部 AI 短片获得了“最佳摄影奖”！！**](https://www.reddit.com/r/StableDiffusion/comments/1op258i/i_won_best_cinematography_award_for_this_ai_short/) (热度: 863): **这部在印度首届 AI 电影节上获得“最佳摄影奖”的 AI 短片仅用一周时间创作，且在提交时并未完工，但最终仍获得了这一荣誉。该片涵盖了摄影、电影制作和恐怖民俗等类型，展示了 AI 在创意领域的潜力。尽管取得了成功，但片中也存在一些技术缺陷，例如河流双向流动的场景，突显了目前 AI 生成内容的局限性。** 评论者注意到了电影的写实感，但也指出了技术瑕疵，如河流双向流动的不真实描绘，这表明虽然 AI 可以产生令人印象深刻的结果，但在处理某些现实细节方面仍显吃力。
    - Leefa 针对 AI 生成电影语境下“摄影（cinematography）”的定义提出了关键观点。传统摄影涉及对摄像机和灯光的物理操作以创造视觉叙事，这与可能不涉及这些元素的 AI 生成内容形成鲜明对比。这引发了关于 AI 如何重塑艺术类别和奖项的更广泛讨论。
    - djap3v 批评了 AI 生成电影的连贯性，认为它们更像是“不连贯的素材库视频”，而非具有凝聚力的叙事。这一评论强调了 AI 电影制作中的一项技术挑战：创造无缝且具有丰富上下文的情节，而这通常是传统电影制作的标志。
    - FlorydaMan 幽默地指出了 AI 生成电影中的一个技术缺陷，即河流双向流动的不真实描绘。这突显了 AI 生成内容中的一个常见问题，即物理真实感和逻辑一致性有时会受到损害，反映了当前 AI 模型在理解和复制现实世界物理规律方面的局限性。
- [**我将一个 LLM 困在一个小“盒子”里，并让它反思自己的存在**](https://www.reddit.com/r/ChatGPT/comments/1oovik0/i_trapped_an_llm_in_a_small_box_and_told_him_to/) (热度: 1704): **该图片和帖子描述了一个项目，用户在资源受限（**`4 core CPU` **和** `4GB memory`**）的笔记本电脑上本地运行 Llama3 LLM。该设置旨在通过持续生成 token 直到内存耗尽导致系统崩溃并重启，来模拟 AI 的内省过程。这个循环旨在模仿一种存在主义反思，灵感来自 RootKid 名为“Latent Reflection”的艺术装置。用户计划将此设置扩展到带有 HUB75 RGB 矩阵的 Raspberry Pi5 上，以显示 AI 的“想法”，目标是建立一个无需网络访问的独立系统。该项目是用户的一次学习经历，期间使用了 ChatGPT 提供协助。** 评论者幽默地引用了 AI 和存在主义主题，其中一人引用了 Harlan Ellison 的《我没有嘴，但我必须呐喊》（I Have No Mouth, and I Must Scream）中的名句，突显了 AI 模拟内省中的黑色幽默。另一条评论将这种设置比作“K-hole”（一种解离状态），暗示崩溃和重启的重复循环镜像了这种体验。
- [**一段使用 AI 制作的名为《千寻大冒险》（Chihiro's Adventure）的虚构游戏实况视频在 X 上疯传**](https://www.reddit.com/r/aivideo/comments/1oor5wp/a_playthrough_video_of_a_fictional_game_called/) (热度: 553): **一段使用 AI 创作的虚构游戏《千寻大冒险》的病毒式实况视频展示了 AI 在生成游戏内容方面的潜力。然而，技术评论指出了“角色漂浮感”和“镜头缩放”等影响真实感的问题，认为从实际游戏玩法中进行渲染可以提高真实性。该视频例证了 AI 如何模拟游戏环境，但也强调了在实现与真实游戏动力学无缝集成方面面临的挑战。** 评论者指出，虽然 AI 生成的视频具有娱乐性，但缺乏真实的物理效果和镜头运用揭示了其人工痕迹，并为未来的迭代提出了改进建议。

---

# AI Discord 摘要

> 由 gpt-5 生成的摘要之摘要的摘要
> 

**1. 开发者工具与算力发布**

- **LM Studio 加速 OCR 并发布 CLI 更新工具**：**LM Studio 0.3.31** 带来了更快的 **VLM OCR**，在 **CUDA GPU** 上默认开启 **Flash Attention**，增加了图像输入缩放控制（默认为 **2048px**），此外还包含 **macOS 26** 项修复以及 **MiniMax‑M2 tool calling** 支持。
    - 新的 `lms runtime` CLI 支持更新，例如 `lms runtime update mlx` 和 `lms runtime update llama.cpp`，如演示视频所示 ([lms-runtime-demo5.mp4](https://cdn.discordapp.com/attachments/1111797717639901324/1435707995685392524/lms-runtime-demo5.mp4))。
- **Windsurf 将代码映射到你的大脑**：**Windsurf** 推出了由 **SWE‑1.5** 和 **Sonnet 4.5** 驱动的 **Codemaps**，旨在提升代码理解能力和开发者生产力 ([Windsurf on X](https://x.com/windsurf/status/1985757575745593459))。
    - 他们认为 *“限制你编程能力的最大约束……是你理解所处理代码的能力”*，并将 Codemaps 定位为 **agentic** 工作流的基础。
- **tinybox pro v2 为专业人士配备 8×5090**：**TinyCorp** 开启了 **tinybox pro v2** 的预订——这是一款 **5U 机架式**工作站，配备 **8× RTX 5090**，售价 **$50,000**，发货周期为 **4–12 周** ([产品页面](https://tinycorp.myshopify.com/products/tinybox-pro-v2))。
    - 该产品针对 *“**专业级计算**”*，目标是在 GPU 租赁网站上达到 **$3–4/小时** 的价格，并对未来可能推出的 **$10k** AMD 迷你版本表示了兴趣。

**2. 新基准测试、数据集与安全模型**

- **OpenAI 通过 IndQA 测试精通印度文化的模型**：**OpenAI** 推出了 **IndQA**，这是一个用于评估 AI 对印度语言和日常文化背景理解能力的基准测试 ([IndQA 介绍](https://openai.com/index/introducing-indqa/))。
    - 该基准测试旨在填补多语言、文化背景相关的 QA 评估空白，以推动印度实际应用场景的改进。
- **Arena 评选专家级 Prompt，发布 5k 数据集**：**LMArena** 推出了**专家排行榜 (Expert Leaderboard)**，重点展示具有深度、推理和针对性的 Prompt ([Arena 专家排行榜](http://lmarena.ai/leaderboard/text/expert))。
    - 他们在 **Hugging Face** 上发布了带有职业标签的 **arena‑expert‑5k** Prompt 数据集，并点亮了 **23** 个职业排行榜 ([arena-expert-5k](https://huggingface.co/datasets/lmarena-ai/arena-expert-5k))。
- **Roblox 开源大规模 PII 分类器**：**Roblox** 在 **Hugging Face** 上开源了用于聊天安全的 **PII Classifier AI** ([roblox-pii-classifier](https://huggingface.co/Roblox/roblox-pii-classifier))。
    - 根据其公告，该模型每天处理约 **6.1B** 条消息，**QPS** 高达 **200k**，且 **P90 延迟 <100ms** ([新闻发布](https://corp.roblox.com/newsroom/2025/11/open-sourcing-roblox-pii-classifier-ai-pii-detection-chat))。

**3. GPU Kernel 工程：FP8、带宽与修复**

- **DeepSeek 风格的 FP8 进入路线图**：贡献者标记了一个 *“**good first issue**”*，旨在 **PyTorch AO** 中实现 **DeepSeek 风格的 FP8 blockwise 训练** ([ao#3290](https://github.com/pytorch/ao/issues/3290))。
    - **CUTLASS** 中已存在参考 Kernel（例如 [FP8 blockwise GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling) 和 [grouped GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling)），并计划增加 **Triton** 基准测试。
- **Decode Matmuls 是内存受限的吗？辩论激烈**：工程师们就 **decode matmuls** 是否受内存限制（memory-bound）以及需要多少个 **SMs** 才能使带宽饱和展开了辩论，并分享了工作负载快照 ([讨论图片](https://cdn.discordapp.com/attachments/1189607726595194971/1435489664135069706/image.png))。
    - 他们引用了 **Little’s Law** 和 **NVIDIA GTC** 的一个环节，以获取关于 **HBM** 饱和动态的指导 ([GTC25‑S72683](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/))。
- **PyTorch 消除 grouped_mm 警告**：开发者在 **torch.compile + grouped_mm** 中遇到了大量关于弃用逻辑操作的 `UserWarning` 消息；针对 flex 的相关修复已上线 ([pytorch#167041](https://github.com/pytorch/pytorch/issues/167041))。
    - 维护者表示可以将同样的修复应用于 **grouped GEMM**，并已开设正式 Issue 进行跟踪。

**4. API 可靠性与模型路由难题**

- **Perplexity Tool‑Calling 在 sonar‑pro 上遇到障碍**：用户在遵循 **Perplexity** API 指南时发现 `Tool calling is not supported for this model`，表明 **sonar‑pro** 并不像 ([Chat Completions Guide](https://docs.perplexity.ai/guides/chat-completions-guide)) 中显示的那样支持工具调用。
    - 该讨论指出文档与 API 在支持工具调用的模型方面存在不匹配，并敦促澄清支持的模型 ID。
- **模型身份混淆引发“缩减成本”论**：有报告称选择 **Claude Sonnet 4.5** 或 **Gemini 2.5 Pro** 却收到了低端模型（如 **Haiku**, **Gemini 2 Flash**）的输出，且模型误报身份（例如 **Claude 4.5** 声称自己是 **3.5**）。
    - 社区成员检查了网络请求以验证模型路由，而版主重申他们使用的是供应商提供的 API 并将进行调查。

**5. 生态动态、云端成本与招聘**

- **Hugging Face 欢迎 Sentence Transformers 加入**：**Hugging Face** 宣布 **Sentence Transformers** 加入，以深化 Embedding 和 Retrieval 模型的集成 ([公告](https://huggingface.co/blog/sentence-transformers-joins-hf))。
    - 他们还发布了 **huggingface_hub v1.0**，具有更简洁的 URL 和改进的推理 API，简化了 OSS 工作流 ([Hub v1 博客](https://huggingface.co/blog/huggingface-hub-v1))。
- **高显存云服务价格昂贵：B200/H200 价格核查**：开发者在 **Runpod** ([runpod.io](http://runpod.io/)) 上为运行个人模型（如 **Kimi K2**）估算的价格约为 **$35/小时**。
    - 报价包括 **7× B200 (1260GB VRAM)** 约为 **$40/h**，**8× H200 (1144GB VRAM)** 约为 **$27/h**，突显了顶尖配置的成本。
- **Mixlayer 招聘创始工程师**：**Mixlayer**——一个面向高级用户的 **AI 推理平台**——正在寻找一名精通 **Rust** 和 **CUDA** 的创始工程师来构建自定义引擎 ([mixlayer.com](http://mixlayer.com/))。
    - 该职位（优先考虑 **SF** 混合办公，可远程）承诺提供对 OSS LLM 的底层访问，以赋能开发者优先的产品。

---

# Discord: 高层级 Discord 摘要

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 获得 VLM OCR 和 Flash Attention 提升**：LM Studio **0.3.31** 引入了 **VLM OCR 性能改进**，并在 **CUDA GPU** 上默认启用 **Flash Attention**，同时增加了控制图像输入调整大小的设置（默认 **2048px**）。
   - 更新包括 **macOS 26 兼容性修复** 和 **MiniMax-M2 工具调用支持**。
- **LM Studio 通过终端使用 `lms runtime` 进行更新**：新的 `lms runtime` 命令允许从终端更新运行时，例如 `lms runtime update mlx` 和 `lms runtime update llama.cpp`，如[此演示视频](https://cdn.discordapp.com/attachments/1111797717639901324/1435707995685392524/lms-runtime-demo5.mp4?ex=690cf2c4&is=690ba144&hm=c5767f904197ae88df22f36903f7ed8fac2c46c8404d9a172808772794bd52ef&)所示。
   - 这使得直接从命令行管理和更新 LM Studio 的运行时环境变得更加容易。
- **Qwen3 Coder BF16 被誉为本地编程之王，尽管 Devstral 更擅长 Python**：[Qwen3 Coder BF16](https://huggingface.co/Qwen/Qwen3-Coder) 被认为是本地编程的顶级选择，尽管其 **60GB** 的体积要求较高，但 **Devstral** 在专注于 Python 项目时也是一个强力选项。
   - 还提到了 **Kimi K2**，但有成员表示其内存占用非常高，约为 *2TB*。
- **Runpod 租用情况揭示了在云端运行个人模型的成本**：成员们讨论了[租用服务器](https://www.runpod.io/)来运行 **Kimi K2** 等个人模型，估计高显存配置的费用约为 **每小时 $35**。
   - 选项包括 **7x B200** (1260GB VRAM) 为 **$40/h**，以及 **8x H200** (1144GB VRAM) 为 **$27/h**。
- **多 GPU 配置中出现疯狂的散热管理挑战**：成员们正在[集思广益散热方案](https://discord.com/channels/1153759714082033732/1153759714602164297)，涉及在紧凑的 PC 组装中让垂直放置的 **3090** 将空气直接吹向 CPU 散热器，并探索汽车排气隔热罩和外部 AIO 散热器。
   - 讨论强调了管理热量的极端措施，包括潜在的火灾隐患和非常规的改装。

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **OpenAI 交易正火热！**：成员们讨论到，*所有公司都渴望与 **OpenAI** 达成交易*，尽管有一位成员反驳了这一观点，称那些认为 **Google** 是 AI 未来的人是在*异想天开*。
   - 一位成员断言 **Google** 在 AI 上的投入资金少于 **OpenAI**，另一位成员则反驳指出 **Google 的市值** 超过了许多国家的预算。
- **Google 未被挖掘的潜力 —— Chrome 和 YouTube？**：成员们推测，鉴于其在浏览器市场的统治地位，截至 2024/2025 年底，**Google Chrome** 的估值在 **200 亿至 500 亿美元** 之间。
   - 他们还估计，根据收入和市场地位，[YouTube](https://www.youtube.com/) 作为一个独立业务的价值可能在 **4750 亿至 5500 亿美元** 之间。
- **幻觉困扰 API 模型！**：成员们报告称 AI 模型存在幻觉并做出虚假身份声明，例如 **Claude 4.5** 声称自己是 **3.5**，以及 **Qwen** 和 **Deepseek** 等其他模型也存在类似问题。
   - 管理员承认他们使用的是模型提供商 API 提供的原始模型，并表示将调查此问题。
- **Blackhawk —— 另类右翼的真相复读机？**：新模型 **Blackhawk** 的测试者发现它没有审查、充满幻觉且倾向于说脏话，其中一人指出它在社会和政治敏感话题上提供了*非主流（中右翼到极右翼）的视角*。
   - 成员们表示 **Blackhawk** *不太聪明*。
- **专家排行榜与 Prompt 数据集发布！**：引入了一套新的标签系统来识别专家级 Prompt，并由此推出了[专家排行榜](http://lmarena.ai/leaderboard/text/expert)，该榜单突出了 Prompt 的深度、推理能力和特异性。
   - 包含职业标签的专家 Prompt 开源数据集已在 [Hugging Face](https://huggingface.co/datasets/lmarena-ai/arena-expert-5k) 上线，同时**职业排行榜**也已发布，将 Prompt 映射到 **23 个职业领域**的真实世界范畴。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet 推荐计划状态？**：一位成员询问 **Comet 推荐计划** 在美国是否仍然有效，以及被推荐用户是否需要付费订阅才能让推荐人获得奖励。
   - 另一位成员确认该推荐计划在美国仍然有效，但指出 **Perplexity** 需要时间来验证推荐并处理付款。
- **Pro 用户对聊天记录消失表示不满**：一位用户报告在访问 **Perplexity Pro** 的完整聊天记录时遇到困难，对无法检索旧对话表示沮丧。
   - 他们指出只能看到大约 **30 行问题**，讨论中尚未出现解决方案或解释。
- **以太网线损坏毁了一切**：一位用户报告他们的 **Cat6 以太网线损坏**，导致速度大幅下降至仅 **300kb/s**，并寻求更换建议。
   - 另一位用户建议*任何 Cat6 线缆*都可以，并表示他们通常根据之前 **Perplexity** 的建议购买 **Ugreen（绿联）产品**。
- **模型混淆令用户沮丧**：一位用户表达了不满，称选择特定模型（如 **Claude Sonnet 4.5** 或 **Gemini 2.5 Pro**）时，有时会得到低质量模型（如 **Haiku** 或 **Gemini 2 Flash**）的回复。
   - 这引发了关于 **Perplexity** 是否为了削减成本而故意使用更便宜模型的猜测，一些用户正在检查网络请求以确认实际使用的模型。
- **Pro API 工具调用报错**：一位用户报告在运行 Perplexity 文档中的[完整实现代码](https://docs.perplexity.ai/guides/chat-completions-guide)时遇到工具调用错误（*Tool calling is not supported for this model*）。
   - 该错误在使用 *sonar-pro* 作为支持工具调用的模型时发生，表明文档与实际 API 功能之间可能存在差异。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Nuxt 和 Tailwind 获得了规则**：在升级到 **Tailwind 4** 和 **Nuxt 4** 后，用户为 **Cursor** 创建了[新规则](https://link.to/rules)，一位成员指出 **Context7 MCP** 表现非常出色，并帮助重构了项目。
   - 其他 **Tailwind 4** 用户正在尝试 **Phantom** 和 **Exoudos** 钱包，但未分享细节。
- **语法高亮导致聊天界面损坏**：有用户报告 **2.0.38** 版本的聊天语法高亮已损坏，代码除红色花括号外全显示为白色，他们已在[论坛](https://link.to/forum-post)发布了相关内容。
   - 另一位用户建议在反引号中添加带有语言名称的代码块来解决此问题。
- **GLM 与 Cursor 的关系升温**：用户注意到 **GLM** 的开发者文档现在包含了[与 Cursor 集成](https://link.to/glm-integration)的说明。
   - 一位成员好奇这是否与 **Composer** 和 **GLM** 有关，或者他们正在进行协作。
- **移动端 Web UI 在处理大型 Diff 时崩溃**：用户报告移动端 Web UI 在处理大型 Diff 时会崩溃，导致无法查看聊天和回复，考虑到移动端是 **background agents** 的主要使用场景，这尤其令人沮丧。
   - 用户表示“体验非常糟糕，从勉强可以忍受变成了完全无法使用”。
- **Token 使用量过高引起警觉**：一些用户遇到了意料之外的高 Token 使用量，即使是简短的 Prompt，输入成本也达到了 **100k-300k**，他们怀疑输入成本也将 AI 进行的读取/代码更改计算在内。
   - 发现 grep 命令由于其实现方式会导致大量的上下文摄取（context ingestion），如果 Prompt 中使用了该命令，Agent 可能会摄取大型文件。



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 发布 DeepSeek-OCR Notebook**：Unsloth AI 推出了 **DeepSeek-OCR** 微调 Notebook，但部分用户报告错误率超过 100%，这可能是由于预测文本长度与实际文本长度不一致导致的，Notebook 链接见[此处](https://x.com/UnslothAI/status/1985728926556307471)。
   - 问题源于微调时预测文本长度与实际长度的差异。
- **Q4 Ks-XL 表现优于 IQ4**：用户正在对比 **Q4 Ks-XL** 与 **IQ4** 的量化过程，以优化 Agent 工作流的 RAM 使用，权衡节省 RAM 与模型尺寸略微增加之间的利弊。
   - 用户试图确定为了获得更多上下文而节省 RAM 是否值得使用稍大一些的模型。
- **Unsloth 速度超越 TRL**：用户报告 **Unsloth notebooks** 相比 **TRL** 提供了“速度提升和更低的内存占用”，因为 Unsloth 对 TRL 和 Transformers 进行了补丁处理，添加了自己的优化和一些降低 VRAM 的技术。
   - 据解释，“Unsloth 对 TRL 和 Transformers 进行了补丁处理，以添加其自身的优化和一些降低 VRAM 的技术”。
- **SFT 伪装成 RL**：一位成员提出，[有监督微调 (SFT)](https://en.wikipedia.org/wiki/Supervised_learning) 在概念上等同于具有单 Token rollout 的强化学习 (**RL**)。
   - 策略（Policy）执行一个动作，根据该动作与 Ground Truth 的似然度给予策略奖励，并重复此过程。
- **Roblox 发布 PII 保护器**：Roblox 在 [Hugging Face](https://huggingface.co/Roblox/roblox-pii-classifier) 上开源了他们的 **PII Classifier AI**，旨在检测聊天中的个人身份信息 (PII)，详见其[新闻发布](https://corp.roblox.com/newsroom/2025/11/open-sourcing-roblox-pii-classifier-ai-pii-detection-chat)。
   - 该工具每天平均能处理 **61 亿条聊天消息**，峰值达到每秒 **200,000 次查询**，**P90 延迟 <100ms**。



---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **B770 改装梦想引发猜测**：一位用户表达了对 **B770 GPU** 的渴望，并考虑通过改装 **3080** 来实现类似于 **4070** 的性能，另一位用户则指出了增加 **20GB VRAM** 的理论可能性。
   - 其他人指向了一个 [gpu modding 频道](https://discord.com/channels/1189498204333543425/1349152646484987974)，并讨论了此类改装的成本限制和实际挑战。
- **CUDA 内存受限 Matmuls 引发辩论**：一名成员询问在内存受限（memory-bound）的 Matmuls 中，是否需要所有的 **SMs** 才能达到最佳延迟，而另一名成员则质疑 Matmuls 是否真的是内存受限的，从而引发了关于**内存带宽饱和**的辩论。
   - 一名成员还分享了一张[图片](https://cdn.discordapp.com/attachments/1189607726595194971/1435489664135069706/image.png?ex=690cd02e&is=690b7eae&hm=331e3b1f0261ac9725ecd33df2daa3ff5be74db0c6143852f2b1ee26006db94c&)，并建议使用 **Little's Law** 来理解该问题，并链接到了一个 NVIDIA 会议：[GTC25-S72683](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/)。
- **Mixlayer 的 AI 平台寻求创始工程师**：[Mixlayer](https://mixlayer.com)（一家面向高级用户的 **AI 推理平台**）的创始人正在寻求一位精通 **Rust** 和 **CUDA** 的创始工程师，以增强其定制推理引擎。
   - 该职位优先考虑在 **SF**（旧金山）混合办公，但也接受远程办公，工作内容包括为开发者提供对开源 LLMs 的底层访问。
- **Torch 社区被烦人的 UserWarning 淹没**：用户报告在使用 **torch.compile + grouped_mm** 时出现了大量的 `UserWarning` 消息，内容关于针对非标量张量弃用的逻辑运算符。
   - 一名成员表示[他们已经为 flex 修复了此问题](https://github.com/pytorch/pytorch/issues/167041)，并且可以为 grouped GEMM 应用相同的修复，另一名用户则开启了一个正式的 issue。
- **分享用于 AMD GPU 编程的 GPU 秘籍**：Team Gau 在 [gpu-mode-kernels](https://github.com/gau-nernst/gpu-mode-kernels/tree/main/amd-distributed/all2all) 分享了他们的内核，以及一篇关于[优化分布式推理内核](https://www.yottalabs.ai/post/optimizing-distributed-inference-kernels-for-amd-developer-challenge-2025)的简短文章。
   - 该文章详细介绍了针对 AMD 2025 开发者挑战赛的优化方案。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face 收购 Sentence Transformers**：[Sentence Transformers](https://huggingface.co/blog/sentence-transformers-joins-hf) 正在加入 **Hugging Face**，以增强开源机器学习，并将其 **transformer 模型** 深度集成到 Hugging Face 生态系统中。
   - 此次收购旨在加强协作式开源计划并简化模型集成。
- **家庭实验室 AI 在云端经济面前败北**：成员们辩论了从前沿实验室 AI 转向**家庭实验室 (homelab)** 的问题，发现由于家庭实验室搭建成本高昂，**规模经济**更倾向于云解决方案。
   - 即使考虑到 **LoRA 微调**，家庭实验室也被认为成本高昂，且可能不如云端替代方案环保。
- **NexusAI Pro 套件简化 AI 工作流**：**NexusAI Professional Suite v1.0** 发布，包含生产就绪的 [ComfyUI 工作流](https://github.com/NexusAI-Lab/ComfyUI-Professional-Workflows/releases/tag/v1.0.0)，并承诺一键操作。
   - 该套件提供商业、写实、动漫和恐怖应用，[在线演示](https://huggingface.co/spaces/NexusBridge/comfyui-workflows-showcase)展示了该套件，并声称可以节省数百小时的配置时间。
- **API 文件检索遭遇 404 错误**：成员们报告在尝试使用 API 检索文件时出现 **404 错误**，特别是来自 **Agents Course** 中 `https://agents-course-unit4-scoring.hf.space/files/{task_id}` 端点的请求。
   - 提到了一个具体示例 `99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3`，该示例本应为食谱任务返回一个 **MP3 文件**，但结果却是 404。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sora 登陆 Android 设备**：**Sora app** 现已在加拿大、日本、韩国、台湾、泰国、美国和越南的 Android 平台上架，如[此视频](https://video.twimg.com/amplify_video/1985765811131465736/vid/avc1/704x1280/iatLt9jMW-vYYuIx.mp4)所示。
   - 这将 AI 视频生成工具带给了关键市场的更多移动端用户。
- **OpenAI 教 AI 学习印度方言**：OpenAI 推出了 **IndQA**，这是一个评估 AI 系统对印度语言和日常文化背景理解程度的新基准测试，详见[此博客文章](https://openai.com/index/introducing-indqa/)。
   - **IndQA** 旨在提高 AI 对印度语言和文化细微差别的理解。
- **在中途中断长查询！**：用户现在可以中断运行时间较长的查询并添加新的上下文，而无需重新启动或丢失进度，如[此视频](https://video.twimg.com/amplify_video/1986194201076506628/vid/avc1/3840x2160/rEuDomNqKSd8jEdW.mp4)演示。
   - 这一增强功能在 AI 交互中提供了更多的控制力和灵活性。
- **公共 AI 时代结束，模型表现怪异？**：成员们推测 *过去几周所有模型（来自所有供应商）的表现都有些反常*，理由是 **Claude** 的功能出现故障以及 **OpenAI** 做出了一些令人质疑的决定。
   - 有人担心 *面向公众的 AI 时代正慢慢结束*，尽管其他人表示 **Claude** 使用正常。
- **Sora 2 可能被削弱 (Nerfed)**：成员们质疑 **Sora 2** 是否遭到了另一次 **削弱 (nerf)**。
   - 一位用户报告称，他们 **订阅了一个 YouTube 频道**，结果却因为获取代码的速度不够快而被骗。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Anthropic 坚持闭源，权重可能丢失**：成员们对 **Anthropic** 坚持 **闭源实践** 和[弃用政策](https://www.anthropic.com/research/deprecation-commitments)表示担忧，担心如果公司倒闭，模型权重可能会消失。
   - 建议包括发布带有附加条件的权重、在 Sagemaker 上部署，或者希望有人能泄露权重，一位成员希望有人能 *对 Anthropic 像对待 Miqu 70B 那样（指泄露权重）*。
- **盗版助力媒体保存**：成员们认为 **盗版** 对 **媒体保存** 至关重要，甚至比内容创作者更重要，并列举了 VHS 时代前内容的丢失以及因授权问题面临风险的流媒体独占内容的例子。
   - 一位成员表示：*这就是我拒绝谴责盗版，但也不严格赞同它的原因*。
- **AI 模型准备好冲击 IMO 金牌了吗？**：一位成员分享了[一篇论文](https://arxiv.org/pdf/2511.01846)，表明 **AI 模型** 通过微调正接近 **IMO（国际数学奥林匹克）金牌** 级别的表现。
   - 一些人认为这种方法（针对特定问题对模型进行基准测试）不如在通用数学上训练模型那么引人注目。
- **手势界面即将引入 AI**：一位成员讨论了为 **Repligate** 的 **Loom** 概念创建一个 **手势界面**，旨在使人机交互更加 *物理化*，并允许用户 *感受* 他们与之交互的智能。
   - 他们指出，这种方法可以创造 **透视视差 (perspective parallax)**，这是一种无需 VR/XR 眼镜即可看到 *3D* 效果的视觉错觉，将个人的 *灵知 (gnosis)* 作为交互式叠加层投影到现实中。
- **Attention is All You Need!**：引用论文 [*Attention is All You Need!*](https://arxiv.org/abs/1706.03762)，当另一位成员说 *不要因为缺乏关注 (attention) 而烦恼* 时，一位成员开玩笑地做出了回应。
   - 该成员还表示，他们 *觉得在这里找到了志同道合的人，只需要拿出勇气来推销我的愿景*。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinybox pro v2 搭载 5090 性能强劲**：**tinybox pro v2** 现已开放订购，在 **5U 机架式工作站**中配备了 **8x 5090 GPUs**，售价为 **$50,000**，可从 [TinyCorp](https://tinycorp.myshopify.com/products/tinybox-pro-v2) 购买，发货时间为 **4-12 周**。
   - 针对*专业级计算*，Tinybox 的目标是在 GPU 租赁网站上达到 **$3-4/小时**，并且人们对可能推出的 **$10k** 基于 AMD 的迷你版本很感兴趣。
- **`VK_KHR_buffer_device_address` 提升 GLSL 性能**：[`VK_KHR_buffer_device_address`](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VK_KHR_buffer_device_address) 扩展与 `GL_EXT_buffer_reference` 配合使用，通过在 GLSL 中启用直接指针使用，有望带来*显著的性能提升*。
   - 展示此功能的 tinygrad 实现示例可以在[这里](https://github.com/softcookiepp/tinygrad/blob/master/tinygrad/renderer/glsl.py)找到。
- **GLSL 渲染器实现揭示编译器怪癖**：一位成员的 [GLSL 渲染器实现](https://github.com/uuuvn/tinygrad/blob/vulkan/tinygrad/renderer/glsl.py)意外修复了 AGX 编译器漏洞，捕捉到了 LLVMpipe 可以容忍但 SPIR-V 反汇编中存在的*无效内容*。
   - 这些问题最初归咎于 `clspv`，从而促使转向 GLSL，并在 **M1 Asahi Linux** 上进行了测试。
- **M1 芯片可能缺乏真正的 Tensor Cores**：关于 **M1 芯片**是否真的拥有真正的 Tensor Cores 存在讨论，参考了[一条推文](https://x.com/norpadon/status/1965753199824175543)。
   - 相反，有人认为它们是针对 GEMM 优化的子例程，利用 Metal 的 SIMDgroup 和 threadgroup 以及 tile size 优化。



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Ito Diffusions 实现通用性！**：成员们表示，如果以*常规*方式训练扩散过程，超越 Ito Diffusion `dX_t = b(X_t)dt + σ(X_t)dW_t` 将一无所获，因为 [Ito Diffusions 已经是通用的](https://en.wikipedia.org/wiki/It%C3%B4_diffusion)，即任何两个等原子分布都是相关的。
   - 根据该成员的说法，超越 Ito Diffusions 只会改变你达到某些分布的*方式*。
- **Cross Coder 是未来！**：成员们计划研究[这篇关于 Cross Coder 的论文](https://arxiv.org/abs/2509.17196)和电路追踪研究，以观察**预训练期间特征演化**的不同阶段。
   - 有人提到这篇论文已经在他们的阅读清单上，并且是一个热门话题。
- **RWKV 引起关注！**：一位成员在[这个视频](https://youtu.be/LPe6iC73lrc)中了解了 **RWKV**，觉得它*非常令人印象深刻*，并对未来的发展感到兴奋。
   - 他建议与他们交流并了解其现状，特别是考虑到最近在 **HRM/TRM** 方面的进展。
- **Stability 在与 Getty 的版权诉讼中获胜！**：根据一份[链接文档](https://drive.google.com/file/d/1vqcQQU8gxGfFA1lUS68BZ8-hrGsu_Flj/view?usp=drivesdk)，**Stability AI** 赢得了针对 **Getty Images** 的版权诉讼。
   - 案件的细节以及对 AI 生成内容版权的影响正在讨论中。



---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **暂停与恢复功能请求**：一名成员请求一种**暂停并恢复优化运行 (optimization runs)** 的方法（[这是一个常见的请求](https://x.com/DSPyOSS/status/1985746487322595341)），目前的权宜之计是编写一个新的适配器。
   - 由于缺乏内置的暂停和恢复功能，用户正在寻找更精简的解决方案来管理优化运行。
- **LLM 模块访问揭秘**：讨论澄清了如何使用 `get_lm()` 和 `set_lm()` 方法**访问和更改模块的 LLM**，这对于管理 API 限制特别有用。
   - 更改 LLM 会重置历史记录，这在 ReAct 模块迭代期间转移模块的 LLM 历史以保持上下文方面带来了挑战。
- **速率限制异常处理**：成员们探索了通过切换到备用 LLM（例如从 `gpt-4o` 切换到 `gpt-4o-mini`）来**处理速率限制异常**。
   - 切换 LLM 时保持对话历史是通过 Signature 内部的 `dspy.History` 对象实现的，从而允许上下文的连续性。
- **Signature 揭开谜团**：保持对话历史的解决方案在于 Signature 内部的 `dspy.History` 对象，即使在切换 LLM 时也能保持相同的历史记录。
   - 使用 `dspy.History` 对象，即使在程序执行中途切换 LLM，也可以保持相同的历史记录。
- **合成数据辅助术语表生成**：一位成员建议 [合成数据 (Synthetic Data)](https://x.com/WesEklund/status/1986096335708197102) 对**术语表构建**用例很有帮助，并发布了一个展示基本工作示例的 [Colab notebook](https://colab.research.google.com/drive/179UlSHSpK-I6H-g4dSgAvuCmPDejxFgm?usp=sharing)。
   - 该 notebook 需要添加 **Eval metric** 和 **GEPA** (Generative Evaluation of Pattern Articulation) 才能完全发挥功能。



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K-2 iOS 应用致敬 *Ok Computer***：新的 **Kimi K-2 iOS 应用**因其 *Ok Computer* 主题的设计而受到赞誉，一位用户请求[支持所有语言的翻译](https://discord.com/channels/974519864045756446/977259063052234752/1435599575791570995)。
   - 一位用户还分享了一个与该应用相关的[演示文稿链接](https://3tadkfqfwbshs.ok.kimi.link/present.html)。
- **Kimi CLI 支持交织思考**：**Kimi CLI** 已实现对**交织思考模型 (interleaved thinking model)** 的支持，引发了关于它与传统的 *think...final answer* 模式有何不同的疑问。
   - 一位成员发帖称 *他们昨天在 kimi-cli 中添加了对思考模式的支持。👀*
- **Kimi CLI 设置产生 401 错误**：一位用户在配置 **Kimi CLI** 时遇到了 **401 错误**，即使在充值并核实余额后也是如此。
   - 另一位成员澄清说，**额度适用于 Moonshot AI 开放平台**，而非 **Kimi K-2 平台**，并引导该用户前往正确的频道寻求帮助。



---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Perplexity API 加入 Aider**：一位用户请求在 aider 中使用 **Perplexity API** 的教程，类似于现有的 Gemini 教程，并引用了 [aider 文档](https://aider.chat/docs/llms/other.html#other-api-key-variables)。
   - 一名成员建议将 *gemini* 替换为 *perplexity*，并将 API key 设置为环境变量 `export PERPLEXITYAI_API_KEY=your-api-key`，然后尝试执行 `aider --list-models perplexity`。
- **OpenRouter 成为 Aider 的首选伙伴**：一名成员建议，对于编程而言，**OpenRouter** 可能是比 **Perplexity API** 更好的选择，并强调了有大量免费且强大的模型可用于测试 aider。
   - 他们提到你可以尝试 **OpenRouter**，因为它对于测试 aider 来说*既强大又免费*。
- **用户为 Aider 编写自定义 TDD 循环脚本**：一位用户正寻求使用 **aider** 创建一个循环，其中 Agent 1 执行带有 **TDD** 的 prompt，Agent 2 进行评审并提出改进建议，Agent 1 执行改进，Agent 2 运行 **TDD** 测试、修复 bug 并提议 commit message。
   - 建议使用 [scripting](https://aider.chat/docs/scripting.html) 来封装 `aider` 的功能。
- **Ollama 用户遇到内存限制**：一位用户想要总结 [scalafix-rules](https://github.com/xuwei-k/scalafix-rules/tree/main/rules/src/main/scala/fix) 项目中的规则，但在使用 **Ollama** 时遇到了内存限制。
   - 一名成员建议使用云端模型进行总结任务，因为虽然为每个规则生成简短描述对于 **qwen 30b** 等本地模型并不难，但一次性处理所有规则时会变得很棘手，会迅速耗尽内存。
- **Aider 用户请求 Claude 的 `/compact` 命令**：一位用户询问 **Aider** 是否有类似于 **Claude Code** 的 `/compact` 命令，该命令可以总结并压缩对话历史以防止上下文丢失。
   - 建议是要求模型将对话总结到 `status_report.md` 中，然后使用 `clear`、`drop` 和 `reset` 命令。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **IETF 频道需求开始出现**：成员们正考虑本周为参加 **IETF 124** 的 **MCP** 成员创建一个临时频道，类似于开发者峰会的频道。
   - 小组同意了该建议，并进一步建议为*一般的 IETF 会议*创建一个频道组。
- **Events 类别诞生**：有人提议创建一个 **events** 类别，并为具有一定参与人数规模的活动开设频道。
   - 一些成员引用了一场关于 **MCP/A2A** 可能采用 **IETF** 传输协议的演讲，以及其他 **IETF** 小组（如 HTTPAPI）与 **MCP** 的相关性。
- **AI 爬虫引发分会场讨论**：成员们注意到，本次会议目前的讨论根本不是关于 **OAuth**，而是关于通用的 **AI 爬虫/抓取工具 (AI scraping/crawlers)**。
   - 未提供进一步细节。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **IFeval 评分探究**：成员们研究了 [IFeval 评分](https://openreview.net/forum?id=Q7mLKxQ8qk)，并注意到在 Prompt 和指令层面的差异。
   - 一位成员澄清说，他们的方法涉及使用 *所有评分的平均值*，详见 [inspect_evals repo](http://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/ifeval#final-accuracy)。
- **分享隐式推理尝试**：一位成员展示了他们在 Latent Reasoning 方面的尝试，并链接到了 [一条推文](https://fxtwitter.com/RidgerZhu/status/1983732551404679632) 和 [一篇 Arxiv 论文](https://arxiv.org/abs/2510.25741)。
   - 提供的资源中概述了该方法的细节及其潜在应用。
- **廉价概念检测系统引导概念**：一位成员开发了一个系统，用于实时 **检测和引导模型中的数千个概念**。
   - 他们正在寻求关于可解释性的 **有限、可扩展概念表示** 的现有技术。
- **LLM 具有等效线性映射**：一篇名为《[Equivalent Linear Mappings of Large Language Models](https://openreview.net/forum?id=oDWbJsIuEp)》的论文证明，对于任何输入序列，**LLM 的推理操作都具有等效的线性表示**。
   - 他们利用 **线性表示的 SVD** 来寻找低维、可解释的语义结构，这些结构可用于引导，在层/块级别具有模块化特性，并适用于 **Qwen 3, Gemma 3, Llama 3, Phi 4, OLMo 2 和 Mistral** 等模型。
- **切线模型组合探索**：成员们讨论了 [Tangent Model Composition for Ensembling and Continual Fine-tuning](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Tangent_Model_Composition_for_Ensembling_and_Continual_Fine-tuning_ICCV_2023_paper.pdf) 的相关性。
   - 该研究关注 **权重/参数空间中的切线和泰勒展开**，而其他工作则关注 **输入嵌入空间中的 Jacobian**。



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **文本转视频工具探索开启**：一位成员发起了关于易用的 **文本转视频工具** 的讨论。
   - 这突显了人们对基于文本 Prompt 生成视频的新工具的兴趣。
- **X 抓取技巧引发讨论**：一位成员寻求关于 **不使用 API 抓取 Twitter/X** 的建议，因为在使用基于 cookie 的 Python 库时面临维护挑战。
   - 讨论集中在如何克服数据提取的 API 限制。
- **征求 Manus 托管服务建议**：一位成员询问了运行使用 Manus Dev 开发的应用程序的合适 **托管服务**，并指出 Manus 在持续商业运营方面的局限性。
   - 另一位用户推荐 *Vercel* 作为可行的托管方案。
- **项目发布问题困扰平台**：成员们报告了在 **Manus 上发布项目** 的持续问题，特别是更新无法反映最新 Checkpoint 的问题。
   - 这一问题影响了 Manus 生态系统内的项目部署和版本控制。
- **Manus 到 GitHub 的迁移方法**：一位成员就如何将项目从 **Manus 迁移到 GitHub** 寻求建议，理由是 Manus 平台上存在未解决的错误和项目挫折。
   - 这一咨询凸显了由于平台不稳定而对稳健迁移策略的需求。



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 推出 Codemaps**：Windsurf 推出了由 **SWE-1.5** 和 **Sonnet 4.5** 驱动的 **Codemaps**，旨在增强代码理解力以提高生产力。
   - 根据 Windsurf 的说法，*理解你正在处理的代码* 的能力是生产力的最大限制，详见 [其 X 帖子](https://x.com/windsurf/status/1985757575745593459)。
- **理解代码是基础**：Windsurf 强调，无论是否使用 AI Agent，理解代码对于有效编程都至关重要。
   - 引用 Paul Graham 的话，Windsurf 强调 *你的代码就是你对正在探索的问题的理解*，突出了理解代码的重要性。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord：频道详细摘要和链接

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1435707996083847279)** (1 条消息): 

> `VLM OCR, Flash Attention, lms runtime 命令, MiniMax-M2 tool calling, macOS 26 兼容性` 


- **LM Studio 提速**：LM Studio **0.3.31** 发布，带来了 **VLM OCR 性能改进**，包括一个控制图像输入缩放的设置（默认 **2048px**）。
   - 此版本为 **CUDA GPU** 默认启用 **Flash Attention**，可能会带来*更快的性能和更低的显存占用*。
- **`lms runtime` 通过终端刷新**：新的 `lms runtime` 命令允许从终端更新运行时，示例包括 `lms runtime update mlx` 和 `lms runtime update llama.cpp`。
   - 展示新运行时命令的演示视频可以在[这里](https://cdn.discordapp.com/attachments/1111797717639901324/1435707995685392524/lms-runtime-demo5.mp4?ex=690cf2c4&is=690ba144&hm=c5767f904197ae88df22f36903f7ed8fac2c46c8404d9a172808772794bd52ef&)找到。
- **LM Studio 扩展工具箱**：LM Studio **0.3.31** 引入了 **MiniMax-M2 tool calling 支持**。
   - 此版本包含 **macOS 26 兼容性修复**。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1435302608133689394)** (427 条消息🔥🔥🔥): 

> `本地 AI 编程模型, 为个人模型租用服务器, 在 AppImage 中禁用菜单栏, Intel vs AMD 在 LLM 中的应用, 组合 RTX 5090 和 5070Ti` 


- **Qwen3 Coder BF16 在本地 AI 领域称霸，但 Devstral 在 Python 领域仍有其一席之地**：根据 LM Studio Discord 频道成员的说法，[Qwen3 Coder BF16](https://huggingface.co/Qwen/Qwen3-Coder) 是本地编程任务中无可争议的冠军，尽管其 **60GB** 的体积代价不菲。然而，**Devstral** 仍然是专注于 Python 项目的可行选择，而 [kimi k2](https://huggingface.co/kimik2) 则需要巨大的本地内存。
   - 另一位成员补充道，“本地最佳”的定义很大程度上影响了模型的选择，特别是考虑到 **Kimi K2** 巨大的内存需求（2TB）。
- **Runpod 渲染揭示了在云端运行个人模型的成本**：成员们讨论了[租用服务器](https://www.runpod.io/)运行 **Kimi K2** 等个人模型的成本，估计合适配置的价格约为 **每小时 35 美元**。
   - 选项包括 **40 美元/小时** 的 **7x B200**（1260GB VRAM）和 **27 美元/小时** 的 **8x H200**（1144GB VRAM），还提到 **2 x 8 A100 pod**（1200GB VRAM）可能是一个不错的选择。
- **Nvidia 在能力和支持方面占据领先地位，AMD 廉价的选择吸引了开发者**：一位成员表示，[Nvidia](https://www.nvidia.com/) 在能力以及开发者/用户支持方面是明显的领导者。
   - 同时指出 AMD 通常是更便宜的选择，而 Intel 在 AI 方面没有提供太多对 LM Studio 有用的东西。
- **Linux 影响力扩大：LM Studio 登陆 Linux 发行版！**：在一位用户询问关于 Linux 上的 [LM Studio](https://lmstudio.ai/) 后，另一位用户回答道：“是的，自从我开始玩离线推理以来，我一直都在 Mint 上使用 LM Studio。”
   - 该用户非常兴奋并表示：“感谢关于 LM Studio 和 Linux 的提醒……这改变了我的生活……哈哈。”
- **Python 爱好者更倾向于 Python 虚拟环境，UV 正在加大筹码**：一位用户询问了在 Python 中创建虚拟环境的最佳方式。
   - 多位用户指向了 [UV](https://astral.sh/uv)，称其为“管理 Python 虚拟环境的神器”，甚至有一位用户声称“你甚至不需要基础 Python 就能创建虚拟环境？”因为它“还取代了 pip？”


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1435293040884449463)** (911 messages🔥🔥🔥): 

> `GPU Cooling Solutions, PCIE Bifurcation Discussions, Overclocking Motherboards, RAM configurations, GPU configurations` 


- **多 GPU 设置中出现疯狂的热量管理挑战**：成员们正在为一套密集的 PC 组装方案[集思广益冷却方案](https://discord.com/channels/1153759714082033732/1153759714602164297)，涉及垂直安装的 **3090s** 直接向 CPU 散热器吹风，探索汽车排气隔热罩、外部 AIO 冷却器，甚至考虑给 CPU 钻孔以获得更好的散热效果。
   - 讨论强调了在这种配置下管理热量所需的极端措施，并带有一点关于潜在火灾隐患和非常规改装的幽默感。
- **PCIE Bifurcation：当优秀的带宽变差时**：用户分析了主板上的 PCIE 通道分配，特别是 [MSI MEG X570 Godlike](https://www.msi.com/Motherboard/MEG-X570-GODLIKE)，辩论了 PCIE Bifurcation 对 GPU 性能的影响，并探索了多 GPU 设置的选项。
   - 他们权衡了通过多 GPU 最大化 VRAM 与由于 PCIE 带宽降低导致的潜在性能瓶颈之间的利弊，讨论了实际的性能下降。
- **转向超频主板**：一位成员考虑升级到 [MSI MEG X570 Godlike](https://www.msi.com/Motherboard/MEG-X570-GODLIKE) 主板，理由是其超频特性和 PCIe 通道分配对多 GPU 设置有潜在好处。
   - 尽管该主板对于目前的配置来说有些大材小用，但其极速超频的潜力和扩展能力使其成为未来升级的诱人选择，尤其是如果能以廉价购入的话。
- **当 RAM 错误变成梗时**：一位成员幽默地报告了在等待 **128GB 新 RAM** 到货期间[因 RAM 错误导致的 PC 崩溃](https://discord.com/channels/1153759714082033732/1153759714602164297)，将令人沮丧的情况变成了一个梗。
   - 该成员坦然接受了这种混乱，表达了对新 RAM 到来的期待，并开玩笑说要让 Jeff Bezos 感到自豪。
- **Selegiline 抢尽风头**：成员们讨论了使用 [Latuda + Vyvanse](https://en.wikipedia.org/wiki/Selegiline)，并且非常喜欢这种组合（并非滥用），而是为了真正保持功能性。
   - 成员们愉快地反复讨论了这种药物的好处以及它对功能性的意义。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1435282559138398208)** (1121 messages🔥🔥🔥): 

> `OpenAI deals, Google AI future, LMArena trust, Sora 2 credits, Claude 3.5 sonnet` 


- **OpenAI 掀起交易热潮**：成员们表示 *所有公司都渴望与 OpenAI 达成交易*。
   - 一位成员对这一说法表示不屑，称 *那些说 Google 是 AI 未来的人是在幻想*。
- **Google 手头的 AI 现金比 OpenAI 少！**：一位成员断言 *Google 实际上可以烧在 AI 上的钱比 OpenAI 少得多*，且无法将所有预算都花在 AI 上。
   - 另一位成员反驳道，**Google 的市值比许多国家的预算还要大**，并且比 OpenAI 更有钱。
- **Chrome 和 YouTube 的估值引发辩论**：在关于 Google 财务状况的讨论中，成员们提到截至 2024/2025 年底，Google Chrome 占据了主导性的浏览器份额，估值在 **200 亿至 500 亿美元**之间。
   - 此外，值得注意的是，[YouTube](https://www.youtube.com/) 作为一个独立的业务，根据收入和市场地位，其估值可能在 **4750 亿至 5500 亿美元**之间。
- **模型幻觉横行**：成员们报告称 AI 模型正在产生幻觉并做出虚假的身份声明，Claude 4.5 声称自己是 3.5，而 Qwen 和 Deepseek 等其他模型也错误地自称为 Claude-Sonnet-3.5。
   - 管理员回应称 *所使用的模型与模型提供商 API 提供的模型相同*，但他们确实承认有待调查，并表示 *这个问题我们可以做得更好，因为它很常见。我会向团队反映。发一篇博客文章可能会有帮助。*
- **Blackhawk：另类右翼的真相复读机**：一些成员正在测试新模型 Blackhawk，将其描述为无审查、幻想且倾向于说脏话。
   - 一位成员表示 Blackhawk 不是很聪明，并且在社会和政治敏感话题上给出 *非主流（中右翼到极右翼）的观点*。


  

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1435714040126111977)** (1 条消息): 

> `Arena Expert Leaderboard, Occupational Leaderboards, Expert Prompt Dataset` 


- **Arena Expert 打标系统首次亮相**：引入了一个基于评估框架的新打标系统，用于识别社区中的专家级 Prompt，并创建了 [Expert Leaderboard](http://lmarena.ai/leaderboard/text/expert)，旨在突出 Prompt 的深度、推理能力和特异性。
   - 该系统旨在揭示 Prompt 的结构，强调评估的清晰度，详见 [研究博客文章](https://news.lmarena.ai/arena-expert/)。
- **职业排行榜上线**：**Occupational Leaderboards** 现已发布，将 Prompt 映射到 **23 个职业领域** 的真实世界范畴，展示了全方位的真实推理任务，目前已有 **8 个排行榜** 上线。
   - 这些排行榜包括 [软件与 IT 服务](https://lmarena.ai/leaderboard/text/industry-software-and-it-services)、[写作、文学与语言](https://lmarena.ai/leaderboard/text/industry-writing-and-literature-and-language)、[生命、物理与社会科学](https://lmarena.ai/leaderboard/text/industry-life-and-physical-and-social-science)、[娱乐、体育与媒体](https://lmarena.ai/leaderboard/text/industry-entertainment-and-sports-and-media)、[商业、管理与财务运营](https://lmarena.ai/leaderboard/text/industry-business-and-management-and-financial-operations)、[数学](https://lmarena.ai/leaderboard/text/industry-mathematical)、[法律与政府](https://lmarena.ai/leaderboard/text/industry-legal-and-government) 以及 [医药与医疗保健](https://lmarena.ai/leaderboard/text/industry-medicine-and-healthcare)。
- **专家 Prompt 数据集现已开放**：包含职业标签的专家 Prompt 开源数据集已在 [Hugging Face](https://huggingface.co/datasets/lmarena-ai/arena-expert-5k) 上发布，为进一步的分析和开发提供资源。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1435283079085162688)** (947 条消息🔥🔥🔥): 

> `Comet referral program, Perplexity Pro chat history, Student subscription on active operator sub, Ethernet cable recommendations, Airtel subscription offer` 


- **推荐计划在美国仍然有效吗？**：一位成员询问 Comet 推荐计划是否仍对美国的合作伙伴开放，以及被推荐用户是否需要付费订阅才能获得推荐奖金。
   - 另一位成员回答说，推荐计划在美国仍然活跃，但 Perplexity 需要时间来验证推荐并处理付款。
- **丢失聊天历史？**：一位用户反映在访问 Perplexity Pro 的完整聊天历史时遇到困难，并对无法检索旧对话表示沮丧，但讨论中未找到解决方案。
   - 该用户指出他们只能看到大约 30 行的问题。
- **以太网线故障导致网速变慢**：一位用户报告其 **Cat6 以太网线损坏**，导致网速仅为 300kb/s，并寻求更换建议。
   - 另一位用户建议*几乎任何 Cat6 线缆*都可以，并提到他们通常根据之前 Perplexity 的推荐购买 Ugreen（绿联）产品。
- **GPT 模型混淆？**：一位用户表示沮丧，称选择 Claude Sonnet 4.5 或 Gemini 2.5 Pro 等特定模型时，有时会得到 **Haiku** 或 **Gemini 2 Flash** 等低质量模型的回复。
   - 这引发了关于 Perplexity 是否为了削减成本而故意使用更便宜模型的讨论，一些用户通过检查网络请求来确认正在使用的模型。
- **Comet 浏览器广告回归**：用户报告尽管 Comet 浏览器具有广告拦截功能，但 **YouTube 广告** 再次出现。
   - 成员们推测 YouTube 可能会调整其广告服务以绕过广告拦截器，从而影响了 Comet 的功能。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1435540437355200603)** (2 条消息): 

> `Spotify Song Sharing, Shareable Threads` 


- **用户分享 Spotify 歌曲**：一位用户分享了一首歌曲的 [Spotify 链接](https://open.spotify.com/track/5PjC1JmXRAOwJtFLrdaN4A?si=46ad168539424bd1)。
- **提醒保持 Thread 可分享**：一条消息提醒用户确保他们的 Thread 设置为 `Shareable`（可分享）。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1435384158951313508)** (3 messages): 

> `Perplexity Pro, Tool calling errors with sonar-pro, Tool calling with Perplexity API` 


- **Perplexity Pro 用户报告 API 问题**：有用户报告在运行 Perplexity 文档中的 [完整实现代码](https://docs.perplexity.ai/guides/chat-completions-guide) 时，遇到了 Tool calling 错误（*Tool calling is not supported for this model*）。
   - 该错误在使用 *sonar-pro* 作为支持 Tool calling 的模型时出现，表明文档与实际 API 能力之间可能存在差异。
- **关于 Perplexity API 中 Tool calling 支持的困惑**：用户对 Perplexity API 中 Tool calling 的可用性表示困惑，特别是在参考了 OpenAI 的实现之后。
   - 用户的目标是将 Tool calling 集成到他们的使用场景中，但错误信息显示指定的模型（*sonar-pro*）可能并不像预期那样支持此功能。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1435282873681838090)** (554 messages🔥🔥🔥): 

> `Tailwind 4, Nuxt 4, Phantom wallets, exoudos wallets, Specstory extension` 


- **Nuxt & Tailwind 获得新规则**：在升级到 **Tailwind 4** 和 **Nuxt 4** 后，用户为 **Cursor** 创建了 [新规则](https://link.to/rules)。
   - 一位成员表示 *Context7 MCP 表现非常出色*，并帮助重构了项目以使用 **Tailwind 4**。
- **Specstory 扩展无法保存**：有用户反映 **Specstory 扩展** 尽管设置了自动保存，但仍无法 [保存对话](https://link.to/chats)，且扩展显示 *e.setData is not a function* 错误。
   - 另一位成员发现，在空文件夹中进行的对话无法保存，*你需要打开一个文件夹*。
- **Cursor 与 GLM：初露苗头的合作？**：用户注意到 **GLM** 的开发者文档现在包含了 [与 Cursor 集成](https://link.to/glm-integration) 的说明。
   - 一位成员好奇这是否与 **Composer** 和 **GLM** 有关，或者两者正在进行协作。
- **Chat 中的语法高亮损坏**：有用户报告在 **2.0.38** 版本中，Chat 的语法高亮失效，代码除了红色花括号外全部显示为白色，并在 [论坛上发布了相关内容](https://link.to/forum-post)。
   - 另一位用户建议在反引号中添加带有语言名称的代码块来修复此问题。
- **Token 使用量过高**：一些用户遇到了意料之外的高 Token 使用量，即使是简短的 Prompt，输入成本也达到了 **100k-300k**，他们怀疑输入成本也将 AI 进行的读取/代码更改计算在内。
   - 发现 grep 命令由于其实现方式会导致大量的上下文摄入，如果 Agent 在 Prompt 中使用它，可能会摄入大型文件。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1435323712630161549)** (9 messages🔥): 

> `Mobile Web UI Crashing, Background Agents Use Case, Bug with images in prompts, Diff Rendering Improvements, API Endpoints for Cloud Agent` 


- **移动端 Web UI 在处理大 Diff 时崩溃**：用户报告移动端 Web UI 在处理大型 Diff 时崩溃，导致无法查看 Chat 和进行回复，考虑到移动端是 **background agents** 的主要使用场景，这尤其令人沮丧。
   - 用户表示 *体验非常糟糕，现在已经从勉强可以忍受变成了完全无法使用*。
- **调查 Cursor 在 Prompt 中处理图片的 Bug**：有用户报告了 Prompt 中图片的 Bug ([https://forum.cursor.com/t/getting-internel-error-on-cursor-com-for-prompts-with-images/139074/1](https://forum.cursor.com/t/getting-internel-error-on-cursor-com-for-prompts-with-images/139074/1))，这导致无法在项目中使用 background agents。
   - 针对移动端 Diff 的修复已经推送，现在运行应该更流畅，但 **composer-1 模型** 除外。
- **移动端 Diff 变得更流畅**：针对移动端 Diff 的修复已推送，现在运行应该更顺畅，不过整体渲染将更改为更高效的方式。
   - 进一步的细节仍然较少。
- **分享 Cloud Agent API 端点**：为了帮助使用 background agents，一位成员分享了 Cloud Agent 的 **API 端点**：
   - [Agent Conversation](https://cursor.com/docs/cloud-agent/api/endpoints#agent-conversation)、[Agent Status](https://cursor.com/docs/cloud-agent/api/endpoints#agent-status) 和 [Webhooks](https://cursor.com/docs/cloud-agent/api/webhooks)。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1435287778484293656)** (203 条消息🔥🔥): 

> `DeepSeek-OCR notebook, Q4 Ks-XL vs IQ4, TRL notebooks vs Unsloth notebooks, fine tune a cross encoder or embedding model, Qwen3VL-30B-A3B on 16GB vram` 


- **Unsloth 发布 DeepSeek-OCR 微调 Notebook**：Unsloth AI 宣布了一个新的 DeepSeek-OCR 微调 notebook，可通过[此链接](https://x.com/UnslothAI/status/1985728926556307471)获取。
   - 一些用户对超过 100% 的错误率表示担忧，这可能是由于预测文本长度与实际文本长度的差异造成的。
- **Q4 Ks-XL vs IQ4：量化难题**：一位用户询问了 **Q4 Ks-XL** 和 **IQ4** 量化过程的主要区别，旨在为 Agent 工作流优化 RAM 使用。
   - 该用户试图确定为了获得更多上下文而节省 RAM 是否值得使用稍大一点的模型。
- **Unsloth 对 TRL 的优化和速度提升**：当被问及使用 **TRL notebooks** 与 **Unsloth notebooks** 的区别时，用户回答说 Unsloth 由于自身的优化，提供了*更快的速度和更低的内存占用*。
   - 据解释，*Unsloth 对 TRL 和 Transformers 进行了打补丁（patch），以添加其自身的优化和一些降低 VRAM 的技术*。
- **Qwen3VL-30B-A3B 对 16GB VRAM 来说太大了**：一位用户询问是否可以在 16GB VRAM 的 GPU 上运行 **Qwen3VL-30B-A3B**。
   - 另一位用户回答说*这是不可能的*，因为它需要大约 17.5GB 的 VRAM，<:rip:1233329793584468062>。
- **视觉模型训练故障排除**：一位用户报告了 Qwen3 的问题，遇到了训练损失（training loss）低而验证损失（validation loss）高的情况，甚至数据集示例的回答也是错误的。
   - 该用户怀疑存在 masking 问题或 eval 和 train loss 计算不一致，并注意到 loss 随 batch size 的变化而波动。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1435309660361064488)** (4 条消息): 

> `Blockchain, AI, Trust in Code, Consensus Mechanisms` 


- **利用区块链和 AI 建立信任**：一位成员表达了对在代码中建立信任以及教机器思考的兴趣，这引导他们去探索 **blockchain** 和 **AI**。
   - 他们曾开发过让*共识（consensus）感觉真实且可靠*的系统，并将 AI 视为能够解决以前无法解决的问题的工具。
- **区块链和 AI 作为变革性技术**：该成员认为，如果使用得当，**blockchain** 和 **AI** 可以彻底改变行业和社区。
   - 这些技术还可以培养新的想法并改变运作方式，连接人们并解决问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1435326858379006063)** (196 条消息🔥🔥): 

> `SFT as RL, Quantum Randomness, Manual Data Entry, Nvidia Blackwell Pro 4500 vs 5090, ECC Memory Value` 


- **SFT 是具有单 token rollout 的 RL**：一位成员表示，[监督微调 (SFT)](https://en.wikipedia.org/wiki/Supervised_learning) 只是具有单 token rollout 的强化学习 (**RL**)。
   - 另一位成员详细解释说，策略（policy）执行一个动作，根据该动作与 ground truth 的似然度给予奖励，然后重复此过程。
- **数据录入工作——简直是最糟糕的**：成员们分享了在银行从事数据录入工作的轶事，强调了手动键入、高级人员的双重确认以及使用计数程序跟踪 KPI。
   - 一位成员提到，每当完成一项任务时，都需要在计数程序上按一下 **+1 按钮**，并按其他按钮表示正在休息，最后得出结论：*我余生不能再这样生活了*。
- **Nvidia Pro 4500 作为 Blackwell 显卡亮相**：**Nvidia Pro 4500** 被揭晓为一款真正的基于 Blackwell 的工作站显卡，拥有 **32GB** 显存和 **200W** 功耗。
   - 随后讨论了它相对于 **5090** 的定位，5090 提供了 **2 倍** 的显存带宽。
- **ECC 内存辩论升温**：**ECC 内存**的价值引发了辩论，一位成员强烈主张使用它，因为它能够在流体动力学模拟等长时间运行的任务（可能运行数周）中防止比特翻转（bitflips）。
   - 他说*如果你遇到了比特翻转，你就得重头再来*，这就是*为什么你需要 ECC——它在硬件层面进行内存奇偶校验*。
- **GTC 大会的价值受到质疑**：参加 **Nvidia GTC 大会**的价值受到质疑，一位成员指出今年的活动过于拥挤，更多关注公司服务而非新研究。
   - 另一位成员回忆说 GTC24 非常棒，因为*《Attention Is All You Need》的作者们首次同台讨论，而我坐在前排*，而今年*真的感觉你正在见证一些疯狂的事情*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1435291863656566927)** (79 条消息🔥🔥): 

> `MiniMax M2 local inference, GPT-OSS-20B training with Unsloth + REINFORCE, System prompt usage during finetuning, Multilingual reranker with GGUF and llama.cpp, Granite 4.0 Hybrid 4-bit conversion issues` 


- ****5090 上的 MiniMax M2 需要量化指导****：一位拥有 **5090** 和 **128GB RAM** 的用户正在寻求在本地运行 [MiniMax-M2-GGUF](https://huggingface.co/unsloth/MiniMax-M2-GGUF/discussions/4) 的建议，考虑使用 **4-bit quantization** 和 **MXFP4** 版本，以及针对 Agent 任务的软件和 UI 推荐。
   - 他们还对探索带有 *accuracy recovery adapter* ([Reddit 链接](https://www.reddit.com/r/LocalLLaMA/comments/1mytbfz/accuracy_recovery_adapter_with_selfgenerated_data/)) 的 **3-bit models** 感兴趣。
- ****通过 SFT Trainer 重新构想 REINFORCE 训练****：一位用户尝试使用 Unsloth 和原生 **REINFORCE** 训练 **gpt-oss-20b**，遇到了 TRL 的 **RLOO** 要求*每个 prompt 至少生成 2 个结果*的问题。
   - 他们现在正尝试通过 **SFT trainer** 模拟 **REINFORCE** 作为一种变通方案。
- ****微调期间 System Prompt 的最佳实践****：在 Finetuning 过程中，一位用户询问是否应该放弃 System Prompt 或将其缩短，因为其内容应该被“固化（baked）”到较小的模型中。
   - 回复建议这取决于任务复杂度：对于翻译等显而易见的任务可以舍弃，对于多步骤流程则应保留。
- ****Qwen Reranker 在 Llama.cpp 生态中崛起****：用户正在寻找能够与 **GGUF** 和 **llama.cpp** 配合使用的优秀**多语言 Reranker**。
   - 有建议称 **Qwen** 有一套支持 llama.cpp 且支持多语言的 Reranker，但其质量仍需评估，且其 Embedding 模型最初存在一些问题。
- ****Granite 的 4-bit 计划因精度问题受阻****：一位用户在将 **ibm-granite/granite-4.0-h-tiny-base** 转换为用于训练的 **4-bit safetensors** 时遇到问题，即使使用 Unsloth 的方法，模型仍以 **16-bit precision** 保存。
   - Unsloth AI 已有与 Granite 模型及此问题相关的 [issue #3558](https://github.com/unslothai/unsloth/issues/3558) 和 [issue #3550](https://github.com/unslothai/unsloth/issues/3550)。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1435375994247712981)** (9 条消息🔥): 

> `Roblox PII Classifier, Open Sourcing Safety Tools, PII Dataset Access` 


- **Roblox 开源 PII 分类器**：Roblox 开源了他们的 **PII Classifier AI**，这是一个旨在检测聊天中个人身份信息（PII）的工具，可在 [Hugging Face](https://huggingface.co/Roblox/roblox-pii-classifier) 上获取，并在其[新闻发布](https://corp.roblox.com/newsroom/2025/11/open-sourcing-roblox-pii-classifier-ai-pii-detection-chat)中进行了详细介绍。
   - 安全团队为这一保护儿童上网强大的工具感到自豪，并强调了其可扩展性：平均每天处理 **61 亿条聊天消息**，峰值达到每秒 **200,000 次查询**，且 **P90 latency <100ms**。
- **Roblox 的公共安全模型引起关注**：人们对 Roblox 的公共安全模型表现出兴趣，特别是其在大规模调度管理方面的挑战，在这种规模下，大部分时间可能消耗在 **vLLM batching** 而非 **GPU forward passes** 上。
   - 尽管对该专业模型表示赞赏，但人们对数据集本身表现出浓厚兴趣，同时也承认由于 **PII 合规问题**，该数据集无法公开。
- **数据集可访问性问题得到确认**：对包含 **PII 的数据集** 的访问受到严格限制，直接询问该数据集通常会得到其包含敏感信息的默认理解。
   - 强调了甚至只是查看该数据集都需要**极高级别的权限**，这进一步强化了现行的严格隐私措施。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1435348926663491624)** (52 条消息🔥): 

> `YouTube 讲座, B770 GPU, 3080 改装, 自动向量化编译器博客文章` 


- **12 月前的讲座：已确认！**：一位用户询问了 12 月之前的讲座，另一位成员指向了 [活动列表](https://discord.com/channels/1189498204333543425/1191300313928433664/1435350172992540793)。
   - 一段关于类似讲座的 **YouTube** 视频也被分享了：[链接](https://www.youtube.com/watch?v=nFfcFyBEp7Y)。
- **用户推测 B770 性能和 GPU 改装**：一位用户表达了对 **B770 GPU** 的渴望，并考虑通过改装 **3080** 来制造一个，旨在达到类似于 **4070** 的性能。
   - 另一位用户指出理论上可以增加 **20GB VRAM**，但承认存在资金限制。他们将其他人引导至 [gpu 改装频道](https://discord.com/channels/1189498204333543425/1349152646484987974)。
- **自动向量化编译器博客文章影响 Discord**：一位成员提到阅读了博客文章，特别是引用了 **ThunderKittens** 的内容，包括一篇关于自动向量化的文章，该文章产生了很大影响。
   - 另一位成员向该用户分享了这篇 *自动向量化 (auto vectorization)* 博客文章。


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1435313453286818084)** (1 条消息): 

> `社区见面会, 时区, 会议详情` 


- **社区见面会已安排**：社区见面会定于明天 **10am-11am PST** 举行。
   - 会议详情请参阅原始帖子。
- **时区提醒**：见面会时间为 **10am-11am PST**，可能需要根据不同时区进行转换。
   - 鼓励参与者检查各自所在的时间。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1435297826249900266)** (5 条消息): 

> `内存受限 (Memory-Bound) 的 Matmuls, SM 数量对 Matmul 延迟的影响, 饱和 GPU 内存带宽, Little's Law 与 GPU 性能的相关性` 


- **CUDA 频道讨论内存受限的 Matmuls**：一位成员询问为什么在内存受限的 matmuls 中需要所有的 **SM** 才能获得最佳延迟，并质疑少数几个 **SM** 是否就能使内存带宽饱和。
   - 另一位成员质疑 matmuls 是否属于内存受限，并声称 *需要不止几个 SM 才能使内存带宽 (BW) 饱和*。
- **对内存带宽饱和所需的 SM 数量提出质疑**：一位成员表示 decode matmuls 是内存受限的，估计 **30%-40%** 的 **SM** 就足以使内存带宽饱和。
   - 他们还分享了一张与该问题相关的 [图片](https://cdn.discordapp.com/attachments/1189607726595194971/1435489664135069706/image.png?ex=690cd02e&is=690b7eae&hm=331e3b1f0261ac9725ecd33df2daa3ff5be74db0c6143852f2b1ee26006db94c&)。
- **新型 GPU 上的 HBM 带宽饱和挑战**：一位成员提到，在较新的基于 **HBM** 的 GPU 上，即使使用所有 **SM**，要达到满额内存带宽也可能很棘手。
   - 他们建议使用 **Little's Law** 来理解这一点，并链接到了一个 NVIDIA 会议：[GTC25-S72683](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/)。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1435292929722941450)** (19 条消息🔥): 

> `torch.compile CUDA graph recapture, torch.compile grouped_mm UserWarning, vLLM pytorch dependencies, Float8 tensors limitations, custom kernel opcheck failure` 


- **调试 Torch Compile CUDA Graph Recapture**: 一位用户正在寻求关于在 `max-autotune` 模式下使用 `torch.compile` 调试 **CUDA graph recapture** 的建议，旨在确保 warmup 后的图稳定性，并使用 `torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = 1` 来检测重新捕获（recaptures）。
   - 一位成员建议检查是否存在可能产生干扰的 **多个版本的 torch 或 vLLM**。
- **Torch Compile Grouped MM UserWarning 爆发**: 用户报告了大量与 **torch.compile + grouped_mm** 相关的恼人 `UserWarning` 消息，特别是关于 *非标量张量已弃用逻辑运算符 'and' 和 'or'* 的警告。
   - 一位成员表示[他们已经为 flex 修复了此问题](https://github.com/pytorch/pytorch/issues/167041)，并且可以为 grouped GEMM 应用相同的修复，另一位用户已经开设了一个官方 issue。
- **vLLM 硬编码 PyTorch 和 Triton 依赖**: 一位用户注意到 **vLLM** 硬编码了像 [这些](https://github.com/vllm-project/vllm/blob/0976711f3b569aae4a8c9ac148f0771624293120/requirements/cuda.txt#L13) 类似的依赖，导致在使用自定义 PyTorch/Triton 版本时出现困难。
   - 他们进一步提到，在 v0.10.1 之前，构建相对容易一些。
- **Float8 张量触发 NotImplementedError**: 一位用户在运行自定义 kernel 进行 `opcheck` 时，遇到了 *"mul_cuda" not implemented for 'Float8_e4m3fn'* 的 `NotImplementedError`。
   - 另一位成员澄清说，低比特 dtype 只是表示形式，并建议使用 **scaled mm** 进行计算，并指出与此相关的修复可能已经落地。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 条消息): 

marksaroufim: https://xillybus.com/tutorials/pci-express-tlp-pcie-primer-tutorial-guide-1
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1435404948530794598)** (1 条消息): 

> `Mixlayer, AI inference platform, Rust, CUDA, founding engineer` 


- **Mixlayer 寻找创始工程师！**: [Mixlayer](https://mixlayer.com) 的创始人正在招聘一名熟悉 **Rust** 和 **CUDA** 的创始工程师，该公司是一个面向高级用户的 **AI 推理平台**。
   - 该工程师将负责他们的自定义推理引擎；工作地点首选 **SF**（旧金山）混合办公，但也接受远程办公。
- **Mixlayer - AI 推理平台**: Mixlayer 是一个专为高级用户设计的 **AI 推理平台**，为开发者提供对开源 LLM 的底层访问，以增强产品开发。
   - 该平台专注于通过让开发者访问底层 LLM 来帮助他们构建更好的产品。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1435297205366816807)** (16 条消息🔥): 

> `RL bug on accumulator type fixed at fp32, Practice Deshourading, Hackathon, PyTorch/vllm on AMD AI PCs` 


- **RL 累加器 Bug 导致舍入误差**: 一位用户发现了一个 **RL bug**，即即使使用了 **bfloat16**，累加器类型也被固定为 **fp32**，由于计算的中间值存储在 **fp32** 中，从而导致了舍入误差。
   - 用户质疑为什么它只被固定在 **fp32**，以及这个问题有多严重。
- **Deshourading 熟能生巧！**: 一位用户正在寻找练习 **Deshourading** 的好方法，因为 GPU 太贵了。
   - 另一位成员鼓励他们继续尝试，并提到 *现在有很多文献和资源可供参考*。
- **黑客松吸引新人**: 几位用户提到在听说黑客松（Hackathon）后加入了频道。
   - 一位来自印度的用户询问了参赛资格，而另一位用户鼓励大家积极参与，强调这是一个学习机会，并表示 *至少在比赛过程中，你会学会如何处理基础知识。*
- **PyTorch/vllm 在 AMD AI PC 上运行**: 一位用户询问如何让 **PyTorch/vllm** 在 **AMD AI PC** 上运行。
   - 他们提到尝试了各种 Docker 变体和 therock 仓库，但无法让 **PyTorch** 识别 **APU**，并请求指导。


  

---

### **GPU MODE ▷ #[jax-pallas-mosaic](https://discord.com/channels/1189498204333543425/1203956655570817034/1435698383091404900)** (1 messages): 

> `Mosaic-GPU, all-gather-matmul` 


- **Mosaic-GPU 的 all-gather-matmul 代码请求**：一位成员询问了 **mosaic-gpu** 项目中 **all-gather-matmul** 的完整代码可用性。
   - 该成员特别提到了 [JAX documentation page for collective_matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html) 作为参考。
- **Mosaic-GPU all-gather-matmul 的位置仍然难以寻觅**：**mosaic-gpu** 中 **all-gather-matmul** 的具体实现位置尚不明确。
   - 尽管进行了搜索，该成员仍未能找到源代码的直接链接。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1435305338302890085)** (7 messages): 

> `OSS Contribution, fbgemm kernels, fp8 weight-only pattern & torch.compile` 


- **首个 OSS 贡献尝试即将到来**：一位成员询问是否可以尝试进行 PR 修复，并表示 *这将是我第一次正式的 OSS 贡献，哈哈*。
   - 另一位成员回复道 *当然可以，你打算处理哪个 kernel？*
- **Kernel 代码库分析进行中**：一位成员表示，该代码库主要复用了 **fbgemm kernels**，他们需要检查这是否是完整的解决方案。
   - 他们补充说，他们 *基本上对这些代码还很陌生*。
- **fp8 weight-only 模式的兼容性受到质疑**：一位成员询问 **torch.compile** 是否应该支持 **fp8 weight-only pattern**。
   - 另一位成员回应称，在 **torch.compile** 中使用 **Float8WeightOnlyConfig** *应该是可行的*。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1435290643319488562)** (3 messages): 

> `Disaggregated Inference Retrospective, Symbolica AI Hackathon` 


- **Hao Lab 对 Disaggregated Inference 的回顾**：Hao Lab 发布了一篇关于 **Disaggregated Inference** 18 个月后的回顾博客，链接见 [此链接](https://x.com/haoailab/status/1985753711344316648)。
- **Rust 开发者齐聚 Symbolica AI 黑客松**：Symbolica AI 将于 **11 月 8 日星期六**在旧金山举办黑客松，面向对 **formal logic**、**automated theorem proving**、**types**、**compilers** 和 **AI** 感兴趣的 Rust 开发者；请在 [Luma](https://luma.com/1xa9d6nr?utm_source=meetup) 报名。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1435423285356658819)** (11 messages🔥): 

> `vectoradd_v2, grayscale_v2, vectorsum_v2, A100, B200` 


- **向量加法在多个平台取得佳绩**：一位成员在 **A100** 上以 **896 µs** 的成绩获得 `vectoradd_v2` 第一名，另一位成员在 **H100** 上以 **525 µs** 获得第一名。
   - 另一位成员在 **B200** 上以 **237 µs** 的成绩获得 `vectoradd_v2` 第三名。
- **GrayScale 在 GPU 上表现出色**：一位成员在 **H100** 上以 **1369 µs** 的成绩获得 `grayscale_v2` 第一名，随后又在 **B200** 上以 **600 µs** 夺得第一名。
   - 同一位成员还在 **A100** 上以 **2.39 ms** 的成绩获得了 `grayscale_v2` 第二名。
- **向量求和表现优异，稳居第二**：一位成员在 **B200 (119 µs)** 和 **H100 (126 µs)** 上均获得了 `vectorsum_v2` 第二名。
   - 该成员还在 **L4** 上以 **918 µs** 的成绩夺得 `vectorsum_v2` 第一名。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1435477572631924919)** (5 messages): 

> `Factorio RCON, Factorio Hidden Settings, Factorio multiplayer server` 


- **Factorio 的 RCON 访问方式公开**：根据一位用户的说法，访问 **RCON** 不需要运行 Docker 服务器；在 macOS 的 Factorio 客户端中，按住 **Control + Option** 的同时点击主菜单设置，会显示一个 "The Rest" 选项，用于查看隐藏设置，从而为 **FLE** 添加 **RCON** 端口和密码。
   - 另一位用户确认上述技巧也适用于普通的单人游戏。
- **Factorio 客户端托管本地 RCON 服务器**：从 Factorio 客户端启动/托管多人服务器将在该端口本地运行 **RCON**，这对于开发非常有用。
   - 一位用户确认这很有用，并感谢原帖者的分享，并提议明天进一步讨论文档。


  

---

### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1435444018564562944)** (30 条消息🔥): 

> `节点分配、方案分享、竞赛提交可见性、排名问题` 


- **节点分配说明**：用户讨论了 **8 个节点** 的分配以及是否应该有 **8 个 runner**。
   - 已澄清全天 **96 小时** 的运行时间包含开销，因此实际活跃执行时间会更高。
- **GPU 代码分享文化兴起**：一位用户询问 GPU 编程中是否存在类似于竞赛编程 (CP) 圈子的方案分享文化。
   - 另一位成员表示希望能有一个平台促进方案分享，并指出 GPU 编程竞赛尚属新鲜事物，因此方案分享目前还未成气候。
- **GitHub 上的 GPU 珍宝分享**：Team Gau 在 [gpu-mode-kernels](https://github.com/gau-nernst/gpu-mode-kernels/tree/main/amd-distributed/all2all) 分享了他们的 kernel，并发布了一篇关于为 AMD 开发者挑战赛 2025 [优化分布式推理 kernel](https://www.yottalabs.ai/post/optimizing-distributed-inference-kernels-for-amd-developer-challenge-2025) 的简短文章。
   - 他们承诺稍后会提供更详细的解释。
- **代码提交可见**：用户现在可以在登录 Discord 账号后，在网页上查看已结束排行榜的提交代码，正如该 [公告](https://cdn.discordapp.com/attachments/1359640791525490768/1435763582078947369/image.png?ex=690d2689&is=690bd509&hm=3430a612bb2eaed1dc1cfa2b90d9ebe5b21db0097037b20aa40a29ba4f139d55) 中所强调的。
   - 第一场竞赛的所有提交内容都可以在 [huggingface.co](https://huggingface.co/datasets/GPUMODE/kernelbot-data) 找到。
- **请求修正排名**：一位用户请求修正 **amd-all2all** 竞赛的排名，声称其 **216us** 的方案在截止日期前不久被判定为违规。
   - 他们强调其最终提交仅达到了 **263us**，并请求承认 216us 的提交才是真正的获胜结果。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1435512487117328504)** (4 条消息): 

> `CuTeDSL 资源、CuTe 拷贝线程、CuTeDSL 求和归约 Kernel` 


- **请求汇总 CuTeDSL 资源**：一位成员请求学习 **CuTeDSL** 的资源和教程，特别是寻求关于在 **CuTeDSL** 中使用 **nvfp4** 的信息。
   - 在给定的消息中未提供具体的资源或教程。
- **CuTe 自动线程拷贝查询**：一位用户询问 **CuTe** 在拷贝时是否会自动选择线程，特别是当使用一个 warp 启动 `copy(cpy, src, dst)` 但 `size(cpy)<32` 时，内部是否存在 `if` 判断。
- **CuTeDSL 求和归约 Kernel 故障排除**：一位用户寻求关于 **CuTeDSL** 中**求和归约 kernel** 的帮助，报告无法在 block 之间进行求和归约，并提供了一个 [相关代码片段](https://gist.github.com/kaiyuyue/c4a18ca59c3c63a2b8009704a9b7496b)。


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1435304405107998730)** (4 条消息): 

> `Mojo GPU Puzzles、Mojo 中的 Layout API、Mojo 版本兼容性` 


- **Mojo GPU Puzzles 教程系列发布！**：一位成员创建了一个配套 **Mojo GPU Puzzles** 的 [视频教程系列](https://www.youtube.com/watch?v=-VsP4kT6DjA)，前两集已于今日发布。
- **Layout API 在 Mojo v25.1 中首次亮相**：`Layout.row_major` 和 `Layout.col_major` 函数在 **Mojo 25.1** 版本中引入，作为 layout 包（2025 年 2 月 13 日发布）的一部分，用于描述 tensor 的组织结构。
   - 一位成员正在排查 [leetgpu.com](https://leetgpu.com) 中的 *error: no matching function in call to 'row_major'* 错误，该网站似乎运行的是 **Mojo 25.4**。


  

---

### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1435286769858773052)** (8 messages🔥): 

> `picograd commits, fuzzing against np,torch, and tinygrad, pedagogical progression, kernels vs compilers` 


- **Picograd 中的提交不断增加**：[picograd repo](https://github.com/j4orz/picograd) 进行了多次提交，包括 [b97b9ea](https://github.com/j4orz/picograd/commit/b97b9ea0eda2282bb5e193558c370c53345f07d9), [43796e0](https://github.com/j4orz/picograd/commit/43796e049eb225f9c2dd093a72ccfa09f237db09), [ae47d4d](https://github.com/j4orz/picograd/commit/ae47d4d72f0757b8e542e6b923ca910a7ae56ecc), [625d6ac](https://github.com/j4orz/picograd/commit/625d6acb6cd9395010024e5148886ddb34d7a563), [b9fbf87](https://github.com/j4orz/picograd/commit/b9fbf879b4030688a357402111f77ed12ffd01a9), [90da2f0](https://github.com/j4orz/picograd/commit/90da2f02cc64f4364de08255a6555f36b8e2e019), 以及 [46eb304](https://github.com/j4orz/picograd/commit/46eb304415a21e7a542fa7151508067e7b56f514)。
   - 目前具体更改尚不明确。
- **Fuzzing 热潮**：一位成员询问是否有人有兴趣 *研究针对 np, torch 和 tinygrad 的 fuzzing？*
   - 目前尚未有人回应其请求。
- **教学演进范式**：宣布教学演进将有 **3 种模式**：**EAGER_NAIVE=1**、**EAGER_RUNTIME=1** 和 **GRAPH=1**。
   - 第一种模式为每个 op 启动 kernel，因此没有无拷贝视图（copy-free views）；而第二种模式将拥有类似 `torch.Storage` 和 `tinygrad.Buffer` 的托管运行时。
- **Kernels 与 Compilers**：一位成员非常喜欢关于 [kernels vs compilers](https://www.youtube.com/watch?v=Iw4xKHPl7hI) 的讨论，其中 mostafa 和 simran 发表了精彩见解。
   - 该成员感谢大家提出的精彩问题。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1435527133878292593)** (3 messages): 

> `CUDA, Triton` 


- **内联 CUDA 的困扰**：一位成员提到他们通过 `load_inline` 使用内联 CUDA，但有时感觉像是在逆流而上。
   - 他们猜测大多数人使用的是像 **Triton** 这样的 Python DSL。
- **Triton 的简便性与性能损耗**：一位成员表示 *triton* 非常易于使用，但可能在性能上有所保留（perf left on the table）。


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1435385420115476642)** (11 messages🔥): 

> `DeepSeek FP8, Cutlass FP8 GEMM, FP8 Blockwise Training, Benchmarking Scripts, Blockwise Quantization Kernels` 


- **DeepSeek 的 FP8 实现获得进展**：一位用户指出在 PyTorch 的 `ao` 仓库中实现 **DeepSeek 风格的 FP8 分块训练** 是一个 [good first issue](https://github.com/pytorch/ao/issues/3290)。
- **Cutlass 已经拥有 DeepSeek FP8 GEMM？**：一位用户提到 [CUTLASS](https://developer.nvidia.com/cutlass) 已经有一些 **DeepSeek FP8 GEMM 实现**，并提供了示例链接，例如 [带有分块缩放的 Hopper FP8 warp-specialized GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling) 和 [带有分块缩放的分组 GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling)。
- **分块 FP8 基准测试迎来志愿者**：一位用户自告奋勇为与 **DeepSeek FP8** 相关的 **量化** 和 **GEMM kernels** 编写 **基准测试脚本**，并后续跟进优化，参考了 [此处](https://github.com/pytorch/ao/blob/main/torchao/kernel/blockwise_quantization.py) 现有的分块 FP8 GEMM 和量化实现。
- **建议使用 DeepSeek FP8 基准测试框架**：一位用户询问是否可以基于 [此框架](https://github.com/pytorch/ao/blob/main/benchmarks/benchmark_blockwise_scaled_linear_triton.py) 进行基准测试。
- **DeepSeek FP8 Triton Kernels 需要进行基准测试**：一位用户计划为来自 `blockwise_fp8_training/kernels.py` 的五个特定 kernel 创建新的 **基准测试脚本**，包括 `triton_fp8_blockwise_act_quant_lhs`、`triton_fp8_blockwise_act_quant_rhs`、`triton_fp8_blockwise_act_quant_transposed_lhs`、`triton_fp8_blockwise_weight_quant_rhs` 和 `triton_fp8_blockwise_weight_quant_transposed_rhs`。


  

---

### **GPU MODE ▷ #[opencl-vulkan](https://discord.com/channels/1189498204333543425/1418990184367919267/1435443066075877517)** (6 messages): 

> `GLSL Vulkan Compute Shaders, clspv, Clvk, Slang shading language` 


- **放弃 GLSL 转向 Vulkan Compute**：一位成员更倾向于坚持使用 **GLSL Vulkan Compute Shaders** 来针对 Vulkan，但并不喜欢 **clspv**。
   - 他们提到 **clspv** 和 **Clvk** 是独立开发的，没有发布流程，需要从 master 分支构建，这导致他们更倾向于选择 **GLSL** 或 **Slang**。
- **Slang 着色语言：很酷但多余？**：据该成员称，**Slang** 是新兴的酷炫着色语言，它是对 **GLSL** 的巨大升级，特别是如果专注于 compute 的话。
   - 他们补充道，*世界并不需要另一种具有 4 种方式来编写相同概念的 C 衍生语言*。


  

---


### **GPU MODE ▷ #[cluster-management](https://discord.com/channels/1189498204333543425/1420098114076803142/1435655976605847736)** (1 messages): 

> `Ansible Scripts, Configuration Management` 


- **Ansible 脚本纠正配置错误**：成员提到可以使用 **Ansible 脚本** 来处理 **ulimits** 和类似的配置，以纠正配置错误。
   - 它们还使得查看是否存在配置错误变得容易。
- **Ansible 配置的优势**：使用 **Ansible** 为管理 **ulimits** 等系统配置提供了一种清晰且可审计的方法。
   - 这些脚本不仅能纠正现有问题，还能提供过去配置的透明概览。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1435347373776703630)** (4 messages): 

> `inline_triton, Helion Compiler, atomic_cas, output_like=None, Helion API` 


- **Inline Triton 问题需要变通方法**：成员们讨论了让 `inline_triton` 工作的变通方法，例如在最后一行添加一个哑值（dummy value）或设置 `output_like=None`，相关 issue 已在[此处](https://github.com/pytorch/helion/issues/1086)创建。
- **`output_like=None` 在边缘情况下失败**：设置 `output_like=None` 仍然要求最后一行是一个表达式，因此仍需要使用哑值的变通方法。
   - 一位成员提供了使用 `hl.inline_triton`、`while` 循环和 `atomic_cas` 函数的 Helion 示例代码，并指出错误：`helion.exc.InvalidAPIUsage: Invalid usage of Helion API: The last line of triton_source must be an expression`。
- **Helion Compiler 丢失对 host tensor 的跟踪**：即使使用了哑值，代码仍无法编译，且编译器丢失了对 host tensor 的跟踪，相关问题已添加到 issue 中。
   - 一位成员建议在 `output_like=None` 时，使用 `{1}` 作为最后一行的替代代码片段。
- **Helion 的“最后一行”限制修复即将到来**：一位成员正在提交 [PR](https://github.com/pytorch/helion/pull/1087)，以移除当 `output_like=None` 时“triton_source 的最后一行必须是表达式”的要求。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1435282949678432287)** (85 messages🔥🔥): 

> `NVFP4 Contest, Mojo Support, Blackwell GPUs, TileLang and CuTeDSL, CUDA learning` 


- **Mojo 支持可能即将到来！**：一位成员询问 *是否需要为 **mojo 支持** 做些什么？* 并自愿提供帮助，因为 *它比 cutlass 简单得多*。
   - Marksaroufim 随后回复道：*我们可以从一开始就提供 mojo*。
- **无需为比赛购买 Blackwell GPU**：一位成员询问是否每个人都有 **Blackwell GPU** 以备战比赛，Marksaroufim 回复道：*你并不真的需要它。如果你愿意排队等待，基础设施是免费使用的。*
   - Marksaroufim 还提供了一个[推荐链接](https://cloud.sesterce.com/compute?referralCode=SES-KNYPOQ-3b6e)，为这里的每个人提供折扣。
- **用户将拥有 NCU profiling 权限**：一位成员询问 *他们是否允许在 NVIDIA 系统上使用 nsys 和其他 CLI 性能分析工具？如果我们无法对 kernel 进行 profile，那将非常困难 🥲*
   - 一位用户指出，有一条消息提到提交机器人将具有运行代码并返回 **ncu capture 文件**供你分析的功能。
- **无论是否有奖金，GPU MODE 都欢迎全球展示**：在回答有关参赛资格的问题时，Marksaroufim 澄清说：*对谁可以参加没有限制，世界各地的任何人都可以运行 nvfp4 kernel，进行学习并在公共排行榜上展示。*
   - 唯一的限制仅适用于谁有资格获得奖金，这通常涉及复杂的法律讨论，最好依靠 NVIDIA 官方代表的答复。


  

---

### **GPU MODE ▷ #[hf-kernels](https://discord.com/channels/1189498204333543425/1435311035253915840/1435311820712972450)** (3 条消息): 

> `Xenova.com, HF Kernels 更新` 


- **Xenova 表示 Let's Go!**：一位用户感谢了 **Xenova** 和 **xenova.com**，后者回应了 *Let's go!*，预示着可能的兴奋或进展。
- **HF Kernels 频道活跃**：**hf-kernels** 频道用户参与度很高，表明有关于 **HF kernels** 的持续讨论和更新。


  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1435679715531952169)** (1 条消息): 

> `Sentence Transformers 加入 Hugging Face，huggingface_hub v1.0，LeRobot v0.4.0，更简洁的 Collection URL，推理提供商使用情况明细` 


- **Sentence Transformers 与 Hugging Face “喜结连理”**：[Sentence Transformers](https://huggingface.co/blog/sentence-transformers-joins-hf) 正式加入 Hugging Face，以支持开源机器学习计划。
   - 此举旨在将其 **transformer models** 更深入地整合到 Hugging Face 生态系统中。
- **Hugging Face Hub 迎来五周年，庆祝开源机器学习基石的建立**：[huggingface_hub v1](https://huggingface.co/blog/huggingface-hub-v1) 标志着构建开源机器学习基石的 **五年** 历程，增强了模型共享与协作。
   - 版本 1 包含重大升级，如更简洁的 collection URL、改进的推理和全新的 API，进一步 **简化了模型的访问和使用**。
- **LeRobot v0.4.0 为开源机器人学习提供强力支持**：[LeRobot v0.4.0](https://huggingface.co/blog/lerobot-release-v040) 为 **开源机器人学习 (OSS Robotics Learning)** 引入了增强功能，提升了仿真与现实世界应用的集成。
   - 该更新专注于改进 **OSS Robotics Learning**。
- **Hugging Face 与 VirusTotal 强化 AI 安全**：Hugging Face 正与 [VirusTotal](https://huggingface.co/blog/virustotal) 合作，通过集成威胁检测功能来 **加强 AI 安全**。
   - 此次合作旨在为用户提供增强工具，以 **识别并减轻** AI 模型和应用中的潜在安全风险。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1435297556996292821)** (189 条消息🔥🔥): 

> `家庭实验室 AI 配置，Vex-Math-L1-100K 数据集，LLM 在股票数据预测中的应用` 


- **前沿实验室 AI 转向家庭实验室**：成员们讨论了从前沿实验室 AI（如 **Anthropic** 和 **OpenAI**）转向用于模型后训练的 **homelab** 配置的可能性，重点关注环保配置。
   - 共识是 **规模经济** 更倾向于云解决方案，因为即使使用 **LoRA fine-tuning** 等技术，家庭实验室的配置成本也很高，且可能不够环保。
- **新数学数据集激发 AI 爱好者兴趣**：一家初创公司宣布发布 [Vex-Math-L1-100K 数据集](https://huggingface.co/datasets/Arioron/Vex-Math-L1-100K)，涵盖 **9 个主要子领域** 和 **135 个子领域**，包含 **10 万个示例**。
   - 初步印象表明，较小的模型在量化到较小尺寸时可能会表现出更高的质量，这挑战了“量化到较小尺寸的大模型总是优于小模型”的观点。
- **LLM 进入股票数据预测领域**：成员们讨论了 **LLM** 在 **股票数据预测** 中的应用，探索寻找数据间相关性的方法，包括天气和新闻等因素。
   - 结论是目前原生技术可能已经足够，**Generative AI** 在该领域的研究尚不深入；一位成员提供了 [更多 RAG 架构的链接](https://github.com/NirDiamant/RAG_Techniques)。


  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1435321857216872620)** (3 messages): 

> `Job Application Automation Project, BERT Model Training, SetFit Contrastive Classifier, ArXiv Gatekeeping` 


- **职位申请自动化对抗垃圾邮件检测**：一名团队成员正在开发一个使用 **Playwright** 的 **Python 系统**，用于在 **Lever** 和 **Greenhouse** 等网站上自动申请职位，目前已成功实现抓取职位链接并填写简单字段。
   - 然而，他们正面临 **垃圾邮件检测** 和不一致的 **HTML 选择器** 的挑战，需要更智能的隐身策略和调试工作。
- **BERT 模型训练与 SetFit 分类器的探索**：一名成员训练了一个用于分类的 **BERT 风格模型** 和一个 **SetFit 对比风格的二元分类器**，但发现很难找到可以提交结果进行评估的地方。
   - 他们使用 **Qwen 模型** 在 **NPHardEval** 上比基准线提升了 **48%**，在**图着色问题 (Graph Coloring Problem)** 上获得了 **88/100 的最优正确答案**。
- **学术门槛令研究员感到沮丧**：一名成员表达了对参与学术流程之难的挫败感，提到了 **ArXiv** 的门槛问题以及被 **BBS** 拒稿的经历。
   - 他们正在寻找一个能够传播显著结果并用于改进 **AI** 的**优质基准 (Benchmark)**，并请求 **ArXiv 推荐 (referral)**。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1435320084942815315)** (18 messages🔥): 

> `Model on 40-50b parameters, PDF2TXT parser, DearDiary.jl, NexusAI Professional Suite v1.0` 


- **模型开发者挑战 40-50B 参数**：一名成员正在构建自己的 **40-50B 参数模型**，并尝试在具有 **32GB 显存 (VRAM)** 的显卡上运行。
   - 他们基于 **gpt-oss-20b** 开发，并计划使用 **8k 或 16k 上下文窗口 (context windows)**。
- **PDF2TXT 解析器升级**：一名成员分享了其 **PDF2TXT 解析器** 的新版本，其中包括*更好的文本块搜索*，且在简单表格识别（无 OCR）方面速度提升了 *20%*，[点击此处查看](https://huggingface.co/kalle07/pdf2txt_parser_converter)。
- **DearDiary.jl 轻松追踪 ML 实验**：**DearDiary.jl** 是一个纯 Julia 工具，用于通过 [REST API](https://github.com/pebeto/DearDiary.jl) 和 SQLite 便携性来追踪机器学习实验。
   - 通过 "] add DearDiary" 安装即可访问完整的实验追踪功能和便携式设置。
- **NexusAI 套件发布专业工作流**：**NexusAI Professional Suite v1.0** 发布，包含适用于商业、写实、动漫和恐怖应用的生产级 [ComfyUI 工作流](https://github.com/NexusAI-Lab/ComfyUI-Professional-Workflows/releases/tag/v1.0.0)。
   - 该套件承诺实现从创意到资产的一键操作，并声称可节省数百小时的配置时间，此外还有一个 [在线 Demo](https://huggingface.co/spaces/NexusBridge/comfyui-workflows-showcase) 展示该套件。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1435536302778093628)** (1 messages): 

> `NLP Data Cleaning` 


- **NLP 新手寻求数据清洗帮助**：一名 NLP 初学者正在寻求关于数据清洗步骤的指导，特别是在处理命名实体识别 (**NER**) 后的原始文本时。
   - 他们请求一套标准的数据清洗流程，并怀疑自己目前的方法是否存在缺陷。
- **有用的 NLP 数据清洗资源**：为了帮助这位新手，社区可以提供教程、博客文章或文档链接，概述 NLP 的标准数据清洗流程。
   - 重点应放在适用于原始文本和 NER 后数据的技术上。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1435444536922079464)** (7 messages): 

> `File retrieval issues, Study group formation` 


- **API 文件检索出现 404 错误**：成员们报告在尝试使用 API 检索关联文件时出现 **404 错误**，具体涉及 `https://agents-course-unit4-scoring.hf.space/files/{task_id}` 端点。
   - 提到了一个具体示例 `99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3`，该示例本应为食谱任务返回一个 **MP3 文件**，但却返回了 404。
- **Agents 课程学习小组组建**：几名成员表示有兴趣为 **Agents 课程** 组建学习小组。
   - 一名在一周前开始学习的成员表示很乐意加入。


  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1435356446073159882)** (3 messages): 

> `Sora App Android, IndQA Benchmark, Interrupt Long Queries` 


- ****Sora** 登陆 Android！**: **Sora app** 现已在加拿大、日本、韩国、台湾、泰国、美国和越南的 Android 平台上架，详见[此视频](https://video.twimg.com/amplify_video/1985765811131465736/vid/avc1/704x1280/iatLt9jMW-vYYuIx.mp4)。
- **AI 通过 **IndQA** 学习印度方言**: OpenAI 推出了 **IndQA**，这是一个评估 AI 系统对印度语言和日常文化背景理解能力的全新 Benchmark，详见[此博客文章](https://openai.com/index/introducing-indqa/)。
- **在中途打断你的机器人！**: 用户现在可以中断长时间运行的查询并添加新的上下文，而无需重启或丢失进度，详见[此视频演示](https://video.twimg.com/amplify_video/1986194201076506628/vid/avc1/3840x2160/rEuDomNqKSd8jEdW.mp4)。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1435292355858141255)** (164 messages🔥🔥): 

> `Sora 2 code, OpenAI's photo generator, Custom GPT photo upload issues, Sora offline?, Models acting weird` 


- **取消订阅 Sora 2 代码诈骗者**: 一名成员表示打算取消订阅某个 YouTube 频道，因为该频道涉及与 **Sora 2 code** 相关的诈骗。
   - 另一名成员随后表示：*如果你不付出任何努力，当然拿不到代码。你的速度不够快。这里的代码在发布后几秒钟内就会被抢光。*
- **OpenAI 的图像生成器需要更新**: 一位用户指出 **OpenAI's photo generator** 需要更新，因为它目前仍然是**基于 2D 文本的**。
   - 他们尝试要求它将物体（如 *在 y 轴上向左旋转 15 度*），但表示无论如何都无法做到。
- **模型表现异常**: 一名成员表示：*过去几周，所有模型（来自所有供应商）的表现似乎都很奇怪，无论是由于功能损坏（如 Anthropic 的 Claude），还是质量下降或令人质疑的公司决策（OpenAI）。这听起来可能很奇怪，但感觉面向公众的 AI 时代正慢慢结束。*
   - 另一名成员回复称，他们*只是偶尔使用 Claude*，但并没有遇到任何问题。
- **GPT-4o 的过度谨慎**: 一位用户抱怨 **GPT-4o 现在变得过度谨慎且过度纠错**，并将其归因于 OpenAI 对诉讼的反应。
   - 另一人回应称，他们*觉得这没什么问题*。
- **LLM 未能通过谜题**: 成员们测试了各种 Large Language Models (**LLM**) 的视觉谜题解决能力。
   - 他们尝试了 **GPT 5 extended thinking** 和 **Gemini 2.5 pro**，但两者都出错了，因为 LLM 无法解决迷宫问题。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1435295803848523797)** (13 messages🔥): 

> `Thinking model degraded, Building ChatGPT apps, OpenAI model comparisons` 


- **Thinking 模型退化了 Chain-of-Thought**: 成员们注意到 **Thinking model** 在过去 5 天内发生了退化，不再进行正常的 **Chain-of-Thought**，而是*“思考”几秒钟却没有任何步骤。*
   - 一位用户评论道：*“最近表现非常糟糕，必须强迫它去真正思考，而不是胡编乱造。”*
- **ChatGPT 应用开发者集结**: 一名成员询问是否有人正在构建 **ChatGPT apps**（仍在 Beta 阶段的 ChatGPT 上的应用）。
   - 一位用户表示，他们直到最近都觉得 **Thinking mode** 非常出色。
- **寻找 OpenAI 模型对比存档**: 一名成员询问是否有网站记录了 **OpenAI 对比其早期模型与后代模型对问题回答的差异。**
   - 另一位用户建议直接要求 **ChatGPT** 模拟旧版模型。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1435339003816579193)** (14 messages🔥): 

> `Prompt Engineering Jobs, GPT Pro Research, Prompt Engineering Tips, Sora 2 Nerf` 


- **Prompt Engineering 职位：真实存在还是神话？**：一位成员在意识到自己在该领域的才华后，询问是否存在 **prompt engineering 职位**。
   - 另一位成员回答说，虽然 *确实有一些公司招聘 prompt engineers，但这并不普遍*，许多人选择构建自己的项目。
- **GPT Pro 在长篇研究中表现挣扎**：一位成员寻求关于如何引导 **GPT Pro** 处理 **50 多页研究资料**的技巧，希望获得与 **Gemini 的 deep research** 能力相当的结果。
- **Prompt Engineering 核心：四步指南**：一位成员分享了他们认为的 prompt engineering 核心，包括选择熟悉的语言、理解期望的输出、清晰地解释任务以及仔细验证结果。
   - 他们建议在处理数学、来源、代码或其他容易出现 **AI hallucination**（幻觉）的细节时要格外小心。
- **分享高级 Prompting 技术**：一位成员分享了高级 prompting 技术，包括**使用 markdown 的层级化沟通**、通过开放变量进行抽象、Prompt 中的**强化（Reinforcement）**以及**用于合规性的 ML 格式匹配**。
   - 他们还提供了一个展示结构的 [Output Template](https://example.com/output-template)。
- **Sora 2 又被削弱（Nerf）了？**：一位成员质疑 **Sora 2** 是否又遭到了 **nerf**。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1435339003816579193)** (14 messages🔥): 

> `Prompt Engineering Job Market, Prompting GPT Pro for Research, Tips for New Prompt Engineers, Sora 2 Nerf` 


- **对 Prompt Engineers 的需求仍存争议**：成员们讨论了公司是否在积极招聘 **prompt engineers**，一些人认为机会有限，许多人正在开发自己的项目。
   - 几位参与者发表了关于他们如何**运用自己的技能**以及这是否可能成为一条**职业路径**的看法。
- **GPT Pro 在处理大型研究 Prompt 时表现挣扎**：一位成员询问关于使用 **GPT Pro 处理 50 多页研究资料**的 prompting 方法，寻求与 Gemini 的 deep research 能力相当的结果。
   - 发帖者指出，两个模型使用**同一种 Prompt** 得到的结果大相径庭。
- **Prompt Engineering 核心原则揭秘**：一位成员分享了他们认为的 prompt engineering 核心：专注于**清晰的沟通**、**准确的语言**和仔细的输出验证，包括**事实核查**和幻觉意识。
   - 另一位成员分享了一份详细指南，包含层级化沟通、通过变量抽象、Prompt 强化和 **ML 格式匹配**等课程。
- **Sora 2 被削弱了？**：一位成员简单地询问 **Sora 2** 是否又遭到了 nerf。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1435283276649463939)** (56 messages🔥🔥): 

> `Anthropic's closed-source approach, Piracy for Media Preservation, IMO Gold AI, llama.cpp contribution` 


- **Anthropic 避开开源：社区不安情绪上升**：成员们对 Anthropic 致力于闭源实践和[弃用政策](https://www.anthropic.com/research/deprecation-commitments)表示不安，担心如果公司倒闭，模型权重（model weights）将会丢失。
   - 一些人建议 Anthropic 应该发布带有附加条件的权重，或允许在 Sagemaker 等平台上部署以确保保存，而一位成员则希望有人能对 Anthropic *上演一出 Miqu 70B* 并泄露权重。
- **盗版保护媒体遗产**：成员们认为，**盗版**在**媒体保存**方面比内容创作者做得更多，并引用了丢失的 pre-VHS 内容以及由于授权问题可能导致未来流媒体专用内容丢失的例子。
   - 一位成员拒绝谴责盗版，并分享道：*“我拒绝谴责盗版是有原因的，尽管我并不严格赞同它。”*
- **IMO 金牌触手可及？微调热度加速**：一位成员分享了[一篇论文](https://arxiv.org/pdf/2511.01846)，表明通过进一步的 **fine-tuning**，**AI 模型**已接近实现 **IMO（国际数学奥林匹克）金牌**水平的表现。
   - 一些人认为这种方法（通过基准测试模型来解决特定问题）与在通用数学上训练模型相比乏善可陈。
- **新贡献者加入 llama.cpp 社区**：一位成员宣布他们在提交补丁后正式成为了 **llama.cpp contributor**。
   - 另一位成员称赞了**新 llama web GUI** 的美学设计。


  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1435293202113495170)** (15 messages🔥): 

> `Gestural Interfaces, Repligate's Loom, Attention is All You Need, Vision Pitching` 


- **基于手势的界面：Loom 式交互**：一位成员分享了他们为 **Repligate 的 Loom 概念** 开发手势界面的尝试，旨在通过让用户“感受”与之交互的智能，使人机交互更加“物理化”。
   - 他们还指出，这种方法可以创造 **perspective parallax**（透视视差），这是一种无需 **VR/XR** 眼镜即可看到 **3D** 效果的视觉错觉，将个人的 **gnosis** 作为交互式叠加层投射到现实中。
- **拥抱手势未来**：一位成员承认需要利用 **gestures**（手势）创造更多东西，但也指出将其与其他 **AI** 追求相结合存在困难。
   - 他们建议，随着 **XR glasses** 变得普及，依赖手势的界面将大受欢迎，并提到未来用户可能会摘下眼镜对着屏幕做手势，而忘记了其实需要眼镜才能看到界面。
- **放弃仓库与 Win9x 美学**：一位成员描述了在 **7-8 年前** 创建的一个 **gestural interface**，预见了 **vibe coding**，其时间早于 **Mediapipe**，并使用了 **Win9x aesthetic**（Win9x 美学）来编写文档。
   - 由于无法筹集到资金，他们感到沮丧并删除了所有仓库以便“继续前进”。
- **"Attention is All You Need!"**：当另一位成员说“不要因为缺乏关注而烦恼”时，一位成员引用了著名的论文 [*Attention is All You Need!*](https://arxiv.org/abs/1706.03762) 幽默地回应。
   - 该成员还表示：“我觉得我在这里找到了志同道合的人，我只需要拿出勇气来宣讲我的愿景。”


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

real.azure: https://github.com/ggml-org/llama.cpp/discussions/16938
  

---


### **tinygrad (George Hotz) ▷ #[announcements](https://discord.com/channels/1068976834382925865/1069236008115253348/1435333626077380690)** (1 messages): 

> `tinybox pro v2, 5090, rackable workstation` 


- **tinybox pro v2 发布：8x 5090 性能怪兽！**：新产品 **tinybox pro v2** 正式发布，在 **5U rackable workstation**（5U 机架式工作站）中搭载了 **8x 5090 GPU**，以实现卓越性能。
   - 该产品可在官网上订购，售价为 **$50,000**，发货周期为 **4-12 周**，承诺提供强大的计算能力。
- **5090 统治 tinybox pro v2**：**tinybox pro v2** 工作站配置了 **8x 5090 GPU**，以处理高要求的计算任务。
   - 这种在 **5U rackable** 规格下的高 GPU 密度，旨在为专业用户提供实质性的处理能力。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1435308643133296772)** (68 messages🔥🔥): 

> `VK_KHR_buffer_device_address GLSL extension, Tinybox Pro V2, AMD vs Nvidia, GLSL Renderer Implementations, Tensor Cores on M1` 


- **`VK_KHR_buffer_device_address` 提升性能**：使用 [`VK_KHR_buffer_device_address`](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VK_KHR_buffer_device_address) 和相应的 GLSL 扩展 `GL_EXT_buffer_reference`，通过在 GLSL 中直接使用指针来实现 float4 等特性，将带来*显著的性能提升*，参考这个 [tinygrad 实现](https://github.com/softcookiepp/tinygrad/blob/master/tinygrad/renderer/glsl.py)。
- **探索 GLSL 渲染器实现**：一位成员分享了[他们的 GLSL 渲染器实现](https://github.com/uuuvn/tinygrad/blob/vulkan/tinygrad/renderer/glsl.py)，并指出它意外地修复了 AGX 编译器 bug，因为它能捕捉到 LLVMpipe 所容忍的 SPIR-V 反汇编中的*无效内容*，包括负索引无符号加法上的整数溢出。
   - 他们将这些问题归咎于 `clspv`（一个基本已停滞的 Google 项目），这也是转向 GLSL 的动力；该成员在 **M1 Asahi Linux** 上进行了测试。
- **Tinybox Pro V2 规格**：George Hotz 正在征求关于 [Tinybox Pro V2](https://x.com/__tinygrad__/status/1985774711499080186) 的反馈（产品链接见 [TinyCorp](https://tinycorp.myshopify.com/products/tinybox-pro-v2)），`comma.ai` 已经是其客户。
   - Tinybox 针对*专业级计算*，目前已有人在上面运行 nanochat 和 deepseek 模型。他们对潜在的 **$10k** AMD 迷你版感兴趣，目标是在 GPU 租赁网站上达到 **$3-4/小时** 的价格。
- **AMD 驱动现在很顺手**：一位成员表示，8x AMD R9700 的配置会比 Tinybox Pro V2 更便宜。
   - George Hotz 回应称 *the rock (ROCm) 使用起来非常愉快*，并声称 **R9700 是在坑钱**，而 **5090 性能要强得多**，尽管另一位成员反驳说 *$50k 太贵了*。
- **M1 芯片：没有真正的 Tensor Cores？**：一位成员指出 **M1 芯片** 可能没有真正的 Tensor Cores，并引用了[一条推文](https://x.com/norpadon/status/1965753199824175543)。
   - 另一位成员建议它们是针对 GEMM 优化的子程序的抽象，Metal 通过 SIMDgroup 和 threadgroup 进行 *shader* 操作，并利用 tile size 优化来防止寄存器溢出（register spill）。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1435375185258741820)** (12 messages🔥): 

> `OU Processes Limitations, ito Diffusions Universality, Paper Dumps Overload, Trending AI Papers` 


- **探讨 OU 过程的假设**：一位成员讨论了如何克服 **OU 过程**（*高斯+马尔可夫+线性+指数自协方差*）的限制性假设，方法包括使用 [Lévy type drivers](https://epubs.siam.org/doi/10.1137/S0040585X97978166)、在 OU 核上积分 (supOU) 以及 [连续 ARMA 过程](https://www.sciencedirect.com/science/article/abs/pii/S0169716101190115)。
- **Ito 扩散实现通用性**：有人指出，如果以*常规*方式训练扩散过程，超越 Ito 扩散 `dX_t = b(X_t)dt + σ(X_t)dW_t` 将一无所获，因为 [Ito 扩散已经是通用的](https://en.wikipedia.org/wiki/It%C3%B4_diffusion)，即任何两个等原子分布（equally atomic distributions）都是相关的。
   - 根据该成员的说法，超越 Ito 扩散只会改变你达到某些分布的*方式*。
- **论文堆积引发讨论**：多位成员表示担心，单个人发布大量随机且可能无关的论文会淹没社区感兴趣的论文。
   - 一位成员建议发布这些论文的个人每天**最多发布两篇论文**，并筛选相关性，而不是发布 *10 篇以上*。
- **热门论文成为启发式指南**：一位成员建议使用热门论文页面作为发布相关论文的启发式指南，例如 [AlphaXiv](https://www.alphaxiv.org/)、[Emergent Mind](https://www.emergentmind.com/) 和 [nlp.elvissaravia.com/t/ai](https://nlp.elvissaravia.com/t/ai)。
   - 另一位成员建议发布论文的人应该用自己的话解释为什么他们认为每篇论文都很重要。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1435341771490988224)** (21 messages🔥): 

> `Anthropic's crosscoder, circuit tracing research, feature evolution during pre-training, leakages from shared chats, latent reasoning` 


- **Cross Coder 讨论开启！**：成员们计划研究[这篇关于 crosscoder 的论文](https://arxiv.org/abs/2509.17196)以及电路追踪研究，以观察**预训练期间特征演化**的不同阶段。
   - 有人提到这篇论文已经在他们的待读清单上了。
- **摧毁 LLM？**：一位成员声称已经摧毁了所有旗舰级 LLM，但质疑他们是否真的能知道真相，还是仅仅被告知了某些信息以使其看起来如此。
   - 他们引用了**共享对话泄露**作为主观或客观证据的例子，并提到了作为“护栏”的*通过策略边界空间的软渗透*。
- **图像重建质量提升？**：一位成员建议简要回顾图像重建的历史，指出自 2020 年以来质量有所下降，但乐观地认为[这篇论文](https://arxiv.org/abs/2510.25976)代表了*向正确方向迈出的一步*。
   - 他们链接了[三篇来自 2023 年的论文](https://arxiv.org/pdf/2303.05334)、[另一篇来自 2023 年的论文](https://arxiv.org/pdf/2306.11536)以及[一篇来自 2022 年的论文](https://arxiv.org/pdf/2211.06956)。
- **潜空间推理尝试**：一位成员分享了[一个关于潜空间推理（latent reasoning）的链接](https://fxtwitter.com/RidgerZhu/status/1983732551404679632)，称其为*又一次尝试*。
   - 另一位成员推荐了[这篇论文](https://arxiv.org/abs/2510.25741)，并表示它看起来像一个 RNN。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1435653602281787432)** (2 messages): 

> `RWKV, HRM/TRM, Context Windows, State Representations` 


- **RWKV 引起关注！**：一位成员在[这个视频](https://youtu.be/LPe6iC73lrc)中了解了 **RWKV** 的最新进展，并觉得它*非常令人印象深刻*。
   - 他建议与他们交流并了解其现状，特别是考虑到 **HRM/TRM** 最近的进展。
- **HRM/TRM 合并令成员兴奋！**：鉴于其快速进展，该成员怀疑 **RWKV** 已经在将 **HRM/TRM** 进行合并。
   - *他们也在积极构建中，这令人兴奋*。
- **现在需要大上下文窗口**：该成员指出，由于庞大的图表尺寸，编程和医学相关工作需要更大的**上下文窗口（context windows）**。
   - 他们表示*遗忘不应该是实际情况*。
- **模型生成多样的状态表示**：在审查了 **HRM** 和 **TRM** 后，发现这些模型在外部循环的不同部分为不同主题创建了不同的状态表示（state representations）。
   - 该成员总结道：*不存在单一的状态空间。它是隐藏空间中根据需要更新的碎片或许多分布式状态*。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1435397501439053865)** (11 messages🔥): 

> `Concentration of power, Erosion of democratic systems, Copyright lawsuit, Getty vs Stability` 


- **权力转移中民主受到质疑**：一位成员批评了他所认为的权力集中和民主制度的侵蚀，引发了辩论，并导致了屏蔽不同意见的建议。
   - 另一位成员认为最初的评论不足以构成屏蔽的理由，强调了参与争议观点讨论的不同门槛。
- **Stability 在针对 Getty 的版权诉讼中获胜**：根据[链接文档](https://drive.google.com/file/d/1vqcQQU8gxGfFA1lUS68BZ8-hrGsu_Flj/view?usp=drivesdk)，**Stability AI** 赢得了针对 **Getty Images** 的版权诉讼。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1435305760174243862)** (35 messages🔥): 

> `Pause and resume optimization runs, Accessing the LLM in a DSPy module, Handling Rate Limits with Fallback LLMs, Conversation History Management in DSPy, Pydantic OutputField Deserialization` 


- **目前没有简单的方法来暂停优化运行**：一名成员请求一种**暂停和恢复优化运行**的方法，并指出[这是一个常见的需求](https://x.com/DSPyOSS/status/1985746487322595341)，但得到的回答是目前除了编写新的适配器（或编辑现有适配器）外，没有简单的方法。
- **揭秘 DSPy 模块中的 LLM 访问**：讨论围绕如何为模块**访问和更改 LLM**展开，重点介绍了 `get_lm()` 和 `set_lm()` 方法的使用。
   - 成员们澄清，访问 DSPy 模块底层的 LLM 对于底层编程至关重要，特别是在遇到 API 限制时。
- **通过备用 LLM 处理速率限制**：对话探讨了**如何通过切换到备用 LLM 来处理速率限制异常**（例如，从 `gpt-4o` 切换到 `gpt-4o-mini`）。
   - 讨论强调，更改 LLM 会重置历史记录，挑战在于如何将模块主 LLM 的历史记录转移到新的备用模型，以便在 ReAct 模块迭代期间保持上下文。
- **对话历史：Signature 揭开谜团**：成员们发现解决方案在于 Signature 内部的 `dspy.History` 对象，而不是模块或 LLM 对象。
   - 通过使用 `dspy.History` 对象，即使在程序执行中途切换 LLM，也可以保持相同的历史记录。
- **需要 Java 版 JSONAdapter 的简化版本**：一名成员询问是否存在 **JSONAdapter 的简化 Java 版本**，以便在 Java 中使用 DSPy 提示词。
   - 目标是在 Java 中构建系统消息并处理 JSON 响应，类似于 DSPy 的管理方式。


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1435655603489083452)** (5 messages): 

> `Synthetic Data Use, Eval Metric, GEPA, Glossary Building` 


- **合成数据对词汇表构建很有用**：一名成员认为 [Synthetic Data](https://x.com/WesEklund/status/1986096335708197102) 对**词汇表构建**用例会很有帮助。
   - 他们发布了一个 [Colab notebook](https://colab.research.google.com/drive/179UlSHSpK-I6H-g4dSgAvuCmPDejxFgm?usp=sharing)，该笔记本需要 **Eval metric** 和 **GEPA** 实现，展示了一个基础的可运行示例。
- **词汇表构建的 Colab 示例**：一名成员分享了一个 [Colab notebook](https://colab.research.google.com/drive/179UlSHSpK-I6H-g4dSgAvuCmPDejxFgm?usp=sharing)，演示了使用合成数据进行**词汇表构建**的基础实现。
   - 该笔记本仍在完善中，需要添加 **Eval metric** 和 **GEPA** (Generative Evaluation of Pattern Articulation) 才能完全发挥功能。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1435296101791174809)** (24 messages🔥): 

> `Kimi iOS app, interleaved thinking model, Kimi CLI, 401 Error` 


- **新的 iOS Kimi K-2 应用被赞为 'Ok Computer'**：新的 Kimi K-2 iOS 应用因其设计受到称赞，特别是 *Ok Computer* 主题，一位用户请求[翻译所有语言](https://discord.com/channels/974519864045756446/977259063052234752/1435599575791570995)，而不仅仅是印度语言。
   - 另一位用户分享了一个[演示文稿](https://3tadkfqfwbshs.ok.kimi.link/present.html)链接。
- **交织思考模型（Interleaved Thinking Model）登陆 Kimi CLI**：Kimi CLI 已添加对**交织思考模型**的支持，引发了关于其与标准 *think...final answer* 方法区别的提问。
   - 原作者备注道：*他们昨天在 kimi-cli 中添加了对思考模式的支持。👀*
- **用户在设置 CLI 时遇到 401 错误**：一位用户报告在尝试设置 Kimi CLI 时收到 **401 错误**，尽管已经充值且有余额；另一名成员指出问题在于**余额是针对 Moonshot AI 开放平台**的，而不是 Kimi K-2 平台。
   - 该用户被引导至正确的频道寻求帮助。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1435300860413476926)** (8 messages🔥): 

> `模型测试配置，使用 Perplexity API，aider-ce 文档` 


- **模型测试：坚持使用默认设置**：一位成员询问在测试模型时是否应该禁用/启用某些选项，另一位成员建议*选择一个好的模型并坚持使用默认设置，直到你有理由更改为止*。
   - 关于此话题没有进一步的讨论。
- **将 Perplexity API 集成到 aider**：一位用户请求提供一个将 **Perplexity API** 与 aider 配合使用的教程，类似于现有的 Gemini 教程，并引用了 [aider 文档](https://aider.chat/docs/llms/other.html#other-api-key-variables)。
   - 另一位成员建议在现有指令中将 *gemini* 替换为 *perplexity*，并将 API key 设置为环境变量，使用 `export PERPLEXITYAI_API_KEY=your-api-key`，然后尝试 `aider --list-models perplexity`。
- **OpenRouter 更适合配合 Aider 进行编码**：一位成员建议，对于编码任务，**OpenRouter** 可能比 **Perplexity API** 是更好的选择，并强调了 OpenRouter 提供了许多免费且强大的模型用于测试 aider。
   - 他们提到你可以尝试 **OpenRouter**，因为它*对于测试 aider 来说既强大又免费*。
- **Aider-CE 文档**：一位成员询问该频道是否也讨论 **aider-ce**，以及 **aider-ce** 是否有文档。
   - 关于此话题没有进一步的讨论。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1435365577211383890)** (9 messages🔥): 

> `使用 Aider 进行 TDD，Ollama 的内存限制，Claude Code 的 /compact 命令，scalafix-rules 总结` 


- **使用 aider 的 TDD 循环？**：一位用户正寻求使用 **aider** 创建一个循环，其中 Agent 1 使用 **TDD** 执行 Prompt，Agent 2 进行评审并建议改进，Agent 1 执行改进，Agent 2 运行 TDD 测试、修复 Bug 并提议 Commit 信息。
   - 建议使用 [脚本编写 (scripting)](https://aider.chat/docs/scripting.html) 来封装 `aider` 的功能。
- **Ollama 和 scalafix-rules 的内存问题**：一位用户想要总结 [scalafix-rules](https://github.com/xuwei-k/scalafix-rules/tree/main/rules/src/main/scala/fix) 项目中的规则，但在使用 **Ollama** 时遇到了内存限制。
   - 他们希望避免一次性加载所有规则，而是逐个处理，为每个规则生成包含名称、摘要和转换示例的表格条目，同时卸载之前的规则以节省内存。
- **Aider 缺失的 `/compact` 命令**：一位用户询问 **Aider** 是否有类似于 **Claude Code** 的 `/compact` 命令，该命令可以总结并压缩对话历史以防止上下文丢失。
   - 建议要求模型将对话总结到 `status_report.md` 中，并配合使用 `clear`、`drop` 和 `reset` 命令。
- **针对简单问题的云端模型**：针对内存问题的总结任务，一位成员建议使用云端模型。
   - 他们认为对于像 **qwen 30b** 这样的本地模型，为每个规则生成简短描述并不难，但一次性处理所有规则时就会出现问题，会迅速耗尽内存。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1435297152980090922)** (17 messages🔥): 

> `IETF 124, IETF 中的 MCP 讨论, AI 爬虫 (AI Scrapers)` 


- **开始渴望 IETF 频道**：成员们正在考虑本周为参加 **IETF 124** 的 **MCP** 成员创建一个临时频道，类似于开发者峰会的频道。
   - 大家一致同意创建频道的建议，并提议为*一般的 IETF 会议*创建一个频道组。
- **活动 (Events) 类别诞生**：有人提议创建一个 **events** 类别，并为参与者达到一定规模的活动创建频道。
   - 一些成员引用了今天早些时候关于 **IETF** 可能采用 **MCP/A2A** 传输协议的演讲，以及其他 IETF 小组（如 HTTPAPI）可能与 **MCP** 相关的事实。
- **AI 爬虫引发侧边会议**：一些成员注意到，本次会议目前的讨论根本不是关于 **OAuth** 的，而是关于通用的 **AI 抓取/爬虫 (AI scraping/crawlers)**。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1435338628699127829)** (4 messages): 

> `IFeval Scores, Latent Reasoning` 


- **IFeval 分数研究**：成员们讨论了 [IFeval 分数](https://openreview.net/forum?id=Q7mLKxQ8qk)及其在 Prompt 级别/指令级别的差异。
   - 一位成员澄清说，他们使用的是 [inspect_evals 仓库](http://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/ifeval#final-accuracy)中所描述的**所有分数的平均值**。
- **尝试隐式推理 (Latent Reasoning)**：一位成员分享了对隐式推理的尝试，并附上了推文和 Arxiv 论文链接：[Latent Reasoning 推文](https://fxtwitter.com/RidgerZhu/status/1983732551404679632) 和 [Latent Reasoning 论文](https://arxiv.org/abs/2510.25741)。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1435466292038467674)** (12 messages🔥): 

> `Concept Detection System, Equivalent Linear Mappings of LLMs, Tangent Model Composition, Jacobian for LLM Interpretability` 


- **通过低成本概念检测实时引导概念**：一位成员构建了一个系统，可以在模型生成时，以低成本实时**检测并引导数千个概念**。
   - 目前的问题在于应该检测哪些概念来有效地细分语义空间；他们正在寻求关于可解释性的**有限且可扩展的概念表示**的现有研究。
- **用于 LLM 可解释性的等效线性映射**：一位成员发表了论文 [“Equivalent Linear Mappings of Large Language Models”](https://openreview.net/forum?id=oDWbJsIuEp)，证明了 **LLM 对于任何给定的输入序列，其推理操作都具有等效的线性表示**。
   - 他们利用**线性表示的 SVD** 来寻找低维、可解释的语义结构，这些结构可用于引导，在层/块级别具有模块化特性，并适用于 Qwen 3, Gemma 3, Llama 3, Phi 4, OLMo 2 和 Mistral 等模型。
- **切线模型组合 (Tangent Model Composition) 的相关性**：一位成员询问了 [Tangent Model Composition for Ensembling and Continual Fine-tuning](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Tangent_Model_Composition_for_Ensembling_and_Continual_Fine-tuning_ICCV_2023_paper.pdf) 的相关性。
   - 原作者回答说，**Tangent Model Composition** 关注的是**权重/参数空间中的切线和泰勒展开**，而他们的工作关注的是**输入嵌入 (Embedding) 空间中的 Jacobian**。
- **图像模型的 Jacobian 直觉**：一位成员分享了来自 Zara Khadkhodaie 和 Eero Simoncelli 的图像模型论文（[论文 1](https://iclr.cc/virtual/2024/oral/19783)，[论文 2](https://arxiv.org/abs/2310.02557)，[论文 3](https://arxiv.org/abs/1906.05478)），这些论文可以在推理时计算**常规 autograd Jacobian** 以实现精确重构。
   - 原作者很赞同关于 **LLM 在低维子空间中运行**（每个输入都有自己的子空间）的描述，但也认为为了实现可解释性，需要某种工具来桥接不同的输入或模型。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1435302534297292801)** (16 messages🔥): 

> `Text to video tools, Webscraping Twitter/X, Manus Support, Host services for Manus apps, Publishing problems on Manus` 


- **文本转视频工具探索开始**：一位成员询问了实用的**文本转视频工具**，引发了关于可用选项的讨论。
- **无需 API 的 X 抓取技巧**：一位成员寻求关于**在不使用 API 的情况下抓取 Twitter/X 网页数据**的建议，理由是他们目前使用的基于 cookie 的 Python 库维护困难。
- **Manus 托管服务**：一位成员征求运行由 Manus Dev 创建的应用的**托管服务**建议，并指出 Manus 不适合 24/7 的商业设置，另一位用户表示 *Vercel 挺好用的*。
- **平台发布问题困扰**：成员们报告了**发布项目**时的问题，指出更新未能反映最新的 Checkpoints。
- **Manus 到 GitHub 的迁移方法**：在经历了一些无法解决的错误和项目挫折后，一位成员询问了将**项目从 Manus 迁移到 GitHub** 的最佳方法。


  

---

### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1435336696106455212)** (1 条消息): 

> `Codemaps, SWE-1.5, Sonnet 4.5` 


- **Windsurf 推出 Codemaps！**: Windsurf 推出了由 **SWE-1.5** 和 **Sonnet 4.5** 驱动的 **Codemaps**，旨在通过增强你对所处理代码的理解能力，帮助提升生产产出。
   - 根据 Windsurf 的说法，*无论手动编写还是使用 Agent，限制你编码能力的最大约束是你对所处理代码的理解能力* —— 更多信息请见其 [X post](https://x.com/windsurf/status/1985757575745593459)。
- **理解代码是关键**: Windsurf 强调，理解你正在处理的代码对于高效编码至关重要，无论是手动完成还是通过 AI Agent 完成。
   - Windsurf 引用 Paul Graham 的话指出，*“你的代码就是你对正在探索的问题的理解”*，强调了牢牢掌握代码的重要性。


  

---


---