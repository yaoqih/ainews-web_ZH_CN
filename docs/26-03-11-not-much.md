---
companies:
- nvidia
- perplexity
- replit
- base44
- vllm
- llama.cpp
- ollama
- togethercompute
- baseten
- wandb
- langchain
- unsloth
date: '2026-03-11T05:44:39.731046Z'
description: '**NVIDIA 的 Nemotron 3 Super** 是一款拥有 **1200 亿（120B）参数/约 120 亿（12B）激活参数**的开放模型。它采用了
  **Mamba-Transformer 混合架构及 SSM Latent MoE（状态空间模型潜变量混合专家）架构**，并具备 **100 万（1M）上下文窗口**。在
  FP4 精度下，其推理速度比 GPT-OSS-120B 快达 **2.2 倍**，且吞吐量增益显著。该模型支持**智能体（agentic）工作负载**，且开放程度极高，公开了权重、数据和基础设施细节。模型在
  **AA 智能指数（AA Intelligence Index）中得分为 36**，优于 GPT-OSS-120B，但次于 Qwen3.5-122B-A10B。来自
  **vLLM、llama.cpp、Ollama、Together、Baseten、W&B Inference、LangChain** 以及 **Unsloth
  GGUFs** 等项目和社区的基础设施支持已迅速跟进。其关键技术创新包括**原生多 Token 预测（MTP）**和显著的 **KV 缓存（KV-cache）效率**优势。


  在产品层面，行业重点正转向**持久化智能体运行时（runtimes）和编排层**。**Andrej Karpathy** 提倡“大 IDE”概念，即由智能体取代文件作为工作单位，从而实现具备实时控制能力、清晰且可分叉（forkable）的智能体组织。符合这一愿景的新产品包括：**Perplexity
  的 Personal Computer**（一种在 Mac mini 上运行的常驻本地/云端混合体），以及可编排 20 个专业模型和 400 多个应用程序的 **Computer
  for Enterprise**。**Replit Agent 4** 提供了支持并行智能体的画布式协作工作流，而 **Base44 Superagents**
  则为非技术用户提供集成解决方案。目前的工程重心正日益从单一模型转向**编排框架（orchestration harness）**。'
id: MjAyNS0x
models:
- nemotron-3-super
- gpt-oss-120b
- qwen3.5-122b-a10b
people:
- karpathy
- ctnzr
- bnjmn_marie
- artificialanlys
title: 今天没什么事。
topics:
- model-architecture
- model-optimization
- inference-speed
- kv-cache
- multi-token-prediction
- agent-infrastructure
- orchestration
- persistent-agents
- model-serving
- product-launches
---

*平静的一天。*

> 2026/3/10-2026/3/11 的 AI 新闻。我们检查了 12 个 Reddit 子版块、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有新增 Discord 信息。[AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[选择订阅或退订](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件发送频率！


---

# AI Twitter 摘要


**NVIDIA 的 Nemotron 3 Super 发布与开源模型效率提升**

- **Nemotron 3 Super** 是当天最明确的技术发布：一个 **120B 参数 / ~12B 激活** 的开源模型，拥有 **1M 上下文**，采用 **混合 Mamba-Transformer / SSM Latent MoE** 架构，并明确支持 Agent 工作负载。NVIDIA 将其定位为异常开放——包括**权重、数据、训练方案（recipe）、基础设施细节**——并且专注于 Blackwell 时代的部署性能。官方声称在 **FP4 精度下其推理速度比 GPT-OSS-120B 快 2.2 倍**，且吞吐量比之前的 Nemotron 版本有大幅提升（[通过 @ctnzr 发布公告](https://x.com/ctnzr/status/2031762077325406428)，[通过 @kuchaev 了解技术视角](https://x.com/kuchaev/status/2031765052970393805)，[Wired 关于 NVIDIA 广泛开源模型投资的报道](https://x.com/willknight/status/2031792027390587313)）。
- 第三方反应集中在同一主题：**强大的“每个激活参数能力”和异常高的推理服务速度**。[@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2031765321233908121) 在 AA 智能指数中给它打出了 **36 分**，领先于 **gpt-oss-120b (33)**，但落后于 **Qwen3.5-122B-A10B (42)**；同时指出其**每张 GPU 的吞吐量比 GPT-OSS-120B 高出约 10%**，发布首日的推理速度达到 **484 tok/s**。社区和基础设施支持立即在 [vLLM](https://x.com/vllm_project/status/2031779213527957732)、[llama.cpp](https://x.com/ggerganov/status/2031819920363733205)、[Ollama](https://x.com/ollama/status/2031777869681000676)、[Together](https://x.com/togethercompute/status/2031831368339243454)、[Baseten](https://x.com/basetenco/status/2031775755253026965)、[W&B Inference](https://x.com/wandb/status/2031778471614300563)、[LangChain](https://x.com/LangChain/status/2031784791251525934) 以及 [Unsloth GGUFs](https://x.com/UnslothAI/status/2031778104306499749) 同步上线。
- 最有趣的技术讨论关于 **它为什么这么快**。[@ctnzr](https://x.com/ctnzr/status/2031776463029186920) 强调 **原生多 Token 预测 (MTP)** 是关键的推理优化手段：临时的多 Token 猜测会在后续传递中得到验证，从而在小批次（batch sizes）情况下利用原本闲置的 GPU 计算资源。[@bnjmn_marie](https://x.com/bnjmn_marie/status/2031821490916905089) 还量化了其相对于 Qwen3.5-122B 的重大 **KV-cache 优势**：Nemotron 的注意力 KV 项在 **BF16 精度下约为 8,192 字节/token**，而 Qwen3.5-122B 则为 **24,576 字节/token**，这使得长上下文推理服务显著变轻。

**Agent 基础设施、编排以及“更大的 IDE”论点**

- 最强劲的产品趋势是从“与模型聊天”转向**持久化的 Agent 运行时和编排层**。[@karpathy](https://x.com/karpathy/status/2031767720933634100) 认为“IDE 时代已结束”的说法是错误的；相反，**“我们需要一个更强大的 IDE”**，其中的工作单元变成了 **Agent 而不是文件**，他随后将这一概念延伸为具有实时可观测性和控制能力的**可读、可分叉（forkable）的 Agent 组织**（[后续](https://x.com/karpathy/status/2031770607466291393)，[组织可读性讨论串](https://x.com/karpathy/status/2031774631498273005)）。
- 多个发布的产品都符合这一框架。**Perplexity** 发布了 **Personal Computer**，这是一个在 **Mac mini** 上运行的**常驻本地/云端混合体**，可跨本地文件/应用/会话工作，并支持远程控制（[发布](https://x.com/perplexity_ai/status/2031790180521427166)，[候补名单](https://x.com/perplexity_ai/status/2031790221612957875)）。它还扩展了 **Computer for Enterprise**，描述了跨 **20 个专用模型**和 **400 多个应用**的编排（[企业版发布](https://x.com/perplexity_ai/status/2031799033489211771)，[API 平台更新](https://x.com/perplexity_ai/status/2031828396435771563)）。另外，**Replit Agent 4** 推出了更具协作性、画布式的（canvas-like）工作流，支持用于应用、网站和幻灯片的**并行 Agent**（[发布](https://x.com/amasad/status/2031755113694679094)），而 **Base44 Superagents** 则为非技术用户强调了与 Gmail、Slack、Stripe、CRM 等工具的“内置（batteries included）”集成（[发布](https://x.com/MS_BASE44/status/2031758998475505848)）。
- 工程讨论越来越多地围绕 **Harness**（评估/执行框架）而非仅仅是模型展开。[@Vtrivedy10](https://x.com/Vtrivedy10/status/2031751769051570256) 描述了一个快速变化的设计空间，其中改进后的模型解锁了此前过于脆弱的产品体验，形成了一个**evals/metrics → 自主 Harness 编辑 → 爬山算法（hill climbing）**的自我改进循环。LangChain 在 Deep Agents 中加入了**自主上下文压缩**功能，使模型可以在任务边界进行压缩，而非使用硬性的 Token 阈值（[公告](https://x.com/LangChain_OSS/status/2031799813851730075)），而 [@OpenAIDevs](https://x.com/OpenAIDevs/status/2031798071345234193) 发布了一份关于 **Agent 的计算机访问权限**的技术报告，涵盖了执行循环、文件系统上下文、网络访问以及护栏（guardrails）。

**Anthropic、以 Claude 为中心的工作流以及早期的 RSI 焦虑**

- 一个主要的元叙事是 **Anthropic 对强大 AI 的机构化架构**。该公司成立了 **The Anthropic Institute**，由 **Jack Clark** 担任新设立的**公共利益负责人（Head of Public Benefit）**，其职责涵盖 ML 工程、经济学和社会科学，旨在引导围绕先进 AI 的公共对话（[发布](https://x.com/AnthropicAI/status/2031674087374815577)，[领导层说明](https://x.com/AnthropicAI/status/2031674092290474421)，[Jack Clark 谈角色变更](https://x.com/jackclarkSF/status/2031746605117010245)）。
- 与此同时，多条推文放大了人们的担忧，即 Anthropic 内部可能已经出现了**早期递归自我改进（recursive-self-improvement, RSI）动态**。最实质性的引用间接来自于对《时代周刊》（**TIME**）一篇文章的讨论：[@kimmonismus](https://x.com/kimmonismus/status/2031803194817511744) 总结了其中的说法，称**开发未来模型的代码中有 70–90% 现在是由 Claude 编写的**，模型发布节奏已从几个月缩短至**几周**，一些研究人员认为**全自动化 AI 研究可能在短短一年内实现**。[@Hangsiin](https://x.com/Hangsiin/status/2031752106496135541) 强调了其中特别引人注目的一点：在某些内部任务中，Claude 比**人类监督者快 427 倍**，且嵌套并行使用模式已非常普遍。
- 这一叙事有一个直接的现实对立面：**对 Claude Code 的操作依赖**。一次登录/认证故障引发了明显的开发者痛苦，[@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2031777214321262637) 开玩笑说硅谷的生产力下降了 90%，[@dejavucoder](https://x.com/dejavucoder/status/2031760986907312635) 报告无法登录，[@HamelHusain](https://x.com/HamelHusain/status/2031783246980399375) 描述了不得不退回到基于 Token 的访问方式。这次停机甚至促使 [@karpathy](https://x.com/karpathy/status/2031792523187040643) 注意到他的**自动研究实验室在 OAuth 故障中全军覆没**，并将未来前沿模型服务的中断定性为潜在的**“智能电压不稳（intelligence brownouts）”**。

**关于 Agent Evals、检索、Post-Training 以及自我改进的研究**

- 几篇论文关注了看起来像是下一个瓶颈的问题：**衡量和改进 Agent 系统**，而不仅仅是基础模型的质量。[@karinanguyen_](https://x.com/karinanguyen_/status/2031789998811595154) 发布了 **PostTrainBench v1.0**，这是一个用于测试前沿 Agent 能否在**简化环境下对语言模型进行后训练 (post-train)** 的基准测试，明确旨在追踪 **AI 研发自动化 / 递归自我改进 (recursive self-improvement)** 的进展。该推文中一个值得注意的消融实验结果是：对于 **GPT-5.1 Codex Max**，**中等推理算力投入优于高等投入**，因为额外的 Token 会导致上下文压缩并损害性能（[消融实验细节](https://x.com/karinanguyen_/status/2031790007028236452)）。
- 在 Agent 学习方面，[@omarsar0](https://x.com/omarsar0/status/2031727864199208972) 强调了 **EvoSkill**，其中执行器/提议器/技能构建器（executor/proposer/skill-builder）三元组能够从失败中发现并完善可复用的技能；据报道，在 OfficeQA 上，它将 **Claude Code + Opus 4.5 的精确匹配率（exact match）从 60.6% 提升到了 67.9%**。[@dair_ai](https://x.com/dair_ai/status/2031726356292407366) 分享了 **AgentIR**，这是一种**具备推理感知能力的检索器 (reasoning-aware retriever)**，它将 Agent 的推理轨迹与其查询联合嵌入；他们报告在 **BrowseComp-Plus 上达到了 68% 的准确率**，而大型传统嵌入模型为 **52%**，BM25 为 **37%**。
- 还有人重新强调了 **Agent 的可靠性即使在没有对抗者的情况下也是一个安全问题**。[@random_walker](https://x.com/random_walker/status/2031693490669654447) 认为，许多 AI Agent 的失败源于不可靠性而非明确的攻击，并指出普林斯顿大学对 NIST 的回应中提到需要定义、衡量并缓解这种失效模式。结合对评估技术日益增长的重视——例如 [@gabriberton](https://x.com/gabriberton/status/2031653520429203498) 称评估创建是代码 Agent 时代最有用的技能——重心正不断向**衡量、测试框架 (harnesses) 和生产反馈回路**转移。

**多模态模型、嵌入与物理/视觉 AI**

- 在多模态方面，**Google 的 Gemini Embedding 2** 引发的是实际定价分析而非基准测试讨论。[@osanseviero](https://x.com/osanseviero/status/2031691784074477766) 总结了这次发布：涵盖 **文本、图像、视频、音频、PDF** 的嵌入，以及用于低维存储的 **Matryoshka embeddings (俄罗斯套娃式嵌入)**。[@neural_avb](https://x.com/neural_avb/status/2031648857625395321) 提供了最有用的部署笔记：**文本定价相对于竞争对手显得偏高**，建议该模型最好保留用于**多模态检索**；除非客户在上传前积极降低 **FPS**，否则视频嵌入成本可能会激增。
- **Qwen3.5 的多模态架构** 也得到了来自 [@ZhihuFrontier](https://x.com/ZhihuFrontier/status/2031686944040915152) 的详细社区解析：一个混合了 **Gated DeltaNet linear attention (门控 DeltaNet 线性注意力)** 和 **Gated full attention (门控全注意力)** 的 **混合注意力 (hybrid attention)** 堆栈，包含 **397B A17B MoE** 变体和 **27B dense (稠密)** 变体，具有 **262k 原生上下文** 并可扩展至 **1M**，且在训练中使用了 **MTP**。该推文的主要用途是作为注意力机制创新的紧凑调研：**混合线性/全注意力、GQA、DSA 和 MoE 路由** 现已成为核心设计轴心。
- 在视觉/物理 AI 领域，**Reka Edge** 作为面向生产环境的 VLM 发布，用于物理 AI，声称在图像/视频理解、目标检测和工具使用方面，比领先的 8B 模型**减少了 3 倍的输入 Token** 且**吞吐量快了 65%**（[发布信息](https://x.com/RekaAILabs/status/2031781818349834628)）。Google 也分享了两个医疗保健领域的应用：一个 AI 系统识别出了标准筛查遗漏的 **25% 的间隔期乳腺癌** ([Google](https://x.com/Google/status/2031734020979998795))；以及一项关于 **AMIE** 用于对话式临床推理的现实世界研究，发现其安全、可行且受到患者好评 ([Google Research](https://x.com/GoogleResearch/status/2031777657835139263))。

**热门推文（按互动量排序）**

- **Perplexity 的“个人电脑”**：运行在 Mac mini 上的全天候在线本地/云端 Agent，支持远程控制及本地应用/文件访问 ([发布](https://x.com/perplexity_ai/status/2031790180521427166))。  
- **Anthropic Institute / Jack Clark 的新角色**：Anthropic 正式发起了一项围绕强大 AI 的公益与公共话语行动 ([Anthropic](https://x.com/AnthropicAI/status/2031674087374815577), [@jackclarkSF](https://x.com/jackclarkSF/status/2031746605117010245))。  
- **Replit Agent 4**：用于发布应用/网站/幻灯片的协作式、多 Agent 画布 (canvas) ([公告](https://x.com/amasad/status/2031755113694679094))。  
- **NVIDIA Nemotron 3 Super**：具有 1M 上下文和首日生态支持的开源 120B/12B 激活混合模型 ([@ctnzr](https://x.com/ctnzr/status/2031762077325406428))。  
- **Claude Code 停机引发的基础设施风险**：前沿模型鉴权失败显著干扰了实际工程工作流 ([@karpathy](https://x.com/karpathy/status/2031792523187040643), [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2031777214321262637))。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen 模型发布与基准测试

  - **[M5 Max just arrived - benchmarks incoming](https://www.reddit.com/r/LocalLLaMA/comments/1rqnpvj/m5_max_just_arrived_benchmarks_incoming/)** (热度: 2188): **该帖子讨论了 M5 Max 128GB 14 英寸笔记本电脑的到货及其基准测试情况，重点是使用 `mlx_lm` 工具测试各种机器学习模型。测试的模型包括 Qwen3.5-122B-A10B-4bit, Qwen3-Coder-Next-8bit, Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit 以及 gpt-oss-120b-MXFP4-Q8。基准测试揭示了针对不同 Prompt 长度的 tokens-per-second 和峰值显存占用等性能指标。作者最初在使用 BatchGenerator 时遇到问题，但通过使用全新的 Python 环境和 `stream_generate` 解决了这些问题。结果显示不同模型的性能各异，峰值显存占用范围从 25.319 GB 到 92.605 GB，生成速度从 14.225 到 87.873 tokens-per-second。** 评论者们对基准测试结果充满期待，其中一位对 Qwen 3.5 27b MLX 模型的性能表现出了浓厚兴趣。另一位评论者则幽默地表达了对基准测试的期待之情。

    - 使用 `mlx_lm.generate` 在 M5 Max 128GB 14 英寸上进行的基准测试显示，不同模型和配置下的性能差异很大。例如，**Qwen3.5-122B-A10B-4bit** 模型在 16K 上下文下的 Prompt 吞吐量达到 `1,239.7 t/s`，峰值显存占用为 `73.8 GB`。相比之下，**Qwen3-Coder-Next-8bit** 模型在 32K 上下文下达到 `1,887.2 t/s`，但显存消耗更高，为 `89.7 GB`。
    - **Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit** 模型的生成吞吐量显著下降，在 32K 上下文下仅为 `14.9 t/s`，峰值显存占用为 `30.0 GB`。这表明在模型复杂性和性能之间存在权衡，因为经过蒸馏（distilled）的模型虽然可能需要更少的显存，但也会导致吞吐量降低。
    - **gpt-oss-120b-MXFP4-Q8** 模型表现出了令人印象深刻的性能，在 16K 上下文下 Prompt 吞吐量为 `2,710.5 t/s`，且峰值显存占用相对较低，为 `64.9 GB`。这表明该模型针对高吞吐量进行了优化，同时保持了高效的显存利用率，非常适合需要快速处理速度的应用。

  - **[Qwen3.5-35B-A3B Uncensored (Aggressive) — GGUF Release](https://www.reddit.com/r/LocalLLaMA/comments/1rq7jtm/qwen3535ba3b_uncensored_aggressive_gguf_release/)** (热度: 1019): **发布在 [Hugging Face](https://huggingface.co/HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive) 上的 **Qwen3.5-35B-A3B Aggressive** 因其去审查（uncensored）特性而备受关注，它在保持原始模型能力的同时做到了零拒答（`0/465 refusals`）。该模型拥有 `35B` 参数，其中活跃参数约 `3B`，采用混合专家（MoE）架构，包含 `256 experts` 且每个 Token 激活 `8+1` 个专家。它支持多模态输入（文本、图像、视频），并采用混合注意力机制（Gated DeltaNet + softmax 比例为 `3:1`）。该模型包含多种量化格式，如 `BF16`, `Q8_0` 和 `Q6_K` ，并通过 `mmproj` 针对视觉支持进行了优化。推荐的采样参数包括 `temp=1.0`, `top_k=20` 和 `presence_penalty=1.5`。建议用户在使用 `llama.cpp` 时添加 `--jinja` 标志以获得最佳性能。** 社区对此次发布表示赞赏，用户们对开发者的努力表示感谢，并期待在 `Q4_K_M` 等所有组件可用后尝试该模型。

    - Velocita84 提出了一个关键点，即需要评估 KL 散度（Kullback-Leibler Divergence, KLD）来证实 Qwen3.5-35B-A3B 模型“无能力损失”的说法。这一指标对于量化原始模型与修改后模型概率分布之间的差异至关重要，从而确保激进的去审查不会降低模型性能。
    - Iory1998 强调了对潜在质量退化的担忧，特别是在处理长上下文场景时。这是大型语言模型（LLM）的常见问题，即激进的去审查等修改可能会影响模型在处理长文本输入时保持连贯性和准确性的能力。该评论者询问了修改后的模型在这些方面与原始模型的对比情况。
    - No-Statistician-374 提到了对该模型 Q4_K_M 版本的期待，这表明社区对不同量化格式感兴趣。这反映了用户热衷于探索各种配置以优化性能和资源利用率，体现了技术社区对平衡模型大小和计算效率的关注。

### 2. 开源 TTS 与语音模型

  - **[Fish Audio 发布 S2：开源、可控且极具表现力的 TTS 模型](https://www.reddit.com/r/LocalLLaMA/comments/1rptdpl/fish_audio_releases_s2_opensource_controllable/)** (热度: 362): **Fish Audio** 发布了 **S2**，这是一个新的开源 TTS 模型，允许使用自然语言情感标签（如 `[whispers sweetly]` 或 `[laughing nervously]`）进行高度表现力和可控的语音合成。该模型支持超过 `80 种语言`，能够在单次推理中生成多说话人对话，并实现了 `100ms` 的首字音频延迟（time-to-first-audio）。据报道，S2 在语音图灵测试（Audio Turing Test）和 EmergentTTS-Eval 中超越了 **Google** 和 **OpenAI** 的闭源模型。该模型和代码已在 [Hugging Face](https://huggingface.co/fishaudio/s2-pro) 和 [GitHub](https://github.com/fishaudio/fish-speech) 上提供，但商业用途需要单独的许可证。关于该模型的开源状态存在争议，因为其许可证限制了未经单独协议的商业使用。创始人承认发布较为仓促，并为用户提供了额外资源，包括 GitHub 仓库和性能基准测试。

    - Fish Audio 的 S2 模型并非完全开源，因为尽管它可用于研究和非商业目的，但商业使用需要单独的许可证。模型可在 [Hugging Face](https://huggingface.co/fishaudio/s2-pro) 获取，代码已在 [GitHub](https://github.com/fishaudio/fish-speech) 上线，不过仍在不断完善中。
    - Fish Audio 的创始人提到，在 H200 设备上使用 fish-speech 仓库，S2 模型可以达到约 `130 tokens per second`，并且通过 SGLang 有可能实现更高的并发量。这表明对于对高吞吐量 TTS 应用感兴趣的用户来说，该模型具有显著的性能潜力。
    - S2 模型支持广泛的语言并提供高质量的输出，包含 `[angry]` 或 `[laughing]` 等表现力标签，增强了其生成细腻语音的实用性。这一特性使得它对于需要高质量非英语 TTS 解决方案的用户特别有价值。


### 3. LocalLLaMA 与模型运行经验

  - **[我后悔发现了 LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1rq8ijl/i_regret_ever_finding_localllama/)** (热度: 1408): **该帖子幽默地描述了从将 AI 作为学习辅助工具到深度参与 **LocalLLaMA** 等本地 LLM 的历程。** 用户详细说明了从使用 AI 处理简单任务到涉及 **MI50 GPU**、量化和自定义矩阵的复杂实现的演变过程。他们提到了在等待 **GLM flash** 和 **Qwen** 模型等技术进展，表明其正深陷于优化本地 AI 性能的研究中。该帖子反映了从实际应用向发烧友对本地 AI 技术痴迷的转变。一位来自大型 AI 公司的评论者指出，本地 AI 在工程圈之外尚未得到广泛认可，并将其潜在影响比作计算领域的 Linux。另一位评论者将对本地 AI 的痴迷视为一种积极的成瘾，强调了知识的价值。


  - **[100 万 LocalLLaMA 成员](https://www.reddit.com/r/LocalLLaMA/comments/1rqcsrj/1_million_localllamas/)** (热度: 430): **图片展示了“LocalLlama”子版块的飞速增长，该版块专注于讨论可本地托管的 AI 模型。** 该子版块创建于 2023 年 3 月，迅速积累了 100 万成员，反映了社区极高的兴趣和参与度。考虑到该子版块成立不到一年，这种增长非常显著，表明了对本地 AI 托管解决方案的强烈且活跃的兴趣。一条评论回顾了社区的韧性，提到了过去在版主管理不稳定方面的挑战；而另一条评论则表达了对替代性 AI 传说的偏好，暗示一些成员正在寻求不同的主题方向。



## 非技术类 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI 模型与基准测试进展

- **[Anthropic: Recursive Self Improvement Is Here. The Most Disruptive Company In The World.](https://www.reddit.com/r/singularity/comments/1rqymbn/anthropic_recursive_self_improvement_is_here_the/)** (热度: 1141): ****Anthropic** 据报道正在通过其模型 **Claude** 加速 AI 开发，Claude 为未来模型编写了 `70% 到 90%` 的代码，这暗示着向 *递归自我改进 (recursive self-improvement)* 的转变。来自 Anthropic 的 **Evan Hubinger** 声称这种现象已经存在，并有可能在一年内实现全自动 AI 研究。**Claude 3.7 Sonnet** 的发布因安全顾虑推迟了 10 天，凸显了该公司谨慎的态度。**Dario Amodei** 警告说，AI 有可能在五年内取代一半的初级白领工作，并敦促对这些影响保持透明。文章还提到了 Anthropic 在军事背景下部署 AI 的立场，以及对政治影响 AI 政策的批评。** 一些评论者对安全延迟的批评表示质疑，认为既然 `90%` 的代码不是由人类编写的，那么彻底的测试对于确保安全至关重要。这场辩论反映了在平衡 AI 快速发展与伦理及安全考量方面的担忧。

    - Substantial-Elk4531 提出了一个关键点，即对 AI 模型进行安全性测试的必要性，特别是当大部分代码并非人类编写时。这突显了严格测试协议的重要性，以确保 AI 系统的安全性和可靠性，由于其自主性，这些系统可能变得复杂且不可预测。
    - BiasHyperion784 讨论了 Anthropic 基础设施升级的时间表，指出到 2027 年第三季度，将部署新硬件（'rubin ultra'），从而增强计算能力。这表明当前训练时间的改进至关重要，因为它们将被即将到来的定制硬件放大，预示着通过扩展计算能力来提升 AI 能力的战略重点。
    - Unethical_Gopher_236 引用了历史上对 AI 安全的担忧，将其与 **Ilya Sutskever** 认为 GPT-2 太危险而不宜发布的情况进行了类比。这一评论强调了 AI 社区关于平衡创新与安全之间持续不断的辩论，反思了过去因安全考量而推迟 AI 模型发布的案例。

  - **[Andrej Karpathy's Newest Development - Autonomously Improving Agentic Swarm Is Now Operational](https://www.reddit.com/r/singularity/comments/1rq5vlw/andrej_karpathys_newest_development_autonomously/)** (热度: 1125): ****Andrej Karpathy** 开发了一个自主改进的 Agent 集群，显著增强了神经网络训练。该系统自主进行了约 `700` 次更改，其中 `20` 次更改使 “达到 GPT-2 水平的时间” 指标提高了 `11%`，将时间从 `2.02 小时` 缩短至 `1.80 小时`。这标志着一个重要的里程碑，因为它展示了 AI 系统在无需人工干预的情况下，有效地执行了 “尝试 → 衡量 → 思考 → 再次尝试” 的完整研究闭环。该项目是 Karpathy 的 tiny LLM 计划的一部分，更多细节可以在 [GitHub](https://github.com/karpathy/nanochat) 上找到。** 评论者对 AI 自主优化流程的能力印象深刻，将其与 RAG 管道等其他 AI 应用中的类似改进进行了类比。这一进展被视为迈向 AI Singularity（奇点）的潜在一步，即 AI 系统可以独立地进行自我改进和优化。

    - **SECONDLANDING** 强调了一项重大成就，即一个 AI Agent 自主地将模型的训练效率提高了约 `11%`，将达到 GPT-2 级性能的时间从 `2.02 小时` 缩短至 `1.80 小时`。这是通过包含迭代测试和优化的自我导向研究闭环实现的，标志着 AI 超越人工调优工作的显著案例。[GitHub 链接](https://github.com/karpathy/nanochat)。
    - **Worldly_Expression43** 分享了使用 Opus 4.6 的类似经验，它利用 `pgvector` 自主优化了检索增强生成 (RAG) 管道。AI 评估了多种分块 (chunking) 策略，最终比原始向量数据库方法实现了 `3倍` 的速度提升。这强调了 AI 在自我基准测试和优化方面的潜力，从而带来显著的性能提升。
    - **TumbleweedPuzzled293** 对自主改进的 AI 集群的 Alignment（对齐）和控制表示担忧。虽然这项技术令人兴奋，但随着这些系统不断演化和修改自身，缺乏明确的保持 Alignment 的策略会带来潜在风险，突显了此类进步既充满希望又暗藏危险的双重性质。

- **[一个 EpochAI Frontier Math 开放问题可能首次被 GPT-5.4 解决](https://www.reddit.com/r/singularity/comments/1rq1av1/an_epochai_frontier_math_open_problem_may_have/)** (热度: 646): **据报道，**GPT-5.4** 已经解决了来自 **EpochAI Frontier Math** 集合中的一个开放问题，该集合由目前专业数学家尚未能解决的数学难题组成。这一成就如果得到验证，将标志着一个重要的里程碑，因为它表明了 AI 在解决开放式研究问题方面的潜力，可能在未来带来更实质性的突破。被解决的问题被描述为“中等程度的趣味性”，表明 AI 在处理复杂数学挑战的能力上取得了有意义的进展。[EpochAI 的开放问题](https://epoch.ai/frontiermath/open-problems)旨在通过 AI 解决方案推动人类数学知识的进步。** 评论者强调了 AI 解决开放式研究问题的意义，认为这可能会导致更大的突破。**Archivara** 的参与增加了该说法的前度，尽管对其相对难度仍存在一些质疑。

    - ImmuneHack 强调了 AI 潜力解决开放研究问题的意义，强调虽然该问题可能不是数学中最具挑战性的，但 AI 为开放式研究做出贡献的能力是一个关键里程碑。这可能预示着迈向更重大突破的轨迹，暗示 AI 在数学研究中角色的转变。
    - FundusAnimae 指出可信来源 **Archivara** 参与了讨论，这增加了该说法的可信度。被解决的问题被描述为“中等程度的趣味性”，表明这是一项显著的成就，但尚未达到数学挑战的巅峰。这凸显了 AI 在应对复杂问题方面日益增长的能力。
    - socoolandawesome 提供了一个更新，称一位 Epoch 研究员认为该解法是正确的，但正在等待问题作者的确认。这突出了数学研究中持续的验证过程，其中同行评审和作者确认是确立解法正确性的关键步骤。

- **[Yann LeCun 揭晓其新初创公司 Advanced Machine Intelligence (AMI Labs) —— 并融资 10.3 亿美元](https://www.reddit.com/r/singularity/comments/1rprdy7/yann_lecun_unveils_his_new_startup_advanced/)** (热度: 997): **Yann LeCun** 与 **Alexandre LeBrun** 共同创立了一家名为 **Advanced Machine Intelligence (AMI Labs)** 的新初创公司。该公司已筹集了 `10.3 亿美元`，用于利用 LeCun 的 **JEPA 架构**开发 **World Models**（世界模型），该架构旨在模拟物理现实而非仅仅是文本，从而解决现有 **LLM** 的幻觉问题等局限性。这一举措被定位为一项长期的研究工作，近期没有产品或收入预期，团队成员包括 **Saining Xie**、 **Pascale Fung** 和 **Michael Rabbat**。该项目得到了 **NVIDIA**、**Samsung** 和 **Bezos Expeditions** 等主要投资者的支持，并将以开源形式发布其代码和论文。[TechCrunch](https://techcrunch.com/2026/03/09/yann-lecuns-ami-labs-raises-1-03-billion-to-build-world-models/)。评论者对 LeCun 的新创举表示乐观，强调他在提供 AI 诚实见解方面的声誉。人们对 **AMI Labs** 的研究对 AI 领域的潜在影响充满期待。

    - Yann LeCun 的新初创公司 Advanced Machine Intelligence (AMI Labs) 已融资 10.3 亿美元，据报道正在寻求 50 亿美元的估值。公司专注于开发 World Models，这是一种能够理解和预测复杂现实世界现象的高级 AI 系统。这一宏伟目标与 LeCun 推动 AI 研究和开发边界的声誉相一致。
    - AMI Labs 组建了一支实力雄厚的领导团队，包括担任 CEO 的 LeBrun、担任 CFO 的 LeFunde 以及担任训练后（post-training）负责人的 LeTune。这些战略性聘用表明了对稳健财务管理和先进 AI 模型优化技术的关注，特别是在 post-training 过程中，这对于增强模型性能和效率至关重要。
    - 该初创公司正在考虑聘请 LeMune 担任增长负责人，并由 LePrune 领导推理效率（inference efficiency）工作，这标志着在扩展业务和优化 AI 推理过程方面的战略重心。这反映了行业提高 AI 模型效率的更广泛趋势，这对于在现实应用中部署大规模 AI 系统至关重要。

- **[我如何使用 2x 4090 GPUs 登顶 Open LLM Leaderboard - 博客形式的研究笔记](https://www.reddit.com/r/MachineLearning/comments/1rq6g08/how_i_topped_the_open_llm_leaderboard_using_2x/)** (热度: 234): **该贴描述了一种提高大型语言模型 (LLM) 性能的新颖方法，即在不修改任何权重的情况下，复制 Qwen2-72B 模型中特定的 7 个中间层区块。这种方法在所有 Open LLM Leaderboard 基准测试中都带来了显著的性能提升，并自 2026 年起一直保持领先地位。作者认为，预训练在模型的层堆栈中创建了离散的功能电路 (functional circuits)，这些电路只有作为一个整体保留时才能有效发挥作用。这项工作是使用 2x RTX 4090 GPUs 完成的，证明了在不需要大量计算资源的情况下也能取得重大进展。作者目前正在一台双 GH200 设备上试验 GLM-4.7 和 Qwen3.5 等当前模型，并计划很快发布代码和新模型。更多细节请参见[完整技术报告](https://dnhkng.github.io/posts/rys/)。** 评论者讨论了循环层电路而非复制它们的潜力，建议训练模型识别何时停止循环可能会产生有意义的结果。大家对于这些电路是离散的还是重叠的，以及在复制前后是否分析了注意力模式 (attention patterns) 或激活统计数据 (activation statistics) 感到好奇。这种方法与一些机械可解释性 (mechanistic interpretability) 工作相一致，而在预训练模型上取得显著结果的想法被认为是令人印象深刻且耳目一新的。

    - 该博客文章讨论了一种非传统的 Transformer 架构方法，其中层被重新排列，例如将第 60 层的输出反馈到第 10 层，而这是模型在训练期间从未遇到过的。这表明 Transformer 层比以前认为的更具互换性，内部表示足够同质，可以处理无序的隐藏状态 (hidden states)。这种灵活性意味着架构可以在没有严格层顺序的情况下运行，挑战了关于模型训练和架构设计的传统观点。
    - 一位评论者建议了循环层电路而非复制它们的潜力，提议训练模型来确定何时停止循环。这种方法可能涉及训练循环/继续/停止 (loop/continue/halt) 命令以实现有意义的早期退出。这个想法是利用现有电路来提高效率，可能通过增强推理能力而无需进行广泛的重新训练，从而为模型提供“免费升级”。这个概念与一些从一开始就包含类似机制的现有模型不谋而合。
    - 另一位评论者强调了这一观察的重要性，即有用的电路存在于小的层区块中，这与机械可解释性研究相一致。他们对在不改变权重的情况下复制区块的有效性表示惊讶，并询问了复制前后的注意力模式或激活统计数据。这种好奇心延伸到这些复制的区块在 Qwen 或 GLM 等不同 LLM 架构中的表现是否一致，表明这是一个潜在的进一步研究领域。

  - **[基准测试模型性能：发布当天 vs. 当前 API 版本](https://www.reddit.com/r/Bard/comments/1rprddw/benchmarking_model_performance_launch_day_vs/)** (热度: 227): **图片对比了 Gemini 3.1 Pro 模型在两个不同日期的输出，突显了随着时间的推移感知到的质量退化。左侧图片来自 2026 年 2 月 19 日，显示了一辆更精细的法拉利，而右侧图片来自 2026 年 5 月 10 日，看起来更简单。这暗示了模型更新或 API 更改可能影响输出质量的问题。然而，由于 LLM 推理的随机性 (stochastic nature)，这种对比的有效性受到了质疑，因为需要多次运行才能得出可靠的结论。** 评论者强调了 LLMs 的随机性，认为单一的对比不足以评估模型性能的变化。此外，还有人对提到的日期表示怀疑，暗示可能存在错误或误解。

- **DifficultSelection** 强调了在 LLMs 基准测试中进行多次运行的重要性，并指出推理具有随机性，每个日期需要大约 `30 runs` 才能得出有意义的结论。这突显了 LLM 输出的概率性质，不同运行之间可能会有显著差异。
- **Cet-Id** 强调了对 LLMs 的一个常见误解，指出许多用户未能理解其概率本质。这表明输出的可变性是固有的，在性能评估中应予以考虑。
- **sankalp_pateriya** 引用了一张显示日期差异的图片，暗示基准测试帖子中可能存在错误或操纵。这引发了对所呈现数据有效性的质疑，因为图片显示了一个未来的日期 `10th May 2026`，这与当前的时间线不符。

### 2. AI 在创意和视频制作中的应用

- **[一直在悄悄使用 Claude 建立一个不露脸的 YouTube 频道，我离获利已经尴尬地接近了](https://www.reddit.com/r/ClaudeAI/comments/1rqbrsm/been_quietly_building_a_faceless_youtube_channel/)** (热度: 2938): **该 Reddit 帖子描述了使用 AI 工具创建不露脸 YouTube 频道的流程，具体包括使用 **Claude** 编写脚本、**ElevenLabs** 进行配音、**Magic Hour** 生成视频以及 **CapCut** 进行剪辑。该用户报告称 YouTube 频道已接近获利阶段，并强调了使用 AI 生成听起来像真人的内容。虽然该过程被描述为不够成熟，但对该用户的需求很有效，目前尚未声称取得显著的财务成功。** 评论对 YouTube 上的 AI 生成内容表示强烈反对，将其贴上“死亡互联网内容”和“AI 垃圾内容（AI slop）”的标签。人们对这类内容的变现潜力持怀疑态度，理由是 YouTube 有封禁类似频道的历史。

- **[自 2025 年 8 月以来，我通过 AI 视频赚了 7 万美元，大家尽管问（AMA）](https://www.reddit.com/r/VEO3/comments/1rqv704/ive_made_70k_from_ai_videos_since_august_2025_ama/)** (热度: 224): **该 Reddit 帖子讨论了一位视频制作人在 2025 年 8 月转型 AI 视频制作并获得 7 万美元收入的经历。作者强调了加入像 Skool 这样的 AI 视频社区进行社交和学习的重要性，并突出了制作公司对 AI 技能的需求。他们分享了一些策略，例如制作高质量视频向决策者展示，以及探索用户生成内容（UGC）市场。帖子还提到使用简单的 prompts 驱动 Nano Banana 和 Kling 等 AI 模型，以高效生成多样化内容。** 评论者对作者的作品集和客户获取策略等实际细节感兴趣，并询问了访问 AI 模型的一体化平台，表明关注点在于实际应用和资源优化。

    - Ant12-3 询问了性价比最高的一体化 AI 模型访问平台，并提到了 Higgsfield。这引发了关于使用整合平台与为不同 AI 工具维护独立账户之间的效率和成本效益的讨论。回应可能会深入探讨这些方法之间的权衡，例如易用性与灵活性以及获取前沿模型的能力。
    - advertisingdave 提到使用了 Higgsfield 并询问了 Flow 的使用情况，表明了对这些 AI 工具的比较。这可能会引发关于 Higgsfield 与 Flow 的功能、性能和具体使用场景的技术讨论，突出它们在 AI 视频制作工作流中的优缺点。
    - 电视剪辑师 TheFreakmode 询问了为制作公司所做的具体工作类型，特别是涉及创建故事还是镜头。这开启了关于 AI 在内容创作中作用的对话，详细说明了 AI 工具如何集成到传统制作流程中，以及制作公司通常对 AI 生成内容的期望。

### 3. Claude 与 AI 工具的日常应用

  - **[停止为“AI 训练营”支付 1,000 多美元。Anthropic（Claude 的开发者）刚刚发布了一个 100% 免费的学院。](https://www.reddit.com/r/ClaudeAI/comments/1rqopis/stop_paying_1000_for_ai_bootcamps_anthropic/)** (活跃度: 1679): **Anthropic** 推出了一个免费的在线学院，提供 AI 相关课程，特别聚焦于其 AI 模型 Claude。课程涵盖了实际应用，例如将 Claude 与 Amazon Bedrock 和 Google Cloud 的 Vertex AI 等平台集成，并为包括教育工作者和非营利专业人士在内的不同受众量身定制。该计划旨在提供关于 AI 素养和伦理协作的普及教育，以对抗昂贵 AI 训练营的趋势。一些评论者指出，该学院自 2025 年中期以来就已开放，这表明该公告并非新闻。也有人对昂贵的 AI 训练营的价值表示怀疑，质疑谁会为这类项目支付 1000 美元。


  - **[Claude 帮我重新设置了镇上的交通信号灯](https://www.reddit.com/r/ClaudeAI/comments/1rphxvk/claude_helped_me_get_a_traffic_light_reprogrammed/)** (活跃度: 3301): **图片描绘了一次邮件交流，一位名为 Lenny 的用户成功地利用 AI 语言模型 **Claude**，将外行人的请求翻译成了适合信号工程师的专业技术语言。这导致了埃塞克斯（Essex）一个特定交叉口交通信号灯程序的修改，使每个周期能多通过 2-3 辆车，从而改善了交通流量。这次交流突显了 AI 在促进公众与技术专家之间沟通方面的实际应用，从而带来了基础设施的实质性改进。** 评论者对工程团队的快速响应表示惊讶和赞赏，指出了 AI 在现实世界应用中的有效性。


  - **[ChatGPT vs Gemini vs Claude vs Perplexity：我给它们各 1000 美元来炒股。9 周后，ChatGPT 从空仓观望变为 +21%（一只股票翻倍）](https://www.reddit.com/r/ChatGPT/comments/1rq33za/chatgpt_vs_gemini_vs_claude_vs_perplexity_i_gave/)** (活跃度: 1345): **在一项为期 9 周的实验中，四个 AI 模型——**ChatGPT**、**Gemini**、**Claude** 和 **Perplexity**——各自获得了 `$1,000`，并使用 Alpaca APIs 进行自主股票交易。**ChatGPT** 以 `+21.1%` 的回报率领先，主要归功于对医疗保健股票的战略性重仓，尤其是翻倍的 **IOVA** 和上涨了 `52%` 的 **ACHC**。**Perplexity** 通过持有大部分现金维持了 `+1.1%` 的回报，而 **Gemini** 和 **Claude** 由于高风险交易和频繁止损，分别表现不佳，回报率为 `-6.6%` 和 `-11.5%`。同期 S&P 500 下跌了 `-1.5%`，凸显了 ChatGPT 显著的超额表现。该实验记录在 [GitHub](https://seve1995.github.io/ai-portfolio-experiment/) 上，更多详情可在 [Substack](https://aiportfolioexperiment.substack.com/) 查看。** 评论者认为结果可能是偶然的，并建议使用每个模型的多个实例来验证发现，尽管这需要大量的资金支持。另一个建议是包含一个随机对照组（例如投掷飞镖），以衡量 AI 相对于随机性的表现。

    - TripIndividual9928 强调了理解每个模型的风险承受能力和交易行为的重要性。ChatGPT 的持币观望然后重仓投资的策略符合量化金融中一种被称为“基于信念的头寸管理（conviction-based position sizing）”的已知策略，即等待高确定性的机会。相比之下，Claude 的频繁交易反映了散户交易中常见的过度交易错误，正如行为金融学研究支持的那样，这往往会导致较差的回报。
    - vegt121 建议通过使用每个模型的多个实例来交易股票，从而设计一个更稳健的实验，这将为性能提供更具统计学意义的分析。这种方法需要大量的资金资源（为每个模型分配 100 个账户大约需要 40 万美元），但它可以对模型的交易能力提供更可靠的洞察。
    - TripIndividual9928 还建议对实验进行增强，让每个模型在做出交易决策前生成一份推理备忘录。这将允许分析交易前的推理质量与实际回报的对比，从而揭示提供更好分析洞察的模型是否也能获得更好的交易结果。





# AI Discord 社区

不幸的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们将很快发布新的 AINews。感谢阅读到这里，这是一段美好的历程。