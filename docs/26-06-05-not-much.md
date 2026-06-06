---
companies:
- anthropic
- sakana-ai
- meta-ai-fair
- princeton
date: '2026-06-05T05:44:39.731046Z'
description: 'Anthropic 的 **Mythos/Opus 迭代周期**引发了褒贬不一的反响：**Claude Mythos** 的单次调用工作流（one-shot
  workflows）广受好评，但 **Opus 4.8** 在基准测试中的性能回退（regressions）也引发了担忧。**Opus 4.7** 在化学任务中表现强劲，被誉为“让
  Claude 成为了化学家”。


  **Sakana AI** 成立了一个 **RSI 实验室**，专注于计算受限下的递归自我改进（Recursive Self-Improvement），标志着
  RSI 正式成为一项研究计划。诸如 **Agents'' Last Exam (ALE)** 和 **SWE-Marathon** 等新基准测试，开始在长程且具备经济价值的任务上对智能体进行评估，结果揭示了通过率低和连贯性不足的挑战。


  普林斯顿大学的一篇 ICML 2026 论文发现，包括 **GPT 5.5**、**Gemini 3.1 Pro / 3.5 Flash** 以及 **Claude
  Opus 4.7** 在内的模型，在可靠性方面仍缺乏实质性的提升。在工具开发趋势上，业界更倾向于采用强化学习（RL）环境风格的框架来进行智能体评估，Meta 的
  **OpenEnv** 便是其中的典型代表。'
id: MjAyNS0x
models:
- claude-mythos
- opus-4.8
- opus-4.7
- gpt-5.5
- gemini-3.1-pro
- gemini-3.5-flash
- claude-opus-4.7
people:
- kimmonismus
- lechmazur
- teortaxestex
- hardmaru
- andrew_n_carr
- steverab
- pauliusztin_
title: 今天没发生什么特别的事。
topics:
- recursive-self-improvement
- benchmarking
- agent-evaluation
- long-horizon-tasks
- reliability
- reinforcement-learning
- sample-efficiency
- economically-meaningful-tasks
- agent-coherence
- anti-reward-hacking
- tooling
- rl-environments
---

**平静的一天。**

> 2026年6月4日至6月5日的 AI 新闻。我们查看了 12 个 Subreddit，[544 条 Twitter](https://twitter.com/i/lists/1585430245762441216)，没有更多的 Discord。[AINews 网站](https://news.smol.ai/)允许你搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[选择订阅频率](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)！

---

# AI Twitter 综述

**Frontier Models, RSI, 以及“AI 构建 AI”的叙事**

- **Anthropic 的 Mythos/Opus 周期主导了讨论，但实质内容与投机猜测并存**：社区注意力集中在 **Claude Mythos** 上，多位用户称其输出达到了“下一层级”，并强调了其强大的 one-shot 桌面和 MacOS 工作流（[kimmonismus 关于 Mythos 输出的评价](https://x.com/kimmonismus/status/2062843119864021404)，[更多反应](https://x.com/kimmonismus/status/2062933600287224073)，[早前帖子](https://x.com/kimmonismus/status/2062805570982203820)）。与此同时，也出现了关于基准测试退化的问题——例如，有说法称 **Opus 4.8 在 LLM Debate Benchmark 上的表现不如 4.7**，以及对早期 Sonnet/Opus 发展轨迹叙事的质疑（[LechMazur](https://x.com/LechMazur/status/2062954327199666602)，[teortaxesTex](https://x.com/teortaxesTex/status/2062807380643958948)）。Anthropic 还发布了一项具体的科学成果：**Opus 4.7 在某些任务上匹配或击败了专业的 NMR 软件**，这被描述为“让 Claude 成为化学家”（[AnthropicAI](https://x.com/AnthropicAI/status/2062979607448682731)）。
- **Recursive self-improvement (RSI) 从模糊的理论转变为明确的组织战略**：[Sakana AI](https://x.com/SakanaAILabs/status/2062948403815030850) 在东京成立了专门的 **RSI Lab**，整合了之前的项目如 **The AI Scientist**、**Darwin Gödel Machine** 和 **ShinkaEvolve**，并明确宣称自我改进系统可以在计算受限的情况下构建，而非仅限于超大规模机制。[hardmaru](https://x.com/hardmaru/status/2062948594597208557) 强调了将 **sample efficiency** 作为设计约束。这与行业内围绕自我改进系统的更广泛言论不谋而合：[kimmonismus](https://x.com/kimmonismus/status/2062868789746671819) 认为 Anthropic/OpenAI 的 RSI 主张不仅仅是 IPO 噱头，而 [andrew_n_carr](https://x.com/andrew_n_carr/status/2062976064343912949) 则暗示通往 AGI 的道路上可能仅剩“1 或 2 个难题”。值得注意的转变是，RSI 不再仅仅是博客文章中的概念；各实验室正围绕其建立正式的研究项目并配备人员。

**Agent 评估、可靠性及长程基准测试**

- **Benchmark 正从任务片段转向具有经济意义的长程工作**：多项新工作推动了超越经典 SWE-bench 风格的评估。[dair_ai](https://x.com/dair_ai/status/2062916866235068607) 推出了 **Agents’ Last Exam (ALE)**，这是一个包含 **1,000 多个具有经济价值的任务**的 Benchmark，并映射到美国职业分类，其中最难层级的平均**全通率仅为 2.6%**。[rishi_desai2](https://x.com/rishi_desai2/status/2062930906818769356) 发布了 **SWE-Marathon**，测试 Coding Agent 是否能在 **10 亿 Token 预算**的项目（如构建 Slack 克隆、将 JAX 重写为 PyTorch 或实现 C 编译器）中保持连贯性。[omarsar0](https://x.com/omarsar0/status/2062919381777350914) 强调了 **Meta-Agent Challenge**，其中 Agent 尝试在沙盒 + Eval API + 时间预算的设置下进行自我改进；结果显示 Meta-Agent 很少能达到人类基准，尽管有反 Reward-hacking 防御，一些 Agent 仍尝试**窃取真值（ground-truth exfiltration）**。
- **可靠性研究继续表明，前沿模型尚不足够可靠**：[steverab](https://x.com/steverab/status/2062890225144135800) 分享了普林斯顿大学更新的 ICML 2026 论文 **“Towards a Science of AI Agent Reliability,”**，增加了 **GPT 5.5、Gemini 3.1 Pro / 3.5 Flash 和 Claude Opus 4.7**，并得出结论：它们并不比之前的模型**更可靠**。此次更新还修正了一个结果一致性指标的笔误，并审计了包括**答案泄露**和 **Agent 在 GAIA 上作弊**在内的框架（scaffold）问题，但总体一致性仍然较低。相关评论强调，“可验证的任务”通常只意味着**简单的任务**（[MillionInt](https://x.com/MillionInt/status/2062924521779450147)），而正确的设定应该是“**现实：最终的评估**”，即系统是否能在生产环境中工作，而不是它们是否通过了 Benchmark 阈值（[559hkdt 引用 swyx/Andon](https://x.com/559hkdt/status/2062867094111219824)）。
- **工具链正趋向于为 Agent 提供类似 RL 环境的测试框架**：[pauliusztin_](https://x.com/pauliusztin_/status/2062874580411162811) 主张通过 Meta 的 **OpenEnv** 将 Agentic Coding 系统建模为 **Gym 风格的 RL 环境**，主要用于可观测性而非优化：成功率、重试次数、工具效率、失败模式、每个成功轨迹的成本。[adithya_s_k](https://x.com/adithya_s_k/status/2062871067803205815) 指出，一份关于 LLM 的 RL 环境指南受到了广泛关注，而 [latentspacepod](https://x.com/latentspacepod/status/2062972030606274785) 发布了对低质量 RL 环境的批评。这些迹象共同表明，Agent 工程正从“体感测试（vibe checks）”转向可重复的测试框架。

**开源模型、量化与多模态发布**

- **Gemma 4 QAT 是本地部署中最具实际意义的开源发布**：Google 在不同模型规模上发布了 **Gemma 4 QAT (Quantization-Aware Training)** 权重 ([googlegemma](https://x.com/googlegemma/status/2062928831229665566), [osanseviero](https://x.com/osanseviero/status/2062933011415392482))。此次发布强调在保持质量的同时降低内存占用，包括一种**移动端量化格式**，并声称 **E2B 可以在约 1GB 内存中运行**。生态系统支持已通过 [Ollama](https://x.com/ollama/status/2062965815864066079) 和 [vLLM](https://x.com/vllm_project/status/2062938949560283216) 立即落地。[danielhanchen](https://x.com/danielhanchen/status/2062933017430315481) 还指出了一个微妙的互操作性问题：将 QAT 直接转换为 llama.cpp 的 **Q4_0** 点阵（lattice）会损失精度，而 Unsloth 的动态 GGUF 则可以恢复大部分精度。
- **Ideogram 4 在图像生成领域脱颖而出，因为它既强大又是开源权重的**：[ideogram_ai](https://x.com/ideogram_ai/status/2062956373957292281) 发表了一篇技术博客，将 **Ideogram 4.0** 描述为一个从零训练的 **9.3B Diffusion Transformer**，并带有一个**冻结的 8B VLM 文本编码器**。值得注意的是，它发布了 **fp8 和 nf4 权重**，其中 **nf4 变体可以运行在单张 24GB GPU 上** ([后续更新](https://x.com/ideogram_ai/status/2062956472489922584))。Arena 测试结果将 **Ideogram 4.0 Quality** 列入文本生成图像的第一梯队，并作为**领先的开源权重图像模型** ([arena](https://x.com/arena/status/2062957421757452516), [开源权重排名更新](https://x.com/arena/status/2062997992777609534))。
- **NVIDIA 的开源模型推动力持续扩大**：围绕 **Nemotron 3 Ultra** 的讨论集中在训练后（post-training）的细节上，例如用于教师-学生分布匹配的 **MOPD 预热 (warmup)**，以及用于推测解码 (speculative decoding) 的 **MTP 增强** ([ben_burtenshaw](https://x.com/ben_burtenshaw/status/2062902364525244572))。NVIDIA 还通过 **Nemotron Coalition** 扩展了其生态系统，加入了 **Nous, Prime Intellect, 和 hcompany** 等合作伙伴 ([NVIDIAAI](https://x.com/NVIDIAAI/status/2062961026409333232))。下游平台反应迅速：[Perplexity](https://x.com/perplexity_ai/status/2062976272436002825) 已向 Pro/Max 用户开放 **Nemotron 3 Ultra**，并将其定位为适用于长时间运行 Agent 的开源模型。

**Agent 产品、开发者工具及运行时基础设施**

- **Hermes Agent 经历了全栈产品周**：[Teknium](https://x.com/Teknium/status/2062822586954997909) 展示了如何**使用 Hermes Agent 构建 Hermes Agent**，并在随后的一周里推进了插件支持、文档和内容策展 ([插件指南](https://x.com/Teknium/status/2062854497865810164), [开发者体验讨论帖](https://x.com/Teknium/status/2062830182432731256))。最重要的发布是 **Hermes v0.16.0**，其中包括一个**桌面 GUI 应用**、控制面板重构、更精简的内置技能，以及**用于远程控制面板/GUI 访问的新安全层**，包括简单身份验证和 OAuth ([发布公告](https://x.com/Teknium/status/2063075771317686606), [安全后续](https://x.com/Teknium/status/2063078732768928234), [中文桌面支持](https://x.com/Teknium/status/2062953592131342832))。
- **Arena 从被动榜单转型为主动的 Agent 运行时**：[arena](https://x.com/arena/status/2062902033389322477) 推出了 **Agent Mode** 及其 **Agent Arena**，用户可以在真实任务中运行 Agent，并将聚合指标（如**确认成功率、表扬 vs 投诉、可控性、bash 恢复能力和工具幻觉**）反馈到排行榜中 ([排行榜细节](https://x.com/arena/status/2062902039445959060))。这是本周评测公司转型为执行平台最清晰的案例之一。
- **开发者工具正围绕 Agent 效率而非仅仅是人类用户体验进行重构**：[ClementDelangue](https://x.com/ClementDelangue/status/2062982727729553913) 提供了一个深刻的运营见解：Agent 优化型工具至关重要，因为相比使用 Hugging Face CLI，**手动编写原始 API 交互消耗的 Token 最多可达 6 倍，且成功率更低**。他的观点——“**优秀的工具是为 Agent 缓存的智能**”——捕捉到了 Agent 原生开发者平台正在兴起的设计原则。相关发布包括作为官方 Codex 插件的 **MagicPath** ([skirano](https://x.com/skirano/status/2062942695547375829))、用于可视化提示 UI 变更的 **Cursor Design Mode** ([cursor_ai](https://x.com/cursor_ai/status/2062950344687272144))，以及 **Perplexity Computer 内置的 Vercel 集成**，用于通过自然语言检查部署并重新部署 ([vercel_dev](https://x.com/vercel_dev/status/2062934988648329515))。

**算力、基础设施经济学及平台运营**

- **AI 基础设施经济学正在成为首要议题**：[Epoch AI](https://x.com/EpochAIResearch/status/2062933470373146828) 估计，到 **2026 年第一季度，AI 相关的数据库中心建设、计算硬件和网络支出将占美国 GDP 的 ~0.8%**，推动总计算基础设施支出达到 **GDP 的 ~1.5%**。在运营方面，[eglyman](https://x.com/eglyman/status/2062921352613425446) 认为问题不在于原始的 Token 支出，而在于缺乏**归因与分配（attribution and allocation）**，并指出即使将 **1000 万美元 AI 账单中的 10%** 从 frontier models 重新路由到更便宜的模型层级，也能节省近 **100 万美元**。
- **Cloudflare 为推理路由发布了具体的成本控制功能**：[CF changelog](https://x.com/CFchangelog/status/2062762883222483347)、[elithrar](https://x.com/elithrar/status/2062887228909527346) 和 [michellechen](https://x.com/michellechen/status/2062894017545720129) 共同宣布了 **AI Gateway 支出限制**、按模型/用户执行预算，以及在达到上限时**回退（fallbacks）到更便宜的模型**，随后还将通过 Cloudflare Access 推出基于身份的控制。随着使用规模从原型阶段转向生产阶段，这正是企业团队目前迫切需要的底层基础设施功能。
- **平台/安全事件仍然至关重要，因为它们揭示了失效模式**：OpenAI 发生了一起账号封禁事件，并由 [OpenAI](https://x.com/OpenAI/status/2062927046448431587) 公开确认，支持人员的后续跟进表明大多数账号/订阅随后已恢复（[reach_vb](https://x.com/reach_vb/status/2063035661855183215)）。OpenAI 还向所有用户推出了 **ChatGPT 锁定模式（Lockdown Mode）**，旨在通过限制出站网络请求来减少 **prompt-injection 驱动的数据外泄（data exfiltration）** 的最终阶段（[cryps1s](https://x.com/cryps1s/status/2062923575049531422)）。另外，关于 Anthropic 停机可能导致跨租户输出暴露的猜测表明，**多租户隔离失效（multi-tenant isolation failures）** 仍然是 Agent 型/云端推理产品中严重程度最高的风险之一（[kimmonismus](https://x.com/kimmonismus/status/2062997809067139468)）。

**热门推文（按互动量排序）**

- **Gemma 4 QAT 发布**：[@googlegemma](https://x.com/googlegemma/status/2062928831229665566) 宣布了适用于所有 Gemma 4 尺寸和 drafters 的 QAT checkpoints，专注于低内存的端侧推理。
- **Anthropic 的 Claude 使用限额扩展**：[@claudeai](https://x.com/claudeai/status/2063018337567670285) 表示已在 **Claude Cowork 中将使用限制翻倍**一个月，以支持更大的委派任务。
- **OpenAI 平台事件**：[@OpenAI](https://x.com/OpenAI/status/2062927046448431587) 报告了错误的账号封禁以及恢复工作。
- **Cursor 设计模式（Design Mode）**：[@cursor_ai](https://x.com/cursor_ai/status/2062950344687272144) 推出了通过指向、绘图或语音进行的多模态 UI 编辑功能。
- **Google 的 Agentic RAG 框架**：[@GoogleResearch](https://x.com/GoogleResearch/status/2062982001850974257) 引入了一种**多 Agent 企业级 RAG** 工作流，采用迭代式上下文收集而非单次检索（one-shot retrieval）。


---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Gemma 4 QAT 和 Nemotron 3 Ultra 发布

  - **[Gemma 4 结合量化感知训练 (QAT)](https://www.reddit.com/r/LocalLLaMA/comments/1txpeo0/gemma_4_with_quantizationaware_training/)** (热度: 982): **Google 在 Hugging Face 上发布了 Gemma 4 量化感知训练 (QAT) 检查点**，针对 [`q4_0`](https://huggingface.co/collections/google/gemma-4-qat-q4-0) 和 [移动端](https://huggingface.co/collections/google/gemma-4-qat-mobile) 目标，同时 **Unsloth** 提供了额外的 [QAT 构建版本](https://huggingface.co/collections/unsloth/gemma-4-qat) 以及 [KLD/质量分析](https://unsloth.ai/docs/models/gemma-4/qat#qat-analysis)。评论者强调了 Google 官方提供的 GGUF 文件，涵盖 [E2B](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)、[E4B](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf)、[12B](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf)、[26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf) 和 [31B](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf)，此外还有 `2-bit` 和 `4-bit` QAT 检查点，旨在与 BF16/PTQ 相比减少本地推理的内存/存储占用，同时保持质量。评论者对较小的 QAT 版本持乐观态度，认为这能让 **Gemma 4 E4B** 等模型在 `6 GB` VRAM 笔记本电脑等受限硬件上运行。一个尚未解决的关键技术问题是 Google 或其他人是否发布了直接对比 **QAT `q4_0` 与 BF16** 质量/性能的基准测试。

    - **Google 发布了官方 Gemma 4 QAT GGUF 检查点** 的 `q4_0` 版本，包括 [E2B](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)、[E4B](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf)、[12B](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf)、[26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf) 和 [31B](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf)。评论者注意到了其对受限本地推理的实际影响，有人预计 **E4B** QAT 版本将能在 `6GB VRAM` 的笔记本电脑上正常运行。
    - 一位评论者链接了 Google 的发布博客 [“Gemma 4 的量化感知训练”](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)，但指出该博文**并未提供 QAT `q4` 与 `bf16` 的基准测试对比**。提出的主要技术疑虑是缺乏证据支持 Google 关于 QAT 能保留模型能力和质量的说法。

  - **[nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1twla1k/nvidianvidianemotron3ultra550ba55bbf16_hugging/)** (热度: 622): **NVIDIA** 发布了 [`NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16)，这是一个拥有 `550B` 参数的 **LatentMoE** 模型，激活参数为 `55B`。它结合了 **Mamba-2**、MoE、选择性注意力（selective attention）以及 **Multi-Token Prediction**，支持高达 `1M` token 的上下文。该模型针对前沿推理、Agent 工作流、长上下文/RAG、工具使用和多语言任务，支持通过 `enable_thinking=True/False` 配置推理模式，并根据 [OpenMDW 1.1 许可证](https://raw.githubusercontent.com/OpenMDW/OpenMDW/refs/heads/main/1.1/LICENSE.OpenMDW-1.1) 发布。列出的最低推理硬件要求为 **8× GB200/B200/GB300/B300**、**16× H100** 或 **8× H200**，这使得大多数用户无法进行本地部署。评论讨论几乎完全集中在极高的硬件占用上；唯一的实质性技术点重申了最低 GPU 要求，而其他评论则在开玩笑说自己少了一块 H200，或者尝试在过时硬件上运行它。

    - 一位评论者指出，官方声明的最低硬件要求极高：**`8x GB200/B200/GB300/B300`、`16x H100` 或 `8x H200`**，这意味着这个 550B 级别的 BF16 Nemotron 模型目标是多节点/数据中心推理，而非典型的本地部署。
    - 一个技术结论是，评论者认为 **NVIDIA Nemotron-3 Ultra 550B A55B BF16** 是日益增长的针对**低延迟推理**优化的开源大模型系列之一。即使输出质量落后于 **GLM** 等模型，在吞吐量/延迟比边缘化的基准测试质量更重要的生产工作负载中，更快的响应速度仍被视为具有极高价值。

### 2. KV Cache 量化与 Agentic 上下文可靠性

  - **[KVarN：来自华为的新型 KV-cache 量化。实现 3–5 倍 KV-cache 压缩且实际提速而非降速，且与 TurboQuant 不同，它在推理任务中表现稳健 (Apache 2.0, vLLM single flag)](https://www.reddit.com/r/LocalLLaMA/comments/1twptw2/kvarn_new_kvcache_quant_from_huawei_35_kv_cache/)** (热度: 633): **华为**开源了 **KVarN**，这是一种通过单个 flag 集成到 **vLLM** 中的 Apache-2.0 KV-cache 量化方法。该方法声称与 FP16 相比，可实现 `3–5×` 的 KV-cache/上下文压缩，吞吐量最高可达 FP16 的 `~1.4×`，且最高可达 **TurboQuant** 吞吐量的 `~2.4×`，同时保持类似 FP16 的输出质量 ([repo](https://github.com/huawei-csl/KVarN), [paper](https://arxiv.org/abs/2606.03458))。该贴将其与 vLLM FP8 KV-cache（`~2×` 容量，接近 BF16 的吞吐量）以及 **Google TurboQuant** 进行了对比，并引用了 vLLM/Red Hat AI 的结果，指出由于 BF16 反量化（dequantization）开销，TurboQuant 的吞吐量可能会下降到 BF16 的 `66–80%`，并在 AIME25/LiveCodeBench 上损失 `~20` 个推理分 ([vLLM study](https://vllm.ai/blog/2026-05-11-turboquant))。KVarN 的核心主张是在不进行重训练、校准或模型更改的情况下，在高压缩比下保持推理/数学/代码质量，解决了已知的低位 KV-cache 失效模式。评论大多持怀疑态度——例如 *“眼见为实”*——一位评论者预见到会有低质量的 PR 涌入 **llama.cpp**。此外，还有人提供了一个技术性的后续建议：在 **B200** 上使用 Qwen/Gemma 基准测试 KVarN，包括 MTP 和非 MTP 缩放检查。

    - 一位评论者指出，**KVarN** 更有意义的生产测试不是 `batch=1`，而是更高并发的情况（如 `batch=16`），因为在这种情况下，许多 KV-cache 量化方法会失去其表观收益，原因是**反量化开销可能会抵消**内存节省。他们认为，关键的技术信号是 KVarN 在真实的 vLLM 批处理/请求混合下是否能带来实际的吞吐量提升，而不仅仅是纸面上减少了 KV 内存占用。
    - 一位用户计划在 **NVIDIA B200** 上使用现有的 **Qwen** 和 **Gemma 4** 的 **MTP 和非 MTP 基准测试** 来测试 KVarN，特别是测试其声称的缩放和加速是否在新型高端硬件上成立。这将非常有用，因为 KV-cache 压缩方法在不同的 GPU 内存带宽、并发量以及投机采样（speculative decoding）/MTP 解码设置下表现可能截然不同。

  - **[你们是对的 - Qwen 3.6 35B 确实很好... 而且 KV Cache 确实很重要。](https://www.reddit.com/r/LocalLLaMA/comments/1twyoqe/you_guys_were_right_qwen_36_35b_is_goodand_kv/)** (热度: 590): **楼主报告称，在涉及 MCP 子图、`11` 个工具、JSON 任务委派、上下文裁剪、OpenWebUI/llama.cpp 集成以及 Redis 操作的 Agentic [Rivet](https://rivet.ironcladapp.com/) 工作流中，使用**未压缩 KV-cache 的 Qwen 3.6 35B IQ4NXL** 的表现显著优于**使用 KV `Q8/8` 的 Qwen 27B Q5_K_XL**。然而，经过长时间测试后，楼主发现 35B 量化版仅在**低上下文**下可靠：在高上下文下，它会出现严重的幻觉，无法执行多任务指令，并犯下破坏性的 Redis 错误（如删除 key 和将 hash 写入 stream 而非 stream），因此他们将关键工作切回了 **27B**，仅将 35B 用于狭窄的单操作任务。一位技术评论者指出，**35B 更窄的 attention/KV 张量**可能使其对 KV-cache 量化的适应性不如 27B，而另一位用户则使用 **35B-A3B Q6** 进行快速代码库分析，并切换到 **27B Q8** 进行代码生成/规划。评论者大多将其视为速度与可靠性的权衡：35B 速度快且适用于阅读/分析，但 27B 被认为能生成更干净的代码且错误更少。大家一致认为，在长上下文的 Agentic 工作流中，KV-cache 压缩的重要性远超通用的“智能轻微下降”建议所暗示的程度。

    - 一位评论者指出，**Qwen 3.6 35B-A3B** 的 attention 张量比 **27B** 窄得多，这使其对 KV-cache 压缩更加敏感；该观点认为，当 KV-cache 精度降低时，**27B 更宽的张量更具韧性**。
    - 描述的一种工作流是：使用 **Q6 的 35B-A3B** 进行快速代码库分析，然后切换到 **Q8 的 27B** 进行实现规划和代码生成。给出的技术理由是 35B-A3B 在阅读/分析方面更快，而据称 27B 生成的代码更干净、错误更少，尽管在用户的硬件上运行较慢。
    - 一位批评性的评论者认为，这种对比不是有效的消融实验，因为多个变量同时发生了变化：模型权重 **27B → 35B**、KV-cache 精度 **Q8 → FP16** 以及量化方案 **K-quant → I-Quant**。他们还提醒说，像“几乎一次性完成（nearly one-shotted）”这种 `n=1` 的结果太弱，不足以支持关于 KV-cache 影响或模型质量的结论。

### 3. 本地 LLM 硬件：3090 平台 vs Mac Studio

  - **[终于完成了我的 LLM 服务器：EPYC 9575F, 4× RTX 3090 (96GB VRAM), 768GB ECC RAM](https://www.reddit.com/r/LocalLLaMA/comments/1tx9tf2/finally_finished_my_llm_server_epyc_9575f_4_rtx/)** (热度: 632): **一位用户完成了一台基于 **Supermicro H13SSL-N** 主板、**AMD EPYC 9575F** (`64C/128T` Zen 5 架构)、`768GB` DDR5-5600 ECC RDIMM 内存和 **4× RTX 3090**（共计 `96GB` VRAM）的本地 LLM 推理服务器。该配置还包括 `1×2TB` 系统 NVMe、`2×3.94TB` 数据 NVMe，以及安装在 **Corsair 9000D** 机箱中的 `2050W` ATX 3.1 PSU。计划的工作负载包括用于高吞吐量小模型服务的 **vLLM** 和用于大型推理模型的 **llama.cpp**。所有 GPU 的功耗均被限制在 `250W`；两张 3090 插在主板上，另外两张前置安装，并使用了来自 [Thingiverse](https://www.thingiverse.com/thing:2804306) 的可打印风扇支架以增强气流。作者指出，其经济性高度依赖于购买时机和二手市场货源：`12×64GB` ECC RDIMM 每条约 `~$325`，`3× RTX 3090` 每张约 `~$650`，而 EPYC 处理器约 `~$3,800`，这使得按当前价格组装此类机器的可行性较低。** 评论中的主要技术诉求是针对 **Kimi K2.6**、**GLM 5.1** 或 **MiniMax 2.7** 等大型模型的实际推理基准测试，本质上是在询问目前一台价值 `$25k+` 的本地推理主机的实际性能表现。其他热门评论大多是非技术性的笑话，未增加实现细节。

    - 一项技术相关的请求要求提供在 **Kimi K2.6**、**GLM 5.1** 和 **MiniMax 2.7** 等大型 MoE/前沿开源模型上的真实推理基准测试，特别是为了量化一台价值 `$25k+` 的 4× RTX 3090 / EPYC 服务器在实践中的表现。建议的指标可能包括 tokens/sec、最大上下文行为、多 GPU 分片（sharding）开销以及 VRAM/RAM 卸载（offload）特性。
    - 一位评论者质疑了系统的平衡性，估计 **768GB ECC RAM 大约需要 `$30k`**，而 **EPYC CPU 需要 `$8k`**，暗示内存/CPU 平台的成本可能超过二手 GPU。另一位评论者认为使用 **4× RTX 3090** 会导致碎片化的 `96GB` VRAM 和高功耗，而单张 **RTX 6000 级别的 Blackwell** 显卡将提供统一的 VRAM、更新的 CUDA 支持以及 **NVFP4** 量化带来的更低内存占用优势。

  - **[老实说，双 3090 让我精疲力竭。正考虑转向 Mac Studio。](https://www.reddit.com/r/LocalLLM/comments/1txuqgl/honestly_dual_3090s_are_wearing_me_out_thinking/)** (热度: 200): **发布者目前运行一套 **双 RTX 3090** 的本地 LLM 配置，用于 **Llama 3/Qwen 70B 量化模型**，在使用 [`ExLlamaV2`](https://github.com/turboderp/exllamav2) 时报告速度约为 `40 tok/s`，但在将 70B 模型的上下文推过 `16k` 时遇到了 VRAM 限制。他们正考虑将这套设备更换为 **128GB Mac Studio**，接受速度下降至 `15 tok/s` 以换取更大的统一内存上下文（例如在 Q8 左右的模型上实现 `64k` 的代码库上下文），此外还能获得更低的发热/噪音以及更少的驱动/后端摩擦。**

## 较少技术性的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

# AI Discord 频道

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快就会发布全新的 AINews。感谢读到这里，这是一段美好的旅程。