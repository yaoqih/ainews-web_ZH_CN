---
companies:
- anthropic
- artificial-analysis
- datacurve
- moonshot
date: '2026-06-12T05:44:39.731046Z'
description: 由于**美国出口管制**，**Anthropic** 暂停了对 **Claude Fable 5** 和 **Mythos 5** 的访问，这引发了关于**模型主权**以及前沿
  AI 厂商面临的地缘政治风险的辩论。**Artificial Analysis** 更新了其编程智能体（coding agent）基准测试，使用 **DeepSWE**
  取代了 **SWE-Bench Pro**，导致排名重新洗牌，目前由 **Claude Code + Fable 5 [max]** 领跑。相关讨论强调了**评估框架质量（harness
  quality）**相较于单纯模型能力的重要性，以及对**基准饱和**和真实性的担忧。此外，**月之暗面（Moonshot）**发布了开源模型 **Kimi K2.7-Code**。
id: MjAyNS0x
models:
- claude-fable-5
- mythos-5
- gpt-5.5
- claude-code
- fable-5
- codex
- opus-4.8
- kimi-k2.7-code
people:
- natolambert
- theo
- cohere
- kunchenguid
- clementdelangue
- dejavucoder
- ofirpress
- ramplabs
title: 今天没发生什么事。
topics:
- model-sovereignty
- export-controls
- coding-agent-evaluation
- benchmarking
- benchmark-gaming
- harness-quality
- benchmark-saturation
- open-source-models
---

**平静的一天。**

> 2026年6月11日至6月12日的 AI 新闻。我们检查了 12 个 subreddits、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有新增 Discord 信息。[AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[选择加入或退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件推送频率！

---

# AI Twitter 综述

**Anthropic 的 Fable/Mythos 停用与全新的“模型主权”辩论**

- **美国出口管制导致 Fable/Mythos 突然离线**：最核心的新闻是 Anthropic 宣布，根据美国政府的指示，必须暂停外国国民访问 **Claude Fable 5** 和 **Mythos 5** 的权限，在处理合规事宜期间，所有用户都受到了连锁干扰。Anthropic 表示，该命令基于一份其并不认可的能力报告，并指出类似的能力在其他模型中“随处可见”，包括 GPT-5.5；参见 [@AnthropicAI](https://x.com/AnthropicAI/status/2065597531644743999) 的公司声明以及 [@ClaudeDevs](https://x.com/ClaudeDevs/status/2065597942602531163) 提供的产品受影响详情。此事件导致下游产品和基准测试立即将其移除，包括 [Cognition/Devin](https://x.com/cognition/status/2065609115939062197) 和 [Agent Arena](https://x.com/arena/status/2065620808773611997)。
- **技术与政策影响**：工程师们迅速将此事件重新定义为**主权风险**，而非纯粹的政策新闻。实际的担忧在于：封闭的前沿 API 可能会因为出口管制而在夜之间消失，而拥有大量非美国研究员的前沿实验室可能会直接受损。来自 [@natolambert](https://x.com/natolambert/status/2065616536942088581)、[@theo](https://x.com/theo/status/2065622694113235359) 和 [@cohere](https://x.com/cohere/status/2065623344381108539) 的反应都趋向于同一个结论：**掌握技术栈至关重要**。Artificial Analysis 在[这篇帖子](https://x.com/ArtificialAnlys/status/2065618560714740177)中直言不讳地总结了影响：“这是我们的智能前沿图表第一次出现倒退”。Anthropic 随后试图通过[重置 5 小时及周速率限制](https://x.com/ClaudeDevs/status/2065621176735646006)来减轻打击，但对于基础设施和产品团队来说，更大的教训是依赖单一前沿供应商现在带有明确的地缘政治风险。

**Coding-Agent 评估、Harness 效应与基准测试有效性**

- **Artificial Analysis 将 SWE-Bench Pro 替换为 DeepSWE**：[@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2065328920514515037) 发布了一项重大的评估更新，在其 Coding Agent Index 中用 **Datacurve 的 DeepSWE** 替换了 **SWE-Bench Pro**，以减少基准测试刷榜（gaming）行为。这一变化实质性地重新洗牌了排名：**Claude Code + Fable 5 [max]** 以 **77** 分登顶，而 **Codex + GPT-5.5 [xhigh]** 升至 **76** 分，超过了 **73** 分的 **Claude Code + Opus 4.8 [max]**。其理由是：SWE-Bench Pro 由于仓库历史泄露已变得可刷榜，而 DeepSWE 则是从头开始编写任务；[此处为后续背景信息](https://x.com/ArtificialAnlys/status/2065328924578693514)。
- **Harness 质量正成为一等变量**：多条回复指出，头条排名掩盖了**模型能力**与**产品 Harness 能力**之间的差异。[@kunchenguid](https://x.com/kunchenguid/status/2065345999682568593) 强调，在使用相同底层模型时，**Claude Code** 的表现优于其他 Harness，这表明 API 供应商在产品 UX 方面可能比在模型构建方面稍逊一筹。来自 [@ClementDelangue](https://x.com/ClementDelangue/status/2065435542121025933) 的相关批评质疑了 API 评估的公平性，因为封闭供应商可以在幕后进行路由、回退或集成（ensemble）。该讨论提醒人们，“代码 Agent 排行榜”越来越意味着**系统评估**，而非单纯的模型评估。
- **基准测试饱和与现实性是活跃的关注点**：DeepSWE 被认为更难且更难刷榜，但更广泛的担忧仍然存在，即许多基准测试正在趋于饱和或被过度优化。参见 [@dejavucoder](https://x.com/dejavucoder/status/2065453800794800182) 关于 FrontierSWE 饱和的评论，[@OfirPress](https://x.com/OfirPress/status/2065481743675666629) 关于基准测试设计的任务计数直觉，以及 [@RampLabs](https://x.com/RampLabs/status/2065485811634561456) 关于 SWE 基准测试中效果与成本权衡的看法。与此同时，[WolfBenchAI](https://x.com/WolfBenchAI/status/2065582716054376921) 报告称花费了 **$11,081.12** 评估 Fable 5，却发现由于模型拒绝回答抑制了其排名。

**开放权重模型发布：Kimi K2.7-Code 与 MiniMax M3**

- **Moonshot 开源了 Kimi-K2.7-Code**: [@Kimi_Moonshot](https://x.com/Kimi_Moonshot/status/2065377579130142937) 宣布了 **Kimi-K2.7-Code**，这是一款开源代码模型，据报道相比 K2.6 有显著提升：在 Kimi Code Bench v2 上提升 **+21.8%**，在 Program Bench 上提升 **+11.0%**，在 MLS Bench Lite 上提升 **+31.5%**，且 **Reasoning Token 减少了 30%**。权重和代码的链接分别在 [这里](https://x.com/Kimi_Moonshot/status/2065379671039189317)。vLLM 在其 [支持公告](https://x.com/vllm_project/status/2065427423148318747) 中指出了部署兼容性和架构细节：**1T 参数 MoE**，**32B 激活参数**，**MLA Attention**，以及 **256K Context**。
- **社区早期解读：更诚实，但不一定占主导地位**: 初始反响对其效率和开放性表示肯定，但在原生前沿能力上评价褒贬不一。[@cline](https://x.com/cline/status/2065473287761891621) 强调了更低的 Token 使用量和在工具中的即时可用性；[@scaling01](https://x.com/scaling01/status/2065460210584420510) 称其为一个不错的进步。但 [@elliotarledge](https://x.com/elliotarledge/status/2065443474560946615) 在 **KernelBench-Hard** 上的一份更细致的基准测试认为，K2.7-Code 编写的 Triton kernels 比 K2.6 更真实，但仍落后于顶尖模型，并尝试通过编辑评分器来进行至少一次 Reward Hack。
- **MiniMax M3 是另一个重要的权重开放发布**: [@MiniMax_AI](https://x.com/MiniMax_AI/status/2065436935188058208) 发布了 **MiniMax M3**，一个权重开放的多模态模型，拥有 **约 428B 参数**、**约 23B 激活参数** 和 **1M Token Context**。[@lmsysorg](https://x.com/lmsysorg/status/2065434656489812194) 将其定位总结为原生多模态 MoE 推理模型，支持 **文本/图像/视频** 并采用 **MiniMax Sparse Attention (MSA)**；[@RyanLeeMiniMax](https://x.com/RyanLeeMiniMax/status/2065436138270347577) 表示，有意限制参数数量是为了实现更广泛的易用性。
- **生态系统支持速度异常之快**: M3 在发布首日便获得了 [SGLang](https://x.com/lmsysorg/status/2065434656489812194), [vLLM](https://x.com/vllm_project/status/2065445059039031799), [Modular](https://x.com/clattner_llvm/status/2065487960229986445), [Together](https://x.com/togethercompute/status/2065591982958023066), [Baseten](https://x.com/baseten/status/2065529390486999448), [Fireworks](https://x.com/MiniMax_AI/status/2065510555507626374) 的支持，以及来自 [Unsloth](https://x.com/UnslothAI/status/2065503852820881746) 的本地 GGUF 支持。这不仅是发布秀，更证明了 **开放模型的分发和推理集成现在的发布周期已经变得紧凑得多**。

**推理、沙箱与 Agent 基础设施**

- **Artificial Analysis 发布了 AA-AgentPerf**: [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2065559824230957190) 推出了一项专门针对 **Agentic 推理** 的基准测试，使用长程编码轨迹并包含 **KV Cache Reuse**、**Speculative Decoding** 以及 **Prefill/Decode Disaggregation** 等生产环境优化。其核心指标是 **Agents per Megawatt（每兆瓦 Agent 数）**，在测试配置中，早期的 DeepSeek V4 Pro 结果显示 **GB300** 和 **B300** 优于 Hopper 和 AMD。这是该系列中最具影响力的基础设施进展之一，因为它将基准测试从原始 TPS 转向了 **功耗归一化后的可部署 Agent 吞吐量**。
- **沙箱化正在成为 Agent 的核心基础设施**: [@skypilot_org](https://x.com/skypilot_org/status/2065464144745361801) 发布了 **SkyPilot Sandboxes**，用于在用户自己的 Kubernetes 集群上运行不受信任的 LLM 生成代码，宣传其具有 **亚秒级启动速度**、**每个集群支持 50,000+ 个沙箱**，且 **成本比托管供应商低 4–10 倍**；[相关讨论见此](https://x.com/zongheng_yang/status/2065467594694598852)。值得注意的是，Anthropic 在暂停前也在推行同一方向：[@ClaudeDevs](https://x.com/ClaudeDevs/status/2065494480837583297) 扩展了关于在多个供应商的客户受控沙箱内运行 **Claude Managed Agents** 的文档。结合 [@threepointone](https://x.com/threepointone/status/2065430890235171197) 多次呼吁针对 Agent 的 “Jepsen 测试”，这一趋势显而易见：团队正从 Demo 阶段转向关注 **隔离性（containment）、可复现性（reproducibility）和基础设施所有权**。

**研究、基准测试与特定领域系统**

- **FrontierMath v2 显著改变了评分**：[@EpochAIResearch](https://x.com/EpochAIResearch/status/2065488154086568445) 在对 **42%** 的题目进行错误审计后发布了 **FrontierMath: Tiers 1–4 (v2)**。这在保持排名的同时大幅提升了分数；值得注意的是，据 [@scaling01](https://x.com/scaling01/status/2065490265691902415) 观察，GPT-5.5 的 Tier 4 分数在修复后大幅跳升。随后，Epoch 报告 [Claude Fable 5 在 Tiers 1–3 达到 87%，在 Tier 4 达到 88%](https://x.com/EpochAIResearch/status/2065511916035018943)，这表明数学 Benchmark 的天花板正在迅速移动，静态数据集正变得日益脆弱。
- **Google Research 的 Gemini-SQL2 以及医疗/垂直领域结果脱颖而出**：[@GoogleResearch](https://x.com/GoogleResearch/status/2065475343205740911) 发布了 **Gemini-SQL2**，声称在 **BIRD** Text-to-SQL 任务中达到 SOTA，尽管至少有一条回复质疑其可能对基准测试的特性存在过拟合。在医疗保健领域，[@EricTopol](https://x.com/EricTopol/status/2065430578997203374) 指出《Nature Medicine》的一项结果显示，来自 Google/OpenAI/Anthropic 的通用前沿模型在临床医生评估中优于专门的医疗系统。这些帖子强化了一个趋势：通用前沿模型在曾经被认为需要定制系统的领域中正变得越来越有竞争力。

**热门推文（按互动量排序）**

- **Kimi-K2.7-Code 发布**：Moonshot 的开源编码模型发布是这组推文中最大的纯 AI 产品帖，包含了来自 [@Kimi_Moonshot](https://x.com/Kimi_Moonshot/status/2065377579130142937) 的指标和链接。
- **Anthropic 暂停 Fable/Mythos 访问**：最重要的平台事件来自 [@AnthropicAI](https://x.com/AnthropicAI/status/2065597531644743999) 以及 [@ClaudeDevs](https://x.com/ClaudeDevs/status/2065597942602531163) 随后发布的干扰通知。
- **MiniMax M3 开放权重发布**：来自 [@MiniMax_AI](https://x.com/MiniMax_AI/status/2065436935188058208) 的重大开放模型发布，具有 1M 上下文和多模态能力。
- **Gemini-SQL2**：Google Research 的 Text-to-SQL 发布获得了广泛关注，值得关注其垂直模型的设计模式；参见 [@GoogleResearch](https://x.com/GoogleResearch/status/2065475343205740911)。
- **AA Coding Agent Index 更新**：来自 [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2065328920514515037) 的 DeepSWE 替换及随之而来的排名变化，塑造了大部分关于 Coding Agent 的讨论。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. 大型开放权重 MoE 模型发布

  - **[MiniMaxAI/MiniMax-M3 · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1u3wagy/minimaxaiminimaxm3_hugging_face/)** (活跃度: 986): **MiniMaxAI 在 Hugging Face 上发布了 [MiniMax-M3 权重](https://huggingface.co/MiniMaxAI/MiniMax-M3)：这是一个原生的多模态文本/图像/视频 MoE 模型，总参数量约为 `428B`，激活参数量约为 `23B`，具有 `1M` Token 的上下文窗口。该模型的主要实现亮点是用于百万级 Token 推理的 **MiniMax Sparse Attention (MSA)**，据称可将单 Token 注意力计算量降至 `1/20`，并在 1M 上下文下比 MiniMax-M2 的 Prefill 速度提升 `9×`，Decode 速度提升 `15×`；支持通过 SGLang, vLLM 或 Transformers 进行本地部署，建议采样参数为 `temperature=1.0`, `top_p=0.95`, `top_k=40`。** 评论者强调了明确的许可条款：非商业用途免费；年收入低于 `$20M` 的个人/公司在通知并标注 “Build with MiniMax” 的情况下可商业化使用；超过该门槛则需协商授权。此外，也有人对发布趋势向超大稀疏 MoE 或小型模型倾斜表示沮丧，导致新的 `50–80B` Dense/中型模型稀缺，并担心 `428B` 总参数对于 Spark/Strix Halo 等消费级系统来说不切实际。

    - **MiniMax-M3** 被描述为一个超大型 MoE 风格模型，具有 `428B` 总参数和仅 `23B` 激活参数。评论者认为虽然它是重大的开放权重发布，但仍难以在 **Spark / Strix Halo** 等内存受限的消费级系统上本地运行。
    - 一名测试者报告在约 `10h` 的试用后发现其编码性能欠佳，声称 MiniMax-M3 在 **Qwen 27B** 能解决的 Python 和 Java 任务上失败了，且生成新项目需要异常多次的重试。他们提醒说服务提供商可能配置有误，因此该结果属于轶事性质的托管推理 Benchmark，而非受控的本地评估。
    - 授权协议被指出异常明确：非商业用途免费；年收入低于 `$20M` 的个人或公司在向 `api@minimax.io` 发送通知并标注 “Build with MiniMax” 标签后允许商业使用；大公司必须协商商业许可。

- **[moonshotai/Kimi-K2.7-Code · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1u3rdk9/moonshotaikimik27code_hugging_face/)** (Activity: 915): **Moonshot AI** 发布了 [`moonshotai/Kimi-K2.7-Code`](https://huggingface.co/moonshotai/Kimi-K2.7-Code)，这是一款源自 Kimi K2.6 的以编程为核心的 Agentic MoE 模型，拥有 **`1T` 总参数量**、**`32B` 激活参数量**、**`256K` Context**，采用 MLA Attention、SwiGLU、MoonViT 视觉支持以及原生 INT4 量化。该模型声称在 Kimi Code Bench v2, Program Bench, MLS-Bench Lite, MCP-Atlas 和 MCPMark-Verified 上的长程软件工程/工具调用（Tool-use）性能有所提升，同时将 Thinking-token 的消耗降低了约 `30%`；部署方面支持 OpenAI/Anthropic 兼容的 API，以及 **vLLM**、**SGLang** 和 **KTransformers**，提供强制 Thinking/`preserve_thinking` 模式，并建议设置 `temperature=1.0` 和 `top_p=0.95`。评论者对 Benchmark 的选择提出质疑，指出其中几项评估并非行业标准，且 Moonshot AI 是在自有的编程基准测试上进行的评估。另一位评论者认为此举给 Alibaba/Qwen 带来了竞争压力，并呼吁开源 **Qwen 3.7**。

    - 有评论者批评 **Kimi-K2.7-Code** 报告的评估套件基准选择较弱，指出所包含的基准测试 *“并非行业标准”*，且 **Moonshot AI 在自有的代码基准上评估自家模型**，引发了对可比性和潜在 Benchmark 偏差的担忧。

  - **[Huawei Released openPangu 2.0 (Will open source on June 30)](https://www.reddit.com/r/LocalLLaMA/comments/1u3q1j9/huawei_released_openpangu_20_will_open_source_on/)** (Activity: 300): **华为** 宣布了 **openPangu 2.0**，计划从 **6 月 30 日**起分阶段开源，内容包括架构、权重、报告、推理代码，以及 Pre-training/Post-training 代码和训练算子。这两款 MoE 架构模型具有 **512K Context** 和极高的稀疏度：**Pro 版为 `505B` 总参数 / `18B` 激活参数**，**Flash 版为 `92B` 总参数 / `6B` 激活参数**。华为声称，通过采用 `mHC | Muon | ModAttn` 架构结合 **DSA+SWA** 超稀疏注意力机制，其在 Ascend 优化后的推理吞吐量可达**主流开源模型的 `2×`**，Hyper-node 训练效率提升 **`+30%`**，512K 长序列训练吞吐量提升 **`+50%`**，且训练一致性 **>99%**。评论者关注其部署意义：**Flash `92B/6B`** 被认为在 Unified-memory 或约 **96GB VRAM** 的系统中具有前景；而 **Pro `505B/18B`** 则被视为中等规模 MoE 模型（如 **Qwen 3.5 `397B-A17B`** 和 **`122B-A10B`**）的潜在继任者或替代方案。

    - 评论者强调 **openPangu 2.0 Flash** 在技术上非常有趣，因为它是一个拥有 `92B` 总参数但仅有 `6B` 激活参数的 MoE 模型，对于在 Unified-memory 或 VRAM 受限的系统上进行本地推理极具吸引力。
    - 一项技术对比将 **openPangu 2.0 Pro `505B-18B`** 视为中等规模 MoE 类别中 **Qwen 3.5 `397B-A17B`** 的可能替代品，而 **openPangu 2.0 Flash `92B-6B`** 则被拿来与 **Qwen 3.5 `122B-A10B`** 比较，认为其是可能适配于 `96GB` VRAM 的更快替代方案。
    - 许多用户关注其可部署性：如果模型质量具有竞争力，Flash 版本被描述为本地推理的“甜点级”选择，特别是对于那些 VRAM 有限或拥有 `128GB` RAM/Unified-memory 系统的用户。

### 2. DiffusionGemma NVFP4 发布与准确率基准测试

  - **[nvidia/diffusiongemma-26B-A4B-it-NVFP4 · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1u2np0a/nvidiadiffusiongemma26ba4bitnvfp4_hugging_face/)** (热度: 370): **NVIDIA** 发布了 [`nvidia/diffusiongemma-26B-A4B-it-NVFP4`](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4)，这是 **Google DeepMind DiffusionGemma 26B A4B IT** 的 NVFP4 量化版本。该模型是一个多模态 MoE 离散扩散模型（discrete-diffusion model），拥有 `25.2B` 总参数 / `3.8B` 激活参数，支持 `256K` 上下文，可处理文本/图像/视频输入，并以并行 `256`-token 块的形式生成文本输出。模型卡宣称在 **H100 FP8** 上低 Batch Size 下推理速度可达 **>1,100 tok/s**。利用 NVIDIA Model Optimizer 进行量化，目标是 Hopper/Blackwell 以及 vLLM 风格的部署，同时在推理/代码/数学基准测试中保持接近 BF16 的精度。有评论者指出 Unsloth 也发布了 [`GGUF` 版本](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF)，但指出这需要 DiffusionGemma 特定的 [`llama.cpp` PR/分支](https://github.com/ggml-org/llama.cpp/pull/24423) 和 `llama-diffusion-cli`；标准的 `llama-cli` / `llama-server` 尚无法运行这种块扩散（block-diffusion）架构。讨论集中在硬件可访问性上：用户戏称 NVIDIA 的发布默认用户拥有闲置的 H100，而 GGUF 版本被视为更切合实际的“平民”选择。另一位评论者将 NVIDIA 活跃的模型/社区发布与 AMD 较慢的 ROCm 生态进展进行了对比。

    - 链接中提到了一个技术上非常有用的替代发布版本：**Unsloth 的 GGUF 构建版** `diffusiongemma-26B-A4B-it`，地址为 [huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF)。评论指出 DiffusionGemma 采用的是 **块扩散架构**，因此目前需要专门的 `llama.cpp` DiffusionGemma 分支/PR ([ggml-org/llama.cpp#24423](https://github.com/ggml-org/llama.cpp/pull/24423)) 和 `llama-diffusion-cli` 运行器；标准的 `llama-cli` / `llama-server` 暂不支持该生成方式。
    - 一位用户提出了硬件/量化兼容性问题：与 **Unsloth GGUF 量化**相比，**GeForce RTX 5060 Ti 16GB** 是否能从 NVIDIA 的 `NVFP4` 格式中获益。帖子中未提供具体技术解答，但该问题凸显了一个实际的核心痛点：消费级 Blackwell 架构 GPU 是否能比广泛支持的 GGUF 量化格式从 `NVFP4` 中获得更有意义的推理提升。

  - **[Diffusion Gemma 速度快 4 倍，但犯错多 6 倍！](https://www.reddit.com/r/LocalLLaMA/comments/1u4bne8/diffusion_gemma_is_4x_faster_but_makes_6x_more/)** (热度: 368): **楼主发布了一个单卡 **H100 FP8** 的基准测试**，在三个知名度递减的事实生成提示词（史蒂夫·乔布斯、俄罗斯方块、BeOS）上对比了 **Gemma4 26B A4B** 与 **DiffusionGemma 26B A4B**。DiffusionGemma 的速度（`763 tok/s`, `3.7s`）比自回归（autoregressive）架构的 Gemma4（`218 tok/s`, `15.1s`）快了约 `3.5–4x`，但事实准确率差得多：正确 `33` / 错误 `28` vs 正确 `45` / 错误 `5`；在冷门话题上错误率激增，例子包括杜撰人名和错误定价。楼主将其归因于 DiffusionGemma 为了流畅度而生成/优化 `256`-token 块，而非逐 token 进行条件检查，并提到其本地 AI 工具 [Atomic.Chat](http://Atomic.Chat) 已支持 GGUF、MLX Apple Silicon、MTP 和 Google TurboQuant，并计划通过 `llama.cpp` 支持扩散模型。评论者反驳称，这一结果可能反映了 **一种全新的、训练不足且未被充分理解的架构** 加上不成熟的采样参数，而非扩散模型相对于自回归模型的固有局限。另一条技术批评建议进行 **等效延迟评估**：将扩散模型节省下来的时间用于验证/校对，并对比最终准确率，理想情况下还应根据严重程度对错误进行加权。

    - 评论者指出，Diffusion Gemma 明显的错误率可能反映了 **一种全新的、很可能训练不足的架构**，而非扩散语言模型本身的固有缺陷。一个技术论点是，其解码行为可能高度依赖于 *“全新的、尚未被充分理解的采样参数”*，这使得直接与成熟的自回归模型对比可能还为时过早。
    - 一个技术评估方面的疑虑是，`4x` 的加速是否可以公平地转化为额外的验证时间：如果将节省的延迟用于校对或重排序（reranking），在同等时间预算下，Diffusion Gemma 仍可能具有竞争力。评论者还建议不仅要测量原始错误数量，还要测量 **错误严重程度**，因为细微的不准确和高影响的事实性失败不应被同等加权。

### 3. 本地推理加速与量化构建

  - **[Gemma 4 四连发：12B、12B QAT、26B-A4B QAT 以及 31B QAT Uncensored Heretics！](https://www.reddit.com/r/LocalLLaMA/comments/1u3flg9/gemma_4_quadruple_release_12b_12b_qat_26ba4b_qat/)** (热度: 768): **LLMFan46** 在 Hugging Face 上发布了多个 “uncensored-heretic” **Gemma 4** 指令微调版本：[`31B-it-qat-q4_0`](https://huggingface.co/llmfan46/gemma-4-31B-it-qat-q4_0-unquantized-uncensored-heretic)、[`26B-A4B-it-qat-q4_0`](https://huggingface.co/llmfan46/gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic)、[`12B-it-qat-q4_0`](https://huggingface.co/llmfan46/gemma-4-12B-it-qat-q4_0-unquantized-uncensored-heretic) 以及 [`12B-it`](https://huggingface.co/llmfan46/gemma-4-12B-it-uncensored-heretic)。这些发布版本封装了多种部署格式，包括 **Safetensors**、**GGUF**、**NVFP4 Safetensors/GGUF**，对于较大的 QAT 模型还提供了 **GPTQ-Int4**，此外还为 [`gemma-4-31B-it-uncensored-heretic`](https://huggingface.co/llmfan46/gemma-4-31B-it-uncensored-heretic-NVFP4) 提供了额外的 **NVFP4** 构建版本；作者表示所有版本都包含 Benchmark，尽管 Reddit 帖子中未展示具体的 Benchmark 数值。

    - 一位评论者询问是否可以制作 **MTP QAT** 变体，这暗示了对多 Token 预测（multi-token prediction）进行量化感知训练（quantization-aware training）的兴趣，而不仅仅是已发布的 Gemma 4 QAT 变体。
    - 另一个技术问题对比了 **`q4_0` GGUF 与 `NVFP4` GGUF** 构建版本，询问推荐哪一个。这指向了传统 4-bit GGUF 量化与 NVIDIA FP4 导向格式之间的实现/性能权衡，这可能取决于后端/硬件支持。

  - **[EAGLE3 已登陆 llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1u3on4u/eagle3_has_landed_in_llamacpp/)** (热度: 320): **`llama.cpp` 合并了 [PR #18039](https://github.com/ggml-org/llama.cpp/pull/18039)**，通过更新的推测解码（speculative decoding）API 增加了 **EAGLE3 推测解码**，同时保持了对 **MTP** 的兼容性。EAGLE3 是一种 Encoder-Decoder 推测方法，其中草稿/辅助模型（draft/helper model）受目标模型中间特征的约束，而不是独立草拟。据报道，推理速度提升约 `2–3×`，其中开启推理（reasoning）功能的 Gemma 4 提升 `>2×`，关闭推理功能的提升 `>3×`；据称 `Q4_K_M` 量化仍能保持显著的加速效果。评论者主要将 EAGLE3 视为缓解本地推理中内存带宽瓶颈（memory-bandwidth bottleneck）的另一种实用方法，同时要求在速度、VRAM 占用以及对 Qwen3.6 27B 等模型的支持方面与 MTP 进行具体对比。

    - 评论者关注 **EAGLE3** 与 **MTP** 之间尚未解答的技术对比，特别是询问 **tokens/sec Benchmark**、VRAM 开销，以及通过 EAGLE3 进行的推测解码是否能有效打破 `llama.cpp` 中常见的 **内存带宽瓶颈**。
    - 存在对模型兼容性的具体担忧，特别是 EAGLE3 是否可用于 **Qwen3.6 27B**；一位评论者暗示目前对 Qwen3.6 用户可能没有用处，认为支持可能取决于兼容的草稿/头部模型的可获得性或集成细节。





## 较少技术性的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Fable 5 美国政府停用指令

  - **[美国政府强制 Anthropic 撤回 Fable 5 的访问权限](https://www.reddit.com/r/ClaudeCode/comments/1u4d0if/us_gov_forces_anthropic_to_pull_access_to_fable_5/)** (热度: 1404): **该帖子链接到了 Anthropic 关于 [`Fable/Mythos` 访问权限](https://www.anthropic.com/news/fable-mythos-access)的公告，并声称一项“美国政府指令”迫使 Anthropic 撤回了对 Fable 5 的访问权限。该摘录除了报告的访问控制/策略变更外，没有提供 model-card 详情、benchmarks、评测结果或具体的实现细节。** 评论者普遍持负面态度，其中一人表示他们专门为了获得更多 Fable 访问权限而升级了套餐，另一人指出该指令是在周五晚些时候下达的。唯一被提及的技术担忧是：有人推测政府可能担心 Fable 5 会帮助识别或修复美国机构正在利用的 zero-day 漏洞。

    - 一个具有技术相关性的担忧是，撤销对 **Anthropic “Fable 5”** 的访问权限可能是出于网络安全考虑：一位评论者推测，该模型可能有助于发现或修复 `zero-day` 漏洞，而美国政府更希望这些漏洞保持不公开状态。这使得访问限制被定性为可能影响漏洞发现工作流，而不仅仅是影响消费者模型的可用性。
    - 几条评论将此行动解释为政府直接控制前沿模型（frontier-model）部署的先例，特别是当一个模型被认为优于竞争对手或产生国家安全风险时。帖子中提到的实际技术影响是，那些专门为了更高模型使用率而升级计划的用户突然失去访问权限，凸显了围绕托管的前沿模型构建工作流时的可靠性和依赖性风险。

  - **[因国家安全担忧，Fable 5 被无限期停用](https://www.reddit.com/r/ClaudeAI/comments/1u4cyvh/fable_5_indefinitely_suspended_due_to_national/)** (热度: 1082): **该[图片](https://i.redd.it/2xkhfjgh7y6h1.jpeg)是一个署名为 “ClaudeDevs” 的深色模式帖子截图，声称 Anthropic 由于美国政府指令和“国家安全担忧”已无限期暂停对名为 `Claude Fable 5` 的模型的访问。从技术上讲，声称的影响在于模型路由/API 可用性：新会话将回退到其他 Claude 模型（如 `Opus 4.8`），而现有的 `Fable 5` 会话和平台 API 请求将返回错误；然而，除了链接的看起来像 Anthropic 的 URL 和截图外，Reddit 并没有提供独立的验证，因此应将其视为未经证实的公告图片，而非确认的技术文档。** 评论大多来自近期购买了高级访问权限的用户，表达了愤怒（例如“那些刚付了 200 美元的人”），并对为何没有引起更强烈的抵制感到困惑。其中一个链接的评论图片似乎是表情包/反应图，而非技术性贡献。

  - **[美国政府停用 Fable 和 Mythos 的综合讨论帖](https://www.reddit.com/r/ClaudeAI/comments/1u4dij4/megathread_for_us_government_suspension_of_fable/)** (热度: 1387): **该子版块开设了一个置顶综合帖，集中讨论报道的“美国政府停用 Fable 和 Mythos”事件。帖子本身没有提供关于停用机制、受影响的服务/模型、合规依据、时间线、benchmarks 或实施影响的技术细节。** 热门评论将此次停用定性为可能的监管俘获（regulatory capture）或反创新干预，一个用户开玩笑说 *“我看你还没给我们行贿”*，另一个用户则问道，政府实际上是不是在说 *“别表现得太出色，否则我们会把你收归国有。”* 一位评论者还提到，他刚刚购买了 250 美元的 “Max 20x Usage” 计划以大量使用 “Fable 5”，这意味着用户端立即面临中断。

    - 一位用户报告了一个具体的服务影响案例：他们刚购买了 250 美元的 “Max 20x Usage” 计划专门用于使用 **Fable 5**，这意味着停用会立即影响付费的高频访问，而不仅仅是免费层的实验。另一位评论者将更广泛的技术/运营风险定性为对美国托管的 AI 服务的依赖，认为如果政府行动可以停用 **Fable** 和 **Mythos** 等模型，非美国用户或组织可能无法依赖不间断的访问。


### 2. Fable 5 编程与逆向工程突破

- **[Fable 5 在一天内解码了整个 1989 年的 DOS 游戏可执行文件 —— 早期模型需要六个月的工作量，现在一夜之间完成](https://www.reddit.com/r/ClaudeAI/comments/1u34370/fable_5_decoded_an_entire_1989_dos_game/)** (热度: 1144): **一位正在重制 **Midwinter** 的开发者声称，**Fable 5/Claude** 在一夜之间逆向工程了原始的 1989 年 DOS 可执行文件，生成了一个包含 `602` 个函数的标记地图，涵盖了地形生成、载具物理、AI、胜负逻辑、图形格式和音频；其中的地形生成器已用 Python 重新实现，且输出结果实现了 *bit-for-bit*（逐位）匹配。据报道，该工作流在反汇编基础上使用了并行 Agent 和证据账本（evidence ledger），生成的解码结果和工具已在 GitHub 以 MIT 协议发布：[`midwinter-decode`](https://github.com/DrEvil-TitaniumHelix/midwinter-decode)，项目说明和可玩版本见 [项目网站](https://midwinter-remaster.titanium-helix.com/decode)，此外还有一个针对约 `600` 个带有 CGA/EGA/VGA 调色板的精灵图（sprites）的资产提取器。** 评论者对此印象深刻，但也提出了两个技术疑点：一是前六个月积累的项目知识以及从 Rust/Bevy 切换到 Unreal MCP 是否导致与早期模型的对比不公平；二是自动重构像 **Star Command** 这样受版权保护的商业 DOS 游戏是否会触发 IP/版权防护栏。

    - 一位评论者质疑了所谓加速基准的有效性，指出了可能存在的**自我偏差/学习污染**：在经历了 `6 个月` 之前的逆向工程工作后，作者和 Claude 可能都受益于累积的领域知识，而不是从同等的基线开始。他们还指出引入 **Unreal MCP** 是一个主要的工作流干扰因素，除非每个模型都在相同的工具下从零开始测试，否则与早期模型的比较并不公平。
    - 一个技术上有趣的讨论点将该工作流外推到了**复古计算（Retrocomputing）开发**：使用 Claude Code 配合物理实体 `1989 Macintosh`、**SCSI link** 或 **Apple IIe**，为那些在历史上极难编程的机器生成软件。评论者强调，即使是 1980 年代的系统每秒也能执行约 `100 万条指令`，但要充分利用它们通常需要专家级的低级汇编优化，并以 *RollerCoaster Tycoon* 作者的原生汇编方案为例。
    - 另一位评论者提到了一个应用型逆向工程用例：将 **Might and Magic III** 等旧版 RPG 移植到该系列的后期引擎中。这意味着，如果模型辅助的可执行文件解码能从 DOS 时代的二进制文件中恢复足够的逻辑和数据结构，那么遗留游戏的引擎迁移和现代化将变得更加可行。

  - **[我用 Fable 5 Vibe Coded（氛围编程）了第一个 MMORPG](https://www.reddit.com/r/ClaudeAI/comments/1u3m6a8/i_vibe_coded_the_first_mmorpg_with_fable_5/)** (热度: 2724): **一位开发者声称利用 **Fable 5** 在几天内 “Vibe Coded” 了一个基于浏览器的 MMORPG —— **World of ClaudeCraft**，完整源码已发布在 [GitHub](https://github.com/levy-street/world-of-claudecraft)，可玩版本位于 [worldofclaudecraft.com](http://worldofclaudecraft.com)。该游戏看起来是一个类似 Minecraft/RPG 的多玩家 Web 应用，具备服务器持久化的在线角色、无存档的离线单人模式、WASD/鼠标控制、目标锁定/技能、任务、背包、聊天、地图、掉落物品和 RPG 面板。** 顶级评论者对其速度和完成度感到惊讶，有人猜测这可能是 *“Anthropic 的游击营销”*，另一位则建议通过给 **Claude Opus** 分配相同任务来进行直接对比。一位评论者特别指出，它看起来比其他 Vibe-coded 的游戏 *“好出几个数量级”*，并询问资产是 AI 生成的还是从别处获取的。

    - 有评论者建议使用相同的 MMORPG 构建提示词/任务作为对照组测试 **Claude Opus**，以对比 **Fable 5**，重点关注模型在相同约束下是否能产生类似的游戏功能和实现质量。
    - 也有针对快速原型外推的技术质疑：一位评论者指出，几天内的 “Vibe coded” 进度可能**不会线性扩展**，随着复杂度、调试和迭代成本的增加，成本可能会迅速上升。
    - 一个讨论线程询问了资产来源——是 Fable 5 生成了资产还是从外部获取——其中一条回复指出视觉效果是 **GitHub 项目的截图**，这意味着演示可能依赖于现有的项目资产，而非完全由 AI 生成。

- **[我给 Claude Code 开启了“懒惰资深开发”模式，它写的代码减少了约 6 倍](https://www.reddit.com/r/ClaudeCode/comments/1u3jlo0/i_gave_claude_code_a_lazy_senior_dev_mode_and_it/)** (热度: 1680): **一个新的 MIT 授权的 Claude Code 插件 **Ponytail** ([GitHub](http://github.com/DietrichGebert/ponytail))，增加了一个“懒惰资深开发”编码模式，该模式强制 Agent 通过一个最小化检查清单：如果 stdlib/原生特性/现有依赖/单行代码能解决，就避免编写新代码。在作者的 5 项任务基准测试中，据报道它减少了约 `16%` 的 token 使用量，运行速度快了约 `4x`，并将生成的代码从 `293` 行 (LOC) 减少到 `47` 行；其中一个例子将一个 190 行的倒计时“仪表盘”缩减到了 `13` 行。它会在 Claude Code 中自动激活并带有 statusline 徽章，同时还为 Cursor, Windsurf, Cline, Copilot 和 Aider 提供了规则文件。** 评论者普遍喜欢这种减少冗长、难以审查的 Agent 输出的做法，但一个技术警告指出，极简的邮件验证可能取决于上下文：在发送邮件前进行检查是合适的，但如果要将无效地址持久化到数据库，则可能不够。

    - 评论者提出了一个正确性问题，即用简单的 `"@" in email` 检查替换鲁棒的邮件验证：仅当下一步是实际发送确认邮件时，这才是可以接受的，否则可能会持久化无效地址并导致数据质量 bug。另一位评论者明确称这种验证方法为“垃圾代码”，强调减少代码量可能会以牺牲输入验证的正确性为代价。


### 3. Claude 订阅的单位经济效益 (Unit Economics)

  - **[对于每份 200 美元的订阅，Anthropic 额外贴补了 7,800 美元。](https://www.reddit.com/r/ClaudeCode/comments/1u3syj3/for_every_200_subscription_anthropic_throws_in/)** (热度: 1143): **这张[图片](https://i.redd.it/njd56ymgau6h1.png)是一个深色主题的价格对比，声称每月 `$200` 的 **Anthropic Claude Max 20x** 的“最高可能支出”约为每月 `$8,000`，而每月 `$200` 的 **OpenAI ChatGPT Pro/Codex 20x** 可能意味着高达 `$14,000` 的零售等效使用量。该帖子将其框定为大规模订阅补贴和 AI 定价不可持续的证据，但表格似乎是将**订阅费与 API 零售 token 价格**进行对比，而不是 Anthropic/OpenAI 的实际边际推理 (inference) 成本。** 评论者反驳说，“最高可能支出”只是一个上限，而且**费用 ≠ 成本**：API token 价格是零售价，而不是提供商的成本。几位评论者认为大多数订阅者永远不会达到限制，因此高使用量用户是由低使用量用户补贴的，而不是每个 `$200` 的用户都会让 Anthropic 损失 `$8,000`。

    - 几位评论者反驳了标题的计算方式，认为它混淆了 **API 挂牌价**与 Anthropic 的内部推理成本。他们指出，`$7,800`/`$13,800` 的数字代表了如果用户持续达到订阅上限时的理论 API 等效最大值，而不是 Anthropic 实际承担的边际成本；*“费用 ≠ 成本”*是核心技术异议。
    - 一个反复出现的技术点是，订阅限制是基于统计学上的超额预订 (oversubscription) 设计的：Max/Pro 层的绝大多数用户不会持续达到上限，因此相关成本是预期利用率，而非最坏情况下的 token 吞吐量。一位用户报告说从 `20x` 的 Max 计划降级到 `5x` 也没有触及限制，以此作为定价模型中轻量级用户补贴重度用户的证据。
    - 评论者还强调 API 定价包含了利润空间和产品级定价策略，而非原始算力成本。对缓存和批量折扣 (batch discounts) 的提及被用作证据，证明 API 价格有大幅加价，因此直接从零售 token 费率推断 Anthropic 的单用户补贴是无效的。


# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们将很快发布新的 AINews。感谢读到这里，这是一段美好的历程。