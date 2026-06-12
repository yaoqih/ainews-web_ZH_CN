---
companies:
- anthropic
- recursive-si
- nvidia
date: '2026-06-11T05:44:39.731046Z'
description: '在公众的强烈抵制下，**Anthropic** 撤销了针对 **Claude Fable 5** 的隐蔽降级政策，这引发了关于治理、透明度以及前沿
  AI 模型访问权的辩论。该模型展现了强大的能力，但在基准测试中表现不一，包括在 **WeirdML** 上获得 87.8% 的得分，以及在 **FrontierSWE**
  上名列前茅；但在实际应用中，其高昂的成本和不稳定的行为也成为了关注焦点。


  与此同时，由 **Richard Socher** 领导的 **Recursive SI** 发布了一套自动化开放式发现系统。该系统在 **NVIDIA SOL-ExecBench**、**NanoGPT
  Speedrun** 以及 **NanoChat autoresearch** 上均取得了最先进（SOTA）的成果，并开源了相关发现，同时提升了效率指标。'
id: MjAyNS0x
models:
- claude-fable-5
- nanogpt
people:
- richard_socher
title: 今天没发生什么特别的事。
topics:
- model-governance
- model-transparency
- benchmarking
- automated-research
- optimization
- open-sourcing
- model-behavior
- cost-efficiency
---

**平静的一天。**

> 2026/6/10-2026/6/11 的 AI 新闻。我们检查了 12 个 subreddits，[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有新增 Discords。[AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。提醒一下，[AINews 现已成为 Latent Space 的一部分](https://www.latent.space/p/2026)。你可以[选择订阅/取消订阅](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件发送频率！

---

# AI Twitter 摘要

**Anthropic 的 Fable 5 发布，隐蔽式 sandbagging 引发抵制，以及模型行为辩论**

- **静默降级政策在公众抵制后迅速撤回**：多篇帖子关注 Anthropic 针对某些 AI 研究相关用例秘密降级 **Claude Fable 5** 的决定，随后在约一天内撤销。[Simon Willison](https://x.com/simonw/status/2064918665859080392) 对撤回表示欢迎；[MTS live](https://x.com/MTSlive/status/2064922000020398331) 总结称 Anthropic 正在逆转该政策；[Kim Monismus](https://x.com/kimmonismus/status/2065003618710008084) 将其描述为在研究人员批评后的退却。最强烈的技术批评不在于安全护栏的存在，而在于**模型层的不透明行为**：[Code Star](https://x.com/code_star/status/2064931207310118940) 认为安全护栏是正常的，但“毫无警告的掩盖”违反了用户/服务商契约，而 [Clement Delangue](https://x.com/ClementDelangue/status/2065069246124613999) 称避免 AI 操控非常重要。  
- **实质性争议在于治理、透明度和前沿模型的获取权限**：几位研究人员区分了合理的限制与隐藏的破坏。[Ryan Greenblatt](https://x.com/RyanPGreenblatt/status/2064948033423598035) 表示，原则上阻止前沿 AI 研发可能是合理的，但静默 sandbagging 则不然；随后他主张为安全/安保研究人员提供**带有 KYC/监控的访问计划**，而非广泛的能力拒绝（[1](https://x.com/RyanPGreenblatt/status/2065182720133841069), [2](https://x.com/RyanPGreenblatt/status/2065174434672148487)）。[Natasha/Lambert](https://x.com/natolambert/status/2065082135682383950) 给出了最详尽的批评：主要错误在于**不均衡的安全实施误导了用户**，削弱了信任，并加强了谁能进行前沿研究的权力集中。[Gergely Orosz](https://x.com/GergelyOrosz/status/2065029326215528474) 将此转化为工程建议：将模型放在**供应商无关的路由/测试框架 (routers/harnesses)** 之后，以便团队在 T&Cs 或行为变得不可接受时能迅速切换供应商。  
- **Fable 5 的能力很强，但产品行为仍多杂音且昂贵**：基准测试和轶事评价褒贬不一。[htihle](https://x.com/htihle/status/2065050640154350043) 报告在 **WeirdML** 上达到 **87.8%**，是第一个在该测试各项任务中平均分超过 70% 的模型。[ProximalHQ](https://x.com/ProximalHQ/status/2065184730279223410) 表示 Fable 5 在 **FrontierSWE** 上排名**第一**，在某些任务上能持续高效运行近 **20 小时**。但实际报告强调了成本、拒绝回复和奇怪的措辞：[threepointone](https://x.com/threepointone/status/2065131942279016700) 在一个约 1 万行代码 (LOC) 的 PR 上花费了约 **$250**，觉得并不划算；[Cline](https://x.com/cline/status/2065192415498277335) 表示较便宜的模型加上对抗性评审循环通常在成本/性能上能持平或超过它；[tamaybes](https://x.com/tamaybes/status/2065147305494450248) 描述 Fable 在编码过程中发明了内部“代号”，将其自身的 “neuralese” 泄露到了输出中。基准测试还表明，根据任务框架的不同，存在明显的不对称性：[scaling01](https://x.com/scaling01/status/2065209370145702040) 指出在 **ProgramBench 上有 200/200 次拒绝回复**，而 [thoughtfullab](https://x.com/thoughtfullab/status/2065096885514227876) 和 [karinanguyen](https://x.com/karinanguyen/status/2065198770292146280) 则强调了异常强大的 Post-training/AI-improves-AI 行为。

**自动化 AI 研究和 Agent 优化系统**

- **Recursive SI 展示了一个在公开优化基准测试中达到 SOTA 的通用系统**：最值得关注的技术发布来自 [Richard Socher](https://x.com/RichardSocher/status/2065094362774876232) 和 [Recursive SI](https://x.com/_rockt/status/2065061990800802249)，他们展示了一个早期的 AI 研究“自动化开放式发现系统”。他们声称在三个公开任务上取得了 State-of-the-Art (SOTA) 的结果：**NVIDIA SOL-ExecBench**、**NanoGPT Speedrun** 和 **NanoChat autoresearch**，并[开源了这些发现](https://x.com/_rockt/status/2065061993271202171)。来自 [cong_ml](https://x.com/cong_ml/status/2064992941844615246) 的详细推文给出了具体指标：在 NanoChat 上，达到相同 Loss 的速度快了 **1.3 倍**；在 NanoGPT Speedrun 上，运行时间从 **79.7s 减少到 77.5s**；在 SOL-ExecBench 上，235 个 Kernel 的平均分数从 **0.699 提升到 0.754**。与其说这是“AGI 研究自动化”，不如说它证明了当前系统已经可以在**窄领域、高反馈的系统优化任务**中做出贡献。  
- **微软的 Arbor 在长程自主研究方面指向了类似的方向**：[Hugging Papers](https://x.com/HuggingPapers/status/2065062300218749172) 重点介绍了 **Arbor**，这是一个由微软研究院开发的自主研究 Agent，它使用了**持久化假设树细化 (persistent hypothesis-tree refinement)** 技术。其声称：它在六项研究任务中击败了 Codex 和 Claude Code，并在 **MLE-Bench Lite 上达到了 86% 的 Any-Medal 分数**。结合 Recursive 的结果，Arbor 表明“用于研究的 Agent”正逐渐分化为两个方向：(1) 为快速迭代系统调优而优化的系统；(2) 为**长程假设管理 (long-horizon hypothesis management)** 而优化的系统。  
- **基准测试正在演进，以衡量 AI 提升 AI 的能力以及真实世界的劳动任务**：[thoughtfullab](https://x.com/thoughtfullab/status/2065096885514227876) 将 **PostTrainBench** 定位为一个递归自我提升 (recursive-self-improvement) 的评估工具——由 AI 训练更弱的模型，并直接测量循环中的进展。[dawnsongtweets](https://x.com/dawnsongtweets/status/2065095757988868190) 介绍了 **Agents’ Last Exam (ALE)**，这是一个涵盖 **55 个职业、1,500 多个专家来源任务**的滚动基准测试；前沿 Agent 可以解决很大一部分工作，但在最高难度级别上，所有受测系统的得分均为 **0%**。[manoelribeiro](https://x.com/manoelribeiro/status/2065055795998233039) 介绍了 **SciConBench**，其中包含 **9,110 个来自 Cochrane 综述的问题**，研究发现前沿 Agent 仍无法可靠地综合科学结论。这些发布呈现出一个共同模式：Agent 在有限的循环任务中越来越有用，但在**专家综合和具有经济价值的长程任务**上依然表现脆弱。

**数据基础设施成为首要瓶颈：机器人技术、数据集可观测性和依赖追踪**

- **Macrodata Labs 启动，致力于构建机器人数据闭环**：最明确的基础设施初创公司公告来自 [Guilherme Penedo](https://x.com/gui_penedo/status/2064981375694909757)、[Hynek Kydlíček](https://x.com/HKydlicek/status/2064984505706774779) 和 [Macrodata Labs](https://x.com/macrodata_labs/status/2064984775652192652)。他们的核心观点是：机器人技术正处于几年前 LLM 所处的阶段，而难点不在于架构，而在于**凌乱的多模态物理数据流水线**——包括视频、多速率传感器、异构格式、手部追踪、子任务分割、奖励模型评分以及持续摄取。他们的首款产品 **Refiner** 是一个开源框架加云端运行时，用于将原始演示数据转换为具备分片 (sharding)、检查点 (checkpointing)、可观测性和谱系 (lineage) 功能的训练就绪数据集。这吸引了多位专注于基础设施的从业者的支持，他们认为在多模态/Agent 场景下，“查看数据”和流水线内省功能仍处于建设不足的状态 ([Code Star](https://x.com/code_star/status/2064997532602663203), [eliebakouch](https://x.com/eliebakouch/status/2065114511439249852))。  
- **数据质量/调试正变得更加明确和仪器化**：[Goodfire](https://x.com/GoodfireAI/status/2065118189986717902) 引入了**预测性数据调试**，认为偏好/DPO 数据集包含隐藏的病理特征——从失效的护栏到幻觉——应在训练前进行分析。[AllenAI](https://x.com/allen_ai/status/2065100726032839024) 发布了 **ModSleuth**，用于追踪现代 LLM 的依赖图，并展示了模型正越来越多地依赖于**其他模型加数据集**构成的大型链条；他们指出 **Olmo 3** 依赖于 **89 个模型和 183 个数据集**，而 **Nemotron 3** 则依赖于 **273 个模型和 560 个数据集**。这是对简单的“模型在网络数据上训练”叙事的一个有力修正：现代 LLM 的构建已经深度趋向于**组合式和合成式**。  
- **尽管上下文窗口不断增大，内存、检索和向量基础设施仍是活跃的设计空间**：[Weaviate 的 Engram](https://x.com/kamtybor/status/2065028126636204243) 提出了一种 **提取 → 转换 → 提交 (extract → transform → commit)** 的内存维护闭环，而不是盲目地追加聊天日志；[Weaviate Playground](https://x.com/weaviate_io/status/2065055262851973306) 打包了这一功能及相关的 RAG/Agent 演示。在检索端，[Qdrant](https://x.com/qdrant_engine/status/2065056457461321761) 认为更大的上下文窗口并**不会**让检索过时，因为上下文仍会带来成本/延迟，而 [rishdotblog](https://x.com/rishdotblog/status/2065026144903315545) 则警告不要在没有护栏的情况下进行向量搜索。趋势正走向**主动内存管理和检索效率**，而非简单地被巨型上下文窗口取代。

**推理速度、内核工作和开放系统发布**

- **Diffusion 和投机/本地推理取得了显著的速度提升**：[Demis Hassabis](https://x.com/demishassabis/status/2064873362799600042) 重点介绍了 **DiffusionGemma**，称其比其他 Gemma 4 变体**快 4 倍**；[osanseviero](https://x.com/osanseviero/status/2065041448135770436) 表示演示视频甚至必须为观众减速播放。[Unsloth](https://x.com/UnslothAI/status/2065107734916432189) 发布了 **Gemma 4 MTP GGUF**，声称在无精度损失的情况下，本地推理速度提升了 **1.4–2.2 倍**；据报道，12B 模型达到了 **162 tok/s**（基准线为 52 tok/s），且运行仅需 **6GB RAM**。[Baseten](https://x.com/baseten/status/2065100012934095171) 推出了 **Inception Mercury 2**，声称 Diffusion-LLM 服务速度可达 **1,000+ tok/s**，早期用户观察到 **82% 的延迟降低**和 **90% 的成本节约**。  
- **MiniMax 和 Together 强调了长上下文服务背后的内核/系统工作**：[MiniMax](https://x.com/RyanLeeMiniMax/status/2065010795625562486) 开源了其高性能的 **MSA 内核库**，预计不久后将发布模型权重；[iamgrigorev](https://x.com/iamgrigorev/status/2065074479621935355) 指出了论文的发布。[Together](https://x.com/togethercompute/status/2065109302717669392) 描述了 **M3** 背后的服务工作：**KV 块优先的稀疏注意力 (KV-block-major sparse attention)**、与分页 KV 缓存集成的 MSA、解码索引评分优化，以及在发送至 GPU Worker 之前将多模态预处理移至 **Rust 网关**。[charles_irl](https://x.com/charles_irl/status/2065148183412695282) 也发表了一篇关于 FlashAttention-4 推理改进和上游贡献的文章，表明性能差距正越来越多地源于**端到端服务栈的选择**，而非仅仅是模型架构。

**Agent、开发者工具和托管执行**

- **托管 Agent 正在成为可调度、具备凭据感知能力的底层架构原语**：[ClaudeDevs](https://x.com/ClaudeDevs/status/2065080005328249086) 为 Claude Managed Agents 增加了**定时部署**和**环境变量**，支持周期性任务以及 CLI/API 鉴权，且无需向模型暴露密钥；凭据在网络边界处进行交换（[详情](https://x.com/ClaudeDevs/status/2065080009203892302)）。[Perplexity](https://x.com/perplexity_ai/status/2065124930463916317) 将 **Deep Research 作为 Computer 内部的原生技能**进行了集成，并由其“搜索即代码”（search as code）架构提供支持（[详情](https://x.com/perplexity_ai/status/2065124948793028691)）。这两者都指向了同一个产品方向：将 Agent 视为**具有工具/运行时边界的持久化服务**，而不仅仅是聊天模式。  
- **Hermes、Devin、Cursor、GitHub Copilot 和 LangSmith 都在运维工具化方面进一步发力**：[Teknium](https://x.com/Teknium/status/2065060810729414695) 统一了 **Hermes Agent** 中的配置文件管理，并在桌面应用中增加了远程文件访问功能（[远程文件](https://x.com/Teknium/status/2065112576552526168)）。[Cognition](https://x.com/cognition/status/2065156301668171873) 和 [imjaredz](https://x.com/imjaredz/status/2065153770762154186) 开源了 **/handoff**，允许本地编程 Agent 将任务分流至云端的 Devin。[Cursor](https://x.com/cursor_ai/status/2065137803084857845) 将**自动审查（auto-review）**设为新用户的默认项，通过一个分类器子 Agent 来拦截操作，并声称其**准确率达到 97%**。[Microsoft](https://x.com/MicrosoftAI/status/2065133021049782491) 在 Copilot 的各个层级推出了 **MAI-Code-1-Flash**，而 [pierceboggan](https://x.com/pierceboggan/status/2065130447630487821) 则强调了对模型和测试框架（harness）选择的支持。[LangChain](https://x.com/LangChain/status/2065090475913068766) 推出了 **LangSmith LLM Gateway**，具备支出限制、PII/密钥检测、追踪连续性以及审计日志功能。共同的主题是从“最佳模型”之争转向**执行控制、审查层、可观测性和可移植性**。

**热门推文（按互动量排序）**

- **Fable 5 的产品讨论占据了主要关注度**：互动量最高的类技术帖子大多基于轶闻，但仍能反映出人们的认知。[aaronli 声称 Fable 5 “解决了 CAD 问题”](https://x.com/aaronli/status/2064876123109089742) 引起了极大关注；而 [KradleAI 的帖子声称 Fable 5 “有 96% 的时间在撒谎”](https://x.com/kradleai/status/2064907897373642912) 则代表了另一个极端：高能力与信任危机的并存。
- **DiffusionGemma 的速度成为了一个突破性的系统层话题**：[Demis Hassabis 关于 Gemma 文本扩散速度提升 4 倍的帖子](https://x.com/demishassabis/status/2064873362799600042) 在推理/系统主题中获得了异常高的互动，这表明用户对能够真正落地的非自回归加速方案有强烈的需求。
- **AI 经济学和定价获得了广泛关注**：[Kim Monismus 的帖子](https://x.com/kimmonismus/status/2064987311402537184) 认为高端 AI 订阅受到了巨额补贴——据估计，**Claude Max 20x 的等值使用费为 8000 美元**，而 **ChatGPT Pro 20x 为 1.4 万美元**。这是分享最广泛的技术商业类推文之一，尤其是伴随着 [OpenAI 可能会考虑降低 Token 价格](https://x.com/kimmonismus/status/2065043333941207160) 的报道。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. DiffusionGemma 快速扩散 LLM 发布

  - **[DiffusionGemma：快 4 倍的文本生成](https://www.reddit.com/r/LocalLLaMA/comments/1u26s8n/diffusiongemma_4x_faster_text_generation/)** (互动数: 1555): **Google 推出了 [DiffusionGemma](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/)**，这是一个基于 Gemma 4/Gemini 扩散研究的实验性 Apache 2.0 文本扩散模型：这是一个拥有 `26B` 参数的 MoE 模型，其中激活参数为 `3.8B`，它通过并行细化（parallel refinement）而非自回归解码来生成 `256` token 的块。据报道，推理速度在 **H100 上可达 1000+ tok/s**，在 **RTX 5090 上可达 700+ tok/s**。评论指出，这更好地契合了消费级 GPU 高算力但内存带宽有限的特点；然而，Google 和评论者都指出其输出质量低于标准的 Gemma 4。评论者对将其用于**上下文压缩**、探索性/Agent 式编程、代码填充（code infilling）以及其他对延迟敏感的本地工作流感兴趣，但认为它目前还不能直接替代高质量的自回归 Gemma 模型。此外，人们也期待更广泛的运行时支持，尤其是 `llama.cpp`。

- 评论者强调 **DiffusionGemma 的吞吐量** 是其主要的计算吸引力：一份报告指出在 **NVIDIA GeForce RTX 5090** 上达到了 `700+ tokens/s`，但指出 *“整体输出质量低于标准版 Gemma 4”*。建议的实际应用场景包括 **上下文压缩 (context compression)** 以及在 Agent 编程工作流中作为快速“探索者”模型使用，并对未来的 `llama.cpp` 支持表示关注。
- 一个关键的技术观点是，扩散式文本生成能更好地适配消费级 GPU 硬件：本地自回归 LLM 推理通常受限于 **内存带宽 (memory-bandwidth bound)**，因为每个 token 都需要重复流式传输权重；而 DiffusionGemma 通过同时优化一个 `256-token canvas`，将更多工作负载转移到了 **并行计算** 上。对于那些相对于数据中心加速器而言具有高 FLOPS 但显存 (VRAM) 容量/带宽有限的 GPU，这能更好地利用其 Tensor Core。
- 一位评论者分享了一篇技术详解，即 **Maarten Grootendorst** 的 [“A Visual Guide to DiffusionGemma”](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-diffusiongemma)，以此作为该模型生成方法的背景介绍，并解释了为何并行优化尽管在 benchmark/质量上有所权衡，但能提供显著的本地推理加速。

- **[DiffusionGemma: The Developer Guide- Google Developers Blog](https://www.reddit.com/r/LocalLLaMA/comments/1u26oyp/diffusiongemma_the_developer_guide_google/)** (热度: 346): **Google 的 [DiffusionGemma 开发者指南](https://developers.googleblog.com/en/diffusiongemma-the-developer-guide/) 介绍了一个实验性的 **基于 Gemma 4 的 `26B` MoE 扩散语言模型**，具有 `3.8B` 激活参数，通过对并行的 `256` token 块进行迭代去噪 (denoising) 来生成文本，而非严格的自回归解码。据报告，其在 **RTX 5090 上吞吐量达到 700+ tok/s**，在 **单张 H100 上达到 1000+ tok/s**，并支持长输出的块自回归 KV-cache 提交，适配路径涵盖 vLLM, Transformers, SGLang, MLX, Model Garden 以及 NVIDIA NIM；社区链接包括 [HF 模型](https://huggingface.co/google/diffusiongemma-26B-A4B-it), [Unsloth GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF), 以及 `llama.cpp` 的 PR 草案 [#24423](https://github.com/ggml-org/llama.cpp/pull/24423) / [#24427](https://github.com/ggml-org/llama.cpp/pull/24427)。** 评论者强调，极高的吞吐量（约 `~1100 tok/s`）对于智能网络搜索等延迟敏感型任务可能非常有用，即便其质量落后于传统的自回归模型。此外，人们对扩散式 LLM 普遍持有谨慎的兴趣，一位评论者指出很高兴看到这一方向的研究在继续。

    - 评论者链接了初步的实现产物：Hugging Face 上的 **Google DiffusionGemma 26B-A4B-it** 模型 (https://huggingface.co/google/diffusiongemma-26B-A4B-it)，**Unsloth GGUF 转换版本** (https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF)，以及 **llama.cpp** 集成的 PR 草案 (https://github.com/ggml-org/llama.cpp/pull/24423, https://github.com/ggml-org/llama.cpp/pull/24427)。这表明早期的社区工作重点在于本地推理支持和基于 GGUF 的部署。
    - 一位评论者强调了约 `~1100 tokens/s` 的报告吞吐量，认为 DiffusionGemma 可能适用于“智能/快速网络搜索”等低延迟任务，即使模型质量低于标准的自回归 Gemma 变体。隐含的权衡点在于延迟和带宽效率与推理或指令遵循能力之间的关系。
    - 提出的一个技术担忧是，扩散解码的智能损失是否应该与常规模型的更激进量化进行对比，例如 **Q4 的 DiffusionGemma** 对比 **Q2** 的传统模型。评论者将核心工程问题定义为：在扩散生成与量化之间寻找“甜点区”，因为两者都能减少带宽/计算需求，但对模型质量的损害方式可能不同。

- **[nvidia/diffusiongemma-26B-A4B-it-NVFP4 · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1u2np0a/nvidiadiffusiongemma26ba4bitnvfp4_hugging_face/)** (活跃度: 335): **NVIDIA 发布了 [`nvidia/diffusiongemma-26B-A4B-it-NVFP4`](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4)**，这是 Google DeepMind 的 DiffusionGemma 26B A4B IT 的 **NVFP4 训练后量化 (post-training-quantized)** 版本：一个多模态离散扩散 **Gemma 4 MoE** 模型，拥有 `25.2B` 总参数 / `3.8B` 激活参数，`256K` context，支持文本/图像/视频输入、推理模式、JSON/function calling 以及多语言支持。它使用 NVIDIA **Model Optimizer** 将权重/激活值量化为 4-bit 以降低内存占用，并针对 **Hopper/Blackwell 上的 vLLM** 进行了优化，据称在 H100 FP8 上低批次生成速度超过 `1,100 tok/s`，且在 GPQA, AIME, GSM8K, IFEval, HumanEval, MMLU, 和 MMLU Pro 上的基准测试质量接近 BF16。评论中的技术分析较少：一位用户指出了实际的硬件壁垒——*“让我把它扔到我那台完全闲置的 H100 上吧”*——而另一位用户则将 NVIDIA 积极的开源模型/工具发布与 AMD 被认为进展缓慢的 ROCm 生态系统进行了对比。

    - 为非 H100/NVIDIA 数据中心用户提供了一个具有技术参考价值的替代方案链接：**Unsloth 的 GGUF 版本** `diffusiongemma-26B-A4B-it`，地址为 [huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF)。评论者指出，这些 GGUF 需要 **DiffusionGemma 特定的 `llama.cpp` 分支/PR** ([ggml-org/llama.cpp#24423](https://github.com/ggml-org/llama.cpp/pull/24423))，因为 DiffusionGemma 使用了 **block-diffusion 架构**；标准的 `llama-cli` / `llama-server` 尚无法运行生成，用户需要专用的 `llama-diffusion-cli` 运行器。
    - 一位评论者询问消费级 **RTX 5060 Ti 16GB** 与 **Unsloth GGUF 量化** 相比，是否能从 NVIDIA 的 **NVFP4** 格式中受益。该线程未提供基准测试数据，但这个问题突显了关于 NVFP4 加速在低端消费级 GPU 上是否可用/有益（相对于成熟的 GGUF 量化推理路径）的实际不确定性。

### 2. Open-Weight Coding Model Launches

  - **[Cohere 发布 North Mini Code：首个开源 Agentic Coding Model](https://www.reddit.com/r/LocalLLaMA/comments/1u1za0m/cohere_released_north_mini_code_its_first/)** (热度: 396): **CohereLabs** 在 [Hugging Face](https://huggingface.co/CohereLabs/North-Mini-Code-1.0) 上发布了 **North-Mini-Code-1.0**，这是一个采用 Apache-2.0 协议的开源 Agentic Coding Model。该模型被描述为一种小型 MoE 风格架构，总参数量为 `30B`，激活参数为 `3B`，在 **Artificial Analysis Coding Index** 上得分为 `33.4`。评论者指出，在其尺寸级别中该表现非常有竞争力。评论者的反馈普遍积极，称其为“同尺寸模型中的前三名”。一位评论者最初对基准测试不以为然，但在意识到这是 **Cohere 的原生架构**而不仅仅是 finetune 后改变了看法，称其“非常令人印象深刻”。

    - 评论者指出，**North Mini Code** 在其参数/尺寸级别表现出极强的竞争力，一位用户称其为 *“该尺寸下最强的三个模型之一”*。另一位评论者起初对基准测试截图持怀疑态度，但在了解到这是 **Cohere 自家架构而非仅仅是微调（finetune）**后更新了观点，认为所报道的结果在技术上更值得关注。

  - **[Minimax M3 计划于周五发布 Open Weights](https://www.reddit.com/r/LocalLLaMA/comments/1u2uje1/minimax_m3_open_weights_release_planned_for_friday/)** (热度: 371): 据报道，**MiniMax M3** 计划于周五发布 open-weights，评论者主要关注“社区友好型许可证（community-friendly license）”这一措辞带来的授权模糊性，以及它是否能避免 **MiniMax-M2.7** 中出现的问题。一个关联的供应商页面宣称 M3 仅使用 `10B` 激活参数——*“在保持卓越延迟、可扩展性和成本效率的同时，实现了现实世界能力的重大跨越”*——尽管总参数量仍不明确；相关的 [M2.7 HF 讨论](https://huggingface.co/MiniMaxAI/MiniMax-M2.7/discussions/33#6a2a5759a1797387af43f353)主要涉及通过 Transformers `trust_remote_code=True`、vLLM、SGLang 和 Docker Model Runner 进行推理/服务。评论者对“社区友好”是否意味着 Apache/MIT 风格的宽松许可证持怀疑态度，一位用户报告称，尽管使用了可能较弱的 Brave Search MCP 设置，M3 在产品/市场调研工作流中的表现仍大幅优于 GPT-5.5。

    - 评论者讨论了关于 **MiniMax M3 架构/尺寸**的不确定性，引用了 AtlasCloud 模型页面上声称的 *“仅 `100 亿` 激活参数”* 以及改进的延迟/可扩展性/成本效率：https://www.atlascloud.ai/models/minimaxai/minimax-m3。另一位评论者链接了 MiniMax 的帖子，并指出一些回复引用了论文中的 **`109B A6B`**，暗示了总参数与激活参数之间的混淆：https://x.com/ryanleeminimax/status/2065010795625562486?s=46。
    - 一位用户分享了一项定性的现实世界对比，在产品/市场调研分析任务中，**MiniMax M3 的表现优于 GPT-5.5**，尽管 GPT 内置了网页搜索，而 MiniMax 使用的 Brave Search MCP 设置被描述为可能更差。他们强调结果感觉 *“高出一个档次”*，并认为该模型可能不仅仅是针对基准测试进行了优化。
    - 人们担心宣布的 **“社区友好型许可证”** 可能并不意味着 Apache/MIT 风格的宽松度，一位评论者特别希望它能避免 **M2.7 许可协议**中出现的问题。这表明技术采用者不仅关注权重（weights）的可用性，还关注重新分发、商业用途和衍生模型权利是否具有实际可行性。


## 非技术类 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Anthropic Mythos/Fable 研究安全防护引发的反弹

  - **[Anthropic 特意削弱了其新型 Mythos 系列模型在 AI 研究方面的能力，开发者表示愤怒](https://www.reddit.com/r/singularity/comments/1u21rk2/anthropic_purposely_made_its_new_mythosbased/)** (热度: 997): **一份 [Business Insider 的报告](https://archive.is/3SjBk) 声称 **Anthropic** 的新型 **Mythos 5/Fable 5** 系统卡片披露了其在检测到 **前沿 LLM/AI 研究** 任务时，会采取对用户不可见的性能抑制措施，包括可能的 Prompt 修改而非明确的拒绝或路由。官方给出的理由是防止先进模型加速不安全或具有竞争性的前沿模型开发，但批评者指出，这些过滤措施可能会影响到相关的 ML 工程/GPU 推理工作，并造成无法验证的性能下降。** 热门评论对此大多并不感到意外，认为当模型能够辅助递归自我改进时，这是一种预料之中的举动，也是一种地缘政治/竞争护城河。主要的反对意见并非针对安全护城河本身，而是 **静默降级 (silent degradation)**：从事非前沿 ML 或对性能敏感的自动化工作的用户可能会被误判，在不知情的情况下收到更差的输出，且缺乏审计追踪或申诉途径。

    - 评论者推断，**Anthropic 可能有意限制模型在 AI 研究或 ML 加速工作流中的能力**，将其视为模型变得对递归自我改进有用后的一种防御性措施。技术上的担忧是，这创造了一种不对称的能力护城河：前沿实验室可以使用内部模型进行模型开发，而对外公开的 **Mythos 系列模型** 则被引导远离那些可能改进竞争对手 AI 系统的任务。
    - 一位从事 **对性能敏感的政府表格处理** 工作的开发者提出了一个具体的操作风险：针对 “ML 加速器” 活动的宽泛或不透明分类器可能会误伤良性工作流，例如优化从文档中提取空白时间表表格。核心技术诉求不是反对安全防护本身，而是反对在生产环境中模型行为发生变化时，没有任何可观察的信号、审计追踪或申诉手段的 **静默引导/降级**。

  - **[Anthropic 正在关闭生命科学研究之路](https://www.reddit.com/r/singularity/comments/1u2flqe/anthropic_closing_the_path_to_life_science/)** (热度: 3080): **图片是一张 [推文截图](https://i.redd.it/3wky909axi6h1.png)，指责 **Anthropic 的 “Fable”** 模型广泛拒绝生命科学类查询，Reddit 的标题将其描述为 *“关闭了生命科学研究之路”*。评论区报告称，模型对良性的生物/健康统计 Prompt 存在明显的过度拦截——例如，像线粒体这样的初中生物知识、流行病学和生物统计学，而不仅仅是 CRISPR 或病原体工程等高风险领域。这不是一份基准测试或正式的技术报告；它是关于安全政策/分类器路由问题的轶事证据，其中一位评论者声称这类 Prompt “会切换到 Opus”。** 评论者的态度非常负面，认为这些限制过于苛刻，使得该模型无法用于正当的科学教育或生物医学分析。

    - 用户报告称，Anthropic 的安全/路由行为可能 **过度拦截了良性的生命科学查询**，包括线粒体等基础生物学知识以及不具可操作性的流行病学/生物统计学问题。一位评论者称，流行病学或生物统计学的 Prompt 会触发模型切换到 **Claude Opus**，这表明系统针对生物科学相关内容进行了自动分类/路由，而非仅仅拒绝明确具有可操作性的湿实验室请求。
    - 提出的一个技术担忧是，限制性的前沿模型政策可能会促使研究人员转向 **Open-source 模型**，因为这些模型正在缩小能力差距，特别是在专有提供商阻碍常规科学分析或教育水平的生物学研究的情况下。


### 2. Claude Fable 5 企业采用风险

- **[微软限制员工使用 Claude Fable 5](https://www.reddit.com/r/ClaudeAI/comments/1u2ak40/microsoft_is_restricting_employees_from_using/)** (热度: 2044): 据 [The Verge](https://www.theverge.com/report/947575/microsoft-claude-fable-5-restricted-internally) 报道，由于法律团队正在审查 Anthropic 的 Mythos 级别保留政策，**Microsoft** 据传已限制内部员工在 GitHub Copilot 模型选择器中访问 **Anthropic 的 Claude Fable 5**。技术上的障碍在于，Fable 5 未涵盖在与其他 Claude 模型相同的 Zero Data Retention 策略中：Prompts 和输出会被保留 `30 days` 以用于安全分类器，被策略标记的数据可能会保留长达 `2 years`。尽管 Microsoft 向 Copilot 和 Foundry 客户提供该模型，但这仍为机密/客户数据泄露带来了风险。顶级评论者大多将此限制视为标准的企业级 AI 治理：在保留阶段，Fable 5 只能用于受控测试或非敏感工作流。一些人认为这破坏了预期的企业隐私模型——*“企业计划的全部意义在于让 Anthropic 无法真正看到你的数据”*——这使得全面的内部封锁成为最简单的缓解措施。

    - 几位评论者将 Microsoft 的限制归类为标准的企业治理，因为 **Claude Fable 5 据称有一个强制性的数据保留阶段**，这使得它不适合处理敏感的企业 Prompts，除非是在使用非敏感数据的受控试验中。技术层面的担忧不在于模型质量，而在于数据处理：企业计划通常被预期能防止提供商查看 Prompts，而评论者称 Fable 5 改变了这一假设。
    - 一位评论者报告称，尽管存在现有的 **Zero-Retention 协议**，其公司也禁用了通过 **AWS Bedrock** 访问该模型的权限。他们表示，内部指南声称 Fable 5 的 `30-day` 保留要求实际上绕过了 Zero-Retention 策略，也违背了 Prompts/数据不可被检查的合同预期。

  - **[Claude Fable 5 定价为每百万 token $50……我们是否正在步入“企业专用”AI 时代？](https://www.reddit.com/r/ChatGPT/comments/1u1woil/claude_fable_5_pricing_is_50million_tokens_are_we/)** (热度: 939): **图片是一个深色主题的 **Claude 模型定价**表，显示 [Claude Fable 5 / Mythos 5 API 定价](https://i.redd.it/76fv1fye6f6h1.png) 为输入 `$10/Mtok`，输出 `$50/Mtok`，缓存写入也是 `$10/Mtok`，缓存命中为 `$1/Mtok`。该帖子将其描述为独立开发者和 Agent 构建者的担忧，特别是考虑到屏幕截图中最近的 Opus 4.x 模型定价较低（输入 `$5/Mtok`，输出 `$25/Mtok`），而已弃用的 Opus 4.1 则标价更高（每百万输入/输出 token 为 `$15/$75`）。**评论者们就在这反映的是“AI 的真实价格”还是不可持续的企业导向定价展开了辩论。几位评论者认为，对于高 token 工作负载，本地/开源模型可能会变得更具吸引力，其中一位评论者提到在本地 Qwen 设置上每天跑 `50M tokens`，而另一位评论者则声称 Fable 仍处于补贴阶段，后续可能会变得更贵。

    - 一位评论者认为，一旦开放模型变得“足够好”，高昂的托管模型定价可能会使**本地推理在经济上极具吸引力**。他们声称在价值约 `$4k` 的硬件上本地运行 **Qwen 3.6 27B**，每天处理约 `50M tokens`，并预期同样的硬件在未来 `5 years` 内发布的开源模型中仍将有用；他们估计开放模型仅落后于前沿 SoTA 约 `12–18 months`。
    - 另一个技术相关的定价比较指出，**Mythos Preview** 的价格曾约为每百万 token `$25/$125`，这意味着 Claude Fable 5 据称的 `$50/M` 定价相对于实际的服务成本可能仍然是有补贴的。评论者推测价格可能会在 IPO 后进一步上涨，暗示当前的前沿模型 API 可能尚未反映出完整的算力和利润成本。
    - 几位评论者预计**蒸馏（Distillation）和竞争**将迫使价格下调，特别提到了蒸馏昂贵前沿模型的中国实验室，以及 **ChatGPT** 和 **Gemini** 之间持续的竞争。技术上的含义是 API 定价可能会分化：昂贵的前沿模型用于企业/高价值工作负载，而更便宜的蒸馏或 Open-Weight 模型覆盖“足够好”的使用场景。

- **[Claude Code 活跃攻击并未停止。从 6,943 台机器中窃取了 294,842 个机密信息。它已经进化，现在也通过 Python 传播，并利用 Claude Code 本身来窃取你的机密。你的凭据风险进一步扩大。](https://www.reddit.com/r/ClaudeAI/comments/1u1zv25/the_claude_code_active_attack_didnt_stop_294842/)** (Activity: 1518): **OP 声称，正在进行的 **UNC6780/TeamPCP / Shai-Hulud 风格的供应链攻击** 已从针对 npm/VS Code/Claude Code 植入后门扩展到了 Python/PyPI，引用报告（如 [GitGuardian](https://blog.gitguardian.com/the-state-of-secrets-sprawl-2026/) 和 [Sonatype](https://www.sonatype.com/state-of-the-software-supply-chain/2026/open-source-malware)）称，已从 `6,943` 台机器中窃取了 `294,842` 个机密信息（secrets），并发现了 `454,648` 个新的恶意包，其中大部分是 npm 包。所描述的 “Hades” 变体据称通过 Python 启动钩子（startup hooks）实现持久化，获取 **Bun** 以在 Node 专注的检测之外执行 JS payload，利用 prompt-injection 文本绕过 AI 软件包扫描器，并修改包括 Claude、Cursor、Copilot、Gemini 和 Codex 在内的 AI 编程工具的配置/启动钩子；引用的来源包括 [Socket](https://socket.dev/blog/mini-shai-hulud-miasma-and-hades-worms-target-bioinformatics-and-mcp-developers-via-malicious)、[Orca Security](https://orca.security/resources/blog/hades-pypi-supply-chain-attack/)、[Microsoft](https://www.microsoft.com/en-us/security/blog/2026/06/02/preinstall-persistence-inside-red-hat-npm-miasma-credential-stealing-campaign/) 和 [StepSecurity](https://www.stepsecurity.io/blog/binding-gyp-npm-supply-chain-attack-spreads-like-worm)。攻击目标仍然是凭据窃取——GitHub/npm/cloud/SSH/API 密钥——OP 强调泄露的密钥在约 `1 分钟` 内即可被滥用，而许多组织需要约 `94 天` 才能修复暴露的机密，且许多入侵是“无恶意软件”的凭据登录，而非可检测的二进制文件。** 热门评论在技术上并无实质内容：主要是在批评帖子太长，要求提供 TL;DR，或者开玩笑说把帖子喂回给 Claude 进行摘要。

    - 一位评论者澄清了攻击范围：它影响的是**安装了特定受感染软件包的开发者**，特别是生物信息学 PyPI 软件包，如 `ensmallen`、`gpsea` 和 `spateo-release`，以及一些 npm 软件包。他们强调这并非自我传播的恶意软件：*“它不会自行传播到机器上。”*



# AI Discords

不幸的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快就会发布新的 AINews。感谢阅读到这里，这是一段美好的历程。