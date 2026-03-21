---
companies:
- cursor
- openai
- anthropic
- langchain
- cognition
date: '2026-03-19T05:44:39.731046Z'
description: '**Cursor** 发布了 **Composer 2**，这是一款前沿级的编程模型，不仅大幅降低了成本，还在基准测试中取得了优异成绩（如
  **CursorBench** 得分为 **61.3**，**SWE-bench Multilingual** 为 **73.7**）。该模型通过**首次持续预训练（continued
  pretraining）**并结合强化学习进行了优化，由一个约 **40 人**的团队在全球 **3-4 个集群**上训练完成。


  **OpenAI** 收购了 Python 工具 **uv、ruff 和 ty** 的幕后团队 **Astral**，进一步加强了其开发者平台。**Anthropic**
  为 **Claude Code** 扩展了即时通讯应用频道，以实现持久化的开发者工作流。


  AI 智能体（Agents）的重心正从单一智能体转向受管集群（fleets）和运行时（runtimes）。**LangChain** 推出了用于企业级智能体管理的
  **LangSmith Fleet**，重点强调**智能体身份**、**凭证管理**和可审计性。其他发布项目还包括 **Cognition 的 Devin 团队版**、**lvwerra**
  开发的 **AgentUI**，以及关于具备**检查点（checkpointing）**和**回滚（rollback）**等功能的智能体运行时的讨论。安全与权限正成为智能体系统设计中的关键约束。'
id: MjAyNS0x
models:
- claude-code
- composer-2
people:
- kimmonismus
- mntruell
- theo
- ellev3n11
- amanrsanger
- charliermarsh
- gdb
- yuchenj_uw
- neilhtennek
- simonw
- yuvalinthedeep
- lvwerra
- hrishioa
title: 今天没发生什么特别的事。
topics:
- reinforcement-learning
- developer-tooling
- agent-systems
- agent-runtimes
- security
- credential-management
- multi-agent-systems
- model-training
- benchmarking
- software-engineering
- enterprise-ai
---

**平静的一天。**

> 2026年3月14日至3月16日的 AI 新闻。我们查看了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有发现更多 Discord 动态。[AINews 网站](https://news.smol.ai/) 允许你搜索过往所有期刊。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[选择开启/退订](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) 邮件接收频率！

---

# AI Twitter 回顾

**AI 编程 Agent、开发者工具，以及 IDE 之争**

- **Cursor 的 Composer 2 似乎是当天最重磅的开发者模型发布**：[@cursor_ai](https://x.com/cursor_ai/status/2034668943676244133) 发布了 **Composer 2**，将其定位为前沿级别的编程模型，并大幅降低了成本。Cursor 表示，质量的提升源于其**首次持续预训练 (continued pretraining)** 运行，为 RL 提供了更强的基座（[详情](https://x.com/cursor_ai/status/2034668950240329837)）。第三方反应强调了性价比和基准测试竞争力：[@kimmonismus](https://x.com/kimmonismus/status/2034667869816979645) 强调了 **$0.50/M 输入** 和 **$2.50/M 输出**，并报告其在 **CursorBench 上得分为 61.3**，**Terminal-Bench 2.0 上为 61.7**，**SWE-bench Multilingual 上为 73.7**；[@mntruell](https://x.com/mntruell/status/2034729462211002505) 将 Cursor 描绘为一种将 API 模型与**领域特定自研模型**相结合的新型公司。此次发布还包括在 [Glass](https://x.com/cursor_ai/status/2034719920710103452) 上的**早期 Alpha 版 UI**，[@theo](https://x.com/theo/status/2034780545134256205) 评论称行业可能会向这种更具 Agent 原生感的 UX 趋同。几位工程师还提到了训练和基础设施情况：[@ellev3n11](https://x.com/ellev3n11/status/2034778708163404102) 表示 RL 运行分布在**全球 3–4 个集群**上，[@amanrsanger](https://x.com/amanrsanger/status/2034704792925479356) 表示这个约 **40 人**的团队专注于软件工程任务。

- **OpenAI 通过 Astral 向底层布局；Anthropic 扩大 Claude Code 的覆盖面**：[@charliermarsh](https://x.com/charliermarsh/status/2034623222570783141) 宣布 **Astral**（**uv, ruff 和 ty** 背后的团队）正在加入 OpenAI 的 Codex 团队；[@gdb](https://x.com/gdb/status/2034662275391320472) 从 OpenAI 侧确认了这笔交易。这次收购被广泛解读为 OpenAI 通过掌握基础 Python 工具链来强化其开发者平台护城河；参见 [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2034661120599101498) 和 [Simon Willison 的评论](https://x.com/simonw/status/2034672725088997879)。与此同时，Anthropic 扩展了 **Claude Code** 的 **channels** 功能，让开发者可以通过消息应用进行交互，目前处于研究预览阶段（[公告](https://x.com/neilhtennek/status/2034762196576805123)，[文档](https://x.com/neilhtennek/status/2034762489951658190)）。这一产品方向值得注意：OpenAI 和 Anthropic 都在超越“模型 API”，向持久的开发者工作流和环境感知型 Agent 接入迈进。

**Agent、多 Agent 运行时以及企业级 Agent 控制平面**

- **重心正从单个 Agent 转向托管机群、运行时和 Agent 操作系统**：[@LangChain](https://x.com/LangChain/status/2034679590250258855) 推出了 **LangSmith Fleet**，这是一个企业级工作区，用于创建和管理具有记忆、工具、权限和频道集成的 **Agent 机群 (fleet of agents)**；发布会反复提到的主题包括 **Agent 身份**、**凭据管理**、共享控制、Slack 接入以及可审计性（[概览](https://x.com/LangChain/status/2034694530478612777)，[更多解读](https://x.com/Vtrivedy10/status/2034690067839521114)）。这与更广泛的论调一致，即“Agent”本身已不再是一个有用的抽象：[@YuvalinTheDeep](https://x.com/YuvalinTheDeep/status/2034624197528269085) 认为正确的比喻是 **AI 操作系统**，负责分配工作、资源和执行上下文。配套的发布活动强化了这种技术栈视角：[@cognition](https://x.com/cognition/status/2034679897084264659) 增加了 **Devin 团队**功能，Devin 可以分解工作并委派给独立 VM 中的并行 Devin；[@lvwerra](https://x.com/lvwerra/status/2034666400007016590) 发布了 **AgentUI**，一个协调代码、搜索和多模态专家的多 Agent 界面；[@hrishioa](https://x.com/hrishioa/status/2034666470932922745) 则认为长程 Agent 任务现在需要一个专门的运行时，具备**检查点 (checkpointing)、回滚、特定提供商的测试框架切换以及执行修复**功能。

- **安全与权限正成为 Agent 系统的“一等公民”级设计约束**：在近期的发布中，一个反复出现的主题是，生产环境的 Agent 部署瓶颈与其说是“模型能否做到？”，不如说是**权限、爆炸半径控制和可观测性**。[@swyx](https://x.com/swyx/status/2034667846505214295) 指出，**基于身份的授权 (identity-based authorization)** 正在成为 AI 安全的新兴共识；[@baseten](https://x.com/baseten/status/2034649896523874356) 则介绍了 **NemoClaw**，这是 NVIDIA 对 OpenClaw 式安全担忧的回应，具备**默认零权限**、沙箱化子 Agent 以及由基础设施强制执行的私有推理 (private inference)。LangChain 的 Fleet 发布也重点强调了权限控制和审计追踪。核心逻辑是：Agent 技术栈正在成熟，演变为更接近企业级软件基础设施的东西，而不再仅仅是 Chatbot 封装程序。

**模型发布、基准测试以及检索/推理结果**

- **MiniMax M2.7 被定位为实用型 Agent 模型，而非单纯的“前沿巨头”**：MiniMax 与 OpenClaw 合作，通过一场深入的技术直播预告了关于**自我演进 (self-evolution)** 和支持 **100,000 个运行集群**的基础设施 ([公告](https://x.com/MiniMax_AI/status/2034520321466978488))；同时，早期使用报告强调其提升了**情商 (emotional intelligence)**、**角色一致性**以及强大的 Agent 工作流能力 ([MiniMax note](https://x.com/MiniMax_AI/status/2034528945962696948))。来自 [ZhihuFrontier](https://x.com/ZhihuFrontier/status/2034543142234628318) 的更专业第三方评估表示，M2.7 的整体性能与前代基本持平，但在**指令遵循**、**上下文幻觉处理**以及**大代码/多轮对话**表现上有所升级，尽管在**硬核推理 (hard reasoning)** 方面略有退步且 Token 消耗更高。集成势头立竿见影：[@Teknium](https://x.com/Teknium/status/2034658808870621274) 将 M2.7 接入了 **Hermes Agent**，用户报告在某些工作流中，其长期运行的 Agent 表现优于 OpenClaw ([示例](https://x.com/populartourist/status/2034653545287348266))。

- **Qwen 3.5 Max Preview 和以检索为中心的系统在榜单上表现亮眼**：[@arena](https://x.com/arena/status/2034653740465336407) 报告 **Qwen 3.5 Max Preview** 在 **数学类排名第 3**，**Arena Expert 前 10**，以及**总榜前 15**，与之前的 Max 变体相比，在文本、写作和数学方面进步显著 ([详情](https://x.com/arena/status/2034658045113065603))；[@Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2034658901321560549) 确认更多优化即将到来。同时，最具技术看点的结果之一围绕着**延迟交互检索 (late interaction retrieval)**：[@antoine_chaffin](https://x.com/antoine_chaffin/status/2034649565614272925) 声称使用 **Reason-ModernColBERT** 后，**BrowseComp-Plus** 的解决率已接近 **90%**。这是一个仅有 **150M** 参数的模型，在深度研究型检索上的表现超越了体积达其 **54倍** 的系统。来自 [@lateinteraction](https://x.com/lateinteraction/status/2034651175023157550) 等人的多项后续研究认为，这并非偶然，而是另一个强力信号，表明在推理密集型搜索中，**多向量/延迟交互检索**正系统性地优于稠密单向量方法。

**多模态模型、OCR、文档解析与创意工具**

- **一批强力的文档/OCR 工具发布，涵盖了基于模型和非模型的方法**：[@nathanhabib1011](https://x.com/nathanhabib1011/status/2034565076963991910) 标注 **Chandra OCR 2** 为新的 **SOTA OCR** 发布，其在 **olmOCR bench 上达到 85.9%**，支持 **90 多种语言**，拥有 **4B** 参数，并支持手写体、数学公式、表单、表格和图像描述提取。另外，[@skalskip92](https://x.com/skalskip92/status/2034658568117309600) 强调了 **GLM-OCR 0.9B**，据报道这款小型 OCR 模型在基准测试中击败了 Gemini。在解析方面，LlamaIndex 开源了 **LiteParse**，这是一个本地的、布局感知的 PDF、Office 文档和图像解析器，**零 Python 依赖**，内置 OCR 选项，保留空间布局，并明确针对 **Agent 流水线 (pipelines)** 设计 ([发布](https://x.com/llama_index/status/2034661997644808638), [详细说明](https://x.com/jerryjliu0/status/2034665976428724267))。这在技术栈上形成了一个实用的划分：高端 OCR/VLMs 用于处理复杂页面，轻量级本地解析器用于处理常规场景。

- **图像/视频和世界模型工作不断加速，但有趣的部分在于延迟和可部署性**：Google 推出了显著升级的 **AI Studio** “vibe coding” 体验，配备了新的 **Antigravity** 编程 Agent 以及 **Firebase** 集成，支持多人应用、后端服务、Auth 和持久化构建（[Google AI Studio 帖子](https://x.com/GoogleAIStudio/status/2034655113961455651)，[Google 摘要](https://x.com/Google/status/2034658419202744614)）。在图像方面，Microsoft 发布了 **MAI-Image-2**，在 Image Arena 中首次亮相便位列 **第 5**，并在多个子类别中较 MAI-Image-1 有大幅提升，特别是在**文本渲染**和**肖像**方面（[竞技场排名](https://x.com/arena/status/2034660389284360585)，[Microsoft 公告](https://x.com/MicrosoftAI/status/2034661558492557386)）。在视觉/视频理解方面，[@skalskip92](https://x.com/skalskip92/status/2034606226902827228) 展示了 **MolmoPoint** 直接通过 VLM 进行基于点的多目标跟踪，这与 SAM 等分割优先的方法不同。[@kimmonismus](https://x.com/kimmonismus/status/2034659158843072893) 提出了一个有用的系统观点：在生成式媒体中，低于 **100ms** 的提示词到输出循环对于实际生产工作流来说，可能比原始模型质量更重要。

**训练、架构、推理与系统研究**

- **持续预训练和 RL 环境质量正重新成为核心竞争杠杆**：Composer 2 团队将性能提升明确归功于 **RL 之前的持续预训练**（[Cursor](https://x.com/cursor_ai/status/2034668950240329837)），多位研究人员认为这种模式在专业化模型中将变得更加普遍（[@code_star](https://x.com/code_star/status/2034672762263060562), [@cwolferesearch](https://x.com/cwolferesearch/status/2034713982515179672)）。与此相关，[@pratyushmaini](https://x.com/pratyushmaini/status/2034653569706811782) 提出了 **“微调者的谬误”（Finetuner’s Fallacy）**：早期的训练数据会在模型表示中留下持久的烙印，后期的微调很难将其消除。在系统方面，[@skypilot_org](https://x.com/skypilot_org/status/2034681533051855173) 在 K8s GPU 集群上扩展了 Karpathy 风格的 autoresearch，在 **8 小时内运行了约 910 个实验**，而不是按顺序运行约 96 个，这是基础设施直接改变自动化研究循环形态的一个例子。

- **架构探索在标准 Transformer 之外依然活跃**：[@MayankMish98](https://x.com/MayankMish98/status/2034681226217595333) 发布了 **M²RNN**，重新审视了用于可扩展语言建模的**具有矩阵值状态的非线性循环**；[@tri_dao](https://x.com/tri_dao/status/2034696258938708438) 指出，非线性 RNN 层似乎添加了与 Attention 和线性 SSMs 不同的独特特性。NVIDIA 的 **Nemotron 3** 技术栈也因混合了 **Transformer + Mamba 2**、**MoE/LatentMoE**、**多 Token 预测**以及 **NVFP4** 精度而受到关注，旨在降低推理成本并支持长上下文 Agent 工作负载（[摘要](https://x.com/TheTuringPost/status/2034668980892479993)）。在基础设施层，[@rachpradhan](https://x.com/rachpradhan/status/2034576637359161365) 报告 **TurboAPI** 达到了 **150k req/s**，声称经过一天的优化后吞吐量达到 **FastAPI 的 22 倍**，而 [@baseten](https://x.com/baseten/status/2034681788724019700) 推出了 **Baseten Delivery Network**，旨在将大模型的冷启动减少 **2–3 倍**。

**热门推文（按互动量排序）**

- **OpenAI 收购 Astral**：[@charliermarsh](https://x.com/charliermarsh/status/2034623222570783141) 宣布 Astral 加入 OpenAI 的 Codex 团队，这是 AI 实验室现在将掌握核心开发者工具视为战略重点的最明确信号之一。
- **Cursor Composer 2 发布**：[@cursor_ai](https://x.com/cursor_ai/status/2034668943676244133) 在该系列中拥有互动量最高的技术产品发布，反映了编程模型的性价比已变得多么核心。
- **Google AI Studio 升级的 vibe coding 技术栈**：[@GoogleAIStudio](https://x.com/GoogleAIStudio/status/2034655113961455651) 和 [@OfficialLoganK](https://x.com/OfficialLoganK/status/2034656376450908203) 围绕具有持久化构建、多人协作和后端集成的全栈应用生成，推动了大量的互动。
- **LlamaIndex LiteParse**：[@jerryjliu0](https://x.com/jerryjliu0/status/2034665976428724267) 引起了强烈共鸣，表明对用于 Agent 流水的实用、本地优先的解析基础设施有着持续的需求。
- **Late interaction retrieval on BrowseComp-Plus**：[@antoine_chaffin](https://x.com/antoine_chaffin/status/2034649565614272925) 发布了当天最重要的基准测试结果之一：一个 **150M** 的 late-interaction 检索器将一项高难度的深度研究基准测试推向了 **90%**。


---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要


### 1. 模型与基准测试发布

- **[MiniMax-M2.7 Announced!](https://www.reddit.com/r/LocalLLaMA/comments/1rwvn6h/minimaxm27_announced/)** (Activity: 1078): **该图片展示了新发布的 **MiniMax-M2.7** 模型与 M2.5、Gemini 31 Pro、Sonnet 4.6、Opus 4.6 和 GPT 5.4 等其他模型在 SWE Bench Pro、VIBE-Pro 和 MM-ClawBench 等各种 Benchmark 上的对比分析。**MiniMax-M2.7** 以红色标出，并在多个类别中展现了卓越的性能。该模型的开发强调自主迭代（autonomous iteration），通过分析、规划、修改和评估的迭代周期来优化其性能，在内部评估集上实现了 `30% performance improvement`。这一过程包括优化采样参数（sampling parameters）和增强工作流指南，标志着向全自动 AI 自我演进（AI self-evolution）的转变。** 一位评论者强调了现实世界可用性比 Benchmark 性能更重要，对那些在评估中表现出色但在实际应用中可能表现不佳的模型表示怀疑。另一条评论幽默地指出了新模型发布的速度之快，并对未来的发展表示兴奋和期待。

    - Recoil42 强调了 MiniMax-M2.7 模型的自主迭代能力，它可以通过迭代周期优化自身性能。该模型能够自主分析失败路径、规划变更、修改代码并评估结果，在内部评估集上实现了 30% 的性能提升。这一过程包括优化采样参数和增强工作流指南，预示着向全自动 AI 自我演进的迈进。
    - Specialist_Sun_7819 提出了一个关于 Benchmark 性能与现实世界可用性之间差异的关键点。他们强调，许多模型在评估中表现出色，但在处理偏离训练分布的任务时却很吃力。这一评论强调了用户测试对于验证 MiniMax-M2.7 等模型实际效果的重要性。
    - Lowkey_LokiSN 对模型的量化抗性（quantization resistance）表示担忧，并引用了之前 M2.5 模型的 UD-Q4_K_XL 变体出现的问题。量化会影响模型性能，而这方面的改进对于在资源受限的环境中部署时保持 MiniMax-M2.7 能力的完整性至关重要。

  - **[Omnicoder-Claude-4.6-Opus-Uncensored-GGUF](https://www.reddit.com/r/LocalLLaMA/comments/1rwy5sl/omnicoderclaude46opusuncensoredgguf/)** (Activity: 397): **该帖子介绍了 **OmniClaw** 模型，它是利用 DataClaw 数据集从真实的 **Claude Code / Codex** 会话中构建的，可在 [Hugging Face](https://huggingface.co/LuffyTheFox/OmniClaw-Claude-4.6-Opus-Uncensored-GGUF) 上获取。文中还展示了由 Claude Opus 蒸馏的 **Omnicoder** 模型，以及用于创意写作的 **OmniRP** 模型。所有模型均为 Uncensored（无审核），且由于其他量化版本存在质量问题，仅提供 `Q8_0` 量化版。这些模型是使用 [Pastebin](https://pastebin.com/xEP68vss) 上提供的 Python 脚本合并的，保留了 GGUF Header 和 Metadata 以确保兼容性。Omnicoder 模型是通过合并多个模型创建的，包括 Jackrong 和 HauhauCS 的 Qwen 3.5 9B 模型、Tesslate 的 Omnicoder，并以 Bartowski 的 Qwen 3.5-9B 作为 Base 模型。OmniClaw 和 OmniRP 模型则进一步分别与 empero-ai 和 nbeerbower 的模型进行了合并。该帖子声称，这些模型代表了基于 Qwen 3.5 9B 架构的小型 9B 模型中 **Uncensored General Intelligence (UGI)** 的最高水平。** 一条评论强调了对 Omnicoder 9B 模型进行的基准测试，指出其在 Aider Benchmark 中的 pass@1 为 `5.3%`，pass@2 成功率为 `29.3%`，每个问题的运行时间为 `402 seconds`，这表明人们对 Claude 蒸馏在提升 Omnicoder 性能方面的有效性持怀疑态度。

- grumd 提供了 Qwen3.5 35B-A3B 和 Omnicoder 9B 之间详细的 Benchmark 对比，使用了包含 225 个困难编程问题的 Aider Benchmark。Qwen3.5 35B-A3B 达到了 `26.7% pass@1` 和 `54.7% pass@2`，平均每个问题耗时 `95 seconds`。相比之下，Omnicoder 9B 在完成 75 个问题后，其 `5.3% pass@1` 和 `29.3% pass@2`，平均每个问题耗时长达 `402 seconds`。这突显了模型之间巨大的性能差距，特别是在效率和准确性方面。
- grumd 对通过 Claude Distillation（蒸馏）解决 Omnicoder 性能问题的潜力表示怀疑，认为当前的结果并不乐观。与 Qwen3.5 9B 的对比预计将进一步揭示性能问题是 Omnicoder 固有的，还是可以通过模型调整或 Distillation 技术来缓解。
- jack-in-the-sack 提出了一个关于模型可互换性的问题，特别是 Claude Code 是否可以被 Omnicoder 替代。这反映了社区对于在不同 AI 模型之间切换时，特别是在编程等专业任务中的兼容性和性能权衡的共同关注。

### 2. AI 模型的硬件和设置

  - **[我公司刚刚给了我一台配备 2x H200 (282GB VRAM) 的机器。帮我挑选一下“智能”天花板。](https://www.reddit.com/r/LocalLLaMA/comments/1rwwqbm/my_company_just_handed_me_a_2x_h200_282gb_vram/)** (热度: 854): **用户拥有了一台配备双 Nvidia H200 GPU 的服务器，每个 GPU 具有 `141GB HBM3e`，总计 `282GB VRAM`。他们的任务是测试用于本地编程任务的 LLM，包括代码补全、生成和评审。推荐的模型是 **Qwen 3.5 397B**，使用 `vLLM` 在 `Q4` Quantization 下进行高效的 Context 处理。建议避免使用 `ollama` 或 `llama.cpp` 等模型，因为它们对 Batched Inference 的处理较差，而这对于并发编程任务至关重要。相反，建议使用 `vLLM` 或 `sglang` 以在多用户环境中获得更好的稳定性和性能。** 一位评论者强调了在实验前定义明确目标和结果的重要性，以确保能继续使用该硬件。另一位分享了使用 `ollama` 的负面体验，理由是其不稳定且性能差，并推荐 `vLLM` 因其稳定性和对多用户环境的适用性。

    - Zyj 建议将 `vLLM` 与 `Qwen 3.5 397B` 模型结合使用，这应该允许在 `Q4` 精度下获得显著的 Context Window。该建议基于可用的 VRAM 以及平衡模型大小与 Context 能力的需求。
    - TUBlender 建议不要在需要 Batched Inference 的设置中使用 `ollama` 或 `llama.cpp`，因为它们对并发请求的处理较差。他们分享了使用 `ollama` 运行 `qwen2.5 72b` 的个人经验，结果导致了不稳定和崩溃，并推荐 `vLLM` 或 `sglang` 作为多用户环境下更稳定的替代方案。
    - Mikolai007 警告不要使用耗尽 GPU VRAM 的模型，并强调了维持健康的 Context Window 的重要性。他们推荐将 `Minimax M2.5` 和 `Qwen 3.5` 作为最佳选择，并指出 `GLM 5` 虽然能力很强，但参数量达到 `800b` 实在太大了。

### 3. Open-Source AI Tools and Applications

  - **[两周前，我在这里发帖询问大家是否对开源本地 AI 3D 模型生成器感兴趣](https://www.reddit.com/r/LocalLLaMA/comments/1rx8327/two_weeks_ago_i_posted_here_to_see_if_people/)** (热度: 366): **该帖子介绍了一个开源桌面应用的 Beta 版本，旨在通过图像生成 3D 网格 (meshes)，目前支持 Hunyuan3D 2 Mini 模型。该应用采用模块化设计，围绕扩展系统构建，开发者正在寻求有关功能、文件导出扩展以及额外模型支持的反馈。GitHub 仓库地址在[这里](https://github.com/lightningpixel/modly)。** 评论者建议了诸如多图像输入、基于文本的编辑、Checkpoint 保存以及对 `glTF` 等格式的支持。他们还建议支持 **Trellis 2** 以实现最先进的开源 3D 模型生成，并提议为非 CUDA GPU 开发 `ggml` 后端。此外，还讨论了自定义网格导入、纹理生成和基础编辑工具等额外功能。

    - New_Comfortable7240 概述了本地 AI 3D 模型生成器的一套完整功能，强调需要一个用户友好的界面，允许添加图像和文本来创建初始网格。他们建议实现一个用于迭代编辑的聊天界面、保存 Checkpoint，并通过修复功能确保与 glTF 格式的兼容性。评论还强调了在 glTF 中重命名节点以避免混淆的重要性，并提出了纹理生成、动画和 Level of Detail (LOD) 管理等可选功能。
    - Nota_ReAlperson 提到 Trellis 2 是目前免费开源 3D 模型生成的顶尖水平 (state-of-the-art)，并建议对其进行支持。他们还提出了为非 CUDA GPU 开发 `ggml` 后端的挑战性任务，这将为没有高端硬件的用户扩大可访问性。这突显了在开发模型生成器时考虑多样化硬件能力的重要性。
    - ArtifartX 强调了导入自定义网格并为其生成纹理的必要性，建议增强混合和基础笔刷工具。他们参考了过去一个使用 SDXL 和 ControlNet 结合自定义着色器进行投影的项目，表明了高级纹理处理功能的潜力。评论还建议将重点放在常用的文件格式上，如 OBJ、FBX、GLTF 和 USD 作为导出选项。

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI Model and Tool Releases

  - **[Harmonic 发布 Aristotle，全球首个免费的形式化数学家 Agent](https://www.reddit.com/r/singularity/comments/1rxdu0c/harmonic_unleashes_aristotle_the_worlds_first/)** (热度: 446): **图片宣布了由 Harmonic 开发的 "Aristotle Agent" 的发布，被誉为全球首个自主数学家 Agent，并提供免费使用。该 Agent 以其解决和形式化复杂数学问题的能力而闻名，通过提供证明的形式化验证 (formal verification)，确保在无需人工干预的情况下实现正确性，从而与其他 AI 数学工具区分开来。这与 DeepMind 的 AlphaProof 等其他 AI 系统形成对比，后者仍属于专有技术。该工具已与近期解决 Erdős 问题的尝试联系起来，突显了其在应对重大数学挑战方面的潜力。** 评论者强调了形式化验证功能的重要性，它确保了证明在构建时即是正确的，消除了人工验证的需要。人们对其处理教科书级别挑战之外的复杂开放问题的能力感到好奇。

    - **ikkiho** 强调了 Harmonic 的 Aristotle 中形式化验证的意义，并将其与其他 AI 数学工具进行了对比。与生成可能存在错误的自然语言证明的 LLM 不同，Aristotle 使用 Lean 证明来确保构建时的正确性，消除了人工验证的必要。这种方法特别值得注意，因为它免费提供，而不像 DeepMind 的专有软件 AlphaProof。
    - **ikkiho** 还对 Aristotle 当前的能力提出了疑问，想知道它是否已在具有挑战性的开放问题上进行了测试，还是主要在解决教科书级别的数学问题。这一询问指向了 Aristotle 未来应对更复杂数学挑战的潜力。
    - **omegahustle** 表示希望 Aristotle 保持免费并被负责任地使用，强调了对于那些能够有效利用它的人来说，其可用性的重要性。这一评论强调了免费获取先进数学工具对研究界的潜在影响。

- **[Gemini 应用的新版本刚刚发布。](https://www.reddit.com/r/GeminiAI/comments/1rx09kr/a_new_version_of_the_gemini_app_was_just_released/)** (活跃度: 425): **图片宣布了 Google Gemini 应用的一个更新，版本号为 `1.2026.1062300`，为美国的免费用户引入了 'Personal Intelligence' 功能。该功能旨在增强 Google 各个应用之间的连接性，提供个性化的回复。更新还包括 UI 改进和错误修复，下载大小为 `196.2 MB`。这表明 Google 生态系统内的用户体验和集成能力得到了显著增强。** 评论者对隐私表示担忧，特别是政府可能通过 'Personal Intelligence' 功能访问个人数据的可能性。还有人对 Gemini 应用的必要性持怀疑态度，一些用户认为它与现有的 Google 应用功能重叠。

    - Technical_Train_9821 提出了对 Gemini 应用数据隐私的担忧，强调了允许该应用访问和连接个人数据的潜在风险。他们认为，如果政府获得访问权限，可能会使个人的整个线上活动变得可搜索，从而引发严重的隐私问题。
    - brandeded 分享了 Gemini 应用的实际使用案例，强调了其与其他服务集成并执行复杂任务的能力。他们描述了该应用可以根据邮件内容创建日历预约、搜索特定的财务交易以及从 Google Drive 检索信息的场景，展示了其在高效管理个人数据方面的实用性。

  - **[基本官宣：Qwen Image 2.0 不开源](https://www.reddit.com/r/StableDiffusion/comments/1rwpyou/basically_official_qwen_image_20_not_opensourcing/)** (活跃度: 495): **Reddit 帖子中的图片是阿里巴巴发布的下一代图像生成模型 **Qwen-Image-2.0** 的发布公告。该模型最初在 Qwen 研究页面上被标记为 "Open-Source"，现在已被重新分类为 "Release"，表明它将不会开源。这一变化与阿里巴巴最近的内部变动相符，包括核心工程师的离职以及由于收入考量而从开源模型转向的战略调整。该模型具有专业的排版渲染功能，支持 `1k-token` 指令，以及原生 `2K` 分辨率，旨在创建详细的信息图表和漫画。** 评论者对阿里巴巴不开源 Qwen-Image-2.0 的决定表示困惑和失望，认为在与 Midjourney 等模型的竞争环境下，闭源会降低其价值。此外，据指出阿里巴巴 CEO 对开源模型缺乏收入表示不满，从而影响了这一战略转型。

    - **Skystunt** 强调了 Qwen Image 2.0 闭源方案的一个关键问题，即与 Midjourney 或 Nano Banana 等提供更成熟 UI 和开源优势的模型相比，其竞争优势被削弱了。该模型的闭源性质加上数据隐私担忧，使其尽管作为 7B 参数模型具有技术实力，但吸引力有所下降。
    - **_BreakingGood_** 提供了阿里巴巴战略转向闭源的背景，引用了 CEO 对开源模型缺乏收入的不满。这导致了重大的内部变动，包括核心工程师的离职，暗示阿里巴巴未来可能不会发布开源模型，从而影响社区对前沿技术的获取。
    - **LeKhang98** 对模型发布频率的看法发表了评论，指出虽然有些人对新模型感到应接不暇，但实际的发布率相对较低，每年只有 2-3 个重要的模型。这一观点建议社区应该珍惜目前新模型的步伐和可用性，尽管发布速度可能会放缓。

### 2. AI 在创意与技术领域的应用

- **[一位澳大利亚 ML 研究员使用 ChatGPT+AlphaFold 缩小了其爱犬 75% 的致命性 MCT 癌性肿瘤，仅用两个月便开发出个性化 mRNA 疫苗——此前他花费 2,000 美元对其爱犬的 DNA 进行了测序](https://www.reddit.com/r/singularity/comments/1ry961j/an_australian_ml_researcher_used_chatgptalphafold/)** (Activity: 498): **澳大利亚机器学习研究员 **Paul Conyngham** 利用 **ChatGPT** 和 **AlphaFold** 为他的爱犬 Rosie 开发了一种个性化 mRNA 疫苗，Rosie 患有危及生命的肥大细胞瘤（MCT）。通过花费约 `$2,000` 进行肿瘤 DNA 测序，Conyngham 利用 ChatGPT 识别新抗原（neoantigens），并使用 AlphaFold 预测蛋白质结构。他与来自新南威尔士大学（UNSW）的 **Martin Smith** 合作进行基因组测序，并与 **Pall Thordarson** 合作进行 mRNA 合成，尽管没有生物学或医学的正式背景，他仍成功在两个月内使肿瘤缩小了 `75%`。这一案例凸显了 AI 在个性化医疗和疫苗快速开发中的潜力 ([来源](https://www.the-scientist.com/chatgpt-and-alphafold-help-design-personalized-vaccine-for-dog-with-cancer-74227))。** 评论者们正在辩论此案例的影响，质疑这究竟代表了医疗民主化的重大转变，还是仅仅是被过度炒作。一些人认为，监管障碍正在阻碍医学进步，正如本案例中实现的快速开发所证明的那样。

    - **DepartmentDapper9823** 认为此案例说明了监管机构可能会阻碍医学进步。他们指出，当绕过这些障碍时，进步会发生得更快，利用 ChatGPT 和 AlphaFold 为狗快速开发个性化 mRNA 疫苗就证明了这一点。
    - **AngleAccomplished865** 呼吁专家评估此案例更广泛的影响，质疑这代表了民主化医疗的重大转变，还是仅仅是炒作。他们强调需要专业见解来确定在医学研究中使用 ChatGPT 和 AlphaFold 等 AI 工具的真实影响。
    - **682463435465** 提出了一个担忧，即癌症患者可能会尝试在自己身上复制这种方法，这表明在没有适当医疗指导的情况下存在自我实验的潜在风险。这强调了在使用 AI 进行个性化医疗时，需要仔细考虑伦理和安全影响。

  - **[构建了一个开源工具，可以找到任何图片的精确坐标](https://www.reddit.com/r/singularity/comments/1rx0abd/built_an_open_source_tool_that_can_find_precise/)** (Activity: 837): ****Netryx** 是由一名大学生开发的开源工具，旨在利用视觉线索和自定义机器学习流水线，从街景照片中确定精确的地理坐标。该工具可在 [GitHub](https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git) 上获取，旨在联系对地理定位技术感兴趣的开发者和公司。该工具的能力通过一个自定义 Web 版本得到展示，该版本可以对卡塔尔袭击等事件进行地理定位，尽管其核心流水线在不同版本中保持一致。** 评论者对该工具的潜在用途表达了复杂的情绪，指出它可能既有益又有害。人们对其依赖 Google Street View 等现有数据源来实现功能也感到好奇。


  - **[我构建了一个 Claude skill，可以为任何 AI 工具编写准确的 prompt。为了停止在糟糕的 prompt 上浪费额度。我们在 GitHub 上刚刚突破 600 星‼️](https://www.reddit.com/r/ClaudeAI/comments/1rxyarx/i_built_a_claude_skill_that_writes_accurate/)** (Activity: 728): **`prompt-master` 是一个 **Claude skill**，旨在优化各种 AI 工具的 prompt 生成，在 GitHub 上已获得超过 `600 颗星`。它能智能检测目标 AI 工具并应用特定策略，例如从用户输入中提取 `9 个维度`并识别 `35 个常见 prompt 问题`，以提高 prompt 的准确性和效率。该工具支持广泛的平台，包括 **Claude, ChatGPT, Midjourney 和 Eleven Labs**，并且是开源的，允许社区驱动的改进。最新版本 `v1.4` 吸收了用户反馈，`v1.5` 的计划正在进行中，重点是基于 Agent 的增强。[GitHub 仓库](http://github.com/nidhinjs/prompt-master)。** 评论者强调，该工具能够针对特定的 AI 模型（如 **Midjourney** 和 **Claude Code**）定制 prompt，这是其区别于通用 prompt 工具的关键所在。人们对其与开源模型的兼容性很感兴趣，这表明其具有更广泛的应用潜力。

- 该工具执行特定于工具的路由（routing）能力被强调为核心功能，使其比通用的 Prompt 增强器更有效。这至关重要，因为像 Midjourney 和 Claude Code 这样的不同 AI 工具需要独特的 Prompt 结构，而大多数通用工具无法解决这个问题。
- 一位用户询问了该工具与开源模型的兼容性，特别提到了在 `5090` GPU 上通过 ComfyUI 本地运行。这表明用户有兴趣在专有模型之外利用该工具的功能，从而可能在不同的 AI 环境中扩展其效用。
- 另一位用户指出，虽然之前已有类似工具的尝试，但它们通常需要手动调整 Prompt。然而，如果该工具能有效管理特定于工具的细微差别（例如 Cursor 和 Claude Code 之间的差异），它将显著提高可用性和效率。

- **[我厌倦了为我的 AI 音乐视频手动给每个片段编写 Prompt，所以我构建了一个 100% 本地开源的（LTX Video desktop + Gradio）应用来自动化这一过程，它就是 - Synesthesia](https://www.reddit.com/r/StableDiffusion/comments/1rx1w7d/i_got_tired_of_manually_prompting_every_single/)** (热度: 306): ****Synesthesia** 是一款开源应用程序，旨在通过集成 `Qwen3.5-9b` 等本地 LLMs 来自动化 AI 生成的音乐视频的创建过程。它处理三个输入文件：独立的干声、全乐队演奏和文本歌词，以生成一个在人声和故事片段之间交替的分镜表（shot list）。该应用与 **LTX-Desktop** 连接进行视频生成，在 `5090` GPU 上以 `540p` 分辨率在不到一小时内完成 3 分钟视频的首轮渲染。用户可以手动调整分镜表或让其自动运行，并为每个镜头选择多个备选素材以进行最终编辑。该项目托管在 [GitHub](https://github.com/RowanUnderwood/Synesthesia-AI-Video-Director) 上。** 一位评论者建议增加 **LoRA support** 以保持角色表现的一致性，而另一位则批评了自动化，认为它无法取代手动编写 Prompt 的创作过程。

    - Loose_Object_8311 建议该应用可以从 **LoRA support** 中获益，以在不同片段中保持一致的角色外观。LoRA (Low-Rank Adaptation) 是一种用于高效微调模型的技术，可以增强应用在 AI 生成的音乐视频中生成一致视觉元素的能力。
    - InternationalBid831 询问了与 **Wan2GP running LTX2**（而非 LTX Desktop）的兼容性，特别是对于使用 `5070ti` GPU 的用户。这表明该应用需要支持不同的硬件配置，以及可能不同版本的 LTX 软件，以适应更广泛的用户群体。
    - Diadra_Underwood 提议在应用中增加一个**样式下拉菜单（styles drop-down menu）**，强调了用户能够轻松切换不同视觉风格（如黏土动画、木偶或 CGI）的潜力。这一功能可以通过允许对 AI 生成内容中的各种艺术风格进行快速实验来提升用户体验。

### 3. AI 与法律/伦理挑战

  - **[多家词典编纂商起诉 OpenAI 存在“大规模”版权侵权，称 ChatGPT 正在剥夺出版商的收入](https://www.reddit.com/r/OpenAI/comments/1rx6o2i/the_dictionaries_are_suing_openai_for_massive/)** (热度: 718): **Britannica** 和 **Merriam-Webster** 已在纽约南区联邦法院对 **OpenAI** 提起诉讼，指控 **OpenAI** 的 **ChatGPT** 在未经许可的情况下使用其研究内容，侵犯了其版权。诉讼称，**ChatGPT** 直接从吸收的内容中提供答案的能力正在剥夺出版商的网络流量和广告收入，而这些收入对他们的生存至关重要。此案加剧了目前关于 AI 使用在线内容以及公共知识与专有信息边界的法律争论。[阅读更多](https://fortune.com/2026/03/18/dictionaries-suing-openai-chatgpt-copyright-infringement/)。** 评论者们质疑允许公司拥有定义权的潜在影响，以及对信息获取便利性的更广泛影响。针对词汇使用的货币化（monetization）出现了一些讽刺的论调，反映了人们对该诉讼前提的怀疑。


  - **[CEO 就如何废除 2.5 亿美元合同咨询 ChatGPT，无视律师建议，在法庭上惨败](https://www.reddit.com/r/ChatGPT/comments/1rxtt72/ceo_asks_chatgpt_how_to_void_250_million_contract/)** (热度: 465): **在最近的一场法律溃败中，**Krafton CEO Changhan Kim** 试图通过咨询 **ChatGPT** 而非其法律团队，来废除与 **Unknown Worlds Entertainment** 签署的一份价值 `$250 million` 的合同。法院果断做出了对他不利的裁决，强调了在没有专业监督的情况下将 AI 用于复杂法律策略的危险性。此案表明，虽然 AI 可以通过压力测试论点和总结判例来辅助法律准备，但它缺乏直接法律行动所必需的法律责任承担和背景理解。欲了解更多详情，请参阅 [404 Media 报告](https://www.404media.co/ceo-ignores-lawyers-asks-chatgpt-how-to-void-250-million-contract-loses-terribly-in-court/)。** 评论者强调了误用 AI 替代专业判断的问题，指出 AI 应该用于增强法律策略而非取代律师。他们强调了人类监督的重要性，特别是在复杂的法律事务中，并建议将 AI 用于识别潜在挑战，而不是作为法律建议的直接来源。

    - **RobinWood_AI** 强调了法律背景下对 AI 的误用，指出 AI 应被用于加强法律策略而非取代专业判断。AI 可以协助对论点进行压力测试和起草框架，但缺乏人类律师的法律责任和背景理解。这位 CEO 的错误在于在没有法律监督的情况下直接使用 AI 来废除合同，这说明了 AI 作为工具与法律责任主体之间的差距。
    - **chiqu3n** 讨论了 AI 在理解特定法律背景方面的局限性，指出像 **ChatGPT** 这样的通用 AI 模型可能无法考虑到可能影响合同条款的特殊立法。他们将其与专门的法律 LLM —— **'justicio'** 进行了对比，后者提供了更细致且法律准确的回复，凸显了在关键法律事务中由人类专家审查的重要性。
    - **Dailan_Grace** 指出了 AI 的“权威语气”问题，这可能会误导用户信任错误信息。AI 模型通常自信地呈现信息而没有任何保留，如果用户缺乏识别错误的专业知识，这可能会产生问题。这种对 AI 输出的过度自信可能导致了该 CEO 的错误决策。

  - **[Jeremy O. Harris 在名利场奥斯卡派对上醉酒大骂 OpenAI 的 Sam Altman 是纳粹](https://www.reddit.com/r/ChatGPT/comments/1rx9rqh/jeremy_o_harris_drunkenly_called_openais_sam/)** (热度: 650): **在名利场（Vanity Fair）奥斯卡派对上，剧作家 **Jeremy O. Harris** 与 **OpenAI** 的 CEO **Sam Altman** 发生对峙，指责他类似于纳粹人物，原因是 **OpenAI** 与战争部（Department of War）达成了新协议。Harris 随后澄清了他的言论，将 Altman 比作被判犯有战争罪的德国工业家 **Friedrich Flick**，而非 **Joseph Goebbels**。这一事件凸显了围绕 AI 及其军事应用的持续伦理辩论。** 评论反映了对纳粹类比是否恰当的怀疑，提到了 Altman 的犹太背景，并包含了一些题外话式的幽默。






# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们将不再以这种形式恢复它，但我们很快就会发布全新的 AINews。感谢阅读到这里，这是一段美好的历程。