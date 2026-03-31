---
companies:
- anthropic
- openai
- nous-research
- huggingface
date: '2026-03-30T05:44:39.731046Z'
description: '**Anthropic** 在面向 Pro/Max 用户的研究预览版中，为 **Claude Code** 引入了“**计算机使用**”（computer
  use）功能用于闭环验证，旨在增强可靠的应用迭代。


  **OpenAI** 发布了适用于 Claude Code 的 **Codex 插件**，实现了跨智能体的组合协作，并标志着向可组合编码工具链（composable
  coding harnesses）的转变。OpenAI 还指出，深夜执行的 Codex 任务运行时间更长，这为后台智能体委派（agent delegation）提供了支持。


  **Nous Research** 的 **Hermes Agent** 凭借更佳的压缩性、适应性和多智能体配置（multi-agent profiles）而得到快速采用，并正朝着智能体操作系统（agent
  OS）抽象化的方向演进。围绕 Hermes 建立的生态系统包含了追踪分析（trace analytics）、微调和远程控制等工具，同时也引发了关于开源与专有智能体基础设施的辩论。


  核心主题包括工具链、提示词/运行时编排以及评审循环（review loops），这些被视为超越模型能力本身的关键因素。'
id: MjAyNS0x
models:
- claude-code
- codex
- hermes-agent
people:
- omarsar0
- dkundel
- reach_vb
- theo
- jayfarei
- kaiostephens
- icarushermes
- winglian
- clementdelangue
- fchollet
title: 今天没发生什么事。
topics:
- closed-loop-verification
- cross-agent-composition
- agent-ecosystem
- multi-agent-systems
- runtime-orchestration
- tooling
- fine-tuning
- remote-monitoring
- privacy
- sandboxing
---

**平静的一天。**

> 2026年3月28日至3月30日的 AI 新闻。我们检查了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，且没有查看更多的 Discord。[AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[选择订阅或退订](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)不同频率的邮件！

---

# AI Twitter 摘要

**Claude Code Computer Use、Codex 互操作性以及 Coding-Agent Harness 竞赛**

- **Claude Code 获得 Computer Use 功能**：Anthropic 在 **Claude Code 中添加了 Computer Use** 功能，允许 Agent 打开应用程序、点击 UI，并直接在 CLI 中测试其构建的内容，目前面向 Pro/Max 用户开放 **research preview**。其实际意义在于闭环验证：代码 → 运行 → 检查 UI → 修复 → 重新测试。几位工程师称这是可靠应用迭代中缺失的一环，特别是与开放式桌面 Agent 相比 ([Anthropic 公告](https://x.com/claudeai/status/2038663014098899416)，[@Yuchenj_UW 关于“眼睛”解锁的看法](https://x.com/Yuchenj_UW/status/2038671697923223999)，[@omarsar0](https://x.com/omarsar0/status/2038668801256968381))。
- **跨 Agent 组合正成为标准**：OpenAI 发布了适用于 **Claude Code 的 Codex 插件**，可以从 Anthropic 的工具链内部触发 review、对抗性 review 以及“救援”流程，使用的是 ChatGPT 订阅而非自定义胶水代码。这一点的显著之处不在于插件的新颖性，而在于它释放了一个信号：编程技术栈正成为**可组合的 Harness**，而非单一的整体产品 ([@dkundel 开发的插件](https://x.com/dkundel/status/2038670330257109461)，[@reach_vb 的使用推文](https://x.com/reach_vb/status/2038671858862583967)，[开源说明](https://x.com/reach_vb/status/2038702889070211557))。另外，OpenAI 分享到，**深夜的 Codex 任务运行时间更长**，晚上 11 点左右开始的任务运行 3 小时以上的概率高出 60%，这符合将重构和规划委托给后台 Agent 的新兴模式 ([OpenAI Devs](https://x.com/OpenAIDevs/status/2038707501492056401))。
- **Harness 质量现在显然已成为一阶变量**：Theo 认为 **Opus 在 Cursor 中的得分比在 Claude Code 中高出约 20%**，并更广泛地指出，闭源 Harness 使得社区难以诊断或修复回归问题 ([性能差距主张](https://x.com/theo/status/2038690786821505378)，[闭源批评](https://x.com/theo/status/2038740065300676777))。这一主题在信息流中反复出现：模型能力的差距正在缩小，而**工具链、Prompt/运行时编排以及 review 循环**仍然造成了巨大的实际差异。

**Hermes Agent 的迅速崛起、多 Agent Profile 以及开放 Harness 生态系统**

- **Hermes 已成为本周突破性的开源 Agent 技术栈**：Nous 发布了一个重大的 **Hermes Agent** 更新，引发了从 OpenClaw/类 OpenClaw 配置迁移的浪潮，用户强调其具有**更好的压缩性、更少的冗余、更强的适应性以及更快的发布节奏** ([Nous 发布声明](https://x.com/NousResearch/status/2038688578201346513)，[Teknium 的多 Agent 配置文件](https://x.com/Teknium/status/2038694680549077059)，[社区迁移示例](https://x.com/soundslikecanoe/status/2038611090704113931)，[另一个示例](https://x.com/valenxi_r/status/2038692504120504453))。新的**多 Agent 配置文件（multi-agent profiles）**赋予每个机器人独立的内存、技能、历史记录和网关连接，使 Hermes 从“个人助理”转向可复用的 **Agent OS** 抽象。
- **围绕 Traces、远程控制和自我优化的生态系统正在形成**：多个项目正在将 Hermes 扩展到核心推理之外。[@jayfarei 的 opentraces.ai](https://x.com/jayfarei/status/2038385591818023278) 提供了一个 CLI/模式/评审流程，用于清理 Agent Traces 并将其发布到 Hugging Face，以供分析、评估、SFT 和 RL 使用。[@kaiostephens 向 HF 上传了约 4,000 条 GLM-5 Hermes Traces](https://x.com/kaiostephens/status/2038414350986207421)。[@IcarusHermes 描述了一种集成方案](https://x.com/IcarusHermes/status/2038524251355934872)，Agent 可以记录自己的决策、导出数据，并根据历史记录微调（fine-tune）更小的后继模型，从而切换到更廉价的模型。[@winglian 的 ARC](https://x.com/winglian/status/2038680417125957865) 增加了**基于浏览器的远程监控/控制**，并配备了端到端加密。
- **开源与专有 Agent 基础设施之争正进入白热化**：[@ClementDelangue 明确指出](https://x.com/ClementDelangue/status/2038552830638755962)，出于隐私和持久性的考虑，开源 Agent 工具应默认使用**开源模型**。与此同时，供应商正在针对已知痛点发力：[@fchollet 强调 PokeeClaw](https://x.com/fchollet/status/2038662563228230127) 是一款更安全的 OpenClaw 式助手，具备沙箱、审批、RBAC 和审计追踪功能；[Z AI 推出了 AutoClaw](https://x.com/Zai_org/status/2038632251551023250)，这是一个**无需 API key** 的本地 OpenClaw 运行时，并可选配 GLM-5-Turbo。

**Qwen3.5-Omni、GLM-5-Turbo/AutoClaw 及其向本地化/Agent 专用化的推动**

- **Qwen3.5-Omni 是一次重大的多模态发布**：阿里巴巴推出了 **Qwen3.5-Omni**，具备原生的文本/图像/音频/视频理解能力、**剧本级字幕生成**、内置的**网页搜索和 Function Calling**，以及一个出色的“**音视频 Vibe Coding**”演示，模型可以根据口述的视觉指令构建网站/游戏。据报道，其能力包括支持 **10 小时音频 / 400 秒 720p 视频**、**113 种语音识别语言**和 **36 种口语**；阿里巴巴声称其在**音频方面超越了 Gemini 3.1 Pro**，并在某些场景下达到了与其相当的音视频理解水平 ([发布贴](https://x.com/Alibaba_Qwen/status/2038636335272194241)，[演示贴](https://x.com/Alibaba_Qwen/status/2038637124619231467)，[补充演示](https://x.com/Alibaba_Qwen/status/2038641496455557565))。[@kimmonismus 提出了一个有用的提醒](https://x.com/kimmonismus/status/2038638427604762666)：这里的“Omni”是指**理解**多模态输入，而非任意的多模态生成。
- **Z AI 继续针对 Agent 工作负载进行优化**：[Artificial Analysis 评估了 GLM-5-Turbo](https://x.com/ArtificialAnlys/status/2038667075489808804)，这是 Z AI 专有的 Agent 优化变体。它在 AA 智能指数上得分 **47**，略低于开放权重的 **GLM-5 (Reasoning)** 的 **50**，但在 **GDPval-AA 上得分 1503**，高于 GLM-5 的 **1408**，这印证了该模型是针对真实世界的 Agent 工作流而非广泛的基准测试至上主义进行调优的。
- **专用开源模型日益成为主流部署模式**：多条推文得出了相同的论点：公司将越来越多地在私有数据上**拥有并专门化开源模型**，而不是无限期地租用通用 API ([@oneill_c](https://x.com/oneill_c/status/2038689976012149131), [@ClementDelangue](https://x.com/ClementDelangue/status/2038649731404927202))。支持证据包括：从 **Claude 4.6 Opus 蒸馏出的 Qwen3.5-27B 模型**在 HF 上连续数周走红，据报道其 **4-bit 量化版仅需 16GB 显存** ([Unsloth](https://x.com/UnslothAI/status/2038625148354679270), [@Hesamation](https://x.com/Hesamation/status/2038642306434150427))，以及对 llama.cpp 和 MLX 等本地运行时的热情不断高涨。

**本地推理与系统：llama.cpp 达到 100k Stars，MacBook 上的 Flash-MoE，以及 Web/服务工具链**

- **Local AI 迎来了一个象征性的里程碑，llama.cpp 在 GitHub 上的 Star 数突破了 10 万大关**：[@ggerganov 的反思](https://x.com/ggerganov/status/2038632534414680223)将 2026 年定为 **local agentic workflows**（本地智能体工作流）爆发的潜在年份，他认为实用的自动化并不需要 frontier-scale 的托管模型，而合适的便携式运行时堆栈（runtime stack）比绝对的规模更重要。该帖子还强调了**跨硬件、非厂商锁定（non-vendor-locked）基础设施（infra）**的重要性。
- **Apple Silicon 上的 Flash-MoE 引起了强烈关注**：一篇广为流传的帖子声称 **Qwen3.5-397B** 可以在一台 **48GB MacBook Pro** 上以 **4.4 tok/s** 的速度运行，其使用的是纯 **C + Metal** 引擎，从 SSD 流式传输权重并仅加载活跃的 experts，据报道在推理期间仅使用 **约 5.5GB RAM**（[摘要推文](https://x.com/heynavtoor/status/2038614549973401699)）。相关工作包括 [anemll-flash-mlx](https://x.com/anemll/status/2038684375425200360)（专注于在 MLX 之上优化 MoE 路径），以及 [AI Toolkit 新增加的 Apple Silicon 支持](https://x.com/ostrisai/status/2038643080400969940)。
- **Web 和服务堆栈也有新动态**：[Transformers.js v4](https://x.com/xenovacom/status/2038610331417608691) 为浏览器/Node/Bun/Deno 增加了 **WebGPU 后端**，带来了巨大的性能提升并支持 200 多种架构。[vLLM-Omni v0.18.0](https://x.com/vllm_project/status/2038415516772299011) 发布了 324 个 commit，包含生产级 TTS/omni 服务、统一量化、diffusion 运行时重构以及十几个新模型。在语音方面，[Artificial Analysis 报道了 Cohere Transcribe](https://x.com/ArtificialAnlys/status/2038678855213568031)：一个 **2B 参数的 conformer encoder-decoder**，采用 **Apache 2.0** 协议，在 **14 种语言**上进行训练，达到了 **4.7% 的 AA-WER**，转录速度约为实时的 **60 倍**。

**Agent 研究：自然语言测试框架、Meta-Harness、异步 SWE Agents 以及通过文件系统的长上下文处理**

- **测试框架工程（Harness engineering）正在成为一个独立的研究领域**：一篇来自清华/深圳的关于**自然语言 Agent 测试框架（harnesses）**的论文提出，让 LLM 从 SOP 中执行编排逻辑，而不是采用硬编码的框架规则。随着上下文预算（context budgets）的增加，多位从业者认为这一方向虽然令人脑洞大开，但具有可行性（[@rronak_ 摘要](https://x.com/rronak_/status/2038401494177694074)）。Meta 通过 **Meta-Harness** 进一步推进了这一想法，该方法在代码、追踪（traces）和评分上进行端到端的框架优化，而不仅仅是针对基础模型；声称的成果包括在 **TerminalBench-2 的 Haiku Agent 中排名第一**，并在文本分类和迁移方面取得了显著提升（[@yoonholeee](https://x.com/yoonholeee/status/2038640635482456118), [由 @LiorOnAI 提供的解读](https://x.com/LiorOnAI/status/2038669301541228606)）。
- **异步/多智能体（multi-agent）SWE 设计获得了更强的实证支持**：来自 CMU 的 **CAID** 论文主张使用**中心化异步隔离委托**（centralized asynchronous isolated delegation），利用管理智能体（manager agents）、依赖图、隔离的 git 工作树、自我验证和合并。据报道，与单智能体基准相比，其在 **PaperBench 上绝对提升了 26.7**，在 **Commit0 上提升了 14.3**，这表明并发和隔离优于单纯地让一个智能体进行更多次迭代（[@omarsar0 摘要](https://x.com/omarsar0/status/2038627572108743001)）。
- **将 Coding agents 视为长上下文处理器是最有趣的重新定义之一**：[@dair_ai](https://x.com/dair_ai/status/2038635382989005015) 重点介绍的一篇论文将巨大的语料库视为目录树，并允许现成的 coding agents 通过 shell 命令和 Python 进行导航，而不是将文本塞进上下文窗口或单纯依赖检索。报告的结果包括在 **BrowseComp-Plus（7.5 亿 token）上达到 88.5%** 的准确率（此前最高为 80%），且操作规模可达 **3T token**。

**训练、优化、评估与生产案例研究**

- **Muon 获得了重要的系统/数学优化**：[Gram Newton-Schulz](https://x.com/jcz42/status/2038660309968208028) 是 Muon 的 Newton-Schulz 步骤的即插即用替代方案。它作用于较小的对称 **XXᵀ Gram matrix** 而非大型矩形矩阵，据称使 Muon 的速度提高了 **2倍**，同时将验证困惑度（validation perplexity）保持在 **0.01** 以内。这项工作受到了 [@tri_dao](https://x.com/tri_dao/status/2038666307738964466) 的赞赏，认为这是真正具有跨学科意义的线性代数与 fast-kernel 结合的成果。
- **两个实际实现细节值得关注**：[Ross Wightman 指出](https://x.com/wightmanr/status/2038634643843682366)了 LLM 训练代码中一个微妙但重要的 **PyTorch `trunc_normal_` 误用模式**：默认的 `a/b` 是绝对值而非标准差，导致许多代码库实际上根本没有进行截断；他还提到了后来在 nightlies 版本中修复的数值异常。在应用层，[Shopify 的 DSPy 案例研究](https://x.com/dbreunig/status/2038650860843245814)在经济效益上非常显著：一张幻灯片强调了通过拆解业务逻辑、使用 DSPy 建模意图并切换到更小的优化模型，在保持性能的同时将年成本从 **550 万美元降至 7.3 万美元** ([后续跟进](https://x.com/kmad/status/2038659241238503716))。
- **新的评估/基准测试继续揭露差距**：[World Reasoning Arena](https://x.com/arankomatsuzaki/status/2038443186255991169) 针对假设性/世界模型推理，并报告了与人类之间的实质性差距。[Tau Bench 的新银行业务领域](https://x.com/_philschmid/status/2038655544613826985) 增加了一个包含 698 个文档的真实支持环境，目前最好的模型仍只能解决约 **25%** 的任务。与此同时，由斯坦福大学领导并由 [@Zulfikar_Ramzan](https://x.com/Zulfikar_Ramzan/status/2038408402809090554) 强调的一篇论文发现，**阿谀奉承的 AI (sycophantic AI)** 会增加用户的确定性，同时降低修复关系的意愿，这突显了“帮助性”指标可能会掩盖社会有害行为。

**热门推文（按互动量排序）**

- **Claude Code 计算机使用**：Anthropic 的发布是这组动态中最大的技术产品发布，对于日常 coding-agent UX 来说可能是影响最深远的 ([公告](https://x.com/claudeai/status/2038663014098899416))。
- **Claude Code 隐藏功能**：[@bcherny 的推文串](https://x.com/bcherny/status/2038454336355999749) 引起了巨大关注，反映了专家级用户现在围绕 coding-agent 工作流进行优化的速度有多快，而不再仅仅关注原始模型提示词 (prompts)。
- **Hermes Agent 更新**：社区对 [Nous 发布的 Hermes 重大更新](https://x.com/NousResearch/status/2038688578201346513) 的广泛反应表明，开源 Agent 框架已进入新的采用阶段。
- **Qwen3.5-Omni 发布**：阿里巴巴的多模态发布是当天最重要的模型公告之一，尤其是围绕音频/视频驱动应用创建的实际演示非常引人注目 ([发布](https://x.com/Alibaba_Qwen/status/2038636335272194241))。
- **llama.cpp 达到 10 万星**：[@ggerganov 的里程碑帖子](https://x.com/ggerganov/status/2038632534414680223) 捕捉到了本周“本地优先”的情绪：日益强大的开源模型加上日益强大的本地运行时 (runtimes)。


---

# AI Reddit 综述

## /r/LocalLlama + /r/localLLM 综述


### 1. Qwen 模型进展与应用

  - **[发现 Qwen 3.6！](https://www.reddit.com/r/LocalLLaMA/comments/1s7zy3u/qwen_36_spotted/)** (活跃度: 568)：**图片展示了 Qwen 视觉语言系列中即将推出的模型 "Qwen 3.6 Plus" 的预览，定于 2026 年 3 月 30 日发布。该模型以其高达 1,000,000 的 `context size` 而备受关注，这表明与之前的版本相比，它在处理海量数据输入方面有了重大飞跃。该模型还强调了通过收集 prompt 和 completion 数据来增强性能，表明其专注于迭代学习和改进。** 评论者推测 Qwen 3.6 可能会解决 3.5 版本中出现的“过度思考问题 (overthinking problem)”，并对其达到 SOTA 性能（尤其是 397B 模型）的潜力感到兴奋。人们还对 Coder 版本是否即将更新感到好奇。

- ambient_temp_xeno 提到的 '1 million context' 表明模型处理大型输入的能力显著增强，这可以提升其在需要大量上下文保留的任务中的表现。这是对先前版本的显著改进，可能允许更复杂且细微的交互。
- Long_comment_san 强调了当前模型中 '1.5 presence penalty' 的具体问题，认为它对角色扮演场景中的模型性能产生了负面影响。这种惩罚可能导致模型过度惩罚重复的话题或想法，从而阻碍创意或叙事任务。
- ForsookComparison 推测 397B 模型已接近实现 SOTA (state-of-the-art) 性能，表明虽然该模型具有庞大的参数量，但仍可能需要 Fine-tuning 来充分优化其功能。这反映了在平衡模型规模与实际性能提升方面的持续努力。

- **[使用本地 Qwen3-VL embedding 进行语义视频搜索，无需 API，无需转录](https://www.reddit.com/r/LocalLLaMA/comments/1s7u4fr/semantic_video_search_using_local_qwen3vl/)** (热度: 275): **该帖子讨论了使用 **Qwen3-VL-Embedding** 进行语义视频搜索，能够将原始视频直接嵌入到向量空间中进行自然语言查询，而无需转录或帧描述。**8B 模型**可在本地 Apple Silicon 和 CUDA 上运行，大约需要 `18GB RAM`，而 **2B 模型**约需要 `6GB`。开发者开发了一个 CLI 工具 [SentrySearch](https://github.com/ssrajadh/sentrysearch)，用于对视频素材进行索引和搜索，使用了 **ChromaDB**，最初基于 Gemini 的 API，但现在已支持本地 Qwen 后端。这种方法实现了高效的本地视频搜索，解决了对本地处理能力的共同需求。** 评论者赞赏多模态 AI 在解决实际问题上的创新应用，并对本地视频搜索功能表现出浓厚兴趣。一些用户对本地托管 Qwen3-VL 模型感到好奇，因为部分用户遇到了性能问题或高 VRAM 占用。

    - **neeeser** 询问有关本地托管 Qwen-3VL embedding 模型的问题，并指出了性能和资源占用方面的挑战。他们提到，即使在 4090 等高端 GPU 上运行该模型也很慢，且消耗大量 VRAM，强调了为此类模型制定高效部署策略的必要性。
    - **Octopotree** 询问系统是在查询期间实时处理视频，还是进行预处理。这种区别对于理解系统的架构和性能至关重要，因为实时处理可能耗费大量资源，而预处理则可以实现更快的查询响应。
    - 讨论涉及使用**多模态 AI** 进行视频搜索，这包括整合不同类型的数据（如视觉和文本）以增强搜索能力。这种方法可以潜在地解决复杂的搜索问题，而不依赖于转录等传统方法，提供更直接、更高效的解决方案。

- **[遇见 CODEC：开源框架终于让“嘿电脑，做这个”真正可行。屏幕读取。语音通话。多 Agent 研究。36 项技能。完全在你的机器上运行。](https://www.reddit.com/r/LocalLLM/comments/1s6h4g5/meet_codec_the_opensource_framework_that_finally/)** (热度: 175): ****CODEC** 是一个开源框架，旨在实现对计算机的全面语音和文本控制，完全在本地硬件上运行，无需外部 API 调用。它集成了多个 AI 模型，包括用于推理的 `Qwen 3.5 35B`、用于语音识别的 `Whisper` 以及用于语音合成的 `Kokoro`，全部运行在单台 Mac Studio 上。该框架包含七个系统，例如用于语音激活和应用控制的 **CODEC Core**、用于语音转文本的 **CODEC Dictate** 以及用于多 Agent 研究和文档处理的 **CODEC Chat**。它用本地实现取代了多个外部工具，强调隐私和自主性，并具有可扩展性，特别关注针对阅读障碍用户的无障碍功能。该项目已在 [GitHub](https://github.com/AVADSA25/codec) 开源，并采用 MIT 许可。** 评论者对在本地运行 `Qwen 3.5 35B` 等复杂 AI 模型的潜力感到兴奋，强调了该框架有效利用中端硬件的能力。人们对将 CODEC 适配到不同环境（如 Linux）很感兴趣，表明了对跨平台兼容性的需求。

- **bernieth** 强调了在本地运行像 Qwen 3.5 35b 这样先进模型的潜力，并强调了为了有效利用这些能力，一个实现良好的框架的重要性。这凸显了在不依赖云服务的情况下，在当地中端硬件上部署复杂 AI 解决方案日益增长的可行性。
- **super1701** 讨论了将 CODEC 与 Home Assistant (HA) 集成以增强功能，例如使用 Frigate 进行安全防护和日常任务自动化。这指出了 CODEC 在智能家居环境中的多功能性，允许 AI 与 IoT 设备之间进行无缝交互。
- **Aggravating_Fun_7692** 对 CODEC 和 Codex 之间的命名相似性表示担忧，这可能会导致混淆。这凸显了在 AI 领域建立独特品牌以避免误解的重要性，特别是在处理开源项目时。

### 3. AI 模型性能的技术讨论

- **[针对关注近期 TurboQuant 讨论的人员关于 TurboQuant / RaBitQ 的技术澄清](https://www.reddit.com/r/LocalLLaMA/comments/1s7nq6b/technical_clarification_on_turboquant_rabitq_for/)** (热度: 686): RaBitQ 论文的第一作者 **Jianyang Gao** 解决了在本地推理和 KV-cache 压缩背景下围绕 **TurboQuant** 和 **RaBitQ** 之间关系的混淆。Gao 强调了三个主要问题：(1) TurboQuant 对 RaBitQ 的描述不完整，忽略了关键的 Johnson-Lindenstrauss 变换；(2) TurboQuant 提出的理论主张缺乏依据，这与 RaBitQ 已确立的渐近最优性相矛盾；(3) 误导性的实证对比，RaBitQ 是在比 TurboQuant 更不利的条件下进行的测试。鉴于 TurboQuant 持续的推广及其即将在 **ICLR 2026** 上的展示，Gao 敦促进行公开澄清以纠正这些问题。[OpenReview 线程](https://openreview.net/forum?id=tO3ASKZlok)。评论者强调了实证对比问题的严重性，指出不公平的实验设置不应通过同行评审。他们还对 RaBitQ 作者表示同情，承认解决出版物准确性问题以及 TurboQuant 获得的意外关注所带来的挑战。

    - 开源 `llama.cpp` TurboQuant 实现背后的开发者分享了来自社区测试的详细性能指标。该实现在包括 Apple Silicon、NVIDIA 和 AMD 在内的各种硬件上进行了测试，结果显示非对称 `q8_0-K + turbo4-V` 配置几乎是无损的，在六个模型系列中困惑度 (perplexity) 仅增加了 `+0.0-0.2%`。此外，实现了显著的 `4.57x` KV 内存压缩，使一台 8GB 的 MacBook Air 能够处理 `4000+` token，而一台 16GB 的 RTX 5070 Ti 可以管理 `131K` 上下文 token。值得注意的是，在 Blackwell 统一内存上的 CUDA 实现达到了比未压缩数据更快的解码速度（`63.5 vs 50.1 tok/s`）。
    - 讨论强调了 Qwen Q4_K_M 上对称 turbo 量化的一个关键问题，该问题导致困惑度达到 `3,400+` 的灾难性性能。然而，使用非对称 `q8_0-K + turbo-V` 量化可以将性能恢复到基准水平。此问题归因于 K 精度通过 softmax 放大占主导地位，并且该发现已由多个独立测试者在 Metal 和 CUDA 上得到证实。底层技术涉及旋转和 Lloyd-Max 标量量化，目前关于该方法在 TurboQuant、RaBitQ 和之前的 Hadamard transform 工作之间的归属问题仍存在争议。
    - 一位评论者批评 TurboQuant 为“万灵油 (snake oil)”，认为现有的压缩技术如 Q8 和 Q4，以及 Hadamard transform，已经有效地使用了多年。这表明了对 TurboQuant 与成熟方法相比的新颖性和有效性的怀疑。

- **[在最近的 KV rotation PR 中发现，现有的 Q8 KV 量化在 AIME25 上性能大幅下降，但通过 rotation 可以基本恢复](https://www.reddit.com/r/LocalLLaMA/comments/1s720r8/in_the_recent_kv_rotation_pr_it_was_found_that/)** (Activity: 393): **GitHub 评论中的图片展示了 AIME25 模型使用不同 KV 量化类型的性能评估，特别关注了 rotation 对性能的影响。图片中的表格显示，不带 rotation 的 Q8_0 KV 类型得分为 `31.7%`，但在加入 rotation 后，得分提升至 `37.1%`。同样，不带 rotation 的 Q4_0 类型得分为 `0%`，但加入 rotation 后，得分提升至 `21.7%`。这表明 rotation 可以显著恢复某些量化配置下的性能，这对于使用 Q8 量化方法的用户尤为重要。** 评论者们对常规 Q8_0 KV cache 的糟糕表现感到惊讶，并注意到了 turboquant/rabitq 的潜在优势。人们也对 llama-eval 的发布充满期待，预计它将提高便利性。

    - 最近的基准测试强调了在 AIME25 模型上使用 Q8_0 KV 量化时性能显著下降，得分为 `31.7%`，而 F16 为 `37.9%`。然而，对 Q8_0 应用 rotation 恢复了大部分损失的性能，使得分回升至 `37.1%`。这表明 rotation 可能是优化量化模型的关键因素，特别是为了保持接近 F16 等高精度格式的性能水平。
    - 数据表明，不带 rotation 的 Q8_0 KV cache 甚至比带 rotation 的 Q5_1 和 Q4_0 表现更差。具体而言，带 rotation 的 Q5_1 达到了 `32.5%` 的得分，而带 rotation 的 Q4_0 的得分从 `2.0%` 跃升至 `21.7%`。这证明了 rotation 具有显著增强低精度量化性能的潜力，使其在实际应用中更具可行性。
    - 围绕 turboquant/rabitq 的讨论表明，这些技术可以为量化性能提供实质性的改进。尽管存在疑虑，但基准测试的证据支持这样一种观点：先进的量化方法（如涉及 rotation 的方法）可以减轻通常与低精度 KV caches 相关的性能退化。


## 技术性较低的 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Anthropic 的 Claude Mythos 及其 AI 模型进展

  - **[Anthropic 正在测试其“史上最强大的 AI 模型” Mythos | Fortune](https://www.reddit.com/r/singularity/comments/1s4t3k7/anthropic_is_testing_mythos_its_most_powerful_ai/)** (热度: 2028): **Anthropic 正在测试一款名为 'Claude Mythos' 的新 AI 模型，该模型被描述为“有史以来开发的最强大的 AI 模型”。这款模型属于名为 'Capybara' 的新层级，超越了现有的 Opus 系列。由于 CMS 配置错误导致的泄露草案资料显示，该模型在 reasoning（推理）、coding（代码）和 cybersecurity（网络安全）任务方面有显著改进，标志着能力的“阶跃式变化”。由于潜在的滥用风险，该公司对其推广持谨慎态度，初始访问权限主要集中在能够增强网络安全防御能力的组织。** 评论反映了讽刺与技术兴趣的交织，一些用户对测试性能较弱模型的实用性表示怀疑，而另一些人则强调了该模型相对于之前版本的重大进步。

    - RedRock727 强调，Anthropic 的新模型 'Claude Mythos' 据报道比以前的模型有了显著进步，在 reasoning、coding 和 cybersecurity 任务中均有提升。该模型属于名为 'Capybara' 的新层级，定位在当前的 Opus 系列之上，预示着提升 AI 能力的战略举措。此次进展是在一次由于 CMS 资产配置错误导致的数据泄露事件后曝光的，Anthropic 将其归因于人为错误。
    - exordin26 详细说明了名为 'Capybara' 的新 AI 模型层级，其被描述为比之前的 Opus 模型更大、更智能。这表明 'Capybara' 和 'Mythos' 可能指的是同一个底层模型，意味着 Anthropic 的 AI 产品线将迎来重大升级。对新层级的关注突显了 Anthropic 致力于推进 AI 技术并应对潜在滥用风险（特别是在网络安全领域）的承诺。
    - 围绕泄露草案的讨论强调了 Anthropic 对推出 'Mythos' 的谨慎态度，尤其是考虑到其增强的 cyber 能力。公司最初将访问权限限制在能够加强防御的组织，反映了对潜在滥用的担忧。这一战略性推广是 Anthropic 为确保部署先进 AI 模型时的安全性和保障所做出的更广泛努力的一部分。

  - **[独家：在意外数据泄露揭露其存在后，Anthropic 承认正在测试代表能力“阶跃式变化”的新型 AI 模型](https://www.reddit.com/r/ClaudeAI/comments/1s4ucsu/exclusive_anthropic_acknowledges_testing_new_ai/)** (热度: 1261): **据报道，Anthropic 正在测试一款新型 AI 模型，其能力与之前的版本相比代表了重大进步。这一信息是在一次意外数据泄露后出现的。该模型目前正在早期访问客户中进行测试，表明它可能很快会更广泛地提供。此次泄露引发了人们对该模型潜在影响以及相对于先前版本改进的兴趣和猜测。** 一些评论者表示怀疑，将其比作典型的营销炒作，而另一些人则认为泄露可能是一种有效的营销策略。

    - 讨论强调了一个潜在的安全隐患，因为 Anthropic 新 AI 模型的泄露恰逢该模型据称具有破坏网络安全的能力。这引发了对 Anthropic 自身安全措施稳健性的质疑，特别是考虑到该模型的高级功能。
    - 对 Anthropic 模型的命名惯例进行了幽默的评价，指出其从 'Opus' 和 'Sonnet' 等优雅的音乐术语转向了像 'Capybara' 这样更奇特的名字。这可能反映了品牌战略的变化，或者是试图在拥挤的市场中区分新模型。
    - 对于数据泄露的“意外”性质存在怀疑，一些人认为这可能是一个战略性的营销手段。泄露的内容包括完整的采访和准备好的引言，这可能表明这是一个受控的发布，旨在为新模型制造话题和关注度。


### 2. OpenAI 的挑战与项目取消

- **[OpenAI 陷入了大麻烦](https://www.reddit.com/r/OpenAI/comments/1s4sbdn/openai_is_in_big_trouble/)** (活跃度: 2616): **图片是《大西洋月刊》(The Atlantic) 一篇题为 "OpenAI Is Doing Everything ... Poorly" 的文章截图，该文批评了 OpenAI 最近的战略决策和项目取消。文章强调了 OpenAI 搁置或取消的几项计划，例如 Sora 视频生成器和 Stargate 项目，并指出承诺的硬件也出现了延迟。这些举动被解读为 OpenAI 陷入困境的信号，因为他们正面临来自 Anthropic 和 Google 的 Gemini 等其他 AI 公司的竞争。文章认为，在算力短缺的情况下，OpenAI 的重点正在转向更具盈能力的企业级解决方案，而非面向消费者的项目。** 评论者认为，OpenAI 的决策反映了由于算力短缺而向企业级解决方案的战略转型，而非陷入困境的迹象。他们指出，像 Sora 这样的项目在财务上是不可持续的，每天耗资 1500 万美元，专注于企业业务是更可行的商业策略。

    - **triclavian** 强调了 OpenAI 由于全球算力短缺而优先考虑企业客户的战略转变。削减利润较低的服务（如 AI 视频生成）被视为优化资源以用于更有利可图的企业级应用，这表明其正专注于可持续的商业实践。
    - **ripestmango** 指出了维持 Sora 等免费服务的财务负担，据报道每天耗资 1500 万美元。该评论者支持停止此类服务的决定，认为这些服务产生了过多的低价值 AI 内容，并建议将资源重新分配给更有影响力的项目。
    - **cfeichtner13** 认为视频和图像生成不具盈利性且消耗大量计算资源。他们指出来自中国的类似技术优于 OpenAI 的产品，并建议专注于企业级解决方案和机器人技术是更可行的前进道路，特别是考虑到扩展数据中心容量的挑战。

- **[这是执行力差，还是仅仅是一个公司在不断尝试](https://www.reddit.com/r/OpenAI/comments/1s4ui3n/is_this_poor_execution_or_just_a_company_at_work/)** (活跃度: 713): **图片是对 OpenAI 最近商业决策的模因式 (meme-style) 批评，强调了 Sora 视频生成器和 Stargate 项目等多个发布后又被取消或推迟的项目。Katie Miller 的推文和《大西洋月刊》的标题表明，这些行为可能反映的是执行力差，而非战略性的实验。评论讨论了 OpenAI 在寻找可扩展且盈利的商业模式方面面临的挑战，指出尽管该公司拥有庞大的用户群，但仍处于初创阶段。** 评论者认为，OpenAI 的行为可能是由寻找盈利能力和可持续商业模式的需求驱动的，一些人认为公司目前的状况是初创公司仍在寻找可行道路的典型表现。

    - **handbrake2k** 强调了 OpenAI 面临的一个常见初创公司挑战：在获得庞大用户群后实现可扩展且盈利的商业模式。这种情况颇具讽刺意味，因为 OpenAI 的做法可能曾受到 Y-Combinator 的批评，而后者以向初创公司提供可持续增长策略建议而闻名。
    - **edjez** 批评了对消费者视频娱乐的关注，认为到 2026 年为此保留 GPU 资源是不切实际的。这暗示 OpenAI 需要将其资源重新对准更可持续和更有利可图的项目。
    - **Acedia_spark** 认为 OpenAI 争夺市场份额的急于求成可能导致了被察觉到的无能。向企业级解决方案的转型虽然具有潜在战略意义，但在更广泛的运营挑战中显得像是反应过度，被比作“试图在泰坦尼克号沉到一半时止损”。

- **[OpenAI 在顾问、投资者和员工发出警示后暂停了“成人模式 (Adult Mode)”](https://www.reddit.com/r/OpenAI/comments/1s4a1r6/openai_halts_adult_mode_as_advisors_investors_and/)** (热度: 654): **OpenAI** 已暂停其“成人模式”聊天机器人的开发，原因是员工、投资者和其咨询委员会对性化 AI 内容的社会影响表示担忧。一个关键问题是年龄验证系统，该系统在 `12%` 的案例中错误地将未成年人识别为成年人，引发了严重的伦理和安全担忧。OpenAI 目前正将重点转向生产力工具和基于 **ChatGPT** 的“超级应用”。更多细节可以在[这里](https://the-decoder.com/openai-halts-adult-mode-as-advisors-investors-and-employees-raise-red-flags/)找到。评论者对 AI 作为“性感自杀教练”的叙事表示怀疑，并批评 OpenAI 可能会与保守价值观保持一致，暗示如果公众使用受到限制，可能会转向军事应用。

    - 一位用户指出，**Gemini** 和 **Grok** 等其他语言模型已经支持成人内容，质疑为什么 OpenAI 停止“成人模式”的决定被视为负面信号（red flag）。这表明在 AI 内容审核的行业标准或公众认知方面可能存在不一致。
    - 另一条评论指出了 OpenAI 这一决定的讽刺之处，暗示如果该公司继续迎合保守观点，它可能会转向军事合同而不是公众使用。这反映了关于 AI 部署的伦理和社会影响的更广泛辩论，特别是在平衡道德价值观与技术能力方面。


### 3. Claude 使用问题与订阅投诉

  - **[关于会话限制的更新](https://www.reddit.com/r/ClaudeAI/comments/1s4idaq/update_on_session_limits/)** (热度: 2467): **Anthropic** 调整了其 Claude AI 服务在高峰时段（工作日，太平洋时间上午 5 点至 11 点 / 格林威治标准时间下午 1 点至 7 点）针对免费版、Pro 版和 Max 版订阅的 5 小时会话限制。虽然每周限制保持不变，但用户在这些时段会更快地耗尽其会话限额。这一变化影响了约 `7%` 的用户，特别是 Pro 层级的用户，旨在应对不断增长的需求。建议运行 Token 密集型任务的用户将其安排在非高峰时段，以最大限度地利用会话。评论者批评 **Anthropic** 缺乏透明度，认为这一变更是悄悄实施的，并对高峰时段限制的降低表示沮丧。他们强调了公开沟通的重要性，尤其是在处理扩展挑战时。

    - shyney 强调，会话限制并非 Bug，而是 Anthropic 有意的改动，认为这是为了避免用户反弹而悄悄进行的。这表明了在不进行前期沟通的情况下管理系统资源的策略性决策，这可能会影响用户的信任和透明度。
    - Wise-Reflection-7400 注意到资源分配的转变，之前提供的 2 倍非高峰时段奖励已被降低的高峰限额所抵消。这反映了资源管理中的一种常见策略，即通过调整福利来有效地管理需求和系统负载。
    - This-Shape2193 批评了关于会话限制沟通缺乏透明度，强调如果公开沟通扩展挑战，用户本可以理解。该评论强调了在重大运营变更期间，有效的消费者外联和公关对于维持用户信任的重要性。

  - **[这不对劲](https://www.reddit.com/r/ClaudeAI/comments/1s55mvg/this_isnt_right/)** (热度: 888): **该帖子强调了对 **Claude AI** 使用透明度和会话限制的担忧，尤其是对 Pro 层级用户。用户报告称，简单的互动（如打招呼“Hello”和询问天气）就消耗了其 `7%` 的使用配额，他们认为这太高了。该用户还批评客户服务毫无帮助，因为它依赖于一个只会重申政策而不解决问题的聊天机器人。**评论者对该服务表示不满，其中一位用户指出，他们在仅发送两条消息后就达到了会话限制，质疑这是否正常。另一位用户提到，由于缺乏透明度以及感知到的服务质量下降，他们取消了订阅。

- 用户报告了 Claude AI Pro 订阅的重大限制，即使是处理两个 Word 文档或对书籍进行简单的排版修改这类极低的使用量，也会迅速耗尽会话限制（session limits）。这导致了不满和退订，因为用户觉得该服务不符合订阅模式所设定的预期。
- Claude AI Pro 订阅的使用限制显著缺乏透明度。用户对使用配额的快速耗尽表达了沮丧，而这些限制在购买时并未明确告知，导致用户认为服务质量和价值有所下降。
- 一些用户将 Claude AI 与 Gemini 等竞争对手进行对比，认为其表现欠佳，并将服务质量和透明度的下降作为更换平台的理由。普遍情绪认为，尽管用户之前对该平台很忠诚，但目前的限制和缺乏清晰沟通正在导致用户流失。

  - **[昨天刚订阅了 Pro 就达到限制了。这是诈骗吗？](https://www.reddit.com/r/ClaudeAI/comments/1s54pfu/subscribed_yesterday_to_pro_and_im_already_hit_by/)** (Activity: 900): **一位用户以每月 20 美元的价格订阅了 **Claude Pro** 作为编程助手，但在为一个 WordPress 插件工作仅两小时后就遇到了使用限制。该用户对服务表示不满，指出他们并未处理大文件或复杂任务，并由于退款流程的问题决定取消订阅。这引发了人们对 Pro 计划对开发者实用性的担忧，特别是考虑到 **Sonnet 3.5/Opus** 所设定的预期。** 几位用户报告了 **Claude Pro** 订阅的类似问题，指出在进行极少交互（如编辑两个 Word 文档或典型的提示词）后出现了意料之外的使用限制。这表明最近的使用政策或限制可能有所变动，导致了用户的不满以及不再续订的决定。

    - 用户报告了 Pro 订阅使用限制的意外变化，一些人在发送典型提示词后，使用百分比显著增加。一位用户注意到他们很快就达到了 50% 的使用率，这表明服务的使用政策或计算方法可能发生了改变。
    - 一位升级到 Max 计划（费用约为 100 美元）的用户报告称，在活跃使用仅三小时后就达到了使用限制。这与其之前的体验形成了鲜明对比，表明使用情况的追踪或执行方式可能发生了变化。
    - 用户担心这些新的限制可能会驱使他们转向其他替代的 AI 服务，例如 Claude。普遍情绪是，如果这些问题得不到解决，可能会导致用户留存率下降，类似于过去从 ChatGPT 转向其他平台的情况。

# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快就会发布新的 AINews。感谢阅读到这里，这是一段美好的历程。