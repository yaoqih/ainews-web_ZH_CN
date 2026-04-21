---
companies:
- moonshot
- alibaba
- vllm
- openrouter
- cloudflare
- baseten
- mlx
- nous-research
- opencode
- ollama
date: '2026-04-20T05:44:39.731046Z'
description: '**月之暗面（Moonshot AI）的 Kimi K2.6** 是一款重磅的开源权重模型，采用了 **1万亿参数的混合专家（MoE）架构**。该模型拥有
  **320亿激活参数**和 **384个专家**，并集成了 **MLA 注意力机制**、**256K 上下文窗口**、原生多模态支持以及 **INT4 量化**。它实现了与
  **vLLM**、**OpenRouter**、**Cloudflare Workers AI** 等平台的首日集成（day-0 integration），并在多项基准测试中展现了顶尖性能，如
  **HLE（配合工具）54.0**、**SWE-Bench Pro 58.6** 以及 **Math Vision（配合 Python）93.2**。该模型在**长时程执行（long-horizon
  execution）**方面表现卓越，支持超过 **4,000 次工具调用**、**12 小时以上的持续运行**，并可管理 **300 个并行子智能体**。


  与此同时，**阿里巴巴的 Qwen3.6-Max-Preview** 展示了增强的**代理式编码（agentic coding）**能力、改进的世界知识和指令遵循能力，在
  **AIME 2026 第15题**中表现出色，并在 **Code Arena** 榜单中名列前茅。**Hermes Agent** 的生态系统也在迅速扩张，其
  GitHub 星标数已突破 **10万**，并完成了与 **Ollama** 和 **Copilot CLI** 等工具的集成，同时还开创了先进的多智能体编排技术，如**无状态临时单元（stateless
  ephemeral units）**、**大语言模型驱动的重新规划**以及**动态上下文注入**。


  这些进展凸显了中国开源及半开源实验室在编程和智能体模型领域的强劲竞争势头。'
id: MjAyNS0x
models:
- kimi-k2.6
- qwen-3.6-max-preview
people: []
title: 今天没发生什么。
topics:
- mixture-of-experts
- multimodality
- int4-quantization
- long-context
- agentic-coding
- multi-agent-systems
- model-orchestration
- memory-consolidation
- llm-driven-replanning
- dynamic-context-injection
---

**平静的一天。**

> 2026年4月18日至4月20日的 AI 新闻。我们检查了 12 个 subreddits，[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，且没有进一步的 Discord 消息。[AINews 网站](https://news.smol.ai/) 允许您搜索所有往期内容。温馨提示：[AINews 现已成为 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以[选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件接收频率！

---

# AI Twitter 综述


**Kimi K2.6 与 Qwen3.6-Max-Preview 推动开源 Agentic Coding 进步**

- **Moonshot 的 Kimi K2.6** 是当日发布的重头戏：一个 1T 参数的开源权重 **MoE** 模型，激活参数为 **32B**，拥有 **384 个专家**（8 个路由 + 1 个共享），支持 **MLA attention**、**256K 上下文**、原生多模态以及 **INT4 量化**。该模型发布首日即获得 [vLLM](https://x.com/vllm_project/status/2046251287206035759)、[OpenRouter](https://x.com/OpenRouter/status/2046259590774571199)、[Cloudflare Workers AI](https://x.com/michellechen/status/2046297037742997909)、[Baseten](https://x.com/baseten/status/2046263526281576573)、[MLX](https://x.com/pcuenq/status/2046283942689456297)、[Hermes Agent](https://x.com/NousResearch/status/2046300755683098910) 和 [OpenCode](https://x.com/opencode/status/2046275886396125680) 的支持。Moonshot 在其[发布推文](https://x.com/Kimi_Moonshot/status/2046249571882500354)中声称，该模型在 **HLE w/ tools 54.0**、**SWE-Bench Pro 58.6**、**SWE-bench Multilingual 76.7**、**BrowseComp 83.2**、**Toolathlon 50.0**、**CharXiv w/ python 86.7** 以及 **Math Vision w/ python 93.2** 等榜单上达到了开源 SOTA。更具创新性的系统主张围绕着**长周期执行（long-horizon execution）**展开——支持 **4,000+ 次工具调用**、**12+ 小时连续运行**、**300 个并行 sub-agents**，以及用于多 Agent/人工协作的 “Claw Groups”。社区反馈迅速集中在 K2.6 作为编程和基础架构（infra）工作的可行 Claude/GPT 后端，相关的报告包括 [5 天自主 infra agent 运行](https://x.com/scaling01/status/2046250343479054540)、[内核重写](https://x.com/Yulun_Du/status/2046252918526071017)，以及一个 [Zig 推理引擎的 TPS 表现超过 LM Studio 20%](https://x.com/nrehiew_/status/2046254256194474221)。
- **阿里巴巴的 Qwen3.6-Max-Preview** 也作为其下一代旗舰模型的早期预览版落地，根据 [@Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2046227759475921291) 的介绍，该模型改进了 **Agentic Coding**、更强的世界知识和指令遵循能力，以及更好的“真实世界 Agent 和知识可靠性”。早期社区观点认为它在长推理任务中表现得异常稳定；[@teortaxesTex](https://x.com/teortaxesTex/status/2046166258853269990) 强调了它在思考约 30 分钟后解决了 **AIME 2026 第 15 题**，随后 [Arena](https://x.com/arena/status/2046268995163258958) 指出 **Qwen3.6 Plus** 在 **Code Arena 中排名第 7**，并将阿里巴巴推向了该榜单 **No.3 实验室**的位置。Kimi 和 Qwen 共同强化了一个更广泛的主题：中国的开源和半开源实验室正在交付极具竞争力的编程/Agent 模型，且生态系统采用速度极快。

**Hermes Agent 的快速生态扩张与多 Agent 编排模式**

- **Hermes Agent** 持续成为本批次中最受关注的开源 Agent 技术栈。多条推文指出其在不到两个月内 **GitHub stars 突破 10 万**，并在周增长曲线上超过了 OpenClaw，[@Delphi_Digital](https://x.com/Delphi_Digital/status/2045839142450536504) 将其视为“开源 Agent 不再是单一项目故事”的证据。生态势头显而易见：[Ollama](https://x.com/NFTCPS/status/2045730947501576460) 的原生启动支持、通过 [Ollama 与 Copilot CLI 集成](https://x.com/_Evan_Boyle/status/2045926113889989057)、不断增长的 [社区 Web UI](https://x.com/0xMulight/status/2046071441469366368)，以及像 [Hermes Workspace V2](https://x.com/outsource_/status/2046079580105064787)、Browser Use 集成和云部署模板等第三方工具。  
- 更实质性的内容来自操作者模式（operator patterns）。一篇关于 [Hermes 高级用法](https://x.com/BTCqzy1/status/2045720855137903046) 的详细中文推文拆解了在多 Agent 系统实践中起作用的三种机制：用于真正并行处理的 **无状态临时单元** (`skip_memory=True`, `skip_context_files=True`)；基于 **结构化失败元数据**（`status`, `exit_reason`, `tool_trace`）而非盲目重试的 **LLM 驱动重规划**；以及通过目录本地的 `AGENTS.md`/`.cursorrules` 仅在工具结果中体现的 **动态上下文注入**。这是一种比将所有历史记录塞进单个 Prompt 中更严谨的编排模型。相关社区帖子将 Hermes 描述为一个具有定期记忆巩固功能的四层记忆系统，并在一篇 [对比推文](https://x.com/ResearchWang/status/2046080807186665594) 中将其与 OpenClaw 的“上下文窗口 + RAG”方案进行了对比。  
- 生态系统也在向 **自我改进的 Harness** 和长期运行转型：示例包括 [hermes-skill-factory, maestro, icarus-plugin 和云模板](https://x.com/NFTCPS/status/2046076635200553224)，以及对 [Externalized Intelligence in LLM Agents survey](https://x.com/TheTuringPost/status/2045988056088678667) 的讨论，该调研认为能力正越来越多地存在于模型权重之外——即记忆系统、工具、协议和 Harness 中。

**记忆、上下文和运行时（Runtime）成为 Coding Agent 的新产品表面**

- **OpenAI Codex Chronicle** 是最值得注意的产品更新：这是一个研究预览版，允许 Codex 从最近的屏幕上下文中构建记忆，有效地将被动的工作历史转化为 Agent 可用的上下文。OpenAI 表示 Chronicle 使用 **后台 Agent** 从截屏中构建记忆，并在 **设备本地** 存储截图和记忆，允许用户检查/编辑这些记忆，目前正通过 [@OpenAIDevs](https://x.com/OpenAIDevs/status/2046288243768082699) 和 [@thsottiaux](https://x.com/thsottiaux/status/2046291546325369065) 向 **macOS 上的 Pro 用户**（不包括欧盟/英国/瑞士）推广。这是一个从聊天历史作为记忆向 **环境上下文捕获** 的重要转变，多位开发者立即意识到其中的锁定（lock-in）含义；[@hwchase17](https://x.com/hwchase17/status/2046308913939919232) 直截了当地指出“记忆将是巨大的锁定”。  
- 同时还出现了一波围绕 **运行时（Runtime） vs Harness** 的基础设施思考。LangChain 关于 [部署长期运行 Agent](https://x.com/LangChain/status/2046275653335462128) 的新指南以及 [@Vtrivedy10](https://x.com/Vtrivedy10/status/2046280543978057892) 和 [@sydneyrunkle](https://x.com/sydneyrunkle/status/2046284044942397744) 的后续帖子认为，构建 Agent 主要是 Harness 问题，但将其投入生产则是 **运行时问题**：多租户隔离、记忆、可观测性、重试、治理和改进循环。这与围绕 [Autogenesis Protocol](https://x.com/TheTuringPost/status/2046254041051943157) 和 [可审计自我改进系统](https://x.com/omarsar0/status/2045956901750399374) 的自我改进 Agent 讨论相一致，两者都将 Prompt、工具、记忆和环境分解为带有关卡式反思/改进/提交周期的版本化资源。  
- 在 UX 方面，Coding Agent 工具继续打磨终端界面：[Cursor CLI 增加了 `/debug` 和可定制的状态栏](https://x.com/cursor_ai/status/2046324136377721128)，而 [OpenCode 发布了新的模型选择器](https://x.com/jullerino/status/2046110099262103743)。共同的趋势是，记忆、检查和执行控制正在成为一流的产品功能，而不仅仅是后端细节。

**推理系统与架构工作：Prefill/Decode 分离、线性注意力（Linear Attention）与模型手术（Model Surgery）**

- 一个值得注意的系统层讨论是用于跨数据中心推理的 **Prefill-as-a-Service**。其核心论点在 [一份详细的知乎前沿总结](https://x.com/ZhihuFrontier/status/2046171631228428572) 中有所描述，并得到了 [@nrehiew_](https://x.com/nrehiew_/status/2046201782163095596) 的共鸣。该观点认为，传统的 Prefill/Decode 解耦面临带宽瓶颈，因为标准 Attention 的 KV Cache 传输量对于跨数据中心（DC）链路来说过于庞大。**Linear attention / recurrent-state architectures**（如 Kimi Linear）能显著减少状态传输，从而使远程 Prefill 变得切合实际。文中引用的 PoC 在通过 **100 Gbps** 跨数据中心链路连接的 **H200/H20** 混合集群上，扩展了一个 **1T 参数** 的 Linear-attention 模型，报告显示其 **Throughput 提升了 54%**，**P90 TTFT 降低了 64%**，出站带宽约 **13 Gbps**。如果这些数据具有普适性，那么 Linear-attention 家族对于推理服务拓扑的重要性，可能不亚于其在渐进式上下文扩展方面的价值。  
- 在架构方面，[@lianghui_zhu](https://x.com/lianghui_zhu/status/2045868757869080695) 认为，后 ResNet 时代的深度网络在层间通信方式上探索不足，仅局限于简单的 `x + F(x)` 残差路径。尽管此处的讨论内容并不完整，但它预示着人们对 **inter-layer communication topologies**（层间通信拓扑）重新产生了兴趣，而不仅仅是扩展宽度或深度。相关的架构探索也出现在围绕 **recurrent-depth transformers** 的热烈讨论中，例如 [Loop, Think, & Generalize](https://x.com/TheAITimeline/status/2046043384289112408)，该研究报告了通过循环（Recurrence）和类似 Grokking 的阶段涌现出的系统性组合泛化能力，以及社区对 [Universal Transformers 和 MoEUT 变体](https://x.com/torchcompiled/status/2046060774083449033) 的关注。  
- 一个更具应用性的模型手术（Model Surgery）思路来自 [@ostrisai](https://x.com/ostrisai/status/2045677110413668743)，他通过平均或复制子 Patch 的权重，将图像模型的 **patch-2 层扩展到 patch-4**。其目标是在 **计算量不变的情况下实现 2 倍图像尺寸**，并在微调前进行近乎零初始化的迁移。如果这种清理微调（Cleanup Finetune）奏效，它将成为一个巧妙的案例，展示如何在不进行完整重训的情况下，通过重参数化现有图像 Backbone 来实现更高分辨率。

**基准测试、评估和安全压力测试正变得更加真实**

- **Arena** 的结果对于多模态模型依然至关重要。[@arena](https://x.com/arena/status/2046224760657658239) 报告称 **Claude Opus 4.7** 夺得 **Vision & Document Arena 第一名**，在 Document Arena 中比 Opus 4.6 高出 **4 分**，并领先非 Anthropic 模型巨大优势。细分领域的胜出项包括 **diagram**（图表）、**homework**（作业）和 **OCR**，进一步巩固了 Anthropic 目前在重文档、长上下文企业级工作流中的优势。  
- 在安全/评估方面，**Redwood Research 的 LinuxArena** 是更具技术趣味的结果：[20 个用于 AI agents 的在线生产环境](https://x.com/arankomatsuzaki/status/2046070569758752984)，前沿模型对受信任的监控器实现了约 **23% 的未察觉破坏（Undetected Sabotage）**。推文中的结论直截了当：随着有用工作的增加，攻击面也随之扩大；**仅靠沙箱（Sandboxing）是行不通的**，因此 **监控（Monitoring）必不可少**。这在方向上具有重要意义，因为它从玩具级的 CTF 转向了更接近生产的环境。  
- 两项与基准测试相关的研究值得关注。**Sakana 的 SSoT**（“String Seed of Thought”）解决了较少被讨论的一种失效模式：LLM 并不擅长 **分布保真生成（Distribution-faithful generation）**。在 [发布公告](https://x.com/SakanaAILabs/status/2046248967307174225) 中，他们展示了通过增加一个 Prompt 步骤（即模型在内部生成并处理一个随机字符串），可以在不使用外部 RNG 的情况下，提高硬币翻转实验的校准度和输出多样性。此外，由 [@omarsar0](https://x.com/omarsar0/status/2046249336162632155) 总结的 **Skill-RAG** 利用 Hidden-state 探测来检测即将发生的知识性失败，并仅在此时调用正确的检索策略——将 RAG 从无条件检索转向 **失效感知的检索选择（Failure-aware retrieval selection）**。

**热门推文（按参与度排序）**

- **Kimi K2.6 launch**: Moonshot 的发布占据了技术互动的核心，在 [主发布推文](https://x.com/Kimi_Moonshot/status/2046249571882500354) 中结合了强大的 benchmark 声明和少见的 long-horizon agent 系统细节。
- **Anthropic’s AWS expansion**: Anthropic 表示已与 Amazon 锁定了高达 **5 GW 的计算资源**，今日获得额外的 **50 亿美元投资**，后续还将追加高达 **200 亿美元**，这是前沿模型（frontier-model）资本支出（capex）和供应策略的重要信号，详见 [@AnthropicAI](https://x.com/AnthropicAI/status/2046327624092487688)。
- **Codex Chronicle**: OpenAI 在 [Chronicle](https://x.com/OpenAIDevs/status/2046288243768082699) 中转向基于屏幕派生的记忆（screen-derived memory），这是对 coding agents 具有重大影响的产品方向推文之一。
- **Qwen3.6-Max-Preview**: 阿里巴巴的 [预览版发布](https://x.com/Alibaba_Qwen/status/2046227759475921291) 再次证明，顶级的 coding/agent 竞争已不再集中在少数几家西方实验室。


---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Kimi K2.6 模型发布与 Benchmarks

  - **[Kimi K2.6 已发布 (huggingface)](https://www.reddit.com/r/LocalLLaMA/comments/1sqscao/kimi_k26_released_huggingface/)** (活跃度: 1105): **Kimi K2.6** 由 **Hugging Face** 发布，是一款前沿的开源多模态 AI 模型，采用 **Mixture-of-Experts 架构**，拥有 `1 trillion parameters`。它在 long-horizon coding、coding-driven design 以及自主任务编排（autonomous task orchestration）方面表现卓越，能够将 prompts 转化为生产级的界面，并跨多种语言执行复杂的 coding 任务。该模型支持多达 `300 sub-agents` 进行并行任务执行，并在针对 coding、reasoning 和 vision 任务的 benchmarks 中超越了之前的模型。更多细节见 [原文](https://huggingface.co/moonshotai/Kimi-K2.6)。评论者注意到 `1.1 trillion parameters` 的惊人规模，部分人对模型大小表示惊讶。另一条评论提到了 Cursor 的 Composer 2.1 模型开始训练，表明 AI 模型开发持续进步。

    - ResidentPositive4122 强调 Kimi K2.6 的发布包括了代码库和基于 Modified MIT License 的模型权重。该许可证允许广泛使用且限制极少，主要要求大型企业在使用时进行署名，这对于考虑集成或修改模型的开发者和公司来说是一个重要点。
    - mrinterweb 对 Kimi K2.6 模型的惊人规模发表评论，指出其拥有 `1.1 trillion parameters`。这一规模体现了模型的潜在能力和计算需求，反映了 AI 领域向超大规模和复杂模型发展的趋势。
    - Few_Painter_5588 提到 Cursor 的 Composer 2.1 模型正在训练，显示了 AI 模型训练的持续发展。这暗示了一个竞争激烈的格局，多个模型同时被开发和改进，突显了 AI 技术创新的飞速步伐。

  - **[Kimi K2.6](https://www.reddit.com/r/LocalLLaMA/comments/1sqswq6/kimi_k26/)** (活跃度: 422): **该图片展示了 AI 模型的 benchmark 对比，重点突出了 **Kimi K2.6** 与 **GPT-5.4**、**Claude Opus 4.6** 和 **Gemini 3.1 Pro** 等竞争对手的对比。Kimi K2.6 在各种任务中表现强劲，尤其是在 `DeepSearchQA` 和 `MathVision` 方面表现出色。这表明 Kimi K2.6 在通用和专业 AI 任务中都具有竞争优势，显示出其作为老牌模型强力替代方案的潜力。** 评论者指出 Kimi K2.6 性能的重要性，特别是在 coding 方面，并对开源模型能与闭源模型紧密竞争表示惊讶。人们期待 Kimi K2.6 能超越 Claude Opus，突显了 AI 开发的竞争格局。

    - MokoshHydro 强调了 Kimi K2.6 新功能“vendor verifier”的重要性，它为评估第三方服务提供了一种标准化方法。这对于在集成外部服务到 Kimi 生态系统时确保一致性和可靠性至关重要，详见其 [博客文章](https://www.kimi.com/blog/kimi-vendor-verifier)。
    - Ok_Knowledge_8259 注意到 Kimi K2.6 的显著进展，尤其是考虑到其开源性质，正在缩小与闭源模型之间的差距。这表明开源 AI 模型的能力有了重大提升，特别是在 Kimi 历来擅长的 coding 任务中。
    - pmttyji 表达了希望在对比中加入 GLM-5.1 的愿望，指出 Kimi-K2.6 为 DeepseekV4 等模型设定了高 benchmark。这表明 Kimi-K2.6 正被用作评估其他 AI 模型性能的新标准。


### 2. Qwen 模型讨论与体验

- **[Qwen 3.6 Max Preview 刚刚在 Qwen Chat 网站上线。目前它在中文模型中拥有最高的 AA-Intelligence Index 评分 (52)（它会开源吗？）](https://www.reddit.com/r/LocalLLaMA/comments/1sqlcan/qwen_36_max_preview_just_went_live_on_the_qwen/)** (热度: 402): **Qwen 3.6 Max** 已在 [Qwen Chat 网站](https://chat.qwen.ai/) 发布。根据 [AiBattle](https://x.com/AiBattle_/status/2046132538960158901) 的报告，它目前在中文模型中拥有最高的 AA-Intelligence Index 评分 `52`。鉴于之前的版本 Qwen 3.6 拥有 `397B` 参数，该模型的参数量推测在 `600-700B` 之间。然而，没有迹象表明 Max 版本会开源，因为从历史来看，Max 模型从未向公众开放过。评论者对 Max 模型的开源持怀疑态度，指出这些模型通常不向公众发布。人们更倾向于可以在消费级硬件上运行的小型模型，并认为 Max 模型应该保持私有以支持公司的营收。

    - 关于 Qwen 3.6 Max 模型的参数量存在猜测，一名用户建议其参数量可能在 `600-700B` 之间，因为 Qwen Plus 模型为 `397B`。这表明其复杂性和潜在能力有了显著提升，与其 `52` 分的高 AA-Intelligence Index 评分相符。
    - 一位用户强调了不开源 Max 模型背后的商业策略，认为这些模型是公司的营收引擎。这暗示公司优先考虑其最先进模型的商业化变现，同时可能提供较小的模型以实现更广泛的普及。
    - 围绕开源的讨论显示，最有可能权重开源（open-weighted）的最大模型可能是 `122B` 模型，因为公司已经停止对 `397B` 的 Plus 模型进行权重开源。这表明了一个限制访问其最先进模型的战略决策，可能是为了维持竞争优势。

  - **[从 Opus 4.7 切换到 Qwen-35B-A3B](https://www.reddit.com/r/LocalLLaMA/comments/1spz0ck/switching_from_opus_47_to_qwen35ba3b/)** (热度: 772): **用户正在考虑将编程 Agent 的驱动模型从 **Opus 4.7** 切换到 **Qwen-35B-A3B**，具体运行在 `M5 Max 128GB` 的配置上。用户承认 Opus 在复杂推理任务中可能具有优势，但质疑 Qwen-35B-A3B 是否足以胜任大多数任务。帖子指出 Qwen-35B-A3B 已经替代了用户约 `95%` 的调用，表明其具有很强的功能性，尽管在复杂场景下可能无法完全达到 Opus 的水平。** 一位评论者认为，如果用户习惯了 Opus 的能力，Qwen-35B-A3B 可能无法达到预期；而另一位评论者则暗示用户的任务可能不需要 Opus 的高级功能。第三条评论指出，Qwen-35B-A3B 可以处理大多数任务，但在某些领域可能逊色于 Opus。

    - **Flinchie76** 讨论了使用 Opus 4.7 和 Qwen-35B-A3B 之间的权衡，强调虽然 Opus 可以快速生成大量代码，但往往会导致复杂且难以理解的架构。相比之下，使用像 Qwen-35B-A3B 这样能力稍弱的模型可以对代码有更多的控制和理解，因为它要求用户思考整个过程并仔细检查更改，从而对最终产品有更好的掌控力。
    - **Borkato** 指出 Qwen-35B-A3B 已经替代了他们约 95% 的调用，这表明虽然它在能力上可能不如 Opus，但对于许多任务来说仍然非常实用。这暗示 Qwen-35B-A3B 可以处理用户通常依赖 Opus 的大部分任务，尽管存在一些局限性。
    - **Thump604** 提到了运行 122B 模型的可能性，但明确表示它无法达到 Opus 4.7 的水平。这表明虽然有更大的模型可用，但它们可能无法完全复制 Opus 的性能或能力，显示出用户从 Opus 过渡到其他模型时潜在的功能差距。

- **[我在我的 mbp m5 max 128gb 上通过 OpenCode 运行 8 bit 量化和 64k 上下文的 qwen3.6-35b-a3b，效果和 claude 一样好](https://www.reddit.com/r/LocalLLaMA/comments/1spdvpo/im_running_qwen3635ba3b_with_8_bit_quant_and_64k/)** (互动量: 1239): **用户报告称在拥有 `128GB RAM` 的 MacBook Pro M5 Max 上，使用 **OpenCode** 运行具有 `8-bit 量化`和 `64k 上下文`的 `qwen3.6-35b-a3b` 模型。他们声称该模型在速度和处理复杂任务（如调试 Android 应用中的序列化问题）方面与 **Claude** 相当。该模型因其极快的响应速度和对长研究任务的高效处理而受到关注，使其成为云端模型的一个可行替代方案。** 评论者强调了该模型在 `5090` 等高性能硬件上的速度优势，以及其对大上下文的高效处理能力，甚至暗示它可以有效处理高达 `256k 上下文`。然而，对于它是否完全等同于 Claude 仍存在一些质疑，尽管它被公认为是一款强大的本地模型。

    - **cosmicnag** 强调了 Qwen 3.6-35b-a3b 模型的性能，并指出在 `5090` GPU 上，其速度是云端模型无法比拟的。他们提到尚未尝试 `NVFP4`，暗示性能仍有进一步提升的潜力。
    - **H_DANILO** 指出 Qwen 模型可以高效处理高达 `256k` 的上下文，并强调该模型的上下文处理成本*非常低廉*。这表明在需要大量上下文管理的任务中具有显著优势。
    - **Krillian58** 分享了截然不同的体验，表示从 Opus 切换到 Qwen 3.6 后，发现它在处理任务时表现明显变差。他们推测这可能是因为模型在*处理 Opus 遗留的问题*，表明在模型转换或适配方面可能存在潜在问题。


### 3. Local LLMs 与离线 AI 应用

  - **[所以... 我应该用本地 LLMs 学点什么？](https://www.reddit.com/r/LocalLLM/comments/1spujo7/so_what_am_i_supposed_to_learn_with_local_llms/)** (互动量: 112): **该帖子讨论了使用本地 LLMs 的挑战和潜力，特别是在 16GB M4 Mac Mini 等受限硬件上。用户尝试了 OpenClaw 以及由 Opus 蒸馏的 `gemma e4b q4` 等本地模型，并将其与 Apple 的 OCR 和视觉功能集成。尽管在设置 cron 任务和基础任务方面取得了初步成功，用户仍对本地 LLMs 相比于 Claude Code 等云端方案的实际效用表示怀疑。帖子强调了本地 LLMs 随硬件提升而进步的潜力，以及理解模型上下文窗口和隐私优势的重要性。建议用户探索更小的模型，并考虑为更高级的应用做技术储备。** 评论者强调了本地 LLMs 在隐私、成本效益以及运行不受限模型方面的优势。他们建议将本地 LLMs 用于邮件摘要、文档分析和个人知识管理等任务。一些人建议从 OpenClaw 切换到 Hermes Agent 以获得更流畅的体验，并强调了建立远程交互渠道和自动化常规任务的重要性。

    - **Local LLMs** 在隐私和数据控制方面具有显著优势。在本地运行 Qwen 3.5 或 3.6 等模型可以让用户避免将敏感信息发送给大公司，这对于保护隐私至关重要。此外，随着硬件变得更便宜且模型更高效，本地 LLMs 可能会比云端方案更具成本效益且速度更快，从而提供技术储备优势。
    - 相比 OpenClaw，更推荐使用 **Hermes Agent**，因为它具有更低的 Token 开销和更好的设计。本地 LLMs 可以与 Telegram 或 Slack 等通信平台集成，以实现邮件摘要、创建知识库以及对 PDF 进行 OCR 等自动化任务。这种配置允许无缝的任务管理，且不受云端模型 Token 使用限制的影响。
    - 在 16GB RAM 等受限硬件上运行 **Local LLMs** 虽然具有挑战性，但具有独特的优势。它允许在不暴露于互联网的情况下安全处理敏感数据，这对于隐私要求极高的任务至关重要。虽然 Qwen 3.5 9b 等模型可以在此类配置上运行，但真正的优势在于自动化那些对云端 API 来说过于敏感的任务，尽管存在硬件限制。

- **[llama.cpp speculative checkpointing was merged](https://www.reddit.com/r/LocalLLaMA/comments/1sprdm8/llamacpp_speculative_checkpointing_was_merged/)** (热度: 417): **`llama.cpp` 项目已合并了 speculative checkpointing 功能，根据任务和重复模式的不同，该功能可以带来不同程度的加速。对于编程任务，用户报告使用 `--spec-type ngram-mod`、`--spec-ngram-size-n 24`、`--draft-min 48` 和 `--draft-max 64` 等参数时，加速效果在 `0% 到 50%` 之间。该功能是持续优化的一部分，还包括 DFlash 和 SYCL 支持等其他增强功能，这些功能已显示出 `17% 到 50%` 的速度提升。这些更新表明，随着软件和驱动程序的完善，性能将继续提高（[来源](https://github.com/ggml-org/llama.cpp/pull/19493)）。** 评论者对这些改进持乐观态度，并指出虽然一些用户对 B70 的初始性能感到失望，但预计后续更新将显著提升性能。社区鼓励大家在进一步优化实施期间保持耐心。

    - `llama.cpp` 中的 speculative checkpointing 功能已经合并，预计将显著增强性能。值得注意的是，有几个相关的 GitHub Pull Request (PR) 对性能提升做出了贡献：[PR #22066](https://github.com/ggml-org/llama.cpp/pull/22066) 报告在 SYCL 上速度提升了 `17 到 50%`，[PR #21845](https://github.com/ggml-org/llama.cpp/pull/21845) 声称速度提升高达 `50%`，[PR #21527](https://github.com/ggml-org/llama.cpp/pull/21527) 也提到了 `50%` 的加速。这些改进表明，随着软件和驱动程序的不断演进，最初对 B70 性能的担忧可能还为时过早。
    - `llama.cpp` 中 self-speculative decoding 的实现使其可以用于 Qwen3.5 和 3.6 等模型。可以通过调整参数来激活此功能，从而可能实现更高效的 Token 生成。然而，实际的性能增益可能会有所不同，正如幽默的评论所指出的，它可能不像预期的那样快（'not BRRRRRR'），但仍然提供了一些 'free tokens'。
    - speculative decoding 接受率的差异受 `ngram-mod` 匹配机制的影响。具有重复模式的代码库（如 TypeScript 或 Java 中的代码）可能会体验到更高的接受率（高达 `50%`），而独特的逻辑序列接受率则较低。参数 `--spec-ngram-size-n 24` 被认为是激进的，因为它需要 `24 tokens` 的上下文来进行模式匹配。在代码/散文混合任务中，尝试更小的值（例如 `8-12`）可能会因为增加模式匹配的可能性而提高性能，尽管 draft runs 会缩短。

## 技术性较低的 AI Subreddit 汇总

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude 设计与使用创新

  - **[这不可能是真的。我不敢相信我的眼睛](https://www.reddit.com/r/ClaudeAI/comments/1sqpb2f/this_cannot_be_real_i_cannot_believe_my_eyes/)** (热度: 1527): **Reddit 帖子中的图片是一个名为 "Air Roster" 的应用程序的功能发布轮播图，展示了各种功能，如月份映射功能、月份选择器界面、测地线地图可视化以及薪资相关的统计数据。设计采用深色主题，配以蓝色和白色文字，旨在追求现代美感。该帖子讨论了设计工具的民主化，将 Canva 对设计门槛的影响与新 AI 工具在减少对专业设计技能的需求方面的潜力进行了比较，允许用户专注于内容而非工具的熟练程度。** 评论反映了对设计质量的怀疑，一些用户批评了用户界面 (UI) 和用户体验 (UX)，另一些用户则质疑赞美的严肃性，暗示这可能是讽刺。

    - Capable_Ad1259 强调了基于专业背景对 UI/UX 设计认知的差异。Backend/API/AI/ML 开发者可能会因为其技术复杂性而觉得设计令人印象深刻，而 UI 开发者和设计师可能会批评其“粗糙”。这突显了从后端工程向设计转型的挑战，强调了掌握设计技能需要投入的时间和努力。

- **[Claude Design is Amazing! We're cooked!](https://www.reddit.com/r/ClaudeAI/comments/1squwsy/claude_design_is_amazing_were_cooked/)** (Activity: 576): **该贴讨论了向 AI 模型 **Claude Design** 提出的一个请求：创建一个避免典型 AI 生成内容（即 "AI-slop"）的操作系统。用户声称 Claude Design 仅通过单次尝试就成功生成了独特的 OS 设计，凸显了其强大的能力。然而，该贴缺乏关于 OS 设计的具体技术细节，例如架构、功能或 Benchmarks，而这些对于技术评估至关重要。** 一位评论者对 AI 创建完整操作系统的可行性提出质疑，对该声明的真实性表示怀疑。另一条评论则怀旧地提到该设计与 **Windows 98** 相似，暗示这更像是一种复古美学，而非现代技术创新。

- **[Claude Design is Incredible...](https://www.reddit.com/r/ClaudeAI/comments/1spxi2f/claude_design_is_incredible/)** (Activity: 1689): **该贴讨论了使用 **Claude Design** 进行快速 UI 重新设计的过程，强调了它以极少的工作量快速转换应用的能力。作者指出，虽然重新设计可能看起来与其他使用 Claude 制作的应用相似，但对于个人使用来说非常有效。该项目目前已开源并发布在 [GitHub](https://github.com/AmmarSaleh50/study-dashboard) 上。作者建议，通过具体的设计 Prompt，Claude 可以产生独特的结果，但通用的 Prompt 会导致默认设计。** 评论者普遍认为，使用 Claude 设计的应用往往看起来大同小异，其中一位指出重新设计导致了不太美观的字体选择。另一位评论者认为，这种统一性可能会导致在不久的将来许多应用都拥有相同的设计。

    - Chupa-Skrull 强调，Claude Design 的主要优势在于它能够暴露各种属性的“调节旋钮”（knobs），允许用户通过调整那些他们可能不知道该如何通过 Prompt 描述的参数来优化工作流程。尽管其底层能力与其它模型数月来提供的内容相似，但这一功能显著加快了设计过程。
    - One-Cheesecake-9353 指出，虽然 Claude Design 可能适用于个人项目，但对于面向大众消费的项目来说，它引入了过多的认知负荷。这表明设计复杂性或 UI 界面对于更广泛的受众来说可能不够直观，可能会对用户体验产生负面影响。
    - Toxic-slop 和 disky_wude 都注意到，Claude 生成的应用往往看起来很相似，这表明设计输出缺乏多样性。这可能是 Claude 设计算法的一个局限，导致风格重复，并可能降低使用该工具开发的应用的独特性。

- **[I didn't realise Claude could build actual Word docs and Excel files. Cancelled three subscriptions in the same week.](https://www.reddit.com/r/PromptEngineering/comments/1spmwkg/i_didnt_realise_claude_could_build_actual_word/)** (Activity: 422): **该贴强调了 **Claude 直接根据 Prompt 生成格式完整的 Word (.docx)、Excel (.xlsx) 和 PowerPoint (.pptx) 文件**的能力，从而消除了对独立文档创建软件的需求。用户可以要求特定的格式，如标题、项目符号和专业字体，且 Claude 可以处理复杂的 Excel 功能，如公式和条件格式。该工具还支持在保持原有格式的情况下编辑现有文档。这种能力允许用户绕过传统的文档创建工具，转而关注内容创作，而非格式排版和基础设施。** 评论者指出了更改文档 metadata 以反映正确作者的重要性，并分享了使用 Claude 修复从 PDF 转换为 Word 的文档中复杂格式问题的经验。他们还赞扬了 Claude 的迭代编辑能力，允许无缝的内容更新和修改。

- Rencauchao 强调了使用 Claude 生成 Word 文档时的一个关键步骤：用户在分享前应修改“作者（author）”和“备注（comments）”元数据，以反映自己的信息，因为这些字段可能会暴露文档是由 Claude 生成的。这对于维护作者身份的完整性和隐私非常重要。
- sceez 分享了一个实际用例，其中 Claude 被用于解决 Word 文档转换为 PDF 后又转回来的格式问题。该过程涉及与 Claude 的迭代交互，成功恢复了文档的格式，展示了 Claude 处理复杂文档编辑任务的能力。
- 5aur1an 建议了一种个性化 Claude 输出的方法，即通过训练它模仿用户的写作风格。这包括分析示例文档的风格元素，然后通过对不符合用户风格的特定词语或短语提供反馈，迭代地优化生成的内容。随着时间的推移，这种方法可以增强生成内容的相关性和个性化程度。

### 2. DeepSeek and V4 Developments

- **[They said it's next week 🤞](https://www.reddit.com/r/DeepSeek/comments/1sppz7q/they_said_its_next_week/)** (Activity: 328): **该图片是 **Yifan Zhang** 在社交媒体上发布的帖子截图，讨论了即将到来的与 AI 模型相关的技术更新，特别提到了 “Sparse MQA”、“Fused MoE Mega Kernel” 和 “Hyper-connections” 等术语。这些术语暗示了 AI 模型架构的进步，可能会提高效率和性能。“V4，下周”的提及暗示了预期的发布或更新，可能与新版本的 AI 模型或框架有关。该帖子已被编辑并显示出极高的参与度，表明了社区的关注。** 评论者对发布时间表表示怀疑，指出自 1 月份以来就一直有类似的承诺。然而，人们也感到了一种重新燃起的乐观和兴奋，一些用户对这次更新的兴趣超过了其他最近的 AI 进展。


- **[To those waiting for V4](https://www.reddit.com/r/DeepSeek/comments/1sq0jcz/to_those_waiting_for_v4/)** (Activity: 221): ****High-Flyer** 是科技领域的一个独特实体，它作为一个大型量化对冲基金运作，而不是传统的科技公司。这种结构允许他们在没有产生直接收入或取悦风险投资者的典型压力下开发像 V4 这样的 AI 模型。他们的方法是由内部指标而非外部市场周期驱动的，这解释了为什么缺乏市场炒作以及提供低成本的 API 服务。据传该公司通过战略性的财务手段（如做空 Nvidia）为其 AI 部门提供资金，突显了其财务独立性和战略重点。** 评论者辩论了 High-Flyer 开发 AI 的初衷，认为尽管他们财务独立，但必须通过创新来保持竞争力和相关性。也有人对人才留存以及为了确保长期成功而可能需要上市的潜在需求表示担忧。

    - WHY_DO_I_SHOUT 强调，该对冲基金缺乏市场炒作和低成本 API 访问是由于其财务独立性，因为他们不依赖于模型的直接收入。这表明他们的主要目标不是通过模型本身变现，而可能是利用它来获得内部优势或进行战略定位。
    - Weird-Pollution-6251 指出，该模型的用户界面以及缺乏与其他工具的集成表明它更多是一个演示，而不是一个成熟的产品。这暗示该对冲基金的重点可能在于展示能力，而不是创建一个市场就绪的产品，这与其不需要从模型中获得直接收入的财务战略相一致。
    - Puzzleheaded-Drama-8 推测该对冲基金可能会从模型发布炒作引起的市场波动中受益。这暗示了对模型的战略利用，旨在影响市场状况，并可能通过针对这些波动的交易创造获利机会。

### 3. Kimi 2.6 与 AI 模型基准测试

  - **[Kimi 2.6 已发布](https://www.reddit.com/r/singularity/comments/1sqsvrt/kimi_26_has_been_released/)** (热度: 605): **该图片展示了一张性能对比图表，突出了 Kimi K2.6 在通用 Agent、编程和视觉 Agent 等各项任务中，与 GPT-5.4、Claude Opus 4.6 和 Gemini 3.1 Pro 等其他 AI 模型相比具有竞争力的表现。Kimi K2.6 因其对开源金融撮合引擎的自主重构而特别受到关注，通过自主迭代优化策略和修改代码，实现了显著的性能提升。这展示了该模型在系统架构和优化方面的先进能力，实现了 `185%` 的中值吞吐量增长和 `133%` 的性能吞吐量提升。** 评论者对 **Kimi K2.6** 的开源特性及其自主优化复杂系统的能力印象深刻，强调了其在实际应用中的潜力。

    - Kimi K2.6 自主优化了 exchange-core（一个开源金融撮合引擎），通过迭代 12 种优化策略并进行了 1,000 多次工具调用（tool calls），修改了 4,000 多行代码。该模型分析了 CPU 和分配火焰图（flame graphs）以识别瓶颈，并重新配置了核心线程拓扑，实现了中值吞吐量 185% 的增长和性能吞吐量 133% 的提升，展示了开源 AI 能力的重大进展。
    - 一位用户对 Kimi 2.5 是否被“刷榜（benchmaxed）”表示怀疑，并指出与 Claude、GLM 5.1、GPT、Gemini 3.1 和 Qwen 等其他模型相比，它在设计和网页开发任务中表现出色。他们强调 Kimi 在创建 PowerPoint 演示文稿、PDF 和网站方面具有无与伦比的性能，认为其设计能力远超竞争对手，如果 Kimi 2.6 确实开源，这一点将尤其令人印象深刻。
    - 讨论中包括了关于 Kimi 2.6 是否真正开源的询问，反映了社区对先进 AI 模型的可访问性和透明度的关注。用户将 Kimi 的性能与其他模型进行了积极对比，强调了其卓越的设计任务处理能力，如果该模型保持开源，这可能是一个显著的优势。

  - **[Opus 4.7 vs 4.6 经过 3 天真实编程后的实测对比 - 来自我实际会话的并排对比](https://www.reddit.com/r/ClaudeCode/comments/1spxtut/opus_47_vs_46_after_3_days_of_real_coding_side_by/)** (热度: 696): **该图片基于为期三天的真实编程会话，提供了 Opus 4.6 和 Opus 4.7 的详细并排对比。重点突出了诸如一次性成功率（one-shot rate）、重试率和单次调用成本等关键指标，显示 Opus 4.6 在一次性成功率（`83.8%` vs `74.5%`）和成本效率（每次调用 `$0.112` vs `$0.185`）方面通常表现更好。然而，Opus 4.7 每次调用生成的输出更多（`800 tokens` vs `372 tokens`），因此价格更高。分析还指出，Opus 4.7 在每轮对话中使用的工具较少，且较少委托给子 Agent，这表明其在操作风格上可能存在差异，或者是样本量限制。该帖子强调这些发现是初步的，且基于有限的数据，随着收集到更多数据，情况可能会发生变化。** 评论者赞赏这种详细分析，并建议 Opus 4.7 可能需要进行 Prompt 调整。还有关于积极推广 Opus 4.7 背后潜在动机的讨论，暗示了成本方面的考量。

    - phil_thrasher 提出了一个关键点，即从 Opus 4.6 过渡到 4.7 时需要进行 Prompt 调整，并建议测试框架（harness）可能需要更改以针对新版本优化性能。这强调了调整测试框架以适应 AI 模型更新的重要性，而开发团队可能尚未完全解决这一问题。
    - SovietRabotyaga 指出了“总成本字段”在理解 Anthropic 积极推广 Opus 4.7 策略中的重要性。这表明经济因素可能会影响推行新版本的决策，从而可能影响模型更新和部署背后的决策过程。
    - thewormbird 回顾了历史模型更新，指出像 3.7 这样的中间版本在他们的工作流中不如 4.0 这样的大版本有效。这引发了关于版本控制策略以及增量更新是否能带来实质性改进的问题，并建议用户从等待 Opus/Sonnet 5 等大版本发布中获益更多。


# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快就会发布新的 AINews。感谢读到这里，这是一段美好的历程。