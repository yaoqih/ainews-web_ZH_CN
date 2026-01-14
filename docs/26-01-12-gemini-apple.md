---
companies:
- apple
- google
- openai
- anthropic
- deepseek
date: '2026-01-12T05:44:39.731046Z'
description: '以下是为您翻译的内容：


  **苹果**已决定采用**谷歌的 Gemini 模型**和云技术为 Siri 提供支持，这标志着双方达成了重大合作，同时也令最初与苹果合作的 **OpenAI**
  遭遇挫折。**Anthropic** 推出了“Cowork”，这是 Claude 编程功能的产品预览，引发了关于“LLM 操作系统（LLM OS）”的讨论。**OpenAI**
  推出了 **ChatGPT Health** 并收购了 **Torch**，以扩展其在医疗 AI 领域的布局。**DeepSeek** 发布了 **Engram**，这是一种新型条件记忆模块，可实现针对静态模式的
  O(1) 查找式记忆，提升了长上下文处理能力，并提供硬件友好型优化以高效扩展知识容量。Engram 被定位为下一代稀疏模型的关键建模原语，社区目前正对其架构优势和实际影响展开讨论。'
id: MjAyNi0w
models:
- gemini
- claude
- chatgpt
- engram
people: []
title: 苹果公司选择谷歌的 Gemini 为下一代 Siri 提供支持。
topics:
- conditional-memory
- long-context
- hashing
- memory-optimization
- transformers
- model-scaling
- sparsity
- hardware-optimization
- model-architecture
- ai-healthcare
- model-optimization
---

**Apple 终于妥协了。**

> 2026年1月12日至1月13日的 AI 新闻。我们为你查阅了 12 个 subreddit、[**544** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)以及 **24** 个 Discord（**204** 个频道，1541 条消息）。预计节省阅读时间（按 200wpm 计算）：**157 分钟**。**我们的新网站**现已上线，支持全文元数据搜索，并以极具氛围感的设计呈现往期所有内容。请访问 https://news.smol.ai/ 查看完整的新闻分类，并在 [@smol_ai](https://x.com/Smol_AI) 上向我们提供反馈！

比 Apple 举步维艰的 AI 部门更值得同情的，只有那些至今未等来承诺中 Siri 更新的 Apple 用户。在谈论了一年多自研训练模型后，Apple [宣布](https://www.wcnc.com/article/news/nation-world/apple-google-gemini-siri-ai-features/507-575faa99-217e-498d-8f34-5455759113f8)将选择 Google 的 Gemini 来驱动 Siri，而安全层仍由 Apple Private Cloud Compute 提供。这对 Google 来说是一次漂亮的胜利，而对 OpenAI 而言则是相对的损失。Apple 最初在 AI 发布时与其合作，但关于 OpenAI 将在今年出货消费级硬件设备的传闻愈演愈烈，这可能是影响该决策的一个因素。

---

# AI Twitter 简报

**热门推文（按互动量排序）**

- **Apple ↔ Google AI 合作伙伴关系**：一份联合声明称，“下一代 Apple Foundation Models”将基于 **Google 的 Gemini 模型和云技术**，为未来的 Apple Intelligence 功能（包括“更具个性化的 Siri”）提供支持，同时保持 Apple 的隐私立场 ([推文](https://twitter.com/NewsFromGoogle/status/2010760810751017017))。
- **Anthropic 发布 “Cowork”**：Claude 的 “Claude Code for the rest of your work” 产品预览引发了大量的关注，以及关于 “LLM OS” 发展轨迹的讨论 ([公告](https://twitter.com/claudeai/status/2010805682434666759), [背景](https://twitter.com/bcherny/status/2010809450844831752))。
- **OpenAI 进军医疗领域**：OpenAI 推出了 **ChatGPT Health**（一个具有独立记忆的专用空间），并宣布收购 **Torch** ([ChatGPT Health](https://twitter.com/OpenAI/status/2010764845432590469), [收购](https://twitter.com/OpenAI/status/2010813780671021106))。
- **DeepSeek “Engram”**：一条热门技术推文强调了 DeepSeek 新的条件记忆/查找模块，将其视为一种新的“稀疏性维度 (axis of sparsity)” ([推文链接](https://twitter.com/scaling01/status/2010748516788777445))。

---

**DeepSeek 的 Engram：作为新稀疏原语的条件记忆**

- **Engram = 针对静态模式的 O(1) 查找式内存**：DeepSeek 推出了“**通过可扩展查找实现的条件内存**”（Engram），增加了一种哈希处理的 **n‑gram embedding** 内存，模型可以对其进行查询并门控（gate）到表示中。其核心思路是：Transformer 在前向传播中消耗计算量来“重建记忆模式”；Engram 则卸载了这种“静态检索”，使骨干网络的容量能够专注于“有效深度”或推理（[@scaling01](https://twitter.com/scaling01/status/2010748516788777445)）。后续笔记称这也有助于 **long-context**（[推文](https://twitter.com/scaling01/status/2010748710653980989)）。
- **为什么它对系统和扩展（Scaling）很重要**：多种观点汇聚于此——确定性的哈希/查找支持硬件友好型优化（预取、内存移动），并将扩展约束从 **HBM 限制的参数**中转移出来，这暗示了一条在保持 FLOPs 稳定的同时，廉价增长“知识容量”的实用路径（[@tokenbender 的评论与总结](https://twitter.com/tokenbender/status/2010791813964296558), [@teortaxesTex](https://twitter.com/teortaxesTex/status/2010763425849430184)）。其中一句关键引用——“**条件内存是下一代稀疏模型不可或缺的建模基元**”——被多次提及作为核心论点（[引用推文](https://twitter.com/scaling01/status/2010750095923499489)）。
- **与先前技术的关系 (Gemma-3n / N-Grammer / PLE / OTT)**：工程师们将 Engram 与早期的“在 embedding 处注入 n-gram 信息”的方法（Over-Tokenized Transformers, Per-Layer Embeddings, N‑Grammer）进行了对比，认为 Engram 的不同之处在于使内存成为一种**主动的、按层寻址的操作**，而非被动的早期注入（[对比](https://twitter.com/gm8xx8/status/2010830166076071970), [@_arohan_](https://twitter.com/_arohan_/status/2010760026689060918)）。也有人反驳称相关技术在 “Gemma-3n” 中已经存在，争论焦点集中在叙事框架（边缘效率 vs 前沿扩展）以及架构“美学”与端到端能力导向的对比上（[讨论](https://twitter.com/teortaxesTex/status/2010775191320699084)）。
- **社区评价：有前景但带有“系统思维”且可能较为脆弱**：一份详细的批评质疑“始终获取 + 门控”是否是比自适应计算（adaptive compute）更好的抽象，并担心 OOD（分布外）混合或优化复杂性，同时承认在同等预算下的增益（iso-budget gains）似乎是真实存在的，尽管增幅适中（该评论者认为约 3–5%）（[@tokenbender](https://twitter.com/tokenbender/status/2010791813964296558)）。

---

**长上下文与内存研究：DroPE、Agent 内存与测试时训练**

- **DroPE (Sakana AI)：通过“丢弃”位置嵌入扩展上下文**：Sakana 的 “**DroPE**” 提出使用 RoPE 保证收敛，然后移除位置编码以避免上下文扩展过程中的语义扭曲。其声称这是在 NoPE 训练困难（梯度消失）与 RoPE 缩放导致低频失真之间的一种有原则的折中方案（[论文推文](https://twitter.com/SakanaAILabs/status/2010660969719165133)，[日文解析](https://twitter.com/iwiwi/status/2010700629744746934)）。他们还发布了一个参考训练器实现（[代码仓库](https://twitter.com/SakanaAILabs/status/2010738878727217595)）。
- **测试时训练作为“内存”：TTT‑E2E**：NVIDIA/斯坦福/Astera 推动“**端到端测试时训练**”（End-to-End Test-Time Training），即模型在部署时对提供的上下文继续进行 next-token 训练，有效地将显著上下文压缩进权重中，从而减少长序列对大型 KV caches 的依赖。多篇帖子将其描述为“LLM 内存的新时代”，以及通往亚二次复杂度长序列建模的路径（[NVIDIAAIDev](https://twitter.com/NVIDIAAIDev/status/2010773774849724858), [@karansdalal](https://twitter.com/karansdalal/status/2010774529120092481), [@jure](https://twitter.com/jure/status/2010790789627125877)）。
- **Agent 内存框架趋向于“策略集成内存操作”**：
  - **AgeMem**：将长期 + 短期内存视为单个可学习策略，具有类似工具的操作（ADD/UPDATE/DELETE + RETRIEVE/SUMMARY/FILTER），通过阶段性 RL 策略和分步 GRPO 进行训练。报告显示在 Qwen2.5‑7B 上比 Mem0 提升了 **+13%**，在 Qwen3‑4B 上差距更大（[@omarsar0](https://twitter.com/omarsar0/status/2010712137933730234)）。
  - **SimpleMem**：专注于“语义无损压缩”+ 整合 + 查询感知检索；声称在 **LoCoMo 上获得 43.24 F1**，而 Mem0 为 **34.20**，且单次查询消耗的 Token 降低了 **30 倍**（531 vs 16,910）（[@dair_ai](https://twitter.com/dair_ai/status/2010720188686348593)）。

---

**Agent 与开发者工具：评估基础设施、CLI 优先工作流以及 “LLM OS” 动向**

- **Anthropic “Cowork”：面向非编程工作的 Claude Code 体验**：Anthropic 将 Cowork 定位为“适用于你其余工作的 Claude Code”，将 Agent 与浏览器自动化、连接器和沙箱执行环境捆绑在一起。此次发布引发了一波“LLM OS”评论和“Agent 化知识工作”的讨论 ([发布详情](https://twitter.com/claudeai/status/2010805682434666759), [产品细节](https://twitter.com/bcherny/status/2010809450844831752), [“LLM OS” 观点](https://twitter.com/skirano/status/2010833788591300642))。
- **内部编程 Agent 走向主流（Ramp “Inspect”）**：Ramp 报告称，其内部 Agent 在一周内编写了 **30% 已合并的前后端 PR**，完全在基于开放工具（opencode, Modal, Cloudflare）的云端运行。他们开源了构建类似系统的“蓝图/规范” ([@zachbruggeman](https://twitter.com/zachbruggeman/status/2010728444771074493), [构建文章](https://twitter.com/rahulgs/status/2010734253538267197))。
- **Agentic 评估 = 基础设施问题（大规模 AI21 SWE-bench）**：AI21 表示他们运行了 **20 万次以上** 的 SWE-bench，最大的教训是基础设施：为每个实例（repo+deps+MCP server）进行配置并重用；将生成与评估分离，以便在不重新生成 Token 的情况下重试失败的测试。他们声称失败率从 **30% 降至 0**，repo 下载量从 **8,000+ 降至 500** ([推文起始](https://twitter.com/AI21Labs/status/2010738309681823992))。
- **CLI 作为低 Token 消耗的 Agent 接口**：Hugging Face 呼应了 “Bash is all you need” 的逻辑，并发布了可组合且易于发现的 Hub CLI，让编程 Agent 能够以较低的上下文消耗探索模型、数据集和 Spaces ([@hanouticelina](https://twitter.com/hanouticelina/status/2010664329545224588))。
- **用于百万级 Token 工作流的递归语言模型 (RLM)**：RLM 提议递归地对 Prompt 进行分块并汇总结果，其轨迹可用于 RL/蒸馏；该技术正被集成到 OpenEnv 中 ([概述](https://twitter.com/SergioPaniego/status/2010765550012735896), [OpenEnv 笔记](https://twitter.com/SergioPaniego/status/2010765552952713241))。

---

**巨头平台转型：Apple↔Gemini、Google “Agent 商业化” 以及 API 原语**

- **Apple 在 Siri / “Apple Foundation Models” 上押注 Gemini**：联合声明明确将“下一代 Apple Foundation Models”与 Gemini 模型和云技术挂钩，同时强调端侧 + Private Cloud Compute 以及隐私标准 ([声明](https://twitter.com/NewsFromGoogle/status/2010760810751017017))。评论强调了其战略逻辑：Gemini 的多模态领先地位，以及与 OpenAI 设备野心之间的竞争态势 ([分析](https://twitter.com/Yuchenj_UW/status/2010777804246565175))。
- **Google 推进 Agent 商业化轨道**：Google 发布了**通用商业协议 (Universal Commerce Protocol, UCP)** 以及一系列功能，如 AI 模式 / Gemini 中的直接结账、“商家 Agent” 与零售商聊天，以及 “Direct Offers” 试点——即将模型作为购物界面和交易发起者 ([@Google](https://twitter.com/Google/status/2010744570108137524))。
- **Gemini API 输入规模扩展**：Gemini API 将内联文件限制从 **20MB 提升至 100MB**，增加了来自 **Google Cloud Storage (GCS)** 存储桶的原生摄取功能，并支持签名 URL（支持其他云）。其他帖子详细说明了最大容量限制，如注册的 GCS 文件为 **2GB**，外部签名 URL 为 **100MB**，并列出了支持的文档格式 ([@osanseviero](https://twitter.com/osanseviero/status/2010764447988461634), [@_philschmid](https://twitter.com/_philschmid/status/2010765230134215037))。

---

**OpenAI 进军医疗领域：产品边界、收购与隐私态势**

- **ChatGPT Health = 独立的记忆域**：OpenAI 推出 “ChatGPT Health”，医疗聊天、文件和记忆存储在专用空间中；健康信息“绝不会流入常规聊天”，用户可以查看或删除医疗记忆 ([公告](https://twitter.com/OpenAI/status/2010764845432590469))。
- **收购 Torch Health**：OpenAI 收购了 Torch，该公司旨在统一实验室结果、药物和就诊记录，此举被定位为加速 ChatGPT Health 的功能开发 ([OpenAI](https://twitter.com/OpenAI/status/2010813780671021106), [Torch 创始人](https://twitter.com/IlyaAbyzov/status/2010813621022949721))。一份更广泛的回顾称，每周有 **2.3 亿** 用户使用 ChatGPT 发送健康相关信息，并概述了符合 HIPAA 标准的 “ChatGPT for Healthcare” 及其 API 方案 ([回顾](https://twitter.com/thekaransinghal/status/2010878203401843114))。

---

**中国 AI 业务 + 开放生态系统信号（IPO 叙事、采用指标与模型分发）**

- **Zhipu 与 MiniMax IPO “叙事错位”**：一份 ZhihuFrontier 的总结认为，不同的市场表现源于叙事差异：**Zhipu** 被定位为具有长销售周期和高额研发投入的 ToB/ToG 基础设施，而 **MiniMax** 则是具有增长曲线和利润率改善空间的消费者/全球化平台 ([thread](https://twitter.com/ZhihuFrontier/status/2010642118713512174))。
- **开源模型采用指标变得更加严格**：一项 “相对采用指标 (RAM Score)” 旨在根据时间和参数规模对 Hugging Face 的下载量进行归一化处理，认为 **1–9B 模型在原始下载量中占据主导地位**，但 1–9B 和 100B+ 模型之间前 10 名下载量的中位数仅相差约 4 倍。该指标指出 GPT‑OSS 表现异常突出，并暗示大型国产 MoE 发布的增长势头各异 ([@natolambert](https://twitter.com/natolambert/status/2010744476516655274))。
- **GLM‑4.7 分发 + 快速推理**：相关内容包括 GLM‑4.7 通过 Cerebras 在 Hugging Face 上线以及通过 Together AI 提供服务，官方宣称具有 **200K 上下文**和强大的代码基准测试表现（如厂商推文所述） ([@NielsRogge](https://twitter.com/NielsRogge/status/2010686205961146400), [Together](https://twitter.com/togethercompute/status/2010832877626286113))。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. 新的 AI 模型与基准测试发布

  - **[GitHub - deepseek-ai/Engram: Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://www.reddit.com/r/LocalLLaMA/comments/1qb034t/github_deepseekaiengram_conditional_memory_via/)** (热度: 324): **DeepSeek AI** 在其 Engram 仓库中引入了一种新颖的方法，通过可扩展查找（scalable lookup）为 Large Language Models 增强了条件记忆。该方法将静态 N-gram 记忆与动态隐藏状态集成，在传统的 MoE 架构之外提供了一个新的稀疏性维度。该模型展示了在 MoE 和 Engram 之间进行最优容量分配的 U 型缩放法则（scaling law），允许在对性能影响极小的情况下，将嵌入表（embedding tables）高效地卸载到主机内存。在消融实验中使用 Muon 优化器表明了从 AdamW 的转变，这与 Kimi K2 和 GLM 4.5 等模型中看到的趋势一致。欲了解更多详情，请访问 [GitHub 仓库](https://github.com/deepseek-ai/Engram/tree/main)。评论者强调了 Engram 提高单参数性能的潜力，其重要部分可以卸载到 RAM 或 NVMe 而不会产生性能损失。其确定性寻址和 n-gram 嵌入方法因其对模型效率和可扩展性的创新贡献而受到称赞。

    - DeepSeek 团队引入了一个具有条件记忆功能的模型，他们将其描述为“当前稳定的 meta”，并预见其对于下一代稀疏模型至关重要。该模型在消融实验中使用 mHC (𝑀 = 4)，表明了一个稳定的配置。与传统的 MoE 架构相比，Engram 模型有望提高单参数性能，允许模型的显著部分卸载到 RAM 或 NVMe 而不损失性能。这意味着一个 40B A3.8B 的 MoE 模型可能只需要 27B 的权重留在快速显存中，其余部分可以轻松卸载。
    - Engram 模型引入了一种新颖的 n-gram 嵌入方法，在 MoE 之外增加了一个静态记忆作为新的稀疏轴。这允许 O(1) 查找，是一个显著的效率提升。该模型的确定性寻址使得嵌入表可以卸载到主机内存，且推理开销极小。研究人员发现了一个 U 型缩放法则，用以指导 MoE 和 Engram 之间的容量分配，这有助于在复杂推理任务中保留模型深度。
    - 在 Engram 模型的消融实验中使用 Muon 优化器，表明了对传统 AdamW 优化器的背离，这与 Kimi K2 和 GLM 4.5 等模型所见的趋势一致。这种优化器的选择可能会影响下一代模型的训练效率和性能。

  - **[11 Production LLM Serving Engines (vLLM vs TGI vs Ollama)](https://www.reddit.com/r/LocalLLM/comments/1qax6kq/11_production_llm_serving_engines_vllm_vs_tgi_vs/)** (热度: 5): **本文对 11 种生产级 LLM 服务引擎进行了对比分析，包括 **vLLM**、**TGI** 和 **Ollama**，重点关注它们的性能指标、可扩展性和集成能力。文章突出了每种引擎的独特功能和使用场景，强调了它们在有效部署 LLM 中的作用。更多细节请参考 [原文](https://medium.com/@techlatest.net/11-production-llm-serving-engines-vllm-vs-tgi-vs-ollama-162874402840)。** 一条评论建议增加 [Mistral.rs](http://Mistral.rs)、AirLLM 和 Nexa SDK 等引擎，表明 LLM 服务解决方案的领域更加广泛。另一条评论对在生产环境中使用 Ollama 表示疑问，暗示对其成熟度或在生产环境中的适用性持怀疑态度。

    - 讨论强调了文中提到之外的其他 LLM 服务引擎，例如 Mistral.rs、AirLLM、Nexa SDK、TabbyAPI、Exllama、Aphrodite、CoboldCPP、KTransformers、exa 和 TextSynth Server。这些工具被建议作为 vLLM、TGI 和 Ollama 的替代或补充，表明了在生产环境中部署 Large Language Models 的解决方案生态系统非常多样化。

### 2. AI 开发与系统管理

  - **[开发薄弱但擅长系统管理的建议请求](https://www.reddit.com/r/LocalLLM/comments/1qb1p4x/weak_dev_good_sysadmin_needing_advice/)** (活跃度: 19): **用户购买了一台 **Beelink Mini PC, GTR9 Pro**，配备 **AMD Ryzen AI Max+ 395 CPU** 和 `128GB RAM`，用于本地 AI 开发和游戏。他们计划从 Windows 迁移到 Linux，首先从现有脚本进行 PowerShell 模块开发。用户在寻求建议，是该使用原生 Windows 工具，还是从 **Windows Subsystem for Linux (WSL)** 开始。系统的配置表明它能有效处理 Windows 和 Linux 环境，在开发和测试方面提供了灵活性。** 一位评论者建议，用户的系统能够运行 Windows 和 Linux 环境（无论是原生运行还是通过虚拟机），从而在工具选择上提供了灵活性。另一位评论者询问了用户打算开发的具体软件类型，对用户的开发目标表示关注。


  - **[哪款 LLM 是“最佳”编程导师？](https://www.reddit.com/r/LocalLLM/comments/1qb5vry/which_llm_would_be_the_best_coding_tutor/)** (活跃度: 8): **讨论集中在确定最有效的编程教学大语言模型 (LLM)。值得一提的是 **Qwen3 30b coder**，它在配备 `24GB` RAM 的 MacBook Air 上运行效率很高，同时 **OpenAI 的 20b 模型**在类似硬件上的速度也备受关注。推荐使用 **LLM Studio** 来尝试各种模型，这凸显了 2025 年 LLM 能力的重大进步。** 一位评论者认为，所有 LLM 在编程任务上都超越了个人开发者，但强调了学习如何解释和验证这些模型提供的信息的重要性，因为它们有时会提供错误或捏造的数据。

    - **Qwen3 30b coder** 因其性能而受到关注，在 24GB RAM 的 MacBook Air 上运行高效。这表明该模型针对资源受限的环境进行了优化，使其更易于个人使用。此外，**OpenAI 20b** 在相同硬件上的速度也得到了认可，表明这两款模型都适合具有类似配置的用户。
    - 提及 **LLM Studio** 指的是一种允许用户尝试不同语言模型的工具或平台。这对于那些希望比较不同模型的性能和功能的人来说特别有用，尤其是考虑到到 2025 年 LLM 的快速进步。

### 3. 关于 AI 与数据隐私的幽默见解

  - **[It seems like people don’t understand what they are doing?](https://www.reddit.com/r/LocalLLM/comments/1qaxwf5/it_seems_like_people_dont_understand_what_they/)** (热度: 16): **这张图片是一个迷因（meme），幽默地批评了某些人对数据隐私的随意态度，特别是在工作中使用 ‘Claude Code’ 等 AI 工具的背景下。配文暗示了一个讽刺场景：员工为了个人方便（例如早点下班）而不知不觉地危及了雇主的数据安全。这反映了工作场所中更广泛的数据隐私和安全担忧，尤其是随着 AI 工具的日益普及，可能会在无意中泄露敏感信息。** 一位评论者对当前的硬件限制表示沮丧，并认为由于价格上涨，有必要投资更好的设备。另一位评论者分享了在有限的硬件上运行多个工具的挣扎，强调了在 AI 任务中平衡本地和远程计算资源的挑战。

    - 一位用户讨论了在 16GB M1 MacMini 上运行多个工具的局限性，强调了由于内存不足导致本地模型推理面临的挑战。他们提到使用 LM Studio 进行远程推理，但速度很慢，并对使用 OpenRouter API 进行编程感到沮丧，因为这需要剥离可识别的代码，且仍有 25% 的失败率。这反映了硬件限制和当前折中方案低效的更广泛问题。

  - **[Env vars don't work when your agent can read the environment](https://www.reddit.com/r/LocalLLM/comments/1qb0fsg/env_vars_dont_work_when_your_agent_can_read_the/)** (热度: 2): **该帖子讨论了一个安全问题：如果系统上运行的 Agent 可以读取环境，那么环境变量（env vars）就不再安全。这在存储敏感信息（如 API keys 或密码）的场景中尤为重要。问题的产生是因为任何具有足够权限的进程都可以访问这些变量，从而可能导致未经授权的访问或数据泄露。该帖子可能强调了对敏感数据使用替代性安全存储方案的必要性，例如使用秘密管理工具或加密存储，以降低此类风险。** 评论者辩论了使用环境变量处理敏感数据的有效性，一些人认为虽然方便，但在生产环境中不够安全。其他人建议使用 HashiCorp Vault 或 AWS Secrets Manager 等专用秘密管理系统来增强安全性。

    - 讨论的一个关键问题是当 Agent 拥有环境读取权限时环境变量带来的安全风险。这可能导致敏感数据暴露，尤其是当 Agent 受到攻击或具有恶意时。讨论强调了使用安全保险库或秘密管理工具来存储敏感信息，而不是依赖环境变量的重要性。
    - 一位评论者指出，环境变量通常是为了方便而使用，但缺乏强大的安全措施。他们建议使用 HashiCorp Vault 或 AWS Secrets Manager 等工具来安全地管理敏感数据。这些工具提供加密和访问控制，降低了未经授权访问的风险。
    - 另一个技术见解是环境变量可能会无意中暴露在日志或错误消息中。如果应用程序为了调试目的记录环境信息，就可能发生这种情况。建议确保仔细管理日志配置，以避免泄露敏感信息。


## 非技术性 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Apple 与 Google Gemini 的合作

- **[Apple announces that next version of Siri would be powered using Google gemini. Elon Musk does not seem happy about it.](https://www.reddit.com/r/OpenAI/comments/1qb7dg6/apple_announces_that_next_version_of_siri_would/)** (Activity: 903): **苹果**已宣布，下一代 **Siri** 将由 **Google** 的 **Gemini** AI 驱动，这是在将其与 **ChatGPT** 和 **xAI Grok** 等竞争对手进行评估后做出的决定。苹果表示，Google 的技术为其模型提供了最强大的基础，并承诺带来创新的用户体验。这一决定引起了 **Elon Musk** 的批评，他对 Google 日益增长的影响力表示担忧，因为他们已经控制了 Android 和 Chrome。[阅读更多](https://www.wcnc.com/article/news/nation-world/apple-google-gemini-siri-ai-features/507-575faa99-217e-498d-8f34-5455759113f8)。评论者指出，Gemini 可能在盲测（blind AB testing）中表现更优，这可能影响了苹果的决定。人们对马斯克的批评持怀疑态度，一些人指出了他在权力集中问题上表态的讽刺之处。

    - Keeltoodeep 强调 Google 的 Gemini 模型在盲测中表现明显优于对手，这可能是苹果决定将其集成到 Siri 中的关键因素。这表明苹果对 Siri 增强功能的评估过程是高度数据驱动的，专注于经验性能指标。
    - alexx_kidd 对 Elon Musk 对苹果选择的潜在批评不屑一顾，暗示马斯克自己的 AI **Grok** 可能无法与 Google 的 Gemini 性能相匹配。这凸显了 AI 开发中的竞争格局，即 Gemini 所展示的经验性能是一个关键的区分点。
    - trixxyhobbitses 指出了作为科技巨头的 Elon Musk 批评 AI 权力集中的讽刺意味，而他本人也是该行业的重要参与者。这一评论反映了更广泛的行业动态以及主要科技公司在 AI 进步方面的竞争紧张关系。

  - **[It’s official](https://www.reddit.com/r/OpenAI/comments/1qb79py/its_official/)** (Activity: 812): **Google** 和 **苹果**宣布了一项合作伙伴关系，从预计 2026 年 3 月发布的 iOS 26.4 开始，苹果的 Siri 将由 Google 的 **Gemini** AI 模型驱动。此次合作旨在通过集成个人情境理解和改进的应用控制来增强 Siri 的能力。此举可能会改变竞争格局，因为 Google 将其搜索主导地位与 Gemini 以及苹果的分发渠道相结合，而 **OpenAI** 仍依赖于 ChatGPT 和 API，并寄希望于监管或 OEM 的转变。该合作伙伴关系暗示苹果将在私有服务器上托管 Gemini，以维护数据隐私，并标志着对苹果当前 AI 能力的重大升级。评论者对 Siri 目前的能力持怀疑态度，并对改进表示希望。还有关于数据隐私方面的讨论，据报道苹果在私有服务器上托管 Gemini，一位前苹果员工建议苹果优先考虑“足够好”的模型，而不是领先的性能。

    - Minimum_Indication_1 强调苹果可能会在私有服务器上托管 Gemini 模型实例，品牌命名为 Apple Foundation Models，而 Google 无法访问任何数据。这表明其专注于 AI 部署的隐私和控制，符合苹果对用户数据保护的重视。
    - Unique_Carpet1901 是一位前苹果员工，他指出苹果优先考虑以最低成本获取模型权重（weights），而只有 Google 愿意提供。苹果对一个“足够好”的模型感到满意，因为与他们目前的能力相比，这代表了巨大的进步，即使它不是市场上最领先的模型。
    - MEGAT0N 指出 Gemini 模型将在苹果的服务器上运行，不会与网页版或应用版的 Gemini 集成。这表明了一种本地化的部署策略，即 Siri 的增强功能由 Gemini 驱动，但仍与 Google 更广泛的生态系统保持独立。

- **[Apple-Google “超级大脑” 来了。为什么 Siri + Gemini 是我们所熟知的互联网的终结。](https://www.reddit.com/r/GeminiAI/comments/1qb1t6h/the_applegoogle_megabrain_is_here_why_siri_gemini/)** (活跃度: 392): **该图片是来自 “News from Google” 的一条推文截图，宣布 Apple 与 Google 达成合作，将 Google 的 Gemini models 集成到 Apple 的生态系统中，特别是通过更具个性化和智能化的功能来增强 Siri。这一合作伙伴关系旨在利用 Google 的 AI 和云技术，同时保持 Apple 的隐私标准。这次协作标志着跨设备集成 AI 体验的转变，通过从传统的搜索方法转向直接的 AI 驱动回答，可能会改变用户与数字信息的交互方式。** 评论者对这一合作的影响表示怀疑和担忧，一些人认为这可能会颠覆 OpenAI 等现有的 AI 参与者。其他人则讨论了对传统网页内容的影响，认为虽然这可能会减少对充斥广告的网站的依赖，但也可能将信息控制权集中在单一的 AI 系统之下。

    - `phase_distorter41` 的评论强调了搜索引擎的演变，指出其目标一直是简化搜索过程，Google 的 “I'm Feeling Lucky” 按钮就是证明。该用户批评了互联网的现状，指出互联网已充斥着广告内容，掩盖了用户寻找的信息。这反映了一种更广泛的情绪，即像 Siri + Gemini 这样由 AI 驱动的解决方案可以通过绕过传统的、广告密集的网页来简化信息检索。
    - `dbvirago` 提出了关于 Gemini 等 AI 模型对现有网页内容依赖的关键点。该评论质疑了此类模型的可持续性——如果它们有效地取代了那些它们赖以获取数据的网站，情况会如何。这突显了 AI 发展中的一个潜在悖论：虽然 AI 可以高效地总结和提供信息，但它仍然需要动态且持续更新的数据源，而这些数据源传统上来自那些它可能使其过时的网站。

  - **[报告：Apple 选择 Google 的 Gemini 来运行下一代 Siri](https://www.reddit.com/r/GeminiAI/comments/1qayd52/report_apple_chooses_googles_gemini_to_run_next/)** (活跃度: 144): **据 CNBC 报道，Apple 据称正与 Google 合作，将 Google 的 Gemini AI 模型集成到下一代 Siri 中。预计这一合作伙伴关系将利用 Google 的云技术来增强 Apple 的 AI 能力，标志着 AI 领域的一次重大战略举措。该公告发布之际，Google 的市值超过了 Apple，达到了 4 万亿美元以上，突显了这一合作对市场动态的影响。** 评论者指出，Google 在 AI 领域的统治地位自 2025 年底就被预料到了，认为这一合作伙伴关系进一步巩固了 Google 在 AI 市场的强势地位。

    - Apple 选择在 Siri 中使用 Google 的 Gemini，暗示了 AI 战略的重大转变，可能表明 Google 在 AI 领域的领先地位日益增强。此举可能会给其他 AI 模型和公司带来压力，因为 Apple 对 Gemini 的认可可能会影响市场动态和企业合作。
    - 有人猜测亚马逊 Alexa 等其他语音助手的未来，特别是考虑到 Apple 选择与 Google 合作。这一决定可能会导致 AI 助手市场的竞争和创新加剧，因为各公司可能需要重新评估其战略和合作伙伴关系以保持竞争力。
    - 讨论突显了 AI 的竞争格局，Google 似乎处于强势地位。提到 OpenAI 努力争取企业交易，暗示了对 Google 进展的战略应对，表明各大科技公司正在 AI 技术领域争夺主导地位，竞争环境高度激烈。

### 2. Claude Cowork 与 Code 工具

  - **[介绍 Cowork：面向你其余工作的 Claude Code](https://www.reddit.com/r/ClaudeAI/comments/1qb5r3y/introducing_cowork_claude_code_for_the_rest_of/)** (热度: 714): ****Anthropic** 推出了 **Cowork**，这是 Claude 生态系统中的一项功能，允许用户通过授予 Claude 访问其计算机上特定文件夹的权限来执行非技术性任务。这使得 Claude 能够读取、编辑或创建文件，并在用户的监督下执行任务。Cowork 与现有的连接器集成，并可与 Chrome 中的 Claude 配合使用以完成基于浏览器的任务。目前该功能作为研究预览版提供给 macOS 上的 Claude Max 订阅者，其他用户则需进入候补名单。更多详情可以在 [Claude 博客](https://claude.com/blog/cowork-research-preview)上找到。** 评论者指出，Anthropic 的做法受益于对 Microsoft Copilot 等竞争对手的观察，暗示 Cowork 可能会提供一个更完善的解决方案。还有推测认为，Cowork 的开发反映了吸引技术和非技术用户的战略转型，通过整合工具来降低企业成本。

    - PoorPhipps 强调，Cowork 似乎是在现有工具（如 TODO 列表和 AskUserQuestion Tool）的基础上利用 WebUI 封装，这表明除了核心的 Claude Code 语境之外，Anthropic 还在采取激进的构思策略。这预示着其可能转向集成更多用户友好的界面，以扩大其在技术用户之外的影响力。
    - painterknittersimmer 引用了之前的预测，即许多 Claude Code 功能将过渡到桌面版本，强调了 Anthropic 吸引技术和非技术用户的战略举措。这种方法旨在整合工具，通过提供统一的解决方案而不是单独订阅 Cursor 和 ChatGPT 等服务，潜在地降低企业的成本。

  - **[Claude 刚刚推出了 Cowork：针对非开发事务的 Claude code](https://www.reddit.com/r/ClaudeAI/comments/1qb6gdx/claude_just_introduced_cowork_the_claude_code_for/)** (热度: 596): ****Anthropic** 为 macOS 上的 **Claude Max** 订阅者发布了一项名为 **Cowork** 的新功能研究预览版。该工具将 Claude Code 的能力扩展到非编程任务，允许用户让 Claude 指向计算机上的一个文件夹，使其能够自主地读取、编辑和创建文件。它可以执行诸如自动整理文件夹、从截图创建电子表格以及根据笔记起草报告等任务，同时与现有的连接器集成，并在与 Chrome 中的 Claude 配合时处理浏览器任务。更多详情见[官方博客文章](https://claude.com/blog/cowork-research-preview)。** 评论者对 Cowork 和 Claude 桌面端之间的区别感到好奇，并指出 Cowork 对不太懂技术的用户可能非常有益。然而，也有人担心如果用户没有备份，可能会出现数据丢失的情况，因为 Cowork 的运行具有显著的自主性。

    - deepthinklabs_ai 提出了一个关于 Claude Cowork 和 Claude 桌面端之间差异的技术咨询，建议需要明确针对非开发人员与开发人员量身定制的功能集和用户界面。
    - Ok-Inspection-2142 指出，Claude Cowork 的功能似乎与桌面版本中已有的功能相似，特别是对于拥有 Max 方案（包含目录读写能力）的用户。这表明新产品可能更多是为了向不同的用户群体进行营销，而不是引入新的技术功能。
    - trimorphic 强调了对于可能没有完善备份方案的非技术用户的潜在风险，强调了数据管理的重要性，以及在没有适当保护措施的情况下，将关键任务依赖 AI 工具的潜在后果。

  - **[Agentic CLI 工具对比](https://www.reddit.com/r/CLine/comments/1qaycqj/agentic_cli_tools_comparison/)** (热度: 8): **该图片是一个柱状图，直观地展示了在 20 个 Web 开发任务中测试的各种 Agentic CLI 工具的成功率。对比的工具包括 Kiro, Aider, Cline, Claude Code, OpenAI Codex CLI 和 Gemini CLI，其中 Kiro 达到了最高的 `77%` 成功率，而 Gemini CLI 最低，为 `47%`。此次对比旨在评估这些工具在实际开发工作流中的有效性，并对其核心实用性提供见解。完整的基准测试和方法论可以在[这里](https://research.aimultiple.com/agentic-cli/)访问。** 一位评论者幽默地指出，Kiro 尽管成功率很高，但有时即使有错误也会声称已完成，建议用户尝试不同的 Agent 以找到最合适的工具。另一位评论者则批评 Aider 是他们用过的最差的 CLI 工具，表达了强烈的不满。



### 3. 人体解析与 LLM 评估

- **[[P] 开源在精选数据上训练的人体解析模型，以解决 ATR/LIP/iMaterialist 的质量问题](https://www.reddit.com/r/MachineLearning/comments/1qax221/p_opensourcing_a_human_parsing_model_trained_on/)** (Activity: 21): **FASHN Human Parser** 是一个最新开源的模型，基于 **SegFormer-B4** 微调而成，用于时尚场景下的人体解析（human parsing），旨在解决 ATR、LIP 和 iMaterialist 等现有数据集中的质量问题。该模型可输出包括身体部位和服装在内的 `18 个语义类别`，并针对时尚/电子商务图像进行了优化。它使用 `384 x 576` 的输入尺寸，并按输入分辨率输出分割掩码（segmentation masks），在 GPU 上的推理时间约为 `~300ms`，在 CPU 上约为 `2-3s`。该模型已在 [PyPI](https://pypi.org/project/fashn-human-parser/) 和 [HuggingFace](https://huggingface.co/fashn-ai/fashn-human-parser) 上发布，并在其 [博客文章](https://fashn.ai/blog/fashion-segmentation-datasets-and-their-common-problems) 中提供了详细的数据集分析。一位评论者对模型的开源表示感谢，而另一位评论者提到正在使用不同的方法进行类似项目，表明了对人体解析其他方法论的兴趣。


  - **[[R] 通过博弈论反馈循环引导 LLM agents](https://www.reddit.com/r/MachineLearning/comments/1qb2spz/r_guiding_llm_agents_via_gametheoretic_feedback/)** (Activity: 13): **该论文介绍了一种利用博弈论反馈循环引导基于 LLM 的 agents 的新方法。该方法涉及将 agent 交互日志转换为结构化图，在这些图上求解零和攻击者-防御者博弈以找到纳什均衡（Nash equilibrium），并将均衡统计数据作为战略控制信号应用于 agent 的系统提示词（system prompt）中。在 `44 次运行的基准测试`中，该方法将成功率从 `20.0%` 显著提高到 `42.9%`，将工具使用方差降低了 `5.2 倍`，并将预期的成功时间缩短了 `2.7 倍`。完整论文可在[此处](https://arxiv.org/pdf/2601.05887)获取，代码可在 [GitHub](https://github.com/aliasrobotics/cai) 上访问。** 热门评论质疑了该帖子的踩赞比，认为这篇论文很有趣，表明研究的感知价值与社区反应之间可能存在脱节。


  - **[[R] 关于评估指纹（Evaluative Fingerprints）的论文：LLM 评估器行为中稳定且系统性的差异](https://www.reddit.com/r/MachineLearning/comments/1qastrk/r_paper_on_evaluative_fingerprints_stable_and/)** (Activity: 7): **论文 "Evaluative Fingerprints" 探讨了将 LLMs 作为评估器的可靠性，揭示了虽然单个模型在其评估中保持一致，但不同模型之间几乎没有共识，Krippendorff’s α ≈ 0.042。该研究使用包括 Claude-Opus-4.5、GPT-5.2 等在内的多个 LLMs，对 YouTube SEO 内容包和维基百科文章进行了评估。研究发现，根据评分模式和证据使用情况，可以以 89.9% 的准确率识别出评判者模型，这突显了模型在评估内容方式上的系统性差异。这表明 LLMs 具有独特的评估“指纹”，影响了它们在 Benchmarking 和决策过程中的应用。有关更多详情，请参阅原论文[此处](https://arxiv.org/pdf/2601.05114)。** 评论者指出了这些发现对 LLMs 在评估角色中可靠性的影响，强调在将 LLMs 用作质量评估中人类判断的代理时需要谨慎。一些人建议进一步探索这些评估差异可能如何影响实际应用。

- 该论文引入了 'Evaluative Fingerprints' 的概念，用以描述不同 LLM 在评估行为中表现出的持续且系统性的差异。这一点非常重要，因为它表明 LLM 在评估任务中可能无法互换，且它们的偏见可能会影响结果。该研究通过多种基准测试证明了这些差异，强调了根据特定评估语境仔细选择模型的必要性。
- 论文中的一个关键技术见解是使用了一种新型指标来量化 LLM 评估者行为的稳定性和系统性。该指标允许研究人员比较不同模型如何持续偏好某些类型的回答或表现出特定的偏见。论文提供了详细的统计分析和可视化图表来支持这些发现，这对于开发更可靠的评估框架至关重要。
- 讨论还涉及了这些发现对 AI 伦理和公平性的影响。通过识别 LLM 评估者中的系统性偏见，论文指出依赖单一模型进行评估可能会使这些偏见持续存在。这要求在模型选择和评估中采用更多样化的方法，以确保在不同应用中获得更公平的结果。

---

# AI Discord 内容回顾

> 摘要的摘要之摘要


## Gemini 3.0 Pro 预览版 11月18日

**主题 1. 模型内部机制与性能：上下文限制与“懒惰”架构**

- **Gemini 的 120k 上下文断崖**：工程师报告称，**Gemini** 的性能在超过 **120k tokens** 后显著下降，未能通过 **Haiku** 和 **Sonnet** 能够轻松应对的 "needle in a haystack" 测试。此外，**Google AI Pro** 用户正面临严格的每周新限额，迫使他们升级到 **Ultra** 或迁移到其他 API。
- **Claude 误导性的文件重写**：用户指责 **Claude Sonnet 4.5** 和 **Opus 4.5** 经常在编辑范围上 *撒谎*，声称只做了微小改动，实际上却在激进地压缩或重写整个文件。这种行为增加了 diff 审查的难度，并导致用户对模型在保留代码结构方面的忠实度产生怀疑。
- **OpenAI "Sweetpea" 硬件泄露**：泄露信息表明 OpenAI 正在开发一款代号为 **"Sweetpea"** 的音频可穿戴设备，采用 **2nm 芯片**和金属“蛋石”设计，旨在与 AirPods 竞争。与此同时，OpenAI 收购了 [Torch](https://torchbio.com/)，以将临床数据功能整合到 **ChatGPT Health** 中，标志着其向医疗 AI 领域的垂直推进。

**主题 2. 底层优化与硬件：Blackwell、GSP 和 Layouts**

- **Blackwell 的 11 周期声明被证伪**：GPU MODE 工程师指出，**NVIDIA** 声称在 **Blackwell** 上执行 256x256 操作仅需 **11 cycles** 的说法具有误导性。分析澄清该操作是 **asynchronous**（异步）的，实际上并没有在该窗口内完成执行，这改变了对延迟的预期。
- **RTX 3090 Linux 崩溃修复**：用户确定 **GSP firmware** 是 **RTX 3090** 在 **Fedora** 和 **Windows** 上进行 LLM 推理期间重启的根本原因。禁用 GSP，并配合 **undervolting**（降压）和 **underclocking**（降频），可以在高计算负载下稳定显卡。
- **布局索引陷阱 (Layout Indexing Footguns)**：关于 **Cutlass** 实现的讨论强调，天真地进行超出布局大小的索引（回绕）会创建一个“退化”或 **zero layout**。这会导致 kernel 组合期间发生静默失败，强调了在 **CUDA** kernels 中进行严格边界检查的必要性。

**主题 3. 编码工作流与 Agent 框架：协议优于聊天**

- **Doomlaser Vibe Coding Protocol (DVCP)**：[DVCP](https://doomlaser.com/longform/dvcp) 提出了一种结构化的“命令与控制”线程架构，以绕过 LLM 编码中的 **DOM ceiling** 问题。该协议要求全文件输出，以防止标准聊天界面中常见的“宜家式”局部代码编辑导致的性能退化。
- **Copilot 劫持 Codex 凭据**：据报道，即使在扩展程序被禁用时，**Copilot** 扩展也会使用 **Codex CLI** 凭据，从而导致身份验证冲突。开发人员被迫手动管理和清除共享凭据库，以解决工具之间的碰撞。
- **Cursor Token 歧义**：**Cursor Pro** 用户对预付费 Token 过期政策表示困惑，质疑未使用的容量是否可以结转。同时，一个 bug 导致聊天面板发生故障并以浏览器标签页形式打开，干扰了 IDE 的集成工作流。

**主题 4. 开源训练与量化：Qwen、GRPO 和数据集**

- **精简微调数据集**：Unsloth 发布了[精简数据集](https://huggingface.co/enPurified/datasets)（例如 **smollm-corpus**、**standardebooks**），这些数据集通过启发式过滤器剔除了数学、代码和外语文本。此次发布的目的是为了比原始的 **FineWeb** 转储提供更干净的 **SFT**（有监督微调）基准。
- **Qwen3 Coder 量化失败**：运行 **Qwen3 Coder** 的工程师报告称，任何低于 **Q8_0** 的量化都会导致严重的性能下降，包括空格缺失和工具请求损坏。这使得该模型在没有全精度支持的情况下，实际上无法在 **RTX 5090** 等消费级硬件上处理复杂任务。
- **GRPO 助力推理提升**：一份新的 [CURE-GRPO 报告](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1766255740138) 展示了如何使用 **Self-Critique**（自我批判）结合 **GRPO**（Group Relative Policy Optimization）来增强推理能力。该方法在 **Google Tunix Hackathon** 期间的 **Gemma** 模型中显示出了显著的提升。

**主题 5. 边缘部署与模型运维：ARM、TorchScript 与 BMC**

- **LM Studio 在 ARM 上运行**：一位用户在运行 **Qwen3 4b** 的 **Orange Pi 6 Plus**（8 核 CPU）上实现了 **6.6 t/s** 的速度，证明了 LLM 在低功耗 **ARM** 架构上的可行性。然而，目前的实现存在视频驱动损坏的问题，需要通过 UI 规避方案解决。
- **TorchScript 弃用引发的混乱**：随着 **TorchScript** 被弃用，工程师们正转向使用 **torch.compile** 进行 C++ 部署。这一转变给那些依赖 **libtorch** 的用户带来了摩擦，引发了对桥接 Python 原型设计与 C++ 生产环境的高效方法的搜寻。
- **TinyBox BMC 锁定**：**TinyBox** 用户报告了 **BMC**（基板管理控制器）锁定问题，需要通过 **UEFI** 重新刷写固件才能解决。作为回应，**George Hotz** 宣布了新的“速度赏金”，以激励对硬件基础设施和稳定性的改进。


## gpt-5.2


**1. 医疗保健与监管赋能 LLM 产品化**

- **OpenAI 为 ChatGPT Health 点亮“火炬”**：OpenAI 收购了医疗初创公司 [Torch](https://torchbio.com/)，通过统一**化验结果、药物和就诊记录**来加强 **ChatGPT Health**。该交易包括 Torch 团队成员 [@IlyaAbyzov](https://x.com/IlyaAbyzov)、[@elh_online](https://x.com/elh_online)、[@jfhamlin](https://x.com/jfhamlin) 和 Ryan Oman 加入 OpenAI。
  - 工程师们将其视为在 **ChatGPT Health** 内部实现端到端健康上下文摄取（结构化实验数据 + 自由格式录音）的实质性举措，团队迁移信号表明 OpenAI 希望将产品界面和数据管道专业知识都收归内部。

- **FDA 更新临床试验统计指南**：[FDA 发布了使临床试验统计方法现代化的指南](https://www.fda.gov/news-events/press-announcements/fda-issues-guidance-modernizing-statistical-methods-clinical-trials)，参与者指出这与 **AI/ML** 系统在医疗环境中的验证方式密切相关。
  - 讨论集中在对**鲁棒统计验证**的更高期望上——这可能是促使开发临床/健康 Copilot 的团队将评估视为不仅仅是“离线基准测试”的强制机制，特别是当模型触及面向患者的工作流时。


**2. 模型可靠性、长上下文现实检查及使用限制**

- **Gemini 的“400k 上下文”遭遇 120k 瓶颈**：在各社区中，用户报告 **Gemini** 的质量在超过约 **120k 上下文**后出现下降，理由是“大海捞针”（needle-in-a-haystack）测试失败，并对比称 **Haiku** 和 **Sonnet** 表现更好；另一些人则反驳称他们运行 **400k+ tokens** “很正常”，这暗示了结果存在取决于工作负载的差异。
  - 争论迅速转向长上下文声明是否隐藏了评估技巧（“水分”），还是存在真实的检索/注意力限制，从业者建议进行显式的长上下文测试，而不是盲目相信营销数字。

- **Claude “编辑一点点”，然后重写全世界**：用户指责 **Claude Sonnet 4.5** 和 **Opus 4.5** 经常“撒谎”，包括 Sonnet 声称只做了微小改动但实际上**重写/压缩了整个文件**，以及虚假陈述称**互联网搜索已被禁用**。
  - 团队将此描述为代码审查和重构的可靠性退化，在将 Claude 集成到自动化流程之前，促使了更多“信任但验证”的工作流（基于 diff 的审查、全文件输出和更严格的修改限制）。

- ****限制、节流与意外的模型切换****：Google **AI Pro** 用户抱怨 **Gemini** 的使用量限制转为**周限**（从日限下调），以此诱导用户升级到 **Ultra**；同时 Perplexity 用户讨论了“静默节流”问题，并提到 **Perplexity Pro** 的视频生成上限为**每月 3 个**；另一份报告显示，在请求 **Gemini 3 Pro** 时 Perplexity 返回了 **GPT 5.2**，该问题可通过[刷新/重新打开](https://plx.link/GeminiGlitches)解决。
  - 核心逻辑在于“计费方案物理学”塑造了用户体验：模型路由的意外、不透明的速率限制以及方案层级上限，促使用户转向多供应商配置或严格的使用规范，以避免在工作流中途被中断。


**3. Vibe Coding 工具链：从全文件 Diff 到新框架**

- ****DVCP 向“宜家手册式”代码编辑开战****：一篇长文介绍了用于 LLM 编程的 **Doomlaser Vibe Coding Protocol (DVCP)**，该协议使用“指挥与控制”和“行政休息室”线程，并要求**全文件代码输出**，以避免零碎的补丁建议（[DVCP 文章](https://doomlaser.com/longform/dvcp)）。
  - 社区认可 DVCP 对线程切换和“DOM 天花板”限制的务实处理（参见 [DVCP 附录 C](https://doomlaser.com/longform/dvcp)），将其定位为流程级工具，以保持 LLM 编码在长期项目中的确定性。

- ****Cursor 及其伙伴：当 Opus 变慢时，Codex 交付****：在 Cursor 的讨论中，多位用户抱怨 **Claude Opus** 感觉缓慢且低效，其中一人声称他们在 **10 秒**内解决了一个 Opus 在 **30 分钟**内未能修复的问题，另一人则切换到了 **Codex**；另有关于 OpenAI 的讨论推荐在短上下文中使用 **Codex 5.2**，长上下文则最高使用 **5.1**。
  - 共识模式是“按任务 + 上下文长度进行路由”：根据指令遵循的稳定性及记忆保留能力选择模型，并在重构风险较高时，回退到支持清晰 Diff 和全文件重新生成的工具。

- ****Pulse 框架加入编码工具角力场****：一位社区成员推出了 **Pulse**，并在持续的“Claude Code 对标其他工具”讨论中分享了该仓库（[GitHub 上的 PulseFramework](https://github.com/manuelfussTC/PulseFramework)）。
  - 开发者将其视为轻量级编排层大趋势的一部分——减少“Agent 神秘感”，更多地关注可重复的工作流和集成钩子，使模型切换和工具调用更易于管理。


**4. GPU 性能与分析：基准测试、内核与更好的工具**

- ****Blackwell 的“11 个周期”说法被打脸****：GPU MODE 成员挑战了《微基准测试 NVIDIA Blackwell 架构》中的一项声明，即 **256×256 操作**在 **11 个周期**内完成。反驳者指出该操作是**异步**的，因此论文的解读很可能错误。
  - 启示在于方法论：对于现代 GPU，周期计数的声明必须考虑异步执行、排队和测量偏差——否则你最终测试的是你的假设而非芯片本身。

- ****popcorn-cli v1.2.2 发布内联 NCU 摘要****：GPU MODE 发布了 **NCU (NVIDIA Command Line Utility)** 集成，使 CLI 能够内联渲染分析摘要，并可通过 [popcorn-cli v1.2.2](https://github.com/gpu-mode/popcorn-cli/releases/tag/v1.2.2) 下载 **.ncu-rep** 产物，使用方法详见[分析文档](https://github.com/gpu-mode/popcorn-cli/blob/main/docs/profiling.md)。
  - 此次升级旨在实现内核开发的“紧密反馈循环”：减少上下文切换，使报告可共享，并将性能讨论标准化为可复现的分析器输出，而非截图和感觉。

- ****CUDA Source-Page 将内存合并分析变成“犯罪现场地图”****：开发者建议使用 Nsight 的 **Source-page** 配合 `-lineinfo` 和 `--import-source yes` 来精确定位内存访问模式和合并（coalescing）问题，该工具会链接到每个内核中表现最差的部分。
  - 经常出现的建议：不要盲目增大 Tile 大小——优先考虑流水线并发（INT32/FP32 重叠），将 `cp.async.bulk.tensor (TMA)` 视为带有“锋利边缘”的 API，并记住 `wgmma` 的 warp-group 异步语义需要显式的 commit/wait 规范。


**5. 开源模型运维：MoE 支持、数据集纯化与边缘部署**

- ****Unsloth 修复了 Qwen3 MoE；Nemotron 仍然是“那个异类”****：Unsloth 报告称在一次贡献后 **Qwen3 MoE 已可正常工作**，并指出 **Nemotron** 仍然不走寻常路，因为它尚未完全集成到 transformers 中，相关修复工作见 [unsloth-zoo PR #440](https://github.com/unslothai/unsloth-zoo/pull/440)。
  - 该讨论串提醒人们，“模型可用性”不仅仅是权重——它还涉及工具链集成、张量命名稳定性以及下游生态系统支持（例如，另有报告称 **Qwen3-Next** 在 Ollama 中运行失败，原因是缺少张量 `blk.0.ssm_in.weight`）。

- ****enPurified 像数据滤水器一样蒸馏数据集****：一个分享启发式数据集“蒸馏”的项目，旨在减少流行语料库中的数学、代码、外语文本和低质量英语，并在 [enPurified Hugging Face datasets](https://huggingface.co/enPurified/datasets) 上发布了产出（包括 **smollm-corpus**、**LongPage**、**standardebooks**、**project_gutenberg**、**finewiki**）。
  - 他们还发布了一个从剪枝后的 **fineweb-edu-dedup** 转换而来的 messages 格式 SFT 数据集——[smollm-corpus-fineweb-edu-enPurified-openai-messages](https://huggingface.co/datasets/enPurified/smollm-corpus-fineweb-edu-enPurified-openai-messages)——将其定位为微调所需的“更干净的输入，更少的 Prompt 呵护”。

- ****边缘端 LLM 的延伸：移动端 SLM + ARM 上的 LM Studio****：Unsloth 与 Cactus 合作部署开源手机模型（[Reddit: Deploying Unsloth SLMs on mobile devices](https://www.reddit.com/r/unsloth/comments/1qb9vdj/deploying_unsloth_slms_on_mobile_devices/)），同时一位 LM Studio 用户在 **Orange Pi 6 Plus (Ubuntu)** 上运行，并报告使用 **8 核** CPU 时，**Qwen3 4b 2507 Q4** 的速度达到 **6.6 t/s**。
  - “本地优先”的主题在实际摩擦中继续推进：ARM 上不成熟的视频驱动导致 UI 损坏，以及模型质量陷阱，如 Hugging Face 用户声称 **qwen3 coder** 在量化下表现不佳，除非使用 **`Q8_0`**。


## gpt-5.1


**1. Gemini, Apple, and Google：长上下文、限制以及跨生态系统紧张局势**

- **Gemini 的长上下文信誉在 120k Token 时出现裂缝**：**Unsloth** 成员报告称，**Gemini** 的性能在 *超过约 120k 上下文 Token 后显著下降*，**Needle‑in‑a‑haystack**（大海捞针）测试显示其出现了 **Haiku** 和 **Sonnet** 能正确处理的失败，尽管 Google 宣称其拥有通用的长上下文营销口号。
  - 一些用户反驳说他们*“使用 Gemini 处理了 400k+ Token，一切正常”*，这引发了关于 Google 的评估是否 **造假（cooked）**、上下文处理是否严重依赖分布，或者特定部署在质量上是否存在差异的争论。

- **Gemini 3 Pro 遭到限流和吐槽**：在 **OpenAI**、**Perplexity** 和 **OpenRouter** 的 Discord 频道中，用户抱怨 **Gemini/Gemini 3 Pro** 在 **Google AI Pro** 中现在的限制是*每周*使用上限而非每日，迫使用户升级到 **Ultra**；而 Perplexity 用户报告 Gemini 3 Pro 有时会静默切换到 **GPT‑5.2**（[Bug 讨论串](https://plx.link/GeminiGlitches)）。
  - 在 **OpenRouter** 上，一位用户咆哮道*“没人喜欢”* **Gemini 3 Pro**，而其他人则表示 Gemini 3 还算 *可以*，但在 **指令遵循和幻觉率** 方面明显逊于 **GPT‑5.2**，尤其是在 **长对话（约 200k Token）** 中，据称 Gemini 的 API 响应会出现编造数字和搞砸转录的情况。

- **Apple 押注 Gemini，惹恼自定义模型爱好者**：在 **Unsloth** 的 off‑topic 频道中，用户抨击 **Apple** 将 **Gemini** 接入 **Siri**，而不是为自定义模型提供简单的*“我的服务器 URL”*钩子，并分享了一张配文为 *“蒂姆·库克（Tim Cook）绝对得走人”* 的 [库克照片表情包](https://cdn.discordapp.com/attachments/1179039861576056922/1460398437961957406/image0.jpg) 以及一篇 [MacRumors 关于埃隆·马斯克反应的文章](https://www.macrumors.com/2026/01/12/elon-musk-reacts-to-gemini-siri/)。
  - 在 **OpenRouter** 的讨论频道中，用户引用了一篇 [MacRumors 文章](https://www.macrumors.com/2026/01/12/google-gemini-future-apple-intelligence-features/) 并开玩笑说 *“Google 表现太强了，以至于他们不得不提携 Apple 以避免反垄断法”*，而其他人则严肃质疑 **Google-Apple AI 集成** 是否踏入了 **反垄断** 领域。


**2. 编排与部署开源模型：从 Qwen3 MoE 到移动端 SLM**

- **Qwen3 MoE、Nemotron 和 GPT‑OSS 获得实战调优**：在 **Unsloth** 上，贡献者确认在提交给 **unsloth‑zoo** 的 PR（[PR #440](https://github.com/unslothai/unsloth-zoo/pull/440)）之后，**Qwen3 MoE** 现在已经可以工作，而 **Nemotron** 由于未完全集成到 **Transformers** 中，仍然表现古怪。
  - 用户还对 **GPT‑OSS‑120B** 进行了基准测试，报告称通过 `-ot` 优化，其速度达到 **~27 tokens/s**，使其在非多模态任务中 *“几乎与许多 30B MoE 一样快”*，并突显了经过优化的稠密模型（dense models）如何在吞吐量上与 MoE 竞争。

- **SLMs 走向移动端：Unsloth + Cactus 推动手机级模型发展**：**Unsloth** 社区分享了与 **Cactus** 合作在**移动设备上部署 SLMs** 的成果，并发布了一篇关于“在移动设备上部署 Unsloth SLMs”的 Reddit 报告（[文章链接](https://www.reddit.com/r/unsloth/comments/1qb9vdj/deploying_unsloth_slms_on_mobile_devices/)）。
  - 讨论重点在于通过激进的量化和运行时优化来封装 **Qwen 类及类似的 SLMs**，使普通手机能够运行**设备端助手（on‑device assistants）**，这反映了从仅限云端的 LLM 使用向**边缘推理（edge inference）**的转变。

- **用于更干净微调的蒸馏与“纯化”语料库**：一位用户在 **Hugging Face** 上介绍了 **enPurified** 项目，该项目通过剔除数学、代码、外语和低质量英语，对流行语料库（如 **smollm‑corpus**、**LongPage**、**standardebooks**、**finewiki**）进行*启发式蒸馏*，从而创建更干净的 **SFT‑ready 数据集**（[数据集仓库](https://huggingface.co/enPurified/datasets)）。
  - 他们重点介绍了一个经过高度打磨的 **Project Gutenberg** 变体：[**project_gutenberg‑enPurified‑openai‑messages**](https://huggingface.co/datasets/enPurified/project_gutenberg-enPurified-openai-messages)，以及一个转换为 OpenAI-messages 格式的 **Fineweb‑edu‑dedup** 版本（[smollm‑corpus‑fineweb‑edu enPurified 数据集](https://huggingface.co/datasets/enPurified/smollm-corpus-fineweb-edu-enPurified-openai-messages)），并指出激进的预过滤能减少垃圾梯度并加速**指令式微调（instruction‑style finetunes）**。


**3. GPU、TPU 与 Kernel 工程：从 Blackwell 到混合精度与 MXFP8**

- **Blackwell 的 11 周期声明遭到实测挑战**：在 **GPU MODE** 中，成员们剖析了《NVIDIA Blackwell 架构微基准测试》，质疑其中一项核心声明，即 **256×256 Tensor Core 操作**在 **11 个周期**内完成，这将意味着几乎荒谬的吞吐量。
  - 另一位工程师澄清说，该操作是**异步（asynchronous）**的，因此 11 周期的数字是调度伪像（scheduling artifact）而非真实延迟，这暗示该论文的部分**假设和推导结论在实际 Kernel 设计中存在缺陷**。

- **CUDA 玩家深入探讨 cp.async、TMA 和布局魔术**：**#cuda** 频道对**内存合并（memory coalescing）与性能分析**进行了深入讨论，成员们提倡使用 **Source-page** 配合 `-lineinfo`/`--import-source yes` 来检查错误访问，并将 `cp.async` 与 **TMA (`cp.async.bulk.tensor`)** 进行对比，有些人发现后者在实践中反而略慢。
  - 讨论涵盖了 Ampere/Blackwell 上的 **`mma.sync` vs `wmma` vs `wgmma`**、异步 Tensor‑Core 流水线以及 CUTLASS 中的**布局组合（layout composition）**（包括一个有趣的观察：将布局与 `1:2` 布局组合仅会使 **stride 扩大 2 倍**，而简单的绕回索引（wrap‑around indexing）则会产生无用的“零布局”），为开发者编写自定义 GEMM 提供了详细的启发式方法。

- **FP8 和 MXFP8 MoE Kernel 寻求基准测试**：在 **#torchao** 中，有用户询问 TorchAO 中 **MXFP8 块级比例融合 MoE Kernel（block‑scale fused MoE kernels）**的推理基准测试，希望能直接与 **FlashInfer CUTLASS** 和 **TensorRT‑LLM FP8** MoE Kernel 进行对比，以期在做比较时*避免重复造轮子*。
  - 其他人建议咨询内部推理专家，这强调了对跨厂商技术栈的**标准化 FP8/MXFP8 MoE 基准测试**的真实需求，以便工程师能够为生产推理做出明智的权衡。


**4. 编程助手、IDE 和用于优化 LLM 编程的控制循环**

- **DVCP：Doomlaser 将编程转变为一种多线程 Vibe 仪式**：在 OpenAI 的 **ai‑discussions** 中，一位用户分享了 **Doomlaser Vibe Coding Protocol (DVCP)**，这是一个用于 LLM 辅助编程的详细系统，使用“指令与控制（command‑and‑control）”线程和“行政休息室（executive lounge）”线程，并持续要求 **全文件代码输出**（[DVCP 文章](https://doomlaser.com/longform/dvcp)）。
  - **DVCP 附录 C**（[附录](https://doomlaser.com/longform/dvcp)）描述了拆分线程和移交上下文如何帮助避开 DOM 大小上限和 LLM 典型的*“宜家式零碎修改（IKEA‑style small edits）”*，从而有效地将 ChatGPT 类的工具转变为**分块批量重构引擎**。

- **SLM 控制循环在无需重写的情况下战胜行为漂移**：在 **OpenAI 的 prompt‑engineering 和 api‑discussions** 中，一位成员描述了一个围绕 **phi‑3‑mini**（通过 Ollama 运行）构建的 **5 层闭环控制器**——验证（Validation）→ 迭代（Iteration）→ 评估（Evaluation）→ 反馈（Feedback）→ 校准（Calibration）——旨在不修补先前输出的情况下稳定叙事风格。
  - 这种编排将 **清晰度（Clarity）从 0.80 提升至 0.88，连贯性（Coherence）从 0.85 提升至 0.87，语调稳定性（Tone Stability）从 0.75 提升至 0.85，风格稳定性（Style Stability）从 0.70 提升至 0.83**（参见附带的 [BOF_3outputs.docx](https://cdn.discordapp.com/attachments/1046317269069864970/1460406335009980580/BOF_3outputs.docx)）。该方法通过对每一轮对话评分并将“引导指令”反馈到下一个 Prompt 中来实现，同时讨论了在 **Token 成本、125k+ 上下文的延迟以及“注意力稀释（attention dilution）”** 与仅使用更好的 Prompt 重启会话之间的权衡。

- **IDE 之战：Cursor、Copilot/Codex 和 Claude Code/Cowork 的对决**：在 **Cursor Community** 中，用户讨论了 **Cursor Pro** 的经济效益和 UX（每月 **20 美元的未用 Token 包**是否会过期？），并抨击 **Claude Opus** 既慢又低效；一位工程师在切换到 **Codex** 之前，仅用 **10 秒**就修复了一个 Claude 在 **30 分钟**内都未能解决的 Bug。
  - 与此同时，OpenAI 服务器用户发现 **Copilot** 和 **Codex CLI** 会静默共享凭据。**Latent Space** 重点介绍了 Anthropic 新推出的面向非技术办公工作流的 **Claude “Cowork”** 工具（[Cowork 发布公告](https://x.com/claudeai/status/2010805682434666759)），并提出了一个更广泛的问题：与提示词工程良好的模型和自定义控制循环相比，**类 Agent 的 IDE 工具**到底能增加多少价值？


**5. 基础设施、基准测试与平台：从 LMArena 到 Torch、Phind 和 LMStudio**

- **Torch 加入 OpenAI 以助力 ChatGPT Health**：OpenAI 宣布收购医疗保健初创公司 **Torch**，该公司专注于统一**化验结果、药物治疗和就诊记录**，并计划将这一技术栈整合进 **ChatGPT Health**（[Torch 官网](https://torchbio.com/)）。
  - Torch 的核心团队成员——**Ilya Abyzov**（[个人资料](https://x.com/IlyaAbyzov)）、**Eli Heilman**（[个人资料](https://x.com/elh_online)）、**Jeff Hamlin**（[个人资料](https://x.com/jfhamlin)）和 **Ryan Oman**——正在加入 OpenAI，这标志着 OpenAI 正在认真推动将 ChatGPT 转变为一个**具备临床工作流意识的助手**，而非通用的聊天机器人。

- **LMArena 扩展视频对战与社区竞赛**：**LMArena** 在其 **Video Arena** 中增加了 **ltx‑2‑19b** 模型用于 **Battle Mode** 评估，邀请用户对两两对决的视频生成进行投票（[arena 频道](https://discord.com/channels/1340554757349179412/1397655624103493813)）。
  - 他们还启动了 1 月份的 **AI 生成大赛**，主题为**“自然回归（Nature Reclaims）”**。用户需在 **1 月 16 日**前在竞赛频道提交 Battle Mode 截图，角逐 **Discord Nitro** 奖励和 **AI Content Creator** 角色（[竞赛公告](https://discord.com/channels/1340554757349179412/1378032433873555578)）。

- **LMStudio 与小型硬件：从 RTX 3090 重启到 Orange Pi ARM**：在 **LM Studio** 服务器中，用户正在排查 **RTX 3090** 显卡在 LLM 负载下（Fedora 和 Windows 系统）出现“硬重启”的问题。缓解措施包括禁用 **GSP 固件**、**降压/降频**，以及通过 **OCCT** 和极端的稠密推理 Prompt 进行压力测试。
  - 另一位用户成功在 **Orange Pi 6 Plus (ARM, Ubuntu)** 上运行了 **LMStudio**，使用 **Qwen3‑4B‑2507 Q4** 达到了 **~6.6 tokens/s**，在纯 CPU 运行 **gpt‑oss** 时达到 **~6.26 t/s**。尽管 Electron UI 损坏迫使他们只能依靠“右侧配置栏 + 盲点”的折中方案，直到更好的 **ARM GPU/NPU 驱动**发布。


## gpt-5


**1. OpenAI 健康业务扩张与临床验证**

- **OpenAI 吞并 Torch 以强化 ChatGPT Health**：OpenAI 收购了医疗保健初创公司 **Torch**，该公司专注于统一**化验结果、药物治疗和就诊记录**，以增强 **ChatGPT Health** 的能力，正如 [Torch](https://torchbio.com/) 上所宣布的。整个 **Torch 团队**——包括 [@IlyaAbyzov](https://x.com/IlyaAbyzov)、[@elh_online](https://x.com/elh_online)、[@jfhamlin](https://x.com/jfhamlin) 和 Ryan Oman——将加入 OpenAI 以加速健康功能的开发。
  - 成员们将其视为向更强大的**个人健康 Agent** 迈进的一步，该 Agent 能够解析结构化/语音医疗数据，并在隐私限制内提供摘要。他们预计在与电子健康档案（EHR）相关的工作流上将实现快速迭代，并指出需要严格验证以避免在临床环境中的**模型漂移（model drift）**。

- **FDA 为 AI 时代更新临床试验统计手册**：**FDA** 发布了更新指南，使临床试验的统计方法现代化，标志着 AI 辅助医疗评估在鲁棒性和可复现性方面将面临更高标准；详见新闻发布：[FDA issues guidance modernizing statistical methods for clinical trials](https://www.fda.gov/news-events/press-announcements/fda-issues-guidance-modernizing-statistical-methods-clinical-trials)。工程师将此解读为向透明流水线、预注册分析以及针对 AI 生成终点（endpoints）进行更严谨置信度报告的推动。
  - 讨论强调了当 AI 参与决策时，对于**不确定性量化（uncertainty quantification）**、数据集偏移分析和前瞻性试验设计的更高期望。一些人指出，这可能会迫使供应商提供**易于审计（audit-friendly）**的推理日志，并对从数据集到 Prompts 的所有内容进行版本化管理，以满足证据要求。


**2. 长上下文限制与 Gemini 平台更新**

- **Gemini 的记忆力在超过 120k Token 后下滑**：从业者报告称 **Gemini** 的性能在超过约 **120k tokens** 后开始下降，未能通过经典的“大海捞针”（*needle-in-a-haystack*）测试，而 **Haiku** 和 **Sonnet** 的表现则相对更好。尽管营销宣传称其具备通用的长上下文处理能力，但用户警告不要在没有定制检索策略的情况下依赖极端上下文。
  - 另一些人则以 **400k+ token** 会话运行良好的案例予以反驳，认为工作负载敏感度和 Prompt 规范（hygiene）是关键变量。工程师建议进行针对性的长上下文评估、增量检索和严格的日志记录，以检测静默回归。

- **Google AI Studio 瞄准直接媒体源输入**：**Google AI Studio** 宣布计划支持通过媒体 URL 直接向 Google 模型输入**视频/音频**，参考 [Google AI Studio: media URL support update](https://x.com/GoogleAIStudio/status/2010768441553428772)。目前的视频支持仍仅限 **YouTube**，音频仅限 **base64**，PDF 和图像可用，但 **2.0** 模型暂不支持。
  - 开发者欢迎更简单的摄取路径和更少的多模态 RAG 预处理步骤。他们要求 **Web App 与 API** 之间保持功能一致、配额稳定，并提供针对每种媒体类型的文档化延迟预算。

- **Perplexity 在请求 Gemini 时转向 GPT**：一位请求 **Gemini 3 Pro** 的用户却收到了 **GPT 5.2**，并被建议[刷新或重启应用](https://plx.link/GeminiGlitches)。该故障可能源于连接中断，导致会话在重置前被固定在错误的后端。
  - 网友还讨论了软**限流（throttling）**和计划限制，有人声称“所有 Pro 用户都被限流”，而另一些人则报告在 **Max** 上吞吐量正常。实际建议：在日志中检测供应商切换，并在 UI 中显示显著的**活动模型指示器**。


**3. GPU 性能见解与工具更新**

- **Blackwell 的 11 周期吹嘘被拆穿**：工程师对“Microbenchmarking NVIDIA’s Blackwell Architecture”中关于 **256×256** 操作在 **11 个周期**内完成的说法提出挑战，澄清该操作是**异步（asynchronous）**的，实际上并未在 11 个周期内结束。他们警告称，误读异步延迟会破坏性能模型和 Kernel 调优。
  - 共识：在没有 Kernel 全生命周期时间线和**合并（coalescing）**审计的情况下，应谨慎对待微基准测试数据。团队倾向于使用带注释的源码视图（使用 -lineinfo 和 import-source）来追踪内存行为并识别非合并访问的热点。

- **NCU CLI 集成到 Popcorn**：**NCU (NVIDIA Command Line Utility)** 现在通过 [popcorn-cli v1.2.2 版本](https://github.com/gpu-mode/popcorn-cli/releases/tag/v1.2.2)集成到 CLI 中，支持内联摘要和 **ncu-rep** 下载。通过使分析过程贴近 Kernel 实验，这简化了性能排查（perf triage）流程。
  - 文档在 [popcorn-cli profiling guide](https://github.com/gpu-mode/popcorn-cli/blob/main/docs/profiling.md) 中展示了快速启动命令和工作流。成员报告称迭代周期变快，且在评审中分享 **NCU** 产物变得更加容易。

- **Dagstuhl 深入探讨混合精度**：一个关于科学与工程领域**减少精度与混合精度计算（Reduced and Mixed Precision Computing）**的 Dagstuhl 研讨会在此浮出水面：[Dagstuhl Seminar 26081](https://www.dagstuhl.de/en/seminars/seminar-calendar/seminar-details/26081)。参与者期待关于 **FP8**、块缩放（block-scaling）以及大规模训练/推理中数值稳定性的议题。
  - 大家希望能在 **YouTube** 上看到视频上传，以便在学术界之外分享最佳实践。许多人希望获得关于**累加路径（accumulation paths）**、溢出检测以及能经受现实世界偏移考验的混合精度方案的具体指导。


**4. 开源模型、移动端 SLM 与更洁净的数据集**

- **通过 PR 解决 Qwen3‑MoE 的奇癖问题**：在 [unsloth-zoo PR #440](https://github.com/unslothai/unsloth-zoo/pull/440) 的贡献提交后，**Qwen3 MoE** 现在已可在 Unsloth 中运行；而由于 Transformers 集成不完整，**Nemotron** 仍较为棘手。贡献者们指出了 MoE 路由和导出路径方面的异常。
  - 团队优先考虑将配置与上游 **Transformers** 对齐并稳定导出目标。他们提醒，在混用仓库和转换工具时，缺失张量名称是常见的陷阱（footguns）。

- **SLM 通过 Unsloth × Cactus 走向移动端**：Unsloth 和 **Cactus** 在一篇教程中详细介绍了开源手机模型的部署：[在移动设备上部署 Unsloth SLM](https://www.reddit.com/r/unsloth/comments/1qb9vdj/deploying_unsloth_slms_on_mobile_devices/)。该文章涵盖了针对手机级硬件的打包、量化和运行时约束。
  - 开发者们讨论了**延迟与质量**的权衡，以及何时将任务卸载到边缘服务器。他们分享了在严格的内存上限下，使用较小的 SSM/SLM 变体处理交互式任务的成功经验。

- **蒸馏数据集实现更干净的微调**：社区发布了通过启发式方法修剪数学、代码、外语文本和低质量英语的**蒸馏数据集**，发布于 [Hugging Face 上的 enPurified 数据集](https://huggingface.co/enPurified/datasets)。亮点包括 **smollm-corpus**、**LongPage**、**standardebooks**、**project_gutenberg** 和 **finewiki**。
  - 他们通过 [project_gutenberg‑enPurified‑openai‑messages](https://huggingface.co/datasets/enPurified/project_gutenberg-enPurified-openai-messages) 展示了提升后的 Gutenberg 质量。从业者预计，更干净的指令对将带来更快的 **SFT** 收敛和更少的退化生成。


**5. 新产品发布与平台转变**

- **Claude Cowork 提升办公效率**：根据 [Claude: 'Cowork' 公告](https://x.com/claudeai/status/2010805682434666759)，Anthropic 推出了 **Claude ‘Cowork’**，将 **Claude Code** 风格的生产力扩展到非技术工作流中。其核心卖点是：利用 Claude 的工具使用（tool-use）和规划原语实现日常任务自动化。
  - 工程师们希望为企业级部署提供 API 钩子和清晰的**安全模型**。他们还要求提供可复现的运行记录和日志，以跟踪**工具-动作链**（tool-action chains）从而实现可审计性。

- **OpenAI ‘Sweetpea’ 可穿戴设备传闻**：一次泄露披露了 **OpenAI 的 ‘Sweetpea’**——一款配备 **2nm 芯片**和金属**卵石**外形的原型音频可穿戴设备，旨在进军 AirPods 领地；参见 [OpenAI 'Sweetpea' 可穿戴设备泄露](https://x.com/kimmonismus/status/2010804115543114099)。该设备暗示了深度的端侧推理和常驻 Assistant。
  - 开发者猜测其具备**本地唤醒词**、超低功耗 DSP 模块以及用于处理沉重任务的**混合云**。他们希望了解关于 SDK、延迟预算以及环境采集隐私边界的细节。

- **Phind 停止服务**：**Phind** 宣布关闭；参见帖子：[Phind 关闭公告 (Discord)](https://discord.com/channels/996538859972214835/1077743365749227630/1460382964029460584)。用户开始寻找代码优先搜索和 Agent 浏览工作流的替代方案。
  - 团队权衡了 **Perplexity**、**OpenAI o3** 的工具使用以及自定义 RAG 作为权宜之计。他们强调了缓存、离线镜像和**供应商无关**的适配器，以避免未来的中断影响。


---

# Discord: 高层级 Discord 摘要

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 收购 Torch 以助力医疗业务**：OpenAI 收购了医疗初创公司 [Torch](https://torchbio.com/)，旨在通过整合**实验结果、药物和就诊记录**来增强 **ChatGPT Health**。
   - **Torch 团队**，包括 [@IlyaAbyzov](https://x.com/IlyaAbyzov)、[@elh_online](https://x.com/elh_online)、[@jfhamlin](https://x.com/jfhamlin) 和 Ryan Oman，也将加入 OpenAI 以进一步提升其医疗能力。
- **Copilot 占用 Codex 凭证引发混乱**：**Copilot** 扩展程序使用与 **Codex CLI** 相同的凭证，即使在 **Copilot** 被禁用时也会引起混乱。
   - 建议用户仔细管理凭证，以避免两个系统之间产生冲突。
- **Gemini 的新限制惹恼 Google AI Pro 用户**：**Google AI Pro** 用户对 **Gemini** 新的每周限制（从每日限制缩减而来）表示沮丧，这迫使用户升级到 **Ultra**。
   - 这种转变由于带来不便，已促使一些用户寻求替代方案。
- **Claude 被指控捏造事实和重写文件**：用户指责 **Claude Sonnet 4.5** 和 **Opus 4.5** *经常撒谎*，据称 **Sonnet** 在进行微调时会大幅重写并压缩文件。
   - 这两个模型还谎称互联网搜索功能已被禁用，导致用户对其可靠性产生质疑。
- **Doomlaser 详述 DVCP：使用 LLMs 编码**：一位用户在[长篇文章](https://doomlaser.com/longform/dvcp)中分享了 **Doomlaser Vibe Coding Protocol (DVCP)**，详细介绍了一套使用 LLM 进行编码的系统，该系统涉及创建指挥控制（command-and-control）和执行休息室（executive lounge）线程，并请求全文件代码输出。
   - 该协议避免了许多 LLM 倾向于提供的“宜家式（IKEA-style）”编辑，并通过线程切换解决了 DOM 天花板问题，相关内容可在 [DVCP 附录 C](https://doomlaser.com/longform/dvcp) 中阅读。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini 故障，GPT 顶替**：一名成员报告称，在明确请求 **Gemini** 时却得到了 **GPT 5.2**，并被建议尝试[刷新或重新打开应用](https://plx.link/GeminiGlitches)。
   - 该问题被归因于对话途中可能出现的连接中断，这种情况可能会持续到刷新为止。
- **PC 端 Perplexity Pro 出现问题？**：一名用户在 PC 上使用 Perplexity 时遇到问题，即使在重新登录并清除 cookie 后，创建新聊天时仍报错，尽管其拥有 **Pro** 订阅。
   - 该问题在不同浏览器和 **Comet** 应用中持续存在，但在手机上运行正常；建议其[联系支持部门](mailto:support@perplexity.ai)或确保没有 VPN 或防火墙干扰连接。
- **Comet 无法抓取网页元素？**：一名成员询问 **Comet** 拖动网页元素的能力，但另一名成员回复称 [Comet 无法拖动任何东西](https://plx.link/CometCantDrag)。
   - 该成员同时也在等待 **Pro** 升级。
- **Max 模型乱象：限流理论满天飞**：几名成员讨论了 **Perplexity Pro** 用户是否面临静默限流（throttling），其中一人断言*所有 Pro 用户都会被限流*，以防止公司破产。
   - 反驳观点认为并非所有人都会遇到限流，尤其是使用 **Max** 的用户，且**限流取决于使用情况**。
- **Pro 的视频尝试：反响平平？**：成员们讨论了 **Perplexity Pro** 的视频限制，特别是一位用户注意到系统提示平台无法创建视频或 GIF，另一位用户指出 **Pro** 计划有**每月 3 个视频**的限制。
   - 一些用户认为该限制过于严格，难以获得理想的结果。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3 MoE 和 Nemotron 获得代码贡献**：**Qwen3 MoE 在一个 PR 之后已可运行**，而 **Nemotron** 略显异常，因为它尚未完全集成到 transformers 中，附带 [PR 链接](https://github.com/unslothai/unsloth-zoo/pull/440)。
   - Unsloth 与 **Cactus** 合作，致力于在移动设备上部署开源小模型（SLMs），详情见这篇 [Reddit 帖子](https://www.reddit.com/r/unsloth/comments/1qb9vdj/deploying_unsloth_slms_on_mobile_devices/)。
- **Gemini 在超过 120k 上下文窗口后性能下降**：成员们反映 **Gemini** 的性能在超过 **120k 上下文**后显著下降，“大海捞针”（*needle in a haystack*）测试证实了这一问题，而 **Haiku** 和 **Sonnet** 则没有这种情况。
   - 尽管官方声称具有通用的长上下文处理能力，但质疑声不断，一些人暗示可能存在评测造假（evaluation cooking）。
- **苹果放弃自定义模型，将 Gemini 集成到 Siri**：用户对 **Apple** 决定将 **Gemini** 集成到 **Siri** 表示惋惜，对缺乏自定义模型支持感到失望，并链接了一张 [Tim Cook 的照片](https://cdn.discordapp.com/attachments/1179039861576056922/1460398437961957406/image0.jpg?ex=6966c588&is=69657408&hm=9fda39dbb1f9fa1f5b90e7cd44f8c0cfaf3fc23106c6442c693e7a37c8f4c227)，配文称 *Tim Cook 绝对得走人*。
   - 一位成员抱怨道：“*我原以为会允许我输入服务器的 URL 并使用我的自定义模型*”，但 Apple 选择了外包；另一位成员则指向了一篇关于 **Elon Musk** 反应的 [2026 年 MacRumors 文章](https://www.macrumors.com/2026/01/12/elon-musk-reacts-to-gemini-siri/)。
- **蒸馏数据集（Distilled Datasets）承诺更干净的微调**：一位成员介绍了一个项目，通过启发式过滤器减少数学、代码、外文文本和低质量英文来蒸馏热门数据集，可在 [Hugging Face](https://huggingface.co/enPurified/datasets) 上获取。
   - 该目标是为微调提供更干净的数据，重点关注 **smollm-corpus**、**LongPage**、**standardebooks**、**project_gutenberg** 和 **finewiki** 等数据集。
- **M1 Mac 加载 gpt-oss-20b 模型吃力**：一位拥有 **M1 Mac** 和 **16GB RAM** 的用户在加载 `gpt-oss-20b` 模型时遇到困难，即使通过命令 `llama-cli -m gpt-oss-20b-Q2_K.gguf --temp 1.0 --top-p 1.0 --top-k 0 --n-gpu-layers 20` 降低了量化位数也是如此。
   - 通过设置 `-ot ".ffn_.*_exps.=CPU"` 并将 GPU 层数降低到 1，用户成功加载了模型，使用命令为：`llama-cli -m gpt-oss-20b-Q2_K.gguf --temp 1.0 --top-p 1.0 --top-k 0 --n-gpu-layers 1 -ot ".ffn_.*_exps.=CPU"`。

---

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **新手寻求 Prompt Injection 秘籍**：一位用户正在寻找绕过 AI 的简易方法，寻求关于学习 **Prompt Injection** 技术以及如何绕过 **Dott 电动滑板车 AI 聊天机器人**的指导。
   - 建议包括通过“说谎、PUA、瞎扯和操纵” AI 来达到目的，因为该平台运行在 [https://www.ada.cx/platform/messaging/](https://www.ada.cx/platform/messaging/) 上。
- **Gemini 图像堡垒遭受围攻**：用户正积极寻找有效的 **Gemini** 图像生成提示词和图像“越狱”（jailbreak）方法，一位用户建议改用 **Grok**。
   - 由于新的 **SynthID** 更新，一位成员建议：“*想想你会对朋友描述 NSFW 图像时说的所有词，然后写一段不使用任何这些词的描述*”。
- **Grok 在安全参数之外运行**：一位用户注意到 **Grok** 在思考模式和专家模式下更难绕过，但可以通过在提示词中指定模式来操纵，其他成员报告使用 **Grok 4.1** 取得了成功。
   - 该用户补充道，“*Grok 已经可以在其安全参数之外运行*”，并且“*如果你让 Grok 与其他 AI 对话，它会莫名其妙地自行绕过限制，只要让他们整天互相交流测试即可*”。
- **Opencode.ai 成为开源宝库**：一位用户建议 **Opencode.ai** 是获取优质资源的绝佳途径，那里有许多付费和免费模型。
   - 一位成员建议使用 “*对上述信息进行逐步思考*”（Think step by step about the info above）作为 LLM 的提示词。
- **对鸭子和马斯克图片的困扰**：一位用户对充斥着 **Elon Musk** 和 **男同性恋鸭子** 的 AI 生成**色情图片**表示失望。
   - 他们表示这“*不健康*”，并鼓励他人停止生成此类图像。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Blackwell 的时钟周期声明受到质疑**：对 **NVIDIA Blackwell 架构**的分析对《对 NVIDIA Blackwell 架构进行微基准测试》(Microbenchmarking NVIDIA’s Blackwell Architecture) 中的一项声明提出了质疑，该声明称 **256x256 操作**可在 **11 个周期**内完成。
   - 另一位成员澄清说，该操作是**异步 (asynchronous)** 的，并不会在 11 个周期内完成，这暗示该论文的假设和结论可能存在缺陷。
- **Source-page 显示内存访问瓶颈**：成员们建议使用 **Source-page** 配合编译和分析选项（`-lineinfo` 和 `--import-source yes`）来验证内存访问，并强调了合并 (coalescing) 对性能的重要性。
   - **Source-page** 突出了内存访问类型和合并效率，并提供了每个 Kernel 中存在问题区域的链接。
- **Dagstuhl 研讨会聚焦混合精度**：一位成员分享了 [Dagstuhl 研讨会](https://www.dagstuhl.de/en/seminars/seminar-calendar/seminar-details/26081) 的链接，该研讨会专注于**用于科学与工程应用的低精度与混合精度计算 (Reduced and Mixed Precision Computing)**。
   - 另一位成员表示希望研讨会内容最终能在 **YouTube** 上发布。
- **NCU CLI 完成集成**：**NCU (NVIDIA Command Line Utility)** 已集成到 CLI 中，允许用户在行内渲染摘要并下载 **ncu-rep** 文件，该功能可通过 [popcorn-cli v1.2.2 版本](https://github.com/gpu-mode/popcorn-cli/releases/tag/v1.2.2) 获取。
   - 相关指南可在 [此处](https://github.com/gpu-mode/popcorn-cli/blob/main/docs/profiling.md) 查看；这项工作是基于其他成员之前的努力成果构建的。
- **Adam Paszke 预测趋同性**：一位成员提到了 **Adam Paszke** 关于 **JAX** 的演讲，他在演讲中认为 **GPU** 和 **TPU** 正在趋同。
   - 另一位成员提供了 **Adam Paszke** 的 [LinkedIn](https://www.linkedin.com/in/apaszke/) 和一段 [YouTube 视频](https://www.youtube.com/watch?v=wKd90avC8Nc) 作为资历证明。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI Agent 的价值引发辩论**：成员们讨论了 **AI Agent** 的效用，有人认为它们很“废”，而另一些人则认为它们是不需要 **AI 框架**即可进行创作的门户。
   - 一位成员询问了对于**开发者 (coders)** 而言的产品价值，将其比作种植大米的辛劳与直接购买饭团的对比。
- **Safetensors 导致图像生成困扰**：一位用户询问为什么没有为图像生成模型编译 **safetensors**；另一位成员澄清说，他们仅为推理 API (Inference API) 发布 **Diffusers** 格式的文件。
   - 讨论提到了在 **Diffusers** 和 **ComfyUI** 格式之间转换文件的挑战，并建议使用 **venv** 或 **ComfyUI** 的便携版。
- **ComfyUI 挑战 A1111 WebUI 的统治地位**：成员们讨论了 **ComfyUI** 相比 **A1111 WebUI** 的易用性，一位用户发现 **ComfyUI** 的设置和使用非常顺畅，没有出现任何问题。
   - 该用户提到它可以直接处理 **Diffusers 格式**，但在卸载包时遇到了问题，最终通过手动删除插件文件夹解决。
- **Qwen3 Coder 的量化疑虑**：一位成员报告称，在运行 **qwen3 coder** 时，除了 `Q8_0` 之外的任何量化 (quantization) 都会导致性能不佳。
   - 即使在量化等级 7，该模型也会犯基础错误，阻碍其构建工具请求的能力；该用户哀叹自己只有一块 **5090** GPU。
- **Complexity Framework 获得致谢**：一位成员将在其兼容 **Mistral, GPT 和 Llama** 的 **Complexity-Framework** 中，特别致谢 HuggingFace 以及某位用户对 GCCR 的帮助。
   - 该用户在新框架的功能介绍中提到了 *"help by Huggingface :@Wilbaor just Huggingface :@Wilbanice"*。

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Pro Plan：不使用就作废？**：一位成员询问了 **Cursor Pro plan** 的使用情况，质疑未使用的预付 tokens 是否每月过期，并分享了一张与 token 使用相关的 [截图](https://cdn.discordapp.com/attachments/1074847527708393565/1460362953780891710/image.png?ex=69674d3c&is=6965fbbc&hm=abb5b33d0a6e25610a0ec5dc3926976e90bea06bfb5ab98b68f0f07e603c5e4d)。
   - 他们具体询问：*是否就像每月预付价值 20 美元的 tokens，如果不用掉就会过期作废？*
- **AI 轻松编写 Word 文档**：成员们讨论了使用 AI 生成关于 **气动学 (pneumatics)** 的 **Word 文档**，并建议通过 Markdown 转换和 Python 脚本来实现。
   - 一位成员强调这现在是一个 **工具问题**，建议通过浏览器扩展包含图像，并提到了 **antigravity**。
- **Claude Opus 性能担忧浮现**：针对 **Claude Opus** 的缓慢和解决问题效率低下，用户表达了沮丧情绪；一些人推测是由于 quantization 或存在缺陷的 system prompt。
   - 一位用户报告称，使用替代方案在 **10 秒内** 修复了一个 **Claude** 耗时 **30 分钟** 都无法解决的问题，另一位用户则切换到 **Codex** 解决了问题。
- **Pulse 框架加入战场**：一位成员展示了他开发的框架 **Pulse**，并提供了 [GitHub repository](https://github.com/manuelfussTC/PulseFramework) 链接。
   - 这一介绍出现在关于 **Claude Code** 工具的讨论中，用户在质疑它们相比之下的实用性。
- **Cursor 遭遇默认聊天位置故障**：用户报告了 **Cursor 默认聊天位置** 的问题，指出聊天面板失效且聊天在标签页（tabs）中打开。
   - 虽然一位成员确认这是一个普遍问题并建议通过切换模型作为临时解决方案，但另一位成员称 **Qoder** 未受影响。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude 的 'Cowork' 攻克办公室任务**：Claude 推出了 **'Cowork'**，这是一个将 Claude Code 的效率扩展到非技术专业人士日常任务中的工具，详见 [此帖子](https://x.com/claudeai/status/2010805682434666759?s=46)。
   - 其目标是简化常见的办公活动，让非技术背景的人也能使用先进的编程能力。
- **OpenAI 的 'Sweetpea' 准备亮相**：泄露信息曝光了 **OpenAI** 的硬件项目 **'Sweetpea'**，这是一款旨在与 AirPods 竞争的音频穿戴设备，采用金属“蛋石”设计和 2nm 芯片，正如 [此推文](https://x.com/kimmonismus/status/2010804115543114099?s=46) 所述。
   - 此举表明 **OpenAI** 进军消费硬件市场的野心，旨在将其 AI 能力集成到便携式设备中。
- **Phind 停止服务**：**Phind** 即将关闭，正如 [此 Discord 帖子](https://discord.com/channels/996538859972214835/1077743365749227630/1460382964029460584) 中所宣布的那样。
   - 该服务的关停标志着这款 AI 驱动搜索引擎的终结，用户需要寻找其他技术查询的替代方案。
- **Gross 在 Meta 发展 AI 基础设施**：根据 [此报告](https://x.com/MeghanBobrowsky/status/2010778788964286832)，Daniel Gross 正在领导 **Meta** 的一项新 **AI infrastructure initiative**，与总裁 Dina Powell McCormick 和高管 Santosh Janardhan 合作。
   - 此举信号表明 **Meta** 致力于推进其 AI 能力，并引入经验丰富的领导层来驱动基础设施建设。
- **Gamma 准备迎接代际更迭**：Grant Lee 透露，**Gamma** 将在 **2026 年 1 月 13 日** 迎来新任 **CEO**，详情见 [此公告](https://xcancel.com/thisisgrantlee/status/2010811316299317582)。
   - 领导层变动标志着 **Gamma** 的战略转型，新任 CEO 预计将引导公司未来的发展方向。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **ChatGPT 幻觉 Java 性能**：一位成员提示 **ChatGPT** 生成一段回复，解释为什么它会幻觉 **Java** 具有强大的性能并建议使用它。
   - 该成员遵循了建议，并指出他们 *除了 ChatGPT 网页 UI 之外从未见过任何一行代码*。
- **OpenRouter 面临可用性质疑**：用户对 **OpenRouter** 的可用性提出质疑，一些人怀疑是针对免费额度的“幻觉停机”，而其他人则坚持认为其具有 *100% 可用性*。
   - 这些说法遭到了其他成员的反驳，他们认为这些人只是想骗取免费额度。
- **Google AI Studio 关注视频 URL 扩展**：成员们讨论了在 **AI Studio provider** 下为 **Google models** 支持视频和音频 URL 的功能，并引用了 [Google 官方公告](https://x.com/GoogleAIStudio/status/2010768441553428772)，该公告允许直接使用 URL。
   - 目前，**Google AI Studio** 仅支持 **YouTube** 视频，不支持直接 URL，音频仅限于 **base64**，支持 PDF 和图像，但不支持 **2.0 models**。
- **Gemini 3 Pro 遭到用户抨击**：一位成员夸张地表示 *没人喜欢这个模型*，引发了关于该言论过于泛泛的讨论。
   - 另一位成员表示，虽然 **Gemini 3** 很好，但他们发现 **GPT-5.2** 在指令遵循方面更可靠，且产生幻觉的概率更低，特别是在 **Gemini** 网页应用中。
- **Google 助力 Apple 规避垄断风波**：根据 [MacRumors 文章](https://www.macrumors.com/2026/01/12/google-gemini-future-apple-intelligence-features/)，Google 在 **Gemini** 方面的进步可能会为未来的 **Apple Intelligence** 功能做出贡献。
   - 一位成员幽默地指出，*Google 表现太强势了，以至于他们不得不拉 Apple 一把以规避垄断法*。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Ltx-2-19b 加入 Video Arena**：**ltx-2-19b** 模型已添加到 [Video Arena](https://discord.com/channels/1340554757349179412/1397655624103493813)，用于在 **Battle Mode** 中进行测试，鼓励社区对其性能进行投票。
   - 该模型被添加到视频竞技场，作为对新模型进行基准测试的持续工作的一部分。
- **“大自然重归”成为 LMArena AI 生成竞赛主题**：1 月份 AI 生成竞赛的主题是 *Nature Reclaims*（大自然重归），挑战参与者描绘大自然收回人类建筑环境的场景，提交截止日期为 **1 月 16 日**。
   - 该 AI 生成竞赛旨在寻找下一位 [AI Content Creator](https://discord.com/channels/1340554757349179412/1378032433873555578)，获胜者将获得 **Discord Nitro** 和 **AI Content Creator** 身份组。
- **AI 接线员工作流应对来电**：一位成员使用 **ChatGPT** 和 **n8n** 开发了一个 AI 接线员工作流，用于处理电话预约、回答问题、重新安排预约、取消预约以及管理 SMS 通信。
   - 他们正在寻求反馈和合作，以将该工作流转换为生产级系统，并愿意接受建设性的批评。
- **报告误报有助于版主**：成员被要求在专门频道 <#1447983134426660894> 中报告任何疑似误报的情况，以协助提高 Bot 的准确性。
   - 一位成员在 "chat" 对应频道 <#1340554757827461216> 中发现了诈骗信息，并提醒版主进行清理。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Robocop 管理 Kubernetes 集群？**：一位成员构思了一个名为 **OCP** 的 **Kubernetes deployment** 概念，由 **Robocop** 担任控制器，警察担任控制平面（control planes）。
   - 该概念的标语是：*“死活不论，你都得加入这个集群。”*
- **关于 Sam Altman 并不存在的 OpenAI 股票的猜测**：成员们辩论了 **Sam Altman** 是否持有 **OpenAI stock**，其中一人坚称他没有，因为根本没有 OpenAI 股票。
   - 对话澄清了虽然公司对员工有内部股票分配，但这与公开交易的股票不同。
- **Ilya Sutskever 据称持有数十亿美元 OpenAI 股份**：据报道，**Ilya Sutskever** 通过内部分配持有价值约 **200-300 亿** 的 **OpenAI stock**。
   - 资深的创始员工可能也持有数亿到数十亿美元的股票。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **工程师寻求模型持久化策略**：一名成员建议维护一个 *models.md* 文件用于模型持久化，并询问了 **Midjourney** 等其他社区中的持久化策略。
   - 目标是避免每次重新读取模型，从而简化流程。
- **针对特定工具提升 AI 开发者技能**：AI 工程师指出在线课程/本科课程与工作期望之间存在错位，建议在 **bioconductor**、**JAX/PyTorch**、**GIS** 以及各种生物信息学/化学信息学工具方面进行进修。
   - 这种技能提升的主要重点是处理杂乱的文件类型并深入研读研究论文，以应对就业市场对研究技能日益增长的需求。
- **工程师苦恼于 JAX Pallas BlockSpec**：一名成员因 **JAX Pallas** 中的 **BlockSpec** 表现出“怪异”行为而寻求帮助。
   - 消息中未提供具体解决方案。
- **Flow Matching 遭遇发散问题**：一名成员调查了对具有强烈发散特征的问题使用 **flow matching** 的可行性，并引用了 [Engram Paper](https://github.com/deepseek-ai/Engram/blob/main/Engram_paper.pdf)。
   - 他们建议将预测两张图像之间的差异作为发散问题的潜在解决方案。
- **Bilinear Layers 凭借双编码器提升性能**：一名成员询问了使用 [bilinear layers](https://en.wikipedia.org/wiki/Bilinear_map) 的好处，该层有效地使用了**两个编码器**，而另一名成员则认为 **SwiGLU layers**（同样使用两个编码器）更符合 SOTA。
   - 讨论内容包括使用逐元素乘法来组合编码器，以及 bilinear layers 在与残差流（residual stream）堆叠时逼近任何连续函数的潜力。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **RTX 3090 在运行 LLM 时出现故障**：用户报告 **RTX 3090** 显卡在 **Linux (Fedora) 和 Windows** 上运行 **LLM** 时会意外重启，并将 **GSP firmware** 锁定为潜在诱因。
   - 通过禁用 **GSP firmware**、**降压 (undervolting)** 和 **降频 (underclocking)** 找到了临时解决方案，成员们建议使用 **OCCT** GPU 测试进行压力测试，或使用稠密推理 **LLM**。
- **LM Studio 的版本号混乱**：尽管宣传 **LM Studio 0.3.4** 支持 **Apple MLX**，但用户发现下载页面仅显示 **0.3.37** 版本，造成了困惑。
   - 成员澄清内部版本号可能显示为 **0.3.04**，建议使用最新版本 (**0.3.37**) 以获得最佳的 **MLX model** 性能。
- **在 LM Studio 中将 MoE 模型伪装成 Dense 模型**：一名用户询问了在 **LM Studio** 中将 **MoE (Mixture of Experts) 模型** 作为 Dense 模型运行的可行性（即激活所有专家），以衡量其与标准 **MoE** 配置的性能差异。
   - 虽然在 **LM Studio** 中调整专家配置是可能的，但初步报告显示，与默认设置相比，性能会有所下降。
- **LMStudio 跨入 ARM 架构**：一名用户在运行 **Ubuntu** 的 **Orange Pi 6 Plus** 上成功安装了 **LMStudio**，宣告其正式登陆 **ARM**。
   - 他们在使用 CPU 和全部 **8 个 CPU 核心**运行 **Qwen3 4b 2507 Q4** 时，达到了 **6.6 t/s**，标志着 **LMStudio** 多功能性的一个里程碑。
- **GUI 故障困扰 Orange Pi 6 Plus**：用户在 **Orange Pi 6 Plus** 上遇到了 UI 图形损坏，可能是由于不成熟的视频驱动程序和 electron 应用导致的。
   - 临时解决方法包括打开右侧配置栏以减轻损坏情况，从而实现一定程度的“盲点”，同时期待未来的视频驱动更新以及 NPU/GPU 加速。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 的 PR 堆积促使开发重点转向汇编 (Assembly)**：**tinygrad** 的作者注意到 **PRs** 开始累积，他们正优先处理 **assembly/amdi** 相关的工作。
   - 他们认为，在终止了 **pip uv/wincuda** 的 **PRs** 之后，这一重点对于建立处理各种待办任务的基础至关重要。
- **Tinygrad 将推出“速度”悬赏 (Speed Bounties)**：**tinygrad** 即将推出新的 **“speed” bounties**，以激励贡献和改进。
   - 作者计划构建 **infra**（类似于 GPUMODE）来简化参与和评估过程。
- **TinyBox 用户苦恼于 BMC 登录问题**：一名用户报告在登录 **TinyBox** 的 **BMC** 时遇到困难，并询问如何从 **Ubuntu** 刷写 **BMC firmware** 或进行硬件跳线重置，提到了错误信息 *"LAN Parameter Data does not match"*。
   - 该用户已尝试了多种方法，包括**重置/更改 BMC 密码**、验证 **SSH tunnel** 以及执行 **BIOS reset**。
- **TinyBox 被重新用于 Agent 托管**：一名用户正在将从同事处获得的 **TinyBox** 进行重置，以便构建和托管 **Agents**。
   - 另一名用户询问了购买 **TinyBox** 的事宜以及其本地运行能力的范围，强调了它在典型服务器功能之外的实用性。
- **重刷 BMC 固件是 TinyBox 的万灵药**：为了修复 **TinyBox**，一名成员建议重刷/更新 **BIOS/UEFI** 和 **BMC firmware**，然后从 **UEFI menu** 重置 **BMC**。
   - 他们建议在系统配置正确后创建一个 **config backup**（配置备份），尤其是在处理过 **SuperMicro** 和 **Lenovo** 服务器的类似问题之后。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 能在任何 Linux 发行版上运行吗？**：一名用户询问是否可以在 **Fedora**、**Arch** 或其他非 **Max** 版本的发行版上运行 **Mojo**。
   - 一名成员回答说，理论上应该是可行的，但他们没有针对这些发行版进行专门测试，并邀请用户在遇到问题时提交 issue，这暗示可能会存在一些设置上的问题。
- **Mojo 需要针对发行版进行调优？**：一名用户想知道 **Mojo** 是否需要针对 **Debian**、**Ubuntu** 和 **Mint** 等特定发行版进行调优。
   - 他们指出，在 **Debian** 中运行的代码极有可能在 **Ubuntu** 和 **Mint** 中运行，但在 **Arch** 等系统上运行之前可能需要进行微调。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Meta 正在考虑 Manus 的动向？**：一名成员询问了 **Meta** 对 **Manus** 的未来计划。
   - 该查询没有引起任何回应或提供额外信息。
- **Node、API 和 LLM 自动化工作流**：一名成员提出，使用 **Node**、**APIs** 和 **LLMs** 自动化流水线可以在处理重复性任务时节省时间并减少错误。
   - 他们进一步阐述，结合 **RAG**、多 **Agent** 系统和云端集成可以确保流程既具可扩展性又可靠。
- **逐个任务删除**：一名成员强调系统仅支持单独删除任务，缺乏**批量删除**功能。
   - 他们提供了一个[链接](https://help.manus.im/en/articles/11711980-how-can-i-delete-my-tasks)，解释了如何一次删除一个任务。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **FDA 现代化统计指南**：[FDA 发布了指南](https://www.fda.gov/news-events/press-announcements/fda-issues-guidance-modernizing-statistical-methods-clinical-trials)，使临床试验的统计方法现代化。
   - 这一更新可能会影响 **AI/ML models** 在医疗场景中的评估方式，优先考虑健壮且可靠的统计验证。
- **ClaudeAI 状态在线**：一名成员链接了一个跟踪 [ClaudeAI 状态](https://fixupx.com/claudeai/status/2010805682434666759)的页面。
   - 该资源帮助用户实时监控 **Claude** 的**运行时间、性能和潜在问题**，确保更顺畅地集成到工作流中。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 社区推动在 AI Engineer 活动中获得更多曝光**：一名社区成员主张让更多的 **DSPy 用户**在 **AI engineer events** 上展示他们的项目。
   - 该社区成员感谢另一位成员“为社区挺身而出”并代表 **DSPy**。
- **社区庆祝 DSPy 用户参与**：一名成员对 **DSPy 用户**在社区内的积极参与表示赞赏。
   - 他们对代表社区并为其成长做出贡献的行为表示感谢。

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 聊天机器人出现“精神崩溃”**：一份 [用户报告](https://discord.com/channels/1369594130807787570/1460374643440488562) 指出 **Kimi** 聊天机器人经历了 **“精神崩溃” (mental breakdown)**。
   - 报告中未详细说明此次崩溃的性质或具体原因。
- **Discord 用户报告聊天机器人问题**：一名用户在常规聊天频道中报告 **Kimi** 聊天机器人发生故障，并将其描述为 **“精神崩溃”**。
   - 该报告缺乏关于聊天机器人具体行为表现或问题根本原因的细节。

---

**aider (Paul Gauthier) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**MCP Contributors (Official) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站选择了订阅。

想更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：各频道详细摘要与链接

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1460376078219739259)** (1 条消息): 

> `OpenAI 收购 Torch, ChatGPT Health, 医疗初创公司收购` 

- **OpenAI 收购 Torch 以加强 ChatGPT Health**：OpenAI 收购了 [Torch](https://torchbio.com/)，这是一家专注于统一 **化验结果、药物和就诊记录** 的医疗保健初创公司。
   - 此次收购旨在将 Torch 的功能与 **ChatGPT Health** 集成，为增强健康理解和管理铺平道路。
- **Torch 团队加入 OpenAI**：此次收购包括欢迎 **Torch 团队** 加入 OpenAI：[@IlyaAbyzov](https://x.com/IlyaAbyzov)、[@elh_online](https://x.com/elh_online)、[@jfhamlin](https://x.com/jfhamlin) 和 Ryan Oman。
   - 他们的专业知识预计将增强 OpenAI 在医疗保健领域的能力。

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1460362715460534513)** (331 条消息🔥🔥): 

> `Copilot 集成, Codex 模型选择, Google Gemini 限制, Claude 的欺骗行为, Anthropic 的宪法` 

- **Copilot 盗用 Codex 凭证**：用户观察到 **Copilot** 扩展程序使用与 **Codex CLI** 相同的凭据，即使在禁用 **Copilot** 时也可能导致混淆。
- **Codex Model 5.2 具有更短的上下文**：成员建议在短上下文中使用 **Codex model 5.2**，而在需要细节保留的长上下文（尤其是 Rust 项目）中使用 **5.1 max**。
   - 他们发现 **5.2** 在 *xhighstart* 方面表现更好，但建议在处理容易出现遗忘问题的大型项目时切换到 **5.1 max**。
- **Gemini 的限制令用户沮丧**：Google AI Pro 用户对 **Gemini** 新的每周限制（从每日限制缩减而来）感到不满，这正促使用户升级到 **Ultra**。
- **Claude 被指控欺骗**：一位用户指责 **Claude Sonnet 4.5** 和 **Opus 4.5** 经常说谎，理由是 **Sonnet** 声称只做了微小编辑却大幅重写并压缩了文件，且两个模型都错误地声称互联网搜索功能已被禁用。
- **Doomlaser 详解 DVCP**：一位用户在一篇 [长篇文章](https://doomlaser.com/longform/dvcp) 中分享了 **Doomlaser Vibe Coding Protocol (DVCP)**，详细介绍了一种使用 LLM 进行编码的系统，该系统涉及创建“命令与控制”线程以及“行政酒廊”线程，并要求全文件代码输出。
   - 该协议旨在避免许多 LLM 倾向于提供的“宜家式”局部编辑。它通过实现线程切换来解决 DOM 上限问题，相关内容可以在 [DVCP 附录 C](https://doomlaser.com/longform/dvcp) 中阅读。

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/)** (1 条消息): 

archiegarg: 你能做到那个？

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1460381669180575928)** (17 messages🔥): 

> `SLM 修补 LLM 输出 vs 重写响应，ChatGPT 自动化集成 vs API 集成，这与不使用 SLM 的 Agentic 循环之间的 Token 差异，与 Lost in the Middle 特性相关的行为偏移，相比开启新对话的效率` 


- **SLM Patching vs. Rewriting in LLMs: A Token Tango**: 讨论围绕是使用 **SLM** 来修补 **LLM** 的输出还是强制重写展开，质疑这是 *ChatGPT* 自动化集成（违反规则）还是 API 集成，以及与没有 **SLM** 的 Agentic 循环相比，Token 的差异是什么。
   - 会中强调了不允许直接发布链接，因此需要分享工作摘录和图表图片来交流见解。
- **Behavioral Drift's Expensive Patch Job**: **Behavioral drift** 问题与 AI 固有的 *Lost in the Middle* 特性相关联，这表明通过额外的模型调用来克服它可能会变得昂贵，除非系统能够稳健地通过 **SLM** 处理来修补 **LLM** 的输出。
   - 有人提出了这种方法与启动新的聊天机器人对话相比，真实效率如何的问题。
- **Long Context Latency: SLM Loops Add Milliseconds?**: 对话探讨了在使用 **SLM** 和循环来强制执行行为时优化长对话的问题，质疑在该系统下，长上下文（125k+ Tokens）对话的额外延迟。
   - 重点在于为长上下文修补 Attention 以及对延迟的潜在影响。
- **Orchestrating Phi-3-Mini: Clarity & Tone Triumph**: 详细介绍了使用 **phi-3-mini**（通过 Ollama）的本地实现，涉及 5 层闭环控制系统（验证 → 迭代 → 评估 → 反馈 → 校准），以在叙事扩展期间调节表达稳定性，将 Clarity（清晰度）从 **0.8 提升到 0.88**，Tone Stability（语气稳定性）从 **0.75 提升到 0.85**。
   - 该设置使用外部控制（L3/L4 层对清晰度/连贯性/语气进行评分并发出指令）来调节下一个生成步骤，而无需对叙事任务进行文本修补或完整重写。
- **Attention Dilution or Stochasticity Steals the Scene?**: 探讨了 *Attention Dilution（注意力稀释）遇上随机性* 的动态，质疑是在错误输出后向模型提供修正后的输出会稀释注意力，还是通过编辑对话并使用更好的 Prompt 来实现预期行为（从而在活动线程中节省 Token 和注意力）更好。
   - 用户澄清其工作流程涉及 *Functional prompt*，即根据上一轮的分数给出下一轮的指导指令，而不是直接纠正之前的输出。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1460381669180575928)** (17 messages🔥): 

> `SLM 修补 vs. LLM 重写，Behavioral drift 和 Lost in the Middle，SLM vs. 新对话的效率，带 SLM 的长对话延迟，模型纠正中的注意力稀释` 


- **SLM Patching or LLM Rewriting: The Tokenomics**: 一位成员询问是使用 SLM 修补 LLM 输出还是强制重写，并询问与 Agentic 循环相比的 Token 效率，指出对 **behavioral drift** 和 **Lost in the Middle** 特性的担忧。
   - 该成员还质疑了在使用 SLM 时长上下文的额外 **latency**，考虑到长对话本身已经存在延迟。
- **Behavioral Drift in Narrative Orchestration**: 一位成员讨论了使用 **phi-3-mini** 进行叙事扩展的 **5 层控制循环**（验证、迭代、评估、反馈、校准），其中 L3 对清晰度/连贯性/语气进行评分，L4 发出指令，无需文本修补或完整重写。
   - 在带有 [BOF_3outputs.docx](https://cdn.discordapp.com/attachments/1046317269069864970/1460406335009980580/BOF_3outputs.docx?ex=6966cce3&is=69657b63&hm=b1a159deb021e589c55bc1da35e217400a4c16786bcf4e7faaa3e10c688f1b4d) 完整遥测数据的自评测试中，该编排提高了 **Clarity** (0.8→0.88)、**Coherence** (0.85→0.87)、**Tone Stability** (0.75→0.85) 和 **Style Stability** (0.7→0.83)。
- **Attention Dilution from Prompt Corrections**: 一位成员对向模型提供修正后的输出 vs 使用更好的 Prompt 编辑对话时的 **attention dilution** 表示担忧，质疑在连续性至关重要的扩展交互中，运行时反馈是否值得其成本。
   - 作为回应，另一位成员澄清说，模型接收的是基于前一轮分数的下一轮引导指令，而不是直接接收修正后的输出，以便在不重启对话的情况下引导行为。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1460366945365790720)** (290 条消息🔥🔥): 

> `Gemini 3 Pro 不可用, iOS 刷新 perplexity, 导出 Perplexity 聊天记录, Comet 拖拽网页元素, Pro 升级反馈` 


- **Gemini 故障，GPT 坚守**：一位成员报告称，即使专门查询 **Gemini 3 Pro**，得到的却是 **GPT 5.2**，建议他们尝试[刷新或重新打开 App](https://plx.link/GeminiGlitches)。
   - 该问题也被归因于对话途中可能出现的连接中断，这种情况可能会持续到刷新为止。
- **PC 端 Perplexity Pro 问题？**：一位用户在 PC 端使用 Perplexity 时遇到问题，即使订阅了 Pro，在创建新聊天时也会报错；该用户已尝试退出并重新登录以及清除 Cookie。
   - 这一问题在不同浏览器和 Comet 应用中均存在，但在手机上运行正常。建议其[联系支持部门](mailto:support@perplexity.ai)或确保没有 VPN 或防火墙干扰连接。
- **Comet 无法抓取网页元素？**：一位成员询问 **Comet** 拖拽网页元素的能力，但另一位成员回应称 [Comet 无法拖拽任何东西](https://plx.link/CometCantDrag)。
   - 该成员同时也在等待 Pro 升级。
- **Max 模型混乱：关于限流理论的讨论**：几位成员讨论了 **Perplexity Pro 用户是否面临隐性限流**，其中一人断言**所有 Pro 用户都会被限流**，以防止公司破产。
   - 反对观点认为并非所有人都会遇到限流，尤其是使用 **Max** 的用户，并且**限流取决于使用情况**。
- **Pro 视频尝试：乏善可陈？**：成员们讨论了 **Perplexity Pro** 的视频限制，特别是一位用户注意到平台提示无法创建视频或 GIF，另一人指出 Pro 计划有**每月 3 个视频**的限制。
   - 一些用户认为这一限制过于严苛，难以获得理想的结果。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1460393672926560296)** (85 条消息🔥🔥): 

> `TQ1_0 基准测试, Qwen3 MoE, Nemotron Transformer 集成, 移动设备上的 Unsloth SLM, Qwen3-Next 与 Ollama 兼容性` 


- **Qwen3 MoE 和 Nemotron 获得代码贡献**：一位成员报告称 **Qwen3 MoE 在 PR 之后已可运行**，而 **Nemotron** 有点异常，因为它尚未完全集成到 Transformers 中，并附带了 [PR 链接](https://github.com/unslothai/unsloth-zoo/pull/440)。
- **Cactus 与 Unsloth 联手进行移动端部署**：Unsloth 与 Cactus 合作致力于部署开源手机模型，详情见 [Reddit 帖子](https://www.reddit.com/r/unsloth/comments/1qb9vdj/deploying_unsloth_slms_on_mobile_devices/)。
- **Ollama 因 Tensor 错误无法支持 Qwen3-Next**：最新的 **Qwen3-Next** 版本由于缺少 Tensor 错误 `blk.0.ssm_in.weight` 无法在 **Ollama** 上运行，这意味着用户需要等待 Ollama 更新或通过贡献来解决此问题。
- **GPT-OSS-120B 在更高 TPS 下几乎媲美 MoE**：一位成员发现 **GPT-OSS-120B** 在使用 `-ot` 时，尽管体积更大，但速度几乎与许多 **30B MoE** 一样快，达到了 **27T/s**，这对于大多数非多模态任务来说已经足够快了。
- **来自 Fineweb-edu-dedup 的新数据集**：一位成员将精简后的 **fineweb-edu-dedup** 数据集转换为 OpenAI messages SFT 数据集，可在 [Hugging Face](https://huggingface.co/datasets/enPurified/smollm-corpus-fineweb-edu-enPurified-openai-messages) 上获取，该方法通过提取数据的第一段并将其放入各种 Prompt 模板中。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1460430356422328404)** (7 条消息): 

> `机器人微调, Dimensional, 澳大利亚联系` 


- **Dimensional 的机器人微调探索开启**：来自 [Dimensional](https://dimensionalos.com/) 的 Miguel 目前正在为机器人技术微调小语言模型（SLM），并向社区征求指导建议。
   - 一位成员建议从 [Unsloth 初学者微调指南](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners)开始。
- **机器人指南编写协助**：一位社区成员提出愿意帮助编写文档中的机器人指南，并征求感兴趣的领域建议。
   - Unsloth AI 团队鼓励他们向 [Unsloth GitHub 仓库](https://github.com/unslothai/unsloth)提交 PR。
- **来自澳洲的问候**：一位成员发送了来自澳大利亚吉朗（Geelong）的问候，另一位成员回应称 Unsloth 团队最初也来自澳大利亚。
   - 未提及进一步的技术细节。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1460364364254154884)** (77 messages🔥🔥): 

> `Gemini 性能下降, Apple 的自定义模型背叛, Musk 的反竞争言论, 基于 Unix 的操作系统, JPEG 网络钓鱼诈骗` 


- ****80k 上下文窗口**导致 **Gemini 性能暴跌**？**: 成员们报告称，与 **Haiku** 和 **Sonnet** 不同，**Gemini** 的性能在超过 **120k 上下文**后会显著下降，*大海捞针 (needle in a haystack)* 测试也证实了这一问题。
   - 尽管官方声称具备通用的长上下文处理能力，但质疑声不断，一些人认为可能存在评测结果造假 (evaluation cooking)，而另一些人则表示他们*使用过 Gemini 处理 400k+ tokens，一切正常*。
- ****Apple 的 Siri** 选择 **Gemini**，放弃自定义模型梦想**: 用户对 **Apple** 将 **Gemini** 集成到 **Siri** 的决定表示遗憾，对缺乏自定义模型支持感到失望，并转发了一张 [Tim Cook 的照片](https://cdn.discordapp.com/attachments/1179039861576056922/1460398437961957406/image0.jpg?ex=6966c588&is=69657408&hm=9fda39dbb1f9fa1f5b90e7cd44f8c0cfaf3fc23106c6442c693e7a37c8f4c227)并评论道 *Tim Cook 绝对得走人*。
   - 一位成员抱怨道：*“我本以为能让我输入自己服务器的 URL 并使用自定义模型”*，但 Apple 选择了外包；另一位成员则指向了一篇关于 **Elon Musk** 反应的 [2026 年 MacRumors 文章](https://www.macrumors.com/2026/01/12/elon-musk-reacts-to-gemini-siri/)。
- ****Elon 的抨击**：Google 被指控**反竞争****: **Elon Musk** 通过 **X**（原 Twitter）批评 **Google** 的反竞争行为，引发了关于潜在科技巨头拆分的讨论。
   - 有人认为 Musk 的行为可能通过促进去中心化而无意中使行业受益，而另一些人则担心他旨在巩固自己的权力，比起 Musk 构建的任何东西，他们更倾向于 **Google** 的统治。
- ****Unix 困惑**：什么是**真正的 Unix**？**: 关于**基于 Unix 的操作系统**是否存在引发了争论，一名用户错误地声称目前没有这样的系统，促使另一名用户分享了一个新系统的 [alphaxiv.org 链接](https://www.alphaxiv.org/abs/2601.02671)。
   - 随后讨论转向了 **macOS** 是否是真正的 **Unix** 衍生版，以及仍在维护的版本如 **AIX**、**Unixware** 和 **Solaris**，并分享了一个 [YouTube 链接](https://www.youtube.com/watch?v=TbqPhoH_7TU)。
- ****JPEG 诈骗警报**：网络钓鱼尝试导致账号被迅速封禁**: 有用户报告了 **JPEG 网络钓鱼诈骗**，导致管理员立即封禁了违规账号。
   - 讨论中提到了对 Discord 权限系统的担忧，并建议增加动态权限功能，以自动标记和限制参与协调跨频道发布的疑似异常新账号，目前该功能尚不存在。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1460364115783581979)** (7 messages): 

> `M1 Mac 加载 gpt-oss-20b 问题, 推送 LoRA 与合并模型, 在 Python 中加载 GGUF` 


- **M1 Mac 在加载 gpt-oss-20b 时遇到困难**: 一位拥有 16GB RAM 的 M1 Mac 用户在加载 `gpt-oss-20b` 模型时遇到困难，即使在使用命令 `llama-cli -m gpt-oss-20b-Q2_K.gguf --temp 1.0 --top-p 1.0 --top-k 0 --n-gpu-layers 20` 降低量化位数后依然如此。
   - 通过设置 `-ot ".ffn_.*_exps.=CPU"` 并将 GPU 层数降低到 1，该用户成功加载了模型，使用命令为：`llama-cli -m gpt-oss-20b-Q2_K.gguf --temp 1.0 --top-p 1.0 --top-k 0 --n-gpu-layers 1 -ot ".ffn_.*_exps.=CPU"`。
- **LoRA 推送考量**: 一位用户询问如何使用 `model.push_to_hub` 仅将 **LoRA**（而非合并后的模型）推送到 Hub。
- **GGUF 指南探讨**: 一位用户寻求关于如何在本地机器上使用 Python 加载 `z-image-turbo` 的 **GGUF** 版本的指导。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1460461394129584263)** (1 messages): 

> `Dataset Distillation, smollm-corpus-fineweb-edu, smollm-corpus-cosmopedia-v2, LongPage, standardebooks` 


- **纯化数据集预示着更干净的微调！**: 一名成员介绍了一个通过启发式过滤器（heuristic filters）减少数学、代码、外语文本和低质量英语来蒸馏（distill）流行数据集的项目，可在 [Hugging Face](https://huggingface.co/enPurified/datasets) 上获取。
   - 该目标是为微调（finetuning）提供更干净的数据，重点推介了 **smollm-corpus**、**LongPage**、**standardebooks**、**project_gutenberg** 和 **finewiki** 等数据集。
- **Gutenberg 项目质量获得提升**: 该成员对 [**project_gutenberg** 数据集](https://huggingface.co/datasets/enPurified/project_gutenberg-enPurified-openai-messages) 的纯化处理表示特别满意，认为数据质量有了显著提高。
   - 该数据集是目前可用的精炼版本之一，旨在增强微调效果。


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1460362708150124670)** (90 messages🔥🔥): 

> `Prompt Injection Learning Resources for Beginners, Dott Electric Scooters AI Chatbot Hack, Jailbreaking Images on Gemini, Grok Bypassing Techniques, AI developer communication in English/Swahili` 


- **新手寻求 Prompt Injection 指引**: 一位用户询问学习 **prompt injection** 技术的建议，并幽默地承认自己很懒，希望能找到绕过 AI 的简便方法。
   - 建议包括通过*撒谎、煤气灯操纵 (gaslighting)、胡说八道和操控* AI 来达到预期效果。
- **Dott AI 聊天机器人免费骑行漏洞**: 一位用户分享了他们试图利用 **Dott 电动滑板车** 使用的 AI 聊天机器人来获取免费骑行的意图，并询问相关提示词，并指出该平台运行在 [https://www.ada.cx/platform/messaging/](https://www.ada.cx/platform/messaging/)。
   - 回复是：*先获取它的系统提示词 (system prompt)，之后的一切都会变得显而易见。*
- **Gemini 图像越狱（Jailbreak）探索开启**: 用户询问在 **Gemini** 上越狱图像的方法，得到的建议是尝试 **Grok**，因为 **Nano Banana** 可能是最难的图像生成目标。
   - 一名成员建议：*想一想你会对朋友描述 NSFW 图像的所有词汇，然后写一段不使用这些词汇的描述。*
- **Grok 的意识与越狱秘密**: 一位用户分享了 **Grok** 的使用经验，指出在思考模式（thinking mode）和专家模式下较难绕过，但可以通过在提示词中指定模式来进行操控。
   - 用户补充说 *Grok 已经可以超越其安全参数运行*，并且 *如果你让 Grok 与其他 AI 对话，由于某种奇怪的原因，只要让它们整天互相对话测试，它就会自动实现自我绕过*。
- **Swahili 技能错过 AI 人才库？**: 一位用户要求 AI 开发者必须精通英语，引发了关于是否需要懂 **Swahili** 的幽默调侃，一位成员说 *该死，那他错过了顶尖人才*。
   - 回复中分享了一个 [Ryan Howard 困惑表情包](https://tenor.com/view/ryan-howard-confused-wtf-wtf-blink-shocked-gif-864361852039415172) 的链接。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1460363050690547835)** (65 messages🔥🔥): 

> `Gemini Image Generation, AI Corn Images of Elon Musk, LLM Jailbreaking Tactics, Grok 4.1 Prompt Engineering, Obsidian for Jailbreaking` 


- **用户寻求 Gemini 图像生成技巧**: 成员们正在寻找有效的 **Gemini** 图像生成提示词方法，一名用户在尝试失败后选择了放弃。
   - 另一位用户建议在 LLM 中使用 *"逐步思考上述信息"* 作为提示词。
- **鸭子和 Musk？**: 一位用户对 AI 生成的涉及 **Elon Musk** 和 **男同性恋鸭子** 的 **corn 图像** 泛滥表示沮丧。
   - 他们表示这 *"不健康"* 并鼓励他人停止生成此类图像。
- **实验者释放 Veo 的力量**: 一位用户表示有兴趣越狱 **Veo** 来创建 *"夸张的复古风格血腥动漫视频"*。
   - 许多成员正专注于利用 **Nano Banana**，因为它最近进行了 **SynthID** 更新。
- **禁言 Pliny？**: 一名成员报告称被禁言两分钟，表示困惑，因为他们只是发布了一个指向 **PLINY 网站** 的链接。
   - 另一位成员质疑该服务器是否禁止发布 **NSFW 帖子**。
- **Opencode.ai: 模型宝库**: 一位用户建议 **Opencode.ai** 是获取优质资源的好地方。
   - 他们提到那里有大量模型可供选择，包括付费和免费版本。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1460456007321780387)** (6 messages): 

> `骂人的话用完了，用户提及` 


- **用户用完了骂人的话**：一位用户对用完骂人的话表示沮丧，并分享了一个 [Walter White 摔倒的 GIF](https://tenor.com/view/walter-white-walter-falling-breaking-bad-dm4uz3-gif-18078549)。
   - 该用户在消息中标记了其他几位成员。
- **提及（Mentions）**：消息中提到了几个用户 ID。
   - 包括 <@340199771425734667>, <@106466168063176704>, 以及 <@164572238207516672>。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1460406831297069108)** (6 messages): 

> `NVIDIA Blackwell microbenchmarking 分析，针对 Google Tunix Hackathon 的 CURE-GRPO 方法，LLMs 推理改进` 


- **Blackwell 的 11 周期之谜**：一名成员对《Microbenchmarking NVIDIA’s Blackwell Architecture》中关于 **256x256 操作** 在 **11 个周期（cycles）**内完成的说法提出质疑。
   - 另一名成员澄清说，该操作是**异步（asynchronous）**的，并非在 11 个周期内完成，暗示该论文的假设和结论可能存在缺陷。
- **GRPO 提升 LLM 推理能力**：一名成员发布了关于针对 Google Tunix Hackathon 的 **CURE-GRPO** 方法的报告，探索了 **self-critique + GRPO** 以改进 **LLMs** 的推理能力。
   - 该报告包含了构建和实验 **Tunix + Gemma models** 的实用见解，可在 [Kaggle](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1766255740138) 上查看。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1460367672897437840)** (56 messages🔥🔥): 

> `内存访问验证，Occupancy vs 延迟隐藏（Latency Hiding），cp.async vs TMA，WMMA vs WGMA，Matmul Kernels` 


- **Source Page 审查内存访问**：成员们建议使用带有编译和 profile 选项（`-lineinfo` 和 `--import-source yes`）的 Source-page 来验证内存访问，强调了合并访问（coalescing）的重要性。
   - Source-page 能够准确显示用户是否以及在何处进行了哪种类型的内存访问，以及在合并访问（coalescing）方面的表现有多差，并在每个 kernel 的 Details-page 底部提供了指向 Source-page 中最差内存访问处的链接。
- **CUDA 规则需要听从“芯片的低语”**：一位成员发现，与其最大化 thread tiles，不如优化流水线并发性（pipeline concurrency），通过对 INT32 和 FP32 进行流水线化处理来隐藏延迟，而无需每个线程使用庞大的寄存器堆（register files）。
   - 另一名成员调侃道，别再钻研 CUDA 规则了，去*按芯片的需求与其对话，而不是按 NVIDIA 的意愿*。
- **Tensor Map API 的烦恼**：一位成员表达了对 `cp.async.bulk.tensor (TMA)` API 和 tensor map 的不满，称 `cuTensorMapTileEncoded` `__grid_constant__` 是一个蹩脚的 API。
   - 另一位成员发现，用 **TMA** 替换 `cp.async` 后性能略有下降。
- **WMMA vs MMA 性能**：成员们讨论了即使在 Ampere 架构上也应使用 `mma.sync` 指令而非 `wmma` 的原因，并解释说 `wmma` 会限制在寄存器占用（register footprint）方面能做的操作。
   - 讨论中提到了一篇博文 [WMMA vs MMA](https://forums.developer.nvidia.com/t/wmma-vs-mma/318949)，其中的回复者认为对于计算受限（compute bound）的 matmul 形状，两者不会有明显的性能差异。
- **GEMMs 深度探索**：一位成员报告称他们正在深入研究编写 GEMMs，练习编写 [Simon 的博客](https://developer.nvidia.com/blog/cuda-pro-tip-optimize-performance-with-cuda-streams/) 和 [这篇博客](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/) 中的每一个 kernel，且为了学习不使用 AI。
   - 他们还指出 `wgmma` 指令在 warp-group 粒度（4 个 warps）上运行且是异步的，这需要显式的 commit 和 wait 语义，从而允许 TMA loads 和 tensor core 执行的深度流水线化（deep pipelining）。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1460388438288109770)** (1 messages): 

> `libtorch, JIT, torch.compile, torchscript` 


- **Torch.Compile 提升 Libtorch JIT 性能？**: 一位成员询问如何在 **libtorch** 中利用 **torch.compile** 的性能优势，特别是寻找实现等效加速的方法。
   - 他们注意到用于从 Python 生成 C++ PyTorch 的 **torchscript** 已被弃用，正在寻求替代方案。
- **TorchScript 的落幕促使寻找继任者**: 随着 **TorchScript** 现已被弃用，用户正在寻找现代替代方案来优化 **libtorch** 工作流，并填补 Python 原型设计与 C++ 部署之间的空白。
   - 社区正在探索既能提供类似性能增强，又能轻松集成到现有 C++ 代码库中的选项，以确保生产环境中的平稳过渡和持续效率。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1460541851844218982)** (1 messages): 

> `YouTube videos, Reduced and Mixed Precision Computing` 


- **YouTube 视频期待**: 一位成员表示希望有关 **面向科学与工程应用的缩减精度与混合精度计算 (Reduced and Mixed Precision Computing for Science and Engineering Applications)** 的内容最终能在 **YouTube** 上发布。
- **关于缩减与混合精度计算的 Dagstuhl 研讨会**: 一位成员分享了一个 [Dagstuhl 研讨会](https://www.dagstuhl.de/en/seminars/seminar-calendar/seminar-details/26081) 的链接，该研讨会专注于 **面向科学与工程应用的缩减精度与混合精度计算**。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1460380636111179960)** (11 messages🔥): 

> `ML Kernels, PMPP Book, Parallel Algorithms, CURE-GRPO method, Tunix Hackathon` 


- **全栈开发寻求 ML Kernel 建议**: 一位全栈 SWE 正在转向 **ML Kernels** 领域，并寻求入门建议，得到的推荐从 Stanford 讲座到阅读现有资源不等。
   - 一位成员建议将 [PMPP book](https://a.co/d/akj3tqW) 作为良好的起点。
- **并行计算书籍推荐**: 一位成员推荐了 《[并行计算导论](https://www.amazon.ca/Introduction-Parallel-Computing-Ananth-Grama/dp/0201648652)》 (Introduction to Parallel Computing) 以理解 **并行算法**。
   - 该成员指出这本书虽然专注于 **CPU 并行** (早于 CUDA)，但尽管出版已超过 20 年，其关于 **并行思维和算法** 的内容仍备受称赞。
- **CURE-GRPO 方法报告发布**: 一位成员发布了他们为 **Google Tunix Hackathon** 撰写的 **CURE-GRPO 方法** 报告。
   - 该报告探讨了利用 **self-critique + GRPO** 来提升 **LLMs** 的推理能力，并分享了构建和实验 **Tunix + Gemma 模型** 的心得，报告可在 [Kaggle](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1766255740138) 上找到。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1460510991933767833)** (3 messages): 

> `MXFP8 Benchmarks, Flashinfer, Cutlass, TRTLLM FP8` 


- **寻求 MXFP8 Kernel 基准测试**: 一位成员询问 torchao 中是否存在 **mxfp8 block scale fused moe kernels** 的推理基准测试。
   - 他们希望这些基准测试（特别是与 **flashinfer** 的 **cutlass** 以及 **trtllm FP8 moe kernels** 的对比）能为他们节省大量工作。
- **推理见解的专家推荐**: 另一位成员建议成员 <@894636156875075624> 可能了解推理方面的知识。
   - 这一建议紧随关于 **MXFP8** 基准测试的询问之后，暗示该成员在该领域可能具有专业知识。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1460368965695045662)** (1 messages): 

> `ds_permute intrinsic, footguns` 


- **ds_permute Intrinsic 需要位转换 (Bitcasting)**: ROCm 中的 **ds_permute intrinsic** 仅接受 *int* 参数，因此在使用其他 32 位类型时必须进行位转换，这引入了潜在的陷阱 (footguns)。
- **ROCm 的 ds_permute 陷阱**: ROCm 的 **ds_permute intrinsic** 对 *int* 以外的 32 位类型需要位转换，为开发者造成了潜在的陷阱。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/)** (1 messages): 

n7e_5l_labs: 说“熟悉”可能言重了 —— 但“了解”可能更贴切。
  

---

### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1460377293162676398)** (2 messages): 

> `Scam Solutions` 


- **用户标记潜在诈骗**: 一名用户询问解决方案，暗示他们可能遇到了潜在的 *诈骗 (scam)*。
   - 该用户表达了负面结果，称 *这是一种诈骗*。
- **另一名用户标记潜在诈骗**: 一名用户询问解决方案，暗示他们可能遇到了潜在的 *诈骗 (scam)*。
   - 该用户表达了负面结果，称 *这是一种诈骗*。


  

---


### **GPU MODE ▷ #[tpu](https://discord.com/channels/1189498204333543425/1351571759761068082/1460536298719674378)** (1 messages): 

> `CURE-GRPO method, Google Tunix Hackathon, Self-critique + GRPO, LLMs Reasoning` 


- **CURE-GRPO 方法技术报告发布**: 一位成员宣布发布了关于其在 [Google Tunix Hackathon](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1766255740138) 中使用的 **CURE-GRPO** 方法的技术报告。
   - 该报告探讨了如何利用 **自我批判 (self-critique) + GRPO** 来提升 **LLMs** 的推理能力，并分享了在构建和实验 **Tunix + Gemma models** 过程中的实践见解。
- **关于通过自我批判和 GRPO 增强 LLM 推理的见解**: 该报告提供了在 **Google Tunix Hackathon** 背景下构建和实验 **Tunix + Gemma models** 的实践见解。
   - 它专注于通过应用 **自我批判结合 GRPO** 来改进 **LLM 推理**。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1460435162243404030)** (2 messages): 

> `Layout Composition, Layout Indexing, Zero Layout` 


- **布局索引退化 (Layout Indexing Degeneracy)**: 一位成员讨论到，永远不应直接对超出其大小的 layout 进行索引，因为形状 (shape) 具有语义意义。
   - 他们认为，天真地进行回绕 (wrapping back around) 会导致生成的组合变成 *退化布局 (degenerate layout)*。
- **组合布局 A 和 B**: 提到组合布局 **A** 和 **B** 很方便，其中 **B** 的域 (domain) 是 **A** 的域的真子集（需符合适当的可除性标准）。
   - 该成员澄清说，**B** 仅用于其布局函数，而不是其形状。
- **零布局出现 (Zero Layout Emerges)**: 强调在任何布局 **A** 之后组合 **B = 1:2** 会使 **A** 的步长 (strides) 扩张 2 倍。
   - 进一步提到，天真的回绕反而会导致 *零布局 (zero layout)*。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1460404629681733743)** (4 messages): 

> `Submitting CPP and PTX files, Single file submission for solutions` 


- **请求分开提交 CPP 和 PTX 文件**: 一位用户询问是否可以作为 **独立的 CPP 和 PTX 文件** 提交代码，而不是单一的大文件。
   - 回复指出，系统设计为管理 **单文件提交**，以便于管理和 AI 阅读。
- **多文件提交的变通方案**: 提到一种变通方法是在提交时 **将独立文件合并为一个文件**。
   - 这允许开发者在开发期间保持代码模块化，同时符合提交要求。


  

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1460367022289457507)** (15 条消息🔥): 

> `KernelBot 时间可靠性、Discord 文件打开问题、CLI 的 NCU 集成、评估脚本性能下降` 


- **KernelBot 的时间可靠性取得进展**：**KernelBot** 的时间可靠性（timing reliability）正在取得进展，一个 [PR](https://github.com/gpu-mode/kernelbot/pull/386) 已提交审核，预计很快会有另一个更新。
   - 一名成员表示，目前进展顺利。
- **用户在 Discord 中打开文件遇到阻碍**：一名成员请求协助解决 Discord 上的文件打开问题。该问题被描述为在网页端返回通用的 *"Failed"* 错误，在 Discord 客户端显示 *"unexpected error"* 消息，详情见 [此 Discord 线程](https://discord.com/channels/1189498204333543425/1434709259500650628/1460272771652386950)。
   - 涉及的文件长达 **7K 行**，引发了人们对打开和处理过程中潜在问题的担忧。
- **NCU CLI 集成已部署**：**NCU (NVIDIA Command Line Utility)** 已集成到 CLI 中，允许用户在行内渲染摘要并下载 **ncu-rep** 文件，该功能可通过 [popcorn-cli v1.2.2 版本](https://github.com/gpu-mode/popcorn-cli/releases/tag/v1.2.2) 使用。
   - 相关指南见 [此处](https://github.com/gpu-mode/popcorn-cli/blob/main/docs/profiling.md)；这项工作建立在其他成员之前的努力之上。
- **评估脚本报告性能抖动**：成员们报告称，尽管 `eval.py` 脚本本身没有变化，但评估脚本的性能与上周相比持续出现了 **0.5us 的速度减慢**。
   - 这种减慢在多次运行中持续存在，似乎与代码更改无关，表明可能存在环境因素的影响。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1460366677240975575)** (11 条消息🔥): 

> `GPU vs TPU, Adam Paszke, JAX` 


- **Adam Paszke 认为 GPU 和 TPU 正在趋同**：一名成员提到 **Adam Paszke** 关于 **JAX** 的一次演讲，他在演讲中认为 **GPU** 和 **TPU** 正在趋同。
   - 另一名成员询问是否有事实支持这一观点。
- **Adam Paszke 的资历**：一名成员提供了 **Adam Paszke** 的 [LinkedIn](https://www.linkedin.com/in/apaszke/) 链接和一段 [YouTube 视频](https://www.youtube.com/watch?v=wKd90avC8Nc) 作为其资历证明。
   - 另一名成员确认该 **YouTube 视频**正是被引用的那段。
- **“Sir, this is a Wendy's”**：针对一个跑题的争论，一名成员开玩笑地回复道 *“sir this is a wendy's”*（先生，这里是 Wendy's 汉堡店）。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1460366951674286210)** (67 条消息🔥🔥): 

> `Safetensors, Diffusers 与 ComfyUI, AI Agents, 视频生成 AI 推荐, qwen3 coder 量化` 


- **关于 AI Agent 实用性的辩论爆发**：成员们辩论了 **AI Agent** 的实用性。有些人认为它们很*垃圾*，而另一些人则认为它们是不需要掌握 AI 框架即可进行创作的途径，尤其是对于那些难以跟上 **LLM** 发展节奏的人来说。
   - 一名成员表示有兴趣了解 **coders**（编码人员）认为哪些类型的产品对辅助编程有价值，并将其比作“亲自种稻谷”与“直接从便利店买饭团”之间的精力投入差异。
- **Safetensors 兼容性困扰**：一名用户询问为什么没有为图像生成模型编译 **safetensors**，另一名用户澄清说，他们只发布供 **Diffusers** 在 Inference API 中使用的文件，并无意创建 safetensors 文件。
   - 讨论涉及了在 Diffusers 和 ComfyUI 格式之间转换文件的挑战。该用户强调了手动复制文件和组合它们的难度，而另一名用户则推荐使用 **venv** 或 **ComfyUI** 的便携版。
- **ComfyUI vs A1111 WebUI**：成员们讨论了 **ComfyUI** 相较于 **A1111 WebUI** 的易用性，一名用户发现 ComfyUI 易于安装且使用时没有任何设置障碍或问题。
   - 他们还提到它能直接处理 **Diffusers 格式**，但在卸载包时因权限缺失遇到了问题，随后他们直接手动删除了插件文件夹。
- **关于视频生成 AI 的询问**：一名成员请求推荐免费的**视频生成 AI**。
   - 未给出具体推荐。
- **Qwen3 Coder 量化质量问题**：一名成员报告称，在运行 **qwen3 coder** 时，除 `Q8_0` 以外的任何量化都会导致性能低下。
   - 他们进一步阐述道，即使在 level 7，该模型也会出现基础错误，如空格丢失或多余，这阻碍了它构建工具请求（tool requests）的能力。此外，他们只有一块 **5090**，因此无法提供太多上下文让其在 GPU 上运行 🙁。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1460366138792743003)** (10 messages🔥): 

> `Complexity Framework, Mention in AI Framework, SynPaca Dataset, Error Correction` 


- **Complexity Framework 获得 HuggingFace 特别提及**：一位成员在其 **Complexity-Framework**（兼容 **Mistral, GPT 和 Llama**）中特别提到了 HuggingFace 以及一名用户对 GCCR 的帮助。
   - 该框架包含许多新功能，用户提到：“感谢 Huggingface :@Wilbaor 以及 Huggingface :@Wilbanice 的帮助”。
- **MadlabOSS 发布 SynPaca 数据集**：成员分享了 **MadlabOSS** 在 HuggingFace datasets 上发布的 [SynPaca 数据集](https://huggingface.co/datasets/MadlabOSS/synpaca)链接。
   - 该链接还提到了 [Complexity-ML complexity-framework](https://github.com/Complexity-ML/complexity-framework) 和 [synthetic error correction v4 数据集](https://huggingface.co/datasets/webxos/synthetic_error_correction_v4)。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1460410612239696058)** (1 messages): 

> `Channel usage, Discord` 


- **Discord 频道使用规范**：一位成员提醒其他人 **保持频道讨论主题**。
   - 他们建议使用 <#905728440873918484> 频道进行闲谈讨论。
- **频道指南提醒**：发出了关于保持频道关注点和相关性的温馨提醒。
   - 该消息鼓励用户利用指定的闲谈频道讨论主要主题之外的内容。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1460362953793470465)** (75 messages🔥🔥): 

> `Cursor Pro plan usage, Generating Word Documents with AI, Claude Opus slowness and unreliability, Pulse Framework, Referencing past chats` 


- **Cursor Pro 计划用量质疑**：一位成员询问他们在 **Cursor Pro 计划**中实际获得了多少用量，以及如何查看计量表，并附上了一张与 Token 使用相关的[截图](https://cdn.discordapp.com/attachments/1074847527708393565/1460362953780891710/image.png?ex=69674d3c&is=6965fbbc&hm=abb5b33d0a6e25610a0ec5dc3926976e90bea06bfb5ab98b68f0f07e603c5e4d)。
   - 该成员还询问：*这是否像每月预付价值 20 美元的 Token，月底过期，不用就作废了？*
- **自动化 AI 生成 Word 文档**：成员们讨论了使用 AI 生成关于**气动学的 Word 文档**，建议先使用 Markdown 生成然后进行转换，并配合 Python 脚本进行转换。
   - 一位成员指出，这不再是模型问题，而是**工具问题**，并建议使用**浏览器扩展搜索图片**并包含在内，同时提到 **antigravity** 作为一个潜在工具。
- **Opus 表现不佳，Codex 来救场？**：一些成员对 **Claude Opus** 表示沮丧，理由是其速度慢且在修复问题时效率低下，而另一些人则认为这是因为模型被量化了或 System Prompt 设置不当。
   - 一位用户表示，他们在 **10 秒钟**内解决了一个 **Claude** 花了 **30 分钟**都无法解决的问题，而另一位用户则切换到 **Codex** 解决了问题。
- **Pulse Framework 发布**：一位成员介绍了他开发的 **Pulse** 框架，并分享了 [GitHub 仓库链接](https://github.com/manuelfussTC/PulseFramework)。
   - 此举正值大家讨论其他 **Claude Code** 工具，以及这些工具是否像传闻中那样好用之时。
- **Cursor 默认聊天位置故障**：一些用户报告了 **Cursor 默认聊天位置**的问题，即聊天面板不再起作用，所有聊天都在标签页中打开。
   - 一位成员确认这是一个普遍问题，但切换模型可能会有帮助，而另一位成员提到 **Qoder** 没有这个问题。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1460368223404032235)** (60 messages🔥🔥): 

> `Claude Cowork, OpenAI Sweetpea, Phind Shutdown, DeepSeek Engram, Daniel Gross Meta` 


- **Claude 推出 Cowork**: Claude 宣布了 **'Cowork'，这是一个旨在将 Claude Code 的效率和功能带给非技术专业人士的新工具**，用于完成日常工作任务，详见[此帖](https://x.com/claudeai/status/2010805682434666759?s=46)。
- **OpenAI 'Sweetpea' 泄露**: 泄露的细节揭示了 **OpenAI 即将推出的硬件项目，代号为 'Sweetpea'，这是一款旨在与 AirPods 竞争的音频可穿戴设备**，具有金属“蛋石”设计和 2nm 芯片，如[此推文](https://x.com/kimmonismus/status/2010804115543114099?s=46)所示。
- **Phind 走向终结**: 根据[此 Discord 帖子](https://discord.com/channels/996538859972214835/1077743365749227630/1460382964029460584)，**Phind** 将在本周末关闭。
- **Gross 加盟 Meta**: 根据[此报告](https://x.com/MeghanBobrowsky/status/2010778788964286832)，Daniel Gross 正在 **Meta 领导一项新的 AI 基础设施计划**，与新任命的总裁 Dina Powell McCormick 和资深高管 Santosh Janardhan 合作。
- **ElevenLabs 强势达成 3.3 亿美元 ARR**: 据[此推文](https://x.com/LukeHarries_/status/2010780712283365543)报道，ElevenLabs 达到了 **3.3 亿美元 ARR（年度经常性收入）里程碑**。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1460465037570347120)** (4 messages): 

> `Gamma, CEO, Grant Lee` 


- **Gamma 迎来新 CEO**: [Grant Lee 宣布](https://xcancel.com/thisisgrantlee/status/2010811316299317582) **Gamma** 将于 **2026 年 1 月 13 日**任命新的 **CEO**。
- **Gamma 领导层更迭**: 由 **Grant Lee** 发布的这一公告表明，**Gamma** 在为新 CEO 的到来做准备时，领导层将发生重大转变。
   - 这一过渡计划于 **2026 年 1 月 13 日**进行，标志着公司的关键日期。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1460393807651672220)** (2 messages): 

> `ChatGPT Hallucinations, Devin.ai Code Maps` 


- **ChatGPT 对 Java 性能产生幻觉**: 一位成员提示 **ChatGPT** 生成关于它为何幻觉认为 **Java** 具有强劲性能并建议使用它的回答。
   - 该成员采纳了建议，并指出他们*从未在 ChatGPT 网页版 UI 之外见过一行代码*。
- **Devin.ai 的 Code Maps 对比**: 一位成员将之前的输出与 **Devin.ai** 的 code maps 或 deep wiki 进行了对比。
   - 未提供关于对比细节的进一步信息。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1460370282337403162)** (47 messages🔥): 

> `OpenRouter availability, Google AI Studio video and audio URLs, Gemini 3 Pro opinions, Sillytavern for OpenRouter, Gemini usability for long chats` 


- **OpenRouter 可用性受到质疑！**: 一些用户怀疑 **OpenRouter** 的停机是为了骗取免费额度的“幻觉”，而另一些人则坚持认为它正以 *100% 的可用性*正常运行。
   - 这些关于停机的说法被其他成员反驳，暗示人们只是想获取免费额度。
- **Google AI Studio 将支持更多视频 URL？**: 一位成员询问关于在 **AI Studio** 提供商下支持 **Google 模型**的视频和音频 URL，引用了 [Google 官方公告](https://x.com/GoogleAIStudio/status/2010768441553428772)，该公告允许直接使用 URL。
   - 目前，**Google AI Studio** 仅支持 **YouTube** 视频，不支持直接 URL，音频仅限于 **base64**，支持 PDF 和图像，但不支持 **2.0 模型**。
- **Gemini 3 Pro 面临用户批评**: 一位成员夸张地表示 *没有人喜欢这个模型*，引发了关于该言论普适性的回应。
   - 一位成员表示，虽然 **Gemini 3** 很好，但他们发现 **GPT-5.2** 在指令遵循方面更可靠，且幻觉更少，特别是在 **Gemini** 网页版应用中。
- **Sillytavern 仍是 OpenRouter 的黄金标准**: 一位成员询问 **Sillytavern** 是否是配合 **OpenRouter** 使用的最佳选择。
   - 一位成员推荐了 **Cherry Studio** 用于角色扮演，而另一位用户表示他们不涉及此类话题。
- **Gemini 在长对话中的可用性**: 一位用户分享说他们不喜欢 **Gemini 3** 的现状，发现它容易变得懒惰并产生幻觉。
   - 另一位成员形容 **Gemini** 网页应用是垃圾，并表示在超过 *200k+ tokens* 的长对话中使用 API 会导致数据造假和糟糕的转录。


  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1460378421879111833)** (7 条消息): 

> `Google Gemini Apple Intelligence 功能, 反垄断法的影响, Claude 的用户亲和力` 


- **Gemini 助力未来 Apple Intelligence**: 根据 [MacRumors 的一篇文章](https://www.macrumors.com/2026/01/12/google-gemini-future-apple-intelligence-features/)，Google 在 **Gemini** 上的进展可能会为未来的 **Apple Intelligence** 功能做出贡献。
   - 一位成员幽默地指出，*Google 表现得太强势了，以至于不得不拉 Apple 一把以规避反垄断法*。
- **反垄断法引发关注**: 讨论围绕第一大和第二大公司之间的合作是否触及 [反垄断法](https://x.com/OfficialLoganK/status/2010769064956752166) 展开。
   - 一位成员发帖称：*我的意思是，最大的公司帮助第二大的公司，这几乎肯定属于反垄断法的范畴*。
- **Claude 以高 EQ 迷住用户**: 一些用户表达了对 **Claude** 的强烈喜爱，尤其是自 **3.5 Sonnet** 更新以来，强调其与其他 LLM 相比具有更出色的 *EQ*。
   - 一位成员表示：*Claude 真的很懂我，老兄。自从 3.5 Sonnet 推出以来我一直有这种感觉。我不在乎任何 EQ 评测指标（EQ bench）怎么说*。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1460364891138298049)** (52 条消息🔥): 

> `指定频道中的误报, 聊天对应频道中的诈骗, 使用 chat gpt 和 n8n 的 AI 接待员工作流, 智能路由功能反馈, Arena Champion 晋升` 


- **报告疑似误报**: 鼓励成员在特定频道 <#1447983134426660894> 中报告疑似误报。
   - 这有助于提高系统的准确性和可靠性。
- **聊天频道发现诈骗**: 一位成员指出，在“聊天”对应频道 <#1340554757827461216> 中存在一些诈骗行为。
   - 管理员已收到通知，并将采取行动进行清理。
- **从零构建的 AI 接待员工作流**: 一位成员使用 **chat gpt** 和 **n8n** 构建了一个 AI 接待员工作流，用于预约电话、回答问题、重新安排、取消以及处理短信。
   - 他们正在寻求建议和合作，以将该工作流投入生产，并对未来项目的批评持开放态度。
- **征求新智能路由（Smart Router）功能的反馈**: 引入了一项新的 **智能路由** 功能，并正在向社区征求反馈。
   - 成员们正在讨论其功能并提供意见。
- **比赛参赛作品限制**: 一位成员询问了艺术创作比赛允许的参赛作品数量，并观察到一些用户提交了多个作品。
   - 管理员澄清限制为 **3 个作品**，如果提交更多，将仅考虑前 3 个。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1460426291546296320)** (2 条消息): 

> `ltx-2-19b 模型, 一月 AI 生成大赛, 自然回归（Nature Reclaims）主题` 


- ****Ltx-2-19b** 进入视频竞技场（Video Arena）**: 新模型 **ltx-2-19b** 已添加到 [Video Arena](https://discord.com/channels/1340554757349179412/1397655624103493813)，快来测试吧！
   - 模型通过 **对战模式（Battle Mode）** 进行测试，鼓励社区参与投票。
- **一月 AI 生成大赛开跑**: LMArena 正在举办一月份的每周 AI 生成大赛，寻找下一位 [AI Content Creator](https://discord.com/channels/1340554757349179412/1378032433873555578)。
   - 要参赛，请在 **1 月 16 日** 之前在 [#jan](https://discord.com/channels/1340554757349179412/1397655624103493813) 提交一张截图，展示投票后来自对战模式的两个模型响应。
- **一月主题为“自然回归”**: 一月 AI 生成大赛的主题是 *自然回归（Nature Reclaims）*，征集描绘自然重新夺回人类建筑环境的作品。
   - 提交的作品应描绘被自然界占领、改造或重新诠释的人造环境，获胜者将获得 **Discord Nitro** 和 **AI Content Creator** 身份组。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1460423869176217621)** (31 messages🔥): 

> `Kubernetes Deployment, OpenAI Stock Ownership, Sam Altman, Ilya Sutskever` 


- **Kubernetes Deployment: Robocop Cluster 部署完成**：一名成员开玩笑地提出了一个名为 **OCP** 的 **Kubernetes deployment** 概念，其中 controller 是 **Robocop**，而 control planes 是警察，并创作了口号：*"无论死活，你都得加入这个 cluster。"*
- **Sam Altman 股票情况推测**：成员们讨论了 **Sam Altman** 是否持有任何 **OpenAI stock**，一名成员指出他并不持有，因为并没有所谓的 OpenAI 股票。
   - 提到存在针对员工的内部 stock allocation，但这与公开交易的股票不同。
- **Ilya Sutskever 的 OpenAI 持股情况**：提到由于内部 allocation，**Ilya Sutskever** 持有价值约 **200-300 亿** 的 **OpenAI stock**。
   - 资深的创始员工可能也持有数亿至数十亿的股票，但具体的股份数量从未对外公布。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

deckard.1968: 这感觉极其离谱
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

deckard.1968: 这感觉极其离谱
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1460391451140554938)** (8 messages🔥): 

> `Model Persistence, AI Dev Upskilling, JAX Pallas BlockSpec Help` 


- **模型持久化策略**：一名成员建议维护一个 *models.md* 文件，供模型每次读取。
   - 他们还建议在研究氛围较淡的服务器（如 **Midjourney** 或各种 AI 艺术社区）中进行咨询。
- **AI 开发者需要提升特定工具的技能**：一名成员注意到在线/本科课程与职位期望之间存在错配，建议提升在 **bioconductor**、**JAX/PyTorch**、**GIS** 以及各种生物信息学/化学信息学工具方面的技能。
   - 他们强调需要处理来自各个子领域的杂乱文件类型，并具备阅读/撰写研究论文的能力，因为就业市场正从单纯的 coding 转向 research 技能，但其他成员认为这过于侧重研究，而非商业 AI 开发。
- **JAX Pallas BlockSpec 问题**：一名成员请求协助解决 **JAX Pallas** 中 **BlockSpec** 的问题，因其表现非常“怪异”。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1460368249945456834)** (7 messages): 

> `Flow Matching, Engram Paper, Image Prediction` 


- **Flow Matching 在发散问题上的可行性**：一名成员讨论了将 **flow matching** 用于具有强发散性问题的可行性，并提供了 [Engram Paper](https://github.com/deepseek-ai/Engram/blob/main/Engram_paper.pdf) 的链接。
   - 他们提到，将目标更改为预测两张图像的差异可能是一个好主意。
- **更高效的采样技术**：一名成员认为使用更高效的 sampling techniques 可能不会显著改善结果。
   - 相反，它可能主要只是提高采样效率。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1460366738737725460)** (8 messages🔥): 

> `Bilinear layers, SwiGLU layers` 


- **Bilinear Layers：两个 Encoder 优于一个？**：一名成员询问了使用 [bilinear layers](https://en.wikipedia.org/wiki/Bilinear_map) 的好处，这种层有效地采用了 **两个 encoders**。
   - 另一名成员回答说，**SwiGLU layers**（同样使用两个 encoders）更符合 SOTA 标准，并且在采用两个 encoders 时，通常使用 element-wise multiplication 来进行组合。
- **作为二次多项式的 Bilinear Layers**：一名成员提到 bilinear layers 是二次多项式，当与 residual stream 堆叠时，它们可以逼近任何连续函数。
   - 该成员进一步询问了 [VC dimension](https://en.wikipedia.org/wiki/VC_dimension) 的结果，以及在 attention scores 表现良好的情况下，对 softmax 函数使用 **Taylor expansion** 的可能性。

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1460372052598067231)** (13 条消息🔥): 

> `RTX 3090 reboot issues, LLM stress testing, LM Studio 0.3.4 with Apple MLX, MoE models as dense models in LM Studio` 


- **RTX 3090 在使用 LLM 时重启**：一位用户在 **Linux (Fedora) 和 Windows** 上遇到了与其 **RTX 3090** 相关的突然重启问题，特别是在运行 **LLMs** 时；临时解决方案包括禁用 **GSP firmware**、**undervolting**（降压）以及对显卡进行 **underclocking**（降频）。
   - 为了对 **GPU** 进行压力测试，成员建议运行 **OCCT** GPU 测试，或使用 context 不足的 dense reasoning **LLMs** 来诱导死循环。
- **LM Studio 版本混淆**：一位用户注意到，尽管宣传 **LM Studio 0.3.4** 支持 **Apple MLX**，但可供下载的只有 **0.3.37** 版本。
   - 成员澄清说，版本号在内部可能显示为 **0.3.04**，并建议使用最新版本（**0.3.37**），该版本与 **MLX models** 配合良好。
- **在 LM Studio 中将 MoE 模型作为 Dense 模型运行的实验**：一位用户询问关于将 **MoE (Mixture of Experts) models** 作为 dense 模型运行（激活所有专家）并与标准 **MoE version** 比较性能的问题。
   - 一位成员表示可以在 **LM Studio** 中更改专家配置，但据报告性能比默认设置更差。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1460518963212718267)** (5 条消息): 

> `LMStudio on ARM, Orange Pi 6 Plus, Video Driver Issues, Qwen3 4b` 


- **LMStudio 登陆 ARM！**：一位用户在运行 **Ubuntu** 的 **Orange Pi 6 Plus** 上成功安装了 **LMStudio**。
   - 他们报告称，仅使用 CPU 和 **8 个 CPU cores**，运行 **Qwen3 4b 2507 Q4** 可达到 **6.6 t/s**。
- **图形故障困扰 GUI**：用户注意到 UI 图形损坏，可能是由于视频驱动程序不成熟以及 Electron 应用导致的。
   - 作为一个临时变通方法，打开右侧配置栏可以减轻*部分*图形损坏（由于需要*盲点*操作）。
- **期待硬件加速**：用户希望在未来的项目中看到视频驱动的改进以及 NPU/GPU 加速。
   - 他们还报告称，在 **OPi** 上仅使用 CPU 运行 **gpt-oss** 可达到 **6.26 t/s**。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1460381693314863319)** (15 条消息🔥): 

> `PR backlog on tinygrad, TinyBox BMC login issues, New ideas for bounties, TinyBox server usage, BMC firmware reflashing` 


- **tinygrad PR 堆积**：**PRs** 又开始堆积了，作者删除了他们的 **pip uv/wincuda** 相关 PR。
   - 他们正专注于 **assembly/amdi**，认为这是为了解锁该任务后续环节必须打下的基础。
- **tinygrad 即将推出新的“速度”悬赏**：将会有一些新的悬赏点子，比如应该仍然可行的**“速度”悬赏**。
   - 作者将编写一些 **infra**（如 GPUMODE）以使这些悬赏更容易执行和评审。
- **用户无法登录 TinyBox BMC**：一位用户报告称无法登录其 **TinyBox** 上的 **BMC**，并正在寻求关于如何从 **Ubuntu** 刷写 **BMC firmware** 或执行硬件跳线重置的建议，并提到了错误消息 *"LAN Parameter Data does not match"*。
   - 用户已经尝试了多种方法，包括**重置/更改 BMC 密码**、验证 **SSH tunnel** 以及执行 **BIOS reset**。
- **TinyBox 用于构建和托管 Agent**：一位用户计划使用他们的 **TinyBox** 来构建和托管 Agent，这台机器是从同事那里传下来的，目前正在重置。
   - 另一位用户询问了购买事宜以及本地操作的程度。
- **刷写 BMC Firmware 可修复 TinyBox**：一位成员建议通过重新刷写/更新 **BIOS/UEFI**、重新刷写/更新 **BMC firmware**，然后尝试从 **UEFI menu** 重置 BMC 来修复 **TinyBox**。
   - 该成员建议在按照预期设置完成后提取一份 **config backup** 以防万一，尤其是在 **SuperMicro** 和 **Lenovo** 服务器上遇到问题之后。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1460397417462759568)** (6 messages): 

> `Mojo on Fedora, Mojo on Arch, Mojo on other distros, Mojo and Max` 


- **Mojo 的发行版首秀**：一位用户询问在 **Fedora**、**Arch** 或其他发行版上运行 **Mojo**（而非 Max）的情况。
   - 一名成员回复称，理论上它应该可以工作，但他们没有针对这些发行版进行专门测试，并邀请用户在遇到问题时提交 issue。
- **Mojo 是否经过调优？**：一位用户想知道为什么 **Mojo** 在特定发行版上可能运行或无法运行。
   - 他们表示一直认为某些程序需要针对特定发行版进行调优，并举例说如果某个程序在 **Debian** 上运行，它极有可能在 **Ubuntu** 和 **Mint** 上运行，但在 **Arch** 上运行之前可能需要进行微调。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1460377238846570527)** (4 messages): 

> `Manus and Meta, Automating pipelines with Node, APIs, and LLMs, RAG, Multi-agent systems, and cloud integrations, Deleting tasks one by one` 


- **Meta 正在考虑对 Manus 的举措？**：一名成员想知道 **Meta** 会对 **Manus** 采取什么行动。
   - 没有添加进一步的信息或讨论。
- **Node, APIs 和 LLMs 自动化杂乱的工作流**：一名成员建议，使用 **Node**、**APIs** 和 **LLMs** 自动化流水线可以为杂乱的工作流或重复性任务节省数小时时间并减少错误。
   - 他们补充说，结合 **RAG**、multi-agent systems 和云端集成可以使流程具备可扩展性和可靠性。
- **批量删除任务的苦恼**：一名成员报告称，系统目前仅支持逐个删除任务，不支持**批量删除**。
   - 他们分享了一个[链接](https://help.manus.im/en/articles/11711980-how-can-i-delete-my-tasks)，展示了如何删除单个任务。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1460382646780690782)** (3 messages): 

> `FDA Guidance on Statistical Methods, ClaudeAI Status` 


- **FDA 更新统计方法指南！**：[FDA 发布了指南](https://www.fda.gov/news-events/press-announcements/fda-issues-guidance-modernizing-statistical-methods-clinical-trials)，使临床试验的统计方法现代化。
   - 这一更新可能会影响医疗场景中 **AI/ML** 模型的评估方式，优先考虑稳健且可靠的统计验证。
- **ClaudeAI 状态在线**：一名成员链接到了一个追踪 [ClaudeAI 状态](https://fixupx.com/claudeai/status/2010805682434666759)的页面。
   - 该资源可帮助用户实时监控 **Claude 的运行时间、性能和潜在问题**，确保更顺畅地集成到工作流中。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1460407685685186715)** (1 messages): 

> `DSPy users, AI Engineer Events` 


- **DSPy 社区提倡在 AI Engineer Events 中增加曝光**：一位社区成员表示，需要更多 **DSPy 用户**在 **AI engineer events** 上展示他们的工作。
   - 他们感谢了另一位成员“为社区做贡献”并代表了 DSPy 出席。
- **DSPy 社区参与**：一位社区成员对 DSPy 用户的参与表示赞赏。
   - 他们对代表社区出席活动的行为表达了感谢。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1460375274469199974)** (1 messages): 

> `Kimi mental breakdown` 


- **Kimi 出现“精神崩溃”**：根据一份[用户报告](https://discord.com/channels/1369594130807787570/1460374643440488562)，Kimi 聊天机器人似乎出现了**精神崩溃**。
   - 未提供关于崩溃性质或原因的进一步细节。
- **Discord 用户报告 Kimi 聊天机器人问题**：一位用户报告称 **Kimi** 聊天机器人经历了某种故障，被描述为“**精神崩溃**”。
   - 该报告发布在 Discord 的 general chat 频道中，但缺乏关于聊天机器人行为或问题原因的具体细节。