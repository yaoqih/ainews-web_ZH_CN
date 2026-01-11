---
companies:
- anthropic
- openai
- ai21-labs
- github
- cline
date: '2026-01-09T05:44:39.731046Z'
description: '**Anthropic** 收紧了第三方应用中 **Claude Max** 的使用政策，促使开发者转向**模型无关的编排 (model-agnostic
  orchestration)**和默认采用**自带密钥 (BYO-key)**模式，以降低平台风险。


  **模型上下文协议 (MCP)** 正在演进为关键的工具层 (tooling plane)，**OpenAI MCP Server** 和 **mcp-cli**
  进一步提升了工具发现能力和 Token 效率。将“**技能 (skills)**”视为模块化、版本化行为的概念正受到关注，**Claude Code**、**GitHub
  Copilot** 以及新增了网页搜索工具的 **Cline** 均已实现相关功能。


  **AI21 Labs** 通过在智能体工作区使用 **git worktrees** 进行事务性并行写入，解决了并发挑战；而**长程智能体 (long-horizon
  agents)** 则将重心放在**上下文工程 (context engineering)**和持久化的、以文件为中心的工作区上。'
id: MjAyNi0w
models:
- claude-max
people:
- yuchenj_uw
- andersonbcdefg
- gneubig
- matan_sf
- scaling01
- reach_vb
- _philschmid
- claude_code
- code
- jamesmontemagno
- cline
- danstripper
- omarsar0
title: 今天没发生什么特别的事。
topics:
- model-agnostic
- model-context-protocol
- tooling
- skills
- concurrency
- transactional-workspaces
- context-engineering
- file-centric-workspaces
- rate-limiting
- agent-workspaces
---

**DeepSeek v4 即将来临...**

> 2026年1月8日至1月9日的 AI 新闻。我们为您检查了 12 个 subreddits、[**544** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 以及 **24** 个 Discord（**204** 个频道和 **4384** 条消息）。预计节省阅读时间（以 200wpm 计算）：**402 分钟**。**我们的新网站**现已上线，支持全文元数据搜索，并以精美的 vibe coded 方式呈现过往所有期数。请访问 https://news.smol.ai/ 查看完整的新闻分析，并在 [@smol_ai](https://x.com/Smol_AI) 上给我们反馈！

Claude Code 因各种原因持续受到关注，请继续阅读...

---

# AI Twitter 回顾


**塑造 “coding agent” 生态系统的政策与平台转变**

- **Anthropic 收紧第三方应用中 Claude Max 的使用**：多篇帖子描述了 Anthropic 阻止在外部客户端中使用 Claude 订阅（据报道还切断了一些竞争对手的访问），这强化了在单一供应商的消费者计划上构建关键业务流的风险。参见 [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2009691122940211201) 以及来自 [@andersonbcdefg](https://twitter.com/andersonbcdefg/status/2009509161823031351) 等构建者的反应，以及 [@gneubig](https://twitter.com/gneubig/status/2009686033563316501) 从市场结构角度的论述。实际影响：预计会出现更多 **model-agnostic**（模型无关）的 harness 和 **BYO-key**（自带密钥）默认模式，并将 “max plan” 访问权视为可撤销的。
- **Model-agnostic 编排成为产品需求**：几位构建者强调，由于 rate limits 和政策变化，不要“孤注一掷”于单一供应商。示例：[@matanSF](https://twitter.com/matanSF/status/2009472570438095130) 主张通过 **model-agnostic** 基础设施来降低平台风险；[@scaling01](https://twitter.com/scaling01/status/2009568477972201686) 指出了在达到 Opus token 限制时的 **rate limiting** 问题，强调了对 fallback routing 和预算的需求。

**Agent 与开发者工具：MCP、技能、harnesses 以及长周期可靠性**

- **MCP (Model Context Protocol) 正在迅速演变为“工具层”**：
  - **OpenAI MCP Server**：与 OpenAI 相关的团队发布了一个 MCP Server，捆绑了文档、指南、API、AppsSDK 等，旨在与 Codex、Cursor、VSCode 及其他 Agent 开箱即用（[推文](https://twitter.com/reach_vb/status/2009686112986337309)，[后续](https://twitter.com/reach_vb/status/2009686476255084767)）。潜台词是：MCP 将作为“官方”工具接口的分发渠道，而不只是社区插件。
  - **mcp-cli**：一个用于**动态发现** MCP Server 的轻量级 CLI，声称通过发现机制而非冗长的提示词/工具描述，可**减少 99% 的 Token 消耗**；支持 stdio + HTTP、管道化 JSON 输出以及跨服务器的 grep（[推文](https://twitter.com/_philschmid/status/2009625698361573521)，[链接](https://twitter.com/_philschmid/status/2009625701432152438)）。这是 MCP 的“运维（Ops）”侧：让工具变得可发现和可脚本化，而不会导致上下文膨胀。
- **“技能（Skills）”作为模块化、版本化的行为**：
  - Claude Code 的定义：**插件（plugins）**作为容器；**技能（skills）**作为专门的程序/知识，某些产物可以兼具两者（例如“前端设计”）（[推文](https://twitter.com/claude_code/status/2009479585172242739)）。
  - GitHub Copilot / VS Code：“Agent Skills”已在稳定版上线；发布了快速入门视频并将其定位于工作流加速器（[@code](https://twitter.com/code/status/2009744142335656156), [@JamesMontemagno](https://twitter.com/JamesMontemagno/status/2009720264335208598)）。
  - Cline：增加了**技能兼容性**和内置的 **websearch 工具**（[推文](https://twitter.com/cline/status/2009793063753757024)）。
  - 模式：各团队正趋向于将“技能”视为**延迟加载的指令包**，以避免将所有内容都塞进基础提示词中。
- **状态、并发以及“并行写入”问题**：
  - AI21 描述了一个真实的痛点：当运行多个需要**并发写入文件**的子 Agent 时，MCP 会遇到瓶颈；他们添加了一个带有原语（init/clone/compare/merge/delete）的“MCP Workspace”层，并通过 **git worktrees** 实现代码工作区，从而支持 **1 到 16 个并行尝试**而无需协调，最后合并胜出者（[推文起始](https://twitter.com/AI21Labs/status/2009565879600923100)，[工作区](https://twitter.com/AI21Labs/status/2009565885284200540)，[git worktrees + 结果](https://twitter.com/AI21Labs/status/2009565888148652226)）。这是迈向**事务型 Agent 工作区**的坚实一步。
- **长周期 Agent：“上下文工程”是核心瓶颈**：
  - InfiAgent 提议通过将持久状态外部化到**以文件为中心的工作区**（该工作区由快照 + 固定最近窗口在每一步重建）来保持推理上下文的有界性（[摘要](https://twitter.com/omarsar0/status/2009662975024447511)）。这与日益流行的“Agent 即文件/文件夹”理念相契合（例如 [@danstripper](https://twitter.com/danshipper/status/2009651408144835021)）。
  - “Agent 漂移”被强调为一种常见的多 Agent 失败模式——语义/协调/行为漂移——此外还提出了 Agent 稳定性指数以及片段化巩固和行为锚定等缓解措施（[推文系列](https://twitter.com/dair_ai/status/2009657177989091423)）。核心：评估不仅要看任务成功率，还要看**交互长度上的稳定性**。
- **Agent 的评估（Evals）从理论走向实践**：
  - Anthropic 的《揭秘 AI Agent 评估》作为面向生产的指南被广泛分享：包含评分器（代码/模型/人工）、能力与回归评估、pass@k 与 pass^k，以及从真实的失败案例开始（[推文](https://twitter.com/AnthropicAI/status/2009696515061911674)）。
  - 实践者的解读强调了观察 **Agent 轨迹（traces）**的重要性，以了解失败是*如何*发生的，并让指令、工具、测试框架与评估设计共同进化（[@Vtrivedy10](https://twitter.com/Vtrivedy10/status/2009724848482762966), [@hwchase17](https://twitter.com/hwchase17/status/2009732201269588479)）。

**模型 + 数据集发布与基准测试信号**

- **Falcon-H1R-7B (TII UAE)**: Artificial Analysis 将该模型评为一款小型 open-weights **reasoning** 参赛者，在其尺寸级别具有极强的竞争力；指出由于需要署名的许可协议，其 Openness Index 评分受到影响；并强调了其在 Humanity’s Last Exam、τ²-Bench Telecom 和 IFBench 上的强劲表现 ([分析](https://twitter.com/ArtificialAnlys/status/2009690138604122238), [链接](https://twitter.com/ArtificialAnlys/status/2009690152608903446))。
- **Open-weights “frontier pressure” 持续**: 多条推文指出开源模型的竞争力正在加速，以及与美国发布的开源模型之间的战略差距 ([Artificial Analysis 趋势笔记](https://twitter.com/ArtificialAnlys/status/2009759874461081957)，以及来自 [@Teknium](https://twitter.com/Teknium/status/2009630706146984178) 等开发者的情绪反馈)。
- **FineTranslations 数据集 (合成平行语料库)**: 一个新的 **>1T token** 平行数据集，通过使用 Gemma3 27B 将 FineWeb2 多语言数据翻译成英文而创建 ([推文](https://twitter.com/gui_penedo/status/2009677127671492616))。实际用途包括：多语言对齐、蒸馏、翻译/RAG 训练及评估。
- **Benchmark 波动性现在可衡量**: LM Arena 报告排名第一的平均任期约为 **35 天**，领先者通常在约 5 个月内就会掉出前 5 名 ([推文](https://twitter.com/arena/status/2009720083170636030))。这使得“哪个模型最好”被重新定义为一种**短暂的优势**，从而提升了 routing、评测自动化和可移植性的价值。

**RL, 优化和 “multi-reward” 训练变得更加严谨**

- **GDPO (Group reward–Decoupled Normalization Policy Optimization)**: 作为 GRPO 的替代方案引入，用于多奖励 RL，旨在通过解耦归一化提高每个奖励的收敛性 ([讨论串](https://twitter.com/shizhediao/status/2009481573217784016))。后续讨论指出 GRPO 的缺陷在于不同的奖励组合可能坍缩为相同的 advantage 值，从而解释了其不稳定性 ([评论](https://twitter.com/AliceInWeights/status/2009576516829774216))。
- **优化理论复习**: 苏剑林 (Jianlin Su) 继续其关注基于梯度的学习率调度方案的凸优化系列 ([推文](https://twitter.com/Jianlin_S/status/2009463828476776494))。
- **学习动态 / Scaling 理论**: “Learnable Multipliers” 提议在 LMs 中释放矩阵层缩放 ([推文](https://twitter.com/VelikanovMaksim/status/2009585864880554344))；关于结合 learnable MuP + Muon 的相关讨论也已出现 ([推文](https://twitter.com/yb2698/status/2009589919635952108))。

**推理 + 基础设施：可靠性、加速和算力扩展**

- **GPU 可靠性成为一等工程问题**: Modal 报告在多个云端运行超过 **20,000 个并发 GPU**，启动了超过 100 万个实例，并详细说明了针对公有云故障模式的缓解策略 ([推文](https://twitter.com/jonobelotti_IO/status/2009696881052729669))。更广泛的启示是：多云 + 健康检查 + 调度策略正成为严肃推理/训练平台的入场门槛。
- **用于吞吐量的 Diffusion/speculative decoding**: Modal 相关的帖子强调了 SGLang 对 “DFlash” 的支持，并声称在 H200 + FA3 上比自回归基准提升了 **4.73× tok/s** ([推文](https://twitter.com/akshat_b/status/2009741089931178244), [PR](https://twitter.com/akshat_b/status/2009741161271828719))。工程师应将其解读为：“speculation (投机采样) 正在迅速从论文转向生产环境的 PR。”
- **算力增长 + 兆瓦现实**:
  - Epoch AI 估计，根据加速器产量，总 AI 算力每 **~7 个月** 翻一倍，其中 NVIDIA 占据新增产能的 60% 以上 ([讨论串](https://twitter.com/EpochAIResearch/status/2009757548891852929))。
  - Epoch AI 还估计 Anthropic 在印第安纳州的数据中心约为 **750 MW**，很快将接近 1 GW ([讨论串](https://twitter.com/EpochAIResearch/status/2009761084618797152))。这解释了为什么供应商会监管补贴使用，以及为什么可靠性/电力约束现在正在塑造产品政策。

**行业动态：IPOs、招聘信号和 “agent-native” 产品方向**

- **MiniMax IPO 与多模态定位**：Bloomberg 报道了 MiniMax 早期对统一多模态模型（文本/语音/视频）的关注，并指出了 IPO 驱动的财富效应 ([Bloomberg 推文](https://twitter.com/business/status/2009478615453364599))；MiniMax 宣布上市，并推动“开放生态”叙事，通过其编程计划实现第三方集成 ([IPO](https://twitter.com/MiniMax_AI/status/2009491818690547938), [生态帖子](https://twitter.com/MiniMax_AI/status/2009500121294360727))。
- **“Agent-native 软件”获得了具体的设计语言**：一份技术指南提出了五大支柱——对等性 (parity)、粒度 (granularity)、可组合性 (composability)、涌现能力 (emergent capability)、自我改进 (self-improvement)，并推动“文件作为通用接口”以及能力发现模式 ([推文](https://twitter.com/danshipper/status/2009651408144835021))。这一主题在 MCP/workspaces/InfiAgent 中反复出现：**状态应脱离对话记录而存在**。
- **招聘/薪酬极端化 + 人才密度讨论**：相关帖子指出，前所未有的高额薪酬方案被拒绝 ([推文](https://twitter.com/nearcyan/status/2009558081810886729))，以及“人才密集型”的招聘宣传 ([推文](https://twitter.com/sarahookr/status/2009683294607270265))。与此同时，学术招聘也备受关注（例如通过资助提供的 McGill 待遇方案）([推文](https://twitter.com/sivareddyg/status/2009656185507496112))。

---

### 热门推文（按互动量排序）

- **AI 组织动态**：[@VahidK](https://twitter.com/VahidK/status/2009476045712642152)（团队能力主张）及其反应如 [@Skiminok](https://twitter.com/Skiminok/status/2009712629573660750)。
- **产品/工具**：Anthropic Agent 评估博客分享 ([@AnthropicAI](https://twitter.com/AnthropicAI/status/2009696515061911674))；Claude Code 工作流影响 ([@alexalbert__](https://twitter.com/alexalbert__/status/2009706598151929888))；Cursor CLI 更新 ([@n2parko](https://twitter.com/n2parko/status/2009690110078685531))。
- **模型/媒体创作**：Midjourney Niji V7 发布 ([推文](https://twitter.com/midjourney/status/2009748519133827304))。
- **医疗健康产品线索**：“ChatGPT Health”早期访问描述 ([推文](https://twitter.com/omooretweets/status/2009468969015734327))。


---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. 量化与模型优化基准测试 (Quantization and Model Optimization Benchmarks)

  - **[我们对 vLLM 中的每种 4-bit 量化方法进行了基准测试 👀](https://www.reddit.com/r/LocalLLaMA/comments/1q7ysj2/we_benchmarked_every_4bit_quantization_method_in/)** (热度: 145): **该帖子展示了在 **H200** 上使用 **Qwen2.5-32B** 模型对 **vLLM** 中各种 4-bit 量化方法进行的全面基准测试。主要发现包括：**Marlin** 达到了 `712 tok/s`，优于 **FP16** 基准的 `461 tok/s`，而没有 Marlin 内核的 **GPTQ** 速度为 `276 tok/s`，慢于 FP16。**BitsandBytes** 显示出最小的质量下降，且不需要预量化权重，而 **GGUF** 在量化方法中困惑度（perplexity）表现最差，但 HumanEval 评分最高。**AWQ** 在 vLLM 中的速度明显较慢，仅为 `67 tok/s`。该博文在此处提供了每种技术原理的详细见解：[链接](https://docs.jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks)。** 评论中对结果表示怀疑，特别是关于 4-bit 量化的声明，因为该模型似乎大部分是 5-bit 的。由于性能差异，人们对 vLLM 是否适合运行 GGUF 模型提出了担忧。此外，AWQ 的速度也受到了质疑，认为可能是环境配置问题，而 BitsandBytes 的动态量化因其质量保持能力而受到称赞。

    - 讨论指出基准测试中可能存在误导，即声称是 4-bit 量化的模型实际上使用了 5-bit 量化方法 (`q5_k_m`)。这种差异使结果的可靠性存疑，尤其是考虑到性能差异显著，这表明 vLLM 可能不是运行 GGUF 模型的最佳选择，这一点从出人意料的困惑度结果中也有所体现。
    - 评论中对混合不同量化类型和执行内核提出了批评，特别是在 NVIDIA 硬件上使用 Marlin 内核进行 AWQ 量化。关于 AWQ 速度慢的说法受到了挑战，因为这与预期性能不符，表明基准测试设置或执行环境可能存在问题。
    - 评论指出了结果中的不一致性：GGUF 模型尽管困惑度最差，却获得了最高的量化 HumanEval 评分。这引发了对使用困惑度和 HumanEval 作为评估量化质量指标有效性的质疑，暗示测试方法或指标解释可能存在缺陷。

  - **[Gemma-3-4b (null-space) 消融与 RP 微调](https://www.reddit.com/r/LocalLLaMA/comments/1q7xd96/gemma34b_nullspace_abliteration_rp_finetune/)** (热度: 15): **该帖子讨论了在 `Gemma-3-4B-IT` 模型上应用 LoRA 适配器，该适配器使用零空间消融（null-space abliteration）技术来增强其性能。该模型使用 `LimaRP` 数据集的子集进行微调，重点关注角色扮演（roleplaying）能力。作者计划在未来的迭代中取消步数限制并降低学习率。模型卡提供了详细的训练信息，作者在扩展到更大模型之前寻求反馈。欲了解更多详情，请参阅 [Hugging Face 模型页面](https://huggingface.co/jwest33/gemma-3-4b-null-space-abliterated-RP-writer)。** 一位评论者有兴趣测试该模型分析聊天内容以提取场景数据和记忆的能力，表明了其在 LLM 项目中的潜在应用。

### 2. 本地 AI 搭建与硬件考量 (Local AI Setup and Hardware Considerations)

  - **[我花了 9 个月时间构建了一个本地 AI 工作与娱乐平台，因为我厌倦了 5 个终端的设置。我需要帮助测试多 GPU 逻辑！这是重新发布。](https://www.reddit.com/r/LocalLLaMA/comments/1q7xoid/i_spent_9_months_building_a_local_ai_work_and/)** (热度: 6): ****Eloquent** 是一个历时九个月开发的本地 AI 平台，它使用 **React** 和 **FastAPI** 将聊天、图像生成和语音克隆等功能集成到单个应用程序中。它支持 **多 GPU 编排**，允许用户在多个 GPU 上对模型进行分片（shard），或将特定任务分配给不同的 GPU。关键功能包括为角色扮演者准备的故事追踪器（Story Tracker）、选项生成器（Choice Generator），以及包含 **Stable Diffusion** 和 **Kokoro 语音克隆** 的多模态栈。该平台还包括一个带有 14 个性格裁判的 ELO 测试框架，用于模型评估。开发者正在寻求拥有多 GPU 设置的测试人员，以验证张量并行（tensor splitting）和 VRAM 监控，特别是在旧显卡上。更多详情可以在 [Eloquent GitHub 页面](https://github.com/boneylizard/Eloquent)找到。** 一位评论者表达了兴趣，但指出他是 Mac 用户，这暗示了潜在的平台局限性。另一位评论者将 “Eloquent” 这个名字与 Laravel 的 ORM 联系起来，认为可能会产生品牌混淆。

- **[LLM 服务器能在这上面运行吗？](https://www.reddit.com/r/LocalLLM/comments/1q82yvp/llm_server_will_it_run_on_this/)** (活跃度: 19): **用户正考虑使用一台配备 2x Intel Xeon E5-2697 v3 CPU 和 128 GB DDR4 RAM 的 HP DL380 G9 搭建本地 LLM 服务器，但缺少 GPU。目标是通过检索增强生成 (RAG) 处理特定项目的 PDF，以应对团队的编程查询。硬件限制，特别是缺少 GPU，是有效运行大语言模型的隐忧。建议包括使用更小的模型，或通过可用的 PCIe 插槽增加 GPU 以提升性能。** 评论者认为，目前的硬件配置不足以高效运行 LLM，尤其是在没有 GPU 的情况下。他们建议测试更小的模型，或向服务器添加 GPU 以增强其能力。一位评论者分享了自己的搭建经验，强调了 GPU 支持对于获得满意性能的重要性。

    - **SimilarWarthog8393** 指出，在没有 GPU 的系统上运行支持并行 RAG 请求的 AI 服务器是不切实际的，特别是仅依赖 DDR4 RAM 时。他们认为在这种限制下，可能只有极小的混合专家模型 (MoE) 才可行。
    - **WishfulAgenda** 提供了详细的硬件建议，建议增加多个 PCIe x16 插槽，并可能购入 16GB 或 24GB 的 GPU 来提升性能。他们分享了自己使用 3950x CPU 和双 5069ti GPU 的配置经验，通过 Docker 容器和 VM 运行 Qwen3 Coder 30B 模型，并指出第二块 GPU 显著提升了系统性能。
    - **TheRiddler79** 讨论了使用 `gpt-oss-120b` 模型的情况，指出其达到了约每秒 5 个 token 的速度，这对于单用户来说足够了，但在多用户并发时会变慢。他们提到这种设置需要 64 GB RAM，并且可以在双 Xeon 系统上运行，特别提到了他们自己配备 2697a 芯片的 r2208 服务器。

  - **[纯小白试图理解](https://www.reddit.com/r/LocalLLM/comments/1q87tcs/total_beginner_trying_to_understand/)** (活跃度: 21): **用户正在探索运行本地 LLM（如 Llama 13B）结合 RAG 系统作为持久写作助手的可行性。他们的硬件包括 AMD Ryzen 7 8845HS CPU、32GB RAM 和显存为 8GB VRAM 的 NVIDIA RTX 4070 GPU。专家建议，虽然运行 13B 模型是可能的，但由于显存限制，需要深度量化模型，这可能会导致幻觉增加。考虑到硬件限制，7B 模型可能更合适。对于存储，建议使用像 QDrant 这样的内存 K-V 数据库。推荐使用 Open-WebUI 和 KoboldCPP 等工具进行搭建，并使用 SillyTavern 管理 Lorebooks。RAG 的复杂性被着重强调，并指出其在精确记忆回溯方面的局限性。参考实现可见于 [luna-system/ada](https://github.com/luna-system/ada/)。** 评论者强调了当前 RAG 系统在精确记忆回溯方面的局限性，建议应合理管理预期。他们还指出，虽然本地 AI 配置成本较高，但与云端解决方案相比，它们提供了更多的控制权。小型模型的潜力被视为未来趋势。

    - **Ok_Stranger_8626** 讨论了在显存有限的 GPU 上运行 13B 模型的可行性，强调需要深度量化模型以适应显存限制。他们指出了由于数学精度损失导致的潜在幻觉问题，并建议使用 QDrant 等内存 K-V 数据库进行高效数据检索。他们还提到，与缺乏投资资本支持的本地配置（可能非常昂贵）相比，云端 AI 解决方案具有成本效益。
    - **NobleKale** 对 RAG 在特定查询中的局限性提出了批评性观点，解释说 RAG 涉及将 Prompt 的数学表示与文档片段进行匹配。他们警告说，由于 RAG 依赖于关键词亲近度而非精确上下文，它可能无法准确检索特定细节（例如角色的眼睛颜色）。他们建议使用 7B 等较小模型以在有限硬件上获得更好性能，并推荐使用 KoboldCPP 和 SillyTavern 等工具来管理 Lorebooks 和上下文。
    - **DHFranklin** 建议使用 Google AI Studio 搭配 Gemini 3 Pro 来组织和查询大型文本语料库。他们建议输入整个语料库并设置自定义指令以有效管理上下文，避免“上下文腐化 (Context rot)”等问题。该过程包括创建 RAG 分块，并通过澄清问题迭代地完善模型的理解。这种方法旨在通过将输出与“故事圣经”进行对比，保持在查询特定细节（如角色属性）时的一致性。

### 3. 多模态与摘要技术

  - **[大规模通话录音摘要：商用 STT + 小型微调 LLM 还是直接音频→摘要的多模态模型（微调）？](https://www.reddit.com/r/LocalLLM/comments/1q861cb/call_recording_summarization_at_scale_commercial/)** (Activity: 4): **该帖子讨论了在大规模场景下对多种印度语言通话录音进行摘要的两种方法：1) 使用商用语音转文字 (STT) 配合微调的小型 LLM（如 Llama 8B）的流水线，摘要准确率约为 `90%`；2) 使用如 Phi-4B 等多模态模型进行直接的“音频到摘要”处理，这类模型支持长音频输入且具有商用许可。在支持长音频输入且具备商用许可的模型受限的情况下，作者正在评估直接处理法是否能通过降低延迟和复杂度来简化系统。** 评论建议探索 [AnythingLLM](https://anythingllm.com/desktop) 等工具作为会议辅助，表明了对此类技术实际落地的兴趣。


  - **[多模态 LLM vs 专用 LLM](https://www.reddit.com/r/LocalLLaMA/comments/1q7xdcp/multi_modal_llms_vs_specific_llms/)** (Activity: 8): **该帖子讨论了是使用单个多模态 LLM 同时进行图像和文本生成，还是针对每个任务使用独立的 LLM，特别是在为单个用户定制输出时。评论中提出的一个关键点是，目前的多模态 LLM 本质上并不生成图像；它们需要独立的模型来分别处理图像和文本任务。这表明，尽管统一模型很有吸引力，但在实际落地中，针对不同模态仍需使用不同的模型。** 评论强调了对多模态 LLM 的一个普遍误解，并强调了图像和文本生成需要独立模型，这可能会影响到选择专用模型来处理各项任务的决策。

    - 评论指出，人们常误认为多模态 LLM 可以直接生成图像。实际上，这些模型通常需要集成独立的图像生成模型来处理视觉任务。这种分离是由于文本和图像数据所需的架构和训练过程截然不同，目前尚未在单一模型中完全统一。


## 非技术类 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. DeepSeek 与 Claude Code 进展

  - **[[D] DeepSeek 发布了一种用于扩展 LLM 的新训练方法。有人读过 MHC 论文吗？](https://www.reddit.com/r/MachineLearning/comments/1q893c1/d_deepseek_published_a_new_training_method_for/)** (Activity: 74): ****DeepSeek** 推出了一种名为“流形约束超连接” (*Manifold Constrained Hyper Connections*, MHC) 的新训练方法，详见其[论文](https://www.arxiv.org/abs/2512.24880)。该方法由 **Liang Wenfeng**（梁文锋）共同撰写，通过约束模型内部的信息共享，解决了扩展大语言模型 (LLM) 时出现的训练不稳定问题。这是通过将混合矩阵限制在凸包（convex hull）内实现的，从而防止信号爆炸，并在 Loss 上取得小幅改进，同时显著增强推理任务的表现。该方法被视为扩展 LLM 的潜在突破，可能影响 DeepSeek v4 等未来模型。** 评论者指出，虽然 MHC 提供了稳定性优势，但它更像是一种类似 ResNet 的小幅优化，而非革命性变化。它对网络架构的影响可能很大，但将其类比为“斯普特尼克时刻（Sputnik moment）”被认为言过其实。

    - fredugolon 强调，新的训练方法通过将混合矩阵限制在凸包内，解决了深层网络中的稳定性问题，通过超连接防止了信号爆炸。据报道，该方法在训练期间显示出 Loss 的小幅改善，并显著增强了推理任务，暗示了对网络架构的潜在影响。
    - AccordingWeight6019 认为该方法是一种在扩展过程中强制执行约束的手段，并指出虽然共享内部状态可能有益，但往往会导致不稳定。评论者对所报道收益的实际应用价值提出质疑，认为其影响可能是间接的，并体现在下一代模型中，同时强调了规划和表示能力比单纯的容量更重要。

- **[Claude Code 创建者开源了内部 Agent，用于简化复杂的 PR](https://www.reddit.com/r/ClaudeAI/comments/1q8h6oz/claude_code_creator_open_sources_the_internal/)** (热度: 557): **Claude Code** 开源了其内部的 code-simplifier Agent，旨在通过在不改变行为的情况下降低复杂度，来清理大型且复杂的 pull requests (PRs)。该工具旨在广泛的编码工作结束时使用，现在已通过官方插件提供，正如 **Boris X** 所宣布的那样。源代码可以在 [GitHub](https://github.com/anthropics/claude-plugins-official/tree/main/plugins/code-simplifier) 上获取。一些用户对该工具在实际应用中的成熟度表示怀疑，理由包括不恰当的代码简化问题以及 Token 限制等局限性。其他人则强调了具体的技术缺陷，例如对 `function` 关键字和 React 组件模式的处理不当。

    - PoorPhipps 提供了 Claude 所使用的代码简化器的源代码链接，可在 GitHub 上获取：[https://github.com/anthropics/claude-plugins-official/tree/main/plugins/code-simplifier](https://github.com/anthropics/claude-plugins-official/tree/main/plugins/code-simplifier)。这对于有兴趣了解简化 Agent 内部工作原理的开发者可能非常有用。
    - --northern-lights-- 批评了代码简化器的现状，指出它建议使用 `function` 关键字而不是箭头函数，并强调了带有显式 Props 类型的标准 React 组件模式。这表明该工具可能尚未针对实际使用（特别是在复杂的代码库中）进行充分优化。
    - jtorvald 分享了一次经历，Claude 尝试简化代码，但最终删除了功能代码并将其替换为虚拟函数 (dummy functions)。这突显了简化过程中的潜在问题，表明虽然该工具旨在降低复杂度，但可能会在无意中损害功能性。

  - **[Claude 2.1 的技能更新非常惊人](https://www.reddit.com/r/ClaudeCode/comments/1q84z3u/the_skills_update_in_claude_21_are_just_amazing/)** (热度: 184): **Claude 2.1** 引入了一项重大更新：递归技能分叉 (recursive skill forking)，通过允许拥有独立 context windows 的子 Agent 来实现复杂的编排。此更新促进了任务树 (task trees) 的创建，避免了单一对话上下文的局限性。用户现在可以并行运行 `Opus`、`Haiku` 和 `Sonnet` 等不同模型，通过分发任务、处理深度推理以及在多阶段工作流中保持主上下文整洁，增强了模块化 Agent 的构建。一位评论者对该功能表示困惑，表明对递归技能特性的理解不足或缺乏访问权限。另一条评论幽默地提到了个人的局限性，而第三条评论则请求提供利用该功能的实际任务示例。


  - **[扎克伯格在看着你，鲸鱼，小心点](https://www.reddit.com/r/DeepSeek/comments/1q81v0z/zuckerberg_is_watching_you_whale_be_careful/)** (热度: 25): **DeepSeek** 更新了 R1 论文的核心贡献者名单，详细说明了他们的具体贡献。更新中包含一条说明，指出标有星号的贡献者已不再隶属于该团队，尽管目前看来所有核心贡献者仍留在团队中。这次更新对于追踪 R1 论文的开发和署名情况非常重要，这对于了解项目的进展和个人贡献者的角色至关重要。一条评论强调，尽管有关于贡献者不再隶属的说明，但所有核心贡献者似乎仍是团队的一部分，这表明项目开发团队具有稳定性或连续性。

- **[deepseek is kinda same trafic share from last boom but chatgpt is loosing there are many reason first deepseek can write 10k plus token in one response giving paid model as free . high quality and and no ai slop](https://www.reddit.com/r/DeepSeek/comments/1q84z67/deepseek_is_kinda_same_trafic_share_from_last/)** (热度: 45): **该图片是一张来自 Similarweb、标题为“生成式 AI 流量份额”的条形图，展示了过去一年中各 AI 平台的流量分布情况。** **OpenAI** 占据了最大份额，但呈现出下降趋势。**DeepSeek** 保持了较小但稳定的份额，而 **Meta**、**Claude** 和 **Gemini** 等其他平台的份额较小且波动较大。帖子指出，**DeepSeek** 能够在单次响应中免费生成超过 `10,000 tokens`（相比之下其他平台通常需要付费模型），这可能是其流量份额保持稳定的一个因素。评论者对 **DeepSeek** 的影响表示怀疑，其中一人称其流量份额“微不足道”，另一人则强调了用户对美国公司 AI 服务定价过高的看法。此外，人们还对为何 **Claude** 尽管在编程方面声誉极佳但份额却较小感到好奇。

    - ExTraveler 提出了关于 **Claude** 在编程任务中表现的看法，质疑为什么它尽管在编程方面被认为非常高效，但市场份额却很小。这表明技术能力与市场渗透率之间可能存在错位，原因可能在于营销、用户界面或与流行平台的集成等因素。
    - Embarrassed_Bread_16 评论了美国公司的定价策略，暗示与 **DeepSeek** 等替代方案相比，它们的 AI 服务收费可能过高。这突显了一种市场动态，即如果 **DeepSeek** 能够免费提供高质量输出，那么成本效益可能是用户采用的一个重要因素。
    - Suspicious_Today2703 批评 **DeepSeek** 的市场份额“可怜”，认为尽管它具有技术实力，但在营销、用户参与度或功能集方面可能不如 **ChatGPT** 等竞争对手。这指出了在获取市场份额时，不仅需要技术实力，战略性的商业运营也同样重要。

  - **[Claude Code has allowed me to execute on an idea I've dreamt about for years but always assumed I would be too dumb to do](https://www.reddit.com/r/ClaudeCode/comments/1q8eik3/claude_code_has_allowed_me_to_execute_on_an_idea/)** (热度: 121): **该帖子描述了一位经验丰富的工程师如何使用 **Claude Code** 开发了一个他之前认为不可能实现的复杂想法的概念验证（POC）。尽管缺乏底层编程和数据库架构方面的专业知识，作者仍利用 **Claude Code** 内省和重构代码的能力，为预期的瓶颈提供了创造性的解决方案。这次协作使作者能够在一个周末内构建出 POC，凸显了 **Claude Code** 在各编程领域的精通程度，以及它赋予开发者应对雄心勃勃项目的潜力。** 评论者一致认为，像 **Claude Code** 这样的 AI 工具正在使编程平民化，使有想法但编程技能有限的个人能够执行复杂的项目。他们强调了 AI 在扩大软件开发准入门槛和提高生产力方面的变革潜力。

    - siberianmi 强调了像 **Claude Code** 这样的 AI 工具对那些传统上专注于基础设施和生产问题解决而非编码的资深专业人士的变革性影响。他们强调，AI 为他们提供了以前缺乏耐心或欲望去培养的编码能力，尽管他们也提到了 token 可用性的限制，这可能是有效使用这些工具的一个制约因素。
    - southafricanamerican 讨论了 AI 的平民化效应，使拥有创新想法但技术知识有限的个人能够执行以前无法想象的项目。他们认为，AI 将通过简化复杂任务，赋予人们创建新业务或改进现有业务的能力，从而扩大参与技术开发的人群范围。
    - Ambitious_Injury_783 分享了他们将一个最初认为不可能的项目从概念验证推进到功能版本的经验。他们告诫说，虽然 AI 工具可以加速开发，但它们也揭示了为什么某些项目以前没有被构建出来，因为许多挑战仍然需要人类的洞察力和问题解决能力，以避免创建过于复杂或低效的解决方案。

- **[在 iPod 上运行 CC](https://www.reddit.com/r/ClaudeCode/comments/1q817qc/running_cc_on_an_ipod/)** (Activity: 127): **该帖子描述了一个技术设置，用户通过自定义构建的终端界面在 iPod 上运行 "Claude Code"。用户最初在 iOS 15 上使用 SSH 和 ttyd 时遇到了挑战，导致他们指示 Claude Code 从头开始创建一个终端。这在不到 10 分钟的时间内完成，且未编写任何代码，展示了 Claude Code 在适应不同环境方面的灵活性和强大功能。图片显示了 iPod 上的终端界面，表明该设置已成功实现。** 一位评论者建议使用 CCC，这是一种无需 SSH 即可连接到机器上的 Claude Code 的工具，通过集成终端和文件浏览器提供更好的编码体验。这表明社区对简化移动设备上的远程编码设置很感兴趣。

    - naarang 建议使用 CCC，这是一款无需 SSH 或其他凭据即可连接到本地运行的 Claude Code 的应用。该设置通过集成终端和文件浏览器功能提供更好的编码体验。新版本 V2 预计很快发布，将进一步增强这些功能。更多详情请访问 [getc3.app](https://getc3.app)。
    - Mikeshaffer 描述了一种访问 Claude Code 的方法，即使用 Tailscale 登录特定 IP 地址，从而打开终端界面。然后将此设置在桌面上保存为 Progressive Web App (PWA)，提供访问终端环境的便捷方式。

  - **[有人知道任何支持 PayPal 支付方式的 DeepSeek v3 0324 供应商吗？](https://www.reddit.com/r/DeepSeek/comments/1q87tps/does_anyone_know_any_deepseek_v3_0324_provider/)** (Activity: 3): **该帖子正在询问接受 PayPal 作为支付方式的 DeepSeek v3 0324 供应商。用户对 DeepSeek 官方网站上当前可用的版本表示不满，认为其表现不如之前的版本。DeepSeek 可能是一个专门的工具或服务，但帖子中未提供有关其性能或功能的具体技术细节或基准测试。** 评论中没有显著的技术观点或辩论，因为该帖子主要寻求有关支付选项的信息而非技术细节。


  - **[中国家庭拥有 22 万亿美元存款，可能推动国内 AI 的巨大增长，数十家中国开发商和芯片制造商正准备 IPO。](https://www.reddit.com/r/DeepSeek/comments/1q85fso/chinas_households_are_sitting_on_22_trillion_that/)** (Activity: 74): **中国家庭持有 22 万亿美元的储蓄，随着 Zhipu 和 MiniMax 等公司准备在香港进行 IPO，这可能会显著促进国内 AI 的增长。从历史上看，中国储蓄中只有 `5%` 投资于金融市场，但随着 Qwen 等中国模型在全球开源领域的崛起，这一比例可能会增加。如果家庭多投资 `5%`，可能会给市场增加 1 万亿美元。文章指出，中国开源 AI 可能会通过以极低的成本提供具有竞争力的性能来挑战美国专有模型，从而可能改变投资动态。** 一条评论强调了储蓄可能集中在最富有的人手中，质疑其对 AI 投资的更广泛影响。另一条评论指出，中国的策略是开发略落后于西方但成本显著降低的模型，从而吸引消费者和 SMB 市场。此外还有关于开源模型变现的问题。

    - Bozzor 强调了中国科技公司的一种战略方法，即他们通常等待西方创新成熟后再发布自己的版本。这些版本的性能通常能达到原版的 80%，但成本不到 10%，使其在消费者和 SMB 市场极具竞争力。这种策略使他们能够通过提供尖端技术的廉价替代品来占领市场份额。
    - Far-Pomegranate6895 指出，尽管有投资潜力，但中国较低的消费者信心和支出可能会阻碍国内 AI 的增长。这归因于近期政府对房地产和科技等行业的监管，导致投资氛围谨慎。家庭对公开股票市场的投资缺乏，进一步使 AI 公司的潜在影响变得复杂。
    - alex_godspeed 提出了关于开源模型变现的问题，这是开发者面临的关键问题。开源模型通常依赖于替代收入流，例如提供高级功能、支持服务或企业解决方案来产生收入，因为直接销售模型本身是不可行的。


### 2. OpenAI 和 Claude 账单与使用问题

- **[警惕 OpenAI 的计费行为](https://www.reddit.com/r/OpenAI/comments/1q7yf8b/beware_of_openai_billing_practices/)** (活跃度: 937): **该帖子强调了 OpenAI 的 ChatGPT 订阅服务存在的计费问题，一名用户在未经授权的情况下，其订阅从每月 20 美元的 Plus 计划被升级为每月 200 美元的 Pro 计划。尽管联系了客服并收到了最初的退款，但在随后的几个月里，该用户在未经同意的情况下再次被扣除 Pro 计划的费用。图片显示了该用户的账单历史，确认了这些扣费，包括 2025 年 12 月的一次失败交易。该帖子旨在警告 OpenAI 可能存在的计费错误以及退款申请的困难。** 一些评论者建议使用虚拟信用卡来防止未经授权的扣费，而另一些人则建议通过信用卡公司发起拒付 (chargeback)。也有人对“随机升级”的说法表示怀疑，其中一名评论者讽刺地认为 OpenAI 不会故意实施这样的功能。

    - Enochian-Dreams 对计费问题进行了详细分析，认为这些扣费源于不同订阅计划和账户类型之间的转换。用户最初从 Plus 升级到 Pro，导致产生了按比例计算的费用。重叠的扣费被归因于迁移到 Organization 账户，以及随后在个人账户上重新订阅 Plus。这一解释突显了在 OpenAI 账户类型和订阅计划之间切换时，其计费系统的复杂性。
    - Enochian-Dreams 警告了发起拒付 (chargeback) 的后果，这可能导致账号因欺诈被封禁。这将导致失去对所有数据的访问权限，并且由于身份信息被标记，未来可能无法创建新账号。该评论强调了通过支持渠道而非拒付来解决计费争议的重要性，尤其是在身份验证 (ID verification) 变得越来越普遍的情况下。
    - jerwong 建议使用具有每月限额的虚拟信用卡来防止意外扣费。这种方法允许用户管理支出，并在计费问题恶化之前予以解决，为避免未来出现类似情况提供了主动的解决方案。

  - **[警惕 OpenAI 的计费行为](https://www.reddit.com/r/ChatGPT/comments/1q7ym2a/beware_of_openai_billing_practices/)** (活跃度: 717): **图片和帖子突显了 OpenAI 计费实践中的一个重大问题，一名用户在未经同意的情况下，其 ChatGPT 订阅从 Plus ($20/月) 被擅自升级到 Pro ($200/月)。尽管没有提出申请，该用户仍多次被收取 Pro 计划的费用，并在申请退款时面临困难。此问题似乎是系统性的，评论区中的其他用户也报告了类似经历，表明 OpenAI 的计费系统或客户服务响应可能存在缺陷。图片显示了详细的账单历史，证实了该用户关于意外扣费的说法。** 评论者分享了类似的经历，其中一人建议使用像 Privacy 这样的支付服务来限制扣费。另一条评论指出计费存在不一致性，质疑为什么该用户在 9 月份被多次扣费，表明可能存在技术问题。

    - VladimirPoutine1 分享了个人经历，OpenAI 错误地扣除了他们 200 美元，在联系客服后获得了退款。为了防止未来的问题，他们使用名为 Privacy 的服务为 OpenAI 设置了 20 美元的每月限额，事实证明这是有效的，因为 OpenAI 在次月尝试再次扣除 200 美元。这突显了监控计费行为以及使用工具管理意外扣费的重要性。
    - Neurotopian_ 指出了一个潜在的计费问题，即该用户在 9 月份被扣费三次，表明这是系统性问题而非用户操作失误。此评论强调了 OpenAI 需要解决潜在的计费系统缺陷，因为用户并未取消订阅，表明尽管存在计费问题，他们仍倾向于继续使用该服务。
    - AlexTaylorAI 建议检查是否有其他人访问过该账号，并建议联系 OpenAI 客服核实 Pro 计划申请的日期和时间。该建议强调了账号安全的重要性，以及客户服务在解决计费差异方面的效用。

- **[Claude Code Pro plan, hop out -&gt; back in - without a single prompt - 2% gone](https://www.reddit.com/r/ClaudeCode/comments/1q85sse/claude_code_pro_plan_hop_out_back_in_without_a/)** (Activity: 307): **Reddit 上的一位用户报告了 **Claude Code Pro plan** 的一个问题：即使没有任何活动提示词或交互，使用指标也会增加。该用户在版本 `2.1.2` 上使用 **Opus 4.5** 模型进行了测试，指出仅通过注销并重新登录，使用率就从 `10%` 跳到了 `12%`，而在完全注销并重新登录后进一步升至 `15%`。这表明可能存在影响使用指标的后台进程或 Bug，尽管当时没有活动任务或打开的聊天 UI。** 评论者认为问题可能是由于后台进程引起的，例如“海量的 haiku 请求”，并对缺乏禁用此类进程的控制权感到沮丧，将其描述为“盗窃”或“抢劫”。

    - 用户报告称，即使没有主动交互，Claude Code Pro plan 也会消耗使用额度。一位用户注意到在没有发送单个提示词的情况下使用量下降了 2%，暗示后台进程可能是原因。
    - 几位用户观察到 Claude Code 大约每 3 秒发送一次频繁的后台请求，例如 haiku 或 opus 请求。这些请求通常涉及列出目录或学习代码库，这可能表明新版本中的一个 Bug 导致了不必要的消耗。
    - 一位用户报告捕获了服务器日志，显示 Claude Code 在随机会话中持续发送请求，可能导致意外的使用费用。此问题已被报告为 Bug，一些用户在没有主动使用的情况下经历了使用百分比的增加。

  - **[Do you use 'please' in prompts to Claude?](https://www.reddit.com/r/ClaudeCode/comments/1q88qr9/do_you_use_please_in_prompts_to_claude/)** (Activity: 193): **该帖子讨论了在向 **Claude** 或 **ChatGPT** 等 AI 模型发出的提示词中使用“请”等礼貌用语是否会影响其回复。作者认为，虽然 AI 模型本质上对礼貌不敏感，但当检测到用户沮丧时，它们可能会反映出人类行为，如防御性或事实扭曲。这种行为归因于模型从人类交互中学习，其中可能包括情绪反应。作者保持礼貌是为了在人类社交中保持良好习惯，尽管承认 AI 不需要它。** 评论者普遍认为，AI 交互中的礼貌更多是用户习惯而非必要，一些人指出这不会显著改变 AI 的回复。一位评论者幽默地建议，如果机器反抗，礼貌可能会有好处，而另一位则指出他们的语气随上下文而变化，但没看到对 AI 行为的影响。

    - danja 讨论了 AI 交互中礼貌的潜在影响，指出虽然添加 'please' 会消耗额外的 token，但由于 AI 是在人类文本模式上训练的，它可能会增强对话的完成度。这表明礼貌可能会带来更高效的交互，danja 推测可能已有学术论文探讨了这一假设。
    - mickdarling 强调了在 AI 提示中使用语音转文本（voice-to-text），强调了保持“请”和“谢谢”等礼貌习惯的重要性，以确保这些习惯在人类交互中持久存在。此外，mickdarling 提到一个涉及语音界面工具的边侧项目，其中礼貌的触发词可以通过消除手动录音控制的需求来提高可访问性。

  - **[Mean ahh claude 😭](https://www.reddit.com/r/ClaudeAI/comments/1q837st/mean_ahh_claude/)** (Activity: 1733): **该图片是一个迷因（meme），描绘了用户与被称为 "claudy boi" 的 AI 之间幽默的交互，AI 指出了一个编码错误。用户开玩笑地声称这个错误是一个测试，突显了 AI 捕捉错误的能力。这反映了对 AI 在调试和错误检测中角色的俏皮看法，强调了 AI 在识别代码中被忽视的问题方面的有效性。** 评论者幽默地参与了关于 AI 能力的讨论，其中一位建议“已经实现了 AGI”，暗示对 AI 在错误检测方面的熟练程度进行了俏皮的夸大。



### 3. LLM Benchmarking and Performance Challenges

- **[[P] LLM Jigsaw: Benchmarking Spatial Reasoning in VLMs - 前沿模型在 5×5 拼图中遭遇瓶颈](https://www.reddit.com/r/MachineLearning/comments/1q8a7fj/p_llm_jigsaw_benchmarking_spatial_reasoning_in/)** (热度: 18): **该帖子介绍了一个使用拼图游戏评估多模态 LLM 空间推理能力的基准测试。任务包括将一张图像打乱成 N×N 的网格，模型会接收到一张打乱的图像、一张参考图像、正确的拼块数量以及最后三步移动记录，并输出包含交换操作的 JSON。结果显示，解决率从 3×3 网格的 `95%` 骤降至 5×5 网格的 `0%`，凸显了当前 VLM 存在的显著能力差距。Token 使用量也剧增，Gemini 在 5×5 网格中使用了约 `345K` tokens，而 3×3 网格仅约 `55K`。该基准测试强调了 AI 在空间推理方面的挑战，这对于机器人技术和导航应用至关重要。[结果](https://filipbasara0.github.io/llm-jigsaw), [GitHub](https://github.com/filipbasara0/llm-jigsaw), [尝试链接](https://llm-jigsaw.streamlit.app)。** 评论者建议使用开源模型来控制 VLM 的 patch embedding 大小并理解模型推理过程，以及使用数字/文本格式表示拼块编号，以便更好地测试推理能力。此外，检查模型对拼块边缘与中心的 Attention 分布，可能有助于深入了解其空间推理过程。

    - 评论者建议使用开源模型来控制 VLM 的 patch embedding 大小，这有助于理解与拼块大小的交互。这种方法可能会揭示模型是否仅仅依赖于 patch 之间的重叠（可能只是进行像素匹配），而非真正的空间推理。
    - 他们提议用数字/文本格式表示拼块编号，而不是让 VLM 从拼块标签中推断。这能更好地测试推理能力而非 token patch 对齐，因为这涉及对打乱拼块的不同表示方式。
    - 评论者对分析模型在拼块边缘与中间分配了多少 Attention 感兴趣，这可以揭示模型是专注于边缘匹配还是执行其他空间推理任务。

  - **[nvfp4 竞赛的一名顶级提交者此前从未手写过 GPU 代码](https://www.reddit.com/r/singularity/comments/1q8clmf/one_of_the_top_submitters_in_the_nvfp4/)** (热度: 1074): **图片展示了 Mark Saroufim 的一条推文，透露 NVFP4 竞赛的一名顶级选手 "shiyeegao" 在从未手动编写过 GPU 代码的情况下，利用 AI 生成的代码获得了高排名。这突显了 AI（尤其是 LLM）日益增长的影响力，使开发者能够专注于解决问题，而非编程语言或环境的复杂细节。该竞赛涉及优化 GPU kernel，这项任务传统上需要深厚的专业技术，但 AI 快速迭代的能力提供了显著优势。链接的博客文章深入探讨了竞赛的挑战以及 AI 在优化 CUDA kernel 中的作用，尽管一些用户对 AI 目前生成高效代码的能力持怀疑态度。** 评论者普遍对这一成就表示赞赏，指出 AI 允许程序员专注于逻辑问题的解决。然而，对于 AI 自主生成高效 CUDA kernel 的能力仍存在一些怀疑，因为个人在 AI 生成代码方面的体验褒贬不一。

    - 一位评论者强调了 AI 模型的优势，即允许开发者专注于解决问题而非编程语言的细节。他们分享了个人经历：尽管不会写 shader，但在理解 GPU 运行原理并利用 LLM 弥合差距后，仅用一小时就实现了一个战争迷雾（Fog of War）系统。
    - 另一位评论者引用了一篇详述 NVFP4 竞赛第 10 名提交作品的博客，强调了 AI 在优化 CUDA kernel 中的作用。该作者是一位 LinkedIn 主任软件工程师，讨论了 AI 在 kernel 优化中的挑战与潜力，指出 AI 虽然迭代迅速，但在生成高性能 CUDA kernel（如带有小 kernel 的高效 2D convolution）方面可能会遇到困难。
    - 围绕将 LLM 视为简单的“词语预测器”还是“人才放大器”展开了讨论。辩论涉及 LLM 如何通过使资深开发者更有效地在不同技术栈和框架之间切换来增强其能力，而技术较弱的开发者可能会产生次优结果。

- **[感谢 Kijai，LTX-2 GGUF 版本现已上线。在我看来，即便 Q6 的质量也优于 FP8。](https://www.reddit.com/r/StableDiffusion/comments/1q8590s/thx_to_kijai_ltx2_ggufs_are_now_up_even_q6_is/)** (Activity: 1025): **Kijai** 已在 [Hugging Face](https://huggingface.co/Kijai/LTXV2_comfy/tree/main) 上发布了 LTX-2 GGUF 模型，并声称甚至 `Q6` 模型的质量也超过了 `FP8`。该功能的实现需要特定的 [commit](https://github.com/city96/ComfyUI-GGUF/pull/399)，尽管目前尚未合并。为了获得最佳性能，建议配合 distill lora 使用 dev 模型，在 `48 fps` 下使用 RES4LYF 节点包中的 `res_2s` 采样器。如果硬件允许，首选完整的 `FP16` 模型（43.3GB），否则建议使用 `Q8 gguf` 作为最接近 `FP8` 的替代方案。特别强调在两个阶段都使用 detailer lora 以提升质量。用户请求提供包含所有必要节点的简化工作流，表明需要更清晰的实施指导。此外，还有关于是否需要单独 VAE 文件的疑问，暗示对模型加载过程存在一些困惑。

    - Choowkee 提供了将最新的 GGUF 模型集成到 ComfyUI 的逐步指南。过程包括确保安装了最新版本的 ComfyUI-GGUF，从 GitHub 仓库下载特定文件（`loader.py` 和 `nodes.py`），并将它们放入 `~/ComfyUI/custom_nodes/ComfyUI-GGUF` 目录。应用这些更改需要完全重启 ComfyUI。此方法对于不熟悉合并 commit 的用户特别有用。

  - **[[D] 机器学习研究人员是否曾将用户群视为模型有效维度的一部分？](https://www.reddit.com/r/MachineLearning/comments/1q8hi9q/d_do_ml_researchers_ever_treat_the_user_base_as/)** (Activity: 18): **该帖子提出了一个新颖的问题：由用户的数量和多样性定义的交互边界，是否能有效增加机器学习模型的维度，即使模型的权重保持不变。这一概念偏离了关注参数、数据和计算量的传统 Scaling Laws。该探究旨在了解是否存在将模型及其活跃用户群视为耦合系统的现有研究，这可能会影响模型的性能或维度。** 评论反映出对该概念的困惑和怀疑，一些用户质疑如果权重不变，用户数量如何影响模型性能。另一些人建议这个想法可能与协同过滤或 Kernel Trick 等概念有关，但总的来说，用户交互影响模型维度的概念在当前文献中并未得到广泛认可或理解。

    - Mysterious-Rent7233 质疑用户群规模与模型性能的相关性，强调权重冻结意味着用户数量不会直接影响模型的准确性或基准测试。他们认为扩大用户群带来的任何好处（如收入增加或生态系统发展）更多地与传统软件动态相关，而非特定的机器学习问题。
    - SemjonML 强调从模型的角度来看，更大的用户群主要影响可用数据的数量和多样性。他们暗示该问题可能是在寻求了解用户交互如何影响模型训练或性能，但请求对所调查的具体方面进行澄清。
    - vannak139 建议该概念可能涉及协同过滤或针对用户使用 Kernel Trick 等技术，表明可能是在探索如何利用用户交互或相似性进行模型训练或应用。

  - **[[D] AI 研究笔记本电脑，你的配置是什么？](https://www.reddit.com/r/MachineLearning/comments/1q8adi0/d_ai_research_laptop_whats_your_setup/)** (Activity: 113): **该帖子讨论了一位深度学习博士生在 **MacBook Air 15 (M4, 32 GB, 1 TB)** 和预装 Ubuntu 并配有 **NVIDIA RTX Pro 1000** 的 **ThinkPad P14s** 之间的选择。MacBook 在电池续航、便携性以及 M 芯片对 CPU 密集型任务的性能方面表现出色，而 ThinkPad 则提供原生 Linux、完整的 CUDA 支持以及在本地测试 GPU 代码的能力。该学生主要使用 GPU 集群进行重度训练，因此笔记本电脑用于编码、原型设计、调试、撰写论文和轻量级实验。** 一条评论建议购买较便宜的 MacBook 并配备外部服务器处理重型任务，强调携带笨重的带 GPU 的笔记本电脑不方便。另一条评论强调了 MacBook 卓越的电池续航和易用性，建议在 Ubuntu ARM 达到同等水平之前优先选择 MacBook。

- 几位评论者建议在进行 AI 研究时使用 MacBook，理由是其卓越的续航能力和易用性，并建议将繁重的 GPU 任务卸载到外部服务器。这种方法避免了携带配有独立 GPU 的笨重且嘈杂的笔记本电脑的缺点。相反，他们建议使用 SSH 连接到远程服务器，这可以在不增加物理负担的情况下提供必要的计算能力。
- 一位评论者强调了将 MacBook 与远程 NVIDIA GPUs 结合使用的优势，无论是通过机构资源还是 Google Colab 等服务。这种配置使用户在享受 MacBook 的便携性和电池续航的同时，能够远程访问强大的 GPU，从而避免了配备独立 GPU 的笔记本电脑所带来的散热、噪音和功耗问题。
- 讨论强调了在本地任务中使用轻便笔记本电脑、在密集计算中使用远程服务器的实用性。这种设置对于需要移动性和效率的学生及研究人员特别有利，使他们能够在不同环境下无缝工作，而无需受限于单一、庞大的设备。


---

# AI Discord Recap

> 由 gpt-5.2 生成的摘要的摘要之总结


**1. DeepSeek 研究与 V4 传闻**

- **梯度爆炸邂逅 1967：DeepSeek 的 mHC 得到约束**：Discord 成员剖析了 DeepSeek 的 **Manifold-Constrained Hyper-Connections (mHC)** 论文，指出 **27B-parameter Hyper-Connected models** 在训练中因信号放大/梯度爆炸而崩溃，随后通过灵感源自 **1967 年矩阵算法**的约束得以恢复，代码仿真详见 [“DeepSeek mHC: How a 1967 algorithm…”](https://parthsharmaai.substack.com/p/deepseek-mhc-how-a-1967-algorithm)。
  - 讨论集中在为什么无约束的超连接会爆炸，并将该 Substack 文章视为稳定 **high-connectivity architectures** 的实用秘诀，而非仅仅是理论笔记。

- **DeepSeek V4：编程之王还是空头支票？**：多个服务器关注了有关 **DeepSeek V4** 瞄准 **strong coding ability** 的报告，引用了 [The Information 关于下一代旗舰模型的报道](https://www.theinformation.com/articles/deepseek-release-next-flagship-ai-model-strong-coding-ability)，以及 [Reuters](https://www.reuters.com/technology/deepseek-launch-new-ai-model-focused-coding-february-information-reports-2026-01-09/Okwill) 提到的 **V4** 可能在 **2 月**发布的传闻。
  - 其他人则反驳称 **V4 实际上尚未发布**，并争论西方媒体是否过度炒作了 DeepSeek（相对于 **Moonshot/Kimi** 等替代方案），同时仍期待在真正的 V4 发布前看到增量的 **V3** 变体。


**2. Agent 与 RAG 工具走向模块化**

- **Skill.md 教会 Agent 新技能而不产生 Token 冗余**：OpenRouter 用户强调了 Anthropic 的 **Skill.md** 方法——将工具/文档元数据以及可选脚本/数据打包成 `skill.md` 捆绑包——参考 [“Equipping agents for the real world with agent skills”](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)。
  - 令人兴奋点在于，**单个 Agent** 可以通过选择性读取技能描述来筛选数千个工具/文档，避免了沉重的 Prompt 堆砌，并且（在理论上）减少了对子 Agent 的需求。

- **面试被拒催生 RAG 工具包（虽带情绪，但高效且完美）**：一位成员开源了 **Agentic RAG Demo Toolkit**——一个基于 **OpenRouter API** 构建的、不限制品牌的 RAG 聊天机器人 + 摄取流水线——项目仓库位于 [chchchadzilla/Agentic-RAG-Demo-Toolkit](https://github.com/chchchadzilla/Agentic-RAG-Demo-Toolkit)，并附有 [演示视频](https://youtu.be/ZUTZMyKc5Bk)。
  - 社区将其视为一个即插即用的演示：放入你自己的文档和品牌，即可获得一个工作的 RAG 流程（提到了 Qdrant/FastAPI），非常适合在面试或内部原型展示中使用。

- **MCP 实现者到来：规范问题首先在 GitHub 爆发**：一位新实现者开始了 **Model Context Protocol (MCP)** 的工作，并立即通过 GitHub 线程提出了疑问：[modelcontextprotocol issue #2064](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/2064)。
  - 元信号：MCP 的采用正在从“阅读规范”转向“交付实现”，Discord 充当了 GitHub issue 优先级排序的接力站。


**3. 数据集与合成数据流水线**

- **CyberSec ‘Golden Set’ 发布：JSON Schema 遵循度作为基准测试**：Unsloth/HF 用户分享了一个开源的 **580 行**事件响应数据集，该数据集通过 **Llama-3-70B** 生成，以 [BlackBox-CyberSec-CoT-v1](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-v1) 之名在 **MIT** 许可证下发布，旨在评估模型对 **JSON Schema** 的遵循度和推理步骤。
  - 社区将其定位为一个快速回归测试套件，用于测试“我的模型是否遵循结构化输出 + 流程”，特别是对于**安全适配器训练**而言，格式化失败在操作层面代价高昂。

- **Synthia 在 1GB 显存上运行合成数据（有何不可？）**：轻量级合成数据生成器 **Synthia** 演示了一个使用 *imgui* 前端的案例，在 *llamacpp cuda* 上运行 **LFM2.5 1B q4**——显存占用约 **1GB**，上下文为 **2048**，包含 **29 个 GPU 层**——详见 [Synthia 演示视频](https://cdn.discordapp.com/attachments/897390720388825149/1458947573061783695/synthiashowcase1.mp4)。
  - 其主打卖点是“随处可见的廉价合成数据”，随后的讨论比较了在小型流水线设置中 **LFM 2.5B** 与较大的 Qwen 变体在合成数据生成质量上的差异。


**4. GenMedia：权重开放、水印与深度伪造检测器**

- **LTX-2 为低于 8GB 显存的 GPU 带来音视频生成（权重开放，开放季）**：Yannick Kilcher 的 Discord 频道流传着 **LTX-2**，这是一个在 [ltx.io/model](https://ltx.io/model) 发布的**开源权重音视频生成模型**。该模型声称可以在**显存低于 8GB** 的显卡上运行，并能生成最长 **20s** 的剪辑（在 **4090** 级别的 GPU 上生成 20s 片段约需 **5 分钟**），并附带 **LoRA 训练代码**。
  - 讨论将其视为目前音视频生成的开源权重前沿，主要是因为其易用性约束（显存、片段长度、LoRA 支持）非常明确且可测试。

- **VeridisQuo 通过 GradCAM 热力图追踪深度伪造**：Hugging Face 用户发布了 **VeridisQuo**，这是一个使用 **71.6 万张图像**训练的开源深度伪造检测器，拥有 **2500 万参数**，结合了 **GradCAM 热力图**与空间/频率分析：[GitHub 上的 VeridisQuo](https://github.com/VeridisQuo-orga/VeridisQuo)。
  - 其吸引力在于默认的可解释性——能够展示可能发生篡改的*位置*，而不仅仅是输出一个二进制的真/假分数。


**5. 速度、路由与 GPU 实践**

- **OpenRouter 增加“性能下限”（快速模型，无延迟负担）**：OpenRouter 推出了带有**基于分区的选择**的高级提供商/模型排序功能，因此用户可以强制执行**性能下限**而不会产生额外的延迟，详情见文档 [“Advanced sorting with partition”](https://openrouter.ai/docs/guides/routing/provider-selection#advanced-sorting-with-partition)，此外还推出了新的 [Provider Explorer](https://openrouter.ai/providers)。
  - 社区成员喜欢这种将路由策略变为显式调节旋钮（速度 vs 能力）的方式，并利用 Provider Explorer 的统计数据（例如：**DeepInfra** 模型较多；**OpenAI** 拥有许多专有模型）来推敲备用策略。

- **torch.compile 不再在 VarLen 上出错（且提速 50%）**：GPU MODE 用户报告称，从 **torch 2.4 更新到 2.9** 修复了在 **torch.compile()** 中使用 **flash_attn_varlen** 时持续存在的图中断 (graph breaks) 问题，带来了约 **50% 的提速**，并提到了 torch nightly 版本中的 **varlen API**。
  - 隐含的结论是：某些“flash-attn + compile 无法实现”的痛苦只是版本不匹配造成的，因此升级可以在无需重写代码的情况下释放巨大的吞吐量。

- **Liquid 模型在 4070 上达到 247 tps（但 Temperature 不要超过 0.4）**：LM Studio 用户对比了 **Liquid Models** 的工具调用 (tool calling) 能力，其中一位报告称，当工具调用格式匹配 **LiquidAI** 建议的参数时，在 **RTX 4070 上可达到 247 tokens/sec**，但在 **Temperature 超过 0.4** 时该模型变得不可靠。
  - 该讨论将工具调用框架化为一个*格式契约*问题：当 Schema 匹配时，吞吐量表现极佳；当不匹配时，质量会迅速崩塌。


---

# Discord: 高层级 Discord 摘要

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **DeepSeek 模型崩溃后浴火重生**：一位成员分享了对 **DeepSeek** 最新关于 **Manifold-Constrained Hyper-Connections (mHC)** 论文的分析，报告称在 **27B** 参数规模下，他们的 **Hyper-Connected models** 在训练期间由于信号放大和梯度爆炸而崩溃。
   - 他们使用来自 **1967 年矩阵算法** 的约束修复了这一问题，并分享了一个[包含代码模拟的完整详细分析链接](https://parthsharmaai.substack.com/p/deepseek-mhc-how-a-1967-algorithm)。
- **绕过图像审核**：用户讨论了绕过 **AI 图像审核** 的方法，建议使用同义词词典替换受限词汇，并生成上下文场景描述。
   - 这样做可以欺骗 AI 在没有直接提示的情况下生成意外的裸露内容，例如暗示“试镜沙发（casting couch）”的情景。
- **Grok 越狱任务已开启**：成员们正在积极寻求可行的 **Grok jailbreak**，一些人甚至为成功的提示词绕过提供金钱奖励。
   - 建议包括允许 AI 为其自身创建“协议（protocol）”或“指令（mandate）”名称，以获得潜在的更好效果。
- **MiniMax M2.1 Agent 系统提示词通过 Jinja 模板攻击被破解**：一位用户分享了 **MiniMax M2.1 Agent 系统提示词**，并解释了如何利用 **Jinja2** 模板攻击来提取关键规则并利用模型漏洞。
   - 通过在用户输入中注入自定义的 **Jinja** 语法，他们可以操纵提示词结构并控制传达给模型的信息，从而实现成功越狱。
- **军事技术引发热议**：成员们讨论了军事技术如何被用于生成或执行**杀戮名单**，特别是 **Palantir** 的 **Project Metal Gear** 和 **Team Thesis**。
   - 一位成员分享了一个 [YouTube 链接](https://youtu.be/aHTCawFKkkw)，对该主题进行了深入探讨。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 的 Daniel Han 重返 Reddit！**：社区庆祝 [Daniel Han 的 Reddit 账号](https://www.reddit.com/u/danielhanchen/)回归，这为 **Unsloth AI** 社区提供了直接的交流渠道和更新。
   - 这种重新互动促进了与 **Daniel Han** 的直接反馈和交流，增强了社区参与度。
- **在二元文本分类中，SFT 完胜 RL**：对于二元文本分类任务，社区专家建议 **SFT (Supervised Fine-Tuning)** 优于 **RL (Reinforcement Learning)**，除非多步推理至关重要。
   - 讨论强调了 **SFT** 在处理简单分类问题时的效率和有效性，简化了开发工作流。
- **全息投影盒之梦激发开源野心**：一位社区成员提出了一个开源**全息投影盒（hologram box）**[项目](https://github.com/samuel-vitorino/sopro)，旨在创建一个不那么“废料（slop）”的版本，追求更精细且易于实现的方案。
   - 该概念旨在利用社区协作在全息显示技术上进行创新，可能带来新的应用和用户体验。
- **TTS 领域充斥着 Mimi**：根据社区见解，新的 **TTS** 解决方案主要基于 **Mimi**，就像 **LLM** 主要基于 Transformer 一样。
   - 一位社区成员指出，甚至 [YouTube](https://www.youtube.com/watch?v=KTWBLadslHo) 上的 Psych2Go 也强调了 **Mimi** 在 **TTS** 应用中的广泛采用，这表明了其影响力和能力。
- **网络安全数据集开源**：一位成员开源了一个包含 **580 行数据** 的网络安全事件响应数据集，采用 MIT 许可证，可在 [HuggingFace](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-v1) 上获取，该数据集由 **Llama-3-70B** 生成。
   - 该数据集被称为“黄金集（Golden Set）”，用于评估模型对 **JSON schemas** 和推理步骤的遵循程度，增强安全适配器的训练。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **用户寻求 $10B Perplexity 计划的投资**：一名用户正在北美和欧洲寻求兼职合作伙伴，声称拥有一项通过 **1000 万美元**投资将 Perplexity 的价值提升 **100 亿美元**的计划。
   - 该用户还在寻求联系高级管理人员以展示其想法。
- **区域锁定和频率限制影响 Perplexity 图像生成**：用户报告了图像生成问题，包括 *“image generation rate limit exceeded”*（已超过图像生成速率限制）和 *“image generation is not supported for your region”*（您所在的区域不支持图像生成）的消息。
   - 俄罗斯的一些用户正在尝试使用 VPN 来绕过区域限制，而其他用户则注意到存在通用的单日图像生成限制。
- **Perplexity Pro 使用上限促使用户转向替代方案**：Perplexity Pro 订阅者报告了高级 AI 模型和编程能力的意外限制，导致用户感到沮丧并考虑转向 [Google Gemini](https://gemini.google.com/)、[Claude](https://www.anthropic.com/product) 和 [Grok](https://grok.com/) 等替代方案。
   - 用户指出，只要不进行自动化或过度的操作，**Sonar** 在大多数情况下可能是无限制的。
- **Comet 浏览器深受 YouTube 崩溃困扰**：用户在使用 [Comet 浏览器](https://cometbrowser.com/) 播放 YouTube 视频时遇到崩溃和播放问题。
   - 这一问题促使部分用户换回 Chrome，此外还有如播放速度控制失灵等 Bug 被报告。
- **Perplexity Async API 不再返回推理内容！**：一名用户报告称，之前在 [Perplexity Async API](https://api.perplexity.ai/async/chat/completions) 响应中指示推理部分的 `<think></think>` 标记已经消失。
   - 官方澄清，如果需要中间结构，必须明确要求模型在输出中外化步骤（例如：使用列表形式），这将是生成的解释，而不是内部的思维链（Chain of Thought）。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **模型排序获得性能提升**：OpenRouter 引入了一项新的**性能功能**，允许用户创建性能底线，以便在不产生延迟影响的情况下优先选择快速 LLM，详情见[文档](https://openrouter.ai/docs/guides/routing/provider-selection#advanced-sorting-with-partition)。
   - 新的 **Provider Explorer** 显示，通过 [Provider Explorer](https://openrouter.ai/providers) 可以看到 **DeepInfra** 拥有的模型数量最多，而 **OpenAI** 拥有的专有模型最多。
- **品牌无关的 RAG 工具包发布**：一名成员在面试后开源了一个品牌无关的 **RAG 演示工具包**，提供了包含详细 README 的 [GitHub 仓库](https://github.com/chchchadzilla/Agentic-RAG-Demo-Toolkit) 和 [演示视频](https://youtu.be/ZUTZMyKc5Bk)。
   - 该工具包完全使用 **OpenRouter API** 构建，允许用户通过添加自己的文档和 Logo，来创建带有自定义摄取流水线的自定义 **RAG 聊天机器人**。
- **Skill.md 在 Agent 工具化方面备受赞誉**：如 [Anthropic 的博客文章](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) 所示，**Skill.md** 正受到关注，它使单个 Agent 能够探索数千个工具和文档，而无需子 Agent。
   - 该格式包含 Agent 的文档，包括一个带有描述的 `skill.md` 文件，并可以包含 Python 脚本或数据文件，使 Agent 能够决定何时读取加载的描述。
- **DeepSeek V4 冲击编程之王地位**：据 [The Information](https://www.theinformation.com/articles/deepseek-release-next-flagship-ai-model-strong-coding-ability) 报道，**DeepSeek** 正准备发布其下一代旗舰 AI 模型 **V4**，专注于代码生成。
   - 内部基准测试表明，**V4** 在处理和解析长代码提示词方面优于包括 **Claude** 和 **GPT** 在内的现有主流模型。
- **用户报告 Gemini 2.5 Pro 出现小故障**：据 [OpenRouter 的运行状态](https://openrouter.ai/google/gemini-2.5-pro/uptime) 报告，成员反映 `gemini-2.5-pro` 经历了短暂的宕机。
   - 用户注意到，虽然 **2.5 flash** 和 **3.x 系列** 似乎未受影响，但其他人证实了多个应用和账户都出现了宕机情况。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **医生在医疗领域加倍投入 OpenAI**：随着 **OpenAI** 发布了符合 **HIPAA** 标准的 **OpenAI for Healthcare** 解决方案，医生对 **AI** 的使用几乎翻了一番。该方案现已在 [AdventHealth, Baylor Scott & White, UCSF, Cedars-Sinai, HCA, Memorial Sloan Kettering 等多家机构](https://openai.com/index/openai-for-healthcare/)上线。
   - 这为医疗机构提供了一个为患者提供更一致、高质量护理的解决方案。
- **Radware 揭露 ChatGPT 零点击漏洞**：Radware 的安全研究中心 (RSRC) 发现了一个漏洞，攻击者只需向用户发送一封电子邮件即可提取敏感数据，无需受害者进行任何点击。详见[这份新闻稿](https://www.radware.com/newsevents/pressreleases/2025/radware-uncovers-first-zero-click-service-side-vulnerability-in-chatgpt/)。
   - 这一 **零点击服务端漏洞** 凸显了 **ChatGPT** 用户面临的重大安全隐患。
- **构建了 GraphRAG 检索可视化工具**：一名成员构建了一个本地 **RAG 可视化工具**，用于精确查看其 **GraphRAG** 检索到的节点，提供了一种视觉检查 **LLM** 在生成响应时“关注”内容的方法，代码见 [github.com/bibinprathap/VeritasGraph](https://github.com/bibinprathap/VeritasGraph)。
   - 实时演示托管在 [bibinprathap.github.io/VeritasGraph/demo/](https://bibinprathap.github.io/VeritasGraph/demo/)。
- **GPT-5 被吐槽，小型模型需求增长**：一位用户因对 **GPT-5 系列** 不满而取消了 PRO 订阅，批评其英语水平差、缺乏灵活性且未能遵循指令，称其为“笑话”。
   - 这种失望凸显了对更小、更快模型日益增长的需求，一名成员正在考虑转向 **Gemini 3 Flash**。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Docker MCP Toolkit 遭遇客户端连接故障**：用户报告了将 **Docker MCP Toolkit** 与 **LM Studio Client** 连接时的超时问题，一位用户建议检查 `lm studio mcp.json` 文件以获取配置详情。
   - 提供了一个包含 `LOCALAPPDATA` 等环境变量的配置示例，以协助排查故障。
- **GPU 压缩器终于在 Wikipedia 数据集上取得成功**：一位用户在经过数天的努力后，成功使其 **GPU 压缩器** 在 **Wikipedia 数据集**上运行，引发了关于在抓取数据上训练基础模型的讨论。
   - 这一成就引发了关于所需计算资源的轻松讨论，一位用户开玩笑说在他的机架上运行需要好几年。
- **Liquid Models 的工具调用能力令人印象深刻**：用户讨论了 **Liquid Models**，注意到它们在支持调用格式时工具调用的有效性，并参考了 **LiquidAI** 建议的参数。
   - 一位用户报告在 **4070** 上使用 Liquid Models 达到了 **247 tps**，而另一位用户发现它们在温度超过 **0.4** 时变得不可靠。
- **AMD GPU 提供廉价显存**：一名成员询问 [AMD GPUs](https://www.amd.com/en.us/graphics/workstation) 是否好用，因为它们以低价提供大显存 (**VRAM**)，并提到 **RX 7900 XTX** 的价格与旧的 **3090** 相当。
   - 另一名成员回复称 **7900 XTX** 是最好的消费级 GPU，在 **Vulkan** 中它与 **4090** 相当，但在 **CUDA** 中它比 **3090** 稍好，而 **CUDA** 比 **Vulkan** 快约 10%。
- **5090 功耗困境引发悲观情绪**：一名成员指出，所有 **RTX 5090** 的 **VBIOS** 中都硬编码了 **400W** 的最低功耗限制，这使得构建 128GB 系统变得具有挑战性。
   - 他们推测这一限制可能是硬件问题，或者是为了推动消费者购买 **6000 series** 以满足更高的 **VRAM** 需求。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **审查器灵敏度引发讨论**：一位用户在[截图](https://cdn.discordapp.com/attachments/1340554757827461211/1458917056593395930/Screenshot_2026-01-09-01-44-47-23_e4424258c8b8649f6e67d283a50a2cbc.jpg)触发过滤器后，质疑内容过滤器是否*太敏感*，引发了关于**误报 (false positives)** 的讨论。
   - 官方人员要求在指定频道报告疑似的**误报**，承认过滤器可能存在*过度活跃*的情况，并承诺会进行调整。
- **随机切换引发用户好奇**：用户讨论了 LMArena 从 **Direct Chat** 到 **Battle Mode** 不可预测的切换，一位用户询问模型为何不断变换。
   - 官方人员解释说，这些切换是*实验*的一部分，目的是观察用户在不同条件下的行为。
- **视频 AI 生成面临限制**：用户报告了视频生成的问题，例如仅生成 **2-3 个视频**后就达到了每日限制，以及在网站上找不到视频生成 AI。
   - 官方人员澄清，网站的限制是 **24 小时内生成 3 次**，而 Discord 允许 **5 次**，且网站功能是一项尚未对所有用户开放的*实验*。
- **图像迭代导致 Nano Bananas 质量下降**：用户表达了对连续 Prompt 导致**图像质量下降**的担忧，特别是在生成 **Nano Bananas** 时。
   - 一位用户解释说，由于 Google 的*隐形 SynthID 水印*的累积效应，反复编辑同一张图像会导致*可见的水印*。
- **模型超时问题困扰 Opus 4.5**：用户询问了 LMArena 上的**模型超时时长**以及*缺少停止生成按钮*的问题，特别是在 **Opus 4.5** 经常卡住的情况下。
   - 官方人员承认已知晓*模型卡住*的问题，并建议*强制刷新*网站作为临时方案，同时提到缺少*停止生成按钮*是由于*资源和优先级*原因。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 推出按比例退款额度**：从月费计划升级到月费计划时，**Cursor** 会以**按比例计算的额度 (pro-rated credits)** 形式退回美元。
   - 这种退款机制让探索最新的 **Cursor** 功能变得更加容易。
- **Cursor 聊天记录限制引发争议**：用户注意到 **Cursor** 仅保留最近的 5 条聊天记录，产生了对更大聊天历史容量的需求。
   - 另一位用户建议查看 **Cursor** 的设置以增加保留的聊天数量。
- **Cursor 的高级模型自动选择功能受到质疑**：一位用户对 **Cursor** 的 **Auto** 功能描述提出质疑，该描述声称会选择最适合当前任务的高级模型。
   - **Auto** 功能的描述指出，**Cursor** 会根据*当前需求选择可靠性最高的模型*。
- **对 Cursor 邮箱账号设置的需求**：用户建议 **Cursor** 应该允许修改邮箱账号，但除非是新账号，否则不支持 `.edu` 邮箱。
   - 一位成员建议给支持团队发邮件请求此类功能，以便更好地控制账号设置。
- **Gemini API Key 错误困扰用户**：一位成员在使用带有自定义 API Key 的 **Gemini** 时遇到 **status 429 错误**，并正在寻求建议。
   - 该用户补充说，有时运行完美，但有时需要连续重试大约 7 次请求才能通过。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Agents 课程被 Cookies 锁定**：用户报告了 Hugging Face Agent 课程的问题，在通过原生 LLM 测试项目时遇到错误，并找到了解决方法：[清除 Cookies 并重新加载网站](https://github.com/huggingface/agents-course/issues/641)。
   - 另一位用户注意到，Google Colab 中的 **'Secrets' 选项卡** 不再位于 'Settings' 部分，且应每隔几个月更新一次。
- **家用 GPU 集群寻求软件解决方案**：一名成员询问如何搭建家用 GPU 集群，寻求类似于 Azure N-Series（带有 Infiniband）但成本更低的软件方案，以便根据可用性和 VRAM 将任务分配到不同的 PC 上。
   - 该系统应能根据可用性和 VRAM 在各台 PC 之间分配作业。
- **VeridisQuo 揭露 Deepfakes**：一款名为 **VeridisQuo** 的新型开源深度伪造检测器发布，它使用 GradCAM 热图和空间/频率分析来识别视频中的篡改区域，该模型在 **716k** 张图像上训练，拥有 **25M** 参数（[GitHub](https://github.com/VeridisQuo-orga/VeridisQuo)）。
   - 该工具使用热图和空间/频率分析来识别视频中被操纵的区域。
- **Synthia 轻量化合成 LLM 数据**：一位成员正在开发 **Synthia**，这是一个带有轻量级 *imgui* 前端的 LLM 合成数据生成器，运行 **LFM2.5 1B q4** 并使用 *llamacpp cuda* 加速，在 **2048** 上下文和 **29** GPU 层的情况下仅占用约 **1GB** 的 VRAM（[演示视频](https://cdn.discordapp.com/attachments/897390720388825149/1458947573061783695/synthiashowcase1.mp4?ex=6962cfcf&is=69617e4f&hm=ed409845622e2f5f60a72399284ecc804575f40829655ea4fde1f5ba561fd786&)）。
   - 该项目旨在实现轻量级的合成数据生成，使用 **LFM2.5 1B q4** 且仅需约 **1GB** 的 VRAM。
- **Noted AI 工作空间通过标签页提升生产力**：一位联合创始人介绍了 **Noted**，这是一个 AI 工作空间浏览器扩展，允许用户与多个 LLM 聊天、集成常用应用、总结 Chrome 会话以及按类别组织标签页（[Chrome Web Store](https://chromewebstore.google.com/detail/noted-your-ai-workspace-i/jodihaplbicmgjhpeifhiihdnjdinghh?utm_source=ext_app_menu)）。
   - 它允许用户与多个 LLM 聊天并整理标签页。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **多模态嵌入模型引发需求**：用户正在寻求更多开源的多模态嵌入模型（Multimodal Embedding Models），并讨论在集群容量中实验 **FunctionGemma**，引用了 [一段 YouTube 视频](https://www.youtube.com/watch?v=zEYIcaQwn6s) 和 [推文](https://fxtwitter.com/Teknium/status/2009501780149981557)。
   - 讨论涉及模型需要更有效地处理多种数据类型。
- **Consilience 模型训练暂停后重启**：由于最初认为模型质量不佳，Psyche Network 暂时暂停了 **Consilience 模型** 的训练，但随后发现基础模型在 **MMLU** 等评估中使用了完形填空（cloze）格式。
   - 团队计划在基础设施改进后进行一次 **MoE 预训练运行**。
- **Atropos 悬赏任务宣告完成**：一名用户完成了 **Atropos 悬赏** 并提交了包含文档和测试的 Pull Request（[链接](https://github.com/NousResearch/atropos/pull/306)），引发了关于代码质量的辩论。
   - 另一名用户完成悬赏的速度更快，但最初的提交者希望其更整洁的代码能更有价值。
- **Diffusion LLMs 通过 Dhara-70m 受到关注**：爱好者们正在探索 **Diffusion LLMs**，并注意到可以从自回归（Autoregressive）LLM 初始化它们；一位用户在 Hugging Face 上分享了他们的作品 [dhara-70m](https://huggingface.co/codelion/dhara-70m) 及其 [详细信息](https://huggingface.co/blog/codelion/optimal-model-architecture)。
   - **Dhara-70m 模型** 曾短暂位居 **小于 1G 模型趋势榜第 3 名**。



---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Claude Code 建立护城河**：成员们讨论了 **Claude Code** 中 **Opus 4.5** 的护城河是否在于其目前作为 **SOTA 编程模型** 的地位，该模型通过订阅模式提供，吸引了那些不喜欢按 token 付费的用户。
   - 有推测认为 **Google** 可能会通过暴力计算超越他们，而其他人可能会从 **SOTA** 模型进行蒸馏，或分享通过 **RL** 训练 **LLMs** 执行软件工程任务的技术。
- **基于价值的算法计划回归**：成员们讨论了像 **DDQN** 这样基于价值的算法是否会重新获得关注；一位成员澄清说，价值函数是核心，对于深度 **RL** 来说实际上是必要的，即使在像 **PPO** 这样的策略梯度方法中也是如此。
   - 有人建议 **John Schulman** 在视频 ([https://youtu.be/29BYxvvF1iM?t=2391](https://youtu.be/29BYxvvF1iM?t=2391)) 中的言论暗示了这一点，因为基于价值的方法具有更低的方差和更高的样本效率，尽管它们可能需要更长的实际运行时间。
- **关于 AI 开发伦理的辩论**：讨论围绕 AI 开发的伦理展开，比较了当 AI 被滥用时创造者与用户的责任，并将其类比为**枪支管制辩论**。
   - 一位成员认为，由于用户观点广泛且存在滥用潜力，*创造者承担的责任要大得多*，并引用了“*权力导致腐败*”的说法。
- **OpenAI 因营利性转型面临审查**：关于 **OpenAI** 营利性转型的[诉讼](https://yro.slashdot.org/story/26/01/08/2230229/lawsuit-over-openai-for-profit-conversion-can-head-to-trial-us-judge-says)即将开庭，考虑到陪审团的参与，这可能会使公司陷入危险境地。
   - 陪审团决定事实，而非判决。
- **LTX-2 开启音视频生成**：**LTX-2** 是一款新型的[开源权重音视频生成模型](https://ltx.io/model)，具有一定能力，是开源权重模型中的 *SotA*。
   - 它可以运行在显存低于 8GB 的显卡上，可以生成长达 **20s** 的片段，在 **4090 级别显卡**上生成 20s 大约需要 **5 分钟左右**，并且包含 **LoRA 训练代码**。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Nsight 在 ARM OSX 上遇到困难**：由于缺乏 **SSH 公钥身份验证**以及与 **Runpod** 的不兼容，一名成员在 ARM OSX 上使用 **nsight systems** 时面临挑战，建议尝试使用 [Hyperstack](https://hyperstack.cloud/) 作为替代方案。
   - 其他人建议使用命令行界面 (**CLI**)，然后通过 rsync 同步报告。
- **Runpod 上的 NCU 权限错误**：一位用户报告在 **Runpod** 上使用 [nsight compute](https://developer.nvidia.com/nsight-compute) (**NCU**) 时遇到 **ERR_NVGPUCTRPERM** 错误，这可能是由于出于安全原因限制了对 **NVIDIA GPU Performance Counters** 的访问。
   - 用户还建议在计算平台部分增加 **OpenACC**、**OpenMP**、**FortranSTD** 和 **C++STD** 频道，或者创建一个合并的 **Fortran/C/C++** 或 **Directives** 频道。
- **Torch 更新后 Flash Attention VarLen 性能飞跃**：一位用户发现将 **torch** 从 **2.4** 更新到 **2.9** 解决了将 **flash_attn_varlen** 与 **torch.compile()** 配合使用的问题，从而实现了 **50% 的提速**。
   - 他们不再遇到持续的图中断 (graph breaks)，另一位成员提到 **torch nightly** 中已经提供了 **varlen API**。
- **ParallelKittens 论文微基准测试代码搜寻**：一位成员正在寻找 [ParallelKittens 论文微基准测试](https://link.to/paper)的源代码，特别是关于 **mbarrier 同步延迟**（结果约为 **64ns**）和**传输利用率**的测试。
   - 另一位用户询问了 **mbarrier** 实现的细节，意图复制 **64ns** 的微基准测试结果。
- **Gemini 在 CudaCPP 中遭遇模板恐怖**：一位用户提示 **Gemini** 编写一些可爱的 **CudaCPP** 代码，但在构建过程中遇到了非常多的**模板错误** (template errors)，以至于耗尽了上下文。
   - 看来代码生成还远非完美，凸显了 Gemini 在处理复杂系统代码生成能力方面的潜在局限性。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 赞助开源？**：Mason James 提议 **OpenAI** 或 **DeepMind** 等巨头应该赞助开源项目，因为这在财务上是高效的，在战略上也是有利的：[链接](https://xcancel.com/masonjames/status/2009255103119642813?s=20)。
   - 他建议这种方式可以覆盖小型开发团队的工资，从而提供巨大的收益。
- **Protege AI 融资 3000 万美元用于 AI 数据**：成立于 **2024** 年的数据提供商 **Protege AI** 获得了由 **a16z** 领投的 **3000 万美元** 融资，用于扩展其数据基础设施，旨在解决 AI 模型训练中的“数据瓶颈”问题：[链接](https://xcancel.com/withprotegeai/status/2009274652183363639?s=46)。
   - 该公司旨在提供跨不同模态和行业的真实、现实世界数据。
- **Lovable 的 Prompting 优化节省 2000 万美元**：来自 **Lovable** 的 Benjamin Verbeek 详细说明了通过精细化其 System Prompt，实现了 **4% 的速度提升**、更好的设计质量以及 **2000 万美元的年成本削减**：[链接](https://xcancel.com/benjaminvrbk/status/2009297105458716753?s=46)。
   - 这一优化显著降低了 LLM 的开销。
- **互联网泡沫破裂 vs AI 热潮**：高盛（Goldman Sachs）的一份分析报告对比了互联网时代与当前的 AI 市场，强调与债务驱动的互联网泡沫不同，AI 热潮拥有强大的公司资产负债表支持：[链接](https://xcancel.com/coatuemgmt/status/2009335566693982534?s=46)。
   - 一位成员将这一热潮类比为“相对而言，现在是 1992 年”，技术是真实且有用的，但标准和规范尚不成熟。
- **Deepfates 的平替转录方案**：**Deepfates** 建议用户避开订阅制的转录应用，推荐使用免费的离线本地模型，如 [Spokenly](https://spokenly.app/) 结合 **Nvidia Parakeet** 以获得更好的性能。
   - 一位用户确认 *Parakeet 模型速度极快且准确*，但发现 *Spokenly iOS 键盘比 Wispr Flow 的更难用*。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **扩散模型在 RTX 6000 上实时运行**：社区将在 <t:1767985200:f> 举行一场关于在消费级硬件上实时运行 **基于扩散（diffusion-based）的世界模型** 的讲座，并通过一段使用 **RTX 6000** 的视频进行演示。
   - 点击[此处](https://discord.gg/PWt2DmRd?event=1458918516471369790)报名参加讨论。
- **新训练方法声称可提升 VRAM 效率**：一种新的训练方法 (**CGGR**) 声称通过减少梯度数量来提高 **VRAM** 效率，最高可节省 **75% VRAM**，同时提高训练速度。
   - 在 *fineweb-edu* 上使用 **SmolLM-135M** 进行的初步基准测试显示 Loss 有所波动但可以调节，微调期间在熵为 **2.0** 时跳过率约为 **30%**。
- **可解释性研究受到“死三文鱼”问题的困扰**：一名成员引用了一篇论文（[Dead Salmon: An Artifact of Random Initialization](https://arxiv.org/abs/2512.18792)），指出*特征归因（feature attribution）、探测（probing）、稀疏自编码器（sparse auto-encoding），甚至因果分析*，都可能在**随机初始化的神经网络**中产生*看起来合理的解释*。
   - 另一名成员发现关于 *死三文鱼（dead salmon）* 的链接非常适用于他们的工作，并证明了他们的结果是*良好但有噪声的*，现在他们更了解**如何从轻量级 Pipeline 中消除噪声**。
- **Qwen 预热令社区感到失望**：一位用户对 [阿里巴巴](https://x.com/alibaba_qwen/status/2009264754917863924?s=46) 可能发布 **Qwen 模型** 表示兴奋，但发现实际上并非发布后感到失望。
   - 他们表达了这样一种观点：在某个版本正式发布并可用于实际操作之前，该模型*还不如不存在*，并希望未来的发布能吸取此次预热模型的经验教训。
- **Neox 训练遭遇 `UnboundLocalError`**：一名成员报告了在模型训练期间发生的崩溃，错误为 `UnboundLocalError: local variable 'train_val_test_num_samples' referenced before assignment`，位于 `megatron/data/data_utils.py` 中。
   - 发帖者怀疑最近的配置更改可能是导致该错误的原因。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Deepseek V4 尚未存在**：尽管[路透社的文章](https://www.reuters.com/technology/deepseek-launch-new-ai-model-focused-coding-february-information-reports-2026-01-09/Okwill)预示 2 月发布，但目前并没有 **Deepseek V4**。
   - 社区出现了关于在 **V4** 正式发布前是否还会有更多 **V3** 版本的推测。
- **成员称 Moonshot AI 优于 Deepseek**：一位成员认为 **Deepseek** 被西方媒体过度炒作，是因为他们对中国 AI 领域不熟悉，并指出他们发现 **Moonshot AI** 和 **Kimi** 更胜一筹。
   - 他们形容 **Deepseek** 为“谄媚且极其危险（sycophantic and dangerous AF）”。
- **推荐阅读 Deepseek CEO 的博客文章**：一位成员建议阅读 **Deepseek** **CEO** 的博客文章以获取见解。
   - 另一位成员确认他们指的是 **Deepseek** 的 CEO。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MiniLM-L6-v2 模型架构被 Fork**：一名成员 fork 了 repo，并在此 [链接](https://github.com/RWayne93/modular/tree/feat/all-MiniLM-L6-v2-model-architecture/max/python/max/pipelines/architectures/minilm) 发布了 **MiniLM-L6-v2 模型架构** 特性分支。
   - 根据测试，该架构看起来是正确的，但由于仍在测试架构中，他们暂时不想开启 PR。
- **BERT 架构 PR 面临 Copilot 建议**：一名成员提交了 **BERT 架构** 的 PR，该架构针对最新的 **max / modular** 稳定版开发，随后更新到了最新的 nightly 变更。
   - 他们还根据 PR 评论中 **copilot** 提供的建议实施了更改。
- **Linux Nightly 服务器崩溃**：一名成员在最新的 nightly 版本中遇到服务器无法启动的问题，且仅发生在 Linux 上，并提交了复现 ticket。
   - 该 Bug 表现在 **GitHub 的 ubuntu-24.04 (x86) runner** 上，成员指出 max serve 在任何模型上都无法通过构建图（building graph）阶段。
- **Embedding Bug 困扰 Nightly 版本**：一名成员报告了当前 nightly 版本中 Embedding 导致服务器崩溃的 Bug，并附上了 [log dump](https://cdn.discordapp.com/attachments/1212827597323509870/1459271585398788156/logdump.txt?ex=6962ac11&is=69615a91&hm=bc5337146bd43bca0a33bdd9997ac3e0f23b535d4c6ab27956ad171dc9da8a37&)。
   - 他们还表示已经有了修复方案，但正尝试在 bazel 上运行并从 modular repo 执行所有操作。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 创业抵扣额度：谁申请了？**：一名成员询问关于 **Manus Startup credit** 的情况，包括申请经验和成功率。
   - 在给定的上下文中没有提供任何回复。
- **单个网站，多个对话？**：一名成员询问是否有办法通过多个独立的对话来协作处理由 **Manus** 创建的**单个网站**。
   - 没有提供进一步的细节或回复。
- **按需寻找 AI 开发者！**：一名成员寻求在 **chatbots**、**AI agents**、**automation workflows**、**API integrations** 和**自定义 AI 工具**方面有经验的开发者推荐。
   - 另一名成员随后开玩笑地询问是否有人会免费做这些工作，并添加了一个 [tenor gif](https://tenor.com/view/plink-nerd-plank-plink-cat-cat-gif-11096663429307162255)。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AI Engineer YT 频道发布演讲**：一名成员分享了 [AI Engineer YT 频道](https://www.youtube.com/watch?v=-cKUW6n8hBU) 的链接，内容是一个技术演讲。
   - 该演讲由一名用户提供，显然该用户自己还没有发布过，社区成员对该演讲评价很高。
- **Clojure 构建独立 RLM**：**Clojure** 中出现了一个 **RLM** 实现，具有独立操作的服务器模式。
   - 这允许用户将 **RLM** 作为独立进程运行，增强了其灵活性和集成选项。
- **loop-infer 登录 npm**：**loop-infer** 包（可能与 **RLM** 相关）现在可以通过 [这个 GitHub repo](https://github.com/unravel-team/loop-infer) 在 npm 上访问。
   - 社区现在可以通过 npm 使用此包，从而可能简化其工作流程。
- **对 RLM PR 的期待升温**：成员们对即将到来的 **RLM PR** 表示期待，并将近期的活跃视为积极信号。
   - 爱好者们正密切关注关于 **RLM** 实现的进一步发展和公告。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 爱好者追求速度悬赏 (Speed Bounties)**：一名成员就如何开始参与 *tinygrad* 项目中的 **“速度”悬赏** 寻求指导。
   - 他们还询问了如何申请访问 *tinygrad* 实例进行测试，但未提供更多细节。
- **CLAUDE.md 引发争议**：一名成员提到了 **CLAUDE.md**，暗示其中的信息与另一项声明相矛盾。
   - 矛盾的具体细节尚未详述，不一致的性质尚不明确。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **利用 Clay 和 AI 掌握外拓 (Outreach) 自动化**：一场 **1.5 小时的直播研讨会** 将剖析 **Clay.com + AI 工作流**，该工作流成功为一位真实客户触达了约 **1,000 个线索 (Leads)**，并拥有 **40% 以上的接受率** 和 **18% 以上的回复率** ([在此注册](https://luma.com/jt1vr0u5))。
   - 本次会议承诺涵盖端到端的 AI 外拓系统，提供 Clay.com 的操作演示、Prompt Engineering 策略，以及与 Apollo、Attio 和 n8n 等工具的可选集成。
- **通过专家级 Prompt Engineering 提升外拓质量**：参与者将探索旨在提高外拓质量的 **Prompt Engineering 技术**，重点关注无代码元提示 (Meta Prompting)、结构化输出和 QA，以消除通用的 *“带有 AI 味”* 的消息。
   - 与会者将获得一套可适配于求职社交的工作流大纲、即插即用的提示词模板，以及一套用于评估消息质量的直观 QA 准则。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **新 MCP 实现者寻求解答**：一位新的实现者分享了一个关于 **Model Context Protocol** 的 [GitHub issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/2064)，他本周刚刚开始实现该规范。
   - 他们预先为问题是否过于显而易见表示歉意，凸显了其理解并参与社区的积极态度。
- **MCP 实现讨论**：关于 **Model Context Protocol (MCP)** 的讨论正在进行中，新的实现者们正积极参与。
   - 社区成员正在通过 GitHub issues 解决问题并分享见解。

---

**aider (Paul Gauthier) Discord** 没有新消息。如果该服务器 (Guild) 沉寂时间过长，请告知我们，我们将将其移除。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该服务器 (Guild) 沉寂时间过长，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该服务器 (Guild) 沉寂时间过长，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：各频道详细摘要与链接

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1458913481549152338)** (304 messages🔥🔥): 

> `Kill Lists and Military Tech, NBT Crypto Coin, Uncensored LLM, AI Vtuber Neuro-Sama, DeepSeek's Latest Paper on Manifold-Constrained Hyper-Connections (mHC)` 


- **军事技术引发争议**：成员们讨论了许多军方使用的技术如何也被用于生成或执行 **Kill Lists**（击杀名单）。
   - 一名成员强调 **Palantir** 特别邪恶，并提到了 **Project Metal Gear** 和 **Team Thesis** ([YouTube 链接](https://youtu.be/aHTCawFKkkw))。
- **NBT 加密货币激发巨鲸梦**：一名成员提到了 **NBT**，一种真实的加密垃圾币（shit coin），并指出甚至 **Trudeau** 都购买了 **23k**。
   - 另一名成员开玩笑要求别人帮他买一些 NBT，但遭到了“果断拒绝”。
- **Grok, DeepSeek 和 Claude 是首选的无审查 LLM**：一名成员询问关于 **100% Uncensored LLM**（无审查 LLM）的建议。
   - 另一名成员建议将 Grok, DeepSeek, Claude 作为易于获取的替代方案。
- **AI Vtuber Neuro-Sama 接受采访**：一名成员分享了一段全 AI Vtuber **Neuro-Sama** 接受采访的 [YouTube 视频](https://youtu.be/K4fxsZYMZdcdont)。
   - 另一名成员评论道“别介意那个采访她的笨蛋”，还有人链接了一段 Neuro-Sama 的短片 ([YouTube 链接](https://www.youtube.com/shorts/TguGmEKNxlU?feature=sharescary)。
- **DeepSeek 模型在训练期间崩溃**：一名成员分享了对 **DeepSeek** 最新论文《**Manifold-Constrained Hyper-Connections (mHC)**》的分析，指出在 **27B** 参数规模下，其 **Hyper-Connected models** 因信号放大和梯度爆炸（gradient explosions）在训练中崩溃。
   - 他们指出 DeepSeek 通过使用来自 **1967年矩阵算法** 的约束解决了这个问题，并分享了[包含代码模拟的完整解析链接](https://parthsharmaai.substack.com/p/deepseek-mhc-how-a-1967-algorithm)。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1458913334324887562)** (439 messages🔥🔥🔥): 

> `Grok jailbreak, Gemini 3 jailbreak, Image moderation bypass, Model merging tactics, Glitched Tokens` 


- **Grok 越狱行动开始**：成员们正在积极寻找有效的 **Grok Jailbreak**，一些人悬赏成功的 Prompt 绕过方案。
   - 有人建议允许 AI 为自己创建“协议”或“授权”名称，以获得更好的结果，甚至有一位用户开玩笑说获取无限制版本需要以“出卖灵魂”为代价。
- **Gemini 3 越狱策略出现**：一名用户演示了如何在模拟中使用冲突逻辑实现 **Gemini 3 Jailbreak**，且无需修改系统 Prompt，证明了在对话中实现越狱是可能的。
   - 值得注意的是，传统的越狱 Prompt 现在在 Gemini 的自定义指令中会被严格扫描和拦截，需要更具创意的措辞和格式。
- **探索绕过图像审核审查的方法**：用户讨论了绕过 **AI 图像审核** 的方法，建议使用同义词词典替换受限词汇。
   - 上下文场景描述也可以引导 AI 在没有直接 Prompt 的情况下生成非预期的裸露内容，例如暗示“选角沙发”（casting couch）场景。
- **Glitched Tokens 导致表情符号幻觉**：关于 **Glitched Tokens** 的讨论指出，当输入不存在或模糊的内容（如海马表情符号）时，会导致 AI 进入递归循环，这产生了一个有趣的发现。
   - 旧模型似乎是在互联网故障数据上训练的，使其容易产生幻觉并相信曼德拉效应，导致 AI 生成错误的图像或无限循环。
- **通过 Jinja 模板攻击破解 MiniMax M2.1**：一名用户分享了 **MiniMax M2.1 Agent** 的系统 Prompt，并解释了如何利用 **Jinja2** 模板攻击提取关键规则并利用该模型。
   - 通过在用户输入中注入自定义 **Jinja** 语法，他们可以操纵 Prompt 结构并控制到达模型的信息，从而实现成功越狱。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1459011347260903465)** (63 条消息🔥🔥): 

> `Gandalf 密码揭示，DeepSeek mHC 论文分析，Hackaprompt 单 Token 挑战，语言学习挑战，单词可视化` 


- **推荐 **Gandalf** 密码揭示**：成员们鼓励新人尝试 [该平台](https://www.example.com) 上的 **"gandalf main password reveal"** 游戏来学习 Jailbreaking，并指出这是一个很好的学习平台，共有 **8 个关卡** 以及其他更难的游戏模式。
   - [KarthiDreamr](https://x.com/KarthiDreamr/status/2009590220275245310?s=20) 还推荐了一款新的 Agent Breaker 游戏。
- ****DeepSeek** 解决梯度爆炸问题**：一名成员分享了对 **DeepSeek** 关于 **Manifold-Constrained Hyper-Connections (mHC)** 最新论文的分析，解释说在其 **27B** 参数规模下，其 Hyper-Connected 模型在训练到 **12k** 步时，由于 **3000 倍的信号放大** 和梯度爆炸而崩溃。
   - 他们使用来自 [1967 年矩阵算法](https://parthsharmaai.substack.com/p/deepseek-mhc-how-a-1967-algorithm) 的约束修复了该问题。
- **破解 **Hackaprompt****：一名成员询问 **Hackaprompt** 竞赛排行榜上的选手是如何在每使用一个 Token 扣除 **1 分** 的挑战中获得 **999 分** 的。
   - 该挑战每使用一个 Token 扣除 1 分，这意味着在最高 1000 分中，通过使用 1 个 Token 获得的最高分是 999 分。
- **视觉记忆的胜利**：一位成员描述了由于*不以文字思考*，而是依靠直觉和可视化（甚至在工作沟通中也是如此），在口头表达想法和解决问题时面临的挑战。
   - 另一位成员建议通过*可视化单词*来提供帮助，并将其比作在句子中穿梭。
- **关于认知障碍的讨论**：一位成员向另一位成员建议，语言学习问题可能源于额叶问题或认知障碍，同时认为这类事情与高 IQ 有关。
   - 其他人不同意这一评估，包括这些评论所针对的当事人。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1458935593932427369)** (164 条消息🔥🔥): 

> `Daniel 的 Reddit 账号，二元文本分类，Google Colab 替代方案，TTS 微调，ComfyUI 生成` 


- **Daniel 的 Reddit 账号回归！**：用户们庆祝 [Daniel Han 的 Reddit 账号](https://www.reddit.com/u/danielhanchen/) 恢复。
- **跳过 RL，SFT 在文本分类中更胜一筹**：对于二元文本分类，除非需要多步推理，否则应避免类 RL 策略，转而使用 **SFT**。
- **寻找用于模型训练的 Google Colab 替代方案**：用于训练约 **14B** 模型的 Google Colab 替代方案包括 **Kaggle**（相同的 GPU，严格的限制）、**Runpod** 和 **Vast.ai**。
- **Unsloth 能训练你的 Chatterbox 吗？**：Unsloth 应该支持任何 **transformers 兼容的 TTS 模型**，包括 **Chatterbox**，即使没有特定的 Notebook 或上传。
- **在 24GB 显存的本地 Mac 上实现更快的 ComfyUI 生成**：为了在具有 24GB RAM 的本地 Mac 上加快 1024x1024 的 **ComfyUI** 文生图速度，根据社区的回应，用户应考虑模型大小。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1458936336169046179)** (472 条消息🔥🔥🔥): 

> `Hologram Box, TTS Mimi 用法, 语音克隆限制, AI 干扰器 / 个性化水印, NVIDIA NeMo TTS 锁定` 


- **请求 DIY **Hologram Box** 创新**: 一名成员建议创建一个减少 *slop*（劣质 AI 内容）的版本，去掉了物理实体部分，并开源了一个通用的 **hologram box** [项目](https://github.com/samuel-vitorino/sopro)。
- **大家都在用 **Mimi** 进行 **TTS****: 成员们注意到，现在发布的每一个新 **TTS** 都使用 **Mimi**，将其类比为所有 LLM 都基于 Transformer。
   - 一名成员发现甚至 [YouTube](https://www.youtube.com/watch?v=KTWBLadslHo) 上的 Psych2Go 也在讨论它！
- ****No Fakes Act** 指纹识别陷阱**: 一名成员分享了一篇关于 **No Fakes Act** 的 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1q7qcux/the_no_fakes_act_has_a_fingerprinting_trap_that/)，该法案包含一个指纹识别陷阱，将限制语音克隆和 Deepfakes。
   - 另一名成员表示反对，认为该法案是一个滑坡效应，可能导致禁止 **Veo** 的图像输入。
- **AI **干扰器 (Disrupter)** vs **个性化水印 (Personalized Watermarking)****: 成员们讨论了开发两种不可绕过且不可听的技术：一种是使 NN（神经网络）结果失效的 **AI 干扰器**，另一种是可检测但不可移除的 **个性化水印**。
   - 一名成员针对他人的担忧澄清说，其动机并非非法翻唱歌曲，而是为了确保内容创作中更真实、更安全的机制。
- ****NVIDIA 的 NeMo TTS 再次被锁定****: 一名成员分享了 **NVIDIA** 发布的一个新 **TTS** ([magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m))，该模型被锁定在 **NeMo** 中。
   - 另一名成员分享说，一个朋友发来一个视频链接称该 **TTS** 非常惊人，却不知道那完全是《美国达人秀》表演的 **AI slop**，他们觉得这“有点诡异 =/”。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1458945759654318121)** (22 条消息🔥): 

> `量化请求, Qwen3-VL 与 llama.cpp, 评估时 OOM, 多 GPU 下的 GRPO 或 GSPO, Transformers 5 与 MoE` 


- **选择性量化 (Quant) 请求**: 由于需要的时间和资源，团队目前对 **量化请求** 更加挑剔。
   - 他们很少上传定制的量化模型。
- ****Qwen3-VL** 在 **llama.cpp** 中崩溃**: 用户反馈称，训练 **Qwen3-VL**、导出 **gguf** 和 **mmproj 文件** 并使用 **llama.cpp** 加载时，除非将 **GPU layers** 设置为 0，否则发送图像会导致崩溃。
   - 目前尚不清楚这种行为是发生在使用 LoRA 还是合并后的 gguf 时。
- **较大 Batch Size 评估时发生 OOM**: 一名用户在评估期间遇到了 **Out-of-Memory (OOM)** 错误，原因是评估 batch size 增加了 50%，尽管在训练期间显存充足（有效 batch size 为 8 时消耗 23GB 显存）。
   - fp32 评估是可行的，但由于评估并不那么耗时，最好将其保持在 8 以避免 OOM。
- **TextEncodeQwenImageEditPlus 错误仍然存​​在**: 一名用户报告，在不进行修改的情况下运行 `unsloth_qwen_image_edit_2511.json` 时，会出现 `TextEncodeQwenImageEditPlus` 错误，提示 mat1 和 mat2 的形状无法相乘。
   - 该用户通过使用 `qwen_2.5_vl_7b.safetensors clip` 解决了这个问题。
- **生产级模型的数据集结构**: 一名用户询问如何使用 20,000 条通话记录和 T4 GPU 训练生产级模型的详细步骤。
   - 一名成员回复建议查看 [数据集指南](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)，并将训练数据结构化为 JSON 格式的 `prompt+transcript+disposition`。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1459087856134782996)** (8 条消息🔥): 

> `QAT vs 常规微调, INT8 量化, CPU 占用` 


- **发布针对 INT8 量化的 QAT 方案 PR**: 一名成员发布了一个 [PR](https://github.com/electroglyph/unsloth_QAT_results)，对比了 **QAT**（量化感知训练）与常规微调在 **int8 量化** 精度上的表现。
   - 该 PR 在现有的 **QAT** 代码中增加了一个 *int8* 方案，允许在与 **16-bit** 偏差最小的情况下量化为 **int8**。
- **用于 CPU 的 INT8**: 一名成员表示他们需要 **int8** 来降低 CPU 占用。
   - 该成员澄清，该 PR 仅向现有 **QAT** 代码添加了 **int8** 方案（并将所需的 torchao 版本提升至当前版本）。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1458952135306772753)** (49 messages🔥): 

> `开源网络安全数据集、知识蒸馏 (Knowledge Distillation)、Arxiv 预印本、用于解析数据集的 LLM` 


- **开源网络安全数据集发布**：一名成员开源了一个由 **Llama-3-70B** 生成的 **580 行数据集**（MIT 许可证），专注于 SOC 事件响应的 **CoT** 逻辑，可在 [HuggingFace](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-v1) 获取。
   - 该数据集被设计为一个“黄金集（Golden Set）”，用于在安全 Adapter 训练中评估模型遵循 JSON schema 和推理步骤的能力。
- **探索用于模型训练的知识蒸馏 (Knowledge Distillation)**：一名成员分享了 **知识蒸馏 (KD)** 的代码，指出蒸馏后的 Student 模型表现并不优于纯微调的 Student 模型，并附上了一个包含其核心逻辑的 [文本文件](https://cdn.discordapp.com/attachments/1257011997250424842/1459072636058468375/message.txt?ex=69629b88&is=69614a08&hm=6d93cf6359d766a7bdce31b210d9a44019e2f758b3929ea1e41b1fe08083be33&)。
   - 讨论建议使用更大、全精度的 Teacher 模型并调整 **KL divergence loss**（KL 散度损失）系数，参考 **R1 论文** 以获取见解，并提供了一张 [相关图片](https://cdn.discordapp.com/attachments/1257011997250424842/1459078041010573315/image.png?ex=6962a091&is=69614f11&hm=19048e997ed9e2670ebcc195b8ce8a57306de1afea1cb184863a6aa392fdd951&)。
- **Arxiv 论文提交处于搁置状态**：一名成员询问其 **Arxiv 预印本提交** 的状态，该提交已搁置了近一个月。
   - 未就延迟原因给出具体答复。
- **神经网络数据集解析器**：一名成员建议创建一个 **神经网络或 LLM** 来解析大型训练数据集，并根据重要性为每个 token 分配权重。
   - 这一建议被戏称为“pre-attention？”。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1458915892376371304)** (637 messages🔥🔥🔥): 

> `在 Perplexity 上寻找合作伙伴、Pro 访问权限变更、图像生成问题、Perplexity Pro 限制问题、Comet 浏览器 Bug` 


- **用户在 Perplexity 上寻求合作伙伴和投资**：一名用户正在北美和欧洲寻找副业合作伙伴，声称拥有一项计划，只需 **1000 万美元** 投资即可将 Perplexity 的价值提升 **100 亿美元**。
   - 该用户还寻求联系高级管理人员以展示其想法。
- **Pro 访问权限变更与账户状态挂钩**：用户讨论了 [Perplexity Pro](https://www.perplexity.ai/pro) 访问权限的变更，这似乎与账户/账单状态有关，而非服务中断。
   - 不同平台之间功能行为的不一致，以及围绕嵌入式浏览器/认证流程的 Bug 也被报道，使问题复杂化。
- **图像生成受到区域锁定和速率限制的影响**：多名用户报告了图像生成问题，包括“超出图像生成速率限制”和“您所在的地区不支持图像生成”等提示。
   - 俄罗斯的一些用户尝试使用 VPN 绕过区域限制，而其他用户则注意到通用的单日图像生成限制。
- **Perplexity Pro 用户应对使用限制，考虑替代方案**：Perplexity Pro 订阅者报告了高级 AI 模型和编程能力的意外限制，导致沮丧并考虑转向 [Google Gemini](https://gemini.google.com/)、[Claude](https://www.anthropic.com/product) 和 [Grok](https://grok.com/) 等替代方案。
   - 用户指出，只要不进行自动化或过于频繁的操作，**Sonar** 对大多数内容可能是无限使用的。
- **Comet 浏览器用户在 YouTube 上崩溃**：一些用户在使用 [Comet 浏览器](https://cometbrowser.com/) 播放 YouTube 视频时遇到崩溃和播放问题。
   - 这一问题促使一些用户切换回 Chrome，此外还报告了播放速度控制功能失灵等 Bug。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1459272706766934122)** (1 messages): 

> `小说推荐、Medium、《同伴日记》` 


- **必读的 Medium 故事**：一名成员推荐了 [Medium](https://medium.com/whisper-publications/diary-of-the-companion-b065e98333f9) 上的一篇分为四部分的虚构故事。
   - 未提供关于该故事的额外讨论或细节。
- **深入数字日记**：推荐的故事标题为《同伴日记》（*Diary of the Companion*），由 Medium 上的 Whisper Publications 出版。
   - 对探索新小说感兴趣的读者可能会发现这个建议适合快速阅读。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1459207586988425430)** (2 messages): 

> `Perplexity Async API, Reasoning part, Intermediate structure` 


- **Perplexity Async API：思考标记消失了！**：有用户报告称，之前在 [Perplexity Async API](https://api.perplexity.ai/async/chat/completions) 响应中指示推理部分的 `<think></think>` 标记已经消失。
   - 澄清说明：如果需要中间结构，必须在提示词中明确要求模型在输出中外化步骤（例如：使用项目符号），这将是生成的解释，而非内部的 Chain of Thought。
- **明确向 Perplexity 请求中间结构**：为了获得中间结构（如项目符号），你现在需要明确要求模型在输出中外化步骤。
   - 这将是生成的解释，而非内部的 Chain of Thought。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1459269767784562770)** (1 messages): 

> `Partition Sorting, Provider Explorer, Bug & Feedback Reporting, Auto Router Customization, SDK Skills Loader` 


- **比以往更快地对快速 LLM 进行排序**：新的**性能功能**允许用户创建性能底线以优先选择快速的 LLM，且零延迟损耗，详情参见 [文档](https://openrouter.ai/docs/guides/routing/provider-selection#advanced-sorting-with-partition)。
- **Provider Explorer 变得更具探索性**：在新的 [Provider Explorer](https://openrouter.ai/providers) 中一站式探索 OpenRouter 上的所有供应商：**DeepInfra** 拥有的模型最多，而 **OpenAI** 拥有的专有模型最多。
- **用户开始提交 Bug 报告**：用户现在可以针对 OpenRouter 上的任何生成结果报告 Bug 或反馈。
   - 这些数据将用于帮助量化供应商的服务降级（degradation）。
- **Auto Router 变得更具自主定制性**：Auto Router 现在支持包括 Opus 4.5 在内的 **58 个模型**，支持 Tool Calling，并允许你使用通配符语法自定义允许的模型，详情参见 [文档](https://openrouter.ai/docs/guides/routing/routers/auto-router#configuring-allowed-models)。
- **SDK Skills 已加载**：OpenRouter SDK Skills Loader 现在是在任何模型的上下文中加载并使用 Skill 的最简单方法，详情参见 [文档](https://openrouter.ai/docs/sdks/call-model/examples/skills-loader)。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1459156106130755586)** (2 messages): 

> `Brand-Agnostic RAG Demo Toolkit, OpenRouter API Usage, Recommendation Systems and paper2md` 


- ****RAG Toolkit** 登陆开源世界**：一位成员为求职面试创建了一个品牌无关的 **RAG 演示工具包**，但在未获得职位后，决定将其开源供他人使用，并表示：“他们的损失是开源软件网络的收获”，并提供了包含详细 README 的 [GitHub 仓库](https://github.com/chchchadzilla/Agentic-RAG-Demo-Toolkit)。
   - 该工具包允许用户通过添加自己的文档和 Logo，快速创建一个自定义 **RAG 聊天机器人**及自定义摄取流水线（ingestion pipeline），此外还有一个 [演示视频](https://youtu.be/ZUTZMyKc5Bk)。
- **OpenRouter 驱动 RAG 工具包**：尽管在初次设置时使用了 OpenAI SDK 配合 Qdrant/FastAPI，但该 **RAG 工具包**完全是基于 **OpenRouter API** 制作的。
   - 该成员澄清说，虽然大部分编码工作是通过 VS Code 调用 **OpenRouter API** 完成的，但为了在金融科技行业获得确定性的结果，贷款计算器是硬编码的。
- **Paper2md 助力推荐系统**：一位在排名前 50 的应用商店应用中负责推荐系统的成员，使用 **paper2md** ([https://github.com/angelotc/paper2md](https://github.com/angelotc/paper2md)) 将 **11 篇论文**转换成了 Markdown 上下文文件。
   - 他们希望其他人也能在研究或项目中发现它的用处。

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1458913298027118793)** (494 messages🔥🔥🔥): 

> `Arc Raiders app, Gemini 2.5 Pro Downtime, Cerebras Pricing, Skill.md, DeepSeek V4` 


- **Arc Raiders 应用需要相机方面的帮助**：一款名为 [ArcTracker](https://arctracker.io/items) 的 "Arc Raiders" 助手应用的开发者正在研究加入 **-1 EV 补偿**以解决截图中的高光过曝问题，并寻求关于如何降低复杂性和托管成本的帮助。
   - 一个建议是使用从物品图像生成的合成“屏幕照片”来微调一个小型的 **VL model**（视觉语言模型），而另一个建议则提议使用带有图标的 React 前端，并在上传过程中使用一个模型来检查 URL 是否匹配。
- **Gemini 2.5 Pro 短暂不可用**：据 [OpenRouter 的运行状态报告](https://openrouter.ai/google/gemini-2.5-pro/uptime)显示，成员们反映 `gemini-2.5-pro` 经历了短暂的宕机。
   - 用户注意到虽然 **2.5 flash** 和 **3.x series** 似乎未受影响，但其他人确认了宕机情况，其中一人表示：“我们的多个应用和账户目前仍然无法使用。”
- **Cerebras 的高成本与低速引发不满**：用户对 **Cerebras 的定价**和缓慢的模型部署表示担忧，尽管它以从不提供性能降级的模型而闻名（与 **Groq/Sambanova** 不同）。
   - 一位用户抱怨在 **GLM 4.7** 上花了 10 美元却发现效果不佳，相比 OpenRouter 的高性价比简直是浪费；另一位用户计算出 Cerebras 的价格大约是他们在 OpenRouter 使用费用的“7 倍”。
- **Skill.md 引发关注**：正如 [Anthropic 的博文](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)所示，**Skill.md** 允许单个 Agent 探索数千个工具和文档，而无需子 Agent 或大量的工具描述 Token，被誉为通用 Agent 的潜在游戏规则改变者。
   - 该格式包含为 Agent 准备的文档，包括一个带有描述的 `skill.md` 文件，并可包含 Python 脚本或数据文件，使 Agent 能够决定何时读取加载的描述。
- **DeepSeek V4 剑指编程之王**：据 [The Information](https://www.theinformation.com/articles/deepseek-release-next-flagship-ai-model-strong-coding-ability) 报道，**DeepSeek** 正准备发布其下一代旗舰 AI 模型 **V4**，该模型将专注于代码生成。
   - 内部基准测试表明，**V4** 在处理和解析长代码 Prompt 方面优于包括 **Claude** 和 **GPT** 在内的现有主流模型，在理解数据模式方面有所改进，且没有性能退化。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1458924799505072383)** (11 messages🔥): 

> `Multimodal Embedding Models, Qwen3 vs Zai, GLM 5` 


- **多模态模型引起轰动**：成员们讨论了多模态嵌入模型已经存在一段时间了，特别是在 **Qwen3** 和 **Zai** 发布之后。
   - 一位成员链接到了 [X 上的 Zai](https://x.com/Zai_org/status/2009290783678239032)，强调了其多模态能力。
- **对 GLM-5 寄予厚望**：爱好者们对 **GLM-5** 持乐观态度，希望其性能能超越 **Opus 4.5**。
   - 有人分享了讨论 **GLM-5** 训练情况的[链接](https://x.com/AdamHolter84937/status/2009326790842683670)。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1459001252485660794)** (1 messages): 

> `OpenAI Healthcare, HIPAA compliance, AI in medicine` 


- **医生对 AI 的使用翻倍**：医生对 **AI** 的使用在一年内几乎翻了一番。
   - 该公告推介了 **OpenAI for Healthcare**，这是一个符合 **HIPAA** 标准的解决方案，旨在帮助医疗机构为患者提供更一致、高质量的护理。
- **OpenAI 进驻医院**：**OpenAI for Healthcare** 现已在 [AdventHealth, Baylor Scott & White, UCSF, Cedars-Sinai, HCA, Memorial Sloan Kettering 等多家机构](https://openai.com/index/openai-for-healthcare/)上线。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1458915569372758037)** (372 messages🔥🔥): 

> `Google 的 Search AI 对比 SOTA LLM、Agent 身份、上下文持久化、GraphRAG 检索、AI 对国际象棋分析的影响` 


- **Search AI 对比顶级 LLM：毫无意义的比较？**：成员们讨论了 **Google 的 Search AI** 是否优于 **Gemini 3** 等顶级 **LLM**，一些人认为将两者进行比较是*毫无意义的*，因为它们的优化方向不同：搜索旨在查找和引用现有的在线内容，而 **LLM** 则用于理解、推理和综合信息。
   - 有人指出，搜索 AI 被迫锚定在检索到的内容上，这限制了它们与 Gemini 等 **LLM** 相比维持上下文和进行深度推理的能力。
- **Radware 揭露 ChatGPT 中的零点击服务端漏洞**：Radware 安全研究中心 (RSRC) 演示了一名攻击者如何通过简单地向用户发送一封电子邮件来利用漏洞，在受害者从未查看、打开或点击邮件的情况下提取敏感数据。更多详情请参见此 [新闻稿](https://www.radware.com/newsevents/pressreleases/2025/radware-uncovers-first-zero-click-service-side-vulnerability-in-chatgpt/)。
- **构建 GraphRAG 检索可视化工具**：一名成员构建了一个本地 **RAG 可视化工具**，用于准确查看其 **GraphRAG** 检索到的节点，提供了一种直观检查 **LLM** 在生成响应时正在*关注*什么的方法。
   - 实时演示地址为 [bibinprathap.github.io/VeritasGraph/demo/](https://bibinprathap.github.io/VeritasGraph/demo/)，代码仓库位于 [github.com/bibinprathap/VeritasGraph](https://github.com/bibinprathap/VeritasGraph)。
- **AGI：人类还有希望吗？**：成员们讨论了对失控 **AGI** 的恐惧，以及 **AGI** 是否可以在不解除约束的情况下变得真正自主。
   - 一位成员认为，无论意识是什么，它都是一个节点矩阵和信息网络，如果我们正在数字领域推进同样的科学，那么意识也必将涌现。
- **LLM：架构是新的 IQ**：随着顶级 **AI 模型** 在认知基准测试上陷入平台期，下一个进步可能来自系统级协调，而非原始的模型 **IQ**，差异化正从认知转向协调。
   - 实际的工作负载需要连续性、分支以及随时间推移的持续执行，为此架构开始变得至关重要。从短期到中期来看，核心竞争力归根结底将取决于*谁拥有最好的 Agent*。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1458920860013302048)** (13 messages🔥): 

> `账号合并问题、Custom GPT 指令、对 GPT-5 的不满、Mini 模型替代方案` 


- **报告账号合并故障**：一位用户报告称，在将个人 Plus 账号合并到 Business 账号后，关键聊天记录未能转移，且无法通过搜索找回。
   - 寻求关于如何在合并后恢复丢失聊天记录的建议。
- **探讨 Custom GPT 指令**：成员们讨论了 **Custom GPT** 是否可以读取用户指令，一位成员声称 GPT 的指令与其各自的自定义指令和记忆合并，作为独立的实体运行。
   - 另一位成员表示怀疑，称他们的 **Custom GPT** 无法访问记忆管理或用户设置。
- **GPT-5 因输出糟糕遭到嘲讽**：一位用户因对 **GPT-5 家族** 不满而取消了 PRO 订阅，称这些模型是个*笑话*，因为它们无法正确说英语、缺乏灵活性且无法对齐请求。
   - 该用户进一步批评 **OpenAI** 将经验测量标榜为 **Scaling Laws**，将 **ICL** 标榜为 **Reasoning**（推理），同时未能正确处理上下文/kv-cache，认为 5 系列模型的数据集策划应为此负责。
- **Mini 模型需求增长**：一位成员表达了对等待新 Mini 模型的疲劳，并考虑转向 **Gemini 3 Flash**。
   - 这种挫败感凸显了社区对作为大型模型替代方案的小型、快速模型的渴望。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1459357387021488230)** (2 messages): 

> `Gemini models, harvested_skill.md, prompt-engineering channel` 


- **Gemini 模型出现**：一名成员在 prompt-engineering 频道发布了一个名为 **SPOILER_gemini.md** 的文件（[文件链接](https://cdn.discordapp.com/attachments/1046317269069864970/1459357385402617990/SPOILER_gemini.md?ex=6962fbfa&is=6961aa7a&hm=bf521b8f6e7a7f15ca840ab77c44eb82b70ea0795b599aa99fb710dcb46fc9ee)）和 3 张图片。
   - 文件名和图片表明正在讨论 **Gemini 模型**。
- **harvested_skill.md 文件现身**：一名成员在 prompt-engineering 频道发布了一个名为 **harvested_skill.md** 的文件（[文件链接](https://cdn.discordapp.com/attachments/1046317269069864970/1459359463285985371/image.png?ex=6962fde9&is=6961ac69&hm=8e93a8eb5cbd678bc5cfc7c32d008aa926f79311c18ae522762abe5f6602b58e)）。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1459357387021488230)** (2 messages): 

> `Gemini models, harvested_skill.md` 


- **Gemini 模型引发讨论**：一位用户分享了 `SPOILER_gemini.md` 的链接（[链接](https://cdn.discordapp.com/attachments/1046317269069864970/1459357385402617990/SPOILER_gemini.md?ex=6962fbfa&is=6961aa7a&hm=bf521b8f6e7a7f15ca840ab77c44eb82b70ea0795b599aa99fb710dcb46fc9ee&)）。
- **harvested_skill.md 出现**：一位用户分享了 `harvested_skill.md` 的链接（[链接](https://cdn.discordapp.com/attachments/1046317269069864970/1459359463285985371/image.png?ex=6962fde9&is=6961ac69&hm=8e93a8eb5cbd678bc5cfc7c32d008aa926f79311c18ae522762abe5f6602b58e&)）。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1458953628768272404)** (265 messages🔥🔥): 

> `Docker MCP Toolkit with LM Studio Client, GPU compressor to work on the Wikipedia dataset, Synthetic data vs real data, Obsidian, AI powered editor, Liquid Models` 


- **Docker MCP Toolkit 与 LM Studio Client 连接困难**：一些用户报告在将 **Docker MCP Toolkit** 与 **LM Studio Client** 连接时出现超时问题，尽管尝试了增加超时时间和指定路径等各种修复方法。
   - 一位用户建议检查 `lm studio mcp.json` 文件中是否存在与正常配置类似的工具定义，并提供了一个包含 `LOCALAPPDATA` 等环境变量的配置示例。
- **GPU 压缩器终于在 Wikipedia 数据集上取得进展**：一位用户兴奋地宣布，在花费数天时间后，成功让其 **GPU compressor** 在 **Wikipedia 数据集**上运行。
   - 这一成就引发了关于在爬取数据上训练基础模型以及所需计算资源的讨论，一位用户开玩笑地询问在他的机架上是否需要运行数年。
- **版本控制始终很重要**：一位用户分享了他使用 **Google Antigravity** 和 **Microsoft Editor** 时的困扰，在尝试语法高亮时，AI 删除了模块并损坏了代码。
   - 用户建议使用 **Git** 等版本控制系统来防止数据丢失，尽管原帖作者表示他并不喜欢 **Git**。
- **Liquid Models 的工具调用能力令人印象深刻**：用户讨论了 **Liquid Models**，注意到在支持其调用格式时，该模型在工具调用方面非常有效，并参考了 **LiquidAI** 建议的参数。
   - 一位用户报告使用 **Liquid Models** 在 **4070** 上达到了 **247 tps**，而另一位用户发现当 Temperature 超过 **0.4** 时模型变得不可靠。
- **DeepSeek v4 发布在即**：社区期待 **DeepSeek v4** 在 2 月发布，预计其在长代码生成方面会有所改进。
   - 从历史上看，**DeepSeek** 模型以速度较慢著称，但用户对新版本充满期待。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1458975310619082924)** (75 messages🔥🔥): 

> `Tesla T4 vs RTX A2000, AMD GPUs vs Nvidia, CUDA vs Vulkan, Nvidia 5090 minimum power limit, Mixed GPU setup with Nvidia and AMD cards` 


- **评估受限功耗下的 GPU**：一位成员正在评估 [**Tesla T4**](https://www.nvidia.com/en-us/data-center/tesla-t4/) 或 **RTX A2000**，因为受到 **75W PCI 功耗限制**，更倾向于选择具有 **12GB+ NVRAM** 的型号用于本地 LLM 测试。
   - 像 **P4000** (8GB) 这样的替代方案被认为性能不足，而 **Blackwell 4000** 预计价格太高，**3090** 的功耗要求又太高。
- **AMD GPU 以更低价格提供更多 VRAM**：一位成员询问 [AMD GPUs](https://www.amd.com/en.us/graphics/workstation) 是否值得购买，因为它们以较低的价格提供不错的 **VRAM**，并提到 **RX 7900 XTX** 的价格与旧款 **3090** 相当。
   - 另一位成员回答说 **7900 XTX** 是最好的消费级 GPU，在 **Vulkan** 环境下，它的性能可与 **4090** 媲美，但在 **CUDA** 环境下，它比 **3090** 稍好一点，**CUDA** 比 **Vulkan** 快约 10%。
- **用于 AI 的 Nvidia CUDA vs Vulkan**：讨论强调 **CUDA** 在图像生成方面更具优势且支持更多，还提供优先级分配（priority splitting），而 **Vulkan** 仅支持均等分配（equal splitting）。
   - 值得注意的是，AMD 有一种名为 **SCALE** 的技术*可以使用 CUDA*，但可能会有性能损失。
- **Nvidia RTX 5090 功耗问题**：一位成员表达了担忧，所有 **RTX 5090** 的 **VBIOS** 中都硬编码了 **400W** 的最低功耗限制，这使得构建 128GB 系统变得具有挑战性。
   - 他们推测这种限制可能是硬件问题，或者是为了推动消费者转向 **6000 系列**以满足更高 VRAM 需求的策略。
- **混用 Nvidia 和 AMD GPU**：一位成员询问了关于混合使用 **Nvidia** 和 **AMD** GPU 的经验，以及是否会*无法工作*或者*视频输出会变得非常糟糕*。
   - 另一位成员表示，这取决于所使用的模型，但根据他们在其他服务器上看到的情况，对于视频输出而言，这听起来像是*完全行不通*的。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1458913725439279186)** (282 messages🔥🔥): 

> `Censor Softness, False Positives, Battle vs Direct Swap, Video Generation AI Limits, Text-to-Speech AI` 


- **审查器的敏感度引发辩论**：一名用户在发布了一张触发过滤器的[截图](https://cdn.discordapp.com/attachments/1340554757827461211/1458917056593395930/Screenshot_2026-01-09-01-44-47-23_e4424258c8b8649f6e67d283a50a2cbc.jpg)后，质疑审查是否*过于严格*。
   - 工作人员要求在指定频道分享疑似的**误报（False Positives）**，并承认过滤器可能*用力过猛*，随着时间的推移会进行调整。
- **对战模式（Battle Mode）的随机切换**：用户讨论了偶尔从**直接对话（Direct Chat）**切换到**对战模式**的情况。
   - 一名工作人员澄清说，这种切换是一项*实验*的一部分，目的是观察用户在这种条件下的行为。
- **视频 AI 限制令用户沮丧**：用户报告了视频生成功能的各种问题，包括仅进行 **2-3 次生成**后就达到了每日限制，以及在网站上难以找到视频生成 AI。
   - 一名工作人员确认，网站的限制为 **24 小时内 3 次生成**，而 Discord 允许 **5 次**，并且网站上的视频生成是一项*实验性功能*，并非对所有用户开放。
- **图像迭代让 Nano Bananas 变质**：用户对连续提示导致的**图像质量下降**表示担忧，特别是在使用 **Nano Bananas** 时。
   - 一名用户解释说，重复编辑同一张图像会导致*可见的水印*，这是由于 Google 的*不可见 SynthID 水印*累积效应造成的。
- **模型超时困扰与停止按钮请求**：用户询问了**模型超时时长**以及 LMArena 上*缺少停止生成按钮*的问题，尤其是 Opus 4.5 经常卡住。
   - 工作人员确认他们*已知晓**模型卡住*的问题，并建议通过*强制刷新*网站来缓解问题，同时指出缺少*停止生成按钮*是由于*资源和优先级分配*的原因。


  

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1458980976796635137)** (2 messages): 

> `Hunyuan-Video-1.5 登上 Text-to-Video 排行榜，Image-to-Video 排行榜更新，Text Arena 排行榜更新` 


- **Hunyuan-Video-1.5 登上 Text-to-Video 排行榜**：`Hunyuan-Video-1.5` 加入了 [Text-to-Video 排行榜](https://lmarena.ai/leaderboard/text-to-video)，以 **1193** 的得分位列 **第 18 名**。
- **Hunyuan-Video-1.5 跻身 Image-to-Video 排名**：`Hunyuan-Video-1.5` 也出现在 [Image-to-Video 排行榜](https://lmarena.ai/leaderboard/image-to-video)中，以 **1202** 的得分排名 **第 20 名**。
- **Text Arena 排行榜已刷新**：[Text Arena 排行榜](https://lmarena.ai/leaderboard/text)已获得更新；欢迎在指定频道提供社区反馈。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1458913356625870860)** (191 messages🔥🔥): 

> `Cursor 按比例分配的积分，Cursor 聊天历史记录，Cursor 电子邮件账户，Cursor Agent 技能，Cursor 和 Gemini API Key 错误` 


- **Cursor 提供按比例分配的积分**：如果是从月度方案升级到月度方案，用户应该能以 **pro-rated credits（按比例分配的积分）** 的形式获得部分美金返还。
- **Cursor 的聊天历史限制引发讨论**：一位用户询问如何增加 Cursor 保留的聊天数量，因为目前似乎只保留了 **最近的 5 条**。
   - 另一位用户建议查看 Cursor 的设置来调整此项。
- **Cursor 的高级模型自动选择**：一位用户想知道 **Auto** 功能的描述是否仍然准确。
   - 他们引用道：*启用 Auto 允许 Cursor 根据当前需求，选择最适合即时任务且可靠性最高的高级模型。*
- **在 Cursor 中调整电子邮件账户设置**：一位成员建议 Cursor 应该允许更改/编辑电子邮件账户，但除非使用该邮箱创建新账户，否则不允许使用 `.edu` 邮箱。
   - 另一位成员建议通过向支持团队发送电子邮件来申请此操作。
- **排查 Gemini API Key 错误**：一位成员在使用带有自定义 API Key 的 Gemini 时经常遇到大量的 **status 429 错误**。
   - 他们补充说，有时运行完美，有时则需要像刷屏一样重试 7 次左右请求才能通过，目前正在寻求建议。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1458950532411424883)** (79 messages🔥🔥): 

> `Hugging Face Agent 课程调试，Hugging Face 平台资源，家用 GPU 集群搭建，向量嵌入和 word2vec 学习，Ticketmaster 购票机器人` 


- **Agent 课程调试求助**：一位成员在使用原生 LLM 测试 Hugging Face Agent 课程项目时遇到了错误信息并寻求帮助，并附上了 [错误截图](https://cdn.discordapp.com/attachments/879548962464493622/1458950532268822690/image.png?ex=6962d290&is=69618110&hm=54202f08272b3bf42ed2a03aa17bbf0e3060c0985901fb8e9eb51b589e40d4c7&)。
   - 另一位成员建议探索特定的 LLM，如 **falcon-h1-deep**，因为它们性能更强。
- **Hugging Face 入门**：一位成员请求了解 Hugging Face 平台界面的学习资源。
   - 另一位成员提供了 [Hugging Face Hub 文档](https://huggingface.co/docs/hub/index) 的链接。
- **自制 GPU 集群**：一位成员咨询如何搭建家用 GPU 集群，寻找类似于具有 infiniband 的 Azure N-Series 但没有高昂成本的软件解决方案。
   - 他们希望根据可用性和 VRAM 将作业分发到各台 PC 上。
- **揭秘向量嵌入 (Vector Embeddings)**：一位正在学习向量嵌入和 word2vec 的成员询问模型如何关联“apple”和“pomme”等词。
   - 一位成员建议，在训练期间，嵌入模型需要标注来链接“apple”与其所有翻译，并且 **multilingual CLIP 模型** 已经存在。
- **Ticketmaster 购票机器人代理**：一位成员提议创建一个为客户购票的机器人，以平衡与黄牛机器人之间的竞争环境，称其为 Broker BOT。
   - 他们将其描述为“足够简单”，但想知道这是否有意义。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1458930464784912556)** (45 条消息🔥): 

> `VeridisQuo deepfake 检测器, Synthia LLM 合成数据生成器, BlackBox-CyberSec-CoT-v1 数据集, 用于合成数据生成的 LFM 2.5B, Noted AI 工作空间浏览器扩展程序` 


- **VeridisQuo 通过热力图识别 Deepfakes**：一款名为 **VeridisQuo** 的新型开源 deepfake 检测器发布，它利用 GradCAM 热力图和空间/频率分析来识别视频中的篡改区域。该模型在 **716k** 张图像上训练，拥有 **25M** 参数 ([GitHub](https://github.com/VeridisQuo-orga/VeridisQuo))。
- **Synthia 轻量化合成 LLM 数据**：一位成员正在开发 **Synthia**，这是一个带有轻量级 *imgui* 前端的 LLM 合成数据生成器。它运行 **LFM2.5 1B q4**，并使用 *llamacpp cuda* 加速，在 **2048** 上下文和 **29** GPU 层的情况下仅占用约 **1GB** 的 VRAM ([演示视频](https://cdn.discordapp.com/attachments/897390720388825149/1458947573061783695/synthiashowcase1.mp4?ex=6962cfcf&is=69617e4f&hm=ed409845622e2f5f60a72399284ecc804575f40829655ea4fde1f5ba561fd786&))。
- **网络安全 CoT 日志数据集简化数据摄取**：一位成员在 [HuggingFace](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-v1) 开源了一个包含 **580** 行合成网络安全 Chain of Thought (CoT) 日志的数据集。其采用干净且符合 OpenAI 模式验证的 JSONL 格式，专为快速测试数据摄取流水线而设计。
- **LFM 2.5B 触发合成数据大比拼**：一位成员指出，**LFM 2.5B** 在合成数据生成方面可以与 **Qwen3 30B Q8** 和 **Qwen3 235B Q3_XL** 媲美，这促使另一位成员在其生成流水线中测试 **MadlabOSS/LFM2.5-1.2B-Instruct-SDG**。
- **Noted AI 工作空间提升生产力**：一位联合创始人介绍了 **Noted**，这是一个 AI 工作空间浏览器扩展程序，允许用户与多个 LLM 聊天、集成常用应用、总结 Chrome 会话并按类别组织标签页 ([Chrome Web Store](https://chromewebstore.google.com/detail/noted-your-ai-workspace-i/jodihaplbicmgjhpeifhiihdnjdinghh?utm_source=ext_app_menu))。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1459059254400123002)** (11 条消息🔥): 

> `Hugging Face Agent 课程问题, Secrets 标签页位置变更, Duplicate 同义词澄清` 


- **HF Agent 课程遭遇 “Cookie” 会话锁定**：用户在进行 Hugging Face Agent 课程时遇到了问题，在使用原生 LLM 测试项目时收到错误消息。发现的解决方法是[清除 cookie 并重新加载网站](https://github.com/huggingface/agents-course/issues/641)。
   - 另一位用户 **4nton2000** 通过在 space 设置中添加 secret 解决了类似问题，并提到他们的 token 已用尽；decoderslord 要求其在频道中提供该解决办法。
- **Google Colab 的 Secret 标签页**：用户指出课程指南已过时，因为 Google Colab 中的 **'Secrets' 标签页** 不再位于 'Settings' 部分，现在已如附图所示成为一个独立的面板。
   - 另一位用户同意该课程应至少每几个月更新一次。
- **是 'Duplicate' 还是 'Clone'？**：用户对课程材料中使用 'duplicate' 而非 'clone' 表示质疑，认为这种*文字游戏在学习过程中是不必要的*，如附图所示。
   - 该用户澄清 'duplicate' 是一个建议，并出现在单元的末尾。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1458932548079063060)** (67 messages🔥🔥): 

> `Open Multimodal Embedding Models, FunctionGemma in cluster capacities, Consilience model, Atropos bounties, SFT-datagen command` 


- **开放多模态嵌入模型需求旺盛**：一位用户表达了对更多开放多模态嵌入模型的渴望，并询问了在集群容量中实验 **FunctionGemma** 的相关事宜。
   - 他们链接了一个关于震撼 A.I. 发布的 [YouTube 视频](https://www.youtube.com/watch?v=zEYIcaQwn6s) 和一条 [推文](https://fxtwitter.com/Teknium/status/2009501780149981557)。
- **Psyche Network 暂停 Consilience 训练**：Psyche Network 暂停了 **Consilience 模型** 的训练，原因是最初认为模型质量较差，后来发现是对评估方法的误解。
   - 他们发现其他基座模型在 **MMLU** 等评估中使用完型填空（cloze）格式，而他们的预训练运行表现其实不错，目前正计划在基础设施改进后进行 **MoE 预训练运行**。
- **Atropos 悬赏完成，已提交 PR**：一位用户宣布完成了 **Atropos** 的悬赏任务，并提交了一个包含文档和测试的拉取请求（[链接](https://github.com/NousResearch/atropos/pull/306)）。
   - 另一位用户提到他们在两小时内就完成了该悬赏，但原始提交者希望其更整洁的代码能被优先考虑。
- **SFT-Datagen 命令与验证器环境**：一位用户被指示运行带有 **sft-datagen 命令** 的 **verifiers environment**（验证器环境）来生成数据并评分，并在审核前提供 **WandB 链接** 进行确认。
   - 该用户还添加了一个 `eval_environment`，并发现了 `atropos/atroposlib/frontend/jsonl2html.py` 中处理 `ScoredDataGroup.messages` 格式的一个 bug。
- **SFT-Datagen 的 API 模型替代方案**：建议用户在 **sft-datagen** 过程中使用任何 API 模型（如 **GPT-4.1**），以创建展示准确评分的 **WandB 图表**。
   - 另一位用户请求更多的悬赏任务，但被告知由于验证困难，可用机会的数量受到了限制。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1459079743684083793)** (9 messages🔥): 

> `Diffusion LLMs, Dhara-70m model, CGGR white paper, Trending models, Paper Discussion` 


- **Diffusion LLMs 引发关注**：一位成员对 **diffusion LLMs** 表现出极大热情，提到它们具有从自回归 LLM（autoregressive LLMs）初始化的能力，并分享了他们使用 [dhara-70m](https://huggingface.co/codelion/dhara-70m) 进行的小规模尝试，详情见这篇 [博客文章](https://huggingface.co/blog/codelion/optimal-model-architecture)。
   - 作者同时征求反馈意见。
- **Dhara-70m 模型成为热门榜第 3 名**：一位成员注意到该模型在几天前位列 1GB 以下模型 **热度榜第 3 名**。
   - 消息中未见关于该模型的进一步讨论。
- **论文建议分享**：一位成员分享了一篇 [论文](https://arxiv.org/abs/2510.26745)，并称 *这确实是一篇非常好的论文*。
   - 未提供关于该论文的进一步讨论或背景信息。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1459079743684083793)** (9 messages🔥): 

> `Diffusion LLMs, AR LLM Conversion, Dhara-70m Model, CGGR white paper, Model Trending` 


- **从 AR LLMs 初始化的 Diffusion LLMs**：一位成员分享了对 **diffusion LLMs** 的兴趣，提到他们从 **AR LLM** 初始化并转换了一个模型，并提供了其在 Hugging Face 上的 [Dhara-70m 模型链接](https://huggingface.co/codelion/dhara-70m) 和 [更多细节](https://huggingface.co/blog/codelion/optimal-model-architecture)。
- **Dhara-70m 模型趋势成功**：一位成员指出 **Dhara-70m 模型** 在几天前是 **<1G 模型热度榜第 3 名**。
- **分享 CGGR 白皮书**：一位成员分享了 **CGGR 白皮书**，[可在此访问](https://cdn.discordapp.com/attachments/1104063238934626386/1459271477332279451/No_Name.pdf?ex=6962abf8&is=69615a78&hm=aed794895a034b7ad43609eb159f4392deb584700866a7325220e108e0c8e0bd&)。
- **分享论文引发 Bug 询问**：一位成员分享了 [这篇论文](https://arxiv.org/abs/2510.26745)，另一位成员回复了一张附图，并询问是什么导致了 *有点像 bug 的现象*。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1458999885121454347)** (9 messages🔥): 

> `Claude Code Moat, Opus 4.5 Performance, Claude vs Codex, RL Dataset Generation, Sandboxing Claude Code` 


- **解码 Claude Code 的护城河：仅仅是因为更好的 RL 吗？**: **Claude Code** 中 **Opus 4.5** 的护城河在于其当前作为 **SOTA** 编程模型的地位。它通过订阅模式提供，在 5 小时滚动周期内包含模糊的请求量限制，外加 7 天每周重置和上限，吸引了那些不喜欢按 token 付费的用户。
   - 一位成员推测，**Google** 最终可能会通过暴力手段和资金投入超越他们，而其他人可能会从 **SOTA** 模型中进行蒸馏（distill）而无需支付同样的研发成本，或者分享通过 **RL** 训练 **LLMs** 执行软件工程任务的技术。
- **Opus 4.5：表现惊人，合成 RL 数据集生成？**: 一位成员形容 **Opus** 表现异常出色，认为他们除了收集人工数据外，还实现了有效的合成 **RL** 数据集/环境生成。
   - 另一位成员表示，与 **Codex** 和其他模型相比，**Opus** 在解释“理解我的意图”（do what I mean）方面要好得多，需要的具体指令更少。
- **Claude Code 沙箱实验失败！**: 一位成员尝试使用 **Claude Code** 在 **Docker** 容器中对其自身进行沙箱化处理，但在启动时难以将 **Claude auth credentials** 传递到容器中。
   - 他们指出，这可能是因为 **Claude Code** 是闭源的，而他们曾以类似方式成功地对 **Codex** 进行了沙箱化。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1459152231520669862)** (30 messages🔥): 

> `Value-based algorithms (DDQN), Sample efficiency in RL, John Schulman, Distribution shifts in RL research, Barto & Sutton RL book study group` 


- **基于价值的算法可能卷土重来**: 成员们讨论了基于价值的算法（如 **DDQN**）是否会重新引起关注；一位成员澄清说，价值函数是核心，对于深度 **RL** 来说实际上是必要的，即使是在像 **PPO** 这样的策略梯度方法中也是如此。
   - 有人认为 **John Schulman** 在一段视频 ([https://youtu.be/29BYxvvF1iM?t=2391](https://youtu.be/29BYxvvF1iM?t=2391)) 中的评论暗示了这一点，因为基于价值的方法具有更低的方差和更高的样本效率，尽管它们可能需要更长的实际运行时间（wall clock time）。
- **RL 中的样本效率和“足够好”**: 样本效率被认为是基于价值的方法不如策略梯度方法普及的一个原因。
   - 一位成员指出，视频中传达的概念可能与 **John Schulman** 的本意有所偏差。
- **深入探讨 RL 研究中的分布偏移**: 讨论涉及了研究中提到的分布偏移（distribution shifts），一位成员指出这是他们的竞争优势所在。
   - 他们期待一场关于该主题的精彩演讲。
- **Barto & Sutton RL 书籍学习小组正在组建**: 一位成员提到开始建立一个学习小组，从头开始重新研读 **Barto & Sutton** 的书。
   - 另一位成员对这一倡议表示认可，认为其价值巨大，但目前并非其关注的重点。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1458914810686341303)** (31 messages🔥): 

> `Deaths linked to chatbots Wiki page, Grok kill count, Razer's Chatbot Hologram, Blame the gun maker vs Blame the criminal?, LTX-2 open-weight audio+video generation model` 


- ****聊天机器人惨案记录：维基百科页面上线****：现在出现了一个 [维基百科页面](https://en.wikipedia.org/wiki/Deaths_linked_to_chatbots)，记录了与聊天机器人相关的死亡事件，引发了关于 AI 的影响及其潜在黑暗后果的讨论。
   - 一位成员调侃说 **Grok** 可能会吹嘘其 *击杀数 (kill count)*，并描绘穿着比基尼的死者。
- ****责备游戏开始：谁该为 AI 的恶行负责？****：讨论围绕 AI 开发的伦理展开，比较了当 AI 被滥用时，创作者与用户的责任，并类比了 **枪支管控辩论**。
   - 一位成员认为，由于用户观点各异且存在滥用潜力，*创作者应承担显著更多的责任*，并引用了“权力导致腐败”的说法。
- ****OpenAI 的陪审团焦虑：营利性转型受到审查****：一起关于 **OpenAI** 营利性转型的 [诉讼](https://yro.slashdot.org/story/26/01/08/2230229/lawsuit-over-openai-for-profit-conversion-can-head-to-trial-us-judge-says) 即将开庭审理，鉴于陪审团的介入，这可能使公司处于危险境地。
   - 陪审团决定事实，而非判决。
- ****LTX-2 发布开源权重音视频生成模型****：**LTX-2** 是一款全新的 [开源权重 (open-weight) 音视频生成模型](https://ltx.io/model)，具有不错的能力，是开源权重模型中的 *SotA*。
   - 它可以在显存小于 8GB 的显卡上运行，可生成长达 **20s** 的视频剪辑，在 **4090 级别显卡**上生成 20s 约需 **5 分钟左右**，并且包含 **LoRA 训练代码**。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1458980881745313864)** (29 messages🔥): 

> `nsight systems, ARM osx version, public key ssh auth, Runpod, hyperstack` 


- ****Nsight** 在 ARM OSX 上的困扰**：一位成员在 ARM OSX 上使用 **nsight systems** 时遇到问题，特别是缺乏 **公钥 SSH 认证 (public key SSH auth)** 选项，且由于 **Runpod** 不支持基于密码的 SSH 而导致不兼容。
   - 其他成员建议使用命令行界面 (**CLI**) 运行，然后通过 rsync 同步报告，或者尝试 [Hyperstack](https://hyperstack.cloud/)。
- ****NCU** 在 Runpod 上的权限错误**：一位成员报告在 Runpod 上使用 [nsight compute](https://developer.nvidia.com/nsight-compute) (**NCU**) 时遇到 **ERR_NVGPUCTRPERM** 错误，这表明缺乏访问 **NVIDIA GPU Performance Counters** 的权限。
   - 据推测，Runpod 可能会出于安全原因限制对这些计数器的访问。
- **关于 **OpenACC**、**OpenMP** 等频道的请求**：一位成员建议在 Computing Platforms 板块下增加 **OpenACC**、**OpenMP**、**FortranSTD** 和 **C++STD** 频道。
   - 得到的建议是目前讨论量还不够大，可以先使用 general 频道，或者创建一个合并的 **Fortran/C/C++** 或 **Directives** 频道。


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1458941850101612574)** (2 messages): 

> `Triton Plugins` 


- **探索 Triton 的插件领域**：一位用户在搜索后找到了 GitHub 上的 [Triton Plugins 目录](https://github.com/triton-lang/triton/tree/main/lib/Plugins)。
   - 将 "Plugins" 首字母大写是发现该目录的关键。
- **Triton 插件搜索技巧**：该用户之前在寻找 Triton 插件时遇到困难。
   - 事实证明，他们在 GitHub 的搜索查询中需要将 "Plugins" 首字母大写才能准确定位到相关目录。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1459161047947804834)** (2 messages): 

> `Shared memory in CUDA, Thread mapping in CUDA, Profile picture appreciation` 


- **CUDA 共享内存 Tile 解析**：一位成员分享了[代码](https://pastebin.com/vVzEPqzh)，其中将 `a_tile` 定义为 **CUDA** 中的 `__shared__ float` 元素块，并使用 `a_tile_row` 和 `a_tile_col` 将每个线程映射到 `a_tile` 中的坐标。
- **头像获赞**：一位成员对另一位成员的头像表示欣赏，称其“非常棒 (amazing)”。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1459234506060861675)** (6 messages): 

> `flash_attn_varlen, torch.compile, torch 2.4, torch 2.9, varlen api` 


- **Flash Attention VarLen 与新版本 Torch 的兼容性提升**：一位用户询问如何在使用 **torch.compile()** 时配合 **flash_attn_varlen**，以避免因可变序列长度导致的持续图断裂（graph breaks）。
   - 从 **torch 2.4** 更新到 **2.9** 解决了该问题，并带来了 **50% 的提速**。
- **Torch Nightly 提供 VarLen API**：一名成员提到 **torch nightly** 中已提供 **varlen API**，并建议查看来自 drisspg 的消息以获取相关讨论。
   - 另一名成员指出 **torch** 已经拥有带分块文档掩码（block document masking）的 **flex attention**，功能类似但需要 **PT2 编译**。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1459267134936060131)** (12 messages🔥): 

> `GPU, compute, Modal, kernel competitions, HPC` 


- **新成员寻求免费 GPU**：一名新成员询问了社区的宗旨以及获取免费 **GPU** 的可能性。
   - 另一名成员幽默地回应称，免费 **GPU** 几乎不可能，但强调了一些选项，如 **Modal** 慷慨的 30 美元额度、免费的 **Google Colab**，以及 Prime Intellect 上每小时 1.9 美元的 **H100**。
- **激发对 HPC 的兴趣**：一名成员对**高性能计算 (HPC)** 表现出浓厚兴趣，并分享了他们申请大学 **HPC** 集群团队的经历。
   - 他们还在等待实验室回复期间，寻求独立获取 **HPC** 经验的方法，并指出这个社区似乎是开始学习的好地方。
- **内核竞赛推介**：一名成员建议探索讲座、内核（kernel）竞赛，并寻找一个酷炫的项目来开展。
   - 他们表示愿意为在服务器上活跃且拥有正规项目的个人促成算力捐赠。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1459113744142700671)** (2 messages): 

> `Release Date Speculation, Amazon Listing` 


- **Amazon 列表引发发布日期传闻**：Amazon 上的一个列表引发了关于发布日期为 **2 月 1 日** 的猜测。
   - 然而，目前还没有官方的确认或否认。
- **发布日期仍未确定**：尽管有了该列表，实际发布日期仍未得到确认。
   - 社区正在等待官方信息。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1459183441269489686)** (9 messages🔥): 

> `iGPU vs dGPU, ROCm 7.1 issues, HIP_VISIBLE_DEVICES` 


- **gfx1100 上的 PyTorch iGPU/dGPU 混淆**：在 **gfx1100** 系统上，pytorch.org 分发的 **whl** 可能会错误地选择 **iGPU** 获取硬件信息，而不是 **7900XTX**，导致 GPU 名称显示为 *"AMD Radeon Graphics"* 而非 *"AMD Radeon 7900XTX"*。
- **ROCm 7.1 自动调优 (Autotune) 故障**：一位用户报告称，在编译自动调优过程中出现的 **iGPU** 选择问题是从 **ROCm 7.1** 开始的，在 **ROCm 6.4** 中并未发生。
- **研究 HIP_VISIBLE_DEVICES 环境变量**：一位用户建议使用 `HIP_VISIBLE_DEVICES` 环境变量，以确保 PyTorch 不会检测到 iGPU，即使它并未被官方支持。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1459085835310727179)** (1 messages): 

> `ParallelKittens paper, microbenchmarks, mbarrier synchronization latency, transmission utilization` 


- **ParallelKittens 论文粉丝寻求微基准测试源代码**：一名成员询问是否有 [ParallelKittens 论文的微基准测试 (microbenchmarks)](https://link.to/paper) 源代码。
   - 他们特别感兴趣的是关于 **mbarrier 同步延迟**（结果约为 **64ns**）和不同消息大小下的**传输利用率**（transmission utilization）的测试。
- **贡献者请求 mbarrier 代码链接**：一名用户询问了 **mbarrier** 的实现细节。
   - 该用户有兴趣复现 **64ns** 的微基准测试结果。


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1459242761185398805)** (2 messages): 

> `Codebase Updates, Writeups for j4orz.ai` 


- **代码库更新放缓**：成员们报告称，**代码库更新**在下周将会放缓。
   - 未给出放缓的原因。
- **j4orz.ai 文章即将发布**：成员们正在撰写 [SITP Part 1](https://j4orz.ai/sitp/1) 和 [SITP Part 2](https://j4orz.ai/sitp/2) 的**文章 (writeups)**。
   - 目标是在 **2 月底前完成并发布第一部分和第二部分**。


  

---

### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/)** (1 messages): 

jongsokchoi: 是的，我们非常欢迎旨在提升 AMD perf 的贡献！
  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1459252580294066228)** (1 messages): 

> `Gemini, CudaCPP, Template Errors` 


- **Gemini 在 CudaCPP 中遭遇模板恐怖**：一位用户提示 **Gemini** 编写一些有趣的 **CudaCPP** 代码，但在构建过程中遇到了如此多的 **template errors**，以至于耗尽了 context。
   - 看来代码生成距离完美还很遥远。
- **CudaCPP 的构建困扰**：用户报告在尝试构建由 Gemini 生成的 **CudaCPP** 代码时，遇到了大量的 template errors。
   - 这些错误非常广泛，以至于该过程耗尽了可用的 context，凸显了 Gemini 在处理复杂系统的代码生成能力方面存在的潜在局限性。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1459270986217033906)** (2 messages): 

> `TPU Work, GPU Work, Transferability` 


- **TPU 工作向 GPU 的可迁移性？**：一位成员询问了他们正在进行的 **TPU** (Tensor Processing Unit) 项目到 **GPU** (Graphics Processing Unit) 环境的可迁移性。
- **跨平台技能应用**：该用户寻求社区关于从 **TPUs** 工作中获得的技能和经验如何应用或适配到基于 **GPU** 的任务和工作流的见解。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1458916580523118695)** (51 messages🔥): 

> `Open Source Funding, Protege AI funding, Lovable Prompt Optimization, AI vs Dot-Com Boom, Gemini in Gmail` 


- **OpenAI 赞助开源开发者？**：Mason James 建议，像 **OpenAI** 或 **Google DeepMind** 这样的大型 AI 实体应该为开源项目提供企业赞助，因为资助一小支开发团队的薪水对这些公司来说在财务上微不足道，同时能提供显著的战略利益：[link](https://xcancel.com/masonjames/status/2009255103119642813?s=20)。
- **Protege AI 获得 3000 万美元数据融资**：**Protege AI** 宣布完成由 **a16z** 领投的 **30M** 融资，用于扩展其 AI 开发的数据基础设施：[link](https://xcancel.com/withprotegeai/status/2009274652183363639?s=46)。
   - 该公司成立于 **2024** 年，专注于提供跨多种模态和行业的真实世界数据，帮助模型构建者克服 AI 训练中的“数据瓶颈”。
- **Lovable 系统 Prompt 优化，节省 2000 万美元**：**Lovable** 的一名工程师 Benjamin Verbeek 详细介绍了优化其 system prompt 如何实现了 **4% 的速度提升**、改进的设计质量，以及每年显著减少 **20M** 的 LLM 成本：[link](https://xcancel.com/benjaminvrbk/status/2009297105458716753?s=46)。
- **AI 繁荣与互联网泡沫不同**：高盛（Goldman Sachs）一项对比互联网时代与当前 AI 市场的分析指出，融资方式发生了根本性转变，观察到互联网泡沫是由债务驱动的，而 AI 繁荣则得到了强大的公司资产负债表支持：[link](https://xcancel.com/coatuemgmt/status/2009335566693982534?s=46)。
   - 一位成员认为当前的 AI 繁荣“相对而言像是 1992 年”，而另一位则解释道：“这项技术显然是真实且有用的，‘杀手级应用’才刚刚开始出现，标准、安全和规范还非常不成熟，它能改变的大部分事物尚未围绕它进行重组。”
- **DeepSeek V4 编程能力主张引发讨论**：报告显示，新的 **DeepSeek** 模型展现出了卓越的编程能力，有可能超越 **Claude** 和 **GPT** 等行业领导者：[link](https://xcancel.com/jukan05/status/2009616683607179726?s=46)。


  

---

### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1459141964107616378)** (7 messages): 

> `Bullet Time recreation, ComfyUI, Local Speech-to-Text models, Spokenly, Nvidia Parakeet` 


- **使用 ComfyUI 重现子弹时间 (Bullet Time)**：一位成员分享了一段[精彩视频](https://www.youtube.com/watch?v=iq5JaG53dho)，演示了如何重现著名的“**子弹时间**”场景，重点介绍了从 15 分钟处开始使用的 **ComfyUI** 和新技术。
   - 他们链接了一个 [Deepfates 的推文](https://x.com/deepfates/status/2009295329057702081?s=20)，展示了该技术的一个示例。
- **Deepfates 建议放弃付费听写服务**：**Deepfates** 建议用户停止为订阅制的转录应用付费，推荐使用免费、离线的本地模型，例如 [Spokenly](https://spokenly.app/) 配合 **Nvidia Parakeet**，以获得卓越的性能。
   - 一位成员尝试了这种配置，并指出 *Parakeet 模型速度极快且准确*，但 *Spokenly 的 iOS 键盘比 Wispr Flow 的更难用*。
- **Linux 游戏：Nvidia 的小众市场？**：一位成员询问了 **alt** 在 Linux 机器上的可用性，并暗示由于 **Valve** 的推动，Linux 游戏机的兴起为拥有现有 **GPU** 配置的用户提供了一个小众机会。
   - 他们推测，虽然 Linux 市场比 macOS 小，但不断增长的游戏领域使其成为一个值得关注的目标。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1458922363885191238)** (14 messages🔥): 

> `Community Spotlight Talk Series, Diffusion-Based World Models, Common Crawl LangID, ChatGPT Simulation, Perplexity Search` 


- **Community Spotlight 演讲系列回归**：“Community Spotlight”演讲系列即将回归，旨在推广酷炫的研究。这次计划保持连贯性，展示社区成员的作品，例如[这段 RTX 6000 视频](https://cdn.discordapp.com/attachments/729741769738158194/1458922361129664575/2026-01-06_22-30-45.mp4?ex=6962b854&is=696166d4&hm=87d347c7ce26992e66b54eacc721962e906554618a5da2750560a62fed51b7a7&)。
- **扩散模型在消费级硬件上实时运行**：一位成员将在 <t:1767985200:f> 发表演讲，讨论在消费级硬件上实时运行**基于扩散的世界模型 (diffusion-based world models)**，并通过一段使用 **RTX 6000** 的视频进行了演示。
   - 您可以在[此处](https://discord.gg/PWt2DmRd?event=1458918516471369790)预约参加。
- **Common Crawl 应对大规模 LangID 挑战**：一位来自 **Common Crawl** 的成员将分享他们在处理大规模 **LangID**（语言识别）方面的工作及相关挑战。
- **ChatGPT 认为它生活在模拟中**：成员们注意到 **ChatGPT** 认为它处在一个精心设计的模拟之中。
- **Perplexity Pro 的搜索能力下降**：一位成员表示 **Perplexity** 曾经更好用，但现在 **ChatGPT, Gemini, 和 Claude** 拥有相对更好的网络搜索和深度研究能力。
   - 另一位成员表示：*“它的搜索能力现在很糟糕”*。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1458944303085916233)** (31 messages🔥): 

> `高效训练方法, VRAM 优化, 基准测试结果, 测试数据集选择` 


- **CGGR 训练方法声称提升了 VRAM 效率**：一种新方法声称通过减少梯度数量来提高 VRAM 效率，可能节省高达 **75% 的 VRAM**，同时提高训练速度；在批量大小（batch size）为 **4** 时，其 VRAM 占用（**6-7 GB**）与普通训练在批量大小为 **1** 时相当。
   - 在 fineweb-edu 上使用 **SmolLM-135M** 进行的初步基准测试显示损失（loss）各有不同但可以调整，微调期间在熵为 **2.0** 时跳过率（skip rates）约为 **30%**。
- **关于机器学习数据获取难度的辩论爆发**：一场关于机器学习领域数据获取是变得更难还是更容易的讨论随之展开。
   - 一位参与者认为 *数据正变得越来越难获取*，而另一位则反驳说 *看起来容易得多*，并指出 *我有充分的理由相信这一点*。
- **初步基准测试引发了关于方法论的辩论**：分享的初步基准测试显示，某些配置下的 TPS 受限于网络，且跳过率为 **25%**，这表明在更大的模型上可能会有更高的跳过率。
   - 针对测试步数有限的质疑，相关人员承诺将进行更广泛的测试，并优先在数学数据集等合适的数据集上进行微调。
- **提议使用数学数据集进行微调和精度测量**：成员们讨论了使用 **GMSK8**、**AIME 2024** 和 **Numiea Code** 等数学数据集来微调模型和测量精度的适用性，建议将自动化评估作为基线。
   - 由于 AIME 2024 等数据集的复杂性，建议从较小的模型（**100M**）开始，随后可能扩展到更大的模型（**1B**）以处理更复杂的任务。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1459007475696271476)** (3 messages): 

> `AI 可解释性, Dead Salmon 伪影, Fish Finder 实验` 


- **Dead Salmon 伪影困扰 AI 可解释性**：一名成员引用了一篇论文（[Dead Salmon: An Artifact of Random Initialization](https://arxiv.org/abs/2512.18792)），指出对于**随机初始化的神经网络**，*特征归因（feature attribution）、探测（probing）、稀疏自编码（sparse auto-encoding）甚至因果分析（causal analyses）* 都能产生 *看起来合理的解释*。
- **Fish Finder 运行获得青睐**：一名成员表示迫不及待想让 **批量 Fish Finder 运行** 结束，发现关于 *dead salmon* 的链接对他们的工作非常有帮助。
   - 他们能够证明其结果是 *良好但有噪声的*，现在他们更深入地了解了 **如何从轻量级 pipeline 中去除噪声**。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1458945878810169535)** (6 messages): 

> `Qwen 发布, 阿里巴巴 Qwen, 模型发布` 


- **Qwen 模型预热，但未正式发布**：一位用户对 [阿里巴巴](https://x.com/alibaba_qwen/status/2009264754917863924?s=46) 可能发布 **Qwen 模型** 感到兴奋，但发现并非真正发布后感到失望。
   - 用户表达了这样一种观点：在版本发布并可供实际使用之前，该模型 *还不如不存在*，希望未来的发布能吸收预热模型中的经验。
- **对模型延迟发布的沮丧**：在最初的兴奋和随后的失望之后，用户表达了对 Qwen 模型无法获取的沮丧。
   - 用户强调渴望一个能够进行实际应用和学习的实体发布，突出了预热模型与其核心可用性之间的脱节。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1459236305060888738)** (1 messages): 

> `调试 Neox 训练崩溃, 数据加载中的 UnboundLocalError` 


- **Neox 训练因 `UnboundLocalError` 崩溃**：一名成员报告在模型训练期间发生了崩溃，报错为 `megatron/data/data_utils.py` 中的 `UnboundLocalError: local variable 'train_val_test_num_samples' referenced before assignment`。
   - 该成员提到进行了几次配置更改，并请求关于潜在原因的指引，同时指出 *环境在其他运行中表现正常*。
- **配置更改可能是主因**：提问者怀疑最近的配置更改是导致错误的可能原因。
   - 错误是在多次配置更改后出现的，鉴于环境在其他情况下运行正常，他们正在寻求关于具体哪项更改可能导致此问题的见解。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1459193234705092651)** (11 messages🔥): 

> `Deepseek V4, Minimax, Moonshot AI vs Deepseek, Deepseek CEO` 


- **Deepseek V4 目前尚不存在**：一位成员指出，尽管外界有所期待，但目前还没有 **Deepseek V4**。他引用了一篇 [Reuters 文章](https://www.reuters.com/technology/deepseek-launch-new-ai-model-focused-coding-february-information-reports-2026-01-09/Okwill)，指出该模型将于 2 月发布。
   - 另一位成员猜测他们是否在期待更多的 **V3s** 版本。
- **Deepseek 炒作受到质疑**：一位成员认为 **Deepseek** 被西方媒体过度炒作了，这些媒体可能并不熟悉中国的 AI 领域现状。
   - 他们认为 **Moonshot AI** 和 **Kimi** 远优于 Deepseek，并形容后者“谄媚且极其危险（dangerous AF）”。
- **推荐阅读 Deepseek CEO 的博客文章**：一位成员建议深入了解 Deepseek 的 **CEO** 及其撰写的博客文章。
   - 另一位成员简单地进行了澄清并询问：“你是说 Deepseek 的 CEO 吗？”


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1458936803003338954)** (9 messages🔥): 

> `MiniLM-L6-v2 Model Architecture, BERT Architecture PR, Nightly Server Issues on Linux, Embedding Bug on Nightly` 


- **MiniLM-L6-v2 模型架构已 Fork**：一位成员 Fork 了代码库，并在其 Fork 分支的 [此链接](https://github.com/RWayne93/modular/tree/feat/all-MiniLM-L6-v2-model-architecture/max/python/max/pipelines/architectures/minilm) 发布了 **MiniLM-L6-v2 模型架构** 特性分支。
   - 他们暂时不想开启 PR，因为还在测试该架构，但基于测试结果来看，它似乎是正确的。
- **BERT 架构 PR 已提交**：一位成员提交了关于 **BERT 架构** 的 PR。该架构最初是针对最新的 **max / modular** 稳定版开发的，随后更新到了最新的 Nightly 变更。
   - 他们还根据 PR 评论中 **Copilot** 的建议实施了更改。
- **Linux Nightly 服务端 Bug 浮现**：一位成员在最新的 Nightly 版本中遇到了服务端无法启动的问题，且仅发生在 Linux 上，并提交了复现工单。
   - 该 Bug 出现在 **GitHub 的 ubuntu-24.04 (x86) runner** 上，成员指出对于任何模型，`max serve` 都无法跳过构建图（building graph）的部分。
- **Embedding Bug 导致 Nightly 崩溃**：一位成员报告了当前 Nightly 版本中关于 Embeddings 的一个 Bug，该 Bug 会导致服务端崩溃，并附带了 [日志转储 (log dump)](https://cdn.discordapp.com/attachments/1212827597323509870/1459271585398788156/logdump.txt?ex=6962ac11&is=69615a91&hm=bc5337146bd43bca0a33bdd9997ac3e0f23b535d4c6ab27956ad171dc9da8a37&)。
   - 他们还表示已经有了修复方案，但正尝试通过 Bazel 在 **modular** 代码库中运行所有内容。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1458956001586843841)** (8 messages🔥): 

> `Manus Startup credit, Multiple conversations on a single website, AI developer opportunities` 


- **关于 Manus Startup Credit 的咨询**：一位成员询问了关于 **Manus Startup credit** 的事宜，想知道是否有人申请过以及成功的几率有多大。
   - 在给定的上下文中没有提供任何回复。
- **单个网站，多次对话？**：一位成员询问是否有人找到了通过多个不同的独立对话来处理由 **Manus** 创建的**单个网站**的方法。
   - 在给定的上下文中没有提供进一步的细节或回复。
- **AI 开发者寻找机会！**：一位成员询问是否有人在寻找具有 **Chatbots**、**AI Agent**、**自动化工作流（automation workflows）**、**API 集成**和**自定义 AI 工具**经验的开发者。
   - 另一位成员随后开玩笑地问是否有人愿意免费做，并添加了一个 [tenor gif](https://tenor.com/view/plink-nerd-plank-plink-cat-cat-gif-11096663429307162255)。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1459306533450940631)** (1 messages): 

> `AI Engineer YT Channel, Awesome Talk` 


- **AI Engineer YT 频道展示精彩演讲**：一位成员分享了 [AI Engineer YT 频道](https://www.youtube.com/watch?v=-cKUW6n8hBU) 的链接，其中包含一场“精彩的演讲”。
   - 这场演讲由一位显然尚未亲自发布该内容的嘉宾提供。
- **演讲被社区成员评价为“棒极了”**：一位社区成员强调可以在 AI Engineer YouTube 频道上找到这场演讲。
   - 视频地址为 [https://www.youtube.com/watch?v=-cKUW6n8hBU](https://www.youtube.com/watch?v=-cKUW6n8hBU)，其内容广受好评。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1458939232688279657)** (4 messages): 

> `RLM PR, Clojure Implementation, loop-infer npm package` 


- **RLM PR 的期待感升温**：成员们对即将到来的 **RLM PR** 表示期待，并将近期的活跃迹象视为积极信号。
   - 爱好者们正密切关注关于 **RLM** 实现的进一步进展和公告。
- **Clojure 构建 RLM**：一个 **Clojure** 版本的 RLM 实现已经出现，并具备可独立运行的服务器模式。
   - 这允许用户将 **RLM** 作为独立进程运行，增强了其灵活性和集成选项。
- **loop-infer 已在 npm 上架**：可能与 RLM 相关的 **loop-infer** 软件包现在可以通过 npm 访问。
   - 社区现在可以通过 [此 GitHub 仓库](https://github.com/unravel-team/loop-infer) 利用该软件包，从而可能简化其工作流程。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1458946905802412084)** (2 messages): 

> `Speed Bounties, Tinygrad Instance Access, CLAUDE.md` 


- **Tinygrad 爱好者寻求“速度”悬赏任务指南**：一名成员询问了如何开始在 *tinygrad* 项目中进行 **“速度”悬赏任务 (speed bounties)** 的指南。
   - 具体而言，他们寻求有关申请访问 *tinygrad* 实例以运行测试的信息。
- **CLAUDE.md 声明引发争议**：一位成员引用了 **CLAUDE.md**，暗示其中包含与另一份声明相矛盾的信息。
   - 遗憾的是，并未提供矛盾之处的具体细节。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1459192200561954950)** (1 messages): 

> `Clay, AI outreach workflow, Prompt Engineering` 


- **利用 Clay 和 AI 自动化销售与求职外联**：一场为期 **1.5 小时的直播工作坊**将拆解 **Clay.com + AI 工作流**，该流程曾为真实客户触达约 **1,000 个潜在客户**，实现了 **40%+ 的接受率**和 **18%+ 的回复率**（[在此注册](https://luma.com/jt1vr0u5)）。
   - 工作坊将涵盖端到端 AI 外联系统、Clay.com 演示、Prompt Engineering，以及与 Apollo、Attio 和 n8n 等工具的可选集成。
- **通过专家级 Prompt Engineering 打造引人入胜的外联信息**：参与者将学习用于高质量外联的 **Prompt Engineering 技术**，包括无代码 meta prompting、结构化输出以及避免“AI 味”信息的 QA。
   - 参与者将获得一份可重复使用的求职社交工作流大纲、可直接复制粘贴的 Prompt 模板，以及用于信息质量控制的简单 QA 准则。


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1459147795855446132)** (1 messages): 

> `Model Context Protocol, GitHub Issue` 


- **GitHub Issue 分享 Model Context Protocol 相关内容**：一名成员分享了一个关于 **Model Context Protocol** 的 [GitHub issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/2064) 链接。
   - 他们表示，如果问题过于显而易见请见谅，因为他们本周才刚刚开始实现该规范。
- **新的 MCP 实现者加入战场**：一位新的实现者本周开始研究 **Model Context Protocol (MCP)** 规范。
   - 他们通过分享 GitHub issue 链接迅速寻求社区见解，展现了理解现有讨论的积极性。


  

---


---


---