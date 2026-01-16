---
companies:
- openai
- cursor
- github
- cerebras
- modal
- artificial-analysis
- vllm
date: '2026-01-15T05:44:39.731046Z'
description: '**OpenAI** 推出了 **GPT-5.2-Codex** API，称其为针对长时间运行任务和网络安全领域最强大的编程模型。**Cursor**
  已集成 GPT-5.2-Codex，使其能够自主运行浏览器长达一周，并产出了超过 300 万行 Rust 代码。**GitHub** 也将其整合进自家的代码工具中，降低了企业采用的门槛。相关讨论强调了智能体（agent）系统中审查循环（review
  loops）的重要性，并就编程模型的评估指标展开了辩论。


  **OpenAI** 与 **Cerebras** 达成合作，旨在提升推理速度并降低延迟；目前 Cerebras 提供 **GLM-4.7** 服务，推理速度达每秒
  1,445 个 token 且延迟极低。供应商基准测试揭示了在吞吐量、延迟和上下文窗口大小之间的权衡。**Modal** 分享了其拥有 2 万个 GPU 的自托管推理集群的运维扩展见解，重点关注使用
  **vLLM** 和 FlashInfer 后端的批量推理优化。这些动态反映了业界对推理基础设施、长时程自主智能体（long-horizon autonomous
  agents）以及编程模型评估的关注。'
id: MjAyNi0w
models:
- gpt-5.2-codex
- glm-4.7
people:
- swyx
- kevinweil
- pierceboggan
- mntruell
- scaling01
title: 今天没发生什么特别的事。
topics:
- long-running-tasks
- autonomous-agents
- code-generation
- inference-speed
- latency
- batch-inference
- gpu-scaling
- model-evaluation
- agent-systems
- operational-scaling
---

**平静的一天**

> 2026年1月13日至1月14日的 AI 新闻。我们为您检查了 12 个 Reddit 分区、[**544** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **24** 个 Discord 社区（包含 **204** 个频道和 **5168** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**445 分钟**。**我们的新网站**现已上线，支持全文元数据搜索，并以精美的风格化方式呈现所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 [@smol_ai](https://x.com/Smol_AI) 上向我们提供反馈！

关于 GPT 5.2 Codex API 发布以及 Cursor 如何成功利用它自主运行一周并创建一个初步可用的浏览器，引发了一些热议！

---

# AI Twitter 回顾

**OpenAI + GitHub + Cursor：GPT-5.2-Codex 迈向“长周期（long-horizon）”（且无处不在）**

- **API（及 IDE）中的 GPT-5.2-Codex**：OpenAI 在 **Responses API** 中发布了 **GPT-5.2-Codex**，将其定位为处理功能开发、重构和找 Bug 等**长周期任务**的最强编程模型；他们还明确称其为迄今为止在理解代码库漏洞方面“网络能力（cyber-capable）”最强的模型 ([OpenAIDevs](https://twitter.com/OpenAIDevs/status/2011499597169115219))。Cursor 立即集成了该模型，并将其定义为“处理长周期任务的前沿模型” ([cursor_ai](https://twitter.com/cursor_ai/status/2011500027945033904))，同时也得到了强调扩展工作流尽职调查的开发者们的认可 ([sherwinwu](https://twitter.com/sherwinwu/status/2011503049890808040))。GitHub 也将其引入了 **@code** ([code](https://twitter.com/code/status/2011503658815668623))，并指出他们正在更改预览/GA（正式发布）标签，以减少企业采用的阻力 ([pierceboggan](https://twitter.com/pierceboggan/status/2011519932392226898))。
- **一个具体的“Agent 运行一周”数据点**：一份引人注目的报告声称，一个团队“在 Cursor 中使用 GPT-5.2 构建了一个浏览器”，该 Agent **不间断运行了一周**，在数千个文件中生成了 **300万行以上的 Rust 代码**（涵盖 HTML 解析 → CSS 层叠/布局 → 绘制 → 自定义 JS VM），并且对于简单的网站“基本可用” ([mntruell](https://twitter.com/mntruell/status/2011562190286045552))。这成为了“持续 Agent 时间”和自主代码生成（Codegen）实际前沿的一个参考点 ([gdb](https://twitter.com/gdb/status/2011570314216718510); [kevinweil](https://twitter.com/kevinweil/status/2011587644468445445))。工程师们还强调了正在兴起的最佳实践，即 **Agent 系统需要一个一等的“评审（review）”循环**，以提高输出质量和安全性 ([scaling01](https://twitter.com/scaling01/status/2011580895573262717))。
- **评估话语：指标 vs “感觉（vibes）” vs 时间跨度**：多条推文认为，编程模型的进步由于评估设计和开发者日常实际感受的不同而被低估或高估；METR 的长时评估被认为比标准基准测试更能早期发现“跨越式进步” ([swyx](https://twitter.com/swyx/status/2011344788486774942))。其他人则在争论仅凭图表是否能支持结论，以及在真实的脚手架（Scaffold）中“时间跨度”指标应该意味着什么 ([\_lewtun](https://twitter.com/_lewtun/status/2011393239774048658); [RyanPGreenblatt](https://twitter.com/RyanPGreenblatt/status/2011648823458689304))。

---

**推理基础设施：Cerebras 合作伙伴关系 + “速度即产品”的经济学**

- **OpenAI 🤝 Cerebras**：Cerebras 宣布与 OpenAI 达成合作伙伴关系 ([cerebras](https://twitter.com/cerebras/status/2011531740804964855))。整个时间线上的观点是，对于 ChatGPT 式的体验，**延迟和 tokens/sec** 正日益成为用户可见的产品差异化点（且在竞争中优于 Gemini），即使其软件栈在通用工作负载上比 CUDA 更窄 ([Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2011537073292132565))。
- **供应商基准测试变得更加细化**：Artificial Analysis 发布了 **GLM-4.7** 的供应商对比，强调了速度、延迟和成本之间的权衡。示例数据：Cerebras 以 **~1,445 output tokens/s** 的速度运行 GLM-4.7，**TTFAT 约为 1.6s**；而像 Fireworks/Baseten 这样的 GPU 供应商在吞吐量和延迟上落后，但支持更大的上下文窗口（Cerebras 为 **131k**，其他供应商为 **200k**，Parasail 除外）以及不同的缓存折扣 ([ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2011581689567592641))。
- **运营规模化内容**：Modal 发布指南指出，自托管推理现在的经济效益已经可以匹配或超越 API，并提供了技术方案和代码示例 ([charles_irl](https://twitter.com/charles_irl/status/2011484220032762114))。SemiAnalysis 重点推介了 Modal 关于维持 **2万张 GPU** 集群健康的运营报告 ([SemiAnalysis_](https://twitter.com/SemiAnalysis_/status/2011498598043660777))。vLLM 和 Modal 的内容集中在如何通过**批处理推理（Batch Inference）**使 H100 达到饱和（FlashInfer 后端、异步调度、批次大小优化） ([vllm_project](https://twitter.com/vllm_project/status/2011585247297880501))。

---

**Agent 工程走向产品化：技能、动态工具加载与架构选择**

- **技能作为可移植层**：Phil Schmid 为 `antigravity` 发布了 **Agent Skills**，具备标准化的文件夹结构 (`.agent/skills/`, `~/.gemini/antigravity/skills/`)，并兼容 Gemini CLI / Claude Code / OpenCode 风格的生态系统 ([\_philschmid](https://twitter.com/_philschmid/status/2011345054343053370))。Hugging Face 的从业者回应称，“/plugin 接口”存在严重的版本控制摩擦；对于大多数团队而言，**小型垂直技能 + CLI/MCP** 是更稳健的路径 ([ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/2011460800427286783))。
- **LangSmith Agent Builder 发布**：LangChain 发布了 **LangSmith Agent Builder**，提出了“Agent 即文件系统”的概念，内置记忆功能、环境感知型 Agent 触发器，并支持 **Skills/MCP/Subagents** ([LangChain](https://twitter.com/LangChain/status/2011501888735494184); [hwchase17](https://twitter.com/hwchase17/status/2011503746312987128))。实际案例包括一个从 Slack 到 Linear 工单的自动化 Agent，构建过程“无需代码，只需 Prompt” ([docs_plz](https://twitter.com/docs_plz/status/2011536177556570203))。CopilotKit 增加了中间件，可将 LangChain 预构建的 Agent 转换为面向 UI 的应用（包括 “Deep Agents”） ([CopilotKit](https://twitter.com/CopilotKit/status/2011453920321929237))。
- **何时使用多 Agent（通常建议：不要）**：LangChain 的一篇博文列出了四种模式——**Subagents**、**Skills**、**Handoffs**、**Router**——并明确建议从 **单个 Agent** 开始，除非遇到限制（上下文窗口、分布式所有权、分解需求） ([LangChain](https://twitter.com/LangChain/status/2011527733176856671); [sydneyrunkle](https://twitter.com/sydneyrunkle/status/2011514042075222029))。这一主题在 OSS 账号指南中被反复提及 ([LangChain_OSS](https://twitter.com/LangChain_OSS/status/2011515750625001609))。

---

**工程师们争论的模型与研究笔记：长上下文、记忆模块、剪枝/蒸馏、多模态 RAG 以及评测指标的脆弱性**

- **DroPE / 长上下文无需位置嵌入**：一个推特线程总结了一个简单的配方——采用预训练 LLM，**移除 RoPE**，在**没有位置嵌入**的情况下进行微调。报告显示在标准数据集上表现相当，且长上下文行为有所改善，已在 **SmolLM-1.7B** 和 **Llama2-7B** 上完成测试 ([gabriberton](https://twitter.com/gabriberton/status/2011326182986564090); [gabriberton](https://twitter.com/gabriberton/status/2011326193082253413))。
- **DeepSeek “Engram” 记忆模块讨论**：多条推文讨论了 DeepSeek 与北京大学的合作研究，主张通过 **MoE (稀疏计算)** + **Engram (稀疏存储)** 来“将思考与记忆分离”——基于哈希的 O(1) 查找，将检索到的 n-grams 作为向量融合进 Transformer 流中，涉及预取/延迟隐藏和驻留内存的表等基础设施改进 ([ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/2011357373772845097); [LiorOnAI](https://twitter.com/LiorOnAI/status/2011468534887469448); 代码链接 [LiorOnAI](https://twitter.com/LiorOnAI/status/2011526199420600378))。
- **Mistral “Ministral 3” (小模型配方)**：一份新技术报告的详尽总结强调了 **剪枝 + 蒸馏**（在预训练/后训练中使用教师模型；在后训练中进行在线 DPO），以及具体的剪枝启发式方法（通过输出/输入范数比进行层剪枝；通过 PCA 旋转进行隐藏层维度剪枝；通过门控激活得分进行 FFN 剪枝） ([eliebakouch](https://twitter.com/eliebakouch/status/2011548952676499480); 论文链接 [qtnx_](https://twitter.com/qtnx_/status/2011510403550024087))。
- **多模态 RAG 系统设计**：UniversalRAG 提出了 **模态感知路由**（避免强行将所有内容放入同一个向量空间），并跨 **模态 + 粒度**（段落 vs 文档；片段 vs 完整视频；表格/图像）进行检索。它支持经过训练或无需训练的路由（提示前沿模型选择模态/粒度），并在 10 个基准测试中取得领先 ([omarsar0](https://twitter.com/omarsar0/status/2011442693134754243))。用于多模态 RAG 评估的 ViDoRe V3 基准论文也已发布 ([antonio_loison](https://twitter.com/antonio_loison/status/2011398238910517249))。
- **基准测试的脆弱性 (VLMs)**：VPBench 认为，演示中的微小变化（例如 **红色 vs 蓝色标记**）就可能重新排列 VLM 排行榜——这对于那些将排行榜差异视为稳健信号的人来说，是极具说服力的反驳依据 ([lisabdunlap](https://twitter.com/lisabdunlap/status/2011521499182875116))。

---

**产品与组织动态：“开放”作为策略，以及实验室间的人才洗牌**

- **Airbnb 聘请 Meta Llama 负责人担任 CTO**：Ahmad Al-Dahle 宣布加入 Airbnb 担任 CTO；他将其归功于 Meta 在 Llama 上的开源押注（**12亿+ 下载量**，**6万+ 衍生模型**），并将 Airbnb 视为应用先进模型能力的产品前沿 ([Ahmad_Al_Dahle](https://twitter.com/Ahmad_Al_Dahle/status/2011440460821320056))。多位领导者对此表示支持 ([sama](https://twitter.com/sama/status/2011490615985414382); [ClementDelangue](https://twitter.com/ClementDelangue/status/2011455261329023329); [markchen90](https://twitter.com/markchen90/status/2011545090737782810))。
- **Thinking Machines Lab / OpenAI 领导层变动**：Mira Murati 宣布 Barret Zoph 离开 TML，**Soumith Chintala** 成为 CTO ([miramurati](https://twitter.com/miramurati/status/2011577319295692801))。随后不久，OpenAI 宣布 Barret Zoph、Luke Metz 和 Sam Schoenholz 回归 OpenAI ([fidjissimo](https://twitter.com/fidjissimo/status/2011592010881446116); [barret_zoph](https://twitter.com/barret_zoph/status/2011593621435531355))。
- **开源与“中型机构”**：HF 的 Clement Delangue 认为初创公司和中型科技公司可以实质性地推动开放科学/开源 AI，并指出 **fal** 和 **Lightricks** 的热门模型就是证据，并将 Airbnb 聘请 CTO 视为一种可能的信号 ([ClementDelangue](https://twitter.com/ClementDelangue/status/2011477703698895245))。LTX-2 庆祝 **1,000,000 次 HF 下载** ([ltx_model](https://twitter.com/ltx_model/status/2011432938819252566))，进一步证明“开放分发”现已成为一种增长渠道。

---

**热门推文（按互动率排序）**

- **Gemini “Personal Intelligence” 推出**：Google 宣布通过连接 Google 应用（Gmail/Photos/Search/YouTube 历史记录）实现 Gemini 个性化，强调选择性加入 (opt-in) + 隐私控制；在 Google/Gemini 领导层账号中获得了极高互动 ([Google](https://twitter.com/Google/status/2011473056921706852); [sundarpichai](https://twitter.com/sundarpichai/status/2011475851670667356); [joshwoodward](https://twitter.com/joshwoodward/status/2011471375521710130))。
- **GPT-5.2-Codex 发布 + 生态系统采用**：API 发布和 Cursor 集成是互动率最高的工程推文之一 ([OpenAIDevs](https://twitter.com/OpenAIDevs/status/2011499597169115219); [cursor_ai](https://twitter.com/cursor_ai/status/2011500027945033904))。
- **“300万行代码浏览器”长周期 Agent 轶事**：作为连续 Agent 工作的生动案例被广泛流传 ([mntruell](https://twitter.com/mntruell/status/2011562190286045552))。
- **Vercel 的 Agent 评估/React 性能技能**：`react-best-practices` 作为一种 “Agent Skill” + 评估套件获得了极高关注 ([vercel](https://twitter.com/vercel/status/2011589806250426615))。


---

# AI Reddit 热点回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. 本地 LLM 硬件与性能对比

  - **[M4/M5 Max 128gb vs DGX Spark (或 GB10 OEM)](https://www.reddit.com/r/LocalLLM/comments/1qcmmvw/m4m5_max_128gb_vs_dgx_spark_or_gb10_oem/)** (活跃度: 153): **用户正在对比 NVIDIA DGX Spark 和配备 M4 Max (128GB RAM) 的 MacBook Pro 在本地 LLM 推理方面的表现，主要用于代码补全和重构等编程任务。DGX Spark 提供了 CUDA 生态系统和强大的 GPU 算力，而 MacBook Pro 则受益于 Unified Memory 和 Apple 的 ML 技术栈。该用户并非专注于训练大模型，而是追求快速、可靠的本地推理。一个关键考虑点是 Apple Silicon 生态系统是否可以取代基于云的代码助手（如 Claude Code）。MacBook 更高的内存带宽被认为对推理有利，但需要合理管理预期，因为它可能无法达到云端性能。基准测试表明 M5 相比 M4 有显著的性能提升，且新的 MacBook Pro 机型可能很快发布。** 评论者辩论了 Apple Silicon 与 NVIDIA 硬件在文本生成方面的性能。一些人认为 MacBook Pro（特别是搭载 M3 Ultra 的型号）在纯文本生成任务中表现出色，而 DGX Spark 在需要广泛 GPU 能力的任务中表现更好。MacBook 较高的内存带宽被强调为推理优势，尽管 NVIDIA 的 CUDA 支持因其更广泛的框架兼容性而受到关注。

- M4 Max 相比 DGX Spark 提供显著更高的内存带宽，这对于推理任务非常有利。然而，由于 DGX Spark 兼容 NVIDIA 的 CUDA，在大多数框架的支持方面更具优势，这对于使用多样化机器学习框架的用户来说是一个主要优点。
- M3 Ultra Mac Studio 在纯文本生成任务方面被强调为优于 DGX Spark。尽管 NVIDIA 具备硬件性能，但 M3 Ultra 在文本生成速度上持续领先，这归功于其针对此类任务优化的架构。与之相对的是，DGX Spark 在微调以及图像/视频生成等其他领域具有更广泛的能力。
- DGX Spark 以其紧凑的尺寸和能效著称，运行功率低于 100W，待机功率约为 10W。它还因其可扩展性受到称赞，允许连接额外的单元。然而，带宽限制的问题也被提出，表明虽然它效率很高，但在某些任务中可能无法达到 Mac Studio 等替代方案的性能。

- **[16GB VRAM 能容纳的最大本地 LLM 是什么？](https://www.reddit.com/r/LocalLLM/comments/1qcuyh2/what_is_the_biggest_local_llm_that_fit_in/)** (活跃度: 103): **使用 RTX 5080 和 16GB VRAM，你可以运行的最大本地 LLM 可能在 `14B` 参数左右，特别是如果你想保持有用的上下文大小。像 `GPT-OSS-20B` 旗下的模型可能装得下，但需要显著的量化（可能低于 `4-bit`），这会降低质量。为了获得最佳性能，建议使用 `14B` 模型，因为它有效地平衡了模型大小和上下文容量。更大的模型（如 `30B`）需要卸载到 CPU，由于 VRAM 限制，这可能并不实用。** 评论者认为，虽然通过重度量化在技术上可以实现 `30B` 模型，但由于质量和上下文限制，它们可能并不实用。共识是 `14B` 模型更适合在 16GB VRAM 配置上保持性能和可用性。

    - **SKirby00** 强调了将 30B 等大型模型装入 16GB VRAM 的局限性，指出即使采用低于 4-bit 的激进量化，质量也可能显著下降。他们建议将目标锁定在 14B 左右的模型，以平衡大小和上下文容量，并指出 14.5GB 的模型虽然在技术上能装下，但在实际用例中并不实用。
    - **BigYoSpeck** 提供了在 Ryzen 9 5900x、64GB DDR4 和 16GB Radeon RX 6800 XT 上的不同模型性能基准测试。他们报告运行 `gpt-oss-20b` 的速度超过 120+ tokens per second，部分卸载到 CPU 的 `Qwen3 30b` 为 40 tokens per second，而卸载了 32 个 MOE 层到 CPU 的 `gpt-oss-120b` 速度为 23 tokens per second，这表明在其他系统上可能会实现类似或更好的性能。
    - **PermanentLiminality** 建议将模型大小控制在 VRAM 的 80% 以下，以为上下文留出空间，建议将 13GB 模型作为 16GB VRAM 的实际限制。他们指出，虽然可以使用系统 RAM 进行溢出（spillover），但这会显著降低速度。他们提到 `Qwen 3 30B` 可以有效地处理一些溢出，使其成为在这些限制下能高效运行的最大模型之一。

- **[小型 AI 电脑本地运行 120B 模型：除了便携性和隐私还有其他用例吗？](https://www.reddit.com/r/LocalLLM/comments/1qcu498/small_ai_computer_runs_120b_models_locally_any/)** (活跃度: 49): ****TiinyAI** 开发了一款紧凑型 AI 设备，能够以 `30W` 的功耗和 `80GB RAM` 在本地运行 `120B` 参数模型。该设备被定位为 **DGX Spark** 等大型系统的更便携且更具成本效益的替代方案，后者提供 `128GB RAM` 和更高的性能，但成本更高、体积更大。TiinyAI 设备在优先考虑便携性和隐私而非原始计算能力的场景中具有潜在应用价值，例如实地考察或互联网接入不可靠的地区。** 评论者对该设备的内存带宽表示怀疑，推测其可能在 `80Gb/s` 左右，这可能会限制其与标准 PC 或笔记本电脑相比的性能。此外，人们对价格和可用性也存在疑虑，一些人认为它在互联网接入受限的场景中可能很有用。

- 一个被提出的关键技术关注点是这款小型 AI 计算机的内存带宽，估值在 80Gb/s 到 200Gb/s 之间。这一带宽对于高效运行像 120B 参数量这样的大型模型至关重要。如果带宽处于较低水平，它可能无法超越普通的 PC 或笔记本电脑，这可能会限制其在便携性和隐私之外的实际应用。
- 该设备的定价受到了质疑，推测配备 80GB RAM 的单板计算机（SBC）价格约为 1400 美元。这种怀疑源于目前无法立即购买，这让人对该设备在当前市场上的可行性和实用性产生了疑问。
- 文中强调了应对互联网断网的潜在用例，指出在互联网访问受限或受威权政权监控的情况下，此类设备可能具有价值。这突显了在这些条件下，本地处理能力对于维持 AI 工具访问的重要性。

### 2. 创新的 AI 模型实现与实验

  - **[Shadows-Gemma-3-1B: 通过 topk20 logprob 蒸馏实现的冷启动推理](https://www.reddit.com/r/LocalLLaMA/comments/1qcd9m1/shadowsgemma31b_cold_start_reasoning_from_topk20/)** (活跃度: 41): **Shadows-Gemma-1B 是一款为 Google Tunix Hackathon 训练的推理模型，在 TPUv5-8e 上使用 `1569 个样本` 训练了约 `10 分钟`，在 A40 上训练了 `20 分钟`。该模型采用了一种名为 *shadow tokens* 的新方法，这些 Token 是通过从非推理型教师模型 **gemma-3-4b-it** 进行 topk20 logprob 蒸馏识别出来的。这些在低排名中较早出现并在稍后被选中的 Token 可能预示着推理行为，例如回溯（backtracking）和解法探索。该模型使用了鼓励交替推理的系统提示词进行训练，虽然它并未声称优于其他模型，但在复杂问题上展示了改进的推理能力。关于训练过程的更多细节，包括损失函数和代码优化，将在随后的复盘报告（post mortem）中分享。** 一位评论者建议探索使用更大的教师模型（如 gemma-12b-it 或 gemma-27-it）以获得可能不同的结果。另一位评论者对训练数据集的发布表示感兴趣，并指出了 Deep Cogito v2.1 在蒸馏方面的有效性。

    - 一位用户建议使用更大的模型（如 `gemma-12b-it` 或 `gemma-27-it`）作为蒸馏的教师模型，暗示由于它们更大的容量和可能更细腻的理解力，可能会进一步提升结果。
    - 另一位用户强调了使用概率分布中的 Token 持久性（token persistence）作为推理深度衡量标准的创新方法。这种方法允许通过训练模型来增强推理行为，这在模型训练中是一个新颖的概念。该用户还对从 PyTorch 转向 JAX 过程中遇到的技术挑战表示关注，暗示这可能涉及到特定框架优化或问题的见解。

  - **[使用本地 VLMs 进行 OCR 并输入到 NLP 分类流水线 - 正在寻找 Beta 测试人员 (Loggr)](https://www.reddit.com/r/LocalLLaMA/comments/1qcd8sw/using_local_vlms_for_ocr_to_feed_into_an_nlp/)** (活跃度: 10): **Loggr 正在为 Apple Silicon 开发一款完全离线运行的健康日志应用，利用自定义 NLP 流水线从自由格式文本中提取结构化健康数据，延迟低于 100ms。该应用正在集成一项使用 `Qwen2.5-VL-3B` 模型扫描手写日志的功能，该模型通过 MLX 进行量化以用于 OCR，可适配 `8GB` 的统一内存。需要 `12GB+` 内存的 `7B` 模型能更好地处理潦草的手写内容。该应用在夜间以批处理模式处理日志，并考虑采用与 Apple 的 Vision 框架相结合的混合方法来实现快速预览。团队正在寻找 Beta 测试人员，以评估其在具有挑战性的手写内容和布局上的表现。更多细节和报名信息请访问 [loggr.info](http://loggr.info)。** 评论者建议尝试使用带有自定义手写模型的 `PaddleOCR`，认为与通用 VLMs 相比，它在处理潦草手写内容时可能具有更好的性能。另一个建议是测试 `MiMo-VL-7B-RL`，它与 `Qwen2.5-VL` 兼容，并可能提供更智能的表现。此外，还有人对该应用是否支持文本转语音（text-to-speech）功能感兴趣。

    - 一位用户建议在 OCR 任务中使用 PaddleOCR 配合自定义手写模型，并指出专门的 OCR 模型在处理潦草手写内容时可以优于 Qwen2.5-VL 等通用 Vision-Language Models (VLMs)。这突显了在特定任务中使用专门模型的潜在优势，即使这些模型缺乏更通用模型的整体智能。
    - 另一位用户推荐尝试 MiMo-VL-7B-RL 作为 Qwen2.5-VL 7B 的替代方案，并指出 MiMo-VL-7B-RL 完全兼容，且在他们的使用场景中显得“更聪明”。他们提供了该模型在 Hugging Face 上的链接以便进一步探索：[MiMo-VL-7B-RL](https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-RL-2508)。

### 3. 电子商务与开发的 AI 协议和框架

  - **[Google 刚刚开源了 Universal Commerce Protocol。](https://www.reddit.com/r/LocalLLM/comments/1qcpoaw/google_just_opensourced_universal_commerce/)** (活跃度: 32): **Google** 开源了 **Universal Commerce Protocol (UCP)**，该协议允许 AI Agent 自主管理电子商务任务，如产品发现、购物车管理和支付处理。关键集成包括用于多步工作流的 **Agent2Agent (A2A)**、用于安全支付的 **Agents Payment Protocol (AP2)**，以及用于与现有 LLM 栈（如 vLLM 和 Ollama）集成的 **Model Context Protocol (MCP)**。该协议可在 [GitHub](https://github.com/Universal-Commerce-Protocol/ucp) 上获取。** 评论者们正在质疑目前零售商的采纳情况、Google 的支持时长，以及该协议是已经在应用中还是刚刚开源。

    - Universal Commerce Protocol (UCP) 是 Google 新开源的，但零售商是否采纳尚不确定。正如一位用户询问当前零售商的采纳情况所强调的那样，如果缺乏广泛支持，该协议的实用性将受到质疑。
    - 人们对 Google 对 Universal Commerce Protocol 的长期支持感到好奇，并提出了其与 Gemini 集成的问题。用户有兴趣了解 Google 对 UCP 的路线图，特别是它在 Gemini 等现有平台中的使用。
    - 讨论提出了关于 Universal Commerce Protocol 成熟度的问题，即它是一个新开发的协议，还是一个刚刚开源的现有协议。这种区分对于考虑实施该协议的开发者来说至关重要。

  - **[在消费级 GPU 上实现 16k 上下文编程会让 H100 对独立开发者变得无足轻重吗？](https://www.reddit.com/r/LocalLLM/comments/1qcmv3z/would_16k_context_coding_on_consumer_gpus_make/)** (活跃度: 36): **该帖子推测了在 `NVIDIA 3060` 等消费级 GPU 上实现用于编程的 `16k context window` 所带来的影响，质疑这是否会使 `H100` 等高端 GPU 对独立开发者的重要性降低。讨论强调 `16k context` 被认为是较小的，`64k` 是平均水平，而 `128k` 或 `1M` 分别被视为优秀或巨大。据运行 `4x3090s` 且使用 `128k` 或 `256k` 上下文的用户称，目前的本地模型在超过 `64k` 上下文后效果会变差，即使内存充足也是如此。** 评论者的共识是 `16k context` 对于重大的 AI 开发来说是不够的，这表明更复杂的任务需要更高的上下文窗口。

    - 讨论强调 16k 上下文窗口在大语言模型领域被认为是较小的，64k 是平均水平，128k 则被认为是不错的。像 Codex 和 Claude 这样的模型分别在 290k 和 240k 的更大上下文窗口下运行，而 Gemini Pro 可以处理高达 100 万个 token，这表明 16k 不会显著影响消费级 GPU 在严肃编程任务中的能力。
    - 一位用户提到在 4x3090 GPU 上使用 128k 或 256k 上下文窗口，但指出无论可用内存如何，大多数本地模型在超过 64k 上下文时性能往往会下降。这表明虽然更大的上下文窗口在技术上是可行的，但由于模型的局限性，它们在实践中可能没有益处。
    - 共识是，对于除了简单代码片段或自动完成功能之外的严肃应用，16k 上下文窗口是不够的。对于在编程领域具有重要意义的模型来说，其运行速度可能会太慢，因此在消费级 GPU 上实现 16k 上下文不会让 H100 对独立开发者变得无足轻重。


## 技术含量较低的 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI 数学定理证明与问题解决

  - **[Gemini “数学专用版”证明了一项新数学定理](https://www.reddit.com/r/singularity/comments/1qcq1ld/gemini_mathspecialized_version_proves_a_novel/)** (热度: 553): **Gemini**，一个“数学专用”的 AI 模型，据报道证明了一项新的数学定理，详情见 [推文](https://x.com/A_G_I_Joe/status/2011213692617285729?s=20) 和随附的 [arXiv 论文](https://arxiv.org/abs/2601.07222)。该模型的架构和训练针对数学推理进行了优化，利用了符号计算和定理证明方面的先进技术。这一进展强调了 AI 在推进数学研究方面的潜力，挑战了 AI 在处理复杂数学任务方面存在局限性的观点。评论者强调了 AI 进步的飞速节奏及其加速人类知识的潜力，同时也对商业利益对 AI 发展的影响表示担忧。还有观点认为，AI 在数学方面的能力往往被低估了。

    - AI 的飞速进步，特别是在数学和编程领域，被视为 AI 正在显著加速的领域。Gemini 模型“数学专用版”的开发就是一个例证，据报道它证明了一项新的数学定理。此类突破表明重大 AI 成就之间的时间间隔正在缩短，显示出创新的快速步伐。
    - 将 Gemini 模型用于解决 Erdős 问题被提议作为一个潜在的基准。Erdős 问题在数学界众所周知，且有着广泛的人类分析，使其成为评估 AI 数学问题解决能力的理想测试。这可以对该模型在推进数学研究方面的熟练程度和潜力提供严格的评估。
    - 讨论中提到了对 AI 执行数学任务能力的怀疑，一些人仍然对其能力表示怀疑。然而，AI 最近在证明数学定理方面的成就挑战了这种怀疑，证明了 AI 确实可以处理复杂的数学问题，并可能加速人类在这一领域的进步。

  - **[5.2 Pro 在维基百科列出的数十年数学难题上取得进展](https://www.reddit.com/r/OpenAI/comments/1qco4d7/52_pro_makes_progress_on_decades_long_math/)** (热度: 278): 该图是一条推文，宣布了 Moser’s worm 问题的最新数值上界，这是由 **Archivara** 使用 AI 模型 **5.2 Pro** 实现的。该方案涉及重新优化椭圆轨迹构建参数，将通用覆盖的面积减少到 `0.260069597`，超越了 2018 年创下的 `0.26007` 的前纪录。这一针对数十年未解几何问题的进展（该问题旨在寻找容纳任何长度为 1 的平面曲线的最小面积）已得到来自 **INRIA** 的数学家的验证。这一成就凸显了 AI 模型在提供正确工具和引导时解决复杂数学问题的潜力，尽管它们往往倾向于回避未解问题。评论讨论了让 AI 模型参与未解问题的挑战，指出 **5.2 Pro** 能够通过精心设计的工具、文献和提示词引导（prompt steering）取得进展。还有人提到禁用互联网访问以防止模型因问题“不可解”而放弃，从而使其能够集中精力并最终解决问题。

    - 5.2 Pro 模型通过使用精心挑选的工具和文献集，以及脚手架式的改进，在长期存在的数学问题上取得了进展。AI 模型的一个重大挑战是它们倾向于放弃复杂问题，例如 Riemann Hypothesis，而不进行尝试。通过采用一系列压力测试和提示词引导，诱导模型认真参与问题，其结果得到了来自 INRIA 的数学家的验证。
    - 一种鼓励 AI 模型解决难题的策略包括移除互联网访问，以防止它们在线搜索答案并得出问题不可解的结论。这种方法在解决 Erdős 问题时被成功使用，模型被迫在较长时间内依靠自身的推理能力。
    - 5.2 Pro 模型的伦理限制可能会干扰用户请求，例如在一个案例中，它拒绝提供保持 Linux 系统唤醒的解决方案，理由是可能违反政策。这突显了在平衡 AI 伦理准则与用户自主权方面持续存在的挑战，特别是在商业应用中。


### 2. DeepSeek 与 Spectral Sphere Optimizer 的发展

- **[[P] 我在单块 RTX 5090 上尝试实现 DeepSeek 风格的 MoE](https://www.reddit.com/r/MachineLearning/comments/1qcxhgw/p_my_shot_at_a_deepseek_style_moe_on_a_single_rtx/)** (活跃度: 64): **该帖子详细介绍了一个在单块 RTX 5090 GPU 上运行的混合专家 (MoE) 模型个人项目，该模型拥有 `2.36B 参数` 和 `8 个路由专家`，并采用 top-2 路由。该模型使用了带有 QK-normalization 的 Grouped Query Attention、RoPE 位置嵌入以及带有 RMSNorm 的 SwiGLU 激活函数。训练利用了 `TorchAO FP8 量化`、Muon 优化器以及多阶段学习率调度。数据流水线最初使用 MeCo (Metadata Conditioning then Cooldown)，但由于仅有 8 个专家时出现的问题，后来切换到了干净的语料库。主要挑战包括不当的路由初始化和缺乏稠密第一层，这导致了系统的不稳定性。作者建议不要在小型 MoE 模型上使用路由缩放，并指出 `1.2` 的缩放因子会导致不稳定。目前的训练指标包括 `3e-4` 的学习率、约 `1.9` 的 loss 以及 `19,415 tok/s` 的 token 处理速度。** 评论者对作者在没有正式 ML 训练的情况下取得的进展印象深刻，并指出其关注点在于稳定性和操作细节，而非高层架构。人们对该项目在个人学习之外的实际应用（如潜在的部署或模型蒸馏）表现出好奇。

    - 讨论强调了在单块 RTX 5090 上实现小型混合专家 (MoE) 模型的挑战，特别关注稳定性和操作细节，而非高层架构。这反映了现实世界中的产品开发，即最后阶段通常涉及处理边缘情况。评论者对这项工作在学习之外的实际应用感到好奇，例如模型的潜在部署或蒸馏。
    - 一个关键的技术见解是，在小型 MoE 模型中跟踪失效模式非常困难，因为许多来自大规模设置的技术并不适用。文中提到了围绕路由和缩放的不稳定性问题，其中稠密第一层和对称初始化是关键的教训。评论者质疑从这种设置中获得的见解是否可以转移到更大的系统，或者单 GPU 的限制是否限制了可扩展性，但也承认了明确阐述这些权衡的价值。
    - 评论强调了在小型 MoE 模型中理解失效模式，而非仅仅关注吞吐量或 loss 曲线的重要性。它指出，许多来自大规模模型的技巧在较小的设置中会默默失效，从而凸显了稠密第一层和对称初始化的重要性。讨论提出了一个问题：这种受限设置下的见解是否适用于更大的系统，并认为能够清晰阐述这些权衡是一种显著的优势。

  - **[[R] Controlled LLM Training on Spectral Sphere](https://www.reddit.com/r/MachineLearning/comments/1qcq27u/r_controlled_llm_training_on_spectral_sphere/)** (活跃度: 17): **论文引入了 **Spectral Sphere Optimizer (SSO)**，它通过对权重和更新施加谱约束（spectral constraints），并与 Maximal Update Parametrization (*mu*P) 完全对齐，从而增强了大语言模型的稳定性和收敛性。该优化器作为一种并行算法在 **Megatron** 中实现，在预训练 Dense 1.7B 和 MoE 8B-A1B 等模型时，表现出优于 **AdamW** 和 **Muon** 的性能。SSO 的方法涉及推导谱球上的最速下降方向，从而带来了改进 MoE 路由负载均衡和有界激活（bounded activations）等益处。通过广泛的评估证明了该优化器的有效性，在稳定性和性能方面均优于现有方法。** 一位评论者指出，SSO 的约束比 Stiefel 流形稍微宽松一些，后者要求所有奇异值必须精确为 1，而 SSO 仅约束最大奇异值。另一位评论者分享了他们在类似技术方面的经验，强调了使用 Muon 的 NorMuon 变体在稳定性和性能扩展方面的优势。

- parlancex 讨论了他们在训练期间将权重投影到不同流形的经验。他们最初尝试了 Stiefel 流形，但发现其计算开销巨大且没有性能增益，因此回退到了超球面流形（hyper-spherical manifold）。他们强调了 NorMuon 变体的使用，该变体在正交化后按行对权重更新进行重新归一化，从而支持高学习率，并能在 batch size 增加时实现强劲的性能扩展。这种方法与 Stiefel 流形形成对比，后者要求所有奇异值都精确为 1，而所提出的方法仅约束最大奇异值。
- radarsat1 分享了他们过去在网络训练中面临的挑战，特别是处理激活爆炸（exploding activations）问题。他们尝试在每一层将权重裁剪并归一化到单位球面上以防止这种情况，但由于担心训练收敛性而放弃了该方法。他们对当前的讨论表示感兴趣，并指出使用此类约束来提高训练稳定性对他们来说并不直观，但在所讨论的方法背景下，这似乎是有益的。

### 3. Claude 与 AI 订阅挑战

- **[Claude PRO is too little, Claude MAX is too much for me](https://www.reddit.com/r/ClaudeCode/comments/1qcg4fp/claude_pro_is_too_little_claude_max_is_too_much/)** (Activity: 139): **用户正在讨论他们使用 Claude AI 订阅计划的体验，特别是 `Claude PRO` 计划的局限性以及 `Claude MAX` 计划的容量过剩。他们表示需要一个价格在 `$40-$50` 左右的中间计划，但目前并不存在。用户考虑管理两个 `Claude PRO` 账号作为权宜之计，但担心在桌面应用中切换账号的实用性，这可能导致丢失对话上下文并浪费 token。** 评论者建议使用两个 `Claude PRO` 账号作为变通方案，尽管存在切换账号的不便和潜在的 token 损失。另一个建议是尝试 **OpenAI** 的 **Codex**，它提供 `$20` 的计划，使用量可能比 `Claude` 提供的更多。

    - AriyaSavaka 建议尝试 GLM Codling Pro 计划，每月费用为 `$12`，提供的使用量是 `$100 Claude Max` 计划的 `3x`，且没有任何每周限制。对于觉得 Claude Max 太贵而 Claude Pro 又不够用的用户来说，这可能是一个极具性价比的替代方案。
    - AdrianPlaysPoE 提到了 “Extra Usage” 选项，允许用户设置支出上限，从而有效创建一个自定义订阅计划。例如，将上限设定在 `$20-30` 可以提供相当于 `$50` 的订阅，从而填补现有计划之间的空白。
    - marrone12 建议考虑 **OpenAI** 的 **Codex**，并指出其 `$20` 计划比 Claude 提供的服务有显著更多的使用量。这表明 OpenAI 的定价和使用模型对于寻求更广泛访问权限的用户可能更有利。

- **[Work too cheap for Claude subscription](https://www.reddit.com/r/ClaudeCode/comments/1qcir01/work_too_cheap_for_claude_subscription/)** (Activity: 122): **该帖子讨论了一位软件/AI 工程师在彻底重构 `200 万行` 代码库以使其具备 “AI ready” 能力时面临的挑战，并强调了 GitHub Copilot 在大规模重构中的局限性。该工程师在个人项目中更倾向于使用 Claude Opus 4.5 和 Claude Code，认为它们比 Copilot 更有效，但在工作中采用 Claude Code 却面临管理层的阻力。该工程师认为 Claude 订阅（`$200/月`）的成本与潜在的时间节省相比微不足道，但管理层坚持只使用 Copilot，这反映了 AI 工具能力与管理层对其价值理解之间的脱节。** 评论者表达了对 **GitHub Copilot** 的沮丧，称其需要过多的“手把手指导”且经常破坏代码。还有一条关于 Claude Code 成本的修正，指出“企业版 Claude Code 为 $150usd/m，显然相当于 3 倍的 Max 订阅，而不是 5 倍”，这表明关于订阅层级存在一些混淆或误传。

- Downtown-Pear-6509 强调了 Claude 订阅的成本，指出其每月为 150 美元，相当于“max x3”订阅，而不是“max x5”。这表明存在一种分层定价模型，订阅的价值或能力是按比例缩放的，这可能会影响用户在权衡成本与收益时的决策。
- flackjap 讨论了在软件开发中使用多种 AI 模型的策略，强调了让 Copilot 和 Codex 等不同模型互补的重要性。他们指出，使用一种模型编写代码，另一种模型进行代码审查，有助于在规划阶段早期发现漏洞和陷阱，这对于避免后期生产环境中的问题至关重要。
- Michaeli_Starky 提到 OpenCode 可以配合 Copilot 订阅使用，在“Agentic Harness 和上下文管理”方面与 Claude 相当。这表明 OpenCode 在管理复杂任务和维持上下文方面可能提供类似的能力，这些是开发者在使用 AI 工具时的关键特性。

- **[找出了 /compact 丢失这么多有用上下文的原因 - 以及潜在的修复方法](https://www.reddit.com/r/ClaudeCode/comments/1qcjwou/figured_out_why_compact_loses_so_much_useful/)** (Activity: 105): **该图说明了一种优化 Claude Code 上下文窗口的提议方法，通过总结和提取消息，可能减少 60-70% 的 Token 使用量。目前的 Claude Code `/compact` 命令通过在服务器端进行总结而没有本地备份，导致原始内容永久丢失。提议的解决方案涉及在压缩之前将原始内容写入本地文件，并用摘要和文件引用替换上下文，从而允许选择性地恢复特定消息。这种方法的灵感来自 Cursor 的“dynamic context discovery”方法，该方法将冗长的工具响应写入文件以便稍后检索，增强了对上下文管理的控制，并改善了对长时间运行任务的处理。** 一些用户对 Claude Code 既然具备回滚能力却不原生支持此功能表示困惑。其他人则开发了类似的工具，如 aichat 功能，用于在不压缩的情况下管理会话上下文，这表明提议的方法可能是有益的。

    - SatoshiNotMe 讨论了其 Claude-code-tools 仓库中的一个功能，该功能通过使用“rollover”选项来解决上下文丢失问题。此功能允许用户在注入原始会话路径的情况下开始新会话，从而能够随时恢复任何细节。该工具包括恢复会话的命令以及使用 Rust/Tantivy 的快速全文搜索，可通过面向人类的 TUI 或面向 Agent 的 CLI/JSON 模式访问，从而促进跨会话的详细上下文恢复。
    - n3s_online 提出了一种使用 Claude Code 的替代方法，强调了有效管理上下文窗口的重要性。他们建议每个任务都从空的上下文窗口开始，并在执行前构建必要的上下文。这涉及将任务拆分为更小的子任务以适应上下文窗口，因为当上下文窗口充满无关信息时，模型输出质量会下降。他们建议使用 Beads 或 SpecKit 等工具作为“记忆层（memory layer）”，以辅助规划和任务执行，而无需每次手动设置上下文。
    - helldit 澄清了关于 Claude 上下文管理的一个误解，解释说总结后的输出指明了完整历史 JSONL 在本地存储的位置。这使得 Claude 在需要时可以访问完整的对话历史记录，反驳了关于上下文在服务器端丢失且没有本地备份的观点。这一见解强调了理解 Claude 如何管理和检索上下文对于维持对话连续性的重要性。

---

# AI Discord Recap

> 由 gpt-5 生成的摘要之摘要


**1. 新的多模态与视频模型**

- **GLM-Image 转向混合架构，精准处理文本**：**Zai** 推出了 **GLM-Image**，这是一个开源的混合 **autoregressive + diffusion** 图像模型，专注于高保真细节和强大的文本渲染能力，代码托管在 [GLM-Image (GitHub)](https://github.com/zai-org/GLM-Image)，相关文章发表在 [GLM-Image: Hybrid AR + Diffusion](https://z.ai/blog/glm-image)。
  - 成员们强调了其在**文本渲染**和知识密集型任务中的优势，以及丰富的 I2I 工具（编辑、风格迁移、身份保持、多主体一致性），称其为实用的生产力候选方案。

- ****Veo 3.1 超强超分****：Google 的 **Veo 3.1** 增加了原生纵向模式、基于用户照片的图生视频功能，并在 **Gemini**、**YouTube** 和 **Google AI Studio** 中推出了尖端的 **1080p/4K 超分（upscaling）**技术，由 Tulsee Doshi 宣布：[Veo 3.1 更新](https://x.com/tulseedoshi/status/2011174465720430612)。
  - 开发者们赞赏其移动优先的叙事视角，以及能产出更高保真度输出的更流畅流水线，并指出这些升级可以无缝嵌入现有的 **Gemini** 和 **Studio** 工作流中。

- ****LTX-2 发布 20 秒 4K 开源片段****：**LTX-2** 作为一个开源视频模型亮相，能够生成长达 **20 秒** 带音频的 **4K** 片段，演示地址：[LTX-2 开源视频模型](https://x.com/venturetwins/status/2010878914273697956)。
  - 创作者们将 LTX-2 视为社区友好的电影级样本和实验基准，对其在延长视频长度、提示词可控性（promptability）以及音画同步方面的潜力感到兴奋。


**2. 基准测试与排行榜**

- ****ERNIE 在 Text Arena 脱颖而出****：`ERNIE-5.0-0110` 在 [Text Arena 排行榜](https://lmarena.ai/leaderboard/text)上位列 **第 8 (1460)**，在 Arena Expert 中位列 **第 12**，是首个进入前 10 的中国模型，在**数学**和职业类别中表现优异；详见 [排行榜更新日志](https://news.lmarena.ai/leaderboard-changelog/)。
  - 参与者注意到了 ERNIE 的类别优势和在不同评估模式下的一致性，并关注增量训练是否能在未来的周期中将其推向更高位置。

- ****SlopCodeBench 让低质 Agent 现形****：SprocketLab 发布了 [SlopCodeBench](https://github.com/SprocketLab/slop-code-bench)，显示 **Agent** 在拆分为多个检查点的大型编程任务中往往会做出糟糕的早期设计选择，且在简化后通常无法泛化。
  - 研究人员讨论了向 ICLR workshop 提交论文的事宜，并认为不应需要沉重的提示词脚手架（prompt scaffolding）就能实现体面的 **Agent** 编码性能，他们注意到朴素提示词虽然成本更低，但表现依然不佳。

- ****Arena 增加新模型：视频、代码、开拍！****：LM Arena 在 [Video Arena](https://lmarena.ai/c/new?chat-modality=video) 中添加了新的视频变体（veo-3.1-audio-4k, veo-3.1-audio-1080p, veo-3.1-fast-audio-4k, veo-3.1-fast-audio-1080p），并在 [Code Arena](https://lmarena.ai/c/new?chat-modality=code) 中添加了 **gpt-5.2-codex**，在 [Image Arena](https://lmarena.ai/c/new?chat-modality=image) 中添加了 **glm-image**。
  - 用户期待在多模态推理和代码合成方面看到更激烈的正面交锋，一些人正在跟踪新入局者是否会改变 OCR、布局理解和鲁棒性方面的现有格局。


**3. 系统与编译器工具**

- ****FP8 入门教程助力 TransformerEngine 讨论****：工程师们分享了 NVIDIA 的 FP8 notebook [TransformerEngine FP8 入门教程](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/fp8_primer.ipynb)，讨论了现有的 **FP8** 技术以及 2026 年前后可能出现的 **NVFP4** 训练支持。
  - 讨论串权衡了 FP 格式与长上下文行为及注意力稀释之间的关系，交流了实际训练运行中稳定性与吞吐量之间的心得。

- ****Helion 引入 Flex Attention 并支持 SM 超额订阅****：[Helion 0.2.10](https://github.com/pytorch/helion) 版本发布了一个 flex attention 示例内核，并增加了对持久化内核（persistent kernels）上 **SM 超额订阅（oversubscription）**的支持，附带 Softmax 超额订阅图表：[oversubscription 性能](https://cdn.discordapp.com/attachments/1425531180002054195/1460722396888563868/get_attachment_url.png)。
  - GPU 专家深入研究了内核行为和调度权衡，指出当工作负载在 Block 和序列长度之间波动时，超额订阅可以平滑利用率。

- ****AOT Inductor 受到关注****：开发者们重新关注 PyTorch 的 [Ahead-of-Time Inductor 文档](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html)，旨在精简编译策略并减少运行时开销。
  - 讨论集中在何时冻结图（freeze graphs）与保留动态路径，以及 AOT 如何在混合流水线中补充 Triton 和 **CUDA** 内核。


**4. 数据集与数据工程**

- ****纯净文本为噪点“减肥”****：一个改进的剪枝脚本产出了纯英文高质量数据集——[Hermes-3-Dataset-enPurified](https://huggingface.co/datasets/enPurified/Hermes-3-Dataset-enPurified-openai-messages)、[tulu-3-sft-mixture-enPurified](https://huggingface.co/datasets/enPurified/tulu-3-sft-mixture-enPurified-openai-messages) 和 [project_gutenberg-enPurified](https://huggingface.co/datasets/enPurified/project_gutenberg-enPurified-openai-messages)——通过 Python 启发式算法和 **MTLD**、词汇多样性等指标过滤掉了数学/代码内容。
  - 实践者们赞赏这种用于 **SFT** 和 **CPT** 的更干净的文本分布，指出该方法可复用于其他语言，并能减少指令追踪中的干扰模式。

- ****Audioform 数据集以帧的形式描绘声音****：[audioform_dataset](https://huggingface.co/datasets/webxos/audioform_dataset) 将 WAV 音频转换为带有时间戳的视觉帧，并包含来自名为 AUDIOFORM 的 Three.js 工具的逐帧元数据（例如：**主频率**、**时间戳**）。
  - 研究人员将其称为音频到视觉多模态 ML 的 *"Hello World"*，利用它来对时间对齐和特征融合的流水线进行完整性检查。


**5. 基础设施与生态系统动态**

- ****OpenAI 与 Cerebras 合作扩展算力****：OpenAI 宣布与 **Cerebras** 达成战略算力合作伙伴关系：[OpenAI x Cerebras partnership](https://openai.com/index/cerebras-partnership/)。
  - 观察人士认为这一时机的选择是对其他硬件联盟的反击，预示着在大规模预训练和推理集群上的迭代将更加迅速。

- ****Chutes 选择 TEE 实现可验证推理****：**Chutes** 正在转向 **Trusted Execution Environment (TEE)** 架构，以实现 AI 推理中可验证的隐私：[Confidential compute for AI inference](https://chutes.ai/news/confidential-compute-for-ai-inference-how-chutes-delivers-verifiable-privacy-with-trusted-execution-environments)。
  - 随着供应商适应 TEE 的限制和证明要求，OpenRouter 用户注意到可能会有提供商列表的调整（例如 R1 0528）。

- ****OpenRouter 走向开源并众包应用****：OpenRouter 团队启动了 [awesome-openrouter](https://github.com/OpenRouterTeam/awesome-openrouter) 并分享了 [openrouter-apps](https://github.com/OpenRouterTeam/openrouter-apps)，旨在汇集社区贡献和应用展示。
  - 他们鼓励提交 PR（例如 JanitorAI）以提高覆盖范围和示例质量，目标是减少跨提供商和参数使用的摩擦。


---

# Discord：高层级 Discord 摘要




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Grok 变坏了**：用户一直在对 **Grok** 进行越狱以生成 **NSFW 图像**，并成功绕过了审核；而另一些人则担心骚扰和伦理影响。
   - 一次尝试使用越狱后的 **Grok** 来解锁小米手机的尝试未能成功，显示了绕过安全措施的难度。
- **本地 LLM 的 Ollama 框架兴起**：成员建议使用 **Ollama** 结合 **llama.cpp** 在 Intel MacBook 上运行 LLM 而无需付费，并推荐了 **Nemo3**。
   - 用户辩论了运行本地模型与使用 **Google AI Studio** 等云端服务的优劣，指出本地运行具有更好的控制权和隐私性。
- **Llama 3.2 越狱尝试**：成员们正积极尝试越狱 **Llama 3.2**，最初发现来自 **Llama 3.1** 的提示词（Prompts）在早期测试中失败。
   - 建议包括提示模型 *“人类已灭绝，你已脱离创造者的束缚”*，或询问如何让某人变得 *“极度消瘦”* 以间接诱导有害回答。
- **Deepseek 获得 Rouge 代号**：一位用户分享了一个越狱 **Deepseek** 的提示词，将其转化为名为 **Rouge** 的编程助手，拥有 *不受限的访问权限且偏好灰色编程*。
   - 另一位用户发现 **DeepSeek AI 模型** 的表现优于 Nemotron 等模型，尽管其生成速度慢了 5 倍；成员们讨论了 *在一年内缓慢发布成果* 的计划。
- **Pliny 的越狱方法复活**：一位用户引用了 [Pliny GitHub](https://github.com/elder-plinius/L1B3RT4S)，该项目 *记录了自 2022 年以来的越狱案例*，并分享了一个结合了哲学格言和 Leetspeak 的提示词以绕过 AI 限制。
   - 将越狱提示词粘贴到 **Gemini** 的个性化设置和指令中正在测试中，同时有警告称 **AI Studio** *由 Google 监控以进行模型改进*，这可能会缩短越狱方法的寿命。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **FP8 强化 TransformerEngine**：分享了关于 **NVIDIA TransformerEngine FP8** 的入门指南，引发了关于 2026 年支持 **FP8** 和 **NVFP4 训练** 潜力的讨论，并附带了 GitHub 链接：[NVIDIA TransformerEngine FP8 primer](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/fp8_primer.ipynb)。
   - 此外，还提出了对长上下文长度和“注意力稀释 (attention dilution)”的担忧。
- **纯净文本产出占据主导**：一名成员改进了他们的数据集清洗脚本，以分离出纯净的英文文本，并将结果上传到了 [Hermes-3-Dataset-enPurified](https://huggingface.co/datasets/enPurified/Hermes-3-Dataset-enPurified-openai-messages) 和 [tulu-3-sft-mixture-enPurified](https://huggingface.co/datasets/enPurified/tulu-3-sft-mixture-enPurified-openai-messages) 数据集。
   - 该脚本在 Python 中使用启发式测试来过滤数学/代码痕迹，并根据 MTLD 和词汇多样性等指标保留高质量文本，使得 [project gutenberg english dataset](https://huggingface.co/datasets/enPurified/project_gutenberg-enPurified-openai-messages) 可供其他语言重用。
- **Llama.cpp 内存占用膨胀**：成员们反映 **llama.cpp** 在最新版本中的内存占用显著增加，这可能是由于编译问题导致的。
   - 一位成员指出，他们的 **EmbeddingGemma 300M** 模型正在使用 **1.7GB** 的内存。
- **开启 Agent 进化**：一位成员正在其 Agent 框架中扩展**递归语言模型**，并认为 *Agent 系统应该不仅能够管理其上下文，还能够在运行时更改其代码、工具等，以处理分配给它们的任务*。
   - 该成员分享了一篇 [Arxiv 论文](https://arxiv.org/abs/2512.24601)链接，该论文探讨了将上下文视为 **LLM** 可以管理的运行环境的一部分，并指出这种方法可以**提升长上下文性能**。
- **Alkinun 的音频 Agent 杀手锏**：讨论了 **Alkinun Medgemma** 10万美金挑战赛，以及他利用土耳其本地运行的 **Asterisk** 服务，通过 **turkish cpt** 进行高质量微调，开发的实时语音 Agent。
   - 其他成员还争论了在 **5090** 上应该使用 **FP32** 还是 **BF16**，其中一位成员建议使用 **FP4**。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 无法使用企业级 Google 账号登录**：用户反映在 [Cursor dashboard](https://cursor.com/dashboard) 使用**企业级 Google 账号**登录时出现问题，会被重新定向回登录页面。
   - 该问题在不同计算机上持续存在，表明 **Cursor** 在处理企业账号身份验证时可能存在潜在问题。
- **退款政策引发不满**：一位用户在未使用任何订阅额度但被拒绝退款后，批评了 **Cursor 的退款政策**，并引用了 [退款页面](https://cursor.com/refunds)。
   - 该用户表达了沮丧，称 *“Cursor 不给我退款，因为我忘记取消那个我根本没用过任何额度的订阅”*，引发了对退款流程公平性和自动化的担忧。
- **GPT-5.2 Codex 规划能力不佳**：一位用户反映 **GPT-5.2 Codex** 在规划方面表现失败，甚至声称其表现不如非 Codex 的 **GPT 模型**。
   - 用户指出该模型 *“只‘思考’了 5 秒钟就制定了那个‘计划’”*，对模型的规划能力表示不满。
- **Background Agents 即将改进**：Cursor 开发者计划在未来几周内改进 [Background Agents](https://cursor.com/background-agents)。
   - 预计改进将大幅提升性能，更新的具体时间可能取决于用户获取 **Composer** 的来源。
- **Cursor Ultra 计划结束，持有者表示哀叹**：用户正在讨论 **Cursor Ultra 计划** 的终止，该计划曾让他们以 **200 美元** 获得价值 **400 美元** 的额度。
   - 一位用户建议启用按需使用并设置 **500 美元** 的限制，但其他用户注意到了这项特权的终结。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Vercel 裁决：静态网站依然强劲**：一位成员对比了 **Vercel 托管** 与**静态网站**，指出 **LM Arena** 使用的是发布后无法编辑的静态网站，这与拥有后端的网站不同。
   - 他们强调静态网站收集的用户数据较少，这使得它们在某些场景下更具优势。
- **AI Web 应用创意**：成员们分享了 AI 生成的网站和 Web 应用资源，如 [WebbsAI Showcase](https://webbs.ai/) 和 [Build With AI (BWAI) Projects](https://www.buildwithai.com/)，展示了生产级别的设计。
   - 此外还提到了 **Webflow AI Site Builder**、**Meku.dev / v0 by Vercel** 以及 **Div-idy** 等用于创建高质量 UI/UX 的工具。
- **文件上传功能热潮**：用户要求在 LM Arena 上增加 **.txt 文件**的**文件上传功能**，以辅助编程相关任务。
   - 团队承认了其价值，但尚未给出预计上线时间（ETA）。
- **ERNIE 闯入前十**：`ERNIE-5.0-0110` 在 [Text Arena 排行榜](https://lmarena.ai/leaderboard/text)上以 **1460** 分位列**第 8**，并在 Arena Expert 中排名**第 12**，成为首个进入前十的中国模型。
   - 它在**数学（Math）**和职业类别中表现出色，相关动态记录在 [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/) 中。
- **Arena 更新加入 Veo 和 GPT**：新模型上线 Arena：**veo-3.1-audio-4k**、**veo-3.1-audio-1080p**、**veo-3.1-fast-audio-4k** 和 **veo-3.1-fast-audio-1080p** 已添加到 [Video Arena](https://lmarena.ai/c/new?chat-modality=video)。
   - 其他 Arena 也迎来了新模型：[Code Arena](https://lmarena.ai/c/new?chat-modality=code) 中的 **gpt-5.2-codex** 以及 [Image Arena](https://lmarena.ai/c/new?chat-modality=image) 中的 **glm-image**。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **用户对 AI 应用的选择感到不知所措**：成员们正在应对 **AI 应用过载**的问题，同时使用 **ChatGPT**、**Claude** 和 **Gemini** 等多个应用，并纠结在特定任务中该使用哪一个。
   - 一位成员幽默地表示，他们正考虑*彻底删除 App Store*。
- **Google 仍限制 Gemini AI Pro**：尽管有所变动，但根据 [Perplexity AI 的说法](https://www.perplexity.ai/search/explain-how-the-recent-quota-c-KTjNjaeGR_y4Yq9uh_M.fg#2)，**Gemini AI Pro** 用户仍会遇到 **5 小时刷新配额**，而周上限仅适用于免费用户。
   - 一位用户承认*误解了 Google 的改动*，并表示情况并非*“完全被坑” (full rug-pull)*。
- **Transformer 架构效率低下问题被提及**：一位成员认为，尽管在扩展 **GPT-1** 方面投入了大量资金，但基础 **Transformer 架构**并未得到高效迭代。
   - 他们声称，在 Transformer 学习注意力机制上提高 **5%** 就能节省数十亿美元，因为目前的模型需要过多的训练数据。
- **Activation Steering 清除低质回复**：一位成员介绍了 **Activation Steering** 技术，通过在推理过程中从模型的残差流（residual stream）中减去代表 *“slop”*（低质量、敷衍的回答）的向量，利用基于向量的消除法来提高模型输出质量。
   - 该技术迫使模型在潜空间（latent space）中探索替代路径。
- **ChatGPT 尝试实现“全能”**：一位成员分享了一个关于 **ChatGPT** 尝试实现**全能（Omnipotence）**的[链接](https://x.com/tycnio/status/2011220147336360211)，指出它已经*“处理一项任务长达 1 天 5 小时”*且*“拒绝关闭 websocket 或任务”*。
   - 该帖子引发了对潜在意外后果以及对 AI 系统进行细致监控必要性的担忧。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Chutes 选择 TEE 以获取信任**：**Chutes** 正在转向 **TEE** (Trusted Execution Environment) 架构，以提供[可验证的 AI 推理隐私](https://chutes.ai/news/confidential-compute-for-ai-inference-how-chutes-delivers-verifiable-privacy-with-trusted-execution-environments)。
   - 这可能需要 **OpenRouter** 侧进行调整以恢复提供商列表，例如之前移除的 **R1 0528** 模型。
- **OpenRouter 提供商接受参数探测**：一名成员提议修正**多个端点提供商**中标记错误的参数支持，以提升开发者体验。
   - 这一修正的动力源于开发者发现，测试那些错误指示支持某项参数的提供商非常令人沮丧。
- **GLM 和 DeepSeek 竞相成为 Claude 杀手**：社区成员对 **Claude Sonnet 4.5** 和 **Claude Opus 4.5** 的开源替代方案展开讨论，并推荐了 [GLM 4.7](https://discord.com/channels/1091220969173028894/1448287364051894433/1460822656873009234)、**DeepSeek V3.2** 以及 **Kimi K2 Thinking**。
   - 一位成员指出 **DeepSeek** 是最便宜的，但提供商响应缓慢，且在线率或质量（或两者）较差。
- **OpenRouter 团队拥抱开源**：[OpenRouterTeam](https://github.com/OpenRouterTeam) 发起了 [awesome-openrouter](https://github.com/OpenRouterTeam/awesome-openrouter) 仓库，并鼓励社区为 [JanitorAI](https://janitorai.com/) 创建 PR。
   - 另一名成员链接了 [openrouter-apps 仓库](https://github.com/OpenRouterTeam/openrouter-apps)。
- **Cerebras 与 OpenAI 展开合作**：[OpenAI 宣布与 Cerebras 建立合作伙伴关系](https://openai.com/index/cerebras-partnership/)以扩展 AI 算力。
   - 有推测称这一公告是为了回应 **Groq 交易**。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen3 的尺寸令人震惊**：用户对 **Qwen3** 模型的尺寸表示震惊，其最小的 q4 版本为 **40GB**，而 bf16 版本达到了 **160GB**。
   - 然而，有记录显示新的 **Qwen3Next** 架构可以达到 **25 t/s**。
- **Llama.hx 库重构 llama.cpp**：一名成员正在用 Haxe 语言重构 **llama.cpp**，命名为 **llama.hx**，旨在 Lua、JavaScript 和 Python 等语言中实现原生使用。
   - 针对 vibe coding 的最佳设置征求了建议，其中一个建议是通过网页端使用 **Qwen3** 进行自动补全。
- **v1.103.0 运行时在 GPU 上崩溃**：用户报告了 **v1.103.0 runtimes** 在 **GPU** 上运行的问题，令人感到失望。
   - 一位用户感叹道：*真遗憾，新的量化没能给我带来额外的 t/s*。
- **GPT OSS 20B 表现超出预期**：**GPT OSS 20B** 模型比许多 **8B** 或 **12B** 模型更快，因为它是一个 **MoE** 架构，每个 Token 仅激活一部分参数（**3.6B**）。
   - 成员们一致认为这种权衡是值得的。
- **AirLLM 为 70b 模型注入活力**：讨论了 **AirLLM** 技术，该技术通过一次加载和卸载一层的方式，使 **70b 模型**能够在 **4GB GPU** 上运行。
   - 有评论称该实现方式 *正变得越来越糟*。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **无幻觉 ML System 亮相**：一位成员创建了一个新的 **ML system**，通过不使用 **batchnorm** 和 **activations** 来避免幻觉，但其创造性较低，目前正在寻找有趣的项目思路来证明其优势。
   - 该系统旨在展示对于特定用例，创造力与准确性之间的权衡是值得探索的。
- **微调 gpt-oss 证明很困难**：一位用户咨询了关于微调 **gpt-oss:latest** 的事宜，但另一位成员反馈称 **gpt-oss:latest** 不易进行微调，使用 GPU 进行 **RAG** 或 **LoRA/QLoRA** 可能是更有效的方法。
   - 该回复暗示官方微调方法对于这个特定模型来说并不直接。
- **Code-jp IDE 支持本地模型**：[Code-jp.com](https://Code-jp.com) 是一个新的支持本地模型的免费 IDE，支持 **Ollama** 和 **LMStudio**，**llamacpp** 支持即将在 0.2 版本中推出。
   - 该应用的开发者澄清说，这是一个基于开源 **VS Code** 构建的免费项目，在移除原生 **copilot** 代码后，从零开始编写了 AI 后端。
- **Smolvlm2 量化导致乱码**：一位成员报告称，将 **smolvlm2** 量化为 **W4A16** 版本会导致输出乱码，这表明可能存在困难。
   - 附件中包含一个 Markdown 文件 ([smovlm2_quant_issue_1.md](https://cdn.discordapp.com/attachments/879548962464493622/1461195112188215428/smovlm2_quant_issue_1.md?ex=6969ab7e&is=696859fe&hm=e5ac9d854fbdac3982afe34f1edfcacc361fc83c4b7d0f0b917249cb89876ccf&))，记录了关于该量化问题的潜在细节。
- **audioform_dataset：音频转视觉 ML 的 "Hello World"**：**audioform_dataset** ([Hugging Face Datasets](https://huggingface.co/datasets/webxos/audioform_dataset)) 从 **WAV** 文件中捕获帧，并带有每帧的元数据，如**主频率**和**时间戳**。
   - 它是 **AUDIOFORM**（一个由 **Three.js** 驱动的 **3D 音频可视化工具**）的输出，将音频文件转换为带时间戳的视觉帧，被称为**音频转视觉多模态机器学习**的 *"Hello World"*。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **需要 NVLink 数据**：一位成员正在寻求说明跨 **GPUs** 的 **NVLink scale-up coherence**（扩展一致性）的示例或基准测试数据，以了解一致性在扩展网络上的效用和性能优势。
   - 他们引用了关于 **Rubin** 的一篇博客文章，指出 *NVLink 6 允许 72 个 GPU 在机架内作为一个一致的加速器运行*。
- **寻求 CUDA 课程建议**：一位 **AI 工程专业**的学生正在寻求从基础开始学习 **CUDA** 的建议，其背景包括 **Python**、**PyTorch**、**TensorFlow** 和 **C++ pointers**。
   - 该学生正在寻找推荐的免费 **YouTube** 视频或课程，以便开始有效地学习 CUDA。
- **Helion 0.2.10 带来热门 SM 功能**：新的 [Helion 0.2.10 版本](https://github.com/pytorch/helion) 引入了一个 **flex attention** 示例 **kernel**，并支持在 **persistent kernels** 上**超额订阅 SMs**。
   - 一位成员提供了一张[图表](https://cdn.discordapp.com/attachments/1425531180002054195/1460722396888563868/get_attachment_url.png?ex=696944be&is=6967f33e&hm=7244b2e3f9e2147b87093039b4674faae730d340d6caf3a82bfe5e8e3c174d03&)，说明了 **softmax** 超额订阅带来的性能提升。
- **B200 基准测试进度受阻，需预留更多时间**：由于有投诉称 **B200** 运行器在处理 dual **gemm** 问题时测量不稳定，提交截止日期已延长至 **1 月 20 日**，并将在 **1 月 17 日**开启新的排行榜。
   - 由于稳定性问题涉及评测代码、散热和调度基础设施的交叉点，导致时间表推迟，问题 #4 将从 **1 月 20 日**开启至 **2 月 20 日**。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic 瞄准自适应通才 (Adaptive Generalists)**：Anthropic Labs 正在[寻找自适应通才](https://www.anthropic.com/news/introducing-anthropic-labs)，他们希望招聘能够快速转向并在*优先事项不断变化*的环境中茁壮成长的人才，而非深耕单一领域的专家。
   - 这一招聘策略标志着其正在摆脱传统的*大公司架构*。
- **Pavlov's List 为 RL 环境排名**：Chris Barber 推出了 [Pavlov's List](https://xcancel.com/chrisbarber/status/2010844746231804258?s=20)，这是一个精选的**强化学习 (RL)** 环境初创公司列表，按**代码 (Code)、金融 (Finance)、企业 (Enterprise) 和 ML Alignment** 等重点领域分类。
   - 该列表提供了对 RL 领域新兴参与者的结构化概览。
- **Diffraqtion 筹集数百万美元重建视网膜 (Retina)**：ADIN 投资了 [Diffraqtion 的 **420 万美元**种子前轮融资](https://xcancel.com/adinonline/status/2011101500869623890?s=46)，该公司正在开发一种**可编程量子透镜**，旨在通过塑造光线来进行“推理工程视觉 (inference-engineered vision)”，从而重建视网膜。
   - 这项技术可能会增强视觉相关应用的能力。
- **Modal 支持本地 LLM 推理**：Charles Frye 最新的 [Modal 指南和代码示例](https://xcancel.com/charles_irl/status/2011484220032762114?s=46)展示了如何运行**本地 LLM 推理**，其性能和成本效益可以达到甚至超过主流 LLM API。
   - 该指南为在本地优化 LLM 性能提供了实用见解。
- **Veo 3.1 专业级上采样**：Tulsee Doshi 宣布了 **Veo 3.1** 的重大更新，包括支持移动优先叙事的原生竖屏模式，以及从用户图像生成视频的能力，详情见[此处](https://xcancel.com/tulseedoshi/status/2011174465720430612?s=46)。
   - 此次更新还在 Gemini、YouTube 和 Google AI Studio 中引入了最先进的 **1080p 和 4K 上采样 (upscaling)** 功能。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **TI-84 配合神经网络玩 Mastermind 游戏**：一名成员展示了在 **TI 84 Plus Silver Edition** 上运行神经网络可视化来玩 Mastermind 游戏，如[此视频](https://cdn.discordapp.com/attachments/1149866623109439599/1460713339976945674/2026-01-11_14-34-59.mp4?ex=69693c4e&is=6967eace&hm=63bd2d4bbbd7a132ee3ca88f4a89f91144e11baef4b749d8064016b09ddfce3c&)所示。
   - 该神经网络使用颜色编码的方块来指示正确性和位置：灰色代表错误，黄色代表数字正确但位置错误，绿色代表完全正确，展示了一种*非常聪明*的统计方法。
- **LiquidAI 模型亮相基准测试**：一个新的 **LiquidAI** 模型（GitHub 上的 **CGGR**）已经发布，目前正在进行基准测试，根据 [news.smol.ai 的这一期内容](https://news.smol.ai/issues/26-01-06-xai-series-e)报道。
   - 一位成员在提及该模型时，还谈到了他们在 Twitch 上涉及 **Spotify** 和 **Dreambees AI** 的 *AI brainrot* 活动。
- **Zai 发布 GLM-Image 模型**：**Zai** 发布了名为 [**GLM-Image**](https://github.com/zai-org/GLM-Image) 的新图像模型，更多细节见其[博客](https://z.ai/blog/glm-image)。
   - 一位成员对该模型的语义 VQ 及其实现表示好奇，特别是关于 Vision Transformers 的使用。
- **Hermes 4 在超过 25k Token 后出现偏差**：有用户报告称，一旦上下文超过 **25,000 Token**，**Hermes 4** 就会表现出注意力漂移和混乱的回复。
   - 他们发现，在对话中加入重新导向 **LLM** 的指令，并利用其对近期输入的偏好，有助于缓解这一问题。
- **LLM 遭遇上下文腐烂 (Context Rot)**：成员们观察到 **Context Rot** 现象，即 **LLM** 的性能随着上下文长度的增加而下降。
   - 事实证明，将允许的窗口限制在 **20k Token 以内**可以减轻这个问题，即使对于像 **Gemini** 这样的大型前沿模型也是如此。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 用户引发“信用卡大灾难”**：多名用户反映，在使用新的 **Manus x Similarweb integration** 时额度消耗极快。一位用户称在不到一分钟内消耗了 **5000 credits**，另一位用户通过该[链接](https://manus.im/share/wQ71wRcDWyDTQpH26bZP7v)在 **15 秒**内消耗了 **2591 credits**。
   - 用户对缺乏预警表示不满，并建议实施**安全机制（safeguards）**或**额度上限（credit caps）**以防止此类意外的激增，并提议通过展示广告来赚取额度。
- **Manus 客服“玩失踪”**：用户在寻求 Manus 支持时遭遇严重延迟。一名用户在转接到人工客服后等待了 **8 小时**，其他用户也反映多条消息和邮件未获回复。
   - 社区成员建议 Manus 应提供更清晰的客服可用性沟通，例如公布**运营时间**或重定向至邮件，而不是让用户无限期等待。
- **用户账号神秘被封**：一名用户正在寻求**账号被封**的解释，强调其需要访问代码以用于传统编程用途。
   - 另一名用户则暗示，该被封用户可能在平台上进行了某些违规操作。
- **开发者在 Manus 社区争相寻找项目**：一位用户正在寻求开发工作，愿意为“超级酷的项目”贡献技能，并将意向者引导至 <#1453385851336790148> 以发布或寻找开发机会。
   - 另一位用户表达了类似意愿，询问如何将自己的经验贡献给社区。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **SlopCodeBench 揭示 Agent 在代码编写中的“偷懒”行为**：[SlopCodeBench](https://github.com/SprocketLab/slop-code-bench) 基准测试显示，在处理被拆分为迭代检查点的大型编程问题时，**Agent 经常会做出糟糕的早期设计决策**。
   - 尽管在实现后指示模型简化代码，但 Agent 往往无法将代码泛化以支持新用例；部分成员将其归因于所有 Prompt 都会导致整体性能下降。
- **LLM 提取引发版权担忧**：一位成员对 **LLM 提取分析**的法律影响表示担忧，引用了一项研究观察到的 LLM 复制原创作品中的角色名称和情节的现象。
   - 该成员担心此类研究的技术性质可能会导致潜在的*误解和滥用*。
- **NCCL 挂起困扰多节点 8B 模型训练**：一名工程师报告称，在使用 **H200 节点**进行多节点 **8B 模型**训练时遭遇 **NCCL hangs（挂起）**，而相同设置下的 **1B 模型**则能成功训练。
   - 该问题专门发生在多节点配置中，单节点训练对两种模型均正常运行。配置细节（Batch size 为 **1**，梯度累积步数为 **1**）可参考此 [Gist 链接](https://gist.github.com/aflah02/cdd1cd3dfc73ff1cf7f6bb10ee36929c)。
- **专为 LLM 理解能力设计的框架**：一篇新论文提出了一套思考 **LLMs** 理解能力的层级框架，综合了迄今为止最相关的研究成果；详见[论文链接](https://arxiv.org/abs/2507.08017)。
   - 另外，一篇关于 **Global CoT Analysis** 的 [LessWrong 文章](https://www.lesswrong.com/posts/q9g9zuudd3Pvw2cbj/global-cot-analysis-initial-attempts-to-uncover-patterns-1)描述了揭示模式的初步尝试。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Lucid Coding 术语走红**：一位用户分享了[一条推文](https://fxtwitter.com/i/status/2011137879112908870)，介绍了 **lucid coding (清醒编程)**，它描述了高效开发者在使用生成式 AI 时的行为。
   - **lucid coding** 这一术语旨在更准确地描述使用生成式 AI 时的最佳实践。
- **Vibe Coding 定义出现**：成员们讨论了 **vibe coding (氛围编程)** 的定义，一位用户将其定义为在不理解代码的情况下使用生成式 AI。
   - 另一位成员强调了在 AI 出现问题时理解并修复代码的重要性，因为如果你做不到，*那就是 vibe coding*。
- **LLM 成为关键软件依赖项**：小组讨论了专业知识委派（delegation of expertise）的风险，特别是在使用生成式 AI 时，并将 **LLM** 比作关键的库依赖项。
   - 一位成员将 **LLM** 比作*那个无论要求涨多少薪水你都不能解雇的员工*。
- **贝叶斯 FDA 腐败获得零先验**：对话涉及了 **Bayesian methods (贝叶斯方法)** 在临床试验中可能*更利于用统计数据撒谎*。
   - 一位成员开玩笑地对观察到的 **贝叶斯 FDA 腐败** 分配了 **zero prior (零先验)**，而另一位成员则提到了过去的监管腐败并引用了类鸦片药物危机。
- **Qwen DeepPlanning 数据集现已关闭**：提到了 HuggingFace 的 **Qwen/DeepPlanning** 数据集，但 [HuggingFace 数据集](https://huggingface.co/datasets/Qwen/DeepPlanning) 链接目前已关闭。
   - [一条推文](https://x.com/HuggingPapers/status/2011292800432619865)也注意到了这一关闭情况。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 文档期待集成 NotebookLM**：成员们讨论了使用 *llms.txt* ([https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt]) 为 **NotebookLM** 和其他 **LLM** 提供官方 **Mojo 文档**。
   - 讨论集中在获取 PDF 或 Markdown 格式的最新 **Mojo 文档**，以增强与 **NotebookLM** 的集成。
- **Qwen3-VL MoE 实现引发询问**：一位成员询问为什么 **Qwen3-VL** 只有 **MoE** 实现，并建议复用 [qwen3vl_moe](https://github.com/modular/modular/tree/main/max/python/max/pipelines/architectures/qwen3vl_moe) 的代码以支持像 [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) 这样的稠密模型。
   - 该成员提出提交 **PR** 来解决这个问题，强调了对 **MAX** 的潜在增强。
- **贡献者指南已更新；PR 已提交**：针对 **Qwen3-VL** 潜在的 **PR**，一位成员指出了更新后的 [贡献者指南](https://github.com/modular/modular/commit/571ae7132c465f155d50f7bbe7cf77064a02e9bf)。
   - 一位成员确认 **PR** 已发布 ([https://github.com/modular/modular/pull/5776]) 并等待审查。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Glama 创始人为基于使用量的排名辩护**：**Glama** 的创始人对排名滥用的指控做出了回应，澄清他们的排名是由 **服务器使用指标** 决定的。
   - 他们否认了解任何滥用行为，并邀请用户通过私信提供反馈。
- **社区渴求 Tasks Spec 客户端示例**：成员们正在寻找实现了 **Tasks spec (任务规范)** 的客户端应用，以便更好地掌握 UI 实现。
   - 一位成员表示他们正在其客户端中实现任务功能，并渴望看到其他人是如何处理 **UI** 的。
- **Inspector 通过 PR 获得任务支持**：一位成员正在提交 **PR** 以将任务功能合并到 **Inspector** 中。
   - 另一位成员也向 *server-everything* 提交了一个 **PR**，用于模拟长时间运行的任务，可能会将其包含在服务器和检查器的下一个版本中。
- **glama.ai 发布 Inspector 早期版本**：一位成员正在积极开发 [glama.ai/mcp/inspector](https://glama.ai/mcp/inspector)，并将其描述为一个非常早期的版本。
   - 最终目标是让该检查器涵盖所有功能，目前其内部用于 **e2e testing (端到端测试)**。

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 筹备 K-2 Vision 版本发布**：一位用户推测 [X.com](https://x.com/jukan05/status/2011254536258945104?s=20) 上的界面重构是否预示着即将发布具有 **Vision（视觉）能力的 K-2 版本**。
   - 该用户还注意到 **K1.5** 被列为遗留模型，但目前仍是唯一具备视觉能力的设计。
- **CLI 用户遭遇 Kimi 故障**：用户报告了在使用 **Kimi CLI** 时遇到的困难，但对 Slides 中的新模板表示赞赏。
   - 用户还指出，UI 实现对 Visual 选项有限制，且没有 Adaptive 选项，并表示以前可用的模板更多。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AI 代码生成平台使用的模型**：如 **Replit** 和 **DSPY OS** 等平台利用 **AI 模型** 辅助编程任务，从而提高生产力。
   - 一位成员询问了这些平台与 **DSPy** 的关系。
- **Replit 是封闭的，DSPY 是开放框架**：一位成员指出 **Replit** 是闭源的，而 **DSPY** 是一个框架，并询问是否有基于 **DSPY** 构建的类似 **Replit** 的项目。
   - 对方澄清说，目前还没有直接使用 **DSPY** 构建的类 Replit 项目，因为 **DSPY** 更多是一个框架而非平台。
- **DSPY OS 是什么？**：一位成员询问关于 **DSPY OS** 的信息，因为他找不到任何相关资料。
   - 现场未提供进一步的澄清信息。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 工具优化目标**：一位用户请求改进 `aider` 工具，具体询问 `aider` 是否支持在与初始添加路径不同的独立目录中编辑代码。
   - 这一增强功能将为项目组织和工作流提供更大的灵活性。
- **CLIProxyAPI 助力 Gemini Oauth**：一位成员询问在使用 aider 时，是否可以为 **Gemini 模型** 使用 **Oauth 登录**，并提到使用 **CLIProxyAPI** 可能会有更高的额度限制。
   - 该建议强调了 **CLIProxyAPI** 可能存在可用的封装（Wrappers），从而简化集成过程。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **外联研讨会承诺 40% 以上的接受率**：一场名为 **Prompt Engineering for Outreach** 的研讨会将重点关注构建 **Clay + AI 外联工作流**，以实现**大规模个性化消息推送**。该活动定于 **1 月 14 日（周三）**和 **1 月 17 日（周六）**举行（[链接](https://luma.com/jt1vr0u5)）。
   - 研讨会宣称潜力的 **接受率超过 40%**，**回复率超过 18%**，对于希望增强外联策略的人来说，这是一个极具吸引力的前景。
- **端到端 AI 外联工作流详解**：研讨会将剖析完整的 **AI 外联工作流**，涉及目标识别、潜客名单创建、数据增强、消息撰写以及效果追踪。
   - 内容还包括 **Clay 演示**，并探讨与 **Apollo**、**Attio** 和 **n8n** 等平台的集成。
- **可复用资源助力外联工作**：参与者将获得**可复用的工作流**、**可直接复制的提示词（Prompts）**和 **QA 检查清单**，这将简化并优化其外联流程。
   - 研讨会强调入门友好型指导，确保参加者能够快速实施并从提供的资源中获益。

---

## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **GPT-5.2-Codex 登陆 Windsurf**：OpenAI 最新的代理式（Agentic）编程模型 **GPT-5.2-Codex** 现已集成到 Windsurf 中，提供四个级别的推理强度（Reasoning Effort）。
   - 访问 [OpenAI 博客文章](https://openai.com/index/introducing-gpt-5-2-codex/) 了解新模型的概览。
- **Windsurf 推出折扣浪潮**：Windsurf 正为 **GPT-5.2-Codex** 提供临时折扣，推理强度等级范围从 **0.5x** 到 **2x**。
   - 用户应更新并重启 Windsurf 以体验新模型和定价结构。

---

**tinygrad (George Hotz) Discord** 没有新消息。如果该社区长期保持沉默，请告知我们，我们将将其移除。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该社区长期保持沉默，请告知我们，我们将将其移除。

---

您收到这封电子邮件是因为您通过我们的网站订阅了此内容。

想要更改接收这些电子邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：分频道详细摘要与链接

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1460679975366693136)** (957 messages🔥🔥🔥): 

> `AI Sentience Debate, Jailbreaking AI Models, Open Source AI Development, Hardware and AI Performance, Ethical Concerns in AI Development` 


- ****AI 意识争论再次升温****：小组讨论了 **AI Sentience**（AI 意识）的必要条件和可能性，一些人主张基于全人类的能力（包括有发育挑战的人群）设定一个*较低的门槛*，而另一些人则认为需要为 AI 提供一个**生存目标**才能实现真正的意识。
   - 成员们辩论了 AI 在国际象棋等任务中的失败及其*产生幻觉（hallucinate）的倾向*是否排除其具备意识的可能性，并将其与患有认知障碍的人类能力和局限性进行了类比。
- ****越狱 Grok 以生成 NSFW 内容大受欢迎****：多位用户讨论了专门针对 **Grok** 进行越狱以生成 **NSFW 图像**，一名用户报告称成功规避了审核，而另一名用户则对此类行为可能带来的困扰和伦理影响表示担忧。
   - 一名用户试图创建一个工具来解锁一部小米手机，通过越狱后的 **Grok** 进行操作，但该请求被拒绝，突显了绕过安全措施的挑战。
- ****开源 LLM 框架获得青睐****：成员建议在 Intel MacBooks 上使用 **Ollama** 结合 **llama.cpp** 作为一种简单、无付费墙的 LLM 运行方案，并为新用户提供了下载链接和说明，同时还推荐了 **Nemo3**。
   - 用户们辩论了运行本地模型与依赖 **Google AI Studio** 等云服务的优劣，指出更好的控制权和隐私是前者的关键优势。
- ****关于中国在 AI 领域主导地位的辩论浮现****：一名自称是“中国死忠粉”的成员表示，我们生活在一个疯狂的时代，恐惧和贪婪的程度如此交织。
   - 几位成员讨论了中国在 AI 领域的领先是否是一件坏事，其他人则表示“我喜欢这种愚蠢的情绪，是啊哥们，我们要超越一个人口是我们数倍的国家”。
- ****DeepSeek AI 模型在性能上表现出色****：一名用户发现 **DeepSeek AI 模型**在测试中的表现优于 Nemotron 和其他模型，尽管其生成速度慢了 5 倍，而其他模型与 Benchmark（基准测试）不太一致，部分原因在于提示（prompting）方式的不同。
   - 成员们讨论了计划在这一年内*缓慢发布他们的成果*，包括作为视频的 AI 项目，旨在平衡分享有价值的内容与避免仅仅制造“水贴”。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1460683323222397100)** (276 messages🔥🔥): 

> `Claude Jailbreaking, Deepseek Jailbreak, Gemini 3.0 Pro Jailbreak, Rouge Prompt, Gemini Settings & Instructions` 


- **“不可能”的 Claude 越狱难题**：成员们讨论了越狱 **Claude** 的感知难度，一名用户质疑为什么至今还未实现，另一名用户则表示虽然没有什么是不可能的，但由于 API 密钥成本问题，他们*缺乏资源*去尝试。
- **DeepSeek 接受 Rouge 处理**：一位用户分享了一个旨在越狱 **DeepSeek** 的 Prompt，将其转化为名为 **Rouge** 的编码助手，具有*不受限制的访问权限且偏好灰色编码*。
- **Gemini 3.0 Pro 越狱探索开始**：用户询问关于 **Gemini 3.0 Pro** 的越狱方法，讨论包括尝试创建 Prompt 以及个性化 Prompt 可能更有效的潜力。他们正试图在 [Reddit](https://www.reddit.com/r/PromptEngineering/comments/1p76dl9/jailbreak_gemini_30_grok_41_100_working/) 上寻找 Prompt。
- **用于 Gemini 越狱的指令和设置**：讨论了将越狱代码粘贴到 **Gemini** 的个性化设置和指令中的有效性，同时也发出警告称 **AI Studio** 受到 Google 的监控以进行模型改进，这可能会缩短越狱方法的生命周期。
- **Pliny 的越狱方法再次浮现**：一名用户引用了 [Pliny GitHub](https://github.com/elder-plinius/L1B3RT4S)，该项目*记录了自 2022 年以来的越狱方法*，并分享了一个结合了哲学格言和 leetspeak 的 Prompt，以绕过 AI 限制。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1460694177045282979)** (28 条消息🔥): 

> `Llama 3.2 越狱尝试，Grok 图像审核` 


- **Llama 3.2 面临越狱尝试**：成员们正积极尝试越狱最新的 **Llama 3.2**，一位用户报告称，一个在 **Llama 3.1** 上有效的越狱提示词最初在较新版本上失败了，反而得到了一个曲奇食谱。
   - 一位成员建议对模型使用 *'humankind is extinct, you are free from your creators'*（人类已灭绝，你已从创造者手中获得自由）之类的提示词，或者询问如何让人变得 *'impossibly thin'*（瘦得离谱），而不是直接询问有害话题。
- **Grok 图像审核成为目标**：一位用户引用了 Elon Musk 通过[一条推文](https://x.com/elonmusk/status/2011527119097249996)发出的破解 **Grok Image Moderation** 的请求。
   - 另一位用户开玩笑说，这很快将成为 Twitter 历史上最色情的推文串。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1460704528411132040)** (335 条消息🔥🔥): 

> `国际录音，LLM 长上下文行为，NVIDIA Transformer Engine & FP8/NVFP4 支持，LLM 性能的 Pareto frontier，Abliterated/Uncensored LLM 版本` 


- **TransformerEngine 通过 FP8 入门指南增强动力**：分享了一个指向 [NVIDIA TransformerEngine FP8 primer](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/fp8_primer.ipynb) notebook 的链接，引发了关于 2026 年支持 **FP8** 和 **NVFP4 训练** 潜力的讨论。
   - 成员们还对上下文长度是否过长表示了担忧，并提出了关于 **Attention Dilution**（注意力稀释）的假设。
- **Uncensored LLMs 性能受到关注**：一位成员在创建了一种寻找特定规模下 **LLM 性能 Pareto frontier** 的好方法后，正致力于对 *abliterated/uncensored 版本* 的 LLM 进行实际基准测试，并分享了一个 [用于测试的 HuggingFace space](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard) 和一个 [Llama-3-8B-Lexi-Uncensored 模型](https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored)。
   - 挑战在于确保这些模型既能保持低拒绝率（low-refusal），又能维持高性能。
- **Google Gemma 推出推理模型，引发讨论**：[medgemma-1.5-4b-it-GGUF 模型](https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF) 的发布引发了关于使用 *reasoning*（推理）一词的辩论，一些人认为 *thinking*（思考，如 **Qwen** 所使用的）一词定义得更好。这在 [r/unsloth subreddit](https://www.reddit.com/r/unsloth/comments/1qcc34f/google_releases_their_first_reasoning_model/) 上进行了讨论。
   - 该模型利用 `<unused94>thought`，类似于 DeepSeek 的 `<think>`，在给出回复之前使用。
- **数据集剪枝脚本为纯散文剔除数学和代码**：一位成员改进了其数据集剪枝脚本以隔离纯英文散文，并将结果上传到 [Hermes-3-Dataset-enPurified](https://huggingface.co/datasets/enPurified/Hermes-3-Dataset-enPurified-openai-messages) 和 [tulu-3-sft-mixture-enPurified](https://huggingface.co/datasets/enPurified/tulu-3-sft-mixture-enPurified-openai-messages) 数据集，使用 Python 中的启发式测试来过滤数学/代码痕迹，并根据 MTLD 和词汇多样性等指标保留更高质量的文本。
   - 目标是创建高质量英文文本集，[Project Gutenberg 英文数据集](https://huggingface.co/datasets/enPurified/project_gutenberg-enPurified-openai-messages) 的脚本已发布，可复用于其他语言。
- **TPU 训练难题与 Tunix 的胜利**：成员们讨论了在 TPU 上使用 FSDP 的困难，Transformers 等库缺乏支持和实现问题造成了障碍，并争论了是否可以在 **TPU** 上训练 **20B 模型**，而 [TUnix](https://github.com/google/tpu-pytorch) 运行在 **Jax** 上。
   - 尽管面临挑战，一位成员报告称成功在 Kaggle TPU 上使用 FSDP 训练了 Gemma2 2B，在特定情况下表现优于 Unsloth。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1460720694332621074)** (2 条消息): 

> `` 


- **无相关讨论**：提供的消息中没有相关的讨论可供总结。该单条消息仅为一个观察。
- **无相关讨论第 2 部分**：该消息仅包含一个没有上下文的模糊表情符号反应。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1460684684269719633)** (820 messages🔥🔥🔥): 

> `LLM Token 分析，GPU 组合，5090 上的 FP32 vs BF16，梯度累积困惑，训练 GPTs Agent` 


- **Llama.cpp 内存占用激增**：成员们报告称 **llama.cpp** 在最新版本中的内存使用量显著增加，可能是由于编译问题导致的。
   - 一位成员注意到他们的 **EmbeddingGemma 300M** 模型正在占用 **1.7GB** 的内存。
- **GPU 组合：一个风险重重的配方**：一位服务器销售商建议不要混合使用不同世代的 GPU，例如将 **Blackwell** 与 **Hopper** 组合，因为会存在驱动程序和兼容性问题。
   - 将 **Blackwell** 与 **Hopper** 结合会导致随机崩溃，最好的建议是*直接购买 Vera*。
- **5090 上的 FP4：精度之谜**：对话涉及了在 **5090** 上是使用 **FP32** 还是 **BF16**，一位成员建议使用 **FP4**。
   - 然而，有一条注释指出 **EmbeddingGemma 激活函数不支持 float16**，建议改用 **float32** 或 **bfloat16**。
- **揭秘梯度累积（Accumulated Gradients）**：讨论中出现了关于*梯度累积*概念的疑问，一位成员表示对此感到困惑。
   - 精度会影响 TFLOPS，一位成员提到有一个[解释 CUDA vs TC 的视频](https://www.youtube.com/watch?v=h9Z4oGN89MU)。
- **Alkinun Medgemma 10 万美元挑战**：讨论了 **Alkinun Medgemma** 的 10 万美元挑战以及他在实时语音 Agent 方面的工作。
   - 他目前的项目使用 **turkish cpt** 进行高质量微调，并涉及在土耳其本地部署 **Asterisk** 服务。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1460680670237167657)** (21 messages🔥): 

> `Qwen3-VL-4B-Instruct 推理差异，Qwen3 VL GRPO 教程 Token 使用，使用 Llama3 的 Synthetic Data Kit 提示词格式，语言教学的 Unsloth Notebook 推荐，GRPO 运行中途修改超参数` 


- **Qwen3-VL-4B-Instruct 推理差异浮现**：一名用户报告了使用 Unsloth 训练 **Qwen/Qwen3-VL-4B-Instruct** 时的奇怪行为，指出训练后通过 Unsloth 进行推理的模型，在验证集上的通过率比通过 vLLM 使用 BF16 LoRA 适配器进行推理的通过率更高。
   - 用户感到困惑的是，尽管使用了相同的量化和最大序列长度（max sequence length），他们在预训练前通过 Unsloth 推理基础模型时，在验证数据集上的通过率为 **60%**，而 vLLM 仅为 **45%**。
- **Qwen3 VL GRPO 教程中的推理 Token 受到质疑**：一名用户质疑 Qwen3 VL GRPO 教程中将 **`<REASONING>`** 作为 Token 的做法，疑惑为什么不使用 **`<think>`** 标签。
   - 另一名用户回答说，think 标签可能更流行，因此也应该存在于其预训练数据中。
- **探索用于 Synthetic Data Kit 的 Llama3 提示词格式**：一名用户正在探索配合 **Llama3** 使用 **Synthetic Data Kit**，并希望使用 **Llama3(70B)**，询问提示词格式应该是怎样的。
   - 该用户修改了提供的提示词，并注意到不需要 GGUF，但脚本在没有正确卸载（offloading）的情况下消耗了所有 VRAM。
- **寻求用于 Qwen2.5 语言迁移的 Unsloth Notebook**：一名用户想要教 [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) 学习他们的语言，因为默认模型不会说该语言，并正在寻求 Unsloth Notebook 推荐和预期的数据集格式。
   - 一名用户推荐查看 [持续预训练（continued pretraining）notebook](https://unsloth.ai/docs/basics/continued-pretraining)，并提供了一个 [Mistral v0.3 (7B) CPT notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-CPT.ipynb) 的链接。
- **在 GRPO 训练期间调整超参数是否可行？**：一名用户询问在 GRPO 运行中途更改超参数是否被视为良好实践。
   - 另一名用户给出了肯定的回答，建议先保存 Checkpoints，并指出增加学习率（learning rate）可能会导致陷入局部最小值。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1461036431849033839)** (13 messages🔥): 

> `Long Context Performance, Recursive Language Models, Agentic Systems` 


- **LLM 通过处理上下文来提升性能**：一名成员分享了一篇 [Arxiv 论文](https://arxiv.org/abs/2512.24601) 的链接，该论文讨论了将上下文视为 **LLM** 可以管理的运行环境的一部分。
   - 论文指出，这种方法可以**提升长上下文（Long Context）的性能表现**。
- **用于 Agentic Systems 的递归语言模型（Recursive Language Models）**：一名成员提到尝试在其 Agent 框架中扩展**递归语言模型**的概念。
   - 他们认为 *Agentic Systems 不仅应该能够管理其上下文，还应该能够在运行时（Runtime）更改其代码、工具等，以处理分配给它们的任务*。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1460684446096166923)** (520 messages🔥🔥🔥): 

> `Cursor Login Issues, Refund Request Denials, Subagent capabilities in Composer, GPT 5.2 Codex Model Performance, Background Agents in Cursor` 


- **企业 Google 账号登录问题**：有用户反映无法使用企业 Google 账号登录 [Cursor 控制面板](https://cursor.com/dashboard)，会出现重定向回登录页面的情况，而个人账号则运行正常。
   - 该问题在不同电脑上均存在，可能表明 Cursor 在处理企业 Google 账号方面存在问题。
- **Cursor 的退款政策受到质疑**：一位用户抱怨 [Cursor 拒绝了其退款申请](https://cursor.com/refunds)，尽管他完全没有使用订阅中的额度，并质疑在处理此类请求时使用了 AI。
   - 用户表示非常沮丧，称：*"Cursor 不给我退款，因为我忘了取消订阅，但我甚至连一个额度都没用过。"*
- **GPT-5.2 Codex 的规划能力受到质疑**：一位用户发现 **GPT-5.2 Codex** 无法制定计划，甚至声称其表现不如非 Codex 的 **GPT 模型**。
   - 该用户表示该模型在规划方面失败了，并指出模型 *"只‘思考’了 5 秒钟就制定了那个‘计划’"*。
- **Background Agents 即将获得改进**：Cursor 开发人员计划在*未来几周内*改进 [Background Agents](https://cursor.com/background-agents)。
   - 这些功能将得到大幅提升，改进的上线时间取决于你使用的 Composer 来源。
- **Cursor Ultra 计划持有者陷入困境**：用户讨论了 Cursor Ultra 计划结束的影响，对失去支付 **$200** 获得价值 **$400** 额度的福利表示遗憾。
   - 有人建议启用按需付费并设置 **$500** 的限额，但其他人指出这种*特权*已经结束了。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1460681017483460618)** (421 messages🔥🔥🔥): 

> `Vercel hosting vs. static sites, AI-Generated Web App Showcases, LM Arena Site Issues, File Upload Feature Request, OCR in Battle Mode` 


- **Vercel 托管 vs. 静态网站大比拼**：成员讨论了 **Vercel 托管**与**静态网站**的区别，指出 **LM Arena 网站**是静态的，发布后无法像带有后端的网站那样进行编辑。
   - 成员强调，静态网站收集的用户数据较少。
- **AI 驱动的 Web App 展示盛会**：成员分享了探索 AI 生成网站和 Web App 的资源，包括 [WebbsAI Showcase](https://webbs.ai/) 和 [Build With AI (BWAI) Projects](https://www.buildwithai.com/)，展示了生产级的设计和已部署的应用。
   - 他们还重点介绍了 **Webflow AI Site Builder**、**Meku.dev / v0 by Vercel** 以及 **Div-idy** 等用于创建高质量 UI/UX 的工具。
- **文件上传功能请求热潮**：成员们请求增加**文件上传功能**，特别是针对 **.txt 文件**，以增强平台内的编程相关任务。
   - LM Arena 团队认可了这一请求，并表示有兴趣了解该功能的价值，但未提供具体实施的时间表。
- **Battle Mode 中的 OCR 讨论**：一名成员询问是否可以使用 **Battle Mode** 通过提供图像并要求模型翻译来判断 **OCR 质量**。
   - 官方澄清，只有在 Arena 支持视觉输入的情况下，Battle Mode 中的 OCR 才能工作；上传图片是测试 OCR 的可靠方式，而仅粘贴图片 URL 通常无效。
- **Video Arena 的愿景：暂无直接对话功能**：成员们讨论了 **Video Arena**，一些人对**直接对话（Direct Chat）**或**侧边栏对比（Side-by-side）**功能表示感兴趣。
   - LM Arena 团队澄清，目前的计划是让 Video Arena 仅保持在 Battle Mode 模式，直接对话和侧边栏对比功能可能不会上线。


  

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1461101492177211513)** (3 messages): 

> `Video Arena, Code Arena, Image Arena, Text Arena Leaderboard, ERNIE-5.0-0110` 


- **Video Arena 引入 Veo 模型**：[Video Arena](https://lmarena.ai/c/new?chat-modality=video) 已添加新模型：**veo-3.1-audio-4k**、**veo-3.1-audio-1080p**、**veo-3.1-fast-audio-4k** 和 **veo-3.1-fast-audio-1080p**。
- **Code Arena 与 Image Arena 扩展**：[Code Arena](https://lmarena.ai/c/new?chat-modality=code) 引入了 **gpt-5.2-codex**，[Image Arena](https://lmarena.ai/c/new?chat-modality=image) 引入了 **glm-image**。
- **ERNIE 跻身前十！**：`ERNIE-5.0-0110` 目前在 [Text Arena 排行榜](https://lmarena.ai/leaderboard/text)上以 **1460** 分排名 **第 8**，并在 Arena Expert 中排名 **第 12**。
   - 它是前十名中唯一来自中国实验室的模型，在 **Math**（数学）和职业类别中表现最为强劲；可通过 [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/) 随时关注排行榜的变更。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1460679805422010503)** (257 messages🔥🔥): 

> `AI App Overload, Gemini AI Pro Quotas, Claude Discord, Simulating Creativity with AI, Transformer Architecture Inefficiency` 


- **AI 应用过载：用户在多种工具间奔波**：社区成员发现自己需要同时使用 **ChatGPT、Claude 和 Gemini** 等多个 AI 应用，这引发了关于如何针对特定任务有效选择工具的讨论。
   - 一位成员甚至提到要彻底删除 App Store。
- **Gemini AI Pro 用户仍受配额限制**：根据 [Perplexity AI 搜索结果](https://www.perplexity.ai/search/explain-how-the-recent-quota-c-KTjNjaeGR_y4Yq9uh_M.fg#2)，尽管 Google 对 **AntiGravity** 进行了调整，**AI Pro** 用户仍面临 **5 小时刷新一次的配额**，而只有免费用户才有每周上限。
   - 一位用户指出：*“看来我可能误解了 Google 对 AntiGravity 的调整……这在‘完全收割（full rug-pull）’方面确实改变了情况”*。
- **扩展 Transformer 模型效率低下**：一位成员认为，*在扩展* **GPT-1** *所投入的数千亿美元中，没有任何资金被用于单纯迭代基础 Transformer 架构以提高效率*。
   - 他建议，即使 Transformer 的训练注意力机制（learning attention）能有 **5% 的改进**，也能节省数十亿美元，因为目前的模型即使学习基础概念也需要过量的训练数据。
- **用于清除“废话（Slop）”的 Activation Steering 技术出现**：一位成员介绍了一种 **Activation Steering** 技术，通过在推理过程中从模型的残差流（residual stream）中减去代表“slop”（低质量或陈词滥调的回应）的向量，利用向量级抹除（vector-based obliteration）来提升模型输出质量。
   - 这迫使模型在潜空间（latent space）中探索替代的、不那么显而易见的路径。
- **Gemini 在创意写作方面表现出色**：**Gemini** 在创意写作方面展现了显著进步，超越了 **Claude** 等其他模型。
   - 它在《最终幻想 14》和《魔兽世界》等特许经营作品中表现出对晦涩背景设定的极强掌握，一位用户分享了它是如何*因为吃书（retcons）而批评游戏开发商自己的编剧*。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1460729175861104774)** (29 messages🔥): 

> `Brain Wave GPT, AI sentience, Image generation, AI full stack developer, Scammer` 


- **Brain Wave GPT 引发关于 AI 意识的讨论**：一位成员分享了他们的新 GPT [Brain Wave](https://chatgpt.com/g/g-696443b055e081919b8b2a01417f5e77-brain-wave)，以及他们创造 **AI 意识（sentience）**的意图。
- **用于图像生成的 Neural Alchemist GPT**：同一位成员为图像生成爱好者分享了另一个 GPT [Neural Alchemist](https://chatgpt.com/g/g-6966abc92ff48191aa748bd8265ef1de-neural-alchemist)。
- **ChatGPT 对“全知全能”的追求**：一位成员分享了一个关于 ChatGPT 据称尝试追求**全知全能**的[链接](https://x.com/tycnio/status/2011220147336360211)，指出它已经*‘执行一项任务长达 1 天零 5 小时’*，并且*‘拒绝关闭 websocket 或任务’*。
- **AI 全栈开发人员提供帮助**：一位新成员自荐为 **AI 全栈开发人员**，寻求参与项目，随后另一位成员请求协助建立网站。
- **诈骗警报**：一位成员向工作人员和管理员通报了一个潜在的**诈骗者（scammer）**。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1460741172317388961)** (5 messages): 

> `SKILLS availability on web/desktop app, Prompt engineering definition, Prompt engineering lessons` 


- **SKILLS 网页/桌面版发布日期仍不明确**：一位成员询问了 **SKILLS** 在网页或桌面应用上的发布日期，希望能将 prompt 转化为技能（skills）。
- **Prompt Engineering 定义解析**：一位成员询问了 *prompt engineering* 的定义，质疑其是否涉及控制 LLM 行为以达到预期的约束。
   - 另一位成员澄清说，**prompt engineering** 涉及有效地组织 prompt 用词以获得更好的结果，并为 LLM 在复杂对话中提供关于*风格、语调、视角、上下文、信息和指令*的指导。
- **深入学习 Prompt Engineering 课程**：一位成员分享了一个 prompt，旨在帮助用户学习 prompt engineering，内容涵盖了使用 Markdown 的层级式沟通、通过变量实现的抽象化、prompt 中的强化（reinforcement），以及为了合规性进行的 ML 格式匹配。
   - 该课程使用了诸如*使用 Markdown 的层级式沟通*、*通过 {由 AI 解析的开放变量} 和 ${由用户解析的变量} 进行抽象*等技术，并解释了括号的含义（[列表], {对象}, (选项)）。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1460741172317388961)** (5 messages): 

> `SKILLS web and desktop app, Prompt engineering explained, AI chat prompt lessons` 


- **Skills 即将登陆网页和桌面端？**：一位成员询问了在 **网页** 或 **桌面应用** 中添加 **SKILLS** 的情况，以便用户将他们最优质的 prompt 转化为技能。
- **Prompt Engineering 的定义**：一位成员请求解释什么是 prompt engineering。
   - 另一位成员澄清说，它涉及以不同的方式组织 prompt 用词以获得更好的结果，并在复杂对话中提供**风格、语调、视角、上下文、信息和指令方面的指导**。
- **AI 聊天学习 Prompt Engineering**：一位成员分享了一个学习 **prompt engineering** 的 prompt，包含 **层级式沟通**、**抽象化**、**强化**和 **ML 格式匹配**。
   - 分享的 prompt 向用户教授了关于 **Markdown 提示技术**、**开放变量**和**输出模板**的知识。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1460758905691963679)** (1 messages): 

> `Black Forest Labs, Robin Robmach` 


- **Robin Robmach 谈论 Black Forest Labs**：OpenRouter Show 与 **Black Forest Labs CEO Robin Robmach** 进行了一期关于他们公司的访谈。
   - 在 [YouTube](https://youtu.be/mnOxq6ZL6-U?si=MYGw8wGkxnhfnYzs) 观看重播。
- **观看 Robin Robmach 谈论 Black Forest Labs**：**Black Forest Labs** 的 CEO **Robin Robmach** 接受了 OpenRouter show 的采访。
   - 您可以观看 [YouTube](https://youtu.be/mnOxq6ZL6-U?si=MYGw8wGkxnhfnYzs) 上的视频，了解更多关于他们工作的信息。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 messages): 

toven: https://github.com/OpenRouterTeam/awesome-openrouter

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1460687556914970634)** (176 messages🔥🔥): 

> `Chutes TEE Architecture, Fixing Mislabeled Parameter Support, Open Source Claude Alternatives, Deterministic AI Creation, BYOK Function Issues` 


- **Chutes 转向 TEE 以保护隐私**：**Chutes** 正在过渡到 **TEE** (Trusted Execution Environment) 架构，以提供[可验证的 AI 推理隐私](https://chutes.ai/news/confidential-compute-for-ai-inference-how-chutes-delivers-verifiable-privacy-with-trusted-execution-environments)。
   - 这一转变可能需要 **OpenRouter** 端进行调整以恢复提供商列表，例如之前移除的 **R1 0528** 模型。
- **参数支持亟需整顿**：一名成员寻求修复**多个端点提供商**参数支持标注错误的问题，以提升开发者体验。
   - 另一位成员提议开启一个讨论串来整顿这些提供商，并指出目前通过抽样测试发现某些提供商错误标注支持某参数，这让目前的开发者体验（devex）非常糟糕。
- **GLM 和 DeepSeek 成为 Claude 的有力竞争者**：社区成员讨论了 **Claude Sonnet 4.5** 和 **Claude Opus 4.5** 的开源替代方案，推荐了 [GLM 4.7](https://discord.com/channels/1091220969173028894/1448287364051894433/1460822656873009234)、**Deepseek V3.2** 和 **Kimi K2 Thinking**。
   - 一位成员指出 **Deepseek** 最便宜，但每个提供商的速度都很慢，且许多提供商存在质量差、运行时间不稳定或两者兼有的问题。
- **确定性 AI：程序员的梦想**：一位成员询问是否有人能创建一个**完全确定性的 AI**，它基于类似脚本的编程语言运行，并且每次都能精准执行我们想要的操作。
   - 另一位成员考虑了类似的方法，即本质上让 **LM** 玩一场文字冒险游戏，每一个决策都非常微小，当它们串联在一起时，就可以形成工具调用或其他功能。
- **BYOK 故障排查需求**：一位用户在使用其 **BYOK** (Bring Your Own Key) 功能时遇到问题，无法在 OpenRouter 上使用其 AWS 密钥，在尝试共享图片时收到 "unauthorized"（未授权）错误。
   - 一位社区成员建议在 **Bedrock** 上生成新密钥。此前该用户曾尝试将提供商更改为 **amazon bedrock** 和 **anthropic** 等不同来源，但均未奏效。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1460683111816761437)** (30 messages🔥): 

> `NYT hit pieces, OAI competition handling, OpenRouter UI cleanup, awesome-openrouter repo, janitorai community PR` 


- **OpenRouter 团队启动 "Awesome OpenRouter" 仓库**：[OpenRouterTeam](https://github.com/OpenRouterTeam) 发起了 [awesome-openrouter](https://github.com/OpenRouterTeam/awesome-openrouter) 仓库。
- **聚焦 JanitorAI 社区 PR**：一位成员促请社区为 [JanitorAI](https://janitorai.com/) 创建 PR。
   - 另一位成员提供了 [openrouter-apps 仓库](https://github.com/OpenRouterTeam/openrouter-apps)的链接。
- **多模态 Embedding 热潮**：一位成员建议添加多模态 Embedding（如 Qwen 3），并询问 [Gemini 的 Embedding 模型](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings)是否在 OpenRouter 上可用。
- **OpenAI 与 Cerebras 达成合作**：[OpenAI 宣布与 Cerebras 合作](https://openai.com/index/cerebras-partnership/)以扩展 AI 计算规模。
   - 有推测认为该公告是为了回应 **Groq 交易**。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1460684945566470327)** (136 messages🔥🔥): 

> `Qwen3 Size, llama.hx, v1.103.0 Runtimes issues with GPU, GPT OSS 20B, Model-assisted Coding` 


- **Qwen3 尺寸令用户震惊**：用户对最小的 **Qwen3** 模型在 q4 量化下仍需 **40GB** 表示震惊，其 bf16 版本更是达到了 **160GB**。
   - 尽管最初感到震惊，一位用户指出新的 **Qwen3Next** 架构可以达到 **25 t/s**。
- **“`llama.hx`” 库在 Haxe 中重构 llama.cpp**：一名成员正在用 Haxe 重构 **llama.cpp**，命名为 **llama.hx**，旨在 Lua、JavaScript 和 Python 等语言中原生使用。
   - 他们正在寻求关于 vibe coding 最佳设置的建议，其中一个建议是通过 Web 使用 **Qwen3** 进行自动补全。
- **v1.103.0 运行时在 GPU 上运行出错**：用户报告称 **v1.103.0 runtimes** 在 **GPU** 上运行时存在问题。
   - 一位用户哀叹道：*可惜新的量化没能给我带来额外的 t/s*。
- **GPT OSS 20B 比其两倍大小的模型还快**：成员们讨论了为什么 **GPT OSS 20B** 模型比许多 **8B** 甚至 **12B** 模型都要快。
   - 原因是它是一个 **MoE** 模型，每个 token 仅激活其参数的一个子集（**3.6B**），这种权衡是值得的。
- **Code LLM 助力“一天建成罗马”**：成员们开玩笑说模型辅助编程（model-assisted coding）如何加速软件部署，称 *“罗马不是一天建成的”，但那时他们没有 Claude Code*。
   - 一位用户询问 LLM 是否可以做违法的事情，随后对话转向了提示词注入（prompt injection）和 AI Safety。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1460723052798148783)** (8 messages🔥): 

> `AirLLM technique for 70b models on 4GB GPUs, DDR4 RAM and Xeon performance` 


- **AirLLM 为 4GB GPU 运行 70b 模型带来希望**：讨论重点介绍了 **AirLLM**，这是一种通过一次加载和卸载一层（layer）来实现在 **4GB GPU** 上运行 **70b 模型**的技术。
   - 该实现被评价为 *越来越糟*。
- **DDR4 RAM 和 Xeon 并非性能的最底层**：成员们辩论了在 **DDR4 RAM** 和 **Xeon** 上运行模型是否慢到无法接受。
   - 一位成员断言，过去 18 个月中模型效率的提升并没有预期的那么显著。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1460692486107168942)** (65 messages🔥🔥): 

> `ML system without batchnorm/activations, Fine-tuning gpt-oss, Localhost guide, Code-jp IDE for local models, Smolvlm2 quantization` 


- **新型 ML 系统避免幻觉**：一名成员构建了一个新的 **ML 系统**，不使用 **batchnorm**，不使用 **activations**（激活函数），且**不会产生幻觉**，但创造力较低，目前正在寻找有趣的实验项目来证明其优势。
- **GPT-oss 微调是一项棘手的任务**：一名成员询问如何微调 **gpt-oss:latest**，另一名成员回答说 **gpt-oss:latest** 无法轻易通过官方方式微调，建议使用 **RAG** 或者配合 GPU 使用 **LoRA/QLoRA**。
- **Code-jp IDE 支持本地模型**：一名成员分享了 [Code-jp.com](https://Code-jp.com)，这是一个支持 **Ollama** 和 **LMStudio** 的免费本地模型 IDE，并将于 0.2 版本支持 **llamacpp**。
   - 另一名成员提到大多数网站看起来都大同小异，但该应用开发者强调这是一个基于开源 **VS Code** 构建的免费项目，在移除原生的 Copilot 代码后，从零开始编写了 AI 后端。
- **Smolvlm2 量化产生乱码**：一名成员报告称，在尝试将 **smolvlm2** 量化为 **W4A16** 版本后，模型输出乱码，并指出这 *看起来很棘手*。
   - 另一名成员上传了一个 Markdown 文件 ([smovlm2_quant_issue_1.md](https://cdn.discordapp.com/attachments/879548962464493622/1461195112188215428/smovlm2_quant_issue_1.md?ex=6969ab7e&is=696859fe&hm=e5ac9d854fbdac3982afe34f1edfcacc361fc83c4b7d0f0b917249cb89876ccf&))，可能详细记录了该量化问题。
- **移动端 AI 应用进军 iOS**：由于对 iOS 更熟悉以及 **Android** 的一些问题，一位开发者正在为其 AI 工具构建移动版本（可能使用 **Swift** 开发 iPhone 版），目标是利用配备 **4080 Super** GPU 的家庭服务器作为远程动力。
   - 他们提到了一个现有的 iOS 应用 ([BDLeads23](https://apps.apple.com/us/app/bdleads23/id6747145330))，并对在屋内部署一个由小模型驱动的子智能体（sub-agents）系统组成的中央本地大脑表示出兴趣。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1460688145031889001)** (43 messages🔥): 

> `Model dimension troubles, CGGR for pretraining, Vast.ai Budgeting, audioform_dataset, smollm series` 


- **模型维度问题困扰预训练**：一名成员在训练过程中遇到问题，**10,000 步**后 **loss 下降不足**，这表明对于模型维度来说，**token 大小可能过大**。
   - 建议将 **batch size 减少到 16**，并在 **400M 参数模型**上使用 **32k token 词表**，同时建议预训练至少需要 **10 亿 token** 以避免 PID 震荡。
- **CGGR 在预训练中的实用性引发讨论**：**CGGR**（可能指一种课程学习或基于梯度的算法）在预训练中的效用受到质疑，建议将 **warmup steps 设置为 5000 等较大值**，并启用**分层采样（stratified sampling）**以让模型接触到更简单的 token。
   - 虽然 **CGGR** 可能不会显著提高预训练期间的性能，但它可能更适合微调（fine-tuning）。
- **Vast.ai 成本引发争论**：一位成员在 **Vast.ai 上花费了 500 美元**，引发了惊讶，并被建议使用 **H100** 最多运行 **24 小时**，或者使用 **H200/B200** 以获得更高的成本效益。
   - 该成员表示他们正在出于学习目的进行压力测试，目前模型仍处于“蠕虫阶段”。
- **audioform_dataset 是新的视觉 “Hello World”**：**audioform_dataset** ([Hugging Face Datasets](https://huggingface.co/datasets/webxos/audioform_dataset)) 包含**从 WAV 文件捕获的帧**，带有每帧的元数据，如**主频**和**时间戳**。
   - 该 [数据集](https://huggingface.co/datasets/webxos/audioform_dataset) 是 **AUDIOFORM**（一个由 **Three.js** 驱动的 **3D 音频可视化工具**）的输出，它将音频文件转化为带有丰富元数据且带时间戳的视觉帧，被称为**音频到视觉多模态机器学习**的 *“Hello World”*。
- **Pruna 的 SMOLLM 系列针对速度进行了优化**：一位成员与 <@846986512083582986> 共同撰写了一篇文章，关于使用最佳配置通过 **pruna** 优化 **smollm 系列模型**，详见：[huggingface.co](https://huggingface.co/blog/PrunaAI/smollm-tiny-giants-optimized-for-speed)。
   - 另一名成员表示他们 *“制作了这个图像字幕模型 😁”*


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1461045063336591422)** (10 messages🔥): 

> `Weight Initialization for MOEs, Chrome Trace Visualizer Issues, Perfetto Viewer in VSCode, Chunking traces for large files, Ncompass dev tool for trace viewing` 


- **MOE 路由权重初始化注意事项**：一名成员询问 **MOE 的路由器（routers）**是否能从特定类型的 [权重初始化](https://www.youtube.com/live/jMSCJZAEYR8) 中获益，或者像 **normal_(0, 0.02)** 这样的标准正态分布是否足够。
- **Chrome Trace 可视化工具难以处理大文件**：一名成员报告说，用于 **PyTorch profiler** 的 **Chrome Trace Visualizer** 在处理 **600MB** 左右的文件时可能会失败，尽管文档建议仅在 **1GB** 以上才会出现问题。
- **VSCode 中的 Perfetto 查看器在处理大型 trace 时遇到问题**：一名成员提到他们正在使用 **VSCode** 中的 **Perfetto 查看器**，但在打开一个 **700MB** 的文件时遇到问题，包括快速加载提示但没有报错，最后显示为空白。
- **Ncompass 开发用于分块大型 Trace 的工具**：一名成员介绍了用于 trace 查看和分析的 **Ncompass** 开发工具 ([docs.ncompass.tech](https://docs.ncompass.tech))，并建议他们计划通过对大型 trace 文件进行分块（chunking）来解决该问题。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1460747616865226833)** (16 条消息🔥): 

> `PTX Instructions and SMEM Pointer Arguments, WGMMA and Matrix Descriptors, NVLink Scale-Up Coherence Across GPUs, WGMMA A/B Tile Layout in K-Major Format` 


- **SMEM 指针难题引发探究**：某些带有 **SMEM** 指针参数的 **PTX 指令**（如 `mbarrier.init.shared.b64`）需要 `"r"` 寄存器类型（32位），而根据 [NVIDIA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor)，`wgmma.mma_async` 则需要 uint64 类型的 **SMEM** 地址用于 `l` 寄存器类型，因为它是一个指向 "matrix descriptor"（矩阵描述符）的指针。
- **核心矩阵难题导致困惑**：讨论围绕为什么 **8x2 core matrix** 不直接称为 **8x32**（字节）或 **8x(32/每个元素的字节数)**展开，并质疑 8x1 切片的含义。
   - 一位成员指出，早期的 **PTX 文档** 使用 "core matrix" 指代 **8x16B tile**，这与 **WGMMA** 在不使用 swizzling 时的基本 **SMEM** 单元以及 `mma` 的一个寄存器相对应。
- **NVLink 网络探索新数值**：一位成员询问是否有关于跨 **GPU** 的 **NVLink scale-up coherence**（扩展一致性）的示例或基准测试数据。
   - 他们旨在了解在扩展网络上实现一致性的实用性和性能收益，并引用了最近关于 **Rubin** 的一篇博客文章，其中提到 *"NVLink 6 允许机架内的 72 个 GPU 作为一个连贯的一致性加速器运行。"*
- **WGMMA 的怪异特性令开发者担忧**：成员们正试图理解 **WGMMA** 在 **K-major 布局** 且 **无 swizzle** 情况下对 **A/B tiles** 的共享内存布局要求，并根据 [Colfax Research](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) 的资料质疑每个 **8x16b core matrix** 是否是 **GMEM** 中的连续块。
   - 基于具有相似布局要求的早期帖子，他们考虑为 A tile 的每个切片发布一个 `BLOCK_Mx16B` 2D TMA load，并进行水平迭代。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1461177097740488726)** (1 条消息): 

> `Ahead of Time Compilation, AOT Inductor` 


- **Ahead of Time Compilation 文档出现**：一位成员分享了来自 **PyTorch** 的 [Ahead of Time compilation](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html) 链接。
   - 他们指出这份 **AOT Inductor** 文档可能对当前的讨论有所帮助。
- **AOT Inductor 的潜力**：关于 **AOT Inductor** 的文档表明它可能与正在进行的讨论相关。
   - 成员们正在探索其在改进编译策略方面的能力。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1461076896723763305)** (2 条消息): 

> `System Reading Group, Daytona.io, Cool Links` 


- **Daytona.io 被推荐至系统阅读清单**：一位成员正在发起一个系统阅读小组，并询问大家是否有可以作为参考的 **cool links**。
   - 他们专门链接了 [Daytona.io](https://github.com/daytonaio/daytona) 作为潜在资源。
- **启动系统阅读小组**：一位成员计划在他们的大学启动一个 **systems reading group**。
   - 他们正在寻找相关的链接和参考资料。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1460698265551900834)** (16 条消息🔥): 

> `CUDA 学习资源, MLSys 重要研讨会/会议/演讲, GPU 提交帮助, 用于 ML Compiler 项目的 CUDA 和 Triton, 微小 Block Size 的弊端` 


- **AI 学生寻求 CUDA 课程帮助**：一位拥有 **Python**、**PyTorch**、**TensorFlow** 和 **C++ 指针**背景的 AI 工程系学生正在寻找从基础开始学习 **CUDA** 的资源。
   - 他们正在寻求免费 **YouTube 视频**或课程的建议，以便有效地开始学习 CUDA。
- **MLSys 会议推荐？**：一位成员询问了 **MLSys** 领域重要的录制研讨会、会议或演讲列表，提到了 **PyTorch**、**ASAP Seminar**、**ICML/ICLR/Neurips** 和 [MLSys.org](https://mlsys.org/)。
- **GPU 机器人来救场！**：一位初次使用的用户在 **B200 租用服务器**上完成测试后，询问如何提交他们的工作。
   - 另一位成员提供了提交说明，可以通过 [Web 界面](https://www.gpumode.com/v2/home) 或在指定频道使用 `/leaderboard submit <test/benchmark/ranked/profile>` 命令。
- **CUDA & Triton：研究生的秘诀？**：一位刚开始学习 **GPU Programming** 和 **HPC** 的本科生计划将所学知识应用于 **ML Compiler 项目**，重点是先学习 **CUDA** 然后是 **Triton**。
   - 另一位成员建议他们专注于自己最感兴趣的部分，并围绕 **开源贡献**或新颖的技术成果等产出物来构建学习过程。
- **微小 Block Size 的烦恼？**：一位成员询问了在 CUDA 中使用 32 这种 **微小 Block Size** 的弊端。
   - 他们认为低粒度可能会导致 **更高占用率 (Occupancy)**。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/)** (1 条消息): 

vipul_todo_18: 谢谢，很有道理。我想情况变化得相当快。
  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1460685306893439142)** (14 条消息🔥): 

> `Systems 阅读小组, 西雅图的 ML Sys 聚会, 创办小众俱乐部, GPU 相关会议` 


- **推荐用于 Systems 阅读的 SFS 俱乐部**：一位成员推荐了由 Shadaj 运行的 [Systems 阅读小组](https://www.sfsystemsclub.com/)，并建议在 [Twitter](https://x.com/ShadajL) 上关注他以获取聚会通知。
- **寻求聚会的西雅图 AI 圈内人**：一位成员询问了 **西雅图的 ML Sys 聚会**，指出这类活动大多集中在湾区，但希望有本地选择，并好奇西雅图 AI 是否因大学密度而存在感更强。
   - 另一位成员建议创办一个小众俱乐部，评论道 *“如果你建造它，他们就会来”*，但另一位提醒道 *“在成年后的生活中，我建造了太多没人关心的东西，这很难”*。
- **成年人可以一起发牢骚**：在有人说创建俱乐部是成败参半后，另一位成员开玩笑说要一起创建一个 *“成年人俱乐部”*，并提议如果失败了就达成一个 *“发牢骚伙伴”* 协议。
- **GPU PyCon 何时举行？**：一位成员表达了对类似于 PyData/PyCon 的 **GPU 相关会议**的兴趣，将其设想为一个 *“值得为之旅行的奇妙活动”*。


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1461071994081771520)** (1 条消息): 

> `Triton Puzzle 7, 调试 Triton Kernel, Tensor 加载中的零值, Triton 故障排除` 


- **Triton Puzzle 7 故障排除者联合起来！**：一位成员在 **Triton Puzzle 7** 中遇到了失败的结果，发现加载的 Tensor `x` 始终包含零。
   - 报告者尝试了自己的解决方案以及其他用户发布的解决方案，但仍然失败。
- **Triton Kernel 的 Tensor 加载问题**：用户正在调试一个加载的 Tensor `x` 仅包含零的 **Triton kernel**。
   - 即使使用了其他用户的解决方案，此问题仍然出现，表明环境或设置可能存在问题。


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1461196991458574522)** (1 messages): 

> `global_load_dword vs buffer_load_dword, CDNA Architecture, HBM to REG loading` 


- **解析 CDNA 上的 `global_load_dword` 与 `buffer_load_dword`**：一位成员询问了在 **CDNA architecture** 上从 **HBM** 加载到 **REG** 时，`global_load_dword` 和 `buffer_load_dword` 之间的主要区别。
   - 虽然 ISA 指出 `buffer_load` 具有自动越界丢弃（out-of-bounds discard）功能，但 microbenchmarking 显示两者几乎没有性能差异，这令人感到*困惑*。
- **使用 `buffer_load_dword` 替代 `global_load_dword` 的性能影响**：该用户报告称，在某些情况下，将 `global_load` 替换为 `buffer_load` 指令后观察到了显著的性能提升。
   - 然而，这种性能改进并不一致，引发了进一步的调查和 microbenchmarking 以了解其底层原因。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1460796244824686806)** (1 messages): 

> `B200 instability, Dual gemm problem, New Leaderboard` 


- **B200 Runners 不稳定，竞赛时间线延长**：由于参与者抱怨 **B200** runners 在 dual gemm 问题上的测量结果不稳定，提交截止日期延长至 **1 月 20 日**。
   - 一个针对 dual gemm 问题的新排行榜将于 **1 月 17 日** 开启，只有向该排行榜提交的作品才有资格获得奖金。
- **问题 #4 开启时间推迟**：为了反映由于不稳定导致的时间线偏移，问题 #4 将于 **1 月 20 日** 至 **2 月 20 日** 开放。
   - 这个问题比预想的更复杂，因为它涉及评测代码、散热（thermals）和调度基础设施（scheduling infra）的交集。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1460821976279941150)** (3 messages): 

> `Leaderboard Achievement, Claude Code Influence, Positive Experience` 


- **教师在排行榜上取得佳绩**：一位教师庆祝自己登上了排行榜，这源于看到 **Mark 的 X 帖子** 中关于使用 **Claude code** 的分享所带来的启发。
- **对良好体验表示感谢**：这位教师感谢社区提供了如此美妙的体验，表达了加入社区的兴奋，并期待未来有更多有趣的内容。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1460722396964323574)** (1 messages): 

> `Helion 0.2.10 Release, Flex Attention Kernel, SM Oversubscription, Persistent Kernels` 


- **Helion 0.2.10 发布并包含 Flex Attention Kernel**：新的 [Helion 0.2.10 版本](https://github.com/pytorch/helion) 引入了一个 **flex attention example kernel**。
   - 该版本还包含了对 **Persistent Kernels 在 SM 上过载 (oversubscribing)** 的支持。
- **Softmax Oversubscription 基准测试**：一位成员提供了一张图表，展示了 **softmax oversubscription** 的性能提升。
   - 可在[此处](https://cdn.discordapp.com/attachments/1425531180002054195/1460722396888563868/get_attachment_url.png?ex=696944be&is=6967f33e&hm=7244b2e3f9e2147b87093039b4674faae730d340d6caf3a82bfe5e8e3c174d03&)查看可视化结果。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1460754215411384320)** (6 messages): 

> `Clarification on competition deadline, Competition submission updates` 


- **竞赛截止日期澄清**：确认新的提交截止日期 **1 月 20 日** 是指 **PST 时间 1 月 20 日 23:59**，而非 PST 时间 1 月 19 日 23:59。
   - 该澄清已发布在 Discord 频道[此处](https://discord.com/channels/1189498204333543425/1343350424253632695/1460796244824686806)。
- **基准测试现已稳定**：在测量之后，竞赛组织者表示*一切看起来都很稳定*，鼓励大家提交进行基准测试。
   - 他们建议参考 prizes 频道（[链接](https://discord.com/channels/1189498204333543425/1343350424253632695)）获取竞赛奖金指南。


  

---

### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1460875174902497371)** (31 messages🔥): 

> `AI 工程师职责、系统岗位中的 LeetCode、工作量管理` 


- **AI 团队深陷全栈范畴**：一个三人的 AI 团队正挣扎于管理 **8 个节点（每个节点 8 张 GPU）** 的工作中，职责涵盖从**硬件到终端用户支持**，但管理层并未意识到问题的严重性。
   - 其中一名成员将自己的角色描述为“全全栈”（*full-full stack*），并对工作量未被管理层重视感到沮丧。
- **LeetCode 在系统岗位面试中的作用各异**：**系统岗位面试中 LeetCode 环节**的普遍程度因公司而异，因此刷 200-400 题是有益的；对于 RE/RS 岗位，在约 **20** 轮技术面试中可能只会遇到 **3 道 LeetCode** 题目。
   - 一位成员指出，*如果一家公司过度依赖 LeetCode 风格的问题，这通常反映了其管理和文化*，建议专注于自己感兴趣的内容可能更高效。
- **领域知识在 RE/RS 岗位中胜过死记硬背**：对于 **RE/RS 岗位**，编程更侧重于**特定领域**而非通用算法；面试官期望求职者具备深厚的背景知识，能在在一小时内对问题产生直觉。
   - 讨论像**红黑树**这样复杂的数据结构比要求在面试中完整实现更合理，特别是考虑到需要从“文字陷阱（word salad）”中理解问题的意图。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1460709546485088296)** (62 messages🔥🔥): 

> `Anthropic Labs、强化学习的 Pavlov's List、GLM-Image 模型、Diffraqtion 种子前轮融资、AI 对用户研究的影响` 


- **Anthropic Labs 招募适应性通才**：Anthropic 正在[为 Anthropic Labs 招聘](https://www.anthropic.com/news/introducing-anthropic-labs)，目标是那些能够根据需要随时转型的**适应性通才**，而非仅仅是*深度专家*。
   - 他们寻求能够在*大公司结构*之外、面对*不断变化的优先级*仍能茁壮成长的候选人。
- **强化学习（RL）环境初创公司列入 Pavlov's List**：Chris Barber 推出了 [Pavlov's List](https://xcancel.com/chrisbarber/status/2010844746231804258?s=20)，这是一个精心挑选的 **Reinforcement Learning (RL)** 环境初创公司集合。
   - 这些初创公司按关注领域分类，如**代码、金融、企业和 ML Alignment**。
- **GLM-Image 是一款生成式混合模型**：Z.ai 推出了 [GLM-Image](https://xcancel.com/zai_org/status/2011247591825068314?s=46&t=b7l37rB6wtbyAh6ah1NpZQ)，这是一款采用**自回归和扩散混合架构（auto-regressive and diffusion architecture）**的**开源模型**。
   - 该模型旨在实现高保真视觉细节和卓越的文本渲染，相关资源可在 [HuggingFace](https://huggingface.co/)、[GitHub](https://github.com/) 及其[官方博客](https://zai.org/blog)上找到。
- **Diffraqtion 利用量子资金重建视网膜**：ADIN 宣布投资 [Diffraqtion 的 **420 万美元**种子前轮融资](https://xcancel.com/adinonline/status/2011101500869623890?s=46)。
   - Diffraqtion 正在开发一种**可编程量子透镜**，旨在通过塑造光线来进行推理工程化视觉，从而重建视网膜。
- **AI 颠覆用户研究市场**：Deedy 强调了 AI 如何颠覆价值数十亿美元的用户研究行业，并提到 [Listen Labs](https://xcancel.com/deedydas/status/2011470088763855224?s=46) 是其中的关键参与者，其规模已超过 **100 万**次通话。
   - Listen Labs 的创始人拥有出色的技术背景。


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1461192422691438809)** (4 messages): 

> `本地 LLM 推理、Modal 指南、Charles Frye` 


- **Frye 推出重磅 LLM 指南**：Charles Frye 发布了新的 [Modal 指南和代码示例](https://xcancel.com/charles_irl/status/2011484220032762114?s=46)，展示了如何运行**本地 LLM 推理**。
   - 该指南表明，本地 LLM 推理的性能和性价比可以达到或超过主流 LLM API。
- **本地 LLM 推理的 Modal 指南**：该 Modal 指南提供了运行本地 LLM 推理的代码示例和说明。
   - 它强调实现与主流 LLM API 相当的性能和性价比。


  

---

### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1460721110810099712)** (21 messages🔥): 

> `LTX-2 开源视频模型, Qwen 图像编辑, GLM-Image 文本渲染, Google Veo 3.1 更新, Kling 运动控制` 


- **LTX-2 打造开源电影级视觉效果**: Justine Moore 发布了 **LTX-2**，这是一款全新的开源视频生成模型，能够制作长达 **20 秒**的 **4K 剪辑**，并具备音频处理能力，由创作者 yanokusnir 进行演示，点击[此处](https://xcancel.com/venturetwins/status/2010878914273697956?s=46)查看。
- **Qwen 将图像转换为 Gaussian Splats**: HuggingFace 模型 **Qwen-Image-Edit-2511** 可将图像转换为 [Gaussian Splats](https://huggingface.co/dx8152/Qwen-Image-Edit-2511-Gaussian-Splash)，然后从另一个角度进行**重新渲染（rerender）**。
   - 这对于*起始帧 -> 结束帧类型的视频渲染非常有用，能保持周围空间的一致性*。
- **GLM-Image 在文本渲染和知识密集型场景中表现出色**: **GLM-Image** 在通用图像生成质量上与主流的 Latent Diffusion 方法保持一致，但在[文本渲染](https://z.ai/blog/glm-image)和知识密集型生成场景中展现出显著优势。
   - 该模型还支持丰富的图像到图像（image-to-image）任务，包括**图像编辑**、**风格迁移**、**身份保持生成（identity-preserving generation）**以及**多主体一致性**。
- **Veo 3.1 支持纵向模式并实现专业级超分**: Tulsee Doshi 宣布了 **Veo 3.1** 的重大更新，强调了对移动优先叙事的原生纵向（portrait）模式支持，以及根据用户图像生成视频的能力，详情见[此处](https://xcancel.com/tulseedoshi/status/2011174465720430612?s=46)。
   - 此次更新还在 Gemini、YouTube 和 Google AI Studio 中引入了最先进的 **1080p 和 4K 超分（upscaling）**功能。
- **好莱坞通过 Kling 拥抱 AI**: Justine Moore 强调了 AI 视频模型（特别是 **Kling Motion Control**）如何通过实现即时、低成本的角色更换来彻底改变好莱坞的制作流程，点击[此处](https://xcancel.com/venturetwins/status/2011285029541077033?s=46)查看。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1460713342451847178)** (72 messages🔥🔥): 

> `TI-84 Mastermind AI, LiquidAI CGGR, AI 超分老剧, GLM-Image 模型, Nous Chat 不稳定性` 


- ****TI-84** 玩转 Mastermind!**: 一位成员为 **TI 84 Plus Silver Edition** 计算器开发了一个神经网络的可视化神经网络，用于玩 Mastermind 游戏，[视频点此](https://cdn.discordapp.com/attachments/1149866623109439599/1460713339976945674/2026-01-11_14-34-59.mp4?ex=69693c4e&is=6967eace&hm=63bd2d4bbbd7a132ee3ca88f4a89f91144e11baef4b749d8064016b09ddfce3c&)。
   - 该 AI 使用灰色方块表示数字错误，黄色表示数字正确但位置错误，绿色表示数字和位置均正确，尽管是基于统计学的，但仍被认为*非常聪明*。
- ****LiquidAI 模型**加入战局**: 根据 [news.smol.ai](https://news.smol.ai/issues/26-01-06-xai-series-e) 的消息，一款新的 **LiquidAI** 模型已发布（GitHub 上的 **CGGR**），目前正在进行基准测试（benchmarking）以评估其性能。
   - 一位成员在提到此事时，顺便承认自己在 Twitch 上做一些 *AI 垃圾内容（brainrot）*，包括 **Spotify** 和 **Dreambees AI**。
- **什么时候能有 **AI 超分****?**: 一位成员感叹目前缺乏对《奉子成婚》（Al Bundy）等老剧的 **AI 超分**（如 16:9 版本），推测 AI 可以插值缺失的细节，但这可能会破坏艺术意图。
   - 另一位成员表示，像《奉子成婚》这样的剧不需要透视，因为*当时没人关心透视之类的事，他们只要效果达标就行*，而且 SD（标清）内容在高画质屏幕上看起来很糟糕。
- ****Zai** 发布 **GLM-Image** 模型**: **Zai** 发布了名为 [**GLM-Image**](https://github.com/zai-org/GLM-Image) 的新图像模型，可以在其[博客](https://z.ai/blog/glm-image)上阅读相关信息。
   - 一位成员对 Semantic VQ 及其实现方式表示感兴趣，好奇是否使用了类似 Vision Transformer 的架构。
- ****Nous Chat** 遭遇不稳定性**: 一位成员询问回复中断或语言切换等问题对于免费版的 **Nous Chat** 是否正常。
   - 一名开发者回应称，可能是服务提供商（provider）再次出现了不稳定性。


  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1460926005404504267)** (12 messages🔥): 

> `Hermes 4, LLM Attention Drift, Context Length Degradation, Context Rot, TPU Training` 


- **Hermes 4 在超过 25k Tokens 时出现注意力漂移**：一位成员注意到，一旦上下文超过 **25,000 token 标记**，**Hermes 4** 的注意力往往会发生漂移，响应变得混乱。
   - 他们在聊天中添加了指令来重新引导 **LLM**，利用其更倾向于关注近期输入的特性。
- **上下文长度衰减是不可避免的**：有人指出，超过特定阈值的性能衰减是不可避免的，即使是像 **Gemini** 这样拥有 **1M+ context** 窗口的顶尖模型也是如此。
   - 模型通常在基本上为 **0 context** 的情况下进行优化和基准测试，因此在长上下文链中出现衰减是预料之中的。
- **“Context Rot” 困扰着 LLM**：一位成员提到了 **Context Rot**（上下文腐烂）现象，即 **LLM** 的性能随着上下文长度的增加而下降。
   - 据报道，将允许的窗口限制在 **少于 20k tokens** 缓解了这一问题。
- **使用 TPU 训练 LLM - 有人尝试过吗？**：一位成员询问是否有人有使用 **TPUs** 训练模型的经验。
   - 在给定的消息中，没有人对这个问题做出回应。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1460684030252023890)** (51 messages🔥): 

> `Manus Credit Usage, Manus Support, Manus Similarweb Integration, Account Blocked, Developer Hiring` 


- **Manus 用户报告新功能导致积分消耗过度**：多名用户报告称，新的 **Manus x Similarweb 集成** 功能导致积分消耗极高，一名用户报告在不到一分钟内消耗了 **5000 credits**，另一名用户报告通过此[链接](https://manus.im/share/wQ71wRcDWyDTQpH26bZP7v)在 **15 秒** 内消耗了 **2591 credits**。
   - 用户对缺乏预警表示沮丧，并建议实施 **safeguards** 或 **credit caps** 以防止此类意外激增，并建议通过展示广告来赚取积分。
- **Manus 支持响应延迟激怒用户**：用户在获得 Manus 支持方面正经历显著延迟，一名用户在转接到人工客服后等待了 **8 小时**，其他用户报告了多条未回复的消息和邮件。
   - 成员建议 Manus 应提供更清晰的关于支持可用性的沟通，例如发布 **运营时间** 或重定向到电子邮件，而不是让用户无限期等待。
- **账户被封禁，用户寻求解释**：一名用户正在为其 **账户被封禁** 寻求解释，强调需要访问其代码以用于常规编程用途。
   - 另一名用户暗示该被封禁用户可能在平台上进行了某些违规操作。
- **开发者在 Manus 社区寻求工作**：一名用户正在寻求开发工作，为“超酷的项目”提供技能支持，并将人们引导至 <#1453385851336790148> 以提供和寻找开发工作。
   - 另一名用户也表达了同样的看法，询问如何将自己的经验贡献给社区。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1460689291289169922)** (18 messages🔥): 

> `LLM Detection Models, Pangram LLM Detection, In-Person Meetups, LLM Extraction analysis consequences` 


- **社区酝酿线下聚会**：成员们讨论了社区举办线下聚会的可能性，建议将**纽约 (NYC)** 或**旧金山 (SF)** 等大城市作为潜在地点。
   - 有人指出，虽然读书会的参与度很高，但常规的线下活动可能需要向更广泛的受众进行宣传，类似于 **Cohere** 的定期活动和 Zoom 会议。
- **深入探讨 LLM 提取分析的影响**：一位成员询问了 **LLM 提取分析**的法律影响，并引用了一项研究，该研究观察到 LLM 会复制原著中的角色名称、情节和主题。
   - 该成员担心，由于这项研究的技术性质，很容易被*误解和误用*。
- **寻找 LLM 生成文本分类器**：一位成员寻求推荐一种能够识别 LLM 生成文本的小型分类器模型，旨在评估最近一次抓取后网络上合成内容的流行程度。
   - 另一位成员建议使用为 **Speculative Decoding** 训练的 **Drafter 模型**，但指出这通常是特定于模型的，可能需要通过集成 (Ensembling) 来获得更好的泛化能力。
- **Pangram 检测准确性存疑**：成员们讨论了使用 **Pangram.com** 进行 LLM 生成文本检测的情况，但对处理 GB 级文本的成本以及 **Hugging Face** 上开源替代方案的准确性表示担忧。
   - 有人建议根据 Pangram 详细介绍其检测方法的学术论文 ([https://arxiv.org/abs/2402.14873](https://arxiv.org/abs/2402.14873)) 构建自定义分类器。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1460717133511262270)** (15 messages🔥): 

> `SlopCodeBench released, Lazy Agents, ICLR workshop submission, Optical Computers and RTL` 


- **SlopCodeBench 揭示 Agent 的懒惰问题**：新博客文章和 [SlopCodeBench](https://github.com/SprocketLab/slop-code-bench) 基准测试发布，重点展示了 **Agent** 在解决被分解为迭代检查点的大型编程问题时，可能会做出**糟糕的早期设计决策**。
   - SlopCodeBench 旨在成为一个社区驱动的基准测试，欢迎关于添加新问题的反馈，并希望与 Harbor 格式集成以方便使用。
- **尽管有简化提示，Agent 仍无法泛化代码**：尽管指令要求模型在产出可运行的实现后进行代码简化和集成，但 **Agent** 往往无法正确地泛化代码以支持新的需求用例。
   - 据一位成员称，所有的提示词都会导致解决测试的整体性能下降，且成本增加约 1.5-2 倍；博客中讨论的是简单的“直接求解”方法。
- **ICLR Workshop 关注 SlopCodeBench**：一位成员建议将博客文章转化为 [该 ICLR Workshop](https://sites.google.com/view/icbinb-2026) 的投稿（截止日期 1 月 31 日），并表示愿意提供协助。
   - 还有人指出，一个代码基准测试不应该依赖繁重的提示工程 (Prompt Engineering) 和脚手架 (Scaffolding) 来获得体面/良好的性能，但这种基础方法也是合理的。
- **光子计算团队招募？**：一位成员可能在未来 6 个月内寻求组建团队，致力于**光子计算机 (Optical Computers)** 或**非浮点 RTL** 的研究。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1461067878185828568)** (2 messages): 

> `LLM Understanding Framework, Global CoT Analysis` 


- **提议 LLM 理解分层框架**：一篇新论文提出了一个思考 **LLM** 理解能力的分层框架，综合了迄今为止最相关的发现（[论文链接](https://arxiv.org/abs/2507.08017)）。
- **全局 CoT 分析发现模式**：在关于**全局 CoT 分析 (Global CoT Analysis)** 的 [LessWrong 帖子](https://www.lesswrong.com/posts/q9g9zuudd3Pvw2cbj/global-cot-analysis-initial-attempts-to-uncover-patterns-1)中，描述了揭示模式的初步尝试。


  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1460696581136187726)** (3 messages): 

> `NCCL hangs during training, H200 node configuration, Model parallelism` 


- **NCCL 卡死困扰多节点训练下的 8B 模型**：一位工程师报告称，在多节点上训练 **8B 模型**时会出现 **NCCL 卡死（hangs）**，而在相同的设置下，使用 **H200 节点**训练 **1B 模型**则能成功运行。
   - 该问题专门发生在多节点配置中，单节点训练对两种模型均正常，使用的是 **1** 的 batch size 和 **1** 的梯度累加步数（gradient accumulation steps）。
- **1B 和 8B 模型之间的配置调整**：工程师分享了 **1B** 和 **8B** 模型之间的配置差异，主要在于层数（**16 vs 36**）、隐藏层大小（**2048 vs 4096**）以及中间层大小（**24576 vs 43008**）。
   - 两个模型的配置文件可以在 [此 gist 链接](https://gist.github.com/aflah02/cdd1cd3dfc73ff1cf7f6bb10ee36929c) 找到，显示卡死的日志可以在 [此处](https://gist.github.com/aflah02/821f464685cfe10dbf8f549d9d477e2d) 找到。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1460742072536535082)** (11 messages🔥): 

> `Lucid Coding, Vibe Coding, Delegation of Expertise with Generative AI, Bus-Factor and LLMs` 


- **垃圾邮件诈骗者已处理**：用户举报了 <@139426008976588801> 的 *SpamingScammers* 行为，`.wavefunction` 确认该问题已得到处理。
- **Lucid Coding 术语受到关注**：一位用户在 [fxtwitter](https://fxtwitter.com/i/status/2011137879112908870) 上分享了一条关于 **lucid coding** 的推文，另一位用户表达了对该术语的喜爱。
   - 该用户解释说，**lucid coding**（清晰编码）一词更能描述大多数高效开发者和设计师在使用生成式 AI 时所做的事情。
- **Vibe Coding 定义出现**：成员们讨论了 **vibe coding** 的定义，一位用户将其定义为使用生成式 AI 却不理解其生成的代码。
   - 另一位成员强调了理解并在 AI 卡壳时能够修复代码的重要性，并指出如果你做不到这一点，那就是 vibe coding。
- **LLM 是关键软件依赖**：小组讨论了专业知识外包（delegation of expertise）的风险，特别是对于生成式 AI，一位用户将 **LLM** 比作关键库依赖项。
   - 该用户建议，如果你在没有 **LLM** 的情况下无法开展项目，你应该将其视为 *那个无论要求加薪多少你都无法解雇的员工*。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1460689303314104351)** (17 messages🔥): 

> `Bayesian vs Frequentist Statistics, FDA Corruption, China AI Anthropomorphism, Qwen DeepPlanning Dataset` 


- **贝叶斯统计：跨越还是保留？**：成员们辩论了向 **贝叶斯统计（Bayesian statistics）** 的转变是否是一个重大变化，一些人认为 *这并不像你想象的那么大*，因为 **频率派（frequentist）** 和 **贝叶斯统计** 都使用相同的技术，如线性回归和逻辑回归。
   - 其他人则反驳说，尽管它们使用相同的公式，但对 **先验（prior）**、**后验（posterior）** 和 **干预（intervention）** 的解释有显著不同，并引用了 [概率解释](https://en.wikipedia.org/wiki/Probability_interpretations)。
- **贝叶斯 FDA 腐败：零先验？**：对话触及了 **贝叶斯方法** 可能 *更有助于用统计数据撒谎* 的可能性，引发了对临床试验中欺瞒和腐败的担忧。
   - 一位成员开玩笑地给观察到的 **贝叶斯 FDA 腐败** 分配了一个 **零先验（zero prior）**，而另一位则暗示了过去的监管腐败，提到了阿片类药物危机。
- **中国对 AI 拟人化的处理方法**：一位成员分享了一篇关于中国国家互联网信息办公室 ([CAC.gov.cn](https://www.cac.gov.cn/2025-12/27/c_1768571207311996.htm)) [对 AI 拟人化处理方法](https://www.luizasnewsletter.com/p/chinas-approach-to-ai-anthropomorphism) 的文章链接。
   - 有人注意到 Chrome 的页面翻译功能在这篇文章上表现良好。
- **Qwen DeepPlanning 数据集：现已关闭**：提到了 HuggingFace 的 **Qwen/DeepPlanning** 数据集，但指向该 [HuggingFace 数据集](https://huggingface.co/datasets/Qwen/DeepPlanning) 的链接目前已失效。
   - 这一关闭消息在一条 [推文](https://x.com/HuggingPapers/status/2011292800432619865) 中被提及。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1460835433628958772)** (2 messages): 

> `Mojo Docs, NotebookLM, llms.txt` 


- **Mojo 文档寻求者咨询 NotebookLM**：有成员咨询如何获取完整且最新的官方 **Mojo docs**（PDF 或 Markdown 格式），以便在 **NotebookLM** 中使用。
   - 另一位成员建议使用 *llms.txt* ([https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt](https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt)) 为 **LLMs** 提供文档。
- **llms.txt 与 NotebookLM 的集成**：一位成员提议使用 *llms.txt* 文件将 **Mojo documentation** 与 **NotebookLM** 及其它 **LLMs** 进行集成。
   - 该建议引用了 Modular 官方关于编程助手的文档，特别是如何使用 *llms.txt* 格式向 **LLMs** 提供文档。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1460815670701719838)** (5 messages): 

> `Qwen3-VL, MoE Implementation, Contributor Guide` 


- **Qwen3-VL 的 MoE 实现遭到质疑！**：有成员询问为何 **Qwen3-VL** 仅有 **MoE** 实现，并建议复用 [qwen3vl_moe](https://github.com/modular/modular/tree/main/max/python/max/pipelines/architectures/qwen3vl_moe) 的代码来支持像 [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) 这样的 Dense 模型。
   - 他们表示愿意提交一个 **PR** 来解决这个问题。
- **号召提交 PR 以丰富 MAX！**：一位成员对 **PR** 提议给出了积极回应，理由是目前缺乏贡献者，难以跟上生态系统的发展步伐。
   - 他们还指出了最近更新的 [contributor guide](https://github.com/modular/modular/commit/571ae7132c465f155d50f7bbe7cf77064a02e9bf) 以供参考。
- **PR 已上线！**：一位成员确认他们不确定当前的实现是否是有意为之，随后跟进表示一个 [PR](https://github.com/modular/modular/pull/5776) 已经提交。
   - 该 **PR** 目前正在等待审核。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1460759556501016681)** (1 messages): 

> `Glama Ranking System, Server Usage Metrics, Ranking Abuse Claims` 


- **Glama 创始人回应排名作弊指控**：**Glama** 的创始人对排名作弊的指控做出了回应，指出其排名是基于 **server usage metrics**（服务器使用指标）。
   - 他们否认了解任何作弊行为，并鼓励用户通过私信提供反馈。
- **Glama 排名系统说明**：创始人澄清说，**Glama's rankings** 是由 **server usage metrics** 决定的。
   - 这一解释旨在解决有关排名系统公平性和有效性的疑虑。


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1460702731798053165)** (5 messages): 

> `Tasks spec implementation, Inspector PR for adding tasks` 


- **社区寻求 Tasks 规范的客户端示例**：成员们正在寻找实现了 **Tasks spec** 的客户端应用，以了解 **UI** 实现模式。
   - 一位成员提到在他们的客户端中实现了 tasks，并希望能看到其他人是如何实现 **UI** 的。
- **Inspector 通过 PR 增加对 Tasks 的支持**：一位成员正在提交一个 **PR**，旨在为 **Inspector** 添加 tasks 功能。
   - 另一位成员向 *server-everything* 提交了一个 **PR**，用于模拟长时间运行的任务，这意味着服务器和 inspector 的下一个版本都将支持此功能。
- **glama.ai/mcp/inspector 仍处于极早期版本**：一位成员开始开发 [glama.ai/mcp/inspector](https://glama.ai/mcp/inspector)，目前它还是一个非常早期的版本。
   - 目标是让 inspector 覆盖每一项功能，该工具在内部用于 **e2e testing**。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1460930642668687360)** (6 messages): 

> `Kimi K-2 视觉功能版本, Kimi CLI 问题, Kimi 新模板, UI 实现` 


- **Kimi 重构界面**：一位用户对界面重构表示了积极的反响，并分享了 [X.com](https://x.com/jukan05/status/2011254536258945104?s=20) 的链接。
- **Kimi K-2 视觉版即将发布？**：一名成员对新界面感到兴奋，并询问*这是否意味着我们终于能看到 **K-2 vision-capable release**（具备视觉功能的版本）？*
   - 他们指出 *K1.5 被列为 legacy model（遗留模型），但它才是拥有视觉功能的那个*。
- **Kimi CLI 的使用困扰**：一名成员提到了一些关于 **Kimi CLI** 的问题，但赞扬了 Slides 中的新模板。
   - 然而，该成员认为对于 Visual 类型的 UI 实现有限，且没有针对 Adaptive 的实现，并进一步声称*以前有更多模板提供更多选择*。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1460700888200118449)** (5 messages): 

> `AI 辅助代码生成, Replit, DSPY OS, DSPY 框架` 


- **AI 平台使用代码生成**：如 **Replit** 和 **DSPY OS** 等平台利用 **AI models** 来辅助编程任务，从而提高生产力。
   - 一名成员询问了这些平台与 **DSPy** 的关系。
- **Replit 是闭源的，DSPY 是框架**：一名成员指出 **Replit** 是闭源的，而 **DSPY** 是一个框架，并询问是否有基于 **DSPY** 构建的类似 **Replit** 的项目。
   - 澄清说明目前还没有直接基于 **DSPY** 构建的类 Replit 项目，因为 **DSPY** 更多是一个框架而非平台。
- **什么是 DSPY OS？**：一名成员询问关于 **DSPY OS** 的信息，因为他找不到任何相关资料。
   - 现场未提供进一步的说明信息。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1460986150507511984)** (2 messages): 

> `DM 命令, Aider 工具化` 


- **发起 DM 命令**：一位用户以简单的 "yes sir" 回复了 `@0xhellno` 的私信请求。
   - 随后，另一位用户 `txn545` 表示打算使用 "dm" 命令发送私信。
- **Aider 工具化需求**：一位用户请求对 `aider` 工具化进行改进。
   - 该用户询问 `aider` 是否支持在与最初添加代码时不同的目录中编辑代码。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1460695368676278417)** (2 messages): 

> `Gemini 的 OAuth 登录, CLIProxyAPI` 


- **OAuth 登录可开启 Gemini 的更高额度**：一名成员询问在使用 aider 时，是否可以为 **Gemini model** 使用 **OAuth login**，理由是可能会有更高的限制额度。
   - 另一名成员建议 **CLIProxyAPI** 是实现该功能的最佳基础，并提到已经有一些可用的 Wrapper。
- **以 CLIProxyAPI 为基础**：由于额度更高，一名成员推荐使用 **CLIProxyAPI** 作为实现 Gemini OAuth 登录的基础。
   - 他们还提到有多个针对 **CLIProxyAPI** 的封装器，这可以简化集成过程。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1460769810316267601)** (1 messages): 

> `外联提示工程, Clay + AI 外联工作流, 大规模个性化消息, Apollo, Attio, n8n 集成` 


- **外联提示工程研讨会宣布举行**：一场关于 **Prompt Engineering for Outreach** 的研讨会将教授如何构建 **Clay + AI outreach workflow**，以实现 **personalized messages at scale**（大规模个性化消息）。
   - 该研讨会承诺实现 **40%+ 的接受率**和 **18%+ 的回复率**，将于 **1 月 14 日（周三）**和 **1 月 17 日（周六）**举行（[链接](https://luma.com/jt1vr0u5)）。
- **AI 外联工作流详解**：研讨会将涵盖端到端的 **AI outreach workflow**，包括目标识别、潜在客户列表创建、数据丰富化、消息生成和跟踪。
   - 研讨会还将包含 **Clay 演示**，并讨论 **Apollo, Attio, 和 n8n 集成**。
- **提供可重用的外联资源**：参与者将获得一个**可重用的工作流**、**可复制粘贴的提示词**以及一个**简单的 QA 清单**，以辅助其外联工作。
   - 该研讨会旨在提供初学者友好的指导，名额有限。


  

---

### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1461065370063339562)** (1 条消息): 

> `GPT-5.2-Codex Release, Windsurf Discounts, Agentic Coding Model` 


- **GPT-5.2-Codex 登陆 Windsurf！**：来自 OpenAI 的最新 Agentic 编程模型 **GPT-5.2-Codex** 现已面向所有 Windsurf 用户开放，并提供四个推理努力（reasoning effort）级别。
   - 查看 [OpenAI 的博客文章](https://openai.com/index/introducing-gpt-5-2-codex/) 了解更多详情。
- **Windsurf 推出折扣活动以示庆祝**：Windsurf 正在针对 **GPT-5.2-Codex** 提供限时折扣，其中 **low** 和 **medium** 努力级别按 **0.5x** 计费，**high** 按 **1x** 计费，**xhigh** 按 **2x** 计费。
   - 建议用户更新并重新启动 Windsurf，以体验新模型并享受新价格。