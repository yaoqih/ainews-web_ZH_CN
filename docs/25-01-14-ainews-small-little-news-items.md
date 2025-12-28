---
companies:
- ollama
- cohere
- togethercompute
- openbmb
- qwen
- langchain
- openai
date: '2025-01-15T02:19:30.206234Z'
description: '**Ollama** 通过集成 **Cohere 的 R7B** 模型增强了其功能，该模型针对 **RAG**（检索增强生成）和**工具使用任务**进行了优化；同时，Ollama
  发布了 **v0.5.5 版本**，带来了质量更新和全新的引擎。**Together AI** 推出了 **Llama 3.3 70B 多模态模型**，提升了推理和数学能力；而
  **OpenBMB** 发布的 **MiniCPM-o 2.6** 在视觉任务上的表现超越了 **GPT-4V**。


  有关**过程奖励模型 (PRM)** 的见解被分享，旨在提升 **LLM（大语言模型）的推理能力**，同时 **Qwen2.5-Math-PRM** 模型在数学推理方面表现卓越。**LangChain**
  发布了 **ChatGPT Tasks** 的测试版，支持预约提醒和摘要，并推出了用于邮件辅助的开源**环境智能体 (ambient agents)**。**OpenAI**
  也向 Plus、Pro 和 Teams 用户推出了 **ChatGPT 中的“任务” (Tasks)** 功能，用于安排自动化操作。


  AI 软件工程正飞速发展，预计在 18 个月内将达到人类水平。关于 **LLM 缩放法则 (scaling laws)** 的研究强调了幂律关系和改进速度进入平台期的趋势，而
  **GANs**（生成对抗网络）正迎来复兴。'
id: 36f8e198-8bb3-4914-abea-5eabbf2a49ca
models:
- r7b
- llama-3-70b
- minicpm-o-2.6
- gpt-4v
- qwen2.5-math-prm
original_slug: ainews-small-little-news-items
people: []
title: '根据语境，可以翻译为：


  1. **简讯**（最常用的正式表达）

  2. **零星的小新闻**（强调零散、琐碎）

  3. **短讯**

  4. **小条新闻**


  如果是在描述报纸或网页上的小板块，也可以译为：**新闻点滴**。'
topics:
- rag
- tool-use-tasks
- quality-of-life
- new-engine
- multimodality
- improved-reasoning
- math-capabilities
- process-reward-models
- llm-reasoning
- mathematical-reasoning
- beta-release
- task-scheduling
- ambient-agents
- email-assistants
- ai-software-engineering
- codebase-analysis
- test-case-generation
- security-infrastructure
- llm-scaling-laws
- power-law
- plateauing-improvements
- gans-revival
---

<!-- buttondown-editor-mode: plaintext -->**耐心是你所需要的一切。**

> 2025年1月13日至1月14日的 AI 新闻。我们为你检查了 7 个 subreddits、[433 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 32 个 Discord 服务器（219 个频道，2161 条消息）。预计节省阅读时间（按 200wpm 计算）：**256 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

[ChatGPT Tasks 发布](https://techcrunch.com/2025/01/14/chatgpt-now-lets-you-schedule-reminders-and-recurring-tasks/)。[Cursor 完成了 B 轮融资](https://x.com/sarahdingwang/status/1879279307119608142)。[Sakana 宣布了一项针对 LoRAs 的精美改进，但性能提升较小](https://x.com/hardmaru/status/1879331049383334187)。Hailuo 发布了一个[巨大的 456B MoE 模型](https://www.reddit.com/r/LocalLLaMA/comments/1i1a88y/minimaxtext01_a_powerful_new_moe_language_model/)，类似于 Deepseek v3。

虽然没有值得作为头条新闻的内容，但这些都是不错的增量进展。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有回顾均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**模型发布与更新**

- **Ollama 模型增强**：[@ollama](https://twitter.com/ollama/status/1879216139542434055) 宣布加入 Cohere 的 **R7B**，这是其 **Command R 系列**中最小的模型，针对 **RAG** 和 **tool use 任务**进行了优化。此外，[@ollama](https://twitter.com/ollama/status/1879212554435911978) 发布了 **Ollama v0.5.5**，包含多项**易用性更新**并迁移到了**新引擎**。[@ollama](https://twitter.com/ollama/status/1878950123625214081) 还重点介绍了即将在旧金山举行的 **2025 Ollama 见面会**，吸引了 **31,592 次曝光**，引起了广泛关注。

- **Together AI 和 OpenBMB 模型**：[@togethercompute](https://twitter.com/togethercompute/status/1879231968434684254) 推出了 **Llama 3.3 70B**，这是一款在 **Together AI** 上免费提供的**多模态模型**，具有**更强的推理**和**数学能力**。与此同时，[@OpenBMB](https://twitter.com/_philschmid/status/1879163439559389307) 发布了 **MiniCPM-o 2.6**，这是一个 **8B 参数的多模态模型**，在视觉任务上**超越了 GPT-4V**。

- **Process Reward Models 和 Qwen 进展**：[@_philschmid](https://twitter.com/theTuringPost/status/1879101437663154194) 分享了关于 **Process Reward Models (PRM)** 的见解，强调了它们在增强 **LLM 推理**方面的作用。**Qwen 团队**也展示了他们的 **Qwen2.5-Math-PRM** 模型，在**数学推理**方面表现出卓越性能。

- **LangChain 和 Codestral 更新**：[@LangChainAI](https://twitter.com/LangChainAI/status/1879235381545386309) 发布了 **tasks 的 Beta 版本**，允许 **ChatGPT** 处理**未来任务**，如提醒和摘要。[@dchaplot](https://twitter.com/dchaplot/status/1878952042498334944) 发布的 **Codestral 25.01** 在 **LMSys Copilot Arena** 中并列第一，展示了较之前版本显著的**性能提升**。

**AI 功能与工具**

- **OpenAI Task 推出**：[@OpenAI](https://twitter.com/OpenAI/status/1879267276291203329) 宣布推出 **Tasks** 功能，允许用户为 **ChatGPT** 安排操作，例如**每周新闻简报**和**个性化健身计划**。该功能目前处于 **Plus、Pro 和 Teams 用户的 Beta 阶段**，最终将面向所有 **ChatGPT 账号**开放。

- **Ambient Agents 和邮件助手**：[@LangChainAI](https://twitter.com/LangChainAI/status/1879218070008570213) 推出了一款**开源邮件助手 Agent**，这是其全新的 **"Ambient Agents"** 范式的一部分。这些 Agent **始终处于活动状态**，处理诸如**邮件分类**和**草拟回复**等任务，在无需传统 **UX 界面**的情况下提高**生产力**。

- **AI 软件工程进展**：[@bindureddy](https://twitter.com/bindureddy/status/1879017155423080482) 讨论了 **AI 软件工程师**的快速成熟，强调了它们在**代码库分析**、**测试用例生成**和**安全基础设施**方面的能力，并预测 AI 将在未来 **18 个月**内达到 **SWE 能力**。

**AI 研究与论文**

- **LLM Scaling Laws**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1878929929611448588) 深入探讨了 **LLM Scaling Laws**，解释了 **算力 (compute)**、**模型大小**和**数据集大小**之间的**幂律关系**。研究强调，虽然 **测试损失 (test loss)** 随规模扩大而降低，但**改进会趋于平缓**，这挑战了 **AI 指数级进步**的观点。

- **GANs 复兴**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1879111514210402681) 报道了通过论文 "**The GAN Is Dead; Long Live the GAN! A Modern GAN Baseline**" 实现的 **GANs** 复兴，重点介绍了 **R3GAN architecture** 及其在 **FFHQ** 和 **CIFAR-10** 等基准测试上优于某些 **diffusion models** 的 **superior performance**。

- **Multimodal RAG 和 VideoRAG**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1878932305177453037) 介绍了 **VideoRAG**，这是 **multimodal RAG** 的扩展，可以 **real-time** 检索 **videos**，利用 **visual** 和 **textual data** 来增强 **response accuracy**。

- **Tensor Product Attention**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1879091791753843064) 展示了 "**Tensor Product Attention (TPA)**" 机制，它将 **inference-time cache size** 降低了 **10x**，并在 **performance benchmarks** 中优于之前的 **attention methods**（如 **MHA** 和 **GQA**）。

**AI 社区与活动**

- **Ollama Meetup 和社区参与**：[@ollama](https://twitter.com/ollama/status/1878950123625214081) 推广了在 **San Francisco** 举办的 **2025 Ollama meetup**，促进了 **AI enthusiasts** 之间的 **community engagement**。此外，[@gdb](https://twitter.com/gdb/status/1879059072135414236) 等人通过 **joining initiatives** 和 **hiring announcements** 鼓励 **community participation**。

- **LangChain AI Meetup**：[@LangChainAI](https://twitter.com/LangChainAI/status/1879235381545386309) 在 **San Francisco** 组织了一场 **evening meetup**，邀请了 **@hwchase17** 和 **Bihan Jiang** 等 **industry leaders** 进行 **fireside chat**，重点讨论了 **deploying production-ready AI agents**。

- **招聘公告**：多条推文（包括来自 [@WaveFormsAI](https://twitter.com/alex_conneau/status/1879240315342880826) 和 [@LTIatCMU](https://twitter.com/gneubig/status/1879261395646398872) 的推文）分享了 **software engineers** 和 **research positions** 的 **job openings**，涉及 **multimodal LLMs**、**full-stack development** 和 **AI safety** 等领域。

**AI 行业新闻与政策**

- **AI 政策与经济影响**：[@gdb](https://twitter.com/gdb/status/1879059072135414236) 发布了一份 **Economic Blueprint**，概述了优化 **AI benefits**、增强 **national security** 并推动 **U.S.** **economic growth** 的 **policy proposals**。与此同时，[@NandoDF](https://twitter.com/NandoDF/status/1878949902383985105) 提倡在 **UK** **removal of non-compete clauses**，以 **boost AI competitiveness**。

- **AI 劳动力转型**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1878955184459518297) 强调了 **AI Engineers and Consultants** 正在成为 **top jobs on the rise**，这是由于 **AI's transformative impact** 跨越各行各业，强调了在这一领域 **gaining expertise** 的重要性。

- **中美 AI 竞争**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1879251029084225966) 等人讨论了 **China vs. US** 之间日益激烈的 **intensifying AI competition**，强调了 **geopolitical implications** 和 **race for AI dominance**。

- **数据中心营收预测**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1878944624477376920) 预测 **FY2026** 的 **data center revenue** 为 **$236 billion**，比 **market consensus** 增长了 **28%**，表明 AI 领域 **growing infrastructure investments**。

**梗/幽默**

- **编程与每日提醒**：[@hkproj](https://twitter.com/hkproj/status/1878954137171415346) 分享了一个 **daily reminder**：**eat veggies** 并 **code triton kernels**，将 **health tips** 与 **coding humor** 结合在一起。

- **AI 与个人生活笑话**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1878915601244336450) 幽默地评论了 **a model's consciousness**，调侃在 **examining AI capabilities** 时需要 **better epistemology**。

- **开发者梗**：[@nearcyan](https://twitter.com/nearcyan/status/1878915854269890911) 发布了一个 **"two space" meme**，引起了开发者对 **coding standards** 内部梗的共鸣。

- **关于 AI Agents 的幽默看法**：[@bindureddy](https://twitter.com/bindureddy/status/1878983646532825320) 调侃 **AI agents** 接管工作任务，思考 **working would become obsolete**。

- **通用技术幽默**：[@saranormous](https://twitter.com/saranormous/status/1879170235019952575) 调侃了关于生孩子的 **reading readiness**，将 **life advice** 与 **humorous skepticism** 交织在一起。

---

# AI Reddit Recap

## /r/LocalLlama Recap

**主题 1. Qwen 的 Math Process Reward Models 与创新**

- **Qwen 在其最近的数学模型中发布了 72B 和 7B 的过程奖励模型 (PRM)** ([Score: 145, Comments: 16](https://reddit.com/r/LocalLLaMA/comments/1i0ysa7/qwen_released_a_72b_and_a_7b_process_reward/))：**Qwen** 发布了两个新的**过程奖励模型 (PRM)**，即 **Qwen2.5-Math-PRM-7B** 和 **Qwen2.5-Math-PRM-72B**，旨在通过识别和纠正中间错误来增强大语言模型 (LLM) 的数学推理能力。这些模型在 **Best-of-N (BoN)** 评估中表现强劲，并在 **ProcessBench** 的错误识别方面表现出色，详情见其题为 *The Lessons of Developing Process Reward Models in Mathematical Reasoning* 的论文 ([arXiv:2501.07301](https://arxiv.org/abs/2501.07301))。
  - **Qwen2.5-Math-PRM-72B** 主要用于学术目的，通过提供推理质量和中间步骤的反馈来训练其他模型，而非用于典型的文本生成任务。**Zealousideal-Cut590** 强调在编程、法律和医疗等非数学领域也需要**过程奖励模型 (PRMs)**，以优化测试时计算 (test time compute)。
  - **-p-e-w-** 讨论了跟上新模型快速发布的挑战日益增大，预测即使是无限的互联网连接可能很快也会不够用。**Useful44723** 建议 **Hugging Face** 应该提供种子 (torrent) 链接作为替代下载方式，以管理海量数据。
  - 模型发布的快速节奏备受关注，**-p-e-w-** 指出每周都会出现多个重大的新发布，导致下载队列可能出现饱和。**Caffeine_Monster** 和 **Threatening-Silence-** 对当前的网速是否充足以及未来的潜在限制发表了评论。


- **[MiniCPM-o 2.6：一个 8B 大小、GPT-4o 级别的全能模型 (Omni Model)，可在设备上运行](https://x.com/OpenBMB/status/1879074895113621907)** ([Score: 158, Comments: 29](https://reddit.com/r/LocalLLaMA/comments/1i11961/minicpmo_26_an_8b_size_gpt4o_level_omni_model/))：**MiniCPM-o 2.6** 是一个拥有 **80 亿参数的模型**，声称达到了 **GPT-4o 级别**的性能。它旨在本地设备上运行，增强了各种应用的访问性和可用性。
  - 讨论中充满了对 **MiniCPM-o 2.6** 达到 **GPT-4o 级别性能**这一说法的怀疑，用户认为尽管它具有可访问性和本地运行能力，但在基准测试或能力上仍无法与 GPT-4 相提并论。**AaronFeng47** 和 **Aaaaaaaaaeeeee** 对其性能表示怀疑，认为它与 **GPT-4o** 不在一个水平，并指出在设备上运行它的技术挑战，需要 ≥12GB 内存的设备。
  - 用户们争论小型模型是否能超越像 GPT-4 这样的大型模型，**MoffKalast** 和 **Radiant_Dog1937** 讨论了像 **Gemma 2 9B** 和 **Gemini 1.5 Flash 8B** 这样的小型模型如何在 **Hugging Face 排行榜**上名列前茅，但可能无法匹配 GPT-4 的全面能力。他们认为，虽然小型模型在特定任务中表现良好，但由于参数容量的物理限制，它们无法与大得多的模型的知识和应用能力相匹配。
  - **Many_SuchCases** 分享了 **MiniCPM-o 2.6** 在 **Hugging Face** 上的链接，并对其推理引擎的兼容性提出了疑问，同时讨论还涉及了 **MiniCPM-o 2.6** 的 **MMMU 分数**，该分数为 50.4，而 GPT-4o 为 69.2，这表明存在显著差距。


**主题 2. MiniMax-Text-01：MoE 和长上下文能力**

- **MiniMax-Text-01 - 一个强大的新型 MoE 语言模型，拥有 456B 总参数（45.9B 激活参数）** ([Score: 93, Comments: 48](https://reddit.com/r/LocalLLaMA/comments/1i1a88y/minimaxtext01_a_powerful_new_moe_language_model/)): **MiniMax-Text-01** 是一款新型的 **Mixture-of-Experts (MoE) 语言模型**，拥有 **4560 亿总参数**，其中每个 token 激活 **459 亿参数**。它采用结合了 **Lightning Attention**、**Softmax Attention** 和 MoE 的混合架构，并使用了 **Linear Attention Sequence Parallelism Plus (LASP+)** 和 **Expert Tensor Parallel (ETP)** 等先进的并行策略，使其在推理过程中能够处理高达 **400 万个 token**。
  - **硬件要求与本地运行**：运行 **MiniMax-Text-01** 需要大量的 RAM，建议范围从 **基础操作的 96GB** 到 **更具实用性的 384/470GB**。尽管其体量巨大，但 **Mixture-of-Experts (MoE)** 架构可能允许通过将激活的专家卸载到 GPU 来实现更可控的本地执行，类似于 **deepseek v3**。
  - **许可与可访问性**：该模型的限制性许可引发了关注，特别是其对使用输出结果来改进其他模型的限制以及分发要求。尽管有这些限制，它仍然开放商用，但一些用户对其强制执行力表示怀疑，并将其与用于军事应用的 **Apache 2.0** 进行了类比。
  - **性能与能力**：该模型处理 **高达 400 万个 token** 的能力被强调为开源长上下文处理领域的一项重大成就。其 **linear 和 softmax attention** 层的混合架构，结合先进的并行策略，被认为与仅依赖 softmax attention 的模型相比，有可能降低上下文要求并增强检索和外推能力。


**主题 3：LLM 驱动的新开源倡议带来的灵感**

- **今天我成立了自己 100% 致力于开源的组织——这一切都要归功于 LLM** ([Score: 141, Comments: 44](https://reddit.com/r/LocalLLaMA/comments/1i148es/today_i_start_my_very_own_org_100_devoted_to/)): 该帖子的作者拥有 **生物学** 背景，他成立了一个完全致力于 **开源** 项目的新组织，并将这一成就归功于 **Large Language Models (LLMs)** 的影响以及 **r/LocalLlama** 社区的支持。他们向社区表达了感谢，并强调了开源生态系统在支持他们从生物学转向这一新事业中的重要性。
  - **自筹资金与财务挑战**：包括 **KnightCodin** 和 **mark-lord** 在内的几位评论者讨论了自筹资金（bootstrapping）创业的挑战和好处。**Mark-lord** 强调通过降低生活成本来有效管理财务，避免投资者的压力，并分享了克服冒充者综合征和财务障碍的个人经历。
  - **社区支持与鼓励**：社区对作者的创业表示了强烈的支持和鼓励，**Silent-Wolverine-421** 和 **NowThatHappened** 等用户表达了祝贺。“This is the way”的情绪得到了多位评论者的共鸣，凸显了追求独立开源项目的共同价值观。
  - **建议与工具**：**Mark-lord** 为那些向 AI 转型的人分享了实用建议，推荐使用 **Claude 3.5** 处理各种任务，并建议使用 **Cursor** 以获得无限请求。他们邀请通过私信进行进一步的讨论和交流，体现了支持他人进行类似转型的意愿。

- **为什么他们要免费发布开源模型？** ([Score: 283, Comments: 166](https://reddit.com/r/LocalLLaMA/comments/1i11hre/why_are_they_releasing_open_source_models_for_free/))：尽管涉及高昂成本，**开源 AI 模型**仍被免费发布，因为它们可以推动**社区协作**并**加速创新**。公司或开发者发布这些模型的动机包括获得**声誉**、鼓励**广泛采用**，以及潜在地**刺激改进**，从而使原始创作者受益。
  - 讨论强调，**开源 AI 模型**通过使模型成为广泛使用的标准，帮助 **Meta** 和 **Google** 等公司巩固市场主导地位，从而降低成本并吸引人才。这一策略被比作 **Google** 的 **Android** 和 **Microsoft** 的 **GitHub**，强调了社区参与和心智占有率（mindshare）带来的长期利益，而非直接从模型本身获取收入。
  - 几条评论认为，免费发布这些模型可以打击竞争对手，并为新玩家设置准入门槛。这可以被视为一种**“焦土策略”**，其目标是用免费资源饱和市场，使他人难以将类似产品货币化，正如在 **Meta** 的 **LLaMA** 和 **GitHub Copilot** 的背景下所讨论的那样。
  - 评论者还指出，**“开源”**标签有时具有误导性，因为许多模型只是**权重开放（open weights）**，而不具备完整的重新训练能力。这种部分开放允许公司从社区反馈和创新中获益，同时仍能保持对其专有技术和战略优势的控制。


**主题 4. RTX Titan Ada 48GB：揭示新 GPU 潜力**

- **RTX Titan Ada 48GB 原型机** ([Score: 52, Comments: 13](https://reddit.com/r/LocalLLaMA/comments/1i0rdx1/rtx_titan_ada_48gb_prototype/))：据推测，**RTX Titan Ada 48GB** 比 **5090** 更具吸引力，潜在价格为 **$3k**。它启用了全部 **144 个 SM**，混合精度训练性能翻倍，并且可能配备了来自 **L40** 的 **Transformer engine**，这与 **4090** 不同。尽管显存带宽较慢，但它提供了 **48GB** 显存、**300W TDP** 以及 **1223.88** 的 **GFLOPS/W**，使其在多卡配置中非常高效。[更多详情点击此处](https://videocardz.com/newz/alleged-nvidia-rtx-titan-ada-surfaces-with-18432-cuda-cores-and-48gb-gddr6-memory-alongside-gtx-2080-ti-prototype)。
  - **显存带宽担忧**：讨论强调了对带宽下降到不到一半的“残酷”担忧，但一些用户认为 **904GB/s** 并不慢，并强调了显存带宽相对于每个 Token 使用的显存容量的重要性。
  - **定价和市场吸引力**：人们对该卡的定价策略持怀疑态度，有人建议以 **$500** 的亏本价销售会更有吸引力。然而，对于优先考虑 Prompt 处理的潜在买家来说，**273GB/s** 的数据被视为一个缺点。
  - **原型与特性**：该卡被确认为一个旧的原型，类似于禁用了 **ECC** 并使用 **GDDR6** 和 **PCIe 4.0** 的 **L40**。它在一年前与 **4090 Ti** 一起被传闻，最近的 **GPU-Z** 截图为其存在提供了一定的可信度。

## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. AGI：营销炒作还是真正的创新？**

- **我们是否实际上离 AGI 还很远，而这一切都只是营销？** ([评分: 161, 评论: 260](https://reddit.com/r/OpenAI/comments/1i0v95g/are_we_actually_far_from_agi_and_this_is_all/))：该帖子质疑 **AGI** 目前是否可以实现，或者关于它的说法仅仅是营销策略。作者认为，尽管 **Transformers** 带来了变革性的影响，但通往 **AGI** 的真正飞跃可能需要 **neuroscience**（神经科学）或 **cognitive science**（认知科学）的突破，以开发出一种能与现有技术互补的新架构。
  - **AGI 定义与怀疑论**：包括 **insightful_monkey** 和 **Deltanightingale** 在内的许多评论者对 **AGI** 的定义展开了辩论，认为虽然像 **o3** 这样的当前 AI 模型在特定领域展示了先进的能力，但它们缺乏作为真正 **AGI** 特征的通用问题解决能力和自主推理技能。共识是 AI 的现状距离实现 **AGI** 还很远，一些人如 **PatrickOBTC** 强调，大部分关于 **AGI** 的讨论都是由营销驱动的。
  - **技术与财务约束**：讨论强调了实现 **AGI** 的技术和财务障碍，**vertigo235** 和 **Deltanightingale** 指出了现有 AI 模型相关的高昂成本和缓慢速度。**JonnyRocks** 指出，**OpenAI** 对 **AGI** 的定义与业务目标挂钩（例如达到 **1000 亿美元**的收入），而非真正的技术里程碑，这表明 **AGI** 主张背后存在财务动机。
  - **进展与未来展望**：虽然像 **ZillionBucks** 这样的人对 **AGI** 的未来保持乐观，但许多其他人持怀疑态度，**TheInfiniteUniverse_** 和 **Economy-Bid-7005** 认为，虽然 **o3** 等模型在特定领域表现良好，但它们缺乏诸如 **recursive learning**（递归学习）等关键要素。**o3-mini** 和 **o3-Full** 的发布备受期待，但是...

---

# AI Discord 回顾

> 由 o1-preview-2024-09-12 生成的摘要之摘要的摘要

**主题 1. 新 AI 模型：Codestral、MiniMax-01 和 DeepSeek V3**

- [**Codestral 模型首次亮相，具备 256k 上下文**](https://mistral.ai/news/codestral-2501)：新的 **Codestral** 模型现在在 **Mistral API** 上免费提供，拥有巨大的 **256k context window**（上下文窗口），被用户描述为“*极快且好用*”。预计它将显著加快大规模代码生成任务的速度。
- [**MiniMax-01 发布具备 4M Tokens 的开源模型**](https://x.com/minimax__ai/status/1879226391352549451)：**MiniMax-01**，包括 **MiniMax-Text-01** 和 **MiniMax-VL-01**，可以处理高达 **400 万 tokens**，超过现有模型 **20–32 倍**。定价设定为 **每百万输入 token 0.2 美元**，**每百万输出 token 1.1 美元**，并在 [Hailuo AI 上提供免费试用](https://hailuo.ai)。
- [**DeepSeek V3 在编程任务中表现优于 Claude**](https://huggingface.co/unsloth/DeepSeek-V3-GGUF)：用户报告称 **DeepSeek V3** 在代码生成和推理方面超越了 **Claude**，尽管在本地运行它需要大量资源——大约 **380GB RAM** 和多个 GPU。它因在复杂任务上“*无可挑剔的推理*”而受到称赞。

**主题 2. AI 工具与 IDE：性能波动与用户创新**

- **Cursor IDE 面临运行缓慢及用户规避方案**：用户在使用 **Cursor IDE** 时遇到了显著的请求缓慢和停机情况，将错误报告过程比作“*幼儿园*”。在监控 [Anthropic's Status](https://status.anthropic.com/) 的同时，一些开发人员使用 [Beyond Compare](https://github.com/sksarvesh007) 创建了脚本来管理代码快照，以应对 Cursor 的问题。
- **Codeium 的 Windsurf 困境与对清晰度的追求**：参与者正在努力解决 **Windsurf** 中导致开发循环的 **AI 生成代码错误**。他们强调使用详细的 **.windsurfrules** 文件，寻求更结构化的方法来优化输出，并参考了 [Codeium Docs](https://docs.codeium.com/supercomplete/overview)。
- **LM Studio 用户对比 Qwen 2.5 和 QwQ 模型**：对 **Qwen 2.5 32B Instruct** 和 **QwQ** 的测试表明，Qwen 提供了更好的代码生成，且回答不那么冗长。用户推荐使用带有 **GGUF 编码**的模型，以便在消费级硬件上获得最佳性能，如 [本地 LLM 建议](https://gist.github.com/shermanhuman/2b9a82df1bab242a8edffe504bb1867c) 中所述。

**主题 3. AI 功能的进展：从任务调度到环境智能体 (Ambient Agents)**

- [**ChatGPT 推出 Task Scheduling 功能**](https://www.theverge.com/2025/1/14/24343528/openai-chatgpt-repeating-tasks-agent-ai)：2025年1月14日，**ChatGPT** 为 **Plus**、**Team** 和 **Pro** 用户推出了新的 **Task Scheduling** 功能。这允许设置一次性和循环提醒，旨在将 ChatGPT 重新定位为主动的 **AI agent**。
- [**Ambient Agents 实现邮件管理自动化**](https://blog.langchain.dev/introducing-ambient-agents/)：一个新的 **AI email assistant** 可以自主分类和起草邮件，减少收件箱过载。在 [Harrison Chase 的公告](https://x.com/hwchase17/status/1879218872727015644)中详细介绍了这一点，它代表了向低干扰、仅需极少监督的 AI 辅助迈进。
- [**提出 Hyper-Connections 以改进神经网络**](https://arxiv.org/abs/2409.19606)：研究人员引入了 **Hyper-Connections** 作为 residual connections 的替代方案，解决了 gradient vanishing 等挑战。早期实验表明，它们达到或超过了现有方法，有可能增强 **language** 和 **vision** 模型。

**主题 4. AI 基础设施：GPU 访问与支持挑战**

- [**Thunder Compute 提供实惠的云端 GPU**](https://thundercompute.com)：**Thunder Compute** 推出 **A100 instances**，价格为 **$0.92/hr**，测试期间另有 **$20/month** 的免费额度。通过简单的 CLI (`pip install tnr`)，它简化了 GPU 工作流，旨在让高性能计算更加触手可及。
- **Unsloth AI 仅限于 NVIDIA GPUs，AMD 用户仍在等待**：用户发现 **Unsloth** 目前仅支持一个 NVIDIA GPU，这让希望获得 AMD 支持的用户感到困惑和沮丧。参考 [SHARK-AI 的 AMD 优化指南](https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md#glossary)，突显了社区对更广泛 GPU 兼容性的兴趣。
- [**OpenRouter 用户面临模型的 Rate-Limiting 问题**](https://openrouter.ai/docs/limits)：高需求导致了 **rate-limiting** 障碍，尤其是 **DeepSeek V3** 等模型。虽然 OpenRouter 邀请供应商通过 [support@openrouter.ai](mailto:support@openrouter.ai) 集成模型，但用户对性能瓶颈表示沮丧。

**主题 5. AI 在代码开发中：实践与哲学**

- **开发者辩论测试压力：“听天由命（Jesus Take the Wheel）”方法**：一些程序员承认测试极少，在推送代码更改前幽默地依赖“*听天由命*”。其他人则强调严格测试的重要性，特别是在缺乏编译检查的语言中，以避免风险部署。
- **社区强调 AI 代码协作的明确指南**：在 **Windsurf** 等工具中，用户强调通过 **.windsurfrules** 提供的详细指南对于减少模糊的 AI 响应至关重要。分享这些规则并通过 [Codeium 的 Feature Requests](https://codeium.canny.io/feature-requests) 提出改进建议，促进了一个寻求更好 AI 交互的主动社区。
- **对游戏开发中实时 Bug 修复 AI 的兴趣**：用户推测未来的视频游戏可能会配备能够实时修复 Bug 的 **real-time AI**。他们幽默地想象 AI 修复旧作，将其视为迈向完全 **polished** 游戏体验的一步。

---

# 第一部分：Discord 高层摘要

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 的烦恼与收获**：多位参与者遇到了反复出现的 **AI-generated code** 错误，在开发中造成了“末日循环”，并寻求更好的 **structured approaches** 来优化输出，参考了 [Codeium Docs](https://docs.codeium.com/supercomplete/overview)。
   - 一些人表示，采用专门的指令而非宽泛的 prompt，可以显著提高可靠性，并促进围绕 **Windsurf** 潜力的更积极的社区讨论。
- **.windsurfrules 规则大显身手**：用户强调，精心定义的 **.windsurfrules** 指南有助于澄清项目需求，并减少代码协作过程中模糊的 AI 响应。
   - 社区成员建议分享这些规则并在 [Feature Requests | Codeium](https://codeium.canny.io/feature-requests) 提交请求，以加速 **Windsurf** 能力的提升。
- **量子井字棋吸引技术爱好者**：通过 [YouTube 视频](https://youtu.be/-qa7_oe5uWQ)展示了一个新的 **Quantum Computing Assembly Language** 演示，名为“量子井字棋（Quantum TicTacToe）”。
   - 爱好者们认为这个预告片是更广泛实验的火花，暗示了 **Windsurf** 的 AI 驱动代码生成与量子导向项目之间潜在的协同作用。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth GPU 支持与 AMD 的愿景发生冲突**：用户发现 **Unsloth** 目前仅支持单个 NVIDIA GPU，这引发了对未来 AMD 支持的困惑，并参考了 [SHARK-AI 的 AMD 优化指南](https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md#glossary)。
   - 社区成员认为目前没有立即的解决方案，而一些人将希望寄托在专门的 GPU 论坛上，以产生可用的补丁。
- **Mistral 的 Codestral 2501 引发许可热议**：Mistral 发布了 **Codestral 2501** 的[官方公告](https://mistral.ai/news/codestral-2501)，但用户对其受限的仅限 API 发布和商业化倾向表示遗憾。
   - 他们质疑**企业许可**是否会限制开源协作，引发了关于模型访问权限的热烈辩论。
- **DeepSeek V3 现身本地测试**：几位成员成功在本地运行了 **DeepSeek V3**，并报告在与 **llama3.1 405B** 配合使用时 VRAM 和 RAM 消耗很高。
   - 他们交流了性能技巧，承认在极简配置下速度较慢，且在大规模 Fine-tuning 中可能存在沉重的开销。
- **Llama 3.1 微调遭遇阻碍**：一位用户在微调 **Llama 3.1** 时遇到了验证损失（validation loss）指标缺失的问题，即使在调整了数据集大小和评估频率后也是如此。
   - 他们参考了 [Unsloth 的梯度累积修复博客文章](https://unsloth.ai/blog/gradient)，并将问题归因于 LLM 训练循环中的棘手问题。
- **4bit 格式引发体积争议**：一些人对 **4bit** 模型保存表示热衷，希望能减少内存占用并让较小的 GPU 保持竞争力。
   - 他们引用了 [Unsloth Notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks) 获取说明，尽管对压缩形式下的模型性能仍存顾虑。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor IDE 性能迟缓**：Discord 参与者指出 **Cursor IDE** 存在严重的请求缓慢问题，将其比作“幼儿园”级别的 Bug 报告场景，如 [Cursor 社区论坛](https://forum.cursor.com/)中所述。
   - 他们监控了 [Anthropic Status](https://status.anthropic.com/) 以了解可能的干扰，但仍对损害编程效率的停机时间感到沮丧。
- **Claude 表现优于 O1**：用户讨论了对 **Claude** 优于 **O1** 的偏好，指出 Claude 在 Agent 模式任务中表现出色。
   - 开发者引用了 O1 更高的资源需求，引发了关于实际使用中**模型性能**的辩论。
- **用于代码快照的批处理文件**：一位开发者创建了一个脚本，用于生成带编号的文件夹，并使用 *Beyond Compare* 在 **Cursor IDE** 出错时快速回滚。
   - 他们分享了[自己的 GitHub 主页](https://github.com/sksarvesh007)，鼓励他人采用该策略来有效追踪代码修改。
- **测试紧张：“交给上帝”的方法**：一些开发者承认测试极少，开玩笑说在推送代码更改前让“耶稣接管方向盘（Jesus take the wheel）”。
   - 其他人则强调在缺乏内置编译的语言中进行严格检查的重要性，并警告说无头部署（headless deployment）虽然风险大，但有时是不可避免的折中方案。
- **MCP 服务器与请求缓慢**：社区成员期待 **MCP** 服务器，认为它们可能会改善 Cursor 现有的响应缓慢问题。
   - 尽管存在等待时间，许多人仍然更喜欢该系统的宽松限制，而不是其他平台上更严格的并发限制。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen 2.5 vs QwQ 对决**：一位用户测试了 **Qwen 2.5 32B Instruct** 与 **QwQ**，报告称 Qwen 在代码生成方面表现更好，且回答不那么冗长，但结果仍然存在差异。
   - 参与者指出 **QwQ** 偶尔会出现不一致的情况，总体反应更倾向于 Qwen，认为其代码建议更清晰。
- **面向开发者的本地 LLM 推荐**：一名成员分享了[一份本地 LLM 指南](https://gist.github.com/shermanhuman/2b9a82df1bab242a8edffe504bb1867c)，强调了针对消费级硬件的 **GGUF 编码**。
   - 其他人提到了 **Hugging Face** 上的 [bartowski/Sky-T1-32B-Preview-GGUF](https://huggingface.co/bartowski/Sky-T1-32B-Preview-GGUF)，称其在经过仔细调整的量化下表现尚可。
- **生成式 AI 助力游戏开发**：用户推测未来的**视频游戏**可能会搭载实时 AI，用于即时修复 Bug，从而减少发布时的崩溃。
   - 他们幽默地想象了一种针对老游戏的突发 AI 修复方案，称其为向完全**打磨**后的经典作品迈进了一步。
- **多 GPU 乱战：RTX 5090 vs 4090**：参与者讨论了将 **5090** 与 **4090** 结合是否能提升处理能力，尽管旧显卡可能会限制性能。
   - 他们强调了逐层任务中的同步问题，这可能导致 **RTX 5090** 在等待较慢的 GPU 追赶时出现空闲时间。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **真核细胞穹窿中的量子特性**：研究人员注意到**真核细胞穹窿 (cell vaults)** 在**嘈杂**环境中能保持相干性，引发了关于量子计算用途的推测。
   - Neoxah 对更深层的细节保持沉默，暗示一旦工作正式发布，未来将会进行扩展。
- **Hyper-Connections 对抗残差阻碍**：研究人员提出 **Hyper-Connections** 作为标准残差连接的替代方案，理由是**梯度消失**和**表示崩溃**的挑战，并参考了[这篇论文](https://arxiv.org/abs/2409.19606)。
   - 初步测试显示它们达到或超过了现有方法，引发了对在**语言**和**视觉**流水线中扩展的乐观情绪。
- **过程奖励与 VinePPO 优化 LLM 推理**：新的**过程奖励模型 (PRMs)** 侧重于 Token 级检查，以增强 LLM 的数学能力，如[这项研究](https://arxiv.org/abs/2501.07301)所述。
   - 对话还探讨了用于思维链 (CoT) 任务的 **VinePPO**，确认它不依赖显式的 CoT 示例即可在 **LATo** 等扩展中获得持续收益。
- **MLQA 惊艳亮相**：一个新的 **MLQA** 基准测试实现通过 Pull Request 出现，为社区增加了多语言 QA 覆盖，尽管目前还有一个 **AST 错误**等待代码审查。
   - 提交者指出 **lm-eval-harness** 包含**多数投票 (majority voting)**，并引用了[此配置片段](https://github.com/EleutherAI/lm-evaluation-harness/blob/bb098f13b05e361f01a5afe7b612779ce362b3f2/lm_eval/tasks/gsm8k/gsm8k.yaml#L30)来设置重复采样。
- **Llama 2 的奇特配置与 Tokenizer 轶事**：开发者发现 **Llama 2** 的 **padded_vocab_size** 在 **NeoX** 和 **HF** 之间存在巨大差异（11008 vs 32768），参考了[此配置详情](https://github.com/EleutherAI/gpt-neox/blob/main/configs/llama2/7B.yml#L9)。
   - 他们还观察到 HF 使用 **silu** 而非 **swiglu**，一些人认为这与早期的激活函数选择不匹配，同时在构建日志中遇到了令人费解的哑 Token。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Supabase 设置意外**：成员们注意到，派生 (fork) 一个 Bolt 项目每次都需要重新部署 **Supabase**，这阻碍了与现有项目的重新连接，并打乱了正常工作流。
   - 他们将其与 **Loveable** 的重用方法进行了比较，希望开发团队能启用一种更简单、更直接的连接方式。
- **Perplexity 聊天机器人构思**：一位用户提议通过将 Hugging Face 模型集成到 Bolt 中来创建一个 **Perplexity 风格**的聊天机器人，引发了对开源 AI 解决方案的兴趣。
   - 其他人建议使用 **OpenAI API** 以实现更快的设置，但他们也讨论了处理不同 API 服务的挑战。

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeVries AI 推出 Telegram LLM 中心**：全新的 [DeVries AI](https://devriesai.com/) 在 Telegram 中提供 **200+ Large Language Models**，价格为 24.99 美元/月，并提供免费试用。
   - 用户可以在单个 Telegram 界面中快速切换 **ChatGPT** 和 **Claude**，并即将加入 **image/video generation** 功能。
- **OpenRouter 提供商设置势头强劲**：**OpenRouter** 邀请潜在提供商发送邮件至 [support@openrouter.ai](mailto:support@openrouter.ai) 以集成其模型，引发了关于创意用途的讨论。
   - 一位用户开玩笑说某个 *提供商秘密使用 OpenRouter*，引发了关于无意中构建出 **AGI** 的幽默推测。
- **Deepseek V3 响应缓慢引发关注**：用户报告 **Deepseek V3** 的响应失败率高达 **7/10**，指出过载导致回复迟缓。
   - 一些人建议切换到 **Together AI endpoint** 以获得更快的性能。
- **MiniMax 456B 参数模型备受瞩目**：**MiniMax** 推出了一款拥有 **456 billion** 参数的模型，尽管未在基准测试中夺冠，但在处理上下文方面表现稳健。
   - 其高效的规模引起了探索更大性能可能性的开发者们的兴趣。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **是机器人还是真人？滑稽的难题**：成员们对那些只打招呼的疑似新成员开玩笑，怀疑这些单行问候背后潜伏着 **Discord bots**。
   - 一些人提议采取更严格的注册步骤，但担心这可能会劝退真正的参与者。
- **DEIS BETA 大幅提升 Flux 采样**：爱好者们称赞 **DEIS BETA** 在 **Stable Diffusion** 场景中有效地引导了 Flux 采样。
   - 他们还在寻找其他工具，旨在改进各种任务中的采样参数。
- **审美分类器吸引好奇粉丝**：一位用户寻求将艺术风格与数字评分相结合的数据集，以构建一个可靠的 **aesthetic classifier**。
   - 建议包括利用 **ollama** 进行流线化提示，希望统一主观和客观的评分方法。
- **FP8 vs FP16：位宽之战**：社区成员辩论了新 GPU 中 **FP8** 与旧设备中更常见的 **FP16** 的优劣。
   - 他们注意到 **FP8** 的内存优势，但担心在高细节的 **Stable Diffusion** 任务中会出现精度权衡。
- **Intel B580 遇挫寻求解决方案**：一位贡献者抱怨由于 subreddit 限制，无法发布关于 **Intel B580** 在 **Stable Diffusion** 上的基准测试。
   - 其他人建议联系管理员或探索其他论坛，以收集更广泛的反馈和见解。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Qwen 的 PRM 在过程监督方面取得进展**：新的 **Qwen2.5-Math-PRM** 在 [ProcessBench](https://huggingface.co/papers/2412.06559) 的数学任务中间错误检测中表现优异，参考了 [Hugging Face 上的 72B 模型](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-72B)，该模型使用 **human-annotated** 数据以实现更强的推理。
   - 开发者警告说，**Monte Carlo** 合成方法落后于人工方法，强调了进行仔细 **evaluation** 的必要性。
- **Claude Sonnet 与 MiniCPM-o 引起轰动**：**Claude Sonnet 3.5** 在 SWE-Bench Verified 上达到 **62.2%**，落后于 OpenAI 的 o3（**71.7%**），这让许多将其视为上一代编程竞争者的人感到惊讶。
   - 与此同时，来自 OpenBMB 的 **MiniCPM-o 2.6** 拥有 **8B-size Omni** 设计，其双语实时音频能力令人印象深刻，详见 [GitHub](https://github.com/OpenBMB/MiniCPM-o) 和 [Hugging Face](https://huggingface.co/openbmb/MiniCPM-o-2_6)。
- **高等教育聊天机器人与 Stripe 的税务技巧**：一场面向 **higher-ed CIOs** 的演讲重点介绍了 **U-M GPT** 和 **Maizey**，密歇根大学倡导为多样化的校园需求提供量身定制的 AI 服务。
   - 在税务方面，成员们称赞了 **Stripe** 的 Non-Union One Stop Shop，让外部企业可以一站式处理 **EU VAT**。
- **合成 CoT 与 O1 的争议**：成员们发现 **synthetic chain-of-thought training** 效果平平，尤其是当它仅仅是没有 **RL** 的监督微调时。
   - 他们对 **O1 models** 的前景表示怀疑，暗示 **Big Molmo** 或 **Tulu-V** 在视觉任务上可能会表现得更好。
- **政策重击：AI 蓝图与数据中心热潮**：一份 [Economic Blueprint](https://openai.com/global-affairs/openais-economic-blueprint/) 建议利用 **AI** 促进国家安全和增长，呼应了 OpenAI 多次提出的政策建议。
   - 拜登总统的行政命令开放了联邦土地用于建设 **gigawatt-scale datacenters**，并要求配套现场 **clean energy** 以匹配产能。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **ChatGPT 推出调度功能**：2025 年 1 月 14 日，**ChatGPT** 推出了全新的**任务调度（Task Scheduling）**功能，可处理一次性和周期性提醒。据 [The Verge 报道](https://www.theverge.com/2025/1/14/24343528/openai-chatgpt-repeating-tasks-agent-ai)，该功能最初面向 **Plus**、**Team** 和 **Pro** 用户推出。
   - 此举旨在将 **ChatGPT** 重塑为更具主动性的 **AI Agent**，实现每日天气更新或新闻警报等任务，正如 [TechCrunch](https://techcrunch.com/2025/01/14/chatgpt-now-lets-you-schedule-reminders-and-recurring-tasks) 所述。
- **Cursor 获得 B 轮融资**：**Cursor** 宣布完成由 **a16z** 领投的 **B 轮（Series B）**融资，展示了投资者对先进编程工具和 AI 驱动开发平台的强劲信心，更多背景见 [Sarah Wang 的推文](https://x.com/sarahdingwang/status/1879279307119608142)。
   - 这笔资金注入突显了业界对 AI 辅助开发日益增长的热情，并为 **Cursor** 工具生态系统的进一步改进奠定了基础。
- **Ambient Agents 实现电子邮件自动化**：一种新型 **AI 邮件助手**可自主分类并起草邮件。这种“环境智能体（Ambient Agents）”背后的理念在[博客文章](https://blog.langchain.dev/introducing-ambient-agents/)中进行了详细阐述，并在 [Harrison Chase 的推文](https://x.com/hwchase17/status/1879218872727015644)中得到了进一步讨论。
   - 这种方法有望通过在后台处理常规任务来减少电子邮件过载，让用户专注于更高层级的决策，且仅需极少的直接监督。
- **Claude 的速率限制障碍**：用户报告在 **Cursor** 上使用 **Claude** Sonnet 3.6 模型时遇到了**速率限制（Rate-limiting）**障碍。[论坛讨论](https://forum.cursor.com/t/anthropic-cannot-sustain-additional-slow-request-traffic-on-claude-3-5-sonnet-please-enable-usage-based-pricing/41361/24)将其归咎于超出 Anthropic **GPU 可用性**的高流量。
   - 开发者透露 **Cursor** 是 Anthropic 最大的客户，这加剧了对更强大 GPU 供应的需求。
- **Magnetar 的算力换股权策略**：对冲基金 **Magnetar** 通过与 **Coreweave** 合作，向 AI 初创公司提供算力资源以换取股权，正如[近期播客](https://podcasts.apple.com/ca/podcast/how-the-hedge-fund-magnetar-is-financing-the-ai-boom/id1056200096?i=1000679726051)所报道。
   - 该策略旨在缓解新兴 AI 企业的融资困局，强调了基础设施准入在推动下一代 AI 发展中的重要性。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Google 的 10 美元音频概览调查**：Google 团队发布了一份 [5 分钟筛选表单](https://forms.gle/NBzjgKfGC24QraWMA)，旨在收集关于**音频概览（Audio Overviews）**的反馈，完成后可获得 **10 美元礼品码**。
   - 参与者需年满 18 岁，调查结果将用于指导未来的 **NotebookLM** 更新。
- **Akash：AI 生成播客的新平台**：一位用户展示了 [Akash](https://akashq.com)，这是一个上传和分享 **AI 生成播客**的便捷网站，省去了复杂的权限步骤。
   - 他们提供了分发基于 NotebookLM 内容的示例，并将其描述为“一种更简单的方法”。
- **播客时长困境**：社区成员讨论了如何限制 NotebookLM 的**播客时长**，并引用了一个 [Reddit 链接](https://www.reddit.com/r/notebooklm/comments/1gmtklt/longest_pod_i_have_ever_got/)寻找解决方案。
   - 其他人讨论了直接进行*音频转录*，建议增加内置功能而非通过上传文件作为来源。
- **付费版 NotebookLM 的公开分享需求**：关于**付费版 NotebookLM** 是否提供完全公开访问（无需为每个用户手动设置权限）的问题被提出。
   - 一些成员指出目前仅支持组织范围内分享，这引发了对更*开放发布*功能的呼吁。
- **针对 PDF 的 NoCode RAG 构想**：一位用户提出了使用 **NoCode** 方法从 Google Drive 中的 **PDF** 检索答案的想法，并将其与 NotebookLM 的检索工作流相结合。
   - 参与者认识到整合该方法的复杂性，希望在未来的迭代中获得更深度的支持。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 评价褒贬不一**：用户对**Pro 功能**中奖励码兑换失败、**代码辅助**效果有限以及无法无缝激活 **Pro search** 表示担忧。一些人称赞其研究优势，但批评了 UI 的变化，并推荐使用 [Ublock Origin](https://ublockorigin.com) 来屏蔽广告和不必要的内容。
   - 成员询问是否可以通过 API 访问 **Pro search**，但官方回复确认目前无法使用，这令工作流受阻。其他人担心**私有内容**不会出现在 Google 上，以及会失去对之前上传文档的访问权限。
- **代码助手越界**：尽管用户给出了明确指令，**代码助手**仍反复坚持要求确认免责声明和部分代码。这导致了摩擦，用户对该助手响应迟钝的设计感到不满。
   - 社区成员建议采用更具适应性的对话流，以减少重复的免责声明。一些人认为这种行为是*不必要的摩擦*，增加了开发任务的复杂性。
- **TikTok 面临额外监管博弈**：根据[这篇文章](https://www.perplexity.ai/page/chinese-officials-consider-tik-51PEvvekQxqVuhd74SkSHg)，中国官员正在考虑围绕 **TikTok** 制定可能的指南，重点关注内容审查和用户隐私。他们强调了对数据处理和监管行动日益增长的关注。
   - 观察人士预计政府实体将进行更多审查，并可能产生全球性后果。用户仍不确定这些规则何时或将如何全面实施。
- **德语摘要请求引发翻译讨论**：一位用户请求对[此讨论](https://www.perplexity.ai/search/kannst-du-mir-eine-zusammenfas-aZ8WnXOORK.hH5cZSr0qqg)中引用的数据进行**德语**摘要。他们强调了本地化覆盖的重要性。
   - 一些人质疑 **Perplexity** 如何大规模管理多语言查询。其他人则认为这是对跨语言 AI 知识共享的一次有趣测试。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepSeek 胜过 Claude**：尽管 **Anthropic** 不断调整 Claude 的后训练方法，**DeepSeek v3** 在编程任务中仍使 **Claude** 显得逊色。
   - 成员们分享了一个[握拳庆祝的 GIF](https://tenor.com/view/never-give-up-fist-pump-motivation-motivational-gif-4892451)，并赞赏 Claude 极具人性化的风格，暗示它可能仍是用户的首选。
- **私有数据 vs 开源之争**：参与者辩论了**开源**是否能与**专有**训练集相媲美，一些人建议建立政府数据中心以公平竞争。
   - 他们认为数据集的质量优于数量，并对完全合成的语料库持怀疑态度。
- **Gemini 轻松抓取数据**：**Gemini** 在**数据提取**方面赢得了赞誉，在准确性上超过了 **4o-mini** 和 **Llama-8B**。
   - 参与者提议使用 **Jina** 进行专门的文本转换，并参考程序化方法以确保精确的结果。
- **关注 Attention 替代方案**：一篇新的[论文](https://arxiv.org/abs/2501.06425)提出了标准 Attention 之外的方法，引发了推测。
   - 该小组将本月称为 **Attention 替代方案之月**，期待在即将发布的版本中看到更稳健的方法。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek v3 需要大内存**：贡献者报告称，有效运行 **DeepSeek v3** 需要约 **380GB** 的 RAM 和多块 GPU 显卡，并建议查看 [Hugging Face 官方仓库](https://huggingface.co/unsloth/DeepSeek-V3-GGUF)。
   - 他们将其与 **Qwen** 等较小的选项进行了比较，指出了硬件资源受限时的性能权衡。
- **Qwen 以更少资源实现本地运行**：成员推荐 **Qwen** 作为本地使用的较小开源替代方案，强调其资源需求低于 **DeepSeek v3** 等大型模型。
   - 他们表示它提供了平衡的性能并避免了沉重的内存开销，尽管没有明确分享基准测试数据。
- **Gemini 擅长用户故事创作**：讨论表明 **Gemini** 模型是根据特定需求生成用户故事（User Story）的有效开源工具。
   - 参与者赞赏其在叙事任务中的专门能力，但未提供明确的指标或链接来证实这些说法。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **MiniMax 的 4M Token 突破**：**MiniMax-01**，包括 **MiniMax-Text-01** 和 **MiniMax-VL-01**，已正式开源，可处理高达 **4M tokens**，远超现有模型 20–32 倍（[论文](https://filecdn.minimax.chat/_Arxiv_MiniMax_01_Report.pdf)）。
   - 其定价为每百万输入 token **$0.2**，每百万输出 token **$1.1**，并在 [海螺 AI (Hailuo AI)](https://hailuo.ai) 提供免费试用，引发了对下一代 AI Agent 工具链的热烈讨论。
- **Thunder Compute 的云端优惠**：一位联合创始人宣布了 [Thunder Compute](https://thundercompute.com)，在 Beta 测试期间提供 **$0.92/小时** 的 **A100** 实例，并赠送 **$20/月** 的免费额度。
   - 他们强调了一个简单的 CLI (`pip install tnr`) 用于快速设置实例，简化了 GPU 工作流并征求用户反馈。
- **Kaiko & Prior Labs 寻求模型构建者**：Kaiko AI 正在阿姆斯特丹和苏黎世招聘 **Senior ML Platform Engineers** 和 **Data Engineers**，专注于癌症治疗的 Foundation Models，不提供签证赞助（[ML Engineer 职位发布](https://jobs.kaiko.ai/jobs/4829222-senior-ml-platform-engineer)）。
   - 同时，**Prior Labs** 正在构建针对**表格数据**、**时间序列**和数据库的 Foundation Models，并引用了一篇 [Nature 文章](https://www.nature.com/articles/s41586-024-08328-6)，强调了其在医疗和金融领域的广泛影响。
- **TorchAO 尝试 int8**：社区成员确认 **int8_weight_only** 使用了由 [torch.compile](https://github.com/pytorch/ao/tree/main/torchao/quantization#workaround-with-unwrap_tensor_subclass-for-export-aoti-and-torchcompile) 优化的融合反量化与矩阵乘法 (fused dequant-and-matmul) 方法。
   - 他们演示了如何通过 **torch.export** 或 **ONNX** 导出这些量化模型，并强调了与 **TorchScript** 的兼容性以提升性能。
- **DeepSeek 2.5 表现出色**：成员们赞扬了 **DeepSeek 2.5** 在一项共享任务中表现出的“完美的推理能力”，展示了显著先进的逻辑。
   - 他们分享了一张图片进行验证，展示了强劲的结果，并引发了对该模型更广泛能力的关注。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Codestral 推出 256k 上下文**：新的 **codestral** 模型在 **Mistral API** 上免费提供，拥有 **256k** 上下文，测试其效率的用户称其“极快且好用”。
   - 用户预计大规模代码生成将获得显著的速度提升，理由是其庞大的上下文窗口和易于部署的特性。
- **ChatGPT 4o Canvas 令人困惑**：鉴于 OpenAI 模糊的推广方式，成员们质疑 **ChatGPT 4o with Canvas** 究竟是 9 月份的旧模型还是新发布的变体。
   - 一些人观察到他们之前的 **4o canvas** 对话回退到了 **4o mini**，引发了关于系统更新的进一步猜测。
- **AI 对齐失效引发讨论**：一段关于 **AI misalignment**（AI 对齐失效）的 [YouTube 视频](https://www.youtube.com/watch?v=K8p8_VlFHUk) 引起了关注，展示了潜在风险的动画场景。
   - 围绕视频的相关性出现了一些疑问，促使观众探索它如何与对高级 AI 系统的广泛担忧相一致。
- **PDF 不是 API**：**prompt-engineering** 和 **api-discussions** 频道的贡献者正在寻求比 **PDF** 更好的数据格式，提倡使用 **JSON**、**YAML** 或纯文本。
   - 一位用户开玩笑说 *PDFs are not an API*，呼应了大家对 AI 任务中繁琐的文档转换的共同挫败感。
- **为非母语人士简化语言**：一个新的 **de-GPTing**（去 GPT 化）提示词有助于重新组织文本，在保留核心技术术语的同时省略罕见词汇。
   - 用户在 [OpenAI Playground](https://platform.openai.com/) 中分享了一种自定义技术，用于减少重复引用，旨在提高回复的清晰度。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Konkani 语言协调与语言保护**：在 **Cohere 讨论**中，一位用户关注了果阿邦 **250 万**人使用的 **Konkani** 语，开发者 **Reuben Fernandes** 正在寻求专业合作，以提高其语言保护项目的认可度。
   - 他计划创建一个能以 **Konkani** 语交流的 **AI 模型**，并强调现有系统都无法充分处理该语言，这引起了参与者的好奇。
- **Rerank 微调定价谜团**：成员们对 [Cohere 定价](https://cohere.com/pricing)中缺失的 **Rerank FT** 成本提出疑问，并参考[官方文档](https://docs.cohere.com/docs/rerank-fine-tuning)寻求澄清。
   - 他们分享了 **FAQ 链接**，并建议这些资源可能会阐明专门的政策，显示出对更清晰成本结构的需求。
- **Cohere 的 128k 上下文限制探讨**：参与者明确了 **128k tokens** 的容量（约 **42,000** 字）涵盖了所有交互，强调这不仅仅是单次聊天的记忆。
   - 讨论对比了**长期 vs 短期**记忆，并指出使用率限制是在 [Cohere 文档](https://docs.cohere.com/v2/docs/rate-limits)中规定的，而不是基于 token 长度。
- **《爱丽丝梦游仙境》机器人趣谈**：**Cmd R Bot** 否认了 *corvo* 和 *escrivaninha* 之间有任何联系，但一个关于**《爱丽丝梦游仙境》**的引用暗示了隐藏的语言转折。
   - 其 **Cohere 文档**搜索结果为空，凸显了在处理文化或文学视角方面的空白。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 的异步愿景**：Owen 提交了两个 [pull requests](https://github.com/modularml/mojo/pull/3945) 和 [pull request #3946](https://github.com/modularml/mojo/pull/3946)，旨在为 **Mojo** 引入**结构化异步 (structured async)**和**效应处理器 (effect handlers)**，强调需要标准化诸如 **oom** 和 **divbyzero** 等异常。
   - 与会者讨论了多种执行器设计，建议在分支到高级并发之前，基础 API 层至关重要。
- **Zed Zoom：Mojo 扩展稳步推进**：一位开发者创建了一个专门的 [Mojo in Zed 扩展](https://github.com/freespirit/mz)，解决了 **stdlib** 路径检测问题，并提供了改进的 LSP 功能。
   - 其他人交流了优化自动补全的建议，一些人强调了该扩展对扩大 **Mojo** 采用率的潜力。
- **Mojodojo 的 Int8 故障**：一位用户在 **Mojodojo** 中将 **Int8** 转换为 **String** 时遇到了[转换错误](https://github.com/modularml/mojo/issues/3947)，并引用了来自[文档](https://mojodojo.dev/guides/intro-to-mojo/basic-types.html#strings)的部分代码。
   - 社区成员分享了 [String 结构体文档](https://docs.modular.com/mojo/stdlib/collections/string/String/#write_bytes)和[参数化概念 (Parameterization concepts)](https://docs.modular.com/mojo/manual/parameters/)的参考资料以解决匹配问题。
- **会议与直播：快速回顾**：一位参与者因课程冲突错过了部分会议，但感谢其他人提供的**更新**让他们保持同步。
   - 他们分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=PYtNOtCD1Jo)，作为无法观看全程对话的人的有用资源。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tiny corp 融资 500 万美元，助力普及计算资源**：一篇[博客文章](https://geohot.github.io/blog/jekyll/update/2023/05/24/the-tiny-corp-raised-5M.html)透露，**tiny corp** 筹集了 **500 万美元**，以加速其在先进计算芯片开发方面的进展。
   - 创始人强调了人类大脑 **20 PFLOPS** 的计算量与当前 HPC 成本之间的巨大差距，引发了关于弥合公众获取更大计算资源途径的讨论。
- **tinygrad 在解决硬件差距中的作用**：成员们讨论了 **tinygrad 的用途**，重点关注其以极低开销处理 GPU 和 CPU 后端的能力。
   - 他们指出，熟悉 **LLVM** 有助于理解 tinygrad 如何编排底层操作，这是基于分布式系统的视角。
- **堆叠 Tensor 遭遇递归瓶颈**：用户在堆叠超过 **6,000** 个 tensor 并调用 `.numpy()` 时遇到了 **RecursionError**。
   - 他们将数量减少到 **1,000** 个 tensor 并绕过了堆叠限制，并建议分块操作以避免**内部递归深度**问题。
- **维度混淆引发错误**：一位用户发现在 tinygrad 中对 1D tensor 调用 `transpose()` 会导致 **IndexError**。
   - 其他人解释说，指定维度参数对于安全操作至关重要，强调了在 **tensor 属性**中具备维度意识的重要性。

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **医疗精通的扩展冲刺 (Scaling Sprints for Medical Mastery)**：在 **500 个样本**的小型训练集上将 **inference time** 增加 **6%-11%**，显著提升了 LLM 在医疗基准测试中的表现，详见 [O1 Replication Journey -- Part 3: Inference-time Scaling for Medical Reasoning](https://arxiv.org/abs/2501.06458)。**任务复杂度**要求更长的推理链，因此排除了内部数据以避免逻辑推导中的混淆。
   - **Nature Communications** 在 [Large Language Models lack essential metacognition for reliable medical reasoning](https://www.nature.com/articles/s41467-024-55628-6) 中强调了模型元认知方面的差距，指出了额外计算成本与临床鲁棒输出之间的矛盾。
- **O1 越狱焦虑 (O1 Jailbreak Jitters)**：怀疑论者对为了训练新模型而**越狱 O1** 的有效性表示怀疑，批评现有基准测试与现实需求脱节。
   - 其他人要求进行更彻底的风险评估，警告不谨慎的越狱行为会削弱对所得系统的信任，并有必要重新思考整个 O1 方法。
- **医疗自动化评估的焦虑 (Healthcare's Automated Assessment Angst)**：成员们认为医疗领域中**基于多选题**的评估将 AI 限制在模式识别和记忆任务中，忽略了更深层次的临床能力。
   - 他们呼吁建立更细致的测试协议，以衡量未来 AI 如何参与实际的诊断和治疗场景。
- **揭穿医疗测试神话 (Debunking Medical Testing Myths)**：关于将**多选题考试**的成功等同于真正的临床技能引发了辩论，指出资深医生在现实世界中的审查远超考试分数。
   - 爱好者们推动将提示词驱动的 AI 与实践专业知识相结合，旨在对模型的临床能力进行更现实的评估。
- **重新定义 AI 在医学中的未来 (Redefining AI’s Future in Medicine)**：参与者强调 **AI** 应该重塑既定的医疗规范和培训，而不是取代医生，以推进患者护理。
   - 他们敦促设计者挑战过时的常规，构想基于伦理保障和真实临床需求的 AI 与人类平衡协作。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 获得视频处理优势**：在一位用户解决了通过 `pipx` 和 **brew** 安装 **Open Interpreter** 的问题后，他们确认该工具可以处理**视频编辑**指令。
   - 他们还注意到当 **Open Interpreter** 输出大量内容时，**Cmder** 会出现性能故障，导致频繁清屏。
- **Deepseek 模型与集成见解**：一位用户询问了 **Deepseek** 模型名称以及如何设置 **DEEPSEEK_API_KEY** 以明确使用方法。
   - 他们还询问了如何将 **Deepseek** 集成到 **Open Interpreter** 中，显示出对连接这两个工具的兴趣。



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **2024 MOOC 引发怀旧情怀**：一位用户怀着强烈的钦佩之情回顾了 **2024** 年，称该 MOOC 是这一年的亮点。
   - 他们分享了重新观看 [Fall 2024 MOOC lectures](https://llmagents-learning.org/f24) 并与有同样感受的人建立联系的兴奋之情。
- **MOOC 在初学者中受到关注**：一位新手在完成之前的机器学习课程后询问了**初学者友好度**，寻求平稳过渡。
   - 其他人回应称 [Fall 2024 MOOC lectures](https://llmagents-learning.org/f24) 在**核心概念**和技能提升的高级技巧之间取得了平衡。
- **2024 秋季课程为 2025 春季奠定基础**：一位成员敦促潜在学习者观看 [Fall 2024 lectures](https://llmagents-learning.org/f24)，为即将到来的 **Spring 2025** 模块积累背景知识。
   - 他们指出下一期课程不会严格要求先验知识，但提前开始总没有坏处。
- **证书发放缓解学生忧虑**：一位用户询问了 **Fall 2024 MOOC certificate**，担心会错过正式颁发。
   - 另一位用户确认证书将于**本月晚些时候发放**，缓解了大家对认可问题的焦虑。



---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **AMD 在 GPT4All 中更接近 NPU 集成**：有人提问 GPT4All 是否很快会利用 AMD 处理器上的 **NPU**，暗示未来会有性能提升。
   - 一位开发者提到 AMD 的软件栈仍然是一个限制因素，但表示一旦软件栈最终确定，支持前景将非常乐观。
- **远程使用 GPT4All 的 VPN 解决方案**：参与者建议在推理机上使用 **VPN** 或 **反向代理**，以便从其他设备访问 GPT4All 的界面。
   - 他们将其描述为一种无需复杂硬件即可实现多设备交互的实用方法。
- **Hugging Face 澄清 GPT4All 模型变体**：对话强调了 **Hugging Face** 上存在多个 **quantization**（量化）变体，例如 codellama q4_0。
   - 将模型文件放在单个文件夹中显然解决了使用不同版本时的困惑。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Agent 配合 Weaviate 在 RAG 竞赛中胜出**：在 [Tuana 最近的一份 notebook](https://twitter.com/tuanacelik) 中，一个使用 **Weaviate** 和 **LlamaIndex** 的 **Agent** 在检索相关数据方面表现优于朴素 **RAG**。
   - 社区成员将此归功于该 **Agent** 结合数据源以获得更强覆盖范围的方法，并重点展示了其决策能力。
- **QAE 获得自定义 Prompt 能力**：一位用户探索了在 `QuestionsAnsweredExtractor` 中使用 `self.llm.apredict()` 的额外变量，引用了 [LlamaIndex 高级 Prompt 文档](https://docs.llamaindex.ai/en/stable/examples/prompts/advanced_prompts/#3-prompt-function-mappings)。
   - 另一位成员分享了函数映射（function mappings）如何提供动态变量，展示了 **LlamaIndex** 可以流利地将多个数据点注入到 **prompt** 模板中。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Meta 的 JASCO 点燃音乐生成热潮**：**Meta AI 的 FAIR 团队**推出了 **JASCO**，这是一个在 **2024 年 11 月**训练的新音乐模型，它使用 [EnCodec](https://huggingface.co/facebook/jasco-chords-drums-melody-1B) 进行音频 **tokenization**，并能处理**和弦**、**鼓点**和**旋律**。
   - 它提供 **400M** 和 **1B** 两种变体，采用 **flow-matching** 主干网络和 **condition dropout**，引发了人们对灵活的 **text-to-music** 生成的关注。
- **JASCO 论文强调技术基础**：一篇题为 [Joint Audio And Symbolic Conditioning for Temporally Controlled Text-To-Music Generation](https://arxiv.org/pdf/2406.10970) 的论文概述了 **JASCO** 基于 **Transformer** 的架构和特性。
   - 工程师们讨论了其专门的音频和符号 **conditioning**，指出其在**下一代**音乐创作和模型复杂性方面的潜力。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Ambient Agents 仍缺少示例**：一位成员询问如何使用 **DSPy** 配置 **ambient agent**，并请求提供代码示例，但聊天中未出现相关示例。
   - 其他人也表达了对 **DSPy** 实际应用案例的兴趣，希望能从其他开发者那里获得**共享资源和经验**。
- **DSPy 实现展示与交流**：另一位参与者邀请更多 **DSPy** 演示，重点展示针对 **ambient agent** 场景的任何动手尝试或部分原型。
   - 他们鼓励社区分享相关细节或开源 **repos**，旨在推动 **DSPy** 解决方案的发展。

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI 在医疗与金融领域加速发展**：在 **2025年1月16日** **下午 4:00 至 5:30 (IST)**，一个全球专家小组将讨论 **AI** 如何影响医疗和金融领域，注册链接见 [此链接](http://bit.ly/3PxJXbV)。
   - 组织者邀请了 **AI 赋能解决方案** 的 **构建者、运营者和所有者** 来解决成本优化和数据管理问题，希望能加速 AI 在这些行业的应用。
- **专家小组关注现实世界的 AI 部署**：该小组计划强调运营细节，包括医疗和金融领域的数据互操作性、合规性和实时分析。
   - 他们强调这些行业之间的 **交叉融合 (cross-pollination)**，期待以最小的开销扩展机器学习模型的新策略。

---

**Axolotl AI Discord** 没有新消息。如果该公会沉寂太久，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该公会沉寂太久，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该公会沉寂太久，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该公会沉寂太久，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会沉寂太久，请告知我们，我们将将其移除。

---

# PART 2: 频道详细摘要与链接

{% if medium == 'web' %}

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1328453553307521074)** (16 条消息🔥): 

> `Support Ticket Issues, Codeium Connectivity Problems, Technical Assistance, Freedom of Speech Concerns, Telemetry Errors on VS Code` 

- **支持工单困扰**：在近 **10 天** 没有收到回复后，一名成员建议管理员应该升级他们未解决的支持工单，以便更快解决。
   - 另一名成员提到他们找不到提交工单的日期，强调了支持响应的延迟。
- **Codeium 连接困扰**：一名成员报告了 Codeium 上 **服务器异常关闭连接** 的情况，这表明可能存在服务器端问题。
   - 他们敦促其他人如果问题持续存在，请 **下载诊断信息 (diagnostics)** 以进行进一步的故障排除。
- **对技术问题协助的感谢**：一位用户对另一名成员在 **电子邮件冲突** 方面的帮助表示感谢，该冲突导致了账号归属权的混乱。
   - 他们强调了在此技术问题期间收到的支持，并对所提供的 **耐心和时间** 表示感谢。
- **关于言论自由的辩论**：一名成员对被删除的消息表示沮丧，认为该频道缺乏言论自由。
   - 作为回应，其他人强调了 **尊重** 和遵守社区准则的重要性。
- **VS Code 上的 Telemetry 问题**：一名成员对他们的 **Codeium VS Code** 扩展中的 Telemetry 错误表示担忧，尽管他们已经启用了该功能。
   - 他们指出了 GitHub 上一个相关的问题，寻求处理相同问题的其他人的帮助。

**提到的链接**：<a href="https://github.com/Exafunction/CodeiumVisualStudio/issues/111.">Exafunction/CodeiumVisualStudio</a>：Codeium 的 Visual Studio 扩展。通过在 GitHub 上创建账号来为 Exafunction/CodeiumVisualStudio 的开发做出贡献。

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1328453559829794919)** (350 条消息🔥🔥): 

> `Windsurf 性能、AI 建议改进、对清晰指南的需求、应用开发挑战、用户体验与反馈` 


- **AI 生成代码的困扰**：用户对 Windsurf 反复出现的问题表示沮丧，指出 AI 生成的代码在最初成功后往往会引入错误，在开发过程中造成“毁灭循环”（loop of doom）。
   - 参与者正尝试实施结构化方法和文档来引导 AI 并提高输出的准确性。
- **清晰指南的重要性**：大家达成共识，以 `.windsurfrules` 文件形式提供的清晰且具体的指南对于与 AI 的有效协作至关重要，因为模糊的 Prompt 会导致不理想的结果。
   - 鼓励用户像对待开发者一样清晰地表达需求，旨在避免 AI 回复中的歧义。
- **经验分享**：几位用户分享了他们使用 Windsurf 创建应用程序的个人经验，范围从创建新闻网站到开发复杂的系统（如日志记录和 XML 解析器）。
   - 虽然许多参与者对 Windsurf 的功能感到兴奋，但他们也表达了由于功能限制所面临的挑战。
- **反馈与建议**：用户对 AI 的建议行为提出了担忧，特别是选择的随机性，引发了关于潜在改进的讨论。
   - 鼓励建立反馈机制和增强建议，以直接解决用户在使用该工具时的体验。
- **用户协助与资源**：新用户（如非开发角色的用户）寻求有效利用 Windsurf 的指导，强调了对资源和教程的需求。
   - 社区成员正在协作并分享关于高效使用和 Prompt 结构化的技巧，以减少困惑并最大限度地发挥 AI 的能力。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/wink-eye-turn-around-chewing-gif-23703707">Wink Eye GIF - Wink Eye Turn - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/power-starwars-unlimited-power-gif-15939349">Power Starwars GIF - Power Starwars Unlimited Power - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://docs.codeium.com/supercomplete/overview">概览 - Codeium 文档</a>：未找到描述</li><li><a href="https://codeium.canny.io/feature-requests">功能请求 | Codeium</a>：向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://codeium.com/blog/pricing-windsurf">方案与价格更新</a>：Cascade 定价模型的一些变更。</li><li><a href="https://youtu.be/-qa7_oe5uWQ">量子井字棋</a>：使用我们新的量子计算汇编语言进行游戏
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1328454686663245864)** (128 条消息🔥🔥): 

> `Unsloth GPU 支持, Mistral 代码生成模型, DeepSeek V3 本地运行, 模型中的 Chat Templates, 模型许可与可用性` 


- **Unsloth 在 GPU 支持方面面临困难**：用户指出 **Unsloth 目前仅支持单个 NVIDIA GPU**，且缺乏对 AMD 的支持，这导致了一些挫败感。
   - 关于未来支持 AMD 的可能性存在疑问，目前尚无明确答案。
- **Mistral 的 Codestral 模型发布存在不确定性**：有推测称 Mistral 的 **Codestral 2501** 尚未向公众开放，用户对其目前仅限 API 访问表示失望。
   - 团队似乎在优先考虑模型的商业许可，这引发了关于对普通用户影响的讨论。
- **在本地运行 DeepSeek V3**：用户分享了在本地运行 **DeepSeek V3** 的经验，以及硬件限制带来的挑战，特别是 RAM 和 VRAM 的要求。
   - 优化设置（如使用 **llama3.1 405B**）是常见的变通方法，但在资源受限的情况下，效率仍存在不确定性。
- **Chat Templates 的重要性**：讨论了在推理设置中加入 **chat template** 的必要性，特别是针对 **Phi-4** 等指令微调模型。
   - 一些成员指出，不使用模板会导致奇怪的补全结果，且不同的模型可能使用各种指令指示符。
- **对模型许可的担忧**：对话强调了围绕模型**许可和可用性**的复杂性，特别是对于来自 Mistral 的中小型模型。
   - 参与者对潜在的延迟以及限制个人和研究人员访问的商业导向方法表示担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://grok.com/">Grok</a>：了解宇宙</li><li><a href="https://mistral.ai/news/codestral-2501">Codestral 25.01</a>：以 Tab 的速度编写代码。现已在 Continue.dev 上线，即将登陆其他领先的 AI 代码助手。</li><li><a href="https://x.com/iScienceLuvr/status/1879091791753843064">Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：Tensor Product Attention Is All You Need。提出 Tensor Product Attention (TPA)，一种利用上下文张量分解对 Q, K, V 激活进行因子分解的机制，可实现 10 倍或更多的减少...</li><li><a href="https://docs.unsloth.ai/basics/reward-modelling-dpo-orpo-and-kto">奖励建模 - DPO, ORPO &amp; KTO | Unsloth 文档</a>：要在 Unsloth 中使用 DPO, ORPO 或 KTO，请遵循以下步骤：</li><li><a href="https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md#glossary">shark-ai/docs/amdgpu_kernel_optimization_guide.md at main · nod-ai/shark-ai</a>：SHARK 推理建模与服务。通过在 GitHub 上创建账号为 nod-ai/shark-ai 的开发做出贡献。</li><li><a href="https://huggingface.co/microsoft/phi-4/blob/main/tokenizer_config.json#L774>">tokenizer_config.json · microsoft/phi-4 at main</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/bigcode/santacoder-fim-task">bigcode/santacoder-fim-task · Datasets at Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1328478985574809670)** (7 条消息): 

> `LLAMA 中的动态过滤器, 语言学习, 视频资源, 发音练习` 


- **动态过滤器让 LLAMA 更有趣**：一位用户建议在 **LLAMA** 中尝试动态的用户定义过滤器，认为与长 Prompt 相比，它们提供了一种更快速的表达想法的方式。
   - 他们提供了一个幽默的例子，涉及电影剧本场景中的“尖叫的疯狂猫夫人”和“超级英雄邮递员”等角色。
- **探索语言学习选项**：几位成员讨论了他们偏好的学习语言，建议包括**波兰语**和**日语**。
   - 他们强调了练习发音和打字技能的有效方法的必要性。
- **分享视频资源**：一位用户发布了一个 YouTube 视频链接，可能与正在进行的讨论有关。
   - 具体来说，分享的链接是 [这个视频](https://www.youtube.com/watch?v=wXGZC-fCrco)。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1328464452118642782)** (144 条消息🔥🔥): 

> `在 Unsloth 上进行 Fine-tuning，训练 VLMs，用于幻觉控制的 RAG，使用 4bit 保存，LLM 的数据集准备` 


- **Llama 3.1 的 Fine-tuning 问题**：一位用户报告了他们在进行 Llama 3.1 Fine-tuning 时遇到的问题，具体表现为尽管尝试了不同的方法，但在训练过程中无法获取 validation loss。
   - 建议调整评估数据集的大小和频率，以提高训练效率。
- **训练 VLMs 与内存问题**：一位在训练 VLM 时遇到系统内存限制的用户被告知，减小训练数据规模会有所帮助，但这无法满足其使用完整数据集的需求。
   - 他们还指出，由于在每个训练 epoch 期间处理庞大的评估数据集，导致速度显著变慢。
- **用于幻觉控制的 RAG 方法**：在讨论将 Llama 3B 用于法律数据集时，有人建议使用更大的模型（70B+）结合 RAG 可能会减轻幻觉等问题。
   - 人们对语言和信息上下文中的幻觉表示担忧，强调了进行稳健验证的必要性。
- **使用 4bit 和模型配置**：一位用户询问了是否可以以 4bit 格式保存模型，并得到了确认，即提供的 notebook 中包含此类功能。
   - 讨论还涉及了如何使用 FastLanguageModel 在低端 GPU 上配置模型加载。
- **LLM 的数据集准备**：对于有兴趣根据特定作者风格创建 LLM 的用户，强调了数据集准备工作占总工作量的 85%。
   - 有人指出，理解 LLM 的功能对于有效的数据集策划和训练过程至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discuss.huggingface.co/t/perhaps-your-features-output-in-this-case-have-excessive-nesting-inputs-type-list-where-type-int-is-expected/135553">Perhaps your features (`output` in this case) have excessive nesting (inputs type `list` where type `int` is expected)</a>: 我在这里也遇到了类似的问题。ValueError: Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batche...</li><li><a href="https://cohere.com/llmu/what-is-semantic-search">What is Semantic Search?</a>: 在本节 LLM University 章节中，你将学习如何使用 embeddings 和相似度来构建语义搜索模型。</li><li><a href="https://unsloth.ai/blog/gradient">Bug Fixes in LLM Training - Gradient Accumulation</a>: Unsloth 的 Gradient Accumulation 修复解决了 LLM 训练中的关键错误。</li><li><a href="https://colab.research.google.com/drive/1mf3lqz2ga80p_rIufDvBPvyFqtn9vcdS?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: 以下是我们所有 notebook 的列表：
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 条消息): 

fjefo: https://youtu.be/SRfJQews1AU?si=s3CSvyThYNcTetX_
  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1328462335102287933)** (209 条消息🔥🔥): 

> `Cursor IDE 性能问题，模型对比，快照批处理文件，使用 Claude vs O1，Cursor 用户体验` 


- **Cursor IDE 遭遇性能问题**：用户报告了 Cursor IDE 持续存在的性能问题，特别是“请求缓慢”影响了响应时间。
   - 停机和减速导致了用户不满，一些用户将他们的体验比作在“幼儿园”环境中报告 Bug。
- **Claude 与 O1 等模型的对比**：关于不同模型有效性的辩论已经展开，许多用户表示 Claude 优于 O1，特别是在 Agent 模式下。
   - Claude 被认为更有能力处理请求，而 O1 则表现挣扎，导致用户群更倾向于 Claude。
- **使用批处理文件管理代码更改**：一位用户开发了一个批处理文件工具来跟踪代码更改，为快照创建编号文件夹，并允许使用 Beyond Compare 进行轻松对比。
   - 这种方法可以帮助用户快速识别并回滚 Cursor IDE 产生的不正确代码更改，从而简化调试过程。
- **测试的经验与挫败感**：关于测试实践的讨论凸显了不同的态度；一些用户表达了随性的态度，称他们在部署时“听天由命”（let 'Jesus take the wheel'）。
   - 其他人则强调了测试的重要性，特别是在缺乏自动编译和测试能力的语言中。
- **用户对 Cursor 功能的探索**：用户讨论了他们对 Agent 模式等各种 Cursor 功能的体验，以及即将推出的 MCP 服务器的影响。
   - 尽管存在一些停机时间，但人们认识到，相比于其他平台强制执行的限制，缓慢的请求系统是可以接受的。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/sksarvesh007">sksarvesh007 - 概览</a>: sksarvesh007 拥有 62 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://forum.cursor.com/">Cursor - 社区论坛</a>: 讨论 Cursor 的地方（Bug、反馈、想法等）</li><li><a href="https://status.anthropic.com/">Anthropic 状态</a>: 未找到描述</li><li><a href="https://tailwindflex.com/u/contribution">来自 Login 的推文</a>: 未找到描述</li><li><a href="https://go.fb.me/h0q0ke">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1328764424286244986)** (2 条消息): 

> `Hugging Face 搜索功能，问题解决` 


- **确认 Hugging Face 搜索问题**：报告了一个关于**应用内 Hugging Face 搜索**的已知问题，团队表示正在与 HF 团队合作解决。
   - 这个问题一直困扰着用户，但团队正专注于快速修复。
- **Hugging Face 问题已解决**：团队确认，感谢给力的 HF 团队，**搜索问题现在应该已经解决**。
   - *“Awesome HF team ❤️🤗”* 表达了对协作解决问题的赞赏。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1328457022961876992)** (72 条消息🔥🔥): 

> `Voodoo 显卡怀旧、Qwen 2.5 与 QwQ 对比、本地 LLM 推荐、生成式 AI 与游戏开发、LM Studio 聊天与开发模式响应差异` 


- **Voodoo 显卡怀旧回忆**：一位用户深情地回忆了 **Voodoo 1** 如何改变了游戏体验，并分享了它在 LAN party 上与各种 OpenGL 游戏的兼容性，这在当时促使许多人购买了该显卡。
   - 他们感叹现有的在线视频无法完全还原他们记忆中那种图形质量，言语中充满了怀旧之情。
- **Qwen 2.5 与 QwQ 的混合体验**：一位用户寻求 **Qwen 2.5 32B Instruct** 与 **QwQ** 的对比，并指出根据其初步测试，结果尚不明确。
   - 普遍共识是 **Qwen 2.5** 在代码编写方面表现更好且废话较少，这与 **QwQ** 模型中发现的问题形成对比。
- **针对编程的本地 LLM 推荐**：一位成员分享了测试本地托管编程模型的见解，强调了良好的 GGUF 编码对于获得最佳性能的重要性。
   - 他们推荐了特定的量化版本和模型，并讨论了在消费级硬件上使用高参数模型时面临的挑战。
- **生成式 AI 增强游戏开发**：用户推测，未来利用 **gen-AI** 开发的视频游戏是否会因为具备实时 AI Bug 修复能力而减少 Bug。
   - 这种可能将 AI Bug 修复应用于旧游戏标题的功能让人感到兴奋，从而确保更完美的体验。
- **LM Studio 界面响应不一致**：一位用户报告了 LM Studio 聊天部分与开发部分之间的响应差异，并注意到模型回复中的异常行为。
   - 另一位用户确认为类似问题正在处理中，并指向了之前关于优化沟通的相关讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://gist.github.com/shermanhuman/2b9a82df1bab242a8edffe504bb1867c">Coding local LLM recommendations that meet some minimum useful standard</a>: 满足最低实用标准的编程本地 LLM 推荐 - MinimumStandardsLLM.md</li><li><a href="https://huggingface.co/bartowski/Sky-T1-32B-Preview-GGUF">bartowski/Sky-T1-32B-Preview-GGUF · Hugging Face</a>: 暂无描述</li><li><a href="https://youtu.be/VkzO2w6EqK4?si=dONXQA4qc6VCdUvk">The 3Dfx Voodoo Difference: This is why we love them</a>: 在这段视频中，我们将了解为什么 3DFX Voodoo 是一款如此特殊的显卡！💙 考虑支持我 💙Patreon: 获取独家早期访问权限...
</li>
</ul>

</div>

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1328455306749022301)** (77 messages🔥🔥): 

> `RTX 5090 vs 4090 multi-GPU setups, GPU memory bandwidth impacts, Image generation GPU setups, Dual GPU setups for cost efficiency, Model training complexities` 


- **讨论将 RTX 5090 添加到 4090 中的收益**：许多人对在现有的 **RTX 4090** 配置中添加一块 **RTX 5090** 带来的性能影响感到好奇，特别是 **5090** 卓越的内存带宽是否能提高处理速度。
   - 然而，有人指出在多 GPU 设置中，**最慢的 GPU** 可能会成为整体性能的瓶颈，这引发了关于有效并行处理的疑问。
- **多 GPU 系统中的内存带宽和 VRAM**：讨论提到在多 GPU 设置中，模型的**每一层**是顺序处理的，这可能会产生同步瓶颈并限制更快的 GPU 所带来的优势。
   - 这意味着即使是像 **RTX 5090** 这样更快的显卡，在等待较慢的 **RTX 4090** 完成处理时，最终也可能处于闲置状态。
- **探索用于推理的 GPU 配置**：考虑到更高的内存带宽和更快的处理时间，使用两块 **RTX 4090** 可能会比单块 **A6000 48GB** 提升推理速度。
   - 讨论指出，尽管有性能优势，但从能效比来看，**较少数量的 GPU** 配置可能更具优势，能以更低的成本提供相似的性能。
- **模型层并行化的挑战**：参与者辩论了跨 GPU 拆分层以实现并行推理速度是否可行，但结论是固有的同步性可能会阻碍这种方法。
   - 层计算后完全同步的挑战引发了关于架构在实践中是否能有效处理此类划分的问题。
- **双 GPU 设置的考虑因素**：许多人正在权衡选择双 **4060ti** 设置而非高端显卡的方案，以获得更好的显存性价比（VRAM-per-dollar），同时解决电源供应限制问题。
   - 用户经验表明，虽然品牌差异很小，但在选择显卡配置时，散热和物理安装等实际问题非常重要。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.amazon.com/PNY-GeForce-RTXTM-Verto-Graphics/dp/B0CG2MX5H9?ref_=ast_sto_dp&th=1">未找到标题</a>：未找到描述</li><li><a href="https://vladmandic.github.io/sd-extension-system-info/pages/benchmark.html">SD WebUI Benchmark Data</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1328461997280329864)** (40 messages🔥): 

> `PhD in Statistical Learning Theory, Orch-Or Theory and AI, AI Autocomplete Speed, Developer Opportunities, Off-Topic Channel Dynamics` 


- **博士候选人寻求对齐（Alignment）见解**：一位成员宣布他们即将开始专注于 Transformer 中**统计学习理论（statistical learning theory）**的博士学业，并表示渴望扩展对**对齐方法（alignment approaches）**的理解。
   - 他们认为拥有更多的**对齐测试和探测（probes）**将对研究大有裨益。
- **独立研究员探索 Orch-Or Theory**：另一位成员讨论了他们对 **Orch-Or Theory** 及其与 AI 联系的关注，特别是在**诺斯科学（noetic science）**领域。
   - 他们提到之前在 **LLMs** 方面取得的成功，并愿意分享见解并从其他研究领域学习。
- **AI 自动补全的优化挑战**：一位用户询问了提高 AI 自动补全性能的方法，特别是针对一个遇到**延迟问题**的类 **Cursor Tab** 项目。
   - 他们提到目前使用的是 **Groq** 和一个小模型，但正在寻求进一步的优化策略。
- **开发者为项目提供支持**：一位全栈区块链和 AI 开发者表示愿意在新年为项目和团队提供协助。
   - 另一位成员建议他们查看相应的频道以获取参与项目的机会。
- **关于使用 off-topic 频道的见解**：成员们讨论了 **off-topic 频道**的价值，称其为闲聊和严肃讨论的枢纽。
   - 观点强调，**有价值的见解**往往产生于这些对话中，强调了该频道对社区的影响。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1328511336539488307)** (73 messages🔥🔥): 

> `Eukaryotic Cell Vaults, Research Channel Guidelines, Hyper-connections in Neural Networks, Process Reward Models, VinePPO and CoT Trajectories` 


- **Neoxah 讨论真核细胞穹窿 (Eukaryotic Cell Vaults)**：Neoxah 分享了关于真核细胞穹窿的见解，指出它们在噪声环境中保持相干性的独特能力，这对于量子计算至关重要。
   - 然而，为了保持其研究的机密性，他没有在当前的聊天中透露具体细节。
- **研究频道指南**：成员们强调，研究频道旨在讨论已发表的研究并使用科学方法，避开个人经历或资历。
   - 虽然“已发表”一词的使用范围可能较广，但重点仍然是那些有可查阅的报告或文档支持的想法。
- **Hyper-connections 作为残差连接的替代方案**：一种名为 Hyper-connections 的新方法为传统的残差连接提供了替代方案，解决了梯度消失和表示坍缩等挑战。
   - 初步实验表明，该方法在增强语言和视觉任务的模型性能方面，与现有方法相比具有很强的竞争力。
- **Process Reward Models 提升 LLM 的推理能力**：研究引入了 Process Reward Models (PRMs)，旨在通过识别影响结果的关键 token 来增强大语言模型的数学推理能力。
   - 研究表明，在各种基准测试中，操纵这些关键 token 可以显著提高模型的准确性。
- **关于 VinePPO 应用的讨论**：有人询问 VinePPO 如何应用于生成思维链 (CoT) 轨迹，特别是在 LATo 的背景下。
   - 与最初的看法相反，讨论明确了 VinePPO 不需要 CoT 轨迹的示例即可有效运行。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.06425">Tensor Product Attention Is All You Need</a>：扩展语言模型以处理更长的输入序列通常需要大型键值 (KV) 缓存，从而导致推理过程中的巨大内存开销。在本文中，我们提出了 Tensor...</li><li><a href="https://arxiv.org/abs/2411.19943">Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM&#39;s Reasoning Capability</a>：数学推理任务对大语言模型 (LLMs) 构成了重大挑战，因为它们需要精确的逻辑推导和序列分析。在这项工作中，我们引入了...的概念。</li><li><a href="https://arxiv.org/abs/2409.19606">Hyper-Connections</a>：我们提出了 Hyper-connections，这是一种简单而有效的方法，可以作为残差连接的替代方案。这种方法专门解决了在残差连接中观察到的常见缺点...</li><li><a href="https://arxiv.org/abs/2501.07542">Imagine while Reasoning in Space: Multimodal Visualization-of-Thought</a>：思维链 (CoT) 提示已被证明在增强大语言模型 (LLMs) 和多模态大语言模型 (MLLMs) 的复杂推理方面非常有效。然而，它在处理复杂...时仍面临困难。</li><li><a href="https://arxiv.org/abs/2501.06282">MinMo: A Multimodal Large Language Model for Seamless Voice Interaction</a>：大语言模型 (LLMs) 和多模态语音文本模型的最新进展为无缝语音交互奠定了基础，实现了实时、自然且类人的对话...</li><li><a href="https://arxiv.org/abs/2501.07301">The Lessons of Developing Process Reward Models in Mathematical Reasoning</a>：Process Reward Models (PRMs) 成为大语言模型 (LLMs) 数学推理中过程监督的一种极具前景的方法，旨在识别并减少中间过程中的错误...</li><li><a href="https://arxiv.org/abs/2411.07501">LAuReL: Learned Augmented Residual Layer</a>：高效深度学习方法的核心支柱之一是架构改进，例如残差/跳跃连接，这显著提高了模型的收敛性和质量。自...以来。</li><li><a href="https://openreview.net/forum?id=uRHpgo6TMR">Sampling weights of deep neural networks</a>：我们为全连接神经网络的权重和偏置引入了一种概率分布，并结合了一种高效的采样算法。在监督学习背景下，无需迭代...</li><li><a href="https://www.minimaxi.com/en/news/minimax-01-series-2">MiniMax - Intelligence with everyone</a>：未找到描述
</li>
</ul>

</div>

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1328810231123415100)** (6 messages): 

> `Induction Head Bumps, Loss vs Compute, Circuit Interoperability` 


- **Induction Head Bumps 的对齐问题**: 一位用户思考在绘制 **Loss vs Compute** 图表时，Scaling Law 图中的 **Induction Head Bumps** 是否会对齐。
   - 另一位成员澄清说，这通常发生在相同的 Token 数量之后，表明答案是 **否**。
- **训练中 Circuit Interoperability 的分析**: 一位成员引用了之前的 **Anthropic 帖子** 和 **Circuit Interoperability 论文**，展示了不同 Pythia 模型在训练过程中电路出现的图表。
   - 他们提供了论文链接 [这里](https://arxiv.org/abs/2407.10827) 并附上了相关图片以供进一步参考。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1328510849316552847)** (2 messages): 

> `Neel Nanda podcast, Mechanistic interpretability, Neural network understanding` 


- **Neel Nanda 在播客中讨论 SAEs**: Google DeepMind 的高级研究科学家 Neel Nanda 主持了一场播客，讨论了 [SAEs](https://open.spotify.com/episode/5XjHhNQxIb16eJZXGmbaCk?si=Z8LTnSo7QHGJkBxgGZbIJA) 及其在 Mechanistic Interpretability 方面的工作。
   - 他强调了在不清楚内部工作原理的情况下，理解神经网络如何执行复杂任务所面临的挑战。
- **Nanda 反思 AI 的独特挑战**: Nanda 认为 AI 是独特的，因为神经网络可以在人类不理解其编程逻辑的情况下完成卓越的任务。
   - 他将其比作拥有能够完成任何开发者都无法显式编写的任务的软件。



**提到链接**: <a href="https://open.spotify.com/episode/5XjHhNQxIb16eJZXGmbaCk?si=Z8LTnSo7QHGJkBxgGZbIJA">Neel Nanda - Mechanistic Interpretability (Sparse Autoencoders)</a>: Machine Learning Street Talk (MLST) · Episode

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1328473233376153682)** (12 messages🔥): 

> `Pre-commit mixed line ending issue, MLQA benchmark implementation PR, LM Evaluator's majority voting, Filters in LM Evaluation Harness` 


- **Pre-commit 混合换行符 (Mixed Line Ending) 失败**: 一位成员报告称，由于 Unix 和 Windows 系统之间的行字符差异，在 Pre-commit 检查期间未能通过 **'mixed line ending'** 测试。
   - 另一位成员建议运行 `pre-commit run --all-files` 来自动修复该问题。
- **MLQA 基准测试实现 Pull Request**: 一位成员宣布添加了 **MLQA 基准测试实现** 并提交了 PR 进行评审。
   - 他们随后指出了代码中发现的一个 **AST 错误**，并请求对此提供反馈。
- **LM Evaluator 支持 Majority Voting**: 在回答一个问题时，一位成员确认 **lm-eval-harness** 支持 **Majority Voting**，并可以在配置中设置重复次数。
   - 设置此功能的文档可以在 [这里](https://github.com/EleutherAI/lm-evaluation-harness/blob/bb098f13b05e361f01a5afe7b612779ce362b3f2/lm_eval/tasks/gsm8k/gsm8k.yaml#L30) 找到。
- **关于 LM Evaluation Harness 中 Filters 的讨论**: 一位成员引导大家关注关于在 **lm-evaluation-harness** 中实现 Filters 的讨论，并提供了参考链接。
   - 对于那些希望针对特定任务修改现有实现的人来说，这个资源可能会很有用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/">EleutherAI</a>: EleutherAI 有 156 个可用的代码库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/bb098f13b05e361f01a5afe7b612779ce362b3f2/lm_eval/tasks/gsm8k/gsm8k.yaml#L30">lm-evaluation-harness/lm_eval/tasks/gsm8k/gsm8k.yaml at bb098f13b05e361f01a5afe7b612779ce362b3f2 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/6d62a69cb5db963f998c486af6efee43fca63dd3/docs/task_guide.md?plain=1#L57)">lm-evaluation-harness/docs/task_guide.md at 6d62a69cb5db963f998c486af6efee43fca63dd3 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1328621469030289418)** (5 条消息): 

> `AINews Newsletter 的增长、Llama 2 配置差异、梯度裁剪关注点、Tokenizer 填充词表问题、激活函数差异` 


- **AINews Newsletter 专题讨论**：一位成员分享了一个幽默的记录，提到最近的一次讨论登上了 **AINews Newsletter**。
   - 这突显了社区内正在进行的议题具有越来越高的参与度和相关性。
- **对 Llama 2 配置中间层大小（Intermediate Size）的困惑**：一位成员对 **Llama 2** 在 NeoX 和 HF 配置之间的 **padded_vocab_size** 值差异提出质疑，指出存在显著不同（11008 对比 32768）。
   - 这反映了不同实现之间配置缺乏标准化的困惑。
- **在运行期间记录梯度幅值**：一位成员询问如何在模型运行期间记录 **梯度幅值（gradient magnitudes）**，以便在不使用梯度裁剪（gradient clipping）的情况下检测梯度激增。
   - 这表明了一种主动了解模型训练行为和潜在问题的方法。
- **Tokenizer 填充词表大小的怪癖**：一位成员表示沮丧，当将 **padded_vocab_size** 设置为 **50304** 时，它仍然填充到 **50432**，导致了困惑。
   - 日志显示构建过程包含了 **152 个哑标记（dummy tokens）**，暗示处理过程中可能存在配置错误。
- **HF 与 NeoX 激活函数使用的对比**：讨论涉及了激活函数的差异，HF 使用 **silu** 而 NeoX 使用 **swiglu**，后者与原始研究保持一致。
   - 成员们仔细检查了 HF，发现其激活列表中未包含 **swiglu**，认为这可能是一个疏忽。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/EleutherAI/gpt-neox/blob/main/configs/llama2/7B.yml#L9">gpt-neox/configs/llama2/7B.yml at main · EleutherAI/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库在 GPU 上实现模型并行自回归 Transformer —— EleutherAI/gpt-neox</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py">transformers/src/transformers/activations.py at main · huggingface/transformers</a>：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的先进机器学习库。—— huggingface/transformers
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1328492906670981243)** (15 条消息🔥): 

> `代码移除问题、在编辑器中报告问题、启用 Diffs、Google Analytics 4 API 集成、Netlify Function 挑战` 


- **对代码被移除的沮丧**：多位成员对更新时代码意外被移除表示沮丧，这阻碍了项目进度并导致重复劳动。
   - 一位成员指出，更改前导（preamble）部分导致先前内容丢失，具体表现为在一次简单的更新中 **移除了 8 个章节**。
- **简化问题报告流程**：一位成员分享了如何通过“撤销（undo）”选项旁边的按钮直接在编辑器中报告问题，方便遇到持续问题的用户。
   - 这促使另一位用户询问报告方法，并分享了对影响多个项目的持续功能性问题的担忧。
- **启用 Diffs 以改进工作流**：一位成员激活了 Diffs 功能，希望它能准确引用更改，并避免在执行新命令时 **移除现有代码**。
   - 另一位用户发出了警告，称 Diffs 在现有项目上可能会发生故障，导致丢弃的代码比预期的更多。
- **Google Analytics 4 的集成挑战**：一位用户报告了在部署到 Netlify 的 React/Vite 应用中集成 Google Analytics 4 API 的困难，在部署后获取分析数据时收到“Unexpected token”错误。
   - 尽管本地测试运行良好，但他们尝试了多种故障排除步骤，包括确认权限和环境变量设置，但均未成功。
- **Netlify 上 GA4 的替代方案**：同一位用户正在寻求将 GA4 API 与 Netlify Functions 集成的替代建议，考虑潜在的客户端解决方案或不同的分析服务。
   - 这反映了用户对于寻找有效方法来解决在 Netlify 环境中遇到的集成问题的广泛兴趣。


  

---

### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1328453545543995487)** (108 条消息🔥🔥): 

> `Token 消耗问题、Supabase 项目集成、Chatbot 开发策略、订阅计划困惑、Bug 报告与持久性` 


- **订阅升级后的 Token 消耗问题**：用户报告在升级到付费计划后，Token 消耗出现异常增加，部分用户甚至难以完成之前可以实现的简单任务，如修改 UI 元素。
   - 用户担心这些变化可能与公司政策或系统 Bug 有关，因为有用户感觉到功能性显著下降。
- **Supabase 项目管理挑战**：在 Bolt 中 Fork 项目每次都需要创建一个新的 Supabase 项目，限制了重新连接现有项目的能力，这与 Loveable 的功能不同。
   - 用户正在等待团队的解决方案，因为此问题干扰了工作流并使项目管理变得复杂。
- **创建 Perplexity 风格的 Chatbot**：一名成员寻求关于将 Hugging Face 模型集成到 Bolt 的建议，表现出对 Perplexity 风格 Chatbot 的兴趣，而其他人则建议直接使用 OpenAI 的 API 以简化操作。
   - 讨论强调了部署和连接多个 API 平台的潜在复杂性。
- **订阅计划困惑**：用户对订阅升级和 Token 分配流程表示困惑，特别是关于计划变更期间的费用以及随后获得的 Token 数量。
   - 建议针对不同订阅层级转换期间的价格结构和 Token 分配进行澄清。
- **Bug 报告与用户反馈**：用户注意到更新后重复出现的 Bug，这导致了挫败感，并认为平台对这些问题的响应不足。
   - 建议用户记录错误并考虑进行手动检查以确保系统的可靠性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://support.bolt.new/WebContainer-Startup-Error-159d971055d680fa9af5dafcdb358f42">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://www.npmjs.com/package/emoji-picker-react">emoji-picker-react</a>：适用于 Web 端 React 应用的表情选择器组件。最新版本：4.12.0，最后发布于 4 个月前。通过运行 `npm i emoji-picker-react` 开始在您的项目中使用。</li><li><a href="https://support.bolt.new/Prompting-Effectively-How-to-talk-to-Bolt-13fd971055d6801b9af4e965b9ed26e2">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://remix.run/guides/errors">错误处理 | Remix</a>：未找到描述。
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1328454835724488745)** (1 条消息): 

> `Telegram AI Chatbot、DeVries AI 订阅模式、Telegram 中的多模型访问` 


- **DeVries AI 将 Telegram 转换为 LLM 界面**：全新的 [DeVries AI](https://devriesai.com/) 允许用户以低廉的订阅费用直接在 Telegram 中与 **200 多个 Large Language Models** 进行对话。
   - 在决定订阅之前，可以先免费试用 AI 进行聊天。
- **单一聊天中的流式 AI 访问**：DeVries AI Chatbot 在单一且熟悉的 Telegram 环境中提供了对 **ChatGPT** 和 **Claude** 等热门模型的访问。
   - 用户可以进行文本交互，并且很快将支持使用该集成方案进行**图像和视频生成**。
- **经济实惠的 AI 订阅方案**：每月仅需 **$24.99**，用户即可访问所有当前和未来的生成式 AI 模型，无需多个订阅。
   - 该模式允许在模型之间轻松切换，并能抢先体验新发布的模型。



**提到的链接**：<a href="https://devriesai.com/">devriesai</a>：您的 Telegram AI Agent

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1328491156748632115)** (106 messages🔥🔥): 

> `OpenRouter 提供商设置, Deepseek 性能问题, OpenRouter 速率限制, 动漫讨论, MiniMax 模型发布` 


- **如何成为 OpenRouter 提供商**：要成为提供商并在 OpenRouter 上部署模型，可以通过电子邮件 `support@openrouter.ai` 联系以获取帮助。
   - 一位用户富有创意地推测创建一个秘密使用 OpenRouter 的提供商，引发了关于可能发明了 AGI 的幽默回应。
- **Deepseek V3 性能下滑**：用户报告称 **Deepseek** 响应经常缓慢，一位成员表示 **10 次中有 7 次** 无法获得回复。
   - 据信响应缓慢是由于**过载**引起的，一些用户建议切换到 Together AI endpoint 等替代方案。
- **了解 OpenRouter 的速率限制**：新用户会获得一些免费额度用于测试，但速率限制取决于购买的额度数量，详见 [OpenRouter 文档](https://openrouter.ai/docs/limits)。
   - 几位用户讨论了缺乏企业账户的问题，强调了为了测试目的需要更高的速率限制。
- **动漫系列推荐引发对话**：关于各种动漫系列（尤其是 **Fate 系列**）展开了热烈讨论，用户们热情地推荐了不同的作品。
   - 对话幽默地转向了用户与动漫作品的互动，展示了共同的经历和偏好。
- **MiniMax 模型发布公告**：拥有惊人的 **4560 亿参数** 的新 **MiniMax** 模型的发布因其出色的上下文处理能力而受到关注。
   - 虽然在基准测试中不是 SOTA，但其效率和容量使其成为 AI 领域中一个潜在的有价值工具。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models?q=free">Models: &#x27;free&#x27; | OpenRouter</a>: 在 OpenRouter 上浏览模型</li><li><a href="https://tenor.com/view/dum-suspense-climax-monkey-shocked-gif-8054274">Dum Suspense GIF - Dum Suspense Climax - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>: 设置模型使用限制</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: 未找到描述</li><li><a href="https://en.m.wikipedia.org/wiki/Fate/Zero">Fate/Zero - Wikipedia</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/ClaudeAI/comments/1i0eoje/mentionaiio_get_answers_easily_from_multiple_ai/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://huggingface.co/MiniMaxAI/MiniMax-Text-01">MiniMaxAI/MiniMax-Text-01 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/MiniMaxAI/MiniMax-VL-01">MiniMaxAI/MiniMax-VL-01 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1328566927374684180)** (104 messages🔥🔥): 

> `Discord 机器人问题, 使用 DEIS BETA 进行 Flux 采样, 美学分类器开发, FP8 vs FP16 精度, Intel B580 在 Stable Diffusion 中的性能` 


- **Discord 机器人引发讨论**：成员们对 Discord 的机器人问题表示担忧，指出那些加入只是为了打个招呼的用户本身看起来就很像机器人。
   - *一位成员提到*，更严格的入职流程可以阻挡机器人，但也可能让真实用户更难加入。
- **关于使用 DEIS BETA 的建议**：出现了关于利用 **DEIS BETA** 进行 Flux 采样设置的讨论，一些成员分享了他们的积极体验。
   - *一位成员询问其他人是否也有* 类似的设置或工具推荐以供探索。
- **训练美学分类器的想法**：一位成员正在寻找结合了艺术和客观评分的资源，用于训练**美学分类器**。
   - *建议包括使用像 ollama 这样的工具* 来辅助提示词编写，这可能会增强工作流。
- **关于 FP8 与 FP16 支持的辩论**：成员们讨论了为什么 **FP8** 支持仅出现在 40 系列等较新的显卡中，并质疑其相对于 **FP16** 的潜在计算优势。
   - 对话显示，虽然**低精度**可以节省内存，但并不总能为所有任务产生所需的准确度。
- **Intel B580 的性能担忧**：一位成员分享了由于 subreddit 过滤器的原因，在发布关于其 **Intel B580 性能** 在 Stable Diffusion 中计算结果时遇到的困难。
   - *另一位成员建议联系* 管理员寻求帮助，将对话转向寻找更好的发布替代方案。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1328573006812811355)** (28 messages🔥): 

> `Qwen2.5-Math-PRM, Claude Sonnet 3.5 性能, MiniCPM-o 2.6 介绍, Midwit 模型, 模型评估方法` 


- **Qwen2.5-Math-PRM 正式亮相**：新的过程奖励模型 (Process Reward Model) **Qwen2.5-Math-PRM** 与 **ORM** 一道，通过识别中间错误，在 [ProcessBench](https://huggingface.co/papers/2412.06559) 上表现出色，展现了 LLM 过程监督的前景。
   - 更多详情请查看他们的 [论文](https://arxiv.org/pdf/2501.07301)，其中概述了他们的评估方法。
- **Claude Sonnet 缩小差距**：在 **OpenAI 的 o3 模型** 在 SWE-Bench Verified 上达到 **71.7%** 后，使用 CodeStory 的 **Claude Sonnet 3.5** 达到了 **62.2%**，展示了其强大的编程能力。
   - *一个上一代的非推理模型能达到尚未发布的未来模型 10% 以内的差距*，凸显了 Sonnet 在编程方面的极高效率。
- **MiniCPM-o 2.6 表现惊艳**：由 OpenBMB 推出的 **MiniCPM-o 2.6** 是一款 **8B 尺寸的 Omni Model**，其能力在多种模态下可媲美 GPT-4o，并支持**实时双语语音对话**。
   - 请在 [GitHub](https://github.com/OpenBMB/MiniCPM-o) 和 [Hugging Face](https://huggingface.co/openbmb/MiniCPM-o-2_6) 上查看 Demo 和代码。
- **Midwit 模型利用 Claude 进行评分**：**Midwit** 模型实现了以 **Claude** 作为其过程奖励模型的 **MCTS**，对动作进行 -100 到 100 的评分，其有效性引发了关注。
   - *“看起来这不应该像现在表现得这么好”* 表达了对其在如此设定下仍能取得高性能的疑虑。
- **评估方法引发讨论**：社区分享了对评估方法的看法，指出目前的 **vibe**（直观感受）是当前模型在训练后可能显得**训练不足 (undercooked)**。
   - 观察到的挑战包括“毁灭循环 (doom loops)”以及多轮对话中的性能下降，引发了对评估完整性的质疑。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenBMB/status/1879074895113621907">OpenBMB (@OpenBMB) 的推文</a>: 💥 介绍 MiniCPM-o 2.6：一个 8B 尺寸、GPT-4o 级别的 Omni Model，可在设备端运行 ✨ 亮点：在视觉、音频和多模态实时流方面基本媲美 GPT-4o-202405 ~ 端到端实时双语语音...</li><li><a href="https://x.com/TheXeophon/status/1879254534465490984">Xeophon (@TheXeophon) 的推文</a>: Vibe-Testing：- 2024 年 5 月截止日期 - HF UI 表现更好，可能是另一个模型（非常不确定）- 非常喜欢对所有事情进行 CoT - 因此质量波动很大 - 至少可以 unde...</li><li><a href="https://x.com/deedydas/status/1877549539781128319?t=hFlLBI6S6s0xaB2ciDeztw&s=19">Deedy (@deedydas) 的推文</a>: 相当疯狂，在 OpenAI o3 在 SWE-Bench Verified 达到 71.7% 后，昨天使用 CodeStory 的 Claude Sonnet 3.5 达到了 62.2%。一个“上一代”非推理模型达到尚未发布的未来模型的 10% 以内...</li><li><a href="https://huggingface.co/MiniMaxAI/MiniMax-Text-01">MiniMaxAI/MiniMax-Text-01 · Hugging Face</a>: 暂无描述</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-Math-PRM-72B">Qwen/Qwen2.5-Math-PRM-72B · Hugging Face</a>: 暂无描述</li><li><a href="https://x.com/teortaxestex/status/1879273615960743995?s=46">Teortaxes▶️ (@teortaxesTex) 的推文</a>: 相比于由于 SYNTHETIC DATA（合成数据）导致的“审美坍塌 (TASTE COLLAPSE)”这一灾难，由于合成数据导致的模型崩溃完全是微不足道的（被 Marcus 式的反统计应对和受伤的艺术家们高估了）...</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6">openbmb/MiniCPM-o-2_6 · Hugging Face</a>: 暂无描述</li><li><a href="https://minicpm-omni-webdemo-us.modelbest.cn/">MiniCPM-omni</a>: 暂无描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/)** (1 messages): 

420gunna: https://x.com/aidan_mclau/status/1878944278782890158
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1328574394892554271)** (31 条消息🔥): 

> `高等教育中的 AI、CIO 职能、密歇根大学的聊天机器人服务、LLM 的局限性、Stripe 税务注册` 


- **为 CIO 导航 AI 演讲**：一位成员正在准备一场针对高等教育 CIO 的 **AI** 演讲，并正在考虑 LLM 功能、Prompting 和使用案例等主题。
   - 建议集中在处理 **敏感数据管理** 以及将讨论与 AI 的实际应用对齐。
- **密歇根大学的聊天机器人计划**：一位成员强调了密歇根大学一位管理员关于其 **定制 GenAI 工具** 的演讲，重点强调了公平性和隐私。
   - 这些服务面向大学社区开放，提供如 **U-M GPT** 和 **Maizey** 等工具以实现个性化体验。
- **CIO 们直面 LLM 的局限性**：讨论中提到 CIO 经常面临对 **LLM** 能力的不切实际要求，呼吁制定指南来区分可行用途与科幻概念。
   - 与会者建议，明确不同 AI 实现中涉及的 **成本和资源** 有助于设定合理的预期。
- **在欧洲使用 Stripe 处理税务**：一位成员分享了关于欧洲税务注册的见解，特别是使用 **Stripe** 的 Non-Union One Stop Shop (OSS) 进行高效的 VAT 管理。
   - 这允许非欧盟企业在一个国家注册，管理整个 **EU** 的 VAT，并简化了申报流程。
- **Stripe 简化欧盟税务注册**：针对 Stripe 的 **Non-Union OSS** 命名出现了一些讨论，该服务为外部企业提供了流线化的税务计算和支付流程。
   - 该方案使这些企业无需在每个欧盟国家注册即可代收和缴纳 VAT，只需提交一次年度付款。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2025/1/14/24343528/openai-chatgpt-repeating-tasks-agent-ai">ChatGPT 现在可以处理提醒和待办事项</a>：ChatGPT 刚刚获得了针对未来任务的新自动化功能。</li><li><a href="https://x.com/__tinygrad__/status/1879034330284192050">来自 tiny corp (@__tinygrad__) 的推文</a>：我们从 AMD 的驱动切换到了我们的 AM 驱动……现在 4 GPU Llama 在红盒子（red box）上比绿盒子（green box）更快！</li><li><a href="https://genai.umich.edu">欢迎 | 密歇根大学生成式 AI</a>：未找到描述</li><li><a href="https://docs.stripe.com/tax/supported-countries/european-union#outside-eu-businesses">在欧盟征收税款</a>：了解如何使用 Stripe Tax 在欧盟计算、征收和申报税款。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/1328754889945518182)** (16 条消息🔥): 

> `Synthetic Chain-of-Thought Training, O1 Models Discussion, Tulu-R and Vision Generalists, Naming Conventions for Models, Molmo and Reinforcement Learning` 


- **合成 Chain-of-Thought 训练表现平平**：一位成员对某篇论文仅涉及 **synthetic chain-of-thought training data** 和 **supervised fine-tuning**，而缺乏任何 **RL** 或类似训练表示失望。
   - 这引发了关于标题中带有 **O1** 的模型普遍质量的讨论，暗示它们的质量可能较低。
- **O1 模型难成大器**：另一位成员指出 **O1 mode** 不会实现，暗示了对当前模型局限性的沮丧。
   - 他们更倾向于使用 **Big Molmo** 或 **Tulu-V** 作为视觉任务的更好替代方案，强调了对性能提升的渴望。
- **命名规范引发批评**：讨论转向了命名规范，一位成员将一个不理想的名字比作 **olmo-Flash**，表示这听起来很尴尬。
   - 另一个例子是 **olmo-Gemini**，强调了荒谬的命名如何削弱可信度。
- **尽管受到批评，仍预期会有高引用量**：一位参与者推测，尽管命名荒谬引发了抵制，但使用像 **O1mo** 这样的名字可能仍会获得相当数量的引用。
   - 这进一步强调了模型名称与其在学术界潜在影响力之间的脱节。
- **Molmo 与 Reinforcement Learning 的协作**：人们对 **Molmo** 可能与 **Reinforcement Learning** 结合感到兴奋，并对适用的视觉任务产生了好奇。
   - 这引发了关于通过新的、更具动态性的模型来增强视觉能力的推测。



**提到的链接**：<a href="https://arxiv.org/abs/2501.06186v1">LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs</a>：推理是解决复杂多步问题的基本能力，特别是在视觉语境中，连续的逐步理解至关重要。现有方法缺乏全面的...

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1328579778290389096)** (17 messages🔥): 

> `Process Reward Models (PRMs), ProcessBench Benchmarking, Nathan Lambert's Post-Training Insights, LLaMA 3 Paper Discussion` 


- **探索过程奖励模型 (PRMs)**：关于过程奖励模型 (PRMs) 的论文讨论了它们在解决 **LLM** 推理中中间错误方面的潜力，但指出了在 **数据标注 (data annotation)** 和 **评估方法论 (evaluation methodologies)** 方面的重大挑战；研究发现蒙特卡洛数据合成逊于人工标注方法。
   - 作者强调了 **Best-of-N 评估策略** 中潜在的偏见，即不可靠的策略模型可能会生成正确的答案，但推理过程存在缺陷。
- **引入 ProcessBench 用于错误识别**：**ProcessBench** 论文专注于评估模型识别 **数学推理** 错误的能力，提供了 **3,400 个测试用例** 及其标注的解决方案用于评估。
   - 测试了两种类型的模型：**PRMs** 和 **critic models**；研究指出，现有的 PRMs 通常无法泛化到更复杂的数学问题。
- **Nathan Lambert 分享 Post-Training 见解**：Nathan Lambert 在一篇文章中分享了他的见解，详细介绍了他在 **Allen Institute for Artificial Intelligence** 从事 **Post-Training** 和推理工作的经验，预示着现代模型训练不断演进的本质。
   - 在 12 月 17 日的一次对话中，他详细阐述了训练现代模型的步骤，并讨论了其组织对开放模型训练的承诺。
- **LLaMA 3 论文赞赏**：一位成员表示 **LLaMA 3 论文** 让通往 **TULU 3** 的路径更加清晰，即使他们主要阅读了相关部分并略读了其余内容。
   - 强调了阅读完整论文的难度，突显了在当前研究背景下资料量是多么庞大。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.07301">The Lessons of Developing Process Reward Models in Mathematical Reasoning</a>: 过程奖励模型 (PRMs) 成为 **LLM** 数学推理中过程监督的一种有前景的方法，旨在识别和缓解中间错误...</li><li><a href="https://arxiv.org/abs/2412.06559">ProcessBench: Identifying Process Errors in Mathematical Reasoning</a>: 由于语言模型在解决数学问题时经常出错，自动识别推理过程中的错误对于其可扩展监督变得越来越重要。在本文中...</li><li><a href="https://open.substack.com/pub/aisummerpodcast/p/nathan-lambert-on-the-rise-of-thinking?r=68gy5&utm_medium=ios">Nathan Lambert on the rise of &quot;thinking&quot; language models</a>: 传统的 **LLM** 缩放“在目前这个阶段，大多数 ChatGPT 用户不会感受到任何区别。”
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1328634401718009897)** (3 messages): 

> `Economic Blueprint, Executive Order for Datacenters` 


- **经济蓝图旨在推动 AI 进步**：一份新发布的 [经济蓝图 (Economic Blueprint)](https://openai.com/global-affairs/openais-economic-blueprint/) 概述了美国的政策建议，旨在最大限度地发挥 **AI** 的效益、加强国家安全并推动经济增长。
   - 讨论中一位成员强调了该蓝图，并评论说这些政策建议是熟悉叙事的一部分，且具有持续性。
- **拜登签署数据中心建设行政命令**：拜登总统签署了一项行政命令，允许在联邦土地上建设吉瓦级数据中心，**DoD** 和 **DoE** 都将为此出租土地。
   - 该命令规定，必须在现场建设足够的 **清洁能源** 以匹配数据中心的容量，从而确保采用环保方法。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/gdb/status/1879059072135414236">Greg Brockman (@gdb) 的推文</a>: 刚刚发布了我们的经济蓝图 —— 关于美国如何最大限度发挥 **AI** 效益、加强国家安全并推动经济增长的政策建议。https://openai.com/global-affairs/openais-econo...</li><li><a href="https://x.com/AndrewCurran_/status/1879174379718004881">Andrew Curran (@AndrewCurran_) 的推文</a>: 今天早上拜登总统签署了一项行政命令，开放联邦土地用于建设吉瓦级数据中心。**DoD** 和 **DoE** 都将出租土地，并提供足够的清洁能源以匹配容量...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1328464883150360597)** (82 messages🔥🔥):

> `ChatGPT 任务自动化，Cursor B 轮融资，AI 邮件助手，模型性能与容量问题，GPU 投资策略` 


- **ChatGPT 推出任务调度功能 (Task Scheduling Feature)**：ChatGPT 现在允许用户设置一次性和周期性提醒，增强了其作为数字助手的功能。该功能正向 Plus, Team 和 Pro 用户推出，支持每日天气更新和提醒通知等任务。
   - OpenAI 旨在将 ChatGPT 定位为不仅仅是一个聊天界面，并试图通过这一新的 Task 功能探索 AI Agent 的潜力。
- **Cursor 完成由 a16z 领投的 B 轮融资**：Cursor 宣布了由 a16z 共同领投的 B 轮融资，表明了对其编程工具和 AI 产品的强力支持。这笔投资凸显了市场对 AI 驱动的开发工具日益增长的兴趣以及潜在的高回报。
   - 本轮融资反映了对创新编程解决方案不断增长的需求，以及对 Cursor 在该领域未来发展的积极展望。
- **推出环境 AI 邮件助手 (Ambient AI Email Assistants)**：一种 AI 邮件助手已被开发出来，可以在无需用户直接参与的情况下对邮件进行分类和起草，展示了被称为“环境 Agent (ambient agents)”的新概念。该技术已开源，允许更广泛的应用尝试。
   - 这一创新旨在减少与传统邮件管理相关的开销，为用户提供更高效、干扰更少的邮件处理方式。
- **Cursor 上的 Sonnet 3.6 出现速率限制 (Rate Limiting) 问题**：用户在 Cursor 上使用 Claude 模型 Sonnet 3.6 时遇到了速率限制问题，引发了对请求处理能力的担忧。开发者表示，这源于超过 Anthropic 的 GPU 可用性的高流量。
   - 据报道，Cursor 是 Anthropic 最大的客户，这使得增加 GPU 资源以应对需求变得迫在眉睫。
- **Magnetar 在 AI 投资领域的策略**：对冲基金 Magnetar 正在通过提供算力资源换取股权的方式，解决 AI 初创公司融资难的挑战，旨在打破 AI 行业的融资瓶颈。这种方法代表了一种支持在获得投资前需要算力资源的初创公司的新颖策略。
   - 他们与 Coreweave 的合作突显了基础设施在推动 AI 进步中的重要性，并展示了科技行业中创新的融资解决方案。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/__tinygrad__/status/1879034330284192050">来自 tiny corp (@__tinygrad__) 的推文</a>：我们从 AMD 的驱动切换到了我们的 AM 驱动...现在 4 GPU Llama 在 red box 上比 green box 更快了！</li><li><a href="https://x.com/gdb/status/1879059072135414236">来自 Greg Brockman (@gdb) 的推文</a>：刚刚发布了我们的经济蓝图（Economic Blueprint）——关于美国如何最大化 AI 收益、加强国家安全并推动经济增长的政策建议。https://openai.com/global-affairs/openais-econo...</li><li><a href="https://www.theverge.com/2025/1/14/24343528/openai-chatgpt-repeating-tasks-agent-ai">ChatGPT 现在可以处理提醒和待办事项</a>：ChatGPT 刚刚获得了一项针对未来任务的新自动化功能。</li><li><a href="https://www.reforge.com/blog/ai-native-product-teams">Reforge</a>：未找到描述</li><li><a href="https://blog.langchain.dev/introducing-ambient-agents/">介绍环境 Agent (ambient agents)</a>：当今大多数 AI 应用都遵循熟悉的聊天模式（"chat" UX）。虽然易于实现，但它们创造了不必要的交互开销，限制了我们人类扩展自身的能力，并且...</li><li><a href="https://x.com/sarahdingwang/status/1879279307119608142">来自 Sarah Wang (@sarahdingwang) 的推文</a>：很高兴宣布 @a16z 共同领投了 @cursor_ai 的 B 轮融资。我们非常激动能继续与 Cursor 团队合作，看他们席卷编程世界。</li><li><a href="https://x.com/hwchase17/status/1879218872727015644">来自 Harrison Chase (@hwchase17) 的推文</a>：在过去的六个月里，我没有直接检查电子邮件，而是依靠一个 AI 邮件助手来帮我分类和起草邮件。这是一个“环境 Agent”（ambient agent）的例子。我们...</li><li><a href="https://x.com/swyx/status/1838663794320642328">来自 swyx.io (@swyx) 的推文</a>：2024 年 9 月更新 https://x.com/Smol_AI/status/1838663719536201790 引用 Smol AI 的 AI 新闻：Lmsys Elo 与价格曲线的预测性是多么显著，以及该策略是多么...</li><li><a href="https://x.com/ilanbigio/status/1878940258349510764?s=46">来自 ilan bigio (@ilanbigio) 的推文</a>：宣布我们全新的 @openai function calling 指南！我们听取了你们的反馈并做了一些关键改进：缩短了 50% 且更清晰、新的最佳实践（详见下文 👇）、文档内函数生成...</li><li><a href="https://x.com/kevinweil/status/1879275151969141193">来自 Kevin Weil 🇺🇸 (@kevinweil) 的推文</a>：💥 ChatGPT 的新功能 💥 现在你可以安排任务，包括一次性和重复性任务。* 每天早上 8 点和下午 5 点给我发送最新的 AI 新闻 * 每日天气、星座运势等 * 安排一些有趣的...</li><li><a href="https://x.com/fofrAI/status/1878807358887154044">来自 fofr (@fofrAI) 的推文</a>：听听 Kokoro-82M... 在 T4 上仅用 4.5 秒就生成了 2 分 25 秒的语音。https://replicate.com/jaaari/kokoro-82m https://replicate.com/p/k3cg51x8vdrga0cmbzj9ttswzg 如此优雅，如此精准。</li><li><a href="https://x.com/btibor91/status/1876923634675315100">来自 Tibor Blaho (@btibor91) 的推文</a>：最近几小时部署了 3 个新的 ChatGPT Web 应用版本 - 新的自定义指令 UX（“ChatGPT 应该如何称呼你？”、“你是做什么的？”、“ChatGPT 应该具备哪些特质？” -...</li><li><a href="https://x.com/satyanadella/status/1878578314115473577?s=46">来自 Satya Nadella (@satyanadella) 的推文</a>：GitHub Copilot Workspace 不再有等待名单——这是最先进的 Agentic 编辑器。今天就开始使用 Agent 进行构建吧。</li><li><a href="https://podcasts.apple.com/ca/podcast/how-the-hedge-fund-magnetar-is-financing-the-ai-boom/id1056200096?i=1000679726051&l=fr-CA">对冲基金 Magnetar 如何资助 AI 热潮</a>：播客剧集 · Odd Lots · 2024-12-09 · 50 分钟</li><li><a href="https://forum.cursor.com/t/anthropic-cannot-sustain-additional-slow-request-traffic-on-claude-3-5-sonnet-please-enable-usage-based-pricing/41361/24?">Anthropic 无法承受 Claude 3.5 Sonnet 额外的慢速请求流量。请启用基于用量的计费</a>：毫无疑问，我们是他们最大的客户。</li><li><a href="https://x.com/BlackHC/status/1878883222911877375">来自 Andreas Kirsch 🇺🇦 (@BlackHC) 的推文</a>：NeurIPS 2024 的程序委员会（PCs）简直是一群小丑 🤡 ML 的现状 🙄 在提出担忧一个月后，你得到的只有：</li><li><a href="https://x.com/svpino/status/1878797424590012907">来自 Santiago (@svpino) 的推文</a>：这是我见过的运行最快的 Llama 3.3！Llama 3.3 70B 以 652 t/s 的速度运行，简直快如闪电。如果你想要 Llama 3.1，以下是我能达到的速度：• Llama 3.1 8B: 1006 t/...</li><li><a href="https://x.com/btibor91/status/1869119330560147690">来自 Tibor Blaho (@btibor91) 的推文</a>：还记得 “Jawbone” 吗？它是 ChatGPT “任务”（Tasks/“jawbones”）的代号 - “让 ChatGPT 在未来执行任务” - 选择未来的日期、任务名称和指令...</li><li><a href="https://huggingfa">

ce.co/Qwen/Qwen2.5-Math-PRM-72B">Qwen/Qwen2.5-Math-PRM-72B · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2501.07301">The Lessons of Developing Process Reward Models in Mathematical Reasoning</a>：过程奖励模型 (PRMs) 成为 Large Language Models (LLMs) 数学推理中过程监督的一种极具前景的方法，旨在识别和减轻中间错误...</li><li><a href="https://arxiv.org/abs/2412.06559">ProcessBench: Identifying Process Errors in Mathematical Reasoning</a>：由于语言模型在解决数学问题时经常出错，自动识别推理过程中的错误对于其可扩展监督变得越来越重要。在本文中...</li><li><a href="https://techcrunch.com/2025/01/14/chatgpt-now-lets-you-schedule-reminders-and-recurring-tasks/">ChatGPT now lets you schedule reminders and recurring tasks | TechCrunch</a>：OpenAI 的 ChatGPT 付费用户现在可以要求 AI 助手安排提醒或重复请求。这项名为 tasks 的新 Beta 功能将开始...</li><li><a href="https://docs.google.com/spreadsheets/d/1x9bQVlm7YJ33HVb3AGb9qlDNkvTy9CyOFZoah0kr3wo/edit?gid=0#gid=0">LLM elo vs pricing chart</a>：未找到描述</li><li><a href="https://youtu.be/SN4Z95pvg0Y?si=wyrwJ1VeV2BFElLG">How AI Took Over The World</a>：一个见解改变了一切……智能可以从模式预测中产生。这是一段总结视频，包含了整个 AI 系列的核心见解……</li><li><a href="https://www.minimaxi.com/en/news/minimax-01-series-2">MiniMax - Intelligence with everyone</a>：未找到描述</li><li><a href="https://www.gov.uk/government/publications/ai-opportunities-action-plan/ai-opportunities-action-plan">AI Opportunities Action Plan</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1328466097116282910)** (2 条消息): 

> `HN 上的阅读清单，用户成就` 


- **HN 首页展示阅读清单**：一位用户分享了对 **Hacker News 首页阅读清单** 的兴奋之情，并附上了 [此处](https://cdn.discordapp.com/attachments/1075282504648511499/1328466096898048031/Screenshot_2025-01-13_at_12.49.35_PM.png?ex=67881f77&is=6786cdf7&hm=b3feed5680389cbe58f5d83bd76ec7564561c423a8af357ed9b8ae21cd1c5730&) 的截图链接。
   - 这一讨论突显了社区对精选内容的重视和兴趣。
- **用户庆祝个人里程碑**：一位成员幽默地惊呼 **"FINALLY MADE IT lmao"**，表达了对个人成就的兴奋。
   - 这反映了社区互动和共同体验的趣味性。


  

---


### **Notebook LM Discord ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1328458497070731334)** (1 条消息): 

> `音频概览 (Audio Overviews) 反馈调查，用户研究参与` 


- **参与音频概览反馈调查**：Google 团队正通过一份 5 分钟的快速筛选问卷征求对 **Audio Overviews** 的反馈，可通过 [此表单](https://forms.gle/NBzjgKfGC24QraWMA) 访问。
   - 完成调查的用户将通过电子邮件收到 **10 美元礼品码**，以感谢其提供的意见。
- **用户研究参与详情**：符合条件的参与者必须年满 18 岁才能参加调查，这将有助于根据用户需求进行未来的产品改进。
   - 需要注意的是，奖励仅在完成完整调查后发放，而非填写意向表。



**提到的链接**：<a href="https://forms.gle/NBzjgKfGC24QraWMA">登记您的意向：Google 反馈调查</a>：您好，我们正通过一项简短调查征求对 NotebookLM 的反馈。这将帮助 Google 团队更好地了解您的需求，以便将其纳入未来的产品增强功能中。如需登记...

  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1328463586476949614)** (29 条消息🔥): 

> `AI 生成的播客，在 Google Classroom 中组织笔记本，生成播客的长度控制，音频转录选项，NotebookLM 的共享权限` 


- **AI 生成播客的分享网站**：一位成员讨论了一个名为 [Akash](https://akashq.com) 的新平台，旨在**轻松分享 AI 生成的播客**，无需账户权限。
   - 它允许用户无缝上传、分享和分发从 NotebookLM 等平台创建的播客。
- **无法在 Google Classroom 中组织笔记本**：一位用户对无法在 Google Classroom 的 NotebookLM 中**分类笔记本**表示沮丧，并建议增加文件夹创建功能。
   - 他们提出，更多的组织功能将提高用户体验和效率。
- **音频文件的转录选项**：一位用户发现了一个通过将文件作为来源上传来转录音频的变通方法，但建议增加一个直接转录按钮会更高效。
   - 这一想法旨在简化 NotebookLM 中将音频转换为文本的过程。
- **控制播客长度**：一位成员询问了限制**生成播客长度**的方法，寻求设定所需时长的选项。
   - 另一位参与者分享了一个 [Reddit 链接](https://www.reddit.com/r/notebooklm/comments/1gmtklt/longest_pod_i_have_ever_got/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)，表明用户在控制输出长度方面取得了一些成功。
- **以音频格式嵌入演讲**：另一位用户分享了 **Martin Luther King Jr. 的 'I Have a Dream'** 演讲和 **Gettysburg Address** 的音频文件，并强调了它们的可用性。
   - 这反映了用户对于通过音频交付方式在平台上使用历史或文学演讲的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://akashq.com">Akas: AI 播客之家</a>: 未找到描述</li><li><a href="https://notebooklm.google.com/notebook/6dd1946b-561b-446c-818a-e9e17e332aac/audio">未找到标题</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/notebooklm/comments/1gmtklt/longest_pod_i_have_ever_got/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.akashq.com/post/51eae1b7-e011-4d66-83af-873d763a203d">1 月 14 日发生了什么？</a>: 1 月 14 日发生了什么？ 选自 This Day in History</li><li><a href="https://www.akashq.com/post/2e2231bf-907b-4805-84ae-71f8a7a45c19">1 月 13 日发生了什么？</a>: 1 月 13 日发生了什么？ 选自 This Day in History
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1328455331592016012)** (52 条消息🔥): 

> `NotebookLM 功能、付费版特性、用户反馈环节、音频摘要问题、项目共享` 


- **讨论增强 NotebookLM 功能的方法**：用户正在探索添加自定义功能的选项，例如修改播客对话或更改声音和个性。
   - 建议使用 custom instructions 作为权宜之计，尽管许多功能尚未得到完全支持。
- **关于付费版 NotebookLM 特性的查询**：一位用户询问付费版本是否允许公开共享项目，而无需单独的访问权限。
   - 已确认目前尚不支持公开共享，但可以在组织内部进行共享。
- **关于音频摘要生成 Bug 的反馈**：用户对 audio overview 功能在生成的摘要中无法保留指向已上传 PDF 文档链接的问题表示担忧。
   - 据指出，该功能目前已失效，团队正在努力恢复。
- **探索 NoCode RAG 解决方案**：一位用户表示有兴趣寻找 NoCode 解决方案，以便从存储在 Google Drive 中的 PDF 生成答案。
   - 社区承认将此类功能与 NotebookLM 集成的复杂性。
- **用户使用 NotebookLM 的体验**：用户分享了使用 NotebookLM 进行研究任务的积极体验，但也指出了需要改进的地方。
   - 建议包括改进引用导出功能以及修复与文档摘要相关的当前 Bug。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://thedrive.ai?">The Drive AI: Revolutionizing File Management &amp; Knowledge Bases</a>: 探索 The Drive AI 在智能文件组织方面的突破。我们的平台借助尖端 AI 将您的文件转化为动态知识库。提升您的业务运营...</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en">Upgrading to NotebookLM Plus - NotebookLM Help</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/notebooklm/comments/1gmtklt/longest_pod_i_have_ever_got/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1328459603049971774)** (74 条消息🔥🔥): 

> `Perplexity Pro 功能、用户体验问题、编程辅助挫败感、API 访问限制、内容可见性担忧` 


- **Perplexity Pro 的评价褒贬不一**：一些用户对 **Perplexity Pro** 表示不满，认为它虽然对研究有益，但在编程辅助方面效果不佳。
   - 针对无法使用兑换的奖励代码以及无法无缝访问 Pro 功能的问题，用户提出了担忧。
- **UI 更改与用户控制**：成员们抱怨不想要的广告和 Perplexity 界面的更改，认为这降低了可用性。
   - 一位用户建议使用 **Ublock Origin** 来消除干扰内容，这表明了对平台发展方向的沮丧。
- **对编程助手的挫败感**：一位用户详细描述了对编程助手的挫败感，因为它在不应该询问时，反复要求确认是否提供完整代码。
   - 尽管提供了具体的指令，AI 仍然无视用户请求，导致挫败感增加。
- **对 API 访问和功能的需求**：有用户询问现在是否可以通过 API 访问 **Pro search**，用户渴望获得该功能的更新。
   - 回复显示目前 API 尚不支持 Pro search，这让希望将其集成到工作流中的用户感到沮丧。
- **对内容可见性的担忧**：用户担心他们发布的页面是私密的且未被 Google 收录，并质疑 **enterprise subscription** 是否会影响可见性。
   - 一位用户指出无法访问之前上传的文档，导致难以有效地利用内容。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai">未找到标题</a>: 未找到描述</li><li><a href="https://apps.microsoft.com/detail/9P02M3TKS0RJ?hl=ko&gl=KR&ocid=pdpshare">AI Search - powered by perplexity - 在 Windows 上免费下载安装 | Microsoft Store</a>: 基于 perplexity API 的强大 AI 搜索。知识从哪里开始。你需要的答案就在指尖。抛开所有噪音，直接获取可靠且最新的答案...</li><li><a href="https://newsletter.moneylion.com/subscribe?ref=yJmsSyv2l7">MoneyLion Markets Daily Newsletter</a>: 你的每日市场新闻剂量
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1328694498729000972)** (2 条消息): 

> `TikTok 监管、德语摘要请求` 


- **中国官员关注 TikTok 监管**：中国官员正在考虑围绕 **TikTok** 的新法规，重点关注其影响和控制措施，如[此处](https://www.perplexity.ai/page/chinese-officials-consider-tik-51PEvvekQxqVuhd74SkSHg)的文章所述。
   - 讨论暗示了对该平台相关的 **content moderation** 和 **data privacy** 问题的日益关注。
- **用户寻求德语摘要**：一位用户请求对特定主题的可用信息进行德语 **summary**，详见[此处](https://www.perplexity.ai/search/kannst-du-mir-eine-zusammenfas-aZ8WnXOORK.hH5cZSr0qqg)的讨论。
   - 该请求的具体细节强调了信息传播中对 **multilingual support** 的需求。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1328454395481817118)** (69 条消息🔥🔥): 

> `Claude 的性格与训练，开源模型的局限性，AI 数据中的防御性实践，训练数据的质量及其影响，模型中的类人交互` 


- **Claude 展示了独特的性格特征**：成员们注意到 **Claude** 拥有一种让某些人觉得讨厌但又很独特的性格，这在很大程度上提升了它的编程能力，直到最近被 **DeepSeek v3** 抢了风头。
   - *Anthropic 持续进行的训练后调整可能使 Claude 保持为 AI 模型中最好的编程工具之一。*
- **开源模型在面对专有数据时表现挣扎**：人们对像 Claude 这样的开源模型（注：原文如此）的未来持怀疑态度，许多人认为专有训练数据最终会产生更好的结果。
   - *对话强调，克服合成数据的挑战不仅仅需要增强算法，更强调了底层数据集的影响力。*
- **AI 训练中的防御性实践**：小组讨论了国防公司寻求 AI 服务日益增长的趋势，一些成员建议政府支持的数据中心可以解决数据访问不平等的问题。
   - *人们开始担心，由于公司利用优越的数据进行训练，可能会导致 AI 开发的垄断。*
- **评估训练数据质量及其影响**：对话强调了这样一个观点：虽然 Anthropic 可能没有最多的数据，但其整体质量可能足以与 Meta 等科技巨头以及其他中国公司竞争。
   - *参与者认识到，仅仅拥有大量数据并不等同于更好的模型性能。*
- **AI 中类人交互的魅力**：一些成员对 **Claude** 与用户的互动方式表示赞赏，注意到其类人交互背后的刻意设计，使其区别于仅专注于智能的模型。
   - *委员会认为 Claude 的设计理念可能通过创建一个更具共情能力的 AI 来增强用户体验。*



**提到的链接**：<a href="https://tenor.com/view/never-give-up-fist-pump-motivation-motivational-gif-4892451">Never Give Up Fist Pump GIF - Never Give Up Fist Pump Motivation - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1328821149030944900)** (3 条消息): 

> `用于数据提取的 Gemini，4o-mini 和 Llama-8B，用于文本转换的 Jina` 


- **Gemini 在数据提取方面表现出色**：一位成员强调 **Gemini** 非常适合**数据提取**，展示了它在这一角色中的有效性。
   - 这一说法表明人们对 Gemini 处理此类任务的能力越来越有信心。
- **评估 4o-mini 和 Llama-8B 的可靠性**：另一位成员建议，虽然 **4o-mini** 和 **Llama-8B** 可能适用于数据任务，但对于它们能否提供**准确的原始内容**，信任度较低。
   - 这反映了在选择数据提取模型时的一种谨慎态度。
- **探索使用 Jina 进行文本转换**：一位成员提议，如果数据以特定方式格式化，可以使用 **Jina** 将数据转换为文本，并建议编程方法可能会产生结果。
   - 这一想法指向了 **Jina** 的多功能性以及创新数据处理方法的潜力。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

real.azure: 看来这个月是 Attention 替代方案之月。 

https://arxiv.org/abs/2501.06425
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

real.azure: 看来这个月是 Attention 替代方案之月。 

https://arxiv.org/abs/2501.06425
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1328468365995544797)** (57 条消息🔥🔥): 

> `.aider.conf.yml 在 .gitignore 中的排除、DeepSeek v3 在本地机器上的性能、自定义主机的 API 配置、开源模型推荐、单个模型基准测试` 


- **关于 .gitignore 中 .aider.conf.yml 的辩论**：一名成员表示，默认情况下不应从 **.gitignore** 中排除 **.aider.conf.yml**，强调了它对团队决策的重要性。
   - 另一名成员反驳称，排除它可以简化个人设置并避免团队冲突。
- **运行 DeepSeek v3 需要大量资源**：讨论透露，为了有效运行 **DeepSeek v3**，用户可能需要约 **380GB RAM** 和多块 **GPU** 卡。
   - 用户探索了 **Qwen** 和其他较小模型等替代方案，以缓解硬件限制。
- **自定义端点的 API 配置**：分享了一种配置 Aider 以利用自定义 API 端点的明确方法，即利用 **OPENAI_API_BASE** 变量。
   - 成员们交换了成功测试 API 交互的命令。
- **开源模型推荐**：成员建议探索像 **Qwen 32b** 这样的小型模型进行本地运行，因为它们的资源需求较低。
   - 对话强调了在选择模型时性能与硬件能力之间的权衡。
- **Gemini 模型的性能查询**：一位参与者请求有关根据设定需求创建用户故事的最佳开源模型的信息。
   - 总体反馈指向 **Gemini** 模型，称赞其在特定任务中的有效性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI 兼容 API</a>：aider 是你终端里的 AI 配对编程工具</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-GGUF">unsloth/DeepSeek-V3-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1328741524296437760)** (6 条消息): 

> `Aider 错误处理、切换编辑格式、Copy Context 命令限制` 


- **Aider 在 LLM 编辑格式方面遇到困难**：一位用户报告在使用 aider 配合 **gpt-4o** 时，频繁出现错误：**'The LLM did not conform to the edit format'**，并寻求解决此类问题的建议。
   - 回复指出，当 LLM 误解系统提示词（system prompts）时会出现此问题，并建议减少添加到聊天中的文件数量以避免混淆。
- **在 Aider 中切换全量编辑格式（Whole Edit Format）**：一位用户询问是否可以在 aider 内部切换 **全量编辑格式**，以便在不通过命令行的情况下解决搜索/替换 bug。
   - 结果表明，用户确实可以通过使用命令 `/chat-mode whole` 切换到全量编辑模式。
- **Copy Context 命令的不一致性**：一位用户询问为什么 **/copy-context** 命令不包含聊天中的命令（如 Linux 或 Python 命令），而 **/ask** 和 **/code** 等其他命令却包含。
   - 这引发了关于 copy context 功能与其他命令相比的功能性和完整性的疑问。



**提到的链接**：<a href="https://aider.chat/docs/troubleshooting/edit-errors.html">文件编辑问题</a>：aider 是你终端里的 AI 配对编程工具

  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1328782458832814162)** (5 messages): 

> `Machete kernels across multiple GPUs, lm_head quantization issues, Bitpacking incompatibilities, Model monkey-patching` 


- **Machete Kernels 的进展仅限于单 GPU**：讨论强调了目前 vLLM 中 **Machete kernels** 的实现仅支持 **tp=1**，正如 [GitHub comment](https://github.com/vllm-project/llm-compressor/issues/973#issuecomment-2536261163) 中所指出的。
   - 此外，**NeuralMagic 的博客**暗示了改进的潜力，但强调跨多 GPU 的能力仍是一个探索领域。
- **lm_head 量化疑虑**：一位成员提到他们的 **lm_head** 似乎是**未量化**的，并对其在所使用的 **Llama 模型**中的配置提出了疑问 ([链接](https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4/tree/main))。
   - 他们报告在引用此 [代码行](https://github.com/vllm-project/vllm/blob/87054a57ab39bad6c7fe8999e7d93566ded713e3/vllm/model_executor/layers/quantization/kernels/mixed_precision/machete.py#L28-L31) 时，在 vLLM 代码的特定行遇到了错误。
- **Bitpacking 导致的不兼容问题**：一位成员建议，问题可能源于 **bitpacking** 与跨第一维度的 Tensor Parallel 不兼容，从而影响了性能。
   - 他们建议降级到 **Marlin** 作为遇到问题的潜在修复方案。
- **Monkey-Patching 建议**：有人提议对模型代码的一部分（特别是初始化文件）进行 **monkey-patch**，作为成功加载模型的变通方法。
   - 这部分代码可以在此 [GitHub 链接](https://github.com/vllm-project/vllm/blob/87054a57ab39bad6c7fe8999e7d93566ded713e3/vllm/model_executor/layers/quantization/kernels/mixed_precision/__init__.py#L14-L19) 中找到。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/hugg">Hugg (Yy)</a>: 未找到描述</li><li><a href="https://github.com/vllm-project/vllm/blob/87054a57ab39bad6c7fe8999e7d93566ded713e3/vllm/model_executor/layers/quantization/kernels/mixed_precision/__init__.py#L14-L19">vllm/vllm/model_executor/layers/quantization/kernels/mixed_precision/__init__.py at 87054a57ab39bad6c7fe8999e7d93566ded713e3 · vllm-project/vllm</a>: 一个用于 LLM 的高吞吐量且内存高效的推理与服务引擎 - vllm-project/vllm</li><li><a href="https://github.com/vllm-project/vllm/blob/87054a57ab39bad6c7fe8999e7d93566ded713e3/vllm/model_executor/layers/quantization/kernels/mixed_precision/machete.py#L28-L31).">vllm/vllm/model_executor/layers/quantization/kernels/mixed_precision/machete.py at 87054a57ab39bad6c7fe8999e7d93566ded713e3 · vllm-project/vllm</a>: 一个用于 LLM 的高吞吐量且内存高效的推理与服务引擎 - vllm-project/vllm</li><li><a href="https://github.com/vllm-project/llm-compressor/issues/973#issuecomment-2536261163).">ValueError: Failed to find a kernel that can implement the WNA16 linear layer. · Issue #973 · vllm-project/llm-compressor</a>: 描述 bug：无法使用 vllm 成功部署 GPTQ-W4A16 量化模型 (Qwen-72B)，其中仅模型的 FFN 部分被量化。环境包版本 -----------------...</li><li><a href="https://neuralmagic.com/blog/introducing-machete-a-mixed-input-gemm-kernel-optimized-for-nvidia-hopper-gpus/)">Introducing Machete: Optimized GEMM Kernel for NVIDIA Hopper GPUs</a>: Machete 是 Neural Magic 为 NVIDIA Hopper GPU 优化的 kernel，在 vLLM 中通过混合输入量化实现了 4 倍的内存节省和更快的 LLM 推理。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1328579814227181651)** (1 messages): 

> `Blockwise Matrix Multiplication, Linear Layer Kernel` 


- **关于分块 Matmul Kernel 的咨询**：一位成员询问用于线性层中**分块 matmul** 的共享 kernel 是否尚可，并提供了[线性层 kernel](https://cdn.discordapp.com/attachments/1189607595451895918/1328579813891641374/linear_layer_kernel.txt?ex=6787e09f&is=67868f1f&hm=cf29d6ada51bf2b4daac4239743fec6768b9a487bd5b0faf24f785b48eddbafa&) 的链接。
   - 该请求引发了关于该 kernel 在优化线性层性能方面有效性的讨论。
- **预计对 Kernel 性能的讨论**：该 kernel 文件的分享预示着对其**性能**反馈及潜在改进领域的期待。
   - 成员们可能会将此 kernel 与现有的分块矩阵乘法实现进行比较。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1328546607746256951)** (3 messages): 

> `FA3 Profiling, H200 vs H100 Performance` 


- **H200 和 H100 上的 FA3 性能差异引发讨论**：一位用户提出了在 **H200** 上对 **FA3** 进行 Profiling 时，相比 **H100** 是否有**显著的性能提升**的问题。
   - 另一位用户确认，**FA3** 与之前的 **FA2** 模型之间确实存在**显著差异**，但讨论重点仍集中在与 H200 的具体比较上。
- **FA2 的显著差异得到认可**：一位用户指出 **FA3** 与 **FA2** 之间已确定的**显著差异**，证实了后者在性能上的改进。
   - 这一交流表明社区对最新模型之间性能指标的关注。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1328601182918152294)** (2 messages): 

> `torch.cuda.max_memory_allocated, torch.cpu, Memory allocation functions` 


- **内存分配缺乏 CPU 等效项**：一位成员注意到 `torch.cuda.max_memory_allocated()` 缺乏对应的 CPU 版本，指出这在功能上存在缺失。
   - *这引发了关于* **PyTorch** 中 CPU 使用率内存分配追踪的问题。
- **对 torch.cpu 功能的担忧**：另一位成员对 `torch.cpu` 表示不满，认为其功能*相当匮乏*。
   - 这种情绪可能凸显了对 **PyTorch** 库中 **CPU 功能** 增强的需求。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1328846464159453194)** (2 messages): 

> `MiniMax-01 Open Source, Lightning Attention Architecture, Ultra-Long Context, Cost-Effectiveness of MiniMax Models` 


- **MiniMax-01 开源发布**：MiniMax-01 系列突破性开源模型已发布，包括 **MiniMax-Text-01** 和 **MiniMax-VL-01**。
   - 该公告通过 [MiniMax's Twitter](https://x.com/minimax__ai/status/1879226391352549451) 发布，强调了模型的创新能力。
- **革命性的 Lightning Attention 架构**：MiniMax-01 采用的 **Lightning Attention** 机制代表了对传统 Transformer 架构的重大突破，展示了大模型规模的实现。
   - 该架构旨在将模型性能提升至远超现有能力的水平。
- **高效处理高达 4M Tokens**：MiniMax-01 支持处理高达 **4M Tokens**，大幅超越领先模型 **20 到 32 倍**。
   - 这一特性旨在满足 AI Agent 应用对超长上下文处理日益增长的需求。
- **极具竞争力的定价策略**：MiniMax-01 模型的推出极具性价比，价格为 **每百万输入 Token 0.2 美元**，**每百万输出 Token 1.1 美元**。
   - 定价反映了模型架构和基础设施方面的优化，促进了持续创新。
- **立即体验免费试用**：感兴趣的用户现在可以通过 [Hailuo AI](https://hailuo.ai) 免费试用 MiniMax-01 模型。
   - 更多详情和见解可以在关于 [MiniMax-01 的完整论文](https://filecdn.minimax.chat/_Arxiv_MiniMax_01_Report.pdf)中找到。



**提到的链接**：<a href="https://x.com/minimax__ai/status/1879226391352549451">来自 MiniMax (官方) (@MiniMax__AI) 的推文</a>：MiniMax-01 现已开源：为 AI Agent 时代扩展 Lightning Attention。我们很高兴推出最新的开源模型：基础语言模型 MiniMax-Text-01 和视觉模型...

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

remek1972: 新内容：https://salykova.github.io/sgemm-gpu
  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1328736812541411421)** (2 messages): 

> `Kaiko AI 职位空缺，Prior Labs 的 Foundation Models` 


- **Kaiko AI 招聘 Senior ML 和 Data Engineers**：Kaiko AI 正在阿姆斯特丹和苏黎世招聘 **Senior ML Platform Engineer** 和 **Senior Data Engineer** 职位，重点是开发 Foundation Models 以增强癌症治疗（[ML Engineer 职位发布](https://jobs.kaiko.ai/jobs/4829222-senior-ml-platform-engineer)，[Data Engineer 职位发布](https://jobs.kaiko.ai/jobs/5007361-senior-data-engineer)）。值得注意的是，他们不提供签证赞助。
- **Prior Labs 的激动人心机会**：Prior Labs 是一家资金充足的初创公司，正在为 **tabular data**（表格数据）、**time series**（时间序列）和 **databases**（数据库）构建 Foundation Models，并正在积极招募熟练的 ML 工程师（[研究文章](https://www.nature.com/articles/s41586-024-08328-6)）。这项工作预计将对从 **healthcare**（医疗保健）到 **finance**（金融）的各个领域产生广泛影响，同时允许工程师进行创新并加速模型性能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://jobs.kaiko.ai/jobs/4829222-senior-ml-platform-engineer">Senior ML Platform Engineer - Kaiko</a>: 关于 Kaiko：在癌症护理中，治疗决策可能需要很多天——但患者没有那么多时间。延迟的原因之一？癌症患者的数据分散在许多地方：医生...</li><li><a href="https://jobs.kaiko.ai/jobs/5007361-senior-data-engineer">Senior Data Engineer - Kaiko</a>: 关于 Kaiko：在癌症护理中，治疗决策可能需要很多天——但患者没有那么多时间。延迟的原因之一？癌症患者的数据分散在许多地方：医生...</li><li><a href="https://www.notion.so/priorlabs/ML-Engineer-Foundation-Models-Freiburg-Berlin-London-1425be1f3b4980598ef0faa9e47ec0e1">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>: 一款将日常工作应用融合为一的新工具。它是为您和您的团队打造的一体化工作空间。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1328508190417096726)** (8 messages🔥): 

> `CUDA 原子函数，CFD 求解器开发` 


- **关于 Double 类型 CUDA 原子函数的讨论**：成员们讨论了缺乏专门用于寻找 **double** 类型元素最小值的 CUDA 原子函数，目前该功能仅有针对整数的实现。
   - 一位成员建议对正数 double 使用整数原子函数，而另一位成员分享了一个 [Stack Overflow 回答](https://stackoverflow.com/a/72461459)，该回答解决了处理负零的问题。
- **在 GPU 上构建 CFD 求解器**：一位成员分享了他们利用 GPU 构建 **CFD 求解器** 以提高并行化效率的项目。
   - 这突显了在流体动力学中应用先进计算技术日益增长的兴趣。



**提到的链接**: <a href="https://stackoverflow.com/a/72461459">如何在 CUDA 中对浮点值使用 atomicMax？</a>: 我在 CUDA kernel 中使用了 atomicMax() 来寻找最大值：&#xA;&#xA;__global__ void global_max(float* values, float* gl_max)&#xA;{&#xA;    int i=threadIdx.x &#x2B; blockDim.x * blockIdx.x;&...

  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1328662257298702348)** (1 messages): 

> `DeepSeek 2.5 推理，图像分析` 


- **DeepSeek 2.5 展示了完美的推理能力**：一次讨论强调了 **DeepSeek 2.5** 在处理共享任务时表现出的 **完美推理能力**。
   - 成员们似乎对该模型处理材料的方式印象深刻，这表明其在推理方面具有强大的能力。
- **链接用于分析的图像**：分享了一张与 DeepSeek 2.5 性能讨论相关的图像进行分析。
   - 它似乎包含支持有关该模型推理能力主张的相关数据。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1328712225027915827)** (6 messages): 

> `int8_weight_only, torch.compile and optimization, torchao compatibility with TorchScript/ONNX` 


- **int8_weight_only 利用融合操作**：有人提问 **int8_weight_only** 是否使用自定义 Kernel 进行反量化（dequantization）和矩阵乘法。
   - 确认了 **torch.compile** 会有效地将这些操作结合起来。
- **使用 torch.compile 融合操作**：进一步阐明 **torch.compile** 确实会按顺序融合反量化和矩阵乘法操作。
   - 这种优化确保了模型执行期间性能的提升。
- **torchao 在 TorchScript 和 ONNX 中的实用性**：有人询问了 **torchao** 与 **TorchScript/ONNX** 的兼容性，并确认了其功能。
   - 为了导出图（graph），成员们讨论了如何配合 **torch.compile** 使用 [torch.export 功能](https://github.com/pytorch/ao/tree/main/torchao/quantization#workaround-with-unwrap_tensor_subclass-for-export-aoti-and-torchcompile)。



**提及的链接**：<a href="https://github.com/pytorch/ao/tree/main/torchao/quantization#workaround-with-unwrap_tensor_subclass-for-export-aoti-and-torchcompile">ao/torchao/quantization at main · pytorch/ao</a>：PyTorch 原生量化和稀疏化，用于训练和推理 - pytorch/ao

  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1328640191115821106)** (3 messages): 

> `GTC Attendance, CUDA Talks, Networking at GTC` 


- **GTC 参会引起关注**：一位成员询问：*“有人去 GTC 吗？”*，表达了想在活动中与其他成员建立联系的愿望。
   - 另一位成员确认参会，表示：*“我会去。”*
- **碰面计划**：发起者提到了他们通常的习惯，即在 GTC 的 **CUDA 相关演讲**附近逗留，并尝试随机结识参会者。
   - 他们坦言，有时这种社交尝试并不会产生结果。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1328847355726004264)** (1 messages): 

> `MPS Profiler, Kernel Profiling, GPU Trace Issues, PyTorch and MPS, MTL_CAPTURE_ENABLED` 


- **MPS Profiler 难以捕获 GPU Trace**：一位用户报告了 **MPS Profiler** 在 Kernel Profiling 期间无法捕获 GPU Trace 的问题，尽管在调度到 GPU 的前后使用了 `getMPSProfiler().startCapture()` 和 `getMPSProfiler().stopCapture()`。
   - 他们确保设置了 **MTL_CAPTURE_ENABLED=1** 环境变量，但从 **Xcode** 获取的 Trace 仍然为空。
- **PyTorch 与 MPS Profiler 的交互**：提到该问题特别发生在从 **Python** 使用 **PyTorch** 运行操作时，引发了关于可能存在集成问题的疑问。
   - 用户正在寻求解决方案，或了解在此场景下将 **MPS** 与 **PyTorch** 配合使用时是否存在常见的误区。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1328839217790586880)** (1 messages): 

> `Thunder Compute, Cloud GPU Pricing, CLI Instance Management` 


- **Thunder Compute 发布，提供更便宜的云端 GPU**：联合创始人宣布推出 [Thunder Compute](https://thundercompute.com)，致力于让**云端 GPU** 更便宜且更易于使用，特别是 **A100** 的定价为 **$0.92/小时**。
   - 团队正在进行 **Beta 测试**，并为用户提供 **$20/月免费额度**以收集服务反馈。
- **简单易用的 CLI 实现无忧实例管理**：用户可以使用 CLI (`pip install tnr`) 轻松管理云端实例，只需一条命令即可实现自动化实例创建。
   - 该功能简化了入门流程，旨在让复杂的配置对每个人都变得可控。
- **由顶级云基础设施支持**：实例托管在 **GCP** 和 **AWS** 的**美国中部地区**，以确保最佳的运行时间和性能。
   - 这种架构增强了机器学习工作流所必需的可靠性和可扩展性。



**提及的链接**：<a href="https://thundercompute.com">Thunder Compute: Low-cost GPUS for anything AI/ML</a>：在 Thunder Compute 上训练、微调和部署模型。开始使用每月 $20 的免费额度。

  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1328833651156455445)** (1 messages): 

> `Onboarding Documentation` 


- **基础入门文档已添加**：TK 的入门文档已添加，你可以点击[此处](https://docs.google.com/document/d/15-Zvf6e0NLX1si4ml4sUOWCDlXNMtOWKiuo6CKZMEYA/edit?usp=sharing)查看。
   - 评论模式已开启，欢迎针对任何不清晰的章节或需要解决的其他问题提供反馈。
- **鼓励对文档提供反馈**：鼓励成员对入门文档提供反馈，以澄清任何不明确的点。这种主动的方法旨在提升未来用户的入门体验。



**提到的链接**：<a href="https://docs.google.com/document/d/15-Zvf6e0NLX1si4ml4sUOWCDlXNMtOWKiuo6CKZMEYA/edit?usp=sharing">TK onboarding</a>：摘要。本文档详细说明了如何开始使用 TK 编写 Kernel。请随时在文档中针对改进区域/缺失信息留下评论。摘要 1 Back...

  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1328522006064271455)** (4 messages): 

> `Nvidia Cosmos, Jetson, Transformer Engine, Mamba, 3D Vision Stack` 


- **Nvidia Cosmos 在 Jetson 上无缝运行**：一位成员分享了一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/johnnycano_nvidia-cosmos-nvidiacosmos-activity-7283774665943109632-VQDa?utm_source=share&utm_medium=member_ios)，强调 **Nvidia Cosmos** 已成功在 **Jetson** 设备上运行。
   - 这一进展为各种 AI 驱动的项目开启了新的应用可能。
- **成功移植 Transformer Engine 及相关库**：一位成员报告称，他们已经移植了 **Transformer Engine** 以及 **30 多个库**，以增强兼容性和性能。
   - 这包括 **Mamba** 和 **3D vision stack** 等显著的补充。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1328501686200700950)** (20 messages🔥): 

> `New codestral model, ChatGPT 4o with Canvas, OpenAI model changes, AI Misalignment Videos, App token issues` 


- **推出 codestral 模型**：名为 **codestral** 的新模型已在 **Mistral API** 上免费提供，具有 **256k context**，并以速度和效率著称。
   - 讨论其能力的成员将其描述为“极快且好用”。
- **了解 ChatGPT 4o with Canvas**：成员们对于 **ChatGPT 4o with Canvas** 究竟是集成了 Canvas 的旧模型，还是一个全新的变体感到困惑。
   - 一些人认为它使用了 9 月份的旧模型，旨在辅助用户与 Canvas 功能进行交互。
- **OpenAI 模型之谜**：社区成员对 OpenAI 模型更新的实施表示困惑，特别是涉及 Canvas 功能的部分。
   - 一位用户注意到，之前与 **4o canvas** 模型的交互现在已回退到 **4o mini**。
- **AI Misalignment 讨论**：一位成员分享了一个关于 **AI misalignment**（AI 对齐失效）动画的 [YouTube 视频](https://www.youtube.com/watch?v=K8p8_VlFHUk)，引发了对该话题的兴趣。
   - 这引发了关于视频内容及其与当前 AI 讨论相关性的提问。
- **Token 相关的应用问题**：在经历应用功能问题后，一位用户提到应用关闭了，但在重新打开后恢复正常，这表明可能存在 **token exhaustion**（Token 耗尽）。
   - 另一位用户建议这种行为可能预示着崩溃，而非典型的 Token 限制。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/)** (1 messages): 

jlgri: Apple Watch 上会有 ChatGPT 应用吗？
  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1328465845529214979)** (8 messages🔥): 

> `数据格式讨论、PDF 局限性、Assistant 响应问题、改写技巧` 


- **讨论数据格式选项**：成员们正在探索更好的数据格式，建议包括 **JSON**、**YAML** 和纯文本，同时批评了 **PDF** 的局限性。
   - *一位成员自嘲道*：“PDF 不是 API 😂。”
- **重新思考 Assistant 的文档引用**：一位成员正在寻求解决方案，以防止他们的 Assistant 在每个结果末尾不断引用文档，他们被建议实现特定的代码。
   - 他们向社区寻求帮助，询问在 [OpenAI playground](https://platform.openai.com/) 的何处实现该代码。
- **为非母语人士简化语言的技巧**：另一位成员分享了他们自己的 **de-GPTing** prompt，专注于改写文本，通过避免生僻词和复杂结构，使其更易于第二语言使用者阅读。
   - 他们强调不要改写特定领域的术语，以保持上下文。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1328465845529214979)** (8 messages🔥): 

> `改进数据格式、减少 PDF 使用、Assistant 行为自定义、用户友好的改写` 


- **寻找更好的数据格式**：成员们讨论了如何提高数据的可用性，建议请求比 PDF 更好的格式，如 **JSON、YAML** 甚至简单的**纯文本**。
   - 一位成员指出 *PDF 不是 API*，强调需要更具适应性的数据格式。
- **社区关于 Assistant 引用的见解**：一位成员寻求帮助，希望让他们的 Assistant 停止在结果末尾引用文档，并分享了一张包含建议代码更改的图片。
   - 另一位成员热情回应，对愿意提供帮助的人预先表示了感谢。
- **为非母语人士简化语言**：一位用户分享了他们的 **de-GPTing prompt**，旨在通过避免复杂的词汇来改写 prompt，使其对第二语言使用者更友好。
   - 其目的是在简化语言的同时，保持技术术语的完整性。


  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1328707329163657226)** (9 messages🔥): 

> `Konkani 语言保护、针对 Konkani 语言的 AI 模型` 


- **Konkani 语言保护倡议**：一位成员介绍了来自果阿的 CSE 本科生 **Reuben Fernandes**，他正在致力于一个保护 **250 万**人使用的 **Konkani** 语言的项目。
   - *他正在寻求与行业专业人士的合作*，以增强项目的影响力和获得批准的机会。
- **理解 Konkani 的 AI 模型**：**Reuben 的项目**旨在开发一个能够使用 **Konkani** 进行对话和理解该语言的 AI 模型，以提升该语言的文化意义。
   - *他指出了项目的独特性*，因为目前还没有现有的模型能够深入理解 Konkani，这使得该项目对语言保护至关重要。


  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1328693517777764462)** (16 messages🔥): 

> `Rerank 微调定价, AI 记忆限制, API key 限制, Cohere 定价与文档` 


- **Rerank 微调定价困惑**：一位成员询问了微调 Rerank 模型的定价，提到该定价未列在 [Cohere 定价页面](https://cohere.com/pricing)上。另一位成员提供了关于 [Rerank FT](https://docs.cohere.com/docs/rerank-fine-tuning) 的文档链接和常见问题解答以供进一步协助。
- **AI 记忆范围说明**：讨论透露 AI 的上下文长度（context length）为 **128k tokens**，在遗忘前大约可转换成 **42,000 个单词**。澄清说明这一记忆容量适用于整个交互时间线，而不仅仅是单个聊天。
- **长期与短期 AI 记忆**：成员们讨论了 AI 系统中长期记忆和短期记忆的区别。明确了上下文长度和保留能力适用于总计多次的交互。
- **理解 API Key 限制**：确认了 API key 的使用限制是基于请求数（requests）而非 token 长度。关于测试版（trial）和生产版（production）API key 的速率限制（rate limits）详情可以在 [Cohere 文档](https://docs.cohere.com/v2/docs/rate-limits)中找到。
- **Cohere API Keys 概览**：Cohere 提供使用受限的评估 key 和约束较少的生产 key。成员们被引导至 [API keys 页面](https://dashboard.cohere.com/api-keys)创建 key，并参考 [定价文档](https://docs.cohere.com/v2/docs/how-does-cohere-pricing-work)获取更多信息。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.cohere.com/v2/docs/rate-limits">不同类型的 API Key 和速率限制 — Cohere</a>：此页面描述了 Cohere API 针对生产和评估 key 的速率限制。</li><li><a href="https://youtu.be/B45s_qWYUt8">Cohere 今年将如何提升 AI 推理能力</a>：Cohere CEO Aidan Gomez 揭示了他们如何应对 AI 幻觉并提升推理能力。他还解释了为什么 Cohere 不使用任何外部...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1328611499744170015)** (11 messages🔥): 

> `Bot 交互, 爱丽丝梦游仙境引用, Cohere 文档搜索` 


- **关于语言的 Bot 交互**：一位用户向 Cmd R Bot 提问 **corvo** 和 **escrivaninha** 之间是否有任何相似之处，Bot 最初回答没有。
   - *Commonwealth* 通过引用 **《爱丽丝梦游仙境》** 挑战了 Bot 的回答，暗示了更深层的联系。
- **《爱丽丝梦游仙境》俏皮话**：对话引用了 **《爱丽丝梦游仙境》**，暗示尽管 Bot 给出断言，但这些词之间可能存在某种幽默或古怪的联系。
   - 这表明了文化引用如何引发关于语言的更深层讨论。
- **Cohere 文档搜索无效**：Cmd R Bot 试图通过在 **Cohere 文档** 中搜索术语 *corvo* 和 *escrivaninha* 的信息来增强对话，但最终一无所获。
   - 这展示了 Bot 在处理文化或文学查询时，其数据访问权限的局限性。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1328454892544589884)** (2 messages): 

> `会议出席, 更新分享, YouTube 链接` 


- **在课程期间分享更新**：一位成员对更新表示感谢，并提到由于课程冲突错过了一部分会议。
   - “感谢更新”反映了社区在面临日程挑战时的支持性氛围。
- **提供 YouTube 视频资源**：另一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=PYtNOtCD1Jo) 链接，可能与会议讨论相关。
   - 该视频可以作为那些错过会议的人的宝贵资源。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1328454911213441145)** (26 条消息🔥): 

> `Async Mojo 提案，Mojo 的 Zed 扩展，Mojodojo Int8 转换问题` 


- **Owen 的 Async Mojo 提案引发热烈讨论**：在一个关于预测在 Mojo 中实现结构化异步（structured async）挑战的讨论串中，一名成员强调了在不同库之间建立效应处理（effect handling）标准的必要性。
   - 讨论中提出了关于一致处理异常的担忧，同时确保像 `oom` 和 `divbyzero` 这样的效应由标准库定义。
- **关于 Mojo 中多个执行器（executors）的深入讨论**：成员们讨论了多个并存执行器运行的影响，并指出不同实现之间需要保持正交性。
   - 一位参与者建议，为了采取更务实的方法，基础 API 的开发应该先于这些讨论。
- **Mojo 的 Zed 扩展开发取得进展**：一位参与者分享了他们为 Zed 开发的 Mojo 扩展，并指出之前的扩展无法识别 `stdlib` 路径的挑战，正在寻求集成方面的指导。
   - 另一位成员分享了他们自己的扩展，该扩展提供了改进的自动补全和 LSP 功能，并建议进行潜在增强以供更广泛使用。
- **寻求 Mojodojo 的 Int8 到 String 转换解决方案**：一位用户提出了关于 Mojodojo 中 Int8 到 String 转换功能的问题，并请求社区协助。
   - 回复中强调了理解 Mojo 中参数类型的重要性，并提供了相关文档的链接。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/collections/string/String/#write_bytes">String | Modular</a>: struct String</li><li><a href="https://docs.modular.com/mojo/manual/types/">Types | Modular</a>: 标准 Mojo 数据类型。</li><li><a href="https://github.com/modularml/mojo/issues/3947">[mojo-examples] Mojodojo Int8 to string conversion example not working · Issue #3947 · modularml/mojo</a>: 问题出在哪里？https://mojodojo.dev/guides/intro-to-mojo/basic-types.html#strings 我们能做些什么改进？这个转换 var word = List[Int8]() word.append(78) word.append(79) word.append(0...</li><li><a href="https://github.com/freespirit/mz">GitHub - freespirit/mz: Support for Mojo in Zed</a>: Zed 对 Mojo 的支持。通过创建一个账户为 freespirit/mz 的开发做出贡献。</li><li><a href="https://docs.modular.com/mojo/manual/parameters/">Parameterization: compile-time metaprogramming | Modular</a>: 参数和编译时元编程简介。</li><li><a href="https://github.com/modularml/mojo/pull/3945">[proposal] Structured Async for Mojo by owenhilyard · Pull Request #3945 · modularml/mojo</a>: 提议为 Mojo 添加结构化异步，遵循 Rust 的异步传统，因为 Mojo 有能力解决 Rust 异步中的许多问题，其中一些是生态系统影响...</li><li><a href="https://github.com/modularml/mojo/pull/3946">[proposal] Provided Effect Handlers by owenhilyard · Pull Request #3946 · modularml/mojo</a>: 该提案包含了一个效应系统的替代方案，我认为它更适合在上下文可能不明确的系统语言中抽象异步、raises 和类似的函数颜色...
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1328523014958219345)** (4 messages): 

> `Tinygrad 概览、对 Tinygrad 的需求、Tiny Corp 的融资与愿景、AI 芯片公司` 


- **探索对 Tinygrad 的需求**：一位用户询问了关于理解 **tinygrad** 目的的相关资源，提到他们查看了网站和文档，但缺乏更深入的见解。
   - 他们指出 *理解 LLVM 具有挑战性*，这引发了关于 tinygrad 重要性的讨论。
- **Tiny Corp 的新创项目获得融资**：一名成员分享了一篇 [博客文章](https://geohot.github.io/blog/jekyll/update/2023/05/24/the-tiny-corp-raised-5M.html)，透露 *tiny corp* 已获得 **$5M** 的融资，标志着一个正在进行的商业项目。
   - 该公司的目标是最终专注于 **芯片开发**，其更广泛的目标是让先进计算变得触手可及。
- **理解人类大脑的算力**：文章强调人类大脑的运行速度约为 **20 PFLOPS**，这在很大程度上仍然是难以企及的，同等算力的成本约为 **$1M** 或 **$100/hr**。
   - 创始人通过多篇 *博客文章* 讨论了这一概念，反映了大多数人面临的算力限制。



**Link mentioned**: <a href="https://geohot.github.io/blog/jekyll/update/2023/05/24/the-tiny-corp-raised-5M.html">the tiny corp raised $5.1M</a>：我们又开始了。我创办了另一家公司。钱已经到账了。

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1328698975758581891)** (16 messages🔥): 

> `学习 Tinygrad、Tensor 堆叠问题、Tinygrad 中的递归错误、获取 Tensor 属性、使用 Tensor 函数` 


- **通过直播学习 Tinygrad**：一位成员分享了他们通过观看 George 的直播和分析聊天记录来学习 **Tinygrad** 的历程，这帮助他们更好地掌握了相关概念。
   - 他们强调从 **distributed systems** 的角度来理解硬件故障的影响。
- **Tensor 堆叠问题**：一位用户在对 Tensor 使用 **.stack** 函数时遇到挑战，并在尝试对堆叠后的 Tensor 调用 .numpy() 时遇到了 **RecursionError**。
   - 据观察，将堆叠的 Tensor 数量从 **6131 减少到 1000** 解决了该问题。
- **处理 Tinygrad 中的递归限制**：成员们讨论了由于在一次操作中堆叠了过多的 Tensor 而导致触发 **recursion depth exceeded** 错误。
   - 提出了一种解决方案，即限制操作次数并一次堆叠较少的 Tensor，以避免超过内部限制。
- **检索 Tensor 属性**：一位用户尝试使用 **dir()** 函数访问 Tensor 对象的不同属性，但遇到了 **IndexError**。
   - 发生该错误是因为默认的 `transpose()` 调用尝试对一个 1D Tensor 不存在的维度进行操作。
- **识别 Tensor 函数问题**：会议澄清了获取 Tensor 属性的问题源于在没有正确维度参数的情况下调用了 **transpose()** 等函数。
   - 成员们建议，谨慎处理 Tensor 维度对于避免此类错误至关重要。



**Link mentioned**: <a href="https://blog.codinghorror.com/because-reading-is-fundamental-2/">Because Reading is Fundamental</a>：大多数讨论在每个用户旁边都会显示一些信息：这传达了什么信息？ * 你名字旁边显示的唯一可以控制的数字是发帖数。 * 每个读到这篇文章的人都会看到 ...

  

---

### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1328651997229875231)** (13 条消息🔥): 

> `Inference-time scaling in LLMs, Jailbreaking O1 for model training, Automated evaluation methods in healthcare, False equivalence in medical training, Role of AI in medicine` 


- **推理时间扩展（Inference-time scaling）提升医学推理能力**：讨论的论文指出，对于大型语言模型（LLMs），增加**推理时间**可以显著提高性能。在仅有 **500 个样本**训练集的情况下，模型在医学基准测试上的表现提升了 **6%-11%**。
   - 论文强调，**任务复杂性**需要更长的推理链，并讨论了排除内部数据的原因，因为无关信息会干扰逻辑推导。
- **对越狱 O1（Jailbreaking O1）的担忧**：一位成员对通过**越狱 O1**来训练模型的做法表示怀疑，认为现有的基准测试大多与有效评估无关。
   - 这引发了关于模型训练中所采用方法论稳健性的质疑。
- **医疗保健中的自动化评估引发关注**：目前医疗保健领域的自动化评估很大程度上依赖于**多项选择题**，这使得评估局限于模式识别和信息召回。
   - 呼吁开发能够更准确反映医学训练复杂性和临床实践现实的方法。
- **医学训练与 AI 之间的虚假等价**：关于**多项选择考试**的高分与医生及医学生在临床环境中的实际能力之间是否存在虚假等价，目前存在持续争论。
   - 参与者指出，医学生会接受广泛的临床实践并受到多位医生的严格审查，这与目前尽管经过 Prompt Engineering 但仍表现不佳的 AI 模型不同。
- **AI 在未来医学中的角色**：对话还强调，医疗保健领域的 AI 不应只关注取代医生，而应挑战现有的规范和方法论。
   - 参与者赞赏那些反映了对 AI 在医学领域影响更细致入微观点的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.06458">O1 Replication Journey -- Part 3: Inference-time Scaling for Medical Reasoning</a>：基于我们之前对 O1 复制的研究（第一部分：Journey Learning [Qin et al., 2024] 和第二部分：Distillation [Huang et al., 2024]），本工作探索了推理时间扩展在医学推理中的潜力...</li><li><a href="https://www.nature.com/articles/s41467-024-55628-6">Large Language Models lack essential metacognition for reliable medical reasoning - Nature Communications</a>：大型语言模型在医学考试中表现出专家级的准确性，支持了将其纳入医疗环境的潜力。在这里，作者揭示了它们的元认知能力尚不足...
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1328470667925913681)** (10 条消息🔥): 

> `Open Interpreter Setup, Video Editing Capabilities, Cmder Performance Issues` 


- **Open Interpreter 安装成功**：在最初遇到通过 **pipx** 安装 **open-interpreter** 的问题后，一位成员确认在安装 **brew** 和 **pipx** 后成功完成设置。
   - 他们征求了关于 **open interpreter** 最有用功能的建议，特别是它是否可以处理视频编辑。
- **Open Interpreter 可以执行视频命令**：针对有关视频编辑的问题，对方澄清 **open interpreter** 可以运行任意命令，包括与视频编辑相关的命令。
   - 这意味着用户可以通过提供适当的命令，使用它来自动化视频编辑流程。
- **Cmder 中的性能故障**：一位成员提出了在 **Cmder** 中使用时的性能问题，当 **open interpreter** 输出大量文本时，会导致屏幕闪烁且文本移动异常。
   - 他们注意到在输出较短时性能尚可，表明该问题可能是由较长的文本输出触发的。
- **输出显示问题**：同一位成员观察到，当 **open interpreter** 写入大量文本时，它似乎会频繁清除并重写终端输出，从而影响速度。
   - 他们分享说，使用 `--plain` 选项并未缓解该问题，导致 **Cmder** 中出现了奇特的显示格式。


  

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1328599946122756106)** (2 messages): 

> `Deepseek Model Name, API Key Variable for Deepseek` 


- **关于 Deepseek 模型名称的查询**：一名成员提出了关于 **Deepseek** **模型名称**的问题。
   - 该查询强调了需要明确讨论中使用的具体模型。
- **关于 DEEPSEEK_API_KEY 变量的问题**：一名成员请求有关如何设置 **DEEPSEEK_API_KEY** 变量的信息。
   - *“值应该是什么？”* 表示对使用 API 的正确配置存在疑问。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

velinxs: 如何将 Deepseek 集成到 OI 中？
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1328467198670409773)** (7 messages): 

> `2024 MOOC nostalgia, Beginner friendliness of the MOOC, Fall 2024 MOOC lectures, Certificate release for Fall 2024 MOOC` 


- **2024 MOOC 引发怀旧情结**：一位用户表达了对 **2024 课程**的怀旧之情，表明了深厚的情感联系。
   - 这突显了参与式学习体验的持久影响。
- **MOOC 初学者友好性讨论**：一位在修读完本科 **Machine Learning** 课程的新用户询问了该 MOOC 的初学者友好程度。
   - 回复建议查看 [2024 秋季 MOOC 讲座](https://llmagents-learning.org/f24) 以评估难度级别。
- **了解 2024 秋季 MOOC 内容**：一位成员建议潜在学习者回顾 **2024 秋季讲座**，因为它们提供了将在 **2025 春季**课程中进一步构建的基础知识。
   - 他们保证春季课程不需要先验知识，只需要参与材料的意愿。
- **证书发放公告**：一位成员询问了 2024 秋季 MOOC 的**证书**情况，寻求关于课程结束后证书可用性的说明。
   - 另一位成员确认证书将于**本月晚些时候发放**，缓解了对遗漏的担忧。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1328493597456072805)** (5 messages): 

> `NPU Support in GPT4All, VPN and Reverse Proxy for GPT4All Access, Model Versions on Hugging Face` 


- **GPT4All 的 NPU 支持即将到来？**：一位用户询问 **GPT4All** 是否很快能够利用 **NPU**，特别是针对 **AMD 处理器**。
   - *Victor Gallagher* 指出 AMD 仍需完善**软件栈**，但表示如果能实现将非常有益。
- **通过 VPN 使 GPT4All 可访问**：一位成员建议在推理机上运行 **VPN** 或**反向代理**，以便从其他设备访问 GPT4All 的本地界面。
   - 该方法被提议作为确保跨机器连接的实用解决方案。
- **Hugging Face 模型版本说明**：一位用户注意到了像 **codellama q4_0** 这样的模型，但提到具有不同**量化（quantization）**的变体在 **Hugging Face** 上也有提供。
   - 他们得出结论，将模型添加到文件夹中即可解决有关使用的疑问。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1328786126525300768)** (1 messages): 

> `Agent vs. Naive RAG, Weaviate Integration, LlamaIndex Performance` 


- **在 Tuana 的 Notebook 中，Agent 表现优于 Naive RAG**：在 [Tuana 最近的 Notebook](https://twitter.com/tuanacelik) 中，将使用 **Weaviate** 和 **LlamaIndex** 的 Agent 性能与 Naive RAG 进行了对比，展示了更优异的结果。
   - Agent 在数据源及其组合方面做出**决策（decision-making）**选择的能力，在增强整体有效性方面发挥了关键作用。
- **使用 LlamaIndex 探索 Weaviate**：Tuana 强调了将 **Weaviate** 集成到 **LlamaIndex** 工作流中，展示了这种组合如何增强数据管理。
   - 讨论强调了此类集成带来的多功能性和潜在改进。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1328456163301261362)** (3 条消息): 

> `QuestionsAnsweredExtractor 定制, Prompt 模板函数映射, LlamaIndex 包安装` 


- **使用自定义 Prompt 增强 QuestionsAnsweredExtractor**：一位成员询问如何在 `self.llm.apredict()` 中为 `QuestionsAnsweredExtractor` 添加自定义 Prompt 模板和额外的上下文变量。
   - 成员们对于是否可以动态添加更多变量以增强功能表示好奇。
- **使用函数映射处理动态变量**：另一位成员建议在 Prompt 模板中使用函数映射（function mappings），以便在连接到函数时附加所需的任何变量。
   - 他们引用了 [LlamaIndex 高级 Prompt 文档](https://docs.llamaindex.ai/en/stable/examples/prompts/advanced_prompts/#3-prompt-function-mappings) 以获取进一步指导。
- **解决 LlamaIndex 包安装问题**：在讨论过程中，分享了有关包安装的细节，包括下载 `llama-index-llms-openai` 及其依赖项。
   - 安装过程涉及多个包版本，并显示从镜像 URL 成功获取。



**提及的链接**：<a href="https://docs.llamaindex.ai/en/stable/examples/prompts/advanced_prompts/#3-prompt-function-mappings">Advanced Prompt Techniques (Variable Mappings, Functions) - LlamaIndex</a>：未找到描述

  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1328756606778867834)** (2 条消息): 

> `Meta JASCO, 联合音频与符号调节, 音乐建模` 


- **Meta JASCO 模型发布**：**Meta AI 的 FAIR 团队**宣布发布 **JASCO**，这是一个在 **2024 年 11 月**训练的新音乐模型。
   - JASCO 采用 **EnCodec 模型**进行音频 Tokenization，并提供两个变体，包含对**和弦 (chords)、鼓点 (drums)** 和**旋律 (melody)** 的控制。
- **JASCO 技术详解**：JASCO 使用基于 **Transformer** 架构的 **flow-matching 模型**，并支持带有 **condition dropout** 的推理。
   - 它提供两种尺寸：**400M** 和 **1B**，以满足不同的音乐生成需求。
- **研究论文可用性**：对于对技术细节感兴趣的人，题为 [Joint Audio And Symbolic Conditioning for Temporally Controlled Text-To-Music Generation](https://arxiv.org/pdf/2406.10970) 的论文提供了深入的信息。
   - 该论文概述了开发 JASCO 模型所采用的方法论和创新方法。



**提及的链接**：<a href="https://huggingface.co/facebook/jasco-chords-drums-melody-1B">facebook/jasco-chords-drums-melody-1B · Hugging Face</a>：未找到描述

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1328849017811238984)** (1 条消息): 

> `Ambient Agents, DSPy 实现示例` 


- **关于使用 DSPy 实现 Ambient Agents 的提问**：一位成员询问如何使用 DSPy 实现 **Ambient Agent**，并向已经实现过的成员征求示例。
   - *讨论中未分享具体的示例。*
- **对 DSPy 实现的兴趣**：另一位成员表示有兴趣查看 DSPy 的**实现示例**，特别是与 Ambient Agents 相关的示例。
   - 邀请其他成员分享任何相关的资源或经验。

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1328734718702911510)** (1 条消息): 

> `Healthcare and AI, Finance and AI, AI-enabled solutions` 


- **关于医疗保健与金融领域 AI 的即将举行的会议**：请在 **2025 年 1 月 16 日**，**IST 时间下午 4:00 至 5:30** 加入我们，与全球专家小组共同探讨 **AI 的影响** 及其在医疗保健和金融领域的应用。
   - 本次活动旨在吸引 **AI-enabled solutions** 的构建者、运营者和所有者，共同讨论这些行业的机遇与挑战，注册链接请点击 [此处](http://bit.ly/3PxJXbV)。
- **关于 AI 解决方案的小组讨论**：会议将由专家组成的小组讨论 AI 如何改变 **医疗保健** 和 **金融** 行业。
   - 鼓励参与者就这些领域的交叉点进行前瞻性讨论。



**提到的链接**：<a href="http://bit.ly/3PxJXbV">欢迎！邀请您参加会议：医疗保健、金融与人工智能。注册后，您将收到一封关于加入会议的确认电子邮件。</a>：医疗保健、金融和人工智能 (AI) 在当今世界日益交织。这些领域之间的互连性也带来了机遇和挑战……

  

---


---


---


---


---


{% else %}


> 完整的各频道详情已针对电子邮件进行截断。 
> 
> 如果您想查看完整的详情，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}