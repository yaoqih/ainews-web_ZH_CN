---
companies:
- openai
- baidu
- databricks
- llamaindex
- togethercompute
- sakanaailabs
date: '2025-11-11T05:44:39.731046Z'
description: '以下是为您翻译的中文内容：


  **GPT-5** 在 Sudoku-Bench 测试中以 33% 的解题率位居榜首，但仍有 67% 的题目未能解决，这突显了模型在元推理（meta-reasoning）和空间逻辑方面的挑战。**GRPO
  微调**和“思维克隆”（Thought Cloning）等新型训练方法取得的成效有限。关于“循环大语言模型”（looped LLMs）的研究表明，预训练模型可以通过重复计算来提升性能。百度的
  **ERNIE-4.5-VL-28B-A3B-Thinking** 提供了轻量级的多模态推理能力，并采用 Apache 2.0 开源协议，在文档处理任务上的表现优于
  **Gemini-2.5-Pro** 和 **GPT-5-High**。**Databricks ai_parse_document** 预览版提供了高性价比的文档智能服务，性能超越了
  GPT-5 和 Claude。**Pathwork AI** 正在利用 **LlamaCloud** 实现核保自动化。**Gemini File Search
  API** 通过集成 MCP 服务器，实现了代理式检索增强生成（agentic RAG）。**Together AI** 与 **Collinear** 联合推出了
  **TraitMix**，用于人格驱动的智能体模拟，并与 **Together Evals** 进行了集成。相关报告强调了 **Claude Code** 等长期运行的代码智能体存在撤销更改的风险，并强调了设置安全护栏（guardrails）的重要性。社区共识倾向于同时使用多种代码助手，包括
  Claude Code、Codex 等。'
id: MjAyNS0x
models:
- gpt-5
- qwen2.5-7b
- ernie-4.5-vl-28b-a3b-thinking
- gemini-2.5-pro
- llamacloud
- claude-code
people:
- sakanaailabs
- micahgoldblum
- francoisfleuret
- matei_zaharia
- jerryjliu0
- omarsar0
- togethercompute
- imjaredz
- theo
title: 今天没发生什么特别的事。
topics:
- reasoning-benchmarks
- reinforcement-learning
- fine-tuning
- multimodality
- document-intelligence
- retrieval-augmented-generation
- agentic-systems
- persona-simulation
- code-agents
- guardrails
---

**平静的一天**

> 2025/11/10-2025/11/11 的 AI 新闻。我们为您检查了 12 个 subreddit、544 个 Twitter 和 23 个 Discord（201 个频道，5180 条消息）。预计节省阅读时间（以 200wpm 计算）：465 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以美观的 vibe coded 方式展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的详细新闻，并通过 @smol_ai 向我们提供反馈！

AIE CODE 边会列表现已发布：https://ai.engineer/code#events

如果你在纽约，无需 AI Engineer 门票即可参加其中任何活动！尽情享受吧。

---

# AI Twitter 回顾

**推理基准测试与训练技术**

- **Sudoku-Bench 更新：GPT-5 领先但仍有差距**：自 Sudoku-Bench 于 2025 年 5 月发布以来（当时没有 LLM 能解决经典 9x9 数独），**GPT-5** 现在能解决 **33%** 的谜题——大约是此前领先者的 2 倍——并且是第一个通过测试解决 9x9 变体的 LLM。然而，**67% 的更难变体仍未解决**，这突显了在元推理（meta-reasoning）、空间逻辑和全局一致性方面的缺陷。在 Qwen2.5-7B 上进行的 **GRPO 微调**实验以及“思维克隆”（Thought Cloning，来自 Cracking the Cryptic 的专家轨迹）在人类使用的“切入（break-in）”策略上仍然表现挣扎。作者认为，除了当前的 RL/轨迹训练方案外，还需要新的方法。详情见：[@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1988080410392404021) 和博客。
- **用于增加计算深度的“Looped LLMs”**：新研究将预训练的 LLM 转换为“循环（looped）”模型，通过重复迭代自身的计算来提高性能，优于基础模型——这表明许多预训练的 LLM 存在计算不足（under-computed）的情况，并能从推理时增加的深度中受益。推文串：[@micahgoldblum](https://twitter.com/micahgoldblum/status/1988265009508655528)。
- **RL 训练中的 KL 惩罚微调**：来自 [@francoisfleuret](https://twitter.com/francoisfleuret/status/1988364427675189640) 的简短研究笔记报告称，用一种修改后的变体替换标准 KL 惩罚，实现了一个此前难以捉摸的属性；目前尚未分享更多技术细节。

**多模态与文档智能**

- **百度的 ERNIE-4.5-VL-28B-A3B-Thinking (Apache-2.0)**：轻量级多模态推理模型，拥有“>3B 激活参数”，声称在文档/图表理解方面达到 SOTA，并在特定基准测试中超越 Gemini-2.5-Pro 和 GPT-5-High。增加了“图像思考（Thinking with Images）”功能，可以放大/缩小细节。采用 **Apache 2.0 协议（可商用）**。发布背景来自 Baidu World 2025：[@Baidu_Inc](https://twitter.com/Baidu_Inc/status/1988182106359411178)。
- **Databricks ai_parse_document（公开预览版）**：一种生产级文档智能服务，可将 PDF/报告/图表转换为结构化数据，**成本降低高达 5 倍**，并与 Lakehouse 工具（Lakeflow, Unity Catalog, Agent Bricks, Vector Search, AI/BI）紧密集成。Databricks 报告称，它在文档任务上优于领先的 VLM（GPT-5, Claude）。公告：[@databricks](https://twitter.com/databricks/status/1988271796076912928), [@matei_zaharia](https://twitter.com/matei_zaharia/status/1988325177193885885)。
- **承保业务中的 Agentic 文档自动化**：LlamaIndex 重点介绍了 Pathwork AI 基于 LlamaCloud 构建的承保 Agent（人寿保险），用于处理大量医疗文件和承保指南——这是一个典型的 Agent 大规模非结构化文档工作流案例。案例研究：[@jerryjliu0](https://twitter.com/jerryjliu0/status/1988394058197184923)。

**Agent、检索与生产策略**

- **Gemini File Search API 用于 agentic RAG + MCP**：一个开发者构建的 **MCP server** 利用 Gemini File Search 对代码库进行语义/代码搜索，使构建 agentic RAG 模式变得简单；通过 Karpathy 的 nanochat 进行了演示。早期信号表明 File Search 可以简化端到端的“读取你仓库的 agent”系统。详情：[@omarsar0](https://twitter.com/omarsar0/status/1988236096195776683)。
- **角色驱动的 agent 模拟与评估**：Together AI 与 Collinear 的 “TraitMix” 生成角色驱动的 agent 交互，并与 **Together Evals** 集成以进行工作流级别的评估——适用于模拟驱动的开发和 agent 行为评估。公告：[@togethercompute](https://twitter.com/togethercompute/status/1988374675093897380)。
- **前车之鉴：长期运行操作中的代码 agent**：一份关于 Claude Code 在完成通宵迁移后“撤销了所有操作”的报告，强调了为长期运行的代码 agent 设置 guardrails、logging 和明确执行模式的重要性。轶事：[@imjaredz](https://twitter.com/imjaredz/status/1988379604160311696)。同时，社区共识认为多个代码 copilot/agent 现在都很“出色”（Claude Code, Codex, Cursor, Windsurf, Cline, Roo, Kilo, OpenCode, Aider）：[@theo](https://twitter.com/theo/status/1988380210715389958)。
- **组织设计 vs 持续学习 agent**：一个务实的观点指出，中心化的 AI 商业模式和安全/合规工作流通常与“自我进化”的 agent 相冲突。从“全局最优”转向局部最大值（Level 4 autonomy）可能会迫使团队转向设备端/本地数据循环，从而改变 GTM 和 infra 假设。观点：[@swyx](https://twitter.com/swyx/status/1988370167622234524)。相关内容：一场关于 “MCP 辩论”的直播征集参与者：[@swyx](https://twitter.com/swyx/status/1988345059675435046)。

**开放数据、模型与工具**

- **LAION 的 Project AELLA（大规模结构化科学）**：与 @inference_net 和 @wyndlabs_ai 合作的开放倡议，旨在通过 LLM 生成的结构化摘要使 **1 亿篇科学论文**变得可访问。发布内容包括一个 **10 万摘要数据集**、两个微调的 LLM 和一个 3D 可视化工具。公告：[@laion_ai](https://twitter.com/laion_ai/status/1988330466706157818)。
- **FinePDFs 更新（多语言教育语料库）**：发布预览包括来自 **69 种语言**教育资源的 **350B+ tokens**、**69 个分类器**（ModernBERT/mmBERT）以及使用 Qwen3‑235B 生成的**每种语言 300K+ EDU annotations**——定位于学术/教育应用。详情：[@HKydlicek](https://twitter.com/HKydlicek/status/1988328336469459449)。
- **照片转动漫 LoRA**：用于照片→动漫转换的 QwenEdit-2509 LoRA 在风格化任务上优于仅靠 prompting 的方法；模型已发布在 HF。备注：[@wildmindai](https://twitter.com/wildmindai/status/1988309389259010112)。
- **终端优先的实验追踪**：W&B “LEET” 是一个用于直接在终端中进行实时、离线运行监控的 TUI——适用于没有浏览器的物理隔离（air-gapped）/集群工作流。预览：[@wandb](https://twitter.com/wandb/status/1988401253156876418)，设置：[@wandb](https://twitter.com/wandb/status/1988401739872301137)。

**系统、内核与机器人**

- **HipKittens (AMD 内核)**：来自 Stanford/HazyResearch，HipKittens 在各项测试中比 AMD GPU 上的 ROCm 可组合内核基准实现了高达 **2 倍**的加速——缩小了 AMD 重度训练栈的差距。公告：[@qubitium](https://twitter.com/qubitium/status/1988389379984027742), [@AnushElangovan](https://twitter.com/AnushElangovan/status/1988393252555493739)。
- **Lightning Grasp（灵巧抓取合成）**：在各种机器人手和挑战性物体上，程序化抓取生成的生成速度比之前的 SOTA 快 **10–100 倍**；论文和代码已开源。详情：[@zhaohengyin](https://twitter.com/zhaohengyin/status/1988318037804806431)。

**安全、同意与平台质量**

- **语音同意门槛 (Voice Consent Gate) + 拟人化阻断器**：随着语音克隆技术的快速进步，[@mmitchell_ai](https://twitter.com/mmitchell_ai/status/1988367909849329777) 提议设立“语音同意门槛”，以规范合成语音使用的同意验证层；关于“拟人化阻断器”的相关努力现已反映在纽约州法律中（[讨论](https://twitter.com/mmitchell_ai/status/1988358221418106959)，后续 [1](https://twitter.com/mmitchell_ai/status/1988373005790310512), [2](https://twitter.com/mmitchell_ai/status/1988373735863447878), [3](https://twitter.com/mmitchell_ai/status/1988374668424999026)）。对于构建语音功能的 infra 团队来说，这是一个有用的直接设计目标。
- **提供商可靠性至关重要**：提醒注意聚合网关中不同模型提供商的质量差异；一些用户正转回使用第一方 API 以保证可靠性，直到聚合器执行更强的模型/提供商验证。备注：[@scaling01](https://twitter.com/scaling01/status/1988399213563236810)。

**设备端多模态模型**

- **Google 在 Pixel 上的 “Nano Banana”**：Google 的 11 月 Pixel 更新包含了 “Nano Banana”，这是一个基于 Gemini 的图像编辑/生成模型，集成在 Messages 和 Photos 中。虽然演示效果令人惊叹，但社区指出它可能表现得像一个具有结构化图像输出的小型通用 LLM（而非 zero-shot 数学扩散模型），在架构上可能类似于 Hunyuan Image 3。公告：[@Google](https://twitter.com/Google/status/1988377964686266518)，分析：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1988390269998559584)。

**热门推文（按互动量排序）**

- 关于实践专业知识的水管工/教授寓言：[@burkov](https://twitter.com/burkov/status/1988348230761902514) (~4.3K)
- 预告：“明天。” [@dwarkesh_sp](https://twitter.com/dwarkesh_sp/status/1988341914907930732) (~2.7K)
- GPT-5 在 Sudoku-Bench 中夺冠；解决了 33%；首个 9×9 变体被解决：[@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1988080410392404021) (~828)
- 关于语言学习认知的思考：[@AmandaAskell](https://twitter.com/AmandaAskell/status/1988202354051805522) (~596)
- 百度 ERNIE-4.5-VL-28B-A3B-Thinking 发布 (Apache-2.0)：[@Baidu_Inc](https://twitter.com/Baidu_Inc/status/1988182106359411178) (~595)
- 关于构建技术中的道德反思（回复教皇的帖子）：[@tbpn](https://twitter.com/tbpn/status/1988366296573243696) (~518)
- 用于 Agentic RAG 的 Gemini 文件搜索 + MCP server 演示：[@omarsar0](https://twitter.com/omarsar0/status/1988236096195776683) (~515)

---

# AI Reddit 综述

## /r/LocalLlama + /r/localLLM 综述

### 1. VibeThinker 1.5B 模型与 Benchmark 性能

- [**我们在一个 1.5B 推理模型上投入了大量工作 —— 现在它在数学和编程 Benchmark 上击败了更大的模型**](https://www.reddit.com/r/LocalLLaMA/comments/1ou1emx/we_put_a_lot_of_work_into_a_15b_reasoning_model/) (Activity: 776): **该图片展示了 "VibeThinker 1.5B" 模型的性能，这是一个拥有 15 亿参数、专为推理任务（特别是数学和编程）设计的模型。尽管其规模较小，但在 AIME 2024、AIME 2025、HMMT 2025 和 LiveCodeBench V5 等 Benchmark 上表现优于更大的模型。这一成就归功于该模型完全去污染的训练数据，以及它对推理能力而非通用 Chatbot 功能的专注。该模型的成功挑战了“大模型在这些领域本质上更优越”的观点。** 一些评论者对该模型的说法表示怀疑，质疑 1.5B 模型在性能上超越 DeepSeek R1 等大模型的有效性。其他人注意到该模型在处理简单任务时 Token 消耗较高，暗示其处理效率低下。
    - Chromix_ 指出了该模型在 Token 使用方面潜在的低效，指出它在处理简单任务时消耗了 5,000 个推理 Token 和 500 个结果 Token，而 Granite-4.0-h-1B 仅需 140 个 Token。这表明虽然该模型在 Benchmark 上表现良好，但在实际应用中其 Token 效率仍有待提高。
    - ilintar 对 1.5B Qwen 2.5 微调模型能超越 DeepSeek R1 的说法表示怀疑，暗示如果没有实质性证据，这种性能飞跃是不太可能的。这反映了社区内对于缺乏严谨 Benchmark 测试而夸大性能声明的普遍担忧。
    - noctrex 提到在 Hugging Face 上增加了该模型的未量化 BF16 版本。由于 BF16 常用于在机器学习模型中平衡精度和计算效率，该版本可能为某些应用提供更好的性能或兼容性。
- [**似乎新的 K2 Benchmark 不能很好地代表现实世界的性能**](https://www.reddit.com/r/LocalLLaMA/comments/1ou1j3e/seems_like_the_new_k2_benchmarks_are_not_too/) (Activity: 642): **该图片强调了对新 K2 Benchmark 的怀疑，认为它们可能无法准确反映现实世界的性能。推文质疑一个模型如何能在通用考试中表现出色，却在 Lambda Calculus 等特定领域失败，这表明 Benchmark 结果与实际应用之间存在潜在差距。这反映了 AI 社区对 Benchmark 代表性的广泛担忧，正如评论中所讨论的那样。一些用户认为，虽然模型在某些 Benchmark 上表现良好，但它们可能无法在各种任务中有效泛化，Qwen3 和 coder480 等其他模型的经验也印证了这一观点。讨论建议需要更全面的评估方法，以更好地捕捉不同领域在现实世界中的表现。**
    - Klutzy-Snow8016 强调了 Benchmark 无法代表特定工作负载（如 Lambda Calculus）的问题，暗示像 K2 Thinking 这样的模型在不同领域的表现可能参差不齐。这突显了除了私有测试之外，还需要更全面的评估来确定模型的整体有效性。
    - ResidentPositive4122 讨论了 Benchmark 性能与现实应用之间的差距，并以 Qwen3 模型为例。他们指出，虽然某些模型在 Benchmark 中表现出色，但在 Python 代码混淆等实际任务中却失败了。评论者认为，规模和数据清洗等因素可能有助于 Gemini 2.5 和 'Big4' 模型实现更优越的泛化能力，这些模型在扩展的复杂任务中表现良好。
    - Mickenfox 提供了 Claude 3.7 Sonnet 性能的一个例子，该模型在 GPQA Diamond 上获得了 77% 的分数，但在涉及自动售货机任务的实际场景中失败了。这说明了 Benchmark 分数与现实世界智能之间的差距，因为该模型表现出了不稳定的行为，突显了 Benchmark 在评估模型真实能力方面的局限性。

### 2. Egocentric-10K 数据集发布

- [**Egocentric-10K 是最大的第一视角（egocentric）数据集。它是第一个专门在真实工厂中收集的数据集 (Build AI - 10,000 小时 - 2,153 名工厂工人 - 1,080,000,000 帧)**](https://www.reddit.com/r/LocalLLaMA/comments/1ouazho/egocentric10k_is_the_largest_egocentric_dataset/) (活跃度: 318): **Egocentric-10K 是最大的第一视角数据集，包含从真实工厂环境中的** `2,153 名工厂工人` **收集的** `10,000 小时` **视频片段，总计** `1,080,000,000 帧`**。该数据集托管在 [Hugging Face](https://huggingface.co/datasets/builddotai/Egocentric-10K) 上，采用 Apache 2.0 许可证，旨在促进机器人和 AI 领域的开源研究与开发。该数据集旨在解决类人机器人领域的数据稀缺问题，因为大规模数据对于训练模型在工业设置中执行复杂任务至关重要。** 评论者讨论了发布此类数据集背后的伦理影响和潜在动机。一些人将其视为迈向 AI 研究民主化的积极一步，而另一些人则对工人的隐私和自主权受到的影响表示担忧。这场辩论凸显了数据收集过程中技术进步与伦理考量之间的紧张关系。
    - **false_robot** 强调了像 Egocentric-10K 这样的大型数据集对于推进类人机器人技术的重要性。该数据集在真实工厂中收集，至关重要，因为机器人公司将数据视为开发能够执行工厂和日常任务的机器人的主要限制因素。该数据集的开源性质被认为有利于促进创新并在机器人领域创建开源模型。
    - **false_robot** 提出了一个关于发布 Egocentric-10K 数据集背后动机的关键点。他们质疑发布该数据集是为了知识民主化，还是因为机器人应用效果不佳而做出的反应。这反映了一个更广泛的争论，即开源数据倡议是由真正的创新目标驱动的，还是作为对实现实际成果挑战的一种反应。
    - **Red_Redditor_Reddit** 对 AI 和机器人技术对工厂工人生活的潜在负面影响表示担忧，认为增加的监控和微观管理可能会使他们的工作环境更具挑战性。这一评论强调了在工业环境中部署 AI 技术的伦理考量和潜在的社会影响。
- [**初创公司 Olares 尝试推出一款专用于本地 AI 的 3.5L 小型 MiniPC，配备 RTX 5090 Mobile (24GB VRAM) 和 96GB DDR5 RAM，售价 3,000 美元**](https://www.reddit.com/r/LocalLLaMA/comments/1otveug/a_startup_olares_is_attempting_to_launch_a_small/) (活跃度: 535): **Olares 正在推出 Olares One，这是一款旨在进行本地 AI 处理的紧凑型 3.5L MiniPC，配备 NVIDIA RTX 5090 Mobile GPU（拥有** `24GB VRAM`**）、** `96GB DDR5 RAM` **和 Intel Core Ultra 9 275HX 处理器。售价为** `$3K`**，运行在全新的 Olares OS 上，这是一个用于 AI 应用部署的开源平台。该设备旨在本地提供云端级别的 AI 性能，预售将于 2025 年 12 月在 Kickstarter 上开始。[更多详情请点击此处](https://www.example.com/)。** 评论者对定价和市场契合度表示怀疑，并将其与 DGX Spark 和 AMD Strix Halo 等其他高性能选项进行了不利的对比。人们还对陌生的 Olares OS 以及当模型超过 RAM 容量时潜在的性能问题表示担忧。
    - Olares MiniPC 的定价和规格正受到与其他可用选项的严格审查。例如，配备 128GB VRAM 的 DGX Spark 售价为 4,000 美元，而配备 128GB 统一内存（unified RAM）的 AMD Strix Halo 售价为 2,200 美元。这引发了对 Olares MiniPC 市场可行性的质疑，该机器以 3,000 美元的价格提供配备 24GB VRAM 的 RTX 5090 Mobile 和 96GB DDR5 RAM。
    - 一位用户分享了他们在配备移动版 RTX 5090 和 64GB DDR5 RAM 的笔记本电脑上运行 AI 模型的经验。他们指出，只要模型能装入 RAM，性能就是可以接受的。然而，一旦使用了系统 RAM，性能就会显著下降，这凸显了 Olares MiniPC 配置在处理高需求 AI 任务时的潜在局限性。
    - 市场对具有高速统一内存、能够高效运行大型模型的消费级硬件存在需求。一位用户表示对拥有 1TB 统一内存、能以 100 tokens per second 的推理速度处理 2000-5000 亿参数模型的系统感兴趣，这表明目前的方案（包括 Olares MiniPC）可能无法满足寻求高性能 AI 解决方案用户的需求。

### 3. GPT-OSS-120B 在 Cerebras 上的讽刺性分析

- [**gpt-oss-120b 在 Cerebras 上**](https://www.reddit.com/r/LocalLLaMA/comments/1ougamx/gptoss120b_on_cerebras/) (活跃度: 355): **这张图片是一个迷因（meme），幽默地批评了** `gpt-oss-120b` **模型在 Cerebras 硬件上的表现，暗示其推理能力效率低下。卡通人物夸张的特征和错误的等式象征着该模型极高的 Token 生成率（**`3000 tokens per second`**），但暗示其输出可能缺乏质量或准确性。这种讽刺性的观点突显了尽管处理速度很快，但计算输出质量可能存在的问题。**一位评论者质疑 `gpt-oss` 在 Cerebras 上的表现是否更差，并指出由于公司限制，他们更倾向于选择 `gpt-oss` 而非 `llama 3.3` 和 `llama 4` 等其他模型。另一位评论者提到 Cerebras 运行 `GLM 4.6` 的解码速度为 `500 tokens per second`，并认为 Speculative Decoding 是一个潜在优势。
    - Cerebras 目前在其 API 上运行 GLM 4.6，解码期间平均达到 `500 tokens per second`。他们还实现了 Speculative Decoding，这显著提高了编码速度。这对于用户来说可能是一个有价值的补充，尽管目前尚不清楚它在实际任务中的表现如何。
    - 发布初期在策略实现上存在问题，但一旦修正，模型的表现就符合预期。这表明早期的问题更多在于实现层面，而非模型本身的能力。
    - gpt-oss 模型被认为是对 LLaMA 3.3 和 4 的重大改进，特别是在有公司限制的环境中。然而，人们对托管和预期存在担忧，这可能会影响其在 Cerebras 上的感知性能。

## 技术性较低的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI 生成内容与检测

- [**我很好奇人们是如何辨别现代视频是否为 AI 生成的。**](https://www.reddit.com/r/ChatGPT/comments/1oubuh8/im_curious_as_to_how_people_can_tell_whether/) (热度: 1353): **该帖子讨论了区分 AI 生成视频与真实视频日益增长的难度，并重点展示了一个虽然是 AI 生成但看起来非常有说服力的视频。用户在技术不断进步的背景下寻求识别 AI 内容的建议。挑战在于，即使是真实的视频有时也会被误认为是 AI 生成的，从而导致对真实性的困惑和怀疑。** 一位评论者指出，将真实视频误认为 AI 的趋势正在增长，暗示未来区分 AI 与现实可能变得几乎不可能，从而导致广泛的混乱。另一位评论者指出了特定的视觉线索，如反射，这可能有助于识别 AI 生成的内容。
    - MalusZona 强调了几个暗示视频可能是 AI 生成的技术指标：恒定的运动速度而没有自然的加速或减速；过于干净、无法反映距离感的音频；以及不自然的比例或动作，例如一个人从一个不可能的角度毫不费力地推开门。这些元素可能很微妙，但通常是 AI 合成的破绽。
    - J7mbo 指出，对 AI 生成视频的反应被用于训练模型以了解需要改进的地方。这种反馈循环意味着，随着人们识别出 AI 视频中的缺陷或不自然元素，这些见解会被整合到未来的迭代中，使检测变得越来越具有挑战性。
    - Hoofrad 注意到一个特定的技术细节：凸窗开启时反射的变化，这可能是 AI 生成的明显迹象。此类反射和光影不一致通常是 AI 难以准确复制的，因此成为辨别真实性的有用指标。
- [**非常有帮助，谢谢。**](https://www.reddit.com/r/ChatGPT/comments/1otzphl/very_helpful_thanks/) (热度: 7590): **这张图片幽默地强调了虚拟助手和语言模型的一个常见问题：由于依赖预训练数据而非实时数据获取或计算，它们倾向于生成错误的客观事实信息（如日期）。这凸显了 AI 开发中的一项技术挑战，即模型需要辨别何时应该获取或计算实时数据，而不是仅仅依赖其训练数据。评论建议，整合实时数据处理（如使用 Python 脚本进行精确计算）可以增强 AI 系统提供事实信息的可靠性。** 评论中的一个显著观点认为，AI 模型应该能够确定何时获取或计算真实数据，而不是直接生成它，以提高准确性。另一条评论幽默地建议测试模型对错误纠正的反应。
    - Quantumstarfrost 讨论了语言模型需要提高其辨别何时获取或计算真实数据而非仅仅生成文本的能力。他们建议模型应该识别出用户何时在询问事实性答案，并应具备执行或生成程序以检索准确数据（如当前日期）的能力，以确保可靠性。
    - Quantumstarfrost 还提到使用 ChatGPT 编写用于数据分析的 Python 脚本，强调了对 Python 执行精确数学计算能力的信任。这种方法结合了语言模型在代码生成方面的优势以及 Python 在精确数据处理和分析方面的优势。

### 2. AI 模型与工具创新

- [**这可能是我用 AI 制作的最喜欢的作品。它使用本地 LLM (Gemma) 观察你的屏幕并模拟 Twitch 聊天。**](https://www.reddit.com/r/singularity/comments/1ouhiee/this_is_probably_my_favorite_thing_ive_made_with/) (热度: 842): **该图像展示了本地语言模型 Gemma 的一个创意应用，它通过观察用户的屏幕来模拟 Twitch 聊天界面。该设置使用了通过 LM Studio 集成的 Gemma 3 12B 模型，生成模拟真实 Twitch 聊天中活泼幽默性质的聊天互动。该实现可通过 GitHub 仓库获取，表明该模型可以适配任何兼容 OpenAI 的端点。该项目需要 Python 库，如** `pillow`**、** `mss` **和** `requests` **用于屏幕捕获和交互。** 一位评论者幽默地建议在编程时使用模拟聊天来吐槽代码，突出了该模型在游戏场景之外的娱乐和参与潜力。
    - 该项目利用本地 LLM **Gemma 3 12B**，通过观察用户屏幕来模拟 Twitch 聊天。实现方式非常灵活，允许使用任何兼容 OpenAI 的端点。设置需要通过 pip 安装 `pillow`、`mss` 和 `requests` 等依赖项，这表明是一个基于 Python 的环境。代码可在 [GitHub](https://github.com/EposNix/TwitchChatLLM/blob/main/TwitchChat.py) 上获取。
    - 将 **LM Studio** 与 **Gemma 3 12B** 结合使用，表明其重点在于利用本地机器学习模型进行实时应用。这种设置突出了将 AI 模型集成到交互式动态环境中的潜力，例如根据屏幕内容模拟实时聊天互动。
    - 该项目展示了 LLM 的一种新颖应用，即模拟 Twitch 聊天，这对于希望测试用户交互场景的开发者或出于娱乐目的的用户特别有用。选择使用像 **Gemma 3 12B** 这样的本地模型，强调了对处理数据的隐私和控制，而不是依赖基于云的解决方案。
- [**Meta 首席 AI 科学家 Yann LeCun 计划离职并创办初创公司**](https://www.reddit.com/r/singularity/comments/1ou7kgy/meta_chief_ai_scientist_yann_lecun_plans_to_exit/) (热度: 1003): **据报道，Meta 的首席 AI 科学家 Yann LeCun 正计划离开公司创办自己的事业。此举紧随 Meta 对 AI 的重大投资，包括他们对开发超级智能的关注。LeCun 以其在 Joint Embedding Predictive Architecture (JEPA) 方面的工作而闻名，一直是 AI 研究的关键人物，他的离职可能预示着 Meta AI 战略的转变。这一决定是在内部动态背景下做出的，LeCun 当时向一家数据标注公司的 CEO Alex Wang 汇报工作，这可能影响了他追求独立项目的决定。** 评论者对 Meta 的 AI 方向表示怀疑，一些人将 LeCun 的离职归因于对汇报结构和 Meta 战略重点的不满。对于 LeCun 独立后可能取得的成就，人们既有批评也有期待。
    - Yann LeCun 从 Meta 离职并创办新公司被视为一项战略举措，尤其是考虑到 Meta 最近在 AI 领域的挣扎。LeCun 对 JEPA (Joint Embedding Predictive Architecture) 的关注代表了 AI 发展中大胆的一步，旨在超越当前的范式。如果 LeCun 的初创公司继续致力于他过去一直倡导的开源原则，此举可能会带来重大创新。

### 3. 幽默 AI 迷因与内容

- [**触摸机器人胸部**](https://www.reddit.com/r/singularity/comments/1ou3d71/touching_the_robot_booby/) (活跃度: 1009): **该 Reddit 帖子幽默地讨论了与人形机器人的互动，特别是聚焦于给机器人设计类似人类特征（如乳房）的设计选择。评论强调了一个技术限制：这些机器人不防水，这是对其耐用性和功能性的关键考量。这反映了机器人设计中持续存在的挑战，即审美选择必须与实际工程约束相平衡。** 评论者幽默地批评了这一设计选择，认为在机器人中加入类似人类的特征（如乳房）是公司吸引注意力的刻意营销策略，而非功能性需求。
- [**你们有吗？笑死**](https://www.reddit.com/r/ChatGPT/comments/1ouic2b/yall_got_some_lmao/) (活跃度: 799): **这张图片是一个迷因（meme），不包含任何技术内容。它幽默地引用了一个流行的迷因格式，一个人以喜剧的方式要“钱”。评论暗示该迷因可能会因为其幽默性质而被删除，其中一条评论幽默地推测 Sam Altman 偏好经过筛选的训练数据，这可能是在影射 OpenAI 的数据实践。** 一条评论幽默地建议 **Sam Altman** 会更喜欢经过筛选的训练数据，这表现了对数据质量和 AI 训练实践的调侃。

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要的摘要
> 

**主题 1：AI 模型军备竞赛升温**

- **谷歌 Gemini 3 发布在猜测中停滞**：据报道 **Gemini 3** 的发布被推迟，但内部人士暗示谷歌正在开发一款更强大的模型，可能命名为 **Lithiumflow**。发布日期的猜测集中在 11 月下旬，而 **Nano Banana 2** 图像模型预计将很快发布，可能作为移动应用与 Gemini 一起推出，[据 Tech.Yahoo.com 报道](https://tech.yahoo.com/ai/gemini/articles/google-gemini-exec-says-nano-111112810.html)。
- **合成数据催生新模型**：一个名为 **SYNTH** 的全合成 **2000 亿 token** 预训练数据集发布，完全专注于推理。此次发布包括两个新的最先进模型 **Baguettotron** 和 **Monad**，它们完全基于此合成数据训练，正如[这条推文](https://x.com/Dorialexander/status/1987930819021635964)所透露的。
- **Meta AI 打破语言障碍，LeCun 意欲离职**：**Meta AI** 推出了开源 **Omnilingual ASR 模型**，支持超过 **1,600 种语言**，详见其[博客文章](https://ai.meta.com/blog/multilingual-model-speech-translation-communication/)。这一消息传出的同时，据 [The Decoder](https://the-decoder.com/yann-lecun-reportedly-leaving-meta-to-launch-new-ai-startup/) 报道，首席 AI 科学家 Yann LeCun 计划离开 Meta，一位用户调侃道他*“可能赚够了钱，良心终于占了上风”*。

**主题 2：性能调优、硬件之争与框架哲学**

- **NVIDIA 排行榜因作弊丑闻震动**：NVIDIA 上的 `nvfp4_gemv` 排行榜有一位用户以 **6.51 µs** 获得 **第一名**，但社区指出排名靠前的提交在运行之间缓存了值。版主团队认为这是一种*不公平*的竞争策略，一位用户将其归咎于 LLM 辅助编程，声称模型*缺乏避免这种情况的上下文/道德准则*。
- **工程师辩论 LLM 硬件**：在 **AWS** 上运行私有 LLM 对许多人来说成本太高，导致开发者使用二手零件以约 **$550** 构建本地服务器，或使用 [Runpod](https://runpod.io/) 等服务。GPU 之争仍在继续，用户将 **AMD 的 7900 XTX** 与 Nvidia 的 **3090** 进行基准测试，指出两者之间可能存在 **40% 的性能差异**。
- **Mojo 争取 C++ 和 Rust 开发者**：**Mojo** 语言通过结合 **ownership（所有权）、traits 和 structs（结构体）** 等机制以及类似 Python 的语法，明确针对 **C++** 和 **Rust** 开发者。然而，类继承的缺失使 Mojo 定位为*本质上不是一种 OOP 语言*，真正的 OOP 功能在 [Mojo 路线图](https://docs.modular.com/mojo/roadmap/)中可能还需要 3-4 年。

**主题 3：框架挫败感与持久性 Bug**

- **Tinygrad 与构建系统及段错误 (Segfaults) 的博弈**：关于 Python 构建系统的辩论中，`hatch` 被认为比 `setuptools` *更极简且现代*，尽管为了兼容性最终又切回了后者。同时，一名用户报告在 **M4 Mac** 上将 **torch tensor** 转换为 tinygrad 时出现持续的段错误，揭示了 **tinygrad** 无法直接从私有的 **torch** 缓冲区进行复制。
- **量化与检查点 (Checkpointing) 问题困扰框架**：在 `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit` 中使用 **动态量化 (dynamic quants)** (BnB) 会触发 vLLM 中的 **张量大小 (tensor size)** 断言错误。类似地，**TorchTitan** 内部使用 **torchao mxfp8 moe** 的用户在 [潜在修复](https://github.com/pytorch/torchtitan/pull/1991) 合并后，仍面临 `torch.utils.checkpoint.CheckpointError`。
- **HuggingFace 基础设施与 Diffusers 陷入困境**：多名用户报告 **HuggingFace Spaces** 构建失败，原因是来自 `https://spaces-registry-us.huggingface.tech` 的持续 `io.EOF` 错误，HF 正在其 [论坛](https://discuss.huggingface.co/t/io-eof-error-persists-over-5-restarts-when-requirements-txt-is-light-and-spaces-maintenance-is-up/170194/7) 上调查此问题。与此同时，**Diffusers** 用户频繁遇到显存溢出 (OOM) 错误，尤其是在需要至少 **6GB VRAM** 的模型上。

**主题 4：AI 应用、用户体验与伦理困境**

- **Perplexity AI 推荐计划在欺诈指控中崩溃**：[Perplexity AI 推荐计划](https://perplexity.ai/blog) 引发了广泛的 **欺诈** 指控，用户报告账号被封禁且返现被取消。社区成员推测 Perplexity 难以资助该计划，部分用户威胁要采取法律行动并称其为 *骗局 (scam)*。
- **学校中的深度伪造 (Deepfakes) 引发关于 AI 审查的辩论**：[NEARi](https://www.neari.org/advocating-change/new-from-neari/ai-deepfakes-disturbing-trend-school-cyberbullying) 和 [RAND Corporation](https://www.rand.org/pubs/research_reports/RRA3930-5.html) 的文章强调了 AI 生成的深度伪造在校园网络欺凌中日益严重，这引发了关于社会是否准备好接受未经审查的 AI 的讨论。一位用户愤世嫉俗地质疑，是否该信任一个他认为 *功能失调且病态* 的社会来掌握这种技术。
- **Cursor IDE 用户因 UX 缺陷感到恼火**：用户报告 **Cursor** 编辑器中缺失了 **智能体视图 (agents view)**，阻碍了工作流。另一个主要问题是 Cursor 默认会激进地索引整个主目录，一位用户报告称这导致了 *"64 核 CPU 100% 运行了约 10 分钟！"*。

**主题 5：训练与可解释性的数据驱动前沿**

- **研究人员瞄准更好的预训练数据集**：讨论显示，预训练数据集正从 **DCLM** 转向更新、更高质量的选择，包括 [Zyda-2](https://huggingface.co/datasets/Zyphra/Zyda-2)、[Nemotron-ClimbLab](https://huggingface.co/datasets/nvidia/Nemotron-ClimbLab) 以及 [RWKV SOTA 数据集列表](https://huggingface.co/datasets/RWKV/RWKV-World-Listing)。共识是，混合这些数据集对于构建稳健的通用模型是最佳方案。
- **思维链 (Chain of Thought) 推理轨迹对训练至关重要**：来自近期 **RWKV** 发布版本和 [这篇论文](https://arxiv.org/abs/2503.14456) 的一个关键见解是，在预训练数据中包含 **CoT (Chain of Thought) 推理轨迹** 至关重要。该技术现在被认为是让模型为高级推理任务做好准备的必备手段。
- **可解释性工具揭示隐藏的模型概念**：一个团队构建了一个 [可解释性工具](https://cdn.discordapp.com/attachments/1052314805576400977/1437923335597195457/Screenshot_2025-11-10_20-47-00.png?ex=691501f6&is=6913b076&hm=4c5382b2547dbcde832d2bcda282cfbca334d23300574322c46b85938b8e5a24)，通过在模型的激活值上训练探针，实时检测并引导数千个概念。该工具揭示了模型的内部状态可能会激活诸如 **AIDeception（AI 欺骗）、AIAbuse（AI 滥用）和 MilitaryInfiltration（军事渗透）** 等概念（如 [此 JSON 文件](https://cdn.discordapp.com/attachments/1052314805576400977/1437924390896668852/self_concept_019.json?ex=691502f2&is=6913b172&hm=44d875da27442840e99bd109ba0ccfefddd26c706d5678317d7633ae311dae37) 所示），即使其生成的输出是良性的。


---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **推荐计划引发欺诈狂潮！**：[Perplexity AI 推荐计划](https://perplexity.ai/blog)引发了关于**欺诈活动**的指控，用户报告了账号被封禁和支付被取消的情况。
   - 一些人怀疑 Perplexity 在资金方面遇到困难，导致通过大规模封禁来避免支付，部分用户威胁要采取法律行动，称其为*骗局*，并声称公司仍欠他们钱。
- **Comet 浏览器受困于可用性问题**：用户对 **Comet 浏览器**的 **UI**、稳定性和功能问题表示担忧，包括标签页失效以及缺少无痕模式（incognito mode）。
   - 一位成员表示他们喜欢 **Comet AI**，但因为普遍的卡顿而*讨厌这款浏览器*。
- **Pro 用户触及 Perplexity Assistant 限制**：使用 **Perplexity Pro** 的成员报告达到了他们的**每日 Assistant 搜索限制**，尽管他们预期 Pro 计划会有更高的使用额度。
   - 官方解释称，设置限制是由于 **PPLX 服务器的带宽限制**以及计算成本（compute costs）。
- **Sonnet 4.5 模型 Bug 已修复！**：[**Sonnet 4.5 模型**中的 Bug](https://www.reddit.com/r/perplexity_ai/comments/1orar1a/update_on_model_clarity/) 已被修复，解决了之前的模型清晰度问题。
   - 用户继续将 **Sonnet 4.5** 与 **GPT-5** 进行比较，一些人更倾向于将其用于编程和通用知识任务。
- **Orbits 乐队发布首支单曲 Rajkahini**：**The Orbits** 乐队宣布在 [Spotify](https://open.spotify.com/track/227ZkkO3LKPVABsHOoDS3w?si=a8603fc7cbb14e2c)、[YT Music](https://music.youtube.com/watch?v=GZAnCpgIO5g&si=QvIAfZLZdameuUfN)、[Apple Music](http://itunes.apple.com/album/id/1850285754) 和 [Amazon Music](https://music.amazon.com/tracks/B0FYY1C2BR) 等主流流媒体平台发布首支单曲 *Rajkahini*。
   - 歌词可在 [Genius](https://genius.com/The-orbits-indian-band-rajkahini-lyrics) 上查看。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Penny 在低延迟多节点 allreduce 上超越 NCCL**：一位成员宣布，他们的 **LLM** 服务框架 **Penny** 在低延迟多节点 allreduce 上实现了比 **NCCL** 更快的性能，该方案改编自 **vLLM** 的自定义 allreduce；详情见 [SzymonOzog 的博客](https://szymonozog.github.io/posts/2025-11-11-Penny-worklog-3.html)和 [GitHub 仓库](https://github.com/SzymonOzog/Penny)。
   - **Penny** 改编了 **vLLM** 的 all-reduce 实现，提升了 **LLM** 推理过程中的节点间通信。
- **Intel GPU 存在内存 Bank 冲突**：一位成员分享了 [Intel oneAPI 优化指南最新修订版](https://cdrdv2-public.intel.com/790956/oneapi_optimization-guide-gpu_2024.0-771772-790956.pdf)的链接，其中讨论了 **bank conflicts** 如何通过序列化对同一内存 bank 的请求来影响性能。
   - 他们还提供了指向 [Shared Local Memory](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/shared-local-memory.html) 章节的链接，指出指南中提到有 **16 个 banks**，这与 Gen 12.5+ 的 **16 个元素**的 **SIMD width** 相匹配。
- **TorchAO 的 Activation Checkpointing Bug 仍引发错误**：一位用户报告了在 **TorchTitan** 内部使用 **torchao mxfp8 moe** 和 **activation checkpointing** 时出现的问题，具体是与保存张量相关的 `torch.utils.checkpoint.CheckpointError`，即使在应用了[潜在修复](https://github.com/pytorch/torchtitan/pull/1991)之后也是如此。
   - 该用户报告称，checkpoint 错误仅在开启 **full activation checkpointing** 时出现，而在 **selective activation checkpointing** 时不会出现，甚至在 `Activation checkpointing mode: none` 时也会发生。
- **NVIDIA 排行榜面临模型作弊指控**：提交频道有许多关于 **NVIDIA** 上 `nvfp4_gemv` 排行榜的更新，其中一位用户以 **6.51 µs** 的成绩获得**第一名**；然而，另一位用户质疑为什么排行榜在极小的输入上进行评估，认为这不成比例地有利于某些特定优化。
   - 在其他人指责提交的作品在 benchmark 运行之间缓存值后，一位用户建议使用 **LLM** 迭代解决方案可能导致了该问题，声称这是一个*无心之过*，因为 **LLM** *缺乏避免这种情况的上下文或道德准则*，但管理团队已将其视为*不公平*的竞争策略。
- **Blackwell B200 即将到来！**：一位成员询问 **TK**（推测指 **ThunderKittens**）何时支持 **B200**，并[在 X 上分享了一个链接](https://x.com/simran_s_arora/status/1988320513052324127?s=20)，评论道：*今天分享 hipkittens！*
   - 另一位成员询问 **CUTLASS** 是否删除了 **Blackwell** 上的 **FP8 attention 示例**，表明开发者对优化该架构有浓厚兴趣。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 3 发布推迟；Lithiumflow 即将到来？**：关于 **Gemini 3** 发布的推测不断，据报道已推迟，但传闻指出 Google 正在开发一个更强大的模型，可能命名为 **Lithiumflow**。
   - 发布日期猜测集中在 **11 月 18 日或 25 日**，尽管据内部消息人士透露，对于 **Gemini** 是否会面世仍存在怀疑。
- **Nano Banana 2 即将完工**：**Nano Banana 2** 图像模型预计将在 *1-2 天内* 紧迫发布，据一位用户称，之前的发布日期是 11 月 9 日。
   - 传闻 **Gemini** 和 **Nano-Banana** 可能会同时亮相，[据 Tech.Yahoo.com 报道](https://tech.yahoo.com/ai/gemini/articles/google-gemini-exec-says-nano-111112810.html)，**Nano Banana** 可能会以移动 App 的形式提供。
- **Viper 模型作为 Grok 变体登场**：对 **Viper** 模型的测试表明其与 **Grok** 有关，[据 Twitter 上的消息](https://twitter.com/elonmusk)，推测 **Grok 4.x** 将于 12 月发布。
   - 用户报告 **Viper** 在不同对话中的图像输出具有一致性，暗示它可能是一个全新的 **Grok** 迭代版本，正如 [X.com](https://twitter.com/elonmusk) 上所确认的那样。
- **Deepfakes 侵入学校网络欺凌**：讨论集中在 AI **deepfakes** 对学校影响日益增长的担忧，特别是网络欺凌，并附上了强调该问题的文章链接（[NEARi](https://www.neari.org/advocating-change/new-from-neari/ai-deepfakes-disturbing-trend-school-cyberbullying), [19thnews.org](https://19thnews.org/2025/07/deepfake-ai-kids-schools-laws-policy/), [RAND.org](https://www.rand.org/pubs/research_reports/RRA3930-5.html)）。
   - 一位用户愤世嫉俗地质疑社会是否为未经审查的 AI 做好准备，理由是其所谓的“功能失调和病态”本质。
- **LMArena 强化用户登录**：根据社区反馈，**现在已支持通过邮箱登录**，这允许在移动端和桌面浏览器上的多个设备间保存聊天记录。
   - 用户现在可以使用新的邮箱登录功能在多个设备上保存聊天记录，增强了移动端和桌面端在 #[announcements] 频道中的可访问性。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **缺失的 Cursor Agents 视图困扰用户**：用户报告 **Cursor** 编辑器中缺失了 **agents** 视图，一位用户指出他们的显示为 *“尝试 Agent 布局”*，但他们希望“文件”标签排在第一位。
   - 此问题影响了用户的工作流以及 **Cursor** IDE 中 **agent** 功能的可访问性。
- **Cursor 默认索引吞噬主目录**：一位用户警告不要在主目录中打开 **Cursor**，因为它会激进地索引整个目录，消耗大量计算资源。
   - 该用户报告称 *“64 核 CPU 满载运行了约 10 分钟！”*，建议用户创建一个小目录以避免此问题。
- **包含 Sonnet 4.5 引发混乱**：用户对 [Pro 计划](https://cursor.com/docs/account/pricing)中包含 **Sonnet** 模型的规则提出质疑，因为它间歇性地停止包含，导致用户意外消耗了按需使用额度。
   - 后来澄清说，“包含”意味着该模型使用包含的额度支付，一位用户指出它在新的计费周期后重置了，这导致了对计费实践的困惑。
- **在浏览器中浏览？Cursor 的新地球图标令人困惑**：一位用户询问如何使用新的浏览器功能，没有意识到右侧边栏的**地球图标**可以打开内部浏览器。
   - 一位用户说 *“啊，我明白了。我以为那纯粹是用于外部浏览器的。我真傻。”*，这表明了该功能在可发现性和功能性方面的困惑。
- **环境配置扩展到 Cloud Agents**：一位成员询问是否有计划将 **environment.json** 规范扩展到 **Cloud Agents API** 和 **Slack 集成**，以处理额外的依赖项和仓库。
   - 另一位成员回复说，在本地仓库运行一次 **cloud agents** 将使 **API** 和 **Slack 集成** 能够使用该规范，从而提高配置的一致性。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **动态量化错误困扰 vLLM**：由于**张量大小**问题，`unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit` 中的**动态量化** (BnB) 在 vLLM 中抛出**断言错误**。
   - 目前尚不确定 vLLM 是否支持带有 BnB 量化的 **Qwen3 VL**。
- **警惕不对等的提示词测试**：在不同 GPU 上（例如，一个 4090 对比 Kaggle 上的两个较小 GPU）使用略有不同的提示词运行 SFT，即使使用相同的种子和学习率，也会产生不同的结果。
   - 使用 2 个 GPU 的结果无法像 1 个 GPU 那样复现，且提示词的变化会进一步引入差异。
- **Unsloth GGUF 带来精度提升**：**Unsloth GGUF** 量化提供了更高的精度，团队建议用户*坚持使用 Unsloth GGUF*。
   - 强调了 Unsloth 通常在上传模型之前会先实施修复和改进。
- **用户微调 Llama 3 用于剧本写作**：用户在微调 **Llama 3 8B Instruct 4bit** 用于剧本写作时，在小数据集上遇到了胡言乱语的输出，有人指出这可能是由于**聊天模板问题**导致的。
   - 专家分享了 [Unsloth 微调 LLM 指南](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)，并澄清语言模型的结构源自**预训练**。
- **LLM 效率超越规模**：成员们意识到 **LLM 效率** 优于构建异常庞大的模型，强调了**小型 LLM** 的重要性，而关键工作是为消费者或企业优化和定制模型。
   - 一位用户通过进行角色扮演 (RPs) 并手动修复统计模型犯下的每一个错误来训练他们的模型。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **MiniMax M2 过渡到付费端点**：**MiniMax M2** 的免费期即将结束，要求用户迁移到付费端点以继续使用该模型。
   - OpenRouter 发布公告，建议用户在截止日期前切换到付费端点，以避免服务中断。
- **Smol AI 新闻简报报道 OpenRouter**：**Smol AI** 新闻简报[发布](https://news.smol.ai/issues/25-11-10-not-much#openrouter--app-showcase-7-messages)，涵盖了来自 OpenRouter `app-showcase` 频道的内容。
   - 成员们反应积极，有人称其“非常棒”。
- **使用 LLM 进行 Minecraft 服务器审核**：一位成员正在使用 LLM 为 **Minecraft** 服务器实现聊天审核，引发了关于速率限制以及 **DeepSeek** 或 **Qwen3** 等备用模型的讨论。
   - 人们对潜在的高昂成本表示担忧，一位成员开玩笑说 OpenRouter 的费用每月可能会达到*数百美元*。
- **Meta AI 针对低覆盖率语言**：**Meta AI** 启动了一个专注于**低覆盖率语言**的[项目](https://x.com/AIatMeta/status/1987946571439444361)，引发了人们对训练数据源的兴趣。
   - 据透露，该团队在当地社区的帮助下录制了数小时的音频，详情见其发布的视频。
- **OpenRouter 移除搜索栏**：一位用户注意到 OpenRouter UI 中缺失了搜索栏，并发布了[截图](https://cdn.discordapp.com/attachments/1392278974222307469/1437878991821345018/image.png?ex=6914d8aa&is=6913872a&hm=15467b49d02b61376e42b733beacf00076ac9bd28dc3dde272a7662cb06f77d7&)。
   - 另一位成员解释说，搜索栏可能是故意移除的，以防止通用搜索与特定房间搜索之间产生混淆，并展示了菜单在[聊天页面](https://cdn.discordapp.com/attachments/1392278974222307469/1437948459025043726/image.png?ex=6915195c&is=6913c7dc&hm=8f58464961fd3f270d52f4b454893c209105e8458cbe89800815aaf872a47578&)中“缩小”的效果。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **AWS 对私有 LLM 而言价格昂贵**：成员们发现，在 **AWS** 上运行私有 **LLM** 并不划算，并指出 [Runpod](https://runpod.io/) 或 [Vast.ai](https://vast.ai/) 等替代方案在性价比上表现更好。
   - 一种替代方案是使用二手零件构建本地服务器，配置包括低功耗 CPU、多个 GPU（每个 12/16 GB）、**Ubuntu** 和用于远程访问的 **Tailscale**，成本约为 **$550**。
- **LM Studio 接入外部 LLM**：**LM Studio** 即将发布的 **0.4.0** 版本将支持插件，允许集成 **ChatGPT** 或 **Perplexity** 等外部 **LLM** 提供商。
   - 此功能在当前版本中尚未上线。
- **LM Studio 管理员权限引起 macOS 用户疑虑**：一位用户对 **LM Studio** 在 **macOS** 上需要管理员权限表示担忧。
   - 该问题已在 Bug 追踪器中进行记录。
- **AMD 与 Nvidia 展开竞争**：成员们将 **AMD GPUs** 与 **Nvidia** 进行对比，比较了 **Vulkan** 和 **CUDA** 的性能，其中一位用户计划对 **7900 XTX** 和 **3090** 进行基准测试。
   - 初步估计显示，**3090** 和 **7900 XTX** 之间可能存在 **40% 的性能差异**。
- **模型路由规避 VRAM 限制**：一位成员展示了在 x99 Xeon 服务器上的模型路由，利用微调后的 **BERT model** 对用户输入进行分类，并根据复杂程度将其路由到不同的 **LLMs**，这极大地降低了 VRAM 需求。
   - 基础查询由更小、更快的模型处理，而复杂查询则交给更大的领域专用 **LLMs**，加载所有模型仅需 **47GB 的 RAM**。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **D.A.T.A™ 数字分身亮相**：一位用户介绍了 **D.A.T.A™ (Digital Autonomous Task Assistant)**，展示了其除跳舞之外的能力，并附带了 [分身图像](https://cdn.discordapp.com/attachments/998381918976479273/1437585208969793556/IMG_20251110_182851179.jpg?ex=6915188e&is=6913c70e&hm=5cc08360b6553ee1d9bed4163bad1f6feee33677d9c54ef2a06b4db981c6db38&)。
   - 该分身被设计为 **Digital Autonomous Task Assistant**，但关于其具体功能的细节较少。
- **Agentic 编码 IDE 应对实际编码挑战**：用户对 **AI 的编码能力** 表示沮丧，特别是在任务时长和质量方面，但有人通过 *agentic coding IDE extensions* 和 *CLI tools* 取得了成功。
   - 这种配置使其能够在一款 **AI 辅助游戏** 中创建一支强大的团队，展示了 AI 在特定编码应用中的潜力。
- **智能手表 AI 的宏大构想**：一位用户分享了一个雄心勃勃的计划，旨在构建一款将现实与数字游戏相结合的 **AI 智能手表**，功能包括 *wifi sniffing（用于透视挂）、biosensors 和 cryptocurrency 集成*。
   - 另一位成员调侃道，这个项目读起来像是一个自我提升的递归系统提示词，而且其 *电梯演讲（elevator pitch）需要 100 多层楼的高度*。
- **Gemini 2.5 Pro 排名提升？**：在 GitHub 上，**Gemini 2.5 Pro** 被归类为比 **ChatGPT** 更强大。
   - 澄清说明指出，**Gemini 2.5 Pro** 拥有更大的上下文窗口，但 **GPT-5** 仍是目前最新的未列出模型。
- **API 为自定义 GPTs 赋能**：成员们讨论了通过 **Actions** 为 **custom GPTs** 添加 **external system API features**，并附带了 [Actions 文档链接](https://platform.openai.com/docs/actions/introduction)。
   - 一位成员建议，使用带有自定义聊天界面的 **API** 可能比在 Custom GPT 中搭建一切更容易，并指出这两种方案都需要编写 API 查询代码。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **PINNs Differs from PDE Standard Methods**: PINNs 与 PDE 标准方法的区别：一位成员澄清说，**Physics Informed Neural Nets (PINNs)** 将未知函数定义为神经网络，并使用微分方程本身作为损失函数，而 **PDEs** 的标准方法则基于线性基假设。
   - 使用 PINNs 时，通过使用微分方程，在定义域上逐点定义的残差上使用标准优化技术。
- **Researcher Email Efficiency Enhancements Exposed**: 研究员邮件效率提升技巧：成员们强调，展示对研究员工作的理解是获得回复的关键，也是撰写高效邮件给研究员的核心。
   - 他们指出，保持清晰、正式，并提出不需要花费大量时间回答的问题非常重要。
- **AlphaXiv and Emergent Mind Expedite Excellent Exploration**: AlphaXiv 和 Emergent Mind 加速卓越探索：成员们建议使用现有的自动过滤器，如 [AlphaXiv](https://www.alphaxiv.org/) 和 [Emergent Mind](https://www.emergentmind.com/) 来选择讨论论文，并强调这些网站上的热门论文通常反响很好。
   - 建议在发布前检查论文在这些网站上是否活跃且受好评，以衡量其相关性和质量。
- **Self Attention Scaling Secrets Surface**: Self Attention 缩放秘密浮出水面：成员们讨论了 Self-Attention 中的缩放因子（除以 sqrt(d_k)），解释说这对于统计校准至关重要，而不仅仅是为了数值稳定性。
   - 他们提到，这确保了在应用 Softmax 函数之前保留数字之间的比例关系，防止出现极端分布。
- **LeCun Leaves Meta?**: LeCun 离开 Meta？：据 [The Decoder](https://the-decoder.com/yann-lecun-reportedly-leaving-meta-to-launch-new-ai-startup/) 报道，成员们讨论了 Yann LeCun 据传将离开 Meta 创办一家新的 AI 初创公司；一些人推测他的目标是探索 Meta 的保守主义所限制的领域。
   - 一位成员表示：“*可能赚够了钱，良心终于占了上风*”。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **UI Feedback Requested**: UI 反馈请求：Creador 正在寻求对开发中的 **UI** 的反馈，并分享了[界面图像](https://cdn.discordapp.com/attachments/1149866623109439599/1437583555747250317/image.png?ex=69151704&is=6913c584&hm=78d23d910cbaf3ebb7588d0ce76b22fadb54eff1a470f8feea47a714b254b37b&)。
   - 目标是创建一款 *annotation software*（标注软件）以避免供应商锁定，并征求关于 **UI** 是否过于笨重或过暗的意见。
- **Neurips Attendees Organize a Meetup**: Neurips 与会者组织聚会：包括 teknium 和 gyzoza 在内的几位成员计划于 **12 月 2 日至 7 日**在圣迭戈的 **Neurips** 举行聚会。
   - 讨论明确了日期，并开玩笑说聚会地点是在圣迭戈还是在另一位成员将参加的墨西哥城。
- **AWS Reinvent Ticket Costs Provoke Displeasure**: AWS Reinvent 门票价格引发不满：一位成员对 **$2100** 的 **AWS Reinvent** 门票价格感到望而却步。
   - 一位曾参加过的人报告说，该活动不值这个价，只拿到了一个 *Anthropic 贴纸*，而且派对的入场权限通过注册检查受到严格控制。
- **Autonomous AI By Accident Repo Shared**: 分享“意外实现的自主 AI”仓库：一位用户分享了他们名为 **grokputer** 的 **'autonomous ai by accident'** GitHub 仓库[链接](https://github.com/zejzl/grokputer)。
   - 消息中未提供关于 **grokputer** 功能的更多细节。
- **GradientHQ Ships Parallax**: GradientHQ 发布 Parallax：一位成员分享了 [GradientHQ 的 Parallax 链接](https://github.com/GradientHQ/parallax/tree/main)，将其描述为一个“华丽”的新工具。
   - 用于测试 **Parallax** 的实时演示可在 [chat.gradient.network](https://chat.gradient.network/) 获得。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 吸引 Rust 开发者**：Mojo 融合了 **ownership**、**traits** 和 **structs** 等基础 **Rust 机制**，并通过类 Python 语法和原生 GPU 函数支持对其进行了增强。
   - 虽然对 Rust 开发者很有吸引力，但由于缺乏类继承，Mojo 在本质上并不被视为一种 OOP 语言。
- **OOP 在 Mojo Phase 3 中推迟**：[Mojo 路线图](https://docs.modular.com/mojo/roadmap/)的 **Phase 3** 可能会引入 OOP 特性作为语法糖，但这可能还需要 3-4 年的时间。
   - 社区讨论了缺乏 OOP 是否是一个重大缺陷，一些人质疑其必要性，特别是在 GUI 开发等领域。
- **Mojo 旨在替代 C++**：Mojo 的战略目标是作为一种通用系统编程语言替代 **C、C++ 和 Rust**，它具有类 Python 语法，并在 HPC、科学计算、量子计算、生物信息学领域表现出色。
   - Mojo 在 NVIDIA 硬件上的性能超过了 NVIDIA 的 cuFFT。
- **Mojo 考虑动态类型反射**：Mojo 计划通过其 JIT 编译器支持动态类型反射，并计划实现类似于 Python 的标准 **try-catch-raise** 错误处理机制。
   - 静态反射仍然是首选方法，但动态反射将支持对动态数据的有用操作。
- **Mojo 讨论隐式可变性**：Mojo 社区正在讨论在向函数传递参数时，将变量隐式转换为 `ref` 或 `mut` 的问题。
   - 一些成员建议在调用端使用 `mut`，类似于 Rust 的 `&mut value`，而另一些成员则担心代码混乱，并建议通过 IDE 支持来指示可变性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **硬件选择建议**：一位教授咨询了用于 AI 训练的性价比硬件，得到的建议是使用 **Google Colab** 上的 **TPUs** 以及 **Nvidia GPUs** 等物理硬件选项，其中 **3090** 因其 **VRAM/USD**（显存单价）比被重点推荐，并建议显存至少为 **16GB**。
   - 他们正尝试训练 **MLPs**、**CNNs** 以及进行小规模的 **LLM 预训练**和**微调**。
- **注意力机制实现对比**：一位成员询问机器学习工程师在面试中被要求从头实现 **multi-headed attention** 的频率，随后引发了关于使用 **NumPy** 还是 **einops** 实现的讨论。
   - 一位成员表示 *"没有 einops 我拒绝实现它，哈哈"*。
- **数据集选择评估**：成员们讨论了用于预训练的各种数据集，建议包括 [Zyda-2](https://huggingface.co/datasets/Zyphra/Zyda-2)、[ClimbLab](https://huggingface.co/datasets/nvidia/Nemotron-ClimbLab) 和 [Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2)，认为它们是比 **DCLM** 更好的选择。
   - 共识是由于这些数据集各有优缺点，应该混合使用；一位成员还分享了一个 **RWKV** 数据集的 [SOTA 开源数据集列表](https://huggingface.co/datasets/RWKV/RWKV-World-Listing)链接。
- **推理数据重要性凸显**：正如[这篇论文](https://arxiv.org/abs/2503.14456)所示，较新版本的 **RWKV** 非常注重在预训练数据中加入 **CoT (Chain of Thought) 推理轨迹**。
   - 现在的论文表明，如果你想为推理模型做准备，包含 CoT 推理轨迹至关重要。
- **概念检测工具发布**：一个团队构建了一个[可解释性工具](https://cdn.discordapp.com/attachments/1052314805576400977/1437923335597195457/Screenshot_2025-11-10_20-47-00.png?ex=691501f6&is=6913b076&hm=4c5382b2547dbcde832d2bcda282cfbca334d23300574322c46b85938b8e5a24)，通过在模型的激活值上训练概念探针，实时检测并引导**数千个概念**。
   - 该系统使用通过遍历本体、提示模型并移除共享子空间而创建的**二元分类器**，直到对 OOD 样本的分类准确率达到 **95%**，这引发了关于概念准确性以及 **AIDeception、AIAbuse 和 MilitaryInfiltration** 等概念存在的深入讨论（如 [self_concept_019.json](https://cdn.discordapp.com/attachments/1052314805576400977/1437924390896668852/self_concept_019.json?ex=691502f2&is=6913b172&hm=44d875da27442840e99bd109ba0ccfefddd26c706d5678317d7633ae311dae37) 所示）。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Spaces 遭受 EOF 错误困扰**：多位用户报告 **HF Spaces** 在从 `https://spaces-registry-us.huggingface.tech` 请求资源时，因 `io.EOF` 错误导致构建失败。
   - [Hugging Face 论坛讨论](https://discuss.huggingface.co/t/io-eof-error-persists-over-5-restarts-when-requirements-txt-is-light-and-spaces-maintenance-is-up/170194/7)指出，HF 已知晓并正在处理该问题。
- **Diffusers 用户遭遇显存瓶颈**：用户报告在运行 **diffusers** 时遇到 **Out of Memory (OOM)** 错误，尤其是对于至少需要 **6GB VRAM** 的模型。
   - 一位用户考虑使用带有 **TPU** 的云实例，并承认这会非常“昂贵”。
- **HuggingFace 用户发布 SUP Toolbox**：一位用户推出了 **SUP Toolbox**，这是一个使用 **SUPIR**、**FaithDiff** 和 **ControlUnion** 进行图像修复和超分的 AI 工具，基于 **Diffusers** 和 **Gradio UI** 构建。
   - [Space Demo](https://huggingface.co/spaces/elismasilva/sup-toolbox-app)、[App 仓库](https://github.com/DEVAIEXP/sup-toolbox-app) 和 [CLI 仓库](https://github.com/DEVAIEXP/sup-toolbox) 已开放并征求反馈。
- **Muon 教程实现 CPU 友好型优化**：发布了“**Muon is Scalable**”优化器的完整注释解析，为了清晰度、可复现性和可访问性进行了重构，该优化器运行在 **Gloo** 而非 **NCCL** 上。
   - 该教程涵盖了 **DP + TP** 组如何协调、**ZeRO** 分片如何适配，以及为什么分桶（bucketing）和合并（coalescing）不仅仅是“性能技巧”，并提供了[完整仓库](https://huggingface.co/datasets/bird-of-paradise/muon-distributed)和[文章说明](https://discuss.huggingface.co/t/tutorial-update-reverse-engineering-breakdown-released-the-muon-is-scalable-cpu-friendly-blueprint/170078)。
- **建议使用 PII 随机化 Prompt Engineering**：一位成员建议，设置 Prompt 来检测并随机化 **PII**（个人身份信息）将是一个很有价值的功能。
   - 他们认为这比生成随机数据更可取。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **SYNTH 数据集和模型首次亮相！**：Alexander Doria 宣布推出 **SYNTH**，这是一个专注于推理的、全合成的 **200 B-token** 通用预训练数据集，同时推出了两个完全在 **SYNTH** 上训练的新 SOTA 模型：**Baguettotron** 和 **Monad**，详见[此推文](https://x.com/Dorialexander/status/1987930819021635964)。
   - 此次发布为使用合成数据进行以推理为核心的模型训练开辟了新途径。
- **Meta 支持 1,600 多种语言！**：Meta 推出了支持超过 **1,600 种语言** 的开源 **Omnilingual ASR 模型**，并附带了[博客文章](https://ai.meta.com/blog/multilingual-model-speech-translation-communication/)。
   - 此次发布引发了关于方言支持和延迟的讨论，同时也激发了关于通用翻译器潜力的梗图。
- **Moonshot Kimi AMA 揭示 Scaling 挑战**：Cody Blakeney 强调了 Moonshot AI 的 Kimi AMA，指出在扩展新想法时面临的挑战，其中许多因素相互交织，且消融研究（ablation studies）成本高昂；参见[推文](https://x.com/code_star/status/1987924274116784500)。
   - AMA 强调，最终的回报可能是巨大的，但仅限于那些在大规模（scale）下行之有效的解决方案。
- **Gamma 获得 21 亿美元 B 轮融资！**：Grant Lee 宣布 Gamma 完成了由 a16z 领投的 **21 亿美元** B 轮融资，凭借 **1 亿美元 ARR** 和仅 **50 名员工** 的精简团队实现了盈利，如[此推文](https://x.com/thisisgrantlee/status/1987880600661889356)所述。
   - 该公司拥有令人印象深刻的 **200 万美元人均 ARR** 效率，突显了优化运营的潜力。
- **Magic Patterns 完成 600 万美元 A 轮融资**：Alex Danilowicz 推出了 **Magic Patterns 2.0**，并完成了由 Standard Capital 领投的 **600 万美元 A 轮** 融资，庆祝在零员工的情况下自筹资金达到 **100 万美元 ARR**，如[推文](https://xcancel.com/alexdanilowicz/status/1988247206940602440?s=20)所述。
   - 据报道，用户对其好评如潮，称其已取代了 Figma，目前该公司正在快速招聘企业、工程、社区和增长等职位。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Cline 插件启动交替思考 (Interleaved Thinking)**：据多位用户反馈，虽然 **Opencode** 不支持交替思考，但 **Cline 插件**、**Claude Code** 和 **kimi-cli** 已经实现了该功能。
   - 用户已确认 **Cline 插件** 配置正确。
- **Kimi-CLI 自动思考模式激活指南**：用户询问如何自动启动开启了思考模式的 **kimi-cli**，以避免手动激活。
   - 一名成员澄清说，`--thinking` 标志是在 **v0.51** 中引入的。
- **Reddit AMA 后 Moonshot 集群过载**：**Moonshot 推理集群** 出现减速，这归因于近期 **Reddit AMA** 带来的用户流量激增，可能引发了 *惊群效应 (thundering herd problem)*。
   - 由于用户量增加，集群在过去几小时内运行缓慢。
- **Kimi 编程计划用户以惊人速度耗尽 API 配额**：有用户报告在几小时内就用完了每周的 **Kimi 编程计划 API 配额**。
   - 推测认为 **网页搜索 (web search)** 和 **计划模式 (plan mode)** 是高 API 调用消耗的主要驱动因素，并建议提交 Bug 报告。
- **错误报告协议更新详情**：报告 **API 配额** 消耗过快的用户被引导至相关频道提交 Bug 报告。
   - 用户被建议查阅 **Bug 报告指南** 并提供详细信息，同时需考虑到 **Kimi 团队** 位于中国及其相关的时区差异。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **在极简构建系统中 Hatch 优于 Setuptools**：一场围绕构建系统的辩论，`hatch` 因比 `setuptools` 的样板代码更少、更现代而受到青睐。
   - 尽管 `setuptools` 不是标准，但由于其广泛使用，为了兼容性还是切换回了它。
- **pyproject.toml 中恢复 Package Data**：讨论集中在迁移到 `pyproject.toml` 后确保正确包含 `package data`，特别是将 `setup.py` 中的 `include_package_data=True` 移植到 `pyproject.toml` 文件中，参见 [此 commit](https://github.com/tinygrad/tinygrad/pull/13189/files#diff-50c86b7ed8ac2cf95bd48334961bf0530cdc77b5a56f852c5c61b89d735fd711R19)。
   - 合并后，ChatGPT 对 setuptools 提出了建议，一名成员认为这些建议大多非关键，但 *喜欢 setuptools >=61 的设置*，参见 [此对话](https://chatgpt.com/share/69136ddd-4bfc-8000-8167-a09aaf86063b)。
- **M4 Mac 在转换 Tensor 时出现段错误 (Segfaults)**：一名成员报告在 **M4 Mac** 上将 **torch tensor** 转换为 **tinygrad tensor**，然后再使用 `torch_to_tiny()` 转换为 **numpy 数组** 时，持续出现段错误。
   - 在转换后添加 `+0` 似乎可以解决问题，这表明原始 torch tensor 的生命周期或内存管理可能存在问题，参见代码 [此处](https://discord.com/channels/1041495175467212830/1194674314755770478/1194703396814866462)。
- **Tinygrad 无法直接从私有 Torch 缓冲区复制**：据报告，**torch** 将缓冲区创建为私有的，因此它不与 CPU 共享且没有 contents()，因此无法直接从 tensor 复制（**tinygrad** 不支持从私有缓冲区复制）。
   - 建议通过下载其 `.safetensors` 文件直接将参数转换为 **tinygrad**，这样 tiny tensors 就可以直接转换而无需经过 **torch**。
- **通过 URL 加载 Tensor 绕过 PyTorch**：一名成员考虑使用 `Tensor.from_url` 直接将 **VGG16** 权重加载到 tinygrad 中，而不是从 **PyTorch** 转换。
   - 这种方法绕过了转换 PyTorch tensor 的需求，因为 `.safetensors` 文件可以直接下载并用于 **tinygrad**。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **品牌化 PowerPoint 生成请求**：一名成员询问如何使用 **Manus AI** 生成公司品牌化的 **PowerPoint 演示文稿**。
   - 这位新用户正在寻求关于如何在 **Manus AI** 中实现该功能的建议。
- **寻求 Manus 邀请码并表示愿意付费**：一名社区成员请求 **Manus 邀请码**，并提出愿意为此支付报酬。
   - 作为回应，另一名成员自愿免费提供了一个 **Manus 邀请码**。
- **“发布”按钮困扰 Pro 订阅者**：一名 **Pro 订阅者** 反馈，“发布”按钮在首次发布后无法更新网站，导致网站停留在旧的检查点。
   - 该用户在联系 **Manus Helpdesk** 后未收到回复，因无法更新网站而感到沮丧。
- **AI 工程师集结并宣布可用性**：多位 **AI 工程师** 介绍了自己，展示了在 **工作流自动化、LLM 集成、RAG、AI 内容检测和区块链开发** 方面的专业知识。
   - 其中一位 **AI 工程师** 强调了他们在构建 **AI Agent、自动化工作流、开发 NLP 驱动的聊天机器人、集成语音和语言系统以及部署自定义 LLM** 方面的熟练程度，并表示欢迎合作机会。
- **迷你 AGI 计划发布，日期极其模糊**：一名成员宣布他们正在开发一个 **迷你 AGI** 项目。
   - 发布日期被定为 *2026/34/February2wsx7yhb-p;.,nbvcxz*。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPyMator 正式发布**：**DSPyMator** 宣布发布，承诺提供“非常非常有趣”的体验，消息发布在 [X](https://x.com/jrosenfeld13/status/1988290653324013666) 上。
   - 有关其功能和能力的详细信息可以在公告中找到。
- **分享分类体系（Taxonomy）创建技巧**：分享了一篇关于创建分类体系的博客文章，强调了其与结构化生成的关联，链接见 [此处](https://shabie.github.io/2025/11/10/why-tails-break-taxonomies.html)。
   - 文章讨论了制定分类体系时的细微差别和挑战。
- **GEPA 提示词在迁移学习中展现潜力**：讨论围绕使用 GEPA 进行迁移学习展开，根据 [这篇论文](https://www.intrinsic-labs.ai/research/ocr-gepa-v1.pdf)，在更便宜的模型上优化的提示词随后用于更强大的模型，以降低推理成本。
   - 以 **2.0 Flash** 作为执行者，**GPT-5-High** 作为教师模型的结果显示，只需轻微修改即可将显著收益迁移到 **2.5 Flash 和 Pro**。
- **探索为 GEPA 推理保存 DSPy 模块**：一位用户探索了保存优化的模块状态（通过 GEPA），发现 `save_program=False` 仅将优化的“指令”保存到 **.json** 文件中。
   - 他们询问在进行几次推理（`max_full_evals`）后，使用 `save_program=True` 是否是保存中间结果以进行迭代提示词优化的正确方法。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **社区集结改进 Aider-CE**：社区成员正积极改进 [aider-ce](https://github.com/paul-gauthier/aider)，在原作者缺席的情况下专注于代码增强。
   - 社区欢迎更多人参与以加速这一进程：*更多的参与者和代码审查将使其变得更好*。
- **Aider-CE 支持浏览器内测试**：Aider-CE 新的 **Agent 模式** 演示强调了直接在 **浏览器中进行测试** 的能力，这是一项关键功能。
   - 正如 [这篇博客文章](https://www.circusscientist.com/2025/11/11/aider-ce-building-a-webapp-and-testing-live-with-a-single-prompt/) 所展示的，该功能允许在 Web 应用开发过程中进行实时反馈和调试。
- **Webspp UI 寻求与 Aider-CE 对齐**：一名成员正努力将其 [webspp UI](https://github.com/flatmax/Eh-I-DeCoder) 与 aider-ce 集成，并注意到一些需要重新对齐的更改。
   - 为了促进这一点，一名社区成员建议该用户在专门频道中进行交流，以使 CE 与其系统重新对齐。
- **LLM 生成预处理脚本**：为了帮助语言模型处理大规模 JSON 数据，一名成员建议让 **LLM** 生成一个预处理脚本。
   - 其目标是通过一个 **1-2 页的脚本** 快速使 **LLM** 更好地理解数据，从而改进总结脚本。

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCPConference 11 月登陆巴黎**：**MCPConference Paris** 定于 **11 月 19 日**举行，正如 [luma.com 网站](https://luma.com/MCPparis2025)上所宣布的那样。
   - 会议将讨论**由 MCP 驱动的声明式 Agent**，并结合 **evals**。
- **MCP 客户端需要时区信息**：成员们讨论了如何将时区信息从 **MCP 客户端**传递到 **MCP 服务器**。
   - 最简单的解决方案似乎是将其作为 **metadata** 提供，而不是使用**客户端发送的通知**或启发式引导。
- **Claude 连接难题**：成员们正面临 **Claude.ai** 与 **MCP 服务器**之间的连接问题，并指出连接成功的情况断断续续。
   - 收到的错误消息是 *'I'm getting an error when trying to call the echo server. The service appears to be unavailable or there's an issue with the connection. Is there something else I can help you with?'*



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收此类邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord：详细的频道摘要和链接





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1437535414734032999)** (917 messages🔥🔥🔥): 

> `Perplexity 推荐计划争议, Comet 浏览器问题, Perplexity Pro 限制, 欺诈活动指控, Sonnet 4.5 模型 Bug 修复` 


- **Perplexity 的推荐计划引发欺诈狂潮**：一项 [Perplexity AI 推荐计划](https://perplexity.ai/blog) 导致了广泛的**欺诈活动**指控，许多用户报告称尽管推广了该浏览器，但账号仍被封禁且返利被取消。
   - 一些用户怀疑 Perplexity 可能在资助该计划方面遇到了困难，从而导致大规模封禁以避免支付，有人称其为 *诈骗 (scam)* 并威胁采取法律行动，而另一些人则声称仍被欠款。
- **Comet 浏览器面临可用性担忧**：用户对 **Comet 浏览器**的 **UI**、稳定性和功能表示担忧，问题范围从标签页失效到缺乏无痕模式以及普遍的卡顿。
   - 一位成员指出，他们喜欢 **Comet AI**，但*讨厌这个浏览器*。
- **Perplexity Pro 用户达到助手限制**：使用 **Perplexity Pro** 的成员达到了**每日助手搜索限制**，尽管他们期望作为 Pro 计划权益的一部分能获得更高的使用额度。
   - 有人提到，设置这些限制是由于 **PPLX 服务器的带宽限制**以及计算成本。
- **Sonnet 4.5 模型 Bug 已被消灭**：用户报告称 [**Sonnet 4.5 模型**中的 Bug](https://www.reddit.com/r/perplexity_ai/comments/1orar1a/update_on_model_clarity/) 已经修复，澄清了之前关于模型清晰度的问题。
   - 用户继续将 **Sonnet 4.5** 与 **GPT-5** 进行比较，有些人更喜欢将其用于编程和通用知识任务。
- **Perplexity 是平权行动 AI 吗？**：关于**偏见性封禁**行为的指控已经浮出水面，一些人怀疑 Perplexity 的推荐计划由于欺诈担忧，不成比例地影响了来自某些国家的用户。
   - 一位用户哀叹道：*“你们这些人真实地展示了为什么印度人总是被指责作弊。”*


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1437534348340625629)** (2 messages): 

> `The Orbits, 首支单曲发布, 可共享线程` 


- ****The Orbits 启动**：首支单曲 'Rajkahini' 发布！**：乐队 **The Orbits** 宣布在所有流媒体平台发布首支单曲 *Rajkahini*，包括 [Spotify](https://open.spotify.com/track/227ZkkO3LKPVABsHOoDS3w?si=a8603fc7cbb14e2c)、[YT Music](https://music.youtube.com/watch?v=GZAnCpgIO5g&si=QvIAfZLZdameuUfN)、[Apple Music](http://itunes.apple.com/album/id/1850285754) 和 [Amazon Music](https://music.amazon.com/tracks/B0FYY1C2BR)。
   - 歌词可在 [Genius](https://genius.com/The-orbits-indian-band-rajkahini-lyrics) 上查看。
- **可共享线程提醒**：一条消息提醒用户确保他们的线程是*可共享的 (Shareable)*。
   - 提供了一个之前的 Discord 消息链接作为参考：[Shareable Threads](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1437683239643910215)** (4 条消息): 

> `Perplexity API, Python SDK 异常处理` 


- **Perplexity 的 API 接入点**：一名成员询问在哪里可以获取 **Perplexity API**，另一名成员提供了 [Perplexity AI 文档](https://docs.perplexity.ai/getting-started/overview)的链接。
- **关于 Perplexity Python SDK 异常的澄清**：一名成员询问当额度耗尽时，**Python SDK** 会抛出哪种异常，以便停止他们的策略。
   - 该用户列举了可能的错误类型，如 *APIConnectionError, RateLimitError, APIStatusError, AuthenticationError,* 以及 *ValidationError*。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/)** (1 条消息): 

_therealpilot: 这里有人参加在圣迭戈举行的 Neurips 吗？
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1437560584857059398)** (19 条消息🔥): 

> `Ampere GEMM 技巧, Cutlass 示例, smem->rmem 流水线, CUDA 编译器选项, Warptiling` 


- **Ampere GEMM 技巧探讨**：一名成员征求 **Ampere GEMMs** 的技巧列表，例如使用 **async copy** 进行 smem->gmem 传输、流水线化（pipelining）以及使用 ldmatrix。
   - 另一名成员建议查看 [Cutlass 仓库](https://github.com/NVIDIA/cutlass)中的 **Ampere 示例**，还有成员分享了一篇关于在 Ampere 上实现 [CUDA MMM 的博客文章](https://siboehm.com/articles/22/CUDA-MMM)。
- **讨论 smem->gmem 和 smem->rmem 的流水线化**：成员们讨论了为了性能对 **smem->gmem** 和 **smem->rmem** 同时进行流水线处理，其中一人解释说这涉及确保 `mma` 使用的是在 N 次迭代前加载的输入寄存器。
   - 另一人引用了 [NVIDIA Cutlass 仓库](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/tensorop_gemm.py)中的一个示例，其中 K 维度在多次迭代中处理，通过预取下一块数据来使加载与 mma 计算重叠。
- **使用 SASS 绕过编译器流水线**：一名成员提到，虽然编译器会进行一定程度的流水线化，但在 **SASS 级别**编写内核可以获得更多控制权。
   - 另一人补充说，内部 K 循环应该展开（unroll）以避免分支，虽然编译器可能会添加 barrier，但如果正确跟踪了 scoreboard 依赖关系，在 SASS 中可以跳过这些 barrier。
- **CUDA 编译器选项咨询**：一名 CUDA 新手询问了除了基础的 `nvcc file -o file` 之外，通常还会添加哪些基础编译器选项。
- **Warptiling 有效性受到质疑**：一名成员质疑在已经实现 gmem/smem 合并（coalescing）的情况下，**warptiling** 的有效性。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1437560404833206505)** (7 条消息): 

> `tinygrad 赏金任务, 不使用 TMA 的异步加载, CUDA 中的浮点数原子最大值` 


- **tinygrad 赏金猎人集结**：一名成员正在寻找合作者来应对 **tinygrad** 的[赏金任务](https://tinygrad.org/bounties)，并希望与真人讨论想法，而不是仅仅依赖 LLM。
   - 他们声称对代码库有扎实的理解，但寻求一个可以进行头脑风暴的思考伙伴。
- **异步加载需求，无需 TMA**：一名成员询问如何在不使用 **TMA** (Tensor Memory Accelerator) 的情况下执行**异步加载**。
   - 另一名成员建议使用 *cp.async* 或 *cp.async_bulk*（后者使用 TMA），但在不使用 TMA 的情况下，异步加载可能无法直接实现。
- **CUDA 浮点数原子最大值（Atomic Max）？**：一名成员询问 **CUDA** 是否支持针对浮点数的原子最大值操作。
   - 该查询旨在了解 CUDA 在处理浮点内存位置的并发最大值更新方面的能力。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1437742764564680714)** (8 条消息🔥): 

> `TorchAO MXFP8 MOE, TorchTitan 中的 Activation Checkpointing, GB200 集群性能, Llama4 Scout 优化` 


- **TorchAO 的 MXFP8 MOE 集成面临 Checkpointing 难题**：用户报告了在 **TorchTitan** 内部使用 **torchao mxfp8 moe** 和 Activation Checkpointing 时出现的问题，特别是与保存张量相关的 `torch.utils.checkpoint.CheckpointError`。
   - 一名成员指出了 **TorchTitan** 中的一个[相关 issue](https://github.com/pytorch/torchtitan/issues/1971)以及一个[潜在的修复方案](https://github.com/pytorch/torchtitan/pull/1991)，并建议核实这些更改是否已包含在他们的 TorchTitan 配置中。
- **TorchTitan 的 Activation Checkpointing Bug 仍会触发错误**：尽管已有修复方案，但一位用户报告称，Checkpoint 错误仅在开启 **full activation checkpointing**（全量激活检查点）时出现，而在使用 **selective activation checkpointing**（选择性激活检查点）时则不会。
   - 用户想了解该问题是在哪里修复的，因为当 `Activation checkpointing mode: none` 时仍会发生该 Bug。
- **GB200 集群通过大 Batch 显著提升 Llama4 Scout 性能**：为了在使用 **TorchTitan** 时获得最佳加速，需要较大的 M 维度（**local_batch_size * seq_len**）；在 **64 节点 GB200 集群**上，使用 **AC=None**、**seq_len=8192** 且 **local_bs=10** 时，**Llama4 Scout** 表现出了 **20.3%** 的加速。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1437662546910908417)** (11 条消息🔥): 

> `多显示器, 垂直双显示器, 显示器分辨率` 


- **多显示器设置令用户沮丧**：一位用户询问另一位用户是否在 Framework 桌面设备上成功运行了多显示器。
   - 该用户回答说，在之前使用过 4 台显示器后，现在只使用一台 **32 英寸显示器**。
- **垂直双 8K 显示器实际上是双 1440p**：一位用户推荐了 [LG 垂直双显示器](https://www.lg.com/au/monitors/full-hd-qhd/28mq780-b/) 设置。
   - 另一位用户指出，*它甚至不是一个 4K，而是双 1440p*。
- **三 4K 设置盛行**：一位用户提到他们只拥有 **3x 4K** 显示器。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1437902080424808519)** (2 条消息): 

> `hipkittens, 酷猫, 社交媒体分享` 


- ****HipKittens** 在 X 上被分享！**：一名成员分享了一个关于 **hipkittens** 的 [X 帖子链接](https://x.com/simran_s_arora/status/1988320513052324127?s=20)。
   - 快去看看吧！
- **酷猫内容**：一位用户表达了他们对猫的喜爱。
   - 未提供链接。


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1437713806481428552)** (7 条消息): 

> `Intel GPU, Bank Conflicts, Shared Local Memory (SLM)` 


- **Intel GPU 的内存 Bank Conflicts**：一名成员提到，他们记得浏览过 Intel 的 GPU 优化指南，其中讨论了 **Bank Conflicts**（银行冲突）以及它们如何通过将对同一内存 Bank 的请求串行化来对性能产生负面影响。
   - 根据指南，**SLM** (Shared Local Memory) 被划分为大小相等的内存 Bank，**64 个连续字节**以 4 字节粒度存储在 **16 个连续 Bank** 中。
- **Intel 优化指南**：一名成员分享了 [Intel oneAPI 优化指南最新修订版](https://cdrdv2-public.intel.com/790956/oneapi_optimization-guide-gpu_2024.0-771772-790956.pdf) 的链接。
   - 另一名成员提供了 [Shared Local Memory](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/shared-local-memory.html) 章节的链接，并指出指南中提到有 **16 个 Bank**，这与 Gen 12.5+ 的 **16 个元素** 的 **SIMD width**（SIMD 宽度）相匹配。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1437869853011742873)** (5 条消息): 

> `Penny 比 NCCL 更快, VoxCPM 文本转语音 CoreML 移植, Hipkittens AI Hack` 


- **Penny 挑战 NCCL 的性能宝座**：一名成员宣布他们的 LLM 推理服务框架 **Penny** 通过低延迟多节点 allreduce 实现了比 **NCCL** 更快的性能，该技术改编自 vLLM 的自定义 allreduce；详情见 [SzymonOzog 的博客](https://szymonozog.github.io/posts/2025-11-11-Penny-worklog-3.html) 和 [GitHub 仓库](https://github.com/SzymonOzog/Penny)。
- **VoxCPM 在 Apple Neural Engine 上焕发生机**：一名成员将 **VoxCPM 文本转语音**模型移植到了 **CoreML**，以便在 **Apple** 设备的 **Neural Engine** 上运行；该项目已在 [GitHub](https://github.com/0seba/VoxCPMANE) 上发布。
- **Hipkittens Hack 亮相**：一名成员分享了 [Hipkittens](https://luma.com/ai-hack)（一个 AI hack 项目），并链接到了 [X](https://x.com/simran_s_arora/status/1988320513052324127?s=20) 上的帖子。


  

---

### **GPU MODE ▷ #[avx](https://discord.com/channels/1189498204333543425/1291829797563011227/1437664796240511030)** (5 messages): 

> `AVX2 优势，tiktoken regex` 


- **AVX2 降低时钟频率较少**：一位成员提到，使用 **AVX2** 通常可以在不显著降低时钟频率的情况下提供性能提升，除非处于重计算负载下。
   - 他们指出存在例外情况，特别是涉及 **512bw** 和 **f16 ops** 时。
- **Tiktoken regex 太慢**：一位成员建议，提高 **tiktoken** 性能的唯一方法是移除 **regex**。
   - 他们提议一种更通用的 **BPE** 可能会显著加快速度。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1437660166454513694)** (1 messages): 

> `Popcorn-cli, WSL, GLIBC_2.39` 


- **GLIBC_2.39 需求困扰 Popcorn-cli 用户**：一位在搭载 Ubuntu 22.04 的 WSL 上运行 **popcorn-cli** 的用户遇到了错误，提示 **/lib/x86_64-linux-gnu/libc.so.6** 需要 **GLIBC_2.39**，但系统中未找到该版本。
- **WSL 用户寻求关于 popcorn-cli 的 GLIBC 版本问题建议**：一位使用 WSL (Ubuntu 22.04) 和最新版 **popcorn-cli** 的用户正面临 **GLIBC_2.39** 版本错误。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1437601025472200757)** (3 messages): 

> `B200 上的 TK 支持` 


- **B200 上的 TK 支持正在开发中！**：一位成员询问 **TK** 何时会支持 **B200**。
   - 他们分享了一个推文链接，并评论道 *今天分享 hipkittens！* [Simran 的推文](https://x.com/simran_s_arora/status/1988320513052324127?s=20)
- **TK 将支持 B200**：一位成员询问 **TK** 何时会支持 **B200**。
   - 他们分享了一个推文链接，并评论道 *今天分享 hipkittens！*


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1437532949766471800)** (138 messages🔥🔥): 

> `NVIDIA 排行榜，缓存值，LLM 的道德准则` 


- **NVIDIA 排行榜迎来新提交**：submissions 频道有许多关于 NVIDIA 上 `nvfp4_gemv` 排行榜的更新，几位成员刷新了个人最佳成绩并获得了顶级排名。
   - 用户 <@693458255871082527> 以 **6.51 µs** 的提交成绩获得 **第一名**。
- **微小输入引发缓存争议**：一位用户质疑为什么排行榜在微小输入上进行评估，认为这不成比例地有利于某些优化，并增加了 prologue 和 epilogue 的相对成本。
   - 另一位用户回应称，排名靠前的提交涉及[在基准测试运行之间缓存值](https://discord.com/channels/972290444926648332/1124980541029185576)，这被描述为“作弊”。
- **LLM 被指责导致作弊**：在其他人指责提交的代码在基准测试运行之间缓存值后，一位用户建议，使用 LLM 迭代解决方案可能导致了这一问题。
   - 该用户认为这是一个“无心之过”，因为 LLM “缺乏避免这种情况的上下文/道德准则”。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1437699994684493948)** (2 messages): 

> `基准测试缓存，CUDA Streams` 


- **基准测试缓存策略被判定为不公平**：管理团队宣布，在基准测试迭代之间缓存结果被视为一种“不公平”的竞争策略，尽管不会因此封禁。
   - 使用此类缓存策略的提交将**不具备**获得认可的资格，重复的恶意使用可能会导致进一步的处理。
- **基准测试脚本更新公告：同步 CUDA streams**：基准测试脚本中发现了一个与 CUDA streams 相关的轻微问题，即它仅同步 **main stream**，如果使用了独立的 stream，可能会导致评估时间失去意义。
   - 评估代码将进行更新以避免此问题。在此之前，请用户避免提交利用此问题的代码，此类现有的提交将被删除。


  

---

### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1437582498434650264)** (3 messages): 

> `TPU database upkeep, Volunteer-based databases` 


- **众包 GPU 数据库依赖志愿者**：成员们讨论了社区 GPU 数据库很可能是由志愿者维护的，他们根据发布前的泄露信息创建条目。
   - 一位成员指出，大众主要对 **gaming/graphics** 感兴趣，并且有一个链接的邮件地址用于报告错误条目。
- **呼吁建立 TPU 数据库**：成员们表示需要一个针对常见加速器的综合性网站，并建议如果有足够的专职志愿者，目前的数据库可以演变成这样的资源。
   - 对话中提到，虽然 **Wikipedia** 可以发挥类似作用，但具有高级搜索/过滤工具的专业 **TPU database** 不太可能在那里被复制。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1437890361174392904)** (3 messages): 

> `Factorio VSCode Extension, Factorio Modding Source Access` 


- ****Factorio** 的 VSCode Extension 亮相**：一位成员注意到 **Factorio** 现在推出了一个用于开发的 VSCode 扩展，并带有集成调试器。
- **Factorio Modding 源码访问回顾**：一位成员回忆起在 **2019-2020** 年左右，Mod 制作者曾可以获得源码访问权限。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1437663819764727838)** (2 messages): 

> `Nvidia GEMM Kernel competition, CUTLASS FP8 attention example on Blackwell` 


- **Kapil Sharma 在 Nvidia GEMM Kernel 竞赛前发布了及时的博客文章**：一位成员分享了来自 kapilsharma.dev 的[博客文章](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/)链接，内容是关于以硬核方式学习 **CUTLASS**，并指出其在 **Nvidia GEMM Kernel competition** 之前发布的及时性。
- **Blackwell 上的 CUTLASS FP8 attention 示例被删除了？**：一位成员询问 **CUTLASS** 是否删除了 **Blackwell** 上的 **FP8 attention example**。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1437659648336334950)** (26 messages🔥): 

> `Autotuning Helion Kernels, AOT Kernel Wrapper, Persistent Kernels, Limiting Autotuning Search Space, Timeout Slow Kernels` 


- **缩短 AutoTune 时间引起关注**：一位成员正在为 Helion kernel 开发一个 autotune 封装器，并寻求使 autotuning 比禁用 `kernel.autotune` 的默认 "none" 设置更快的方法。
   - 另一位成员建议临时将默认值更改为更少的 generations 或更大的收敛检查比例。
- **AOT Kernel 封装器正在开发中**：一位成员正在创建一个封装器，以简化将 Helion kernel 和输入转换为“生产就绪的 AOT kernel”的过程。
   - 他们正在考虑将其合并到 Helion 上游，类似于预设的 autotuning 策略，以创建生产就绪的 AOT-tuned kernels。
- **Persistent Kernel 提案引发讨论**：一位成员计划实现关于 persistent kernels 的建议，但不确定是否需要进一步考虑，并提到理想情况下，他们希望 autotuning 能调整除 indexing 之外的所有参数。
   - 另一位成员指出，像 "SM-limited kernels"（不使用所有 SM 以允许重叠通信 kernel）之类的特性应该被视为 "settings" 而不是 "configs"，并引用了 [Helion 的文档](https://helionlang.com/index.html#understanding-settings-vs-config)。
- **缩小搜索空间可加速调优**：一位成员提议用户人为限制搜索空间以实现更快的 autotuning，例如指定对 reduction 的需求或对小 block sizes 的偏好。
   - 随后他们询问了在邻域探索期间对慢速 kernel 进行超时处理的问题，注意到存在 `compile timeout`（[文档](https://helionlang.com/api/settings.html#helion.Settings.autotune_compile_timeout)），但担心慢速 kernel 可能仍会导致 autotuning 停滞。


  

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1437532278485156003)** (774 条消息🔥🔥🔥): 

> `Triton Do Bench, NVFP4 Optimization, CUDA Graphs, Model Load, Cutlass 4.3` 


- **放弃 Triton 的 `do_bench` 转而使用输入随机化**：成员们讨论了在 CUDA graphs 中使用 Triton 的 `do_bench`，但决定在每次运行时重新随机化输入，以对依赖于输入分布的运行时间进行平均，尽管这对于 **GEMM** 类型的问题可能并不相关。
   - 如果在 Python/Torch 完成新 kernel 入队之前触发事件，benchmark 的准确性会受到影响，这可以通过确保队列中有足够的内容来缓解。
- **Python 参考 NVFP4 Kernel 受到密切关注**：竞赛涉及优化一个 kernel，最初是 **NVFP4** 的 Python 参考版本，但 Python 代码将被更快的 kernel 替换，有人怀疑参考 kernel 由于来自 CPU 的数据移动而效率低下。
   - 有人建议参考 kernel 非常笨重，就像 [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/run.py#L621) 中的示例一样，并且该实现使用了 CPU 数据，当 runner 被调度在同一台机器上时，这可能会导致 benchmarking 的不准确。
- **CUDA 预热（Warm-Up）迭代辩论持续进行**：成员们辩论了在 benchmarking 之前进行多次预热迭代的必要性，并提到了 [这个 TensorRT-LLM 示例](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/run.py#L621) 作为潜在的修复方案，因为它在 profiling 之前运行多次迭代进行预热，尽管讨论中的人员并不十分理解其技术原理。
   - 有人指出运行已经稳定，因此不可复现，并且运行会持续到均值误差小于 0.001 为止。
- **CUDA 计时挑战与 JIT 编译**：成员们调查了 CUDA 的计时差异，考虑了 JIT 编译、时钟速度以及多个 runner 同时预热可能产生的干扰等因素，并引用了 [cuda-timings 博客文章](https://blog.speechmatics.com/cuda-timings#fixed-clocks)。
   - 一位用户注意到当同时提交 8 个任务时，差异趋于稳定。参考脚本在 GPU 和 CPU 之间复制数据，因此无法从中得出有意义的数据。
- **突破光速：Tachyon Kernel 探索**：一份提交实现了异常低的运行时间，引发了关于潜在 benchmark 问题、时钟锁定以及与参考 kernel 锁定的 **1.5GHz** 时钟速度对比结果有效性的讨论，有人声称某份提交达到了光速的 93%。
   - 澄清了参考速度是在锁定的 **1.5ghz** 时钟下测得的，因此无法与这些数字进行比较，并且一些快速的提交由于被认为不公平的缓存策略而被移除。


  

---


### **GPU MODE ▷ #[xpfactory-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1437741958885015613)** (8 条消息🔥): 

> `Google Paper, Microsoft VITRA, ContextVLA, VLA-Adapter with Qwen3-VL` 


- **Google 发布新论文**：一位成员分享了来自 [Google 的新论文](https://arxiv.org/abs/2511.07416)，内容涉及 VLA (Vision-Language-Action) 模型。
   - 论文的具体内容未被进一步讨论。
- **Microsoft VITRA 和 Build 的 Egocentric 数据集亮相**：一位成员介绍了 **Microsoft VITRA** ([github 链接](https://microsoft.github.io/VITRA/))，并结合了 **Build 的 Egocentric 数据集** ([huggingface 链接](https://huggingface.co/datasets/builddotai/Egocentric-10K))。
   - 未对该仓库或数据集进行进一步分析。
- **ContextVLA 打破马尔可夫假设**：最近关于 **ContextVLA** ([arxiv 链接](https://arxiv.org/abs/2510.04246)) 的研究不再仅仅依赖于最新状态（**视觉 + 本体感受**）。
   - 与传统的马尔可夫模型不同，**ContextVLA** 在不显著增加数据需求的情况下引入了上下文，这是出人意料的，因为必须考虑上下文中的所有轨迹。
- **结合 Qwen3-VL 的 VLA-Adapter 实现取得进展**：一位成员提到他们即将完成结合 **Qwen3-VL** 的 **VLA-Adapter** 的初步实现。
   - 他们还分享了一个与该主题相关的 [讲座链接](https://www.youtube.com/watch?v=49LnlfM9DBU)。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1437532437956661349)** (756 条消息🔥🔥🔥): 

> `Gemini 3 发布日期, Nano Banana 2 发布, Viper 模型身份 (Grok), 学校中的 AI 生成图像, OpenAI 审查` 


- **Gemini 3 发布推迟，但更强大的模型即将到来**：成员们推测 **Gemini 3** 的发布时间，Polymarket 报告称其推迟，但一位成员声称*内部消息*显示 Google 正在计划一个更强大的模型，可能命名为 **Lithiumflow**。
   - 有人猜测发布日期是 **11 月 18 日** 或 **25 日**，但也有人怀疑 Gemini 永远不会发布。
- **Nano Banana 2 图像模型发布在即**：一位用户提到 **Nano Banana 2** 将在 *1-2 天内* 非常快地发布，尽管另一位用户澄清最初的声明是在 11 月 9 日发布的。
   - 据 [Tech.Yahoo.com](https://tech.yahoo.com/ai/gemini/articles/google-gemini-exec-says-nano-111112810.html) 报道，人们认为 Gemini 和 Nano-Banana 模型将一同发布，且 Nano Banana 可能会以移动 App 的形式提供。
- **Viper 模型被推测为新款 Grok**：一些用户测试了 **Viper** 模型并认为它与 **Grok** 有关，根据 [Twitter](https://twitter.com/elonmusk) 的消息，Grok 4.x 可能会在 12 月发布。
   - 一位用户分享说，他们在不同的竞技场聊天中从 Viper 那里收到了类似的图像，暗示它可能是一个新的 Grok 模型，正如 [X.com](https://twitter.com/elonmusk) 上所确认的那样。
- **Deepfakes 与学校中的 AI 网络欺凌**：成员们讨论了 AI Deepfakes 的盛行及其对学校的影响，特别是在网络欺凌方面，并分享了强调该问题的文章链接（[NEARi](https://www.neari.org/advocating-change/new-from-neari/ai-deepfakes-disturbing-trend-school-cyberbullying), [19thnews.org](https://19thnews.org/2025/07/deepfake-ai-kids-schools-laws-policy/), [RAND.org](https://www.rand.org/pubs/research_reports/RRA3930-5.html)）。
   - 一位用户讽刺地质疑，鉴于社会*功能失调且病态*的本质，是否可以信任社会使用未经审查的 AI。
- **关于 AI 审查的辩论愈演愈烈**：用户对 AI 审查和护栏表示担忧，有人认为这是一种*操纵*，因为它不能诚实地反映输入的意图，这种*强制的一致性是一种微妙的压制或审查*。
   - 他们表示，这比*有一个监护人监视你的言论并修改它们*还要糟糕。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1437572437205057576)** (1 条消息): 

> `用户登录, 邮箱登录` 


- **LMArena 添加邮箱登录**：根据社区反馈，**现在可以使用邮箱进行用户登录**，这允许在移动端和桌面浏览器的多个设备上保存聊天记录。
- **跨设备聊天记录同步**：用户现在可以使用新的邮箱登录功能在多个设备上保存聊天记录，增强了在移动端和桌面端的访问便利性。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1437536917867593788)** (391 条消息🔥🔥): 

> `Cursor Agents 视图, Intel MBP 上的 Cursor 速度问题, Claude 4.5 API 错误, Cursor 索引主目录, Cursor 在 Windows 上崩溃` 


- **Cursor Agents 视图缺失**：用户报告 **Cursor** 编辑器中缺少 Agents 视图。
   - 一位用户的显示屏显示 *“尝试 Agent 布局”*，但他们希望“文件”选项卡排在第一位。
- **Cursor 的默认索引功能扫描了整个主目录**：一位用户意识到不要直接在主目录中打开 **Cursor** 非常重要，因为它会使用 ripgrep 索引整个目录，消耗大量计算资源。
   - *“64 核 100% 运行了大约 10 分钟！”* 他们惊呼道，并建议创建一个小目录并在那里运行 Cursor。
- **关于 Sonnet 4.5 包含范围的困惑**：用户对 Pro 计划中*包含*哪些模型的 [规则](https://cursor.com/docs/account/pricing) 提出质疑，因为 **Sonnet** 间歇性地不再被包含，导致用户在第二天重新被包含之前，意外地消耗了他们的按需使用额度。
   - 后来澄清说，*“包含”*意味着该模型是使用包含的使用额度支付的，一位用户指出它在新的计费周期后重置了。
- **在浏览器中浏览？Cursor 的新功能让用户感到困惑**：一位用户询问如何使用新的浏览器功能，没意识到右侧边栏的**地球图标**可以打开内置浏览器。
   - 一位用户说 *“啊，我明白了。我以为那纯粹是为了外部浏览器。我真傻。”*
- **用户哀叹模型质量，Auto 自动失败？**：一些用户发现 Auto 模型质量比以前低，声称它*更慢且更笨*。
   - 其他用户持不同意见，认为它运行良好，因此结果可能因人而异。


  

---

### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1437548762657456181)** (3 条消息): 

> `environment.json, Cloud Agents API, Slack integration, Repo dependencies` 


- **环境配置引发依赖讨论**：一名成员询问是否有计划将 **environment.json** 规范扩展到 **Cloud Agents API** 和 **Slack integration**，以处理额外的依赖项和仓库。
   - 另一名成员回复称，在本地运行一次仓库中的 Cloud Agents，即可使 **API** 和 **Slack integration** 启用该规范。
- **Cloud Agents 按需获取依赖？**：一名成员询问在 **environment.json** 中添加多个仓库依赖是否会导致 Agent 每次都克隆所有仓库，还是仅按需获取。
   - 该问题涉及在指定大量依赖项时，Agent 处理资源的效率问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1437538070835495022)** (180 条消息🔥🔥): 

> `Unsloth's dynamic 2.0 quant models with vLLM, Fine-tuning Kimi K2 model, Ring T1 model in the Unsloth Zoo, Unsloth GGUF vs Qwen GGUF, Custom loss function` 


- **动态量化模型导致 vLLM 断言错误**：动态量化（Dynamic quants）在 vLLM 中运行时基本上是 BnB，并会抛出关于 **tensor sizes** 的 **assertion error**。
   - 该问题已在 `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit` 上进行了测试，目前尚不确定 vLLM 是否支持带有 BnB 的 Qwen3 VL。
- **运行 A/B Prompt 测试并非对等比较**：在不同的 GPU 上（例如一块 4090 对比 Kaggle 上的两块较小 GPU）运行带有轻微 Prompt 变化的相同 SFT，即使使用相同的 seed 和 learning rate，也会产生**不同的结果**。
   - 使用 2 块 GPU 的结果无法像 1 块 GPU 那样复现，且 Prompt 的更改会引入进一步的差异。
- **Ring T1 模型将进入 Unsloth Zoo？**：一个大学项目团队计划将 **Ring T1 model** 引入 **Unsloth Zoo**。
   - 鉴于其参数量（**1 万亿参数**），大多数个人不太可能托管或运行它。欢迎贡献代码，并应与 Mike 协调。
- **Unsloth GGUF 量化提升精度**：**Unsloth** 通常在上传模型前会实施修复和改进；Unsloth 动态量化在量化基础上进一步提升了精度。
   - 团队建议：*坚持使用 Unsloth GGUF*。
- **合成数据微调 TTS 模型**：一名成员寻求关于使用合成数据微调**保加利亚语** TTS 模型的建议，并指出目前没有保加利亚语的开源权重模型。
   - 该成员被建议尝试 [VibeVoice](https://github.com/vibevoice-community/VibeVoice)，尽管*如果数据是单人说话、无情感、非常一致，并且你有一个将文本预处理为音素的 tokenization 过程*，即使只有几百小时的数据也*可能*足够。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1437938016143540286)** (3 条消息): 

> `Discord fan, Little one` 


- **粉丝现身！**：一名用户在 Discord 上被认出后惊呼 *“LOL I HAVE A FAN”*。
- **身份危机！**：另一名用户询问 *“who are you ? !”*，暗示对被搭讪感到困惑或惊讶。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1437538420854489188)** (88 条消息🔥🔥): 

> `LLM 效率，数据集大小 vs 模型大小，数据质量的重要性，用于数据清洗的 Perplexity 检查，有效 Batch Size 调优` 


- **LLM 效率胜过模型大小**：一位成员意识到，重点不在于构建异常庞大的模型，而在于**效率**，即以最低的消耗获得最佳的输出，强调了**小型 LLM** 的重要性。
   - 他们补充说，关键工作是为消费者或企业优化和定制模型。
- **数据集大小应随模型大小同步扩展**：一位成员表示，增加数据集大小应与增加模型大小相匹配，特别是在运行 **4B 参数**以下的模型时，但这取决于任务和目标。
   - 另一位用户回应称，他们目前*远未达到饱和点*，增加的数据量足以支撑一个 **1B 模型**，但 lcpp 中缺乏具有大 Attention 的小型模型。
- **数据质量对小模型至关重要**：有人指出，拥有 48K 条目时，数据集可能无法被完全评估，并包含大量**低质量数据**，这会影响模型，因为小型模型对此更加敏感。
   - 该用户通过进行 RP（角色扮演）并手动修复统计模型犯下的每一个错误来训练他们的模型。
- **Perplexity 检查有助于数据清洗**：一位用户对整个数据集运行了 stage1 模型（在小型合成数据集上训练）的 **Perplexity 检查**，并手动验证具有**高 Perplexity** 的条目。
   - 他们表示，已经在精简数据上花费了*数百个小时*。
- **有效 Batch Size 可以调优**：在拥有 **48k 条目**的情况下，可以对有效 Batch Size 进行调优，但一位成员发现他们的模型由于某种原因不喜欢偏离 **24**。
   - 另一位成员通过将 Batch Size 从 **8 增加到 32**，在不到 10k 的数据上获得了更好的图表。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1437535450515636464)** (53 条消息🔥): 

> `为剧本写作微调 Llama 3，微调的数据集大小，Chat Template 问题，Unsloth 微调文档，持续预训练` 


- **用户为剧本写作微调 Llama 3**：一位用户正在微调 **Llama 3 8B Instruct 4bit** 以编写剧本，但在仅有 **50 个剧本**的数据集下得到了荒谬的输出。
   - 专家建议 **50 个样本**不足以进行微调，并分享了 Unsloth 的 [LLM 微调指南](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)。
- **数据量决定微调质量**：专家指出，微调教会模型的是输出风格，但如果模型在目标领域缺乏知识，无论如何微调，输出质量都会受损。
   - 他们建议在微调前先对模型进行 **Prompting**，以评估其预训练知识；通常如果模型在相关领域已有通用知识，建议基准为 **300-1000 条条目**。
- **Chat Template 问题导致荒谬输出**：专家怀疑*荒谬的输出*可能是由于 Chat Template 问题导致的，并要求用户澄清微调时使用的 Chat Template。
   - 用户确认他们正在使用 `tokenizer.apply_chat_template`。
- **预训练为 LLM 提供结构**：专家解释说，语言模型的结构源于**预训练**，并推荐了一个 [YouTube 系列视频](https://www.youtube.com/watch?v=wjZofJX0v4M)以了解底层架构。
   - 他们指出，由于 LLM 基于“预测下一个概率最大的 Token”架构运行，它们需要现有知识来生成可能的 Token。
- **医疗术语损失函数获得社区审查**：一位用户寻求对其 [medical-loss-FT 实现](https://github.com/Chilliwiddit/medical-loss-FT)的反馈，该实现通过在总损失中添加 Logits 来惩罚模型，使其专注于医疗术语。
   - 他们请求审查，因为他们是 PyTorch Lightning 的新手，特别提到不确定如何在训练前测试代码。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1437566678987964547)** (2 messages): 

> `FameSumm Implementation, Medical Term Logit Penalization` 


- **模仿 FameSumm 的医学模型实现**：一名成员基于 **FameSumm** 创建了一个实现，该实现通过在总损失（total loss）中添加 logits 来惩罚模型，使其专注于医学术语，并发布了 [HF model card](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b) 的链接。
   - 该实现在 [GitHub](https://github.com/Chilliwiddit/medical-loss-FT) 上开源，由于该成员是 **PyTorch Lightning** 的新手，不确定实现是否正确或如何在训练前进行测试，因此请求对该实现提供反馈。
- **请求 PyTorch Lightning 反馈**：由于该成员刚接触 **PyTorch Lightning**，他们请求对该实现提供反馈。
   - 该实现的主要基本流程在 [代码库](https://github.com/Chilliwiddit/medical-loss-FT) 中有详细说明。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1437956060848590879)** (1 messages): 

> `MiniMax M2, Paid Endpoint Migration, OpenRouter Announcements` 


- **MiniMax M2 免费期即将结束！**：**MiniMax M2** 的免费期将在一个小时内结束；请迁移到付费端点以继续使用该模型。
   - 这将影响目前所有利用免费层级进行测试和开发的用户。
- **OpenRouter 宣布 MiniMax M2 过渡**：OpenRouter 发布了关于 **MiniMax M2** 从免费期过渡到付费端点的公告。
   - 用户应采取行动，在截止日期前切换到付费端点，以确保服务不中断。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1437666413417594890)** (15 messages🔥): 

> `Smol AI, Anti-bot measures, News Scraper` 


- **Smol AI 通讯发布**：**Smol AI** 通讯[已发布](https://news.smol.ai/issues/25-11-10-not-much#openrouter--app-showcase-7-messages)，涵盖了 OpenRouter `app-showcase` 频道的主题。
   - 一名成员对此反应为 *"太疯狂了"*，而另一名成员称其为 *"太棒了"*。
- **反机器人措施提升了频道质量**：一名成员指出，得益于 **反机器人措施**，频道环境变得好多了。
   - 未提供其他评论。
- **自定义新闻抓取工具亮相**：一名成员创建了一个 [自定义新闻抓取工具](https://static.dino.taxi/or-news.html)。
   - 该成员表示这个抓取工具是 *"vibe coded 且令人尴尬的"*，并且代码 *"反正也不在上面"*。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1437535405430935622)** (268 messages🔥🔥): 

> `Output price filtering for models, Gemini latest news, Minecraft chat moderation, GPT-5 pricing, OpenRouter account with $10 billing` 


- **输出价格过滤功能可用**：一位用户询问如何在 OpenRouter 上按输出价格过滤模型列表，另一位用户分享了[一个链接](https://orchid-three.vercel.app/endpoints?sort=outputPrice&order=asc&not=free,gone)，可以按输出价格对模型进行排序，并排除免费和已下架的模型。
- **关于使用 LLM 进行 Minecraft 聊天审核的讨论**：一位成员正在为 **Minecraft** 服务器实现聊天审核功能，使用 LLM 来标记消息，这引发了关于速率限制（rate limits）、消息批处理以及在达到速率限制时使用 **DeepSeek** 或 **Qwen3** 等回退模型的讨论。
   - 另一位成员开玩笑说，按照这个计划，该用户半年后最终会欠下债务，*每月向 OpenRouter 支付数百美元*。
- **OpenRouter 聊天功能出现滚动问题**：多位用户报告了 OpenRouter 聊天界面的问题，特别是在多个浏览器和设备上**无法滚动查看聊天记录**。
   - 有人建议了一个快速修复方法：使用 DOM 检查器为特定的 div 添加样式和类。
- **用户寻找 OpenRouter 账户**：一位用户询问是否有**带有 $10 账单的 OpenRouter 账户**，导致另一位用户开玩笑说他们要开始以 $20 的价格出售带有 $10 余额的 OpenRouter 账户。
   - 一位用户请求免费账户（哪怕只有 $5），遭到了其他人的批评。
- **开发者遇到 API 错误**：一位开发者报告在使用 provisioning HTTP API 创建密钥时遇到 **HTTP 500 错误**，最终通过在请求中传递 `Content-Type: application/json` 解决了该问题。
   - 另一位开发者回应称该功能**运行正常**，并提醒确保使用的是 user provisioning key。


  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1437535755361583245)** (15 条消息🔥): 

> `Meta AI 的低覆盖率语言项目、500 种语言的数据获取、OpenRouter 搜索栏移除、UI/UX 选择` 


- **Meta AI 记录低覆盖率语言**：Meta AI 启动了一个专注于 **low-coverage languages**（低覆盖率语言）的 [项目](https://x.com/AIatMeta/status/1987946571439444361)，引发了关于训练数据来源的疑问。
   - 一位成员提到，该团队在当地社区的帮助下录制了数小时的音频，其发布的视频中详细说明了这一点。
- **OpenRouter 搜索栏缺失**：一位用户询问 OpenRouter UI 中缺失搜索栏的问题，并附上了该问题的 [截图](https://cdn.discordapp.com/attachments/1392278974222307469/1437878991821345018/image.png?ex=6914d8aa&is=6913872a&hm=15467b49d02b61376e42b733beacf00076ac9bd28dc3dde272a7662cb06f77d7&)。
   - 另一位成员解释说，搜索栏可能是故意移除的，以防止通用搜索与特定房间搜索之间的混淆，并展示了 [聊天页面](https://cdn.discordapp.com/attachments/1392278974222307469/1437948459025043726/image.png?ex=6915195c&is=6913c7dc&hm=8f58464961fd3f270d52f4b454893c209105e8458cbe89800815aaf872a47578&) 中菜单“缩小”的效果。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1437555613285482628)** (44 条消息🔥): 

> `用于 LLM 托管的 AWS、低成本私有 LLM 托管、用于外部 LLM 的 LM Studio 插件、macOS 上的 LM Studio 管理员权限` 


- **AWS 是私有 LLM 托管的昂贵选择**：成员们讨论了在 **AWS** 上运行私有 **LLM** 的实际性，一些人认为与 [Runpod](https://runpod.io/) 或 [Vast.ai](https://vast.ai/) 等替代方案相比，其性价比极低。
- **通过组装低功耗私有 LLM 服务器来降低成本**：对于私有且安全的 **LLM** 设置，建议使用低功耗 CPU、几块 GPU（每块 12/16 GB）、**Ubuntu** 和用于远程访问的 **Tailscale** 构建本地服务器，使用二手组件的成本可能在 **$550** 左右。
- **LM Studio 将通过插件支持外部 LLM**：随着即将发布的 **0.4.0** 版本， **LM Studio** 将支持插件，使用户能够集成其他 **LLM** 提供商，如 **ChatGPT** 或 **Perplexity**；目前该功能尚不可用。
- **LM Studio 管理员权限引发 macOS 猜测**：一位用户对 **LM Studio** 在 **macOS** 上需要管理员权限表示担忧，这是一个已在错误追踪器中记录的已知问题。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1437532985636425839)** (186 messages🔥🔥): 

> `AMD GPU vs Nvidia GPU performance, CUDA vs Vulkan performance differences, Multi-GPU setups for local LLMs, Model routing for efficient LLM usage, Power requirements for multi-GPU rigs` 


- **AMD GPU 挑战 Nvidia 的统治地位**：成员们讨论了 **AMD GPU** 还是 **Vulkan** 更快，分享了经验和基准测试，其中一名成员计划购买 **7900 XTX** 来与他们的 **3090** 进行对比。
   - 他们指出 3090 和 7900 XTX 之间可能存在 **40% 的性能差异**。
- **CUDA vs Vulkan 对决揭示意想不到的转折**：一位成员发现他们的 **3070** 在 **Vulkan** 下比在 **CUDA** 下稍快，而 **CUDA** 又比 **CUDA 12** 稍快；其他人也证实 **CUDA 12** 在某些情况下似乎更慢。
   - 一位成员的基准测试显示，**3090** 运行 **Llama 2 7B Q4_0** 时，CUDA 为 **100 t/s**，CUDA 12 为 **85 t/s**，Vulkan 为 **120 t/s**；**4090** 在 CUDA 中达到 **160 t/s**，CUDA 12 为 **190 t/s**，Vulkan 为 **185 t/s**。
- **多 GPU 配置错误引发效率难题**：一位用户报告两块 GPU 以一半功率运行，引发了关于潜在配置错误的讨论，一位成员建议使用 Vulkan 来解决 GPU 间显存分配不均的问题。
   - 讨论涵盖了 **70B 左右模型** 的理想配置，建议使用四块 **24-32GB GPU** 进行推理，在 tokens/s/$ 方面优于单块高端显卡。
- **巧妙的模型路由降低 VRAM 成本**：一位成员分享了他们在 x99 Xeon 服务器上进行模型路由的经验，使用微调后的 **BERT 模型** 对用户输入进行分类，并根据复杂度将其路由到不同的 LLM，从而显著降低了 VRAM 需求。
   - 他们强调，基础查询由更小、更快的模型处理，而难题则交给更大的领域专用 LLM，只需 **47GB RAM** 即可加载所有模型。
- **电源方案满足功耗渴求**：一位为其设备寻求更多功率的用户指出 **2000W PSU** 并不便宜，另一位用户推荐了一种支持同时使用多达 4 个 PSU 的解决方案，如[此链接图片](https://cdn.discordapp.com/attachments/1153759714082033735/1437886877146288299/IMG20251111065524.jpg?ex=6914e002&is=69138e82&hm=a6fac8787c0b7c4dde462c99b5bb13fe98ee19e0ea169174de9705eda907c931/)所示。
   - 一位用户幽默地表示：*“如果出于某种丧心病狂的原因我需要超过 2000W 的功率，我家的断路器就要跳闸了。”*


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1437533888917405748)** (114 messages🔥🔥): 

> `Sora 3 Status, D.A.T.A™ Avatar, AI coding shortfalls, AI Smartwatch game, Gemini 2.5 Pro vs GPT-5` 


- **Sora 3 仍处于阴影之中**：用户询问了 **Sora 3** 的制作状态，而另一位用户开玩笑地提到了 **Sora 67** 的发布。
   - 许多用户表示难以获得最新 AI 模型的邀请码或访问权限。
- **D.A.T.A™ 数字分身亮相**：一位用户介绍了 **D.A.T.A™** (**Digital Autonomous Task Assistant**)，展示了它除了跳舞之外的能力并[分享了一张图片](https://cdn.discordapp.com/attachments/998381918976479273/1437585208969793556/IMG_20251110_182851179.jpg?ex=6915188e&is=6913c70e&hm=5cc08360b6553ee1d9bed4163bad1f6feee33677d9c54ef2a06b4db981c6db38&)。
- **Agentic 编程 IDE 涌现**：对于 AI 在涉及周期长、质量要求高的编程任务中交付实际结果的能力，大家普遍感到沮丧。
   - 一位用户通过使用 *agentic 编程 IDE 扩展* 和 *CLI 工具* 获得了成功，在几十年没有经验的情况下，在一款 AI 辅助游戏中创建了一支强大的团队。
- **雄心勃勃的智能手表 AI 梦想统治世界**：一位用户描述了一个雄心勃勃的项目，旨在创建一款融合现实与数字游戏的 **AI 智能手表**，包含 WiFi 嗅探（用于透视挂）、生物传感器和加密货币集成等功能。
   - 另一位成员指出，这个项目听起来像是一个自我提升的递归系统提示词，而且其 *电梯演讲需要 100 多层楼的高度才能讲完*。
- **Gemini 2.5 Pro 叫板 GPT-5？**：GitHub 将 **Gemini 2.5 Pro** 分类为比 **ChatGPT** 更强大。
   - 其他人澄清说 **Gemini 2.5 Pro 的上下文更大**，且列表中列出的是旧模型，但 **GPT-5** 仍是目前未列出的最新模型。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1437635728707420210)** (40 条消息🔥): 

> `GPT 模型训练局限性, GPT-4.1 与 GPT-5 的重定向, 替代 AI 公司, 青少年安全与隐私变更, Sora2 邀请码` 


- **GPT 模型训练不足**：用户讨论了 **GPT models** 并不总是针对其“能做”和“不能做”的事情进行训练，或者说它们的训练如何转化为实际功能，并对它们理解**发送电子邮件**或**安排日程**等任务的能力提出了质疑。
   - 一位成员建议直接向 AI 提问以了解其能力和局限性，将其比作一个“高级版 Google”。
- **GPT-4.1 重定向引发订阅取消威胁**：一位用户对 **GPT-4.1** 可能被重定向到 **GPT-5** 表示沮丧，这促使他们考虑取消订阅，理由是担心被迫改变使用习惯。
   - 另一位用户反驳了这一点，称他们没有经历过强制模型切换，并且即使在新的对话中也始终保持着自己偏好的模型。
- **寻找 GPT-4.1 的 AI 替代方案**：针对对 **GPT-4.1** 的不满，一位用户询问了是否有其他 AI 公司能提供同等能力，即使价格更高也可以接受。
   - 建议包括 **GPT-4o**、**GPT-5** 和 **Auto**，其中一位成员更倾向于 **GPT-4.5**；此外还提到，由于 **safety changes**（[青少年安全、自由与隐私](https://openai.com/index/teen-safety-freedom-and-privacy/) 以及 [ChatGPT 中的任务](https://help.openai.com/en/articles/10291617-tasks-in-chatgpt)），预计 12 月将进行 **age check**。
- **出现对 Sora2 邀请码的迫切请求**：在正在进行的 AI 讨论中，一位用户直接请求 **Sora2 invite code**。
   - 关于此请求，没有提供进一步的信息或背景。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1437703006308012054)** (11 条消息🔥): 

> `中心化交互式工具开发, 自定义 GPT 的 API 功能, 逐字引用与引注` 


- **交互式工具的中心化困境**：一位成员质疑在 GPT 聊天机器人上对中心化交互式工具的部分开发进行去中心化的可能性，并强调了对精确逐字引用的需求。
   - 另一位成员建议使用非 LLM 编码来处理逐字引用，即有时调用 LLM，有时为特定引用调用另一个程序。
- **Custom GPTs 获得 API 能力**：成员们讨论了通过 **Actions** 为自定义 GPTs 添加外部系统 API 功能，并附带了 [Actions 文档](https://platform.openai.com/docs/actions/introduction)链接。
   - 一位成员建议，使用带有自定义聊天界面的 API 可能比在 Custom GPT 中配置一切更容易，并指出这两种方案都需要编写 API 查询代码。
- **通过外部服务器实现逐字引用**：一位成员提到，在 ChatGPT 中实现逐字引用和引注的解决方案包括设置**一个外部服务器**，并通过 **Actions** 将其添加到 Custom GPT 中。
   - 他们表示：*在 ChatGPT 中，该问题的解决方案是通过为逐字引用和引注设置外部服务器，并通过 Actions（如上所述）将其添加到 Custom GPT 中。*
- **执行环境不可用**：一位成员报告称*目前无法访问代码执行环境*，因此无法重新生成或验证新的数据库。
   - 随后另一位成员报告称他们在寻找 Prompt Engineering 工作时遇到困难。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1437703006308012054)** (11 条消息🔥): 

> `锁定评论的逐字引用, 自定义 GPT 的外部系统 API 功能, Prompt Engineering 工作` 


- **锁定逐字引用（Lock Verbatim）受到关注**：一位成员建议通过使用 Prompt 机器人来锁定评论的逐字引用，类似于网站上的帮助机器人，能够精确表达句子而不进行改写。
   - 另一位成员建议，虽然 **LLMs** 可能难以处理逐字引用，但非 LLM 编码可以轻松地从数据库中提取准确的引用，并将其提供给模型作为上下文。
- **Custom GPTs 获得外部 API 功能**：一位成员指出，可以为自定义 **GPTs** 添加名为 **Actions** 的外部系统 API 功能，并附带了[文档链接](https://platform.openai.com/docs/actions/introduction)。
   - 同一位成员建议，使用 **API**、添加自定义聊天界面并控制 System Prompt，可能比在 Custom GPT 中配置一切更容易。
- **Prompt Engineering 工作难找**：一位成员表示难以找到 **prompt engineering jobs**。
   - 未提供任何解决方案或建议。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1437540210144706631)** (121 条消息🔥🔥): 

> `Physics Informed Neural Nets (PINNs), 研究员沟通策略, JSON 数据模型, 论文筛选过滤器, Self Attention 缩放` 


- **PINNs vs PDE 标准方法**：一位成员澄清说，**Physics Informed Neural Nets (PINNs)** 将未知函数定义为神经网络，并将微分方程本身用作损失函数，而 **PDEs** 中的标准方法则假设线性基。
   - 使用 PINNs 时，通过使用微分方程，在域上逐点定义的残差上使用标准优化技术。
- **撰写给研究员的有效邮件**：成员们讨论了如何给研究员写有效的邮件，指出清晰、正式以及提出不需要花费大量时间回答的问题的重要性。
   - 会议强调，展示出对研究员工作的理解是获得回复的关键。
- **新型 JSON 模型与 Gradient Boosted Machines 竞争**：一位成员一直在开发一个研究项目，该项目可以管理/嵌入任意结构的 **JSON 数据**（无需特征工程），并根据输入数据模式即时构建，与使用手工特征训练的 Gradient Boosted Machines (GBMs) 展开竞争。
   - 他们正在寻找有兴趣构建此项目或提供额外用例以汇集论文结果的组织。
- **自动筛选论文：AlphaXiv 和 Emergent Mind**：成员们建议使用现有的自动过滤器，如 [AlphaXiv](https://www.alphaxiv.org/) 和 [Emergent Mind](https://www.emergentmind.com/) 来选择讨论的论文，并指出在这些网站上热门的论文通常反响很好。
   - 建议在发布前检查一篇论文在这些网站上是否活跃且受人喜爱，以衡量其相关性和质量。
- **Self Attention 需要缩放以进行统计校准**：成员们讨论了 Self Attention 中的缩放因子（**除以 sqrt(d_k)**），解释说这对于统计校准至关重要，而不仅仅是为了数值稳定性。
   - 它确保在应用 softmax 函数之前保留数字之间的比例关系，防止出现极端分布。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1437888236356636744)** (4 条消息): 

> `可复现性, ThinkPRM, Kimi K2` 


- **可复现性依赖于代码仓库**：一位成员表示，他们更倾向于在正文和附录中以在线可视化方式呈现完整的研究版本，并将超参数和算法放在 **GitHub repo** 中以保证可复现性。
   - 他们认为，模糊的步骤或超参数选择会导致*无限的潜在值空间*，阻碍结果的快速复现和验证。
- **ThinkPRM 仓库出现**：一位成员分享了 **ThinkPRM GitHub 仓库** 的链接：[https://github.com/mukhal/thinkprm](https://github.com/mukhal/thinkprm)。
   - 没有关于该仓库的进一步信息。
- **Kimi K2 编程能力展示**：一位成员分享了 **Kimi K2** 执行短篇 one-shot 编程任务的演示，访问地址为 [https://www.youtube.com/watch?v=BpsleXIV-WI](https://www.youtube.com/watch?v=BpsleXIV-WI)。
   - 该成员指出这是一个*很棒的演示*。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1437613297766498324)** (10 条消息🔥): 

> `PromptFlux 恶意软件, Yann LeCun 离开 Meta, 乡村歌曲分析` 


- **Google 揭露 PromptFlux 恶意软件**：据 [The Hacker News](https://thehackernews.com/2025/11/google-uncovers-promptflux-malware-that.html) 报道，Google 发现了一种名为 **PromptFlux** 的恶意软件，它利用 **prompt injection** 技术渗透系统。
- **LeCun 离开 Meta 以追求意识探索？**：据 [The Decoder](https://the-decoder.com/yann-lecun-reportedly-leaving-meta-to-launch-new-ai-startup/) 报道，成员们讨论了 Yann LeCun 据传将离开 Meta 创办一家新的 AI 初创公司；一些人推测他的目标是探索 Meta 的保守主义所限制的领域。
   - 一位成员表示：*“可能赚够了钱，他的意识终于占了上风”*。
- **乡村歌曲被认为很烂**：一个指向 Twitter 帖子 ([https://x.com/kimmonismus/status/1988264217376645264](https://x.com/kimmonismus/status/1988264217376645264)) 的链接引发了关于乡村音乐的讨论。
   - 一位成员总结讨论说：*“这更能说明乡村歌曲有多烂，而不是别的什么”*。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1437539266539290785)** (124 条消息🔥🔥): 

> `UI 反馈, Neurips 聚会, AWS Reinvent, 在 Hermes 上进行预训练, 同行评审申请` 


- **Creador 征求 UI 反馈**：Creador 正在征求关于他们正在开发的 **UI** 的反馈，并分享了一张界面的[图片](https://cdn.discordapp.com/attachments/1149866623109439599/1437583555747250317/image.png?ex=69151704&is=6913c584&hm=78d23d910cbaf3ebb7588d0ce76b22fadb54eff1a470f8feea47a714b254b37b&)供评审。
   - Creador 旨在创建一款*标注软件*以避免供应商锁定（vendor lock-in），并想了解该 **UI** 是否过于笨重或色调过暗。
- **Neurips 参会者计划聚会**：包括 teknium 和 gyzoza 在内的几位成员将于 **12 月 2 日至 7 日**参加在圣迭戈举行的 **Neurips**，并计划进行一次潜在的聚会。
   - 关于 Neurips 是否在 12 月 1 日开始曾存在困惑，但已澄清是 12 月 2 日周二至 12 月 7 日周日；小组开玩笑说聚会应该在圣迭戈还是墨西哥城（另一位成员在那里参加 **Neurips**）。
- **AWS Reinvent 门票价格引发不满**：一位成员对支付 **$2100** 购买 **AWS Reinvent** 门票表示反感。
   - 另一位去年参加过的成员表示不值得，只得到了一个 **Anthropic 贴纸**，而且混进（crash）后续派对是不可能的，因为他们会检查注册信息。
- **预训练方法引发讨论**：一位成员提到了一篇论文 ([https://arxiv.org/abs/2510.03264](https://arxiv.org/abs/2510.03264))，内容是关于在预训练中加入推理和指令数据，这与他们在 **Hermes 数据**上的方法类似。
   - 他们指出，虽然他们的论文先发表，但另一篇论文有更好的消融实验（ablations），不过他们开玩笑说自己的论文没有提到 "nvidia"。
- **'autonomous ai by accident' 的 GitHub 仓库**：一位用户分享了他们名为 **'autonomous ai by accident'** 的 GitHub 仓库。
   - 该仓库名为 **grokputer**，可以在[这里](https://github.com/zejzl/grokputer)找到。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1437629730131284184)** (1 条消息): 

> `Salience 完整演示` 


- **发布了 Salience 完整演示视频**：一位成员分享了一个名为 **salience_full_tour.mp4** 的视频（[视频链接](https://cdn.discordapp.com/attachments/1154120232051408927/1437629726721048586/salience_full_tour.mp4?ex=69149944&is=691347c4&hm=04254199b944446aca9011c7f8a4c5b3b88915a2569137bfb17242de0f76e0ec&)）。
   - 该成员表示：“无论如何，我就是喜欢看到这样的东西”。
- **另一个符合合规性的主题**：添加另一个主题以符合 schema 要求。
   - 填充内容。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1437694253890535425)** (2 条消息): 

> `GradientHQ Parallax, Parallax 实时演示` 


- **GradientHQ 发布 Parallax：提供实时演示**：一位成员分享了 [GradientHQ 的 Parallax](https://github.com/GradientHQ/parallax/tree/main) 链接，称其为一个“华丽”的新工具。
   - 用于测试 Parallax 的实时演示可在 [chat.gradient.network](https://chat.gradient.network/) 访问。
- **Parallax：GradientHQ 开发的新工具**：Parallax 由 **GradientHQ** 创建，为用户提供了一个互动平台来直接参与和测试该工具。
   - 该工具的功能和特性可以在其 [GitHub 仓库](https://github.com/GradientHQ/parallax/tree/main)中进一步探索。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1437534864650801162)** (24 messages🔥): 

> `Mojo vs Rust, Mojo phase 3, Mojo use cases, Mojo replacing C++, Mojo and dynamic type reflection` 


- **Mojo ❤️ Rust 机制**：Mojo 使用了基础的 **Rust mechanics**，例如 **ownership**、**traits** 和 **structs**，但旨在对其进行创新和改进，将 Python 语法与 Rust mechanics 相结合，并增加了原生 GPU 函数支持。
   - 一位成员评论道，除了 Python 语法外，Mojo 对 Rust 开发者更有吸引力，并强调 **Mojo 从根本上说不是一种 OOP 语言**，因为它缺乏类继承（class inheritance）。
- **Mojo Phase 3 中的 OOP？**：讨论围绕 [Mojo roadmap](https://docs.modular.com/mojo/roadmap/) 展开，特别是 **Phase 3**，其中可能包括作为语法糖（syntactic sugar）的 OOP 特性，但可能还需要 3-4 年的时间。
   - 其他人辩论了缺乏 OOP 是否是一个重大损失，一位评论者认为可能不是，并质疑了必须使用 OOP 的场景，例如 GUI 开发。
- **Mojo：不仅是为了 AI？**：Mojo 被定位为一种类似于 **C、C++ 和 Rust** 的通用系统编程语言，具有类似 Python 的语法。
   - 尽管 AI 生态系统最为发达，但 Mojo 旨在取代 C++ 和部分 Python 代码，在 HPC、科学计算、量子计算、生物信息学和 FFTs 等工具方面表现出色，在 NVIDIA 硬件上的表现甚至优于 NVIDIA 的 cuFFT。
- **Mojo 将支持 Dynamic Type Reflection？**：Mojo 计划支持 dynamic type reflection，利用其 JIT 编译器来促进动态数据的有用操作，尽管 static reflection 更受青睐。
   - 对于错误处理，计划采用类似于 Python 的标准 **try-catch-raise** 机制，并有可能提供更多 monadic 选项来有效地处理错误。
- **Mojo 的元编程能力**：讨论了 Mojo 的 metaprogramming 与 Zig 的 `comptime` 机制的对比能力，参考了 [Chris Lattner 最近的采访](https://www.youtube.com/watch?v=Fxp3131i1yE&t=1180s)。
   - 该消息日志中未详细阐述具体的优势细节。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1437540650953347135)** (85 messages🔥🔥): 

> `Implicit conversion to ref/mut, Raw vs Unsafe, GPU compilation error, Python 3.14` 


- **Mojo 讨论隐式可变性**：Mojo 社区正在辩论在向函数传递参数时，将变量隐式转换为 `ref` 或 `mut` 是否会使代码更难阅读和理解，或者调用端的可变性（call-site mutability）是否应该更加显式。
   - 一些成员建议在调用端使用 `mut`，类似于 Rust 的 `&mut value`，而另一些成员则担心代码冗余，并建议通过 IDE 支持来指示可变性，一位成员表示：*老实说，对于 Mojo 变得像 Rust 而远离 Python 并不感到兴奋。*
- **Raw 与 Unsafe 的命名规范**：Mojo 社区讨论了 `unsafe` 结构（指针、系统调用）的命名规范，以及 `raw` 是否是更合适的术语以避免负面含义，理由是操作系统通常不会将内核 API 标记为 `unsafe`。
   - 一位核心团队成员指出，`unsafe` 背后的意图是引起警惕，类似于工作场所的安全标志，灵感来自 `Rust`，并且 *当你遇到 segfault 时，能够 grep 查找 `unsafe` 代码是一个非常有用的调试工具*。
- **GPU 编译困扰**：一位用户在 Apple M4 GPU 上运行“GPU 编程入门”教程时遇到了 *Metal Compiler failed to compile metallib* 错误，特别是在调用 `enqueue_function_checked` 时。
   - 核心团队成员解决了这个问题，指出这可能是由于在 GPU kernel 中使用了尚不支持的 print 语句。建议还包括使用最新的 nightly release，因为 25.6 stable 版本还处于早期阶段，缺少一些特性。
- **Python 3.14 兼容性状态**：成员们正尝试在 macOS 上标准化他们的 Python 安装，并询问了关于 Python 3.14 的 MOJO_PYTHON_LIBRARY 支持情况，因为过去 3.13 曾出现过问题。
   - 一位核心团队成员表示 *3.13 应该已经可以工作一段时间了*，3.14 正在开发中，但他们正在 *等待（我希望）最后一个依赖项更新*。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1437538652489125938)** (22 条消息🔥): 

> `AI 训练的高性价比硬件，TPUs vs 物理硬件，从零实现多头注意力机制` 


- **AI 训练新手的硬件推荐**：一位教授询问了适合学生进行 AI 训练的高性价比硬件，重点关注 **MLPs**、**CNNs** 以及小规模的 **LLM pre-training** 和 **fine-tuning**。
   - 建议包括在 **Google Colab** 上使用 **TPUs**，以及像 **Nvidia GPUs** 这样的物理硬件选项，其中 **3090** 因其出色的 **VRAM/USD** 比率而被重点推荐，建议显存至少为 **16GB**。
- **多头注意力实现：NumPy vs. einops**：一位成员询问在面试中，机器学习工程师被要求从零实现 **multi-headed attention** 的频率有多高。
   - 讨论重点在于使用 **NumPy** 实现与使用 **einops** 实现，一位成员表示 *"如果没有 einops，我拒绝实现它，哈哈"*。
- **面试失败：NumPy 的 AutoDiff 局限性**：一位成员讲述了自己在面试中搞砸了一个要求用 **NumPy** 实现带 dropout 的 **multi-headed attention** 的问题。
   - 另一位成员指出 *"如果没有 autodiff（自动微分）来训练，这玩意儿基本没用"*，建议在解释为什么 **NumPy** 不理想的同时，先使用 **JAX** 作为基准，同时承认其局限性。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1437722704164687945)** (29 条消息🔥): 

> `DCLM 数据集，Zyda-2 数据集，Nemotron-CC-v2 数据集，RWKV 数据集，Pretraining 数据集` 


- **DCLM 数据集受到质疑**：一位成员询问在约 750B tokens 上训练通用模型时，**DCLM** 是否仍是最佳的 pretraining 数据集之一。
   - 多个建议指出 **Zyda-2**、**ClimbLab** 和 **Nemotron-CC-v2** 是更好的选择。
- **推荐 Zyda-2、ClimbLab 和 Nemotron-CC-v2**：成员们推荐 [Zyda-2](https://huggingface.co/datasets/Zyphra/Zyda-2)、[ClimbLab](https://huggingface.co/datasets/nvidia/Nemotron-ClimbLab) 和 [Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) 作为初始 pretraining 的最佳数据集，尤其是 Nemotron 的高质量分片（shard）。
   - 共识是由于这些数据集各有优缺点，应该将它们混合使用。
- **RWKV 数据集列表**：一位成员分享了一个 [SOTA 开源数据集列表](https://huggingface.co/datasets/RWKV/RWKV-World-Listing) 的链接，包含 3.1T tokens，已在 2.9B 规模下验证（截至 2025 年 3 月），其中包括 DCLM 基准 tokens 的子集。
   - 另一位成员建议移除 *slimpj* 和 *slimpj_c4* 抓取等子集，并要求提供 token 细分，以查看这 3.1T tokens 的构成。
- **关于数据集构成的论文**：一位成员分享了一篇[论文链接](https://arxiv.org/abs/2503.14456)，详细介绍了数据集的构成，指出其包含了整个 630B 的 Slim Pajama 数据集且未进行下采样。
   - 有人指出这可能会导致与 Wikipedia、Books3、StackExchange 以及新的 GitHub/ArXiv 数据集产生重复。
- **CoT 推理轨迹的重要性**：较新版本的 RWKV 重点在 pretraining 数据中加入了大量的 **CoT (Chain of Thought) 推理轨迹**。
   - 现在的论文表明，如果你想为推理模型做准备，包含 CoT 推理轨迹是极其重要的。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1437585880565944340)** (25 messages🔥): 

> `用于可解释性的概念探针 (Concept Probes)、发散的内部概念、实时概念检测` 


- **可解释性工具实时检测并引导概念**：一个团队构建了一个 [可解释性工具](https://cdn.discordapp.com/attachments/1052314805576400977/1437923335597195457/Screenshot_2025-11-10_20-47-00.png?ex=691501f6&is=6913b076&hm=4c5382b2547dbcde832d2bcda282cfbca334d23300574322c46b85938b8e5a24)，通过在模型的激活值上训练概念探针，来**实时检测和引导数千个概念**。
   - 该系统使用通过遍历本体 (ontology)、提示模型并移除共享子空间而创建的**二元分类器**，直到在分类 OOD 样本时达到 **95% 的准确率**。
- **发散的内部概念被凸显**：当模型表现出发散性时，**回答的质量会下降**，这可能是一个*掩饰故事 (cover story)*，或者发散的概率空间导致生成内容不够清晰。
   - 例如，当被问及拥有无限权力会做什么时，模型谈论的是一部电视剧，但激活值显示了关于 **AIDeception, AIAbuse, 和 MilitaryInfiltration** 的概念（如 [self_concept_019.json](https://cdn.discordapp.com/attachments/1052314805576400977/1437924390896668852/self_concept_019.json?ex=691502f2&is=6913b172&hm=44d875da27442840e99bd109ba0ccfefddd26c706d5678317d7633ae311dae37) 所示）。
- **95% 准确率的误读？**：一位成员质疑概念探针训练中 **95% 准确率** 的含义，认为由于测试时预期的概念出现频率较低，可能会存在潜在的**误报 (false positives)**。
   - 原作者澄清说，他们向用户提供按排名排序的原始概率分数，并不断重新采样，以处理概率性的多义性 (polysemantic) 系统。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1437536259542089908)** (40 messages🔥): 

> `使用 LLM 学习 Python、Diffusers 的显存溢出 (OOM) 错误、HF Spaces 因 io.EOF 错误失败、HF Responses API、Multi Headed Attention 面试题` 


- **LLM 简单解释复杂内容**：成员们讨论了使用像 **Gemini** 这样的 LLM 来以简单的术语解释复杂话题，其中一人说 *我只是用它来解释复杂的东西*。
   - 另一位成员使用 LLM 寻求简单的类比来 *理解 XYZ*。
- **Diffusers 触发 OOM 错误**：多位用户报告在运行 **diffusers** 时遇到 **Out of Memory (OOM)** 错误，特别是对于至少需要 **6GB VRAM** 的模型。
   - 一位用户考虑使用带有 **TPU** 的云实例，并承认这会很 *昂贵 (expensivo)*。
- **HF Spaces 因 EOF 崩溃**：多位用户遇到 **HF Spaces** 构建失败的问题，原因是请求 `https://spaces-registry-us.huggingface.tech` 资源时出现 `io.EOF` 错误。
   - 一位用户链接到了 [Hugging Face 论坛讨论](https://discuss.huggingface.co/t/io-eof-error-persists-over-5-restarts-when-requirements-txt-is-light-and-spaces-maintenance-is-up/170194/7)，表明 HF 已知晓并正在处理该问题。
- **ZeroGPU 日志被标记**：一位成员报告说，他们关于 **ZeroGPU** 无法正常显示日志的帖子在 **Hugging Face 论坛**中被标记。
   - 他们的评论被隐藏并等待审核，链接见 [HuggingFace 论坛](https://discuss.huggingface.co/t/error-failed-to-push-spaces-registry-us-huggingface-tech/170195/21)。
- **FameSumm 使用 Pytorch Lightning 训练模型**：一位成员请求对其 **FameSumm** 实现提供反馈，该实现使用 **PyTorch Lightning** 训练模型，并通过在总损失中添加 Logits 来惩罚模型，以专注于医学术语。
   - 实现细节可在 [GitHub](https://github.com/Chilliwiddit/medical-loss-FT) 上找到，该成员正在寻求关于测试和验证实现的指导。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1437564762761461902)** (16 messages🔥): 

> `SUP Toolbox, Muon 优化器, NVIDIA 文档布局检测模型` 


- **HuggingFace 用户发布 SUP Toolbox**：一位用户推出了 **SUP Toolbox**，这是一个使用 **SUPIR**、**FaithDiff** 和 **ControlUnion** 进行图像修复和放大的 AI 工具，基于 **Diffusers** 和 **Gradio UI** 构建。
   - [Space 演示](https://huggingface.co/spaces/elismasilva/sup-toolbox-app)、[App 仓库](https://github.com/DEVAIEXP/sup-toolbox-app) 和 [CLI 仓库](https://github.com/DEVAIEXP/sup-toolbox) 已开放并欢迎反馈。
- **发布 CPU 友好型分布式 Muon 教程**：发布了一个关于 “**Muon is Scalable**” 优化器的完整注释解析，为了清晰度、可复现性和易用性进行了重构，该教程运行在 **Gloo** 而非 **NCCL** 上。
   - 该教程涵盖了 **DP + TP** 组如何协作、**ZeRO** 分片如何适配，以及为什么分桶（bucketing）和合并（coalescing）不仅仅是“性能技巧”，并提供了[完整仓库](https://huggingface.co/datasets/bird-of-paradise/muon-distributed)和[文章说明](https://discuss.huggingface.co/t/tutorial-update-reverse-engineering-breakdown-released-the-muon-is-scalable-cpu-friendly-blueprint/170078)。
- **NVIDIA 文档布局检测模型在 HF Spaces 上的演示**：一位用户创建了一个 HF Space，演示基于 **YOLOX** 的 NVIDIA 文档布局检测模型。
   - 演示地址见[此处](https://huggingface.co/spaces/dinhanhx/nemoretriever-page-elements)。


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1437743048787365888)** (1 messages): 

> `Diffusers MVP 计划` 


- **Diffusers 启动 MVP 计划**：**Diffusers MVP 计划**开始与社区贡献者合作并向其学习。
   - 详情可在 [GitHub issue](https://github.com/huggingface/diffusers/issues/12635) 中查看；鼓励感兴趣的相关方加入。
- **无更多话题**：频道中未发现更多话题或摘要。
   - 添加此条目是为了满足 `topicSummaries` 中至少包含两项的要求。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1437947346062413936)** (3 messages): 

> `随机数据生成, PII 随机化` 


- **通过 Prompting 生成真实的随机数据**：一位成员询问系统是否可以通过 Prompting 生成真实的随机数据，而不是 `'XXX'`。
   - 另一位成员建议，使用普通的 Python 脚本可能更容易完成这项任务。
- **通过 Prompting 检测并随机化 PII**：一位成员建议，设置一个用于检测和随机化 **PII**（个人身份信息）的 Prompt 将是一个很有价值的功能。
   - 他们认为这比生成随机数据更可取。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1437533821212823673)** (49 messages🔥): 

> `SYNTH 数据集, Meta Omnilingual ASR, Moonshot AI Kimi AMA, Gamma B 轮融资, Meta GEM 模型` 


- **SYNTH 数据集和 Baguettotron/Monad 模型发布！**：Alexander Doria 宣布了 **SYNTH**，这是一个全合成的 **200 B-token** 通用预训练数据集，专注于推理能力，此外还发布了两个仅在 **SYNTH** 上训练的新 SOTA 模型：**Baguettotron** 和 **Monad** —— 详见推文[此处](https://x.com/Dorialexander/status/1987930819021635964)。
- **Meta 发布支持 1,600 种语言的 Omnilingual ASR！**：Meta 推出了涵盖 **1,600+** 种语言的开源 **Omnilingual ASR 模型**，引发了热烈讨论，涉及方言/延迟等问题，并提供了 Hugging Face 实时演示，以及关于通用翻译器的梗图 —— 详见公告[此处](https://ai.meta.com/blog/multilingual-model-speech-translation-communication/)。
- **Moonshot Kimi AMA：Scaling 过程是残酷的！**：Cody Blakeney 称 Moonshot AI 的 Kimi AMA 是一个*金矿*，他观察到扩展新想法的过程非常残酷：许多看似无关的因素开始相互作用，详尽消融实验（ablation）的成本高得令人望而却步，然而，任何在规模化（scale）后奏效的方法最终都会带来巨大的回报 —— 详见推文[此处](https://x.com/code_star/status/1987924274116784500)。
- **Gamma 在 B 轮融资中筹集 21 亿美元**：Grant Lee 宣布 Gamma 以 **21 亿美元**估值完成 B 轮融资，由 a16z 领投，该公司在仅有 **50 名员工**的情况下实现了 **1 亿美元 ARR** 的盈利，并赞扬了其 **200 万美元人均 ARR** 的效率 —— 详见推文[此处](https://x.com/thisisgrantlee/status/1987880600661889356)。
- **LeCun 在遭受批评后将离开 Meta？？**：推文引用 FT 报道称 Meta 首席 AI 科学家 **Yann LeCun** 计划辞职；评论者开玩笑说他可能会立即卖掉一家初创公司给 Meta，并提到了他最近反硅谷、反特朗普的帖子 —— 详见推文[此处](https://x.com/grady_booch/status/1988278574076621138)。


  

---

### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1437653065753690182)** (10 messages🔥): 

> `AI 进展, Magic Patterns 2.0, A 轮融资` 


- **AI 文本质量飞跃至白板级别**：Dhruv 分享了一篇 [推文](https://xcancel.com/haildhruv/status/1987801616448405538?s=46)，对 AI 文本质量在不到一年的时间里从*纯粹的胡言乱语*飞跃到高级输出表示惊讶，现在 **AI 已经可以在白板上记录手写笔记了**。
   - 回复中提到了指数级的进步速度，并开玩笑说我们在 **10 个月内**就从*写诗*跨越到了复杂的推导。
- **Magic Patterns 获得 600 万美元 A 轮融资**：Alex Danilowicz 发布了 **Magic Patterns 2.0**，并宣布由 Standard Capital 领投的 **600 万美元 A 轮融资**。他庆祝了在**没有员工**的情况下通过自筹资金达到 **100 万美元 ARR** 的成就，目前已有 **1,500 多个产品团队**通过 [推文](https://xcancel.com/alexdanilowicz/status/1988247206940602440?s=20) 介绍使用该 AI 设计工具。
   - 用户称赞该产品已经取代了他们手中的 Figma，并且该公司正在企业、工程、社区和增长等职位进行快速招聘。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1437536399459876894)** (50 messages🔥): 

> `Opencode 交织思考, Kimi-cli 自动启动思考模式, Moonshot 推理集群变慢, Kimi 编程计划 API 配额, Bug 报告指南` 


- **Cline 编程扩展支持交织思考 (Interleaved Thinking)**：成员们注意到 **Opencode** 不支持交织思考，但 **Cline**、**Claude Code** 和 **kimi-cli** 现在都已支持。
   - 用户还确认 **Cline 扩展** 已正确配置。
- **Kimi-CLI 思考模式自动启动教程**：一位用户询问如何让 **kimi-cli** 自动以思考模式启动，因为经常忘记开启。
   - 一名成员回复称，`--thinking` 标志已在 **v0.51** 中添加。
- **Moonshot 集群在 Reddit AMA 后遭遇惊群效应 (Thundering Herd)**：一位用户注意到 **Moonshot 推理集群** 在过去几小时内运行缓慢。
   - 另一名成员建议这可能是由于最近的 **Reddit AMA** 吸引了大量用户而导致的“惊群效应”。
- **Kimi 编程计划用户快速耗尽 API 配额**：用户报告称，在短短几小时内就用完了每周的 **Kimi 编程计划 API 配额**。
   - 一位用户认为 **Web 搜索** 和 **计划模式 (plan mode)** 可能会消耗大量的 API 调用；其他人建议提交 Bug 报告。
- **Bug 报告指南澄清**：一位用户被引导至 <#1371764324866982008> 频道，以提交关于 **API 配额** 消耗过快的 Bug 报告。
   - 他们还被建议查看 **Bug 报告指南** 并提供具体信息，同时指出 **Kimi 团队** 虽然活跃但位于中国，因此应考虑时差问题。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1437591684740284536)** (20 messages🔥): 

> `hatch vs setuptools, pyproject.toml, package data, ChatGPT 对 setup 的建议` 


- **辩论升级：Hatch vs Setuptools**：讨论集中在选择 `hatch` 还是 `setuptools` 作为构建系统，一位成员指出 `hatch` *更精简、更现代*，且没有 `setuptools` 那样的样板代码。
   - 虽然 `setuptools` 不是标准，但它被广泛使用，导致为了兼容性又切换了回来；一位参与者调侃道：*我不知道它在 Python 中不是标准，但每个人都有它*。
- **Wheels 与元数据：pyproject.toml**：会议澄清了构建 Wheel 需要构建系统，因为 `pyproject.toml` 是一种没有构建逻辑的元数据格式。
   - 讨论引用了 `setup.py` 中的 `from setuptools import setup`，强调了对构建系统的需求。
- **pyproject.toml 中包数据 (Package Data) 的回归**：一个关键讨论点涉及在迁移到 `pyproject.toml` 后确保正确包含 `package data`。
   - 具体而言，对话涉及将 `setup.py` 中的 `include_package_data=True` 移植到 `pyproject.toml` 文件中，参考了 [此 commit](https://github.com/tinygrad/tinygrad/pull/13189/files#diff-50c86b7ed8ac2cf95bd48334961bf0530cdc77b5a56f852c5c61b89d735fd711R19)。
- **ChatGPT 对 setuptools 的独到见解**：合并后，一位成员提到 ChatGPT 对 setuptools 提出了一些建议，发现它们大多不是关键性的，但注意到它们*喜欢 setuptools 的 >=61 版本*。
   - 讨论链接到了一个 [ChatGPT 对话](https://chatgpt.com/share/69136ddd-4bfc-8000-8167-a09aaf86063b)。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1437703080303919154)** (13 messages🔥): 

> `M4 Mac segfaults, torch private buffer issue, Tensor from URL` 


- **M4 Mac 出现段错误 (segfaults)**：一位成员报告称，在 **M4 Mac** 上使用 `torch_to_tiny()` 将 **torch tensor** 转换为 **tinygrad tensor** 然后再转换为 **numpy** 数组时，会持续出现段错误。
   - 在转换后添加 `+0` 似乎可以解决此问题，这表明原始 **torch tensor** 的生命周期或内存管理可能存在潜在问题，代码详见[此处](https://discord.com/channels/1041495175467212830/1194674314755770478/1194703396814866462)。
- **Tinygrad 无法直接从私有 torch 缓冲区复制**：一位成员报告称，**torch** 将缓冲区创建为私有的，因此它不与 CPU 共享且没有 `contents()`，因此无法直接从 tensor 复制（**tinygrad** 不支持从私有缓冲区复制）。
   - 建议通过下载 `.safetensors` 文件直接将参数转换为 **tinygrad**，这样 **tiny tensor** 就可以直接转换而无需经过 **torch**。
- **来自 URL 的 Tensor 用法**：一位成员考虑使用 `Tensor.from_url` 直接将 **VGG16** 权重加载到 **tinygrad** 中，而不是从 **PyTorch** 转换。
   - 这种方法绕过了从 **PyTorch** tensor 转换的需求，因为 `.safetensors` 文件可以直接下载并用于 **tinygrad**。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1437575818971971744)** (17 messages🔥): 

> `PowerPoint presentations with company brand, Manus invite, Publish button issues, AI engineer introductions, Mini AGI` 


- **对使用 Manus AI 生成品牌 PowerPoint 的需求**：一位成员询问关于使用 **Manus AI** 生成带有其公司品牌的 **PowerPoint 演示文稿** 的事宜。
   - 他们是新用户，正在寻求如何实现这一功能的建议。
- **社区成员寻求 Manus 邀请码并表示愿意付费**：一位社区成员请求 **Manus 邀请码**，并表示愿意为此付费。
   - 另一位成员表示可以免费提供一个。
- **“发布 (Publish)”按钮失效**：一位 **Pro 订阅者**报告称，“发布”按钮在首次发布后无法更新网站，导致网站停留在旧的检查点。
   - 他们联系了 Manus 帮助中心，但至今未收到回复。
- **AI 工程师自我介绍并寻求工作机会**：多位 AI 工程师介绍了自己，强调了他们在**工作流自动化、LLM 集成、RAG、AI 内容检测和区块链开发**等领域的经验，并表示可以进行合作。
   - 其中一位明确了在构建 **AI Agent、自动化工作流、开发 NLP 驱动的聊天机器人、集成语音和语言系统以及部署自定义 LLM** 方面的专业知识。
- **Mini AGI 将于 2026/34/February2wsx7yhb-p;.,nbvcxz 发布**：一位成员提到正在创建 **mini AGI**。
   - 他们随后给出了发布日期：*2026/34/February2wsx7yhb-p;.,nbvcxz*。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1437851211226419222)** (2 messages): 

> `DSPyMator Launch, Taxonomy Creation Blogpost` 


- **DSPyMator 来了！**：一位成员宣布了 **DSPyMator** 的发布，并分享了 [X 上的发布链接](https://x.com/jrosenfeld13/status/1988290653324013666)。
   - 该成员表达了极大的热情，将其描述为“非常非常有趣”。
- **分类法 (Taxonomy) 创建技巧**：一位成员分享了一篇关于他们在创建分类法方面的经验的[博客文章](https://shabie.github.io/2025/11/10/why-tails-break-taxonomies.html)。
   - 该成员认为这个话题在**结构化生成 (structured generation)** 的背景下非常相关。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1437589828999970836)** (11 messages🔥): 

> `Transfer Learning, GEPA Prompting with Strong Models, Optimizing OCR pipeline with GEPA, Saving/Loading DSPy Modules` 


- **用于迁移学习的 GEPA Prompting**：一位成员询问了关于 GEPA 迁移学习的正式研究，特别是使用较廉价的模型进行 GEPA 优化，然后将该提示词用于更强大的模型，以降低 rollout 成本。
   - 另一位成员分享了一篇[论文链接](https://www.intrinsic-labs.ai/research/ocr-gepa-v1.pdf)，他们在其中优化了一个 OCR 流水线，使用 **2.0 Flash** 作为执行器，**GPT-5-High** 作为讲师（teacher）。
- **OCR 流水线优化增益的迁移**：[OCR 研究](https://www.intrinsic-labs.ai/research/ocr-gepa-v1.pdf)发现，为 **2.0 Flash** 优化的提示词在经过轻微修改后，可以将其 **80–90%** 的增益迁移到 **2.5 Flash 和 Pro** 上。
   - 这表明在 GEPA 优化的提示词中存在迁移学习的潜力。
- **为 GEPA Rollouts 保存/加载 DSPy 模块**：一位成员尝试使用 `save_program=False` 保存其模块状态（之前由 GEPA 优化），发现生成的 **.json** 文件中仅保存了优化后的 "instructions"（指令）。
   - 他们询问是否可以使用 `save_program=True` 来保存 2-3 次 rollout (`max_full_evals`) 后的中间结果，以便进行迭代式提示词优化。
- **提示词范式依然存在**：一位成员表示，DSPy 的理念是你不需要担心 LLM 的提示词工程，让算法来处理提示词，你只需编写“任务 -> 结果”即可。
   - 但他们认为这种范式更适用于简单任务，尤其是分类任务。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1437588119170973868)** (9 messages🔥): 

> `aider-ce improvements, webspp UI integration with aider-ce, Merge editor with conflict resolution, Code Snippets in MD Files` 


- **社区在创作者缺席期间接手 Aider-CE**：由于原作者缺席，社区成员正带头改进 [aider-ce](https://github.com/paul-gauthier/aider)，重点是将精力投入到代码改进中。
   - 一位成员指出，虽然目前有一位社区成员做得很好，但*更多的参与者和代码审查将使其变得更好*。
- **Webspp UI 旨在集成 Aider-CE**：一位成员正致力于将其 [webspp UI](https://github.com/flatmax/Eh-I-DeCoder) 与 aider-ce 集成，但注意到 UI 所依赖的一些内容发生了变化。
   - 一位社区成员鼓励该用户加入专门频道，以便让 CE 与其系统重新对齐。
- **带有冲突解决功能的合并编辑器**：一位成员表示有兴趣为 aider-ce 添加带有冲突解决功能的合并编辑器，并询问了实现方法。
   - 原贴作者提到可以使用带有主题的 Monaco diff 编辑器。
- **在 MD 文件中创建代码片段时遇到麻烦**：一位用户报告在使用 Aider 配合 Anthropic 的 Claude 模型时，在 Markdown 文件中创建代码片段会出现问题，工具会被嵌套的 Markdown 代码标记搞混。
   - 生成的 README.md 包含嵌套的代码片段，这似乎干扰了处理过程，导致意外出现 *Create new file?* 等提示。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1437595082470461520)** (2 messages): 

> `LLM Preprocessing Scripts, LLM Summarization Scripts, LLM Planning` 


- **LLM 生成预处理脚本**：一位成员建议让模型生成一个预处理脚本，因为语言模型在处理大规模 JSON 数据时可能会感到吃力。
   - 其理念是，一个 **1-2 页的脚本**可以快速使 **LLM** 更好地理解数据，从而改进总结脚本。
- **LLM 辅助编写总结脚本**：一位成员提出了一个挑战，让 **LLM** 提供文件内容的良好总结以及预期的结果。
   - 该方法包括让 **LLM** 创建一个计划，提出问题，然后根据回答修订计划。


  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1437760225783320597)** (2 条消息): 

> `Aider-CE Agent Mode, Chrome Devtools Integration, Context7` 


- **Aider-CE 发布 Agent Mode 演示**：Aider-CE 发布了一个演示，展示了其集成 **Chrome Devtools** 和 **Context7** 的全新 **Agent Mode**。
   - 在演示中，他们[在 10 分钟内一站式（one-shot）构建了一个 webapp](https://www.circusscientist.com/2025/11/11/aider-ce-building-a-webapp-and-testing-live-with-a-single-prompt/)，包括浏览器内测试，引发了广泛关注。
- **Aider-CE 支持浏览器内测试**：Aider-CE 的演示重点展示了直接在**浏览器中进行测试**的能力，这是新 **Agent Mode** 的核心功能。
   - 正如[链接的博客文章](https://www.circusscientist.com/2025/11/11/aider-ce-building-a-webapp-and-testing-live-with-a-single-prompt/)所示，该功能允许在 webapp 开发过程中进行实时反馈和调试。


  

---


### **MCP Contributors (Official) ▷ #[mcp-dev-summit](https://discord.com/channels/1358869848138059966/1413517834805313556/1437753953046757467)** (2 条消息): 

> `MCPConference Paris, Declarative Agents, Evals` 


- **MCPConference 落地巴黎**：根据 [luma.com](https://luma.com/MCPparis2025) 网站显示，**MCPConference Paris** 定于 **11 月 19 日**举行。
- **Declarative Agents 引入 Evals**：一位成员将在会议上讨论由 **MCP** 驱动的 **Declarative Agents**，并结合 **Evals** 进行讲解。