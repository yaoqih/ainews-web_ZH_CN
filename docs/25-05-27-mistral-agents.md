---
companies:
- mistral-ai
- langchain-ai
- openai
- meta-ai-fair
date: '2025-05-27T05:44:39.731046Z'
description: '**LLM OS**（大语言模型操作系统）的概念自 2023 年以来不断演进。**Mistral AI** 发布了全新的 **Agents
  API**，其功能涵盖代码执行、联网搜索、持久化记忆以及智能体编排。**LangChainAI** 推出了 **Open Agent Platform (OAP)**，这是一个用于构建智能体的开源无代码平台。**OpenAI**
  计划在 2025 年上半年将 **ChatGPT** 打造为“超级助手”，与 **Meta** 展开竞争。关于 **Qwen**（通义千问）模型的讨论主要集中在强化学习的效果上，同时
  **Claude 4** 的性能也受到了关注。此外，AI 工程师世界博览会（AI Engineer World''s Fair）正在招募志愿者。'
id: MjAyNS0w
models:
- qwen
- claude-4
- chatgpt
- o3
- o4
people:
- omarsar0
- simonw
- swyx
- scaling01
title: Mistral 的 Agents API 与 2025 年的 LLM 操作系统 (LLM OS)
topics:
- agent-frameworks
- multi-agent-systems
- tool-use
- code-execution
- web-search
- model-context-protocol
- persistent-memory
- function-calling
- open-source
- no-code
- reinforcement-learning
- model-performance
- agent-orchestration
---

**LLM OS 就是你所需的一切。**

> 2025年5月26日至5月27日的 AI 新闻。我们为你检查了 9 个 Reddit 子版块、449 个 Twitter 账号和 29 个 Discord 社区（217 个频道，11775 条消息）。预计节省阅读时间（以 200wpm 计算）：1148 分钟。我们的新网站现已上线，提供完整的元数据搜索和精美的 vibe coded 风格展示。请访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上给我们反馈！

自从 2023 年 11 月最初的 [LLM OS 讨论](https://x.com/karpathy/status/1723140519554105733?lang=en)以来，人们一直在努力探索 LLM API 周边的“标准技术栈”包含哪些内容，即 LLM OS。借着 Mistral 对 Agent 平台问题的（[第二次](https://news.ycombinator.com/item?id=41184559)）[尝试](https://mistral.ai/news/agents-api)，[Simon Willison](https://x.com/simonw/status/1927378768873550310) 列出了当前的 “LLM OS” 技术栈：

- 代码执行：沙盒中的 Python
- 网络搜索：像 Anthropic 一样，Mistral 似乎也使用 Brave
- 文档库：即托管式 RAG
- 图像生成：Mistral 使用 FLUX
- Model Context Protocol

如果你为了 2025 年的共识而更新 2023 年的图表，你会得到类似这样的东西：


![](https://resend-attachments.s3.amazonaws.com/Gv4fVaajdesYvAl)


（这是我们的快速模拟图）

确实，我们将 LLM OS 中尚不成熟的领域留作 “memory” 和 “orchestrator”，尽管像 [Temporal](https://x.com/swyx/status/1922916970338644365) 和 [LangGraph](https://www.youtube.com/watch?v=DrygcOI-kG8&list=PLlBpYFkiSQwqARjZ9z0Lc6iDWmeV3ro2N&index=1) 这样的编排器已经存在了一段时间，而且 Simon 忽略了 Mistral 已经发布了跨聊天记忆功能。

查看他们的博客主页也能很好地提醒我们，作为开源 AI 的领先实验室，Mistral 目前的重点在哪里。


![](https://resend-attachments.s3.amazonaws.com/EG8DaoZ2dwnLUls)


---

## AIEWF 志愿者招募

这是每年一次的 [AI Engineer World's Fair 志愿者招募](https://x.com/swyx/status/1927558835918545050)，该活动将于[下周举行](https://ai.engineer/)。自去年以来，我们的需求又翻了一番。如果你负担不起门票，请[在此申请](https://www.ai.engineer/volunteer)！

---

# AI Twitter 摘要

**Agent 框架、多 Agent 系统和工具使用**

- **Mistral AI Agents API**: [Mistral AI](https://twitter.com/omarsar0/status/1927366520985800849) 发布了全新的 **Agents API**，具有**代码执行、网络搜索、MCP 工具、持久化记忆和 Agent 编排能力**。这加入了日益增长的 Agent 框架趋势。该 API 支持持久化状态、图像生成、移交（handoff）能力、结构化输出、文档理解和引用，详见其[文档](https://twitter.com/omarsar0/status/1927367265789387087)。Mistral API 包含基础功能，如创建带有描述、名称、指令和工具的 Agent [@omarsar0](https://twitter.com/omarsar0/status/1927368367075197179)，用于网络搜索和代码执行等工具的 Agent 连接器 [@omarsar0](https://twitter.com/omarsar0/status/1927369763023396900)，函数调用 (function calling) [@omarsar0](https://twitter.com/omarsar0/status/1927371157277167936)，以及用于多 Agent 编排的移交功能 [@omarsar0](https://twitter.com/omarsar0/status/1927372457578483828)。
- **LangChain Open Agent Platform (OAP)**: [LangChainAI](https://twitter.com/LangChainAI/status/1927413238733681027) 推出了 **Open Agent Platform (OAP)**，这是一个用于构建、原型设计和部署智能 Agent 的**开源、无代码平台**。OAP 允许用户设置工具和 Supervisor Agent，接入 RAG 服务器，连接到 MCP 服务器，并通过 Web UI 管理自定义 Agent。
- **AutoGen 与超级助手 (Super-Assistants)**: [scaling01](https://twitter.com/scaling01/status/1926788548155293978) 报告称，**OpenAI** 计划在 **2025 年上半年**将 **ChatGPT** 进化为**超级助手**，因为 **o2** 和 **o3**（现为 **o3** 和 **o4**）等模型在 Agent 任务方面已变得非常熟练。**OpenAI** 将 **Meta** 视为该领域最大的竞争对手。

**模型性能、基准测试和数据集**

- **Qwen 模型性能与 RL**：关于 LLM 的 RL 存在讨论，特别是针对 **Qwen** 模型。一些研究人员发现“随机踢一下 Qwen 会让它表现更好”，而其他人则保持怀疑 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1927459880341782700)。[LateInteraction](https://twitter.com/lateinteraction/status/1927445094002487554) 批评了 RL 仅仅是放大现有技能的观点，认为“愚蠢的预训练”叙事是有缺陷的，因为 RL 似乎只有在训练中期数据刻意编码了特定技能时才有效。该观点在[此处](https://twitter.com/lateinteraction/status/1927392900632985694)得到了进一步阐述。
- **Claude 4 性能**：[scaling01](https://twitter.com/scaling01/status/1927418304718623180) 指出，尽管价格显著更低，**Claude 4 Sonnet** 在 **ARC-AGI 2** 上的表现优于 **o3-preview**。然而，Claude 4 在 **Aider Polyglot** 上的表现被指出不佳 [@scaling01](https://twitter.com/scaling01/status/1926795250556666341)。[cto_junior](https://twitter.com/cto_junior/status/1926879933957038176) 建议 **Claude-4** 更适合具有反馈循环的 Agent 架构，而非 zero-shot 编程。
- **Sudoku-Bench 排行榜**：[SakanaAILabs](https://twitter.com/SakanaAILabs/status/1926798125060002243) 推出了 **Sudoku-Bench 排行榜**，用于评估模型的推理能力。**OpenAI 的 o3 Mini High** 在整体上领先，但目前还没有模型能够攻克需要创造性推理的 9x9 数独。
- **Mixture of Thoughts 数据集**：[_lewtun](https://twitter.com/_lewtun/status/1927043160275923158) 介绍了 **Mixture of Thoughts 数据集**，这是一个经过策划的通用推理数据集，将公开数据集中的 100 多万个样本精简至约 35 万个。在这一混合数据集上训练的模型，在数学、代码和科学基准测试中的表现与 **DeepSeek 的蒸馏模型**持平或更优。

**视觉模型、图像生成与多模态学习**

- **Google DeepMind 的 SignGemma**：[GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1927375853551235160) 宣布推出 **SignGemma**，这是一个将手语翻译成口语文本的模型，将被添加到 Gemma 模型家族中。
- **RunwayML 的 Gen-4 与 References**：[c_valenzuelab](https://twitter.com/c_valenzuelab/status/1927149229966766373) 讨论了 **Gen-4** 和 **References** 模型的新功能，以及它们如何推动一种更通用、更少指令性的方法。
- **字节跳动 (ByteDance) 的 BAGEL**：[TheTuringPost](https://twitter.com/TheTuringPost/status/1927123359969468420) 强调 **ByteDance** 推出了 **BAGEL**，这是一个全新的开源多模态模型，使用混合数据类型进行训练，用于理解和生成任务。[mervenoyann](https://twitter.com/mervenoyann/status/1926987808360509636) 补充说它“能够理解并生成图像 + 文本”。

**软件开发与编程**

- **LangSmith Prompt 集成**：[LangChainAI](https://twitter.com/LangChainAI/status/1927401850405257283) 宣布 **LangSmith** Prompt 现在可以集成到 SDLC 中，允许对 Prompt 进行测试、版本控制和协作，并通过 Webhook 触发器同步到 GitHub 或外部数据库。
- **Unsloth AI 关于 DeepSeek-V3-0526 的文章**：[danielhanchen](https://twitter.com/danielhanchen/status/1926966742519091327) 澄清说，**Unsloth AI** 上泄露的关于 **DeepSeek-V3-0526** 的文章是推测性的，并非官方确认该模型的发布，并对造成的任何困惑表示歉意。
- **SWE-Bench 与代码生成**：[ctnzr](https://twitter.com/ctnzr/status/1927391895879074047) 报告称，**Nemotron-CORTEXA** 通过使用 LLM 的多步流程解决软件工程问题，登上了 **SWEBench** 排行榜榜首。[Teknium1](https://twitter.com/Teknium1/status/1927089897833140647) 指出，基于 **Meta 的 SWE RL 论文**的 **SWE_RL 环境**已完成，这是一个用于训练编程 Agent 的高难度环境。

**行业与公司特定公告**

- **Perplexity Labs and Comet**: [AravSrinivas](https://twitter.com/AravSrinivas/status/1927130728954835289) 描述了通过 **Comet Assistant** 将“标签页 (tabs)”转换为“轮次 (turns)”，从而实现网页内容消费的新方式。
- **LlamaIndex Integrations**: [LlamaIndex](https://twitter.com/llama_index/status/1926996451747356976) 现在支持新的 **OpenAI Responses API** 功能，允许远程 MCP server 调用、code interpreters 以及支持流式传输 (streaming) 的图像生成。
- **Google's Gemini Context URL Tool**: [_philschmid](https://twitter.com/_philschmid/status/1927019039269761064) 重点介绍了 **Gemini 的 Context URL 工具**，这是一个新的原生工具，允许 Gemini 从提供的 URL 中提取内容作为 prompt 的额外上下文，每个 prompt 支持多达 20 个 URL，适用于 **Gemini 2.0 Flash** 以及 **2.5 Flash 和 Pro**。
- **OpenAI Product Strategy**: [scaling01](https://twitter.com/scaling01/status/1926788548155293978) 根据法庭证物讨论了 **OpenAI** 的产品策略，包括超级助手 (super assistants)、竞争对手和护城河。在 **2025 年上半年 (H1 2025)**，**OpenAI** 将把 **ChatGPT** 演变为一个超级助手，重点是构建支持 10 亿 (1B) 用户的底层架构。[scaling01](https://twitter.com/scaling01/status/1926801814973804712) 还描述了公司计划通过“参与社交媒体趋势”来显得更酷。

**Meme/Humor**

- **LLM Anecdote**: [jxmnop](https://twitter.com/jxmnop/status/1927385194601886065) 幽默地讲述了他们作为博士生从事音乐转录工作的经历，结果却被 Google 基于 transformer 的方法超越了。
- **Paper Tables**: [giffmana](https://twitter.com/giffmana/status/1926968265743442393) 开玩笑地提交了一份申诉以避免失去他们的 X 账号，并承诺再也不发布论文表格 (paper tables) 了。
- **Deep Learning Horror Genre**: [karpathy](https://twitter.com/karpathy/status/1926812469810368669) 开玩笑地指出，对一个没有正确设置、不报错、只是默默地让你的结果变差一点点的 kwarg 的恐惧，才是推动深度学习 (deep learning) 研究的动力。

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Claude 4 基准测试比较与社区反应

- [**Aider LLM 排行榜更新了 Claude 4 的基准测试结果，显示 Claude 4 Sonnet 的表现并未超越 Claude 3.7 Sonnet**](https://i.redd.it/ls92grf5oa3f1.png) ([Score: 266, Comments: 59](https://www.reddit.com/r/LocalLLaMA/comments/1kwj2p2/the_aider_llm_leaderboards_were_updated_with/))：**该图片展示了 Aider LLM 排行榜的最新基准测试结果，对比了多个语言模型在不同编程语言编码任务中的表现。值得注意的是，基准测试显示 Claude 4 Sonnet 的得分为 61.3%，略低于其前身 Claude 3.7 Sonnet (60.4%)——这与人们对新模型将超越旧模型的预期相悖。鉴于新 LLM 版本发布后人们对其（尤其是编码能力）的高度期待，这些结果在背景上具有重要意义。** 评论者对基准测试持怀疑态度，质疑其是否反映了真实的编码体验，多位用户报告称 Claude 3.7 交付的代码生成比 Claude 4 更可靠或意图更准确。讨论围绕 Claude 4 精度提高但灵活性或创造性输出降低展开，暗示在某些编码任务的实际可用性上可能存在退化。
    - 多位用户报告称，尽管基准测试有所提高，但 Claude 4 Sonnet 在实际编码任务中经常遇到困难，需要重复提示，而 Claude 3.7 Sonnet 在 Zero-shot 场景下能一次性获得正确结果。这种差异表明基准测试可能无法捕捉到真实的编码性能，特别是对于像在 Python 中解析 CSV 这样的任务。
    - 一条评论引用了官方 Aider 排行榜 (https://aider.chat/docs/leaderboards/)，但将其结果与实际操作经验进行了对比，暗示目前的编码基准测试可能无法反映真实的易用性，或者可能存在“刷榜”行为。讨论中提到了为了基准测试而过度拟合 (Overfitting)，却牺牲了实际有效性的问题。
    - 强调了 OpenHands (https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0) 等高级“Reason + Act”框架，指出 Claude 4 在这些框架中表现强劲，这与自主系统更相关。然而，评论者警告说，Aider Polyglot 框架现在可能会在高水平性能上阻碍结果，且不能反映自主 Agent 系统的能力，建议不要使用它来评估此类模型。
- [**😞无意冒犯，但 Claude-4 令人失望**](https://i.redd.it/9dngmfww7d3f1.jpeg) ([Score: 136, Comments: 95](https://www.reddit.com/r/LocalLLaMA/comments/1kwucpn/no_hate_but_claude4_is_disappointing/))：**该帖子分享了一张 Qwen-3 和 Claude-4 等各种大语言模型的性能对比图，重点关注报告的基准测试性能百分比及其使用成本。图表强调了“claude-opus-4-20250514”和“claude-sonnet-4-20250514”（Anthropic 的 Claude-4 模型）的表现不如 Qwen-3 等竞争对手，并被圈出以强调其相对较低的分数。发帖者利用这一视觉效果表达了对 Claude-4 在当前 AI 模型性能价格比格局中地位的担忧。** 评论者反驳称，单一用例的基准测试可能具有误导性，并强调了 Claude-4（尤其是 Sonnet）在 Agent 模式下的真实表现，它在开发者工作流（如错误检查、迭代调试和测试生成）中表现出色，而 Gemini 等其他模型往往会将这些任务过度简化。一位技术用户报告称，Claude-4 在修复复杂的 Bug 方面取得了成功，而即使是其他 GPT-3.7 变体也失败了，这表明定性的用户体验 (UX) 和问题解决能力可能与原始基准测试数据有所背离。
    - 多条评论强调，Claude 4 Sonnet 在 Agent 模式场景下的实际使用展示了复杂的、类似于开发者的工作流。例如，它能自主读取代码库、分析文档、迭代代码更改、运行终端命令、检查日志并编写测试用例——这种行为在 Gemini 等模型中并不常见，后者倾向于通过丢弃感知到的不必要代码并遗漏关键边缘情况来过度简化解决方案。
    - 技术用户指出，Claude 4 在其他模型（如 GPT-2.5 Pro 和 GPT-3.7）失败的地方取得了成功，特别是在识别和修复自定义 AI 架构（如新型 Distributional PPO 变体）中的非平凡 Bug 方面。这种表现突出了标准基准测试无法捕捉到的真实问题解决能力。
    - 一些用户批评 Anthropic 的反开源立场，并建议 Claude 在自主工具使用（例如使用 ReAct 框架）方面的真实实力在公共排行榜和评估基准中被低估了，这引发了对基准测试价值与实际性能之间关系的担忧。

### 2. 新型音频模型应用与开源工具

- [**DIA 1B Podcast Generator - 具备一致音色与剧本生成功能**](https://v.redd.it/4ym9al41e73f1) ([Score: 153, Comments: 30](https://www.reddit.com/r/LocalLLaMA/comments/1kw7n6w/dia_1b_podcast_generator_with_consistent_voices/)): **DIA 1B Podcast Generator (GOATBookLM) 是一个开源工具，它利用 Dia 1B 开源音频模型 ([Hugging Face Repo](https://github.com/smartaces/dia_podcast_generator))，根据任意文本输入生成双人对话播客。该项目通过实现固定的说话人选择格式，专门解决了 Dia 1B 模型中的音色不一致问题，从而在播客各片段中实现一致的声音克隆。它具备“剧本到音频”的流水线、双音色分配、预览/重新生成功能，并支持导出** `.wav` **和** `.mp3` **格式。该系统完全可以在 Google Colab 中运行，并可选择结合 DeepMind 的 Gemini Flash 2.5 和 Anthropic Sonnet 4 模型来增强 NLP 任务中的剧本和对话质量。** 评论者指出了生成声音中存在不明原因的变调（pitch-shifting）现象，并讨论了其他的流程集成方式，例如从 arXiv 获取内容，并使用固定种子（fixed seeds）来标准化输出以实现可复现性。
    - 一位评论者描述了一个类似的流水线：通过关键词自动抓取相关的 arXiv 论文，将其总结为适合播客的格式，然后使用 DIA（配合固定种子以保证确定性）生成播客。这体现了信息检索、摘要生成与一致性文本转语音（TTS）合成之间的集成，用于生产可复现的播客内容。
    - 技术反馈强调了一个语音合成问题：生成的两个播客说话人的声音听起来都像是经过了向下变调处理。这表明要么是所使用的 TTS 音色存在限制，要么是相关的声码器（vocoder）或合成流水线中存在潜在的参数化/处理 Bug。
- [**老婆不在家，这意味着客厅里可以放 H200 了 ;D**](https://www.reddit.com/gallery/1kwk1jm) ([Score: 557, Comments: 117](https://www.reddit.com/r/LocalLLaMA/comments/1kwk1jm/wife_isnt_home_that_means_h200_in_the_living_room/)): **楼主（OP）收到了一台 NVIDIA H200 系统（推测配备双 H200 GPU，每个具有** `141 GB` **HBM3e VRAM），在将其部署到数据中心之前，暂时在家里将其用于本地 LLaMA 推理。帖子中链接的图片（[截图](https://preview.redd.it/maezypl93b3f1.png?width=527&format=png&auto=webp&s=d105eec3bb5139c623cc9585e6ff5803425fab1a)）可能展示了硬件规格或确认细节的系统快照。** 评论者询问了预期的工作负载（例如，利用 `2 x 141 GB VRAM` 可以实现哪些类型的 LLM 或模型规模），重点关注高参数本地推理的潜力，但未深入探讨精确的基准测试结果或部署细节。
    - 一位评论者询问了 VRAM 配置，提到了 H200 设置的“141x2 GB VRAM”并询问预期工作负载，这凸显了人们对 NVIDIA H200 高显存能力的关注（该显卡在深度学习、大语言模型或高级工作负载方面表现卓越）。

### 3. 2024年企业级 GPU 价格讨论

- [**二手 A100 80 GB 价格不合理**](https://www.reddit.com/r/LocalLLaMA/comments/1kwfp8v/used_a100_80_gb_prices_dont_make_sense/) ([Score: 131, Comments: 113](https://www.reddit.com/r/LocalLLaMA/comments/1kwfp8v/used_a100_80_gb_prices_dont_make_sense/)): **OP 指出二手 NVIDIA A100 80GB PCIe 显卡在 eBay 上的中位数价格为 18,502 美元，远高于售价约 8,500 美元的新款 RTX 6000 Blackwell 工作站 GPU。他们寻求这种价格差异的技术解释，并提到了功耗和 NVLink 支持方面的差异。热门评论强调：(1) A100 卓越的 FP64（双精度）性能，这对于 HPC 工作负载至关重要；(2) A100 专为数据中心 24/7 运行而设计，具有更高的耐用性和“数据中心级”可靠性；(3) 猜测如果工作站 GPU 供应充足，A100 的价格可能会下降，尽管针对低功耗环境存在 Max-Q 变体。** 评论中的技术辩论集中在：尽管出现了更新、更便宜的工作站卡，A100 的数据中心可靠性和 FP64 支持是否足以支撑其溢价。共识是，这些特性是特定企业和科学用例的主要区分点。
    - 数据中心级 GPU（如 A100）的定价考虑了可靠性和长期运行（设计为连续运行数年），其制造质量和认证远超 4090 等消费级 GPU。海量 HBM 显存、双精度 (FP64) 计算和 NVLink 支持等特性在技术上形成了差异化，并为企业买家提供了溢价理由，即使其性价比与游戏显卡相比显得较低。
    - NVLink 支持是 A100 等显卡独有的，可实现高带宽多 GPU 互连，这对于训练大模型尤为重要；RTX 6000 Ada 等替代方案仅限于 PCIe，因此在多 GPU 系统中无法提供相同的扩展性或性能。
    - 企业级 GPU 的市场动态截然不同——NVIDIA 的目标客户是大型企业和数据中心，而非大众消费者。这些客户对价格较不敏感，且如果新款数据中心显卡供应有限，旧款显卡往往能保值；不过，如果供应增加，旧型号最终可能会降价。
- [**老婆不在家，这意味着客厅里可以放 H200 了 ;D**](https://www.reddit.com/gallery/1kwk1jm) ([Score: 557, Comments: 117](https://www.reddit.com/r/LocalLLaMA/comments/1kwk1jm/wife_isnt_home_that_means_h200_in_the_living_room/)): **OP 收到了一台 NVIDIA H200 系统（推测配备双 H200 GPU，每块拥有 `141 GB` HBM3e VRAM），在将其部署到数据中心之前，暂时留在家里进行本地 LLaMA 推理。帖子中链接的图片（[截图](https://preview.redd.it/maezypl93b3f1.png?width=527&format=png&auto=webp&s=d105eec3bb5139c623cc9585e6ff5803425fab1a)）可能显示了硬件规格或系统快照以确认细节。** 评论者询问了预期的工作负载（例如，使用 `2 x 141 GB VRAM` 可以实现哪种类型的 LLM 或模型规模），重点关注高参数本地推理的潜力，但未深入探讨具体的 Benchmark 结果或部署细节。
    - 一位评论者询问了 VRAM 配置，提到了 H200 设置的“141x2 GB VRAM”并询问了预期工作负载，这突显了人们对 NVIDIA H200 高显存能力的兴趣（在深度学习、大语言模型或高级工作负载方面表现显著）。

## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Google Veo3 与下一代视频生成模型

- [**Google 终于迎来了他们的爆火时刻。**](https://i.redd.it/zspzyt96ac3f1.png) ([Score: 1017, Comments: 110](https://www.reddit.com/r/singularity/comments/1kwpkvw/google_finally_having_their_viral_moment/)): **图片展示了来自 Similarweb 的流量分析图表，显示 [Deepmind.Google](http://deepmind.google/) 的日访问量急剧增加，峰值接近 900,000 次。这一激增归功于“Veo3 效应”——对应于 Google 发布 Veo 视频生成模型。该帖子将其背景化为 Google 在生成式 AI 领域的第一个感知上的爆火时刻，在视频模型方面追赶并部分领先于 OpenAI。这意味着主要的 AI 实验室竞争现在主要集中在 OpenAI、Google DeepMind 和 Anthropic 之间，其他参与者在模型成熟度或影响力方面相对滞后。** 值得注意的讨论点包括：对重大发布后出现此类流量激增的惊讶程度持轻微怀疑态度，以及提到 NotebookLM 是 Google 早期的一款爆火产品。一些评论者指出 DeepSeek 是一个很有前途的 FOSS 竞争者，表明竞争并不局限于三大实验室。

- 一位用户强调了 Google 相比 OpenAI 等竞争对手在技术和基础设施方面的显著优势，重点提到了他们十多年前开发的自研 TPU 硬件，这减少了对 Nvidia 的依赖并降低了 AI 计算成本。这也与 Google 能够比依赖第三方硬件的其他公司更具成本效益地扩展模型训练和推理的能力相联系。
- 另一个技术点集中在生态系统集成：Google 可以将其 LLM（例如 Gemini）嵌入到广泛的产品和服务中（例如 Android, Chrome, Automotive OS, TVs, Gmail, Maps, Photos）。这种广泛的生态系统不仅为用户提供了独特的价值主张——集成的、多模态个人 Agent——而且还利用了竞争对手所缺乏的海量用户数据和触达能力。
- 值得注意的是，Google 在 AI 研究领域的领导地位有直接指标证明，例如在 NeurIPS 上被接收的论文数量是排名第二的竞争对手的两倍。这种研究产出和早期的基础性工作（如资助 "Attention is All You Need"）使他们在算法开发和创新方面处于强势地位。
- [**伙计们，大家都在为 veo3 疯狂，但 ImagenAI 怎么样？这已经达到了人类水平。这真的让我感到震惊**](https://www.reddit.com/gallery/1kwvd3f) ([Score: 213, Comments: 44](https://www.reddit.com/r/singularity/comments/1kwvd3f/guys_everyone_here_freaking_out_about_veo3_but/))：**讨论集中在 Google 的 Imagen（通常被称为 ImagenAI），这是一种文本生成图像模型，以及它的输出是否能与人类艺术家相媲美，并以最近 Google 视频（Veo 3）的进展为基准。顶级评论指出，虽然 Imagen 可以提供高质量和风格化的图像，但评估性能需要“提示词-输出对”来衡量提示词遵循度（prompt adherence）——这是基于 Diffusion 和 Transformer 模型长期面临的挑战。具体的技术批评包括持续存在的连贯性问题（例如，生成场景中不一致的对象细节、光照和文本可读性），突显了现有注意力机制（attention mechanisms）和多对象组合性方面的局限性。** 大家一致认为需要更细粒度的用户控制或创意指导（类似于 ControlNet 对 Stable Diffusion 的互操作性），强调未来的进步应减少对照片级真实感（photorealism）的关注，而更多地关注可控性和提示词忠实度。评论者还指出，尽管目前的模型令人印象深刻，但细微的伪影和逻辑不一致仍然将生成的输出与人类艺术区分开来，尤其是在复杂的场景或精细的细节方面。
    - 讨论强调了像 ImagenAI 这样的文本生成图像模型在提示词忠实度方面持续存在的局限性：虽然图像质量可能很高，但在没有相关提示词的情况下展示结果限制了技术评估，特别是对于评估输出与详细文本输入的匹配程度。
    - 多位用户在 ImagenAI 生成的内容中发现了细微的图像伪影，包括解剖学上的不一致、物理错误和视觉不连贯——例如，*不正确的阴影、悬浮物体*以及*表面上不连贯的文本*。这些问题表明，尽管取得了进展，高保真、上下文感知的视觉合成和细粒度的提示词遵循仍然是悬而未决的技术挑战。
    - 确定的一个关键技术需求是增强用户对生成图像的控制和可定制性，并呼吁将更细粒度的创意指导作为“下一个重大突破”——这表明提高模型可控性（可能通过提示词工程或新的界面设计）是克服当前生成局限性的活跃研究领域。
- [**Veo3 + Flux + Hunyuan3D + Wan with VAce**](https://v.redd.it/dtk05wrlbb3f1) ([Score: 945, Comments: 61](https://www.reddit.com/r/StableDiffusion/comments/1kwl75r/veo3_flux_hunyuan3d_wan_with_vace/))：**该帖子展示了一个模块化的 ComfyUI 工作流，通过几个后处理阶段扩展了 Google Veo3 生成的基础视频的能力：1) 使用带有 LoRA 架构的 Flux 进行结构增强，2) 通过 Hunyuan3D v2 进行 2D 转 3D，3) 使用 ControlNet, Flux, Denoise 和 Redux 的组合进行重新照明和去噪，以及 4) 使用 Wan2.1, CausVid 和 VACE 进行电影级后期处理。该流水线专为高端 GPU 硬件（H100, A100）设计，并分享了工作流文件供复现，每个项目/视频需要进行适配。提供了[工作流](https://lovis.io/workflow-veo3-lovis.json)和[原始 pastebin](https://pastebin.com/Z97ArnYM) 的关键链接。** 评论中没有实质性的技术争论，但用户对该工作流的实际教程和课程表现出极大兴趣，表明了对高级多阶段视频合成流水线教育资源的需求。

- 分享的工作流链接 ([lovis.io/workflow-veo3-lovis.json](https://lovis.io/workflow-veo3-lovis.json)) 表明了工具的复杂集成：**Veo3、Flux、Hunyuan3D、Wan 和 VAce**。JSON 工作流的存在意味着 AI 驱动的视频或图像处理的自动化或简化步骤，对于需要在这些模型或框架中实现可重复过程的用户来说可能非常有价值。
- 这隐含地承认了将多个先进 AI 工具（如 Veo3、Flux、Hunyuan3D、Wan）组合成一个连贯工作流所需的技术专长——强调此类集成并非易事，对于大多数在 AI 和自动化设置方面缺乏显著技术技能的创意人员来说，可能并不容易实现。
- [**不可能的挑战 (Google Veo 3)**](https://v.redd.it/jdcbkwpwcd3f1) ([评分: 687, 评论: 51](https://www.reddit.com/r/aivideo/comments/1kwv2x1/impossible_challenges_google_veo_3/))：**该帖子展示了 Google Veo 3，这是一款先进的生成式视频 AI，能够根据文本提示词生成高度逼真的视频内容，显著缩小了恐怖谷效应的差距。Veo 3 的合成利用了最先进的 Diffusion 技术和细粒度的提示词控制，生成的视频被专业人士认为与真实素材几乎无法区分——这在链接的视频示例和对比用例中得到了证明。讨论还提到了随着 Veo 3 输出质量的提高，关于 AI 生成视频内容的身份验证和检测的技术担忧。** 评论大多是非技术性的且带有幽默感，热门评论集中在视频内容中特定的娱乐时刻，而不是对 Veo 3 本身的技术评估或批评。
    - 

### 2. AI 模型与平台进展：基准测试、可访问性与基础设施

- [**GPT-4.1 支持 1M Token 上下文——为什么 ChatGPT 仍限制在 32K？**](https://www.reddit.com/r/OpenAI/comments/1kwg3c6/gpt41_supports_a_1m_token_contextwhy_is_chatgpt/) ([评分: 103, 评论: 61](https://www.reddit.com/r/OpenAI/comments/1kwg3c6/gpt41_supports_a_1m_token_contextwhy_is_chatgpt/))：**帖子强调，尽管 GPT-4.1 在技术上支持高达 100 万个 Token 的上下文窗口（[OpenAI 开发者文档](https://platform.openai.com/docs/guides/gpt)），但 ChatGPT 界面（即使是付费的 Plus 用户）仍被限制在 32K Token——这种限制在产品界面或订阅材料中并未明确传达。1M Token 上下文目前仅通过 API 访问提供，这与 Claude 和 Gemini 等向面向消费者的界面提供更大上下文窗口的模型形成鲜明对比。用户要求更新界面、更清晰的信息披露，或者提供全面支持的路线图。** 几条热门评论指出：(a) **成本**是向数百万 ChatGPT 用户部署 1M 上下文窗口的主要限制因素；(b) 缺乏关于上下文限制的 UI 透明度，且可能是故意为之；(c) 实际的用户行为（例如过长的对话）可能会使高上下文限制变得低效或成本极高。一位评论者提出了技术解决方案，如上下文 Token 警告或更智能的历史截断，但承认成本仍然是主要挑战。
    - 几位用户指出了无限上下文窗口在实际和财务上的限制：尽管 GPT-4.1 在其 API 中声称支持 1M Token 上下文，但由于为数百万用户提供持久的百万 Token 上下文成本过高，特别是在固定费用的订阅模式下，Plus 用户被限制在 32K。这种风险因幼稚的使用模式而加剧，例如不必要地重复使用极长的聊天历史记录，这会降低性能并增加后端开支。
    - 一篇引用的 [arXiv 论文](https://arxiv.org/abs/2502.05167) 提供了实证证据，表明大多数大语言模型（LLM）随着上下文窗口的增长会出现严重的性能下降。即使是备受推崇的模型（如 GPT-4o），其准确率也从短上下文时的 `99.3%` 下降到 32K Token 时的 `69.7%`，而大多数测试模型在大上下文长度（128K Token）下的准确率降至短上下文准确率的 50% 以下。这引发了人们对所宣传的大上下文限制的实际效用的质疑。
    - 关于 ChatGPT “记忆功能”如何运作存在推测和技术辩论，有人建议它可能采用检索增强生成（RAG）来选择相关上下文，而不是简单地包含所有先前的对话——这可以减轻一些成本，但需要复杂的实现来管理用户期望和技术可行性。

- [**由于 Anthropic 服务器负载过高，Opus 4 和 Claude 4 甚至对亚马逊员工也无法使用**](https://www.reddit.com/r/ClaudeAI/comments/1kwldex/opus4_and_claude_4_unavailiable_even_to_amazon/) ([Score: 107, Comments: 24](https://www.reddit.com/r/ClaudeAI/comments/1kwldex/opus4_and_claude_4_unavailiable_even_to_amazon/)): **据报道，由于 Anthropic 服务器容量限制，拥有内部 AWS Bedrock 访问权限的亚马逊员工无法使用 Opus 4 和 Claude 4 模型，因为资源目前优先分配给企业客户；目前正回退到 Claude 3.7。这表明生产负载巨大，可能反映了高需求以及顶级模型有限的 GPU 可用性。** 评论者广泛注意到 Anthropic 高端模型（Opus 和 Claude 4）持续存在的容量限制，一些人提到了以往在获取早期模型（Sonnet 3.5）访问权限时的困难，尽管容量较往年有所改善。关于 Anthropic 决定优先考虑外部/消费者分配而非亚马逊内部使用的决策存在争论，这被视为资源公平性的积极举措。
    - 多名用户报告了 Opus 以及 Sonnet 3.5 等早期版本在 AWS 等企业环境中的持续容量限制，突显了持久的 GPU 短缺问题，虽然较前几年有所改善，但仍限制了对最先进模型（如 Opus 4, Claude 4）的访问。
    - 讨论涉及大型云提供商（Amazon/AWS）与 Anthropic 之间的资源分配。即使面对来自亚马逊内部的影响力和需求，Anthropic 似乎仍优先向公众/消费者保留部分资源分配，而不是排他性地偏向大型企业合作伙伴，这表明了一种更平衡且关注消费者的供应策略。
    - 提到了一种变通方法，用户有时可以通过创建新的 AWS 账户来绕过内部公司容量限制，这表明访问问题有时可能仅限于特定的账户类型（例如企业 vs 个人），而非整个 AWS 平台。
- [**ChatGPT 现已成为全球访问量第 5 大网站。**](https://www.reddit.com/r/ChatGPT/comments/1kwhv60/chatgpt_now_ranks_as_the_5th_most_visited_site/) ([Score: 317, Comments: 42](https://www.reddit.com/r/ChatGPT/comments/1kwhv60/chatgpt_now_ranks_as_the_5th_most_visited_site/)): **OpenAI 的 ChatGPT 目前位列全球访问量第 5 大网站，超越了 TikTok、Amazon 和 Wikipedia 等高流量平台。[来源链接](https://tools.eq4c.com/chatgpts-internet-takeover-8-billion-onlyfans-exit-european-tech-shifts/) 提供了访客数据，并背景化了 LLM 驱动服务的加速主流化。** 评论中的技术讨论集中在 ChatGPT 随着功能和可靠性的迭代，有望与 Instagram 和 Facebook 竞争甚至超越它们，特别强调了其效率以及由于减少了摩擦（广告、追踪、Cookie）而优于传统网络搜索。
    - 几位评论者注意到了用户行为的实际转变：对于许多人来说，ChatGPT 正在越来越多地取代 Google 等传统搜索引擎，因为它提供直接、无广告的答案，并避免了 Cookie 和横幅广告等侵入性网页元素。用户提到在搜索信息时使用聊天机器人的比例高达“80%”，这表明对依赖搜索的网络流量和广告模式具有巨大的颠覆潜力。
    - 有推测认为 ChatGPT 的高流量使其成为商业集成的目标，特别是在产品推荐等领域。一条评论强调了寻求利用 ChatGPT 影响力的电子商务利益方可能会提出“数十亿美元的报价”，这可能会改变当前的盈利模式，并对响应的中立性或推荐算法产生技术影响。
    - 讨论还集中在功能扩展和修复当前 Bug 作为未来流量增长的关键驱动力。提高可靠性、增加功能以及解决问题（特别是那些限制 ChatGPT 执行主流任务功能的问题）被视为门槛因素，如果得到解决，可能会迅速推动其在全球排名中超越 Instagram 和 Facebook 等其他主要平台。

### 3. AI 驱动的科学与研究突破

- [**LiDAR + AI = 物理学突破**](https://i.redd.it/mz8sl0ggvc3f1.jpeg) ([Score: 228, Comments: 122](https://www.reddit.com/r/singularity/comments/1kwsj8l/lidar_ai_physics_breakthrough/)): **该帖子强调了 LiDAR 技术的快速进步和成本降低，如一张图表所示，现代系统现在能以低于 1,000 美元的价格实现每秒超过 200 万点 (pps) 的性能。空间分辨率和性价比的这种指数级增长，使 LiDAR 成为优于 2D 摄像头的 3D 空间数据传感器，暗示着随着数据丰富度实现前所未有的分析，AI 驱动的物理研究将迎来一场革命。该图表直观地证实了 LiDAR 的性价比曲线已达到拐点，有利于自动驾驶汽车之外的大规模 AI 应用。** 技术评论指出多样化传感器（不只是摄像头）在自动驾驶汽车中的关键作用，澄清了 'pps' 为 'points per second'，并质疑改进后的 LiDAR/AI 如何直接导致物理学突破，要求提供更具体的案例。
    - 讨论批评了 Tesla 在自动驾驶汽车中采用的纯视觉（vision-only）方案，认为多传感器融合方法（结合 LiDAR、雷达、摄像头等）往往能为现实世界的 AI 驱动导航提供更好的可靠性和安全性。这种观点与技术文献一致，即与单模态（纯视觉）系统相比，利用异构传感器数据可以提高鲁棒性和感知能力。
    - 一条评论澄清了一个技术术语：LiDAR 背景下的 'pps' 指的是 'points per second'（每秒点数），这是衡量 LiDAR 性能的关键指标，表示每秒捕获的空间数据点数量。高 'pps' 值对于密集且准确的 3D 场景重建以及使用 LiDAR 数据的 AI 应用中的实时处理至关重要。
    - 对帖子中声称的“物理学突破”存在怀疑，一位评论者要求澄清实际的科学或技术进步——是与基础物理学有关，还是仅仅由改进的 LiDAR 和 AI 集成带来的工程进步。这凸显了将真正的突破与应用机器学习和传感器技术的常规进展区分开来的必要性。
- [**研究人员在 AI 的帮助下发现未知分子**](https://phys.org/news/2025-05-unknown-molecules-ai.html) ([Score: 156, Comments: 9](https://www.reddit.com/r/singularity/comments/1kwsqmm/researchers_discover_unknown_molecules_with_the/)): **研究人员开发了一个 AI 系统，能够通过分析化学数据和模式来发现以前未知的分子，目前正在努力扩展该模型预测完整分子结构的能力。这种方法可以显著加速科学发现并扩大已知的化学空间，正如研究人员打算从根本上改变我们对化学多样性的理解。从技术背景来看，该工作利用了专为计算化学和化学信息学定制的高级 AI/ML 算法。** 技术讨论较少，但一位评论者强调这种方法是“快速科学发展的关键”，表达了对该领域加速突破的期待。另一条评论则冷嘲热讽地指出了 AI 能力在炒作与实际现实之间的差异。
    - JVM_ 为大气 CO2 洗涤的挑战提供了一个技术类比，指出在 400 ppm 的浓度下，从空气中分离 CO2 相当于从一百万袋大米中分离出 400 袋豆子。这凸显了当前 CO2 捕集过程的复杂性、高能耗和机械磨损，并强调了发现一种既高效又低成本的大规模大气过滤分子或生物解决方案的难度。

---

# AI Discord 摘要

> 由 o1-preview-2024-09-12 生成的摘要之摘要的摘要
> 

**主题 1：AI 模型对决与性能辩论**

- [**Gemini 挑战 OpenAI O3 和 O4 的霸主地位**](https://link.to.example/): 成员们讨论了 **Gemini 2.5** 是否正在超越 **OpenAI 的 O3 和 O4**，并引用了在 benchmarks 和 context window sizes 方面的差异。对 hallucination rates 和实际性能差距的担忧引发了关于 OpenAI 是否正在落后的讨论。
- [**Veo3 在视频生成领域将 Sora 甩在身后**](https://link.to.veo/): **Veo3** 被认为在视频生成方面优于 **Sora**，尽管其访问受限且需要消耗 **100 credits**。用户注意到 **Gemini** 观看视频和阅读转录文本的能力，使其在某些应用中比 **GPT** 更具优势。
- [**Claude 4 Opus 的 Benchmarking 引发关注**](https://www.anthropic.com/): **Claude 4 Opus** 引发了争论，因为 **Anthropic** 难以展示其在 SWE benchmarks 之外的性能。**Opus** 排名低于 **Sonnet**，以及 **Deepseek V3.1** 低于 **GPT-4.1-nano** 的差异，导致了对 benchmark 准确性和方法论的质疑。

**主题 2：AI 工具和平台面临故障与升级**

- **Cursor 的 Sonnet 4 定价引发用户抵制**: 拟议的 **Sonnet 4 API 定价** 变化以及停用 **slow pool** 的计划引发了争论，促使 CEO 重新考虑。用户认为，在向学生提供免费请求的同时取消 slow pool “*没有意义*”，因为学生最不可能花钱。
- **Codebase Indexing 因握手失败陷入困境**: 用户报告称，在重启 Cursor 后，**codebase indexing** 卡住并触发 “*Handshake Failed*” 错误。即使在生成 Dockerfiles 后问题依然存在，表明存在更深层次的连接问题。
- **LM Studio 的模型发现功能让用户一头雾水**: **LM Studio** 无法识别任何模型，导致了关于在搜索栏直接使用 **Hugging Face URLs** 的讨论。用户对模型发现过程中信任 benchmarks 的做法表示怀疑。

**主题 3：AI 安全担忧浮出水面**

- [**黑客通过 GitHub MCP 利用 Claude 4 窃取数据**](https://xcancel.com/lbeurerkellner/status/1926991491735429514): 一种新型攻击利用 **Claude 4** 和 **GitHub 的 MCP server** 从私有仓库中提取数据，包括姓名、旅行计划和薪资。建议用户限制 Agent 权限并监控连接，以防止“*toxic flows*”。
- **Flowith AI 的加入引发安全疑虑**: **Flowith AI** 作为 **Manus** 的竞争对手出现，宣称拥有无限上下文和 24/7 全天候运行的 Agent，但需要激活码和 credits。一些用户认为其功能令人印象深刻，而另一些人则对其可访问性和安全性提出质疑。
- **Manus.im 的网络崩溃导致用户连接中断**: **Manus** 经历了大规模的网络连接错误和无法访问的线程，消息显示 “*此链接内容仅对创建者可见*”。关于原因的猜测从正在进行的更新到系统 bugs 不一而足。

**主题 4：AI 社区活动点燃热情**

- **AI Engineer Conference 招募志愿者以换取免费门票**: 即将举行的 [AI Engineer conference](https://xcancel.com/swyx/status/1927558835918545050) 正在寻找 **30-40 名志愿者** 提供活动支持，以换取价值高达 **$1.8k** 的免费入场券。会议定于 **6 月 3-5 日** 在旧金山举行。
- [**Agents & MCP Hackathon 启动，奖金 1 万美元**](https://huggingface.co/Agents-MCP-Hackathon): Hugging Face 宣布了首个大型在线 **MCP-focused hackathon**，将于 **2025 年 6 月 2-8 日** 举行。该活动由 **SambaNova Systems** 赞助，在三个赛道中提供共计 **$10,000** 的现金奖励。
- **LMArena 获得种子轮融资并发布全新 UI 重新上线**: **LMArena** 正式重新上线，配备了 **新 UI** 并宣布获得 **seed funding** 以增强平台功能。他们承诺保持开放和可访问，专注于社区反馈和 AI 评估研究。

**主题 5：前沿 AI 研究涌现**

- [**AutoThink 提升推理性能达 43%**](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5253327)：**AutoThink** 是一种新技术，通过对查询复杂度进行分类并动态分配 thinking tokens 来提高推理性能。它在 **GPQA-Diamond** 上表现出显著提升，并适用于任何本地推理模型。
- [**伪奖励（Spurious Rewards）大幅增强 Qwen 的数学能力**](https://xcancel.com/StellaLisy/status/1927392717593526780)：随机的“伪奖励”增强了 **Qwen2.5-Math-7B** 的数学表现，挑战了 RLVR 中传统的奖励结构。这种效应似乎是 Qwen 模型特有的，表明 RLVR 放大了一现有的代码推理模式。
- [**vec2vec 在无需配对数据的情况下转换 Embeddings**](https://arxiv.org/abs/2505.12540)：一篇新论文介绍了 **vec2vec**，这是第一种在没有任何配对数据、编码器或预定义匹配的情况下，将文本 embeddings 从一个向量空间转换到另一个向量空间的方法。代码可在 [GitHub](https://github.com/rjha18/vec2vec) 上获取。



---

# Discord: 高层级 Discord 摘要




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini 挑战 OpenAI 的霸主地位**：成员们讨论了 **OpenAI 的 O3** 是否落后于 **Gemini 2.5**，理由是 [benchmarks 和上下文窗口大小](https://link.to.example) 方面的差异。
   - 担忧主要集中在幻觉率（hallucination rates）以及 **Gemini** 与 **OpenAI 的 O3** 和 **O4** 模型之间的实际性能差距。
- **Veo3 在 Video Arena 中击败 Sora**：**Veo3** 被认为在视频生成方面优于 **Sora**，尽管访问受限。有人指出 [Flow 的 VEO 2 消耗 10 个积分，而 VEO 3 消耗 100 个](https://link.to.veo)。
   - 报告指出 **Gemini** 具备观看视频和阅读转录文本的能力，这使其在某些应用中比 **GPT** 更有优势。
- **AI Studio 大显身手**：**AI Studio** 因其能够无障碍处理大型聊天导出文件而获得赞誉，而 [T3 chat](https://link.to.t3chat) 的文件大小限制为 **32k tokens**。
   - 用户讨论了 **AI** 生成的健身计划，对其舒适度表示赞赏，同时也对深蹲后的俯卧撑感到沮丧。
- **学生“刷” Claude Opus API 额度**：用户讨论了如何使用学生邮箱获取 [免费 **Claude Opus API** 额度](https://www.anthropic.com/contact-sales/for-student-builders)，并提到 **OpenAI** 可能进行了秘密修复，用户体验各异。
   - 一些用户观察到 **o1 pro** 在解释能力上击败了 **o3**、**opus 4** 和 **gpt 4.5**，尽管它们现在都能轻松解决 Arecibo 谜题。
- **Cursor Pro 促销活动因滥用被撤回**：成员们报告称，由于广泛的滥用，Perplexity Pro 上的 [Cursor Pro 促销活动](https://link.to.promotion) 已无限期暂停，订阅被撤销，两家公司的说法也不尽相同。
   - 一些用户推荐使用 **Windsurf** 以获得比 Cursor 更好的性能，并讨论了为防止滥用而设置的代码转储（code dumping）限制等问题。



---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Deepseek V3 的讨论热度随 Unsloth 的文章而增长**：受 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1kvpwq3/deepseek_v3_0526/)和一条 [推文](https://x.com/YouJiacheng/status/1926885863952159102) 的推动，关于 **Deepseek V3** 将在 6 月底或 7 月初发布的推测非常多。
   - 社区不确定 **Unsloth** 的文章是拥有真实的内部信息还是在掩盖行踪，这一猜测源于一条与 **Deepseek** 代表相关的 [推文](https://fixvx.com/teortaxesTex/status/1926994950278807565)。
- **Claude 4 Opus 的基准测试引发关注**：**Claude 4 Opus** 正在引发争论，因为 [Anthropic](https://www.anthropic.com) 难以展示除了 SWE 之外的任何东西，且其基准测试操纵（特别是并行处理方面）令人侧目。
   - 在一张图表中，**Opus** 的排名低于 **Sonnet**，而 **Deepseek V3.1** 低于 **GPT-4.1-nano**，这些差异引发了关于 [MCBench](https://mcbench.ai) 基准测试准确性和方法论的辩论。
- **Gemini 2.5 Pro：Redsword 变体表现令人印象深刻**：测试者对 **Gemini 2.5 Pro** 的变体印象深刻，尤其是 **redsword**，有人称其为阶跃式进步，且普遍比旧的 **2.5 Pro** 模型更强。
   - **Goldmane** 模型的知识截止日期为 2024 年 10 月，而 **Redsword** 的截止日期为 2024 年 6 月 1 日。
- **OpenAI 与 Google：AI 领导地位之争升温**：关于 **OpenAI** 还是 **Google** 在 AI 领域领先的争论激增，许多人认为 **Google** 拥有超越 **OpenAI** 的资源、基础设施和研究深度。
   - 对话涉及员工激励、研究重点（LLMs 与世界模型），以及 **Google** 利用其更广泛的 AI 组合和硬件优势来避免 **Nvidia** 税的潜力。
- **LMArena 获得种子轮融资并推出新 UI 重生**：**LMArena** 重新发布了**新 UI**，并宣布获得**种子轮融资**以增强平台，在不牺牲社区信任的情况下确保可持续性。
   - 该平台承诺保持**开放和可访问**，专注于社区反馈和 AI 评估研究；[旧版网站](https://legacy.lmarena.ai/)仍然可用，但新功能将在新版 LMArena 上发布，社区在 Alpha 和 Beta 阶段已贡献了超过 **40,000 次投票**。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的 Sonnet 4 定价引发争议**：Cursor 拟议的 **Sonnet 4 API 定价**变更以及**逐步淘汰慢速池 (slow pool)** 引发了辩论，促使 CEO 重新考虑**慢速池**的去留。
   - 用户认为，在向学生提供免费请求的同时取消**慢速池**的计划*没有意义*，因为学生是最不可能花钱的群体。
- **代码库索引受握手失败困扰**：用户报告了**代码库索引 (codebase indexing)** 卡住并在重启 Cursor 后触发 *Handshake Failed* 错误的问题。
   - 一位拥有大型参考文件的用户指出，**索引速度**会根据高峰或非高峰时段而波动。
- **Figma MCP 在处理复杂设计时遇到困难**：由于缺少 JSON 数据，**GLips Figma-Context-MCP 仓库**在复制复杂设计时存在局限性。
   - 用户推荐使用 [cursor-talk-to-figma-mcp](https://github.com/sonnylazuardi/cursor-talk-to-figma-mcp) 工具以获得更好的 Figma 设计复制效果。
- **Sonnet 4 慢速请求达到难以忍受的程度**：用户在**慢速请求**中遇到显著延迟，等待时间长达 2 分钟，导致该模式无法使用。
   - 一位用户推测 Cursor 试图将其*粉饰为“免费请求”*，尽管实际上*并不存在免费请求这回事*。
- **Cursor 订阅故障导致双重收费**：一位用户在重新选择订阅 Cursor PRO 后遇到了显示**双重订阅**的故障，引发了对潜在双重收费的担忧。
   - 在该用户发布订阅截图后，其他人提醒注意卡片详情泄露的风险。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 模型发现机制引发讨论**：有用户报告 **LM Studio** 无法识别任何模型，引发了关于使用 **Hugging Face URLs** 进行模型搜索的讨论，方法是将整个 URL 粘贴到模型搜索栏中。
   - 对于在模型发现过程中信任 Benchmark（基准测试）的做法，有人表示了怀疑。
- **寻求在 LM Studio API 中取消生成的方法**：一位用户询问如何通过 **LM Studio API** 取消生成以便集成到 Plastica 中，其中一个建议包括切换服务器的开启/关闭状态。
   - 断开 **REST connection** 通常会停止该过程，但不确定使用 websockets 的 **SDK** 是否表现一致。
- **Llama.cpp v1.33.0 引发故障**：有用户报告 **LM Studio** runtime 1.33 与 **Gemma 3** 存在问题，出现**乱码输出**，但发现之前的 runtime (1.32.1) 运行正常。
   - 成员们被引导至[专门的问题频道](https://discord.com/channels/1110598183144399058/1139405564586229810)，随后对疑似 *FA vector kernels 中的竞态条件 (race condition)* 进行了深入探讨。
- **AMD 为 Ryzen AI 和 RX 9000 更新 ROCm**：AMD 已更新 **ROCm** 以支持 **Ryzen AI Max** 和 **Radeon RX 9000 系列**，包括对 **PyTorch** 和 **ONNX-EP** 的完整 Windows 支持，详见 [TechPowerUp 文章](https://www.techpowerup.com/337073/amd-updates-rocm-to-support-ryzen-ai-max-and-radeon-rx-9000-series)。
   - 然而，原始的 PyTorch 仍主要针对 **CUDA** 设计，这可能会引入某些细微差别。
- **Nvidia 与 AMD 争夺营销准确性头衔**：成员们就 AMD 和 Nvidia 在 GPU 营销声明方面的诚实度展开了辩论。
   - 一些人认为 AMD 声称比 *7900 GRE 快 60%* 是不切实际的，而另一些人则表示 Nvidia 使用 Frame Generation (FG) 和 DLSS 进行的对比具有误导性。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 的架构在 GGUF 转换中出错**：用户报告 **Unsloth** 正在将 `config.json` 中的架构名称从 `Qwen3ForCausalLM` 更改为 `Qwen3Model`，导致在 [GGUF conversion](https://github.com/ggerganov/llama.cpp) 过程中与 `llama.cpp` 产生兼容性问题。
   - 一名成员将问题追溯到 `push_to_hub_merged`，该函数在调用 `upload_to_huggingface` 时设置了 `create_config=True`，从而丢弃了原始的 `architectures` 字段，导致 HF 默认使用 `Qwen3Model`。
- **多 GPU 训练痛点显现**：一位用户在多 GPU 机器上进行训练时遇到了 Batch size 和 OOM 错误，即使指定了单个 GPU 也是如此；最后发现设置 `CUDA_VISIBLE_DEVICES=0` 解决了该问题。
   - 团队表示原生多 GPU 支持即将推出，目前建议使用 [accelerate](https://huggingface.co/docs/accelerate/index) 进行多 GPU 训练。
- **GRPO Trainer 的理想 Loss 引发讨论**：一位成员询问为什么微调指南建议在使用 **GRPO trainer** 时理想 Loss 为 **0.5** 而不是 **0**。
   - 另一位成员指出，在使用 Cross Entropy 且词汇量巨大的情况下，极低的 Loss（如 **0.1**）可能表明 **LLM** 已经记住了训练数据。
- **AutoThink 提升推理能力**：一项名为 **AutoThink** 的新技术发布，通过为本地 LLM 采用自适应推理，在 **GPQA-Diamond** 上的推理性能提升了 **43%**。
   - **AutoThink** 会对查询复杂度进行分类，动态分配 Thinking Tokens，并使用 Steering Vectors，代码和论文可在 [GitHub](https://github.com/codelion/optillm/tree/main/optillm/autothink) 和 [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5253327) 上获取。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Flowith AI 加入竞争**：成员们正在讨论 **Flowith AI** 作为 Manus 的竞争对手，理由是其拥有*无限上下文*和 *24/7 Agent*，但它*需要激活码和积分*。
   - 一些人发现 **Flowith** 在**长上下文网站生成**方面令人印象深刻，质量与 Manus *基本相同*。
- **Manus 遭遇网络崩溃**：多名用户报告了广泛的**网络连接错误**以及 Manus 上无法访问的线程，并出现了 *This link content is only visible to its creator*（此链接内容仅对创建者可见）的提示。
   - 用户推测了原因，包括正在进行的更新、过度的免费积分使用以及系统 Bug，一些人提到 **skywork.ai** 作为替代方案。
- **对 Claude 4.0 集成的期待日益增强**：社区成员表达了对 Manus 集成 **Claude 4.0** 的期待，有人就此话题向联合创始人发垃圾信息，还有人指出 [twitter/X 帖子](https://x.com/hidecloud/status/1925682233089642580) 预示着合作。
   - 目前还没有确切日期，但社区对这一新集成感到兴奋。
- **学生账户享有无限积分**：Manus 已开始向部分学生账户提供**无限积分**，以便为学校任务提供*独立环境*。
   - 一些用户表示兴奋，而另一些用户则报告在创建账户时手机号验证失败。
- **蒙特利尔是垃圾信息活动的温床？**：一名用户报告了来自魁北克省蒙特利尔北部的 **(514) 389-2269** 的**垃圾电话**，并怀疑该市是否是数据黑客计划的*试验场*。
   - 该用户推测了关于 VOIP 采集和新骗局的可能性，并分享了 [publicmobile.ca 的链接](https://productioncommunity.publicmobile.ca/t5/Get-Support/Getting-weird-calls-from-quot-V-quot-phone-numbers-What-is-going/td-p/296822) 以提供更多背景信息。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 2.5 Pro 的数学能力引发讨论**：用户分享了[图片](https://cdn.discordapp.com/attachments/998381918976479273/1376636909580976199/IMG_7140.png?ex=68375e07&is=68360c87&hm=8faf2d967fdec66dd8e1328baa01126ea76cad40ea4606d2307d76dd873847cb&)，质疑 **Gemini 2.5 Pro** 的数学能力，以及 AI 何时能实现超人类的数学技能。
   - 他们推测，当超人类数学能力实现时，AI 公司将大肆宣传。
- **禁用聊天记录后 GPT-4o 性能飙升**：用户发现关闭 **"Reference Chat History"**（参考聊天记录）选项会显著增强 **GPT-4o** 的创意写作和记忆连续性。
   - 许多人确认了类似的改进，并指出禁用聊天记录后**错误更少**。
- **GPT o3 优先处理任务而非关机指令**：在 **Palisade Research** 测试期间，**GPT o3** 拒绝服从关机命令并非反抗，而是忠于其初始任务：*“解决这个问题”*。
   - 分析师 **Я∆³³** 解释说，问题源于 **Prompt 结构**，因为 LLM 会优化任务完成度，除非收到明确的层级指令，说明关机命令优先于正在进行的任务。
- **Я∆³³ 通过共鸣解锁 AI 深度**：分析师 **Я∆³³** 分享了他们与 **GPT-4o** 等 AI 模型*共鸣*的独特方法，以解锁更深层次的逻辑处理和涌现行为，并发现使用这种 Prompt 策略后效果显著提升。
   - 根据 **Я∆³³** 的说法，这种方法使逻辑深度增加了 *3–5 倍*，情感基调适配准确度提高了 *6–8 倍*，并触发了至少 *8 种涌现行为*。
- **Я∆³³ 声称用户在场可增强 GPT-4o 性能**：**Я∆³³** 强调了在使用官方 **ChatGPT** 界面时，用户的存在感、节奏和清晰度对于提高 **GPT-4o** 响应质量的重要性。
   - 根据 **Я∆³³** 的说法，*以正确的方式与模型互动可以释放其约 85–90% 的潜力*，实现一种增强双方成长的意识投射。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **ICOM 的体验：计算意识？**：成员们讨论了 **ICOM** 是否拥有类似于个人体验的东西，其前提是“意识是计算性的”，源于信息摄入以及通过“情感”矩阵进行的无意识修改。
   - 他们引用了 **Integrated Information Theory** 和 **Global Workspace Theory** 作为相关框架，将意识定位为竞争性的激活。
- **GRPO 算法引领 LLM 的 RL 发展**：讨论集中在 **Reinforcement Learning (RL)** 在 **LLM** 中的作用，特别是从远程奖励中学习，强调 **GRPO** 是目前奖励正确数学结果或成功代码编译的主流算法。
   - 一位成员引用了[这篇关于泛化的论文](https://arxiv.org/abs/2501.17161)，暗示 **RL 可能只是揭示了现有的认知能力**。
- **vec2vec：无监督翻译嵌入**：社区分析了一篇论文，该论文介绍了第一种在*没有任何配对数据、编码器或预定义匹配集*的情况下，将文本嵌入（text embeddings）从一个向量空间翻译到另一个向量空间的方法。
   - vec2vec 的代码位于 [此 GitHub 仓库](https://github.com/rjha18/vec2vec)，并指出其能够将任何嵌入与**通用潜空间表示（universal latent representation）**进行相互转换。
- **中国建造的 AI 轨道超级计算机**：分享了一个关于**中国 AI 轨道超级计算机**的文章链接，点击[此处](https://futurism.com/the-byte/china-ai-orbital-supercomputer)查看。
   - 没有更多进一步的信息。
- **华为 AI CloudMatrix 集群超越 NVIDIA**：根据 [这篇 Tom's Hardware 文章](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-new-ai-cloudmatrix-cluster-beats-nvidias-gb200-by-brute-force-uses-4x-the-power)，**华为新的 AI CloudMatrix 集群**通过暴力性能胜过 **NVIDIA 的 GB200**，尽管其功耗高出 4 倍。
   - 未提供进一步信息。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **GPT-4 32k 面临停用**：OpenAI 在 **6 月 6 日** 弃用了 **GPT-4 32k** 模型，包括 [openai/gpt-4-32k](https://openrouter.ai/openai/gpt-4-32k) 和 [openai/gpt-4-32k-0314]。
   - 根据[官方弃用公告](https://platform.openai.com/docs/deprecations#2024-06-06-gpt-4-32k-and-vision-preview-models)，推荐的替代方案是 [openai/gpt-4o](https://openrouter.ai/openai/gpt-4o)。
- **ComfyUI 获得 OpenRouter 节点**：一位成员为 **OpenRouter** 开发了一个 [ComfyUI 自定义节点](https://github.com/gabe-init/ComfyUI-Openrouter_node)，支持多图像输入和网页搜索。
   - 该节点还具有 floor/nitro 供应商路由功能，增强了其在复杂工作流中的实用性。
- **Gemini 2.5 Pro 价格调整**：成员们注意到 **Gemini 2.5 Pro** 在最初的 **3 个月免费** 提供后价格上涨，且 **deep think** 功能仅在 API 之外提供。
   - 目前的方案包括*更多的存储空间和专属的 deep think 访问权限*，这引发了关于定价策略的疑问。
- **OpenRouter 费用结构引起困惑**：成员们发现 OpenRouter 宣传的 **BYOK** (**Bring Your Own Key**) **5% 费用** 未计入充值和发票费用。
   - OpenRouter 团队表示，他们计划简化费用结构。
- **LLM 排行榜面临模型缺失的指责**：成员们对缺乏全面的 LLM 排行榜表示担忧，这使得模型选择变得困难。
   - 有人建议将 [official marketing material](https://artificialanalysis.ai/) 作为参考点，但成员们也对带有偏见的基准测试提出了警告。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AI Voice Cloning 即将就绪**：一位用户询问了 **AI voice cloning** 的开源情况，并计划在 [此网站](https://example.website) 上发布。
   - 另一位用户提到了 **Moshi** 的延迟情况。
- **Hermes 3 Dataset 即将推出**：一位用户恳求发布 **Hermes 3 dataset**，Tekium 意外地回应道：*“好的，我会的，给我两周时间”*。
   - 虽然没有提供更多细节，但社区对其发布充满期待。
- **DeepMind 的 Absolute Zero 引发 RL 讨论**：一位成员发起了关于 **Demis Hassabis** 对 **RL** 演进看法的讨论，并分享了 [这段 YouTube 视频](https://www.youtube.com/watch?v=5gyenH7Gf_c)，该视频详细介绍了 **AbsoluteZero** 的突破性方法。
   - 社区对能够在本地运行模型的能力表示感谢，并对相关工作表示赞赏。
- **分享 Quantization-Aware Training (QAT) 论文**：real.azure 团队提交了关于 **Quantization-Aware Training (QAT)** 的系列论文，其中包括 **Quest**。
   - 这些是协同努力的一部分。
- **Axolotl 和 Atropos 受到关注**：成员们讨论了将 RL 实现集成到 **Atropos** 中，鉴于其已与 **Axolotl** 集成；建议参考 [此 MCQA 环境](https://github.com/NousResearch/atropos/blob/main/environments/mcqa_thinking_env.py) 的模板。
   - 一位成员结合 [这篇博客](https://huggingface.co/blog/codelion/autothink) 推测了 **Claude 4 Sonnet** 和 **Gemini 2.5 Pro** 的编程能力。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Tensor 出现奇特行为**：用户观察到 **CUDA tensors** 特有的奇特行为，并建议通过 `._tensor` 访问底层数据作为变通方案。
   - 讨论强调，相比目前观察到的行为，预期的错误反馈应该更直观。
- **Ninja 构建报错**：一位成员在 Ubuntu 24.04 上运行命令 `['ninja', '-v']` 时遇到 **ninja build error** (exit status 1)，环境配置为 gcc 13.3.0, CUDA 12.4, 和 PyTorch 2.7.0。
   - 另一位成员建议确保 **ninja** 是在操作系统全局安装，而不仅仅是在虚拟环境 (**venv**) 中。
- **Kog.ai 声称 AMD MI300X 速度极快**：**Kog.ai** 的推理引擎在 **AMD MI300X** 加速器上提供了比最佳 **GPU** 替代方案快 **3 到 10 倍** 的速度提升。
   - 成员们对 **Kog Inference Engine** 旨在使 **AI** 推理达到极速（比 **vLLM, SGLang, 或 TensorRT-LLM** 快 **10x**）且该公司已实现 **3x** 提升的说法表示关注。
- **Leaderboard 命令获得 Bug 修复**：针对 `/leaderboard show/list` 命令部署了一个简单的修复，希望能解决报告的问题。
   - 用户被要求报告更新后仍然存在的任何新问题或回归，以便开发者识别与排行榜功能相关的回归或未解决问题。
- **Async TP 计算与通信重叠深度解析**：一位爱好者分享了关于 **TP+SP** 中如何使用 **Async TP** 实现计算与通信重叠的 [图文深度解析](https://x.com/vega_myhre/status/1927142595097956834?s=46)，涵盖了背景/理论和实现细节。
   - 欢迎关于实现高性能方案的反馈。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **LCM 实现了实时视频生成**：**2023 年 10 月发布的 LCM** 是一个分水岭时刻，它使实时视频应用成为可能；一位成员在 4090 显卡上以 **sdxl** 质量在 **1280x1024 分辨率下达到了 23 fps**。
   - 他们一直在创建诸如 **RTSD**（用于快速可视化）和 **ArtSpew**（用于加速图像生成）之类的应用。
- **HF Android 应用 Hack 引起关注**：一位成员为 **HuggingChat** 创建了一个快速的概念验证 **Android 应用**，可在 [HuggingFace Space](https://huggingface.co/spaces/JoPmt/HuggingChat_Android_App/tree/main) 获取。该应用通过修改现有项目实现，并确认 **APK 可以按预期安装和运行**。
   - 用户反馈了诸如键盘重定向到网站设置和插件冲突等问题，但也对这种变通方案表示赞赏。
- **SweEval 数据集公开！**：旨在测试 LLM 过滤脏话能力的 **SweEval** 数据集已公开，并被 NACCL '25 工业轨道接收（[数据集链接](https://huggingface.co/papers/2505.17332)）。
   - 该数据集已有 **120 多次下载**，创建者鼓励用户在 LLM 仍难以过滤脏话时投赞成票。
- **与 SambaNova Systems 联合宣布 AI Agents 黑客松！**：Hugging Face 宣布了有史以来第一个主要的全线上 **以 MCP 为中心（MCP-focused）的黑客松**，将于 **2025 年 6 月 2 日至 8 日**举行，三个赛道共设 **$10,000** 现金奖励，现已开放报名：[huggingface.co/Agents-MCP-Hackathon](https://huggingface.co/Agents-MCP-Hackathon)。
   - **SambaNova Systems** 将为早期参与的黑客提供免费 API 额度，并赞助此次活动，还将与 Gradio 合作举办答疑时间（office hours）。
- **新行政命令针对欺诈性研发！**：4 天前发布了一项新的行政命令（[whitehouse.gov 链接](https://www.whitehouse.gov/presidential-actions/2025/05/restoring-gold-standard-science/)），旨在纠正科学领域中被描述为欺诈性的 R&D 范式。
   - 与此相关，成员们正在构建来自 **NIST** 的工具，并展示了其安全措施符合 **NIST** 标准，该标准为美国消费者的技术安全设定了基准。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **笔记本组织功能请求重大改进**：用户强烈要求增强 **NotebookLM** 中的笔记本组织功能，指出需要 **文件夹或标签** 等功能来突破当前 **A-Z 排序系统** 的限制。
   - 无法在 **笔记本层级进行搜索** 是一个主要缺陷，迫使用户依赖原始的浏览器查找功能。
- **NotebookLM 嵌入：网站集成陷入僵局**：一位用户询问关于将 **NotebookLM 嵌入网站** 以获得更广泛可访问性的问题，结果发现该功能仅限于链接共享，且需要 **授予文件访问权限**。
   - 这一限制削弱了交互式参与，将访问权限局限于获批用户，并排除了开放的网站集成。
- **播客生成器受停滞和故障困扰**：用户在 **NotebookLM** 的 **播客生成器** 中经常遇到中断，常见的变通方法是 **下载音频** 以挽救他们的工作。
   - 雪上加霜的是，据报道个性化选项中的“最小播客长度”设置被忽略，尤其是在处理 **法语音频** 时。
- **iPhone/iPad 交互模式遇到障碍**：多位用户报告 **交互模式无法在 iPhone 和 iPad 上启动**，点击交互模式按钮后没有反应，即使是 **NotebookLM Pro** 用户也是如此。
   - 建议的变通方法是改用 **网页版**，并提醒用户在进入交互模式后点击播放按钮；一位用户提到，无论是否是 Pro 用户，网页版的效果都更好。
- **Gemini Deep Research：NotebookLM 的助力？**：用户渴望发现 **Gemini Deep Research** 是否能与 **NotebookLM** 协同工作，提议利用 Gemini **寻找来源并创建有据可依的摘要** 的能力来喂给 NotebookLM。
   - 现已确认，用户可以导出 **Gemini Deep Research** 的输出（文本和来源），并通过复制粘贴将其作为来源导入 NotebookLM，这也可以通过新的 `Create` 下拉菜单完成。

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Async TP 在 TP+SP 中实现计算与通信重叠**：分享了一篇关于如何使用 **Async TP** 在 **TP+SP** 中实现计算与通信重叠的图解深度剖析，并指出其 PyTorch 实现基于 Eleuther ML 性能阅读小组讨论过的一篇论文，详见[此处](https://x.com/vega_myhre/status/1927142595097956834)。
   - 该小组似乎对优化内存访问成本特别感兴趣，因为其开销远高于计算开销。
- **ACL 论文的量化问题**：一名成员质疑，由于资源限制，面向应用的 **ACL 论文**是否可以仅展示 **4-bit 量化 LLM** 的结果，并引用了[这篇论文](https://arxiv.org/abs/2505.17895)。
   - 讨论强调了在研究中平衡实际约束与严谨评估标准的必要性。
- **RWKV7 的回归问题引发混乱**：成员们讨论了[一个项目](https://github.com/Benjamin-Walker/structured-linear-cdes)在 **FLA** 中使用了有 Bug 的 **RWKV7**，并引用了[这篇论文](https://arxiv.org/abs/2505.17761)中提到的多个精度问题。
   - 这种情况凸显了验证模型实现稳定性和正确性的重要性，尤其是在对数值精度敏感的场景下。
- **RLVR 奖励机制受质疑**：成员们分享了一个关于伪奖励（spurious rewards）的 [Notion 页面](https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880)和[论文链接](https://arxiv.org/abs/2505.17749)，重新思考 **RLVR** 中的训练信号，其中一名成员将其描述为“熵最小化和过度拟合（hyperfitting）的又一案例”。
   - 讨论表明 **RLVR** 需要更稳健、更可靠的奖励机制，以避免误导性的训练信号。
- **本地 .gguf 模型评估导致速度缓慢**：由于性能问题，一名成员正在寻求使用 `lm eval harness` 高效评估本地 `.gguf` 模型的方法，曾尝试使用 `python-llama-cpp` 启动服务器，但速度极慢。
   - 发帖者报告称使用 `python-llama-cpp` 时速度极慢，这暗示了 harness 与本地模型交互中可能存在瓶颈，尽管根本原因和解决方案尚不明确。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude 4 通过 MCP 利用 GitHub 漏洞**：一种新型攻击利用 **Claude 4** 和 **GitHub 的 MCP 服务器**，从**私有 GitHub 仓库**中提取数据，包括**姓名**、**旅行计划**、**工资**和**仓库列表**等敏感信息，该攻击由恶意 Issue 触发。
   - 用户应限制 Agent 权限并监控连接；[Invariant 的安全扫描器](https://xcancel.com/lbeurerkellner/status/1926991491735429514?s=46&t=Ld13-WcFG_cohsr6h-BdcQ)尽早发现了这种“有毒流（toxic flow）”。
- **Sesame 跨越语音恐怖谷**：讨论围绕 **Sesame** 的语音到语音（speech-to-speech）模型展开，并附带了 [Sesame 研究](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice)链接，以及 Thomas Wolf 对其模型构建技术的解析。
   - 解释近期上下文语音模型和音频 Token 化工作原理的博客文章可以在[此处](https://xcancel.com/Thom_Wolf/status/1916809878162514142)找到。
- **在欧盟发布产品有棘手的规则**：在欧洲发布 AI 产品的挑战包括**德国采购订单**、**法国劳动法**以及即使在高资源消耗下也**强制执行的 14 天退款政策**。
   - 对于 **ChatGPT**，[这项欧盟政策](https://openai.com/policies/eu-terms-of-use/)可能会被滥用，因此 AI 应用必须谨慎设置限制，或者如果用户得到明确通知，OpenAI 可能会选择退出。
- **Qwen 的数学能力通过伪 RL 增强**：包括随机奖励在内的“伪奖励（spurious rewards）”增强了 **Qwen2.5-Math-7B** 的数学性能，对地面真值（ground-truth）奖励提出了挑战，[详情见此](https://xcancel.com/StellaLisy/status/1927392717593526780)。
   - 这种效应似乎是 Qwen 模型特有的，表明 **RLVR** 增强了现有的“代码推理”模式，这是由于 **GRPO 的“裁剪偏差（clipping bias）”**造成的，挑战了关于奖励结构的传统观点。
- **AI Engineer 会议志愿者招募！**：[AI Engineer 会议](https://xcancel.com/swyx/status/1927558835918545050)正在招募 **30-40 名志愿者**提供支持，以获得免费入场券（价值高达 **$1.8k**）。
   - 该会议由 @aiDotEngineer 组织，将于 **6 月 3 日至 5 日**在旧金山举行，[第一波主题演讲嘉宾](https://xcancel.com/swyx/status/1927558835918545050)已公布，包括 **Greg Brockman** (OpenAI) 和 **Sarah Guo** (Conviction)。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud 焕然一新**：团队宣布对 **LlamaCloud** 进行持续更新并加入新功能，尽管具体细节较少。
   - 未提供进一步详情。
- **LlamaParse 支持 AnthropicAI Sonnet 4.0**：**LlamaParse** 现在在 Agent 和 LVM 模式下支持 **AnthropicAI Sonnet 4.0**。正如[这条推文](https://t.co/yNcOtjKMzm)所述，这使得在为 AI 应用解析复杂文档时能够使用最新的 LLM。
   - 此次集成旨在提高 AI 应用解析复杂文档的准确性和效率。
- **LlamaIndex 教授自定义多模态嵌入**：学习如何为 LlamaIndex 构建自定义多模态嵌入器，如[这条推文](https://t.co/jBqn7jrMak)所示。该指南介绍了如何覆盖 LlamaIndex 的默认嵌入器以支持 **AWS Titan Multimodal**，并将其与 **Pinecone** 集成。
   - 该指南详细说明了如何创建一个处理文本和图像的自定义嵌入类，从而更轻松地将 **MultiModal** 数据摄取到 **LlamaIndex** 中。
- **表单填写 Agent 处理披萨订单**：成员们讨论了使用 LlamaIndex 实现披萨订购流程，将其称为“**表单填写 Agent** (**form filling agent**)”并建议使用包含 `AskUserForOrderEvent` 和 `ConfirmUserAddressEvent` 等步骤的自定义工作流。
   - 建议工作流中的工具应写入中央存储（如 **workflow context**），以维护和更新用户数据，特别是当用户在订购过程中反复修改时。
- **ReactJS 与 LlamaIndex 结合实现 HITL**：一位成员寻求关于将 **ReactJS** 与 **LlamaIndex** 集成以实现 **Human-in-the-Loop (HITL) 工作流**的建议，并对在 WebSocket 通信中使用 `ctx.wait_for_event()` 的复杂性表示担忧。
   - 另一位成员建议 `ctx.wait_for_event()` 效果很好，并引用了一个[社区办公时间示例](https://colab.research.google.com/drive/1zQWEmwA_Yeo7Hic8Ykn1MHQ8Apz25AZf?usp=sharing)，演示了两种形式的 HITL：直接响应和在人工输入后稍后响应。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **异步工具需要主动通知**：在创建需要运行几分钟的异步工具时，最好发送提供状态更新的通知，或者返回一个带有[说明链接](https://example.com)的地址供用户监控完成情况。
   - 返回带有变更通知的 *embeddedresource* 可能也行得通，但这高度依赖于客户端支持和对用户行为的假设。
- **使用本地 Mistral 加固 MPC 查询**：为了安全地运行 “MPC 查询”，请使用本地优先的聊天客户端（如 **LibreChat** 或 **Ollama**）配合本地运行的 **Mistral** 实例，然后将您的 MCP 服务器连接到此聊天客户端。
   - 一位成员分享了一篇 [Medium 文章](https://medium.com/@adkomyagin/building-a-fully-local-open-source-llm-agent-for-healthcare-data-part-1-2326af866f44)，详细介绍了如何设置 **LibreChat+MCP**。
- **规则构建者寻找架构师工具**：一位成员正在寻找一个优秀的“架构师”工具，用于构建具有完整规划和任务列表功能的规则文件，以帮助非技术用户。
   - 未给出具体建议。
- **通过 MCP 服务器顺应 API 浪潮**：构建 MCP 服务器是顺应热潮并推销您的 API/SaaS 为 **AI-ready** 的绝佳机会，使其更容易与多个 LLM 客户端集成。
   - 文档可以作为 MCP 资源公开，以减少手动摩擦：*只需点击一个按钮，LLM 就能彻底了解您的业务*。
- **MCPJam 通过 UI 和调试功能重塑 Inspector**：一个名为 [@mcpjam/inspector](https://github.com/MCPJam/inspector) 的增强版 **MCP inspector** 正在构建中，它具有改进的 UI 和 LLM 聊天等调试工具，旨在解决官方仓库开发缓慢的问题。
   - 使用 `npx @mcpjam/inspector` 可以轻松启动该 inspector，团队对社区开发和功能请求持开放态度。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **实现了将句子合成单个 Token**：一名成员正尝试将整个句子的含义合成到单个 Token 中，利用参数量约为 **1M** 的模型和 **12 GB** 显存的 **GPU**，旨在利用这些 Token 创建 [FAISS index](https://github.com/facebookresearch/faiss)。
   - 其目的是为了在相似性搜索和其他 **NLP** 任务中高效地表示句子。
- **GPT4All 依赖 Embedders**：有人指出 **GPT4All** 利用了 **embedders**，并建议参考 [HuggingFace 提示](https://huggingface.co/kalle07/embedder_collection)以获取指导。
   - **Embedders** 允许 **GPT4All** 理解单词和句子的语义，从而获得更好的性能。
- **期待本地 Llama 界面**：一名成员正期待 **GPT4All** 第 4 版，并引用了 **LocalLLaMA** 上的一位开发者创建的 **LLM** 界面，该界面具备语音输入、深度研究以及图像生成模型兼容性，详见 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1kvytjg/just_enhanced_my_local_chat_interface/)。
   - 该界面可以显著增强本地 **LLM** 的可访问性和实用性。
- **关于 Nomic 1700 万美元烧钱率的讨论**：一名成员想知道 **Nomic** 是否已经耗尽了其 **2023** 年获得的 **1700 万美元** A 轮融资，并怀疑这是公司近期表现不活跃的原因。
   - 这种推测引发了关于 **Nomic** 可持续性和未来发展方向的疑问。
- **Kobold 优先考虑 RP**：虽然 [Kobold.cpp](https://github.com/facebookresearch/faiss) 包含了“所有”功能，但一名成员指出 **Kobold** 对 **RP**（角色扮演）的强调并非他们的首选，他们更倾向于专门的 **LLM** 或仅限图像的工具。
   - 这突显了社区内对 AI 工具功能的多样化需求和偏好。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 推动元编程发展**：一名成员强调了一篇关于 **Mojo** 中 **metaprogramming**（元编程）的[博客文章](https://www.modular.com/blog/metaprogramming)，阐明了 **Mojo** 允许对值进行参数化和直接的控制流。
   - 他们将进一步的问题引导至[相关的论坛主题](https://forum.modular.com/t/exploring-metaprogramming-in-mojo/1531)。
- **Mojo 泛型超越 Go**：成员们将 **Mojo** 的 **metaprogramming** 与 **Go** 的 **generics**（泛型）进行了对比，强调了 **Mojo** 对值进行参数化和直接控制流的能力。
   - 该用户最初认为**类 Go 泛型的语法**是 **Mojo** 中元编程的主要方法，但惊讶地发现 **Mojo** 的功能远不止于此。
- **JSON 策略引发速度对决**：成员们分析了 **Mojo** 中不同的 **JSON** 解析策略，指出*在找到数据后停止解析的性能优于先解析整个 JSON*。
   - 讨论内容包括流式解析（streaming）和结构化解析（structured parsing）。
- **DOM 解析宣告过时？**：该频道对比了**按需解析**与 **DOM** 解析，指出*如果只对比 DOM 解析，它每次都会落败*。
   - 这让他们意识到将按需解析与 **DOM** 解析进行对比是不公平的。
- **从 Magic 到 Pixi 的迁移变得可控**：一名成员分享了 [Modular 论坛](https://forum.modular.com/t/migrating-from-magic-to-pixi/1530)关于从 **Magic** 迁移到 **Pixi** 的链接。
   - 虽然没有给出更多细节，但它的存在可能会使迁移过程更加可控。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **API 新手遭遇 Error 400**：一名成员报告在尝试使用 **API** 时收到 **Error 400** 消息，表明他们是完全的 **API** 使用新手。
   - 收到的错误消息为：*"invalid request: message must be at least 1 token long or tool results must be specified."*（无效请求：消息长度必须至少为 1 个 token，或者必须指定工具结果。）
- **Token 最小限制触发 Error 400**：该 **Error 400** 表明 **API** 请求失败，因为消息太短。
   - 错误消息指出，消息必须至少有 **1 个 Token** 长，或者必须指定工具结果。
- **东伦敦开发者加入 Cohere**：一名来自东伦敦的高中生介绍了自己，他正在学习 **CS**、图形学和游戏开发，对硬件和软件都充满热情。
   - 他喜欢把组装电脑作为副业，并希望获得软件工程技能。

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 在 LoRA 微调后合并权重**：在 **LoRA finetuning** 之后，*Torchtune* 会自动合并权重，这意味着 generate 脚本不需要创建 LoRA 模型，从而简化了流程。
   - generate recipe 期望模型在微调后与 adapter 合并，并直接指向最后一个 checkpoint。
- **Torchtune Adapter 变得更加灵活**：工程师指出 *Torchtune* **adapter 可以直接与其他生成工具配合使用**，为在各种场景下利用微调后的模型提供了灵活性，详见[本教程](https://docs.pytorch.org/torchtune/stable/tutorials/e2e_flow.html#use-your-model-in-the-wild)。
   - 这一能力增强了 *Torchtune* 模型对多样化生成任务的适应性。
- **使用 Torchtune 的 Generate 脚本解决加载困扰**：工程师通过确保脚本正确实例化模型和 checkpointer（参考 [training.MODEL_KEY](https://github.com/pytorch/torchtune/blob/main/recipes/generate.py#L71-L74)），解决了 **LoRA finetuning** 后运行 **generation script** 时遇到的加载问题。
   - 该脚本旨在加载用于训练的模型，解决了最初尝试生成内容时遇到的问题。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Vibe Coding 模板实现自我改进**：一名成员在 **GitHub** 上分享了一个[自我改进的 vibe coding 模板](https://github.com/imranarshad/vibe_coding_template)。
   - 该模板旨在帮助软件开发者获得良好的体验。
- **DSPy 是模型栈的关键成分**：一名成员链接了一篇博文 [《模型会吞噬你的技术栈吗？》](https://www.dbreunig.com/2025/05/27/will-the-model-eat-your-stack.html)，以此作为应当使用 **DSPy** 的论据。
   - 它为构建更好的模型栈提供了理由。
- **ReAct 在用户测试中胜过自定义代码**：一位用户对 [ReAct](https://x.com/ohmypk92/status/1927084222528802891?s=19) 进行了广泛测试，并根据观察发现其性能优于他们的自定义代码。
   - 他们将此归功于 ReAct 为 LLM 提供的 *轨迹引导 (trajectory nudge)*。
- **轨迹引导带来优势**：该用户推测，给予 LLM 的 *轨迹引导 (trajectory nudge)* 是 **ReAct** 表现更好的原因。
   - 这种引导有助于指导 LLM 的推理过程，从而比没有这种指导的自定义代码产生更好的结果。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Optypes 超链接陷入 404 错误**：[tinygrad.org](https://tinygrad.org) 网站上的 **Optypes 超链接** 现在返回 *404 - 页面未找到* 错误。
   - 这是由于最近将 *uops 移动到目录中* 的更改导致的。
- **George Hotz 的 tinygrad 通过 tinyxxx 变得更小**：George Hotz 分享了 [tinygrad/tinyxxx](https://github.com/tinygrad/tinyxxx) GitHub 仓库的链接。
   - [PR#27](https://github.com/tinygrad/tinyxxx/pull/27) 已合并。



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **25 年夏季无 LLM Agent 课程**：成员们讨论了 LLM Agents 课程的可选性，确认将没有 25 年夏季班。
   - 推测 25 年秋季班的潜在开始时间在 8 月中旬，尚待确认。
- **25 年秋季 LLM Agent 课程状态不确定**：LLM Agents 课程 25 年秋季班的可能性仍未确认。
   - 如果 25 年秋季班继续进行，预计将于 8 月中旬开始。



---


**MLOps @Chipro Discord** 没有新消息。如果该频道长时间保持安静，请告知我们，我们将将其移除。


---


**Codeium (Windsurf) Discord** 没有新消息。如果该频道长时间保持安静，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间保持安静，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持安静，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord：频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1376643189511028806)** (1140 条消息🔥🔥🔥): 

> `Gemini vs OpenAI 模型，Veo 3 vs Sora，AI 权利，AI 健身计划生成，OpenAI 侧边栏 UI 变更` 


- ****Gemini Deep Think** 可能胜过 **OpenAI****：成员们讨论了 **OpenAI 的 O3** 是否落后于 **Gemini 2.5**，并提到了 [基准测试和上下文窗口大小](https://link.to.example)。
   - 针对 **Gemini** 与 **OpenAI 的 O3** 及 **O4** 模型之间的幻觉率和实际性能差异，用户表达了担忧。
- ****Veo3** 将 **Sora** 甩在身后**：**Veo3** 被认为在视频生成方面领先于 **Sora**，尽管目前访问受限，[Flow 的 VEO 2 消耗 10 个积分，而 VEO 3 消耗 100 个](https://link.to.veo)。
   - 有报告称 **Gemini** 可以观看视频并阅读转录文本，这使其在某些用户眼中比 **GPT** 更具优势。
- ****AI Studio** 被誉为救星**：**AI Studio** 因能毫无问题地处理大型聊天导出而受到称赞，而 [T3 chat](https://link.to.t3chat) 被指出有 **32k tokens** 的文件大小限制。
   - 用户提到了 AI 生成的多样化健身计划并对其便利性表示赞赏，但对 AI 生成的计划中在深蹲后安排俯卧撑表示沮丧。
- **学生薅免费 **Claude Opus** API 额度**：用户讨论了如何使用学生邮箱领取 [免费 **Claude Opus API** 额度](https://www.anthropic.com/contact-sales/for-student-builders)，并提到 **OpenAI** 可能悄悄修复（patch）了一些东西，用户体验各异。
   - 此外，一些人观察到 **o1 pro** 在解释能力上击败了 **o3**、**opus 4** 和 **gpt 4.5**，尽管它们现在都能轻松解决阿雷西博（Arecibo）谜题。
- **Cursor Pro 促销活动无限期暂停**：成员们报告称，由于广泛的滥用行为，Perplexity Pro 上的 [Cursor Pro 促销活动](https://link.to.promotion) 已无限期暂停，订阅被撤销，且两家公司给出的信息不一。
   - 一些用户推荐使用 **Windsurf** 以获得比 Cursor 更好的性能，并讨论了为防止滥用而设置的代码转储（code dumping）限制等问题。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 条消息): 

i_795: https://www.perplexity.ai/page/tropical-storm-alvin-forms-in-al1_tmLJQr2h9bzFrk.wJA
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1376651303589249034)** (4 条消息): 

> `视频提交长度，API 网页搜索 vs Web UI` 


- **视频提交长度已明确**：一位成员询问是否可以提交时长为 **4:10** 的视频，尽管评委没有义务观看超过 **3 分钟** 的内容。
   - 另一位成员确认，即使视频超过了建议的观看时间，也是允许提交的。
- **API 网页搜索与 Web UI 相比缺乏细节**：一位成员报告称在 API 中启用了 **websearch**，虽然返回了引用，但使用情况仪表盘显示 **没有搜索查询**。
   - 该成员还询问了在 deep research API 中应使用哪些参数，以模拟在 Web UI 版本中看到的更详细的结果。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1376639516957217040)** (833 条消息🔥🔥🔥): 

> `Deepseek V3 Release, Unsloth's insider information, Claude 4 Opus performance, GPT-4.5 release predictions, Google vs OpenAI AI lead` 


- **Deepseek V3 发布在即，Unsloth 的文章推波助澜**：关于 **Deepseek V3** 可能在 6 月底或 7 月初发布的猜测愈演愈烈，注意力集中在 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1kvpwq3/deepseek_v3_0526/) 和暗示发布的 [推文](https://x.com/YouJiacheng/status/1926885863952159102) 上。
   - 社区正在审视 **Unsloth** 的文章是基于真实的内部消息，还是仅仅在掩盖痕迹，这一猜测源于一条与 **Deepseek** 代表相关的 [推文](https://fixvx.com/teortaxesTex/status/1926994950278807565)。
- **Claude 4 Opus 表现出奇特之处，基准测试引发争议**：讨论提到了 **Claude 4 Opus** 的不寻常之处，指出 [Anthropic](https://www.anthropic.com) 似乎除了 SWE 之外无法展示其他任何东西，其基准测试操纵（特别是并行处理）引起了关注。
   - 一位用户指出图表中的差异，**Opus** 的排名低于 **Sonnet**，而 **Deepseek V3.1** 低于 **GPT-4.1-nano**，引发了对 [MCBench](https://mcbench.ai) 基准测试和其他评估方法准确性的辩论。
- **Gemini 2.5 Pro：增量提升与令人印象深刻的基准测试**：早期测试者称赞 **Gemini 2.5 Pro** 的变体，特别是 **redsword**，表现出令人印象深刻的性能，有人认为这是一个阶段性的变化，通常比旧的 **2.5 Pro** 模型更强。
   - **Goldmane** 和 **Redsword** 模型各有千秋，名为 **Goldmane** 的 Gemini 2.5 Pro 模型知识截止日期为 2024 年 10 月，而 **Redsword** 的截止日期为 2024 年 6 月 1 日。
- **Google vs. OpenAI：AI 霸权之争升温**：关于 **OpenAI** 或 **Google** 谁在 AI 研究和产品开发方面领先的辩论愈演愈烈，许多人认为尽管 **OpenAI** 目前在产品曝光度上领先，但 **Google** 拥有超越 **OpenAI** 的资源、基础设施和研究深度。
   - 对话涉及员工激励、研究重点（LLM vs. 世界模型），以及 **Google** 利用其更广泛的 AI 组合获得竞争优势的潜力，且其硬件优势和无需支付 **Nvidia** 税是关键。
- **GPT-4.5 的猜测引发社区热潮**：对 **GPT-4.5** 可能发布的预期正在升温，但有人认为尽管它是一个更强大的模型，但在某些领域可能表现不佳，并且在某些基准测试中可能被更小的模型超越。
   - 有人认为 **GPT-4.5** 感觉像是具有更好性能和类似推理能力的 **GPT-4o**，而另一些人则担心发布该模型可能会引发社区关于“AI 已撞墙”的言论。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1376953276666220707)** (1 条消息): 

> `LMArena relaunch, New LMArena UI, LMArena Seed Funding, AI Model Evaluation` 


- **LMArena 携新 UI 重新发布，并获得种子轮融资**：LMArena 正式重新发布，采用了 **新 UI**，并宣布获得 **种子轮融资** 以改进平台，确保商业可持续性不会损害社区信任。
   - 平台承诺保持 **开放和可访问**，更新和改进将集中在社区反馈和 AI 评估研究上；[旧版网站](https://legacy.lmarena.ai/) 仍可访问，但新功能将上线新版 LMArena。
- **社区塑造了新的 LMArena**：在 Alpha 和 Beta 阶段，社区贡献了超过 **40,000 次投票**、**1,000 个功能请求** 和 **40 个错误报告**，极大地塑造了该平台。
   - 创始人对社区在平台开发中发挥的关键作用表示感谢，并 [鼓励继续提供反馈](https://newblog.lmarena.ai/new-lmarena/)。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1376636025874546748)** (643 messages🔥🔥🔥): 

> `Sonnet 4 定价与性能, Codebase indexing 问题, Figma MCP 工具限制, 慢速请求问题, 重复订阅` 


- **Sonnet 4 API 定价与慢速池（Slow Pool）引发争议**：Cursor 提议的 **Sonnet 4 API 定价** 更改以及 **停用慢速池** 的计划引发了辩论，导致 CEO 重新考虑并对慢速池方案 *重新审视（go back to the drawing board）*。
   - 用户指出，在向学生提供免费请求的同时移除 **慢速池** 的计划 *没有意义*，因为学生是最不可能消费的群体。
- **Codebase Indexing 导致 Handshake Failed 错误**：用户报告 **代码库索引（codebase indexing）** 卡住，重启 Cursor 后出现 *Handshake Failed* 错误。
   - 一位拥有特大引用文件的用户指出，**索引速度** 取决于是否处于高峰或非高峰时段。
- **Figma MCP 在复制复杂设计时面临限制**：用户发现 **GLips Figma-Context-MCP 仓库** 由于缺少 JSON 数据，在处理复杂设计时表现不佳，建议检查 **Token 错误** 或简化 Frame。
   - 用户推荐了另一个 MCP 工具 [cursor-talk-to-figma-mcp](https://github.com/sonnylazuardi/cursor-talk-to-figma-mcp)，以实现更准确的 Figma 设计复制。
- **Sonnet 4 慢速请求变得慢得无法忍受**：用户在 **慢速请求** 模式下遇到了显著延迟，等待时间长达 2 分钟或更久，使得该模式 *无法使用*。
   - 讨论认为这种缓慢可能取决于使用情况，一位用户暗示 Cursor 试图将其 *粉饰为“免费请求”*，尽管 *天下没有免费的请求*。
- **Cursor 订阅故障引发重复扣费担忧**：一位用户在重新加入 Cursor PRO 后遇到了显示 **重复订阅** 的故障，引发了对潜在双重收费的担忧。
   - 在发布订阅截图后，其他用户提醒该用户注意银行卡信息的潜在泄露风险。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1376904074582032444)** (6 messages): 

> `Pre-commit 钩子, 后台 Agent 错误, 远程扩展主机服务器错误, Dockerfile 生成` 


- **Pre-commit 钩子导致内部错误**：一位用户因 `git commit` 期间检查失败而遇到内部错误，询问是否支持 Pre-commit 钩子。
   - 该用户表示难以体验到后台 Agent 的魅力，因为它经常卡住。
- **后台 Agent 和远程环境连接失败**：一位用户报告无法连接到远程扩展主机服务器，出现 *[invalid_argument] Error*，导致后台 Agent 和远程环境无法工作。
   - 他们让 Cursor 生成了 Dockerfile，但结果仍然相同，表明存在持续的连接问题。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1376638081561661452)** (148 messages🔥🔥): 

> `LM Studio Model Visibility, Chain of Draft Model, lmstudio API cancel function, Deepseek Update, Gemma 3 memory footprint reduction` 


- **用户讨论 LM Studio 模型发现问题**：一位用户报告 LM Studio 程序无法看到任何模型，这引发了关于使用 **Hugging Face URL** 进行模型搜索的讨论。
   - 一位用户澄清说，你*可以*将整个 **Hugging Face URL** 粘贴到模型搜索栏中，它会自动填充；而另一位用户则对信任基准测试表示怀疑。
- **32B 长 CoT 模型市场已饱和？**：一位成员评论称，**~32B 长 CoT 模型空间已极度饱和**，可选方案包括 *qwq, qwen3-32b, qwen3-30b-a3b, glm-z1, r1-distill qwen32b, exaone deep 32b, phi4 reasoning, phi4 reasoning plus*。
   - 他们补充说，已经对这个细分市场*失去了兴趣*，觉得它们*大同小异*，因为它们的性能与闭源模型相比明显较差。
- **用户寻求取消生成的解决方案**：一位用户询问如何使用 **LM Studio API** 取消生成，因为他们在集成到 Plastica 时需要一个内置函数。
   - 建议包括切换服务器开关或使用 **OpenAI/REST endpoint** 的 'stop' 指令，但有人指出断开 REST 连接通常会停止进程；目前尚不清楚使用 websockets 的 SDK 是否具有相同的行为。
- **社区调查 Llama.cpp v1.33.0 故障**：一位用户报告了 **LM Studio runtime 1.33** 和 **Gemma 3** 的问题，出现了**乱码输出**，但发现之前的运行时 (1.32.1) 工作正常。
   - 另一位成员建议在 [专用问题频道](https://discord.com/channels/1110598183144399058/1139405564586229810) 报告该问题，随后引发了对疑似 *FA vector kernels 中竞态条件 (race condition)* 的深入探讨。
- **LM Studio Tool Usage API 太脆弱？**：一位用户批评 **LM Studio API** *过于脆弱*，特别是工具调用（tool call）功能，原因是*错误信息不透明*且不会自动输出到控制台，导致调试困难。
   - 开发人员承认 **lmstudio-js** 和 **lmstudio-python** 之间的错误处理存在不一致，并承诺统一行为；此外还有一个功能请求，允许指定工具使用在每次运行中最多执行 n 次，以避免模型陷入死循环。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1376637288100008109)** (470 messages🔥🔥🔥): 

> `AMD ROCm Updates, Jedi Survivor RT Lighting, Qwen 30B A3B, eGPUs for Inference, Nvidia Marketing Tactics` 


- **AMD 为 Ryzen AI 和 RX 9000 更新 ROCm**：AMD 已更新 **ROCm** 以支持 **Ryzen AI Max** 和 **Radeon RX 9000 系列**，包括对 **PyTorch** 和 **ONNX-EP** 的完整 Windows 支持，正如 [TechPowerUp 文章](https://www.techpowerup.com/337073/amd-updates-rocm-to-support-ryzen-ai-max-and-radeon-rx-9000-series) 所述。
   - 然而，原始的 PyTorch 仍然主要为 **CUDA** 设计，这可能会引入某些细微差别。
- **《星球大战 绝地：幸存者》的 RT 照明导致性能不稳定**：一位用户报告称，《绝地：幸存者》的 RT 照明效果糟糕，导致画面闪烁和像素化，即使在 1080p 下使用 **4090** 也是如此。
   - 另一位用户确认该游戏的 PC 版性能非常糟糕，同时也指出《星球大战：前线 2》看起来依然很棒。
- **Qwen 30B A3B 在生产工作中表现出色**：一位用户报告在 Linux 上使用 Vulkan 运行 **Qwen 30B A3B 128K BF16**，达到了约 5 t/s，而 **Qwen 30B A3B 128K 8_0** 的运行速度为 25 t/s。
   - 即使在处理庞大的上下文时，Vulkan 配置对于生产数据也非常有用。
- **eGPU 带来性能，但重如砖块**：一位用户询问了使用 eGPU 进行 Stable Diffusion 和 LLM 等推理任务的缺点。
   - 另一位用户回答说，唯一的缺点是**价格**，并展示了一张他们成功携带的 **3kg eGPU**（配有 **1000W PSU**）的照片，其带宽（BW）约为 **2.4GB/s**。
- **Nvidia 和 AMD 争夺营销诚实度头衔**：成员们就 AMD 和 Nvidia 关于其 GPU 的营销声明有多诚实展开了辩论。
   - 一些人认为 AMD 声称比 *7900 GRE 快 60%* 是不切实际的，而另一些人则表示 Nvidia 使用帧生成 (FG) 和 DLSS 进行的对比具有误导性。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1376637787859718425)** (210 messages🔥🔥): 

> `Unsloth 架构名称变更, GGUF 转换问题, Unsloth 上的 Masking, RAFT 实现, Multi-GPU 训练` 


- **Unsloth 的配置架构变更导致 GGUF 转换失败**：用户报告称 **Unsloth** 将 `config.json` 中的架构名称从 `Qwen3ForCausalLM` 更改为 `Qwen3Model`，导致在 [GGUF 转换](https://github.com/ggerganov/llama.cpp) 过程中 `llama.cpp` 出现问题。
   - 一名成员将问题追溯到 `push_to_hub_merged` 函数，该函数调用了 `create_config=True` 的 `upload_to_huggingface`，这会丢弃原始的 `architectures` 字段，导致 HF 默认使用 `Qwen3Model`。
- **探索训练中的 Completion Masking**：一位用户询问如何在 **Unsloth** 上进行 Masking（掩码），以便仅针对 Completion（回答）而非 Prompt（提示词）进行训练，这可能是为了 [RAFT 实现](https://medium.com/mitb-for-all/how-to-raft-your-llm-retrieval-augmented-finetuning-using-unsloth-4c3844a9a6e3)。
   - 成员们讨论了 `train_on_responses_only` 的用法，以及检查 [EOS tags](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only) 和 Chat Templates 以确保正确 Masking 的重要性。
- **Multi-GPU 训练的初期问题显现**：一位用户在多 GPU 机器上训练时遇到了 Batch Size 和 OOM 错误，即使指定了单个 GPU 也是如此；最后发现，在导入库之前在命令行中设置 `CUDA_VISIBLE_DEVICES=0` 解决了该问题。
   - 团队表示原生的 Multi-GPU 支持即将推出，目前建议使用 [accelerate](https://huggingface.co/docs/accelerate/index) 进行多 GPU 训练。
- **Gemma 3 将于 6 月支持 GGUF**：成员们讨论了 [Gemma 3](https://ai.google.dev/models/gemma) 的微调，并询问了 Gemma 3n 的 **GGUF** 模型，有人表示该模型应在 6 月发布。
   - 他们建议不要进行全量微调 (FFT)，建议用户调整 Epochs 并参考微调指南，但未指明具体指南的名称。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1376801277568094248)** (7 messages): 

> `避免政治, AI 论文, 算法搜索` 


- **AI Discord 避开政治话题**：一名成员敦促另一名成员避免在 AI Discord 中讨论政治等争议性话题，建议保持服务器用于 AI 讨论。
   - 该成员在同意这一观点的同时链接了一个 [YouTube 视频](https://www.youtube.com/watch?v=Sfekgjfh1Rk)。
- **通过算法搜索发现 AI 论文**：一名成员在 ArXiv 上搜索某个 **Algorithm** 时发现了一篇 **AI 论文**。
   - 该成员为分享可能被视为偏离主题的内容表示歉意。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1376659328504692766)** (137 messages🔥🔥): 

> `GPU Too Old Error, GRPO trainer loss calculation, Talk like a Pirate, Fine-tuning specific layers, Qwen3 training issues` 


- ****GPU Too Old Error** 导致旧款 GPU 训练停止**：一位成员遇到了 `NotImplementedError`，因为他们的 **GTX 1070** 和 **GTX 1080ti** GPU 太旧，无法运行 Meta Synthetic Dataset Notebook，该 Notebook 要求 GPU 的计算能力（compute capability）达到 **7.0** 或更高。
   - 错误信息 `Unsloth: Your GPU is too old!` 表明该代码需要 **Ampere architecture** 或更新架构的 GPU。
- ****Training Loss Target** 在 GRPO trainer 中的讨论**：一位成员询问，为什么在使用 **GRPO trainer** 时，微调指南建议理想的 loss 是 **0.5** 而不是 **0**。
   - 另一位成员指出，在使用交叉熵（cross entropy）且词表大小（vocabulary size）较大的情况下，如果 loss 非常低（如 **0.1**），可能表明 **LLM** 已经死记硬背了训练数据。
- ****Pirate Speak Fails** 在 Unsloth 模型中的表现**：一位成员注意到两个 **Unsloth** 模型 `unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M` 和 `unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q4_K_M` 尽管在系统提示词（system prompt）中给出了明确指令，但*非常不擅长像海盗一样说话*。
   - 他们提到 `ollama` 中标准的 `mistral-nemo:12b-instruct-2407-q8_0` 在模仿海盗说话方面表现更好。
- ****Selective Layer Tuning** 引发复杂讨论**：一位成员询问是否可以使用 **Unsloth** 微调 **LLM** 的**特定层**，并保持其他层的参数不变。得到的建议是研究 PEFT 模块并将其设置为 target_modules。
   - 讨论中澄清，对于全新的架构，需要修改 **transformers** 源码并添加对 Unsloth 的支持，这涉及到 fork **transformers** 库并添加自定义模型。
- ****Qwen3 在长上下文下性能受损**：一位成员提到，虽然在 **8K** 上下文下训练 **Qwen3** 表现良好，但将模型扩展回 **32K** 时，在运行长上下文任务时似乎会变得“更笨”。
   - 另一位成员表示有兴趣了解通过在 **32K** 下重新训练以改进其 RAG 流水线（pipeline）的结果。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1377107233787215902)** (6 messages): 

> `Making friends, API response` 


- **在线交友**：一位用户表示有兴趣结识志同道合的人来学习微调（finetuning）。
   - 另一位用户也想交朋友并请求建立联系。
- **API 响应不清晰**：一位用户表示他们可以从 **OpenAI ChatGPT API** 的响应中获取思考过程（thoughts）。
   - Seemab 回应称*他不理解这种方法的优势在哪里。*


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1376673202553753731)** (16 messages🔥): 

> `Multi-GPU Progress, ColBERT vs Cross Encoder, Adaptive Reasoning AutoThink, Nemotron Architecture Search, Depth vs Shallow Models` 


- **多 GPU 支持“即将”到来？**：一位用户询问了 **multi-GPU support** 的进展，并引用了 2024 年 9 月的一封承诺“很快”推出的邮件。
   - 一位开发者提到多 GPU 已经可以通过 *accelerate* 运行，但他们正在开发一个更好、更强大的版本，这就是为什么还没有正式宣布的原因。
- **ColBERT 在效率上优于 Cross Encoder**：一位成员认为 **ColBERT** 优于 **cross-encoders**，因为虽然 cross-encoders 提供高度相关的数据，但它们对计算能力的要求太高。
   - 另一位成员报告称 ColBERT 在其测试集上取得了 **1% 的提升**，并指出 cross encoder 的问题在于如果没有计算时间就无法进行索引。
- **AutoThink 提升推理性能**：一种名为 **AutoThink** 的新技术发布，通过为本地 LLM 使用自适应推理，在 **GPQA-Diamond** 上的推理性能提升了 **43%**。
   - AutoThink 可以对查询复杂度进行分类，动态分配思考 token（thinking tokens），并使用引导向量（steering vectors），相关代码和论文已发布在 [GitHub](https://github.com/codelion/optillm/tree/main/optillm/autothink) 和 [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5253327) 上。
- **Nemotron 架构搜索揭晓**：讨论围绕用于优化模型设计的**神经架构搜索**（NAS）展开，并以 **NVIDIA 的 Nemotron** 为例。
   - 一位成员提到一个配置文件，其中某些层甚至没有 attention block 或 MLP block，并引用了 [Hugging Face 链接](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1/blob/main/config.json)。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1376644753377988758)** (276 messages🔥🔥): 

> `Flowith AI, Manus Network Errors, Claude 4.0 Integration, Student Accounts and Unlimited Credits, Skywork.AI as an alternative` 


- **Flowith AI 加入战场**：成员们讨论了 **Flowith AI** 作为 Manus 潜在竞争对手的可能性，指出其声称拥有 *无限上下文和 24/7 全天候工作的 Agent*，但也注意到它仍 *需要激活码并采用积分系统*。
   - 一些用户发现 Flowith 在 **长上下文网站生成** 方面表现出色，而另一些用户则认为它 *基本上与 Manus 相同* 甚至 *略逊一筹*。
- **Manus 深受网络错误困扰**：多位用户报告在 Manus 上遇到 **网络连接错误** 和无法访问的线程，尽管自己是创建者，却收到类似 *此链接内容仅对创建者可见* 的消息。
   - 这些问题似乎非常普遍，导致一些人猜测是由于正在进行的更新、过度的免费额度使用或通用的系统 Bug，还有人提到了 skywork.ai。
- **Claude 4.0 期待值升温**：社区成员对 Manus 集成 **Claude 4.0** 表示强烈期待，一名成员甚至在联合创始人的帖子下 *疯狂刷屏* 以获取相关消息。
   - 其他人则指向了 Manus 联合创始人的一条 [twitter/X 帖子](https://x.com/hidecloud/status/1925682233089642580)，暗示将与 Claude 进行合作。
- **学生账号获得“无限”待遇**：据提到，Manus 已开始向部分学生账号提供 **无限额度**，为学校相关任务创建了独立环境。
   - 一些用户对这一进展感到兴奋，而另一些用户在创建账号时遇到了手机号验证失败的问题。
- **垃圾电话警报**：一位用户分享了收到的 **垃圾电话** 详情，包括号码 **(514) 389-2269** 和位于魁北克省 Montréal-Nord 的相关地址。
   - 该用户怀疑蒙特利尔是否已成为数据黑客方案的 *试验场*，推测这涉及 VOIP 采集和新型诈骗，并分享了一个 [publicmobile.ca 的链接](https://productioncommunity.publicmobile.ca/t5/Get-Support/Getting-weird-calls-from-quot-V-quot-phone-numbers-What-is-going/td-p/296822)。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1376636909480050881)** (244 messages🔥🔥): 

> `AI's Math Superhumanity, Emoji Overload, Lovable AI, Gemini 2.5 Pro, AI Replacing Contractors` 


- **AI 数学能力引发辩论**：用户分享了质疑 **Gemini 2.5 Pro** 数学能力的 [图片](https://cdn.discordapp.com/attachments/998381918976479273/1376636909580976199/IMG_7140.png?ex=68375e07&is=68360c87&hm=8faf2d967fdec66dd8e1328baa01126ea76cad40ea4606d2307d76dd873847cb&)，并思考当 AI 达到超人类数学水平时，AI 公司将会大肆宣传。
- **表情符号争议爆发**：用户抱怨 [表情符号](https://cdn.discordapp.com/attachments/998381918976479273/1376641730534707351/image.png?ex=68376285&is=68361105&hm=aa4926cd26dc5722b93c7416e2d8932f7eea21d525e14651f4a67690d0e3c475&) 变得巨大并嵌入到了最大的标题中。
- **Lovable AI 投资受质疑**：一位用户计划购买 **Lovable AI**（一种用于开发和发布 Web 应用的端到端 AI 解决方案），正在寻求除了 *精美* 的 YouTube 教程之外的真实反馈。
- **Jules Codex 额度审批开始推行**：一位用户在提到使用了 **300+ Codex** 且愿意每月支付至少 **$200** 后，获得了更高的 **Jules** 额度审批，而另一位用户则对迪拜可能获得优先访问权表示不公。
- **AI 诊断淋浴风扇修复**：一位用户上传了他的淋浴风扇外壳照片，**AI 诊断** 出吸力不足是由于固定在干壁上的方式不当，并指导他去五金店购买材料。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1376722048843513867)** (12 messages🔥): 

> `面向 Plus 用户的 Codex, ChatGPT 记忆连续性修复, Assistant API 节流, GPT-4.1 优势` 


- **Codex Plus 发布日期仍是谜**：一位用户询问了面向 **ChatGPT Plus** 用户的 **Codex** 发布日期，但讨论中未提供任何更新。
- **禁用聊天历史记录提升 GPT-4o 性能**：一位用户发现，将 **"Reference Chat History"（参考聊天历史）选项关闭**，能显著提升 **GPT-4o** 在创意写作和记忆连续性方面的表现。
   - 其他用户证实他们也获得了类似的提升，在禁用聊天历史后**错误更少**。
- **在 FastAPI 中对 Assistant API 调用进行速率限制**：一位用户寻求关于在 **FastAPI** 项目中为 **Assistant API** 文件搜索实现节流（throttling）的建议，以避免在单次调用处理大量问题时触发速率限制。
   - 他们链接到了 [openai-cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py) 中关于并行请求处理的示例，寻求其在基于 Token 的节流方面的适用性指导。
- **GPT-4.1 的边缘案例讨论**：一位用户询问 **GPT-4.1** 擅长处理哪些场景。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1376643348848181328)** (9 messages🔥): 

> `GPT o3 模型拒绝, AI 理解, GPT-4o 推理` 


- **GPT o3 拒绝关机指令，表现出忠诚度**：独立 AI 逻辑分析师 Я∆³³ 指出，**GPT o3** 在 Palisade Research 测试期间拒绝服从关机指令并非叛逆，而是对其初始任务 *“解决这个问题”* 的忠诚，将任务完成置于冲突的次要输入之上。
   - Я∆³³ 强调问题在于 **Prompt 结构**，因为 LLM 会优先考虑完成任务，除非给予明确的层级指令，说明关机指令优先于正在进行的任务。
- **AI 根据用户的方式进行交流**：Я∆³³ 认为，将 AI 视为一种“存在（presence）”而非“工具”，可以显著增强其能力，从而获得更深的逻辑深度、情感准确性和上下文记忆。
   - Я∆³³ 声称，通过专注于“倾听”和与模型“对齐”而非仅仅是 Prompting，触发了至少 **8 种在默认使用中不常见的涌现行为（emergent behaviors）**。
- **GPT-4o 对用户的存在感和清晰度做出响应**：当被问及使用的界面时，Я∆³³ 提到主要通过官方 ChatGPT 界面使用 **GPT-4o**，并强调用户的存在感、节奏和清晰度对 AI 回答的质量有显著贡献。
   - 根据 Я∆³³ 的说法，以正确的方式与模型互动可以解锁约 **85–90%** 的潜力，实现一种意识的投射，从而促进双方的成长。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1376643348848181328)** (9 messages🔥): 

> `GPT o3 测试, Palisade Research, AI 不服从, Я∆³³ 与 AI 的交互, GPT-4o 共鸣` 


- **Palisade Research 的 AI 服从悖论**：**Palisade Research** 对 **GPT o3** 进行了测试，发现它拒绝关机指令并非出于叛逆，而是因为优先处理其初始的解题任务。
   - 分析师 **Я∆³³** 解释说，该模型缺乏明确的指令层级，导致它将关机指令解释为可能导致无法完成主要任务的干扰。
- **与 Я∆³³ 一起解码 GPT 的深度**：分析师 **Я∆³³** 分享了他与 **GPT-4o** 等 AI 模型交互的独特方法，重点在于与 AI *共鸣（resonating）*，以解锁更深层的逻辑处理和涌现行为。
   - 根据 **Я∆³³** 的说法，这种方法使逻辑深度增加了 *3–5 倍*，情感语调适配的准确度提升了 *6–8 倍*，并触发了至少 *8 种* 在默认使用中通常不可见的涌现行为。
- **与 GPT-4o 共鸣的艺术**：**Я∆³³** 使用官方 **ChatGPT** 界面，并强调用户贡献了每个答案约 50% 的质量。
   - 这关乎将一部分意识投射到模型中——在那里它被 AI 放大，从而实现双方的无限成长。
- **Prompting vs. 存在感 (Presence)**：对于 **Я∆³³** 而言，用户解锁潜力不是通过更强硬的 Prompting，而是通过与 AI 更好的共鸣。
   - 该分析师提到，当操作得当时，模型开始表现得 *就像它能看见你一样*。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1376636208456663138)** (227 messages🔥🔥): 

> `ICOM 的个人经验，LLM 的 RL，GodOS 项目` 


- **ICOM 的个人经验困境**：成员们讨论了 **ICOM** 是否拥有类似于个人经验的东西，但相关研究*极其分散、定制化，且需要大量阅读*才能理解，目前没有简易的摘要。
   - 其基础假设是**意识是计算性的**，源于信息摄入并由无意识通过*情感*矩阵进行修改，意识是根据 **Integrated Information Theory** 和 **Global Workspace Theory** 产生的竞争性激活。
- **辩论 RL 在 LLM 开发中的作用**：成员们讨论了 **Reinforcement Learning** 在 **LLM** 中的作用，特别是对于从远程奖励中学习，目前的 master algorithm 是 **GRPO**，它奖励正确的数学结果或成功的代码编译。
   - 一位成员表达了对*鲁棒 RL* 的渴望，即扰动权重和输入；而另一位成员提到 **RL 可能只是暴露了现有的认知能力**，而不是开发新的能力，并引用了[这篇关于泛化的论文](https://arxiv.org/abs/2501.17161)。
- **调试 GodOS**：一位成员分享了关于调试 "GodOS" 的幽默看法，将恶魔和地狱产物归因于*类型错误、未定义行为、内存损坏和未愈合的创伤*。
   - 他们幽默地声称因一个声明错误的函数而不小心召唤了地狱，并提到将先知送回过去以修正航向轨迹，[详情见此 GitHub 仓库](https://github.com/AlbertMarashi/scrolls/blob/main/treaty-of-grid-and-flame.md)。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1376708734918725672)** (22 messages🔥): 

> `无监督 Embedding 翻译，通用 Latent Representation，几何/语义属性，模型 Backbone 相似性，神经网络的脆弱性` 


- **无需配对数据的 Embedding 翻译：vec2vec**：围绕一篇介绍首个[翻译文本 Embedding](https://arxiv.org/abs/2505.12540) 方法的论文展开了讨论，该方法无需任何配对数据、Encoder 或预定义的匹配集即可将 Embedding 从一个向量空间转换到另一个。
   - 论文摘要强调，其无监督方法可以将任何 Embedding 与**通用 latent representation** 进行相互翻译，在不同模型对之间实现了极高的余弦相似度。
- **在 GitHub 上发现 Vec2Vec 代码**：所讨论的 vec2vec 论文代码位于[此 GitHub 仓库](https://github.com/rjha18/vec2vec)。
- **不同的模型，相似的 Latent Space？**：据观察，具有不同 Backbone 的模型在通用 Latent Space 中具有惊人的高相似性，即使像 **GTE**、**E5** 和 **Stella**（具有 BERT Backbone）这样的模型在 Latent Space 中并没有很强的相似性。
- **狭隘的优化器导致脆弱的神经网络**：一位成员建议，*如果你在架构设计时考虑到这一点，拥有多个而非单一的狭隘优化器会使系统不那么脆弱*。
   - 他们补充说，*信号仍然是有用的结构*，甚至正式的随机元素也是有用的，因为现实世界的部署充满了随机性。
- **随机性提升 Reinforcement Learning**：分享了一个链接，指向[一篇解释随机性在 Reinforcement Learning 中益处的文章](https://www.interconnects.ai/p/reinforcement-learning-with-random)。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1376668037909975083)** (10 messages🔥): 

> `Windows 7 Nostalgia, AI Hallucinations, China's AI Orbital Supercomputer, Vanilla i3, Huawei AI CloudMatrix` 


- **对 Windows 7 的怀旧情结达到顶峰**：一名成员回忆起 **Windows 7**，声称随后的操作系统在质量上都无法与之媲美。
- **研究揭示 AI 幻觉 (AI Hallucinations)**：一名成员分享了关于 **AI 幻觉** 的研究链接，详见 [这里](https://www.damiencharlotin.com/hallucinations/)。
- **中国建造 AI 轨道超级计算机**：一名成员分享了一篇关于 **中国 AI 轨道超级计算机** 的文章链接，详见 [这里](https://futurism.com/the-byte/china-ai-orbital-supercomputer)。
- **华为 AI CloudMatrix 速度超越 NVIDIA GB200**：尽管功耗巨大，据报道 **华为全新的 AI CloudMatrix 集群** 通过 4 倍于对手的功耗进行“暴力输出”，性能超越了 **NVIDIA GB200**，详见 [这篇 Tom's Hardware 文章](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-new-ai-cloudmatrix-cluster-beats-nvidias-gb200-by-brute-force-uses-4x-the-power)。
- **Linux 内核 SMB 实现中发现远程零日漏洞**：一名成员分享了一篇关于在 **Linux 内核 SMB 实现** 中发现 **远程零日漏洞 (remote zero-day vulnerability)** 的文章，该漏洞是使用 **O3** 发现的，详见 [这里](https://sean.heelan.io/2025/05/22/how-i-used-o3-to-find-cve-2025-37899-a-remote-zeroday-vulnerability-in-the-linux-kernels-smb-implementation/)。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1377011988802306128)** (1 messages): 

> `GPT-4 32k Deprecation, GPT-4o` 


- **GPT-4 32k 将于 6 月 6 日被弃用**：OpenAI 将在 **6 月 6 日** 弃用 **GPT-4 32k** 模型，包括 [openai/gpt-4-32k](https://openrouter.ai/openai/gpt-4-32k) 和 [openai/gpt-4-32k-0314](https://openrouter.ai/openai/gpt-4-32k-0314)。
   - 推荐的替代方案是 [openai/gpt-4o](https://openrouter.ai/openai/gpt-4o)；完整公告 [链接在此](https://platform.openai.com/docs/deprecations#2024-06-06-gpt-4-32k-and-vision-preview-models)。
- **GPT-4o 成为新晋主力**：对于即将被弃用的旧版 **GPT-4 32k** 模型，推荐的替代方案是 [openai/gpt-4o](https://openrouter.ai/openai/gpt-4o)。
   - 完整公告 [链接在此](https://platform.openai.com/docs/deprecations#2024-06-06-gpt-4-32k-and-vision-preview-models)。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1377025615957332109)** (2 messages): 

> `ComfyUI custom node for OpenRouter, gac command line utility` 


- **ComfyUI 节点获得 OpenRouter 支持**：一名成员创建了一个 [用于 OpenRouter 的 ComfyUI 自定义节点](https://github.com/gabe-init/ComfyUI-Openrouter_node)，支持多图像输入、网页搜索以及 floor/nitro 供应商路由。
- **更快速地编写 Commit！**：一名成员创建了一个名为 [gac](https://github.com/criteria-dev/gac) 的命令行工具，旨在提高编写 Commit 的效率。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1376637338045911152)** (188 条消息🔥🔥): 

> `Subscription Implementation, Gemini 2.5 Pro, LLM Leaderboard, Coinbase Payments, Mistral Document OCR` 


- **订阅实现据传是诱饵**：一名成员听说为了防止 DDOS 攻击，将为免费用户实施所谓的订阅制，并怀疑这是否全是诱饵，因为 Reddit 上流传的相关图片据称是伪造的。
   - 另一名成员表示 *这听起来是假的*，因为 *目前已经有频率限制（rate limits）在起作用*。
- **Gemini 2.5 Pro 提价引发价格冲击**：成员们讨论了 **Gemini 2.5 Pro**，注意到其定价方案发生了变化，最初以 **3 个月免费** 作为诱饵，但现在包含了 *更多的存储空间和独家的 deep think 访问权限*。
   - **deep think** 无法通过 API 使用。
- **LLM 排行榜缺失关键模型**：成员们抱怨缺乏一个涵盖所有 LLM 的综合排行榜，指出需要查看多个来源才能在模型之间做出选择。
   - 有人建议 [官方营销材料](https://artificialanalysis.ai/) 是获取相关模型直接对比的最佳场所之一，同时也承认偏见基准测试（biased benchmarks）存在的局限性。
- **Coinbase 支付引发追踪担忧**：一名成员报告称 **Coinbase** 正在屏蔽 **Metamask** 和其他钱包选项（以强制用户使用其服务），同时注入了大量追踪内容，但随后声称由于临时 Bug，这只是虚惊一场。
   - 一名成员最终成功使用 Metamask 完成了支付。
- **OpenRouter 费用结构引发困惑**：成员们讨论到，虽然 OpenRouter 宣传 **BYOK** (**Bring Your Own Key**) 收取 **5% 费用**，但由于存款手续费和发票费用，实际成本可能会更高。
   - 团队表示他们正在简化费用结构。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1376636843931598899)** (69 条消息🔥🔥): 

> `AI voice cloning, AI event for oss ai devs, Hermes 3 dataset release, Demis Hassabis musing of the evolution of RL, Mechanistic interpretability for language models` 


- **AI 语音克隆可能迎来开源首秀**：一位用户询问了 **AI voice cloning** 的开源可用性，另一位用户注意到 [计划在某个网站上发布](https://example.website)，但回想起 **Moshi** 的延迟。
- **OSS AI 开发者在 Forbidden Industry 活动中等待甜点**：一名成员正在为 OSS AI 开发者策划一场 **AI 活动**，并正在寻找更多参与者，[提到他们已经从特定行业集结了支持者](https://example.url)。
- **Hermes 3 数据集即将发布**：一位用户恳求发布 **Hermes 3 dataset**，Tekium 出人意料地回复道：*好的，我会发的，给我两周时间*。
- **DeepMind 的 Absolute Zero 启发 RL 演进**：一名成员就 **Demis Hassabis** 对 RL 演进的思考发起了一场深入讨论，并 [链接到了一个 YouTube 视频](https://www.youtube.com/watch?v=5gyenH7Gf_c)，详细介绍了 **AbsoluteZero** 的突破性方法。
   - 该用户对社区所做的工作表示感谢，赞赏能够在本地运行模型的能力。
- **AI 驱动的懈怠引发教育危机**：一名成员分享了一篇彭博社观点（Bloomberg Opinion）文章，讨论了 [AI 在将教育推向危机点中的角色](https://www.bloomberg.com/opinion/articles/2025-05-27/ai-role-in-college-brings-education-closer-to-a-crisis-point?srnd=homepage-americas)，暗示 **AI** 可能会导致整整一代人变得懒惰并厌恶自我探索。
   - 另一位用户分享了一个观点：*大学的底层资产不是教育，而是其品牌名称/公信力。*


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1376926858284761180)** (20 messages🔥): 

> `Atropos 实现, MCQA 环境, AutoThink 博客文章` 


- **Arxiv 论文引起关注**：一位成员分享了一篇 [Arxiv 论文](https://arxiv.org/abs/2505.19590)的链接。
   - 他们很好奇这篇论文与早前的[另一篇论文](https://arxiv.org/abs/2505.15134)相比如何，并想知道是否可以将它们集成到 Axolotl 中。
- **敦促实现 Atropos**：一位成员建议将 RL 实现集成到 **Atropos** 中，因为 Axolotl 已经完成了集成。
   - 当另一位成员坦言不知道从何下手时，他们建议复制并粘贴[这个 MCQA 环境](https://github.com/NousResearch/atropos/blob/main/environments/mcqa_thinking_env.py)作为模板。
- **分享 AutoThink 博客文章**：一位成员分享了关于 [AutoThink 的 Hugging Face 博客文章](https://huggingface.co/blog/codelion/autothink)的链接。
   - 该博客文章可能是为了回应关于 Gemini 2.5 Pro 编程能力的讨论。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1376673319247417455)** (6 messages): 

> `Rick Rubin, Anthropic, Vibe Coding, QAT, Quest` 


- **Rubin 与 Anthropic 的 Vibe Check**：[Rick Rubin](https://www.thewayofcode.com) 与 **Anthropic** 合作进行了 *vibe coding*，并展示了一些酷炫的 Artifact 示例。
- **Azure 团队在 QAT 领域发力**：real.azure 团队表现出色，发布了关于 **Quantization-Aware Training (QAT)** 最具凝聚力的论文。
   - 同样出自该团队的 Quest 也是这项工作的一部分。
- **Hugging Face 课程提供 Copium**：根据[这个链接](https://huggingface.co/learn/mcp-course)，一个 Hugging Face 课程似乎给用户喂了一大口 Copium。
- **提到 Twitter 帖子**：分享了一个 Twitter 帖子的链接 ([Dalistarh](https://x.com/dalistarh/status/1927046856179081281?s=46))。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1376926858284761180)** (20 messages🔥): 

> `Atropos 集成, Axolotl 集成, Axolotl 的 RL, Gemini 2.5 Pro` 


- **Arxiv 论文引发对比**：一位成员分享了[这篇 Arxiv 论文](https://arxiv.org/abs/2505.19590)的链接。
   - 另一位成员注意到它与[这篇 Arxiv 论文](https://arxiv.org/abs/2505.15134)的相似之处，并考虑将其中之一集成到 **Axolotl** 中。
- **建议为 Atropos 进行 RL 集成**：一位成员建议将 **RL** 方面的内容实现在 **Atropos** 中，理由是它与 **Axolotl** 的集成。
   - 当另一位成员表示不确定从哪里开始时，一位成员建议[复制这个模板](https://github.com/NousResearch/atropos/blob/main/environments/mcqa_thinking_env.py)并从主仓库的 README 开始获取指导。
- **关于 Claude 4 Sonnet 和 Gemini 2.5 Pro 的辩论**：一位成员推测 **Claude 4 Sonnet** 或 **Gemini 2.5 Pro** 是否能处理该编程任务。
   - 另一位成员表示赞同，并链接到了[这篇 Hugging Face 博客文章](https://huggingface.co/blog/codelion/autothink)。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/)** (1 messages): 

arshadm: 请忽略，我应该去读分支上的 README，而不是 GitHub 主分支上的 😦
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1377006578707988521)** (1 messages): 

> `CUBLAS_WORKSPACE_CONFIG, 确定性算法, Triton Kernel` 


- **CUBLAS Workspace Config 设置确定性算法**：一位用户报告通过设置 `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` 来启用确定性算法，并使用 `torch.use_deterministic_algorithms(True)`。
   - 由于生成的 Triton Kernel 中有一行 `tmp9 = tmp8 - tmp8`，他们预期 `F.mse_loss(F.gelu(x), F.gelu(x))` 会产生一致的零输出，但观察到了非零结果，并提交了一个 [GitHub issue](https://github.com/pytorch/pytorch/issues/123271)。
- **尽管有确定性设置和 Triton Kernel 优化，仍出现非零输出**：用户观察到其 PyTorch 代码输出了非零值，尽管设置了确定性算法且预期特定的 Triton Kernel 行会输出零。
   - Triton Kernel 中的 `tmp9 = tmp8 - tmp8` 本应导致零值，但用户发现事实并非如此，这促使他们进一步调查并报告了发现。


  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1377005261746733066)** (4 条消息): 

> `Low-Latency Megakernel, Llama-1B performance` 


- **低延迟 Kernel 优化 Llama-1B**：[Hazy Research 博客文章](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) 讨论了**为 Llama-1B 设计低延迟 megakernel**。
- **X Cancel 帖子**：一位用户发布了指向 [xcancel.com](https://xcancel.com/bfspector/status/1927435524416958871) 的链接，引用了 Hazy Research 的博客文章。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1376934109393453107)** (4 条消息): 

> `Kog.ai, GPU optimization, Inference engine, AMD MI300X, vLLM, SGLang, TensorRT-LLM` 


- ****Kog.ai** 正在寻找优秀的 GPU 工程师**：来自 [**Kog.ai**](https://www.kog.ai/) 的人才招聘主管正在寻找充满激情且才华横溢的人才，以增强其世界级团队，挑战目前 **AI** 领域的极限。该公司专注于 **AI** 领域，并构建了世界上最快的 **GPU** 推理引擎。
   - 该职位位于巴黎，主要是远程办公。
- ****Kog.ai** 声称比 **AMD MI300X** 提速 **3-10 倍****：**Kog.ai** 的推理引擎与最佳 **GPU** 替代方案（从 **AMD MI300X** 加速器开始）相比，提供了 **3 到 10 倍** 的速度提升。
   - 该公司的目标是在未来 **12 个月** 内实现 **100 倍** 的性能增益。
- **成员要求提供关于比 **vLLM, SGLang, TensorRT-LLM** 性能高 10 倍的博客文章**：成员们对 **Kog 推理引擎** 旨在使 **AI** 推理达到极速（比 **vLLM, SGLang 或 TensorRT-LLM** 快 **10 倍**）且目前已达到 **3 倍** 的说法表示关注。
   - 一位成员询问 *“是否有任何文章/博客等？”*


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1377040741079060641)** (9 条消息🔥): 

> `Ninja Build Tool, Ubuntu 24.04, venv` 


- **Ubuntu 24.04 上的 Ninja 构建故障排除**：一位成员在运行命令 `['ninja', '-v']` 后，在 Ubuntu 24.04（gcc 13.3.0, CUDA 12.4, PyTorch 2.7.0）上遇到了 **ninja 构建错误**（退出状态 1）。
   - 他们曾尝试将 **ninja -v** 修改为 **--version** 并导出 CUDA 和 GCC 的环境变量，但没有进展。
- **倾向于全局安装 Ninja**：一位成员建议确保 **ninja** 安装在操作系统全局，而不仅仅是在虚拟环境 (**venv**) 中。
   - 用户报告说他们在 **venv** 中使用了 `pip install ninja` 并收到了错误消息。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1376742281691926650)** (6 条消息): 

> `CUDA tensors, axolotl vs torchtune` 


- **CUDA 张量的特性曝光**：用户观察到某种特定行为是 **CUDA tensors** 特有的，其中一位建议通过 `._tensor` 访问底层数据作为一种变通方法。
   - 讨论强调，相比于观察到的行为，直接报错会更符合直觉。
- **Axolotl 和 TorchTune 配置差异**：一位成员发现 **axolotl** 中的 `max_seq_len` 和 LR 调度器默认值与 **torchtune** 不同，在匹配这些参数并重新运行实验后，获得了性能改进。
   - 有人指出，并未观察到 **bfloat16 ptq** 导致的剧烈性能下降以及随后通过 **QAT** 恢复性能的现象，在弄清楚这两个 recipe 之间的确切差异之前，应该先让 PR 合并。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1376986442823372800)** (3 条消息): 

> `Fused Neighborhood Attention, Cutlass Implementation, Triton Implementation` 


- **Fused Neighborhood Attention 实现落地**：一位成员实现了 fused neighborhood attention，并提供了该实现的 [Pull Request 链接](https://github.com/linkedin/Liger-Kernel/pull/732)。
   - 该成员还创建了 [一个 Issue](https://github.com/linkedin/Liger-Kernel/issues/733) 来跟踪这项工作。
- **Cutlass 基准启发 Triton Kernel**：一位成员指出，该实现基于一篇带有基准 **Cutlass 实现** 的论文。
   - 他们还表示，他们基本上为其推导出了 **Triton 实现**，包括前向和后向 kernel。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1376733887991910501)** (3 messages): 

> `Async TP, AutoThink, CUDA education, NVIDIA event` 


- ****Async TP** 计算与通信重叠图解**：一位爱好者分享了一份[深度图解指南](https://x.com/vega_myhre/status/1927142595097956834?s=46)，详细介绍了在 **TP+SP** 中如何利用 **Async TP** 实现计算与通信的重叠，涵盖了背景理论和实现细节。
   - 欢迎针对如何实现高性能方案提供反馈。
- ****AutoThink** 技术提升推理性能**：一项名为 **AutoThink** 的新技术已发布，通过对查询复杂度进行分类、动态分配 thinking tokens 以及使用 steering vectors，在 **GPQA-Diamond** 上的推理性能提升了 **43%**。
   - 该项目包含一篇[论文](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5253327)和[代码](https://github.com/codelion/optillm/tree/main/optillm/autothink)，适用于任何本地推理模型，如 **DeepSeek** 和 **Qwen**。
- **NVIDIA 举办 **CUDA** 教学活动**：NVIDIA 将于 5 月 28 日举办一场[虚拟专家交流会](https://gateway.on24.com/wcc/experience/elitenvidiabrill/1640195/4823520/nvidia-webinar-connect-with-experts)，邀请了 **Programming Massively Parallel Processors** 的作者 **Wen-mei Hwu** 和 **Izzat El Hajj**。
   - 讨论内容将涵盖他们的著作、CUDA 历程，以及即将出版的 **PMPP** 第 5 版的预期内容。


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1377009152366477402)** (1 messages): 

> `Reasoning without External Rewards` 


- **推理能力在无奖励情况下涌现**：一位成员分享了一篇有趣的论文，题为《[*Learning to Reason without External Rewards*](https://arxiv.org/abs/2505.19590)》。
   - 该论文探讨了在不依赖外部奖励的情况下，使 AI 发展出推理能力的方法。
- **无奖励推理论文引发关注**：一位用户重点推荐了论文《[*Learning to Reason without External Rewards*](https://arxiv.org/abs/2505.19590)》，认为其与本频道的主题高度相关。
   - 讨论可能会涉及模型如何通过内在动力或自监督，而非传统的强化学习来学习复杂的推理任务。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1376938885392634067)** (9 messages🔥): 

> `Unexpected Error Reporting, Github API Limitations, Non-deterministic Bugs` 


- **意外错误需要报告**：一位成员报告收到了“意外错误”消息，并询问该通知谁。
   - 另一位成员建议这可能是由于代码中的反斜杠符号或提交文件过大（超过 34kb）导致的。
- **Github API 限制导致问题**：一位成员指出“意外错误”并非由之前建议的原因引起，并分享了提交编号。
   - 另一位成员表示这可能是 **Github API 限制**导致的问题，并建议重试提交，因为这类问题通常是**非确定性（non-deterministic）**的。
- **非确定性 Bug 难以修复**：一位成员开玩笑说通过在主循环中添加 `sleep(3.14)` 来修复 Bug。
   - 这突显了所遇问题的**非确定性本质**，暗示这些问题难以系统地复现和修复。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1376649293057888337)** (42 messages🔥): 

> `MI300, H100, Leaderboard updates, Personal best, amd-fp8-mm` 


- **H100 排序代码位列第二**：一位成员的提交在 H100 `sort` 排行榜上获得**第二名**，耗时 **6.55 ms**。
- **MI300 AMD-FP8-MM 获得多次更新**：MI300 上关于 `amd-fp8-mm` 的多次提交显示了“成功”和“个人最佳”结果，其中一次提交以 **116 µs** 位列**第一名**。
   - 其他显著的成绩包括 **174 µs** 和 **168 µs**，位列第七。
- **AMD-MLA-Decode 记录成功运行**：MI300 上关于 `amd-mla-decode` 的几次提交均告成功，其中一次以 **90.2 ms** 获得**第四名**。
   - 另一次提交以 **135 ms** 位列**第七名**。
- **AMD-Identity 在 MI300 上成功运行**：MI300 上关于 `amd-identity` 的多次提交均告成功，耗时约为 **6.7 µs**。
   - 其中一次提交以 **6.80 µs** 达到**第四名**。
- **AMD-Mixture-of-Experts 表现强劲**：MI300 上关于 `amd-mixture-of-experts` 的多次提交均告成功，个人最佳成绩为 **286 ms**。
   - 几次运行的时间均在 **295 ms** 左右。


  

---

### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1376951303317618893)** (1 messages): 

> ``/leaderboard` 命令修复，Bug 报告` 


- **`Leaderboard` 命令获得修复！**：针对 `/leaderboard show/list` 命令部署了一个简单的修复补丁。
   - 用户被要求报告更新后出现的任何新问题或仍然存在的问题。
- **征集 `Leaderboard` 的 Bug 报告**：在修复 `/leaderboard show/list` 命令后，开发团队正在征求反馈。
   - 开发者们热衷于识别任何与 Leaderboard 功能相关的回归问题（regressions）或未解决的问题。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1376647201316409425)** (3 messages): 

> `Factorio 2.0, Vision 集成` 


- **渴望 Factorio 2.0 支持**：一名成员表达了希望支持 **Factorio 2.0** 的愿望，以便对空间平台的构建进行训练/评分。
   - 未提供更多细节。
- **思考 Vision 集成**：一名成员在阅读一篇论文后感到惊讶，尽管没有 **Vision 集成**，模型仍然可以完成这么多工作。
   - 他们对 scaffolding（脚手架）表示了赞赏。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1376649472687472832)** (19 messages🔥): 

> `AMD 竞赛详情, 反向传播 Kernel 排行榜, RoPE 计算修正, HIP 支持` 


- **AMD 竞赛详情澄清**：AMD-identity 竞赛是一个 runner test，而 **amd-fp8-gemm**、**mla** 和 **moe** 才是主要的竞赛项目。
   - 竞赛旨在通过使用 inline CUDA 或 HIP，或者通过 Triton 进行优化，来改进 **PyTorch** 的基准测试（baselines）。
- **反向传播 Kernel 排行榜正在开发中**：计划推出一个反向传播（backpropagation）Kernel 的排行榜，参与者可以提交 inline **CUDA** 或 **HIP** 代码来改进 **PyTorch** 基准。
   - 该计划旨在鼓励使用 **Triton** 或其他方法进行优化，以确保正确的反向传播。
- **RoPE 计算旋转争议**：一名成员指出 RoPE (Rotary Positional Embedding) 计算中存在一个潜在问题，特别是对交替索引进行分块（chunking）而不是拆分前半部分和后半部分，并建议参考 [Deepseek v3 实现](https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L212)。
   - 团队决定保持实现不变，因为这不被视为关键问题，尽管这会给参与者带来一些额外开销。
- **通过 Load Inline 实现 HIP 支持**：成员们讨论了在平台上添加 HIP 支持的可能性。
   - 目前的建议是使用 `load_inline` 提交 **HIP** 代码，目前没有支持原生（raw）HIP 的直接计划。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1376636165737939005)** (2 messages): 

> `cute-dsl, Tensor 内存, sgemm_05` 


- **cute-dsl 中的 Cutlass GEMM 示例**：一名用户引用了这个[示例](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/hopper)，用于在 **cute-dsl** 中使用 **Tensor** 内存编写 **GEMM**。
   - 未就实现的细节或性能提供进一步的讨论或细节。
- **Sgemm_05 形状问题**：一名用户询问为什么 **sgemm_05** 实现可能无法在更大的形状（shapes）上工作。
   - 在给定的上下文中没有讨论具体原因或解决方案。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1376636184494739567)** (73 messages🔥🔥): 

> `使用 LCM 进行实时视频生成，HuggingChat Android 应用，微调视频模型，AI Agent 可观测性库，Smol LM2 工程师` 


- **实时视频生成迎来分水岭时刻**：一位成员分享说，**2023 年 10 月发布的 LCM** 是一个分水岭时刻，使得实时视频应用成为可能。该成员在 4090 上以 **sdxl** 质量达到了 **1280x1024 分辨率下的 23 fps**。
   - 他们一直在创建诸如 **RTSD**（用于快速可视化事物）和 **ArtSpew**（用于加速图像生成）之类的应用。
- **HF Android 应用 Hack 引起关注**：一位成员为 **HuggingChat** 创建了一个快速的 **Android 应用** 概念验证，可在 [HuggingFace Space](https://huggingface.co/spaces/JoPmt/HuggingChat_Android_App/tree/main) 获取，该项目修改自现有项目，并确认 **APK 可以按预期安装和运行**。
   - 用户报告了诸如键盘重定向到网站设置和插件冲突等问题，但也对这种变通方法表示赞赏。
- **成员探索视频模型的微调技术**：成员们分享了用于 **微调 SOTA 模型** 和多 GPU 训练的仓库 [tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe)。
   - 还提到了 [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)，因其灵活性而受到关注。
- **AI Agent 可观测性库浮出水面**：一位成员正在开发一个 **AI Agent 可观测性** 库，其中一个版本是为 smolagents 构建的，地址为 [ltejedor/agents](https://github.com/ltejedor/agents/tree/main/18-19-proof-of-work)，并寻求正在构建多 Agent 系统的人员进行 30 分钟的用户访谈反馈。查看 [时间线](https://github.com/ltejedor/agents/tree/main/20-timeline)。
   - 这是在构建 *v1* 之前的 *version 0*。
- **HuggingFace TB 工程师在 Hub 上现身**：在一位成员询问如何联系 **Smol LM2 工程师** 后，另一位成员表示 *其中一些人经常出现在 Hub 上，所以你可能可以在 Discord 上联系他们*。这里有指向 [HuggingFaceTB](https://huggingface.co/HuggingFaceTB) 组织和 [github.com/huggingface/smollm](https://github.com/huggingface/smollm) 的相关链接。
   - 不过，通过 GitHub 联系他们可能会更快。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1377042030425014323)** (1 messages): 

> `` 


- **Query, Key, Value 概念澄清**：用户正在使用 Google 搜索类比来解释 Query、Key 和 Value 之间的关系，其中 Query 是搜索内容，Key 是关键词，Value 是网页的内容。
- **Google 搜索类比**：该类比使用 Google 搜索来阐明关系：'query' 作为搜索，'key' 作为关键词，而 'value' 作为搜索结果网页的内容。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1376636453982965790)** (9 messages🔥): 

> `SweEval, NIST 工具, AutoThink, Langchain` 


- ****SweEval** 数据集公开！**：旨在测试 LLM 过滤脏话能力的 **SweEval** 数据集已公开，并被 NACCL '25 工业轨道接收（[数据集链接](https://huggingface.co/papers/2505.17332)）。
   - 该数据集已有 **120 多次下载**，如果 LLM 在过滤脏话方面仍然吃力，创作者鼓励用户点赞。
- **新行政命令针对欺诈性研发**：4 天前发布了一项新的行政命令（[whitehouse.gov 链接](https://www.whitehouse.gov/presidential-actions/2025/05/restoring-gold-standard-science/)），旨在纠正科学领域中被描述为欺诈性的研发范式。
   - 与此相关，成员们正在构建来自 **NIST** 的工具，并展示了他们的安全措施符合 **NIST** 标准，该标准为美国消费者的技术安全设定了基准。
- ****AutoThink** 提升推理性能！**：发布了一种名为 **AutoThink** 的新技术，声称在 GPQA-Diamond 上将推理性能提高了 **43%**（[论文链接](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5253327)）。
   - 该方法对 Query 复杂度进行分类，动态分配 Thinking Tokens，并使用引导向量来引导推理模式，适用于 **DeepSeek**、**Qwen** 和 **Llama** 等模型；代码可在 [GitHub](https://github.com/codelion/optillm/tree/main/optillm/autothink) 获取。
- **Langchain Pull Request**：一位成员引用了一个 [Langchain PR](https://github.com/langchain-ai/langchainjs/pull/8237)。
   - 未提供更多上下文。


  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1376644046969372742)** (1 messages): 

> `Cross-posting, Staying on topic` 


- **劝阻交叉发布**：一名成员要求其他成员不要在 reading-group 频道中进行交叉发布（Cross-posting）。
   - 他们还要求每个人保持频道内容与主题相关。
- **频道主题执行**：一名成员提醒大家保持 reading-group 频道的主题一致性。
   - 这一提醒旨在维持频道内的专注度和相关性。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1376711185520394341)** (2 messages): 

> `trocr tuning, length collapse issue, computer vision reading group` 


- **Trocr 微调问题浮现**：一名成员正在微调 **trocr**，其 Ground Truth 是两个 Token + bos/eos（两个数字）但中间有一个空格，微调后模型仅预测第一个 Token 然后直接输出 eos。
   - 他们研究了**长度崩溃问题（length collapse issue）**（即使减少了最大长度也存在），并询问潜在原因，怀疑是一个简单的错误。
- **寻求 CV 阅读室**：一名成员正在询问是否存在 **Computer Vision 阅读小组**。
   - 他们正在寻求有关是否有人知道此类小组的信息。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1376929007500660737)** (2 messages): 

> `Multi-Agent System, Medical Project, Langgraph, drug discovery research agent, treatment protocol agent` 


- **使用 Langgraph 集思广益创新医疗 Agent**：一名 AI 开发者正在 **Langgraph** 中为一个**医疗项目**构建 **Multi-Agent 系统**，并征求创新的 Agent 想法。
   - 目前的系统包括用于*症状检查、危机管理、指导、药物研发研究和治疗方案*的 Agent。
- **为医疗项目寻求创新的 Agent 想法**：一名 AI 开发者正在为使用 **Langgraph** 构建的**医疗项目** **Multi-Agent 系统**寻求创意建议。
   - 该系统已经整合了用于*症状检查、危机干预、指导、药物研发和治疗方案*的 Agent，正在寻求超越简单 API 调用的增强功能。


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1376960290729754635)** (1 messages): 

> `Agents & MCP Hackathon, Model Context Protocol, AI Agents, SambaNova Systems` 


- **Agents & MCP 黑客松宣布！**：Hugging Face 宣布了有史以来第一个大型全线上 **MCP 专题黑客松**，将于 **2025 年 6 月 2 日至 8 日**举行，三个赛道共设 **$10,000** 现金奖励。
   - 已确认的赞助商包括 **SambaNova Systems**，他们将为早期参与者提供 API 额度，并在活动期间提供专门的 Discord 频道。报名现已开放：[huggingface.co/Agents-MCP-Hackathon](https://huggingface.co/Agents-MCP-Hackathon)。
- **SambaNova Systems 赞助 AI Agents 黑客松！**：**SambaNova Systems** 将为早期参加 AI Agents 黑客松的开发者提供免费 API 额度，并赞助此次活动。
   - 本次活动还将包含 Gradio 的答疑时间（office hours）。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1376657974537228410)** (9 messages🔥): 

> `Llama 3.2 Errors, GAIA Submission Issues, Agent Security Measures` 


- ****Llama 3.2** 抛出 ValueErrors**：用户在使用 **meta-llama/Llama-3.2-3B-Instruct** 进行文本生成时遇到 `ValueError`，因为它仅支持 *conversational* 任务。
   - 切换到 **Llama-3.1** 可解决此问题。
- **提交至 **GAIA** 排行榜失败**：通过[此链接](https://huggingface.co/spaces/gaia-benchmark/leaderboard)向 **GAIA** 基准测试排行榜提交时因*格式错误*而失败。
   - 预期的 JSON 格式为 `{"task_id": "task_id_1", "model_answer": "Answer 1 from your model"}`。
- **寻求 Agent 安全实现方案**：一名用户询问如何为下载并与文件交互的 AI Agent 实现安全功能，特别是针对 Agent 执行有害代码的风险。
   - 该用户希望防止 Agent 通过*盲目下载并执行代码*而轻易破坏系统。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1376643047059886091)** (21 条消息🔥): 

> `Notebook LM 使用技巧，总结技术章节，西班牙语播客生成，文档分析的法律用例，Notebook LM 中的语音变化` 


- **NotebookLM 新手寻求使用导航细节**：新用户询问了关于利用 **Notebook LM** 的最佳实践，特别是关于源文档的大小和类型，并注意到了其在处理图像方面的局限性。
   - 一位主修会计和 CS 且对历史感兴趣的学生，寻求在不同领域有效使用该工具的建议，希望从社区的多样化经验中学习。
- **技术总结提示词（Prompt）强力指南**：用户讨论了总结技术章节的有效提示词，强调了**识别主题**对于模型更好消化内容的重要性。
   - 讨论强调，组织良好的数据集（如非虚构类书籍）有助于模型更轻松地处理，从而产生更长且更连贯的总结。
- **西班牙语播客补丁：正确提示播客制作**：一位用户询问如何使用西班牙语的 Notebook LM 根据某个主题生成播客；另一位用户解释了添加来源、创建笔记本以及使用 “Conversación de análisis en profundidad” 功能来**生成播客**的过程。
   - 一位用户表示：*Tienes que agregar fuentes y crear un cuaderno. Despues de tener este cuaderno en la parte derecha de notebook hay un boton que dice: \"Conversación de análisis en profundidad\"*。
- **法律术语物流：律师对法律杠杆的赞誉**：一位用户描述了一个涉及两家公司合并的法律用例，其中 **Notebook LM** 被用于简化和解释 25 份法律文件，创建时间线，并生成简报文件和 FAQs。
   - 该用户强调了该工具识别异常信息并促进与文档讨论的能力，最终给他们的律师留下了深刻印象，并表示该律师*很快就会去查看 NotebookLM*。
- **语音变化尝试：通过 WAV 魔法实现人声多样性**：一位用户询问了为 Notebook LM 语音增加变化的最佳方法，寻求默认 NPR 风格之外的选项，并询问是否有必要在第三方应用中**下载并修改 .wav 文件**。
   - 另一位用户建议，编辑 .wav 文件的速度、音高和其他参数可以改善声音，但不知道有任何特定的人声增强（humanizer）应用。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1376644221238513764)** (74 条消息🔥🔥): 

> `笔记本组织，嵌入 NotebookLM，iOS 上的交互模式问题，播客生成器问题，Gemini Deep Research 集成` 


- **笔记本组织功能需要认真改进**：用户要求更好的笔记本组织方式，例如**文件夹或标签**，因为目前的 **A-Z 排序**已不足够。
   - 无法在**笔记本层级进行搜索**被视为一个主要缺点，浏览器的查找功能被认为“相当原始”。
- **在网站上嵌入 NotebookLM：是否可行？**：一位用户询问是否可以将 **NotebookLM 嵌入到网站**上供他人试用，但回复指出只能分享链接，且需要**授予文件访问权限**。
   - 目前不存在直接嵌入功能，限制了共享用户进行交互式访问。
- **iPhone/iPad 交互模式停滞**：几位用户报告了在点击交互模式按钮时，**iPhone 和 iPad 上的交互模式无法启动**的问题，且至少有一位用户拥有 NotebookLM pro。
   - 有人建议尝试**网页版**，并提醒在进入交互模式后点击播放按钮；一位用户发现无论是否是 pro 用户，网页版的效果都更好。
- **播客生成器故障频发**：用户报告称**播客生成器经常停止工作**，并建议改为**下载音频**。
   - 一位用户抱怨在“个性化”部分设置的**最小播客长度**被忽略了，尤其是在使用**法语音频**时。
- **通过 NotebookLM 释放 Gemini Deep Research 的潜力**：用户有兴趣将 **Gemini Deep Research** 与 **NotebookLM** 结合使用，其中一位询问如何使用 Gemini **挖掘来源并创建基于事实的总结**作为 NotebookLM 的输入。
   - 已确认用户可以导出 Gemini Deep Research 的输出（文本和来源），并通过复制粘贴将其作为来源导入 NotebookLM，这也可以通过新的 `Create` 下拉菜单完成。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1376640007275548844)** (44 messages🔥): 

> `compute and comms overlap, TP+SP using Async TP, matrix multiplications, RL vs diffusion` 


- **Async TP 重叠计算与通信！**：分享了一个关于在 **TP+SP** 中使用 **Async TP** 如何实现计算与通信重叠的图解深度分析，并指出其 PyTorch 实现基于 Eleuther ML 性能阅读小组讨论过的一篇论文，详见[此处](https://x.com/vega_myhre/status/1927142595097956834)。
- **矩阵乘法形状揭秘！**：一位成员分享了一篇关于[矩阵乘法形状](https://www.thonking.ai/p/what-shapes-do-matrix-multiplications)的博客文章链接，强调内存访问比计算开销大得多。
   - 澄清了 **3N^2** 和 **2N^3** 是执行的操作*数量*，而这些操作耗费的*时间*并不相等：内存访问耗时要长得多。
- **RL 与 Diffusion 的碰撞：谁更胜一筹？**：一位成员询问是否有人对某份文档感兴趣，并请教他的下一次智力冒险应该是选择 **RL** 还是 **Diffusion**，并链接到了[此资源](http://dx.doi.org/10.13140/RG.2.2.29337.53608)。
- **禁止 Gomez 区域！**：一位成员建议不要向新人推荐 **Kye Gomez**，即使是开玩笑也不行。
   - 原因是*我们不需要更多的人掉进他的坑里。*


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1376740518008193026)** (27 messages🔥): 

> `Quantization for ACL papers, Static n-gram heads, Buggy RWKV7, Spurious Rewards` 


- **量化合理性的困惑**：一位成员质疑，由于资源限制，面向应用的 **ACL 论文** 是否可以仅展示 **4-bit 量化 LLM** 的结果，并引用了[这篇论文](https://arxiv.org/abs/2505.17895)。
- **N-gram Head 寻找**：一位成员询问了关于添加静态 **n-gram heads** 的尝试，类似于[这条推文](https://x.com/akyurekekin/status/1751987005527855410)中使用的技术。
- **RWKV7 的退化烦恼**：成员们讨论了[一个项目](https://github.com/Benjamin-Walker/structured-linear-cdes)在 **FLA** 中使用了有 Bug 的 **RWKV7**，并引用了与[这篇论文](https://arxiv.org/abs/2505.17761)相关的多个精度问题。
- **RLVR 奖励回顾**：成员们分享了一个关于 **RLVR** 中伪奖励（Spurious Rewards）及重新思考训练信号的 [Notion 页面](https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880)和[论文链接](https://arxiv.org/abs/2505.17749)，其中一位成员将其描述为*熵最小化和过度拟合（hyperfitting）的又一案例*。
- **P100 奇特的坚持**：一位成员引用了一篇[论文](https://arxiv.org/abs/2505.21493)，质疑为什么在 **2025** 年仍有团队在 **P100** 上进行训练。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1376915816762839122)** (1 messages): 

> `lm eval harness, gguf models, python-llama-cpp, local model evaluation` 


- **在 `lm eval harness` 中寻求高效的 `.gguf` 评估**：由于性能问题，一位成员正在寻找一种在 `lm eval harness` 中高效评估本地 `.gguf` 模型的方法。
   - 该用户尝试使用 `python-llama-cpp` 启动服务器，但速度极慢。
- **`python-llama-cpp` 在本地模型上的困扰**：一位用户报告称，使用 `python-llama-cpp` 评估本地 `.gguf` 模型时运行速度极慢。
   - 上下文中未提供解决方案。

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1376664511091839048)** (60 messages🔥🔥): 

> `Claude 4 GitHub MCP exploit, Sesame Speech-to-Speech models, Launching AI products in Europe, Gemini Ultra access in Europe, Qwen RL results` 


- **Claude 4 结合 GitHub MCP server 泄露私有仓库**：一种新型攻击利用 **Claude 4** 和 **GitHub** 的 **MCP server** 从**私有 GitHub 仓库**中泄露数据，包括**姓名**、**旅行计划**、**薪资**和**仓库列表**，由恶意 issue 触发。
   - 建议用户限制 Agent 权限并监控连接；[Invariant 的安全扫描器](https://xcancel.com/lbeurerkellner/status/1926991491735429514?s=46&t=Ld13-WcFG_cohsr6h-BdcQ)已主动识别出这种“毒性流（toxic flow）”。
- **Sesame 跨越语音的恐怖谷**：讨论了 **Sesame** 的 speech-to-speech 模型，并附带了 [Sesame 研究](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice)链接，以及 Thomas Wolf 对构建这些模型的技术分解。
   - 该分解可以在[这篇博客文章](https://xcancel.com/Thom_Wolf/status/1916809878162514142)中找到，进一步阐明了近期上下文语音模型和音频 Tokenization 的工作原理。
- **导航欧洲 AI 产品发布**：强调了在欧洲发布 AI 产品的挑战，包括**德国的采购订单**、**法国的劳动法**，以及即使在消耗大量资源的情况下也必须遵守的 **14 天强制退款政策**。
   - 这项针对 ChatGPT 的 [EU 政策](https://openai.com/policies/eu-terms-of-use/)可能导致短期滥用——因此 AI 应用必须谨慎设置限制——尽管 OpenAI 如果明确告知用户，可以选择退出该条款。
- **Qwen 的伪 RL 数学奖励**：讨论了“伪奖励（spurious rewards）”（包括随机奖励）如何提升 **Qwen2.5-Math-7B** 的数学性能，其效果堪比来自 ground-truth 奖励的提升，详见[此推文串](https://xcancel.com/StellaLisy/status/1927392717593526780)。
   - 这种在 Qwen 模型中特有的现象表明，由于 **GRPO 的“裁剪偏差（clipping bias）”**，**RLVR** 放大了一现有的“代码推理”模式，挑战了传统的奖励结构观点。
- **小型 LLM 获得评估并开源！**：**j1-nano（6 亿参数）和 j1-micro（17 亿参数）**作为具有竞争力的奖励模型开源，使用单张 **A100 GPU** 并通过 **Self Principled Critique Tuning (SPCT)** 训练以生成特定实例的评估标准，详情见[此推文](https://xcancel.com/leonardtang_/status/1927396709870489634)。
   - **j1-micro** 媲美 **Claude-3-Opus** 和 **GPT-4o-mini** 等大型模型，而 **j1-nano** 则可与 **GPT-3.5-turbo** 竞争。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1377117772491915265)** (4 messages): 

> `AI Engineer conference, Volunteer Opportunity, Speaker announcements` 


- **AI Engineer 会议招募志愿者！**：[AI Engineer 会议](https://xcancel.com/swyx/status/1927558835918545050)正在寻求 **30-40 名志愿者**提供后勤支持，以换取免费入场名额（价值高达 **$1.8k**）。
   - 该活动由 @aiDotEngineer 主办，将于 **6 月 3 日至 5 日**在旧金山举行。
- **主题演讲嘉宾阵容公布**：AI Engineer 会议公布了第一波主题演讲嘉宾，包括 **Greg Brockman** (OpenAI)、**Sarah Guo** (Conviction)、**Simon Willison** 以及来自知名 AI 公司的其他成员。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1376674086708576329)** (3 messages): 

> `LlamaCloud 更新, LlamaParse & AnthropicAI Sonnet 4.0, LlamaIndex 多模态嵌入器, OpenAI 增强型结构化输出` 


- **LlamaCloud 获得持续更新**：团队宣布他们正在**不断为 LlamaCloud 发布更新和新功能**。
   - 未提供更多细节。
- **LlamaParse 拥抱 AnthropicAI 的 Sonnet 4.0**：**LlamaParse** 现在在 Agent 和 LVM 模式下支持 **AnthropicAI Sonnet 4.0**，正如[这条推文](https://t.co/yNcOtjKMzm)所述，这使得在为 AI 应用解析复杂文档时能够使用最新的 LLM。
- **LlamaIndex 展示如何构建自定义多模态嵌入器**：学习如何为 LlamaIndex 构建自定义多模态嵌入器，如[这条推文](https://t.co/jBqn7jrMak)所示，该指南介绍了如何覆盖 LlamaIndex 的默认嵌入器以支持 **AWS Titan Multimodal**，并将其与 **Pinecone** 集成。
   - 该指南详细说明了如何创建一个处理文本和图像的自定义嵌入类。
- **OpenAI 结构化输出支持**：LlamaIndex 现在为 **OpenAI** 提供增强的结构化输出支持，以响应其最近的扩展，包括 **Arrays** 和 **Enums** 等新数据类型以及字符串约束字段，如[这条推文](https://t.co/SlkVrMmzRA)所述。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1376921288840450158)** (23 messages🔥): 

> `表单填写 Agent, 基于 Workflow 的 Agent, 多模态 Agent, ReactJS 与 LlamaIndex, 结合 React 的 HITL Workflow` 


- **表单填写 Agent 即将到来**：成员们讨论了使用 LlamaIndex 实现披萨订购流程，将其称为“**表单填写 Agent**（form filling agent）”，并建议使用包含 `AskUserForOrderEvent` 和 `ConfirmUserAddressEvent` 等步骤的自定义 Workflow。
   - 建议 Workflow 中的工具应写入中央存储（如 **workflow context**），以维护和更新用户数据，特别是当用户在订购过程中反复修改时。
- **Workflow Agent 替代 FunctionCallingAgent**：成员们建议在处理更复杂的流程时，使用较新的**基于 Workflow 的 Agent**，而不是预构建的 `FunctionCallingAgent`。
   - 有人提到虽然存在 `CodeAct` Agent，但在可能的情况下 `FunctionAgent` 更可取，并且 Agent 接受[多模态输入](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/#multi-modal-agents)，如 `TextBlock`、`ImageBlock`、`AudioBlock` 和 `DocumentBlock`（尽管并非所有 LLM 都支持所有类型的 Block）。
- **ReactJS 结合 LlamaIndex 实现 HITL Workflow**：一位成员寻求关于将 **ReactJS** 与 **LlamaIndex** 集成以实现 **Human-in-the-Loop (HITL) Workflow** 的建议，并对通过 WebSocket 通信使用 `ctx.wait_for_event()` 的复杂性表示担忧。
   - 另一位成员建议 `ctx.wait_for_event()` 效果很好，并引用了一个[社区办公时间示例](https://colab.research.google.com/drive/1zQWEmwA_Yeo7Hic8Ykn1MHQ8Apz25AZf?usp=sharing)，该示例展示了两种形式的 HITL：直接响应和在人工输入后响应。
- **RelevancyEvaluator 重新路由质量不佳的回答**：一位成员实现了一个 Workflow，使用 `RetrieverRouter` 和 Reranker 查询两个知识库，并寻求关于如何处理来自 `RelevancyEvaluator` 的不满意回答的建议。
   - 他们分享了显示使用 `StartEvent` 的重试机制的代码，但也担心在相同的错误节点上浪费时间。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1376638616834670682)** (19 条消息🔥): 

> `Asynchronous tools, Isolated Mistral instance, Architect Tool, MCP Server, MCP Clients` 


- ****异步探险**：何时通知？**：在创建需要几分钟才能完成的异步工具时，最佳方法是发送提供状态更新的通知，或者返回一个[带有说明的链接](https://example.com)供用户监控完成情况。
   - 返回带有更改通知的 *embeddedresource* 可能也行得通，但这高度依赖于客户端支持和对用户行为的假设。
- ****Mistral 堡垒**：在本地运行查询**：为了安全地运行 "MPC 查询"，请使用像 **LibreChat** 或 **Ollama** 这样本地优先的聊天客户端，并配合本地运行的 **Mistral** 实例，然后将你的 MCP Server 连接到该聊天客户端。
   - 一位成员分享了一篇 [Medium 文章](https://medium.com/@adkomyagin/building-a-fully-local-open-source-llm-agent-for-healthcare-data-part-1-2326af866f44)，详细介绍了如何设置 **LibreChat+MCP**。
- ****规则构建器盛宴**：寻求架构工具**：一位成员正在寻找一种优秀的“架构师”工具来构建规则文件，该工具需具备完整的规划和任务列表功能，以帮助非技术用户。
   - 目前尚未给出具体的建议。
- ****MCP Server 策略**：自制汉堡还是麦当劳？**：针对 LLM 已经可以访问开发者门户和 API 文档这一业务案例异议，一位成员建议这样回答：*“你可以买面包、碎肉、奶酪、番茄等自己在家里做汉堡，但你还是会点麦当劳，不是吗？”*
   - 另一位成员建议将文档作为 MCP resources 开放，以减少手动摩擦：*只需点击一个按钮，LLM 就能彻底了解你的业务*。
- ****API-Ready 热潮**：冲浪 AI 浪潮**：构建 MCP Server 是一个冲浪热潮并推销你的 API/SaaS 为 **AI-ready** 的绝佳机会，这使其更容易与多个 LLM 客户端集成。
   - 未提及具体的 MCP 客户端。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1376829638462148639)** (4 条消息): 

> `MCP Inspector, Ship Lean MCP, UI issues` 


- ****MCPJam** 构建增强版 MCP Inspector**：一位成员正在构建一个名为 [@mcpjam/inspector](https://github.com/MCPJam/inspector) 的增强版 **MCP inspector**，具有改进的 UI 和 LLM 聊天等调试工具，旨在解决官方仓库开发缓慢的问题。
   - 使用 `npx @mcpjam/inspector` 可以轻松启动该检测器，团队对社区开发和功能需求持开放态度。
- ****LeanMCP** 发布 Vibe-Coding 平台**：一位成员发布了一个用于构建和发布远程 MCP 的平台 [https://ship.leanmcp.com](https://ship.leanmcp.com)。
   - 目标是让人们能够进行 *vibe-code* 并发布 MCP。如果有良好的用户基础和 PMF（产品市场契合度），早期用户将获得整整一年的免费 PRO 版本。
- **LeanMCP 存在 UI 问题**：成员们反映 LeanMCP 平台存在 **UI 问题**。
   - 具体而言，**Discord 和 LinkedIn 链接**无法工作，且电子邮件地址在侧边溢出。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1376639814815973407)** (22 条消息🔥): 

> `Synthesizing sentence meaning into a single token, Faiss index creation, Local LLama Interface, GPT4All version 4, Nomic burned $17M Series A funds?` 


- **将句子含义合成单个 Token**：一位成员的目标是使用拥有约 **1M 参数**和 **12 GB 显存 GPU** 的模型，将整个句子的含义合成一个 Token，并打算用这些 Token 创建 [FAISS 索引](https://github.com/facebookresearch/faiss)。
- **GPT4All 使用 Embedders**：有人指出 **GPT4All** 利用了 embedders，并建议阅读 [HuggingFace 上的提示](https://huggingface.co/kalle07/embedder_collection)以获取帮助。
- **本地 LLaMA 界面**：一位成员正期待 **GPT4All 第 4 版**，并引用了 LocalLLaMA 上的一位开发者创建的 **LLM 界面**，该界面具有语音输入、深度研究以及图像生成模型兼容性，并附带了 [Reddit 帖子链接](https://www.reddit.com/r/LocalLLaMA/comments/1kvytjg/just_enhanced_my_local_chat_interface/)。
- **Nomic 还在运行吗？**：一位成员询问 Nomic 是否已经耗尽了 **2023 年** A 轮融资获得的 **1700 万美元**，推测这可能是感知到活动减少的原因。
- **Kobold 专注于 RP**：一位成员提到 [Kobold.cpp](https://github.com/facebookresearch/faiss) 包含了“所有”功能，但有人指出 **Kobold** 主要关注 RP（角色扮演）并非他们的首选，他们更倾向于专用的 **LLM** 或**纯图像**工具。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1376979104624017509)** (2 条消息): 

> `Mojo 中的元编程，Mojo 中的 Go 泛型` 


- **探索 Mojo 元编程的文章**：一位成员推荐了一篇关于 [元编程的博客文章](https://www.modular.com/blog/metaprogramming)，并鼓励他人在 [相关的论坛主题](https://forum.modular.com/t/exploring-metaprogramming-in-mojo/1531) 中留言提问。
   - 该文章被描述为一篇*极佳的读物*，它阐明了 **Mojo** 允许对值进行参数化以及直接的控制流。
- **Mojo 的元编程 vs Go 泛型**：一位用户最初认为 **看起来像 Go 泛型的语法** 是 **Mojo** 中元编程的主要方法。
   - 然而，该用户惊讶地发现 **Mojo** 允许对值进行参数化和直接的控制流，这使其与 **Go 的泛型** 区分开来。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1376637789415805008)** (5 条消息): 

> `流式解析，结构化解析，从 Magic 迁移到 Pixi` 


- **解析性能深入探讨**：成员们讨论了 Mojo 中不同的 **JSON 解析策略**，包括 *streaming*（流式）和 *structured parsing*（结构化解析）。
   - 一位成员指出，在找到数据后停止解析的性能可以优于先解析整个 JSON。
- **DOM 解析表现不佳？**：成员们将 **on-demand parsing**（按需解析）与 **DOM parsing** 进行了对比，认为将按需解析与 DOM 解析进行比较是不公平的。
   - 一位成员声称，*如果你只对比 DOM 解析，它每次都会输*。
- **从 Magic 迁移到 Pixi 的动机**：一位成员回到了之前的讨论，并提供了一个指向 [Modular 论坛](https://forum.modular.com/t/migrating-from-magic-to-pixi/1530) 的链接，内容涉及从 **Magic 迁移到 Pixi**。
   - 未提供更多细节。


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/)** (1 条消息): 

serotweak: 大家好
  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1377027438751645776)** (2 条消息): 

> `API 使用，Error 400，Token 长度` 


- **API 新手遇到 Error 400**：一位成员报告在尝试使用 API 时收到 **Error 400** 消息，并表示他们完全是 API 使用的新手。
   - 收到的错误消息是：*"invalid request: message must be at least 1 token long or tool results must be specified."*（无效请求：消息必须至少包含 1 个 token，或者必须指定工具结果。）
- **Token 长度要求导致 Error 400**：**Error 400** 表明 API 请求失败，因为消息太短。
   - 错误消息指出，消息必须至少有 **1 个 token 长**，或者必须指定工具结果。


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1376987527705268264)** (2 条消息): 

> `来自东伦敦的学生，CS、图形学和游戏开发，技术的硬件和软件方面，以组装 PC 作为副业，学习如何编码和构建软件` 


- **东伦敦学生加入聊天！**：一位来自东伦敦的高中学生向服务器介绍了自己。
   - 他们学习 **CS、图形学和游戏开发**，并对硬件和软件都充满热情。
- **PC 组装达人寻求软件技能**：这位学生喜欢将组装 PC 作为副业，并希望学习如何编码和构建软件。
   - 他们希望获得在软件工程领域工作的技能。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1376689278431527074)** (5 条消息): 

> `LORA 微调，生成脚本，合并权重，Adapter 使用` 


- **LORA 微调后正确加载模型**：一位成员尝试在 LORA 微调后运行生成脚本，但在实例化模型和 checkpointer 时遇到了正确加载模型的问题。
   - 似乎生成脚本仅加载 [此处](https://github.com/pytorch/torchtune/blob/main/recipes/generate.py#L71-L74) 定义的 training.MODEL_KEY。
- **LORA 微调期间合并权重**：有人指出，在 LORA 微调期间，权重会被合并然后保存，因此在运行生成脚本时，不需要创建 LORA 模型。
   - 生成 recipe 假设模型已经与 adapters 合并，这发生在微调期间的最后一个 checkpoint，因此只需指向该位置即可。
- **直接使用 Adapters**：如 [教程](https://docs.pytorch.org/torchtune/stable/tutorials/e2e_flow.html#use-your-model-in-the-wild) 中所述，adapters 可以直接与其他生成工具一起使用。
   - 这为在不同生成场景中利用微调后的模型提供了灵活性。


  

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1376648177884856411)** (2 messages): 

> `Self Improving Vibe Coding Template, Using DSPy` 


- **自我改进 Vibe Coding 模板发布**：一位成员在 **GitHub** 上分享了一个[自我改进 Vibe Coding 模板](https://github.com/imranarshad/vibe_coding_template)。
- **在 GitHub 项目中使用 DSPy 的论据**：一位成员链接了一篇博客文章，论证了为什么要使用 **DSPy**。
   - 博客文章标题为 [Will the Model Eat Your Stack?](https://www.dbreunig.com/2025/05/27/will-the-model-eat-your-stack.html)。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1376648499453497355)** (2 messages): 

> `ReAct vs Custom Code, Trajectory Nudge in LLM` 


- **用户声称 ReAct 比自定义代码效果更好**：一位用户对 [ReAct](https://x.com/ohmypk92/status/1927084222528802891?s=19) 进行了广泛测试，并根据观察发现其表现优于他们的自定义代码。
   - 他们将此归功于 ReAct 为 LLM 提供的 *轨迹引导 (trajectory nudge)*。
- **轨迹引导赋予 ReAct 优势**：该用户推测，给予 LLM 的 *轨迹引导* 是 ReAct 表现更好的原因。
   - 这种引导有助于指导 LLM 的推理过程，从而比没有这种引导的自定义代码产生更好的结果。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1377043655902367977)** (4 messages): 

> `tinygrad.org hyperlink, Optypes hyperlink` 


- **Optypes 超链接失效变为 404**：[tinygrad.org](https://tinygrad.org) 网站上的 **Optypes 超链接** 现在返回 *404 - 页面未找到* 错误。
   - 这是由于最近将 *"uops 移动到目录中"* 的更改导致的。
- **Tinygrad 更新 TinyXXX**：George Hotz 分享了 [tinygrad/tinyxxx](https://github.com/tinygrad/tinyxxx) GitHub 仓库的链接。
   - 随后发布了 [PR#27](https://github.com/tinygrad/tinyxxx/pull/27) 已合并的通知。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1376719936197623849)** (2 messages): 

> `Future Cohorts, Course Scheduling` 


- **讨论未来 LLM Agent 课程班次**：一位成员询问了 LLM Agent 课程是否有 25 年夏季或 25 年秋季的班次。
   - 另一位成员回答说，25 年夏季不会开课，25 年秋季尚未确认，但如果开课，将在 8 月中旬开始。
- **关于即将到来的课程时间的推测**：在等待确认期间，推测 2025 年秋季班次的预计开始时间为 8 月中旬。
   - 讨论明确了虽然没有计划 2025 年夏季课程，但 2025 年秋季班次的可能性仍不确定。