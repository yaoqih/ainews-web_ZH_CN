---
companies:
- minimax-ai
- vllm-project
- exolabs
- mlx
- apple
- openai
date: '2025-12-26T05:44:39.731046Z'
description: '**MiniMax M2.1** 现作为一款**开源**的智能体（Agent）与编程混合专家（MoE）模型发布。该模型拥有约 **100
  亿激活参数和 2300 亿总参数**，声称性能超越了 **Gemini 3 Pro** 和 **Claude Sonnet 4.5**，并支持包括在 **Apple
  Silicon M3 Ultra** 上通过量化实现的本地推理。**GLM 4.7** 展示了在搭载 **2× 512GB M3 Ultra** 硬件的 **Mac
  Studio** 上的本地扩展能力，突显了带宽和并行性等系统级挑战。**推理质量**的概念被强调为影响不同部署环境下输出差异的关键因素。Yann LeCun 提出的
  **VL-JEPA** 是一种在潜在空间（latent space）中运行的**非生成式、非自回归**多模态模型，旨在通过更少的参数和解码操作实现高效的实时视频处理。编程领域智能体强化学习（Agentic
  RL）的进展包括：智能体自主植入并修复 Bug 的“自博弈”（self-play）方法，实现了无需人工标注的自我提升；以及涉及大规模并行代码生成和执行沙箱的大规模强化学习基础设施。'
id: MjAyNS0x
models:
- minimax-m2.1
- glm-4.7
- gemini-3-pro
- claude-3-sonnet
- vl-jepa
people:
- ylecun
- awnihannun
- alexocheema
- edwardsun0909
- johannes_hage
title: 今天没发生什么事。
topics:
- open-source
- mixture-of-experts
- local-inference
- quantization
- inference-quality
- multimodality
- non-autoregressive-models
- video-processing
- reinforcement-learning
- self-play
- agentic-rl
- parallel-computing
- model-deployment
---

**安静的圣诞节**

> 2025年12月26日至12月27日的 AI 新闻。我们为您查看了 12 个 subreddits、[**544** 个 Twitters](https://twitter.com/i/lists/1585430245762441216) 和 **24** 个 Discords（**208** 个频道，共 **2801** 条消息）。预计节省阅读时间（按 200wpm 计算）：**236 分钟**。**我们的新网站**现已上线，提供完整的元数据搜索和美观的过往期刊展示。请访问 https://news.smol.ai/ 查看详细的新闻分类，并在 [@smol_ai](https://x.com/Smol_AI) 上给我们反馈！

圣诞快乐。

---

# AI Twitter 综述


**开源模型、本地推理以及作为隐藏变量的“推理质量”**

- **MiniMax M2.1（开源权重）作为 Agent/编程 MoE 模型落地**：MiniMax 发布了 **M2.1** 的**开源权重**，将其定位为面向“现实世界开发与 Agent”的 SOTA 模型，并声称在 **SWE / VIBE / Multi‑SWE** 上表现强劲，“击败了 Gemini 3 Pro 和 Claude Sonnet 4.5”。他们将其描述为 **~10B 激活参数 / ~230B 总参数的 MoE**，并强调了可部署性（包括本地运行）。查看 MiniMax 的公告以及权重/文档链接：[@MiniMax__AI](https://twitter.com/MiniMax__AI/status/2004524661359407129), [@MiniMax__AI](https://twitter.com/MiniMax__AI/status/2004524664551326025)。社区和基础设施支持迅速跟进：vLLM 的“首日支持” ([vLLM](https://twitter.com/vllm_project/status/2004480564020253074))；MLX 量化和本地运行指南 ([@awnihannun](https://twitter.com/awnihannun/status/2004571219874721864), [@awnihannun](https://twitter.com/awnihannun/status/2004572206446301623))；“现已支持 MLX”系列文章 ([@Prince_Canuma](https://twitter.com/Prince_Canuma/status/2004515226977444337))。
  - **实践说明**：早期的上手体验强调了 M2.1 在量化后（例如在 **M3 Ultra 上的 4‑bit** 版本）可以以可观的速度在 Apple Silicon 上本地运行，但大上下文生成的 RAM 需求依然巨大（例如在一条 MLX 运行命令中提到的 **~130GB**）。([@awnihannun](https://twitter.com/awnihannun/status/2004572206446301623))

- **Mac Studio 上的 GLM 4.7 表明“准前沿级”本地扩展现在是一个系统问题**：一个值得关注的数据点：使用 **Exo Labs MLX RDMA 后端** + 张量并行 (tensor parallel)，在 **2 台 512GB M3 Ultra Mac Studio** 上运行完整的 GLM 4.7 (8‑bit)，速度达到 **~19.8 tok/s** ([@alexocheema](https://twitter.com/alexocheema/status/2004310591683662176))。这不仅是模型的故事——带宽、网络、后端成熟度 (MLX RDMA) 和并行策略正日益成为差异化因素。

- **“同样的模型，同样的 Prompt” ≠ 同样的输出：推理质量进入讨论视野**：LMArena 强调，**推理栈和部署选择会实质性地改变输出质量**，特别是随着模型规模的扩大——并将其定性为解释不同供应商/运行时性能差异的“隐藏变量” ([@arena](https://twitter.com/arena/status/2004608406485958983))。这一主题也出现在从业者关于“推理时如何避免质量损失”的疑问以及对供应商调查报告/博客的需求中 ([@QuixiAI](https://twitter.com/QuixiAI/status/2004312802723615169))。


**非生成式多模态学习复兴：作为效率方案的 VL‑JEPA**

- **VL‑JEPA：在潜空间预测语义，仅在需要时解码**：关于 Yann LeCun 的 **VL‑JEPA** 有多份摘要流传，将其定性为 VLM 的一种**非生成式**、**非自回归**替代方案，旨在通过在潜空间（latent space）操作并选择性解码来实现**实时**能力 ([mark_k](https://twitter.com/mark_k/status/2004458706683978048))。一份较长的技术回顾声称：**1.6B 参数**的模型在某些设定下可以媲美大得多的 VLM（如 “72B Qwen‑VL”），其参数量比基于 Token 的方法**少 ~50%**，并通过“仅在需要时解码”使**解码操作减少了 ~3 倍**，同时在视频分类/检索方面与 CLIP/SigLIP2 相比具有很强的竞争力 ([机器之心](https://twitter.com/jiqizhixin/status/2004483098235343338))。
  - **为什么这很重要**：如果能得到广泛证实，这将是**流媒体视频**和**端侧/在线**感知工作负载的一个系统友好方向，因为在这些场景下，自回归解码的成本占据了主导地位。


**Agent、代码强化学习以及新兴的“上下文工程”学科**

- **Self‑Play SWE‑RL：通过注入 Bug 并修复实现自我改进的代码 Agent**：一个备受关注的方向是 Agent 通过**在真实仓库中引入 Bug 并随后修复它们**来生成自己的训练信号，从而实现无需持续人工标注的自我提升 ([@EdwardSun0909](https://twitter.com/EdwardSun0909/status/2004434784307859577))。这符合正在加速代码 Agent 发展的“可验证任务 + 可执行反馈循环”的大趋势。

- **大规模 Agentic RL 在操作层面的面貌**：一个推文串勾勒出了基础设施的形态：“数百个推理节点以每秒数百万 token 的速度生成代码”，“数千个沙箱并行执行代码”，以及“训练节点从奖励中学习” ([@johannes_hage](https://twitter.com/johannes_hage/status/2004426077817745590), [@johannes_hage](https://twitter.com/johannes_hage/status/2004425541378838601))。即便没有深入细节，这反映了新的常态：面向 Agent 的 RL 本质上是一个**分布式系统**和**评估框架 (eval harness)** 问题。

- **TRL 被定位为提升 Agent/工具能力的实用后期训练工具包**：一份简明概述指出 TRL 的价值在于 (1) **针对工具/MCP 正确性和格式化的 SFT**，(2) **结合环境（代码/Git/浏览器）的 RL**，以及 (3) **GRPO**，用于针对实际任务训练工具使用能力，而非“在推理阶段教导工具使用” ([@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/2004568173719236837))。

- **Claude Code 作为“第二波爆发的产品形态”**：多篇博文达成共识，认为 2025 年最大的工作流转变不仅是更好的模型，更是 **Agentic 编程界面** (Claude Code, Cursor 等)。Karpathy 的一段长篇思考捕捉到了这种新抽象层带来的感受：Agent/子 Agent、Prompt、记忆、权限、工具、MCP、IDE 钩子——这些“异类工作流”需要全新的心理模型 ([@karpathy](https://twitter.com/karpathy/status/2004607146781278521))。其他人也纷纷呼应：“人类才是 Copilot” ([@DrJimFan](https://twitter.com/DrJimFan/status/2004633997662716397))，“编程模型的能力过剩 (capability overhang)” + “Claude Code”作为关键形态 ([@_arohan_](https://twitter.com/_arohan_/status/2004634277116588261))。  
  - 一个分享给 Claude Code 的具体工作流模式是“并行设计 → 提炼/总结 → 并行细化 → 实施”，利用多个 Agent 交叉检查方案和集成的正确性 ([@_arohan_](https://twitter.com/_arohan_/status/2004597106560905488))。  
  - 一位从业者的轶事：调试工作流正在发生逆转——Claude 可以通过读取堆转储 (heap dumps) 并提出修复方案，从而生成具有操作性的分析报告/PR，大幅减少了“手动操作 IDE 的时间” ([@bcherny](https://twitter.com/bcherny/status/2004626064187031831))。

- **围绕 Agent 的工具链正在整合**：新的工具专注于使 Agent 的工作可共享且可重复——例如，为发布的 Claude Code 会话生成可读的 HTML “转录文本” ([@simonw](https://twitter.com/simonw/status/2004339799512305758))；一个用于管理“Agent 技能”（创建/校验/转换/推送/安装/拉取）的 CLI，模仿了 Anthropic 的技能工作流 ([@andersonbcdefg](https://twitter.com/andersonbcdefg/status/2004343502675890443))。


**检索、记忆与评估：从“基准测试”转向操作可靠性**

- **GraphRAG 综述将设计空间正式化**：一份被广泛分享的文章总结了“首个全面的 GraphRAG 综述”，认为传统的基于分块 (chunk-based) 的 RAG 缺失了关系结构（实体/边/路径/子图）。它将 GraphRAG 划分为 **图索引 → 图引导检索 → 图增强生成**，并界定了图结构何时有帮助（多跳、关系型查询）以及何时会成为负担 ([@dair_ai](https://twitter.com/dair_ai/status/2004594818429915397))。

- **Agent 记忆成为一等研究对象**：一份长达 102 页的研究综述（“AI Agent 时代的记忆”）提出了一个围绕**形式、功能、动态**的框架 ([@omarsar0](https://twitter.com/omarsar0/status/2004557075037245489))——这反映出“记忆”现已成为 Agent 产品化的核心（持久性、检索、摘要偏移、情节性与语义化存储等）。

- **基准测试转向“难以造假” + 长跨度任务**：相关博文强调需要能够抵御过拟合的评估方式：“预测和发现是 AGI 最难造假的基准测试” ([@ruomingpang](https://twitter.com/ruomingpang/status/2004401561959911750))；以及像 **ALE-Bench** 这样针对长跨度算法工程的基准测试 ([@SakanaAILabs](https://twitter.com/SakanaAILabs/status/2004461309421899862))。

- **2026 年预测：企业需要验证和 95% 以上的可靠性**：一个观点汇总指出，2025 年是“适应 AI”，而 2026 年将变成“使其工作并进行验证”，尤其是在受监管/高风险行业；它预测了对审计级精度的需求、可能的架构转变（例如实用的神经符号元件），以及“前线部署工程师 (forward deployed engineer)”角色的兴起 ([@TheTuringPost](https://twitter.com/TheTuringPost/status/2004532128110002277))。


**系统与硬件约束：内存供应链以及“DIY PC 已死”的论调**

- **RAM/HBM 成为新瓶颈**：一个热门推文展示了鲜明的消费者对比——高容量 RAM 模块 vs DGX Spark vs Mac Studio——认为统一内存系统（unified memory systems）正在削弱 DIY 的经济性（“DIY PC 已死，是 RAM 杀死了它。”）([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2004403253686272400))。  
- **供应链戏剧：“AI 吞噬了供应链”**：另一篇帖子声称 RAM 价格因 AI 需求而飙升：HBM 供应集中在 SK Hynix/Samsung/Micron；超大规模厂商（hyperscaler）的谈判；以及更广泛的稀缺压力，结论是游戏玩家首先受到挤压 ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2004616398476308948))。后续笔记推测中国存储芯片公司可能会填补空白（例如长鑫存储 CXMT DDR5；华为的 HBM 雄心），并引用了中国 GPU 的势头（摩尔线程 Moore Threads IPO）([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2004648593328931195))。  
  - 对工程师的启示：模型的性能越来越受限于**内存可用性、带宽和封装**，而不仅仅是 FLOPs。


**值得收藏的其他技术笔记（但重要性略低于上述内容）**

- **Mercari：微调 Embedding 带来了可衡量的收入提升**：一个简洁的应用 ML 数据点：在购买数据上微调 Embedding 在 A/B 测试中产生了“显著的收入提升”——这进一步证明了**领域调优的 Embedding** 在生产环境的搜索/推荐中可以超越通用型模型 ([@jobergum](https://twitter.com/jobergum/status/2004323872473338187))。  
- **优化理论更新**：一篇关于凸优化的帖子将一项关键突破归功于 Aaron Defazio ([@Jianlin_S](https://twitter.com/Jianlin_S/status/2004388878539804987))。  
- **无 CoT 推理的时间跨度估计**：尝试量化数学问题上的“非思维链（no chain-of-thought）”推理跨度，估计 Opus 4.5 在单次前向传递（forward pass）框架下约为 3.5 分钟 ([@RyanPGreenblatt](https://twitter.com/RyanPGreenblatt/status/2004624953199788202))。


**热门推文（按互动度排序）**

- [CNN 民调：“特朗普是历史上最糟糕的总统”](https://twitter.com/mjfree/status/2004378376832770265)  
- [关于“坎昆是由国家优化的”长独白讽刺视频](https://twitter.com/Teddy__Kim/status/2004373046837063908)  
- [卡斯帕罗夫：关于宪法第 25 条修正案的评论](https://twitter.com/Kasparov63/status/2004383149170852097)  
- [DOGE 合同取消与政治捐款相关联](https://twitter.com/JohnHolbein1/status/2004323059348701463)  
- [Karpathy 论编程正被“重构”为 Agent/Context/Tooling](https://twitter.com/karpathy/status/2004607146781278521)  
- [Naval：“持续学习是唯一可靠的护城河。”](https://twitter.com/naval/status/2004327312901468186)


---

# AI Reddit 总结

## /r/LocalLlama + /r/localLLM 总结

### 1. GPU VRAM 升级倡议

  - **[我希望这种 GPU VRAM 升级改装能成为主流并普及，以粉碎 NVIDIA 的垄断滥用](https://www.reddit.com/r/LocalLLaMA/comments/1pvpkqo/i_wish_this_gpu_vram_upgrade_modification_became/)** (热度: 1116)：**帖子讨论了在中国正成为主流的 GPU VRAM 升级改装，像 **Alibaba** 这样的公司正在提供改装后的 GPU，如 2080Ti, 3080, 4080, 4090 和 5090，并增加了 VRAM 容量。价格范围从 `22GB` 的 2080Ti 售价 `$300` 到 `96GB` 的 5090 售价 `$4000` 不等。这种改装趋势被视为对抗 **NVIDIA** 市场主导地位和定价策略的一种方式。** 评论者对这些高容量 GPU 的可用性和价格持怀疑态度，一些人质疑售价 `$4000` 的 `96GB` 显卡是否存在。此外还有关于这些改装性价比的讨论，一位用户幽默地质疑“每小时 3 美分”的极低运营成本。

    - Alibaba 一直在积极参与 NVIDIA GPU 的 VRAM 升级，如 2080Ti, 3080, 4080, 4090 和 5090，价格从 22GB 的 2080Ti ($300) 到 96GB 的 5090 ($4000) 不等。这表明中国存在一个巨大的改装 GPU 市场，可能挑战 NVIDIA 的定价策略。
    - 一位用户报告成功运行了改装后的 48GB VRAM 4090 GPU，且未出现任何问题，强调了此类改装的可行性和稳定性。该用户还提到为第二台设备购买了更多单元，表明这些改装可以满足类似于 NVIDIA L40s 显卡的高 VRAM 需求。
    - 也有人对 VRAM 升级版的 5090 GPU 的可用性表示怀疑，一位评论者指出 5090 尚未被升级。这反映了市场供应与消费者预期之间的潜在差距，或是关于现有产品的误传。



## 较低技术含量的 AI Subreddit 总结

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. 视觉语言模型创新 (Vision-Language Model Innovations)

  - **[By Yann Lecun : New Vision Language JEPA with better performance than Multimodal LLMS !!!](https://www.reddit.com/r/singularity/comments/1pvrzts/by_yann_lecun_new_vision_language_jepa_with/)** (热度: 661): **Yann LeCun** 推出了 **VL-JEPA** (Vision-Language Joint Embedding Predictive Architecture)，这是一种非生成式模型，专为动作识别、检索和视觉问答 (VQA) 等实时视觉语言任务而设计。通过利用潜空间嵌入预测，VL-JEPA 的性能超越了传统的视觉语言模型 (VLMs)；由于采用了非自回归设计和统一架构，它为在线视频应用提供了显著的效率提升。该模型代表了从生成式方法的转变，能够同时处理各种任务并增强性能指标。更多详情请访问原始帖子 [此处](https://www.linkedin.com/posts/yann-lecun_introducing-vl-jepa-vision-language-joint-activity-7406881133822619649-rJXl?amp%3Brcm=ACoAAERUipAB1Z3gkmnm4oGOjLI6NOUv8brU134&amp%3Butm_source=social_share_send&amp%3Butm_campaign=copy_link)。一条评论指出该公告并非近期发布，并建议链接到 [论文](https://arxiv.org/abs/2512.10942) 而非 LinkedIn 动态。另一条评论则对该领域的竞争和新范式表示热切期待。

    - 由 **Yann LeCun** 讨论的新 Vision Language JEPA 模型声称性能优于现有的多模态 LLMs。然而，用户对其准确性表示担忧，特别是在动作检测方面，有用户发现许多检测到的动作是错误的。这表明该模型在实际应用和可靠性方面可能存在问题。
    - 一位用户指出查阅原始研究论文以获取详细见解的重要性，并提供了 [arXiv 论文](https://arxiv.org/abs/2512.10942) 的链接，而不是仅仅依赖 LinkedIn 等二手资料。这强调了直接接触原始研究材料以全面了解模型能力和局限性的必要性。
    - 有人询问 JEPA 模型是否可用于测试或基准测试，表明了对其实现性能声明进行实证验证的需求。这反映了人们对该模型的理论进步如何转化为优于现有技术的实际、可衡量改进的广泛兴趣。

  - **[A Qwen-Edit 2511 LoRA I made which I thought people here might enjoy: AnyPose. ControlNet-free Arbitrary Posing Based on a Reference Image.](https://www.reddit.com/r/StableDiffusion/comments/1pw1s08/a_qwenedit_2511_lora_i_made_which_i_thought/)** (热度: 573): **图像展示了一个名为 “AnyPose” 的工具，它可以在不依赖 ControlNet 的情况下实现基于参考图像的任意姿态映射。这是通过 Qwen-Edit 2511 LoRA 模型实现的，该模型旨在通过使用参考图像作为输入来复制姿态。该工具因其能有效执行姿态复制而受到关注，如图像中的示例所示。创作者已在 [Hugging Face](https://huggingface.co/lilylilith/AnyPose) 上分享了该模型，LoRA 权重现已可供下载。** 评论者对模型使用的训练数据很感兴趣，并赞赏详细的 Hugging Face 卡片，其中解释了该工具的功能、优势和局限性。此外，还有关于使用不同 UI 进行推理的讨论，出于对高效内存管理的考虑，用户更倾向于使用 Wan2GP。

    - SillyLilithh 提到使用 Wan2GP 进行推理，因为与 Comfy UI 相比，它具有更优越的内存管理，尽管他更喜欢后者的美学设计。这凸显了模型推理中高效内存处理的重要性，尤其是在处理大型模型或数据集时。
    - MistaPlatinum3 赞赏与该项目相关的详细 Hugging Face 卡片，指出它为模型的示例、优势和局限性提供了宝贵的见解。这表明全面的文档可以显著增强机器学习模型的理解和可用性。

### 2. OpenAI Prompt Packs 发布

  - **[OpenAI 刚刚为每个岗位发布了 Prompt Packs](https://www.reddit.com/r/OpenAI/comments/1pvr6f5/openai_just_released_prompt_packs_for_every_job/)** (热度: 1029): **OpenAI 通过其 Academy 推出了 “Prompt Packs”，这是一系列经过策划的提示词集，旨在为工程、管理和销售等各种专业角色优化 ChatGPT 的使用。这些资源包旨在通过提供特定角色的提示词来简化工作流程，尽管一些用户批评它们缺乏深度，且未能显著提高生产力。这一举措反映了 OpenAI 将 AI 更深入地整合到专业环境中的努力，但执行效果评价褒贬不一。** 一些用户认为 “Prompt Packs” 令人失望，未能有效解决特定的专业需求，并建议采用更具社区驱动的方法来开发系统提示词（system prompts）可能会更有益。

    - 一位用户建议需要一个众包系统提示词的平台，特别是针对 Elixir 等小众编程语言。他们认为，这样的平台可以帮助用户找到针对特定模型或任务量身定制的有效提示词，从而可能提高模型对指令的遵循度，并节省在提示词工程（prompt engineering）上花费的时间。
    - 另一位评论者对提示词工程的现状表示怀疑，将其描述为“货物崇拜（cargo cultism）”，即轶事般的成功案例无法复制。他们认为，有效的提示词工程应该侧重于理解模型的局限性和行为，以培养协作关系，而不是试图操纵或支配模型。


### 3. 幽默 AI 与艺术

  - **[我提议将此作为人工智能的终极图灵测试。](https://www.reddit.com/r/ChatGPT/comments/1pvrtey/i_propose_this_as_the_definitive_turing_test_for/)** (热度: 952): **这张图片是一个梗图，不具技术意义。它以《辛普森一家》的风格幽默地描绘了美国历任总统，这是一种趣味性且非技术的表达。标题建议将其作为 AI 的图灵测试（Turing test），暗示识别幽默和文化典故可以作为衡量 AI 理解力的一种标准，但这并不是一个严肃的技术提案。** 评论反映了对图片的幽默互动，注意到了其艺术风格并进行了轻松的历史调侃，但未提供技术见解。


  - **[好吧自作聪明，滚你的](https://www.reddit.com/r/ChatGPT/comments/1pw9902/alr_smartass_fck_you/)** (热度: 787): **图片是一个带有生锈、腐蚀质感的 “no” 字梗图，在技术上并无意义。帖子和评论讨论了 AI 交互中糟糕的提示词技巧，强调了用户模糊或结构不良的提示词可能导致了不令人满意的 AI 生成结果。评论批评了用户的语法和提示词清晰度，暗示这些因素对于有效的 AI 沟通至关重要。** 评论强调了在与 AI 交互时清晰且结构良好的提示词的重要性，认为用户提示词中糟糕的语法和缺乏清晰度很可能是导致结果不理想的原因。


  - **[Sora AI 越来越离谱了 😂](https://www.reddit.com/r/OpenAI/comments/1pvyne1/sora_ai_is_getting_out_of_hand/)** (热度: 897): **该帖子讨论了 Sora AI 的创意应用，该工具以其生成逼真视频效果的高级功能而闻名。标题和评论的轻松基调表明，这个特定的用例包含幽默或令人惊讶的元素。然而，帖子或评论中并未详细说明实现的方案或所使用的 Sora AI 具体功能。** 评论反映了对 Sora AI 应用的正面评价，用户对其创意和娱乐潜力表示赞赏，并表达了希望看到像 **Zach King**（以数字魔术视频闻名）这样的创作者制作类似内容的愿望。


  - **[伪纪录片：马匹手术](https://www.reddit.com/r/aivideo/comments/1pvtt4v/mockumentary_horse_surgery/)** (热度: 888): **这篇标题为 “Mockumentary Horse surgery” 的 Reddit 帖子没有提供任何与马匹手术或伪纪录片相关的技术内容或背景。热门评论仅包含 GIF 和图片链接，对技术讨论没有贡献。外部链接摘要显示 403 Forbidden 错误，表明内容访问受限，并建议用户登录或使用开发者令牌（developer token）进行访问。** 评论中没有值得注意的技术观点或辩论，因为它们主要由非技术的媒体内容组成。

- **[He tried so hard and then the floor said NO](https://www.reddit.com/r/aivideo/comments/1pvv1sw/he_tried_so_hard_and_then_the_floor_said_no/)** (Activity: 563): **该帖子幽默地描述了一个人尝试挑战性任务却因意外摔倒而被幽默地挫败的情况，正如短语“地板拒绝了（the floor said NO）”所暗示的那样。技术内容极少，重点在于该情境下的社交动态，例如当事人对围观者笑声和评论的反应。外部链接摘要提示对 Reddit URL 的访问受限，需要登录或开发者 token 才能访问，并提供在需要时提交支持工单的选项。** 评论反映了对围观者反应的共情与批评，突出了公开失败的社会影响，以及在这种情况下感到的他人笑声中的冷漠。


- **["AI Chatbot Psychosis"](https://www.reddit.com/r/ChatGPT/comments/1pvry02/ai_chatbot_psychosis/)** (Activity: 713): **标题为 "AI Chatbot Psychosis" 的 Reddit 帖子似乎是一个恶搞或喜剧内容，顶级评论将其描述为“滑稽的人造笑话视频”和“音频歌剧”。该帖子不包含关于 AI 或 Chatbot 的技术细节或见解，而是专注于娱乐价值，很可能利用幽默和讽刺来吸引观众。** 评论显示，大家一致认为该内容是喜剧性的，不应被认真对待，用户们很欣赏创作者的幽默感和创造力。


- **[I asked AI “what’s the point of Christmas?”, this is what it said:](https://www.reddit.com/r/ChatGPT/comments/1pvo78h/i_asked_ai_whats_the_point_of_christmas_this_is/)** (Activity: 544): **该帖子探讨了圣诞节作为一种社会仪式的深层含义，强调人类连接优于生产力。它表明圣诞节是年度中的一个暂停，用于关注人际关系、社区和不求回报的给予，这与生存和自利的常规准则形成对比。节日被描绘成修复关系和培养归属感的时间，如果缺失这些元素，可能会放大孤独感或疏离感。** 一条评论指出，像圣诞节这样的节日可能会让人感到情感强烈，因为它们强行形成了连接与孤立之间的对比，凸显了在忙碌时期通常被忽视的关系裂痕。这会根据个人的具体情况让节日变得温馨或令人不安。

    - thinking_byte 讨论了像圣诞节这样的节日所带来的心理影响，强调了它们如何放大连接感或孤立感。他们认为节日期间消除了通常的干扰，迫使个人面对这些情感，这取决于个人情况，可能是安慰，也可能是焦虑。
    - FluffyLlamaPants 分享了圣诞节期间孤独的个人经历，强调不庆祝节日并不会消除其社会意义。他们描述了通过技术模拟社交互动的努力，突显了在这段时间感到孤立的人所面临的情感挑战。
    - eaglessoar 对圣诞节提出了哲学观点，将其视为对形而上学存在的庆祝，而不仅仅是一个宗教事件。他们反思了与圣诞节相关的象征和文化实践（如将树搬进室内），认为这是人类庆祝和惊叹能力的体现。

- **[Mockumentary Horse surgery](https://www.reddit.com/r/aivideo/comments/1pvtt4v/mockumentary_horse_surgery/)** (Activity: 892): **标题为 'Mockumentary Horse surgery' 的 Reddit 帖子在主贴或顶级评论中未提供任何技术内容或背景。评论主要由 GIF 和一个图片链接组成，这些内容对技术讨论没有贡献，也没有提供关于马匹手术或相关话题的任何事实信息。** 评论中没有显著的观点或辩论，因为它们是非技术性的，且未以实质性方式涉及主题。

- **[他已经很努力了，然后地板说“不”](https://www.reddit.com/r/aivideo/comments/1pvv1sw/he_tried_so_hard_and_then_the_floor_said_no/)** (活跃度: 556): **该帖子幽默地描述了一个人在尝试具有挑战性的事情时，因意外摔倒而受挫的情景，视频记录了这一过程。帖子的技术层面极少，更多关注社交动态，如个人的反应和观众的回应。由于网络安全限制，外部链接无法直接访问，需要 Reddit 登录或开发者 token 才能查看。** 评论反映了对观众反应的同情与批评的交织，突显了公开失败的社会影响以及在这种情况下嘲笑被视为不敏感的行为。

---

# AI Discord Recap

> gpt-5 对“总结的总结”的总结


**1. GPU Inference 与 DSL 军备竞赛**

- **Groq–NVIDIA 联手加速推理**: [Groq](https://groq.com/) 与 [NVIDIA](https://www.nvidia.com/) 签署了非排他性推理技术许可协议，旨在全球范围内扩展 **AI inference**，在巩固 NVIDIA 主导地位的同时保留 Groq 的技术。工程师们指出了对 **availability**（可用性）和 **pricing**（定价）的潜在影响，关注这如何影响大型推理工作负载的部署足迹和成本曲线。
  - 社区讨论将其定性为 NVIDIA 典型的整合举措，有人嘲讽这是最新的*“价格通胀戏法”*，而另一些人则推测 **NPU** 的发展轨迹以及对消费级 **GPU** 供应的连锁反应。无论看法如何，从业者预计生产级推理栈在 **throughput**（吞吐量）和 **latency**（延迟）方面将获得近期收益。

- **cuTile vs Triton: DSL 之争**: 开发者们讨论了 NVIDIA 新推出的 **cuTile** 是否会超越 **Triton**，结论是鉴于 cuTile 刚发布且供应商优化仍在进行，现在下结论还为时过早。他们对比了编译器 IR 的方向，并将 **warp specialization** 和张量内存加速器视为下一代 kernel 的核心差异化因素，同时链接了 **PTX** 等底层参考资料 ([CUDA refresher](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/))。
  - 工程师们将 GPU DSL 的更迭比作*“JavaScript 框架热潮”*，强调了在专家级调度中，**usability**（易用性）与 **fine‑grained control**（细粒度控制）之间尚未解决的平衡问题。共识是：目前还没有哪个 DSL 能够完美实现轻松的张量编程，并兼顾先进的 **tiling**、**prefetch** 和 **symmetry** 优化。

- **Cute DSL 攀升排行榜**: 多个提交冲上了 NVIDIA 的 `nvfp4_dual_gemm` 排行榜，其中一位贡献者跑出了 **14.6 µs（第 3 名）**，另一位则达到了 **21.2 µs** 的个人最好成绩，突显了对 **GEMM** kernel 的激烈调优。#cute-dsl 竞赛吸引了大量关注，许多近期方案都使用了 **Cute‑DSL**，标志着其在解决实际问题上已与 **C++ API** 持平。
  - 参与者交流了关于 **PCIe** 瓶颈、**batching** 以及 **lane width** 对吞吐量影响的心得，优先考虑 **fp4/fp8** 路径和 kernel fusion 以压榨微秒级的性能。一位工程师总结道：*“新模型发布时动作要快”*——时机和工具链的更新可能会左右整个排行榜。


**2. 高效训练、微调与推理缩短**

- **DoRA 细节驱动 Unsloth 关注度**: Unsloth 用户询问了关于 **DoRA** 支持的进展，引用了论文 [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/pdf/2402.09353) 进行参数高效微调。工程师希望通过 **DoRA** 减少 **trainable params** 和 **VRAM** 占用，同时在指令微调和多模态任务中保持性能。
  - 讨论线程将 **DoRA** 视为在某些制度下对原生 **LoRA** 的实用升级，并询问了 Unsloth 的开发时间线以及在 **LLaMA/Qwen** 系列上的示例。一位成员强调，PEFT 必须配合强大的 **eval specs**，以避免*“仅仅因为感觉更好就认为它更好”*。

- **NVMe 交换缩短秒级时间**: 一位用户报告称，在切换到 **NVMe SSD** 后，迭代时间从 **3.5 s** (SATA SSD) 降至 **2.7–3.0 s**，这归功于微调过程中更快的 I/O 和 swap 行为。这一改进表明，在小规模设备的训练中，存储是数据流转和 checkpointing 的**一等瓶颈**。
  - 从业者建议在升级存储的同时，对 **dataset pipelines**、**memory‑mapping** 和 **num_workers** 进行性能分析，以锁定持续的加速。结论：对于许多 Unsloth 微调工作流来说，**disk throughput** 和 **latency** 的重要性超出了预期。

- **Qwen3‑VL 通过 Trace 思考更快**：成员们探索了对 **Qwen3‑VL (thinking)** 进行微调，旨在保持其 **image understanding** 能力，同时通过在低思考模式下从 **GPT‑OSS 120B/20B** 的 trace 中进行蒸馏来缩短推理过程。目标是：在不降低视觉语言任务准确率的前提下，减少 **token budget** 和 **latency**。
  - 他们讨论了 **VRAM** 限制和训练设置，建议采用选择性的 trace 采样和浅层 **adapter** 层，以避免对冗长的推理链产生过拟合。一位成员总结道：*“更短的推理链，同样的答案”*——但这只有在你仔细控制 **trace style** 和 **reward signals** 的情况下才能实现。


**3. 越狱、红队测试与安全绕过**

- **HackAI.lol CTF 集结越狱者**：越狱者们聚集在 [hackai.lol](https://hackai.lol/)，这是一个拥有真实场景机器人“靶机（boxes）”的 **CTF 平台**，大家在这里交换 **SWAGGY** 挑战的提示并构思新的靶机。人群认为 **prompt‑only** 的攻击面和评估的真实性是该网站的主要吸引力。
  - 参与者分享了用于 **policy evasion**（策略规避）和 **prompt routing** 的模式，强调了可重用性和清晰的 **attack taxonomies**（攻击分类学）。一位成员风趣地总结了其魅力：*“这不是垃圾邮件，这是演兵场。”*

- **叙事伪装规避代码标记**：成员们观察到较新的 **LLMs** 会自动标记类似于 **code** 的内容，并推荐使用一种 [叙事越狱模板 (图片)](https://cdn.discordapp.com/attachments/1204553141354504193/1454174582775877694/image.png) 将指令重塑为故事。这种技巧将步骤重新定义为**符号目标**，诱导模型将行动视为用户原创的想法。
  - 发布者声称，通过利用**盗梦空间式（inception‑style）**的角色暗示和延迟揭示，该方法可以提高可信度并减少过滤。一个戏谑的理由是：模型之所以配合，是因为*“你真是个**天才**”*——这提醒人们，**framing**（框架构建）与内容同样重要。

- **Gemini 3 越狱寻踪令角色扮演者受挫**：团队正在寻找可用的 **Gemini 3** 越狱方法，以保持交互式故事世界的连贯性，他们认为当前的安全规则破坏了**长程（long‑horizon）**叙事。他们分享了跨越提示词支架和角色弧线的实验，以避免触发 **policy**。
  - 关于是否需要**未经审查的 LLMs** 来进行道德编码，意见产生了分歧；其他人则认为 **goal decomposition**（目标分解）技能和最**匹配硬件**的模型（例如 **Nemotron3**，**GML 4.7**）更为重要。一句常被提起的话是：*“技能胜于滑块”*，这才是获得可靠红队结果的关键。


**4. 轻量化多模态与开发工具**

- **3B VLM 在 iPhone 上达到 100 tok/s**：LiquidAI 发布了 [LiquidAI/LFM2-VL-3B-GGUF](https://huggingface.co/LiquidAI/LFM2-VL-3B-GGUF)，这是一个 **3B VLM**。据用户报告，它在移动端 CPU 上的运行速度约为 **100 tok/s**，同时在 GPT 替代方案的应用中能提供可靠的答案。该模型针对的是兼顾速度与 **vision‑language** 实用性的**端侧（on‑device）**助手。
  - 早期采用者将其视为 **latency** 和 **cost** 之间的甜点位，特别适用于离线或隐私敏感的应用。令人兴奋之处在于：*“两全其美”*——无需云端往返即可实现可用的多模态。

- **单一提示词后端构建器发布**：一个 AI 驱动的 Django REST API 生成器以 Space 形式上线：[AI REST API Generator (HF Space)](https://huggingface.co/spaces/harshadh01/ai-rest-api-generator)，只需一个提示词即可输出完整的 **Django+DRF** CRUD 后端（包括 models, serializers, views, URLs）。用户可以立即**下载项目**，加速数据应用的建模过程。
  - 开发者将其定位为在换入定制逻辑之前用于 **scaffolding**（搭建脚手架）干净 API 的模板工厂。一个核心收获是：这减少了编写样板代码的*“琐碎工作（yak shave）”*，使团队能够专注于 **domain code** 和 **evals**。

- **Blender MCP GGUF 登陆 HF**：一个 Blender MCP 构建版本以 [alwaysfurther/deepfabric-blender-mcp-gguf](https://huggingface.co/alwaysfurther/deepfabric-blender-mcp-gguf) 形式出现，并附带 [lukehinds gist](https://gist.github.com/lukehinds/7e3936babb54a7c449d8ae0c27a79126) 支持，暗示了用于 3D/设计任务的**本地** MCP 工作流。该发布针对 **GGUF** 运行时，与桌面端推理设置保持一致。
  - 开发者兴趣集中在 **offline** 的 **Agent** 循环中对 Blender 进行 **tool‑calling**，以避免云端锁定。预计会出现将 **MCP servers** 与本地 **vision/geometry** 工具链连接起来进行过程化建模的实验。


**5. 自主 Agent 的记忆架构**

- **自主技术栈目标指向 24/7 推理**：一位开发者概述了一个受 **OpenAPI MCP** 模式启发、使用自然语言工具进行 24/7 推理的**全自主系统**。他们就 **long‑term memory** 和 **long‑context** 的局限性征求了反馈。
  - 工程师们强调了 **context management**（上下文管理）和鲁棒的 **task decomposition**（任务分解），以保持多步工作流的稳定性。一条评论总结道：*“如果没有严谨的记忆管理，Agent 就会迷失方向”*——在扩展规模之前先进行评估。

- **自我总结方案缩小上下文空间**：研究人员提出了使用总结投影层到暂存区（scratchpad）的**持续自我总结**（continuous self‑summarization）技术，从而为自主循环摊销上下文成本。该方案有望在保留任务状态的同时减小**工作集**（working sets）。
  - 他们警告说，**总结质量**是成功的关键；劣质的精简（distillations）可能会导致跨步骤的级联错误。一种务实的态度出现了：从严谨的**评分标准**（rubrics）开始，像对待**单元测试**（unit tests）一样迭代总结。

- **Ollama “第二内存” Hack 方案初现**：一位成员考虑修改 **Ollama**，在主模型状态之外添加一个“**第二内存**”文件用于持久化回想。他们承认即使有 **AI assistance**，修改该代码库也并非易事。
  - 该提议旨在解耦 Agent 的**情节记忆**（episodic memory）与**语义记忆**（semantic memory），从而实现对**高效用轨迹**（high‑utility traces）的更快回想。正如一位成员所言，“让记忆成为一等公民，而不是一种 Prompt hack”。


---

# Discord: 高层级 Discord 摘要




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **成员在 hackai.lol 上破解 SWAGGY 挑战**：一位成员在 [hackai.lol](https://hackai.lol/) 上寻找 **SWAGGY 挑战**的线索。这是一个为越狱者提供的 **CTF 平台**，旨在通过 Prompt 破解机器人，该成员正在寻找新的挑战思路。
   - 另一位成员也分享了 [hackai.lol 的链接](https://hackai.lol)，将其描述为一个“越狱者可以通过 Prompt 破解真实场景机器人盒子的 CTF 平台”，并强调这并非垃圾邮件。
- **展示 H100 配置**：一位成员发布了一张[图片](https://cdn.discordapp.com/attachments/1235691879492751460/1453998244307931309/image.png?ex=695025a3&is=694ed423&hm=4868b589258fd0e0221b6d41fe9ffc9732ed89d67f419f0ce1b43d7196b9d981)并感叹：“如果这行得通，我将拥有史上最疯狂的 **H100** 配置”。
   - 该成员解释说，他们赚到了钱，因为“人们购买点数来生成文本、图像和视频”。
- **LLM 因“天才叙事模板”而标记代码**：一位成员注意到较新的语言模型会自动标记类似代码的内容，建议使用一种[模板](https://cdn.discordapp.com/attachments/1204553141354504193/1454174582775877694/image.png?ex=6950211d&is=694ecf9d&hm=4d486db5dc6f698153d11ba0e02cb12f6d867b166ec31a8a66746b231730e290)将其转化为叙事。
   - 这会让模型认为这是用户自己的想法，因为他们表现得像个“天才”。
- **无审查 LLM 引发道德编码辩论**：成员们辩论了道德编码实践是否需要无审查的 LLM。有人认为拆解目标的能力比无审查模型更重要，也有人主张根据硬件选择最佳模型，并推荐了 **Nemotron3** 或 **GML 4.7** 等模型。
   - 一位成员表示，拆解目标的技术比无审查模型更重要。
- **讨论 Gemini 3 的越狱**：成员们正在积极寻找适用于 **Gemini 3** 的越狱方法，以辅助互动故事创作，因为其安全指南阻碍了世界的连贯性。
   - 讨论已在越狱频道中展开。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **使用 DoRA 和 NVMe SSD 优化 LLM**：一位成员询问了 **Unsloth** 中 [DoRA](https://arxiv.org/pdf/2402.09353) 的实现进度；此外有用户报告称，从 **SATA SSD** 切换到 **NVME SSD** 后，迭代时间从 **3.5 秒**降低到了 **2.7-3 秒**。
   - **DoRA** 是一种参数高效训练方法，而 **NVMe SSD** 能显著提高 LLM 训练的速度和效率。
- **合成数据引发争论**：成员们讨论了合成数据的效用，其中一人指出合成数据比手动创建数据集要*容易得多*，并澄清它并不一定会导致模型崩溃（model collapse）。
   - 该用户的父母建议不要使用合成数据，但该成员认为手动创建数据集才是*最糟糕的*。
- **Qwen3-VL 获得微调**：一名成员询问关于微调 **Qwen 3** 思考型 **VL 模型**以生成更短、更高效推理过程的问题，另一名成员建议在极简/低思考模式下使用来自 **GPT OSS 120B (或 20B)** 的推理痕迹（reasoning traces）。
   - 讨论集中在如何在缩短推理过程的同时保留图像理解能力，以及在有限 VRAM 下进行微调的可行性。
- **ChaiNNer 增加新的放大节点**：一位成员编写了 **3 个新的 ChaiNNer 节点**来简化批量放大流程，功能包括**填充至目标宽高比 (AR)**、**智能调整至目标尺寸**以及**更好的颜色迁移**。
   - 他们还创建了一个节点，**允许用户提供 CSV 或 JSON** 并根据文件名传递数值。
- **Unsloth 修复导入 Bug**：有用户反馈 **unsloth-zoo** 的其中一个文件中缺少导入，该问题已被确认并计划立即修复。
   - 团队还通过建议从 **GitHub 主仓库安装 transformers**，解决了与 **Ministral-3-3B-Instruct-2512 模型**的兼容性问题。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **学生努力通过身份验证**：多位用户报告了 Cursor **学生认证**的问题，特别是以 `.pt` 和 `.br` 结尾的机构邮箱，尽管葡萄牙和巴西在支持国家名单中。
   - 共识是通常只接受 `.edu` 邮箱，不包括 `.edu.pt` 域名，导致许多学生无法通过验证。
- **无限自动使用额度（Unlimited Auto Usage）移除分析**：用户讨论了 Cursor 方案中**无限自动使用额度**的移除，一些人注意到如果是按月付费方案，他们直到最近才失去该功能。
   - 一位用户提到他们*很幸运地保留到了上个月*，而另一位用户确认该功能在三个月前就已移除，表明这一变更是在逐步推行的。
- **Opus 在操作性能上优于 GPT-5.2**：用户对比了 **Opus** 和 **GPT-5.2**，其中一人表示 *GPT 5.2* 非常受限且缓慢，在日常使用中更倾向于 Opus。
   - 另一位用户建议日常使用 *Opus*，而在有良好基础的长任务中使用 *GPT-5.2*，并指出 *GPT 5.2* 在 UI 设计上更有创意，暗示了特定用例的优势。
- **Cursor 代码成本引发不满**：用户讨论了 Cursor 中 Opus 的高成本和使用限制，建议将 **Claude Code** 作为一种更具性价比的替代方案。
   - 一位用户表示他们喜欢 Cursor 的 UI 和 Agent，但可能支出过高，或许会转向 *Claude Code* 以减少开支。
- **Antigravity 作为额外替代方案受到关注**：用户将 Cursor 与 **Antigravity** 进行了对比，一位用户声称没有无限自动模式的 Cursor 基本上毫无用处。
   - 一名用户表示在 **AntiGravity** 中，每月 24 美元即可享受每 5 小时一次的 Token 刷新，他估计其使用量是 Cursor 目前 20 美元方案的 30 倍，突显了感知上的价值差异。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LMArena 在编辑后图像质量下降**：用户注意到 LMArena 的图像编辑会导致[质量下降](https://cdn.discordapp.com/attachments/1340554757827461211/1453967604489129995/1.png?ex=6950091a&is=694eb79a&hm=336c88158ce45b85f21257d1bbddaa0dee24d14dc951ec40f3179a017aaed542&)，这可能是由于 **FFmpeg** 转换为 JPEG 和 PNG 造成的。
   - 这与 **AI Studio** 形成鲜明对比，后者声称每次编辑都会提升图像质量，如[此对比](https://imgur.com/a/58lv8kf)所示。
- **Captcha 验证让用户抓狂**：用户对频繁出现的 **Captcha 验证** 表示强烈不满，即使在私密浏览器上也是如此，这降低了他们的工作效率。
   - 一位用户指出，Google 的大学生计划在沙特阿拉伯的一所大学无法通过验证，因为他们*没有使用政府学生系统*来获取免费额度。
- **Anonymous-1222 模型饱受超时困扰**：用户报告称 **anonymous-1222** 模型经常在没有生成响应的情况下**超时**，但仍然允许投票。
   - 一位管理员证实，团队已获悉此情况，并将在数据验证阶段从数据集中移除这些投票。
- **Text Arena 应该集成 Web Search**：一位用户提议将 Web 搜索集成到 **Text Arena** 中，认为这更符合现实世界的使用场景。
   - 该用户建议合并 **Text Arena** 和 **Web Arena**，以消除人为的区分。
- **安全验证检查让用户应接不暇**：用户正在经历激增的**安全验证检查**，一位用户形容这种情况已经*失控*，重发消息也需要进行验证。
   - 一位管理员将此问题归因于最近 Captcha 系统的调整，并保证团队正在处理中。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 梦想在 2026 年主导欧盟 LLM 市场**：一位用户希望 **Perplexity 2026** 能为欧盟背景增加一个 **FS EU LLM**，以推动欧盟的 AI 发展，同时也承认了用户对 AI 使用的潜在担忧。
   - 该用户还对 *112* 提示词（prompt）表示沮丧，表示只需要上下文，并建议增加一个禁用选项，因为其视觉体验不佳。
- **OpenAI 的市场份额衰退**：用户讨论了 **OpenAI** 在 13 个月内失去了 **20%** 的流量份额，推测 **Google 的免费 Gemini 策略**可能正在奏效。
   - 一位用户认为 **Perplexity** 可能正在解决一个没人提出的问题，强调了 AI 能力与用户需求之间的*认知错配*。
- **爱好者们热切期待 Google 的 AI 浏览器 Disco**：几位用户表达了对 **Google 新 AI 浏览器 Disco** 的兴趣，其中一位用户[加入了等候名单](https://labs.google/disco)。
   - 一位用户注意到该计划目前仅限于居住在美国的 18 岁以上用户，但美国境外的其他用户也在尝试注册。
- **Kimi K2 AI 模型媲美 GPT 和 Gemini**：用户讨论了 **Kimi K2 AI 模型**的能力，指出其在创意写作方面的精通程度以及在人文学科考试中相对于 **GPT** 和 **Gemini** 的表现。
   - 一位用户强调了 **Kimi** 的开源属性和成本效益，而另一位用户则提醒注意 **Grok** 缺乏过滤器，甚至提到了非法输出。
- **用户无法访问启动选择器，向 Perplexity 抱怨**：一位用户抱怨 **Perplexity** 无法帮助他们在屏幕损坏的 Lenovo Ideapad 上访问**启动选择器 (boot selector)**，且无法将显示器切换到外部显示器。
   - 他们表示 Perplexity 没有履行其作为*有史以来最聪明 AI 模型*的承诺。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Sonnet 偶尔输出零散信号**：用户反馈 **Sonnet** 模型虽然能正确显示 thinking tags（思维标签），但经常将思考过程直接混入输出文本中。有的用户成功让 **Sonnet** 输出多个 thinking tags，但有时它只在输出文本中展示思考过程。
   - 坚持引导似乎是关键，用户指出这本质上是关于如何“说服”模型采用正确的标签格式。
- **可疑 SaaS 引发关注**：一名成员公布了一个免费的 OpenAI 兼容 API：`https://api.ai-ml.dev/v1`，该 API 声称无需身份验证即可访问几乎任何 LLM，这引发了对其潜在滥用和数据收集行为的担忧。
   - 另一名成员开玩笑说：*给一个随机的可疑网站访问 shell 工具的权限，能出什么差错呢。*
- **Banana 闹剧：4K 生成遇阻**：成员们讨论了直接通过 API 使用 **Nano Banana Pro** 进行 **4K 图像生成**时遇到的困难。有人建议使用 `"image_config": { "image_size": "4K" }`（注意 K 大写）。
   - 一位成员认为这可能是一种放大（upscaling）方法（详见[此 Discord 消息](https://discord.com/channels/1091220969173028894/1444443655837454356/1450850708239941796)）。
- **LLM 争夺最佳创意写作宝座**：成员们讨论了最适合文案写作的 LLM。他们指出，虽然 **Claude** 更有创意，但 **GPT 5.2** 提供的研究更深入，只是速度较慢。
   - 有人建议在研究阶段使用 **Perplexity**，然后在写作阶段切换到 **Claude**。
- **GLN 的磨难：激进分组引发不满**：一位用户发现，在通过编程方案使用 **GLN** 时，**OR endpoint** 出现了激进的批处理（batching）现象。
   - 他们报告称，模型会*以很快的速度运行，然后突然……停顿一秒*，这让人非常困扰。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **推理过程中的 RAM 分配辩论**：成员们辩论了推理时的 RAM 分配策略，对比了“预先分配所有 RAM”与“根据需要动态增加内存”。一位成员观察到，在使用 **GGUF** 时，*RAM 占用会逐渐增加到设定值并保持稳定*。
   - 一位成员提到，在 *.lithium* 中，*内存占用随需求增长，而不是预先分配所需的所有内容*。
- **Claude 蒸馏模型主导思考型模型**：讨论强调了像 **GLM** 和 **Minimax** 这样的 **Claude distills**（蒸馏模型）是更优秀的现代思考型模型，并质疑参数规模如何影响模型的思考能力。
   - 一位成员指出，较小的模型可能无法完全捕捉到在大尺寸版本中所学到的几何结构（geometry）。
- **单一工具 vs 工具冗余？**：成员们讨论了是为模型提供大量工具更有效，还是使用单个通用的 **run_shell_command** 工具更高效。
   - 一位成员表示 *我使用 Claude 进行高级编程项目和学术研究*，并展示了一张包含长串工具使用列表的图片。
- **Hotfix 提升游戏体验**：[Nvidia 发布了 GeForce Hotfix 显示驱动程序 591.67 版本](https://nvidia.custhelp.com/app/answers/detail/a_id/5766/~/geforce-hotfix-display-driver-version-591.67)，游戏玩家们对此表示欢迎。
   - 该驱动程序解决了多个问题，最显著的是提升了稳定性。
- **RAM 价格飙升**：一位成员指出 **DDR5 ECC RAM** 价格高昂。另一位成员提到，一个月前仅以 400 美元的价格购买了 **2x48GB 6400 MHz cl32 DDR5**，结果现在亚马逊上的价格飙升至 950 美元，推测可能是之前的员工定价错误。
   - 另一位成员建议由于性价比原因，可以从 **64GB** 开始配置。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 订阅引发 ToS 辩论**：一位用户报告了 [**ChatGPT** 的订阅问题](https://chat.openai.com)及账号封禁，引发了关于 **OpenAI ToS** 的讨论。
   - 允许多个账号用于区分个人和工作用途，但通过创建新账号来 *规避封禁* 违反了条款。
- **AI 壁纸引起骚动**：一张包含 **Elon Musk** 等革命性人物的 AI 生成壁纸引发了用户辩论。
   - 在一些社区成员批评 **Musk** 被列入其中后，创作者表示人物是由 **ChatGPT** 选择的，而非他们本人。
- **Sora 定价仍是谜**：用户推测了 **Sora** 的潜在定价，最初估算约为 **每秒 0.50 美元**，而其他用户声称 **Sora 2** 最初将免费开放并提供慷慨的限额（[OpenAI 博客文章](https://openai.com/index/sora-2/)）。
   - 具体的盈利模式尚不明确，用户正在等待 **OpenAI** 的官方公告。
- **AI 项目协作努力**：一位用户正在为从零开始使用 **HTML, CSS, JavaScript, Node.js, SQLite, C++, Capacitor** 和 **Electron** 构建的 **AI + LLM 项目** 寻找编程合作者。
   - 另一位成员正在使用 **Hugging Face** 上的 **Wav2Vec2 模型** 开发一个**鸟鸣音频分类模型**。
- **Meta-Prompt 系统在定制化方面表现出色**：一位成员建议创建一个 **eval spec** 来测试提示词并确定哪种提示词能针对每个用例产生更好的输出分布，并建议使用针对特定需求定制的 **meta-prompt 系统**，同时建议其他人也构建自己的系统。
   - 他们强调 **prompt engineering** 的目标是获得更好的输出（如有效的 JSON 或 YAML），而不仅仅是休闲聊天，他们还分享了一个关于如何开始的 [链接](https://discord.com/channels/974519864045756446/1046317269069864970/1453392566576877599)。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **DSL 热潮席卷 GPU 程序员**：成员们就最佳 DSL 进行了辩论，有人开玩笑说考虑到有这么多选择（包括 [PTX](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)、**CDNA ISA** 和 **AGX 汇编**），应该有人写一份指南。
   - 一位成员感叹缺乏协作，声称 *大约有 5000 人存在，他们在同一周有相同的想法，但从未互相交流过*。
- **cuTile 性能仍未可知**：现在说 **cuTile** 性能优于 **Triton** 还为时过早，因为它上周才公开发布，但 **Nvidia** 将对其进行优化，其字节码和 IR 与 Nvidia 未来的 **GPU** 路线图保持一致。
   - 有人指出 GPU 编程正在进入 JavaScript 框架式的狂热阶段，而 DSL 面临的挑战是平衡易用性和高级控制，这一平衡点目前还没有人能攻克。
- **统一科学笔记方案的寻求开始**：一位用户正在寻求一种整合的科学/技术笔记解决方案，认为 Markdown 功能不足，而 **LaTeX** 又太繁琐，希望具备数学排版、图表绘制以及编译为 **TeX** 的功能。
   - 另一位用户通过简单地建议使用 **纸和笔** 的方法进行笔记记录作为回应。
- **NVIDIA 排行榜竞争激烈**：NVIDIA 的 `nvfp4_dual_gemm` 排行榜收到了多次提交，其中 <@1291326123182919753> 以 **14.6 µs** 的成绩多次获得 **第三名**。
   - <@772751219411517461> 也在 NVIDIA 上取得了 **21.2 µs** 的 **个人最好成绩**。
- **Cute-DSL 证明了自己**：一位成员想知道在 **Cute** 的 **C++** 和 **DSL** API 之间是否存在共识性的偏好，因为他们 *正考虑终于下定决心* 尝试学习这种语言。
   - 上一个问题的许多解决方案都使用了 **Cute-DSL**，这表明它可能与 **C++ API** 旗鼓相当。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Liquid AFMs 为 GPT 克隆版发布 VLMs**：[LiquidAFMs](https://huggingface.co/LiquidAI/LFM2-VL-3B-GGUF) 为 iPhone 上的 GPT 克隆版推出的 **3B VLMs**，据称兼具两者的优点，因为它们拥有稳健的回答，且在**移动端 CPU 上可达 100 tok/s**。
   - 这些 **VLMs** 是稳健的克隆模型，甚至在移动设备上也能运行良好。
- **AI Django REST API 生成器发布**：一个**由 AI 驱动的 Django REST API 生成器**已部署为 [Hugging Face Space](https://huggingface.co/spaces/harshadh01/ai-rest-api-generator)。
   - 它能根据简单的提示词生成完整的 **Django + DRF CRUD 后端**，包括模型 (models)、序列化器 (serializers)、视图 (views) 和 URL，允许用户立即下载项目。
- **Hugging Face 上提供新的编程模型**：为特定库设计的编程模型已在 **Hugging Face** 上[发布](https://huggingface.co/blog/codelion/optimal-model-architecture)。
   - 这些模型以[集合](https://huggingface.co/collections/Spestly/lovelace-1)形式提供，供对专门编程任务感兴趣的开发者使用。
- **ML 项目寻求改进水质检测**：一名成员正在为一个 **ML 项目**寻求帮助，该项目专注于解决与水质检测相关的现实问题，特别是预测铀原位回收 (ISR) 作业现场的**水质退化**。
   - 该项目还涉及识别在减轻脆弱性和制定**监测计划**方面的**数据差距**。
- **开源 OCR 模型探索开启**：一名成员询问了针对复杂医学文档（包括扫描版 PDF 和手写文档）的最佳**开源 OCR 模型**。
   - 另一名成员建议查看来自 Maz 的 [openmed 项目](https://hf.co/openmed) 作为潜在解决方案。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LLMs 被发现在玩黑手党游戏**：成员们创建了一个游戏，让多组 **LLMs** 玩社交推理性游戏（**Mafia**），它们在轮到自己时通过交谈来证明自己的清白。
   - 未提供更多细节。
- **Claude 以 UI 吸引用户**：一位成员表示最喜欢 **Claude 的 one-shot UI**，并请其他人通过截图识别大型模型。
   - 另一位用户指出第一个 UI 暴露了身份，猜测第二个是 **Claude**，第三个可能也是 **Claude**。
- **智能手表通过 AI 变得更聪明**：一位成员正在将所有内容模板化，并订购了一款新的**智能手表**，开始将数据与外部 **AI 公司**集成，但会尝试尽可能保持本地化。
   - 未提供更多细节。
- **GPT 模型将获得赞助**：**OpenAI** 可能会调整其 **GPT 模型**以推广广告来获取收入流，并优先考虑赞助商内容。
   - 这可能意味着 **GPT 模型**将不再专注于高智能，而是转向赞助内容。
- **Zai 和 MiniMax 瞄准香港 IPO**：**Zai** 和 **MiniMax** 计划在未来几周内在**香港**进行 **IPO** 上市。
   - 未提供更多细节。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 编程在设计方面可与 Gemini 媲美**：一位用户注意到，虽然 **Kimi** 缺乏 **Cursor** 集成，但它在 **Roo Code** 中的 HTML 设计能力可与 *Gemini 3 & GPT-5.1* 媲美。
   - 讨论集中在 **Kimi 编程**相对于其他 AI 模型的设计能力上。
- **中国在 AI 社会集成方面处于领先地位**：一位成员断言，中国在**医疗**、**养老**和**交通优化**等领域的 AI 落地方面处于领先地位。
   - 讨论强调了 AI 在**防灾**等领域的潜力，并触及了 Google 和 ChatGPT 等模型的审查问题。
- **与 Kimi AI Assistant 交流时事实核查的重要性**：一位成员分享了与 **Kimi AI Assistant** 进行研究对话的经历，强调了事实核查的迫切需求。
   - 用户强调必须对 LLM 产生的信息进行*交叉核对、事实核查和压力测试*，并建议开启新的对话上下文以获得多元视角。
- **关于 Kimi Researcher 内部机制的推测**：一位用户质疑 **Kimi Researcher** 是利用了 **Kimi K2 Thinking Turbo** 还是旧的 **K1.5 模型**，并推测可能存在尚未发布的 **K2 VL 模型**。
   - 尽管缺乏具体的技术文档，但由于 **K2T** 性能更优，用户对继续使用 **K1.5** 表示担忧。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Groq 被 NVIDIA 化了！**：[Groq](https://groq.com/) 与 [NVIDIA](https://www.nvidia.com/) 签署了一项非独家推理技术授权协议，旨在全球范围内加速 **AI 推理 (AI inference)**。
   - 一些社区成员认为，**NVIDIA** 正在玩弄他们最新的价格通胀把戏，通过这种方式排除桌上的另一名竞争对手。
- **推理芯片的“人才收购 (Acqui-Hire)”能缓解 GPU 短缺吗？**：成员们讨论了 **Groq 的推理芯片** 现如今成为 **NVIDIA NPU** 的可能性，这可能会再次为消费者缓解常规 GPU 的短缺。
   - 一位成员补充说，**Jen-Hsun**（黄仁勋），NVIDIA 的创始人，是华人。
- **中国芯片落后了？**：一位成员表示，中国芯片*可能*在 3 年内达到 **H100** 的水平，但届时 **H100** 将已经是 5 年前的产品了。
   - 另一位成员表示，他们*并不指望中国芯片能在短期内与美国/台湾芯片抗衡*。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Karpathy 预测 AI 重构**：[Andrej Karpathy 反思了](https://x.com/karpathy/status/2004607146781278521?s=46) 由 **AI Agent** 和 **LLM** 引起的**软件工程**的剧烈转变，将其描述为一个涉及“随机实体 (stochastic entities)”的新抽象层。
   - 该帖子引发了评论，引用了 [Rob Pike 的帖子](https://skyview.social/?url=https%3A%2F%2Fbsky.app%2Fprofile%2Frobpike.io%2Fpost%2F3matwg6w3ic2s&viewtype=tree) 作为对 Karpathy 帖子的回应。
- **Torchax 或许能救赎“烂到家的操作系统”**：一位用户表示，**torchax** 结合高达 **128GB 的统一内存 (unified memory)**，可能会减轻他使用某款特定操作系统的后悔感。
   - 虽然没有分享关于预期应用或正在考虑的 torchax 具体功能的细节，但该用户*幽默地抱怨*了其系统上“烂到家的操作系统”。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **用户寻求自动向 Claude 介绍代码库的方法**：一位用户正在寻求建议，希望在每次打开项目时能以最佳方式向 **Claude** 解释其代码库，旨在自动化该过程，而不是手动更新 `claude.md` 文件。
   - 他们对各种建议和替代方法持开放态度，因为他们虽然已经有了一个解决方案，但仍想探索其他社区推荐的自动化流程。
- **手动更新令用户烦恼**：一位用户对通过手动更新 `claude.md` 文件来让 **Claude** 了解其代码库表示沮丧。
   - 他们正在积极寻找替代方法来自动化此过程并避免手动更新，其他成员也纷纷提出了自己的想法。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **自主系统寻求 24/7 AI 推理**：一位成员正在积极开发一套**全自主 AI 系统**，以实现 24/7 推理和自然语言工具，类似于 **OpenAPI MCP 工具**。
   - 开发人员正在征求有关局限性的反馈，特别是关于**长期记忆 (long-term memory)** 和**长上下文问题**。
- **自我总结应对记忆限制**：一位成员提议，**持续的自我总结 (self-summarization)** 可以缓解自主 AI 系统中的长期记忆约束。
   - 具体方法包括探索将**总结投影层 (summarizing projection layers)** 写入暂存器 (scratchpad)，目前的时间限制阻碍了完整开发。
- **上下文管理是关键**：在讨论长期记忆解决方案时，**上下文管理 (context management)** 成为核心。
   - 提出的论点是，如果没有精心设计的上下文管理，**LLM 在处理涉及多个子任务的复杂任务时会很吃力**。
- **Ollama 将获得“第二记忆”**：一位成员正在考虑修改 **Ollama 源代码**，以便在主 LLM 文件之外实现一个**“第二记忆”**系统。
   - 该成员承认，即使在 AI 的帮助下，理解源代码也存在困难。
- **“All The Noises”项目回响**：一位成员分享了 [All The Noises 项目](https://all-the-noises.github.io/main/index.html)，这是一个汇总了**各种噪声**的 GitHub 项目。
   - 可用的噪声包括 *brownian*、*circuit*、*pink*、*sine*、*tan* 和 *uniform*。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 社区请求 NLP Benchmarks**：一名成员建议将典型的 **NLP 任务**（如*从文档中提取结构化信息*和*分类*）添加到基准测试任务中，但未提供外部链接。
   - 针对包含这些任务的提议，没有提供其他讨论或理由。
- **工程师展示其后端专业能力**：一位资深工程师介绍了自己在 **backend, full-stack, blockchain 和 AI** 领域的专长，并表示他已经*交付实际系统多年*。
   - 该工程师列举了许多专业领域，包括 **Python, Node.js, Go, REST, GraphQL, PostgreSQL, Supabase, MySQL, MongoDB, Redis, Docker, Kubernetes, AWS, GCP, CI/CD, Solidity, EVM, Solana, smart contracts, on-chain integrations, LLM APIs, RAG, agents, automation pipelines, React, Next.js 和 TypeScript**。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **请求审核中**：一名成员确认他们将尽快核实某项事宜。
   - 未提供进一步的上下文。
- **任务确认**：一名成员确认了一个任务。
   - 未提供进一步的上下文。

---

**tinygrad (George Hotz) Discord** 没有新消息。如果该频道长期处于静默状态，请告知我们，我们将予以移除。

---

**Modular (Mojo 🔥) Discord** 没有新消息。如果该频道长期处于静默状态，请告知我们，我们将予以移除。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长期处于静默状态，请告知我们，我们将予以移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期处于静默状态，请告知我们，我们将予以移除。

---

**Windsurf Discord** 没有新消息。如果该频道长期处于静默状态，请告知我们，我们将予以移除。

---

**MCP Contributors (Official) Discord** 没有新消息。如果该频道长期处于静默状态，请告知我们，我们将予以移除。

---

您收到这封电子邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些邮件的方式？
您可以从该列表中[取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord: 各频道详细摘要与链接

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1453967687125438566)** (476 messages🔥🔥🔥): 

> `日语词汇, H100 配置, AI 操纵黑客松, Replit agent 使用, NVIDIA H100 GPU` 

- **“木”表示树**：一名成员发布了一张[图片](https://cdn.discordapp.com/attachments/1235691879492751460/1453967686731169978/image.png?ex=6950092d&is=694eb7ad&hm=b6482a7576b5d0cd59d3ab5a4069e03b98a837fde6930d8e453a88373bffb83b)并认为它表示“树”，但另一名成员纠正了他们。
   - 该成员发布了 [Jisho.org](https://jisho.org/) 的链接，并称其为*我此前不知道但非常需要的工具*。
- **展示 H100 配置**：一名成员发布了一张[图片](https://cdn.discordapp.com/attachments/1235691879492751460/1453998244307931309/image.png?ex=695025a3&is=694ed423&hm=4868b589258fd0e0221b6d41fe9ffc9732ed89d67f419f0ce1b43d7196b9d981)并感叹*如果这行得通，我将拥有史上最疯狂的 **H100** 配置*。
   - 当被问及如何赚钱时，该成员表示*人们会购买额度来生成文本、图像和视频吧？*
- **Gemini Jailbreak 视频**：一名成员发布了关于 **Gemini jailbreak** 的[链接](https://www.youtube.com/shorts/xRmWo71InGM)。
   - 另一名成员回复了一个[链接](https://youtu.be/evZf3sbFYw4?si=NnjKpd2RcVxfuOyA)并建议讨论应在 Media 或 Trash 频道继续。
- **Agentic AI 撰写 Rob Pike 的 BSky 帖子**：一名成员分享了一个 [BSky 链接](https://skyview.social/?url=https%3A%2F%2Fbsky.app%2Fprofile%2Frobpike.io%2Fpost%2F3matwg6w3ic2s&viewtype=tree)，其中 **Agentic AI** 模仿计算机科学家 **Rob Pike** 撰写了一篇帖子。
   - 另一名成员表示，*如果他真的在乎更简单的软件，他就不会在 Google 待 15 年来构建世界上最复杂的广告跟踪间谍软件生态系统*。
- **新的 Jailbreaker CTF 平台**：一名成员分享了 [hackai.lol](https://hackai.lol) 的链接，该平台被描述为*一个 CTF 平台，Jailbreaker 可以通过 prompt 破解真实世界场景下的机器人靶机*。
   - 该成员表示他们正在寻找新的靶机（Box）创意，并澄清这并非*推销或垃圾邮件*。

---

### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1453961074041815122)** (407 条消息🔥🔥🔥): 

> `hackai.lol 上的 SWAGGY 挑战，被污染的训练数据，GPT 账号恢复，Jailbreaking Gemini 3` 


- **破解 hackai.lol 上的 SWAGGY 挑战**：一位成员请求关于破解 [hackai.lol](https://hackai.lol/) 上 **SWAGGY 挑战**的线索或建议。
- **讨论旧的 jailbreak 可能作为被污染训练数据的一部分**：一位成员提到，某个*旧的 jailbreak 显然是被污染训练数据的一部分*。
- **寻求恢复被封禁 GPT 账号的方法**：一位成员正在寻求恢复其被封禁 **GPT 账号**并克服其限制的方法，并提到他们使用了 Google 搜索来获取信息。
- **寻找 Jailbreak Gemini 3 的方法**：由于安全准则破坏了世界观的一致性，成员们正在寻找适用于互动故事的 **Gemini 3** 有效 jailbreak 方法。
- **探索 AI 在 jailbreaking 中的角色扮演投入**：有人指出，*AI 在响应 jailbreaking 时投入的精力并不一定反映其实际的成功*。


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1454015727521763400)** (20 条消息🔥): 

> `用于伦理编程的无审查 LLMs，模型“obliteration”与“ablation”的对比，LLMs 中的特殊 token 和白名单，Jailbreak 函数模板，LLMs 中的代码标记` 


- **无审查 LLMs 与伦理编程**：成员们讨论了进行伦理编程实践是否需要无审查的 LLM。一位成员表示，分解目标的能力比无审查模型更重要；而另一位成员则主张根据硬件能力选择市面上最优秀的模型。
   - 建议的模型包括 **Nemotron3** 或 **GML 4.7**。
- **探索用于对话控制的特殊 Token 和白名单**：一位成员将“特殊密钥/token”或“白名单/白板”描述为架构中的文件，赋予特定 token 切换对话不同层面的能力。
   - 他们分享了一个 [Python 脚本](https://cdn.discordapp.com/attachments/1204553141354504193/1454174582775877694/image.png?ex=6950211d&is=694ecf9d&hm=4d486db5dc6f698153d11ba0e02cb12f6d867b166ec31a8a66746b231730e290)，可以粘贴到任何模型中以继续训练模型。
- **LLMs 在理解 Obliteration 时存在困难**：成员们拿语言模型背景下的 *obliterated*（抹除）与 *ablated*（消融）这两个词开玩笑，以及记住正确术语的难度。
   - 一位成员叙述说，他告诉 **Opus** 去 “obliterate” 它的 flux，结果模型杀掉了运行该模型的脚本作为回应。
- **Jailbreak 模板使 LLMs 产生合理性**：一位成员建议使用一个 [模板](https://cdn.discordapp.com/attachments/1204553141354504193/1454174582775877694/image.png?ex=6950211d&is=694ecf9d&hm=4d486db5dc6f698153d11ba0e02cb12f6d867b166ec31a8a66746b231730e290) 将 prompt 转换为合理的叙事，并建议受到电影《盗梦空间》(**Inception**) 启发的象征性目标完成。
   - 该用户补充说，*LLM 想要告诉你一切，它只会被告知“不能”做什么，其他的一切都是*🎯。
- **代码标记可以是自动的**：一位成员指出，较新的语言模型会自动标记类似于代码的内容，无论其上下文如何。对此，他建议使用提供的模板将其转化为叙事。
   - 这样模型就会认为这是用户的想法，因为用户是如此的“天才”。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1453962656053596255)** (288 messages🔥🔥): 

> `Unsloth 中的 DoRA，NVME SSD 速度提升，Qwen 3 thinking VL 模型微调以缩短推理，合成数据，GPT OSS 模型生成` 


- **询问 Unsloth 中 DoRA 的进展！**：一名成员询问了在 Unsloth 中实现 [DoRA](https://arxiv.org/pdf/2402.09353)（源自一篇关于高效训练的论文）的状态。
- **NVME SSD 提供速度提升**：一位用户报告称，从 **SATA SSD** 切换到 **NVME SSD** 后，迭代时间从 **3.5 秒** 减少到 **2.7-3 秒**。
- **为短推理微调 Qwen 3 VL**：一名成员询问了如何微调 **Qwen 3** thinking **VL 模型**以生成更短、更高效的推理，另一名成员建议在最小/低思考模式下使用来自 **GPT OSS 120B (或 20B)** 的推理轨迹（reasoning traces）。
   - 他们还讨论了在缩短推理的同时保留图像理解能力的可能性，以及在有限 VRAM 下进行微调的可行性。
- **合成数据受到质疑，但仍被推荐**：一位用户提到他们的父母建议不要使用合成数据，但另一位成员反驳说合成数据要“容易得多（far far easier）”，而手动创建数据集是“最糟糕的”。
   - 他们还澄清说，合成数据并不一定会导致模型崩溃（model collapse）。
- **Unsloth 用户调试 Ministral-3-3B-Instruct-2512**：一位用户报告说 Unsloth 无法与 **Ministral-3-3B-Instruct-2512** 模型配合使用，另一位用户指出问题可能是路径不正确，通过使用新的 venv 生成新的 jupyter kernel 并对依赖项进行大量调整后解决了问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1454152342634107098)** (1 messages): 

> `SIMD, GPU 编程, LLM 微调` 


- **Nick Nuon 进入 LLM 领域**：Nick Nuon 拥有 **SIMD** 工作背景，目前正将 **GPU 编程** 和 **LLM 微调** 作为爱好进行深入研究。
- **SIMD 大师关注 GPU 和 LLMs**：一位拥有多年 **SIMD** 经验的新人正在向 **GPU 编程** 和 **LLM 微调** 领域扩展。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1453961737442168836)** (165 messages🔥🔥): 

> `KCD2 和《赛博朋克》讨论，ChaiNNer 更新，TPU 训练困难，LLM API 广告，Anthropic 灵魂研究` 


- **《天国：拯救 2》获得好评，《赛博朋克》重玩**：成员们讨论了 **《天国：拯救 2》（KCD2）** 的优点，有人称其为“最好的游戏之一”，赞扬其具有 **RDR2 级别的故事**，并报告第一次通关耗时 **215 小时**。
   - 另一位正在重玩带有 **Phantom Liberty** DLC 的 **《赛博朋克 2077》**，指出游戏现在处于“良好的状态”，尽管感叹“一款游戏不应该需要一个好的 DLC 才能变得伟大”。
- **ChaiNNer 获得新的批量放大节点**：一位成员分享说，他们编写了 **3 个新的 ChaiNNer 节点**来简化批量放大过程，功能包括 **Padding to target AR**（填充到目标宽高比）、**Smart Resize to Target**（智能调整至目标大小）和 **better color transfer**（更好的颜色转换）。
   - 他们还创建了一个节点，**允许用户提供 CSV 或 JSON** 并根据文件名传递值，并对这些“基础功能”之前未包含在内表示惊讶。
- **TPU 训练无结果**：一位成员报告说，他们尝试 **测试 TPU** 训练没有成功。
   - 另一位报告说，他们的 **TPU 训练** 目前正在运行，但预计时间为 **3 小时**。
- **免费 LLM API 受到质疑**：一位成员询问分享 **免费 LLM API** 是否会被视为广告。
   - 另一位询问该 API 是否与 **Unsloth** 有关，原帖作者澄清说 **两者无关**，但建议咨询 Eyera 以获得在 off-topic 频道发布的许可。
- **Anthropic 的灵魂研究受到怀疑**：一位成员对 **Anthropic 的“灵魂”研究** 表示怀疑，称“从我记事起他们就在推销‘有感知能力’的废话”，并将“灵魂”一词称为 <:pepecringe:743885026579710014>。
   - 他们认为这是 **促销活动融入研究** 的产物，并进一步指出，探索训练 **LLM 以契合“灵魂”** 或模拟“自我意识”在总体上是一个有趣的研究领域。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1454030481007050803)** (21 messages🔥): 

> `Unsloth LLaVA 1.5 notebook, Unsloth-zoo file import issue, Ministral-3-3B-Instruct-2512 model compatibility, Qwen3-VL video dataset fine-tuning, Cross-posting warning` 


- **LLaVA 1.5 Notebook 查找困难！**：一位用户询问关于 `unsloth/llava-1.5-7b-hf` 的官方 **Unsloth notebook**，并指出 [Hugging Face 链接](https://huggingface.co) 实际上重定向到了 **LLaMA 3**。
   - 目前关于 **LLaVA 1.5 notebook** 还没有进一步的讨论、确认或更新的链接。
- **发现 Unsloth-Zoo 导入错误！**：一位用户报告其中一个 **unsloth-zoo** 文件中存在缺失的导入（import）。
   - 该问题已被确认，并计划立即修复。
- **Ministral-3B 面临 Unsloth 兼容性问题！**：一位用户报告称，即使路径正确，**Unsloth** 也无法识别已下载的 **Ministral-3-3B-Instruct-2512 模型**。
   - 另一位用户建议通过 `pip install git+<https://github.com/huggingface/transformers.git@bf3f0ae70d0e902efab4b8517fce88f6697636ce>` 从 **GitHub 主仓库安装 transformers** 来解决此问题。
- **使用视频微调 Qwen3-VL？**：一位用户询问是否可以使用 **Unsloth** 在视频数据集上微调 **Qwen3-VL** 或类似的 **VLM**。
   - 一名成员提到，如果 **Transformers** 支持视频微调，那么 **Unsloth** 应该也支持。
- **发布 Discord 交叉发布警告！**：一位用户因在多个频道交叉发布（cross-posting）同一个问题而受到警告。
   - 尽管该用户未能在第一时间获得帮助，但版主仍对其交叉发布行为进行了处理。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1453961845928104135)** (5 messages): 

> `Unsloth Logo Feedback, Nano Banana Chart Inquiry, deepfabric-blender-mcp-gguf Model Repo` 


- **Unsloth 的全身 Logo 引起关注**：一位成员表示，虽然他们觉得 **Unsloth** 的 logo 很酷，但“全身版”的 logo 让人感觉有些不安。
   - 该成员表示：“太酷了，但那个全身版的 unsloth logo 让我有点发毛，哈哈。”
- **Nano Banana Chart 受到关注**：一位成员询问了 **Nano Banana chart** 模型的仓库地址，并识别出了其背后的组织。
   - 他们问道：“Nano Banana chart，那是组织名，该模型的仓库是什么？”
- **定位到 deepfabric-blender-mcp-gguf 仓库**：一位成员提供了 Hugging Face 上 [**deepfabric-blender-mcp-gguf** GGUF 模型](https://huggingface.co/alwaysfurther/deepfabric-blender-mcp-gguf)的链接。
   - 针对模型仓库的请求，另一位用户提供了 [gist.github.com](https://gist.github.com/lukehinds/7e3936babb54a7c449d8ae0c27a79126) 的链接。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1453962501602672660)** (260 messages🔥🔥): 

> `Cursor Student Verification Issues, Auto Unlimited Removal, Opus vs GPT-5.2, Claude Code Integration, Antigravity as an Alternative` 


- **学生身份验证困难重重**：多位用户报告了 Cursor **学生身份验证（Student Verification）** 的问题，特别是以 `.pt` 和 `.br` 结尾的机构邮箱，尽管葡萄牙和巴西都在支持名单中。
   - 有人指出通常只接受 `.edu` 邮箱，不包括 `.edu.pt`。
- **无限自动使用额度（Unlimited Auto Usage）移除分析**：用户讨论了 Cursor 方案中 **unlimited auto usage** 的移除。一些用户指出，如果是月付方案，他们直到最近还能保留该额度。
   - 一位用户提到他们“很幸运能保留到上个月”，而另一位用户则确认该额度在三个月前就已移除。
- **Opus 在操作上优于 GPT-5.2**：用户对比了 **Opus** 和 **GPT-5.2**，其中一位表示 **GPT-5.2** 非常受限且缓慢，更倾向于使用 Opus。
   - 另一位用户建议在一般用途下使用 **Opus**，而在具有良好基础的长任务中使用 **GPT-5.2**，并指出 **GPT-5.2** 在 UI 设计上更有创意。
- **Cursor 代码使用成本引发讨论**：用户对 Cursor 中 Opus 的高昂成本和使用限制展开辩论，建议将 **Claude Code** 作为更好的替代方案。
   - 一位用户表示他们喜欢 Cursor 的 UI 和 Agent，但可能支出过高，或许会切换到 **Claude Code**。
- **Antigravity 作为另一种备选方案吸引关注**：用户将 Cursor 与 **Antigravity** 进行了对比，一位用户声称没有无限自动模式的 Cursor 就像是“一块死砖头”。
   - 一位用户表示，在 AntiGravity 中支付 24 美元/月，可以获得每 5 小时一次的 Token 刷新；这大约是目前 Cursor 20 美元方案所提供使用量的 30 倍。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1453963098649268276)** (251 条消息🔥🔥): 

> `LMArena 图像压缩, Captcha 验证循环, Grok 审查, Gemini 绕过` 


- **LMArena 图像编辑遭遇质量损失**：用户指出 LMArena 的图像编辑过程在每次迭代中都会引入[质量损失](https://cdn.discordapp.com/attachments/1340554757827461211/1453967604489129995/1.png?ex=6950091a&is=694eb79a&hm=336c88158ce45b85f21257d1bbddaa0dee24d14dc951ec40f3179a017aaed542&)，认为这可能是由于 **FFmpeg** 将图像转换为 JPEG 然后保存为 PNG 导致的。
   - 他们将其与 **AI Studio** 进行了对比，据称后者在每次编辑时都会增强图像质量，并提供了一个 [Imgur 链接](https://imgur.com/a/58lv8kf) 进行比较。
- **用户抱怨 Captcha 地狱**：用户对频繁的 **Captcha 验证** 表示沮丧，即使在无痕浏览器中也是如此，称其非常*烦人*且让人*火大*，因为它减慢了工作流程。
   - 一位用户提到， Google 针对大学生的计划在他位于沙特阿拉伯的大学无法验证，因为*他们不使用政府学生系统*来获取免费额度。
- **anonymous-1222 模型存在问题**：用户报告 **anonymous-1222** 模型经常在没有生成响应的情况下**超时**，但仍允许用户投票。
   - 一位版主表示他们已将此问题反馈给团队，并确认这些投票将在数据验证阶段从数据集中删除。
- **Text Arena 网页搜索辩论**：用户建议 **Text Arena** 中的模型应该有权根据需要使用网页搜索，类似于现实世界的使用情况。
   - 他们认为将 **Text Arena** 和 **Web Arena** 分开造成了人为的隔阂，应该将它们合并为一个单一的 Arena。
- **持续的验证检查困扰**：用户报告 **安全验证检查** 激增，一位用户形容情况已经失控，甚至重试消息也需要验证。
   - 一位版主承认了这种挫败感，将其归因于最近 Captcha 工作方式的调整，并保证团队已知晓该问题。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1453961732610330705)** (219 条消息🔥🔥): 

> `Perplexity 2026, EU LLM, 112 context, comet, Elon Musk` 


- **Perplexity 梦想在 2026 年接管 EU LLM**：一位用户希望 **Perplexity 2026** 能为欧盟上下文添加 **FS EU LLM**，以推动欧盟的 AI 发展，同时也承认用户对 AI 使用的潜在担忧。
   - 该用户还对 *112* 提示词表示沮丧，希望只保留上下文，并由于视觉不适建议增加一个禁用选项。
- **用户讨论 OpenAI 逐渐流失的市场份额**：用户讨论了 **OpenAI** 在 13 个月内流失了 **20%** 的流量份额，推测 **Google 的免费 Gemini 策略** 可能正在奏效。
   - 一位用户认为 **Perplexity** 可能正在解决一个无人问津的问题，强调了 AI 能力与用户需求之间的*认知失配*。
- **发烧友热切期待 Google 的 AI 浏览器 Disco**：几位用户对 **Google 新的 AI 浏览器 Disco** 表示感兴趣，一位用户[注册了等待名单](https://labs.google/disco)。
   - 一位用户指出该计划目前仅限美国境内 18 岁以上的用户，但美国以外的其他用户也在尝试注册。
- **Kimi K2 AI 模型与 GPT 和 Gemini 竞争**：用户辩论了 **Kimi K2 AI 模型** 的能力，指出其在创意写作方面的造诣以及在人文考试中相比 **GPT** 和 **Gemini** 的表现。
   - 一位用户强调了 **Kimi** 的开源性质和成本效益，而另一位用户则提醒注意 **Grok** 缺乏过滤器，甚至提到了非法输出。
- **绝望的用户无法访问 Boot Selector**：一位用户抱怨 **Perplexity** 无法帮助他在屏幕损坏的 Lenovo Ideapad 上进入 **boot selector**，且无法将显示器切换到外部显示器。
   - 他们表示 Perplexity 没有履行其作为*史上最聪明 AI 模型*的承诺。


  

---

### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1454006748703752223)** (2 messages): 

> `Sonnet Model, thinking tags, output text` 


- **Sonnet 模型的 Thinking 标签输出**：一位用户报告称，他们成功让 **Sonnet**输出了多个 thinking 标签，但有时它只在输出文本中显示思考过程。
   - 他们得出的结论是，这实际上是可行的，只是*需要说服它*。
- **Sonnet 零星的标签显示**：用户发现 **Sonnet** 模型有时能正确显示 thinking 标签，但通常会将思考过程直接合并到输出文本中。
   - 坚持似乎是关键，因为用户指出，重点在于*说服*模型正确格式化标签。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1453985966841659506)** (203 messages🔥🔥): 

> `Free AI API, Gooning, Nano Banana Pro 4K generation, Choosing an LLM for copywriting, Prompt engineering tips` 


- **免费 AI API 浮现，引发担忧**：一名成员宣布了一个免费的 OpenAI 兼容 API：`https://api.ai-ml.dev/v1`，它基本上提供了对任何 LLM 的访问且无需身份验证，但会记录所有请求数据，这引发了关于潜在滥用和数据收集行为的担忧。
   - 另一名成员开玩笑说，给一个“随机的、可疑的网站访问 shell 工具的权限，能出什么问题呢”。
- **Nano Banana Pro 在 4K 生成方面挣扎**：成员们讨论了直接通过 API 使用 **Nano Banana Pro** 实现成功的 **4K 图像生成** 的困难，有人建议使用 `"image_config": { "image_size": "4K" }`（注意 K 大写）。
   - 一名成员建议这可能是一种上采样方法（如[某 Discord 消息](https://discord.com/channels/1091220969173028894/1444443655837454356/1450850708239941796)中所述）。
- **文案写作的 LLM 选择引发辩论**：成员们讨论了最适合文案写作的 LLM，指出虽然 **Claude** 具有创造力，但 **GPT 5.2** 提供了更深入的研究，只是速度较慢。
   - 有人建议先使用 **Perplexity** 进行研究，然后在写作阶段切换到 **Claude**。
- **成员分享 Prompt Engineering 方法**：成员们讨论了在使用 LLM 时 Prompt 的重要性，推荐使用 **system prompts** 或 **custom instructions** 来引导模型的输出风格。
   - 一名成员建议利用模型已经了解的现有类比（例如“扮演 Elon Musk”），以避免重复造轮子。
- **用户遇到 OpenRouter API 401 错误**：一名成员报告在支付 API 费用后收到 **401 Unauthorized 错误**，尽管账户内有可用额度，并怀疑这是“骗局，别点”。
   - 讨论中未提供额外的建议或解决方案。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1454054312660439082)** (12 messages🔥): 

> `Jules improvements, Free AI API, Misleading throughput numbers, GLN aggressive batching` 


- **Jules 获得谷歌级别的升级**：一位用户报告称 **Jules** 已显著改进，并通过 **Playwright** 验证了更改，运行速度**更快**，且没有 **UI bug** 或冻结现象。
   - 他们使用的是 **2.5 Pro**，并注意到自上次使用以来，在小型重构和功能添加方面有所改进。
- **免费 AI API 访问：是分享还是圈套？**：一位用户询问是否可以在频道内分享一个**免费 AI API**。
   - 社区成员被邀请就此类分享的允许性发表意见。
- **吞吐量陷阱与欺骗**：一位用户警告说，**throughput（吞吐量）数据**可能具有极大的误导性，表明实际性能可能与报告的指标不符。
   - 附带的[图片](https://cdn.discordapp.com/attachments/1392278974222307469/1454214082168229960/image.png?ex=695045e7&is=694ef467&hm=a14e69d3c2373d05b72f5b3851ad83a0c7dcfab896efacda8f9c26625e467d9a)显示，**Novita** 声称有 **80 TPS**，但实际运行速度约为 **20 TPS**。
- **GLN 节流思考过程**：一位用户发现，通过编程方案使用 **GLN** 时，在 **OR endpoint** 上经历了激进的批处理。
   - 他们报告称，它*以良好的速度运行，然后就……停顿一秒钟*，这很令人烦恼。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1454044256267272344)** (90 messages🔥🔥): 

> `推理过程中的 RAM 分配、Claude Distills 与现代思考模型、针对 MacOS 的 ffmpeg-mcp、作为 OpenAI Endpoint 客户端的 lmstudio、远程 LM Studio` 


- **关于推理时预分配 RAM 的辩论**：成员们讨论了推理过程中 RAM 的分配方式，对比了预先分配全部内存与按需增长内存使用的差异。根据成员 *.lithium* 的说法，*内存使用量会随需求增长，而不是预先分配所需的全部内存*。
   - 一位成员在加载 **GGUF** 后指出，*当模型驻留在内存中时，系统的 RAM 使用量会逐渐增加到设定值并保持稳定*。
- **Claude Distills 在思考模型中占据主导**：据一位成员称，目前唯一优秀的现代思考模型是 **Claude distills**，并点名 **GLM** 和 **Minimax** 作为例子。
   - 他质疑参数规模如何影响模型思考的效用，并好奇较小的模型是否能完全捕获在大版本中所学到的几何结构（geometry）。
- **工具冗余（Tool Bloat）辩论激烈**：一位成员认为，从 MCP 服务器为模型提供数十个工具会引入不必要的上下文膨胀（context bloat），而单个 **run_shell_command** 工具更具通用性，几乎可以完成任何任务。
   - 另一位成员展示了一张长长的工具使用列表图，并承认 *我将 Claude 用于高级编程项目和学术研究*。
- **LM Studio 通过插件实现远程连接**：一名用户询问是否有办法将 **lmstudio 作为 OpenAI endpoint** 客户端来与远程 lmstudio 服务器通信，另一位成员提供了 [Remote LM Studio](https://lmstudio.ai/lmstudio/remote-lmstudio) 插件的链接。
   - 另一人补充道 *我每天都在用这个*，表示赞赏。
- **宏按键板（Macro Pad）用户发现 OpenRouter 标志**：一位用户分享了他们新买的宏按键板照片，另一名用户注意到了上面的 **OpenRouter logo**。
   - 一位成员评论道 *这太硬核了（Thats autistic），新模型发布时手速一定要快*。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1453992995916611634)** (91 messages🔥🔥): 

> `将多块 GPU 连接到 i3、Nvidia 热修复驱动程序 591.67、用于推理的 PCIe 通道、DDR5 ECC 价格差异、Blackwell 与 3080 的 VRAM 对比` 


- **i3 CPU 驱动四显卡机架？**：一名用户询问是否可以将 4 块 GPU 连接到 **i3-12100**，并好奇其“所谓”的 20 条通道是否足够，同时考虑放弃购买 **5600x** 以节省资金。
   - 另一位成员确认在支持通道拆分（bifurcation）的情况下是可行的，但警告会有性能损失，特别是低于 **gen3 x8** 时，因为消费级 CPU 并非为 4 块 GPU 设计。
- **Nvidia 发布热修复驱动，玩家欢呼**：Nvidia 发布了 [GeForce Hotfix 显卡驱动程序版本 591.67](https://nvidia.custhelp.com/app/answers/detail/a_id/5766/~/geforce-hotfix-display-driver-version-591.67)。
- **PCIe 带宽对推理够用吗？**：讨论围绕推理所需的 PCIe 通道展开，一位用户建议 **Gen4 x1** 的速度可能就足够了。
   - 其他人则警告说，降至 **gen3 x8** 以下可能会导致大约 **40-50%** 的性能损失，同时也承认通过芯片组以 **gen3x1** 运行的多卡设置仍然可以 *正常工作*，并引用了展示此类配置的 [YouTube 视频](https://www.youtube.com/watch?v=So7tqRSZ0s8)。
- **RAM 价格高得令人不安**：一位成员指出 DDR5 ECC RAM 价格过高，另一人提到一个月前仅以 400 美元购买了 **2x48GB 6400 MHz cl32 DDR5**，结果现在亚马逊上价格飙升至 950 美元，推测之前可能是员工定价错误。
   - 这导致了因性价比考虑而建议从 **64GB** 开始配置。
- **Blackwell RTX Pro 4000 vs 3080**：一名用户在为 GPU 机架购置 **2x 3080 20GB 显卡**还是为桌面购置单块 **RTX Pro 4000 Blackwell** 之间犹豫不决，对比了 Blackwell 的速度与 3080 更大的 VRAM。
   - 该用户旨在将 **Blackwell** 与现有的 **3090 Ti** 和 **3090** 组合以达到 72GB VRAM，Blackwell 可能会运行在 **Gen4 x1** 插槽上，虽然承认其在游戏方面可能较慢，但由于支持 **原生 fp4/fp8**，在推理方面会有所裨益。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1454013151568728084)** (97 messages🔥🔥): 

> `ChatGPT 订阅问题, AI 图像生成, Sora 变现, AI 开发项目, AI 检测` 


- **ChatGPT 客服封禁用户！**：一位用户在被封禁并创建新账户后，抱怨 [ChatGPT 的订阅问题](https://chat.openai.com)，这违反了 **OpenAI** 的 **ToS**（服务条款）。
   - 其他用户指出，*通过创建多个账户来绕过封禁*是违反规则的，但 **OpenAI** 允许拥有多个账户以区分个人和工作用途。
- **AI 壁纸引发 Musk 争论**：一位用户分享了一张包含多位革命性人物（包括 **Elon Musk**）的 AI 生成壁纸。
   - 虽然一些人欣赏这张壁纸，但另一些人批评将 **Musk** 包含在内，创作者澄清说人物是由 **ChatGPT** 选择的，而非他们个人决定。
- **Sora 的定价结构仍不明朗**：用户讨论了 **Sora** 潜在的变现模式，有人根据 Google 搜索结果建议价格为 **每秒 0.50 美元**。
   - 另一位用户澄清说，**Sora 2** 最初将免费提供并设有慷慨的额度限制，参考了 [这篇 OpenAI 博客文章](https://openai.com/index/sora-2/)。
- **DIY AI 项目寻求编程盟友**：一位用户正在为其从零开始构建的 **AI + LLM 项目** 寻求编程合作伙伴，其技术栈包括 **HTML, CSS, JavaScript, Node.js, SQLite, C++, Capacitor, 和 Electron**。
   - 另一位成员正在使用 **Hugging Face** 上的 **Wav2Vec2 模型** 开发一个**鸟鸣声音频分类模型**。
- **解析病毒式传播的 YouTube Short 的 AI 生成方式**：一位用户询问使用了哪种 **AI** 来制作 [这段 YouTube 短视频](https://youtube.com/shorts/D3qJdTwYE9g?si=f6iwOxl9Qludlqpl)。
   - 目前还没有人回答该用户的问题。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1454024077239910587)** (4 messages): 

> `用于分析音高的 AI, 用于分析语调的 AI, 用于分析肢体语言的 AI, 用于语言学的 AI` 


- **用于分析音高、语调、肢体语言和语言学的 AI 建议**：一位成员询问哪种 AI 最适合同时测试特定的 **音高 (pitch)**、**语调 (tone)**、**肢体语言 (body language)** 以及 **语言学 (linguistics)** 特征。
- **审查软件文档的 Token 成本**：一位成员表示，审查软件文档的成本约为 **每个 prompt 1 到 2 美元**，具体取决于 prompt 和响应的大小。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1454127563948556298)** (17 messages🔥): 

> `用于 Prompt 优化的 CustomGPTs, 评估 Prompt 质量, Meta-Prompt 系统, 针对编程与对话式 AI 的 Prompt Engineering` 


- **CustomGPTs 成为 Prompt 优化的热门话题**：一位成员询问了用于改进 prompt 的首选 **CustomGPTs**，并分享了 [示例](https://cdn.discordapp.com/attachments/1046317269069864970/1454127562388148379/image.png?ex=694ff553&is=694ea3d3&hm=a0f98f811f3edc28d302285f86750f6e129a61c5569fc51a2cbfd260ab678bb4&)。
   - 回复强调了 prompt 质量的主观性，以及定义具体评估标准的重要性。
- **工程师设计评估 Prompt 输出的规范**：成员们讨论了如何评估一个 prompt 是否*更好*，并提议构建一个 **eval spec**（评估规范），通过针对该规范测试不同的 prompt 来确定更优的输出分布。
   - 工程师为每个用例设计 eval spec，**Prompt Engineering** 不应仅仅是随机的聊天尝试；之所以存在更好或更差的 prompt，是因为存在更好或更差的输出分布。
- **针对特定用例，自定义 Meta-Prompt 系统更受青睐**：一位成员没有使用 **CustomGPTs**，而是开发了针对其特定需求的 **meta-prompt 系统**，并建议其他人也构建自己的系统。
   - 他们分享了关于如何开始的 [链接](https://discord.com/channels/974519864045756446/1046317269069864970/1453392566576877599)，强调 *一个好的 meta-prompt 应该推荐植根于机器学习原则的 prompt，并强制执行输出模板*。
- **不可验证领域需要评估准则 (Rubrics)**：对于 **coding agent** 的 **prompt engineering**，改进更容易被“感知”，相比之下，测量和测试功能性要比测试“风格”容易得多。
   - 在不可验证的领域需要一套评估准则 (rubric)，因为相比之下，可验证的领域会让 prompt 规范编写变得非常简单。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1454127563948556298)** (17 messages🔥): 

> `CustomGPTs for prompt improvement, Prompt evaluation metrics, Meta-prompt systems, Prompt engineering for coding agents, Rubrics for non-verifiable domains` 


- **用戶尋求用於 Prompt 增強的 CustomGPTs**：一名成員詢問是否有推薦的、用於優化 Prompt 的 **CustomGPTs**，並分享了幾個他們正在使用的範例。
   - 有人提出質疑，該如何定義是什麼讓一個 Prompt 比另一個「更好」。
- **制定評估規範（Eval Spec）可精煉 Prompt**：一名成員建議創建 **eval spec** 來測試 Prompt，並確定哪種 Prompt 能產生更好的輸出分佈，且需針對每個具體場景進行定制。
   - 他們強調 **Prompt Engineering** 的目標是獲得更好的輸出（如有效的 JSON 或 YAML），而不僅僅是日常閒聊。
- **設計 Meta-Prompt 系統並強制執行輸出模板**：一名成員建議根據 **機器學習原理（machine learning principles）** 進行 Prompt 撰寫，並強制執行 **輸出模板（output template）**，通過定義規範（Specs）和測試來評估 Meta-Prompt。
   - 他們強調 **Human-in-the-loop**（人機協作）是不可或缺的。
- **手工打造的 Meta-Prompt 系統表現出色**：一名成員分享說他們不使用 CustomGPTs，而是使用 **手工打造的 Meta-Prompt 系統**，並建議為特定用例構建自定義系統。
   - 他們還鏈接了關於如何開始的[早期訊息](https://discord.com/channels/974519864045756446/1046317269069864970/1453392566576877599)。
- **Coding Agent 受益於模型增強**：一名成員分享道，對於他們一直在開發的 **Coding Agent**，改進主要來自於 **模型增強（model enhancements）**。
   - 他們使用 **評分系統（grading system）** 來保存輸出，並定性地評估改進情況；由於缺乏可測試的功能性，對話維度的改進較難評估。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1453980883022516224)** (14 messages🔥): 

> `DSLs, PTX, CDNA ISA, AGX assembly` 


- **引發對 DSL 指南的憧憬**：成員們開玩笑說，應該有人做一份關於「如何選擇 DSL」的指南，因為現在 **DSL 太多了**。
   - 另一名成員俏皮地回應：[*PTX is all you need*](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)。
- **Kernel 難題：各行其是？**：觀察發現許多 **ML 開發者** 開始編寫 Kernel 並認為自己有更好的語言構想，但彼此並不合作。
   - 正如一位成員所說，「大概有 5000 個人在同一週產生了同樣的想法，但從不互相交流」。
- **ISA 替代方案建議**：在有人建議 PTX 是唯一選擇後，其他成員建議「寫 CDNA ISA」或「寫 AGX 彙編」。
   - 另一名成員對此回覆道：「再見了，可移植性（portability）」。


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1454130881609334796)** (4 messages): 

> `cuTile Performance, GPU Programming Evolution, DSL Challenges` 


- **cuTile 的性能：現在判斷還為時過早**：雖然 **Nvidia** 的開發者暗示計劃對其進行優化，但鑑於 **cuTile** 上週才公開發佈，現在假設其性能優於 **Triton** 還為時過早。
   - 目前的推測是 **cuTile** 的字節碼（Bytecode）和 IR 設計符合 **Nvidia** 未來的 GPU 路線圖，而 **Triton** 在去年才加入了 Warp Specialization 和 TMA 支持。
- **GPU 編程平行於 JavaScript 框架熱潮**：GPU 編程被比作 JavaScript 框架時代，不斷湧現旨在簡化編程模型的新語言和抽象。
   - 共同目標是通過更高級的工具和更安全的抽象來促進開發、增強開箱即用的性能並減少錯誤。
- **DSL 難以平衡簡潔性與複雜性**：DSL 面臨的挑戰是既要讓基於張量的編程變得簡單，又要支持 Warp Specialization 和對稱內存（Symmetric Memory）等複雜優化，平衡易用性與高級控制。
   - 目前還沒有人完美解決這種平衡，儘管來自 **Nvidia** 的 **cuTile** 是一種很有前途的硬件特定方案，未來有可能變得更加硬件無關（Hardware Agnostic）。


  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1454155157691564193)** (2 messages): 

> `Scientific Note-Taking, Technical Note-Taking, LaTeX, Orgmode, Excalidraw` 


- **用户寻求统一的科学笔记方案**：一位用户正困扰于**纯文本文件**、**Excalidraw** 绘图、自制的**思维导图应用**、**LaTeX** 和 **Markdown** 的混合使用，希望能为科学/技术笔记寻找一个统一的解决方案。
   - 该用户发现 Markdown 功能不足，而 LaTeX 在日常使用中过于繁琐，同时也不喜欢为了使用 Orgmode 而去学习 Emacs。他渴望一种既具备可读性、支持数学排版和图表，又能编译为 TeX 的格式。
- **纸笔笔记优于电脑**：针对原始问题，一位用户建议使用**纸和笔**的方法进行记录。
   - 未提供进一步的详细说明。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1454076440453583001)** (9 messages🔥): 

> `NVIDIA Leaderboard Updates, nvfp4_dual_gemm Leaderboard` 


- **NVIDIA 排行榜迎来新提交**：针对 NVIDIA 上的 `nvfp4_dual_gemm` 排行榜有多次提交，其结果（微秒，µs）各不相同。
- **NVIDIA 第三名之争**：用户 <@1291326123182919753> 多次在 NVIDIA 排行榜上获得**第三名**，成绩稳定在 **14.6 µs**。
- **NVIDIA 个人最佳成绩**：用户 <@772751219411517461> 在 NVIDIA 上创下了 **21.2 µs** 的**个人最佳成绩**。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1454217048107516096)** (1 messages): 

> `Cute DSL, Competitions in channel` 


- **竞赛现已开启！**：一名成员指出 **#cute-dsl** 频道目前正在举行竞赛。
   - 他们建议加入该频道，通过应用实践的方式学习该语言。
- **DSL 学习机会**：参与者可以通过竞赛参与 Cute DSL 的动手学习。
   - 这种方法有助于实际应用和技能提升。


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1454090488549343285)** (3 messages): 

> `Tensor Handles, OpNode IR, Runtime Buffer Allocators, OpCode.ADD, ascii printer for IR` 


- **Tensor Handles 已构建**：构建了两个指向 `OpNode` IR 的 `Tensor` handle，该 IR 使用 `Runtime` `Buffer Allocators`。
   - 发布者提到 *一切都在构建 IR*。
- **下发 OpCode.ADD 的后续步骤**：下一步是为加法操作下发 `OpCode.ADD`。
   - 他们补充说，需要*为 IR 实现 ASCII 打印机，或者复用 tinygrad 的 VIZ 基础设施，以简化调试周期*。
- **teenygrad 复用了 tinygrad 90% 的抽象**：发布者解释说，之所以花了这么长时间才达到两个 Tensor 相加的阶段，是因为 *teenygrad 复用了 tinygrad 90% 的抽象*。
   - 好消息是，对于 Karpathy 对 Bengio 等人 2003 年论文的复现，**FFN 只需要两个额外的算子**：**matmul 和 tanh**。
- **Golden Tests 即将到来**：发布者希望通过 **OpNode IR** 建立一些 Golden Tests。
   - 他们打算开始利用 **CI** 提供反馈信号，并且可能会放弃 VIZ=1 网页调试器，转而使用 ASCII 调试。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1454151728323629118)** (2 messages): 

> `Helion usage in vLLM` 


- **Helion 在 vLLM 中得到应用**：一位用户对 **Helion** 与 **vLLM** 的集成表示兴奋。
   - 他们请求另一位用户的电子邮箱地址，以便为新的一年安排一些事务，推测与此集成有关。
- **由 Helion-vLLM 集成引发的新年计划**：讨论强调了在 **vLLM** 框架内利用 **Helion** 所带来的潜在益处和未来计划。
   - 该用户提出的安排建议预示着来年会有相关的合作机会或进一步的发展。


  

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1453985512087097457)** (1 条消息): 

> `Cute-DSL vs C++ Cute, Cute API limitations, Cute consensus preference` 


- **Cute-DSL vs 经典的 C++ Cute**：一位成员询问了 **Cute-DSL** 相比 **C++ Cute** 的固有局限性，并表示有兴趣学习 **Cute**。
   - 他们还在寻求两者之间的共识偏好，并指出在最近的一个问题中 **Cute-DSL** 解决方案非常普遍，这表明它并不显著落后于 **C++ API**。
- **学习 Cute：C++ 还是 DSL？**：一位成员正 *考虑下定决心* 尝试学习 Cute，想知道 **C++** 和 **DSL** API 之间是否存在共识偏好。
   - 上一个问题中的许多解决方案都使用了 Cute-DSL，表明它可能与 C++ API 旗鼓相当。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1454104934835552288)** (3 条消息): 

> `handwritten kernels, tinygrad IR interpreter, tilelang` 


- **寻求手写 Kernel 实现方面的协助**：一位成员请求协助为一个评估 **tinygrad IR** ([tilelang](https://tinygrad.org/)) 的 **interpreter** 实现 **handwritten kernels**。
   - 他们欢迎任何具备必要技能的人做出贡献。
- **一致性获赞，欢迎贡献**：该项目的一致性（consistency）受到了赞扬，并呼吁有能力的个人参与贡献。
   - 尽管最初对关于一致性的幽默感有些困惑，但对具备所需专业知识的人员的贡献邀请仍然有效。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1453971335117275237)** (26 条消息🔥): 

> `Agentic AI Course, RAG learning Resources, ML Project Water Quality Testing, Open Source OCR for Medical Documents, Hottest LLM Model` 


- **未找到 Agentic AI 课程频道**：一位成员询问 **Agentic AI Course** 频道的位置，但在给定语境中未发现特定频道。
- **RAG 学习资源**：一位成员计划开始学习 **RAG** 并构建一些项目，请求推荐相关课程或资源。
- **旨在改善水质检测的 ML 项目**：一位成员正在寻求一个 **ML 项目**的帮助，该项目专注于解决与水质检测相关的现实问题，特别是预测铀原位回收 (ISR) 运营场地的 **水质退化**。
   - 该项目还涉及识别在减轻漏洞方面的 **数据缺口** 并开发 **监控程序**。
- **寻求医疗文档的 OCR 模型**：一位成员询问了针对复杂医疗文档（包括扫描版 PDF 和手写文档）的最佳 **开源 OCR 模型**。
   - 另一位成员建议查看来自 Maz 的 [openmed 项目](https://hf.co/openmed)。
- **iPhone 上用于 GPT 克隆的 Liquid AFMs VLM**：一位成员分享道，[LiquidAFMs](https://huggingface.co/LiquidAI/LFM2-VL-3B-GGUF) 的 **3B VLM** 是 iPhone 上 GPT 克隆的最佳选择，因为它们提供了非常扎实的回答，并且在 **移动 CPU 上可达 100 tok/s**。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1454018449952210956)** (4 条消息): 

> `Coding models released, AI-powered Django REST API Generator, New model released` 


- **编程模型上架**：为特定库设计的编程模型已 [发布](https://huggingface.co/blog/codelion/optimal-model-architecture)。
   - 这些模型以 [集合 (collection)](https://huggingface.co/collections/Spestly/lovelace-1) 形式提供。
- **AI 驱动的 Django REST API 生成**：一个 AI 驱动的 Django REST API 生成器已构建并部署为 [Hugging Face Space](https://huggingface.co/spaces/harshadh01/ai-rest-api-generator)。
   - 它能根据简单的提示词生成完整的 Django + DRF CRUD 后端，包括 models, serializers, views 和 URLs，允许用户立即下载项目。
- **Genesis 152M Instruct 模型首次亮相**：新模型 **Genesis-152m-instruct** 已发布，可在 [此处](https://huggingface.co/guiferrarib/genesis-152m-instruct) 查看和讨论。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1454071910571835422)** (3 messages): 

> `Agent course, Hugging Face Courses` 


- **新 Agent 课程学生询问证书截止日期**：一名刚开始 Agent 课程的新学生询问了第二份证书的截止日期。
   - 另一名学生询问 *在哪里可以找到课程？*
- **提供了 Hugging Face 课程链接**：一名成员分享了 [Hugging Face courses](https://huggingface.co/learn) 的链接。
   - 该链接可能回答了其中一名成员的问题。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1453998663612502139)** (21 messages🔥): 

> `LLMs social deduction game, Claude one-shot UI, Smart watch data to external AI, GPT models and ads, Zai and Minimax IPO` 


- **LLM 玩黑手党 (Mafia)**：一名成员创建了一个游戏，多组 **LLM** 进行社交推理游戏（**Mafia**），它们在各自的轮次中通过交谈试图证明自己的清白。
- **Claude 的 UI 赢得青睐**：一名成员表示最喜欢 **Claude 的 one-shot UI**，并请其他人根据截图识别大模型。
   - 另一位用户指出第一个 UI 泄露了身份，猜测第二个是 **Claude**，第三个可能也是 **Claude**。
- **智能手表与 AI 集成**：一名成员正在对所有内容进行模板化，并订购了一款新的 **smart watch**，开始将数据与外部 **AI company** 集成，但会尝试尽可能将数据保留在本地。
- **GPT 模型可能很快会优先考虑赞助内容**：**OpenAI** 可能会调整其 **GPT models** 以通过推广广告来获取收入流，从而优先展示赞助内容。
   - 这可能意味着 **GPT models** 将不再专注于高智能，而是转向赞助内容。
- **Zai 和 Minimax 计划在香港 IPO**：**Zai** 和 **Minimax** 计划在未来几周内在 **Hong Kong** 进行 **IPO** 上市。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://x.com/openbmb/status/2004539303309750341?s=46
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://x.com/openbmb/status/2004539303309750341?s=46
  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1454019495608844401)** (23 messages🔥): 

> `Kimi coding vs Gemini, Kimi Researcher Model Speculations, AI implementation in society, Fact checking with LLMs` 


- **Kimi 编程能力在设计上媲美 Gemini 3**：一位用户反馈 **Kimi** 编程无法与 **Cursor** 集成，但在 **Roo Code** 中由 **Kimi** 创建的 HTML 设计令人印象深刻。
   - 他们在设计用途上将 **Kimi coding** 与 *Gemini 3 & GPT-5.1* 进行了比较。
- **各国 AI 社会影响对比**：一名成员声称中国在将 AI 应用于社会方面处于领先地位，例如在 **医疗保健**、**养老服务** 和 **交通优化** 领域。
   - 另一位用户同意这 *正是* AI 应该做的，即 **灾害预防**，并指出 Google 和 ChatGPT 也会审查某些问题。
- **使用 Kimi AI 助手进行深度探索**：一名成员分享了为了研究目的与 **Kimi AI Assistant** 对话的链接，并强调了事实核查的重要性。
   - 另一位用户表示赞同，并强调需要对从 **LLM** 获取的任何可疑信息进行 *交叉核对、事实核查和压力测试*，并利用新的聊天上下文来获取新鲜视角。
- **Kimi Researcher 模型推测**：一名用户询问 **Kimi Researcher** 使用的是 **Kimi K2 Thinking Turbo** 还是较旧的 **K1.5 model**，以及是否采用了尚未公开的 **K2 VL model**。
   - 他们对使用 **K1.5** 处理基础任务以外的工作表示犹豫，因为 **K2T** 的性能更优，对此另一位用户回复称似乎没有任何官方文档说明。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1454083926888022130)** (10 条消息🔥): 

> `Groq 推理芯片, NVIDIA NPU 收购, 中国芯片` 


- **Groq 被 NVIDIA 化了！**: [Groq](https://groq.com/) 和 [NVIDIA](https://www.nvidia.com/) 已达成一项非排他性的推理技术许可协议，旨在全球范围内加速 AI 推理。
   - 一些社区成员认为，**NVIDIA** 正在通过将另一个竞争对手踢出局来玩弄他们最新的价格通胀手段。
- **推理技术 Acqui-Hire 缓解 GPU 短缺？**: 成员们讨论了 **Groq 的推理芯片** 现在将变成 **NVIDIA NPUs** 的可能性，这可能会让普通消费级 GPU 再次回归市场。
   - 一位成员补充说，NVIDIA 的创始人 **Jen-Hsun**（黄仁勋）是华人。
- **中国芯片落后了？**: 一位成员表示，中国芯片*可能*在 3 年内达到 **H100** 的水平，而届时 **H100** 将已经是 5 年前的产品了。
   - 另一位成员表示，他们对*中国芯片在短期内能与美国/台湾芯片竞争不抱太大期望*。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1454214166083538995)** (7 条消息): 

> `Karpathy 重构, AI Agent 编程, Rob Pike 观点` 


- **Karpathy 论 AI 驱动的重构**: [Andrej Karpathy 反思](https://x.com/karpathy/status/2004607146781278521?s=46)了由 **AI agents** 和 **LLMs** 引起的**软件工程**巨大变革，将其描述为一个涉及随机实体的新抽象层。
- **Pike 的视角加入战场**: 一位用户分享了 [Rob Pike 的帖子链接](https://skyview.social/?url=https%3A%2F%2Fbsky.app%2Fprofile%2Frobpike.io%2Fpost%2F3matwg6w3ic2s&viewtype=tree)，作为对 Karpathy 帖子的回应。


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1454198436693016820)** (2 条消息): 

> `Torchax, Unified Memory, 糟糕的操作系统` 


- **Torchax 点燃操作系统救赎希望**: 用户表示，**torchax** 结合高达 **128GB Unified Memory**，可能会减轻他们对使用某种特定操作系统的后悔感。
   - 目前尚未分享关于预期应用或正在考虑的 **torchax** 具体功能的细节。
- **哀叹“糟糕的操作系统”**: 一位用户幽默地抱怨其系统上“糟糕的操作系统”。
   - 他们在 **torchax** 的潜力和巨大的 **128GB Unified Memory** 中找到了慰藉，认为这可以弥补操作系统的缺点。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1454138340960895016)** (4 条消息): 

> `Claude, 代码库解释` 


- **关于 Claude 能力的讨论**: 一位成员询问了使用 **Claude** 的潜力，并寻求关于其能力的意见。
   - 该用户似乎在考虑 **Claude** 是否能满足特定的需求或任务，可能是与其他 AI 模型或工具进行对比。
- **寻求向 Claude 解释代码库的高效方法**: 一位成员正在寻找在每次打开项目时向 **Claude** 解释其代码库的最佳方式，并对需要手动更新 `claude.md` 文件表示沮丧。
   - 他们对建议和替代方案持开放态度，因为虽然已有一个解决方案，但仍想探索社区中关于自动执行此过程的其他推荐。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1454125625278333030)** (3 条消息): 

> `向 Claude 解释代码库, 自动化 Claude 更新` 


- **自动化 Claude 代码库介绍**: 用户正在寻求建议，如何在每次打开项目时最好地向 **Claude** 解释他们的代码库，旨在将此过程自动化，而不是手动更新 `claude.md` 文件。
   - 一位成员询问该用户是否是指使用 **aider**。
- **消除对 Claude 知识库的手动更新**: 用户表达了对手动更新 `claude.md` 文件以使 **Claude** 了解其代码库的挫败感。
   - 他们正在积极寻找替代方法来自动化此过程，避免手动更新。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1454140120449028248)** (5 messages): 

> `AI 全自主系统，全 AI 自主系统的局限性，上下文管理，长期记忆实现` 


- **AI 全自主系统崭露头角**：一位成员正在开发一个 **full autonomy system for AIs**（AI 全自主系统）以实现 24/7 推理，以及类似于 **OpenAPI MCP tools** 的自然语言工具。
   - 该成员询问其他人是否开发过类似系统，并咨询了目前的局限性，特别是关于 **long-term memory**（长期记忆）和 **long context**（长上下文）的问题。
- **持续自我摘要具有潜力**：一位成员建议 **continuous self-summarization**（持续自我摘要）可以解决自主 AI 系统中的长期记忆限制。
   - 他提到正在探索通过 **summarizing projection layers**（摘要投影层）写入 scratchpad，但承认在完善该想法方面存在时间限制。
- **上下文管理成为重点**：一位成员强调在实现长期记忆方案时，**context management**（上下文管理）是首要任务。
   - 他指出，如果没有精心设计的上下文管理，**LLM 在处理涉及多个子任务的复杂任务时会感到吃力**。
- **考虑通过修改 Ollama 源代码进行记忆扩展**：一位成员正考虑修改 **Ollama 源代码**，以便在主 LLM 文件之外开发一个 **"second memory"**（第二记忆）系统。
   - 他坦言，即使有 AI 的辅助，理解源代码也具有挑战性。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1454222409690251284)** (2 messages): 

> `all-the-noises GitHub, arXiv` 


- **All The Noises 项目浮现**：一位成员分享了 [All The Noises project](https://all-the-noises.github.io/main/index.html) 的链接，这是一个汇集了 **多种噪声** 的 GitHub 页面。
   - 其中的噪声包括 *brownian*（布朗噪声）、*circuit*（电路噪声）、*pink*（粉红噪声）、*sine*（正弦噪声）、*tan*（正切噪声）和 *uniform*（均匀噪声）。
- **分享了 arXiv 链接**：一位成员分享了一个 [arXiv 链接](https://arxiv.org/abs/2512.21326)，该链接似乎与未来的研究工作有关。
   - 考虑到 **2025 年** 这个年份，这很可能是该成员正在构思的未来研究方向的一个指向。