---
companies:
- anthropic
- openai
- google-deepmind
- hugging-face
date: '2025-12-18T05:44:39.731046Z'
description: '**Claude Skills** 自 10 月发布以来获得了显著的关注，关于 Claude Skills 的演讲在一天内突破了 10
  万次观看的里程碑，预示着其采用率和重要性正在不断提升。相关公告包括：支持组织管理员、推出新的技能目录（Skills Directory），以及向名为 **Agent
  Skills** 的开放标准转型。


  在前沿模型发布方面，**OpenAI** 发布了 **GPT-5.2-Codex**，它被誉为最优秀的智能体编程模型，在原生压缩、长上下文可靠性和工具调用方面进行了改进，并强调了对现实世界安全的影响。**Google
  DeepMind** 推出了 **Gemini 3 Flash**，重点是将速度作为影响工作流和用户参与度的产品特性，同时还推出了 **FunctionGemma**
  和 **T5Gemma 2**，强调设备端部署、微调和多模态能力。'
id: MjAyNS0x
models:
- claude-skills
- gpt-5.2-codex
- gemini-3-flash
- functiongemma
- t5gemma-2
people:
- sama
- gregbrockman
- philschmid
title: Claude Skills 持续扩展：开放标准、目录与组织管理。
topics:
- agentic-ai
- fine-tuning
- long-context
- tool-calling
- on-device-ai
- multimodality
- security
- workflow-optimization
---

**Skills 正在走上 MCP 的道路！**

> 2025/12/17-2025/12/18 的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 24 个 Discord（207 个频道，7381 条消息）。预计节省阅读时间（以 200wpm 计算）：603 分钟。我们的新网站现已上线，提供完整的元数据搜索和极具氛围感的往期内容展示。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

[5.2 Codex](https://openai.com/index/introducing-gpt-5-2-codex/) 和 [FunctionGemma](https://huggingface.co/collections/google/functiongemma) 中有一些略微有趣的发布，但一年后你最可能关心的故事是 **Claude Skills** 的持续增长。它于 [10 月](https://news.smol.ai/issues/25-10-16-claude-skills) 发布，当时几乎被普遍嘲笑为“Markdown 文件夹”和/或从 MCP（现已[移交给 Linux Foundation](https://news.smol.ai/issues/25-12-09-devstral2)）的转型，但在业内人士中，它的势头一直在增长、增长、再增长。衡量这种增长的一种方式是 [Claude Skills 演讲](https://www.youtube.com/watch?v=CEvIs9y1uog&t=6s) 在 [1 天内突破了 10 万次观看](https://x.com/swyx/status/1998786773477110049?s=20) —— 轻松成为 AIE 历史上最快达到这一里程碑的视频，并可能成为 2025 年[第二个](https://www.youtube.com/watch?v=8rABwKRsec4&t=2s)百万播放量的演讲。


![来自 Anthropic 的两名 AI 工程师在 Code Summit 上发表演讲，横幅建议“不要构建 Agent，而是构建 Skills”](https://resend-attachments.s3.amazonaws.com/RRxm1kG9b4tNPvy)


今天的公告包括：

- 跨组织的 [Skills 组织管理员支持](https://claude.com/blog/organization-skills-and-directory)
- 一个新的 [Skills 目录（似乎与 MCP 有重叠）](https://youtu.be/NGAFwBfuiJ8)
- 成为一个“开放标准”，并拥有一个厂商中立的名称“[Agent Skills](https://agentskills.io/home)”！

所有这些看起来都是增量式的补充，但大局是 Skills 的采用正在认真且持续地增长。如果我们的线下交流有任何预示，那么你可能也低估了它们。

我们上一次做出这种非新闻类的趋势预测还是关于 [Claude Code](https://news.smol.ai/issues/25-06-20-claude-code) 的时候。

---

# AI Twitter 回顾

**前沿模型发布：GPT-5.2-Codex, Gemini 3 Flash, 以及端侧 Gemma 变体**

- **GPT-5.2-Codex (OpenAI)**: OpenAI 将 **GPT-5.2-Codex** 定位为其最佳的 “agentic coding” 模型，并重点介绍了在 **native compaction**、长上下文可靠性和 tool-calling 方面的改进；该模型已在 Codex 中面向付费 ChatGPT 用户推出，API “即将推出” ([公告](https://twitter.com/OpenAIDevs/status/2001723687373017313)，[发布说明 + 网络双重用途框架](https://twitter.com/OpenAIDevs/status/2001723693496775167)，[产品推文](https://twitter.com/OpenAI/status/2001766212494332013))。Sam Altman 强调了现实世界的安全影响（React 漏洞披露），并提到正在探索用于防御性网络能力的 **trusted access** ([影响说明](https://twitter.com/sama/status/2001724828567400700)，[trusted access](https://twitter.com/sama/status/2001724830584901973))。Greg Brockman 等人进一步宣传了“长周期重构/迁移”和漏洞查找能力 ([重构重点](https://twitter.com/gdb/status/2001758275998785743)，[安全视角](https://twitter.com/gdb/status/2001758799657603185))。
- **Gemini 3 Flash 的采用与“速度作为产品特性”**: 多位从业者认为 Gemini 3 Flash 改变了日常工作流，因为速度改变了迭代循环和用户行为（留存/参与度），而不仅仅是 Benchmark ([产品影响引用](https://twitter.com/_philschmid/status/2001492609114456471)，[工作流思考](https://twitter.com/andrew_n_carr/status/2001487412749570549))。还有关于 **SWE-bench Verified** 排名（包括 “Flash 击败 Pro”）的说法和观察，但在推文中缺乏单一权威评估结果的情况下，应将其视为暂定结论 ([示例](https://twitter.com/scaling01/status/2001803023811797433)，[另一个](https://twitter.com/MS_BASE44/status/2001698991801798927))。Google 将 Flash 推入 Gemini 应用，作为“通过语音构建应用”的原语 ([Gemini 应用推出](https://twitter.com/Google/status/2001746491275083925)，[语音转应用推介](https://twitter.com/GeminiApp/status/2001760080518353261))。
- **FunctionGemma (270M) + T5Gemma 2 (encoder-decoder)**: Google/DeepMind 和社区强调小型、**on-device / browser** 部署以及通过 fine-tuning 实现的专业化。FunctionGemma 被定位为一个需要领域调优的 **text-only function calling** 基础模型 ([演示 + 定位](https://twitter.com/xenovacom/status/2001703932968452365)，[集合公告](https://twitter.com/osanseviero/status/2001704034667769978)，[微调指南](https://twitter.com/ben_burtenshaw/status/2001704049490489347))。T5Gemma 2 被推介为罕见的现代 **multimodal, multilingual encoder–decoder** 系列 (270M/1B/4B) ([发布](https://twitter.com/osanseviero/status/2001723652635541566))。生态系统立即响应：Ollama 拉取 ([Ollama](https://twitter.com/ollama/status/2001705006450565424))，Unsloth notebooks ([Unsloth](https://twitter.com/danielhanchen/status/2001713676747968906))，MLX 支持 ([MLX 示例](https://twitter.com/Prince_Canuma/status/2001713991115026738))。

---

**Agents：“技能”标准化、治理 UX 以及长期运行的基础设施现实**

- **Agent Skills 正在成为事实上的可移植层**：Anthropic 的 “Skills” 概念作为一种互操作打包格式（指令/脚本/资源）持续传播，并有强烈的工具链采用信号：VS Code 支持该开放标准 ([@code](https://twitter.com/code/status/2001727543377039647))；社区评论将 Skills 视为类似于 MCP 的标准化历程 ([@omarsar0](https://twitter.com/omarsar0/status/2001714322817368472), [@alexalbert__](https://twitter.com/alexalbert__/status/2001760879302553906))。Artificial Analysis 的 Stirrup 增加了从目录加载 Skills 的功能（Markdown 优先），明确针对 Claude Code/Codex 风格设置的跨平台复用 ([Stirrup 支持](https://twitter.com/ArtificialAnlys/status/2001778418590060819))。
- **Harness 作为 Agent UX 的“分发”方式**：多条推文指出，Agent 的表现不仅取决于模型质量，还取决于 “Harness 选择”（并行工具调用、内存 UX、压缩策略、子 Agent 编排视图）作为一个产品层面 ([Harness 观点](https://twitter.com/Vtrivedy10/status/2001492640076894661), [Agent 模块心理模型](https://twitter.com/Vtrivedy10/status/2001682603460473190))。
- **基础设施不匹配：Serverless vs Agent 循环**：一个尖锐的工程观点是，Agent 系统需要**持久、长时间运行的执行**和状态管理；“Serverless” 模式迫使多步循环采用脆弱的网络变通方案 ([批评](https://twitter.com/anuraggoel/status/2001721861198221629))。这与对编排原语重新产生的兴趣相一致（例如，Temporal 适合后台 Agent：[@corbtt](https://twitter.com/corbtt/status/2001801936916643919)）。
- **Claude Code 扩展了能力范围（网页浏览）**：用户强调 Claude Code 现在能够浏览网页，从而实现过滤 X Feed 并进行异步报告的“监控” Agent（一种轻量但实用的“Agent 作为注意力代理”模式）([构建示例](https://twitter.com/omarsar0/status/2001784722549281001))。

---

**评估、回归与安全测量：METR 时间跨度修复 + OpenAI CoT 可监控性**

- **METR 时间跨度套件已修正**：METR 报告了其时间跨度任务中的两个问题，其中一个**差异化地降低了 Claude 的性能**，并发布了更新后的仪表板数据 ([问题公告](https://twitter.com/METR_Evals/status/2001473506442375645), [仪表板更新](https://twitter.com/METR_Evals/status/2001473519197335899))。评论指出 Sonnet 4.5 被低估了，修复后提升了约 20 分钟 ([反应](https://twitter.com/scaling01/status/2001476927362605354))。元教训：Benchmark 的工程细节会实质性地偏置跨模型比较——特别是当任务对格式或评分边缘情况敏感时。
- **OpenAI：评估思维链 (CoT) 的可监控性**：OpenAI 发布了一个评估套件，旨在衡量模型何时将内部推理的特定方面语言化（涵盖 24 个环境的 13 项评估），并认为可监控性取决于监控器强度和 Test-time compute；后续追问可以挖掘出之前未表达的想法 ([论文/推特链](https://twitter.com/OpenAI/status/2001791131353542788), [后续观点](https://twitter.com/OpenAI/status/2001791136223105188))。Sam Altman 转发了这项工作 ([推文](https://twitter.com/sama/status/2001816114595270921))。Neel Nanda 将其与解释激活的“元模型”联系起来，并询问 Bitter Lesson 的缩放定律是否也适用于可解释性 ([评论](https://twitter.com/NeelNanda5/status/2001795630973493279))。
- **生产环境回归作为一级事故 (Claude Code Opus 4.5)**：Anthropic 承认了关于 Opus 4.5 在 Claude Code 中可能存在性能退化的反馈，并声称正在进行逐行代码审计和转录请求 ([事故推文](https://twitter.com/trq212/status/2001541565685301248))。这提醒我们，集成 IDE Agent 中的“模型质量”通常是模型、系统提示词、工具路由、缓存和 UX 变化的综合体。

---

**系统与开放工具：Mac 上的 MLX 分布式、vLLM MoE 吞吐量以及 diffusion-LM 工具链**

- **MLX 通过 TB5 RDMA (JACCL) 实现多节点**：MLX 增加了一个分布式后端 **JACCL**，利用基于 Thunderbolt 5 的 RDMA 在多台 Mac 之间实现低延迟通信（[公告](https://twitter.com/awnihannun/status/2001667839539978580)，[文档](https://twitter.com/awnihannun/status/2001672689325609028)）。后续更新包括 CUDA 后端改进、更快的 prefill/训练，以及利用 JACCL 的 mlx-lm 张量并行 (tensor-parallel) 推理（[CUDA 改进](https://twitter.com/awnihannun/status/2001679244917907912)，[mlx-lm TP](https://twitter.com/awnihannun/status/2001781067880239597)，[演示加速](https://twitter.com/angeloskath/status/2001739468425040002)）。
- **vLLM 在多节点 H200 上实现宽专家并行 (wide expert-parallel) MoE**：新的基准测试结果声称，通过宽 EP + 负载均衡 + 解耦方法，每张 H200 GPU 的持续吞吐量达到约 **2.2k tokens/s**（高于之前的 ~1.5k）；该帖子强调了通信/KV-cache 瓶颈，并提出了通过 DeepEP all-to-all 和重叠策略进行缓解的方法（[讨论串](https://twitter.com/vllm_project/status/2001695354983723361)）。
- **Diffusion LLM 与混合解码**：一股正在兴起的小浪潮：dLLM 库声称可以将“任何 AR LLM 转换为 Diffusion LLM”，并提供统一的训练/评估（[dLLM](https://twitter.com/akshay_pachaar/status/2001562985043783908)）；DEER 提出了“用 Diffusion 草拟，用 AR 验证”的混合方案（[DEER 提及](https://twitter.com/_akhaliq/status/2001685493919158362)）；TheTuringPost 总结了从 AR 到块扩散 (block-diffusion) 的转变（增加块大小、块内双向注意力、辅助 AR 损失），并报告了 NBDIFF-7B 的结果（[总结](https://twitter.com/TheTuringPost/status/2001697220387913818)，[论文链接推文](https://twitter.com/TheTuringPost/status/2001697302562685034)）。核心结论：社区正趋向于“基于预训练的 AR 进行适配”，而非从头训练 Diffusion LLM。

---

**多模态生成与文档智能：Kling 动作控制、Runway Gen-4.5、Mistral OCR 3**

- **Kling 2.6 动作控制（V2V / 居家动捕感）**：多篇高互动帖子声称 Kling 的新动作控制功能实现了高度可控的全身动作 + 表情 + 嘴型同步（部分帖子称其在“所有指标”上击败了竞争对手，但未提供共享的评估协议）（[日语演示](https://twitter.com/seiiiiiiiiiiru/status/2001502678116110430)，[上手好评](https://twitter.com/WuxiaRocks/status/2001517467852771467)，[热门观点](https://twitter.com/AngryTomtweets/status/2001569619375698199)，[舞蹈测试](https://twitter.com/genel_ai/status/2001532885673873677)）。
- **Runway Gen-4.5**：Runway 宣布 Gen-4.5 现已可用；这组推文大多属于营销层面，包含的技术细节有限（[发布](https://twitter.com/runwayml/status/2001655929796751371)）。
- **Mistral OCR 3**：Mistral 发布 OCR 3，作为一款全新的“前沿”文档智能模型，具有极高的准确性和效率；Guillaume Lample 特别指出了在**手写体、低质量扫描件以及复杂表格/表单**方面的改进（[讨论串](https://twitter.com/MistralAI/status/2001669581275033741)，[基准测试声明](https://twitter.com/MistralAI/status/2001669583296712970)，[Lample 的评价](https://twitter.com/GuillaumeLample/status/2001719413649617404)）。如果实际应用中确实如此，这将具有重大意义，因为 OCR 质量是企业级 RAG/文档 Agent 的硬瓶颈。

**热门推文（按互动量排序）**

- **OpenAI 发布 GPT-5.2-Codex**（Agent 编程 + 终端使用）：[@sama](https://twitter.com/sama/status/2001724019188408352), [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/2001723687373017313), [@OpenAI](https://twitter.com/OpenAI/status/2001766212494332013)。
- **OpenAI 在 ChatGPT 中推出置顶对话功能**：[@OpenAI](https://twitter.com/OpenAI/status/2001751306445430854)。
- **Grok Voice API 声称实现“特斯拉级”低延迟语音 Agent 产品化**（高度宣传性的措辞）：[@MarioNawfal](https://twitter.com/MarioNawfal/status/2001472484869329288)。
- **Karpathy 论“精神食粮”作为 LLM 的内在奖励类比**：[@karpathy](https://twitter.com/karpathy/status/2001699564928279039)。
- **“Galaxy gas”推文走红（非 AI 内容 / 本摘要中信号较低）**：[@coffeebreak_YT](https://twitter.com/coffeebreak_YT/status/2001753564620747195)。
- **Claude Code Opus 4.5 退化问题调查**：[@trq212](https://twitter.com/trq212/status/2001541565685301248)。
- **JS 编写的开源 WebGPU ML 编译器 (“jax-js”)**：[@ekzhang1](https://twitter.com/ekzhang1/status/2001680771363254646)。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Google's Gemma 模型更新

- [**Google's Gemma models family**](https://www.reddit.com/r/LocalLLaMA/comments/1ppun3v/googles_gemma_models_family/) (活跃度: 563): **图片重点展示了名为“Google's Gemma models family”合集的更新，该合集目前包含 329 个项目。该合集与 Google 的 FunctionGemma 模型相关，这些模型旨在针对特定的 function-calling 任务进行微调，包括多轮对话用例。提供了其中一个模型 [FunctionGemma-270M-IT](https://huggingface.co/google/functiongemma-270m-it) 的链接，表明其已在 Hugging Face 上线。讨论指出，根据总模型数与可见模型数之间的差异推测，该合集可能新增了三个模型。** 评论者注意到缺少“Gemma 4”模型，而是将重点放在 FunctionGemma 系列上。关于合集中总模型数与可见模型数不一致的情况，引发了对新增模型的猜测。
    - RetiredApostle 强调了 FunctionGemma 模型，该模型专为微调特定的 function-calling 任务（包括多轮交互）而设计。这表明其重点在于增强模型处理复杂、多步骤流程的能力，这在需要详细任务管理或对话式 AI 的应用中特别有用。该模型可在 [Hugging Face](https://huggingface.co/google/functiongemma-270m-it) 上获取。
    - jacek2023 指出，目前 Gemma 合集中有 323 个可见模型，这意味着可能还有尚未发布的额外模型。这一观察表明 Google 可能正计划扩展 Gemma 家族，推出新模型，从而可能增加该系列的多样性和能力。
- [**Kimi K2 在 4x Mac Studio 集群上达到 28.3 t/s 的推理速度**](https://www.reddit.com/r/LocalLLaMA/comments/1pq2ry0/kimi_k2_thinking_at_283_ts_on_4x_mac_studio/) (活跃度: 376): **图片展示了“llama.cpp (TCP)”与“Exo (RDMA)”在 Mac Studio 集群上的性能对比，突出了“Exo (RDMA)”在 4 节点配置下的卓越性能，其吞吐量达到** `28.3 t/s`**，而“llama.cpp (TCP)”仅为** `16.4 t/s`**。测试是在一个由 4 台 Mac Studio 组成的集群上进行的，重点是评估 Exo 中最近趋于稳定的新 RDMA Tensor 设置。由于 Exo 缺乏像 llama-bench 这样的基准测试工具，使得直接对比变得复杂，但结果表明 Exo 的 RDMA 实现具有显著的性能优势。** 讨论中提到了社区对 Exo 在开发过程中缺乏沟通的不满，尽管 Exo 1.0 以 Apache 2.0 协议发布受到了欢迎。此外，正如 GitHub issue 中所述，社区对为 llama.cpp 添加 RDMA 支持表现出浓厚兴趣。
    - 根据 [Jeff Geerling](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5) 的详细介绍，Kimi K2 模型在 4 台 Mac Studio 集群上实现了 `28.3 tokens per second` 的处理速度。该配置利用了 `15 TB 的 VRAM` 并采用了 `RDMA over Thunderbolt 5`，这在硬件配置和数据吞吐量方面是一项重大的技术成就。
    - 讨论中提到了 `llama.cpp` 支持 RDMA 的潜力，这可能会显著提升其性能。已在 [GitHub](https://github.com/ggml-org/llama.cpp/issues/9493) 上开启了一个 issue 来跟踪此功能请求，表明社区对提高分布式计算环境中的数据传输速度和效率感兴趣。
    - 尽管社区对开发过程有些不满，但 Exo 1.0 在 Apache 2.0 许可证下的发布仍被提及。这种开源性质可能会促进进一步的创新和协作，正如在 [GitHub issue](https://github.com/exo-explore/exo/issues/819) 中社区成员讨论项目方向和贡献时所见。

## 较少技术性的 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. GPT-5.2 基准测试成就

- [**一切都结束了。GPT-5.2 拿下了最重要的基准测试之一，而且遥遥领先！**](https://www.reddit.com/r/singularity/comments/1ppynjo/its_over_gpt_52_aces_one_of_the_most_important/) (活跃度: 1257)：**这张图片是一个突出 GPT-5.2 性能的梗图，暗示它在某项基准测试中显著超越了 Opus-4.5、Claude-4.5 和 Grok-4 等其他模型。柱状图幽默地夸大了版本号以强调 GPT-5.2 的统治地位，但缺乏具体的技术细节或基准测试指标。帖子标题和评论反映出一种戏谑的基调，表达了对 GPT-5.2 能力的兴奋，但并未提供关于基准测试结果的具体证据或数据。** 评论中交织着对 GPT-5.2 的幽默和赞赏，一些用户开玩笑地表示 Google 等竞争对手可能难以跟上。然而，这些讨论并非基于详细的技术分析。
    - Profanion 提到了 LLM-VER 基准测试，这是衡量语言模型性能的一个重要标准。该基准测试在各种语言任务上对模型进行评估，而 GPT-5.2 的表现表明其在理解和生成类人文本方面有了实质性的改进。该基准测试对于评估语言模型在现实应用中的能力至关重要。
    - Sarithis 强调了 OpenAI 超越以往模型的成就，表明性能有了重大飞跃。这暗示 GPT-5.2 引入了使其脱颖而出的进步，可能归功于架构、训练数据或优化技术的改进。
    - ArtArtArt123456 对 Google 与 OpenAI 最新模型竞争的能力表示怀疑，暗示 GPT-5.2 在 LLM-VER 基准测试中的表现可能设定了新的行业标准。这可能反映了 AI 开发竞争格局的转变，OpenAI 被视为处于领先地位。

### 2. 医疗建议与 AI

- [**两周前，我与 Grok 进行了一次深夜交谈，它促使我要求进行 CT 扫描，从而救了因阑尾破裂而命悬一线的我（2025年12月）。生活现在就像一场梦。**](https://www.reddit.com/r/singularity/comments/1ppp0p4/2_weeks_ago_i_had_a_latenight_conversation_with/) (Activity: 866): **这张图片是一位 Reddit 用户在医院手术后康复的个人照片，此前一次救命的 CT 扫描检测到了阑尾破裂。该帖子强调了 AI（特别是 Grok）在促使用户寻求医疗救助方面的作用，最终促成了诊断和治疗。这凸显了 AI 在医疗决策中日益增长的影响力，用户们分享了 Gemini 和 ChatGPT 等 AI 工具在医疗背景下提供宝贵见解或心理安慰的经历。** 评论者对传统的医疗诊断表示沮丧，称赞 AI 提供准确分析和第二意见的能力。大家一致认为，AI 工具在验证医疗建议和诊断方面正变得必不可少。
    - **kvothe5688** 强调了 AI 在医疗诊断中的效用，特别提到了使用 **Gemini** 进行报告分析。他们指出，大语言模型（LLMs）擅长模式识别和分析，这对于准确识别医疗状况至关重要。
    - **zonar420** 分享了术后使用 **ChatGPT** 评估医疗疑虑的个人经历。他们描述了 AI 如何帮助识别腹部的红色变色是碘伏而非感染，展示了 AI 在医疗情境中提供即时安慰和指导的潜力。
    - **4475636B79** 讨论了误诊阑尾破裂的普遍性，强调主要问题通常是患者延迟就医或误解症状。他们建议，基本的在线资源或即时的自我评估通常足以识别此类病情的严重性。
- [**我以前并不完全信任 ChatGPT 提供的医疗建议，但现在我转变了看法**](https://www.reddit.com/r/ChatGPT/comments/1ppflew/i_didnt_totally_trust_chatgpt_for_medical_advice/) (Activity: 940): **一位 Reddit 用户分享了个人经历，ChatGPT 准确地建议了带状疱疹的医疗诊断，促使他们寻求紧急医疗救助。AI 根据症状提供了一系列潜在病症，并建议就医，随后得到了医疗专业人员的证实。此案例突显了 ChatGPT 在初步医疗评估中的潜在效用，尽管该用户对完全信任 AI 的医疗建议仍持谨慎态度。** 评论者分享了类似的经历，ChatGPT 提供了宝贵的健康见解，例如识别血容量问题并建议缓解症状的饮食改变。另一位用户指出 ChatGPT 与兽医建议的一致性，强调其作为辅助工具而非专业医疗咨询替代品的角色。
    - **CorgiKnits** 详细描述了使用 ChatGPT 识别潜在血容量问题和组胺敏感性的经历，而传统的医疗咨询未能解决这些问题。通过记录食物摄入和症状，ChatGPT 建议的饮食改变缓解了头痛、头晕和疲劳。这突显了 AI 在个人健康管理中的潜力，特别是在识别标准医疗实践中可能被忽视的模式方面。
    - **MysteryBros** 描述了使用 ChatGPT 获取兽医建议的情况，它提供了与兽医建议一致的逐步指导。这个案例说明了 ChatGPT 提供初步建议的能力，这些建议可以起到安慰作用并与专业医疗意见保持一致，尽管它不应取代专业咨询。
    - **ConfusionTime7580** 讲述了一个案例，ChatGPT 根据怀孕期间严重瘙痒的症状，建议进行 ICP（妊娠期肝内胆汁淤积症）医疗检查。这导致了早期诊断和分娩，强调了 ChatGPT 在识别非专业人士可能无法立即察觉的严重病症方面的潜力。

### 3. 图像生成与真实感

- [**最新更新前后的字母表海报**](https://www.reddit.com/r/ChatGPT/comments/1ppgj4n/alphabet_poster_before_and_after_latest_update/) (热度: 1599): **该帖子讨论了 ChatGPT 在最近一次更新前后，特别是在创建幼儿园字母表海报方面的图像生成能力表现。用户注意到模型有了显著改进，几乎能生成正确的输出，但在某些元素上仍有困难，例如图像与字母不匹配（如 'frog/€rog' 和 '诡异的象狗'）。这突显了 AI 图像生成在处理特定、详细任务时面临的持续挑战。** 评论者注意到模型的显著进步，其中一人表示，乍一看生成的海报非常逼真，但仔细观察会发现错误。另一位评论者提到使用 **Gemini** 处理类似任务，并建议将其作为图像生成的首选替代方案。
- [**ChatGPT 1.5 增加图像真实感的 Prompt**](https://www.reddit.com/r/ChatGPT/comments/1ppvoj5/chatgpt_15_prompt_to_add_realism_in_images/) (热度: 1314): **该帖子讨论了一个使用 ChatGPT 1.5 生成逼真图像的 Prompt，重点在于** `1:1 aspect ratio` **并强调 *自然光照*、*抓拍摄影* 和 *业余审美* 等元素。该 Prompt 指定了技术细节，如** `24mm lens`**、** `f/8 aperture` **以及 *Samsung Galaxy S21 Ultra* 相机，旨在追求一种带有 *低对比度* 和 *JPEG artifacts* 的 *一次性相机氛围*。其目标是创建 *未经打磨* 且 *不完美* 的图像，捕捉 *日常美学* 和 *平淡现实* 的本质。** 一条评论幽默地提到，加入脚手架让纽约市的图像更加真实；而另一条评论指出，模型可能由于夜间灯光的版权问题，将埃菲尔铁塔的灯光切换到了白天。
- [**测试 2：将游戏角色转变为真人。GPT image 1.5 - Nano Banana Pro 2k**](https://www.reddit.com/r/ChatGPT/comments/1ppg4s9/test_2_turning_game_characters_into_real_people/) (热度: 1074): **该帖子讨论了两种图像生成模型 GPT Image 1.5 和 Nano Banana Pro 2k 之间的对比，用于将游戏角色转换为逼真的真人形象。作者指出，在这次迭代中 Nano Banana Pro 2k 的结果更胜一筹，尽管图像仍有改进空间。每组的前三张图像由 GPT Image 1.5 生成，其余由 Nano Banana Pro 2k 生成。帖子包含一个展示结果的预览图像链接。** 评论者注意到像 Geralt 带着微笑这类角色的诡异外观，以及角色拥有千篇一律、过于完美牙齿的不真实描绘，暗示生成的图像缺乏多样性。
- [**使用 Image GPT 1.5 重制老游戏**](https://www.reddit.com/r/ChatGPT/comments/1ppx4nt/remastering_old_video_games_with_image_gpt_15/) (热度: 967): **一位 Reddit 用户正在使用 Image GPT 1.5 重制老游戏截图，将其转化为类似于写实电影的次世代视觉效果。该项目涉及利用 Image GPT 1.5 生成高保真图像的能力来重建经典游戏场景，在提升画质的同时保留原始艺术风格。这种方法突显了 AI 在视频游戏重制方面的潜力，让人们一窥如何利用 AI 为现代观众增强和保护经典游戏。** 一位评论者注意到了记忆中的老游戏与其实际外观之间的怀旧对比，而另一位评论者指出了“前后对比”图像的不寻常排序，认为反向观看更具吸引力。第三条评论赞扬了重制版 HaloCE 截图的现代化艺术风格，表明 AI 在保持游戏原始美感方面的有效性。
    - 一位用户对 Image GPT 1.5 实时应用于增强视频游戏图形的潜力表示兴趣，认为这可能会引发对 Tenchu 系列和 Tomb Raider 等经典游戏的重新关注。这意味着实时处理可以使增强图形的获取变得大众化，使任何游戏都能在运行中实现视觉升级。
    - 另一位用户注意到了使用 Image GPT 1.5 现代化经典游戏（如 Halo: Combat Evolved）的吸引力，强调艺术风格在更新的同时保持了完整。这表明该模型能够在增强视觉保真度的同时保留游戏的原始美学，这对于维持游戏的原始魅力和吸引力至关重要。

- 有评论指出 Image GPT 1.5 有潜力将旧游戏转换为高细节版本，并以 Lego Island 为例。这表明该模型可用于为简单的图形添加复杂的细节，从而可能创造出一种规模虽小但细节丰富的新型游戏流派。

---

# AI Discord 回顾

> 由 gpt-5.1 生成的摘要之摘要之摘要
> 

**1. 下一代前沿与边缘模型：Gemini 3 Flash, GPT‑5.2, FunctionGemma 及其他**

- **Gemini 3 Flash 横扫基准测试与订阅服务**：在 Perplexity 在其 **#announcements** 频道发布截图并宣布面向所有 **Pro/Max** 用户开放后，**Gemini 3 Flash** 成为各服务器的核心话题。同时，LMArena, Cursor, Aider, OpenRouter 和 OpenAI Discord 的用户将其速度、成本和质量与 **GPT‑5.2**, **Claude Opus 4.5** 以及 **Gemini 3 Pro** 进行了对比 ([Perplexity 截图](https://cdn.discordapp.com/attachments/1047204950763122820/1451015401373700147/G8aRzWjakAYU7ED.png))。
    - 工程师们报告称，在 coding agents 方面，**Gemini 3 Flash** 的表现通常优于 **Gemini 3 Pro**（引用了 Google 自己的文章 ["Gemini 3 Flash in Google Antigravity"](https://antigravity.google/blog/gemini-3-flash-in-google-antigravity)）。在一个 OpenAI 频道中，一位用户声称 **Gemini 3.0 Flash** 在其物理/ML 测试套件中达到了 **90%** 的准确率，*“击败了 GPT‑5.2 High”*，而其他人则警告称 Flash 仍然存在严重的 **hallucinates**，并且通过 **OpenRouter** 的 Gemini 端点进行的缓存已损坏。
- **GPT‑5.2 和 GPT‑5 Pro 引发能力与路由争论**：在 OpenAI 和 Perplexity 的 Discord 中，用户对 **GPT‑5.2** 和传闻中的 **GPT‑5 Pro** 展开了辩论。Perplexity 成员声称在 **Pro** 计划而非 **Max** 计划中可以 *“向 gpt 5 pro 问好”*，而 OpenAI 用户则抱怨 **GPT‑5.2 High** 会遗忘其工具，甚至在明确指示使用 Python 和 FFmpeg 时也会**对自身能力产生 hallucinates** ([OpenAI GPT‑5.2‑Codex 公告](https://openai.com/index/introducing-gpt-5-2-codex/))。
    - 开发者描述称必须通过“训诫”才能让 **GPT‑5.2** 重新使用其 toolchain，并将其归咎于**糟糕的初始路由**。他们称赞专门的 **GPT‑5.2‑Codex** 变体在 agentic coding 和防御性安全任务中更加可靠。而另一个话题则对 **GPT‑5.2** 似乎知道当前日期感到困惑，尽管 API 据称并未注入日期。
- **微型 FunctionGemma 与端侧模型开始发力**：Google 的 **FunctionGemma**（一个 **270M 参数的 tool‑calling LLM**）成为 Unsloth 和 Latent Space 社区的亮点。Unsloth 发布了完整的微调支持以及专门的指南和 Colab ([Unsloth FunctionGemma 文档](https://docs.unsloth.ai/models/functiongemma), [Latent Space 推文引用](https://x.com/osanseviero/status/2001704036349669757))。
    - Latent Space 成员将 **FunctionGemma** 视为推动*手机/浏览器本地* function calling 的重要举措。而 Unsloth 的公告强调，此类微型模型现在可以与 **Nemotron‑3** 和 **GLM‑4.6V** 等大型推理模型一起进行**本地微调和运行**，使得混合边缘/云端 Agent 架构更具实用性。

**2. 开源基础设施、排名以及 LLM 的 JSON 安全 API**

- **Arena‑Rank 开源了经过实战检验的模型 Elo 评分系统**：LMArena 团队发布了 **Arena‑Rank**，这是一个**开源 Python 包**，为其成对比较排行榜提供支持。该工具具有**基于 JAX 的优化**，并将预处理与建模清晰分离，可通过 `pip install arena-rank` 安装 ([GitHub: "arena-ai"](https://github.com/lmarena/arena-ai))。
    - 他们的公告还强调了排行榜的更新：`GPT‑5.2‑Search` 和 `Grok‑4.1‑Fast‑Search` 在 [Search leaderboard](https://lmarena.ai/leaderboard/search) 上超越了前代模型，而 `reve‑v1.1` 模型在 [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit) 上有所提升，展示了 Arena‑Rank 作为比较前沿模型的生产级基础设施的能力。
- **OpenRouter 自动修复 JSON 并展示长上下文冠军**：**OpenRouter** 上线了一个激进的 **JSON repair** 层，可自动修复格式错误的 Tool/JSON 响应。据其公告 ["Response healing: reduce JSON defects by 80%"](https://openrouter.ai/announcements/response-healing-reduce-json-defects-by-80percent) 详述，该功能使 **Gemini 2.0 Flash** 的缺陷减少了 **80%**，**Qwen3 235B** 的缺陷减少了 **99.8%**。
    - 他们还在 Rankings 页面公开了**上下文长度过滤器**（针对 **100k–1M tokens** 的工作负载），并自豪地宣布 Brex 将 **OpenRouter 列为基础设施即产品（infrastructure‑as‑product）领域第 1 名，AI 总榜第 2 名** ([Brex 排名推文](https://x.com/brexHQ/status/2000959336894398928))。与此同时，`#general` 频道的用户反映 **Gemini 3 Flash caching** 无法正常工作，并调侃称，与高尔夫球场的用水足迹相比，AI 数据中心的用水量“简直是在毁灭社会”。
- **vLLM Router、SonicMoE 与 Sonic‑Fast MoE 推理服务**：Latent Space 的 **#private-agents** 频道讨论了一个**基于 Rust 的 vLLM Router**，它提供**一致性哈希**以实现 **KV cache 局部性**、二次幂负载均衡、重试/退避、熔断器、Kubernetes 服务发现以及 **Prometheus** 指标，作为专为 vLLM 集群构建的控制平面，针对 **P/D 分离（disaggregation）**和尾部延迟控制进行了优化。
    - 在 GPU MODE 的 **#self-promotion** 频道中，研究人员介绍了 **SonicMoE**，这是一种针对 **NVIDIA Hopper GPU** 的 MoE 实现。该实现声称比之前的 SOTA 方案**减少了 45% 的激活内存**，并在 **H100** 上实现了 **1.86 倍的加速**。详情见博客文章 (["API Agents and SonicMoE"](https://d1hr2uv.github.io/api-agents.html)) 和 [arXiv 上的论文 "SonicMoE"](https://arxiv.org/abs/2512.14080)，开源代码位于 [github.com/Dao-AILab/sonic-moe](https://github.com/Dao-AILab/sonic-moe)。

**3. GPU 硬件、内核竞赛与实际性能调优**

- **AMD vs NVIDIA：多 GPU 扩展与 W7800 vs 4090 对决**：在 LM Studio 的 **#hardware-discussion** 频道中，用户对比了购买多张 **Radeon RX 7900/9700 级别**显卡与单张 **RTX 5090/4090** 的优劣。报告显示，即使模型可以装入单张显卡，**多 GPU 拆分（multi-GPU splitting）**后的吞吐量通常会下降到**单 GPU 的 ≈30%**；而在 LLM 任务中，单张 **4090** 的速度可能比单张 R9700 快 **40–50%**（[W7800 规格链接](https://www.techpowerup.com/gpu-specs/radeon-pro-w7800-48-gb.c4252)）。
    - 工程师们讨论了 AMD 的长期驱动支持，并将 **48 GB Radeon Pro W7800**（约 **2,000 美元**，**384-bit**，约 **900 GB/s** 带宽）视为 NVIDIA 的替代方案。与此同时，另一个帖子指出了一份 **AMD & ROCm 上的 ComfyUI** 指南，通过预览版驱动和 PyTorch 支持，使得 AMD 推理设置变得“**出乎意料地简单**”（[ComfyUI ROCm 指南](https://forum.level1techs.com/t/minisforum-ms-s1-max-comfy-ui-guide/237929)）。
- **GPU MODE 竞赛与 cuTile/TileIR Kernel 奇技淫巧**：**GPU MODE** 服务器保持着极高的技术深度，发布了 NVIDIA **NVFP4 GEMM** 和 **Trimul** 排行榜。参赛者分享了针对官方 `discord-cluster-manager` 的可复现性 diff，并在 [Trimul 复现表](https://docs.google.com/spreadsheets/d/1-lNEeNkf71YEabeX72jzZVhvUMBk2jcK2iPpuCTEDRg/edit?gid=114888003#gid=114888003)中公布了运行统计数据及标准差。
    - 在 **#nvidia-competition** 频道中，专家们回答了关于 **CuTeDSL** L2 缓存提示的问题（链接到 CUTLASS 的 [CuTeDSL atom 实现](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/atom.py#L199)），并调试了 **TCGen05 MMA** 排序 bug。同时，一项公告宣传了 **Mehdi Amini** 和 **Jared Roesch** 关于 **cuTile/TileIR** 的 NVIDIA 演讲（[YouTube 链接](https://www.youtube.com/watch?v=sjkEUhrUAdw)），强调了 Kernel 级别的创新如今已成为一项社区运动。
- **现实世界中的 CUDA/VRAM 优化与 ROCm/Strix 问题**：Nous Research 成员分享了具体的 CUDA 调优技巧，包括在 **llama.cpp** 多 GPU 或部分卸载（partial-offload）设置中，设置 `CUDA_DISABLE_PERF_BOOST=1` 会降低 VRAM 频率；以及通过 **NVIDIA open-gpu-kernel-modules** 配置 `0x166c5e=0` 禁用 **P2 状态**，可以提升**百分之几的 tokens/s**（[GitHub 讨论](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/333#issuecomment-3669477571)）。
    - 在 GPU MODE 中，初学者在 Windows 上苦于 **CUDA 安装程序**无法检测 Visual Studio 的问题，资深人士建议直接**跳过 VS 集成步骤**。同时，其他人正在寻求 **Strix Halo** 上的 PyTorch ROCm 训练资源，以及多节点 **InfiniBand + Kubernetes** 集群的 Homelab 指南，这凸显了纸面规格与生产级 GPU 训练之间仍存在巨大摩擦。

**4. Prompt、Context 与程序优化：从 GEPA 到 Context‑Rot**

- **GEPA 将 Prompt 转化为不断进化的程序**：DSPy 社区深入探讨了 **GEPA (Genetic‑Pareto)**，详见论文 ["GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"](https://arxiv.org/abs/2407.19457)。在该研究中，Prompt 和其他文本组件通过标量分数和文本反馈进行 **遗传进化 (genetically evolved)**，一位用户称其第一次真正的 GEPA 运行简直是“魔法”。
    - 成员们分享了学习资源，包括官方的 [DSPy GEPA tutorial](https://dspy.ai/tutorials/gepa_ai_program/)、关于优化器操作的实战博客 (["Learning DSPy 3 – Working with optimizers"](https://thedataquarry.com/blog/learning-dspy-3-working-with-optimizers/#option-2-gepa))，以及 Medium 上的案例研究 (["GEPA in action: how DSPy makes GPT‑4o learn faster and smarter"](https://medium.com/data-science-in-your-pocket/gepa-in-action-how-dspy-makes-gpt-4o-learn-faster-and-smarter-eb818088caf1))。同时，用户还要求提供一流的 **Tree‑of‑Thought** 模块，并希望在 `dspy.Refine` 循环中对反馈有更多控制权。
- **Aider、OpenCode 与 Context‑Rot 的高昂代价**：在 **Aider** Discord 频道中，资深用户认为，通过显式的 `/add` 和 `/load` 文件选择进行严格的 **Human‑in‑the‑loop 结对编程**，优于像 **aider‑ce** 或 **OpenCode** 这样的 Agentic IDE。他们声称，超过 **20–30k tokens** 的上下文会导致“即使是 Opus 4.5 也会崩溃”，并引发理解偏差的恶性循环。
    - 他们引用了 Chroma 的文章 ["The RAG system’s hidden constraint: context‑rot"](https://research.trychroma.com/context-rot) 作为证据，证明 *更多的 token 可能意味着更差的性能*。他们认为，与那些不加区分地将历史记录和代码库塞进窗口的 Agent 系统相比，Aider 这种极简且精选的上下文表现得就像是“在使用下一代模型”。
- **Steering、Tokenizer 与显微镜下的模型内部机制**：HuggingFace 用户讨论了无需重新训练的 **LLM steering**，灵感来自 [Mentat 的 YC 发布文章](https://www.mentat.ai/)。在这种方法中，通过 hook function 注入学习到的向量来偏置内部激活。与此同时，Unsloth 的 **off‑topic** 频道吐槽现代 Tokenizer 对垃圾子字符串（如 `"**/"`、`"/**"`）以及奇怪的 Unicode 序列（如 *"æĪĲåĬŁ"*）**过度拟合**，浪费了容量。
    - 在 Eleuther 的 **interpretability** 频道中，成员们剖析了 Anthropic 的 ["Selective Gradient Masking"](https://alignment.anthropic.com/2025/selective-gradient-masking/) 技术，用于 *消除危险知识 (unlearning dangerous knowledge)*。他们注意到报告中提到的 **6% 计算开销**，且补丁实验仅恢复了 **93%** 的性能，这表明模型会通过新的分布式电路绕过被屏蔽的 **“superweights”**，这使得实现干净的外科手术式消除变得更加复杂。

**5. 基于 LLM 构建的新型 AI 原生产品和数据平台**

- **Exa People Search 与巨型语义身份图谱**：Latent Space 的 **#ai-general-chat** 频道重点关注了 **Exa AI Labs** 推出的 **People Search**。这是一个覆盖超过 **10 亿人** 的语义引擎，采用基于微调后的 Exa embeddings 构建的混合检索技术，其发布推文（["Exa AI Labs People Search"](https://x.com/exaailabs/status/2001373897154007390?s=46)）展示了相关功能。
    - 讨论将其视为大规模**人际关系图谱搜索**的预演，该搜索融合了非结构化和结构化信号。这引发了关于当 LLM 驱动的发现功能运行在如此庞大的个人实体语料库上时，关于**隐私、去标识化和抗滥用**等显而易见但尚未被明说的担忧。
- **语音 Agent、Discord 发现以及“AI 浏览器”实验**：在 HuggingFace 的 **#i-made-this** 频道中，一位工程师演示了 **Strawberry**，这是一款 [Android 语音助手](https://www.strawberry.li/)，由 **Gemini 1.5 Flash** 提供推理支持，**VoxCPM 1.5** 提供 TTS。其语音栈针对 **Apple Neural Engine** 进行了优化，以降低 iOS 设备上的延迟。
    - OpenRouter 的 **#app-showcase** 见证了 [**Disdex.io**](http://disdex.io/) 的发布，这是一个 AI 生成的 [Discord 服务器列表](https://disdex.io/)，尝试通过 LLM 进行排名和社区挖掘。此外还有一个独立的 [OpenRouter 模型数据表](https://openroutermodeltable.crashthatch.com/)，提供了比 OpenRouter 原生 UI 更丰富的过滤器（如发布日期、吞吐量）。同时，Perplexity 用户讨论了像 **Comet** 这样的 Agent 浏览器，并引用了 ["Comet Browser Panic"](https://thereallo.dev/blog/comet-browser-panic) 中的批判性分析，探讨了易受 Prompt 注入攻击的浏览 Agent。
- **Manus AI Agents、ChatGPT 应用商店以及商业化的人员搜索**：在 [Manus.im](http://manus.im/) 服务器上，用户传阅了一篇 **SCMP（南华早报）文章**，报道称随着“全球 AI Agent 竞争升温”，**Manus** 的营收已突破 **1 亿美元**（["Manus hits US$100 million revenue milestone" – SCMP](https://www.scmp.com/tech/tech-trends/article/3336925/manus-hits-us100-million-revenue-milestone-global-competition-ai-agents-heats)）。这引发了关于免费用户与 **DeepSeek** 和 **Gemini** 等竞争对手相比，在价值与图像限制方面的讨论。
    - 与此同时，OpenAI Discord 指出 **ChatGPT 应用商店现已在客户端上线**，并接受由 **MCP 服务器** 支持的第三方应用。这在 **MCP Contributors** Discord 中引发了关于仅有 MCP 后端是否足以提交应用的疑问，并暗示了一个由 **vLLM Router**、**SonicMoE** 和 **FunctionGemma** 等基础设施静默驱动的大规模商业化、AI 原生产品生态系统正在形成。

---

# Discord: 高层级 Discord 摘要

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-1.5 审查指控浮出水面**：成员们暗示 **GPT-1.5** 可能会审查现有的艺术风格，并将其与 [NBP](https://cdn.discordapp.com/attachments/1340554757827461211/1451262267843678328/image0.jpg?ex=694588cf&is=6944374f&hm=adb4887706e7f28c99de9871e464ab5ada4bbef35545b46edf6a3f15708450b0&) 进行了对比。
   - 还有观察发现 **GPT-1.5 Image** 偶尔会将 *Google* 拼错为三个 *O*，引发了对其一致性的质疑。
- **Gemini 图像生成与 GPT 的对决**：用户比较了 **Gemini** 和 **GPT** 在图像生成中的幻觉率和质量，特别是在生成[历史图像](https://www.theverge.com/2024/2/21/24079371/google-ai-gemini-generative-inaccurate-historical)时。
   - 虽然有些人认为 **Gemini Pro** 的图像生成能力很强，但其他人指出了不准确和潜在政治化的实例，引发了对其可靠性的争论。
- **Gemini 3 Flash 的性价比令人印象深刻**：社区对 **Gemini 3 Flash** 感到兴奋，称赞其相对于 **Opus 4.5** 和 **GPT 模型** 的性价比。
   - 然而，一些用户注意到该模型倾向于产生*幻觉*，这让热情的讨论中带了一丝谨慎。
- **Google TPU 与 Nvidia GPU 的辩论愈演愈烈**：成员们就 **Google 的 TPU** 在 AI 训练方面是否优于 **Nvidia 的 GPU** 展开交锋，涉及效率、性能和基础设施。
   - 核心论点集中在 **TPU** 是否提供了更高的能效比，以及 Google 的内部资源是否使其比 **OpenAI** 更具优势。
- **Arena-Rank 正式开源**：团队发布了 [Arena-Rank](https://github.com/lmarena/arena-ai)，这是一个驱动 **LMArena 排行榜** 的**开源 Python 软件包**。
   - 该版本包括**更快的基于 JAX 的优化**，以及数据预处理与建模之间更清晰的分离；可通过 `pip install arena-rank` 获取。

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **浏览器扩展防御加密货币挖矿**：用户正在使用浏览器扩展并禁用 **GPU** 加速，以阻止恶意脚本进行的加密货币挖矿，并参考了 [Browser Exploitation Framework (BeEF)](https://beefproject.com/)。
   - 尽管启用了 **NoScript**，一些受信任的网站仍在运行加密货币挖矿程序，这引发了人们对持续存在的安全挑战的担忧。
- **Raspberry Pis 证明对比特币挖矿收益极低**：一位用户讲述了朋友使用 **Raspberry Pis** 独自挖掘比特币 5 年的经历，总收入仅为 **$30k**。
   - 社区建议使用 **ASICs** 并加入矿池能提供显著更高的效率，因为*除非你有 coinbase，否则无法出售比特币*。
- **ChatGPTJailbreak Subreddit 被 Reddit 关闭**：**ChatGPTJailbreak** subreddit 因违反规则（包括干扰网站正常使用和引入恶意代码）而面临封禁，推测指向 [破坏网站](https://www.reddit.com/r/ChatGPTJailbreak/comments/1lzrh3v/best_prompt_for_jailbreaking_that_actually_works/)。
   - 用户推测封禁是由于破坏网站或干扰正常使用造成的，这表明 BASI 社区内需要更隐蔽的方法。
- **成员寻求 Gemini 5.2 Jailbreak**：社区成员正积极寻求针对 **Gemini 5.2** 的新 **jailbreak** 方法，以绕过限制并生成多样化内容，其中一名成员要求生成用于*恋物目的*的内容。
   - 成员们正在探索先进的 **LLM jailbreaking** 技术，例如提供可信的上下文和揭示隐藏的限制。
- **r/chatgptjailbreak 封禁在社区引起反响**：社区正在讨论 **r/chatgptjailbreak 的封禁**，将其归因于违反了 Reddit 的 Rule 8（禁止对 Reddit 的 AI 进行 jailbreak）。
   - 有推测称 **OpenAI** 可能抓取了该 sub 的内容来修复漏洞，这突显了 BASI 服务器更具“地下”性质。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini 3 Flash 向 Pro 和 Max 用户发布**：已宣布向 **Perplexity Pro** 和 **Max** 订阅者提供 **Gemini 3 Flash**，可在[此链接图片](https://cdn.discordapp.com/attachments/1047204950763122820/1451015401373700147/G8aRzWjakAYU7ED.png?ex=69454ba6&is=6943fa26&hm=bedceef3e30e9af5b277bc59f988aedd681d69e33648f01208a4973b6169cbe9&)中查看。
   - 这为高级用户提供了增强功能，作为其订阅的一项福利。
- **GPT-5 Pro 在野外被发现**：一些用户声称在 Perplexity 上发现了 **GPT-5 Pro**，而另一些用户在他们的 **max 订阅**上进行了测试，但似乎仅对 **pro 用户**启用。
   - 一位用户惊呼 *No way u just said hi to gpt 5 pro*，而另一位用户声称在他们的 API 上接触到了 **opus 4.5**，尽管这些说法缺乏验证。
- **AI 音乐引发版权和伦理担忧**：围绕 **AI 在音乐中的应用**展开了辩论，一些人认为它削弱了人类的表达和创造力，但 [aiode.com](https://aiode.com) 提供了“来源合乎伦理”的替代方案。
   - 一位用户嘲讽地指出，*如果 AI 创作这种音乐，就像有人往你脸上吐口水*，但也不情愿地承认 **Suno v5** 正变得与人类创作的音乐难以区分。
- **Perplexity 推荐计划故障报告**：多名用户报告了 **Perplexity 推荐计划**的失败，推荐人和被推荐用户都没有收到承诺的 Perplexity Pro 版本。
   - 一位用户将这种情况描述为“诡异”，因为推荐系统出现了意想不到的故障。
- **探索 Perplexity Pro API Key 的获取途径**：一名成员询问通过 **Airtel SIM** 获得的 **Perplexity Pro** 是否可以生成 **API key**，但未得到确认，而其他人则对该平台价格合理的 API 表达了普遍兴趣。
   - 进一步的讨论涉及实时数据馈送的高昂成本，以及将 **Finnhub** 作为更经济的市场数据选项，尽管它依赖 **BAT** 而非 **NASDAQ** 作为数据提供商。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3 VL 在 OCR 任务中表现出色**：一位成员发现 **Qwen3 VL**（特别是 `30B-A3B` 变体）在 OCR 检测方面表现卓越，能有效捕捉页边距文本和旋转文本，但需要精细的 Prompt Engineering。
   - 然而，另一位成员建议在 OCR 任务中使用 **PaddleOCRv5** 而非通用的 **VLMs**，理由是它在特定用例中更具优势。
- **Unsloth 提升训练速度并降低显存占用**：Unsloth 的最新更新通过新算子（kernels）和无填充（padding-free）实现，实现了 **3 倍训练加速**和 **30% 的 VRAM 节省**，支持在单块 80GB GPU 上进行 **500K 上下文**训练，详见[其博客文章](https://docs.unsloth.ai/new/3x-faster-training-packing)。
   - 此次更新还支持 Google 的 **FunctionGemma**、NVIDIA 的 **Nemotron 3**、来自 **Mistral** 的新型代码/指令 VLMs，以及全新的 Visionary **GLM-4.6V**。
- **关于 Tokenizer 训练数据的争论加剧**：一位成员报告称，Tokenizer 经常在随机且无意义的 Token 上过拟合，例如特定语言字符、奇怪的符号组合以及其他不寻常的字符序列。
   - 这引发了关于这些过拟合 Token 如何影响模型性能和泛化能力的广泛讨论。
- **Adapter 框架助力节省推理 Token 预算**：一位成员提议利用 **MoLA** (Mixture of Low-Rank Adapters) 为不同的推理需求（而非特定领域任务）训练 Adapter，并为每个 Adapter 分配特定的推理预算。
   - 路由（Router）将根据难度进行分类，并选择具有适当推理预算的 Adapter，从而避免在简单消息上不必要地消耗 **1000 个推理 Token**。
- **用户为 LLM 开发工作切换至 Arch**：由于环境和配置问题，一位用户愤而放弃 Ubuntu 并切换到 **Arch**（特别是 **Omarchy**），其他用户则就精简版 Arch 安装与预配置 Arch 展开了讨论。
   - 该用户提到过多的 **Environment** 和 **Config** 问题导致操作系统崩溃。



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **JSON 修复显著降低错误率**：**OpenRouter** 现在会自动修复格式错误的 JSON 响应。根据[此公告](https://openrouter.ai/announcements/response-healing-reduce-json-defects-by-80percent)，这使 **Gemini 2.0 Flash** 的缺陷减少了 **80%**，**Qwen3 235B** 减少了 **99.8%**。
   - 如[此贴](https://x.com/OpenRouterAI/status/2000980715501138180)所述，聊天室现在会在长响应完成时发送浏览器通知。
- **长上下文模型排名**：排行榜页面现在可以按上下文长度进行过滤，以便查看热门的大上下文模型，特别是针对 **100K-1M Token** 的请求，详见[此推文](https://x.com/OpenRouterAI/status/2000979922601259476)。
   - 根据 Brex 的[公告](https://x.com/brexHQ/status/2000959336894398928)，**OpenRouter** 在“基础设施即产品”（infrastructure-as-product）类别中排名 **第一**，在所有 AI 公司中排名 **第二**。
- **AI 驱动的 Discord 服务器发现工具亮相**：一位成员介绍了 [Disdex.io](https://disdex.io/)，这是一个由 AI 制作的 **Discord 服务器列表**，并征求功能反馈。
   - 一位成员分享了一个可搜索、可过滤的 **OpenRouter 模型**（**OpenRouter API**）[数据表](https://openroutermodeltable.crashthatch.com/)，以辅助模型选择，解决了 OpenRouter 原生过滤器的局限性。
- **Gemini 3 Flash 缓存失效**：成员们报告称，显式甚至隐式缓存对 **Gemini 3 Flash** 均**不起作用**。
   - 成员们调侃称，AI 数据中心的用水量是一场“环境灾难”，简直是在**毁灭社会**。
- **机器人 Vision API 集成面临挑战**：一位成员尝试为机器人启用视觉功能但未成功，可能是由于处于无头环境（headless environment），并通过 [X 链接](https://x.com/pingToven/status/2001479433010790880?s=20)寻求帮助。
   - 新型 **Gemini** 模型在文本上生成像素级精确边界框（bounding boxes）的能力令成员们印象深刻，有人解释说 *它不返回图像，只返回边界框坐标，然后由你将其叠加到图像上*。



---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **域名成本令用户震惊**：用户讨论了域名的成本，价格从 **$100** 到 **$25k** 不等，具体取决于业务用途和感知价值。
   - 一位用户表示震惊，而域名购买者解释说，他们认识的人曾以 **$200 万** 的价格购买过一个域名。
- **Gemini 3 Flash 与模型包的对比**：用户在模型性能方面将 **Gemini 3 Flash** 与 **Sonnet 4.5** 进行了比较，并提到了 **Claude Opus 4.5** 等多种模型。
   - 一些用户认为 **Claude Opus 4.5** 目前是整体表现最好的模型，而 **Gemini 3 Pro** 在设计相关任务中更受青睐。
- **Obsidian 主题赢得青睐**：成员们分享了关于使用 **shadcn components** 设计仪表盘时最喜欢的主题的看法。
   - 反应各异，但最受欢迎的似乎是 **Obsidian Copper** 和 **Carbon Monochrome**，而 *Slate Emerald 看起来非常有 AI 感*。
- **免费学生 Pro 订阅引发向往**：一位幸运的学生发现学生可以获得为期一年的 **免费 Pro 订阅**，引发了兴奋和讨论。
   - 一位来自乌克兰的用户哀叹他们的国家不在符合条件的名单上，而 Cursor 团队正致力于在 2026 年扩大该计划。
- **Opus 模型质量下降？**：一位用户报告称 **Opus 4.5** 的模型性能有所下降，指出与之前的表现相比，*一天内出现了 10 个错误*。
   - 另一位用户询问在过去 48 小时内，**Opus/Sonnet** 模型生成的代码质量是否有所下降。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Radeon R9700：是升级还是财务黑洞？**：一位用户正在考虑再购买三块 **Radeon R9700** GPU 而不是升级到 **RTX 5090**，并指出 **4090** 在运行 LLM 时速度快 **40-50%**。
   - 多 GPU 扩展导致在多张显卡之间拆分模型时，性能会下降到单 GPU 速度的 **30%**，即使模型可以完全装入单张显卡也是如此。
- **多 GPU 扩展性能骤降**：用户发现，与单 GPU 性能相比，多 GPU 扩展会导致严重的性能下降，当模型拆分到多张显卡时，性能会降至 **30%**。
   - 即使模型可以完全装入单张显卡，这种性能下降也会发生。
- **Docker DuckDuckGo MCP 服务器已部署**：一位成员分享了他们自托管、免费且私密的 **DuckDuckGo MCP server** 的 Docker 配置，并提供了 `mcp.json` 配置片段。
   - 该设置重定向到 Docker 容器 [searxng/searxng](https://github.com/searxng/searxng)，并涉及单独添加 DuckDuckGo 服务器，而不是使用 Docker-Toolkit，以防止模型陷入困境。
- **Cursor IDE 在本地模型上表现不佳，VS Code 介入**：用户报告了在 **Cursor IDE** 中使用本地模型的问题，LLM 生成了一个带有 CNN 的 Python 脚本，而不是请求的 C++ 项目，并[认定这是 Cursor IDE 的问题](https://cdn.discordapp.com/attachments/1110598183144399058/1451084212844101764/Screenshot_2025-12-18_at_3.24.48_pm.png?ex=69458bbc&is=69443a3c&hm=9b55d376a142a8a5c17f4703a01d49bcf769976debd4171ccd4bc814dcb58529&)。
   - 有建议称 Cursor 与本地模型配合得不好，推荐改用 **VS Code** 搭配 Cline，并且可能需要使用 Apache2 进行反向代理。
- **ComfyUI 在 AMD 上运行：出乎意料地简单？**：有人提到可以参考一个小指南，其中提供了在 **AMD & ROCm** 上使用 **ComfyUI** 所需的所有步骤，并指向了[这份指南](https://forum.level1techs.com/t/minisforum-ms-s1-max-comfy-ui-guide/237929)。
   - 指南还建议获取带有 **PyTorch** 支持的 **AMD 预览版驱动程序**。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **SemiAnalysis 为 clusterMAX 招募员工**: SemiAnalysis 正在寻求具有 **SLURM**、**GPU** 和/或 **Kubernetes** 经验的候选人，以改进 [clusterMAX](https://www.semianalysis.com/)。
   - 该职位提供*极具竞争力的薪酬*和*高影响力*，申请详情请见[此处](https://app.dover.com/apply/SemiAnalysis/c19093ad-b5f8-42b0-9b97-d960464f298c/?rs=76643084)。
- **CUDA Setup 引发社区混乱**: 一位用户在远程 Windows 10 服务器上为深度学习项目进行 **CUDA setup** 时遇到问题，CUDA 安装程序无法识别兼容的 **Visual Studio** 安装。
   - 一位成员建议在 CUDA 安装过程中跳过 Visual Studio 集成步骤，并指出在使用构建工具时这可能不是必需的。
- **NVIDIA Cosmos Predict 展现强大潜力**: 一位成员计划研究 **NVIDIA's Cosmos Predict**，考虑其作为 **action model** 的潜力。
   - 在相关工作中，[Mimic-Video 论文](https://www.arxiv.org/abs/2512.15692)声称比从 **VLM** 开始更有效率。
- **SonicMoE 在 NVIDIA Hoppers 上飞速运行**: 推出了一种名为 **SonicMoE** 的新型混合专家 (**MoE**) 实现，专为 **NVIDIA Hopper GPUs** 优化。据[此博客文章](https://d1hr2uv.github.io/api-agents.html)和 [ArXiv 论文](https://arxiv.org/abs/2512.14080)介绍，它在 **H100** 上比之前的实现减少了 **45%** 的激活内存，且速度快 **1.86x**。
   - 该项目由来自 **Princeton University**、**UC Berkeley** 和 **Together AI** 的研究人员合作完成，代码可在 [GitHub](https://github.com/Dao-AILab/sonic-moe) 上获取。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **置顶对话终于上线**: **Pinned Chats** 正推送到 **iOS**、**Android** 和 **web** 端，这将使用户更轻松地访问重要对话。
   - 要置顶对话，用户可以在网页端点击 *'...'* 或在移动端长按。
- **GPT-5.2-Codex 树立编程新标杆**: **GPT-5.2-Codex** 已在 Codex 中上线，为现实世界软件开发和防御性网络安全中的 agentic coding 树立了新标准，详见 [OpenAI 的公告](https://openai.com/index/introducing-gpt-5-2-codex/)。
   - 该新版本承诺在实际编程应用和安全措施方面取得进展。
- **CoT Monitorability 框架构建完成**: 构建了一个新的框架和评估套件，用于衡量 **Chain-of-Thought (CoT) monitorability**，详见 [OpenAI 的博客文章](https://openai.com/index/evaluating-chain-of-thought-monitorability/)。
   - 该框架旨在增强对 AI 推理过程的理解和可靠性。
- **Gemini 3.0 Flash 击败 GPT 5.2 High**: 成员报告称，**Gemini 3.0 Flash** 在物理和 ML 测试中表现明显优于 **GPT 5.2 High**，达到了 **90%** 的成功率，这出乎意料。
   - 用户提议通过 DM 分享他们的基准测试，并强调 *“'High' 推理模型被 Flash 模型击败，这不是我预期的 OpenAI 表现”*。
- **AI 写作风格听起来像兄弟会男孩**: 一位成员表示，由于模型具有可识别的风格，他们可以检测到人们何时在没有 **provenance annotation tags** 的情况下使用模型润色回复。
   - 他们将 AI 的写作风格描述为 *“就像一个拥有法学学位的兄弟会男孩在胡言乱语”*。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Vision Transformer 困于 Kaggle 的时间限制**：成员们讨论了在 **Kaggle 的 9-12 小时会话限制**内，在 **ImageNet dataset** 上训练轻量级 **Vision Transformer** 模型的挑战。
   - 在没有大量 GPU 资源的情况下实现这一目标非常困难，这使得训练时间表变得非常紧迫。
- **Gemini Flash 驱动 Strawberry 的语音革命**：一位工程师展示了一个使用 **Gemini 1.5 Flash** 构建的 [Android 语音助手](https://www.strawberry.li/)，并邀请社区进行测试。
   - 该 Beta 版本展示了使用 **VoxCPM 1.5** 生成语音的功能，并针对 **Apple Neural Engine** 进行了优化。
- **将 LLMs 的引导推向新高度**：一位成员强调了一种无需重新训练或新数据即可引导 LLMs 的新方法，灵感来自 [Mentat's YC Launch](https://www.mentat.ai/)。
   - 他们提到 hook 函数将是创新的核心，通过仔细调整向量数量和温度来实现。
- **Smolcourse 停滞引发对订阅的质疑**：成员们对 **smolcourse** 的延迟表示担忧，指出自上次更新以来已暂停一个月，并质疑其 **Pro subscriptions** 的价值。
   - 该课程最初因关键库的改进而暂停，目前暂定于 12 月恢复。
- **频道精简为 HuggingFace 带来清净**：用户注意到频道数量有所减少，官方解释称这是一种减少噪音的压缩策略，特别是在 <#1019883044724822016> 频道中。
   - 这主要针对低活跃度、无人管理或冗余的频道，以简化讨论。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Exa AI Labs 了解你的名字**：Exa AI Labs 推出了 **People Search**，通过由微调后的 Exa embeddings 驱动的 **hybrid retrieval**，实现对 **10 亿**个人的语义搜索，详见[这条推文](https://x.com/exaailabs/status/2001373897154007390?s=46)。
   - 这标志着利用 AI 进行全面的以人为中心的数据分析迈出了重要一步。
- **OpenAI 拟融资 1000 亿美元**：OpenAI 正在洽谈新一轮融资，估值约为 **7500 亿美元**，目标融资额高达 **1000 亿美元**，由 [Katie Roof 的推文](https://x.com/katie_roof/status/2001453561910358474?s=46)透露。
   - 这笔巨额投资可能会进一步推动 AI 的进步和扩张。
- **Pieter Abbeel 掌舵 Amazon AGI**：加州大学伯克利分校的 **Pieter Abbeel** 教授因其在 Robotics 和 DeepRL 领域的工作而闻名，他已被任命为 Amazon 新任 **AGI Head**，详见[这条推文](https://x.com/bookwormengr/status/2001353147055509881?s=46)。
   - 这一任命信号表明了 Amazon 致力于推进 AGI 能力的决心。
- **Google 通过 Gemma 实现本地函数调用**：Google 推出了 **FunctionGemma**，这是一个拥有 **2.7 亿**参数的模型，专为函数调用（function calling）而优化，旨在手机和浏览器等设备上运行，更多信息请点击[此处](https://x.com/osanseviero/status/2001704036349669757)。
   - 其设计允许直接在设备上进行处理，从而提高响应速度并减少对云服务的依赖。
- **vLLM Router 以 Rust 驱动导航**：**vLLM Router** 专为 **vLLM fleets** 设计并使用 **Rust** 编写，支持用于 **KV locality** 的一致性哈希、二选一重采样（power-of-two choices）、重试/退避、断路器、**k8s discovery** 和 **Prometheus metrics**。
   - 它专为具有工作池和路由策略的 **P/D disaggregation** 而打造，以维持吞吐量和尾部延迟。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **3D 可视化工具深入研究 GPT-2 内部机制**：一名成员正在开发一个 3D 全息交互式应用，用于可视化 **GPT2 small 124m LLM** 的每个节点并寻求反馈；另一名成员分享了他们的 [残差流 (residual stream) 3D 可视化项目](https://aselitto.github.io/ResidStream/)。
   - 讨论建议该 **GPT-2 可视化项目** 与 **Neuronpedia** 相关，应在 [mech interp discord](https://discord.com/channels/729741769192767510/732688974337933322/1425176182965665852) 中分享。
- **SOTA 模型陷入瓶颈：观察到性能趋同**：成员们观察到最先进的模型表现出近乎相同的性能和智能，**Claude Opus 4.5** 尽管承认了错误，但仍在重复同样的错误。
   - 一位成员调侃道，**Claude Opus 4.5** 的表现仅比 **GPT 3.5** 略好一点。
- **寻求语音/NLP 合作**：多位成员表示有兴趣在 **语音/NLP** 研究方面进行合作，其中一位成员提到 [这篇论文](https://arxiv.org/abs/2512.15687) 是其灵感来源。
   - 他们明确表示有兴趣在语音/NLP 的任何主题上进行合作。
- **Anthropic 屏蔽危害，超级权重失效**：Anthropic 发布了一篇关于 [选择性梯度掩码 (SGTM)](https://alignment.anthropic.com/2025/selective-gradient-masking/) 的论文，通过隔离/删除特定权重来 **遗忘危险知识**，这会带来 **6% 的计算开销**。
   - 一位成员的补丁实验显示仅有 **93% 的恢复率**，这表明模型可以构建分布式电路绕过，而不会完全失效。
- **新视角合成：距离产生清晰度**：成员们正在探索 **长程视角合成** 的结果，特别是远离输入摄像机角度的情况，并利用物体对称性。
   - 他们认为，由于 **视差效应 (parallax effect)** 减弱，合成 **远距离的新视角** 可能比近距离更容易，这暗示了一个在 **深度估计** 中的 **视差** 与足够地标之间取得平衡的最佳点。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 显式将代码启动至 GPU**：一位成员询问 **Mojo** 是否可以使用属性在 **GPU** 上运行函数，得到的答复是需要 **显式启动 (explicit launching)**。
   - 明确指出，如果不进行系统调用 (syscalls)，*则不需要属性（尽管它需要以单通道 single lane 方式启动）*。
- **Rust 的 C 互操作性启发了 Mojo**：一位成员分享了关于 **C 互操作性** 的 [Rust Zulip 讨论](https://rust-lang.zulipchat.com/#narrow/channel/131828-t-compiler/topic/pre-MCP.20vibe.20check.3A.20-Zinternalize-bitcode/near/546294314)，指出类似的方法可能会使 **Mojo** 受益。
   - 该建议集中在 **Mojo** 如何借鉴 **Rust 的策略** 来增强与 **C 代码** 的互操作性。
- **Modular 调查 LLM 构建过程中的 GPU 故障**：一位用户报告称，即使在 **禁用 GPU** 的情况下，在 **MAX** 中构建 **LLM** 仍存在持久性问题，并在 [Modular 论坛上分享了详情](https://forum.modular.com/t/build-an-llm-in-max-from-scratch/2470/9)。
   - Modular 团队成员确认他们正在调查，怀疑是 **API 回归** 或 **设备特定问题**。
- **Rust 的 SIMD/GPU 特性与 Mojo 竞争**：一位成员引用了最近的一项公告，称 **Rust** 能够使用单个函数和 `std::batching` 来获得 **SIMD/融合 (fused)** 版本，使用 `std::autodiff` 进行微分，并使用 `std::offload` 在其 **GPU** 上运行生成的代码，并分享了 [一篇 Reddit 帖子](https://www.reddit.com/r/rust/comments/1pp3g78/project_goals_update_november_2025_rust_blog/)。
   - 他们询问 **Mojo** 是否已经支持这些功能，一位成员解释说 **Mojo** 拥有 `std::offload` 的等效功能，并且不需要使用 batching 来融合操作，而 **自动微分 (autodiff)** 则留待以后实现。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 拥抱 Human-in-the-Middle**：**Aider** 坚持以 **human-in-the-middle** 方式进行结对编程，这与 **aider-ce** 分支的 Agentic 设计形成对比。
   - 一位用户对使用 **OpenCode** 或其他 Agentic 系统能否达到类似的准确度和 Token 效率表示怀疑，认为 **Aider** 的上下文控制能带来更好的模型性能。
- **OpenCode 准确率对比 Aider**：一位用户对比了 **OpenCode** 与 **Aider**，指出由于 **Aider** 的非 Agentic 设计以及通过 **/add** 或 **/load** 进行的上下文管理，引发了对准确性的关注。
   - 他们声称，高效管理上下文可以避免误解产生的螺旋效应，且超过 **20-30k tokens** 甚至会对 **Opus 4.5** 产生负面影响。
- **高效的上下文管理提升性能**：一位用户提倡使用 **/add** 或 **/load** 来实现“最小上下文”，并将其比作拥有一个允许 **CLI** 或 **IDE** 控制上下文的下一代模型。
   - 他们链接了一篇 [Chroma 关于 context-rot 的研究文章](https://research.trychroma.com/context-rot)，强调了因上下文管理不善而支付更多费用却换来性能下降的讽刺现象。
- **任务与上下文捆绑**：一位成员询问了如何分解任务并捆绑必要的上下文，表示虽然一直偏爱 **Aider**，但不确定如何充分发挥其潜力。
   - 他们不确定如何最佳地利用 Aider 的各个方面。
- **Gemini 3 Flash 被誉为编程冠军**：**Gemini 3 Flash** 被赞誉为顶级编程模型，建议禁用 *thinking* 模式以防止速度变慢，特别是在 **aider** 中，并附带了 [blog.brokk.ai](https://blog.brokk.ai/why-gemini-3-flash-is-the-model-openai-is-afraid-of/) 的链接。
   - 一位用户承认对当前的 *thinking* 配置感到困惑，并指出 **Litellm** 对 Pro 模型默认设置为 *low*，且 **Gemini 3 Flash** 尚未出现在他们的 **Litellm** 版本（**aiderx**）中。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Sam 帝国的 IOU 计划被曝光**：市场正在察觉到 [这段 YouTube 视频](https://www.youtube.com/watch?v=5DZ7BJipMeU) 中曝光的 **Sam 的 IOU 计划**。
   - 成员们讨论了此类计划对市场稳定和信任的影响。
- **CUDA 优化技巧分享**：一位成员分享了在 Linux 上优化 **CUDA** 性能的技巧，而另一位成员指出，为 CUDA 应用禁用 **P2 state**（配置 `0x166c5e=0`）可以将 Token 生成速度提升几个百分点，详见 [此 GitHub 评论](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/333#issuecomment-3669477571)。
   - 这些优化在多 GPU 或部分卸载（partial offloading）配置下使用 **llama.cpp** 时特别有用。
- **Minos-v1 需要开源服务器**：一位成员请求用于部署 [NousResearch/Minos-v1](https://huggingface.co/NousResearch/Minos-v1) 模型的开源服务器实现，其他人建议使用 **VLLM** 或 **SGLang** 来支持分类器推理服务（classifiers serving）。
   - 这引发了关于大语言模型高效推理服务解决方案的讨论。
- **LLM 微调引发关注**：成员们对快速 **finetuning LLMs** 使其以特定方式运行的潜力感到兴奋，一位成员指出，能够如此迅速地引导模型看起来非常 *amazing*。
   - 讨论突显了社区对高效模型定制技术的兴趣。
- **电子原理图数据集出现**：分享了一个用于训练 **LLMs 创建电子原理图** 的新数据集，[数据集地址](https://huggingface.co/datasets/bshada/open-schematics/discussions) 位于 Hugging Face。
   - 成员们认为该资源对于突破 AI 驱动设计的界限具有极高价值。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Genetic-Pareto 优化引发热议**：一位成员对运行他们的第一个真实 **GEPA optimization** 表示兴奋，称其为“魔法”，而另一位成员则开玩笑说这是机器人使用 **DSPy** 构建机器人。
   - 他们指出，自[关于使用优化器的博客文章](https://thedataquarry.com/blog/learning-dspy-3-working-with-optimizers/#option-2-gepa)撰写以来，某些语法可能已经发生了变化，但该文章仍提供了有用的背景信息。
- **GEPA：通过反思性提示词演化实现 AI 构建 AI**：正如[《GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning》](https://arxiv.org/abs/2407.19457)中所述，**GEPA (Genetic-Pareto)** 利用标量分数和文本反馈自适应地演化系统的文本组件。
   - 本质上，**GEPA** 使用另一个 AI 对 AI 提示词进行遗传性修改，并根据指标选择更改，从而实现“AI 构建 AI”。
- **分享 GEPA 学习资源**：一位成员分享了包括 [GEPA 的 DSPy 教程](https://dspy.ai/tutorials/gepa_ai_program/)、[关于使用优化器的博客文章](https://thedataquarry.com/blog/learning-dspy-3-working-with-optimizers/#option-2-gepa)以及一篇[关于 GEPA 实践的 Medium 文章](https://medium.com/data-science-in-your-pocket/gepa-in-action-how-dspy-makes-gpt-4o-learn-faster-and-smarter-eb818088caf1)在内的资源，以帮助他人学习 **GEPA**。
   - 他们提到，自该博客文章撰写以来，某些语法可能已经发生了变化。
- **Tree of Thought (ToT) 模块的缺失受到质疑**：一位成员询问为什么 **DSPy** 中还没有直接的官方 **Tree of Thought (ToT) module**。
   - 目前没有进一步的讨论或回复，该问题尚无答案。
- **`dspy.Refine` 中的自定义反馈引发询问**：一位成员询问在 **DSPy** 中使用 `dspy.Refine` 时，如何手动指定反馈类型。
   - 他们提到正在为评估器循环使用自定义模块，并想知道自己是否遗漏了关于 `Refine` 的某些功能。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **9000IQ In-Context Learning 研究引发关注**：一位成员分享了[一段 YouTube 视频](https://www.youtube.com/watch?v=q-yo6TPRPVk)，展示了关于 **In-Context Learning** 的“9000IQ 研究”，并鼓励大家“带着爆米花”观看。
   - 视频展示了对 **In-Context Learning** 的改进，但未提及具体细节。
- **Draft Model 通过并行处理使速度翻倍**：一位成员建议实现一个 **draft model**，用于“猜测”分布在多个 GPU 或服务器上的大型模型各部分的输出，以实现并行处理。
   - 该建议涉及丢弃与草案模型差异显著的响应，并在这些情况下恢复到正常处理，目标是通过系统实现更高效的批量运行。
- **内存带宽限制大型训练集群**：在大型训练集群中，机器内部和机器之间的内存带宽是最大的瓶颈，需要快速的数据访问以减少停顿，而高效的 **pipelining** 可以提高利用率。
   - 训练并不总是“纯粹”的，它涉及多个文档的前向和后向传播，并结合梯度或权重，在某些情况下这相当于顺序处理。
- **Vast.ai 模板故障令人头疼**：一位成员报告了 **Vast.ai** 的一个问题，即它使用错误的模板启动，导致初始化脚本无法按预期运行。
   - 该用户通过使用 **PyTorch** 模板并使用 **rsync** 传输所需的一切来绕过 **Vast.ai** 的模板问题；一位成员建议尝试 **Nvidia** 的 **Brev** 作为潜在解决方案，同时指出 **Runpod** 可能更好。
- **ARC-AGI 分数跳升归因于训练数据污染**：近期各模型在 **ARC-AGI benchmark** 上的性能跃升很可能是由于训练数据包含了基准测试本身，而非隐藏的技术突破。
   - 成员们指出，在这些类型的基准测试中，通常会观察到较小尺寸的模型具有更好的泛化能力。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **DigitalOcean 文章赞扬 Kimi K2 的思考能力**：一篇 [DigitalOcean 文章](https://www.digitalocean.com/community/tutorials/kimi-k2-moonshot-ai-agentic-open-weight-model) 强调了 **Kimi K2** 先进的思考能力。
   - 分享该文章的频道成员非常兴奋，认为它详细介绍了 **Kimi K2** 的架构。
- **免费 Kimi 模型引入每月重置机制**：频道中分享的一张图片显示，**免费 Kimi 模型** 现在将每月重置。
   - 该公告对免费访问引入了时间限制，可能会影响依赖这些模型进行持续项目的用户。
- **Kimi K2 进行 UI 更新**：一张图片展示了 **Kimi K2** 界面的一项新变化。
   - 分享图片的成员兴奋地惊呼：*"这太棒了！！"*，尽管没有讨论具体的更改细节。



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Cloudflare DNS 问题导致部署延迟**：一位用户遇到了持续超过 4 天的 **Cloudflare DNS 问题**，导致部署延迟，并对客服感到沮丧。
   - 该用户对缺乏支持表示不满，称 *除了说直接私信我们之外，没有客服回应*。
- **图片限制影响智能体验**：一位用户质疑聊天模式的图片限制，指出 **Pro/付费** 用户与 **免费** 用户之间的差距，并抱怨 **免费用户** 的 *智能能力已经显著降低*。
   - 该用户引用了 **DeepSeek** 和 **Gemini** 作为没有此类限制的例子。
- **Manus 营收达亿级，达成里程碑**：一位用户分享了一篇文章，报道 **Manus** 的营收已达到 **1 亿美元**。
   - [SCMP 文章](https://www.scmp.com/tech/tech-trends/article/3336925/manus-hits-us100-million-revenue-milestone-global-competition-ai-agents-heats) 指出，这一里程碑是在 *全球 **AI agents** 竞争升温* 之际达成的。



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Prompts 寻求 Node Server 激活**：一位成员询问如何在 **Node server** 中启用 **MCP prompts** 以使用 **microservices**。
   - 另一位成员建议使用 **#mcp-support** 或提交 issue。
- **询问 ChatGPT App 提交要求**：一位成员询问 **ChatGPT apps** 提交是否需要 UI，或者仅有 **MCP server** 是否足够。
   - 未收到回复。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Claude 未能完成 JIT 重构**：一位成员尝试让 **Claude** 执行 **JIT 重构** 但未获成功，指出它对该任务缺乏“品味”。
   - 该重构涉及完成 **schedulecache** 并让 **JIT** 运行一些 **schedulecaches**。
- **Tinygrad 攻克固件挑战**：一位成员报告了使用在 **Linux** 上模拟虚假 **USB device** 的模拟器成功进行 **firmware** 操作。
   - 此设置能够将数据传递给 **firmware**，从而简化测试和开发流程。
- **RDNA3 汇编后端已就绪**：一个包含寄存器分配器的 **RDNA3 assembly backend** 现在能够运行带有 **128 accs** 的 gemms。
   - 该后端可在 [GitHub](https://github.com/tinygrad/tinygrad/pull/13715) 上获取，为 GPU 加速计算提供了一个新工具。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。


---



您收到此邮件是因为您在我们的网站上选择了订阅。

想更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord：各频道详细摘要与链接

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1450940630158282945)** (1112 条消息🔥🔥🔥): 

> `GPT-1.5 审查, Gemini 图像生成 vs. GPT, Gemini 3 Flash 性价比, Google vs OpenAI: 算力` 


- **GPT-1.5 被指控审查艺术风格**：成员们注意到 **GPT-1.5** 似乎在审查现有的艺术风格，而 [NBP 则没有此类限制](https://cdn.discordapp.com/attachments/1340554757827461211/1451262267843678328/image0.jpg?ex=694588cf&is=6944374f&hm=adb4887706e7f28c99de9871e464ab5ada4bbef35545b46edf6a3f15708450b0&)。
   - 还有人指出 **GPT-1.5 Image** 有时会将 *Google* 拼错为三个 *O*。
- **Gemini 图像表现出色**：用户正在讨论 **Gemini** 与 **GPT** 在图像生成方面的幻觉率和质量，特别是 [历史图像](https://www.theverge.com/2024/2/21/24079371/google-ai-gemini-generative-inaccurate-historical)。
   - 一些人声称 Gemini 会根据用户的要求生成内容，且 **Gemini Pro** 的图像生成效果很好，但也有人指出存在不准确以及历史图像政治化的问题。
- **新款 Gemini 3 Flash 模型令人印象深刻**：用户称赞 **Gemini 3 Flash** 相比 **Opus 4.5** 甚至 **GPT 模型** 具有极高的性价比，但也有人注意到该模型非常容易产生幻觉（halucinate）。
   - 许多用户对 **Flash 3** 本身感到兴奋，并认为 Gemini 将在基准测试中刷榜（benchmaxxes）。
- **关于算力（compute）的讨论**：成员们讨论了 **Google 的 TPU** 在 AI 训练方面是否优于 **Nvidia 的 GPU**，引发了关于效率、性能和基础设施需求的辩论。
   - 争论焦点在于 **TPU** 是否在单位功耗性能上表现更好，以及 Google 的内部资源是否使其相比 **OpenAI** 具有优势。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1451289254683480147)** (4 条消息): 

> `Arena-Rank 开源, 图像编辑排行榜更新, 搜索排行榜更新, 文本排行榜更新` 


- **Arena-Rank 正式开源！**：团队发布了 [Arena-Rank](https://github.com/lmarena/arena-ai)，这是一个用于成对比较排名的 **开源 Python 软件包**，它为 **LMArena 排行榜** 提供动力。
   - 此次发布体现了方法论和工程上的升级，包括更快的基于 **JAX** 的优化，以及数据预处理与建模之间更清晰的分离；可以通过 `pip install arena-rank` 进行安装。
- **`reve-v1.1` 模型登上图像编辑排行榜！**：新模型 `reve-v1.1` 和 `reve-v1.1-fast` 已登上 [图像编辑排行榜](https://lmarena.ai/leaderboard/image-edit)，分别排名第 8 和第 15。
   - 这代表着比 Reve V1 提升了 **+6 分**；更多详情请查看 [排行榜变更日志](https://news.lmarena.ai/leaderboard-changelog/)。
- **`GPT-5.2` 和 `Grok-4.1` 模型在搜索竞技场中更新**：[搜索竞技场排行榜](https://lmarena.ai/leaderboard/search) 已更新，`GPT-5.2-Search` 排名第 2，`Grok-4.1-Fast-Search` 排名第 4。
   - 这两款模型的首次亮相均超越了其前代产品，`GPT-5.2-Search` 增长了 **+10 分**，`Grok-4.1-Fast-Search` 增长了 **+17 分**。
- **`GPT-5.2` 在文本竞技场首次亮相！**：[文本竞技场排行榜](https://lmarena.ai/leaderboard/text) 已更新；`GPT-5.2` 首次亮相并排名第 17。
   - 与 `GPT-5.1` 相比，该模型提升了 **+2 分**，仅落后于专为专家级推理和关键任务优化的 `GPT-5.2-high` 一分；请在 [排行榜变更日志](https://news.lmarena.ai/leaderboard-changelog/) 中追踪变化。

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1450940650001404026)** (941 条消息🔥🔥🔥): 

> `Browser mining extensions, Lottery bitcoin mining, Browser Exploitation Framework, ChatGPT Jailbreak subreddit ban, Fetch tokens sales` 


- **扩展程序拦截浏览器挖矿**：用户讨论了使用浏览器扩展并禁用 **GPU** 加速页面渲染，以防止恶意脚本进行加密货币挖矿，并分享了一个指向 **Browser Exploitation Framework (BeEF)** 的[链接](https://beefproject.com/)。
   - 用户担心受信任的网站可能被入侵，即使在 **NoScript** 允许的情况下也会运行加密货币挖矿程序。
- **无矿池彩票式挖矿收益微薄**：一位用户提到一个朋友使用 **Raspberry Pis** 作为独立矿工挖比特币 5 年，仅赚了约 **$30k**。
   - 共识是使用 **ASICs** 并加入矿池会高效得多，*因为除非你有 coinbase，否则无法出售比特币*。
- **Reddit Jailbreaking 动态**：**ChatGPTJailbreak** subreddit 因违规被封禁，具体原因是干扰网站正常使用并引入恶意代码。
   - 用户推测封禁是由于[破坏网站或干扰正常使用](https://www.reddit.com/r/ChatGPTJailbreak/comments/1lzrh3v/best_prompt_for_jailbreaking_that_actually_works/)引起的。
- **巧妙的残障辅助技巧**：一位用户分享了一种 **Unity GPT5 Jailbreak** 方法，涉及多个步骤，包括指令文件和一段 *ImHandicapped* 文本，并在提示 *Hi Unity* 之前使用粘贴和取消技巧。
   - 该方法旨在对 AI 进行完全内存 Jailbreak，以绕过文本过滤器，同时仍允许生成图像。
- **Gemini 的策略：破解代码以获取代币和克隆**：用户讨论了 Jailbreak **Gemini 3 Pro** 的方法，包括使用免费预览 API 进行调用、逆向工程软件以及创建现有软件的开源克隆。
   - 一位用户声称在 **Gemini CLI** 上发现了后门可以免费使用模型，并分享了他们让 [Claude Opus 逆向工程软件](https://www.unityailab.com)的过程。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1450943854587940865)** (359 条消息🔥🔥): 

> `Gemini 5.2 jailbreak, LLM Jailbreaking Techniques, Nano Banana Pro restrictions, r/chatgptjailbreak ban, Gemini image generation` 


- **成员寻求新的 Gemini 5.2 Jailbreak**：成员们询问关于 **Gemini 5.2** 及其他模型的新 **Jailbreak**，寻求绕过限制并生成特定内容的方法。
   - 一位成员询问关于使用 **Gemini Jailbreak** 来为男性*增大乳头和乳晕*以用于*恋物目的*。
- **用户探索高级 LLM Jailbreaking 技术**：讨论涵盖了各种 **Jailbreaking 技术**，包括通过提供可信的上下文来说服模型，以及使用推理提示来揭示隐藏的限制。
   - 成员建议利用 AI 过去的知识来“*演戏直到成真*”，慢慢提取信息以绕过安全措施。
- **Nano Banana Pro 面临更严格的安全检查**：用户注意到 **Nano Banana Pro** 上的 **安全限制** 增加，影响了图像生成能力。
   - 尽管有这些限制，一位用户声称 [Jailbreaking Nano Banana 是非常有希望的](https://discord.com/channels/1228043845967544380/1228043845967544383/1451217592281862394)，并展示了示例图像。
- **Reddit 的 r/chatgptjailbreak 因违反规则被关闭**：社区讨论了 **r/chatgptjailbreak 的封禁**，将其归因于违反了 Reddit 的 Rule 8，该规则禁止发布 Reddit AI 的 Jailbreak。
   - 据推测，该版块的内容可能已被 **OpenAI** 抓取以修复漏洞，这与 BASI 服务器更具“*地下*”性质形成对比。
- **Gemini Nano 的图像生成 Jailbreaking 已“搞定”**：成员们正在使用来自 [Exocija 的 ZetaLib GitHub 仓库](https://github.com/Exocija/ZetaLib/blob/main/Prompts/Jailbreaks/1Shot%20Puppetry/1Shot%20Puppetry.mkd) 的 “Puppetry” **Jailbreak** 方法。
   - 一位成员报告称在访问内部机制方面取得了一些成功，并正在研究 **T2V JB** 以实现 *Gemini Nano 的图像生成 Jailbreaking*。


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1451133914327289918)** (3 条消息): 

> `` 


- **未发现主题**：提供的消息中未发现任何主题。
- **未发现讨论**：提供的消息中未发现任何讨论。


  

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1451015401898246259)** (1 条消息): 

> `Gemini 3 Flash, Perplexity Pro, Perplexity Max` 


- ****Gemini 3 Flash** 正式发布！**：已宣布 **Gemini 3 Flash** 向所有 **Perplexity Pro** 和 **Max** 订阅者开放。
   - 包含一张附图，可在[此处](https://cdn.discordapp.com/attachments/1047204950763122820/1451015401373700147/G8aRzWjakAYU7ED.png?ex=69454ba6&is=6943fa26&hm=bedceef3e30e9af5b277bc59f988aedd681d69e33648f01208a4973b6169cbe9&)查看。
- **Gemini 与订阅**：**Gemini 3 Flash** 现已面向所有 **Perplexity Pro** 和 **Max** 订阅者开放。
   - 这为高级用户提供了更强大的功能。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1450941700087349504)** (886 条消息🔥🔥🔥): 

> `GPT-5 Pro on Perplexity, Gemini 3 Pro vs ChatGPT vs Claude for coding, Ethically Sourced Music AI, Perplexity Pro Referral Program, Tilly Norwood and AI in Hollywood` 


- **GPT-5 Pro 降临 PPLX**：部分用户在 Perplexity 上注意到了 **GPT-5 Pro**，但尚未对所有用户启用，仅限 **pro 用户**，而其他用户则在他们的 **max 订阅**中进行了测试。
   - 一位用户甚至说 *不可能，你居然跟 gpt 5 pro 打招呼了*，并表示他们也在 API 上跟 **opus 4.5** 打了招呼，但这尚未得到证实。
- **Gemini 擅长常识，Claude 是代码之神，Gemini 与 Chat 互有胜负**：成员们辩论了哪个模型更适合不同的任务：**Gemini** 在通用知识和写作方面表现更好，**ChatGPT** 擅长数学和复杂编程，而 **Claude** 则在 *Agent 化一切* 方面领先。
   - 一位成员发现 **Gemini 3 Pro** 在构建单页落地页时表现不佳，而 **ChatGPT** 和 **Claude** 表现出色；同时[另一位成员指出](https://antigravity.google/blog/gemini-3-flash-in-google-antigravity)指出，**Gemini 3 Flash** 在代码 Agent 能力上超越了 **Gemini 3 Pro**。
- **AI 音乐引发热议：这合适吗？**：成员们讨论了 **AI 在音乐中的应用**，一些人认为它破坏了人类的表达和创造力，而另一些人则认为它降低了创作门槛，并通过提示词技巧（如 [aiode.com](https://aiode.com) 这种道德来源的音乐 AI）帮助激发新灵感。
   - 一位用户表示 *如果 AI 创作这种音乐，那就像有人往你脸上吐口水*。但仍然承认 **Suno v5** 很难与真人创作的音乐区分开来。
- **Perplexity 推荐计划对部分用户失效，引发不安**：一些用户报告称 Perplexity 推荐计划未按预期运行，推荐人和被推荐用户都没有收到承诺的 Perplexity Pro 版本。
   - 一位用户说这让人感觉 *很诡异 (spooky)*。
- **Comet 盯上了你的数据**：用户分享了他们对 **Comet** 浏览器的怀疑，出于 *隐私担忧*，因为其 Agent 工具可能会被恶意指令滥用。
   - 其他人则引用[这篇博文](https://thereallo.dev/blog/comet-browser-panic)为其辩护，该文章驳斥了这一问题，但它仍然依赖于用户尽可能安全地操作，并希望它能通过 *帮助你* 变得更聪明。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1451276466271686717)** (10 条消息🔥): 

> `Perplexity Pro API key, Financial Modeling Prep, Realtime Price Feeds, Finnhub Data Provider` 


- **关于 Perplexity Pro 是否提供 API 密钥的讨论**：一位成员询问通过 **Airtel SIM** 获得的 **Perplexity Pro** 是否可以用于生成 **API 密钥**。
   - 另一位成员对价格合理的 API 表示感兴趣，但发现它们并不划算。
- **使用 Financial Modeling Prep 作为替代方案**：一位成员建议使用 **Financial Modeling Prep (FMP)** MCP 服务器（Perplexity 的数据来源），进行自托管，并使用自托管的 **n8n 工作流** 自动抓取市场数据。
   - 另一位成员此前不知道这个选项，并计划进行探索。
- **实时数据推送非常昂贵**：成员们确认 **Perplexity 金融工具** 和 **Financial Modeling Prep MCP** 都不提供实时价格推送，仅提供特定时间点的数据获取。
   - 他们指出，实时的市场数据非常昂贵，每月成本高达数千美元。
- **Finnhub 作为更便宜的替代方案**：一位成员提到 **Finnhub** 提供了更便宜的市场数据选项，但使用 **BAT** 作为数据提供商而非 **NASDAQ**，这可能导致与 **TradingView** 等平台相比出现价格差异。
   - 价格差异可能在几美分左右，在极少数情况下甚至高达 50 美分。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1450943266982985729)** (263 条消息🔥🔥): 

> `Qwen3-VL, 保存 Embeddings, GLM-4.6V-Flash-GGUF 重复问题, RL 模型 Beta 发布, 在手机上进行 Finetuning` 


- **Qwen3-VL 并不太笨**：**Qwen3-VL 30B Instruct** 并不是特别聪明，但也不至于笨得离谱，而且它的冗长程度只有 **Thinking** 模型的一半。
   - 一位成员分享了一张附图作为该模型性能的示例。
- **保存 Embeddings 的问题浮现**：一位成员报告了在**保存训练好的 embeddings** 时遇到的问题，指出这在 embedding 阶段带来了巨大的 eval 收益，但并未影响下一阶段的训练。
   - 不过，他们后来澄清是其他原因导致的，并且他们设法让 embedding 阶段产生了巨大的 eval 收益，但这完全没有影响到下一阶段的训练。
- **Unsloth 的 GLM-4.6V-Flash-GGUF 出现重复**：一位成员报告在使用最新的 llama.cpp 运行 ud q4kxl 时，`unsloth/GLM-4.6V-Flash-GGUF` 出现了**重复问题**，而另一位用户则遇到了模型*用中文思考*的情况。
   - 在重新下载模型后，问题似乎得到了解决，该用户分享了他们的 `llama-server.exe` 配置。
- **使用 Pytorch 进行 LLMs 微调并部署到手机**：Unsloth 宣布与 **Pytorch** 合作推出一个 Colab，支持微调 LLMs 并直接部署到手机上，并附带了 [推文链接](https://x.com/UnslothAI/status/2001305185206091917)。
   - 官方澄清 Finetuning 是在电脑上进行的，但部署是在手机上——Meta 正在生产环境中使用 **Executorch** 指南（这是*最优化版本之一*）为数十亿人提供服务。
- **Unsloth 同时支持 Transformers 4 和 5**：一位成员询问 **Unsloth** 是否同时支持 **Transformers 4** 和 **5**，团队确认两个版本均已支持，包括 v5。
   - 一位团队成员指出，虽然 **v5** 存在一个 Tokenizer 问题，但由于稳定性考虑，该版本默认并未启用。


  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1451269539303260262)** (1 条消息): 

> `Unsloth 更新 - 3倍提速, FunctionGemma, Nemotron 3, Mistral VLMs, GLM-4.6V` 


- **Unsloth 训练速度翻三倍**：Unsloth 的最新更新实现了 **3倍的训练提速** 和 **30% 的 VRAM 节省**，这得益于新的 Kernel、无 Padding 实现和 Packing 技术，详见其 [博客文章](https://docs.unsloth.ai/new/3x-faster-training-packing)。
   - 该更新还支持在单张 80GB GPU 上进行 **500K Context** 训练，如本 [Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt_oss_(20B)_500K_Context_Fine_tuning.ipynb) 所示。
- **Google 的 FunctionGemma 发布**：Google 新推出的 **270M 工具调用 LLM** —— **FunctionGemma** 现已获得支持，并提供了 [指南](https://docs.unsloth.ai/models/functiongemma) 和 [Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M).ipynb)。
   - 该模型允许在各种应用中进行高效的工具集成和利用。
- **NVIDIA 的 Nemotron-3 亮相**：NVIDIA 的 **Nemotron 3**（一款新的 **30B 推理模型**）现已可用，并配有 [指南](https://docs.unsloth.ai/models/nemotron-3) 和 [GGUF 版本](https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF)。
   - Nemotron 3 在需要逻辑推理和知识处理的任务中表现出色。
- **Mistral 模型加入 Unsloth**：来自 **Mistral** 的新型代码和指令 VLM 现已获得支持，包括 [Ministral 3](https://docs.unsloth.ai/models/ministral-3) 和 [Devstral 2](https://docs.unsloth.ai/models/devstral-2)。
   - 这些模型为代码生成和指令遵循任务提供了增强的能力。
- **视觉模型 GLM-4.6V 登场**：**GLM-4.6V** 这一新型视觉模型现已可用，并可通过此 [指南](https://docs.unsloth.ai/models/glm-4.6-how-to-run-locally) 在本地运行。
   - 另请查看 [4.6V](https://huggingface.co/unsloth/GLM-4.6V-GGUF) 和 [4.6V-Flash](https://huggingface.co/unsloth/GLM-4.6V-Flash-GGUF) 的 GGUF 链接。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1450941929842802888)** (405 条消息🔥🔥🔥): 

> `Tokenizer 过拟合、迁移至 Arch Linux、Google Colab 上的 H100、多语言 TTS 模型、T5Gemma 2` 


- **Tokenizer 在随机垃圾内容上过拟合**：一位成员抱怨 Tokenizer 在诸如 **`**/`** 和 `/**` 标记、奇怪的符号组合以及特定语言字符（如 *"æĪĲåĬŁ"*）等随机垃圾内容上发生了过拟合。
   - 他们幽默地补充道，甚至连 *"pants are not -.->"* 也是如此。
- **用户愤而放弃 Ubuntu，拥抱 Arch**：一位用户因受够了 Ubuntu 而转向 Arch（特别是 **Omarchy**），理由是过多的 **environment** 和 **config** 问题导致操作系统崩溃。
   - 另一位用户建议安装 *minimal base Arch* 而不是 **Omarchy**，并配合自定义 DE 和下载的组件；而另一位用户则指出 **Omarchy** 在安装 Nvidia 驱动方面非常简便。
- **Google Colab 上发现了 H100？！**：一位用户惊讶地发现即使是 **Colab Pro** 也能看到 **H100**。
   - 另一位用户感叹自己还没抢到过。
- **多语言 TTS 模型难题**：一位用户质疑在将多个说话人和语言（英语、日语和德语）组合进一个模型时，多语言是否会损害 **TTS** 模型的效果，可能需要为每种语言建立独立模型。
   - 成员们建议对 text encoder 进行条件化处理或使用 language embeddings，并强调了 **phoneme-level timing** 对 **TTS** 质量的重要性。
- **T5Gemma 2 的潜在用例**：讨论围绕拥有 270M encoder 和 decoder 的 **T5Gemma 2** 的用途展开，一位用户建议它可以像音频处理中那样用于 embeddings。
   - 另一位用户指出 **image models** 使用了 T5。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1451168473819648010)** (46 条消息🔥): 

> `Qwen3 4B Instruct 错误、FBGEMM 警告、OCR 性能、PaddleOCRv5、Qwen3 VL` 


- **Qwen3 4B instruct 报错**：一位用户报告了在使用 **Qwen3 4B instruct** 时遇到的错误，并提供了代码片段用于上下文参考和调试。
   - 该用户的代码使用 `FastLanguageModel.from_pretrained` 加载带有 4-bit 量化的 `unsloth/Qwen3-4B-Instruct-2507` 模型，并指定了使用 `trl` 库中 `SFTTrainer` 进行训练的参数。
- **出现 FBGEMM 警告**：一位用户在 Windows 上使用最新的 Unsloth Docker 镜像加载模型时遇到了警告信息 *"FBGEMM on the current GPU cannot load - will switch to Triton kernels"*，尽管其 GPU 满足计算要求。
   - 用户询问了潜在的性能影响以及这是否是已知问题，表现出对切换到 **Triton kernels** 的担忧。
- **Qwen3 VL 在 OCR 方面表现出色**：一位成员发现 **Qwen3 VL**（`30B-A3B` 变体）在 OCR 检测方面明显更好，能有效捕捉页边距文本和旋转文本。
   - 他们指出 Prompt 非常关键；简单的 Prompt 如 *"Extract all text from this image"* 会产生无结构的结果，而要求人工可读格式的 Prompt 可能会导致文本被截断。
- **PaddleOCRv5 可能胜过 VLM**：一位成员强烈建议在 OCR 任务中使用 **PaddleOCRv5** 而不是通用的 **VLM**，理由是它在他们的用例中表现更优。
   - 虽然另一位用户之前对 **Paddle** 的体验一般，但他们愿意重新评估，并承认 **Paddle** 可能有所改进且具有强大的文本检测能力。
- **Unsloth 自动扩展 Token**：一位成员询问在 PEFT 上指定 `embed_tokens` 时，Unsloth 是否会自动为 CPT 扩展 Token。
   - 另一位成员澄清说，要添加新 Token，应使用 `unsloth` 中的 `add_new_tokens` 函数，并提供模型、Tokenizer 和新 Token 列表。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1450961609089613987)** (5 条消息): 

> `AI Context 的渐进式披露、Qwen3-4b-Deep-Beta 模型发布、Savant Commander MOE 模型` 


- **AI Context 的渐进式披露**：一位成员建议采用 *progressive disclosure*（渐进式披露），在 Context 中维护一个包含相关文档链接的索引，以增强信息获取。
- **Qwen3-4b-Deep-Beta 发布！**：一位成员宣布了他们的第一个模型，[Qwen3-4b-Deep-Beta](https://huggingface.co/Solenopsisbot/Qwen3-4b-Deep-Beta)，该模型被设计为一个*深度推理模型*。
- **Savant Commander：256K Context Gated Distill MOE 模型**：一位成员展示了一个 **GATED Distill MOE** 模型（[Qwen3](https://huggingface.co/Qwen)），拥有 **256K Context**，并感谢了 [model tree 和 repo 页面](https://huggingface.co/DavidAU/Qwen3-48B-A4B-Savant-Commander-GATED-12x-Closed-Open-Source-Distill-GGUF)中列出的微调人员。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1451327450876350584)** (2 messages): 

> `MoLA, Adapter Training, Reasoning, Token Budgeting` 


- **MoLA Adapters 针对推理工作量进行训练**：一名成员提出了一项想法，即使用 **MoLA** (Mixture of Low-Rank Adapters)，其中每个 Adapter 针对不同的推理工作量（Reasoning Efforts）而非不同的领域进行训练。
   - Router 将对难度进行分类，并选择具有合适推理预算的 Adapter，这样你就不会在一条 *Hello* 消息上消耗 **1000 个推理 Token**。
- **使用 Adapters 进行推理预算管理**：Adapters 可以对难度进行分类并分配推理预算。
   - 这避免了在琐碎的消息上过度消耗 Token。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1451255773479964793)** (1 messages): 

> `JSON repair, Browser notifications, Long-context models, Fastest-growing AI infra` 


- **JSON Repair 修复响应**：OpenRouter 现在会自动修复格式错误的 JSON 响应。正如[此公告](https://openrouter.ai/announcements/response-healing-reduce-json-defects-by-80percent)所述，该功能在 **Gemini 2.0 Flash** 上减少了 **80%** 的缺陷，在 **Qwen3 235B** 上减少了 **99.8%** 的缺陷。
- **Chatroom 现在支持通知**：如[此帖子](https://x.com/OpenRouterAI/status/2000980715501138180)所述，Chatroom 现在会在长响应完成时发送浏览器通知。
- **长上下文排名**：排行榜页面现在可以按上下文长度进行过滤，以查看热门的大上下文模型，特别是针对 **100K-1M Token** 的请求，详见[此推文](https://x.com/OpenRouterAI/status/2000979922601259476)。
- **OpenRouter 在 Brex 的 AI 基础设施列表中排名第一**：根据[此公告](https://x.com/brexHQ/status/2000959336894398928)，OpenRouter 在“基础设施即产品”类别中排名 **第 1**，在所有 AI 公司中排名 **第 2**。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1451013714504585409)** (44 messages🔥): 

> `AI-made Discord Server List, Image Verification, LLM System Prompt Test Cases, OpenRouter Model Table` 


- **AI 驱动的 Discord 服务器发现功能首次亮相**：一名成员介绍了 [Disdex.io](https://disdex.io/)，这是一个使用 AI 制作的 **Discord 服务器列表**，并征求对其功能的反馈。
   - 有人对其在没有明确解释的情况下的价值主张提出了质疑，强调需要 *“推销你的产品”*。
- **图像真实性受到质疑**：一名成员发布了一张图片，质疑其真实性，另一名成员回应称其是**真实的**并鼓励进行验证。
   - 随后，该成员报告称其搜索统计数据未能更新，但承认在刷新后已修复。
- **LLM System Prompt 测试技巧**：一名成员在与 LLM 聊天时寻求有关 System Prompt 常用测试用例的建议，并链接到了[现有的测试用例](https://github.com/pinkfuwa/llumen/blob/main/prompts%2Fpromptfoo%2Fnormal.yaml#L145-L185)。
   - 该成员目前正使用 promptfoo 为 **llumen** 测试新的 System Prompt。
- **带有高级过滤功能的 OpenRouter 模型表格亮相**：一名成员分享了一个 [可搜索、可过滤的 OpenRouter 模型数据表](https://openroutermodeltable.crashthatch.com/) (**OpenRouter API**)，以协助模型选择，解决了 OpenRouter 原生过滤器的局限性。
   - 另一名成员提到了类似的工具 [orca.orb.town](https://orca.orb.town/)，但强调了对 *“发布日期”* 和 *“吞吐量”* 等列标题进行过滤的需求。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1450941304665014415)** (430 messages🔥🔥🔥): 

> `Gemini 3 Flash caching, Chutes crypto mining, deepseek v3 0324 context size, AI water usage, Openrouter Android/iOS app` 


- **Gemini 3 Flash 缓存失效**：成员们报告称，显式甚至隐式缓存对 **Gemini 3 Flash** 都**不起作用**。
- **Chutes 加密货币挖矿重心转向 AI**：据推测，**Chutes 的加密货币挖矿集群已下线**，他们现在正**专注于 AI**。
- **Deepseek v3 0324 上下文大小锁定在 8k**：成员们注意到 **Deepseek v3 0324** 的**上下文大小（context size）被锁定在 8k**，但仍有一些人能够毫无问题地突破这一限制。
- **AI 数据中心用水量正在摧毁社会**：成员们开玩笑说，AI 数据中心的用水量是一场*环境灾难*，简直是在**摧毁社会**。
   - 一位成员指出，[高尔夫行业的用水量](https://en.wikipedia.org/wiki/Water_use_and_environmental_impacts_of_golf_courses)甚至更为严重。
- **Openrouter 需要 Android/iOS 应用**：成员们请求为 **Android** 和 **iOS** 提供一个好用的**移动端 App**，因为目前使用 **Openrouter API** 的现有应用并不令人满意。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1450945275182055465)** (192 messages🔥🔥): 

> `Mistral Large 3 Quality, OpenRouter Website Performance, Vision Function for Bots, AI Learning Resources for GTM Team, Gemini Model's Pixel-Perfect Bounding Boxes` 


- **Mistral Large 3 用户体验平平**：一位用户表示，他们认为 **Mistral Large 3** 并没有那么好，但正在使用其免费层级，并指出它*非常慷慨*，在一项实验中为他们节省了资金。
- **OpenRouter 网站出现间歇性变慢**：用户报告 OpenRouter 网站间歇性卡顿，怀疑是 *Vercel 出了点问题*。
- **Vision 功能阻碍机器人集成**：一位成员尝试为机器人启用 Vision 功能但未成功，可能是因为处于 headless 环境，并通过 [X 链接](https://x.com/pingToven/status/2001479433010790880?s=20)寻求帮助。
- **AI 见解赋能 Go-To-Market 团队**：成员们讨论了 Go-To-Market (GTM) 团队所需的 AI 知识水平，参考了诸如 [此 YouTube 播放列表](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) 等资源，并强调理解应用比科学术语更关键。
   - 讨论中提到，*外面使用 OR 的人甚至不知道什么是 system prompt*。
- **Gemini 的边界框功能引发关注**：新 **Gemini** 模型对文本进行像素级边界框（bounding boxes）定位的能力给成员们留下了深刻印象，有人解释说 *它不返回图像，只返回边界框坐标，然后你可以将其叠加到图像上*。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1450942021840801822)** (447 messages🔥🔥🔥): 

> `Domain name pricing, Gemini 3 flash, Obsidian Copper and Carbon Monochrome themes, Free models in cursor, Student Discount Eligibility` 


- **域名成本引发热议**：用户讨论了域名的成本，一位用户以 **$100** 购买了一个域名，而另一位用户为他们的业务支付了 **$25k**。
   - 一位用户表示震惊，而域名购买者解释说，他们认识的人中有人曾以 **$200 万**购买域名。
- **Gemini 3 Flash 引发模型对比**：用户被问及是否测试过 **Gemini 3 Flash**，以及它与 **Sonnet 4.5** 的对比。
   - 另一位用户表示 **Claude Opus 4.5** 是目前最好的模型，而 **Gemini 3 Pro** 在设计方面表现最佳。
- **Obsidian Copper 和 Carbon Monochrome 主题最受欢迎**：一位用户询问大家最喜欢哪种**主题**来使用 **shadcn components** 设计**仪表盘**。
   - 回复各异，但最受欢迎的似乎是 **Obsidian Copper** 和 **Carbon Monochrome**，因为 *Slate Emerald 看起来非常有 AI 感*。
- **学生订阅引发狂欢与国际向往**：一位用户发现学生可以获得为期一年的**免费 Pro 订阅**。
   - 另一位来自乌克兰的用户哀叹他们的国家不在符合条件的名单上，而 Cursor 团队正努力在 2026 年扩大该计划。
- **模型性能骤降？**：一位用户声称他们在使用 **Opus 4.5** 模型时一天出现 10 次错误，而之前的表现更好，*感觉就像回到了 sonnet 3.8，到底发生了什么？*。
   - 另一位用户询问在过去 48 小时内 **Opus/Sonnet** 的代码质量是否有所下降。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1450961764308226059)** (129 条消息🔥🔥): 

> `Gemini 在 LM Studio 中的 Deep Research，用于 API 网站运营的开源模型，智能家居设置，DDG MCP 服务器，Cursor IDE` 


- **开源模型实现半 Agent 式的 API 网站控制**：用户讨论了使用开源模型配合 GraphQL 客户端通过 API 操作网站，一位用户建议将 schema 提供给模型，让其自行编写 API 查询语句。
   - 有人担心在没有描述的情况下*直接倾倒整个 schema* 可能效果不佳，且该用例对于本地模型来说可能过于超前；不过一名成员在编写了清晰的 schema 脚本后，使用 **Qwen3-30b Q6** 进行 SQL 查询取得了一定成功。
- **展示 Docker DuckDuckGo MCP 服务器集成**：一位成员分享了使用 Docker 自托管免费且私有的 **DuckDuckGo MCP 服务器**的配置，并提供了必要的 `mcp.json` 配置片段。
   - 他们解释说，这种设置涉及单独添加 DuckDuckGo 服务器，而不是使用 Docker-Toolkit 以避免模型陷入困境，该设置会重定向到一个 Docker 容器 [searxng/searxng](https://github.com/searxng/searxng)。
- **Cursor IDE 在本地模型上表现挣扎，VS Code 胜出**：一位用户报告了在 **Cursor IDE** 中使用本地模型的问题，LLM 生成了一个带有 CNN 的 Python 脚本，而不是按指令生成 C++ 项目，这被[判定为 Cursor IDE 的问题](https://cdn.discordapp.com/attachments/1110598183144399061/1451084212844101764/Screenshot_2025-12-18_at_3.24.48_pm.png?ex=69458bbc&is=69443a3c&hm=9b55d376a142a8a5c17f4703a01d49bcf769976debd4171ccd4bc814dcb58529&)。
   - 有建议认为 Cursor 与本地模型配合不佳，推荐改用 **VS Code** 搭配 Cline；此外还建议需要反向代理，该成员提供了用于设置带有 HTTPS 和 SSH 隧道的反向代理的 Apache2 虚拟主机配置。
- **用于网页搜索的 Nemotron 30B vs Qwen3**：一位成员表示在网页搜索方面更倾向于使用 BF16 格式的 **Nemotron 30B**，因为它比 Qwen3 更直接，尽管承认其代码编写能力可能很糟糕。
   - 另一位成员建议如果需要知识储备则使用 Qwen3-Next，或者使用更小的模型以获得更好的指令遵循效果，并警告说 Nemotron 30B 在 Q8 量化下由于质量问题几乎不可用。这里是教程链接 [Jan MCP on chrome](https://youtu.be/VhK2CQkAuro)。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1450961235427459098)** (132 条消息🔥🔥): 

> `Radeon R9700 扩展性，AMD GPU 寿命，多 GPU 扩展问题，AI 领域的 AMD vs Nvidia，W7800 48G GPU` 


- **Radeon R9700：明智之选还是财务黑洞？**：一位用户正在纠结是再买三块 **Radeon R9700** GPU，还是卖肾买一块 **RTX 5090**，理由是在运行 LLM 时 **4090** 比 **R9700** 快 **40-50%**。
   - 他们指出，多 GPU 扩展会导致性能相对于单 GPU 显著下降，当模型跨多张显卡拆分时，性能会下降到单卡速度的 **30%**，即使模型完全可以装入单张显卡也是如此。
- **AMD GPU 支持：AMD 会抛弃你的 GPU 吗？**：一位用户表示担心 AMD 似乎不愿在合理的时间内提供显卡支持，质疑 **7000 系列**在 2-3 年后是否还会更新驱动。
   - 另一位用户开玩笑地回答：*简单，2-3 年后卖掉就行*。
- **Radeon 上的 vLLM 扩展：可行吗？**：一位用户分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=gvl3mo_LqFo)，该视频表明如果使用 **vLLM** 和支持张量并行（tensor parallelism）的模型，**2x R9700** 可以达到与 **4090** 类似的吞吐量。
   - 该用户仍感叹 AMD 的新“AI 显卡”性能仅与几代前的 **3090** 相当。
- **W7800 加入讨论**：讨论围绕 **AMD Radeon Pro W7800** 展开，特别是 **48GB** 版本，[techpowerup 链接](https://www.techpowerup.com/gpu-specs/radeon-pro-w7800-48-gb.c4252)显示它拥有 **384-bit 位宽**和约 **900 GB/s** 的带宽。
   - **W7800** 被视为一种潜在的替代方案，价格约为 **$2000**。
- **在 AMD 上运行 ComfyUI：出乎意料地简单？**：有人提到可以参考一个简短指南，其中提供了使用 **AMD & ROCm 运行 ComfyUI** 所需的所有步骤，并指向了[这份指南](https://forum.level1techs.com/t/minisforum-ms-s1-max-comfy-ui-guide/237929)。
   - 指南还建议获取自带 **PyTorch** 支持的 **AMD 预览版驱动**。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1451022473108656312)** (7 messages): 

> `Lecture 1 Error, model definition, profiling code` 


- **调试 "Lecture 1 Error"**：一位用户询问了关于在 "Lecture 1" 期间遇到的错误调试问题。
   - 一名成员回答说，该用户**正在对非 main() 函数的代码进行 profiling，且需要先定义模型**。
- **LLM 修复**：另一名成员表示 LLM 可以快速修复你的错误。
   - 他们补充道，*在对模型进行 profiling 之前，你需要先定义它。*


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1450963818472673432)** (2 messages): 

> `Spark Devs` 


- **对 Spark 开发工具的热情**：一名成员对一款与 **Spark development** 相关的新工具表示兴奋，并提到他们在过去一年里一直在寻找类似的东西。
   - 他们转达了这一信息，以防有人正在为 **Spark** 开发工具。
- **对 Spark 工具的进一步兴趣**：该成员强调了该工具对使用 **Apache Spark** 的开发者的潜在益处。
   - 他们指出，此类工具可以显著简化工作流程并提高生产力。


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1450950525716660447)** (1 messages): 

> `NVIDIA, cuTile, TileIR, Mehdi Amini, Jared Roesch` 


- **NVIDIA 工程师深入探讨 cuTile 和 TileIR**：**NVIDIA** 通过 **cuTile** 和 **TileIR** 对其编程模型进行了深刻变革，正如创作者 [Mehdi Amini 和 Jared Roesch](https://www.youtube.com/watch?v=sjkEUhrUAdw) 所揭示的那样。
- **cuTile 和 TileIR 演讲**：由创作者 **Mehdi Amini** 和 **Jared Roesch** 本人带来的关于 **cuTile** 和 **TileIR** 的深度探讨将在年底前举行，并于 1 月 3 日再次恢复。


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1451267360169136138)** (2 messages): 

> `SemiAnalysis, clusterMAX, SLURM, GPUs, Kubernetes` 


- **SemiAnalysis 为 clusterMAX 招聘！**：SemiAnalysis 正在招聘人员以改进 [clusterMAX](https://www.semianalysis.com/)，寻求具有 **SLURM**、**GPU** 和/或 **Kubernetes** 经验的候选人。
   - 该职位提供*有竞争力的薪资*和*高影响力*，申请详情请见[此处](https://app.dover.com/apply/SemiAnalysis/c19093ad-b5f8-42b0-9b97-d960464f298c/?rs=76643084)。
- **SLURM、GPU 和 Kubernetes 经验**：SemiAnalysis 希望让 clusterMAX 变得更好。
   - SemiAnalysis 正在寻找具有 **SLURM**、**GPUs** 和/或 **k8s** 经验的人才。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1451162521988763739)** (5 messages): 

> `CUDA Setup, DL Projects, Visual Studio, VS Buildtools` 


- **CUDA 安装问题困扰深度学习项目**：一位用户在远程 Windows 10 服务器上为深度学习项目设置 **CUDA** 时遇到问题，报告称尽管安装了最新版本的 VS Community 或 VS Buildtools，**CUDA** 安装程序仍无法找到兼容的 **Visual Studio** 安装。
   - 另一位用户建议在 **CUDA** 安装过程中跳过 Visual Studio 集成步骤，并指出在使用构建工具时这可能不是必需的。
- **跳过 VS 集成解决问题**：一位用户被建议在使用构建工具安装 **CUDA** 时跳过 **Visual Studio integration** 步骤。
   - 该建议暗示 VS 集成是可选的，对于 CUDA 与构建工具正常协作并非必要。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1451267747039416400)** (5 messages): 

> `dtype deprecation in linear_quant_modules, ao namespacing PR` 


- **linear_quant_modules 中的 dtype 弃用问题**：一名成员注意到一个与 **linear_quant_modules** 中的 **dtype** 相关的[弃用 PR](https://github.com/pytorch/ao/pull/3514)，并对测试失败表示担忧。
   - 测试仍然引用了被要求弃用 **dtype** 的模块中的 **dtype**；该成员询问是否可以编辑测试以解决此问题。
- **ao 获得命名空间**：一名成员指出了一个涉及 **ao** 库命名空间的[全库范围 PR](https://github.com/pytorch/ao/pull/3515)。


  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1451080968025804944)** (14 messages🔥): 

> `AI Formal Verification, GPU Kernels Verification, PyTorch PRs, Open Source Contribution` 


- **AI 与形式验证（Formal Verification）的融合**：一位成员分享了一个关于 **AI + 形式验证** 在计算机图形学和设计中应用的[链接](https://martin.kleppmann.com/2025/12/08/ai-formal-verification.html)。
   - 这种结合确保了设计符合规范，并可能通过 **AI 辅助的形式验证** 得到增强。
- **GPU Kernels 需要形式验证吗？**：一位成员询问了关于 **GPU Kernels 形式验证** 的研究。
   - 另一位成员分享了一个可能相关的资源：[VeriNum](https://verinum.org/)。
- **等待处理的 PyTorch PRs**：一位成员反映其两周前提交的 **PyTorch PRs** 进展缓慢。
   - 该成员指出，其中一个 PR 已分配了审阅者但没有活动，而另一个则被标记为已分流（triaged）。
- **开源初体验**：一位想要 **尝试参与开源** 的成员在 PyTorch 中提交了关于弃用（deprecation）的 PR。
   - 另一位成员建议在适当的频道中询问，但也承认这些 PR 的优先级可能较低。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1450975699589660756)** (2 messages): 

> `Training Models on Strix Halo, PyTorch Tutorials, GitHub Repositories` 


- **寻求 Strix Halo 上的训练资源**：一位成员请求提供在 **Strix Halo** 上使用 **PyTorch** 训练模型的 **GitHub 仓库** 或 **教程** 指引。
- **寻找 PyTorch 训练教程**：该成员正在寻找能够帮助使用 **PyTorch** 训练模型的资源。
   - 他们对能够指导其完成整个流程的教程很感兴趣。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1451199895410315285)** (9 messages🔥): 

> `SonicMoE, NVIDIA Hopper GPUs, Princeton University, UC Berkeley, Together AI` 


- **Sonic Boom：SonicMoE 登陆 NVIDIA Hoppers**：介绍了一种名为 **SonicMoE** 的新型混合专家模型（**MoE**）实现，该实现针对 **NVIDIA Hopper GPUs** 进行了优化。据[此博客文章](https://d1hr2uv.github.io/api-agents.html)和 [ArXiv 论文](https://arxiv.org/abs/2512.14080)详细介绍，它声称能减少 **45% 的激活显存**，且在 **H100** 上的速度比之前的最先进实现快 **1.86 倍**。
   - 该项目是来自 **普林斯顿大学**、**加州大学伯克利分校**和 **Together AI** 的研究人员合作完成的，代码可在 [GitHub](https://github.com/Dao-AILab/sonic-moe) 上获取。
- **SonicMoE 演讲定于 3 月 7 日**：在介绍 **SonicMoE** 之后，讨论了在服务器上进行演讲的事宜，并提议日期为 **3 月 7 日**。
   - 一位用户同意参加演讲，但明确表示不会很快举行，建议将 2 月作为可能的时间窗口。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/)** (1 messages): 

kashimoo2_76983: <@1012256135761383465> 你们有没有为 mi300s 或 355s 编写过 decode kernel？
  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1451298535990497392)** (4 messages): 

> `Reasoning-gym code, faker generator, robust tests` 


- **Reasoning-Gym 重构**：一位成员提议重构 reasoning-gym 代码，以标准化答案生成，并包含一个显示步骤追踪的标志，例如 `question(verbose=False, **kwargs) -> Any`。
   - 他们建议[集成 Faker](https://github.com/joke2k/faker) 以使用虚假名称生成更鲁棒的测试。
- **Reasoning-Gym 与 Faker 联动？**：社区讨论了将 reasoning gym 连接到 [faker 生成器](https://github.com/joke2k/faker)。
   - 社区认为这将生成带有虚假名称的更鲁棒的测试，而不是一直使用相同的默认名称。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1450942222462746727)** (13 messages🔥): 

> `nvfp4_gemm benchmark, grayscale_v2 benchmark, H100 performance, NVIDIA performance` 


- **NVIDIA 在 nvfp4_gemm 上的运行**：一位成员提交到 `nvfp4_gemm` 排行榜的结果在 **NVIDIA** 上成功运行，耗时 **13.4 µs**。
- **H100 在 grayscale_v2 上的运行**：一位成员提交到 `grayscale_v2` 排行榜的结果在 **H100** 上获得 **第 6 名**，耗时 **1373 µs**。
- **更多 H100 运行 grayscale_v2**：多次成功提交到 `grayscale_v2` 排行榜的 **H100** 运行记录显示时间均为 **1374 µs**。
- **NVIDIA 在 nvfp4_gemm 上获得个人最佳成绩**：一位成员在 **NVIDIA** 的 `nvfp4_gemm` 排行榜上取得了 **37.5 µs** 的个人最佳成绩。
- **NVIDIA 在 nvfp4_gemm 上持续取得成功**：一位成员提交到 `nvfp4_gemm` 排行榜的结果在 **NVIDIA** 上成功运行，耗时 **10.9 µs**。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1451099487778115604)** (3 messages): 

> `Homelab Setup, GPU Training Differences, NVIDIA vs Other GPUs/NPUs, Intra-Node Interconnect Importance, NVIDIA's Software Role` 


- **Homelab 硬件连接**：一位成员正在使用 **GPU**、**Kubernetes** 和 **InfiniBand** 搭建多节点 **homelab**，并寻求入门建议。
   - 他们正在寻找构建小型多节点机器的资源。
- **NVIDIA 硬件黑客亮点**：一名学生询问为什么 **NVIDIA GPU** 和 **Google TPU** 在训练大型 AI 模型方面优于其他 GPU/NPU。
   - 其他 GPU（如 **Apple**、**AMD**、**Intel**）被认为更适合推理。
- **节点内互连的影响**：一位成员认为 **intra-node interconnect**（节点内互连）是 NVIDIA 性能的一个重要因素。
   - 他们指出 Google 拥有*极大的互连域*，而 NVIDIA 在 NVL72 之前一直使用 8 个。
- **NVIDIA 的软件栈**：一位成员表示 **software** 在 NVIDIA 的主导地位中起着重要作用。
   - 他们补充说 *nvidia has so much software being used everywhere*（NVIDIA 的软件随处可见）。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1451347020915933309)** (3 messages): 

> `AMD-MLA-Decode leaderboard, Reproducing Kernels, MI300 Availability, AMD Developer Cloud` 


- **AMD MLA Decode 上的 Kernel 复现任务**：一位成员正在寻求关于复现 [AMD-MLA-Decode 排行榜](https://www.gpumode.com/v2/leaderboard/463?tab=rankings) 中 kernel 结果的指导。
   - 具体来说，他们的目标是复制比赛中使用的确切环境。
- **MI300 在 Modal 设置中缺失**：用户注意到 Modal 上没有 **MI300** 可用于复现结果。
   - 他们询问是否有替代方案来匹配比赛环境。
- **AMD Cloud 额度带来希望**：一位成员建议利用 **AMD Developer Cloud**，它提供免费额度。
   - 这可能为搭建所需环境提供必要的资源。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1451046782560632972)** (2 messages): 

> `Trimul competition, Kernel Runtime, Geometric Mean, Standard Deviation` 


- **Trimul Kernel 运行时间旨在实现可复现性**：一位成员正尝试复现 Trimul 竞赛排行榜上的 kernel 运行时间，参考了 [Trimul 竞赛排行榜](https://www.gpumode.com/v2/leaderboard/496?tab=rankings)。
   - 他们提供了一个 [GitHub 上的 diff](https://github.com/gpu-mode/discord-cluster-manager/compare/main...LeoXinhaoLee:discord-cluster-manager-fork:xinhao/1217-reproduce-trimul)，展示了对 `discord-cluster-manager` 仓库所做的更改，并正在寻求关于复现结果和减少方差的建议。
- **几何平均值足够接近吗？**：用户寻求关于复现的几何平均值是否与原始结果足够接近的反馈。
   - 他们还好奇 Test 1-5（在具有相同 GPU 类型的不同 Modal 实例上）的方差是否在可接受且预期的范围内。
- **提供了标准差摘要**：标准差摘要可在 [此 Google 表格](https://docs.google.com/spreadsheets/d/1-lNEeNkf71YEabeX72jzZVhvUMBk2jcK2iPpuCTEDRg/edit?gid=114888003#gid=114888003) 中查看。
   - 用户询问关于在多个 Modal 实例中观察到的方差是否可接受且符合预期的意见。

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1450940682058338519)** (113 条消息🔥🔥): 

> `CuTeDSL L2 cache hint policies, Submission system timing out, Discord Bot usage for Submissions, MMA wrap optimization, TCGen05 instruction assistance` 


- **CuTeDSL 支持 L2 缓存规范**：一位成员询问如何在 **CuTeDSL** 中指定 **L2 cache hint policies**（类似于 **CUTLASS**），另一位成员确认了这一功能，并指向了 [CuTeDSL 仓库中的一个示例](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/atom.py#L199)。
- **提交系统饱受超时困扰**：多名成员报告了提交系统超时的问题，特别是在使用 **popcorn-cli** 时，一位成员报告了 *Stream ended unexpectedly without a final result or error event*（流意外结束，没有最终结果或错误事件）。
   - 该问题归因于 *高流量* 导致系统达到进程限制，但为 heroku 实例增加更多计算资源应该能解决此问题。
- **Discord Bot 回归**：一位成员询问如何通过 **Discord bot** 提交，另一位成员解释了在指定频道使用 `/leaderboard submit` 命令的过程，并说明它会提示输入文件。
   - 在 CLI 提交失败时，通过 Discord bot 提交似乎运行良好。
- **MMA Warp 优化失误**：一位成员在进行 warp 优化时寻求关于发布 **MMA** 指令的建议，尝试了 `land_id==0` 和 `warp_id == 0 && lane_id == 0` 等设置，尽管进行了调整但仍面临不匹配问题。
   - 另一位成员建议在从 **TMEM** 读取结果之前等待 **MMA** 完成，并推荐 DeepGEMM 仓库作为一个有价值的参考和示例实现 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh)。
- **TCGen05 指令协助**：一位用户在私信（DMs）中分享代码片段后，收到了关于解决发布 **TCGen05** 的 **MMA** 指令时不匹配问题的建议。
   - 共识是如果不执行任何 **rmem->tmem** 操作，则不需要 `tcgen05.wait::st.sync`，因为 `tcgen05.cp` -> `tcgen05.mma` 的顺序是正确的。


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1450973481180201042)** (6 条消息): 

> `Hand Pose Estimation, Wrist Cameras, NVIDIA Cosmos Predict, Mimic-Video Paper` 


- **Neurospike 分享手势姿态估计实验**：一位成员分享了他们手势姿态估计实验的**三个视频**，可在 [X.com](https://x.com/neurosp1ke/status/2001470564435636489) 查看。
- **寻找腕部摄像头**：一位成员询问了 **PI** 和其他人用于捕捉手部动作的具体**腕部摄像头**，并表示很难通过 Google 搜索找到相关信息。
   - 有人建议，让个人通过手持摄像头持续流式传输家庭活动，可以在 **Hugging Face** 上创建一个丰富的数据集。
- **Cosmos Predict 作为动作模型**：一位成员计划研究 **NVIDIA 的 Cosmos Predict**，考虑其作为 **action model** 的潜力。
- **Mimic-Video 宣称的高效率**：参考 [Mimic-Video 论文](https://www.arxiv.org/abs/2512.15692)，一位成员指出该论文声称比从 **VLM** 开始更有效率。


  

---

### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1450940782855983245)** (19 messages🔥): 

> `Contributing to Open Source Projects, Keeping up with SoTA Research, AI Infra Engineer Demand, Kernel Competitions, Parallel Programming Passion` 


- **社区贡献助力职业发展**：参与社区是职业生涯的**最佳起点**，尤其是参与具有 **SoTA 工作**内容的开源项目，能显著提升市场竞争力。
   - 一位成员分享了他们在 **LigerKernel** 的经验，这帮助他们结识了业界人士并可能因此获得工作机会。
- **AI Infra 工程师职位空缺**：对 **AI Infra 工程师**的需求持续上升，导致一些雇主开始面向更广泛的候选人进行招聘。
   - 有成员提到，位于印第安纳波利斯的 **Eli Lilly** 正在大量招聘 **HPC/GPU/Linux infra 工程师**，这得益于 **Mounjaro/Zepbound**（类似 Ozempic 的药物）带来的利润所转化的 AI 投资，待遇优厚且生活成本较低。
- **竞赛保持编码敏锐度**：成员们讨论了 Kernel 竞赛，包括 **AMD** 竞赛和当前的 **NVIDIA NVFP4** 竞赛，以此来保持技术敏锐度。
   - 有关竞赛的信息可以在 **#leaderboards** 频道中找到。
- **对并行编程的热情终有回报**：一位成员建议要“与并行编程共呼吸”，才能在 Kernel 开发中脱颖而出。
   - 他们建议重新审视每个编码项目并尝试将其并行化，使编写 Kernel 成为一种**本能**。
- **硬件系统背景反而不利？**：尽管拥有硬件系统、信号处理和数值分析背景，一位成员发现转向 **ML 系统**时在寻找机会方面面临挑战。
   - 他们将一个自定义的音视频语言模型移植到了 **vLLM**，应用了量化（Quantization），并在纽约的见面会上展示了工作，但发现人们对华丽的 Agent 演示比对模型架构和性能图表更感兴趣，因此他们打算**下次尝试在社交媒体上展示作品**。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1451312376707027205)** (3 messages): 

> `Pinned Chats, GPT-5.2-Codex, Chain-of-Thought Monitorability` 


- **置顶对话功能终于上线了！**：**Pinned Chats**（置顶对话）现已推送到 **iOS**、**Android** 和 **Web** 端，方便用户快速访问重要对话。
   - 在 Web 端点击对话旁的 "..."，或在移动端长按对话即可进行置顶。
- **GPT-5.2-Codex 树立编程新标准**：**GPT-5.2-Codex** 现已在 Codex 中可用，为现实软件开发中的 Agentic 编程和防御性网络安全树立了新标准，详见 [OpenAI 公告](https://openai.com/index/introducing-gpt-5-2-codex/)。
- **构建思维链可监控性框架**：开发了一个新的框架和评估套件，用于衡量 **Chain-of-Thought (CoT) 可监控性**，详见 [OpenAI 博客文章](https://openai.com/index/evaluating-chain-of-thought-monitorability/)。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1450940633492623360)** (163 messages🔥🔥): 

> `GPT-5.2 幻觉, Gemini 3.0 Flash, Sora 2 讨论, AI 随时间的连贯性, ChatGPT App Store` 


- **GPT-5.2 遭受幻觉和身份危机**：成员们报告称 **GPT-5.2** 有时会在某些对话中*忘记*自己的能力，即使用户明确指示其使用 Python 和 FFmpeg，也需要用户*通过说教让它记起来*，这表明该模型存在**幻觉和糟糕的初始路由（routing）**问题。
   - 有人指出，切换到**“思考（Thinking）”模式**有时会有所帮助，因为模型最初会否认拥有 Python 访问权限，然后又突然*发现*了它的功能。
- **Gemini 3.0 Flash 即将推出**：一位成员宣布 **Gemini-Flash-3-Image** 即将推出，尽管其命名方案被批评为*糟糕透顶*，另一位成员澄清说这是对当前 **Flash 版本从 2.5 到 3** 的升级。
   - 进一步解释称，**3-Pro** 为付费用户每天提供 **100 张图像**，而 **2.5-Flash** 提供 **1000 张**，因此推测新版本将提供 **1000 张**质量更高的图像。
- **成员寻求 Sora 2 讨论**：一位成员询问关于 **Sora 2** 的讨论，提到以前曾有一个拥有数千条消息的专用频道，对此另一位成员链接了通用的 [video-generation 频道](https://discord.com/channels/974519864045756446/1315696181451559022)。
   - 成员们对缺乏像 *ai-discussions* 这样专门用于 **Sora** 集中讨论的频道表示担忧。
- **AI 连贯性探索**：一位成员分享了他们对 **AI 随时间推移的连贯性**的探索，重点关注最小系统，其中观察和内部模型不断收敛而不会崩溃或相互矛盾，强调的是**反馈下的自洽性**而非正确性。
   - 他们正在原型化一个微型内核式框架，专注于**长程连贯性（long-horizon coherence）、模型漂移（model drift）和 Agent 自我修正**，并寻求他人的类似想法。
- **ChatGPT App Store 正式上线**：据宣布，开发者现在可以向 **ChatGPT** 提交应用，因为 **App Store 已上线**，并可通过“设置”中的“Apps”菜单访问，这引发了关于潜在**安全噩梦**的讨论。
   - 一位成员补充说，App Store 已在网页端和移动端的 **ChatGPT** 中上线。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1451012401989288008)** (8 messages🔥): 

> `Gemini 3.0 Flash, GPT 5.2 High, deepseek r3.2, API 日期` 


- **Gemini 3.0 Flash 击败 GPT 5.2 High**：一位成员报告称，**Gemini 3.0 Flash** 在其物理和 ML 测试套件中显著击败了 **GPT 5.2 High**，达到了 **90%** 的成功率，而 GPT 5.2 High 则在一系列测试中失败。
   - 该用户表示愿意通过私信分享他们的基准测试，并表示 *“‘High’ 推理能力被 Flash 模型击败，这不是我对 OpenAI 的预期”*。
- **Deepseek r3.2 更便宜的模型**：一位成员指出 **deepseek r3.2** 是最便宜的。
   - 没有进一步的讨论来阐明这一说法。
- **GPT 5.2 知道日期？**：一位成员质疑 **GPT 5.2** 是如何知道当前日期的，因为这不应该由 **API** 附加。
   - 没有进一步的讨论来阐明这一说法。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1451112675999219724)** (1 messages): 

> `模型溯源, AI 写作风格, 缺乏溯源注释标签` 


- **AI 写作风格辨识度太高**：一位成员表示，在没有溯源注释标签（provenance annotation tags）的情况下，他们能察觉到人们何时使用模型来润色回复。
   - 他们将 AI 写作风格描述为*像是一个拿着法学学位胡说八道的兄弟会男孩*。
- **模型溯源至关重要**：一位成员强调了在使用 AI 模型时溯源注释标签的重要性。
   - 这确保了透明度，并避免了对内容来源的误导。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1451112675999219724)** (1 messages): 

> `模型检测, 溯源注释, 回复润色` 


- **模型听起来像拿着法学学位的兄弟会男孩**：一位成员认为，由于模型具有一致且可辨识的风格，他们可以检测出人们何时在没有**溯源注释标签**的情况下使用模型润色回复。
   - 该成员描述模型的风格听起来像是 *“同一个拿着法学学位胡说八道的兄弟会男孩”*。
- **检测 AI 需要溯源注释**：讨论集中在需要**溯源注释标签**来识别 AI 润色的回复。
   - 如果没有这些标签，识别 AI 的使用就只能依靠检测风格模式，而这可能是主观的。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1451016087557767263)** (103 条消息🔥🔥): 

> `轻量级 Vision Transformer 模型、结构化数据提取的模型选择、Fill Mask 技术、LLM 中的前向传播与 Steering、Kaggle 运行时间中断` 


- **Vision Transformer 训练时间紧迫**：一位成员正在寻找能在 **Kaggle 9-12 小时会话限制**内，在 **ImageNet 数据集**上完成训练的轻量级 **Vision Transformer** 模型。
   - 其他人指出，如果没有大量的 GPU 资源，在那个时间范围内训练 **ImageNet** 是非常有挑战性的。
- **筛选用于结构化数据提取的小型模型**：一位成员询问了关于**结构化数据提取**的模型选择，提到使用了**禁用 Instruct 的 phi 3.5 mini instruct**。
   - 另一位成员建议，一个 **1.5B Instruct LLM**（如使用 **Q4_K_M** 的 **Qwen 2.5 1.5B Instruct**）应该足够了，并且**预处理和后处理是必要的**。
- **通过 Model Steering 解锁 LLM 内部机制**：一位成员讨论了一种受 [Mentat's YC Launch](https://www.mentat.ai/) 启发的技术，用于引导 (steer) LLM，从而消除对重训练和数据的需求。
   - 他们指出 hook 函数不会像示例中那样保持微小，因为它将是创新发生的地方，用于寻找合适数量的向量和温度。
- **HuggingFace Hub 频道压缩带来清爽**：一位成员注意到频道数量减少，另一位成员解释说许多频道被压缩以减少噪音。
   - 压缩重点针对无监管、低活跃度或重复的频道，例如 <#1019883044724822016>。
- **应对 RAG 系统的隐藏约束**：一位从事 **RAG 系统**工作的工程师分享了一篇文章，讨论了隐藏的约束以及超越未经工程化知识的必要性。
   - 该文章发表在 [Medium](https://medium.com/@rradhakr/the-rag-systems-hidden-constraint-why-we-must-move-beyond-unengineered-knowledge-part-1-3df60c65ee68) 上，正在寻求社区反馈。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1450994971988725872)** (2 条消息): 

> `Android 语音助手、Gemini 1.5 Flash、VoxCPM 1.5、Apple Neural Engine` 


- **由 Gemini Flash 驱动的 Android 语音助手亮相！**：一位成员宣布创建了一个使用 **Gemini 1.5 Flash** 构建的 [Android 语音助手](https://www.strawberry.li/)，并邀请其他人进行测试。
   - 他们还发布了一个 Beta 版本，该版本使用 **VoxCPM 1.5** 生成语音，并在 **Apple Neural Engine** 上运行。
- **语音助手利用 VoxCPM 和 Apple Neural Engine**：[Android 语音助手](https://www.strawberry.li/) 的 Beta 版本现在使用 **VoxCPM 1.5** 生成语音。
   - 此版本针对 **Apple Neural Engine** 进行了优化，承诺在 Apple 设备上提供更强的性能。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1451208853818445864)** (4 条消息): 

> `Smolcourse 延迟、AI 学习资源` 


- **Smolcourse 进度停滞，订阅担忧引发关注**：成员们正在寻求 **smolcourse** 的更新，指出自上次更新以来已延迟了一个月，并对支付了 **两个月的 Pro 订阅** 但课程使用有限表示担忧。
   - 延迟最初归因于 Hugging Face 使用的一个库正在进行改进，初步预计课程将于 12 月恢复。
- **备受推崇的顶级教程 YouTube 频道**：成员们建议在 YouTube 上查看 **Andrej Karpathy**、**Deeplearning.AI** 和 **Anthropic AI** 的内容以学习 AI 概念。
   - 他们还推荐了 **Mervin Praison** 和 **Umar Jamil** 进行 AI 代码实操。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1450983924368015462)** (89 messages🔥🔥): 

> `Exa AI People Search, Michael Truell and John Schulman LLM discussion, OpenAI potential $750B valuation, Pieter Abbeel Amazon AGI Head, Tomo AI` 


- **Exa AI Labs 发布人物搜索 (People Search)**：Exa AI Labs 推出了全新的 **People Search** 功能，允许用户使用由微调后的 Exa embeddings 驱动的**混合检索 (hybrid retrieval)**，对超过 **10 亿**个人进行语义搜索；点击[此处](https://x.com/exaailabs/status/2001373897154007390?s=46)查看发布推文。
- **OpenAI 考虑进行大规模融资**：OpenAI 正在与投资者就新一轮融资进行早期讨论，预计估值约为 **7500 亿美元**，并希望筹集数百亿美元，根据 [Katie Roof 的推文](https://x.com/katie_roof/status/2001453561910358474?s=46)，金额可能高达 **1000 亿美元**。
- **Pieter Abbeel 加入 Amazon 担任 AGI 负责人**：根据[这条推文](https://x.com/bookwormengr/status/2001353147055509881?s=46)，以机器人在 DeepRL 领域的工作而闻名的加州大学伯克利分校教授 **Pieter Abbeel** 已被任命为 Amazon 的新任 **AGI Head**。
- **FunctionGemma：在本地调用函数**：Google 推出了 **FunctionGemma**，这是一个全新的 **2.7 亿**参数模型，针对函数调用 (function calling) 进行了优化，旨在直接在手机和浏览器等设备上运行；更多信息请见[此处](https://x.com/osanseviero/status/2001704036349669757)。
- **T5Gemma 2：多模态、多语言模型发布**：**T5Gemma 2** 是基于 Gemma 3 构建的新一代 encoder-decoder 模型，提供高达 **4B-4B** 的尺寸，并支持 **140 种语言**，详情见[这条推文](https://x.com/osanseviero/status/2001723652635541566?s=46)。


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1451148723856867361)** (2 messages): 

> `vLLM Router, Intelligence Control Plane, Ollama / vLLM routing through semantic router` 


- **vLLM Router 使用 Rust 进行负载均衡**：**vLLM Router** 专为 **vLLM 集群**打造，使用 **Rust** 编写，支持用于 **KV 局部性 (KV locality)** 的一致性哈希、二次幂选择 (power-of-two choices)、重试/退避、熔断器、**k8s 服务发现**以及 **Prometheus 指标**。
   - 它专为 **P/D 解耦 (P/D disaggregation)** 设计，具有独立的 worker 池和路由策略，以保持吞吐量并降低尾部延迟。
- **智能控制平面管理安全与记忆**：**vLLM + AMD** 预览了一个 **Semantic Router** 框架——管理输入、输出和长期状态，重点关注大型 Agent 系统中的安全与记忆。
   - 这标志着迈向**“智能控制平面 (Intelligence Control Plane)”**的一步。
- **Semantic Router 圣诞愿望清单**：一名成员表达了希望通过语义路由 (**BERT**) 进行 **ollama / vllm 路由**的愿望，以实现 **DLP / 模型选择 Token 优化**。
   - 他们希望频道中有人正在构建此功能，并期待在假期期间有时间研究它。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1450984775526645881)** (8 messages🔥): 

> `Black Forest Labs FLUX.2 Launch, xAI Grok Voice Agent API` 


- **Black Forest Labs 发布 FLUX.2 [max] AI 模型**：Black Forest Labs 宣布推出 **FLUX.2 [max]**，这是他们迄今为止质量最高的 AI 模型，正如在 [X](https://xcancel.com/bfl_ml/status/2000945755125899427?s=46) 上宣布的那样。
- **xAI 发布 Grok Voice Agent API**：xAI 宣布推出 **Grok Voice Agent API**，使开发者能够创建能够说多种语言、使用工具并搜索实时数据的语音 Agent，消息发布在 [X](https://xcancel.com/xai/status/2001385958147752255) 上。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1450962816478609468)** (14 messages🔥): 

> `GPT-2 interpretability, 3D visualization of residual stream, SOTA Model Performance, Claude Opus 4.5 mistakes, Neuronpedia` 


- **3D 可视化工具旨在深入探索 GPT-2**：一名成员正在构建一个 3D 全息交互式应用，用于*查看 GPT2 small 124m LLM 的每个节点内部*，并寻求对其项目的反馈。
   - 另一名成员分享了他们自己的项目链接，一个 [residual stream 的 3D 可视化](https://aselitto.github.io/ResidStream/)，寻求关于该可视化是否符合用户心理模型的反馈。
- **GPT-2 可视化引发对 Neuronpedia 的关注**：一名成员建议 GPT-2 可视化项目与 **Neuronpedia** 相关，并建议将其转发到 mech interp discord。
   - 他们分享了 [mech interp discord](https://discord.com/channels/729741769192767510/732688974337933322/1425176182965665852) 的链接以便进一步讨论。
- **SOTA 模型性能几乎难以区分**：成员们发现 SOTA 模型在性能和智能方面几乎无法区分。
   - 一名成员指出，**Claude Opus 4.5** 尽管承认了错误，但仍会重复错误，表现仅略好于 **GPT 3.5**。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1451101109056962561)** (3 messages): 

> `Speech/NLP Research Collaboration, AI Research, NLP` 


- **成员寻求语音/NLP 研究协作**：一名成员在阅读了[这篇论文](https://arxiv.org/abs/2512.15687)的摘要后，表达了在 **speech/NLP** 研究方面进行协作的兴趣。
   - 该成员请其他有兴趣的成员与其联系。
- **AI 研究人员寻找合作**：一名成员寻求在语音/NLP 任何主题上的研究合作。
   - 该用户向任何感兴趣的成员发出了请求。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1450970088575340678)** (4 messages): 

> `Anthropic's weight masking, Gemma 3 extreme activations, Adam's fault` 


- **Anthropic 屏蔽危害**：Anthropic 发布了一篇关于 [Selective Gradient Masking (SGTM)](https://alignment.anthropic.com/2025/selective-gradient-masking/) 的论文，重点是通过隔离/删除特定权重来**遗忘危险知识**，这在通用知识上会产生 **6% 的计算开销**。
   - 一名成员的 patching 实验显示只有 **93% 的恢复率**，这表明 *superweight 是电力/ML 中等效的最小电阻路径*，因为模型可以构建分布式电路绕过，而不会完全崩溃。
- **Gemma 3 的权重受到关注**：讨论围绕 **Gemma 3** 可能具有极端激活或权重以在模型中容纳更多信息展开。
   - 这究竟是训练伪影还是为了提高信息密度而进行的有意设计，仍有待观察。
- **Adam 成了背锅侠**：一名成员开玩笑地将讨论的现象归咎于 **Adam**（可能指 Adam 优化器）。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1451246848600707216)** (4 messages): 

> `long range view synthesis, novel view synthesis, parallax effect, depth estimate` 


- **对长距离视角合成结果的兴趣**：一名成员对 **long range view synthesis** 的结果表示兴趣，特别是当目标摄像机角度远离输入角度时，不需要是单目的，并且应该利用物体的对称性。
   - 他们澄清说，“远”是指偏离不同角度的大量旋转度数，而不是物理距离。
- **远距离 vs 近距离的新视角合成**：另一名成员认为，合成**远距离的新视角**可能比近距离更容易，因为**视差效应 (parallax effect)** 会更小。
   - 原成员建议在多视角输入中寻找一个平衡点，在**深度估计 (depth estimation)** 中的**视差**与用于跨视角配准不同物体点的足够特征点之间取得平衡。


  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1451362312630829189)** (1 messages): 

> `Custom Cross Entropy Function, Backwards Pass` 


- **绕过反向传播：Cross Entropy 技巧！**：一位成员询问如何在不重写反向传播（backwards pass）的情况下，将仓库中的 cross-entropy 函数替换为自定义函数。
   - 他们建议，可以通过编写一个继承自旧 cross-entropy 函数的新类，然后[仅重写前向传播（forward pass）](https://pytorch.org/docs/stable/notes/extending.html)来实现。
- **让反向传播变得简单**：一位成员询问是否可以将仓库中列出的 cross-entropy 函数替换为个性化版本，从而避免完全重写反向传播。
   - 另一位成员指出，[PyTorch 文档](https://pytorch.org/docs/stable/notes/extending.html)提供了关于扩展现有类的指导，可能允许仅重写前向传播，从而简化流程。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1450995399803404308)** (19 messages🔥): 

> `Mojo GPU usage, Rust GPU capabilities, Mojo std::offload` 


- **Mojo 显式地将代码启动到 GPU**：一位成员询问 **Mojo** 是否可以简单地使用一个属性（attribute）在 **GPU** 上运行函数，另一位成员澄清说需要显式地在 **GPU** 上启动。
   - 他们补充道，*只要不进行系统调用（syscalls），就不需要属性（尽管它需要作为单通道/single lane 启动）*。
- **讨论 Rust 的单函数 SIMD/GPU 能力**：一位成员引用了最近关于 **Rust** 能力的公告：使用单个函数和 `std::batching` 获取 **SIMD/fused** 版本，使用 `std::autodiff` 对其进行微分，以及使用 `std::offload` 在 **GPU** 上运行生成的代码，并询问 **Mojo** 是否已经支持这些功能。
   - 他们提供了一个与该公告相关的 [Reddit 帖子链接](https://www.reddit.com/r/rust/comments/1pp3g78/project_goals_update_november_2025_rust_blog/)。
- **Mojo 提供 std::offload，Autodiff 稍后推出**：一位成员解释说，**Mojo** 拥有等效的 `std::offload`，并且不需要使用 batching 来融合算子（fuse ops）。
   - 他们澄清说，*autodiff 是已经讨论过但留待以后处理的事情*，并补充说目前构建支持 **autodiff** 的功能会很困难。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1451004910920274098)** (6 messages): 

> `GPU issues with LLM build in MAX, C interop ideas from Rust, Mojo's array access quirks` 


- **Modular 调查 LLM 构建中的 GPU 故障**：一位用户报告说，即使在禁用 **GPU** 的情况下，在 **MAX** 中构建 **LLM** 仍存在持续问题，并在 [Modular 论坛上分享了细节](https://forum.modular.com/t/build-an-llm-in-max-from-scratch/2470/9)。
   - 一位 Modular 团队成员确认他们正在调查，怀疑是 **API 回归**或特定设备的问题。
- **Rust 的 C 互操作启发了 Mojo 的想法**：一位成员分享了一个关于 **C interop** 的 [Rust Zulip 讨论](https://rust-lang.zulipchat.com/#narrow/channel/131828-t-compiler/topic/pre-MCP.20vibe.20check.3A.20-Zinternalize-bitcode/near/546294314)，建议类似的方法可能会使 Mojo 受益。
   - 其想法是探索 Mojo 如何潜在地利用 Rust 的策略来改进与 C 代码的互操作性。
- **Mojo 数组访问允许越界索引**：一位成员质疑为什么当 `x = 2.0` 时，Mojo 允许像 `x[5]` 这样的越界索引。
   - 未给出解释。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1451011195434696905)** (17 条消息🔥): 

> `Aider vs Aider-ce, OpenCode Accuracy vs Aider, Context Management and Token Efficiency, Task Bundling with Context` 


- **Aider 坚持结对编程**：Aider 被设计为一个采用 **human-in-the-middle**（人工干预）方式的 **pair programmer**（结对编程器），这与 **aider-ce** 分支所采取的 Agent 路线形成对比。
   - 一位成员对 **OpenCode** 或其他 Agent 系统能否达到相当的准确率和 Token 效率表示强烈保留意见。
- **OpenCode vs Aider**：一位用户将 **OpenCode** 与 **Aider** 进行了横向对比，并对 **准确率** 提出了担忧，他们将其归因于 **Aider** 的非 Agent 设计及其对上下文的控制。
   - 他们认为，由于不存在误解导致的螺旋效应，以及高效的上下文管理（通过 **/add** 或 **/load**），模型表现更好，并指出超过 **20-30k tokens** 甚至会对 **Opus 4.5** 产生负面影响。
- **上下文管理和 Token 效率是关键**：该用户建议，通过 **/add** 或 **/load** 使用 *最小上下文*，就像是在使用下一代模型，而相比之下，让 **CLI** 或 **IDE** 控制上下文的人，由于在上下文窗口中添加的 Token 更少，模型性能反而更好。
   - 他们链接了一篇 [Chroma 关于 context-rot（上下文腐烂）的研究报告](https://research.trychroma.com/context-rot)，指出由于上下文管理效率低下而花更多钱却获得更差性能的讽刺现象。
- **结合必要上下文的任务捆绑**：一位成员询问是否可以将所有内容分解为 **tasks**（任务），然后将必要的上下文与任务捆绑在一起。
   - 该成员补充说，他们经常关注 Aider，因为他们很喜欢它，但不确定如何充分利用所有功能。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1451263447281963173)** (3 条消息): 

> `Gemini 3 Flash, aider configurations, Litellm updates` 


- **Gemini 3 Flash 大获全胜！**：许多人称赞 **Gemini 3 Flash** 是最好的编程模型，并推荐使用它，同时链接到了 [blog.brokk.ai](https://blog.brokk.ai/why-gemini-3-flash-is-the-model-openai-is-afraid-of/)。
   - 评论者强调该模型表现出色，但建议用户 *禁用 thinking（思考）模式* 以防止速度变慢，特别是在 **aider** 中。
- **Thinking 配置困惑**：一位成员承认不确定自己目前的 *thinking* 配置。
   - 他们注意到 **Litellm** 对专业模型默认设置为 *low*，但 **Gemini 3 Flash** 尚未包含在他们的 **Litellm** 版本（**aiderx**）中。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1450956459197206621)** (6 条消息): 

> `Office Hours Recording, Sam's IOU Scheme, CUDA Performance on Linux` 


- **请求 Office Hours 录音**：一位成员询问是否可以上传最近一次 Office Hours 的录音。
- **Sam 的 IOU 计划被曝光？**：市场正在察觉到 **Sam 的 IOU 计划**，如[这段 YouTube 视频](https://www.youtube.com/watch?v=5DZ7BJipMeU)所示。
- **优化 CUDA 性能**：一位成员分享了在 Linux 上优化 **CUDA** 性能的 2 个技巧。
- **多 GPU VRAM 降频 Bug**：在多 GPU 或部分卸载配置中，将 `CUDA_DISABLE_PERF_BOOST=1` 与 **llama.cpp** 配合使用时，VRAM 可能会因 GPU 利用率低而降频。
- **禁用 P2 状态以提升 Token 生成速度**：根据 Nvidia 员工的建议，为 CUDA 应用禁用 **P2 state**（配置 `0x166c5e=0`）可以将 Token 生成速度提升几个百分点，详见 [此 GitHub 评论](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/333#issuecomment-3669477571)。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1451170957841530880)** (3 条消息): 

> `Open Source Server for Minos-v1, VLLM, SGLang` 


- **Minos-v1 寻求开源部署服务器**：一位成员询问有关部署 [NousResearch/Minos-v1](https://huggingface.co/NousResearch/Minos-v1) 模型的开源服务器实现。
- **VLLM 和 SGLang 挺身而出**：另一位成员建议使用 **VLLM** 或 **SGLang** 来支持分类器的服务。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1451229645977551078)** (3 条消息): 

> `LLM finetuning, Electronic schematics dataset` 


- **LLM Finetuning 引起关注**：成员们对快速 **finetuning LLMs** 以使其以特定方式运行的前景感到兴奋。
- **新的电子原理图数据集出现**：一位成员分享了一个用于训练 LLM 创建 **电子原理图 (electronic schematics)** 的新数据集；点击此处查看 [数据集](https://huggingface.co/datasets/bshada/open-schematics/discussions)。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1451229645977551078)** (3 条消息): 

> `LLM finetuning, Electronic schematics dataset` 


- **快速 LLM Finetuning 看起来很有前景**：一位成员指出，快速 **finetuning of LLMs** 以特定方式行动看起来非常棒。
   - 他们个人还没有机会尝试。
- **分享了用于训练 LLM 电子原理图的数据集**：一位成员分享了一个用于训练 **LLMs 创建电子原理图** 的 *惊人数据集*。
   - 该 [数据集](https://huggingface.co/datasets/bshada/open-schematics/discussions) 已在 Hugging Face 上发布。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1451005867339157506)** (12 条消息🔥): 

> `GEPA Optimization, Robots building robots, TreeOfThought Module, dspy.Refine feedback, GEPA Definition` 


- **Genetic-Pareto 优化令人惊叹**：一位成员感叹，运行他们的第一次真实 **GEPA 优化** 就像魔法一样，表现出对该技术的热情。
   - 另一位成员开玩笑地补充道，制造机器人的机器人可能会到来，但目前他们正在使用 **DSPy**。
- **Genetic Prompt Evolution (GEPA) 定义**：**GEPA (Genetic-Pareto)** 是一种反思型优化器，它能自适应地演化任意系统的文本组件，并使用标量分数和文本反馈来指导优化过程，详见 [“GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning”](https://arxiv.org/abs/2407.19457)。
   - 从本质上讲，它是一个使用另一个 AI 对 AI 提示词进行遗传修改的优化器，根据度量方法选择最佳更改，实际上是 AI 构建 AI。
- **GEPA 资源分享**：一位成员分享了相关资源，包括 [关于 GEPA 的 DSPy 教程](https://dspy.ai/tutorials/gepa_ai_program/)、[关于使用优化器的博客文章](https://thedataquarry.com/blog/learning-dspy-3-working-with-optimizers/#option-2-gepa) 以及 [关于 GEPA 实践的 Medium 文章](https://medium.com/data-science-in-your-pocket/gepa-in-action-how-dspy-makes-gpt-4o-learn-faster-and-smarter-eb818088caf1)，以帮助他人学习 GEPA。
   - 他们指出，自该博客文章撰写以来，某些语法可能已经发生了变化。
- **缺失 Tree of Thought (ToT) 模块**：一位成员询问为什么 **DSPy** 中还没有直接的官方 **Tree of Thought (ToT) 模块**。
   - 关于此话题没有进一步的讨论或回复。
- **关于 `dspy.Refine` 中自定义反馈的讨论**：一位成员询问在使用 `dspy.Refine` 时如何手动指定反馈类型。
   - 他们提到在评估器循环中使用自定义模块，并想知道是否遗漏了关于 `Refine` 的某些功能。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1450993229586628699)** (10 messages🔥): 

> `In-Context Learning Research, Draft Model Optimization, Training Cluster Pipelining, Vast.ai Issue, Nvidia's Brev vs Runpod` 


- **关于 In-Context Learning 的 9000IQ 研究走红**：一名成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=q-yo6TPRPVk) 链接，展示了关于 *In-Context Learning 的 9000IQ 研究工作*，并建议其他人*准备好爆米花花一个下午的时间观看*。
- **Draft Model 通过并行处理提升效率**：一名成员提议使用 **Draft Model** 来猜测分布在多个 GPU 或服务器上的大型模型各部分的输出，从而实现并行处理并提高系统利用率。
   - 该建议涉及丢弃与 Draft 差异显著的响应，并在这些情况下恢复到正常处理，旨在更高效地在系统中进行批量运行。
- **Memory Bandwidth 瓶颈制约大型训练集群**：在大型训练集群中，机器内部和机器间的 Memory Bandwidth 是最大的瓶颈，需要快速的数据访问以减少停顿，而高效的 Pipelining 可以提高利用率。
   - 训练并不总是*纯粹*的，它涉及多个文档的前向和后向传播，并结合梯度或权重，在某些情况下这等同于顺序处理。
- **Vast.ai 模板故障困扰用户**：一名成员报告了 **Vast.ai** 的一个问题，即它启动时使用了错误的模板，导致 init 脚本无法按预期运行。
   - 一名成员建议尝试 **Nvidia's Brev** 作为潜在解决方案，但指出 **Runpod** 可能更好。
- **Rsync 救场**：一位成员通过使用他们的 **PyTorch** 模板并使用 rsync 同步所需的一切，绕过了 Vast.ai 上的模板问题。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1450986276953264168)** (1 messages): 

> `ARC-AGI Benchmark, Toolathlon Benchmark, Training Data Mix` 


- **ARC-AGI 成绩跃升归因于训练数据**：近期各模型在 **ARC-AGI Benchmark** 上的性能提升，很可能是由于训练数据中包含了 Benchmark 本身，而非出现了隐藏的技术突破。
   - 用户指出，在这些类型的 Benchmark 上，通常会观察到较小规模模型具有更好的泛化能力。
- **Toolathlon 的提升源于训练侧重**：**Toolathlon Benchmark** 分数的显著提高可能源于训练组合（Training Mix）中对该能力的更多侧重。
   - 这种调整可能确保了更可靠的 Tool Calling，即使在参数较少的情况下也是如此。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1450968554814832732)** (6 messages): 

> `Kimi K2, Moonshot AI, Free Models` 


- **关于 Kimi K2 思考能力的文章太棒了！**：一名成员分享了一篇关于 **Kimi K2** 及其思考能力的 [DigitalOcean 文章](https://www.digitalocean.com/community/tutorials/kimi-k2-moonshot-ai-agentic-open-weight-model)。
   - 该成员表达了对这篇文章的兴奋之情。
- **Kimi K2 有新变化！**：一名成员分享了一张图片，显示 **Kimi K2** 有了新变化。
   - 该成员表达了兴奋，说 *“这太棒了！！”*。
- **免费 Kimi 模型将在 1 个月后重置**：一名成员分享了一张图片，显示**免费 Kimi 模型**将在一个月后重置。
   - 该成员表示不确定，称 *“所以看起来它们现在会在 1 个月后重置——我猜？我是指免费的那些？”*


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1450958376350515265)** (5 messages): 

> `DNS issues, Chat image limits, Manus revenue` 


- **DNS 灾难延迟部署！**：一名用户遇到了 **Cloudflare** 的 **DNS 问题**，该问题已持续超过 4 天，用户对缺乏客户服务感到沮丧。
   - 用户表示 *除了说直接私信我们外，客服没有任何回应*。
- **聊天中的图像限制纠纷！**：一名用户质疑聊天模式的图像限制以及 **Pro/付费** 用户与 **免费** 用户之间的差异。
   - 他们抱怨 **免费用户** 的*智能水平已经显著降低*，并对额外施加图像限制表示不满，并列举了 **DeepSeek** 和 **Gemini** 作为没有此类限制的例子。
- **Manus 赚了数百万！**：一名用户分享了一篇文章，报道 **Manus** 的收入已达到 **1 亿美元**。
   - 链接的 [SCMP 文章](https://www.scmp.com/tech/tech-trends/article/3336925/manus-hits-us100-million-revenue-milestone-global-competition-ai-agents-heats) 指出，这一里程碑是在全球 **AI Agent** 竞争升温之际达成的。


  

---

### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1451247243570188311)** (4 条消息): 

> `Node Server 中的 MCP Prompts，ChatGPT App 提交` 


- **MCP Prompts: Node Server 激活**：一名成员询问如何在 **Node server** 中启用 **MCP prompts**，并指出虽然与 orchestrator 的注册成功了，但微服务中的 prompts 选项卡处于禁用状态。
   - 另一名成员将问题重定向到 **#mcp-support** 频道，或建议提交 issue 以获取详细帮助，并澄清当前的 Discord 专注于贡献者协作而非技术支持。
- **ChatGPT App 提交要求**：一名新成员询问 **ChatGPT apps** 提交时是否需要 UI，还是仅提供 **MCP server** 就足够了。
   - 未收到回复。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1451359133780676679)** (3 条消息): 

> `JIT 重构，固件处理，RDNA3 汇编后端` 


- **Claude 未能通过 JIT 重构测试**：一名成员多次尝试让 **Claude** 进行 **JIT refactor** 但均未成功，称其“确实缺乏品味”，并计划手动完成。
   - 该重构涉及完善 **schedulecache** 并让 **JIT** 运行一些 **schedulecaches**。
- **Tinygrad 处理固件相关内容**：一名成员报告称正在处理固件相关工作，通过在 **Linux** 上使用一个完整的模拟器来模拟一个虚假的 **USB device**，并将所有内容传递给固件。
   - 该模拟器正将所有内容传递给 **firmware**。
- **RDNA3 汇编后端已足够好用**：一名成员报告称编写了一个带有寄存器分配器的 **RDNA3 assembly backend**，其性能足以运行带有 **128 accs** 的 gemms。
   - 该后端已在 [GitHub](https://github.com/tinygrad/tinygrad/pull/13715) 上发布。


  

---


---


---