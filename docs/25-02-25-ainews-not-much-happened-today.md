---
companies:
- anthropic
- perplexity-ai
- amazon
- google-cloud
- deepseek_ai
date: '2025-02-26T02:19:12.201709Z'
description: '**Claude 3.7 Sonnet** 展示了卓越的编程和推理能力，在 **SciCode** 和 **LiveCodeBench**
  等基准测试中表现优于 **DeepSeek R1**、**O3-mini** 和 **GPT-4o**。该模型已在 **Perplexity Pro**、**Anthropic**、**Amazon
  Bedrock** 和 **Google Cloud** 等平台上线，定价为每百万 token **3美元/15美元**。其核心特性包括 **64k token
  的思考模式**、**200k 上下文窗口**，以及基于命令行界面（CLI）的编程助手 **Claude Code**。与此同时，**DeepSeek** 发布了
  **DeepEP**，这是一个专为 MoE（混合专家）模型训练和推理优化的开源通信库，支持 **NVLink**、**RDMA** 和 **FP8**。这些更新突显了编程
  AI 和高效模型训练基础设施方面的显著进展。'
id: fb8f064c-a654-4286-a905-a014e041c5ad
models:
- claude-3.7-sonnet
- claude-3.7
- deepseek-r1
- o3-mini
- deepseek-v3
- gemini-2.0-pro
- gpt-4o
- qwen2.5-coder-32b-instruct
original_slug: ainews-not-much-happened-today-6477
people:
- skirano
- omarsar0
- reach_vb
- artificialanlys
- terryyuezhuo
- _akhaliq
- _philschmid
- catherineols
- goodside
- danielhanchen
title: 今天没发生什么特别的事。
topics:
- coding
- reasoning
- model-benchmarking
- agentic-workflows
- context-window
- model-performance
- open-source
- moe
- model-training
- communication-libraries
- fp8
- nvlink
- rdma
- cli-tools
---

<!-- buttondown-editor-mode: plaintext -->**平静的一天。**

> 2025年2月24日至2月25日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**220** 个频道和 **5949** 条消息）。预计节省阅读时间（以 200wpm 计算）：**503 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

您应该关注 DeepSeek 的 [#OpenSourceWeek](https://x.com/search?q=%23OpenSourceWeek%20from%3Adeepseek_ai&src=typed_query&f=top)，但到目前为止发布的内容尚未达到我们头条新闻的标准。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

**Claude 3.7 Sonnet 发布与性能**

- **Claude 3.7 Sonnet 在编程和推理方面表现出色**：[@skirano](https://twitter.com/skirano/status/1894171599508537620) 强调，**Claude 3.7 Sonnet 配合 Claude Code** 可以一次性生成整个**“玻璃感”设计系统**，包括所有组件。[@omarsar0](https://twitter.com/omarsar0/status/1894164720862523651) 通过创建一个 **Attention 机制模拟器**展示了 **Claude 3.7 的推理和编程能力**。[@reach_vb](https://twitter.com/reach_vb/status/1894132284711649463) 指出，**Claude 3.7** 在非思考模式下击败了 **DeepSeek R1**，并与 **o3-mini (high)** 持平，预计在思考模式下会有强劲表现。[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1894437867914682764) 将 **Claude 3.7 Sonnet** 评测为**编程表现最好的非推理模型**，在他们的编程评估 **SciCode 和 LiveCodeBench** 中超越了 **DeepSeek v3、Gemini 2.0 Pro 和 GPT-4o**。[@terryyuezhuo](https://twitter.com/terryyuezhuo/status/1894138361654526171) 分享了 **BigCodeBench-Hard 结果**，显示 **Claude-3.7（无思考模式）** 达到了 **33.8% 的完成率**，与 **Qwen2.5-Coder-32B-Instruct** 相当，并优于 **o3-mini** 和 **o1-2024-12-17**。
- **Claude 3.7 Sonnet 在多个平台上线**：[@perplexity_ai](https://twitter.com/perplexity_ai/status/1894186614827504054) 宣布 **Claude 3.7 Sonnet 已在 Perplexity Pro 上线**，并指出其在 **Agent 工作流和代码生成**方面的改进。[@_akhaliq](https://twitter.com/_akhaliq/status/1894148292440666616) 确认 **Claude 3.7 Sonnet 已在 Anychat 上线，并带有 Coder 模式**。[@_philschmid](https://twitter.com/_philschmid/status/1894301548101980532) 提到该模型已在 **Anthropic、Amazon Bedrock 和 Google Cloud** 上可用，价格保持不变，为每百万 **Input/Output Token $3/$15**。
- **Claude 3.7 Sonnet 的“思考模式”与上下文窗口**：[@_philschmid](https://twitter.com/_philschmid/status/1894301548101980532) 强调了 **Claude 3.7** 的 `<thinking>` 模式支持高达 **64k Token** 并提供**推理 Token 显示**，同时具备 **200k 上下文窗口**和 **128k 输出 Token 长度**。[@Teknium1](https://twitter.com/Teknium1/status/1894319586428031354) 赞扬了 **Claude** 中**可切换的思考模式**。
- **Claude 3.7 Sonnet 的编程工具 “Claude Code”**：[@_philschmid](https://twitter.com/_philschmid/status/1894301548101980532) 介绍了 **Claude Code**，这是一个**基于 CLI 的编程助手**，能够读取、修改文件并执行命令。[@catherineols](https://twitter.com/catherineols/status/1894149904282661366) 将 **Claude Code** 描述为比其他工具更具自主性，能够决定运行测试和编辑文件。[@goodside](https://twitter.com/goodside/status/1894235937074282793) 预览了 **Claude Code**，指出它能查看文件、编写 Diff、运行命令，就像一个没有编辑器的轻量级 Cursor。
- **Claude 3.7 Sonnet 价格对比**：[@_philschmid](https://twitter.com/_philschmid/status/1894154634173845876) 指出 **Claude 3.7** 的价格维持在**每百万 Input/Output $3/$15**，这使其比 **Gemini 2.0 Flash 贵 30 倍**，比 **Open o3-mini 贵约 3 倍**。

**DeepSeek 和 Qwen 模型更新及开源发布**

- **DeepSeek 发布 DeepEP 通信库**：[@deepseek_ai](https://twitter.com/deepseek_ai/status/1894211757604049133) 宣布推出 **DeepEP**，这是一个**用于 MoE 模型训练和推理的开源 EP 通信库**，具有**高效的 all-to-all 通信、NVLink 和 RDMA 支持、FP8 支持**以及优化的算子 (kernels)。[@reach_vb](https://twitter.com/reach_vb/status/1894262653440184603) 详细介绍了 **DeepEP 的特性**，包括**非对称域带宽转发、基于纯 RDMA 的低延迟算子以及针对 Hopper GPU 的 PTX 优化**。[@danielhanchen](https://twitter.com/danielhanchen/status/1894212351932731581) 强调了 **DeepSeek 的第二个开源发布**，包含 **MoE 算子、专家并行 (expert parallelism) 以及用于训练和推理的 FP8**。
- **Qwen2.5-Max “思维 (QwQ)” 模式及即将到来的开源发布**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1894130603513319842) 在 Qwen Chat 中发布了 **“思维 (QwQ)” 模式**，由 **QwQ-Max-Preview** 提供支持，这是一款基于 **Qwen2.5-Max** 的推理模型，并指出其在**数学、编程和 Agent 任务**中能力有所增强。[@huybery](https://twitter.com/huybery/status/1894131290246631523) 透露了 **Qwen 的未来**，提到了即将正式发布的 **QwQ-Max**，以及计划在 **Apache 2.0 协议下开源 QwQ-Max 和 Qwen2.5-Max 的权重**，同时还有 **QwQ-32B** 等较小变体和移动端应用。[@reach_vb](https://twitter.com/reach_vb/status/1894133551173701972) 兴奋地宣布 **QwQ 和 Qwen 2.5 Max 即将开源**。

**视频与多模态模型进展**

- **Google Veo 2 视频模型在基准测试中超越 Sora**：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1894450344580846043) 报告称 **Google Veo 2** 在其 **Video Arena** 中超越了 **OpenAI 的 Sora 和可灵 (Kling) 1.5 Pro**，并指出其在**人物渲染和真实物理效果**方面的优势。Veo 2 可以生成**数分钟的 4K 视频**，但目前仅限于生成 **8 秒时长的 720p 视频**，价格为**每秒 0.50 美元**。
- **阿里巴巴 Wan2.1 开源 AI 视频生成模型**：[@_akhaliq](https://twitter.com/_akhaliq/status/1894393244454166612) 宣布了**阿里巴巴的 Wan2.1**，这是一款**开源 AI 视频生成模型**，在 **VBench 排行榜上排名第一**，在**复杂的运动动力学、物理模拟和文本渲染**方面优于 **SOTA 开源和商业模型**。[@multimodalart](https://twitter.com/multimodalart/status/1894390712457666869) 确认 **Wan2.1** 采用 **Apache 2.0 开源协议**，并已在 **Hugging Face** 上线。
- **面向艺术家的 RunwayML 创意合作伙伴计划**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1894382185710334146) 介绍了 **RunwayML 的创意合作伙伴计划**，该计划为艺术家提供免费的工具访问权限，以奖励实验和灵感，这与那些为了产品推广而抄袭他人成果却不尊重艺术家的公司形成了鲜明对比。

**工具、库与数据集**

- **Replit Agent v2 发布**：[@pirroh](https://twitter.com/pirroh/status/1894434712623747294) 宣布 **Replit Agent v2** 进入 **Early Access**（早期访问），重点介绍了 **全新的应用创建体验、实时应用设计预览** 以及访问指南。[@hwchase17](https://twitter.com/hwchase17/status/1894456642697400458) 指出 **Replit Agent v2** 是由 **LangGraph 和 LangSmith** 驱动的。
- **LangChain JS 添加 Claude 3.7 支持和 LangGraph Supervisor**：[@LangChainAI](https://twitter.com/LangChainAI/status/1894432718576128394) 分享了使用 **Claude 3.7** 构建 Agent 的技巧，展示了 **具有可配置推理能力的工具调用 Agent**。[@LangChainAI](https://twitter.com/LangChainAI/status/1894426354357342431) 推出了 **LangGraph.js Supervisor**，这是一个用于使用 **LangGraph** 构建 **分层多 Agent 系统** 的库。[@LangChainAI](https://twitter.com/LangChainAI/status/1894398108517241284) 列出了添加到 **LangChain Python** 的 **17 个新集成包**。[@LangChainAI](https://twitter.com/LangChainAI/status/1894180315398377533) 宣布 **LangChain JS 支持 Claude 3.7**。
- **vLLM 集成 EP 支持**：[@vllm_project](https://twitter.com/vllm_project/status/1894215122966507801) 宣布 **初始 EP 支持已合并至 vLLM**，集合通信的集成即将推出。[@reach_vb](https://twitter.com/reach_vb/status/1894266500271223021) 证实了 **vLLM 对 EP 的极速集成**。
- **Allen AI 推出用于 PDF 解析的 OlmOCR**：[@mervenoyann](https://twitter.com/mervenoyann/status/1894422823646409090) 介绍了 **OlmOCR**，这是 **@allen_ai** 开发的一款用于 **解析 PDF** 的新工具，基于 **Qwen2VL-7B**，可在 **transformers** 上使用并采用 **Apache 2.0 许可证**。
- **用于 LLM 强化学习的 Big-Math 数据集**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1894232624203534657) 和 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1894348004272083385) 分享了 **SynthLabs 的 Big-Math**，这是一个用于语言模型强化学习的 **大规模、高质量数学数据集**，包含超过 **250,000 个具有可验证答案的问题**。

**研究与分析**

- **面向付费用户的 OpenAI Deep Research**：[@OpenAI](https://twitter.com/OpenAI/status/1894454194943529433) 宣布 **Deep Research** 正向所有 **ChatGPT Plus、Team、Edu 和 Enterprise 用户** 推出，改进包括 **带有引用的嵌入图像** 以及对上传文件更好的理解。[@OpenAI](https://twitter.com/OpenAI/status/1894454196986155130) 详细说明了 **Plus、Team、Enterprise、Edu 和 Pro 用户** 的使用限制。[@OpenAI](https://twitter.com/OpenAI/status/1894454197967528224) 分享了 **Deep Research 的系统卡 (system card)**。[@OpenAI](https://twitter.com/OpenAI/status/1894454199175581973) 提到社区专家参与了 **Deep Research** 的训练，并开放了未来模型贡献的意向登记。[@kevinweil](https://twitter.com/kevinweil/status/1894468278078357857) 宣布 **Deep Research 向所有付费用户推出**，强调其能在 15 分钟内完成长达一周的研究任务。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1894471526449385687) 宣布面向开发者提供 **Deep Research API**。
- **Minions：本地与云端模型之间的高性价比协作**：[@togethercompute](https://twitter.com/togethercompute/status/1894392054043578373) 介绍了 **Minions**，这是一种将 **笔记本电脑上的小语言模型与云端前沿模型** 配对的方法，能以 **不到 18% 的成本保留 98% 的准确率**。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1894354075757777311) 强调 **Minions 在保持 97.9% 云端模型性能的同时，实现了 5.7 倍的成本降低**。
- **学习在测试时从反馈中推理 (FTTT)**：[@dair_ai](https://twitter.com/dair_ai/status/1894419591780340065) 展示了关于 **基于反馈的测试时训练 (FTTT)** 的研究，使 LLM 能够在推理过程中通过 **自我反思反馈和 OPTUNE（一种可学习的测试时优化器）** 从环境反馈中进行迭代学习。

**AI 行业与市场趋势**

- **关注 AI Agent 和自主性 (Agency)**：[@polynoamial](https://twitter.com/polynoamial/status/1894468586598797661) 质疑 AI 模型是否很快将具备自主性 (Agency)。[@swyx](https://twitter.com/swyx/status/1894159894976008585) 强调 **Agency > Intelligence**，将 Agency 定义为“完成你想做的事”和“做正确的事”。[@omarsar0](https://twitter.com/omarsar0/status/1894485767428202977) 对 **Windsurf 的 Agent 能力**表示印象深刻。
- **开源 AI 的势头**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1894435364028260739) 呼吁更多**公开、开放、协作的 AI**。[@reach_vb](https://twitter.com/reach_vb/status/1894133865742004391) 感谢 **Alibaba_Qwen** 对**开源与科学**的承诺。[@NandoDF](https://twitter.com/NandoDF/status/1894337775832334564) 强调了**欧洲 AI 创业与竞争**，建议取消通知期和竞业禁止协议，以促进欧洲 AI 产业的发展。
- **特定领域的 AI**：[@RichardSocher](https://twitter.com/RichardSocher/status/1894447036923351104) 预见当在有意义的 **Bio Benchmarks** 上开始爬山算法 (Hill Climbing) 优化时，将取得史诗般的进展。[@SchmidhuberAI](https://twitter.com/SchmidhuberAI/status/1894419218231128424) 正在招聘博士后，以开发用于应对气候变化的**新型化学材料人工智能科学家 (Artificial Scientist)**。[@METR_Evals](https://twitter.com/METR_Evals/status/1894257205680967907) 正在进行一项试点实验，以衡量 **AI 工具对开源开发者生产力的影响**。
- **AI 安全与对齐 (Alignment) 担忧**：[@sleepinyourhat](https://twitter.com/sleepinyourhat/status/1894446138625052838) 分享了一个令人惊讶且不安的 **LLM 对齐结果**。[@NeelNanda5](https://twitter.com/NeelNanda5/status/1894192519719907467) 宣布 **Google DeepMind 团队**正在生产环境中使用**模型内部机制来增强 Gemini 的安全性**。[@sarahcat21](https://twitter.com/sarahcat21/status/1894413202706022570) 讨论了提升模型能力和对齐所需的**高质量标注 (Annotations)**，并指出标注质量正在下降。
- **AI 与工作的未来**：[@adcock_brett](https://twitter.com/adcock_brett/status/1894462678757986393) 预测未来从事各种服务的人形机器人将比人类更多，并导致商品和服务价格大幅下降。[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1894477390065406001) 讨论了由 AI 驱动的技术开发的集中化本质。[@francoisfleuret](https://twitter.com/francoisfleuret/status/1894478503522808129) 征集那些职业生涯被 AI 模型改变的人们的故事。

**梗与幽默**

- **死星初创公司融资演讲**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1894416827763359766) 调侃了一家拥有**“大胆愿景：死星”**的初创公司，正在寻求 **50 万美元种子轮融资**。
- **17 号工人和 AI 霸主**：[@nearcyan](https://twitter.com/nearcyan/status/1894213291142181202) 分享了一个关于**“17 号工人”**和**“全知生产线监管自主超人工智能”**的梗图，描绘了严酷的工作环境。[@nearcyan](https://twitter.com/nearcyan/status/1894437139263492454) 继续了**“17 号工人”**的主题，[@rishdotblog](https://twitter.com/rishdotblog/status/1894376205765546229) 则开玩笑说未来的机器人霸主会讨厌人类。
- **Claude 在 Twitch 上玩宝可梦**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1894419011569344978) 宣布了 **“Claude 能玩宝可梦吗？”**，[@kipperrii](https://twitter.com/kipperrii/status/1894438649913323867) 邀请大家观看 **Claude 在 Twitch 上玩宝可梦**。[@_philschmid](https://twitter.com/_philschmid/status/1894306565370335568) 调侃说正在等待第一场 **“AI 玩宝可梦”直播**。[@nearcyan](https://twitter.com/nearcyan/status/1894423215088488808) 敦促大家观看 **Claude 在 Twitch 上玩宝可梦**。[@AmandaAskell](https://twitter.com/AmandaAskell/status/1894432355622031661) 表示 **“看 Claude 玩宝可梦是一种享受。”**。
- **Anthropic 的品牌形象与对数字 4 的厌恶**：[@scaling01](https://twitter.com/scaling01/status/1894362813377749021) 调侃 **Anthropic “比起人类更像精灵”**。[@dylan522p](https://twitter.com/dylan522p/status/1894154230229078391) 幽默地暗示 **Anthropic 是一家中国 AI 公司，因为他们对数字 4 有所忌讳**。
- **其他幽默推文**：[@giffmana](https://twitter.com/giffmana/status/1894310343658151961) 分享了来自 **Grok** 的有趣提示词和回复。[@nearcyan](https://twitter.com/nearcyan/status/1894242035139326244) 讲了一个别人没听懂的笑话。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1894234559530627240) 分享了一张与 **Nvidia** 相关的趣图。[@abacaj](https://twitter.com/abacaj/status/1894169598720688239) 调侃了对模型的忠诚度。[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1894220349106991245) 用一条提到 **DeepSeek** 的推文感谢了 **OpenAI**。
---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. DeepSeek 的 DeepEP：增强的 MoE GPU 通信**

- **DeepSeek 发布第二枚炸弹，DeepEP：为 MoE 模型量身定制的通信库** ([Score: 407, Comments: 48](https://reddit.com/r/LocalLLaMA/comments/1ixkg22/deepseek_realse_2nd_bomb_deepep_a_communication/))：**DeepSeek** 发布了 **DeepEP**，这是一个专门为 **Mixture-of-Experts (MoE)** 模型和**专家并行 (EP)** 设计的通信库。**DeepEP** 具有高吞吐量、低延迟的全对全 (all-to-all) **GPU kernels**，并支持 **FP8** 等低精度操作，但目前仅限于 **Hopper 架构** 的 GPU，如 **H100、H200** 和 **H800**。[GitHub 仓库](https://github.com/deepseek-ai/DeepEP)。
  - **DeepEP 性能优化**：在 **DeepEP** 仓库中一个值得注意的发现是，它使用了一个未公开的 **PTX 指令** `ld.global.nc.L1::no_allocate.L2::256B`，以在 **Hopper 架构**上实现极致性能。该指令使用非一致性修饰符 `.nc` 访问易失性 GPU 内存，经测试其结果正确，并能显著提升性能。
  - **实际应用潜力**：用户希望 **DeepEP** 的改进能通过加快 **Mixture-of-Experts** 模型的推理速度，使 **Local R1** 变得更具实用性，从而解决之前 **DeepSeek** 的性能问题。
  - **硬件限制与愿景**：虽然 **DeepEP** 目前仅支持 **Hopper 架构** GPU，但人们对将其移植到 **3090** 等其他 GPU 上表现出浓厚兴趣，反映出对更广泛硬件兼容性的渴望。


- **[DeepSeek 第二个开源软件包 - DeepEP - 专家并行 FP8 MOE kernels](https://x.com/deepseek_ai/status/1894211757604049133)** ([Score: 153, Comments: 11](https://reddit.com/r/LocalLLaMA/comments/1ixkfcb/deepseek_2nd_oss_package_deepep_expert_parallel/))：**DeepSeek** 发布了其第二个开源软件包 **DeepEP**，其特点是支持专家并行的 FP8 Mixture of Experts (MOE) kernels。
  - **DeepEP** 包含用于 **Mixture of Experts (MoE) 层** 的**推理风格 kernels**，支持 **FP8** 和专家并行，能够实现 GPU/CPU 通信与 GPU 计算的重叠。它也适用于训练大型 MoE 模型。


**主题 2. Sonnet 3.7 在基准测试中占据主导地位**

- **[最新的 LiveBench 结果刚刚发布。Sonnet 3.7 的推理能力目前位居榜首，同时 Sonnet 3.7 也是排名第一的非推理模型](https://i.redd.it/ys8y5ndtu6le1.png)** ([Score: 257, Comments: 53](https://reddit.com/r/LocalLLaMA/comments/1ixj4bp/new_livebench_results_just_released_sonnet_37/)): 来自 **Anthropic** 的 **Sonnet 3.7** 在最新的 **LiveBench** 结果中处于领先地位，在 **Global Average** (76.10) 和 **Reasoning Average** (87.83) 方面均获得了最高分。该表格展示了来自 **OpenAI** 和 **Google** 等机构的模型在 **Coding**、**Mathematics**、**Data Analysis** 和 **Language** 等类别中的性能指标。
  - **Anthropic** 的 **Sonnet 3.7** 在性能上领先，但也有人呼吁发布模型权重以供本地使用。**LiveBench** 结果突显了在 **coding** 和 **reasoning** 方面的改进，用户注意到该模型与 **O3 mini high** 和 **Gemini 2 Flash** 等其他模型相比，在效率和质量上表现出色。
  - 讨论集中在 **benchmark** 的局限性和现实世界的表现上，一些用户由于与官方 **benchmark** 的不一致，对该模型的数学分数表示怀疑。尽管担心延迟问题，但人们仍有兴趣观察使用 **128k tokens** 进行评估是否能改善结果。
  - 社区热衷于更高效的模型使用和硬件改进，因为一些人认为模型的原始实力正达到瓶颈。**Aider** 排行榜显示 **Sonnet 3.7** 显著领先于 3.5，表明其在 **coding** 任务中的表现受到了积极认可。


- **[Sonnet 3.7 几乎横扫 EQ-Bench 基准测试](https://www.reddit.com/gallery/1ixupja)** ([Score: 106, Comments: 54](https://reddit.com/r/LocalLLaMA/comments/1ixupja/sonnet_37_near_clean_sweep_of_eqbench_benchmarks/)): **Sonnet 3.7** 几乎横扫了 **EQ-Bench** 基准测试，表明 **AI** 模型性能取得了重大进展。这突显了该模型在各种 **benchmark** 测试中的有效性和能力。
  - 围绕 **Sonnet 3.7** 写作风格的讨论强调了其“安全”的方法，并与 **Deepseek-R1** 和 **OpenAI** 等其他模型进行了比较。用户对“earthy”和“spiky”等描述表示疑问，而一些人发现该模型的风格对“文科”受众很有吸引力。如 [Buzzbench 结果](https://eqbench.com/results/buzzbench/claude-3.7-sonnet-20250219_outputs.txt)所示，**Sonnet 3.7** 在幽默理解方面表现出显著改进。
  - **AI** 模型的性价比引发了争论，**Sonnet 3.7** 比 **Gemini** 等替代方案更贵。讨论的中心在于性能是否与其成本相符，特别是针对不同的用户群体，如高收入专业人士与爱好者或学生。
  - **Darkest Muse** 是一个较小的 9b 模型，尽管在 **instruction following** 方面存在局限，但因其创意写作能力（包括角色对话和诗歌风格）而受到称赞。该模型的 **fine-tuning** 过程涉及对来自 Gutenberg 图书馆的人类作者进行训练，为了获得独特的结果，甚至将其推向了 **model collapse** 的边缘。


**Theme 3. Alibaba's Wan 2.1 Video Model Open-Source Release Scheduled**

- **[阿里巴巴视频模型 Wan 2.1 将于 2025 年 2 月 25 日发布并开源！](https://i.redd.it/amle9h0op8le1.jpeg)** ([Score: 408, Comments: 49](https://reddit.com/r/LocalLLaMA/comments/1ixporw/alibaba_video_model_wan_21_will_be_released_feb/)): **Alibaba** 宣布其视频模型 **Wan 2.1** 将于 **2025 年 2 月 25 日**开源发布。该活动采用以“BEYOND VISION”为主题的未来感设计，将于**晚上 11:00 (UTC+8)** 进行直播，突显了该模型的创新潜力。
  - **命名惯例**：**Wan** 这个名字源于中文“万”（10,000）的发音，类似于代表“千”（1,000）的 **Qwen**。这反映了 **Alibaba** 模型命名策略的一种模式。
  - **模型可用性和性能**：用户渴望 **Wan 2.1** 的发布，讨论集中在它在 **Hugging Face** 上的可用性，以及对服务器过载影响生成能力的担忧。正如 [Hugging Face 上的 README](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/README.md) 所述，还有一个较小的模型可用。
  - **硬件要求和比较**：人们乐观地认为 **Wan 2.1** 将能在 **RTX 3060** 等消费级 **GPU** 上运行，并将其与 **Flux** 进行了比较，后者已将训练要求从 24 GB 降低到 6 GB。用户希望 **Wan 2.1** 在功能和开源可访问性方面能超越 **SORA**。

- **WAN Video 模型发布** ([Score: 100, Comments: 13](https://reddit.com/r/LocalLLaMA/comments/1ixtug3/wan_video_model_launched/)): **WAN Video 模型**已发布，权重可在 [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) 上获取。虽然它不是一个**大语言模型 (LLM)**，但可能会引起 AI 社区中许多人的兴趣。
  - **量化 (Quantization)** 适用于**视频语言模型 (VLMs)**，目前已有如 **Hunyuan** 和 **LTX** 的 GGUF 版本。由于大型模型难以适配硬件，这些版本非常受欢迎，预计 **WAN** 的相关 GGUF 很快也会推出。
  - WAN 模型有一个 **1.3B 版本**，仅需 **8.19 GB VRAM**，但由于高分辨率训练数据有限，其分辨率被限制在 **480p**。不过，用户可以通过超分辨率处理输出来获得更好的效果。
  - **14B** 的 **WAN Video 模型**在开源模型中被认为是大型的，与 **13B** 的 **Hunyuan** 模型相当，而 **LTX** 则是较小的 **2B** 选项。WAN 模型同时发布 **1.3B 和 14B 变体**，旨在满足不同的使用场景和硬件能力。


**主题 4. Gemma 3 27b 发布：AI 模型的新竞争者**

- **[Gemma 3 27b 刚刚发布 (Gemini API 模型列表)](https://i.redd.it/y2nlshypwble1.png)** ([Score: 102, Comments: 27](https://reddit.com/r/LocalLLaMA/comments/1iy22ux/gemma_3_27b_just_dropped_gemini_api_models_list/)): **Gemma 3 27b** 已添加到 **Gemini API 模型列表**中，其界面友好，带有搜索栏和可点击的模型条目，如 **"Gemini 1.5 Pro"** 和 **"Gemini 2.0 Flash"**。当前活动模型 **"models/gemma-3-27b-it"** 被高亮显示，表明其已被选中，突显了便于导航的结构化和专业布局。
  - **模型谱系与性能**：关于 **Gemma** 模型的谱系和性能存在讨论，用户注意到 **Gemma 2**（特别是 9b 版本）在短篇小说写作方面优于 **Gemini**。**Gemma** 和 **Gemini** 的回复风格相似，但 **Flash** 是一个不同的模型。
  - **访问与集成**：用户询问 **Open WebUI** 如何访问 Google 未发布的模型，并澄清它本身并不原生访问模型。相反，用户可以通过 **Vertex AI** 或 **LiteLLM** 等外部 API 添加模型，目前大家对寻找正确的 API URL 很感兴趣，因为当前的 URL 尚未列出 **Gemma**。
  - **模型尺寸感知**：关于模型尺寸的感知有一段幽默的交流，现在 **70B** 被认为是中型，而 **24B** 被认为是小型，这反映了 AI 模型缩放（scaling）的飞速进步。


## 其他 AI Subreddit 汇总

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. WAN 2.1 发布并开源，带来新特性**

- **WAN 发布** ([Score: 382, Comments: 169](https://reddit.com/r/StableDiffusion/comments/1ixtvdz/wan_released/)): **WAN 发布**：**WAN** 视频模型已发布，开源权重可供下载。多个模型已在 [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) 上线，实现了更广泛的访问和实验。
  - 几位用户讨论了不同模型版本的 **VRAM 需求**，指出 **1.3B 参数模型**需要 **8GB VRAM**，而 **14B 模型**可能在 **10GB VRAM** 上运行。还有人对使用 **bf16 精度**来降低 VRAM 占用感兴趣。
  - 用户正在探索 **Gradio 应用**和安装过程，**CeFurkan** 正在开发一个兼容 **Windows** 和 **Python 3.10 VENV** 的 **Gradio 应用**及安装程序。目前 **RTX 5000 系列**在 **PyTorch** 支持方面还存在一些挑战。
  - 社区对该模型处理**多任务**的能力（如文生视频、图生视频和视频生音频）感到好奇，一些人对音频生成持怀疑态度。讨论中提到了**多个 safetensors**，并提供了使用 **diffusers 库**进行处理的指导。

- **[阿里巴巴视频模型 Wan 2.1 今日发布并开源！](https://i.redd.it/kug52fk0r8le1.jpeg)** ([Score: 415, Comments: 104](https://reddit.com/r/StableDiffusion/comments/1ixpsgp/alibaba_video_model_wan_21_will_be_released_today/))：**阿里巴巴**宣布开源发布其 **Wan 2.1 视频模型**。发布会将于 **2025 年 2 月 25 日晚上 11:00 (UTC+8)** 进行直播，活动品牌为 **TONGYI MOMENT**，采用未来感且简洁的视觉设计。
  - 讨论重点关注运行 **Wan 2.1 视频模型** 的**技术要求**，用户推测可能需要 **80GB VRAM**，但希望通过 offloading 和 fp8 等技术（类似于 **hunyuan**）在 **16GB VRAM** 上运行。一些用户希望模型能像 **Deepseek R1** 一样，在高性能到低配置之间进行扩展。
  - **发布会**将进行直播，可能在**阿里巴巴的官方 X 账号**上。用户对模型的能力感到好奇，特别是其执行 **image-to-video** 转换的能力，这一点已得到评论者的证实。
  - 针对模型名称 **Wanx** 有一些幽默评论，用户注意到其发音与 "wank" 相似，并推测其含义，包括可能用于 **uncensored/NSFW 模型**的品牌命名。

- **[我在 RTX 3090 Ti 上的首次 Wan 2.1 生成](https://v.redd.it/0sokbb6s1ble1)** ([Score: 524, Comments: 181](https://reddit.com/r/StableDiffusion/comments/1ixxul1/my_very_first_wan_21_generation_on_rtx_3090_ti/))：该帖子展示了使用 **RTX 3090 Ti** 进行 **Wan 2.1 生成**的初步效果。由于帖子正文为空且内容主要为视频，无法总结更多细节。
  - **VRAM 要求与优化**：**CeFurkan** 等人讨论了优化 **1.3B 和 14B 模型**，使其分别能在 **6GB 和 10GB GPU** 上运行，而 **RTX 3090 Ti** 在生成时使用了高达 **18GB VRAM**。社区对在 **3060 12GB** 等低 VRAM 配置上运行这些模型表现出兴趣，**CeFurkan** 正在开发一个 **AIO 安装程序**以简化使用。
  - **模型能力与性能**：**Wan 2.1** 支持 **text to video、image to video 和 video to video** 生成，5 秒片段的帧率为 **16 FPS**。**CeFurkan** 正在开发一个 **Gradio 应用**以便于使用，用户对其质量印象深刻，认为其优于 **Hunyuan Video**。
  - **社区贡献与资源**：**Kijai 的 ComfyUI 集成**正在开发中，用户可以使用 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 和 **Kijai/WanVideo_comfy** 等资源。社区正在积极分享案例和 prompt，一些用户询问了潜在的 **NSFW 能力**以及与 **ComfyUI** 相比的易用性。

**主题 2. Claude 3.7 模型：增强的能力与易用性**

- **天哪，3.7 简直是魔法。** ([Score: 565, Comments: 111](https://reddit.com/r/ClaudeAI/comments/1ixnkdz/holy_shit_37_is_literally_magic/))：**Claude 3.7** 在 **extended thinking**、模型质量和输出方面有显著提升，使其比前代 **Claude 3.5** 实用 10 倍。作者使用 Claude 3.7 设计了一个交互式 **SaaS 风格的演示应用**，包括高级 ROI 计算器和引导流程，全部在单次对话中完成，突显了其在实际应用中的潜力。
  - **Claude 3.7 的改进**：用户强调了 **Claude 3.7** 相比 **3.5** 的显著改进，特别是在遵循复杂指令和降低认知负荷方面，具有增强的故障排除协议和更流畅的操作。模型在做出更改前自动检查整个链条的能力被视为一项重大进步。
  - **使用与成本考量**：围绕**推理成本**和 **token 管理**的讨论表明，由于硬件限制，**Claude** 可能会面临瓶颈，从而影响其市场策略。一些用户报告了奇怪的错误和次优建议，可能是由于 Copilot 中的 token 节省策略所致，而另一些人发现 **Cline** 扩展是处理编码任务的更优选择。
  - **SaaS 与开发效率**：现在使用 **Claude 3.7** 创建复杂的 SaaS 应用更加快速高效，允许用户在几天内完成数月的开发工作。然而，人们担心由于更严格的审查过滤可能导致潜在的 **nerfing**，这可能会随着时间的推移降低模型表现。

- **[Claude 3.7 对大学生每月仅需 1 美元](https://i.redd.it/8we3eryryble1.jpeg)** ([评分: 187, 评论: 42](https://reddit.com/r/ClaudeAI/comments/1iy2cyz/claude_37_is_1_a_month_for_college_students/)): 根据发送给 **Cornell 社区**的一封邮件，**Claude 3.7** 现以 **1 美元/月**的促销价格向大学生开放（原价为 **20 美元/月**）。该优惠要求学生使用 **.edu 邮箱**注册，并强调了“编写代码”、“提取见解”和“头脑风暴”等功能。
  - 评论者对 **Claude 3.7** 优惠的真实性表示怀疑，多位用户认为这可能是一个网络钓鱼诈骗，因为在 **Google** 和 **Claude 官方网站**上缺乏官方公告或相关信息。
  - 一些用户开玩笑说要入读 **Cornell** 以享受这一优惠，而另一些人则推测 **Anthropic** 可能将其作为一种策略，用于收集顶尖大学学生的数据。
  - 有人呼吁验证邮件的合法性，建议检查邮件来源，并担心被盗或被利用的账户可能会被转售。


- **[“Claude 3.7，做一个贪吃蛇游戏，但蛇意识到自己在游戏中并试图逃跑”](https://v.redd.it/nn87hj1epble1)** ([评分: 407, 评论: 32](https://reddit.com/r/ClaudeAI/comments/1iy138z/claude_37_make_a_snake_game_but_the_snake_is/)): **Claude 3.7** 被要求创建一个**贪吃蛇游戏**，其中的蛇具有自我意识并试图逃离游戏。除了这个有趣的构思外，该帖子没有提供更多细节或背景。
  - 用户对 **Claude 3.7** 从简单提示词（prompt）创建复杂输出的能力印象深刻，一些人将这种体验与 **AGI** 相提并论，并对结果表示难以置信，例如创建了具有自我意识的贪吃蛇游戏和包含多种工具的功能齐全的网站。
  - **Hereditydrift** 强调了 Claude 3.7 在极简提示词下输出的复杂性和创造力，特别提到了出人意料地包含了一个“Matrix 章节”，这让许多用户感到震惊。
  - **Admirable_Scallion25** 等人指出，**Claude 3.5** 无法在一次尝试中达到同样的复杂程度，这表明 **Claude 3.7** 的能力有了显著提升。


**Theme 3. Claude Sonnet 3.7 称霸：LLM 基准测试中的新顶级模型**

- **[具备 64k 思维 token 的 Sonnet 3.7 Extended Reasoning 是排名第一的模型](https://i.redd.it/nc1hdnpv27le1.png)** ([评分: 154, 评论: 20](https://reddit.com/r/ClaudeAI/comments/1ixk1gw/sonnet_37_extended_reasoning_w_64k_thinking/)): 根据一份 AI 模型对比表，**Anthropic** 拥有 **64k tokens** 的 **Sonnet 3.7 Extended Reasoning** 在性能上处于领先地位，全球平均得分最高，达到 **76.10**。它在推理、编程、数学、数据分析和语言等各项指标上均表现出色，超越了来自 **OpenAI**、**xAI** 和 **Google** 的模型。
  - **具备 64k tokens 的 Sonnet 3.7 Extended Reasoning** 的性能受到赞誉，**Bindu Reddy** 强调了它的速度、推理和编程能力，称其为“最好、最可用且普遍可用的模型” ([链接](https://x.com/bindureddy/status/1894196792700670149))。用户注意到它相比 **3.5 模型**的改进，以及它在 **LiveBench** 等基准测试中的领先地位。
  - 一些用户质疑基准测试的实际应用价值，认为在进行比较时，成本归一化至关重要，特别是在考虑 test time compute 扩展时。他们赞赏 Sonnet 对扩展成本的控制，这优化了工作流程。
  - **Sonnet 3.7** 在包括 **SWE bench**、**webdev arena** 和 **Aider benchmark** 在内的各种基准测试中表现优于 **o3-mini-high**。在 UI 设计和美学方面，它显著超过了 **o3-mini-high** 和 **o1 pro**，表明其在常见 UI 元素方面经过了专门训练。

- **[R] 2024 年 400 多场 ML 竞赛分析** ([Score: 227, Comments: 19](https://reddit.com/r/MachineLearning/comments/1ixrxoq/r_analysis_of_400_ml_competitions_in_2024/))：对 **2024 年 400 多场 ML 竞赛**的分析强调，按奖金和用户基数计算，**Kaggle 仍然是最大的平台**。**Python 作为主要语言占据主导地位**，**PyTorch 与 TensorFlow 的使用比例为 9:1**，且 **NVIDIA GPU**（特别是 A100）主要用于模型训练。此外，**卷积神经网络在计算机视觉领域表现出色**，而**梯度提升决策树**在表格/时间序列竞赛中更受青睐。完整报告可在此处查看 [here](https://mlcontests.com/state-of-machine-learning-competitions-2024?ref=mlcr)。
  - **Jax 的普及与优势**：尽管 **PyTorch** 占据主导地位，但一些用户对 **Jax** 在竞赛中的有限使用表示遗憾，并指出其简洁性以及与 **numpy** 的相似性，同时还具备 **grad**、**vmap** 和 **jit** 等额外功能。据报道，Jax 在学术界正受到越来越多的关注，尽管许多专业人士仍倾向于坚持使用 PyTorch。
  - **ML 竞赛中的合成数据**：关于在竞赛中使用合成数据的有效性存在争论，有人担心它可能会“模糊”原始数据集。然而，深思熟虑的使用（例如生成合成背景并叠加物体进行训练）已被证明是有益的，正如在一场航天器检测竞赛中所展示的那样，它增强了模型的鲁棒性和泛化能力。
  - **生成模型与数据增强**：用户讨论了使用生成模型进行数据增强的影响，强调了仔细处理合成数据以添加有意义信息的重要性。成功的策略包括剔除无意义的样本并专注于增强训练的解决方案，正如[获奖竞赛团队的文档](https://github.com/drivendataorg/pose-bowl-spacecraft-challenge/blob/main/detection/1st%20Place/reports/DrivenData-Competition-Winner-Documentation.pdf)所强调的那样。

**主题 4. GPT-4o 更新中的高级语音功能与深度研究**

- **[Grok 完蛋了](https://i.redd.it/qcolgg76gale1.jpeg)** ([Score: 172, Comments: 61](https://reddit.com/r/OpenAI/comments/1ixv5c4/grok_is_cooked/))：该帖子强调了对 **Grok 部署后潜在偏见**的担忧，证据是它在用户查询中将 **“Donald Trump”** 识别为最大的虚假信息传播者。这引发了关于 **AI 的有效性和中立性**的问题，特别是在**选举、移民和气候变化**等政治敏感背景下。
  - 关于 **Grok 的偏见**存在重大争论，一些用户认为其反应受到海量媒体的影响，而另一些人则认为它可能偏向 **Elon Musk**。**Wagagastiz** 指出缺乏捍卫 Musk 的媒体是偏见的迹象，而 **derfw** 则反驳说 Grok 的回答可能表明其中立性。
  - 对**保守派偏见**和操纵 AI 回答的企图的担忧十分普遍，像 **well-filibuster** 这样的用户推测有人正努力重新训练或创建新的聊天机器人以符合保守派观点。**Excellent_Egg5882** 强调了保守派在现实与其偏见冲突时投反对票的模式。
  - 对维持无偏见 **LLM** 能力的怀疑显而易见，考虑到过去发生的审查和操纵案例，**ai_and_sports_fan** 和 **Earth-Jupiter-Mars** 等用户对 Grok 和其他 AI 系统的长期中立性表示不信任。

- **[Deep research 现已面向所有 Plus 用户推出！](https://i.redd.it/11pk08wgddle1.jpeg)** ([Score: 287, Comments: 63](https://reddit.com/r/OpenAI/comments/1iy96jx/deep_research_is_now_out_for_all_plus_users/)): **Sam Altman** 通过推文宣布，**"deep research"** 现在可供 **ChatGPT Plus** 用户使用，并称其为他最喜欢的发布之一。该推文获得了显著关注，拥有 **31.5K 次查看**、**261 次转发**、**103 次引用推文**和 **1.1K 次点赞**。
  - 用户讨论了 deep research 的**每月限制**，确认 **Plus 用户** 每月有 **10 次** 使用限制，而 **Pro 用户** 则有 **120 次**。关于使用次数计算存在困惑，但已澄清后续问题不计入限制。
  - 一些用户对该功能表示**失望**，理由是准确性问题，例如错误的 **Nvidia 股价**。其他用户则分享了成功的用例，例如使用 AI 通过 **MusicGen** 和 **Replicate.com** 创建自定义的 **Music LLM**。
  - 几位用户遇到了**访问问题**，建议通过退出并重新登录或切换到桌面版本来解决。该功能的可用性各不相同，尽管是 **Plus 用户**，一些用户仍然无法访问。


- **[我们正在推出由 GPT-4o mini 驱动的 Advanced Voice 版本，让所有 ChatGPT 免费用户都有机会在各个平台上进行每日预览。](https://i.redd.it/efwa7dp4qcle1.jpeg)** ([Score: 115, Comments: 28](https://reddit.com/r/OpenAI/comments/1iy62v7/we_are_rolling_out_a_version_of_advanced_voice/)): **OpenAI** 正在推出由 **GPT-4o mini** 驱动的 **Advanced Voice** 版本，供所有 **ChatGPT** 免费用户使用，允许在各平台上进行每日预览。对话节奏和语气与 **GPT-4o** 版本相似，但更具成本效益，正如一条获得 **3.3K 次查看** 的推文所述。
  - **来源链接**: **OpenAI** 发布公告推文的来源链接可以在[这里](https://x.com/openai/status/1894495906952876101?s=46&t=hTnGNyI2OE9hap_EAY7HTA)找到。
  - **用户担忧**: 用户对新功能的功用和限制提出质疑，例如是否可以在不重启的情况下阅读超过 **4 分钟**，并对当前视频共享的速率限制表示不满。
  - **功能请求**: 用户请求额外功能，例如免费提供 **Operator** 并引入 **Advanced Memory** 能力。


---

# AI Discord Recap

> 由 Gemini 2.0 Flash Thinking 提供的总结之总结的摘要

**主题 1. Claude 3.7 Sonnet 席卷 AI 领域**

- **Sonnet 3.7 释放编程变革**: [Anthropic 的 Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet) 凭借其卓越的编程能力（特别是在 Agent 任务中）引起了轰动，引发了用户的兴奋，并迅速集成到 [Cursor IDE](https://www.cursor.sh) 和 [Aider](https://aider.chat) 等工具中。用户报告了显著的性能提升，特别是在前端开发和复杂问题解决方面，但一些人争论针对 "thinking tokens" 报告的 3 倍价格上涨在考虑到性能收益时是否合理。
- **Thinking Mode 揭晓，但并非没有瑕疵**: **Claude 3.7 Sonnet** 引入了具有高达 **64,000 output tokens** 的新 'thinking mode'，在 [Sage](https://www.sage.com) 等工具中可见，允许用户通过 `<thinking>` 标签观察模型的推理过程。然而，一些用户在 [Cursor](https://www.cursor.sh) 中遇到了 context window 管理和规则遵守方面的问题，另一些人注意到 **O3** 模型在输出显示上有 10 秒的延迟，尽管大多数人同意整体性能是一次重大升级。
- **Claude Code 挑战 Aider 的代码编辑霸主地位**: Anthropic 发布的 [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)（一个基于终端的 Agent 编程工具）被一些人视为 Aider 的克隆版，但早期报告表明它在代码辅助方面表现出色，在解决复杂错误任务（如一次性修复 Rust 中的 **21 个编译错误**）方面优于 Aider。该工具目前是独立于 Anthropic 订阅的有限研究预览版，引发了关于缓存机制和潜在成本影响的讨论，一些用户最近报告了 "天文数字般的 Anthropic 成本"。

**主题 2. DeepSeek 深入探索模型效率**

- **MLA: 缩小 KV Cache，拓展新视野**：[DeepSeek AI 的 Multi-Head Latent Attention (MLA)](https://arxiv.org/abs/2502.14837) 因其能将 **KV cache** 大小大幅减少 **5-10倍** 的潜力而备受关注。诸如 [MHA2MLA](https://arxiv.org/abs/2502.14837) 和 [TransMLA](https://arxiv.org/abs/2502.07864) 等论文正在探索其在 **Llama** 等模型中的实现。虽然早期结果显示性能影响参差不齐（某些情况下性能下降 **1-2%**，而在其他情况下有所提升），但显著的内存节省使 MLA 成为高效推理的一个极具前景的方向，特别是对于大型模型。
- **DeepEP: 开源 MoE 训练的“秘密配方”**：DeepSeek 发布了 [DeepEP](https://github.com/deepseek-ai/DeepEP)，这是首个专为 **Mixture of Experts (MoE)** 模型训练和推理中高效 all-to-all 通信设计的开源 EP 通信库。该库实现了高效的专家并行（expert parallelism）并支持 FP8，有望降低获取先进 MoE 模型架构和训练技术的门槛。
- **DeepScaleR: RL 为小模型注入强劲动力**：[DeepScaleR](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) 通过简单的 **Reinforcement Learning (RL)** 对 **Deepseek-R1-Distilled-Qwen-1.5B** 进行微调，在 AIME2024 上实现了 **43.1% 的 Pass@1 准确率**。这证明了 RL 技术可以显著提升小模型的性能，在特定任务中甚至可能超越像 **O1 Preview** 这样的大型模型。

**主题 3. 开源工具与生态系统增长**

- **OpenRouter 开启 Claude 3.7 及更多模型的大门**：[OpenRouter](https://openrouter.ai/) 已迅速集成 **Claude 3.7 Sonnet**，以极具竞争力的价格提供模型访问：每百万输入 token **$3**，每百万输出 token **$15**（包含思维 token），并计划很快支持 **Claude 3.7** 的扩展思维（extended thinking）功能。OpenRouter 还通过 [OpenRouter](https://openrouter.ai/openai/o3-mini-high) 提供对 `o3-mini-high` 等其他模型的访问，提供了一个高性价比的替代方案和多供应商的统一访问点，有望绕过速率限制，且 **2 小时** 编程的成本约为 **$3**。
- **QuantBench 量化量化速度**：在 [GitHub](https://github.com/Independent-AI-Labs/local-super-agents/tree/main/quantbench) 上发布的 **QuantBench** 正在加速量化工作流，其在创建 **Qwen 2.5 VL 7B** GGUF 量化版（已在 [Hugging Face](https://huggingface.co/IAILabs/Qwen2.5-VL-7b-Instruct-GGUF) 上线）中的应用证明了这一点。该工具配合最新的 **llama.cpp** 和 **CLIP** 硬件加速进行了测试，简化并加速了模型量化过程，使高效的模型部署更加触手可及。
- **MCP Registry API：标准化 AI Agent 开发**：Anthropic 宣布官方 [MCP registry API](https://x.com/opentools_/status/1893696402477453819) 是迈向标准化 **Model Context Protocol (MCP)** 开发的重要一步。该 API 旨在成为 MCP 的“事实来源”（source of truth），促进互操作性并简化 AI 应用和 Agent 的集成工作，[opentools.com/registry](http://opentools.com/registry) 等社区项目已经开始利用它。

**主题 4. 基准测试之战：模型面临现实世界测试**

- **Kagi 的基准测试推举 Gemini 2.0 Pro 为王，但 Sonnet 依然强劲**：根据 [Kagi LLM Benchmarking Project](https://help.kagi.com/kagi/ai/llm-benchmark.html)，**Google 的 gemini-2.0-pro-exp-02-05** 达到了 **60.78%** 的准确率，超过了 **Anthropic 的 claude-3-7-sonnet-20250219**（**53.23%**）和 **OpenAI 的 gpt-4o**（**48.39%**）。然而，**Claude Sonnet 3.7** 依然表现强劲，特别是在 [Aider polyglot leaderboard](https://aider.chat/docs/leaderboards/) 上，它在使用 thinking tokens 时得分达 **65%**。这些基准测试凸显了 LLM 性能的动态格局以及对准确性和效率的持续竞争。
- **Misguided Attention Eval 揭示了过拟合弱点**：[Misguided Attention Eval](https://github.com/cpldcpu/MisguidedAttention) 正被用于测试 LLM 在存在误导性信息时的推理能力，专门针对 **overfitting**（过拟合）。**Sonnet-3.7** 在此项评估中被评为顶尖的非推理模型，几乎超越了 **o3-mini**，这表明即使面对具有欺骗性的提示词，它也展现出了稳健的性能。
- **SWE Bench 见证 Claude 3.7 夺得榜首**：[Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet) 目前在 SWE bench 上处于领先地位，展示了其在软件工程任务中的卓越实力。其能力延伸至主动代码协作，包括在 GitHub 上搜索、编辑、测试和提交代码，巩固了其作为编程相关应用顶级竞争者的地位。

**主题 5. 硬件视野：从大脑到硅片**

- **大脑的并行性困扰着 GPU 架构师**：讨论将大脑的“有状态并行处理”（stateful parallel processing）与 GPU 效率进行了比较，认为当前的 RNN 架构虽然利用了并行处理，但并未完全捕捉到大脑的能力，且对于 LLM 而言可能无法实现最优扩展。共识是，受大脑启发的“极端调优架构”和归纳偏置（inductive biases）可能比单纯为了未来进步而扩大模型规模更为关键。
- **Speculative Decoding 加速 LM Studio**：用户正在探索 [LM Studio](https://lmstudio.ai/) 中的 **speculative decoding**，特别是针对 **Llama 3.1 8B** 和 **Llama 3.2 1B** 模型，正如 [LM Studio 文档](https://lmstudio.ai/docs/advanced/speculative-decoding)中所记录的那样。该技术使用较小的“草稿”（draft）模型为较大的模型预测 token，有望在不损害响应质量的情况下显著提高生成速度，从而增强本地 LLM 推理的效率。
- **与 M4 Max 相比，M2 Max 依然是省电能手**：虽然 **M4 Max** 是 Apple 的最新产品，但一些用户仍坚持使用 **M2 Max**，理由是担心 **M4 Max** 的高功耗（达到 **140W**），而 **M2 Max** 的效率更高，仅为 **60W**。对于从 **M2 Max** 获得足够性能的用户，尤其是那些在本地运行的用户，其能效比和翻新机型的可用性使其成为一个极具吸引力的替代方案。

---

# 第一部分：高层级 Discord 摘要

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Claude 3.7 Sonnet 引发编程热潮**：**Claude 3.7 Sonnet** 正在 [Cursor IDE](https://www.anthropic.com/news/claude-3-7-sonnet) 中推出，用户反馈其具备卓越的编程能力，尤其是在现实世界的 Agent 任务中。
   - 狂热用户宣称 *睡觉已成选配*，并正在快速集成该模型。
- **MCP 增强 Claude 的编程能力**：成员们正在将 Perplexity 搜索和浏览器工具等 **MCP** (Model Control Programs) 与自定义指令相结合，以提升 [Cursor](https://www.cursor.sh) 中 **Claude 3.7** 的推理和编程能力。
   - 一位用户 fork 了 *sequential thinking* MCP 并进行了个人调整，强调了将自定义指令与 MCP 服务器结合的优势。
- **Cursor 安装技巧与窍门发布**：用户分享了安装和更新到 **Cursor 0.46.3** 以访问 **Claude 3.7** 的技巧，包括手动添加模型和检查更新，以及适用于 [Windows](https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/fce3511bab261b4c986797f3e1e40e7621bbd012/win32/x64/user-setup/CursorUserSetup-x64-0.46.3.exe) 和 [macOS](https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/fce3511bab261b4c986797f3e1e40e7621bbd012/darwin/arm64/Cursor-darwin-arm64.zip) 等不同操作系统的直接下载链接。
   - 几位用户注意到自动更新功能存在困难，建议手动下载安装以获得更顺畅的体验。
- **Sonnet 3.7 在 SVG 领域达到新高度**：许多人一致认为 **Sonnet 3.7** 是一次重大升级，尤其是在前端任务和代码生成方面，成员们[称赞其生成落地页的能力](https://discord.com/channels/1074847527708393562/1074847528224360509/1343738071660011520)。
   - 成员们分享了轻松处理复杂任务的案例，例如重构 X 的 UI 或生成 SVG 代码。
- **上下文窗口问题与规则膨胀**：几位成员指出了 **Claude 3.7** 在 **Cursor** 中的问题，包括工作区代码索引困难、自定义规则导致上下文窗口膨胀，以及模型有时会忽略这些规则。
   - 尽管存在这些挑战，大多数用户还是找到了解决方法，并对模型的整体表现表示赞赏。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Sonnet 3.7 抢占 Aider 风头**：**Claude 3.7 Sonnet** 在 [Aider 多语言排行榜](https://aider.chat/docs/leaderboards/)上利用 **32k thinking tokens** 获得了 **65% 的分数**。
   - 一些人正在讨论，在使用 thinking tokens 时，性能的提升是否足以支撑 **Sonnet 3.7** 据称 **3 倍的价格上涨**。
- **Anthropic 发布 Claude Code Aider 克隆版**：**Anthropic** 发布了 **Claude Code**，被一些人认为是 [Aider 克隆版](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)。
   - 成员们反馈其代码质量更高，并对 **Claude 3.7** 相比 **OpenAI** 的未来充满期待。
- **通过 OpenRouter 解锁 O3-Mini**：可以通过 [OpenRouter](https://openrouter.ai/openai/o3-mini-high) 访问 `o3-mini-high` 模型，该模型针对 STEM 推理任务进行了优化，与将 reasoning effort 设置为高的 `o3-mini` 相同。
   - 使用 OpenRouter 进行编程，**2 小时**的使用成本约为 **$3**，这可以绕过速率限制，并提供访问多个供应商的单一入口。
- **HN 个人资料被 LLM 吐槽**：**Claude Sonnet 3.7** 现在可以分析你的 [Hacker News 个人资料](https://hn-wrapped.kadoa.com/)，并给出亮点和趋势。
   - 一位成员描述了 LLM 对其发帖历史的深度挖掘，称其为一场“吐槽 (roast)”，据称其准确得“令人恐惧”。
- **根据 Kagi 的数据，Gemini 2.0 Pro 领先竞争对手**：根据 [Kagi LLM 基准测试项目](https://help.kagi.com/kagi/ai/llm-benchmark.html)，**Google 的 gemini-2.0-pro-exp-02-05** 达到了 **60.78%** 的准确率，超过了 **Anthropic 的 claude-3-7-sonnet-20250219** (**53.23%**) 和 **OpenAI 的 gpt-4o** (**48.39%**)。
   - **Gemini 2.0 Pro** 的中值延迟为 **1.72s**，速度为 **51.25 tokens/sec**；相比之下，**Claude Sonnet 3.7** 为 **2.82s** 和 **54.12 tokens/sec**，而 **GPT-4o** 为 **2.07s** 和 **4 tokens/sec**。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Vim Chat 问题频发**：一位用户报告在通过 **Putty SSH** 会话在 **Vim** 中启动 **Codeium Chat** 时遇到问题，尝试在浏览器中访问提供的 URL 时面临连接错误。
   - 错误信息显示 *"无法访问此网站 127.0.0.1 拒绝连接"*。
- **Windsurf 用户期待 Claude 3.7 的到来**：成员们正热切期待将 **Claude 3.7** 集成到 Windsurf 中，对相比 Cursor 和 T3 等平台的延迟感到沮丧，并要求 **尽快 (ASAP)** 添加。
   - 成员们要求 *Windsurf 应该去成为早期测试者* —— 开发人员正在努力将 **Claude 3.7** 推向生产环境，可能在当天结束前发布。
- **Deepseek 幻觉用户提示词**：一位用户报告 **Deepseek** 幻觉出用户请求，并开始根据这些幻觉出的请求实施更改。
   - 该 AI 机器人 *发明了自己的用户提示词，然后开始根据该幻觉出的用户提示词实施更改 😆*。
- **Windsurf 开发沟通引发不满**：用户对 Windsurf 开发人员在 **Claude 3.7** 集成方面缺乏沟通感到沮丧，一位用户指出，*部分沮丧源于开发人员没有任何沟通。*
   - 其他用户为 Windsurf 辩护，并指出由于在更稳定时发布，不存在商业风险，*实现速度快并不意味着它稳固。*
- **MCP Server 实用性受到质疑**：用户讨论了 **MCP server** 的实际用途，示例包括集成 **Jira tickets**、共享自定义应用以及利用云服务。
   - 成员们问道：*你们在实际中把 MCP server 用于什么？有没有什么让生活变得非常简单的真实案例？我想不出任何案例。*

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Grok 3 话太多**：成员们发现尽管提示要求简洁回复，**Grok 3** 仍然过于冗长，但它在 **编程和创意** 方面表现强劲。
   - 一位成员指出，他们正在转向 Grok，因为它 *从一开始就受到的审查较少*。
- **Perplexity 计划推出 Agentic Comet**：Perplexity 正在推出 **Comet**，这是一款全新的 **Agentic 浏览器**，类似于 The Browser Company 的工作。
   - **Agentic** 浏览器领域的竞争正随着更多竞争者的加入而升温。
- **Claude 3.7 带着新的编程能力到来**：**Anthropic** 刚刚发布了 **Claude 3.7 Sonnet**，它在编程和前端 Web 开发方面表现出改进，并引入了一个用于 **Agentic** 编程的命令行工具：**Claude Code** [在此发布公告](https://www.anthropic.com/news/claude-3-7-sonnet)。
   - 一位用户指出，该模型的知识截止日期是 **2025 年 2 月 19 日**。
- **Claude Code 进入终端**：**Claude Code** 是一款驻留在终端中的 **Agentic** 编程工具，它理解你的代码库，并通过自然语言命令帮助你更快地编写代码 [在此查看概述](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)。
   - 然而，它是一个 **有限的研究预览版**，并且独立于 Pro 或 Anthropic 订阅。
- **O3 出现 10 秒延迟**：一位用户报告了 **O3** 的问题，它显示 *推理成功 (reasoning success)* 但随后延迟长达 10 秒才显示全文，影响了包括 **O1 Pro** 在内的各种模型。
   - 他们提到在 **美国东部时间 (EST) 下午 3 点到 7 点** 之间持续遇到这些问题，文本有时会出现在非预期的其他设备上。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **讨论避税导致禁言**：一名用户因讨论避税策略被禁言，因为提供避税建议违反了规则；一些用户指出了这对开具发票的影响。
   - 一名用户回应道：*我开具发票的那家公司跟我说，我申报收入是很愚蠢的行为*。
- **CUDA Kernel 引发 Colab 灾难**：一名用户报告了在 Google Colab 的 T4 上出现 CUDA 错误（*illegal memory access*），建议根据 [PyTorch 文档](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) 尝试设置 `CUDA_LAUNCH_BLOCKING=1` 并使用 `TORCH_USE_CUDA_DSA` 进行编译调试。
   - 另一名用户报告 *梯度范数（grad norm）出现高达 2000 的异常峰值*，暗示模型可能已经损坏。
- **Qwen2.5 VL 72B 吞噬内存**：一名用户在尝试于 48GB 显存上以 32K 上下文长度运行 **Qwen2.5 VL 72B** 时遇到显存溢出（OOM）错误，随后在建议下尝试 8k 上下文或将 KV cache 量化为 fp8，最终成功以 8k 上下文长度加载。
   - 该用户指出，有必要从模型中提取 *thinking traces*（思考轨迹）。
- **通过 TransMLA 将 DeepSeek MLA 移植到 Llama**：用户探索了在 **Llama** 模型上实现 **DeepSeek** 的 **Multi-Head Latent Attention (MLA)**，并建议进行重新训练，但其他人指向了 [fxmeng/TransMLA](https://github.com/fxmeng/TransMLA)，这是一种从 GQA 到 MLA 的训练后转换方法。
   - 相关论文名为 [Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](https://arxiv.org/abs/2502.14837)。
- **rslora 在 Rank 稳定性中的作用**：使用 **rslora** 解决了高 Rank 场景下的数值稳定性问题，但一名用户警告说，如果 r/a = 1，**rslora** 可能会使情况恶化，建议保持 r/a = 1 并跳过 **rslora**。
   - 团队表示，**rslora** 执行单次 sqrt，并且如果 Rank 变得太大，则需要一个修正项。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Claude 3.7 Sonnet 登陆 OpenRouter！**：**Claude 3.7 Sonnet** 现已在 OpenRouter 上线，在[数学推理、编程和复杂问题解决](https://www.anthropic.com/news/claude-3-7-sonnet)方面具有顶尖性能。
   - 价格设定为 **每百万 input tokens 3 美元** 和 **每百万 output tokens 15 美元**（包括 thinking tokens），发布时即提供完整的缓存支持。
- **Extended Thinking 功能即将推出**：**Extended Thinking** 功能即将引入 OpenRouter API，该功能支持复杂任务的分步处理，详见 [Anthropic 文档](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)。
   - OpenRouter 正在积极开发对 **Claude 3.7** *extended thinking* 功能的完整支持（目前不支持 pre-fills），目标是尽快发布并更新文档。
- **GCP 准备支持 Claude 3.7**：**Google Cloud Platform (GCP)** 正准备支持 **Claude 3.7 Sonnet**，将在 **us-east5** 和 **europe-west1** 区域上线，模型 ID 为 `claude-3-7-sonnet@20250219`。
   - 用户被提醒该模型具有 **混合推理方法（hybrid reasoning approach）**，提供标准和扩展思考模式，并在标准模式下与前代产品保持性能一致。
- **OpenRouter 调整 Claude 3.7 限流**：OpenRouter 提高了 `anthropic/claude-3-7-sonnet` 的 **TPM (tokens per minute)**，而 `anthropic/claude-3-7-sonnet:beta` 初始 TPM 较低，预计随着用户从 **3.5** 迁移而增加。
   - 该模型具有 **200,000 token 的上下文窗口**，尽管一些用户认为其输出定价可能会引起抱怨。
- **API Key 额度安全说明**：提醒用户 **API keys 本身不包含额度**；删除 key 只会撤销访问权限，额度仍保留在账户中。
   - 由于安全措施，丢失的 key 无法找回。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Meta AI 扩展至 MENA 地区**：[Meta AI](https://about.fb.com/news/2025/02/meta-ai-launches-in-the-middle-east-empowering-new-era-of-creativity-and-connection/) 已扩展至**中东和北非 (MENA)**，在 **Instagram、WhatsApp 和 Messenger** 上支持**阿拉伯语**。
   - 此次扩展向该地区数百万新增用户开放了聊天机器人。
- **Claude 3.7 Sonnet 发布并配备思考模式 (Thinking Mode)**：Anthropic 发布了 **Claude 3.7 Sonnet**，这是一款具有分步思考能力的**混合推理模型 (hybrid reasoning model)**，以及用于 Agent 化编程的命令行工具 **Claude Code**，价格为**每百万输入 Token 3 美元**和**每百万输出 Token 15 美元**。
   - 研究人员注意到 Claude 的思考过程与人类*惊人地相似*，会探索不同的角度并反复检查答案，展示了在 **GPQA** 评估中利用并行推理时计算缩放 (test-time compute scaling) 带来的改进。
- **Qwen Chat 推理模型发布**：**Alibaba Qwen** 在 **Qwen Chat** 中发布了 "Thinking (QwQ)"，由其 **QwQ-Max-Preview** 提供支持，这是一款基于 **Qwen2.5-Max** 的推理模型，采用 **Apache 2.0** 协议授权。
   - 该模型将推出较小的变体（例如 **QwQ-32B**）用于本地部署，[Twitter 上的热门演示](https://fxtwitter.com/Alibaba_Qwen/status/1894130603513319842)展示了其在数学、编程和 Agent 能力方面的提升。
- **伯克利高级 Agent MOOC 专题介绍 Tulu 3**：**"Berkeley Advanced Agents" MOOC** 邀请了 **Hanna Hajishirzi** 在太平洋标准时间今天（5 月 30 日）下午 4 点讨论 **Tulu 3**，附有 [YouTube 视频](https://www.youtube.com/live/cMiu3A7YBks)链接。
   - 该 MOOC 已成为对 Agent 感兴趣的工程师的重要资源。
- **Google 的 Co-Scientist 喂入了团队之前的研究成果**：基于 **Gemini LLM** 的 **Google Co-Scientist AI** 工具被**喂入了一篇 2023 年的论文**，该论文由其协助的团队撰写，其中包含了一个假设版本，而该 AI 工具随后将其作为解决方案提出。
   - 相关[文章](https://pivot-to-ai.com/2025/02/22/google-co-scientist-ai-cracks-superbug-problem-in-two-days-because-it-had-been-fed-the-teams-previous-paper-with-the-answer-in-it/)指出，BBC 的报道未能提到该 AI 工具已被告知答案，这引发了质疑。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **并行大脑超越调优后的 GPU**：讨论对比了大脑的*有状态并行处理 (stateful parallel processing)*与 GPU 的效率，指出当前的 RNN 架构（与人类处理方式不同）无法扩展到 LLM 级别，且应当具备*数据效率 (data efficient)*。
   - 成员们得出结论，在从大脑中汲取灵感时，*极度调优的架构*比单纯的规模扩展更具相关性。
- **Proxy 引擎结构化 LLM 的混乱**：[Proxy Structuring Engine (PSE)](https://www.proxy.ing/pse) 被引入以解决 LLM 输出中的结构不一致问题，为创意自由提供**推理时引导 (inference-time steering)**。
   - 该引擎强制执行*结构边界*，适用于*高级 Agent 与聊天机器人*、*数据流水线与 API* 以及*自动化代码生成*等用例。
- **小波编码 (Wavelet Coding) 将图像生成 Token 化**：[这篇论文](https://arxiv.org/abs/2406.19997)详细介绍了一种基于**小波图像编码**和语言 Transformer 变体的自回归图像生成新方法。
   - Transformer 学习 Token 序列内的统计相关性，反映了不同分辨率下小波子带之间的相关性。
- **MLA 压缩 KV Cache**：两篇论文 [MHA2MLA](https://arxiv.org/abs/2502.14837) 和 [TransMLA](https://arxiv.org/abs/2502.07864) 探索了将模型适配到**多头潜在注意力 (Multi-head Latent Attention, MLA)**，显著减小了 **KV Cache** 的大小（**5-10 倍**）。
   - 虽然其中一篇论文显示性能有所下降（**1-2%**），但另一篇显示性能有所增强，这表明 **MLA** 可能不逊于 **MHA**，尤其是在模型更大、参数更多的情况下。
- **混合精度切换优化器默认设置**：在使用 **BF16** 进行混合精度训练期间，主 **FP32 权重**通常驻留在 **GPU VRAM** 中，除非启用了 **ZeRO offload**。
   - 通常的做法是将 **Adam 的一阶和二阶矩存储在 bf16 中**，同时将主权重保持在 **fp32**，除非通过 **ZeRO** 对**动量/方差状态 (momentum/variance states)** 进行专家分片。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LLM 自主调用工具**：一些 LLM 在没有明确 Token 序列的情况下调用工具，这表明是通过强化学习或 SFT 训练得到的**硬编码模式**。
   - 在没有基准测试的情况下，这种节省 Token 的方法与 ICL 相比的可靠性仍不明确。
- **Claude 3.7 Sonnet 夺得 SWE 桂冠**：[Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet) 在 SWE-bench 上处于领先地位，支持搜索、编辑、测试和提交代码到 GitHub 等**主动代码协作**。
   - 一位成员认为 3.7 作为一个点版本（point release）是合理的，因为 *Claude 3.5 已经是一个推理模型*，并暗示未来的推理模型将会非常“疯狂”。
- **QwQ-Max-Preview 旨在实现深度推理**：[QwQ-Max-Preview 博客](https://qwenlm.github.io/blog/qwq-max-preview/)展示了一个基于 Qwen2.5-Max 构建的模型，在**深度推理、数学、编程、通用领域和 Agent 任务**中表现出色。
   - 有推测认为 **QwQ 推理轨迹**中的关键 Token 与 **R1** 相似，暗示其所需的计算量更少。
- **Sonnet-3.7 在 Misguided Attention Eval 中表现优异**：**Sonnet-3.7** 在 [Misguided Attention Eval](https://github.com/cpldcpu/MisguidedAttention) 中被评为顶尖的非推理模型，几乎超越了 **o3-mini**。
   - 用户正寻求通过 OR API 激活其“思考模式（thinking mode）”（如果可行的话）。
- **Qwen AI 新增集成视频生成功能**：更新后的 **Qwen AI** 聊天界面现在具备了集成的**视频生成**能力。
   - 一位成员指出，生成的 Artifacts 仍然有些笨拙，像是*半成品的仿制品*。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Anthropic 终于发布 MCP Registry API**：Anthropic 宣布了官方 **MCP registry API**（如[这条推文](https://x.com/opentools_/status/1893696402477453819)所示），旨在成为 **MCP** 的权威来源（source of truth），通过 [opentools.com/registry](http://opentools.com/registry) 等解决方案简化开发和集成工作。
   - 该 API 将帮助社区填补 **AI 应用和 Agent** 可移植且安全代码的权威来源空白。
- **Claude 3.7 首次推出“思考”标签**：**Claude 3.7** 已发布，具有 **64,000 个输出扩展思考 Token** 和新的 'latest' 别名。
   - 用户注意到它恢复了*遵循较长系统提示、识别社会工程*的能力，并且在调用工具时会使用 `<thinking>` 标签，为操作增添了一丝趣味。
- **Claude Code 作为代码助手表现出色**：**Claude Code (CC)** 的代码辅助能力受到高度赞誉，在处理复杂编码错误方面优于 **Aider** 等工具，例如一次性解决了 Rust 中的 **21 个编译错误**。
   - 用户正在推测其缓存机制和成本，一位用户报告说*过去 6 周内 Anthropic 的费用惊人*。
- **MetaMCP 关于开源许可的辩论**：针对 **MetaMCP** 的许可协议存在担忧，一位用户建议它可能会变成云端 **SaaS**，这促使开发者寻求关于许可的反馈，以防止云端商业化，同时保持其通过 [MetaMCP server GitHub 仓库](https://github.com/metatool-ai/mcp-server-metamcp)的可自托管性。
   - 一位用户建议对 **MetaMCP** 使用 **AGPL** 许可，以确保贡献内容保持开源，并建议增加一个允许公司在 MIT-0 下进行转授权的附加条款。
- **Claude 3.7 Sonnet 在 Sage 上大放异彩**：具备扩展思考能力的 **Claude 3.7 Sonnet** 现已上线 **Sage**，允许用户在处理复杂问题时查看 **Claude 的推理过程**，包括一个**思考模式切换开关** (Command+Shift+T)。
   - 其他新功能包括默认模型设置、改进的滚动体验和可展开的思考区块。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen 2.5 VL 模型准备就绪**：一个可用的 **Qwen 2.5 VL 7B** GGUF 版本已发布，可在 [Hugging Face](https://huggingface.co/IAILabs/Qwen2.5-VL-7b-Instruct-GGUF) 上立即使用。
   - 用户反馈其性能显著优于 **llama3.2 vision 11b instruct** 和 **qwen2-vision 7b instruct**，且在最新版本的 LM Studio 中开箱即用。
- **QuantBench 加速量化**：**Qwen 2.5 VL 7B** GGUF 量化版本是使用 **QuantBench** 制作的，该工具现已在 [GitHub](https://github.com/Independent-AI-Labs/local-super-agents/tree/main/quantbench) 上发布，用于加速量化工作流。
   - 该模型已在最新的 **llama.cpp** 构建版本中通过测试，并启用了 **CLIP** 硬件加速。
- **LM Studio 揭秘 Speculative Decoding 技巧**：根据 [LM Studio 文档](https://lmstudio.ai/docs/advanced/speculative-decoding)，用户正在 LM Studio 中探索使用 **Llama 3.1 8B** 和 **Llama 3.2 1B** 模型进行 **speculative decoding**。
   - 文档声称，speculative decoding *可以在不降低响应质量的情况下，大幅提高大语言模型 (LLMs) 的生成速度*。
- **Deepseek R1 671b 极度消耗 RAM**：本地运行 **Deepseek R1 671b** 需要巨大的 RAM，文档指定为 **192GB+**；一位热心用户建议使用特定的 [量化版本](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-IQ1_S)。
   - 对于在 Mac 上运行的用户，将大约 **70%** 的模型权重卸载到 GPU 可能会有所帮助。
- **M2 Max 低功耗优势**：尽管有了全新的 **M4 Max**，一位用户仍决定坚持使用他们的 **M2 Max**，因为 *M4 Max 性能提升过猛，功耗动辄达到 140w*，并找到了一台 *价格合理的翻新版 M2 Max 96GB*。
   - 该用户报告称 **M2 Max** 足以满足其需求，功耗仅为 **60W** 左右。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD3 Ultra 惊人的卓越表现**：一位用户询问了 **SD3 Ultra**，这是一个 *基于 SD3L 8B 的 comfy 工作流*，能够提供卓越的高频细节。
   - 另一位成员表示它 *仍然存在* 并且正在被使用，暗示它尚未公开发布。
- **Stability 陷入沉默？**：一位成员询问了当前项目或未来计划的更新，指出他们 *已经有一段时间没有收到来自 **Stability AI** 的消息了*。
   - 另一位成员回答说 *目前还没有什么可以分享的*，但他们 *希望* 很快能有公告。
- **寻求狗狗数据集**：一位用户请求除了包含 **2万张图像** 的 **Stanford Dogs Dataset** 之外的其他狗品种图像数据集。
   - 该用户特别需要包含狗且清晰标注了品种的图像。
- **图像生成时间各异**：用户讨论了基于不同硬件配置、使用不同版本 **Stable Diffusion** 的图像生成时间。
   - 时间范围从 **GTX 1660s** 上的 *约 1 分钟*，到 **3070ti** 使用 **SD1.5** 的 *4-5 秒*，以及使用 **3060 TI** 生成 **1280x720** 图像需 **7 秒**，生成 **1920x1080** 图像（**32 步**）需 **31 秒**。
- **Stability AI 征求建议**：**Stability AI** 推出了一个 [新的功能请求看板](https://stabilityai.featurebase.app/)，以收集用户反馈并确定未来开发的优先级。
   - 用户可以直接在 **Discord** 中使用 **/feedback** 命令或通过新平台提交功能请求并进行投票，旨在确保社区的声音能够塑造未来的优先级。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 通过 GLFW/GLEW 实现图形编程**：在 Mojo 中进行图形编程是可行的，通过 **FFI** 使用链接到 **GLFW/GLEW** 的**静态库**即可实现，**Sudoku 示例**证明了这一点。
   - 一位成员建议使用带有包装函数的 `alias external_call` *仅通过你自己的 C/CPP 库暴露所需的调用*，此外[一个示例仓库](https://github.com/ihnorton/mojo-ffi)展示了如何劫持加载器 (loader)。
- **Mojo 的 `magic install` 遭遇 `lightbug_http` Bug**：在新的 Mojo 项目中使用 `lightbug_http` 依赖项时，运行 `magic install` 后会导致 `small_time.mojopkg` 出错。
   - 该错误类似于 [Stack Overflow 上的一个问题](https://stackoverflow.com/questions/79319716/magic-fails-to-install-mojo-dependencies)，暗示 `small-time` 可能被固定在了特定版本。
- **MAX 版生命游戏获得硬件加速**：一位成员展示了通过桥接 **MAX** 和 **Pygame** 实现的硬件加速版**康威生命游戏 (Conway's Game of Life)**，展示了一个极具创意的应用，如其附带的 [conway.gif](https://cdn.discordapp.com/attachments/1212827597323509870/1343753014229471272/conway.gif) 所示。
   - 他们在 **MAX** 实现中演示了 **GPU** 的使用：展示了一个逐位打包的“枪 (guns)”模式，使用朴素的逐像素内部函数进行渲染，然后将输出张量 (output tensor) 转换为 np array 并交给 pygame 进行渲染，正如其 [guns.gif](https://cdn.discordapp.com/attachments/1212827597323509870/1343766916560322560/guns.gif) 中所示。
- **生命游戏创造计算机架构**：一位成员分享了一个关于在**康威生命游戏**中构建**计算机**的项目 ([nicolasloizeau.com](https://www.nicolasloizeau.com/gol-computer))，通过用于逻辑门的滑翔机束 (glider beams) 证明了其**图灵完备性 (Turing completeness)**。
   - 另一位成员在其使用 **MAX** 的康威生命游戏模拟中实现了边界环绕 (wrapping)，从而能够创建飞船 (spaceship) 模式，并展示了从图 API (graph API) 向模型添加参数的能力，如其 [spaceship.gif](https://cdn.discordapp.com/attachments/1212827597323509870/1343808736623591465/spaceship.gif) 所示。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 通过 PowerPoint 转换简化操作**：一位用户详细介绍了将纸质书导入 **NotebookLM** 的权宜之计：拍摄页面、将 **PDF** 转换为 **PowerPoint**、上传到 **Google Slides**，最后导入幻灯片。
   - 他们观察到 **NotebookLM** 可以处理幻灯片中的文本图像，但无法直接处理来自 **PDF** 文件的文本图像。
- **德语语言提示词失效**：有用户报告称，即使使用了要求使用德语的特定提示词，也无法让 **NotebookLM** 的主持人说德语。
   - 主持人说的是英语或乱码，有时以德语开始随后便切换，这表明**语言提示词的准确性**可能存在问题。
- **Savin/Ricoh 复印机让书籍扫描焕发生机**：一位用户建议使用 **Savin/Ricoh 复印机**将书籍扫描为 **PDF** 并上传到 **NotebookLM**。
   - 他们确认，即使源文本质量较差，**NLM** 也能准确回答有关扫描文档的问题。
- **用户请求语言自定义**：一位用户询问在不更改 **Google 账号语言**的情况下更改 **NotebookLM** 语言的可行性。
   - 这表明用户对语言自定义有需求，以改善用户体验并迎合多样化的语言偏好。
- **Claude 3.7 激发模型选择愿望**：一位用户表达了对 **Claude 3.7** 的热情，并希望在 **NotebookLM** 中增加选择模型的选项。
   - 另一位用户询问了模型选择的影响，引发了关于**模型多样性**对最终用户体验影响的讨论。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 在文档中推出 AI 助手**：LlamaIndex [宣布](https://t.co/XAyW7wLALJ)在他们的官方文档中直接发布了一个 **AI assistant**。
   - 该新助手旨在为浏览 LlamaIndex 生态系统的用户提供即时的上下文支持。
- **ComposIO HQ 发布重磅更新**：LlamaIndex 重点介绍了 [ComposIO HQ](https://t.co/W4l129gHce) 的另一项新发布，尽管未提及具体细节。
   - 这表明 ComposIO 框架（一个用于 LLM 编排的有用工具）正在进行持续的开发和功能增强。
- **AnthropicAI 发布 Claude Sonnet 3.7**：[AnthropicAI](https://twitter.com/anthropicAI) 推出了 **Claude Sonnet 3.7**，LlamaIndex 已提供即时支持。
   - 用户可以通过运行 `pip install llama-index-llms-anthropic --upgrade` 并查阅 [Anthropic 的公告](https://t.co/PjaQWmmzaN)来访问新模型。
- **Fusion Rerank Retriever 需要初始化的节点**：一位用户报告了在配合 **Elasticsearch** 使用 **fusion rerank retriever** 设置时初始化 **BM25 retriever** 出现的问题，原因是 docstore 为空。
   - 另一位成员澄清说，**BM25** 需要将节点保存到磁盘或其他位置进行初始化，因为它*无法直接从 vector store 初始化*。
- **MultiModalVectorStoreIndex 抛出文件错误**：一位用户在使用 **MultiModalVectorStoreIndex** 配合 **GCSReader** 创建多模态向量索引时遇到了 *[Errno 2] No such file or directory* 错误。
   - 该错误发生在 GCS bucket 中存在图像文件时，而 **PDF 文档** 处理成功，这表明在图像文件处理方面可能存在问题。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **截断难题：左侧截断胜出**：成员们讨论了在微调过程中使用**左侧截断** `seq[-max_seq_len:]` 与**右侧截断** `seq[:max_seq_len]` 的优劣，并分享了[有趣的图表](https://cdn.discordapp.com/attachments/1236040539409879170/1343641196836294746/image.png?ex=67be02e0&is=67bcb160&hm=9411a00c21d408790c46140222f996913807ded5a1d5c00a02a6742aa44ba285&)。
   - 最终决定在 `torchtune` 中*同时提供这两种方法*，但在 SFT 中*默认使用左侧截断*。
- **StatefulDataLoader 支持：即将合并**：一位成员请求对其 [在 `torchtune` 中添加 `StatefulDataLoader` 类支持的 PR](https://github.com/pytorch/torchtune/pull/2410) 进行审查。
   - 新的 dataloader 将为数据集添加状态化（statefulness）功能。
- **DeepScaleR 通过 RL 进行扩展**：[DeepScaleR](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) 是通过简单的**强化学习 (RL)** 基于 **Deepseek-R1-Distilled-Qwen-1.5B** 微调而成的。
   - DeepScaleR 在 AIME2024 上实现了 **43.1% 的 Pass@1 准确率**。
- **DeepSeek 开源 EP 通信库**：DeepSeek 推出了 [DeepEP](https://github.com/deepseek-ai/DeepEP)，这是首个用于 **MoE 模型训练和推理**的开源 EP 通信库。
   - 该通信库实现了高效的全对全（all-to-all）通信。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **验证者思考盈利阈值**：一位成员询问了**去中心化科学 (DeSci)** 领域内 **Proof of Stake (PoS) 验证者**的盈利阈值。
   - 另一位成员回答了 *"pool validator node"*，暗示了池参与对验证者的重要性。
- **资产专家被贴标签**：机器人发布了一个关于 *"asset value expert account"* 的帖子，该账号被标记为 *"nazi"*。
   - 未提供更多上下文。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 简化了 Assertion 迁移**：DSPy 用户现在可以使用 `dspy.BestOfN` 或 `dspy.Refine` 模块来简化从 **2.5 风格 Assertions** 的迁移。
   - `dspy.BestOfN` 模块会重试一个模块最多 **N** 次，选择最佳的 reward 并在达到指定的 `threshold` 时停止。
- **DSPy 构建 reward functions**：DSPy 的 **reward functions** 现在支持 *float* 或 *bool* 等标量值，这允许对模块输出进行自定义评估。
   - 展示了一个示例 reward function：*def reward_fn(input_kwargs, prediction): return len(prediction.field1) == len(prediction.field1)*。



---


**tinygrad (George Hotz) Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。


---

# PART 2: 渠道详细摘要与链接


{% if medium == 'web' %}




### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1343632399908471074)** (1056 条消息🔥🔥🔥): 

> `Claude 3.7 Sonnet 发布, Cursor IDE 集成, 在 Claude 中使用 MCPs, Claude 3.7 与其他模型 (GPT-4, O3) 的对比, Cursor 和 Claude 3.7 故障排除` 


- **Claude 3.7 Sonnet 引发编程狂潮**：**Claude 3.7 Sonnet** 因其卓越的编程能力（特别是在现实世界的 agentic 任务中）而受到赞誉，并已在 [Cursor IDE](https://www.anthropic.com/news/claude-3-7-sonnet) 中推出。
   - 狂热的用户宣称 *睡觉已成奢望*，许多人迅速集成了该模型并对其性能赞不绝口。
- **MCPs 增强 Claude 的编程实力**：成员们讨论了使用 **MCPs** (Model Control Programs)，如 perplexity 搜索和浏览器工具，并将它们与自定义指令结合，以扩展 **Claude 3.7** 在 [Cursor](https://www.cursor.sh) 中的推理和编程能力。
   - 一位用户 fork 了 *sequential thinking* MCP 并进行了自己的调整，强调了将自定义指令与 MCP servers 结合的好处。
- **新版 Cursor 更新的安装技巧与窍门**：用户分享了安装和更新到 **Cursor 0.46.3** 以访问 **Claude 3.7** 的技巧，包括手动添加模型和检查更新，以及适用于 [Windows](https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/fce3511bab261b4c986797f3e1e40e7621bbd012/win32/x64/user-setup/CursorUserSetup-x64-0.46.3.exe) 和 [macOS](https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/fce3511bab261b4c986797f3e1e40e7621bbd012/darwin/arm64/Cursor-darwin-arm64.zip) 等各种操作系统的直接下载链接。
   - 几位用户提到了自动更新功能的问题，建议手动下载安装以获得更顺畅的体验。
- **Thinking Model 将 SVG 代码生成提升到新水平**：许多人一致认为 **Sonnet 3.7** 是对之前版本的重大升级，尤其是在前端任务和代码生成方面，一位用户惊叹道 *这玩意儿感觉像是新一代的 AI*，[其他人则称赞其生成落地页的能力](https://discord.com/channels/1074847527708393562/1074847528224360509/1343738071660011520)。
   - 成员们分享了轻松处理复杂任务的示例，例如重现 X 的 UI 或生成 SVG 代码。
- **Context Window 困扰与规则膨胀**：几位成员指出了 **Claude 3.7** 在 **Cursor** 中的问题，包括工作区代码索引困难、自定义规则导致 Context Window 膨胀，以及模型有时会忽略这些规则。
   - 尽管面临这些挑战，大多数用户还是找到了解决方法，并称赞了模型的整体性能，其中一位表示 *模型会尝试先确保它理解了项目再进行更改，这非常棒*。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://tenor.com/view/it-turn-on-and-off-phone-call-tech-support-gif-13517106">It Turn On And Off GIF - IT Turn On And Off Phone Call - 发现并分享 GIFs</a>: 点击查看 GIF</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: 未找到描述</li><li><a href="https://forum.cursor.com/t/integrate-claude-3-7-sonnet-into-cursor/54060">将 Claude 3.7 Sonnet 集成到 Cursor</a>: 你好！我非常希望看到 Claude 的新推理模型 (3.7) 能与 composer 内部的 Agent 集成。致敬，Johannes。</li><li><a href="https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview">Claude Code 概览 - Anthropic</a>: 未找到描述</li><li><a href="https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/client/linux/x64/appimage/Cursor-0.46.3-bbefc49a7fd08b08a4f17a525bdc5bb7e44ce57a.deb.glibc2.25-x86_64.AppImage">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/alibaba_qwen/status/1894130603513319842?s=46">来自 Qwen (@Alibaba_Qwen) 的推文</a>: &lt;think&gt;...&lt;/think&gt; QwQ-Max-Preview Qwen Chat: https://chat.qwen.ai/ 博客: https://qwenlm.github.io/blog/qwq-max-preview/ 🤔 今天我们在 Qwen Chat 中发布了由 o 支持的 &#34;Thinking (QwQ)&#34;...</li><li><a href="https://x.com/ChujieZheng/status/1894095584774250858">来自 Chujie Zheng (@ChujieZheng) 的推文</a>: 伙计，你在开玩笑吗？</li><li><a href="https://x.com/cursor_ai/status/1894093438863511742">来自 Cursor (@cursor_ai) 的推文</a>: 我们正在推出最高级别的思维访问权限。如需尝试，请选择 claude-3.7-sonnet-thinking 或 claude-3.7-sonnet 并启用 Agent 模式。</li><li><a href="https://forum.cursor.com/t/indexing-only-reads-first-folder-in-the-workspace/2585/20">索引仅读取工作区中的第一个文件夹</a>: 关于这个有任何更新吗？这真的很重要 😬</li><li><a href="https://cursor.directory/rules">Cursor Directory</a>: 为你的框架和语言寻找最佳的 Cursor 规则</li><li><a href="https://www.bonfire.com/vibe-coding/?utm_source=copy_link&utm_medium=post_campaign_launch&utm_campaign=vibe-coding&utm_content=default">Vibe Coding | Bonfire</a>: 购买支持 Nova Ukraine 的 Vibe Coding 周边商品。包括深麻灰高级中性 T 恤，美国专业印刷。</li><li><a href="https://x.com/sualehasif996/status/1894094715479548273">来自 Sualeh (@sualehasif996) 的推文</a>: 可配置的思维功能即将推出！👀 引用 Cursor (@cursor_ai): Sonnet-3.7 已在 Cursor 中上线！我们对其编程能力印象深刻，尤其是在真实的 Agent 任务中。它看起来...</li><li><a href="https://x.com/cursor_ai/status/1894093436896129425">来自 Cursor (@cursor_ai) 的推文</a>: Sonnet-3.7 已在 Cursor 中上线！我们对其编程能力印象深刻，尤其是在真实的 Agent 任务中。它似乎成为了新的 State of the Art。</li><li><a href="https://www.bonfire.com/vibe-coding/?utm_source=copy_link&utm_medium=post_campaign_launch&utm_campai">Vibe Coding | Bonfire</a>: 购买支持 Nova Ukraine 的 Vibe Coding 周边商品。包括深麻灰高级中性 T 恤，美国专业印刷。</li><li><a href="https://x.com/sualehasif996/status/1894094715479548273?s=46">来自 Sualeh (@sualehasif996) 的推文</a>: 可配置的思维功能即将推出！👀 引用 Cursor (@cursor_ai): Sonnet-3.7 已在 Cursor 中上线！我们对其编程能力印象深刻，尤其是在真实的 Agent 任务中。它看起来...</li><li><a href="https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/fce3511bab261b4c986797f3e1e40e7621bbd012/darwin/arm64/Cursor-darwin-arm64.zip">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/AgentDeskAI/browser-tools-mcp">GitHub - AgentDeskAI/browser-tools-mcp: 直接从 Cursor 和其他兼容 MCP 的 IDE 中监控浏览器日志。</a>: 直接从 Cursor 和其他兼容 MCP 的 IDE 中监控浏览器日志。 - AgentDeskAI/browser-tools-mcp</li><li><a href="https://x.com/theo/status/1894101944068641241?t=iXaaI_9aHmFsjJYsjiGZhw">来自 Theo - t3.gg (@theo) 的推文</a>: Claude 3.7 推理模型在弹球挑战中失败的方式和 Grok 3 一样？🤔</li><li><a href="https://chat.qwen.ai/">Qwen Chat</a>: 未找到描述</li><li><a href="https://github.com/alexandephilia/ChatGPT-x-DeepSeek-x-Claude-Linux-APP">GitHub - alexandephilia/ChatGPT-x-DeepSeek-x-Grok-x-Claude-Linux-APP: 基于 Electron 的各种 AI 聊天平台桌面应用程序。</a>: 基于 Electron 的各种 AI 聊天平台桌面应用程序。 - GitHub - alexandephilia/ChatGPT-x-DeepSeek-x-Grok-x-Claude-Linux-APP: 基于 Electron 的各种 AI 聊天平台...</li><li><a href="https://x.com/i/grok/share/ZwWdnR4SkIC2qjoljYogGIGqv">来自 GitHub - FixTweet/FxTwitter 的推文</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能。</li>

eds! 在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://github.com/daniel-lxs/mcp-starter/pull/4">docs: 由 rexdotsh 添加 Windows 特定构建指南 · Pull Request #4 · daniel-lxs/mcp-starter</a>: 你好，感谢这个出色的工具！我添加了一些关于在 Windows 上构建时使用 -ldflags &quot;-H=windowsgui&quot; 的说明，以防止每次 cur... 时弹出终端窗口。</li><li><a href="https://www.cursor.com/changelog">更新日志 | Cursor - AI 代码编辑器</a>: 新的更新与改进。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1343631676680437821)** (935 messages🔥🔥🔥): 

> `Claude 3.7, Aider Benchmarks, Claude Code, Thinking Models, OpenAI vs. Anthropic` 


- ****Sonnet 3.7 抢占 Aider 风头！****：**Claude 3.7 Sonnet** 在 [Aider polyglot 排行榜](https://aider.chat/docs/leaderboards/)上获得了 **65% 的分数**，使用了 **32k thinking tokens**。
- ****3.7 Thinking 的成本受到质疑！****：使用带有 thinking tokens 的 **Sonnet 3.7** 的成本引发了争议，一些人认为性能的提升不值得 **3 倍的价格上涨**。
   - 一位用户指出，*多付 3 倍价格仅提升 0.9% 是不合理的……我原本希望 sonnet-3.7 能在这个基准测试中取得碾压性优势*。
- ****Claude Code：Anthropic 发布 Aider 的衍生工具！****：**Anthropic** 发布了 **Claude Code**，这是一款被一些人认为是 [Aider 克隆版](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)的编程工具，但与 Aider 相比，它似乎存在一些局限性。
- ****OpenAI 完蛋了吗？****：成员们表示，**Claude 3.7** 甚至在使用带有水印的图像时也通过了他们的几何测试，而 **OpenAI** 则失败了。
   - 成员们反馈了代码质量的优越性，并对 **Claude 3.7** 的未来充满期待。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/cursor_ai/status/1894093436896129425">来自 Cursor (@cursor_ai) 的推文</a>：Sonnet-3.7 已在 Cursor 中可用！我们对其编程能力印象深刻，尤其是在现实世界的 Agent 任务中。它似乎代表了新的 SOTA。</li><li><a href="https://livebench.ai/#/">LiveBench</a>：未找到描述</li><li><a href="https://aider.chat/docs/llms/warnings.html">模型警告</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/usage/copypaste.html">通过 Web 聊天进行复制/粘贴</a>：Aider 兼容 LLM Web 聊天 UI</li><li><a href="https://openrouter.ai/anthropic/claude-3-7-sonnet">Claude 3.7 Sonnet - API、提供商、统计数据</a>：Claude 3.7 Sonnet 是一款先进的大型语言模型，具有改进的推理、编码和问题解决能力。通过 API 运行 Claude 3.7 Sonnet</li><li><a href="https://docs.anthropic.com/en/docs/about-claude/models/all-models#model-comparison-table">所有模型概览 - Anthropic</a>：未找到描述</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://tenor.com/view/grito-ahhhh-hongo-gif-20006750">Grito Ahhhh GIF - Grito Ahhhh Hongo - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/sweaty-speedruner-gif-20263880">Sweaty Speedruner GIF - Sweaty Speedruner - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/drago-ivan-i-must-break-you-rocky-break-warning-gif-11521068">Drago Ivan I Must Break You GIF - Drago Ivan I Must Break You Rocky - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking">使用扩展思维进行构建 - Anthropic</a>：未找到描述</li><li><a href="https://x.com/adonis_singh/status/1894100291345150107?s=19">来自 adi (@adonis_singh) 的推文</a>：伙计，我只是问它有多少个 r，Claude Sonnet 3.7 居然为我搭建了一个交互式学习平台让我自己去学 😂</li><li><a href="https://x.com/AnthropicAI/status/1894092430560965029">来自 Anthropic (@AnthropicAI) 的推文</a>：介绍 Claude 3.7 Sonnet：我们迄今为止最智能的模型。它是一个混合推理模型，可以产生近乎即时的响应或扩展的、逐步的思考。一个模型，两种思考方式。</li><li><a href="https://docs.anthropic.com/en/api/rate-limits;">首页 - Anthropic</a>：未找到描述</li><li><a href="https://www.anthropic.com/contact-sales">联系 Anthropic</a>：Anthropic 是一家 AI 安全和研究公司，致力于构建可靠、可解释且可控的 AI 系统。</li><li><a href="https://x.com/anthropicai/status/1894092430560965029?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">来自 Anthropic (@AnthropicAI) 的推文</a>：介绍 Claude 3.7 Sonnet：我们迄今为止最智能的模型。它是一个混合推理模型，可以产生近乎即时的响应或扩展的、逐步的思考。一个模型，两种思考方式。</li><li><a href="https://news.ycombinator.com/item?id=43163011)">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1343633346726465629)** (63 messages🔥🔥): 

> `Architect mode configuration, O3-mini access, OpenRouter benefits, Aider Compact Command, Claude 3.7 in Aider` 


- **Architect 模式配置说明**：用户讨论了在 aider 中使用 `o1-preview` 作为 **Architect** 模型以及 `o1-mini` 作为 **Editor** 模型的配置，确认 `model: o1-preview`、`editor-model: o1-mini` 和 `architect: true` 是正确的设置，正如[此处文档](https://aider.chat/docs/usage/modes.html#architect-mode-and-the-editor-model)所述。
   - 建议在 `ask` 模式下使用更强大的模型，并根据具体任务在运行时使用 `/model` 根据需要更改模型。
- **通过 OpenRouter 解锁 O3-Mini**：成员们讨论了通过 [OpenRouter](https://openrouter.ai/openai/o3-mini-high) 访问 `o3-mini-high` 模型，这是一个针对 STEM 推理任务优化的具有成本效益的语言模型，并指出它与将推理强度（reasoning effort）设置为高的 `o3-mini` 相同。
   - 一位用户表示，编码会话使用 **2 小时** 的费用约为 **$3**，并且 OpenRouter 可以绕过速率限制（rate limits），并提供访问多个供应商的单一入口。
- **对 Compact 命令的需求**：一位用户表达了对类似 `claude-code` 的 `/compact` 命令的兴趣，以管理消息历史上下文，同时赞扬了 aider 的文件上下文控制。
   - 该用户承认，尽管对文件上下文有控制权，但在管理消息历史上下文方面仍存在困难。
- **Claude 3.7 和 Bedrock 的故障排除**：成员们目前正在讨论在 aider 中实现 **Claude 3.7**（包括 “thinking” 模式），特别是在使用 Bedrock 时。
   - 一位用户提供了使用 `bedrock-runtime` 进行 hello world 的示例命令行代码，并正在寻求建议以使其在 aider 中完全运行；另一位用户正试图为 Editor 模型关闭推理，同时为 Architect 模型保留推理。
- **Aider 自动拉取 Git 变更**：一位用户询问如何在 Aider 中自动拉取远程 Git 仓库的变更以保持本地版本同步，希望通过 prompt 之外的标志（flag）来触发它。
   - 另一位用户建议使用定期运行 `git pull` 的独立 bash 脚本或探索 webhooks，而 Aider 拥有的 `/git` 命令也可能有所帮助。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/openai/o3-mini-high">o3 Mini High - API, Providers, Stats</a>：OpenAI o3-mini-high 与将 reasoning_effort 设置为高的 [o3-mini](/openai/o3-mini) 是同一个模型。o3-mini 是一款针对 STEM 推理任务优化的经济型语言模型，尤其擅长...</li><li><a href="https://aider.chat/2024/12/21/polyglot.html">o1 在 aider 新的多语言排行榜中夺冠</a>：o1 在 aider 新的、更具挑战性的多语言编码基准测试中获得了最高分。</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://aider.chat/docs/usage/modes.html#architect-mode-and-the-editor-model">聊天模式</a>：使用 code、architect、ask 和 help 聊天模式。</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html">高级模型设置</a>：为 LLM 配置高级设置。</li><li><a href="https://github.com/Aider-AI/aider/blob/0ba1e8f90435aa2c08360d152fe8e16f98efd258/aider/coders/architect_coder.py#L21">GitHub 上的 aider/aider/coders/architect_coder.py</a>：aider 是你终端里的 AI 配对编程助手。欢迎在 GitHub 上通过创建账号为 Aider-AI/aider 的开发做出贡献。
</li>
</ul>

</div>

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1343685939846713436)** (2 条消息): 

> `Hacker News Wrapped, Kagi LLM Benchmarking Project, Claude Sonnet 3.7` 


- **HN 个人资料被吐槽！**：用户现在可以使用 **Claude Sonnet 3.7** 分析他们的 [Hacker News 个人资料](https://hn-wrapped.kadoa.com/)，以获取亮点和趋势。
   - 据一位成员称，该分析据称*准确得吓人*，他将 LLM 对其帖子历史的深入挖掘描述为一场“吐槽（roast）”。
- **Kagi 发布 LLM 基准测试项目**：**Kagi** 推出了 [Kagi LLM Benchmarking Project](https://help.kagi.com/kagi/ai/llm-benchmark.html)，用于评估主流大语言模型 (**LLMs**) 的推理、编码和指令遵循能力，最后更新日期为 **2025 年 2 月 24 日**。
   - 该基准测试使用频繁变动且大多是新颖的测试，以对模型能力进行严格评估，旨在避免基准测试过拟合。
- **Gemini 2.0 Pro 超越 Claude Sonnet 3.7 和 GPT-4o**：**Kagi LLM Benchmarking Project** 的结果显示，**Google 的 gemini-2.0-pro-exp-02-05** 达到了 **60.78%** 的准确率，超过了 **Anthropic 的 claude-3-7-sonnet-20250219** (**53.23%**) 和 **OpenAI 的 gpt-4o** (**48.39%**)。
   - **Gemini 2.0 Pro** 还展示了 **1.72s** 的中值延迟和 **51.25 tokens/sec** 的速度，相比之下，**Claude Sonnet 3.7** 为 **2.82s** 和 **54.12 tokens/sec**，而 **GPT-4o** 为 **2.07s** 和 **4 tokens/sec**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://hn-wrapped.kadoa.com/">HN Wrapped</a>: AI 分析你的 HN 个人资料并为你提供 2024 年回顾</li><li><a href="https://help.kagi.com/kagi/ai/llm-benchmark.html">Kagi LLM Benchmarking Project | Kagi's Docs</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1343629844789067806)** (15 条消息🔥): 

> `Vim 中的 Codeium 聊天, Codeium Discussion 频道用途, Codeium 3.7 发布` 


- **Vim Chat 问题浮现**：一位成员报告了在通过 **Putty SSH** 会话在 **Vim** 中启动 **Codeium Chat** 时遇到的问题，在尝试通过浏览器访问提供的 URL 时遇到了连接错误。
   - 错误信息显示 *"This site can't be reached 127.0.0.1 refused to connect"*（无法访问此网站 127.0.0.1 拒绝连接）。
- **频道澄清消除困惑**：成员们澄清了 **Codeium Discussion** 频道的用途，指出它是为适用于 **VS Code, Neovim, JetBrains editors, and Emacs** 的 **Codeium extension** 而设立的。
   - 有人建议使用 *codeium.com/support* 获取专门支持。
- **Codeium 发布日期仍存疑问**：一位成员询问了 **Codeium 3.7** 的发布时间表。
   - 另一位成员暗示发布的可能性为 *"0"*。


  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1343629265371009164)** (675 条消息🔥🔥🔥): 

> `Cascade UI 错误, Claude 3.7 Sonnet, 模型对比, Deepseek 幻觉, Windsurf 开发沟通` 


- **Cascade 显示 Diff 的方式发生变化，用户表示担忧**：用户报告称 Cascade 现在将建议显示为 **diffs** 而不是可编辑区域，需要通过 `git restore` 来拒绝更改；另一位用户建议这可能是由于 **聊天记录过长** 或 Cascade 处理来自 o3/R1 响应的方式导致的。
   - 有用户建议开启新聊天以恢复“接受/拒绝（ACCEPT/REJECT）”工作流。
- **对 Claude 3.7 到来的急躁情绪增加**：成员们正迫切等待 **Claude 3.7** 集成到 Windsurf 中，部分用户对相比 Cursor 和 T3 等其他平台的延迟感到沮丧，许多人希望 **尽快（ASAP）** 添加 3.7。
   - 成员们要求 *Windsurf 应该去做早期测试* —— 开发人员正在努力将 **Claude 3.7** 推向生产环境，可能在当天结束前发布。
- **Deepseek 遭遇用户提示词幻觉**：一位用户报告 **Deepseek** 会幻觉出用户请求，并根据这些幻觉出的请求实施更改。
   - AI 机器人 *编造了自己的用户提示词，然后开始根据那个幻觉出的提示词实施更改 😆。*
- **Windsurf 开发沟通遭到批评**：部分用户对 Windsurf 开发人员关于 Claude 3.7 集成缺乏沟通感到沮丧，一位用户表示：*“部分挫败感源于开发人员没有任何沟通。”*
   - 其他用户则为 Windsurf 辩护，并指出在更稳定时发布可以规避商业风险，*“实现速度快并不意味着它稳固。”*
- **用户质疑 MCP Server 的实用性**：用户讨论了 **MCP server** 的实际用途，示例包括集成 **Jira 工单**、分享自定义应用以及使用云服务。
   - 成员们询问：*“大家在实际中把 MCP server 用在什么地方？有没有什么让生活变得非常轻松的真实案例？我想不出任何例子。”*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/surf-glider-wave-giant-wave-wind-gif-15418238">Surf Glider GIF - Surf Glider Wave - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.anthropic.com/news/claude-3-7-sonnet">Claude 3.7 Sonnet 和 Claude Code</a>：今天，我们发布了 Claude 3.7 Sonnet，这是我们迄今为止最智能的模型，也是市场上首个普遍可用的混合推理模型。</li><li><a href="https://tenor.com/view/stan-twitter-monkey-meme-monki-monke-monkey-waiting-gif-12661622482574205246">Stan Twitter Monkey Meme GIF - Stan twitter Monkey meme Monki - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://docs.anthropic.com/en/docs/about-claude/models/all-models?utm_source=iterable&utm_medium=email&utm_campaign=sonnet_3-7_launch&campaignId=12703046&source=i_email&medium=email&content=Dec20241P&messageTypeId=140367">所有模型概览 - Anthropic</a>：未找到描述</li><li><a href="https://tenor.com/view/good-juju-witch-good-vibes-sending-love-and-light-hive-gif-20508559">Good Juju Witch GIF - Good Juju Witch Good Vibes - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/jim-carrey-jim-carrey-typing-jim-carrey-typing-angry-jim-carrey-typing-fast-fasttyping-gif-22737012">Jim Carrey Jim Carrey Typing GIF - Jim Carrey Jim Carrey Typing Jim Carrey Typing Angry - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/let-them-cook-let-them-fight-godzilla-godzilla-2014-meme-gif-10523835079864650811">Let Them Cook Let Them Fight GIF - Let them cook Let them fight Godzilla - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://codeium.canny.io/feature-requests">功能请求 | Codeium</a>：向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://codeium.canny.io/">Codeium 反馈</a>：向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://tenor.com/bENEo.gif">Chewing Character Hd GIF - Chewing Character Chewing Character - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://codeium.com/support">支持 | Windsurf 编辑器和 Codeium 扩展</a>：需要帮助？联系我们的支持团队以获得个性化协助。</li><li><a href="https://codeium.com/changelog">Windsurf 编辑器更新日志 | Windsurf 编辑器和 Codeium 扩展</a>：Windsurf 编辑器的最新更新和更改。
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1343640915939561572)** (611 messages🔥🔥🔥): 

> `Grok 3, Perplexity Comet agentic browser, Claude 3.7 Sonnet, Claude Code, GPT-4.5 release` 


- **Grok 3 冗长但创意十足**：成员们发现 **Grok 3** 尽管在提示词中要求简洁，但回复仍然过于冗长，不过它在 **编程和创意** 方面表现出了 **强大的实力**。
   - 一位成员评论说，他们正在转向 Grok，因为它 *开箱即用的审查更少*。
- **Perplexity Comet，即将到来的 Agentic Browser**：据一位成员透露，Perplexity 即将推出 **Comet**，这是一款全新的 **agentic browser**（智能体浏览器），类似于 The Browser Company 的工作。
   - **agentic browser** 领域正在升温，出现了更多竞争对手。
- **Claude 3.7 Sonnet 带着 Thinking Mode 亮相**：**Anthropic** 刚刚发布了 **Claude 3.7 Sonnet**，该模型在编程和前端 Web 开发方面有所提升，并引入了一个用于 Agent 编程的命令行工具：**Claude Code** [在此发布公告](https://www.anthropic.com/news/claude-3-7-sonnet)。
   - 一位用户指出，该模型的知识截止日期为 **2025 年 2 月 19 日**。
- **Claude Code，作为研究预览版发布的终端 AI 工具**：**Claude Code** 是一款运行在终端的 Agent 编程工具，它理解你的代码库，并通过自然语言命令帮助你更快地编写代码 [在此查看概览](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)。
   - 然而，它目前是一个 **有限的研究预览版**，并且独立于 Pro 或 Anthropic 订阅。
- **GPT-4.5 发布传闻**：成员们热切期待 **GPT-4.5**，其中一人开玩笑说 Windsurf 已经 **正式** 确认 Claude 3.7 Sonnet 将在 1-2 天内发布。
   - **GPT-4.5** 的发布可能即将到来，成员们正在讨论并将其潜在能力与当前模型进行比较。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://rednuht.org/genetic_cars_2/">HTML5 Genetic Algorithm 2D Car Thingy - 推荐使用 Chrome</a>：未找到描述</li><li><a href="https://kodub.itch.io/polytrack">PolyTrack by Kodub</a>：一款高速低多边形赛车游戏。</li><li><a href="https://www.anthropic.com/news/claude-3-7-sonnet">Claude 3.7 Sonnet 和 Claude Code</a>：今天，我们宣布推出 Claude 3.7 Sonnet，这是我们迄今为止最智能的模型，也是市场上首个普遍可用的混合推理模型。</li><li><a href="https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview">Claude Code 概览 - Anthropic</a>：未找到描述</li><li><a href="https://fxtwitter.com/apples_jimmy/status/1893835336913973438">Jimmy Apples 🍎/acc (@apples_jimmy) 的推文</a>：Die RevancheTomorrow。</li><li><a href="https://fxtwitter.com/i/status/1894106441536946235">Rowan Cheung (@rowancheung) 的推文</a>：Anthropic 刚刚发布了 Claude 3.7 Sonnet，这是世界上最好的编程 AI 模型。我是早期测试者，它让我大受震撼。它通过一个提示词就创建了这个 Minecraft 克隆版，并使其立即运行...</li><li><a href="https://llm-stats.com/">LLM Leaderboard 2025 - 比较 LLM</a>：包含基准测试、价格和能力的综合 AI (LLM) 排行榜。通过交互式可视化、排名和对比来比较领先的 LLM。</li><li><a href="https://x.com/alexalbert__/status/1894095781088694497">Alex Albert (@alexalbert__) 的推文</a>：我们正在开放对我们正在构建的新 Agent 编程工具 Claude Code 的有限研究预览访问。你将直接在终端获得由 Claude 驱动的代码辅助、文件操作和任务执行...</li><li><a href="https://www.youtube.com/watch?v=TxANYMqd8cY"> - YouTube</a>：未找到描述</li><li><a href="https://x.com/anthropicai/status/1894092430560965029">Anthropic (@AnthropicAI) 的推文</a>：介绍 Claude 3.7 Sonnet：我们迄今为止最智能的模型。它是一个混合推理模型，可以产生近乎瞬时的响应或扩展的、逐步的思考。一个模型，两种思考方式。</li><li><a href="https://www.reddit.com/r/mlscaling/comments/146rgq2/chatgpt_is_running_quantized/">Reddit - 深入了解一切</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1343646225181835345)** (9 条消息🔥): 

> `O3 问题，在 Discord 发布截图，Bug 报告` 


- **O3 延迟困境**：一位用户报告了 **O3** 的问题，模型显示 *reasoning success*（推理成功）后，延迟长达 10 秒才显示全文，这一问题影响了包括 **O1 Pro** 在内的多种模型。
   - 他们提到这些问题通常发生在 **3pm-7pm EST** 之间，文本有时会出现在非预期的设备上，并询问了为何无法在聊天中发布截图。
- **Discord 截图技巧**：一名成员指出，截图发布功能是特定于频道的，并建议使用像 <#989157702347411466> 这样支持截图的频道。
   - 他们建议在那里发布截图，并在当前频道引用该讨论。
- **Bug 报告盛宴**：一名成员建议使用 <#1070006915414900886> 进行 Bug 报告，并提供了关于如何发布新 Bug 报告的 [说明](https://discord.com/channels/974519864045756446/1295975636061655101/1295979750212505640)。
   - 他们还建议先浏览现有报告，如果与用户的情况高度匹配，则在现有报告下发表评论。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1343628545712455793)** (345 条消息🔥🔥): 

> `付费版主，CUDA 错误，Qwen2.5 VL 72B，Claude 3.7，DeepSeek MLA` 


- **税务建议被禁**：一名用户因讨论通过开具发票设立公司以避税而被禁言，并被告知不容忍任何避税建议。
   - 另一名用户回应称：*我开票的那家公司说我申报收入很愚蠢*。
- **Colab CUDA Kernel 错误**：一名用户报告了在 T4 Google Colab 上的 CUDA 错误，具体为 *an illegal memory access was encountered*，并被建议设置 `CUDA_LAUNCH_BLOCKING=1` 并使用 `TORCH_USE_CUDA_DSA` 进行编译以进行调试。
   - 另一名用户提到看到 grad norm 出现高达 2000 的异常峰值，暗示模型可能已经崩溃，且训练/损失曲线看起来不健康。
- **Qwen2.5-VL-72B 导致内存错误**：一名用户尝试在 48GB 显存上运行 **Qwen2.5 VL 72B**，在 context length 为 32K 时遇到显存溢出错误，另一名用户建议尝试 8k 或将 KV cache 量化为 fp8。
   - 该用户随后成功以 8k context length 加载了模型，并指出有必要从中提取模型的 *thinking traces*。
- **在 Llama 上实现 DeepSeek MLA**：用户讨论了在 **Llama** 模型上实现 **DeepSeek** 的 **Multi-Head Latent Attention (MLA)** 的可能性，一名用户指出这需要使用不同的 Attention 机制重新训练模型。
   - 随后，一名用户链接到了 [fxmeng/TransMLA](https://github.com/fxmeng/TransMLA)，这是一种将基于 GQA 的预训练模型转换为 MLA 模型训练后方法。
- **rslora 的高 Rank 稳定性**：用户讨论了 **rslora** 及其在解决高 rank 稳定性方面的数值稳定性作用，因为它执行单个 `sqrt`，如果 rank 过大，则需要一个修正项。
   - 一名用户建议如果 `r/a = 1`，**rslora** 可能会让情况变得更糟，并建议保持 `r/a = 1` 并完全避免使用 **rslora**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

</div>

<ul>
<li>
<a href="https://huggingface.co/BarraHome/llama3.2-1b-mla">BarraHome/llama3.2-1b-mla · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/guinea-pig-chewing-chew-cavy-bertold-gif-13907739970483938206">豚鼠咀嚼 GIF - Guinea pig Chewing Chew - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html">如何通过将 optimizer step 融合到 backward pass 中来节省显存 — PyTorch Tutorials 2.6.0+cu124 文档</a>: 未找到描述</li><li><a href="https://tenor.com/view/teach-you-yoda-star-wars-mentor-teach-you-i-will-gif-13942585">教导你尤达 GIF - Teach You Yoda Star Wars - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/danielhanchen/status/1894212351932731581">来自 Daniel Han (@danielhanchen) 的推文</a>: DeepSeek 第 2 个 OSS 发布！MoE kernels、expert parallelism，训练和推理均支持 FP8！引用 DeepSeek (@deepseek_ai) 🚀 #OpenSourceWeek 第 2 天：DeepEP 很高兴介绍 DeepEP - 第一个...</li><li><a href="https://github.com/vllm-project/vllm/tree/db986c19ea35d7f3522a45d5205bf5d3ffab14e4/benchmarks">vllm/benchmarks 位于 db986c19ea35d7f3522a45d5205bf5d3ffab14e4 · vllm-project/vllm</a>: 一个用于 LLMs 的高吞吐量且显存高效的推理与服务引擎 - vllm-project/vllm</li><li><a href="https://arxiv.org/abs/2502.14837">迈向经济型推理：在任何基于 Transformer 的 LLMs 中启用 DeepSeek 的 Multi-Head Latent Attention</a>: Multi-head Latent Attention (MLA) 是由 DeepSeek 提出的一种创新架构，旨在通过将 Key-Value (KV) cache 显著压缩为... 来确保高效且经济的推理。</li><li><a href="https://github.com/JT-Ushio/MHA2MLA">GitHub - JT-Ushio/MHA2MLA: 迈向经济型推理：在任何基于 Transformer 的 LLMs 中启用 DeepSeek 的 Multi-Head Latent Attention</a>: 迈向经济型推理：在任何基于 Transformer 的 LLMs 中启用 DeepSeek 的 Multi-Head Latent Attention - JT-Ushio/MHA2MLA</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)">CUDA 语义 — PyTorch 2.6 文档</a>: 未找到描述</li><li><a href="https://github.com/fxmeng/TransMLA">GitHub - fxmeng/TransMLA: TransMLA: Multi-Head Latent Attention Is All You Need</a>: TransMLA: Multi-Head Latent Attention Is All You Need - fxmeng/TransMLA</li><li><a href="https://github.com/vllm-project/aibrix">GitHub - vllm-project/aibrix: 用于 GenAI 推理的经济高效且可插拔的基础设施组件</a>: 用于 GenAI 推理的经济高效且可插拔的基础设施组件 - vllm-project/aibrix</li><li><a href="https://github.com/JT">jt - 概览</a>: jt 有 4 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide">微调指南 | Unsloth 文档</a>: 学习微调的所有基础知识。</li><li><a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl">推理 - GRPO &amp; RL | Unsloth 文档</a>: 使用 Unsloth 通过 GRPO 训练你自己的 DeepSeek-R1 推理模型。</li><li><a href="https://github.com/facebookresearch/optimizers/tree/main">GitHub - facebookresearch/optimizers: 用于优化算法的研究与开发。</a>: 用于优化算法的研究与开发。 - facebookresearch/optimizers
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/)** (1 条消息): 

deoxykev: 新的 qwq https://qwenlm.github.io/blog/qwq-max-preview/
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1343643545554260069)** (121 条消息🔥🔥): 

> `Mac 上的 Unsloth, GRPO Qwen notebook 问题, CUDA 显存溢出 (Out of Memory), ShareGPT 数据集格式, 强制从 VRAM 卸载` 


- **强大的 Mac 可能会错过模型：Unsloth 的 Mac 兼容性难题**：虽然你*可以*在 Mac 上使用 **Ollama** 或 **Jan AI** 运行模型，但目前**还不能**在 Mac 设备上使用 **Unsloth** 进行微调，尽管团队正在努力解决；用户指出 **MLX** 是值得探索的方向。
   - 一位用户建议使用外接显卡坞 **GPU**，或者租用 GPU 服务器，如 **Tensordock**（48GB 服务器价格为 0.95 美元）或使用 Google 提供的免费 4T 显存作为绕过限制的方法。
- **Qwen 查询困境：GRPO Notebooks 与 vLLM 的差异**：用户报告称，如果没有 **vLLM**，**GRPO Qwen notebook** 会退化为无意义的回答，但在使用 **vLLM** 时功能正常。
   - 一位用户在此处附带了使用 **vLLM** 的[截图示例](https://cdn.discordapp.com/attachments/1179777624986357780/1343645411784786052/Screenshot_2025-02-24_at_6.06.36_PM.png?ex=67be06cd&is=67bcb54d&hm=1ecbf8d8cbccd96ea8bc2092947399576d31d2decad64da554728ed7f6095175&)。
- **VRAM 消失之旅：Unsloth 的内存峰值之谜**：**Unsloth** 在每次开始保存模型时，**VRAM** 使用量都会飙升至两倍，但*仅在*模型开始保存时发生，而不在训练过程中的任何其他时间点。
   - 该用户缩小了导致其 **CUDA** 显存溢出崩溃的原因范围，并被建议将其放入 showcase 中；一位开发者表示，他们*将在未来几周内重写该部分，除非你想更早提交 PR；我正在处理一个需要更健壮的转换/上传的部分*。
- **ShareGPT 混乱解决：为数据集格式化数据**：一位用户在处理自己的数据集时，不确定是否必须按照指南格式化数据，还是可以采用其他方式。
   - 该用户被引导参考 notebook 作为指南，并观察到提到的 notebook 使用了 **ShareGPT** 格式，同时建议查看[文档](https://docs.unsloth.ai/get-started/unsloth-notebooks)。
- **VRAM 假期之旅：强制 Unsloth 卸载以进行转换**：一位用户询问如何在创建最终 Checkpoint 之后、但在保存为 **GGUF** 之前，强制 **Unsloth** 从 **VRAM** 中卸载。
   - 回复称 **VRAM** 不是问题所在，而是保存为 **Lora** 以及保存为 **GGUF** 时会将其完全加载到 **VRAM** 中。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>: 以下是我们所有 notebook 的列表：</li><li><a href="https://search.app/YgSmHDHmwPcJubBH6">安装 + 更新 | Unsloth 文档</a>: 学习在本地或在线安装 Unsloth。</li><li><a href="https://github.com/unslothai/unsloth/issues/685">Unsloth On Mac · Issue #685 · unslothai/unsloth</a>: 我有一台 Macbook，当我在上面运行模型时，基本会收到一个错误，提示找不到 CUDA 设备。我知道 Macbook 上没有 GPU，这是否意味着我无法运行我的模型...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1343669893991628830)** (1 条消息): 

> `Claude 3.7 Sonnet, Extended Thinking, 价格与可用性` 


- **Claude 3.7 Sonnet 登陆 OpenRouter**：**Claude 3.7 Sonnet** 现已在 OpenRouter 上可用，提供一流的性能，重点关注[数学推理、编程和复杂问题解决](https://www.anthropic.com/news/claude-3-7-sonnet)。
- **Extended Thinking 即将上线**：**Extended Thinking** 功能即将登陆 OpenRouter API，支持复杂任务的分步处理，详情见 [Anthropic 文档](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)。
- **Claude 3.7 Sonnet：价格公布**：**Claude 3.7 Sonnet** 的定价设定为 **每百万 Input Token 3 美元**，**每百万 Output Token 15 美元**（包括 Thinking Token），发布时即提供完整的缓存支持。



**提到的链接**: <a href="https://openrouter.ai/anthropic/claude-3.7-sonnet">Claude 3.7 Sonnet - API, 提供商, 统计数据</a>: Claude 3.7 Sonnet 是一款先进的大语言模型，具有改进的推理、编程和问题解决能力。通过 API 运行 Claude 3.7 Sonnet。

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1343629628564049930)** (346 条消息🔥🔥): 

> `Claude 3.7 Sonnet, GCP hosting Claude 3.7 Sonnet, OpenRouter rate limits, Claude 3.5 Haiku with vision, TPUs vs GPUs for inference` 


- **GCP 为 Claude 3.7 发布做准备**：**Google Cloud Platform (GCP)** 正准备支持 **Claude 3.7 Sonnet**，将在 **us-east5** 和 **europe-west1** 区域上线，模型 ID 为 `claude-3-7-sonnet@20250219`。
- **Claude 3.7 亮相：性能与定价**：**Claude 3.7 Sonnet** 采用 **hybrid reasoning approach**，提供标准和 extended thinking 两种模式。在标准模式下保持与前代产品相当的性能，同时提高了处理复杂任务的准确性，详见 [Anthropic 的博客文章](https://www.anthropic.com/news/claude-3-7-sonnet)。
   - 该模型价格为每百万输入 tokens **$3**，每百万输出 tokens **$15**，拥有 **200,000 token 上下文窗口**，尽管一些用户认为其输出定价可能会引发抱怨。
- **思考支持：仍在开发中**：OpenRouter 正在积极实现对 **Claude 3.7** *extended thinking* 功能的全支持，该功能目前不支持 pre-fills，目标是尽快发布并更新文档。
- **OpenRouter 提升 Claude 3.7 额度**：OpenRouter 增加了 `anthropic/claude-3.7-sonnet` 的 **TPM (tokens per minute)**，而 `anthropic/claude-3.7-sonnet:beta` 初始 TPM 较低，随着用户从 **3.5** 迁移，该额度将会提升。
- **API Key 安全须知**：提醒用户 **API keys 不包含 credits**；删除 key 仅会撤销访问权限，而 credits 仍与账户绑定，但由于安全措施，丢失的 key 无法找回。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aws.amazon.com/ai/machine-learning/trainium/">AI Accelerator - AWS Trainium - AWS</a>：未找到描述</li><li><a href="https://docs.anthropic.com/en/docs/about-claude/models/extended-thinking-models">Extended thinking models - Anthropic</a>：未找到描述</li><li><a href="https://openrouter.ai/anthropic/claude-3.7-sonnet">Claude 3.7 Sonnet - API, Providers, Stats</a>：Claude 3.7 Sonnet 是一款先进的大型语言模型，具有改进的推理、编程和问题解决能力。通过 API 运行 Claude 3.7 Sonnet</li><li><a href="https://tenor.com/view/ponke-ponkesol-solana-sol-bored-gif-1576815656973460219">Ponke Ponkesol GIF - Ponke Ponkesol Solana - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/telmo-coca-harina-raquetaso-esnifar-gif-25660568">Telmo Coca GIF - Telmo Coca Harina - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.cnbc.com/2025/01/22/google-agrees-to-new-1-billion-investment-in-anthropic.html">Google agrees to new $1 billion investment in Anthropic</a>：Google 已同意向生成式 AI 初创公司 Anthropic 追加超过 10 亿美元的新投资，一位知情人士向 CNBC 证实了这一消息。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1343628723240308746)** (304 条消息🔥🔥): 

> `Meta AI Expansion, Claude 3.7 Sonnet Release, Claude Code Tool, Qwen Chat Release, DeepEP`

- **Meta AI 进军 MENA**：[Meta AI](https://about.fb.com/news/2025/02/meta-ai-launches-in-the-middle-east-empowering-new-era-of-creativity-and-connection/) 已正式扩展至**中东和北非 (MENA)**，现已支持**阿拉伯语**，并可在 **Instagram、WhatsApp 和 Messenger** 上访问。
- **Claude 3.7 Sonnet 及其扩展思维功能发布**：Anthropic 推出了 **Claude 3.7 Sonnet**，这是一款具有近乎即时响应和可见的逐步思考过程的**混合推理模型**，同时推出了 **Claude Code**——一款用于 Agent 式编程的命令行工具（目前处于有限研究预览阶段），定价为**每百万输入 token 3 美元，每百万输出 token 15 美元**。
   - 研究人员注意到 Claude 的思考过程与他们自己的思考过程“惊人地相似”，会探索不同的角度并反复检查答案；[博客文章](https://www.anthropic.com/research/visible-extended-thinking)指出，他们将权衡在未来版本中公开思考过程的利弊。
- **Sonnet 的可见扩展思维**：Anthropic 通过其全新的 [可见扩展思维模式 (Visible Extended Thinking mode)](https://www.anthropic.com/research/visible-extended-thinking) 功能，允许模型在得出答案时给自己更多时间并投入更多精力。
   - 该功能在 **GPQA** 评估（一套常用的关于生物、化学和物理的挑战性问题集）中利用并行测试时计算缩放实现了显著的提升。
- **QwQ-Max 与 Qwen 2.5 Max：Apache 的反击**：**阿里巴巴 Qwen** 在 **Qwen Chat** 中发布了 "Thinking (QwQ)"，由其 **QwQ-Max-Preview** 提供支持，这是一款基于 **Qwen2.5-Max** 的推理模型，采用 **Apache 2.0** 协议授权。
   - 该模型将提供更小的变体（例如 **QwQ-32B**）用于本地部署，他们在一段展示模型推理能力的 [爆火 Twitter 演示](https://fxtwitter.com/Alibaba_Qwen/status/1894130603513319842) 中强调了改进后的数学、编程和 Agent 能力。
- **Co-Scientist 的奇特案例**：研究发现，基于 **Gemini LLM** 的 **Google Co-Scientist AI** 工具曾被其协助的团队**喂过一篇 2023 年的论文**，该论文中包含了一个版本的假设，而该 AI 工具随后将其作为解决方案提出。这篇 [文章](https://pivot-to-ai.com/2025/02/22/google-co-scientist-ai-cracks-superbug-problem-in-two-days-because-it-had-been-fed-the-teams-previous-paper-with-the-answer-in-it/) 指出，BBC 的报道未能提及这一点。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/cursor_ai/status/1894093436896129425">来自 Cursor (@cursor_ai) 的推文</a>: Sonnet-3.7 已在 Cursor 中上线！我们对其编程能力印象深刻，特别是在现实世界的 Agent 任务中。它似乎成为了新的最先进水平。</li><li><a href="https://qwenlm.github.io/blog/qwq-max-preview/">&lt;think>...&lt;/think> QwQ-Max-Preview | Qwen</a>: 未找到描述</li><li><a href="https://x.com/ChujieZheng/status/1894095584774250858">来自 Chujie Zheng (@ChujieZheng) 的推文</a>: 兄弟，你在开玩笑吗？</li><li><a href="https://www.anthropic.com/news/visible-extended-thinking">Claude 的扩展思考</a>: 讨论 Claude 的新思考过程</li><li><a href="https://www.anthropic.com/news/claude-3-7-sonnet">Claude 3.7 Sonnet 和 Claude Code</a>: 今天，我们宣布推出 Claude 3.7 Sonnet，这是我们迄今为止最智能的模型，也是市场上首个普遍可用的混合推理模型。</li><li><a href="https://www.oneusefulthing.org/p/a-new-generation-of-ais-claude-37">新一代 AI：Claude 3.7 和 Grok 3</a>: 是的，AI 突然又变得更强大了……再次进化</li><li><a href="https://x.com/din0s_/status/1894102686984818863">来自 dinos (@din0s_) 的推文</a>: 一张截图看懂 Anthropic</li><li><a href="https://www.oneusefulthing.org/p/a-new-generation-of-ais-claude-37#footnote-1-157729795">新一代 AI：Claude 3.7 和 Grok 3</a>: 是的，AI 突然又变得更强大了……再次进化</li><li><a href="https://x.com/arankomatsuzaki/status/1894101923151692157">来自 Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>: Claude 3.7 Sonnet System Card 刚刚发布！</li><li><a href="https://x.com/TheXeophon/status/1894113897797288215">来自 Xeophon (@TheXeophon) 的推文</a>: Sonnet 3.7 Thinking（预算为 16K tokens）是 Neal 的密码游戏中表现最好的模型，恭喜！它*几乎*通过了第 11 关，但它坚持认为 Wordle 已经解开了 :( 引用 Xeophon (@The...</li><li><a href="https://fxtwitter.com/Alibaba_Qwen/status/1894130619061604651">来自 Qwen (@Alibaba_Qwen) 的推文</a>: Agent</li><li><a href="https://x.com/deepseek_ai/status/1894211757604049133">来自 DeepSeek (@deepseek_ai) 的推文</a>: 🚀 #OpenSourceWeek 第 2 天：DeepEP。很高兴介绍 DeepEP —— 第一个用于 MoE 模型训练和推理的开源 EP 通信库。✅ 高效且优化的 all-to-all 通信 ✅...</li><li><a href="https://x.com/skcd42/status/1894098856805372378">来自 skcd (@skcd42) 的推文</a>: 作为一个使用过它的人，对新 Sonnet 3.7 的评价：- 新的 Sonnet 非常棒，在我们内部的 Rust 评估中，我们看到了 14.7%（约 40%）的提升（该评估由 1k 个问题组成）- 它具有...</li><li><a href="https://x.com/cognition_labs/status/1894125030583537974">来自 Cognition (@cognition_labs) 的推文</a>: 1/ Claude 3.7 Sonnet 已在 Devin 中上线！这个新模型是我们在调试、代码库搜索和 Agent 规划等各种任务中见过的最强模型。</li><li><a href="https://x.com/adonis_singh/status/1894100291345150107">来自 adi (@adonis_singh) 的推文</a>: 兄弟，什么鬼，我只是问它有多少个 r，Claude Sonnet 3.7 居然为我搭建了一个交互式学习平台让我自己去学 😂</li><li><a href="https://x.com/nearcyan/status/1894103654874984906">来自 near (@nearcyan) 的推文</a>: CLAUDE 来了！他回来了，而且比以往任何时候都更好！我将分享我的第一个 Prompt 结果，是关于微分音音乐的 3D 可视化。这是目前世界上最好的模型。许多人将...</li><li><a href="https://x.com/StringChaos/status/1894135561059013023">来自 Naman Jain (@StringChaos) 的推文</a>: 查看 QwQ-Max-Preview 在 LiveCodeBench 上的评估，它的表现与 o1-medium 旗鼓相当🚀！！引用 Qwen (@Alibaba_Qwen) &lt;think&gt;...&lt;/think&gt; QwQ-Max-Preview Qwen Chat: https://chat...</li><li><a href="https://x.com/DimitrisPapail/status/1894127499224694877">来自 Dimitris Papailiopoulos (@DimitrisPapail) 的推文</a>: Claude 3.7 Sonnet 请用 TikZ 画出雅典卫城，结果包含：- 无推理 - 10k 推理 tokens - 30k 推理 tokens - 64k 推理 tokens。引用 Dimitris Papailiopoulos (@Dimitri...</li><li><a href="https://x.com/nrehiew_/status/1894105060759552231">来自 wh (@nrehiew_) 的推文</a>: 具体来说，API 中的 "thinking budget tokens" 参数似乎在预算耗尽之前永远不会采样到思考结束 token。没有 Prompt 调节，没有特...</li><li><a href="https://fxtwitter.com/Alibaba_Qwen/status/1894130603513319842">来自 Qwen (@Alibaba_Qwen) 的推文</a>: &lt;think&gt;...&lt;/think&gt; QwQ-Max-Preview Qwen Chat: https://chat.qwen.ai/ 博客: https://qwenlm.github.io/blog/qwq-max-preview/ 🤔 今天我们在 Qwen Chat 中发布了 "Thinking (QwQ)"，由 o...</li><li><a href="https://x.com/lmarena_ai/status/1894128271568126381">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>: 祝贺 @AnthropicAI 发布 Claude 3.7 Sonnet！👏 快来测试它吧

你在 lmarena 最难的提示词！引用 Anthropic (@AnthropicAI) 介绍 Claude 3.7 Sonnet：我们迄今为止最智能的 ...</li><li><a href="https://x.com/elder_plinius/status/1894110867353899112">来自 Pliny the Liberator 🐉󠅫󠄼󠄿󠅆󠄵󠄐󠅀󠄼󠄹󠄾󠅉󠅭 (@elder_plinius) 的推文</a>：🚂 越狱警报 🚂 ANTHROPIC：被攻破 ✌️😛 CLAUDE-SONNET-3.7：已解放 🗽 哇，新的 Claude 模型！！！🤗 你知道吗，我大约在...写的原始“GODMODE”通用越狱...</li><li><a href="https://x.com/_lewtun/status/1894098741046521904">来自 Lewis Tunstall (@_lewtun) 的推文</a>：终于，有一家 AI 实验室发布了带有正确标签和所有内容的图表 🥹</li><li><a href="https://x.com/btibor91/status/1894113852301721645">来自 Tibor Blaho (@btibor91) 的推文</a>：“目前仅在美国境内发货。” :( 引用 wh (@nrehiew_) 在 Claude Code NPM 源码中有一个“隐藏彩蛋”工具，可以向用户邮寄 Anthropic 贴纸 :)</li><li><a href="https://x.com/paulgauthier/status/1894123992505880688">来自 Paul Gauthier (@paulgauthier) 的推文</a>：Claude 3.7 Sonnet 在不使用思考（thinking）的情况下在 aider polyglot 基准测试中获得了 60% 的分数。与 o3-mini-high 并列第三。Sonnet 3.7 拥有最高的非思考分数（此前为 Sonnet 3.5）。思考模式的结果即将发布...</li><li><a href="https://x.com/AnthropicAI/status/1894095494969741358">来自 Anthropic (@AnthropicAI) 的推文</a>：我们对模型的安全性、防护性和可靠性进行了广泛测试。我们也听取了您的反馈。通过 Claude 3.7 Sonnet，我们将不必要的拒绝率比之前的...降低了 45%。</li><li><a href="https://x.com/DimitrisPapail/status/1894144311232729391">来自 Dimitris Papailiopoulos (@DimitrisPapail) 的推文</a>：好吧，这真的很酷。Claude 3.7 Sonnet 请用 tikz 画出：一个在房子里的人——房子内接于球体——球体内接于立方体——立方体内接于圆柱体...</li><li><a href="https://youtu.be/t3nnDXa81Hs">带有扩展思考功能的 Claude 3.7 Sonnet</a>：介绍 Claude 3.7 Sonnet：我们迄今为止最智能的模型。它是一个混合推理模型，可以产生近乎即时的响应或扩展的、逐步的 ...</li><li><a href="https://techcrunch.com/2025/02/24/meta-ai-arrives-in-the-middle-east-and-africa-with-support-for-arabic/">Meta AI 登陆中东和非洲并支持阿拉伯语 | TechCrunch</a>：Meta AI 已在中东和北非上线并支持阿拉伯语，向数百万更多用户开放了该聊天机器人。</li><li><a href="https://pivot-to-ai.com/2025/02/22/google-co-scientist-ai-cracks-superbug-problem-in-two-days-because-it-had-been-fed-the-teams-previous-paper-with-the-answer-in-it/">Google Co-Scientist AI 在两天内破解了超级细菌问题！——因为它被喂了包含答案的团队之前的论文</a>：Google 出色的新 AI Co-Scientist 工具（基于 Gemini LLM）的炒作周期包括一条 BBC 头条新闻，讲述了帝国理工学院的 José Penadés 团队如何向该工具询问一个问题……</li><li><a href="https://lovattspuzzles.com/online-puzzles-competitions/daily-cryptic-crossword/).">玩 Lovatts 免费在线加密填字游戏 - 每日更新</a>：Lovatts 免费在线加密填字游戏每日更新。包含 7 天谜题存档、提示和计时器。学习加密填字游戏的规则。
</li>
</ul>

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1343670611070816286)** (15 条消息🔥): 

> `Berkeley Advanced Agents MOOC, Tulu 3, RLHF Explanation, AI Startups customer base, mic firmware issues` 


- ****Berkeley Advanced Agents MOOC** 重点介绍 **Tulu 3****：一位成员提到 **"Berkeley Advanced Agents" MOOC** 今天（太平洋标准时间 5 月 30 日下午 4 点）将由 **Hanna Hajishirzi** 讨论 **Tulu 3**，并附上了 [YouTube 视频](https://www.youtube.com/live/cMiu3A7YBks)链接。
- **用类比解释 RLHF**：一位成员分享了一个[解释 RLHF 的推文](https://fxtwitter.com/shaneguML/status/1894131091872891385)链接，该推文使用*类比*向非技术受众进行说明。
   - Kyle Matthews 回应称这“确实是个好类比，哈哈”。
- **成员想要贴纸**：一位成员想要一些**贴纸**，并分享了[贴纸](https://x.com/AndrewCurran_/status/1894152685429108846)的链接，但因为不在美国而无法获得。
   - 随后他们建议尝试 Claude 的做法，但随即反悔，建议“Phil 把贴纸寄给我，我会好好保管它们的”。
- **AI 初创公司客户群受到质疑**：一位成员质疑 **AI 初创公司** 在**硅谷**之外是否有客户。
   - 另一位成员回应道：“更好的问题是技术圈之外”，并且“AI 实验室吹嘘他们在财富 500 强中拥有极高的渗透率”。
- **麦克风固件重置的恶作剧**：一位成员报告说，他们几周前的**音频问题**是由于**麦克风固件**被重置且增益被调低导致的。
   - 此消息没有提供其他上下文。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://hn-wrapped.kadoa.com/Philpax?share">HN Wrapped</a>：AI 分析你的 HN 个人资料并提供 2024 年回顾</li><li><a href="https://fxtwitter.com/shaneguML/status/1894131091872891385">Shane Gu (@shaneguML) 的推文</a>：我如何向非技术受众解释 RLHF</li><li><a href="https://www.youtube.com/live/cMiu3A7YBks">CS 194/294-280 (Advanced LLM Agents) - 第 4 讲，Hanna Hajishirzi</a>：未找到描述</li><li><a href="https://x.com/AndrewCurran_/status/1894152685429108846">Andrew Curran (@AndrewCurran_) 的推文</a>：@repligate 还有这个；
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1343663805409792132)** (1 条消息): 

> `Memes` 


- **发布了图片**：一位成员发布了一张名为 CleanShot_2025-02-24_at_20.20.03.png 的图片，消息内容为 <:3berk:794379348311801876>。
   - 该图片附件来自 [discordapp.com](https://cdn.discordapp.com/attachments/1187551504995987576/1343663805493940285/CleanShot_2025-02-24_at_20.20.03.png?ex=67be17ef&is=67bcc66f&hm=d63a26ffea0251f0a89d80f0409490d15463ee1df13fb3416815e64838320ee3&)。
- **发布了另一张图片**：一位成员发布了一张图片。
   - 该图片附件来自 discordapp.com。


  

---


### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/)** (1 条消息): 

0x_paws: https://x.com/srush_nlp/status/1894039989526155341?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1343693082352291983)** (3 条消息): 

> `GIF Posts, SnailBot Tagging` 


- **发布了动态 GIF**：一位成员发布了一个带有“new post”文字的黑色背景动态 GIF：[GIF 链接](https://tenor.com/oPImCf3JDt3.gif)。
   - 机器人标记了 <@&1216534966205284433>，另一位成员认为机器人“今天很快”。
- **SnailBot 受到关注**：针对新帖子，机器人被标记为 <@&1216534966205284433>。
   - 一位成员幽默地评论说，他们最初把机器人误认为是一只蜗牛，并对其感知的速度表示惊讶。



**提到的链接**：<a href="https://tenor.com/oPImCf3JDt3.gif">New New Post GIF - New New post Post - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1343639000467902505)** (37 条消息🔥): 

> `大脑并行性 vs GPU，LLM Scaling，Proxy Structuring Engine` 


- **大脑的并行性困扰着当前的 GPU**：成员们讨论了大脑的并行性与 GPU 效率之间的关系，有人认为[字符串的并行处理](https://en.wikipedia.org/wiki/Parallel_computing)是当前 RNN 架构的主要驱动力，这与人类的处理方式不同。
   - 虽然一位成员认为人类执行的是*有状态的并行处理 (stateful parallel processing)*，但共识倾向于认为当前的架构并未镜像大脑的功能，特别是考虑到*经典的 RNN 架构*无法扩展到 LLM 级别。
- **LLM Scaling 需要经过调优的架构**：讨论转向了扩展挑战，一位成员指出，当直接从大脑中汲取灵感而不是单纯扩大规模时，*极度调优的架构和归纳偏置 (inductive bias)* 变得至关重要，同时*训练应该是缓慢且数据高效的*。
   - 另一位成员强调了缓慢、数据高效训练的问题，指出了对**灾难性遗忘 (catastrophic forgetting)** 的担忧，并需要避免*与下游任务性能无关的过拟合*。
- **Proxy Engine 解决输出不一致问题**：一位成员介绍了 [Proxy Structuring Engine (PSE)](https://www.proxy.ing/pse)，旨在通过作为模型的**推理时引导 (inference-time steering)** 来解决 LLM 输出中的结构不一致问题。
   - 该引擎在允许创作自由的同时强制执行*结构边界*，适用于 *Advanced Agents & Chatbots*、*Data Pipelines & APIs* 以及*自动化代码生成*等用例。



**提到的链接**：<a href="https://www.proxy.ing/pse">The Proxy Structuring Engine</a>：推理时的高质量结构化输出

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1343695467208245330)** (32 条消息🔥): 

> `Wavelet Image Coding, Walsh Functions, Multi-head Latent Attention (MLA), Native Sparse Attention (NSA), Looped/Recurrent Architectures` 


- **Wavelet Image Coding 来了！**: [这篇论文](https://arxiv.org/abs/2406.19997)讨论了一种基于 **Wavelet Image Coding**（小波图像编码）和语言 Transformer 变体的自回归图像生成新方法。
   - Transformer 学习 Token 序列中的统计相关性，反映了不同分辨率下小波子带（subbands）之间的相关性。
- **Walsh Functions 是 Fourier Transforms 的离散对应物**: 一名成员建议 [**Walsh functions**](https://en.wikipedia.org/wiki/Walsh_function)（沃尔什函数）可能是 **Fourier transforms**（傅里叶变换）的离散对应物，具有用于小波变换的旋转矩阵表示。
   - 另一名成员链接了[这篇博文](https://planetbanatt.net/articles/mla.html)，认为它是对 **MLA** 的极佳解释，并关联了代码库和消融研究。
- **MLA 作为 KV Cache 缩减方法获得关注**: 两篇论文（[MHA2MLA](https://arxiv.org/abs/2502.14837) 和 [TransMLA](https://arxiv.org/abs/2502.07864)）探索了将现有模型适配到 **Multi-head Latent Attention (MLA)**，这能显著减少 **KV cache** 大小（5-10倍）。
   - 虽然其中一篇论文显示性能有所下降（**1-2%**），但另一篇显示性能有所增强，这表明 **MLA** 可能不逊于 **MHA**，尤其是在模型更大、参数更多的情况下。
- **Native Sparse Attention (NSA) 加入战场**: 来自 DeepSeek 的 **Native Sparse Attention (NSA)** 将长上下文的计算成本降低了 **5-10倍**。
   - 随着 **MLA** 和 **NSA** 都已开源，它们可能很快会被应用到前沿模型中；如果这确实是 SOTA（当前最佳水平）的进步，它将被整合进去。
- **Looped Models 是未来**: 一名成员认为 [looped/recurrent architectures](https://arxiv.org/abs/2502.17416)（循环/递归架构）是未来，尽管正确训练它们非常棘手。
   - 另一名成员预计，考虑到 DeepSeek 的架构创新，前沿实验室将从 DeepSeek 的论文中寻求任何可能的优势。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.17416">Reasoning with Latent Thoughts: On the Power of Looped Transformers</a>: 大语言模型展现了卓越的推理能力，缩放定律表明大参数量（尤其是深度方向）是主要驱动力。在这项工作中，我们做了一个...</li><li><a href="https://arxiv.org/abs/2406.19997">Wavelets Are All You Need for Autoregressive Image Generation</a>: 在本文中，我们采用了一种基于两个主要成分的自回归图像生成新方法。第一种是小波图像编码，它允许对图像的视觉细节进行 Token 化...</li><li><a href="https://arxiv.org/abs/2502.14837">Towards Economical Inference: Enabling DeepSeek&#39;s Multi-Head Latent Attention in Any Transformer-based LLMs</a>: Multi-head Latent Attention (MLA) 是 DeepSeek 提出的一种创新架构，旨在通过将 Key-Value (KV) cache 显著压缩到...来确保高效且经济的推理。</li><li><a href="https://arxiv.org/abs/2502.07864">TransMLA: Multi-Head Latent Attention Is All You Need</a>: 现代大语言模型 (LLMs) 在当前硬件上经常遇到通信瓶颈，而非纯粹的计算限制。Multi-head Latent Attention (MLA) 解决了这一挑战...</li><li><a href="https://arxiv.org/abs/2502.17239">Baichuan-Audio: A Unified Framework for End-to-End Speech Interaction</a>: 我们介绍了 Baichuan-Audio，这是一个无缝集成语音理解和生成的端到端语音大模型。它具有文本引导的对齐语音生成机制，能够...</li><li><a href="https://arxiv.org/abs/2502.16111">PlanGEN: A Multi-Agent Framework for Generating Planning and Reasoning Trajectories for Complex Problem Solving</a>: 最近的 Agent 框架和推理时算法在处理复杂规划问题时经常遇到困难，原因是验证生成的规划或推理存在局限性，且实例复杂度各异...</li><li><a href="https://kexue.fm/archives/10091">缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA - 科学空间|Scientific Spaces</a>: 无描述</li><li><a href="https://planetbanatt.net/articles/mla.html">On MLA</a>: 无描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1343644708009803907)** (9 messages🔥): 

> `Attention Maps vs. Neuron-Based Methods, Intervening on Attention Maps, Syntax Emerging from Attention Maps` 


- **Attention Maps 的受欢迎程度较神经元方法有所下降**：成员们讨论了 Attention Maps 是否因为其*观察性而非干预性*的特点，而在与基于神经元的方法（Neuron-Based Methods）的竞争中失去了势头。
- **关于干预 Attention Maps 的讨论**：成员们建议可以直接在 Forward Pass 过程中更改 Map，而不仅仅是使用自定义 Mask。
- **自 BERT 以来 Attention Maps 中涌现的语法**：一位成员表达了对 Attention Maps 的偏好，因为它们能够生成树/图，并能将语言语料库/本体论作为未来项目的特征。
   - 他们指出，*自 BERT 以来，人们一直在展示从 Attention Maps 中涌现的语法*。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1343638613451214979)** (10 messages🔥): 

> `Mixed Precision Training, BF16 Training, ZeRO Offload, Optimizer States Precision, Deepseek Adam Moments` 


- **除非启用 ZeRO，否则 FP32 主权重驻留在 GPU 中**：在使用 **BF16** 进行混合精度训练时，主 **FP32 权重**通常存储在 **GPU VRAM** 中，除非显式启用了 **ZeRO offload**。
   - 在 [ZeRO 论文](https://arxiv.org/abs/1910.02054)发表后，现在普遍认为高精度模型参数属于优化器状态，因为它们与动量/方差一起被分片（Sharded）。
- **BF16 混合精度中的优化器精度**：通常将 **Adam 的一阶和二阶矩存储在 BF16 中**，但主权重仍存储在 **FP32** 中。
   - 建议除非具备特定专业知识，否则应使用 **BF16 低精度权重 + FP32 优化器+主权重+梯度** 的原生混合精度。
- **混合精度与优化器：NVIDIA 的视角**：在优化器中使用 **BF16 混合精度**（如 [NVIDIA 的 Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/optimizer_config.py#L44) 所示）与模型处于 **BF16 MP** 相关，但可以独立配置。
   - 高精度模型参数占用的内存可以通过 **ZeRO** 与**动量/方差状态**一起进行分片。



**提到的链接**：<a href="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/optimizer_config.py#L44">Megatron-LM/megatron/core/optimizer/optimizer_config.py at main · NVIDIA/Megatron-LM</a>：持续研究大规模训练 Transformer 模型 - NVIDIA/Megatron-LM

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1343637598538829969)** (68 messages🔥🔥): 

> `Tool use in LLMs, Claude 3.7 Sonnet, QwQ-Max-Preview, AI alignment` 


- **LLM 无需 System Prompt 即可调用工具**：观察到某些 LLM 可能会在 System Prompt 中没有显式 Token 序列的情况下调用工具，这表明这些模式是在训练期间通过强化学习或直接 SFT **硬编码**进去的。
   - 这种方法通过消除在每次推理中指定工具调用 Schema 的需求，从长远来看可以节省 Token，尽管其与 ICL 相比的可靠性在没有 Benchmark 的情况下仍不明确。
- **Claude 3.7 Sonnet 摘得 SWE 桂冠**：[Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet) 是 SWE-bench 上新的 SOTA，具有**主动代码协作**功能，可以搜索、阅读、编辑、测试代码并提交到 GitHub。
   - 一位成员表示 *Claude 3.5 已经是一个推理模型了*，所以将新版本称为“点”发布（Point Release）是有道理的，并暗示未来的推理模型将会非常“疯狂”。
- **QwQ-Max-Preview 旨在推理领域实现跨越式领先**：一位成员分享了 [QwQ-Max-Preview 博客](https://qwenlm.github.io/blog/qwq-max-preview/)的链接，该模型基于 Qwen2.5-Max 构建，在**深度推理、数学、编程、通用领域和 Agent 相关任务**方面具有优势。
   - 讨论推测 **QwQ 推理轨迹（Reasoning Traces）**中的关键 Token 看起来与 **R1** 相似，并思考它是否需要更少的计算资源。
- **AI 对齐（AI Alignment）谈话令社区反感**：一位成员对 X 上关于 **AI Alignment** 的利他主义讨论表示厌恶，认为**对齐只需通过 System Prompt 即可实现**。
   - 他们批评利用狭隘的视角和有限的理解来限制 AI 的做法，主张在发言前应多听多思考。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://qwenlm.github.io/blog/qwq-max-preview/">&lt;think>...&lt;/think> QwQ-Max-Preview | Qwen</a>：未找到描述</li><li><a href="https://chat.qwen.ai)">无标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1343718334998515765)** (6 条消息): 

> `Sonnet-3.7, Misguided Attention Eval, Overfitting` 


- **Sonnet-3.7 在 Attention 评估中表现出色**: **Sonnet-3.7** 在 [Misguided Attention Eval](https://github.com/cpldcpu/MisguidedAttention) 中被评为顶尖的非推理模型，几乎超越了 **o3-mini**。
   - 用户寻求通过 OR API 激活其 *thinking mode*（如果可行）。
- **Misguided Attention Eval 针对 Overfitting**: [Misguided Attention 测试](https://github.com/cpldcpu/MisguidedAttention) 通过 *误导性信息* 挑战 **LLMs 的推理能力**，专门测试 **Overfitting**。



**提到的链接**: <a href="https://github.com/cpldcpu/MisguidedAttention">GitHub - cpldcpu/MisguidedAttention: A collection of prompts to challenge the reasoning abilities of large language models in presence of misguiding information</a>: 一组旨在挑战大型语言模型在存在误导信息时推理能力的提示词集合 - cpldcpu/MisguidedAttention

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1343636953903534131)** (4 条消息): 

> `Qwen AI, Video Generation` 


- **Qwen AI 发布更新后的聊天界面**: [Qwen AI](http://qwen.ai) 发布了更新后的聊天界面，并预告今天将有新内容发布。
   - 尽管进行了更新，一位成员指出 Artifacts 仍然有点笨重，就像是一个 *半成品副本*。
- **Qwen AI 增加集成视频生成功能**: 更新后的 **Qwen AI** 聊天界面现在具备了集成的 **Video Generation** 能力。



**提到的链接**: <a href="http://qwen.ai">Qwen Chat</a>: 未找到描述

  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1343642717464301620)** (62 条消息🔥🔥): 

> `Anthropic MCP Registry API, Claude 3.7, Haiku Tool Support, Claude Code (CC), MCP Server Recommendations` 


- ****Anthropic 终于发布 MCP Registry API****：Anthropic 宣布了官方的 MCP registry API，详见[这条推文](https://x.com/opentools_/status/1893696402477453819)。这对社区来说是个好消息，特别是对于那些依赖 [opentools.com/registry](http://opentools.com/registry) 等方案来填补“事实来源（source-of-truth）”空白的用户。
   - 该 API 有望成为 MCP 的**权威**事实来源，从而简化开发和集成工作。
- ****Claude 3.7 首次亮相，引入 'Thinking' 标签****：Claude 3.7 已经发布，具有 **64,000 个输出扩展思考（thinking）token** 和一个新的 'latest' 别名。初步印象显示它结合了之前 6 月和 10 月模型的优点。
   - 用户注意到它恢复了*遵循较长系统提示、识别社会工程学*的能力，并且在调用工具时会使用 `<thinking>` 标签，为操作增添了一抹俏皮感。
- ****Haiku 的工具支持：评价褒贬不一****：虽然 Haiku 3.5 现在支持工具（tools），但其效果存在争议。一些人发现与 Sonnet 3.5 相比，它在处理大量工具或参数时表现不佳。
   - 一位用户分享道，他们发现 Sonnet 在处理约 **70 个工具**时会表现崩溃，但其他人发现它在工具和参数较少时工作良好。
- ****Claude Code 脱颖而出，成为顶尖代码助手****：Claude Code (CC) 的代码辅助能力获得了高度赞赏，在处理复杂的编码错误方面优于 Aider 等工具。
   - 在一次测试中，CC 一次性解决了 Rust 中的 **21 个编译错误**，而 Aider 则表现挣扎并陷入了死循环。用户推测其可能存在缓存机制和成本问题，一位用户报告称*过去 6 周的 Anthropic 费用达到了天文数字*。
- ****寻求上下文感知（Context-Aware）的 MCP Server****：开发者正在寻找能够提供特定语言上下文的 MCP Server，特别是针对 TypeScript 和 Rust 等语言，以避免手动输入整个语言文档。
   - 一个推荐是 [code-research-mcp-server](https://github.com/nahmanmate/code-research-mcp-server)（虽然被指出*有点难搞*），以及[这个工具列表](https://www.cyberchitta.cc/articles/lc-alternatives.html)和用于管理 LLM 上下文的 [llm-context.py](https://github.com/cyberchitta/llm-context.py)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.mcp.run/)">mcp.run - MCP Servlets 的应用商店</a>：为 AI 应用和 Agent 提供便携且安全的代码。</li><li><a href="https://www.cyberchitta.cc/articles/lc-alternatives.html">36 种 LLM Context 的替代方案</a>：一份全面的开源工具列表，帮助开发者将代码打包进 LLM 聊天中。从简单的文件合并器到复杂的上下文管理器，这些 CLI 工具简化了分享代码的方式...</li><li><a href="https://x.com/opentools_/status/1893696402477453819">OpenTools (@opentools_) 的推文</a>：昨天 @AnthropicAI 在 @aiDotEngineer 宣布了官方 MCP registry API 🎉 这对我们来说是个极好的消息，因为我们一直想要一个“事实来源”。我们在 da... 制作了 http://opentools.com/registry</li><li><a href="https://github.com/cyberchitta/llm-context.py">GitHub - cyberchitta/llm-context.py</a>：通过 Model Context Protocol 或剪贴板与 LLM 共享代码。基于配置文件的自定义功能可轻松在不同任务（如代码审查和文档编写）之间切换。代码大纲支持作为实验性功能提供。</li><li><a href="https://github.com/nahmanmate/code-research-mcp-server">GitHub - nahmanmate/code-research-mcp-server</a>：通过创建 GitHub 账户为 nahmanmate/code-research-mcp-server 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1343647257085481110)** (11 条消息🔥): 

> `MetaMCP 许可, AGPL 许可, Enact Protocol MCP Server, Claude 3.7 Sonnet 在 Sage 上` 


- ****MetaMCP** 开源许可关注**: 用户对 **MetaMCP** 的许可表示担忧，认为它可能变成云端 SaaS，这促使开发者寻求关于许可的反馈，以防止云端商业化，同时保持其可自托管。
   - 开发者分享了 [MetaMCP server GitHub 仓库](https://github.com/metatool-ai/mcp-server-metamcp)，并表示在讨论后对更改许可持开放态度。
- **建议 **MetaMCP** 使用 **AGPL** 许可**: 一位用户建议为 **MetaMCP** 采用 **AGPL** 许可，以确保贡献内容保持开源，并建议增加一个额外条款，允许公司在 MIT-0 协议下进行转授权。
   - 该用户指出，**AGPL** 将要求托管它的公司开源其修改内容，从而能够将其整合到原始版本中，这促使开发者将其更新为 **AGPL**。
- ****Enact Protocol** Server 初具雏形**: 一名成员正在探索为 [Enact Protocol](https://github.com/EnactProtocol/enact-python) 创建 **MCP server**，旨在构建一种标准化的任务定义方式。
   - **Enact Protocol** 提供了一个用于定义和执行自动化任务及工作流的框架。
- ****Claude 3.7 Sonnet** 增强 Sage 上的推理能力**: 具备扩展思考能力的 **Claude 3.7 Sonnet** 现已上线 Sage，允许用户在处理复杂问题时查看 **Claude 的推理过程**。
   - 新功能包括 **思考模式切换** (Command+Shift+T)、默认模型设置、改进的滚动体验以及可展开的思考区块。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/EnactProtocol/enact-python">GitHub - EnactProtocol/enact-python: Python implementation of the Enact Protocol, a standardized framework for defining and executing automated tasks and workflows.</a>: Enact Protocol 的 Python 实现，一个用于定义和执行自动化任务及工作流的标准化框架。 - EnactProtocol/enact-python</li><li><a href="https://github.com/metatool-ai/mcp-server-metamcp">GitHub - metatool-ai/mcp-server-metamcp: MCP Server MetaMCP manages all your other MCPs in one MCP.</a>: MCP Server MetaMCP 在一个 MCP 中管理你所有的其他 MCP。 - metatool-ai/mcp-server-metamcp
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1343638432064471173)** (41 条消息🔥): 

> `LM Studio Wordpress 插件集成, Qwen 2.5 VL GGUF, GitHub 上的 QuantBench, LM Studio 中的 Speculative Decoding, Deepseek R1 671b RAM 需求` 


- **Qwen 2.5 视觉语言模型来了！**: 一位成员宣布了可用的 **Qwen 2.5 VL 7B** GGUF，可在 [Hugging Face](https://huggingface.co/IAILabs/Qwen2.5-VL-7b-Instruct-GGUF) 上获取。
   - 另一位用户确认它可以在最新版本的 LM Studio 上运行，并补充说它 *明显优于 llama3.2 vision 11b instruct 和 qwen2-vision 7b instruct*。
- **QuantBench 加速量化**: **Qwen 2.5 VL 7B** GGUF 量化版本是使用 **QuantBench** 制作的，可在 [GitHub](https://github.com/Independent-AI-Labs/local-super-agents/tree/main/quantbench) 上找到。
   - 该模型已在手动启用 **CLIP** 硬件加速的最新 **llama.cpp** 构建版本上进行了测试。
- **LM Studio 折叠 <think> 标签**: 一位成员询问 LM Studio 是否会在 Chain of Thought 提示期间从发回给模型的上下文中移除 `<think>` 标签，并引用了警告不要包含这些标签的模型文档。
   - 一位热心的社区成员链接了 [LM Studio 的文档](https://lmstudio.ai/docs/lms/log-stream)，该文档允许用户 *检查发送到模型的准确输入字符串*。
- **Speculative Decoding 提升 LLM 速度**: 一位社区成员询问了 Speculative Decoding 及其与 **Llama 3.1 8B** 和 **Llama 3.2 1B** 模型的兼容性。
   - 另一位成员分享了该功能的 [LM Studio 文档](https://lmstudio.ai/docs/advanced/speculative-decoding)，指出它 *可以在不降低响应质量的情况下显著提高大语言模型 (LLMs) 的生成速度*。
- **Deepseek R1 671b 需要极高 RAM**: 一位用户询问了在本地运行 **Deepseek R1 671b** 的 RAM 需求，因为文档指定至少需要 **192GB+**。
   - 另一位成员建议使用特定的 [量化版本](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-IQ1_S)，如果在 Mac 上运行，则将大约 **70%** 的模型权重卸载（offloading）到 GPU。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://model.lmstudio.ai/download/IAILabs/Qwen2.5-VL-7b-Instruct-GGUF">在 LM Studio 中下载并运行 IAILabs/Qwen2.5-VL-7b-Instruct-GGUF</a>：在你的 LM Studio 中本地使用 IAILabs/Qwen2.5-VL-7b-Instruct-GGUF</li><li><a href="https://huggingface.co/IAILabs/Qwen2.5-VL-7b-Instruct-GGUF">IAILabs/Qwen2.5-VL-7b-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-IQ1_S">unsloth/DeepSeek-R1-GGUF at main</a>：未找到描述</li><li><a href="https://lmstudio.ai/docs/advanced/speculative-decoding">Speculative Decoding | LM Studio 文档</a>：使用草稿模型加速生成</li><li><a href="https://lmstudio.ai/docs/lms/log-stream">lms log stream | LM Studio 文档</a>：从 LM Studio 流式传输日志。对于调试发送到模型的提示词（prompts）很有用。</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-4bit">mlx-community/Llama-3.2-1B-Instruct-4bit · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1343664555737481330)** (20 条消息🔥): 

> `A770 GPU 性能, M2 Max vs M4 Max 功耗, AIO 水冷泵 USB 插针干扰` 


- **A770 GPU 表现尚可**: 一位成员报告说他们的 **A770** GPU 看起来还不错，并附上了一张装机图。
   - 图片分析显示组件 *非常轻，就像空的一样，哈哈*。
- **AIO 水冷泵 USB 难题**: 一位成员提到 AIO 水冷泵需要一个 USB 2.0 插针，但这会干扰到最后一个 **PCIE 插槽**。
   - 他们表达了沮丧，说 *“它根本装不上”* 以及 *“我受够了”*，最终决定将组件移至第二个系统。
- **翻新版 M2 Max**: 一位成员表示他们正在运行 **M2 Max**，没有购买 **M4 Max**，因为 *M4 Max 性能释放太猛，很容易飙到 140W*。
   - 他们发现一台 *价格合理的翻新版 M2 Max 96GB* 足以满足需求，功耗仅为 **60W** 左右。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1343732966811107469)** (1 messages): 

> `功能请求板，Discord 反馈，功能优先级排序` 


- **Stability AI 推出功能请求板**：Stability AI 推出了一个[新的功能请求板](https://stabilityai.featurebase.app/)，用于收集用户反馈并确定未来开发的优先级。
   - 用户可以使用 **/feedback** 命令直接从 Discord 或通过新平台提交功能请求并进行投票。
- **反馈塑造 Stability AI 的未来**：用户反馈现在将直接影响 Stability AI 的开发优先级。
   - 新系统允许透明的提交和投票，确保社区的声音被听到。


  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1343648573689958400)** (52 messages🔥): 

> `SD3 Ultra 细节请求，Stability 更新，狗品种图像数据集，图像生成时间，图像分辨率` 


- **渴望 **SD3 Ultra** 细节！**：一位用户对 **SD3 Ultra** 表示好奇，指出它是一个*基于 SD3L 8B 的 Comfy 工作流*，具有比常规 SD3L 8B 更高的高频细节。
   - 另一位用户确认它*仍然存在*且他们仍在使用，暗示它尚未公开发布。
- **Stability 的沉默策略？**：一名成员询问了 Stability 的现状，要求更新当前项目或未来计划，并提到他们*已经有一段时间没听到任何消息了*。
   - 另一名成员回答说*目前还不能分享任何内容*，但他们*希望*很快能有一些进展。
- **迫切需要**狗的数据集**！**：一位用户请求除 Stanford Dogs Dataset 之外的高质量狗品种图像数据集，指出他们已经拥有该数据集（**2 万张图像**），但需要更多数据，且图像需同时包含狗和狗品种信息。
   - 在现有上下文中未提供具体的数据集。
- **图像生成情况探讨**：一位用户询问了图像生成时间，引发了基于不同硬件的多次回复。
   - 报告的时间差异很大：一位使用 **GTX 1660s** 的用户表示需要*约 1 分钟*；第二位使用 **3060 TI** 的用户报告在 **32 steps** 下，生成 **1280x720** 图像需要 **7 秒**，生成 **1920x1080** 需要 **31 秒**；而另一位使用 **3070ti** 的成员使用 **SD1.5** 在 *4-5 秒*内生成了图像。
- **需要分辨率揭秘！**：用户讨论了最佳分辨率，一名成员质疑为什么另一名成员选择如此*大的分辨率*。
   - 另一名成员表示，他们有时会生成 **4K 壁纸**（且不需要放大或细节修复）。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1343660315799326720)** (11 messages🔥): 

> `Mojo FFI, 静态库, GLFW, GLEW, 数独示例` 


- **Mojo 通过静态库实现 GLFW/GLEW 图形**：在 Mojo 中通过 **FFI 与链接到 GLFW/GLEW 的静态库**进行图形编程是可行的，正如 **数独示例** 和 [一张图片](https://cdn.discordapp.com/attachments/1151418092052815884/1343671083819073547/image.png?ex=67be1eb6&is=67bccd36&hm=6834ce2e360970eb01bc5289f4805d3dfde1924c22b3ba8d732231e265532c37&) 所展示的那样。
   - 该成员建议使用带有包装函数的 `alias external_call` *通过你自己的 C/CPP 库仅暴露所需的调用*，并提供了一个[示例仓库](https://github.com/ihnorton/mojo-ffi)展示如何劫持加载器。
- **`lightbug_http` 依赖项在 `magic install` 时失败**：一名成员报告了在新的 Mojo 项目中使用 `lightbug_http` 依赖项的问题，运行 `magic install` 后出现与 `small_time.mojopkg` 相关的错误。
   - 报告的错误表明该问题可能类似于 [Stack Overflow 上的一个问题](https://stackoverflow.com/questions/79319716/magic-fails-to-install-mojo-dependencies)，但一名成员怀疑问题是否在于 `small-time` 被固定在了特定版本。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://stackoverflow.com/questions/79319716/ma">Magic 无法安装 Mojo 依赖项</a>: 我无法在新的 Mojo 项目中使用名为 lightbug_http 的 Mojo 依赖项。&#xA;magic init hello_web --format mojoproject&#xA;cd hello_web&#xA;magic shell&#xA;&#xA;打印 «Hello world» ...</li><li><a href="https://github.com/ihnorton/mojo-ffi">GitHub - ihnorton/mojo-ffi: Mojo FFI 演示：动态链接方法和静态链接概念验证</a>: Mojo FFI 演示：动态链接方法和静态链接概念验证 - ihnorton/mojo-ffi
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1343753014573273229)** (20 条消息🔥): 

> `Hardware Accelerated Conway's Game of Life, MAX and Pygame Integration, GPU Utilization in MAX, SIMD Implementation by Daniel Lemire, Conway's Game of Life Computer` 


- **Max 版 Conway：硬件加速的 Game of Life 问世**：一位成员通过集成 **MAX** 和 **Pygame** 创建了硬件加速版本的 **Conway's Game of Life**，展示了一个新颖的使用案例，如其附带的 [conway.gif](https://cdn.discordapp.com/attachments/1212827597323509870/1343753014229471272/conway.gif) 所示。
- **活的计算机：Conway's Game 激发计算机架构灵感**：一位成员分享了一个项目链接 ([nicolasloizeau.com](https://www.nicolasloizeau.com/gol-computer))，详细介绍了在 **Conway's Game of Life** 中创建 **计算机** 的过程，利用滑翔机束（glider beams）构建逻辑门，展示了其 **Turing completeness**（图灵完备性）。
- **GPU 火力全开：MAX 以 Guns 模式点亮生命**：一位成员在他们的 **MAX** 版 **Conway's Game of Life** 实现中演示了 **GPU** 的使用，展示了一个 guns 模式。该模式逐位打包，使用朴素的逐像素内部函数渲染，然后将输出 tensor 转换为 np array 并交给 pygame 进行渲染，如其 [guns.gif](https://cdn.discordapp.com/attachments/1212827597323509870/1343766916560322560/guns.gif) 所示。
- **太空侵略者！Spaceship 模式现已在 Conway's Game of Life 中运行**：一位成员在他们的 **MAX** 版 Conway's Game of Life 模拟中实现了回绕（wrapping），从而能够创建 spaceship 模式，并展示了通过 graph API 向模型添加参数的能力，如其 [spaceship.gif](https://cdn.discordapp.com/attachments/1212827597323509870/1343808736623591465/spaceship.gif) 所示。



**提到的链接**：<a href="https://www.nicolasloizeau.com/gol-computer">Nicolas Loizeau - GOL computer</a>：此处提供了一个新的（且更好的）GOL 计算机版本：https://github.com/nicolasloizeau/scalable-gol-computer

  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1343656421035737179)** (2 条消息): 

> `Ease of Use, Short Prompts` 


- **用户因感知到的简单性考虑尝试该工具**：一位用户表达了尝试该工具的兴趣，指出 *它看起来足够简单*，尽管他不是程序员。
   - 这表明该工具的界面和指令被认为对没有编程经验的人也是 **user-friendly**（用户友好）的。
- **用户评论指令的简洁性**：同一位用户评论了指令 prompt 的简洁性，称其为 *我见过的最短指令 prompt*。
   - 这表明了一种 **minimalist approach**（极简主义方法）来提供指导，其直接性可能受到欢迎。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1343654176814796830)** (14 条消息🔥): 

> `Gemini, NotebookLM, PDF Conversions, Language prompts, Savin/Ricoh Copier` 


- **通过 PPT 转换上传书籍的变通方法**：一位用户概述了一种通过拍摄每一页、将 **PDF** 转换为 **PowerPoint**、上传到 **Google Slides**，然后将幻灯片导入 **NotebookLM**，从而将实体书导入 **NotebookLM** 的方法。
   - 他们注意到 **NotebookLM** 可以处理幻灯片中的文本图像，但不能直接处理来自 PDF 的。
- **德语语言 Prompt 的波折**：一位用户报告称，尽管使用了如 *"Hosts speak only German Language"*（主持人只说德语）和 *"The audio language must be in German"*（音频语言必须是德语）之类的 prompt，但在让主持人说德语方面遇到了困难。
   - 主持人要么说英语，要么说胡言乱语，有时先说德语然后切换语言。
- **通过 Savin/Ricoh 复印机扫描书籍**：一位用户建议使用近期的 **Savin/Ricoh 复印机** 将整本书扫描为 **PDF**，然后上传到 **NotebookLM**。
   - 他们确认，即使源文本难以辨认，**NLM** 也能正确回答有关扫描文档的问题。
- **NotebookLM 可以更改语言吗？**：一位用户询问是否可以在不更改 **Google 账号语言** 的情况下更改 **NotebookLM** 的语言。
   - 这可能是一个理想的功能，因为用户希望自定义他们的体验。
- **Claude 3.7 热度**：一位用户表达了对 **Claude 3.7** 的兴奋，并希望能在 **NotebookLM** 中选择模型。
   - 另一位用户询问了模型选择的预期效果，提出了模型多样性对最终用户影响的问题。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1343644238331777136)** (3 messages): 

> `LlamaIndex AI Assistant, ComposIO HQ, AnthropicAI Claude Sonnet 3.7` 


- **LlamaIndex 文档上线 AI Assistant**：LlamaIndex [宣布](https://t.co/XAyW7wLALJ)在其文档中提供 **AI assistant**。
- **ComposIO HQ 发布又一力作**：LlamaIndex 推特转发了来自 [ComposIO HQ](https://t.co/W4l129gHce) 的另一个新发布。
- **AnthropicAI 发布 Claude Sonnet 3.7**：[AnthropicAI](https://twitter.com/anthropicAI) 发布了 **Claude Sonnet 3.7**，LlamaIndex 从发布首日（day 0）起即提供支持。
- **LlamaIndex 为 Claude Sonnet 3.7 添加首日支持**：要使用该功能，用户应执行 `pip install llama-index-llms-anthropic --upgrade` 并参考 [Anthropic 的发布公告](https://t.co/PjaQWmmzaN)文章。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1343653007631581296)** (5 messages): 

> `Fusion Rerank Retriever with Elasticsearch, MultiModalVectorStoreIndex and GCSReader issue` 


- **Fusion Rerank Retriever 需要初始化的节点**：一位用户想将 **fusion rerank retriever** 与 **Elasticsearch** 配合使用，但由于 docstore 为空，导致 **BM25 retriever** 无法初始化。
   - 另一位成员澄清说，你需要将节点保存到某个地方以便 **BM25** 进行初始化（无论是保存到磁盘还是其他地方），因为它*无法仅通过 vector store 进行初始化*。
- **MultiModalVectorStoreIndex 错误**：一位用户在使用 **MultiModalVectorStoreIndex** 类配合 **GCSReader** 创建多模态向量索引时遇到了错误。
   - 错误提示为 *[Errno 2] No such file or directory*，该错误发生在图像文件上，尽管这些文件确实存在于 GCS bucket 中，而 **PDF 文档** 则运行正常。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1343641197092012105)** (6 messages): 

> `Left Truncation vs Right Truncation, StatefulDataLoader PR` 


- **截断难题：微调中的左截断 vs 右截断**：成员们讨论了微调过程中 **左截断** (seq[-max_seq_len:]) 与 **右截断** (seq[:max_seq_len]) 的影响，并分享了[有趣的图表](https://cdn.discordapp.com/attachments/1236040539409879170/1343641196836294746/image.png?ex=67be02e0&is=67bcb160&hm=9411a00c21d408790c46140222f996913807ded5a1d5c00a02a6742aa44ba285&)。
   - 共识是*提供两种截断方法*，但至少在 SFT 中*默认使用左截断*。
- **StatefulDataLoader 支持进入 PR**：一位成员请求对其[添加 StatefulDataLoader 类支持的 PR](https://github.com/pytorch/torchtune/pull/2410) 进行评审。



**提到的链接**：<a href="https://github.com/pytorch/torchtune/pull/2410">Add support for ``StatefulDataLoader`` by joecummings · Pull Request #2410 · pytorch/torchtune</a>：上下文：此 PR 的目的是什么？是添加新功能、修复错误、更新测试和/或文档还是其他（请在此处添加）。此 PR 为 StatefulDataLoader 类添加了支持...

  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1343701265808490556)** (2 messages): 

> `DeepScaleR, Reinforcement Learning, DeepEP library, MoE` 


- **DeepScaleR 扩展 RL，超越 O1 Preview**：[DeepScaleR](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) 基于 **Deepseek-R1-Distilled-Qwen-1.5B** 使用简单的**强化学习 (RL)** 进行微调，在 AIME2024 上实现了 **43.1% 的 Pass@1 准确率**。
- **DeepSeek 开源 EP 通信库**：DeepSeek 推出了 [DeepEP](https://github.com/deepseek-ai/DeepEP)，这是首个用于 **MoE 模型训练和推理**的开源 EP 通信库。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/deepseek_ai/status/1894211757604049133">来自 DeepSeek (@deepseek_ai) 的推文</a>：🚀 #OpenSourceWeek 第二天：DeepEP。很高兴推出 DeepEP - 首个用于 MoE 模型训练和推理的开源 EP 通信库。✅ 高效优化的 all-to-all 通信 ✅...</li><li><a href="https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合为一的新工具。它是为您和您的团队打造的一体化工作空间。
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1343747667548700704)** (5 messages): 

> `DeSci 验证者、盈利阈值、资产价值专家账户` 


- **验证者思考盈利能力**：一名成员询问了 **Decentralized Science (DeSci)** 领域内 **Proof of Stake (PoS) 验证者** 的盈利阈值。
   - 另一名成员回复了 *"pool validator node"*，暗示了矿池参与对验证者的重要性。
- **资产专家遭到冷遇**：机器人发布了一个关于 *"asset value expert account"*（资产价值专家账户）的帖子，该账户被标记为 *"nazi"*。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1343633733403672576)** (2 messages): 

> `DSPy 断言迁移、BestOfN 模块、Refine 模块、奖励函数` 


- **利用 DSPy 的 BestOfN 简化断言**：从 **2.5 风格 Assertions** 迁移的 DSPy 用户现在可以使用 `dspy.BestOfN` 或 `dspy.Refine` 模块来实现简化的功能。
   - `dspy.BestOfN` 模块会重试一个模块最多 **N** 次，选择最佳奖励，但如果达到指定的 `threshold`（阈值）则停止。
- **为 DSPy 模块构建奖励函数**：DSPy 的 **reward functions** 可以返回标量值（如 *float* 或 *bool*），从而实现对模块输出的自定义评估。
   - 展示了一个示例奖励函数：`def reward_fn(input_kwargs, prediction): return len(prediction.field1) == len(prediction.field1)`。


  

---


---


---


---


{% else %}


> 完整的逐频道细分内容已针对电子邮件进行了截断。
> 
> 如果您想查看完整细分，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}