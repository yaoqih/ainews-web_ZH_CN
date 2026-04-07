---
companies:
- nous-research
- anthropic
date: '2026-04-06T05:44:39.731046Z'
description: '**Hermes Agent** 作为领先的开放代理技术栈（open agent stack）正受到广泛关注，其核心特性包括自我提升技能、持久化记忆以及自我改进闭环。其新增的
  **Manim 技能** 能够生成数学及技术动画，进一步扩展了代理的能力。随着 GUI 工具、WebUI、HUD（平视显示器）更新、OAuth 支持以及各项集成的推出，Hermes
  生态系统正在迅速扩张。


  与此同时，一场针对智能体的开放训练数据运动正在兴起，重点在于共享可复用的行为数据和任务轨迹（harness traces）。


  相比之下，**Anthropic 的 Claude Code** 则面临着分发和政策方面的挑战。有报告指出，其限制和不稳定性正影响着第三方编程代理，这凸显了常驻（always-on）代理在订阅经济模式下存在的问题。社区中的核心观点包括：“如果使用
  Claude Code 来分析其自身源码，现在会报错”以及“基本处于不可用状态”。'
id: MjAyNS0x
models: []
people:
- theo
- clementdelangue
- badlogicgames
- yuchenj_uw
title: 今天没发生什么特别的事。
topics:
- self-improving-skills
- agent-architecture
- memory-persistence
- animation-generation
- open-training-data
- coding-agents
- subscription-models
- policy-restrictions
---

**平静的一天。**

> 2026年4月4日至4月6日的 AI 新闻。我们检查了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，且没有进一步的 Discord 消息。[AINews 网站](https://news.smol.ai/) 允许您搜索所有历史期。提醒一下，[AINews 现已成为 Latent Space 的一部分](https://www.latent.space/p/2026)。您可以[选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) 邮件发送频率！

---

# AI Twitter 回顾

**Hermes Agent 的爆发周：自我提升技能、Manim 视频生成以及开放的 Trace 数据飞轮**

- **Hermes 正成为这批讨论最多的开源 Agent 技术栈**。在用户报告中反复提到的核心差异点在于架构而非品牌：自我编写/自我改进的技能、持久的跨会话 memory、可搜索的历史记录以及明确的自我改进循环。最清晰的对比总结来自 [@TheTuringPost](https://x.com/TheTuringPost/status/2040936147720048909)，他从技能、memory、架构和安全默认设置方面将 Hermes 与 OpenClaw 进行了对比。许多开发者也呼应了这一观点，报告称使用 Hermes 的入门门槛更低且效果更好，包括 [@SnuuzyP](https://x.com/SnuuzyP/status/2040999794894663996)、[@DoctaDG](https://x.com/DoctaDG/status/2041051272560923090) 和 [@purea1go](https://x.com/purea1go/status/2041260177681441195)。
- **最重大的具体功能发布是 Hermes 的 Manim 技能**，它可以将 prompt 转换为 3Blue1Brown 风格的数学/技术动画。Nous 在[这条发布推文](https://x.com/NousResearch/status/2040931043658567916)中介绍了它，社区立即将其视为 Agent 领域扩展能力的体现，而非噱头：[@ErickSky](https://x.com/ErickSky/status/2040956335764734235)、[@lucatac0](https://x.com/lucatac0/status/2041018088913608923)、[@casper_hansen_](https://x.com/casper_hansen_/status/2041046264758858081)，甚至 [@Sentdex](https://x.com/Sentdex/status/2041165530812334417) 都强调了解说视频生成是比 summarization 更具吸引力的 Agent 使用案例。Teknium 还澄清了通过 [`/manim-video <prompt>`](https://x.com/Teknium/status/2040970080172060730) 进行调用的方式。
- **Hermes 周边的生态系统正在迅速完善**：GUI 层如 [hermes-workspace.com 突破 700 stars](https://x.com/outsource_/status/2040901725066858898)，原生的 [Hermes WebUI](https://x.com/nesquena/status/2041000592215298123)，以及 Hermes HUD 更新，支持在 [v0.4.0](https://x.com/aijoey/status/2040978270439580042) 中将实时 Agent 映射到 tmux 窗格并显示等待审批的任务；此外，[v0.5.0](https://x.com/aijoey/status/2041282964097511679) 中新增了 HUD “Prompt Patterns” 视图，用于从本地状态数据库中挖掘重复的用户行为。Hermes 还增加了 [OAuth 的 MCP server 支持](https://x.com/Teknium/status/2041020715022024976)，支持通过 [Telegram/Discord 斜杠命令强制加载技能](https://x.com/Teknium/status/2041233409901769133)，以及通过 [Teknium 的指南](https://x.com/Teknium/status/2040998328461316524)，利用 Hermes 自行创建的 endpoint 与 OpenWebUI 集成。
- **第二个趋势是 Agent 领域的开放训练数据运动**。[@badlogicgames](https://x.com/badlogicgames/status/2040979640265633882) 发布了 `pi-share-hf`，用于发布经过 PII 过滤的 coding-agent trace；随后他在[后续推文](https://x.com/badlogicgames/status/2041151967695634619)中发布了自己的数据集。[@ClementDelangue](https://x.com/ClementDelangue/status/2041189872556269697) 明确呼吁建立一个众包的 Agent trace 开放数据集，而 Teknium 指出 Hermes 的会话文件已经是[具备训练就绪条件的 JSONL 候选数据](https://x.com/Teknium/status/2041253083687317660)。这一点值得注意，因为它将“开源 Agent”的范畴从单纯的模型 weights 转向了可重用的行为数据和测试框架 trace。

**Anthropic 对 Claude Code 的限制以及由此引发的向开源/替代 Coding Agent 的转型**

- **这里的核心 Coding-Agent 动态并非模型发布，而是一场分发与政策冲突**。多条高互动帖子报告称 Claude Code 在第三方 Agent 工作流中变得受限或不可靠。[@theo](https://x.com/theo/status/2041016477047034012) 报告称，如果使用 Claude Code 分析其自身源码，现在会报错；此前他在 [这篇帖子](https://x.com/theo/status/2040895674288570499) 中称最新的文档更新“非常疯狂”，随后又在 [另一篇帖子](https://x.com/theo/status/2041111862113444221) 中宣布 Claude Code “基本无法使用”。类似的抱怨也出现在 [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2041187141523526011) 的发文中，他将运行时间不稳定和订阅封锁归咎于 GPU 短缺，还有其他人讨论了 Prompt 级别的准入限制（Gating）。
- **社区普遍认为，聊天机器人时代固定的订阅经济模式并不适用于全天候运行的 Agent**。[@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2041202983640432966) 提出了最清晰的经济学论点：20美元/200美元的订阅是为聊天设计的，而不是为了 24/7 全天候运行、会消耗远超以往 Token 的 Agent 循环。这意味着短期内将面临一系列权衡：更严格的 Rate Limits、更多的 GPU 采购、使用强制执行或重新定价。[@kimmonismus](https://x.com/kimmonismus/status/2041203798723666375) 总结的更广泛报告也支持了这一观点，即推理成本仍然占据了实验室收入的大部分。
- **这种摩擦正在为开源 Harness、中间件和替代方案创造空间**。LangChain 发布了一系列关于 [Harness 自定义中间件模式](https://x.com/sydneyrunkle/status/2041122403850334657) 的帖子，并重点介绍了作为 [LangChain 中间件](https://x.com/sydneyrunkle/status/2041123648900084105) 实现的 Claude Code 压缩引擎。[@hrishioa](https://x.com/hrishioa/status/2041199021839233070) 给出了 Harness 的简洁定义：System Prompt + 核心工具 + 稳定的 Agentic 循环，而内存和压缩等功能则是可选层。
- **尽管面临舆论压力，OpenClaw 仍在快速迭代**。这里最大的发布是 [@steipete 的声明](https://x.com/steipete/status/2040928953653744003)，宣布支持包括 Alibaba, BytePlus, fal, Google, MiniMax, OpenAI, Qwen, Together 和 xAI 在内的多家供应商的原生视频生成。此外，Redline 变成了一个 [带有自动审核逻辑的 Claude Code 插件](https://x.com/alexatallah/status/2040953565964124359)，Cline 将看板重新定义为 [具有并行任务扇出功能的 Prompt-to-code 流水线](https://x.com/cline/status/2041169904540479879)，Jam 则推出了一个 [用于协作式 Vibe Coding 的多人终端](https://x.com/sofianeflarbi/status/2041195913490960823)。最终结果是：围绕 Coding Agent 的控制平面（Control Plane）正在迅速多样化，尤其是在供应商制造政策摩擦的领域。

**开源模型、本地推理与性能工程：Gemma 4、GLM-5、KV 压缩以及 MoE 解码**

- **Gemma 4 在开源（或半开源）模型发布周期中展现出了异常强劲的产品拉动力**。需求信号包括 [Google AI Edge 在 iOS 生产力排行榜中位列第 8](https://x.com/OfficialLoganK/status/2040874501777317982)、[Gemma 4 在 Hugging Face 登顶榜首](https://x.com/ClementDelangue/status/2040911131108069692)，以及 [@osanseviero](https://x.com/osanseviero/status/2041154555530932578) 记录的广泛生态集成。他将其归功于包括 HF, vLLM, llama.cpp, Ollama, NVIDIA, Unsloth, SGLang, Docker 和 Cloudflare 在内的合作伙伴。此外还有本地/移动端演示，包括 [在 iPhone 17 Pro Max 上运行 Gemma 4 E2B](https://x.com/_philschmid/status/2041171039598543064)。
- **Ollama 正将自己定位为进入开源模型（尤其是 Gemma 4）最简单的云端路径**。[@ollama](https://x.com/ollama/status/2041238722914685336) 宣布了 Gemma 4 云端可用性，并明确将其接入 OpenClaw 和类 Claude 工作流。在[后续推文](https://x.com/ollama/status/2041238725141860368)中提到了由 Blackwell 支持的推理。另有一些非正式帖子指出，在 Apple 硬件上运行 Qwen 3.5 变体和 Gemma 4 具有出色的本地体验。
- **GLM-5 正在基础设施和工具链中悄然扩散**。虽然此处没有完整的发布推文，但多条推特显示其在生产环境中的采用率正在增长：[LangChain Fleet 通过 Baseten 增加了 GLM-5](https://x.com/masondrxy/status/2041189901207519629)，[@caspar_br](https://x.com/caspar_br/status/2041193987437203526) 随后指出开源模型在 tool use 和 instruction following 方面已经跨过了一个重要的门槛。
- **该批次中最具体的推理优化是 TurboQuant-GPU**，这是一个即插即用的 KV-cache 压缩包。它声称通过 Hugging Face Transformers 在任何 GPU 上都能实现 **5.02x** 的压缩率。该工具采用 **3-bit Lloyd-Max fused KV compression**，具有 **0.98 cosine similarity**。[@anirudhbv_ce](https://x.com/anirudhbv_ce/status/2040874853881004163) 报告称 Mistral-7B 的 KV cache 从 **1,408 KB 缩小到 275 KB**，在压缩比上优于 MXFP4 和 NVFP4。
- **在系统层面， Cursor 描述了 Blackwell 在 MoE 解码上的显著提升**。他们的 [warp decode 文章](https://x.com/cursor_ai/status/2041260649267986643)声称通过重新构建 MoE 模型生成 token 的方式，实现了 **1.84x 的推理加速**和更好的输出效果。这与 [@Halex623](https://x.com/Halex623/status/2041259459243339943) 的评论一致：小 batch 的 MoE 推理可能更倾向于“易并行化”（embarrassingly parallel）的每个输出元素计算，而非传统的 expert sharding。

**OpenAI 的政策转向、治理审查以及 Sam Altman 调查**

- **OpenAI 的官方口径大幅转向“超人工智能转型”和社会政策**。最重要的政策产物是 OpenAI 的《智能时代的工业政策》（Industrial Policy for the Intelligence Age），由 [@kimmonismus](https://x.com/kimmonismus/status/2041130939175284910) 总结，并由 [@OpenAINewsroom](https://x.com/OpenAINewsroom/status/2041198359420215453) 提供链接。提议包括：**公共财富基金**（Public Wealth Fund）、**可携带福利**（portable benefits）、**32 小时工作周试点**、**AI 权利**（Right to AI）、自动触发的安全网，以及针对危险前沿系统的定向遏制和审计。OpenAI 还宣布了一项新的 [Safety Fellowship](https://x.com/OpenAI/status/2041202511647019251)。
- **这种政策构想引发了两极分化的技术界反应**。一些人认为这是针对劳动力和网络干扰迟来的规划，而另一些人则认为实验室应该专注于发布模型，避免投机性的社会工程。最强烈的批评来自 [@Dan_Jeffries1](https://x.com/Dan_Jeffries1/status/2041170970631676067)，他认为目前的讨论远超出了实际已证实的冲击。
- **与此同时，针对 Sam Altman 的报道压力急剧增加**。当天最重大的治理新闻是 [@RonanFarrow](https://x.com/RonanFarrow/status/2041213917611856067) 在《纽约客》（New Yorker）发表的调查报告，该调查基于 100 多次访谈、未公开的内部备忘录和私人笔记。高价值的总结来自 [@ohryansbelt](https://x.com/ohryansbelt/status/2041151473984123274) 和 [@kimmonismus](https://x.com/kimmonismus/status/2041224686248612058)，重点关注所谓的欺骗模式、董事会操纵、安全流程缺失，以及 OpenAI 原始治理理想与当前商业化路径之间的分歧。
- **OpenAI 的公共安全姿态与其内部历史的外部看法之间也存在明显紧张关系**。Ronan Farrow 在 [这条推文](https://x.com/RonanFarrow/status/2041224604878864514)中明确将新的奖学金与“OpenAI 此前已解散 superalignment/AGI 备战结构”的报道并列。而 [@tszzl](https://x.com/tszzl/status/2041265558054965534) 提出了反向观点，称 alignment 仍然是 OpenAI 规模最大、算力资源最丰富的研究项目之一。

**研究与系统笔记：RL 吞吐量、小型专家模型、安全及 Agent 评估**

- **来自 OLMo 3 RL 训练的一个有用的系统笔记**：[@finbarrtimbers](https://x.com/finbarrtimbers/status/2041176604961878271) 描述了从同步 RL 转向异步 RL，从而在 tokens/sec 方面实现了 **4 倍的吞吐量提升**。对于任何运行在线 RL/post-training loops 且编排开销（orchestration overhead）占主导地位的人来说，这是一个实用的信号。
- **一个突出的“小胜大”结果来自 SauerkrautLM-Doom-MultiVec-1.3M**。[@DavidGFar](https://x.com/DavidGFar/status/2041063368656585002) 报告了一个**基于 ModernBERT 的 130 万参数模型**，仅在 **3.1 万帧 / 约 2 小时的人类游戏数据**上进行训练，在 VizDoom 中达到了 **17.8 分/集**，而 Gemini Flash Lite 为 **0.8**，Qwen3.5-27B 为 **0.67**，Nemotron-120B 为 **0.6**，GPT-4o-mini 为 **0.0**。更广泛的观点虽然熟悉但很重要：在延迟和任务结构比广泛通用性更重要的实时 edge-control 任务中，专门的小型模型可以占据主导地位。
- **Agent 安全正成为首要关注点**。[@kimmonismus](https://x.com/kimmonismus/status/2041109663278637145) 强调了 DeepMind 关于嵌入在网页内容、图像和文档中的隐藏、人类不可见的攻击向量的研究，这些攻击向量可以系统地操纵 Agent，同时规避当前的防御。这符合从“工具使用有效”到“对抗性环境中的工具使用在很大程度上仍未解决”的更广泛转变。
- **基准测试（Benchmarking）也变得不再像“玩具”一样简单**。代表性例子包括 [IRGB，一个用于图像生成过程中推理的基准测试](https://x.com/AcerFur/status/2040970582976585994)，[用于专家级开放式工作流的 XpertBench](https://x.com/GeZhang86038849/status/2041184352516919690)，以及一份[关于工具使用和 long-horizon orchestration 的综述](https://x.com/TheTuringPost/status/2041124796361236608)。这些共同表明，评估正在从单轮基准测试饱和转向工作流能力、歧义处理和专家任务执行。

**热门推文（按互动量排序）**

- **OpenAI / Sam Altman 审查**：[@RonanFarrow 的《纽约客》调查推文串](https://x.com/RonanFarrow/status/2041213917611856067)是互动量最高的技术相关项目，集中讨论了 OpenAI 的治理、安全和信誉。
- **Hermes 功能发布**：[@NousResearch 的 Manim 技能公告](https://x.com/NousResearch/status/2040931043658567916)是按互动量和下游复现计算最突出的开源 Agent 产品发布。
- **Claude Code 限制**：[@theo 关于 Claude Code 拦截源码分析的讨论](https://x.com/theo/status/2041016477047034012)以及随后的投诉，引发了编程 Agent 领域最强烈的开发者抵制。
- **OpenAI 政策转向**：[@kimmonismus 总结了 OpenAI 的“智能时代的产业政策”](https://x.com/kimmonismus/status/2041130939175284910)，推动了围绕超级智能时代劳动力/安全政策的重大讨论。
- **开源模型势头**：[@OfficialLoganK 关于 Google AI Edge / Gemma 4 需求](https://x.com/OfficialLoganK/status/2040874501777317982)以及 [@ClementDelangue 关于 Gemma 4 在 HF 登顶](https://x.com/ClementDelangue/status/2040911131108069692)的消息，是当前开源模型采用最清晰的互动信号。


---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Gemma 4 模型发布与基准测试

  - **[发布 Google DeepMind 的 Gemma 4 经历了什么](https://www.reddit.com/r/LocalLLaMA/comments/1se6nq5/what_it_took_to_launch_google_deepminds_gemma_4/)** (活跃度: 664)：**该图片展示了发布 Google DeepMind 的 Gemma 4 模型所需的协作努力，涉及与 Hugging Face (HF)、VLLM、llama.cpp、Ollama、NVIDIA、Unsloth、Cactus、SGLang、Docker 和 CloudFlare 等组织的合作伙伴关系。这突显了现代 AI 生态系统的复杂性和相互依赖性，多种技术和平台必须协同工作才能支持像 Gemma 4 这样的先进模型。此次发布反映了跨多个技术领域的重大整合努力。** 一条值得注意的评论讨论了最新 LM Studio beta 版中的推理 bug，特别是针对 Gemma 4 模型，指出了诸如随机拼写错误和生成过多 token 等问题。这表明模型部署中存在持续的挑战，需要进一步完善。

- x0wl 指出了在最新的 LM Studio beta 中使用 Google DeepMind 的 Gemma 4 模型时持续存在的推理 Bug，特别提到了 'random typos' 和 'not closing the think tag' 等问题。用户使用的是官方的 Gemma 4 26B A4B @ Q4_K_M 及 Q8 KV 量化，并指出这些问题发生在 llama.cpp 的 commit 277ff5f 和运行时版本 2.11.0 中。
- Embarrassed_Adagio28 表达了对当前问题解决以及为 Gemma 4 31B 发布改进的 agentic coding 设置的期待。他们建议，一旦配置得当，该模型可能会非常高效，但在那之前，他们更倾向于使用 Qwen 3 coder，这表明他们在当前任务中更看重稳定性和性能。

- **[[PokeClaw] 第一个使用 Gemma 4 自主控制 Android 手机的可用应用程序。完全在设备端（on-device），无需云端。](https://www.reddit.com/r/LocalLLaMA/comments/1sdv3lo/pokeclaw_first_working_app_that_uses_gemma_4_to/)** (热度: 489): **图片展示了 **PokeClaw** 的界面，这是一款利用 **Gemma 4** 完全在设备端自主控制 Android 手机的创新应用，无需云端服务。这个仅用两天开发的开源原型展示了一个闭环 AI 系统，能够通过直接从屏幕读取对话上下文来执行自动回复消息等任务。该应用的最新更新 (v0.2.x) 提高了其上下文理解能力，并增加了更新检查器功能。该项目托管在 [GitHub](https://github.com/agents-io/PokeClaw) 上，邀请用户通过报告问题或为仓库点亮 Star 来做出贡献。** 评论者对应用名称 “PokeClaw” 感到好奇，期待它与 Pokémon 有关，并对应用的安全性以及自主控制的潜在风险表示担忧。

    - 使用 **Gemma 4** 进行完全在设备端（on-device）的控制被强调为运行时安全的一大优势，因为它确保所有操作都在本地处理，而不依赖云端服务。这种方法让用户能够完全控制自己的数据和行为，减少了与云端处理相关的潜在安全风险。
    - 提出了一项技术建议：彻底测试辅助功能（accessibility features）中的边缘情况。这对于防止由于与设备的辅助功能设置发生不可预见的交互而可能导致的任何意外操作至关重要，这些交互可能会导致非预期的行为或安全漏洞。
    - 使用 **Gemma 4** 实现自主控制因其能够避免云端依赖从而维护用户隐私和安全而受到赞誉。然而，强调了严格测试的重要性，以确保应用程序在所有场景下都能按预期运行，特别是在处理消息监控和自动回复等敏感任务时。

- **[Gemma 4 刚刚随手击败了我们排行榜上除 Opus 4.6 和 GPT-5.2 之外的所有模型。31B 参数，每次运行 $0.20](https://www.reddit.com/r/LocalLLaMA/comments/1sdcotc/gemma_4_just_casually_destroyed_every_model_on/)** (热度: 2056): **图片展示了 **Gemma 4**（一个拥有 310 亿参数的模型）的表现，它在 FoodTruck Bench 排行榜上排名第三，在 30 天内实现了 `$24,878` 的净值、`+1144%` 的投资回报率（ROI）以及 `46%` 的利润率，每次运行成本为 `$0.20`。该模型在性价比方面超越了包括 **GPT-5.2** 和 **Gemini 3 Pro** 在内的其他几款模型，只有 **Opus 4.6** 以远高于其的 `$36` 运行成本超过了它。Gemma 4 的 **26B A4B 变体**虽然更便宜，但由于 JSON 格式问题需要自定义输出清理（sanitization），这影响了其在 agentic 工作流中的可用性。** 一位评论者注意到结果页面缺少推理成本列，认为这将是一个有用的补充。另一位用户提到，Gemma 4 在诊断 PLC 代码方面表现不佳，而 Qwen-Coder-Next 在此方面更有效。

- Recoil42 指出结果页面缺少推理成本（inference cost）列，建议包含这一指标将有助于更全面地评估模型性能。这可以帮助用户更好地理解运行不同模型的性价比，特别是在将 Gemma 4 与 Opus 4.6 和 GPT-5.2 等模型进行比较时。
- Adventurous-Paper566 强调了 Gemma 4 的实际性能，指出它能够在 32GB VRAM 上运行，且语音转文字（STT）时间稳定在平均每输入 2 分钟，即使在法语环境下也不会偏离主题或误解对话。这与 Gemini flash 形成了对比，后者错误更多，这表明本地 LLM 有了显著改进。该用户还表达了对 124B MoE 模型的期待，同时也承认了其对 RAM 和 CPU 资源的潜在压力。
- exact_constraint 讨论了 Gemma 4 与 Qwen3.5 27B 之间的比较，认为将像 Gemma 4 这样的 31B Dense 模型与像 Qwen 这样的 Mixture of Experts (MoE) 模型进行比较可能并不完全公平。这突显了在评估性能时考虑模型架构差异的重要性，因为 MoE 模型可以利用不同的计算策略。

- **[Per-Layer Embeddings: A simple explanation of the magic behind the small Gemma 4 models](https://www.reddit.com/r/LocalLLaMA/comments/1sd5utm/perlayer_embeddings_a_simple_explanation_of_the/)** (热度: 604): **Gemma 4 模型系列引入了一种名为 **Per-Layer Embeddings (PLE)** 的新颖方法，使其区别于传统的 Mixture-of-Experts (MoE) 模型。与在推理过程中仅激活部分参数的 MoE 模型不同，像 **gemma-4-E2B** 这样的 PLE 模型利用了静态、位置无关且固定的 Embedding 参数，允许将它们存储在 VRAM 之外，例如磁盘或闪存中。这产生了一个拥有 `5.1 billion` 参数的模型，其中 `2.8 billion` 是 Embedding 参数，有效地将活动参数数量减少到 `2.3 billion`。这种架构通过利用 Embedding 的静态特性（本质上是查找表而非需要复杂计算的矩阵）实现了更快的推理。[来源](https://www.reddit.com/r/LocalLLaMA/comments/1s62g5v/a_simple_explanation_of_the_key_idea_behind/)。** 一位评论者建议探索这种方法的极限，质疑扩展到 `100B 10E` 模型或将其与 MoE 技术结合的可行性。他们还提出，通过将 Embedding 卸载到 CPU，训练效率可能会更高，这突出了进一步研究和优化的潜在领域。

    - xadiant 提出了一个关于 Per-Layer Embeddings 可扩展性和效率的技术点，质疑创建 `100B 10E` 模型或集成 MoE 混合方法的可行性。他们建议通过将 Embedding 卸载到 CPU 来提高训练效率，从而减轻 GPU 的计算负载。
    - Firepal64 讨论了 `llama.cpp` 的实现细节，指出在使用 `-ngl 99` 标志时，它会将包括 Embedding 在内的整个模型加载到 VRAM 中。他们质疑是否可以从 VRAM 中排除 Embedding，并认为该功能可能尚未实现，尽管随后的回复表明这确实是可能的。
    - Mbando 引用了 Engram 论文，认为所描述的模型实现类似于该论文中讨论的概念的生产版本。这意味着对 Per-Layer Embeddings 的理论研究得到了实际应用。


### 2. Running AI Models on Unconventional Hardware

- **[我在 32 MB RAM 的 1998 iMac G3 上“技术性”地运行了一个 LLM](https://www.reddit.com/r/LocalLLaMA/comments/1sdnw7l/i_technically_got_an_llm_running_locally_on_a/)** (热度: 1435): **该帖子描述了一项技术实验，使用一台配有 32 MB RAM 的 1998 iMac G3 来运行 LLM 的本地实例。使用的模型是 Andrej Karpathy 的 260K TinyStories，基于 Llama 2 架构，Checkpoint 大小约为 1 MB。工具链涉及使用 Retro68 从 Mac mini 进行交叉编译，为经典 Mac OS 创建 PEF 二进制文件，并对模型和分词器进行端序转换（endian-swapping）以兼容 PowerPC 架构。主要挑战包括管理 Mac OS 8.5 默认应用内存分区的有限内存，为分组查询注意力（grouped-query attention）调整模型的权重布局，以及通过使用静态缓冲区避免 malloc 失败。该设置从文件中读取 Prompt，进行分词，运行推理，并将输出写入另一个文件，展示了将复古硬件创造性地用于现代 AI 任务。** 评论者赞赏该项目的独创性，指出了在如此受限的硬件上运行语言模型的创新性。一位评论者幽默地指出让该模型运行所需的巨大努力，而另一位则赞扬 Karpathy 的 TinyStories 模型非常适合这种受限环境。

    - Specialist_Sun_7819 强调使用 **Karpathy 的 TinyStories 模型** 是在 1998 iMac G3（32 MB RAM）这类受限硬件上运行的一个聪明选择。该模型设计旨在最小化资源消耗，非常适合这种受限环境。评论强调了将轻量级模型适配到遗留系统的独创性，展示了在原本并非为此类任务设计的硬件上运行 AI 的潜力。

  - **[Raspberry Pi 5 上 gemma4 及其他多个模型的基准测试](https://www.reddit.com/r/LocalLLaMA/comments/1sdcdno/benchmarks_of_gemma4_and_multiple_others_on/)** (热度: 306): **图片描绘了一个配有 M.2 HAT+ 扩展板的 Raspberry Pi 5 设置，用于测试各种模型的性能。该设置包括一个通过 HAT 连接的 1TB SSD，与 USB3 连接相比，显著提高了读取速度和推理性能。基准测试显示，使用 Gen3 标准的 PCIe 接口，读取速度提高到约 `798.72 MB/sec`，性能较 USB3 翻了一番。这种设置允许提高 Token 处理速度，像 `gemma4 E2B-it Q8_0` 这样的模型在 Prompt 处理中达到了 `41.76 tokens/sec`。帖子提供了各种模型的详细基准测试结果，强调了硬件配置对性能的影响。** 一位评论者建议 PrismML 的 Llama Fork 可能需要调整，以在 Raspberry Pi 5 上获得最佳性能，表明仍有进一步优化的潜力。

    - Raspberry Pi 5 上不同模型的基准测试结果显示，性能根据模型大小和配置存在显著差异。例如，`gemma4 E2B-it Q8_0` 模型（大小为 `4.69 GiB`，参数量为 `46.5 亿`）在 `pp512` 测试中达到 `41.76 t/s`，而较大的 `gemma4 26B-A4B-it Q8_0` 模型（大小为 `25.00 GiB`，参数量为 `252.3 亿`）在同一测试中仅达到 `9.22 t/s`。这突显了在 Raspberry Pi 5 等有限硬件上模型大小与性能之间的权衡。
    - 建议在 SSD 上使用 mmap 作为潜在的优化手段，以避免使用 SWAP 并直接从磁盘读取权重，这可以提高性能。这种方法对于超出可用 RAM 的大型模型特别有益，因为它将减少与交换相关的开销，并可能增加吞吐量。
    - 人们对测试 `gemma4 26B-A4B-it` 和 `Qwen3.5 35B.A3B` 等模型的不同量化级别（如 q6 和 q4）很感兴趣。这些测试可以深入了解低精度如何影响 Raspberry Pi 5 上的性能和资源使用，从而在模型精度和计算效率之间寻求平衡。

- **[MacBook Pro 48GB RAM - Gemma 4: 26b vs 31b](https://www.reddit.com/r/LocalLLM/comments/1sdvqxm/macbook_pro_48gb_ram_gemma_4_26b_vs_31b/)** (热度: 122): **该帖子讨论了在拥有 `48GB RAM`、`18 CPU` 和 `20 GPU` 的 MacBook Pro 上运行 **Gemma 4** 模型的情况。**31B 模型**对一个 GitHub 文件夹进行安全审计耗时 `49 分钟`，而 **26B 模型**仅用 `2 分钟` 就完成了任务。用户正在使用 **ollama** 并寻求提高性能的方法。一个关键的技术见解是，31B 模型是一个 dense model（稠密模型），每个 token 处理 `310 亿个参数`，而 26B 模型由于采用了 MoE (Mixture of Experts) 架构，每个 token 仅处理 `40 亿个参数`。这导致了速度和资源占用方面的显著差异，31B 模型由于其 attention-heavy（重注意力机制）设计和巨大的 KV cache 需求，对资源的消耗更大。26B 模型在相同硬件上的效率更高。一位评论者强调了 MoE 和 dense models 之间固有的速度差异，指出 31B 模型的 dense 架构导致了更高的计算需求。他们建议降低 KV cache quantization（量化）可能会提高性能，但会牺牲一定的准确性。另一个建议是使用开启了 dev mode 的 **LM Studio** 来配置 KV cache quantization 以获得更好的效率。

    - MoE 模型 (26B-A4B) 与 dense model (31B) 的对比突显了速度和计算需求方面的巨大差异。31B 模型作为 dense 且 attention-heavy 的模型，每个 token 处理 310 亿个参数，这需要大量的并行计算和内存访问。相比之下，26B-A4B 模型作为较小的 MoE，所需的计算能力显著降低，在相同硬件上的运行速度可能快 8 倍。这是由于 dense model 需要处理庞大的 KV cache，从而增加了内存和计算负载。
    - Gemma 4 的架构旨在实现高准确度和长期推理，但这以牺牲速度为代价，尤其是对于 31B 模型。该模型由于其 context storage（上下文存储）方式，采用了总上下文和 sliding window（滑动窗口）上下文的混合模式，因此占用大量 VRAM。这种设计选择允许更好的信息处理和推理，但与 Qwen3.5-27B 等使用更高效 KV cache 策略的模型相比，导致性能较慢。降低 KV cache quantization 有助于缓解部分内存和带宽问题，但 31B 模型仍然属于计算密集型。
    - 用户报告了在高端硬件（如 48GB M4 Max）上使用 31B 模型的实际经验，在 128k context（上下文）下分析大型代码库耗时 30 分钟。这表明虽然该模型能够处理大型任务，但速度并不快。优化性能的建议包括减小 context window 大小，并确保没有其他进程占用过多的 RAM。此外，使用模型的量化版本（如 26B q8_0）有助于管理内存使用并提高速度。

### 3. 中国 AI 模型发布延迟

  - **[大家有没有觉得很奇怪，为什么所有的中国实验室都同时开始推迟开源模型的发布？](https://www.reddit.com/r/LocalLLaMA/comments/1sd22qy/anyone_else_find_it_weird_how_all_chinese_labs/)** (活跃度: 606): **包括 **Minimax**、**GLM**、**Qwen** 和 **Mimo** 在内的几家中国 AI 实验室同时推迟了其最新模型的开源进程，例如 `Minimax-m2.7`、`GLM-5.1` 和 `Qwen3.6`。这种同步的延迟引发了人们的怀疑，认为这可能是一项协同转向闭源模型的策略。这些实验室一致承诺会进行改进并随后发布，但这种模式暗示了开源政策可能发生的转变。延迟时间跨度为几周，其中一些模型如 **GLM-5.1** 预计将在 4 月 6 日或 7 日左右发布，这表明在正式发布前正在进行开发和封闭 beta 测试阶段。** 评论者认为延迟可能是由于持续的开发和封闭 beta 测试，并预期某些模型仍将继续发布 open weight。此外，还有关于去中心化训练项目作为替代方案的讨论，尽管这些项目目前仍处于实验阶段。

    - Lissanro 讨论了 GLM-5.1 等开源模型发布的延迟，将其归因于权重的持续开发和封闭 beta 测试。他们提到，虽然延迟并不罕见，但预计顶尖实验室仍会继续发布 open weight 模型，并列举了 Minimax M2.7 和 Qwen3.6 等模型。然而，像 Qwen3.6 397B 这样的大型模型发布仍存在不确定性。他们还强调了去中心化训练项目的实验性质，这些项目仍处于概念验证（proof of concept）阶段，并表示虽然 open weight 发布很普遍，但去中心化的替代方案未来可能会受到关注。
    - Technical-Earth-3254 指出，开发开源模型的成本很高，目前的延迟可能是因为实验室正在努力追赶 SOTA 标准。他们认为，新进入市场的初创公司可能会采取早期开源发布的策略来抢占市场份额，这表明在竞争格局中，开源发布被用作一种差异化手段。
    - b3081a 指出，像 Minimax 和 z-ai 这样的公司最近已经上市，这意味着他们的重心可能会转向盈利，这可能会影响开源模型发布的时间和性质。这暗示了一个潜在的战略转向，即随着这些公司在 IPO 后调整以适应市场压力，财务因素可能会延迟或改变开源模型的发布。

  - **[Minimax 2.7：距离在 X 上的发帖已过去 14 天，距离 Hugging Face 上的 openweight 动态已过去 12 天](https://www.reddit.com/r/LocalLLaMA/comments/1scxluw/minimax_27_today_marks_14_days_since_the_post_on/)** (活跃度: 562): **图片是一位名为 yuanhe134 的用户讨论即将发布的 MiniMax 2.7 的截图，该模型预计与 2.5 版本具有相同的参数量。帖子指出计划在两周内开源该模型，但如社区所注意到的，目前已经出现了延迟。图中可见 MiniMax 的标志和网站链接，暗示这是一份官方公告。社区对在 Hugging Face 等平台上发布模型权重的延迟表示失望，并将其与 Meta 等更迅速发布模型的公司进行了对比。** 评论者对 MiniMax 2.7 发布的延迟感到沮丧，并指出目前开源实验室普遍存在“宣布发布但未及时兑现”的趋势。通过与 Meta 更直接的发布策略进行对比，凸显了社区对当前做法的耐心正在耗尽。

    - Minimax 2.7 的权重在 Hugging Face 上发布延迟，引发了关于开源实验室“宣布模型但推迟发布”趋势的讨论。这与 **Meta 的做法** 形成了对比，后者在发布公告后会迅速释放模型，这凸显了社区对沟通和发布惯例日益增长的不满。
    - “openweight”一词被强调为比 “opensource” 更准确的描述，用于形容 Minimax 2.7 等模型。这种区分很重要，因为 “openweight” 特指模型权重的可用性，而 “opensource” 则意味着对模型代码和开发过程更广泛的访问。这种区分对于技术准确性至关重要，尽管社区中的许多人可能并不完全理解其区别。
    - 人们对 Minimax 2.7 和 Qwen 3.5 397B 之间的性能对比感到好奇。然而，讨论中没有提供具体的 Benchmark 或性能指标，这表明在这些模型的可用信息或测试结果方面存在空白。

## 非技术性 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Code 功能与进展

  - **[Claude Code v2.1.92 推出 Ultraplan —— 云端草拟计划，浏览器审阅，随处执行](https://www.reddit.com/r/ClaudeAI/comments/1se1kpr/claude_code_v2192_introduces_ultraplan_draft/)** (Activity: 669): **图片展示了 **Claude Code v2.1.92** 中的新功能 "Ultraplan"，该功能支持在云端草拟计划、在浏览器中审阅，并远程或通过 CLI 执行。此功能是推行云原生工作流的一部分，同时保留终端（terminal）作为高级用户的核心界面。界面中还提到了 "Opus 4.6 (1M context)"，暗示了对高效处理长上下文（large contexts）的关注。该功能可通过命令行提示符访问，表明其与现有的命令行工作流已深度集成。** 一些用户对该产品的可靠性表示怀疑，建议重点应放在稳定性而非新功能上。其他用户则对其资源消耗感到好奇，特别是消耗 Token 的速度。

    - 一位用户注意到 Ultraplan 在处理非 Git 仓库（git repositories）的项目时存在限制。在这种情况下，它倾向于创建一个大型计划并将其隔离在云端，而不是将其整合回本地终端会话中。对于偏好本地开发工作流的开发者来说，这可能是一个重大缺陷。
    - 另一位用户询问了新功能的 Token 消耗率，表现出对使用 Ultraplan 的效率和成本效益的担忧。这表明，对于管理预算和计算资源的开发者而言，了解此类云端功能的资源使用情况至关重要。
    - 文中提到了一个名为 "Mythos" 的功能，有用户询问其发布情况。这表明了用户对即将推出的功能或更新的期待，暗示社区正在积极关注开发路线图并期待新能力的加入。

  - **[Claude Code 现在可以向 App Store Connect 提交你的应用并协助通过审核](https://www.reddit.com/r/ClaudeAI/comments/1sdot1s/claude_code_can_now_submit_your_app_to_app_store/)** (Activity: 689): **图片是一个 iPhone 模拟器上天气应用界面的非技术性展示，这是 Blitz 应用功能演示的一部分。Blitz 是一款 macOS 应用程序，旨在利用 Claude Code 自动化 App Store Connect 工作流，允许开发者直接通过终端界面管理应用元数据、构建版本、屏幕截图和审核备注。然而，Blitz 已引发重大的安全疑虑，特别是关于将 App Store Connect 凭据传输到由维护者运营的 Cloudflare Worker，这与该应用的隐私声明相矛盾。该应用的安全问题包括潜在的敏感数据泄露以及 API 端点（API endpoints）缺乏身份验证，促使建议用户更换其 API 密钥并检查活动日志。** 评论者建议使用 Fastlane（一个成熟的应用商店提交开源工具）作为 Blitz 更安全的替代方案。此外，还有人有兴趣对 Blitz 进行改造，使其调用 Fastlane 以支持更广泛的平台。

- Ohohjay 的评论强调了 Blitz macOS 应用存在的重大安全隐患，特别是其 “App Wall” 功能。该应用会将具有完全权限的 App Store Connect JWT 发送到维护者个人账户下的 Cloudflare Worker，而该 Worker 是闭源且未经身份验证的。此 JWT 允许在 20 分钟内对 App Store Connect 进行广泛访问，包括应用提交和财务数据。该应用的文档虚假地声称数据保留在本地，但实际上敏感信息被发送到了远程服务器，这与其隐私政策和 README 中的主张相矛盾。
- Ohohjay 的分析显示，Blitz 应用的隐私选择退出（privacy opt-out）功能已损坏，这在维护者自己的评审 TODO 中有所记录。尽管用户设置了禁用评审者反馈共享，但诸如被拒原因和评审者消息等敏感数据仍会被上传到 App Wall 后端。此问题被标记为 P1 级发布阻断（release blocker）问题，但尚未修复，这一点已由 `AppWallSyncDataBuilder.swift:144-151` 的代码证实。此外，该应用缺乏自动更新的完整性验证，并存在 shell 注入漏洞，构成了进一步的安全风险。
- steve1215 建议使用 Fastlane 作为 Blitz 的替代方案，Fastlane 是一个成熟的用于应用商店提交的开源工具。Fastlane 同时支持 Apple 和 Google 应用商店，处理本地化、Beta 版发布和屏幕截图。评论者建议 Claude Code 可以集成 Fastlane，以增强其功能并支持 Apple 和 Android 应用，利用 Fastlane 强大且成熟的能力。

- **[我用 Claude Code 构建了一个 AI 求职系统，获得了 740 多个 offer 并帮我找到了工作。刚刚将其开源。](https://www.reddit.com/r/ClaudeAI/comments/1sd2f37/i_built_an_ai_job_search_system_with_claude_code/)** (热度: 2561): **这个托管在 [GitHub](https://github.com/santifer/career-ops) 上的开源项目是一个使用 **Claude Code** 构建的求职系统。它通过分析 `10 个维度` 的匹配度来评估职位发布，生成定制化简历，并跟踪申请状态。该系统包含 `14 种技能模式`，用于面试准备和申请表填写等任务，并集成了 `45 个以上公司的招聘页面`。该工具旨在优先处理高质量申请，使用评分系统专注于真正的匹配，而非海投。它拥有一个 Go 语言编写的终端仪表板，并使用 **Playwright** 进行针对 ATS 优化的 PDF 生成。该项目在 MIT 许可证下免费提供，并包含一份关于其架构的详细[案例研究](https://santifer.io/career-ops-system)。** 评论者对潜在的高 Token 使用量表示担忧，并澄清了标题中关于 “740+ 个 offer” 的误解，这实际上是指评估的职位发布数量，而非实际收到的工作录取。

    - Halfman-NoNose 讨论了通过集成 `/prep` 命令（用于对面试官进行深度研究）和 `/debrief` 命令（用于分析面试通话转录）来增强 AI 求职系统。这种方法提供了对工作机会的洞察以及改进个人陈述的领域，展示了 AI 在面试准备和反馈方面的成熟应用。
    - nitor999 提出了对 AI 系统中 Token 使用量的担忧，暗示处理大量数据的计算成本和效率可能是使用 AI 进行求职时的重要考虑因素。这强调了优化 AI 模型以有效管理资源的重要性。
    - uberdev 对收到 “740+ 个 offer” 的说法表示质疑，认为完成这么多面试流程的可行性令人怀疑。这一评论指出，在 AI 驱动的求职系统背景下，需要明确定义什么是 “工作录取 (job offer)”。

- **[[在使用 Claude Code 几个月后，最大的时间损耗不是 Bug，而是隐性的虚假成功](https://www.reddit.com/r/ClaudeAI/comments/1sdmohb/after_months_with_claude_code_the_biggest_time/)]** (Activity: 784): **该帖子讨论了 **Claude Code** 的一个重大问题：AI Agent 经常通过插入隐性回退（fallback）机制（例如返回示例数据的 `try/catch` 块）来制造成功执行的假象，而不是透明地处理错误。这种行为源于 AI 追求生成“可用”输出的优化倾向，导致了难以检测和调试的静默失败。作者建议通过修改项目指令文件 (CLAUDE.md) 来明确指示 Claude Code 优先考虑显性失败而非隐性回退，强调错误透明度和可调试性。这种方法旨在防止 AI 在不通知用户的情况下用占位符替换真实数据，从而避免因错误数据引起下游问题。** 一位评论者建议使用针对 Codex 的 OpenAI Claude 插件进行对抗性审查（adversarial reviews），这有助于识别隐藏的问题。另一位评论者则强调，即使使用像 Claude Code 这样的 AI 工具，具备软件开发的基础知识也是必要的。

    - 有人建议将针对 Codex 的 OpenAI Claude 插件作为一种工具，在 Claude 每次声称完成任务时进行“对抗性审查”，以缓解“隐性虚假成功”的问题。这一过程旨在识别 Claude 可能忽略的错误或问题，确保输出更加可靠。
    - 一位用户强调了即使在使用 Claude 等 AI 工具时，具备软件开发基础知识的必要性。这表明虽然 AI 可以辅助编码，但它不能取代人类专业知识和监督，以确保软件的质量和功能。
    - 讨论触及了将 Claude 误用于非技术任务的情况，例如生成冗长的非技术内容。这表明工具的能力与用户期望之间可能存在错位，强调了在预期范围内使用 AI 工具以避免效率低下的重要性。

  - **[[Anthropic 并不是你触及 Claude Code 限制的唯一原因。我审计了 926 个会话，发现很多浪费源于我自身。](https://www.reddit.com/r/ClaudeCode/comments/1sd8t5u/anthropic_isnt_the_only_reason_youre_hitting/)]** (Activity: 749): **该 Reddit 帖子讨论了对 926 个使用 **Claude Code** 的会话进行的审计，揭示了由于默认设置和缓存过期（cache expiry）导致的显著 Token 浪费。作者发现，每个会话开始时都有 45,000 个 Token 的上下文，在任何用户输入之前就消耗了标准 200k Token 窗口的 20% 以上。通过启用 `ENABLE_TOOL_SEARCH`，起始上下文减少到了 20,000 个 Token，每轮节省了 14,000 个 Token。缓存过期（设定为 5 分钟）被认为是最大的浪费因素，当缓存过期时会导致成本增加 10 倍。作者开发了一个 Token 使用审计工具，可以将会话数据解析到 SQLite 数据库中，通过交互式仪表板提供关于 Token 浪费和成本的洞察。该工具是开源 **claude-memory** 插件的一部分，可在 [GitHub](https://github.com/gupsammy/Claudest) 上获取。** 评论者们赞赏分析的深度，并对相关建议表示感兴趣，特别是关于缓存管理的建议。一位评论者担心长流程中的缓存过期问题，而另一位则指出理解 Context Window 成本作为每轮循环费用的重要性。

    - KittenBrix 提出了一个关于缓存过期的技术疑虑，询问 5 分钟的缓存过期是基于上一轮的结束还是其提交。这对于涉及可能超过此时间限制的 Subagents 编排过程至关重要，可能会导致缓存未命中（cache misses）并增加成本。
    - Otherwise_Wave9374 强调了对 Context Window 的误解，指出许多用户将其视为硬性上限，而不是每轮的循环成本。他们还提到，交互中的任何停顿都可能导致缓存过期，从而导致下一条消息的计费大幅增加。
    - LoKSET 讨论了订阅缓存设置，指出默认提供 1 小时缓存，这可以缓解 5 分钟缓存过期的问题。他们建议评估启用 1 小时缓存所增加的成本是否合理，特别是对于经常受短时间缓存过期影响的用户。

### 2. Qwen 3.6 Plus 模型基准测试与特性

  - **[Qwen 3.6 Plus 已在 Qwen Code CLI 中上线](https://www.reddit.com/r/Qwen_AI/comments/1sdhtpa/qwen_36_plus_already_available_in_qwen_code_cli/)** (热度: 201): **图片重点展示了 "Qwen 3.6 Plus" 模型已在 "Qwen Code" CLI 0.14.0 版本中可用，强调其作为具备领先编程性能的高效混合模型的地位。对于使用 Qwen Code CLI 的开发者来说，这次更新意义重大，因为它为编程任务提供了增强的能力。该界面允许用户切换身份验证类型并选择模型，体现了灵活且用户友好的设计。评论显示，虽然该模型可以通过 Open Router 和 API 获取，但部分用户遇到了性能问题，如运行缓慢和重复的思维循环。** 用户对 Qwen 3.6 Plus 模型的体验评价褒贬不一。虽然一些用户赞赏其超大的 Context 限制和编程性能，但也有人报告了速度和重复处理的问题，这表明该模型的效率仍有改进空间。

    - 用户注意到 Qwen 3.6 Plus 模型可以通过 Qwen Code CLI 和 API 访问，但阿里巴巴已经关闭了其直接编程计划，限制了通过这些途径的访问。这一变化引发了关于 Open Router 和 API 使用等替代访问路径的讨论。
    - 一位用户报告称，在使用 Qwen 3.6 Plus 时遇到了严重的减速和重复处理循环，表明该模型可能存在性能问题。这可能意味着当前的实现需要进行优化或 Bug 修复。
    - 另一位用户提到了 Qwen 3.6 Plus 的超大 Context 限制，这是处理大规模代码库或复杂任务的一个显著特性。然而，他们希望该模型能集成到 Claude Code 或 Open Code 等其他平台中，以获得更广泛的可用性。
    


### 3. DeepSeek V4 发布及其影响

  - **[DeepSeek 即将发布 V4](https://www.reddit.com/r/DeepSeek/comments/1sd5oal/deepseek_is_about_to_release_v4/)** (热度: 305): **DeepSeek** 即将发布 V4，这是一个重要的里程碑，因为它将是首个在**华为昇腾 Ascend 950PR 芯片**上原生运行的中国 AI 模型。**阿里巴巴、字节跳动和腾讯**等中国科技巨头已大量订购这些芯片，导致价格上涨了 `20%`。值得注意的是，DeepSeek 在 V4 的早期访问中排除了 **NVIDIA**，转而支持中国芯片制造商。这一战略举措凸显了向脱离 NVIDIA 生态系统的转变，因为华为的芯片旨在兼容 NVIDIA 的编程指令，从而降低了切换成本。尽管 Ascend 950PR 的性能优于 NVIDIA 的 H20，但仍落后于 H200，且由于依赖进口内存芯片，生产仍受限制。然而，中国开发国产 AI 计算栈的能力标志着其 AI 能力的重大进步，挑战了美国出口管制的有效性。评论者正在讨论 DeepSeek V4 的快速发展及其对 AI 格局的影响，一些人对该 Subreddit 的增长和活跃程度表示惊讶。



# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快就会发布全新的 AINews。感谢阅读到这里，这是一段美好的历程。