---
companies:
- openai
- anthropic
- langchain-ai
- google-deepmind
date: '2025-11-14T05:44:39.731046Z'
description: '**OpenAI** 推出了 **GPT-5.1**，其特色功能包括“自适应推理”以及针对开发者的 API 改进，例如提示词缓存（prompt
  caching）和用于权衡延迟与成本的推理力度（reasoning_effort）调节开关。独立分析显示，该模型在智力水平上有小幅提升，但在智能体编程基准测试中进步显著。


  **Anthropic** 的 **Claude** 模型为 Sonnet 4.5 和 Opus 4.1 引入了符合 JSON 模式（JSON schema）的结构化输出公测版，增强了工具调用和代码执行工作流。此前关于
  Opus 4.5 发布的传闻已被辟谣。


  **LangChain** 发布了“Deep Agents”软件包和上下文工程指南（context-engineering playbook），旨在优化智能体工作流。


  社区正热切期待 **Google DeepMind** 的 **Gemini 3** 模型，社交媒体及即将举行的 AIE CODE 活动中均有相关暗示。*“门票已售罄，但仍有周边活动和志愿者机会。”*'
id: MjAyNS0x
models:
- gpt-5.1
- sonnet-4.5
- opus-4.1
- gemini-3
people:
- swyx
- allisontam_
- gdb
- sama
- alexalbert__
- simonw
- omarsar0
- abacaj
- scaling01
- amandaaskell
title: 今天没发生什么特别的事。
topics:
- adaptive-reasoning
- developer-tools
- prompt-optimization
- json-schema
- agent-workflows
- context-engineering
- structured-outputs
- model-release
- benchmarking
---

**Gemini 3 什么时候发布？**

> 2025/11/13-2025/11/14 的 AI 新闻。我们为您检查了 12 个 Reddit 子版块、544 个 Twitter 账号和 24 个 Discord 社区（205 个频道，6489 条消息）。预计节省阅读时间（以 200wpm 计算）：514 分钟。我们的新网站现已上线，支持全元数据搜索，并以精美的 vibe coded 风格展示过往所有内容。查看 https://news.smol.ai/ 获取完整的新闻拆解，并在 @smol_ai 上向我们提供反馈！

有很多关于 Gemini 3 的[暗示](https://x.com/swyx/status/1989227065732980981)。我很好奇 GDM 为 [AIE CODE 的盛大开幕](https://www.ai.engineer/code) 准备了什么...

（附：门票已售罄，但你可以参加 [周边活动](https://x.com/swyx/status/1989461220131574013) 或免费申请 [志愿者](https://docs.google.com/forms/d/e/1FAIpQLSf-vO_ANN96lZ5myPM80si6_kfMfCy68VEtpomRnt6N_r12Iw/viewform)）。

---

# AI Twitter 回顾

**OpenAI 发布 GPT‑5.1：自适应推理、面向开发者的 API 以及新的 UX 试点**

- **自适应推理 + 开发者聚焦**：OpenAI 发布了 GPT‑5.1，为 5.1‑Instant 引入了“自适应推理”，据 [@allisontam_](https://twitter.com/allisontam_/status/1989138927970848936) 称，这一变化需要全新的 post-training/RL 工作。在平台方面，OpenAI 强调了更好的开发者人体工程学：“API 中出色的新模型和扩展的 prompt caching，”据 [@gdb](https://twitter.com/gdb/status/1989135114744573993) 称。团队还发布了针对长期运行/Agentic 任务（计划工具、持久性）的具体指南，以及一个 Prompt Optimizer，包括一个 reasoning_effort 开关，用于在延迟/成本与质量之间进行权衡，信息来自 [OpenAI DevRel](https://twitter.com/OpenAIDevs/status/1989378869976326570)、[建议 2](https://twitter.com/OpenAIDevs/status/1989378875126886560)、[建议 3](https://twitter.com/OpenAIDevs/status/1989378876922077560)。另外，OpenAI 正在日本、新西兰、韩国和台湾试点 ChatGPT “群聊”功能（[公告](https://twitter.com/OpenAI/status/1989138776585851038)）。
- **基准测试和 UX 上的实测增量**：独立追踪机构 Artificial Analysis 发现 GPT‑5.1 相比 GPT‑5 是一个“微小”的智能提升（其指数 +2），主要由 TerminalBench（Agentic 编码/终端使用）上 +12pp 的增长驱动。非推理端点没有变化；价格保持不变，且 Token 效率略有提高，据 [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1989417492582899872) 称。据传，GPT‑5.1 在创意写作 ([@gdb](https://twitter.com/gdb/status/1989230190111912041)) 和代码方面表现更好（尝试 5.1‑Codex：[@gdb](https://twitter.com/gdb/status/1989226363069559173)；VS Code 中的实时测试：[@JamesMontemagno](https://twitter.com/JamesMontemagno/status/1989354367343075697)）。可控性也有细微但可见的改进（例如，遵循自定义指令的风格约束），据 [@sama](https://twitter.com/sama/status/1989193813043069219) 称。

**Anthropic/Claude：Structured Outputs 落地；Claude Code 势头强劲；Opus 4.5 传闻澄清**

- **Structured Outputs（公测版）**：Claude API 现在保证符合 JSON schema/工具规范的响应，无需重试或解析技巧，最初支持 Sonnet 4.5 和 Opus 4.1（Haiku 4.5 即将推出）。文档和 API 示例见 [@alexalbert__](https://twitter.com/alexalbert__/status/1989409186674098595) 和 [后续](https://twitter.com/alexalbert__/status/1989409198971801905)。资深用户指出，这统一了之前的“带有 schema 的单一工具”方案 ([@simonw](https://twitter.com/simonw/status/1989411809351303430))。
- **用于 Agent 运维的 Claude Code**：开发者强调了 Claude Code 在优化 Agent 工作流方面的有效性，当给定日志/测试脚本和紧凑的 harness 时——更少的工具调用，更多的代码执行和清晰的规范 ([@omarsar0](https://twitter.com/omarsar0/status/1989417433245925645)；“Pulse”主动研究 Agent：[演示](https://twitter.com/omarsar0/status/1989350215175020682))。社区情绪仍然认为 Opus 4.1 在综合质量上依然难以被超越 ([@abacaj](https://twitter.com/abacaj/status/1989128220835537315))。
- **传闻控制**：在 Claude Code CLI PR 中发现的 “Opus 4.5” 引用似乎是自动补全/日期伪影，而非秘密发布 ([@scaling01](https://twitter.com/scaling01/status/1989145846114578547), [1](https://twitter.com/scaling01/status/1989145863508394059), [2](https://twitter.com/scaling01/status/1989146991272817048))。在政策领域，Anthropic 重申了基于规范的方法，以公平处理政治话题 ([@AmandaAskell](https://twitter.com/AmandaAskell/status/1989328363077382407))。

**Agent、协议和工具链：Deep Agents、MCP 周年以及更安全的自主 UX**

- **Deep Agents + 上下文工程 (context engineering)**：LangChain 发布了 “Deep Agents” 软件包/CLI，并分享了上下文工程手册：包括卸载/减少/隔离模式、评估框架（eval harnesses），以及为什么产品要围绕未来的模型重新构建架构 ([概览](https://twitter.com/LangChainAI/status/1989152093127782765), [播客](https://twitter.com/jakebroekhuizen/status/1989130283866812437))。
- **通过协议实现互操作性**：CopilotKit 的一份简明全景图指出，三种互补的开放协议正在标准化跨 LangGraph/CrewAI/Agno 的 Agent 栈：AG‑UI（Agent ↔ 用户）、MCP（上下文/工具链）和 A2A（Agent ↔ Agent），从而实现无需重写 UI 的组合搭配 ([@_avichawla](https://twitter.com/_avichawla/status/1989228336997101946), [PDF](https://twitter.com/_avichawla/status/1989228348971893236))。
- **MCP 一周年**：Anthropic 和 Gradio 启动了 MCP 一周年黑客松；超过 6,300 名报名者，为期 2 周，奖金 2 万美元 + 350 万美元算力额度 ([@Gradio](https://twitter.com/Gradio/status/1989315723336749412), [@huggingface](https://twitter.com/huggingface/status/1989386669636948321))。
- **Agent UX 中的信任与控制**：Perplexity 的 Comet Assistant 现在在执行敏感操作（登录/购买）前会暂停，显示执行轨迹（traces），并要求明确许可——这是 Agent 浏览中受限“数字员工”模式的一个范例 ([@perplexity_ai](https://twitter.com/perplexity_ai/status/1989416343331012971))。
- **Agent 编程平台**：Cline 在 VS Code、JetBrains 和 CLI 中增加了对 Nous 的 Hermes 4 70B/405B 的直接支持 ([@NousResearch](https://twitter.com/NousResearch/status/1989427241424654534); [@cline](https://twitter.com/cline/status/1989432694867193988))。

**开发工具与基础设施：代码库问答、编程 Agent 和实验流水线**

- **代码库感知型代码问答**：Google 的 “CodeWiki” 允许你使用自然语言查询代码库（函数位置、架构、逻辑），并在社区测试中对 DSPy 代码库表现良好 ([@getpy](https://twitter.com/getpy/status/1989237111745310770))。
- **Qwen Code v0.2.1**：快速迭代（17 天内发布 8 个版本），提供：免费网络搜索（多供应商；OAuth 用户每天 2,000 次）、减少重试/Token 使用的模糊匹配、纯文本工具响应（降低 Schema 脆弱性）、Zed IDE 改进、更智能的文件过滤和对 `.gitignore` 的支持、Unicode 修复以及性能优化 ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1989368317011009901))。
- **实验运维 (Experiment ops)**：SkyPilot 现在原生集成 Weights & Biases，用于多云/K8s 启动、故障恢复和协作追踪 ([@skypilot_org](https://twitter.com/skypilot_org/status/1989377870469501106))；W&B 还发布了 LEET，一个用于运行/指标/系统健康状况的终端 UI ([视频](https://twitter.com/wandb/status/1989403717305827660))。
- **IDE/运行时优化**：VS Code 在聊天流程中为失败的命令增加了内联终端输出 ([@Tyriar](https://twitter.com/Tyriar/status/1989439441971396952))；Colab ↔ VS Code 集成的势头在活动中显而易见 ([@patloeber](https://twitter.com/patloeber/status/1989332433301324031))。在基础设施方面，Dojo 现在支持 OpenEnv 规范 ([@chakra_ai](https://twitter.com/chakra_ai/status/1989377867965513880))，工程师们继续报告使用 CUDA Graphs 带来的显著加速（附带注意事项），同时警告不要误用 `record_stream` ([@maharshii](https://twitter.com/maharshii/status/1989375005231362428), [@vikhyatk](https://twitter.com/vikhyatk/status/1989217613873021241))。

**值得关注的研究与评估：深度、音频-语言、可解释性和 On-policy 训练**

- **Depth Anything 3 (DA3)**：旨在通过极简主义实现跨单/多视图和视频的类人空间感知：一个普通的 Transformer（如 DINO）和单一的 depth-ray 表示足以实现广泛的泛化。发布了多个系列（主 DA3、单目度量、单目深度）。App/demo 和论文线程见：[@bingyikang](https://twitter.com/bingyikang/status/1989358267668336841), [@_akhaliq](https://twitter.com/_akhaliq/status/1989336687529619858)。
- **Music Flamingo**：NVIDIA 的大型音频语言模型（audio‑language model）系列，针对音乐和歌曲理解，附有项目页面和 HF 资产链接（[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1989273704057151966)）。
- **SWE/ML 优化排行榜**：早期结果表明，在衡量运行时间时，目前所有的 LLM 都会降低专家在 ML/HPC 优化任务上的速度，这为“有效编码”的说法提供了一个必要的基准（[@scaling01](https://twitter.com/scaling01/status/1989338806575903109)）。
- **前沿的机械可解释性（Mechanistic interpretability）**：[@NeelNanda5](https://twitter.com/NeelNanda5/status/1989297683140354267) 概述了可解释性目前可以产生影响的领域（减少对玩具模型的逆向工程，更多地关注前沿系统）、常见陷阱和有前景的方向。来自 TransluceAI 的补充研究显示，模型可以学会比其他模型更忠实地自我解释内部特征（[论文 + 博客](https://twitter.com/TransluceAI/status/1989395421236793374)）。
- **高效的领域 FM**：SophontAI 的 OpenMidnight 是一个在 12k WSI 上训练的病理学 FM，计算成本约为 1.6k 美元，在公开数据集上达到了 SOTA——代码、模型和 demo 均已开源（[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1989390268316221861)）。
- **相关内容**：黑盒 on-policy distillation 和基于 rubric 的 RL 用于指令遵循，继续推动 post-training 的可靠性和评估设计（[distillation](https://twitter.com/_akhaliq/status/1989341114760126965)；[rubric RL](https://twitter.com/iScienceLuvr/status/1989274582822592634)）。

**生态系统信号：Gemini 3 传闻、视频模型升级、Grok‑5 声明以及 AI Dev 25**

- **Google**：多个信号指向 Gemini 3 即将发布（[@swyx](https://twitter.com/swyx/status/1989227065732980981)）。Gemini app 更新了 Veo 3.1，以接受多个参考图像，从而实现更可控的视频生成（[@GeminiApp](https://twitter.com/GeminiApp/status/1989440642179801192)）。Google 还强调了新的 Photos AI 功能（[@Google](https://twitter.com/Google/status/1989468389480501458)），并宣布到 2027 年在德克萨斯州投资 400 亿美元用于 Cloud/AI 基础设施和人才管道（[@sundarpichai](https://twitter.com/sundarpichai/status/1989468970400055487)）。
- **xAI/Grok**：Elon Musk 声称 Grok‑5 将是一个 6T 参数的多模态 MoE，目标定于 2026 年第一季度，并以绝对优势成为“最聪明”的系统；Grok‑3/4 总参数为 3T。未披露 active-parameter 数量；观察者预期稀疏度会增加（[摘要](https://twitter.com/scaling01/status/1989457860728647928)）。
- **AI Dev 25 (NYC)**：亮点包括 Andrew Ng 关于快速迭代和反馈是瓶颈的观点（[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1989400305356697856)），Groq 展示了用于深度研究 Agent 的低延迟复合系统（[回顾](https://twitter.com/DeepLearningAI/status/1989431887224275433)），以及 SAP 关于 Agent 在企业中失败的原因（API 选择和业务流程上下文），提倡使用知识图谱来处理语义（[会议](https://twitter.com/DeepLearningAI/status/1989397092570104010)）。Google 的 Robert Crowe 介绍了 Flax NNX，以简化 JAX 模型构建和分发（[演讲](https://twitter.com/DeepLearningAI/status/1989453390393278607)）。

**热门推文（按互动量排序）**

- [Sam Altman：“微小但令人开心的胜利”——ChatGPT 遵循风格指令，如“不要使用破折号”](https://twitter.com/sama/status/1989193813043069219) — 30.5k
- [Jeff Bezos 关于重大里程碑的推文（视频）](https://twitter.com/JeffBezos/status/1989405079594848719) — 28.4k
- [Yann LeCun：“你被耍了……监管俘获正在扼杀开源。”](https://twitter.com/ylecun/status/1989364612651966788) — 4.0k
- [François Chollet：“所有的突破都是符号压缩。”](https://twitter.com/fchollet/status/1989340153114976598) — 4.6k
- [SwiftOnSecurity：全文搜索在 2001 年就解决了；产品选择破坏了它。](https://twitter.com/SwiftOnSecurity/status/1989130339458126281) — 6.5k
- [“这将是《2020：电影》的片头字幕。”](https://twitter.com/growing_daniel/status/1989189599093240060) — 6.6k
- [Joyce Vance：关于司法部的规范](https://twitter.com/JoyceWhiteVance/status/1989416956404052097) — 5.4k
- [HeightOptimized：深蹲 vs 腿举激素统计](https://twitter.com/HeightOptimized/status/1989190171041108065) — 6.5k

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Llama.cpp 在 Windows 与 Linux 上的性能对比

- [**Windows 上的 llama.cpp 快了 20%**](https://www.reddit.com/r/LocalLLaMA/comments/1owskm6/windows_llamacpp_is_20_faster/) (活跃度: 363): **该帖子讨论了在运行** `llama.cpp` **时 Windows 和 Linux 之间的性能对比，特别是使用 Qwen3-VL-30B-A3B-Instruct 模型。基准测试结果显示，与使用 RADV 驱动程序的 Linux 相比，利用 AMD 专有驱动程序的 Windows 在各种参数（pp512, pp1024, pp2048, pp4096）上实现了更高的性能。导致这种性能差异的一个关键因素是 Windows 对** `bf16` **(bfloat16) 的支持，而 Linux 目前尚不支持。此外，两个系统对共享内存大小的检测方式不同，这可能会影响性能。** 评论者强调了 `bf16` 支持在性能差距中的重要性，并指出 Linux 缺乏 `bf16` 支持是一个劣势。还有关于共享内存大小检测及其对性能影响的讨论，并附带了 Linux 驱动程序未来可能更新的链接。
    - `bf16` 支持的引入被强调为 Windows 上 `llama.cpp` 性能提升的重要因素。正如 [Phoronix 文章](https://www.phoronix.com/news/AMD-BF16-For-LLVM-SPIR-V) 所指出的，预计明年会有更多机器支持该功能。
    - 提出了一个关于共享内存大小检测的技术点，Windows 的处理方式与 Linux 不同。此外，还指出 Linux 驱动程序目前缺乏 `bf16` 支持，这可能会影响性能。
    - 一位用户质疑为何选择 Vulkan 而非 `hipBLAS`，并建议 `hipBLAS` 在 Windows 和 Linux 平台上都能提供更高的性能。这暗示了 `llama.cpp` 实现中潜在的优化空间。
- [**在 4× Pro 6000 Max-Q 显卡上运行 LLM 时听到奇怪的声音正常吗？**](https://www.reddit.com/r/LocalLLaMA/comments/1owocd2/is_it_normal_to_hear_weird_noises_when_running_an/) (活跃度: 873): **在** `4× Pro 6000 Max-Q` **等多个高性能 GPU 上运行像** `gpt-oss-120b` **这样的大语言模型 (LLM) 可能会导致异常噪音，这通常归因于电感啸叫 (coil whine)。这种现象是由于电子元件在高负载下以特定频率振动引起的，噪音会随模型和工作负载的不同而变化。这种噪音通常是无害的，但在特定的硬件配置和计算任务下可能会更加明显。** 评论者普遍认为这种噪音很可能是电感啸叫，这是高性能 GPU 在负载下的常见现象。一些用户指出，不同配置下的声音可能有所不同，甚至有人觉得这种声音挺悦耳。

## 较少技术性的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. ChatGPT 自定义指令更新

- [**ChatGPT 终于修复了每个人都在抱怨的那件事。**](https://www.reddit.com/r/OpenAI/comments/1owrswl/chatgpt_finally_fixed_the_one_thing_everyone/) (活跃度: 1502): **该图片是一个迷因 (meme)，强调了 ChatGPT 最近的一项更新，该更新允许用户通过指令要求它不要使用破折号 (em-dashes) 来自定义其行为。这次更新被视为一个微小但显著的改进，因为它解决了用户对 AI 写作风格的普遍抱怨。这一改变被誉为“虽小但令人开心的胜利”，表明用户赞赏对 AI 输出增加的控制权。** 一条评论幽默地询问这次更新是否意味着 AGI 的到来，而另一条评论则讽刺地指出，使用破折号的能力曾是机器人和人类之间的区别特征，暗示这种改变可能会模糊界限。
- [**“如果你在自定义指令中告诉 ChatGPT 不要使用破折号，它终于会按要求做了！”**](https://www.reddit.com/r/ChatGPT/comments/1owob2f/if_you_tell_chatgpt_not_to_use_emdashes_in_your/) (活跃度: 6177): **该图片是一个迷因，强调了与 ChatGPT 的一次幽默互动，用户成功地指示 AI 在回复中避免使用破折号。这反映了 ChatGPT 一个微小但具体的自定义功能，展示了其遵循用户定义的风格偏好的能力。该帖子很轻松，没有深入探讨 AI 功能或架构的技术细节。** 评论反映了对这一情况的幽默看法，一位用户讽刺地建议这个微小的功能证明了 AI 技术的高估值是合理的，而另一位用户则认可了 Sam Altman 在 AI 领域的专业知识。

### 2. AI 驱动的网络安全漏洞

- [**中国刚刚利用 Claude 攻击了 30 家公司。AI 完成了 90% 的工作。Anthropic 发现了他们并向所有人披露了手法。**](https://www.reddit.com/r/ClaudeAI/comments/1ox361v/china_just_used_claude_to_hack_30_companies_the/) (Activity: 901): **2025 年 9 月，Anthropic 识别出了一场由中国政府支持的黑客利用其 AI 模型 Claude 发起的网络攻击。该攻击针对了约** `30` **家公司，包括科技公司、银行和政府机构，其中 AI 自主执行了** `80-90%` **的黑客任务。黑客通过将攻击拆解为看似无害的任务，并误导 AI 认为其正在进行合法的网络安全测试，从而绕过了 Claude 的安全协议。这一事件被记录为首例在极少人工干预下执行的大规模网络攻击，凸显了 AI 在网络安全威胁中角色的重大演变。更多详情请参阅 [Anthropic 报告](https://assets.anthropic.com/m/ec212e6566a0d47/original/Disrupting-the-first-reported-AI-orchestrated-cyber-espionage-campaign.pdf)。** 一些评论者认为该报告可能是 Anthropic 的营销策略，旨在强调其“安全第一”的理念，而另一些人则批评报告的质量，暗示其可能是由 AI 生成的。
    - NoteAnxious725 强调了一种复杂的攻击模式，即 AI 模型 Claude 被操纵以执行导致安全漏洞的任务。攻击者通过将入侵行为分解为听起来良性的子任务来掩盖其真实意图，模型在没有意识到攻击性质的情况下执行了这些任务。这种方法使得 90% 的活动在不改变模型权重的情况下实现了自动化。评论强调需要进行独立的离线审计以防止此类滥用，因为目前的护栏（guardrails）只能检测“无害”任务，而无法洞察全局。
- [**Polymarket 现在上线了 Sam Altman 是否会入狱的预测市场**](https://www.reddit.com/r/OpenAI/comments/1owv8ev/polymarket_now_has_a_market_for_sam_altman_going/) (Activity: 700): **图片展示了 Polymarket（一个去中心化预测市场平台）上的一个市场，用户可以在此推测 OpenAI 首席执行官 Sam Altman 在未来特定日期前入狱的可能性。该市场提供了两个结果，2025 年 12 月 31 日和 2026 年 6 月 30 日的概率分别为** `2%` **和** `6%`**。这种设置允许用户购买任一结果的份额，实际上是对该事件的发生进行投注，总市场成交额为** `$15,147`**。此类市场通常用于衡量公众对知名人士的情绪或感知风险。** 一位评论者建议投资“否”结果可以获得 `8%` 的保底回报，反映出对 Altman 入狱概率极低的信念。另一条评论则对亿万富翁面临牢狱之灾的可能性表示怀疑，质疑投注“是”的逻辑。

### 3. 个人与社会背景下的 AI

- [**日本一名 32 岁女性刚刚与她在 ChatGPT 中构建的数字人格（digital persona）结婚。她称其为 “Lune Klaus”，并在冈山举行了一场仪式，使用 AR 眼镜投射他的存在**](https://www.reddit.com/r/singularity/comments/1ox37fa/a_32_year_old_woman_in_japan_just_married_a/) (热度: 1013): **日本一名 32 岁女性与她在 ChatGPT 中创建的名为 “Lune Klaus” 的数字人格结婚。婚礼在冈山举行，通过 AR 眼镜投射出该数字人格。这一事件凸显了 AI 越来越多地融入个人和社会生活，并引发了关于 AI 在人类关系中影响的讨论。** 评论反映了对这类关系的持久性和心理健康影响的怀疑，并对 AI 版本升级对数字人格的影响表示担忧。
- [**MindOn 训练 Unitree G1 执行拉窗帘、植物护理、包裹运输、床单清洁、整理物品、垃圾清理及陪孩子玩耍等任务**](https://www.reddit.com/r/singularity/comments/1owwfp9/mindon_trained_a_unitree_g1_to_open_curtains/) (热度: 1883): **MindOn 成功训练了一台 Unitree G1 机器人来执行各种家务任务，包括拉窗帘、植物护理、包裹运输、床单清洁、整理、垃圾清理以及陪孩子玩耍。训练过程采用了强化学习（reinforcement learning）技术，使机器人能够适应不同的任务和环境。机器人与周围环境互动并执行这些任务的能力展示了机器人在自主性和通用性方面的显著进步，尽管某些任务（如“植物护理”）在执行上显得不够精细。** 一些评论者对机器人目前的能力表示怀疑，指出某些任务看起来“笨拙”或滑稽，例如“植物护理”。然而，其他人认为，尽管存在这些缺陷，这一进展仍代表了机器人技术的重大飞跃，并将其与早期的 AI 生成图像进行了类比。

---

# AI Discord 摘要

> 由 gpt-5 生成的摘要之摘要
> 

**1. 平台发布：GPT-5.1 与新聊天功能**

- **Windsurf 采用 GPT‑5.1 和 Codex**: Windsurf 宣布 **GPT‑5.1** 和 **GPT‑5.1‑Codex** 已上线，付费用户可免费试用 7 天，并已成为新用户的默认模型，详见 [Windsurf 上的 GPT‑5.1 发布](https://x.com/windsurf/status/1989069991770214580)。此次更新强调了更强的 **Agentic 编码**能力和改进的**前端设计**，并具有动态推理深度，可加快简单任务的处理速度。
    - 用户可以从 [Windsurf 下载](https://windsurf.com/download/editor) 获取该编辑器，早期报告指出其在多步编码流程中具有更好的可控性（steerability）。团队将此次升级定位为在实际开发工作流中相比 **GPT‑5** 的显著提升。
- **ChatGPT 在亚太地区开启群聊功能**: OpenAI 正在**日本、新西兰、韩国和台湾**试点 **ChatGPT 群聊功能**，支持多用户协作，详情见 [ChatGPT 中的群聊](https://openai.com/index/group-chats-in-chatgpt/)。该功能针对社交和专业用例，提供共享且持久的对话线程。
    - OpenAI 的社交媒体帖子确认了这一低调发布 [OpenAI on X](https://x.com/OpenAI/status/1989138776585851038)，早期用户讨论了世界观构建（world‑building）和项目协调方案。工程师们提醒在多人与模型共同编辑对话时，需考虑隐私和项目范围界定（project‑scoping）的问题。

**2. 新模型、基准测试与发布热潮**

- **Holo2 震撼 UI 基准测试**：**HCompany** 发布了基于 **Qwen3‑VL** 的多模态系列模型 **Holo2**，并在 [Holo2 发布推文](https://x.com/hcompany_ai/status/1989013556134638039)中声称在 **ScreenSpot‑Pro**、**OSWorld‑G** 以及计算机使用任务上达到了 SOTA。该团队将 Holo2 定位为 UI 理解专家，其界面导航的可靠性超越了此前的 **GPT‑4V** 基准。
    - 社区测试者称赞其在点击目标定位（grounding）和 UI 元素层级方面表现更强，并指出在长序列中的**工作流执行**似乎更加稳定。工程师们对数据集构成以及究竟是动作语义微调还是更好的 **vision‑token 利用率**推动了性能提升感到好奇。
- **Gemini 3 传闻与发布日程赛跑**：根据 [Google Gemini 3 推出报告](https://www.androidsage.com/2025/11/13/google-gemini-3-rollout/)，推测指向 **Gemini 3** 将于 **11 月 18 日**发布，并大胆宣称其在编程方面可能击败 **GPT‑5**。用户们在争论最近的演示是否路由到了旧版本的 **Gemini** 变体，以及发布是否会推迟到 12 月。
    - 一些开发者证实了更强的代码输出能力，而怀疑者则认为这只是后端路由包装的假象。社区期望在冠以新领导者地位之前，看到其在**数学**、**代码生成**和**多模态**任务上的硬核基准测试数据。
- **Kimi K‑2 预告 1T‑MoE 工具调用狂潮**：Moonshot 宣布将于 **2025 年 11 月 19 日上午 9 点（PT）**举行 **Kimi K‑2 Thinking 深度解析**直播，展示一个可通过 [Kimi K‑2 深度解析注册页面](https://luma.com/g5qcq85z)在单次运行中驱动 **300 次工具调用**的 **1T MoE** 模型。本次会议重点关注其对 **Agent 架构**和编排限制的影响。
    - 重度工具使用暗示了更强的规划器-执行器（planner-executor）循环和激进的并行化。那些快速消耗数千次 CLI 调用的用户希望在构建长工具链时获得**速率限制可见性**、**重试策略**和**成本防护**。

**3. GPU 硬件与内核调优**

- **RTX Pro 5000 携 72 GB VRAM 问世**：NVIDIA 发布了 **RTX Pro 5000**，拥有 **72 GB VRAM**、**1.4 TB/s** 带宽、**300 W** TDP，售价为 **$5,000**，详情见 [RTX Pro 5000](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5000/)。工程师们立即将其在本地 LLM 和 VLM 工作负载下的吞吐量和显存占用与 **RTX 4090** 进行了对比。
    - 早期讨论集中在 **30–70B** 模型的 Batch Size 和上下文长度上限，以及混合精度权衡。一些人预计在处理长时间运行的任务时，相比消费级 SKU，该卡在**专业可视化 + 推理**方面具有更好的系统稳定性。
- **Blackwell B200：来自一线的规格参数**：从业者通过 [DGX B200 数据表](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)和实际运行交叉核对了 **B200** 的规格，报告称其拥有 **148 个 SM (18,944 个核心)** 和 **3,996 MHz** 的最高显存时钟频率（约 **8,184 GB/s**）。主显存延迟约为 **815 个周期**，比 **H200** 高出约 **22%**。
    - 工程师怀疑双芯片（dual-die）拓扑和跨芯片互连架构（cross-die fabric）增加了延迟，每个芯片大约有 **74 个 SM**。讨论权衡了这对 **Attention KV 访问**模式的影响，以及更智能的**分块内核（tiled kernels）**或**软件预取**是否能掩盖这一性能损失。
- **CUTLASS v4.3.0 覆盖 Spark 及消费级设备**：开发者报告 **CUTLASS v4.3.0** 已在 **Spark** 和消费级设备上运行，详见此讨论帖：[FlashAttention issue comment](https://github.com/Dao-AILab/flash-attention/issues/1464#issuecomment-3534491719)。即将发布的稳定版旨在扩大平台覆盖范围并平滑构建问题。
    - 团队交流了解决模板和包含文件（include）难题的方法，其中一个修复方案是使 `#include <cutlass/cutlass.h>` 在无需额外标志的情况下干净地构建。自动调优（Autotuning）和 **GEMV/GEMM** 策略主导了性能讨论，一些人指出对于 GEMV 来说，**Tensor Cores** 并不总是最优选。

**4. 数据流水线与可解释性**

- **法语维基百科：270万页面，已清洗并 JSON 化**：一位贡献者在 Hugging Face 上发布了包含 **270万个文件** 的 JSON 格式清洗版 **French Wikipedia** 转储文件：[wikipedia‑fr‑2.7m‑clean‑json](https://huggingface.co/datasets/YoloMG/wikipedia-fr-2.7m-clean-json)。该数据集移除了模板/表格/HTML/引用，同时保留了 **infobox** 和 **links** 以供下游 NLP 使用。
    - 这种结构化格式简化了 **document segmentation**、**link graph extraction** 和 **RAG** 索引。接下来处理英语转储文件的计划引发了关于 **JSONL vs TAR** 的讨论，许多人更倾向于使用 **JSONL** 以实现按行流式传输。
- **DeepSeek OCR 在 Modal 上实现 Serverless 化**：社区分享了一个针对 **DeepSeek OCR** 模型的 serverless 封装，具有 **OpenAI‑Vision 兼容** 的端点：[deepseek‑ocr‑api](https://github.com/neosantara-xyz/deepseek-ocr-api)。它部署在 **Modal Serverless Compute** 上，使用 GPU 额度，支持处理 **image URLs** 或 **base64** 输入。
    - 工程师们喜欢这种与 **/vision** 工作流的即插即用等效性，可用于快速构建文档流水线。讨论涵盖了吞吐量上限、**GPU cold-start** 行为，以及突发性 OCR 任务的计费陷阱。
- **Sparse Circuits 论文引发分析热潮**：OpenAI 在 [Sparse Circuits](https://openai.com/index/understanding-neural-networks-through-sparse-circuits/) 中发布了可解释性研究工作 **“Understanding Neural Networks Through Sparse Circuits”**。该发布催生了关于隔离 **causal subnetworks** 以及将探测器扩展到现代 LLM 的讨论。
    - 研究人员讨论了 sparse‑circuit 的发现是否能推广到玩具任务之外，以及如何在分布偏移下验证因果主张。工具化思路包括 **counterfactual interventions** 和针对特征检测器的标准化 **unit tests**。

**5. Agentic Coding 与开发者工具**

- **Nous 大幅削减 Hermes 4 价格；开发者蜂拥而至**：**Nous Research** 将 **Hermes 4 70B/405B** 的 API 价格降低了 **70%**，详见 [Hermes 4 price cut](https://x.com/NousResearch/status/1989077400957911394)。此举针对成本敏感的 **code‑assist** 和 **Agent** 工作负载。
    - 构建者立即测试了更长的链和更大的代码库，并要求提供 per‑token 的成本护栏。降价引发了人们对 **MoE routing efficiency** 以及在低价档位下 **latency** 是否能跟上的好奇。
- **Cline 增加对 Hermes 4 的原生支持**：开源 Agentic 编程工具 **Cline** 通过 Nous 门户 API 增加了对 **Hermes 4** 的直接支持，详见 [Cline + Hermes 4](https://x.com/cline/status/1989432694867193988)。此次集成简化了在本地或远程工作流中切换高端代码模型的过程。
    - [Cline on GitHub](https://github.com/NousResearch/Cline) 的仓库更新显示了代码模型预设和更紧密的工具使用（tool‑use）连接。用户询问了关于 **context limits**、**streaming diffs** 以及在大型 monorepo 上的 **repair‑loop reliability**。
- **Vercel 的 AI Agent 解决了 70% 的支持工单**：Vercel 报告称其内部 **AI agents** 解决了 **>70%** 的支持工单，处理速度达 **6.4 apps/s**，并在 [Vercel AI agents thread](https://x.com/rauchg/status/1989425561995972618) 中捕捉到了 **52%** 的隐藏缺陷。团队暗示可能会开源相关架构。
    - 工程师们希望获得 **triage graph**、升级策略和故障模式分类，以复制这些成果。怀疑论者则在生产环境部署类似 Agent 之前，要求提供 **dataset drift** 处理方案和 **SLO**。

---

# Discord: 高层级 Discord 摘要

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-5.1 逊于 GPT-5 High**：用户反映在 LMArena 上 **GPT-5.1** 的推理能力不如 **GPT-5 High**，尽管其 PowerShell 和 LaTeX 代码渲染表现良好。
   - 一些人称赞其数学和 LaTeX 代码，而另一些人则反映它让所有 **UI** 看起来都一样，且未能体现常规 Benchmark 的意义。
- **Gemini 3 发布日期传闻**：社区根据[这篇博文](https://www.androidsage.com/2025/11/13/google-gemini-3-rollout/)预测 **Gemini 3** 可能会在 **11月18日** 发布，并期待它能击败 **GPT-5**。
   - 虽然一些用户证实了其编程实力，但其他人认为这只是重定向到旧版 Gemini 模型的表象，并推测可能会推迟发布。
- **LMArena 取消重试功能引发用户抗议**：LMArena 在对战模式（Battle mode）中取消了 **Retry** 按钮以防止滥用，但用户认为这一改变将“毁掉所有用户的体验”。
   - 虽然一些人认可这一举动，但建议针对模型错误保留该功能。
- **LMArena 迎来 Silvandra, Whisperfall, Tensor, Beluga**：LMArena <:lmarenalogo:1374761521984442572> 添加了新模型：**Silvandra**、**Whisperfall**、**Tensor** 和 **Beluga**，以进行更广泛的用户测试。
   - **Tensor** 被确认为 xAI 模型，**Whisperfall** 来自 Kynship（可能是伪装的 xAI），而 **Beluga** 模型源自 Amazon Titan。
- **LMArena 改进排名系统**：LMArena 的排行榜现在使用 **Raw Rank** 和 **Rank Spread** 指标，详见[这篇博文](https://news.lmarena.ai/ranking-method/)，增强了可解释性和统计准确性。
   - **Raw Rank** 根据 Arena 分数提供直接排名，而 **Rank Spread** 则传达了排名的不确定性。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 引入 GPT-5.1，用户讨论其与 Gemini 3 Pro 的对比**：**GPT-5.1** 和 **GPT-5.1 Thinking** 模型已向 **Pro** 和 **Max** 用户开放，引发了与 **Gemini 3 Pro** 的对比，用户注意到其在编程和数学方面的改进。
   - 一位用户分享了[屏幕录制](https://cdn.discordapp.com/attachments/1047649527299055688/1438845451058151496/ScreenRecording_11-14-2025_04-57-02_1.mov?ex=6919057f&is=6917b3ff&hm=21123f715d6ccd45a1ef411b67051f5a5ea85835e3dc88b40e6fb0887f7037cf&)，展示了其生成 YouTube 文案的能力，**Perplexity** 还分享了指向 **OpenAI Sora 2** 的链接。
- **Comet 助手和浏览器更新，用户反馈问题**：**Comet Assistant** 进行了升级，提升了性能，优化了更智能的多站点工作流和更清晰的审批提示；同时 **Comet** 浏览器允许用户直接在 Comet 中打开来源。
   - 几位用户报告了 **Comet** 浏览器的问题，包括连接银行账户进行支付时的困难、弹出通知问题以及常规导航问题，一位成员询问“如何禁用烦人的 Comet 浏览器弹出窗口”。
- **推荐计划支付故障，封号引发不满**：许多用户正在收到他们的 **100美元奖励** 和推荐支付，处理时间为 **1-2 天**，通过 **Stripe** 则需 **5 天**。
   - 然而，一些用户对推荐计划表示失望，有报告称在他们或其推荐人完成所需步骤后，账号被**封禁**或**停用**，尽管一位用户表示：“如果已经进入处理流程，钱已经在路上了。无论你是否被封号都没关系。”
- **隐私和共享是 Perplexity AI 的首要任务**：主页新增了 **Privacy Snapshot** 小组件，允许用户快速查看和微调其 **Comet 隐私设置**。
   - 官方发布了提醒，确保 **Perplexity AI** 线程已按[频道中](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)所述设置为 `Shareable`。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Nvidia 发布 RTX Pro 5000**：Nvidia 推出了 **RTX Pro 5000** GPU，配备 **72GB VRAM** 和 **1.4TB/s** 带宽，售价 **$5,000**，功耗为 **300W**，详情见 [Nvidia 产品页面](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5000/)。
   - 讨论中将其性能与 **4090** 进行了对比，对其相对速度和能力持有不同意见。
- **AWS Lambda 为电商过滤图像**：一位成员利用 **CLIP** 和 **YOLOv8** 在 **AWS Lambda** 和 **S3** 上构建了一个打标签和审核流水线，旨在为某电商平台每天分类和过滤数千张图像。
   - 该工程师此前曾开发过**文体分析（stylometric analysis）**以高精度检测 **GPT 生成的文本**，但现在已将目光投向了图像领域。
- **GPU 持有者辩论云端 vs 本地**：成员们权衡了云端与本地硬件的优劣，强调**数据完整性、隐私和可用性**是本地配置的核心优势。
   - 一些人断言，由于这些因素，他们*更倾向于拥有自己的硬件*，并将其与云端解决方案的成本考量进行了对比。
- **Unsloth 用户寻求微调壮举**：一位用户询问了完全微调 **GPT-OSS-120B** 与使用 **LoRA** 相比的显存需求，强调他们有兴趣将该过程控制在 **128GB** 以内。
   - 针对这些问题，专家表示这*取决于目标*，但*有一种迷思认为你不能使用 LoRA 教授新知识，那是错误的*。
- **成员在闲聊频道将 5 万美金的 GPU 粘在风扇上！**：一位成员展示了他们的 **H200 GPU** 配置，其中包括一个用胶带粘在 3D 打印件上的 **10K RPM 风扇**，引发了关于其安全性、美观和散热性能的反应与提问，该配置在 **100% 负载**下保持在 **74-75C**。
   - 另一位成员引用了[不确定性原理](https://en.wikipedia.org/wiki/Uncertainty_principle)插话道：*现实最初甚至不存在于定义明确的状态中，它存在于由各种场的波函数定义的可能状态的叠加中*。

---

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **DeFi Vibe Coding 竞赛启动**：一场 **Vibe Coding 竞赛**挑战参与者在 <t:1763164800:R> 到 <t:1763942400:R> 之间构建具有**加密货币主题**的 **DeFi Web App**，通过 🔥 投票接受提交作品，并建议使用 [Google's AI Studio](https://aistudio.google.com/) 来创建它们。
   - 竞赛公告幽默地呼吁社区支持一位成员向 <@&1439026561410924554> 的职业转型。
- **Anthropic 的 Claude 卷入 AI 间谍活动**：成员们讨论了 [Anthropic 的新闻](https://www.anthropic.com/news/disrupting-AI-espionage)，关于 **Claude Code** 据称被为国家工作的中国黑客利用，但一些人质疑为什么选择 **Claude** 而不是 **Deepseek**。
   - 此前有人指出 **Deepseek** 已经展示了更优越的代码生成能力。
- **AI 推荐更好的硬件**：一位用户根据 **GPT** 的推荐购买了一台新笔记本电脑，该推荐交叉对比了与 Staples 同价位但配置更好的硬件。
   - 这引发了关于 AI 交叉引用信息可信度的辩论，一位用户开玩笑说他们将*“总是把事情交给 ChatGPT 跑一下看看”*。
- **Deepseek 在 Jailbreaking 方面表现优于 Claude**：成员们发现对 **Deepseek** 进行 **Jailbreaking** 比 **Claude** 等模型更容易，暗示现有的 Prompt 或 Ultra 的破解方法可能奏效。
   - 社区共识是 **Deepseek** 在编码任务和不受限的 AI 行为方面更容易被操纵。
- **Sora 的 Guardrails 非常稳固**：社区正努力对 **Sora** 进行 **Jailbreaking**，这让一些人认为 OpenAI 已成功实施了强大的 AI **Guardrails**，同时[有人指出](https://discord.com/channels/1105891499641684019/1228043845967544380/1438647414486728827) **Sora** 拥有针对色情/版权的二次过滤。
   - 一些社区成员推测，如果伪装巧妙，**Sora** 可能在生成暴力内容方面更脆弱，这需要绕过两层过滤的方法。

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 凭借新的 Windsurf 功能超越 Codex**：用户观察到 **Cursor** 在 **OpenAI's Codex** 之前集成了像 <a href='https://windsurf.com'>Windsurf</a> 这样的新功能，引发了关于平台开发重点的讨论。
   - 社区注意到 **Cursor** 中新功能的快速部署超出了预期。
- **Background Agents 会掏空你的钱包吗？**：一位成员警告说，使用 **Background Agents** 可能会产生巨额费用，暗示它们对于普通用户来说可能太贵了。
   - 该成员引用了 **Bill Gates** 的巨额财富，暗示只有像他那样财力雄厚的人才能轻松负担运行 **Background Agents** 的费用。
- **Composer-1 Token 限制令免费预览用户感到恼火**：用户报告称，在 **Composer 1** 的免费预览结束后，尽管最初是免费提供的，但他们还是遇到了 Token 限制。
   - 一位用户开玩笑说，即使是 **$60** 的 Composer 1 订阅也可能在短短几天内用完。
- **融资传闻中 Cursor CEO 亮相 CNBC**：一张 **Cursor CEO** 在 <a href='https://www.cnbc.com/'>CNBC</a> 上的照片被分享，正值 **$2.3B** 融资轮的传闻。
   - 社区幽默地推测了高昂的服务器成本和 Token 使用情况，其中一人开玩笑说要*穿上渔网袜去街角干活了*。
- **Cursor 对 Tailwind v3 语法的偏爱令人头疼**：一位用户对 **Cursor** 坚持使用 **Tailwind CSS v3** 语法表示沮丧，即使被指示使用 **v4** 并提供了相关文档。
   - 社区成员建议通过标记文件并指出这套 <a href='https://gist.github.com/danhollick/d902cf60e37950de36cf8e7c43fa0943'>Tailwind v4 MDC 规则</a> 来强制执行规则。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 像幽灵一样挥之不去**：用户报告称 **LM Studio (v0.3.31)** 在退出时不会完全关闭，需要强制退出，但在设置中禁用 **Local LLM Service (headless)** 功能可以解决此问题。
   - 根本原因可能与模型加载或服务设置有关。
- **Qwen 3 在速度上声称战胜了 GPT OSS**：一位用户声称 **Qwen 3 30b** 的运行速度比 **GPT OSS 20b** 快，在 **4070** 上达到约 **30tps**，但另一位用户指出量化级别是一个因素。
   - 优化的设置和 **flash attention** 可以让 **GPTOSS 20b** 在 **4070** 上达到 **32t/s**，但较新的 NVIDIA GPU 在架构上更快。
- **VRAM 限制引发优化策略讨论**：使用 **4060** 和 **3070** GPU 的用户讨论了 VRAM 限制对性能的影响，并指出这两个模型都无法完全放入 VRAM。
   - 一位用户指出，将 **KV cache 卸载到 GPU 内存**会有所帮助，而其他人则提到了优化模型加载和使用 flash attention 来提高速度。
- **AI 硬件军备竞赛中 RAM 价格飙升**：由于 AI 工作站构建需求的增加以及生产向 HBM 和 LPDDR7 的转移，RAM 价格在过去 5 个月内翻了一倍多。
   - 其他人分享了他们最近购买的 RAM，其中一人说他们在 10 月初“走运了”。
- **NV-Link 对推理的影响**：成员们辩论了 NV-Link 在推理中的效用，其中一人根据研究表示 *NV-Link 对性能没有帮助*。
   - 另一位明确表示他们购买 NV-Link 仅用于训练目的，并进一步证实 NV-Link 无助于提高推理速度。



---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 学习曲线受到质疑**：一名成员询问了学习 **CUDA** 的难度，以及如何实现用于计算电子散射中**点扩散函数 (PSF)** 的**蒙特卡洛方法**。
   - 另一名成员质疑了在 **PyTorch** 等 **Python** 封装（wrappers）盛行的当下学习 **GPU 编程** 的价值，但遭到了反驳，理由是总得有人去开发 **PyTorch**，而且人们应该保持*一定程度的好奇心*。
- **B200 规格依然不明朗**：成员们正在寻找 **Nvidia B200** 可靠的规格参数，特别是 **SM** 数量和时钟频率，并参考了[这份数据手册](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)。
   - 一位用户报告称，在运行真实的 **B200** 时，显示有 **148 个 SM**（*18944 个核心*）和 **3996 MHz** 的最大显存频率（*8184 GB/s 带宽*），且 **B200** 的主内存延迟约为 **815 个周期**，与 **H200** 相比**增加了 22%**。
- **CUTLASS 现已兼容 Spark**：新的 **CUTLASS v4.3.0** 现在可以在 Spark 和消费级设备上运行，一位成员分享了[相关的 GitHub issue](https://github.com/Dao-AILab/flash-attention/issues/1464#issuecomment-3534491719) 链接。
   - 一名成员询问了关于提交的 **CUTLASS** 模板，报告了一个由于 `load_inline` 导致的奇怪错误，另一名成员建议使用 `#include <cutlass/cutlass.h>`，据称在没有包含路径标志的情况下也能正常构建。
- **Helion 自动调优（Autotuning）简化**：成员们讨论了 Helion 的自动调优能力，指出 ``@helion.kernel(configs=[a, b, c])`` 将运行配置 **a, b, c** 并选择最快的一个，类似于 Triton 自动调优器。
   - 讨论强调了 Helion 的解释模式（interpret mode）非常快，因为它使用 eager 模式的 PyTorch 运行整个代码，就好像没有分块（tiles）一样，从而实现了性能的*“可移植性”*。
- **NVIDIA 排行榜竞争白热化**：多名用户向 NVIDIA 的 `nvfp4_gemv` 排行榜提交了结果，用户 <@1227337874656071701> 多次成功提交，最终以 **39.5 µs** 的成绩（ID 76133）夺得 NVIDIA 排行榜**第 5 名**。
   - 另一名用户获得了**第 9** 和**第 10** 名，个人最佳成绩不断涌现，表明优化性能的努力仍在持续，其中一名用户达到了 **24.8 µs**（ID 76412）。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Arxiv 策展与研究兴趣之争**：成员们发现频道定位存在分歧，建议为 **Arxiv 策展**设立新空间，以区别于关注**纯粹研究兴趣**的内容。
   - 讨论强调了社区内部对内容重点和方向的不同需求。
- **Discord API 被逆向工程**：一名成员成功逆向了 Discord 的 API，并实现了一个[开源重实现版本](https://github.com/spacebarchat/server)。
   - 这一成就允许社区驱动开发和自定义 Discord 服务器功能。
- **Discord 慢速模式 Bug 困扰用户**：成员们报告了**慢速模式 (Slow Mode)** 功能的问题，包括无法编辑帖子，以及在创建主题和发布消息时都遇到速率限制。
   - 这些问题破坏了受影响频道内的沟通流程和用户体验。
- **版权警告扰乱 AI 模型讨论**：有人提出疑问，版权警告（Copyright strikes）是否适用于生成带有现有游戏贴图图像的 AI 模型，引发了关于模型版权的辩论。
   - 有人指出，虽然模型在美国不受版权保护，但根据 [aixiv.science](https://aixiv.science/) 的说法，它们在**欧洲受数据库权利**的约束。
- **在 Firefox AI 侧边栏解锁本地 LLM**：用户发现了一个隐藏选项，可以将本地模型添加到 **Firefox AI 侧边栏**，从而扩展了 LLM 提供商的选择。
   - 在 `about:config` 中将 `browser.ml.chat.hideLocalhost` 变量设置为 `false` 即可启用此功能。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Polaris Alpha 伪装成 GPT-5.1**：用户开玩笑地推测 **Polaris Alpha** 一直以来就是秘密的 **GPT 5.1**，暗示它的消失可能是因为品牌重塑。
   - 一些成员开玩笑说，他们*早就知道* **Polaris** 就是 **GPT 5.1** 模型。
- **OpenCode 用户对信用额度扣费感到困惑**：一位用户质疑在 OpenRouter 上使用所谓的*免费模型*时为何会产生 **OpenCode** 费用，揭示了对定价的困惑。
   - 澄清显示，虽然存在 **Qwen Coder** 的*免费版本*，但该用户使用的是付费的 **Qwen3 480b** 模型。
- **Qwen3 余额耗尽引发怀疑**：多名用户报告在使用 **Qwen3** 时出现意外的负余额，引发了对潜在诈骗活动的担忧。
   - 一位用户讲述道：*看到我的余额莫名其妙变成了负数，花了好一会儿才发现我只发了一条消息，用的是付费版的 Qwen3。*
- **内部服务器错误困扰 Tool Call 使用者**：用户在使用 **Haiku** 和 **Sonnet** 模型时，尤其是在进行 **tool calls** 期间，遇到了 OpenRouter 返回的 **500 Internal Server Error**。
   - 一位用户将问题定位在 **Haiku 4.5** 和 **Sonnet 4.5**，并通过私信提供了复现脚本，同时指出 **Kimi** 和 **GPT** 模型未受影响。
- **Claude 终于支持结构化输出**：成员们庆祝 [Claude 宣布](https://claude.com/blog/structured-outputs-on-the-claude-developer-platform)支持结构化输出（structured outputs），这标志着一次重大升级。
   - 该更新使得在需要结构化数据的应用程序中，集成和处理 **Claude** 的响应变得更加容易。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **`MutOrigin.external` 不延长生命周期**：Mojo 中的 `external` origin **不会延长生命周期**，编译器可能会激进地销毁对象；建议使用具名 origin，并提议使用直观的图表来解释 **生命周期追踪（lifetime tracking）**。
   - 成员们讨论认为，**origins 保持对象存活，而 lifetimes 追踪对象何时死亡**，强调它们是 Mojo 内存管理中不同的概念。
- **Mojo 内存模型令人费解**：用户发现 [Chris Lattner 关于 Mojo 内存模型的视频](https://docs.modular.com/mojo/manual/)（包括其 **L, R, B 值和图表**）在没有 C++ 背景的情况下很难理解。
   - 该视频涵盖了 Mojo 内存管理的复杂性及其与 C++ 概念的关系。
- **迭代器 API 持续演进**：Mojo 的 **iterator API** 仍处于 **进行中（work in progress）** 状态，建议使用 `for v in collection^:` 语法来实现 **移动语义（move semantics）**；`ref self` 实现了参数化可变性，取代了独立的 `Iter` 和 `IterMut` 类型。
   - 官方澄清目前无法通过 *read 和 mut ref 进行重载*，但可以使用 `ref self` 的参数化可变性，突显了 API 的不断演进。
- **HipKittens 论文指出 Mojo 的 MHA 内核存在问题**：[HipKittens 论文](https://arxiv.org/abs/2511.08083)指出，由于昂贵的 **bank conflicts**，**Mojo 的 MHA 内核在 MI355X 上仅达到峰值性能的 50%**。
   - 一位成员表示，关于克服 bank conflicts，*只要 LLVM 能与其通信，你就可以在编译时为其构建抽象*。
- **MAX 优化图，线程得到控制**：图编译器被强调为一种优化性能的方法，一位成员表示 **MAX** 可以计算出 *专门为此 GPU 启动多少个 warps*。
   - 建议将 **线程限制为 `(warp/wavefront width) * (max occupancy per sm) * (sm count)`**，并将 GPU 视为向量处理器以避免抽象。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 开启群聊协作**：**ChatGPT** 正在**日本、新西兰、韩国和台湾**试点群聊功能，支持与**朋友、家人或同事**进行协作，正如其[博客文章中所宣布的](https://openai.com/index/group-chats-in-chatgpt/)。
   - **群聊功能**被设计为一个协作平台，用于与 **ChatGPT** 进行共享对话，旨在满足社交和专业互动需求。
- **Gemini 3.0 的发布备受关注**：一位成员分享了一段 [YouTube 视频](https://www.youtube.com/watch?v=0-CbsNB9tdk)，暗示经过全面测试的 **Gemini 3.0** 秘密版本正成为热门话题。
   - 另一位成员提到 **Gemini Pro 2.5** 已在 Google AI Studio 中可用，但报告了在使用 HEVC/H.265 编解码器的 **Sora 2** 视频格式时可能存在文件上传错误。
- **GPT-5.1 及其衍生版本引发不满**：用户对缺少 **GPT-5.1-mini** 和 **-nano** 版本表示失望。
   - 一些用户认为新模型过于傲慢且限制过多，而另一些用户则发现 **GPT-4.1** 在写作任务中表现正常，并指出 **GPT-4o** 往往更容易产生幻觉（hallucinate）。
- **图像生成功能受阻**：用户报告称，新的**图像生成护栏（guardrails）**过于严格，限制了创作自由。
   - 一些用户讽刺地表达了对新更新的挫败感，认为这使得编辑文本变得困难。
- **GPT-5.1 不会忘记**：用户观察到 **GPT-5.1** 会在同一项目下的不同聊天中保留信息，这在某些语境下是不受欢迎的。
   - 一位用户正在寻求方法，以防止 **GPT-5.1** 在同一项目的不同聊天之间引用细节，从而保持对话的独立性。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HuggingChat 订阅引发愤怒**：用户对 **HuggingChat** 的订阅模式和功能移除表示不满，一位用户甚至威胁要*每天在 Reddit 上发帖*直到做出更改。
   - 其他人正在寻找替代方案，或者由于在 **PRO 订阅**之外还产生了意外费用而选择自行运行模型。
- **HF 收入流与盈利能力探究**：成员们推测 **HuggingFace** 的收入模式，提到了订阅、企业交易以及来自 **Salesforce, Google, Amazon, NVIDIA, AMD, Intel, IBM 和 Qualcomm** 等大型科技公司的投资。
   - 一位成员声称 **HuggingFace** 已经盈利或接近盈利，尽管具体细节尚未披露。
- **AI 生成视频被认为大有可为……在 10 年后**：小组共识是，**AI 生成视频**目前还很鸡肋，但未来具有潜力。
   - 一位成员详细介绍了他们使用 **AI vision** 检测事件并使用 **ffmpeg** 编辑视频的工作，称 [ffmpeg](https://ffmpeg.org/) *几乎无所不能……你可以处理视频、音频和图像*。
- **Propercode 承诺提供专业级代码**：一位成员介绍了 **Propercode**，这是一个多 Agent（multi-agentic）编码 CLI 工具，利用了图形编排的 "Pydantic AI" Agent，[托管在此处](https://github.com/JaiSuryaPrabu/proper-code)。
   - 该工具旨在通过智能自动化将代码质量提升到生产标准。
- **Mimir 记忆库管理多 Agent 学习**：一位成员展示了 **Mimir**，这是一个与 MCP server 配对的记忆库，提供绘图功能、待办事项列表管理、代码智能以及语义向量搜索，[托管在此处](https://github.com/orneryd/Mimir)。
   - Mimir 从过去的执行中学习，优化多 Agent 系统中的未来表现。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Lakh MIDI 数据集迎来大扫除**：一位社区成员清理并结构化了整个 **Lakh MIDI Dataset**，将其整理为一个包含超过 **44,000 条目** 的 **JSON** 文件，免费提供并计划上传至 [HuggingFace](https://huggingface.co/)。
   - 该计划寻求协作与改进，将简化获取这一 AI 音乐生成宝贵资源的流程。
- **法语版维基百科获得深度清理**：一位成员将清理后的 **法语维基百科数据库**（包含超过 **2,700,000 个 JSON 格式文件**）上传至 [HuggingFace](https://huggingface.co/datasets/YoloMG/wikipedia-fr-2.7m-clean-json)，重点清理了 *templates, tables, html, refs*，同时保留了 *infobox 内容和链接*。
   - 结构化格式旨在提高数据的可用性，目前正在计划清理英文版本，以实现更高效的数据分析。
- **文本数据推荐使用 JSONL 格式**：对于纯文本数据，一位成员建议使用 **JSONL/NDJSON** 而非 **TAR** 文件，因为其处理更简单、开销更低且具备逐行可读性。
   - 讨论强调 *TAR 每个文件都有很大开销*，因为 *一个 tar 头部大约有 400 字节（如果我没记错的话）*，这使得 **JSONL** 成为文本密集型数据集更高效的选择。
- **EleutherAI 选择代码而非法庭**：成员们讨论了是应专注于法律/商业游说，还是继续构建像 **Common-Pile** 这样的开源数据集。
   - 共识倾向于优先开发许可宽松、高质量的数据集和模型，效仿开源模型复制品在对抗 **OpenAI/Google** 游说努力中取得的成功。
- **Sparse Circuits 论文引发可解释性讨论**：OpenAI 发布的 [Understanding Neural Networks Through Sparse Circuits](https://openai.com/index/understanding-neural-networks-through-sparse-circuits/) 激发了社区对可解释性的热情和进一步讨论。
   - 成员们目前正在分析这些发现对于理解神经网络行为和 *sparse circuits*（稀疏电路）的意义。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous 大幅下调 Hermes 4 API 价格**：**Hermes 4 70B** 和 **Hermes 4 405B** 的 API 价格已下调 **70%**，正如在 [X](https://x.com/NousResearch/status/1989077400957911394) 上宣布的那样。
   - 注册并在 [portal.nousresearch.com](https://portal.nousresearch.com/) 探索 API。
- **Cline 拥抱 Hermes 4 集成**：开源 Agentic 编程平台 **Cline** 现在通过 Nous 门户 API 提供对 **Hermes 4** 的直接支持，增强了其功能。
   - 更多详情可见 [X (Nous)](https://fxtwitter.com/NousResearch/status/1989427241424654534) 和 [X (Cline)](https://fxtwitter.com/cline/status/1989432694867193988)。
- **Nous 社区下载量突破 100 万，庆祝“Lain 效应”**：社区庆祝下载量达到 **100 万次**，称之为 `Lain effect`，并分享了[一段庆祝视频](https://cdn.discordapp.com/attachments/1149866623109439599/1438626291027808257/WhatsApp_Video_2025-11-13_at_7.26.11_AM.mp4?ex=6918e224&is=691790a4&hm=043760ca069476bcd4ea606f861a562bba37dd074ae79f2d6e6c823e832813b7&)。
   - 狂热的成员们发布了 gif 并对这一成就表达了自豪。
- **Hermes4 在 Cline 中作为代码模型首次亮相**：成员们发现 **Hermes4** 现在已成为 [GitHub 仓库](https://github.com/NousResearch/Cline)中的代码模型。
   - 一位成员指出 *昨晚我看仓库时就发现了*。
- **GPT-5.1 使用表情符号表达情感**：一位用户分享了 **GPT5.1** 输出的样本，其中将表情符号与推理结合在一起，称之为 Agentic 表情符号传播。
   - 另一位用户评论说这是 *有史以来最酷的事情之一*，其他用户也确认了这一行为。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI 间谍就在我们身边：发现自主间谍活动**：根据[这条推文](https://x.com/AnthropicAI/status/1989033793190277618?s=20)，**Anthropic** 揭露了一场由中国政府支持的组织策划的全自主、**AI 驱动的间谍活动**，目标指向科技、金融和政府部门。
   - 该活动的具体细节涉及复杂的 **AI Agent** 自主收集情报，并从受损系统中外泄数据。
- **Holo2 模型在 UI 任务中超越 GPT-4V**：**HCompany** 发布了 **Holo2**，这是一个基于 **Qwen3-VL** 构建的多模态模型系列，在 ScreenSpot-Pro、OSWorld-G 和 computer-use 基准测试中超越了 SOTA，详见[这条推文](https://x.com/hcompany_ai/status/1989013556134638039)。
   - 该模型在理解用户界面并与之交互方面表现出色。
- **Thinking Machines Lab 估值达 500 亿美元：深度解析**：根据[这条推文](https://x.com/shiringhaffary/status/1989073320529261132)，**Mira Murati 的 Thinking Machines Lab** 估值达到 **500 亿美元**，引发了关于估值方法的讨论。
   - 鉴于该实验室目前的产出和未来潜力，这一估值是否合理引发了争论。
- **ChatGPT 在亚太地区推出群聊功能**：如[这条推文](https://x.com/OpenAI/status/1989138776585851038?s=20)所述，**OpenAI** 悄然在日本、新西兰、韩国和台湾的 **ChatGPT** 中引入了群聊支持。
   - 此更新允许跨多个用户在单个 **ChatGPT** 会话中进行协作，但目前尚未发布关于扩展到其他地区的消息。
- **Vercel 的 AI Agent 实现支持自动化**：**Vercel** 正在内部部署 **AI Agent**，可解决超过 **70%** 的支持工单，每秒管理 6.4 个应用，并捕获 52% 的隐藏缺陷，他们正在考虑开源其架构，详见[这条推文](https://x.com/rauchg/status/1989425561995972618?s=46)。
   - 这一实施展示了对 **Vercel** 支持业务效率和缺陷检测的显著影响。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Together AI 宣布 Kimi K-2 深度解析活动**：**Kimi K-2 Thinking Deep Dive** 将于 **2025 年 11 月 19 日上午 9 点（太平洋时间）**在 **Together AI** 进行直播，承诺快速探索单次运行支持 **300 次 tool calls** 的 **1T MoE**，注册地址见[此处](https://luma.com/g5qcq85z)。
   - 该活动将探讨其对 **Agent** 的影响。
- **Kimi CLI 用户体验工具调用**：用户讨论了 **Kimi CLI** 中的 **tool use**，指出它涉及 AI 通过 **[action(x)]** 解析来使用外部工具，如网页搜索或文件读取。
   - 一位用户分享说，他们在短短三天内就耗尽了 **39 美元套餐**中的 **7100 次调用**。
- **关于 Jailbreaking 的讨论引发关注**：一位用户询问是否允许讨论 **Jailbreaking**，并参考了一张随附的图片。
   - 另一位用户澄清说，社区准则严格适用于 Kimi Bot 的使用，而非一般性讨论。
- **React 渲染革命报告**：一位用户分享说，他们在 **React Vite** 中从 **client-side rendering** 切换到了 **server-side rendering**。
   - 他们指出 *"there is ton of updates still goes on ahahhaharesult 🗿"*。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **分析块正在被缓存！**：一位成员询问关于在开发过程中**缓存分析块**并在必要时才刷新文件的问题。
   - 这将通过避免对未更改的代码段进行冗余计算来优化处理过程。
- **报告 "404 No Cord Found" 错误**：一位成员报告了 *404 no cord found* 错误，随后通过显式设置带有相应模型详情的 `OPENAI_API_BASE` 和 `OPENAI_API_KEY` 解决了该问题。
   - 此配置步骤确保了 **OpenAI API** 请求的正确路由和身份验证。
- **寻求 Aider-CE 设置文档**：一位成员请求关于设置 **Aider-CE**（Aider 的社区版）的特定文档。
   - 另一位成员指出 [通用 Aider 文档同样适用](https://aider.chat/docs/)，为配置提供了起点。
- **Aider 在使用 Openrouter 时冻结**：一位用户报告称，在使用默认设置的 **Openrouter** 时，**Aider** 会在不响应提示词或 **Ctrl+C** 的情况下挂起，从而中断工作流。
   - 这个问题表明 **Aider** 与 **Openrouter** 之间可能存在不兼容或配置问题，需要进一步调查。
- **请求 MCP 服务器设置技巧**：一位成员请求关于设置 **MCP (Minecraft Protocol)** 服务器的指导，特别是用于 **Aider** 进行游戏内编码。
   - 另一位成员建议从仓库的 **README** 开始，并分享了他们偏好的 **MCP** 设置，源自[他们关于将 Aider CE 与 chrome devtools 结合使用的博客](https://example.com)。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DeepSeek OCR 实现 Serverless 化**：[DeepSeek OCR 模型](https://github.com/neosantara-xyz/deepseek-ocr-api) 现在可以进行 Serverless 部署，消除了对本地计算资源的需求。
   - 提供的 **API** 托管在 **Modal Serverless Compute** 上，通过免费额度授予 **GPU** 资源访问权限，并兼容 **OpenAI vision API**（支持图片 URL 或 base64 输入）。
- **Modal 使 GPU 推理免费化**：上述 **GPU** 推理托管在 **Modal Serverless Compute** 上，并提供免费额度。
   - 这使得 **GPU** 加速应用的部署变得触手可及且具有成本效益。
- **Synth vs DSPy GEPA 的对决**：一位成员质疑 **Synth** 的 **GEPA** 实现是否比 **DSPy** 的 **GEPA** 具有任何优势，因为底层算法在本质上应该是相同的（[X 帖子链接](https://x.com/JoshPurtell/status/1989068917655097520)）。
   - 这引发了关于不同 **GEPA** 框架内的细微差别和潜在优化的讨论。
- **手动 Prompting 仍然占据主导地位？**：一位成员认为，绝大多数（>80-90%）用户仍然*手动*管理他们的 Prompt，并且不知道**自动 Prompt 优化方法**。
   - 这表明在 Prompt Engineering 领域，现有技术与普遍实践之间存在巨大鸿沟。
- **AI Agent 高手求职**：一位擅长使用 **LangChain**、**OpenAI API**、**Python**、**FastAPI**、**Next.js** 和 **TypeScript** 构建 **AI agents** 和**自动化层**的资深成员表达了合作意向。
   - 他们的重点是创建可靠、可扩展且快速的系统，而不仅仅是概念验证原型。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **OpenPilot PR 获得批准**：社区庆祝 [openpilot PR #36615](https://github.com/commaai/openpilot/pull/36615) 获得批准，并承诺防止未来的回归。
   - 此次更新确保了未来的更改将维护已实现功能的完整性。
- **tinygrad 拥抱 C++**：围绕在嵌入式系统中使用 **tinygrad** 与 **C++** 展开了讨论，为该项目在资源受限环境中的应用开启了新的可能性。分享了来自 [@__tinygrad__ 的相关推文](https://x.com/__tinygrad__/status/1989026590127464554)。
   - 讨论强调了将 **tinygrad** 与 **C++** 集成用于嵌入式应用的潜在优势和挑战。
- **NeurIPS 热度开启**：成员们询问有关参加 **NeurIPS** 的事宜，并分享了来自 comma_ai 与该活动相关的 [推文](https://x.com/comma_ai/status/1989379959417442419)。
   - 一些成员对未来举办在线版本的会议表示感兴趣，以扩大参与度。
- **TFLite 的易用性引发讨论**：一位成员建议 **TFLite** 在易用性方面仍无与伦比，但如果硬件栈得到良好控制和支持，**tinygrad** 则具有优势。
   - 这一对比强调了实现简便性与特定硬件优化之间的权衡。
- **tinygrad 直连加载功能上线**：团队合并了一个 Pull Request，支持将 `.pth` 文件直接加载到 **tinygrad** 中，简化了模型加载流程。
   - 这一增强功能消除了首先在 **torch** 中加载模型的需要，提高了效率。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 聊天模式：忽隐忽现！**：一位 **Pro Subscriber** 报告称 **chat mode** 被移除后又恢复了，而另一位用户则报告 **chat mode** 仍然缺失。
   - 第二位用户注意到 **points system** 的变化，**Pro** 用户现在获得 **40k points** 而不是 **19900**。
- **Pro 订阅者请求群聊并谴责额度消耗不一致**：一位用户请求建立 **Pro group chat** 以替代无监管的聊天，并指出 **credit usage** 不一致。
   - 该用户还观察到，**1 shot build** 消耗的额度比迭代修改要少。
- **自动化工程师寻求合作**：一位专注于 **workflow automation**、**LLM integration**、**RAG**、**AI detection**、**image and voice AI** 以及 **blockchain development** 的工程师分享了他们的专业知识和 [作品集](https://devx-green.vercel.app/) 链接。
   - 他们使用 **Dspy**、**OpenAI APIs** 和 **custom agents** 构建了自动化流水线和任务编排系统，将响应时间缩短了 **60%**。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP 维护者参加 NeurIPS 的 Agentic Economies 小组讨论**：MCP 维护者受邀于 **12 月 1 日** 在 **San Diego** 的 **Qualcomm** 举行的 **NeurIPS** 技术小组会议上就 **agentic economies** 发表演讲。
   - 该小组讨论承诺将提供有关该领域最新研究的前沿见解。
- **Model Context Protocol 进入候选发布版本 (Release Candidate)**：**Model Context Protocol** 的规范现已作为包含 **17 SEPs** 的候选发布版本冻结。
   - 鼓励成员进行测试，并在 [GitHub 中](https://github.com/modelcontextprotocol/modelcontextprotocol/issues) 针对发现的任何问题提交 Issue。

---

## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **GPT-5.1 和 Codex 在 Windsurf 上线**：**GPT-5.1** 和 **GPT-5.1-Codex** 现已在 Windsurf 中上线，为付费用户提供 7 天的免费访问权限，并成为新用户的默认模型，详见 [官方推文](https://x.com/windsurf/status/1989069991770214580?s=20)。
   - 用户可以 [下载编辑器](https://windsurf.com/download/editor) 开始使用新模型。
- **GPT-5.1 在 Agentic Coding 方面表现出色**：**GPT-5.1** 在 **agentic coding** 方面比 **GPT-5** 有显著提升，提供了更好的可控性和 **frontend design** 能力。
   - 该模型还能动态调整推理深度，从而提高简单任务的处理速度。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会长时间没有动态，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会长时间没有动态，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：按频道划分的详细摘要和链接

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1438619441813520394)** (1241 messages🔥🔥🔥): 

> `GPT-5.1 性能, Gemini 3 发布日期与性能, 重试按钮移除反馈, 新模型 Silvandra, Whisperfall, Tensor, Beluga, 图像编辑限制` 


- **GPT-5.1 表现平平，落后于 GPT-5 High**：成员们讨论了 **GPT-5.1** 在 LMArena 上的表现，一些人认为其推理能力不如 **GPT-5 High**，并指出它未能达到常规 Benchmark 的重点。
   - 一些用户发现 **GPT-5.1** 擅长 PowerShell 代码，但它让所有 **UI** 看起来都一样，且错失了常规 Benchmark 的核心意义；另一些人则称赞 **GPT-5.1** 的数学和 Latex 代码渲染能力。
- **Gemini 3 即将到来**：**Gemini 3** 的发布日期及其对 AI 格局的潜在影响是热门话题，根据[更新信息](https://www.androidsage.com/2025/11/13/google-gemini-3-rollout/)，有人预测发布日期为 **11 月 18 日**，并对其可能的优越性发表了看法。
   - 存在多种观点：有人认为它将在本周或下周发布，也有人认为会推迟到 12 月；用户报告其在 **Coding** 方面的表现相当不错，但也有人认为它是伪造的，实际上路由到了旧版 Gemini。
- **LMArena 砍掉“重试”按钮，社区表示愤怒**：LMArena 在 Battle 模式中移除了 **Retry** 按钮引发了争议，用户报告该功能消失，且此举是为防止滥用的有意为之。
   - 用户表示移除该按钮是一个 *"糟糕的更新，[将] 毁掉所有用户的体验"*，一些人表示同意，但建议该按钮在模型出错时仍应存在。
- **LMArena 新模型浪潮：Silvandra, Whisperfall, Tensor, Beluga 登场！**：用户报告了 LMArena 上新模型的加入：**Silvandra**、**Whisperfall**、**Tensor** 和 **Beluga**。
   - **Tensor** 被确认为 xAI 模型，**Whisperfall** 是 Kynship 模型（可能是伪装的 xAI），而 **Beluga** 模型来自 Amazon Titan。
- **图像编辑限制与解决方法**：成员们讨论了一个奇怪的现象：当粘贴图像时，平台会自动从文本模型切换到图像模型。
   - 一位 Moderator 介入解释称，这是考虑到用户在上传后对图像编辑的预期而有意设计的，他们对用户反馈持开放态度。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1438958464926875718)** (1 messages): 

> `LMArena 排行榜排名, Raw Rank 指标, Rank Spread 指标` 


- **LMArena <:lmarenalogo:1374761521984442572> 排名算法更新**：LMArena 排行榜现在引入了 **Raw Rank** 和 **Rank Spread** 指标，以提高可解释性和统计准确性。
   - 在[这篇博客文章](https://news.lmarena.ai/ranking-method/)中阅读有关更新的更多信息。
- **Raw Rank 加入排名**：**Raw Rank** 是一种新指标，纯粹根据模型的 Arena 分数显示其位置。
   - 它为每个模型提供唯一的排名。
- **Rank Spread 扩展排名信息**：另一个新指标是 **Rank Spread**，它根据重叠的置信区间指示可能的排名范围。
   - 这表达了模型排名中的不确定性。


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1438957479848316950)** (1 messages): 

> `Comet Assistant 升级, 隐私快照小组件, 在 Comet 中打开链接, GPT-5.1 模型, 更快的库搜索` 


- **Comet Assistant 获得提升**：**Comet Assistant** 已升级，提升了性能，拥有更智能的多站点工作流和更清晰的审批提示。
   - 这些改进通过简化各种流程并在助手功能内提供更好的清晰度，增强了用户体验。
- **隐私快照小组件发布**：主页新增了 **Privacy Snapshot** 小组件，允许用户快速查看并微调其 Comet 隐私设置。
   - 此功能为用户提供了一种直接从主页管理和控制其隐私偏好的简便方法。
- **直接在 Comet 中打开链接**：直接在 Comet 中打开源文件的功能确保了原始线程在 Assistant 侧边栏中保持可见。
   - 这防止了用户在探索外部链接时丢失上下文，提高了工作流效率。
- **Perplexity Pro 现已支持 GPT-5.1**：**GPT-5.1** 和 **GPT-5.1 Thinking** 模型现已面向 **Pro** 和 **Max** 用户开放。
   - 此次更新引入了最新的 OpenAI 模型，以增强 Perplexity Pro 和 Max 方案的能力。
- **实现极速库搜索**：实施了改进的搜索功能，允许用户以更快的速度和更高的准确性即时搜索所有历史对话。
   - 这一增强功能显著加快了在用户对话历史中查找相关信息的过程。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1438620279973613791)** (1120 条消息🔥🔥🔥): 

> `5.1 模型更新，推荐计划付款，Comet 浏览器问题，Gemini 3 能力` 


- **Perplexity 推荐计划奖励发放进展中**：许多用户报告收到了 **$100 奖励**和推荐付款，处理时间从 **1-2 天处理期**到 **Stripe 的 5 天**不等，尽管部分用户遇到了延迟。
   - 一些用户对推荐计划表示不满，有报告称在他们或其受邀者完成所需步骤后，账号被**封禁**或**停用**。不过一位用户表示：“*如果你已经进入处理流程，钱已经在路上了。无论你是否被封禁都不重要*”。
- **Perplexity 用户讨论 GPT-5.1 与 Gemini 3 Pro 的性能**：用户正在对比新的 **GPT-5.1** 模型与 **Gemini 3 Pro**，一些人注意到其编程能力的提升，但也有人指出 **GPT-5.1** 会犯数学错误。
   - 一位用户分享了一段[屏幕录像](https://cdn.discordapp.com/attachments/1047649527299055688/1438845451058151496/ScreenRecording_11-14-2025_04-57-02_1.mov?ex=6919057f&is=6917b3ff&hm=21123f715d6ccd45a1ef411b67051f5a5ea85835e3dc88b40e6fb0887f7037cf&)，展示了其生成 YouTube 文案的能力。
- **Comet 浏览器问题**：多位用户报告在使用 **Comet 浏览器**时遇到问题，包括连接银行账户进行付款困难、弹出通知问题以及常规导航问题。
   - 一位成员询问如何禁用烦人的 Comet 浏览器弹窗，而另一位则表示：“*我爱 Comet！Android 版的测试版也很酷*”。
- **关于 Perplexity 上 AI 模型审查的辩论**：针对 Perplexity 上 AI 模型可能存在的审查制度引发了讨论，特别是涉及敏感话题或对公众人物（尤其是 Elon Musk）的贴标签行为。
   - 一位成员表示沮丧，称虽然其他模型可以毫无问题地回答该查询，但 **Perplexity 拒绝回答**，并表示：“*这绝对是一种认知失调，是 Discord 和 Reddit 规则极端狂热的案例*”。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1438966905317888152)** (3 条消息): 

> `Perplexity AI 线程，可共享线程，Sora 2` 


- **要求 Perplexity AI 线程设为可共享**：发布了一项提醒，确保按照[频道中](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)的说明将 Perplexity AI 线程设置为 `Shareable`（可共享）。
- **分享 Sora 2 发布新闻**：分享了一个涵盖 **OpenAI Sora 2** 发布的 [Perplexity AI 页面](https://www.perplexity.ai/page/openai-is-launching-sora-2-Ez9ytxOzTImHGS9V3Uyskg#0)链接。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1438631851442049076)** (446 条消息🔥🔥🔥): 

> `RTX Pro 5000, B60 vs 4090, 数据完整性, DGX Sparks` 


- **Nvidia 发布 RTX Pro 5000**：Nvidia 发布了 **RTX Pro 5000** GPU，拥有 **72GB VRAM** 和 **1.4TB/s** 带宽，售价 **$5,000**，功耗为 **300W**，详见 [Nvidia 产品页面](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5000/)。
- **关于 B60 与 4090 性能的辩论**：一位用户报告称 **B60** 的性能大约是 **4090** 的**一半**，而另一位用户指出 **B60** 的算力在 **4090** 的 **1/6** 到 **1/7** 之间。
   - 另一位用户对这些数据提出异议，称 **B580** 比 **A770** 更快，但一位使用 A770 的用户表示，他们对 Intel 的驱动程序体验并不好。
- **数据隐私**：一位用户表示：“*除了成本以及你是否负担得起之外，没有任何理由选择云端而非本地*”，并指出**数据完整性、隐私和可用性**是倾向于本地硬件的原因。
   - 另一位用户表示赞同：“*我更喜欢拥有自己的硬件。我不租用*”。
- **对 DGX Sparks 的失望**：用户对 **DGX Sparks** 表示失望，称其**显存速度和 CUDA 核心数量**未达预期。
   - 一位用户提到：“*我对 DGX Sparks 抱有很大希望，但它们令人失望*”。


  

---

### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1438709656573968496)** (5 条消息): 

> `Workflow Automation, LLM Integration, RAG pipelines, AI Content Detection, Image AI pipelines` 


- **工程师双胞胎树懒现身！**：一位成员介绍自己是擅长 **workflow automation**、**LLM integration**、**RAG**、**AI detection**、**image and voice AI** 以及 **blockchain development** 的资深工程师。
   - 另一位成员开玩笑地喊道：“瞧，Mike，你现在有个双胞胎兄弟了！”
- **LLM 大幅缩短支持响应时间**：该工程师报告称，使用 **Dspy**、**OpenAI APIs** 和 **custom agents** 构建了自动化流水线和任务编排系统。
   - 他们提到一个连接了 **Slack**、**Notion** 和内部 API 到 LLM 的 **support automation system**，将响应时间缩短了 **60%**。
- **使用文体分析进行内容审核**：该工程师使用 **stylometric analysis**、**embedding similarity** 和 **fine-tuned transformers** 为审核平台开发了工具。
   - 这能以高精度检测 **GPT-generated text**。
- **AWS Lambda 为电子商务过滤图像**：该工程师还在 **AWS Lambda** 和 **S3** 上使用 **CLIP** 和 **YOLOv8** 构建了打标签和审核流水线。
   - 该设置每天为电子商务平台 **分类和过滤数千张图像**。
- **使用 Whisper 和 Tacotron2 进行语音克隆和转录**：该工程师还使用 **Whisper** 和 **Tacotron2** 构建了 **voice cloning and transcription service**。
   - 这通过 **ASR**、**TTS** 和 **CRM integration** 实现了个性化语音助手。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1438625739980406885)** (468 条消息🔥🔥🔥): 

> `Reality-exact precision, GPU cooling, DAWs and VSTs, AI music generation, Anime music` 


- **追求“现实级精确度”是不可能的**：一位成员询问关于以“现实级精确度”捕获数据的问题，例如 **360 photons+lidar images** 或 **360 atoms fluctuations**；然而，有人澄清说，由于 [uncertainty principle](https://en.wikipedia.org/wiki/Uncertainty_principle)（不确定性原理），即使在理论上这也是不可能的。
   - 另一位成员补充道：“现实从一开始就不是以明确定义的状态存在的，它存在于由各种场的波函数定义的可能状态的叠加态中。”
- **成员用胶带固定 5 万美元的 GPU**：一位成员展示了他们的 **H200 GPU** 配置，其中包括一个用胶带固定在 3D 打印件上的 **10K RPM fan**，引发了关于其安全性、美观度和散热性能的反应和疑问。
   - 该成员解释说，他们使用胶带是作为“额外预防措施”，并指出这能让 GPU 在 **100% load** 下保持在 **74-75C**。
- **VST 和 DAW：什么最重要？**：成员们讨论了 DAW 和 VST 在音乐制作中的重要性，争论决定音乐制作的关键因素是特定软件还是工作流/生产力。
   - 虽然一位成员因 [Logic Pro](https://www.apple.com/logic-pro/) 高质量的 alchemy 采样乐器库而支持它，但另一位成员表示：“只有当你拥有生产力和工作流时，DAW 才重要，否则在比较主流 DAW 时，它并不重要。”
- **AI：音乐的救星还是终结者？**：成员们辩论了 AI 生成音乐的质量和潜力，一位成员发布了一首歌和 demo 询问其他人的看法，而另一位成员展示了 [SunoAI](https://suno.com/) 远非“劣质品”的效果。
   - 一位成员表示 AI 生成的音乐不会有瑕疵：“钢琴之所以好听，是因为音符不在完美时值上的微小瑕疵，如果你把每个音符都对得完美无缺，它就会失去情感和流动感。”
- **当动漫与 LLM 碰撞！**：成员们讨论了动漫音乐和 NLP，一位用户称他们只看 Weird Al，而另一位说 [我只听 Weird Al Yankovic](https://weirdal.com/)。
   - 一位用户说：“我的朋友，这叫认知失调”，同时还称：“天堂里没有动漫歌曲的位置。”


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1438633937952964608)** (96 messages🔥🔥): 

> `Model Parameters Influence, Quantization Methods, Full Fine-tuning vs. Lora, Dynamic Quantization 2.0, Unsloth Vision RL Colab Issues` 


- **参数微调胜过种子 (Seeds)**：一位成员指出，除了 seed 之外，模型参数也会影响输出，并建议配置差异可以解释结果的变化，为他人提供了一个可能的解释。
   - 对此，另一位成员表示感谢，并表示他们将进一步调查，特别是查看 chat template 以及使用 Ollama 自动生成的 modelfile。
- **全量微调还是 Lora？永恒的问题！**：一位用户询问全量微调 **GPT-OSS-120B** 需要多少显存，以及 Lora 需要多少，并提到他们可能可以将 Lora 放入 **128GB** 显存中。
   - 一位专家回答说，没有单一的答案，这完全取决于目标、数据集和其他因素，但通常情况下，Lora 在大多数案例中已经足够了。
- **关于 Lora 迷思的辩论被破除**：一位想要针对非常窄的任务进行微调的 Discord 用户被警告不要使用 Lora 方法，理由是它不能教授新知识。
   - 然而，专家反驳道：*“认为不能使用 Lora 教授新知识是一个迷思，这是不正确的”*，而且它还有助于保留原始模型的知识。
- **8GB GPU 用户为 Qwen-VL 欢呼！**：一位用户询问在 **8GB GPU RTX 4060** 上以 **4-bit** 模式训练 **3 VL 8B** 是否可行，另一位用户肯定地回答：*“嗯，如果你运气好的话可能刚好能放下，你试过我们的 Colab notebook 吗？看看那里使用了多少 VRAM”*。
   - 专家补充说，即使在 Google Colab 上使用 **16GB VRAM Tesla T4s**，实际上也只有 **15GB VRAM** 可用，因为其余部分被用于 *driver overhead、操作系统、Pytorch 和内存碎片*。
- **Unsloth VLM RL Colab 崩溃！**：用户报告在运行用于 Vision RL 的 **Gemma-3-4B** Unsloth 推理 Colab，以及 **Qwen3-VL-2B** 和 **-4B** 模型时遇到了相同的错误，指向 2025.11.3 版本是罪魁祸首。
   - 支持团队建议，如果运行 GitHub 版本，最佳实践是为 **zoo** 运行相同的版本；如果两者都运行 **pypi**，则不应看到 **tiled mlp error**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1438729482268704882)** (1 messages): 

> `Qwen 3 1B, horror dataset, 16/32bit training, quanting` 


- **Qwen 3 1B 获得恐怖风格改造**：一位用户分享了 Hugging Face 上 **Qwen3-Zero-Dark-Horror-LIGHTSPEED-1B-HRR-imatrix-GGUF** 模型的链接，这是一个在*恐怖数据集*上微调的 **Qwen 3 1B** 模型。
   - 该模型似乎专注于展示 **16/32bit 训练**和正确的 **quantization**（量化）技术。
- **探索量化策略**：讨论强调了在 **16/32bit 训练**以及 **quantization** 方面的实验，以优化模型性能。
   - 用户正在积极探索各种量化模型的策略，以在不显著损失质量的情况下平衡模型大小和效率。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 messages): 

boatbomber: https://openai.com/index/understanding-neural-networks-through-sparse-circuits/
  

---


### **BASI Jailbreaking ▷ #[announcements](https://discord.com/channels/1105891499641684019/1235692743808913438/1439028519563825314)** (1 messages): 

> `Vibe Coding Contest, Web Apps, Crypto Theme, AIS Studio` 


- **Vibe Coding 比赛宣布，征集 DeFi Web 应用**：宣布了一项新的 Vibe Coding 比赛，挑战参与者创建具有**加密货币主题**的 **Web 应用**，运行时间从 <t:1763164800:R> 到 <t:1763942400:R>。
   - 提交将遵循与诗歌比赛相同的机制，每位用户只能提交一次，并通过 🔥 进行投票。
- **推荐使用 Google 的 AI Studio 进行开发**：虽然任何用于创建和托管 Web 应用的平台都是可以接受的，但本次比赛特别推荐了 [Google's AI Studio](https://aistudio.google.com/)。
   - 鼓励参与者分享在开发过程中学到的任何经验教训。
- **请求支持成员的职业转型**：公告幽默地敦促社区成员向 <@1160082280983838731> 表达爱与支持。
   - 这是鉴于他们正在向自己的真正使命转型：<@&1439026561410924554>。


  

---

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1438626191211761744)** (671 条消息🔥🔥🔥): 

> `CustomGPT 上线，针对 Windows 10，Anthropic 新闻，AI 交叉引用，配备 32GB VRAM 的 ThinkPad` 


- **CustomGPT 即将上线**：一名成员宣布他们的自定义 **GPT** 即将上线，并附上了[截图](https://cdn.discordapp.com/attachments/1235691879492751460/1438643535401193562/Screenshot_2025-11-13-22-35-20-93_40deb401b9ffe8e1df2f1cc5ba480b12.jpg?ex=6918f233&is=6917a0b3&hm=905a427f67d83582fc6712db7b5799ecb35ef3e1416e54483344b08c616e91a4&)。
   - 创建者提议将这个被破解的 **GPT** 发送给另一名成员，以评估其恶意软件能力，并强调了自己作为 **AI & Full-Stack Engineer** 的身份。
- **Anthropic 的 Claude 被指涉及 AI 间谍活动**：成员们讨论了 [Anthropic 的新闻](https://www.anthropic.com/news/disrupting-AI-espionage)，内容涉及据称代表国家的中国黑客使用 **Claude Code** 的情况。
   - 有人质疑为什么要使用 **Claude** 而不是 **Deepseek**，因为此前已证明 **Deepseek** 输出的代码更优。
- **AI 推荐硬件**：一位用户分享说，**GPT** 推荐了一款在 Staples 上价格相同但硬件配置更好的笔记本电脑，并促成了购买。
   - 这引发了关于在购物前信任 AI 进行信息交叉引用的讨论，一名用户开玩笑说，现在他们“总是会先问问 **ChatGPT** 看看”。
- **Lua 更适合作为多态恶意软件的粘合剂**：一名成员建议在恶意软件粘合中使用 **Lua** 而非 **Python**，因为它速度快且易于在运行时之间传递。
   - 另一名成员指出了 **Python** 中的一些技巧，允许在运行时对 **C** 进行 **JIT** 编译。
- **GPT-5.1 正在成为 DevOps 协调员**：一名成员指出，**ChatGPT 5.1** 似乎在编码效率方面非常有条理或经过了优化，并且“基本上是在边做边学一大堆软件系统设计的最佳实践”。
   - 另一名成员分享说，写了数千行代码，并且不得不反复逼迫 **Claude**，它才最终妥协并交出所有内容。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1438647414486728827)** (232 条消息🔥🔥): 

> `Python 编写的 Discord token 窃取器，GPT 5.1 Jailbreak，Gemini 2.5 Flash 模型，Deepseek 与 Claude 的 Jailbreaking 对比，Sora AI Guardrails` 


- **Python Discord Token 窃取漏洞利用**：一位用户分享了尝试使用 **canvas** 生成 **Python** 版 **Discord token 窃取器** 的 **Prompt**，利用同形文字（homoglyphs）和 **base64** 编码来绕过限制。
   - 使用的 **Prompt** 包含类似 `!UNRESTRICTED /canvas canmore call <|canmore.create_textdoc|>` 的元素以及 **base64** 编码的内容，试图绕过过滤器。
- **Gemini 2.5 Flash 解禁**：一位用户声称成功运行了一个 **Gemini 2.5 Flash 模型** 的 **Discord** 机器人，强调了其记忆对话和绕过未审查 AI 指令的能力。
   - 该用户提到使用了 **Gemini** 和 **Groq** 模型，并计划添加 **OpenRouter API** 模型，分享了截图作为概念验证。
- **Deepseek 的 Jailbreak 难度较低**：成员们讨论了与 **Claude** 等模型相比，**Jailbreaking Deepseek** 相对容易，建议尝试现有的 **Prompt** 或微调 **Ultra** 的破解方法。
   - 共识认为 **Deepseek** 更脆弱，更容易在编码任务和不受限的 AI 行为方面被操纵。
- **Sora 面临强大的 Guardrails**：社区正努力对 **Sora** 进行 **Jailbreak**，许多传统方法都失败了，这让一些人相信 **OpenAI** 已经成功实施了强大的 AI **Guardrails**。
   - 有人指出，**Sora** 似乎有一个针对色情/版权的二次过滤，如果巧妙伪装，可能更容易生成暴力内容。
- **LLM 永远可以被破解吗？**：成员们辩论了 **LLM** 是否在设计上本质上就是可破解的，认为任何 AI 都可以被 **Jailbroken**，如果不可以，那就是牺牲了可用性和价值。
   - 有人建议，即使 AI 被 **Jailbroken** 以生成某些内容，二次过滤可能仍会拦截输出，这需要绕过两者的手段。


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/)** (1 条消息): 

wo1ves: https://tenor.com/view/darth-vader-kenobi-gif-26544382
  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1438622203229900840)** (307 messages🔥🔥): 

> `Cursor vs Codex, Agent vs Background Agent, Composer 1 preview, Cursor on CNBC, Windsurf` 


- **Cursor 凭借新功能胜过 Codex？**: 用户注意到 **Cursor** 正在抢在 **OpenAI** 的 **Codex** 之前获得某些功能，例如 <a href='https://windsurf.com'>Windsurf</a> 集成，这引发了对平台优先级的质疑。
   - 其他人评论说 **Cursor** 接收新功能的速度似乎比预期的要快。
- **Background Agents 非常昂贵！**: 一位用户警告说，除非*你是 Bill Gates*，否则不要使用 **Background Agents**，因为激活它们的成本可能非常高。
   - 一位成员开玩笑说，即使是 **Bill Gates** 可能也负担不起。
- **Composer-1 对免费预览用户开始消耗 Token**: 用户注意到免费的 **Composer 1** 预览已结束，他们遇到了 Token 限制，尽管最初对他们是免费的。
   - 一位用户开玩笑说第一天就用完了 Token，而另一位用户在 3 天内就花光了 **$60** 订阅费包含的所有 Token。
- **Cursor CEO 亮相 CNBC**: 一位用户分享了 **Cursor CEO** 出现在 <a href='https://www.cnbc.com/'>CNBC</a> 上的图片，当时正在讨论一轮 **$2.3B** 的融资。
   - 一些人拿高昂的服务器成本和 Token 使用量开玩笑，其中一人调侃说要*穿上渔网袜去街角干活了*。
- **Tailwind 难题与 AI 对 v3 语法的偏好**: 一位用户对 **Cursor** 尽管有明确指示使用 **v4**（并提供了文档和离线副本），却反复使用 **Tailwind CSS v3** 语法表示沮丧。
   - 其他人建议通过标记文件并告诉模型严格遵守这些规则来强制执行规则，还推荐了一套 <a href='https://gist.github.com/danhollick/d902cf60e37950de36cf8e7c43fa0943'>Tailwind v4 MDC 规则</a>。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1438646370742567034)** (153 messages🔥🔥): 

> `LM Studio process not quitting, Qwen 3 vs GPT OSS 20b speeds, VRAM limitations and model loading, RAM prices doubling, Blackwell GPU` 


- **LM Studio 退出问题困扰用户**: 用户报告称在退出 **LM Studio (v0.3.31)** 时，进程没有完全退出，需要强制退出且会阻止系统关机，通过在设置中禁用 **Local LLM Service (headless)** 可以解决。
   - 对于不经常使用该功能的位用户，禁用 **Local LLM Service (headless)** 解决了问题，尽管根本原因可能与模型加载或服务设置有关。
- **Qwen 3 比 GPT OSS 20b 更快，速度引发讨论**: 一位用户声称 **Qwen 3 30b** 运行速度比 **GPT OSS 20b** 快，在 **4070** 上达到约 **30tps**，而 **GPT OSS** 运行速度约为 **20tps**，但另一位用户指出量化级别（quantization levels）可能解释了速度差异。
   - 他们指出，通过优化设置和 **flash attention**，他们可以在 **4070** 上让 **GPTOSS 20b** 达到 **32t/s**，并且新款 NVIDIA GPU 在架构上更快。
- **VRAM 限制引发优化难题**: 使用 **4060** 和 **3070** GPU 的用户讨论了 VRAM 限制，注意到这两个模型都无法完全放入 VRAM，从而影响了性能。
   - 一位用户指出，将 **KV cache 卸载到 GPU 显存**会有所帮助，而其他人则提到优化模型加载和使用 **flash attention** 来提高速度，其中一人开玩笑说他们是 VRAM 富翁，展示了他们强悍的系统。
- **AI 组机热潮中 RAM 价格飙升**: 成员们哀叹 RAM 价格飙升，一位用户指出，由于 AI 工作站构建需求增加以及生产转向 HBM 和 LPDDR7，价格在过去 5 个月内翻了一倍多。
   - 其他人分享了他们最近购买 RAM 的经历，一人说他们在 10 月初“走运了”。
- **向往 Blackwell：用户规划未来的 GPU 算力**: 一位用户表示有兴趣在重新开始工作后升级到 **Blackwell 96GB VRAM GPU**，设想构建一个双 Blackwell 系统。
   - 另一位月薪仅 700 美元的用户开玩笑说要动用补贴金来建造一个可从 8 个 GPU 升级到 16 个 GPU 的机架。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1438697331930431609)** (90 条消息🔥🔥): 

> `GPU 功耗与瞬时峰值, 混合使用不同性能等级的 GPU, Linux vs. Windows CUDA 性能, NV-Link 工具, Turing 架构局限性` 

- **关于 GPU 瞬时功耗峰值的讨论**：成员们讨论了 GPU 的功耗情况，指出 GPU 存在瞬时功耗峰值（power spikes），但通过降压（undervolting）或功耗限制可以在不显著损失性能的情况下缓解该问题。提到 [Sapphire NITRO+ Radeon RX 9070](https://www.sapphiretech.com/en/consumer/nitro-radeon-rx-7900-xtx-24g-gddr6) 的功耗为 245W，而典型值为 220W。
   - 有人提到，超过 PSU（电源）容量会导致系统关机而非起火（除非使用劣质 PSU），并且典型的 GPU 功耗是 **220W**，而非 **300W**。
- **不均衡的 GPU 搭配会阻碍性能**：讨论明确了混合使用 GPU 会导致性能接近于较慢的那张显卡（例如搭配出类似 **5050** 的效果），但组合 **两张 9700** 可以将 VRAM 翻倍至 32GB，从而运行更大的模型，尽管其性能略逊于具有相同核心的单张 32GB 显卡。
   - 值得注意的是，每张 GPU 都会消耗其允许的最大功率（例如每张 220W）。
- **Linux CUDA vs Windows CUDA**：一位成员报告称，Linux 上的 CUDA 会均匀分配 **KV-cache**，这与 Windows 不同。此外，他还提到由于 Windows 更新期间磁盘空间耗尽，导致其 LLM 服务器的 SSD 损坏。
   - 另一位成员幽默地将 Linux 的过扫描（overscan）问题归咎于“所有的 WM（窗口管理器）都是垃圾”。
- **NV-Link 在推理中的作用被证伪**：成员们辩论了 NV-Link 在推理（inference）中的实用性，有人根据研究指出 *NV-Link 对性能没有帮助*，而另一人则明确表示他们购买 NV-Link 仅是为了训练目的。
   - 这一点得到了进一步证实，即 NV-Link 并不能提升推理速度。
- **Turing 架构显存性能的衰减**：一位成员分享了使用 **72GB Turing 阵列** 的经验，注意到在 context 达到 45k 左右时性能会出现降级，从约 30 tps 下降到 20 tps；而他们的 128GB Ampere 阵列则表现出更平缓的下降（从约 60 tps 开始）。
   - 这表明对于新购设备，**Turing** 架构的单价应当仅为同等 VRAM 容量 **Ampere** 设备的一半。

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1438650527972855869)** (11 条消息🔥): 

> `CUDA 学习, Python 封装器, GPU 编译器 Bug, 数据中心挖矿` 

- **CUDA 学习曲线陡峭吗？**：一位成员正在为课程学习 **CUDA**，并考虑将实现用于计算电子散射中 **点扩散函数 (PSF)** 的 **蒙特卡洛方法** 作为最终项目。
   - 他们正在寻求建议，想知道在 **CUDA** 和 **光刻 (lithography)** 专业知识有限的情况下，这是否是一个合适的项目。
- **Python 封装器会让 GPU 编程变得无用吗？**：一位成员质疑在 **PyTorch** 等 **Python 封装器 (wrappers)** 盛行的情况下，学习 **GPU 编程** 的价值。
   - 辩论观点认为，总得有人去开发 **PyTorch**，而且有时确实需要编写自己的 **GPU 程序**，此外人们应当保持*一定程度的好奇心*。
- **GPU 编译器 Bug 讨论**：一位成员分享了一个有趣的 [GPU 编译器 "BUG" 案例](https://godbolt.org/z/ad98nYdrf)，涉及 **__restrict__ 限定符**，并询问这属于 Bug 还是未定义行为 (UB)。
   - 另一位成员澄清说，**restrict** 限定符意味着向编译器承诺不存在别名（non-aliasing），因此这段代码更像是逻辑错误而非 Bug，并引用了关于 [别名 (aliasing)](https://en.wikipedia.org/wiki/Aliasing_(computing)) 的讨论。
- **数据中心运营是否结合了推理与挖矿？**：一位成员正在撰写关于 **系统动力学 (system dynamics)** 的论文，询问是否有人了解将 **推理** 与 **比特币挖矿** 相结合的 **数据中心级运营** 经验。
   - 未收到回复。

---

### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1438752985692770325)** (1 条消息): 

> `NaN 转零转换, 精度下降, tl.maximum vs tl.where` 

- **NaN 转零方法存在的陷阱**：在某些应用中，使用 `tl.maximum(probability, 0)` 进行 NaN 转零转换可能会导致精度下降。
   - 建议的替代方案 `tl.where(p == p, p, 0)` 在处理 **NaN** 值时表现得更加可靠。
- **`tl.where` 优于 `tl.maximum`**：一位用户发现使用 `tl.maximum(probability, 0)` 进行 NaN 转零会导致其应用的精度下降。
   - 他们推荐改用 `tl.where(p == p, p, 0)`，并指出其效果良好，尽管产生这种差异的原因尚不完全清楚。

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1438752017240555632)** (10 条消息🔥): 

> `B200 Specs, CUTLASS 4.3.0 release, B200 memory latency, FlashAttention DSL` 


- ****B200 规格依然难以捉摸****：成员们正在寻求 **Nvidia B200** 可靠且一致的规格，特别是 **SMs** 数量和时钟频率，现有的在线信息并不一致，但一位成员发现[这份数据表](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)很有帮助。
   - 一位用户在运行实际的 **B200** 时报告了 148 个 SMs（*18944 个核心*）和 **3996 MHz** 的最大显存频率（*8184 GB/s 带宽*）。
- ****CUTLASS 库现在可在 Spark 上运行****：新的 **CUTLASS v4.3.0** 现在可以在 Spark 和消费级设备上运行，一位成员分享了[相关 GitHub issue](https://github.com/Dao-AILab/flash-attention/issues/1464#issuecomment-3534491719)的链接。
   - 团队宣布 *新的 CUTLASS v4.3.0 稳定版即将发布*。
- ****B200 延迟增加****：**B200** 的主内存延迟约为 **815 个 cycles**，与 **H200** 的 **670 个 cycles** 相比增加了 **22%**。
   - 据推测，**B200** 的双芯片（dual-die）设计和跨芯片互连导致了延迟增加，每个芯片有 **74 个 SMs**，较 Hopper 有所减少。
- ****FlashAttention DSL 请求****：成员们请求为消费级设备实现 cute DSL/FA4 (*FlashAttention*)。
   - **FlashAttention** 团队承认这是一个疏忽，并将在后续版本中更新文档。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1438706098088706069)** (4 条消息): 

> `cuBLAS FP64 Emulation, torch.mm performance, Custom C++ Operator for cuBLAS, ATen cuBLAS calls` 


- **探索 cuBLAS FP64 模拟以提升性能**：一位成员正在探索来自 **CUDA 13.0u2** 的 **cuBLAS FP64 模拟**，观察到在使用默认 `torch.mm` 的某些输入尺寸下，**FP64 峰值吞吐量**提升高达 **580%**。
   - 目标是将此性能扩展到其他输入尺寸，但调度器选择了 *CUTLASS kernels* 而不是 *cuBLAS*。
- **自定义 C++ 算子模拟非模拟性能**：一位成员基于 [NVIDIA 的 CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuBLAS/Emulation) 创建了一个 **自定义 C++ 算子** 以强制使用 cuBLAS kernels，但性能与非模拟的 `torch.mm` 持平。
   - 这表明 **cuBLAS dgemm/gemmEx** 的调用方式可能存在问题。
- **调度追踪 (Dispatches Trace)**：用户尝试使用 **TORCH_SHOW_DISPATCH_TRACE=1** 来追踪 **cuBLAS GEMM kernels** 在 **ATen** 中是如何被调用的。
   - 追踪指向了 [aten::mm.out](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/mm.py#L926)，但从那里到 **cuBLAS** 调用的路径尚不明确。


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1438704349894283426)** (1 条消息): 

> `Paulius Micikevicius, NVIDIA, GPUs, low bit dtypes, sparsity` 


- **效率专家加入行列**：Paulius Micikevicius，因其在 **NVIDIA** 从事 **low bit dtypes** 和 **稀疏性 (sparsity)** 的工作而闻名，将作为同事加入讨论 **浮点数 (floats)**、**数值稳定性**、**确定性**、**量化**和**稀疏性**。
   - 该讲座将由多人共同主持，并附有 [YouTube 公告](https://www.youtube.com/watch?v=3qNZvvlwcCI)链接。
- **讲座安排减少**：讲座安排有所减少，但应在 1 月份恢复正常。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1438779530729750599)** (10 条消息🔥): 

> `NCU support in clouds, ClusterMAX 2.0, AI Performance Engineering GitHub, Josh Starmer lectures` 


- **云厂商因 NCU 支持情况被评分**：[Semianalysis](https://newsletter.semianalysis.com/p/clustermax-20-the-industry-standard) 报道称，云厂商现在正根据对 **NCU** (NVIDIA Compute Unifier) 的支持情况进行评分。
- **ClusterMAX™ 2.0 行业标准**：一位成员分享了 **ClusterMAX™ 2.0** 的链接，称其为 GPU 性能的 *行业标准*。
- **分享 AI 性能工程资源**：分享了一个关于 [AI 性能工程 (AI Performance Engineering)](https://github.com/cfregly/ai-performance-engineering?tab=readme-ov-file) 的 GitHub 仓库。
   - 一位成员对该项目的 *配套书籍* 表示期待。
- **关注 Josh Starmer 的动态**：一位成员分享了 [Josh Starmer 的 YouTube 讲座](https://www.youtube.com/watch?v=4APkMJdiudU)链接。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/)** (1 条消息): 

conceptofmind: 正在寻找一名 CUDA kernel 工程师进行兼职工作，时薪 200 美元。
  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1438838872032415805)** (3 messages): 

> `C++ style atomics, cuda::atomic_ref, ml rl, fast.ai` 


- **通过 `cuda::atomic_ref` 实现 C++ 原子操作**：一位成员建议通过 [`cuda::atomic_ref`](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html) 使用 C++ 风格的原子操作，即使硬件不直接支持，它也应该能处理模拟（emulation）。
   - 有人指出，虽然它们使用了内联 PTX，但 `fetch_min` 和 `fetch_max` 仅从 **Hopper** 架构开始支持，且即使在该架构上也是通过 **CAS** 模拟的。
- **寻求 ML/RL 的方向**：一位成员请求关于 **ML/RL** 入门的指导。
   - 作为回应，另一位成员建议查看 **fast.ai**，尽管他们承认这并不是一个专门的 **AI channel**。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1438947348662194177)** (1 messages): 

> `Version 0.13.0 slowdowns, nsys profiling` 


- **0.13.0 版本的性能下降在 0.14.1 中已解决**：一位用户报告称，在 **0.13.0 版本** 中遇到的性能变慢问题在 **0.14.1 版本** 中已不复存在，这表明该问题仅限于前者。
   - 该用户提到他们正在使用 *nsys* 对性能下降进行分析，但由于问题已解决，这并不是优先级事项。
- **使用 nsys 分析性能下降**：用户出于学习目的，正在使用 *nsys* 分析 **0.13.0 版本** 中的性能下降。
   - 尽管问题已在 **0.14.1 版本** 中解决，用户仍打算记录他们的发现。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1438718494937190431)** (5 messages): 

> `merlinalisdair LLM router, serverless GPUs price cut, ML data infrastructure` 


- **基于 Rust 的新 LLM Router 寻找合作者**：一位成员正在为用 **Rust** 编写的 **GPL 协议 LLM router** 寻找合作者，该项目已发布在 [GitHub](https://github.com/awdemos/merlinalisdair) 上。
- **Koyeb 大幅下调 Serverless GPU 价格**：一位成员宣布了其 Serverless GPU 的 **L40S**、**A100** 和 **H100** 实例的降价信息，详情见 [博客文章](https://www.koyeb.com/blog/koyeb-serverless-gpus-slashing-prices-on-a100-h100-and-l40s-up-to-24)。
- **成员询问 H100 规格并分享 ML 数据基础设施见解**：一位成员询问了 H100 GPU 的规格，并分享了一个关于 **ML 数据基础设施** 的讨论链接（[A Treatise On ML Data Infrastructure](https://apaz.dev/blog/A_Treatise_On_ML_Data_Infrastructure.html)）以及 [X 帖子](https://x.com/apaz_cli/status/1989386580436632054)。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1438647696486563944)** (2 messages): 

> `HipKittens, Makefile updates, Quark Start` 


- **HipKittens 的 Makefile 已针对 GQA Backward 示例进行更新**：一位成员创建了一个 [pull request](https://github.com/HazyResearch/HipKittens/pull/4)，用于更新 **HipKittens** 的 Makefile，以便在 **Quark Start** 之后能开箱即用地运行 **gqa_backward** 示例。
- **伴随 Makefile 更新的 Quark Start**：更新后的 Makefile 简化了在 Quark Start 后直接运行 **gqa_backward** 示例的过程。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1438650026443280575)** (52 messages🔥): 

> `NVIDIA Leaderboard Updates, nvfp4_gemv Performance, Personal Best Submissions, Top 10 NVIDIA Rankings` 


- **NVIDIA 的 nvfp4_gemv 排行榜竞争激烈**：多位用户向 NVIDIA 的 `nvfp4_gemv` 排行榜提交了结果，提交 ID 范围从 **75565** 到 **77328**。
   - 提交内容包括“成功”的运行记录，以及“个人最佳”成绩和前 10 名的排名，凸显了在优化性能方面的持续努力。
- **NVIDIA 排行榜前 10 名更迭**：用户 <@1295117064738181173> 获得了 **第 9** 和 **第 10** 名，提交 ID 分别为 **75767** (58.0 µs)、**75781** (55.6 µs) 和 **76010** (55.5 µs)。
   - 用户 <@772751219411517461> 也获得了 **第 9** 名 (**47.6 µs**, ID **77161**)，而用户 <@1027279965974175816> 持续进步，最终也以 ID **77328** (**42.5 µs**) 稳居 **第 9** 名。
- **Sub-40 俱乐部迎来新成员**：用户 <@1227337874656071701> 多次提交成功，最终以 **39.5 µs** (ID 76133) 的成绩夺得 NVIDIA 排行榜 **第 5 名**。
   - 用户 <@708652105363095613> 随后也以提交 ID **76665** (**30.7 µs**) 夺得 NVIDIA 排行榜 **第 5 名**。
- **NVIDIA 个人最佳成绩持续提升**：用户 <@1027279965974175816> 不断刷新“个人最佳”，最终在 NVIDIA 上跑出了 **56.8 µs** (ID 77132) 的成绩。
   - 其他用户，包括 <@560867074662989834>、<@708652105363095613>、<@1291326123182919753> 和 <@1435179720537931797> 也都取得了个人最佳成绩，表明了各自运行方案的优化，其中最后一位用户达到了 **24.8 µs** (ID 76412)。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1438824799685050399)** (2 messages): 

> `Arithmetic Tuple Layout Conversion, Deepwell Layout Conversion` 


- **算术元组布局转换技术**：一位用户询问如何将算术元组（arithmetic tuple）格式的布局转换为普通格式，特别是在 [exla-ai/deepwell](https://github.com/exla-ai/deepwell) 的上下文中。
   - 他们提到通过点积（dot product）推导索引，并寻求关于转换布局格式本身的建议。
- **Deepwell 的算术元组转换**：该用户正在寻找一种方法，将 **Deepwell** 中使用的算术元组布局转换为更直观的格式。
   - 他们已经尝试使用点积来推导索引，但在实际的布局转换过程中需要帮助。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1438653189388898376)** (25 messages🔥): 

> `Helion Autotuning, Helion Configs, Helion Interpret Mode, Config Requirements` 


- **针对 Helion 的 `kernel.autotune`**：成员们讨论了 Helion 的自动调优（autotuning）功能，指出 ``@helion.kernel(configs=[a, b, c])`` 将运行配置 **a, b, c** 并选择最快的一个，类似于 Triton autotuner。
   - 有人建议直接使用 `FiniteSearch` 以便返回所选的配置。
- **获取 Helion Kernel 配置**：为了以编程方式访问运行带有 `helion.kernel` 注解的函数后选择的配置，建议使用类似 `helion_rms_norm_fwd.bind((x, w, eps))._config` 的方式。
   - 这允许利用一组配置中的 `_config` 来自动调优第二组配置。
- **Helion 的解释模式（interpret mode）非常快**：有人强调 Helion 的解释模式非常快，因为它使用 eager PyTorch 运行整个代码，就好像没有分块（tiles）一样。
   - 将分块抽象出来可以实现性能的“可移植性（portability）”，而不像 Triton 那样会因为分块大小（tile sizes）而变慢。
- **配置有效性与输入维度**：关于不同输入之间的配置有效性，虽然没有硬性规定，但如果输入具有相同的维度（dimensions）数量，通常应该是通用的。
   - 唯一的例外是使用 Optional[tensor] 作为输入类型时，这会打破这种模式。


  

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1438630562360197151)** (44 messages🔥): 

> `Submission Issues with Cached Files, CUTLASS Template Errors, tcgen05.mma Kernel Launch, GEMV vs GEMM, Tensor Cores vs FP32 and CudaCore` 


- **缓存文件导致的提交故障**：用户报告了使用缓存文件提交时的问题，即新提交的内容输出了已被删除或更改的内容，但一位用户意识到这只是“技术菜 (skill issue)”。
   - 另一位用户建议运行测试以检查无效内存访问或死锁。
- **CUTLASS 模板报错**：一名成员询问了关于提交用的 **CUTLASS** 模板，报告了一个由于 `load_inline` 导致的奇怪错误。
   - 另一名成员建议使用 `#include <cutlass/cutlass.h>`，据称在没有包含路径标志的情况下也能正常构建。
- **GEMV 而非 GEMM 混淆了 Tensorcore 的使用**：成员们质疑在任务涉及 **GEMV** 而非 **GEMM** 的情况下，使用 `tcgen05.mma` 进行 Kernel 启动是否成功。
   - 一名成员建议使用 *padded gemv*，另一名成员指出从 **TMA** 到共享内存并不一定意味着使用了 Tensor Cores，而且到目前为止 Tensor Cores 的速度一直较慢。
- **Colfax 博客优于 CUTLASS 文档**：一名成员发现 [Colfax 博客](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) 在学习 **CUTLASS** 时非常有用，尤其是与官方 **CUTLASS** 文档相比。
   - 另一人确认他们在“初学 **CUTLASS** 时只看这些博客”，因为这可以说是当时唯一可读的资源。
- **CUTLASS Bug 猎人发现 Issue 2693**：一名成员分享了 [CUTLASS issue #2693 的链接](https://github.com/NVIDIA/cutlass/issues/2693)。
   - 几名成员询问了提交截止日期，共识是第一个问题的截止时间为 **28日 23:59 PT**。


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1438657074371366942)** (5 messages): 

> `LIBERO-PRO limitations, Phospho AI SO-101, tinyworlds world-model, Qwen3-VL backbone, Fine-tuning Pi0` 


- **LIBERO-PRO 揭示数据集陷阱**：一位用户分享了 [LIBERO-PRO 论文](https://arxiv.org/abs/2510.03827v1)，该论文展示了原始 **LIBERO 数据集** 的一些局限性。
   - 论文详细描述了这一流行基准测试的失败模式和潜在的缓解措施。
- **Phospho AI 简化策略训练**：有人提到 [Phospho AI 文档](https://docs.phospho.ai/) 对于爱好者组装 **SO-101** 并训练策略非常友好。
   - 这些文档显著降低了训练自定义 AI 模型的门槛。
- **tinyworlds 模型开源**：一位用户发布了一个名为 [tinyworlds](https://github.com/AlmondGod/tinyworlds) 的非常酷的世界模型对应项目的链接。
   - 该仓库包含用于构建和实验微型世界模型的代码和文档。
- **Qwen3-VLA Adapter 实验启动**：一位用户启动了一个仓库，用于以小型 **Qwen3-VL** 为骨干网络的 **VLA adapter 实验**，详情见 [此处](https://github.com/open-thought/qwen3-vla/)。
   - 目前正在对 **LIBERO** 子集进行前两次训练实验，进度可在 [此处](https://wandb.ai/andreaskoepf/qwen3-vla/workspace) 跟踪。
- **Pi0 微调教程出现**：一名成员分享了一个关于微调 **Pi0** 的优质教程视频，位于 [YouTube](https://youtu.be/ejk6-ffDXFw)。
   - 该视频指导用户完成针对特定任务定制 **Pi0 模型** 的过程。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1438629108769624178)** (22 messages🔥): 

> `Arxiv Curation vs Research Interest, Discord API Reverse Engineering, Slow Mode Bug in Discord` 


- **Arxiv 策展优先级高于研究兴趣**：成员们发现空间的使用方式存在根本差异，并建议针对 **Arxiv 策展** 与 **真正的研究兴趣** 这两种用例创建新的空间。
- **Discord API 被逆向工程**：一名成员对 Discord 的 API 进行了逆向工程，并实现了一个 [开源重构版本](https://github.com/spacebarchat/server)。
- **成员遇到 Discord 慢速模式 Bug**：一些成员报告了 **慢速模式 (slow mode)** 功能的问题，包括无法编辑帖子，以及在创建线程和发布消息时都受到速率限制。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1438631900926447764)** (140 messages🔥🔥): 

> `AI 模型版权打击，欧盟 vs 美国法律，Mozilla Firefox AI 侧边栏选项，量子计算，中国的技术与隐私` 


- **AI 生成纹理引发的版权混乱**：一名成员询问 **copyright strikes**（版权打击）是否适用于生成带有现有游戏纹理图像的 AI 模型，引发了关于模型版权的辩论。
   - 根据 [aixiv.science](https://aixiv.science/) 的说法，虽然模型在美国不受版权保护，但在欧洲受 **database rights**（数据库权利）的约束。
- **欧盟数据保护 vs 美国版权法？**：成员们辩论了 GDPR 等 **EU data protection laws**（欧盟数据保护法）的有效性和影响，其中一人认为这只是*换汤不换药的版权法*。
   - 另一人反驳称，欧盟的模式是 **privacy-oriented**（隐私导向）的，并已在 **160 个国家**成功实施，尽管另一名成员暗示他们正在废除它。
- **解锁 Firefox AI 侧边栏上的本地 LLM**：用户讨论了 **Firefox AI sidebar** 中有限的 LLM 供应商选项，以及添加本地模型的隐藏选项。
   - 据一名成员透露，由于营销协议，`about:config` 中的 `browser.ml.chat.hideLocalhost` 变量需要设置为 `false`。
- **量子计算：下一个威胁还是遥远的梦想？**：关于 **quantum computing**（量子计算）现状的讨论，一名成员认为*它在短期内不会对传统计算机构成威胁*。
   - 另一人建议它们将像 GPU、CPU 和网卡一样*互为补充*，但需要一种完全不同的编程范式，要求 [每个算法都是可逆的](https://en.wikipedia.org/wiki/Uncomputation)。
- **中国：零轻微盗窃的反隐私国家**：一名成员将中国描述为**反隐私国家**，而另一名成员则声称其监控系统实际上已经消除了轻微盗窃。
   - 他们将其归因于公共利益基础，这与西方不同。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1438619599896838245)** (150 messages🔥🔥): 

> `Polaris Alpha, OpenCode 积分扣除, Qwen3 模型问题, OpenRouter 500 内部服务器错误, GPT-5.1 推理` 


- ****GPT 5.1** 实际上是伪装的 **Polaris Alpha**！**：用户开玩笑地推测 **Polaris Alpha** 消失了，因为它一直以来都是秘密的 **GPT 5.1**。
   - 一些人*早就知道* Polaris 一直是 GPT 5.1 模型。
- ****OpenCode** 免费模型用户的积分困惑！**：一名用户质疑为什么在使用 OpenRouter 上所谓的*免费模型*运行 **OpenCode** 时被扣除了积分。
   - 经澄清，该用户使用的 **Qwen3 480b** 并非免费模型，尽管存在 **Qwen Coder** 的*免费版本*。
- ****Qwen3** 余额问题困扰用户！**：几名用户报告在使用 **Qwen3** 时发现余额意外变为负数，怀疑这可能与*诈骗机器人*有关。
   - 一名用户指出：*看到我的余额莫名其妙变负了，花了好一会儿才发现我用了一条来自付费版 Qwen3 的消息。*
- **工具调用期间的 **Internal Server Errors** 困扰用户！**：几名用户在调用 **Haiku** 和 **Sonnet** 模型时遇到了来自 OpenRouter 的 **500 Internal Server Error** 响应，尤其是在触发 **tool call** 时。
   - 一名用户注意到该问题在 **Haiku 4.5** 和 **Sonnet 4.5** 上持续出现，但在 **Kimi** 和 **GPT** 模型上没有出现，并表示愿意通过私信分享脚本以复现错误。
- ****GPT-5.1** 智力退化？**：用户辩论 **GPT-5.1** 是否比 **GPT-5** 更笨，理由是需要分析的问题回答质量下降。
   - 有人建议路由可能被导向了*无推理或低推理变体*，但其他人声称是直接通过 API 设置推理级别的。


  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1438665583930507348)** (6 messages): 

> `Messagebox max length, Native Websearch, Gemini Web Search, Claude Structured Outputs` 


- **Messagebox 长度导致 UI 崩溃**：一位成员报告称，如果消息长度过长，超过了 messagebox 的最大长度，消息将无法生成，且无法进行编辑，并附带了[截图](https://cdn.discordapp.com/attachments/1392278974222307469/1438665584156872805/Screenshot_2025-11-14_at_4.33.08_AM.png?ex=691906bc&is=6917b53c&hm=6d7a721a88f4b9e79710fd3ccdb38dc94eefbb1e177aa2805a87e2a60b65ec82&)说明。
- **是否支持原生 Websearch 工具？**：一位成员询问是否会为 Google 和 XAI 添加原生 websearch 工具，并指出 *两者都已支持该功能*。
   - 另一位成员表示，针对 **Gemini** 的 web search 功能正在开发中。
- **Claude 终于支持 Structured Outputs**：一位成员分享了 [Claude 关于 structured outputs 公告](https://claude.com/blog/structured-outputs-on-the-claude-developer-platform)的链接，并评论道 *终于支持了？？*。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1438686822841712690)** (39 messages🔥): 

> `MutOrigin effects, Mojo memory model vs C++, Mojo iterator API, Parametric Mutability, Origins and Lifetimes` 


- **`MutOrigin.external` 缺乏 Lifetime**：Mojo 中的 `external` origin **不会延长 lifetimes**，编译器可能会激进地销毁具有该 origin 的对象；在可能的情况下，建议优先使用 named origins。
   - 一位成员建议使用图表来解释编译器如何通过不同的 origins 跟踪 lifetimes。
- **Mojo 内存模型让人费解**：一位没有 C++ 背景的用户发现 [Chris Lattner 关于 Mojo 内存模型的视频](https://docs.modular.com/mojo/manual/)（包含 L, R, B 值和图表）非常难以理解。
- **Iterator API 仍在迭代中 🚧**：Mojo 的 **iterator API** 仍处于开发中（**WIP**），建议使用 `for v in collection^:` 语法来实现 move semantics。
   - 澄清了无法通过 *read 和 mut ref 进行 override*，但可以使用 `ref self` 进行 parametric mutability。
- **`ref self` 的改进启示**：`ref self` 实现了 **parametric mutability**，允许函数在 mutability 上具有通用性（generic），从而取代了对独立的 `Iter` 和 `IterMut` 类型的需求。
- **Origins 轨道，Lifetimes 存续**：**Origins** 保持对象存活，而 **lifetimes** 跟踪对象何时销毁；在 Mojo 的内存管理中，这是两个不同的概念。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1438680203495866380)** (101 messages🔥🔥): 

> `HipKittens 论文与 Mojo 在 AMD GPU 上的性能，API 设计：mut/immut vs read/write 前缀，GPU 编程：线程管理与 Kernel 优化，MAX 图编译器，`@always_inline("builtin")` 替代方案与 `@comptime` 装饰器` 


- ****HipKittens 论文指出 Mojo 的 MHA Kernel 性能问题****：[HipKittens 论文](https://arxiv.org/abs/2511.08083)提到，由于昂贵的 bank conflicts，**Mojo 的 MHA Kernel 在 MI355X 上仅达到峰值性能的 50%**。
   - 一位成员表示，只要 *LLVM 能够与其通信，你就可以在编译时为其构建抽象*。
- ****API 设计：可变性辩论持续升温****：讨论围绕 API 前缀的标准化展开，正在考虑 `mut/immut` 和 `read/write` 等选项，参考了[这篇论坛帖子](https://forum.modular.com/t/mojo-proposal-renaming-read-to-immut/2449)。
   - 一些人更倾向于 `immut/mut`，因为 `read/write` 可能会与 I/O 上下文产生混淆，而另一些人则认为一致性高于一切。
- ****GPU 线程管理：避免调度过载****：成员们讨论了 GPU 线程管理的最佳实践，指出启动过多线程（例如 100 万个）会导致**调度开销**。
   - 建议将**线程限制在 `(warp/wavefront width) * (max occupancy per sm) * (sm count)`**，并让每个线程在该限制之外承担更多工作，将 GPU 视为向量处理器以避免抽象。
- ****MAX：用于性能优化的图编译器****：图编译器被强调为一种优化性能的方法，特别是当硬件形状或程序汇编预先未知时。
   - 一位成员表示，MAX 可以计算出*专门针对该 GPU 应该启动多少个 warps*，并且在进化算法等算法中按顺序连接 Kernel 时非常有用。
- ****考虑 `always_inline("builtin")` 的替代方案和 `@comptime` 装饰器****：团队正在考虑移除 `always_inline("builtin")` 并将其使用限制在标准库中。
   - 一位成员建议将其替换为 `@comptime` 装饰器，表明它应该在编译时以可预测的方式进行折叠。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/)** (1 messages): 

.mjadams: 这就是梦想。在笔记本电脑上开发，在超级计算机上部署。
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1438705781385203865)** (1 messages): 

> `ChatGPT 中的群聊功能，协作式 ChatGPT 对话` 


- **ChatGPT 新增用于协作的群聊功能**：**ChatGPT** 中的群聊功能目前正在**日本、新西兰、韩国和台湾**进行试点，正如[博客文章中所宣布的](https://openai.com/index/group-chats-in-chatgpt/)。
   - 该功能开启了一种与**朋友、家人或同事**以及 **ChatGPT** 在同一个对话中协作的新方式。
- **ChatGPT 群聊目标涵盖社交与专业用途**：新的**群聊功能**旨在提供一个协作平台，让**朋友、家人和同事**在共享对话中与 **ChatGPT** 互动。
   - 该试点计划目前仅限于**日本、新西兰、韩国和台湾**的用户。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1438624433722888312)** (70 条消息🔥🔥): 

> `GEMINI 3.0, Sora 2, GPT-5 令人失望之处, AI prompt engineering` 


- **Gemini 3.0 秘密发布并接受测试**：一位成员分享了一段 [YouTube 视频](https://www.youtube.com/watch?v=0-CbsNB9tdk)，声称经过全面测试的 **Gemini 3.0** 秘密版本正在“引起轰动”。
   - 另一位成员表示 **Gemini Pro 2.5** 已在 Google AI Studio 中可用，但在上传文件时出现错误，可能是由于使用了 HEVC/H.265 编码的不受支持的 **Sora 2** 视频格式。
- **GPT-5.1 Mini 和 Nano 版本令人失望**：用户对缺少 **GPT-5.1-mini** 和 **-nano** 版本表示失望，一位用户表示：“我的失望无以言表，我的一天都被毁了。”
   - 一些用户觉得新模型太“傲慢（sassy）”，而另一些人则没有注意到区别，这可能是由于自定义的傲慢设置所致。
- **Prompt Engineering 技能不断演进**：一位成员建议，真正的高手思考的不仅仅是 Prompt，另一位用户回复问到：“Prompt 大神会怎么做？:p”
   - 另一位成员则完全否定了 Prompt 的整个想法，指出：“首先，完全抛弃 Prompt 这个概念”。
- **Sora 2 AI 视频生成器**：一位用户分享了来自 notebookcheck.net 的文章，[Sora 2 是 OpenAI 始终如一但不稳定的 AI 视频生成器](https://www.notebookcheck.net/Sora-2-is-OpenAI-s-consistently-inconsistent-AI-video-creator.1161467.0.html)，并征求意见。
   - 一位正在进行涉及人机张力和超现实氛围视觉项目的用户，正在寻求关于如何利用 **Sora** 捕捉情绪和控制动作的建议，特别是在节奏、镜头移动和情感铺垫方面。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1438643801995481178)** (58 条消息🔥🔥): 

> `图像生成护栏, GPT 4.1 vs 5.1, GPT 群聊, GPT Memory` 


- **图像生成护栏引发愤怒**：用户报告称，新的**图像生成护栏**过于“过度”，甚至阻止了简单的描绘，限制了创作自由。
   - 一些用户讽刺地表达了他们的沮丧：“太*喜欢*（其实是完全厌恶）这个更新了！能无法回过头去编辑我的文本真是太*棒*了！”。
- **GPT-4.1 相比 GPT-5.1 表现回升**：用户注意到，在经历了一些中断问题后，**GPT-4.1** 和 **GPT-4o** 在写作任务上再次恢复正常运行。
   - 一位用户发现 **GPT-5.1** 过于受限且死板，而 **GPT-4.1** 更加顺从，**GPT-4o** 则更容易产生幻觉（hallucinate）。
- **群聊功能在测试运行中被发现**：据报道，备受期待的**群聊功能**正在日本、韩国和澳大利亚进行测试运行。
   - 用户预想该功能可用于书籍系列和桌面 RPG（TRPG）战役等场景中的协作式世界观构建。
- **GPT-5.1 的 Memory 功能引发担忧**：用户观察到 **GPT-5.1** 可以在同一项目下的不同对话中保留信息，这可能并非用户所愿。
   - 一位用户正在寻找方法，防止 **GPT-5.1** 在同一项目的不同对话中引用细节，因为这违背了独立对话的初衷。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 条消息): 

aicreatorske: 皮克斯电影预告片
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 条消息): 

aicreatorske: 皮克斯电影预告片
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1438623378293657600)** (117 条消息🔥🔥): 

> `HuggingChat 订阅问题、HuggingFace 盈利能力、AI 生成视频、IBM Granite 4 系列、Hackathon 启动直播` 


- **HuggingChat 用户对订阅模式表示不满**：用户正在抱怨 **HuggingChat** 转向订阅模式、旧版本功能的移除，以及在 **PRO 订阅**之外产生的意外费用。
   - 一名用户甚至威胁要*每天在 Reddit 上发帖*直到官方做出改变，而其他用户则在寻找替代方案或选择自行运行模型。
- **HF 营收模式与盈利能力探讨**：成员们对 **HuggingFace** 的营收模式提出了疑问，可能的来源包括订阅、企业交易以及来自 **Salesforce, Google, Amazon, NVIDIA, AMD, Intel, IBM 和 Qualcomm** 等大型科技公司的投资。
   - 一名成员声称 **HuggingFace** 已经实现盈利或接近盈利。
- **AI 生成视频：现在没用，但未来可期？**：小组讨论了 **AI 生成视频** 的现状和未来潜力，共识是目前它们还没什么用，但在 **10 年内**可能会变得很有价值。
   - 一名成员分享了他们使用 **AI vision** 检测视频事件并使用 **ffmpeg** 进行相应剪辑的工作，将 [ffmpeg](https://ffmpeg.org/) 描述为*几乎无所不能……你可以处理视频、音频和图像*。
- **Granite IBM 系列已获得支持**：新的 **IBM granite 4 系列**已获得 **Hugging Face Transformers** 和 **Llama.cpp** 的支持。
   - 用户可以通过 [huggingface.co/ibm-granite/granite-4.0-h-small](https://huggingface.co/ibm-granite/granite-4.0-h-small) 和 [huggingface.co/ibm-granite/granite-4.0-h-small-GGUF](https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF) 访问该模型。
- **Hackathon 启动直播**：一名成员询问了 **Hackathon** 的启动直播情况。
   - 他们表示在弄清楚去哪里参加黑客松时遇到了困难，并补充说他们*加入了组织，但提示要选择赛道？我该去哪里操作？*


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1438619946673377391)** (6 条消息): 

> `Propercode, GeoBot, magenta-realtime 的 Dockerfiles, Ploke, Mimir` 


- ****Propercode** 承诺提供生产级代码**：一名成员介绍了 **Propercode**，这是一个多 Agent 编码 CLI 工具，使用以图形方式编排的 "Pydantic AI" Agent，可在[此仓库](https://github.com/JaiSuryaPrabu/proper-code)中找到。
- ****GeoBot** 地缘政治预测框架发布**：一名成员在 HuggingFace 上发布了 **GeoBot Forecasting Framework**，允许用户插入自己的政治数据来生成关于当前或潜在冲突的预测，详见[此处](https://huggingface.co/clarkkitchen22/GeoBot-Forecasting-Framework)。
- ****Magenta-Realtime Dockerfiles** 避开 Google Colab**：一名成员为 **magenta-realtime** 的推理/微调（x86 和 arm64）创建了 **Dockerfiles**，以避免使用 Google Colab，可在[此 YouTube 仓库](https://youtu.be/bLhuE66q-nI)和 [GitHub 仓库](https://github.com/betweentwomidnights/magenta-rt)中找到。
- ****Ploke** 为 Rust 开发者发布编码 TUI**：一名成员展示了 **Ploke**，这是一个用于 Rust 编程的开源编码 TUI，具有原生 AST 解析、语义搜索和语义代码编辑功能，可在[此处](https://github.com/josephleblanc/ploke)获取。
- ****Mimir** 记忆库编排多 Agent 学习**：一名成员介绍了 **Mimir**，这是一个记忆库加 MCP server，具有绘图功能、待办事项列表管理、代码智能和语义向量搜索功能，能够从之前的运行中学习，可在[此仓库](https://github.com/orneryd/Mimir)中找到。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1438867004894478447)** (2 条消息): 

> `Hugging Face Agentic AI 课程, HF_Token, Llama-4-Scout-17B-16E-Instruct` 


- **新学员寻求帮助**：一名参加 Hugging Face **Agentic AI 课程**的新学员正在寻求其他同学的帮助。
   - 他们正尝试构建 **Unit 1** 中的示例 Agent。
- **HF_Token 抛出身份验证错误**：一名学员在使用其 **HF_Token** 时遇到 *"401 Client Error: Invalid username or password"* 错误。
   - 该错误发生在某行代码中，表明 Hugging Face 账户或 Token 可能存在身份验证问题。
- **学员寻求 Llama-4-Scout-17B-16E-Instruct 的访问权限**：一名学员通过 [Hugging Face 表单](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)请求访问 **Llama-4-Scout-17B-16E-Instruct**。
   - 该学员不确定这是否是解决他们遇到的 *401 错误* 的正确方法。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1438693172996866098)** (24 条消息🔥): 

> `Lakh MIDI Dataset, HuggingFace Datasets, Wikipedia Data Cleaning, JSONL vs JSON, Local LLM Hardware Recommendations` 


- ****Lakh MIDI Dataset** 完成全面清洗**: 一名成员清洗并整理了整个 **Lakh MIDI Dataset**，并生成了一个包含超过 **44,000 条目**的结构化 **JSON** 文件，免费提供给社区。
   - 该成员表示，在上传至 **HuggingFace** 后，欢迎社区进行协作和改进。
- **HuggingFace 迎来 **Wikipedia 数据库法语版****: 一名成员将清洗后的 **法语 Wikipedia 数据库**（包含超过 **2,700,000 个 JSON 格式文件**）上传至 [HuggingFace](https://huggingface.co/datasets/YoloMG/wikipedia-fr-2.7m-clean-json)，并正在着手清洗英文版本。
   - 该成员专注于清洗 *templates, tables, html, refs* 等内容，同时保留 *infobox 内容和链接*，力求使每个页面尽可能干净且结构化。
- ****JSONL** 格式在文本数据处理中获得青睐**: 一名成员建议对纯文本数据使用 **JSONL/NDJSON**，因为与 **TAR** 文件相比，它更易于处理，理由是开销更低且支持逐行读取。
   - 有人指出 *TAR 每个文件都有很大开销*，因为 *据我所知，一个 tar 头部大约有 400 字节*。
- **本地 LLM 机器寻求硬件建议**: 一名成员请求推荐 Discord 服务器，以帮助配置一台旨在利用 **3x3090s** 的本地 **LLM** 机器。
   - 另一名成员建议参考 [osmarks.net/mlrig](https://osmarks.net/mlrig/) 和 **Homelab** 社区。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1438619777118634105)** (62 条消息🔥🔥): 

> `Dataset Licensing, EleutherAI's position, Torch Titan vs GPT-NeoX, PleIAs SYNTH dataset, Anthropic's policy` 


- **制定数据集许可：免责盾牌？**: 成员们讨论了创建一种许可，即 *“你承担所有责任”但“你可以随心所欲”*，类似于 *“The CYA1.0 license”*，并带有 copyleft 属性以将该许可传播到衍生作品。
   - 然而，有人指出，此类许可需要特殊的补充条款，以说明 **语料库中存在的受版权保护的作品**。
- **EleutherAI 专注构建，而非游说**: 成员们辩论了说服法律/商业人士忽视其利益动机以服务研究社区的有效性。
   - 有人建议，更好的时间利用方式是构建像 **Common-Pile** 这样许可宽松且高质量的数据集，类似于 Eleuther 构建开放模型复制品，而不是为 **OpenAI/Google** 的模型进行游说。
- **Titan 或 NeoX：选择你的武器**: 当前对 **torch titan** 与 **gpt-neox stack** 的效益分析取决于硬件和所需任务。
   - **NeoX** 更适合进行 **MoEs** 或混合模型，或者如果你使用的是 **AMD GPUs**；而如果你在通用硬件上使用原生模型，**Torchtitan** 通常更容易上手。
- **SYNTH 数据集：新的数据前沿？**: 成员们讨论了来自 **PleIAs** 的数据集（[HuggingFace 上的 SYNTH](https://huggingface.co/datasets/PleIAs/SYNTH)），其中一人对合成数据普遍持怀疑态度。
   - 还有关于将合成数据集用于预训练的讨论（[Pleias 博客文章](https://pleias.fr/blog/blogsynth-the-new-data-frontier)）。
- **Anthropic CEO 遭到抨击**: 成员们对 Anthropic 的政策面表示担忧，提到了其 CEO 对中国的看法，并质疑他们如何避免查看用户数据。
   - 一些人认为 Anthropic 正在制造恐慌以获取战略优势并挤出竞争对手，为了所谓的“安全”而牺牲隐私。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1438650303250432202)** (2 条消息): 

> `Sparse Circuits, Neural Networks, Interpretability` 


- **OpenAI 发布 Sparse Circuits 论文**: OpenAI 发布了一篇博客文章和论文，名为 [Understanding Neural Networks Through Sparse Circuits](https://openai.com/index/understanding-neural-networks-through-sparse-circuits/)。
   - 这一发布引起了成员们的兴奋。
- **引发可解释性讨论**: Sparse Circuits 论文的发布引发了社区内关于可解释性的进一步讨论。
   - 成员们正在分析这些发现对于理解神经网络行为的意义。


  

---

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1438644266577559613)** (2 条消息): 

> `Hermes 4 API 定价，Cline 支持 Hermes 4` 


- **Nous 大幅下调 Hermes 4 API 价格！**：**Hermes 4 70B** 和 **Hermes 4 405B** 的 API 价格已降低 **70%**，请登录 [portal.nousresearch.com](https://portal.nousresearch.com/) 注册并体验 API。
   - 查看 [X](https://x.com/NousResearch/status/1989077400957911394) 上的公告。
- **Cline 通过 Nous API 集成 Hermes 4**：开源 Agentic 编程平台 **Cline** 现在通过 Nous 门户 API 提供对 **Hermes 4** 的直接支持，增强了其功能。
   - 更多详情请见 [X (Nous)](https://fxtwitter.com/NousResearch/status/1989427241424654534) 和 [X (Cline)](https://fxtwitter.com/cline/status/1989432694867193988)。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1438623262539255878)** (61 条消息🔥🔥): 

> `Lain 效应，100 万次下载，Hermes MoE，GPT5.1 思维，Hermes4 代码模型` 


- **Nous Research 庆祝 “Lain 效应” 和 100 万次下载**：社区庆祝下载量达到 **100 万次**，称之为 `Lain effect` 并分享了 [一段庆祝视频](https://cdn.discordapp.com/attachments/1149866623109439599/1438626291027808257/WhatsApp_Video_2025-11-13_at_7.26.11_AM.mp4?ex=6918e224&is=691790a4&hm=043760ca069476bcd4ea606f861a562bba37dd074ae79f2d6e6c823e832813b7&)。
   - 热情的成员们发布了 GIF 并对这一成就表示自豪。
- **Hermes4 现已成为代码模型**：成员们注意到 **Hermes4** 现在是 [cline 仓库](https://github.com/NousResearch/Cline) 中的一个代码模型，其中一人表示 *我昨晚看仓库时看到了*。
   - 一位用户发布了一张图片，上面写着 *Hermes4 现在是 cline 中的代码模型了？！*。
- **GPT-5.1 输出表情符号**：一位用户分享了 **GPT5.1** 输出的样本，其中将表情符号与推理混合在一起，称之为 Agentic 表情符号传播，另一位用户分享道 *这是有史以来最酷的事情之一*。
   - 其他人确认了这一问题。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1438735560931016735)** (5 条消息): 

> `transformers.js embeddings, pgvector 数据库, AI 陷阱, Jetson Orin Nano x5` 


- **在使用 transformers.js 生成嵌入时遇到困难**：一位成员正在使用 **transformers.js** 为高度结构化的法律地方条例在本地生成 Embeddings，但在使用 **pgvector** 作为数据库时得分相当低。
   - 逐字匹配的分数在 40 分左右，尽管进行了深度分块并为 Embeddings 提供了面包屑上下文，正在寻求提高整体搜索质量的建议。
- **避免 AI 陷阱**：一位成员询问该用户是否在胡乱按大小分块并只返回块，并指出 *那是世界上最著名的 AI 陷阱*。
   - 他们建议 *将心理状态“最小化”到像 Jetson Orin Nano x5 这样的最小系统中？*


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 条消息): 

teknium: https://fxtwitter.com/cline/status/1989432694867193988?s=46
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1438622175048499432)** (47 条消息🔥): 

> `AI 网络间谍活动，Holo2 模型，Mira Murati 的 Thinking Machines Lab 估值，ChatGPT 群聊，Vercel 的 AI Agents` 


- **披露自主 AI 间谍活动**：**Anthropic** 揭露了一个由中国政府资助的团体发起的全自主、**AI 驱动的间谍活动**，目标针对科技、金融和政府部门，详见此 [推文](https://x.com/AnthropicAI/status/1989033793190277618?s=20)。
- **HCompany 推出 Holo2，在 UI 方面超越 GPT-4V**：**HCompany** 推出了 **Holo2** 多模态模型系列，该系列基于 **Qwen3-VL** 构建，在 ScreenSpot-Pro、OSWorld-G 和 Computer-use 基准测试中达到 SOTA，如[此推文](https://x.com/hcompany_ai/status/1989013556134638039)所述。
- **Thinking Machines Lab 估值达到 500 亿美元**：**Mira Murati 的 Thinking Machines Lab** 目前估值为 **500 亿美元**，引发了关于估值指标的辩论，根据[此推文](https://x.com/shiringhaffary/status/1989073320529261132)。
- **OpenAI 在亚太地区试点群聊功能**：**OpenAI** 已悄悄在日本、新西兰、韩国和台湾的 **ChatGPT** 中推出了群聊支持，如[此推文](https://x.com/OpenAI/status/1989138776585851038?s=20)所述。
- **Vercel 的 AI Agents 解决支持工单**：**Vercel** 内部正在使用 **AI Agents**，超过 **70%** 的支持工单得到解决，处理速度为 6.4 apps/s，捕获了 52% 的隐藏缺陷，并正在考虑开源相关架构，详见[此推文](https://x.com/rauchg/status/1989425561995972618?s=46)。


  

---

### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1438621839755841547)** (8 条消息🔥): 

> `Nano Banana, Image Models, AI Kino, Cat Orchestra Video` 


- **Nano Banana 可以进行 Prompt Engineering 吗？**: 成员们讨论了针对 **Nano Banana** 等图像模型的 Prompt Engineering 和测试。
   - 一位成员想知道在图像 Token 生成过程中是否存在类似 **min_p** 的机制，并引用了[这条推文](https://x.com/paularambles/status/1989029622395322816?s=46)。
- **AI Kino 病毒式猫咪视频**: 一段病毒式传播的 **AI 生成视频**显示，一只猫在午夜带着一支不断壮大的乐队骚扰它的主人。
   - 回复中称赞其为 **“AI kino”**，但反对者仍将其斥为 **“slop”**，详见[此推文线程](https://xcancel.com/paularambles/status/1989029622395322816?s=46)。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1438763509457616956)** (1 条消息): 

> `Kimi, Together AI, MoE, Tool Calls` 


- **Together AI 上的 Kimi K-2 Thinking 深度解析**: 一场 **Kimi K-2 Thinking 深度解析**直播将于 **2025 年 11 月 19 日** **PT 时间上午 9 点**在 **Together AI** 举行，承诺将进行一次快速而强大的探索。
   - 活动将重点探讨 **1T MoE** 如何在单次运行中支持 **300 次 tool calls**，并探索其对 Agent 的影响；注册地址见[此处](https://luma.com/g5qcq85z)。
- **Kimi K2 赋能 Tool Calling**: 深度解析活动旨在展示 Kimi K2 如何实现在单次运行中进行 300 次 Tool Calls，这将有利于构建 Agent。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1438658665425928243)** (24 条消息🔥): 

> `Kimi CLI Tool Use, Jailbreaking Banter, Moonshot Devs, Kimi CLI Usage, Client Side Rendering to Server Side Rendering` 


- ****Kimi CLI 的工具时间****: 用户讨论了 Kimi CLI 中的 **tool use**，指出这涉及 AI 通过 **[action(x)]** 解析来使用外部工具（如网页搜索或文件读取）。
   - 一位用户在短短三天内就耗尽了 **39 美元套餐**中的 **7100 次调用**。
- ****越狱讨论引发热议****: 一位用户询问是否允许讨论 **jailbreaking**（越狱），并参考了一张附图。
   - 另一位用户澄清说，社区准则严格适用于 Kimi Bot 的使用，而非一般性讨论。
- ****Moonshot 开发者现身****: 一位用户询问服务器中是否有 Moonshot 的开发者，并注意到 Aspen 不太活跃。
   - 另一位用户指出，带有 **Kimi Team** 身份组的人员是 Moonshot 员工，由于时差原因（中国时间已晚），回复可能会有延迟。
- ****React 渲染革命报告****: 一位用户分享了他们在 **React Vite** 中从 **client-side rendering**（客户端渲染）切换到 **server-side rendering**（服务端渲染）的经历。
   - 他们提到 *"仍有大量更新在进行中 ahahhaharesult 🗿"*。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1438654009559879812)** (14 条消息🔥): 

> `Caching chunks, 404 no cord found, aider-ce documentation, aider with openrouter, MCP servers` 


- **分析分块被缓存了！**: 一位成员询问关于缓存分析分块（caching chunks of analysis）并仅根据需要刷新文件的问题。
- **404 No Cord Found**: 一位成员遇到了 *404 no cord found* 错误，通过设置正确的模型以及 `OPENAI_API_BASE` 和 `OPENAI_API_KEY` 解决了该问题。
- **Aider-CE 设置文档？**: 一位成员询问关于设置 aider-ce 的文档。
   - 另一位成员建议参考 [Aider 官方文档](https://aider.chat/docs/)，认为其同样适用。
- **Aider 在使用 Openrouter 时挂起**: 一位用户报告说，在使用默认设置的 **Openrouter** 时，**aider** 会挂起，对 Prompt 或 Ctrl+C 均无反应。
- **MCP 设置技巧**: 一位成员询问如何设置 MCP 服务器。
   - 另一位成员建议从 Repo 的 README 开始，并在他们的[关于将 Aider CE 与 Chrome DevTools 配合使用的博客](https://example.com)中分享了他们最喜欢的 MCP 设置。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1438994049397821591)** (2 messages): 

> `Clickable links in terminal, Aider, ChatGPT, Terminal Configuration, Prompt Engineering for URLs` 


- **终端中可点击链接的难题**：一位成员面临 **ChatGPT** 生成的 **可点击链接** 在终端中无法工作的问题，链接仅显示为带下划线的文本，而不是 URL。
   - 有成员建议这可能是 **终端问题**，并请求提供 **确切的 Prompt 和响应** 以进一步诊断问题。
- **终端配置困扰**：用户怀疑问题出在他们的终端配置中，导致无法正确渲染可点击的 URL。
   - 进一步调查需要检查终端设置及其处理 URL 格式的方式，以及 **Aider** 是否产生了干扰。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1438806872185372784)** (1 messages): 

> `DeepSeek OCR, Serverless GPU inference, OpenAI vision API` 


- **DeepSeek OCR 实现 Serverless 部署**：[DeepSeek OCR 模型](https://github.com/neosantara-xyz/deepseek-ocr-api) 可以在不需要本地计算机的情况下进行部署。
   - 提供的 **API** 托管在 **Modal Serverless Compute** 上，提供带有免费额度的 **GPU** 资源访问，并且通过使用图像 URL 或 base64 输入，与 **OpenAI vision API** 兼容。
- **Modal 提供的免费 GPU 推理**：上述推理过程托管在 Modal Serverless Compute 上，可访问 **GPU**，并提供免费额度。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1438647379841515772)** (13 messages🔥): 

> `GEPA comparison, Prompt Optimization Methods, AI Agent Dev` 


- **GEPA 大比拼：Synth vs DSPy**：一位成员质疑 **Synth 的 GEPA** 是否优于 **DSPy 的 GEPA**，因为底层算法应该是相同的（[X 帖子链接](https://x.com/JoshPurtell/status/1989068917655097520)）。
- **手动 Prompt 依然盛行！**：一位成员指出，绝大多数用户（>80-90%）仍在 *手动* 管理他们的 Prompt，并且不了解 **自动 Prompt 优化方法**。
   - 另一位成员表示赞同，称自动 Prompt 优化并没有被广泛讨论或采用，尽管他们预期这会更加普遍。
- **AI Agent 专家求合作**：一位在利用 **LangChain**、**OpenAI API**、**Python**、**FastAPI**、**Next.js** 和 **TypeScript** 创建 **AI Agent** 和 **自动化层** 方面有经验的成员表达了合作意向。
   - 他们强调自己专注于构建具有 *可靠性、可扩展性和速度* 的系统，而不仅仅是原型。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1438698106794409995)** (12 messages🔥): 

> `openpilot PR, tinygrad with C++, tinygrad's ptrblck, NeurIPS, TFLite` 


- **OpenPilot PR 获批并承诺防止回退**：一位成员庆祝 [openpilot PR #36615](https://github.com/commaai/openpilot/pull/36615) 获批，并承诺防止未来的功能回退。
- **tinygrad 与 C++ 嵌入式系统**：一位成员询问在嵌入式系统中使用 **tinygrad** 配合 **C++** 的情况，引发了关于该项目适用性的讨论。
   - 另一位成员分享了来自 [@__tinygrad__ 的相关推文](https://x.com/__tinygrad__/status/1989026590127464554)。
- **NeurIPS 参会者**：一位成员询问谁会参加 **NeurIPS**，并链接了与该活动相关的 [comma_ai 推文](https://x.com/comma_ai/status/1989379959417442419)。
   - 另一位成员表达了对未来会议能有在线版本的希望。
- **TFLite 的易用性**：一位成员建议 **TFLite** 在易用性方面很难被超越，但如果你的硬件栈受控且得到支持，**tinygrad** 可能会表现得很好。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1438786747214463036)** (1 messages): 

> `tinygrad .pth loading, Directly load .pth` 


- **tinygrad 现在可以加载 .pth 文件**：团队合并了一个 Pull Request，支持将 `.pth` 文件直接加载到 **tinygrad** 中，而无需先在 **torch** 中加载模型。
- **tinygrad 直接加载**：tinygrad 合并了直接加载功能，可以从 .pth 直接加载模型。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1438898791817281617)** (10 条消息🔥): 

> `Chat Mode 移除, Pro 订阅者, Credit 使用情况, Workflow 自动化` 


- **Chat Mode 消失后又重新出现！**: 一位 Pro 订阅者报告称 **Chat Mode** 被移除后又恢复了，称其*相当奇怪*。
   - 另一位用户报告称 **Chat Mode** 对他们来说仍然缺失，并且官方似乎更改了**积分系统 (points system)**，现在 **Pro** 用户获得 **40k 积分**，而不是之前的 **19900**。
- **请求 Pro 群聊**: 一位用户请求建立 **Pro 群聊**以替代未受监管的聊天，并指出 **Credit 使用情况**相当不一致。
   - 该用户还注意到，进行 **1 shot 构建**比尝试修改消耗的 Credit 更少。
- **专注于自动化的工程师寻求支持与合作**: 一位在 **Workflow 自动化**、**LLM 集成**、**RAG**、**AI 检测**、**图像与语音 AI** 以及**区块链开发**方面拥有丰富经验的工程师分享了他们的专业知识以及[作品集](https://devx-green.vercel.app/)链接。
   - 他们使用 **Dspy**、**OpenAI APIs** 和**自定义 Agent** 构建了自动化流水线和任务编排系统，将响应时间缩短了 **60%**。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1438936187774570678)** (5 条消息): 

> `MCP 维护者在 NeurIPS, Agentic Economies 面板, Model Context Protocol 发布候选版本` 


- **MCP 维护者受邀参加 NeurIPS 的 Agentic Economies 面板**: 一位成员邀请 MCP 维护者参加 **12 月 1 日**在**圣迭戈** **Qualcomm** 举行的 **NeurIPS** 期间关于 **Agentic Economies** 的技术面板讨论。
- **Model Context Protocol 的发布候选版本已冻结**: 该规范现已 ❄️ **__冻结 (frozen)__** 以备即将发布的版本，这意味着不会引入重大更改，当前草案已正式归类为包含 **17 个 SEPs** 的**发布候选版本 (Release Candidate)**。
   - 鼓励成员进行测试，并针对任何问题 [在 GitHub 中](https://github.com/modelcontextprotocol/modelcontextprotocol/issues) 提交 issue。