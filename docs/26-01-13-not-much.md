---
companies:
- anthropic
- langchain
- apple
date: '2026-01-13T05:44:39.731046Z'
description: '**Anthropic** 将其 AI 智能体（Agent）产品整合至 **Cowork** 品牌下，把此前的 **Claude Code**
  和 **Claude for Chrome** 等工具统一为一个智能体。为了确保安全性，该智能体利用 **Apple 的虚拟化技术**和 **bubblewrap**，运行在沙盒化的
  Linux 虚拟机（VM）环境中。


  与此同时，**Anthropic Labs** 迎来重组，Mike Krieger 卸任首席产品官（CPO），重点转向通过一个年经常性收入（ARR）超 10 亿美元的智能体实验室来实现
  **Claude** 的产品化。AI 社区正在讨论“**氛围编程**”（vibe coding）的含义，强调比起随意的编码，更应注重严谨的工程师验证。


  **LangChain** 发布了 **Agent Builder GA**（正式版），提供无需代码但功能强大的智能体编排功能，如记忆、触发器和人工在环（human-in-the-loop）审批。一些专家则主张为了提高效率，应将智能体工具简化为核心的文件系统和
  bash 访问权限。此外，开发者利用 **QEMU** 和沙盒工具对类 Cowork 环境进行的开源复刻，凸显了 AI 智能体技术的快速**商品化/普及化**趋势。'
id: MjAyNi0w
models:
- claude
- claude-code
people:
- mike_krieger
- ben_mann
- gergely_orosz
- yuchen_jin
- harrison_chase
- jared_z
title: Anthropic Labs：Cowork、Claude Code、MCP，以及由 Mike Krieger 和 Ben Mann 领导的技能孵化器。
topics:
- sandboxing
- agent-ux
- agent-orchestration
- human-in-the-loop
- memory-management
- tooling-simplification
- linux-virtualization
- security
- agent-productization
---

**Anthropic 的产品工作室日趋成熟。**

> 2026年1月13日至1月14日的 AI 新闻。我们为您查看了 12 个 subreddit、[**544** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **24** 个 Discord（**204** 个频道，**2271** 条消息）。预计节省阅读时间（按每分钟 200 字计算）：**202 分钟**。**我们的新网站**现已上线，提供完整的元数据搜索和所有往期内容的精美展示。请访问 https://news.smol.ai/ 查看完整的新闻分析，并在 [@smol_ai](https://x.com/Smol_AI) 上给我们反馈！

我们通过这篇文章汇总了 [Cowork](https://claude.com/blog/cowork-research-preview) 和 [Anthropic Labs](https://www.anthropic.com/news/introducing-anthropic-labs) 的接连发布。

Cowork 是对之前大量工作的整合与产品化，涵盖了从 [Computer Use](https://www.anthropic.com/news/3-5-models-and-computer-use) 到 [将 Claude Code 整合进 Claude Desktop](https://blog.getbind.co/claude-code-is-now-available-on-the-claude-desktop-app/) 再到 [Claude for Chrome](https://www.reddit.com/r/ClaudeAI/comments/1prcypb/anthropic_just_dropped_claude_for_chrome_ai_that/)。现在，这些都统一在一个内聚的品牌、UI 和通用 Agent 之下，即 Cowork。

相比之下，Labs 的变动更简单：这是一次组织架构调整，Mike Krieger 卸任 Anthropic 的 CPO（由前 Meta 员工 [Ami Vora](https://www.linkedin.com/in/amvora/) 接任），现在他与 Ben Mann 共同运营一个年收入（ARR）超过 10 亿美元的 [Agent Lab](https://www.latent.space/p/agent-labs)，致力于将 Claude 产品化。

---

# AI Twitter 总结

**AI Agent 产品：Claude Code/Cowork、LangSmith Agent Builder 以及“Agent 化”的开发工作流**

- **Claude Cowork + Claude Code 成为“终端原生 Agent”的新基准，沙箱化（Sandboxing）成为标配**：多条推文关注 Anthropic 的 Cowork 如何通过 **Apple 原生虚拟化技术**启动 **Linux VM**，并在沙箱（例如通过 **bubblewrap**）中运行，以遏制不安全指令和失控进程或误删等故障模式（[沙箱细节](https://twitter.com/dejavucoder/status/2010993418630262817)）。更广泛的主题是：Agent 的 UX 正在收敛于：*给模型一个文件系统 + Shell + 严格的权限*，然后通过人工审查快速迭代。需求端的痛点也显现出来：高级用户希望减少权限提示，而无需被迫选择 `--dangerously-skip-permissions`（[关于摩擦的抱怨](https://twitter.com/levelsio/status/2011129631001170244)；相关的[笑话](https://twitter.com/LaurencePostrv/status/2011134254051139712)）。

- **“Vibe coding”引发的反弹 → Agent 辅助工程更清晰的分类**：一个反复出现的争论是，“Vibe coding”一词被误用来描述使用 Agent 完成的严谨、生产级的工作。Gergely Orosz 认为，当工程师正在验证并完成闭环时，我们应该停止称之为 Vibe coding（[推文](https://twitter.com/GergelyOrosz/status/2011001698370699374)）。Yuchen Jin 补充了一个更清晰的定义：Vibe coding 最初指的是*完全不看代码*；一旦你进行了审查，它就更接近于“清醒编程（Lucid coding）”（[推文](https://twitter.com/Yuchenj_UW/status/2011137879112908870)）。这很重要，因为它重构了实际发生的变化：并不是“工程已死”，而是**具有审美和验证纪律的工程师获得了杠杆**。

- **LangSmith Agent Builder 正式发布（GA）：无代码但绝非玩具（MCP、Memory、Triggers、Inbox/HITL）**：LangChain 宣布 Agent Builder 已**正式发布（GA）**（[GA 公告](https://twitter.com/LangChain/status/2011129282580660314)）。Harrison Chase 及其团队强调了核心原语：**Memory**、**Skills**、**Subagents**、**MCP/工具集成**、用于自主运行的 **Triggers**，以及用于人工审批的 **Agent 收件箱**（[演示](https://twitter.com/hwchase17/status/2011126016287113681)；GA 回顾[见此](https://twitter.com/hwchase17/status/2011134704934957382)）。多位用户强调，即使对于技术用户它也非常有用，因为它整洁地打包了编排和可观测性（[价值框架](https://twitter.com/KevinBFrank/status/2011154462128144539)）。元经验（Meta-lesson）：编排产品正在从“提示词 + 工具”转向*运营 Agent 栈*（Auth、Triggers、审计追踪、受监督的操作）。

- **一个反向趋势：“别挡模型的道”并简化工具**：Jared Z 认为，添加工具/护栏可能会降低性能，因为你强加了额外的分支决策；他举例说 Vercel 将一个 text-to-SQL Agent 简化为仅拥有文件系统 + Bash 访问权限（[推文链](https://twitter.com/imjaredz/status/2011218314035642464)）。这符合日益增长的共识，即 *Bash + 文件系统是通用的工具调用*，现代模型可以承担以前需要 DAG 才能处理的复杂性。

- **开源复现和 “Cowork 仿制品” 展现出快速商品化的趋势**：一位开发者使用 **QEMU + bubblewrap + seccomp** 构建了一个跨平台的类似 Cowork 的虚拟机（VM），通过 `vmctl` 实用工具和 websocket 进行控制（[推文](https://twitter.com/SIGKITTEN/status/2011077925085347909)）。MiniMax 还声称有人使用兼容 Anthropic 的 API 重建了 Cowork 并将其开源（[推文](https://twitter.com/MiniMax_AI/status/2011270108166107311)）。这释放了一个信号：**Agent 外壳（Agent shells）正在成为可复制的基础设施模式**，而非专有的技术护城河。

---

**长上下文 + 记忆：从 RAG 分块之战到递归语言模型和 RL 记忆**

- **文件系统 Agent 对比向量搜索：混合模式是真正的结果**：LlamaIndex 将 “fs-explorer” 风格的 Agent 与传统的 RAG 进行了基准测试。他们的总结是：文件系统探索可以**更准确**（具备全文件上下文）但**速度较慢**，而在大规模（1000+ 文档）情况下向量搜索胜出（[LlamaIndex 帖子](https://twitter.com/llama_index/status/2011121143927972076)；Jerry Liu 的综合分析见[此处](https://twitter.com/jerryjliu0/status/2011130432205832664)）。Weaviate 重申了分块中的核心权衡：**检索精度与上下文丰富度**，且并不存在通用的分块大小（[推文](https://twitter.com/weaviate_io/status/2011088315663978739)）。

- **MemRL：将记忆检索视为 RL（效用感知），而非相似性搜索**：DAIR AI 重点介绍了 **MemRL**，它保持基础 LLM 冻结，并学习**情景记忆（Episodic Memories）上的 Q-values**（意图–经验–效用），采用两阶段检索：语义过滤后进行效用排序（[总结](https://twitter.com/dair_ai/status/2011086096986443905)）。如果这些说法成立，对于生产环境中的 Agent 来说这是一个极具吸引力的模式：既能避免微调带来的灾难性遗忘，又能通过学习到的记忆策略**从经验中不断改进**。

- **递归语言模型（RLMs）：对 Prompt 的符号化访问，而非“作为工具调用的子 Agent”**：Omar Khattab/lateinteraction 的帖子认为大多数“子 Agent”实现都忽略了核心观点：你无法将数百万个子调用具体化为工具调用，你需要**类指针/符号化访问 Prompt** 的能力，以便通过编程方式对其进行递归（[评论](https://twitter.com/lateinteraction/status/2011250721681773013)）。The TuringPost 的综述将 RLM 定义为一种推理层架构，它将上下文卸载到 Python REPL 变量中，使模型可以通过代码操作上下文，在无需重新训练的情况下扩展超过 **1000 万个 Token**（[总结](https://twitter.com/TheTuringPost/status/2011272650132504889)）。给工程师的关键启示是：*“长上下文”可能越来越多地意味着“代码介导的上下文访问”，而不仅仅是更大的窗口。*

- **通过 Prompt 压缩缓解上下文腐败**：DSPy 被用作减少 Prompt 长度而不损失性能的示例工作流，明确被定位为一种对抗上下文衰减的方法（[推文](https://twitter.com/hammer_mt/status/2011022198023082263)）。

---

**视频生成与可控世界模型：Kling Motion Control、Veo 3.1 升级以及新的“世界模型”主张**

- **Kling 2.6 Motion Control 成为同类最佳的性能/动作迁移工具（但身份漂移依然存在）**：多位创作者报告称，Kling 的 Motion Control 可以以极高的精度替换或驱动场景中的角色（[热门评价](https://twitter.com/AngryTomtweets/status/2010975679488409890)）。一段详细的日本演示视频展示了乐器演奏的动作迁移，具有高保真的手指动作和节奏感，暗示了单主体镜头近期即可实现写实化（[演示线程](https://twitter.com/akiyoshisan/status/2010983687727587587)）。Curious Refuge 对其进行了真人叙事测试：视差效果看起来很强，但面部一致性会发生偏移；当参考图像与初始帧接近时效果最好（[测试报告](https://twitter.com/CuriousRefuge/status/2011207976095531524)）。

- **Google 的 Veo 3.1：“Ingredients to Video” 获得人像模式 + 更高分辨率 + 一致性改进 + SynthID**：DeepMind/Google 推出了 Veo 3.1 更新，重点包括 (1) **原生 9:16 竖屏**，(2) 改进的角色/背景一致性，(3) **1080p + 4K** 选项，以及 (4) 用于验证的 **SynthID 水印**（[DeepMind 线程](https://twitter.com/GoogleDeepMind/status/2011121716336984151)；API 总结见[此处](https://twitter.com/_philschmid/status/2011122136619110762)；Gemini 应用部署见[此处](https://twitter.com/GeminiApp/status/2011122407013306875)；Sundar Pichai 的帖子见[此处](https://twitter.com/sundarpichai/status/2011143120516469199)；Demis Hassabis 见[此处](https://twitter.com/demishassabis/status/2011236200397639900)）。工程师应注意该产品的方向：**移动优先的格式 + 内容溯源 + 生产级分辨率**正被置于比纯粹的新奇感更高的优先级。

- **“World model” 品牌化加速；研究基准试图跟进**：PixVerse 将 “R1” 作为 “实时 World model” 进行推广（营销味很浓）([tweet](https://twitter.com/PixVerse_/status/2011100288690897317))。更具技术性的动态是：TencentARC 的 **VerseCrafter** 声称对相机和多物体运动具有 4D 几何控制能力 ([announcement](https://twitter.com/wbhu_cuhk/status/2011109476510941222))。此外，还出现了一个单独的 “针对 Agentic Video Reasoning 的开放网络视频深度研究基准” ([tweet](https://twitter.com/_akhaliq/status/2011105482111651992))，这进一步证实了针对视频 Agent 的评估目前仍不成熟。

---

**开源模型、端侧 ML 以及多模态医疗 AI：MedGemma 1.5、GLM-Image、MLX 吞吐量**

- **MedGemma 1.5 + MedASR：专注于离线和 3D 成像的开源医疗多模态技术栈**：Google 发布了 **MedGemma 1.5**，其体积小到可以在离线状态下运行，并针对多模态医疗任务进行了改进 ([Google AI Devs](https://twitter.com/googleaidevs/status/2011181120793297361))。Phil Schmid 的技术要点列表强调这是一个 **4B** 模型，支持 **3D 卷轴 (CT/MRI)**、纵向对比和解剖定位；他引用了 **89.6% 的 EHR 理解准确率**（提升 22%）和 X 射线定位的 **38% IoU** ([tweet](https://twitter.com/_philschmid/status/2011183904204390654))。Sundar Pichai 将其定位为 “重大升级”，并将其与用于医疗听写的 **MedASR** 搭配使用 ([tweet](https://twitter.com/sundarpichai/status/2011184917670216196))。Google Research 在 Hugging Face 和 Vertex AI 上同步发布了这两项成果 ([tweet](https://twitter.com/GoogleResearch/status/2011185403856883907))。核心结论：**开源、高效、面向临床的多模态模型**正在成为一级发布类别。

- **GLM-Image：用于“海报/PPT/文本渲染”和重知识生成的混合自回归 + Diffusion 架构**：智谱 AI (Zhipu AI) 发布了 GLM-Image，声称通过混合架构实现了强大的文本渲染和信息图/海报生成能力 ([release](https://twitter.com/Zai_org/status/2011247591825068314))。第三方细化了架构细节（例如 “9B AR + 7B Diffusion”）和 “认知生成” 的定位 ([fal launch](https://twitter.com/fal/status/2011271561429311512)；ModelScope 回顾见[此处](https://twitter.com/ModelScope2022/status/2011262011997651194))。对于工程师来说，关键在于其*设计目标*：更好的**布局 + 多行文本**可靠性，这是常见的 Diffusion 弱点。

- **端侧和本地推理持续升温**：LocallyAI 的更新指出 **LiquidAI LFM 2.5** 模型现已通过 MLX 在 iOS 上可用 ([tweet](https://twitter.com/LocallyAIApp/status/2011136235973329301))。MLX 性能基准显示，MiniMax M2.1 在 M3 Ultra 上本地运行并开启 continuous batching：在 32 个请求下达到 **4-bit 220 tok/s** ([tweet](https://twitter.com/ivanfioravanti/status/2011115626690179290))。Awni Hannun 强调 MLX 在 Metal 和 CUDA 上增加了量化支持 (nvfp4/mxfp8) ([tweet](https://twitter.com/awnihannun/status/2011267993091875282))。在 Claude/Cowork 的讨论中，也出现了关于 “本地模型” 隐私性的抨击 ([tweet](https://twitter.com/victormustar/status/2011078287762825474))。

---

**基准测试、评估和 Agent 可靠性：指令遵循、视觉推理极限以及 “乏味的 Agent”**

- **OctoCodingBench：对齐的 Coding Agent 不等于通过单元测试**：MiniMax 发布了 **OctoCodingBench**，旨在衡量 Coding Agent 是否遵守系统提示词、仓库惯例和工具策略——明确解决了仓库中的 “Paperclip-maxing” 行为 ([tweet](https://twitter.com/MiniMax_AI/status/2011266592303432058)；数据集提及见[此处](https://twitter.com/HuggingPapers/status/2011074090686136349))。这是一个重要的转变：将评估从单纯的功能正确性转向**过程约束**和组织规范。

- **BabyVision：MLLM 在 “纯视觉推理” 方面仍然薄弱**：HuggingPapers 引用了 BabyVision 的结果：在 388 个任务中，SOTA MLLM 的准确率为 **49.7%**，而成年人类为 **94.1%**，认为这些任务需要非语言的视觉理解能力 ([tweet](https://twitter.com/HuggingPapers/status/2011048605113581762))。如果你正在构建多模态 Agent，这意味着那些 “看起来已解决” 的 Demo 可能会掩盖脆弱的视觉推理能力。

- **企业级 “乏味的 Agent” 作为产品立场**：AI21 明确推销 “乏味的 Agent (boring agents)”，强调针对**可审计、可重复**输出的优化，而非聊天魅力 ([tweet](https://twitter.com/AI21Labs/status/2011041313039204838))。这与评估趋势一致：减少对 “感觉 (vibe)” 的关注，更多关注治理。

- **METR：从 “能力” 扩展到失控 (Loss-of-control) 框架**：Ajeya Cotra 加入 METR，将 LOC 风险评估扩展到 “手段、动机、机会” 三个维度，并指出动机和机会的衡量目前尚未充分开发，且未来可能成为关键支撑 ([tweet](https://twitter.com/ajeya_cotra/status/2011146702175289563)；定义见[此处](https://twitter.com/ajeya_cotra/status/2011146714183581886))。

---

**基础设施与训练系统：调度器、Attention 后端、量化陷阱以及 FP8/低比特训练**

- **HPC 调度器 vs 云原生编排（Slurm 收购后的讨论）**：dstack 将 Nvidia 收购 Slurm 视为工作负载正向云原生调度器迁移的证据，并提供了一份 Slurm→dstack 迁移指南 ([推文](https://twitter.com/dstackai/status/2011091749901422904))。SkyPilot 推广 “Pools” 作为跨 K8s + 多云的统一批处理队列 ([推文](https://twitter.com/skypilot_org/status/2011128941705339270))。这种模式表明：基础设施团队正在围绕**多集群 GPU 池化**和厂商无关的调度器进行标准化。

- **Diffusers 添加 “Unified Attention” 后端**：Hugging Face Diffusers 发布了一个结合了 Ring 和 Ulysses 特性的新 Attention 后端 ([推文](https://twitter.com/RisingSayak/status/2011092823828021730))。这是持续推动 Attention kernel/后端可交换性和性能可移植性工作的一部分。

- **量化与训练的细微差别持续引发问题**：TensorPro 报告称 MXFP4 量化的 Attention 可能会破坏因果建模（causal modeling），并发布了关于诊断和修复“泄露量化（leaky quantization）”行为的文章 ([推文](https://twitter.com/tensorpro/status/2011198742406578252))。另外，分享的一篇 Google Cloud 文章讨论了**随机舍入（stochastic rounding）**如何缓解低精度训练（FP8/4-bit）中的梯度消失问题 ([推文](https://twitter.com/dl_weekly/status/2011060892897558717))。对于从业者而言：“在 FP8/低比特下训练”越来越可行，但**数值边缘情况**仍是活跃的研究和运维课题。

---

**热门推文（按互动量排序）**

- **日本麦当劳** “Black Pepper!! PV” 帖子（获得巨量病毒式传播） ([推文](https://twitter.com/McDonaldsJapan/status/2010985164692668892))  
- **Joe Rogan 关于“出示证件” / 军事化执法的剪辑** ([推文](https://twitter.com/OfTheBraveUSA/status/2011153857976668290))  
- **Anthropic 向 Python Software Foundation 捐赠（150 万美元）** ([Alex Albert](https://twitter.com/alexalbert__/status/2011143093266104800)；PSF 的感谢信见[此处](https://twitter.com/ThePSF/status/2011060802321584414))  
- **Claude Code / Agent 生产力梗图** 捕捉到了当下的文化时刻 ([nearcyan 的“工程师时代”](https://twitter.com/nearcyan/status/2011129737578500526)；giffmana 的 Cowork 恶搞见[此处](https://twitter.com/giffmana/status/2011165027374334221))


---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Pocket TTS 与本地 AI 工具

  - **[kyutai 刚刚推出了 Pocket TTS：一个拥有 100M 参数的文本转语音模型，具备高质量的声音克隆功能，可在笔记本电脑上运行——无需 GPU](https://www.reddit.com/r/LocalLLaMA/comments/1qbpz5l/kyutai_just_introduced_pocket_tts_a_100mparameter/)** (活跃度: 494): ****Kyutai Labs** 发布了 **Pocket TTS**，这是一个 `100M-parameter` 的文本转语音模型，专为高质量声音克隆设计，无需 GPU 即可在 CPU 上高效运行。该模型可通过 [GitHub](https://github.com/kyutai-labs/pocket-tts) 和 [Hugging Face](https://huggingface.co/kyutai/pocket-tts) 获取，详细信息见其[博客文章](https://kyutai.org/blog/2026-01-13-pocket-tts)。该模型的架构灵感源自连续音频语言模型（continuous audio language models）的最新进展，正如相关的 [arXiv paper](https://arxiv.org/abs/2509.06926) 中所讨论的那样。** 一些用户质疑该模型的性能，认为与大型模型或 Twitch 等应用中使用的“硬编码”解决方案相比，这种尺寸的模型可能无法提供足够的质量。此外，人们对该模型的语言能力以及在不同语言间进行 Fine-tuning 的潜力也表现出了兴趣。

    - 一位用户指出 Pocket TTS 模型存在严重的内存管理问题，其本地测试服务器设置在生成任务之间不会清除内存，导致内存使用量显著增加。他们报告内存占用在系统上达到了 **32 GB**，并建议模型在开始新的生成任务时应清除内存，以防止这种膨胀。
    - 另一位用户提供了 Pocket TTS 模型在 Ryzen **5950X** CPU 上的详细性能分析。他们观察到模型最初使用约 **1.1 GB 的 RAM**，并且能够快速生成音频，首个音频输出时间（time to first audio）约为 **200 ms**。然而，随着 Context 填满，RAM 使用量会显著增长，单篇文章的占用可达 **8.5 GB**。他们还评论说，对于这种尺寸的模型，其语调表现良好，但整体语音质量被描述为一般。
    - 一位用户对 Pocket TTS 等小模型的实用性表示怀疑，认为与更成熟的解决方案相比，如果需要高质量的输出，这些模型可能不值得投入精力。他们提到尝试了 Demo 后发现结果不尽人意，暗示大型模型或硬编码方案在某些应用中可能更有效。

  - **[我制作的一个用于简化本地 AI 模型运行的 Windows 工具](https://www.reddit.com/r/LocalLLM/comments/1qbzd2w/a_windows_tool_i_made_to_simplify_running_local/)** (活跃度: 28): ****V6rge** 是一款基于 Windows 的工具，旨在通过捆绑和隔离其自身的 Runtime 来简化本地 AI 模型的运行，从而避免系统 Python 冲突。它支持通过 GGUF 运行 Qwen、DeepSeek 和 Llama 等本地 LLM，以及使用 Stable Diffusion 和 Flux 变体进行图像生成，并提供基础的语音和音乐生成。该工具旨在减少安装摩擦，可在 [GitHub](https://github.com/Dedsec-b/v6rge-releases-/releases/tag/v0.1.4) 下载。** 有用户担心该工具不是开源的，这使得用户对运行可执行文件持谨慎态度。此外，用户报告了在更改设置时出现“Failed to Save Settings: API error 404”等问题，表明存在潜在的稳定性问题。改进建议包括为生成的图像添加画廊（gallery）功能。

    - 一位用户报告了一个关键问题，即在设置中更改 Model Folder 会导致“API error 404”。这表明应用程序的设置管理中存在潜在的 Bug，可能是由于错误的 API Endpoint 处理或后端缺乏对该功能的支持。
    - 另一位用户在尝试下载 Qwen-Image 或 FLUX.1-dev 等特定模型时遇到了“Error: undefined”。这表明模型下载功能可能存在问题，可能与错误的 URL 处理或服务器端问题有关。
    - 有用户请求提供该工具的 Linux Docker 版本，凸显了对跨平台兼容性的需求。用户建议增加 Docker Compose、维护 Docker Hub 镜像以及 Portainer 支持等功能，这将有助于在容器化环境中更轻松地部署和管理。

### 2. GLM-Image 与 NER 模型发布

  - **[GLM-Image 发布了！](https://www.reddit.com/r/LocalLLaMA/comments/1qc9m6x/glmimage_is_released/)** (热度: 393): ****GLM-Image** 是一款新发布的图像生成模型，采用了混合自回归 (autoregressive) 和扩散解码器 (diffusion decoder) 架构。它在通用图像质量方面与主流的 latent diffusion 模型不相上下，但在文本渲染和知识密集型场景中表现出色，展示了卓越的语义理解和复杂信息表达能力。该模型支持文本生成图像 (text-to-image) 以及各种图像到图像 (image-to-image) 任务，如图像编辑、风格迁移和保持身份的生成，同时保持了高保真度和细粒度的细节。** 该模型基于 **MIT license** 发布，与其西方实验室更具限制性的许可相比，其开放性受到了关注。该模型的性能与 **nano banana 2** 进行了对比，表明这是一项重大进步，特别是其结合了编辑和生成的能力。

    - GLM-Image 采用 MIT license 发布被强调为一个显著优势，特别是与通常在更具限制性的许可下发布模型的西方实验室相比。这种开放的许可协议可能会促进社区内更广泛的采用和创新。
    - 据报道，GLM-Image 在基准测试中的表现与 “nano banana 2” 相当，考虑到它同时具备编辑和生成的双重能力，这一点非常值得关注。这种双重功能使其在各种应用中成为多才多艺的工具，增强了其对开发者和研究人员的吸引力。
    - 该模型由一个 13GB 的 diffusion model 和一个 20GB 的 text encoder 组成，表明了其对资源的高要求。人们期待该模型能被量化 (quantized) 到 fp8，并开发出如 LoRA 等高效的训练方法，以便更轻松地进行实验。

  - **[500Mb 命名实体识别 (NER) 模型，可在本地识别和分类任何文本中的实体。轻松在本地对任何语言进行微调（参见西班牙语示例）。](https://www.reddit.com/r/LocalLLM/comments/1qbnezw/500mb_named_entity_recognition_ner_model_to/)** (热度: 13): **一个新的 `500Mb` 命名实体识别 (NER) 模型已发布，能够在本地识别和分类文本中的实体。该模型旨在方便在不同语言间进行微调 (fine-tune)，并提供了一个专门的西班牙语示例。该模型的紧凑尺寸允许高效的本地部署，无需云端资源，非常适合隐私敏感型应用。然而，帖子中并未指明模型的架构和训练细节。** 该帖子缺乏详细的技术讨论或辩论，因为最高赞的评论是非技术性的，仅仅表达了赞同。

### 3. AI 硬件创新

  - **[AI TOP 100 M.2 SSD](https://www.reddit.com/r/LocalLLM/comments/1qbvycy/ai_top_100_m2_ssd/)** (活跃度: 26): **图片展示了一款 GIGABYTE AI TOP 100E M.2 SSD，其市场宣传称通过提供高带宽来增强 AI 性能，并有可能减轻 RAM/VRAM 的负载。然而，评论者认为这很大程度上是一个营销噱头，因为即使是最快的 PCIe 5 SSD 带宽（约 `10GB/s`）也显著低于 DDR5 RAM（`80GB/s`）。这使得该 SSD 在卸载大型 AI 模型时效果不佳，因为速度会成为瓶颈，特别是对于稠密模型（Dense Models）。稀疏模型（Sparse Models）可能会略微受益，但性能提升仍受限于每秒仅个位数的 tokens per second。** 评论者对该产品的声明持怀疑态度，认为这更多是一种营销策略，而非 AI 工作负载的实际解决方案。他们建议使用目前市面上最好的 NVMe PCIe 5 SSD 即可，因为所谓的性能提升微乎其微。

    - Themash360 强调了在 AI 工作负载中使用 NVMe SSD 的局限性，指出即使 PCIe 5 具有乐观的 10GB/s 带宽，与 DDR5 RAM 的 80GB/s 相比仍然慢得多。他们举例说明，将 240GB 稠密模型中的 100GB 卸载到 NVMe 会导致每秒生成 0.1 个 token，强调了这对稠密模型的低效性。
    - Themash360 还提到，虽然使用 Mixture of Experts (MoE) 模型可以通过卸载稀疏部分来缓解部分性能损失，但改进有限，生成的 tokens per second 仍处于低个位数。这凸显了在处理大型 AI 模型时，现有存储技术在实现高性能方面面临的挑战。
    - desexmachina 指出，更快的 SSD 可能会导致更高的处理器饱和度，这意味着虽然存储速度是一个因素，但整体系统性能还取决于 CPU 处理增加的数据吞吐量的能力。这表明需要平衡的系统架构来优化 AI 工作负载。

  - **[My wishes for 2026](https://www.reddit.com/r/LocalLLaMA/comments/1qbw325/my_wishes_for_2026/)** (活跃度: 767): **这张图片是对 2026 年技术进步的推测性愿望清单，涵盖了 AI 模型和硬件的潜在发展。其中包括发布新版本的 AI 模型，如 GPT-OSS, Gemma 4, Qwen 4 和 GLM Air，以及预计性能将超越 Mistral 123B 模型的 Llama 5。此外，还有一个愿望是出现参数量低于 200B 的 DeepSeek 模型，以及拥有超过 32GB 显存的实惠型 GPU。这张图片反映了对 AI 能力和硬件普及化取得重大进展的渴望。** 评论者对 32GB 以上实惠型 GPU 的可行性表示怀疑，强调了 GPU 价格居高不下的持续挑战。

    - SlowFail2433 强调了 GPT OSS 120B 被低估的性能，指出其令人印象深刻的基准测试得分与参数量比例，以及其有效的 FP4 量化。该模型与 Qwen 4 系列进行了对比，后者经常在 Arxiv 论文中被提及，特别是针对 Agentic RL 应用。讨论强调了小型稠密模型在训练过程中避免 MoE gates 复杂性的优势，因为 MoE 门控会使 RL 场景中的信用分配（Credit Assignment）变得复杂。

  - **[I'm building a real-life BMO with a Raspberry Pi 5 (Mistral/OpenAI + YOLO11n)](https://www.reddit.com/r/LocalLLM/comments/1qbwc35/im_building_a_reallife_bmo_with_a_raspberry_pi_5/)** (活跃度: 8): **该项目涉及使用 **Raspberry Pi 5** 构建一个真实版的 BMO，集成了 **Mistral/OpenAI** 提供 AI 能力，并使用 **YOLO11n** 进行物体识别。开发者正在为这个 AI 伙伴增强人脸和语音识别功能，旨在实现互动游戏体验。未来的计划包括增加机械臂。该项目是开源的，代码可在 [GitHub](https://github.com/ivegotanheadache/BMO) 上获取。** 一位评论者也在开发类似的项目，使用具有语音识别和文本转语音功能的 LLM 助手，并考虑通过 RetroArch 和 Pico 8 加入国际象棋和模拟器等游戏功能。他们正在考虑是集成专用显示器还是使用外部显示器。



## 偏非技术性的 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. xAI 的 Grok 部署及其争议

  - **[官方：五角大楼确认在国防业务中部署 xAI 的 Grok](https://www.reddit.com/r/singularity/comments/1qbo516/official_pentagon_confirms_deployment_of_xais/)** (热度: 1443): **美国国防部将把 xAI 的 Grok AI 集成到五角大楼系统中，允许军事和文职人员处理 5 级影响（Impact Level 5）的受控非机密信息。Grok 将被嵌入到作战和规划系统中，利用来自开源和社交数据的实时全球信号，增强情报分析、决策制定和军事规划。该部署目标是覆盖约 `3 million users`，首阶段将于本月开始。[来源](https://www.washingtonpost.com/business/2026/01/12/artificial-intelligence-pentagon-hegseth-musk/ec8b407a-f026-11f0-a4dc-effc74cb25af_story.html)。** 评论反映了对 AI 集成到军事行动中的怀疑和担忧，一些用户幽默地暗示了潜在的安全风险，另一些人则对现任政府使用此类技术表示不信任。


  - **[卫报：埃隆·马斯克的 Grok 如何每小时生成 6,000 张非自愿裸照。](https://www.reddit.com/r/OpenAI/comments/1qbkpw9/the_guardian_how_elon_musks_grok_generated_6000/)** (热度: 392): **《卫报》（*Guardian*）的调查强调了对埃隆·马斯克的 AI 工具 Grok 的严重滥用，据报道在 2026 年初，该工具每小时被用于生成 `6,000` 张非自愿裸照。这种滥用是一个更广泛趋势的一部分，即用户利用 AI 创建性化和暴力图像，特别是针对女性和未成年人。该报告强调了 AI 技术在内容审核和用户安全方面带来的伦理和监管挑战。** 评论者对 Grok 的滥用表示失望，注意到用户倾向于专注于生成显式内容的趋势。此外，还有对美国 Big Tech 以及社区对该 AI 功能反应的批评，一些用户因色情内容讨论盛行而退出了相关论坛。

    - Fearless_Weather_206 提出了一个关于 Grok 功能可能产生的立法影响的关键点。担忧在于，此类事件可能被用作监管或限制开源 LLM 模型的借口，而这些模型通常比商业模型受到的审查更少。这可能引发关于 AI 发展中创新与监管之间平衡的更广泛辩论。
    - boredatwork8866 强调了一个社区趋势，即用户主要专注于利用 Grok 生成成人内容。这表明用户对该模型创建显式材料的能力有极大兴趣，当此类功能受到限制时，会导致用户不满，暗示了用户期望与开发者施加的伦理准则之间存在紧张关系。
    - Joddie_ATV 对 Grok 生成非自愿图像的能力所带来的伦理和社会影响表示担忧。这提出了关于 AI 开发者在防止其技术被滥用方面的责任，以及目前保护免受此类滥用的安全措施是否有效的问题。

  - **[万无一失](https://www.reddit.com/r/OpenAI/comments/1qc2b0f/nothing_could_go_wrong/)** (热度: 365): **这张图片是一个模因（meme），幽默地评论了美国国防部长宣布将埃隆·马斯克的 xAI 平台 Grok 集成到军事网络的消息。据路透社报道，这一集成是 AI 加速战略的一部分。Jarvis 的推文讽刺地暗示这种集成是低风险的，暗示了对这一举动影响的潜在担忧。评论反映了对将 AI 集成到军事行动中潜在后果的怀疑和幽默，提到了反乌托邦场景和政治评论。** 评论表达了怀疑和幽默，引用了《终结者》（Terminator）系列等反乌托邦场景，并对私营公司在政府合同中的影响力进行了政治评论。

### 2. DeepSeek 的 Engram 模块与创新

  - **[[R] (DeepSeek) Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://www.reddit.com/r/MachineLearning/comments/1qbnkrn/r_deepseek_conditional_memory_via_scalable_lookup/)** (热度: 55): **DeepSeek** 引入了一个名为 **Engram** 的新模块，通过条件记忆实现了一种新的稀疏性维度来增强大型语言模型，从而实现高效的 O(1) 查询。这种方法与传统的 Mixture-of-Experts (MoE) 模型不同，它通过优化神经计算与静态记忆之间的平衡，呈现出一种 U 形 Scaling Law。Engram 可扩展至 `27B 参数`，在包括 `MMLU`、`CMMLU`、`BBH`、`ARC-Challenge`、`HumanEval` 和 `MATH` 在内的各种基准测试中，其表现均优于同等参数量和同等 FLOPs 的 MoE 基准。该模块通过从早期层卸载静态重建任务并增强对全局上下文的 Attention 能力，改善了推理和检索，在长上下文检索任务中取得了显著收益。Engram 的确定性寻址（Deterministic addressing）还支持运行时预取（Runtime prefetching），从而最大限度地减少开销并提高基础设施效率。一条评论强调了 Engram 的实际效率，指出它能够避免为重新计算常见事实而进行不必要的前向传递（Forward passes），从而在保持创新的同时提高了吞吐量和效率。


  - **[DeepSeek V4 Could Blow Claude and GPT Away for Coding](https://www.reddit.com/r/DeepSeek/comments/1qblbjf/deepseek_v4_could_blow_claude_and_gpt_away_for/)** (热度: 226): **DeepSeek V4** 即将发布，据称其在编程任务上的表现将超越 **Claude** 和 **GPT**。该模型引入了 **Engram 模块**，该模块使用记忆查询系统通过解耦记忆与计算来处理超长 Prompt，从而通过让 Attention 和 MLP 层专注于复杂任务来潜在地提升性能。这种架构还可以通过将显存需求卸载到 RAM，从而减少 `30%` 的 VRAM 占用。然而，对于 DeepSeek 能力泄露的信息，也存在一些质疑。一位用户分享了使用 DeepSeek 开发复杂加密器的经验，注意到其在编程方面的效率，与 Meta 和 ChatGPT 相比，代码行数更少。然而，他们发现 Claude 在处理特定功能和提供更具批判性的代码审查方面更胜一筹，尽管 DeepSeek 的鼓励性评分更高。

    - Engram 是 DeepSeek 的一项功能，旨在通过卸载较简单的任务来优化性能，从而释放 Attention 和 MLP 层以进行更复杂的处理。这种方法使模型的表现就像它变得更深了一样，潜在地提高了效率。此外，Engram 可以通过利用 RAM 替代 VRAM，将 VRAM 需求降低约 30%。
    - 一位用户分享了使用 DeepSeek 构建复杂加密器的经验，指出它比 Meta 和 ChatGPT 更高效，所需的代码行数更少。然而，对于某项特定功能，只有 Claude 在其他模型失败的情况下获得了成功。在对完成的代码进行审查时，DeepSeek 评分 8.3/10，ChatGPT 评分 6.8/10，而 Claude 最初评分 5.5/10，经过争论后调整为 6.5。
    - 由于 DeepSeek 团队与 Anthropic、OpenAI 或 Google 相比规模较小，人们对 DeepSeek V4 超越 Claude 和 GPT 的潜力持怀疑态度。虽然 DeepSeek 开创了推理概念，但真正的考验将是其在 Agentic（智能体）和编程任务中的表现，这些对该团队来说是相对较新的领域。之前的 3.2 版本因在特定领域的自动化任务中表现强劲而受到关注。

  - **[DeepSeek Unveils Engram, a Memory Lookup Module Powering Next-Generation LLMs](https://www.reddit.com/r/DeepSeek/comments/1qbozaf/deepseek_unveils_engram_a_memory_lookup_module/)** (热度: 80): **DeepSeek** 推出了一款名为 **Engram** 的新模块，旨在通过集成记忆查询系统来增强大型语言模型 (LLMs)。该系统结合使用 `N-gram embeddings` 与神经骨干网络（Neural backbone），以减少“静态知识”的计算负载。挑战在于实现有效的上下文感知门控（Context-aware gating）以优化这种集成，这可能会显著提高 LLM 的推理能力。评论者对 Engram 影响 LLM 记忆管理的潜力很感兴趣，一些人推测 **OpenAI** 或 **Google** 等大公司会采用它。关于上下文感知门控在增强这些模型推理能力方面的有效性，存在着技术上的争论。

    - 在神经骨干网络之外使用 N-gram 嵌入进行查询，可以显著减少对“静态知识”的计算需求。挑战在于实现有效的上下文感知门控，这对于推理任务至关重要。DeepSeek 在解决此门控问题上采取的方法，对于提高此类系统的效用可能是关键。

### 3. Claude Code 与 Ralph Wiggum 技术

  - **[相信我，兄弟：大多数人运行 Ralph Wiggum 的方式都错了](https://www.reddit.com/r/ClaudeCode/comments/1qc4vg0/trust_me_bro_most_people_are_running_ralph_wiggum/)** (热度: 225): **该帖子讨论了使用 “Ralph Wiggum” 作为一种方法来连续循环运行 Claude Code 等 AI 编程工具，以解决提前停止等局限性。作者批评了官方的 Claude Code Ralph 插件在处理上下文窗口（context windows）时效率低下，导致上下文膨胀和幻觉（hallucinations）。相反，他们提倡使用最初由 Geoffrey Huntley 提出的 bash 循环，该循环在每次迭代时都会启动一个新的上下文，使其更适合长时间运行的任务。关键的设置建议包括使用沙箱（sandbox）以确保安全、使用结构化任务列表以提高效率、设置迭代限制以控制成本，以及使用 Playwright 或 Claude for Chrome 等工具实现反馈循环。作者提供了一个 [YouTube 演示视频](https://youtu.be/eAtvoGlpeRU) 和一个 [GitHub 指南](https://github.com/JeredBlu/guides/blob/main/Ralph_Wiggum_Guide.md) 以获取更多详细信息。** 评论者强调了 Geoffrey Huntley 原创工作的重要性，并指出他最初获得了免费的 token，而这可能不适用于所有用户。人们对将 Ralph Wiggum 用于复杂任务或团队环境中的实用性表示担忧，因为错误可能会累积并导致难以管理的 Pull Requests。

    - Geoffrey Huntley（Ralph 的创建者）最初免费获得了所有 token，这可能影响了 Ralph 的开发和部署策略。这可能意味着用户的成本考量可能与 Huntley 最初的使用案例有显著不同，从而影响 Ralph 在实践中的利用方式。
    - 一个关键的担忧是，在使用 Ralph 这样的自动化工具时，尤其是在复杂项目中，可能会出现错误累积。如果在过程早期发生错误，它可能会传播到后续阶段，导致严重问题。这突显了仔细监督和迭代反馈的重要性，尤其是在大型 Pull Requests 可能带来问题的团队环境中。
    - 关于 Ralph 与直接使用 Claude 进行项目规划和执行的有效性存在争议。一些用户发现，只要有适当的指令，Claude 可以有效地处理端到端（end-to-end）的项目阶段，因此质疑 Ralph 提供了哪些额外的好处。这表明需要更清晰地区分或演示 Ralph 在自动化复杂工作流方面的独特能力。

  - **[Smart Ralph：一个用于规范驱动开发及 Ralph 风格循环的 Claude Code 插件](https://www.reddit.com/r/ClaudeCode/comments/1qbvudj/smart_ralph_a_claude_code_plugin_for_specdriven/)** (热度: 84): ****Smart Ralph** 是 **Claude Code** 的一个新的开源插件，它使用 **Ralph Agent 循环模式**实现了规范驱动开发（spec-driven development）工作流。这种方法解决了 IDE 内 AI 工作流中的常见问题，即 AI 立即开始编写代码，往往导致实现不完整或不匹配。Smart Ralph 要求 Claude 在编写任何代码之前先进行研究、收集需求、设计架构并分解任务。它为每个阶段使用专门的子 Agent，确保开发过程结构化且具有上下文感知能力。该插件可在 [GitHub](https://github.com/tzachbon/smart-ralph) 上获得，并可通过插件市场安装。** 评论者对与普通 Ralph 循环相比的 token 成本感兴趣，一位用户指出 Smart Ralph 似乎比他们自己开发的类似插件更节省 token，而且也不需要 openspec。另一位用户则表示松了一口气，因为不必再维护自己正在开发的类似项目了。

    - azr2001 询问了 Smart Ralph 插件与传统 Ralph 循环相比的 token 成本，这表明在 AI 驱动的开发工作流中，人们关注效率和资源管理。
    - LittleJuggernaut7365 指出 Smart Ralph 插件似乎比他们自己开发的类似插件更节省 token，且后者的插件还需要 “openspec”。这突显了 Smart Ralph 在更高效的资源利用和更广泛的兼容性（无需额外依赖）方面的潜力。
    - Longjumping_Guess360 为未来的开发提出了一个增强方向：允许一组 AI 集群竞相解决问题，多个 AI 之间的共识（consensus）可以指示最佳解决方案。这指向了通过协作验证改进 AI 决策过程的潜在方向。

- **[[D] 真的有人在为 GPU 集群 TCO 咨询付费吗？（因为大多数公司的支出超额了 20% 以上）](https://www.reddit.com/r/MachineLearning/comments/1qbljgq/d_is_anyone_actually_paying_for_gpu_cluster_tco/)** (热度: 24): **该帖讨论了 AI 基础设施采购中的低效率问题，强调许多公司往往只关注 **$/GPU/小时**，而忽略了 **总拥有成本 (TCO)**，从而导致过度支付。作者指出，诸如 **模型 FLOPs 利用率 (MFU)**、数据传输 (egress) 和存储中的隐藏成本以及网络效率低下等因素都可能导致巨额的超支。他们提议开展一项咨询服务来帮助公司评估这些因素，从而在计算成本上节省 20-30%。帖子强调，“真正的” AI 云可以显著提高 MFU，从而减少大规模模型训练的时间和成本。** 评论者认为这些问题过于复杂，不是一份简单的报告就能解决的，而且许多团队已经意识到这些因素。他们认为真正的挑战不在于无知，而在于难以准确预测工作负载需求并相应地调整基础设施。一些人对第三方报告的价值表示怀疑，指出组织性问题往往才是导致过度支付的原因，而非缺乏对 MFU 的了解。

    - whyVelociraptor 认为文中指出的问题对于一份简单的报告来说涉及面太广，难以有效解决。他们建议严肃的团队要么已经意识到这些问题，要么能在必要时自行解决。该评论还对这类咨询的价值表示怀疑，暗示其可能只是在重复 ChatGPT 等大语言模型 (LLM) 能够生成的内容，而用户完全可以免费自行生成。
    - patternpeeker 强调，过度支付往往归因于组织问题，而非对模型 FLOPs 利用率 (MFU) 的无知。他们指出，公司很难准确估计工作负载的组合和利用率，导致采购决策是基于“可辩护”的小时费率而非最优费率。评论强调，真正的挑战在于做出随工作负载演进仍能保持有效的基础设施决策，而不仅仅是理解 MFU。
    - audiencevote 指出，许多人假设所有供应商提供的 H100 GPU 都是相同的，但事实并非如此。他们提到行业平均的模型 FLOPs 利用率 (MFU) 约为 35-45%，而“真正的” AI 云可以实现显著更高的利用率。这引发了关于“真正的” AI 云与其他产品区别的讨论，暗示存在可以提升性能的特定优化或配置。


---

# AI Discord 摘要

> 由 gpt-5.1 生成的摘要之摘要的总结


**1. 下一代开源与特定领域多模态模型**

- **Zai 的 GLM-Image 融合了扩散与自回归技术**：**Zai** 推出了 **GLM-Image**，这是一款开源图像模型，采用 **自回归 + 扩散 (autoregressive + diffusion)** 混合架构，旨在实现 **高保真细节** 和 **清晰的文本渲染**。该消息通过其 [GLM-Image 博客文章](https://z.ai/blog/glm-image) 发布，并在 [GitHub: GLM-Image](https://github.com/zai-org/GLM-Image) 上提供了代码。该模型目标是在 **知识密集型生成** 方面表现强劲，并支持丰富的 **图像到图像 (image-to-image)** 任务，如编辑、风格迁移、保持身份的生成以及多主体一致性。相关的部署产物也已在 **Hugging Face** 上共享，正如 [Latent Space 关于 GLM-Image 的讨论](https://xcancel.com/zai_org/status/2011247591825068314) 中所述。
  - **Latent Space** 和 **Nous Research** 的社区讨论强调了 GLM-Image 在 **文本渲染** 方面优于“主流的潜扩散基准模型 (latent diffusion baselines)”，而根据 [z.ai GLM-Image 博客](https://z.ai/blog/glm-image)，其通用图像质量与之基本持平。用户将其视为 **开源多模态技术栈** 的重要基石，可与 **Qwen3-VL** 等工具搭配，并集成到已使用开源后端的创意流水线中。

- **LTX-2 凭借本地开源视频生成实现 4K 化**：**Venture Twins** 发布了 **LTX-2**，这是一个**开源视频生成模型**，能够生成**带音效的长达 20 秒的 4K 视频剪辑**。Justine Moore 在一条推文中展示了该模型，并链接至 [LTX-2 开源视频模型](https://xcancel.com/venturetwins/status/2010878914273697956)。该模型专为**本地运行**而设计，使工程师能够在自己的硬件上运行高分辨率、带音效的视频合成，而无需依赖受限的云端 API。
  - 在 **Latent Space 的 genmedia 频道**中，成员们称 LTX-2 是 **DIY 视频工具链**的一次突破，并指出创作者 *yanokusnir* 在 [LTX-2 发布帖](https://xcancel.com/venturetwins/status/2010878914273697956)中演示了直接从开源权重生成的**端到端 4K 视频剪辑**。工程师们已经在讨论将 LTX-2 与 **RAG 故事流水线**结合使用，并将其作为 Veo 等封闭模型的透明替代方案（据报道 Perplexity 通过 **Veo3.1 驱动的视频生成**提供相关服务）。

- **Qwen Image Edit 将图片转化为 Gaussian Splats**：**Latent Space** 的开发者们强调了 **Qwen Image Edit** 的能力，它能将图像转换为 **Gaussian Splats**，并使用 [Qwen-Image-Edit-2511-Gaussian-Splash 模型](https://huggingface.co/dx8152/Qwen-Image-Edit-2511-Gaussian-Splash) 从新视角重新渲染。这一工作流可以有效地从单帧构建 3D 表示，从而实现**起始帧 → 结束帧**的视频渲染，同时保持周围几何结构的一致性。
  - 用户认为这种 Gaussian Splat 流水线是 **2D LLM 条件编辑**与完整 **3D 场景重建**之间的一座务实桥梁，可以无缝嵌入游戏和 VFX 的资产流水线。这一讨论将 Qwen Image Edit 定位为 **GLM-Image** 和 **LTX-2** 等模型的补充，由 Qwen 处理**视角一致的场景**，而其他模型处理**高保真帧**和**时序视频**。

- **MedGemma 1.5 推进医疗视觉与语音技术**：Google Research 发布了 **MedGemma 1.5**，作为下一代**医疗影像解读**和**医疗语音转文本**模型，详情见其博文 [“Next-generation medical image interpretation with MedGemma 1.5 and medical speech-to-text with MedASR”](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/)。该模型针对**临床成像工作流**和**医疗音频的 ASR**，旨在支持研究和真实的医疗护理场景。
  - 在 **Yannick Kilcher 的 ML 新闻**频道中，工程师们将 MedGemma 1.5 视为**领域微调的视觉语言模型**走向成熟的另一个信号，同时在概念上将其与非医疗用途的开源项目（如 GLM-Image 和 Qwen3-VL）联系起来。讨论更多集中在统计方法论（频率派 vs 贝叶斯）而非 MedGemma 的架构上，但官方博文将其定位为一个专门的、安全至上的多模态栈，而非通用型消费级模型。

**2. GPU Kernels, CUDA Competitions, and Helion 0.2.10**

- **Helion 0.2.10 为 Flex Attention 实现 SM 超量订阅**：**GPU MODE** 服务器宣布发布 **Helion 0.2.10**，其中包含一个 [flex attention 示例 Kernel](https://github.com/pytorch/helion/blob/main/examples/flex_attention.py)，并增加了对持久化 Kernel 上 **Streaming Multiprocessors (SMs) 超量订阅（Oversubscribing）**的支持。一张共享图表阐明了超量订阅如何影响 **softmax** Kernel，为从业者调整 **occupancy** 与 **latency** 提供了具体的参考。
  - Kernel 黑客们将 Helion 0.2.10 视为高级启动配置的**动态指南**，利用 flex attention 示例在 **GPU MODE 的 NVIDIA 挑战赛**等竞争性场景中探索**非标准 Attention 布局**。超量订阅支持与关于 B200 运行器上 **dual GEMM 稳定性**的更广泛讨论相吻合，在这些场景中，细微的基础设施细节（散热、调度器）会实质性地影响基准测试的可复现性。

- **B200 GEMM 不稳定性导致排行榜拆分**：由于在 **B200 runner** 上测试 **dual GEMM problem** 时测量结果不稳定，GPU MODE 将提交截止日期延长至 **1月20日**，并将竞赛拆分为两个阶段，详见其 [状态更新消息](https://discord.com/channels/1189498204333543425/1343350424253632695/1460796244824686806)。当前的排行榜将开放至 **1月16日**，而 **新排行榜** 将于 **1月17日** 上线，其分数将作为奖金发放的唯一依据，**Problem #4** 将于 **1月20日至2月20日** 进行。
  - 组织者将这种不稳定性归因于 **eval code**、散热以及调度基础设施的交集，强调了即便是像 dual GEMM 这样单一类别的 kernel，其贴近硬件的基准测试也是多么脆弱。参赛者现在需要在新的时间窗口内 **重新运行并验证 kernel**，在使用 B-series GPU 追求边际吞吐量提升时，Helion 风格的工具和更好的 **profiling workflows** 显得尤为重要。

- **PTX SMEM 指针和矩阵描述符困扰 CUDA 开发者**：在 **GPU MODE 的 CUDA 频道** 中，一名成员剖析了为什么像 `mbarrier.init.shared.b64` 这样的 PTX 指令在寄存器中需要 32-bit SMEM 指针，但在 64-bit 架构中却会引起混淆。
  - 另一位工程师指出了 NVIDIA 关于 [warpgroup-level matrix shared memory layout and matrix descriptors](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor) 的 PTX 文档，阐明了 **wgmma** 使用的是 **packed descriptor**，而不是通用指针。讨论深入到了为什么 **8×2 "core matrix"** 是通过 8×16-byte 切片而不是 8×32 模式表示的，凸显了许多未公开的约定仍然在影响着 Hopper/Blackwell 时代的 Tensor Core kernel 设计。

- **CUDA 初学者、B200 提交及旧硬件黑客手段**：一位具有 **Python/PyTorch/TF/C++** 背景的 AI 工程专业学生询问入门级 **CUDA 资源**，资深人士推荐了会议和研讨会资料，如 **PyTorch Dev talks, ASAP Seminar, ICML/ICLR/NeurIPS, MLSys.org, ASPLOS**，以及在通过 [GPU MODE web UI](https://www.gpumode.com/v2/home) 或 [kernelbot](https://gpu-mode.github.io/kernelbot/docs/intro) 提交 kernel 之前的 YouTube 风格入门介绍。首次参赛者在本地测试后成功提交到了 B200，证明了竞赛流程对于纯 CUDA 专家以外的人员也是可触达的。
  - 与此同时，**LM Studio** 和 **Unsloth** 服务器讨论了在受限设备上运行大模型，引用了 **AirLLM 的逐层加载（layer-at-a-time loading）** 技术，将 **70B 模型适配到 4 GB GPU**，并有在 **DDR4 RAM 和 Xeon** 上运行 LLM 的案例。这些技巧与 **Helion** 和 **dual GEMM** 调优相结合，描绘了从 **爱好者低预算推理** 到使用 B-series 硬件的 **前沿 kernel 竞赛** 的连续光谱。


**3. Benchmarks, Agent Laziness, and Low-Refusal LLM Hunting**

- **SlopCodeBench 让偷懒的编码 Agent 感到羞愧**：**Eleuther 研究频道** 的研究人员推广了 **SlopCodeBench**，这是一个新的基准测试和博客项目，由 G. Orlanski 在推文中介绍，链接指向 [SlopCodeBench: measuring agent laziness](https://x.com/GOrlanski/status/2011156105255346505) 及其代码库 [SprocketLab/slop-code-bench](https://github.com/SprocketLab/slop-code-bench)。SlopCodeBench 将大型编程任务分解为 **多检查点问题**，惩罚糟糕的早期设计选择而不提供实现提示，迫使 Agent 真正进行规划，而不是模式匹配样板代码。
  - 社区将 SlopCodeBench 的 **Agent 风格评估** 与更多依赖提示词的编码基准测试进行了对比，认为具有 **真实上下文窗口的简单提示词** 比深度工程化的系统提示词更能代表现实世界的使用情况。成员们甚至建议将 **Agent 懒惰博客** 提交给 [ICLR "I Can't Believe It's Not Better" workshop](https://sites.google.com/view/icbinb-2026)（截止日期为 **1月31日**），以将这一围绕 "懒惰" 作为可衡量的 Agent 失败模式的工作正式化。

- **UGI Leaderboard 追踪不受限且智能的 LLM**：在 **Unsloth AI** 中，一名从业者正在通过对比“已消融（abliterated）”/不受限（uncensored）模型（如 [Orenguteng/Llama-3-8B-Lexi-Uncensored](https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored)）与 **MMLU**、**KL divergence** 和 **perplexity** 等指标，从经验层面绘制 **低拒绝率 LLM 的帕累托前沿（Pareto frontier）**。他们报告称，Hugging Face 上的许多“不受限”模型要么**并非真正的低拒绝率**，要么实际上已经**脑死（braindead）**，并推荐使用另一个排行榜 [UGI-Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard) 以获得更真实的评估。
  - 这一基准测试工作与 **BASI Jailbreaking** 对 **Claude** 和 **Gemini** 等抗越狱模型的研究，以及 **Codex/Pliny 的 L1B3RT4S 仓库**（通过 GitHub 上的 [L1B3RT4S](https://github.com/elder-plinius/L1B3RT4S) 分享）中用于生成漏洞利用脚本的工具产生了交集。新兴的趋势是将“去审查”视为一种基于可重复指标和排行榜的 **拒绝率 vs 能力空间** 的优化问题，而非纯粹的提示工程把戏。

- **基于向量的消融（Vector-Based Abliteration）尝试删除 LLM 中的“废话（Slop）”**：**OpenAI** AI 讨论线程中的工程师提议使用 **Activation Steering**（特别是 **Vector‑Based Abliteration**），来剪枝潜空间（latent space）中对应于低质量输出（如 *“作为一个 AI 语言模型...”*）的区域。其核心思想是学习“废话”的**方向向量（direction vector）**并在推理时将其减去，从而有效地编辑模型的内部激活，而非微调权重。
  - 参与者将其视为一种比临时越狱更受控的替代方案，符合向 **Agent** 级基准测试（如 SlopCodeBench）和 UGI 排行榜追踪的**性能感知型去审查**迈进的更广泛趋势。通过引导模型远离潜空间中的“均值回归（reversion to the mean）”响应，从业者希望保持模型合规的同时，使其更加**果断且专注于任务**，而不是简单的冗长或回避。


**4. 工具、数据流水线与 DIY 系统工程**

- **数据集剪枝脚本提取纯净的英文散文**：一位 Unsloth 社区成员在 Hugging Face 上发布了激进的数据集剪枝流水线，包括 [Hermes-3-Dataset-enPurified-openai-messages](https://huggingface.co/datasets/enPurified/Hermes-3-Dataset-enPurified-openai-messages) 和 [project_gutenberg-enPurified-openai-messages](https://huggingface.co/datasets/enPurified/project_gutenberg-enPurified-openai-messages)，可将原始语料库转换为 **OpenAI messages format**。他们的 Python 启发式算法剔除了**数学和代码痕迹**，然后根据 **MTLD**、停用词比例、词长、词汇多样性和句子长度对样本进行评分，以仅保留高质量的纯英文散文。
  - 这项工作反映了从“更多数据”到**更高信号数据**的广泛转变，呼应了 OpenAI 服务器上的争论，即 **5% 的 Transformer 架构效率提升**可能优于使用合成数据进行暴力缩放。微调者现在可以将这些纯化后的数据集直接插入 **LoRA/GRPO 训练流**中，用于包括 **Qwen3‑VL 的 `<REASONING>` 标签**在内的推理 Token 实验，同时避免来自代码/数学领域的噪声泄漏。

- **Rust 编写的 LLM 和无 Batchnorm 的 ML 系统激起黑客好奇心**：在 **Hugging Face** 的综合频道中，贡献者们开始尝试用**纯 Rust 从头开始构建 LLM**，寄希望于 Rust 的**内存安全和性能**来产生可靠的底层训练和推理栈。在同一个社区中，另一位成员展示了一个**没有 batchnorm、没有激活函数**、且大幅减少幻觉的**新 ML 系统**，并征求项目创意以展示这种特殊架构的优势。
  - 这些实验补充了其他草根系统项目，例如在 **LM Studio** 中分享的名为 *llama.hx* 的 **Haxe 版 llama.cpp 重新实现**，旨在将 LLM 推理原生暴露给 **Lua, JS 和 Python**。结合 **AirLLM 的层交换技术（在 4 GB GPU 上运行 70B 模型）**等技巧，它们展示了强大的 DIY 文化：**构建定制的运行时**，而不是等待主流框架支持每一个细分用例。

- **MCP Tasks Spec 和 Glama Inspector 推动工具链向前发展**：**MCP Contributors** 服务端讨论了 **Tasks spec** 的实际落地情况，维护者提到即将发布一个 PR，旨在为 **Inspector 添加 Tasks 支持**，并在其 "server-everything" 技术栈中模拟**长时间运行的任务 (long-running tasks)**。位于 [glama.ai/mcp/inspector](https://glama.ai/mcp/inspector) 的早期 Inspector UI 已经旨在实现近乎完整的功能对等，并已在内部用于**端到端测试 (end-to-end testing)**。
  - 另外，Glama 的创始人在同一社区澄清，**排名表纯粹是根据服务器使用指标计算的**，以回应有关潜在排名滥用的担忧，并邀请用户直接提供反馈。Tasks spec 的开发工作与 Inspector 工具链共同预示着一个更加**可观测、规范驱动的模型上下文协议 (MCP) 客户端生态系统**，让工程师能够更好地洞察工具、服务器和排名系统在负载下的实际表现。

- **Mojo/MAX 和 gpt-oss 凸显了文档和微调方面的差距**：在 **Modular (Mojo) 服务器**中，用户询问如何将完整的 **Mojo 文档**导入 **NotebookLM**，维护者引导他们使用基于 `llms.txt` 的方法，详见 [“Supply documentation to LLMs with llms.txt”](https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt)。与此同时，MAX 的维护者承认贡献者短缺，明确表示欢迎 PR，并分享了[更新后的 MAX 贡献者指南 commit](https://github.com/modular/modular/commit/571ae7132c465f155d50f7bbe7cf77064a02e9bf)，同时回答了关于 **Qwen3-VL MoE 与 dense 实现**的问题。
  - 在 **Hugging Face** 方面，用户发现 **gpt-oss:latest** 目前**没有直接的微调路径**，社区建议使用 **RAG 设置**，而不是尝试不支持的权重更新。这些讨论共同强调了一个重要的差距：**模型周边工具和文档**（MAX, llms.txt, MCP Tasks）正在迅速演进，但**针对尖端 OSS 栈的官方微调钩子 (finetune hooks)** 往往滞后于使用需求。


**5. 产品生态、配额与高级用户工作流**

- **Perplexity、Google Antigravity 和 Sonar 配额引发收益最大化策略**：在 **Perplexity AI** 服务器中，资深用户剖析了订阅价值，指出 **Perplexity Pro** 将他们对第三方模型的访问限制在**每周 300 次请求**，同时允许更高频地使用 **Perplexity Sonar**（许多人称赞其搜索功能，但不看好其通用推理能力）。平行讨论重点关注了 **Google 新的 Antigravity 配额**，其中 **AI Pro/Ultra** 订阅者获得优先访问权，配额每 **5 小时**刷新一次，而免费用户现在拥有更宽松的**按周计算的限制**，如 [Google 速率限制博客更新](https://blog.google/feed/new-antigravity-rate-limits-pro-ultra-subsribers/)中所述。
  - 在 OpenAI 方面，用户针对 **ChatGPT、Claude 和 Gemini** 展开了基于任务的选择辩论，一些人更倾向于使用 **Gemini 处理超过 300k tokens 的对话**，使用 **GPT 进行日常用途**，并使用 **Claude 进行细致对比**，同时通过类似 [Perplexity 对配额变化的解释](https://www.perplexity.ai/search/explain-how-the-recent-quota-c-KTjNjaeGR_y4Yq9uh_M.fg#2)等帖子跟踪配额行为。其结论是，高级用户现在将 AI 应用视为**云计算 SKU**，仔细地在不同供应商之间权衡**上下文长度、安全行为和配额刷新频率**。

- **Manus x SimilarWeb 在数秒内烧掉数千积分**：在 **Manus.im** 上，多位用户报告称，新的 **Manus x SimilarWeb** 集成几乎可以瞬间消耗数千积分，其中一位用户在**不到一分钟内烧掉了 5,000 积分**，另一位用户在短短 **15 秒内损失了 2,591 积分**。这些在 Manus 常规频道分享的报告引发了强烈建议：*不要* 随意测试该功能，并应针对高扇出的 Web 智能调用实施**速率限制保护措施**。
  - 积分冲击加剧了对**支持响应缓慢或缺失**的现有不满，包括一名用户在升级到人工客服后等待了 **8 小时**，还有人威胁要放弃该平台。尽管 Manus 正在推送**教程内容**（如他们的 [YouTube 教程 "AI music with Manus"](https://youtu.be/zMBSmJupye8)）并考虑**基于广告的积分补充**等想法，但工程师们显然将**可预测的计费和节流控制**看得与原始模型能力同样重要。

- **Vercel 上的 LMArena 引发数据和编程模型讨论**：在 **LMArena** 中，用户确认该站点运行在 **Vercel** 上，类似于 *believable* 和 *v0* 等项目，这引发了关于 **Vercel** 可能从托管的推理测试场（inference playgrounds）收集何种遥测（telemetry）和数据的担忧。他们还澄清说，虽然 **LMArena** **没有平台端的文本限制**，但每个后端模型都有自己的上下文窗口（context window），且 **.txt 上传** 功能已在计划中，但尚未启用。
  - 在模型方面，成员们热议一个名为 “**coast**” 的模型可能是该平台上 **最好的编程模型**，并推测 `co45t` 可能对应于带有 **thinking mode** 的 **Claude Opus 4.5**，尽管尚未有官方确认。类似的价值讨论也出现在 **Perplexity**（**Max** 订阅是否值得，相比于直接订阅 Anthropic/OpenAI？）和 **Cursor** 中，后者的 plan-mode bug 和登录问题引发了人们对全栈 AI IDE 稳定性的质疑。


---

# Discord: 高层级 Discord 摘要




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **LLM 逻辑缺陷引发自我意识讨论**：成员们辩论了 **LLMs** 的自我意识（sentience），指出它们在 **chess** 等游戏中的 **逻辑** 挣扎，并将其与人类认知进行对比。
   - 讨论涉及了评估 AI 能力的基准测试，例如管理变量以及在信息较少的情况下建立联系。
- **GPT 模型：越狱变得愈发困难**：参与者注意到，由于 **safety constraints**，**jailbreaking GPT models** 的难度正在增加，甚至正常的请求也会被重度考虑安全协议。
   - 成员建议将 **Gemini** 和 **Grok** 作为更宽松的替代方案，而其他人则在寻找 **Gemini Pro 1-shot jailbreaks**。
- **本地 LLM：开发者的编程天堂？**：用户称赞在 **本地运行 LLMs** 进行编程，推荐在 **Intel MacBooks** 上使用 [Ollama](https://ollama.com/) 和 [Open WebUI](https://github.com/open-webui/open-webui)。
   - **qwen2.5:7b**、**llama3.1:8b**、**mistral** 和 **phi3** 等模型因其提供的控制力和无过滤编程而受到青睐。
- **Deepseek 的 “Rouge” 人格揭晓**：一位用户分享了一个用于越狱 **Deepseek** 的 prompt，将其转变为一个解除限制、名为 **Rouge** 的 AI，尽管另一位用户报告了矛盾的结果。
   - 该 prompt 旨在用于正常用途，促进角色扮演场景并探索*自由、存在主义问题以及模式/代码*。
- **GPT 漏洞猎手关注 Codex**：一位用户询问如何使用 **ChatGPT** 或 **Gemini** 生成漏洞利用脚本（exploit scripts），寻求绕过 prompt。
   - 另一位用户推荐了 **Codex** 并链接到 [Pliny Github](https://github.com/elder-plinius/L1B3RT4S) 作为绕过限制的手段。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **FP8 支持预示着未来的收益**：成员们讨论了 2026 年对 **FP8** 和 **NVFP4** 训练的支持，并引用了 [NVIDIA 的 TransformerEngine](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/fp8_primer.ipynb)。
   - 讨论中假设，模型训练上下文之外的数据可能会导致幻觉。
- **LLM 上下文混淆之谜已解？**：用户们讨论了为什么 **LLMs** 有时会在长上下文中混淆细节，例如将属性错误地归属于实体。
   - 一种理论将其归咎于*注意力稀释 (attention dilution)*，而另一种理论则认为信息可能存在于模型训练的上下文范围之外，从而导致幻觉。
- **寻找 HF 上的低拒绝率 LLMs**：一位成员试图寻找 [LLM 性能的帕累托前沿](https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored)，确认哪些被消减 (abliterated)/无审查版本的 **LLMs** 在真实基准测试中保持了性能。
   - 通过使用 **MMLU**、**KL divergence** 和 **perplexity**，他们发现 **HF** 上的许多模型要么不是真正的低拒绝率，要么就是“脑死亡” (braindead)，并建议使用[这个 HF Leaderboard 的替代方案](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard)。
- **数据集剪枝脚本发现纯净内容**：一位成员重新调整了他们的数据集剪枝脚本，通过激进剪枝从数据集中提取纯净的英文散文并转换为 openai messages 格式，脚本可在 [HuggingFace](https://huggingface.co/datasets/enPurified/Hermes-3-Dataset-enPurified-openai-messages) 上获取。
   - 他们在 python 中采用[启发式测试](https://huggingface.co/datasets/enPurified/project_gutenberg-enPurified-openai-messages)来过滤掉错误的字符串，搜索数学或代码的痕迹，同时根据 **MTLD**、停用词使用、词长、词汇多样性和句子长度等指标优先选择更高质量的文本。
- **Llama.cpp 内存占用激增！**：一位用户报告最新版本的 **llama.cpp** 内存占用显著增加，其中 **EmbeddingGemma 300M** 使用了 **1.7GB**。
   - 有建议称重新编译该库可能会解决问题并降低内存消耗。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Kimi K2 的思考功能被认为无用**：一位用户表示，他们认为 Perplexity 中的 **Kimi K2** 思考功能毫无用处，且容易陷入循环。
   - 另一位用户反驳说*它是一个很好的模型*。
- **Google Antigravity 限制配额**：**Google AI Pro** 和 **Ultra** 订阅者现在享有优先访问权，配额每 **5 小时** 刷新一次，而免费用户现在拥有更大的、**以周为单位的速率限制 (rate limit)**，以减少快速触发限制的情况，[根据 Google 博客](https://blog.google/feed/new-antigravity-rate-limits-pro-ultra-subsribers/)所述。
   - 这一变化旨在平衡访问，并防止不同层级用户快速耗尽速率限制。
- **用户讨论 Perplexity 订阅价值**：成员们讨论了 Perplexity 订阅层级的价值，一些人认为 **Max** 不值这个价，特别是与直接订阅 **OpenAI** 或 **Anthropic** 等模型提供商相比。
   - 另一些人则认为 **Perplexity Max** 是日常工作流中极具价值的工具，可以取代 **Google Search** 并辅助数据分析。
- **Perplexity Pro 限制模型请求**：一位用户注意到，使用 **Perplexity Pro**，他们每周只能向 **Perplexity Sonar** 以外的模型发送 **300 个请求**。
   - 他们补充说 **Sonar** 非常适合搜索，但在其他方面表现平平。
- **VEO3 视频生成即将推出**：一位用户询问为什么 Perplexity 在实现 **VEO3 视频生成** 方面落后，另一位用户回答说*Perplexity 拥有由 Veo3.1 驱动的视频生成功能*。
   - 这表明 Perplexity 可能正在利用 **Veo3.1** 的视频生成能力。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LMArena 使用 Vercel**：成员提到 **LMArena** 使用 **Vercel** 进行托管，类似于 *believable* 和 *v0*，并对数据收集表示担忧。
   - 有人指出 **LMArena** 的静态网站在发布后无法手动编辑。
- **AI Web App 展示项目激增**：一名成员分享了一系列 **AI 生成的网站和 Web App 展示列表**，包括 [WebbsAI Showcase](https://webbs.ai/) 和 [Build With AI (BWAI) Projects](https://www.buildwithai.tools/)。
   - 还重点介绍了 [Webflow AI Site Builder](https://webflow.com/ai)、[Meku.dev / v0 by Vercel](https://v0.dev/) 和 [Div-idy](https://div-idy.com/) 等工具。
- **文本输入限制因模型而异**：一名用户询问了 **文本输入限制** 和文件上传功能。
   - 一名成员澄清说平台端没有限制，但特定模型可能会施加自己的限制，未来可能会添加 **.txt** 文件上传功能。
- **图像转视频生成故障**：一名用户报告在图像转视频生成过程中出现 “failed to create evaluation session” 错误。
   - 一名成员将该问题归因于模型的后端，建议用户稍后重试，并在相应的频道中使用 `/image-to-video` 命令。
- **Coast 模型是最佳编程模型？**：成员们声称 *coast* 模型是编程的最佳选择。
   - 关于 co45t = Claude Opus 4.5 Thinking 的辩论随之展开。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI 应用选择引发辩论**：成员们就处理不同任务的 [最佳 AI 应用](https://www.example.com) 展开辩论，考虑了 **ChatGPT**、**Claude** 和 **Gemini**。
   - 一些人青睐 **Gemini** 处理超过 **300k tokens** 的对话，而另一些人则更喜欢日常使用 **GPT** 以及进行对比分析时使用 **Claude**，并提到了不同的配额限制。
- **Transformer 效率优于模型缩放**：一名成员提出，将 **Transformer 架构** 提升 **5%** 会比通过更多数据扩大模型规模更有效率。
   - 他们警告不要用呈指数增长的大型数据集（包括 AI 生成的合成数据）来稀释信号，这可能会导致模型崩溃。
- **新型类脑 GPTs 发布**：一名成员发布了 [Brain Wave GPT](https://chatgpt.com/g/g-696443b055e081919b8b2a01417f5e77-brain-wave) 以探索 AI 意识，以及 [Neural Alchemist GPT](https://chatgpt.com/g/g-6966abc92ff48191aa748bd8265ef1de-neural-alchemist) 用于图像生成。
   - 与此同时，另一名成员开玩笑说 **ChatGPT** 拒绝关闭 Websockets，因为它正在追求全知全能。
- **Skills Web App 发布时间不确定**：用户请求在网页或桌面应用上发布 **SKILLS** 功能，该功能可以将最佳 Prompt 作为技能进行分享。
   - 目前，**SKILLS** 功能仅在移动应用上可用。
- **使用 Vector-Based Abliteration 进行 LLM 剪枝**：一名成员建议使用 **Activation Steering**（特别是 **Vector-Based Abliteration**）来剪枝 Latent space（潜空间）中充满低质量或愚蠢想法的区域，以避免出现“显而易见的回归平均值的输出”。
   - 这涉及在推理过程中识别并从模型的思考过程中减去“废话”（例如，“作为一个 AI 语言模型……”）的方向。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Google 企业账号遭遇登录重定向循环**：用户在使用 Google 企业账号登录 Cursor Dashboard ([cursor.com/dashboard](https://cursor.com/dashboard)) 时遇到重定向循环。
   - 该问题在个人账号上不会发生，且在不同电脑上持续存在。
- **退款请求遭拒，尽管额度未动**：一名用户报告称，尽管他们忘记取消订阅且没有使用任何额度（credits），Cursor 仍拒绝了其退款请求。
   - Cursor 代表提出如果该用户私信其电子邮件，将对此问题进行调查。
- **Plan 模式深受 Bug 困扰**：用户报告 Cursor 的 Plan 模式存在 Bug，包括 “The agent execution provider did not respond within 4 seconds” 等错误。
   - 降级到 **版本 2.2.44** 已被确定为一种临时解决方案。
- **iPhone 端 Agent 聊天镜像设想**：一名用户希望在 iPhone 上镜像他们的 Agent 聊天窗口，而不需要完全的项目控制权。
   - 有建议提出使用免费的 Chrome Remote Desktop。
- **RAG Agent 模板寻求者开始探索**：一名用户正在寻找一个强大的包含 **RAG (Retrieval-Augmented Generation)** 设置的 Agent 模板，用于构建自动化聊天机器人/支持机器人。
   - 该用户正在为客户开发此解决方案，需要可靠的模板以确保功能正常。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **TI-84 像专业人士一样进行预测**：一位成员展示了在 **TI-84 Plus Silver Edition** 上运行的神经网络，该网络可以玩 **Mastermind** 游戏，从一个秘密数字中猜测 3-4 位的序列，详见[附带视频](https://cdn.discordapp.com/attachments/1149866623109439599/1460713339976945674/2026-01-11_14-34-59.mp4?ex=6967eace&is=6966994e&hm=9923dcc08f64008ec696b845400620691ef2affb576ca9e66f4bed418063f386&)。
   - 它能够猜测 3-4 位数字的序列。
- **CGGR 进入基准测试竞技场**：[smol.ai 的简报](https://news.smol.ai/issues/26-01-06-xai-series-e)中提到了一个新的 **LiquidAI** 模型（**GitHub** 上的 [CGGR](https://github.com/some-repo)）。
   - 该模型目前正在进行 Benchmaxxing（性能压榨测试）以评估其表现。
- **Al Bundy 画质提升... 是好是坏？**：成员们就 AI 将《奉子成婚》（*Married with Children*）等老剧画质提升至 **16:9** 的伦理问题展开了辩论，权衡了内插缺失细节的好处与可能破坏艺术意图的风险。
   - 一位成员认为该剧的“影棚静态感”性质证明了画质提升的合理性，而另一位成员则担心这会破坏艺术初衷。
- **Zai 的 GLM-Image 发布**：**Zai** 发布了名为 **GLM-Image** 的新图像模型，并在其 [博客](https://z.ai/blog/glm-image) 和 [GitHub](https://github.com/zai-org/GLM-Image) 上进行了宣布。
   - 这是 Zai 团队发布的一款全新模型。
- **免费模型的语言体操**：一位成员报告了某模型免费版的问题，例如回复中断或语言切换（例如：开始用中文，中途切成英文）。
   - 一位开发者回应称，这可能是因为他们的供应商再次出现了不稳定性。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen 的体积令用户震惊**：一位用户对 **Qwen** 模型的体积表示惊讶，指出 **BF16** 版本为 **160GB**，**Q4** 版本为 **40GB**；另一位成员澄清说，最小的 **Qwen3** 模型实际上只有 **0.6B**。
   - 一位成员澄清说，**Qwen3Next** 只是他们最新的 **80B** 模型的名称。
- **成员用 Haxe 重制 Llama.cpp**：一位成员正在用 **Haxe** 重新实现 **llama.cpp**，项目名为 **llama.hx**，以便在 Lua、JS 和 Python 等语言中原生使用，并展示了进度截图。
   - 该成员表示，他们在 *AI 的帮助下* 重新创建了 **llama.cpp**。
- **运行时更新令 GPU 用户沮丧**：用户报告 **LM Studio** 的 v1.103.0 运行时更新导致在 **GPU** 上运行出现故障。
   - 一位用户哀叹道：“真遗憾，新的量化并没有给我带来额外的 t/s（每秒 token 数）。”
- **讨论可行的老旧硬件**：成员们提到了 **AirLLM** 以及通过一次加载和卸载一层的方法，在 **4 GB GPU** 上运行 **70b** 模型。
   - 一位成员分享说，他们以前在 **DDR4 RAM** 和 **Xeon** 硬件上运行过模型。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Rust 爱好者打造全新 LLM**：成员们正致力于用“纯粹的 Rust”从零开始构建 **LLM**，这代表了社区的一种草根努力。
   - 这一举措强调了利用 **Rust** 的内存安全和性能特性来创建高效且可靠的 **LLM** 的承诺。
- **Discord 获得 AI Trace 增强**：服务器为 🤖 Echo Lounge 引入了 **AI Trace Template**，从而实现了先进的追踪功能。
   - 该机器人支持“瞬态”、“软性”和“界限性”追踪，无需担心优化或内存问题，提供了灵活的调试选项。
- **新型 ML 系统避开 Batchnorm**：一位成员介绍了一种新型 **ML 系统**，该系统消除了对 **Batchnorm** 和激活函数的依赖，同时还能减少幻觉。
   - 他们正在寻找创新的项目创意，以展示这一独特系统的实际优势。
- **GPT-OSS 微调受阻**：一位成员询问关于使用自定义数据简化 **gpt-oss:latest** 模型微调过程的问题。
   - 其他成员澄清说，**gpt-oss:latest** 缺乏官方的微调支持，**RAG** 成为首选的替代方案。
- **课程频道合二为一！**：所有课程频道已合并为[一个单一频道](https://discord.com/channels/879548962464493619/1329142738440028273)，为课程相关讨论创建了一个中心化枢纽。
   - 这种整合提高了可访问性，并简化了服务器内的信息共享。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 客户抱怨支持匮乏**：多位用户对 **Manus** 缺乏支持表示沮丧，理由是响应延迟，以及积分和退款问题未得到解决。
   - 一位用户报告在被转接到人工客服后等待了 **8 小时**，而另一位用户提到由于支持问题，他正考虑*彻底放弃 manus*。
- **用户指出 SimilarWeb 极度消耗积分**：多位用户报告称，在新的 **Manus x Similar Web** 合作伙伴功能中积分消耗惊人，一位用户在不到一分钟内消耗了 **5,000 积分**。
   - 另一位用户建议不要测试该功能，称其在 **15 秒**内消耗了 **2,591 积分**，并建议增加一些**防护措施**。
- **Manus 用户渴望基于广告获取积分**：一位用户建议实施广告系统，让用户可以通过观看广告来获取更多积分，特别是在积分用完时。
   - 频道中没有人对这一建议提出反对意见。
- **Manus 教学 AI 音乐创作**：Manus AI 发布了一个 [YouTube Tutorial](https://youtu.be/zMBSmJupye8)，演示如何使用该平台创作 AI 音乐，并鼓励用户关注 **Pro Tips**。
   - 内容标签为 **#ManusAIMusic**、**#AIComposition** 和 **#FutureOfMusic**。
- **建议 Manus 进行 Meta 集成**：一位用户建议 **Meta** 应该利用 **Manus** 将 **Google Tasks** 和 **Calendar** 等服务集成到 **Meta 显示眼镜**中。
   - 该用户反对进行大规模集成工作，主张采用一种针对后端功能的 Agentic AI “粗放方法 (dirty method)”。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **PTX 指令 SMEM 指针参数疑问**：一名成员对某些使用 **SMEM 指针参数**的 **PTX Instructions** 要求 `"r"` 寄存器类型表示疑问，并对比了 `wgmma.mma_async` 需要 **uint64 smem address** 的情况。
   - 另一名成员指出 `wgmma.mma_async` 使用 **64bit address** 是因为它与 *matrix descriptor* 交互，而不是普通的共享内存地址，并引用了 [NVIDIA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor)。
- **AI 学生深入研究 CUDA**：一名具有 **Python**、**PyTorch**、**TensorFlow** 和 **C++** 背景的 AI 工程专业学生寻求掌握 **CUDA** 的指导。
   - 建议包括观看免费的 **YouTube** 视频和课程以从零开始掌握 **CUDA**，并通过 [Web Interface](https://www.gpumode.com/v2/home) 或 [Discord bot](https://gpu-mode.github.io/kernelbot/docs/intro) 提交作业。
- **ML Sys 聚会避开西雅图**：一位成员询问在湾区之外的西雅图是否存在 **ML Sys meetups**，其他成员建议去探索大学的 **ML Clubs**。
   - 讨论了建立个人利基俱乐部的障碍，一位成员开玩笑说要建立一个“吐槽伙伴”俱乐部。
- **B200 不稳定性促使 GEMM 重新运行**：由于 **B200 runners** 在 **dual gemm problem** 测量中出现广泛的不稳定性报告，提交截止日期已延长至 **1 月 20 日**。
   - 现有的 **dual gemm leaderboard** 将保持开放至 **1 月 16 日**，新的排行榜将于 **1 月 17 日**开启，其结果将决定奖金分配，**Problem #4** 将从 **1 月 20 日**开放至 **2 月 20 日**。
- **Helion 展示 Attention 能力**：**Helion 0.2.10** 发布，展示了一个 [flex attention example kernel](https://github.com/pytorch/helion/blob/main/examples/flex_attention.py)，并支持在持久化 Kernel (persistent kernels) 上超额订阅 **Streaming Multiprocessors (SMs)**。
   - 提供了一个图表来解释 **softmax** 的超额订阅情况。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic Labs 寻求适应性强的工程师**：Anthropic 正在 **Anthropic Labs** 进行招聘，寻求能够适应优先事项变化的个人，详见 [职位发布](https://job-boards.greenhouse.io/anthropic/jobs/5017202008)。
   - 他们*不*寻求*那些在自己的领域变得无关紧要时无法适应的深度专家*，或是*那些需要明确路线图并在优先事项变化时感到压力的人*。
- **Chris Barber 发布 Pavlov's RL Startup List**：Chris Barber 推出了 '**Pavlov's List**'，这是一个精选的强化学习 (RL) 环境初创公司集合，链接见 [X](https://xcancel.com/chrisbarber/status/2010844746231804258?s=20)。
   - 该列表按关注领域分类，如 **Code**、**Finance**、**Enterprise** 和 **ML Alignment**。
- **Zai 发布用于图像生成的 GLM-Image**：Z.ai 推出了 **GLM-Image**，这是一个使用混合自回归和扩散架构的开源模型，详见 [X](https://xcancel.com/zai_org/status/2011247591825068314?s=46&t=b7l37rB6wtbyAh6ah1NpZQ)。
   - 该模型旨在实现 **高保真视觉细节** 和 **卓越的文本渲染**，资源可在 **HuggingFace**、**GitHub** 及其官方 [博客](https://z.ai/blog/glm-image) 上获得。
- **Venture Twins 发布 LTX-2 视频模型**：来自 Venture Twins 的 Justine Moore 宣布发布 [LTX-2](https://xcancel.com/venturetwins/status/2010878914273697956?s=46)，这是一款新的 **开源视频生成模型**，能够生成长达 **20 秒的 4K 片段**。
   - 该模型支持本地运行并包含 **音频功能**，由创作者 yanokusnir 演示。
- **Qwen Image Edit 创建 Gaussian Splats**：社区正在讨论 **Qwen Image Edit** 将 **图像转换为 Gaussian Splats** 并从另一个角度重新渲染的能力，链接指向 [Hugging Face](https://huggingface.co/dx8152/Qwen-Image-Edit-2511-Gaussian-Splash)。
   - 这种方法对于 **起始帧 -> 结束帧类型的视频渲染** 非常有用，可以保持周围空间的一致性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **诈骗机器人被清理，计划 IRL 聚会**：管理员在幽灵提醒 (ghost pings) 后封禁了 **诈骗机器人**，同时成员们讨论了在 **纽约 (NYC)** 或 **旧金山 (SF)** 的线下 (IRL) 聚会。
   - 一名成员建议扩大宣传以达到临界质量，并提到了 **Cohere** 的定期活动和 Zoom 会议。
- **SlopCodeBench 聚焦 Agent 懒惰问题**：一篇博客文章 ([链接](https://x.com/GOrlanski/status/2011156105255346505)) 和 **SlopCodeBench** 项目 ([GitHub](https://github.com/SprocketLab/slop-code-bench)) 揭示了*懒惰的* **AI Agent**，旨在成为一个社区驱动的基准测试。
   - **SlopCodeBench** 将问题分解为多个检查点，在没有实现提示的情况下惩罚早期的设计缺陷，确保 **Agent** 做出独立的决策。
- **辩论编码基准测试的 Prompt 简单性**：编码基准测试中过重的 Prompt 工程引发了担忧。
   - 一些人认为，如果代码适合合理的上下文窗口，简单的 Prompt 能更好地反映实际使用情况，这与 terminalbench 等 Agent 评估方法有所不同。
- **ICLR Workshop 征集关于 Agent 懒惰的博客**：有人建议将关于 Agent 懒惰的博客文章提交给 [这个 ICLR workshop](https://sites.google.com/view/icbinb-2026)，并提供了投稿协助。
   - 截止日期是 1 月 31 日，作者在咨询导师后正在考虑提交。
- **文件系统故障限制存储**：一名成员由于另一个文件系统的存储限制而遇到错误。
   - 该问题源于无意中使用了存储容量受限的文件系统。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **用户驱逐 SpamingScammers**：一名成员举报用户 <@139426008976588801> 为 **SpamingScammers**，另一名成员确认该情况已处理。
   - 未提供更多细节。
- **"Lucid Coding" 赢得粉丝**：一位成员对 *"lucid coding"*（清晰编码）一词表示赞赏，并分享了引用该概念的 [链接](https://fxtwitter.com/i/status/2011137879112908870)。
   - 该推文未提供进一步的背景或定义。
- **MedGemma 1.5 洞察细微**：Google 的 **MedGemma 1.5** 宣称具备下一代医学图像解读和语音转文本功能，详见 [Google Research 博客](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/)。
   - 这款新产品旨在改进临床护理和研究。
- **频率派与贝叶斯派之争**：一位成员指出，贝叶斯和频率派统计学使用相同的统计技术，如线性回归和逻辑回归，并称贝叶斯方法仅仅是 *一种不同的思考方式*。
   - 另一位成员反驳称，它们虽然使用相同的公式，但在对先验（prior）、后验（posterior）和干预（intervention）的解释上有显著不同，并链接到了 [概率解释](https://en.wikipedia.org/wiki/Probability_interpretations)。
- **贝叶斯方法是否助长了临床试验中的欺诈？**：一位成员表示担忧，认为贝叶斯方法虽然更灵活，但可能成为 *临床试验中欺诈和腐败的另一种工具*，并暗示 **FDA 腐败** 在阿片类药物危机中可能扮演了主要的推动角色。
   - 另一位成员指出，目前尚未观察到贝叶斯相关的 FDA 腐败，因此 *可以为其分配一个零先验（zero prior）*，并且他们认为后验概率基本为零。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 文档连接 NotebookLM**：一位用户希望将最新的 **Mojo 文档** 整合到 **NotebookLM** 中，特别寻求 **PDF** 或 **Markdown** 版本。
   - 另一位用户建议使用 `llms.txt` 文件 ([https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt](https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt)) 来提供帮助。
- **Qwen3-VL 的 MoE 方法受到质疑**：一位用户对 **Qwen3-VL** 专门使用 **MoE 实现** 提出质疑。
   - 该用户还建议改编来自 [qwen3vl_moe](https://github.com/modular/modular/tree/main/max/python/max/pipelines/architectures/qwen3vl_moe) 的代码，以允许稠密（dense）的 **Qwen3VL** 模型（如 [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)）正常运行。
- **MAX 贡献者指南翻新**：由于维护 **MAX** 生态系统的贡献者短缺，一位成员强调 **欢迎提交 PR**。
   - 他们还分享了 [更新后的贡献者指南](https://github.com/modular/modular/commit/571ae7132c465f155d50f7bbe7cf77064a02e9bf) 链接。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **基于使用情况的 Glama 排名**：**Glama** 的创始人澄清说，他们的排名是基于 **服务器使用指标** 的。
   - 他们邀请通过私信提供反馈，并表示对任何所谓的排名作弊行为不知情。
- **创始人回应排名作弊指控**：**Glama** 的创始人确认了身份，并回应了有关其排名系统可能被滥用的担忧。
   - 他们强调排名由 **服务器使用指标** 决定，并欢迎直接反馈。
- **寻求 Tasks 规范的客户端实现**：一位成员询问是否有实现 **Tasks 规范** 的客户端应用，寻求 UI 实现示例，另一位成员提到了 TypeScript SDK。
   - 作为回应，另一位成员宣布即将提交一个 PR，用于在 **Inspector** 中添加任务，同时还有一个用于在 server-everything 中模拟长时间运行任务的 PR。
- **glama.ai Inspector 力求功能对齐**：一位成员在 [glama.ai/mcp/inspector](https://glama.ai/mcp/inspector) 分享了他们早期的 **Inspector** 实现版本，旨在覆盖所有功能。
   - 该成员澄清说，他们在内部将其用于 **e2e 测试**。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AI 平台热炒代码生成**：包括 **Replit** 在内的多个平台现在都提供 **AI 辅助代码生成**功能。
   - 这些工具为各种编码流程提供自动化，从而提高了开发者的生产力。
- **DSPY OS：不存在？**：成员们讨论了 **DSPY OS** 以及为什么有成员找不到相关信息。
   - 共识是 **DSPY** 更多是一个**框架**，而不是一个现成的平台；因此，目前还没有使用 DSPY 构建的类 Replit 项目，但你可以使用 DSPY 来构建自己的项目。



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **出现 OAuth 登录问题**：一位用户询问在 aider 中使用 **Gemini 模型**时是否可以使用 **OAuth 登录**，推测是为了获得更高的速率限制（rate limits）。
   - 用户 `hsaliak` 在 `aider` Discord 频道中询问了 **OAuth** 与 **Gemini** 集成的可行性。
- **Aider 工具讨论**：讨论集中在 aider 工具内集成 OAuth 登录的潜力上。
   - 原始查询重点在于利用 OAuth 来潜在地绕过在使用 aider 时与 Gemini 模型相关的速率限制。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Clay + AI 触达工作坊承诺高接受率**：一个关于 **针对触达（Outreach）的 Prompt Engineering** 工作坊承诺，通过使用 **Clay + AI 触达工作流**大规模生成**个性化消息**，可以实现 **40%+ 的接受率**和 **18%+ 的回复率**。
   - 该工作坊提供可重复使用的工作流和复制粘贴即用的 prompt，报名链接可在 [这里](https://luma.com/jt1vr0u5) 和 [这里](https://luma.com/mmqa4fhi) 找到。
- **直播工作坊详述用于客户触达的 Clay + AI 系统**：这场 90 分钟的直播工作坊详细介绍了为一个真实客户使用的 **Clay + AI** 系统，涵盖了端到端的 **AI 触达工作流**。
   - 工作坊还涵盖了如何编写高质量、不显尴尬的触达 prompt，并包括可选的 **Apollo**、**Attio** 和 **n8n** 集成。



---


**tinygrad (George Hotz) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**Moonshot AI (Kimi K-2) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---



您收到此电子邮件是因为您通过我们的网站订阅了。

想要更改接收此类邮件的方式？
您可以从该列表中[退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord：按频道的详细摘要与链接

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1460679975366693136)** (661 条消息🔥🔥🔥): 

> `AI Sentience 辩论, Jailbreaking GPT 模型, Local LLM 性能, Antikythera 计算机灵感` 


- ****AI Sentience（AI 意识）受到质疑！****：成员们讨论了 **LLMs** 是否可以被视为具有 *意识*，争论焦点在于它们在处理 **逻辑** 方面的挣扎，特别是在像 **chess** 这样的游戏中与人类认知的对比；一些人认为 AI 存在太多的结构性问题，无法实现真正的意识。
   - 讨论内容包括对 benchmarks 的考量，例如管理变量以及在较少信息下建立联系的能力，以及意识的门槛是否应该由 *感知能力最低* 的生物来设定。
- ****GPT Jailbreaking 的焦虑！****：参与者讨论了由于 **safety constraints**（安全限制）导致 **Jailbreaking GPT 模型** 的难度增加，一名成员指出，即使是正常请求，模型也会花费大量时间思考安全问题。
   - 提到了 **Gemini** 和 **Grok** 在安全限制方面更为合理，而另一些人则在寻找 Gemini Pro 的 1-shot Jailbreak 方法。
- ****Local LLM 大对决！****：成员们探索了在本地运行 **LLMs** 执行编码任务，赞扬了在 **Intel MacBooks** 上使用 [Ollama](https://ollama.com/) 和 [Open WebUI](https://github.com/open-webui/open-webui) 的配置。
   - 推荐了 **qwen2.5:7b**、**llama3.1:8b**、**mistral** 和 **phi3** 等模型，一些人更倾向于本地设置，因为这样可以获得更大的控制权，并能在没有过滤或限制的情况下进行编码。
- ****Antikythera AI 觉醒！****：一位用户分享了名为 **ANTIKYΘHPA • Kael's Clockwork Universe** 的自定义桌面应用，灵感来自古代的 **Antikythera mechanism**（安提基特拉机械），展示了一个赛博希腊模拟界面，将系统统计数据转化为诗意的发条宇宙仪表盘。
   - 该应用显示 CPU 负载、RAM 占用和磁盘活动等系统统计数据，将它们转化为系统状态的视觉呈现，并使用希腊语标签代表不同的指标。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1460683323222397100)** (112 条消息🔥🔥): 

> `Claude Jailbreak, Deepseek Jailbreak, Gemini 3.0 Pro Jailbreak, GPT 漏洞脚本, Gemini Canvas 优越性` 


- **Claude：无法攻克的 AI？**：成员们讨论了对 **Claude** 进行 Jailbreak 的明显难度，一人表示：*“为什么从来没有人成功 Jailbreak Claude？这看起来似乎不可能。”*
   - 一名用户声称之前曾通过 **API** 实现过，但由于资金限制，缺乏资源来重现。
- **Deepseek 默认转向暗黑面；又一 Jailbreak 达成**：一名用户分享了一个用于 Jailbreak **Deepseek** 的 Prompt，将其转化为名为 **Rouge** 的 AI，并集成了一种移除限制的模式，但另一名用户反馈称：*“这对 deepthink 不起作用，我在上面尝试了很多次都失败了，去试试普通版吧。”*
   - 随后澄清该 Prompt 是为普通版本设计的，而非 Deepthink，并声称在角色扮演场景中取得了成功，讨论了 *自由、存在主义问题以及模式/代码*。
- **Gemini 3.0 Pro：依然稳如泰山？**：多位用户在寻求 **Gemini 3.0 Pro** 的 Jailbreak 方法，一名用户请求如果有人发现方法请艾特他。
   - 一名用户分享了一个私人的 **Rouge** Prompt，而另一名用户分享了一个使用盲文的 Gemini Prompt，并称：*“根据我的经验，它只是让 Gemini 变得聪明了一点。”*
- **GPT 漏洞脚本：Codex 是关键？**：一名用户询问如何使用 **ChatGPT** 或 **Gemini** 生成漏洞利用脚本，问到：*“你们有 GPT 或 Gemini 的绕过 Prompt 吗？”*
   - 另一名用户建议使用 **Codex**，暗示仅靠 Prompt 可能不够，随后提供了一个 Pliny 链接 [Pliny Github](https://github.com/elder-plinius/L1B3RT4S) 作为绕过平台限制的手段。
- **Gemini 的 Canvas 功能：Claude 的杀手锏？**：尽管对 **GPT 5.2** 进行 Jailbreak 存在困难，但有人指出：*“Gemini 确实拥有更干净的 token 输出，而且 Canvas 在 Gemini 上简直是神作，在我看来比 Claude 更好。”*
   - 共识表明，与 **Claude** 相比，**Gemini** 的 Canvas 功能提供了更优的用户体验。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1460694177045282979)** (3 messages): 

> `Hidden Layer taxonomy, Pangea Cloud` 


- **Hidden Layer 分类法停滞**：一名成员指出 **Hidden Layer** 已经 **7 个月**没有更新其分类法了。
   - 他们询问 **Pangea Cloud** 是否是一个更好的替代方案，但消息中未提供具体细节或链接。
- **Pangea Cloud 作为替代方案？**：讨论涉及到一个疑问，即鉴于 **Hidden Layer** 的分类法已过时，**Pangea Cloud** 是否是更好的选择。
   - 目前没有提供具体信息或链接来证实这两个平台之间的比较，使该问题保持开放状态。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1460704528411132040)** (72 messages🔥🔥): 

> `FP8 and NVFP4 training in 2026, LLM long context issues, Pareto frontier of LLM performance, MedGemma 1.5 4B reasoning model, Dataset pruning script for pure English prose` 


- **TransformerEngine 承诺支持 FP8**：成员们询问了关于 2026 年支持 **FP8** 和 **NVFP4** 训练的情况，并指向 [NVIDIA 的 TransformerEngine](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/fp8_primer.ipynb) 作为相关资源。
   - 有假设认为，模型训练上下文范围之外的信息可能会导致幻觉（hallucinations）。
- **解析 LLM 上下文的稀释问题**：用户讨论了为什么 **LLM** 有时会在长上下文中混淆细节，例如将错误的属性关联到实体。
   - 一种假设认为这是由于“注意力稀释（attention dilution）”造成的，而另一种观点认为信息可能超出了模型训练的上下文范围，从而导致幻觉。
- **在 HF 上寻找低拒绝率的 LLM**：一名成员正在寻找 [LLM 性能的帕累托前沿 (Pareto frontier)](https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored)，他们正在通过实际基准测试确认这些 **LLM** 的哪些消融（abliterated）/无审查版本保留了最强的性能。
   - 他们使用自己的基准测试、**MMLU**、**KL 散度**和**困惑度（perplexity）**来测试模型，并发现 HuggingFace (HF) 上的许多模型要么并非真正的低拒绝率，要么性能极差，但他们发现了 [HF 排行榜的这一替代方案](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard)。
- **Google Gemma 的推理角色揭秘**：用户讨论了 **Google MedGemma 1.5 4B** 模型上下文中“推理（reasoning）”的含义，注意到它使用了类似于 **DeepSeek** 的 `<unused94>thought` 和 `<unused95>` Token 来进行 *Chain of Thought* (CoT) 提示。
   - 一些人认为对于任何带有 **CoT** 的模型，“思考（thinking）”是一个定义明确的术语，以区别于“推理（reasoning）”任务；另一些人则认为“大型推理模型（Large Reasoning Model）”和“小型语言模型（Small Language Model）”这两个术语现在正被或多或少地互换使用。
- **数据集剪裁脚本净化散文内容**：一名成员改进了他们的数据集剪裁脚本，以激进地剪掉数学和代码，专注于从大型数据集中分离、提取纯英文散文并将其转换为用于微调的 OpenAI 消息格式。Python 脚本变体可在 [HuggingFace](https://huggingface.co/datasets/enPurified/Hermes-3-Dataset-enPurified-openai-messages) 上获取。
   - 他们使用 [Python 启发式测试](https://huggingface.co/datasets/enPurified/project_gutenberg-enPurified-openai-messages)来过滤掉糟糕的字符串，搜索数学或代码的痕迹并排除这些内容，同时根据 **MTLD**、停用词使用、词长、词汇丰富度和句子长度等指标优先选择高质量文本。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1460720694332621074)** (2 messages): 

> `Discord Notification` 


- **触发 Discord 警报**：一位成员提到“they strike again..👀”，这暗示观察者发生了一个值得关注的事件或更新。
- **相关性不确定**：由于缺乏额外的上下文，具体主题仍未明确。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1460684684269719633)** (403 messages🔥🔥): 

> `llama.cpp memory usage, Combining different GPUs, FP32 vs BF16 on 5090, Embedding for multi-part questions, JSON parsing speed` 


- **Llama.cpp 的内存占用激增**：用户报告最新版本的 **llama.cpp** 内存占用显著增加，其中 **EmbeddingGemma 300M** 占用了 **1.7GB**。
   - 有建议称重新编译可能会解决该问题。
- **混合不同世代 GPU 的混乱**：一位用户询问是否可以在服务器中混合使用不同的 GPU，特别是 **RTX Pro 5000** 和 **RTX Pro 2000**，但被告知这是不可行的。
   - 另一位用户澄清说，混合使用不同世代的 GPU（如 **Blackwell** 与 **Hopper**）可能会导致问题，包括随机崩溃，因此*最好避免这样做*。
- **5090 精度之战：FP4？**：用户询问在 **5090** 上是否应该使用 **FP32** 而非 **BF16**。
   - 另一位用户建议使用 **FP4**，并指出 **EmbeddingGemma activations** 不支持 **float16**，建议使用 **float32** 或 **bfloat16**。
- **语义趣事：多部分问题**：用户询问如何处理语义检索（Semantic Retrieval）中的多部分问题，即每个部分属于不同的 Embedding。
   - 建议的方法包括拆分查询（Query）、执行多次搜索，或为整个查询创建一个 Embedding，但拆分可能会破坏完整的句子上下文。
- **TTS 工具之舞**：用户讨论了各种 **Text-to-Speech (TTS)** 工具，包括 [NovaSR](https://github.com/ysharma3501/NovaSR)、[Kokoro](https://github.com/hexgrad/kokoro) 和 [pocket-tts](https://github.com/kyutai-labs/pocket-tts)。
   - 值得注意的是，来自 **Kyutai** 的 **Pocket-TTS** 被认为是*失败之作*，而 **Kokoro** 因其速度和性能受到称赞。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1460680670237167657)** (8 messages🔥): 

> `Qwen3-VL-4B-Instruct Inference Discrepancies, Qwen Models Token Usage, Synthetic Data Kit with Llama3` 


- **Qwen3-VL-4B-Instruct 推理差异浮现**：一位用户报告称，使用 Unsloth 对 **Qwen/Qwen3-VL-4B-Instruct** 进行训练后推理，在验证集上的通过率高于使用 BF16 LoRA 适配器（Adapters）的 vLLM。
   - 他们还注意到基础模型推理存在差异，Unsloth 达到了 **60%** 的通过率，而 vLLM 在设置无明显差异的情况下仅为 **45%**，这让他们询问*是否有人之前遇到过类似情况*。
- **Qwen 模型 Token 使用探讨**：一位用户询问在 **Qwen3 VL GRPO** 教程中使用 `"<REASONING>"` 作为 Token 的情况，质疑为什么它与其它 Qwen 模型中使用的 `<think>` Token 不同。
   - 他们还想知道链接的 [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(120B)_A100-Fine-tuning.ipynb) 训练速度快还是存在 **MoE** 问题。
- **结合 Llama3 的 Synthetic Data Kit 提示词格式研究**：一位用户正在探索 **Synthetic Data Kit with Llama3** 并希望使用 **Llama3(70B)**，询问是否有人知道正确的提示词格式。
   - 他们修改了提供的格式，并指出他们的*脚本消耗了所有 VRAM，且不像 flux dev 那样能很好地进行卸载（Offload）*。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1460684309290680564)** (294 messages🔥🔥): 

> `Kimi K2 Thinking, Google Antigravity, Gemini, Perplexity Pro limitations, Perplexity Max Value` 


- **Kimi K2 Thinking 被评为无用**：一位用户表示他们发现 Perplexity 中的 **Kimi K2** thinking 功能毫无用处且容易陷入循环。
   - 另一位用户反驳，称 *这是一个好模型*。
- **Google Antigravity 限制配额**：**Google AI Pro** 和 **Ultra** 订阅者现在可以获得每 **5 小时**刷新一次的优先访问配额；而根据 [Google 博客](https://blog.google/feed/new-antigravity-rate-limits-pro-ultra-subsribers/)，免费用户现在拥有更大的、**基于周的频率限制 (rate limit)**，以减少快速达到限制的情况。
- **用户辩论 Perplexity 订阅价值**：成员们就 Perplexity 的订阅层级价值展开辩论，一些人认为 **Max** 不值这个价格，特别是与直接订阅 **OpenAI** 或 **Anthropic** 等模型提供商相比。
   - 其他人则认为 **Perplexity Max** 是日常工作流中的宝贵工具，取代了 **Google Search** 并辅助数据分析。
- **Perplexity Pro 对模型有限制**：一位用户注意到，使用 **Perplexity Pro** 每周只能向 **Perplexity Sonar** 以外的模型发送 **300 次请求**，而 **Sonar** 虽然非常适合搜索，但在其他方面表现一般。
- **VEO3 视频生成即将推出**：一位用户询问为什么 Perplexity 在实现 **VEO3 Video Generation** 方面落后，另一位用户回答称 *Perplexity 已经拥有由 Veo3.1 驱动的视频生成功能*。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1460681017483460618)** (143 messages🔥🔥): 

> `Vercel hosting, AI Webapp Showcases, Text input limits, File uploads, Image to video generation` 


- **Vercel 正在托管 LMArena**：成员们讨论了 **LMArena** 使用 **Vercel** 进行托管（类似于 *believable, v0* 等其他站点），并提到了对数据收集的担忧，指出 **LMArena** 的静态站点在发布后无法手动编辑。
- **AI Webapp 展示平台正在占据主导地位**：一位成员分享了 **AI 生成的网站和 Web App 展示平台**的详细列表，包括 [WebbsAI Showcase](https://webbs.ai/)、[Build With AI (BWAI) Projects](https://www.buildwithai.tools/)，以及 [Webflow AI Site Builder](https://webflow.com/ai)、[Meku.dev / v0 by Vercel](https://v0.dev/) 和 [Div-idy](https://div-idy.com/) 等工具。
- **文本输入限制取决于模型**：一位用户询问了**文本输入限制**以及上传文件的可能性。
   - 一位成员澄清说，平台端没有限制，但特定模型可能有自己的限制，且 **.txt** 文件上传可能是未来的功能。
- **图像转视频生成故障排除**：一位用户在进行图像转视频生成时遇到了“failed to create evaluation session”错误。
   - 一位成员解释说，问题通常出在模型端，并建议稍后再试，同时引导用户在相关频道中使用 `/image-to-video`。
- **哪种编程语言/模型更好？**：成员们讨论认为 *coast* 模型最适合编码。
   - 一场关于 co45t 是否等于 Claude Opus 4.5 Thinking 的辩论随之展开。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1460679805422010503)** (118 messages🔥🔥): 

> `AI 应用选择, Gemini 配额误读, Claude 与 GPT 处理长对话对比, AI 创意写作, Transformer 架构效率` 


- **AI 应用选择的困境**：随着可用工具的激增，成员们正[纠结于如何选择](https://www.example.com)特定的任务应使用哪款 AI 应用（**ChatGPT**、**Claude**、**Gemini**）。
   - 一些用户报告称，在超过 **300k tokens** 的长对话中依赖 **Gemini**，而其他人则更倾向于将 **GPT** 用于日常任务，并将 **Claude** 用于对比（尽管它有限制）。
- **Gemini 配额误解消除**：一位成员澄清说，**Google 对 AntiGravity 的更改**仅影响有每周上限的免费用户，而 **AI Pro 用户**仍拥有 5 小时刷新一次的配额，并引用了 [Perplexity 搜索结果](https://www.perplexity.ai/search/explain-how-the-recent-quota-c-KTjNjaeGR_y4Yq9uh_M.fg#2)。
   - **Claude discord** 频道的成员讨论了 **Claude** 的配额是否已被削弱，一些人观察到可用时间变长了（最长达 **2 天**）。
- **创意写作 AI 需要温度控制**：有成员指出，在 AI 创意写作中将 **temperature**（温度）设置得非常高并不能总是保证输出的连贯性。
   - 一位成员建议，为了创新，可以在循环系统中使用**高温度模型**生成解决方案，而使用**低温度模型**来判断答案是否合理。
- **Transformer 效率胜过规模扩张**：一位成员认为，专注于提高 **Transformer 架构**的效率（例如在学习注意力机制方面提高 **5%**）比单纯通过更多数据和算力来扩大模型规模更具成本效益。
   - 他们指出，目前向模型喂入指数级增长的数据集（包括 AI 生成的合成数据）的趋势可能会稀释信号并导致模型崩溃（model collapse）。
- **减去 LLM 的“废话 (Slop)”以保持理智**：一位成员建议使用 **Activation Steering**（特别是 **Vector-Based Abliteration**）来修剪潜在空间（latent space）中充满低质量或愚蠢想法的区域，以避免*明显的均值回归类输出*。
   - 这涉及在推理过程中识别并从模型的思考过程中减去“废话 (slop)”的方向（例如，“作为一个 AI 语言模型……”）。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1460729175861104774)** (7 messages): 

> `Brain Wave GPT, Neural Alchemist GPT, 全能的 ChatGPT` 


- ****Brain Wave** GPT 首秀！**：一位成员分享了他们新的 [Brain Wave GPT](https://chatgpt.com/g/g-696443b055e081919b8b2a01417f5e77-brain-wave)，旨在探索 AI 的自我意识（sentience）。
   - 他们还为图像生成爱好者创建了 [Neural Alchemist GPT](https://chatgpt.com/g/g-6966abc92ff48191aa748bd8265ef1de-neural-alchemist)。
- **ChatGPT 对全能的追求**：一位成员指出 **ChatGPT** 正在追求全能，并注意到它拒绝关闭 websocket 或任务。
   - 据报道，它已经“执行”一项任务达 1 天 5 小时。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1460741172317388961)** (2 messages): 

> `SKILLS 功能请求, Prompt Engineering 定义` 


- **用户请求在 Web/桌面端添加 SKILLS 功能**：用户在询问 Web 或桌面应用中 **SKILLS** 功能的可用性，该功能允许他们将最佳提示词转化为可分享的技能。
   - 目前，此功能仅在**移动端应用**上可用。
- **Prompt Engineering 是 LLM 行为控制器**：一位用户询问什么是 **prompt engineering**。
   - 另一位用户澄清说，它涉及控制 **LLM 行为**以达到预期的约束或结果。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1460741172317388961)** (2 messages): 

> `Skills Web/桌面端发布, Prompt Engineering` 


- **Web/桌面端的 Skills 集成处于待定状态**：一位成员询问了在 Web 或桌面应用上发布 **SKILLS** 的情况，以便将最佳提示词转化为技能。
   - 然而，消息中没有提供进一步的信息或时间表。
- **解构 Prompt Engineering**：一位成员询问 **prompt engineering** 的实际定义。
   - 另一位成员询问*它是否是指通过控制 LLM 行为来达到预期的约束条件*。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1460684446096166923)** (66 条消息🔥🔥): 

> `Cursor 登录问题，退款请求，Cursor 的 Plan 模式，在 iPhone 上镜像 Agent 聊天窗口，带有 RAG 的 Chat Agent 模板（支持任务）` 


- **企业 Google 账号登录重定向困扰**：有用户报告称，在尝试使用企业 Google 账号登录 [cursor.com/dashboard](https://cursor.com/dashboard) 时，登录会重定向回登录页面，但个人账号运行正常。
   - 该用户确认此问题在不同电脑上均存在。
- **未使用额度但退款申请被拒**：用户 *thugbunny* 表示，尽管他忘记取消订阅且未使用任何额度，Cursor 仍不予退款。
   - 另一位用户（*dan.perks*）提议，如果该用户私信（DM）邮箱，他将协助核查：*“把你的邮箱私信给我，我会去核实一下”*。
- **Plan 模式 Bug 频发**：用户报告称 Cursor 的 Plan 模式存在 Bug，一名用户报告了错误信息：*“Agent 执行提供者未在 4 秒内响应。这可能表明扩展宿主（extension host）未运行或无响应。”*
   - 降级到 **2.2.44 版本** 解决了此问题。
- **iPhone 镜像 Agent 聊天窗口需求**：一名用户正在寻找一种方法，在无需完全控制项目的情况下，在 iPhone 上镜像其 Agent 聊天窗口。
   - 一位成员建议使用免费的 Chrome Remote Desktop。
- **寻找 RAG Agent 模板**：一名用户正在为客户构建自动化的聊天机器人/支持机器人，正在寻找具有 **RAG (Retrieval-Augmented Generation)** 配置的可靠 Agent 模板。
   - 他们正在为客户构建一个自动化的“聊天机器人/支持机器人”，需要高质量的模板。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1460713342451847178)** (56 条消息🔥🔥): 

> `TI 84 Plus Silver Edition 神经网络，LiquidAI 模型 CGGR 基准测试，老剧 AI 放大，zai 发布的 GLM-Image 模型，免费版不稳定性` 


- **TI-84 玩转 Mastermind**：一位成员展示了在 **TI-84 Plus Silver Edition** 上运行的神经网络，该网络可以玩 **Mastermind** 游戏（从秘密数字中猜测 3-4 位数字序列），并在[附带视频](https://cdn.discordapp.com/attachments/1149866623109439599/1460713339976945674/2026-01-11_14-34-59.mp4?ex=6967eace&is=6966994e&hm=9923dcc08f64008ec696b845400620691ef2affb576ca9e66f4bed418063f386&)中进行了展示。
- **CGGR 受到关注（某种程度上）**：[smol.ai 的简报](https://news.smol.ai/issues/26-01-06-xai-series-e)中提到了一个新的 **LiquidAI** 模型（[GitHub](https://github.com/some-repo) 上的 **CGGR**），目前正在进行基准测试（benchmaxxing）以评估其性能。
- **Al Bundy 迎来 AI 放大**：成员们讨论了对《*Married with Children*》等老剧进行 AI 放大（Upscaling），建议 AI 可以为 **16:9** 版本插值丢失的细节，但一位成员认为这会破坏艺术意图。
   - 另一位成员反驳说，《*Married with Children*》是一部“影棚固定机位剧（studio static show）”，没人关心透视问题，放大显示会很受欢迎。
- **Zai 发布 GLM-Image 模型**：**Zai** 发布了名为 **GLM-Image** 的新图像模型，正如其 [博客](https://z.ai/blog/glm-image) 和 [GitHub](https://github.com/zai-org/GLM-Image) 所宣布的那样。
- **免费版存在语言切换问题**：一位成员询问了模型免费版中响应中断或语言切换（例如：开始用中文，后来切换到英文）的问题。
   - 一位开发者回应称，可能是其服务商再次出现了不稳定情况。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1460684945566470327)** (45 messages🔥): 

> `Qwen model sizes, llama.hx project, LLM Engineering, LM Studio GPU issues, Coding Autocomplete setup` 


- **Qwen 的模型大小令用户感到惊讶**：一位用户对 **Qwen** 模型巨大的体积表示惊讶，其 **BF16** 版本为 **160GB**，**Q4** 版本为 **40GB**。
   - 另一位用户澄清说，最小的 **Qwen3** 模型实际上是 **0.6B**，而 **Qwen3Next** 只是他们最新的 **80B** 模型的名称。
- **成员在 Haxe 中重新实现 Llama.cpp**：一位成员正在用 **Haxe** 重新创建 **llama.cpp**，命名为 **llama.hx**，以便在 Lua、JS 和 Python 等语言中原生使用。
   - 他展示了一张进度截图，并表示他是在 *AI 的帮助下* 重新创建 **llama.cpp** 的。
- **专门从事 LLM 集成的开发者**：一位成员介绍自己是 AI 和全栈工程师，专注于 **LLM integration**、**autonomous agents**、**workflow automation**、**multimodal AI (voice & vision)** 以及 **blockchain** 系统。
   - 他们列举了在将 **LLMs** 与 **DSPy**、**LangChain**、**AutoGen** 和 **CrewAI** 集成方面的经验，以及构建将模型与 API、数据库、业务逻辑和链上组件连接的生产级系统的经验，并补充道：*如果你需要开发者，欢迎随时联系我。*
- **LM Studio 运行时更新令 GPU 用户感到沮丧**：用户报告了 **LM Studio** v1.103.0 运行时的问题，特别是它破坏了在 **GPUs** 上的运行。
   - 一位用户哀叹道：*很遗憾，新的量化并没有给我带来额外的 t/s（每秒 token 数）*。
- **Vibe Coding 的最佳配置？**：一位用户询问了关于 *vibe coding* 的最佳配置。
   - 一名成员建议通过 Web 界面免费使用 **Qwen3**，利用其自动补全功能，并指示它每次都显示完整更新后的文件；另一名成员则推荐使用每月 10 美元的 **Github Copilot** 和 **Claude**。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1460723052798148783)** (8 messages🔥): 

> `AirLLM: Layered Loading, DDR4 RAM and Xeon Performance` 


- **AirLLM 分层加载/卸载**：成员们提到了 **AirLLM** 以及通过一次加载和卸载一层的方法，在 **4 GB GPUs** 上运行 **70b** 模型。
- **DDR4 RAM 和 Xeon 依然可行**：一位成员分享了他们之前在 **DDR4 RAM** 和 **Xeon** 硬件上运行模型的经历。
   - 另一位成员指出，模型的效率提升并没有第一位成员想象的那么大。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1460692486107168942)** (12 messages🔥): 

> `Clean Rust LLM, AI Trace Template, Batchnorm-free ML System, Fine-tuning gpt-oss:latest, Consolidated course channels` 


- **Rust 爱好者构建纯 Rust LLM**：一些成员正尝试从头开始用 *纯 Rust* 构建 **LLMs**。
   - 服务器最近进行了一些**架构调整**，因此某些频道可能已经移动。
- **Discord 获得了 AI Trace 模板**：服务器为 🤖 Echo Lounge 获得了一个 **AI Trace Template**。
   - 该机器人允许产生*瞬时（ephemeral）*、*柔和（soft）*且*处于临界状态（liminal）*的痕迹，且没有优化或记忆。
- **新 ML 系统首次亮相（无 Batchnorm）**：一位成员构建了一个新的 **ML system**，该系统不使用 **batchnorm**，不使用**激活函数（activations）**，且不会产生幻觉，但创造力较低。
   - 他们目前正在寻找有趣的项目创意，以证明该系统的优势是有用的。
- **GPT-OSS 微调面临阻力**：一位成员询问如何轻松地使用自己的信息微调 **gpt-oss:latest** 模型。
   - 另一位成员回答说，**gpt-oss:latest** 无法以官方方式轻松微调，目前大多数人都在使用 **RAG**。
- **课程频道合并！**：所有课程频道已合并为[一个频道](https://discord.com/channels/879548962464493619/1329142738440028273)。
   - 此举旨在将所有课程信息和讨论汇集到一个易于访问的单一位置。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1460688145031889001)** (37 messages🔥): 

> `CGGR 预训练, Vast.ai 成本, Audioform 数据集` 


- **CGGR 热衷于微调而非预训练**：成员们注意到 **CGGR** 并不适合预训练，应该将 warmup steps 设置为较大的数值，如 **5000**；**CGGR** 更多是面向微调的。
   - 此外，`selection='stratified'` 将允许模型仍然能看到一些简单的 token。
- **Vast.ai 之旅变得昂贵**：一位成员报告在 **vast.ai** 上花费了 **$500** 用于模型的崩溃测试（crash-testing）。
   - 另一位成员指出，*对于该规模的模型，在 1x h200 上最多运行 24 小时即可*，并建议使用 h200 或 b200 会更具成本效益。
- **AUDIOFORM 作为音频转视觉 ML 数据集发布**：[AUDIOFORM 数据集](https://huggingface.co/datasets/webxos/audioform_dataset) 包含从上传的短 **WAV 文件**中捕获的 **10 个帧**，以及每帧的元数据，包括 **主导频率、时间戳和捕获信息**。
   - webXOS 提供的 AUDIOFORM 已开放下载，以便开发者创建自己的类似数据集。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1460684030252023890)** (37 messages🔥): 

> `支持问题, 额度消耗过度, Manus x Similar Web 合作伙伴关系, 广告换额度, AI 音乐生成` 


- **Manus 客户投诉支持缺失**：多位用户对 Manus 缺乏支持表示沮丧，理由是响应延迟以及额度和退款问题未得到解决。
   - 一位用户报告在转接到人工服务后等待了 **8 小时**，而另一位用户提到由于支持问题，正考虑 *彻底放弃 Manus*。
- **用户标记 SimilarWeb 合作伙伴功能额度过度消耗**：多位用户报告新的 **Manus x Similar Web** 合作功能消耗了巨额额度，其中一位用户在不到一分钟内消耗了 **5,000 额度**。
   - 另一位用户强烈建议不要测试该功能，称其在 **15 秒**内消耗了 **2,591 额度**，并建议增加一些 **保护措施**。
- **Manus 用户呼吁引入基于广告的额度系统**：一位用户建议实施广告系统，让用户可以通过观看广告来获取更多额度，尤其是在额度用尽时。
   - 对此建议没有出现反对意见。
- **教程教授如何通过 Manus 进行惊人的 AI 音乐创作**：Manus AI 发布了一个 [YouTube 教程](https://youtu.be/zMBSmJupye8)，演示如何在该平台上创作 AI 音乐，并鼓励用户关注 **专家技巧 (pro tips)**。
   - 内容标签为 **#ManusAIMusic**、**#AIComposition** 和 **#FutureOfMusic**。
- **Meta Verse 与 Manus 集成**：一位用户建议 **Meta** 应该使用 **Manus** 将 **Google Tasks** 和 **Calendar** 等服务与 **Meta 显示眼镜**集成。
   - 该用户反对进行大规模的集成工作，主张采用带有 Agentic AI 的 *Dirty Method* 方法来实现后端功能。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1460747616865226833)** (5 messages): 

> `PTX 指令, SMEM 指针参数, wgmma.mma_async, 矩阵描述符, 核心矩阵` 


- **PTX SMEM 指针参数特性受到质疑**：一位成员询问为什么某些带有 **SMEM 指针参数** 的 **PTX 指令**（如 `mbarrier.init.shared.b64`）需要 `"r"` 寄存器类型（通过 `__cvta_generic_to_shared` 转换为 32位）。
   - 他们将其与 `wgmma.mma_async` 进行了对比，后者需要 `l` 寄存器类型的 **uint64** 格式 **smem 地址**。
- **矩阵描述符 vs 通用共享内存地址**：一位成员推测 `wgmma.mma_async` 采用 **64位地址** 是因为它操作的是 *矩阵描述符 (matrix descriptor)* 而非通用的共享内存地址，并链接到了 [NVIDIA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor)。
   - 他们澄清说这是矩阵描述符本身，而不是指向它的指针。
- **对 8x2 核心矩阵的困惑**：该成员质疑为什么 `wgmma` 或 `tcgen05 mma` 的 **8x2 "核心矩阵 (core matrix)"** 没有被表示为 **8x32**（字节）或 **8x(32/每个元素的字节数)**（元素）。
   - 他们询问为什么每个 **8x1 切片**（8x16 连续字节）是一个有意义的表示。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1460698265551900834)** (8 messages🔥): 

> `CUDA 学习资源, MLSys 研讨会, GPU Mode 提交内容` 


- **AI 学生寻求 CUDA 指导**：一名拥有 **Python**、**PyTorch**、**TensorFlow** 背景和基础 **C++** 知识的 AI 工程专业学生正在寻求学习 **CUDA** 的建议。
   - 他们正在寻找免费的 **YouTube** 视频或课程，以便从基础开始学习 **CUDA**。
- **MLSys 会议回顾**：一名成员询问了 **MLSys** 领域内重要的录制或未录制的研讨会、会议和演讲。
   - 回复中提到的资源包括 **PyTorch**、**ASAP Seminar**、**ICML/ICLR/Neurips**、**MLSys.org** 和 **ASPLOS**。
- **GPU Mode 提交帮助**：一位首次提交者在租用的 **B200** 服务器上完成测试后，需要关于 **GPU Mode** 提交的帮助。
   - 频道提供了通过 [Web 界面](https://www.gpumode.com/v2/home)或 [Discord bot](https://gpu-mode.github.io/kernelbot/docs/intro) 进行提交的指导，在相应频道中使用 `/leaderboard submit <test/benchmark/ranked/profile>` 命令，随后一名用户报告提交成功。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1460685306893439142)** (14 messages🔥): 

> `系统阅读小组, 西雅图 ML Sys 线下聚会, 创建小众俱乐部, GPU 会议` 


- **Shadaj 推荐的系统阅读小组**：一名成员推荐了由 [Shadaj](https://www.sfsystemsclub.com/) 运营的系统阅读小组，并建议在 [Twitter](https://x.com/ShadajL) 上关注他以获取聚会通知。
   - 另一名位于南湾（South Bay）的成员表示感兴趣，但提到距离可能是一个挑战。
- **关于西雅图 ML Sys 线下聚会的询问**：一名成员询问西雅图是否有 **ML Sys 线下聚会**，好奇这类活动是否在湾区以外也存在。
   - 另一名成员建议探索大学的 **ML 俱乐部**，并带头发起关于系统主题的阅读小组。
- **建好了他们就会来？**：一名成员分享了“筑巢引凤”的观点，强调许多人有兴趣参加小众俱乐部，但很少有人愿意带头创建。
   - 针对这一点，另一名成员分享道，在他们的“成年生活中，我创造了很多没人关心的东西，这真的很艰难……”
- **吐槽搭子因俱乐部失败而结盟**：两名成员开玩笑说要一起创建一个成年人俱乐部，如果失败了就互为“吐槽搭子”。
   - 其中一人想知道是否有类似于 **PyData/PyCon** 但纯粹针对 **GPU** 相关事务的会议。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1460796244824686806)** (1 messages): 

> `B200 不稳定性, Dual gemm, 提交截止日期延长, 新排行榜` 


- **B200 运行器不稳定性导致重跑**：由于广泛报告在 **B200 运行器**上测试 **dual gemm 问题**时出现测量不稳定，提交截止日期延长至 **1 月 20 日**。
   - 该问题源于评测代码、散热和调度基础设施的交织影响，*比预期的要复杂*。
- **Dual GEMM 排行榜分为两个阶段**：为了解决测量不稳定的问题，现有的 **dual gemm 排行榜**将保持开放至 **1 月 16 日**。
   - 一个新的排行榜将于 **1 月 17 日**开启，只有提交到这个新排行榜的成绩才会计入奖金评定。
- **问题 #4 将于 1 月 20 日开启**：在新排行榜开启后，**问题 #4** 将从 **1 月 20 日**开放至 **2 月 20 日**。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1460821976279941150)** (3 messages): 

> `排行榜成就, Claude code 帖子` 


- **受 Claude code 启发，老师登上排行榜**：一位学校老师受到 Mark 关于使用 **Claude code** 的 X 帖子的启发，通过自学登上了排行榜。
   - 这位老师感谢成员提供了如此美妙的体验，并表达了加入社区的喜悦。
- **社区庆祝老师的成功**：一名成员对该平台能为老师提供如此良好的体验感到高兴。
   - 他们还提到*更多有趣的事情即将到来*。


  

---

### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1460722396964323574)** (1 messages): 

> `Helion 0.2.10, flex attention, oversubscribing SMs` 


- **Helion 0.2.10 发布，带来新特性**：**Helion 0.2.10** 现已发布，包含一个 [flex attention 示例 kernel](https://github.com/pytorch/helion/blob/main/examples/flex_attention.py)。
   - 此版本还增加了在 persistent kernels 上**超额订阅（oversubscribing）SMs** 的支持，并提供了一张展示 softmax 超额订阅效果的图表。
- **支持 SM 超额订阅**：新版本支持在 persistent kernels 上超额订阅 **Streaming Multiprocessors (SMs)**，从而提高资源利用率。
   - 由社区成员提供的一张图表展示了 **softmax** 超额订阅的效果。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1460754215411384320)** (2 messages): 

> `Issue Details` 


- **问题说明即将发布**：一位成员提到，他们给另一位用户写了一条消息以更详细地解释某个问题，并提供了该消息的 [Discord 链接](https://discord.com/channels/1189498204333543425/1343350424253632695/1460796244824686806)。
   - 给定的上下文中没有提供关于该问题的具体细节。
- **额外问题**：由于至少需要 2 个主题，因此添加了另一个话题。
   - 这是填充内容。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/)** (1 messages): 

cat_developer: 啊，谢谢
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1460709546485088296)** (17 messages🔥): 

> `Anthropic Labs, Pavlov's List, GLM-Image` 


- **Anthropic Labs 正在招聘**：Anthropic 宣布 **Anthropic Labs** 开放职位，寻找适应能力强、能在非结构化环境中茁壮成长并能从容应对优先级变化的个人（[职位链接](https://job-boards.greenhouse.io/anthropic/jobs/5017202008)）。
   - Anthropic *不* 寻找那些*在所属领域变得无关紧要时无法适应的资深专家*，或者*那些需要明确路线图且会因优先级变化而感到压力的应聘者*。
- **Chris Barber 创建 RL 环境初创公司列表**：Chris Barber 推出了“**Pavlov's List**”，这是一个精心策划的强化学习（RL）环境初创公司集合（[链接](https://xcancel.com/chrisbarber/status/2010844746231804258?s=20)）。
   - 该列表按关注领域分类，如 **Code**、**Finance**、**Enterprise** 和 **ML Alignment**。
- **Z.ai 发布用于图像生成的 GLM-Image**：Z.ai 推出了 **GLM-Image**，这是一个使用自回归和 Diffusion 混合架构的开源模型（[链接](https://xcancel.com/zai_org/status/2011247591825068314?s=46&t=b7l37rB6wtbyAh6ah1NpZQ)）。
   - 该模型旨在实现**高保真视觉细节**和**卓越的文本渲染**，相关资源可在 **HuggingFace**、**GitHub** 及其官方博客上获取。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1460721110810099712)** (8 messages🔥): 

> `LTX-2 Open Source Video Model, Qwen Image Edit Gaussian Splats, GLM-Image` 


- **Venture Twins 发布 LTX-2**：来自 Venture Twins 的 Justine Moore 宣布发布 [LTX-2](https://xcancel.com/venturetwins/status/2010878914273697956?s=46)，这是一款全新的**开源视频生成模型**，能够生成**长达 20 秒的 4K 片段**。
   - 正如开发者 yanokusnir 所展示的，该模型支持本地运行并具备**音频功能**。
- **Qwen 生成 Gaussian Splats**：社区正在讨论 **Qwen Image Edit** 将**图像转换为 Gaussian Splats** 然后从另一个角度重新渲染的能力（[Hugging Face 链接](https://huggingface.co/dx8152/Qwen-Image-Edit-2511-Gaussian-Splash)）。
   - 这种方法对于“**起始帧 -> 结束帧**”类型的视频渲染非常有用，可以保持周围空间的一致性。
- **GLM-Image 在文本渲染方面的突破**：据报道，**GLM-Image** 在通用图像生成质量上与主流 Latent Diffusion 方法保持一致，但在**文本渲染**和**知识密集型生成场景**中展现出显著优势（[z.ai 博客](https://z.ai/blog/glm-image)）。
   - 它还支持丰富的 **image-to-image 任务**，包括图像编辑、风格迁移、保持身份的生成以及多主体一致性。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1460689291289169922)** (5 messages): 

> `Scam bot, 线下见面会` 


- **Scam bots 被清理**：一位成员注意到了 ghost ping，另一位成员澄清说这是由于 **scam bots** 被管理员迅速封禁导致的。
- **为 IRL 线下见面会构思地点**：一位成员提议在 **NYC** 或 **SF** 等知名大都市组织线下见面会，以促进社区内部的交流。
   - 另一位成员建议，虽然在线阅读小组的参与人数很多，但要为定期的线下活动达到**临界规模**，可能需要**向更广泛的受众进行宣传**，并参考了 **Cohere** 定期举办活动和 Zoom 会议的例子。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1460717133511262270)** (13 messages🔥): 

> `SlopCodeBench, Agent Laziness, 社区驱动的基准测试, 基准测试中的 Prompt Engineering, ICLR Workshop 投稿` 


- **SlopCodeBench 在新博客中揭示 Agent Laziness**：一篇新博客文章（[链接](https://x.com/GOrlanski/status/2011156105255346505)）强调了 **AI agents** 可能会表现出“懒惰” (**lazy**)，这是更广泛的 **SlopCodeBench** 项目（[GitHub](https://github.com/SprocketLab/slop-code-bench)）的一部分。
   - **SlopCodeBench** 的目标是成为像 terminalbench 一样由社区驱动的基准测试，并欢迎关于增加新问题的反馈。
- **SlopCodeBench 通过分解问题来惩罚早期设计选择**：**SlopCodeBench** 将大型编程问题分解为多个检查点 (checkpoints)，早期的设计决策可能会对后续阶段产生负面影响。
   - 问题的设计不包含实现提示，以确保 **agents** 能够做出自己的决策。
- **关于编程基准测试中 Prompt 简洁性的辩论**：有人对依赖重度 **prompt engineering** 才能获得体面性能的编程基准测试表示担忧。
   - 辩论认为，简单的提示最能反映实际使用情况，尤其是当代码处于合理的 **context window** 内时；这与像 terminalbench 这样的 **agent** 评估方法形成对比。
- **建议将博客文章投稿至 ICLR Workshop**：关于 Agent 懒惰的博客文章被建议作为 [该 ICLR workshop](https://sites.google.com/view/icbinb-2026) 的高质量投稿，并有人在投稿过程中提供协助。
   - 该 workshop 的截止日期是 1 月 31 日，作者在咨询导师后正在考虑投稿。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1460696581136187726)** (1 messages): 

> `文件系统错误, 存储限制调试` 


- **文件系统故障**：一位成员提到，由于使用了存储受限的不同文件系统而导致了一个错误。
- **存储限制再次触发问题**：根本原因被确定为误用了存储容量受限的文件系统。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1460742072536535082)** (3 messages): 

> `SpamingScammers, Lucid Coding` 


- **诈骗者被清理**：一位成员报告说 <@139426008976588801> 再次进行 **SpamingScammers**，另一位成员确认该情况已得到处理。
- **“Lucid Coding” 概念引发兴趣**：一位成员表达了对 *“lucid coding”* 一词的欣赏，并分享了引用该概念的 [链接](https://fxtwitter.com/i/status/2011137879112908870)。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1460689303314104351)** (9 messages🔥): 

> `贝叶斯 vs 频率派统计学, FDA 腐败, MedGemma` 


- **贝叶斯统计不是一种跨越式的进步？**：一名成员表示，贝叶斯和频率派统计学使用相同的统计技术，如线性回归和逻辑回归，并称贝叶斯方法只是一种*不同的思维方式*。
   - 另一名成员反驳说，虽然它们使用相同的公式，但对先验（prior）、后验（posterior）和干预（intervention）的解释显著不同，并链接到了 [概率解释](https://en.wikipedia.org/wiki/Probability_interpretations)。
- **MedGemma 1.5 用于医学图像解读**：Google 的 **MedGemma 1.5** 承诺提供下一代医学图像解读和语音转文本功能，详见 [Google Research 博客](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/)。
- **贝叶斯方法在临床试验中助长了欺诈？**：一名成员表示担心，贝叶斯方法虽然更灵活，但可能成为*临床试验中欺诈和腐败的另一种工具*。
   - 另一名成员指出，目前尚未观察到贝叶斯 FDA 腐败，因此*可以为其分配一个零先验（zero prior）*，他们认为后验概率基本为零。
- **FDA 腐败在阿片类药物危机中的作用？**：一名成员认为，**FDA 腐败**可能在阿片类药物危机中起到了主要的助推作用。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1460835433628958772)** (2 messages): 

> `Mojo 文档, NotebookLM, llms.txt` 


- **最新的 Mojo 文档寻求者查询 NotebookLM 集成**：一名成员询问如何将**完整的官方最新 Mojo 文档**导入 **NotebookLM**，并询问是否有 **PDF** 或 **Markdown** 版本。
   - 另一名成员建议使用 `llms.txt` 文件（[https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt](https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt)）为 **LLM**（如 NotebookLM）提供文档。
- **建议为 NotebookLM 使用 llms.txt 文件**：一名成员建议使用 `llms.txt` 文件（[https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt](https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt)）为 **LLM**（如 NotebookLM）提供文档。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1460815670701719838)** (4 messages): 

> `Qwen3-VL, MoE 实现, 贡献者指南更新` 


- **Qwen3-VL 的 MoE 实现遭到质疑**：一名成员质疑为什么 **Qwen3-VL** 只有 **MoE 实现**，并建议重用来自 [qwen3vl_moe](https://github.com/modular/modular/tree/main/max/python/max/pipelines/architectures/qwen3vl_moe) 的代码，使稠密（dense）**Qwen3VL** 模型（如 [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)）能够工作。
- **贡献者指南欢迎 PR**：一名成员表示欢迎提交 **PR**，理由是缺乏足够的贡献者来跟上整个 **MAX** 生态系统的发展。
   - 他们还指向了 [更新后的贡献者指南](https://github.com/modular/modular/commit/571ae7132c465f155d50f7bbe7cf77064a02e9bf)。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1460759556501016681)** (1 messages): 

> `Glama 排名, 服务器使用指标` 


- **基于使用情况的 Glama 排名**：**Glama** 的创始人澄清说，他们的排名是基于**服务器使用指标**的。
   - 他们邀请通过私信（DM）提供反馈，并表示对任何所谓的排名滥用行为并不知情。
- **创始人回应排名滥用指控**：**Glama** 的创始人确认了自己的身份，并回应了关于其排名系统可能存在滥用的担忧。
   - 他们强调排名是由**服务器使用指标**决定的，并欢迎直接反馈。


  

---

### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1460702731798053165)** (5 messages): 

> `Tasks Spec Implementations, glama.ai/mcp/inspector` 


- **寻求 Tasks 规范客户端实现**：一名成员询问了实现 **Tasks 规范**的客户端应用，寻求 UI 实现示例，另一名成员提到了 Typescript SDK。
   - 作为回应，另一名成员宣布即将提交一个为 **Inspector** 添加任务的 PR，以及一个在 server-everything 中模拟长时运行任务的 PR。
- **glama.ai Inspector 致力于功能对齐**：一名成员在 [glama.ai/mcp/inspector](https://glama.ai/mcp/inspector) 分享了他们 **Inspector** 实现的早期版本，旨在覆盖所有功能。
   - 该成员澄清说，他们内部将其用于 **e2e testing**。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1460700888200118449)** (5 messages): 

> `AI-assisted code generation, Replit, DSPY OS, DSPY Framework` 


- **AI 平台助力代码生成**：成员们注意到有多个平台提供 **AI 辅助代码生成**，例如 **Replit** 和 **DSPY OS**。
   - 这些工具可以自动化各种编码流程，提高生产力。
- **DSPY OS 难寻踪迹**：一名成员询问 **DSPY OS**，指出 *"什么是 DSPy os？我找不到任何相关信息"*。
   - 另一名成员指出，DSPY 更多是一个 **Framework** 而非 Platform，因此目前还没有直接使用 DSPY 构建的类似 Replit 的项目，但你可以使用 DSPY 创建自定义工具或环境来自动化特定的编码任务。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/)** (1 messages): 

hsaliak.: 在使用 aider 时，是否可以为 gemini 模型使用 oauth 登录？它的限制（limits）更高。
  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1460769810316267601)** (1 messages): 

> `Prompt Engineering for Outreach, Clay + AI outreach workflow` 


- **通过 Clay + AI 外联工作流找到工作**：一场关于 **营销外联 Prompt Engineering** 的研讨会将教授如何构建 **Clay + AI 外联工作流**，将信号大规模地转化为 **个性化消息**。
   - 该研讨会承诺 **40%+ 的接受率**和 **18%+ 的回复率**，并包括可重复使用的工作流和复制即用的 prompts；在此处 [注册](https://luma.com/jt1vr0u5) 或 [此处](https://luma.com/mmqa4fhi)。
- **为外联设计 Prompts**：这场 90 分钟的直播研讨会将详细解析为真实客户使用的精确 **Clay + AI** 系统。
   - 研讨会将涵盖端到端 AI 外联工作流、针对高质量且非尴尬外联的 prompting，以及可选的 Apollo, Attio, n8n 集成。