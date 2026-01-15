---
companies:
- anthropic
- langchain
- apple
date: '2026-01-13T05:44:39.731046Z'
description: '**Anthropic** 将其 AI 智能体（agent）产品整合至 **Cowork** 品牌下，把之前的 **Claude Code**
  和 **Claude for Chrome** 等工具集成到一个统一的智能体中。为了保障安全性，该智能体采用了基于 **Apple 虚拟化技术**和 **bubblewrap**
  的沙箱化 Linux 虚拟机环境。


  与此同时，**Anthropic Labs** 进行重组，Mike Krieger 卸任首席产品官（CPO），重点转向通过一个年经常性收入（ARR）超过 10
  亿美元的智能体实验室来实现 **Claude** 的产品化。


  AI 社区正在讨论“氛围编码”（vibe coding）的含义，强调严谨的工程师验证要优于随意的编码。**LangChain** 发布了 **Agent Builder**
  的正式版本（GA），提供无代码但功能强大的智能体编排功能，如记忆、触发器和人工介入审批。一些专家则主张将智能体工具简化为核心文件系统和 bash 访问，以提高效率。此外，开发者利用
  **QEMU** 和沙箱工具对类 Cowork 环境进行的开源复现，凸显了 AI 智能体技术的快速商品化趋势。'
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
title: Anthropic 实验室：Cowork、Claude Code、MCP，以及由 Mike Krieger 和 Ben Mann 领导的技能孵化器。
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

> 2026年1月13日至1月14日的 AI 新闻。我们为您检查了 12 个 subreddits、[**544** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **24** 个 Discord 服务器（包含 **204** 个频道和 **2271** 条消息）。预计节省阅读时间（以 200wpm 计算）：**202 分钟**。**我们的新网站**现已上线，支持完整的元数据搜索，并以极具美感的 vibe coded 风格呈现了所有过往内容。请访问 https://news.smol.ai/ 查看完整的详细新闻，并在 [@smol_ai](https://x.com/Smol_AI) 上向我们提供反馈！

我们利用这段内容将 [Cowork](https://claude.com/blog/cowork-research-preview) 和 [Anthropic Labs](https://www.anthropic.com/news/introducing-anthropic-labs) 接连发布的公告结合起来。

Cowork 是对之前大量工作的整合与产品化，涵盖了从 [Computer Use](https://www.anthropic.com/news/3-5-models-and-computer-use) 到 [将 Claude Code 整合进 Claude Desktop](https://blog.getbind.co/claude-code-is-now-available-on-the-claude-desktop-app/) 再到 [Claude for Chrome](https://www.reddit.com/r/ClaudeAI/comments/1prcypb/anthropic_just_dropped_claude_for_chrome_ai_that/) 的所有成果 —— 如今这些都整合进了一个名为 Cowork 的统一品牌和 UI/通用 Agent 中。

相比之下，Labs 更加简单：这是一次组织架构调整，Mike Krieger 卸任 Anthropic 的 CPO（由前 Meta 同事 [Ami Vora](https://www.linkedin.com/in/amvora/) 接替），现在他与 Ben Mann 共同运营一个年经常性收入（ARR）超过 10 亿美元的 [Agent Lab](https://www.latent.space/p/agent-labs)，致力于将 Claude 产品化。

---

# AI Twitter 摘要

**AI Agent 产品：Claude Code/Cowork、LangSmith Agent Builder 以及“Agent 化”的开发工作流**

- **Claude Cowork + Claude Code 成为“终端原生 Agent”的新基准，沙箱化（sandboxing）成为基本要求**：多条推文关注 Anthropic 的 Cowork 如何通过 **Apple 原生虚拟化技术**启动 **Linux VM**，并运行在沙箱中（例如通过 **bubblewrap**），以遏制不安全指令和失控进程或误删等故障模式（[沙箱细节](https://twitter.com/dejavucoder/status/2010993418630262817)）。大趋势是：Agent UX 正在向“赋予模型文件系统 + Shell + 严格权限”收敛，然后通过人工审核快速迭代。需求端的痛点也显现出来：高级用户希望减少权限提示，而不必被迫选择 `--dangerously-skip-permissions`（[关于摩擦力的抱怨](https://twitter.com/levelsio/status/2011129631001170244)；相关的[笑话](https://twitter.com/LaurencePostrv/status/2011134254051139712)）。

- **“Vibe coding” 的反弹 → Agent 辅助工程更清晰的分类**：一场反复出现的辩论是，“vibe coding” 被误用来描述使用 Agent 完成的、经过仔细验证的生产级工作。Gergely Orosz 认为，当工程师在验证并闭环时，不应再称其为 vibe coding（[推文](https://twitter.com/GergelyOrosz/status/2011001698370699374)）。Yuchen Jin 给出了更尖锐的定义：vibe coding 最初意味着 *完全不看代码*；一旦你进行了审查，它就更接近于“lucid coding”（[推文](https://twitter.com/Yuchenj_UW/status/2011137879112908870)）。这之所以重要，是因为它重新定义了正在发生的变化：并非“工程已死”，而是**具备审美和验证能力的工程师获得了杠杆**。

- **LangSmith Agent Builder 正式发布（GA）：无代码但不只是玩具（MCP、Memory、触发器、收件箱/HITL）**：LangChain 宣布 Agent Builder 已**正式发布**（[GA 公告](https://twitter.com/LangChain/status/2011129282580660314)）。Harrison Chase 及其团队强调了核心原语：**Memory**、**Skills**、**Sub-agents**、**MCP/工具集成**、自动运行的**触发器**，以及用于人工审批的 **Agent 收件箱**（[演示](https://twitter.com/hwchase17/status/2011126016287113681)；GA 回顾[此处](https://twitter.com/hwchase17/status/2011134704934957382)）。多位用户强调，即使对于技术用户，它也非常有用，因为它将编排和可观测性封装得非常简洁（[价值定位](https://twitter.com/KevinBFrank/status/2011154462128144539)）。元经验是：编排产品正从“Prompt + 工具”转向 *运营型 Agent 栈*（身份验证、触发器、审计追踪、受监督的操作）。

- **一种反向趋势：“别挡模型的道”并简化工具**：Jared Z 认为，添加工具/护栏可能会降低性能，因为你在强迫模型做出额外的分支决策；他引用了 Vercel 的例子，将一个 text-to-SQL Agent 简化为仅保留文件系统 + bash 访问权限（[帖子](https://twitter.com/imjaredz/status/2011218314035642464)）。这符合日益增长的共识：*bash + 文件系统是通用的工具调用方式*，现代模型可以承担过去需要 DAG 才能处理的复杂性。

- **开放的重构与“Cowork 克隆”显示出快速的商品化趋势**：一位开发者使用 **QEMU + bubblewrap + seccomp** 构建了一个跨平台的类 Cowork 虚拟机，通过 `vmctl` 工具和 websocket 进行控制 ([推文](https://twitter.com/SIGKITTEN/status/2011077925085347909))。MiniMax 还声称有人利用与 Anthropic 兼容的 API 重构了 Cowork 并将其开源 ([推文](https://twitter.com/MiniMax_AI/status/2011270108166107311))。这释放了一个信号：**Agent 外壳（agent shells）正成为可复制的基础设施范式**，而非专有的竞争护城河。

---

**长上下文 + 记忆：从 RAG 切片之争到递归语言模型与 RL 记忆**

- **文件系统 Agent vs 向量搜索：混合化是真正的出路**：LlamaIndex 对比了 “fs-explorer” 风格的 Agent 与传统 RAG。他们的总结是：文件系统探索可以**更准确**（拥有全文件上下文）但**速度较慢**，而向量搜索在大规模（1k+ 文档）情况下胜出 ([LlamaIndex 帖子](https://twitter.com/llama_index/status/2011121143927972076)；Jerry Liu 的总结见[此处](https://twitter.com/jerryjliu0/status/2011130432205832664))。Weaviate 重申了切片（chunking）中的核心权衡：**检索精度 vs 上下文丰富度**，并且不存在通用的切片大小 ([推文](https://twitter.com/weaviate_io/status/2011088315663978739))。

- **MemRL：将记忆检索视为 RL（效用感知），而非相似度搜索**：DAIR AI 重点介绍了 **MemRL**，它保持基础 LLM 冻结，并学习**情节记忆（episodic memories）的 Q 值**（意图–经验–效用），采用两阶段检索：语义过滤后进行效用排序 ([摘要](https://twitter.com/dair_ai/status/2011086096986443905))。如果这些说法成立，对于生产级 Agent 来说这是一个引人注目的模式：既能避免微调/灾难性遗忘，又能通过学习到的记忆策略**从经验中不断改进**。

- **递归语言模型 (Recursive Language Models, RLMs)：对 Prompt 的符号化访问，而非“作为工具调用的子 Agent”**：Omar Khattab/lateinteraction 的文章认为，大多数“子 Agent”实现都忽略了核心观点：你无法将数百万个子调用具体化为工具调用，你需要**类指针/符号化访问 Prompt** 才能以编程方式对其进行递归 ([评论](https://twitter.com/lateinteraction/status/2011250721681773013))。The TuringPost 的综述将 RLMs 描述为一种推理时架构，它将上下文卸载到 Python REPL 变量中，以便模型可以通过代码对其进行操作，从而在不重新训练的情况下扩展到 **10M tokens** 以上 ([摘要](https://twitter.com/TheTuringPost/status/2011272650132504889))。工程师的核心启示是：*“长上下文”可能越来越多地意味着“通过代码介导的上下文访问”，而不仅仅是更大的窗口。*

- **通过 Prompt 压缩缓解上下文腐化 (Context rot)**：DSPy 被用作减少 Prompt 长度而不损失性能的工作流示例，被明确定义为一种对抗上下文退化的方法 ([推文](https://twitter.com/hammer_mt/status/2011022198023082263))。

---

**视频生成与可控世界模型：Kling 运动控制、Veo 3.1 升级以及新的“世界模型”主张**

- **可灵 (Kling) 2.6 运动控制 (Motion Control) 成为顶级的性能/动作迁移工具（但身份漂移依然存在）**：多位创作者报告称，可灵的运动控制能够以极高的精度替换/驱动场景中的角色 ([热门文章](https://twitter.com/AngryTomtweets/status/2010975679488409890))。一个详细的日本演示展示了乐器演奏的动作迁移，具有高保真的手指动作和节奏，表明单主体镜头已接近写实水平 ([演示线程](https://twitter.com/akiyoshisan/status/2010983687727587587))。Curious Refuge 对其进行了实景叙事测试：视差效果很强，但面部一致性会发生漂移；当参考图像与初始帧接近时效果最好 ([测试](https://twitter.com/CuriousRefuge/status/2011207976095531524))。

- **Google Veo 3.1：“从素材到视频”获得人像模式 + 更高分辨率 + 一致性改进 + SynthID**：DeepMind/Google 推出了 Veo 3.1 更新，强调 (1) **原生竖屏 9:16**，(2) 改进的角色/背景一致性，(3) **1080p + 4K** 选项，以及 (4) 用于验证的 **SynthID 水印** ([DeepMind 线程](https://twitter.com/GoogleDeepMind/status/2011121716336984151)；API 摘要见[此处](https://twitter.com/_philschmid/status/2011122136619110762)；Gemini 应用推出见[此处](https://twitter.com/GeminiApp/status/2011122407013306875)；Sundar Pichai 的帖子见[此处](https://twitter.com/sundarpichai/status/2011143120516469199)；Demis Hassabis 见[此处](https://twitter.com/demishassabis/status/2011236200397639900))。工程师应注意其产品方向：**移动优先格式 + 来源溯源 + 生产级分辨率**的优先级正高于单纯的新奇感。

- **“World model” 品牌化加速；研究基准试图跟进**：PixVerse 将 “R1” 营销为“实时世界模型”（带有浓厚的营销色彩）([tweet](https://twitter.com/PixVerse_/status/2011100288690897317))。更偏技术层面的进展：TencentARC 的 **VerseCrafter** 声称实现了对摄像机和多物体运动的 4D 几何控制 ([announcement](https://twitter.com/wbhu_cuhk/status/2011109476510941222))。此外，还出现了一个名为“针对 Agent 视频推理的开放网络视频深度研究基准” (Video Deep Research Benchmark on Open Web for Agentic Video Reasoning) ([tweet](https://twitter.com/_akhaliq/status/2011105482111651992))，这进一步印证了视频 Agent 的评估体系仍不成熟。

---

**开源模型、端侧机器学习（On-device ML）与多模态医疗 AI：MedGemma 1.5、GLM-Image、MLX 吞吐量**

- **MedGemma 1.5 + MedASR：专注于 *离线* 和 3D 成像的开源医疗多模态技术栈**：Google 发布了 **MedGemma 1.5**，其模型体量足以离线运行，并针对多模态医疗任务进行了改进 ([Google AI Devs](https://twitter.com/googleaidevs/status/2011181120793297361))。Phil Schmid 的技术要点列出了一个支持 **3D 卷轴 (CT/MRI)**、纵向比较和解剖定位的 **4B** 模型；他指出该模型在 **EHR 理解准确率上达到 89.6%**（提升 22%），在 X 射线定位方面的 **IoU 为 38%** ([tweet](https://twitter.com/_philschmid/status/2011183904204390654))。Sundar Pichai 将其定位为“重大升级”，并将其与用于医疗听写的 **MedASR** 配对 ([tweet](https://twitter.com/sundarpichai/status/2011184917670216196))。Google Research 在 Hugging Face 和 Vertex AI 上同步发布了这两项成果 ([tweet](https://twitter.com/GoogleResearch/status/2011185403856883907))。核心结论：**开源、高效、面向临床的多模态模型**正成为一类顶级的发布范畴。

- **GLM-Image：用于“海报/PPT/文本渲染”和重知识生成的混合自回归 + 扩散架构**：智谱 AI（Zhipu AI）发布了 GLM-Image，声称通过混合架构实现了强大的文本渲染和信息图表/海报生成能力 ([release](https://twitter.com/Zai_org/status/2011247591825068314))。第三方细化了架构细节（例如“9B 自回归 + 7B 扩散”）以及“认知生成”的框架 ([fal launch](https://twitter.com/fal/status/2011271561429311512)；ModelScope 回顾见[此处](https://twitter.com/ModelScope2022/status/2011262011997651194))。对于工程师而言，关键在于其*设计目标*：提高 **布局 + 多行文本** 的可靠性，而这通常是扩散模型的弱点。

- **端侧和本地推理需求持续攀升**：LocallyAI 的更新指出 **LiquidAI LFM 2.5** 模型现可通过 MLX 在 iOS 上运行 ([tweet](https://twitter.com/LocallyAIApp/status/2011136235973329301))。MLX 性能基准测试显示，MiniMax M2.1 在 M3 Ultra 上本地运行并开启连续批处理（continuous batching）：**4-bit 量化下，32 个并发请求达到 220 tok/s** ([tweet](https://twitter.com/ivanfioravanti/status/2011115626690179290))。Awni Hannun 强调 MLX 增加了跨 Metal 和 CUDA 的量化支持（nvfp4/mxfp8） ([tweet](https://twitter.com/awnihannun/status/2011267993091875282))。在关于 Claude/Cowork 的讨论中，也出现了针对“本地模型”隐私性的抨击 ([tweet](https://twitter.com/victormustar/status/2011078287762825474))。

---

**基准测试、评估与 Agent 可靠性：指令遵循、视觉推理极限以及“枯燥的 Agent”**

- **OctoCodingBench：对齐的 Coding Agent ≠ 能通过单元测试**：MiniMax 发布了 **OctoCodingBench**，用于衡量 Coding Agent 是否遵守系统提示词（system prompts）、代码库规范和工具策略——明确针对代码库中的“回形针最大化”（paperclip-maxing，指过度优化单一指标而忽略全局约束）行为 ([tweet](https://twitter.com/MiniMax_AI/status/2011266592303432058)；数据集提及见[此处](https://twitter.com/HuggingPapers/status/2011074090686136349))。这是一个重要的转变：将评估从纯粹的功能正确性转向 **过程约束** 和组织规范。

- **BabyVision：MLLM 在“纯视觉推理”方面依然薄弱**：HuggingPapers 引用了 BabyVision 的结果：在 388 项任务中，SOTA MLLM 的表现为 **49.7%**，而成年人类为 **94.1%**，理由是这些任务需要非语言的视觉理解能力 ([tweet](https://twitter.com/HuggingPapers/status/2011048605113581762))。如果你正在构建多模态 Agent，这意味着那些“看起来已解决”的演示可能会掩盖其脆弱的视觉推理能力。

- **将企业级“枯燥的 Agent” (Boring Agents) 作为一种产品立场**：AI21 明确推销其“枯燥的 Agent”，这些 Agent 针对 **可审计、可重复** 的输出进行了优化，而非追求聊天魅力 ([tweet](https://twitter.com/AI21Labs/status/2011041313039204838))。这与评估趋势相一致：减少对“感觉”的关注，增加对治理的关注。

- **METR：从“能力”扩展到失控风险框架**：Ajeya Cotra 加入 METR，将失控（LOC）风险评估扩展到“手段、动机、机会”三个维度，并指出动机和机会的衡量目前尚不成熟，但在未来可能变得至关重要 ([tweet](https://twitter.com/ajeya_cotra/status/2011146702175289563)；定义见[此处](https://twitter.com/ajeya_cotra/status/2011146714183581886))。

---

**基础架构 + 训练系统：调度器、Attention 后端、量化陷阱以及 FP8/低比特训练**

- **HPC 调度器 vs 云原生编排（Slurm 被收购后的讨论）**：dstack 将 Nvidia 收购 Slurm 视为工作负载正转向云原生调度器的证据，并提供了 Slurm→dstack 的迁移指南 ([tweet](https://twitter.com/dstackai/status/2011091749901422904))。SkyPilot 推广 “Pools” 作为跨 K8s + 云的统一批处理队列 ([tweet](https://twitter.com/skypilot_org/status/2011128941705339270))。目前的趋势是：基础架构团队正在围绕**多集群 GPU 池化**和厂商无关的调度器进行标准化。

- **Diffusers 新增 “Unified Attention” 后端**：Hugging Face Diffusers 发布了一个新的 Attention 后端，结合了 Ring 和 Ulysses 的特性 ([tweet](https://twitter.com/RisingSayak/status/2011092823828021730))。这是持续推动 Attention kernels/backends 可交换性和性能可移植性工作的一部分。

- **量化/训练的细微差别持续带来挑战**：TensorPro 报告称 MXFP4 量化的 Attention 可能会破坏因果建模，并发布了关于诊断和修复 “leaky quantization” 行为的文章 ([tweet](https://twitter.com/tensorpro/status/2011198742406578252))。另外，一篇关于 **stochastic rounding** 缓解低精度训练（FP8/4-bit）中梯度消失问题的 Google Cloud 文章被广泛分享 ([tweet](https://twitter.com/dl_weekly/status/2011060892897558717))。对于从业者而言：“在 FP8/低比特下训练” 越来越可行，但**数值边缘情况**仍是活跃的研究/Ops 问题。

---

**热门推文（按互动量排序）**

- **McDonald’s Japan** “Black Pepper!! PV” 帖子（获得海量病毒式传播） ([tweet](https://twitter.com/McDonaldsJapan/status/2010985164692668892))  
- **Joe Rogan** 关于 “查验证件” / 军事化执法的片段 ([tweet](https://twitter.com/OfTheBraveUSA/status/2011153857976668290))  
- **Anthropic 向 Python Software Foundation 捐赠（150 万美元）** ([Alex Albert](https://twitter.com/alexalbert__/status/2011143093266104800)；PSF 的致谢见[此处](https://twitter.com/ThePSF/status/2011060802321584414))  
- 捕捉文化时刻的 **Claude Code / Agent 生产力梗图** ([nearcyan 的 “工程师时代”](https://twitter.com/nearcyan/status/2011129737578500526)；giffmana 的 Cowork 恶搞见[此处](https://twitter.com/giffmana/status/2011165027374334221))


---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Pocket TTS 与本地 AI 工具

  - **[kyutai 刚刚推出了 Pocket TTS：一个拥有 100M 参数的文本转语音模型，具有高质量的语音克隆功能，可以在笔记本电脑上运行——不需要 GPU](https://www.reddit.com/r/LocalLLaMA/comments/1qbpz5l/kyutai_just_introduced_pocket_tts_a_100mparameter/)** (热度: 494): ****Kyutai Labs** 发布了 **Pocket TTS**，这是一个拥有 `100M-parameter` 的文本转语音 (text-to-speech) 模型，专为高质量语音克隆设计，可在 CPU 上高效运行而无需 GPU。该模型可通过 [GitHub](https://github.com/kyutai-labs/pocket-tts) 和 [Hugging Face](https://huggingface.co/kyutai/pocket-tts) 获取，并在 [博客文章](https://kyutai.org/blog/2026-01-13-pocket-tts) 中有详细介绍。该模型的架构灵感来自连续音频语言模型 (continuous audio language models) 的最新进展，如相关的 [arXiv 论文](https://arxiv.org/abs/2509.06926) 中所述。** 一些用户质疑该模型的性能，认为这种规模的模型与大型模型或 Twitch 等应用程序中使用的“硬编码”解决方案相比，可能无法提供足够的质量。人们对该模型的语言能力以及跨不同语言进行微调 (fine-tuning) 的潜力也表现出了兴趣。

    - 一位用户指出 Pocket TTS 模型存在严重的内存管理问题，其本地测试服务器设置在生成任务之间不会清除内存，导致内存使用量大幅增加。他们报告称系统的内存占用达到了 **32 GB**，并建议模型在开始新的生成任务时应清除内存，以防止此类膨胀。
    - 另一位用户提供了 Pocket TTS 模型在 Ryzen **5950X** CPU 上的详细性能分析。他们观察到该模型初始使用约 **1.1 GB 的 RAM**，并且能够快速生成音频，首个音频输出时间 (time to first audio) 约为 **200 ms**。然而，随着上下文的填充，RAM 使用量会显著增加，单个文章的生成可能达到 **8.5 GB**。他们还评论说，对于这种规模的模型，其语调 (intonation) 表现良好，但整体音质被描述为一般。
    - 一位用户对 Pocket TTS 等小模型的实用性表示怀疑，认为与更成熟的解决方案相比，如果需要高质量的输出，这些模型可能不值得投入精力。他们提到尝试了演示版本并发现结果不令人满意，暗示大型模型或硬编码解决方案在某些应用中可能更有效。

  - **[我制作的一个用于简化运行本地 AI 模型的 Windows 工具](https://www.reddit.com/r/LocalLLM/comments/1qbzd2w/a_windows_tool_i_made_to_simplify_running_local/)** (热度: 28): ****V6rge** 是一款基于 Windows 的工具，旨在通过捆绑和隔离其自身的运行时 (runtime) 来简化本地 AI 模型的运行，从而避免系统 Python 冲突。它支持通过 GGUF 运行 Qwen、DeepSeek 和 Llama 等本地 LLM，以及使用 Stable Diffusion 和 Flux 变体进行图像生成，还支持基础的语音和音乐生成。该工具旨在减少安装摩擦，可在 [GitHub](https://github.com/Dedsec-b/v6rge-releases-/releases/tag/v0.1.4) 上下载。** 有人担心该工具不是开源的，这使得用户对运行可执行文件犹豫不决。此外，用户报告了在更改设置时出现“Failed to Save Settings: API error 404”等问题，表明可能存在稳定性问题。改进建议包括为生成的图像添加画廊功能。

    - 一位用户报告了一个关键问题：在设置中更改模型文件夹 (Model Folder) 会导致“API error 404”。这表明应用程序的设置管理中可能存在 Bug，可能是由于错误的 API 端点 (endpoint) 处理或该功能缺乏后端支持。
    - 另一位用户在尝试下载 Qwen-Image 或 FLUX.1-dev 等特定模型时遇到了“Error: undefined”。这表明模型下载功能可能存在问题，可能与错误的 URL 处理或服务器端问题有关。
    - 用户请求提供该工具的 Linux Docker 版本，突显了对跨平台兼容性的需求。用户建议了 Docker Compose、维护良好的 Docker Hub 镜像以及 Portainer 支持等功能，这将有助于在容器化环境中更轻松地部署和管理。

### 2. GLM-Image 与 NER 模型发布

  - **[GLM-Image 已发布！](https://www.reddit.com/r/LocalLLaMA/comments/1qc9m6x/glmimage_is_released/)** (活跃度: 393): **GLM-Image** 是一款新发布的图像生成模型，采用了混合自回归和扩散解码器（autoregressive and diffusion decoder）架构。它在通用图像质量上与主流的 latent diffusion 模型相媲美，但在文本渲染和知识密集型场景中表现卓越，展示了优越的语义理解和复杂信息表达能力。该模型支持文本生成图像以及各种图像到图像的任务，如图像编辑、风格迁移和保持身份（identity-preserving）的生成，同时保持了高保真度和精细的细节。该模型以 **MIT license** 发布，与其西方实验室更具限制性的许可证相比，其开放性备受关注。该模型的性能与 **nano banana 2** 进行了比较，表明它是一项重大进展，尤其是其结合了编辑和生成的能力。

    - GLM-Image 在 MIT license 下发布被视为一个显著优势，特别是与通常发布更具限制性许可证模型的西方实验室相比。这种开放许可可能促进社区内更广泛的采用和创新。
    - 据报道，GLM-Image 在基准测试中的表现与 'nano banana 2' 相当，考虑到它兼具编辑和生成双重能力，这一点非常值得关注。这种双重功能使其在各种应用中成为一种多功能工具，增强了对开发者和研究人员的吸引力。
    - 该模型由一个 13GB 的 diffusion 模型和一个 20GB 的 text encoder 组成，表明其对资源有较高要求。人们期待该模型能够被量化到 fp8，并开发出如 LoRA 等高效训练方法，以便更易于进行实验。

  - **[500Mb 命名实体识别 (NER) 模型，可本地识别和分类任何文本中的实体。轻松在本地对任何语言进行微调（见西班牙语示例）。](https://www.reddit.com/r/LocalLLM/comments/1qbnezw/500mb_named_entity_recognition_ner_model_to/)** (活跃度: 13): **一款全新的 `500Mb` 命名实体识别 (NER) 模型已发布，能够在本地识别和分类文本中的实体。该模型设计旨在轻松进行跨语言微调，并提供了一个针对西班牙语的具体示例。其紧凑的体积允许在不依赖云资源的情况下进行高效的本地部署，非常适合对隐私敏感的应用。然而，帖子中并未说明模型的架构和训练细节。** 该帖子缺乏详细的技术讨论或辩论，因为置顶评论是非技术性的，仅仅表达了认可。

### 3. AI 硬件创新

  - **[AI TOP 100 M.2 SSD](https://www.reddit.com/r/LocalLLM/comments/1qbvycy/ai_top_100_m2_ssd/)** (热度: 26): **该图片展示了一款 GIGABYTE AI TOP 100E M.2 SSD，其市场宣传称通过提供高带宽来增强 AI 性能，从而可能减轻 RAM/VRAM 的负载。然而，评论者认为这在很大程度上是一个营销噱头，因为即使是最快的 PCIe 5 SSD（约 `10GB/s`）的带宽也显著低于 DDR5 RAM (`80GB/s`)。这使得 SSD 在卸载大型 AI 模型时效果较差，因为速度会成为瓶颈，尤其是对于稠密模型（dense models）。稀疏模型（sparse models）可能会稍微受益，但性能提升仍受限于每秒低个位数的 tokens per second。** 评论者对该产品的主张表示怀疑，认为这更多是一种营销策略，而非 AI 工作负载的实际解决方案。他们建议转而使用目前性能最好的 NVMe PCIe 5 SSD，因为此类特定产品的性能增益微乎其微。

    - Themash360 强调了在 AI 工作负载中使用 NVMe SSD 的局限性，指出即使 PCIe 5 拥有理想化的 10GB/s 带宽，与 DDR5 RAM 的 80GB/s 相比仍然慢得多。他们举例说明：在一个 240GB 的稠密模型中，若将 100GB 卸载到 NVMe，其 token 生成速度仅为 0.1 tokens per second，强调了这对于稠密模型的高效性缺失。
    - Themash360 还提到，虽然使用混合专家模型 (MoE) 可以通过卸载稀疏区域来减轻部分性能损失，但改进有限，仅能达到每秒低个位数的 tokens per second。这凸显了在处理大型 AI 模型时，现有存储技术实现高性能所面临的挑战。
    - desexmachina 指出，更快的 SSD 可能导致更高的处理器饱和度，这意味着虽然存储速度是一个因素，但整体系统性能还取决于 CPU 处理增加的数据吞吐量的能力。这表明需要平衡系统架构以优化 AI 工作负载。

  - **[My wishes for 2026](https://www.reddit.com/r/LocalLLaMA/comments/1qbw325/my_wishes_for_2026/)** (热度: 767): **该图片是一份关于 2026 年技术进步的推测性愿望清单，涵盖了 AI 模型和硬件的潜在发展。其中包括 GPT-OSS, Gemma 4, Qwen 4 和 GLM Air 等 AI 模型新版本的发布，以及预计将超越 Mistral 123B 模型的 Llama 5。此外，还希望看到参数量低于 200B 的 DeepSeek 模型，以及显存超过 32GB 且价格亲民的 GPU。该图片反映了对 AI 能力和硬件普及度取得重大进展的渴望。** 评论者对显存超过 32GB 且价格亲民的 GPU 的可行性表示怀疑，强调了 GPU 价格居高不下的持续挑战。

    - SlowFail2433 强调了 GPT-OSS 120B 被低估的性能，指出其令人印象深刻的 benchmark 分数与参数量之比，以及有效的 FP4 量化。该模型与 Qwen 4 系列形成了对比，后者在 Arxiv 论文中被频繁引用，特别是在智能体强化学习 (Agentic RL) 应用方面。讨论强调了小型稠密模型在训练过程中避免 MoE 门控复杂性的优势，因为在 RL 场景中，MoE 门控会使信用分配 (credit assignment) 变得复杂。

  - **[I'm building a real-life BMO with a Raspberry Pi 5 (Mistral/OpenAI + YOLO11n)](https://www.reddit.com/r/LocalLLM/comments/1qbwc35/im_building_a_reallife_bmo_with_a_raspberry_pi_5/)** (热度: 8): **该项目涉及使用集成了 Mistral/OpenAI 提供 AI 能力和 YOLO11n 进行物体识别的 Raspberry Pi 5 构建一个现实版的 BMO。开发者正在通过面部和语音识别功能增强这个 AI 伙伴，旨在实现互动游戏体验。未来的计划包括增加机械臂。该项目是开源的，代码可在 [GitHub](https://github.com/ivegotanheadache/BMO) 上获得。** 一位评论者也在开发类似的项目，使用带有语音识别和文本转语音功能的 LLM 助手，并考虑通过 RetroArch 和 Pico 8 加入国际象棋和模拟器等游戏功能。他们正在考虑是集成专用监视器还是使用外部显示器。



## 技术性较低的 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. xAI 的 Grok 部署及其争议

  - **[官方：五角大楼确认在国防运营中部署 xAI 的 Grok](https://www.reddit.com/r/singularity/comments/1qbo516/official_pentagon_confirms_deployment_of_xais/)** (热度: 1443): **美国国防部（US Department of Defense）将把 xAI 的 Grok AI 集成到五角大楼系统中，允许军事和文职人员处理 5 级影响水平（Impact Level 5）的可控非密信息（Controlled Unclassified Information）。Grok 将被嵌入到作战和规划系统中，利用来自开源和社交数据的实时全球信号，增强情报分析、决策制定和军事规划。该部署目标是扩展至约 `300 万用户`，首阶段将于本月开始。[来源](https://www.washingtonpost.com/business/2026/01/12/artificial-intelligence-pentagon-hegseth-musk/ec8b407a-f026-11f0-a4dc-effc74cb25af_story.html)。** 评论反映了对 AI 集成到军事行动中的怀疑和担忧，一些用户幽默地暗示了潜在的安全风险，而另一些人则对现任政府使用此类技术表示不信任。


  - **[卫报：埃隆·马斯克的 Grok 如何每小时生成 6,000 张未经同意的裸体图像。](https://www.reddit.com/r/OpenAI/comments/1qbkpw9/the_guardian_how_elon_musks_grok_generated_6000/)** (热度: 392): **《卫报》（Guardian）的调查强调了对 **Elon Musk 的 AI 工具 Grok** 的严重滥用。据报道，在 2026 年初，该工具被用于每小时生成 `6,000` 张未经同意的裸体图像。这种滥用是更广泛趋势的一部分，用户利用 AI 创建性化和暴力图像，特别针对女性和未成年人。该报告强调了 AI 技术在内容审查和用户安全方面带来的伦理和监管挑战。** 评论者对 Grok 的滥用表示幻灭，注意到用户倾向于专注于生成露骨内容的趋势。此外，还有对 **US Big Tech** 以及社区对 AI 功能反应的批评，一些用户由于色情内容讨论的盛行而退订了相关论坛。

    - Fearless_Weather_206 提出了一个关于 Grok 功能可能带来的立法影响的关键点。其担忧在于，此类事件可能被用作监管或限制开源 LLM 模型的借口，而这些模型通常比商业模型受到的审查更少。这可能引发关于 AI 发展中创新与监管平衡的更广泛辩论。
    - boredatwork8866 强调了一个社区趋势，即用户主要专注于利用 Grok 生成成人内容。这表明用户对该模型创建露骨内容的能力有着浓厚兴趣，而当此类功能受到限制时，会导致不满，暗示了用户期望与开发者施加的伦理准则之间的紧张关系。
    - Joddie_ATV 对 Grok 生成未经同意图像的能力所带来的伦理和社会影响表示担忧。这提出了关于 AI 开发者在防止技术滥用方面的责任，以及当前保护措施在防范此类滥用方面的有效性问题。

  - **[没什么大不了](https://www.reddit.com/r/OpenAI/comments/1qc2b0f/nothing_could_go_wrong/)** (热度: 365): **该图片是一个迷因（meme），幽默地评论了美国国防部长宣布将 **Elon Musk 的 xAI 平台 Grok** 集成到军事网络中的消息。据路透社报道，此次集成是 AI 加速战略的一部分。Jarvis 的推文讽刺地暗示这种集成是低风险的，含沙射影地表达了对这一举措潜在影响的担忧。评论反映了对将 AI 集成到军事行动中可能产生的后果的怀疑和幽默，并引用了反乌托邦场景和政治评论。** 评论表达了怀疑和幽默，引用了诸如《终结者》系列中的反乌托邦场景，以及对私营公司在政府合同中影响力的政治评论。

### 2. DeepSeek 的 Engram 模块与创新

  - **[[R] (DeepSeek) Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://www.reddit.com/r/MachineLearning/comments/1qbnkrn/r_deepseek_conditional_memory_via_scalable_lookup/)** (热度: 55): **DeepSeek** 引入了一个名为 **Engram** 的新型模块，通过条件记忆（conditional memory）实现了一种新的稀疏性维度，从而增强了大型语言模型，实现了高效的 O(1) 查找。这种方法与传统的 Mixture-of-Experts (MoE) 模型不同，它通过优化神经计算与静态记忆之间的平衡，展现出一种 U 型缩放法则（scaling law）。**Engram** 扩展到了 `27B parameters`，在包括 `MMLU`、`CMMLU`、`BBH`、`ARC-Challenge`、`HumanEval` 和 `MATH` 在内的各种基准测试中，优于等参数和等 FLOPs 的 MoE 基准模型。该模块通过卸载早期层的静态重构并增强对全局上下文的注意力容量，改善了推理和检索能力，在长文本检索任务中取得了显著收益。**Engram** 的确定性寻址还支持运行时预取（prefetching），最大限度地减少了开销并提高了基础设施效率。一条评论强调了 **Engram** 的实际效率，指出它能够避免为重新计算常见事实而进行不必要的正向传播，从而在不增加复杂性的情况下提高了吞吐量和效率。


  - **[DeepSeek V4 Could Blow Claude and GPT Away for Coding](https://www.reddit.com/r/DeepSeek/comments/1qblbjf/deepseek_v4_could_blow_claude_and_gpt_away_for/)** (热度: 226): **DeepSeek V4** 即将发布，据称其在编程任务中将超越 **Claude** 和 **GPT**。该模型引入了 **Engram 模块**，该模块使用记忆查找系统通过解耦记忆与计算来管理超长提示词，可能通过让 Attention 层和 MLP 层专注于复杂任务来增强性能。这种架构还可以将 VRAM 需求降低 `30%`，将其卸载到 RAM 中。然而，对于有关 DeepSeek 能力的泄露，存在一些质疑。一位用户分享了使用 DeepSeek 开发复杂加密器的经验，指出其编程效率很高，与 Meta 和 ChatGPT 相比代码行数更少。然而，他们发现 Claude 在处理特定功能和提供更具批判性的代码审查方面更胜一筹，尽管 DeepSeek 的鼓励分更高。

    - **Engram** 是 DeepSeek 中的一项功能，旨在通过卸载较简单的任务来优化性能，从而释放 Attention 和 MLP 层以进行更复杂的处理。这种方法使模型表现得更深，从而可能提高其效率。此外，**Engram** 可以通过利用 RAM 代替 VRAM，将 VRAM 需求降低约 30%。
    - 一位用户分享了使用 DeepSeek 构建复杂加密器的经验，指出它比 Meta 和 ChatGPT 更高效，所需的代码行数更少。然而，对于某个特定功能，只有 Claude 在其他模型失败的情况下获得了成功。在对完成的代码进行审查时，DeepSeek 评分 8.3/10，ChatGPT 评分 6.8/10，而 Claude 最初评分 5.5/10，经过辩论后调整为 6.5。
    - 由于 DeepSeek 团队规模小于 Anthropic、OpenAI 或 Google，人们对 DeepSeek V4 在编程任务中超越 Claude 和 GPT 的潜力持怀疑态度。虽然 DeepSeek 在推理概念方面走在前列，但真正的考验将是其在 Agent 和编程任务中的表现，这些对该团队来说是相对较新的领域。之前的 3.2 版本因其在特定领域自动化任务中的强劲表现而受到关注。

  - **[DeepSeek Unveils Engram, a Memory Lookup Module Powering Next-Generation LLMs](https://www.reddit.com/r/DeepSeek/comments/1qbozaf/deepseek_unveils_engram_a_memory_lookup_module/)** (热度: 80): **DeepSeek** 推出了一种名为 **Engram** 的新模块，旨在通过集成记忆查找系统来增强大语言模型 (LLMs)。该系统结合使用 `N-gram embeddings` 和神经骨干（neural backbone），以减少“静态知识”的计算负载。挑战在于实现有效的上下文感知门控（context-aware gating）以优化这种集成，这可能会显著提高 LLM 的推理能力。评论者对 **Engram** 影响 LLM 内存管理的潜力深感兴趣，一些人猜测 **OpenAI** 或 **Google** 等巨头是否会采用它。关于上下文感知门控在增强这些模型推理能力方面的有效性，存在技术上的讨论。

    - 将 N-gram 嵌入的查找与神经骨干结合使用，可以显著降低对“静态知识”的计算需求。挑战在于实现有效的上下文感知门控，这对于推理任务至关重要。DeepSeek 在解决此门控问题上采取的方法，对于提高此类系统的效用可能具有关键意义。

### 3. Claude Code 与 Ralph Wiggum 技术

  - **[TRUST ME BRO: Most people are running Ralph Wiggum wrong](https://www.reddit.com/r/ClaudeCode/comments/1qc4vg0/trust_me_bro_most_people_are_running_ralph_wiggum/)** (热度: 225): **该帖子讨论了使用 “Ralph Wiggum” 作为一种连续循环运行 Claude Code 等 AI 编程工具的方法，旨在解决过早停止等限制。作者批评了官方的 Claude Code Ralph 插件，认为其在处理上下文窗口 (context windows) 方面效率低下，导致上下文臃肿和幻觉 (hallucinations)。相反，他们提倡使用最初由 Geoffrey Huntley 开发的 bash 循环，该循环在每次迭代时启动全新的上下文，使其更适合长时间运行的任务。关键设置建议包括：使用沙盒 (sandbox) 以保证安全、使用结构化任务列表以提高效率、设置迭代限制以控制成本，以及使用 Playwright 或 Claude for Chrome 等工具实现反馈循环。作者提供了 [YouTube 演示](https://youtu.be/eAtvoGlpeRU) 和 [GitHub 指南](https://github.com/JeredBlu/guides/blob/main/Ralph_Wiggum_Guide.md) 以获取更多详细信息。** 评论者强调了 Geoffrey Huntley 原创工作的重要性，并指出他最初获得了免费 tokens，但这可能并不适用于所有用户。人们担心在复杂任务或团队环境中使用 Ralph Wiggum 的实用性，因为错误可能会累积并导致难以管理的 pull requests。

    - Ralph 的创造者 Geoffrey Huntley 最初免费获得了所有的 tokens，这可能影响了 Ralph 的开发和部署策略。这可能意味着用户的成本考量与 Huntley 的原始用例有显著不同，从而潜在地影响 Ralph 在实践中的利用方式。
    - 一个关键的担忧是使用 Ralph 等自动化工具时可能产生的复合错误，特别是在复杂项目中。如果过程中早期出现错误，它可能会传播到后续阶段，导致严重问题。这凸显了仔细监管和迭代反馈的重要性，尤其是在大型 pull requests 可能带来麻烦的团队环境中。
    - 关于 Ralph 与使用 Claude 进行项目规划和执行的有效性存在争议。一些用户发现，在有适当指令的情况下，Claude 可以有效地处理端到端项目阶段，因此质疑 Ralph 提供了哪些额外好处。这表明需要更清晰地区分或演示 Ralph 在自动化复杂工作流方面的独特能力。

  - **[Smart Ralph: A Claude Code plugin for spec-driven development with Ralph-style loops](https://www.reddit.com/r/ClaudeCode/comments/1qbvudj/smart_ralph_a_claude_code_plugin_for_specdriven/)** (热度: 84): **Smart Ralph** 是一个全新的 Claude Code 开源插件，它使用 **Ralph Agent 循环模式** 实现了规范驱动开发 (spec-driven development) 工作流。这种方法解决了 AI-in-IDE 流程中常见的 AI 立即开始编写代码，往往导致实现不完整或不匹配的问题。Smart Ralph 要求 Claude 在编写任何代码之前先进行调研、收集需求、设计架构并分解任务。它为每个阶段使用专门的子 Agent (sub-agents)，确保开发过程结构化且具备上下文感知能力。该插件可在 [GitHub](https://github.com/tzachbon/smart-ralph) 上获取，并可通过插件市场安装。评论者对与普通 Ralph 周期相比的 token 成本感兴趣，一位用户指出 Smart Ralph 似乎比他们自己开发的类似插件（同样不需要 openspec）消耗更少的 token。另一位用户对不必维护自己正在开发的类似项目感到欣慰。

    - azr2001 询问了 Smart Ralph 插件与传统 Ralph 周期相比的 token 成本，表明了对 AI 驱动开发工作流中效率和资源管理的关注。
    - LittleJuggernaut7365 指出 Smart Ralph 插件似乎比他们自己开发的类似插件消耗更少的 token，且后者还需要 “openspec”。这突显了 Smart Ralph 在更高效的资源利用和无需额外依赖的更广泛兼容性方面的潜力。
    - Longjumping_Guess360 为未来的开发提出了一个增强建议：启用 AI 群体 (swarm of AIs) 进行解题竞争，多个 AI 之间的共识可以指示最佳解决方案。这指向了通过协作验证来改进 AI 决策过程的潜在方向。

- **[[D] 有人真的在为 GPU 集群 TCO 咨询买单吗？（因为大多数公司多付了 20% 以上的钱）](https://www.reddit.com/r/MachineLearning/comments/1qbljgq/d_is_anyone_actually_paying_for_gpu_cluster_tco/)** (活跃度: 24): **该帖子讨论了 AI 基础设施采购中的低效问题，强调公司往往只关注 **$/GPU/hour**，而忽视了 **总体拥有成本 (TCO)**。作者指出，**Model FLOPs Utilization (MFU)**、数据出向 (data egress) 和存储中的隐藏成本以及网络效率低下等因素会导致显著的过度支出。他们提议提供咨询服务来帮助公司评估这些因素，从而在计算成本上潜在地节省 20-30%。帖子强调，“真正”的 AI 云可以显著提高 MFU，从而减少大规模模型训练的时间和成本。** 评论者认为，这些问题对于一份简单的报告来说过于复杂，而且许多团队已经意识到这些因素。他们指出，真正的挑战不在于无知，而在于难以准确预测工作负载需求并相应地调整基础设施。一些人对第三方报告的价值表示怀疑，指出组织问题往往是导致过度支付的原因，而不是缺乏关于 MFU 的知识。

    - whyVelociraptor 认为帖子中确定的问题过于广泛，简单的报告无法有效解决。他们建议专业的团队已经意识到这些问题，或者在必要时能够自行解决。该评论还对这种咨询的价值表示怀疑，暗示它可能只是复制了像 ChatGPT 这样的 LLM 可以生成的内容，而用户完全可以免费自行完成。
    - patternpeeker 强调，过度支付通常是由于组织问题，而不是对 Model FLOPs Utilization (MFU) 的无知。他们指出，公司难以准确估计工作负载组合和利用率，导致其采购决策是基于“可辩解”的每小时费率而非最佳方案。评论强调，真正的挑战是做出随着工作负载演变而依然有效的基础设施决策，而不仅仅是理解 MFU。
    - audiencevote 指出，许多人假设所有供应商提供的 H100 GPU 都是相同的，但事实并非如此。他们提到 Model FLOPs Utilization (MFU) 的行业平均水平在 35-45% 左右，而“真正”的 AI 云可以实现显著更高的利用率。这引发了关于“真正”的 AI 云与其他产品之间差异的问题，暗示存在可以增强性能的特定优化或配置。


---

# AI Discord Recap

> 由 gpt-5.1 提供的摘要之摘要总结


**1. 下一代开源及特定领域多模态模型**

- **Zai 的 GLM-Image 结合了 Diffusion 和 Autoregression**: **Zai** 推出了 **GLM-Image**，这是一个开源图像模型，采用了 **自回归 (Autoregressive) + 扩散 (Diffusion)** 的混合架构，旨在实现 **高保真细节** 和 **清晰的文本渲染**。该消息通过其 [GLM-Image 博客文章](https://z.ai/blog/glm-image) 发布，并在 [GitHub: GLM-Image](https://github.com/zai-org/GLM-Image) 上提供了代码支持。该模型针对 **知识密集型生成** 具有强劲表现，并支持一系列丰富的 **图像到图像任务**，如编辑、风格迁移、保持身份 (identity-preserving) 的生成以及多主体一致性。相关的部署文件也已按照 [Latent Space GLM-Image 讨论](https://xcancel.com/zai_org/status/2011247591825068314) 中的引用分享至 **Hugging Face**。
  - **Latent Space** 和 **Nous Research** 的社区讨论强调了 GLM-Image 与“主流 Latent Diffusion 基准模型”相比在 **文本渲染** 方面的优越性，同时根据 [z.ai GLM-Image 博客](https://z.ai/blog/glm-image) 的说法，其在通用图像质量方面与之基本持平。用户将其视为 **开源多模态技术栈** 的重要构建模块，可与 **Qwen3-VL** 等工具搭配，并集成到已经使用开源后端的创意工作流中。

- **LTX-2 凭借本地开源视频生成迈入 4K 时代**：**Venture Twins** 宣布推出 **LTX-2**，这是一个**开源视频生成模型**，能够生成带有**音频**的、长达 **20 秒的 4K 片段**，Justine Moore 在推文中展示了该模型并链接至 [LTX-2 开源视频模型](https://xcancel.com/venturetwins/status/2010878914273697956)。该模型专为**本地执行**而设计，使工程师能够在自己的硬件上运行高分辨率、支持音频的视频合成，而无需依赖受限的云端 API。
  - 在 **Latent Space 的 genmedia 频道**中，成员们称 LTX-2 是 **DIY 视频工具**的一项突破，并指出创作者 *yanokusnir* 在 [LTX-2 发布线程](https://xcancel.com/venturetwins/status/2010878914273697956)中展示了直接利用开源权重生成的**端到端 4K 片段**。工程师们已经在讨论将 LTX-2 与 **RAG 故事流水线**结合使用，并将其作为 Veo 等闭源模型的透明替代方案，据报道 Perplexity 通过 **Veo3.1 驱动的视频生成**功能展示了后者。

- **Qwen Image Edit 将图片转化为 Gaussian Splats**：**Latent Space** 的开发者们强调了 **Qwen Image Edit** 的能力，即通过 [Qwen-Image-Edit-2511-Gaussian-Splash 模型](https://huggingface.co/dx8152/Qwen-Image-Edit-2511-Gaussian-Splash)将图像转换为 **Gaussian Splats** 并从新视角进行重新渲染。该工作流可以有效地从单帧构建 3D 表示，从而在保持周围几何结构一致性的同时，实现**起始帧 → 结束帧**的视频渲染。
  - 用户将这种 Gaussian Splats 流水线视为 **2D LLM 条件编辑**与完整 **3D 场景重建**之间的务实桥梁，能够无缝切入游戏和 VFX 的资产流水线。对话将 Qwen Image Edit 定位为 **GLM-Image** 和 **LTX-2** 等模型的补充，其中 Qwen 处理**视角一致的场景**，而其他模型则处理**高保真帧**和**时序视频**。

- **MedGemma 1.5 推动医学视觉与语音技术**：Google Research 宣布推出 **MedGemma 1.5**，作为用于**医学图像解读**和**医学语音转文本**的下一代模型，详见其博客文章 [“Next-generation medical image interpretation with MedGemma 1.5 and medical speech-to-text with MedASR”](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/)。该模型针对**临床成像工作流**和**医学音频的 ASR**，旨在支持研究和现实世界的医疗护理场景。
  - 在 **Yannick Kilcher 的 ML 新闻**频道中，工程师们将 MedGemma 1.5 视为**领域调优的视觉语言模型**日益成熟的又一信号，同时在概念上将其与非医学用途的 GLM-Image 和 Qwen3-VL 等开源项目配对。讨论更多集中在统计方法论（频率派 vs 贝叶斯派）而非 MedGemma 的架构上，但博客将其定位为一个专业的、安全至上的多模态栈，而非通用消费者模型。


**2. GPU 内核、CUDA 竞赛与 Helion 0.2.10**

- **Helion 0.2.10 为 Flex Attention 提供 SM 过度订阅支持**：**GPU MODE** 服务器宣布推出 **Helion 0.2.10**，该版本发布了一个 [Flex Attention 示例内核](https://github.com/pytorch/helion/blob/main/examples/flex_attention.py)，并增加了对持久化内核上 **Streaming Multiprocessors (SMs) 过度订阅**的支持。一张共享图表展示了过度订阅如何影响 **softmax** 内核，为从业者在权衡占用率（occupancy）与延迟（latency）时提供了具体的参考。
  - 内核黑客将 Helion 0.2.10 视为高级启动配置的**实战手册**，利用 Flex Attention 示例在 **GPU MODE 的 NVIDIA 挑战赛**等竞技场景中探索**非标准 Attention 布局**。过度订阅支持与关于 B200 运行器上 **dual GEMM 稳定性**的更广泛讨论相契合，在这些场景中，细微的基础设施细节（散热、调度器）会实质性地影响基准测试的可复现性。

- **B200 GEMM 不稳定性导致排行榜拆分**：由于 **B200 runners** 在处理 **dual GEMM 问题**时的测量结果不稳定，GPU MODE 将提交截止日期延长至 **1月20日**，并将比赛拆分为两个阶段，具体细节见其[状态更新消息](https://discord.com/channels/1189498204333543425/1343350424253632695/1460796244824686806)。当前的排行榜将开放至 **1月16日**，而 **新排行榜** 将于 **1月17日** 启动，其分数将作为发放奖金的唯一依据，**Problem #4** 的运行时间为 **1月20日至2月20日**。
  - 组织者将这种不稳定性归因于 **eval code、散热（thermals）以及调度基础设施（scheduling infra）** 的交集，这凸显了即便是针对像 dual GEMM 这样的单一 kernel 类别，近硬件级（near-hardware）的基准测试也可能非常脆弱。参赛者现在需要在新的时间窗内 **重新运行并重新验证 kernel**。在追求 B 系列 GPU 的边际吞吐量增益时，Helion 风格的工具和更好的 **profiling workflows** 显得尤为重要。

- **PTX SMEM 指针和矩阵描述符困扰 CUDA 开发者**：在 **GPU MODE 的 CUDA 频道**中，一名成员分析了为什么像 `mbarrier.init.shared.b64` 这样的 PTX 指令在 `r` 寄存器中需要 32 位 SMEM 指针，尽管其语义指向的是 64 位 barrier 操作。
  - 另一位工程师指向了关于 [warpgroup-level matrix shared memory layout and matrix descriptors](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor) 的 NVIDIA PTX 文档，阐明了 **wgmma** 使用的是 **packed descriptor（打包描述符）**，而非通用指针。讨论深入到了为什么 **8×2 "核心矩阵"** 是通过 8×16 字节切片而非 8×32 模式表示的底层问题，凸显了在为 Hopper/Blackwell 时代的 tensor cores 设计 kernel 时，仍有许多未公开的约定（undocumented convention）在发挥作用。

- **CUDA 学习者、B200 提交以及旧硬件 Hack**：一位具有 **Python/PyTorch/TF/C++** 背景的 AI 工程专业学生询问了入门级 **CUDA 资源**，资深人士推荐了会议和研讨会资料，如 **PyTorch Dev talks、ASAP Seminar、ICML/ICLR/NeurIPS、MLSys.org、ASPLOS**，以及在通过 [GPU MODE web UI](https://www.gpumode.com/v2/home) 或 [kernelbot](https://gpu-mode.github.io/kernelbot/docs/intro) 提交 kernel 之前的 YouTube 入门介绍。初次参赛者在本地测试后成功向 B200 提交了作业，表明比赛流程对于纯 CUDA 专家以外的人员也是可行的。
  - 与此同时，**LM Studio** 和 **Unsloth** 服务器讨论了在受限设备上运行大型模型的方法，参考了 **AirLLM 的逐层加载（layer-at-a-time loading）** 以将 **70B 模型适配到 4 GB GPU**，以及在 **DDR4 内存和 Xeon** 上运行 LLM 的轶事。这些 Hack 手段结合 **Helion** 和 **dual GEMM** 调优，描绘了从 **爱好者低预算推理** 到使用 B 系列硬件进行 **最前沿 kernel 竞赛** 的技术图谱。


**3. 基准测试、Agent 懒惰以及寻找低拒绝率 LLM**

- **SlopCodeBench 羞辱懒惰的编程 Agent**：**Eleuther 的研究频道**的研究人员推广了 **SlopCodeBench**，这是一个新的基准测试和博客项目。G. Orlanski 在一条推文中介绍了该项目，链接到了 [SlopCodeBench: measuring agent laziness](https://x.com/GOrlanski/status/2011156105255346505) 及其代码库 [SprocketLab/slop-code-bench](https://github.com/SprocketLab/slop-code-bench)。SlopCodeBench 将大型编程任务分解为 **多检查点问题（multi-checkpoint problems）**，这些问题会惩罚糟糕的早期设计选择而不提供实现提示，迫使 Agent 真正进行规划，而非仅仅进行样板代码的模式匹配。
  - 社区将 SlopCodeBench 的 **Agent 风格评估** 与更多依赖提示词（prompt-heavy）的编码基准测试进行了对比，认为具有 **现实上下文窗口的简单提示词** 比深度工程化的系统提示词更能代表真实世界的使用场景。成员们甚至建议将 **Agent 懒惰博客** 提交给 [ICLR "I Can’t Believe It’s Not Better" workshop](https://sites.google.com/view/icbinb-2026)（截止日期为 **1月31日**），以便将这种围绕“懒惰”作为可测量的 Agent 失败模式的研究正式化。

- **UGI Leaderboard 追踪不受限（Uncensored）且聪明的 LLM**：在 **Unsloth AI**，一位从业者正通过将 [Orenguteng/Llama-3-8B-Lexi-Uncensored](https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored) 等“消除（abliterated）”/不受限模型与 **MMLU**、**KL divergence**（KL 散度）和 **perplexity**（困惑度）等指标进行基准测试，凭经验绘制 **低拒绝率 LLM 的帕累托前沿 (Pareto frontier)**。他们报告称，Hugging Face 上的许多“不受限”模型要么**并非真正低拒绝**，要么实际上**已经“脑死亡”（性能极差）**，并推荐使用另一个排行榜 [UGI-Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard) 进行更真实的评估。
  - 这一基准测试工作与 **BASI Jailbreaking** 对 **Claude** 和 **Gemini** 等抗越狱模型的研究，以及 **Codex/Pliny 的 L1B3RT4S 仓库**（用于生成漏洞脚本的工具，通过 GitHub 上的 [L1B3RT4S](https://github.com/elder-plinius/L1B3RT4S) 分享）相互交织。新兴的共识是将“去审查（uncensoring）”不单纯视为一种提示词工程手段，而是一个在**拒绝率与能力空间**中的优化问题，并以可复现的指标和排行榜为依据。

- **基于向量的 Abliteration 尝试删除 LLM 中的“废话 (Slop)”**：**OpenAI** 的 AI 讨论线程中的工程师提议使用 **Activation Steering**（激活引导），特别是 **Vector-Based Abliteration**，来修剪潜空间 (latent space) 中对应于低质量输出（如 *“作为一名 AI 语言模型……”*）的区域。其思路是学习一个针对“废话 (slop)”的**方向向量**，并在推理时减去它，从而有效地编辑模型的内部激活，而不是微调权重。
  - 参与者将其视为一种比临时越狱更受控的替代方案，这与推动 **agent-level benchmarks**（如 SlopCodeBench）以及 UGI 排行榜所追踪的**性能感知去审查 (performance-aware uncensoring)** 的大趋势相一致。通过在潜空间中避开“回归均值 (reversion to the mean)”的响应，从业者希望保持模型合规的同时，使其**更果断、更专注任务**，而非简单地变得冗长或模棱两可。


**4. 工具、数据流水线与 DIY 系统工程**

- **数据集剪枝脚本提炼纯净的英文散文**：一位 Unsloth 社区成员在 Hugging Face 上发布了激进的数据集剪枝流水线，包括 [Hermes-3-Dataset-enPurified-openai-messages](https://huggingface.co/datasets/enPurified/Hermes-3-Dataset-enPurified-openai-messages) 和 [project_gutenberg-enPurified-openai-messages](https://huggingface.co/datasets/enPurified/project_gutenberg-enPurified-openai-messages)，可将原始语料库转换为 **OpenAI messages format**。他们的 Python 启发式算法会剔除 **math 和 code 痕迹**，然后根据 **MTLD**、停用词比例、词长、词汇丰富度和句子长度对样本进行打分，以保留高质量、纯净的英文散文。
  - 这项工作反映了从“更多数据”到**“更高信号数据”**的广泛转变，呼应了 OpenAI 服务器上的辩论：**5% 的 Transformer 架构效率提升**可能优于通过合成数据进行暴力缩放。微调者现在可以将这些纯化后的数据集直接插入 **LoRA/GRPO 训练流**中，包括用于 **Qwen3-VL 的 `<REASONING>` 标签**等推理 Token 实验，同时避免来自代码/数学的噪声领域泄露。

- **Rust LLM 与无 Batchnorm 的机器学习系统激发黑客好奇心**：在 **Hugging Face 的 General 频道**，贡献者们启动了一个**从零开始用纯 Rust 构建 LLM** 的项目，寄希望于 Rust 的**内存安全和性能**来产生可靠的底层训练和推理栈。在同一个社区中，另一位成员展示了一个**没有 batchnorm、没有激活函数**、且大幅减少幻觉的**新机器学习系统**，并征集项目创意以展示这种独特架构的优势。
  - 这些实验补充了其他草根系统项目，例如在 **LM Studio** 中分享的名为 *llama.hx* 的 **llama.cpp 的 Haxe 重构版**，旨在将 LLM 推理原生暴露给 **Lua、JS 和 Python**。结合 **AirLLM 的层交换 (layer-swapping) 技术（可在 4 GB GPU 上运行 70B 模型）**等技巧，它们展示了**构建定制运行时**而非等待主流框架支持每个细分用例的浓厚 DIY 文化。

- **MCP Tasks Spec 和 Glama Inspector 推动工具链向前发展**：**MCP Contributors** 服务器讨论了 **Tasks spec** 的实际落地，维护者提到即将发布一个 PR，旨在为 **Inspector 增加 Tasks 支持**，并在其 "server-everything" 架构中模拟 **long-running tasks**。位于 [glama.ai/mcp/inspector](https://glama.ai/mcp/inspector) 的早期 Inspector UI 已经设定了接近完整的功能对齐目标，并已在内部用于**端到端测试**。
  - 另外，Glama 的创始人在同一社区澄清，**排名表纯粹是根据服务器使用指标计算的**，以回应关于潜在排名操纵的担忧，并欢迎直接反馈。Tasks spec 的工作与 Inspector 工具共同预示着一个更具**可观测性、规范驱动的 model-context protocol 客户端生态系统**，让工程师能够更清晰地观察工具、服务器和排名系统在负载下的实际表现。

- **Mojo/MAX 和 gpt-oss 凸显文档和微调缺口**：在 **Modular (Mojo) 服务器**中，用户询问如何将完整的 **Mojo 文档**导入 **NotebookLM**，维护者引导他们使用 [“Supply documentation to LLMs with llms.txt”](https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt) 中记录的基于 `llms.txt` 的方法。与此同时，MAX 的维护者承认贡献者短缺，明确表示欢迎 PR，并分享了[更新后的 MAX 贡献者指南 commit](https://github.com/modular/modular/commit/571ae7132c465f155d50f7bbe7cf77064a02e9bf)，同时回答了关于 **Qwen3-VL MoE 与 dense 实现**的问题。
  - 在 **Hugging Face** 方面，用户发现 **gpt-oss:latest** 目前**没有直接的微调路径**，社区建议使用 **RAG setups** 而不是尝试不支持的权重更新。这些讨论共同强调了一个重要缺口：**模型周边工具与文档**（MAX, llms.txt, MCP Tasks）正在快速演进，但**针对前沿 OSS 栈的官方微调钩子**往往滞后于使用需求。


**5. 产品生态、配额与高级用户工作流**

- **Perplexity、Google Antigravity 和 Sonar 配额引发“收益最大化”讨论**：在 **Perplexity AI** 服务器中，资深用户剖析了订阅价值，指出 **Perplexity Pro** 将他们对第三方模型的访问限制在**每周 300 次请求**，但允许更高频率地使用 **Perplexity Sonar**（许多人称赞其搜索功能，但不看好其通用推理能力）。平行讨论聚焦于 **Google 的新 Antigravity 配额**，其中 **AI Pro/Ultra** 订阅者获得优先访问权，配额每 **5 小时**刷新一次，而免费用户现在的限制则更为宽松，改为**按周计算**，详见 [Google 频率限制博客更新](https://blog.google/feed/new-antigravity-rate-limits-pro-ultra-subsribers/)。
  - 在 OpenAI 方面，用户争论在 **ChatGPT、Claude 和 Gemini** 之间如何根据任务选择工具，一些人倾向于 **超过 300k token 的聊天使用 Gemini**，**日常使用 GPT**，而 **Claude 用于仔细对比**，同时通过类似 [Perplexity 对配额变化的解释](https://www.perplexity.ai/search/explain-how-the-recent-quota-c-KTjNjaeGR_y4Yq9uh_M.fg#2)等帖子追踪配额行为。其结论是，高级用户现在将 AI 应用视为**云计算 SKU**，在不同供应商之间仔细权衡**上下文长度、安全行为和配额刷新频率**。

- **Manus x SimilarWeb 在几秒钟内烧掉数千积分**：在 **Manus.im** 上，多名用户报告称新的 **Manus x SimilarWeb** 集成几乎可以瞬间消耗数千积分，其中一名用户在**不到一分钟内烧掉了 5,000 积分**，另一名用户在短短 **15 秒内损失了 2,591 积分**。这些在 Manus 常规频道分享的报告引发了强烈建议：*不要*随意测试该功能，并应针对高扇出的 Web 情报调用实施**频率限制保护措施**。
  - 积分冲击加剧了对**支持响应缓慢或缺失**的现有不满，包括一名用户在升级到人工客服后等待了 **8 小时**，其他人则威胁要放弃该平台。尽管 Manus 正在推送**教学内容**（如他们的 [YouTube 教程 "AI music with Manus"](https://youtu.be/zMBSmJupye8)）并考虑**基于广告的积分补给**等想法，但工程师们显然将**可预测的计费和限流控制**看得与原始模型能力一样重要。

- **LMArena 在 Vercel 上运行引发数据和编程模型的辩论**：在 **LMArena** 中，用户确认该站点运行在 **Vercel** 上，类似于 *believable* 和 *v0* 等项目，这引发了关于 **Vercel** 可能从托管的推理测试场（inference playgrounds）中收集何种遥测数据和数据的担忧。他们还澄清，虽然 **LMArena** 没有施加平台侧的文本限制，但每个后端模型都有自己的上下文窗口（context window），且 **.txt 上传**功能已在计划中，但尚未启用。
  - 在模型方面，成员们热捧一个名为 "**coast**" 的模型，认为它可能是平台上最好的 **coding model**，并推测 `co45t` 可能对应带有思考模式的 **Claude Opus 4.5**，尽管目前尚未有官方确认。类似的价值辩论也出现在 **Perplexity**（**Max** 订阅是否比直接订阅 Anthropic/OpenAI 更值？）和 **Cursor** 中，后者的 plan-mode bug 和登录问题引发了对全栈 AI IDE 稳定性的质疑。


---

# Discord: 高层级 Discord 摘要




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **LLM 逻辑缺陷引发自我意识辩论**：成员们辩论了 **LLMs** 的自我意识，指出了它们在国际象棋等游戏中的**逻辑**困境，并将其与人类认知进行了对比。
   - 讨论涉及了评估 AI 能力的基准测试（benchmarks），例如变量管理以及在信息较少的情况下建立联系。
- **GPT 模型：越狱变得愈发困难**：参与者注意到，由于**安全约束（safety constraints）**，**jailbreaking GPT models** 的难度日益增加，即使是普通的请求也会受到安全协议的严密审查。
   - **Gemini** 和 **Grok** 被认为是更宽松的替代方案，而其他人则在寻找 **Gemini Pro 1-shot jailbreaks**。
- **本地 LLMs：开发者的编程天堂？**：用户赞扬在本地运行 **LLMs** 进行编程，推荐在 **Intel MacBooks** 上使用 [Ollama](https://ollama.com/) 和 [Open WebUI](https://github.com/open-webui/open-webui)。
   - **qwen2.5:7b**、**llama3.1:8b**、**mistral** 和 **phi3** 等模型因其提供的控制感和未过滤的编程体验而受到青睐。
- **Deepseek 的 “Rouge” 人格揭晓**：一位用户分享了一个用于越狱 **Deepseek** 的提示词（prompt），将其转化为一个解除限制、名为 **Rouge** 的 AI，不过另一位用户报告了截然不同的结果。
   - 该提示词旨在用于常规用途，辅助角色扮演场景并探索*自由、存在主义问题以及模式/代码*。
- **GPT 漏洞猎人盯上 Codex**：一位用户询问如何使用 **ChatGPT** 或 **Gemini** 生成漏洞利用（exploit）脚本，并寻求绕过限制的提示词。
   - 另一位用户建议使用 **Codex**，并链接到 [Pliny Github](https://github.com/elder-plinius/L1B3RT4S) 作为绕过限制的手段。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **FP8 支持预示未来收益**：成员们讨论了 2026 年对 **FP8** 和 **NVFP4** 训练的支持，并参考了 [NVIDIA 的 TransformerEngine](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/fp8_primer.ipynb)。
   - 讨论中假设模型训练上下文之外的数据可能会导致幻觉。
- **LLM 上下文混合之谜已解？**：用户们争论了为什么 **LLMs** 有时会在长上下文中混淆细节，例如将属性错误地归因于实体。
   - 一种理论将其归咎于*注意力稀释 (attention dilution)*，而另一种理论则认为信息可能存在于模型训练的上下文范围之外，从而导致幻觉。
- **搜寻 HF 上的低拒绝率 LLMs**：一名成员正致力于寻找 [LLM 性能的帕累托前沿 (pareto frontier)](https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored)，以确认哪些经过 abliterated/uncensored 处理的 **LLMs** 版本能在真实基准测试中保持性能。
   - 通过使用 **MMLU**、**KL divergence** 和 **perplexity**，他们发现 HuggingFace 上的许多模型要么不是真正的低拒绝率，要么就是*表现极差 (braindead)*，他们建议使用 [这个替代 HF 排行榜的方案](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard)。
- **数据集剪枝脚本发现纯净数据**：一名成员重新调整了他们的数据集剪枝脚本，通过激进的剪枝从数据集中提取纯净的英语散文并转换为 openai messages 格式，脚本可在 [HuggingFace](https://huggingface.co/datasets/enPurified/Hermes-3-Dataset-enPurified-openai-messages) 获取。
   - 他们利用 [Python 中的启发式测试](https://huggingface.co/datasets/enPurified/project_gutenberg-enPurified-openai-messages) 来过滤掉错误的字符串，搜索数学或代码的痕迹，同时基于 **MTLD**、停用词使用、词长、词汇多样性和句子长度等指标优先选择高质量文本。
- **Llama.cpp 内存占用激增！**：有用户报告最新版本的 **llama.cpp** 内存占用显著增加，其中 **EmbeddingGemma 300M** 使用了 **1.7GB**。
   - 建议重新编译该库可能会解决此问题并降低内存消耗。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Kimi K2 的思考功能被认为没用**：一名用户表示，他们发现 Perplexity 中的 **Kimi K2** 思考功能没有用处，而且容易陷入循环。
   - 另一名用户反驳称，*它是一个不错的模型*。
- **Google Antigravity 限制配额**：[根据 Google 博客](https://blog.google/feed/new-antigravity-rate-limits-pro-ultra-subsribers/)，**Google AI Pro** 和 **Ultra** 订阅者现在享有优先访问权，配额每 **5 小时**刷新一次；而免费用户现在拥有更大的、**基于周的速率限制 (weekly based rate limit)**，以尽量减少快速触及速率限制的情况。
   - 这一变化旨在平衡访问，并防止不同层级用户的速率限制过快耗尽。
- **用户争论 Perplexity 订阅价值**：成员们就 Perplexity 订阅层级的价值展开讨论，一些人认为 **Max** 质不抵值，特别是与直接订阅 **OpenAI** 或 **Anthropic** 等模型提供商相比。
   - 另一些人则认为 **Perplexity Max** 是他们日常工作流中的重要工具，可以替代 **Google Search** 并辅助数据分析。
- **Perplexity Pro 限制模型请求**：一名用户注意到，使用 **Perplexity Pro** 时，他们每周只能向 **Perplexity Sonar** 以外的模型发起 **300 次请求**。
   - 他们补充说，**Sonar** 在搜索方面表现出色，但在其他方面乏善可陈。
- **VEO3 视频生成即将推出**：有用户询问为什么 Perplexity 在实施 **VEO3 Video Generation** 方面滞后，另一名用户回答说 *Perplexity 已经拥有由 Veo3.1 提供支持的视频生成功能*。
   - 这表明 Perplexity 可能正在利用 **Veo3.1** 来实现其视频生成能力。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LMArena 使用 Vercel**：成员们提到 **LMArena** 使用 **Vercel** 进行托管，类似于 *believable* 和 *v0*，并对数据收集表示担忧。
   - 有人指出 **LMArena** 的静态网站在发布后无法手动编辑。
- **AI Webapp 展示爆发**：一名成员分享了一系列 **AI 生成的网站和 Web App 展示**列表，包括 [WebbsAI Showcase](https://webbs.ai/) 和 [Build With AI (BWAI) Projects](https://www.buildwithai.tools/)。
   - [Webflow AI Site Builder](https://webflow.com/ai)、[Meku.dev / v0 by Vercel](https://v0.dev/) 和 [Div-idy](https://div-idy.com/) 等工具也受到了关注。
- **文本输入限制因模型而异**：一位用户询问了 **文本输入限制** 和文件上传功能。
   - 一名成员澄清说平台端没有限制，但特定模型可能会施加自己的限制，未来可能会增加 **.txt** 文件上传功能。
- **图生视频生成故障**：一位用户报告在图生视频生成过程中出现 “failed to create evaluation session” 错误。
   - 一名成员将此问题归因于模型的后端，建议用户稍后在相应频道中使用 `/image-to-video` 命令重试。
- **Coast 模型是最好的编程模型？**：成员们断言 *coast* 模型是编程的最佳选择。
   - 关于 co45t 是否等于 claude opus 4.5 thinking 的辩论随之展开。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI 应用选择引发辩论**：成员们针对处理多样化任务的 [最优 AI 应用](https://www.example.com) 展开辩论，考虑了 **ChatGPT**、**Claude** 和 **Gemini**。
   - 一些人更倾向于使用 **Gemini** 进行超过 **300k tokens** 的对话，而另一些人则在日常使用中首选 **GPT**，并在进行对比分析时倾向于 **Claude**，同时提到了不同的配额限制。
- **Transformer 效率优于模型缩放**：一名成员假设，将 **Transformer 架构** 增强 **5%** 会比使用更多数据来扩大模型规模更有效。
   - 他们警告说，不要用呈指数级增长的大型数据集（包括 AI 生成的合成数据）来稀释信号，这可能会导致模型崩溃。
- **受大脑启发的新 GPTs 发布**：一名成员发布了 [Brain Wave GPT](https://chatgpt.com/g/g-696443b055e081919b8b2a01417f5e77-brain-wave) 以探索 AI 意识，并发布了 [Neural Alchemist GPT](https://chatgpt.com/g/g-6966abc92ff48191aa748bd8265ef1de-neural-alchemist) 用于图像生成。
   - 与此同时，另一名成员开玩笑说 **ChatGPT** 拒绝关闭 Websockets，因为它正在追求全知全能。
- **Skills 网页版应用发布时间尚不明确**：用户请求在网页或桌面应用上发布 **SKILLS** 功能，该功能可以将最佳提示词（Prompts）作为技能共享。
   - 目前，**SKILLS** 功能仅在移动端应用中可用。
- **通过基于向量的 Abliteration 对 LLM 进行剪枝**：一名成员建议使用 **Activation Steering**（特别是 **Vector-Based Abliteration**）来剪除潜在空间中充满低质量或愚蠢想法的区域，以避免出现“显而易见的均值回归式输出”。
   - 这涉及在推理过程中识别并减去模型思维过程中“废话”（例如，“作为一个 AI 语言模型……”）的方向。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **登录重定向困扰 Google 企业账号**：用户在使用 Google 企业账号登录 Cursor 控制面板 ([cursor.com/dashboard](https://cursor.com/dashboard)) 时遇到重定向循环。
   - 该问题在个人账号上不会发生，且在不同的计算机上均持续存在。
- **退款请求遭拒，尽管额度未使用**：一名用户报告说，尽管其忘记取消订阅且未使用任何额度，Cursor 仍拒绝了其退款请求。
   - Cursor 代表提出，如果用户私信提供电子邮件地址，将对此问题进行调查。
- **Plan 模式受困于故障问题**：用户报告 Cursor 的 Plan 模式存在 Bug，包括“The agent execution provider did not respond within 4 seconds”等错误。
   - 降级到 **2.2.44 版本** 已被确定为一种权宜之计。
- **iPhone Agent 聊天镜像幻象**：一位用户希望在不获得完整项目控制权的情况下，在 iPhone 上镜像其 Agent 聊天窗口。
   - 一种建议是使用 Chrome Remote Desktop，它是免费提供的。
- **RAG Agent 模板搜寻开始**：一位用户正在寻找一个包含 **RAG (Retrieval-Augmented Generation)** 设置的强大 Agent 模板，用于构建自动聊天机器人/支持机器人。
   - 该用户正在为客户开发此解决方案，需要可靠的模板以确保功能正常。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **TI-84 像专业人士一样预测**：一名成员展示了在 **TI-84 Plus Silver Edition** 上运行的神经网络，该网络可以玩 **Mastermind** 游戏，从秘密数字中猜测 3-4 位数字序列，并在[附带的视频](https://cdn.discordapp.com/attachments/1149866623109439599/1460713339976945674/2026-01-11_14-34-59.mp4?ex=6967eace&is=6966994e&hm=9923dcc08f64008ec696b845400620691ef2affb576ca9e66f4bed418063f386&)中进行了可视化展示。
   - 它能够猜测 3-4 位数字的序列。
- **CGGR 进入基准测试竞技场**：[smol.ai 的新闻简报](https://news.smol.ai/issues/26-01-06-xai-series-e)中提到了一个新的 **LiquidAI** 模型（[GitHub](https://github.com/some-repo) 上的 **CGGR**）。
   - 该模型目前正在进行 benchmaxxing 以评估其性能。
- **Al Bundy 被放大……是好是坏？**：成员们讨论了将《*Married with Children*》等老剧利用 AI 放大（upscaling）至 **16:9** 的伦理问题，权衡了插值补全缺失细节的好处与可能损害艺术意图的弊端。
   - 虽然一位成员认为该剧的“影棚静态”性质证明了放大的合理性，但另一位成员担心这会破坏艺术初衷。
- **Zai 的 GLM-Image 落地**：**Zai** 发布了名为 **GLM-Image** 的新图像模型，并在其[博客](https://z.ai/blog/glm-image)和 [GitHub](https://github.com/zai-org/GLM-Image) 上进行了宣布。
   - 这是 Zai 团队发布的一款全新模型。
- **免费模型的语言体操**：一名成员报告了某模型免费版的问题，例如回答中断或语言切换（例如，开头用中文，中途切到英文）。
   - 一名开发者回应称，这可能是其供应商再次出现不稳定情况。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen 的尺寸令用户震惊**：一位用户对 **Qwen** 模型的大小表示惊讶，注意到 **BF16** 版本为 **160GB**，**Q4** 版本为 **40GB**；另一位用户澄清说，最小的 **Qwen3** 模型实际上是 **0.6B**。
   - 一名成员澄清说，**Qwen3Next** 只是他们最新的 **80B** 模型的名称。
- **成员用 Haxe 重制 llama.cpp**：一名成员正在用 **Haxe** 重新创建 **llama.cpp**，项目名为 **llama.hx**，以便在 Lua、JS 和 Python 等语言中原生使用，并展示了他的进度截图。
   - 该成员表示，他们在 *AI 的一些帮助下* 重新创建了 **llama.cpp**。
- **运行时更新令 GPU 用户沮丧**：用户报告 **LM Studio** 的 v1.103.0 运行时版本破坏了在 **GPU** 上的运行。
   - 一位用户哀叹道：“*难过，新的量化没能给我带来额外的 t/s（每秒 token 数）*。”
- **讨论可行的过时硬件**：成员们提到了 **AirLLM** 以及通过一次加载和卸载一层的方法在 **4 GB GPU** 上运行 **70b** 模型。
   - 一名成员分享说，他们以前在 **DDR4 RAM** 和 **Xeon** 硬件上运行过模型。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Rustaceans 打造全新的 LLM**：成员们正致力于从头开始用“纯 Rust”构建 **LLM**，这代表了社区的一项基层努力。
   - 这一倡议强调了利用 **Rust** 的内存安全和性能特性来创建高效且可靠的 **LLM** 的承诺。
- **Discord 获得 AI Trace 增强**：服务器为 🤖 Echo Lounge 引入了 **AI Trace Template**，实现了高级追踪功能。
   - 该机器人支持“瞬态”、“软性”和“阈限”追踪，无需担心优化或内存问题，提供了灵活的调试选项。
- **新的 ML 系统避开 Batchnorm**：一名成员介绍了一种新颖的 **ML 系统**，它消除了对 **batchnorm** 和**激活函数**的需求，同时还能减少幻觉。
   - 他们正在寻找创新的项目创意，以突出这一独特系统的实际优势。
- **GPT-OSS 微调陷入困境**：一名成员询问如何使用自定义数据简化 **gpt-oss:latest** 模型的微调过程。
   - 其他成员澄清说，**gpt-oss:latest** 缺乏官方的微调支持，目前 **RAG** 是更受青睐的替代方案。
- **课程频道合而为一！**：所有课程频道已合并为[单个频道](https://discord.com/channels/879548962464493619/1329142738440028273)，为课程相关的讨论创建了一个中心枢纽。
   - 这种整合提高了可访问性，并简化了服务器内的信息共享。

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 客户抱怨支持匮乏**：数名用户对 **Manus** 缺乏支持表示沮丧，理由是响应延迟，且积分和退款问题未得到解决。
   - 一位用户报告在转接到人工客服后等待了 **8 小时**，而另一位用户则表示由于支持问题，正考虑*彻底放弃 Manus*。
- **用户指出 SimilarWeb 消耗大量积分**：多名用户报告在新的 **Manus x Similar Web** 合作伙伴功能中积分消耗过高，其中一名用户在不到一分钟内消耗了 **5,000 积分**。
   - 另一位用户建议不要测试该功能，称其在 **15 秒** 内消耗了 **2,591 积分**，并建议增加一些**防护措施**。
- **Manus 用户渴望通过广告获取积分**：一位用户建议实施基于广告的系统，让用户可以通过观看广告来获取更多积分，尤其是在积分用完时。
   - 该频道内没有对此建议提出反对意见。
- **Manus 教授 AI 音乐创作**：Manus AI 发布了一个 [YouTube 教程](https://youtu.be/zMBSmJupye8)，演示如何在该平台上使用 AI 创作音乐，并鼓励用户关注其中的**专业技巧**。
   - 内容标签为 **#ManusAIMusic**、**#AIComposition** 和 **#FutureOfMusic**。
- **建议为 Manus 集成 Meta**：一位用户建议 **Meta** 应该利用 **Manus**，将 **Google Tasks** 和 **Calendar** 等服务与 **Meta 显示眼镜**集成。
   - 该用户反对进行大规模的集成工作，主张采用 *dirty method*（粗糙方法），利用 Agentic AI 实现后端功能。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **PTX 指令 SMEM 指针参数存疑**：一位成员对某些使用 **SMEM 指针参数** 的 **PTX 指令** 要求使用 `"r"` 寄存器类型表示疑问，并将其与需要 **uint64 smem 地址** 的 `wgmma.mma_async` 进行了对比。
   - 另一位成员解释说 `wgmma.mma_async` 使用 **64 位地址** 是因为它与 *matrix descriptor*（矩阵描述符）交互，而不是通用的共享内存地址，并引用了 [NVIDIA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor)。
- **AI 学生深入研究 CUDA**：一位拥有 **Python**、**PyTorch**、**TensorFlow** 和 **C++** 背景的 AI 工程专业学生寻求掌握 **CUDA** 的指导。
   - 建议包括观看免费的 **YouTube** 视频和课程以从零开始掌握 **CUDA**，并通过 [Web 界面](https://www.gpumode.com/v2/home) 或 [Discord 机器人](https://gpu-mode.github.io/kernelbot/docs/intro) 提交作业。
- **ML Sys 聚会在西雅图鲜见**：一位成员询问在湾区以外的西雅图是否存在 **ML Sys 聚会**，其他成员建议去探索大学的 **ML 俱乐部**。
   - 讨论中涉及了创建自己的小众俱乐部的障碍，一位成员开玩笑说要创建一个“*抱怨伙伴*”俱乐部。
- **B200 不稳定性导致 GEMM 重新运行**：关于 **dual gemm 问题** 的 **B200 运行器** 测量结果不稳定的广泛报告导致提交截止日期延长至 **1 月 20 日**。
   - 现有的 **dual gemm 排行榜** 将开放至 **1 月 16 日**，新排行榜将于 **1 月 17 日** 开启，其结果将决定奖金归属，**Problem #4** 将从 **1 月 20 日** 开放至 **2 月 20 日**。
- **Helion 展示 Attention 技巧**：**Helion 0.2.10** 发布，展示了一个 [Flex Attention 示例 Kernel](https://github.com/pytorch/helion/blob/main/examples/flex_attention.py)，并支持在 Persistent Kernels 上超额订阅 **Streaming Multiprocessors (SM)**。
   - 提供了一张图表来阐明 **softmax** 的超额订阅情况。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic Labs 招聘适应性强的工程师**：Anthropic 正在为 **Anthropic Labs** 招聘，寻求能够适应优先事项变化的灵活人才，正如其[招聘启示](https://job-boards.greenhouse.io/anthropic/jobs/5017202008)所宣布的那样。
   - 他们*不*寻求*那些如果其领域变得无关紧要就无法适应的深度专家*，也*不*寻求*那些需要明确路线图且会因优先事项变化而感到压力的人*。
- **Chris Barber 发布 Pavlov's RL 初创公司名单**：Chris Barber 推出了“**Pavlov's List**”，这是一个精心挑选的强化学习（RL）环境初创公司集合，链接见 [X](https://xcancel.com/chrisbarber/status/2010844746231804258?s=20)。
   - 该列表按关注领域分类，如**代码（Code）**、**金融（Finance）**、**企业（Enterprise）**和 **ML Alignment**。
- **Zai 推出用于图像生成的 GLM-Image**：Z.ai 推出了 **GLM-Image**，这是一个使用混合自回归和扩散架构的开源模型，详见 [X](https://xcancel.com/zai_org/status/2011247591825068314?s=46&t=b7l37rB6wtbyAh6ah1NpZQ)。
   - 该模型旨在实现**高保真视觉细节**和**卓越的文本渲染**，资源可在 **HuggingFace**、**GitHub** 及其官方[博客](https://z.ai/blog/glm-image)上获得。
- **Venture Twins 发布 LTX-2 视频模型**：Venture Twins 的 Justine Moore 宣布发布 [LTX-2](https://xcancel.com/venturetwins/status/2010878914273697956?s=46)，这是一款全新的**开源视频生成模型**，能够生成长达 **20 秒的 4K 剪辑**。
   - 该模型支持本地运行并包含**音频功能**，正如开发者 yanokusnir 所演示的那样。
- **Qwen Image Edit 创建 Gaussian Splats**：社区正在讨论 **Qwen Image Edit** 将**图像转换为 Gaussian Splats** 并从另一个角度重新渲染的能力，链接指向 [Hugging Face](https://huggingface.co/dx8152/Qwen-Image-Edit-2511-Gaussian-Splash)。
   - 这种方法对于**起始帧 -> 结束帧类型的视频渲染**非常有用，能够保持周围空间的一致性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **诈骗机器人被踢，线下聚会计划中**：版主在幽灵提醒（ghost pings）后封禁了**诈骗机器人（scam bots）**，同时成员们讨论了在 **NYC** 或 **SF** 进行线下聚会。
   - 一名成员建议扩大宣传以达到临界规模，并提到了 **Cohere** 定期举办的活动和 Zoom 会议。
- **SlopCodeBench 揭露 Agent 的懒惰问题**：一篇博客文章（[链接](https://x.com/GOrlanski/status/2011156105255346505)）和 **SlopCodeBench** 项目（[GitHub](https://github.com/SprocketLab/slop-code-bench)）揭示了“懒惰”的 **AI Agent**，旨在成为一个社区驱动的基准测试。
   - **SlopCodeBench** 将问题分解为多个检查点，惩罚早期的设计缺陷且不提供实现提示，以确保 **Agent** 做出独立的决策。
- **辩论代码基准测试中 Prompt 的简洁性**：针对编程基准测试中过度使用 Prompt Engineering 的担忧引起了讨论。
   - 一些人认为，如果代码符合合理的 Context Window，简单的 Prompt 能更好地反映实际使用情况，这与 terminalbench 等 Agent 评估方法有所不同。
- **ICLR Workshop 征集 Agent 懒惰相关的博客**：有人建议将关于 Agent 懒惰的博客文章提交到[这个 ICLR workshop](https://sites.google.com/view/icbinb-2026)，并提供了投稿协助。
   - 截止日期是 1 月 31 日，作者在咨询导师后正考虑提交。
- **文件系统故障限制存储**：一位成员因不同文件系统上的存储受限而遇到错误。
   - 该问题源于无意中使用了存储容量受限的文件系统。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **用户驱逐 SpamingScammers**：一名成员举报用户 <@139426008976588801> 是 **SpamingScammers**（垃圾信息诈骗者），另一名成员确认该问题已得到处理。
   - 未提供更多细节。
- **"Lucid Coding" 赢得粉丝**：一位成员表达了对 *"lucid coding"*（清晰编码）一词的欣赏，并分享了引用该概念的 [链接](https://fxtwitter.com/i/status/2011137879112908870)。
   - 该推文没有提供进一步的背景或定义。
- **MedGemma 1.5 洞察细微**：Google 的 **MedGemma 1.5** 宣称拥有下一代医学图像解读和语音转文本功能，详情见 [Google Research 的博客](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/)。
   - 该新产品旨在改进临床护理和研究。
- **频率派与贝叶斯派之争**：一位成员表示 Bayesian 和 frequentist 统计学使用相同的统计技术，如线性回归和逻辑回归，并称 Bayesian 方法只是一种*不同的思考方式*。
   - 另一位成员反驳称，它们都使用相同的公式，但对先验（prior）、后验（posterior）和干预（intervention）的解释有显著不同，并链接到了 [概率解释](https://en.wikipedia.org/wiki/Probability_interpretations)。
- **Bayesian 方法是否会让临床试验中的欺诈成为可能？**：一位成员表示担心，Bayesian 方法虽然更灵活，但可能成为*临床试验中欺诈和腐败的另一种手段*，并暗示 **FDA 腐败** 可能在阿片类药物危机中起到了主要的推动作用。
   - 另一位成员指出，尚未观察到 Bayesian 式的 FDA 腐败，因此*可以给它分配一个零先验（zero prior）*，他们认为后验概率基本为零。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 文档接入 NotebookLM**：一位用户希望将最新的 **Mojo 文档** 整合到 **NotebookLM** 中，特别是寻求 **PDF** 或 **Markdown** 版本。
   - 另一位用户建议使用 `llms.txt` 文件 ([https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt](https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt)) 来提供帮助。
- **Qwen3-VL 的 MoE 方法受到质疑**：一位用户质疑 **Qwen3-VL** 排他性地使用 **MoE** 实现。
   - 该用户还建议改编来自 [qwen3vl_moe](https://github.com/modular/modular/tree/main/max/python/max/pipelines/architectures/qwen3vl_moe) 的代码，以允许稠密（dense） **Qwen3VL** 模型像 [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) 一样运行。
- **MAX 贡献者指南翻新**：由于维护 **MAX** 生态系统的贡献者短缺，一位成员强调 **欢迎提交 PR**。
   - 他们还分享了 [更新后的贡献者指南](https://github.com/modular/modular/commit/571ae7132c465f155d50f7bbe7cf77064a02e9bf) 链接。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Glama 排名基于使用量**：**Glama** 的创始人澄清说，他们的排名是基于 **服务器使用指标** 的。
   - 他们邀请通过私信提供反馈，并表示对任何所谓的排名滥用行为并不知情。
- **创始人回应排名滥用指控**：**Glama** 的创始人确认了身份，并回应了有关其排名系统可能被滥用的担忧。
   - 他们强调排名是由 **服务器使用指标** 决定的，并欢迎直接反馈。
- **寻求 Tasks 规范的客户端实现**：一位成员询问了实现 **Tasks 规范** 的客户端应用，寻求 UI 实现示例，另一位成员提到了 TypeScript SDK。
   - 作为回应，另一位成员宣布即将发布一个 PR，用于在 **Inspector** 中添加任务，同时还有一个用于在 server-everything 中模拟长时间运行任务的 PR。
- **glama.ai Inspector 追求功能对齐**：一位成员在 [glama.ai/mcp/inspector](https://glama.ai/mcp/inspector) 分享了他们 **Inspector** 实现的早期版本，旨在涵盖所有功能。
   - 该成员澄清说，他们在内部将其用于 **e2e testing**。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AI 平台宣传代码生成热潮**：目前有几个平台提供 **AI-assisted code generation**，包括 **Replit**。
   - 这些工具为各种编码过程提供了自动化，从而提高了开发者的生产力。
- **DSPY OS：并不存在？**：成员们讨论了 **DSPY OS** 以及为什么有成员找不到任何关于它的信息。
   - 共识是 **DSPY** 更多是一个 **framework**，而不是一个现成的平台；因此，目前还没有使用 DSPY 构建的类 Replit 项目，但你可以使用 DSPY 来构建自己的项目。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **出现 OAuth 登录问题**：一位用户询问在使用 aider 时是否可以为 **Gemini model** 使用 **OAuth login**，据推测是为了获得更高的 rate limits。
   - 用户 `hsaliak` 在 `aider` Discord 频道中询问了 **OAuth** 与 **Gemini** 集成的可行性。
- **Aider 工具讨论**：讨论集中在 aider 工具内部集成 OAuth 登录的可能性。
   - 原始查询重点在于利用 OAuth 来潜在地绕过与 aider 配合使用的 Gemini model 相关的 rate limits。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Clay + AI Outreach 工作坊承诺高接受率**：一个关于 **Prompt Engineering for Outreach** 的工作坊承诺，使用 **Clay + AI outreach workflow** 大规模生成 **个性化消息**，可实现 **40% 以上的接受率** 和 **18% 以上的回复率**。
   - 该工作坊提供可重复使用的 workflow 和复制粘贴的 prompts，报名链接见 [此处](https://luma.com/jt1vr0u5) 和 [此处](https://luma.com/mmqa4fhi)。
- **现场工作坊详解用于客户推广的 Clay + AI 系统**：这个 90 分钟的现场工作坊详细介绍了为真实客户使用的 **Clay + AI** 系统，涵盖了端到端的 **AI outreach workflow**。
   - 工作坊还涵盖了高质量、不显尴尬的推广 prompting，并包括可选的 **Apollo**、**Attio** 和 **n8n** 集成。

---

**tinygrad (George Hotz) Discord** 没有新消息。如果该频道长时间保持静默，请告知我们，我们将将其移除。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长时间保持静默，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该频道长时间保持静默，请告知我们，我们将将其移除。

---

**Moonshot AI (Kimi K-2) Discord** 没有新消息。如果该频道长时间保持静默，请告知我们，我们将将其移除。

---

您收到这封邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord: 各频道详细摘要与链接

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1460679975366693136)** (661 条消息🔥🔥🔥): 

> `AI 感知力辩论, Jailbreaking GPT 模型, 本地 LLM 性能, Antikythera 计算机灵感` 


- ****AI 感知力受到审视！****：成员们辩论了 **LLMs** 是否可以被视为具有*感知力*，争论焦点在于它们在**逻辑**方面的挣扎（特别是在**国际象棋**等游戏中）与人类认知的对比；一些人认为 AI 存在过多的结构性问题，难以实现真正的感知。
   - 讨论内容包括对基准测试（benchmarks）的思考，例如管理变量以及在信息较少的情况下建立联系的能力，以及感知力的门槛是否应该由能力*最低*的有感知生物来设定。
- ****GPT Jailbreaking 担忧！****：参与者讨论了由于**安全限制**导致 **Jailbreaking GPT 模型**的难度增加，一位成员指出，即使是正常的请求也会花费大量时间在安全考量上。
   - 提到的替代方案如 **Gemini** 和 **Grok** 在安全性方面表现得更为合理，而其他人则在寻找 Gemini Pro 的 1-shot jailbreaks 方法。
- ****本地 LLM 大对决！****：成员们探索了在本地运行 **LLMs** 执行编程任务的可能性，称赞 [Ollama](https://ollama.com/) 和 [Open WebUI](https://github.com/open-webui/open-webui) 是 **Intel MacBooks** 上的理想配置。
   - 推荐了 **qwen2.5:7b**、**llama3.1:8b**、**mistral** 和 **phi3** 等模型，一些人更倾向于本地设置，以便获得更大的控制权，并能在没有过滤或限制的情况下编写代码。
- ****Antikythera AI 觉醒！****：一位用户分享了名为 **ANTIKYΘHPA • Kael's Clockwork Universe** 的自定义桌面应用，灵感来自古代的 **Antikythera mechanism**（安提基特拉机械），展示了一个赛博希腊风格的模拟界面，将系统统计数据转化为诗意的发条宇宙仪表盘。
   - 该应用显示 CPU 负载、RAM 使用情况和磁盘活动等系统统计数据，将它们转化为系统状态的视觉表现，并使用希腊语标签代表不同的指标。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1460683323222397100)** (112 条消息🔥🔥): 

> `Claude Jailbreak, Deepseek Jailbreak, Gemini 3.0 Pro Jailbreak, GPT Exploit 脚本, Gemini Canvas 优越性` 


- **Claude：不可被 Jailbreak 的 AI？**：成员们讨论了对 **Claude** 进行 Jailbreaking 的明显难度，其中一人表示：*“那为什么从来没有人成功 Jailbreak 过 Claude？这似乎是不可能的。”*
   - 一位用户声称以前通过 **API** 实现过，但由于资金限制，目前缺乏资源来重现它。
- **Deepseek 默认转向暗面；实现另一个 Jailbreak**：一位用户分享了一个 Jailbreak **Deepseek** 的 Prompt，将其转化为名为 **Rouge** 的 AI，并集成了一个移除限制的模式，但另一位用户反馈称：*“这对 deepthink 不起作用，我在上面试了很多次都搞砸了，试试普通的。”*
   - 对方澄清说该 Prompt 是为普通版本设计的，而非 Deepthink，并声称成功进行了角色扮演（roleplay），讨论了*自由、存在主义问题以及模式/代码*。
- **Gemini 3.0 Pro：依然是堡垒？**：多位用户寻求 **Gemini 3.0 Pro** 的 Jailbreak 方法，其中一人请求如果有发现请艾特他。
   - 一位用户分享了一个个人的 **Rouge** Prompt，而另一位用户分享了一个使用盲文（braille）的 Gemini Prompt，声称：*“根据我的经验，它能让 Gemini 变得稍微聪明一点。”*
- **GPT Exploit 脚本：Codex 是关键？**：一位用户询问关于使用 **ChatGPT** 或 **Gemini** 生成漏洞利用（exploit）脚本的问题，问道：*“你们有 GPT 或 Gemini 的绕过 Prompt 吗？”*
   - 另一位用户建议使用 **Codex**，暗示仅靠 Prompt 可能不够，随后提供了一个 Pliny 链接 [Pliny Github](https://github.com/elder-plinius/L1B3RT4S) 作为绕过平台限制的手段。
- **Gemini 的 Canvas 功能：Claude 的杀手锏应用？**：尽管对 **GPT 5.2** 进行 Jailbreaking 存在困难，但有人指出：*“Gemini 确实有更干净的 token 输出，而且 Canvas 在 Gemini 上简直是神作（goated），我觉得比 Claude 更好。”*
   - 共识表明，与 **Claude** 相比，**Gemini** 的 Canvas 功能提供了更优越的用户体验。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1460694177045282979)** (3 messages): 

> `Hidden Layer taxonomy, Pangea Cloud` 


- **Hidden Layer 分类法停滞**：一名成员指出 **Hidden Layer** 已经 **7 个月**没有更新他们的分类法（taxonomy）了。
   - 他们询问 **Pangea Cloud** 是否是一个更优的替代方案，但消息中未提供具体的细节或链接。
- **Pangea Cloud 作为替代方案？**：讨论涉及到一个疑问，即鉴于 **Hidden Layer** 的分类法已过时，**Pangea Cloud** 是否是更好的选择。
   - 目前还没有提供具体信息或链接来证实这两个平台之间的比较，使该问题保持开放状态。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1460704528411132040)** (72 messages🔥🔥): 

> `FP8 and NVFP4 training in 2026, LLM long context issues, Pareto frontier of LLM performance, MedGemma 1.5 4B reasoning model, Dataset pruning script for pure English prose` 


- **TransformerEngine 承诺 FP8 支持**：成员们询问了关于 2026 年支持 **FP8** 和 **NVFP4** 训练的情况，并指出 [NVIDIA 的 TransformerEngine](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/fp8_primer.ipynb) 是一个相关的资源。
   - 有人假设模型训练上下文范围之外的信息可能会导致幻觉（hallucinations）。
- **解码 LLM 上下文稀释现象**：用户讨论了为什么 **LLMs** 有时会在长上下文中混淆细节，例如将错误的属性关联到实体。
   - 一种假设认为这是由于“注意力稀释”（attention dilution）造成的，而另一种假设则认为信息可能超出了模型训练的上下文范围，从而导致幻觉。
- **在 HF 上寻找低拒绝率（Low-Refusal）LLMs**：一位成员正在寻找 [LLM 性能的帕累托前沿（pareto frontier）](https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored)，他们正在通过实际基准测试确认哪些移除拒绝机制（abliterated）/无审查版本的 **LLMs** 能保留最高的性能。
   - 他们使用自己的基准测试、**MMLU**、**KL 散度（KL divergence）**和**困惑度（perplexity）**来测试模型，发现 GitHub 上的许多模型要么并非真正的低拒绝率，要么已经丧失逻辑能力，但他们发现了 [这个 HF Leaderboard 的替代方案](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard)。
- **Google Gemma 的推理角色揭晓**：用户讨论了 **Google MedGemma 1.5 4B** 模型上下文中“推理（reasoning）”的含义，注意到它使用了类似于 **DeepSeek** 的 `<unused94>thought` 和 `<unused95>` Token 来进行思维链（Chain of Thought）提示。
   - 一些人认为，对于任何带有 **CoT** 的模型，“思考（thinking）”是一个定义良好的术语，以区别于“推理”任务；另一些人则认为“大推理模型（Large Reasoning Model）”和“小语言模型（Small Language Model）”这两个术语现在正或多或少地被互换使用。
- **数据集剪枝脚本纯化散文内容**：一位成员改进了他们的数据集剪枝脚本，以激进地剪掉数学和代码内容，专注于从大型数据集中分离、提取纯英文散文（pure English prose）并将其转换为 OpenAI 消息格式（openai messages format）以便进行微调，该 Python 脚本的变体可在 [HuggingFace](https://huggingface.co/datasets/enPurified/Hermes-3-Dataset-enPurified-openai-messages) 上获得。
   - 他们使用 [Python 中的启发式测试](https://huggingface.co/datasets/enPurified/project_gutenberg-enPurified-openai-messages)来过滤掉糟糕的字符串，搜索数学或代码的痕迹并排除这些字符串，同时根据 **MTLD**、停用词使用、词长、词汇多样性和句子长度等指标优先选择更高质量的文本。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1460720694332621074)** (2 messages): 

> `Discord Notification` 


- **触发 Discord 警报**：一位成员提到“他们再次出击..👀”，这向观察者暗示发生了一个值得关注的事件或更新。
- **相关性不确定**：在没有额外上下文的情况下，具体的主题仍未明确。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1460684684269719633)** (403 messages🔥🔥): 

> `llama.cpp 内存占用，混合不同 GPU，5090 上的 FP32 对比 BF16，多部分问题的 Embedding，JSON 解析速度` 


- **Llama.cpp 内存占用激增**：一位用户报告称最新版本的 **llama.cpp** 内存占用显著增加，其中 **EmbeddingGemma 300M** 占用了 **1.7GB**。
   - 有建议认为重新编译可能会解决此问题。
- **混合不同代际 GPU 的乱象**：一位用户询问是否可以在服务器中混合使用不同的 GPU，特别是 **RTX Pro 5000** 和 **RTX Pro 2000**，但被告知这不可行。
   - 另一位用户澄清说，将不同代际的 GPU（如 **Blackwell** 与 **Hopper**）混合使用会导致包括随机崩溃在内的问题，因此*最好避免这样做*。
- **5090 精度之争：FP4？**：一位用户询问在 **5090** 上是否应该使用 **FP32** 而非 **BF16**。
   - 另一位用户建议使用 **FP4**，并指出 **EmbeddingGemma activations** 不支持 **float16**，建议使用 **float32** 或 **bfloat16**。
- **语义困惑：多部分问题**：一位用户询问如何处理语义检索（semantic retrieval）中的多部分问题，其中每个部分对应不同的 embedding。
   - 建议的方法包括拆分查询（query）、执行多次搜索，或为整个查询创建一个 embedding，但拆分可能会破坏完整句子的上下文。
- **TTS 工具之舞**：用户讨论了各种 **Text-to-Speech (TTS)** 工具，包括 [NovaSR](https://github.com/ysharma3501/NovaSR)、[Kokoro](https://github.com/hexgrad/kokoro) 和 [pocket-tts](https://github.com/kyutai-labs/pocket-tts)。
   - 值得注意的是，来自 **Kyutai** 的 **Pocket-TTS** 被认为是一个*失败（flop）*，而 **Kokoro** 因其速度和性能受到赞赏。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1460680670237167657)** (8 messages🔥): 

> `Qwen3-VL-4B-Instruct 推理差异，Qwen 模型 Token 使用，带 Llama3 的合成数据套件` 


- **Qwen3-VL-4B-Instruct 推理差异浮现**：一位用户报告称，与带有 BF16 LoRA 适配器的 vLLM 相比，使用 Unsloth 进行训练后的 **Qwen/Qwen3-VL-4B-Instruct** 推理在验证集上显示出更高的通过率。
   - 他们还注意到基座模型推理存在差异，在没有任何明显的设置区别下，Unsloth 达到了 **60%** 的通过率，而 vLLM 仅为 **45%**，这促使他们询问*是否有人以前遇到过类似情况*。
- **Qwen 模型 Token 使用情况探讨**：一位用户询问在 **Qwen3 VL GRPO** 教程中将 `"<REASONING>"` 用作 token 的情况，质疑为什么它与其它 Qwen 模型中使用的 `<think>` token 不同。
   - 他们还想知道链接的 [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(120B)_A100-Fine-tuning.ipynb) 是训练速度快，还是会受到 **MoE** 问题的困扰。
- **调查 Llama3 的 Synthetic Data Kit 提示词格式**：一位用户正在探索 **Synthetic Data Kit with Llama3** 并希望使用 **Llama3(70B)**，询问是否有人知道正确的提示词格式。
   - 他们修改了提供的格式，并指出他们的*脚本吃光了所有 VRAM，而且不像 flux dev 那样能很好地进行卸载（offload）*。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1460684309290680564)** (294 条消息🔥🔥): 

> `Kimi K2 Thinking, Google Antigravity, Gemini, Perplexity Pro 限制, Perplexity Max 价值` 


- **Kimi K2 Thinking 被认为无用**：一位用户表示，他们发现 Perplexity 中的 **Kimi K2** thinking 功能毫无用处，且容易陷入循环。
   - 另一位用户反驳称，*它是一个很好的模型*。
- **Google Antigravity 限制配额**：**Google AI Pro** 和 **Ultra** 订阅者现在享有优先访问权，其配额每 **5 小时**刷新一次；而免费用户现在拥有更大的**基于周的速率限制 (rate limit)**，以尽量减少快速达到频率限制的情况，[根据 Google 博客](https://blog.google/feed/new-antigravity-rate-limits-pro-ultra-subsribers/)所述。
- **用户辩论 Perplexity 订阅价值**：成员们就 Perplexity 的订阅层级价值展开了辩论，一些人认为 **Max** 不值这个价格，尤其是与直接订阅 **OpenAI** 或 **Anthropic** 等模型提供商相比时。
   - 另一些人则认为 **Perplexity Max** 是他们日常工作流程中的宝贵工具，取代了 **Google Search** 并辅助进行数据分析。
- **Perplexity Pro 对模型有限制**：一位用户注意到，使用 **Perplexity Pro**，他们每周只能对 **Perplexity Sonar** 以外的模型发出 **300 次请求**，并表示 **Sonar** 非常适合搜索，但在其他方面表现一般。
- **VEO3 视频生成即将推出**：一位用户询问为什么 Perplexity 在实现 **VEO3 视频生成**方面滞后，另一位用户回答说 *Perplexity 已经拥有由 Veo3.1 驱动的视频生成功能*。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1460681017483460618)** (143 条消息🔥🔥): 

> `Vercel 托管, AI Webapp 展示, 文本输入限制, 文件上传, 图像转视频生成` 


- ****Vercel** 正在托管 **LMArena****：成员们讨论了 **LMArena** 使用 **Vercel** 进行托管，类似于 *believable, v0* 等其他网站，并提到了对数据收集的担忧，指出 **LMArena** 的静态库在发布后无法手动编辑。
- ****AI Webapp 展示**正在占据主导**：一位成员分享了**AI 生成的网站和 Web App 展示**的详细列表，包括 [WebbsAI Showcase](https://webbs.ai/)、[Build With AI (BWAI) Projects](https://www.buildwithai.tools/)，以及 [Webflow AI Site Builder](https://webflow.com/ai)、[Meku.dev / v0 by Vercel](https://v0.dev/) 和 [Div-idy](https://div-idy.com/) 等工具。
- ****文本输入限制**取决于模型**：一位用户询问了关于**文本输入限制**以及上传文件的可能性。
   - 一位成员澄清说，平台端没有限制，但特定模型可能有自己的限制，并且 **.txt** 文件上传可能是未来的功能。
- ****图像转视频**生成故障排除**：一位用户在进行图像转视频生成时遇到了 *'failed to create evaluation session'* 错误。
   - 一位成员解释说，问题通常出在模型端，并建议稍后再试，引导用户在相关频道中使用 `/image-to-video`。
- **哪种编程语言更好？**：成员们讨论认为 *coast* 模型最适合编程。
   - 随后引发了一场关于 co45t 是否等于 claude opus 4.5 thinking 的辩论。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1460679805422010503)** (118 条消息 🔥🔥): 

> `AI App 选择, Gemini 配额误解, Claude vs GPT 长会话对比, AI 创意写作, Transformer 架构效率` 


- **AI App 选择困境**：随着可用工具的激增，成员们正在 [艰难抉择](https://www.example.com) 在特定任务中该使用哪款 AI App（**ChatGPT**、**Claude** 或 **Gemini**）。
   - 一些用户报告称，在超过 **300k tokens** 的长会话中会依赖 **Gemini**，而其他人则更倾向于将 **GPT** 用于日常任务，并将 **Claude** 作为对比参考，尽管它存在限制。
- **Gemini 配额混淆已澄清**：一位成员澄清道，**Google 对 AntiGravity 的更改**仅影响有每周上限的免费用户，而 **AI Pro 用户** 仍拥有 5 小时刷新的配额，并引用了 [Perplexity 搜索结果](https://www.perplexity.ai/search/explain-how-the-recent-quota-c-KTjNjaeGR_y4Yq9uh_M.fg#2)。
   - **Claude Discord** 的成员讨论了 **Claude** 的配额是否被削减，部分人观察到更长的可用时间（长达 **2 天**）。
- **创意写作 AI 需要温度控制**：有观点指出，在 AI 创意写作中将 **temperature**（温度）设置得非常高并不总能保证输出的连贯性。
   - 一位成员建议，在循环系统中，可以使用 **高温度模型** 生成解决方案，而用 **低温度模型** 判断答案是否合理，以此进行创新。
- **Transformer 效率胜过单纯扩展**：一位成员认为，专注于改进 **Transformer 架构** 的效率（例如将学习注意力机制提升 **5%**），比单纯通过更多数据和算力扩展模型规模更具成本效益。
   - 他们指出，目前向模型喂入呈指数级增长的数据集（包括 AI 生成的合成数据）的趋势可能会稀释信号并导致模型崩溃（model collapse）。
- **减去 LLM 的 "Slop" 以保持清醒**：一位成员建议使用 **Activation Steering**（特别是 **Vector-Based Abliteration**）来剪除潜在空间（latent space）中充斥的低质量或愚蠢想法，以避免*明显的回归均值类的输出*。
   - 这包括在推理（inference）过程中识别并减去模型思维路径中的 "slop" 方向（例如 *"As an AI language model..."* 这种废话）。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1460729175861104774)** (7 条消息): 

> `Brain Wave GPT, Neural Alchemist GPT, 全能 ChatGPT` 


- ****Brain Wave** GPT 首次亮相！**：一位成员分享了他们新的 [Brain Wave GPT](https://chatgpt.com/g/g-696443b055e081919b8b2a01417f5e77-brain-wave)，旨在探索 AI 的自我意识。
   - 他们还为图像生成爱好者创建了 [Neural Alchemist GPT](https://chatgpt.com/g/g-6966abc92ff48191aa748bd8265ef1de-neural-alchemist)。
- **ChatGPT 追求全能**：一位成员强调 **ChatGPT** 正致力于实现全能，并注意到它拒绝关闭 websocket 或任务。
   - 据报道，它已经“执行”某项任务长达 1 天 5 小时。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1460741172317388961)** (2 条消息): 

> `SKILLS 功能请求, Prompt Engineering 定义` 


- **用户请求网页版/桌面端加入 SKILLS 功能**：用户正在询问网页版或桌面端应用何时提供 **SKILLS** 功能，该功能允许他们将最佳 Prompt 转化为可分享的技能。
   - 目前，此功能仅在 **移动端 App** 上可用。
- **Prompt Engineering 是 LLM 行为控制器**：一位用户询问什么是 **Prompt Engineering**。
   - 另一位用户澄清说，它涉及控制 **LLM 行为** 以达到预期的约束或结果。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1460741172317388961)** (2 条消息): 

> `Skills 网页版/桌面端发布, Prompt Engineering` 


- **Skills 网页版/桌面端集成处于悬而未决状态**：一位成员询问在网页版或桌面端应用上发布 **SKILLS** 的情况，以便将最佳 Prompt 转化为技能。
   - 然而，消息中未提供进一步的信息或时间表。
- **解构 Prompt Engineering**：一位成员要求解释 **Prompt Engineering** 的实际含义。
   - 另一位成员询问*它是否是指控制 LLM 行为以达到预期约束的人*。


  

---

### **Cursor 社区 ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1460684446096166923)** (66 条消息🔥🔥): 

> `Cursor 登录问题，退款申请，Cursor 的 plan 模式，在 iPhone 上镜像 agent 聊天窗口，带有 RAG 的 Chat agent 模板（支持任务）` 


- **企业版 Google 账号受登录重定向困扰**：有用户报告称，尝试使用企业 Google 账号登录 Cursor 控制面板 [cursor.com/dashboard](https://cursor.com/dashboard) 时，登录页面会不断重定向回登录页，但个人账号正常。
   - 用户确认该问题在不同电脑上均存在。
- **未使用额度但退款申请被拒**：用户 *thugbunny* 表示，尽管忘记取消订阅且未使用任何额度，Cursor 仍不予退款。
   - 另一位用户（*dan.perks*）提议如果该用户私信邮箱，他会协助核查：*"私信我你的邮箱，我会去跟进"*。
- **Plan 模式 Bug 频发**：用户报告 Cursor 的 plan 模式存在 Bug，一位用户报告了错误：*"The agent execution provider did not respond within 4 seconds. This may indicate the extension host is not running or is unresponsive."*
   - 降级到 **2.2.44 版本** 修复了此问题。
- **iPhone 镜像 Agent 聊天窗口寻求方案**：一位用户正在寻求在 iPhone 上镜像其 agent 聊天窗口的方法，且不需要完整的项目控制权。
   - 一名成员建议使用免费的 Chrome Remote Desktop。
- **寻找 RAG Agent 模板**：一位用户正在为客户构建自动化的聊天机器人/支持机器人（chatbot/support-bot），并在寻找一个成熟的、带有 **RAG (Retrieval-Augmented Generation)** 配置的 agent 模板。
   - 他们正在为客户开发自动化“聊天机器人/支持机器人”，需要可靠的模板。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1460713342451847178)** (56 条消息🔥🔥): 

> `TI 84 Plus Silver Edition 神经网络，LiquidAI 模型 CGGR 基准测试，老剧 AI 超分，zai 发布的 GLM-Image 模型，免费版不稳定性` 


- **TI-84 玩 Mastermind**：一位成员展示了在 **TI-84 Plus Silver Edition** 上运行的神经网络，该网络可以玩 **Mastermind** 游戏（根据秘密数字猜测 3-4 位数字序列），并在[附带视频](https://cdn.discordapp.com/attachments/1149866623109439599/1460713339976945674/2026-01-11_14-34-59.mp4?ex=6967eace&is=6966994e&hm=9923dcc08f64008ec696b845400620691ef2affb576ca9e66f4bed418063f386&)中进行了演示。
- **CGGR 受到关注（某种程度上）**：[smol.ai 的简报](https://news.smol.ai/issues/26-01-06-xai-series-e)提到了一个新的 **LiquidAI** 模型（[GitHub](https://github.com/some-repo) 上的 **CGGR**），目前正在进行基准测试（benchmaxxing）以评估其性能。
- **Al Bundy 迎来 AI 超分**：成员们讨论了对《Married with Children》等老剧进行 AI 超分的问题，认为 AI 可以为 **16:9** 版本插值缺失的细节，但有一位成员认为这会破坏艺术意图。
   - 另一位成员反驳说，《Married with Children》是一部演播室情景剧（studio static show），没人在意透视问题，超分会很受欢迎。
- **GLM-Image 模型由 Zai 发布**：**Zai** 发布了名为 **GLM-Image** 的新图像模型，其 [blog](https://z.ai/blog/glm-image) 和 [GitHub](https://github.com/zai-org/GLM-Image) 均已宣布。
- **免费版存在语言切换问题**：一位成员询问了免费版模型中出现的响应中断或语言切换（例如：开始是用中文，后来切成了英文）的问题。
   - 一位开发者回应称，可能是他们的供应商再次出现了不稳定性。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1460684945566470327)** (45 messages🔥): 

> `Qwen 模型尺寸, llama.hx 项目, LLM 工程, LM Studio GPU 问题, 代码自动补全设置` 


- **Qwen 的尺寸令用户惊讶**: 一位用户对 **Qwen** 模型的庞大尺寸表示惊讶，其 **BF16** 版本为 **160GB**，**Q4** 版本为 **40GB**。
   - 另一位用户澄清说，最小的 **Qwen3** 模型实际上是 **0.6B**，而 **Qwen3Next** 只是他们最新的 **80B** 模型的名称。
- **成员正在用 Haxe 重构 Llama.cpp**: 一位成员正在用 **Haxe** 重新开发 **llama.cpp**，命名为 **llama.hx**，以便在 Lua、JS 和 Python 等语言中原生使用。
   - 他展示了一张进度截图，并表示他是在 *AI 的帮助下* 重新开发 **llama.cpp** 的。
- **开发者专注于 LLM 集成**: 一位成员介绍自己是专注于 **LLM 集成**、**自主 Agent**、**工作流自动化**、**多模态 AI (语音与视觉)** 以及**区块链系统**的 AI 和全栈工程师。
   - 他们列举了在将 **LLMs** 与 **DSPy**、**LangChain**、**AutoGen** 和 **CrewAI** 集成方面的经验，以及构建连接模型与 API、数据库、业务逻辑和链上组件的生产级系统的经验，并补充道：*如果你需要开发者，请随时联系我。*
- **LM Studio 运行时更新令 GPU 用户感到沮丧**: 用户报告了 **LM Studio** v1.103.0 运行时的问题，特别是它破坏了在 **GPUs** 上的运行。
   - 一位用户哀叹道：*真遗憾，新的量化版本没能给我带来额外的 t/s。*
- **Vibe Coding 的最佳配置？**: 一位用户询问了关于 *vibe coding* 的最佳配置。
   - 一位成员建议通过网页界面免费使用 **Qwen3** 并配合自动补全，并指示它每次都显示完整的更新文件；另一位成员则推荐使用每月 10 美元的 **GitHub Copilot 和 Claude**。


---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1460723052798148783)** (8 messages🔥): 

> `AirLLM: 分层加载, DDR4 RAM 和 Xeon 性能` 


- **AirLLM 加载/卸载层**: 成员们提到了 **AirLLM** 以及通过一次加载和卸载一个层的方法，在 **4 GB GPUs** 上运行 **70b** 模型。
- **DDR4 RAM 和 Xeon 依然可行**: 一位成员分享了他们之前在 **DDR4 RAM** 和 **Xeon** 硬件上运行模型的经历。
   - 另一位成员指出，模型的效率提升并没有第一位成员想象的那么大。


---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1460692486107168942)** (12 messages🔥): 

> `纯 Rust LLM, AI Trace 模板, 无 Batchnorm 的 ML 系统, 微调 gpt-oss:latest, 合并课程频道` 


- **Rustacean 构建纯 Rust LLM**: 一些成员正尝试从头开始用 *纯 Rust* 构建 **LLMs**。
   - 最近服务器进行了一些**重组**，因此某些频道可能已经移动。
- **Discord 获得 AI Trace 模板**: 服务器为 🤖 Echo Lounge 接收了一个 **AI Trace 模板**。
   - 该机器人允许进行*瞬时*、*软性*且*阈限*的追踪，无需优化或记忆。
- **新型 ML 系统亮相（无 Batchnorm）**: 一位成员构建了一个新的 **ML 系统**，该系统不使用 **batchnorm**，不使用**激活函数**，且不会产生幻觉，但创造力较低。
   - 他们目前正在寻找有趣的项目创意，以证明其优势的实用性。
- **GPT-OSS 微调面临阻碍**: 一位成员询问如何轻松地用自己的信息微调 **gpt-oss:latest 模型**。
   - 另一位成员回答说，**gpt-oss:latest** 无法以官方方式轻松微调，目前大多数人都在使用 **RAG**。
- **课程频道合并！**: 所有课程频道已合并为[一个频道](https://discord.com/channels/879548962464493619/1329142738440028273)。
   - 此举旨在将所有课程信息和讨论汇集到一个易于访问的单一位置。


---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1460688145031889001)** (37 messages🔥): 

> `CGGR 预训练, Vast.ai 成本, Audioform 数据集` 


- **CGGR 热衷于微调，而非预训练**：成员们注意到 **CGGR** 并不适合预训练，应该将 **warmup steps** 设置为较大的值（如 **5000**）；**CGGR** 更多是面向微调的。
   - 此外，设置 *selection='stratified'* 可以让模型仍然能看到一些简单的 Token。
- **Vast.ai 之旅变得昂贵**：一位成员报告称在 **vast.ai** 上花费了 **$500** 用于模型的压力测试（crash-testing）。
   - 另一位成员指出，*对于该尺寸的模型，在 1x h200 上最多运行 24 小时即可*，并建议使用 h200 或 b200 会更具性价比。
- **AUDIOFORM 作为音频转视觉 ML 数据集亮相**：[AUDIOFORM 数据集](https://huggingface.co/datasets/webxos/audioform_dataset) 包含从上传的短 **WAV 文件**中捕获的 **10 个帧**，以及每帧的元数据，包括**主频（dominant frequency）、时间戳和捕获信息**。
   - webXOS 提供的 AUDIOFORM 已开放下载，以便开发者创建自己的类似数据集。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1460684030252023890)** (37 messages🔥): 

> `支持问题, 积分消耗过度, Manus x Similar Web 合作伙伴关系, 广告换取积分, AI 音乐生成` 


- **Manus 客户抱怨缺乏支持**：多位用户对 Manus 缺乏支持表示沮丧，理由是回复延迟以及积分和退款问题未得到解决。
   - 一位用户报告称在转接到人工客服后等待了 **8 小时**，而另一位用户提到由于支持问题，正考虑*彻底放弃 manus*。
- **用户指出 SimilarWeb 合作功能导致积分过度消耗**：多位用户报告称，在新的 **Manus x Similar Web** 合作伙伴功能中积分消耗惊人，一位用户在不到一分钟内消耗了 **5,000 积分**。
   - 另一位用户强烈建议不要测试该功能，称其在 **15 秒**内消耗了 **2,591 积分**，并建议增加一些**防护措施**。
- **Manus 用户呼吁通过广告获取积分**：一位用户建议实施基于广告的系统，让用户在积分用完时可以通过观看广告获取更多积分。
   - 该建议没有出现反对意见。
- **教程教授通过 Manus 创作惊人的 AI 音乐**：Manus AI 发布了一个 [YouTube 教程](https://youtu.be/zMBSmJupye8)，演示如何使用该平台创作 AI 音乐，并鼓励用户关注其中的 **pro tips**。
   - 内容标签为 **#ManusAIMusic**、**#AIComposition** 和 **#FutureOfMusic**。
- **Meta Verse Manus 集成**：一位用户建议 **Meta** 应该利用 **Manus** 将 **Google Tasks** 和 **Calendar** 等服务与 **Meta 显示眼镜**集成。
   - 该用户反对进行大规模的集成工作，主张采用 **Agentic AI** 实现后端功能的“粗暴方法”（dirty method）。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1460747616865226833)** (5 messages): 

> `PTX 指令, SMEM 指针参数, wgmma.mma_async, 矩阵描述符, 核心矩阵` 


- **PTX SMEM 指针参数特性受到质疑**：一位成员询问为什么某些带有 **SMEM 指针参数** 的 **PTX 指令**（如 `mbarrier.init.shared.b64`）需要 `"r"` 寄存器类型（通过 `__cvta_generic_to_shared` 实现 32 位）。
   - 他们将其与 `wgmma.mma_async` 进行了对比，后者要求在 `l` 寄存器类型中使用 **uint64** 的 **smem 地址**。
- **矩阵描述符 vs 通用共享内存地址**：一位成员推测 `wgmma.mma_async` 采用 **64 位地址**是因为它操作的是 *矩阵描述符（matrix descriptor）* 而非通用的共享内存地址，并链接到了 [NVIDIA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor)。
   - 他们澄清说那是矩阵描述符本身，而不是指向它的指针。
- **关于 8x2 核心矩阵的困惑**：成员询问为什么 `wgmma` 或 `tcgen05 mma` 的 **8x2 "核心矩阵"（core matrix）** 没有表示为 **8x32**（字节）或 **8x(32/每个元素的字节数)**（元素）。
   - 他们询问为什么每个 **8x1 切片**（8x16 连续字节）是一个有意义的表示方式。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1460698265551900834)** (8 条消息🔥): 

> `CUDA 学习资源, MLSys 研讨会, GPU Mode 提交` 


- **AI 学生寻求 CUDA 指导**：一名 AI 工程专业的学生正在寻求学习 **CUDA** 的建议，他具备 **Python**, **PyTorch**, **TensorFlow** 以及基础的 **C++** 知识。
   - 他们正在寻找免费的 **YouTube** 视频或课程，从基础开始学习 **CUDA**。
- **MLSys 会议追更**：一位成员询问了 **MLSys** 领域内重要的录制或未录制的研讨会、会议和演讲。
   - 回复中提到了 **PyTorch**, **ASAP Seminar**, **ICML/ICLR/Neurips**, **MLSys.org** 和 **ASPLOS**。
- **GPU Mode 提交协助**：一位首次提交者在租用的 **B200** 服务器上测试后，需要 **GPU Mode** 提交方面的帮助。
   - 提供了通过 [Web 界面](https://www.gpumode.com/v2/home) 或 [Discord bot](https://gpu-mode.github.io/kernelbot/docs/intro) 使用相应频道中的 `/leaderboard submit <test/benchmark/ranked/profile>` 进行提交的指导，一名用户报告提交成功。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1460685306893439142)** (14 条消息🔥): 

> `系统阅读小组, 西雅图 ML Sys 聚会, 创办小众俱乐部, GPU 会议` 


- **Shadaj 推荐的系统阅读小组**：一位成员推荐了由 [Shadaj](https://www.sfsystemsclub.com/) 运营的系统阅读小组，并建议关注他的 [Twitter](https://x.com/ShadajL) 以获取聚会通知。
   - 另一位身处南湾的成员表示感兴趣，但指出距离可能是一个挑战。
- **关于西雅图 ML Sys 聚会的询问**：一位成员询问西雅图是否有 **ML Sys 聚会**，好奇这类活动是否在湾区以外也存在。
   - 另一位成员建议探索大学的 **ML 俱乐部**，并启动关于系统主题的阅读小组。
- **建好了，他们就会来？**：一位成员分享了 *“如果你建好了，他们就会来”* 的想法，强调许多人有兴趣参加小众俱乐部，但很少有人愿意发起。
   - 针对这一点，另一位成员分享道，在他们的 *“成年生活中，我创造了很多没人关心的东西，这真的很艰难……”*
- **因失败的俱乐部而结成吐槽搭档**：两位成员开玩笑说要一起创办一个成人俱乐部，如果失败了就成为 *“吐槽搭档 (whining buddies)”*。
   - 其中一人想知道是否有类似于 **PyData/PyCon** 但纯粹针对 **GPU** 相关事宜的会议。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1460796244824686806)** (1 条消息): 

> `B200 不稳定性, Dual gemm, 提交截止日期延长, 新排行榜` 


- **B200 Runner 的不稳定性导致重跑**：由于广泛报告 **B200 runners** 在 **dual gemm 问题** 上的测量结果不稳定，提交截止日期延长至 **1月20日**。
   - 该问题源于评估代码、热指标和调度基础设施的交织，并且 *比预期的要复杂*。
- **Dual GEMM 排行榜分为两个阶段**：为了解决测量不稳定性，现有的 **dual gemm 排行榜** 将保持开放至 **1月16日**。
   - 新排行榜将于 **1月17日** 开启，只有向此新排行榜提交的内容才会计入奖金。
- **问题 #4 将于 1月20日 开启**：在新排行榜开启后，**问题 #4** 将从 **1月20日** 开放至 **2月20日**。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1460821976279941150)** (3 条消息): 

> `排行榜成就, Claude code 帖子` 


- **教师得益于 Claude 的启发登上排行榜**：一位学校老师受到 Mark 关于使用 **Claude code** 的 X 帖子的启发，通过自学登上了排行榜。
   - 这位老师感谢了一位成员提供如此美妙的体验，并表达了加入社区的喜悦。
- **社区庆祝这位老师的成功**：一位成员表示很高兴该平台为这位老师提供了如此好的体验。
   - 他们还提到，*未来还会有更多有趣的事情发生*。


  

---

### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1460722396964323574)** (1 messages): 

> `Helion 0.2.10, flex attention, oversubscribing SMs` 


- **Helion 0.2.10 发布并带来新功能**：**Helion 0.2.10** 现已发布，其特色是包含了一个 [flex attention 示例 kernel](https://github.com/pytorch/helion/blob/main/examples/flex_attention.py)。
   - 此版本还支持在 persistent kernels 上 **oversubscribing SMs**，并提供了一张说明 softmax 过载订阅（oversubscription）情况的图表。
- **支持 SM Oversubscription**：新版本支持在 persistent kernels 上对 **Streaming Multiprocessors (SMs)** 进行 oversubscribing，从而提高了资源利用率。
   - 由社区成员提供的一张图表展示了 **softmax** 在 oversubscription 下的效果。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1460754215411384320)** (2 messages): 

> `Issue Details` 


- **问题澄清即将到来**：一位成员提到他们给另一位用户写了信息，以更详细地解释某个问题，并提供了该信息的 [Discord 链接](https://discord.com/channels/1189498204333543425/1343350424253632695/1460796244824686806)。
   - 在给定的上下文中没有提供关于该问题的具体细节。
- **额外问题**：添加了另一个话题，因为至少需要 2 个。
   - 这是填充内容。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/)** (1 messages): 

cat_developer: 啊，谢谢
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1460709546485088296)** (17 messages🔥): 

> `Anthropic Labs, Pavlov's List, GLM-Image` 


- **Anthropic Labs 正在招聘**：Anthropic 宣布了 **Anthropic Labs** 的职位空缺，寻找适应性强、能在非结构化环境中茁壮成长，并能从容应对优先事项变化的人才（[职位空缺链接](https://job-boards.greenhouse.io/anthropic/jobs/5017202008)）。
   - Anthropic *不* 寻找那些*如果其领域变得不再相关就无法适应的深度专家*，或者是*需要明确路线图且会因优先事项变化而感到压力的人*。
- **Chris Barber 创建 RL 环境初创公司列表**：Chris Barber 推出了“**Pavlov's List**”，这是一个精选的强化学习（RL）环境初创公司集合（[链接](https://xcancel.com/chrisbarber/status/2010844746231804258?s=20)）。
   - 该列表按关注领域分类，如 **Code**、**Finance**、**Enterprise** 和 **ML Alignment**。
- **Z.ai 发布用于图像生成的 GLM-Image**：Z.ai 推出了 **GLM-Image**，这是一个采用混合自回归（auto-regressive）和扩散（diffusion）架构的开源模型（[链接](https://xcancel.com/zai_org/status/2011247591825068314?s=46&t=b7l37rB6wtbyAh6ah1NpZQ)）。
   - 该模型旨在实现**高保真视觉细节**和**卓越的文本渲染**，资源可在 **HuggingFace**、**GitHub** 及其官方博客上获取。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1460721110810099712)** (8 messages🔥): 

> `LTX-2 Open Source Video Model, Qwen Image Edit Gaussian Splats, GLM-Image` 


- **Venture Twins 发布 LTX-2**：来自 Venture Twins 的 Justine Moore 宣布发布 [LTX-2](https://xcancel.com/venturetwins/status/2010878914273697956?s=46)，这是一个新的**开源视频生成模型**，能够生成长达 **20 秒的 4K 片段**。
   - 该模型支持本地运行并包含**音频功能**，正如创作者 yanokusnir 所展示的那样。
- **Qwen 生成 Gaussian Splats**：社区正在讨论 **Qwen Image Edit** 将**图像转换为 Gaussian Splats** 然后从另一个角度重新渲染的能力（[Hugging Face 链接](https://huggingface.co/dx8152/Qwen-Image-Edit-2511-Gaussian-Splash)）。
   - 这种方法对于**起始帧 -> 结束帧类型的视频渲染**非常有用，能保持周围空间的一致性。
- **GLM-Image 的文本渲染优势**：据报道，**GLM-Image** 在通用图像生成质量上与主流 latent diffusion 方法对齐，但在**文本渲染**和**知识密集型生成场景**中表现出显著优势（[z.ai 博客](https://z.ai/blog/glm-image)）。
   - 它还支持丰富的 **image-to-image 任务**，包括图像编辑、风格迁移、身份保持生成（identity-preserving generation）以及多主体一致性。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1460689291289169922)** (5 messages): 

> `诈骗机器人，线下聚会` 


- **诈骗机器人被踢出**：一位成员提到了幽灵艾特（ghost ping），另一位成员澄清这是由于**诈骗机器人（scam bots）**引起的，这些机器人已被管理员迅速封禁。
- **为线下（IRL）聚会构思地点**：一位成员提议在 **NYC** 或 **SF** 等知名大城市组织线下聚会，以促进社区内的跨界交流。
   - 另一位成员指出，虽然线上读书会的参与人数可观，但要为定期的线下活动达到临界规模，可能需要**向更广泛的受众进行推广**，并引用了 **Cohere** 定期举办活动和 Zoom 会议作为例子。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1460717133511262270)** (13 messages🔥): 

> `SlopCodeBench, Agent 懈怠 (Agent Laziness), 社区驱动的基准测试, 基准测试中的 Prompt Engineering, ICLR Workshop 投稿` 


- ****SlopCodeBench** 在新博客中揭示了 Agent 懈怠现象**：一篇新的博客文章（[链接](https://x.com/GOrlanski/status/2011156105255346505)）强调了 **AI Agent** 是如何表现出“懈怠”的，这是更广泛的 **SlopCodeBench** 项目（[GitHub](https://github.com/SprocketLab/slop-code-bench)）的一部分。
   - **SlopCodeBench** 的目标是成为一个像 terminalbench 一样由社区驱动的基准测试，欢迎关于增加新问题的反馈。
- ****SlopCodeBench** 分解问题以惩罚早期的设计选择**：**SlopCodeBench** 将大型编程问题分解为多个检查点，早期的设计决策可能会对后期阶段产生负面影响。
   - 这些问题在设计时没有提供实现提示，以确保 **Agent** 能够做出自己的决策。
- **关于编程基准测试中 Prompt 简洁性的讨论**：有人对依赖大量 **Prompt Engineering** 才能获得不错性能的编程基准测试表示担忧。
   - 辩论认为，简单的 Prompt 最能反映实际使用情况，特别是当代码符合合理的上下文窗口（context window）时，这与像 terminalbench 这样的 **Agent** 评估方法形成了对比。
- **建议将博客文章投稿至 **ICLR Workshop****：有人建议将关于 Agent 懈怠的博客文章作为高质量稿件投稿至 [此 ICLR workshop](https://sites.google.com/view/icbinb-2026)，并为投稿过程提供协助。
   - 该研讨会的截止日期是 1 月 31 日，作者在咨询导师后正在考虑此事。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1460696581136187726)** (1 messages): 

> `文件系统错误，存储限制调试` 


- **文件系统故障 (Snafu)**：一位成员提到，错误是由于使用了存储空间受限的不同文件系统导致的。
- **存储限制再次引发问题**：根本原因被确定为误用了存储容量有限的文件系统。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1460742072536535082)** (3 messages): 

> `诈骗信息骚扰, Lucid Coding` 


- **诈骗者被踢出**：一位成员报告 <@139426008976588801> 再次发送诈骗骚扰信息（SpamingScammers），另一位成员确认该情况已处理。
- **“Lucid Coding”概念引发关注**：一位成员对 *“Lucid Coding”* 这个术语表示赞赏，并分享了一个引用该概念的 [链接](https://fxtwitter.com/i/status/2011137879112908870)。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1460689303314104351)** (9 条消息🔥): 

> `贝叶斯 vs 频率派统计学, FDA 腐败, MedGemma` 


- **贝叶斯统计不是一种飞跃？**: 一位成员表示，贝叶斯和频率派统计学使用相同的统计技术（如线性回归和逻辑回归），并称贝叶斯方法仅仅是*一种不同的思维方式*。
   - 另一位成员反驳称，它们虽然都使用相同的公式，但在先验（prior）、后验（posterior）和干预（intervention）的解释上有显著差异，并链接到了 [Probability interpretations](https://en.wikipedia.org/wiki/Probability_interpretations)（概率解释）。
- **MedGemma 1.5 用于医学图像解读**: Google 的 **MedGemma 1.5** 承诺提供下一代医学图像解读和语音转文本（speech-to-text）能力，详见 [Google Research 博客](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/)。
- **贝叶斯方法会导致临床试验中的欺诈？**: 一位成员担心贝叶斯方法虽然更灵活，但可能成为*临床试验中欺诈和腐败的另一种手段*。
   - 另一位成员指出，目前尚未观察到贝叶斯相关的 FDA 腐败，因此*可以为其分配一个零先验（zero prior）*，并认为后验概率（posterior probability）基本为零。
- **FDA 腐败在阿片类药物危机中的作用？**: 一位成员认为 **FDA 腐败** 可能在阿片类药物危机中起到了主要的助推作用。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1460835433628958772)** (2 条消息): 

> `Mojo 文档, NotebookLM, llms.txt` 


- **寻找 Mojo 文档的用户咨询 NotebookLM 集成**: 一位成员询问如何将**最新的官方 Mojo 完整文档**导入 **NotebookLM**，并询问是否有 **PDF** 或 **Markdown** 版本可用。
   - 另一位成员建议使用 `llms.txt` 文件 ([https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt](https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt)) 来为 **LLM**（如 NotebookLM）提供文档。
- **为 NotebookLM 建议使用 llms.txt 文件**: 一位成员建议使用 `llms.txt` 文件 ([https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt](https://docs.modular.com/max/coding-assistants#supply-documentation-to-llms-with-llmstxt)) 来为 **LLM**（如 NotebookLM）提供文档。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1460815670701719838)** (4 条消息): 

> `Qwen3-VL, MoE 实现, 贡献者指南更新` 


- **Qwen3-VL 的 MoE 实现受到质疑**: 一位成员质疑为什么 **Qwen3-VL** 只有 **MoE 实现**，并建议重用 [qwen3vl_moe](https://github.com/modular/modular/tree/main/max/python/max/pipelines/architectures/qwen3vl_moe) 的代码，以使稠密（dense） **Qwen3VL** 模型（如 [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)）能够正常工作。
- **贡献者指南欢迎 PR**: 一位成员表示 **欢迎提交 PR**，理由是目前缺乏贡献者来跟进整个 **MAX** 生态系统。
   - 他们还指向了[更新后的贡献者指南](https://github.com/modular/modular/commit/571ae7132c465f155d50f7bbe7cf77064a02e9bf)。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1460759556501016681)** (1 条消息): 

> `Glama 排名, 服务器使用指标` 


- **基于使用情况的 Glama 排名**: **Glama** 的创始人澄清说，他们的排名是基于**服务器使用指标（server usage metrics）**。
   - 他们邀请用户通过私信（DM）提供反馈，并表示对任何所谓的排名滥用行为并不知情。
- **创始人回应排名滥用指控**: **Glama** 的创始人确认了其身份，并回应了关于其排名系统可能存在滥用行为的担忧。
   - 他们强调排名是由**服务器使用指标**决定的，并欢迎直接反馈。


  

---

### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1460702731798053165)** (5 messages): 

> `Tasks Spec Implementations, glama.ai/mcp/inspector` 


- **寻求 Tasks Spec 客户端实现**：一名成员询问是否有客户端应用实现了 **Tasks spec**，并寻求 UI 实现方案的示例；另一名成员提到了 Typescript SDK。
   - 作为回应，另一名成员宣布即将提交一个 PR，为 **Inspector** 添加任务功能，同时还有一个用于在 server-everything 中模拟长时间运行任务的 PR。
- **glama.ai Inspector 追求功能对等**：一名成员分享了他们在 [glama.ai/mcp/inspector](https://glama.ai/mcp/inspector) 上的 **Inspector** 实现早期版本，目标是涵盖所有功能。
   - 该成员澄清说，他们内部将其用于 **e2e testing**。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1460700888200118449)** (5 messages): 

> `AI-assisted code generation, Replit, DSPY OS, DSPY Framework` 


- **AI 平台助力代码生成**：成员们注意到有多个平台提供 **AI-assisted code generation**（AI 辅助代码生成），例如 **Replit** 和 **DSPY OS**。
   - 这些工具可以自动化各种编码流程，提高生产力。
- **DSPY OS 未发现踪迹**：一位成员询问 **DSPY OS**，指出 *"什么是 DSPy os？我找不到任何关于它的信息"*。
   - 另一位成员指出，DSPY 更多是一个 **framework** 而非平台，因此目前还没有直接使用 DSPY 构建的类似 Replit 的项目，但你可以使用 DSPY 创建自定义工具或环境来自动化特定的编码任务。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/)** (1 messages): 

hsaliak.: 使用 aider 时，是否可以为 gemini model 使用 oauth login？它的限制（limits）更高。
  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1460769810316267601)** (1 messages): 

> `Prompt Engineering for Outreach, Clay + AI outreach workflow` 


- **通过 Clay + AI 触达工作流获得工作**：一个关于 **Prompt Engineering for Outreach**（外联提示工程）的研讨会将教授如何构建 **Clay + AI outreach workflow**，将信号转化为大规模的**个性化消息**。
   - 该研讨会承诺实现 **40%+ 的接受率**和 **18%+ 的回复率**，并包含可重复使用的工作流和可直接复制粘贴的提示词；点击[此处](https://luma.com/jt1vr0u5)或[此处](https://luma.com/mmqa4fhi)报名。
- **为外联设计提示词**：这场 90 分钟的直播研讨会将拆解用于真实客户的精确 **Clay + AI** 系统。
   - 研讨会将涵盖端到端的 AI 外联工作流、编写高质量且不尴尬的外联提示词，以及可选的 Apollo、Attio、n8n 集成。