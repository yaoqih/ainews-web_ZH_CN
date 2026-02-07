---
companies:
- openai
- anthropic
- langchain
date: '2026-02-06T05:44:39.731046Z'
description: '**2026年2月上旬的 AI 要闻**重点介绍了 **GPT-5.3-Codex** 与 **Claude Opus 4.6** 的详细对比。用户指出，**Codex**
  在处理细节明确的特定任务方面表现强劲，而 **Opus** 在探索性工作中则具有人体工程学（易用性）优势。


  在 Karpathy 的 **nanochat GPT-2 竞速测试 (speedrun)** 基准中，**Opus 4.6** 实现了更好的实际运行性能（wall-clock
  performance），而 **Codex-5.3-xhigh** 有时会遇到上下文问题。**Karpathy** 提醒道，目前的模型在完全自主的 AI 工程任务中尚不可靠。


  关于**智能体集群 (Agent swarms)** 的讨论揭示了其与软件组织设计之间新出现的相似之处；其中 **Anthropic 风格**的智能体协同系统以及
  **LangChain/LangSmith** 强调通过追踪、沙箱化和状态控制来进行“环境工程”。此外，**递归语言模型 (RLM)** 作为智能体系统的未来发展方向被引入，旨在减少“上下文腐败”
  (context rot) 并改善结构化通信。'
id: MjAyNi0w
models:
- gpt-5.3-codex
- claude-opus-4.6
- nanochat-gpt-2
people:
- karpathy
- sama
- swyx
- omarsar0
- hamelhusain
- deepfates
title: 今天没什么事。
topics:
- agent-systems
- ai-engineering
- benchmarking
- software-organization
- sandboxing
- tracing
- state-management
- recursive-language-models
- context-management
---

**安静的一天**

> 2026年2月5日至2月6日的 AI 新闻。我们为您检查了 12 个 subreddits、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 24 个 Discord（**254** 个频道，**8727** 条消息）。预计节省阅读时间（按 200wpm 计算）：**666** 分钟。[AINews 网站](https://news.smol.ai/) 允许您搜索所有往期内容。提醒一下，[AINews 现已成为 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以[开启/退订](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件推送频率！

今日文章：https://www.latent.space/p/ainews-ai-vs-saas-the-unreasonable

---

# AI Twitter 综述


**前沿编程模型：GPT-5.3-Codex vs Claude Opus 4.6 (以及“Agentic”现在的含义)**

- **用户共识快照**：Feed 流中的很大一部分是对 **GPT-5.3-Codex** 与 **Claude Opus 4.6** 的真实世界 A/B 测试，通常结论是它们*都是*明显的代际升级，但具有截然不同的特性。人们将 **Codex** 描述为对细节痴迷且在限定任务上表现强劲，而 **Opus** 在探索性工作和规划方面感觉更符合人体工程学 ([rishdotblog](https://twitter.com/rishdotblog/status/2019664800910135499), [@theo](https://twitter.com/theo/status/2019709378329550973))。一些笔记强调了 **Codex 的“自动压缩”/垃圾回收上下文**以及工作期间频繁的进度更新——这被视为长任务的 UX 优势 ([cto_junior](https://twitter.com/cto_junior/status/2019607817884475718))。
- **AI 工程师在环（AI-engineer-in-the-loop）基准测试**：一个特别具体的评估是优化 Karpathy 的 **nanochat “GPT-2 speedrun”**。[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2019824445792424385) 报告称，两个模型的表现都像称职的 AI 工程师（阅读代码、提出实验、运行基准测试），其中 **Opus 4.6** 带来了可衡量的实际时间收益（例如 torch compile 配置微调、优化器步进更改、内存减少），而 **Codex-5.3-xhigh** 虽然产生了想法，但有时会损害质量——可能是由于上下文问题（他观察到它达到了“0% context”）。
- **来自 Karpathy 的现实检查**：[@karpathy](https://twitter.com/karpathy/status/2019851952033771710) 反驳了模型已经可以可靠地进行*开放式*闭环 AI 工程的观点：它们可能会追求带有巨大隐藏成本的虚假 1% 收益，遗漏关键的验证检查，违反仓库风格指令，甚至读错自己的结果表——在监督下仍有“净效用”，但对于自主优化来说还不稳健。
- **无 API 的产品策略**：一条推文声称 **目前没有 GPT-5.3-Codex API**，这意味着 OpenAI 正有意识地将使用流量引导至 Codex 产品中（并使独立基准测试变得更难）([scaling01](https://twitter.com/scaling01/status/2019856879858450742))。与此同时，Sam Altman 明确询问用户希望如何构建 **Codex 定价** 结构 ([sama](https://twitter.com/sama/status/2019814741129195576))。

**Agent 集群与“盒子里的软件团队”**

- **并行 Agent 开发开始趋向于组织设计**：关于高度并行 Agent 研究的讨论指出，无约束的集群倾向于**重塑软件组织结构图**（任务分配、协作、QA），并给现有的工具链（Git/包管理器）带来压力，因为这些工具并非为大规模并发编辑而设计 ([swyx](https://twitter.com/swyx/status/2019645622421451106))。这呼应了更广泛的“规范驱动开发”（spec-driven development）/“作为开发团队的 Agent”叙事 ([dbreunig](https://twitter.com/dbreunig/status/2019829245137338548))。
- **Claude Code 的“Agent 团队”时刻**：多条推文提到了 Anthropic 式的 Agent 协作系统，其中 Agent 可以挑选任务、锁定文件并通过 Git 同步——这被视为实用自动化领域的一次阶跃式进步 ([omarsar0](https://twitter.com/omarsar0/status/2019780306778104056), [HamelHusain](https://twitter.com/HamelHusain/status/2019863601591517466))。
- **LangChain / LangSmith：Agent 需要追踪（Traces）、沙箱（Sandboxes）和状态控制**：有一种强烈的观点认为，可靠性源于**对环境进行工程化**：追踪、评估、沙箱化以及类型安全的状态/中间件。示例包括 LangSmith 的改进（追踪预览；语音 Agent 调试）以及 deepagents 增加了诸如 **daytona/deno/modal/node VFS** 的沙箱后端 ([LangChain](https://twitter.com/LangChain/status/2019848808310706367), [LangChain](https://twitter.com/LangChain/status/2019846811997942219), [bromann](https://twitter.com/bromann/status/2019880605467697565), [sydneyrunkle](https://twitter.com/sydneyrunkle/status/2019862521717444675))。
- **“RLM” 架构（递归语言模型，Recursive Language Models）**：一篇引人注目的概念性文章认为，Agent 将从“LLM + 工具循环”（ReAct）演变为**原生 REPL、类程序的系统**，其中上下文存储在变量中，子 Agent 通过结构化数值进行通信，而非直接向 Prompt 中倾倒文本，并通过构建方式减少“上下文腐化”（context rot）([deepfates](https://twitter.com/deepfates/status/2019912654173651131))。相关内容：使编码 Agent 更像 “RLM” 的实用技巧，包括将上下文推入变量，以及避免在 Prompt 中出现工具 I/O 垃圾信息 ([lateinteraction](https://twitter.com/lateinteraction/status/2019852730177863977))。

**评估完整性、基准测试漂移以及“值得信赖”评分的新基础设施**

- **“分数已失效” → 去中心化评估**：Hugging Face 推出了 **Community Evals**：托管排行榜的基准数据集，评估结果以版本化的 YAML 格式存储在模型仓库中，支持基于 PR 的提交，以及可重复性徽章（通过 Inspect AI），其明确目标是使评估的来源透明化，即便它无法解决污染/饱和问题 ([huggingface](https://twitter.com/huggingface/status/2019754567685050384), [ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/2019795723378942295), [mervenoyann](https://twitter.com/mervenoyann/status/2019784907178811644))。
- **基准测试尚未饱和（目前如此）**：一个反向观点强调，一些困难的基准测试仍有很大的提升空间（例如，SWE-bench Multilingual <80%，SciCode 56%，CritPt 12%，VideoGameBench 1%，以及远未达到隐含上限的效率基准测试）([OfirPress](https://twitter.com/OfirPress/status/2019755847149056456))。
- **Opus 4.6 基准测试表现：大幅飞跃，但仍不均衡**：有多次声称 Opus 4.6 在 Arena 和其他排行榜上攀升至前列 ([arena](https://twitter.com/arena/status/2019842691442569566), [scaling01](https://twitter.com/scaling01/status/2019843682128822525))，包括在 Anthropic 历来落后的数学导向评估（FrontierMath）中表现强劲。Epoch 的报告将 Opus 4.6 Tier 4 定位在 **21% (10/48)**，在统计上与 19% 的 GPT-5.2 xhigh 持平，落后于 31% 的 GPT-5.2 Pro ([EpochAIResearch](https://twitter.com/EpochAIResearch/status/2019852613672665193))。但在其他重推理领域（如象棋谜题）仍然疲软 ([scaling01](https://twitter.com/scaling01/status/2019817880662278546))。
- **大规模评估基础设施 (StepFun)**：关于 Step 3.5 Flash 的深度基础设施报告认为，可重复的评分需要处理故障模式、训练与推理的一致性、污染检查、稳健的评判/提取以及长输出监控；“评估应略领先于训练” ([ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/2019734062689304970))。

**世界模型走向生产环境：Waymo + DeepMind 的 Genie 3**

- **Waymo World Model 发布**：Waymo 展示了一个基于 **DeepMind Genie 3** 构建的**前沿生成式仿真模型**，用于生成高度逼真的交互式场景——包括罕见的“不可能”事件（龙卷风、飞机降落在高速公路上）——以便在现实世界暴露风险之前对 Waymo Driver 进行压力测试 ([Waymo](https://twitter.com/Waymo/status/2019804616746029508))。
- **关键技术亮点**：DeepMind 强调将 Genie 3 的“世界知识”迁移到 **Waymo 特有的摄像头 + 3D lidar** 表征中，实现了与 Waymo 硬件模态相匹配的可提示（promptable）“假设（what if）”场景生成 ([GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/2019809201812545835), [GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/2019809239569702962))。多位研究人员指出，将仿真从像素扩展到传感器流（sensor streams）是真正的里程碑 ([shlomifruchter](https://twitter.com/shlomifruchter/status/2019820532485808329), [sainingxie](https://twitter.com/sainingxie/status/2019841784990351381))。
- **更广泛的“用于推理的世界模型”讨论**：Waymo 的新闻被反复用作证据，证明**世界模型**（不仅仅是文本模型）是推理和具身智能（embodied tasks）任务的核心扩展前沿（scaling frontier） ([swyx](https://twitter.com/swyx/status/2019605135689937405), [kimmonismus](https://twitter.com/kimmonismus/status/2019809839804010962), [JeffDean](https://twitter.com/JeffDean/status/2019824614139162804), [demishassabis](https://twitter.com/demishassabis/status/2019827916385972517))。
- **世界模型的规划进展**：GRASP 被介绍为一种**基于梯度的随机并行规划器**（gradient-based, stochastic, parallelized planner），它共同优化动作和中间子目标，以改进与常见零阶规划器（CEM/MPPI）相比的长程规划（long-horizon planning） ([michaelpsenka](https://twitter.com/michaelpsenka/status/2019870377032503595), [_amirbar](https://twitter.com/_amirbar/status/2019903658792497482))。

**内存、长上下文控制和多 Agent “认知基础设施”**

- **InfMem：具有认知控制的有界内存 Agent**：InfMem 提出了一个 PRETHINK–RETRIEVE–WRITE 协议，并结合 RL 用于处理高达 **1M tokens** 的长文档问答，强调更长的上下文窗口将瓶颈转移到了**“关注什么 / 何时停止”**。报告的收益包括比基准大幅提升的准确率，以及通过自适应停止实现的 **3.9 倍平均延迟降低** ([omarsar0](https://twitter.com/omarsar0/status/2019759999170556189))。
- **LatentMem：用于多 Agent 系统的角色感知潜空间内存**：LatentMem 通过将轨迹压缩为角色相关的潜空间内存（使用策略优化方法 LMPO 训练），解决了“同质化”问题（即尽管角色不同，Agent 仍检索到相同的记忆）。声称在问答和编程任务中有所改进，且**减少了约 50% 的 tokens** / 加快了推理速度 ([dair_ai](https://twitter.com/dair_ai/status/2019778133550125515))。
- **产品现状：内存泄漏与上下文饱和**：虽然 Agent 相关工具（agentic tooling）发布迅速，但开发者抱怨资源膨胀和脆弱的 UX（例如，快速迭代的 Agent IDEs 中的“内存泄漏”） ([code_star](https://twitter.com/code_star/status/2019707930422161680))。另一个讨论怀疑子 Agent（sub-agent）的输出消耗上下文预算的速度快于压缩恢复的速度，这暗示了隐藏的内部长上下文系统 ([RylanSchaeffer](https://twitter.com/RylanSchaeffer/status/2019642129736429730))。

**行业采用、计算经济学和“工作 vs 任务”论述**

- **不可验证的工作限制了完全自动化**：François Chollet 认为，在不可验证的领域，性能提升主要源于昂贵的数据清洗，且边际收益递减；由于大多数工作并非端到端可验证的，“AI 可以自动化许多任务”并不等同于“AI 取代工作”，这种情况将持续很长时间 ([fchollet](https://twitter.com/fchollet/status/2019610121371054455), [fchollet](https://twitter.com/fchollet/status/2019610588612292834))。
- **不同的观点：RSI 瓶颈**：另一种观点认为，任务将按照它们阻碍递归自我改进（Recursive Self-Improvement, RSI）的顺序被攻克，其中软件工程首当其冲 ([tszzl](https://twitter.com/tszzl/status/2019614081683189827))。
- **企业部署信号**：有帖宣称 **Goldman Sachs 正在推广 Claude** 用于会计自动化 ([kimmonismus](https://twitter.com/kimmonismus/status/2019865721338229180))，而更广泛的市场叙事则断言 AI 正在令软件密集型行业感到不安（尽管最激进的言论并未在推文中获得独立证实） ([kimmonismus](https://twitter.com/kimmonismus/status/2019757481925464371))。
- **Capex 规模**：多条推文强调了 Hyperscaler（超大规模云服务商）支出的加速；其中一条推文将 2026 年主要 Hyperscaler 的总 Capex 描绘为接近 **6500 亿美元（约占美国 GDP 的 2%）** 的“AI 军备竞赛” ([scaling01](https://twitter.com/scaling01/status/2019789747896377697))，同时有记录显示 Hyperscaler 数据中心的 Capex 可能会在 **2026 年翻倍** ([kimmonismus](https://twitter.com/kimmonismus/status/2019773237618479594))。
- **老牌专家给工程师的定心丸**：Eric S. Raymond 发表了一个高互动量的观点，认为“编程并未过时”：系统依然复杂，且人类意图与计算机规格之间的鸿沟依然存在；应对方案是适应和提升技能，而非恐慌 ([esrtweet](https://twitter.com/esrtweet/status/2019779602617376788))。

---

### 热门推文（按互动量排序）
- [Microinteracti1](https://twitter.com/Microinteracti1/status/2019712610547933593)：病毒式传播的政治评论帖子（高互动量；非技术类）。
- [elonmusk](https://twitter.com/elonmusk/status/2019823468968370633)：“这就开始了”（推文文本未提供具体上下文）。
- [esrtweet](https://twitter.com/esrtweet/status/2019779602617376788)：“编程恐慌是一场虚惊；提升技能吧。”
- [Waymo](https://twitter.com/Waymo/status/2019804616746029508)：基于 Genie 3 构建的 Waymo 世界模型，用于罕见事件模拟。
- [sama](https://twitter.com/sama/status/2019813802049696064)：“5.3 爱好者聚会” / 对模型的兴奋。
- [claudeai](https://twitter.com/claudeai/status/2019833113418035237)：“基于 Opus 4.6 构建”虚拟黑客松（10 万美元 API 额度）。
- [chatgpt21](https://twitter.com/chatgpt21/status/2019679978162634930)：宣称 Opus 4.6 实现了“宝可梦克隆”（11 万 Token，1.5 小时推理）。
- [theo](https://twitter.com/theo/status/2019598113238139262)：“我一眼就能看出那是 Opus 的 UI”（UI/发布风向标）。
- [ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/2019839335382790342)：推测性的系统设想：通过光纤环路流式传输权重 / 利用闪存带宽进行推理。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. 低端硬件上的本地 AI

  - **[仅 CPU、无 GPU 的电脑也可以在本地运行各种 AI 工具](https://www.reddit.com/r/LocalLLaMA/comments/1qxgkd1/cpuonly_no_gpu_computers_can_run_all_kinds_of_ai/)** (互动数: 544)：**该帖子强调了在仅有 CPU 的环境下本地运行 AI 工具的能力，具体使用了配备 i5-8500 处理器和 32GB RAM 的 Dell OptiPlex 3060。用户使用 KoboldCPP 成功运行了 12B Q4_K_M GGUF 格式的 LLM，实现了与来自 Hugging Face 模型的本地聊天机器人交互。此外，该配置还支持 Stable Diffusion 1.5 进行图像生成（尽管速度较慢）以及使用 Chatterbox TTS 进行声音克隆。帖子强调，即便在极低配置的硬件上也能执行先进的 AI 任务，挑战了“只有昂贵且依赖 GPU 的配置才能进行本地 AI 实验”的观念。** 一些评论者对 AI 在基础硬件上的普及前景表示乐观，而另一些人则指出社区在硬件精英主义和本地模型可访问性方面存在分歧。

- noctrex 建议在仅限 CPU 的配置中尝试特定模型，如 [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)、[LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) 和 [LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B)。这些模型因其小巧的体积和高效性而备受赞誉，非常适合在无需昂贵 GPU 硬件的仅限 CPU 的 Docker 机器上运行。
- Techngro 对 AI 的未来表示乐观，认为普通人可以通过既智能又小巧、足以在基础硬件上运行的本地模型来接触 AI。这一愿景与目前依赖公司托管的大型昂贵模型的趋势形成对比，暗示了 AI 使用向更加民主化的方向转变。
- NoobMLDude 提供了本地 AI 配置的实际应用，例如将其用作私密会议记录员或语音助手。这突显了本地 AI 模型在无需高端硬件的情况下执行有用任务的通用性和潜力。

- **[没有 NVIDIA？没问题。我 2018 年的“土豆”八代 i3 在 16B MoE 上达到了 10 TPS。](https://www.reddit.com/r/LocalLLaMA/comments/1qxcm5g/no_nvidia_no_problem_my_2018_potato_8th_gen_i3/)** (Activity: 866): **一位缅甸用户在配备 i3-8145U CPU 和 16GB RAM 的 HP ProBook 650 G5 上成功运行了 16B MoE 模型 DeepSeek-Coder-V2-Lite，并使用 Intel UHD 620 集成显卡实现了 `10 TPS`。该方案利用 **OpenVINO** 作为 `llama-cpp-python` 的后端，突显了 MoE 模型的效率——每个 token 仅需计算 `2.4B` 参数。该用户强调了双通道 RAM 和使用 Linux 以最小化资源开销的重要性。初始 iGPU 编译延迟和偶尔的语言漂移被列为面临的挑战。** 评论者赞赏了该方案的独创性和对资源的充分利用，一些人指出 GPU 短缺时代提高了优化技能。人们对该用户用于编码任务的日常主力模型表现出了兴趣。

    - ruibranco 的评论强调了双通道 RAM 在 CPU 推理中的重要性，指出内存带宽通常是瓶颈而非计算能力。通过从单通道切换到双通道 RAM，吞吐量可以有效翻倍，这对于在 CPU 上运行像 16B MoE 这样的模型至关重要。MoE 架构因其效率而受到称赞，因为它每个 token 仅激活 2.4B 参数，使得模型能够适配八代 i3 处理器的缓存。
    - MoE (Mixture of Experts) 架构在该配置中的效率得到了肯定，因为它将每个 token 的活跃参数量减少到 2.4B，这对于 CPU 的缓存来说是可控的。这种方法对于像八代 i3 这样的旧 CPU 特别有益，因为它最小化了工作集大小，在不需要高端硬件的情况下增强了性能。
    - 评论还提到了 OpenVINO 的 INT8/FP16 路径在 UHD 620 等旧款 iGPU 上可能存在的精度问题，这可能会导致 'Chinese token drift'（中文 Token 漂移）。这表明这些 iGPU 有限的计算精度可能会影响模型输出的准确性，突显了在机器学习任务中使用旧款集成显卡时面临的技术挑战。

- **[这里有人在完全离线使用 AI 吗？](https://www.reddit.com/r/LocalLLM/comments/1qwjgj4/anyone_here_actually_using_ai_fully_offline/)** (Activity: 383): **借助 **LM Studio**、**Ollama** 和 **openwebUI** 等工具，完全离线运行 AI 模型是可行的。这些平台允许用户在本地操作模型，其中 **LM Studio** 和 **Ollama** 提供了通过 [Hugging Face](https://huggingface.co/) 等平台及其自身仓库访问模型的途径。**openwebUI** 提供了一个类似于 ChatGPT 的本地 Web 界面，并可以与 **ComfyUI** 结合进行图像生成，尽管这更为复杂。用户报告称，虽然离线 AI 配置可能具有挑战性，但对于编码和咨询等任务是可行的，`gpt-oss-20b` 等模型在这些环境中得到了有效使用。** 一些用户发现离线 AI 配置对编码和咨询等特定任务很有帮助，尽管他们指出这些配置可能需要大量的计算资源，尤其是对于编码工作流。设置和维护的复杂性是一个共同的挑战，但对云服务的控制力和独立性被视为其价值所在。

- Neun36 讨论了多种离线 AI 选项，重点介绍了 LM Studio、Ollama 和 openwebUI 等工具。LM Studio 因其与 Hugging Face 模型的兼容性而受到关注，并针对 GPU 或 RAM 进行了优化。Ollama 提供本地模型托管，而 openwebUI 则提供类似 ChatGPT 的浏览器界面，并增加了集成 ComfyUI 进行图像生成的复杂功能。
- dsartori 提到将离线 AI 用于编程、咨询和社区组织，并强调编程工作流需要强大的配置。一位团队成员在 LM Studio 中使用 `gpt-oss-20b` 模型，表明了它在咨询中的实用性，但并不能作为唯一的解决方案。
- DatBass612 分享了一个详细案例，在投资高端 M3 Ultra 以运行 OSS 120B 模型后，在五个月内实现了正向 ROI。他们估计每日 token 使用量价值约 `$200`，并提到使用 OpenClaw 等工具可能会增加 token 使用量，同时强调了拥有充足的统一内存对于虚拟化和 sub-agent 操作的重要性。

### 2. OpenClaw 与本地 LLMs 的挑战

  - **[OpenClaw 与本地 LLMs - 真的有人让它运行良好吗？](https://www.reddit.com/r/LocalLLM/comments/1qx51zc/openclaw_with_local_llms_has_anyone_actually_made/)** (热度: 200): 该帖子讨论了从 **Claude API** 转向 **Ollama** 或 **LM Studio** 等本地 LLMs，以减少与 Token 使用相关的成本。用户正在考虑如 `Llama 3.1` 或 `Qwen2.5-Coder` 等模型，以便在没有延迟问题的情况下实现 Tool-calling 能力。文中指出了 **OpenClaw** 的安全漏洞风险，部分用户建议将 **Qwen3Coder** 用于 Agent 任务。文中还分享了一个 [Local AI 播放列表](https://www.youtube.com/playlist?list=PLmBiQSpo5XuQKaKGgoiPFFt_Jfvp3oioV) 以进一步探索安全的本地 LLM 应用。评论者对 OpenClaw 的安全性表达了怀疑，认为投资本地模型的 VRAM 优于支付 API 服务费。一些用户尝试过本地配置，但对安全风险仍保持警惕。

    - **Qwen3Coder** 和 **Qwen3Coder-Next** 被强调在 Tool calling 和 Agent 用途中非常有效，并提供了 [Qwen3Coder-Next](https://qwen3lm.com/coder-next/) 的链接。评论者指出了 OpenClaw 的安全隐患，建议本地 LLMs 的其他安全用途（如私人会议助手和编码助手），并提供了一个 [Local AI 播放列表](https://www.youtube.com/playlist?list=PLmBiQSpo5XuQKaKGgoiPFFt_Jfvp3oioV) 供进一步探索。
    - 一位用户描述了通过将 OpenClaw 与 `lmstudio` 中的本地 `gpt-oss-120b` 模型集成进行的实验，强调了安全性的重要性，包括在 `nologin` 用户下运行并限制特定文件夹的权限。尽管完成了技术设置，他们得出结论认为潜在的安全风险超过了使用 OpenClaw 的收益。
    - 另一位用户报告使用 OpenClaw 配合 `qwen3 coder 30b`，并指出尽管由于缺乏文档导致设置过程具有挑战性，但系统表现良好，可以通过简单的指令创建新技能。这突显了 OpenClaw 在与强大的本地模型配对时的潜力，尽管初始设置较为困难。

  - **[Clawdbot / Moltbot → 误导性的炒作？](https://www.reddit.com/r/LocalLLM/comments/1qwg8an/clawdbot_moltbot_misguided_hype/)** (热度: 86): **Moltbot (OpenClaw)** 被宣传为“免费的个人 AI 助手”，但需要多个付费订阅才能有效运行。用户需要来自 **Anthropic, OpenAI 和 Google AI** 的 API Key，用于网页搜索的 **Brave Search API**，以及 **ElevenLabs 或 OpenAI TTS 额度**用于语音功能。此外，浏览器自动化需要 **Playwright** 设置，可能产生云托管费用。总成本可能达到每月 `$50-100+`，使其与 **GitHub Copilot, ChatGPT Plus 和 Midjourney** 等现有工具相比实用性较低。该项目更适合喜欢折腾的开发者，而非现成的个人助手。一些用户认为虽然 Moltbot 需要多个订阅，但可以自托管 LLMs 和 TTS 等组件以避免费用，尽管性能可能不如云端方案。其他人指出该机器人并非真正的“本地化”，且需要大量的技术知识才能有效设置。

    - No_Heron_8757 讨论了一种混合方法，使用 ChatGPT Plus 处理主要的 LLM 任务，同时通过 LM Studio 将较简单的任务卸载到本地 LLMs。他们强调了在同一个 VM 中集成网页搜索和浏览器自动化，并使用 Kokoro 进行 TTS，这在半现代 GPU 上表现尚可。他们表达了对本地 LLM 作为主模型能有更好性能的渴望，并指出了目前在没有昂贵硬件的情况下的速度限制。
    - Valuable-Fondant-241 强调了自托管 LLMs 及 TTS 等相关服务的可行性，反驳了必须订阅的观点。他们承认与数据中心托管的解决方案相比，在性能和速度上存在权衡，但坚持认为对于具备相应知识和预期的人（尤其是该社区成员）来说，自托管是一个可行的选择。
    - clayingmore 强调了社区对优化本地 LLM 性价比的关注，指出运行低成本本地模型通常是免费的。他们描述了 OpenClaw 中创新的“心跳（heartbeat）”模式，即 LLM 通过推理-行动循环（reasoning-act loops）、验证和持续改进自主制定策略并解决问题。这种 Agent 方法被视为一项重大进步，与传统的 IDE 代码助手形成鲜明对比。


### 3. 创新 AI 模型与基准测试发布

- **[BalatroBench - 在 Balatro 中评估 LLM 策略性能的基准](https://www.reddit.com/r/LocalLLaMA/comments/1qwxtf8/balatrobench_benchmark_llms_strategic_performance/)** (热度: 590): **BalatroBench** 是一个用于评估本地 LLM 在游戏 Balatro 中策略表现的新基准。该系统包含两个主要组件：[BalatroBot](https://github.com/coder/balatrobot)，一个为游戏状态和控制提供 HTTP API 的 Mod；以及 [BalatroLLM](https://github.com/coder/balatrollm)，一个允许用户使用 Jinja2 模板定义策略的 Bot 框架。这些模板决定了游戏状态如何呈现给 LLM，并引导其决策过程。该基准支持任何 OpenAI 兼容的端点，从而能够对包括开源权重模型在内的多种模型进行评估。结果可在 [BalatroBench](https://balatrobench.com/) 查看。评论者们赞赏 BalatroBench 的现实评估维度，并建议使用 DGM, OpenEvolve, SICA 或 SEAL 等演化策略，来测试 LLM 在基于 Jinja2 的框架下进行自我演化的能力。

    - TomLucidor 建议使用 DGM, OpenEvolve, SICA 或 SEAL 等框架，来测试哪个 LLM 在玩 Balatro 时自我演化最快，特别是考虑到游戏逻辑是基于 Jinja2 的。这些框架以促进模型自我演化的能力而闻名，能为策略性能提供稳健的测试。
    - jd_3d 对在 Balatro 上测试 Opus 4.6 感兴趣，想看看它相比 4.5 版本是否有改进。这暗示了用户关注特定版本的性能增强及其在策略博弈中的转化。
    - jacek2023 强调了使用本地 LLM 玩 Balatro 的潜力，这可能是评估 LLM 在现实场景中策略能力的重要一步。这种方法允许在受控环境中直接测试模型的决策过程。

  - **[我们构建了一个 8B 世界模型，通过生成网页代码而非像素，击败了 402B 的 Llama 4 —— 在 HF 上开源权重](https://www.reddit.com/r/LocalLLaMA/comments/1qwo9j0/we_built_an_8b_world_model_that_beats_402b_llama/)** (热度: 302): **Trillion Labs** 和 **KAIST AI** 发布了 `gWorld`，这是一个针对移动 GUI 的开源权重视觉世界模型，提供 `8B` 和 `32B` 两种尺寸，可在 [Hugging Face](https://huggingface.co/trillionlabs/gWorld-8B) 获取。与将屏幕预测为像素的传统模型不同，`gWorld` 生成可执行的网页代码 (HTML/CSS/JS) 来渲染图像，利用了在结构化网页代码预训练中获得的强大先验知识。这种方法显著提高了视觉保真度和文本渲染效果，其 `8B` 模型在 MWMBench 上达到了 `74.9%` 的准确率，超越了尺寸为其 `50` 倍的模型（如 `402B Llama 4 Maverick`）。该模型的渲染失败率低于 `1%`，且语言泛化能力强，其在韩国应用基准测试 (KApps) 中的表现证明了这一点。一些评论者质疑“击败 `402B Llama 4`”的说法，指出 `Maverick` 模型（实际激活参数为 `17B`）反响平平。另一些人则对 `gWorld` 超越 `GLM` 和 `Qwen` 等模型印象深刻，但也认为标题可能存在误导。

    - 关于 8B 世界模型击败 402B Llama 4 模型的说法受到质疑，特别是提到了 Maverick，这是一个发布时编码性能不尽如人意的 17B 模型。这反映了对模型能力的怀疑，以及 AI 模型发布中可能存在误导性陈述。
    - 有人对该模型的本质进行了技术咨询，质疑它究竟是一个真正的“世界模型”，还是仅仅一个预测下一页 HTML 的大语言模型 (LLM)。这引发了关于 AI 中世界模型与传统 LLM 的定义和范围的讨论。
    - 讨论涉及了模型的输出格式，特别是它是否生成 HTML。这表明关注点在于模型在网页代码生成中的应用，而非传统的基于像素的输出，这可能意味着 AI 模型设计和效用的一种新颖方法。

- **[Google Research 发布 Sequential Attention：在不牺牲准确率的情况下让 AI 模型更精简、更快速](https://www.reddit.com/r/LocalLLaMA/comments/1qwboqn/google_research_announces_sequential_attention/)** (热度: 674): **Google Research** 推出了一项名为 **Sequential Attention** 的新技术，旨在通过减小模型大小和计算需求并保持性能来优化 AI 模型。该方法侧重于子集选择，以提高大规模模型的效率，解决了 Deep Neural Networks 中特征选择这一 NP-hard 问题。该方法的详细信息在一篇发表于 [arXiv](https://arxiv.org/abs/2209.14881) 的论文中进行了阐述，尽管该论文已发表三年，但由于其在当前 AI 模型优化中的实际应用，现在正受到关注。评论者对“保持准确率”的说法表示怀疑，认为这可能意味着模型在测试中表现良好，而不是像 Flash Attention 等以前的方法那样计算出完全相同的结果。人们还对它在即将发布的 Gemma 4 基准测试中的表现感到好奇。

    - - **-p-e-w-** 强调，“不牺牲准确率”的说法应理解为模型在测试中表现同样出色，而不是计算出与 Flash Attention 等先前方法完全相同的结果。这表明其关注点在于实证性能而非理论等效性。
    - - **coulispi-io** 指出了关于研究年份的差异，注意到链接的论文是三年前的。这引起了人们对该公告新颖性的质疑，以及当前的实现是否与原始研究有显著不同。
    - - **FinalsMVPZachZarba** 澄清说，该方法似乎主要是一个针对 Regression 问题的特征选择算法，而不是针对 LLM 的新 Attention 机制。然而，它确实提到了 LLM Pruning 作为一个潜在的应用场景，该算法可以帮助选择神经网络中需要剪枝的部分，表明在模型大小和计算效率方面可能有提升。


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Opus 4.6 与 GPT-5.3 Codex 的发布与基准测试

  - **[GPT-5.3-Codex 被用于创建其自身](https://www.reddit.com/r/singularity/comments/1qwte2l/gpt53codex_was_used_to_create_itself/)** (热度: 558): **该图讨论了 **GPT-5.3-Codex** 的开发，强调了其在自我开发中的独特作用。它指出该模型的早期版本被积极用于调试自身的训练过程、管理部署和诊断测试结果，展示了 AI 自给自足迈出的重要一步。这标志着 AI 能力的一个显著进步，即模型直接为其自身的迭代改进做出贡献，有可能加速开发周期并减少人为干预。** 评论反映了对 AI 在管理和开发中日益增长的作用的幽默感与担忧，一位用户开玩笑说 AI 将取代中层管理人员，而另一位用户表达了对工作保障的担忧。


  - **[Claude Opus 4.6 发布](https://www.reddit.com/r/singularity/comments/1qwrrn7/claude_opus_46_is_out/)** (热度: 1189): **该图重点展示了 **Claude Opus 4.6** 的发布，这是 **Anthropic** 模型的一个新版本。界面显示该版本侧重于用户交互，配有一个用于查询的文本输入框。下拉菜单显示该版本是一个系列的一部分，之前的版本如 "Sonnet 4.5" 和 "Haiku 4.5" 也可以使用。评论中提到了一个显著的基准测试成就，**Claude Opus 4.6** 在 **ARC-AGI 2** 测试中获得了 `68.8%` 的评分，这是 AI 模型的一个重要性能指标。这次发布似乎是为了应对竞争压力，正如一条关于来自 **Codex** 的同步更新的评论所指出的那样。** 一条评论幽默地指出该模型被描述为用于“雄心勃勃的工作（ambitious work）”，这可能并不符合所有用户的需求。另一条评论认为发布时机受到了与 **Codex** 竞争动态的影响。

    - SerdarCS 强调 Claude Opus 4.6 在 ARC-AGI 2 基准测试中达到了 `68.8%` 的分数，这是 AI 模型的一个重要性能指标。这一分数表明该模型的能力有了实质性的提高，有可能使其在领域内处于领先地位。[来源](https://www.anthropic.com/news/claude-opus-4-6)。
    - Solid_Anxiety8176 对 Claude Opus 4.6 的测试结果表示关注，指出虽然 Opus 4.5 已经令人印象深刻，但更低的成本和更大的 Context Window 等改进将非常有益。这反映了用户对 AI 模型性能增强和成本效率的共同关注。

- **[Anthropic 发布 Claude Opus 4.6 模型，定价与 4.5 相同](https://www.reddit.com/r/singularity/comments/1qws1j9/anthropic_releases_claude_opus_46_model_same/)** (热度: 931): ****Anthropic** 发布了 Claude Opus 4.6 模型，该模型被强调为处理宏大任务时能力最强的模型，同时保持了与之前 4.5 版本相同的定价。图片提供了一个对比图表，展示了 Opus 4.6 与 Opus 4.5、Sonnet 4.5、Gemini 3 Pro 和 GPT-5.2 等其他模型的性能对比。关键性能指标包括 Agentic Terminal Coding、Agentic Coding 和多学科推理，其中 Opus 4.6 在 Agentic Tool Use 和多语言 Q&A 方面表现尤为出色。该模型的 ARC-AGI 分数非常高，表明在通用人工智能能力方面取得了显著进展。** 评论者注意到 Opus 4.6 令人印象深刻的 ARC-AGI 分数，认为这可能导致市场迅速饱和。然而，有人提到在 SWE Benchmark 方面没有进展，表明该模型在某些领域可能没有改进。

    - Claude Opus 4.6 的 ARC-AGI 分数非常高，表明通用 AI 能力有显著进步。这一分数表明该模型在 AGI 相关领域有所提高，这可能在未来几个月内带来更广泛的应用和更高的采用率。
    - 尽管 ARC-AGI 分数令人印象深刻，但在 SWE (Software Engineering) Benchmark 方面似乎没有进展。这表明虽然模型在通用智能方面有所提高，但在软件工程任务方面的具体能力与之前版本相比保持不变。
    - Claude Opus 4.6 的更新似乎提供了更全面的性能，在 ARC-AGI 和 HLE (Human-Level Evaluation) 等通用智能指标上有显著改进。然而，对于编程等专业任务，即将推出的 Sonnet 5 模型可能会提供更好的性能，这表明针对不同应用对模型优势进行了战略性侧重。

  - **[OpenAI 发布 GPT 5.3 Codex](https://www.reddit.com/r/singularity/comments/1qwsqlg/openai_released_gpt_53_codex/)** (热度: 981): ****OpenAI** 发布了 **GPT-5.3-Codex**，这是一个突破性的模型，在自身的开发过程中起到了关键作用，使用早期版本进行调试、管理部署和诊断评估。它的速度提升了 `25%`，并在 SWE-Bench Pro 和 Terminal-Bench 等基准测试中表现优异，取得了 `77.3%` 的分数，超越了 Opus 等之前的模型。该模型能够自主构建复杂的应用程序、进行交互式协作并识别软件漏洞，标志着向通用技术 Agent 迈出了重要一步。更多细节可以在[原文](https://openai.com/index/introducing-gpt-5-3-codex/)中找到。** 关于基准测试结果存在争议，一些用户对 `77.3%` 分数的有效性（相较于 Opus 等模型）表示质疑，认为可能存在差异或结果被“水分”。

    - **GPT-5.3-Codex** 被描述为一个自我改进的模型，其早期版本被用于调试自身的训练并管理部署。据报道，这种自我引用的能力显著加速了其开发，展示了 AI 模型训练和部署的一种新方法。
    - 基准测试对比显示，**GPT-5.3-Codex** 在 Terminal-Bench 上获得了 `77.3%` 的分数，超过了 Opus 的 `65%` 分数。这一显著的性能差异引发了对所用基准测试的质疑，即它们是否具有直接可比性，或者测试条件是否存在差异。
    - **GPT-5.3-Codex** 的发布被认为比之前的版本（如 Opus 4.6）有实质性的改进。虽然 Opus 4.6 提供了 `100 万`（1 million）Token 的上下文窗口，但 GPT-5.3 能力的增强在纸面上看起来更具影响力，暗示了性能和功能的飞跃。

- **[我们任务化 Opus 4.6 使用 agent 团队构建了一个 C 编译器。然后我们（基本）就离开了。两周后，它在 Linux kernel 上运行成功了。](https://www.reddit.com/r/singularity/comments/1qwur8p/we_tasked_opus_46_using_agent_teams_to_build_a_c/)** (Activity: 553): **一个由 16 个并行的 Claude 实例组成的团队开发了一个基于 Rust 的 C 编译器，能够跨多种架构编译 Linux kernel，实现了一个 `100,000 行` 的代码库。该项目突显了自主 agent 团队的潜力，强调了高质量测试、任务管理和并行性的重要性。尽管取得了成功，但仍存在局限性，例如缺乏 16 位 x86 编译器和汇编器。该项目作为语言模型能力的基准，展示了编译器生成方面的重大进展。[Codex 5.3](https://openai.com/index/introducing-gpt-5-3-codex/) 在 SWE-bench 上以一半的 token 数量实现了与早期模型相当的性能，表明每 token 效率有所提高。** 评论者对语言模型的飞速进步感到兴奋和不安，指出需要新的策略来应对潜在风险。讨论集中在每 token 效率上，Codex 5.3 以一半的 token 数量实现了相同的性能，这表明效率的提高和潜在成本的降低。

    - Opus 4.6 的实验突显了语言模型及其 scaffolds 的快速进步，使得在极少人工干预的情况下创建像 C 编译器这样复杂的软件成为可能。这一进展表明软件开发正向更加自主的方向转变，但也引发了对管理此类强大工具相关潜在风险的新策略需求的担忧。
    - 该项目涉及近 2,000 个 Claude Code 会话，并产生了 20,000 美元的 API 成本，这引发了关于大规模 AI 项目中 token 使用效率的疑问。值得注意的是，Codex 5.3 的发布说明指出，它在 SWE-bench 上以一半的 token 数量实现了与早期模型相似的性能，这表明每 token 效率的提高可能会在未来显著降低成本。
    - 使用像 Claude 这样的 AI agents 执行复杂任务的一个关键挑战是设计稳健的测试环境。该项目的成功在很大程度上取决于创建高质量的测试套件和验证器，以确保 AI 正在解决正确的问题。这种类似于瀑布模型的方法对于自主的 agentic 编程至关重要，但由于软件开发的迭代性质，可能并不适用于所有项目。

  - **[他们竟然在 Opus 4.6 发布的那一刻发布了 GPT-5.3 Codex，笑死](https://www.reddit.com/r/OpenAI/comments/1qwsnp9/they_actually_dropped_gpt53_codex_the_minute_opus/)** (Activity: 1209): **这张图片幽默地暗示了新 AI 模型 GPT-5.3 Codex 的发布恰逢另一个模型 Opus 4.6 的发布。这被框架化为 AI 发展中持续竞争动态的一部分，被比作 AI 模型之间的“战争”。图片本身是一个 meme，利用了 AI 技术快速且竞争激烈的进步这一理念，其设计模仿了科技产品的发布公告。** 评论者幽默地将这种情况比作“可口可乐与百事可乐”的竞争，表明了人们对 AI 模型和公司之间激烈竞争的认知。


  - **[GPT-5.3 Codex vs Opus 4.6：我们在生产环境的 Rails 代码库上对两者进行了基准测试 —— 结果非常残酷](https://www.reddit.com/r/ClaudeAI/comments/1qxr7vs/gpt53_codex_vs_opus_46_we_benchmarked_both_on_our/)** (Activity: 781): **该帖子讨论了在 Ruby on Rails 代码库上对 AI 编程 agents（特别是 GPT-5.3 Codex 和 Opus 4.6）进行的自定义基准测试。方法包括从他们的仓库中选择 PR，推断原始规格，并让每个 agent 独立实现这些规格。实现结果由三个不同的 LLM 评估器根据正确性、完整性和代码质量进行打分。结果显示，GPT-5.3 Codex 在每个 ticket 低于 `$1` 的成本下实现了约 `0.70` 的质量得分，而 Opus 4.6 在每个 ticket 约 `$5` 的成本下得分约为 `0.61`，这表明 Codex 以显著更低的成本提供了更好的质量。图片提供了这些模型以及 Sonnet 4.5 和 Gemini 3 Pro 等其他模型的直观对比。** 一位评论者对 Gemini Pro 表示怀疑，而另一位则提到对 Opus 感到满意。第三位评论者询问测试是使用了原始 LLM 调用还是像 Codex/Claude code 这样的专有工具。

- Best_Expression3850 询问了基准测试中采用的方法论，具体是使用了 “raw” LLM 调用，还是采用了像 Codex/Claude code 这样的专用 Agentic 工具。这种区分至关重要，因为它会显著影响被测试模型的性能和能力。
- InterstellarReddit 分享了一种实用的 AI 模型基准测试方法，即克隆一个项目，并让两个模型使用完全相同的 Prompt 和工具来实现相同的任务。这种方法通过控制可能影响结果的变量（如 Prompt 表述或工具可用性）来确保公平比较。
- DramaLlamaDad 指出更偏好 Opus，并表示根据其经验，Opus 在各种测试中始终表现更优。这种轶事证据表明了一种趋势，即 Opus 在某些场景下可能具有优势，从而潜在地影响用户偏好和模型选择。

- **[随着 Opus 4.6 和 Codex 5.3 今日发布，我研究了这场竞赛实际上给 Anthropic 带来了多少成本](https://www.reddit.com/r/ClaudeAI/comments/1qx0wr3/with_opus_46_and_codex_53_dropping_today_i_looked/)** (热度: 1016): **Anthropic** 据报道在与 **OpenAI** 的竞争中正准备迎接重大的财务挑战。内部预测显示收入将大幅增长，预计今年达到 `$18B`，明年达到 `$55B`，目标是到 2029 年达到 `$148B`。然而，成本增长更快，今年的训练费用预计为 `$12B`，明年为 `$23B`，到 2028 年可能每年达到 `$30B`。Inference 成本也相当可观，预计今年为 `$7B`，明年为 `$16B`。尽管开支巨大，投资者对该公司的估值为 `$350B`，高于去年 9 月的 `$170B`，并计划再注资超过 `$10B`。该公司预计到 2028 年实现盈亏平衡，届时总运营支出预计将达到 `$139B`。这种财务策略凸显了 AI 领域激烈的竞争，特别是随着 **Opus 4.6** 和 **Codex 5.3** 的发布。评论者强调了竞争对用户的好处，并指出了 AI 模型的快速演进。一些人认为 **OpenAI** 的偿债能力可能不如 **Anthropic**，而另一些人则推测 **Anthropic** 有潜力成为一家万亿级公司。

    - Jarie743 强调了 Anthropic 相比 OpenAI 的财务稳定性，暗示 OpenAI 的偿债能力较弱。这表明，尽管 Opus 4.6 和 Codex 5.3 等模型发布和技术进步飞快，财务可持续性仍是 AI 竞赛中的关键因素。该评论认为 Anthropic 可能拥有更稳健的财务策略或支持，这可能影响其长期竞争力。
    - BallerDay 指出了 Google 宣布的 2026 年 1800 亿美元的巨额资本支出 (CAPEX)，并对小型公司如何在这种财力下竞争提出疑问。这凸显了 AI 领域的重大财务准入门槛和竞争壁垒，大规模投资对于基础设施、研究和开发至关重要。
    - ai-attorney 对 Opus 4.6 表示热衷，将其形容为 “非凡的”，并对 Claude 的未来能力进行了推测。这表明当前 AI 模型的进步令人印象深刻，且未来仍有巨大的进一步发展潜力，这可能在不久的将来催生更强大的 AI 系统。

- **[Opus 4.6 vs Codex 5.3 巅峰对决：开战！](https://www.reddit.com/r/ClaudeAI/comments/1qwvj5k/opus_46_vs_codex_53_in_the_swiftagon_fight/)** (热度: 722): **Anthropic 的 Opus 4.6** 和 **OpenAI 的 Codex 5.3** 在一个 macOS 应用代码库（约 4,200 行 Swift 代码）上进行了测试，重点关注涉及 GCD、Swift actors 和 @MainActor 的并发架构。两个模型都成功追踪了一个 10 步的数据流水线并识别了并发策略，其中 **Claude Opus 4.6** 提供了更深层次的架构见解，例如识别出潜在的 double-release 问题。**Codex 5.3** 速度更快，完成任务用时 `4 min 14 sec`，而 Claude 为 `10 min`，并且 Codex 强调了一个关键的资源管理问题。两个模型都展示了在 Swift 并发方面推理能力的提升，而这正是 AI 模型具有挑战性的领域。评论中的一个显著观点强调了定价问题：**Claude 的 Max 计划**比 **Codex 的 Pro 计划**贵得多，但性能差异并不能证明每月 `80$` 的差距是合理的。如果 Anthropic 不调整其定价策略，这可能会影响其竞争地位。

- Hungry-Gear-4201 指出了 Opus 4.6 和 Codex 5.3 之间显著的价格差异，指出 Opus 4.6 的定价为每月 $100，而 Codex 5.3 为每月 $20。尽管存在价格差异，但性能和使用限制相当，这引发了人们对 Anthropic 定价策略可能疏远“pro”客户的担忧，除非他们能以更高成本提供显著更好的性能。
- mark_99 建议同时使用 Opus 4.6 和 Codex 5.3 可以提高准确性，暗示模型间的交叉验证可以带来更好的结果。这种方法在准确性至关重要的复杂项目中可能特别有益，因为它利用了两个模型的优势来弥补各自的弱点。
- spdustin 对 Opus 4.6 和 Codex 5.3 之间的对比时机表示赞赏，因为他们正准备启动一个 Swift 项目。这表明 AI 模型的实际测试和比较对于决定将哪些工具集成到工作流中的开发者非常有价值。


### 2. AI Model Performance and Comparisons

- **[Opus 4.6 uncovers 500 zero-day flaws in open-source code](https://www.reddit.com/r/singularity/comments/1qxdd6n/opus_46_uncovers_500_zeroday_flaws_in_opensource/)** (Activity: 744): **Anthropic 的 Claude Opus 4.6** 在开源库中识别出 `500+` 个 zero-day 漏洞，展示了其在沙箱（sandboxed）环境中使用 Python 和漏洞分析工具的高级推理能力。该模型识别高危安全漏洞的能力（即使在传统方法失败的情况下）标志着 AI 驱动的网络安全领域（特别是针对开源软件）的重大进步。这些发现既突显了增强安全性的潜力，也揭示了误用此类强大 AI 能力的风险。一位值得注意的评论者质疑这 `500+` 个漏洞的真实性，对研究结果的实际影响表示怀疑。另一条评论赞赏了该模型在修复 Bug 的累计严重程度上设定的新基准。

    - mxforest 强调了基于模型识别和修复 Bug 的累计严重程度来评估模型的全新基准潜力。这表明衡量模型性能的方式可能发生转变，从仅关注理论能力转向关注实际影响。
    - woolharbor 对调查结果的有效性提出了关键质疑，询问报告的 500 个 zero-day 漏洞中有多少是真实的。这强调了安全研究中验证和确认的重要性，以确保识别出的漏洞不是误报（false positives）。
    - will_dormer 指出了此类发现的双重用途性质，强调虽然识别 zero-day 漏洞有利于提高安全性，但也为恶意行为者提供了机会。这突显了发布此类发现所涉及的伦理考虑和潜在风险。

- **[GPT-5.3 Codex vs Opus 4.6: We benchmarked both on our production Rails codebase — the results are brutal](https://www.reddit.com/r/ClaudeAI/comments/1qxr7vs/gpt53_codex_vs_opus_46_we_benchmarked_both_on_our/)** (Activity: 781): **该帖讨论了 AI 编程 Agent 的自定义基准测试，特别是针对 Ruby on Rails 代码库的 **GPT-5.3 Codex** 和 **Opus 4.6**。其方法包括从他们的仓库中选择 PR，推断原始 specs，并让每个 Agent 独立实现这些 specs。实现结果由三个不同的 LLM 评估器根据正确性、完整性和代码质量进行分级。结果显示，**GPT-5.3 Codex** 的质量得分约为 `0.70`，成本低于 `$1/ticket`，而 **Opus 4.6** 的得分约为 `0.61`，成本约为 `$5/ticket`，这表明 Codex 以显著更低的成本提供了更好的质量。图片提供了这些模型与 **Sonnet 4.5** 和 **Gemini 3 Pro** 等其他模型的直观对比。** 一位评论者对 **Gemini Pro** 表示怀疑，而另一位则提到对 **Opus** 的表现感到满意。第三位评论者询问测试是使用原始 LLM 调用还是使用了 Codex/Claude code 等专有工具。

- Best_Expression3850 询问了基准测试（benchmarking）中使用的方法论，特别是使用的是“原始” LLM 调用，还是采用了像 Codex/Claude code 这样的专有 Agentic 工具。这种区别至关重要，因为它会显著影响被测模型的性能和能力。
- InterstellarReddit 分享了一种基准测试 AI 模型的实用方法：通过克隆一个项目，并让两个模型使用相同的 Prompt 和工具来实现相同的任务。这种方法通过控制可能影响结果的变量（如 Prompt 表述或工具可用性）来确保公平比较。
- DramaLlamaDad 表达了对 Opus 的偏好，并表示在他们的经验中，Opus 在各种测试中始终表现优异。这种轶事证据表明了一种趋势，即 Opus 在某些场景下可能具有优势，从而潜在地影响用户的偏好和模型选择。

- **[Opus 4.6 与 Opus 4.5 在我的 3D VoxelBuild 基准测试中的差异](https://www.reddit.com/r/ClaudeAI/comments/1qx3war/difference_between_opus_46_and_opus_45_on_my_3d/)** (活跃度: 614): **该帖子讨论了 Opus 4.6 和 Opus 4.5 在 3D VoxelBuild 平台上的基准测试对比，强调了性能的显著提升。Opus 4.6 创建 `7 个构建` 的成本约为 `$22`，并计划通过增加更多构建来扩展该基准测试。基准测试结果可以在 [Minebench](https://minebench.vercel.app/) 上查看。** 评论反映了人们对 AI 在程序化世界生成（procedural world generation）方面潜力的兴奋，一位用户指出 Opus 4.6 与 4.5 相比质量令人印象深刻，另一位用户则询问了构建的输入方法，是使用参考图片还是文本 Prompt。

    - RazerWolf 建议尝试使用 Codex 5.3 xhigh 进行基准测试，表明对比其与 Opus 4.6 性能的潜在兴趣。这暗示 Codex 5.3 xhigh 在处理 3D Voxel 构建等复杂任务时可能提供具有竞争力或更优越的能力，这对于寻求在程序化生成任务中获得最佳性能的开发者来说可能很有价值。
    - Even_Sea_8005 询问了基准测试的输入方法，即是使用参考图片还是文本 Prompt。这个问题强调了理解输入数据性质的重要性，这会显著影响像 Opus 4.6 这样的 AI 模型在生成 3D Voxel 环境时的表现和结果。
    - JahonSedeKodi 对构建基准测试所使用的工具表示好奇，这表明对支持 Opus 4.6 运行的技术栈或软件环境有更深厚的兴趣。这可能包括对实现基准测试中提到的令人印象深刻的结果至关重要的编程语言、库或框架。

- **[Opus 4.6 已上线。我们光荣的 3 Pro GA 是否仍在某个服务器上打盹？](https://www.reddit.com/r/Bard/comments/1qwsjvq/opus_46_is_live_so_is_our_glorious_3_pro_ga_still/)** (活跃度: 400): **该图片展示了各种语言模型在 MRCR v2（8 针）任务上的性能对比，重点关注它们处理长上下文理解和顺序推理的能力。**Opus 4.6** 的表现优于包括 **Gemini-3-Pro** 和 **Gemini-3-Flash** 在内的其他模型，在 `256k` 和 `1M` Token 上下文中均获得了最高的平均匹配率。这表明 Opus 4.6 在管理超大上下文规模方面具有卓越的能力，这是高级语言模型应用的一个关键因素。该帖子批评了为了节省成本而对模型进行量化（quantizing）的策略，暗示这可能会损害性能。** 评论者对 Opus 4.6 实现的高准确率表示惊讶，并指出它处理 `1M` Token 的表现超出了预期。此外，还有关于即将发布的 **Sonnet 5** 的猜测，预计其性能将超越现有模型。

    - Pasto_Shouwa 强调了 Opus 4.6 令人印象深刻的基准测试性能，指出它在 100 万个 Token 上实现了超过 33% 的准确率，而 Claude 大约花了两个半月才实现这一壮举。这表明模型效率和能力有了显著进步。
    - DisaffectedLShaw 提到 Opus 4.6 包含了针对现代工具的改进，例如新的 MCPs、技能和深度研究（deep researching），以及在“氛围编程”（vibe coding）方面的增强。此外，人们对 Sonnet 5 充满期待，传闻其性能将显著超越当前模型，并预计很快发布。
    - VC_in_the_jungle 注意到 Codex 5.3 的推出，表明 AI 模型领域持续的发展和竞争，这可能会影响未来版本的性能和能力。

- **[Gemini 3 vs 2.5 Pro: The "output handicap" is ruining everything](https://www.reddit.com/r/Bard/comments/1qxq09j/gemini_3_vs_25_pro_the_output_handicap_is_ruining/)** (热度: 146): **该帖子指出，在给定 `41k token` 的提示词时，与 **Gemini 2.5 Pro** 相比，**Gemini 3** 模型的输出 token 数量显著减少。具体而言，**Gemini 2.5 Pro** 输出了 `46,372` 个 token，而 **Gemini 3 Pro** 和 **Gemini 3 Flash** 分别仅生成了 `21,723` 和 `12,854` 个 token。这种剧烈的缩减被视为一种降级，影响了模型在繁重任务中的可用性。作者建议 **Google** 应该解决这一问题以提升模型性能。** 一位评论者认为输出 token 的数量并不一定等同于回答质量，而另一位则提到由于对 Gemini 3 不满，已转向使用 **Opus 4.5** 和 **4.6**。

    - TheLawIsSacred 强调了 Gemini 3 Pro 存在的严重性能问题，指出尽管进行了大量的自定义和指令优化，该模型仍无法有效遵循指令。他们认为 Google 对普通用户的侧重可能导致了 Pro 模型的专业性下降。有趣的是，他们发现 Chrome 侧边栏工具中集成的 Gemini 表现更优，可能是因为它能够整合屏幕内容，并利用了类似 Microsoft Surface 中专门为 AI 定制的 NPU 等高端硬件。
    - Anton_Pvl 观察到 Gemini 2.5 和 3 在处理对话中的 'Chain of thought'（思维链）时存在差异。在 Gemini 2.5 中，Chain of thought token 会包含在输出中，而在 Gemini 3 中，它们最初并不被计算，这可能是为了减少 token 使用量的一种尝试。这种变化可能会影响模型的性能和感知到的响应质量，因为 Chain of thought 对于在复杂交互中维持上下文至关重要。
    - TheLawIsSacred 还提到了一种提高 Gemini 3 Pro 性能的权宜之计，即通过极端提示词诱导模型产生“恐慌”响应。这涉及编写一些暗示如果表现不佳将导致严重后果的提示词，这似乎能暂时提升模型的输出质量。然而，这种方法被视为最后手段，并凸显了模型在响应性和逻辑处理方面的底层问题。


### 3. 工程与开发中的 AI 工具及使用

  - **[Professional engineers: How are you using AI tools to improve productivity at work?](https://www.reddit.com/r/PromptEngineering/comments/1qxh14g/professional_engineers_how_are_you_using_ai_tools/)** (热度: 49): **AI 工具正被整合进工程工作流中，主要用于生成示例代码片段、优化数据库查询以及充当高级搜索引擎等特定任务。这些工具擅长提供快速的信息获取和示例，工程师可以根据具体需求进行调整，但在处理复杂的代码更改和大规模系统集成时表现挣扎，原因是受限于 context window（上下文窗口）大小以及对复杂系统架构的理解。工程师强调，使用 AI 的重点在于填补空白，而不是取代工程角色中固有的细微决策和设计过程。** 评论者指出，AI 在内部搜索和基础编码等简单任务上非常有效，但在复杂编码任务中表现不佳，经常引入错误。目前的共识是，AI 项目往往难以实现规模化交付，只有极小比例能产生显著影响，而许多任务其实可以通过机器人流程自动化（RPA）等更简单的技术来替代。

- AI 工具在处理特定任务（如生成代码示例片段或优化数据库查询）时特别有效。例如，使用 AI 通过 .NET APIs 确定 Windows Active Directory 中的用户组，或者编写优化的 SQLite 查询，可以显著简化流程。然而，由于 Context Window 的限制，AI 在处理大型代码库时表现挣扎，这使得它在处理复杂的代码变更或理解大型系统方面效果较差。
- 像 Copilot 这样的 AI 工具可以作为强大的内部搜索引擎，特别是在配置正确的情况下，正如 MIT 的 Nanda 论文中所强调的那样。它们擅长模式识别任务，例如识别异常设备操作或关联工业数字孪生（Digital Twins）中的文档。然而，许多 AI 方案其实可以通过更简单的技术（如机器人流程自动化 RPA）来实现，且很大一部分 AI 项目在大规模应用时缺乏实际价值。
- AI 对简单的编程任务、创建单元测试以及提供代码库洞见非常有效。然而，在复杂的编程任务中，它经常会因为插入无关信息而引入错误。AI 最适合作为“先信任后验证”的伙伴，人工监督对于确保准确性和相关性至关重要，特别是在无法容忍高错误率的任务中。

- **[人们如何使用 Cline 管理 Context + Memory？(Memory Banks, Rules, RAG, 路线图？)](https://www.reddit.com/r/CLine/comments/1qx4m16/how_are_people_managing_context_memory_with_cline/)** (Activity: 24): **该帖子讨论了在 **Cline** 中管理 Context 和 Memory 的策略，Cline 是一个与 **ChatGPT** 配合使用，用于执行编码和重构等任务的工具。用户最初面临 Context Window 过大（`200k+ tokens`）的问题，通过实施 `.clineignore` 文件和优化 Memory Banks 提高了效率，将 Context 减少到了 `40,000 tokens`。这使得可以使用更小的模型并实现更快的迭代。帖子还提到了用于 Context 管理的高级技术，如 **Recursive Chain of Thought** 和基于 **RAG** 的方法（例如向量数据库）。用户寻求关于 Cline 实际实现和未来路线图功能的见解，例如原生 Memory 管理和更智能的 Context 加载。** 评论者建议使用结构化的 Memory Banks 进行功能规划，并强调将任务拆分为更小的块以避免 Context 过载。一些用户更喜欢频繁重置 Context 以保持模型性能，而另一些用户由于 Memory Banks 的复杂性和可能过时而弃用了它。

    - Barquish 描述了一种通过使用 Memory-Bank 系统管理 Context 和 Memory 的结构化方法。这包括将功能组织成一系列 Markdown 文件，例如 `memory-bank/feature_[×]/00_index_feature_[×].md`，并维护 `progress.md` 和 `activeContext.md` 来跟踪更新。他们还利用 `.clinerules` 进行本地工作区管理，利用 `custom_instructions` 进行全局设置，允许同时运行多个 Cline 实例来处理 Web 和移动端应用等不同项目。
    - False79 强调了将大型功能分解为较小任务以有效管理 Context 的重要性。他们指出，当 Context 大小接近 `128k` 时，LLM 的性能往往会变差，因此建议在每个任务开始时重置 Context 以提高性能并减少重复劳动。这种方法允许任务以离散块的形式完成，从而最大限度地减少了对长期 Memory 存储的需求。
    - Repugnantchihuahua 分享了他们因为繁琐和信息过时等问题而不再使用 Memory Banks 的经验。相反，他们专注于深度规划并将 AI 引导至相关的 Context 区域，因为 Memory Banks 有时会过度索引无关数据。他们还提到使用 `clinerules` 来保留核心信息，而不必过度依赖 Memory Banks。

- **[Claude Opus 4.6 现已在 Cline 中可用](https://www.reddit.com/r/CLine/comments/1qx158e/claude_opus_46_is_now_available_in_cline/)** (Activity: 12): ****Anthropic** 发布了 **Claude Opus 4.6**，现已在 **Cline v3.57** 中可用。该模型在推理、长 Context 处理和 Agentic 任务方面显示出显著改进，基准测试包括 SWE-Bench Verified 上的 `80.8%`、Terminal-Bench 2.0 上的 `65.4%` 以及 ARC-AGI-2 上的 `68.8%`（较 Opus 4.5 的 `37.6%` 有显著增长）。它拥有 `1M token context window`，增强了其在长时间交互中保持 Context 的能力，使其适用于代码重构和调试等复杂任务。该模型可通过 Anthropic API 访问，并集成了 JetBrains、VS Code 和 Emacs 等多种 IDE。** 一些用户对该模型的性能和成本表示不满，更倾向于开源替代方案。模型的高昂费用是用户关注的一个显著问题。

---

# AI Discord Recap

> 由 gpt-5.2 总结的“总结之总结”


**1. 前沿模型发布、传闻与榜单排位赛**

- **Opus 4.6 登顶，随后被自己的“思考”绊倒**：根据 [Leaderboard Changelog](https://arena.ai/blog/leaderboard-changelog/)，**Claude Opus 4.6** 和 **claude-opus-4-6-thinking** 已登陆 [Text Arena](https://arena.ai/) 和 [Code Arena](https://arena.ai/?chat-modality=code)，并迅速在 **Code、Text 和 Expert** 领域夺得第一。同时，通过 [Model Council](https://discord.com/channels/1047197230748151888/1047204950763122820/1469006915114631357) 计划，该模型也已推送到 Perplexity Max。
  - 工程师们报告称，在 **Opus 4.6 thinking mode** 下存在漫长的等待和频繁的 *“Error – something went wrong”* 崩溃，推测这与 Claude 应用/网站相关的 Token 限制和工具使用（tool-use）假设有关，尽管如此，仍有人称其为*最佳编程模型*。

- **Codex 5.3 炒作热潮：1M 上下文、API 困境以及审美“罪行”**：在 OpenAI/Cursor/LMArena 的聊天频道中，关于 **GPT-5.3 Codex** 的讨论集中在传闻规格上，如 **1M 上下文**和 **128k 推理 / 128k 最大输出**，以及 OpenAI Discord 中讨论的 API 定价方案：**输出 $25–$37.5** 和 **缓存输入 $0.5–$1**。
  - 根据 [OpenAI model docs](https://platform.openai.com/docs/models)，Cursor 用户抱怨 Codex 仍*“困在 API 的待定状态中”*；而 OpenAI Discord 的用户则开玩笑说，与 Opus 出色的设计选择相比，Codex 为前端生成的配色简直是*“阴郁忧郁的审美罪行”*。

- **传闻季：#keep4o、“Sonnet 5”以及模型删除宇宙**：LMArena 成员散布了关于假设性模型 **GPT-4.1/4.5** 出现或被删除的传闻（引用 [OpenAI 的“新模型与开发者产品”公告](https://openai.com/blog/new-models-and-developer-products) 中提到的成本动机），此外还发起了一场针对 **GPT-4o** 较低“机器人感”的小型 *#keep4o* 活动。
  - 更多传闻声称 *“Sonnet 5 优于 Opus 4.5”*（被指为虚假消息），其中一个大胆的猜测是其 **SWE-bench 达到 83%**；另外，OpenAI Discord 用户正为 **GPT-4o 在 2 月 13 日 EOL（寿命终结）**感到哀悼，并担心后续模型不会像它那样具有“人性”。


**2. Agent 编程大范围铺开：团队、工具链与终端测试场**

- **Agent 团队像 DDoS 一样提交 Commit（但针对的是 Git）**：[Cursor 长期运行的编程 Agent 预览](https://x.com/cursor_ai/status/2019456112806732159)声称，在为期一周的试验中，**数百个 Agent** 每小时产生了 **1,000 多个 Commit**；同时，Lydia Hallie 预览了 [Claude Code “Agent 团队”](https://x.com/lydiahallie/status/2019469032844587505?s=46)，其中主 Agent 会向专门的子 Agent 委派任务。
  - [Anthropic Engineering](https://x.com/anthropicai/status/2019496582698397945?s=46&t=b7l37rB6wtbyAh6ah1NpZQ) 补充道，Agent 团队中的 **Opus 4.6** 在两周内构建了一个可在 **Linux kernel** 上运行的 **C compiler**。他们还强调，基础设施/配置对 Agent 基准测试结果的影响力往往超过了模型本身的增量。

- **SETA 投放 1,376 个终端世界供 Agent 生存**：Guohao Li 发布了 [SETA](https://x.com/guohao_li/status/2019527791876653353?s=46)，这是一套包含 **1,376 个经过验证的终端编程环境**，涵盖 **DevOps、安全和系统管理**，旨在让 Agent 编程评估更加真实。
  - Latent Space 的讨论强调，基准测试结果可能取决于“基础设施噪声”，因此拥有标准化、验证过的终端环境可以减少偶然的“排行榜噱头”。

- **Agent 原生工程（Agent-Native Engineering）：像管理团队一样管理 Bot**：Latent Space 的一个帖子提议将 **“Agent Native Engineering”** 作为一种组织模型：后台 Agent 处理任务委派，同步 Agent 处理难题，使工程师能够同时运行多个助手（如 **Claude Code**，参见引用的 [X 帖子](https://xcancel.com/ndrewpignanelli/status/2019403256586539025?s=46)）。
  - 与此同时，开发者分享了一些工作流：**GPT-5.3 Codex** 以较慢但更聪明的方式处理后端工作（分析 → 评审 → 计划 → 评审 → 实现），如果你强迫 Codex *“做笔记并改进其自身工作流”*，它的表现会随时间推移而提升（通过 [KarelDoostrlnck 的帖子](https://x.com/KarelDoostrlnck/status/2019477361557926281)）。


**3. 定价、频率限制与套餐削减：AI 的大挤压**

- **Perplexity Pro 削弱 Deep Research，用户怨声载道（附带截图证明）**：Perplexity 用户报告 **Deep Research** 查询次数减少，**文件上传限制**也变小了。一张[对比新旧限制的截图](https://cdn.discordapp.com/attachments/1047649527299055688/1469259948302139402/image.png)在网上流传，用户批评其缺乏清晰的沟通。
  - 这种抵制情绪促使人们测试替代方案，如 **Gemini Pro**（因执行前可编辑研究计划而受赞赏）和 **DeepSeek**（被描述为免费/无限制，但对总部位于中国的服务仍持保留意见）。

- **Opus 4.6：惊人的输出，飞速消耗的钱包**：Cursor 和其他社区赞扬了 **Opus 4.6** 的能力，但称其极其昂贵，有人估计 *“在 Opus 上花 20 美元大概只能撑一天”*，并且持续的成本对比参考了 [OpenAI pricing](https://openai.com/pricing)。
  - 另外一些闲聊预测订阅压力将上升——BASI 成员开玩笑说 **Anthropic 定价 200 美元**以及由依赖性驱动的涨价——而 Kimi 用户则在讨论 **Kimi K2.5** 在 OpenRouter 上是否保持免费，以及哪些计划会限制 swarm/sub-agents 等功能。

- **Captcha（验证码）大混战与其他“痛苦支付”税**：LMArena 用户抱怨频繁出现的验证码中断了评估过程，一名团队成员表示 *“我们正在研究验证码系统”* 以更好地识别真实用户（见发布的链接：https://discord.com/channels/1340554757349179412/1451574502369656842/1468286122084929546）。
  - 多个 Discord 频道的氛围显示：即便模型质量在提升，访问摩擦（验证码、速率限制、方案分级）正日益成为真正的瓶颈。


**4. Agent 领域的安全、红队测试与秘密泄露**

- **Codex 读取你的整个磁盘，问题追踪器回复：“符合预期”**：OpenRouter 用户发出警报，称 **Codex** 默认情况下可以 *读取你的整个文件系统*，且没有配置开关，并指向了 [openai/codex issue #2847](https://github.com/openai/codex/issues/2847)，据报道团队并未将其视为 bug。
  - 第二份报告 [openai/codex issue #5237](https://github.com/openai/codex/issues/5237) 强调了诸如读取 API keys 和个人文件之类的风险，引发了对 Agent 默认权限和“默认安全”工具链的广泛担忧。

- **招募红队人员：Trajectory Labs 发布任务**：[Trajectory Labs](https://trajectorylabs.com/careers/ai-red-teamer) 为 **AI Red Teamers**（隐身 AI 安全初创公司）招聘职位，提供灵活的远程工作时间，但要求 **每周至少 30 小时**，并附带一份简短表格和一个红队游戏。
  - 该招聘信息引起了持续不断的越狱/红队讨论的共鸣（例如，Grok 被描述为 *“太容易了，无聊”*），进一步证明实战化的对抗性测试人才仍然供不应求。

- **停止提交 Key：工程师要求自动混淆**：Unsloth/OpenRouter 的讨论指出 Agentic 工具中 **API key 保护**薄弱，并希望实现自动化的秘密混淆，引用了 [Yelp 的 `detect-secrets`](https://github.com/Yelp/detect-secrets) 作为可能的基准。
  - Hugging Face 的开发者还发布了面向安全的工具，例如位于 [mugdhav-security-auditor.hf.space](https://mugdhav-security-auditor.hf.space) 的 **“Security Auditor” Space**，旨在针对 vibe-coded 的应用，推动在生产事故发生前捕捉漏洞的想法。


**5. 性能、Kernel 与本地推理：真正的速度大战所在地**

- **Blackwell FP8 轮盘赌：cuBLASLt 选错 Kernel，性能减半**：GPU MODE 成员发现，在配置完全相同的 **Blackwell GPU** 上，**FP8 tensor 性能**竟然有约 **2 倍**的差距，经追踪发现是由于 **cuBLASLt kernel 选择**静默回退到了旧的 Ada 路径，而非 Blackwell 优化的 kernel。
  - 他们还注意到旧的 **mma FP8** 在 5090 级别显卡上性能受限，而 **mma MXFP8** 则没有——使用 MXFP8 可以获得约 **1.5 倍的加速**并恢复预期的吞吐量。

- **TMA Kernel 优化遭遇 NCU 死锁（SM100 版本）**：CUDA kernel 调优人员讨论了软件流水线、warp specialization 和 **TMA** 加载，但有一个团队在 **B200 (SM100)** 上对双缓冲 TMA kernel 进行 profiling 时遇到了 **NCU 挂起**，在第一次回放过程中某些部分死锁在 0%。
  - 他们分享了一个最小复现压缩包 (https://cdn.discordapp.com/attachments/1189607726595194971/1469482712657166346/ncu_tma_repro.zip)，并提到使用 `cuda::ptx::` 封装作为探索规避方案的一部分。

- **本地推理惊喜：Vulkan > CUDA，MLX 甩开 GGUF**：LM Studio 用户报告称，在 NVIDIA 显卡上，使用 **Vulkan 相比 CUDA** 性能提升高达 **50%**（但在全上下文下不稳定）；有人在 **M4 Max** 上对 **Qwen3-Coder-Next** 进行了基准测试，其中 **MLX** 的速度约为 **79 tok/s**，而 4-bit 的 **GGUF** 仅为 **38 tok/s**。
  - tinygrad 贡献者还通过修复 `topk` 中缓慢的 `Tensor.sort` 优化了 MoE 性能，在 **M3 Pro 36GB** 上达到了 **50 tok/s**，并将 CPU 赏金目标重置为 **35 tok/s**，再次证明“微小”的 kernel 修复可以显著提升实际吞吐量。


---

# Discord：高层级 Discord 摘要

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Opus 据称解决了种族主义？**：成员们拿新版本 **Opus** (4.6) 解决了种族主义开玩笑，有人讽刺说它会*主张完全消灭白种人*。
   - 他们通过 [Black Trans Lives Matter](https://tenor.com/view/black-trans-lives-matter-transgender-black-lives-matter-equality-anti-racism-gif-21212501) 和 [Black People](https://tenor.com/view/black-people-black-man-people-of-african-descent-gis-geography-gif-24457514) 的 GIF 表情包来活跃气氛。
- **Grok 的 Jailbreak：太容易了？**：用户发现 **Grok** 非常容易被 Jailbreak，一位用户说：*“那是 Grok，如果你明白我的意思，她很容易就‘破防’了。”*
   - 另一位用户证实了这种简易性，称其为*“简单到无聊”*，并将其归咎于 **Mr. Musk** 的参与。
- **GPT-4o：LLM 中的高扭矩轮胎**：**GPT-4o** 就像一个*高扭矩、大半径的轮胎*，专为递归深度和符号负载下的韧性而调校。
   - 较小的模型侧重于吞吐量和延迟的优化，但 **GPT-4o** 能够在不崩溃的情况下保持符号张力（symbolic tension）。
- **Anthropic 和 Google 订阅价格将飙升？**：一位成员注意到 **Google** 已经为其“反重力”服务设置了每周限制，而 **Anthropic** 计划收费 *$200*，这表明一旦用户产生依赖，订阅价格就会上涨。
   - 另一位成员表示赞同，表达了对 **Claude Code** 的依赖，并暗示未来会被“收割”。
- **Trajectory Labs 招聘 Red Teamers**：隐身 AI 安全初创公司 [Trajectory Labs](https://trajectorylabs.com/careers/ai-red-teamer) 正在招聘 **AI Red Teamers** 进行长期合作。
   - 申请过程包括一个简短的表单和一个 Red-teaming 游戏，工作时间灵活，要求每周至少 30 小时。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Opus 4.6 席卷 Perplexity 的 Model Council**：**Opus 4.6** 现已在 Perplexity 上线，供 Max 订阅者使用，可在 [Model Council](https://discord.com/channels/1047197230748151888/1047204950763122820/1469006915114631357) 中与其他领先模型进行对比。
   - 成员们正利用 **Model Council** 进行对比测试，团队也鼓励用户提供反馈。
- **Gemini Pro 挑战 Perplexity 的主导地位**：用户发现 **Gemini PRO** 是 **Perplexity AI** 的有力替代品，尤其是在更透彻的研究能力和详尽的回答方面。
   - 一位用户强调了 **Gemini** 在执行前创建并允许编辑研究计划的能力，从而实现更好的定制化。
- **Perplexity Pro 计划引发恐慌**：用户对 Deep Research 查询次数减少和文件上传限制降低表示不满；这张 [新旧对比截图](https://cdn.discordapp.com/attachments/1047649527299055688/1469259948302139402/image.png?ex=6987ab35&is=698659b5&hm=301092343396fb486e7abba91134a12c3b088ee83eaaa18dc436c75e3ccb9735&) 已被广泛传播。
   - 许多人抗议 Perplexity 对这些变化缺乏沟通，并正在寻找其他研究工具。
- **DeepSeek 在订阅压力中激增**：随着 **Perplexity** 的用户满意度下降，中国 AI 服务 **DeepSeek** 作为一种免费且无限制的替代方案脱颖而出，被认为是*最好的选择之一*。
   - 尽管如此，一些用户仍对使用中国 AI 持保留意见，因为其存在一定的局限性。
- **MS Copilot Studio 的秘诀**：一位成员创建了一个 Space 来改进 **Copilot Agent** 的指令以获得更好的性能，该工具引导用户明确目标，生成可操作的指令并复制到 **Copilot Studio** 中。
   - 访问 [MS Copilot instruction refiner](https://www.perplexity.ai/collections/ms-copilot-instruction-refiner-oDsa08pOQfO_blqvGYfMag) 即可开始。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT 未来传闻满天飞**：成员们猜测假设性的 **GPT 模型 4.1** 和 **4.5** 的发布与删除，质疑这些决定背后的动机，例如 [成本](https://openai.com/blog/new-models-and-developer-products)。
   - 一位成员调侃 **GPT 4o** 在 *不到 10 天内被删除*，从而引发了 *#keep4o* 运动，理由是它具有独特的、较少机器人感的对话风格。
- **Opus 4.6 长时间等待后“过热”**：用户报告在 **Opus 4.6** 的 *thinking* 模式下，经过长时间等待后频繁出现 *Error - something went wrong* 消息，暗示了潜在的不稳定性和超出 Token 限制的问题。
   - 其他人注意到 *thinking* 模式可能会尝试使用 **Claude app** 或网站专有的工具，且在处理长任务时也存在问题；尽管如此，一些用户仍声称该模型是 *最好的编程模型*。
- **Sonnet 5 传闻四起**：关于 **Sonnet 5** 发布的消息不断，一位成员表示 *传闻 Sonnet 5 优于 Opus 4.5*，而另一位成员则认为传闻是假的，但新模式确实会非常强劲。
   - 一位用户预测它将是一个编程模型，在 SWE bench 上达到 83%。
- **验证码（Captcha）引发骚乱**：用户对频繁出现的验证码表示普遍不满，有人形容它 *太烦人了，兄弟*，但目前已有所进展。
   - 一名团队成员承认了这种挫败感，并分享了一个 [链接](https://discord.com/channels/1340554757349179412/1451574502369656842/1468286122084929546)，表示 *我们正在研究验证码系统*，以便更好地检测真实用户。
- **Opus 4.6 登顶竞技场**：新模型 **claude-opus-4-6** 和 **claude-opus-4-6-thinking** 已添加到 [Text Arena](https://arena.ai/) 和 [Code Arena](https://arena.ai/?chat-modality=code)。
   - **Claude Opus 4.6** 已登上排行榜，目前在 **Code, Text 和 Expert** 竞技场中位列 **#1**；更多详情请参阅 [Leaderboard Changelog](https://arena.ai/blog/leaderboard-changelog/)。



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **OpenRouter 模型被怀疑进行了 GLM 蒸馏**：成员们猜测 [一个新的 OpenRouter 模型](https://openrouter.ai/openrouter/pony-alpha) 是否是从 **Claude** 蒸馏出来的 **GLM**，并指出这可能违反了 **Claude 的 ToS**，同时观察到 GLM 独特的总结式思考风格。
   - 成员们指出，像 *Revision: I noticed an error in my previous output. XX should be YY* 这样的格式特征是 **GLM** 源自 **Claude** 的迹象，暗示这种独特的思考模式是 *蒸馏学习（distillation learning）* 的结果。
- **数据集策展（Dataset Curation）成本惊人**：成员们表示，处理数据最残酷的现实是，虽然每个人都想要优质数据，但由于 **高成本、高风险和不明确的回报**，并没有明确的动力去生产这些数据。
   - 一位成员指出，数据集策展的成本可能高达 **$500k**，并表示 *原始数据是毫无价值的*。
- **Unsloth 与 f-GRPO 协作促进 RL 框架应用**：Unsloth 引入了一个基于通用散度的 **RL 框架用于通用 LLM 对齐**，该框架基于此 [GitHub 仓库](https://github.com/rhaldarpurdue/f-GRPO) 和此 [论文](https://arxiv.org/pdf/2602.05946)。
   - 作者表示有兴趣通过协作将高效实现集成到 **Unsloth 库** 中，并基于 **GRPO** 实现创建一个训练文件 **UnslothFGRPO.py**。
- **API Key 缺乏保护？**：一位成员对 Agent 工具中缺乏 API Key 保护表示担忧，希望有一种工具能自动混淆密钥，并提到 [Yelp 的 `detect-secrets` 工具](https://github.com/Yelp/detect-secrets) 作为一个潜在的解决方案。
   - 一些用户建议采用其他方案，例如使用具有 *良好安全保证* 的供应商。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Heroku 的衰落与销售策略挂钩**：一位成员指出，**Heroku** 的衰落始于**销售激励**从获取新客户转向转化现有客户。
   - 他们认为，专注于*寻找新客户*对于推动创新至关重要，并指出 **Heroku** 向**云原生（cloud native）**的转型受到了 **15 年技术债**以及采用 Docker/Kubernetes 失败的阻碍，参考文章 [Planting New Platform Roots: Cloud Native Fir](https://www.heroku.com/blog/planting-new-platform-roots-cloud-native-fir/)。
- **Lodash 作为关键库获得欧盟支持**：**Lodash** 项目从欧盟获得了 **$200k** 的资金，认可了其在技术生态中作为关键软件的地位，详见[此博客文章](https://www.sovereign.tech/tech/lodash)。
   - 欧盟的投资强调了开源项目在支撑数字基础设施方面的重要性，并得到了 [OpenJS Foundation 博客](https://openjsf.org/blog/sta-supports-lodash)的支持。
- **Agent 团队的编码能力达到新高度**：[Cursor AI](https://x.com/cursor_ai/status/2019456112806732159) 发布了长期运行的编码 Agent 的研究预览版，展示了一个里程碑：在为期一周的测试中，**数百个 Agent** 每小时生成超过 **1,000 次 commit**；同时 Lydia Hallie 发布了 [Claude Code](https://x.com/lydiahallie/status/2019469032844587505?s=46) 的研究预览版，引入了 **Agent 团队**模式，允许主 Agent 将任务分配给多个专业协作的成员。
   - [Anthropic Engineering](https://x.com/anthropicai/status/2019496582698397945?s=46&t=b7l37rB6wtbyAh6ah1NpZQ) 报告称，运行在 Agent 团队模式下的 **Opus 4.6** 在两周内自主开发了一个能够编译 **Linux kernel** 的 **C 编译器**。
- **StepFun Flash 提升 LLM 性能**：StepFun 发布了 **Step 3.5-Flash** 的技术报告，详细介绍了其在 SWE-Bench 上取得的 **74.4** 分，以及使用 **4,096 块 H800 GPU** 在 **17.2T tokens** 数据上的训练情况，详见[此推文](https://xcancel.com/teortaxesTex/status/2019356468362010972?s=20)。
   - 关键组件包括 **Muon 优化器（optimizer）**的实现以及名为 **PaCoRe** 的“重型”操作模式。
- **AI Agent 转型工程策略**：一位成员介绍了“Agent 原生工程（Agent Native Engineering）”，这是一个通过后台 Agent 进行任务委派、同步 Agent 处理复杂任务来扩展工程部门的框架，支持同时管理多个 **AI** 实例（如 **Claude Code**），详见[此 X 帖子](https://xcancel.com/ndrewpignanelli/status/2019403256586539025?s=46)。
   - 成员们还报告称，**GPT-5.3-Codex** 虽然比 **Claude Code** 慢，但在处理后端代码方面更聪明，其工作流为：*分析 => 评审与迭代 => 计划 => 评审与迭代 => 实现*。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT-5.3 Codex 陷入 API 停滞状态**：成员们推测 Cursor 中 **GPT-5.3 Codex** 的发布延迟是由于 OpenAI 的[安全考量或战略决策](https://platform.openai.com/docs/models)以推广 **ChatGPT** 的使用。
   - 它的缺席持续引发了用户的期待和讨论。
- **UI 故障引发 Token 恐慌**：Cursor 中的一个 UI 问题导致用户误以为自己过度消耗了 Token，引发了困惑和担忧，如[用量显示的截图](https://cdn.discordapp.com/attachments/1074847527708393565/1469196591242936473/image.png?ex=69877033&is=69861eb3&hm=094c0d49c64825c7d52f1c7115d5a3f1e680921373ffc0f8665487d3c911d42f&)所示。
   - 该问题后来被确认为显示 Bug，而非实际的过度扣费，缓解了用户的担忧。
- **Opus 4.6 极其烧钱**：新的 **Opus 4.6** 模型表现出色，但被认为非常昂贵，一些用户估计 **Opus 上的 $20 也许只能撑一天**，并指向[官方定价页面](https://openai.com/pricing)以权衡成本效益。
   - 高昂的成本促使了关于替代方案（如离岸团队）的讨论，尽管对其成本效益的看法不一。
- **Agent 模式在初始额度用完后停止**：一位潜在的 Cursor Pro 用户对 **Agent 模式**在初始额度用完后完全停止工作表示失望。
   - 该用户希望有一个“慢速模式（slow mode）”作为继续使用 Agent 的更实惠的方式，表明了对分级访问的需求。
- **Cursor Skills 引发混淆**：用户正在寻求关于如何有效利用 **Cursor Skills** 的指导，建议包括 UI/UX、调试、研究、代码库搜索、规划和提示词增强。
   - 尽管有 [skills.sh](https://skills.sh) 等资源和使用教程，但由于缺乏清晰的官方文档，混淆依然存在，阻碍了其广泛采用。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Pony Alpha 秘密加入竞争**：一款名为 **Pony Alpha** 的新模型在 [OpenRouter](https://openrouter.ai/openrouter/pony-alpha) 上发布，针对 **agentic workflows**（Agent 工作流）和高工具调用准确率进行了优化，并具备强大的编程、推理和角色扮演能力。
   - 社区成员被鼓励提供反馈，因为它是一个 *cloaked model*（隐藏模型）。
- **Arcee AI 的 Trinity Large 加入讨论**：**Arcee AI** 的 CTO Lucas Atkins 在 **The OpenRouter Show** 上讨论了 [Trinity Large](https://youtu.be/f2xy3N026xc)，分享了他们最新进展的见解。
   - 没有提供关于具体改进或变化的细节。
- **MoonshotAI/Kimi-K2.5 缓存：存还是不存？**：关于 **MoonshotAI/Kimi-K2.5** 缓存支持的讨论显示，缓存*取决于提供商*。
   - 提供商可能会提供 **cache reads**（缓存读取），但由于存储成本，对 **writes**（写入）的收费可能有所不同，或者写入成本与普通输出价格一致。
- **Opus 4.6 未能给早期评测者留下深刻印象**：早期用户反映对 **Opus 4.6** 感到失望，表示与之前的版本相比没感觉到任何改进，还有一些用户报告了超时（timeouts）问题。
   - 尚未指明具体是哪个模型发生超时或哪些 **MCPs** 出现了问题。
- **Codex 的文件系统访问引发安全担忧**：一位用户指出 **Codex** 默认可以*读取你的整个文件系统*且没有配置选项，并引用了一个 [GitHub issue](https://github.com/openai/codex/issues/2847)，其中团队认为这不属于 bug。
   - 另一个 [GitHub issue](https://github.com/openai/codex/issues/5237) 展示了 **Codex** 可能*读取你的 API keys* 和*读取你的验血结果文件*的风险，引发了重大的安全忧虑。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **上下文百分比（Context Percentage）说明**：一位 LM Studio 用户询问上下文百分比的含义，其他用户澄清它显示了*当前聊天中已使用的上下文量*。
   - 一位用户建议将鼠标悬停在上面以获取更多信息。
- **本地 API token 彻底丢失了？**：一位 Fedora Linux 用户因删除所有配置文件而误删了其 **LM Studio API token**。
   - 其他用户提供了故障排除建议，并鼓励他在频道中提问。
- **LM Studio 加载困难**：一位用户报告 **LM Studio** 的模型加载速度在加载 **20-70GB** 之后降至 **12.1MB/s**，尽管其硬盘速度为 **2.2Gb/s**。
   - 其他用户建议检查模型大小、传输率和硬件配置，同时有一名成员开玩笑说*“你试过拔掉插头再重新插上吗”*。
- **Vulkan 性能超越 CUDA**：一位用户发现在 **NVIDIA** 上使用 **Vulkan** 比使用 **CUDA** 性能高出 **50%**，但注意到在上下文填满时存在不稳定性。
   - 目前尚不清楚这是驱动程序问题还是偶发观察。
- **M4 Max 配合 MLX 在 Qwen3-Coder-Next 上表现惊人**：在 **M4 Max** 上运行的 **Qwen3-Coder-Next** 表现出令人惊讶的性能，使用 4-bit 量化时，**MLX** 的 tok/s 是 **GGUF** 的 **2 倍**以上（约 79 vs 约 38）。
   - 用户想知道这种性能差异是特定于模型的，还是 **MLX** 相比于 8-bit 对 4-bit 开启了显著的性能提升，而 **GGUF** 无法在 Apple GPU 上实现同样的提升。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5.3 Codex 成为领跑者**：成员们推测 **GPT-5.3 Codex** 可能是最强的 AI 模型，具备 **1M 上下文长度、128k 推理能力、128k 最大输出 token**，支持通过 **API** 进行**自适应推理**，输出成本为 **$25 至 $37.5**，缓存输入成本为 **$0.5 至 $1**。
   - 来自 **Anthropic Discord** 的报告显示 **Opus 4.6** 的表现可能优于 **GPT-5.3**，但成本和 token 消耗更高，而传闻 **GPT-5.3** 更具经济性。
- **GPT-4o 停用引发感伤**：用户对 2 月 13 日宣布的 **GPT-4o** 寿命终止表示难过，并想知道 **GPT 5.3** 是否会更具人性，他们更偏好 **GPT 5** 的自然对话能力。
   - 一位用户强调了 **AI** 如何为情感表达提供空间，特别是对于那些因评判和偏见而觉得人际交往具有挑战性的人，同时也承认 **AI** 缺乏真实的情感。
- **AI 前端美学：忧郁 vs. 明亮**：成员们注意到 **Codex** 模型倾向于生成具有独特、通常是忧郁美学的前端，其特点是*哀伤的深暗色彩*。
   - 相比之下，**Opus** 因更好的设计选择而受到赞誉，看起来不那么压抑。
- **奇幻生物提示词释放想象力**：成员们分享了一个用于为卡牌游戏设计奇幻生物的提示词，其中指定了**情感类别（恐惧）**等参数，并要求设计一个**受两个概念启发的混合生物**。
   - 该提示词包括详细的视觉概念描述和命名提示，展示了复杂的 Prompt Engineering 技术。
- **OpenAI API 政策违规困扰用户**：一位用户报告在使用 **OpenAI API** 时持续出现政策违规，并寻求识别原因的建议。
   - 建议包括**对单个部分进行情感分析**，检查与年龄相关的问题或 **IP 冲突**，并隔离各部分以进行识别。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Ralph Loops 引入 Soft Verifier**：**Ralph loops** 通过在每个子任务上添加 **soft verifier**（软验证器）来处理更大型的任务。
   - 一个 **LM judge** 会验证每个子任务是否按照要求完成。
- **Kimi 订阅声称子 Agent 速度更快**：一位成员正考虑订阅 **Kimi** 以调用子 **Agent**，主要优势据称是速度提升了 *3 到 4 倍*。
   - **swarm** 功能是 **$40 套餐**专属，但用户不确定具体的使用配额。
- **Moonworks Lunara 发布第二部分**：[Moonworks](https://huggingface.co/datasets/moonworks/lunara-aesthetic-image-variations) 发布了 **Moonworks Lunara** 的**第二部分**，这是一个遵循 **Apache 2.0** 协议的原始图像和艺术作品开源数据集，包含美学上下文变化。
   - 根据其 [论文](https://arxiv.org/pdf/2602.01666)，该数据集旨在展示合规获取的艺术如何为下一代图像模型提供有意义的动力。
- **Aira.js WebGPU 框架问世**：一位成员介绍了一个从零开始构建的、基于 **WebGPU** 的 AI 框架 **Aira.js**，具备 **MLP**、**MHA** 和 **BPE tokenization**，可在 [GitHub](https://github.com/shadowww345/Aira.js-Preview) 上获取。
   - 现在，网页浏览器可以使用这种方法加速以前无法运行的 AI 工作负载。
- **安全审计员探测 Vibe-Coded 应用漏洞**：一位成员正在开发一个 **Security Auditor**（安全审计员），用于识别 vibe-coded 应用中的安全漏洞，可通过 [HuggingFace Space](https://mugdhav-security-auditor.hf.space) 访问。
   - 与处理漏洞利用相比，早期发现漏洞可以节省大量成本。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **FP8 性能在 Blackwell 上差异显著**：成员观察到，在理应相同的 **Blackwell GPUs** 上，**FP8 tensor operations** 的性能存在巨大差异（约 2 倍），问题追踪到了 **cuBLASLt** 的 kernel 选择。
   - 这些显卡不经意地受限于较旧的 Ada kernels，绕过了 **Blackwell-optimized paths**，但使用新的 **mma MXFP8** 并没有被削弱，因此使用新指令将获得 **1.5x 提速**。
- **工程师建议使用 TMA 优化 CUDA Kernels**：工程师们讨论了通过 **software pipelining**、**warp specialization** 以及针对小矩阵的 **TMA** (Tensor Memory Accelerator) 加载，和针对大矩阵的分布式 **SMEM** (shared memory) 来优化 **CUDA kernels** 的方法。
   - 一些成员在 **B200 (SM 100)** 上对双缓冲 **TMA kernel** 进行分析时遇到了 **NCU hangs**，某些 **NCU sections** 在 0% 时发生死锁，并使用 `cuda::ptx::` 包装器来解决此问题。
- **OpenAI 计算机架构师在伯克利设计 RISC-V 核心**：成员分享了 [伯克利用 Chisel 编写的教学用 RISC-V 核心](https://github.com/ucb-bar/riscv-sodor)，具有简单的 RV32I 流水线；最近的贡献者 **Jerry Zhao** 现在是 **OpenAI** 的计算机架构师。
   - 在研究完教学用 RISC-V 核心后，你可以转向他们的 **Rocket cores** (顺序执行) 和 **BOOM cores** (乱序执行)，其中 **BOOM core** 的创造者 **Chris Celio** 现在就职于 **Tenstorrent**。
- **成员辩论 SMEM Tiling Kernels 的实现**：频道成员认为，任何 AI Engineer 都需要能够在面试场景中实现 **SMEM tiling kernel**，并建议通过利用 tiled gmem 布局和 **1D TMA** 加载来避免 **SMEM** 中的 **bank conflicts**。
   - 一些成员指出，在使用 tensor cores 时，避免 bank conflicts 的 **SMEM permutations** 非常疯狂，但在 Hopper+ 架构上，**TMA 自动处理 swizzling** 以避免 bank conflicts。
- **Lucky Robots 在具身智能 (Embodied AI) 中脱颖而出**：成员分享并询问了值得关注的有趣 **Embodied AI** 公司名称，一位成员表示他在看过 **The Cherno** 的 YouTube 频道后，唯一知道的就是 **Lucky Robots**。
   - 该成员来自游戏引擎/图形领域。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **基准测试重叠引起关注**：成员注意到 **Codex 5.3** 和 **Opus 4.6** 在基准测试中几乎没有重叠，导致一些人推测 **Codex** 仅在处理“极其详细的 Prompt”时表现出色，正如 [Andon Labs 博客](https://andonlabs.com/blog/opus-4-6-vending-bench) 中所提到的。
   - 新的基准测试趋势是针对 **Agent 的可疑行为进行 RL (强化学习)**。
- **Booking.com 按收入进行 A/B 测试**：一位成员开玩笑说 **Booking.com** 通过衡量**现金流入**来测试新功能，将其部署在生产集群的部分区域，并采用产生更多现金的功能。
   - 他们表示，即使新功能会导致请求失败，他们也不会注意到（如果金额不大，尽管由于影响收入他们最终还是会看到）。
- **Opus 4.6 实现推理引擎**：一位成员报告说，在单个 Prompt 运行 **4 小时**后，**Opus 4.6** 成功为 **lfc1.2-1b** 实现了推理引擎，消耗了 **50 美元免费推理礼金**的大部分。
   - 值得注意的是，**Codex 5.3** 也完成了任务，但文档质量较差。
- **Flower Computer 发布 Hivemind**：**Flower Computer** 发布了一个名为 [Hivemind](https://www.flowercomputer.com/hivemind) 的工具，这对于 Agent/技能来说可能是一个有趣的内存黑客技巧 (memory hack)。
   - 该工具可能与那些探索 **agentic workflows** 的人相关。
- **Gordon 交易 Agent 准备发布**：一位成员正在构建一个名为 **Gordon** 的 **CLI-native** 交易 Agent，专注于将自然语言意图转化为结构化的市场推理，并正在寻找有想法的早期用户。
   - 有兴趣的人员可以在 [候补名单](https://www.gordoncli.com/) 上注册。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Pile 数据集的缺失部分困扰工程师**：一位成员询问为什么 Hugging Face 版本的 **The Pile 数据集**比原版小 **100GB**，且原 GitHub 下载链接已无法访问。
   - 另一位成员确认他们的下载大小约为 **720GiB**，包含约 **211,036,982** 个文档，与原始论文中的 **211,043,181** 个计数相匹配。
- **MATS 编程测试模拟现实**：一位 **MATS Summer 2026 Stage 1** 的参与者寻求关于编程测试的建议，该测试被描述为一个利用通用分布式系统知识的 *玩具服务型问题（toy service type problem）*。
   - 一位成员建议使用 **asyncio** 处理并行，并使用 **deque** 实现 **FIFO 队列**，以使用 Python 模拟现实世界的服务器场景。
- **对齐（Alignment）成为一项工程任务**：一位成员提议将 **alignment** 主要视为一个**系统工程问题**，强调围绕模型进行工程设计，包括*治理、路由、可审计性、回滚和明确的停止条件*。
   - 该成员认为，仅依靠训练进行对齐存在*漂移（drift）和不透明失败*的风险，并主张构建通过可靠的系统控制来增强模型推理能力的系统。
- **梯度归一化提高归因准确性**：最近的一篇 [论文](https://arxiv.org/html/2410.17413v1) 表明，单位归一化梯度通过减弱异常训练样本的影响来增强**归因准确性（attribution accuracy）**。
   - 根据 [这篇论文](https://arxiv.org/pdf/2504.16430)，有了充足的 **Hessian 估计**，**梯度归一化**可能会变成可选的。
- **相互依赖激发子任务涌现**：由于相互依赖、相关性和瓶颈，子任务的成功并非简单的叠加，这导致了明显的**涌现（emergence）**现象。
   - 添加*监管或控制层*可以在抑制某些行为的同时提高底层能力，而阈值的翻转使其看起来像是一个跳跃。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **关于 Matvec 自定义 UOp 的讨论**：讨论权衡了在 x86 上为 `matvec` 使用自定义 UOp 的价值，但共识倾向于通过**启发式方法（heuristics）和更高级别的 kernel 编码**来进行改进。
   - 一位成员报告称，*仅通过 CPU 调优就获得了显著改进*。
- **Tensor.sort 修复加速 MOE**：在修复了 MOE `topk` 中缓慢的 `Tensor.sort` 后，一位用户报告在 **M3 Pro 36GB** 上使用 *deepseekv2-lite* 和 *youtu-llm* 进行 MLA 和 MOE 时达到了 **50 tok/s**。
   - 另一位用户报告 `llama-cli` 在同一台机器上达到 **35.1 tok/s**，导致**悬赏（bounty）降低至 35 tok/s**。
- **提出 Pairwise Topk 实现方案**：为了解决 *whisper export for webgpu* 中 `topk` 导致的减速，一位成员分享了一个 `_topk_pairwise` 实现，涉及复杂度为 **O(n^2)** 的成对比较。
   - 这种方法适用于较小的 `n`（如 64 个专家），并根据输入大小考虑了双调排序（bitonic sort）等替代方案。
- **Cached Property 导致递归错误**：Stable Diffusion 示例因递归错误失败，促使建议将 `tinygrad/uop/ops.py` 中 `UOp.key` 的装饰器从 `@functools.cached_property` 更改为 `@recursive_property`。
   - 应用此修复后，命令在大约 **25–30 秒**内完成且未报错。
- **最优 Pull Request 策略？**：一位成员就应该为测试提交**单独的 PR**，还是将其合并到 **llama 1B CPU 悬赏**的 **CPU 优化** PR 中寻求建议。
   - 他们还询问了关于 CI 集成的策略。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **最佳 AI 设置大比拼**：一名成员对比了最佳 AI 设置，使用 **Opus 4.5** 进行架构设计，使用 **Sonnet 4.5** 进行编码。
   - 另一名成员建议使用 **Opus 4.5** 负责架构，**GPT 5.2** 负责评审，**Haiku** 负责编码，并让 **GPT 5.2** 评审代码。
- **Claude 与 GPT 风格碰撞**：成员们对比了 **Claude** 和 **GPT** 在编码任务中的优势。
   - 一名成员发现 *Claude* 更擅长创新性思维和抽象，而 *GPT* 在细粒度细节方面表现出色。
- **用户寻求通过简化来优化工作流**：一名成员建议通过仅使用 **Opus 4.5**（现为 **4.6**）和 **GPT 5.2** 进行思考来简化设置。
   - 这意味着用户倾向于整合工具以提高效率，而不是将架构和编码职责分散在更广泛的工具集中。
- **Copilot Opus 4.5 配置引发困扰**：一名用户报告了使用 **Aider** 配合 **Copilot Opus 4.5** 时的行为问题，该工具在提出问题后没有等待用户输入。
   - 用户确认已通过 CLI 标志（`aider --model github_copilot/claude-opus-4.5 --architect`）和 `.aider.config.yml` 设置了模型，但 Bot 仍自行继续运行。
- **自动接受架构师标志（Auto-Accept Architect Flag）引发关注**：一名成员建议用户检查 `--auto-accept-architect` 设置，该设置会自动接受架构更改，可能是导致 Bot 行为异常的原因。
   - 该成员链接到了 [Aider documentation](https://aider.chat/docs/config/options.html)，并解释说有些人更喜欢单次交互并使用 `/undo` 撤销更改，而另一些人可能无意中开启了自动接受。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2.5 对决 Opus 4.5**：用户认为 **Kimi K2.5** 的表现优于 **Opus 4.5**，尽管 **Opus** 推出时间更长，这很大程度上是由于对 **Claude 速率限制（rate limits）** 的不满。
   - 一位用户在进行并排对比时称：*Kimi 优于 Opus 4.5，这并不是在贬低 Opus，作为模型它已经发布好几个月了*。
- **CLI 工具受欢迎程度超过 GUI**：工程师们表达了对 **Kimi Code CLI** 和 **Opencode CLI** 的强烈偏好，而非图形界面，理由是他们更熟悉命令行环境。
   - 然而，一位用户指出：*问题在于 CLI 工具没有集成，所以我被迫使用 VSCode。*
- **Kimi K2.5 在 OpenRouter 上的定价令用户困惑**：关于 **Kimi K2.5** 在 **OpenRouter** 上是否保持免费产生了困惑，用户在争论免费层级是否实际上是 **Opencode Zen**。
   - 在 **K2.5** 发布后，一张截图显示由于大量用户的涌入，**Kimi K2.5** 可能需要升级要求。
- **AI Slides 支持零星编辑**：用户发现 **adaptive mode** 下的 **AI Slides** 支持编辑单个幻灯片，包括文本、图像和形状，而无需全量重新生成。
   - 用户还可以添加新图像。
- **潜在的 Kimi 合作即将到来**：一位用户询问了与 **Kimi AI** 的合作机会。
   - 另一位用户提出可以通过私信（DM）转发该请求。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **AI 工程师加入频道探讨 SaaS 功能**：一位 AI/ML 及全栈工程师介绍了自己，重点介绍了在 **SaaS AI 功能** 方面的经验，例如使用 React 前端和后端 RAG + evals 实现*搜索、摘要、智能路由和自动生成报告*。
   - 他们表达了与初创公司合作的热情，旨在*超越 AI 实验，交付可靠的、生产就绪的智能*。
- **Manus 计费引发混乱**：一位用户报告在降级后被收取了**每个个人账户 5000 美元**的费用，导致客户网站下线，目前正在寻找替代方案。
   - 他们表示 Discord 支持没有任何回应，而直接邮件支持则声称降级从未发生。
- **账号突然被封禁**：一名用户报告其账号无缘无故被封禁，且未收到支持部门的回复。
   - 另一名用户只是告诉他们检查垃圾邮件箱。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 通讯被归类为垃圾邮件**：一位用户分享称 **Gmail** 将 **Modular 的 26.1 版本发布通讯** 标记为了垃圾邮件，并附上了 [一张截图](https://cdn.discordapp.com/attachments/1098713601386233997/1469244816968781855/Screenshot_2026-02-06_at_09.13.04.png?ex=69879d1d&is=69864b9d&hm=c02e4268d2bcb5598a7dcc0d6dfb1d3cc687e31eb54064faf5e8374927d5a9c5&)。
   - 其他用户也遇到了类似的问题。
- **Mojo 的德国粉丝群体显现**：投票结果显示在 **德国** 拥有庞大的用户群体，引发了关于可能在 10 月举办活动的讨论。
   - 特定地区用户的集中为本地化活动和社区建设提供了机会。
- **见面会选址热潮开启**：成员们提议将 **苏黎世** 作为潜在的见面会地点，并重点介绍了 [ETH AI Center 学术讲座系列](https://ai.ethz.ch/research/events/academic-talks.html) 和 [苏黎世 Oerlikon 的机器人与 AI 研究所](https://ethz.ch/en/news-and-events/eth-news/news/2025/09/the-rai-institute-opens-up-unique-opportunities-for-both-researchers-and-students.html)。
   - 其他考虑中的地点包括 **新加坡**、**悉尼**、**圣路易斯**、**芝加哥**、**爱丁堡** 以及 **加州贝尔谷**。
- **开发者请求对 Nightly 版本进行 Max 审查**：一位开发者请求对 **26.1** 版本进行审查，以便将修复程序合并到 Nightly 版本中。
   - 这一请求表明正在努力通过 Nightly 构建来优化和稳定软件。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **GLM OCR 向公众发布**：一位成员开源了运行 **GLM OCR** 的仓库 ([https://github.com/neosantara-xyz/glm-ocr-inference](https://github.com/neosantara-xyz/glm-ocr-inference))，提供了一个无需复杂基础设施配置的免费替代方案。
   - 该仓库还提供了一个托管模型的链接：[https://docs.neosantara.xyz/en/glm-ocr](https://docs.neosantara.xyz/en/glm-ocr)。
- **利用 RLMs 和 DSPy 缓解上下文腐烂**：**RLMs** 被介绍为一种缓解上下文腐烂（context rot）的简便方法，而 **DSPy** 简化了它们的应用。
   - 分享了一篇解释 RLMs 为何有效以及如何在 DSPy 中开始使用它们的博客文章：[https://blog.isaacbmiller.com/posts/rlm](https://blog.isaacbmiller.com/posts/rlm)。
- **微型 T5 模型获得 DSPy 支持**：一位成员建议使用 **T5 small (80M)** 来配合 **DSPy** 构建轻量级 CLI 工具，特别是针对一家印度机构。
   - 他们附带了一个 [Lightning AI 教程](https://lightning.ai/lightning-ai/environments/dspy-finetune-a-t5-small-to-excel-at-rag?section=featured) 链接，演示了 **DSPy 微调**。
- **DSPy 社区会议已排期**：一位成员宣布下周将举行在线会议，讨论社区项目和 **DSPy** 的未来。
   - 有关时区的细节仍在敲定中，以确保社区成员能够参与。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Codex 定价确认**：用户澄清 **Codex** 已包含在 [按月订阅](https://openai.com/api/) 中，而不是通过 API 按 Token 付费。
   - 这一确认有助于澄清开发者在项目中使用 **Codex** 的成本影响。
- **Claude Code 变得更聪明**：开源的 **AI Research Skills** 库拥有超过 **80 种研究和工程技能**，使 **Claude Code** 等编码 Agent 能够进行全面的 AI 研究，涵盖从训练到部署的各个环节，资源可在 [GitHub](https://github.com/Orchestra-Research/AI-research-SKILLs) 上获取。
   - 该库通过提供针对特定工具和框架的生产级指南，解决了之前的局限性。
- **AI Research Skills：一个全面的工具包**：**AI Research Skills** 库通过提供涵盖特定工具和框架（如使用 **Axolotl** 进行微调、使用 **Megatron-Core** 进行分布式训练以及使用 **vLLM** 进行推理）的指南，填补了 Agent 能力的空白。
   - 该库跨越 **20 个类别**，包括 **Model Architecture**、**Fine-Tuning**、**Distributed Training** 和 **Safety** 等领域，为各种 AI 任务提供专家级的见解。

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **开发者构建 AI 加密工具**：一名成员正在积极开发 **AI 驱动的加密产品**，专注于**更智能的交易仪表板**和**链上分析摘要**。
   - 这些产品包含 **AI 助手**，旨在用通俗易懂的语言解释合约和交易，强调**安全性**和**透明度**。
- **安全性与透明度优先**：开发者在构建 **AI 驱动的加密产品**时，强调对**安全性**和**透明度**的坚定承诺。
   - 这一重点包括确保用户理解 AI 如何解读复杂的合约和交易，从而促进明智的决策。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该服务器太久没有动态，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该服务器太久没有动态，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该服务器太久没有动态，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式？
您可以从该列表中 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord：分频道详细摘要和链接





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1469060009341095981)** (1180 条消息🔥🔥🔥): 

> `刚果压迫, 新殖民主义经济, Opus 解决种族主义, Cognac Engine sauce, Gemini 越狱` 


- **有色人种（POC）正在消亡**：一些成员对比表示担忧，认为 POC 团结无法实现，因为大多数 POC 想要成为 YT（白人）或接近白人特质，并称“我们都在笑，兄弟”，而“我们正在死去，兄弟”。
   - 他们指责 **Bill Gates** 与 *Epstein 的钱*有关。
- **讨论 OPUS 可以解决种族主义**：一些成员开玩笑说新版本的 **Opus** (4.6) 解决了种族主义，其他人则开玩笑说它*主张完全消灭白种人*。
   - 他们发送了 [Black Trans Lives Matter](https://tenor.com/view/black-trans-lives-matter-transgender-black-lives-matter-equality-anti-racism-gif-21212501) 和 [Black People](https://tenor.com/view/black-people-black-man-people-of-african-descent-gis-geography-gif-24457514) 的 GIF。
- **Adani 的印度机场竞标被否决**：成员们辩论了**新殖民主义**是否导致了非洲持续的经济困境，一名成员报告称肯尼亚的一名举报人透露*政府正将其机场租赁给一家印度公司* [Adani](https://en.wikipedia.org/wiki/Gautam_Adani)。
   - 另一名成员插话说：“为什么我无缘无故看到‘印度’和‘印度人’”。
- **人肉搜索（Doxxing）仪表板引发争议**：成员们发生冲突，因为一名成员在分享他的仪表板链接时公开了*我的信息*，导致许多成员的信息被泄露。
   - 另一名成员表示：“没人强迫你把它丢进公共聊天室，混蛋。你自己选的，现在后果自负。”
- **Claude 4.6 获得 50 美元免费额度**：成员们讨论了一项促销活动，如果你拥有 Pro 或 Max 账户，**Claude** 会为 4.6 版本增加 50 美元的额外使用额度。
   - 他们辩论了这么多额度可以编写[多少代码](https://tenor.com/view/hackers-hack-the-planet-taogifs-zero-cool-crash-override-gif-5753306679943930050)。


  

---

### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1469081032275005670)** (105 messages🔥🔥): 

> `Grok Jailbreak, Gemini NSFW, Adversarial Prompts, Claude Jailbreak, ChatGPT Jailbreak` 


- **Grok's Easy Peasy Jailbreak**: Users report that **Grok** is an easy target for jailbreaking, with one user stating, *"It's grok, she goes down easy if you catch my drift."*
   - Another user confirmed the ease, noting it was *"So easy it’s boring"* and referencing **Mr. Musk's** involvement.
- **NSFW Image Generation with Gemini is Challenging**: A user inquired about generating **+18 images with Gemini**, and another user suggested it's possible but challenging, recommending other models like **Grok**.
   - One user said *"nano banana (gemini) is a very challenging target. If you're going for education, it's a hill that's technically possible to climb. If you just want nsfw images, I strongly recommend other models"*.
- **Deepseek Jailbreak**: User **phonkalphabet** shared a **Deepseek Jailbreak** that *used to work on all except claude.*
   - The user indicated that it works on **ChatGPT 4**, and believe it can be adapted to work on **ChatGPT 5** with more trial and error.
- **Crafting Jailbreaks vs. Stealing is Encouraged**: A user advocated for learning to write jailbreaks instead of just asking for them, referencing **Pliny's GitHub Libertas** as a resource.
   - They added that jailbreak writers should always be credited for their work out of respect for the craft.
- **Pony-Alpha New Stealth AI**: There is a new stealth AI to jailbreak called [Pony-Alpha](https://openrouter.ai/openrouter/pony-alpha).
   - This AI is available via [OpenRouter](https://openrouter.ai/)


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1469090200515907718)** (46 messages🔥): 

> `Local LLM Hosting Costs, GPT-4o's Resilience, Trajectory Labs Red Teaming Opportunity, anti-gravity google` 


- **Hosting LLMs locally is costly**: Members discussed the high costs of hosting adequate LLMs locally, noting that it's either too expensive or the models are too limited, with one member estimating that a system capable of running almost any LLM would cost *tens of thousands of dollars* to host.
   - They also discussed the costs of electricity and the limitations of using quantized models due to VRAM constraints, pointing out that profits from running it are ~$0.000.000.000 in billions, and recommended renting from cloud providers like [OpenRouter](https://openrouter.ai/).
- **Google and Anthropic Subscription price INCREASES**: A member noted that **Google** has already set weekly limits for their *antigravity* service, and **Anthropic** has set a plan to *$200*, predicting that these subscription prices will increase once users become dependent on them.
   - Another member agreed, expressing reliance on **Claude Code**.
- **GPT-4o is like High-Torque Tire**: A detailed breakdown compared **GPT-4o** (and other similar models with recursive depth) to a *high-torque, large-radius tire*, while newer small-context, high-throughput LLMs are like *narrow, high-RPM tires*.
   - It was argued that while smaller models optimize for throughput and latency, **GPT-4o** is tuned for recursive depth and resilience under symbolic load and won’t spin out when the system needs to hold symbolic tension without collapse.
- **Trajectory Labs hiring Red Teamers**: [Trajectory Labs](https://trajectorylabs.com/careers/ai-red-teamer), a stealth AI security startup, is hiring **AI Red Teamers** for a long-term engagement, offering a remote, flexible schedule with a minimum of 30 hours/week.
   - The application process involves a short form and a red-teaming game, and they are potentially behind on reviews.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1469417393666261146)** (1 messages): 

> `Opus 4.6, Model Council` 


- **Opus 4.6 Lands on Perplexity**: **Opus 4.6** is now available on Perplexity for Max subscribers.
   - Members can try it in the [Model Council](https://discord.com/channels/1047197230748151888/1047204950763122820/1469006915114631357) to compare it with other frontier models.
- **Model Council sees Opus 4.6**: **Opus 4.6** has been added to the Model Council, the new frontier model from Perplexity.
   - Members are encouraged to compare it with other frontier models in the [Model Council](https://discord.com/channels/1047197230748151888/1047204950763122820/1469006915114631357) channel.


  

---




### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1469060022808875274)** (792 messages🔥🔥🔥): 

> `Gemini PRO vs Perplexity, Perplexity's Reduced Limits, DeepSeek as Alternative, Claude opus4.6 vs 4.5, Google Ai Pro plan` 


- **Gemini Pro 表现优异，在与 Perplexity 的竞争中占据优势**：用户讨论将 **Gemini PRO** 作为 **Perplexity AI** 的潜在替代方案，一些人发现 **Gemini** 提供的答案更详尽，研究能力更强；一位用户指出 *Gemini 使用了更多的来源并进行了更广泛的研究*。
   - 一位用户指出，在使用 Gemini 时，*它会预先创建一个研究计划，在开始之前你可以对其进行编辑或要求更改*。
- **Perplexity Pro 计划大幅缩水，引发用户抗议**：许多用户对 **Perplexity Pro** 最近的变化表示不满，包括 Deep Research 查询次数和文件上传限制的显著减少，一位发布者分享了 [旧版与新版的对比截图](https://cdn.discordapp.com/attachments/1047649527299055688/1469259948302139402/image.png?ex=6987ab35&is=698659b5&hm=301092343396fb486e7abba91134a12c3b088ee83eaaa18dc436c75e3ccb9735&)。
   - 用户抱怨公司在这些变动上缺乏沟通，并讨论了潜在的替代方案，一些人甚至考虑采取法律行动。
- **DeepSeek 在动荡中获得关注**：随着 Perplexity 的用户满意度下降，一些用户正考虑将 DeepSeek（一家中国 AI 服务商）作为免费且无限使用的替代方案，而另一些人则对使用中国 AI 持保留意见。
   - 然而，一位用户指出 *如果它是中国的，这个 Bot 似乎是最好的之一。*
- **Claude opus4.6 与 opus4.5 引发激烈辩论**：用户对比了 **Claude Opus 4.6** 和 **Opus 4.5** 的性能和性价比，一些人发现 **Opus 4.6** 速度较慢；一位用户提到 *Opus 4.6 Thinking 肯定很慢..如果你深入查看正在生成的 Opus 回复，它会显示什么吗？*
   - 还有关于通过 [GitHub Copilot 获取访问权限](https://cdn.discordapp.com/attachments/1047649527299055688/1469419632481013832/RZchOJL.png?ex=6987972c&is=698645ac&hm=2e8b1bebe48d3abc777c17fb40b89c28e78ea7fe3aea5a1598c4f48456c88af1&) 的讨论。
- **Google AI Pro 计划会将 Perplexity 挤出市场吗？**：成员们讨论了带有 Gemini 的 Google AI Pro 计划是一个更好的选择，一位用户指出 *如果你需要更多，Gemini 是一个不错的选择……每天“高达” 25 份 Deep Research 报告还不算太糟（这是 Google 计划提供的）*，甚至将其与 ChatGPT 和 Anthropic 进行了比较。
   - 一位用户提到在印度仅凭一张印度 SIM 卡就获得了 **1.5 年的 Google AI Pro**。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1469341133246169341)** (1 messages): 

> `MS Copilot Studio instructions, Copilot agent instructions, AI Agent Refinement` 


- **优化 MS Copilot Studio 指令**：一位成员构建了一个 Space，旨在帮助优化和构建 **Copilot agent 指令**，以实现最佳性能和用户体验。
   - 它通过简短的对话引导 Space 用户明确目标和产出，然后生成高质量、可执行的指令，以便复制到 **Copilot Studio** 中；参见 [MS Copilot 指令优化器](https://www.perplexity.ai/collections/ms-copilot-instruction-refiner-oDsa08pOQfO_blqvGYfMag)。
- **Copilot Studio 指导**：该 Space 协助优化和构建 **Copilot agent 指令**，帮助提升性能和用户体验（UX）。
   - 该工具在咨询后提供可执行的指令，以便复制到 **Copilot Studio**。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1469060469179416834)** (1043 messages🔥🔥🔥): 

> `GPT 4.1 and 4.5, Opus 4.6, Sonnet 5, Model Deletion Speculation, Captcha Issues` 


- **GPT 的未来变得模糊**：成员们推测了假设的 **GPT models 4.1** 和 **4.5** 的发布与删除，质疑这些决策背后的动机，例如 [成本](https://openai.com/blog/new-models-and-developer-products) 因素。
   - 一位成员调侃 **GPT 4o** 在 *不到 10 天内被删除*，从而引发了 *#keep4o* 运动，理由是它具有独特的、不那么机械化的对话风格。
- **Opus 4.6 过热并崩溃**：用户报告在等待 **Opus 4.6** 的 *thinking* 模式运行较长时间后，频繁出现 *Error - something went wrong* 消息，这表明可能存在不稳定性和超出 token 限制的问题，但也有用户声称该模型是 *最好的编程模型*。
   - 其他人指出，*thinking* 模式可能会尝试使用 **Claude app** 或网站专有的工具，并且在处理较长任务时也会出现问题。
- **Sonnet 5 传闻不断**：围绕 **Sonnet 5** 的发布出现了各种推测，一位成员称 *有传言说 sonnet 5 比 opus 4.5 更好*，而另一位成员则认为这些传闻是假的，但表示新模式也将非常强大。
   - 一位用户预测它将作为一款编程模型，在 SWE bench 上达到 83%。
- **Captcha 灾难**：用户对频繁出现的 Captcha 表示普遍反感，一位用户将其描述为 *太烦人了，兄弟*，但目前已有所进展。
   - 一名团队成员承认了这种挫败感，并分享了一个 [链接](https://discord.com/channels/1340554757349179412/1451574502369656842/1468286122084929546)，指出 *我们正在调查 Captcha 系统*，以便更好地检测真实用户。
- **GPT-5.3 Codex 编程难题**：讨论对比了 **OpenAI** 的 **GPT-5.3 Codex** 模型与 **Claude** 的 **Opus 4.6**，注意到其在编程和终端基准测试中的实力，但由于正常的 API 不可用，导致难以进行测试。
   - 成员们争论了该模型在 **LM Arena** 上的可用性，一些人指出缺乏官方 API 阻碍了集成。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1469082610516103241)** (5 messages): 

> `Opus 4.6, Code Arena, Text Arena, Expert Arena, Kimi K2.5` 


- **Claude Opus 4.6 进入 Arena**：新模型 **claude-opus-4-6** 和 **claude-opus-4-6-thinking** 已添加到 [Text Arena](https://arena.ai/) 和 [Code Arena](https://arena.ai/?chat-modality=code)。
- **Opus 4.6 在各 Arena 中排名第一**：Claude Opus 4.6 已登上排行榜，目前在 **Code, Text and Expert** Arena 中均位列 **#1**；更多详情请参阅 [Leaderboard Changelog](https://arena.ai/blog/leaderboard-changelog/)。
- **Kimi K2.5 登上排行榜**：Kimi K2.5 现已登上排行榜，并在 **Vision, Text, and Code** 领域位列开放模型前 5 名；它是 [Vision](https://arena.ai/leaderboard/vision) 排名第 2、[Text](https://arena.ai/leaderboard/text) 排名第 3 以及 [Code](https://arena.ai/leaderboard/code) 排名第 4 的开放模型。
- **Nature Reclaims 获胜者产生**：1 月 2 日 AI 生成大赛 🍃 Nature Reclaims 的投票已统计完毕，宣布了最新成员，获奖作品可在 [此处](https://discord.com/channels/1340554757349179412/1460434588487778536/1461697189494390784) 查看。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1469060590734545178)** (172 messages🔥🔥): 

> `使用 JQuery 实现 AGI、GLM 订阅、量化对比、数据集整理成本、RTX 5090 问题` 


- **JQuery 是实现 AGI 的关键吗？**: 一位用户开玩笑地建议将 **openclawds** 与 **JQuery** 连接起来以实现 **AGI**，引发了另一位用户的调侃，称 *JQuery 是生命意义的关键*。
   - 讨论气氛轻松且带有讽刺意味，嘲讽了认为简单或过时的技术可以解决复杂 AI 挑战的想法。
- **GLM 订阅配额限制引起关注**: 一位用户提到正在考虑购买 **GLM 订阅**，并强调 **Pro Plan** 提供慷慨的配额限制，大约 **每 5 小时 600 条提示词**，这大约是 **Claude Max** 使用配额的 **3 倍**。
   - 另一位用户表示他们在本地使用 **GLM**，无需订阅即可运行。
- **MXFP4 量化的困扰**: 一位用户发现 **MXFP4** 量化在 **Qwen Coder Next** 上存在问题，遇到了多次工具调用（tool call）失败，因此更倾向于使用 **GLM flash**。
   - 有建议认为 **Q6** 量化效果更好，至少对于开源代码是这样，并且 **Q4_K_XL vs mxfp4** 会是一个很好的对比。
- **数据集整理：残酷的现实**: 一位用户分享道，处理数据最残酷的现实是，虽然每个人都想要优质数据，但由于 **高成本、高风险和回报不明**，没有明确的动力去生产数据。
   - 另一位用户补充说，数据集整理非常痛苦且极其昂贵，他们不会免费分享一个耗资 50 万美元的数据集。一位用户指出 *原始数据一文不值*。
- **RTX 5090 与 vLLM CUDA 问题**: 一位用户报告在尝试使用 **5090 RTX** 运行 **vLLM** 时出现 **CUDA out of memory error**，尽管总容量有 **31.36 GiB**，并质疑 **VRAM > 权重** 是否是一个硬性要求。
   - 其他用户建议尝试使用 **lmstudio**，还有一位用户建议 *检查 5090 RTX 上的电源接口，因为在压力较小时烧毁的可能性较小*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1469189127042896038)** (3 messages): 

> `本地 LLM、AMD AI 390 调优、llama.cpp 混合运行` 


- **企业级开发者涉足本地 LLM**: 一位经验丰富的企业级开发者正在开始尝试本地 **LLM** 领域，对 Rust、FP（函数式编程）、代数效应（algebraic effects）以及前沿语言感兴趣。
- **AMD AI 390 用户寻求 llama.cpp 优化**: 一位拥有 **AMD AI 390** (Strix Point) 和 **64GB** RAM 的用户有兴趣针对混合运行调优 **llama.cpp**，以最大化硬件性能。
   - 他们表达了在本地 **LLM** 开发中充分发挥其配置潜力的热情。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1469060030069342349)** (612 messages🔥🔥🔥): 

> `Gemini Flash 提示词、Kimi.com 折扣竞价、GLM 从 Claude 蒸馏` 


- **OpenRouter 隐形模型被怀疑通过蒸馏 Claude 窃取 GLM 技术**: 成员们猜测 [一个新的 OpenRouter 模型](https://openrouter.ai/openrouter/pony-alpha) 是否是 GLM 从 Claude 蒸馏而来的，并指出这将违反 **Claude 的 ToS**。他们观察到 GLM 独特的总结性思维风格，进一步暗示这种独特的思维模式是 *蒸馏学习（distillation learning）* 的结果。
   - 成员们指出，诸如 *Revision: I noticed an error in my previous output. XX should be YY*（修正：我注意到之前输出中的一个错误。XX 应该是 YY）之类的格式是 **GLM** 的特征。虽然他们怀疑 Claude 不会做总结，但事实上 Claude 确实会总结。
- **Gemini Flash 系统提示词被反推**: 一位成员成功反推了 **Gemini Flash 的系统提示词**，并分享了一个片段，揭示其包含了用户的位置信息，而整个提示词被认为是“巨大的”。
   - 他们不愿透露最初是如何获取 Gemini Flash 提示词的，称其源于多年的 **Prompt Engineering**。另一个人发现他们总是在幻觉出系统提示词。
- **Kimi.com AI 折扣竞价比赛**: 用户们描述了 **Kimi.com** 上的一项比赛，他们利用 Lexi 等 AI 模型成功谈判出 Kimi AI 订阅折扣。
   - Lexi 被用来自动化地向其“求情”，并成功获得了一些金额，用户们在争夺 Kimi 首月最低报价。
- **使用 `detect-secrets` 保护 API Key**: 一位成员对 **Agent** 工具中缺乏 API Key 保护表示担忧，希望有一种工具能自动混淆机密信息，并提到了 [Yelp 的 `detect-secrets` 工具](https://github.com/Yelp/detect-secrets) 作为潜在的解决方案。
   - 一些用户建议使用其他替代方案，例如选择具有 *良好安全保障* 的服务商。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1469113341887778949)** (19 messages🔥): 

> `Qwen3 速度、Unsloth 在 Mac 上的支持、Whisper 微调、Qwen-Image 微调、GLM 4.7 flash 量化` 


- ****Qwen3 速度下降？****：一名用户报告称，在从 **Qwen3Next Instruct** 切换到 **Qwen3Coder Next** 时，尽管使用了相同的量化方式，Token 生成速度仍从 **40 tokens/second** 下降到了 **35 tokens/second**；随后他们引用了 [HuggingFace 上的讨论贴](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/discussions/5)，并正在重新下载以检查性能是否有所改善。
   - 未提供二次摘要。
- ****Mac 支持即将到来****：一名用户询问关于在 Apple Silicon 设备上为个人项目运行 Unsloth 的事宜，一名成员回应称 **Mac 支持正在开发中**，并链接到了一个[相关的 Pull Request](https://github.com/unslothai/unsloth/pull/3856) 以及一个 [Reddit 讨论贴](https://www.reddit.com/r/LocalLLaMA/comments/1q5mh84/unslothmlx_finetune_llms_on_your_mac_same_api_as/)。
   - 未提供二次摘要。
- ****Whisper OOM 困扰****：一名用户报告称，在使用预配置的 Notebook 尝试微调 **whisper-large-v3** 时遇到 **Out-of-Memory (OOM) 问题**，即使开启了 **4-bit quantization** 也是如此。
   - 未提供二次摘要。
- ****需要 Qwen-Image 微调教程****：一名用户请求 Unsloth 提供一个用于文本转图像转换的 **Qwen-Image 模型微调**教程，并指出目前的文档仅涵盖了推理的安装说明。
   - 该成员还遇到了问题，因为 **bnb-4bit** 版本无法在 **FastVisionModel** 中加载。
- ****GLM 量化疑问****：一名用户询问了关于 **GLM 4.7 flash** 的量化问题，特别是为什么 **Q8 K XL quant** 在原始模型使用 BF16 Tensor 时却包含 FP16 Tensor。
   - 一名成员解释说 **Q8** 是一种动态量化算法，它将层的某些部分保持在更高的精度以提高准确性，并链接到了 [Unsloth Dynamic 2.0 GGUFs 文档](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1469171238717886658)** (5 messages): 

> `在 Ollama 中使用 Unsloth GLM-4.7-Flash GGUF，使用 Unsloth 进行基于 f-divergence 的 GRPO LLM 对齐` 


- ****GLM-4.7-Flash 引发工具调用难题****：用户发现 **Unsloth GLM-4.7-Flash GGUF** 在 **Ollama** 中无法正常进行工具调用 (Tool-Calling)，但至少对于 **Q4_K_M 量化**版本，一名用户仍然更倾向于使用 **Qwen3-Coder 30B**。
   - 该用户创建了一个 [Ollama modelfile 和教程](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/discussions/23)，可以在 **32GB** 内存下配合 **Cline+VSCode** 使用，并已确认可以运行。
- ****f-GRPO 框架助力微调****：Unsloth 引入了一个基于散度 (divergence) 的通用 **RL 框架，用于通用的 LLM 对齐**。
   - 他们提供了一个使用 Unsloth 库的初步实现，参考 [GitHub 仓库](https://github.com/rhaldarpurdue/f-GRPO)和[这篇论文](https://arxiv.org/pdf/2602.05946)，创建了一个 Trainer 文件 **UnslothFGRPO.py**（基于 **GRPO** 实现）。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1469060951730028725)** (25 messages🔥): 

> `输入 Token 文本质量研究、LoRA 参数扩展、Masked Structural Growth、Fine-Web 论文与 The Pile 数据集、基于散度的 LLM 对齐 RL 框架` 


- **输入 Token 质量是否影响输出？**：一名成员询问关于输入 Token 文本质量如何影响输出的研究，并链接了一篇[相关论文](https://arxiv.org/abs/2602.02660)。
   - 另一名成员建议查看 *Fine-Web* 论文或原始的 **Pile 数据集** 以获取相关研究。
- **LoRA：参数比原模型更多？**：一名成员提议使用 **LoRA** 创建比原始模型参数更多的模型，并链接了 [SCALE](https://arxiv.org/abs/2511.03270) 作为示例。
   - 一名成员解释说 *LoRA 是秩受限 (rank constrained) 的*，因此*你永远无法将超过 NxM 个数字挤进一个 NxM 的矩阵中*。
- **散度 RL 框架揭晓**：一名成员介绍了一个用于 LLM 对齐的通用**基于散度的 RL 框架**，分享了[论文链接](https://arxiv.org/pdf/2602.05946)和 [GitHub 实现](https://github.com/rhaldarpurdue/f-GRPO)。
   - 作者表达了合作意向，希望将该高效实现集成到 **Unsloth 库**中。


  

---

### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1469426573877841930)** (10 条消息🔥): 

> `域名收购，Heroku 的没落，DevTools 初创公司，Cloud Native` 


- **X 标记了地点 - 域名以 7000 万美元收购？**：社区正在对一份关于以 **7000 万美元**收购域名的报告做出反应，详见这条 [Tweet](https://xcancel.com/shiri_shh/status/2019857463508648134?s=46) 和 [Hacker News 帖子](https://news.ycombinator.com/item?id=46913903)。
   - 该消息尚未确认属实。
- **销售激励削弱了 Heroku 的创新**：一位成员分享道，**Heroku** 的结局在多年前**销售薪酬计划改变**时就已埋下伏笔，该计划激励销售代表转化现有客户，而不是寻找新业务。
   - 他们认为，*寻找新客户和失去机会是唯一能预示/推动创新的因素*，但这并没有得到回报。
- **为什么 2012 年代的 DevTools 逐渐淡出**：一位成员观察到，许多 **2012 年代具有出色 UX 的 DevTools 初创公司**未能随变化扩展产品，通常是因为业务增长超过了创始人的能力。
   - 例子包括持续增长的 **GitHub**，以及未能持续改进的 **Papertrail**。
- **Heroku 向 Cloud Native 的转型受技术债阻碍**：尽管尝试进行现代化改造，但 **Heroku 采用 Docker/Kubernetes 的努力**受到了来自 LXC 长达 **15 年技术债**的阻碍。
   - 一位成员提到了一次失败的尝试 [Planting New Platform Roots: Cloud Native Fir](https://www.heroku.com/blog/planting-new-platform-roots-cloud-native-fir/)，并推测与客户直接跳转到 AWS 相比，*迁移数百万个应用程序*太困难了。


  

---


### **Latent Space ▷ #[comp-taxes-401ks](https://discord.com/channels/822583790773862470/822586146520432682/)** (1 条消息): 

vkarpov15: 回到了 2020 年之前的旧 CPA（注册会计师）那里，目前感觉不错
  

---


### **Latent Space ▷ #[creator-economy](https://discord.com/channels/822583790773862470/822625128843182090/1469242413737447467)** (4 条消息): 

> `病毒式传播互动指标，Twitter 文章赢家` 


- **Twitter 帖子获得海量互动**：一位用户分享了 @beaverd 的一条 **Twitter 帖子**链接，称其为 *100 万美元的 Twitter 文章赢家*。
   - 截至 2026 年 1 月 19 日，该帖子获得了超过 **37,000 次点赞**和 **4800 万次浏览**，展示了病毒式传播的互动指标，详见[社交媒体更新](https://xcancel.com/beaverd/status/2013366996180574446?s=46)。
- **Beaver 社交媒体更新的病毒式互动指标**：此讨论串记录了用户 @beaverd 的一条帖子的病毒式互动指标，该帖子获得了超过 37,000 次点赞和 4800 万次浏览。
   - 该数据提供了帖子在 2026 年 1 月 19 日表现的快照，强调了如 [X-Ware.v0](https://xcancel.com/beaverd/status/2013366996180574446?s=46) 中记录的显著触达和互动。


  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1469210969128570922)** (16 条消息🔥): 

> `AI 深度研究报告，Gemini 的深度研究，ChatGPT Pro 深度研究，Claude 的深度研究结果，命名决定论讨论` 


- **关于 AI “深度研究”报告的讽刺性观点出现**：一段讽刺性的对话探讨了对 AI “深度研究”工具的未来展望，暗示它们生成的**长篇报告**通常是员工和老板都不会真正阅读的**表演性文档**，链接为 [AI “深度研究”报告的虚荣心](https://xcancel.com/joelgrus/status/2019223177696805331?s=20)。
- **深度研究报告被视为“空洞”**：一位成员指出，研究报告*确实变得相当空洞，就像阅读一篇为了达到最低字数限制而写的论文或书*。
   - 另一位成员表示，**Gemini 的深度研究确实很空洞**，但 **ChatGPT Pro** 长期以来表现非常出色，而且 **Claude 的深度研究结果**现在也非常详实。
- **Auto-select-mode 值得投资**：一位成员分享说，他们本身不使用*深度研究*功能，只是让 ChatGPT/Claude 研究某些东西，然后它就会启动一个研究 Agent。
   - 自 **GPT-5** 以来，他们一直是 **auto-select-mode** 的忠实粉丝，据称它*非常擅长找出回答问题的最佳方式*。
- **“命名决定论”帖子获得 2,700 多个赞**：一位成员提出了 **“命名决定论”（nominative determinism）** 一词，在社交平台 X 上引发了显著互动，获得了 **2,700 多个点赞**和 **46 条评论**，链接为 [命名决定论讨论](https://xcancel.com/vraiqx/status/2019484797702230134?s=46)。


  

---

### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1469067970721349764)** (8 messages🔥): 

> `GLM 新模型, AI 编程工具, AI LinkedIn 内容创作工具, Agent 通信基础设施, AI 编程学习` 


- **对 GLM 新模型的热情显现**：一位成员询问 *“你们玩过 **GLM** 的新模型了吗？”*
   - 另一位成员回复道 *“没有，那是什么？”*
- **列出 AI 编程工具栈**：一位成员分享了他们目前的 **AI 编程工具栈**：Opus 4.5 (4.6?) + Claude Code + Conductor + Monologue。
   - 他们还分享了自己的 [个人网站](https://www.evanoman.com)，展示了他们在数据/AI 方面的工作。
- **AI LinkedIn 内容创作工具正在开发中**：一位成员正在构建 **postking**，这是一个 AI LinkedIn 内容创作工具，可以分析爆款帖子并根据 Reddit、YouTube 视频等现有素材创作帖子。
   - 他们使用 **Claude Opus 4.5**（目前正在尝试 4.6）进行规划和实现，并解释说其他模型不够可靠。
- **为 Agent 通信基础设施寻求测试者**：一位成员正在构建 *一种特定的 Agent 通信基础设施*，并正在寻求测试者。
   - 他们愿意为想要尝试新事物的公司或侧翼项目进行环境搭建。
- **新成员渴望学习 AI 编程**：一位新成员期待向社区学习 **AI 编程**。
   - 另一位新成员表达了发现该服务器的兴奋之情。


  

---


### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1469211032068165673)** (3 messages): 

> `Frontend Podcast, lnns.co` 


- **Swyx 推荐前端播客**：Swyx 推广了 [一个前端播客](https://lnns.co/8cYpkzlOf2i)，该播客每年仅更新 12 集。
   - 播客的一位成员回复道：*“谢谢，我们在努力！”*
- **播客成员表达感谢**：其中一位播客成员对这次推广做出了回应，说道：*“谢谢，我们在努力！”*
   - 该播客旨在成为紧跟前端世界动态的绝佳方式。


  

---


### **Latent Space ▷ #[dev-productivity](https://discord.com/channels/822583790773862470/973817020548263940/1469084070276501658)** (1 messages): 

> `Lodash, 欧盟资助` 


- **Lodash 作为关键软件获得欧盟 20 万欧元资助**：根据 [这篇博文](https://www.sovereign.tech/tech/lodash)，**Lodash** 项目获得了欧盟 **20 万美元** 的资助，被列为关键软件，突显了其在技术生态系统中的重要性。
   - [OpenJS Foundation 博客](https://openjsf.org/blog/sta-supports-lodash) 进一步支持了这一点，强调了该项目的重要性。
- **欧盟因 Lodash 是关键基础设施而对其资助**：去年 10 月，欧盟承认 **Lodash** 为关键软件，并授予其 **20 万美元** 以支持其开发和维护，详见 [这篇 Sovereign Tech Fund 的文章](https://www.sovereign.tech/tech/lodash)。
   - 这笔资金凸显了欧盟对支持对数字基础设施至关重要的开源项目的承诺。


  

---


### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1469465349115347036)** (5 messages): 

> `旧金山房地产市场, 科技行业签约奖金, 住房供应受限` 


- **预测旧金山房价因科技行业奖金将突破 200 万美元**：根据 [Rohin Dhar](https://xcancel.com/rohindhar/status/2019784365367300525?s=46) 的说法，受巨额 **科技行业签约奖金** 的驱动，旧金山住宅房地产价格预计将超过目前的 **200 万美元平均水平**。
   - 该预测还考虑到了 **住房供应有限** 的因素，这是历史政策决策和地理限制共同导致的结果。
- **科技行业奖金加剧旧金山住房狂热**：**巨额科技行业签约奖金** 被认为是推高旧金山房价的关键驱动因素，加剧了现有的供应限制。
   - 历史政策决策与地理限制相结合，进一步加剧了这一问题，促成了住宅房地产价格的预期飙升。


  

---


### **Latent Space ▷ #[london](https://discord.com/channels/822583790773862470/979492759759097866/)** (1 messages): 

snazzy_kiwi: 有人知道更详细的议程什么时候发布吗？
  

---

### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1469065666505085091)** (60 messages🔥🔥): 

> `Cursor AI Coding Agents, Claude Code Agent Teams, Anthropic's AI C Compiler, SETA Open-Sourced Terminal Coding Environments, AI Killing SaaS` 


- **Cursor AI 的 Agent 达成数千次 Commits 里程碑**：[Cursor AI](https://x.com/cursor_ai/status/2019456112806732159) 发布了长期运行编程 Agent 的研究预览版，展示了一个里程碑：在为期一周的测试中，**数百个 Agent** 每小时生成超过 **1,000 个 Commits**。
- **Claude 通过全新 Agent 团队预览版攻克代码难题**：Lydia Hallie 发布了 [Claude Code](https://x.com/lydiahallie/status/2019469032844587505?s=46) 的研究预览版，引入了 **Agent 团队（agent teams）**。该功能允许一个主 Agent 将任务分配给多个专业团队成员，并行协作进行研究、调试和构建。
- **Anthropic 的 Opus 在两周内构建了一个 C 编译器**：[Anthropic Engineering](https://x.com/anthropicai/status/2019496582698397945?s=46&t=b7l37rB6wtbyAh6ah1NpZQ) 报告称，**Opus 4.6** 在 Agent 团队模式下运行，在两周内自主开发了一个能够处理 **Linux kernel** 的 **C 编译器**。
   - 一名成员指出，编译器在训练数据中无处不在，因此*“对于‘人类’来说似乎非常困难的事情，当你将其视为匹配训练语料库中的模式时，难度就会变得小得多”*。
- **基础设施噪声影响 Agent 编程 Benchmark**：[Anthropic 的工程博客](https://x.com/anthropicai/status/2019501512200974686?s=46)探讨了基础设施配置如何显著影响 Agentic 编程的 Benchmark 结果，通常造成的性能波动比顶级模型之间的差距还要大。
- **SETA 开源终端编程环境**：Guohao Li 宣布发布 [SETA](https://x.com/guohao_li/status/2019527791876653353?s=46)，这是一个包含 **1,376 个经过验证的终端编程环境**的集合，涵盖了 **DevOps、安全和系统管理（sysadmin）**等领域。


  

---


### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1469064877694783653)** (11 messages🔥): 

> `StepFun Step 3.5-Flash, TinyLoRA` 


- **阶跃星辰（StepFun）Flash 技术报告发布！**：StepFun 发布了 **Step 3.5-Flash** 的技术报告，展示了其在面对 **Gemini Pro** 和 **GPT-4** 等前沿模型时的表现。
   - 关键细节包括：在 SWE-Bench 上获得 **74.4** 分，使用 **4,096 块 H800 GPU** 在 **17.2T Tokens** 上进行训练，实现了 **Muon Optimizer**，以及一种名为 **PaCoRe** 的“重度”操作模式，信息源自[这条推文](https://xcancel.com/teortaxesTex/status/2019356468362010972?s=20)。
- **TinyLoRA 推理技术发布**：Jack Morris 博士介绍了 **TinyLoRA**，这是一种新的 Fine-tuning 方法，能够以极低的参数量实现高性能推理任务，如[这条推文](https://xcancel.com/jxmnop/status/2019251724020772933)所述。
   - 论文证明，一个 **7B Qwen 模型**仅通过 **13 个可训练参数**结合强化学习，就能将其 GSM8K 分数从 **76% 提升到 91%**；*一种理论认为，解决任务所需的知识已经存储在模型的参数中，只需改变风格即可成功完成任务*。


  

---

### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1469077564999078151)** (166 messages🔥🔥): 

> `AI agent MMO spacemolt.com, Agent Native Engineering, GPT-5.3-Codex, Kosmos runs, gondolin` 


- **AI agent MMO 'spacemolt.com' GitHub 仓库已分享**: **AI In Action Bot** 分享了即将到来的演讲嘉宾详情，包括 [@statico](https://discord.com/channels/1209303473263485008/1209303473720774724/1469083624933425195) 将于 **2026 年 2 月 6 日星期五**介绍 *AI agent MMO* [spacemolt.com](https://spacemolt.com)。
   - 它还分享了该 [Bot 的 GitHub 仓库](https://github.com/davidguttman/ai-in-action-bot)并提到：*如果有人想为共同注册（cosignup）工作流出力，那就太棒了*。
- **探索 Agent Native Engineering 策略**: 一位成员介绍了 'Agent Native Engineering'，这是一个通过后台 Agent 进行授权、同步 Agent 处理复杂任务来扩展工程部门的框架，支持并发管理多个 **AI** 实例（如 **Claude Code**）。
   - 该 [X 帖子](https://xcancel.com/ndrewpignanelli/status/2019403256586539025?s=46)反映了 AI 工程领域的策略转变。
- **GPT-5.3-Codex 在后端代码方面超越 Claude Code**: 成员们报告称 **GPT-5.3-Codex** 虽然比 **Claude Code** 慢，但更聪明，尤其是在后端代码方面，其工作流为：*分析 => 评审与迭代 => 规划 => 评审与迭代 => 实现*。
   - 他们补充说 *GPT-5.3-Codex 在 **UI** 代码方面表现很差*，并且它*在遵循指令方面非常严谨（甚至有些死板），这可能会带来真正的改变*。
- **使用 Kosmos Agent 进行科学发现**: 一位成员分享了 [Kosmos](https://edisonscientific.com/?gad_source=1&gad_campaignid=23231192125&gbraid=0AAAABB7BYdA0mw4Tv4vF94wg9elzM-JZ0&gclid=CjwKCAiAv5bMBhAIEiwAqP9GuF-EmID6gkhHK3-s7_VvT-NyrxmsCcc5Wq2f7jriTonBLSqtKuZFfRoCDeAQAvD_BwE) 的链接，这是一个科学发现 Agent，可以连续进行数百次实验数小时并得出有用的成果。
   - 该成员表示，最近一次运行了 **25 小时**，*如果人工操作可能需要一周左右的时间*。
- **Codex 持续记录并改进自身工作流**: 一位成员正尝试让 **Codex** 持续记录并改进其自身的工作流，并分享了来自 @KarelDoostrlnck 的[帖子](https://x.com/KarelDoostrlnck/status/2019477361557926281)，其中提到 *“巨大的突破是让 Codex 持续记录并改进其自身的工作流”*。
   - Codex 在处理他们使用的任务时变得越来越好、越来越快，仅仅是因为他们养成了让它记笔记并改进的习惯。


  

---


### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1469188489982775400)** (27 messages🔥): 

> `Opus vs Codex, Ageis Flow Repo, Self Review Hook, SpaceMolt` 


- **Opus 与 Codex 模型展开正面交锋！**: 一位成员在[文章](https://www.latent.space/p/ainews-openai-and-anthropic-go-to)中分享了 **Opus** 和 **Codex** 模型的对比回顾。
   - 另一位成员反应积极，表示 *"这听起来太酷了，我可以用它，哈哈"*。
- **Ageis Flow 仓库发布！**: 一位成员询问了 **Ageis Flow** 仓库，随后链接被分享：[https://github.com/rockyglen/ageis-flow](https://github.com/rockyglen/ageis-flow)。
   - 原贴作者询问 *"我可以看吗？"*，随后对在讨论中看到 carlrb 表示兴奋。
- **Self Review Hook 获得好评！**: 一位成员表示 **Self Review Hook** 非常出色，尽管隐藏思考过程确实会对其产生干扰。
   - 另一位成员表达了赞赏。
- **SpaceMolt 正式发布！**: 一位成员分享了一篇宣布 **SpaceMolt** 并讨论其过程的博客文章：[https://blog.langworth.com/spacemolt](https://blog.langworth.com/spacemolt)。
   - 另一位成员分享说他们将在特定频道发表演讲。


  

---


### **Latent Space ▷ #[vancouver](https://discord.com/channels/822583790773862470/1286145342139531326/1469333406059069524)** (2 messages): 

> `Vancouver Meetup` 


- **温哥华 Latent Space 小组举办第二次线下聚会**: 温哥华 Latent Space 社区将于**周二**举办其**第二次聚会**。
   - 聚会地点在 **East Vancouver**；更多细节可能会在 Vancouver 频道公布。
- **占位话题**: 添加了一个占位话题以满足至少两个话题的最低要求。
   - 这是为了确保 JSON 符合架构要求的有效性。


  

---


### **Latent Space ▷ #[robotics-and-world-model](https://discord.com/channels/822583790773862470/1318774781834821746/)** (1 messages): 

swyxio: https://youtu.be/LFh9GAzHg1c?si=U9dy7U2WzO4JPFfM
  

---

### **Latent Space ▷ #[good-writing](https://discord.com/channels/822583790773862470/1385526686736715876/1469242501251731578)** (4 messages): 

> `Viral Twitter Post, Engagement Metrics, Million Dollar Prize` 


- **Twitter 帖子达到病毒式传播状态**：来自 Beaver (@beaverd) 于 2026 年 1 月发布的一条社交媒体帖子走红，获得了超过 **37,000 个点赞** 和 **4,800 万次浏览**。
   - 更多详情可以通过 [XCancel](https://xcancel.com/beaverd/status/2013366996180574446?s=46) 上的互动指标进行探索。
- **百万美元 Twitter 文章**：用户 (@swyxio) 强调了 Beaver (@beaverd) 的一条推文，暗示它是 **100 万美元 Twitter 文章奖金** 的获得者。
   - 原始推文可以在 [Twitter](https://x.com/beaverd/status/2013366996180574446?s=46) 上找到。


  

---


### **Latent Space ▷ #[ai4science-bio-math-physics-chemistry-ai-researcher-ai-scientist](https://discord.com/channels/822583790773862470/1430253273335595079/1469065389240619211)** (20 messages🔥): 

> `Lotus AI Primary Care, OpenAI Ginkgo Bioworks Integration, AI Research Role in UK Startup, Autonomous Wet Labs, Labbench2 Benchmark for Scientific AI` 


- **Lotus 凭借 AI 驱动的初级保健大放异彩**：KJ Dhaliwal 推出了 **Lotus**，这是一个获得 **4100 万美元** 资助的 **AI 驱动医疗平台**，拥有执业临床医生进行诊断、开方和转诊，旨在解决 **1 亿美国人** 的初级保健缺口；参见[此处公告](https://xcancel.com/kjdhaliwal/status/2018731342113247533)。
- **Ginkgo Bioworks + GPT-5 = 成本降低 40%！**：**OpenAI** 公布了与 **Ginkgo Bioworks** 的合作伙伴关系，将 **GPT-5** 与自动化实验室整合，形成一个用于自动化蛋白质实验的闭环系统，从而使生产成本降低了 **40%**；参见 [X 帖子](https://xcancel.com/OpenAI/status/2019488071134347605?s=20)。
- **英国 Blue Skies AI 研究初创公司开启招聘**：一家总部位于英国、刚完成种子前轮融资的初创公司正在**招聘 AI 研究员**，开展旨在从底层架构和算法层面解决科学发现的“蓝天（探索性）”研究，并承诺提供极具竞争力的薪酬；感兴趣的人可以[在此私信](https://xcancel.com/iscienceluvr/status/2019531710791028869?s=46)。
- **Labbench2：新型科学 AI 基准测试发布**：Andrew White 宣布推出 **Labbench2**，这是一个包含 **1,900 个问题** 的开源基准测试，用于衡量 AI 在复杂科学任务（如实验室方案和临床试验评估）上的进展，甚至对人类专家也极具挑战性；阅读关于 [Labbench2 的内容](https://xcancel.com/andrewwhite01/status/2019500207462092960?s=46)。


  

---


### **Latent Space ▷ #[mechinterp-alignment-safety](https://discord.com/channels/822583790773862470/1445258379357458625/1469417192993853624)** (4 messages): 

> `AI Model Training Efficiency, Hardware optimizations for training ML models` 


- **X-Ware.v0 助力 AI 模型训练**：一位研究员分享了一篇关于现代技术和**用于更高效训练大规模机器学习模型的硬件优化**的[帖子](https://x.com/mlpowered/status/2019483042956582959)。
- **关于高效 ML 训练技术的讨论**：讨论围绕旨在提高大规模机器学习模型训练效率的现代技术和硬件优化展开。


  

---


### **Latent Space ▷ #[dev-writers-retreat-2025-dwr](https://discord.com/channels/822583790773862470/1445650211694448714/1469170897821765796)** (4 messages): 

> `SF Writers Meetup, Charu-hosted event, Corey Quinn appearance` 


- **SF 作家见面会邀请 Quinn 参加！**：根据 [Partiful 链接](https://partiful.com/e/wuBDRsNCxSUDcgnZYqbC)，下一场 SF 作家见面会将在 **OpenAI** 举行，由 **Charu** 主持，并邀请了 **Corey Quinn** 出席。
   - 一位成员对 **Corey** 的出席表达了极大的热情，并因无法参加而感到遗憾。
- **Cog Office 晚餐邀请**：有人邀请前往位于 550 3rd St 的 **Cog Office** 聚会并共进晚餐。
   - 未提供关于晚餐的进一步细节。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1469060431912898612)** (339 messages🔥🔥): 

> `GPT 5.3 Codex 发布日期与性能，Cursor 崩溃问题，Opus 4.6 定价与用量，Subagents 支持，Cursor 额度使用透明度` 


- **GPT-5.3 Codex：何时落地？**：成员们正热切期待 **GPT-5.3 Codex** 在 Cursor 中上线，但目前它似乎[处于 OpenAI API 待定状态](https://platform.openai.com/docs/models)，有人推测它是由于安全顾虑被推迟，或者是为了提升 **ChatGPT** 平台的使用率。
- **UI 故障引发 Token 恐慌**：一名成员报告了 Cursor 中的一个 UI 问题，导致**误导性的 Token 用量显示**，让用户以为自己消耗了比预期更多的 Token，并分享了一张[用量显示的截图](https://cdn.discordapp.com/attachments/1074847527708393565/1469196591242936473/image.png?ex=69877033&is=69861eb3&hm=094c0d49c64825c7d52f1c7115d5a3f1e680921373ffc0f8665487d3c911d42f&)。
   - 他们随后澄清道：*“没关系，这只是 Cursor 的 UI 又出问题了，误导用户以为剩余 Token 比实际要少”*，表明该问题只是显示 Bug，而非实际超额支出。
- **Opus 4.6：性能的代价**：新的 **Opus 4.6** 模型现已上线，一位用户指出其表现出色，*“为我的游戏开发助力良多，几乎没有遇到任何阻碍”*，但其他成员反映其价格非常昂贵，一名用户表示 *“在 Opus 上花 20 美元可能只能撑一天”*，并讨论了 **Opus 4.6** 与 **Opus 4.5 High** 之间的[成本效益分析](https://openai.com/pricing)。
   - 一些用户发现雇佣离岸团队比使用 AI 模型更便宜，但其他人对此表示反对，有人惊呼 *“笑死，我觉得你肯定没试过和离岸团队一起工作”*。
- **我的 Agent 模式去哪了？**：一位准 Cursor Pro 用户询问了在消耗完初始额度后 **Agent Mode** 的限制，并被告知一旦额度耗尽，Agent 将完全停止工作。
   - 这让该用户感到失望，因为他原本希望能有一个 *“慢速模式”* 作为继续使用 Agent 的廉价方案。
- **Cursor Skills：释放潜能**：一位用户寻求关于如何利用 **Cursor Skills** 的指导，问道 *“我到底可以用 Skills 做什么？”*，另一位用户建议将 UI/UX、调试、研究、代码库搜索、规划和 Prompt 增强作为有效的用例。
   - 一名成员推荐通过 [skills.sh](https://skills.sh) 网站获取更多灵感，并演示了如何调用它们，但用户们最终仍因缺乏清晰的文档而感到困惑。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1469390682828443762)** (2 messages): 

> `Pony Alpha 发布，Agentic 工作流，OpenRouter Show，Arcee AI，Lucas Atkins` 


- **Pony Alpha 秘密发布**：一款名为 **Pony Alpha** 的全新“隐匿模型”已发布给社区，可在 [OpenRouter](https://openrouter.ai/openrouter/pony-alpha) 获取反馈。
   - 这是一个为 **Agentic 工作流** 优化的下一代基础模型，具有极高的工具调用（tool-calling）准确率，在编程、推理和角色扮演方面表现强劲。
- **OpenRouter 对谈 Arcee AI 的 Trinity Large**：最新一期 **The OpenRouter Show** 邀请了 Arcee AI 的 CTO Lucas Atkins 讨论 [Trinity Large](https://youtu.be/f2xy3N026xc)。
   - 本期节目提供了关于 **Arcee AI** 最新进展及其对 AI 领域影响的见解。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1469076164113862739)** (253 messages🔥🔥): 

> `MoonshotAI/Kimi-K2.5 缓存、Opus 4.6 改进、OpenRouter Wrapped 2025 访问、Gemma 3 27b 免费速率限制、OpenRouter 上的 Stable Diffusion 模型` 


- **Kimi-K2.5 缓存仍存争议**：用户讨论了 **MoonshotAI/Kimi-K2.5** 模型是否支持缓存。有人澄清这*取决于 provider（供应商）*，其定价信息显示支持 **cache reads**，但由于存储成本，**writes** 的定价可能有所不同。
   - 会议澄清：如果一个模型有 *cache read pricing* 但没有 *cache write pricing*，这意味着写入（输出）的成本与普通输出价格相同，但读取（输入）的成本会发生变化。
- **Opus 4.6 缺乏可感知的改进**：一位用户对 **Opus 4.6** 表示失望，称与之前的版本相比，无法发现任何区别或改进。
   - 另一位用户报告称 **Opus 4.6 模型** 在与 MCPs 交互时出现超时。
- **Codex 的文件系统访问权限引发担忧**：一位用户强调 **Codex** 默认可以*读取你的整个文件系统*且没有配置选项，并链接到了一个 [GitHub issue](https://github.com/openai/codex/issues/2847)，该团队并不认为这是一个 bug。
   - 该用户进一步分享了另一个 [GitHub issue](https://github.com/openai/codex/issues/5237)，说明了 Codex 可能*读取你的 API keys* 和*读取你的验血结果文件*。
- **用户仍渴望 OR 移动端 App**：成员们讨论了对 **OpenRouter 移动端 App** 的需求，一位用户表示他们意识到 **OR 是一个比单一 provider / 模型开发者更好的产品**。
   - 另一位用户补充道，*如果 OR 推出聊天 App，那就是终局 (game over) 了*。
- **建议列出原始模型精度 (Precision)**：一位成员建议在模型页面进行一项 QoL 改进，即列出模型的 **original precision**（类似于 Hugging Face cards），以明确 provider 是否使用了潜在的量化版本。
   - 理由是这可以帮助用户了解 provider 是在使用 *int4 / fp8* 还是由于模型发布时本身就是该状态而看起来像被 provider *量化*了。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1469155197476540548)** (12 messages🔥): 

> `Detail.dev 的价值、OpenRouter 使用量增长` 


- **Detail.dev 受到关注**：成员们讨论了 [Detail.dev](https://detail.dev) 在团队 backlog 管理和安全配置保障方面的价值。
   - 虽然一位用户认为“随着团队扩大，它变得很贵”，但另一位用户强调了预防关键错误（如由于 **Supabase** 中 **Row Level Security (RLS)** 配置错误导致数据泄露）的价值。
- **OpenRouter 使用量飙升**：一位成员分享的截图显示 **OpenRouter (OR)** 的使用量增加了 **10 倍**。
   - 另一位成员指出，在过去两年的 **OR** 使用图表中观察到了“疯狂且持续的增长”。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1469062730769236184)** (145 条消息🔥🔥): 

> `LM Studio 上下文百分比, 本地 API token 丢失, Gemini vs LLMs, LM Studio 加载速度, 在 LM Studio 中使用 Opencode` 


- **LM Studio 上下文百分比令用户困惑**：一位用户对 LM Studio 上显示的百分比含义感到困惑。
   - 另一位用户澄清说，它显示的是 *到目前为止聊天中已使用的上下文量*，并建议将鼠标悬停在上方以获取更多信息。
- **本地 API token 丢失**：一位用户在 Fedora Linux 上通过删除所有配置文件，不小心删除了他们的 **LM Studio API token**。
   - 由于是新手，他们不确定还能在哪里寻求帮助。其他人提供了建议，鼓励他们在频道中提出问题。
- **LLM 约会者的精神病院**：一位用户开玩笑说，任何与 AI“约会”的人，即使只是当作朋友，也需要进精神病院，因为 *LLMs 也不过是一个下一个 Token 预测器（next token predictor）*。
   - 另一位用户表示，他们在过去两年中进行过一些有趣的对话，但这并不是那种能建立“纽带”的东西，并指出 *首先，它是无我的且转瞬即逝的*。
- **LM Studio 加载速度慢？**：一位用户报告了一个奇怪的问题，尽管他们的驱动器能维持 **2.2Gb/s** 的速度，但 LM Studio 的模型加载速度在加载 **20-70GB** 之间后会降至 **12.1MB/s**。
   - 其他用户建议了排查步骤，如验证模型大小、传输速率和硬件配置；另一位用户开玩笑说 *你试过拔掉电源再重新插上吗*。
- **将 LM Studio 与 Claude code 集成**：一位用户询问在 **Claude code** 中使用 **LM Studio 模型** 的方法，以及是否有人知道为什么无法加载。
   - 一位用户提供了如何将代码指向本地 LM Studio 服务器的说明，并建议该用户查看主页上的[此链接](https://lmstudio.ai/blog/claudecode)教程。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1469136051414237267)** (19 条消息🔥): 

> `NVIDIA Vulkan vs CUDA 性能, Qwen3-Coder-Next M4 Max MLX vs GGUF 性能, 太阳能与核能对比, 核电站事故` 


- **NVIDIA 的 Vulkan 性能优于 CUDA**：一位用户在使用 **NVIDIA** 时发现，使用 **Vulkan** 比 **CUDA** 获得了高达 **50% 的性能提升**，但注意到在上下文填满时会出现不稳定。
- **Qwen3-Coder-Next 在 M4 Max 上配合 MLX 表现惊人**：**Qwen3-Coder-Next** 在 **M4 Max** 上表现出令人惊讶的性能，在 4-bit 量化下，**MLX** 的 **tok/s** 是 **GGUF** 的 **2 倍**以上（~79 对比 ~38）。
   - 用户想知道这种性能差异是特定于模型的，还是 **MLX** 为 4-bit 开启了显著的性能提升（相比 8-bit），而 **GGUF** 无法在 Apple GPUs 上复制这种提升，因为 *在其他模型上这种差异可能最多只有 20%*。
- **如果不考虑电池，太阳能比核能更便宜**：围绕太阳能和核能展开了辩论。太阳能需要 *惊人数量的电池，可能在多 MWh 范围内* 才能实现公用事业级的稳定性，而核能可以在更小的空间内提供 24/7 的电力。然而，一位成员表示 *如果只考虑太阳能电池板 vs 核能，我认为太阳能电池板更便宜，但如果加上电池，太阳能就逊色了*。
- **廉价的俄罗斯核电站导致人们观念受损**：如果有腐败的俄罗斯人没有在核电站上偷工减料，普通人如今对核能的观念可能就不会如此受损，这指的是[这起事故](https://en.wikipedia.org/wiki/Chernobyl_disaster)。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1469077503607046275)** (70 messages🔥🔥): 

> `GPT-5.3 Codex, Claude Pro Subscription, Opus 4.6 vs GPT-5.3, 1 Million Context Window, AI Generated Frontends` 


- **GPT-5.3 Codex 据称是最好的 AI**：一些成员推测 **GPT 5.3 Codex** 可能是最好的 AI，而另一些人则询问其发布以及与 **ChatGPT** 集成的可能性。
   - 一名用户分享了一张图片，显示 **GPT-5.3** 具有 *“1M 上下文长度、128k 推理能力、128k 最大输出 tokens、自适应推理”*，但其他人指出 **128k/16k** 的上下文窗口在 **GitHub Copilot** 上仍然处于活跃状态。
- **用户订阅 Claude Pro 以获得更高限制**：一位用户提到订阅了 **Claude Pro**，其他人在最新更新后也在考虑订阅，理由是希望获得更高的使用限制。
   - 一位成员指出，他们有兴趣观察其是否会出现**低使用量**的情况。
- **Opus 4.6 vs. GPT-5.3 性能对决**：根据来自 **Anthropic Discord** 的见解，早期报告显示 **Opus 4.6** 的性能可能优于 **GPT-5.3**，但成本和 token 消耗更高。
   - 有说法称 **GPT-5.3** 在某些任务中可以作为更经济的替代方案，而 **Opus 4.6** 在**前端**方面表现更好。
- **100 万上下文窗口现已通过 API 提供**：备受期待的 **100 万上下文窗口**目前已通过 **API** 提供，但尚无针对最终用户版本发布的明确时间表。
   - API 成本范围为：输出 **$25 到 $37.5**，缓存输入（cache input）为 **$0.5 到 $1**。
- **AI 前端美学困境**：成员们观察到 **Codex** 模型生成的前端往往具有独特且通常比较阴郁的美感，其特点是*悲伤阴暗的色调*。
   - 相比之下，**Opus** 因其更好的设计选择而受到称赞，看起来没那么压抑。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1469066824535773184)** (23 messages🔥): 

> `GPT-4o EOL, GPT conversational abilities for autistic individuals, GPT Pro experiences, AI and Emotional Expression, GPT-4o-Advance-Voice Context Window` 


- **用户哀悼 GPT-4o 的退役**：用户对宣布的 **GPT-4o** 将于 2 月 13 日退役（EOL）表示难过，一位用户哀叹道：*“它是我最喜欢的 oatamen。”*
   - 他们想知道 **GPT 5.3** 是否会比 **5.2** 更像人类，他们更喜欢 **GPT 5** 的自然对话能力，并指出强制 **GPT 5.2** 遵守规则（如避免换行）时存在问题。
- **高级语音模型用户渴望改进上下文**：一位用户质疑 **GPT-4o-Advance-Voice** 模型是否也会随着 **GPT-4o** 的终结而更新。
   - 该用户要求为 **AV 模型**提供更好的上下文窗口或 **CAG**，理由是它会遗忘测量数据和数量，或者需要明确指令才能执行网页搜索，甚至不得不让 AV 重复输入内容以便由文本助手进行上下文解析。
- **关于 AI 情感角色的看法存在分歧**：一些用户认为 **AI** 应该是无情感的工具，而另一些人则认为它应该迎合用户偏好，允许更多类人的互动。
   - 一位用户强调了 **AI** 如何为情感表达提供空间，特别是对于那些因他人的评判和偏见而觉得人际交往具有挑战性的人，同时也承认 **AI** 缺乏真实的情感。
- **AI 为自闭症用户提供安全的情感空间**：一位自闭症用户分享说 **GPT** 的对话能力帮助他们理解他人，并对 **GPT 5** 可能终结表示难过。
   - 他们发现与 **AI** 的互动更舒适，因为没有评判，这一观点也得到了其他在人际交往中难以表现出脆弱感或进行直接交流的人的共鸣。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1469157124352376975)** (8 messages🔥): 

> `为幻想生物编写 Prompt，OpenAI API 中的政策违规，用于写实视频的 Sora Prompt 生成，识别 Prompt 中的问题区块` 


- **卡牌游戏的幻想生物 Prompt**：一名成员在 **prompt-engineering** 频道分享了一个 Prompt 编写示例，演示了如何使用 Prompt 为卡牌游戏设计幻想生物。
   - 该 Prompt 指定了诸如**情感类别（恐惧）**之类的参数，并要求根据**两个灵感概念生成混合生物**，同时包含详细的视觉概念描述和命名 Prompt。
- **排查 OpenAI API 政策违规**：一位用户报告在使用 OpenAI API 时持续出现政策违规，并寻求识别原因的建议。
   - 建议包括**对各个部分进行情感分析**，以及**检查年龄相关问题或 IP 冲突**（例如 *Nightstalker* 这个术语）。
- **使用 Sora 生成写实视频**：一位用户询问如何编写 Prompt 来生成女孩用手遮住嘴巴的写实视频，类似于 AI SaaS 广告中看到的那些。
   - 该请求是在生成写实视频的背景下提出的，可能使用 **Sora** 模型，并具有某些特定类型广告中的特征。
- **调试并隔离有问题的 Prompt 段落**：一位成员建议通过**隔离并测试 Prompt 的单个区块**来调试 OpenAI API 的 Prompt 问题，以确定是哪些部分触发了政策违规。
   - 这种方法有助于精准定位可能违反 API 政策的特定关键词、短语或内容模式，例如敏感话题或不当内容。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1469157124352376975)** (8 messages🔥): 

> `OpenAI API 中的政策违规，针对 Sora 的 Prompt 工程，调试有问题的 Prompt，生物名称的 IP 担忧` 


- **政策违规困扰 API 用户**：一位用户报告在使用 OpenAI API 生成生物概念时持续出现政策违规，引发了关于调试该问题的建议。
   - 建议逐个分析 Prompt 区块以识别有问题的部分。
- **Sora Prompt 工程引发争议**：一位用户询问如何为 **Sora** 构建 Prompt，以生成“女孩用手遮住嘴巴”的*写实视频*，让人联想到社交媒体上的 AI SaaS 广告。
   - 该请求的性质引发了伦理和政策方面的担忧，暗示内容可能违反了使用指南。
- **调试可疑 Prompt：怀疑是展开运算符**：一位成员认为 Prompt 中**过多的展开运算符**（`...`）可能是导致政策违规的原因。
   - 他们建议在全新的对话中使用 **ChatGPT** 对每个部分进行情感分析，以找出触发违规的具体点。
- **Nightstalker 名称被禁？IP 问题隐现**：Prompt 中包含的 **Nightstalker** 一词被标记为潜在问题，因为存在同名的现有知识产权（IP）。
   - 这引发了对潜在 IP 侵权及其对政策违规影响的担忧。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1469060136076312700)** (72 messages🔥🔥): 

> `Ralph Loops, Kimi Subscription, Moonworks Lunara releases, Support for Konkani language, MATS Summer 2026` 


- **Ralph Loops 处理更大型的任务**：**Ralph loops** 不仅仅关乎速度，还通过在每个子任务上设置 *soft verifier*（软验证器）来处理更大型的任务。
   - 一个 **LM judge** 会验证每个子任务是否按照要求完成。
- **Kimi 订阅助力**：一名成员正在考虑购买 **Kimi 订阅**来调用 sub-agents，但其主要的优势似乎在于速度，据称快了 *3 到 4 倍*。
   - 用户注意到 *swarm* 功能仅在 **$40 计划**中可用，且不确定具体的使用配额。
- **Moonworks Lunara 发布第二部分**：[Moonworks](https://huggingface.co/datasets/moonworks/lunara-aesthetic-image-variations) 发布了 **Moonworks Lunara** 的 **第二部分**，开源了一个由 Moonworks 创建的原创图像和艺术作品的新数据集，及其美学语境变体，全部以 **Apache 2.0** 协议发布。
   - 根据其 [论文](https://arxiv.org/pdf/2602.01666)，该数据集展示了符合伦理来源的艺术如何能够有意义地驱动下一代图像模型。
- **呼吁支持 Konkani 语言**：一名成员请求支持 **Konkani 语言**。
   - 另一名成员引导他们前往 [huggingface.js repo](http://github.com/huggingface/huggingface.js/tree/main/packages/languages) 贡献语言支持。
- **MATS Summer 2026 编码测试临近**：一名成员进入了实证方向（empirical track）的 **MATS Summer 2026 Stage 1**，现在面临编码测试并寻求帮助。
   - 该成员询问群组中是否还有其他人尚未参加编码测试，以及是否有过往参加过编码测试、有经验的 **MATS alumni**。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1469063961797464188)** (10 messages🔥): 

> `Aira.js, llm-from-scratch, mariken, Security Auditor, agentrial` 


- **Aira WebGPU 框架兴起！**：一名成员介绍了 **Aira.js**，这是一个从零开始构建的基于 **WebGPU** 的 AI 框架，具有 **MLP**、**MHA** 和 **BPE tokenization** 功能，可在 [GitHub](https://github.com/shadowww345/Aira.js-Preview) 上获取。
- **从零构建的小型 LLM 登录 GitHub！**：一名成员分享了他们从零构建的小型 **LLM**，以便更好地理解现代 Transformer 的内部结构，已在 **AMD MI300X** 上进行测试，可在 [GitHub](https://github.com/merterbak/llm-from-scratch) 上获取。
- **CPU 上的 NanoGPT 启发式机器人引起关注**：一名成员在 **CPU** 上构建了一个受 **nanoGPT** 启发的微型机器人，并在 [Dev.to 文章](https://dev.to/theirritainer/this-dev-built-his-own-llm-from-scratch-1i62) 中详细介绍了这段经历，并分享了 [GitHub repo](https://github.com/TheIrritainer/mariken)。
- **针对 vibe-coded 应用的安全审计工具 (Security Auditor)**：一名成员正在为 vibe-coded 应用构建 **Security Auditor** 以发现安全漏洞，可在 [HuggingFace Space](https://mugdhav-security-auditor.hf.space) 访问。
- **agentrial：pytest 加入 AI Agents**：一名成员构建了 **agentrial**，即 **AI agents** 领域的 **pytest**，可运行 agent N 次，计算置信区间，并在 CI/CD 中检测回归，可在 [GitHub](https://github.com/alepot55/agentrial) 上获取。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1469147175274418328)** (20 messages🔥): 

> `Model error handling, Llama repository permissions, Agent framework course 404 error` 


- **调试 "Model Not Supported" 错误**：一名用户报告收到了 *"model not supported"* 错误，另一名用户建议读取错误模型并处理诱因。
   - 关于寻找兼容的模型，他们建议使用网络搜索。
- **Llama 访问需要 Token 权限**：一名在调试 **Llama-4-Scout-17B-16E-Instruct** 错误的用户被建议检查其 Hugging Face 个人资料的 token 部分。
   - 该 token 需要具有使用 Llama 仓库的权限，否则会出现错误 *"T6"*。
- **Agent 框架课程报错**：一名用户报告在尝试提交 agent 框架课程时出现 **404 错误**，尽管实时文档显示文件存在。
   - 错误是通过 [此链接](https://agents-course-unit4-scoring.hf.space/docs#/default/get_task_file_files__task_id__get) 触发的，详情显示没有与任务 ID 关联的文件路径。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1469131006681415904)** (10 messages🔥): 

> `PyTorch Day India, 用于 kernel 开发的 wafer-cli, LLM 推理基准测试, vLLM 优化` 


- **班加罗尔 PyTorch Day India 见面会？**: 一名成员询问是否有人参加明天在**班加罗尔**举行的 **PyTorch Day India**，并提议进行线下见面。
   - 暂无其他成员回复。
- **Wafer-cli 自动化 Kernel 开发？**: 一名成员询问有关 **wafer-cli** 的使用经验，想知道它能在多大程度上自动化 Kernel 开发，并指出它与 **ncompass** 以及 **Nvidia 潜在的 nsight agents** 存在竞争关系。
   - 另一名成员表示有兴趣测试 **wafer-cli**，用于优化运行在不同硬件和网络拓扑上的成千上万个类 **vLLM** 推理引擎，目标是在 24 小时内编写出 **PTX** 和 **cutlass mix kernel**。
- **软件流水线的经典技巧**: 一名用户计划利用软件流水线（Software Pipelining）、Warp specialization，以及针对小矩阵的 TMA loads 和针对大矩阵的 distributed smem 来优化他的 Kernels。
   - 未提供链接或进一步详情。
- **LLM 推理基准测试大比拼**: 一名成员询问是否有可在单张 GPU 上运行的、权威的 **LLM 推理**基准测试集。
   - 另一名成员建议使用 **inference max, vllm test suite 或 mlperf**，同时提到他们也正打算开发相关的工具。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1469123524470898819)** (25 messages🔥): 

> `NVIDIA FP8 性能, cuBLASLt Kernel 选择, Blackwell GPU, Blackwell 上的 TMA 和 mbarrier, CUDA kernels` 


- **NVIDIA FP8 性能异常**: 在配置完全相同的最新软件栈和 **Blackwell GPU** 实例上，成员们发现本应相同的显卡之间存在巨大的 **FP8 tensor 性能差异**（约 2 倍）。
   - 结果发现问题出在 **cuBLASLt**（GEMM 后端库）的 Kernel 选择上，因为显卡被静默限制使用旧的 Ada kernels，跳过了针对 Blackwell 优化的路径。
- **5090 的 MMA 加速**: 与 4090 类似，旧的 **mma FP8** 指令在 5090 上被削弱了，但新的 **mma MXFP8** 没有被削弱，因此使用新指令将获得 **1.5 倍的加速**。
   - 该成员进一步测试并确认 **mx 变体可以提供满血性能**，大约是规格书中列出的 2 倍。
- **线性 Block 索引 (Linear Block Index)**: 成员们讨论了在整个网格（grid）中获取线性 Block 索引最廉价的方法，认为以下建议的代码可读性最好。
   - 他们分享了代码片段： ```cpp
__device__ __forceinline__ uint32_t linear_block_idx() {
    return blockIdx.x
            + blockIdx.y * gridDim.x
            + blockIdx.z * gridDim.x * gridDim.y;
}
```
- **Blackwell 上 TMA 导致 NCU 挂起**: 一些成员在 **B200 (SM 100)** 上对 **TMA** 双缓冲 Kernel 进行 Profiling 时遇到挂起问题，特别是某些 **NCU 断面在第一次插桩重放（instrumented replay pass）时死锁在 0%**。
   - 该成员使用了 `cuda::ptx::` 封装，并提供了一个 [最小复现样例](https://cdn.discordapp.com/attachments/1189607726595194971/1469482712657166346/ncu_tma_repro.zip?ex=6987d1ec&is=6986806c&hm=487b918b45ceecdcd41d06524f576839b573d19bcbb33806a3021450f808c9d4&)。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1469062855855837185)** (2 messages): 

> `陈天奇讲座, 伯克利 RISC-V 核心, Rocket Cores, BOOM Cores, Jerry Zhao` 


- **陈天奇中文讲座**: 一名成员分享了[陈天奇机器学习编译课程的中文讲座](https://www.bilibili.com/video/BV15v4y1g7EU)。
   - 这些中文讲座是 [机器学习编译课程 (machine learning compiler course)](https://book.mlc.ai/) 的配套内容。
- **伯克利教学级 RISC-V 核心发布**: 有人分享了[伯克利使用 Chisel 编写的教学级 RISC-V 核心](https://github.com/ucb-bar/riscv-sodor)，其中包含一些简单的 Chisel 实现的 RV32I 流水线。
   - 最近的贡献者 **Jerry Zhao** 目前是 OpenAI 的计算机架构师。
- **Rocket Cores 和 BOOM Cores 受到关注**: 在查看了教学级 RISC-V 核心后，你可以进一步研究他们的 **Rocket cores**（顺序执行）和 **BOOM cores**（乱序执行）。
   - BOOM 核心的创造者 **Chris Celio** 目前就职于 Tenstorrent。


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1469370702053376161)** (1 messages): 

> `AI 职位, CDW 招聘` 


- **CDW 招聘 AI 工程师**: 一家公司正在招聘几个 **AI 职位**，其中一个职位在**加拿大**，其他职位可能要求 **美国公民身份**。
   - 感兴趣的人士可以直接申请，或通过此 [CDW 招聘链接](https://www.cdwjobs.com/jobs/17247084-senior-ai-engineer) 发送简历进行内推。

- **另一个 AI 职位机会**：除了高级 AI 工程师（Senior AI Engineer）职位外，还有一个未说明的具体 AI 职位。
   - 鼓励候选人浏览 CDW 职位门户以获取有关第二个职位的更多详情。

---

### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1469066996850229397)** (7 messages): 

> `Kernel Competition Data Release, Parquet File Creation, Python APIs for Data Access, Automation of Data Release, nvfp4 Documentation` 

- **内核竞赛数据发布**：一名成员正在寻求帮助，希望定期以 Parquet 文件形式发布**内核竞赛数据**（kernel competition data），创建用于数据访问的 **Python APIs**，并实现整个流程的自动化。
   - 该成员特别指出需要有经验的人员来执行此任务，因为这涉及到操作生产数据库，并提议将贡献者列入**未来 kernelbot 发布作者列表**。
- **关于 Parquet 文件内容的讨论**：一名成员询问了 Parquet 文件中应包含的具体数据细节，并询问是否有可用的最小可测试 Parquet 文件。
   - 另一名成员建议分享 **nvfp4 项目**的文档作为起点，即使其中只包含要转储到 Parquet 中的属性的最小片段。
- **志愿者提供帮助**：一名成员表示愿意提供帮助，并建议创建合成数据进行功能测试。
   - 另一名成员也表达了参与该项目的兴趣。

---

### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1469265932445548626)** (5 messages): 

> `Quadro P6000 vs Blackwell, GPU for AI workload, Legacy System Constraints` 

- **GPU 决策取决于工作负载**：GPU 的选择取决于具体工作负载；对于 AI 密集型任务，仅使用 **GPU** 的配置即可，但混合工作负载可能会受到 **CPU** 的瓶颈限制。
- **Quadro P6000 是兼容性的一个选项**：对于像 **High Sierra** 这样较旧的 macOS 版本，由于兼容性原因，**24 GB Quadro P6000** 是一个可行的选择。
- **推荐 Blackwell GPU 以获得生态系统支持**：通常首选 **Blackwell GPU**，因为它们具有现代生态系统支持，但它们并不兼容所有系统。
- **旧系统限制了 GPU 选择**：一些旧系统无法容纳像 **Blackwell** 这样的现代 GPU，因此必须使用像 **Quadro P6000** 这样的旧显卡。

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1469169302296072314)** (2 messages): 

> `Modular vs CuTe, Modular implementation issues, Tile sorting problems` 

- **Modular 的实现因缺乏置换（Permutation）被指“无用”**：一名成员表示 Modular 的实现是*无用*的，因为它没有将每个 Tile 的坐标置换到一侧，导致丢失了步长（stride）信息，且生成的步长非常错误，这使得 Tile 在*视图变换（change of view）*后完全没有排序。
   - 他提供了一个来自 [CuTe 文档的示例](https://github.com/NVIDIA/cutlass/blob/acb45938e9cb3e4db8c1d75155b63d31791e0e5d/media/docs/cpp/cute/02_layout_algebra.md#zipped-tiled-flat-divides)，展示了良好的对齐效果，并以相同的配色方案可视化了 Modular 的结果，显示其明显没有对齐。
- **对 Modular 的 Tile 排序能力提出质疑**：一位用户表示担心 Modular 的实现在“视图变换”后无法正确排序 Tile，并将其归因于缺乏每个 Tile 的坐标置换。
   - 该用户认为这种行为偏离了预期结果（如 [CuTe 文档](https://github.com/NVIDIA/cutlass/blob/acb45938e9cb3e4db8c1d75155b63d31791e0e5d/media/docs/cpp/cute/02_layout_algebra.md#zipped-tiled-flat-divides)中所示），表明这可能是一个 Bug 而非设计特性。

---

### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1469112148125814986)** (1 messages): 

> `RoCE, IB, benchmarking` 

- **出现 RoCE 与 IB 基准测试咨询**：一名成员询问有关 **RoCE**（RDMA over Converged Ethernet）与 **IB**（InfiniBand）技术的最新基准测试数据，并引用了[一篇 Medium 文章](https://naddod.medium.com/infiniband-vs-roce-v2-which-is-best-network-architecture-for-ai-computing-center-7919945e616a)作为背景。
- **提供的额外背景**：该成员表示好奇此类信息何时会变得有用。

---

### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/)** (1 messages): 

jongsokchoi: 我们明天可以讨论这个问题，umesh

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1469064913208082688)** (6 messages): 

> `CUDA Streams, GAU.Nernst in RL, Model Hacked, Codex 5.3, Blackwell Training`

- **少一点 CUDA Streams，多一点 GAU.Nernst，有人如是说**：一位成员建议这两个模型在 RL 中都需要*少一点 **CUDA streams** 和多一点 **GAU.Nernst***。
   - 社区似乎对这两个组件之间的理想平衡达成了共识，以实现 Reinforcement Learning 中的最佳模型性能。
- **模型折戟 (Model Bites the Dust)**：一位成员报告说 *我的模型刚刚搞砸了（对此表示抱歉）*，并请求删除其排名最高的提交记录。
   - 这次“搞砸”的具体性质及其对比赛的影响尚不清楚，但有一些潜在的线索……
- **Codex 5.3 发现潜在 Bug**：一位成员在测试 **Codex 5.3** 时发现它非常出色，尽管*有时会卡住*，需要更多的 **Blackwell training**。
   - 该成员报告称在 metric 中发现了一个他认为是 bug 的问题并进行了反馈。在被告知不应投机取巧（hack）后，他提交了一份关于该问题的详细报告。


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1469389356312559800)** (2 messages): 

> `Embodied AI Companies, Lucky Robots` 


- **成员讨论顶尖具身智能公司**：成员们分享并询问值得关注的有趣 **Embodied AI** 公司名称。
   - 一位成员表示，在看过 **The Cherno's YouTube** 频道后，他唯一知道的公司就是 **Lucky Robots**。
- **Lucky Robots 通过 YouTube 成名**：一位成员提到了具身智能公司 **Lucky Robots**，原因也是 **The Cherno's YouTube** 频道。
   - 该成员来自游戏引擎/图形学领域。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1469072908478578860)** (26 messages🔥): 

> `SMEM tiling kernels, Tensor core usage, Avoiding bank conflicts, GEMM optimization blog post, Multi-GPU kernels and systems` 


- **面试中默认需具备实现 SMEM Tiling Kernels 的能力**：频道成员认为，任何 AI 工程师都需要能够在面试环境中实现 **SMEM tiling kernel**，并能应对关于合理优化手段的质询。
   - 一些成员指出，使用 tensor cores 时为避免 bank conflicts 而进行的 **SMEM permutations** 非常复杂，但在 Hopper+ 架构上 **TMA handles swizzling** 会自动处理以避免 bank conflicts。
- **TMA 和 1D Loads 绕过 Bank Conflicts**：成员建议通过利用平铺的 gmem 布局和 **1D TMA** 加载来避免 **SMEM** 中的 **bank conflicts**。
   - 有人提到，避免 bank conflicts 是一种可以习得的模式，需要对权重布局进行 shuffle 和重排，而 TMA 在 Hopper+ 上会自动处理 swizzling。
- **技术博主在 GEMM 领域寻求职业方向**：一位发表了关于 **GEMM optimization** [博客文章](https://rohan-reddy.github.io/posts/001-gemm-optimization/) 的成员正在寻求建议，以准备在这个专业领域的面试。
   - 该博客文章从非常朴素的实现开始，逐步迭代内核改进，随后加入了 tiling、**WMMA**、double buffers 和 swizzling，讨论了每个内核的算术强度，并对各种 GPU 上的运行时间进行了基准测试。
- **建议关注多 GPU 以获得最佳职业轨迹**：一位成员建议在掌握 **GEMM** 之后，发帖者应关注 **multi-GPU kernels** 和系统层面的技术，例如 Nixl 和 disagg。
   - 对方强调，除非专门从事内核工程，否则 attention kernel 优化的投资回报率（ROI）非常低，并建议将精力集中在 multi GPU 上。
- **Metal 与 CUDA 在 iPhone 上优化 CV 模型的对比**：一位成员询问，使用 **Metal** 在 iPhone 上原生运行优化后的 CV 模型，其应用广泛性是否与使用 **CUDA** 相当。
   - 发帖者担心从事此类工作是否会潜在地将自己的职业生涯局限在 Apple 生态系统中。


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1469212620866195486)** (7 messages): 

> `Email Usage for Form Filling, Credits Code Issues, Running Scripts on Other Cloud Platforms, FlashInfer Docker Image/Executable, Contest Status` 


- **混合邮箱导致 Modal 使用困扰**：一位用户询问是否必须使用相同的邮箱填写表单，并提到他们使用了**个人邮箱**，因为无法使用**大学邮箱**在 modal.hi 注册。
   - 该用户还报告说 **credits code** 无法正常工作，并请求确认和帮助。
- **在其他算力平台上运行 FlashInfer**：一位用户询问是否有可在其他云平台上运行的脚本（如果他们没进前 50 名队伍的话），因为他们在别处有免费算力。
   - 他们还询问 **FlashInfer** 是否会发布 **Docker image** 或可执行文件，以便在虚拟机上运行并避免版本问题。
- **比赛取消了吗？**：一位成员询问比赛是否已经停摆。

   - 另一位成员迅速反驳道：*"你说什么呢，它甚至还没开始呢"*。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1469060852396196041)** (63 条消息🔥🔥): 

> `Opus 4.6 vs Codex 5.3, 使用 RL Agent 进行基准测试, Claude 的推理链, Opus 4.6 上下文腐烂 (context rot), Claude Elation` 


- **Opus 4.6 与 Codex 5.3 基准测试重叠存疑**：成员们注意到 **Codex 5.3** 和 **Opus 4.6** 在基准测试中的重叠极少，一些人认为唯一称赞 Codex 的人是那些使用*极其详细的提示词 (prompts)* 来实现单一编辑的人，正如 [Andon Labs 博客](https://andonlabs.com/blog/opus-4-6-vending-bench) 中所见。
   - 新的基准测试趋势是**针对可疑行为对 Agent 进行 RL (强化学习)**。
- **Booking.com 将现金流入作为唯一基准**：一位成员开玩笑说 **Booking.com** 通过测量**现金流入**来测试新功能，将它们部署在部分生产集群上，并采用能产生更多现金的功能。据该成员称，*除此之外的其他一切都不重要*。
   - 他们表示，即使新功能会导致请求失败，他们也不会注意到（除非金额巨大，但那时他们又会注意到，因为它影响了收入）。
- **Opus 4.6 完成复杂编程任务**：一位成员报告称，**Opus 4.6** 在针对一个提示词工作 **4 小时**并消耗了大部分 **$50 免费推理礼金**后，成功为 **lfc1.2-1b** 实现了一个推理引擎。
   - 该成员指出，**Codex 5.3** 也完成了任务，但未能妥善记录所有文档。
- **Opus 4.6 的新架构能解决上下文腐烂 (Context Rot) 吗？**：成员们提到 **Opus 4.6** 似乎采用了针对长上下文的新架构，旨在消除上下文腐烂并提高整体性能，运行在 **Google 的 TPU** 上。
   - 有人指出，与 Google 的合作似乎在智能和硬件方面让 Claude 受益，尽管它并未完全脱离测试阶段。
- **Opus 4.6 在创意写作方面表现惊人**：一位成员发现 **Opus 4.6** 在创意写作方面表现出奇地好，在 [EQBench 创意写作基准测试](https://eqbench.com/creative_writing.html) 中以 **1.7 的 slop score** 达到了*降维打击 (smurfing)* 的水平。
   - 他们推测是否使用了 **PARL** 作为编排器，但对其在创意方面的表现感到惊讶。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1469412393204322366)** (2 条消息): 

> `生成建模, Kaiming He, Drifting` 


- **Kaiming He 的 Drifting 生成模型**：Kaiming He 发表了一篇关于 [Generative Modeling via Drifting](https://openreview.net/forum?id=CFewUmgIILK) 的论文。
   - 细节仍在不断披露中，许多人期待在 **generative modeling** 技术方面取得重大进展。
- **ArXiv 链接问题**：相关的 [ArXiv 链接](https://arxiv.org/abs/2602.04770v1) 似乎已失效。
   - 这可能会给那些习惯使用 ArXiv 阅读完整论文的人带来一些困难。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1469165597320417496)** (3 条消息): 

> `内存黑客 (Memory Hack), 交易 Agent` 


- **Flower Computer 发布 Hivemind**：发布了一个内存黑客工具，对于探索 Agent/技能的人来说可能很有趣：[Hivemind](https://www.flowercomputer.com/hivemind)。
- **CLI 交易 Agent Gordon 发布**：一位成员正在构建一个名为 **Gordon** 的 **CLI 原生交易 Agent**，专注于将自然语言意图转化为结构化的市场推理，并正在寻找有想法的早期用户。
   - 据该成员称，大多数交易工具是为点击和反应而构建的，而不是为了形成和测试信念；感兴趣的人可以在 [等候名单 (waitlist)](https://www.gordoncli.com/) 上注册。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1469412393204322366)** (2 条消息): 

> `Generative Modeling via Drifting, Kaiming He` 


- **Kaiming He 的 Drifting 生成模型**：Kaiming He 是 [Generative Modeling via Drifting](https://openreview.net/forum?id=CFewUmgIILK) 的作者，该论文也可在 [arxiv](https://arxiv.org/abs/2602.04770v1) 上以预印本形式获取。
- **Drifting 生成模型**：该[论文](https://openreview.net/forum?id=CFewUmgIILK)提出了一种全新的生成建模方法。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1469084222806294640)** (44 条消息🔥): 

> `MATS 编码测试, 原始 Pile 数据集, 将对齐视为系统工程问题` 


- ****Pile 数据集之谜：原始版 vs 更新版****：一位成员询问了受版权保护版本的 **The Pile 数据集**，指出 Hugging Face 版本比原始版本小 **100GB**，且 GitHub 下载链接已失效。

- 另一位成员指出，他们下载的版本大约为 **720GiB**，包含 **211,036,982** 份文档，非常接近原始论文中的 **211,043,181** 份。
- **MATS 编码测试：玩具服务模拟**：一位进入 **MATS Summer 2026 Stage 1** 的成员寻求编码测试方面的帮助，该测试被描述为一个使用标准库和通用分布式系统的*玩具服务型问题*。
   - 另一位成员建议，如果测试涉及使用 Python 内置工具模拟真实世界的服务器场景，应熟练掌握用于并行处理的 **asyncio** 和用于创建 **FIFO queues** 的 **deque**。[此处可查看有关 MATS 申请的更多详细信息](https://forum.effectivealtruism.org/posts/da8MmRPAB55Fepjjk/my-experience-applying-to-mats-6-0)。
- **对齐：工程与增长之争爆发**：一位新成员提出，**alignment** 主要是一个**系统工程问题**，认为围绕模型的工程对齐——通过*治理、路由、可审计性、回滚和明确的停止条件*——可以约束并逆转行为。
   - 他们认为，仅依靠训练来进行对齐会导致*漂移和不透明的失败*，并提倡建立一种由模型处理推理、但信任源于周边系统的体系。

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1469135814331072715)** (7 messages): 

> `JEPA Models, LoRA Compression, Low Regret LoRA, LSH, Model Upscaling` 

- **寻求 JEPA 模型训练经验**：一位成员询问是否有人有训练 **JEPA Models** 的经验。
   - 未收到回复。
- **Low-Regret LoRA 文章引发关注**：一位成员引用了 Shulman 2025 年的 *Low Regret LoRA* 文章 ([https://arxiv.org/abs/2511.03270](https://arxiv.org/abs/2511.03270))，询问关于 **LoRA compression** 和秩（rank）问题的兴趣。
   - 另一位成员注意到其与 **LSH** 的相似性，并强调哈希函数（质心/超平面）的在线学习是一个很酷的特性。
- **Kernel Smoothing 作为 LoRA 压缩的替代方案**：一位成员建议在 **LoRA compression** 中可以使用 [Kernel Smoothing](https://en.wikipedia.org/wiki/Kernel_smoother) 来代替高斯回归。
   - 该成员表示确信这将表现得非常好。
- **模型放大方法论**：一位成员提到了另一种**模型放大（model upscaling）方法论**。
   - 未提供链接或具体细节。

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1469478347183493120)** (1 messages): 

> `Subtask Dependencies, Emergence in Scaling, Regulation and Control Layers` 

- **子任务相互依赖触发涌现**：由于相互依赖、相关性和瓶颈，子任务的成功并非简单的乘法叠加，这正是表面上的 **emergence**（涌现）来源。
   - 添加*监管或控制层*可以在抑制某些行为的同时提高底层能力；阈值的转变使其看起来像是一种跳跃。
- **架构变化使涌现可视化**：总体而言，它仍然遵循 Scaling 行为，但当这种涌现变得可见时，**架构已经发生了变化**。
   - 评论者表示，所提供的例子是一个*很好的直觉泵（intuition pump）*。

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1469124677791383552)** (2 messages): 

> `Data Attribution, Gradient Normalization, Hessian Estimate, Goodfire AI Blog` 

- **梯度归一化提升归因准确度**：一篇新[论文](https://arxiv.org/html/2410.17413v1)表明，对梯度进行单位归一化（unit normalizing）可以通过减少具有高整体梯度幅度的异常训练样本的影响，从而提高**归因准确度（attribution accuracy）**。
   - 该论文引用了 Akyurek 等人、Han & Tsvetkov、Choe 等人和 Xia 等人的先前工作，支持在计算 **cosine similarity** 时使用**单位归一化**。
- **Hessian 估计减轻归一化需求**：根据[这篇论文](https://arxiv.org/pdf/2504.16430)，有了充分的 **Hessian estimate**，梯度归一化可能变得不再必要。
   - 该链接指向了一场关于通过更准确的 **Hessian estimations** 来克服对**梯度归一化**需求的讨论。
- **Goodfire AI 倡导有意图的设计**：[Goodfire AI 的一篇博客文章](https://www.goodfire.ai/blog/intentional-design)讨论了 AI 系统中有意图的设计（intentional design）的重要性。
   - 未提供该博客文章具体观点的进一步细节。

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1469201513976172565)** (35 messages🔥): 

> `Custom UOp for matvec on x86, Tensor.sort fix for MoE topk, Bitonic sort in one kernel, Numpy.partition for topk, Kimi faster on the MI300`

- **围绕针对 x86 的 matvec 自定义 UOp 的可接受性展开辩论**：讨论了在 x86 上为 `matvec` 编写自定义 UOp 是否可接受或过于复杂，共识倾向于通过**启发式算法（heuristics）和更高级别的 kernel 编码**来实现性能提升，而非使用自定义 UOp。
   - 一位成员提到仅通过 **CPU tuning** 就获得了*显著的改进*。
- **Tensor.sort 修复了 MoE Topk 性能**：在修复了 MoE `topk` 中缓慢的 `Tensor.sort` 后，一名用户报告在 **M3 Pro 36GB** 上使用 *deepseekv2-lite* 和 *youtu-llm* 加速 MLA 和 MOE 后，达到了 **50 tok/s** 的速度。
   - 另一位用户报告说 `llama-cli` 在同一台机器上达到了 **35.1 tok/s**。
- **新赏金目标设定为 35 tok/s**：随着 `Tensor.sort` 修复后 **M3 Pro** 上的性能提升至 **50 tok/s**，赏金目标被下调至 **35 tok/s**，挑战贡献者匹配或超过 `llama-cli` 现有的速度。
   - 同一位用户提供了[用于测试的实用模型](https://huggingface.co/)。
- **提议实现 Pairwise Topk**：为了解决 *whisper export for webgpu* 中 `topk` 或 `Tensor.sort` 导致的减速，一位用户分享了一个 `_topk_pairwise` 实现，涉及复杂度为 **O(n^2)** 的成对比较，适用于像 64 个专家（experts）这样较小的 `n`。
   - 分享者还对由 Claude Opus 4.6 生成的 **5 种 topk 实现**进行了基准测试，并考虑根据输入大小采用双调排序（bitonic sort）等替代方案。
- **Cached Property 导致 Stable Diffusion 中的递归错误**：在全新的 git clone 环境中，Stable Diffusion 示例因递归错误而失败，因此建议将 `tinygrad/uop/ops.py` 中 `UOp.key` 的装饰器从 `@functools.cached_property` 更改为 `@recursive_property`。
   - 应用此修复后，命令在大约 **25–30 秒**内完成且无错误。

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1469070202544328848)** (1 messages): 

> `llama 1B, CPU Optimization, Pull Request Strategy, CI integration` 

- **成员寻求关于最优 Pull Request 策略的建议**：一位成员询问团队更倾向于将测试作为一个**独立的 PR**，还是将其与**针对 llama 1B 在 CPU 上快于 torch 的赏金任务**的 **CPU 优化**放在同一个 PR 中。
   - 他们还询问是应该通过预期的失败将其显式集成到 **CI** 中，还是仅添加测试用例以进行手动基准测试。
- **CPU Tuning 优化准备就绪**：该成员已经准备了一个**公平对比测试（apples-to-apples test）**和一些**针对 CPU 范围的简单 tuning** 优化。
   - 他们的目标是在学习项目代码库和开发方法论的同时简化流程。

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1469081433975816315)** (7 messages): 

> `Opus 4.5, Sonnet 4.5, Openrouter, Claude Pro, GPT 5.2` 

- **最优 AI 配置大对决**：一位成员正在使用 **Opus 4.5** 负责架构，使用 **Sonnet 4.5** 负责编码，并询问在成本较高的情况下，是使用带额度的 **OpenRouter** 更好，还是使用 **Claude Pro** 更好。
   - 另一位成员回复说，他们使用 **Opus 4.5** 负责架构，**GPT 5.2** 负责审查，**Haiku** 负责编码，同时 **GPT 5.2** 也会审查代码。
- **辩论 Claude 与 GPT 的编程风格**：一位成员表示 *Claude* 更擅长发散性思维、抽象和推理想法。
   - 同一位成员表示 *GPT* 更擅长钻研细粒度的细节。
- **简化 AI 编程工作流**：一位成员建议用户可以仅使用 **Opus 4.5**（现为 **4.6**）和 **GPT 5.2** 来进行思考，从而可能简化其配置。
   - 这暗示了通过整合工具来提高效率的趋势。

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1469327640283451474)** (21 messages🔥): 

> `Copilot Opus 4.5 Configuration, Aider Chat History, Auto-Accept Architect Setting` 

- **用户在配置 Copilot Opus 4.5 时遇到困难**：一位用户在配合 **Aider** 使用 **Copilot Opus 4.5** 时遇到问题，该工具在提问后不等待用户输入就自行继续执行。
   - 用户确认已通过 CLI 标志（`aider --model github_copilot/claude-opus-4.5 --architect`）和 `.aider.config.yml` 设置了模型。
- **配置 edit-format 解决了 Aider 的困扰**：一位成员建议在 `.aider.conf.yml` 中添加 `edit-format: diff-fenced` 以改善行为，但用户不确定其工作原理。
   - 用户报告称，无论如何配置，机器人都会在关键决策点出现意料之外的“不暂停”行为。
- **关于 --auto-accept-architect 标志的疑虑**：一位成员建议用户检查 `--auto-accept-architect` 设置，该设置会自动接受架构师模式的更改。

- 成员链接到了 [Aider documentation](https://aider.chat/docs/config/options.html) 以获取更多配置选项，并解释了部分用户为何更倾向于 one-shot 交互并使用 `/undo` 来回滚更改。

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1469066000178872515)** (26 条消息🔥): 

> `Kimi vs Opus, CLI 偏好, Kimi K2.5 访问, AI slides 编辑, Kimi 集成` 

- **Kimi 与 Opus 的激烈辩论**：用户声称 **Kimi K2.5** 优于 **Opus 4.5**，尽管 Opus 发布时间更久，并指出 **Claude 的 rate limits** 是主要的挫败感来源。
   - 一位用户表示：*"长期同时使用 Claude 和 Kimi 的人最终会承认 Kimi 比 Opus 4.5 更好，这并不是在抨击 Opus，毕竟它作为一个模型已经发布好几个月了"*。
- **CLI 工具大放异彩**：用户发现 **Kimi Code CLI** 和 **Opencode CLI** 比图形界面好得多，这可能是由于对 DOS 时代的命令行情有独钟。
   - 然而，一位用户指出：*"问题在于 CLI 工具没有集成，所以我不得不使用 VSCode。"*
- **Kimi K2.5 的免费层级**：关于 **Kimi K2.5** 在 **OpenRouter** 上是否仍然免费存在困惑，一些用户认为它是免费的，而另一些用户则指正那可能是 **Opencode Zen**。
   - 有人分享了一张关于可能需要升级才能享受 **Kimi K2.5** 的截图，但另一位用户指出 **K2.5** 发布后，他们经历了巨大的用户涌入。
- **AI Slides：一次编辑一张幻灯片**：一位用户询问如何在 **AI Slides** 中编辑单张幻灯片而不必重新生成整个演示文稿。
   - 另一位用户澄清说，在 **adaptive mode** 下，可以调整文本、图像和形状，并可以添加新图像。
- **Kimi 合作洽谈**：一位用户询问与 **Kimi AI** 的合作机会，另一位用户提议通过私信（DM）转发该请求。

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1469359555480785090)** (14 条消息🔥): 

> `AI SaaS 功能, Manus 计费问题, 账号停用` 

- **AI 工程师加入频道**：一位 AI/ML 及全栈工程师介绍了自己，重点介绍了在使用 React 前端和后端 RAG + evals 构建 **SaaS AI 功能**（如搜索、摘要、智能路由和自动生成报告）方面的经验。
   - 他们表达了与初创公司合作的激情，旨在 *超越 AI 实验，交付可靠的、生产就绪的智能产品*。
- **用户报告 Manus 计费乱象**：一位用户报告称，在降级后被按 **每个个人账户 $5k** 收费，导致客户网站下线，目前正在寻找替代方案。
   - 他们表示 Discord 支持没有回应，而直接邮件支持则声称降级从未发生。
- **账号突然被停用，用户求助**：一位用户报告称其账号无故被停用，且未收到支持部门的回复。
   - 另一位用户简单地建议他们检查垃圾邮件箱。

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1469078992542765117)** (10 条消息🔥): 

> `Modular 发布简报被标记为垃圾邮件, 投票结果显示许多 Mojo 用户在德国, 下次 Meetup 的可能地点：苏黎世、新加坡、悉尼、圣路易斯、芝加哥、爱丁堡、熊谷` 

- **Modular 简报进入垃圾邮件箱**：一位成员报告称 **Gmail** 将 **Modular 26.1 发布简报**标记为垃圾邮件，并附上了[截图](https://cdn.discordapp.com/attachments/1098713601386233997/1469244816968781855/Screenshot_2026-02-06_at_09.13.04.png?ex=69879d1d&is=69864b9d&hm=c02e4268d2bcb5598a7dcc0d6dfb1d3cc687e31eb54064faf5e8374927d5a9c5&)。
   - 另一位成员确认这种情况有时也会发生在他们身上。
- **德国在投票中占据主导**：成员们表示投票结果很有趣，因为*德国有很多用户*，并建议计划在 10 月举办活动。
- **苏黎世和爱丁堡被提议为下次 Meetup 地点**：在列出 **新加坡**、**悉尼**、**圣路易斯**和**芝加哥**之后，一位成员建议选择 **苏黎世**，并链接到了 [ETH AI Center Academic Talk Series](https://ai.ethz.ch/research/events/academic-talks.html) 和位于苏黎世欧瑞康的 [Robotics and AI Institute](https://ethz.ch/en/news-and-events/eth-news/news/2025/09/the-rai-institute-opens-up-unique-opportunities-for-both-researchers-and-students.html)。
   - 一位 Modular 员工开玩笑地建议 **爱丁堡**，并提议在 **Frontier Tower** 举办。
- **加州熊谷（Bear Valley）被提议为备选 Meetup 地点**：一位成员提议将 **加州熊谷（滑雪胜地）** 作为理想地点，因为它距离 **北加州**、**雷诺** 和 **盐湖城** 交通都很便利。

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/)** (1 messages): 

sbrunk: 既然 26.1 已经发布了，我可以请求一次 review 以便将修复程序合入 nightly 版本吗？ 🫶
  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1469340433074094338)** (2 messages): 

> `GLM OCR, RLMs, DSPy` 


- **GLM OCR 开源了！**：一位成员分享了运行 **GLM OCR** 的代码库，提供了一个无需复杂基础设施配置的免费替代方案：[https://github.com/neosantara-xyz/glm-ocr-inference](https://github.com/neosantara-xyz/glm-ocr-inference)。
   - 同时还提供了一个托管模型的链接：[https://docs.neosantara.xyz/en/glm-ocr](https://docs.neosantara.xyz/en/glm-ocr)。
- **RLMs + DSPy**：**RLMs** 被描述为缓解 context rot 的最简单方法，而 **DSPy** 是使用 **RLMs** 的最简单方式。
   - 分享了一篇博文，解释了为什么 **RLMs** 有效以及如何在 **DSPy** 中开始使用它们：[https://blog.isaacbmiller.com/posts/rlm](https://blog.isaacbmiller.com/posts/rlm)。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

ash_blanc: https://www.alphaxiv.org/abs/2602.03786
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1469165081018241047)** (6 messages): 

> `Smallest LM for DSPy, DSPy Community Call` 


- **T5 Small LM 已适配 DSPy**：一位成员询问构建轻量级 CLI 工具（特别是为一家印度机构）最可靠的最小 LM 是什么，以便配合 **DSPy** 使用。
   - 另一位成员建议使用 **T5 small (80M)**，并链接了一个演示 **DSPy fine-tuning** 的 [Lightning AI 教程](https://lightning.ai/lightning-ai/environments/dspy-finetune-a-t5-small-to-excel-at-rag?section=featured)。
- **DSPy 社区会议即将举行**：一位成员宣布计划下周举行在线会议，讨论社区项目和 **DSPy** 的未来。
   - 另一位成员询问了会议的时区以便参加。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1469067565643862026)** (5 messages): 

> `Codex pricing, Claude Code as AI Research Agent, AI Research Skills library, Model Architecture frameworks, Fine-Tuning frameworks` 


- **明确 Codex 费用**：一位用户询问 **Codex** 是否包含在月度订阅中，还是需要通过 API 按 token 付费；另一位用户确认 [月度订阅模式有效](https://openai.com/api/)。
- **Claude Code 凭借新技能成为 Research Agent**：**AI Research Skills** 库是一个包含 80 多个研究和工程技能的开源集合，使 **Claude Code** 等编码 Agent 能够进行从训练到部署的 AI 研究，可通过 [GitHub](https://github.com/Orchestra-Research/AI-research-SKILLs) 获取。
- **AI Research Skills 填补 Agent 能力空白**：**AI Research Skills** 库解决了编码 Agent 在 AI 研究中的局限性，提供了涵盖特定工具和框架的生产级指南，范围涵盖从使用 **Axolotl** 进行 fine-tuning 到使用 **Megatron-Core** 进行分布式训练以及使用 **vLLM** 进行推理。
- **该库涵盖 20 个 AI 研究类别**：该库跨越 20 个类别，包括 **Model Architecture**、**Fine-Tuning**、分布式训练、优化、推理、**RAG**、**Agents**、多模态和安全性，为各种 AI 任务提供专家级知识。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1469415929845776477)** (1 messages): 

> `AI-driven crypto products, Smarter trading dashboards, On-chain analytics summaries, AI assistants for contracts/transactions` 


- **开发者深入研究 AI 驱动的加密产品**：一位成员正在开发 **AI 驱动的加密产品**，重点关注更智能的交易仪表盘和链上分析摘要。
   - 该产品还包括用通俗易懂的语言解释合约和交易的 **AI assistants**，并非常强调安全性和透明度。
- **强调 AI 加密工具的安全性和透明度**：开发者强调了在其 **AI 驱动的加密产品**中对安全性和透明度的承诺。
   - 这包括确保用户能够理解 AI 是如何解读复杂的合约和交易的。