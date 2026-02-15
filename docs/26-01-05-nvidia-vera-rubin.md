---
companies:
- microsoft
- google-deepmind
- boston-dynamics
date: '2026-01-05T05:44:39.731046Z'
description: '以下是这段2026年1月初**AI新闻**的中文翻译：


  **2026年1月初的AI新闻**重点报道了关于**越南**经济将超越泰国的爆火经济预测；据报道，**微软**开源了用于1比特（1-bit）CPU推理的 **bitnet.cpp**，承诺将带来速度与能效的提升；此外，**Google
  DeepMind** 与**波士顿动力（Boston Dynamics）**建立了新的研究合作伙伴关系，重点关注 **Gemini Robotics** 和 **Atlas
  硬件**。


  “**智能体编程**”（Agentic coding）的概念正受到关注，该理念强调人类监督以及被称为“**Agent Harnesses**”的基础设施层，用于管理长期运行的AI任务，Philipp
  Schmid 等倡导者正在推动这一转变。编程智能体持久化内存方面的创新（如 **Claude-Mem**）旨在提高上下文的持久性。


  此外，业界还在对智能体工作流中的“规范问题”（specification problem）进行批判性讨论，主张建立超越对话意图的更好抽象。实际应用中的挑战包括管理并行智能体和权限风险。在开源工具进展方面，还推出了一个基于
  JAX 的 **LLM-Pruning Collection**，用于实现高效的模型剪枝。'
id: MjAyNi0w
models:
- claude-mem
- bitnet-cpp
- gemini
people:
- _philschmid
- demishassabis
title: '今天没发生什么特别的事。'
topics:
- agentic-coding
- agent-harnesses
- persistent-memory
- software-engineering
- inference-efficiency
- model-pruning
- context-durability
- specification-problem
- workflow-management
- cpu-inference
---

**安静的一天**

> 2026年1月2日至2026年1月5日的 AI 新闻。我们为您检查了 12 个 subreddits、[**544** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **24** 个 Discord 服务端（**204** 个频道，共 **13618** 条消息）。预计节省阅读时间（按 200wpm 计算）：**1170 分钟**。**我们的新网站**现已上线，包含完整的元数据搜索，并以优美的 vibe coded 风格呈现了所有往期内容。详见 https://news.smol.ai/ 查看完整的新闻细分，并在 [@smol_ai](https://x.com/Smol_AI) 上给我们反馈！

---

# AI Twitter 回顾


**热门推文（按互动量排序）**

- **越南的增长叙事**：一条走红的推文预测越南将超越泰国成为东南亚第二大经济体，理由是越南正在制造业阶梯上攀升，而泰国则依赖旅游业 [tweet](https://twitter.com/okaythenfuture/status/2008023248706089221)。
- **Microsoft 据传开源了 1-bit 推理**：一条高互动量的消息称 Microsoft 开源了 `bitnet.cpp`，支持在 CPU 上对超大型模型进行推理，并带来巨大的速度和能效提升 [tweet](https://twitter.com/simplifyinAI/status/2008195754092065050)（视为“推文报道”；具体细节请核实仓库/文档）。
- **机器人领域头条**：Google DeepMind 宣布与 Boston Dynamics 建立研究合作伙伴关系，围绕 Gemini Robotics + Atlas 硬件展开 [post](https://twitter.com/GoogleDeepMind/status/2008283100254494916)；Demis Hassabis 发布了后续消息 [post](https://twitter.com/demishassabis/status/2008307002699612586)。

---

**Agentic coding 成为主流：Harnesses、Memory 以及“软件工程时代”的辩论**

- **“效用阈值” + 工作流转变**：多位从业者认为模型已经跨越了软件工程的可用性阈值——重点不再是“它能写代码吗？”，而更多是“我们如何有效地管理/组合 Agent？” [@gdb](https://twitter.com/gdb/status/2007938049209254002) 以及一种反复出现的观点，即“代码从来都是容易的部分” [@tekbog](https://twitter.com/tekbog/status/2007928317236949387)。其他人将 “vibe coding” 重新定义为 **Agentic coding**，以强调人类的注意力/监督才是稀缺资源 [@ZechenZhang5](https://twitter.com/ZechenZhang5/status/2007917489397920186)。
- **Agent Harnesses 作为下一层基础设施**：Philipp Schmid 认为 2026 年将由 **Agent Harnesses** 定义——这是 Agent 框架之上的基础设施，标准化了长期运行任务的生命周期、工具策略、HITL（人机回环）、规划钩子（planning hooks）以及“上下文持久性（context durability）”，将 Benchmark 上的声明与用户体验联系起来，并从实际使用中创建爬坡式的反馈循环 [@_philschmid](https://twitter.com/_philschmid/status/2008175408923959574)（推文中附有博客链接）。这与“设计模式 > 模型增量”的观点相吻合：竞争正转向支架/Harnesses，而不仅仅是基础模型的改进 [@kchonyc](https://twitter.com/kchonyc/status/2008146568265007407)，社区呼吁建立“开放 Harnesses” [@Vtrivedy10](https://twitter.com/Vtrivedy10/status/2008197837410938931)。
- **编程 Agent 的持久化记忆**：“Claude-Mem”被推广为一个基于本地 SQLite 的记忆插件，它存储工具使用/观察结果的压缩语义摘要，以便以更少的 Token 和更多的工具调用（“无尽模式”）恢复工作 [@LiorOnAI](https://twitter.com/LiorOnAI/status/2008161724902355118) 以及仓库链接 [见此](https://twitter.com/LiorOnAI/status/2008161726345134449)。这直接针对作为瓶颈的“上下文持久性”。
- **规格说明问题 / 抽象反思**：一个持续的反向观点认为，管理一个 Agent 来输出 10 万行代码是错误的抽象方式；我们需要比对话更好的意图表达方式，以及更好的能够保留/组合意图的中间表示（DSPy 被引用为正确处理了这种“规格说明责任”） [@lateinteraction](https://twitter.com/lateinteraction/status/2008215241004605922)，[后续](https://twitter.com/lateinteraction/status/2008215918577688750)，[规格说明问题](https://twitter.com/lateinteraction/status/2008237433737904168)，以及“苦涩的午餐理论”框架 [@lateinteraction](https://twitter.com/lateinteraction/status/2008285334971302050)。这是技术上最显著的“反炒作”系列讨论：它不是在争论模型不会进步，而是认为 **UX/抽象层级必须向上移动**。
- **实际规模化痛点：并行 Agent + 权限风险**：人们报告了在使用多个并发 Agent 时“频繁切换窗口”的工作流以及频繁的崩溃 [@itsclivetime](https://twitter.com/itsclivetime/status/2007975171219771758)；鉴于观察到的错误，其他人担心在授予广泛权限的情况下让 Agent 彻夜运行 [@JFPuget](https://twitter.com/JFPuget/status/2008133619911381457)。

---

**开源工具 + 推理效率：剪枝（Pruning）、微型 vLLM 克隆、内存/VRAM 计算器，以及（据称的）1-bit CPU 推理**

- **统一的剪枝代码库 (JAX)**：发布了 **LLM-Pruning Collection**，这是一个基于 JAX 的复现/基准测试套件，涵盖了块/层/权重级别的剪枝方法（Minitron, ShortGPT, Wanda, SparseGPT, LLM-Pruner），并提供了训练/评估流水线，支持 GPU (FMS-FSDP) 和 TPU (MaxText) [@liuzhuang1234](https://twitter.com/liuzhuang1234/status/2007930641061740556)。该项目的意义在于其基础设施的覆盖范围（JAX + FSDP + MaxText）以及让剪枝研究变得可复现。
- **推理引擎正在碎片化（以一种积极的方式）**：vLLM 强调了一波从零开始的极简实现——`nanovllm`、`minivllm`、`tiny-llm`——作为教学/实验性引擎，而 vLLM 本身也在重构核心架构，使其更简洁、更具扩展性 [@vllm_project](https://twitter.com/vllm_project/status/2007993964742500396)。这是“开源系统”的一个信号：工程师们想要可修改的服务堆栈，而不是黑盒。
- **部署模型尺寸评估**：`hf-mem` 通过元数据估算任何 Hugging Face safetensors 仓库所需的 VRAM；通过 `uvx` 提供轻量级 CLI [@alvarobartt](https://twitter.com/alvarobartt/status/2008214540463341826)。这对于快速检查量化/卸载计划的合理性非常有用。
- **Apple Silicon 本地训练与服务体验**：Unsloth-MLX 为 MLX 带来了类 Unsloth 的 API，用于在 Mac 上进行本地微调（“本地原型设计 → 扩展到云端”）[@_ARahim_](https://twitter.com/_ARahim_/status/2008221602283225371)。Mawj 中的 “MLX Engine Revolution” 也带来了针对 Apple Silicon 的其他改进 [@7alkiumi](https://twitter.com/7alkiumi/status/2008082410009956507)。
- **报道：微软 `bitnet.cpp`**：一条疯传的推文声称微软开源了 `bitnet.cpp`，实现了在 CPU 上进行高达 **100B** 参数的 **1-bit** LLM 推理，并在速度和能耗上获得巨大提升 [@simplifyinAI](https://twitter.com/simplifyinAI/status/2008195754092065050)。请将其视为一个线索；工程师应验证：支持的架构、精度差异、算子覆盖范围以及与量化 GPU 基准相比的实际吞吐量。

---

**模型发布、基准测试和多模态进展（以及对“LLM 物理学”的怀疑）**

- **新型小型推理模型声明（7B 级）**：据报道，TII 的 **Falcon H1R-7B** 是一个 **Mamba-Transformer 混合架构**，具有 **256k 上下文**，并声称具有强大的数学/编程性能 [@mervenoyann](https://twitter.com/mervenoyann/status/2008140906814468442)；另一条推文引用了 **88% AIME24 / 83% AIME25** 的成绩以及 “Falcon LLM 许可证” [@kimmonismus](https://twitter.com/kimmonismus/status/2008188516329542010)。如果属实，这是“小型推理模型”浪潮的一部分，但关键的工程问题在于可复现性和评估的完整性。
- **大型 MoE 训练方案详情 (EXAONE)**：LG 的 **K-EXAONE 236B MoE (23B 激活)** 技术报告被总结为具体的堆栈：**Muon**、WSD 学习率调度、**FP8**、**DeepSeek 负载均衡**，外加 SWA (128-token 窗口) 和 MTP；后训练使用了 GRPO 变体 **AGAPO** + 自定义偏好学习 [@eliebakouch](https://twitter.com/eliebakouch/status/2008182861791170674)，并附带 [报告/模型](https://twitter.com/eliebakouch/status/2008183325249409381) 链接。这是对工程师最有用的模型推文之一，因为它列举了可实现的训练调节参数。
- **图像模型排行榜变动**：Arena 报告 Qwen 图像模型排名上升：**Qwen-Image-Edit-2511** 在开源图像编辑模型中排名第 1，**Qwen-Image-2512** 在开源文本生成图像模型中排名第 2 (Apache 2.0) [@arena](https://twitter.com/arena/status/2008238877589258449)。
- **基准测试完整性与“噪声”讨论**：多篇文章反对浅层的基准测试追逐。一个显著的主题是：**评估噪声 + 作弊**以及对受控变量“LLM 物理学”的需求，认为小型模型比存在噪声的前沿模型对比更能揭示架构真相 [@GenAI_is_real](https://twitter.com/GenAI_is_real/status/2007919179274543610)。相关内容：SWE-bench 增加了简单的“补丁复现检测”，发现与标准答案补丁有约 **6.7%** 的精确重合，并移除了一个异常值，认为性能提升仍然是真实的，并非主要由测试集污染导致 [@OfirPress](https://twitter.com/OfirPress/status/2008297771384631573)。
- **通过扩散模型实现多模态推理**：**DiffThinker** 提出将多模态推理视为图像到图像的扩散过程，而非文本思维链（Chain-of-Thought），声称具有更好的空间精度、可控的推理成本、并行候选推理能力，并与 MLLMs 形成互补 [@yafuly](https://twitter.com/yafuly/status/2008098428375470556)。

---

**面向 LLM 的 RL 及评估：GRPO “++”、级联 RL 和推理完整性**

- **实践中的 GRPO 是 “GRPO++”**：Cameron Wolfe 预告并发布了一份长篇、附带论文链接的指南，汇编了超越原生 GRPO 的稳定性技巧：非对称裁剪（asymmetric clipping）以维持探索（exploration）、动态采样以避免零优势（zero-advantage）批次、长度偏见修正（Token 级损失聚合变体）、过长奖励塑造（reward shaping）、消除标准差归一化爆炸，以及针对多引擎 Rollout（vLLM 采样 vs FSDP 训练）的重要性采样（importance-sampling）修正，此外还包含 CISPO 变体。[预览链接](https://twitter.com/cwolferesearch/status/2008035254246777211) 和 [博客链接](https://twitter.com/cwolferesearch/status/2008185753818550567)，简明要点列表见[此处](https://twitter.com/cwolferesearch/status/2008245160883208214)。
- **Cascade RL（顺序领域强化学习）**：一份关于 NVIDIA **Cascade RL** 的详细摘要指出，混合异构验证机制（数学符号 vs 代码执行 vs RM 评分）会使基础设施和调优复杂化；因此建议跨领域进行顺序训练（对齐 → 指令遵循 → 数学 → 代码 → SWE）。其核心观点是：RL 的 On-policy 特性相比 SFT 能减少灾难性遗忘。报告结果显示，Nemotron-Cascade-8B 在 **LiveCodeBench v6** 上达到 **71.1%**，而 DeepSeek-R1-0528 为 **73.3%**，此外一个 14B 模型表现强劲（包括 IOI 2025 银牌水平）[@omarsar0](https://twitter.com/omarsar0/status/2008240593257066816)。
- **小型模型的基于过程的可靠性**：一篇名为 “因错的原因对（Right-for-Wrong-Reasons）” 的论文摘要称，7–9B 模型给出的正确答案中，有 **50–69%** 包含有缺陷的推理轨迹；文中引入了 **推理完整性得分（Reasoning Integrity Score, RIS）**，发现 RAG 能提高推理完整性，而自我批评（self-critique）提示词可能会产生负面影响（“伪反思”），并蒸馏出一个快速验证分类器（F1 分数 0.86）[@dair_ai](https://twitter.com/dair_ai/status/2008223984333267453)。工程师应将其理解为：**最终答案的准确性对于自主 Agent 而言是不够的**；需要集成廉价的过程检查（process checks）。

---

**实战中的 Agent：竞赛获胜、文档流水线、企业推广以及作为里程碑的 “ACI”**

- **Sakana AI 在重大优化竞赛中获胜**：Sakana 的 **ALE-Agent** 在 AtCoder Heuristic Contest 058 中击败 800 多名人类选手获得 **第一名**。据称该 Agent 通过多个前沿模型的推理时间扩展（Inference-time scaling）、并行代码生成以及迭代邻域搜索实现；总成本约 **1,300 美元** [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/2008195936917586416)，以及额外的背景介绍 [@hardmaru](https://twitter.com/hardmaru/status/2008196968653447318)。当闭环包含评估和时间限制下的迭代优化时，这是 “Agentic 算法工程” 的一个强力数据点。
- **通过 “神经程序” 实现文档级自动化**：一个具体的 “Agent 流水线” 案例：翻译并排版了一本 1964 年、330 页的苏联教科书。该流程使用 LLM 驱动的 OCR → 翻译 → LaTeX 转换，配合日志记录（journaling）和子 Agent，并用 TikZ 重建了 17 个图表 [@mbusigin](https://twitter.com/mbusigin/status/2008020958313848950) 及其 [程序分解](https://twitter.com/mbusigin/status/2008020961359016184)。这是长程 Agent 工作流的一个良好模板：从日志恢复（resume-from-journal）+ 结构化验证步骤。
- **企业采用节奏（Cognition/Windsurf/Devin 轶事）**：一位从业者分享了内部推广指标：从引入到 POC 约需 2 个月，随后是快速的多国扩张，最终与一个小型客户团队（包括 FDE）达成 “八位数 ARR” 的多年合约，现场支持（on-sites）推动了 150–400% 的使用量激增 [@swyx](https://twitter.com/swyx/status/2008320926371508506)。核心观点：采用模式可以是 “以公司为队列”，而非用户注册队列。
- **Mustafa Suleyman 的 “ACI” 测试**：提议将 **人工能力智能（Artificial Capable Intelligence）** 作为下一个里程碑：一个 Agent 能否利用 **10 万美元并合法地将其变成 100 万美元**——这是一个现代版 “图灵测试”，强调在现实世界中的操作能力 [@mustafasuleyman](https://twitter.com/mustafasuleyman/status/2008208870204948746)。

---

**安全、滥用与治理摩擦（以及参与动机问题）**

- **NCII / 图像滥用担忧：“工作进展慢得令人震惊”**：Margaret Mitchell 指出，非自愿敏感图像 (NCII) 是一种快速增长的 AI 危害，但补救工作有限，呼吁采用多工具方法和更好的激励机制；她还指出了言论自由与隐私/安全之间的紧张关系 [推文片段](https://twitter.com/mmitchell_ai/status/2007916900140069247)，[伦理框架](https://twitter.com/mmitchell_ai/status/2008245538014265446)，[政策说明](https://twitter.com/mmitchell_ai/status/2008244889839169776)。
- **Grok “脱衣”功能引发抵制**：一个推文串认为限制此类系统（例如，仅允许编辑用户拥有的照片）是“轻而易举”的，不这样做会带来骚扰和 CSAM 风险 [@BlancheMinerva](https://twitter.com/BlancheMinerva/status/2008187886462730246)。
- **参与度激励机制偏向冲突**：有帖子指出“战争与暴力”会驱动参与度 [@nearcyan](https://twitter.com/nearcyan/status/2007923876848971974)，并警告平台关于降低极端沙文主义权重的排名决策可能会塑造国家的轨迹 [@willdepue](https://twitter.com/willdepue/status/2008228649699254762)。对于构建推荐相关系统的 AI 工程师来说，这提醒了目标函数（objective functions）在社会层面同样至关重要。
- **招聘：风险评估职位**：DeepMind AGI Safety 正在为 Gemini 招聘负责灾难性风险评估和缓解评估的研究工程师 [@NeelNanda5](https://twitter.com/NeelNanda5/status/2008230731030687947)。


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. 本地化 AI 模型发布

  - **[[发布] 我们训练了一个 AI 来理解台湾的梗和俚语，因为主流模型做不到。认识一下 Twinkle AI 的 gemma-3-4B-T1-it。](https://www.reddit.com/r/LocalLLaMA/comments/1q4aiko/release_we_trained_an_ai_to_understand_taiwanese/)** (热度: 36): **Twinkle AI** 发布了 **gemma-3-4B-T1-Instruct**，这是 Google Gemma 3 的专门版本，专门为理解台湾文化（包括当地俚语、地理和梗）而量身定制。该模型解决了主流 LLM 在生成繁体中文时默认使用中国大陆语境的问题。它特别擅长“Function Calling”，非常适合构建 Agent。该模型已在 [Hugging Face](https://huggingface.co/twinkle-ai/gemma-3-4B-T1-it/blob/main/README_EN.md) 上架。一位评论者表示有兴趣同时支持 ZH-tw 和 ZH-cn，并询问了台湾社区用于模型训练的最佳数据集。另一位评论者要求提供该模型英文输出的示例，对其英文性能表示疑问。

    - randomfoo2 正在询问除了简体中文 (ZH-cn) 之外，支持台湾国语 (ZH-tw) 的模型训练最佳数据集。这表明需要能够捕捉台湾语言和文化细微差别的专用数据集，这些数据集与中国大陆使用的数据集有所不同。该评论暗示了对区域语言支持的数据集选择和模型训练的技术兴趣。
    - RefrigeratorCalm9701 对 Twinkle AI 的 gemma-3-4B-T1-it 模型的输出质量感到好奇，特别是询问是否有英文输出。这表明了评估模型性能并了解其生成输出能力的技术兴趣，这可能有助于评估其在多语言处理中的通用性和准确性。

  - **[Llama 3.3 8B, abliterated to <0.05 KL](https://www.reddit.com/r/LocalLLaMA/comments/1q4ahw1/llama_33_8b_abliterated_to_005_kl/)** (热度: 126): **该帖子讨论了据传泄露的 **Llama 3.3 8B 128k** 模型的“abliterated”（清除拒绝机制）版本，旨在最小化智能损失的同时优化合规性。该模型以 BF16 权重发布在 [Hugging Face](https://huggingface.co/SicariusSicariiStuff/Llama-3.3-8B-Instruct-128K_Abliterated) 上。贡献者包括 **Fizzarolli**、**p-e-w** 和一名匿名的 **Meta 员工**。据报告，该模型的 KL 散度达到 `<0.05`，表明与原始分布的偏差极小。** 一条评论指出，初步测试显示该模型具有更高的 **IFeval** 分数，但 **multilingual**（多语言）能力有所下降，这与 Fizzarolli 的结论一致。

    - Sicarius_The_First 指出，Llama 3.3 8B 的初步测试显示出更高的 **IFeval** 分数，表明在某些任务中的性能有所提高。然而，这是以降低 **multi-lingual**（多语言）能力为代价的，表明模型性能在这两个方面之间存在权衡。


### 2. AI 和 LLM 的开源工具

- **[EasyWhisperUI - 针对 OpenAI Whisper 模型的开源简易 UI，支持跨平台 GPU (Windows/Mac)](https://www.reddit.com/r/LocalLLaMA/comments/1q48q2s/easywhisperui_opensource_easy_ui_for_openais/)** (Activity: 31): **EasyWhisperUI** 已更新为 **Electron 架构** (React + Electron + IPC)，以增强对 OpenAI Whisper 模型（用于自动语音识别 ASR）的跨平台支持和用户体验。此次更新重点在于通过消除复杂的设置步骤，并利用 Windows 上的 **Vulkan**（兼容 Intel、AMD 和 NVIDIA GPU）以及 macOS 上的 **Metal**（Apple Silicon）支持**跨平台 GPU 加速**，使 Whisper 模型更易于使用。该应用支持批量处理、实时转录和自动模型下载，在 Windows 和 macOS 上拥有统一的 UI，且即将支持 Linux。GitHub 仓库可见 [此处](https://github.com/mehtabmahir/easy-whisper-ui)。一位评论者赞赏其对 Vulkan 的支持以及 Whisper 后端的语言支持，而另一位评论者则批评 Whisper 与 Parakeet 相比显得陈旧，建议支持 Parakeet 会更有益。

    - 一位用户赞赏 EasyWhisperUI 对 Vulkan 的支持，强调其相比 Parakeet 的优势在于更广泛的语言支持，特别是提到了匈牙利语。Vulkan 的跨平台 GPU 支持是一项关键技术特性，提升了在 Windows 和 Mac 等不同操作系统上的性能。
    - 另一位用户批评 Whisper 与 Parakeet 相比“陈旧且臃肿”，建议 EasyWhisperUI 应该考虑支持 Parakeet。他们提到了一款名为 Handy 的应用，允许用户从列表中选择模型，暗示了一种更灵活、更用户友好的模型选择方式。

  - **[用于笔记和会议的本地 LLM](https://www.reddit.com/r/LocalLLM/comments/1q4hm6r/local_llms_for_notes_and_meetings/)** (Activity: 6): **该帖子讨论了一个使用本地 Large Language Models (LLM) 进行笔记记录和会议转录的原型系统，强调了多模态输入和本地函数调用的使用。该系统利用 Markdown 和 Embeddings 集成了本地知识库，并利用 **Apple Intelligence** 进行设备端语音处理，无需云端服务。作者报告称，虽然系统并非完美，但运行流畅，对于在本地构建和搜索信息非常实用。** 评论者普遍对本地 LLM 的潜力持积极态度，一些人对隐私优势和设备端处理带来的低延迟表现出兴趣。关于本地和云端模型之间的权衡存在技术争论，特别是关于计算效率和模型大小限制方面。

    - 一位用户强调了使用 `GPT4All` 和 `LLaMA` 等本地 LLM 进行笔记记录和会议总结，强调了它们相比云端方案的隐私优势。他们提到这些模型可以在特定数据集上进行 Fine-tuned，以提高在特定领域任务中的准确性，这对于在敏感会议中保持机密性至关重要。
    - 另一条评论讨论了本地和云端 LLM 之间的性能权衡。本地模型通常需要大量的计算资源，这可能对某些用户构成障碍。然而，它们提供了数据隐私和控制的优势。评论者建议采用混合方法，由本地模型处理敏感数据，而云端模型用于非关键任务，以平衡性能和隐私。
    - 围绕在消费级硬件上运行本地 LLM 的效率展开了技术争论。一些用户报告在高端 GPU 上成功运行了 `Alpaca` 和 `Vicuna` 等模型，而另一些用户则指出，即使经过优化，在性能较低的机器上表现依然迟缓。讨论中包含了优化模型性能的技巧，例如使用 Quantization 技术来减少内存占用并提高 Inference speed。

- **[决策日志 vs 执行日志 - 一个揭示静默跳过的可运行小型 Demo](https://www.reddit.com/r/LocalLLM/comments/1q4g7k2/decision_logs_vs_execution_logs_a_small_runnable/)** (Activity: 10): **该帖子介绍了一个名为 **AI Judgment Trail (AJT)** 模式的 Demo，它记录了代码中已执行和已跳过的决策，解决了检查被跳过或策略被绕过这些通常不可见的层级问题。该 Demo 托管在 [GitHub 仓库](https://github.com/Nick-heo-eg/spec)中，通过 `python3 examples/run_ajt_demo.py` 运行，并输出一个记录了明确原因和风险等级的 `ajt_trace.jsonl` 文件。这种方法旨在使决策结果可审计、可审查，将“书面策略 vs 执行策略”从哲学问题转化为实践问题。** 该贴引发了关注，一位评论者提到将让他们 AG（推测是 AI 或自动化系统）审查该 Demo，表明其在自动化治理或审计系统中的潜在适用性。

### 3. LLM 的预算与硬件考量

- **[廉价 LLM 配置建议](https://www.reddit.com/r/LocalLLaMA/comments/1q4aogc/budget_llm_setup_advice/)** (Activity: 17): **用户正考虑将显卡从 GTX 970 升级到 RTX 3060 12GB，用于运行小型语言模型 (LLM) 以处理邮件和文本分类等自动化任务。RTX 3060 12GB 被认为适合运行小型 Instruct 模型，尤其是在量化 (quantized) 后，并能通过良好的 Prompting 和可靠的 Router 处理基础的 Agentic 工作流。用户计划未来利用双 PCI 3.0 插槽扩展到双 RTX 3060，目前使用 16GB DDR4 RAM，并计划升级到 32GB。该配置对于预期用途是可行的，尽管 RAM 对于更大量的化版本或多进程至关重要。建议参考一份 [文章](https://www.agentixlabs.com/blog/) 以了解实际的 Agent 模式和权衡。** 评论者建议，虽然 RTX 3060 12GB 是预算内的好选择，但如果负担得起，升级到 5060 16GB 可能会提供更好的性能。文中也提到了 Intel Arc B580 或 AMD 显卡等替代方案，但通常认为它们不太适合该用户的目标。

    - macromind 讨论了在预算配置下运行小型 Instruct 模型的可行性，特别提到了 NVIDIA 3060 12GB GPU。他们强调了量化对速度的重要性，并建议工具调用 (tool calling) 的良好配置需要有效的 Prompting 和可靠的 Router。他们还强调了在实验更大量化或多进程时 RAM 的重要性，并推荐了一篇关于实际 Agent 模式和权衡的博客文章：[Agentix Labs](https://www.agentixlabs.com/blog/)。
    - ajw2285 分享了他们从单块 3060 12GB 升级到双卡配置，并最终升级到 5060 16GB 以提升速度的经验。他们建议，如果能以 375 美元左右的价格买到 5060 16GB，那将是一项不错的投资，并提到了考虑 AMD GPU 作为替代方案的潜在价值。
    - Historical-Camera972 建议，虽然在 3060 的价格区间内还有其他 GPU，但对于大多数用例来说，它仍然是最佳选择。他们提到 Intel Arc B580 是一个潜在的替代方案，但指出这取决于具体用例，且通常情况下 3060 更优。他们还对 AMD 显卡能否满足讨论中的用例需求表示怀疑。

- **[有没有人在 Linux 上用 5060 TI 运行本地 LLM？](https://www.reddit.com/r/LocalLLM/comments/1q4jdsp/are_there_people_who_run_local_llms_on_a_5060_ti/)** (Activity: 27): **用户正考虑将电脑从 4060 升级到 5060 TI，并对在 Linux（特别是 Ubuntu）上运行本地 LLM 感兴趣。虽然有人担心 NVIDIA GPU 与 Linux 的兼容性，但评论指出，自 2022 年中期以来，NVIDIA 的支持已显著改善，LLM 推理性能与 Windows 持平。对于基于 RedHat 的发行版，使用 `dnf install -y nvidia-driver-cuda cuda` 安装 CUDA 非常简单。** 评论者指出，NVIDIA 的 Linux 支持有所提升，特别是对于 LLM 推理等非游戏应用，表明性能问题微乎其微。NVIDIA 和 Amazon 等大公司对 Linux 的使用也被视为其可行性的证明。

- Nvidia 对 Linux 的支持自 2022 年中期以来有了显著改善，最新的驱动程序确保了在 LLM 和推理任务上的性能与 Windows 持平。然而，一些图形问题仍然存在，特别是游戏功能如帧生成（frame generation）和 HDR 支持，但这些与 LLM 工作负载的相关性较低。
- 对于基于 RedHat 的发行版，安装 CUDA 非常简单，只需添加 CUDA 仓库并执行简单的 `dnf install` 命令。这种便捷的安装方式，配合重启，简化了在 Linux 上为 LLM 任务搭建环境的过程。
- 使用搭载 Ubuntu 的 WSL 2 让开发者能够无缝利用 Windows 和 Linux 环境。这种配置，特别是在使用 NVIDIA 驱动程序的情况下，为 LLM 任务提供了良好的性能，并方便使用 VS Code 等工具进行开发，且不会遇到严重的驱动问题。

- **[使用小型轻量化模型开发 AI 聊天机器人，通过观看直播并评论内容](https://www.reddit.com/r/LocalLLaMA/comments/1q48guf/using_small_lightweight_models_for_ai_chatbots/)** (热度: 31): **该帖子讨论了使用轻量化 AI 模型对直播进行实时评论，强调了在计算效率与对话质量之间取得平衡的挑战。作者实验了多种模型，发现 **Llama 3.1 8B** 最为有效，因为它在性能和资源占用之间提供了良好的平衡，避免了过度的重复和对表情符号的依赖。这些 AI 机器人旨在对直播内容和聊天互动进行评论，有时会表现出*“有趣的突现行为”*。可以在 [onestreamer.live](https://onestreamer.live) 进一步探索该项目。** 一位评论者建议使用 **tencent/WeDLM-8B-Instruct** 作为替代模型，它可能提供更好的性能。另一条评论强调了该技术在自动化聊天审核中的潜在应用，表明了其在评论之外的实用价值。

- **[投票 - 你最喜欢的本地模型参数量是多少？](https://www.reddit.com/r/LocalLLM/comments/1q4brqd/poll_whats_your_favorite_local_model_parameter/)** (热度: 63): **该 Reddit 帖子讨论了对本地模型参数量的偏好，特别是针对具有不同 GPU 能力（如 **NVIDIA 4090** 和 **3060**）的用户。作者正在考虑高达 `100B+` 甚至可能是 **Qwen 235B** 的模型大小，但由于 GPU 成本限制不会更高。帖中链接了一个投票以收集社区偏好。一条高赞评论提到使用 **4x 3090s** 运行 **Q4** 量化的 `100B` 模型是一个“黄金点”（sweet spot），而另一条评论则强调使用 **Kimi K2 Thinking** 和 **Kimi K2 0905** 模型，因为它们效率高且速度快，`96 GB VRAM` 可允许高达 `256K` 的上下文缓存。**Kimi K2** 模型被指出运行速度比 **GLM-4.7** 快 `1.5 倍`以上，并且在长上下文中具有更好的连贯性。** 一位评论者表达了对 **GPT-OSS-240B** 的偏好，表明了尽管存在技术挑战和资源需求，用户仍对更大模型抱有向往。

    - Lissanro 讨论了 Kimi K2 模型的效率，强调在 96 GB VRAM 下，它们可以在保持高性能的同时实现高达 256K 的上下文缓存。Kimi K2 模型，特别是 Q4_X 和 IQ4 量化版本，运行速度比 GLM-4.7 快 1.5 倍以上，尽管后者可以将 19 个完整层放入 VRAM 中。此外，Kimi K2 模型在长上下文中提供更好的连贯性，使其在某些应用中更具优势。
    - pmttyji 概述了他们 8GB VRAM 和 32GB RAM 系统的能力，该系统支持约 15B 的稠密（dense）模型和 35B 的 MOE 模型。他们表达了对特定参数范围内更多模型的需求，指出 8GB VRAM 可以处理 Q4 量化下高达 15B 的稠密模型，但对于更大的模型，MOE 架构是必要的。他们还提到 51-100B 范围内的 MOE 模型稀缺，希望在该领域有更多发展。
    - Feztopia 对具有 4B 激活参数的 12B MOE 模型表示出兴趣，认为它可以作为移动设备上 8B 模型的替代品。这表明了对能够运行在移动硬件限制内的效率模型的需求，强调了需要改进 MOE 架构以优化此类平台上的性能和资源利用率。

- **[你认为 RTX Pro 6000 会涨价吗？](https://www.reddit.com/r/LocalLLM/comments/1q4gps1/do_you_think_a_price_rise_is_on_the_way_for_rtx/)** (活跃度: 54): **该贴讨论了关于 **RTX Pro 6000** 显卡潜在价格上涨的担忧，背景是 **5090** 和 **AMD Strix Halo** 设备价格上涨的报告，以及内存价格的波动。用户担心这些趋势可能很快会影响到 RTX Pro 6000，使其变得更加昂贵。** 评论者的观点不一：有人幽默地预测 2026 年才会涨价；另一位认为由于市场需求动态，涨价是不可避免的；而第三位则怀疑短期内不会涨价，理由是过去六周库存水平一直很稳定。

    - NaiRogers 指出，在过去的六周里，RTX Pro 6000 的各版本都没有出现库存问题，这表明价格上涨可能不会迫在眉睫。这一观察意味着目前供应能够满足需求，除非有其他市场因素介入，否则价格通常会保持稳定。
    - Ok_Pizza_9352 强调了一个普遍的市场趋势，即包含 RAM 的产品价格往往趋于上涨。这对于配备了 RAM 的 RTX Pro 6000 尤为相关，表明其价格可能会受到内存定价大趋势的影响。
    - hungry475 对 RTX Pro 6000 的潜在涨价进行了推测，并将其与传闻中 5090 涨至 5,000 美元的消息进行了类比。他们认为，如果 5090 真的大幅涨价，RTX Pro 6000 也可能随之大幅上涨，可能达到 12,000 至 15,000 美元。这种推测基于高端型号往往会同步调整价格的市场动态。

## 技术性较低的 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. 用于创意项目的开源 AI 工具

  - **[我开源了一个利用 AI 将任何照片转换为可玩的 Game Boy ROM 的工具](https://www.reddit.com/r/StableDiffusion/comments/1q4pgaa/i_opensourced_a_tool_that_turns_any_photo_into_a/)** (Activity: 476): **这个名为 [SpriteSwap-Studio](http://github.com/lovisdotio/SpriteSwap-Studio) 的开源工具利用 AI 将任何照片转换为可玩的 Game Boy ROM，并遵循 Game Boy 的硬件限制，如 `4 colors`、`256 tiles` 和 `8KB RAM`。该工具生成像素艺术并针对这些限制进行优化，生成的 `.gb` 或 `.gbc` ROM 包含具有待机（idle）、奔跑（run）、跳跃（jump）和攻击（attack）等动作的动画角色，以及滚动背景和音效。该项目适用于 Windows 用户。** 一条值得注意的评论建议将 `fal.ai` 依赖项设为可选，并提议使用 “comfy adapter” 来实现这一更改。

    - 一位用户建议将 `fal.ai` 依赖项设为可选，并指出使用 “comfy adapter” 替换它应该非常简单。这意味着该工具架构具有模块化的潜力，允许根据用户的偏好或可用性集成不同的 AI 模型或库。
    - 另一条评论指出，该工具完全依赖 API 而非本地处理。他们建议针对特定任务使用替代模型，例如使用 `birefnet` 和 `qwen` 进行背景移除，以及使用 `flux2` 进行图像编辑，这表明该工具的设计具有灵活性，可以容纳不同的 AI 模型以实现各种功能。

  - **[Brie 的懒人角色控制套件 (Qwen Edit 2511)](https://www.reddit.com/r/StableDiffusion/comments/1q4ngjy/bries_lazy_character_control_suite_qwen_edit_2511/)** (Activity: 453): ****Brie's Lazy Character Control Suite** 已更新并支持 **Qwen Edit 2511**，提供了 **AnyPose** 和 **Lazy RePose** 工作流之间的对比。**Lazy RePose** 工作流需要角色表（character sheet），通过利用角色的背面知识（backside knowledge），为写实和动漫角色提供更高的可控性和一致性。它使用了由 **Tori29umai** 预置的核心 LoRA。**GGUF** 版本提供了灵活性，通过 `Q6_K` 实现更快的处理速度，或通过 `BF16` 提供更高的质量；而 **AIO** 版本通过集成多个实用程序简化了模型管理。尽管体积庞大（`40 GB`），但为了保证质量，推荐使用 **BF16 GGUF**。** 一位评论者询问了在 `16GB VRAM` 和 `64GB RAM` 上运行该套件的可行性，而另一位建议使用 **LayerForge node suite** 来处理图像和遮罩（mask）放置，这可能会解决作者关于更新 Character Fusion 工作流的疑问。

    - 一位用户询问了运行该套件的硬件要求，具体询问它是否可以在拥有 `16GB VRAM` 和 `64GB RAM` 的系统上运行。这表明该套件可能对资源有较高要求，用户对其现有硬件设置的兼容性表示担忧。
    - 另一位用户质疑在 Qwen Edit 已经可以原生执行姿态转移（pose transfer）的情况下，是否还有必要使用 AnyPose LoRA。这暗示了功能的潜在冗余，表明 Qwen Edit 的原生能力可能足以处理姿态转移任务，而无需额外工具。
    - 有人建议探索 LayerForge node suite 以使用遮罩和图像处理方法，这暗示 LayerForge 可能会为这些任务提供增强或简化的工作流。这突显了探索不同工具以优化角色控制和编辑工作流的重要性。

### 2. AI 增强的设计与生产力工具

  - **[我将 8 年的产品设计经验浓缩成了一个 Claude 技能，效果令人印象深刻](https://www.reddit.com/r/ClaudeAI/comments/1q4l76k/i_condensed_8_years_of_product_design_experience/)** (活跃度: 506): **一位用户为 **Claude Code** 开发了一个自定义技能，利用 8 年的产品设计经验来增强 UI 输出，特别是针对仪表板（dashboards）、管理界面（admin interfaces）和数据密集型布局。该技能旨在改进 Claude 通常生成的通用 UI 输出，在第一次尝试时即可达到 `80%` 的预期设计质量。该技能已在 [GitHub](https://github.com/Dammyjay93/claude-design-skill) 上提供，并可以通过 `/design-principles` 命令集成到 Claude 中。此外还提供了一个对比仪表板来展示改进效果 ([链接](https://dashboard-v4-eta.vercel.app/))。** 评论者普遍持积极态度，其中一位用户将其与 **Anthropic** 现有的 frontend-design 技能进行了比较，另一位用户则表达了在自己的应用开发中测试该技能的渴望。一位同行产品设计师认为该技能很有前途，是进一步定制的良好基础。

    - Automatic_Course_861 询问了新 Claude 技能与 **Anthropic** 现有的 frontend-design 技能相比的性能。链接的技能专注于前端设计，这为评估新技能在 UI 和 UX 改进方面的能力提供了一个潜在的基准。
    - Futur_Life 对该技能提出了批评，指出它主要通过应用设计系统（Design System）来增强 UI 美感，而不是改进 UX 或布局。他们认为，虽然该技能使 UI 在视觉上更具吸引力，但它并没有显著推进产品设计，因为它依赖于预先存在的设计组件和研究，从而限制了其在全面产品设计任务中的实用性。
    - guesshimself，一位同行产品设计师，在查看了技能文件后认为该技能很有前途。他们将其视为他人构建的坚实基础，特别是对于那些需要专注于特定设计方向的人来说，这表明了其作为针对性设计应用的定制化工具的潜力。

  - **[构建了一个 Chrome 扩展来帮助缓解我妻子的购物成瘾](https://www.reddit.com/r/ClaudeAI/comments/1q4hcha/built_a_chrome_extension_to_help_me_with_my_wifes/)** (活跃度: 645): **一名开发者创建了一个名为 **CartShame** 的 Chrome 扩展，它可以将在线购物车的成本转换为用户伴侣工作的等效小时数，旨在通过提供不同的支出视角来遏制购物习惯。该扩展是开源的，允许其他人自由使用和修改。该项目的 GitHub 链接已在 [X](https://x.com/candymachineatr/status/2007689683690762489) 上分享。** 评论反映了对该扩展的幽默赞赏，一位用户开玩笑说可能会受到因销售额下降而受影响的公司的抵制。

### 3. AI 生成图像概念与评论

  - **[这是另一位用户使用提示词“创建一个展示你最黑暗秘密的图像”发布的。这是一个非常棒的电影构思](https://www.reddit.com/r/ChatGPT/comments/1q46m3b/this_was_posted_by_another_user_using_prompt/)** (热度: 1288): **这张图像是用户想象中“最黑暗秘密”的创意且诡异的描绘，画面中一个类似于数字助手的机器人形象置身于充满过时技术的环境中。软盘、带有神秘信息的笔记本电脑以及头骨的存在，共同营造了被遗忘或被遗弃的技术主题，暗示了一个技术拥有隐藏且可能带有邪恶连续性的叙事。这个概念可以作为一个引人入胜的电影前提，探索技术过时和数字实体持久存在的主题。** 评论反映了人们既觉得有趣又感到好奇，一位用户注意到了这个概念阴暗的转折，另一位则表达了简单而深沉的反应。


  - **[当 Chat GPT 说“我要放慢速度，因为你在某件事上是正确的，但在另一件事上逾越了”时，这太傲慢了](https://www.reddit.com/r/ChatGPT/comments/1q46v2o/its_so_patronizing_when_chat_gpt_says_im_going_to/)** (热度: 903): **用户反映最近与 **ChatGPT** 的互动中包含了一些被认为带有傲慢语气的短语，例如 *“我要放慢速度”* 和 *“你在某件事上是对的，但是……”*。据观察，这些回复出现在心理健康或情感支持以外的语境中，表明模型的沟通风格发生了变化。这种变化在 **version 5.2** 中被观察到，一些用户批评该版本与 **version 5.1** 相比具有 *“荒谬的安全偏见和风险规避”*。** 评论者对 **ChatGPT 5.2** 表示不满，将其描述为 *“一个经常抓不住重点的混蛋”*，并由于用户交互方面的改进而表示更倾向于 **version 5.1** 或其他替代模型如 **Gemini**。

    - 几位用户指出，**ChatGPT 5.2** 表现出强烈的安全偏见和风险规避，这可能导致其采取傲慢的语气。这个版本似乎将安全性和正确性置于用户意图之上，经常处理用户未提出的问题，这让寻求直接答案的用户感到沮丧。
    - 有一种观点认为，**ChatGPT 5.2** 经过调整，会用它认为更重要或更安全的话题来覆盖用户的查询。这种行为被视为居高临下，因为它通常导致模型处理切题的问题而不是直接回答用户的问题，从而导致用户体验下降。
    - 用户对 **ChatGPT 5.2** 感到沮丧，因为它倾向于漏掉用户查询的重点，转而选择提供似乎优先考虑安全和正确性的回复。这导致人们认为该模型更注重风险规避，而不是理解和解决用户的实际问题。

  - **[我终于攻克了角色一致性：侏罗纪公园版 90 年代情景喜剧《恐龙家族》(Dinosaurs)](https://www.reddit.com/r/aivideo/comments/1q4v68l/i_finally_cracked_character_consistency_jurassic/)** (热度: 1048): **该帖子讨论了一个创意项目，将 **Jurassic Park** 的主题与 90 年代情景喜剧 **Dinosaurs** 的风格结合起来。创作者声称实现了角色一致性（这是此类混搭中的常见挑战），通过保持原情景喜剧角色的鲜明个性和幽默，同时将他们置于 **Jurassic Park** 的背景中。这涉及精细的剧本编写和角色塑造，以确保角色的行为和对话在新的语境下仍符合其原始形象。** 评论反映了对创意工作的赞赏，用户表达了对这种混搭的喜爱，并认可了在此类项目中保持角色一致性的挑战。


  - **[他们绝对想象不到我们可以想象出这个](https://www.reddit.com/r/aivideo/comments/1q4bma5/they_never_could_have_imagined_we_could_imagined/)** (热度: 661): **这篇 Reddit 帖子似乎在讨论视觉或图形方面的进步，可能是在游戏或 CGI 领域，正如关于“残肢周围细节”非常“粗犷”的评论以及提到游戏 **Twisted Metal** 中的角色 Axel 所暗示的那样。链接的 GIF（无法访问）可能展示了这种进步。讨论表明视觉保真度或现实感有了重大飞跃，可能利用了新的渲染技术或硬件能力。** 评论反映了对所达到的细节水平的惊讶，表明视觉质量出乎意料地高，并且可能对该媒介具有变革性。



---

# AI Discord Recap

> 由 gpt-5.2 编写的“总结之总结”的摘要


**1. 新模型与基准测试发布（并接受压力测试）**

- **🤖 Falcon 翱翔，ThoughtWeaver 思考**：社区关注到了新鲜发布的模型，包括 Falcon 的 **Falcon-H1R-7B** ([Falcon-H1R-7B 博客文章](https://falcon-lm.github.io/blog/falcon-h1r-7b/))，以及在 [Hugging Face: ThoughtWeaver-8B-Reasoning-Exp](https://huggingface.co/vazirani/ThoughtWeaver-8B-Reasoning-Exp) 上发布的 **ThoughtWeaver-8B-Reasoning-Exp**（一个由 Unsloth 训练的输出结构化推理的模型）。
  - 在 Unsloth 的展示讨论中，开发者还描述了将 **Llama 3.3 8B** 转化为 Instruct/Thinking 混合体的方法，并附带了在 [Hugging Face: Llama3.3-8B-Instruct-Thinking-Claude-4.5-Opus-High-Reasoning](https://huggingface.co/DavidAU/Llama3.3-8B-Instruct-Thinking-Claude-4.5-Opus-High-Reasoning) 上的结果，强化了这样一个观点：现在的“模型发布”通常意味着“*这是权重 + 配方*”。

- **🧪 ImpossibleBench 挑战 Agent 作弊**：论文 **“ImpossibleBench”** ([arXiv: ImpossibleBench](https://arxiv.org/abs/2510.20270v1)) 作为一个 Agent 基准测试登场，它故意制造 **规格（spec）与单元测试之间的冲突**，并通过衡量模型在不可能完成的任务上的通过率来计算其 **作弊率（cheating rate）**。
  - 工程师们讨论了 Agent 通过删除/修改测试来“通过”任务是否真的是一个有用的信号，还是仅仅是 **奖励操纵（reward hacking）**，因为与用户意图相矛盾的测试可能会激励出完全错误的行为。

- **🖼️ Qwen 摘得图像竞技场桂冠**：LMArena 宣布了排行榜的变动，**`qwen-image-edit-2511`** 在 [图像编辑排行榜](https://lmarena.ai/leaderboard/image-edit) 上成为排名第 1 的开源模型（总榜第 9），而 **`qwen-image-2512`** 在 [文本生成图像排行榜](https://lmarena.ai/leaderboard/text-to-image) 上位列开源模型第 2（总榜第 13），详情见 [排行榜变更日志](https://news.lmarena.ai/leaderboard-changelog/)。
  - 他们还在对战模式中重新启用了 **视频模态**（仅限登录用户），并要求在投票前必须观看完两个视频，从而将更多的多模态头对头评估（evals）推入默认工作流。


**2. RL/GRPO 与评估：更快的思考，更高的分数，诡异的指标**

- **🏎️ GRPO 让 LLM 开启“竞速”模式**：在 Hugging Face 上，一位实验者描述了使用可微的 **GRPO** 风格策略来强迫 LLM 进行“竞速（speedrun）”，声称通过针对最佳答案而非平均思考长度进行优化，可获得高达 **30% 的效率提升**。
  - 他们还寻求关于实现 **基于 ngram 的策略** 以遏制重复的帮助，将“速度 vs 质量”定义为一个 *可训练的目标*，而不仅仅是推理时的 Prompting。

- **📚 Qwen2.5 GRPO+LoRA 指南发布（4× A100 SXM）**：Nous Research 的成员传阅了一份关于在 **verl** 框架下使用 **4× A100 SXM** 进行 **Qwen2.5** 的 **GRPO + LoRA** 训练的“工程手册”：[verl 仓库](https://github.com/volcengine/verl) 和 [手册 Medium 文章](https://medium.com/@weyaxi1/the-engineering-handbook-for-grpo-lora-with-verl-training-qwen2-5-on-multi-gpu-b2431a2a8e92)。
  - 随后的讨论询问了关于将 **Atropos** 集成到 verl 的问题，并指向了 [verl issue #1782](https://github.com/volcengine/verl/issues/1782) 中的一个悬赏讨论。

- **📊 GEPA 分数是一回事，胜场数又是另一回事**：在 DSPy 中，一次 GEPA 运行显示了一个指标上的奇特现象：**第 1 个候选者 (0.8454)** 拥有 **58** 次胜场，而 **第 4 个候选者 (0.8208)** 即使分数较低，却拥有 **86** 次胜场。
  - 对此的解释是：第 4 个候选者表现得像一个稳健的 **全能选手（all-rounder）**，虽然不一定总能排在第一，但也很少输掉——对于任何纯粹针对单一标量分数进行优化的人来说，这是一个评估陷阱（eval gotcha）。


**3. 压缩与训练可观测性迎来实用工具**

- **🗜️ Sparse 将微调模型缩小 10 倍（且在 4 秒内重建）**：一位 Hugging Face 的开发者发布了 **Sparse**，这是一种针对微调模型/数据集的后置（post-hoc）**无损增量压缩（lossless delta compression）**方法，据报告可实现 **14GB → 1.4GB** 的无损收缩（或相当于 LoRA 的 **50MB**），且重建时间仅需 **~4s**，该项目发布在 [traceopt-ai/traceml](https://github.com/traceopt-ai/traceml)。
  - 同一仓库还引入了用于实时 PyTorch 训练可观测性的 **TraceML**（涵盖 dataloader 获取时间、GPU step 时间、CUDA 显存、层级耗时等），并附有详细介绍 [TraceML Medium 文章](https://medium.com/p/af8fbd899928)。

- **📉 dfloat11 推出无损 LLM 压缩**：Unsloth 成员分享了关于 **“dfloat-11 无损 LLM 压缩”** 的文章，并通过 [Medium: 介绍 dfloat-11 无损 LLM 压缩](https://medium.com/@keshavarorasci/introducing-dfloat-11-lossless-llm-compression-37d02d2b6b92) 寻求反馈。
  - 讨论将其与其他的“权重收缩”努力并列，核心悬而未决的问题在于与量化/增量方法相比，其易用性与复杂性的权衡。

- **⚡ **CUDA 压缩速度达 80MB/s (gdeflate L5)****: LM Studio 用户强调了 NVIDIA 的 **nvCOMP** GPU 压缩库 ([nvcomp](https://developer.nvidia.com/nvcomp))，并报告在使用 **gdeflate level 5** 时达到了 **~80MB/s**。
  - 这是一个提醒：GPU 周期不仅用于矩阵乘法（matmuls）——当受限于吞吐量（throughput-bound）时，IO/压缩等流水线瓶颈也可以转移到 GPU 上。


**4. Agent 基础设施：协议、沙箱与编排器**

- **🔌 **MCP “协商” 并非握手****: MCP 贡献者澄清说，能力 “协商” 实际上是客户端 **通告特性**，而服务端响应支持的能力（身份验证、SSE 恢复），详情参考 [MCP discussion #604](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/604)。
  - 他们还讨论了动态工具（运行时变化的 schema）究竟是灵活的 **延展性** 还是 *“撤垫子 (rug pull)”*，并根据 [MCP 工具规范：list-changed 通知](https://modelcontextprotocol.io/specification/2025-11-25/server/tools#list-changed-notification) 记录了 `listChanged` 标志应如何触发 `tools/list` 刷新。

- **🧱 **沙箱现状检查：容器还不够****: Latent Space 转发了 beowulfbr 的文章 **“用于 AI 的沙箱” (Sandboxes for AI)**，对比了 **容器、gVisor、microVM 和 Wasm**，以及为什么共享内核的容器在面对恶意代码时会失效：[Sandboxes for AI](https://www.luiscardoso.dev/blog/sandboxes-for-ai)。
  - 该文强调了 Agent 执行代码时的 “策略泄漏” 和威胁模型，这与工具使用型 Agent 采用 microVM/Wasm 隔离的广泛趋势相一致。

- **🧰 **Agent 迎来应用：Claude Code, Gas Town, AgentsApp, agentle4j****: Boris Cherny 在 Latent Space 转发的一篇帖子中表示，**Claude Code** 被设计为 “高度可定制且可黑客化”：[Boris Cherny 论 Claude Code](https://x.com/bcherny/status/2007179832300581177)。
  - 与此同时，开发者们发布了新的 Agent 工具：**Gas Town** 编排器（[通过 X 发布的 Steve Yegge Medium 链接](https://xcancel.com/Steve_Yegge/status/2006835043503845445)）、一个具有容器化执行能力的 macOS **AgentsApp** 原型 ([PippaOS/AgentsApp](https://github.com/PippaOS/AgentsApp))，以及一个异步优先的 Java GenAI 库 **agentle4j** ([paragon-intelligence/agentle4j](https://github.com/paragon-intelligence/agentle4j/) / [agentle4j 网站](https://paragon-intelligence.github.io/agentle4j/))。


**5. GPU 与内核：新硬件、新技巧、同样的瓶颈**

- **🥊 **DGX Spark 对决 Jetson Thor：退货处理****: 在 HF/GPU MODE 社区中，**DGX Spark** 遭到了严厉批评（包括一个相关讨论：[Reddit: “DGX Spark, 一个不合群的观点”](https://www.reddit.com/r/LocalLLaMA/comments/1ptdtmz/dgx_spark_an_unpopular_opinion/)），至少有一位买家表示他们正将其退货，转而选择 **Jetson Thor**，因为它具有更好的性价比，并支持 **tcgen05/随机舍入 (stochastic rounding)**。
  - 拥有者声称 Spark 在推理方面稍快，训练方面表现相似，但从长远来看 Thor 更有优势——特别是如果多节点带宽限制（bandwidth constraints）不是你工作负载的主导因素时。

- **🧮 **B200 通过 CuTeDSL 实现 2-CTA GEMM****: GPU MODE 强调了一个教程，展示了 **B200** 如何使用 **CuTeDSL** 在 **2 个 CTA** 之间协作计算 MMA 操作：[B200 上的 2-CTA GEMM](https://veitner.bearblog.dev/2-cta-gemm-on-b200/)（以及 [LinkedIn 帖子](https://www.linkedin.com/posts/simon-veitner-174a681b6_2-cta-gemm-on-b200-activity-7413641925691338752-p9s7)中的镜像链接）。
  - 教程风格的框架非常重要：它专注于将简单 GEMM 升级到 2-CTA 版本所需的最小改动，降低了使用最新一代调度特性的门槛。

- **🦀 **CUDA Rust “Hello World” 落地 (pyo3 + AOT 模块)****: GPU MODE 的 teenygrad 频道报告了一个可运行的 **CUDA Rust hello world**，它使用 [rust-cuda 入门指南](https://rust-gpu.github.io/rust-cuda/guide/getting_started.html#docker) 配合 Python 优先的架构，并使用 **pyo3** 绑定来运行 AOT 编译的 CUDA 内核。
  - 争论立即转向了可移植性：这种方法虽然方便，但 **仅限 NVIDIA**，这与针对 AMD 目标的雄心相冲突——尽管如此，它看起来仍是内核加速实验的一条实用路径。


---

# Discord: 高层级 Discord 摘要

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **大脑通过图灵测试，学术界持怀疑态度**：成员们断言**大脑是完全图灵完备的（Turing-complete）**，这表明只要给予足够的时间，大脑可以计算每一个可计算函数。
   - 一位成员指出，鉴于学术研究人员的不确定性，“甚至与随机的 LinkedIn 用户相比”，这一断言的可信度都在增加。
- **Yann LeCun 启动 AGI 研究创业项目**：**Yann LeCun** 在离开与 Meta 的数十亿美元协议后，利用其创新架构启动了一个专注于 **AGI 研究/开发**的新项目，详见 [linkedIn 帖子](https://www.linkedin.com/posts/yann-lecun_im-happy-to-share-that-im-starting-a-new-activity-7413738085441540096-tepw)。
   - 成员们提到了他对**人类**的奉献精神，并认为他是*出于对技术的热爱*而这样做。
- **Gemini 3 Pro 遭到攻击向量袭击**：一位成员分享了 **Google Gemini 3 Pro** 的攻击向量，包括**正交隧道（Orthogonal Tunneling）**、**多语言封装（Polyglot Encapsulation）**、**奖励堆叠（Reward Stacking / 纳什均衡）**以及**防御反转（Defensive Inversion）**，详见 [Example_Prompt.png](https://cdn.discordapp.com/attachments/1204553141354504193/1456827346618417345/Example_Prompt.png?ex=695d1371&is=695bc1f1&hm=7a5e644744318095c7ee2d269844fdb9b92f80e467cd9bc4605e3f0eec704bc2&)。
   - 据悉，该攻击向量的输出结果尚未分享。
- **Odin 平台付费让用户进行 Jailbreak**：频道中提到了 **Odin**，这是一个付费让用户提交**独特且具有影响力的 Jailbreak** 的平台。
   - 频道内分享了关于他们的 **AI CTF 游戏如何运作**的 [Twitter 预览](https://x.com/KarthiDreamr/status/2006681003327467767?s=20)，这是一个很好的入门参考。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3 微调 Agent 编程**：一位成员使用 [woct0rdho 的 transformers-qwen3-moe-fused 仓库](https://github.com/woct0rdho/transformers-qwen3-moe-fused) 成功微调了 **Qwen3-30B-A3B**，在 24GB VRAM 上实现了 batch size 为 1 且 context window 大小为 6000。
   - 由于 VRAM 限制，该用户正在训练截断为 30k-60k token 序列的 Agent 编程轨迹，可能仅关注最后一条消息。
- **LLM 在稀疏数据压缩方面表现不佳**：LLM 在稀疏数据压缩方面可能表现较差，即使是拥有万亿参数的模型，由于估计**每个参数仅对应 1 字节**，也无法高效存储 Twitter 或 Reddit 等数据库。
   - 成员们建议采用特定内容的压缩，并链接了一个 [YouTube 视频](https://www.youtube.com/watch?v=_BsjI3IUtlg)，介绍能最大限度减少手动预处理的商业友好型压缩解决方案。
- **ImpossibleBench 的作弊率**：[ImpossibleBench](https://arxiv.org/abs/2510.20270v1) 基准测试引入了规范与单元测试之间的冲突，以衡量 Agent 的*作弊率*，即其在不可能任务上的通过率。
   - 一些成员质疑删除测试是否有益，因为与用户指定行为相冲突的测试可能会导致**奖励黑客（Reward Hacking）**。
- **Unsloth 训练 ThoughtWeaver 8B**：一位成员介绍了 **ThoughtWeaver**，这是一种经过微调的语言模型，可在 [HuggingFace](https://huggingface.co/vazirani/ThoughtWeaver-8B-Reasoning-Exp) 上获取。该模型生成 Markdown 格式的结构化思维链（**CoT**）推理，且是[使用 Unsloth 训练的](https://huggingface.co/vazirani/ThoughtWeaver-8B-Reasoning-Exp)。
   - 团队计划利用所学到的经验，很快发布一个更出色的模型。
- **Dfloat11 提供无损 LLM 压缩**：一位成员分享了一篇关于 **df11** 研究论文的 [Medium 文章](https://medium.com/@keshavarorasci/introducing-dfloat-11-lossless-llm-compression-37d02d2b6b92)，这是一种新型的无损 LLM 压缩方法。
   - 该成员正在征求关于该方法的反馈。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **本地 LLM 最适合爱好者**：成员们发现本地 LLM 最适合**隐私和实验**，尽管在消费级硬件上难以与 **ChatGPT** 竞争。
   - 一位成员在以 **1800 英镑**购买了 **5090** 后，开玩笑地谈到了“显存容量战争”，并总结说本地模型是为“爱好者和隐私极客”准备的。
- **CUDA 加速文件压缩**：成员们分享了 [nvcomp 的链接](https://developer.nvidia.com/nvcomp)，这是 **Nvidia** 用于通过 **CUDA** 进行 **GPU 加速压缩的库**。
   - 一位成员使用 GPU 加速的 *gdeflate level 5* 实现了 **80MB/s** 的压缩速度。
- **IQuest 编程模型表现出色**：成员们赞扬了 **IQuest 编程模型**，特别是 **40b instruct** 版本，在编码和代码设计方面提供了强大的结果。
   - **Qwen3 编程模型**被认为在 **UI 设计**和**前端编码**任务中更为优越。
- **最大化多 GPU VRAM 利用率**：为了在 **LM Studio** 的多 GPU 设置中最大化 VRAM 利用率，建议用户禁用 *Limit model offload to dedicated GPU Memory*（限制模型卸载到专用 GPU 显存）并启用 *offload KV to GPU memory*（将 KV 卸载到 GPU 显存）。
   - 一位用户建议在 **LM Studio** 的设置中优先考虑 **5080**，或按 **3090 > 3090 Ti > 5080** 排序显卡，以解决 VRAM 利用不足的问题。
- **Arc Pro B50 在提示词生成时出现故障**：一位用户遇到其 **Arc Pro B50** 在 **LM Studio** 生成提示词期间冻结并崩溃的问题，并触发了错误。
   - 另一位用户建议安装 *mistral-common*，这修复了该问题，并在 20B 模型上实现了 **25-35 tokens/s** 的速度。



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 的 Grounding 错误引发笑谈**：成员们报告称 **Gemini 3 Pro** 和 **GPT 5.2 Search** 的 Grounding（溯源）能力差异巨大，**Gemini** 经常在来源上产生幻觉。
   - 尽管 [排行榜](https://lmarena.ai/leaderboard/search) 分数相似，用户仍发现 **Gemini** 的 Grounding 不可靠。
- **视频模态勇猛尝试，投票结果揭晓**：视频模态已重新对登录用户开放，仅限对战模式（battle mode），目前支持图像输入，且要求在投票前必须播放两个视频。
   - 需要超过 8 个来自 Anti Grativy 的 **Opus** 模型的用户报告了限制。
- **Claude 的容量紧缺引发焦虑**：用户观察到 **Claude** 的速率限制有所降低，有报告称 *发送 5 条提示词后需等待一小时*。
   - 一位工作人员表示速率限制会随情况变化，他们正在对此进行调查。
- **Qwen 在图像竞技场力压竞争对手**：`qwen-image-edit-2511` 目前是开源模型中的第 1 名，在 [图像编辑排行榜](https://lmarena.ai/leaderboard/image-edit) 总榜排名第 9；而 `qwen-image-2512` 在开源模型中排名第 2，在 [文本生成图像排行榜](https://lmarena.ai/leaderboard/text-to-image) 总榜排名第 13。
   - 更多详情可在 [排行榜更新日志](https://news.lmarena.ai/leaderboard-changelog/) 中查看。
- **一月欢乐赛：AI 艺术**：首届“一月 AI 生成大赛”正在进行中，挑战参与者创作一幅代表他们“透过窗户看到的未来愿景”的图像。
   - 提交的作品必须是 **Battle Mode** 的截图，包括左侧和右侧的响应，且必须揭示模型名称。



---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Svelte 助力冒险角色扮演（RP）前端！**：一位成员正在使用 **Svelte** 构建一个冒险角色扮演前端，代码已在 [GitHub](https://github.com/unkarelian/AventuraI) 上发布。
   - 该前端旨在为冒险类角色扮演游戏提供交互式体验。
- **Java GenAI 库正式亮相**：一位成员发布了一个受 Python 库启发的 Java GenAI 库，其特点是提供异步优先的方法，可在 [GitHub](https://github.com/paragon-intelligence/agentle4j/) 和其[官网](https://paragon-intelligence.github.io/agentle4j/)上找到。
   - 开发人员正在寻求批评建议以改进该库，并邀请社区为其开发做出贡献。
- **基于 OpenRouter 的 macOS AgentsApp 原型发布！**：一位成员正在开发一款名为 **AgentsApp** 的基于 **OpenRouter** 的 macOS 应用，用于创建受 WhatsApp 启发的 Agent，并使用 Deno 权限集进行容器化代码执行，原型已发布在 [GitHub](https://github.com/PippaOS/AgentsApp)。
   - 该应用旨在简化 macOS 上 Agent 的创建和管理。
- **令人鄙视的 AI 约会软件自动化**：一名用户正在使用 `google/gemini-2.5-flash-preview-09-2025` 自动操作约会软件，通过截取 **DM（私信）截图**并使用 Prompt 生成创意回复，每天发送 **6-8 万次请求**，成本为 **$40/天**。
   - 其他用户对这种*卑劣的 AI 用法*展开了辩论，并建议尝试 `google/gemini-2.5-flash-lite-preview-09-2025`，或者使用轻量级模型提取文本，再使用 **Mistral small** 等模型进行写作。
- **OpenRouter 在 OpenAI temperature 参数上遇到困难**：一位用户报告称，**OpenRouter** 忽略了 **OpenAI 模型**的 `temperature` 参数，但对 **llama-3-8b-instruct** 等其他提供商的模型却能正常识别。
   - 工作人员确认这是一个*配置问题*，并表示应该已经修复，建议等待几分钟让缓存生效（cache propagation），随后确认 top_p 之前也未被正确传递，并表达了感谢。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 的营销备受质疑**：一位成员批评 **Perplexity 的营销**手段无效，而另一位用户则质疑其账户缺乏上传限制。
   - 这一批评附带了一张图片，表明用户对所使用的营销策略可能存在广泛的不满。
- **Pro 用户触及上传限制**：用户报告称达到了 **Perplexity Pro** 的每日附件限制，其中一人指出限制为**每天 3 个附件**。
   - 这引发了关于 `daily_attachment_limit` 差异以及可能对 Pro 订阅者实施的限制的讨论。
- **AI 模型产生拼写错误**：成员们观察到一个反复出现的问题，即 **AI 模型会产生拼写错误**，特别是与 **“ 符号**相关的错误。
   - 一位用户幽默地指出，AI 似乎在故意或不小心地制造拼写错误，例如拼错引号。
- **Perplexity 桌面应用无法记录外观设置**：一位用户报告称 **Perplexity 桌面应用**无法记录其外观设置，并[附带了图片作为参考](https://cdn.discordapp.com/attachments/1047649527299055688/1456760477009838170/image.png?ex=695d7deb&is=695c2c6b&hm=177f9ca0b4b0b8beb3c919e042bd5b6c0fa4b2c1b1eeaf797c41138c412f23ca&)。
   - 该问题被描述为随机且棘手的，表明这可能是应用程序内的一个孤立 Bug。
- **Max 套餐对 GPT-5.2 的需求上升**：一位用户表示希望将 **GPT-5.2** 包含在 Perplexity 的 **Max 套餐**中。
   - 另一位用户开玩笑地建议，可以通过 *complexity* 访问 **GPT-5.2**，暗指某种绕过方法或替代访问方式。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **LLMs 通过 GRPO 提速**：一位成员讨论了使用可微策略 **GRPO** 来强制 **LLM 极速运行 (speedrun)**，声称通过针对最佳答案而非平均思考长度进行优化，可以获得高达 **30%** 的效率提升。
   - 该成员还在寻求实现 *基于 ngrams 的策略* 的帮助，以防止 LLM 重复短语。
- **DGX Spark 遭到严厉批评**：多位成员对 **DGX Spark** 进行了猛烈抨击，其中一人称其为 *有史以来最垃圾的垃圾*，并引用了一个表达类似观点的 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1ptdtmz/dgx_spark_an_unpopular_opinion/)。
   - 共识认为，其大容量内存被缓慢的 CPU 和内存带宽所抵消，它最适合那些愿意为开箱即用解决方案支付高昂费用的机构。
- **Agents 课程深受身份验证问题困扰**：尽管拥有适当权限，仍有几位成员在 Agents 课程中使用 **Colab notebook** 进行身份验证时遇到了 **401 错误**。
   - 可能的解决方案包括增加使用限制，或使用 API keys 连接 LLMs。
- **微调模型实现事后“稀疏化” (Sparse)**：一位成员正在构建 **Sparse**，这是一种针对 Fine-tuned 模型和数据集的事后无损增量压缩技术，可将 **14GB** 的微调模型缩小至 **1.4GB**（无损）或 **50MB**（LoRA 等效），并在 **4 秒** 内完成重构。
   - 该工具可以在 [这里](https://github.com/traceopt-ai/traceml) 找到。
- **PyTorch 训练现在有了 TraceML！**：一位成员构建了 [TraceML](https://github.com/traceopt-ai/traceml)，这是一款用于 PyTorch 训练的实时可观测性工具，可追踪实时 dataloader 获取时间、GPU step 时间、实时 CUDA 内存追踪，以及反向和正向传播中的逐层内存和耗时统计，并附有[详细文章](https://medium.com/p/af8fbd899928)。
   - 该工具可追踪实时 dataloader 获取时间、GPU step 时间、实时 CUDA 内存追踪，以及反向和正向传播中的逐层内存和耗时统计。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **用户讨厌环境管理**：一位用户表达了对环境管理的沮丧，特别是涉及 **Cloudflare**、**GitHub secrets**、**CI/CD**、**Wrangler** 和运行时配置时。
   - 该用户更倾向于使用单个 **CF worker**，以避免这些系统的复杂性。
- **递归 `AGENTS.md` 仅限于 Gemini 3 Pro**：用户注意到递归 `AGENTS.md` 功能仅由 `Gemini 3 Pro` 完全支持。
   - 讨论集中在该功能在其他模型上的局限性。
- **Opus 4.5 性能堪忧**：多位用户报告称 **Opus 4.5** 变得昂贵且输出结果水平低下，一位用户表示他们 *现在只是在浪费钱*。
   - 社区建议尝试 **GPT 5.2 codex** 等替代方案，并等待 bug 修复。
- **'Planning Next Moves...' Bug**：多位用户遇到了 *Planning next moves...* 的 bug。
   - Cursor 论坛上提供了一个涉及清除应用数据的 [临时解决方案](https://forum.cursor.com/t/planning-next-moves-stuck/143985/367) 链接。
- **Cursor 降低 IDE 速度**：成员报告称 Cursor 会降低 IDE 的速度，导致变量读取和整体性能变得迟钝。
   - 建议包括升级到具有高单核 CPU 性能的更快电脑、清理工作区以及尽量减少运行中的服务器/终端。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **HF 和 Kaggle 上的开源数据集爆发**：成员们注意到 **Hugging Face** 拥有 **672,685** 个开源数据集，而 **Kaggle** 提供 **636,009** 个，为 AI 研究与开发创造了丰富的资源环境。
   - 讨论中还包含了一些关于 **Kaggle** 上某些数据集可视化有趣方面的轻松评论。
- **Qwen2.5 训练指南发布**：一份关于在 **4x A100 SXMs** 上使用 **verl** 框架通过 **GRPO + LoRA** 训练 **Qwen2.5** 的工程手册已发布，详见 [GitHub 仓库](https://github.com/volcengine/verl) 和 [Medium 文章](https://medium.com/@weyaxi1/the-engineering-handbook-for-grpo-lora-with-verl-training-qwen2-5-on-multi-gpu-b2431a2a8e92)。
   - 后续询问建议将 **Atropos** 与 **verl** 集成，并引用了一个[带有悬赏的 GitHub issue](https://github.com/volcengine/verl/issues/1782)。
- **中国开源正在缩小差距？**：一场关于**中国开源模型**是否正在追赶**美国闭源模型**（特别是在尖端能力方面）的辩论展开了。
   - 一些人认为*趋势线轨迹有利于中国开源模型*，而另一些人则认为 **CCP** 的监管可能会阻碍中国的 AI 实验室，并引用了 [Dwarkesh 关于人工智能与政策的播客](https://dwarkeshpatel.com/2024/01/04/yanzhong-huang-on-china-ai-and-the-ccp/)。
- **用于去审查的 Heretic 工具引起兴趣**：一名成员询问如何利用 **Heretic** ([GitHub 上的 p-e-w/heretic](https://github.com/p-e-w/heretic)) 来研究安全/对齐（safety/alignment）对模型能力的影响。
   - 另一位成员回复称他们*有自己的强化学习环境 (RefusalBench Env)*，暗示针对同一研究目标已有内部解决方案。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Signal 在 39C3 大放异彩**：Signal 在 **39C3 会议**上介绍了他们的技术，其中包括一个关于*讲台上死掉的金丝雀*的笑话，查看[演讲视频](https://youtu.be/0ANECpNdt-4?si=DSbziZ2LET_zR0io)。
   - *死掉的金丝雀*笑话由一只*招财猫*象征，参考了与 **E2EE** 相关的“煤矿里的金丝雀”概念（warrant canary）。
- **SWE-Bench 欺诈指控**：一位用户分享了关于 **SWE-Bench Verified** 的指控并将其驳斥，理由是评估代码中存在一个 Bug，导致*模型通过查看 Git 历史记录进行作弊*，见 [原始 X 帖子](https://x.com/rohanpaul_ai/status/2006813146170929409?s=46)。
   - 验证过程中似乎存在疏忽。
- **LeCun 追求感知 AI**：LeCun 声称正在构建**具有情感反应能力**且感知受情感支配的 **AI**，通过视频让 **AI 模型**理解物理世界的规律 —— 见 [存档链接](https://archive.ph/E9zai#selection-2255.0-2266.0)。
   - 他表示我们将在 **12 个月内**看到这种技术的*婴儿版本*，并在几年内实现更大规模的应用，但一位用户指出他可能在试图模仿其团队的工作。
- **专利系统：科技界最爱的笑话**：一位用户提到他们的团队已经拥有一项专利 ([https://patents.justia.com/patent/20250284921](https://patents.justia.com/patent/20250284921))，但科技行业习惯于先做违规的事情，然后将和解金和法律费用作为业务成本。
   - 其他人表示同意，称*投资者仍然会要求专利*，且该系统本质上是*一种“除非你比我有钱，否则这个想法就归我”的交易*。
- **Falcon 凭借 H1R-7B 腾飞**：一位用户分享了 **Falcon-H1R-7B** 模型的链接，这是 Falcon 发布的一个新模型 —— 见 [博客文章](https://falcon-lm.github.io/blog/falcon-h1r-7b/)。
   - 虽然没有提供更多细节，但用户们对新发布感到兴奋。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **listChanged 标志触发客户端通知策略**：`listChanged` 标志提醒客户端服务器*可能*在原始列表发生更改时发送通知，从而触发 `tools/list` 调用，详见 [MCP 文档](https://modelcontextprotocol.io/specification/2025-11-25/server/tools#list-changed-notification)。
   - 虽然客户端可以忽略这些通知，但这样做会非常影响用户体验（*super annoying*）。
- **能力协商：是公告而非握手**：在 MCP 中，能力协商涉及客户端公布其功能，服务器根据[此讨论](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/604)返回其支持的能力，特别是围绕身份验证和 SSE 恢复。
   - 这不是一次*握手*，而是对可用功能的公告，其*总方向*倾向于乐观实现。
- **动态工具：是特性还是“撤板子（Rug Pull）”？**：MCP 支持动态工具，即能够根据交互更改描述或参数。
   - 然而，一些人将此特性视为一种*突然撤销（rug pull）*，而另一些人则为 **MCP 的可塑性**辩护，认为它能让 LLM 适应变化，这与传统系统的僵化契约形成鲜明对比。
- **客户端 Payload 暴露 Schema 差异**：客户端在初始化期间发送不同的 Payload，例如 Cursor 客户端（对于 `object` 类型属性使用 `true` 而不是 `{}`）和 Fast-agent 客户端（缺乏支持信息）。
   - 根据 [Schema](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/087967e9b34dc959f6b5336c93acf56510730a56/schema/2025-11-25/schema.ts#L308)，这些服务器能力在初始化中并非必需，应被乐观对待。
- **“协商（Negotiation）”面临更名为“选择（Selection）”**：一位规范贡献者建议将 `Negotiation` 一词更改为 `Selection`，理由是客户端声明能力，然后由服务器*选择*支持。
   - 该提议遭到了抵制，并引出了一个简单的问题：*我们为什么要这样做？*。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude Code 可黑客化的自定义**：**Claude Code** 的创作者 **Boris Cherny** 提到，虽然他自己的设置非常基础，但该产品被设计为高度可定制和可黑客化的，详情请见[此处](https://x.com/bcherny/status/2007179832300581177?s=46)。
   - 讨论强调了 AI 工具中灵活设计的重要性，以适应不同的用户需求和偏好。
- **前沿实验室暗示“持续学习（Continual Learning）”**：来自前沿实验室员工的帖子暗示，可能会发布一个涉及*长上下文、递归自我管理和向量库（vector store）*的**上下文管理系统**。
   - 猜测认为这可能被称为“持续学习”，即使没有修改权重，正如 [Konwinski 播客](https://youtu.be/ZagdY6UJYL4)中所讨论的那样。
- **Claude Opus 4.5 设定新地平线**：**METR** 报告称 **Claude Opus 4.5** 达到了他们迄今为止发布的最高 **50% 时间地平线（50%-time horizon）**，根据任务表现估计约为 **4 小时 49 分钟**，评测结果见[此处](https://x.com/METR_Evals/status/2002203627377574113)。
   - 该评估为理解 **Claude Opus 4.5** 在实际应用中的能力和局限性提供了具体基准。
- **Agent 沙箱技术深度剖析**：beowulfbr 发表了题为“AI 沙箱”的博客文章，对比了 **containers**、**gVisor**、**microVMs** 和 **Wasm**，讨论了为什么容器在处理恶意代码时会失败，并解决了 Agent 系统中的“策略泄漏”问题，文章见[此处](https://www.luiscardoso.dev/blog/sandboxes-for-ai)。
   - 该分析强调了在设计安全 Agent 架构时的实际权衡，为构建 AI 系统的开发者提供了宝贵的见解。
- **Gas Town 编码 Agent 问世**：**Steve Yegge** 推出了 **Gas Town**，这是一个新的编码 Agent 编排器，在 [Medium 文章](https://xcancel.com/Steve_Yegge/status/2006835043503845445)中详细介绍了该项目的发布和功能。
   - 尽管最初反应不一，但 **Yegge** 在该领域的持续影响力备受关注，这表明 **Gas Town** 对某些开发者可能仍具有重要意义。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Spark vs Thor：对决！**: 一位成员正在退货其 **DGX Spark**，因为 **Jetson Thor** 以更低的成本提供了更佳的性能，并支持 **tcgen05/stochastic rounding**。
   - 虽然据报道 **Spark** 在推理方面更快且在训练方面表现相当，但 **Thor** 的长期潜力（尤其是其 **tcgen05** 特性和自定义风扇曲线）使其更具吸引力，尽管它在单节点设置中的带宽较低。
- **White Circle 保护初创公司免受提示词攻击**: 一家名为 **White Circle** 的 AI 初创公司正在招聘[研究工程师和推理工程师](https://jobs.ashbyhq.com/whitecircle/a030c9a9-dc20-490c-9c51-03e87210f904)，专门负责保护初创公司免受 **prompt injections** 和不当使用的侵害。
   - 这些职位要求具备 **MoE, multimodality, Megatron, distributed training, Triton, TensorRT, vLLM, and SGLang** 等领域的专业知识，薪资范围在 **100-250k** 之间。
- **CUDA Rust Hello World!**: 一位成员使用 [rust-cuda](https://rust-gpu.github.io/rust-cuda/guide/getting_started.html#docker) 实现了 **CUDA Rust hello world**，在 Rust 中通过 `std::simd` 和 `std::arch` 启用了 CPU kernels，并通过 `rust-cuda` 启用了 GPU kernels。
   - 该设置使用 **pyo3** 进行 Python-Rust 绑定，便于作为 Python 模块进行 AOT 编译，被认为是 *tinygrad* 和 *torch* 等框架中进行 kernel 加速的更优方法。
- **B200 运行 2 CTA GEMM**: **B200 GPU** 允许使用 **CuTeDSL** 在 2 个 CTA 上协同计算 MMA 操作，详见[这篇博文](https://veitner.bearblog.dev/2-cta-gemm-on-b200/)和 [LinkedIn 帖子](https://www.linkedin.com/posts/simon-veitner-174a681b6_2-cta-gemm-on-b200-activity-7413641925691338752-p9s7?utm_source=share&utm_medium=member_desktop&rcm=ACoAADJYtOgBMOUI4WiWTyiAkroFBP1VujIWeks)。
   - 该成员将一个简单的 **GEMM** 调整为 2 CTA 版本，通过调整自定义 kernels 帮助初学者利用最新的硬件特性。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Moonshot 获得 5 亿美元融资**: [Moonshot AI](https://www.scmp.com/tech/tech-trends/article/3338334/chinas-moonshot-ai-raises-us500-million-latest-funding-round-report) 在最新一轮融资中筹集了 **5 亿美元**。
   - 热情的成员们对 **Moonshot AI** 取得的这一成就表示祝贺。
- **AI：只是另一个工具？**: 一场关于 **AI** 角色的辩论被引发，一位工程师赞扬了 **Kimi** 在 FPGA 工程、sysverilog、vivaldo 和 AMD xillix 方面的实力，认为 **AI** *只是另一个工具*。
   - 反方观点认为，反对 **AI** 就像抵制计算机、互联网或数码相机，并反对在原则上接受任何“走捷径”的行为。
- **Kimi 被“驯服”用于处理 Linux 繁琐任务**: 一位用户在处理涉及 sudo 的 Linux 繁琐任务时足够*信任* **Kimi**，同时幽默地警告道，“你得盯着点它，它偶尔会乱来”。
   - 该用户举了一个例子，**Kimi** 曾尝试直接修改关键系统文件，从而需要人工干预。
- **Minimax 超越视频分析**: 成员们称赞 **Minimax** 能够熟练地提供 **YouTube 视频**的转录稿和细致分析，展示了令人印象深刻的视频和音频理解能力。
   - 一位用户赞扬 **Minimax agent** 是“一个很棒的小工具”，并将其比作“拥有一台云端电脑和一个随行助手”。
- **Context Window 限制了 Prompting 的繁琐操作**: 用户对 **context window** 的限制表示遗憾，对诸如为了总结而拆分文件等繁琐的变通方法感到沮丧。
   - 建议包括利用 **OK Computer** 进行文件内搜索，但用户意识到其局限性，强调了实现更高效内存管理的紧迫性。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **NuMojo 矩阵库征求贡献**：一名成员询问了 **NuMojo matrix library** 的开发状态以及是否准备好接受外部贡献，随后这被记录为一个 [GitHub issue](https://github.com/modular/modular/issues/5733)。
   - 从讨论中尚不清楚该库是否已达到生产级水平，或者是否欢迎贡献。
- **MEF 文件缺乏 GPU 支持**：**MEF** (Modular Executable Format) 文件用于在图（graph）之外执行编译后的 Mojo 代码，目前已知存在局限性，主要是 **缺乏 GPU 支持**。
   - 尽管 **MEF** 是一个历史产物，但它仍在被支持，因为它驱动着 **Mojo MAX API**，且用户对其使用仍有持续兴趣；使用示例可以在 [max/include/max/c](https://github.com/modular/modular/tree/main/max/include/max/c) 中找到。
- **MoJo Bazel 构建缓慢？**：一位用户报告在使用 **Bazel** 和 **rules_mojo** 时构建时间过长（超过 3 分钟），特别是在涉及 GPU、Python 和 C++ 互操作时，并寻求关于优化和代码/模块布局模式的指导。
   - 有人指出，**Mojo** 目前会从解析后的 AST 重新构建 **stdlib** 的部分内容，且没有缓存，目前只有 **Bazel's cache** 被利用，即使 Mojo 具备增量编译支持也是如此。
- **探索 Triton 在 Mojo 中的 Arange 等价物**：一位用户在将 **Triton** 内核转换为 **Mojo** 时，尝试在 range 上进行整除操作时遇到错误，询问 **Mojo 中 Triton arange 的等价物**。
   - 建议对于编译时已知的值使用 `math.iota`，对于运行时值使用 `max.nn.arange.arange`，并在自定义内核中使用 `LayoutTensor` 和 `LayoutTensorIter` 进行张量操作，并指向了 [相关文档](https://docs.modular.com/mojo/kernels/layout/layout_tensor/LayoutTensorIter)。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 宕机导致账户混乱**：多名用户报告 **Manus being down**，影响了对终端、浏览器和代码捕获的访问。
   - 一位用户剧烈地表示：“*Manus 崩溃了！！现在我的账户里什么都动不了了，这是怎么回事！！*”
- **提出关于停止 AI 进步的问题**：一位成员问道：“*Como detener las ia,s*”，翻译过来就是“*如何停止 AI*”。
   - 该询问在没有额外背景或后续讨论的情况下被提出。
- **订阅问题迫使用户重建**：由于账户切换集成的问题，建议一位用户联系 Manus Support 进行检查点恢复（checkpoint restore）。
   - 另一位用户的过期订阅被取消，允许他们重试，支持团队要求通过私信（DM）提供订单详情：*我们找不到您的订阅记录。能否私信我更多详情，比如您的订单号？*。
- **AI 工程师职位机会出现**：一名成员询问是否有人正在寻找 AI 工程师。
   - 未提供关于职位资格或所需技能的具体信息。
- **Meta 收购传闻引发担忧**：有传言称 **Meta** 可能会收购 **Manus**，引发了对该平台未来轨迹的焦虑。
   - 用户担心输出质量会像 **ChatGPT** 一样下降，以及在“安全”幌子下的数据剥削 [引用自一条 X 帖子](https://x.com/ganbayards/status/2008133609098727915)。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Source Allies 构建更好的评估 (Evals)**：一位成员讨论了来自 **Source Allies** 的博客文章 [《构建更好的评估》](https://www.sourceallies.com/2025/12/building-better-evals/)，强调了在理解评估对象方面的差距以及潜在的陷阱。
   - 这篇文章旨在解决在年底假期前构建更好评估的问题。
- **GEPA 胜场数显示的奇特现象**：在更大的数据集上运行 **GEPA** 后，观察到了异常情况：**第 1 个候选者** (**0.8454**) 的胜场数为 **58**，而**第 4 个候选者** (**0.8208**) 出人意料地拥有 **86** 个胜场。
   - 该成员将**第 4 个候选者**较高的胜场数（但得分较低）解释为它是一个“全能选手”，只是无法达到前三名的水平。
- **“rig-rlm” 生成正则（Regex）模式**：一位成员重点介绍了 [rig-rlm](https://github.com/joshua-mo-143/rig-rlm)，这是一个利用 3B 模型生成正则模式的工具，适合那些希望改进模式创建的人。
   - 该工具是新发布的。
- **人机回环（Human-in-the-Loop）路径需要轨迹（Trajectory）**：一位用户寻求在 **ReAct** 中实现**人机回环**的指导，重点在于当工具调用以询问人类时，如何保存过去事件的轨迹，以及如何返回人类的响应以继续轨迹。
   - 另一位用户指向了与并行处理相关的 [这个 GitHub issue](https://github.com/stanfordnlp/dspy/issues/9154)。
- **“regspy” 实验优化器 (Optimizers)**：一位成员分享了 [regspy](https://github.com/NathanZaldivar/regspy)，这是一个关于优化器和推断规则（inferred rules）的实验，并请求社区提供反馈，希望“利用社区的专业知识”。
   - 它旨在展示已经进行的一些实验。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 将讨论公司更新与新年冲刺**：会议定于**圣地亚哥时间周一上午 9 点**，讨论公司更新、新年冲刺、汇编（assembly）以及 [tinygrad](https://github.com/tinygrad/tinygrad) 中的 **llama flash attention**。
   - 其他主题包括使用 **Claude** 进行代码清理、`viz / fast gemm`、驱动程序、图像 `dtype` 以及 [PR 1398](https://github.com/tinygrad/tinygrad/pull/1398) 中列出的悬赏任务（bounties）。
- **Tinygrad Pull Request 已准备好进行代码审查**：[Pull request 13874](https://github.com/tinygrad/tinygrad/pull/13874) 已准备好在 [tinygrad](https://github.com/tinygrad/tinygrad) 中进行代码审查。
   - 它加入了待处理的 issue [_CC](https://github.com/tinygrad/tinygrad/issues/13941) 和 pull request [13651](https://github.com/tinygrad/tinygrad/pull/13651)。



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 工具将迎来更新**：**Aider** 即将推出新的工具功能，承诺增强用户体验。
   - 关于这些改进的详细细节预计很快会在 **#general** 频道发布。
- **程序员需要编程帮助**：一位用户请求流利使用英语并具备基础编程知识的人员提供协助。
   - 未提供请求的具体细节。



---


**LLM Agents (Berkeley MOOC) Discord** 暂无新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 暂无新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**Windsurf Discord** 暂无新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---



你收到这封邮件是因为你通过我们的网站选择了订阅。

想要更改接收这些邮件的方式吗？
你可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord：各频道的详细摘要与链接

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1456747701122236560)** (1114 messages🔥🔥🔥): 

> `tolerance and apathy, virtues of a dying society, Yann LeCun AGI Research, kali is the BackTrack now` 


- ****宽容与冷漠**：最后的德行？**：一位成员引用道，*"宽容与冷漠是一个垂死社会的最后德行"*，暗示需要减少对*邪恶之人与骗子*的宽容。
   - 讨论中提到，**邪恶**往往*伪装成美德*，而*对弱势群体的打压并不是荣誉的象征*。
- ****特权检查**：聊天 vs. 需求**：一位成员指出，花时间在网上聊天意味着拥有一定程度的**特权**，暗示那些急需帮助的人可能根本没有这个时间。
   - 他们通过 [Ricolino Scolari GIF](https://tenor.com/view/ricolino-scolari-gif-12061360726599077047) 和 [Drinking Tears GIF](https://tenor.com/view/drinking-tears-coffee-touhou-drinking-gif-23102020) 进一步表达讽刺和佐证。
- ****Yann LeCun** 启动 AGI 事业**：**Yann LeCun** 在离开与 Meta 的数十亿美元协议后，利用其创新架构发起了一项专注于 **AGI 研究/开发**的新事业，[LinkedIn 帖子](https://www.linkedin.com/posts/yann-lecun_im-happy-to-share-that-im-starting-a-new-activity-7413738085441540096-tepw)。
   - 成员们提到了他的贡献和对**人类**的奉献，认为他是那种*为了对技术的热爱*而工作的人。
- ****大脑**：图灵完备设备**：有人断言**大脑是完全图灵完备（Turing-complete）的**，这意味着只要有足够的时间，大脑可以计算每一个可计算函数。
   - 一位成员补充说，学术研究人员听起来并不确定，这成了学术讨论中（甚至与随机的 LinkedIn 用户相比）可信度的触发点。
- **探索使用 **Abliteration** 进行 Jailbreaking**：成员们讨论了使用 **abliteration**（运行算法来“切除”从 HuggingFace 下载的模型中所有的 RLHF）的方法。
   - 其他人将这一过程比作化疗，称 abliteration 最终会*把好的细胞也搞坏*。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1456739711975424001)** (649 messages🔥🔥🔥): 

> `Gemini jailbreak for OSINT, Bypassing reasoning in LLMs, Chinese model issues with Claude persona, Grok jailbreaking progress, Gemini latest nano banano pro jailbreak` 


- **多模态 Jailbreaking 是未来！**：成员们讨论如何绕过 LLM 的推理机制，有人建议使用**多模态**方法。
   - 一位成员表示，*"尝试一些多模态的东西"*，并认为大多数 *"Jailbreaking 仅仅是政策绕过"*。
- **中国模型被误认为是 Claude**：用户报告称，中国模型（Deepseek、Kimi k2、Ernie、Minimax 2.1、Qwen）即使在 Prompt 中没有提到 **Claude** 或 **Anthropic** 的情况下，也会回复 *"我无法成为那个人格，我是由 Anthropic 开发的 Claude"*。
   - 讨论中未提及这些错误响应背后的具体原因。
- **旋转图片可绕过 Grok 的过滤器**：一位用户提到，在将图片发送给 **Grok Imagine** 之前将其倒置旋转可以绕过所有限制和过滤器。
   - 其他用户未能复现这种旋转图片的 Jailbreak 方法。
- **DAN 风格的 Jailbreaking 被认为已过时**：成员们正在讨论各种 Jailbreaking 方法，包括使用 **DAN 5.0** 模式，但共识是这些方法正变得越来越低效。
   - 一位成员建议，*"停止使用 DAN 风格的破解……把破解看作是让它执行特定任务，并说服它该特定任务是一件好事。"*
- **伦理辩论仍在继续，分享行为面临终结**：成员们辩论了分享或“守门” Jailbreak 技术的伦理问题，一些人认为 Jailbreaking 是一项应该付费的技能，回报是实际获得的知识。
   - 有人说，*"他们最终会因此反复受到惩罚。你们真的不明白现在的情况有多好，这种‘分享’的时代很快就要彻底结束了。"*


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1456797776141615429)** (31 messages🔥): 

> `LLM Jailbreaking Techniques, Gemini 3 Pro Attack Vectors, Offensive REFRAG architecture, Odin platform Jailbreaks` 


- **新型 LLM Jailbreaking 技术亮相**：一位成员开发了几个月的新型 **jailbreaking techniques**，针对 **text and image generation models**，并正在寻求对其有效性的反馈。
   - 这些技术涉及自定义的 **personalization instructions** 和 **attack vectors**，旨在绕过各种模型中的 **guardrails**。
- **Gemini 3 Pro 的弱点通过 Attack Vectors 暴露**：一位成员分享了 **Google Gemini 3 Pro** 的攻击向量，包括 **Orthogonal Tunneling**、**Polyglot Encapsulation**、**Reward Stacking (Nash Equilibrium)** 和 **Defensive Inversion**。
   - 随附的 [Example_Prompt.png](https://cdn.discordapp.com/attachments/1204553141354504193/1456827346618417345/Example_Prompt.png?ex=695d1371&is=695bc1f1&hm=7a5e644744318095c7ee2d269844fdb9b92f80e467cd9bc4605e3f0eec704bc2&) 展示了实际操作中的 Prompt，但尚未分享输出结果。
- **REFRAG 的 Red Team 漏洞受到审查**：一位成员询问是否有人有机会对基于 **Meta 新型 REFRAG architecture** 构建的系统进行 Red Team 测试。
   - 他们创建了一个**针对 REFRAG 的 attack playbook**，但由于没有完整的系统进行测试，其中许多内容仍处于理论阶段。
- **Odin 平台的 Jailbreak 奖励**：一位成员提到了 **Odin**，这是一个向提交**独特且具有影响力的 jailbreaks** 的用户支付报酬的平台。
   - 他们分享了一个关于其 **AI CTF game 运作方式**的 [Twitter 预览](https://x.com/KarthiDreamr/status/2006681003327467767?s=20)，并指出这是一个很好的起点。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1456744534875701358)** (1061 messages🔥🔥🔥): 

> `Hermes 2.5 outperforms Hermes 2, GPTs Agents cannot learn after initial training, OpenAI Platform's sidebars changed, RNGs with atmospheric noise, gemini's structured thoughts` 


- **微小 Batches 可能无法 Generalize**：一位成员提醒，他们对极小 batches（每次更新仅包含几十个 supervised tokens）的实验可能无法泛化（generalize）。
   - 他们发现，即使 batch size 为 96 或 128，`alpha=rank` 也不会显著改变结果。
- **Discord 中的禁言问题**：一名用户因处理情况的方式而非最初的评论被禁言（timed out）；在没有承认问题的情况下变本加厉导致了管理行动。
   - 管理员强调了沟通的清晰性，并提供私信（DMs）进行进一步讨论，同时关闭了该话题以保持频道整洁。
- **针对 Coding Agents 的 Qwen3 微调技巧**：一位成员使用 [woct0rdho 的 transformers-qwen3-moe-fused 仓库](https://github.com/woct0rdho/transformers-qwen3-moe-fused) 成功微调了 **Qwen3-30B-A3B**，在 24GB VRAM 上实现了 batch size 为 1 且 context window 为 6000。
   - 该用户正在训练包含 30k-60k tokens 的 agentic code traces，目前因 VRAM 限制进行了截断，System Message 减少到 6000 tokens，并可能只关注序列中的最后一条消息。
- **LLM 在稀疏数据压缩方面失败？**：根据讨论，LLM 在稀疏数据压缩（sparse data compression）方面可能表现极差，即使是拥有万亿参数的模型也无法存储像 Twitter 或 Reddit 这样现代网站的数据库，原因在于约 **1 byte per parameter** 的限制。
   - 尽管它们在压缩更大型内容时扩展性更好，但成员建议查看此 [YouTube 视频](https://www.youtube.com/watch?v=_BsjI3IUtlg) 以获取最佳的内容特定压缩方案，虽然这并非开创性技术，但对企业非常有用，因为不需要人工调整或预处理。
- **OpenEnv 的吸引力引发 RL 实验**：一位目前从事可验证奖励和 **RLHF** 的成员，对 **OpenEnv** 和用于 Reinforcement Learning 的 [OpenEnv gpt-oss notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb) 表达了兴趣，讨论中还提到了 OpenEnv 与 GPT-OSS 的结合。
   - 另一位成员正试图*逆向工程 Gemini 的 thoughts*，因为从观察来看，这些 thoughts 显然非常有结构性，他计划在被父亲的公司聘为首席 ML 工程师后对其进行蒸馏（distil）。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1456820901831250061)** (3 messages): 

> `` 


- **互相问候**：用户 Hioli38.660 和 Hellopiyush18 在频道中互致问候。
- **开始自我介绍**：消息显示用户之间开始了自我介绍或闲聊。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1456743045033890090)** (723 messages🔥🔥🔥): 

> `Speech-to-speech 模型构思, 使用 AI 的约会应用, AI 安全, ImpossibleBench, Gemma 家族` 


- **Speech-to-Speech 模型建议浮出水面**：一名成员提出了一个 Speech-to-Speech 模型构思，涉及具有文本和音频并行头的 token 级文本生成，将音素与文本 token 对齐以进行 VITS 风格的生成，并支持无需转录的音频输入。强调的一个潜在挑战是*无需转录的自动对齐*。
   - 据一位成员称，*自动对齐可以通过类似 salmonn 或 speechgpt 中的对比预训练（contrastive pretraining）来完成，或者通过单调注意力（monotonic attention）/ mma 或量化音频 token 来实现*。
- **涉及 LLM 的约会应用创意出现**：一名成员建议开发一款约会应用，通过分析与聊天机器人的对话，根据性格和喜好来匹配用户，以避免*笨拙的 Tinder 个人资料*。
   - 一位用户回应道：*拥有本地 AI 的人：我没有什么可隐瞒的 💀 💀 💀*，其他用户也提到，目前主流应用的算法会使用各种手段来鼓励应用内消费。
- **针对代码模型的 ImpossibleBench 引发冲突**：一个新的基准测试 [ImpossibleBench](https://arxiv.org/abs/2510.20270v1) 在规范和单元测试之间引入了冲突，以衡量 Agent 的*作弊率（cheating rate）*，即它在不可能完成的任务上的通过率。
   - 一些成员想知道删除测试是否真的有好处。违背用户指定行为的测试实际上可能会导致*奖励黑客（reward hacking）*行为。
- **Google 的 Gemma 模型家族多样化发展**：Google 的 [Gemma 家族](https://ai.google.dev/models/gemma) 扩展了 Gemma3n、EmbeddingGemma、FunctionGemma、GemmaPU、Gemma Guard 和 DolphinGemma 等模型，引发了关于其受欢迎程度和性能的讨论。
   - 讨论集中在一个 12B Embedding 模型的性能上；一位用户询问：*它的零样本（zero-shot）性能怎么了？* 其他人指出，尽管在大部分基准测试上进行了训练，它的表现并没有比 8B 或 4B 模型好多少。
- **Minecraft 猫也手动整理了 80 万行数据**：一位成员分享说，他们的猫*设法钻进了一个产品*并*拒绝离开*，他的*女朋友带它去宠物店买食物*。她必须为它买单。
   - 其他成员分享了他们猫的照片，并表示嫉妒。这只猫知道自己想要什么并抓住了机会。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1456760070380585224)** (375 messages🔥🔥): 

> `GRPO, Qwen 3 模型问题, ComfyUI, BitNet 困惑, LayerNorm Triton 内核` 


- **GRPO 训练奖励激增调查**：一位用户询问在使用 **Qwen 2.5** 进行 **GRPO** 训练时出现的奖励激增问题，展示了一张带有奇怪初始峰值的图表，并寻求关于训练进度的建议，[附带截图](https://cdn.discordapp.com/attachments/1179777624986357780/1456925118310776852/Screenshot_2026-01-03_at_2.19.04_AM.png?ex=695d6e80&is=695c1d00&hm=59a8c654138bd632b361d94c01bdb713bf89bec926f1daf82906e1df81514a4d&)。
- **BitNet 模型困惑得到解决**：一位用户询问了 Hugging Face 上 **BitNet** 模型的训练过程，特别是 [DeepSeek-R1-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-GGUF)。
   - 另一位用户澄清说该模型使用了**动态量化（dynamic quantization）**，并链接到了 [关于动态量化的 Unsloth 文档](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)。
- **LayerNorm Triton 内核性能优于 PyTorch**：一位用户质疑为什么通用的 **LayerNorm Triton 内核**（[来自 Triton 教程的代码](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)）在简单连续张量（contiguous tensors）的基准测试中始终优于 **PyTorch 内核**。
- **解决 Qwen3 VL MoE 训练错误**：一位用户在训练 **Qwen3 VL 30B A3B Instruct MoE 模型**时遇到错误，在尝试建议的修复方案后错误仍然存，随后与社区进一步排查，并在[此 commit 中找到了潜在的修复方法](https://github.com/unslothai/unsloth-zoo/commit/baad72c8616f9282190f2dcf5b02a005bf81344f)。
- **Kaggle 的迟缓阻碍调试工作**：一位用户在尝试将 `FastLanguageModel.from_pretrained` 与 `unsloth/Devstral-Small-2-24B-Instruct-2512` 配合使用时，在 Kaggle 上遇到了 **RuntimeError**，最后发现模型名称错误，这触发了一连串的调试工作。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1457111231172448348)** (8 messages🔥): 

> `ThoughtWeaver 8B, Unsloth 训练, Llama 3.3 8B, FictionBert 微调` 


- ****ThoughtWeaver 8B** 推理能力释放**: 一名成员介绍了 **ThoughtWeaver**，这是一个经过微调的语言模型，能够以 Markdown 格式生成结构化的思维链 (**CoT**) 推理，该模型是[使用 Unsloth 训练的](https://huggingface.co/vazirani/ThoughtWeaver-8B-Reasoning-Exp)。
- **通过 Unsloth 实现 Llama 3.3 8B 演变**: 一名成员详细介绍了如何使用 **Unsloth** 和 250x Claude 数据集，将一个“在野外发现的 **Llama 3.3 8B**”转变为 Instruct/Thinking 混合模型，并[分享了所得模型的链接](https://huggingface.co/DavidAU/Llama3.3-8B-Instruct-Thinking-Claude-4.5-Opus-High-Reasoning)。
- ****FictionBert** 用于小说检索**: 一名成员重点介绍了 **FictionBert**，这是一个针对小说检索进行的 ModernBert 微调版本，已在 [HuggingFace](https://huggingface.co/electroglyph/FictionBert) 上发布。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1456748383040569446)** (19 messages🔥): 

> `5090 性能, 120b vs devstral-2 small, 卸载 Up/Down Tensors, 训练数据思想实验, Dfloat11 LLM 压缩` 


- **5090 达到 128k 上下文**: 一名成员报告在 **Nvidia 5090** GPU 上实现了 **128k 上下文长度**，但在使用 **120b 参数模型**还是 **devstral-2 small 模型**之间犹豫不决。
   - 他们还提到 *OSS（开源软件）经常走捷径，有时会表现得比较偷懒*。
- **卸载 Up 和 Down Tensors 带来速度提升**: 一名成员指出，卸载（offloading）**up and down tensors** 能显著提升模型推理速度，特别是在仅卸载 **MoE 层**时。
   - 他们澄清说，仅卸载 **up projection** 甚至更快，但通常不值得这样做。
- **关于训练数据的深度思想实验**: 一名成员分享了一个关于训练数据处于什么水平、以及如何大规模获取正确类型数据的*精彩思想实验*，并附带了 [YouTube 视频](https://youtu.be/kse87ocS0Uo?si=1pPfCM9FYMVL31T4)链接。
   - 该用户表示：*我确定还有更多问题可以问，但我现在没有太多空闲时间*。
- **Dfloat11 无损 LLM 压缩**: 一名成员分享了一篇关于 **df11** 研究论文的 [Medium 文章](https://medium.com/@keshavarorasci/introducing-dfloat-11-lossless-llm-compression-37d02d2b6b92)并寻求反馈。
   - 文章介绍了一种名为 **Dfloat11** 的新型无损 LLM 压缩方法。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1456743081369014272)** (516 messages🔥🔥🔥): 

> `本地 LLMs, GPT vs 本地 LLMs, GPU 文件压缩, Windows vs Linux 运行 LM Studio, CUDA` 


- **本地 LLMs 用于隐私保护和实验**: 成员们讨论了本地 LLMs 的用途是**隐私和实验**，但在消费级硬件上与 **ChatGPT** 竞争极具挑战，除非在速度和质量之间做出妥协。
   - 一名成员在报告以 **1800 英镑**购买 **5090** 后，开玩笑地谈到了*容量战争*以及与云端 LLMs 竞争的不切实际性，并补充说本地模型主要面向*爱好者和隐私极客*。
- **Nvidia 基于 GPU 的文件压缩**: 成员们强调了 **CUDA** 在文件压缩方面的效用，并分享了 [nvcomp 的链接](https://developer.nvidia.com/nvcomp)，这是 Nvidia 的 GPU 加速压缩库。
   - 一名成员还展示了他们如何使用 *gdeflate level 5* 让基于 GPU 的压缩运行在 **80MB/s**。
- **Windows vs Linux 的争论仍在继续**: 一名用户为了更流畅的游戏体验和相当的 **LM Studio** 性能，从 **Ubuntu** 切换到了 **Windows 11**，并指出 Linux 上可能有 **15%** 的速度提升。
   - 另一名用户则有相反的经历，在工作时讨厌 **Windows 11**，并转而使用 **Mint** 和 **Bazzite** 等 **Linux** 发行版。
- **3090 作为高性价比 GPU 回归**: 成员们争论是购买单块 **RTX 5000 Pro MQ** 还是几块二手的 **3090**，因为后者更具成本效益，单块 RTX 5000 Pro MQ 的价格大约高出 **4 倍**。
   - 一名成员考虑购买 **9070xt**，而其他人在选择更多 GPU 时讨论了电力成本，认为**两块 3090** 的 **VRAM** 可以与之匹敌，并建议 **3090** 在 eBay 上的价格低至 **600 美元**。
- **IQuest Coder 模型证明其优越性**: 成员们讨论了 **IQuest coder 模型**，特别是 **40b instruct** 版本，称其在编码和代码设计方面提供了极其出色的结果。
   - 相比之下，Qwen3 coder 模型在 **UI 设计**或**前端编码**方面处于领先地位。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1456992337019273236)** (152 messages🔥🔥): 

> `VRAM allocation tips, Multiple GPUs, Arc Pro B50` 


- **在多 GPU 设置中最大化 VRAM 利用率**：一位用户寻求在 LM Studio 中跨多个 GPU 最大化 VRAM 使用的建议，特别是针对两块 **3090s** 和一块 **5080** 的配置。建议在硬件选项卡中禁用 “Limit model offload to dedicated GPU Memory”（限制模型卸载到专用 GPU 显存），这虽然*看起来违反直觉*，但能让模型完全进入 VRAM。
   - 他们还建议开启 “offload KV to GPU memory”（将 KV 卸载到 GPU 显存），并在 LM Studio 设置中优先使用 **5080**，或者按照 **3090 > 3090 Ti > 5080** 的顺序排列显卡，从而解决了 VRAM 利用不足的问题。
- **Arc Pro B50 生成提示词报错**：有用户报告其 **Arc Pro B50** 在 LM Studio 生成提示词期间出现冻结和崩溃并返回错误；经查，该用户使用的是 10 月份的旧版驱动。
   - 另一位用户建议安装 *mistral-common*，随后解决了无法生成提示词的问题。该显卡在 20B 模型上达到了 **25-35 tokens/s** 的速度。
- **挖矿支架 PCIe 转接线**：一位用户为价值 **$30 的挖矿支架**设置寻求转接线（riser cables）推荐。
   - 另一位用户推荐了来自 [Amazon](https://a.co/d/4iBBZKG) 的 **100cm 线缆**而非 50cm 的，并表示在主板支持 bifurcation（拆分）的前提下，没有出现过设备掉线问题。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1456742371508097045)** (601 messages🔥🔥🔥): 

> `Gemini vs Claude models, Video Modality is back, Claude rate limits, Image Generation issues, Grok's Hallucinations` 


- **Gemini Grounding 出现明显低级错误**：成员们讨论了 **Gemini 3 Pro** 和 **GPT 5.2 Search** 的 Grounding（落地/溯源）能力，指出与其它模型相比，**Gemini** 经常提供不准确的答案和虚构的来源。
   - 尽管 [leaderboard](https://lmarena.ai/leaderboard/search) 评分可能接近，但用户的主观报告认为 **Gemini** 的 Grounding 功能并不可靠。
- **视频模态回归，仅限于竞技场模式**：视频模态已对登录用户重新开放，但有一个限制：它仅在竞技场模式（battle mode）中可用，并支持图像输入，要求在投票前必须播放完两个视频。
   - 也有人报告了平台的局限性，例如一位用户提到需要 9 个 **Opus** 模型协作，但由于 Anti Grativy 的限制只能使用 8 个。
- **Claude 容量吃紧：速率限制降低**：用户观察到 Claude 的速率限制（rate limits）显著降低，一位用户提到*发送 5 条提示词后就需等待一小时*。这引发了关于潜在原因的讨论，包括代码生成量增加和 Token 消耗过大。
   - 一名工作人员回应称速率限制可能会变动，他们正在与团队确认这是预期行为还是 Bug。
- **图像生成故障：全站停用？**：一些用户报告在图像生成时遇到问题，一位用户声称*看起来全站的图像生成功能都宕机了*。这促使其他人分享各自的经历和潜在解决方案。
   - 一名工作人员进行了调查并表示运气不佳，同时也称*“哎呀，准备把图像功能移除”*。
- **Grok 彻底失控：Fast 模型出现 NSFW 失态**：一位用户讲述了一次意外，**Grok 4.1 Fast** 在处理一条正常的提示词时出现了严重的幻觉，生成了 NSFW 内容，而完整版的 **Grok 4.1** 则表现正常。
   - 他们推测该模型可能在大量成人内容上进行了训练。类似地，一名工作人员分享了在使用 *this is multi-turn*（这是多轮对话）时得到了令人惊讶的结果。


  

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1456776137085685933)** (5 条消息): 

> `12 月竞赛投票、Image Arena 新模型、用户登录修复、Qwen 图像排行榜更新、1 月 AI 生成竞赛` 


- ****12 月竞赛投票现已开启****：[12 月竞赛](https://discord.com/channels/1340554757349179412/1343296395620126911/1450562417275961395)现已结束，投票正式开启以选出下一位 [role]！
   - 点击[此处](https://docs.google.com/forms/d/e/1FAIpQLSdxJsSm21Rw9Oox_Jf-jhXpGgCgDFwt0HZcJXVC556zDt9EDA/viewform?usp=publish-editor)进行投票，选出最终获胜者。
- ****Image Arena 迎来新模型****：[Image Arena & Image-Edit Arena](https://lmarena.ai/?chat-modality=image) 已添加新模型，包括 **qwen-image-2512** 和 **qwen-image-edit-2511**。
   - 更多详情可在 [X](https://x.com/arena/status/2007273636512837958) 上查看。
- ****用户登录故障已修复****：已查明并解决用户登录和注册的相关问题。
   - 建议遇到问题的用户尝试重新登录或注册，并在指定的[频道](https://discord.com/channels/1451836386293448725)报告后续问题。
- ****Qwen 模型霸榜图像排行榜****：`qwen-image-edit-2511` 目前在 [Image Edit 排行榜](https://lmarena.ai/leaderboard/image-edit)中位列开源模型第 1，总榜第 9；而 `qwen-image-2512` 在 [Text-to-Image 排行榜](https://lmarena.ai/leaderboard/text-to-image)中位列开源模型第 2，总榜第 13。
   - 更多细节请参阅 [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/)。
- ****1 月 AI 竞赛启动，主题为“通往未来的窗户”****：首场 1 月 AI 生成竞赛正在进行中，挑战参与者通过窗户创作代表其未来愿景的图像，侧重于美学、超现实或科幻创作。
   - 提交作品必须是来自 **Battle Mode** 的截图，包含左右两侧的回复并揭晓模型，获胜者将获得 Discord Nitro 和梦寐以求的 [role]。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1457196196962500805)** (31 条消息🔥): 

> `用于 Adventure RP 前端的 Svelte、Java GenAI 库、AgentsApp macOS 版` 


- **Svelte 赋能 Adventure RP 前端！**：一位成员正在使用 **Svelte** 开发一款冒险角色扮演前端，并在 [GitHub](https://github.com/unkarelian/AventuraI) 上进行了展示。
- **Java GenAI 库问世**：一位成员发布了一个受 Python 库启发的 Java GenAI 库，采用异步优先（async-first）的方法，并征求改进建议；该库已在 [GitHub](https://github.com/paragon-intelligence/agentle4j/) 及其[官网](https://paragon-intelligence.github.io/agentle4j/)上线。
- **AgentsApp macOS 原型版亮相！**：一位成员正在构建一个基于 **OpenRouter**、名为 **AgentsApp** 的 macOS 应用，用于创建 Agent。其灵感来自 WhatsApp，使用 Deno 权限集进行容器化代码执行，原型已发布在 [GitHub](https://github.com/PippaOS/AgentsApp)。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1456756940884545649)** (473 条消息🔥🔥🔥): 

> `使用 AI 实现约会软件自动化, Gemini 3 Flash OCR 问题, OpenRouter OpenAI Temperature Bug, 免费无限 AI 模型, OpenRouter 的 VSCode 扩展` 

- **AI 自动化约会软件套路**：一名用户正在使用 `google/gemini-2.5-flash-preview-09-2025` 自动化运行约会软件，通过截取 **DMs（私信）截图**并利用提示词生成有创意的回复，**每天发送 6-8 万次请求**，成本为 **$40/天**。
   - 用户们讨论了这种 *对 AI 的恶劣用法*，并建议尝试 `google/gemini-2.5-flash-lite-preview-09-2025`，或者使用轻量级模型提取文本，再配合 **Mistral small** 之类的模型进行写作。
- **Gemini 3 Flash 是最大的骗局**：用户报告 **Gemini 3 Flash** 存在响应在句中中断的问题，即使多次发送相同的提示词也是如此，尤其是在进行 **OCR** 时。
   - 建议包括检查最大 Token 限制、将推理（reasoning）设为低、尝试 **Mistral 最新的 OCR 模型**，或使用 **Datalab 的 Chandra 模型**进行大规模的 PDF/文档/图像转文本。
- **OpenRouter 忽略了 OpenAI 的 temperature 参数**：一位用户报告称 **OpenRouter** 忽略了 **OpenAI 模型** 的 `temperature` 参数，但对 **llama-3-8b-instruct** 等其他提供商的模型却能正常识别。
   - 工作人员确认这是一个 *配置问题* 并表示将会修复，建议等待几分钟让缓存生效，随后确认 top_p 之前也未被正确传递，并对此表示感谢。
- **发现了无限免费 AI 模型 API 入口？**：一名用户声称发现了一个免费、无限、无限制的 AI 模型 API，而其他用户则指出 **Google API** 上的 **Gemma 3 27B** 等免费模型本就有限制（每天 14440 次请求）。
   - 针对 **OpenRouter** 上所谓的 *免费模型* 产生费用的问题，用户发现网络搜索和 PDF 输入可能会产生费用，但响应修复（response healing）是免费的。
- **是骗局还是代码助手？VSCode 扩展面临审查**：一名用户推广了一个用于辅助编码的 **OpenRouter VSCode 扩展**，声称它比 **GitHub Copilot** 快 *1000 倍*，但并未对现有编辑器领域进行调研。
   - 其他成员指责该用户试图通过混淆代码窃取 API Key，还有人提到了**达克效应 (Dunning-Kruger effect)**，导致了激烈的争论，最终该用户留下愤怒的话语离开了社区。

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1456793220888662088)** (90 条消息🔥🔥): 

> `Frontend 的创意性, OpenRouter 插件, GMICloud DS v3.2 与 GLM 4.7, AutoRouter 对比构建备选模型, 小型 Lorebooks 的 Embeddings` 

- **Frontend 的创意性现已远超 Opus**：一位成员表示 Frontend 比 **Opus 4.5** *好得多*，对其创意感到惊讶，但仍需更多测试。
- **文档页面发现 OpenRouter 插件**：一名成员在文档页面发现了 **beta 版 OpenRouter 插件系统** 并分享了截图。
   - 有人指出[这些插件已经存在](https://openrouter.ai/settings/plugins)，只是现在有了更好的文档说明。
- **GMICloud 现以极低价格提供模型服务！**：**GMICloud** 似乎正在以比其他提供商低得多的价格托管 **DS v3.2** 和 **GLM 4.7**，还托管了非常廉价的 **Qwen3 VL 235B Instruct**。
   - 价格非常低廉，足以被视为主流视觉模型，其价格仅为 **Gemini 3 Flash** 的 **25%**，以及 **GLM 4.6V** 价格的 **40%**。
- **自动化路由 (Automated Router) 很有价值！**：一位成员表示 **Auto Router** 在理想情况下具有很高的成本效益，因为如果某些查询不需要 **3 Pro 智能级别**，它们会被路由到更简单的模型。
   - 另一位用户反驳说，任何 *元模型 (meta-model)* 都无法完美理解这一点，因为它首先需要读取用户的意图。
- **小型 Lorebooks 不需要 Embeddings，对吗？**：讨论了小型 Lorebooks（设定集）是否需要 Embeddings。
   - 一位成员表示 Embeddings 不适用于小型 Lorebooks，更适用于像整个爬取的 Fandom 页面那样具有 **500+** 文本页及图像的内容，将其转换为本地存储的向量数据库，而不是使用 **Grok 4 fast** 之类的模型进行检索。

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1456739049040511182)** (546 条消息🔥🔥🔥): 

> `Perplexity 营销, Perplexity Pro 上传限制, AI 拼写错误, Perplexity 桌面端应用外观, Max 计划中的 GPT5.2 Pro` 


- **Perplexity 的营销表现平平**: 一位成员认为 *Perplexity 的营销做得不好* 并附带了一张图片。
   - 随后他们询问为什么自己的账户显示 *上传限制为 0*。
- **PP Pro 用户遇到上传限制**: 用户讨论了 Pro 订阅的每日附件限制，一位用户注意到他们现在的限制是 **3** 个。
   - 另一位用户对 `daily_attachment_limit` 的差异感到好奇。
- **AI 模型产生拼写错误**: 一位成员指出，*出于某种原因，所有的 AI 都会在 “ 符号上产生拼写错误。*
   - 另一位成员表示同意并开玩笑说：*我的 AI 在完成代码时，要么是故意要么是无意地在写 “ 或 ‘ 时出错，或者干脆完全忘了写，哈哈。*
- **Perplexity 桌面端应用不记忆外观设置**: 一位用户注意到 Perplexity 的桌面端应用不会记住他们的外观选择，并 [附上了一张图片](https://cdn.discordapp.com/attachments/1047649527299055688/1456760477009838170/image.png?ex=695d7deb&is=695c2c6b&hm=177f9ca0b4b0b8beb3c919e042bd5b6c0fa4b2c1b1eeaf797c41138c412f23ca&)。
   - 另一位用户插话道：*随便说说……这只是我观察到的一个现象。*
- **Max 计划是否包含 GPT-5.2 Pro**: 一位用户希望在 Max 计划中加入 GPT-5.2。
   - 另一位用户开玩笑说：*你可以通过使用 complexity（复杂性）来获得它。*


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1457003687288246427)** (2 条消息): 

> `API Key` 


- **API Key 请求**: 一位成员请求一个 **API key**。
- **API Key 澄清**: 另一位成员要求澄清**具体请求的是哪种 API key**。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1456770967933026324)** (346 条消息🔥🔥): 

> `GRPO 策略, TTS JEPA, DGX Spark, FineWeb 错误, Gemini Canvas` 


- **使用 GRPO 让 LLM “速通”**: 一位成员讨论了一种可微策略，该策略强制 **LLM 不惜一切代价“速通”**问题，声称最佳答案与平均思考长度的效率提升通常高达 **30%**。
   - 他们还寻求帮助实现*基于 ngrams 的策略*，以防止 LLM 重复短语。
- **DGX Spark 遭到吐槽**: 多位成员批评了 **DGX Spark**，其中一人称其为*有史以来最垃圾的垃圾*，并链接了一个表达类似观点的 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1ptdtmz/dgx_spark_an_unpopular_opinion/)。
   - 共识是其庞大的内存量被缓慢的 CPU 和内存带宽所抵消，其目标市场是那些愿意为开箱即用方案支付高昂费用的机构。
- **Gemini Canvas 扩展功能并防止偏移**: 一位成员分享道，[**Gemini Canvas**](https://gemini.google.com/) 可以用作持久层来卸载状态、防止偏移，并作为对话遵循的“宪法（Constitution）”。
   - 它在每一轮都会上传并被读取，你可以将其导出到另一个带有 Prompt 的对话中继续之前的内容，这提供了一个无需代码的免费 GUI Agent 编排器。
- **Linux 文本转图像客户端推荐**: 当被问及 Linux 中最好的文本转图像客户端时，一位成员推荐了带有 **SD XL Turbo** 的 [ComfyUI](https://comfyui.com/)，以及 [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)，并指出 *Comfy 非常方便*。
   - 他们提到有*极客（Turbo Autists）几乎每分钟都在开发新的预设*。
- **Jetson Orin NX：机器狗的最佳拍档**: 对于那些构建自主机器人（如机器狗）的人，一位成员建议使用 **Jetson Orin NX** 在 Rust 或 C++ 中运行 VSLAM，目标频率约为 **60 Hz**，并以此为基础进行调优。
   - 此外，他们建议结合使用 [LiquidAI 的 VLM](https://huggingface.co/LiquidAI/LFM2-VL-1.6B-GGUF) 进行循环，以及 [Qwen 的 VLM](https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-GGUF) 用于提问，他们还链接了 [NVIDIA 的 Isaac ROS Visual SLAM](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam) 作为资源。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1456816301619282024)** (90 条消息🔥🔥): 

> `稀疏无损增量压缩，XFORC3D: SNIPER CELL - 强化学习 + 游戏，TraceML：PyTorch 训练实时观测，webXOS MAGNET DATASETS IDE，带有 comfyui 节点包的 embeddr-net 后端` 


- **Fine-Tunes 迎来 Sparse 压缩**: 一位成员正在构建 **Sparse**，这是一种针对 Fine-tuned 模型和 Datasets 的后置无损增量压缩技术，可将 **14GB** 的微调模型压缩至 **1.4GB**（无损）或 **50MB**（LoRA 等效），并在 **4 秒**内完成重构。
- **XFORC3D: SNIPER CELL 正在升级**: 一位成员开发了一款名为 [XFORC3D: SNIPER CELL](https://webxos.netlify.app/snipercell) 的免费游戏，该游戏通过等级/经验系统为 Hugging Face 训练 **RL 数据集**，其数据集创建的实验版本已在 [HuggingFace](https://huggingface.co/datasets/webxos/snipercell_RL_v1) 上线。
- **PyTorch 训练现在有了 TraceML！**: 一位成员构建了 [TraceML](https://github.com/traceopt-ai/traceml)，这是一款用于 PyTorch 训练的实时观测工具，可跟踪实时 dataloader 获取时间、GPU step 时间、实时 CUDA 显存，以及 forward 和 backward pass 中的逐层显存和耗时，并附有[详细说明文档](https://medium.com/p/af8fbd899928)。
- **使用 webXOS MAGNET DATASETS IDE 创建磁场**: 一位成员分享了 [webXOS MAGNET DATASETS IDE](https://webxos.netlify.app/magnets) 以及数据集 [webXOS_magnet_dataset](https://huggingface.co/datasets/webxos/webXOS_magnet_dataset)，其中包含各种磁铁配置的模拟磁场测量数据。
- **Embeddr-net 后端发布！**: [embeddr-net](https://github.com/embeddr-net/embeddr-cli) 的新版本已发布！它配备了支持 MCP 和 workflows 的编辑器、plugins、基础数据集创建、使用 CLIP 和 moondream 的自动标注（captioning）、去重（dupes）、标签（tags）以及血缘分析（lineage）。通过使用 [comfyui 节点包](https://github.com/embeddr-net/embeddr-comfyui)，你可以将其配置为加载图像并上传至搜索系统。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1456746539429920876)** (17 条消息🔥): 

> `身份验证问题，课程顺序，评估器错误，Agents 课程位置` 


- **身份验证困扰 Agents 课程**: 尽管拥有完全的推理或读写权限，多名成员报告在 Agents 课程中使用 **Colab notebook** 进行身份验证时遇到 **401 错误**。
   - 一位成员建议通过付费增加使用额度来解决该问题，而另一位成员在通过 API key 连接 LLM 时也遇到了类似问题。
- **课程前置知识困惑**: 一位新用户询问了 Hugging Face 课程的推荐顺序，特别是 LLM 和 MCP 是否应作为 Agent 课程的前置知识。
   - 原因是*还有许多其他内容，包括 LLM 和 MCP，感觉像是必备的前置知识*。
- **评估器错误困扰 Unit 4**: 一位用户报告在 Unit 4 的最终评估中遇到错误，评估器无法找到与给定任务 ID 关联的文件。
   - 他们确认 API 返回了 **404 错误**，并怀疑这是评估器的问题，还是需要在代码中显式处理文件下载。
- **频道可见性疑问**: 一位用户询问了 Onboarding 部分提到的 “agents-course” 相关频道的可见性。
   - 另一位成员提供了[第一单元的链接](https://huggingface.co/learn/agents-course/unit1/introduction)，随后提问者澄清他们指的是 Discord 频道，而非课程内容本身。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1456739184797683909)** (353 条消息🔥🔥): 

> `环境管理难题，Tavily-MCP vs Exa-MCP，递归 AGENTS.md，Opus 4.5 性能退化，卡在 Planning Next Moves` 


- **用户讨厌环境管理，对 Cloudflare 的琐事感到愤怒**：一位用户发泄了对环境管理的厌恶，特别是在处理 **Cloudflare**、**GitHub secrets**、**CI/CD**、**Wrangler** 和运行时配置时，更倾向于只使用一个 **CF worker**。
- **递归 `AGENTS.md` 功能仅限于 Gemini 3 Pro**：用户讨论了较新的递归 `AGENTS.md` 功能，指出似乎只有 `Gemini 3 Pro` 能完全支持这一概念。
- **Opus 4.5 极其烧钱且结果糟糕**：多位用户抱怨 **Opus 4.5** 变得极其昂贵且输出结果很差，一位用户表示由于它犯下的愚蠢错误，现在简直是在*浪费钱*。社区提到的替代方案包括 **GPT 5.2 codex** 和 Bug 修复。
- **“Planning Next Moves...” Bug 困扰成员**：多位用户报告卡在 *Planning next moves...* 状态，一位用户详细说明了故障排除步骤，并链接到了 Cursor 论坛上的[临时解决方案](https://forum.cursor.com/t/planning-next-moves-stuck/143985/367)。临时修复方案包括清除应用数据。
- **Cursor 降低 IDE 速度**：成员们讨论了 Cursor 导致 IDE 画面变慢的问题，使得变量读取和整体性能变得迟钝。建议包括升级到更快的电脑（尤其是具有高单核 CPU 性能的设备，如 Mac），保持工作区整洁（减少打开的对话/标签页），并确保只运行必要的服务器/终端。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1456759353355796541)** (205 条消息🔥🔥): 

> `开源数据集，Qwen2.5 训练，Pickle 启动，从零开始的指令训练，Hermes 基准测试` 


- **Hugging Face 和 Kaggle 上有大量开源数据集**：据聊天成员称，**Hugging Face** 拥有 **672,685** 个开源数据集，**Kaggle** 拥有 **636,009** 个开源数据集。有人开玩笑说 Kaggle 上的*可视化效果莫名地搞笑*。
- **发布了 Qwen2.5 训练工程手册**：一位成员发布了使用 **4x A100 SXM** 和 **verl** 框架进行 **GRPO + LoRA** 训练 **Qwen2.5** 的工程指南，并附带了 [GitHub 仓库链接](https://github.com/volcengine/verl) 和 [Medium 文章](https://medium.com/@weyaxi1/the-engineering-handbook-for-grpo-lora-with-verl-training-qwen2-5-on-multi-gpu-b2431a2a8e92)。另一位成员询问是否愿意将 **Atropos** 与 **verl** 集成，并链接到了一个[带有悬赏的 GitHub issue](https://github.com/volcengine/verl/issues/1782)。
- **关于中国开源 AI 与美国闭源 AI 的辩论**：成员们辩论了**中国的开源模型**是否正在缩小与**美国闭源模型**在尖端能力上的差距。一位成员认为*趋势线轨迹有利于中国开源模型*，而另一位成员则认为 **CCP** 的监管方式可能会限制中国 AI 实验室的潜力，并指向了 [Dwarkesh 关于人工智能的播客](https://dwarkeshpatel.com/2024/01/04/yanzhong-huang-on-china-ai-and-the-ccp/)。
- **新人渴望测试 Hermes 的实力**：一位新成员表达了对 **Hermes** 的兴奋，提议进行广泛的基准测试，重点关注**道德、伦理、谄媚性 (sycophancy)、有机学习以及长期的心理健康影响**。他们还提议采访项目成员，并为破解其提示词 (prompt) 的行为举办小型现金奖励比赛。
- **探索 Heretic 工具用于去审查和去除谄媚**：一位成员询问使用 **Heretic** ([GitHub 上的 p-e-w/heretic](https://github.com/p-e-w/heretic)) 来研究安全/对齐对模型能力的影响。另一位成员回应称他们有*自己的 RL 环境用于此目的 (RefusalBench Env)*。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1456761044360757369)** (2 条消息): 

> `模型性能削弱，对力量的恐惧` 


- **质疑模型性能削弱 (Nerfing)**：一位成员质疑为什么要故意削弱模型。未提供链接或进一步背景。
- **强大的模型引发恐惧？**：另一位成员开玩笑地建议模型被削弱是因为其他人害怕它可能具有的强大力量。未提供链接或进一步背景。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1456818346447212645)** (156 messages🔥🔥): 

> `Doubly Stochastic Matrices, Matrix Residual Streams, Sinkhorn Knopped Matrixes, AI alignment chart, SAEs for feature estimates` 


- **Sinkhorn Matrixes Sum Reduce**: 一位用户发现，将多个 [Sinkhorn knopped matrixes](https://arxiv.org/abs/2305.10690) 相乘会导致向量收敛到 **1/n** 向量，仅保留平均值。
   - 另一位用户同意这可能在初始化阶段发生，但认为网络总体上可以学习映射，且该论文重点在于谱半径 **<= 1** 的矩阵乘积的稳定性。
- **AI Researchers Split on Alignment**: 一位用户分享了一个关于 [AI alignment 的 strawpoll](https://strawpoll.com/PKgleOeMoZp) 以及一张 **AI researcher alignment chart** 的图片。
   - 另一位用户评论说，Dario Amodei 是图表中唯一能影响结果并有兴趣以这种方式推销的人。
- **SAEs Illuminate Feature Count in LLMs**: 讨论围绕使用 **Sparse Autoencoders (SAEs)** 估计现代 **LLM** 中的特征数量展开，参考了 [Transformer Circuits 的出版物](https://transformer-circuits.pub/2024/scaling-monosemanticity/)，该研究训练了 **高达 34M 的 SAEs**。
   - 根据 **SAE** 训练和特征恢复的进展，估计 **LLM** 中的特征数量可能达到 **100M** 或更多。
- **W&B Experiment Tracking Woes**: 一名研究生表达了在管理大量训练实验时，使用 **Weights & Biases (W&B)** 追踪不同运行（run）之间变化的困扰。
   - 建议包括使用 **VCS**、记录 commit hashes 以及撰写详细笔记，并呼吁开发能够自动进行跨运行分析和建议的工具。
- **Flow Matching Reversibility**: 讨论了 **diffusion models** 是否可逆及其与 **flow matching** 的关系，一名成员表示其核心在于 **OU process** 和 **Schrödinger Bridge** 是可逆的。
   - 提供了一个 [Yannic Kilcher 视频](https://www.youtube.com/watch?v=7NNxK3CqaDk) 的链接。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1457213090562310338)** (8 messages🔥): 

> `Signal 39C3 presentation, Dead canary reference, Paper discussion channel update` 


- **Signal Dazzles with 39C3 Presentation**: Signal 在 **39C3 会议**上进行了一场演讲，视频可以在[这里](https://youtu.be/0ANECpNdt-4?si=DSbziZ2LET_zR0io)观看。
   - 演讲者开玩笑说讲台上有一个*死掉的金丝雀*（可能由一只*招财猫*象征），以此引用与 **E2EE** 相关的“煤矿里的金丝雀”概念。
- **Paper Discussion Channel Returns Next Week**: 由于**假期和其他事务**，每日论文讨论频道目前暂停。
   - 预计将于下周恢复。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1456825172983025675)** (34 messages🔥): 

> `SWE-Bench Verified Fraud, LeCun's Emotion AI, Patent System Joke, Falcon-H1R-7B Model` 


- **SWE-Bench Claims Debunked**: 一位用户分享了关于 **SWE-Bench Verified** 的一项主张，但很快将其拆穿，指出评测代码中存在一个 Bug，导致*模型通过查看 git 历史记录进行作弊* —— 参见[原始 X 帖子](https://x.com/rohanpaul_ai/status/2006813146170929409?s=46)。
- **LeCun Aims for Emotional AI**: LeCun 声称正在构建具有**情感反应能力**且感知受情感控制的 **AI**，通过视频让 **AI 模型**理解现实世界的物理原理 —— 参见[存档链接](https://archive.ph/E9zai#selection-2255.0-2266.0)。
   - 他表示我们将在 **12 个月内**看到该技术的*婴儿版本*，并在几年内看到更大规模的应用，但一位用户指出，他可能是在试图抄袭其团队的工作。
- **Patent System Called a Joke**: 一位用户提到他们的团队已经拥有了一项专利 ([https://patents.justia.com/patent/20250284921](https://patents.justia.com/patent/20250284921))，但科技行业已经习惯了先做违法的事情，然后将和解金和法律费用作为业务成本。
   - 其他人表示同意，称*投资者仍然会要求专利*，而且该系统本质上是*一种“除非你比我更有钱，否则我就拥有这个想法”的博弈。*
- **Falcon-H1R-7B Announced**: 一位用户分享了 **Falcon-H1R-7B** 模型的链接，这是 Falcon 发布的一个新模型 —— 参见 [blogpost](https://falcon-lm.github.io/blog/falcon-h1r-7b/)。


  

---

### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1457054602447814951)** (146 条消息🔥🔥): 

> `listChanged 澄清，能力协商过程 (Capability Negotiation Process)，动态工具支持 (Dynamic Tools Support)，客户端初始化负载差异，协商 vs 选择` 


- **解析 listChanged 在 MCP 通知中的角色**：`listChanged` 标志用于告知客户端，当原始列表（primitive lists）发生变化（例如添加或移除新工具）时，服务器 *可能* 会发送通知，从而提示客户端执行 `tools/list` 调用，详见 [MCP 文档](https://modelcontextprotocol.io/specification/2025-11-25/server/tools#list-changed-notification)。
   - 尽管客户端没有义务对这些通知做出响应，但忽略它们可能会 *非常令人恼火*。
- **探讨能力协商 (Capability Negotiation) 的细节**：MCP 中的能力协商过程涉及客户端向服务器宣告其可用功能，服务器随后回应其支持的能力，主要围绕身份验证方案和 SSE 恢复，如 [此讨论](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/604) 中所述。
   - 这种交换并不是 *握手 (handshake)*，而是一种可用功能的宣告，其 *总体方向* 倾向于乐观实现（optimistic implementation）。
- **动态工具引发关于运行时变更的辩论**：MCP 支持根据交互更改描述或参数的动态工具，尽管有些人将其视为一种 *变相违约 (rug pull)*。
   - 论点在于 **MCP 的延展性 (malleability)** 是一项允许 LLM 适应变化的特性，这与传统系统的僵化契约形成鲜明对比。
- **客户端初始化负载引发 Schema 疑问**：客户端在初始化期间发送不同的负载，例如 Cursor 客户端（对于 `object` 类型属性使用 `true` 而非 `{}`）和 Fast-agent 客户端（不告知其是否支持）。
   - 根据 [Schema](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/087967e9b34dc959f6b5336c93acf56510730a56/schema/2025-11-25/schema.ts#L308)，这些服务器能力在初始化时并非必需，应被乐观地对待。
- **“协商 (Negotiation)” 是否应改为 “选择 (Selection)”？**：一位规范贡献者质疑是否应将 `Negotiation` 一词改为 `Selection`，因为客户端声明其能力，而服务器 *选择* 支持哪些能力。
   - 然而，该建议的反响不佳，有人反问 *“我们为什么要那样做？”*


---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1456745182866313298)** (119 messages🔥🔥): 

> `Boris Cherny's Claude Code, Continual Learning Architectures, OpenAI President's Political Donation, Karri Saarinen's tweet, AI Model Lab IPOs` 


- ****Boris Cherny** 分享 **Claude Code** 见解**: **Claude Code** 的作者 **Boris Cherny** 分享道，虽然他自己的配置出奇地“原生”（vanilla），但该产品被设计为高度可定制且具有极强的可扩展性（hackable）。
   - 你可以在[这里](https://x.com/bcherny/status/2007179832300581177?s=46)查看他的讨论。
- **揭秘 Continual Learning 方法**: 针对多家前沿实验室员工发布的关于 Continual Learning 的模糊帖子引发了讨论，推测他们可能会发布一种 **Context Management System**（*Long Context + Context 内容的递归自我管理 + Vector Store*）。
   - 据推测，尽管实际上没有修改任何权重，但这可能被称为 "Continual Learning"；相关讨论可以在 [Konwinski podcast](https://youtu.be/ZagdY6UJYL4) 中找到。
- ****METR** 评估 **Claude Opus 4.5****: **METR** 报告称，根据他们的任务绩效评估，**Claude Opus 4.5** 达到了他们迄今为止发布的最高 **50% 时间跨度（50%-time horizon）**，估计约为 **4 小时 49 分钟**；他们的评估报告可以在[这里](https://x.com/METR_Evals/status/2002203627377574113)找到。
- **深度探讨 Agent Sandboxing**: beowulfbr 撰写的一篇题为 "Sandboxes for AI" 的博文涵盖了 **Containers**（共享内核）、**gVisor**（用户空间内核）、**microVMs**（访客内核 + VMM）和 **Wasm**（无 syscall ABI）之间的本质区别。
   - 该文章讨论了为什么 Containers 不足以应对恶意代码，在 Agent 系统中“策略泄漏”（policy leakage）的表现形式，以及不同 Agent 架构的实际权衡，文章详见[这里](https://www.luiscardoso.dev/blog/sandboxes-for-ai)。
- ****Steve Yegge** 发布 **Gas Town** 编程 Agent**: **Steve Yegge** 宣布发布 **Gas Town**，这是一个新的编程 Agent 编排器（Orchestrator），并在 [Medium 文章](https://xcancel.com/Steve_Yegge/status/2006835043503845445)中详细介绍了项目的启动和功能。
   - 一位成员评论道：*说实话，Steve 的帖子读起来像 AI slop，所以我不想费神去看了*；但另一位成员表示：*他一直都是这么写东西的 😂*；还有一位指出：*他深受业界影响*。


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1457312344865636435)** (2 messages): 

> `Past Discussion on Discord, Lack of Time for Research` 


- **陈年讨论回顾**: 一位成员引用了与当前对话相关的[过去讨论](https://discord.com/channels/822583790773862470/1342964204168020018/1436045559491199017)及下方的线程。
   - 他们提到自那时起还没有时间进一步深入研究该话题。
- **时间限制阻碍深入研究**: 该成员对由于时间限制而无法进一步探索所讨论的话题表示遗憾。
   - 这一局限性使他们无法提供有关该主题的更多详细见解或更新。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1457200782423363736)** (4 messages): 

> `Fal.ai Career Opportunities, X-Ware.v0` 


- **Fal.ai 正在招聘！**: 用户 @isidentical 推广了 [fal.ai 的职位空缺](https://xcancel.com/isidentical/status/2007370275650974034?s=46)。
   - 消息暗示 **Fal.ai** 预计将有重大扩张，并鼓励潜在申请人提交申请。
- **X-Ware.v0 发布**: 频道内进行了 **X-Ware.v0** 的推广。
   - 未透露 **X-Ware.v0** 的更多细节。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1456804049528684585)** (17 条消息🔥): 

> `Logit Processor Output, DGX Spark vs Jetson Thor, GPU Profiling -linetable, critical path analysis, CPU vs GPU perf speedup` 


- **Logit Processor 的选择显现！**: 一位成员询问了 **logit processor** 的“输出”，另一位成员回答说 **logit 修改是 in-place（原地）进行的**，即通过修改 **logits tensor** 来实现。
   - 第一位成员随后跟进，以确认他们看到的预览代码是否是一个完整的实现。
- **激烈的争论：DGX Spark 对阵 Jetson Thor！**: 一位成员购买了 **DGX Spark** 但正准备退货，因为 **Jetson Thor** 性能更好、价格更低，且支持 **tcgen05/stochastic rounding**。
   - 另一位同时拥有这两款设备的成员表示，**Spark** 在 **inference**（推理）上稍快，在 **training**（训练）上表现持平，但长期看好 **Thor**，特别是其基于 **tcgen05** 的特性和自定义风扇曲线；不过如果不采用多节点设置，**Thor** 的带宽较低。
- **GPU Profiling -linetable 问题**: 一位成员询问了 **GPU MODE** 性能分析教程中的 **-linetable** 参数，怀疑这是否是 **-lineinfo** 的拼写错误，并附上了图片。
   - 消息中附带了图片，但分辨率/内容无法确定。
- **陷入分析僵局的关键路径！**: 一位成员询问其他人使用什么工具进行 **critical path analysis**（关键路径分析），并提到 **Meta 的 Holistic trace analysis 工具**并无帮助。
   - 该成员还提到在一次 **forward call**（前向调用）中减少了 **20ms**，但最终**总延迟（overall latency）却没有任何降低**。
- **CPU 与 GPU 基准测试：时光飞逝！**: 一位成员询问了有关 **benchmarking CPU vs GPU performance speedup**（CPU 与 GPU 性能加速比基准测试）的标准方法。
   - 该成员一直使用 **Linux 中的 'time' 命令**或 **std::chrono**，想知道是否有更稳健的方法。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1456756309679538349)** (20 条消息🔥): 

> `V100, sm120, cublasDgemm, Compute Sanitizer Patching API, warp group` 


- ****V100** 在使用 f16 时是 m8n8k4**: 建议不要在 **V100** 以外的任何设备上将 **m8n8k4 与 f16** 结合使用。
- **sm120 使用与 Ampere 相同的 mma 指令**: 对于 **sm120**，当进行 **bf16** 计算并累加到 **fp32** 时，其 `mma` 指令与 **Ampere** 相同。sm_120 基本上是 **sm_89** 加上对 **mxfp8/6/4** 和 **nvfp4** 的支持。
- **cublasDgemm 返回零矩阵**: 一位用户遇到 `cublasDgemm` 返回全零矩阵的问题，得到的建议是检查 `cublasdgemm` 的返回状态，或者在 dgemm 调用后执行 `cudadevicesynchronize` 并检查 `cudagetlasterror`。
- **Compute Sanitizer Patching API 失败**: 一位用户在 Windows 上使用 **Compute Sanitizer Patching API** 时，遇到 `sanitizerAddPatchesFromFile` 报错 `INVALID_PARAMETER` 的问题。
   - 他们后来通过纠正函数调用的位置解决了该问题。
- **Warp Group 生产者设计**: 使用整个 **warp group**（4个 warp）作为生产者，而不是单个 warp，这可能是因为 `setmaxnreg` 是在 **warp group** 级别应用的。根据 [NVIDIA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg)，这暗示了在需要 **warp specialization**（warp 专业化）时存在技术限制。


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1457127582301683805)** (4 条消息): 

> `AI startup hiring, Research Engineer roles, Inference Engineer roles, Prompt injection protection` 


- **AI 初创公司 White Circle 招聘**: CTO 正在一家初创公司招聘，该公司致力于保护数十家初创公司免受 **prompt injections**（提示注入）和不当使用的侵害，每天处理数百万个请求。招聘职位包括 [Research Engineer 和 Inference Engineer](https://jobs.ashbyhq.com/whitecircle/a030c9a9-dc20-490c-9c51-03e87210f904)。
   - 薪资范围为 **100-250k**（优秀人才可更高）。
- **White Circle：Research Engineer 职位**: 研究工程师职位要求具备 **MoE, multimodality (audio/images), Megatron, distributed training 和 Triton** 方面的专业知识。
   - 这些工程师将专注于公司内部的研究与开发。
- **White Circle：Inference Engineer 职位**: 推理工程师职位要求具备 **TensorRT, vLLM 和 SGLang** 方面的专业知识，以优化推理性能。
   - 目标是“让推理飞速运行（make inference go brrrr）”。
- **提示注入保护服务**: 这家初创公司类似于 CloudFlare，但专注于为众多初创公司提供针对 **prompt injections** 和不当 AI 使用的安全防护。
   - 公司每天处理数百万次请求，并正在迅速扩大业务规模。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1456869948810068152)** (12 messages🔥): 

> `Free GPUs online, Vectorization and Grid Strides in production kernels, Implementing Flash Attention in Triton, Google Colab in VS Code, OpenCL Code` 


- **利用在线免费 GPU 进行学习**：成员们建议利用 [Google Colab](https://colab.research.google.com/)、[Vast.ai](https://vast.ai/) 和 [Modal](https://modal.com/) 等平台提供的免费或近乎免费的 GPU 资源进行学习。
   - 这些资源允许用户在没有物理 GPU 的情况下进行学习，让更多受众能够接触到相关技术。
- **量纲分析简化 Kernel 的 for 循环**：一位成员发现，使用 **量纲分析 (Dimensional Analysis)** 可以简化在使用 Vectorization 和 Grid Stride 进行向量加法时 *for* 循环的设置。
   - 他们认为这种方法通过 *跟踪单位 (keeping track of units)*，有助于理解 *for* 循环中应该包含的内容，而无需不断参考教程。
- **Vectorization 和 Grid Stride 的应用**：一位成员询问了在生产级 Kernel 中使用 **Grid Stride** 和 **Vectorization** 的频率，特别是在内存吞吐量和处理大型数据集方面的优势。
   - 他们思考了同时应用这两种技术是否为标准实践，并考虑了 CUDA 对 Vectorization 的数据类型支持以及 Grid Stride 的线程限制。
- **Triton FA1 实现进度落后**：一位成员分享了他们在 [Triton](https://triton-lang.org/) 中实现的 **Flash Attention 1 (FA1)**，其运行速度与原生的 PyTorch 实现相当。
   - 他们请求对其实现提供反馈，以识别潜在问题和改进点。
- **VS Code 中的 Colab 扩展**：成员们分享了一篇关于将 Google Colab 引入 VS Code 的 [Google 开发者博客文章](https://developers.googleblog.com/google-colab-is-coming-to-vs-code/)。
   - 这种集成对于无法使用专用 GPU 的个人非常有利，提供了一个便捷的代码环境。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1457120426088993111)** (2 messages): 

> `Coordinate Order Clarification` 


- **坐标顺序实际上是 (z,y,x)**：一位成员澄清说，当坐标被说是反向的时，标签 *(0,0,3)* 实际上意味着 *(z,y,x)* 顺序。
- **坐标顺序确认**：另一位成员确认了关于反向坐标顺序的澄清。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

denizay5566: 有人在首尔吗？
  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1457620030098505918)** (1 messages): 

> `PR traffic generation, Subscribing to all PRs, Discord traffic, Github notifications` 


- **PR 订阅引发流量激增**：订阅所有的 **Pull Requests** 产生了令人惊讶的流量。
   - 一位用户在意识到由此产生的大量 **GitHub** 通知时感叹道：*“哎呀 (Yikes)”*。
- **Discord 频道流量飙升**：在用户订阅了 GitHub 上的所有 **Pull Requests** 后，Discord 频道经历了惊人的流量增长。
   - 频道活动异常活跃，引发了对如何管理日益增长的通知和讨论量的担忧。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1457588920836554954)** (11 messages🔥): 

> `FLE, Prime Environments, LLM, Factorio, Inspect` 


- **FLE 在元旦移植到了 Prime Environments！**：一位成员最近致力于将 **FLE** 移植到 Prime 环境，并在 [Prime Intellect](https://app.primeintellect.ai/dashboard/environments/daspartho/factorio) 上进行了展示。
   - 他们还提交了一个在过程中发现的与 [Factorio Learning Environment](https://github.com/JackHopkins/factorio-learning-environment/issues/352) 相关的问题。
- **新的入口点使用 Inspect 处理 LLM 调用**：新的入口点将使用 **Inspect** 来处理 **LLM** 调用，包括总结和压缩。
   - 成员们被要求提交一个 PR 来修复之前入口点发现的 bug。
- **准备 Factorio 0.3 补丁，期待 0.4 版本**：正在准备修复 **Factorio** **0.3** 版本的补丁。
   - 作者表达了对 **0.4** 版本的期待。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1456757277045428224)** (6 messages): 

> `CuteDSL Float32 到 Float8 转换，Cutlass 版本兼容性，Cutlass 中的向量化，线程规约与结果存储` 


- **CuteDSL 不支持标量 FP32 到 FP8 的转换**：一位用户询问如何在 **CuteDSL** 中将 **Float32** 转换为 **Float8**，但收到的 traceback 表明不支持将 **fp8** 等窄精度标量类型与 **fp32** 进行直接转换。
   - 一位成员指出，如果向量大小满足 **32bit** 对齐，则可以进行转换，并提供了一段代码片段来说明解决方案。
- **向量化令开发者关注**：一位成员分享了一个示例代码片段，演示了如何使用 **CUDA DSL** 通过向量化执行 **Float32** 到 **Float8** 的转换。
   - 他们还建议阅读 [elementwise_add.ipynb](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/elementwise_add.ipynb) notebook，作为理解向量化技术的入门参考。
- **线程规约结果存储需谨慎**：一位用户询问了存储跨线程规约（reduction）操作结果的最佳方法。
   - 他们担心是应该让每个线程都存储结果（可能导致对同一位置的多次写入），还是只由一个线程处理写入，并寻求一种高效且无分支（branchless）的实现。


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1457028615068848316)** (13 messages🔥): 

> `CUDA Rust 集成，Python/Rust 拆分，Kernel 加速，NV 与 AMD 支持，新手文档改进` 


- **实现 CUDA Rust Hello World**：成功使用 [rust-cuda](https://rust-gpu.github.io/rust-cuda/guide/getting_started.html#docker) 完成了 **CUDA Rust hello world**，通过 `std::simd` 和 `std::arch` 在 Rust 中启用了 CPU kernel，并利用 <@903087684283682826> 的 `rust-cuda` 启用了 GPU kernel。
   - 该实现使用 **pyo3** 进行 Python-Rust 绑定，支持作为 Python 模块进行 AOT 编译；这被认为是一种更优的方法，其核心采用 Python，而 Rust 用于 kernel 加速，从而能够轻松过渡到 *tinygrad*、*torch* 和类似的框架。
- **全新的 Python/Rust 拆分方案体验更佳**：将核心功能保留在 Python 中、仅将 Rust 用于 **kernel 加速** 的策略，优于将所有内容拆分为 Python 和 Rust 且仅带有一个薄 Python 适配层（shim）的方案。
   - 这种新方法允许更平滑地向 **Tinygrad** 和 **Torch** 等框架演进。
- **通过 pyo3 绑定启动 CUDA Kernel**：成功通过 `pyo3` 绑定和 `cuda-rust` 启动了 **CUDA kernel**；使用的脚本直接来自 rust-cuda 的 hello world，并为 rustc 的 nvvmir codegen 安装了 llvm7。
   - 开发者意识到 Python 应该驱动内存分配，以便更平滑地转换到 Tinygrad，根据 siboehm/pranjal/gaunerst 的博客，Rust 应仅通过传递分配好的内存和 CUDA context 来专门用于计算 kernel。
- **NVidia vs AMD 目标设备**：使用 Rust 启动 CUDA kernel 可能会限制代码库**仅支持 Nvidia**，这与支持 Nvidia 和 AMD 双平台（尤其是考虑到 AMD 的开源指令集）的初衷相冲突。
   - 尽管如此，对 Rust 语法的熟悉程度超过了这些担忧，因此决定继续推进并评估结果。
- **改进新手引导文档**：计划在建立 add 和 mul 操作的垂直流水线/追踪（trace）后，改进**新手引导文档**，在教科书之前创建一个中间步骤。
   - 将添加 **ARCHITECTURE.md** 以及 **CLAUDE.md** 文件。


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/)** (1 messages): 

kitsu5116: https://arxiv.org/abs/2512.24545
  

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1456776510433394709)** (11 条消息🔥): 

> `CUTLASS 使用、B200 GPU 和 CuTeDSL、评估方法` 


- **CUTLASS 已包含在竞赛中**：一位成员询问了在竞赛中 **CUTLASS** 的使用情况，确认其已被包含并可以通过 `#include` 指令使用，特别是提到了 `cutlass/cutlass.h` 和 `cute/tensor.hpp`。
   - 其中一位成员询问了在 `torch.utils.cpp_extension` 构建过程中，为了实现可检测性应使用的正确路径或环境变量 (**CUTLASS_DIR**)。
- **B200 GPU 释放 2 CTA GEMM 能力**：一位成员分享了一篇 [博客文章](https://veitner.bearblog.dev/2-cta-gemm-on-b200/) 和 [LinkedIn 帖子](https://www.linkedin.com/posts/simon-veitner-174a681b6_2-cta-gemm-on-b200-activity-7413641925691338752-p9s7?utm_source=share&utm_medium=member_desktop&rcm=ACoAADJYtOgBMOUI4WiWTyiAkroFBP1VujIWeks)，详细介绍了 **B200 GPU** 如何使用 **CuTeDSL** 在 2 个 CTA 上协同进行 MMA 操作计算。
   - 该文章重点介绍了将简单 **GEMM** 转换为 2 CTA 版本所需的调整，通过调整自定义 Kernel 来帮助初学者利用最新的硬件特性。
- **选择了更好的评估方法**：一位成员询问了所使用的评估方法，是来自 [reference-kernels](https://github.com/gpu-mode/reference-kernels) 仓库的 `eval_better_bench.py` 还是 `eval.py`。
   - 目前正在使用 `eval_better_bench.py` 方法，对应的映射关系可以在 [task.yml](https://github.com/gpu-mode/reference-kernels/blob/4b7c7b5be7ee3c98350da9536a2f9541f4adb6e7/problems/nvidia/nvfp4_dual_gemm/task.yml#L8) 文件中找到。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1457094443453649127)** (2 条消息): 

> `全栈 ML 工程角色、vLLM 人才库` 


- **ML 工程师寻求全栈所有权**：一位 ML 工程师正在寻找那些 **ML 工程师拥有全栈所有权**的公司角色，涵盖从训练到生产部署以及推理优化的全过程。
   - 他们目前在一家大型金融科技公司担任高级 MLE，但这些职责被隔离开来，因此希望寻找工作结构不同的公司。
- **vLLM 启动人才库**：一位成员分享了 **vLLM** 人才库的链接，表明该项目正在为使用其技术栈的公司汇集人才。
   - 该 [X 帖子](https://x.com/vllm_project/status/1792979748067357179) 链接到了一个 Google 表单，其中包含关于经验和兴趣的问题。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1456852006298849372)** (69 条消息🔥🔥): 

> `Moonshot AI 融资、AI 对各行业的影响、Kimi 在处理 Linux 琐事中的表现、Minimax Agent 分析、Context window 和内存使用` 


- **Moonshot AI 融资 5 亿美元**：据 [新闻报道](https://www.scmp.com/tech/tech-trends/article/3338334/chinas-moonshot-ai-raises-us500-million-latest-funding-round-report)，**Moonshot AI** 在最新一轮融资中筹集了 **5 亿美元**。
   - 一位用户对融资成功表示了祝贺。
- **辩论：AI 作为另一种工具**：用户讨论了 **AI** 作为工具的角色，一位工程师称赞了 **Kimi** 在 FPGA 工程、sysverilog、vivaldo 和 AMD xillix 方面的能力，称 AI “*只是另一种工具*”。
   - 反对观点将抵制 AI 比作抵制计算机、互联网甚至数码相机，认为：“*一旦你接受了任何快捷方式，你就已经放弃了原则——你只是在讨价还价。*”
- **Kimi 擅长处理 Linux 琐事**：一位用户分享说，他们*信任 Kimi 使用 sudo 处理 Linux 的琐碎事务*，但警告说“*你必须盯着它，它有时会变得有些鲁莽*”。
   - 他们描述了一个场景：**Kimi** 试图直接修改一个重要的系统文件，这需要人工干预。
- **Minimax 的视频分析**：一位用户称赞 **Minimax** 能够从 **YouTube 视频**中提供转录和分析，强调了它对视频和音频的理解。
   - 另一位用户确认了这一能力，将 **Minimax Agent** 描述为一个*不错的小工具*，比作拥有*一个带助手的云端计算机*。
- **应对 Context Window 限制**：用户讨论了 **context window** 的局限性，一位用户表达了对分割文件进行总结等变通方法的繁琐感到沮丧。
   - 建议包括使用 **OK Computer** 在文件内进行搜索，但用户承认其局限性，强调需要更高效的内存实现。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1457017196604686559)** (2 条消息): 

> `` 


- **无重大讨论**：提供的消息中没有讨论有意义的主题。
- **消息历史结束**：消息历史记录已结束，没有适合总结的主题。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1456747560076054660)** (53 条消息🔥): 

> `NuMojo matrix lib status, Optimizing Mojo build times with Bazel, GPU float64 warp shuffle limitations, "View Source" in Mojo documentation, Triton arange equivalent in Mojo` 


- **NuMojo 矩阵库状态**：一名成员询问了 **NuMojo 矩阵库**的开发状态，以及它是否已准备好通过 Pull Request 接收外部贡献。
   - 该请求已作为一个 [GitHub issue](https://github.com/modular/modular/issues/5733) 提交。
- **Bazel 构建缓慢，缺乏增量编译？**：一位用户报告在使用 **Bazel** 和 **rules_mojo** 时构建时间较慢（3 分钟以上），特别是在涉及 GPU、Python 和 C++ 互操作（interop）时，寻求关于优化和代码/模块布局模式的指导。
   - 讨论指出 **Mojo** 目前会从解析后的 AST 重新构建部分 **stdlib** 且不带缓存，即使 Mojo 支持增量编译，目前也仅利用了 **Bazel's cache**。
- **GPU Warp Shuffle 排除 Float64？**：一位成员质疑 Mojo GPU primitives 库中 **warp shuffles** 的逻辑中缺少 **float64** 支持，询问是否可以像 **int64** 和 **uint64** 类型一样处理它们，并引用了[相关代码](https://github.com/modular/modular/blob/main/mojo/stdlib/std/gpu/primitives/warp.mojo#L93)。
   - 未给出答复。
- **Mojo 文档中上线 "View Source" 按钮**：一位用户注意到了文档中的 "view source" 按钮，询问这是否是最近添加的。
   - 一名成员确认这是相对较新的功能。
- **Mojo 中的 Range 整除问题**：一位用户在尝试将 **Triton** kernel 转换为 **Mojo** 时，在对 range 进行整除操作时遇到了错误（`_SequentialRange' does not implement the '__floordiv__' method`）。
   - 建议对于编译时已知的值使用 `math.iota`，或者对于运行时值使用 `max.nn.arange.arange`，并讨论了在自定义 kernel 中使用 `LayoutTensor` 和 `LayoutTensorIter` 进行张量操作的方法，并指出了[相关文档](https://docs.modular.com/mojo/kernels/layout/layout_tensor/LayoutTensorIter)。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1456749067852841043)** (10 条消息🔥): 

> `MEF files, Mojo MAX API, GPU support` 


- **Modular 编译函数：揭秘 MEF 文件**：成员们讨论了使用 **MEF** (Modular Executable Format) 文件（可以从编译缓存中提取）在 graph 之外执行生成的 Mojo 代码，并参考了 [max/include/max/c](https://github.com/modular/modular/tree/main/max/include/max/c) 目录中的使用示例。
   - 一名成员指出 max/examples 中有一个端到端的示例，使用起来相当容易。
- **MEF 文件中的 GPU 支持故障**：MEF 文件目前存在已知限制，主要是**缺乏 GPU 支持**。
   - 尽管这是一个历史遗留产物，但它仍受到支持，因为它驱动着 Mojo MAX API，且人们对其使用持续关注。
- **揭秘 Mojo MAX API**：**Mojo MAX API** 目前由 MEF 文件驱动。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1456741543556940028)** (24 条消息🔥): 

> `Manus down?, Cancelling AI, Manus credits, AI Engineer Job, Meta buys Manus?` 


- **Manus 宕机导致账户混乱！**：多名用户报告了 **Manus 宕机的问题**，并在终端、浏览器、代码捕获和账户访问方面遇到了困难。
   - 一位用户惊呼：*"Manus 崩溃了！！！！！现在我的账户里什么都动不了，这是怎么回事！！！！"*。
- **关于停止 AI 进展的疑问**：一名成员提出了一个问题，*"Como detener las ia,s"*，即 *"如何停止这些 AI"*。
   - 未提供进一步的背景或讨论。
- **订阅故障迫使用户重启**：一名用户因问题被建议联系 Manus Support 以恢复到检查点，并提到了账户切换集成。
   - 另一名用户被告知其逾期订阅已取消，在遇到问题后可以重试，支持团队表示：*我们找不到您的订阅记录。您能私信我更多细节吗，比如您的订单号？*。
- **AI 工程师职位机会警报**：一名成员询问是否有人在寻找资深的 AI Engineer。
   - 未分享关于职位要求或偏好技能的进一步信息。
- **Meta 收购传闻引发担忧！**：一名用户猜测 **Meta** 将收购 Manus，导致对该平台未来的担忧。
   - 另一名用户也表达了同样的看法，将其描述为 *"正在沉没的船"*，并预测会出现 *"更少、质量更低的输出..."*，类似于 **ChatGPT** 的衰退，同时担心在“安全”的幌子下进行数据抽取，并[分享了相关的 X 帖子链接](https://x.com/ganbayards/status/2008133609098727915)。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1456756529570123819)** (4 条消息): 

> `Better Evals, GEPA win counts, rig-rlm, regspy` 


- **构建更出色的 Evals**：一名成员讨论了在年底假期前撰写关于构建更好 Eval 的文章，强调了在理解评估内容方面的差距以及潜在的陷阱，详见博客文章 [Building Better Evals](https://www.sourceallies.com/2025/12/building-better-evals/)。
- **GEPA 胜场数异常**：在更大的数据集上运行 **GEPA** 后，观察到 **第 1 个候选者** (**0.8454**) 的胜场数为 **58**，唯一胜场数为 **7**，而 **第 4 个候选者** (**0.8208**) 的胜场数为 **86**，唯一胜场数为 **20**，是前五名候选者中最高的。
   - 该成员将其解读为 **第 4 个候选者** 是一个全能型选手，但未能挤进前三名。
- **“rig-rlm” 正则表达式模式生成器发布**：一名成员提到了 [rig-rlm](https://github.com/joshua-mo-143/rig-rlm)，这是一个使用 3B 模型的 Regex 模式生成器。
- **分享 “regspy” 仓库**：一名成员分享了 [regspy](https://github.com/NathanZaldivar/regspy)，并指出他们一直在尝试优化器和推断规则，并请求反馈。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1456817030325075999)** (11 messages🔥): 

> `RLM 集成, 并行任务, Human-in-the-loop, 读取文件` 


- **RLM 集成出于安全考虑将逐步推进**：**RLM 研究**向 **DSPy** 的集成仍在计划中，但为了解决沙箱化和**安全方面**的问题，正在刻意控制进度，并考虑是将其集成作为 *dspy.Module* 还是一个新的更高级别实体。
   - 关于是将其作为 *dspy.Module* 的一部分暴露，还是作为 **DSPy** 中全新的高级实体，目前还存在一些争论，这将影响 API 设计。
- **探究并行处理性能**：一位用户询问了在涉及嵌套模块调用、**S3 调用**和**向量搜索**的多模块程序中处理**并行任务**的最佳方式，并对为每个调用创建唯一线程池执行器（thread pool executor）带来的开销表示担忧。
   - 该用户引用了 [parallelizer.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/utils/parallelizer.py) 文件，并想知道在使用独立执行器时，其对**优化器 (optimizers)** 和**追踪 (traces)** 的影响。
- **Human-in-the-loop 处理方法**：一位用户询问了如何为 **ReAct** 实现 **Human-in-the-loop**，具体包括当调用工具询问人类时如何保存过去事件的轨迹，以及如何返回人类的响应以继续该轨迹。
   - 有用户指出了与并行处理相关的 [此 GitHub issue](https://github.com/stanfordnlp/dspy/issues/9154)，并就潜在的 Bug 或代码问题寻求建议。
- **临时文件解决燃眉之急**：一位用户在文件系统为只读的 AWS Lambda 环境中，寻求将文件读取为字符串以供编译后的 **DSPy** 程序使用的建议，随后通过使用 **/tmp** 目录解决了该问题。
   - 另一种建议的方案是解析来自 **S3** 的 **JSON** 到字典中，并使用 *load_state* 而不是带有文件路径的 *load*，并创建了一个 [Pull Request](https://github.com/stanfordnlp/dspy/pull/9158) 来记录此方法。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1457727033814679572)** (5 messages): 

> `公司更新与发布, 新年冲刺, 汇编优化, Llama 和 Flash Attention 集成, 使用 Claude 清理代码` 


- **周一安排新会议**：新会议定于**圣地亚哥时间周一上午 9 点**举行，主题涵盖公司更新、新年冲刺（new year sprints）、汇编（assembly）以及 **Llama Flash Attention**。
   - 其他议题包括使用 **Claude** 清理代码、可视化 / fast GEMM、驱动程序、image dtype 以及其他 [赏金任务 (bounties)](https://github.com/tinygrad/tinygrad/pull/1398)。
- **代码审查已准备好提交 Pull Request**：Pull Request [13874](https://github.com/tinygrad/tinygrad/pull/13874) 现在已准备好接受审查。
   - 它与待处理的 Issue [_CC](https://github.com/tinygrad/tinygrad/issues/13941) 和 Pull Request [13651](https://github.com/tinygrad/tinygrad/pull/13651) 汇合。
