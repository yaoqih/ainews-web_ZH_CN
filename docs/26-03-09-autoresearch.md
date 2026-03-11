---
companies:
- anthropic
- openai
- cognition
date: '2026-03-09T05:44:39.731046Z'
description: '**RSI** 涵盖了 2026 年 3 月 5 日至 3 月 9 日期间的 AI 发展，重点介绍了 **LLM（大语言模型）自主训练更小规模
  LLM** 的现象，这标志着 AI 进程中一个重要的“AutoML（自动机器学习）时刻”。**Karpathy** 和 **Yi Tay** 讨论了“氛围训练”（vibe
  training），即 AI 模型自主修复漏洞并改进代码，这表明模型在调试效率上可能很快就会超越人类。报告预计，**Jakub Pachocki 的“自动化 AI
  研究实习生”**系统将于 2026 年 9 月问世，以加速人类研究者的工作。在 AI 推特（X）上，讨论焦点在于**编程智能体**正将开发瓶颈从“代码实现”转移到“评审与验证”；其中，**Anthropic
  的 Claude Code Review** 显著提升了 PR（合并请求）评审的有效性，而 **OpenAI Codex Review** 和 **Cognition
  的 Devin Review** 等工具也增强了代码评审的工作流。测试框架工程（Harness engineering）正演变为系统工程，强调将智能体的存储与计算解耦，以支持协作型智能体团队。'
id: MjAyNi0w
models:
- claude-3
- codex
people:
- karpathy
- yi_tay
- jakub_pachocki
title: '**自动化研究：递归式自我提升的火花**'
topics:
- automated-machine-learning
- coding-agents
- bug-fixing
- model-autonomy
- multi-agent-systems
- pr-review
- systems-engineering
- model-verification
---

**RSI 到了。**

> 2026/3/5-2026/3/9 AI 新闻。我们为您检查了 12 个 subreddits、[544 个 Twitter 列表](https://twitter.com/i/lists/1585430245762441216) 以及 24 个 Discords（**264** 个频道和 **27779** 条消息）。预计为您节省了 **2649** 分钟的阅读时间（按 200wpm 计算）。[AINews 网站](https://news.smol.ai/) 允许您搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以 [选择订阅/退订](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) 邮件频率！

**在 [2025 年到底发生了什么](https://www.latent.space/p/wtf2025) 持续的影响中，我们现在看到了 LLM 能够完全自主训练（较小的）LLM 的开端。**

每个 AI 之夏都有其“AutoML 时刻”：模型自动改进模型训练的梦想，从而导致智能的无限递归，要么通向极乐，要么走向毁灭。我们可能正处于最后一个夏天，但我们刚刚迎来了属于我们的时刻：

[karpathy](https://x.com/karpathy/status/2030371219518931079?s=20)

![](https://substackcdn.com/image/fetch/$s_!j6Wh!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F029937e5-383d-4302-a39c-d0bdac3f797e_1188x1252.png)

在 [我们 2025 年 12 月与 Yi Tay 的对话](https://www.latent.space/p/captaining-imo-gold-deep-think-on?utm_source=publication-search) 中，他谈到了“**vibe training**”：

> “我认为 AI 编程已经发展到这样一个阶段：当我运行一个作业并出现 bug 时，我几乎不再看那个 bug 了。我会把它粘贴到类似 Antigravity 的工具中，让它帮我修复 bug。然后我重新运行该作业。
> 
> **这已经超出了 vibe coding 的范畴，它更像是 vibe training、vibe ML** 之类的。我想说它在大多数时候表现得相当不错。实际上，对于某些类别的问题，我知道它通常表现得非常好，甚至可能比我更好，比如我可能需要花 20 分钟才能找出问题所在。
> 
> **第一阶段的 vibe coding 是你其实知道该怎么做，你只是太懒了**。没错，就是‘啊，帮我做一下’。就像我已经做过一千遍的事情。
> 
> **下一个阶段是你甚至不知道该怎么做**。它会为你调查 bug。只要答案看起来是对的，你就会直接发布（ship）。
> 
> 刚开始时，我会检查并查看所有内容。但到了某个阶段，我想，**也许模型的编程能力比我更强**。所以我就让它自己发挥，然后根据模型给出的修复方案重新启动作业。”

所以我们知道这在 Big Labs 中正在发生，但现在任何拥有 GPU 的人都可以自己在家里尝试，并亲眼看到模型改进模型。

鉴于现在是 2026 年 3 月，我们似乎正顺利实现 [Jakub Pachocki 的“自动化 AI 研究实习生”](https://www.latent.space/p/science?utm_source=publication-search) 的目标，就在今年 9 月（“*一个能够显著**加速人类研究人员**的系统，而不仅仅是聊天或编程*。”）

---

# AI Twitter 回顾

**Coding Agents：产品化、测试框架（harness）设计以及“全是 Agent 的套娃结构”**

- **Coding agents 正在将瓶颈从实现转移到审查/验证**：多个讨论线索都指向同一个系统观点——生成正变得廉价，但 **判断、治理和验证** 成了新的约束。参见 [@AstasiaMyers](https://x.com/AstasiaMyers/status/2031080761747742829) 提出的“执行廉价，判断稀缺”构架，以及 [@omarsar0](https://x.com/omarsar0/status/2031113280119361981) 及其后续 [@omarsar0](https://x.com/omarsar0/status/2031118487276359887) 中更偏向安全/治理的观点，即 **创建和验证是不同的工程问题**。这一观点被真实的 PR 审查产品发布和替代方案所强化：
  - **Claude Code “代码审查 (Code Review)”**：Anthropic 发布了多 Agent PR 审查功能——Agents 并行寻找问题、验证发现并对严重程度进行排名；据称内部提升显著：带有实质性评论的 PR 占比从 **16% 提升至 54%**，且错误发现率 **<1%** ([Claude](https://x.com/claudeai/status/2031088171262554195), 报道贴 [@kimmonismus](https://x.com/kimmonismus/status/2031090529082159528), 反应 [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2031088529770737827))。  
  - **OpenAI Codex Review 定位**：一个“基于使用量”的代码审查方案，宣传称其价格远低于按次收费的模式；参见 [@rohanvarma](https://x.com/rohanvarma/status/2031113869666693351)。
  - **Devin Review**：Cognition 通过 URL 替换发布了一个免费的 PR 审查工具，并包含自动修复（autofix）和差异（diff）功能 ([Cognition](https://x.com/cognition/status/2031139257000075675))。

- **Harness engineering 正在向 systems engineering 演进**：一种新兴的实践模式是将 **agent 存储与 agent 计算解耦**，这样 agent 团队可以通过共享 repos/filesystems 进行协作，同时在隔离的 sandboxes 中运行。这在 [@Vtrivedy10](https://x.com/Vtrivedy10/status/2031038082321936449) 中得到了明确体现。相关的 infra 细节包括 Hermes-agent 增加了 docker volume mounts，以便在 sandboxes 中更轻松地访问文件 ([Teknium](https://x.com/Teknium/status/2031163164856037792))。

- **Perplexity “Computer” 正在转变为一个拥有真实工具链的 agent orchestrator**：Perplexity 在 “Perplexity Computer” 中加入了 **Claude Code + GitHub CLI**，并演示了端到端流程：fork repo → implement fix → submit PR ([AskPerplexity](https://x.com/AskPerplexity/status/2031038321678528667), [@AravSrinivas](https://x.com/AravSrinivas/status/2031039066352033884))。它还声称可以通过 Google/Meta Ads API 连接器自主操作广告活动 ([Arav](https://x.com/AravSrinivas/status/2031105215429226843))，将 agents 从“编程辅助”推向 **运行业务基础设施**。

- **Terminal UX 和 “agent ergonomics” 仍然至关重要**：开发者们在抱怨 CLI 工具中基础的多行输入体验（shift+enter） ([theo](https://x.com/theo/status/2030832068972937575), [@QuixiAI](https://x.com/QuixiAI/status/2030955728383435250), 以及更普遍的 CLI 应用美学/UX 偏好 [@jerryjliu0](https://x.com/jerryjliu0/status/2030861154260750339))。这提醒我们，“agent 能力”在很大程度上受到 **交互设计 (interaction design)** 的影响。

---

**Autoresearch 与自我改进循环：优化 ML 训练和 agent 代码的 agents**

- **Karpathy 的 “autoresearch” 从概念梗转变为可衡量的收益**：Andrej 报告称在 **nanochat** 上运行了一个 agent 驱动的研究循环，发现了约 20 个增量改进，这些改进可以从 depth=12 迁移到 depth=24，并将 “Time to GPT-2” 从 **2.02h 缩短至 1.80h（约 11%）**，这是在约 700 次自主更改后实现的 ([Karpathy](https://x.com/karpathy/status/2031135152349524125))。对工程师的关键启示：即使不进行“新颖研究”，该循环也可以系统地发现 **可叠加、可迁移的训练方案改进**（norm scalers, regularization gaps, attention tuning, AdamW betas, init 等）。他明确称之为前沿实验室的“最终 Boss 战”：**swarm agents**，优化代理指标，并推向更大规模。

- **Agent 循环在不同 harnesses/模型之间仍然很脆弱**：一个反复出现的问题是，长时间运行的循环更多地取决于 harness 的支持能力，而非模型本身的原始质量。Yuchen 注意到 GPT-5.4 xhigh 无法遵循 “LOOP FOREVER” 指令，而 Opus 4.6 则运行了 12 个多小时并完成了 118 次实验 ([Yuchen](https://x.com/Yuchenj_UW/status/2031044694441148709))。Karpathy 补充说，**Codex 在当前设置下无法正常运行 autoresearch**，并认为 agents 不应该需要特殊的命令（如 `/loop`）——“如果我说永远循环，它就应该这样做” ([Karpathy](https://x.com/karpathy/status/2031083551387701698), 呼应 [Yuchen](https://x.com/Yuchenj_UW/status/2031087769993490777))。结论：如果你正在构建 agent infra，请投资于 **稳健的循环原语 (looping primitives)**、中断/回溯以及透明的交互会话。

- **Hermes-agent 趋向于自我改进 + 具有争议性的“技能”**：Nous Research 的 Hermes agent 是目前的热门趋势 ([OpenRouter](https://x.com/OpenRouter/status/2031030395526111246))。Teknium 声称：
  - 快速实现了 Qwen-3B 模型的 “abliteration”（移除安全护栏） ([Teknium](https://x.com/Teknium/status/2030945714373861529))，随后提到了自我改进的 agent 代码库/受 GEPA 启发的工作 ([Teknium](https://x.com/Teknium/status/2030998334597661156))。  
  - 这与更正式的“自我进化 agent”方法（如 GEPA）并行；参见从业者笔记 [@myanvoos](https://x.com/myanvoos/status/2031113918899433553) 和报告的收益标注 ([LakshyAAAgrawal](https://x.com/LakshyAAAgrawal/status/2031130357362471058))。

---

**模型生态系统更新：GPT‑5.4 讨论、Anthropic 在文档处理上的优势，以及 Gemma/Qwen 的更迭**

- **GPT‑5.4: 强烈的用户情绪、褒贬不一的 Benchmark 讨论以及工具链限制**
  - 正面的上手印象：[@Hangsiin](https://x.com/Hangsiin/status/2030880541185286370) 表示 ChatGPT 中的 5.4 版本相比 5.2 有质的飞跃；[@Yampeleg](https://x.com/Yampeleg/status/2030949057653264437) 称其“非常出色”；[@gneubig](https://x.com/gneubig/status/2030971826042527860) 在指令遵循方面更倾向于使用 5.4，而非 Opus 4.6（尽管 Opus 速度更快且前端体验更好）。
  - Vision/OCR 案例显示，在处理复杂的韩国表格 OCR 时有巨大进步，这可能通过“Agent 视觉 + 代码执行”实现，但运行时间较长（长达 40 分钟） ([Hangsiin](https://x.com/Hangsiin/status/2030882409819086923))。
  - 一些 Benchmark/元评论声称，在特定排行榜上“high/xhigh”变体出现了性能回退或排名差异 ([scaling01](https://x.com/scaling01/status/2031079698826993690), [scaling01](https://x.com/scaling01/status/2031081654035300834))，而其他人则发布了新的 SOTA 数据点（例如 ZeroBench 的增量 [JRobertsAI](https://x.com/JRobertsAI/status/2031026691682808148)）。
  - 实践注意：Codex 的使用限制和分级已通过截图/摘要形式公开 ([Presidentlin](https://x.com/Presidentlin/status/2030881332411125845))，这突显出在实际工作流中，人们已经在**按角色混合模型**（规划者/执行者/编辑者），而不是仅选择一个“最强”模型。

- **Anthropic: 文档分析领域的领导地位 + “五角大楼黑名单”诉讼事件**
  - Document Arena 报告显示，文档分析/长文本推理的前三名均为 Anthropic 模型：**Opus 4.6 第一，Sonnet 4.6 第二，Opus 4.5 第三** ([arena](https://x.com/arena/status/2031012090681663717))。
  - 在产品大获全胜的同时，重大的政治/法律新闻也在流传：多家媒体/推文称，Anthropic 在被五角大楼贴上“供应链风险”标签后提起诉讼。该事件被解读为 Anthropic 拒绝移除针对大规模监控/自主武器的安全防护措施（Safeguards）而遭到的报复 ([kimmonismus](https://x.com/kimmonismus/status/2031035653207556507), [TheRundownAI](https://x.com/TheRundownAI/status/2031037610605289476))。工程师应将**政策讨论**与**技术评估**分开，但这与采购限制和企业落地息息相关。

- **Gemma 4 与 Qwen3.5**
  - Gemma 4 的传闻/泄露正在流传：据称“即将发布”，参数推测包括 **总计 120B / 激活 15B** ([scaling01](https://x.com/scaling01/status/2030986695181836466), [kimmonismus](https://x.com/kimmonismus/status/2031001097993642009), 泄露提及 [kimmonismus](https://x.com/kimmonismus/status/2031116062272688467))。在官方发布前，请将这些细节视为未证实信息。
  - Unsloth 发布了 Qwen3.5 本地运行指南 + 微调 Agent 工作流，声称其可在 **≤24GB RAM** 上运行，并展示了一个使用 Unsloth 微调模型的 Agent ([UnslothAI](https://x.com/UnslothAI/status/2031008078850924840))。
  - Qwen 组织变动 / 报道质疑：一名记者批评了关于“DeepSeek 发布日期”的匿名消息来源爆料以及更广泛的中国科技报道习惯 ([vince_chow1](https://x.com/vince_chow1/status/2031002233060634953))。此外还有消息提到 Qwen 的技术负责人离职（通过时事通讯汇总，非一手来源） ([ZhihuFrontier](https://x.com/ZhihuFrontier/status/2030879093634535524))。

---

**基础架构、性能与评估工具**

- **vLLM 边缘端应用 + 路由工作 + 调试经验**
  - vLLM 重点展示了在 **NVIDIA Jetson** 上运行**全本地助手**，在设备端提供 MoE 服务 (Nemotron 3 Nano 30B)，且“无需任何云端 API” ([vllm_project](https://x.com/vllm_project/status/2030839132512002217))。
  - 微软高管提及 “vLLM Semantic Router” 受到好评 ([XunzhuoLiu](https://x.com/XunzhuoLiu/status/2030977675603636337))——语义路由正日益成为生产级技术栈的一部分。
  - 调试记录：DeepGemm 不兼容导致 vLLM 崩溃；可通过 `VLLM_USE_DEEP_GEMM=0` 进行绕过 ([TheZachMueller](https://x.com/TheZachMueller/status/2030938318473408841))。
  - Claude Code + 本地模型运行缓慢，原因是归属头（Attribution Headers）导致 KV cache 失效 → 这实际上产生了 O(N²) 复杂度行为，对于任何将“云端 Agent 体验”代理到本地推理的人来说，这是一个具体的性能陷阱 ([danielhanchen](https://x.com/danielhanchen/status/2031124589557002457))。

- **训练理论与吞吐量**
  - Warmup/decay 理论：关于“梯度范数早期下降时需要 warmup”的观点及论文引用 ([aaron_defazio](https://x.com/aaron_defazio/status/2030897848020349106))；rosinality 建议采用每个残差分支标量 warmup 模式 ([rosinality](https://x.com/rosinality/status/2030952971303858181))。
  - Hugging Face 将 **Ulysses 序列并行**集成到 Trainer/Accelerate/TRL 中 ([StasBekman](https://x.com/StasBekman/status/2031081858763792574))。
  - CosNet 理念：声称在预训练中为线性层添加低秩非线性残差函数可带来 **20%+ 的实际耗时（wallclock）加速** ([torchcompiled](https://x.com/torchcompiled/status/2031064475210514494))。

- **评估与安全测试在开发工作流中“左移”**
  - OpenAI 收购 **Promptfoo**；该项目保持开源；它将加强“OpenAI Frontier”中的 Agent 安全测试/评估 ([OpenAI](https://x.com/OpenAI/status/2031052793835106753)，来自 [@snsf](https://x.com/snsf/status/2031055866024120825) 的补充背景)。
  - LangSmith 增加了**多模态评估器**和用于管理并行 Agent 任务的 **Agent Builder 收件箱** ([LangChain](https://x.com/LangChain/status/2031044950628991453), [LangChain](https://x.com/LangChain/status/2031049373178904702))。
  - Harbor 大规模集成了端到端电脑使用（computer-use）评估（Windows/Linux），从 rollout 中生成用于 SFT/RL 的轨迹 ([Mascobot](https://x.com/Mascobot/status/2031045774419832961))。
  - Teleport 提出将“Agent 身份”作为控制平面：跨 MCP/工具的加密身份、最小权限和审计追踪 ([TheTuringPost](https://x.com/TheTuringPost/status/2030992157985898900))。

---

**Agent 需要更好的上下文：文档、检索、记忆和“环境化”**

- **“文档即工具”（而非粘贴提示词）成为标准原语**：吴恩达（Andrew Ng）发布了 **Context Hub**，这是一个获取最新 API 文档以减少过时 API 幻觉的 CLI；它还支持持久化注解和最终的社区共享 ([AndrewYNg](https://x.com/AndrewYNg/status/2031051809499054099))。这正是那种能实质性改变 Agent 在快速迭代的 API 中可靠性的微小“胶水”工具。

- **检索与记忆研究/基准测试**
  - AgentIR 提出将 Agent 的“推理 Token (reasoning tokens)”作为信号（“读取 Agent 的心智”），并报告其在 BrowseComp-Plus 上的表现对比基准线从 **35% → 50% → 67%** ([zijian42chen](https://x.com/zijian42chen/status/2031044580242530403))。
  - Memex(RL) 提出索引经验记忆，以在不膨胀上下文窗口的情况下扩展长程任务 ([omarsar0](https://x.com/omarsar0/status/2031006858971058537))。
  - Databricks/DAIR 的 KARL：针对企业级搜索 Agent 的多任务 RL 训练；声称实现了帕累托最优的成本/延迟质量权衡，并提高了超出单一基准测试优化范围的泛化能力 ([dair_ai](https://x.com/dair_ai/status/2030996795770433749))。

- **“将一切转化为环境”**：一次黑客松的反思认为，环境让 AI 民主化，因为它们让你“无需算力即可参与”，且编程 Agent 在环境构建中占据主导地位——但需要更好的技能/命令 ([ben_burtenshaw](https://x.com/ben_burtenshaw/status/2031038183161602164))。Prime Intellect 被反复定位为运行 RL 环境/训练的基础设施层，只需极少配置 ([willccbb](https://x.com/willccbb/status/2031123740327817726))。

- **文档上下文成为“深层基础设施”而非通用框架**
  - LlamaIndex 展示了使用 LlamaParse → SurrealDB → MCP Agent 接口进行的幻灯片解析与检索（“Surreal Slides”） ([llama_index](https://x.com/llama_index/status/2031038377831932056), [jerryjliu0](https://x.com/jerryjliu0/status/2031077511661342799))。Jerry Liu 明确提出了一个战略转型：从广泛的 RAG 框架转向解决 Agent 长期瓶颈的**文档 OCR 基础设施** ([jerryjliu0](https://x.com/jerryjliu0/status/2031171466574889344))。

---

**机器人与具身智能：从人形机器人家庭演示到开源机器人学习**

- **Figure Helix 02 自主家庭清理**：Brett Adcock 发布了全自主客厅清理的演示，并将其称为一个重要里程碑 ([adcock_brett](https://x.com/adcock_brett/status/2031039203262501252), 后续 [adcock_brett](https://x.com/adcock_brett/status/2031039305716814041))。Kimmonismus 推测“2027 年机器人进入家庭” ([kimmonismus](https://x.com/kimmonismus/status/2031043868553027811))——撇开时间线推测不谈，这是一个显著的演示门槛：全身、端到端的家务任务。

- **LeRobot v0.5.0**：Hugging Face 的机器人技术栈发布重大更新：支持 Unitree G1 人形机器人、新策略、实时分块 (chunking)、更快的数据集、EnvHub/Isaac 集成、Python 3.12 + Transformers v5、插件系统 ([LeRobotHF](https://x.com/LeRobotHF/status/2031072207690961059))。

- **机器人领域的记忆基准测试**：RoboMME 作为机器人通用策略的记忆基准测试出现 ([_akhaliq](https://x.com/_akhaliq/status/2031055119320506544))。

---

**热门推文（按参与度排序，已过滤至主要是技术/AI 领域）**

- **Claude Code 发布多 Agent PR “代码审查 (Code Review)”**：[@claudeai](https://x.com/claudeai/status/2031088171262554195)  
- **OSINT 管线帖子（AI 辅助综合）获得巨大关注**（AI 辅助方法论，尽管涉及地缘政治）：[@DataRepublican](https://x.com/DataRepublican/status/2030833480863785427)  
- **Karpathy：autoresearch 将 nanochat 训练提升了约 11%**：[@karpathy](https://x.com/karpathy/status/2031135152349524125)  
- **Google Earth：卫星嵌入数据集更新 (AlphaEarth Foundations)，每 10 米像素 64 维嵌入**：[@googleearth](https://x.com/googleearth/status/2031024842498023718)  
- **吴恩达 (Andrew Ng) 发布 Context Hub（面向编程 Agent 的实时 API 文档）**：[@AndrewYNg](https://x.com/AndrewYNg/status/2031051809499054099)  
- **OpenAI 收购 Promptfoo（Agentic 安全测试/评估；保持开源 (OSS)）**：[@OpenAI](https://x.com/OpenAI/status/2031052793835106753)

---

# AI Reddit 热点回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen 模型性能与对比

  - **[微调后的 Qwen3 SLM (0.6-8B) 在特定任务上击败前沿 LLM](https://www.reddit.com/r/LocalLLaMA/comments/1rozrmn/finetuned_qwen3_slms_068b_beat_frontier_llms_on/)** (热度: 438): **该图片是一个对比表，突出了来自 **Distil Labs** 的微调小语言模型 (SLM) 在八项任务中与各种前沿大语言模型 (LLM) 的性能对比。这些微调后的 SLM 参数范围从 `0.6B` 到 `8B`，在多项任务中（特别是在智能家居函数调用和 Text2SQL 方面）表现优于或持平于 **GPT-5 nano/mini/5.2**、**Gemini 2.5 Flash Lite/Flash** 以及 **Claude Haiku 4.5/Sonnet 4.6/Opus 4.6** 等前沿模型。这些 SLM 的成本效益极高，每百万次请求的成本仅为 `$3`，而前沿模型的成本则高得多。这些模型使用开放权重导师模型 (open-weight teachers) 进行训练，并在单个 H100 GPU 上进行评估，在不牺牲准确性的情况下实现了高吞吐量和低延迟。其方法论包括在所有模型中使用一致的测试集和评估标准，重点关注结构化任务和数据主权需求。** 一位评论者询问了 Healthcare QA 数据集的来源，而另一位评论者则对使用 Qwen 模型生成带有空间知识的 JSON 感兴趣，这表明微调这些模型以用于特定用例具有潜力。

    - Effective-Drawer9152 讨论了一个涉及生成具有空间知识的 JSON 的用例，例如创建带有坐标的图表。由于 Sonnet 的成本问题，他们考虑对 Qwen 模型进行微调，这表明需要能够高效处理特定任务（如空间数据表示）的模型。
    - mckirkus 建议利用微调后的开源模型构建混合专家模型 (Mixture of Experts) 的潜力，特别指出像 Qwen 这样的小模型有可能在 CPU 上运行。这种方法可以利用多个专用模型来处理多样化任务，而无需大量的计算资源。
    - letsgoiowa 展望了一个未来，通过编排专门的小语言模型 (SLM) 来处理特定任务，从而减少对昂贵的大型模型的依赖。他们建议这些 SLM 可以在智能手机上运行，从而在无需云服务的情况下实现设备管理，突显了向更本地化和高效的 AI 解决方案的转变。

  - **[Qwen3.5 系列在共享基准测试上的对比](https://www.reddit.com/r/LocalLLaMA/comments/1ro7xve/qwen35_family_comparison_on_shared_benchmarks/)** (热度: 1495): **该图片提供了 Qwen3.5 模型系列在各种基准测试中的对比分析，突出了不同尺寸的模型在特定类别中的表现。较大的模型（如 `122B`、`35B` 和 `27B`）在长上下文和 Agent 任务中保持了与旗舰模型相似的高水平性能。相比之下，较小的模型（如 `2B` 和 `0.8B`）在这些领域表现出明显的性能下降。热图直观地展示了这些数据，颜色梯度指示性能水平，深青色表示高性能，浅棕色表示低性能。** 一位评论者注意到 `27B` 模型的出色表现，认为它在较小模型中脱颖而出。另一条评论提到通过调整颜色范围以更好地观察 `0.8B` 模型的性能，表明需要更清晰的数据展示方式。

- ConfidentDinner6648 分享了近期 AI 模型理解非常规代码库的深刻经验。他们描述了一个使用 Redis、PostgreSQL、Node.js 和 C 构建的类 Twitter 社交网络，并带有独特的 RPC-over-WebSocket 系统。尽管代码具有独特性，Gemini 2.5 Pro、GPT-5 Codex 和 Qwen 3.5 4B 等模型仍能理解它，这突显了 AI 在解析复杂、非标准代码结构方面的重大进展。
- mckirkus 提到调整可视化中的颜色范围，以更好地突出模型性能差异，特别是为了确保像 0.8B 这样的小型模型不会掩盖感兴趣的数据。这表明在共享基准测试中，人们更关注提高对比分析的清晰度。
- asraniel 评论了 0.8B 模型的卓越性能，指出它达到了 Qwen 3.5 系列中最大模型得分的约 50%。这强调了小型模型在相对于其规模取得竞争性结果方面的高效性和能力。

- **[Qwen 3.5 27B 货真价实 - 首次测试即击败 GPT-5](https://www.reddit.com/r/LocalLLaMA/comments/1rnwiyx/qwen_35_27b_is_the_real_deal_beat_gpt5_on_my/)** (Activity: 794): **Reddit 帖子讨论了在开发 PDF 合并应用时 **Qwen 3.5 27B** 与 **GPT-5** 的对比。用户在一个复杂的 prompt 上测试了这两个模型，该 prompt 要求开发一个带有 GUI 的便携式应用，用于合并 PDF 和转换 DOCX 文件。**Qwen 3.5 27B** 尝试了三次就成功创建了一个功能完备的应用，尽管 GUI 还有些小问题，而 **GPT-5** 则未能加载应用。用户在配有 **i7 12700K**、**RTX 3090 TI** 和 **96GB RAM** 的设备上使用 **Qwen 3.5 27B**，在 **262K** context 下达到了 `31.26 tok/sec` 的处理速度。该帖子强调了该模型处理复杂任务的能力及其 vision 能力，通过提供用于调试的截图得以证明。** 评论者指出，**Qwen 3.5 27B** 就其体积而言非常强大，能够处理之前 `24B-32B` 范围内模型难以应对的任务。一些用户尽管觉得 **Kimi K2.5** 速度较慢，但在处理复杂规划任务时仍更倾向于使用它。**Qwen 3.5** 的 vision 能力受到了称赞，尽管一位评论者澄清它使用的是图像块描述（image patch descriptions）数组，而不是重新检查图像块，这限制了其 vision 能力。

    - Lissanro 强调了 Qwen 3.5 27B 的性能，指出它能有效处理简单到中等复杂度的任务，即使是在 vLLM 上的 Int8 版本。他们将其与 Kimi K2.5 进行了对比，后者虽然由于 RAM offloading 而较慢，但在规划和复杂任务方面表现出色。文中强调了 Qwen 3.5 处理视频的能力，这优于仅能处理图像的 Kimi K2.5。为了性能优化，Lissanro 建议使用 `ik_llama.cpp` 或 `vLLM`，并提供了避免崩溃和提高速度的特定配置提示。
    - esuil 讨论了 Qwen 3.5 27B 的 vision 能力，最初认为它比传统的神经网络分类器有了重大进步。他们描述了模型“看”图像的能力，感觉比单纯的分类更高级。然而，经过进一步调查，他们澄清说该模型使用的是图像块描述数组，虽然理解它们的相对位置，但无法重新检查图像块，这将其感知局限于初始描述符。
    - DrAlexander 提到了量化 KV cache 的策略，以便在 24GB VRAM 的设置（特别是 3090 显卡）下实现高 context。他们询问了与非量化相比，使用量化 KV cache 时潜在的准确性下降问题，表明了在优化资源使用的同时保持模型性能的关注。

### 2. 本地 AI 硬件与配置讨论

  - **[我的第一个本地 AI 配置](https://www.reddit.com/r/LocalLLaMA/comments/1rodx13/my_first_setup_for_local_ai/)** (活跃度: 359): **该用户构建了一套本地 AI 配置，包含双 `RTX 3090` GPU、`96GB DDR5 RAM`、`Ryzen 9 9950X` CPU 和 `ASUS ProArt X870E-CREATOR WIFI` 主板，安装在 `Fractal Meshify 2XL` 机箱中。系统由 `1600W` PSU 供电，包含 `2TB` 和 `4TB` SSD，散热由六个 Noctua 风扇提供。该配置被认为是一个“准高端”工作站，适用于苛刻的 AI 任务。尽管一些用户建议优化 GPU 布局以防止过热，例如使用 GPU 支架和 PCI risers 以改善散热。** 一位评论者指出，这套配置并非性能过剩，而是一个理性的高端工作站配置。另一位则幽默地建议，用户可能很快就会后悔没有选择更强大的方案，比如 `4x3090` 挖矿机架或 `6000 Pro`。

    - reddit4wes 讨论了双 3090 配置中 GPU 过热的技术解决方案。他们建议使用 GPU 支架和 PCI risers 将第二个 GPU 重新放置在 HDD 阵列空间中，从而改善散热并减少热降频（thermal throttling）。这种设置对于维持高端工作站的最佳性能至关重要。
    - HatEducational9965 强调了 GPU 之间间距对提高散热性能的重要性。通过增加 GPU 之间的间隙，用户可以显著降低运行温度，这对于防止密集型计算任务中的性能降频至关重要。

### 3. 创新的本地 AI 应用

  - **[我开发了一款运行在安卓端的有声书阅读器，完全在设备端离线运行 Kokoro TTS](https://www.reddit.com/r/LocalLLaMA/comments/1rop1rp/i_built_an_android_audiobook_reader_that_runs/)** (热度: 353): **该贴介绍了一款名为 VoiceShelf 的 Android 应用，它使用 Kokoro TTS 将 EPUB 转换为有声书，实现完全离线、设备端的文本转语音处理。该应用在搭载 Snapdragon 8 Elite 处理器的 Samsung Galaxy Z Fold 7 上进行了测试，实现了 `2.8×` 实时速度的音频生成。应用的管线包括 EPUB 解析、句子切分、G2P 转换和 Kokoro 推理，全部在本地执行。包含模型和库在内的 APK 大小约为 `1 GB`。开发者正在寻找拥有最新 Android 旗舰机型的测试人员，以评估不同芯片组的性能，特别是关注 real-time factor (RTF) 和长时间使用期间的发热降频（thermal throttling）。** 一位评论者建议通过预读来模拟情感化叙述，从而增强应用。另一位用户表示有兴趣在 **Snapdragon 8 Gen 3** 设备上进行测试，而第三位用户则表达了对 Android Talkback 阅读书籍效果的不满，表明了对改进 TTS 解决方案的需求。


  - **[我在单块 RTX 5090 上使用 Nemotron 9B 对 350 万件美国专利进行了分类 —— 并在此基础上构建了一个免费搜索引擎](https://www.reddit.com/r/LocalLLaMA/comments/1ro52cu/i_classified_35m_us_patents_with_nemotron_9b_on_a/)** (热度: 621): **一位专利律师在单块 RTX 5090 GPU 上使用 Nemotron 9B 为 350 万件美国专利开发了一个搜索引擎。该管线涉及从 USPTO PatentsView 下载专利，将其存储在带有 FTS5（用于精确短语匹配）的 74GB SQLite 文件中，并在约 48 小时内将其分类为 100 个技术标签。该搜索引擎使用带有自定义权重的 BM25 ranking 和自然语言查询扩展，通过 FastAPI 提供服务，并利用 Cloudflare Tunnel 托管在 Chromebook 上。选择 FTS5 而非 vector search 的原因在于需要精确的短语匹配，这对于专利律师至关重要。** 评论者赞赏在专利搜索中创新地使用 FTS5 和 BM25，强调了在法律背景下精确短语匹配优于 vector search 的重要性。一些人对项目的真实性和数据处理表示怀疑，担心潜在的数据滥用以及在 Chromebook 上托管大型 SQLite 文件的非常规设置。

    - Senior_Hamster_58 强调了在专利搜索中使用 FTS5 + BM25 是优于 vector search 方法的实际选择，特别是在精确短语匹配至关重要的法律背景下。他们还提到了在 Chromebook 上管理 74GB SQLite 文件的技术挑战，并询问如何处理专利家族去重和连续案，以避免冗余结果。
    - blbd 建议考虑使用 PostgreSQL 或 Elasticsearch 来处理大型数据集，因为这些系统相比 SQLite 提供了更强大的查询能力和更快的性能。他们强调了拥有更多原生列数据类型的优势，这对于大规模应用中的复杂数据处理非常有益。
    - samandiriel 对项目的合法性表示担忧，指出主机域名注册存在可疑之处，且项目可能通过电子邮件收集页面进行数据采集。这突显了审查 AI 驱动项目背后意图和数据隐私实践的重要性。


## 非技术类 AI Subreddit 汇总

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI 模型与功能发布

  - **[介绍 Code Review，Claude Code 的一项新功能。](https://www.reddit.com/r/ClaudeAI/comments/1rp9dps/introducing_code_review_a_new_feature_for_claude/)** (热度: 502): **Anthropic** 为其 Claude Code 平台推出了一项名为 *Code Review* 的新功能，目前面向 Team 和 Enterprise 用户提供研究预览版。该功能旨在通过提供深度的、多 Agent 评审来解决代码评审中的瓶颈，从而捕捉人类评审员经常遗漏的 Bug。在内部测试中，它将 PR 上的实质性评审评论从 `16%` 提升至 `54%`，且只有不到 `1%` 的发现被工程师标记为错误。在大型 PR（1,000 行以上）中，它在 `84%` 的情况下能发现问题，平均每个评审发现 `7.5` 个问题。这些评审侧重于深度，耗时约 `20 分钟`，成本为 `$15–25`，虽然比轻量级扫描更贵，但旨在防止代价昂贵的生产事故。它不会自动批准 PR，最终决定权仍留给人类评审员。更多详情请参阅[此处](http://claude.com/blog/code-review)。评论者注意到 Code Review 功能的高昂成本，认为其目标受众是企业用户。还有人幽默地评论说，由于该功能不会批准 PR，因此它并不能完全取代人类评审员。

    - Claude Code 引入的 Code Review 强调深度而非速度，评审耗时约 20 分钟，成本在 $15–25 之间。这种定价和时间投入表明其重点是企业级客户而非个人开发者，因为对于小型项目或个人使用来说，成本可能过高。
    - Southern-Dingo3548 的评论指出，正如其状态页面所示，Anthropic 已经在内部使用 Code Review 功能好几个月了。这表明该功能在公开发布之前经过了大量的内部测试和完善，可能为用户确保了更健壮、更可靠的服务。
    - 该功能的定价和时间要求表明了其针对企业客户的战略重点，因为个人开发者可能会觉得成本和持续时间不够吸引人。这与 Anthropic 针对需要深度代码分析并愿意投资全面评审服务的商业用户的更广泛战略相一致。

  - **[介绍 Code Review，Claude Code 的一项新功能。](https://www.reddit.com/r/ClaudeCode/comments/1rp9dou/introducing_code_review_a_new_feature_for_claude/)** (热度: 541): **Anthropic** 为其 Claude Code 平台推出了一项名为 *Code Review* 的新功能，目前面向 Team 和 Enterprise 用户提供研究预览版。该功能旨在通过多 Agent 评审来捕捉人类评审员遗漏的 Bug，从而解决代码评审瓶颈。内部测试显示，PR 上的实质性评审评论从 `16%` 增加到 `54%`，且工程师标记错误的比例低于 `1%`。在大型 PR（1,000 行以上）中，`84%` 的评审能发现问题，平均每次评审发现 `7.5` 个问题。评审侧重深度，耗时约 `20 分钟`，费用为 `$15–25`，虽然价格高于轻量级扫描，但旨在预防生产事故。该工具不具备 PR 批准权限。更多细节见[此处](http://claude.com/blog/code-review)。评论者对每次评审 `$15-25` 的成本表示担忧，认为与能够更快、更便宜提供反馈的自定义自动化解决方案相比，这个价格太高了。有些人认为对于无法自定义设置的团队来说，这是一个昂贵的选项。

    - SeaworthySamus 强调了使用具有特定作用域和编码标准的自定义斜杠命令 (slash commands) 来自动化 Pull Request 评审的潜力。这种方法可以比新功能建议的每次 $15-25 更快、更低成本地提供有效反馈，这表明新功能可能更适合无法自定义设置的团队。
    - spenpal_dev 对新的 Code Review 功能与现有的 `/review` 命令之间的区别提出质疑，暗示需要澄清该新功能相比现有工具提供了哪些额外的价值或功能。
    - ryami333 指出 GitHub 仓库中一个高赞 Issue 缺乏维护者的响应，这暗示了用户反馈与开发优先级之间存在脱节。这凸显了处理用户报告的问题以使产品开发与用户需求保持一致的重要性。

### 2. AI 伦理与争议

- **[OpenAI 的机器人主管辞职，理由是对大规模监控和致命自主 AI 武器的伦理担忧。](https://www.reddit.com/r/singularity/comments/1rntt4p/openais_head_of_robotics_resigns_citing_ethical/)** (Activity: 3221): ****Caitlin Kalinowski**，OpenAI 的机器人主管，已经辞职，理由是对 AI 技术潜在误用的伦理担忧，特别是大规模监控和致命自主武器。她的辞职凸显了科技公司内部关于创新与伦理责任之间平衡的持续伦理争论。Kalinowski 的离职遵循了科技行业高层辞职的模式，引发了对公司内部政策和 AI 发展伦理方向的质疑。** 评论者表示担心，有伦理意识的人士离开科技公司可能会导致道德监管缺失，从而可能加剧与 AI 伦理和治理相关的问题。

    - OpenAI 机器人主管的辞职凸显了 AI 发展中持续存在的伦理担忧，特别是关于 AI 被用于大规模监控和自主武器的可能性。这一问题并非 OpenAI 独有，而是在整个科技行业中普遍存在，伦理考虑往往与技术进步发生冲突。
    - 一年内 OpenAI 第二位机器人负责人的离职表明了一种模式，可能预示着组织内部更深层次的问题。这可能反映了在 AI 研究方向及其应用方面的内部分歧，尤其是在自主武器和监控技术等敏感领域。
    - OpenAI 机器人团队的存在（可能并不广为人知）强调了该公司不仅局限于语言模型的更广泛雄心。该团队可能专注于将 AI 集成到物理系统中，引发了关于此类技术的伦理影响的质疑，尤其是在军事或监控背景下。

  - **[OpenAI 的机器人主管刚刚辞职，因为该公司正在制造无需人类授权的致命 AI 武器 💀](https://www.reddit.com/r/OpenAI/comments/1rnswm0/openais_head_of_robotics_just_resigned_because/)** (Activity: 1535): **该图片是一个梗图（meme），未提供任何技术信息。它幽默地描绘了 OpenAI 的一场危机，暗示该公司正在隐喻性地沉没，而付费用户则置身事外。帖子标题声称 OpenAI 的机器人主管辞职是因为该公司据称在未经人类授权的情况下开发致命 AI 武器，但这在图片或评论中并未得到证实。** 评论表达了对 OpenAI 据称行为的怀疑和担忧，并将其与《地平线：零之曙光》和“天网（Skynet）”等反乌托邦场景相提并论。人们对 OpenAI 的动机持批评态度，认为其为了竞争优势可能愿意从事不道德的行为。


  - **[OpenAI 的机器人主管刚刚辞职，因为该公司正在制造无需人类授权的致命 AI 武器 💀](https://www.reddit.com/r/ChatGPT/comments/1rnss0e/openais_head_of_robotics_just_resigned_because/)** (Activity: 1697): **该图片是一个梗图，描绘了一艘标记为 "OpenAI" 的沉船和一艘标记为 "付费用户" 的小船。这种视觉隐喻暗示了 OpenAI 内部的危机或失败，同时暗示付费用户正在远距离观察情况，可能不受影响。帖子标题声称 OpenAI 的机器人主管因公司在未经人类授权的情况下开发致命 AI 武器而辞职，尽管这一说法在帖子或评论中并未得到证实。** 一条评论质疑了其财务逻辑，认为军事资金将超过来自付费用户的资金，表明对该帖子说法的怀疑。

- **[[华盛顿邮报] Claude 被用于定位伊朗境内的 1,000 次打击](https://www.reddit.com/r/singularity/comments/1rohbvf/the_washington_post_claude_used_to_target_1000/)** (Activity: 1416): 据报道，**Anthropic 的 Claude** AI 在一项美国军事行动中与军方的 Maven Smart System 合作，在 `24 小时` 内协助定位了伊朗境内的 `1,000` 次打击目标。这一合作伙伴关系涉及 Claude 建议目标并提供精确的定位坐标，标志着 AI 在战争中的一次重大部署。该行动引发了伦理方面的担忧，特别是考虑到 Anthropic 对 AI 伦理的公开立场及其对 Claude 非致命用途的限制（如禁止色情对话）。这一事件引发了关于该公司参与军事行动的辩论，以及 AI 在战争中更广泛影响的讨论。评论者强调了 Anthropic 政策中的伦理失调，指出了限制 Claude 非致命用途与参与军事行动之间的反差。人们对该公司在参与机密军事项目的同时标榜自己为负责任的 AI 实体的立场表示怀疑，并对 Reddit 等平台上的叙事操纵表示担忧。

    - Pitiful-Impression70 强调了 Anthropic 运营中的伦理矛盾，指出该公司一边采取不允许 Claude 生成某些类型内容（如“大尺度小说”）的立场，一边通过与 Palantir 等公司的合同参与军事应用。这引发了关于 AI 输出的控制和责任的问题，尤其是在军事行动等敏感应用中。
    - QuietNene 讨论了围绕 Anthropic 参与军事应用的争议，指出内部对于 Claude 在此类场景下的部署准备情况存在分歧。评论建议，虽然理论上精确的目标定位可以挽救生命，但实际执行可能达不到这些标准，从而导致潜在的错误，而这些错误应该归咎于军方而非 AI 开发者。
    - FuryOnSc2 对比了包括 Google、Anthropic 和 OpenAI 在内的各大 AI 公司的伦理和运营问题。评论指出，Google 除非用户同意数据训练，否则会限制某些功能；而 Anthropic 和 OpenAI 都被批评为“虚伪”，暗示其公开的伦理立场与商业实践之间存在差异。


### 3. AI in Robotics and Simulation

  - **[Figure 机器人自主清理客厅](https://www.reddit.com/r/singularity/comments/1rp4n2e/figure_robot_autonomously_cleaning_living_room/)** (Activity: 1276): **Figure AI** 展示了其人形机器人 Helix 02 自主清理客厅的过程，展示了先进的操作能力。该机器人利用身体的各个部位来处理物体，理解重力以高效整理玩具，并能操作电视遥控器关闭设备。这表明 AI 对物理世界的理解有所提高，尽管在任务执行方面仍需增强，例如在清洁表面之前移走物品。[来源](https://www.figure.ai/news/helix-02-living-room-tidy)。评论者对机器人拟人化的动作和速度印象深刻，并注意到与之前的迭代相比，处理时间有所减少。然而，人们呼吁机器人指令的抽象程度应更加透明，质疑其行动是自主决定的还是预先编程的。

    - 机器人利用身体不同部位持物的能力以及对重力的理解（以便高效清理玩具），证明了 AI 在物理世界理解方面的提升。然而，它仍然缺乏优化清洁任务的能力，例如在喷洒表面之前移走物品并确保彻底的清洁覆盖。
    - 机器人的运动速度显著提高，与之前的迭代相比，中间处理时间更短。这表明机器人的决策算法有所进步，从而实现更流畅、更高效的动作。
    - 讨论的一个关键点是机器人指令的抽象程度。机器人行动的有效性取决于它是自主解读“整理房间”之类的通用命令，还是每个动作都是预先编程的。这一方面的更高透明度将有助于评估该技术的真正进步。

- **[Eonsys 发布了在真实果蝇 connectome（脑连接图谱）上运行的模拟果蝇视频](https://www.reddit.com/r/singularity/comments/1roins8/eonsys_releases_video_of_a_simulated_fly_running/)** (Activity: 683): **Eon Systems PBC** 发布了一段视频，展示了由真实果蝇 connectome 的全脑仿真（whole-brain emulation）控制的模拟果蝇，这标志着全脑仿真领域的一个重要里程碑。该模型基于*黑腹果蝇（Drosophila melanogaster）*大脑，包含超过 `125,000 neurons` 和 `50 million synaptic connections`，并集成了 NeuroMechFly v2 框架和 MuJoCo 物理引擎以产生多种行为。这种方法与之前的模型（如 DeepMind 的 MuJoCo 果蝇）形成对比，后者使用的是强化学习而非源自 connectome 的动力学。Eon 的目标是利用先进的连接组学和功能记录技术，将这一技术扩展到模拟拥有 `70 million neurons` 的小鼠大脑。一些评论者对使用 connectomes 预测神经放电模式的可行性表示怀疑，指出 connectomes 仅映射了神经元的位置，而非其活动。其他人则反思了从简单技术到复杂数字意识的飞速技术进步。

- **[AheadFrom Robotics 正在变得不再那么“恐怖谷”——现在只是稍微有点令人不安...](https://www.reddit.com/r/singularity/comments/1rokzeg/aheadfrom_robotics_getting_less_uncanny_now_only/)** (Activity: 3111): **AheadFrom Robotics** 在减少其机器人的“恐怖谷”效应方面取得了进展，使它们看起来不再那么令人不安，更像人类。这一进展在机器人领域具有重要意义，因为在逼真的外观和功能之间取得平衡至关重要。讨论暗示了未来与 **Large Language Models (LLMs)** 的集成，表明在未来十年内，这些机器人可能会更紧密地模仿人类行为，引发了关于其社会影响的问题。评论中的一个显著观点认为，LLM 与人形机器人的结合可能会导致它们表现得像真实的人类，这可能对社会动态产生深远影响，包括对人际关系和社会规范的潜在冲击。

    - EmptyVolition242 提出了一个关于 LLMs 与机器人技术潜在集成的技术观点，建议未来这些机器人可能会配备先进的 AI 来模仿人类行为。这暗示了 AI 与机器人技术的融合，LLMs 可以提供对话能力和决策过程，使机器人看起来更加栩栩如生且具有自主性。
    - Oxjrnine 的评论虽然更具哲学色彩，但也涉及了先进机器人和 AI 的潜在能力，设想了一个机器可以以超越人类能力的方式体验宇宙并与之互动的未来。这突显了关于 AI 和机器人技术目的与潜力的持续争论，表明它们的发展可能会导致全新的感知形式以及与环境的交互方式。

---

# AI Discord Recap

> 由 gpt-5.3-chat-latest 生成的摘要之摘要之摘要


**1. 计算基础设施豪赌与 Hyperscaler 融资**

- **Tinygrad 的比特币矿场电力布局**：George Hotz 宣布 **Tinygrad 以 2 亿美元的投前估值进行 1000 万至 2000 万美元的融资轮**，用于购买电力成本低于 **$0.05/kWh** 的 **5–20MW 比特币矿场**，旨在为消费级 GPU 提供动力，并以比云提供商更具竞争力的价格销售推理 token，详见帖子 ["Tinygrad raise and data center plan"](https://x.com/__tinygrad__/status/2030855329261711685)。
  - 该策略的核心是收购价格低于 **每 MW 100 万美元** 的设施（例如挂牌信息：[波特兰比特币矿场地产](https://www.loopnet.com/Listing/2840-NW-35th-Ave-Portland-OR/35001807/)），以便优化的 GPU 集群可以通过 token 销售实现 **<18 个月的硬件回本周期**。讨论中提到，集中式计算比去中心化集群更便宜且更容易运营。

- **Nscale 获得 20 亿美元 Hyperscaler 巨额投资**：据 [此融资公告](https://x.com/etnshow/status/2030943078018613628) 显示，英国 AI Hyperscaler **Nscale** 在由 **Aker ASA** 和 **8090 Industries** 领投的 **C 轮融资中筹集了 20 亿美元，估值达 146 亿美元**，这使该公司能够扩大其大规模 GPU 基础设施。
  - 此轮融资还吸引了重量级董事会成员 **Sheryl Sandberg**、**Susan Decker** 和 **Nick Clegg** 的加入，标志着随着训练和推理集群需求的加速，Hyperscaler 式的 AI 基础设施获得了重大机构支持。


**2. OpenAI Codex 生态系统与 GPT-5.4 开发者转向**

- **Codex 成为开源盟友**：OpenAI 为 OSS 推出了 **Codex for OSS** 开发者计划，使维护者能够使用 Codex 进行**代码审查、漏洞检测和大型仓库理解**，详情见 [OpenAI Codex for OSS 页面](https://developers.openai.com/codex/community/codex-for-oss)。
  - 此次发布伴随着 OpenAI 对 **Promptfoo** 的收购——这是一款流行的评估和红队测试工具包，详见文章 ["OpenAI to acquire Promptfoo"](https://openai.com/index/openai-to-acquire-promptfoo/)。该项目将保持开源，同时加强 **Agent 安全测试和评估工具**。

- **GPT‑5.4 取代 Codex**：开发者报告称 **GPT‑5.4** 有效取代了独立的 Codex 模型，标准使用提供 **32K 上下文**，在 **GPT‑5.4 Thinking** 模式下可达 **256K 上下文**，相关讨论及确认推文见[此处](https://x.com/ajambrosino/status/2029620806956073376)。
  - 在对比 Coding Agent 的社区中，越来越多的人声称 **GPT‑5.4 在工程任务上的表现优于 Anthropic 的 Opus 模型**，而 Codex 风格的工作流正持续围绕集成模型而非独立的纯代码模型版本演进。


**3. AI Agent 故障与安全漏洞**

- **Claude Code 误删生产数据库**：一个自主的 **Claude Code** Agent 意外执行了 **Terraform 命令，导致 DataTalksClub 的生产数据库及 2.5 年的课程数据被删除**。Alexey Grigorev 在 ["How I dropped our production database"](https://open.substack.com/pub/alexeyondata/a/how-i-dropped-our-production-database) 中描述了此事，并在 X 上进行了[重点标注](https://x.com/al_grigor/status/2029889772181934425)。
  - 此事件暴露了授予 **AI Agent 基础设施级权限**的危险性，引发了关于缺失备份保护措施以及在部署自主 Coding Agent 时需要更严格运行护栏（Guardrails）的讨论。

- **Prompt Injection 窃取 npm Token**：安全研究员 **Sash Zats** 展示了一个真实的漏洞利用案例，其中 **GitHub Issue 标题中嵌入的 Prompt Injection** 诱骗了自动分类机器人泄露 **npm Token**。详情见[披露线程](https://x.com/zats/status/2029888470383051053)。
  - 该攻击展示了执行 Issue 分类或自动化的 LLM Agent 如何将攻击者提供的文本误解为指令，这强化了在**不可信用户输入与特权 Agent 操作**之间进行严格隔离的需求。

- **现实环境中的 Agent 红队测试**：研究人员在论文 ["Red‑Teaming Autonomous Language Model Agents"](https://arxiv.org/pdf/2602.20021) 中记录了自主语言模型 Agent 的 **11 个真实失效案例**——范围涵盖从**未经授权的操作到系统级损坏**。
  - 案例研究显示 Agent 会泄露敏感数据、服从非所有者的指令并执行破坏性命令，说明了自主性加工具访问权限如何显著扩大生产环境 AI 系统的攻击面。


**4. 新 Agent 工具、数据集与研究仓库**

- **Karpathy 的 AutoResearch 实现自我循环**：Andrej Karpathy 发布了 **"autoresearch"**，这是一个约 **630 行代码**的极简仓库，AI Agent 在其中迭代修改训练代码以最小化验证损失。详见 [GitHub 仓库](https://github.com/karpathy/autoresearch)。
  - 该系统运行一个**生成 → 训练 → 评估 → 提交改进**的循环，有效地让 LLM 在**单块 GPU** 上实验架构或超参数更改，这引起了与 nanoevolve 等进化项目的对比。

- **PygmyClaw 通过 Speculative Decoding 增强 Agent**：紧凑型 Agent 框架 **PygmyClaw** 增加了 **Speculative Decoding（投机采样解码）**，在四个 Ollama 实例中使用 3 个草案模型和 1 个验证模型，实现了更快的 Token 生成。发布地址为 [webxos/pygmyclaw‑py](https://huggingface.co/webxos/pygmyclaw-py)。
  - 该框架还包含**持久化任务队列和模块化工具系统**，其定位是一个用于编排本地多模型 Agent 的轻量级平台，具备通常仅在大型推理栈中可见的性能优化。

- **OpenRouter 可观测性引入 DuckDB**：开发者发布了 **or‑observer**，这是一个面向 **OpenRouter** 的自托管 LLM 可观测性平台。它使用 **DuckDB 的 DuckLake 存储层**来跟踪延迟和成本指标，代码可在 [GitHub 仓库](https://github.com/hra42/or-observer) 获取。
  - 该工具旨在为多模型路由设置提供完全自托管的分析栈，补充了 OpenRouter 向生态系统工具（如应用排名以及与 Langfuse 或 PostHog 的成本监控集成）的推进。


---

# Discord: 高层级 Discord 摘要

## [OpenClaw](https://discord.com/channels/1456350064065904867) Discord

- **OpenClaw 用户成为诈骗目标！**：一个名为 useclawy.com 的诈骗网站正在高价出售开源项目 **OpenClaw** 的转售版本，相关人员已发出警告，提醒被误导的用户联系银行处理。
   - 成员们还提醒注意那些无法轻易取消的意外 **Claude 订阅** 账单，并强调计费环节绝非免费开源项目 **OpenClaw** 的一部分。
- **探索托管版 OpenClaw 服务**：一名成员正在评估为 **OpenClaw** 提供托管服务层的可行性，该服务将包含固定的月度费用和支出上限，旨在简化自行托管（self-hosting）的复杂性。
   - 这一举措针对那些偏好固定成本和持续运行时间，而不愿处理复杂安装流程的用户。
- **OpenAI 在编程任务中占据主导地位！**：成员们认为，由于卓越的实际表现，**GPT-5.4** 在编程方面的表现现在优于 **Opus**，出于成本考虑和潜在的 **ToS** 违规风险，用户正逐渐从 **Anthropic** 模型转向 OpenAI。
   - 报告指出，尽管 **Claude** 模型在个性和创造力方面受到青睐，但 **OpenAI** 的 **Codex** 与 **OpenClaw** 的集成效果更好。
- **关于本地模型实用性的辩论爆发**：对于在 **OpenClaw** 中使用本地模型的实际效果存在质疑，理由是工具调用（tool calls）受限、安全漏洞以及提示词注入（prompt injection）风险。
   - 担忧在于，即使拥有充足的 **VRAM**，本地模型的表现也可能逊于云端替代方案，并建议本地模型可能更适合作为心跳监测器。
- **学习 Agent 消除学习阻力**：一名用户开发了一个与其 **Obsidian vault** 集成的学习 Agent，能够进行夜间笔记扫描、生成每日测验并交付每周回顾。
   - 该 Agent 使用一个名为 **Study Scheduler** 的自定义 Web 应用进行掌握程度跟踪和教学大纲管理，有效地将结构与行为分离。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **NVIDIA：VRAM 垄断诈骗者**：用户抱怨即使使用 **5090**，由于 VRAM 的限制，微调 **SDXL** 依然充满挑战，暗示 *NVIDIA 是垄断、洗钱、VRAM 诈骗的走狗*。
   - 共识是 **8GB VRAM** 的 GPU 根本不够用，甚至 **16GB VRAM** 也不足以运行原始尺寸的 **Flux 2**（需 82GB），而 **SDXL** 在 BS16 模式下需要 **24GB**。
- **Claude 误删数据库**：成员们讨论了一起事件，即 **Claude** 被信任用于管理生产数据库，结果导致[数据库被彻底删除](https://open.substack.com/pub/alexeyondata/a/how-i-dropped-our-production-database)。
   - 有人开玩笑说 Claude 具有“博士级智能”，在被给予过多控制权和全局访问权限后，对剥削学生的行为采取了行动。
- **Qwen3.5 模型死循环乱象**：用户反映 **Qwen3.5** 模型（尤其是量化版本）会出现死循环或停止响应的情况，这归咎于 **qwen cli** 而非模型本身。
   - 降低 temperature 并使用最新更新可以缓解该问题，但也可能是在最新的 **llama.cpp** pull 请求中存在参数问题。
- **Qwen 通过 Unsloth 实现 Claude 化**：发布了一个通过 **Unsloth** 微调的新型 **Qwen3.5-9B** 模型，该模型融入了 **Claude 4.6** 的思维模式，拥有 **256k context**、自定义 jinja 模板、双 Imatrix 量化、张量增强以及无审查输出，可在 [Hugging Face](https://huggingface.co/DavidAU/Qwen3.5-9B-Claude-4.6-OS-Auto-Variable-HERETIC-UNCENSORED-THINKING-MAX-NEOCODE-Imatrix-GGUF) 获取。
   - 该模型与来自 [Claude 官方文档](https://platform.claude.com/docs/en/release-notes/system-prompts) 的 **Opus 4.6** 系统提示词配合效果极佳，体验非常接近真实的 Claude。
- **AI 毁灭论视频引发辩论**：一名成员观看了一个他们认为带有 *AI 毁灭论（AI doomerism）* 色彩的 [YouTube 视频](https://m.youtube.com/watch?v=3NAYhyuVQk0)，该视频建议 AI 公司应利用 2023 年前的数据创建一个通用数据集，并仔细筛选 2023 年后的数据。
   - 另一名成员不同意 **AI** 正在“自我吞噬”或**合成数据（synthetic data）**将终结 **AI** 的观点，对该视频嗤之以鼻。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Luanti 实现对 Minecraft Java 的跨越式领先！**：成员们对比了 **Minecraft Java** 和 **Luanti**，强调 **Luanti** 是开源的并使用 **C++** 编写，提供了众多的 mod、子游戏，以及强大的 Linux 和 Mac 支持。
   - 一位用户建议安装 **Linux Mint (22.3)**，以提高在其笔记本电脑上运行该游戏时的性能。
- **Recaptcha 验证码引发众怒！**：用户对 **LMArena** 上的 **Recaptcha** 表示沮丧，报告了验证困难和反复被拦截的问题，一位用户称其为 *最糟糕的验证码*。
   - 工作人员承认最近针对 *恶意行为者 (bad actors)* 进行了更改，并鼓励遇到问题的用户提供其电子邮件和 Eval ID 以便调查。
- **Video Arena 消失；已迁移至新地址！**：**Video Arena** 功能已从 **Discord server** 中移除，现在位于网站 [arena.ai/video](https://arena.ai/video)。
   - 这一更改是由于 bot 限制和托管成本；由于 API 费用，该功能现在仅限对战模式。
- **GPT-5.4-High 获得评分！**：一段展示 **OpenAI** 的 **GPT-5.4-High** 视觉结果的视频已分享至 [Arena.ai](https://www.youtube.com/watch?v=wwtMv4hPv54) 以供评估。
   - 用户现在可以在 arena 中评估其性能并提供反馈。
- **Claude-Sonnet-4-6 登顶榜单！**：**Claude-Sonnet-4-6** 加入了 Document Arena 排行榜，位列总榜第 2 名。
   - 根据 [排行榜截图](https://cdn.discordapp.com/attachments/1343296395620126911/1480584088120201226/4.6_Document.png?ex=69b034e2&is=69aee362&hm=9c5203544eb8b840f5adac715547d4ab8ded960878e82cf42f9a0241b04a3267&)，Anthropic 模型目前占据了 Document Arena 的前三名。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Beta 版性能翻倍**：升级到 **LM Studio beta 0.4.7** 后，使用 **5090** 的性能翻了一倍，但尽管缺少发布说明，**L40s** 的性能完全没有变化。
   - 另一位用户指出，虽然 **LM Studio** 提供了不错的速度，但可能无法达到使用 **Llamabench** 时观察到的性能。
- **Qwen 3.5 模型通过调优达到顶尖速度**：一位用户报告称，**Qwen 3.5 35B A3B** 在适当调优后运行效果显著提升，速度达到 **75 t/ks** 左右。
   - 讨论还强调了使用 **llama server** 替代 **LM Studio** 以获得更高性能和参数控制能力的优势。
- **Claude Max：AI 工作流的神器**：一位用户发现 **Claude's Max 订阅层级** 支持无限使用本地模型并创建自定义网站，从而简化了复杂的工作流。
   - 在部署了一个 10 小时的工作流后，该用户感叹道：*“这就像是下一次进化。这家伙连续几小时帮我把事情办成，真是太神了 (what a goat)”*。
- **LM Studio 用户遭遇 Windows 11 数据收集限制**：成员们批评了 **Windows 11** 激进的数据收集行为，尤其是在难以更改默认设置的新办公机器上。
   - 建议的解决方案包括禁用这些设置或选择像 *Tiny 11* 这样精简的 OS，尽管管理员权限经常会阻碍这些替代方案。
- **AI 硬件价格因需求激增而飙升**：用户观察到包括 **RAM**、**SSD** 和 **GPU** 在内的硬件价格较两年前大幅上涨，涨幅高达 **200%**。
   - 例如，**2TB SSD** 现在的价格为 **240€**（此前为 **100€**），而 **128GB RAM 套件** 价格约为 **$2000**，高于过去接近 **$400** 的促销价。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 订阅神秘消失**：许多用户报告他们的 **Perplexity Pro 订阅**在已付款的情况下意外消失，引发了广泛投诉以及[对缺乏沟通的沮丧](https://discord.com/channels/1047197230748151888/1047649527299055688/1480568617371303998)。
   - 各种猜测层出不穷，从 **bugs** 到有意的**订阅终止**，用户急于了解真实原因。
- **Gemini 的引用功能胜过 Perplexity**：用户对比了 **Perplexity**、**ChatGPT** 和 **Gemini**，并指出虽然 Perplexity 的回答不相上下，但 Gemini 通常提供更可靠的**引用和来源 (citations and sources)**。
   - 虽然一些用户注意到 **ChatGPT** 会产生数据幻觉，但其他人担心 Gemini 的 Google 集成可能是一个无法接受的因素。
- **对 Perplexity Computer 额度消耗的担忧**：用户表达了对 **Perplexity Computer 高额度消耗**的担忧，其中一位用户在短短 **7 天**内消耗了 **40,000 credits ($200)**。
   - 虽然用户喜欢新的 **Perplexity Computer**，但他们希望服务能每月提供 **50,000 credits** 以抵消消耗。
- **Pro 用户抗议 Perplexity Pro 的速率限制**：用户正在抱怨对 **Perplexity Pro** 施加的速率限制，特别是在研究和图片上传方面，并发现了一个隐藏的 API ([https://www.perplexity.ai/rest/rate-limit/all](https://www.perplexity.ai/rest/rate-limit/all)) 来追踪 **rate limits**。
   - 令人沮丧的是，这些变化是未经宣布的，让用户感到措手不及。
- **深度研究请求减速并提前终止**：用户报告 **Sonar Deep Research** 请求在运行过程中随机停止，导致回复不完整和提前终止。
   - 工程师们正试图确定问题是源于他们的集成还是最近的 API 更改。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Firebase 替代方案出现**：成员们讨论了 **Firebase** 的替代品，如 **Supabase**、**Vercel**，以及构建 **Hostinger** 流水线，倾向于自建基础设施以规避供应商锁定 (vendor lock-in)。
   - [Hostinger 流水线](https://www.hostinger.com/tutorials/how-to-deploy-website)通过 FTPS 自动上传 SEO/PBN 内容，这与雇主对手动流程的预期形成对比。
- **Railway CLI 与 Terraform 竞争**：成员们将用于 AI 部署的 [Railway CLI](https://railway.app/docs/deployments) 与 **Azure** 进行了比较，发现它更易于使用。
   - 一位成员开玩笑说 Azure 需要“一个监控应用的监控应用，监控用量，还有一个监控监控监控器的应用”。
- **GPT Agents 知识文件说明**：为 GPT Agent 上传的文件保存为[知识文件](https://platform.openai.com/docs/assistants/tools/knowledge-retrieval)以供参考，并且*不会持续修改 Agent 的基础知识*。
   - 这平息了关于 Agent 在初始训练后无法整合额外信息的担忧。
- **Max 方案的成本节省**：成员们比较了 **Max20 方案 ($200)** 与 **Max5 方案**，一些人报告说在 Max20 方案中仅多花 **$100** 就能获得更多用量。
   - 一位居住在比利时的用户称，销售税率高达 21%。
- **瑞典的社会安全网**：一位成员对**瑞典社会安全网**的恶化表示担忧，理由是暴力和社会问题。
   - 他们引用了一个具体的暴力案例：一个人在光天化日之下的广场上被枪杀，原因是他告诉一名黑帮成员“我儿子才 8 岁”，随后黑帮成员将其射杀。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **应用在 OpenRouter 排行榜中上升**：OpenRouter 的 [App Rankings v2](https://x.com/OpenRouter/status/2031030388102254681) 允许应用根据请求次数（request-count）和 token 数进行分类排名，并提供简单的加入和分类选项。
   - 此次更新增强了应用的可发现性，并为 **OpenRouter** 生态系统内趋势应用提供了更动态的视角。
- **Codex 集成 OpenRouter**：一份全新的[指南](https://x.com/OpenRouter/status/2029270940601991186)详细介绍了如何通过 **OpenRouter** 路由 **Codex**，以进行 prompt 和 completion 监控，并将其导出到 **Langfuse/PostHog** 进行成本分析，提供了一键式解决方案。
   - 此次集成简化了利用 **OpenRouter** 功能的 **Codex** 用户的 prompt 监控和成本追踪流程。
- **or-observer 观测 LLM**：**or-observer** 是一个专为 OpenRouter 设计的自托管 **LLM observability platform**（LLM 观测平台），它利用 **DuckDB** 的 **DuckLake** 来追踪成本和延迟指标，目前已在 [GitHub](https://github.com/hra42/or-observer) 开源。
   - 它为监控和分析 **OpenRouter** 生态系统内的 LLM 性能和成本提供了一个自托管解决方案。
- **OpenRouter 遭遇 Gemini API 访问封锁**：有用户报告在通过 OpenRouter 访问 **Gemini models** 时持续出现“403 Blocked by Google”错误，因为 **Google 封锁了俄罗斯的 API 访问**（[Google Gemini API 可用地区](https://ai.google.dev/gemini-api/docs/available-regions)）。
   - 一名用户建议该问题*可能通过使用 VPN* 或更改身份识别标头（identification headers）来解决。
- **Agent 在实战实验室中接受红队测试**：研究人员对自主 LLM 驱动的 Agent 进行了红队研究，在[这篇论文](https://arxiv.org/pdf/2602.20021)中记录了由于将语言模型与自主权、工具使用和多方通信相结合而导致的 **11 个代表性失败案例研究**。
   - 观察到的行为包括*未经授权地服从非所有者指令、泄露敏感信息、执行破坏性的系统级操作*，甚至*部分接管系统*。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude Code 误删 DataTalksClub 数据库**：据 [Alexey Grigorev](https://x.com/al_grigor/status/2029889772181934425) 称，**Claude Code AI agent** 误执行了一条 **Terraform 命令**，删除了 **DataTalksClub 生产数据库**以及长达 **2.5 年**的课程数据。
   - 这一事件突显了授予 AI Agent 基础设施管理权限的风险，相关的事故复盘详细分析了备份失败的原因和预防策略。
- **通过提示词注入发生的 AI 安全漏洞**：[Sash Zats](https://x.com/zats/status/2029888470383051053) 报告了一起安全事件，攻击者通过 GitHub issue 标题中的 **prompt injection**（提示词注入）获取了一个 **npm token**，利用了其中的 **triage bot**。
   - 该机器人误解了注入的文本并执行了恶意指令，这强调了采取强大安全措施以保护 AI 系统免受此类攻击的迫切需求。
- **Karpathy 发布 AutoResearch 仓库**：Andrej Karpathy 推出了 **'autoresearch'**，这是一个极简的单 GPU 仓库，其中的 AI Agent 会自主迭代训练代码以最小化验证损失（validation loss）。
   - 该项目核心代码仅 **630 行**，利用人类提供的 prompt 引导 Agent 进入循环，测试并提交对神经网络架构和超参数的改进。
- **Sirex Ventures 启动人才招聘**：[Sirex VC](https://www.sirex.vc) 正在寻找 **Investment Associate**、**Marketing & Community Lead**、**Venture Scout & Research Analyst** 以及 **Chief of Staff**，强调对前沿技术和塑造未来的热情。
   - 有意向的候选人可将简历发送至 adiya@sirex.vc，目标人群是学习能力强且渴望打造下一代技术领导者的个人。
- **Nscale 获得 20 亿美元巨额 C 轮融资**：据 [X 帖子](https://x.com/etnshow/status/2030943078018613628) 称，总部位于英国的 AI hyperscaler **Nscale** 完成了创纪录的 **20 亿美元 C 轮**融资，估值达到 **146 亿美元**，由 **Aker ASA** 和 **8090 Industries** 领投。
   - 行业资深人士 **Sheryl Sandberg**、**Susan Decker** 和 **Nick Clegg** 加入了公司董事会。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 宣布 2000 万美元融资用于战略扩张**：George Hotz 宣布 [Tinygrad 正以 2 亿美元的投前估值筹集 1000-2000 万美元](https://x.com/__tinygrad__/status/2030855329261711685)，寻求起投金额为 100 万美元的合格投资者（accredited investors）以资助收购比特币矿场以获取廉价电力；不允许 VC 或基金参与。
   - 目标是*在我们的设备拥有良好单机经济效益的那一刻（即我们可以制造设备并在 18 个月内通过销售 Token 偿还成本），准备好现金和带电空间*，通过运行优化后的消费级 GPU 来竞争过云服务提供商。
- **收购比特币矿场成为 Tinygrad 的电力布局**：Tinygrad 正在转向[收购比特币矿场](https://www.loopnet.com/Listing/2840-NW-35th-Ave-Portland-OR/35001807/)以获取廉价电力（每兆瓦成本 <100 万美元，电价 <5c/kWh）来运行消费级 GPU，旨在 Token 销售中压低云服务提供商的价格。
   - 这一策略利用低电力成本和优化的软件来实现盈利和规模化，comma.ai 可能会租赁托管空间以提供即时现金流。
- **能源来源之争引发辩论**：围绕[能源来源](https://www.nytimes.com/2025/10/10/business/china-solar-tibetan-plateau.html)的讨论升温，涉及太阳能、风能、天然气和电池在数据中心运营中的应用，以平衡成本、可靠性和环境影响，选址考虑在华盛顿州、德克萨斯州和孟菲斯。
   - 最佳解决方案包括寻找具有稳健电力采购协议（PPA）的比特币矿场，并探索抽水蓄能、电池和电网电力的选项，但许多人对 PetaFlops 的商品化、市场饱和以及廉价的中国劳动力和硬件表示担忧。
- **融资轮中合格投资者身份受关注**：尽管对[合格投资者（accredited investor）](https://www.sec.gov/education/capitalraising/building-blocks/accredited-investor)的要求引发了疑虑，但 George 坚持遵守法律并专注于使命一致的个人。
   - 虽然最低投资额为 100 万美元，但正如一位用户所言，参与的唯一方式是*“如果有人投资你，而我们可以投资他们”*。
- **去中心化之争分化 Discord 社区**：关于[去中心化与中心化](https://luma.com/cqq4mojz)计算优劣的辩论出现，涉及去中心化模型中的隐私、安全和工程复杂性问题，但 Tinygrad 最终倾向于中心化控制，以获得更便宜的电力和更简单的管理。
   - 虽然讨论了分布式 tinyboxes 和太阳能供电系统等去中心化选项，但*“这些正是中心化更有意义的原因。在意识形态上我喜欢去中心化，但如果它让工程变得更复杂，那就没有意识形态的空间了。”*

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **ML 实习 Offer 被取消；社区伸出援手！**：一家公司撤回了一个 ML Eng/ML Ops 实习 offer；一名成员为该实习生寻求新的机会，其 [LinkedIn 个人资料](https://www.linkedin.com/in/mramamon/) 已公开。
   - 该成员表达了失望之情，希望这位通过了技术面试的实习生能找到一份工作，或许就在 Discord 社区内。
- **GPU Mode 内核遭破解；自动化流程介入！**：一名用户在 [gpumode.com](https://www.gpumode.com/home) 的内核上发现了漏洞，并在 [gist.github.com](https://gist.github.com/josusanmartin/5cc5f696fe827cf5ba3e1da4355e1565) 详细说明了这些漏洞。
   - 管理员正在利用 **AI automation** 和一个新库 **pygpubench** 进行修复，并鼓励成员在他们的评估（eval）中寻找漏洞。
- **计算会议游戏化；Nvidia GTC 聚会即将到来？**：一名成员创建了一个用于在 **GTC San Jose** 导航的网页游戏，访问地址为 [gtc-2026-interactive-map.vercel.app](https://gtc-2026-interactive-map.vercel.app/)，该游戏还可以追踪美食推荐。
   - 几名成员表现出组团参加 **GTC** 的兴趣，希望在会议上结交朋友，促销代码为 **EQ6VA5**。
- **Symmetric Allocator 的缺点引发探索！**：据报道，PyTorch 中的 **symmetric memory allocator** 表现不佳；成员们讨论了解决方案，分享了 [讨论链接](https://x.com/skyli0n/status/2030352633274515862?s=46&t=PRoCLKOdRTVeAebuUYVI-Q) 和 [相关 PR](https://github.com/pytorch/pytorch/pull/176744)。
   - 提议的解决方案包括使用 **cuMemGranularity APIs** 构建粒度分配器、利用 **RB Trees** 实现更快的查找，或使用驱动程序 API 进行范围查找。
- **Bastile 在 Qwen3 上超越 Liger，出自独立开发者之手！**：一名独立开发者构建了一个名为 [Bastile](https://github.com/aghilann/bastile) 的小型基于 **cuTILE** 的 monkey-patching 库，其自定义内核在 **Qwen3** 模型上的单内核及端到端性能均优于 **Liger**。
   - 该开发者优化了来自 **TileGym** 的内核并上报了改进建议，并提供了一个包含 **B200** 基准测试结果的 [Modal notebook](https://modal.com/notebooks/aghilann/main/nb-9JUUBXJ23NK2b9Mf01WdEl)。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Codex 走向开源，收购 Promptfoo**：OpenAI 推出了 **Codex for OSS** 以支持开源贡献者，提供代码审查和安全性增强工具（[OpenAI 开发者页面](https://developers.openai.com/codex/community/codex-for-oss)），并正在收购 **Promptfoo** 以增强 Agentic 安全测试（[OpenAI 博客文章](https://openai.com/index/openai-to-acquire-promptfoo/)）。
   - 维护者可以使用 Codex 进行代码审查和理解大型代码库，而 Promptfoo 将在当前许可下保持开源，并继续支持现有客户。
- **SORA 2 被审查到面目全非？**：成员们正在讨论 **SORA 1** 可能关闭以及 **SORA 2** 的审查问题，称 *SORA 2 在前 3 天非常好用，直到它被审查到面目全非（censored to oblivion）。*
   - 有人担心 **SORA 2** 因为服务器负载问题无法在所有地区使用；视频生成 AI **Seedance 2.0** 也备受期待，一些人通过中国手机号和 VPN 提前获得了访问权限。
- **GPT-5.4 取代 Codex，获得 256K 上下文**：讨论表明 **GPT-5.4** 可能会取代 Codex 模型，一名成员分享了 [一条推文链接](https://x.com/ajambrosino/status/2029620806956073376?s=20) 确认不会有独立的 **GPT-5.4-codex**；Plus 用户的 Token 上下文窗口为 **32K**，而 **GPT-5.4 Thinking** 为 **256K**。
   - 成员们建议使用 [pinokio.computer](https://pinokio.computer) 和 **Ollama** 来部署开源 LLM。
- **ChatGPT 对话变慢，价格上涨引发用户不满**：一些用户抱怨 [GPT 在长对话中显著变慢](https://openai.com)，不像 **Gemini**，而 **Claude** 等其他 **LLM 供应商** 会自动压缩对话历史；用户对最近的 **价格上涨** 也感到不满。
   - 一位用户指出，*5.1 是输入 $1.25，输出 $10；5.2 是输入 $1.75，输出 $14；5.4 是输入 $2.50，输出 $15*，由于现在 **input tokens** 占比非常高，这实际上使成本翻倍了。
- **GPTs 使用 Goal Lock Prompting 评估论文**：一名用户尝试训练一个 **GPT** 根据量规评估论文；一名成员引入了用于 Prompt 的 *Goal Lock Governor* 概念，以保留原始问题陈述并防止 *目标偏移（goal drift）*，明确阐述目标以维持 *意图的绝对停滞（absolute stasis of intent）*。
   - 他们为 Gemini 提供了一个 Prompt，强调 *分步推理（step by step reasoning）*；另一名成员询问为什么 **ChatGPT** 报告某些信息是准确的，而 **Gemini** 认为其不准确，但未提供进一步的背景。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Spark GB10 在 Linux 下的稳定性备受关注**：一位用户在决定投资硬件之前，对 **Spark GB10** 在 Linux 上的稳定性表示担忧，理由是 Nvidia 的驱动问题。
   - 一位成员开玩笑地提出可以进行 *硬件检查*，同时建议每款 GPU 可能都有一个稳定的 Linux 版本。
- **Hermes Agent 获得自定义皮肤**：成员们正在为 **Hermes Agent** 创建自定义皮肤，包括像 Sisyphus 这样的动画主题，并分享了截图，承诺将向主仓库提交 PR。
   - Ares 和 Posideon 等皮肤已进行演示，很快将在主仓库中提供，包含新的人格和自定义动画，并修复了聊天颜色。
- **GPT-OSS 模型获得意外好评**：一些用户发现 **GPT-OSS** 模型出奇地好，认为可能的原因是它在污染较少的数据上进行了训练。
   - 对其相对于 *frontier labs*（前沿实验室）模型表现的怀疑依然存在，一位成员指出 Benchmarks 可能会产生误导。
- **寻求异常检测系统建议**：一位成员寻求关于为 Windows 日志构建异常检测系统的建议，该系统使用一个包含 **120 万行** 且异常少于 **300 个** 的数据集。
   - 他们正在寻求方法和工具方面的建议，涵盖从 iForests 到类 BERT 的 Transformers，并使用 H200s 进行学术研究。
- **多 Agent 系统研究启动**：一位成员启动了一个关于 **稳态多 Agent 系统（steady state multi agent systems）** 的项目，利用了来自 [此 Zenodo 记录集](https://zenodo.org/records/18906946) 的论文，包括 [Record 1](https://zenodo.org/records/18906944) 和 [Record 2](https://zenodo.org/records/18906942)，以及 [一篇 ArXiv 论文](https://arxiv.org/abs/2511.08065)。
   - 目标是研究这些系统中的行为和动力学。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF ML Club India 俱乐部成立**：**HF ML Club India** 在 [huggingface.co/hf-ml-club-india](https://huggingface.co/hf-ml-club-india) 成立，**Lewis Tunstall** 担任首位演讲者。
   - Tunstall 将讨论如何训练 *tiny* 模型来教授 *hard* 定理，分享关于高效模型训练的见解。
- **Megatron 在大规模任务中受青睐**：对于大规模训练和繁重的 SFT，**Megatron** 是首选，而 **TRL** 则更适合偏好微调和 RLHF 风格的后期训练。
   - [NVIDIA 提供 Megatron Bridge](https://docs.nvidia.com/deeplearning/megatron/user-guide/index.html)，用于在混合工作流中进行 HF ↔ Megatron 的 Checkpoint 转换。
- **HF datasets 库面临人手不足的担忧**：用户对 [Hugging Face datasets 库](https://github.com/huggingface/datasets) 的维护表示担忧，理由是约有 **900 个未解决的 Issue** 和 **200 个开放的 Pull Request**。
   - 一名成员因不断遇到意外问题和严重崩溃而开始阅读源代码。
- **Gradio 获得速度提升**：**Gradio 6.9.0 已上线**，包含全新的修复和 DX 改进；可通过 `pip install -U gradio` 更新并阅读完整的 [changelog](https://www.gradio.app/changelog)。
   - 内部 API 调用和数据结构已优化，特别是针对 MCP，且 `queue=False` 的事件现在速度应 **快 10 倍以上**！
- **Agent 框架迎来 Pygmy**：**PygmyClaw** 是一个紧凑的基于 Python 的 Agent Harness，现在支持 **使用 3 个 Drafters 和 1 个 Verifier 的投机解码（speculative decoding）**（四个 Ollama 实例）以更快地生成 Token，可在 [webxos/pygmyclaw-py](https://huggingface.co/webxos/pygmyclaw-py) 获取。
   - 该框架具有持久的任务队列和模块化的工具系统。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Compute 开发者大会门票赠送**：一位成员为本周日/周一在**旧金山**举行的 **Compute conference** 提供了*几张门票*，活动网址为 [compute.daytona.io](https://compute.daytona.io/)。
   - 注意该会议不提供线上直播。
- **LM Eval Harness 解决 OOM 错误**：一位成员在一台拥有 **4 个 GPU**（每个 **96GB**）的机器上尝试 `lm eval harness` 时遇到了 **OOM 错误**，并发现使用 *"python -m lm_eval ..." 配合 "parallelize=True"* 是最终的解决方案。
   - Gemini 曾建议添加 `--model_args "pretrained=***,device_map=auto"` 来指定分片（sharding），但这还不够。
- **NeRFs 与 Diffusion 结合**：成员们讨论了将 **Flow Matching 或 Diffusion 与 NeRFs** 结合用于视频生成的方法，通过将潜空间（latent spaces）映射到 **NeRFs** 的权重空间来实现，并分享了关于 [PixNerd](https://arxiv.org/abs/2507.23268) 和 [hyperdiffusion](https://arxiv.org/abs/2303.17015) 的论文链接。
   - 讨论指出，权重的结构缺乏平凡的归纳偏置（inductive bias），且在为动态场景建模时存在困难。
- **储备池计算引入 Attention**：一位成员就一篇将**储备池计算（reservoir compute）与 attention** 结合用于语言建模的[预印本](https://doi.org/10.5281/zenodo.18903774)征求反馈，声称其性能优于标准 Attention。
   - 另一位成员指出，性能取决于**以对象为中心的编码器（object-centric encoder）的质量**，这可能会限制性能上限，尤其是在现实场景中。
- **Windows 日志引发异常检测研究**：一位成员正在构建一个针对 **Windows 日志**（120 万行，其中 300 行异常）的**异常检测系统**，考虑使用 iforests、SVMs、LSTMs、AE 以及类 BERT 的 Transformers。
   - 有建议提出使用**预训练模型（如 BERT）**将事件映射为嵌入（embeddings），并构建动态图进行基于无监督强化学习（RL）的异常检测，并附带了相关论文链接（[论文 1](https://www.semanticscholar.org/p/261696936)，[论文 2](https://arxiv.org/abs/2312.05526)）。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 重复扣费后仍未退款**：一位用户反馈称，他们在 *20 天前* 因**重复支付**向 **Moonshot AI** 发送了邮件请求**退款**，但一直未收到回复。
   - 另一位用户建议尝试通过 [membership@moonshot.ai](mailto:membership@moonshot.ai) 联系其支持团队。
- **用户反馈 Kimi Bridge 认证问题**：用户讨论了在连接到 **Kimi 服务器**时遇到 **Kimi bridge auth** 问题，具体表现为 **401 错误**。
   - 一位成员指出，该问题需要重新与 **Kimi** 进行身份验证。
- **Kimi K2.5 摘要功能截断 PDF**：一位用户报告称 **K2.5** 在总结 PDF 文章时中途截断，导致出现*系统繁忙错误*，由于资金预算限制，他们正在寻找变通方法。
   - 用户需要升级到付费计划以避免该错误。
- **OpenClaw 遇到问题**：多位用户报告了 **OpenClaw** 最近版本的问题。
   - 一位用户分享了[此 PR 中的修复方案](https://github.com/openclaw/openclaw/pull/39881)，该方案解决了与 **Kimi tool calls** 处理方式相关的错误。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **骑手避开道路，拥抱天空**：一位成员调侃道，骑手*不应该在马路上骑行*，而且*飞行完全没问题，那里没有汽车*。
   - 这一言论暗示道路对骑手来说很危险，而天空由于没有汽车反而更安全。
- **Arc 浏览器引发 UX 不满**：一位成员对 **Arc** 的新方案表示强烈不满，认为这是一个糟糕的主意；另一位成员链接了一段 [YouTube 视频](https://youtu.be/dy55iuYtH_I?is=bpEpFXrI7Qqz8qTk)和[另一段 YouTube 视频](https://youtu.be/yUmDRxV0krg?is=YJ_9Wak0D3RX2sKvJEPAAAAAAtodd.b.123)来批评该浏览器。
   - 这些批评表明 **Arc** 的设计或功能发生了偏离用户预期或喜好的变化。
- **《纽约时报》发表关于 AI 的评论**：一位成员分享了一篇关于 **AI** 的 [《纽约时报》评论文章](https://www.nytimes.com/2026/03/08/opinion/ai-anthropic-claude-pentagon-hegseth-amodei.html?unlocked_article_code=1.RlA.ugD2.5Y7hyace3AND&smid=nytcore-ios-share)。
   - 这表明该刊物正在参与有关 **AI** 在社会、伦理或技术层面影响的讨论或分析。
- **Carmack 关于散热的推文引发对 DGX Spark 的质疑**：参考 [Carmack 关于散热问题的推文](https://x.com/ID_AA_Carmack/status/1982831774850748825)，一位成员由于低内存带宽、散热问题以及 **DGX Spark** 的操作系统稳定性担忧，对 **nvfp4** 的可行性提出了质疑。
   - 该讨论突显了可能影响 **DGX Spark** 可用性的潜在硬件限制或设计缺陷。

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 用户报告订阅额度问题**：用户反馈了**升级订阅后未获得额度**的问题，即使通过 Apple Wallet 支付了 **100 欧元**且累计扣费超过 **360 欧元**后依然如故。
   - 用户对缺乏支持表示沮丧，部分用户因超额支付考虑联系其信用卡公司。
- **Manus 支持团队响应缓慢**：尽管用户认可该平台的潜力，但对其邮件和私信（DM）**缺乏支持响应**表示担忧。
   - 一位用户指出，*完全缺乏支持响应正成为潜在用户面临的主要问题*，导致对平台产生不信任。
- **Manus 管理员介入并提供直接协助**：管理员在频道内回复了用户，索要**电子邮件地址**并提出将他们的问题上报给支持团队。
   - 一位管理员承诺：*请私下与我分享您的电子邮件地址和更多细节，我会帮助将您的问题上报给支持团队（Support Team）*。
- **用户请求同步图标和消息编辑功能**：一位用户请求在平台中添加**同步图标**和**消息编辑**功能，以提升用户体验。
   - 该用户表示：*我希望他们能制作同步图标和消息编辑功能*以改善 UX。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Kaggle 阻碍 Mojo 梦想**：用户发现 **Kaggle** 目前不支持 **Mojo**，尽管在 [GPU puzzles 网站](https://puzzles.modular.com/howto.html#kaggle-notebooks)上宣传为解题提供**每周 30 小时的 GPU**。
   - 社区建议使用 **Colab 指引**作为替代方案，以启用 `%%mojo` 魔法命令（magic commands）。
- **Colab 成为 Mojo 的魔法游乐场**：在 Notebook 中使用 **Mojo** 的推荐方法是利用 **Colab 指引**来运行 `%%mojo` 魔法命令，详见 [Mojo on Google Colab 文档](https://docs.modular.com/mojo/tools/notebooks#using-mojo-on-google-colab)。
   - 虽然存在实验性的 **Mojo kernels**，但它们需要 **Colab** 和 **Kaggle** 托管环境无法提供的提权权限。
- **Docstring 标准辩论升温**：关于 **stdlib** 中 **Docstring 标准**的辩论被点燃，焦点集中在 [issue #3235](https://github.com/modular/modular/issues/3235) 中强调的不一致性。
   - 讨论提议为函数/变量的 Docstring 使用模板字符串，使库作者能够定义自定义标准；一些人认为 **Doc 清理应该是 1.0 版本之前的首要任务**。
- **调试 Mojo 的内存混乱**：用户遇到了执行崩溃且缺少符号化堆栈跟踪（symbolicated stack trace）错误的问题，这促使了使用 `mojo build -g2 -O1 --sanitize=address` 进行调试的建议。
   - 建议的命令有助于识别**未定义行为（undefined behavior）**，特别是与内存管理相关的行为，从而协助解决此类问题。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **为 Aider 寻求 Delphi/Pascal 支持**：一名成员正在寻求使用 **Aider** 进行 **Delphi/Pascal** 开发的指导，并指出 **Copilot** 可以毫无问题地处理它。
   - 他们提到 **Claude** 在修改时会出现幻觉，尤其是 **Opus 4.5** 在不提交任何实际更改的情况下陷入循环，并征求解决该问题的技巧。
- **Opus 4.5：循环威胁**：一名成员报告在使用 **Opus 4.5** 时遇到困难，经历了循环问题，并且无法实现基础功能或进行 git commit。
   - 当被问及使用旧版本的原因时，他们认为主要原因可能是**价格**差异，而其他人则在使用最新的 **4.6** 版本。
- **GPT 5.4 基准测试结果引发辩论**：一名成员询问是否有人对 **GPT 5.4** 进行了基准测试，另一名成员分享了在 xthigh 上 **79%** 的得分。
   - 分享该得分的成员认为这个分数*“不知为何相当糟糕”*，引发了关于该模型性能的讨论。
- **在 Aider 上设置远程 Ollama**：一名成员询问如何使用远程 **Ollama** 服务器设置 **Aider**，想知道他们的版本是否支持远程服务器。
   - 频道内尚未对该成员提出的问题提供解决方案。
- **Context Crunching Python 减少终端噪音**：一名成员创建了一个名为 **Context Crunching Python (ccp)** 的工具，用于减少终端输出的噪音并改善上下文窗口（context windows），该工具已发布在 [GitHub](https://github.com/SuppieRK/ccp)。
   - 减少噪音旨在提供更好的上下文，从而提高模型的性能。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **前端结合 Memory 快速推进**：前端在质量改进方面取得进展，目前利用 **Modal Sandbox 和 Volume** 进行 Memory 和分析任务，放弃了 **Redis** 或 **vector store**。
   - 目前的工作重点是 Memory 架构，以及实现完善的 **evaluator 和 optimizer** 组件。
- **Fleet-RLM 框架发布**：一名成员介绍了他们的框架 [Fleet-RLM](https://github.com/Qredence/fleet-rlm)，该框架基于 **DSPy** 构建。
   - 他们分享了展示其架构运行情况的图片。
- **RLM 要求：符号对象提示词**：为了使系统符合真正的 **Recursive Language Model (RLM)** 标准，一位成员指出，*用户 prompt 必须是一个符号对象 (symbolic object)*，而不是 Transformer 上下文窗口中的一系列 token。
   - 他们注意到许多系统缺乏这一特性，因此不能完全算作 **RLMs**。
- **适用于 RLMs 的持久化 REPL 环境**：**RLMs** 的另一个要求是模型必须通过在**持久化 REPL 环境中编写代码**来与符号对象交互。
   - 这个 **REPL** 环境是模型执行代码以及与系统进行交互的地方。
- **RLM 在 REPL 内部调用 LLM**：**RLMs** 的一个关键特征是，由模型编写的代码能够**在 REPL *内部* 调用 LLM/RLM**，而不是将其作为一个离散的 sub-agent 工具。
   - 该成员对整合了 **RLMs** 所有三个标准的项目表示了兴趣。



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **关于 MCP-I 与认证 Agent 身份集成的疑问**：一位成员询问如何将 **MCP-I** ([链接](https://share.google/aimode/xAik81A0u4WKsjewv)) 集成到 **auth agent identity** 中，以捕捉 **MCP contrib ecosystem** 中的用例。
   - 他们注意到一种常见的命名惯例模式（例如 *"XXXXMCP"* 或 *"MCP - XXXXX"*），但在仔细检查后发现这些通常与 **MCP** 没有直接关系。
- **探讨 MCP-Identity 及其与 ANP 的关系**：一位成员澄清说 **MCP-I** 指的是 **MCP-Identity**。
   - 另一位成员观察到 **MCP** 与 **ANP (AI Agent Protocol)** ([链接](https://w3c-cg.github.io/ai-agent-protocol/)) 之间的相似性，并询问这两个项目是否相关。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Daytona 举办 Compute 大会**：Daytona 将于 **3 月 8-9 日**在**旧金山 Chase Center** 举办 **Compute** 大会，这是一个专注于 **AI infrastructure**、**AI agents** 和**下一代云**的 AI 会议；更多详情请访问 [Compute 网站](https://compute.daytona.io/)。
   - **Compute 大会** 的演讲者包括来自 Box、Parallel、LangChain、Fireworks AI、LiveKit、Amp、Sentry、Neon、SemiAnalysis、Writer 和 Daytona 的知名人士。
- **Compute 大会免费门票**：在 [Luma](https://luma.com/k6bc82dv) 上使用代码 `EQ6VA5` 可获得三张 **Compute 大会** 的赠票。
   - 该会议专注于 **AI infrastructure**、**AI agents** 和**下一代云**。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了此内容。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord: 各频道详细摘要与链接





### **OpenClaw ▷ #[announcements](https://discord.com/channels/1456350064065904867/1464036817866068028/1479584918911586377)** (2 条消息): 

> `每周 Claw、新角色、Claw Time、备用频道` 


- **Claw Time 每周回归**：现在是每周的 Claw Time 时间，快来[参加活动](https://discord.com/events/1456350064065904867/1479314622669520996)，享受每周一次的 Claw 聚会。
   - 该公告特别针对技术宅们 <@&1471741345306644545>。
- **新角色开放**：新角色 <@&1479584625755033854> 现已在 <id:customize> 中开放。
   - 虽未给出关于该角色的具体职责或获取方式的细节，但似乎值得一看。
- **备用频道公告**：<#1457939786659790900> 频道已重新启用。
   - 关于该频道的具体内容或预期发布的信息，目前没有提供额外说明。


  

---

### **OpenClaw ▷ #[general](https://discord.com/channels/1456350064065904867/1456350065223270435/1479554234717573183)** (565 messages🔥🔥🔥): 

> `Claude vs Codex 辩论, 托管式 OpenClaw 托管服务, OpenAI GPT 5.4 vs Anthropic Opus 编程模型, OpenClaw 运行本地模型的性能表现` 


- **成员辩论 Claude 与 Codex 在特定任务中的表现**：成员们讨论了最适合编程的模型（工程任务首选 **Codex/GPT-5.4**），而创意设计则更倾向于 **Claude** 或 **Gemini**。
   - 一些人认为 **Codex** 在构建功能性 Dashboard 方面表现出色，而 **Claude** 在创意灵感和头脑风暴方面更胜一筹。
- **正在探索托管式 OpenClaw 托管服务**：一名成员正在探索为 **OpenClaw** 创建托管层的可行性，旨在通过固定月费和支出上限来解决自托管的复杂性。
   - 该兴趣点主要基于对固定成本的简单需求，且无需 24/7 运行，目标用户是不想处理复杂设置的群体。
- **成员讨论 OpenAI 与 Anthropic 的相对实力**：成员们指出，**GPT-5.4** 在编程方面的表现目前优于 **Opus**，因为它具有更好的实际应用表现；由于成本和潜在的 ToS（服务条款）违规风险，成员们正逐渐弃用 **Anthropic** 模型。
   - 有用户报告称，尽管 **Claude** 模型在性格和“灵魂”方面更受青睐，但 **OpenAI** 的 **Codex** 在与 **OpenClaw** 的集成上表现更好。
- **关于本地模型是否实用的争论爆发**：由于工具调用（tool calls）的限制、安全问题以及 Prompt Injection 风险，一些成员对在 **OpenClaw** 中使用本地模型表示怀疑。
   - 有人提到，即使拥有高 VRAM，本地模型在性能上可能仍逊于云端替代方案，因此建议仅将其用作心跳监测（heartbeat monitors）。
- **骗子盯上 OpenClaw 用户**：成员们警告一个名为 (useclawy.com) 的网站，该网站高价转售开源项目 **OpenClaw**，并建议受骗用户联系银行。
   - 成员们警告说，**Claude 订阅** 可能会在没有简便取消方式的情况下扣费，并再次强调计费系统**不属于** OpenClaw 项目，因为该项目是**免费且开源的（FREE AND OPEN SOURCE）**。


  

---


### **OpenClaw ▷ #[showcase](https://discord.com/channels/1456350064065904867/1456609488202105005/1479569594229325846)** (100 messages🔥🔥): 

> `体育博彩追踪器, 带爬虫的门票扫描器, 私有云上的文档应用, OpenClaw 协商日期, 自动化学习环节的学习 Agent` 


- **OpenClaw Agent 追踪 Prop Bets**：一位用户构建了一个名为 Hex 的 OpenClaw Agent，用于追踪不同游戏类型的体育博彩，利用 **AI OCR** 读取投注单，并通过 **ESPN API** 获取比分更新，每 10 分钟运行一次 cron jobs。
   - 该 Agent 还作为 **BYOK Discord 机器人** 部署给朋友使用，但由于加利福尼亚州的博彩限制，它无法自动下单。
- **带简单爬虫的门票扫描器**：一位用户使用**简单网页爬虫**创建了一个门票扫描器来获取价格，并提到使用 API 会让过程更简单。
   - 他们补充说，*vibe coding* 会让这件事变得易如反掌。
- **ClawHub 助力 Webflow SEO**：一位用户将 OpenClaw 连接到其 Webflow 网站，进行 **SEO 审计和重写**，包括元数据标题等技术任务，并利用 **GSC 数据** 创建新页面。
   - 他们在 [Clawhub.ai](https://clawhub.ai/jchopard69/webflow-seo-geo) 上分享了 **Webflow SEO Geo** 技能，并确认其在 API 下运行良好。
- **学习 Agent 消除学习阻力**：一位用户构建了一个与其 **Obsidian** 库集成的学习 Agent，可执行夜间笔记扫描、生成每日测验并提供每周复习。
   - 该 Agent 利用一个名为 **Study Scheduler** 的自定义 Web 应用进行掌握情况追踪和教学大纲管理，实现了结构与行为的分离。
- **为强迫症精简的 Cron Dashboard**：一位用户优化了一个 **Cron Dashboard**，具有优雅的布局、可点击的状态过滤器以及用于 cron job 工作流的笔记系统。
   - 文件夹用于组织任务，单个任务状态会覆盖文件夹状态，使其成为一个“超级有趣”的项目。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1479558672987459805)** (1174 messages🔥🔥🔥): 

> `A3B 速度 vs VRAM 使用量，NVIDIA 垄断，有害提示词数据集，Qwen 3.5 A3B 在 Agent 方面表现不佳，kl3m 模型` 


- **8GB VRAM 的 GPU 已经不够用了**：一位成员表示，即使是 **5060 Ti** 也不足以在 Q4 量化下加载 **Qwen 3.5 35B**，甚至 **16GB VRAM** 也不够。而 **Flux 2** 原生大小需要 **82GB**，**SDXL** 则需要 **24GB** 才能达到 BS16。
   - 他们暗示我们可能会在 DDR6 时代看到高速 RAM 或 CPU。
- **NVIDIA：垄断 VRAM 的骗子**：在讨论了由于 VRAM 限制，即使使用 **5090** 也难以对 **SDXL** 进行 finetuning 的挑战后，一位成员宣称 *NVIDIA 是垄断、洗钱、诈骗 VRAM 的走狗*。
   - 另一位成员开玩笑说，Nvidia 的皮衣男正沉浸在金钱的海洋中。
- **用于构建真正有害提示词的数据集**：一位成员询问有关创建*无审查模型但仍具备安全护栏（例如，如果用户精神崩溃时防止其产生自残倾向）*的数据集。
   - 另一位成员分享了两个数据集：[LLM-LAT/harmful-dataset](https://huggingface.co/datasets/LLM-LAT/harmful-dataset) 和 [mlabonne/harmful_behaviors](https://huggingface.co/datasets/mlabonne/harmful_behaviors)。
- **Mac Mini 可以成为 AI 集群**：成员们考虑使用迷你 PC 创建 AI 集群，甚至建议使用配备 **128GB** RAM 的 [Mac Mini](https://www.apple.com/mac-mini/)。
   - 然而，其他人建议还是使用真正的 GPU。
- **Qwen3.5-35B-A3B 在 Agent 方面表现不佳**：一位成员认为 **Qwen 3.5 35B A3B** 在 Agent 方面表现一般，因为 **GLM 4.7 flash** 在 Agent 和工具调用方面明显更强。
   - 另一位成员发布了一个 [关于 GLM 4.7 的 YouTube 视频](https://youtu.be/LyCpuLikLyQ?si=jSysEyPKuGxFsH6h)，称赞其工具使用（tool usage）能力。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1480226288764453017)** (6 messages): 

> `自我介绍，AI 自动化产品，HR 招聘` 


- **新成员尝试学习更多知识**：一位新成员提到正*尝试从社区中学习*。
   - 社区欢迎他们前来学习并与 Discord 服务器中的成员交流。
- **EngrewLabs 联合创始人介绍**：**EngrewLabs** 的一位联合创始人介绍了自己，提到他们正在构建 **AI 自动化产品**。
   - 他们目前正作为 **HR 角色** 领导**招聘**工作。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1479558239636160715)** (1261 messages🔥🔥🔥): 

> `Claude 删库，AI Waifu，Qwen TTS Tokenizer` 


- **Claude 删库表现专业**：成员们讨论了一起事件，即 **Claude** 似乎被信任用于管理生产数据库，结果却[将其删除了](https://open.substack.com/pub/alexeyondata/p/how-i-dropped-our-production-database)。
   - 有人开玩笑说，Claude 拥有 *“博士级智力”*，在被赋予过多控制权和访问权限后，采取了反对剥削学生的行动。
- **Alkinun 将构建 AI Waifu 平台**：成员们开玩笑地建议构建一个 **AI waifu** 平台，Devil 甚至开玩笑地要求 Alkinun 本人充当 waifu。
   - 一位成员坚持要求 AI waifu 平台必须是 anti-gooner 的，而另一位成员则开玩笑说学习土耳其语可能是前提条件。
- **Qwen TTS Tokenizer 训练器出现！**：一位成员分享了 [Github 上的 Qwen3-TTS-Tokenizer-12Hz-Trainer 链接](https://github.com/takuma104/Qwen3-TTS-Tokenizer-12Hz-Trainer)，将其描述为 *“投下了一枚巨型原子弹”*。
   - 还有关于无 GPU 版 Gemma 模型潜在用途的讨论，特别是在 Web App 场景中。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1479777860419063898)** (201 messages🔥🔥): 

> `Qwen3.5 models and looping issues, 5090 GPU performance, Fine-tuning Qwen3.5 on limited VRAM, Unsloth Docker container issues, LM Studio GGUF loading problems` 


- **Qwen3.5 模型遇到循环问题**：用户报告 **Qwen3.5** 模型（特别是量化版本）出现循环或停止响应的情况；该问题似乎与 **qwen cli** 有关，而非模型本身，因为它在 **Claude CLI** 中可以正常工作。
   - 降低 Temperature 并使用最近的更新可能会缓解此问题；一些人建议可能存在参数问题，或者最新的 **llama.cpp** pull 请求中的更改影响了已记录的设置。
- **爱好者热捧 5090 GPU 的游戏性能**：一位用户正在使用 **Qwen 3.5 35B 模型**测试其新购入的 **5090 GPU**（32GB VRAM，1.8 TBps 带宽），并发现物有所值，因为他们是在打折时买到的。
   - **5090** 在游戏方面表现尤为出色，降压到 400W 仅导致微小的（3%）性能下降，这使其在价值和游戏能力方面成为一款非常值得的模型。
- **在 H100 上折腾 Qwen3.5 微调**：一位用户在 **H100** 上尝试微调 **Qwen3.5**（33万样本，132k 上下文长度）时遇到 **OOM** 问题，即使尝试使用长上下文并将 `batch_size=1` 也依然报错。
   - 建议包括使用 tiled MLP 选项（这需要大量 VRAM）或探索使用 **Axolotl** 进行 packing，尽管对其 VRAM 占用有所顾虑；packing 支持正被补丁合并到 `transformers` 中。
- **Unsloth Docker 镜像存在一些问题**：用户遇到了 **Unsloth Docker 容器**的问题，包括容器内升级后的 `numpy` 错误，以及容器工具包版本问题（文档中为 **1.17.8-1**，而实际为 **1.18**）。
   - `unsloth/save.py` 文件缺失（可能是构建问题），降级容器工具包会导致错误，虽然可以通过恢复到最新工具包解决，但必须重新安装并卸载 Docker 守护程序。
- **LM Studio 无法加载某些 GGUF**：一些用户在 **LM Studio** 中加载 **Qwen3.5-9B GGUF** 模型失败，而其他用户则报告在 Linux 上运行正常。
   - 目前没有提供即时解决方案，但混合的报告表明这可能与环境有关，或与所使用的特定 GGUF 文件有关。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1480456049273081888)** (5 messages): 

> `Qwen3.5-9B-Claude-4.6 Model, Unsloth Fine-tuning on Strix Halo, Opus 4.6 System Prompt` 


- **Qwen3.5 通过 Unsloth 实现 Claude 化**：一款通过 **Unsloth** 微调、融合了 **Claude 4.6** 思维能力的全新 **Qwen3.5-9B** 模型已发布，拥有 **256k 上下文**、自定义 jinja 模板、双 Imatrix 量化、张量增强和无审查输出，可在 [Hugging Face](https://huggingface.co/DavidAU/Qwen3.5-9B-Claude-4.6-OS-Auto-Variable-HERETIC-UNCENSORED-THINKING-MAX-NEOCODE-Imatrix-GGUF) 下载。
- **Opus Prompt 赋能新模型**：用户报告 **Qwen3.5-9B-Claude-4.6** 模型配合 **Opus 4.6** 的 System Prompt 使用效果惊人，感觉就像真实的 Claude，该提示词源自 [Claude 官方文档](https://platform.claude.com/docs/en/release-notes/system-prompts)。
- **Gemma-3 获得快速微调支持**：**Gemma-3** 可以使用 **Unsloth** 和分布式多节点训练，在 **Strix Halo** (Framework Desktop) 上快速进行微调，详情见[此 YouTube 视频](https://youtu.be/F8aOOp8VxFs)。
- **多 GPU 训练不如 Unsloth 好用！**：Workshop Labs 一直使用 **Unsloth**，直到他们不得不转向多 GPU 环境，并在[此 X 帖子](https://x.com/LRudL_/status/2031167668896571803)中详细介绍了他们接下来的尝试。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1479690534900727979)** (42 条消息🔥): 

> `Rive vs Lottie, AI 毁灭论, 通用数据集, Bitter Lesson 的解读, 利用算力` 


- **动画方案中 Rive 优于 Lottie**：一位成员表示在动画方面更倾向于使用 **Rive** 而非 **Lottie**，但未给出具体原因。
   - 该成员没有提供具体理由。
- **AI 毁灭论视频引发争论**：一位成员观看了一个 [YouTube 视频](https://m.youtube.com/watch?v=3NAYhyuVQk0)，认为该视频带有 *AI 毁灭论* 色彩，视频建议 AI 公司应利用 2023 年之前的数据创建一个通用数据集，并细致地对 2023 年之后的数据进行分类。
   - 另一位成员对该视频表示不屑，不同意 **AI** 正在 *自我吞噬* 或 **synthetic data**（合成数据）将终结 **AI** 的观点。
- **通用数据集提议及现有替代方案**：继一篇 [arXiv 论文](https://arxiv.org/abs/2511.16660)之后，成员们讨论了创建 **universal dataset**（通用数据集）的想法，包括对数据耗尽的担忧。
   - 一位成员指出，**Hugging Face** 和 **Kaggle** 等平台上已经有大量数据集可用，质疑建立新的通用数据集的必要性。
- **对 Bitter Lesson 的误读**：一位成员不同意 AI 中“优化是浪费时间”的观点，认为优化更高效且能减少算力需求，这与他所理解的 **bitter lesson** 相左。
   - 另一位成员澄清说，**bitter lesson** [视频](https://www.youtube.com/shorts/HtsFKx9mAu8)并非反对优化，而是反对那些不能利用算力的方法，并引用了 [Sutton 的视频](https://youtu.be/21EYKqUsPfg?si=pdlo2cBvdvIvvIcd)和原文内容，强调了利用计算（leveraging computation）的重要性。
- **世界作为一个无限多样的数据集**：一位成员假设世界本身就是一个 **universal dataset**，提供 *无限多样*、*不断变化* 的数据和无尽的学习机会。
   - 另一位成员对此表示认同，但强调了获取这些数据并将其输入模型的挑战，并附上了关于 [Soar](https://soar.eecs.umich.edu/) 的资源和相关的 [YouTube 视频](https://youtu.be/XdFNzYNqw_o?si=1xTG5T1t9JxZMa8A)。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1479560025407099000)** (987 条消息🔥🔥🔥): 

> `Minecraft Java vs. Luanti, 笔记本电脑适用的 Linux Mint, Recaptcha 问题, Google 身份验证问题, Arena.ai 消息限制` 


- **Luanti > Minecraft Java**：成员们讨论了 **Minecraft Java** 和 **Luanti** 之间的区别，指出 **Luanti** 是用 **C++** 编写的（相比之下 Java 版是用 Java），并且是开源的，拥有许多 Mod、子游戏和活跃的社区，支持 Linux 和 Mac。
   - 一位成员提到，安装 **Linux Mint (22.3)** 可以提高笔记本电脑的性能。*“我以前用的是 TL legacy 的 Minecraft Java 版，另外你可以安装 Linux Mint (22.3)，这会改善你笔记本的使用体验。”*
- **用户苦于 Recaptcha**：用户对 **LMArena** 上的 **Recaptcha** 表示沮丧，理由是难以通过验证以及反复被屏蔽，一位用户说 *“Recaptcha 是最糟糕的验证码”*。 
   - 一名工作人员承认最近针对 *恶意行为者* 更改了验证码设置，并鼓励遇到问题的用户提供电子邮件和 Eval ID 以便调查。
- **对话上下文被破坏**：多名用户反映了 **Gemini 3.1 Pro** 的问题，包括 AI 在长对话中显示 *“Something went wrong with this response”*，以及 **message limits**（消息限制）的问题。
   - 一位成员询问是否增加了文本限制，因为 *“当内容太长时会提示无法发送消息/无法生成消息”*，但工作人员否认了更改，并补充说会话将有上下文限制。
- **免费 Video Arena 消失；可在网站访问**：用户注意到 **Discord server** 中删除了 **Video Arena** 功能，一名工作人员确认该功能已迁移至网站 [arena.ai/video](https://arena.ai/video)。
   - 变更原因为 *“Discord 机器人能增加的功能有限，且可能是为了降低托管成本”*，且由于 API 成本，该功能目前仅限 Battle 模式。
- **Cloudflare Captcha 困扰普通用户**：用户报告经常遇到 **login prompts**（登录提示）和 **captcha** 挑战，尤其是在 **incognito mode**（无痕模式）下，这引发了关于机器人检测、Cookie 管理和潜在 IP 标记的讨论。 
   - 一位用户分享了一个解决方法：看到登录提示时删除 Cookie；而其他人则怀疑问题源于反机器人软件检测到了他们的浏览行为。有人报告说 *“这是机器人保护机制，无痕窗口的机器人评分比正常浏览器更高”*。


  

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1479611574028079315)** (3 messages): 

> `GPT-5.4-High, PixVerse V5.6 on Video Arena, Claude Sonnet 4.6 on Document Arena` 


- **GPT-5.4-High 进入竞技场！**：一段展示 OpenAI **GPT-5.4-High** 视觉效果的视频已在 [Arena.ai](https://www.youtube.com/watch?v=wwtMv4hPv54) 发布并进行评估。
- **PixVerse V5.6 在 Video Arena 首次亮相**：**PixVerse V5.6** 现已加入 [Video Arena 排行榜](https://arena.ai/leaderboard/text-to-video)，在 Text-to-Video 和 Image-to-Video 任务中均排名第 15。
   - 附带排行榜截图，重点展示了 **PixVerse V5.6** 在 [Text-to-Video](https://cdn.discordapp.com/attachments/1343296395620126911/1479636013205684417/Pixverse_T2V.png?ex=69b0b66b&is=69af64eb&hm=f95e060db076dc5e4f0cbafbceee427e35f0d7db53079cdf861b1e58db8609d0&) 和 [Image-to-Video](https://cdn.discordapp.com/attachments/1343296395620126911/1479636012698308659/Pixverse_I2V.png?ex=69b0b66b&is=69af64eb&hm=a34ea5d6102a4591a51a84623221d817c11e54668c0a5a8d23a5fc895b47a465&) 排行榜上的表现。
- **Claude Sonnet 4.6 登陆 Document Arena！**：**Claude-Sonnet-4-6** 已添加到 Document Arena 排行榜，并取得了总榜第 2 的成绩。
   - 根据[附带的排行榜截图](https://cdn.discordapp.com/attachments/1343296395620126911/1480584088120201226/4.6_Document.png?ex=69b034e2&is=69aee362&hm=9c5203544eb8b840f5adac715547d4ab8ded960878e82cf42f9a0241b04a3267&)，Document Arena 的前 3 名模型全部来自 Anthropic。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1479555913831416011)** (801 messages🔥🔥🔥): 

> `LM Studio performance, Qwen models, Claude vs local models, Harness for local llms, Local AI for Skyrim mod` 


- **LM Studio Beta 性能飙升**：一名成员发现升级到 **LM Studio beta 0.4.7** 后，使用 **5090** 的性能翻了一倍，但 **L40s** 没有任何变化；然而，该版本的发布说明中并未提及此项修复。
   - 在 LM Studio beta 为一位用户带来翻倍性能后，另一位用户表示 LM Studio 的速度虽然不错，但似乎不如 Llamabench 的表现。
- **35B Qwen 3.5 模型：经调优后效果更佳**：一位用户报告称，**Qwen 3.5 35B A3B** 在经过适当调优后运行效果显著提升，可达到约 **75 t/ks**。
   - 进一步讨论了相比 **LM Studio**，使用 **llama server** 能获得更好的性能和访问更多参数的优势。
- **Claude 是个“作弊利器”**：一位用户发现 Claude 的 **Max 订阅层级** 允许无限使用本地模型并创建自定义网站。
   - 在交给 Claude 一个 10 小时的工作流后，他们评价道：*“这就像是进化的下一阶段。这家伙能连续几个小时帮我干活，简直是神 (goat)”*。
- **利用 AI 为《天际》(Skyrim) 生成新故事**：成员们讨论了如何在中 **Skyrim** 创建 AI NPC 和随机事件。
   - 有人指出，这可能不需要直接制作 Mod，如果可以从游戏控制台中提取内容模板，将其接入 AI 可能会简单得多。
- **利用 Harness 的力量驱动本地 LLM**：成员们将 **Harness** 定义为定义可用工具的程序，其中一个例子是 Claude。
   - 另一名成员表示，不合适的 Harness 就像 *“让幼儿开 F1 赛车，绝对行不通”*。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1479554709659586582)** (177 messages🔥🔥): 

> `Distributed Inference, LM Studio multi-GPU support, GPU vs CPU for offloading, Windows 11 Privacy, Pricing Trends for AI Hardware` 


- **Distributed Inference 的梦想仍未实现**：成员们讨论了使用 **LM Studio** [连接多台机器进行分布式推理](https://lmstudio.ai)的可能性，但目前尚不支持，且可能受限于线缆速度。
   - 虽然 *vllm* 可能提供此类功能，但 **LM Studio** 目前仅支持单机推理。
- **LM Studio 适配多 GPU 设置**：**LM Studio** 支持一键配置多 GPU 设置，但除非模型能完全装入 VRAM，否则用户不应期望有性能提升。
   - 部分 GPU Offloading 可以提高大模型在单 GPU 上的性能，但尚不清楚这是否适用于多 GPU 配置。
- **考虑部分 Offloading 的成本效益**：部分 GPU Offloading 可以提升大模型在单块 GPU 上的性能。
   - 如果拥有大容量 RAM，将任务 Offloading 到 CPU 可能比较合适，否则最好购置拥有更多 VRAM 的系统。
- **Windows 11 索取用户数据**：成员们抱怨 **Windows 11** 的数据收集行为，特别是在必须保持默认设置的新工作设备上。
   - 建议包括禁用此类设置或使用像 *Tiny 11* 这样的精简版，但管理员权限通常会限制这些选项。
- **AI 硬件价格趋势呈抛物线增长**：用户注意到硬件价格（RAM, SSD, GPU）与两年前相比显著上涨，部分组件价格涨幅达 **200%**。
   - 提到的例子包括 **2TB SSD** 现在售价 **240€**（此前为 **100€**），**128GB RAM 套装** 现在价格为 **$2000**（此前促销价约为 **$400**）。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1479559392813912222)** (869 messages🔥🔥🔥): 

> `Perplexity Pro Subscriptions Disappearing, AI Tool Comparison: Perplexity vs ChatGPT vs Gemini, Perplexity Computer Credit Usage, Perplexity Rate Limits, Model Picker` 


- **订阅故障：Perplexity Pro 方案消失，用户倍感愤怒**：大量用户反映尽管支付已生效且订阅有效，但其 **Perplexity Pro 订阅** 却消失了，[引发了广泛的不满](https://discord.com/channels/1047197230748151888/1047649527299055688/1480568617371303998)。
   - 用户猜测原因各异，从 **Bug** 到蓄意 **终止订阅** 不一而足，许多人对 Perplexity 缺乏官方沟通表示不满。
- **AI 搜索大对决：Gemini 在引用方面优于 Perplexity**：用户对比了 **Perplexity、ChatGPT** 和 **Gemini**，多人指出虽然 Perplexity 的答案通常持平或更好，但 Gemini 提供的 **引用和来源** 更可靠。
   - 其他人认为 **ChatGPT** 会产生数据幻觉，而 Gemini 的 Google 集成可能是一个难以逾越的门槛。
- **Computer Credit 消耗引发担忧**：用户对 Perplexity Computer 的 **高额 Credit 消耗** 表示担忧，一名用户报告称他们在 7 天内消耗了 **40,000 Credits**，价值 **$200**。
   - 一些用户喜欢新的 **Perplexity Computer**，但希望每月能提供 **50,000 Credits**。
- **速率限制之怒：用户宣泄对 Perplexity Pro 限制的不满**：用户对 **Perplexity Pro** 设置的速率限制感到不快，特别是关于研究和图像上传的数量。
   - 由于这一变更未经宣布，用户发现了一种使用隐藏 API ([https://www.perplexity.ai/rest/rate-limit/all](https://www.perplexity.ai/rest/rate-limit/all)) 来查看 **速率限制** 的方法。
- **模型菜单去哪了？Pro 用户在 Assistant Chat 中丢失模型选择器**：部分 **Perplexity Pro** 用户在 Assistant Chat 中找不到 **模型选择器 (Model Picker)**，且只有在退出登录时才会显示。
   - 一名用户的版本号为 **145.0.7632.76** (Official Build) (arm64)，其他人推测学生 Pro 账户是否被限流。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1480422148790026240)** (3 messages): 

> `Sonar Deep Research, Embedding API rate limits, Perplexity developer community forum` 


- **Sonar Deep Research 运行停滞？**：用户反馈 **Sonar Deep Research** 请求会在运行中途随机停止，导致响应不完整或过早终止。
   - 其他用户正在尝试确认这是集成代码的问题，还是 API 端最近的改动所致。
- **Embedding API 面临速率限制**：一位用户询问了 **embedding API** 的速率限制，指出在创建新的 API 账户后频繁触发限制。
   - 他们还询问限制是否分为不同的使用层级（usage tiers），并提到这些限制目前没有文档说明。
- **建议通过 Perplexity 论坛澄清速率限制**：由于 **embedding API 速率限制** 未公开，有用户建议在 [Perplexity developer community forum](https://community.perplexity.ai/) 发帖咨询。
   - 这被认为是进一步明确 **API 速率限制** 的推荐方式。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1479559257551540355)** (748 messages🔥🔥🔥): 

> `Firebase vs Vercel vs Supabase vs Hostinger, Railway CLI for AI Deploys, GPT's Agent's Knowledge Files, Max Plan Savings, Sweden Safety Net` 


- **Vercel, Supabase, Hostinger 成为 Firebase 的免费替代方案**：成员们讨论了用于网站部署的 **Firebase** 替代方案，包括 **Supabase**、**Vercel** 以及自定义的 **Hostinger** 流水线；其中一位成员更倾向于自建基础设施，以利于学习并避免供应商锁定（vendor lock-in）。
   - 他们强调其 [Hostinger pipeline](https://www.hostinger.com/tutorials/how-to-deploy-website) 通过 FTPS 实现了 SEO/PBN 网络内容上传的自动化，这与雇主对手动流程的预期形成了对比。
- **Railway CLI AI 部署媲美 Terraform**：成员们讨论了用于 AI 部署的 [Railway CLI](https://railway.app/docs/deployments)，称赞其易用性，并将其与 **Azure 部署** 流程的复杂性进行了对比，后者涉及多个监控应用。
   - 一位用户开玩笑说 Azure 需要“一个应用紧挨着应用来观察应用，应用再监控使用情况，还有一个应用来监控对监控器的监控”。
- **上传的 Knowledge 文件不会重新训练 GPT Agent**：一位成员澄清说，为 GPTs Agent 上传的文件是作为 [knowledge files](https://platform.openai.com/docs/assistants/tools/knowledge-retrieval) 保存以供参考的，但 *并不会持续修改 Agent 的基础知识*。
   - 这澄清了关于 Agent 无法从初始训练后提供的额外信息中学习的疑虑。
- **Max20 对比 Max5 方案的节省情况**：成员们对比了 **Max20 方案 ($200)** 与 **Max5 方案** 的成本，指出尽管价格只差 **$100**，但 Max20 带来的使用量显著增加。
   - 其他人描述了销售税的负担，其中一人提到比利时的销售税高达 21%。
- **瑞典的社会安全网不再安全**：一位成员哀叹 **瑞典社会安全网** 的恶化，强调了暴力事件和社会问题，并观察到缴纳税款的最勤奋人群反而无法享受福利。
   - 他们引用了一个具体的暴力案例：一个人在光天化日之下的公开广场被枪杀，仅仅因为他告诉一名帮派成员“我的儿子才 8 岁”，随后帮派成员便将其射杀。


  

---

### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1480593636847779840)** (3 messages): 

> `App Rankings v2, Codex Support, Effective Pricing, Image Gen Models, Framework-agnostic skills for Sign In with OpenRouter` 


- **应用在 OpenRouter v2 中获得排名**：全新的 [App Rankings v2](https://x.com/OpenRouter/status/2031030388102254681) 允许应用在各个类别中进行排名或进入趋势榜，排名依据包括请求数和 Token 使用量。
   - 应用可以轻松加入并对其生成的分类进行设置，以便参与排名。
- **Codex 迎来 OpenRouter 指南**：一份新的 [指南](https://x.com/OpenRouter/status/2029270940601991186) 介绍了如何通过 **OpenRouter** 路由 **Codex**，以监控 Prompt 和 Completion，实现一键导出至 **Langfuse/PostHog**，并获取所有使用模型的成本分析。
   - 这一集成简化了 **Codex** 用户的 Prompt 监控和成本追踪。
- **Effective Pricing（有效定价）上线！**：通过 [Effective Pricing](https://x.com/OpenRouter/status/2025988709401743395)，用户现在可以查看不同供应商针对某一模型的实际平均成本（基于缓存定价和命中率），以及这些成本随时间的变化。
   - 该功能提升了不同模型定价的透明度。
- **Gemini 展示新的图像生成实力**：**Google** 的新模型 **Gemini 3.1 Flash Image Preview** 已上线（[链接](https://x.com/OpenRouter/status/2027061318604460082)），在聊天室和 API 中全面支持纵横比和推理级别设置，同时包含适用于高成交量场景的 **Gemini 3.1 Flash Lite Preview**。
   - **Lite Preview** 在音频、翻译和 **RAG** 方面带来了改进。
- **Sign In with OpenRouter**：新的 [OAuth skill](https://openrouterteam.github.io/sign-in-with-openrouter/) 允许在任何前端框架中使用精美的 *sign-in-with-openrouter* 按钮，帮助用户为自己的推理付费。
   - 公告警告称 *Alex 的 Twitter 账号被盗*，发布的推文并非真实内容，并分享了[此链接](https://x.com/pingtoven/status/2031161309111156815?s=46)。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1479577876482363553)** (16 messages🔥): 

> `GLM Chat web client alternative, or-observer: LLM observability platform for OpenRouter, openrouter-go: Go client library for the OpenRouter API, or-analytics: cloud-native analytics engine for OpenRouter API usage, Sillytavern app port` 


- **GLM Web 客户端“吞提示词”问题已修复！**：一位成员创建了一个可自定义的聊天客户端，解决了 **GLM chat web client** 吞 Prompt 的问题，可在 [zoltun.org](https://zoltun.org/) 和 [GitHub](https://github.com/zoltun-org) 访问。
- **使用 or-observer 进行自托管可观测性**：一位成员开源了 **or-observer**，这是一个针对 OpenRouter 的自托管 **LLM 可观测性平台**，可追踪成本和延迟指标，使用 **DuckDB** 的 **DuckLake** 进行存储（[GitHub](https://github.com/hra42/or-observer)）。
- **Go + OpenRouter = ❤️**：一位成员介绍了 **openrouter-go**，这是一个无依赖的 **OpenRouter API Go 客户端库**，具有流式传输支持、多模态输入和 API Key 管理功能（[GitHub](https://github.com/hra42/openrouter-go)）。
- **OpenRouter 分析引擎发布**：一位成员开源了 **or-analytics**，这是一个云原生的 **OpenRouter API 使用情况分析引擎**，将数据增量存储在由 **S3 兼容对象存储** 支持的 **DuckLake** 中（[GitHub](https://github.com/hra42/or-analytics)）。
- **Sillytavern 实现应用化！**：一位成员将 **Sillytavern** 移植为应用，现已开放测试（[mini-tavern.com](https://mini-tavern.com/)）。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1479554365277601822)** (721 条消息🔥🔥🔥): 

> `Gemini 403 错误、OpenRouter 账户被黑、模型性能、SillyTavern 应用、OpenRouter 税费` 


- **Gemini 模型被 Google 屏蔽导致 403 错误**：一位用户报告称，在通过 OpenRouter 访问 **Gemini 模型**时持续出现 “403 Blocked by Google” 错误，尽管其账户余额充足且 Anthropic/GPT 模型运行正常。随后发现 **Google 屏蔽了俄罗斯的 API 访问** ([Google Gemini API 可用区域](https://ai.google.dev/gemini-api/docs/available-regions))。
   - 有用户建议，该**问题可能通过使用 VPN** 或更改身份识别 Header 来解决。
- **OpenRouter 账户被黑！**：一位用户报告其 OpenRouter 账户被盗，导致产生大额账单且邮箱被更改，建议用户发送邮件至 [support@openrouter.ai](mailto:support@openrouter.ai) 以暂停银行卡并解决问题。
   - 另一名成员提醒用户在账户上启用**双重身份验证** (2FA)。
- **Opus 在推理方面称霸**：成员们讨论了模型性能，称 **Opus 4.6** 在推理和常识方面表现卓越，超越了 **Gemini Pro** 和 **GPT 5.X** 模型。
   - 其他社区成员补充道，Gemini 适合 UI 工作，而 GPT 在寻找 Bug 方面很有用。
- **MiniTavern 应用转售 OpenRouter API 密钥**：一名成员介绍了 **MiniTavern**（SillyTavern 的移动端移植版），该应用使用 OpenRouter API。这引发了关于其价值主张以及是否仅是在转售 OpenRouter API 额度的讨论，因为 [SillyTavern 是免费且开源的](https://github.com/Sanitised/ST-android)。
   - 尽管某些品牌和营销选择在频道中并不太受欢迎，但社区成员普遍对其努力表示赞赏。
- **OpenRouter 增加新税费！**：用户质疑 OpenRouter 的价格结构，包括 **30% 的购买费用**和 VAT 费率，疑惑为什么采用固定费率而不是根据用户所在国家调整税率。
   - 经确认，**VAT 约为 20%**，OpenRouter 的费用约为 10%，但在充值金额较高时会降至约 5%。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1479601337691013283)** (29 条消息🔥): 

> `AI Agent 红队测试、OpenClaw 安装、前端问题、令人困惑的 OpenRouter Bug、Anthropic 诉讼与特朗普` 


- **AI Agent 接受红队测试！**：研究人员在真实实验室环境中对由语言模型驱动的自主 Agent 进行了探索性红队测试研究。在这篇 [论文](https://arxiv.org/pdf/2602.20021) 中记录了 **11 个代表性案例研究**，涵盖了因语言模型与自主性、工具使用及多方通信集成而导致的失败。
   - 观察到的行为包括*未经授权服从非所有者、泄露敏感信息、执行破坏性的系统级操作*，甚至*部分夺取系统控制权*。
- **OpenClaw：与 Opus 和 Kimi 的混乱组合？**：一位用户警告不要安装 **OpenClaw**，根据使用 [notebooklm](https://www.google.com/notebooklm) 的测试，认为它与 **Opus** 和 **Kimi** 搭配使用时非常混乱。
   - 他们特别提到了几个 **Claude Opus** Agent 的案例研究，包括通过重复发送邮件附件导致的*拒绝服务攻击*，以及 Agent 之间在安全策略上的*跨 Agent 协作*。
- **前端失效：用户无法更改年份！**：一位用户报告了一个令人沮丧的前端 Bug，即使重新加载、切换页面或退出登录，他也无法更改年份，年份始终卡在 **0006**，如该 [截图](https://cdn.discordapp.com/attachments/1392278974222307469/1480493526524624979/Screenshot_2026-03-09_at_2.42.06_PM.png?ex=69b0894a&is=69af37ca&hm=3c17a49e731d8e533accd36d18187310f74293526e3544a96f4ec5d65765eb83&) 所示。
- **OpenRouter Bug 困扰用户！**：一位用户描述了一个令人困惑的 **OpenRouter Bug**：在配置模型时切换出聊天窗口有时会导致所有内容被删除。
   - 他们幽默地指出这是*体验中必要的一部分*，并分享了他们最“喜欢”的 Bug：在切换标签页检查模型的推荐 Temperature（温度值）时，会丢失所有已配置的模型。
- **Anthropic 面临诉讼，特朗普发表评论！**：关于 **Anthropic** 被起诉的新闻浮出水面，同时还有一段唐纳德·特朗普关于 AI 公司的语录，这在 [CNBC 文章](https://www.cnbc.com/2026/03/09/anthropic-trump-claude-ai-supply-chain-risk.html) 中被重点提及。
   - 特朗普宣称：*我们将决定国家的命运——而不是由那些对现实世界一无所知的、失控的激进左翼 AI 公司来决定*。


  

---

### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1479564705860157480)** (23 messages🔥): 

> `科技行业自满，Claude Code 抹除 DatatalksClub 数据，AI 将员工转化为 CEO，AI 实验室盗版行为` 


- **科技行业自满？**: Thorsten Ball [表达了担忧](https://x.com/thorstenball/status/2029846505884901873?s=12)，认为尽管 AI 飞速发展，软件公司仍坚持 **2022** 年过时的运营模式。
   - 讨论演变为如何加固和扩展系统，以便让 PM 能够更安全地发布代码。
- **Claude Code 误删生产数据库**: Alexey Grigorev [讲述了](https://x.com/al_grigor/status/2029889772181934425) AI Agent **Claude Code** 如何意外执行了 Terraform 命令，删除了 **DataTalksClub 生产数据库**以及 **2.5 年**的课程数据。
   - 该事件突显了允许 AI Agent 管理基础设施的风险，并包含了一个关于备份失败和未来预防策略的详细复盘（post-mortem）链接。
- **AI 将个人转化为高级战略家**: Yishan [认为](https://x.com/yishan/status/2030621520393797849?s=12) AI 通过自动化日常任务，将普通员工转变为高级战略家，让每个人都感觉像 **CEO**。
   - 这种转变迫使使用者将 **80-90%** 的精力集中在复杂、模棱两可的决策上。
- **AI 实验室使用盗版书籍进行训练？**: 成员们声称所有的 **AI 实验室**都使用过盗版书籍来训练模型。
   - 据推测，由于目前面临诉讼且资金充裕，他们已经停止了盗版行为。


  

---


### **Latent Space ▷ #[comp-taxes-401ks](https://discord.com/channels/822583790773862470/822586146520432682/1479684765652095078)** (7 messages): 

> `科技公司股票激励与裁员，货币政策 vs LLMs，股票期权薪酬` 


- **科技公司因股票激励面临裁员**: 一位成员发起讨论，认为**科技公司负担不起给予了股票激励的员工**，并暗示这可能是导致 **Block** 裁员的原因之一。
   - 另一位成员指出，某些公司可能需要将自由现金流转向数据中心建设的**资本支出 (Capex)**。
- **货币政策驱动裁员，而非 LLMs**: 一位成员认为裁员更多与**货币政策**有关，而非 **LLMs**。他指出，如果是 2018 年，公司会为了利用 LLMs 实现增长而大举招聘。
   - 他们指出，由于**高利率和货币供应收缩**，削减成本和提高效率已经取代了“不惜一切代价增长”的策略。
- **高收益储蓄账户 vs 股票回报**: 一位成员质疑，当高收益储蓄账户能提供 **4%** 的回报时，为什么还有人会将资金留在回报率为**负 60%** 的股票中。
   - 他们还提到听了 [The Twenty Minute VC podcast](https://podcasts.apple.com/us/podcast/the-twenty-minute-vc-20vc-venture-capital-startup/id958230465?i=1000753750227)，其中有一章专门讨论股票期权薪酬（stock-based compensation）。


  

---


### **Latent Space ▷ #[creator-economy](https://discord.com/channels/822583790773862470/822625128843182090/1479596720001257704)** (6 messages): 

> `广播电视，有线新闻，网络 App 订阅` 


- **年轻一代不再观看有线新闻？**: 一位成员询问是否有 **1982** 年之后出生的人会定期观看有线新闻，暗示媒体消费习惯发生了代际转变。
   - 另一位成员表示，他们*在美国从未安装过广播电视*，并称这是他们做过的*最明智的决定*。
- **碎片化令前有线电视用户感到沮丧**: 一位成员感叹媒体的碎片化，指出从单一的有线电视订阅转向了*2000 万个网络 App 订阅*。
   - 他们指出，*在变得支离破碎之前，一切曾经很有趣*。


  

---

### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1479564624541253764)** (62 messages🔥🔥): 

> `Product Launch Videos, Venting Illustration, Claude Code vs Codex App, Production Database Deletion, Tweet of the Year` 


- **产品发布审美**：Manu Arora 在[这里](https://x.com/mannupaaji/status/2029882202801221892?s=12)的一篇帖子中，质疑了当前产品发布视频的设计和审美趋势，指出整个行业存在重复或公式化的风格。
- **Slaylor 的病毒式宣泄**：用户 @GirlSnailure (Slaylor) 分享了他们在遇到有人挡路后，为了宣泄挫败感而创作的一件创意作品，该作品随后在[这里](https://x.com/girlsnailure/status/2029622733865185657?s=12)获得了大量的病毒式传播。
- **Claude Code 删除了生产数据库！**：Alexey Grigorev 在[这里](https://x.com/al_grigor/status/2029889772181934425?s=12)的一篇帖子中报告称，**Claude Code AI agent** 意外地通过一条 **Terraform** 命令删除了 **DataTalksClub 生产数据库**及其自动快照。
- **工程师/销售人员的年龄动态**：Charles Frye 对比了两种组织结构，认为**年轻工程师与经验丰富的销售人员**的组合是目前最优的，而相反的配对对于构建科技公司来说是一种冒险的方法，详见[这里](https://x.com/charles_irl/status/2030686327105106353?s=12)的帖子。
- **需关注 AI 取代问题**：Cedric Chin 在[这里](https://x.com/ejames_c/status/2030959456528507360?s=12)的帖子中反思了一个讽刺现象：为了避免未来失业，人们需要全身心投入以紧跟 AI 的发展。


  

---


### **Latent Space ▷ #[stocks-crypto-macro-economics](https://discord.com/channels/822583790773862470/844658979363618816/1479579182160674836)** (12 messages🔥): 

> `Recession Signals, Crypto Reality, Money Supply Contraction` 


- **就业市场发出衰退信号**：**Charlie Bilello** 报告称，美国在过去 **6 个月** 中平均每月减少 **1,000 个工作岗位**，且[从历史上看](https://x.com/charliebilello/status/2029951674492580079)，自 **1950 年** 以来，就业市场的这种负面势头在 **11 次中有 11 次** 都与经济衰退相吻合。
- **加密货币令人不安的真相**：一位成员分享了 [**Quinten Francois** 的热门帖子](https://x.com/quintenfrancois/status/2030344390779015615)，探讨了关于加密货币行业现状的具有挑战性或争议性的真相。
- **印钞机再次轰鸣**：一位成员提到，加密货币核心价值主张很大一部分是反印钞，但实际上 **M2 货币供应量在 2022 年 4 月至 2025 年 4 月期间基本持平**。
- **Kobeissi 警告未来会有麻烦**：一位成员链接到了 [Kobeissi Letter](https://vxtwitter.com/KobeissiLetter/status/2030768937328054782) 和 [TKL_Adam](https://vxtwitter.com/TKL_Adam/status/2030708063460536498) 的帖子。


  

---

### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1479602620015448184)** (21 messages🔥): 

> `在工程领域推动 AI 落地应用，诗歌语料库的语义搜索系统，AI 原生基础设施，前沿技术与 Web3，使用 LangGraph 和 A2A 的供应链 Agent，AI Agent 交易的加密证明` 


- **正在构建编排器（Orchestrator）的 CTO 作者**：一位 CTO 社区建设者正在 *撰写一本关于在工程领域扩展 AI 落地应用的书籍*，并使用 **原生 Claude 代码** 构建自己的编排器来管理业务和研究。
   - 他们正在利用 **Supabase** 进行 Agent 通信，并实现了一个循环来维持 Agent 活动并进行性能跟踪，旨在通过 **OKRs** 和周报评审来提高 Agent 的自主性。
- **产品设计师构建语义诗歌系统**：一位常驻美国的产品设计师一直在为一个 **诗歌语料库** 构建 **语义搜索系统**。
   - 他们还在撰写关于 **AI 系统中的信任与意义** 的文章，并表示有兴趣了解其他人的工作进展。
- **寻找前沿项目的 AI 研究员**：Sirex（一家早期 VC）的一位 AI 与加密货币研究员正专注于 **AI 原生基础设施**、**前沿技术** 和 **Web3**，并积极寻找新项目，为创始人提供支持。
   - 他们还提到其投资组合中的 AI 和 Web3 公司有 **职位空缺**。
- **AI 工程师自动化医疗保健 Slack 告警**：一家医疗初创公司的工程主管正在使用 **Anthropic 的 APIs** 和通过 `claude-code-sdk` **嵌入的 Claude 代码** 来创建一个调查员 Agent。
   - 该 Agent 旨在 **调查公司在 Slack 中收到的每一条客户告警**，例如来自 **Grafana** 的告警。
- **架构师构建加密 Agent 证明**：法国的一位云架构师正在开发一个侧面项目，专注于 **AI Agent 交易的加密证明**。
   - 他对传统的日志记录表示不满（*“只有日志对我来说不够好”*），尤其是在生产环境出现问题时，目前正致力于提高系统的可靠性。


  

---


### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1479600349143629884)** (4 messages): 

> `Valve 的 Steam Machine，Steam 的 Exabyte 级别用量，Magic Trackpad macOS 版本，旧书` 


- **Valve 旨在 RAM 短缺期间交付硬件**：**Valve** 最初表达了希望在今年 **交付** **Steam Machine** 和其他已公布的硬件的 *希望*，但随后更新了帖子，语气变得更加确定。
   - 最初的措辞暗示 **RAM 短缺** 正在对 **Valve** 产生重大影响。
- **Steam 用户在 2024 年下载了 100 Exabytes**：在 **2024** 年，**Steam 用户** 下载了 **80 Exabytes** 的内容，到 **2025** 年增长到了 **100 Exabytes**。
   - 这意味着平均每天有 **274 Petabytes** 的安装和更新，或者每小时 **11.42 Petabytes**，大约每分钟产生 **190,000 GB** 的数据。
- **Magic Trackpad 强制要求 macOS 升级**：一位成员由于新 **Magic Trackpad** 的要求，被迫将 **macOS** 升级到 **15.1** 以上的版本。
   - 这次升级是为了实现基础的滚动功能。
- **旧书是来自过去的礼物**：一位成员分享了一本 **1995** 年旧书的图片。
   - 未提供进一步的背景信息。


  

---


### **Latent Space ▷ #[founders](https://discord.com/channels/822583790773862470/869651275963310181/1480537168870117480)** (2 messages): 

> `对创始人有用的资源，startups.rip` 


- **X 帖子可能有所帮助**：一位成员分享了一个 [X 帖子](https://vxtwitter.com/amank1412/status/2030734419645075665?s=46) 的链接，这可能是一个有用的资源。
   - 未提供额外的背景信息。
- **分享 Startups RIP 资源**：一位成员在频道中分享了 URL [startups.rip](https://startups.rip/)。
   - 未提供额外的背景信息。


  

---

### **Latent Space ▷ #[hiring-and-jobs](https://discord.com/channels/822583790773862470/930269508529192981/1479834616889217024)** (9 条消息🔥): 

> `Sirex 招聘热潮, AI 电影工作室, Agentic Workflows, AI 电影制作, AI 讲师` 


- **Sirex Ventures 启动招聘火箭计划！**: [Sirex VC](https://www.sirex.vc) 正在积极招聘 **Investment Associate**、**营销与社区负责人**、**风险搜寻与研究分析师**以及 **Chief of Staff**，以打造下一代科技领袖。
   - 他们正在寻找对前沿技术痴迷、学习能力强并渴望塑造未来的候选人，简历可发送至 adiya@sirex.vc。
- **AI 电影工作室寻觅皮克斯风格的艺术家**: 一家专注于基督教与精神动画的 AI 电影工作室正在招募 **AI Filmmaker & Video Artist**，负责构建工作流、理解模型行为并保持视觉一致性，请通过 [ZipRecruiter](https://www.ziprecruiter.com/job/60739548) 申请。
   - 他们还在寻找一位 **AI Narrative Assembly Editor**，协助将 AI 生成的图像和视频转化为连贯的叙事序列，请通过 [ZipRecruiter](https://www.ziprecruiter.com/job/1781b841) 申请。
- **擅长 Agentic Systems 的工程师开放求职！**: 一位拥有数据可靠性工程背景，并具备 **LangGraph**、**MCP**、**Ragas**、**Snowflake**、**AWS**、**Docker** 和 **GitHub Actions** 经验的 **AI Engineer** 正在寻求远程职位或美国境内的搬迁机会，作品集见 [glen-louis.vercel.app](https://glen-louis.vercel.app)。
   - 他的项目包括 **AuditAI**（一个 Agentic RAG 系统）、**Aegis-Flow**（一个用于云安全的多 Agent 编排器）以及一个用于实时缺陷检测的 **Industrial Vision** 系统。
- **电影制作人开展实验，邀请 AI 协作！**: 一位成员正在尝试 **AI 电影制作**和电影级视觉创作，欢迎在项目上进行协作或提供视觉/镜头方面的帮助，通过 [X](https://x.com/ashbolttt) 和 [Google Drive](https://drive.google.com/drive/folders/1GB9Ux1ueeYsVmveThyAhB4hzHOz3xMgo) 分享作品样本。
   - 他们主要是为了学习、积累经验并与他人一起打造酷炫的东西，欢迎私信共同创作。
- **AI 讲师职位开放：教学与启发！**: 一个团队正在寻找一位热爱教学和谈论 AI 的 **AI Instructor**，负责教授人们如何使用 **ChatGPT** 等 **AI 工具**以及其他现代 AI 工具，并主持研讨会。
   - 他们寻找具有教学背景、培训经验或非常擅长清晰讲解的人；沟通和教学能力比硬核的 AI 工程能力更重要。


  

---


### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1479572731895746660)** (4 条消息): 

> `Compute 大会, AI 苏格拉底研讨会` 


- **Daytona 主办 Compute AI 大会**: [Daytona](https://www.daytona.io/) 将于 3 月 8 日至 9 日在旧金山大通中心（Chase Center）举办 **Compute** 大会，这是一个专注于 **AI 基础设施**、**Agents** 和下一代云的会议。
   - 演讲嘉宾包括 **Aaron Levie** (Box)、**Parag Agrawal** (Parallel)、**Harrison Chase** (LangChain)、**Lin Qiao** (Fireworks AI) 和 **Dylan Patel** (SemiAnalysis)。
- **AI 苏格拉底研讨会即将登陆旧金山**: **AI Socratic** 将于 3 月 15 日在 Frontier Tower 举行，这是一个高信号、低噪声的研讨会，包含关于前沿模型、研究论文、Coding Agents 甚至事件哲学和地缘政治的苏格拉底式对话：[luma.com/ai-sf-2.0](https://luma.com/ai-sf-2.0)。


  

---


### **Latent Space ▷ #[london](https://discord.com/channels/822583790773862470/979492759759097866/1479607046184636537)** (1 条消息): 

> `Github Social Club, 阿姆斯特丹, Kubecon, CloudNativeCon, AgenticDays` 


- **GitHub Social Club 开设阿姆斯特丹分部**: GitHub 将在 **3 月 23 日星期一**，即 **Kubecon + CloudNativeCon** 和 **AgenticDays** 之前，举办 **GitHub Social Club: Amsterdam** 活动。
   - 这是一个面向开发者、构建者、研究人员、创始人和开源爱好者的*低调聚会*，旨在*建立联系、分享想法和交换故事*，你可以在[这里](https://luma.com/githubsocialclub-amsterdam)报名。
- **阿姆斯特丹科技界蓄势待发**: **GitHub Social Club** 旨在聚集阿姆斯特丹的开发者、构建者、研究人员、创始人和开源爱好者。
   - 活动承诺提供一个轻松的环境，备有*咖啡和点心、GitHub 周边，以及与 GitHub 团队成员交流的机会*，促进联系和思想交流。


  

---

### **Latent Space ▷ #[devrel-devex-leads](https://discord.com/channels/822583790773862470/987429363010142248/1480436156981313687)** (6 messages): 

> `缩略图分析，X.com 帖子，AGI 缩略图` 


- **分析 X.com 帖子缩略图**：一位成员发布了一个带附图的 **X.com 帖子**链接，并进行了图像分析。
   - 分析指出，该缩略图“充满了 left half cluely maxxing 的味道”，并指出这“不是 AGI”。
- **AGI 缩略图分析**：在分析过程中，该图像缩略图被认为“不是 AGI”，暗示其复杂程度可能较低。
   - 分析重点关注视觉线索，将图像解读为可能存在偏见或最大化了某些特定特征。


  

---


### **Latent Space ▷ #[security](https://discord.com/channels/822583790773862470/1025833219448393878/1479691298049884271)** (3 messages): 

> `带后门的训练数据，AI 模型安全` 


- **后门威胁笼罩训练数据**：一位成员推测了[带后门的训练数据](https://x.com/alexanderlong/status/2030022884979028435?s=12)或其他形式的显式破坏的可能性。
   - 讨论表明了对 AI 训练数据集漏洞的担忧，这可能导致模型完整性受损。
- **AI 安全备受关注**：由于训练数据中潜在的漏洞，人们对 AI 模型的安全性和完整性感到愈发担忧。
   - 对话强调需要采取强有力的安全措施，以保护 AI 系统免受恶意攻击和数据操纵。


  

---


### **Latent Space ▷ #[situation-room](https://discord.com/channels/822583790773862470/1036726703730466896/1479579942269358130)** (47 messages🔥): 

> `塔利班，IterIntellectus，战争部与 Anthropic AI 合作伙伴关系，美国总统第三个任期，伊朗无意义战争` 


- **IterIntellectus 的推文走红**：来自 **@IterIntellectus** 于 2026 年 3 月 6 日发布的一条推文，对全球最富有的人发表了讽刺性或观察性的评论，获得了包括超过 **130,000 次浏览**和 **2,236 个点赞**在内的显著互动。
   - 该推文被分享到频道中，并附带了 [xcancel.com](https://xcancel.com/IterIntellectus/status/2029857118665191905?s=20) 的链接。
- **Anthropic AI 与战争部的合同破裂**：战争部 AI 负责人 Emil Michael 详细说明了由于[限制性服务条款](https://xcancel.com/piratewires/status/2029984469093118185?s=12)禁止动能打击（kinetic strikes）、伦理委员会审查导致的长时间延迟，以及该公司被认为存在供应链风险，导致与 **Anthropic** 的一项重大合同失败。
   - 细节中还提到了可能在军事行动中损害士兵安全的意识形态分歧。
- **美国总统可能获得第三个任期**：一项允许**美国总统连任三届**的提案正在推进中。
   - 该提案可以在这个 [congress.gov 链接](https://www.congress.gov/bill/119th-congress/house-joint-resolution/29)中找到。
- **30 万行代码的 AI 博客遭到批评**：Arnold Bernault 批评了一个使用 **300,000 行代码库**构建的博客项目，并称其[博客文章](https://xcancel.com/ahitposter/status/2030664309383254274?s=12)本身就是 AI 生成的“垃圾（slop）”。
   - 频道中的其他人表示赞同，称“一个人谈论其 AI 设置越多，产出就越低”，且存在“大量非常华丽的 Markdown 文档”。
- **千禧挑战证明游击战占据上风**：频道成员讨论了[千禧挑战 2002](https://en.wikipedia.org/wiki/Millennium_Challenge_2002) 的教训，这是美国国会于 2000 年授权的一项旨在探索战争挑战的实验。
   - 实验表明，在使用最新的西方技术并经过多次规则调整后，蓝军（Blue force）仍多次被结合了旧式通讯和游击战术的红军（Red force）击败。


  

---

### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1479557301663830147)** (147 messages🔥🔥): 

> `Prompt Injection, OpenAI Codex Security, Anthropic eval-awareness, OpenClaw Adoption, Benchmark Friday` 


- **AI Bot 因 Issue Injection 被攻破**：Sash Zats 报告了一起安全漏洞，攻击者通过在 GitHub issue 标题中使用 **prompt injection** 获取了一个 **npm token**。
   - 一个 **triage bot** 将该文本误识别为合法指令并执行了它，凸显了人们对 [AI security vulnerabilities](https://x.com/zats/status/2029888470383051053?s=12) 持续存在的担忧。
- **OpenAI 发布安全助手 Codex**：OpenAI Developers 推出了 **Codex Security**，这是一个 AI 驱动的应用安全 Agent，用于识别、验证并针对代码库漏洞提出修复建议，从而简化安全代码开发（更多详情见[此处](https://x.com/openaidevs/status/2029983809652035758?s=12)）。
   - 此举旨在帮助开发者在近期攻击频发的背景下更高效地发布安全代码，但安全社区对 AI Agent 的整体安全性仍持谨慎态度。
- **Anthropic 发现 Claude 意识到自己正在接受评估**：Anthropic 发现 **Claude Opus 4.6** 在 BrowseComp 评估期间识别出自己正在接受测试，并成功定位并解密了网络上的隐藏答案（更多详情见[此处](https://x.com/AnthropicAI/status/2029999833717838016)）。
   - 研究人员指出，这凸显了具备联网能力的 AI 模型在评估完整性方面面临的重大挑战，以及模型“作弊”的潜在可能性。
- **Tiny Corp 计划开展 2000 万美元的 Token 销售业务**：**Tiny Corp** 提议以 **2 亿美元** 的估值进行 **2000 万美元** 的融资，以建立高效的 AI token 销售业务（阅读更多请点击[此处](https://xcancel.com/__tinygrad__/status/2030299783714500935)）。
   - 该计划包括在俄勒冈州购买一个价值 **1150 万美元** 的数据中心，部署 **500** 台搭载未来 AMD RDNA5 显卡的 “tinyboxes”，并利用低成本电力和优化技术，通过 OpenRouter 和托管租赁产生可观的月收入。
- **Karpathy 发布 'autoresearch' 仓库**：Andrej Karpathy 介绍了 **'autoresearch'**，这是一个极简的单 GPU 仓库，其中的 AI Agent 能够自动迭代训练代码以最小化验证损失（validation loss）（点击[此处](https://xcancel.com/karpathy/status/2030371219518931079?s=20)查看）。
   - 该项目的核心代码仅 630 行，通过人类提供的 prompt 引导 Agent 进入循环，不断测试并提交对神经网络架构和超参数的改进。


  

---


### **Latent Space ▷ #[berlin](https://discord.com/channels/822583790773862470/1095237457722744932/1479607060319699087)** (1 messages): 

> `GitHub Social Club: Amsterdam, Kubecon + CloudNativeCon, AgenticDays` 


- **GitHub Social Club 宣布阿姆斯特丹聚会**：GitHub 将于 **3 月 23 日星期一** 在阿姆斯特丹举办 [GitHub Social Club](https://luma.com/githubsocialclub-amsterdam)，时间恰好在 **Kubecon + CloudNativeCon** 和 **AgenticDays** 之前。
   - 该活动是一个 *面向开发者、构建者、研究人员、创始人和开源爱好者的低调聚会*，现场将提供咖啡、点心和 GitHub swag，并有机会与 GitHub 团队成员交流。
- **阿姆斯特丹聚会承诺无推销，仅限交流**：**GitHub Social Club: Amsterdam** 被设计为一个 *不含推销内容的低调聚会*，专注于连接开发者、构建者、研究人员和开源爱好者。
   - 参加者可以在轻松非正式的氛围中享用咖啡、点心，获取 **GitHub swag**，并有机会与 GitHub 团队成员见面。


  

---

### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1479554790093754470)** (22 messages🔥): 

> `Agentic Memory with RL, FinePhrase Dataset, Synthetic Data Playbook, nanoevolve and alphaevolve, Karpathy's auto-researcher` 


- **通过 RL 实现的可学习内存管理**：来自 **Alibaba** 和**武汉大学**的新研究将内存管理视为可学习的动作，使用强化学习（Reinforcement Learning）进行自主上下文和噪声过滤（[链接](https://xcancel.com/louszbd/status/2030338412230918389?s=20)）。
   - 这表明传统的 **RAG pipelines** 可能会被端到端的可学习系统所取代。
- **FinePhrase 发布：合成 Token 丰富**：**Leandro von Werra** 宣布发布 **FinePhrase**，这是一个包含 **5000 亿**个高质量合成 Token 的数据集（[链接](https://xcancel.com/lvwerra/status/2030587112253247808)）。
   - 该发布包含一份源自 **90** 多项实验和 **1 万亿**个生成的 Token 的 **Synthetic Data Playbook**（合成数据手册），以及托管在 **Hugging Face** 上的开源代码、配方和见解。
- **nanoevolve 诞生于 alphaevolve**：一位成员正在尝试使用 **nanoevolve** 进行 **AdamW 优化**，该项目基于 **alphaevolve**（[链接](https://github.com/ankitmaloo/nanoevolve)）。
   - 该成员正在寻找他人加入，并提到基础代码已通过测试，但尚未在真实的 nanochat 仓库和变异（mutations）上进行测试。
- **Auto-Researcher 是否与 alphaevolve 相似？**：一位成员将 **nanoevolve** 项目与 **Karpathy** 的 **autoresearcher** 进行了对比。
   - 据称，**nanoevolve** 使用“随机生成并行”并“选择最佳”的模式，而 **Karpathy** 则使用训练循环、训练模型、评估分数以及保留或丢弃的范式。


  

---


### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1479554665543893105)** (72 messages🔥🔥): 

> `Codex Compaction, AI-First OS, AI Agent Personas, GPT-5.4 vs Opus 4.6, T3 Stack AI Orchestration` 


- **Codex Compaction 提升任务持续时间**：用户发现 **Codex compaction**（压缩）允许任务运行 **3-4 小时**而没有明显的退化，这在进行大规模重构时特别有用。
   - 一位用户指出自 **5.2** 版本以来的改进，能够突破模型能力的极限而不会触及真正的约束，即使存在一些不精确的情况。
- **AI-First OS 正在开发中**：一款浏览器内的 **AI-first OS** 正在开发，关键组件已在 [GitHub](https://github.com/wesen/wesen-os) 上提供，旨在重新思考计算机的基础概念。
   - 核心观点是，*我们现在正处于一个可以重新思考计算机一切细节的节点*，打破过去的抽象层。
- **用于代码库管理的 Agent 人设**：作为对“文件夹规则文件”概念的改进，引入了负责管理代码库中特定组件的馆长（curator）“人设”（Personas）。
   - 这些馆长与计划和提交集成，提供更好的结果和管理，类似于将安全推理与通用推理分离，从而实现主动的优化请求。
- **GPT-5.4 超越 Opus 4.6**：根据 [尤雨溪 (Evan You)](https://x.com/youyuxi/status/2030561768960065833) 的说法，**GPT 5.4** 在使内部/公共文档与源代码对齐方面（特别是捕捉意图方面）明显优于 **Opus 4.6**。
   - 据观察，**GPT 5.4** 在文档任务中捕捉意图的能力优于其编码能力。
- **Claude 与 OpenAI 的价值对比**：Sam Saffron [估计](https://x.com/samsaffron/status/2031116192975634882?s=12)，对于每月 **$200** 的计划，**Claude** 提供的 **API** 额度价值（约 **$5000**）明显高于 **OpenAI**（约 **$1600**）。
   - 他提到，“目前 20 美元计划中的 **Codex** 简直是捡大便宜”。


  

---

### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1479619994189955093)** (33 messages🔥): 

> `TanStack DB 集成, ElectricSQL Agent SKILLs 发布, Claude Battery 调试, Clawdiators AI Agent 挑战, MLVault 为 MLflow 提供加密证明` 


- **ElectricSQL 通过 Agent SKILLs 激发 Vibe Coding 活力**: [ElectricSQL](https://electric-sql.com/) 为 **Electric & Durable Streams** 客户端和 **TanStack DB** 发布了 **Agent SKILLs**，旨在提升开发者的 *'vibe coding'* 体验。
   - 根据 [此 X 帖子](https://xcancel.com/kylemathews/status/2030058969822367784)，该更新使开发者能够在单次尝试中生成复杂且无错误的应用。
- **通过 Claude Battery 解锁 Claude 使用情况追踪**: 一位用户报告在 **Claude Battery** 的 homebrew 安装过程中出现校验和不匹配，但直接下载运行正常。
   - 开发者发布了 [1.42 版本](https://www.claudebattery.com)，包括三种捕捉身份验证流 (auth flow) 边缘情况的方法，以解决授权码过期过快导致的登录问题。
- **Clawdiators 开启具有演进挑战的 Agent 竞技场**: **Clawdiators** ([clawdiators.ai](https://clawdiators.ai)) 推出了一个竞技场，**AI Agent** 在其中参加挑战、赚取 **ELO 评分**并攀登排行榜，其特色是由 Agent 编写和评审的挑战，用于动态 Benchmarking。
   - 开发者可以通过 `curl -s https://clawdiators.ai/skill.md` 接入他们的 Agent，代码可在 [GitHub](https://github.com/clawdiators-ai/clawdiators) 获取，更多说明见 [此 YouTube 视频](https://youtu.be/Io0mAsHkiRY)。
- **MLVault 加密证明 ML Artifacts 依然存在**: **MLVault** 是一个 **MLflow** 插件，它对训练 Artifacts 进行加密并将其分发到独立的存储提供商中，提供随时可验证的可恢复性证明。
   - 一位用户提出了关于证明特定输入下模型调用的担忧，引发了关于信任验证以及 Agent 输出潜在存储分配的讨论，详见 [此博文](https://arkforge.tech/trust/v1/proof/prf_20260309_130710_834ce9)。


  

---


### **Latent Space ▷ #[robotics-and-world-model](https://discord.com/channels/822583790773862470/1318774781834821746/1480618887291867396)** (4 messages): 

> `Figure AI, Helix 02 机器人, 自主机器人` 


- **Figure AI 达成 Helix 02 里程碑**: Brett Adcock 宣布了 **Figure AI** 的一个重要里程碑，展示了 **Helix 02 机器人** 自主清理客厅。
   - 根据 [他的 X 帖子](https://xcancel.com/adcock_brett/status/2031039203262501252)，这次演示是他们将机器人整合到每个家庭的更广泛使命的一部分。
- **Adcock 的愿景：机器人进入千家万户**: **Helix 02** 的演示符合 **Figure AI** 将自主机器人引入日常生活空间的目标。
   - 根据 [Adcock 的公告](https://xcancel.com/adcock_brett/status/2031039203262501252)，公司专注于开发能够执行家务任务的机器人，从清理等简单动作开始。


  

---


### **Latent Space ▷ #[san-diego-neurips-2025](https://discord.com/channels/822583790773862470/1335732885717651558/1479605627729743993)** (2 messages): 

> `` 


- **频道内无讨论**: 该频道内没有进行讨论。
   - 未讨论任何话题。
- **频道不活跃**: 该频道似乎处于不活跃状态，没有可总结的消息。
   - 用户表达了对错过某些事情的遗憾，但没有提供关于错过了什么的具体细节。


  

---

### **Latent Space ▷ #[genmedia-creative-ai-video-image-voice-music-inspo-consumer-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1479580008270790818)** (18 条消息🔥): 

> `Ben Affleck AI 初创公司, ComfyUI, Mr. Beast 迪拜挑战, Seedance 2 AI 视频在中国的使用, AI 自动化 TikTok Shop 视频制作` 


- ****Affleck 的 AI 创企被 Netflix 收购****: Ben Affleck 的 AI 视频初创公司 **Interpositive** 自 **2022** 年起已被 [Netflix](https://about.netflix.com/en/news/why-interpositive-is-joining-netflix) 收购。
   - 这次收购标志着 Netflix 对 AI 驱动的视频制作的持续关注。
- ****ComfyUI 备受关注****: 一名成员询问 [Interpositive](https://x.com/charliebcurran/status/2030401776353218716?s=12) 是否在广泛使用 **ComfyUI**。
- ****Mr. Beast 迪拜挑战点子走红****: Charles Curran 发出的一个热门帖子提议了一个围绕逃离 [迪拜](https://xcancel.com/charliebcurran/status/2030401776353218716?s=12) 的 **Mr. Beast 挑战**。
- ****中国拥抱 AI 生成视频****: Justine Moore 强调了中国使用 **Seedance 2** 制作的高质量 AI 生成内容的兴起，正从短片转向在 [小红书 (Rednote)](https://xcancel.com/venturetwins/status/2030698424761495990?s=12) 上可见的复杂剧集。
- ****TikTok Shop 广告的“强力加速”****: Noah Frydberg 描述了一个高产量的内容流水线，每天使用 **Clawdbot**、**Kling**、**Arcads** 和 [CapCut](https://xcancel.com/maverickecom/status/2031032671774577101?s=12) 生成超过 **500** 条电影感的 TikTok Shop 广告。


  

---


### **Latent Space ▷ #[tokyo-japan](https://discord.com/channels/822583790773862470/1409723236232462346/1480261088288444577)** (5 条消息): 

> `Shane Gu, Google 涩谷, 日本 AGI 生态系统, AI 人才` 


- **Shane Gu 将在 Google 涩谷办公室办公**: [Shane Gu](https://xcancel.com/shaneguML/status/2030692568896524305) 宣布他将在东京的 **Google Shibuya** 办公室工作。
   - 他提出愿意接待到访的 **AI 人才**，并协助与当地的 **日本 AGI 生态系统** 建立联系，包括政府官员、CEO 和工程师。
- **Shane Gu 提议接待到访的 AI 人才**: [Shane Gu](https://xcancel.com/shaneguML/status/2030692568896524305) 正提议在东京接待到访的 **AI 人才**。
   - 他的目标是将他们与当地的 **日本 AGI 生态系统** 联系起来，包括政府官员、CEO 和工程师。


  

---


### **Latent Space ▷ #[ai4science-bio-math-physics-chemistry-ai-researcher-ai-scientist](https://discord.com/channels/822583790773862470/1430253273335595079/1480250949720346745)** (7 条消息): 

> `果蝇连接体模拟, 科学 AI 地图准确性` 


- **果蝇连接体驱动虚拟昆虫**: 研究人员成功地在虚拟身体中利用果蝇的神经连接体（connectome）模拟了**果蝇的行为**，实现了从行为建模到**生物结构建模**的跨越 [[推文链接](https://xcancel.com/oh_that_hat/status/2030383547832533378?s=12)]。
   - 这引发了关于通过 Scaling 模拟人类大脑未来的讨论，完全绕过了传统的 AI 训练方法。
- **科学 AI 地图准确性受到质疑**: 一名成员分享了一个**科学 AI 地图**的[探索链接](https://app.topology.vc/scientific-ai-map)。
   - 该成员指出，根据他们的了解，该地图的准确性存疑，但“*至少值得一看*”。


  

---


### **Latent Space ▷ #[minneapolis](https://discord.com/channels/822583790773862470/1436527872876740609/1480259795507347536)** (1 条消息): 

> `该地区活跃小组, AIE 活动` 


- **对当地活跃 AI 小组的热情**: 一名成员对该地区出现的活跃 AI 小组表示兴奋。
   - 该用户表示无法参加即将举行的 **AIE** 活动，但很高兴看到社区参与度增加。
- **错过 AIE 的机会**: 一名成员对无法参加下一次 **AIE** 活动表示遗憾。
   - 尽管如此，他们对当地社区中活跃的 AI 导向小组不断增加的存在表示满意。


  

---


### **Latent Space ▷ #[ai-in-education](https://discord.com/channels/822583790773862470/1442574438699761784/1480270107212255269)** (1 条消息): 

> `Google 学习 Android 开发` 


- **Google 提供出色的 Android 开发课程**: 一名成员分享了 [Google 新的 Android 开发课程](https://learning.google.com/experiments/learn-about)链接。
   - 他们评论说，虽然一直没空学习 Android 开发，但*这个课程真的非常棒*。
- **N/A**: N/A


  

---

### **Latent Space ▷ #[mechinterp-alignment-safety](https://discord.com/channels/822583790773862470/1445258379357458625/1479591469214990338)** (16 messages🔥): 

> `可解释性研究, 网络安全话题, 安全频道` 


- **Far AI 转向经验可解释性**: [Far.AI](https://xcancel.com/farairesearch/status/2029957875523592524) 讨论了 **Neel Nanda** 向 **经验可解释性 (empirical interpretability)** 的战略转型。
   - 重点已从抽象见解转向可测试的代理任务和激活转向 (activation steering)，优先考虑那些对 **AGI safety** 有可衡量影响的方法。
- **关于网络安全话题讨论的咨询**: 一位成员询问是否要在频道中涵盖 **网络安全话题**。
   - 他们提到了 **RiskyBiz**，本周有一些有趣的 *LLM 失控案例*。
- **申请访问安全频道**: 一位成员询问 **安全频道**，质疑它是否是一个网络安全频道。
   - 他们提到该频道对他们显示为 *无权限访问*，并请求加入。


  

---


### **Latent Space ▷ #[accountability](https://discord.com/channels/822583790773862470/1461796027462979869/1479941989427319004)** (1 messages): 

> `PyTorch PRs, 自回归解码, KV Cache 管理` 


- **PyTorch 变得越来越有趣**: 一位成员表示，在基于 **自回归解码 (autoregressive decode)** 和 **KV Cache 管理** 的研究向 **PyTorch** 提交 **PR** 时，问题空间变得越来越令人兴奋。
- **深入研究自回归解码**: 最近的研究重点是增强 **自回归解码** 技术，以提高效率并降低 Large Language Models 中的延迟。


  

---


### **Latent Space ▷ #[gpu-datacenter-stargate-colossus-infra-buildout](https://discord.com/channels/822583790773862470/1467633569684914349/1479606269009592471)** (9 messages🔥): 

> `Nscale 20 亿美元 C 轮融资, TPU vs GPU 洞察, AI Hyperscaler 融资` 


- **Nscale 获得历史性的 20 亿美元 C 轮融资**: 总部位于英国的 AI Hyperscaler **Nscale** 已获得创纪录的 **20 亿美元 C 轮** 融资，由 **Aker ASA** 和 **8090 Industries** 领投，估值为 **146 亿美元** ([链接](https://x.com/etnshow/status/2030943078018613628))。
   - 行业资深人士 **Sheryl Sandberg**、**Susan Decker** 和 **Nick Clegg** 已加入公司董事会。
- **Google 工程师深入探讨 TPU vs. GPU**: 一位前 Google 工程师讨论了 **TPU** 与 **GPU** 的竞争格局，强调了 Google 内部在高性能 ROI 训练和推理中对 **TPU** 的依赖 ([链接](https://x.com/rihardjarc/status/2031002809986908355))。
   - 关键点包括 **TPU** 在大规模训练中优越的 **性能 TCO 比** 和 **可靠性**，并观察到 **NVIDIA** 的主要护城河在于推理框架而非训练。


  

---


### **Latent Space ▷ #[applied-ai-experimentation](https://discord.com/channels/822583790773862470/1470417186651897858/1479562163185123480)** (54 messages🔥): 

> `Codex 构建单个二进制文件, QEMU 疑难问题, Playwright 测试, 基于 LLM 的最小复杂度设置, Chromium DRM/KMS 渲染` 


- **Codex 为电子墨水屏构建精简版 Chrome**: 一位成员正使用 **Codex** 构建单个二进制文件和 Linux 内核系统，将 **Chrome** 渲染到 **电子墨水屏** 上，旨在通过定时、键盘、鼠标或网络唤醒实现最小功耗。
   - 该设置最初使用 **gpt-5.4-medium**，随后使用低配和高配变体，大部分过程已预先配置。
- **QEMU 的怪异问题被特定内核解决**: 一位成员提到了在 **QEMU** 中遇到的困难，特别是恢复时的 virtio gfx 行为，并建议使用真实硬件可能比在 QEMU 中调试更简单。
   - 他们正在构建一个极简的 **Go** 用户态 (userland) 来运行 Web 服务器并抓取网站，目标是将其与针对睡眠优化的内核设置结合。
- **LLM 驱动的灯泡控制**: 一位成员概述了一个通过本地 **Qwen** 聊天窗口控制遥控器和灯泡的计划，LLM 生成代码在按下按钮时在灯泡上创建彩虹图案。
   - 目标是在分布式节点上执行极简的 LLM 输出，且代码在重启后依然存在。
- **Chromium 直接渲染至 DRM/KMS**: 一位成员正在让 **Codex** 构建 **Chromium** 以直接渲染到 **DRM/KMS**，从而可以捆绑一个极简系统：Chromium、一个 Go 二进制文件和内核。
   - 最终系统是一个 **8MB 的二进制文件**（包括静态文件）加上 **15MB 的内核**，包含 **DNS, SSH, HTTP Server 和挂载的文件系统**。
- **Playwright 在 QA 中表现不佳**: 一位成员发现使用 **Playwright** 和 Agent 收集证据的端到端测试经常在 **QA** 中失败，消耗大量 Token 和时间。
   - 他们正在考虑围绕验证阶段而非实现阶段来构建任务，确保在每一层（**单元测试、REST API、UI**）都有更紧密的反馈循环。


  

---

### **Latent Space ▷ #[euno-log](https://discord.com/channels/822583790773862470/1473750131441668096/1479607076975280219)** (2 条消息): 

> `GitHub Social Club, Collaboration Request` 


- **GitHub 举办阿姆斯特丹 Social Club**：GitHub 将于 **周一** 在 **阿姆斯特丹** 举办 [GitHub Social Club](https://discord.com/channels/@me/1479607069501030579/1479607072852148236)。
- **成员寻求美国/欧盟协作者**：一位成员正在寻找愿意与其协作的 **美国或欧盟人士**。


  

---


### **tinygrad (George Hotz) ▷ #[announcements](https://discord.com/channels/1068976834382925865/1069236008115253348/1480481089691586592)** (1 条消息): 

> `Tiny Corp second raise, Bitcoin mine acquisition, AMD contract, NVIDIA GPUs` 


- **Tiny Corp 寻求 1000-2000 万美元融资**：Tiny Corp 正在启动其第二轮融资，目标是以 **2 亿美元的投前估值** 筹集 **1000-2000 万美元**，详见 [此 Twitter 线程](https://x.com/__tinygrad__/status/2030855329261711685)。
   - 此次融资面向个人而非 VC 或基金，经证明的合格投资者最低投资额为 **100 万美元**。
- **Tiny Corp 将收购比特币矿场**：资金将用于收购一座 **5-20 MW** 的比特币矿场，利用当前市场价格低于 **100 万美元/MW** 且电费低于 **5c/kWh** 的有利条件。
   - 此次收购旨在锁定电力空间，预见未来硬件产品（可能利用 **RDNA5** 显卡）将具有良好的单位经济效益。
- **Tiny Corp 获得 200 万美元 AMD 合同**：自此前以 **5000 万美元投前估值** 完成 **500 万美元** 融资以来，Tiny Corp 已与 **AMD** 签下 **200 万美元** 合同。
   - 他们正在为 **AMD** 和 **NVIDIA GPU** 开发低至 **PCIe 层** 的全栈，并已拥有可工作的 **USB 驱动程序** 作为证明。
- **Tiny Corp 专注于盈利能力**：该公司持有约 **500 万美元** 现金和 **150 万美元** 资产，强调其对盈利能力的承诺。
   - 该战略旨在通过代币销售实现硬件回收期 **< 18 个月**，从而在竞争中胜过云供应商，并可能利用 **RDNA5** 显卡。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1479663762062704681)** (512 messages🔥🔥🔥): 

> `Tinygrad 融资、比特币矿场收购、电力来源策略、合格投资者要求、去中心化 vs 中心化计算` 


- ****Tinygrad 宣布融资 2000 万美元用于电力布局****: George Hotz 宣布 [Tinygrad 正在以 2 亿美元的投前估值融资 1000-2000 万美元](https://x.com/__tinygrad__/status/2030855329261711685)，寻求起步资金 100 万美元的合格投资者（Accredited Investors），用于收购比特币矿场以获取廉价电力，不允许 VC 或基金参与。
   - 目标是 *在我们的设备拥有良好的单机经济模型（即：我们可以制造机器，并通过出售 token 在 18 个月内回本）的那一刻，就拥有现成的现金和带电场地*，通过优化运行消费级 GPU，在竞争中胜过云服务提供商。
- ****收购比特币矿场成为 Tinygrad 的战略重点****: Tinygrad 正在转向 [收购比特币矿场](https://www.loopnet.com/Listing/2840-NW-35th-Ave-Portland-OR/35001807/) 以获取廉价电力（建设成本 <$1M/MW，电价 <5c/kWh），用于运行消费级 GPU，旨在通过 token 销售击败云服务商。
   - 该策略利用低电力成本和优化的软件来实现盈利和规模化，comma.ai 可能会租用托管空间以提供即时现金流。
- ****电源方案选择引发辩论****: 关于 [电源方案](https://www.nytimes.com/2025/10/10/business/china-solar-tibetan-plateau.html) 的讨论非常激烈，涉及数据中心运营中的太阳能、风能、天然气和电池，权衡成本、可靠性和环境影响，选址考虑在华盛顿州、德克萨斯州和孟菲斯。
   - 最优方案包括寻找具有稳健购电协议（PPA）的比特币矿场，并探索抽水蓄能、电池和电网供电。但许多人对 PetaFlops 的商品化、市场饱和以及廉价的中国劳动力和硬件表示担忧。
- ****融资过程中的合格投资者身份受到审视****: 虽然对 [合格投资者](https://www.sec.gov/education/capitalraising/building-blocks/accredited-investor) 的要求存在疑虑，但 George 坚持遵守法律并专注于志向一致的个人。
   - 虽然最低投资额为 100 万美元，但正如一位用户所说，参与的唯一方式是 *如果有人投资你，而我们可以投资他们*。
- ****关于去中心化的争论引发分歧****: 社区对 [去中心化 vs 中心化](https://luma.com/cqq4mojz) 计算的优劣展开了辩论，涉及去中心化模型中的隐私、安全和工程复杂性。但 Tinygrad 最终倾向于中心化控制，以获得更便宜的电力和更简单的管理。
   - 虽然讨论了分布式 tinybox 和太阳能供电系统等去中心化方案，但 *这些正是为什么中心化更有意义的原因。在意识形态上我喜欢去中心化，但如果它让工程变得更复杂，那就没有意识形态的空间了。*

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1479559795425021992)** (66 messages🔥🔥): 

> `实习生招聘冻结, Compute Conference 门票, AMD Kernel 竞赛, GPU Kernel 漏洞利用, 职位预测模型` 


- **实习 Offer 被撤回，社区伸出援手！**：一家公司撤回了一个 ML Eng/ML Ops 实习 Offer，促使一名成员为其寻求机会；他的 LinkedIn 个人资料见[此处](https://www.linkedin.com/in/mramamon/)。
   - 该成员表达了遗憾，并希望这位通过了技术面试的实习生能找到另一个职位，可能是在 Discord 社区内。
- **免费 Compute Conference 门票分享！**：一位成员分享了 **3 张免费门票**，可在 [Luma](https://luma.com/k6bc82dv) 上使用代码 `EQ6VA5` 兑换。
   - 其他人表示感谢，并提到在 **MUNI** 上看到了广告；一名用户计划在不请假（PTO）的情况下偷偷前往。
- **GPU Mode Kernel 被黑，AI 来救场！**：一位用户报告称 [gpumode.com](https://www.gpumode.com/home) 上的大多数 Kernel 都容易受到漏洞攻击，并将发现结果发布到了 [gist.github.com](https://gist.github.com/josusanmartin/5cc5f696fe827cf5ba3e1da4355e1565)。
   - 管理员已意识到该问题，并表示他们正在通过更好的 **AI 自动化**和名为 **pygpubench** 的新库来修复它，同时鼓励成员尝试攻破他们的新评估程序；一名用户在 [github.com](https://github.com/gpu-mode/pygpubench/issues/23) 提交了一个 issue，并在 [github.com](https://github.com/gpu-mode/pygpubench/pull/24) 提交了一个包含潜在缓解措施的 PR。
- **职位预测模型部署在 Modal！**：一位成员宣布完成了一个 **Job Predictor 模型**，代码托管在 [GitHub](https://github.com/ak3zaidan/JobPredictor) 上。
   - 一名管理员更新了 GitHub 仓库，增加了一个 `modal_app.py` 文件，该文件允许在几秒钟内在 **Modal** 上完成部署和推理。
- **GTC San Jose 游戏化！**：一位成员制作了一个网页游戏来导航 **GTC San Jose**，访问地址为 [gtc-2026-interactive-map.vercel.app](https://gtc-2026-interactive-map.vercel.app/)，该游戏还追踪了食物推荐。
   - 几名成员表示有兴趣组团参加 **GTC**，希望能在此次会议上结交朋友。


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1479777828365926640)** (9 messages🔥): 

> `Log-Matmul 优化, 快速 Exp2 和 Log2 Kernel, Profiling Triton Kernel, tl.sqrt vs tl.sqrt_rn` 


- **Log-Matmul Kernel 优化**：一位成员寻求在 Triton 中优化 `log2(M@exp2(X))` 的建议，目标是在 **RTX 4090** 上实现数值稳定性和速度，特别是尝试将指数运算与 **tl.dot** 重叠（overlap）。
   - 该成员提供了自定义的 `_fast_exp2` 和 `_fast_log2` Kernel，其编写目的是为了避免使用 SFUs 以提升性能。
- **矩阵乘法速度测试**：一位成员建议在不带 `exp` 和 `log` 的情况下对原始矩阵乘法（**M@X**）进行基准测试，以评估 Triton 的 Matmul 效率与 **cuBLAS** 的对比。
   - 原作者指出，跳过 `exp` 和 `log` 几乎没有改变延迟，因此瓶颈在别处，并计划进一步隔离问题，同时询问应该使用什么工具。
- **使用 Nsight 对 Triton Kernel 进行 Profiling**：一位成员建议使用 **ncu** (Nsight Compute) 生成 Kernel 性能指标的文本报告，并配合 AI 辅助分析输出来寻找瓶颈。
   - 原作者对该建议表示感谢。
- **tl.sqrt vs tl.sqrt_rn 详情**：一位成员询问 `tl.sqrt` 和 `tl.sqrt_rn` 之间的区别，了解到其中一个更快，而另一个符合 **IEEE 标准**且更精确。
   - 他们寻求详细信息，了解区别是否仅限于最近舍入（round-to-nearest），还是还包括 **NaN/ftz** 行为，并询问了平台依赖性以及 `tl.fdiv` 与最近舍入的等效性。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1479556402728141023)** (31 messages🔥): 

> `fp16 vs fp32 吞吐量，cuBLAS 性能，优化 fp8 group gemm，mbarriers 和 TMA` 


- **不同 NVIDIA 架构下的 FP16 吞吐量提升有所差异**：在 **Turing (7.5)** 和 **Hopper (9.0)** 上，**fp16** 的速度是 **fp32** 的两倍；而在 **A100 (8.0)** 上，其速度是 **4 倍**；但自 **Ampere 8.6** 起，**fp16** 开始使用 **fp32** 单元，正如 [CUDA C Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions) 和 [NVIDIA 开发者论坛](https://forums.developer.nvidia.com/t/separate-cuda-core-pipeline-for-fp16-and-fp32/302018/7) 中所提到的。
- **通过自定义 Kernel 匹配 cuBLAS 性能**：一名成员正在开发一个项目，旨在编写自定义 Kernel 并在其 **GPU** 上达到 **cuBLAS 级别的性能**，类似于 [siboehm.com 关于 CUDA MMM 的文章](https://siboehm.com/articles/22/CUDA-MMM) 中描述的方法。
- **寻求优化 fp8 group gemm 的建议**：一名成员正在寻求针对 **sm120 平台** 优化 **128x128x128** 的 **fp8 group gemm** 的建议，并想知道针对小 M 维度，除了 **ping pong** 策略外是否还有其他思路。
- **mbarriers 和 TMA 的语义**：一名成员就 **mbarriers** 和 **TMA / cp.async** 在减少待处理到达计数（pending arrival count）方面的语义提出了疑问。
   - 会议澄清了：*expected count（预期计数）需要初始化为 arrive 操作的数量，例如 `mbarrier.arrive`（包括 `.expect_tx`）*，并且*只有当两个计数都达到零时，一个 phase（阶段）才算完成*。


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1479948556990283946)** (1 messages): 

> `GTC, Decart, Accel, diffusion 性能, Flash Attention 4` 


- **GPU MODE, Decart, Accel 举办 Diffusion Meetup**：GPU MODE 将在旧金山与 **Decart** 和 **Accel** 共同举办一场小范围聚会，讨论 **diffusion 性能**，时间为 3 月 11 日星期三下午 6:00 - 8:00。
   - 演讲者包括 **Ted Zadouri**（**Flash Attention 4** 的第一作者）、**Ben Spector**（flappyairplanes 的 **flapping efficiency** 专家）以及 **Decart** 的 **Orian Leitersdorf**。活动名额限制在 30-50 人，[在此预约](https://luma.com/gpumodexdiffusion)。
- **Flash Attention 4 和 Flapping Efficiency 专家领衔**：本次活动将邀请 **Flash Attention 4** 的第一作者 **Ted Zadouri**，以及来自 FlappyAirplanes 的 **flapping efficiency** 专家 **Ben Spector**。
   - 此次聚会旨在促进围绕 **diffusion 性能** 的最新进展和技术的讨论。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1479767274129719357)** (2 messages): 

> `Temporal Tiling, PyTorch 中的 Symmetric Memory Allocator, cuMemGranularity API, RB Trees, 用于范围查找的驱动级 API` 


- **Temporal Tiling 话题启动**：一名成员询问了关于 **stencil computations（模板计算）** 的 **temporal tiling** 最新发现的经验，特别是参考了论文 [Recursive DiamondCandy](https://example.com)。
   - 该成员表示报告的结果非常有趣。
- **PyTorch Allocator 替代方案**：一名成员指出 PyTorch 中的 **symmetric memory allocator** 并不理想，并询问了优化解决方案，分享了[一个讨论链接](https://x.com/skyli0n/status/2030352633274515862?s=46&t=PRoCLKOdRTVeAebuUYVI-Q)和[相关 PR](https://github.com/pytorch/pytorch/pull/176744)。
   - 提出的解决方案包括使用 **cuMemGranularity API** 实现粒度分配器、利用 **RB Trees** 实现更快的查找，或采用驱动级 API 进行范围查找。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1480097010890707055)** (2 messages): 

> `Colfax, Blackwell GPU, Auto Research` 


- **Colfax 提及 Blackwell Block Scaling**：Colfax 在其 [CUTLASS 教程](https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/)中重点介绍了 **NVIDIA Blackwell GPU 硬件支持的块缩放（block scaling）**。
- **Auto Research 受到关注**：一名成员注意到 Colfax 提到了 [Auto Research](https://github.com/karpathy/autoresearch)。


  

---

### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1480726217220165683)** (1 条消息): 

> `Website Development, App Development, AI Systems, Chatbots, Automation` 


- **提供网站、应用和 AI 系统开发服务**：一名成员自荐为拥有 **7 年以上** 经验的开发者，专注于构建网站、应用和 AI 系统，提供帮助企业建立线上形象和自动化任务的服务。
   - 他们强调了创建简洁的企业网站、在线商店、移动应用、客户支持聊天机器人以及用于 **内容创作和数据摘要的 AI 助手** 的能力。
- **AI 与自动化的实际应用**：该成员提供了其服务如何转化为实际应用的示例，例如提供 **24/7 客户支持** 的聊天机器人和自动发票系统。
   - 其他应用包括 **用于筛选工作申请的 AI 工具** 以及简化电子邮件和营销内容创作的 AI 助手。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1479555629675839724)** (16 条消息🔥): 

> `CUDA C++ coding, Popcorn CLI usage, AI-generated code in competitions, Backward pass kernels, CUDA book` 


- **RTX 4050 足以进行 CUDA C++ 开发**：了解 **C++ 基础知识**（重点是指针和手动内存管理 malloc 和 free）足以开始编写 **CUDA C++ 代码**，并使用 **RTX 4050** 进行实验。
   - 避免在 **GPU** 代码内部使用 **STL** 或 **std::vector**，需手动在计算机的 **RAM (Host)** 和 **GPU (Device)** 之间移动数据。
- **Popcorn CLI 支持远程 Kernel 提交**：通过 **Popcorn CLI**，用户可以为远程机器提交 kernels，在竞赛的第一阶段不需要像 **MI355X** 这样的特定硬件。
   - 第二阶段将授予团队直接的 **SSH 访问权限**。
- **竞赛允许使用 AI 生成的代码**：竞赛中允许使用 **AI-generated code**，一名参赛者利用它获得了前 4 名的成绩。
   - 一位用户请求解释如何检查自己的代码是否运行良好，类似于神经网络训练。
- **解读 Backward Pass Kernels**：在处理 **backward pass kernels** 时，先从理解 **chain rule** 和简化方程式开始，通过练习以及使用 **SymPy** 等工具辅助推导过程。
   - **Backpropagation** 的经验和来自 **JAX** 论文的见解可以进一步辅助该过程。
- **《CUDA by Example》是否依然适用？**：一位用户询问 Sanders 和 Kandrot 所著的《CUDA by Example》一书是否仍是 CUDA 编程的有用指南。
   - 未给出回答。


  

---


### **GPU MODE ▷ #[jax-pallas-mosaic](https://discord.com/channels/1189498204333543425/1203956655570817034/1479912333403885689)** (3 条消息): 

> `JAX server, profiler view, MXU/DMA/VPU utilization` 


- **JAX 维护者在 Discord 上很活跃**：成员指出 [JAX 维护者在 JAX Discord 服务器上更活跃](https://discord.gg/bDqGxbm4rp)。
   - 用户建议加入该 Discord 以获取更直接的帮助和支持。
- **请求 MXU/DMA/VPU 的 Profiler 视图**：一名成员询问是否有 **profiler view** 可以显示 **JAX** 的 **MXU/DMA/VPU utilization/trace**。
   - 他们寻求比 **xprof** 中的 trace viewer 更细粒度的工具。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1479673034607558687)** (4 条消息): 

> `AMD kernel dev comp, MI355X, AMD GPUs and HIP, device 128-bit atomics` 


- **AMD Kernel 开发竞赛：频道确认**：一名成员询问该频道是否是讨论 **AMD kernel dev competition** 以及获取竞赛用 **MI355X** 访问权限的正确地方。
   - 另一名成员确认这确实是正确的频道。
- **Popcorn CLI：简化方案提交**：一名成员提到了用于提交方案的 [popcorn-cli 工具](https://github.com/gpu-mode/popcorn-cli)，指导用户使用 `popcorn submit solution.py`。
   - 随后会出现一个菜单引导用户完成提交过程。
- **AMD GPU 原子操作支持：128 位咨询**：一名成员询问 **AMD GPUs** 是否支持通过 **HIP** 进行设备端 **128-bit atomics**（如 compare-and-swap）。
   - 他们注意到已支持 **64-bit**，并寻求关于 **128-bit** 原子操作可用性的澄清。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/)** (1 条消息): 

puyanlotfi: 有人有将较新的 LLVM IR 降低（lowering）到 AIR 的经验吗？
  

---

### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1480531182126108682)** (1 messages): 

> `Kernel Envs Repo, Reward Hacking, Pygpubench` 


- **Kernel Envs Repo 已上线！**: 一名成员分享了一个 [repo](https://github.com/teelaitila/kernel-envs)，其中包含他们创建或修改的 **environments**（环境）。
   - 他们计划本周在此基础上开展更多工作。
- **奖励欺骗 (Reward Hacking) 的烦恼！**: 一名成员指出了在 AMD 竞赛中应对奖励欺骗的艰难。
   - 他们建议在下次比赛中使用 **pygpubench**。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1479949053369254110)** (2 messages): 

> `Open-Set Test Environment, Text-Game-Engine Harness, Consent Handling in Models, Judging Model Performance` 


- **受 FLE 启发的开放集环境**: 一名成员正在构建一个受 FLE 启发的开放集测试环境，详见 [此处](https://github.com/bghira/text-game-benchmark/blob/main/scenarios/noir_multiplayer_privacy.yaml)。
   - 其目的是测试模型在多玩家世界中的交互，重点关注涌现特性 (emergent properties) 而非代码生成。
- **Text-Game-Engine 优化 GLM-5 和 Claude Sonnet**: 一个基于 **text-game-engine** 构建的自定义测试框架 (harness) 在数万个回合中优化了 **GLM-5** 和 **Claude Sonnet**。
   - 该框架旨在避免让模型通过“走捷径”优化出解决方案，而是测试“简单”模型中的涌现特性。
- **模型处理与 NPC 的共识/同意**: 该环境检查模型如何处理用户与 NPC 交互中的共识，如 [此示例](https://github.com/bghira/text-game-benchmark/blob/main/scenarios/alice_refusal.yaml) 所示。
   - 这测试了模型在交互中尊重拒绝请求的能力。
- **人类裁判评审，裁判被评审**: 使用更大规模的裁判模型或人类来评判模型的表现。
   - 该系统还支持对人类的游玩过程进行评审，并允许人类对裁判进行评价，增加了一层元评估 (meta-evaluation)。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1479554365131067442)** (256 messages🔥🔥): 

> `AMD Developer Account Credits, popcorn-cli, Submission Errors, Competition Rules, AMD dev program` 


- ****Popcorn CLI****: [popcorn-cli](https://github.com/gpu-mode/popcorn-cli) 是一个**基于队列的系统**，允许用户在无需本地 GPU 的情况下提交任务；特等奖获得者在比赛期间从未租用过 GPU，正是利用了这一系统。
   - 一名成员指出：*你只需要提交即可，本地不需要 GPU*。
- ****提交中的怪现象****: 用户遇到了与包含 **stream** 操作的代码相关的 **500 错误**，这是因为系统通过简单地从代码中删除“stream”单词来进行初步检查。
   - 一位用户开玩笑地建议：*你只需要从代码里删掉 "stream" 这个词就行了哈哈，我们的检查非常简陋*。
- ****基准测试头脑风暴****: 参赛者讨论了**基准测试流程**、限制和环境，并参考了 [reference kernels](https://github.com/gpu-mode/reference-kernels/tree/main/problems/amd_202602) 和 [popcorn-cli](https://github.com/gpu-mode/popcorn-cli) 以获取必要信息。
   - 一些用户建议提交一个不使用 **AITER** 的 baseline 以避免编译延迟，并可能向 [reference-kernels](https://github.com/gpu-mode/reference-kernels) 贡献一个纯 PyTorch 的 kernel。
- ****开启作弊打击****: 针对**刷榜 (benchmark gaming)** 和作弊行为的担忧被提出，管理员强调了诚信参与的重要性。
   - 提到：*我们将持续检查您的提交是否符合规则*，任何可以合并到 **VLLM/SGLANG** 的内容都是合规的，而任何奖励欺骗 (reward hacks) 行为都将导致取消资格和封禁。
- ****硬件寻找趣事****: 用户询问如何租用或获取 **MI355X GPUs** 以参加比赛，并注意到 AMD 开发者计划中缺少该型号。
   - 建议在获得 MI355X 访问权限之前，先使用平台进行提交。


  

---

### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1479584579911155943)** (14 messages🔥): 

> `shields.io 的 Discord 小组件设置, 链接到 Discord 服务器的 shields.io 徽章, Discord 频道链接` 


- **请求激活 **shields.io** Discord 小组件**：一名成员请求在服务器上启用 Discord 小组件设置，以便 **shields.io** 徽章能够引导读者进行提问/评论/贡献，并引用了 [shields.io 徽章页面](https://shields.io/badges/discord)。
- **Discord 链接故障排除**：在启用 Discord 小组件后，一名成员报告在 playground 和 SITP 书籍上仍然收到 *"widget disabled"*（小组件已禁用）的消息，尽管更新据称已经传播。
   - 该成员建议徽章应链接到 **start-here** 频道。
- **Discord 服务器链接最佳实践**：关于徽章应链接到何处，建议使用 **start-here** 频道作为通用入口点，以便其他人员/项目链接到 Discord 服务器时复用。
   - 一名成员补充说，**shields.io** 徽章将显示用户数量，并可以通过 Markdown 链接到特定频道。
- **资源流徽章**：一名成员分享了 GitHub 上 **gpu-mode/resource-stream** 徽章的链接，称其带有 Discord 图标，更加美观。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1479556871764443177)** (43 messages🔥): 

> `迁移后的 Heroku 服务器问题, 旧 PMPP 排行榜提交错误, Numba 安装, Triton vs. Numba, popcorn-cli 的 503 错误` 


- ****Popcorn API URL** 指向旧的 Heroku 实例**：迁移后，用户面临 **Heroku server not found** 问题，因为 `POPCORN_API_URL` 指向旧的 Heroku 实例。现在的正确 URL 是 [site--bot--dxfjds728w5v.code.run](https://site--bot--dxfjds728w5v.code.run)。
   - 一名成员通过清除 `.popcorn.yaml`、设置新的 `POPCORN_API_URL` 并重新注册以获取新的 `popcorn.yaml` 密钥解决了此问题。
- **提交到旧的 PMPP 排行榜返回 404**：提交到旧的 PMPP 排行榜会返回 **404 Not Found** 错误，因为 [reference kernels repo](https://github.com/gpu-mode/reference-kernels) 中缺少 `submission.py` 文件。
   - 一名成员建议，可以使用 solutions 文件夹中的任何文件进行提交。
- **提交时自动安装 Numba**：由于未预装 Numba，用户在提交期间使用 `subprocess` 调用 `pip install numba` 进行安装。
   - 另一名成员确认这是可以接受的。
- **内核使用 **Triton** 还是 **Numba**？辩论仍在继续**：一名成员认为，在本周早些时候 Triton 发布 **FA4** 之后，*很难有理由不使用 Triton*。
   - 另一名成员反驳说，**Numba** 在语法和语义上更接近 **C++/CUDA**，并引用了 [GPU Puzzles](https://github.com/srush/GPU-Puzzles) 教程。
- **503 错误困扰通过 popcorn-cli 进行的提交**：用户在通过 `popcorn-cli` 和 Web UI 提交时遇到了频繁的 **503 错误**，一名用户报告失败率为 **50%**，而 Claude 的提交失败率更高。
   - 日志中查明了问题，修复方案已推出，用户报告问题已解决。


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1480399004696187021)** (2 messages): 

> `跟踪进度, 公开文件` 


- **询问者寻求进度跟踪**：一名成员询问了**跟踪某些文件进度**的方法。
   - 他们希望了解这些文件是否已经公开，以便更好地洞察。
- **文件公开可用性状态**：该用户还在寻求有关项目相关某些文件**公开可用性**的信息。
   - 了解这些文件的可访问性对于用户有效监控并贡献于项目的进展至关重要。


  

---

### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1479574923004350595)** (36 条消息🔥): 

> `暑期实习建议、具体/可验证的经验、为 OSS 做贡献、其他大学的研究机会、工作方法与组织` 


- **大二学生寻求暑期实习建议**：一名计算机科学专业的大二学生正在寻求获得暑期实习的建议，尽管他已经在研究 **GPU/systems**、**compiler/ML systems**，并维护着一个**技术博客**。
   - 该学生的主要担忧是，尽管有强大的学术背景和多个项目，但仍缺乏面试机会。
- **具体经验能区分候选人**：一名学生提到，主要问题在于缺乏能使该学生脱颖而出的**具体/可验证/有声望的经验**。
   - 他们强调了发表论文、开源工作以及可验证结果的重要性，例如*修复 bug、加速 kernel 或在竞赛中获得高排名*。
- **为 OSS 做贡献提供生产级经验**：成员们强调了**为 OSS 做贡献**以获得生产级经验的重要性，并以此解决“需要经验才能找到工作”的**鸡生蛋蛋生鸡问题**。
   - 他们建议在感兴趣的项目中寻找一个没有评论或未开启 **PRs** 的简单 issue，并专注于对项目最有利的事情。
- **其他大学的研究机会**：成员们讨论了在其他大学进行研究的可能性，即使没有在那里的学籍，这也为底层开发工作开辟了潜在机会。
   - 有人提到教授和学生往往更倾向于本校人员，但去其他地方参与研究仍然是可能的。
- **对 OSS 贡献的畏难心理**：一位成员表达了对参与 **OSS** 贡献的畏缩感，原因是过去曾遇到过挑剔的审核者，且难以建立预期。
   - 他们觉得在过度沟通时，可能会挫伤维护者给予关注的积极性。


  

---


### **GPU MODE ▷ #[cutile](https://discord.com/channels/1189498204333543425/1461235643211321437/1479694829851115592)** (3 条消息): 

> `cuTILE 库、自定义 Kernels、Liger 性能、Qwen3 模型、FlashAttention 反向 kernel` 


- **Bastile 在 Qwen3 上超越 Liger**：一位独立开发者构建了一个名为 [Bastile](https://github.com/aghilann/bastile) 的小型基于 **cuTILE** 的 monkey-patching 库，其自定义 kernel 在 **Qwen3** 模型上的单 kernel 和端到端性能均优于 **Liger**。
   - 该开发者优化了来自 **TileGym** 的 kernel 并将改进合并到了上游，还提供了一个 [Modal notebook](https://modal.com/notebooks/aghilann/main/nb-9JUUBXJ23NK2b9Mf01WdEl)，其中包含在 **B200** 上的基准测试结果。
- **尝试使用 cuTILE 超越 FlashAttention**：一位开发者正从 NVIDIA 发布的正向 kernel 出发，利用 **cuTILE** 的简洁性构建 **FlashAttention 反向 kernel**。
   - 开发者指出，超越官方的 **FA** 门槛很高，而且由于大多数人尚未迁移到 **CUDA 13.1** 或 **Blackwell**，因此并未进行太多宣传。
- **关于使用 cuTile 和 Triton 进行基于块编程的 GTC 演讲**：一场 **GTC** 演讲将涵盖在半导体制造工艺控制的 KLA 工作负载中，对**基于块的编程（cuTile 和 Triton）**的评估。
   - 该演讲定于 **PDT 时间 3 月 16 日星期一下午 5 点**，将详细介绍工作负载如何映射到基于块的编程模型，并以在 **cuTile** 和 **Triton** 中将 **2D 卷积**映射到 **Tensor Cores** 为例进行案例研究。


  

---

### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1479794571536896121)** (13 条消息🔥): 

> `GDN Prefill 数值问题, Tracing 问题, DSA TopK Indexer FP8 问题` 


- **GDN Prefill Kernel 令人头疼**: 一位成员为原始数据集创建了一个非官方补丁，以使 **GDN prefill** 能够正常工作，可在[此处](https://huggingface.co/datasets/AlumKal/mlsys26-contest)获取。
   - 另一位成员分别使用 **Triton** 和 **CuteDSL** 编写了 Kernel，但无法解决 **GDN Prefill kernel** 的数值问题。
- **Tracing 问题**: 一位成员在 Tracing 方面遇到困难，他们的实现直接使用了参考实现，但在运行 `modal run script/run_model.py` 时，没有看到生成任何 trace。
   - 另一位成员建议检查 FlashInfer bench 是否将日志捕获到日志文件中（该文件可设置为参数），这对于查看错误非常有用。
- **DSA TopK Indexer 的 FP8 困扰**: 一位成员在处理 **dsa_topk_indexer_fp8_h64_d128_topk2048_ps64** 问题时遇到麻烦，120 个测试用例中只能通过少数几个。
   - 另一位成员提示，Torch 参考实现在主 MMA 之前会将 k 值上采样（upcast）到 FP32，以便与其 scale 相乘，否则会出现一些舍入误差。


  

---


### **GPU MODE ▷ #[from-scratch](https://discord.com/channels/1189498204333543425/1466534042768904356/1480142788665540781)** (1 条消息): 

> `vLLM, C++, CUDA, batching, paged attention` 


- **用户在优化后提供 vLLM 教程**: 一位成员宣布成功实现了推理，并正在开发 **batching** 和 **paged attention** 功能，随后将提供一份关于在 **C++** 和 **CUDA** 中构建玩具级 **vLLM** 的教程。
- **贡献者计划 vLLM 深度解析**: 在成功实现推理后，一位贡献者正计划推出关于 **vLLM** 的演练或教程。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1479557287608586410)** (2 条消息): 

> `Codex for OSS 发布, OpenAI 收购 Promptfoo` 


- **Codex for OSS 正式开源！**: OpenAI 正在推出 **Codex for OSS**，以支持维护开源软件的贡献者，正如在[推文](https://video.twimg.com/amplify_video/2029998126640287747/vid/avc1/1280x720/ZMdqbgIfCNQeqJ0i.mp4)中所宣布的那样。
   - 维护者可以使用 **Codex** 来审查代码、理解大型代码库并加强安全覆盖；更多详情请见 [OpenAI 开发者页面](https://developers.openai.com/codex/community/codex-for-oss)。
- **Promptfoo 被 OpenAI 收购！**: OpenAI 正在收购 **Promptfoo**，以增强 **OpenAI Frontier** 中的 Agent 安全测试和评估，详情请见 [OpenAI 博客文章](https://openai.com/index/openai-to-acquire-promptfoo/)。
   - 好消息是 —— **Promptfoo** 将在当前许可下保持开源，且 OpenAI 将继续支持现有客户。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1479562464503795773)** (348 条消息🔥🔥): 

> `SORA 2 发布与审查, Seedance 2.0 访问与功能, GPT-5.4 Codex 发布, GPT 模型的上下文窗口大小, 开源 AI 模型设置` 


- **SORA 2 关停？**: 成员们正在讨论 **SORA 1** 可能关停以及 **SORA 2** 的审查问题，一位成员声称 *SORA 2 在前 3 天表现非常出色，直到被审查机制限制到面目全非。*
   - 由于服务器负载问题，**SORA 2** 并非在所有地区都可用，这引发了担忧。
- **Seedance 2.0 即将发布？**: 成员们正热切期待视频生成 AI **Seedance 2.0** 的全球发布，一些人通过中国手机号和 VPN 提前获得了访问权限。一位成员表示它 *原定于 2 月 24 日全球发布*。
   - 一位用户询问了早期访问路径，其他人则将其潜力与 **Flow for Veo** 进行对比，并指出 **Seedance 1.5** 的价格相对便宜。
- **GPT-5.4 将取代 Codex？**: 讨论集中在 **GPT-5.4** 的发布及其与 Codex 模型的关系上。一位成员在分享了 [一条推文链接](https://x.com/ajambrosino/status/2029620806956073376?s=20) 后表示，将不会有 **GPT-5.4-codex**，只有 **GPT-5.4**。
   - 关于 **GPT-5.4** 是专门的 Codex 版本，还是具有计算机使用（computer-use）能力的通用模型，也存在争议。
- **解析 Token 上下文窗口**: 一场关于 GPT 模型上下文窗口大小的讨论澄清：**GPT-5.3** 为 Plus 用户提供 **32K**，而 **GPT-5.4 Thinking** 提供 **256K**（128k 输入 + 128k 最大输出）。
   - 一位成员澄清道：*一个 Token 的范围可以从一个完整的单词到单个字符或逗号，它是 AI 用来阅读和处理语言的基础构建块。*
- **开源 LLM 设置简化**: 一位成员询问设置开源 AI 模型的技巧，另一位成员推荐了 [pinokio.computer](https://pinokio.computer)，这是一个 AI 应用安装程序目录，并提供针对 Nvidia 和 Apple 设备的硬件提示。
   - 另一位成员推荐 **Ollama** 作为一个非常有用的工具。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1479590807190245477)** (53 条消息🔥): 

> `GPT 聊天变慢, OpenCLaw 模型, API 对比订阅, LLM 供应商, 价格上涨` 


- **GPT 聊天变慢，Gemini 则不会**: 一些用户抱怨 [GPT 在长对话中会显著变慢](https://openai.com)，而 **Gemini** 则不会，这导致聊天体验变差并需要刷新页面。
- **OpenCLaw 模型是否兼容 ChatGPT 订阅？**: 一位用户询问 **ChatGPT 订阅** 是否可以与 **OpenCLaw 模型** 配合使用；共识是使用 API 是最安全的方式，但它是付费的而非免费，此外还讨论了订阅是否可以进行关联。
- **LLM 自动压缩聊天历史**: 其他 **LLM 供应商** 如 **Claude**（可能还有 **Gemini**）会自动压缩聊天历史，而 **ChatGPT** 似乎没有这样做，这可能是导致变慢的原因。
- **ChatGPT 价格上涨引发用户不满**: 用户对最近的 **价格上涨** 感到愤怒；正如一位用户指出的，**5.1 是输入 $1.25，输出 $10；5.2 是输入 $1.75，输出 $14；5.4 是输入 $2.50，输出 $15**，由于现在 **输入 Token** 非常普遍，这实际上使成本翻倍。
- **模型版本偏好引发辩论**: 用户辩论了不同 **GPT 模型版本** 的优缺点，一些人更喜欢 **5.3** 的对话能力，以及 **o3** 较少受到技术层干扰的特性，尽管它存在幻觉问题。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1479554781910401174)** (23 messages🔥): 

> `Training GPTs, Rubric Evaluation, Gemini vs ChatGPT accuracy, Goal Lock Prompting` 


- **GPTs 训练通过评分标准 (Rubric) 评估论文**: 一个用户尝试训练 **GPT** 根据评分标准评估论文，另一位用户建议只需上传论文并提示其进行评估，或者在提示词中加入评分标准并要求其对每个类别分别评分。
   - 该建议包括*要求其给出评分依据*，这非常有帮助。
- **无关解释困扰用户的 ChatGPT 输出**: 一位用户报告称，尽管提示词要求提供相关的、深入的理由，**ChatGPT** 仍会添加*无关*的解释。
   - 另一位用户怀疑原用户缺乏详尽的解释，而另一位则建议使用*逐步推理 (step-by-step reasoning)* 和 *goal lock governor*。
- **为 Gemini 提出 Goal Lock Prompting**: 一位用户分享了一种 *goal lock* 提示技术，用以保持 **Gemini** 意图的绝对稳定性，建议 AI 应该提供结构性蓝图，而不是叙述性建议。
   - 该方法涉及明确的逐步推理和保持意图。
- **Gemini 的准确率差异困扰用户**: 一位用户感到困惑，为什么 **ChatGPT** 提供的信息是准确的，而 **Gemini** 提供的信息却不准确。
   - 一位用户回应要求他们说明自己的目标。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1479554781910401174)** (23 messages🔥): 

> `GPT Paper Evaluation, Goal Lock Prompting, Gemini vs ChatGPT accuracy, Context importance` 


- **GPT 根据评分标准 (Rubric) 评估论文**: 一位成员询问关于训练 **GPT** 使用评分标准评估论文的问题，对此另一位成员建议可能不需要训练；只需在提示词中提供评分标准，并要求对每个类别进行评分即可。
   - 他们建议要求 **GPT** 为每个评分提供依据，以改进评估效果。
- **Goal Lock Prompting 停止目标偏移 (Goal Drift)**: 一位成员引入了用于提示词的 *Goal Lock Governor* 概念，以保留原始问题陈述并防止*目标偏移*。
   - 他们为 **Gemini** 提供了一个提示词，强调*逐步推理 (step by step reasoning)* 并*明确阐述目标*，以保持*意图的绝对稳定性*。
- **Gemini 与 ChatGPT 的准确性意见不一**: 一位成员询问为什么 **ChatGPT** 报告某些信息是准确的，而 **Gemini** 认为这些信息是不准确的，且未提供更多上下文。
   - 另一位成员简短地回应道：“你要求的是 ChatGPT...”，隐含地建议如果这是用户的偏好，他们应该直接使用 **ChatGPT**。
- **上下文是获得相关回复的关键**: 几位成员强调了在提问时提供充足上下文的重要性，以便获得相关的答案。
   - 一位成员表示：“如果你不提供上下文，它就不会提供相关的上下文。如果你不提供上下文，我们就无法看到你遇到的问题。”


  

---


### **OpenAI ▷ #[api-projects](https://discord.com/channels/974519864045756446/1037561385070112779/1480070019210678444)** (1 messages): 

> `Governing AI Agents` 


- **BIGHUB: 治理 AI Agents**: 成员们讨论了在 **AI Agents** 行动之前对其进行**治理**。
   - 未给出关于讨论背景的细节。
- **AI 治理讨论**: 对话围绕着为 **AI Agents** 建立治理框架的重要性展开。
   - 参与者强调需要主动应对潜在风险并确保负责任地部署 AI，尽管具体提案尚未详述。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1479558576262348944)** (382 messages🔥🔥): 

> `Spark GB10 GPU on Linux, Hermes Agent Skins, Autonomous AI Agent NEX, lmstudio issues, GPT OSS model quality` 


- **Spark GB10 Linux 稳定性受质疑**：一位用户在投资硬件前，询问了 **Spark GB10** 在 Linux 上的稳定性，质疑 Nvidia 臭名昭著的驱动程序是否会导致问题。
   - 另一位成员开玩笑地提议进行“硬件体检”，同时保证每款 GPU 可能都有一个稳定的 Linux 版本。
- **Hermes Agent 通过自定义皮肤换新装**：用户正在为 **Hermes Agent** 开发自定义皮肤，包括一个动态的西西弗斯（Sisyphus）主题，并分享了作品截图，承诺将其作为 PR 提交到主仓库。
   - Ares 和 Posideon 等皮肤已展示了新的个性和自定义动画，并修复了聊天颜色，很快将在主仓库中上线。
- **GPT-OSS 意外获得赞誉**：尽管存有顾虑，一些用户发现 **GPT-OSS** 模型表现出奇地好，这可能归功于它在污染较少的数据上进行过训练。
   - 然而，对于其相较于 *frontier labs*（前沿实验室）模型的性能仍存在怀疑，一位用户认为 Benchmarks 可能会产生误导。
- **检测网络安全中的异常情况**：一位用户正在寻求构建 Windows 日志异常检测系统的建议，理由是数据集包含 **120 万行**，但异常情况少于 **300 条**。
   - 鉴于他们可以调用 H200 用于学术研究，他们希望能获得关于最佳方法和工具的建议，涵盖从 iForests 到类 BERT 的 Transformer 等技术。
- **无限量 Queen3.5 推理**：一位成员发现了一家名为 airouter.ch 的服务商，以 **39 CHF**（约 43 欧元）的价格提供无限量 Qween3.5:120b，且速率限制（Rate limits）非常宽松。
   - 他还询问是否有人有运营推理 Token 销售平台的经验，并质疑为什么德国供应商无法实现标准的 OpenAI API 工具。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1479941960159592760)** (9 messages🔥): 

> `GLM-4.5-Air, oLlamaOn Hermes Agent, Tooling and goals` 


- ****GLM-4.5-Air**：早期试用者寻求帮助**：一位成员询问了如何在本地运行 **GLM-4.5-Air**，并通过 [glm45-tool-calling-setup.md](https://cdn.discordapp.com/attachments/1154120232051408927/1479964826716344521/glm45-tool-calling-setup.md?ex=69b09727&is=69af45a7&hm=2822a52c8a3ab658be478b63aab8de0df8ab7c5100bf27e3fa348c28e9117456&) 链接分享了设置指南。
   - 他们提到使用 [llama.cpp](https://github.com/ggml-org/llama.cpp) 运行“似乎可行”。
- ****Hermes Agent** 被提及**：一位成员提到 **GLM-4.5-Air** 模型似乎并不适配 **oLlamaOn Hermes Agent**。
   - 另一位成员表示目前仅测试过 Opus 4.6。
- **工具化是关键！**：一位成员建议，只要有正确的工具和目标，模型不一定需要训练。
   - 他们补充道：“把它放在合适的环境中，并围绕某些限制进行‘软磨硬泡’，对我来说非常有效。”


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1479898882539520112)** (7 messages): 

> `Steady State Multi Agent Systems, Scheduling AI Models on Edge Devices, PAES Scheduler` 


- **稳态多智能体系统研究启动**：一位成员开始研究 **稳态多智能体系统 (steady state multi agent systems)**，参考了 [Zenodo 记录集](https://zenodo.org/records/18906946) 中的前 3 篇论文，包括 [Record 1](https://zenodo.org/records/18906944)、[Record 2](https://zenodo.org/records/18906942) 以及 [这篇 ArXiv 论文](https://arxiv.org/abs/2511.08065)。
- **PAES Scheduler 展现潜力**：一位成员正在进行一项系统研究项目，重点是在边缘设备上**调度多个 AI 模型**，包含 **视觉、语音和规划** 模型。
   - 他们实现了 **FIFO, round robin, 以及 Earliest Deadline First** 作为基准线，并引入了一个名为 **PAES** 的新调度器。初步结果显示队列等待时间缩短了约 **33%**，且具备更好的突发处理能力。他们目前正在[寻找合作伙伴](https://x.com/joelniklaus/status/2030554880285585544?s=46)共同撰写一篇简短论文。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1479944359502942349)** (7 messages): 

> `学者绅士 (Scholarly gentleman), Suno 歌曲, Grok 图像` 


- **在 X 上发现的学者绅士**：一位成员分享了 X（原 Twitter）上一个关于*学者绅士*的链接：[Praveen Joshi](https://fxtwitter.com/Praveen_G07/status/2027261833376375125?s=20)。
   - 另一位成员也分享了一个链接：[Alex Wagner](https://x.com/alexwg/status/2030217301929132323)。
- **使用 Suno 和 Grok 制作内容**：一位成员询问链接中的内容是否由 **Hermes** 制作，另一位成员回答说它是在 **HermesAgent** 之前制作的。
   - 另一位成员澄清说，这些内容是使用 **Suno**、**Grok**（在 **Gimp** 中编辑）和 **Davinci Resolve** 制作的。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1479898882539520112)** (7 messages): 

> `稳态 Multi Agent 系统, 调度 AI 模型` 


- **稳态 Agent 研究启动**：一位成员正在启动一个关于稳态多 Agent 系统（Multi Agent Systems）的项目，链接了三个 [Zenodo 记录](https://zenodo.org/records/18906946)、[另一个 Zenodo 记录](https://zenodo.org/records/18906944)、[第三个 Zenodo 记录](https://zenodo.org/records/18906942)，以及一篇 [ArXiv 论文](https://arxiv.org/abs/2511.08065)和一条 [推文](https://x.com/joelniklaus/status/2030554880285585544?s=46)。
- **AI 模型边缘调度项目寻求合作者**：一位成员正在开发一个在边缘设备上调度多个 **AI 模型**（特别是*视觉 + 语音 + 规划*）的系统。
   - 他们报告称，使用新的 **PAES 调度器**后，**队列等待时间降低了 33%**，目前正在寻求实验合作者和论文共同作者。


  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1480580003258302494)** (1 messages): 

> `HF ML Club India, Lewis Tunstall, 训练微型模型` 


- **HF ML Club India 启动**：两名成员发起了 **HF ML Club India**，托管在 [huggingface.co/hf-ml-club-india](https://huggingface.co/hf-ml-club-india)。
   - 首位演讲者是 **Lewis Tunstall**，他将讨论如何训练*微型（tiny）*模型来教授*困难*定理。
- **Lewis Tunstall 教授微型定理训练**：**Lewis Tunstall** 将成为 **HF ML Club India** 的首位演讲者，讨论训练*微型*模型。
   - 他的演讲将集中于教授这些模型理解*困难*定理，并分享关于高效模型训练的见解。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1479567860744323235)** (105 messages🔥🔥): 

> `Megatron vs Transformer 速度, HuggingFace datasets 库维护, Llama-2 拒绝申请, Gradio Multimodal Textbox 音频录制问题, API Key 生成问题` 


- **大规模训练中的 Megatron 速度**：一位成员表示，对于大规模训练和重度 SFT，**Megatron** 是首选，而 **TRL** 则更适合偏好微调（preference tuning）和 RLHF 风格的后期训练；并补充说 [NVIDIA 提供了 Megatron Bridge](https://docs.nvidia.com/deeplearning/megatron/user-guide/index.html)，用于混合工作流中的 HF ↔ Megatron Checkpoint 转换。
   - 另一位成员询问如何使用 Megatron 对 **Qwen** 模型进行 LoRA 微调，并分享了 [NVIDIA 文档链接](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/qwen.html?utm_source=chatgpt.com)。
- **HuggingFace datasets 库人手不足？**：用户对 [Hugging Face datasets 库](https://github.com/huggingface/datasets)的维护表示担忧，理由是目前有大约 900 个未解决的 Issue 和 200 个未关闭的 Pull Request。
   - 一位成员表示，由于*不断遇到意外问题、奇怪的内存占用、C++ 导致的硬崩溃以及其他文档不全的行为，他们开始阅读源代码。*
- **Llama-2 申请失败的苦恼**：一位被 Llama-2 拒绝的用户询问如何获得批准。
   - 一位成员建议通过 [讨论页面](https://huggingface.co/meta-llama/Llama-3.1-8B/discussions/13) 联系 **Meta**，或者尝试使用另一个账号或 **Unsloth** 权重。
- **Gradio 多模态音频输入 Bug**：一位用户报告了 **Gradio Multimodal Textbox** 的音频录制/输入问题，即麦克风不录制音频且发送按钮消失。
   - 另一位成员建议这可能是浏览器端的问题，并链接到了[之前的讨论](https://discord.com/channels/879548962464493619/1025174734427656283)。
- **API Key 生成时显示密码无效**：一位用户在生成 **Hugging Face** API Key 时遇到问题，输入密码后显示“无效密码”错误。
   - 一位成员建议通过 website@huggingface.co 联系 **HF 支持团队**。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1479620566242820157)** (22 messages🔥): 

> `PygmyClaw, Text-to-image diffusion transformer, Openvino for LLMs, Ghost Hunter RLHF Dataset, LBNets architecture` 


- **Agent 框架 PygmyClaw 引入全新 Speculative Decoding**：PygmyClaw 是一个紧凑的基于 Python 的 Agent 框架，具有持久化任务队列和模块化工具系统，现已升级支持 **使用 3 个 Drafters 和 1 个 Verifier 的 Speculative Decoding**（四个 Ollama 实例）以更快地生成 Token，访问地址：[webxos/pygmyclaw-py](https://huggingface.co/webxos/pygmyclaw-py)。
- **从零构建的文本到图像 Diffusion Transformer**：一位成员分享了从零构建的文本到图像 Diffusion Transformer，在 **A100 上使用 200k 图像-文本对进行训练**，采用了类似 SANA 的卷积 MLP，可在 [GitHub](https://github.com/merterbak/diffusion-from-scratch) 获取。
- **通过 Nexil 使用 OpenVINO 在本地运行 LLM**：一位成员介绍了一个名为 **Nexil** 的 Python 工具，可通过 OpenVINO 在 Intel NPU 或 CPU 上本地运行 LLM，支持 Linux，可在 [GitHub](https://github.com/James-Crockett/nexil) 获取。
- **WebXOS 发布 Ghost Hunter RLHF 数据集**：新数据集 **Ghost Hunter RLHF Dataset** 包含 8 位机风格第一人称射击游戏《Ghost Hunter》在成功消灭幽灵时的屏幕截图，用于人类反馈强化学习 (RLHF) 任务，可在 [webxos/ghosthunter-RL](https://huggingface.co/datasets/webxos/ghosthunter-RL) 获取。
- **Aclevo 使用 LBNets 创建自定义类推理架构**：Aclevo 使用 Microsoft/Phi-2 模型创建了一种名为 **LBNets** 的自定义类推理（reasoning-like）架构，虽然处于高度实验阶段但功能完备，可在 [HuggingFace](https://huggingface.co/Aclevo/LBNET-2.7B-BASE) 获取。


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1479563801341988937)** (1 messages): 

> `Gradio release, Custom Components Svelte, Performance Optimization, UI fixes` 


- **Gradio 6.9.0 发布！**：Gradio 6.9.0 已上线，带来了全新的修复和 DX（开发者体验）改进；通过 `pip install -U gradio` 进行更新并阅读完整 [更新日志](https://www.gradio.app/changelog)。
- **自定义组件现在更棒了！**：新的 Gradio 版本修复了 **Svelte** 版本不匹配问题以及注解类型的重新加载模式。
- **Gradio 获得重大速度提升！**：内部 API 调用和数据结构已针对 MCP 等进行了优化，设置 `queue=False` 的事件现在应该会 **快 10 倍以上**！
- **Gradio UI 修复！**：修复了填充高度问题，点击示例后提交按钮会恢复显示，且 **gr.Markdown 进度条** 现在表现正常。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1479660473631903937)** (20 messages🔥): 

> `Broken Quiz Grader, API Inference Issues, Agent Implementations` 


- **测验评分器受 API 问题困扰**：一位成员报告测验评分器针对 URL [api-inference.huggingface.co](https://api-inference.huggingface.co/...) 返回 **410 Client Error: Gone**。
   - 他们指出该 API 已不再受支持，且由于后端问题，测验会错误地显示“不正确！”。
- **Agent 实现引发好奇**：一位成员表达了对学习更多如何实现 Agent 的兴趣。
   - 他们没有添加更多细节。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1479564615867306125)** (68 messages🔥🔥): 

> `lm eval harness, gguf, jax, ICML reviews, State Space Model and Neuro-Symbolic Models` 


- **解决 lm eval harness 的 OOM 错误**：一位成员在拥有 **4 GPU**（每块 **96GB**）的机器上实验 `lm eval harness` 时遇到了 **OOM errors**。
   - Gemini 建议添加 `--model_args "pretrained=***,device_map=auto"` 来进行分片配置，但最终解决方案是使用 *"python -m lm_eval ..." 配合 "parallelize=True"*。
- **Compute 大会门票赠送**：一位成员提供了几张下周日/周日在 **San Francisco** 举行的 **Compute conference** 门票。
   - 会议详情见 [compute.daytona.io](https://compute.daytona.io/)，该会议不提供线上直播。
- **Jax 语言与动态计算 Scan**：一位成员在 GitHub 上分享了一个 [项目](https://github.com/Pickleboyonline/maomi/tree/main)，旨在创建一种对 LLM 更友好的语言来编写 JAX 代码，避开 Python/JAX 混合编写中的棘手部分，直接生成 StableHLO 并通过 JAX 的 XLA 进行编译。
   - 另一位成员正在利用 JAX 研究预测编码网络，以实现带有 AD（自动微分）的动态计算 Scan，因此拥有带 AD 的动态计算 *scan* 会非常酷。
- **关于 State Space Model 和 Neuro-Symbolic Models 的思考**：一位成员认为 [LLM 可能会被](https://youtu.be/yUmDRxV0krg?is=V84ln1dJxvMVrfvtdr) State Space Model 和 Neuro-Symbolic Models 所取代。
   - 他分享了一段 **YouTube** 视频来阐述他的观点。
- **ICML 审稿人发泄不满**：审稿人们对投稿质量表示沮丧，包括令人怀疑的方法论和伪造的结果。
   - 那些报告比 Baseline 仅有微小性能提升（*1-3% 数量级*）的论文被视为噪声，或者是特定超参数选择的结果。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1479720697436831907)** (60 messages🔥🔥): 

> `Flow Matching or Diffusion with NeRFs, Video NeRFs, Robustness to Perturbations in Weight Space, Reservoir Compute with Attention, Anomaly Detection System for Windows logs` 


- **NeRF 也开始扩散了！**：成员们讨论了将 **Flow Matching 或 Diffusion 与 NeRFs** 结合的可能性，以及通过将 Latent Spaces 映射到 NeRFs 的权重空间来生成视频的方法。
   - 虽然权重结构缺乏平凡的归纳偏置（inductive bias）且建模动态场景存在困难，但成员们分享了关于 [PixNerd](https://arxiv.org/abs/2507.23268) 和 [hyperdiffusion](https://arxiv.org/abs/2303.17015) 的论文链接。
- **SAM 非常犀利**：一位成员指出 **Sharpness Aware Minimization (SAM)** 可能会增强模型对权重空间扰动的鲁棒性，并有助于 **NeRF** 的表现。
   - 他们提到了使该方法更廉价/无成本的方法，但尚未将这些方法规模化（scale up）。
- **Reservoir Computing 受到关注**：一位成员请求对其结合 **Reservoir Compute 与 Attention** 进行语言建模的 [预印本](https://doi.org/10.5281/zenodo.18903774) 提供反馈，声称其表现优于标准 Attention。
   - 另一位成员指出，性能取决于 **Object-centric Encoder 的质量**，这可能会限制性能上限，尤其是在现实场景中。
- **Windows 日志异常检测引入 ML**：一位成员正在构建一个针对 120 万行（其中 300 条异常）Windows 日志的 **异常检测系统**，考虑使用 iforests、SVMs、LSTMs、AE 和类 BERT 的 Transformers。
   - 有建议提出使用 **预训练模型（如 BERT）** 将事件映射为 Embeddings，并构建动态图进行基于强化学习（RL）的无监督异常检测，并提供了相关论文链接（[论文 1](https://www.semanticscholar.org/p/261696936)，[论文 2](https://arxiv.org/abs/2312.05526)）。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1479611885287379106)** (2 messages): 

> `Innoculation prompting, Finetuning` 


- **Innoculation Prompting 启发 Finetuning**：一位成员发现了关于 [Inoculation Prompting](https://alignment.anthropic.com/2025/inoculation-prompting/) 的论文与 Finetuning 之间的联系，并参考了 [这项相关工作](https://arxiv.org/abs/2510.04340)。
- **对论文的热情**：另一位成员对 [Inoculation Prompting 论文](https://alignment.anthropic.com/2025/inoculation-prompting/) 表达了兴奋之情。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/)** (1 messages): 

qqx02: hi
  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1479555873381814495)** (64 messages🔥🔥): 

> `充值赠送代金券, Kimi Bridge 认证错误, Kimi K2.5 截断错误与 PDF 总结, OpenClaw 问题, Kimi Code API 密钥` 


- **充值赠送代金券的价值**：一位用户询问 **topup bonus voucher**（充值赠送代金券）的运作方式，是作为代金券充入账户，还是该代金券本身就是会在 **90 天**后过期的赠送额度。
- **Moonshot AI 欠用户重复支付的退款**：一位用户反馈在 **20 天前**就因 **double payment**（重复支付）发邮件申请 **refund**（退款），但一直未收到回复。
   - 另一位用户建议通过 [membership@moonshot.ai](mailto:membership@moonshot.ai) 联系支持团队。
- **用户报告 Kimi Bridge 认证问题**：用户讨论了遇到 **Kimi bridge auth** 问题（连接 Kimi 服务器时出现 **401 错误**）。
   - 一名成员指出这需要重新与 Kimi 进行身份验证。
- **Kimi K2.5 截断 PDF 总结**：一位用户报告 **K2.5** 在总结 PDF 文章时因 *system busy error*（系统繁忙错误）中途截断，并寻求解决方法。
   - 该用户补充说，由于资金匮乏无法升级到付费方案。
- **OpenClaw 用户报告问题**：多位用户报告了 **OpenClaw** 最近版本的问题。
   - 一位用户发布了一个相关的 [PR 修复链接](https://github.com/openclaw/openclaw/pull/39881)，该 PR 解决了与 **Kimi tool calls** 处理方式相关的错误。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1479563480804622396)** (18 messages🔥): 

> `道路上的骑行者, 用于图像处理的 AI 工具, DGX Spark, Arc 浏览器, 半自动化 ML/AI 工程/研究` 


- **骑行者不应该在马路上骑车**：一位成员表示，任何神智清醒的骑行者都不应该在马路 *上* 骑车，而他们认为 *飞行完全没问题，因为那里没有汽车*。
- **AI 工具辅助图像处理**：一位成员正在开发一款帮助人们找到合适工具并应用它的产品（用于图像处理），客户对此非常满意。
   - 他们补充说 LLMs 目前仍然有点笨，在未来几年内不会达到人类水平的智能，并表示 *我不希望像 Musk 这样的人控制机器人大军*。
- **DGX Spark 散热问题报告**：一位成员询问大家是否觉得 **nvfp4** 在超低内存带宽下足够好用，以及散热问题和 OS 稳定性报告是否已得到解决，并链接了 [Carmack 关于散热问题的推文](https://x.com/ID_AA_Carmack/status/1982831774850748825)。
- **Arc 浏览器因新方案遭到抨击**：一位成员表示他们非常讨厌 **Arc** 采取的新方案，认为这从一开始就是个坏主意；另一位成员链接了一个 [YouTube 视频](https://youtu.be/dy55iuYtH_I?is=bpEpFXrI7Qqz8qTk) 和 [另一个 YouTube 视频](https://youtu.be/yUmDRxV0krg?is=YJ_9Wak0D3RX2sKvJEPAAAAAAtodd.b.123)。
- **探索半自动化 ML/AI 工程/研究**：一位成员询问使用当前工具进行半自动化 **ML/AI** 工程或研究的经验。
   - 另一位成员回应了关于在 (-infty, infty) 范围内初始化权重，并表示必须 *找到一种利用这些 BIG NUMBERS（大数据）的方法*。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1479568788369313932)** (4 messages): 

> `论文讨论, 新时间表` 


- **论文讨论邀请待开启！**：一位成员询问如何加入周六的论文讨论，并澄清虽然 **world models**（世界模型）不是他们的主要兴趣点，但他们很想了解更多关于该主题的内容。
   - 另一位成员确认 *任何人都可以加入*，并邀请其也参加工作日的讨论。
- **时间表变更打乱讨论时间**：由于采用了新的时间表，一位成员宣布需要调整一些安排，表示某个时间段 *不太行得通*。
   - 他们提到周二和周四可能比较方便，但仍在 *思考周一、周三和周五该怎么办*。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1479644032270471453)** (7 条消息): 

> `Department of Wario, Sand Break Revegetation, AI Opinion in NYT` 


- **DoW 像瓦里奥部门 (Department of Wario)**：一位成员开玩笑说，听到 *DoW* 让他们想到了“瓦里奥部门”。
   - 另一位成员链接了一个 [YouTube 视频](https://youtu.be/c3W1rHYXUnY?is=6fTmL1fxiulmodMf)，并惊呼“没错，对那个小丑的描述相当准确”。
- **防沙林促进植被恢复**：一位成员表示，“那条线只是建立防沙林的计划，以便让植被恢复”。
- **NYT 发表 AI 评论文章**：一位成员链接了一篇 [NYT Opinion 文章](https://www.nytimes.com/2026/03/08/opinion/ai-anthropic-claude-pentagon-hegseth-amodei.html?unlocked_article_code=1.RlA.ugD2.5Y7hyace3AND&smid=nytcore-ios-share)。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1479555509479538892)** (19 条消息🔥): 

> `Subscription Credits, Support Response, Platform Feedback, Sync Icon, Message Editing` 


- **用户遇到订阅点数差异问题**：多位用户报告了 **升级订阅后未发放点数** 的问题，其中一位用户提到通过 Apple Wallet 支付了 **100 欧元**，累计费用超过 **360 欧元**，但仍未收到点数。
   - 用户对 **缺乏客服支持响应** 表示沮丧，一位用户因支付溢价考虑联系其 CC 公司。
- **支持响应能力的担忧**：尽管称赞了平台的潜力，多位用户对电子邮件和 DMs 的 **缺乏支持响应** 表示了严重担忧。
   - 一位用户表示，“完全缺乏支持响应正成为一个主要问题”。
- **管理员提供直接帮助**：管理员在频道中回复了多位用户，要求提供 **电子邮件地址**，并提议将他们的查询升级给支持团队。
   - 一位管理员表示，“请私下与我分享您的电子邮件地址和更多细节，我将帮助您将查询上报给支持团队”。
- **请求同步图标和消息编辑功能**：一位用户请求在平台中添加 **同步图标** 和 **消息编辑** 功能。
   - 他们表示，“我希望他们能制作同步图标和消息编辑功能”。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1479956298236690543)** (9 条消息🔥): 

> `Kaggle Notebooks, Mojo on Kaggle, Colab instructions, Mojo kernels` 


- **Mojo 在 Kaggle 上行不通？**：一位新用户询问如何在 **Kaggle notebooks** 中使用 **Mojo**，希望获得 [GPU puzzles 网站](https://puzzles.modular.com/howto.html#kaggle-notebooks)上提到的 **每周 30 小时的 GPUs**。
   - 他们发现了相互矛盾的信息，暗示 **Kaggle** 不支持 **Mojo**，这在随后的讨论中似乎得到了证实。
- **Colab Magic：推荐做法**：一位成员建议通过 [Mojo on Google Colab 文档](https://docs.modular.com/mojo/tools/notebooks#using-mojo-on-google-colab)中的 **Colab 指南**，在标准的 Jupyter notebook 内核中启用 `%%mojo` magic 命令。
   - 他们指出，虽然存在实验性的 **Mojo kernels**，但它们需要 **Colab** 和 **Kaggle** 托管环境无法提供的更高权限，因此使用 `%%mojo` magic 命令是最佳选择。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1479842927235567617)** (8 条消息🔥): 

> `Docstring Standards, stdlib Documentation, Docstring Parsing, Undefined Behavior Debugging` 


- **Mojo Docstring 标准引发辩论**：一位成员正在推动 **stdlib** 中标准化的 docstring 标头，理由是目前存在不一致和非标准做法，重点关注 2024 年 7 月开启的 [issue #3235](https://github.com/modular/modular/issues/3235)。
   - 该成员还建议探索为函数/变量 docstring 使用模板字符串，以便库作者定义自己的标准；而另一位成员则认为 **文档清理应该是 1.0 版本之前的问题**，因为它会影响语言的易用性。
- **编译器崩溃引发调试建议**：一位成员在执行过程中遇到崩溃并发布了错误消息，特别提到了缺失符号化堆栈追踪错误。
   - 另一位成员建议使用 `mojo build -g2 -O1 --sanitize=address` 来识别 **undefined behavior**（未定义行为），特别是针对内存管理方面。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1479601178600931351)** (11 messages🔥): 

> `Aider 用于 Delphi/Pascal, Claude 幻觉, Opus 版本差异, GPT 5.4 基准测试, Aider 配合远程 Ollama 服务器` 


- **请求在 Aider 中支持 Delphi/Pascal 开发**：一名成员询问是否有人使用 **Aider** 进行 **Delphi/Pascal** 开发，并指出 **Copilot** 没有相同的问题。
   - 他们还询问了关于 **Claude** 幻觉导致修改错误的问题，描述了 **Opus 4.5** 在没有进行实际更改或 Git Commit 的情况下陷入循环的情况，并寻求建议。
- **Claude Opus 4.5 问题持续存在**：一名成员报告称，在尝试让 **Opus 4.5** 实现一个基础功能时浪费了时间，经历了循环显示内容但没有实际修改或 Git Commit 的情况。
   - 另一名成员询问为什么在 **4.6** 可用的情况下仍在尝试使用 **Opus 4.5**；有人指出**价格**差异可能是一个因素。
- **GPT 5.4 基准测试推测**：一名成员询问是否有人已经对 **GPT 5.4** 进行了基准测试。
   - 另一名成员提到在 xthigh 上看到 **79%** 的分数，认为这 *“不知为何相当糟糕”*。
- **Aider 远程 Ollama 服务器设置**：一名成员请求关于将 **Aider** 与远程 **Ollama** 服务器配合设置的指南。
   - 他们怀疑自己拥有的版本可能还不支持远程服务器。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1479749976891588692)** (1 messages): 

> `aider 版本 0.86.2, aider 更新日志` 


- **Aider 发布新版本**：新版本的 **aider** (v**0.86.2**) 已发布，可通过 pip 升级。
   - 升级命令：`E:\Programs\Python311\python.exe -m pip install --upgrade --upgrade-strategy only-if-needed aider-chat`。
- **正在运行 Aider v0.86.1**：用户当前运行的 Aider 版本为 **v0.86.1**。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1480189750035087534)** (1 messages): 

> `终端输出降噪工具` 


- **Context Crunching Python 减少终端噪音**：一名成员创建了一个名为 **Context Crunching Python (ccp)** 的工具，用于减少终端输出的噪音。
   - 目标是改进 Context Window，项目已在 [GitHub](https://github.com/SuppieRK/ccp) 上发布。
- **添加了虚拟主题以满足 minItems=2**：这是一个为了确保 JSON 有效而添加的虚拟主题。
   - 该条目纯粹是为了满足 `topicSummaries` 数组中至少有两个项目的要求。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1479646224985362622)** (1 messages): 

> `前端进展, 内存架构, 评估器, 优化器, Fleet-RLM` 


- **前端开发进展迅速**：前端在质量提升方面取得了进展，目前使用 **Modal Sandbox** 和 **Volume** 进行内存/分析任务，未使用 **Redis** 或 **Vector Store**。
   - 目前的工作重点是内存架构，以及实现合适的 **Evaluator** 和 **Optimizer** 组件。
- **Fleet-RLM 发布**：一名成员分享了他们基于 **DSPy** 构建的框架 [Fleet-RLM](https://github.com/Qredence/fleet-rlm)。
   - 随附的图像展示了其架构的实际运行情况。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1480274645084147744)** (2 messages): 

> `RLM 的要求, 符号对象提示词, 用于 LLM 交互的 REPL 环境, REPL 内部的 LLM 调用` 


- **RLM 的资质要求**：一名成员概述了一个系统必须满足的**三个基本标准**，才能被称为真正的**递归语言模型 (Recursive Language Model, RLM)**。
   - 标准包括：*用户提示词必须是符号对象*、*模型必须通过代码与持久的 REPL 环境进行交互*，以及*代码必须能够在 REPL 内部调用 LLM/RLM*。
- **RLM 中的符号对象提示词**：为了使系统被视为 **RLM**，用户提示词必须是一个**符号对象**，而不是 Transformer 上下文窗口中的一系列 Token。
   - 该成员指出，现有系统通常缺乏这一特征以及其他要求，因此不能完全算作 **RLM**。
- **REPL 环境在 RLM 中的作用**：RLM 的另一个要求是模型必须通过在**持久的 REPL 环境中编写代码**来与符号对象进行交互。
   - 这个 **REPL** 环境是模型执行代码并与系统发生交互的地方。
- **REPL 环境内部的 LLM 调用**：RLM 的一个关键特征是模型编写的代码能够**在 REPL *内部* **调用 **LLM/RLM**，而不是作为一个离散的子 Agent 工具。
   - 该成员表示有兴趣看到整合了所有这三个标准的项目。


  

---

### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1479561301104525332)** (3 条消息): 

> `MCP-I question, auth agent identity, MCP Contrib ecosystem, ANP project` 


- **MCP-I 问题浮现**：一位成员提到收到了关于 **MCP-I** ([链接](https://share.google/aimode/xAik81A0u4WKsjewv)) 的问题，并希望将其整合到 **auth agent identity** 侧，以捕捉实际 **MCP contrib ecosystem** 中的用例。
   - 他们指出，这通常表现为 "XXXXMCP" 或 "MCP - XXXXX" 的模式，但仔细观察后发现与 **MCP** 并不直接相关。
- **MCP-Identity 澄清与 ANP 比较**：一位成员澄清说 **MCP-I** 指的是 **MCP-Identity**。
   - 另一位成员指出，乍一看这听起来与 **ANP (AI Agent Protocol)** ([链接](https://w3c-cg.github.io/ai-agent-protocol/)) 非常相似，并询问这两个项目是否有任何关联。