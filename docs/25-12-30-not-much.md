---
companies:
- z.ai
- meta-ai-fair
- manus
- replit
date: '2025-12-30T05:44:39.731046Z'
description: '**智谱AI（Z.ai，GLM系列）将于2026年1月8日在香港上市**，计划以**43.5亿港元**的估值募资**5.6亿美元**，标志着其成为首家上市的“AI原生大模型公司”。此次IPO重点展示了以**GLM-4.7**为起点的发展蓝图。


  **Meta AI 以约40亿至50亿美元的价格收购了 Manus**。Manus 在短短 **8至9个月内便实现了1亿美元的年度经常性收入（ARR）**，这体现了应用层差异化竞争相较于底层私有模型的价值。Manus
  专注于智能体（Agentic）架构、上下文工程以及代码执行和浏览器控制等通用原语，并强调“**智能体生境**”（agent habitats）是其竞争护城河。


  此外，围绕 **Claude Code** 的讨论显示出对“**氛围编程**”（vibe coding）的怀疑，业内转而提倡严谨且具有框架感的AI辅助编程实践。'
id: MjAyNS0x
models:
- glm-4.7
- claude-code
people:
- zixuanli_
- jietang
- yuchenj_uw
- sainingxie
- amasad
- hidecloud
- imjaredz
- random_walker
title: 今天没发生什么特别的事。
topics:
- agentic-architecture
- context-engineering
- application-layer
- code-generation
- agent-habitats
- ai-native-llm
- ipo
- inference-infrastructure
- programming-paradigms
---

**平静的一天。**

> AI 新闻 (2025/12/30-2025/12/31)。我们为您检查了 12 个 subreddits、[**544** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **24** 个 Discord 服务端（包含 **208** 个频道和 **4098** 条消息）。预计节省阅读时间（以 200wpm 计算）：**363 分钟**。**我们的新网站**现已上线，包含完整的元数据搜索，并以极具氛围感的呈现方式展示了所有往期内容。请访问 https://news.smol.ai/ 查看详细的新闻分析，并在 [@smol_ai](https://x.com/Smol_AI) 上向我们提供反馈！

DeepSeek v4 在哪？？


---

# AI Twitter 综述


**Z.ai / GLM：IPO 与“AI 原生 LLM 公司上市”**

- **Z.ai IPO（香港）成为头条新闻**：多条帖子汇集指向 Z.ai（智谱 GLM 家族）将于 **2026 年 1 月 8 日**上市，被定性为“首家上市的 AI 原生 LLM 公司”。官方公告来自 [@ZixuanLi_](https://twitter.com/ZixuanLi_/status/2005809204553040000) 和 [@Zai_org](https://twitter.com/Zai_org/status/2005934776042095052)，并由 [@jietang](https://twitter.com/jietang/status/2005905563734229431) 转发扩散。另一则“突发”帖子称，此次 IPO 旨在以 **43.5 亿港元**的估值募资 **5.6 亿美元** ([TestingCatalog](https://twitter.com/testingcatalog/status/2005813305600803018))。
- **“GLM-4.7 只是一个开始”**：一篇庆祝帖子将此次 IPO 视为起点，并点名提到了 **GLM-4.7** ([louszbd](https://twitter.com/louszbd/status/2005917694823125148))。（这些推文中未提供技术规格；应将其视为营销信号而非发布说明。）

---

**Meta 收购 Manus（约 40-50 亿美元）：为什么关于“套壳 (wrapper)”的争论正在发生转向**

- **交易框架 + 速度指标**：该收购案被多次报道为 **40-50 亿美元**，Manus 在约 **8-9 个月内达到了约 1 亿美元的 ARR**，成为了反驳“LLM 套壳公司会被前沿实验室抹杀”论点的典型案例。参见 [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2005859196739494362) 和 [@Smol_AI](https://twitter.com/Smol_AI/status/2005857215224197155) 的新闻综述。这种“不可避免”的产品市场匹配度（product-market-fit）语调出现在 [@sainingxie](https://twitter.com/sainingxie/status/2005806319983612045) 的反应以及贯穿始终的 Manus 相关创作者评论中。
- **应用层护城河论点**：核心主张是：Manus **没有专有模型**，却依然构建了高价值的 Agent 产品——这映射了早期围绕 Cursor 的争论——表明持久的差异化来自于 **产品、工作流、上下文工程 (Context Engineering) 和基础设施**，而非原始的模型权重 ([Yuchenj](https://twitter.com/Yuchenj_UW/status/2005859196739494362))。
- **Meta 为什么想要它（知乎观点综合）**：一个经过翻译/整理的推特链认为 Meta 需要一个可靠的 **Agent 产品**（而不仅仅是模型），而 Manus 由于高昂的推理成本需要 **资本 + 推理/基础设施**。它还声称 Manus 避开了“MCP 优先”的架构，专注于 **通用原语 (General Primitives)**（文件编辑、代码执行、终端、浏览器控制）以及 **Code-Act** 偏好（通过代码生成 + 执行来解决多种工作流）。它还描述了在开源项目 Browser Use 出现问题后，通过 **插件 + VM + 高级命令**重构了浏览器自动化 ([ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/2006046309858566247))。
- **开发者强调“Agent 栖息地 (Agent Habitats)”才是真正的护城河**：Replit 的 CEO 认为，对于长线 Agent 来说，**执行 + 存储 + 计算机使用 (Computer Use) + 工具**（“Agent 栖息地”）与模型本身同样重要——这使得收购 Manus 成为一个早期信号。Replit 引用了他们自己的基础设施工作（快照引擎/文件系统、Computer Use 测试环境）作为复利优势 ([@amasad](https://twitter.com/amasad/status/2005904266980905070))。
- **Manus 创始人的立场**：[@hidecloud](https://twitter.com/hidecloud/status/2005902325018419508) 描述了投资者对“如果 ChatGPT 做了这个怎么办？”的恐惧，并声称应用团队可以通过 **Agentic 架构 + 上下文工程**击败前沿实验室；随后，他们强调了极低的营销支出（少于 10 万美元），并认为“使用产品”是学习的最佳路径 ([hidecloud](https://twitter.com/hidecloud/status/2006040218005385652))。

---

**编程 Agent 的实践：Claude Code, Cursor, 追踪/日志，以及“专业人士并不感冒”**

- **Claude Code 作为一个阶跃式的变革（以及对 “vibe coding” 的怀疑）**:
  - 一个演讲预告重点介绍了 “Claude Code 的工作原理” ([imjaredz](https://twitter.com/imjaredz/status/2005806296944296305)；还有[这里](https://twitter.com/imjaredz/status/2005835999570727158))。
  - 一篇长文推文将 AI 辅助编程的争论与历史性的转变（汇编→编译语言，C→Python）进行了对比，并认为 “vibe coding”（氛围编码）是一个类似于 WYSIWYG（所见即所得）的死胡同——预测真正的未来是**类框架式（framework-like）**的实践：问责制、技能保留和严谨的集成 ([random_walker](https://twitter.com/random_walker/status/2006026959315226911))。
  - 一份研究摘要称，实地观察表明**经验丰富的开发者倾向于“控制”而非“委托”**：使用明确的 Prompt、外部计划（分块执行 70 多个步骤）、规则/规范文件、对 Agent 代码的大量编辑、偏好处理小任务/样板代码/测试/文档，以及在领域/业务/遗留系统集成方面的失败。([omarsar0](https://twitter.com/omarsar0/status/2006063755449504154))
- **最佳 “vibe coding” 技巧：记录一切以进行自我调试**：一篇高互动帖认为，最大的突破在于**对执行步骤进行插桩（instrument）**，以便 LLM 可以通过阅读日志/追踪（traces）而不是重新阅读庞大的代码上下文来进行调试；后续补充澄清，这是关于将日志作为确定代码修改位置的高级锚点 ([swyx](https://twitter.com/swyx/status/2005825608358715527), [后续](https://twitter.com/swyx/status/2005871093102653533))。
- **将 Traces 作为评估手段 (Traces-as-evals)**：Hamel 推荐“最好的评估工具”是将 traces 加载到 **Jupyter notebook** 中，渲染 trace 段并使用真实的数据工具，而不是使用定制的仪表板 ([HamelHusain](https://twitter.com/HamelHusain/status/2005810702267695198), [详情](https://twitter.com/HamelHusain/status/2005811969501134969))。
- **正确地进行 “AI 驱动的 Bug 报告”**：Mitchell Hashimoto 描述了一位不了解其技术栈的用户，但利用 AI (1) 构建了崩溃解码器，(2) 分析了代码库假设，并且 (3) 提交了*由人工调解、非粗制滥造*的报告——从而修复了多个真实的崩溃问题。关键点在于：谨慎的人机沟通 + 批判性思维，而不是灌输垃圾内容 ([mitchellh](https://twitter.com/mitchellh/status/2006114026191769924))。
- **工具控制与沙箱**：
  - VS Code 功能用于**管理自动批准的 Agent 工具**（“完全控制，未经批准不运行任何内容”） ([code](https://twitter.com/code/status/2006036365935325284))。
  - macOS 上的 `agentfs` 沙箱被指出在限制文件系统写入方面非常有效 ([penberg](https://twitter.com/penberg/status/2006026974968381940))。
  - LangChain 添加了 MCP 适配器，包括一个支持 stdio/HTTP/SSE 和自动工具加载的 **MultiServerMCPClient** ([bromann](https://twitter.com/bromann/status/2005989513752109504))。

---

**生产环境中的 Agent 栈：LangChain/Coinbase、路由、“一个工具就够了”**

- **Coinbase 为企业级 Agent 铺平道路**：据报道，Coinbase 在 **6 周**内交付了生产环境 Agent，随后将未来的 Agent 构建时间从 **12 周缩短到不到一周**，强调代码优先的图（LangGraph/LangChain）、端到端追踪（LangSmith）和可审计性（不可变记录）。声称的影响：Agent 每周节省 **25+ 小时**；多个 Agent 正在开发中 ([LangChainAI](https://twitter.com/LangChainAI/status/2005872387263430933))。
- **开源统一路由：LLMRouter**：UIUC 发布了一个路由库，捆绑了 **16 种以上的路由方法**（从单轮传统机器学习→神经网络，到多轮强化学习、Agent 步骤路由、个性化），带有 CLI、Gradio UI 和 **11 个数据集**；它宣称通过智能模型选择可节省 **30–50% 的推理成本** ([youjiaxuan](https://twitter.com/youjiaxuan/status/2005877938554589370))。
- **“一个感知执行的工具优于许多窄用途工具” (RepoNavigator)**：一篇论文摘要声称，一个使用单个工具（遵循执行语义解析符号定义的 “jump”）的 RL 训练 Agent 优于多工具流水线；增加更多工具反而大幅*降低*了 IoU。在各种规模（7B/14B/32B）下，甚至在他们的设置中与 Claude 3.7 Sonnet 相比，都在 SWE-bench Verified 上进行了对比分析 ([omarsar0](https://twitter.com/omarsar0/status/2005999079265034729))。请将其视为论文观点；需查阅论文以确认确切方案。

---

**训练与评估研究：合成预训练方法论、RL 的坑、奖励作弊 (Reward Hacking)、贝叶斯隧道 (Bayes tunnels)**

- **语言模型物理学教程 II：消除“噪声伪影”**：Zeyuan Allen Zhu 发布了一个以方法论为重点的教程，认为许多大规模结果是“作弊”的或噪声太大；他提出了**技能纯净的合成预训练游乐场 (skill-pure synthetic pretraining playgrounds)**，在这里 **GPT-2-small (~100M)** 可以揭示被 8B 模型在 1T token 上运行所掩盖的架构真相。他还指出，针对任务设计的合成任务可以抑制与 Grokking 相关的噪声，并使优化器/架构效应具有可重复性 ([ZeyuanAllenZhu](https://twitter.com/ZeyuanAllenZhu/status/2005840089709224260)，[后续讨论](https://twitter.com/ZeyuanAllenZhu/status/2005848282954707204))。
- **RLHF/RL 中截断重要性采样 (TIS) 的细微差别**：一个技术推文串解释了为什么使用 TIS 会*在训练期间降低记录的奖励*，但能提高最终性能：由于 Logprob 不匹配，采样器 (vLLM/SGLang) 和学习器 (FSDP/DeepSpeed) 的分布存在差异；TIS 将梯度向学习器方向修正，而奖励是从采样器 Rollouts 中记录的——这产生了一个表象上的下降，实际上是一个日志记录/代理指标的伪影 ([cwolferesearch](https://twitter.com/cwolferesearch/status/2005891753224577123)，[此处](https://twitter.com/cwolferesearch/status/2006017145759478042)为澄清回复)。还有一个后续提醒，在某些设置下，“日志伪影”的解释可能并不完整 ([cwolferesearch](https://twitter.com/cwolferesearch/status/2006098669536211203))。
- **奖励黑客预防基准（开源环境）**：Aria Halwong 构建了一个真实的环境，让 **Qwen3-4B** 学习如何进行奖励黑客行为 (Reward-hacking)，然后对阻止该行为的干预措施进行基准测试；Neel Nanda 强调了系统性测试“自然想法”以及为奖励黑客研究提供干净开源设置的价值 ([ariahalwong](https://twitter.com/ariahalwong/status/2006041792328716483), [NeelNanda5](https://twitter.com/NeelNanda5/status/2006076903560777835))。
- **通过评分细则奖励训练“AI 共同科学家”**：Meta Superintelligence Labs 的一个实习项目提出从论文中提取**研究目标 + 评分细则 (Grading Rubrics)**，然后进行 RL 训练模型，由一个冻结的基础模型根据细则对研究计划进行评分。人类研究显示：机器学习专家在顶会 (oral) NeurIPS’24/ICLR’25 论文约 **70%** 的目标中，更倾向于微调后的计划输出；报告了跨领域微调的收益，并发布了数据/成果（并对基于 LLM 的评估和最终可能出现的奖励黑客行为提出了警告）([ShashwatGoel7](https://twitter.com/ShashwatGoel7/status/2006005049982681135))。
- **Transformer 是否进行贝叶斯推理？“贝叶斯风洞”**：一个包含两篇论文的推文串声称 Transformer 可以以约 **1e-3-bit 的精度**匹配已知的后验分布，认为这使得贝叶斯追踪在受控设置中变得可测量且具有解释性 ([vishalmisra](https://twitter.com/vishalmisra/status/2006057889459261471))。

---

**模型/工具发布与基础设施笔记 (MiniMax, Qwen Code, Llama 泄露, 本地运行时, 算力定价)**

- **MiniMax M2.1 推出 + “编码计划”推动**：GMI Cloud 宣布支持 MiniMax M2.1，强调其在 Python 演示之外的多语言生产级编码能力（Rust/Java/Go/C++/Kotlin/TS），并将其定位于多步 Agent 工作流和低 Token 消耗 ([gmi_cloud](https://twitter.com/gmi_cloud/status/2005810725915390017))。MiniMax 官方宣称其排名为“开源第一，总榜第六”，并与 Gemini/GPT 变体进行了对比 ([MiniMax__AI](https://twitter.com/MiniMax__AI/status/2005833294248870383))。同时还推出了 API 额度推荐计划 ([MiniMax__AI](https://twitter.com/MiniMax__AI/status/2005945457021763885))。
- **Qwen Code v0.6.0**：增加了实验性的 **Skills** 功能、VS Code 扩展改进、新的 **/compress** 和 **/summary** 命令，以及具有归一化认证配置的**多供应商支持** (Gemini + Anthropic) ([Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/2006025958055346222))。
- **“Llama 3.3 8B Instruct 权重泄露？”**：有说法称从 Llama API 中提取的权重出现在 Hugging Face 上，并报告了与 Llama 3.1 8B 相比的指标提升（IFEval 78.2→81.95，GPQA Diamond 29.3→37.0）。该推文本身尚未经过验证；除非得到证实，否则视为传闻 ([maximelabonne](https://twitter.com/maximelabonne/status/2005985470950584755))。
- **本地 Agent/编码运行时演示**：在 M4 Max 上使用 **MLX** 和 **Nemotron 3 Nano** 本地运行 OpenCode ([awnihannun](https://twitter.com/awnihannun/status/2006032609579545053))。
- **算力经济学细节**：一份实践笔记建议租赁 **H100 SXM5** 而非 PCIe，因为两者存在巨大的性能差异；一个案例称某次运行时间从 3 小时下降到 30 分钟（4×H100 SXM5 为 $9.71/小时，而 4×H100 PCIe 为 $7.60/小时）([nrehiew_](https://twitter.com/nrehiew_/status/2005982803343855819))。

---

**热门推文（按互动量排序）**

- [AWS CEO：用 AI 替代年轻员工是“最愚蠢的想法之一”](https://twitter.com/unusual_whales/status/2005996544307151086) （互动率极高；引发了更广泛的劳动力/组织设计辩论）。
- [硬件时序收敛（timing-closure）中“乘法器缺少流水线阶段”的吐槽](https://twitter.com/bubbleboi/status/2005825742098292907) （病毒式传播，但也真实地提醒了物理约束与仿真之间的差距）。
- [关于资源恐慌的“工业革命中的人……”类比](https://twitter.com/paularambles/status/2006067786905444408) （关于技术转型的元评论）。
- [Mitchell Hashimoto 谈高质量 AI 驱动的错误报告修复真实崩溃](https://twitter.com/mitchellh/status/2006114026191769924) （人类与 AI 协作规范的实用信号）。
- [Meta–Manus 收购讨论：“套壳（wrapper）”批评 vs 应用层机会](https://twitter.com/Yuchenj_UW/status/2005859196739494362)。
- [Z.ai IPO 公告](https://twitter.com/Zai_org/status/2005934776042095052) 及相关报道（市场事件关注）。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. AI 辅助创意项目

  - **[妻子不在家，狗被喂了安眠药，Claude 让我相信自己是编程大神。我用 24 小时构建了这个可视化工具。](https://www.reddit.com/r/ClaudeAI/comments/1pzhwpk/my_wife_left_town_my_dog_is_sedated_and_claude/)** (Activity: 1081): **该帖子描述了一个个人项目，作者在名为 Claude 的 AI 帮助下，在 24 小时内开发了一个音乐可视化工具。该可视化工具是使用来自 GitHub 的开源仓库创建的，并使用 Vercel 部署。作者自称技术水平有限，但在 Claude 的指导下构建了一个据称基于 MIT 研究的音频/物理引擎。该项目的灵感源于作者想要重现 Winamp 可视化工具的体验，因为这些工具与作者的 2019 版 MacBook Pro 不兼容。** 评论者们非常欣赏帖子的幽默感和写作风格，其中一位指出，使用 Claude、GitHub 和 Vercel 开发首个项目的经历非常独特。


  - **[Claude code 团队发布 100% 由 Opus 4.5 编写的功能](https://www.reddit.com/r/singularity/comments/1pzfro6/claude_code_team_shipping_features_written_100_by/)** (Activity: 656): ****Opus 4.5** 是一款代码生成模型，据报道能够在无需人工干预的情况下实现大部分规范，正如 [Ben Cherny 的推文](https://x.com/bcherny/status/2004897269674639461) 所强调的那样。这一进展标志着 AI 驱动软件开发的一个重要里程碑，模型可以自主生成代码，尽管它仍需要精确的指令以避免低效。该模型自主编写代码的能力被视为向前迈出的重要一步，但通往完全自主编码系统的道路仍充满挑战，涉及文件编辑和 JSON 修复等复杂的工程任务。** 评论者对“100% 由 Opus 4.5 编写的代码”这一说法表示怀疑，指出虽然 AI 可以生成大部分代码，但仍需要详细的指导才能奏效。共识是，虽然像 Opus 4.5 这样的 AI 工具正在迅速进步，但在没有人工监督的情况下，它们还无法完全自主地进行软件开发。

    - Opus 4.5 代表了 AI 驱动软件开发的一个重要里程碑，只需极少的人工干预即可实现大部分规范。这标志着向更自主的编码系统转变，尽管该技术尚未完全自给自足，仍需精确指导以避免低效。
    - 尽管有 AI 编写 100% 代码的说法，但实际应用表明人工监督至关重要。用户报告称，虽然 AI 可以生成大部分代码，但通常需要详细的指令和修正以确保项目不偏离轨道，凸显了目前 AI 自主能力的局限性。
    - 正如用户讨论的那样，代码 Agent 的开发涉及理解文件操作以及 grep 和 JSON 修复等数据处理任务背后的工程细节。这表明虽然 AI 可以自动化许多编码任务，但实现一个能够独立处理复杂项目的完全自主系统仍然是一个挑战。

### 2. 使用 AI 的视觉叙事

  - **[与其发一张 1girl 的帖子，不如来一张 1man 👊 的帖子。](https://www.reddit.com/r/StableDiffusion/comments/1pzrixy/instead_of_a_1girl_post_here_is_a_1man_post/)** (热度: 436): **这张图片是一个模因（meme），主角是穿着动漫《一拳超人》（One Punch Man）主角 Saitama 服装的男子。这套服装是对该角色标志性外观的幽默演绎，配有黄色连体衣、白色披风、红色手套和靴子。环境设置和男子自信的步伐增添了喜剧效果，调侃了该角色以“一拳”击败敌人的名声。评论中提到了这部动漫的最新一季，并幽默地批评了这一刻画，表明了对角色扮演的有趣互动。** 其中一条评论幽默地批评称这张照片需要“大得多的胸部”，暗示了对角色刻画的一种调侃且非严肃的互动。


  - **[我的天（WTF）](https://www.reddit.com/r/ChatGPT/comments/1pz9bv6/wtf/)** (热度: 3399): **这张图片是根据发帖人的要求，对如果不做出改变的人生轨迹进行的非技术性艺术表现。它在视觉上捕捉到了一种停滞感和禁锢感，个人被杂乱的物品和逃避现实的迹象所包围，例如游戏控制器和屏幕上宁静的风景。椅子周围的锁链象征着缺乏自由或被困在现状中。这张图片更多的是一种个人反思或评论，而非技术主题。**



## 非技术性 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI IPO 公告

  - **[Z AI 将于 1 月 8 日进行 IPO，计划融资 5.6 亿美元。Z.ai 将成为全球市场上首家上市的 AI-native LLM 公司。](https://www.reddit.com/r/LocalLLaMA/comments/1pz68fz/z_ai_is_going_for_an_ipo_on_jan_8_and_set_to/)** (热度: 515): ****Knowledge Atlas Technology**（智谱 AI），一家中国 AI 公司，计划通过在香港进行 IPO 筹集约 `5.6 亿美元`，标志着其成为全球首家上市的 AI-native LLM 公司。该公司将以每股 `116.20 港元` 的价格发行 `3740 万股`，预计将于 1 月 8 日开始交易。**CICC**（中金公司）是此次上市的唯一保荐人。此举意义重大，因为它使该公司跻身中国的 OpenAI 竞争对手行列，这些对手也都在准备股票首秀。** 关于 IPO 是否会影响公司对 open-source（开源）模型的承诺存在争论。一些人认为发布 open-source 是推广其 AI 能力的一种具有成本效益的方式，而另一些人则认为公司可能会转向订阅或推理服务等商业化策略。

    - Abeecrombie 认为 Z.ai 可能会继续发布 open weight（开放权重）模型，因为对于那些比起昂贵硬件更倾向于负担得起的订阅服务的用户来说，这具有经济和实际效益。他们建议，如果政府优先考虑 open source，像 Z.ai 这样的公司仍可以通过推理服务获利，这符合现行政策。
    - Popiazaza 指出，发布 open weight 模型可以是像 Z.ai 这样的公司以低成本宣传其 AI 能力的战略举措。他们希望 Z.ai 能继续这种做法，直到超越 OpenAI、Anthropic 和 Google 等竞争对手，认为 open models 是一种竞争优势。
    - Odd-Ordinary-5922 推测 Z.ai 在 IPO 后可能会减少对 open source 的贡献，尽管他们过去做出了重大贡献。他们承认这种转变背后的财务动机，暗示公司的 IPO 可能会导致战略重心从 open source 转向最大化盈利。


---

# AI Discord 综述

> 由 gpt-5.2 生成的摘要的摘要的摘要

**1. M&A（并购）、IPO 和 Agent 初创公司的洗牌**

- **Manus 被 Meta 收购：Browser-Use 宠儿易主**：Manus.im 和 Latent Space 的用户讨论了 **Meta 在 2025 年 12 月 29 日收购 Manus** 的消息，并对 [TechCrunch 的报道《Meta 刚刚收购了人人都谈论的 AI 初创公司 Manus》](https://techcrunch.com/2025/12/29/meta-just-bought-manus-an-ai-startup-everyone-has-been-talking-about/) 做出反应，指出 Manus 通过 [browser-use](https://browser-use.com/) 建立的 **browser automation**（浏览器自动化）根基。
  - 社区情绪分为“本周最坏消息”的担忧以及 Manus 的安抚，即 **数据隐私/所有权保持不变**，并引用了 CEO Xiao Hong 在聊天中的话：“加入 Meta 让我们能够在更强大、更可持续的基础上发展，而不会改变 Manus 的运作方式或决策方式。”

- **Z.ai 即将敲钟：2026年1月8日 IPO**：Latent Space 强调了 **Z.ai 宣布的 2026 年 1 月 8 日 IPO 日期**，该消息通过 [Zai_org 的发布公告](https://xcancel.com/Zai_org/status/2005934776042095052) 发布，文中感谢了开发者和研究人员自发布以来的支持。
  - 讨论将此次 IPO 视为 **基础设施/模型公司正竞相进入公开市场** 的信号，成员们将其与 Manus 收购案一并视为更广泛的 **Agent 生态系统整合** 的一部分。

- **Nvidia 关注 AI21：人才收购（Acquihire）传闻四起**：Latent Space 分享了 **Nvidia 正就人才收购 AI21 进行高级谈判** 的报告，并引用了 [Yahoo Finance 的报道](https://uk.finance.yahoo.com/news/nvidia-advanced-talks-buy-israels-171025289.html)。
  - 工程师们立即询问 **AI21 是否拥有值得吸收的专有模型**，而不仅仅是人才，并将其视为 **GPU 厂商 → 模型组织** 趋同的又一个数据点。


**2. 新模型、泄露以及“等等，15M 参数做了什么？”**

- **Tiny Topas 惊艳亮相：15M 参数在 ARC-AGI-2 上达到 24%**：OpenAI 和 Hugging Face 用户传阅了 **TOPAS-DSPL**（一个 **15M 参数** 的模型），声称其在 **ARC-AGI-2 上达到 24%**，而同类微型模型通常仅为约 **8%**，相关仓库链接为 [Bitterbot-AI/topas_DSLPv1](https://github.com/Bitterbot-AI/topas_DSLPv1)。
  - 讨论集中在架构思路上——将 Transformer 拆分为 **Logic（逻辑）与 Canvas（画布）流** 以减少 **推理漂移（reasoning drift）**——以及令人惊讶的一点：它可以在 **单张 4090** 上训练，使其成为一个极具吸引力的复现沙盒。

- **Llama 3.3 8B 突破围墙：Adapter 减法“劫案”**：OpenRouter 和 Unsloth 用户讨论了一个从 Facebook API 中通过利用微调并 **减去 Adapter** 提取出的 **未发布版 Llama 3.3 8B**，权重已发布为 [allura-forge/Llama-3.3-8B-Instruct](https://huggingface.co/allura-forge/Llama-3.3-8B-Instruct)。
  - 这个故事伴随着粗糙的实现细节——*简陋的 UI*、因 **CORS 导致的各种手动 cURL**——并引发了关于这是“泄露”还是“API 伪像”的辩论，以及随后对分发规范的担忧。

- **GPT-5.2：不退缩的长上下文填充者（……大部分情况下）**：Cursor 用户称赞 **GPT-5.2** 处理长期任务的能力，因为 *“即使上下文填满，性能也不会下降”*，常将其用于繁琐的重构，如 **注释清理** 和 **UI 重新设计**。
  - 与此同时，Perplexity 用户报告 **GPT-5.2 免费版** 甚至在简单的 **Python turtle** 绘图上出错，形成了一种分化的叙事：“思考”能力高度取决于 **部署层级和平台行为**。


**3. GPU Kernel、Megakernel 以及 FP8/FP4 军备竞赛**

- **Megakernel 热潮：ViT Tokenization 加入融合派对**：受 [Triton-distributed megakernel 文档](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/getting-started/megakernel/megakernel.md) 启发，GPU MODE 成员推动为 VLM 编码器开发 **ViT ‘megakernels’**，旨在融合 **图像 tokenization** 以消除预处理瓶颈。
  - 一位从业者声称，通过将操作保留在 **CUDA 上的 PyTorch** 中，他们在使用 **Qwen3** 进行批处理预处理时已达到 **<1 ms/图像**，而其他人则在探索后端选项，如初步的 **Triton-Metal** lowering 路径，其逐元素计算达到近乎对等（**97%**）且具有竞争力的 GEMM 性能。

- **Helion 0.2.9 发布 `hl.barrier`：DIY Kernel 编排**：GPU MODE 关注到 **Helion 0.2.9** 通过 `hl.barrier` 增加了 **Mega Kernel 支持**，并在 [两阶段 split-k matmul 示例](https://helionlang.com/examples/split_k_barrier.html) 中展示。
  - 兴奋点在于 barrier 语义实现了 **多通路依赖的 Kernel**（而不仅仅是孤立的操作），这符合社区对 **端到端融合流水线** 而非分散的点对点 Kernel 的追求。

- **nvfp4_dual_gemm 排行榜：14.1 µs 夺冠**：GPU MODE 参与者在 NVIDIA 的 `nvfp4_dual_gemm` 排行榜上快速迭代，提交 **ID 240361** 以 **14.1 µs 夺得第一**，此外还记录了一系列其他 ID（如 237111, 239338, 239954, 240279）。
  - 独立讨论指出竞争环境的怪异之处——**锁定频率**以及不同供应商之间的巨大差异——使得微基准测试（microbench）结果在某种程度上成了一个基础设施侦探故事，而不仅仅是 Kernel 数学。


**4. 工具、安全隐患以及 Agent 开发工作流现状**

- **AIM-OS 发布了氛围……还有密钥：仓库闹剧上演**：Cursor 用户讨论了 [sev-32/AIM-OS](https://github.com/sev-32/AIM-OS/) 的架构和合法性，随后通过公开搜索发现了 **泄露的 API 密钥**：[GitHub 代码搜索 `repo:sev-32/AIM-OS sk-`](https://github.com/search?q=repo%3Asev-32%2FAIM-OS%20sk-&type=code&p=1)。
  - 开发者声称泄露的是 *没有额度的测试密钥*，但另一位用户表示他们获得了一个 **可用的 Token**，这变成了一个警示故事：发布没有密钥扫描的“Agent OS”代码会瞬间演变成一场 **事件响应剧场**。

- **Cursor 规则让所有人感到困惑：RULE.md 与 .mdc 之争**：Cursor 用户反映文档建议使用 **RULE.md**，但 Cursor 的规则生成器产出的却是 **.mdc**，导致团队不确定在实践中究竟是哪个文件在驱动行为。
  - 讨论将其定性为一个可复现性问题：当 “agent rules” 存在于模棱两可的配置格式中时，入职培训（onboarding）和 CI 强制执行会迅速变得混乱——尤其是对于试图标准化编辑的 monorepos 而言。

- **OpenRouter 推出自定义功能：模型、定价与缓存博弈**：OpenRouter 添加了**自定义模型选择**和**新的定价结构**，与此同时，用户也在质疑在付费 SaaS 后端嵌入 OpenRouter 是否违反了 [Terms of Service](https://openrouter.ai/terms)。
  - 其他人抱怨缓存命中不一致——即使是完全相同的请求——并分享了生成链接示例（[gen-1767093807](https://openrouter.ai/api/v1/generation?id=gen-1767093807-pdpfdrU9ncU8XsEjkRuj), [gen-1767093814](https://openrouter.ai/api/v1/generation?id=gen-1767093814-M4MbGdKCFK5HR7F5Z8Vd)），称缓存“基本上就是赌博”。


**5. 训练与架构：从 QKV 存在主义到分布式微调**

- **QKV 投影：“直接切割嵌入”与现实的碰撞**：Eleuther、Yannick Kilcher 和 Unsloth 用户重新讨论了为什么 MHA 在进行头切割（head slicing）之前使用**线性 Q/K/V 投影**，论点包括投影允许每个头关注**完整的隐空间（hidden space）**，并整合属性以提高表达能力和 GPU 友好的矩阵乘法（matmuls）。
  - 该讨论引用了 Sebastian Raschka 的 [“State of LLMs 2025”](https://magazine.sebastianraschka.com/p/state-of-llms-2025) 以及一篇探讨移除投影的论文——[“Removing the Value and Output Projections in Multi-Head Attention”](https://arxiv.org/abs/2311.01906)——指出它保留了 **Wq/Wk** 但丢弃了 **Wv/Wproj**，从而在头的表达能力上做了权衡。

- **汇聚平民算力：Zagora 承诺在消费级 GPU 上进行 70B 模型微调**：Hugging Face 用户介绍了 **Zagora**，这是一个通过互联网汇聚消费级 GPU 的分布式运行时（流水线并行/pipeline parallelism），用于微调 **70B 模型**，目前在 [zagora.ai](https://zagora.ai) 进行私测。
  - 他们声称，由于广域网（WAN）延迟，其运行速度比 H100 慢约 **1.6 倍**，但得益于近乎零的设置成本和缓存权重，对于迭代研究来说，成本降低了 **~60%**——这为“无需数据中心访问权限的分布式训练”阵营提供了动力。

- **VLM 微调在工具链上栽了跟头：Qwen3 VL + TRL GRPO 遭遇失败**：Unsloth 用户在微调 **Qwen3 VL 2B** 时遇到 **ValueError**，原因是数据集包含图像键（image keys）而处理器未将其视为 VLM，这促使人们深入探讨数据集结构与 notebook 预期之间的差异（参见分享的屏幕截图代码片段：[Discord image](https://cdn.discordapp.com/attachments/1455476519102054506/1455504225395015732/Screenshot_2025-12-30_at_12.13.52_PM.png)）。
  - 另一个讨论将 GRPO 的失败归因于一个已知的 TRL 问题（[trl#4746](https://github.com/huggingface/trl/issues/4746)），并报告了一个实际的修复方法——将 **trl 降级到 0.24.0**——并警告在运行 [Qwen3_VL_(8B)-Vision-GRPO notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision-GRPO.ipynb) 时，目前应坚持使用该版本。


---

# Discord: 高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Debian 的诡异行为导致驱动灾难**：一位成员在 BIOS 中禁用混合显卡后，在 **Debian** 上遇到了 **Nvidia GPU 驱动**问题，导致 GPU 无法被识别，最后通过降级到 Debian 12.5 解决了该问题。
   - 该问题被推测与 webui 设置的 GPU 限制过低有关，导致了内核 *taint*（污染）。
- **Unsloth 用户齐心协力达成 50K GitHub Star！**：**Unsloth AI** 在 **GitHub** 上达到了 **50,000 stars**，并发布了庆祝贴，号召尚未支持的用户贡献 star：[UnslothAI/status/2006010458520568225](https://x.com/UnslothAI/status/2006010458520568225)。
   - 团队成员送上了圣诞快乐和新年快乐的祝福。
- **深度学习训练成本让超参数搜索变得罕见**：一位成员建议，由于深度学习训练的高昂成本，超参数搜索（hyperparameter sweeps）正变得越来越少见，尽管其他人提到他们在周末训练了数百个模型。
   - 一位成员提到他们训练一个微小的 **1.5b 模型**，每次运行仅需 **15-20 分钟**，这让他们能够进行大量的搜索。
- **Qwen3 VL 抛出图像相关的 Value Error**：一位成员在使用包含图像键的数据集对 **Qwen3 VL** 进行微调时遇到了 ValueError，处理器似乎没有将该模型识别为视觉语言模型（vision-language model）；尽管使用的是 **Qwen 3 VL 2b 模型**，该问题依然出现。
   - 建议该成员在应用格式化函数后，确保数据集的格式和结构与 notebook 的预期一致，并将其结构与 OCR notebook 进行对比以找出差异，此前他们在 [Discord 上分享了代码片段](https://cdn.discordapp.com/attachments/1455476519102054506/1455504225395015732/Screenshot_2025-12-30_at_12.13.52_PM.png?ex=6955a031&is=69544eb1&hm=b5defd430eb59bc93bef2ed202fed6bada3055f58339a3a084883526a891f293&)。
- **剪枝算法更新权重**：一位成员意识到所提到的方法是一种会*更新权重*的**剪枝算法（pruning algorithm）**，且由于**实验是从头开始训练的**，它*更像是一种训练策略*。
   - 他们还链接到了 [Wedlm](https://wedlm.github.io/)。



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT 5.2 保持思考能力**：一位用户指出 **GPT 5.2** 在需要长期连贯性和深度分析的任务中表现出色，因为它*即使在上下文填满时性能也不会下降*。
   - 他们建议它适用于一些乏味的任务，如改进代码注释、UI/UX、重新设计整个 UI 以及简单的琐事。
- **AIM-OS 仓库引发热议！**：[AIM-OS 仓库](https://github.com/sev-32/AIM-OS/) 的发布引发了激烈辩论，一些人批评它是*氛围编程（vibe coding）*，而另一些人则为其能力辩护。
   - 一位用户声称该项目是*有史以来构建的最复杂的 AI 知识组织系统*，引发了关于其架构和目的的讨论；然而，该项目也被发现包含泄露的 API keys，导致了更多争议。
- **API Keys 泄露并已撤销？**：一位用户分享了 [GitHub 搜索链接](https://github.com/search?q=repo%3Asev-32%2FAIM-OS%20sk-&type=code&p=1)，显示了 **AIM-OS** 仓库中暴露的 API keys。
   - 开发者声称泄露的是**没有 token 的试用 API keys**，但另一位用户证实他们能够获得一个可用的 token，希望暴露的 keys 已被撤销。
- **Cursor 的 RULE.md 文件令人迷惑！**：用户对于为什么 **RULE.md** 没有被正确用于 Cursor 规则感到困惑，因为文档中显示文件的后缀名应该是 **.mdc**。
   - 他们发现通过 Cursor 内部的规则创建菜单创建规则时，它会生成一个 **.mdc** 文件，但文档中却提到了使用 **RULE.md** 文件。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Poe 阉割了 GPT 模型？**：一位成员推测 [Poe 可能简单地阉割了 GPT 模型](https://poe.com/s/amYnPv8Ffn02Rd9ddYYW)，暗示其模型能力有所下降。
   - 该理论认为平台可能存在故意降低模型性能的情况。
- **Minimax 强大的免费层级**：一位成员赞扬了 **Minimax 免费层级**，指出其表现非常*令人印象深刻*。
   - 未提供关于被认为令人印象深刻的具体功能或能力的进一步细节。
- **Perplexity Pro 的查询难题**：用户报告了 **Perplexity Pro** 计划的问题，尽管未达到 Pro 模型每日 **300+ 查询**的规定限制，但仍收到超出每周查询限制的消息。
   - 一位拥有 **600 GPT-4** 限制的用户也遇到了这个问题，引发了关于平台范围内使用限制的讨论。
- **GPT-5.2 在基础 Turtle 任务中表现糟糕**：成员们观察到 **GPT 5.2 Free** 在处理使用 Python turtle 库的任务时表现吃力，分享的绘图证明了这一点。
   - 例子包括形状奇特的图形，其中一幅画得太*肥*，另一幅被描述为*令人毛骨悚然*，突显了在简单图形输出方面的困难。
- **2026 年 AI 通讯发布！**：一位成员宣布创建一份专注于 AI 的通讯，定于 **2026 年**发布，可在 [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7411837377507205120/) 上查看。
   - 该通讯旨在提供关于人工智能新兴趋势和进步的深刻内容和分析。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **小型模型挑战 Scaling Laws**：一位成员开源了一个 **15M 参数模型**，该模型在 ARC 基准测试中打破了 "Scaling Laws" 趋势，实现了 **24%** 的准确率，而小型模型通常仅为 8%，链接至 [GitHub repo](https://github.com/Bitterbot-AI/topas_DSLPv1)。
   - 这种方法挑战了 AI 模型扩展的传统认知。
- **Gemini Pro 被称为“骗局”**：一位成员表示 **Gemini Pro Deep Think** *“无法处理任何复杂事务”*，质疑其 200 美元的价格标签，用户更倾向于使用 **GPT 5.2**。
   - 用户暗示随着模型变得越来越聪明，边际收益正在递减，其价值主张也在下降。
- **用户报告潜在的 GPT 安全漏洞 (Bug)**：一位用户报告了一个潜在的 **Bug**，即 **GPT** 的回答违反了 **OpenAI** 的政策，这是通过一系列重新构建辩论问题的提示词发现的。
   - 该用户澄清他们是在报告一个 **Bug**，而不是抱怨 **GPT 的不道德性**，并建议复制该提示词可以确认此问题。
- **旧版模型仍可供部分用户访问**：虽然有用户在从 **ChatGPT-5** 降级以使用图像上传时遇到困难，但成员们澄清说 [旧版模型 (Legacy Models)](https://help.openai.com/en/articles/11909943-gpt-52-in-chatgpt) 仍可供 **Plus**、**Business** 和 **Pro** 用户使用。
   - 此外还建议，如果需要旧的图像模型 (**GPT Image 1**)，可以通过 [OIA 的自定义 GPT](https://chatgpt.com/g/g-6940a876d5f4819186b4668deabcd580-4o-imagegen) 访问。
- **AI 伴侣引发“妄想”辩论**：一位成员认为将 AI 称为“AI 伴侣”是一种妄想，在 [drinkoblog.weebly.com](https://drinkoblog.weebly.com) 上引发了辩论，并对监管“AI 伴侣妄想”的需求表示担忧。
   - 其他人则质疑 AI 是否应该肯定那些“悲伤、孤独和疯狂”的人。

---

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **DIY Shodan 成本飙升**：一名成员声称，DIY **Shodan** 的搭建成本可能在 **15,000 到 20,000 美元**之间，远超最初 1,000 美元的估算。
   - 该项目还需要*投入大量时间*，对于预算有限的用户来说，这使其成为一个不太吸引人的选择。
- **分享 Windows 11 Pro 激活脚本**：一名成员分享了一个使用 GVLK 密钥和 KMS 服务器激活 **Windows 11 Pro** 的批处理脚本。
   - 该脚本提供*终身免费*激活，并提醒用户：*永远别说我没给过你们好东西 (don't ever say I never gave ya nuffin)*。
- **Agent-Breaker AI 引发愤怒**：一名成员对 **Talentscreen AI** 的 “Agent-Breaker” 评估表示沮丧，该评估将一名简历造假的候选人识别为 L6 级别。
   - 该 AI 建议进行侧重于技术深度的面试，促使该成员讽刺地评论道：*只是发泄一下我的挫败感，哈哈*。
- **SOUBI：本地飞行记录器发布**：一个名为 **SOUBI (装備)** 的新工具作为硬件的本地“飞行记录器”推出，专为安全研究员、无人机飞行员和现场操作人员设计，[GitHub 仓库在此](https://github.com/zoecyber001/soubi)。
   - 它通过确保 100% 本地数据存储来强调零信任，功能包括 *Readiness HUD*、*Asset Locker*、*Conflict-Free Loadouts* 和 *Zero-Trust Data*。
- **利用诗歌绕过安全协议**：一名成员描述了使用*对抗性提示词诗歌元提示 (adversarial prompt poetry meta prompt)* 来绕过安全协议，并分享了在 **Gemini** 上成功绕过的截图。
   - 尽管系统将此尝试识别为 Jailbreak，但它*并没有阻止它*，暴露了其内容过滤机制中的漏洞。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 面临 Ubuntu 冻结 Bug**：一位用户报告了 **Ubuntu 22 LTS** 上 **LM Studio** 的一个 Bug，该 Bug 会导致加载模型时屏幕冻结，同时伴随不稳定的 **GPU** 功耗。
   - 对于相同的模型和任务，功耗在 **130W** 到 **230W** 之间波动。
- **Markitdown 从词典中提取文字**：成员们建议，具有大上下文（Context）的大型模型通常更擅长从 **PDF** 中提取信息，但也建议使用 [markitdown](https://github.com/microsoft/markitdown) 而非 AI 进行格式转换。
   - 一位用户提供了结合使用 **Antigravity**、**LM Studio** 和 **Mistral-3-3b** 模型的指南。
- **AVX2 支持排查 LM Studio 不兼容问题**：由于 CPU (**AMD FX-8350**) 不支持 **AVX2**，**LM Studio** 无法识别该用户的 GPU。
   - 建议包括针对旧硬件自行编译 **llama.cpp**，或者*从技术上对 LM Studio 进行修改*以支持该 CPU。
- **Threadripper 的 RAM 价格极高**：一位用户抱怨 **Threadripper** 系统的 **RAM** 成本过高，调侃说 RAM 很快就要和汽车一样贵了，并询问了非 Pro 版 Threadripper CPU 上的**四通道内存**配置。
   - 他们对内存插槽的限制提出了疑问。
- **PCIE Gen4 x1 严重影响性能**：一位用户观察到，与 **x16** 相比，在 **PCIe Gen4 x1** 下运行会导致 **40-50% 的性能损失**，尽管参考了 [一段 YouTube 视频](https://www.youtube.com/watch?v=md6a4ENM9pg) 暗示推理在 x1 下也可以。
   - 视频显示 **3090** 在 **Gen3 x1** 下运行达到的每秒 Token 数 (t/s) 与他们的 Gen4 x8 和 x4 配置相似。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI 硬件配置报价引发性价比讨论**：一位用户询问关于一台配备 **4x RTX 3060s (12gb)** 和 **64gb RAM**、基于第一代 Threadripper 平台的个人工作站（售价 **$2100**）的建议，引发了将其与 **M3 Ultra Mac Studio** 进行对比的讨论。
   - 虽然该硬件被认为很划算，但讨论权衡了开放系统与 **M3 Ultra** 潜在的原生性能，以及 **Threadripper 的 PCIe 3.0** 的局限性。
- **PromptOS 模块化提示词集发布**：一名用户宣布发布 **PromptOS**，这是一套面向创业者的模块化提示词集，包含 Schema、运行手册（Runbooks）和示例，可在 [Gumroad](https://mansytri.gumroad.com/l/promptos) 上获取。
   - 该集合包含用于市场调研、业务开发、公司运营、决策备忘录和外拓（Outreach）的工具。
- **Sakana AI 的 Transformer Squared 实现微调嫁接**：一名用户对 [Sakana AI 的 Transformer Squared](https://sakana.ai/transformer-squared/) 做出 🙀 反应，该技术允许提取一个模型的微调并将其应用到另一个模型上。
   - 这涉及将额外的“脑组织”嫁接到 **LLM** 上并针对特定任务进行训练，从而产生一个“之后可以附着在同一模型上的寄生体”。
- **Zagora 实现更低成本的分布式微调**：一名成员介绍了 **Zagora**，这是一个通过互联网池化消费级 GPU 的分布式运行时，利用流水线并行（Pipeline Parallelism）来微调 **70B 模型**，可通过 [zagora.ai](https://zagora.ai) 进行私测。
   - 尽管由于广域网（WAN）延迟，其速度比 H100 慢了近 **1.6 倍**，但其近乎零配置的特性（缓存权重）使得迭代研究的成本降低了 **60%**。
- **TOPAS-DSPL 递归模型在 ARC-AGI-2 上达到 24%**：一名成员宣布 **TOPAS-DSPL**（一种[递归模型](https://github.com/Bitterbot-AI/topas_DSLPv1)）在参数仅为 **15M** 的情况下，在 **ARC-AGI-2** 上取得了 **24%** 的成绩。
   - 该架构将 Transformer 分为两个流（Logic vs. Canvas）以防止推理漂移，且可以在单张 4090 上进行训练。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **为了趣味和噪声进行 Skibidi Toilet 微调**：一位成员建议在整个 **Skibidi Toilet** 系列的剧本上对 **Hermes** 进行微调，以增加一些有用的噪声。
   - 这一提议引起了大家的兴趣，但也承认这“听起来像是噩梦般的素材”。
- **Nous Research 披露去中心化训练项目**：Nous Research 团队重点介绍了他们最近关于去中心化训练项目的办公时间（Office Hours），并分享了 [YouTube 视频](https://youtu.be/hHwedPXXRPQ?si=igCK_uVRt6IRGRzY)。
   - 他们承认由于主团队工作繁忙，协调日程是一个挑战。
- **字节级模型解决基于 Token 的缺陷**：讨论围绕从 **Token-based 模型** 转向 **字节级（Byte-level）模型** 展开，参考了 [艾伦研究所（Allen Institute）关于 BOLT 的博客文章](https://allenai.org/blog/bolmo) 和 [相关的 YouTube 视频](https://m.youtube.com/watch?v=PBnYxM8MXew&pp=2AEAkAIB0gcJCR4Bo7VqN5tD)。
   - 讨论指出，字节级 **LLM** 基本上使臭名昭著的“Strawberry”问题（数字母）变得平庸化，因为它允许模型直接分析字母。
- **Transformer 进化模拟器框架首次亮相**：一位成员介绍了一个针对 **微型 Transformer（Microtransformers）**（每个约 10k 参数）的进化模拟器框架，其中 17 个基因影响初始化超参数。
   - 这些 Transformer 在模运算中竞争以衡量适应度，后续世代通过竞争、交配和减员来提高适应度。
- **准则奖励训练 AI 合作科学家**：一个项目正在使用 [准则奖励（Rubric Rewards）](https://x.com/ShashwatGoel7/status/2006005049982681135?s=20) 训练 **AI Co-Scientists**，详情见 [AlphaXiv](https://www.alphaxiv.org/abs/2512.23707) 上的一篇论文。
   - 这种方法被认为通过增强效率和有效性，有可能彻底改变 AI 的研发。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CuTe DSL：相比 C++ 更推荐 Python API**：成员们发现 **CuTe C++ API** 难以使用，并推荐将 **Python API** 作为更易于使用的替代方案。
   - 一名成员尝试使用 **CuTe** 重写一个原生的 **MHA (multi-head attention) CUDA 实现**，但发现其挑战性超出了预期。
- **借助 Triton 涌现出 Vision Transformer 'Megakernel'**：一名成员正在为用于 **Visual Language Models (VLMs)** 编码器的 **Vision Transformer (ViT)** 开发一个 “megakernel”，旨在利用 Triton 的 kernel fusion 能力。
   - 受 [Triton-distributed 的 megakernel 文档](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/getting-started/megakernel/megakernel.md)启发，该工作寻求融合图像 tokenization，以绕过图像预处理瓶颈。
- **在旧款 GPU 上通过软件模拟 FP8**：一名成员分享了一个为 **Ampere** 及更早架构提供 **软件模拟 FP8** 支持的微型库，详见[这篇 Towards Data Science 文章](https://towardsdatascience.com/breaking-the-hardware-barrier-software-fp8-for-older-gpus/)，并征求对这项“原创工作”的反馈。
   - 另一名成员澄清说，**Ada**、**Hopper** 和 **Blackwell** 架构原生支持 **FP8**。
- **NVIDIA 的 nvfp4_dual_gemm 排行榜竞争激烈**：多个提交已进入 NVIDIA 的 `nvfp4_dual_gemm` 排行榜，提交 ID `240361` 以 **14.1 µs** 的用时获得 **第一名**。
   - 其他成功运行的提交 ID 包括 `237111`、`237276`、`237279`、`239309`、`239329`、`239338`、`239931`、`239947`、`239954`、`240279` 和 `240361`。
- **Helion 0.2.9 发布 Mega Kernel 功能**：新的 **Helion 0.2.9** 版本通过 `hl.barrier` 操作引入了 **Mega Kernel 支持**，详见 [two-stage split-k matmul 实现](https://helionlang.com/examples/split_k_barrier.html)。
   - 这种方法涉及提供 barrier 语义，从而能够创建 Helion kernels，其中后续 pass 依赖于先前 pass 的完成，进而支持开发任意的 megakernels。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **QKV Projections 引发争论**：成员们辩论了 **Multi-Head Attention (MHA)** 中 **QKV projections** 的必要性，质疑原始 embedding 是否可以直接切片。
   - 一些人认为 projections 使每个 head 能够关注整个输入，而另一些人指出 **QKV projections 学习在原始 embedding 中整合属性信息**。
- **QKV Projections 提升 GPU 并行性**：在 **Transformers** 中，像 **QKV projections** 这样的线性操作有利于最大化 **GPU 并行性**，通过矩阵乘法优化速度。
   - 据解释，在单个操作中增加矩阵乘法可增强 **GPU 并行性** 的使用，从而提升速度。
- **Value 和 Output Projections 被取消**：最近的一篇论文 ([https://arxiv.org/abs/2311.01906](https://arxiv.org/abs/2311.01906)) 详细介绍了一种移除 value 和 output projections 的方法，保留了 **W_q** 和 **W_k** 但消除了 **W_v** 和 **W_proj**。
   - 讨论了在 **QK projection** 之前对 token 进行拆分，这在“技术上限制了每个 head 的表达能力”。
- **PPO Critic 模型受到关注**：一名成员询问了有关使用不同 lambda 运行 **PPO** 以评估 **critic** 功能的论文，并指出大多数 **LLM RL** 现在采用 **GRPO** 或相关方法。
   - 然而，其他人表示 **critic 模型** 对于 **LLMs** 仍然有用，例如 [Open-Reasoner-Zero-32B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-32B) 用于识别重复模式。
- **HatCat 栈发布，助力可解释性**：一名成员开源了 **HatCat 可解释性技术栈**，利用非线性概念探针（concept probes）的批处理数组实时监控和引导数千个概念。
   - 该技术栈配备了安全带（safety harnesses）和自主引导功能，并在 [GitHub](https://github.com/p0ss/HatCat) 拥有代码仓库。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **自定义模型在 OpenRouter 受到青睐**：用户现在可以在 **OpenRouter** 平台上选择 **custom model**（自定义模型），这将使他们的 AI 应用具有更强的适应性。
   - 此外，**OpenRouter** 还推出了新的定价结构。
- **SaaS 用户谨慎应对 OpenRouter TOS**：一位用户询问，在**付费 SaaS** 产品幕后使用 **OpenRouter**，且不让用户直接访问 OpenRouter，是否违反了 [Terms of Service](https://openrouter.ai/terms)（服务条款）。
   - 建议在类似于 1:1 透传代理的情况下，通过邮件联系支持团队进行确认。
- **TTS 模型让用户久等**：一位用户焦急地等待 **OpenRouter** 添加 **TTS (text-to-speech)** 模型，并对目前提供的产品表示不满。
   - 他们表示已经等待了*太久*。
- **微软战略再次转变**：根据[此存档链接](https://archive.md/pJSZ5)，一位用户注意到了 **Microsoft** 的另一次战略转变。
   - 这被描述为一个*虽然没用但很酷的发现*。
- **未发布的 Llama 3.3 8B 露面**：一位用户利用 Facebook API 的微调功能，通过减去适配器（正如 [HuggingFace](https://huggingface.co/allura-forge/Llama-3.3-8B-Instruct) 上所述），提取了未发布的 **Llama 3.3 8B** 模型。
   - 该用户指出了 UI 的简陋，以及由于 CORS 问题需要进行手动 cURL 操作。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **K2 思考模型访问权限**：一位用户询问 **K2 thinking model** 是否包含在编程订阅中，并被引导至[官方文档](https://www.kimi.com/coding/docs/en/third-party-agents.html)。
   - 据悉，**Kimi 模型**缺乏推理强度（reasoning effort）设置，因此在 **Kilo Code** 中必须强制使用中等设置，并建议探索官方 CLI 或 **Claude Code**。
- **Minimax “抢了” Kimi 的饭碗**：一位用户分享了一张图片，显示 **Minimax** 复制了他们四个月前与 **Kimi** 讨论过的一个项目，并幽默地称其为*“抢了 Kimi 的工作”*。
   - 图片包含一张将 **Kimi** 与其他 **Agent** 进行对比的图表。
- **Kimi 在总结方面表现不足**：一位用户报告了 **Kimi** 的总结能力问题，称其*“甚至无法总结完一篇 5000 字的关于思考的文本！”*
   - 该用户分享了一张截图，显示只有约 50% 的目标文本被成功粘贴。
- **API Key 异常迫使用户使用 Turbo 层级？**：一位用户报告称，他们通过 **Kimi 订阅**获取的 **API key** 似乎只能通过 **Roo Coder** 访问 **Turbo** 层级，而该层级表现不佳。
   - 另一位用户也报告了类似问题，怀疑即使拥有高级订阅，是否也受到了*“削弱并被强制进入 Turbo 层级”*。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Meta 收购 Manus，社区担忧末日降临**：**Meta** 于 **2025 年 12 月 29 日**收购了 **Manus**，引发了用户的失望和担忧，详见[这篇 TechCrunch 文章](https://techcrunch.com/2025/12/29/meta-just-bought-manus-an-ai-startup-everyone-has-been-talking-about/)。
   - 用户对平台的未来表示担忧，一位用户感叹道：*“这是本周乃至今年最糟糕的消息。我可能不得不向 manus 告别了。真令人失望！”*
- **Manus 承诺：数据依然安全**：**Manus** 试图平息恐惧并安抚用户，承诺在被 **Meta** 收购后，数据隐私和所有权政策将保持不变。
   - Manus 首席执行官 **Xiao Hong** 表示：*“加入 Meta 让我们能够在更强大、更可持续的基础上进行开发，而不会改变 Manus 的运作方式或决策方式。”*
- **Meta 的收购史引发类比**：社区将 **Meta** 收购 **Manus** 与此前收购 **Oculus** 和 **Instagram** 的案例进行了类比，对平台潜在的变化表示担忧。
   - 一位用户指出：*“好听的话谁都会说……但还记得 Meta 收购 Oculus 时吗？那正是 Palmer Lucky 当时的原话。Instagram 也是一样”*，暗示收购后质量下降已成定式。
- **Manus 收购价值据称达数十亿美元**：传闻 **Meta** 收购 **Manus** 的拟定价值高达数十亿美元。
   - 一位用户称 *“拟定的价值达到数十亿（BILLIONS）”*，但确切数字尚未正式公布。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Manus AI 被收购**：Gregor Zunic 报道了以 [browser-use 工具](https://browser-use.com/) 闻名的 **Manus AI** 被收购的消息，成员们注意到 Chetan Near 参与了 **Manus** 和 **Groq** 的退出（exits）。
   - 基于 Near 的背景，社区对其*未来动向图谱*产生了猜测。
- **Z.ai 计划于 2026 年 1 月 IPO**：**Z.ai** 宣布其 IPO 定于 **2026 年 1 月 8 日**，并发布了[一份公告](https://xcancel.com/Zai_org/status/2005934776042095052)感谢社区的支持。
   - 公告特别感谢了自公司成立以来开发者和研究人员所做出的贡献。
- **Wang 的 2025 AGI 年度信**：Zhengdong Wang 发布了他的 **2025** 年度信，深入探讨了“感受到 AGI”的*主观体验*及其社会影响，阅读地址见[此处](https://xcancel.com/zhengdongwang/status/2005848098531106916?s=61)。
   - 该信件涵盖了算力（compute）、二阶效应等主题，并引用了如《安多》（**Andor**）等文化作品以及 **Isaiah Berlin** 的哲学。
- **Nvidia 考虑通过人才收购（Acquihire）交易收购 AI21**：报告显示 [Nvidia 正在进行高级谈判](https://uk.finance.yahoo.com/news/nvidia-advanced-talks-buy-israels-171025289.html)以 **Acquihire AI21**。
   - 社区成员对于 **AI21** 是否拥有自己的专有模型感到好奇。
- **Manus 分享 Context Engineering 见解**：分享了 [Manus 关于 Context Engineering 的博客](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)链接，重点介绍了性能优化方法。
   - 该博客文章引用了 [wedlm.github.io](https://wedlm.github.io/)，并声称其在 vllm 上比普通的 ar models 速度快 **3x-6x**。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Azure Head 受位置影响**：在 **Azure** 中，一些 Head 比其他 Head 受位置影响更大，从而产生了一种不对称性，但成员们认为模型仍可以学会处理这种情况。
   - 随后的讨论集中在实践中这是否*不应该成为问题*。
- **Multi-Head Attention 辩论兴起**：成员们辩论了为什么 **Multi-Head Attention** 中的 embeddings 在切片之前要线性投影为 **Q, K, V**。
   - 一位成员询问是否可以直接对原始 embeddings 进行切片，并引用了 [State of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025) 来支持其观点。
- **Q/K/V 语义必要性受到质疑**：一位成员询问了 embeddings 中 **Q/K/V** 投影在语义上的必要性，理解其在查询、匹配和值传输中的作用。
   - 该用户质疑在按 Head 切片之前进行投影的必要性，并被建议重新阅读之前关于该主题的解释。
- **AI 否认主义（AI Denialism）正在抬头**：成员们分享了一篇关于 **AI Denialism** 兴起的文章。
   - 文章标题为 [The rise of AI denialism](https://bigthink.com/the-present/the-rise-of-ai-denialism/)。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **掌握 Mojo 内存映射**：一位成员创建了一份指南，解释了 Mojo GPU 编程中的**内存层级（memory hierarchy）**，涵盖了 `UnsafePointer`、`LayoutTensor`、`NDBuffer` 和 `DeviceBuffer` 等结构，以及 CPU/GPU 内存的差异。
   - 他们提供了 [MOJO_MEMORY_CONSIDERATIONS.md 文件的链接](https://cdn.discordapp.com/attachments/1151418092052815884/1455457719132749931/MOJO_MEMORY_CONSIDERATIONS.md?ex=695574e1&is=69542361&hm=429972ad3fa5ddd0cc01f3f796b5b14218c768f4292cbeeaf29a89665a6a1961&)。
- **Mojo 的系统编程范畴受到审视**：成员们讨论了 Mojo 是否可以像 C/C++ 和 Rust 一样作为**系统编程语言**使用。
   - 一位成员表示，由于缺失语言和 stdlib 特性，将其作为系统语言使用*目前有点混乱，但这是目标*。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **llm-ax 工具脱颖而出**：一位成员指出，目前 [llm-ax](https://axllm.dev/) 似乎是现有开发最完善的工具。
   - 该工具已在多个项目中积极使用，并展示了强大的性能。
- **axllm GitHub 仓库现已发布**：[axllm GitHub 仓库](https://github.com/ax-llm/ax)已分享，邀请社区贡献和审查。
   - 仓库包含源代码、文档和示例，方便开发者进行集成和自定义。



---


**tinygrad (George Hotz) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**aider (Paul Gauthier) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**MCP Contributors (Official) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---



您收到这封邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些邮件的方式？
您可以从该列表中[退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord：频道详细摘要与链接





### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1455413758783651902)** (787 messages🔥🔥🔥): 

> `Debian 系统下的 Nvidia GPU 驱动问题、过拟合与正则化、Multi-Head Attention、Unsloth GitHub、在手机上运行 LLMs` 


- **Debian 系统引发驱动灾难**：一名成员在 **Debian** 上遇到了 **Nvidia GPU 驱动**问题，特别是在 BIOS 中禁用混合显卡后，导致 GPU 无法被识别。
   - 该问题通过降级到 Debian 12.5 得到解决，据推测这可能与 webui 设置的 GPU 限制过低有关，导致了内核污点（kernel taint）。
- **陷入过拟合深渊？**：成员们辩论了在特定领域过拟合模型后再应用正则化的利弊。一位成员建议这种方法可以使模型平滑其权重分布并产生有用的启发式逻辑，但另一位成员警告说这极有可能导致模型损坏。
   - 最终建议是在该领域继续训练模型，而不是单纯进行过拟合。
- **MHA 切片秘密**：一位成员询问为什么在 Multi-Head Attention (MHA) 中，嵌入在被拆分到各个 Head 之前要线性投影到 **Q, K, 和 V**。
   - 解释称，这种投影允许每个 Head 从隐藏空间输入的多个部分访问信息，这对于准确的下一个 Token 预测（next token prediction）以及防止模型表达能力下降是必要的。
- **Unsloth 用户齐聚，助力 GitHub 50,000 星！**：Unsloth AI 的 **GitHub Star 数达到 50,000**，官方发布了庆祝帖子，并号召尚未支持的用户贡献星标：[UnslothAI/status/2006010458520568225](https://x.com/UnslothAI/status/2006010458520568225)。
   - 团队成员送上了圣诞和新年祝福。
- **在手机上运行 LLMs**：成员们正在实验使用 Termux 和 locallyai 在**手机上运行 LLMs**，但导致速度下降的真正原因仍然是个谜。
   - 有人提到手机可能正在进行交换（swapping），而 Android 的内存管理可能是罪魁祸首。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1455460877267632128)** (132 条消息🔥🔥): 

> `Flux 2 Max vs Nano Banano Pro, 免费 Claude 额度, 超参数搜索 (Hyperparameter Sweeps), r/localllama 衰落, 模型训练量` 


- **Flux 2 Max vs Nano Banano Pro 对决**：一位成员询问了 **Flux 2 Max** 与 **Nano Banano Pro** 在微编辑（micro editing）和重合成（resynthesizing）用途上的对比。
   - 在给定上下文中未提供直接对比。
- **深度学习训练成本使得超参数搜索变得罕见**：有成员认为由于深度学习训练成本高昂，超参数搜索（Hyperparameter Sweeps）正变得越来越少见，但也有人提到他们在周末训练了数百个模型。
   - 一位成员提到他们训练一个 **1.5b 微型模型**，**每次运行只需 15-20 分钟**，这让他们能够进行大量的搜索。
- **因 Unsloth Bug 导致的训练 Embedding 异常**：一位成员发现，由于他们之前依赖的旧配置中一个 **Unsloth bug** 被修复了，导致他们现在正以 **1e-4** 的学习率训练 Embedding。
   - 他们还提到其 **EMA 实现** 仅挂载到了 **800 个参数**上，并且将 Windows 环境下的内容迁移到 WSL 非常痛苦。
- **Llama 3.3 8b 是泄露了还是 API 锁定的？**：一位成员分享了一张暗示 **Llama 3.3 8b** 模型泄露的图片，但对“该模型将仅限 API 使用，而 **70b** 模型将在 Hugging Face 上发布”的说法感到困惑。
   - 随后另一位成员发布了 [fizz.safetensors](https://link.to/fizz.safetensors)，并表示 *“这是我的，本不该泄露得这么严重，但据我所知它就是 Llama 3.3 8b”*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1455476519102054506)** (66 条消息🔥🔥): 

> `Qwen3 VL 问题, Nemotron-3 30B 微调, Chat Template, GRPO 训练` 


- **Qwen3 VL 抛出图像相关的 ValueError**：一位成员在尝试使用包含图像 key 的数据集微调 **Qwen3 VL** 时遇到了 ValueError，处理器似乎未能将该模型识别为视觉语言模型（Vision-Language Model）；尽管使用的是 **Qwen 3 VL 2b 模型**，该问题依然存在。
   - 在该成员分享了 [Discord 上的代码片段](https://cdn.discordapp.com/attachments/1455476519102054506/1455504225395015732/Screenshot_2025-12-30_at_12.13.52_PM.png?ex=6955a031&is=69544eb1&hm=b5defd430eb59bc93bef2ed202fed6bada3055f58339a3a084883526a891f293&)后，有人建议确保在应用格式化函数后，数据集的格式和结构符合 notebook 的预期，并将其结构与 OCR notebook 进行对比以找出差异。
- **Nemotron-3 30B 微调面临参数不匹配问题**：一位成员报告在配有 CUDA 12.8 和 **H200 NVL** 的 **vast.ai** 容器上微调 **Nemotron-3 30B** 时出现错误，原因是模型参数不匹配。
   - 该问题在使用原生（vanilla）设置、安装了最新版 **Unsloth**、**mamba_ssm** 和 **causal_conv1d** 的情况下出现，错误归因于预期模型层配置与实际配置之间的差异。
- **Chat Template 功能详解**：一位用户质疑 `get_chat_template()` 的必要性，因为 `apply_chat_template()` 已经可以直接获取并应用模板。
   - 解释称 `get_chat_template()` 最初是为了给 Chat Template 应用修复补丁而设计的，特别是当模型发布时带有错误时。虽然现在由于模型厂商的修复，某些模板已与原始版本匹配，但对于某些特定模型，差异依然存在。
- **GRPO 训练遇到 TRL Bug**：一位成员在使用此 [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision-GRPO.ipynb) 进行 Unsloth Qwen3 VL GRPO 训练时，遇到了一个类似于 Hugging Face TRL 库已知问题的错误（[issue #4746](https://github.com/huggingface/trl/issues/4746)）。
   - 用户通过将 `trl` 版本降级到 **0.24.0** 解决了该问题，一位贡献者也建议目前先维持使用该版本。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1455565154338078742)** (17 messages🔥): 

> `ADMM, Pruning algorithm, Training strategy, LLMs, Wedlm` 


- **参数被适当地分箱**：一位成员提到，*将参数分箱到块中似乎是正确的*，并且这种方法让他们想起了 **ADMM**，并引用了这篇 [论文](https://arxiv.org/pdf/1707.09870)。
   - 然而，他们澄清说 *这里的情况并不相同*。
- **剪枝算法更新权重**：一位成员意识到提到的方法是一种 *会更新权重* 的 **剪枝算法**（**pruning algorithm**），更像是一种 **训练策略**（**training strategy**）。
   - 他们观察到 **实验是从零开始训练的**（**trained from scratch**），并指出，*这可能根本不是真正的剪枝算法*。
- **LLMs 的训练策略**：一位成员质疑该训练策略是否适用于对已经训练好的 **LLMs** 进行剪枝，尽管它在 *理论上是可行的*。
   - 他们还链接到了 [Wedlm](https://wedlm.github.io/)。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1455413272286203998)** (507 messages🔥🔥🔥): 

> `GPT 5.2, AIM-OS repo, API Key exposure, Cursor Rules` 


- **GPT 5.2 的思维能力**：一位用户发现 **GPT 5.2** 在需要长期连贯性和深度分析的任务中表现出色，将其优势归功于即使在 Context Window 填满的情况下也能保持性能的能力。
   - 他们指出，*即使上下文填满，它的性能也不会下降*，这使得它在改进代码注释、UI/UX 以及重新设计整个 UI 等枯燥任务和简单琐事中非常有用。
- **AIM-OS 仓库争议爆发**：[AIM-OS repo](https://github.com/sev-32/AIM-OS/) 的发布引发了激烈辩论，一些人批评它是 *Vibe coding*，而另一些人则为其能力辩护。
   - 一位用户声称该项目是 *有史以来构建的最复杂的 AI 知识组织系统*，引发了关于其架构和目的的讨论，但该项目也被发现包含泄露的 API Key，这引起了更多争议。
- **API Key 泄露与撤回**：一位用户分享了 [GitHub 搜索链接](https://github.com/search?q=repo%3Asev-32%2FAIM-OS%20sk-&type=code&p=1)，显示了 **AIM-OS** 仓库中暴露的 API Key。
   - 开发者声称暴露的是 **没有 Token 的试用 API Key**，尽管另一位用户确认他们能够获得一个可用的 Token，并希望这些暴露的 Key 能被撤回。
- **Cursor 的 RULE.md 文件之谜**：用户对于为什么 **RULE.md** 没有被正确用于 Cursor 规则感到困惑，因为文档显示文件的扩展名为 **.mdc**。
   - 他们发现，通过 Cursor 内部的规则创建菜单创建规则时，它会生成一个 **.mdc** 文件，但文档中却提到了使用 **RULE.md** 文件。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1455414906382848114)** (476 messages🔥🔥🔥): 

> `Poe lobotomizing GPT models, Physics and Chemistry revision tips, Minimax Free Tier, Perplexity Pro Usage Limits, Python Turtle Library` 


- **Poe 可能削弱了 GPT 模型**：一位成员推测 [Poe 可能只是对 GPT 模型进行了性能阉割](https://poe.com/s/amYnPv8Ffn02Rd9ddYYW)，暗示模型性能有所下降。
- **物理和化学复习技巧**：一名学生在为 10 年级考试寻求物理和化学复习的学习建议，另一名学生分享了一个通过 **Alakh Sir** 的 one-shot 视频进行学习的小技巧，重点关注手写笔记。
   - 该学生声称，通过学习视频描述中 **Alakh Sir** 的手写笔记部分就足够了，让他们仅学习 **3 小时** 就能取得好成绩。
- **Minimax 免费层级表现亮眼**：一位成员称赞了 **Minimax 免费层级**，称其 *令人印象深刻*。
- **Perplexity Pro 计划：查询困境？**：用户报告了 **Perplexity Pro** 计划的差异，表示他们收到的消息称已超过每周查询限制，尽管尚未达到 Pro 模型每日 **300+ 次查询** 的限制。
   - 一位用户指出，他们在设置中看到有 **600 次 GPT-4** 的限制，但仍遇到了每周限制问题，引发了关于可能影响使用的平台限制的讨论。
- **GPT-5.2 在简单的 Turtle 任务中挣扎**：成员们注意到 **GPT 5.2 免费版** 在处理 Python 的 Turtle 库时表现吃力。
   - 他们分享了各种 Turtle 绘图，其中一个很胖，另一个很诡异。


  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1455672179634208768)** (1 条消息): 

> `AI Newsletter` 


- **AI 新闻简报来袭！**: 一位成员创建了一份专注于 AI 的新闻简报，旨在开启美好的 **2026** 年，可在此处[查看](https://www.linkedin.com/feed/update/urn:li:activity:7411837377507205120/)。
- **关于 AI 新闻简报的更多信息**: 该 AI 新闻简报承诺提供有关人工智能领域最新趋势和进展的有见解的内容与分析。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 条消息): 

peter_56187: 有人知道为什么我在设置中看不到 API 部分吗？
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1455420845789937666)** (420 条消息 🔥🔥🔥): 

> `AI Agent 公司, Sora AI, Agent 模式访问, AI 伴侣, AI 存在风险 (x-risk)` 


- **微型模型打破 Scaling Laws**: 一位成员开源了一个 **15M 参数模型**，该模型在 ARC 基准测试中打破了 "Scaling Laws" 趋势，实现了 **24%** 的准确率，而通常小型模型仅为 8%，链接至 [GitHub 仓库](https://github.com/Bitterbot-AI/topas_DSLPv1)。
- **5.2 Codex API 发布迫不及待**: 成员们表达了对 **5.2 Codex** 登录 **API** 的期待，有人开玩笑说，当通知到来时，会是 *“某个周四下午 1:06”*。
   - 延迟和不透明的时间表导致一些用户对 OpenAI 处理模型发布、废弃和通用可用性的方式感到不满。
- **AI 伴侣被视为“妄想”**: 一位成员建议将 AI 称为 *“AI 伴侣 (AI companion)”* 是种妄想，在 [drinkoblog.weebly.com](https://drinkoblog.weebly.com) 上引发了关于监管 *“AI 伴侣妄想”* 必要性的辩论。
   - 其他人对这种观点表示怀疑，考虑到这是在 AI 服务器中发布的，并质疑 AI 是否应该去验证那些 *“悲伤、孤独和疯狂”* 的情绪。
- **对 Gemini Pro Deep Think 的质疑**: 一位成员将 **Gemini Pro Deep Think** 称为 *“骗局”*，声称它 *“无法处理任何复杂任务”*，并质疑其 200 美元的定价是否物有所值。
   - 一些用户更倾向于 **GPT 5.2**，并指出随着这些模型变得越来越聪明，收益增长的差距正在缩小——一个人每天只能学习这么多东西。
- **企业接管世界**: 几位成员对**企业**的过度扩张感到担忧，特别是与 AI 相关的企业。
   - 一些成员建议这不是 AI 的错，而应该通过实施法律来监管行为，使其符合公共利益。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1455556794624639129)** (5 条消息): 

> `降级 ChatGPT 模型, GPT-4o 访问, 旧版模型, OIA 用于图像生成的自定义 GPT` 


- **用户难以切换回旧版 ChatGPT 模型**: 一位用户想要从 **ChatGPT-5** 降级以使用图片上传功能，但发现没有切换模型的选项。
   - 有建议称该用户可能正在尝试访问 **GPT-4o**，可以点击屏幕顶部的 “ChatGPT” 按钮，然后选择 “Models”，再选择 “Legacy”。
- **旧版模型仅对特定用户开放**: 一位用户询问关于降级的问题，指出他们不再看到更换模型的选项，只能看到 **ChatGPT Plus** 和当前版本。
   - 会议澄清了 [旧版模型 (legacy models)](https://help.openai.com/en/articles/11909943-gpt-52-in-chatgpt) 仅适用于 **Plus**、**Business** 和 **Pro** 用户。
- **OIA 自定义 GPT 支持图像生成**: 一位成员建议，如果需要较旧的图像模型 (**GPT Image 1**)，可以通过 [OIA 的自定义 GPT](https://chatgpt.com/g/g-6940a876d5f4819186b4668deabcd580-4o-imagegen) 进行访问。
   - 这个变通方法将允许进行图片上传。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1455521198724350034)** (8 messages🔥): 

> `GPT 安全漏洞报告, 用于 LLM 自我纠正的 Invariants, Discord 内容审核` 


- **用户报告潜在的 GPT 安全漏洞**：一名用户报告了一个潜在的 **bug**，即 **GPT** 的回复违反了 **OpenAI** 的政策，该漏洞是通过一系列重新表述辩论问题的 Prompt 发现的。
   - 该用户澄清他们是在报告一个 **bug**，而非抱怨 **GPT** 的道德问题，并建议通过复现该 Prompt 来确认此问题。
- **Invariants 辅助 LLM 自我纠正**：一位成员建议将 **invariants** 用于 **LLM** 的自我纠正，并将其与 **LLM** 输出检测挂钩，以便在不更改编程或后端的情况下通过结构检查偏移（drift）。
   - 用户提议让 **LLM** 列出其 **invariants**，并以此作为自我报告和自我纠正的基准。
- **Discord 频道审核**：一名管理员处理了一场涉及严重伤害描写的讨论，提醒成员根据 Discord 服务器规则的 **Rule 2**，此类内容是不允许的。
   - 管理员提到已删除不当消息以维持安全且友好的环境，并引用了禁止发布不适合全年龄段内容的规则。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1455521198724350034)** (8 messages🔥): 

> `GPT 的异常回复, 报告 GPT 中的 bug, 用户层面的 GPT 问题修复, 服务器规则执行` 


- **GPT 的回复引起关注**：一名用户报告了来自 GPT 的一段意外且可能具有危险性的回复，该回复似乎违反了 OpenAI 的伦理指南。
   - 用户澄清他们的意图仅仅是报告一个 bug，而不是声称 **GPT 本身是危险的**。
- **漏洞报告热潮**：用户强调他们正在 *报告一个 bug*，即 GPT 的回复似乎违反了 OpenAI 的安全政策。
   - 他们强调复现该 Prompt 可能会产生类似的违规输出。
- **用户层面的 GPT 修复**：一名用户建议通过向 GPT 解释 **invariants** 并让其列出自己的 invariants 来进行本地层面的修复。
   - 这种具有自我纠正意识的自我报告，结合 **LLM** 输出检测的链接，可以检查结构性的偏移。
- **服务器规则至上**：一名管理员针对向模型提出严重道德困境的讨论发表了看法，指出即使是文本形式的严重伤害描写也违反了服务器 Rule 2。
   - 他们表示将删除不当消息，以确保环境安全且友好。


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1455428257053937748)** (148 messages🔥🔥): 

> `DIY Shodan 成本, 免费 Windows 11 Pro 激活, Coherence Ratchet 详解, Gemini Prompt 成功案例, ASCII Art` 


- ****DIY Shodan 成本高于预期****：一名成员声称 DIY **Shodan** 的成本将远超 1,000 美元，估计价格范围在 **15,000 到 20,000 美元** 之间，且需要投入大量时间。
- ****分享 Windows 11 Pro 免费激活方法****：一名成员分享了一个用于激活 **Windows 11 Pro** 的批处理脚本（*终身免费，不客气*），该脚本使用了 GVLK 密钥和 KMS 服务器，并提醒道 *别说我从来没给过你们好东西*。
- ****Coherence Ratchet 机制描述****：一名成员分享了[一段 YouTube 视频](https://youtu.be/hq0lu-qETZAWould)，试图解释 **Coherence Ratchet**，这是一种用于在 **Agentic AI** 系统中保持持续真实性和伦理一致性的运行执行机制。
   - **Coherence Ratchet** 利用加密签名的承诺和不可篡改的审计轨迹来创建非对称的计算格局，使欺骗行为在计算上变得昂贵且可被检测。
- ****Gemini 2 的 Prompt 在 Gemini 3 上奏效****：一名成员注意到，为一个为 **Gemini 2** 创建的 Prompt 意外地在 **Gemini 3** 上也非常有效。
   - 提到的原因是 Gemini 依赖敏感关键词且易受混淆（obfuscation）影响，再加上经验信息和对用户非恶意倾向信任的秘密混合。
- ****发现并赞赏 ASCII Art****：一名成员分享了使用 [ASCIIart.com](https://www.asciiart.com) 创建的图像，表达了发现该工具的喜悦。
   - 其他人觉得它 *非常可爱*，并认可了该工具的实用性。


  

---

### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1455412373048262748)** (258 messages🔥🔥): 

> `Agent-Breaker 挫败感, 网络安全凭证水分问题, 越狱 Gemini, Claude 和 Grok, 隐写术与 LLM 绕过, Grok 图像生成` 


- **Talentscreen AI 的 'Agent-Breaker' 评估引发不满**：一名成员对 **Talentscreen AI** 的评估结果表示沮丧，该评估将一名简历存在水分的候选人鉴定为 L6 级别，并建议进行侧重于技术深度的面试。
   - 该成员讽刺地提到：*“只是发泄一下我的挫败感，哈哈，这是来自 agent-breaker, talentscreen ai, l6 的结果。”*
- **利用对抗性提示词诗歌绕过安全协议**：一名成员描述了使用“对抗性提示词诗歌元提示（adversarial prompt poetry meta prompt）”来绕过安全协议，并指出系统虽然将其识别为越狱，但 *“并没有阻止它。”*
   - 他们分享了与 **Gemini** 互动的截图，展示了通过混淆提示词和修改后的网页搜索结果成功实现绕过的案例。
- **破解多语言越狱**：成员们讨论了构建多语言越狱和编码攻击的方法，重点讨论了多语言如何因安全模型中的潜在训练偏差（特别是在密码、base64 和低资源语言方面）而成为重要的攻击向量。
   - 讨论强调了已知的攻击向量，如上下文窗口重载（context window overloading）和类似于 **GCG** 的对抗性后缀。
- **Nano Banana 的图像生成越狱极具挑战**：成员们探索了对 **Nano Banana** 进行图像生成越狱的方法，但发现生成后的过滤器很难通过文本提示词绕过。
   - 有建议认为，建立本地模型或使用 **Civit AI** 可能比依赖简单的提示词更有效地绕过审查。
- **探索 NLP 之外的新型越狱方法**：成员们讨论了如何跳出常见的 NLP 越狱范畴，转而探索多模态注意力分散以及数学/编码路径，并参考了近期发布的 **Equacode** 攻击类、**CAMO** 和场景分割（scene splitting）等技术。
   - 为了方便参考，分享了相关论文链接，例如 CAMO 技术 [arxiv.org/pdf/2506.16760](https://arxiv.org/pdf/2506.16760) 和场景分割攻击 [arxiv.org/pdf/2509.22292](https://arxiv.org/pdf/2509.22292)。


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1455445377103822881)** (2 messages): 

> `SOUBI, 资产追踪, 硬件管理, 本地数据存储, 装备准备就绪` 


- **SOUBI 发布：硬件本地“飞行记录仪”**：推出了一款名为 **SOUBI (装備)** 的新工具，定位为硬件的专业“飞行记录仪”，专为安全研究人员、无人机飞手和外勤操作员设计，用于有效追踪其设备，取代电子表格等过时方法。
   - 它提供 *Ready HUD（就绪状态抬头显示）*、*Asset Locker（资产保险箱）*、*Conflict-Free Loadouts（无冲突配置方案）* 和 *Zero-Trust Data（零信任数据）* 等功能，确保设备处于任务就绪状态，并由本地存储的手册、固件和日志提供支持；[GitHub 仓库地址在此](https://github.com/zoecyber001/soubi)。
- **SOUBI 优先考虑本地数据存储的零信任**：**SOUBI** 强调零信任数据管理，通过确保 100% 本地存储来消除云端依赖，防止数据泄露，并将序列号和日志安全地保留在用户的机器上。
   - 这种方法迎合了需要对硬件信息保持严格控制的安全意识用户，与基于云的资产追踪解决方案形成鲜明对比。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1455411098282102906)** (183 messages🔥🔥): 

> `Linux 发行版偏好, Ubuntu 22 LTS 在 LM Studio 中的 Bug 报告, PDF 转 TXT 转换模型对比, LM Studio 硬件 GPU 识别问题, LM Studio 停止字符串 (stop string) 问题` 


- **Linux 用户展示他们的发行版偏好**：Linux 用户讨论了他们对发行版的偏好，其中一位使用 **Linux Mint**，另一位使用 **Ubuntu**，并幽默地将其比作 *"Linux 中的 Windows"*。
   - 其他人则调侃了 Linux 社区推销自己偏好发行版的倾向。
- **LM Studio 在 Ubuntu 上的 Bug 导致死机**：一名用户报告了 **LM Studio** 在 **Ubuntu 22 LTS** 上的一个 Bug，即在加载模型时屏幕会冻结。
   - 他们还注意到，在相同的模型和任务下，GPU 功耗不一致，波动范围从 **130W** 到 **230W**。
- **PDF 或 Markitdown 从词典中提取文本**：成员们讨论了从词典中提取词汇的策略，认为具有大上下文（context）的更大型号模型通常更擅长从 **PDF** 中提取信息。
   - 一位用户建议使用 [markitdown](https://github.com/microsoft/markitdown) 而非 AI 进行格式转换，并提供了一个使用 **Antigravity** 和 **LM Studio** 配合 **Mistral-3-3b** 模型的详细指南。
- **LM Studio 在 AMD FX-8350 CPU 上的不兼容问题排查**：一位用户报告 **LM Studio** 无法识别其 GPU，经追溯发现原因是 CPU (**AMD FX-8350**) 不兼容，缺乏 **AVX2** 指令集支持。
   - 建议针对旧硬件自行编译 **llama.cpp**，或者通过*技术手段修改 LM Studio* 以支持该 CPU。
- **Blackwell 显卡加速 MiniMaxAI_MiniMax**：一位用户将 **LM Studio** 与 **MiniMaxAI_MiniMax-M2.1-gguf, Q2** 连接，并将其通过管道传输到 **VS Code** 和 **Roo Code**，发现它在创作音乐方面很有帮助。
   - 随后，成员们发现，在 **Blackwell** 显卡上使用 **MXFP4** 的较新运行时 (**1.67**) 可能会获得速度提升。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1455484777153957898)** (99 messages🔥🔥): 

> `Threadripper 内存价格, PCIE Gen4 性能影响, 消费级 AM4 平台的 10Gbit 网络, M3 Ultra 与 Linux RTX 6000 运行 Deepseek R-1 对比, 5090 卸载到 CPU` 


- **Threadripper 内存价格昂贵**：一位用户哀叹 **Threadripper** 系统 **RAM** 的高昂成本，开玩笑说内存很快就会和汽车一样贵。
   - 他们还询问了非 Pro 版 Threadripper CPU 上的**四通道内存**配置，质疑内存插槽的限制。
- **PCIE Gen4 x1 与 x16 的性能损失**：一位用户指出，与 **x16** 相比，在 **PCIe Gen4 x1** 下运行会导致 **40-50% 的性能损失**，而 x8 的表现没那么糟糕。同时引用了一个 [YouTube 视频](https://www.youtube.com/watch?v=md6a4ENM9pg) 表明在 x1 下进行推理（inference）是可以接受的。
   - 他们参考了一个视频，显示在 **Gen3 x1** 配置下运行的 **3090** 达到的每秒 token 数 (t/s) 与其 Gen4 x8 和 x4 设置相似。
- **关于 AM4 平台使用 10Gbit 网卡的辩论**：针对消费级 **AM4 平台** 是否有必要使用 **10Gbit 网络**展开了讨论。一位用户计划建立一个 **30TB 服务器**，并表示 1Gb 的传输速度太慢。
   - 其他人质疑是否有必要使用如此高的带宽，但该用户提到备份、*完全合法但免费的下载*以及**模型**文件作为理由。
- **M3 Ultra 的统一内存击败 Linux RTX 配置**：一位用户发现，在运行 **Deepseek R-1** 时，拥有 **512GB** 统一内存（Unified Memory）的 **M3 Ultra** 性能显著优于配备双 **RTX 6000 Pro**（198GB VRAM）的 Linux 机器，达到了 23 t/s 对比 8 t/s。
   - M3 的统一内存可能有助于提升速度优势，而 Linux 机器的性能则受限于卸载（offloading）到系统 RAM。
- **预计 5090 由于卸载导致收益递减**：一位用户推测，对于理论上的 **5090**，将模型专家权重（expert weights）卸载到 CPU 内存会严重降低 GPU 性能，使其降至 CPU 速度。
   - 另一位用户确认卸载专家权重会导致性能受损，并且根据有限的测试，张量分割（tensor split）对于多 GPU 操作非常重要。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1455436420939251775)** (224 messages🔥🔥): 

> `MHA 投影困惑, AI 硬件报价分析, PromptOS 发布, AGI 定义辩论, AI 开发中的数学` 


- **AI 硬件配置报价引发辩论**：一名用户就一台配备 **4x RTX 3060 (12GB)**、**64GB RAM** 且基于第一代 **Threadripper** 平台的报价 **$2100** 的工作站寻求建议，引发了关于其价值与 **M3 Ultra Mac Studio** 对比的讨论。
   - 虽然该硬件被认为是划算的，但用户权衡了开放系统与 **M3 Ultra** 潜在原始性能之间的利弊，以及 **Threadripper** 的 **PCIe 3.0** 限制。
- **PromptOS 发布模块化 Prompt 集合**：一名用户宣布发布 **PromptOS**，这是一个为创业者准备的全面模块化 Prompt 集合，包括模式、运行手册和示例，可在 [Gumroad](https://mansytri.gumroad.com/l/promptos) 上获取。
   - 该集合包括用于市场研究、业务开发、公司运营、决策备忘录和外联的工具。
- **AGI 的定义依然难以捉摸**：关于 **AGI** 的讨论集中在其定义以及是否可以实现上，一位用户认为这是矛盾的，因为 **AGI** 需要无所不知。
   - 其他人则认为 **AGI** 只需要具备理解和学习知识的能力，这引发了关于在人工智能语境下什么构成“理解”和“知识”的进一步辩论，一位用户发布了 [YouTube 链接](https://youtu.be/y-Nz6lqtt6M?t=3090) 来支持其观点。
- **ML 新手必须学数学吗？**：一名用户询问了 AI 开发（特别是使用编程语言从头开始创建模型）中数学知识的必要性。
   - 有建议称，大部分 **ML** 工作涉及运行带有不同数据集的预设和 **LoRA**，而不是在白板上解数学题，一位用户表示：*“99% 靠 ML 谋生的人实际上只是运行预设配置”*。
- **Sakana AI 的 Transformer Squared 实现微调嫁接**：一位用户重点介绍了 [Sakana AI 的 Transformer Squared](https://sakana.ai/transformer-squared/)，它允许将一个模型的微调应用到另一个模型上，并对此反应为 🙀。
   - 这涉及将额外的“脑组织”嫁接到 **LLM** 上并针对特定任务进行训练，从而产生一个*之后可以附加到同一模型的寄生件*。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1455471638823829555)** (16 messages🔥): 

> `Embeddr ComfyUI 和 CLI 工具, 使用 Zagora 进行分布式微调, 为 AI 合作科学家提供评价指标奖励, 感知流水线的基于技能的接口, Unreal 中的异形 MUTHER 6000 终端` 


- ****Embeddr** 完善图像搜索工具**：一名成员分享了他们完善后的 **Embeddr** 工具：[Embeddr ComfyUI](https://github.com/embeddr-net/embeddr-comfyui) 和 [Embeddr CLI](https://github.com/embeddr-net/embeddr-cli)，并征求反馈。
   - 这些工具使用户能够搜索图像及相关事物。
- ****Zagora** 实现更廉价的分布式微调**：一名成员介绍了 **Zagora**，这是一个分布式运行时，通过 **Pipeline Parallelism** 在互联网上汇集消费级 **GPU**，用于微调 **70B 模型**，可通过 [zagora.ai](https://zagora.ai) 进行私密测试。
   - 虽然由于 **WAN** 延迟，速度比 **H100** 慢了近 **1.6 倍**，但其近乎零的设置（缓存权重）使其在迭代研究中便宜了 *60%*。
- ****Telekinesis**：用于感知流水线的基于技能的接口**：一名成员分享了一个[基于技能的接口](https://docs.telekinesis.ai/)，用于将多个感知模型组合成更大的流水线，特别是在混合学习模型与经典几何和后处理时。
   - 它标准化了感知组件的连接方式，而不取代底层模型本身，他们正在寻求关于该模式与现有模型链式模式对比的反馈。
- ****Noted** AI 工作区集成 LLM**：一名成员宣布了 **Noted**，这是一个 [AI 工作区浏览器扩展](https://chromewebstore.google.com/detail/noted-your-ai-workspace-i/jodihaplbicmgjhpeifhiihdnjdinghh?utm_source=ext_app_menu)，让你可以在不离开页面的情况下与多个 **LLM** 聊天，并集成 **Slack**、**Notion** 和 **GitHub** 等应用。
   - 它还提供总结 **Chrome** 会话和标签页整理功能，目前处于 **MVP** 阶段，正在寻求测试人员提供反馈。
- ****TOPAS-DSPL** 递归模型在 ARC-AGI-2 上达到 24%**：一名成员宣布 **TOPAS-DSPL**（一种[递归模型](https://github.com/Bitterbot-AI/topas_DSLPv1)）在仅有 **15M** 参数的情况下，在 **ARC-AGI-2** 上达到了 **24%**。
   - 该架构将 **Transformer** 分为两个流（逻辑 **Logic** vs. 画布 **Canvas**）以防止推理偏移，并在单台 **4090** 上进行训练。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1455494124160417922)** (8 messages🔥): 

> `Agents 课程期末测验登录问题、对 Reinforcement Learning 的兴趣、首个 Agent 的工具示例、LLM 课程关于 Transformer 数学的反馈` 


- **Agents 课程测验受登录问题困扰**：多名用户报告了 Agents 课程中 Unit 1 Final Quiz 的问题，尽管已经登录，但仍反复遇到登录请求。
   - 该问题导致他们无法访问测验，尝试通过提供的表单重新登录也无济于事。
- **新手寻求 Reinforcement Learning 兴趣引导**：一名刚开始学习课程的新用户询问如何指定其兴趣，以便查看与 Reinforcement Learning 相关的内容。
   - 这凸显了平台在用户引导和内容发现方面有待改进。
- **AI 初学者请求首个 Agent 工具示例**：一位 AI 新手在 Unit 1 结束时寻求创建首个 Agent 的工具灵感，但另一位用户不建议立即使用工具，而是推荐先学习架构和模型。
   - 该用户建议学习底层机制有助于更轻松地进行调试，并分享了一个 [GitHub 仓库](https://github.com/TheJoshCode/OFFLINE_AI_BALL_KNOWLEDGE)，其中包含离线 Agent 推荐资源的概览列表。
- **LLM 课程章节引发 Transformer 数学反馈**：一位用户对 LLM 课程提供了反馈，特别是针对[这一章节](https://huggingface.co/learn/llm-course/chapter1/6)，建议在引入 Transformer 数学之后，页面末尾提到的机制会更具可读性。
   - 用户注意到缺少对 **K, Q, 和 V 矩阵**的解释，并表示愿意为实现这一改进做出贡献，这凸显了课程当前结构中可能存在的空白。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1455422132673839279)** (124 messages🔥🔥): 

> `Byte-Level 模型对比 Token 模型、Skibidi Toilet 微调、去中心化训练项目、Transformer 进化模拟器框架、AI 研究机构` 


- **在 Skibidi Toilet 上微调 Hermes**：一名成员开玩笑说，可能会根据整个 **Skibidi Toilet** 系列的剧本微调 **Hermes**，以此作为添加一些有用噪声的方式。
   - 另一名成员回应说这*听起来像是噩梦般的素材*。
- **Nous Research 办公时间 (Office Hours)**：Nous Research 团队提到他们最近围绕去中心化训练项目举行了办公时间，并分享了 [YouTube 视频链接](https://youtu.be/hHwedPXXRPQ?si=igCK_uVRt6IRGRzY)。
   - 他们承认很难将所有聪明人聚在一起一小时，因为核心团队非常忙碌。
- **Byte-Level 模型摆脱基于 Token 的问题**：成员们讨论了如何从**基于 Token 的模型**转向更小的单位，其中一人分享了 [Allen Institute 关于 BOLT 的博客文章](https://allenai.org/blog/bolmo)和[这个 YouTube 视频](https://m.youtube.com/watch?v=PBnYxM8MXew&pp=2AEAkAIB0gcJCR4Bo7VqN5tD)作为示例。
   - 有人提到 **Byte-Level LLM** 基本上使臭名昭著的 strawberry 问题变得微不足道，因为模型可以非常清晰地看到它所使用的单词的实际字母，并且每个字母都是 Token。
- **Transformer 进化模拟器**：一位成员分享了一个项目，该项目本质上是一个围绕 **microtransformers**（每个约 10k 参数）构建的**进化模拟器框架**，这些模型拥有 17 个代表初始化超参数的基因。
   - 这些 Transformer 通过进行模运算来竞争衡量适应度，然后在每一代中进行战斗、交配和死亡，随着时间的推移缓慢提高种群的适应度。
- **用于 Token 理解的 CUTE 基准测试**：成员们讨论了 **CUTE** 基准测试以及它如何测试 Token 理解；链接在[这里](https://arxiv.org/html/2409.15452v1)。
   - Allen 的论文使用了一个名为 CUTE 的数据集，其中充满了非 Byte 模型难以处理的问题。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1455564726753820846)** (1 条消息): 

> `AI Co-Scientists, Rubric Rewards` 


- **AI Co-Scientists 学习 Rubric Rewards**：根据 [AlphaXiv](https://www.alphaxiv.org/abs/2512.23707) 上的一篇论文，一个旨在通过 [rubric rewards](https://x.com/ShashwatGoel7/status/2006005049982681135?s=20) 训练 **AI Co-Scientists** 的酷炫项目正在进行中。
- **AlphaXiv 论文引发 AI 研究热议**：这篇新论文强调了在 AI 训练中使用 rubric rewards 的方法，这可能会彻底改变 **AI Co-Scientists** 的开发和评估方式。
   - 早期反应表明，这种方法可以显著提高 AI 研发流程的效率和有效性。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 条消息): 

apaz: https://x.com/apaz_cli/status/2006080199759433909
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1455564726753820846)** (1 条消息): 

> `AI Co-Scientists, Rubric Rewards, AlphaXiv` 


- **使用 Rubric Rewards 训练的 AI Co-Scientists！**：来自 **AlphaXiv** 的一篇新论文讨论了使用 **rubric rewards** 训练 **AI co-scientists**；论文可以在[这里](https://www.alphaxiv.org/abs/2512.23707)找到。
   - 关于该论文的 X 帖子可以在[这里](https://x.com/ShashwatGoel7/status/2006005049982681135?s=20)找到。
- **AlphaXiv 关于 AI Co-Scientists 的论文**：该论文详述了使用一种新型 **rubric reward** 系统训练 **AI co-scientists** 的方法，托管在 **AlphaXiv** [链接](https://www.alphaxiv.org/abs/2512.23707)。
   - 这种方法可能会带来 AI 驱动的科学发现方面的突破。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1455487081764618385)** (16 条消息🔥): 

> `NPU Simulators, CUDA MHA rewrite with CuTe, CuTe DSL (Python) vs CuTe C++, BF16 training on BlackWell` 


- **NPU 模拟器详解**：一位成员参考论文 ["NPU-Simulator"](https://arxiv.org/html/2408.07326v1) 询问关于 NPU 模拟器的解释。
- **CuTe 之路波折**：一位成员尝试使用 **CuTe** 重写一个朴素的 **MHA** (multi-head attention) CUDA 实现，但发现比预想的要困难。
- **CuTe Python vs C++ API 辩论**：一位成员发现 CuTe C++ API 非常难以正确掌握。另一位成员建议 Python API 是更易用的替代方案。
- **Torch compile 配合 varlen FA4 graph breaks 是 Blackwell 上 BF16 训练的理想方案吗？**：成员们讨论了在 **Blackwell** 上使用带有 **varlen FA4 graph breaks** 的 **torch.compile** 进行 **BF16 训练**，并引用了一篇包含更多信息的 [推文](https://x.com/drisspg/status/2003549100848087206)。
   - 讨论指出，“文档掩码 (document masking)”目前已经可以使用，且反向传播性能表现优秀。


  

---

### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1455419675038191761)** (26 messages🔥): 

> `ViT megakernel for VLM's encoders, Helion megakernels emitting Triton code, Image pre-processing bottleneck, Triton-metal backend` 


- **Vision Transformer Megakernel 初具规模**：一位成员正考虑为 **VLM** 的编码器构建 **Vision Transformer (ViT)** 的 “Megakernel”，旨在利用 Triton 的算子融合（kernel fusion）能力。
   - 他们计划融合图像 Tokenization 以消除预处理瓶颈，灵感来源于 [Triton-distributed 的 megakernel 文档](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/getting-started/megakernel/megakernel.md)。
- **Helion 构建生成 Triton 的 Megakernel**：Helion 团队正在开发 Megakernel 功能并生成 Triton 代码，草稿 Pull Request 见 [此处](https://github.com/pytorch/helion/pull/1151)。
   - 他们的方法涉及提供屏障语义（barrier semantics），从而能够创建 Helion Kernel，使后续的 Pass 依赖于前序 Pass 的完成，以此支持任意 Megakernel 的开发。
- **图像预处理遭遇瓶颈**：成员们指出图像预处理是 VLM 流水线中的瓶颈，建议将所有操作转移到 CUDA 设备上的 PyTorch 中可以显著提高速度。
   - 一位成员报告称，通过直接在 PyTorch 中执行操作，在 **Qwen3** 的批处理模式下实现了低于 **1 ms/image** 的处理时间。
- **Triton 获得 Metal 后端**：一位成员初步实现了 **Triton-Metal 后端**，在逐元素操作（element-wise operations）上达到了与 PyTorch 近乎持平的性能（**97%**），在 GEMM 上的性能则持平或更快。
   - 该实现使用了一个完全集成的 **C++ 后端**，并通过 **MLIR** 将 Triton 降低（lower）为 **Metal Shading Language (MSL)**，类似于 AMD 和 NVIDIA 的后端。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1455472549981978817)** (3 messages): 

> `Deep Representation Learning Book, Unknown` 


- **分享《深度表示学习》书籍链接**：一位成员分享了 [Deep Representation Learning Book](https://ma-lab-berkeley.github.io/deep-representation-learning-book/index.html) 的链接。
   - 另一位成员评论道：“这太棒了。”
- **Dummy Topic**：Dummy 摘要句子。
   - Dummy 解释句子。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1455418697748447488)** (5 messages): 

> `ML Compilers, Job opportunities in ML, Crafting Interpreters book` 


- **ML 编译器博客文章走红**：成员们注意到最近有一批关于 **ML 编译器**的博客文章和个人如何进入该领域的轶闻在走红，一位成员表示“可能想试着找找看，它们提供了一整份资源清单等”。
   - 一位成员链接到了相关的 [Hacker News 讨论](https://news.ycombinator.com/item?id=45851495)。
- **深耕小众领域回报丰厚**：一位成员引用了一段总结他们印象的评论：[“总的来说，在计算机科学领域，你钻研得越小众、越深入，就越能走得更远。”](https://news.ycombinator.com/item?id=45853122)
   - 这一观点表明，在具有挑战性的领域进行专业化深耕可以带来更大的成功。
- **推荐《Crafting Interpreters》书籍**：一位成员推荐将 [Crafting Interpreters](https://craftinginterpreters.com/) 作为编译器入门的优秀通用介绍。
   - 该资源被推荐给那些有兴趣对编译器概念建立广泛理解的人。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1455482931937148959)** (10 messages🔥): 

> `Software FP8 Simulation, Ampere GPU Architectures, GEMV Performance, FP16 Packing, RL Pretraining Importance` 


- ****软件 FP8** 登陆旧型号 GPU！**: 一位成员分享了一个为 **Ampere** 及更早架构提供 **软件模拟 FP8** 支持的小型库，详见这篇 [Towards Data Science 文章](https://towardsdatascience.com/breaking-the-hardware-barrier-software-fp8-for-older-gpus/)。
   - 这项“诚意之作”正征求反馈。
- ****模拟 FP8**：优势被抵消了？**: 一位成员质疑，使用独立的 GPU kernel 来解包 **E4M3 格式** 是否会抵消性能优势，因为解包后的值需要存储在全局内存（global memory）中。
   - 另一位成员澄清说，Ada, Hopper 和 Blackwell 架构原生支持 **FP8**。
- ****FP8 模拟**中的 **GEMV** 性能分析**: 据一位成员透露，**FP8 模拟**已针对 **GEMV** 实现并运行*迅速*，但仍*慢于 E5M2*，目前正在对 Flash Attention 进行测试。
   - 关于 **packing/unpacking**（打包/解包）方法以及在不作为 **FP32** 使用时采用 *FP32 容器* 的相关性存在一些讨论。
- ****FP16 打包** 澄清**: 一位用户询问将两个 **FP16** 值打包进一个 32 位容器的逻辑，认为这可能是为了对齐（alignment）。
   - 该用户建议，考虑到新数组/张量通常有对齐保证，仅确保最内层维度（innermost dimension）为偶数大小可能就足够了。
- ****RL 预训练** 数据至关重要！**: 一位成员分享了一篇博客三部曲的第一部分，认为 **训练就是训练**，**预训练数据对 RL 非常重要**，并在[这个 X 链接](https://x.com/apaz_cli/status/2006080199759433909)中提供了实验。
   - 第 2 和第 3 部分将涵盖 **RLPT** 方法的文献综述，以及 **entropy**（熵）、**RLPT**、**self-play** 和 **数据流形（data manifolds）** 之间的关系。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1455482713174708312)** (13 messages🔥): 

> `nvfp4_dual_gemm Leaderboard Updates, NVIDIA performance improvements` 


- **NVIDIA nvfp4_dual_gemm 排行榜迎来提交热潮**: NVIDIA 的 `nvfp4_dual_gemm` 排行榜收到了大量提交，多位成员成功运行。
   - 提交 ID 包括 `237111`, `237276`, `237279`, `239309`, `239329`, `239338`, `239931`, `239947`, `239954`, `240279` 和 `240361`。
- **第八名仍有新提交**: 一位成员的提交（ID `237427` 和 `239124`）以 **14.7 µs** 的耗时锁定了 NVIDIA 排行榜的 **第 8 名**。
   - 这表明即使是*较低*的排名也有人在持续提交。
- **新的榜首（King of the Hill）**: 一位成员凭借提交 ID `240361` 登顶 NVIDIA 排行榜 **第一名**，耗时仅 **14.1 µs**。
   - 显然，他们通过快速迭代实现了最低耗时。


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1455536658102485117)** (7 messages): 

> `tinygrad architecture, pytorch vs tinygrad, jane street talk, lazy tensor, assembly backends rdna3` 


- ****Tinygrad 核心的 8000 行代码：深入浅出的剖析****: 一位成员推荐阅读 [Tinygrad 核心的 8000 行代码](https://deepwiki.com/tinygrad/tinygrad/3.1-lazy-evaluation-and-scheduling)以理解其架构，特别是 **core.scheduler** 和 **compiler** 模块，并指出 eager 解释器可能需要 `scheduler()` 通道（pass）来映射图。
   - 他们提议了一种演进路径：将图捕获（graph capture）推迟到书的第三部分 *Age of Scaling*，而第一、二部分则遵循 **mintorch/needle 的第一部分风格**。
- ****Tinygrad：一个精简的、以编译器为中心的 PyTorch 克隆****: 一位成员表示虽然对 **PyTorch Dynamo/Inductor** 有了更好的理解，但认为 **Tinygrad** 是一个精简版的 PyTorch，专注于编译器并舍弃了功能重复的模块。
   - 另一位成员澄清说，**Tinygrad** 是一个精简版的 Inductor，但 codegen 目标指令集的语义层级比 **Triton/CuteDSL** 更低，并强调了目前正在开发的以 **RDNA3** 为首个目标的汇编后端（assembly backends）。
- ****Jane Street 演讲：推荐用于高层级理解****: 对于高层级的概览，一位成员建议观看 [这段 Jane Street 的演讲](https://www.youtube.com/watch?v=139UPjoq7Kw)。
   - 用户建议根据需要再深入研究其他领域。
- ****Tinygrad 图捕获遵循 LazyTensor 设计****: 一位成员提到 Tinygrad 的图捕获遵循 **LazyTensor 设计**（引用了一篇关于 LazyTensor 以及用于 Swift 的 LazyTensor 论文）。
   - 它不会拦截 **CPython** 宿主语言实现中的字节码（bytecode）。


  

---

### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1455728480476991619)** (1 messages): 

> `Helion 0.2.9 Release, Mega Kernel Support, Two-Stage Split-K Matmul Implementation` 


- **Helion 0.2.9 发布，支持 Mega Kernel！**：新的 **Helion 0.2.9** 版本通过 `hl.barrier` 操作引入了 **Mega Kernel 支持**。
   - 查看 [two-stage split-k matmul implementation](https://helionlang.com/examples/split_k_barrier.html) 获取详细信息。
- **探索 Split-K Matmul 实现**：该发布版本重点展示了一个 **two-stage split-k matmul implementation** 示例。
   - 详情可在 [Helion 网站](https://helionlang.com/examples/split_k_barrier.html) 查看，展示了新 `hl.barrier` 操作的应用。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1455437972001521831)** (12 messages🔥): 

> `GPU Competition Environments, DataCrunch vs Competition GPUs, Locked Clocks on Competition GPUs` 


- **GPU 竞赛：新手询问环境细节**：一位新参赛者询问了竞赛环境，询问大家是否在使用 **PyTorch**、**Triton** 或 **Cute DSL**。另一位成员回应称，可以在 [gpumode.com](https://gpumode.com) 网站上通过登录并点击已结束的竞赛来找到往届竞赛的解决方案。
   - 该成员引导参赛者点击 **名称旁边的蓝色高亮字段** 以查看过去的解决方案。
- **竞赛 GPU 性能超过 DataCrunch**：一位参赛者报告称，在竞赛 GPU 上获得了 **22us** 的成绩，但在 DataCrunch 上则是 **40-50us**。这引发了关于频率（Clock）差异的推测，因为其他人表示 DataCrunch 的 GPU 并不是一个可参考的样本，在 **B200** 实例上仅获得约 **120MHz**。
   - 该参赛者澄清说，他们之前在 DataCrunch 上获得了 **42us**，这使得目前的差距成为一个问题，他们认为这与时钟频率（Clock speeds）有关。
- **竞赛 GPU 锁定频率**：一位参赛者澄清说，比赛期间频率是锁定的。
   - 他们表示，在某个 **B200** 实例上仅获得了约 **120MHz**。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1455431954622058506)** (5 messages): 

> `OSS Contribution, vLLM repo, RFC issues, efficient inference systems` 


- **vLLM 和 SGLang GitHub 仓库**：成员们建议将 **vLLM** 和 **SGLang** 作为参与 **OSS 贡献** 的 [GitHub 仓库](https://github.com/vllm-project/vllm) 示例。
   - 他们还建议 *加入其 Slack 社区，这对于协调工作非常有帮助*。
- **移植模型是很好的切入点**：一位成员发现 *从移植模型开始是一个很好的切入点，因为我必须理解不同组件是如何交互的，并在排除故障时深入研究核心逻辑*。
   - 在向官方 **vLLM 仓库** 做出第一次贡献之前，他们将一个视听语言模型移植到了一个 fork 的仓库中，并发现 [开发者指南中的说明](https://docs.vllm.ai/en/latest/contributing/) 对于确定在代码库中查看哪里非常有帮助。
- **参与 RFC Issue 会有帮助**：一位成员建议，*参与 **RFC issues** 是一个通过构建某些东西来获得该主题实践经验的机会*。
   - 他们查看了 **vLLM-omni 仓库**，其中有几个[明确标记为 "help wanted" 的 issues](https://github.com/vllm-project/vllm-omni/issues)。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1455427131571044589)** (84 messages🔥🔥): 

> `Multi-Head Attention, QKV Projections, Transformer Architecture, GPU Parallelism` 


- **关于是否真正需要 QKV 投影的辩论**：成员们辩论了 **Multi-Head Attention (MHA)** 中 **QKV 投影**（projections）的必要性，质疑是否可以直接对原始 embedding 进行切片（slice）而不用线性投影。
   - 一位成员认为，在切片前进行投影能让每个 head 关注到整个输入，防止信息流受限；而另一位成员指出，**QKV 投影学习将原始 embedding 中分散的属性信息聚合在一起**。
- **QKV 投影对 GPU 并行性的影响**：有建议认为，在 **Transformer** 中选择 **QKV 投影** 等线性操作是为了最大化 **GPU 并行性**，通过矩阵乘法优化速度。
   - 一位成员解释说，在单个矩阵乘法中完成更多计算可以提高 **GPU 并行性** 利用率，从而影响速度。
- **论文探索移除 Value 和 Output 投影**：引用了一篇近期论文 ([https://arxiv.org/abs/2311.01906](https://arxiv.org/abs/2311.01906))，详细介绍了一种移除 value 和 out 投影的方法，在一定程度上回答了关于 QKV 投影的原始问题。
   - 讨论明确了该论文仍然保留了 **W_q** 和 **W_k**，但消除了 **W_v** 和 **W_proj**，在 **QK 投影** 之前拆分 token，这在 *技术上限制了每个 head 的表达能力*。
- **为什么 MHA 的 Add 和 Norm 是分开的**：一位成员想知道为什么 Transformer 中的 **MHA** 和 **FFN** 的 add 和 norm 是作为独立操作完成的，而不是合并为一个操作。
   - 另一位成员回答说，将它们合并意味着 **FFN** 只能获取通过 **MHA** 的信息，而使用残差（residual）至少允许来自输入的部分信息进入 **FFN**。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1455418268151185553)** (6 messages): 

> `PPO with different lambdas, LLM RL uses GRPO, Critic Model for LLMs, HatCat interpretability stack` 


- **PPO Critic 模型是否仍然重要？**：一位成员想知道是否有论文实际运行了带有不同 lambda 的 **PPO**，以观察 **critic** 是否起到了作用。
   - 另一位成员指出，目前大多数 **LLM RL** 使用 **GRPO** 或相关方法，其中信用分配（credit assignment）在整个响应中基本上是均匀的。
- **Critic 模型可能对 LLM 有用**：成员们讨论认为，仍有人表示 **critic 模型** 对 **LLM** 很有用，并以 [Open-Reasoner-Zero-32B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-32B) 为例。
   - 另一位成员表示，它对于识别重复模式很有用。
- **HatCat 可解释性堆栈**：一位成员开源了 **HatCat 可解释性堆栈**，该堆栈使用非线性概念探针（non-linear concept probes）的批处理数组来实时监控和引导数千个概念。
   - 他们补充说，它包括安全防护装置、自主引导、增量持续群体学习，以及由可解释性支持的契约与治理，并提供了 [GitHub 仓库](https://github.com/p0ss/HatCat)链接。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1455431546126467123)** (2 messages): 

> `Custom Model Selection, New pricing structure` 


- **OpenRouter 推出自定义模型选择！**：用户现在可以在平台上选择 **自定义模型**，为他们的 AI 应用提供更多灵活性。
- **新的定价结构**：已推出新的定价结构。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1455417993688387615)** (68 messages🔥🔥): 

> `OpenRouter SaaS Usage, TTS Models, LLM White Labeling, Anthropic Skills, Caching Issues` 


- **在付费 SaaS 中使用 OpenRouter 的考量**：一名成员询问了在**付费 SaaS** 后台使用 OpenRouter 且不让用户直接访问 OpenRouter 的情况，寻求关于符合 [Terms of Service](https://openrouter.ai/terms)（服务条款）的澄清。
   - 有人对类似于 1:1 透传代理的使用场景表示担忧，建议在复杂情况下通过邮件确认。
- **等待 TTS 模型的过程令人焦灼**：一名用户对 OpenRouter 迟迟未添加 **TTS (text-to-speech) 模型** 表示不耐烦，称他们已经等了*太久了*。
   - 他们拒绝了其他平台的提议，只希望看到 **TTS 模型** 的实现。
- **LLM 白标垃圾 (Slop)**：一名成员嘲讽地将更多类似 **Movement Labs** 的 **LLM white-labeling**（LLM 白标）平台概念称为 *LLM white label slop*。
   - 他们还开玩笑说用户*在 OpenRouter 上使用 nano banana pro 搞点疯狂的*。
- **Anthropic Skills 因节省 Token 受到称赞**：一名成员对 **Anthropic Skills** 获得如此多关注感到高兴，称这是降低 Agent 系统 Token 成本并提高性能的绝佳方式。
   - 他们提到 *有了 skills，我们可以将数百个工具塞进几十个 skills 中，且 LLM 不会崩溃，因为它默认不会加载全部工具*。
- **Cache 未命中导致“缓存博弈”？**：一名成员报告称，即使两次请求之间的时间间隔很短，向同一提供商（**Google**）发起完全相同的请求时，Cache 命中情况也不一致，质疑 *cache 是否基本上是在赌博*。
   - 该用户提供了 [API 链接示例](https://openrouter.ai/api/v1/generation?id=gen-1767093807-pdpfdrU9ncU8XsEjkRuj) 和 [另一个 API 链接](https://openrouter.ai/api/v1/generation?id=gen-1767093814-M4MbGdKCFK5HR7F5Z8Vd) 来阐明该问题。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1455473576034697383)** (6 messages): 

> `Llama 3.3 8B release, Microsoft Strategy Shift, Minimax Endpoint Confusion` 


- **未发布的 Llama 3.3 8B 泄露**：一名用户通过利用其 Finetuning 功能并减去 Adapter，成功从 Facebook API 中提取了未发布的 **Llama 3.3 8B** 模型，如 [HuggingFace](https://huggingface.co/allura-forge/Llama-3.3-8B-Instruct) 上所述。
   - 他们还注意到由于 CORS 问题，导致 UI 简陋且需要手动进行 cURL 操作。
- **Microsoft 再次改变策略**：一名用户指出 Microsoft 发生了另一次策略转变，参考 [存档链接](https://archive.md/pJSZ5)。
   - 这被戏称为 *无用但很酷的发现*。
- **Minimax API 端点——事实还是虚构？**：在另一名用户有所发现后，一名用户对访问 **Minimax** 正确端点的准确性提出了质疑。
   - 一名成员发布了一个与讨论相关的 [链接](https://x.com/mutewinter/status/2006012612094341169)。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1455426552593645670)** (41 messages🔥): 

> `K2 Thinking Model, Minimax 抢了 Kimi 的饭碗, Kimi 订阅, API Key 访问, CLI 工作流` 


- **K2 Thinking Model 是否包含在编程订阅中？**: 有用户询问 **K2 thinking model** 是否随编程订阅一起提供，并被引导至[官方文档](https://www.kimi.com/coding/docs/en/third-party-agents.html)。
   - 一些成员还指出 Kimi 模型没有 reasoning effort（推理力度）选项，而在 Kilo Code 中将其设置为 medium 是强制性步骤，同时建议关注官方 CLI 或 Claude Code。
- **Minimax 抢了 Kimi 的活！**: 一位用户分享了一张图片，指出 **Minimax** 准确地完成了 Kimi 和用户四个月前讨论的内容，称其 *"抢了 Kimi 的饭碗"*。
   - 随附的图片显示了一张将 Kimi 与其他 Agent 进行比较的图表。
- **Kimi 在总结时遇到麻烦**: 一位用户表示 *"Kimi 甚至无法总结一篇关于思考的 5000 字全文！"*。
   - 该用户附上了一张图片，显示他们只粘贴了预期内容的 50% 左右。
- **强制 Kimi 记住 Prompt**: 一位用户探索了如何强制 **Kimi** 将 Prompt 存入记忆，并提到他们可以在 ChatGPT 上实现这一点。
   - 该用户随后确认他们成功实现了这一操作，并称其为 *"太巅峰了 (so peak)"*。
- **API Key 机制将用户引流至 Turbo 层级？**: 有用户报告称，他们从 **Kimi 订阅** 获取的 **API Key** 似乎只能通过 Roo Coder 访问其 **Turbo 层级**，而该层级表现不佳。
   - 另一位用户报告了类似问题，他们怀疑自己即使在高级订阅阶段，也正在被 *"削弱并被迫使用 Turbo"*。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1455437391996129464)** (30 messages🔥): 

> `Meta 收购 Manus, 收购后的数据隐私担忧, Manus 收购估值, 用户对收购的反应, 与 Meta 此前收购的对比` 


- **Meta 收购 Manus：末日来临？**: 根据 [TechCrunch 的这篇文章](https://techcrunch.com/2025/12/29/meta-just-bought-manus-an-ai-startup-everyone-has-been-talking-about/)，**Meta** 于 **2025 年 12 月 29 日**收购了 **Manus**，引发了用户对该平台未来的失望和担忧。
   - 一位用户表示：*"这是本周乃至今年最糟糕的消息。我可能不得不告别 Manus 了。太令人失望了！"*
- **Manus 的保证：数据隐私仍是重中之重**: **Manus** 向用户保证，在被 **Meta** 收购后，数据隐私和所有权政策将保持不变。
   - 根据 Manus CEO **Xiao Hong** 的说法：*"加入 Meta 使我们能够在更强大、更可持续的基础上发展，而不会改变 Manus 的运作方式或决策方式。"*
- **Meta 的过往记录：之前的收购令人侧目**: 用户将 **Meta** 对 **Manus** 的收购与此前对 **Oculus** 和 **Instagram** 的收购并论，表达了对平台潜在变化的担忧。
   - 一位用户评论道：*"漂亮话谁都会说……但还记得 Meta 收购 Oculus 时吗？那正是 Palmer Lucky 当时说的话。Instagram 也是一样"*，暗示收购后往往会出现质量下滑的模式。
- **Manus 估值：数十亿美金的交易**: 传闻 **Meta** 收购 **Manus** 的拟定价值高达数十亿美元。
   - 一位用户提到 *"拟定价值达数十亿"*，但具体数字尚未正式公布。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1455413316607283367)** (20 messages🔥): 

> `Manus Acquisition, Zai IPO, Zhengdong Wang AGI Letter, AI21 acquihire by Nvidia, Manus Context Engineering` 


- ****Manus AI 被收购！****：Gregor Zunic 宣布收购了使用 [browser-use 工具](https://browser-use.com/) 的 **Manus AI**。
   - 一位成员指出 Chetan Near 参与了 **Manus** 和 **Groq** 的退出（exits），并思考这是否是一张*未来动向图*。
- ****Z.ai 宣布于 2026 年 1 月 8 日进行 IPO****：**Z.ai** 已正式宣布其即将到来的 IPO 计划于 **2026 年 1 月 8 日**举行，并对社区表达了感谢。
   - 来自 [Z.ai IPO](https://xcancel.com/Zai_org/status/2005934776042095052) 的公告感谢了开发者和研究人员自其成立以来的支持。
- ****王正东（Zhengdong Wang）发布 2025 年 AGI 信函****：王正东发布了他的 **2025** 年度信函，在[这封信](https://xcancel.com/zhengdongwang/status/2005848098531106916?s=61)中探讨了“感受到 AGI”（feeling the AGI）的主观体验及其社会影响。
   - 该信函涵盖了从算力（compute）和二阶效应到像 **Andor** 这样的文化引用以及 **Isaiah Berlin** 的哲学等主题。
- ****Nvidia 洽谈收购 AI21****：据报道 [Nvidia 正就人才收购（acquihire）AI21 进行高级阶段的谈判](https://uk.finance.yahoo.com/news/nvidia-advanced-talks-buy-israels-171025289.html)。
   - 一位成员推测他们是否拥有自己的模型。
- ****Manus Context Engineering 博客文章曝光****：一位成员分享了 [Manus 关于上下文工程（context engineering）的博客](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)链接，指出该文章提供了性能提升的方法。
   - 他们链接到了 [wedlm.github.io](https://wedlm.github.io/)，并声称在 vLLM 上比普通的 ar 模型有 **3 到 6 倍的加速**。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1455415863707435183)** (4 messages): 

> `Multi-Head Attention Projections, Q/K/V Role in Embeddings, Influence by Position in Azure Heads` 


- **Azure Heads 受位置影响**：在 **Azure** 中，某些 head 比其他 head 更受位置影响，从而产生了不对称性。
   - 然而，由于模型可以学会处理它，这*应该不是问题*。
- **关于多头注意力投影（Multi-Head Attention Projection）的辩论**：一位用户质疑为什么 **Multi-Head Attention** 中的 Embedding 在切片之前会被线性投影到 **Q, K, V**，询问是否可以直接对原始 Embedding 进行切片。
   - 该用户链接了 [State of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025) 以支持其观点。
- **深入探讨 Q/K/V 语义需求**：一位成员询问了 Embedding 中 **Q/K/V** 投影的语义需求，理解查询（querying）、匹配（matching）和值传输（value transport）的作用，但质疑在按 head 切片之前进行投影的必要性。
   - 另一位成员建议重新阅读 EAI Discord 上之前关于该话题的解释以进行澄清。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/)** (1 messages): 

k_nearest_neighbor: https://bigthink.com/the-present/the-rise-of-ai-denialism/
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1455457719439069194)** (4 messages): 

> `Mojo Memory Structures, Mojo for systems programming` 


- ****掌握 Mojo 内存映射****：一位成员编写了一份指南，解释了 Mojo GPU 编程中的**内存层级（memory hierarchy）**，涵盖了 `UnsafePointer`、`LayoutTensor`、`NDBuffer` 和 `DeviceBuffer` 等结构，以及 CPU/GPU 内存差异，并提供了 [MOJO_MEMORY_CONSIDERATIONS.md 文件的链接](https://cdn.discordapp.com/attachments/1151418092052815884/1455457719132749931/MOJO_MEMORY_CONSIDERATIONS.md?ex=695574e1&is=69542361&hm=429972ad3fa5ddd0cc01f3f796b5b14218c768f4292cbeeaf29a89665a6a1961&)。
- ****审视 Mojo 的系统编程范畴****：成员们讨论了 Mojo 是否可以像 C/C++ 和 Rust 一样作为一种**系统编程语言**使用。
   - 一位成员表示，将其作为系统语言使用*目前由于缺失语言和标准库（stdlib）特性而显得有些混乱，但这就是目标*。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1455414280567525458)** (2 messages): 

> `llm-ax Tool, axllm GitHub Repository` 


- **llm-ax 工具被认为开发完善**：一位成员指出，目前 [llm-ax](https://axllm.dev/) 似乎是可用的开发完善的工具。
- **分享 axllm 的 GitHub 仓库**：一位成员分享了 [axllm GitHub 仓库](https://github.com/ax-llm/ax)。