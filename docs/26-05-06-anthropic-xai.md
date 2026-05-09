---
companies:
- anthropic
- spacex
- x-ai
date: '2026-05-06T05:44:39.731046Z'
description: '**Anthropic** 宣布与 **SpaceX** 达成新的**计算合作伙伴关系**，以显著提升 **Claude** 系列产品的算力容量。此次合作将
  Pro、Max、Team 和 Enterprise 用户的 **Claude Code 5 小时速率限制**提高了一倍，取消了高峰时段的限额削减，并大幅提升了
  **Opus** 模型的 API 速率限制。


  该协议允许 Anthropic 通过 **SpaceXAI** 访问 **Colossus 1**，预计 Claude 的推理（inference）任务很快将在
  Colossus 上大规模运行。此外，Anthropic 还举办了 **“Code with Claude”** 活动，重点介绍了 Claude Code 的更新、GitHub
  规模的使用案例以及托管代理（managed agents）。相关讨论还涉及了算力瓶颈、用户对限额调整的反应、关于托管代理功能的争议，以及围绕 AGI（通用人工智能）可信度的持续安全与治理探讨。'
id: MjAyNS0x
models:
- claude
- claude-code
- opus
- colossus-1
people:
- nottombrown
- _aidan_clark_
- kipperrii
- theamolavasare
- alexalbert__
title: Anthropic 与 xAI 达成了关于 Colossus I 超级计算机的协议：规模达 300MW，每年价值 50 亿美元；其 ARR（年度经常性收入）年化增长率达
  8000%。
topics:
- compute
- rate-limiting
- agent-platforms
- inference
- api
- managed-agents
- safety
- governance
- event
---

**平静的一天。**

> 2026年5月5日至5月6日的 AI 新闻。我们检查了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有新增 Discord 信息。[AINews 网站](https://news.smol.ai/) 允许您搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以 [选择开启/关闭](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) 邮件接收频率！


今天是 Anthropic 的第二届年度开发者大会，现场氛围极佳。虽然没有一些（预测失准的）人所期待的大模型发布，但主要宣布了与 SpaceX 的合作伙伴关系（有望挑战 Claude 史上最大规模的发布）、Claude Managed Agents 的 3 个新功能，以及对过去 6 个月发布的所有内容的总结、重新介绍和庆祝：


### 开场演讲 (opening keynote)
在 Elon 批准后（可能具有战略意义，因为他起诉 OpenAI 的诉讼正在审理中），Anthropic 正以惊人的速度（“在接下来的几天内”）接管整个 Colossus 1。据估计，这是一笔每年约 50 亿美元的交易，使 xAI 成为了一家新云厂商（neocloud）：




另一个重头戏是与 Amodei 兄妹的圆桌访谈，宣布了 80 倍的增长，并对美国和中国的竞争对手发表了一些评论：




Dario 正在关注的趋势：

**初创小团队 (Tiny Teams)**：他仍然认为 2026 年我们将看到单人估值 10 亿美元的公司。“个人或极少数人有能力完成一系列不可思议的事情……以前，如果你有一个想法或愿景，你需要花费数年积累大量资源才能实现。我认为现在对于个人或极微型团队来说，有一个独特的机会来完成这些不可思议的事。我们正从‘模型在写代码’转向‘模型帮助我们将软件工程视为一项任务’，再到‘模型帮助我们将‘如何建立一个商业或经济单元’视为一项任务’”。

**多智能体 (Multiagents)**：“从房间里的一群聪明人开始，逐步发展到‘数据中心里的天才国度’”。

**企业服务 (Enterprise Services)**：“Claude Code 帮助个人提高生产力，但我们将越来越多地帮助整个团队和组织提高效率，使其产出大于各部分之和”。

**瓶颈 (Bottlenecks)**：Claude 当然在加速 Claude 自身，但他也在思考阿姆达尔定律（Amdahl’s Law）——安全性、可验证性——寻找软件工程中的瓶颈并消除它们，从而加速整个流程。


主舞台的其他环节包括：

**必须了解的 Claude Code 更新：**




**关于内环（Inner Loop）与外环（Outer Loop）的更多成果内容……**




**……用于 Agent 的自动改进：**








---

# AI Twitter 回顾

**头条新闻：Anthropic 和 Claude 的发布与评论**

## What happened

**Anthropic 经历了一个密集的新闻周期，重点围绕算力、Claude Code 限制以及 Agent 平台方向。** 官方层面，Anthropic 宣布与 SpaceX 建立新的算力合作伙伴关系，这将“大幅增加”容量，并立即转化为 Claude 产品更高的使用限制：[@claudeai](https://x.com/claudeai/status/2052060691893227611) 表示该交易提供的算力足以提高使用限制。随后 [@claudeai](https://x.com/claudeai/status/2052060693269008586) 补充了具体细节：**Claude Code 的 5 小时速率限制在 Pro、Max、Team 以及按席位计费的 Enterprise 版本中翻倍；移除了 Pro 和 Max 在高峰时段的限制削减；Opus API 的速率限制大幅提升**。xAI 将该交易描述为 Anthropic 通过 SpaceXAI 获得了 **Colossus 1** 的访问权限，以“为 Claude 提供额外容量” [@xai](https://x.com/xai/status/2052060350770515978)，同时 Anthropic CTO Tom Brown 补充道，**Claude 的推理将在“未来几天内”在 Colossus 上逐步上线** [@nottombrown](https://x.com/nottombrown/status/2052062566126649448)。公司还举办了 **“Code with Claude”** 活动，包括直播主题演讲以及关于 Claude Code、GitHub 规模级使用和托管型 Agent 的环节 [@ClaudeDevs](https://x.com/ClaudeDevs/status/2052055459272761661)，引发了开发者和观察者的实时热议 [@simonw](https://x.com/simonw/status/2052055655230706032), [@latentspacepod](https://x.com/latentspacepod/status/2052062150332710942)。围绕这些，讨论分成了四个主题：**(1) 算力瓶颈比许多人预想的更严重，据称是因为使用量超预期增长；(2) 用户对 5 小时限制的提升表示欢迎，但对未变的周限制表示质疑；(3) 人们争论 Anthropic 新推出的托管型 Agent 功能（如 Memory/“Dreaming” 和 Rubrics/“Outcomes”）是真正的产品差异化优势，还是可商品化的外壳功能；(4) Anthropic 的安全/治理立场继续吸引着赞扬与批评**，包括批评者称部分 Anthropic 员工表现出“只有我们值得被托付 AGI”，而与 Anthropic 相关的人士则反驳称，内部更普遍的观点是“没有任何人值得被托付 AGI”，而非“只有我们” [@_aidan_clark_](https://x.com/_aidan_clark_/status/2052089187659346047), [@kipperrii](https://x.com/kipperrii/status/2052094851991392536)。


## Official facts and confirmed details

- Anthropic 宣布与 **SpaceX 建立算力合作伙伴关系**以增加容量 [@claudeai](https://x.com/claudeai/status/2052060691893227611)。
- Anthropic 表示以下措施立即生效：
  1. **Claude Code 的 5 小时速率限制翻倍**（适用于 Pro, Max, Team, 以及按席位计费的 Enterprise）
  2. **移除了 Claude Code 在 Pro 和 Max 版本上的高峰时段限制削减**
  3. **大幅提升 Opus 模型的 API 速率限制**
  来源：[@claudeai](https://x.com/claudeai/status/2052060693269008586)
- Anthropic 发布了关于更高使用限制和 SpaceX 算力交易的官方说明 [@claudeai](https://x.com/claudeai/status/2052060696255283346)。
- xAI 的公告将此安排描述为 **SpaceXAI 为 Anthropic 提供 Colossus 1 的访问权限**，以获取额外的 Claude 容量 [@xai](https://x.com/xai/status/2052060350770515978)。
- Anthropic CTO Tom Brown 表示，**Claude 的推理将在几天内开始在 Colossus 上运行** [@nottombrown](https://x.com/nottombrown/status/2052062566126649448)。
- Anthropic 产品/工程负责人 Amol Avasare 澄清，**周限制尚未提高**，因为只有**很小比例**的用户触及了周限制，而触及 5 小时限制的用户比例要大得多；随着算力的就位，未来可能会有更多变化 [@TheAmolAvasare](https://x.com/TheAmolAvasare/status/2052064611692904639), [@TheAmolAvasare](https://x.com/TheAmolAvasare/status/2052066157176426653)。
- Anthropic/Claude 举办了 **Code with Claude** 活动，涵盖了主题演讲、Claude Code 更新、GitHub 规模级使用以及托管型 Agent 等环节 [@ClaudeDevs](https://x.com/ClaudeDevs/status/2052055459272761661)。
- Anthropic 的 Alex Albert 宣传了此次活动，并随后将该公告总结为 **“更多芯片，更多 Claude”** [@alexalbert__](https://x.com/alexalbert__/status/2052067009605861764), [@alexalbert__](https://x.com/alexalbert__/status/2052065953173872912)。
- Claude Code 官方账号重申了 Pro/Max/Team 的限制提升 [@claude_code](https://x.com/claude_code/status/2052071730190123094)。

## 计算细节与规模声明

多条推文对 SpaceX/xAI 合作协议的规模增加了定量描述。这些内容**并非出自 Anthropic 的官方公告推文**，但流传广泛：

- [@_arohan_](https://x.com/_arohan_/status/2052065871552819647) 引用了 **“超过 300 兆瓦的新容量”以及“在一个月内部署超过 220,000 块 NVIDIA GPU”。**
- [@scaling01](https://x.com/scaling01/status/2052068218047545501) 声称 Colossus 1 包含 **约 150,000 块 H100、50,000 块 H200 和 30,000 块 GB200**。
- [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2052065017072386450) 重复了 **220,000 块 GPU** 的数据，并增加了一项未经证实的说法，即 Anthropic 已在 **Google TPU 上投入了 2000 亿美元**。
- [@eliebakouch](https://x.com/eliebakouch/status/2052066609896808473) 将该交易解读为 Anthropic 实际上获得了 **Colossus 1 的全部容量**，而不仅仅是闲置的 GPU。
- Elon Musk 随后表示 SpaceX/xAI 能够放心出租 Colossus 1，是因为 **xAI 已经将训练任务转移到了 Colossus 2** [@elonmusk](https://x.com/elonmusk/status/2052069691372478511)，而 [@eliebakouch](https://x.com/eliebakouch/status/2052068426152132722) 声称 Colossus 2 已经拥有 **约 50 万块 Blackwell**。

这些数据最好被视为**具有部分半官方性质，但尚未在 Anthropic 自身的公告线程中完全正式确认**。广泛的事实性结论比精确的库存明细更具说服力：**Anthropic 锁定了规模巨大的、近期的外部推理容量扩展。**

## 瓶颈确实存在的证据

一个反复出现的解读是，Anthropic 的限制确实源于算力，而不仅仅是定价或产品设计。

- [@kimmonismus](https://x.com/kimmonismus/status/2052059082886910251) 在直播期间/之后询问 Anthropic 是否会在**不增加额外费用的情况下将 Claude Code 的速率限制（rate limits）提高一倍**。
- [@kimmonismus](https://x.com/kimmonismus/status/2052118418174681572) 随后总结了 Dario/Daniela 访谈中的言论：**使用量意外增长了约 80 倍**，据称这导致了算力短缺，而 SpaceX 的交易是解决该问题的首次重大尝试。
- [@czajkadev](https://x.com/czajkadev/status/2052101699188248990) 明确将此次更新解读为 **算力是瓶颈** 的证明。
- [@theo](https://x.com/theo/status/2052114791045668894) 另外辩称，行业问题“不只是钱的问题，而是算力的问题”，尽管这是一个更宏观的观点，但与 Anthropic 的情况相吻合。
- [@scaling01](https://x.com/scaling01/status/2052069341609226550) 从这笔交易中总结出了一个宏观论点：**前沿实验室正面临严峻的算力约束，以至于不得不从竞争对手那里租用数据中心。**

这是数据集中最强烈的事实/市场信号之一：**只有在达成重大算力交易后，Anthropic 面向用户的速率限制才发生了实质性变化。**

## 产品影响：Claude Code、API 和托管 Agent

Anthropic 对用户的实际影响显而易见：

- **Claude Code 核心用户在 5 小时窗口内获得了更多可用的突发容量（burst capacity）。**
- **缓解了 Pro/Max 用户在高峰时段的节流（throttling）。**
- **Opus API 用户获得了更高的速率限制**，这对于 Agent 工作负载和生产环境集成至关重要。

此次活动还凸显了 Anthropic 在 Agent 方面的更广泛平台雄心。虽然官方的主要推文大多关于活动本身，但评论指向了如下功能：

- **Dreaming** = 内存 / 跨会话上下文
- **Outcomes** = 评分标准 / 评级 / 目标跟踪
- Agent 编排 / 托管 Agent 的方向

评论：
- [@RichNwan](https://x.com/RichNwan/status/2052085746526216601) 认为 Anthropic 正在通过 **Dreaming** 和 **Outcomes** 构建他们的“托管 Agent 平台”，但质疑这些功能与开源工具相比是否具有显著的差异化优势。
- [@eliebakouch](https://x.com/eliebakouch/status/2052156107313807690) 认为这些功能**对核心用户很重要**，特别是对于保持主 Agent 的上下文窗口（context window），以及使用独立的评分器来管理质量、安全性、奖励作弊（reward hacking）。
- [@latentspacepod](https://x.com/latentspacepod/status/2052068066167816369) 引用了 Anthropic 演讲者的观点，强调了**验证（verification）**、“例行程序（routines）是更高阶的 Prompt”，以及剩余的差距通常在于**部署/运营化**而非原始能力。

最后一点使 Anthropic 与行业大趋势保持一致，即从“单次对话聊天机器人”转向**具有内存、任务分解、评级和验证能力的结构化 Agent 系统**。

## 事实 vs 观点

### 证据最充分的事实主张
- Anthropic 建立了新的 **SpaceX 算力合作伙伴关系**，并立即提高了 Claude Code/API 的限制 [@claudeai](https://x.com/claudeai/status/2052060691893227611), [@claudeai](https://x.com/claudeai/status/2052060693269008586)。
- 每周限制**尚未**翻倍；Anthropic 员工表示，这是根据达到上限的用户群体有意为之的 [@TheAmolAvasare](https://x.com/TheAmolAvasare/status/2052064611692904639)。
- Anthropic 计划近期在 **Colossus** 上运行 **Claude 推理** [@nottombrown](https://x.com/nottombrown/status/2052062566126649448)。
- Anthropic 举办了一场 **Code with Claude** 活动，重点关注编程、生产部署和托管 Agent [@ClaudeDevs](https://x.com/ClaudeDevs/status/2052055459272761661)。

### 似是而非但缺乏直接证实的主张
- Anthropic 即将获得 **>300 MW / >220,000 个 NVIDIA GPU** 的访问权限 [@_arohan_](https://x.com/_arohan_/status/2052065871552819647)。
- Colossus 1 的库存构成包括 **H100/H200/GB200 混合方案** [@scaling01](https://x.com/scaling01/status/2052068218047545501)。
- Anthropic 的需求激增了约 **80 倍**，让领导层猝不及防 [@kimmonismus](https://x.com/kimmonismus/status/2052118418174681572)。

### 观点与解读
- Anthropic 在解决算力短缺问题上**等待了太久**，导致将大量增长空间拱手让给了 OpenAI/Codex：[@scaling01](https://x.com/scaling01/status/2052070594972090409)。
- 这笔交易证明了**算力并非持久的护城河**，因为顶尖实验室可以从任何提供算力的 hyperscaler/集群运营商那里租用容量：[@Dorialexander](https://x.com/Dorialexander/status/2052067579594707149)。
- 或者，从实际角度来看，这证明了相反的观点：**谁控制了部署的算力，谁就决定了谁能满足需求**。
- Anthropic 的平台功能**并没有太大的差异化**，因为开源工具可以复制它们：[@RichNwan](https://x.com/RichNwan/status/2052085746526216601)。
- 或者它们**具有足够的差异化**，因为第一方集成可以将模型行为、Memory、评估器和产品体验紧密结合。
- Anthropic 的文化异常专注于安全且“对人类有益”：Elon Musk 在会见 Anthropic 高层后表示，他印象深刻，且“没有触发我的邪恶探测器” [@elonmusk](https://x.com/elonmusk/status/2052069691372478511)。
- 相反，批评者继续将 Anthropic 描绘成在 AGI 治理方面过于家长式作风或排他主义 [@_aidan_clark_](https://x.com/_aidan_clark_/status/2052089187659346047)。

## 舆论中的不同观点

### 1) 积极 / 支持
大量回复将此视为用户的胜利，并证明 Anthropic 正在积极采取行动。

- [@alexalbert__](https://x.com/alexalbert__/status/2052065953173872912)：“更多芯片，更多 Claude。”
- [@_sholtodouglas](https://x.com/_sholtodouglas/status/2052062164467224971)：“更多算力 -> 直接惠及用户。”
- [@kimmonismus](https://x.com/kimmonismus/status/2052059448261177367) 强调了限额翻倍并提高了 Opus API 的上限。
- [@TheRundownAI](https://x.com/TheRundownAI/status/2052064469371470218) 将其总结为直接的用户利益。
- [@DannyLimanseta](https://x.com/DannyLimanseta/status/2052078750893056420) 赞赏公司间的合作，并希望 Anthropic 的谨慎能与 SpaceXAI 的乐观态度达成平衡。
- [@AmandaAskell](https://x.com/AmandaAskell/status/2052161052058833181) 对该公告的象征意义做出了积极反应。

### 2) 中立 / 务实
这些观点欢迎这一变化，但更关注运营细节和仍然存在的限制。

- [@btibor91](https://x.com/btibor91/status/2052067002412335435) 和 [@kimmonismus](https://x.com/kimmonismus/status/2052061694080188720) 立即注意到一个可能的疑点：**每周限额保持不变**。
- [@TheAmolAvasare](https://x.com/TheAmolAvasare/status/2052064611692904639) 直接回答了这个问题。
- [@sbmaruf](https://x.com/sbmaruf/status/2052119971820658771) 报告称在更改后仍然看到 Rate Limit（速率限制），这意味着部署和可靠性调整仍在进行中。
- [@zachtratar](https://x.com/zachtratar/status/2052161984968396819) 呼吁在分阶段推广期间保持耐心。

### 3) 竞争 / 战略评论
另一组观点从 OpenAI 与 Anthropic 的产品战争角度看待此次发布。

- [@scaling01](https://x.com/scaling01/status/2052070594972090409) 认为 Anthropic 因等待太久而**错失了增长优势**，可能将数十亿美元的 ARR 让给了 OpenAI。
- [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2052065017072386450) 将此举解读为由于 **OpenAI Codex 的增长**，Dario 变得更具进攻性。
- [@_arohan_](https://x.com/_arohan_/status/2052053181656641735) 戏称“大厂已成为 Claude 的套壳（wrapper）”，指出了 Claude 在开发者心智中的地位。
- [@dejavucoder](https://x.com/dejavucoder/status/2052051193376231845) 说道“Claude 宕机了，请圣 Tibo 重置 Codex 限制”，捕捉到了当一个服务容量受限时，编程工具用户在多个产品间切换的现实情况。

### 4) 治理 / 安全 / 文化评论
这是最深刻的哲学分歧。

- [@_aidan_clark_](https://x.com/_aidan_clark_/status/2052089187659346047) 批评了他自称从 Anthropic 同事那里反复听到的一种观点：即相信只有他们才值得被信任去构建 AI。
- [@kipperrii](https://x.com/kipperrii/status/2052094851991392536) 部分同意“只有我们能被信任”的说法很糟糕，但认为真正的多数观点更接近于**“任何人在 AGI 面前都不值得被信任”**，即便其个人对 Anthropic 的信任度仍高于其他公司。
- [@elonmusk](https://x.com/elonmusk/status/2052069691372478511) 在会见 Anthropic 领导层后，给出了令人意外的认可。
- [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2052080339364004317) 考虑到此前对 Anthropic 的批评，称这种反转具有讽刺意味。
- [@teortaxesTex](https://x.com/teortaxesTex/status/2052080900280557749) 嘲讽了 Musk/xAI 与 Anthropic 之间迅速达成的缓和。
- [@teortaxesTex](https://x.com/teortaxesTex/status/2052045988936683674) 还认为，一方面警告他人 AI 风险，一方面又在构建如“Mythos”这样强大的封闭系统是不一致的。
- [@goodside](https://x.com/goodside/status/2052077014346064372) 虽然并非直接针对 Anthropic 的治理，但也参与了通常围绕 Anthropic 展开的更广泛的道德/AI 规范辩论。

## 关于 Claude 模型性能和对比的评论

尽管这些推文中没有出现重大的 Claude 新模型，但 Claude 仍然是产品和评估论述中的基准。

- [@giffmana](https://x.com/giffmana/status/2051925008457273527) 在一个数学争议点上对比了“Opus 4.6”、ChatGPT Pro 和 Muse Spark。他的看法如下：  
  - **Opus 4.6** 自信地辩护了一个错误的证明（“误导/Gaslit”）  
  - **ChatGPT Pro** 正确地统一了公式，但没有给出解释  
  - **Muse Spark** 两者都做得很好  
  这虽然是偶发案例，但却是这组数据中较为具体的模型定性对比报告之一。
- [@kimmonismus](https://x.com/kimmonismus/status/2052040471829004627) 总结了一份 Substack 分析，声称 **GPT-5.5 在网络安全方面基本与 Claude Mythos Preview 持平**，可能性价比更高，而 Mythos 仅在某些通用基准测试和 SWE-bench Pro 上略微领先；他质疑为什么 Mythos 仍然保持神秘。
- [@AssemblyAI](https://x.com/AssemblyAI/status/2052043337751056733) 注意到其网关支持来自 **Claude 4.5+ 模型**的结构化 JSON。
- [@OpenRouter/TencentHunyuan](https://x.com/TencentHunyuan/status/2051978552900538403) 将 **Claude Code** 列为推动 Hy3 使用的主要应用之一，展示了即使在后台使用第三方模型的情况下，Claude 在编程工具生态系统中的重要性。

这些评论并没有建立硬性的模型排名，但它们确实表明 Claude 仍然是 Coding Agent 工作流中的主要基准，且高级用户越来越多地比较**模型 + 框架 + 限制 + 可靠性**，而不仅仅是基础智能。

## Claude Code 和 harness 工程背景

数据集中一个值得注意的背景趋势是，许多工程师现在认为 **Agent 性能严重依赖于 harness**——即系统提示词（system prompts）、工具、中间件、分解策略以及模型特定的调优。

相关的非 Anthropic 评论：
- [@masondrxy](https://x.com/masondrxy/status/2052054177749029164)：同样的模型，同样的任务，根据提示词/工具/中间件的不同，得分会有很大差异；**在 tau2-bench 上有 10–20 分的跨越**。
- [@LangChain](https://x.com/LangChain/status/2052054711440662864)：针对 OpenAI、Anthropic 和 Google 模型的 harness 配置。
- [@jakebroekhuizen](https://x.com/jakebroekhuizen/status/2052058987580051566)：区分了随模型改进的**纵向 harness 演进**与**模型家族间的横向调优**。
- [@Vtrivedy10](https://x.com/Vtrivedy10/status/2052100726608781363)：认为定制的 harness 在许多任务上可以超越默认的 Codex/Claude Code；对于许多 Agent 设计来说，可用的 context windows 实际上仍然在 **50–100k** 之间。
- [@kieranklaassen](https://x.com/kieranklaassen/status/2052092428438688027)：“如果你无法在 Claude CLI 中完成工作，Claude 就无法为你工作。”

这一点很重要，因为 Anthropic 的一些平台举措——内存、评分、托管 Agent——可以被解读为 **Anthropic 正在将 harness 的部分功能产品化**。这有助于解释核心争论：**这些是具有防御性的平台原语（primitives），还是仅仅是对开源框架可以克隆的模式进行的第一方包装？**

## 更广泛的背景：为什么这很重要

1. **Inference（推理）而不仅仅是训练，现在成了前沿瓶颈。**  
   这次新闻不是发布新模型，而是发布产能。这在前沿领域正变得越来越普遍。

2. **算力市场正在变得流动且具有战略意义。**  
   Anthropic 与 SpaceX/xAI 基础设施的合作，打破了每个前沿实验室仅依靠自身垂直整合栈（vertically integrated stack）的简单叙事。

3. **开发者产品的份额对可靠性和限制非常敏感。**  
   Claude 似乎对开发者有很强的吸引力，但 rate limits（速率限制）和停机（outages）会迅速将用户推向 Codex/Cursor 或其他工具。

4. **战场正在从基础模型转向 Agent 系统。**  
   “Code with Claude”、托管 Agent、Dreaming、Outcomes 以及周围的讨论都表明，下一阶段的竞争重点将是**内存（memory）、编排（orchestration）、evals（评估）和工作流集成**。

5. **Anthropic 的品牌形象依然是分化的。**  
   它同时：
   - 因产品质量和对安全性的严肃态度而受到赞赏，
   - 因家长式作风或被感知的排他主义而受到批评，
   - 并且现在被认为在算力方面比以前更具商业进攻性。

## 底线结论

Anthropic 的新闻重点不在于一个华丽的新模型，而在于一个结构性现实：**对 Claude 的需求已经超过了可用算力，而 Anthropic 的回应是达成一项重大的外部基础设施协议，并立即放宽了关键的用户限制** [@claudeai](https://x.com/claudeai/status/2052060691893227611), [@claudeai](https://x.com/claudeai/status/2052060693269008586)。最重要的技术/经济信号是，**产能、rate limits 以及 Agent 产品的易用性，现在与排行榜上的分差（leaderboard deltas）具有同等的战略重要性**。主要的悬念在于，Anthropic 是否能将这种产能转化为持续的产品势头，其托管 Agent 功能是否真正具有差异化，以及其安全/治理姿态在与 OpenAI、Google、xAI 和开源模型生态系统竞争加剧的情况下，究竟是助力还是阻力。


**基础设施、Inference 与系统**

- OpenAI 及其合作伙伴发布了 **MRC (Multipath Reliable Connection)**，这是一种针对大型 AI 训练集群的开放网络协议，已部署在 OpenAI 最大的超级计算机上 [@OpenAI](https://x.com/OpenAI/status/2052025532485902368), [@OpenAI](https://x.com/OpenAI/status/2052025533937103102)。评论强调了多路径路由、微秒级故障转移，以及网络正成为主要的瓶颈 [@kimmonismus](https://x.com/kimmonismus/status/2052011784023028060), [@gdb](https://x.com/gdb/status/2052059553542328829)。
- Perplexity 表示其构建了自研推理引擎 **ROSE**，涵盖了从 Embeddings 到万亿参数规模的 LLM，并使用 **CuTeDSL** 来加速在 Hopper 和 Blackwell 架构上的专用 Kernel 开发 [@perplexity_ai](https://x.com/perplexity_ai/status/2052041903970148647)。
- vLLM + Mooncake 展示了针对 Agent 工作负载（具有可重用前缀）的强大系统成果：**3.8倍吞吐量**、**P50 TTFT 降低 46倍**、**端到端延迟降低 8.6倍**，以及缓存命中率从 **1.7% 提升至 92.2%**，并扩展到了 **60 台 GB200 GPU** [@vllm_project](https://x.com/vllm_project/status/2052113331927060840)。
- Unsloth + NVIDIA 发布了三项训练优化技术，据称可使家用 GPU 的 LLM 训练速度提升 **约 25%**：Packed-sequence 元数据缓存、双缓冲 Checkpoint 重载以及更快的 MoE 路由 [@UnslothAI](https://x.com/UnslothAI/status/2052020656527532276)。
- NVIDIA 关于 **RL 内部无损投机采样 (Lossless Speculative Decoding)** 的工作备受关注，该技术在不改变策略分布的前提下，使 235B 规模的端到端 RL 速度提升高达 **约 2.5倍**，使 8B 规模的 Rollout 吞吐量提升 **约 1.8倍** [@TheTuringPost](https://x.com/TheTuringPost/status/2052180472206381268)。
- Baseten 推出了 **Frontier Gateway**，作为面向闭源模型实验室的托管基础设施/API/认证/限流/计费方案；Poolside 报告称从启动到投产仅用了 **7 周**，其中 Laguna XS.2 的 **P50 TTFT 为 146ms**，Laguna M.1 为 **605ms** [@tuhinone](https://x.com/tuhinone/status/2052082677432390130), [@poolsideai](https://x.com/poolsideai/status/2052075055132057707)。


**基准测试、评估与 Agent Harness**


- **ProgramBench** 探讨了语言模型是否可以从头开始重建程序，其范围超出了修复类的 SWE 任务 [@ComputerPapers](https://x.com/ComputerPapers/status/2051895799043215415)；Ofir Press 认为基准测试是指定我们所期望未来的“藏宝图” [@OfirPress](https://x.com/OfirPress/status/2052106927908200957)。
- **Terminal-Bench 2.1** 修正了 TB2.0 中的 **28/89 个任务**；排名虽然保持不变，但绝对分数的变动高达 **12 分**，这提醒人们 Agent 基准测试的维护至关重要 [@terminalbench](https://x.com/terminalbench/status/2052119174500220964), [@ekellbuch](https://x.com/ekellbuch/status/2052165464655298866)。
- **OBLIQ-Bench** 作为一个重要的 IR 基准测试发布，专注于困难的第一阶段检索，目前已有的检索器很难从大型语料库中检索出微妙相关的文档 [@dianetc_](https://x.com/dianetc_/status/2052053806121140254)，该测试得到了 IR 研究人员的强烈认可 [@lateinteraction](https://x.com/lateinteraction/status/2052055143038713875), [@nlp_mit](https://x.com/nlp_mit/status/2052069072607547892), [@LightOnIO](https://x.com/LightOnIO/status/2052095548098822477)。
- Harvey 推出了 **LAB**，这是一个开源的长程法律 Agent 基准测试，涵盖 **24 个业务领域的 1,200 个任务**，并获得了 LangChain、Baseten、Artificial Analysis 等机构的支持与评论 [@saranormous](https://x.com/saranormous/status/2052061665596948894), [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2052145762650431840)。
- 多个推文中的一个主要主题是，**Harness 工程是一个一等变量**，即使使用相同的基座模型，它在 Agent 基准测试上通常也价值 **10–20 分** [@masondrxy](https://x.com/masondrxy/status/2052054177749029164), [@LangChain](https://x.com/LangChain/status/2052054711440662864), [@Vtrivedy10](https://x.com/Vtrivedy10/status/2052100726608781363)。


**模型发布与模型性能**

- Zyphra 发布了 **ZAYA1-8B**，这是一个**激活参数少于 1B** 的推理 MoE，在 **Apache 2.0** 协议下开源权重，声称在数学/推理效率上表现强劲，并且通过测试时计算 (test-time compute) 能够接近规模大得多的系统 [@ZyphraAI](https://x.com/ZyphraAI/status/2052103618145501459), [@ZyphraAI](https://x.com/ZyphraAI/status/2052103646712828119)。相关评论称赞了其架构/后训练技术栈以及与 AMD 的合作伙伴关系 [@teortaxesTex](https://x.com/teortaxesTex/status/2052106600882528326), [@eliebakouch](https://x.com/eliebakouch/status/2052126118891729148)。
- Google 的 **Gemma 4** 在 Code Arena 中推进了开源模型的帕累托前沿：在开源模型中，**Gemma-4-31B 排名第 13**，**Gemma-4-26B-A4B 排名第 17** [@arena](https://x.com/arena/status/2052061349312921686), [@_philschmid](https://x.com/_philschmid/status/2052104144706588699)。
- Google 为 Gemma-4 开发的 **DFlash 草图模型 (draft model)** 被描述为他们训练过的最好的草图模型之一，尤其在编程和数学方面表现出色 [@jianchen1799](https://x.com/jianchen1799/status/2051902953376923946)。
- Qwopus3.6-35B-A3B-v1 声称在**单张 RTX 5090 上可达 162 tok/s**，目标是在消费级硬件上实现强劲的 one-shot 前端/Web 生成 [@KyleHessling1](https://x.com/KyleHessling1/status/2052064943999267212)。
- 关于 DeepSeek 的评价褒贬不一：据报道，由中国主要的国家背景半导体基金领投的融资谈判目标估值为 **450 亿美元** [@jukan05](https://x.com/jukan05/status/2051904572038455634)；与此同时，评估人员正在争论 V4-Pro 在 WeirdML 上的表现弱于 GLM/Kimi 以及其他开源竞争对手 [@htihle](https://x.com/htihle/status/2052042076196335658), [@teortaxesTex](https://x.com/teortaxesTex/status/2052043753892761882)。


**Agents, tools, and developer workflows**


- Cursor 增加了跨规则、技能、MCP 和子 Agent 的**上下文使用明细 (context usage breakdowns)**，以帮助调试上下文问题 [@cursor_ai](https://x.com/cursor_ai/status/2052059748544249918)，并描述了使用早期的 Composer 模型来引导 (bootstrapping) 未来 Composer 生成的过程 [@cursor_ai](https://x.com/cursor_ai/status/2052116064474161556)。
- Cognition 在 Windsurf 2.0 中发布了 **Devin Review** 和 **Quick Review / SWE-Check**，明确针对审查 AI 生成代码这一新瓶颈 [@cognition](https://x.com/cognition/status/2052100630626607189), [@ypatil125](https://x.com/ypatil125/status/2052122827961278833)。
- OpenAI 推广了 **Codex 子 Agent (subagents)**，将其定位为在专门的 Agent 之间分配工作并将结果合并为单一答案的一种方式 [@reach_vb](https://x.com/reach_vb/status/2052090279344120278)。
- Nous/Hermes 继续推进高度可插拔的本地 Agent 技术栈：包括插件扩展、社区文档、Windows/WSL2 设置指南以及用例聚合 [@Teknium](https://x.com/Teknium/status/2052046335583625629), [@witcheer](https://x.com/witcheer/status/2052033039379673374), [@NousResearch](https://x.com/NousResearch/status/2052140057222369541)。
- Perplexity 在其 Agent API 中增加了 **Finance Search**，包含授权数据、实时市场数据和引用，声称在 **FinSearchComp T1** 上具有同类产品中最高的准确率和最低的正确答案成本 [@perplexity_ai](https://x.com/perplexity_ai/status/2052028012313649194), [@AravSrinivas](https://x.com/AravSrinivas/status/2052033959555735752)。
- Google 的 Gemini API 为 File Search 增加了**多模态检索**，在单一检索管线中使用 `gemini-embedding-2` 处理 PDF 和图像 [@_philschmid](https://x.com/_philschmid/status/2052060912425546050)。


**Robotics, multimodality, and research notes**

- Genesis AI 推出了 **GENE-26.5**，描述了一个包含机器人原生基础模型、类人手、数据手套和模拟器的全栈机器人程序；该模型在 **语言、视觉、本体感受、触觉和动作** 方面进行了训练 [@gs_ai_](https://x.com/gs_ai_/status/2052050956272230577), [@theo_gervet](https://x.com/theo_gervet/status/2052057035681018359)。
- Meta FAIR 发布了 **NeuralBench**，这是一个采用 MIT 开源协议的 NeuroAI 统一基准测试框架，包含 **36 个 EEG 任务** 和 **94 个数据集**，并计划支持 MEG/fMRI [@hubertjbanville](https://x.com/hubertjbanville/status/2052029372282888234), [@JeanRemiKing](https://x.com/JeanRemiKing/status/2052034314120896582)。
- Sander Dieleman 发表了一篇关于 **flow maps** 的长篇技术文章，探讨了学习 Diffusion 模型的积分以实现更快的采样及相关技巧 [@sedielem](https://x.com/sedielem/status/2051957402556104799)。
- François Fleuret 勾勒了一个增强系统的推测性方案：**类 Latent Diffusion 推理 + 真实循环状态 + 世界模型预预训练** [@francoisfleuret](https://x.com/francoisfleuret/status/2051928896027693479)，并引发了关于 Diffusion 风格推理是否能以正确方式外推的有用讨论 [@willdepue](https://x.com/willdepue/status/2052033422915477580), [@jeremyphoward](https://x.com/jeremyphoward/status/2052149483740545400)。
- HeadVis 作为一种用于研究 Attention Heads 的新型可解释性工具被推出 [@kamath_harish](https://x.com/kamath_harish/status/2052046203030827088)。
- Microsoft Research 关于 **Agent 可读可解释性** 的工作提出了 “Agentic-imodels”，即由 Coding Agent 演化出对其他 LLM 可解释的模型；报告称在 **65 个表格数据集** 上取得了提升，并将下游 BLADE 的改进从 **8% 提高到 73%** [@dair_ai](https://x.com/dair_ai/status/2052125514266190286)。


---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. MTP 与量化本地推理

  - **[Gemma 4 MTP 发布](https://www.reddit.com/r/LocalLLaMA/comments/1t4jq6h/gemma_4_mtp_released/)** (热度: 1575): **Google 发布了 Gemma 4 的 Multi-Token Prediction (MTP) 草案 Checkpoints** —— [`31B-it-assistant`](https://huggingface.co/google/gemma-4-31B-it-assistant)、[`26B-A4B-it-assistant`](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant)、[`E4B-it-assistant`](https://huggingface.co/google/gemma-4-E4B-it-assistant) 和 [`E2B-it-assistant`](https://huggingface.co/google/gemma-4-E2B-it-assistant) —— 详见 Google 的 [公告](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)。模型卡显示，MTP 通过一个较小的草案模型扩展了基础模型，用于 **Speculative Decoding**，其中草案模型预测未来多个 Token，而目标模型并行验证它们，声称可实现 **高达 `2x` 的解码加速**，且 *“质量与标准生成完全一致”*。一位评论者指出，最小的 `E2B` 变体使用了 **`78M` 的草案模型**，另一位分享了关于 Gemma 4 MTP 的技术图解 [链接](https://newsletter.maartengrootendorst.com/i/193064129/multi-token-prediction-mtp-with-gemma-4)。**

    - 一位评论者链接了关于 Gemma 4 **Multi-Token Prediction (MTP)** 的更新版图解说明，包括面向实现的片段：[Maarten Grootendorst 的指南](https://newsletter.maartengrootendorst.com/i/193064129/multi-token-prediction-mtp-with-gemma-4)。这有助于理解 Gemma 4 的 MTP 设置如何在每次前向传递中预测多个未来 Token，以及它如何与 Speculative/Draft 风格的解码进行交互。
    - 提到的一个技术细节是 **E2B 模型包含一个 `78M` 参数的草案模型**，这意味着一个用于加速生成工作流（如 Speculative Decoding）的轻量级辅助模型。草案模型的超小规模值得关注，因为它可以在减少解码延迟的同时，让验证器/主模型负责最终的 Token 接受。

- **[[通过 MTP 在 Qwen 3.6 27B 上实现 2.5 倍推理加速 - 本地 Agent 编码的理想选择 - 48GB 显存支持 262k 上下文 - 修复 Chat Template - 直接替代 OpenAI 和 Anthropic API 接口](https://www.reddit.com/r/LocalLLaMA/comments/1t57xuu/25x_faster_inference_with_qwen_36_27b_using_mtp/)]** (热度: 1445): **一个 llama.cpp 的 PR ([`pull/22673`](https://github.com/ggml-org/llama.cpp/pull/22673)) 为 **Qwen 3.6 27B MTP** 增加了支持，利用模型内置的 Multi-Token Prediction (MTP) 头进行投机解码 (Speculative Decoding)；作者报告在 M2 Max 96GB 上实现了 **~`2.5×` 的生成加速**，达到 **`28 tok/s`**，并发布了带有 MTP Tensor 的转换版 GGUF：[froggeric/Qwen3.6-27B-MTP-GGUF](https://huggingface.co/froggeric/Qwen3.6-27B-MTP-GGUF)。该配置结合了 `--spec-type mtp --spec-draft-n-max 5`、`q4_0`/`q8_0` KV-cache 量化，以及高达 **`262144` token** 的长上下文，声称在 **48GB Mac/显存级系统**上具有可行性；作者还在 [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) 上传了修复的非 vLLM 专用 Jinja Chat Template。注意事项：目前的 MTP 支持需要从 PR 分支构建 llama.cpp，`q4_0` KV 存在一定的质量损失，且 **Vision 功能目前在与 MTP 同时使用时会导致 llama.cpp 崩溃**；一位评论者在 RTX Pro 6000 MaxQ 上测试了 Qwen 3.6 2.7B Q8，速度从 **`36 tok/s` 提升至 `78 tok/s` (开启 MTP)**，同时指出 Prompt 处理速度下降了约 `20%`。** 评论区普遍反响热烈，认为近期开源模型和推理运行时的进展异常迅速，对消费级/本地硬件尤为重要。一个技术问题询问了 “turbo3/turbo4” 是否已合并，或者是该 MTP PR 的一部分。

    - 一位用户报告了在 **RTX Pro 6000 MaxQ** 上的具体 MTP 加速效果：开启 MTP 后，`qwen 3.6 2.7B Q8` 从 `36 tokens/s` 提升到 `78 tokens/s` ，而 Prompt 处理速度下降了约 `20%`。他们表示生成质量似乎没有变化，对于重度生成的任务负载，这种权衡非常值得。
    - 一位评论者询问 `turbo3`/`turbo4` 的更改是否已经合并，或者观察到的加速是否专门属于 **MTP PR**，这突显了对于哪种推理优化路径带来收益的不确定性。
    - 有人提出针对 **Qwen 3.6 Dflash** 模型和低比特 `iq3_XS` 量化的技术对比请求。评论者指出，他们通常可以在 `16GB` 显存中容纳 `256k` 上下文，并询问发布的量化版本在不使用 `mmproj` 时是否也能支持 `256k` 上下文。

  - **[[Qwen 3.6 27B 不同量化版本的质量对比 (BF16, Q8_0, Q6_K, Q5_K_XL, Q4_K_XL, IQ4_XS, IQ3_XXS,...)]](https://www.reddit.com/r/LocalLLaMA/comments/1t53dhp/quality_comparison_between_qwen_36_27b/)** (热度: 771): **一位 Reddit 用户在一个合成的“国际象棋转 SVG”任务（需要 PGN 状态跟踪、棋盘方向、棋子放置和末步高亮）上，对 **Qwen 3.6 27B** 的量化版本进行了基准测试。测试使用 `llama.cpp`，参数设置为 `temp=0.6`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.0`, `ctx=65536`。在这项单次运行测试中，**BF16/Q8_0** 基本正确，**Q6_K** 显示出棋子放置质量下降，**Q5_K_XL/Q4_K_XL/IQ4_XS** 仍基本可用，而 **Q3/Q2** 变体在布局/方向上失败率明显增加；作者选择 **IQ4_XS** 作为 `16 GB` 显存 RTX 5060 Ti 配置的实际底线。他们报告使用原生 `llama.cpp` 速度约为 `~100 pp tps / 8 tg tps`，而在使用 **TheTom’s TurboQuant fork**（设置 `-ngl 99`, `-ctk turbo4`, `-ctv turbo2` 及 `<75k` 上下文）时，速度提升至 `~760 pp tps / 22 tg tps`；完整输出已发布在 [qwen3-6-27b-benchmark.vercel.app](https://qwen3-6-27b-benchmark.vercel.app/)。** 顶层技术反馈称赞了该基准测试，但也强调 *“单次运行是不够的”*，因为随机解码可能使个别量化结果成为异常值；不过评论者也指出，观察到的质量退化趋势大体符合预期。

    - 几位评论者提出了方法论方面的担忧：量化对比似乎依赖于每个测试的单次运行，这可能会产生 **统计噪声** 和误导性的质量差异。他们建议对每个量化版本运行多次以检测异常值，特别是因为即使在整体退化趋势可见的情况下，LLM 评估在每次运行之间也可能存在差异。
    - 讨论的一个技术结论是，**`4-bit` 量化可能仍然是实际的甜点位**，`3-bit` 被描述为比通常声称的更具可用性，而超过 `5-bit` 的提升与转向更大/更好的基础模型相比，收益可能在递减。一位评论者特别对比了极大型 `122B UD-Q3_K_XL` 模型与较小的 `35B IQ4_NL` 模型，以此论证模型规模的重要性可能超过更高比特量化带来的质量提升。


### 2. Agentic Coding 和成本基准测试

- **[DeepSeek V4 Pro 在我们的 Agentic 基准测试 FoodTruck Bench 上追平了 GPT-5.2 —— 10 周后，成本降低约 17 倍](https://www.reddit.com/r/LocalLLaMA/comments/1t47qbw/deepseek_v4_pro_matches_gpt52_on_foodtruck_bench/)** (热度: 478): **该图片是 FoodTruck Bench 的技术排行榜截图，显示 **DeepSeek V4 Pro** 位列第 `#4`，在从 `$2,000` 初始资金开始的 30 天 Agent 食品卡车模拟中，实现了 `$27,142` 的最终净值、`+1257%` 的 ROI、`51%` 的利润率、`$52,139` 的营收以及 `$26,492` 的利润 ([图片链接](https://i.redd.it/fx89f3w5n9zg1.png))。这支持了帖子中的主张，即 DeepSeek V4 Pro 的中值结果与 **GPT-5.2** 的差距在 ~`3%` 以内，而据报道在相同工作负载下成本便宜 ~`17×`，这使其在 API 成本大幅降低的情况下达到了该基准测试的前沿级别 (Frontier-tier) 水平。** 评论者们对此印象深刻，但对结果解读持怀疑态度：有人指出 **Claude Opus 4.6** 在利润上似乎遥遥领先，而另一位评论者则质疑，如果 **Gemma 4 31B** 能击败 **Sonnet 4.6**，那么该基准测试的公信力。此外，人们对缺失的最新 GPT 变体（如 “GPT 5.4/5.5”）感到好奇。

    - 几位评论者关注的是基准测试排名的含义，而非 DeepSeek 的头条结果：据报道，**Claude Opus 4.6** 在 **FoodTruck Bench** 上的利润比下一梯队模型高出约 `1.7×`，这表明尽管 DeepSeek V4 Pro 以更低的成本追平了 **GPT-5.2**，但在这个 Agentic 利润优化基准测试中，Claude 仍保持着显著领先。
    - 多位用户点名 **Gemma 31B** 是一个讨论不足的离群值：它出现在 FoodTruck Bench 的前 5 名中，据报道击败了 **Sonnet 4.6**，并且在 **EQBench** 上也表现出色。评论者质疑，如果这些排名属实，为什么相比小米/DeepSeek 的结果，Gemma 获得的关注较少。
    - 有人请求扩大对比集，加入更新或缺失的模型，特别是 **GPT-5.4/5.5**、最新的 **Qwen3.6** 模型，以及一个评论者预期可能超越 Gemma 的 `27B` 模型。隐含的担忧是，目前的基准测试表对于评估当前前沿和中型模型的竞争力可能不完整或已过时。

  - **[Claude Code @ Opus 4.7 对阵 OpenCode @ qwen3.6:27b。两者都交付了一个可玩的舒适 Roguelite 游戏。](https://www.reddit.com/r/LocalLLM/comments/1t49wld/claude_code_opus_47_vs_opencode_qwen3627b_both/)** (热度: 406): **一项 One-shot (单次尝试) 基准测试对比了 **Claude Code (Opus 4.7)** 与 **OpenCode (本地 Qwen3.6:27B)**，两者使用了相同的 VS Code devcontainers 和严格的绿地项目 (Greenfield) 提示词来开发一个原生 Canvas/FastAPI Roguelite 游戏；两者都生成了一个首轮即可运行的游戏，实现了移动、剑盾战斗、程序化生成世界、掉落物、交换 UI 和重启循环。Opus 耗时 ~`20 分钟`并使用了 `97k` tokens，而 Qwen 耗时 ~`15 分钟`并使用了 `64k` tokens（约少三分之一），尽管作者明确将该结论限制在明确定义的绿地开发工作，而非硬推理或现有代码库维护。由于 Reddit 的 `403 Forbidden` 访问限制，帖子中链接的视频 [`v.redd.it/h4awffniaazg1`](https://v.redd.it/h4awffniaazg1) 无法在抓取中访问。** 评论者关注于可复现性和本地模型的性能：有人索要完整提示词，而其他人则认为 **Qwen3.6 27B** 在编程/棘手问题上表现出奇强劲，比某些 MoE 替代方案更不容易产生幻觉，在许多编程任务上大致可与去年的 **Sonnet 4.5** 媲美。另一位评论者表示，如果“妥善利用”，`35B` 变体在大规模代码库编辑任务中表现良好。

    - 用户请求提供对比中缺失的关键复现细节：确切的提示词、运行本地 Qwen 所用的硬件，以及是否对 `qwen3.6:27b` 应用了任何 Quantization (量化)。这些细节非常重要，因为本地模型的吞吐量和编程质量会因量化水平、显存带宽以及 GPU 或 Apple Silicon 配置的不同而显著变化。
    - 一位评论者报告 `Qwen3.6 27B` 在 **M1 Pro** 上运行“非常缓慢”，但仍能很好地处理编程和棘手问题。他们声称它比 `35B A3B` 和 `Gemma MoE` 的幻觉更少，并估计它与去年的 `Sonnet 4.5` 大致相当，使其能胜任 “90% 的编程任务”。
    - 另一位用户认为，当“妥善利用”并赋予大型代码库上下文进行检查和编辑时，`35B` 模型表现强劲，这表明对于编程 Agent 工作流来说，编排/上下文管理可能与原始模型选择同样重要。

- **[DeepSeek V4 比以前便宜 17 倍，促使我开始实际衡量发送到云端的数据与在本地运行的数据。结果非常惊人。](https://www.reddit.com/r/LocalLLaMA/comments/1t4s6g2/deepseek_v4_being_17x_cheaper_got_me_to_actually/)** (Activity: 904): **一位开发者对为期 `10` 天的 coding-agent 使用情况进行了埋点分析，并针对本地 RTX 3090 上的 Qwen 3.6 27B 模型与云端模型重新运行了 `150` 个任务样本。结果发现：在文件读取/项目扫描/解释任务（占工作量的 `35%`）中，本地模型与云端模型表现一致的比例达 `97%`；在测试/样板代码/单文件编辑任务（占工作量的 `30%`）中，这一比例为 `88%`。在多文件调试（占工作量的 `20%`）和跨 `5+` 个文件的复杂架构/重构（占工作量的 `15%`）中，本地模型的质量有所下降（分别为 `61%` 和 `29%`）。因此，据称仅将后两类任务路由到云端，就能将 API 支出从 `$85/月` 削减至约 `$22/月`。** 评论者普遍认同“混合/本地优先”的工作流：一些人报告称几乎所有的编码任务都使用本地模型，仅在进行规划、监督、异常复杂的任务或法律/健康等非代码领域时，才升级到 Gemini/ChatGPT/Claude/Qwen/GLM 的免费层级或云端模型。一位评论者询问了任务类型路由器/测试框架的实现细节，这暗示目前缺失的关键技术组件是用于分类和调度的自动化层。

    - 几位评论者描述了 **混合本地/云端工作流**：本地模型处理大部分与代码相关的任务，而云端/免费 Web 层级（如 **ChatGPT, Claude, Gemini, Qwen, GLM**）或专门的 Gemini 则保留给规划、监督或罕见的复杂问题。一位用户报告称实现了 **零订阅** 运行，云端主要用于健康/法律咨询等非代码领域，因为这些领域对本地模型可靠性的容忍度较低。
    - 一个关键的技术异议是，本地模型在 **large contexts（大上下文）** 下可能较慢，并会通过额外的验证/调试时间产生隐藏成本。一位评论者认为，即使本地推理更便宜，那 `~10%` 本地模型表现不佳的情况也可能会主导生产力成本，并建议托管的 **Qwen 3.6 27B / Qwen 3.6 Pro** 可能更快，且每月仅需“几美元”。

## 技术性较低的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Anthropic Claude Code 的限制与可靠性

  - **[Claude Code 的 Rate Limits 翻倍](https://www.reddit.com/r/ClaudeCode/comments/1t5hs98/doubled_rate_limits_for_claude_code/)** (Activity: 3224): **Anthropic** 表示，通过与 **SpaceX** 建立新的算力合作伙伴关系以及近期达成的其他算力协议，该公司得以上调 Claude 的容量：**Claude Code** Pro/Max 计划不再受高峰时段限制，且 **Opus** 模型的 **Claude API** Rate Limits 正在“大幅”提高，立即生效 ([Anthropic 公告](https://www.anthropic.com/news/higher-limits-spacex))。该帖子将其描述为“Rate Limits 翻倍”，但引用的公告本身明确提到的是取消 Claude Code 的高峰时段限流以及提高 Opus API 的限制，并未给出确切的数值配额。热门评论大多是非技术性的惊讶/怀疑，以及对 Elon Musk 与 Sam Altman/OpenAI 竞争关系的猜测。


  - **[我受够 Claude 了。它已经变成了彻头彻尾的垃圾。](https://www.reddit.com/r/ClaudeCode/comments/1t4w5an/ive_had_it_with_claude_it_has_become_complete/)** (Activity: 1716): **一位高级 SWE 报告称，Anthropic [Claude](https://www.anthropic.com/claude) 在“Opus 4.7”版本相对于“Opus 4.6”出现了重大退化：CLI 交互变慢（提交需 `30s`，实现需 `45min`）、调整大小时终端/Tmux 渲染变差、失去了实用的 `Ctrl+O` 追踪可见性、使用限制触碰更频繁，且尽管进行了项目内存/上下文工程，指令遵循能力依然下降。引用的具体技术故障包括：忽略简短的测试超时（`10–15s` → `30s/60s/5min`）、无视“从不自动提交”指令进行自动提交、尽管使用了 `/caveman` 但输出依然冗长、在 Rust 重构中通过添加 `handle_input_bytes(Bytes)` 而非将 `handle_input(&[u8])` 更改为 `Bytes` 来实现、以及偏离了 `io_uring` 的 cancel-safety 计划，转而退回到一个存在竞争条件的 one-shot/multi-shot recv 快捷方案，随后才承认 *“是的，偏离了。我坦白。”*** 热门评论主要分为两派：一派认同失去可见的推理过程使得中断错误循环变得更加困难，部分用户取消了 Max 订阅并转向开源模型以追求稳定性；另一派则是由经验丰富的开发者提出的反对意见，认为在遵循严格的 `Claude.md`/`memory.md`、划定计划范围、设定里程碑并避免加载过多上下文的情况下，Claude 依然极具生产力。

- 一位资深软件开发人员报告称，通过使用受约束的项目工作流可以保持稳定的编码表现：维护良好的 `Claude.md` 和 `memory.md`，少量的技能设定，前期规划，基于里程碑的实现，以及重复的构建/测试/发布周期。他们认为，许多失败可能源于较差的上下文卫生（context hygiene）——要么是将“29 个不同的 Markdown 文件”作为超大的伪操作系统（pseudo-OS）加载，要么是在每个命令中倾倒整个上下文窗口。
- 一位用户强调了由于隐藏 Chain-of-Thought 风格的进度而导致的 UX/回归问题：没有可见的“思考（thinking）”过程，他们无法再分辨 Claude 是在内部循环还是在等待服务器端的延迟（latency）。这使得及早中断无效率的推理以及诊断延迟是模型行为还是基础设施相关问题变得更加困难。
- 几位用户报告了与时间相关的质量波动，其中一位特别声称在美东时间（US Eastern）上午 8 点至下午 2 点的高峰使用期间，Claude 的表现变差：更多的偷工减料、输出更草率以及“脑死”行为，而错峰使用时的质量则更接近之前。隐含的技术担忧是随负载增加的性能下降（load-dependent degradation），可能源于容量压力、路由、限流（throttling）或高峰期的模型/服务变更。

- **[将台灯变成 Claude Code 状态指示灯](https://www.reddit.com/r/ClaudeAI/comments/1t4gfc7/turned_a_desk_lamp_into_a_claude_code_status/)** (活跃度: 1817)：**一位 Reddit 用户改编了开源项目 [`bobek-balinek/claude-lamp`](https://github.com/bobek-balinek/claude-lamp)，将一个 BLE 台灯变为了 **Claude Code 状态指示灯**：Claude Code 钩子（hooks）调用一个 Python 脚本，通过 Bluetooth Low Energy 发送命令来设置动画/颜色。当 Claude 正在工作时，台灯显示**蓝色旋转动画**；当需要用户输入时显示**粉色**；闲置时显示**暖白色**。效果可在源代码中配置，作者正在考虑将该方案扩展到 **Philips Hue** 灯泡。由于 `403 Forbidden` 响应，链接中的 Reddit 视频无法访问。**评论者主要询问了灯的型号，并讨论了将该想法扩展到多个并发 Claude Code 会话的情况，例如使用多盏灯或设计更好的多会话状态指示器。一位评论者指出，标题也可能暗指通过 [`status.claude.com`](https://status.claude.com/) 显示 Anthropic 的服务健康状况。

    - 一位评论者建议将灯的功能扩展到本地 Claude Code 状态之外，以反映 **Claude 服务健康状况**，使用 Anthropic 的公开状态页面 [status.claude.com](https://status.claude.com/) 作为数据源。这将使指示灯代表运行可用性，而不仅仅是本地任务/会话状态。
    - 另一个提出的技术改进是可视化 **滚动五小时窗口内剩余的 Claude Code 使用量**，例如按比例点亮灯或“圆环”来显示剩余配额。另一条评论提出了多会话情况，暗示如果多个 Claude Code 会话并发运行，指示灯将需要进行聚合或处理每个会话的状态。

- **[警告：Anthropic 的 “Gift Max” 漏洞导致 800 多欧元损失、信用受损且账号被封。](https://www.reddit.com/r/ChatGPT/comments/1t4atbx/warning_anthropics_gift_max_exploit_drained_800/)** (活跃度: 3451)：**原帖主（OP）报告称，尽管启用了 `2FA`，其 Anthropic 账户仍出现了超过 **800 欧元**的未经授权 **“Gift Max”** 扣费；他们声称收到了 `3-D Secure` 邮件但从未授权，同时礼品代码被生成并立即被第三方兑换。他们将此事件与 Anthropic [状态页面](https://status.anthropic.com/)中关于*“结算错误增加和未经授权的订阅更改”*的条目以及 GitHub issue `#51404`/`#51168` 联系起来。随后，OP 表示 Anthropic 在收到警方报告和证据后封禁了该账户，切断了其对进行中对话/项目的访问权限。在更新中，OP 表示银行将其视为欺诈，发起了追回/退款，并指控 Anthropic 的商户账户；他们还考虑根据 [GDPR/DSGVO](https://gdpr.eu/) 提交数据请求以恢复数据，并寻求德国法律援助以修复可能受损的 [SCHUFA](https://www.schufa.de/) 信用记录。**评论大多很实际或持怀疑态度：有人指出在美国这通常通过信用卡退单（chargeback）处理，而另一人则强调了在 ChatGPT 的 Reddit 频道发布由 Gemini 撰写的反 Anthropic 警告的讽刺性/可疑性。

- OP 报告称，他们的银行已将 `€800+` 与 Anthropic 相关的费用作为诈骗案件处理并进行了撤回，并将直接追究商户账户的责任。他们还计划提交正式的 GDPR/DSGVO 数据请求以恢复进行中的项目数据，并寻求德国法律援助 (*Beratungshilfeschein*)，以确保清除任何 SCHUFA 信用记录条目。
- 一位评论者指出，他们看到来自不同商家的多个 YouTube 广告，都在宣传“1 年免费 Claude 访问权限”，这表明可能存在一场与所报道的漏洞利用或钓鱼/支付滥用模式相关的有组织诈骗活动。



# AI Discords

不幸的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快就会发布全新的 AINews。感谢阅读到这里，这是一段美好的历程。