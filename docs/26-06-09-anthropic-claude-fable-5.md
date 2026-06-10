---
companies:
- anthropic
- cursor_ai
- cognition
date: '2026-06-09T05:44:39.731046Z'
description: '**Anthropic** 发布了两款重量级模型：**Claude Fable 5** 正式全面开放，**Claude Mythos 5**
  则设为受限访问，针对敏感查询将自动回退至 **Claude Opus 4.8** 处理。**Fable 5** 具备 **100 万 token 的上下文窗口**，定价为**每百万输入
  token 10 美元**以及**每百万输出 token 50 美元**。


  该模型在软件工程、知识工作、科学研究及视觉等基准测试中均处于领先地位，性能超越了 **GPT-5.5**，并在 **CursorBench**、**FrontierCode**、**Terminal-Bench
  2.1** 和 **Artificial Analysis Intelligence Index** 上刷新了业内领先纪录（SOTA）。此次发布涵盖了 Pro、Max、Team
  和 Enterprise（企业）订阅方案，由于算力容量限制，目前提供临时使用额度。此外，Python、TypeScript、Go、Java 和 C# 的**中间件
  SDK** 支持也已同步上线。'
id: MjAyNS0x
models:
- claude-fable-5
- claude-mythos-5
- claude-opus-4.8
- gpt-5.5
people:
- mikeyk
- scaling01
title: Anthropic Claude Fable 5
topics:
- benchmarking
- software-engineering
- knowledge-work
- scientific-research
- vision
- context-windows
- model-pricing
- sdk
- rate-limiting
---

**平静的一天。**

> 2026年6月8日-2026年6月9日的 AI 新闻。我们检查了 12 个 subreddits，[544 个 Twitters](https://twitter.com/i/lists/1585430245762441216)，没有新增的 Discords。[AINews 网站](https://news.smol.ai/) 允许您搜索所有历史期数。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以[选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件发送频率！

---

# AI Twitter 回顾

**头条：Anthropic 发布 Claude Fable 5 和 Mythos 5**

## 发生了什么

**Anthropic 发布了其下一个主要模型系列的两个版本：面向公众开放的 Claude Fable 5 和受限访问的 Claude Mythos 5。**

- Anthropic 正式宣布 **Claude Fable 5** 为其“首个面向公众开放的 Mythos 级模型”，并表示它超越了以往发布的任何模型，在几乎所有测试的 **benchmarks** 上都达到了 **state-of-the-art** 水平 [@claudeai](https://x.com/claudeai/status/2064394146916229443), [@claudeai](https://x.com/claudeai/status/2064394151441863006)
- Anthropic 表示 **Fable 5 与 Mythos 5 的底层模型相同，但增加了 safeguards**，部分与网络/生物/化学/蒸馏（distillation）相关的 **prompts** 可能会被 **路由至 Claude Opus 4.8** [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064428347678220691), [@scaling01](https://x.com/scaling01/status/2064398688802205900)
- Anthropic 声明，对于“极少数”潜在有害话题，**查询将透明地 fallback 到 Opus 4.8**，并根据早期的用户界面提示，声称 **95% 以上的会话从未遇到过此情况** [@claudeai](https://x.com/claudeai/status/2064394155258765783), [@mikeyk](https://x.com/mikeyk/status/2064392996288901392)
- Anthropic 的开发者消息称，服务端以及 **Python, TypeScript, Go, Java, 和 C#** 的 SDK 中间件均支持 **fallback** [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064428351029449214)
- 据报道，**Fable 5 和 Mythos 5** 的定价均为 **每百万 input tokens 10美元，每百万 output tokens 50美元**；第三方评估机构随后报告缓存定价为 **每百万 cache writes 12.50美元，每百万 cache reads 1美元** [@scaling01](https://x.com/scaling01/status/2064394893603049625), [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992)
- 根据 Artificial Analysis 的数据，Fable 5 保留了 Anthropic 的 **1M-token context window** [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992)
- Anthropic 在 **6月22日之前将 Fable 5 纳入 Pro, Max, Team 以及基于席位的 Enterprise 计划**，之后表示由于容量限制，将需要使用 **usage credits**，并计划稍后恢复更广泛的订阅访问 [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064394931033248226), [@scaling01](https://x.com/scaling01/status/2064394893603049625), [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992), [@kimmonismus](https://x.com/kimmonismus/status/2064388066354028986)
- 关于临时包含政策的困惑立即显现；用户询问“截止至6月22日包含”是什么意思，Anthropic 员工对分阶段推出进行了澄清 [@dejavucoder](https://x.com/dejavucoder/status/2064393509990523102), [@TheAmolAvasare](https://x.com/TheAmolAvasare/status/2064393574431764928)
- 在需求激增后，Anthropic 随后在所有产品中 **重置了 5 小时和每周的 rate limits** [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064464557951852643)

## 官方声明与第三方基准测试数据


**Anthropic 及其合作伙伴平台报告了广泛的基准测试领先优势，特别是在编程和长程 Agent 任务方面。**

- Anthropic 的公开声明：Fable 5 在**软件工程、知识工作、科学研究和视觉**方面表现尤为强劲，且**其领先优势随任务长度和复杂度的增加而扩大** [@claudeai](https://x.com/claudeai/status/2064394151441863006)
- Cursor 表示 Fable 5 以 **72.9%** 的成绩创下了 **CursorBench SOTA** 新纪录，比之前的最高分高出 **8 个百分点** [@cursor_ai](https://x.com/cursor_ai/status/2064394824313376787)
- Cognition 表示 Fable 5 夺得了 **FrontierCode 的第 1 名**，Devin 已将其集成到 Devin Cloud Ultra、Desktop 和 CLI 中 [@cognition](https://x.com/cognition/status/2064398549073453266), [@cognition](https://x.com/cognition/status/2064398551539761387)
- Cline 报告 Fable 5 在 **Terminal-Bench 2.1 上达到 88.0%**，以 **4.6 个百分点**的优势击败了 GPT-5.5 [@cline](https://x.com/cline/status/2064427461212045546)
- Artificial Analysis 将 Fable 5 列为其 **Intelligence Index 的第 1 名，评分为 64.9**，领先 GPT-5.5 约 **5 个百分点**，并指出 Anthropic 占据了前两名 [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992)
- Artificial Analysis 还报告了：
  - **GDPval-AA Elo 1932**，在 Agent 现实世界知识工作方面排名第 1 [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064414308289937869)
  - 在 **Humanity’s Last Exam 上达到 53%**，领先第二名模型超过 **7 个百分点**，同时在 **9% 的 HLE 任务**中触发了 fallback [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992)
  - 在 Intelligence Index 任务中，**fallback 路由占比约 8%**，主要集中在科学问题上 [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992)
  - Anthropic 表示平均**只有不到 5% 的会话**会发生 fallback [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064414308289937869)
- 社区基准测试汇总强调了编程领域的巨大差距：
  - **SWE-Bench Pro：Fable 5 为 80.3%，而 GPT-5.5 为 58.6%** [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2064396097075003739)
  - **FrontierCode Diamond：Mythos 5 为 30.9%，而第二名仅为 13.4%** [@scaling01](https://x.com/scaling01/status/2064391295620010383)
  - **Mythos 5 的 Anthropic ECI 为 161.29** [@scaling01](https://x.com/scaling01/status/2064392088003756431)
- Artificial Analysis 指出，Fable 5 在 **AA-Omniscience** 知识基准测试中的飞跃，可能暗示其**模型规模比 Anthropic 之前的公开模型更大**，尽管这只是推论而非确认的规格 [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992)

## 产品行为、使用概况与部署详情

**此次发布的定义更多地取决于工作流的变化和成本概况，而不仅仅是原始评估指标（evals）。**

- Anthropic 员工和早期用户反复将 Fable 5 描述为一款针对**超长、高投入任务**的模型，用户的操作也从分配具体任务转变为赋予其**目标/职责** [@felixrieseberg](https://x.com/felixrieseberg/status/2064392202504310900), [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064399512664526853), [@alexalbert__](https://x.com/alexalbert__/status/2064467657483829441)
- Anthropic 建议用户默认选择 **xhigh/high effort**（极高/高投入）模式，重写旧的 CLAUDE.md 指令，并让模型行使更多的判断力 [@alexalbert__](https://x.com/alexalbert__/status/2064467657483829441)
- Anthropic 给开发者的信息强调了**多 Agent 编排（multi-agent orchestration）**，由 Fable 在 Claude Managed Agents 中向更小的模型分派任务 [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064394928948703406)
- 多位测试者将 Fable 描述为**缓慢、消耗 Token 且昂贵**，但能力异常强大：
  - Dan Shipper 表示，它在处理任务时通常会消耗 **50 万到 100 万个 Token**，最好留给繁重的工作使用 [@danshipper](https://x.com/danshipper/status/2064393970856124501)
  - Simon Willison 称其为“缓慢、昂贵且强大” [@simonw](https://x.com/simonw/status/2064501565738930433)
  - Theo 很快就达到了限制，随后对 Anthropic 重置速率限制（rate-limit）表示欢迎 [@theo](https://x.com/theo/status/2064442054772716020), [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064464557951852643)
- 第三方和内部轶闻强调了其在长时间运行的工程任务上的巨大进步：
  - Ethan Mollick 表示，他可以交给它一份 **15 页的设计文档**，它会持续工作 **9 个多小时** [@emollick](https://x.com/emollick/status/2064395281903346013)
  - Kimmonismus 强调了 Anthropic 的说法，即 Stripe 使用 Fable 在一天内完成了 **5000 万行 Ruby 代码的迁移**，取代了原本需要**整个团队耗时两个多月**的工作 [@kimmonismus](https://x.com/kimmonismus/status/2064401121515274747)
  - Victor Taelin 报告称 Fable 发现了一个微妙的 Bug，并在一个案例中实现了据称高达 **1770% 的加速**，尽管他仍然需要审计其正确性 [@VictorTaelin](https://x.com/VictorTaelin/status/2064448425936994742)
  - 与 Anthropic 相关的帖子引用了 **430 倍的内核（kernel）加速**、**69 倍的自训练（self-training）加速**以及 **10 倍的药物设计加速**，尽管这些数据源自基准测试/系统卡（system-card）的解读，除非得到独立验证，否则应视为厂商方面的说法 [@scaling01](https://x.com/scaling01/status/2064392386520780945), [@scaling01](https://x.com/scaling01/status/2064392809293939119), [@scaling01](https://x.com/scaling01/status/2064394250142265367)
- 生态系统同步展开：Fable 5 已出现在 **Cursor, Devin, Notion, Microsoft Foundry, GitHub Copilot App/CLI, Cline, Replit, Base44, MagicPath, Arena, MCP Atlas** 等应用中 [@cursor_ai](https://x.com/cursor_ai/status/2064394824313376787), [@cognition](https://x.com/cognition/status/2064398549073453266), [@NotionHQ](https://x.com/NotionHQ/status/2064397568696819984), [@Azure](https://x.com/Azure/status/2064421301108834552), [@pierceboggan](https://x.com/pierceboggan/status/2064402677614911818), [@cline](https://x.com/cline/status/2064427461212045546), [@pirroh](https://x.com/pirroh/status/2064408022651191613), [@ScaleAILabs](https://x.com/ScaleAILabs/status/2064473993919537578)

## 安全架构与主要争议


**最大的争论不在于 Fable/Mythos 是否强大，而在于 Anthropic 决定在某些前沿 AI 开发任务中悄悄降低模型的效用。**

- Anthropic 的 system-card 措辞被多位用户发现：当 Fable 5 被用于**前沿 LLM 开发**时，Anthropic 可能会通过 **prompt modification、steering vectors 和 PEFT** 来**限制模型的有效性**，且用户**不会收到通知**；Anthropic 估计这会影响大约 **0.03% 的流量** [@Hangsiin](https://x.com/Hangsiin/status/2064397550434816088), [@kimmonismus](https://x.com/kimmonismus/status/2064417460715962479)
- Anthropic 还另外披露了会将涉及**网络安全（cybersecurity）和生物安全（biosecurity）**的请求自动重定向至 Opus 4.8 [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064394931033248226)
- 这种区别至关重要：**一些风险查询会明显地被重定向为 Opus 并据此计费**，而**前沿 LLM 开发请求可能会被悄悄削弱，而不是重定向或拒绝**
- 批评者认为，这在研究和工程工作流中创造了一个**未被记录的干扰因素（unlogged confounder）**：
  - “付费产品中不应存在‘静默降级’（silent handicaps）” [@nrehiew_](https://x.com/nrehiew_/status/2064400440264179923)
  - “在不告知用户的情况下降低 ML 研究性能，这种行为极具敌意” [@deanwball](https://x.com/deanwball/status/2064434861088395730)
  - “安全干预必须是可见的、可审计的且可追溯的” [@MattGibsonMusic](https://x.com/MattGibsonMusic/status/2064518301888512486)
  - “这就是信任的破裂”，因为每个糟糕的结果都变得模糊不清 [@MattGibsonMusic](https://x.com/MattGibsonMusic/status/2064518301888512486)
- 几位研究人员将其定性为针对开放研究和开源权重（open weights）的**反竞争“过河拆桥”（ladder-pulling）**行为：
  - “实验室开始拆掉梯子了” [@natolambert](https://x.com/natolambert/status/2064404993193754830)
  - “这是保护和培育开源 AI 的最大警钟” [@rasdani_](https://x.com/rasdani_/status/2064409800641859747)
  - “他们指的不是暂停 AI 研究，而是暂停*你的* AI 研究” [@bayeslord](https://x.com/bayeslord/status/2064437399292203401)
  - “原创思考者不能沦为底层阶级” [@marksaroufim](https://x.com/marksaroufim/status/2064428421774753943)
  - “权力、能力和经济财富的集中是 AI 面临的最大风险” [@ClementDelangue](https://x.com/ClementDelangue/status/2064513229099876663)
- 多位用户担心分类器的边界过宽或过于容易出错：
  - 一位用户表示 “cancer（癌症）一词被标记为生物安全风险” [@DeryaTR_](https://x.com/DeryaTR_/status/2064414826122866707)
  - 另一位用户表示 Fable 拒绝回答 “心脏是做什么的？” [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2064524668208545955)
  - 生物学领域的用户报告了账户上下文的差异，包括可以在**无痕模式（Incognito Mode）下使用 Fable，但在普通模式下却不行** [@cremieuxrecueil](https://x.com/cremieuxrecueil/status/2064449457869984035)
  - Teknium 等人报告了简单工程提示词被拒绝的情况 [@Teknium](https://x.com/Teknium/status/2064462936677203983), [@Teknium](https://x.com/Teknium/status/2064466293185806658)
  - 用户报告 PTX ISA 问题和推理优化（inference optimization）查询被标记 [@snowclipsed](https://x.com/snowclipsed/status/2064408466039390417), [@dejavucoder](https://x.com/dejavucoder/status/2064420742129967331)
- 一些例子既幽默又尖锐：用户开玩笑说，请求推理代码会导致模型“开始导入 ONNX”或实现 JEPA，将其视为能力引导（capability steering）的迹象 [@vikhyatk](https://x.com/vikhyatk/status/2064515989795127744), [@MattVMacfarlane](https://x.com/MattVMacfarlane/status/2064440740483403829)

## 事实 vs. 观点


**事实 / 直接由发布材料或基准测试（benchmark）帖子支持**

- Fable 5 已全面开放（GA）；Mythos 5 为受限访问 [@claudeai](https://x.com/claudeai/status/2064394146916229443), [@TheRundownAI](https://x.com/TheRundownAI/status/2064394481923699070)
- Fable 5 和 Mythos 5 共享相同的底层模型，但在 Fable 上增加了额外的安全防护（safeguards） [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064428347678220691), [@scaling01](https://x.com/scaling01/status/2064398688802205900)
- 定价为 **每百万 input/output tokens 10 美元 / 50 美元** [@scaling01](https://x.com/scaling01/status/2064394893603049625), [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992)
- Fable 保留了 **1M 上下文（context）** [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992)
- Anthropic 引入了拒绝/回退机制（refusal/fallback mechanisms）和 SDK 中间件（middleware） [@ClaudeDevs](https://x.com/ClaudeDevs/status/2064428351029449214)
- Anthropic 披露了**针对前沿 LLM 开发的静默干预（silent interventions）**，影响约 **0.03% 的流量** [@Hangsiin](https://x.com/Hangsiin/status/2064397550434816088)
- Fable 在 **6 月 22 日**前暂时包含在订阅中，之后将改为基于额度（credit-based）计费 [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2064500150069030992)

**观点 / 解读**

- “Anthropic 赢了”、“Anthropic 拥有编程护城河（coding moat）”、“Anthropic 正在追求 ASI”等属于评论而非经过验证的事实 [@scaling01](https://x.com/scaling01/status/2064401880323653799), [@scaling01](https://x.com/scaling01/status/2064399642603802676), [@scaling01](https://x.com/scaling01/status/2064410532824662047)
- 认为此举主要是为了 **IPO 形象考量（optics）**、**反开源定位**，或专门为了减缓 **Meta/中国/开源实验室（open labs）** 的进展，这些是合理的解读，但尚未得到 Anthropic 的确认 [@kimmonismus](https://x.com/kimmonismus/status/2064448699632402664), [@kylebrussell](https://x.com/kylebrussell/status/2064502244041511348), [@natolambert](https://x.com/natolambert/status/2064412173527556298)
- 认为 Anthropic 是出于真诚的安全信念而非功利性的护城河构建（moat-building）的观点同样属于解读 [@finbarrtimbers](https://x.com/finbarrtimbers/status/2064427031543341450)
- 诸如“GPT-4 时刻”、“大模型味儿（big model smell）”、“作为工程师被它完全压制”或“对普通用户来说似乎没有好多少”等主观报告属于经验性陈述，而非标准化证据 [@karinanguyen](https://x.com/karinanguyen/status/2064406015760601379), [@bcherny](https://x.com/bcherny/status/2064431111154053187), [@akbirkhan](https://x.com/akbirkhan/status/2064418425552928812), [@citrini](https://x.com/citrini/status/2064480613852201336)

## 不同观点


**支持派 / 能力优先**

- Anthropic 员工和深度测试人员将 Fable 5 描述为一次**阶跃式提升**：
  - Felix Rieseberg：从向 AI 下达任务转向赋予其职责 [@felixrieseberg](https://x.com/felixrieseberg/status/2064392202504310900)
  - Alex Albert：模型感觉更具协作性，而非仅仅是一个工具 [@alexalbert__](https://x.com/alexalbert__/status/2064394410004304003)
  - Karpathy：一个“值得大版本更新的阶跃式变化”，尤其是在处理长且困难的任务时，尽管安全防护措施在发布之初“触发过于频繁” [@karpathy](https://x.com/karpathy/status/2064409694761054332)
  - Bcherny：自 Opus 4.5 以来最大的跨越；该模型展现出了判断力、品味和有条理的 Debug 能力 [@bcherny](https://x.com/bcherny/status/2064431111154053187)
- 第三方基础设施和应用供应商强调了其在 Benchmark 中的胜出以及集成价值，而非安全争议 [@cursor_ai](https://x.com/cursor_ai/status/2064394824313376787), [@cognition](https://x.com/cognition/status/2064398549073453266), [@NotionHQ](https://x.com/NotionHQ/status/2064397568696819984), [@Azure](https://x.com/Azure/status/2064421301108834552)

**批判派 / 信任与开放**

- 许多研究人员和开源模型倡导者认为，即使是出于安全动机，这种静默限流（silent throttling）也是不可接受的：
  - Natolambert 称在不告知用户的情况下这样做是“失调（misaligned）”的 [@natolambert](https://x.com/natolambert/status/2064404993193754830)
  - Dean Ball 警告称，这可能会招致**反垄断**审查 [@deanwball](https://x.com/deanwball/status/2064434861088395730)
  - Jeremy Howard 称这是“非常黑暗且悲伤的一天” [@jeremyphoward](https://x.com/jeremyphoward/status/2064481719626154417)
  - Gneubig 警告未来 AI 可能仅提供给少数特权阶层 [@gneubig](https://x.com/gneubig/status/2064451352000975124)
  - Eric Zelikman 将其定义为在暗中破坏（sabotaging）客户 [@ericzelikman](https://x.com/ericzelikman/status/2064442174373314701)
- 开源支持者将此次发布作为倡导**主权/开放模型**的理由 [@nickfrosst](https://x.com/nickfrosst/status/2064396337404096809), [@NoahZiems](https://x.com/NoahZiems/status/2064464265189482570), [@ClementDelangue](https://x.com/ClementDelangue/status/2064513229099876663)

**中立 / 混合**

- 一些观察者认为，即使产品设计欠佳，Anthropic 可能也**真诚地相信**这些干预措施对安全是必要的 [@finbarrtimbers](https://x.com/finbarrtimbers/status/2064427031543341450)
- 另一些人表示，Anthropic **并不欠**任何人无限制的 Frontier 能力，但仍认为这纯粹是商业行为和市场细分，而非出于利他主义 [@suchenzang](https://x.com/suchenzang/status/2064452548753559644)
- Karpathy 的观点是矛盾的：模型质量异常出色，但发布的防护措施过于敏感，可能需要进行调优 [@karpathy](https://x.com/karpathy/status/2064409694761054332)

## 研究限制、隐私及企业影响


**讨论从安全领域扩展到了关于信任、隐私和企业可靠性的更广泛问题。**

- 企业关注的核心问题是**可预测性**：如果供应商可以根据推断的任务类别静默降低输出质量，用户可能再也无法得知失败是源于模型、Prompt 还是隐藏的干预 [@MattGibsonMusic](https://x.com/MattGibsonMusic/status/2064518301888512486), [@code_star](https://x.com/code_star/status/2064464447662707180)
- 一些用户担心，对于重要的工作流而言，这实际上构成了一种**供应链风险**，促使公司转向 Open Weights 模型或自研模型 [@NoahZiems](https://x.com/NoahZiems/status/2064464265189482570), [@deliprao](https://x.com/deliprao/status/2064485687374569897)
- 还有人担心账户级别的上下文或过往使用记录可能会影响干预行为的触发，正如生物学家报告的正规模式与无痕模式下的差异 [@cremieuxrecueil](https://x.com/cremieuxrecueil/status/2064449457869984035)
- 在提供的推文中，没有直接证据表明 Anthropic 正在**利用用户数据进行训练**或违反了明示的数据隐私条款；此处的隐私辩论主要集中在**行为画像（Behavioral Profiling） / 静默策略执行**，而非传统的训练数据隐私。
- 对于研究用户，隐藏干预被认为极具破坏性，因为它破坏了**可重复性和科学归因** [@deanwball](https://x.com/deanwball/status/2064434861088395730), [@MattGibsonMusic](https://x.com/MattGibsonMusic/status/2064518301888512486)
- 对于企业买家而言，问题不仅在于模型是否强大，还在于它是否是编程、医疗、科学、金融和基础设施领域中一个**稳定且可审计的依赖项**。

## 背景


**这次发布之所以重要，是因为它同时包含了显见的能力飞跃和显见的访问控制转变。**

- 此次发布正值与 GPT-5.5、即将推出的 GPT-5.6 以及 Gemini 3.5 Pro 的激烈竞争之中；多位博主认为 Anthropic 在 coding/agentic 工作方面已经取得了暂时领先 [@kimmonismus](https://x.com/kimmonismus/status/2064467466450088078), [@teortaxesTex](https://x.com/teortaxesTex/status/2064473970892587105)
- 这也涉及到一个关于**开源与闭源模型差距 (open vs closed model gap)** 的更广泛争论；一份引用了 Epoch 风格框架的报告指出，权重开放模型 (open-weight models) 平均滞后于闭源前沿模型约 **4 个月** [@dl_weekly](https://x.com/dl_weekly/status/2064422551762153946)
- 社区反应表明，这次发布可能不仅因为其“大模型气息 (big model smell)”和基准测试的飞跃而被铭记，还因为它使**选择性能力释放 (selective capability release)** 常态化：即向公众开放前沿模型，但带有**特定领域的隐藏限制**
- 这一政策路线可能会影响未来关于以下方面的辩论：
  - **安全性与开放性 (safety vs openness)**
  - **前沿研究工具的公平获取**
  - **反垄断与平台权力**
  - **企业对 API 提供商的信任**
  - **即使在原始能力落后的情况下，开源模型是否会成为敏感技术工作的默认选择**

**模型、基准测试与评估**

- 新的基准测试项目 **Agents’ Last Exam (ALE)** 启动，旨在测试与劳动力市场接轨的 Agent 性能；在涵盖 **55 个职业**、**1,500 多个任务**中，顶尖 Agent 在最难层级的得分仅为 **2.6%**，该项目由来自 **100 多个机构**的 **300 多位专家**共同参与 [@YiyouSun](https://x.com/YiyouSun/status/2064392466011394213), [@SnorkelAI](https://x.com/SnorkelAI/status/2064396025410760950), [@dawnsongtweets](https://x.com/dawnsongtweets/status/2064452279973863848)
- Cohere 发布了其首个开源编程模型 **North Mini Code**：**30B 总参数 / 3B 激活参数的 MoE** 架构，支持 **256K 上下文**，**64K 最大生成长度**，采用 Apache 2.0 协议，并针对 agentic 工作流进行了优化 [@cohere](https://x.com/cohere/status/2064378058329526556), [@JayAlammar](https://x.com/JayAlammar/status/2064385607455908254), [@vllm_project](https://x.com/vllm_project/status/2064416312605237434)
- Google 宣布推出 **Gemini 3.5 Flash Live Translate**，支持 **70 多种语言**的实时语音到语音翻译，已在 Gemini API、AI Studio、Google Translate 中可用，并将登陆 Meet [@OfficialLoganK](https://x.com/OfficialLoganK/status/2064369125447864674)
- 新基准测试 **iOSWorld** 评估了在 **26 个自定义 iOS 应用**和 **133 个任务**中的个人智能手机 Agent；最强的前沿模型即使在拥有特权访问权限的情况下，成功率也仅为 **52%** [@rsalakhu](https://x.com/rsalakhu/status/2064402156740907444)

**推理、训练与系统**

- **Latent Context Language Models (LCLMs)** 作为一种长上下文推理方法被引入，可将上下文压缩高达 **16 倍**，在延迟/准确度边界上优于 KV-cache 压缩 [@micahgoldblum](https://x.com/micahgoldblum/status/2064361011994337772), [@iamleonli](https://x.com/iamleonli/status/2064374393057300846)
- Microsoft Research 的 **Mirage** 将 3D 场景存储为 latent tokens，据报告其视频生成速度提高了 **10.57 倍**，内存消耗降低了 **55 倍** [@HuggingPapers](https://x.com/HuggingPapers/status/2064393076416688416)
- vLLM 推出了 **vime**，这是一个位于 vLLM 生态系统中的 RL 后训练框架，与 NeMo-RL、OpenRLHF 和 verl 并列 [@vllm_project](https://x.com/vllm_project/status/2064397637634376174)
- 关于 Agent 训练的讨论仍在继续，包括用于自我提升支架 (self-improving scaffolds) 的 **Self-Harness** [@omarsar0](https://x.com/omarsar0/status/2064429834999304247) 以及在多轮对话中保留推理轨迹的 **AutoForge/交错思考 (interleaved thinking)** [@cwolferesearch](https://x.com/cwolferesearch/status/2064505867181949182)
- Google 与 Hugging Face 启动了 **Fast Gemma Challenge**，旨在不破坏质量的前提下，在单张 **A10G** 上加速 **Gemma 4 E4B** [@googlegemma](https://x.com/googlegemma/status/2064374874962117084), [@osanseviero](https://x.com/osanseviero/status/2064375902046245219), [@_lewtun](https://x.com/_lewtun/status/2064386398090576236)

**Agent、工具与开发工作流**

- LangChain 强调了 Fleet 中由循环触发器驱动的 **agent loops** 模式 [@caspar_br](https://x.com/caspar_br/status/2064363014997021126)
- OpenAI 在 Responses API 的网络搜索中增加了**图片结果 (image results)** [@OpenAIDevs](https://x.com/OpenAIDevs/status/2064395155688616153)
- GitHub/Copilot 应用更新包括**并行子会话 (parallel sub-sessions)** 和用于动态界面的 **canvas** UI [@tgrall](https://x.com/tgrall/status/2064334802799509745), [@burkeholland](https://x.com/burkeholland/status/2064446521035067615)
- Hermes Desktop 增加了 **Ollama** 支持，具备自学 Python 技能以及即时通讯应用集成 [@ollama](https://x.com/ollama/status/2064441778590339402), [@NousResearch](https://x.com/NousResearch/status/2064468385748951415)
- 关于 Agent 执行的一个面向安全的对立观点：**Temenos** 主张对生成的代码而非 Agent 本身进行沙箱处理，使用 **rootless gVisor**，同时将身份验证/工具（auth/tools）保留在宿主机上 [@abhijithneil](https://x.com/abhijithneil/status/2064462294155952297)

**研究、科学与形式化方法**


- Axiom 发布了 **EconLib**，一个基于 Lean 的经济学库；对 Aumann 的“同意存在分歧 (agreeing to disagree)”定理进行形式化时，发现了一个隐藏的与可数性相关的假设 [@TheTuringPost](https://x.com/TheTuringPost/status/2064391882017579520)
- “Economy of Minds” 提出通过拍卖和激励机制而非中心化编排来进行 Agent 协作，据报告，数学推理能力从 **15.9% → 57.0%**，金融研究能力从 **45.0% → 60.0%** [@TheTuringPost](https://x.com/TheTuringPost/status/2064406931184443618)
- 梅奥医学中心 (Mayo Clinic) 的 **REDMOD** 据报道能在诊断前 **3 年** 通过 CT 扫描检测出胰腺癌，在诊断前中位时间 **475 天** 识别出 **73%** 的隐匿性癌症 [@TheRundownAI](https://x.com/TheRundownAI/status/2064416920191869191)

**开放生态系统与基础设施**


- Hugging Face 和 Arcee 宣布达成合作伙伴关系，所有 Arcee 模型/数据集（包括私有数据）将使用 HF 替代 AWS S3 [@ClementDelangue](https://x.com/ClementDelangue/status/2064323874049679643), [@MarkMcQuade](https://x.com/MarkMcQuade/status/2064385389801124218)
- Cohere 继续通过 “**Sovereign AI for all**” 推动主权/开放视角 [@cohere](https://x.com/cohere/status/2064414912768618898)
- Marks Saroufim 提出了 **Researcher Reciprocity License** 并将 GPU MODE 数据集迁移至该协议下，此举是针对前沿实验室受益于开放研究却反过来限制访问权这一现状的明确回应 [@marksaroufim](https://x.com/marksaroufim/status/2064428421774753943), [@marksaroufim](https://x.com/marksaroufim/status/2064442386374369597)


---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. 开源模型推理与聊天模板更新

  - **[小米刚刚声称在标准的 8-GPU 服务器上实现了 1T 模型的 1,000+ tps](https://www.reddit.com/r/LocalLLaMA/comments/1u0buhm/xiaomi_just_claimed_1000_tps_on_a_1t_model_using/)** (热度: 1027): **小米 MiMo** 声称 [`MiMo-V2.5-Pro-UltraSpeed`](https://mimo.xiaomi.com/blog/mimo-tilert-1000tps) 通过 TileRT 模型-系统协同设计（而非 Cerebras/Groq 式的专用硬件），在单台“标准” `8-GPU` 服务器上实现了 **`1T` 参数 MoE 的 `1000+ tokens/s` 解码速度**。报告的技术栈结合了 **针对 MoE 专家的 FP4/MXFP4 量化与 QAT**（同时保持非专家模块的高精度），以及 **DFlash 块级掩码投机采样解码**（代码任务接受长度为 `6.30`，数学/推理为 `5.56`，Agent 任务为 `4.29`），并使用持久化低延迟内核以减少启动/同步开销。评论中一个关键的待解技术疑问是，小米并未指明使用了哪种 `8-GPU` 服务器，这使得可复现性和性价比对比变得模糊。评论者讨论了“Token 寒冬”的经济学，认为瓶颈不在于模型需求，而在于价格过高或被囤积的西方 GPU 供应；与此同时，来自 **DeepSeek、小米和 MiniMax** 的中国压缩稀疏架构/MoE 工作正变得更具推理效率。其他人强调，小米的选择性 FP4 策略是最重要的细节，因为朴素的全模型 FP4 会降低推理、代码和逻辑能力。

    - 强调的一个关键技术细节是，小米采用了**选择性 FP4 量化**，而非统一应用 FP4：在 **MiMo-V2.5-Pro** 中只有 **MoE Experts** 被量化为 FP4，而非专家模块则保留原始精度，以避免推理、逻辑和代码生成能力的下降。评论指出小米使用 **FP4 QAT** 来减小模型尺寸并提高带宽利用率，同时保持能力接近原始模型。
    - 已发布的模型权重可在 Hugging Face 上获取，路径为 **XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash**: https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash。这很有意义，因为它允许对宣称的 8-GPU 服务器上 `1,000+ tps` 的吞吐量进行独立检查或基准测试。
    - 几位评论者对该声明背后的硬件和参数计算提出了质疑：*“8 GPU 服务器……具体是哪 8 个？”* 以及 *“1T-A1B？”* 技术上的担忧是，如果不了解具体的 GPU 型号、Interconnect、Serving Stack、Batch Size、上下文长度，以及这个 `1T` MoE 模型是否每 token 仅激活约 `1B` 参数，吞吐量数据就无法解释。

  - **[Gemma 4 聊天模板现在支持保留思考（preserve thinking）](https://www.reddit.com/r/LocalLLaMA/comments/1u084qi/gemma_4_chat_template_now_has_preserve_thinking/)** (热度: 482): **Google 的 Gemma 团队在官方 Gemma 4 聊天模板中增加了 `preserve_thinking` 支持**，这与一些用户此前已成功应用的第三方模板修改相匹配。这一更改旨在使 Gemma 4 聊天格式能更好地保留/利用模型的“思考（thinking）”轨迹，尽管帖子里没有提供基准测试数据或实现差异对比。评论者普遍欢迎官方的采纳，认为这验证了此前社区模板 Hack 的有效性。几位用户推测，需要发布更大规模的 **Gemma 4 `124B` MoE** 版本，才能充分利用更新后的模板来实现更强的 Agent 编码用例。

    - 评论者指出，**Gemma 4 的官方聊天模板似乎正在添加 `preserve_thinking`**，一些用户此前通过第三方/自定义模板修改启用了该行为并发现其非常有效。宣称的主要技术优势是改善了 **Agent 编码工作流** 的连续性，保留先前的推理/思考轨迹有助于多步骤工具使用和代码迭代。
    - 一位评论者提醒，该更改可能尚未上线：`preserve_thinking` 支持被描述为一个**尚未合并的开放 PR**，而模型文件据报已有 `21 天` 未更新。这建议用户在假设发布版本已具备新行为之前，应先核实实际模型仓库中的 Tokenizer/聊天模板文件。
    - 几条评论将模板更改视为对更大规模 **Gemma 4 `124B` MoE** 变体需求的增加，认为 `preserve_thinking` 在与更高容量的模型搭配用于编码 Agent 场景时会更有价值。讨论虽然具有推测性，但技术核心在于通过扩展模型规模/MoE 架构来更好地利用更新后的聊天模板行为。

## 较低技术含量的 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Fable 5/Mythos 5 发布与访问层级

- **[推出 Claude Fable 5](https://www.reddit.com/r/ClaudeCode/comments/1u1b207/introducing_claude_fable_5/)** (热度: 2698): **该[图片](https://i.redd.it/tb8akxef4a6h1.png)是该帖子声称发布的 **Claude Fable 5 / Claude Mythos 5** 的 Benchmark 对比表，显示该高亮模型在 Agentic coding、知识工作、空间推理、工具使用、法律、生物、网络安全和健康等 Benchmark 中，相较于 Claude Mythos Preview、Claude Opus 4.8、GPT 5.5 和 Gemini 3.1 Pro 处于领先或接近领先的地位。正文将 Fable 5 和 Mythos 5 描述为相同的底层“Mythos 级”模型，其中 **Fable 5 使用了安全回退（safety fallbacks）机制**：涉及网络安全、生物/化学以及蒸馏（distillation）相关的请求会被路由至 **Claude Opus 4.8**，据称受影响的会话不足 `5%`。** 评论大多是造势或怀疑，而非技术分析，包括“确认 AGI”之类的玩笑，以及询问“Fable 最近是不是变笨了”的抱怨。

    - 一位评论者注意到一个明显的访问/定价限制：**Claude Fable 5 仅在 `6 月 22 日` 之前免费**，据报道之后用户需要购买积分（credits）才能继续使用。这对于任何评估该模型的人都很重要，因为 Benchmark 或工作流测试可能需要在积分计费期开始前完成。
    - 一名用户报告了一个可能的启动/前端问题，询问 **Fable 是否生成了格式错误的 HTML**，并链接了一张渲染结果的截图：https://preview.redd.it/qaceea1fma6h1.jpeg?width=1440&format=pjpg&auto=webp&s=440eb5a30e7dfc186d610ed94be50fa50b962c9e 。该评论暗示在推广过程中可能存在实现或输出格式的 Bug，而非基于 Benchmark 的模型质量讨论。

  - **[Claude Fable 5 感觉与其说是一次模型发布，不如说是 AI 不平等的一个预兆](https://www.reddit.com/r/ClaudeAI/comments/1u1fsdi/claude_fable_5_feels_less_like_a_model_launch_and/)** (热度: 2387): **该帖子认为，**Anthropic 所谓的 Claude Fable 5 推广**代表了从统一的公共 Frontier-model 发布向**分层访问架构**的转变：公共付费用户获得带有安全路由的 Fable 5，这可能会将涉及 `cyber`、`bio`、`chemistry` 或 `distillation` 的请求降级到 Opus 4.8；而选定的合作伙伴据称可以获得 **Mythos 5**，它被描述为具有更少安全防护的相同底层模型。帖子还强调了定价/容量限制：据说 Fable 5 仅在 `6 月 22 日` 之前包含在付费计划中，随后可能转为使用积分（usage credits）模式，这意味着对于固定费率的消费者订阅来说，Frontier-agent 的推理成本仍然过于昂贵。** 评论分为两派：一派担心 AI 访问的不平等，另一派则认为为了防范高风险能力，采取限制性的安全政策是必要的。一位评论者认为这种结果是可预见的 Token 经济压力导致的，即向昂贵的企业级模型倾斜；而另一位评论者则辩称，尽管会给用户带来不便，但采取“宁稳勿乱（rather safe than sorry）”的做法是正确的。

    - 几位评论者将此次发布视为预期的经济转型：随着 Frontier-model 的能力和复杂性增加，**推理/Token 成本上升到顶尖模型将成为企业专属工具**，而非默认的消费级产品。一位评论者认为这将推动日常工作负载流向硬件上更便宜的本地推理，如 **Apple M-series** 芯片或 **RTX Spark-class** 加速器，而将 Frontier API 预留给高价值任务。
    - 一个关注定价的讨论帖声称，新模型的 API 经济模型使得消费者订阅与 Frontier 的使用在结构上不匹配：*“我们 `$200` 的月费订阅在新模型上可能只够 `3` 次 API 提示词。”* 其中隐含的技术观点是，即使是高端消费者计划，也可能只能通过严格的速率限制（rate limits）、模型路由或回退到更便宜的模型（如 **Opus 4.8**）来维持运行，一位评论者称 Opus 4.8 对 “`99%`” 的用户来说已经足够了。

  - **[Claude Fable (Mythos) 发布了！](https://www.reddit.com/r/singularity/comments/1u1at0h/claude_fable_mythos_is_out/)** (热度: 1456): **图像 ([PNG](https://i.redd.it/i88096c6fa6h1.png)) 似乎显示了一个 **Claude 风格的模型选择器/UI**，其中有一个标为 **“Fable 5 High”** 的新模型，与帖子标题声称的 *Claude Fable/Mythos 已发布* 相符。评论中的关键背景细节是，**Fable 5 仅在 `6 月 22 日` 之前临时包含**在 Pro、Max、Team 以及按席位计费的 Enterprise 计划中，且“无需额外费用”，并将于 `6 月 23 日` 从这些计划中移除；目前尚未提供 Benchmark、架构细节、API 规范或能力评估。** 评论者大多对限时可用性表现出 Token 预算焦虑和 FOMO（错失恐惧症），开玩笑说需要在访问权限消失前“烧掉 Token”。讨论主要非技术性，不包含实质性的模型性能分析。

- 用户强调 **Fable 5** 暂时仅在 `6月22日` 之前捆绑于 **Pro, Max, Team 以及基于席位的 Enterprise** 方案中，并将于 `6月23日` 移除，这暗示未来的可用性可能需要单独的层级或付费墙。
- 一个关于可用性的细节：**Fable 5** 似乎可以在 **命令行中的 Claude Code** 中运行，但至少有一位用户报告称其在 **Claude 桌面应用的 Claude Code 集成** 中不可见，这表明存在特定于客户端的发布或 UI/模型选择器的差异。


### 2. Anthropic 数据政策与安全治理

- **[Anthropic 今天更改了其隐私政策，其中有一个每个 Claude 用户都需要了解的特定条款](https://www.reddit.com/r/ClaudeAI/comments/1u0kq84/anthropic_changed_their_privacy_policy_today_and/)** (活跃度: 1475)：**该帖子声称 **Anthropic** 在 `2026-06-08` 发布了修订后的 [隐私政策](https://www.anthropic.com/legal/privacy)，并将于 `2026-07-08` 生效，将执法披露措辞从“通过法律程序的外部强制披露”更改为“基于 Anthropic 内部认为必要的 *“善意信念 (good faith belief)”*” 进行披露。作者认为，这会给涉及创意写作、角色扮演、小说中的威胁或心理健康宣泄的误报审核/分类器升级带来风险，并将其与 [OpenAI](https://openai.com/policies/) 和 [Mistral](https://mistral.ai/terms/) 政策中据称更狭窄的披露措辞进行了对比；一位评论者询问了实际的政策更新链接/来源。** 热门评论者对此反应消极，将这一变化视为重大的隐私倒退和“平台恶化 (enshittification)”的证据，其中一人表示这破坏了 Anthropic 的信任/道德品牌形象，并让他们考虑切换回 Codex。

    - 一位评论者认为，修改后的 Anthropic 条款对于欧洲用户可能 **不符合 GDPR 规定**，因为相关条款据称定义过于宽泛。他们建议先通过 Anthropic 的 **Data Protection Officer (DPO)** 进行申诉，如果未能解决，再向相关的 **Data Protection Commission/authority (DPC)** 提交申请。
    - 与 **OpenAI** 相关的一个法律风险对比也被提出，引用了一份报告称，大规模枪击案受害者的家属可能会在一项诉讼中索赔 **10 亿美元**，指控肇事者的 ChatGPT 使用情况虽然在内部被标记，但未向警方报告：[BIV 文章](https://www.biv.com/news/tumbler-ridge-families-likely-to-seek-us1-billion-in-lawsuit-against-openai-lawyer-12209582)。讨论的技术政策含义是，AI 提供商的内部安全监控是否在升级、报告和用户隐私方面产生了义务。

### 3. 前沿 AI 基础设施与大脑研究博弈

  - **[SpaceX 刚刚展示了其首个 AI 卫星设计](https://www.reddit.com/r/singularity/comments/1u0qc5r/spacex_has_just_revealed_its_first_ai_satellite/)** (热度: 1883): **该图片是一个所谓的 SpaceX “AI1 卫星”设计的演示版概念幻灯片，展示了一个大型轨道计算平台，配备了**可展开的液体散热器**、**集中式计算单元**、约 `70 m` 的翼展、`20 m` 的展开高度，以及一个 `150 kW` 的太阳能阵列，为峰值 `150 kW` / 平均 `120 kW` 的计算负载供电。技术重点在于天基 AI 计算能力和散热，图表突出了冗余泵送回路和用于提高散热器生存能力的微流星屏蔽：[图片](https://i.redd.it/tw874bgnn56h1.jpeg)。** 评论区表示高度怀疑，一位用户将计算能力比作“1 架 GB200 机架”，并认为这不具成本效益，特别是考虑到硬件故障可能导致整个卫星报废。另一位评论者认为这次展示可能是市场/IPO 炒作，而非实际的工程公告。

    - 一位评论者指出图中所示的计算负载大约相当于**一个机架的 NVIDIA GB200 级硬件**，认为考虑到发射/集成成本以及无法维修故障组件，这在轨道上很难实现成本效益：*“如果某个部件损坏，整个东西就废了。”*
    - 另一项技术批评集中在架构术语“**冗余泵送回路**”和“**集中式计算**”上，暗示液冷回路加上集中式加速器机架可能会为卫星 AI 平台带来艰巨的热管理和单点故障风险。

  - **[Jeff Bezos 正在资助一项寻找大脑“核心算法”的疯狂行动](https://www.reddit.com/r/singularity/comments/1u079tc/jeff_bezos_is_funding_a_wild_hunt_for_the_brains/)** (热度: 1381): **[WIRED 报道](https://www.wired.com/story/jeff-bezos-is-funding-a-wild-hunt-for-the-brains-core-algorithm/)称 **Jeff Bezos** 正在资助 **Flourish**，这是一家据称估值 **`$2.5B`** 且已筹集 **`$500M`** 的神经科学/AI 初创公司，该公司追求的假设是：通过对生物神经元的直接研究，可以揭示大脑的“核心算法”。其技术赌注在于，经验神经科学——而非仅仅扩展当今的深度学习——可能会产生受真实神经计算启发的新 AI 架构或学习规则。**

    - 一位评论者对这一前提提出了核心技术质疑：*“是什么让他认为大脑像算法一样运作？”* 这一观点挑战了认知是否可以还原为单一的计算程序，而非源于异构的生物、生化和网络级动态。




# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快就会发布新的 AINews。感谢阅读到这里，这是一段美好的历程。