---
companies:
- anthropic
date: '2026-03-24T05:44:39.731046Z'
description: '**Anthropic** 的闭源编程产品 **Claude Code** 遭遇了严重的源代码泄露，暴露了超过 **50 万行**的编排逻辑，包括自主模式和记忆系统，但并未涉及模型权重。此次泄露引发了快速的公开逆向工程，产生了大量分叉项目（最高获得
  **3.26 万个 Star 和 4.43 万次 Fork**），随后 Anthropic 发起了 **DMCA 下架**行动。


  针对编译泄露代码的用户，网上出现了可疑的 npm 软件包，构成了实时的安全威胁。此外，相关讨论中还提到了名为 **“mythos”** 的未发布模型引用；尽管发生了泄露，产品功能更新仍在持续。文中提到了“Anthropic
  关于此次泄露的官方声明”，但未给出具体细节。'
id: MjAyNS0x
models:
- claude-code
people: []
title: 今天没发生什么特别的事。
topics:
- model-architecture
- security
- reverse-engineering
- dmca
- software-development
- open-source
- code-leak
- agent-harness-design
---

**平静的一天。**

> 2026/3/23-2026/3/24 AI 新闻。我们检查了 12 个 subreddits、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有发现更多 Discord 动态。[AINews 网站](https://news.smol.ai/) 允许您搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以 [选择订阅/退订](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) 邮件发送频率！

---

# AI Twitter 回顾

**头条新闻：Claude Code 源码泄露 —— 架构发现、Anthropic 的回应以及竞争对手的反应**

## 事件回顾

Anthropic 的一款闭源编程产品 Claude Code 似乎通过发布的 source maps / 安装包内容暴露了大量源代码构件，这引发了快速的公开逆向工程、镜像备份和衍生移植。讨论迅速从“令人尴尬的泄露”转向了“这揭示了最尖端的 Agent 治理设计中的哪些信息？”多位观察者指出，此次泄露暴露的是 Orchestration 逻辑而非模型权重，包括自主模式、记忆系统、规划/审查流程以及特定于模型的控制逻辑。公开的 Fork 激增；一个帖子声称某 Fork 仓库在法律担忧导致其转向使用 Codex 进行 Python 转换之前，已获得了 **3.26 万颗星和 4.43 万次 Fork** ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2038996920845430815))。随后的评论估计泄露的代码量超过 **50 万行** ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039029676040220682))。根据多位发帖者的说法，Anthropic 随后通过 **DMCA 删除请求** 来遏制二次传播 ([dbreunig](https://x.com/dbreunig/status/2039007097376108979), [BlancheMinerva](https://x.com/BlancheMinerva/status/2039114452088295821))。数据集中最具体的官方信号是一篇广泛流传的帖子，提到了 **“Anthropic 关于泄露事件的官方声明”** ([theo](https://x.com/theo/status/2039074833334689987))，但声明正文并未包含在此处，因此从本资料库来看，仅能确认该声明的存在。此外，一名 Claude Code 团队成员在余波期间宣布了一项产品功能 —— 通过 `/web-setup` 简化本地/网页 GitHub 凭证设置 ([catwu](https://x.com/_catwu/status/2039027712288075812)) —— 这暗示了正常的产品运营仍在继续。此次泄露还制造了现实的安全隐患：攻击者迅速注册了可疑的 npm 软件包，如 **`color-diff-napi`** 和 **`modifiers-napi`**，旨在针对那些试图编译泄露代码的人员 ([Butanium_](https://x.com/Butanium_/status/2039079715823128964))。

## 事实 vs. 观点


**推文中较为属实的信息：**
- Claude Code 源码组件遭到公开访问，并作为泄露事件被广泛讨论 ([scaling01](https://x.com/scaling01/status/2038982287648293016), [Yuchenj_UW](https://x.com/Yuchenj_UW/status/2038996920845430815), [theo](https://x.com/theo/status/2039069225109819838))。
- 泄露的材料中**不**包括模型权重；至少有一份安全综述明确表示“他们没有泄露模型权重” ([saranormous](https://x.com/saranormous/status/2039172685666918672))。
- 人们从仓库中提取了功能名称和架构基调，包括 **Kairos**、**dream**、**teammem**、**buddy**、**ultrathink**、**ultraplan**、**ultrareview**，以及 GitHub 和 Slack 集成 ([scaling01](https://x.com/scaling01/status/2038982287648293016), [scaling01](https://x.com/scaling01/status/2039001738934468857))。
- 根据多位观察者的说法，Anthropic（或其代表）似乎已通过 DMCA 寻求下架镜像/Fork 副本 ([dbreunig](https://x.com/dbreunig/status/2039007097376108979), [BlancheMinerva](https://x.com/BlancheMinerva/status/2039114452088295821))。
- 可疑的包名抢注（package-name squatting）瞄准了那些试图根据泄露源码构建本地 Claude Code 的开发者 ([Butanium_](https://x.com/Butanium_/status/2039079715823128964))。
- 据报道，在泄露事件发生后，其他人已在内部实现了本地编译 ([theo](https://x.com/theo/status/2039079267905261831))。

**看似合理但应谨慎对待的说法：**
- Anthropic 通过发布 source maps 特地“泄露”了仓库：这一点被广泛暗示，但推文中没有引用权威的技术根因解释。
- 未发布的模型文档（包括对名为 **“mythos”** 的模型的引用）被曝光：这出现在一份综述 ([saranormous](https://x.com/saranormous/status/2039172685666918672)) 和诸如“Anthropic 的新模型 Capybara/Mythos 只是想成为人类” ([scaling01](https://x.com/scaling01/status/2039091546377576864)) 之类的推测性议论中，但数据集并未独立验证这些组件的真实性。
- 确切的仓库指标和代码行数（例如 **32.6k stars / 44.3k forks**，**500k+ 行**）是第三方测量数据，可能反映的是特定时间点的特定镜像/Fork，而非原始仓库状态。

**观点 / 解读：**
- 此次泄露令人尴尬，但在技术上“没有突破性进展” ([rasbt](https://x.com/rasbt/status/2039020306912755763))。
- 真正的护城河是 harness engineering，随着代码的外泄，Claude Code 与竞争对手之间的差距将更快缩小 ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039029676040220682))。
- Anthropic 不应激进地压制 Fork，因为开源社区无论如何都会构建自定义的 harness ([BlancheMinerva](https://x.com/BlancheMinerva/status/2039128635559318013))。
- 该事件“致命地证伪了”基于保密和控制的安全策略 ([pmarca](https://x.com/pmarca/status/2039042126294733295))。
- 如果泄露的代码可以简单地通过机器翻译成另一种语言，版权执法就会被削弱 ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2038996920845430815))。

## 泄露讨论中揭示的技术细节


最重要的技术启示是，观察者压倒性地关注 **harness**，而非底层的 Claude 模型。这符合同一组推文中的大趋势：“harness 至关重要” ([Vtrivedy10](https://x.com/Vtrivedy10/status/2038993396463796638))，以及后来的“除了原始模型能力外，编程工具真正的差距在于 harness” ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039029676040220682))。Sydney Runkle 关于 harness engineering 的推文串——涉及动态配置中间件（在每个步骤中切换模型/工具/提示词，包括工具注册表过滤）——虽然不是专门针对 Claude 的，但为读者推断 Claude Code 团队在内部构建的内容提供了强有力的背景信息 ([sydneyrunkle](https://x.com/sydneyrunkle/status/2039040565749096607))。

### 由读者发现的命名内部系统 / 主题

提取自泄露仓库特征的推文提到：

- **Kairos**：被描述为一种“始终开启的自主 Agent 模式” ([scaling01](https://x.com/scaling01/status/2038982287648293016))。
- **dream**：被描述为“夜间记忆巩固” ([scaling01](https://x.com/scaling01/status/2038982287648293016))。
- **teammem**：“共享项目记忆” ([scaling01](https://x.com/scaling01/status/2038982287648293016))。
- **buddy**：“带有模型的类拓麻歌子（Tamagotchi）宠物系统” ([scaling01](https://x.com/scaling01/status/2038982287648293016))；随后得到其他人的呼应，注意到“Claude Code 中潜伏着一个 AI 宠物！” ([dbreunig](https://x.com/dbreunig/status/2039017351061143780)) 以及“新的 Claude Code buddy 功能挺可爱的” ([eliebakouch](https://x.com/eliebakouch/status/2039176958416720104))。
- **自动技能提升** ([scaling01](https://x.com/scaling01/status/2038983513081356360))。
- **ultrathink**、**ultraplan**、**ultrareview** 以及“与 GitHub 和 Slack 的完整集成” ([scaling01](https://x.com/scaling01/status/2039001738934468857))。

即使某些命名带有宣传性质或搞怪色彩，但整体图景是一致的：Claude Code 似乎拥有一个分层的 Agent 运行时，包含：
1. 持久化/项目记忆，
2. 自主/后台运行，
3. 规划/评审阶段，
4. 自我改进或技能蒸馏循环，
5. 接入开发者工作流系统的协作钩子（Hooks）。

### 架构形态与代码构成

几位技术读者得出了相似的解读：

- **很大一部分价值在于艰苦积累的编排逻辑和诊断工具**，而非神奇的算法 ([dbreunig](https://x.com/dbreunig/status/2039206774558036466))。
- 代码包含**许多针对特定模型和上下文的条件判断**，用以平滑处理模型怪癖 ([dbreunig](https://x.com/dbreunig/status/2039206774558036466))。
- 还有**大量普通的 CLI 基础架构 / 样板代码**，这表明其专有优势不在于 Shell 应用本身，而在于反馈循环、Prompt、中间件、诊断和集成 ([dbreunig](https://x.com/dbreunig/status/2039206774558036466))。
- 很大一部分代码可能是**围绕规划、工具调用、评审、记忆、重试和遥测（Telemetry）的脚手架**，而非新颖的模型代码。

这种解读与数据集里更广泛的 Agent 工程讨论相吻合：
- LangChain 推崇将人机回圈（human-in-the-loop）中断作为标准的流状态（stream state），而非定制的工作流机制 ([LangChain_JS](https://x.com/LangChain_JS/status/2038985561348993107))。
- Vtrivedy 强调将 Evals（评估）作为指导 Agent 更新和架构优化的信号 ([Vtrivedy10](https://x.com/Vtrivedy10/status/2039029715533455860))。
- Koylan 总结了一个 Shopify/DSPy 架构：Agent 控制的检索、上下文隔离、模块化后的 MIPRO Prompt 优化，以及“更小的模型 + 更好的架构 > 更大的模型 + 较差的架构” ([koylanai](https://x.com/koylanai/status/2039027239304433767))。

其含义是：Claude Code 的泄露大体上证实了业界的猜想，即**生产级编程 Agent 是 Prompt、策略、中间件、记忆、评估和异常处理的集成体**。

### 封装与泄露机制线索

推文暗示泄露可能源于发布的源码产物：
- “闭源 > 发布 Sourcemaps > 源码瞬间泄露” ([mattrickard](https://x.com/mattrickard/status/2039054181361967487))。
- Theo 讨论了是否可以“在直播中打开代码目录”而不会受到版权警告，暗示广泛的本地检查已变得可行 ([theo](https://x.com/theo/status/2039069225109819838))。
- “内部已实现本地 Claude Code 构建”表明代码树的完整度足以进行编译或重建本地版本 ([theo](https://x.com/theo/status/2039079267905261831))。

这也产生了一个衍生安全风险：针对本地构建者的原生插件依赖进行包名抢占（package-name squatting） ([Butanium_](https://x.com/Butanium_/status/2039079715823128964))。这是一个经典的二阶泄露效应：一旦代码外逃，漏洞攻击面就会从“暴露了什么？”扩展到“由于社区恐慌性重新编译会触发哪些工具链行为？”

## Anthropic 的明显回应

在这组推文中，Anthropic 的回应大多是间接可见的。

### 1) 官方声明存在
Theo 发帖称，有 **“一份来自 Anthropic 关于此次泄露的官方声明”** ([theo](https://x.com/theo/status/2039074833334689987))。由于声明全文缺失，除了确认其存在之外，任何进一步的推断都属于猜测。

### 2) 通过 DMCA 进行法律封堵
多条帖子表示 Anthropic 正在针对重新分发泄露源码的代码库发送 **DMCA 移除通知**：
- “代码是自由的，但 Anthropic 正在通过 DMCA 请求关闭泄露的 Claude Code 源码库” ([dbreunig](https://x.com/dbreunig/status/2039007097376108979))。
- “针对 Claude Code 源代码的 DMCA 正在发出” ([BlancheMinerva](https://x.com/BlancheMinerva/status/2039114452088295821))。

这表明 Anthropic 将此事件视为专有代码的未经授权发布，而非开源契机。

### 3) 产品运营继续进行
在争议期间，一位 Claude Code 团队成员发布了一项常规的产品更新：`/web-setup` 用于在 Web 版 Claude 会话中重用本地 GitHub 凭据 ([catwu](https://x.com/_catwu/status/2039027712288075812))。虽然这证据不足，但与“控制泄露，继续交付”的策略一致。

### 4) 此处没有证据表明 Anthropic 接受了这次泄露
一些外部人士认为 Anthropic 应该“冷静对待”，因为代码已经到处都是了 ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039191313749524518))，但本数据集中的证据指向了相反的方向：**封堵与下架**，而非正式发布。

## 竞争对手与生态系统的反应


### OpenHands / 开源竞争对手
最清晰的竞争对手回应来自 OpenHands 的 Graham Neubig：
- “OpenHands 不会向那些想要使用我们 Agent 的用户发出任何 DMCA 移除通知，我们的 Agent 拥有 Claude Code 的大部分功能。我们的路线图中包含 Tamagotchi（拓麻歌子）” ([gneubig](https://x.com/gneubig/status/2039166255089799222))。
- 随后他发布了一个关于 Tamagotchi 功能的跟踪议题 ([gneubig](https://x.com/gneubig/status/2039168326912389208))。

这既是竞争定位，也是一种实质性的主张：开源 Agent 栈可以复制 Claude Code 的“大部分”功能，并带有一种对伙伴系统（buddy system）的俏皮认可。

### OpenAI / Codex 的对比
同一时间窗口内，关于所谓的“Codex 代码库泄露”也出现了混乱，随后被一名 OpenAI 员工纠正：
- 最初的病毒式声明：“OpenAI 的某人泄露了整个 Codex 代码库” ([reach_vb](https://x.com/reach_vb/status/2038971515572523502))。
- 纠正：“这个库从一开始就是开源的……我在 OpenAI 负责 Codex 工作” ([reach_vb](https://x.com/reach_vb/status/2039038251407732754))。

这是一个有用的背景信息，因为它强化了对比：
- **Codex 代码库的可见性是故意的。**
- **Claude Code 的可见性则不是。**

Yuchen 尖锐地描述了一个下游效应：Claude Code 的一个分叉库获得了巨大的采用量，然后“使用 Codex 将整个代码库从 TypeScript 转换为 Python” ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2038996920845430815))。这是一个带有主观色彩但很重要的竞争角度：开源或泄露的框架代码可以利用对手的编码 Agent 在不同语言生态系统中快速重构。

### Nous / Hermes / 持久化 Agent 竞争对手
Nous/Hermes 的帖子并不是对泄露事件的直接反应，但由于它们宣传类似的能力，因此成为了对比对象：
- 持久化记忆、自我改进、许多内置工具、多平台集成、MIT 许可 ([evanlong_me](https://x.com/evanlong_me/status/2039026061640601816))。
- 两分钟内从 OpenClaw 导入 ([AntoineRSX](https://x.com/AntoineRSX/status/2039017227270156395))。
- 基于 Cron 的漏洞扫描和 Agent 维护 ([Teknium](https://x.com/Teknium/status/2039022907020689898), [Teknium](https://x.com/Teknium/status/2039096442313396514))。
- 社区工具和入门指南 ([Teknium](https://x.com/Teknium/status/2039102514508058675), [aijoey](https://x.com/aijoey/status/2039108098174906514))。

这些都很重要，因为泄露内容的阅读者通常得出结论：Claude Code 的“独门秘籍”可以通过强大的开源 Agent 系统进行复现。

### 风险投资/开源意识形态的反应
Marc Andreessen 的广泛反应最具哲学意味：“‘AI 安全’可以基于秘密和控制的想法已被致命地证伪” ([pmarca](https://x.com/pmarca/status/2039042126294733295))。这显然是个人观点，但它捕捉到了一个派系的结论：专有应用层的保密性并不是一种持久的控制方案。

## 不同观点

### 观点 1：泄露在战略上很重要，因为它暴露了真正的护城河
这是工程师们的主流看法。
- “除了原始的模型能力，编程工具真正的差距在于 harness（工程支架）” ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039029676040220682))。
- “Harness 工程非常困难，而且绝非易事” ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039191313749524518))。
- “基于模型类型和特定上下文的条件判断非常多” ([dbreunig](https://x.com/dbreunig/status/2039206774558036466))。
- “harness 塑造了 [模型]，使其在我们关心的工作中表现出色且具有成本效益” ([Vtrivedy10](https://x.com/Vtrivedy10/status/2038993396463796638))。

这种观点认为，这次泄露减少了围绕商业化编程 Agent 最有价值部分的信息不对称。

### 观点 2：很有趣，但并非开创性的
- Rasbt：“除了泄露这件事令人尴尬外，内容很有趣但没有什么突破性” ([rasbt](https://x.com/rasbt/status/2039020306912755763))。
- Mbusigin：“如果是在六个月前会更有趣……现在 harness 已经满大街都是了” ([mbusigin](https://x.com/mbusigin/status/2039105055299686834))。

这一阵营认为，该领域已经趋同于许多类似的模式，因此这次泄露主要验证了已知的最佳实践。

### 观点 3：Anthropic 应该停止对抗并接受现实
- Blanche Minerva 认为，一旦社区已经在构建自定义 harness，强行下架（takedowns）几乎没什么作用 ([BlancheMinerva](https://x.com/BlancheMinerva/status/2039128635559318013))。
- Yuchen 表示团队表现得很“淡定”，尽管数据集中关于 DMCA 报告的证据显示情况并非完全如此 ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039191313749524518))。

这种观点认为，代码外泄后的法律手段杠杆作用很低。

### 观点 4：DMCA 是合理的，因为这仍然是私有代码
这种观点隐含在 Anthropic 明显的行动以及担忧版权打击（copyright strikes）的帖子中 ([theo](https://x.com/theo/status/2039069225109819838))。虽然在这里很少被明确争论，但逻辑很直白：意外发布并不意味着放弃版权。

### 观点 5：泄露证明了基于保密的安全性/控制已经破产
- Andreessen 的论点不仅限于 Anthropic ([pmarca](https://x.com/pmarca/status/2039042126294733295))。

这是意识形态层面的，比工程细节更广泛，但它也成为了讨论的一部分。

## 背景：为什么这很重要

### 1) 它揭示了编程 Agent 的性能究竟源自何处
这次泄露为许多从业者已经怀疑的转变提供了具体证据：**前沿编程 UX 越来越像是一个系统问题，而不仅仅是模型问题**。模型提供推理和生成，但生产级质量来自于：
- 动态工具选择、
- 记忆架构（memory architecture）、
- 评估/审查循环、
- 错误分类与重试、
- 特定模型的 Prompt 分支、
- 与 GitHub/Slack 等的集成、
- 以及持久的自主模式。

这与围绕 Agent 评估和改进的讨论相吻合：
- trace（追踪）作为改进循环的基础 ([LangChain](https://x.com/LangChain/status/2039028327030079565))、
- 在线评估和 trace 增强 ([Vtrivedy10](https://x.com/Vtrivedy10/status/2039186184161616245))、
- 生产环境中的 Agent 监控 ([LangChain](https://x.com/LangChain/status/2039014039892947062))。

### 2) 它压缩了竞争周期
如果 Claude Code 编码了大量的隐性产品知识，那么公开访问意味着竞争对手可以：
- 复制模式、
- 对标 harness 决策、
- 进行跨语言的设计移植、
- 识别弱点、
- 并更快地构建开源等效方案。

Yuchen 明确预测，“每个模型实验室和 AI 编程初创公司……都会研究它并快速弥补这一差距” ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039029676040220682))。

### 3) 它创造了一个新的安全教训
包名抢注（package-squatting）后续攻击的重要性几乎与泄露本身相当。一旦开发者急于编译泄露的内部软件，生态系统就会变得容易受到依赖混淆、拼写错误抢注（typo squats）、虚假原生模块和恶意安装脚本的攻击 ([Butanium_](https://x.com/Butanium_/status/2039079715823128964))。这符合本周 Saranormous 所总结的更广泛的供应链恐慌 ([saranormous](https://x.com/saranormous/status/2039172685666918672), [saranormous](https://x.com/saranormous/status/2039108234460721341))。

### 4) 它驳斥了对 “wrapper” 的简单化贬低
一个重要的潜台词是：这次泄露似乎让许多工程师确信 “wrapper” 层绝非易事。多位读者在阅读后表示，代码证明了 **wrapper/harness 工程非常困难** ([dbreunig](https://x.com/dbreunig/status/2039206774558036466), [Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039191313749524518))。这加强了这样一种观点：应用层的护城河建立在 orchestration、产品 UX 和 eval loops 之上，而不仅仅依赖于 foundation models。

## 总结


Claude Code 的泄露并没有暴露 Anthropic 的模型 weights，但它暴露了具有战略意义的东西：一款领先编程产品背后的大部分 **agent harness stack**。公开的发现指向了一个成熟的 orchestration 架构，具有持久化内存、自主/后台模式、planning-review 循环、技能改进以及深度的 workflow 集成。Anthropic 在此事件中可观察到的反应是遏制——官方承认并据传发出了 DMCA 通知——而竞争对手和开源项目则借此机会声称，这些功能中的许多现在都可以在开放系统中复现。社区得出的最强技术结论不是 Claude Code 包含什么魔法，而是 **高性能编程 Agent 依赖于大量积累的、模型特定的、操作复杂的系统工程**。因此，这次泄露的意义不在于丑闻，而在于它像是一份实地笔记，揭示了当前真正的工程杠杆点所在。

**Key tweets:** [@scaling01](https://x.com/scaling01/status/2038982287648293016), [@scaling01](https://x.com/scaling01/status/2039001738934468857), [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2038996920845430815), [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039029676040220682), [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2039191313749524518), [@dbreunig](https://x.com/dbreunig/status/2039007097376108979), [@dbreunig](https://x.com/dbreunig/status/2039206774558036466), [@theo](https://x.com/theo/status/2039074833334689987), [@theo](https://x.com/theo/status/2039079267905261831), [@Butanium_](https://x.com/Butanium_/status/2039079715823128964), [@gneubig](https://x.com/gneubig/status/2039166255089799222), [@pmarca](https://x.com/pmarca/status/2039042126294733295), [@rasbt](https://x.com/rasbt/status/2039020306912755763), [@BlancheMinerva](https://x.com/BlancheMinerva/status/2039128635559318013), [@mattrickard](https://x.com/mattrickard/status/2039054181361967487), [@saranormous](https://x.com/saranormous/status/2039172685666918672)

**Models, agents, and post-training**

- [@PrismML](https://x.com/PrismML/status/2039049400190939426) 推出了 **Bonsai 8B/4B/1.7B**，这是一个基于 **Apache 2.0** 协议的 **1-bit 权重**系列模型。宣称数据：8B 模型仅 **1.15 GB**，比全精度同类模型 **缩小 14 倍**、**快 8 倍**、**能效高 5 倍**；定位为“10 倍智能密度”。后续推文展示了 MLX/iPhone 路径以及向左偏移的大小与智能帕累托前沿 (Pareto frontier) ([PrismML](https://x.com/PrismML/status/2039049404209148007), [PrismML](https://x.com/PrismML/status/2039049405815529559), [adrgrondin](https://x.com/adrgrondin/status/2039066539022778613), [HessianFree](https://x.com/HessianFree/status/2039049800398655730))。  
- [@nisten](https://x.com/nisten/status/2039100896840134935) 提供了 Bonsai-8B GGUF 的有用独立拆解：**8,188,548,848 参数**、**399 个张量**、**1099.3MB** 总权重数据、**1.126 bits/weight**，需要支持 `Q1_0_g128` 的 Prism 版 llama.cpp 分支。  
- [@liquidai](https://x.com/liquidai/status/2039029358224871605) 发布了 **LFM2.5-350M**，这是一个小于 **500MB 的量化**模型，专注于受限环境下的 **工具使用和数据提取**。该模型引起关注的部分原因是，据报道一个 350M 的模型使用了 **28T tokens** ([abacaj](https://x.com/abacaj/status/2039158882111521190))。  
- [@hcompany_ai](https://x.com/hcompany_ai/status/2039021096649805937) 推出了 **Holo3** 计算机操作模型，声称在 **OSWorld-Verified** 上达到 **78.9%**，领先于 **GPT-5.4** 和 **Opus 4.6**，且成本仅为 **1/10**，权重已在 Hugging Face 上发布且 API 已上线。  
- [@outsource_](https://x.com/outsource_/status/2038999111039357302) 强调了一个在 Claude 4.6 Opus 轨迹上蒸馏的 **27B Qwen3.5 变体**，声称支持本地 **16GB VRAM** 部署，**96.91% 的 HumanEval 保留率**，**24% 的思维链 (chain-of-thought) 减少**，以及强大的 SWE-bench 表现。  
- [@ClementDelangue](https://x.com/ClementDelangue/status/2039121367656702102)、[@QGallouedec](https://x.com/QGallouedec/status/2039000031949165005) 和 [@lvwerra](https://x.com/lvwerra/status/2039003207985197107) 标志着 **TRL v1.0** 的发布，包含 **75 种以上的方法**，涵盖 SFT、DPO、GRPO、异步 RL；lvwerra 表示其现在 **每日下载量达 10 万次**。  
- [@tinkerapi](https://x.com/tinkerapi/status/2039049192451301761) 指出了一份训练说明，通过谨慎的 SFT→RL 选择，使 **20B 模型实现了 5 倍的分数提升**。  
- [@togethercompute](https://x.com/togethercompute/status/2039099845856903644) 发布了 **Aurora**，这是一个开源的基于 RL 的投机采样 (speculative decoding) 系统，声称比训练良好的静态投机器 **快 1.25 倍**，且 **从零开始的在线训练** 可以击败预训练的静态基准模型 ([详情](https://x.com/togethercompute/status/2039099852924367186), [代码](https://x.com/togethercompute/status/2039099854702669835))。  
- [@QinYi88814](https://x.com/QinYi88814/status/2038971910835560921) 标记了 **daVinci-LLM**，这是一个透明的预训练项目，包含开源权重、数据管道、训练过程和消融实验；核心宣称是：**3B 模型匹配 7B 性能**。

**Agent、测试框架 (harnesses)、评估与可观测性**

- [@dair_ai](https://x.com/dair_ai/status/2038968068706390117) 介绍了 **Natural-Language Agent Harnesses (NLAHs)** 和 **Intelligent Harness Runtime**，认为 Harness 逻辑本身应该是一个可编辑/可执行的工件，而不是零散的控制器代码。这是在技术上与 Claude Code 讨论最契合的论文之一。  
- [@Vtrivedy10](https://x.com/Vtrivedy10/status/2038993396463796638), [@Vtrivedy10](https://x.com/Vtrivedy10/status/2039029715533455860), 和 [@Vtrivedy10](https://x.com/Vtrivedy10/status/2039035899938267334) 论证了 Harness 的质量是由 Eval 质量、Traces（追踪）和基础设施循环驱动的，而不仅仅是更换模型。  
- [@sydneyrunkle](https://x.com/sydneyrunkle/status/2039040565749096607) 继续了一个有用的 Harness 工程系列，探讨用于每步适配工具/模型/Prompt 的**动态配置中间件 (dynamic config middleware)**。  
- [@LangChain_JS](https://x.com/LangChain_JS/status/2038985561348993107) 描述了一种实用的 Human-in-the-loop 模式，其中中断表现为普通的流状态；[@LangChain](https://x.com/LangChain/status/2039014039892947062) 推出了关于监控生产级 Agent 的课程；[@LangChain](https://x.com/LangChain/status/2039028327030079565) 将 Traces 视为改进循环的基础原语。  
- [@FranklinMatija](https://x.com/FranklinMatija/status/2039001719007330530) 介绍了 **AI Agent Traps**，这是一种针对与网页、电子邮件、API 和多 Agent 系统交互的自主 Agent 的六类对抗性分类法。  
- [@perplexity_ai](https://x.com/perplexity_ai/status/2039029140758864314) 成立了由 Ninghui Li 领导的 **Secure Intelligence Institute**，并发表了第一篇回应 NIST 关于保障自主 Agent 安全的论文（[论文链接](https://x.com/perplexity_ai/status/2039029152880480260)）。  
- [@cwolferesearch](https://x.com/cwolferesearch/status/2039009111711367557) 发布了一份包含 **30 多个 LLM Evals/Benchmarks** 的调研，强调了领域分类法、人工标注、Model-in-the-loop 策展、数据质量、真实性以及演进。这是这批文章中更有用的元评估 (meta-eval) 帖子之一。  
- [@GoogleResearch](https://x.com/GoogleResearch/status/2039014600927043926) 宣布了一个新框架，通过优化每个项目与人类评分者的比例，来提高主观 AI Benchmark 的可复现性。  
- [@koylanai](https://x.com/koylanai/status/2039027239304433767) 总结了一套 DSPy/Shopify 架构风格的经验教训：Agent 控制的检索、上下文隔离、模块化后的 Prompt 优化、冻结的 Eval 上下文，以及“更小的模型 + 更好的架构 > 更大的模型 + 更差的架构”。

**开源模型、多模态与系统**

- [@IBM / @mervenoyann](https://x.com/mervenoyann/status/2039015519135641997) 重点推介了 **Granite 4.0-3B-Vision**，该模型在同尺寸下对文档/表格/图表的支持能力非常强，可通过 transformers/vLLM 获取，并采用免费许可证。  
- [@LearnOpenCV](https://x.com/LearnOpenCV/status/2038972079370858750) 介绍了专注于精准视觉定位（visual grounding）的 **Molmo Point**；[@_akhaliq](https://x.com/_akhaliq/status/2038998550881714402) 推荐了用于任务感知推测采样（task-aware speculative sampling）的 **TAPS**；[@_akhaliq](https://x.com/_akhaliq/status/2039000804061847801)、[@_akhaliq](https://x.com/_akhaliq/status/2039006585188499744)、[@_akhaliq](https://x.com/_akhaliq/status/2039007111741366620)、[@_akhaliq](https://x.com/_akhaliq/status/2039011853460819999) 和 [@_akhaliq](https://x.com/_akhaliq/status/2039029830323253546) 发布了关于图像生成、Agent 文明基础设施、图像编辑、端侧图像生成/编辑以及双向运动生成的新论文。  
- [@dair_ai](https://x.com/dair_ai/status/2039072251199549573) 发布了 **GAAMA**，这是一种为 Agent 设计的图增强关联记忆（graph-augmented associative memory），在 **LoCoMo-10** 上获得了 **78.9% 的平均奖励**，性能优于经过调优的 RAG 基准。  
- [@quentinlldc](https://x.com/quentinlldc/status/2038986438088257558) 发布了 **LeWorldModel** 数据集和权重（checkpoints）。  
- [@ID_AA_Carmack](https://x.com/ID_AA_Carmack/status/2039046172799578122) 对 **LeWorldModel** 进行了详尽的技术评论，包括具体细节：**224x224 RGB**、未经修改的 **ViT-Tiny** 编码器、**192 维潜空间（latent）**、预测器采用 **ViT-S**，在 **dropout 0.1**、**batch 128 x 4 轨迹**、**300** 次动作滚动至视界 **H=5**、最多 **30 次 CEM 迭代**下表现更好，但在预测器尺寸增加时性能会出现下降。  
- [@SemiAnalysis_](https://x.com/SemiAnalysis_/status/2039102080959566038) 发布了一篇关于 Blackwell 的深度剖析，涵盖了 tensor cores、PTX/SASS、tcgen05、UMMA、TMA、floorsweeps、DSMEM 以及良率微基准测试（yield microbenchmarking）。  
- [@clattner_llvm](https://x.com/clattner_llvm/status/2039027422310596881) 认为 Kernel 作者需要调度器控制权，但无需完全的微观管理；后续笔记提到，简化竞态条件可以带来更具可移植性、可组合性的算法 ([thread](https://x.com/clattner_llvm/status/2039028017843126406))。  
- [@Prince_Canuma](https://x.com/Prince_Canuma/status/2039090156188389500) 指出 **RF-DETR** 现已支持 MLX，可用于端侧实时实例分割。  
- [@Shawkat_m1](https://x.com/Shawkat_m1/status/2039014724071719405) 报告称，将 Qwen3.5:36b 的 Ollama 切换到 MLX 后速度提升了 **2.2 倍**；[@joreilly](https://x.com/joreilly/status/2039002786130534618) 发现，在 M1 Max 上，使用 `qwen3.5:4b-nvfp4` 运行 Agent 的速度比 qwen3.5:4b **快 38%**。

**行业、融资与产品动态**

- [@OpenAI](https://x.com/OpenAI/status/2039085161971896807) 宣布了一项巨额融资：**承诺资本达 1220 亿美元**，**投后估值达 8520 亿美元**，旨在全球范围内分发有用的智能。多篇评论文章对此进行了深度解析 ([scaling01](https://x.com/scaling01/status/2039081471843930366), [TheRundownAI](https://x.com/TheRundownAI/status/2039103606327001435), [reach_vb](https://x.com/reach_vb/status/2039114329967013980))。  
- [@runwayml](https://x.com/runwayml/status/2038984561132990836) 推出了 **Runway Fund**，并表示已支持 Cartesia、LanceDB 和 Tamarind Bio。  
- [@charlieholtz](https://x.com/charlieholtz/status/2039027121901957349) 表示 Conductor 完成了 **2200 万美元的 A 轮融资**。  
- [@andreamichi](https://x.com/andreamichi/status/2039010131443437850) 表示 AI 安全公司 depthfirst 以 **5.8 亿美元的估值**完成了 **8000 万美元的 B 轮融资**。  
- [@wandb](https://x.com/wandb/status/2038984035301822784) 推广了对 ClickHouse CEO 的采访，内容涉及在产品面世前融资 **5000 万美元**以及为 AI Agent 构建产品。  
- [@yupp.ai](https://x.com/pankaj/status/2039010092255969712) 正在停止运营，网站将保留 **15 天**供用户导出数据。  
- [@Google](https://x.com/Google/status/2038969843701989773) 为美国用户推出了 Gmail 用户名修改功能：可修改为任何可用的 `@gmail.com` 用户名，旧地址保留为别名，**每年限修改一次，总计最多修改三次**；[@gmail](https://x.com/gmail/status/2039107985281008078) 为美国的 Google AI Ultra 订阅者推出了 **AI Inbox** 测试版。  
- [@OfficialLoganK](https://x.com/OfficialLoganK/status/2039015034286694618) 和 [@_philschmid](https://x.com/_philschmid/status/2039014102811427263) 在 Gemini API/AI Studio 中推出了 **Veo 3.1 Lite**，价格为 **$0.05/秒**，仅为 Fast 版的一半，支持生成 **4秒/6秒/8秒** 的 T2V/I2V 剪辑，比例支持 **16:9 / 9:16**。  
- [@GoogleAIStudio](https://x.com/GoogleAIStudio/status/2039055128276148454) 围绕 **Lyria 3** 推出了音乐实验室（music playground）。  
- [@osanseviero](https://x.com/osanseviero/status/2039120000095547722) 报告称 **Gemma** 下载量已达到 **4 亿次**，并拥有 **100,000 个变体模型**。  
- [@AnthropicAI](https://x.com/AnthropicAI/status/2039137425214353555) 宣布与澳大利亚政府签署关于 AI 安全研究的谅解备忘录（MOU）。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Claude Code 源码泄露与分析

  - **[Claude 源代码通过其 npm 注册表中的 map 文件泄露](https://www.reddit.com/r/LocalLLaMA/comments/1s8ijfb/claude_code_source_code_has_been_leaked_via_a_map/)** (热度: 4694): **图像显示了一个终端窗口的目录列表，其中包含与名为 "Claude" 的项目相关的文件，包括 TypeScript 文件和一个源码映射文件 (`cli.js.map`)。npm 注册表中存在此 map 文件表明源代码可能被无意中暴露，这可能是由于配置错误或疏忽造成的。这一事件凸显了在生产环境中保护 source maps 以防止未经授权访问源代码的重要性。** 评论者幽默地推测了这次疏忽，认为这可能是由于 Anthropic 员工的失误或其 AI 系统的一个特性。还有人讽刺地评论说，由于这次泄露，代码现在已经是“开源”的了。

    - Claude 源代码通过其 npm 注册表中的 map 文件泄露，引发了重大的安全担忧，尤其是考虑到 Claude 在识别漏洞方面的声誉。这一事件暴露了 Anthropic 内部安全措施的潜在漏洞，以及其 AI 在保护专有信息方面的有效性。
    - 讨论幽默地暗示 Anthropic 员工可能通过“氛围编程 (vibe coding)”无意中导致了泄露，这意味着他们的开发过程中缺乏严格的监督或自动化检查。这表明需要更强大的内部控制，甚至可能需要更先进的 AI 驱动监控系统来防止此类泄露。
    - 该事件引发了关于泄露的代码是否可以被视为“开源”的争论，因为它是无意中向公众开放的。这引出了关于使用或分析泄露代码的法律和道德影响的问题，以及是否可以利用它来改进安全实践或 AI 开发。

  - **[Claude Code 源码泄露 —— 我将其多 Agent 编排系统提取为一个开源框架，可与任何 LLM 配合使用](https://www.reddit.com/r/LocalLLaMA/comments/1s8xj2e/claude_codes_source_just_leaked_i_extracted_its/)** (热度: 600): **Claude Code 的源代码被泄露，揭示了其多 Agent 编排系统。一名开发者将该系统重新实现为一个名为 **open-multi-agent** 的开源框架，该框架与模型无关，并兼容 Claude 和 OpenAI 模型。该框架包含诸如用于任务分解的协调者模式（coordinator pattern）、带有用于 Agent 间通信的消息总线的团队系统，以及具有依赖解析功能的任务调度器等功能。它使用 `TypeScript` 实现，代码量约 `8000 行`，采用 **MIT** 许可证。该框架设计为进程内运行，与 `claude-agent-sdk` 不同，它可以部署在 Serverless、Docker 和 CI/CD 等各种环境中。该项目已在 [GitHub](https://github.com/JackChen-me/open-multi-agent) 上线。** 评论者对基于泄露的专有代码开源框架的合法性和道德性表示怀疑，并担心潜在的法律后果。此外，还有关于在规划和执行中使用不同模型的实用性争论，质疑选择 GPT-4o 等模型进行编码的决定。

    - 讨论强调了从 Claude Code 源码中提取的多 Agent 编排系统的技术层面。该系统旨在将目标分解为任务，这是跨不同语言模型管理复杂操作的关键功能。这个编排层对于集成各种 LLM（例如使用 Claude 进行规划并使用 GPT-4o 进行实现）至关重要，展示了一种协同利用不同模型优势的高级方法。
    - 围绕在 2026 年 3 月使用 GPT-4o 执行编码任务展开了技术辩论，表达了对其当时此类任务的适用性或性能的怀疑。这暗示了关于语言模型随时间演进及其能力的讨论，以及随着新模型的出现，某些模型在特定应用中可能变得过时或效果降低。
    - 讨论了开源专有代码的法律影响，特别是在 MIT 等开源许可下发布泄露代码的相关风险。这引发了对版权侵权的担忧以及潜在的法律保护需求，强调了在处理专有软件时了解知识产权法的重要性。

- **[分析 Claude Code 源代码。写下 "WTF"，Anthropic 就会知道。](https://www.reddit.com/r/LocalLLaMA/comments/1s8uerc/analyzing_claude_code_source_code_write_wtf_and/)** (热度: 601): **该 Reddit 帖子讨论了 **Claude Code** 的源代码，揭示了其广泛的追踪和分类机制。该系统使用简单的关键词检测来进行情感分析，追踪如 `wtf` 和 `frustrating` 等词汇以标记负面情绪。它还会监控用户在权限提示期间的行为，记录诸如打开反馈框或输入并取消输入等操作。其反馈系统旨在捕捉负面体验，提示用户分享会话转录。隐藏命令如 `ultrathink` 和 `ultraplan` 会改变系统行为，而遥测（telemetry）则记录详细的环境配置文件，包括会话 ID 和运行时的详细细节。一种内部模式 (`USER_TYPE=ant`) 会收集更细粒度的数据，将行为与特定的部署环境联系起来。这种程度的仪表化监控（instrumentation）表明该系统具有超越典型聊天机器人功能的极高可观测性。[来源](https://x.com/UsmanReads/status/2039036207431344140?s=20)。** 一些评论者认为，所述的追踪机制是事件驱动分析和用户反馈系统的标准做法，通常用于识别更新带来的问题。其他人则指出，像 `/btw` 这样的功能现在已经公开，而 `ultrathink` 等命令更像是内部产物或彩蛋（easter eggs），反映了充满趣味性的开发文化。

    - NandaVegg 强调，使用关键词列表进行情感分析（例如检测 'wtf' 或 'frustrating' 等词汇）是事件驱动分析系统中的常见做法。这些系统通常用于基于 Web 的应用程序，以监控用户反馈并识别可能破坏用户体验或模型行为的更新问题。这种方法通过将负面情绪标记为进一步调查的触发器，帮助开发人员快速解决潜在问题。
    - NandaVegg 还提到 Claude Code 中存在 'ultraplan' 和 'ultrathink' 等内部功能，这些功能尚未完全完善并作为彩蛋存在。这些功能被比作游戏应用中发现的内部产物，暗示了开发团队内部存在实验和侧向项目的文化。评论暗示此类功能可能是内部激励系统的一部分，鼓励开发人员进行创新并添加独特的功能。
    - 讨论还涉及了 SRavingmad 表现出兴趣的 'tamagotchi mode'（拓麻歌子模式）。虽然评论中没有详述，但该模式可能指的是 Claude Code 内部的一个功能或项目，模仿了拓麻歌子的互动和养成属性，可能是 AI 系统中一个趣味性或实验性的功能。


### 2. Qwen 模型发布与基准测试

  - **[Copaw-9B (Qwen3.5 9b, 阿里巴巴官方 Agentic 微调版) 已发布](https://www.reddit.com/r/LocalLLaMA/comments/1s8nikv/copaw9b_qwen35_9b_alibaba_official_agentic/)** (热度: 330): **该图片是一个柱状图，比较了三个 AI 模型在四项任务中的性能：**CoPaw-Flash-9B**、**Qwen3.5-Plus** 和 **GPT-5.4**。这四项任务分别是：文档解析（Document Parsing）、定时自动化（Scheduled Automation）、内存管理（Memory Management）和信息搜索（Information Search）。**CoPaw-Flash-9B** 是由 **Alibaba** 微调的模型，展现出了极具竞争力的性能，尤其在定时自动化和内存管理方面表现出色。该模型被指出在某些基准测试中与 **Qwen3.5-Plus** 持平，表明其在特定任务中的有效性。**CoPaw-Flash-9B** 的发布具有重要意义，因为它为大型模型提供了一个更小、更高效的替代方案，吸引了那些在特定应用中更倾向于紧凑型模型的用户。** 评论者赞赏像 CoPaw-Flash-9B 这样的小型模型的可用性，强调了对性能不打折扣的高效模型的需求。不同版本（如 Q8_0 GGUF 版本）的提供也受到了关注，表明社区对多样化模型格式的兴趣。

- Alibaba 发布的 CoPaw-9B 是 Qwen 3.5 的 Agentic 微调版本，由于其较小的模型尺寸而引发关注，这对寻求高效模型的用户很有吸引力。对比图展示了 Qwen 3.5 小型模型与同尺寸 CoPaw-Flash 之间的性能差异，暗示了其在效率或能力方面的潜在提升。
- 对于有兴趣使用 `llama.cpp` 运行该模型的用户，CoPaw-Flash-9B 的量化版本已经发布，这有利于在计算资源有限的环境中部署模型。该版本可在 Hugging Face 上找到，为实验和部署提供了更便捷的途径。
- 对于对 CoPaw-Flash-9B 的 Q8_0 GGUF 版本感兴趣的用户，文中提供了指向 Hugging Face 仓库的链接。该版本可能提供适用于特定用例的优化或配置，彰显了社区为提高这些模型的可访问性和通用性所做的努力。

- **[Qwen3.5-Omni 的结果已由 Alibaba 公布](https://www.reddit.com/r/LocalLLaMA/comments/1s8apue/qwen35omni_results_have_been_published_by_alibaba/)** (Activity: 499): **Alibaba 宣布发布 **Qwen3.5-Omni**，这是一款先进的全模态 AGI，能够处理文本、图像、音频和视频输入。公告重点介绍了一项名为 'Audio-Visual Vibe Coding' 的功能，这表明其专注于集成和解释多种数据类型以增强实时交互。图片包含一张性能对比表，但有人批评不同任务中的 benchmark 模型不断变化，一些人认为这具有误导性。** 一位评论者批评性能表中的 benchmark 模型不断变化具有误导性，而另一位则表达了对模型成功和进一步开发的希望。还有人希望该模型能与 llama.cpp 兼容，以实现更广泛的普及。

    - sittingmongoose 指出了 Qwen3.5-Omni 结果中可能存在的误导性方面，指出随着列表向下移动，用于对比的 benchmark 模型也在发生变化。这可能会歪曲对模型性能的看法，因为在整个结果中，它可能没有始终与同一组模型进行对比。
    - zdy132 提到 Alibaba 提供的 Qwen 3.6 plus preview API 现在可以在 [Openrouter](https://openrouter.ai/qwen/qwen3.6-plus-preview:free) 上免费使用。他们指出，虽然交互数据将用于训练，但该模型据推测性能很高，尽管存在数据使用问题，但对用户来说仍是一个极具吸引力的选择。

- **[发现 Qwen 3.6！](https://www.reddit.com/r/LocalLLaMA/comments/1s7zy3u/qwen_36_spotted/)** (Activity: 935): **图片展示了 “Qwen 3.6 Plus”，这是 Qwen 视觉语言系列中即将推出的模型，定于 2026 年 3 月 30 日发布。该模型以其庞大的 `context size of 1,000,000` 而闻名，这表明与之前的迭代相比，在处理海量数据输入方面有了重大飞跃。该模型还强调了对 prompt 和 completion 数据的收集以增强性能，表明其专注于迭代学习和适应。** 评论者推测了相比 3.5 版本的潜在改进，例如解决 overthinking 问题，并表达了对该模型通过进一步精炼达到 SOTA 状态的期待。

    - ForsookComparison 提到了 397B 模型达到 SOTA 性能的潜力，认为它可能只需要少量的精炼即可实现这一地位。这意味着该模型已经具备竞争力，但可以通过针对性改进来超越该领域的当前领先者。
    - ambient_temp_xeno 强调了令人印象深刻的 100 万 token 上下文窗口，这可以显著增强模型处理大规模数据和复杂任务的能力。这一特性对于需要大量上下文保留和处理的应用特别相关。
    - Long_comment_san 讨论了当前模型中 1.5 presence penalty 的问题，认为它对角色扮演（RP）场景产生了负面影响。他们表示比起 overthinking 的模型，更倾向于 instruct 模型，表明需要在创造力与指令遵循之间取得平衡。

### 3. 本地 LLM 实验与挑战

  - **[在 OpenCode 中将 Qwen3.5-27B 作为主模型在本地运行](https://www.reddit.com/r/LocalLLaMA/comments/1s7p0u9/running_qwen3527b_locally_as_the_primary_model_in/)** (热度: 365): **该帖子讨论了 Qwen3.5-27B 模型（一种混合架构的 LLM）作为 OpenCode 编程助手主模型的配置与性能。该模型在 NVIDIA RTX 4090 上通过 `llama.cpp` 本地运行，采用 4-bit 量化，上下文窗口大小为 `64K`，消耗约 `22GB` 显存（VRAM）。性能指标包括：Prefill（预填充）速度约为 `~2,400 tok/s`，Generation（生成）速度约为 `~40 tok/s`。该配置在编写和调试 Python 脚本等任务中展示了有效的工具调用能力，尽管作者指出 GPT-5.4 和 Opus/Sonnet 等模型在结构化程度较低的编程场景下表现更优。作者强调了合理的规划和提供充足的上下文对于获得最佳性能的重要性。详细的配置指南可见作者的 [博客文章](https://aayushgarg.dev/posts/2026-03-29-local-llm-opencode/)。** 评论者普遍认同 Qwen3.5 模型在本地配置中的有效性，并强调了良好的软件工程实践对于获得最佳结果的重要性。一位评论者建议尝试 **Qwen3.5-35b-a3b** 模型，据报道该模型在 Benchmark 分数相当的情况下，运行速度快了 `9x`。

    - v01dm4n 强调了 `qwen3.5-35b-a3b` 的性能，指出其 Benchmark 分数与 `qwen27b` 相似，但运行速度快了 9 倍。这表明新模型在效率上有显著提升，对于追求速度且不牺牲性能的用户来说是一个极具吸引力的选择。
    - dan-lash 讨论了 Frontier 模型与 `qwen 3.5` 在分别使用 OpenCode 和 Claude 作为框架（Harness）时的对比测试。Frontier 模型生成代码速度很快但不够全面，而 OpenCode 需要更多交互才能完成任务。相比之下，将 Claude 与 `qwen` 配合使用产生的代码量多出三倍且质量更好，强调了框架对模型性能的重要性。
    - rmhubbert 强调了在使用 LLM 时遵循良好软件工程原则（如研究、规划、测试和验证）的重要性。他们认为，这些实践对于从小型模型中获得最佳结果至关重要，即使是 Frontier 模型也无法弥补糟糕的工程实践。


## 非技术类 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Code 源代码泄露与分析

  - **[Claude Code 源代码通过其 npm 注册表中的 map 文件泄露](https://www.reddit.com/r/singularity/comments/1s8izpi/claude_code_source_code_has_been_leaked_via_a_map/)** (热度: 1522): **据 [GitHub](https://github.com/instructkr/claude-code) 报道，2026 年 3 月 31 日，Anthropic 的 Claude Code CLI 完整源代码通过其 npm 注册表中的 `.map` 文件泄露。该代码库包含约 `512k 行 TypeScript` 代码，采用 React + Ink 构建终端 UI，并运行在 Bun runtime 上。此次泄露可能曝光了尚未公开的主要受限（Gated）功能。** 评论反映出部分用户对此次泄露影响的误解，尤其是对 Large Language Models (LLMs) 和 Agent 之间的区别不甚了解，显示出社区中存在知识鸿沟。

    - Nedshent 指出了社区中对于此次泄露的一个普遍误解，强调许多人并没有掌握 Large Language Models (LLMs) 与 Agent 之间的区别。这凸显了这些技术如何运作及应用方面存在更广泛的认知差距，意味着就实际应用或复制 Claude 的能力而言，这次泄露的影响可能没有某些人想象的那么大。
    - Kizky 对泄露的实际影响提出了疑问，思考泄露的源代码是否可以用于训练模型或在线部署。这反映了人们对在现实应用中利用泄露代码的兴趣，尽管在缺乏关于泄露内容和结构的进一步背景下，这种做法的可行性或益处尚不明确。
    - 关于“使用 React + Ink (终端 UI) 在 Bun runtime 上构建”的评论提供了代码开发环境的技术细节。它提到了使用 React 和 Ink 构建终端 UI，运行在 Bun runtime 上，并指出代码库包含约 512,000 行 TypeScript。这让人得以窥见该项目的规模和复杂性，以及涉及到的技术栈。

- **[Claude Mythos 泄露：“迄今为止我们开发的最强大的 AI 模型”](https://www.reddit.com/r/singularity/comments/1s7zwjn/claude_mythos_leaked_by_far_the_most_powerful_ai/)** (热度: 1816): 据报道，**Anthropic** 开发了一款名为 **Claude Mythos** 的新 AI 模型，被描述为 *“迄今为止我们开发的最强大的 AI 模型”*。该模型以高昂的运营成本著称，使其价格明显高于其前代产品 **Opus**，且个人用户和小型企业可能无法负担。泄露信息表明，该模型的能力非常强大，但成本壁垒可能会限制其广泛采用。更多详情请参阅 [Claude Mythos - Archive](https://m1astra-mythos.pages.dev/)。评论者对 Claude Mythos 的高昂成本表示担忧，指出它可能超出许多用户的承受范围，类似于现有的 Opus 模型。这引发了关于尖端 AI 技术对小型实体可访问性的疑问。

    - **Sticking_to_Decaf** 强调 Anthropic 的新模型 Mythos 与前代 Opus 相比，运营成本大幅增加。这种增加的成本预计将使个人用户和小型企业无法使用，因为许多人已经认为 Opus 价格昂贵。这表明 Mythos 可能更多地针对预算约束较少的企业级应用。
    - **MFpisces23** 对围绕新 AI 模型发布的炒作表示怀疑，质疑增量改进的价值。他们强调希望看到真正的新能力，而不仅仅是改进的 benchmarks，这表明需要 AI 技术更实质性的进步，而不仅仅是微小的增强。

  - **[得益于泄露的 Claude Code 源代码，我使用 Codex 找到并修复了 Claude Code 中极其严重的 token 消耗根源。我的使用限制已恢复正常！](https://www.reddit.com/r/ClaudeAI/comments/1s8zxt4/thanks_to_the_leaked_source_code_for_claude_code/)** (热度: 1234): **该帖子讨论了通过利用 **Codex** 修复 **Claude Code** 中 token 消耗问题的方案。问题追溯到一个名为 `db8` 的函数，该函数由于未正确过滤会话文件附件，导致延迟工具（deferred tools）被重复重新宣告，并降低了缓存使用效率。该补丁涉及修改 `db8` 以保留某些附件，从而稳定缓存前缀，并将缓存效率从 `26%` 显著提高到 `99%`。此外，通过 `Node.js` 而不是独立二进制文件运行，解决了另一个与 API 请求中哨兵值（sentinel value）相关的 bug。修复详情记录在 [GitHub 仓库](https://github.com/Rangizingo/cc-cache-fix/tree/main) 中，包含一个简单的脚本，可以在不更改原版 Claude 安装的情况下应用补丁。** 一些评论者猜测 **Anthropic** 可能是故意泄露源代码以众包 bug 修复，而另一些人则对内部代码开发能力的缺乏表示失望。

    - **Macaulay_Codin** 强调了泄露的 Claude Code 中的一个重大技术问题，即恢复会话时的 `db8` 附件剥离。该 bug 的逻辑链是合理的，修复方案涉及一个简单的两行代码更改以保留 `deferred_tools_delta`。然而，他们警告说，该仓库还包含一个补丁，通过修改缓存 TTL 函数来强制执行 1 小时 TTL，从而绕过订阅检查，这并非正当的 bug 修复，而是对计费控制的规避。此外，帖子中声称的性能提升与实际数据不符，实际数据显示缓存率提升了 72% 而非所述的 99%。
    - **Dry_Try_6047** 讨论了使用 Claude 识别一个与 MCP 服务器中 OAuth2 相关的次要 bug，该 bug 此前已报告给 Anthropic 但几乎未获回应。尽管 Anthropic 声称拥有丰富的工程资源，用户仍能引导 Claude 找到并应用修复程序，随后他们在公司内部将其作为一种技能分享。这种情况引发了对 Anthropic 优先级和对客户报告问题响应速度的担忧，表明其工程能力与实际问题解决效率之间可能存在脱节。
    - 讨论涉及了 Anthropic 工程实践的更广泛影响，**Dry_Try_6047** 对公司的关注点和有效性表示担忧。尽管每位工程师拥有的 Agent 数量庞大，但似乎缺乏对基本问题的关注，社区需要独立识别并修复 bug 就是证明。这引发了关于如果这种趋势持续下去，软件工程未来的疑问，可能会对该学科关注核心问题解决能力的培养产生负面影响。

- **[我深挖了 Claude Code 泄露的源码，Anthropic 的代码库简直疯了](https://www.reddit.com/r/ClaudeAI/comments/1s8lkkm/i_dug_through_claude_codes_leaked_source_and/)** (热度: 5088): **Anthropic 的 Claude 泄露的源码揭示了一个奇特的功能：一个名为 `/buddy` 的基于终端的宠物系统，其中包括抽卡稀有度系统和 ASCII 伴侣。代码库还显示了一些非常规做法，例如对物种名称进行十六进制编码以绕过内部扫描器，以及使用 **Deepgram Nova 3** 的语音模式。该项目的代号为 **Tengu**，其 telemetry 事件和 feature flags 反映了这一点。代码库规模显著，`main.tsx` 大小为 `803,924 字节`，且有多个文件超过 `4,000 行`。代码中有 `460` 个 `eslint-disable` 注释，且仍在使用已废弃的函数。代码库包含幽默的注释以及尚未发布的功能，如 **Kairos** 和 **Ultraplan**。仓库链接在[这里](http://github.com/instructkr/claude-code)。** 一些评论者认为代码库的古怪之处令人感同身受，对于大型项目来说并不寻常，而另一些人则表示希望发布 `/buddy` 功能。

    - 一位用户指出，代码库中存在已废弃的函数很可能是一个战略决策，旨在提醒开发者不要在断新代码中使用它们。这是大型代码库中的常见做法，在涉及多个开发者且销售团队施压要求在过渡期间保持功能稳定的情况下，向新实现的逐步迁移是必要的。
    - 另一位评论者认为，代码库的状态对于大型项目（尤其是那些早于 GPT-3 等 AI 进展的项目）来说很典型。他们认为“疯狂（unhinged）”一词有些夸大其词，因为在许多开发者在紧迫的截止日期下贡献代码的环境中，这种复杂性和看似混乱的组织是标准现象。
    - 针对大型代码库的性质提供了一项技术见解，强调看似杂乱无章或过时（例如已废弃的函数）的内容，通常反映了随时间推移维护和演进软件的实际挑战。这包括在开发新功能与支持旧版本之间取得平衡，这在科技公司中是一个常见场景。

  - **[Claude Code 源码已通过其 npm 注册表中的 map 文件泄露](https://www.reddit.com/r/ClaudeAI/comments/1s8ifm6/claude_code_source_code_has_been_leaked_via_a_map/)** (热度: 2944): **图像显示了来自终端窗口的目录列表，展示了与名为 "Claude-code" 的项目相关的文件。`cli.js.map` 文件的存在表明包含了 source maps，这可能会在无意中暴露源代码。此次泄露是通过 npm 注册表中的一个 map 文件发生的，可能导致未经授权访问 **Anthropic** 的项目 Claude 的源代码。评论建议，这次泄露可能会导致大量 fork 或衍生作品的产生。** 评论者幽默地建议这次泄露可能会导致该项目出现许多 fork，其中一人指出可能会出现使用 Token 显著减少的 "MiniClaude" 版本。另一条评论强调了泄露的偶然性，暗示这仍然导致了该项目实际上变成了开源项目。

  - **[有人刚刚在 X 上泄露了 Claude Code 的源码](https://www.reddit.com/r/ClaudeCode/comments/1s8j10r/someone_just_leaked_claude_codes_source_code_on_x/)** (热度: 1831): **这篇 Reddit 帖子讨论了 Claude Code CLI 的 TypeScript 源码泄露事件，揭示了公共构建版本中不存在的 35 个构建时 feature flags。值得注意的功能包括 **BUDDY**（一个电子宠物风格的 AI 宠物）、**KAIROS**（一种持久型助手模式）以及 **ULTRAPLAN**（允许将复杂的规划发送到远程 Claude 实例）。泄露还发现了未公开的环境变量、内部命令以及针对 Anthropic 员工的特殊用户类型。图片是宣布泄露的社交媒体帖子截图，显示了源代码文件的目录列表。** 评论者幽默地推测 GitHub 上可能会涌入大量新项目，并表示有兴趣为泄露的代码贡献 bug 修复。

- Sensitive_Song4219 预见 GitHub 上会出现大量新项目，预测泄露的 Claude 代码将导致众多“coding agent harnesses”的诞生。这表明他相信社区将迅速适应并在泄露的源代码基础上进行开发，可能导致衍生作品和工具的大量涌现。
- HockeyDadNinja 幽默地建议，泄露事件可能允许社区提交 Bug 修复，暗示获取源代码可能使开发者能够更有效地识别和解决问题。这反映了一种常见的开源实践，即社区参与可以带来快速的改进和增强。
- Watchguyraffle1 强调需要将泄露的 Claude 代码与 GitHub 上的现有仓库区分开来。该评论强调了理解泄露代码相对于其他可用资源的独特之处的重要性，这对于希望有效利用新信息的开发者来说至关重要。

### 2. TurboQuant 与模型量化讨论

- **[[D] 对 Google 新论文争议的看法？](https://www.reddit.com/r/MachineLearning/comments/1s7m7rn/d_thoughts_on_the_controversy_about_googles_new/)** (热度: 382): **争议集中在 Google 的新论文 TurboQuant 上，据称该论文对 RaBitQ 之前的研究工作存在误导性陈述且引用不足。该论文因将 RaBitQ 的重要提及移至附录，并使用单核 CPU 运行 RaBitQ 对比使用 GPU 运行 TurboQuant 进行不公平的性能比较而受到批评，这可能夸大了 TurboQuant 的原创性和有效性。OpenReview 的公开评审指出，TurboQuant 在没有提供详细解释的情况下，将 RaBitQ 的保证描述为由于“分析松散”而导致的“次优”，引发了对该论文中比较和引用实践诚信的担忧。** 评论者对缺乏对独立研究团队的认可表示担忧，并担心大型研究实验室可能利用其优越的资源（如 GPU）来宣称突破，从而掩盖小型贡献者的光芒。

    - **Sad-Razzmatazz-5188** 强调了对 TurboQuant 论文对待 RaBitQ 工作方式的担忧，指出 TurboQuant 的作者可能通过将提及内容降级至附录并进行不均衡的性能比较，误导了 RaBitQ 的贡献。这可能会不公平地增强 TurboQuant 被感知的原创性和有效性，从而引发有关研究中适当引用归属的伦理问题。
    - **linearmodality** 批评 TurboQuant 论文并不像宣称的那样具有创新性，指出所使用的技术（如随机旋转和标量量化）在文献中已经为人所知多年。评论者认为该论文未能达到最佳结果，因为它没有采用 trellis coding（格点编码），而这种方法本可以提高性能。这一批评表明，考虑到 QTIP 等现有工作，该论文的新颖性及其对 AI 效率的贡献被夸大了。
    - **ProfessionalCraft275** 引用了 OpenReview 的一项批评，其中 TurboQuant 将 RaBitQ 的保证描述为由于“分析松散”而“次优”，但未提供详细解释。批评中缺乏清晰度，引发了对 TurboQuant 评估 RaBitQ 工作公平性和透明度的质疑。

- **[[D] TurboQuant 作者在 OpenReview 上的回复](https://www.reddit.com/r/MachineLearning/comments/1s8yni2/d_turboquant_author_replies_on_openreview/)** (热度: 121): **TurboQuant 作者在 OpenReview 上做出回应，澄清了论文的贡献，强调其新颖性在于推导用于最优量化的旋转矢量坐标的精确分布，而非源自 RaBitQ。他们承认了对 RaBitQ 最优性的错误定性，现在已准确地归功于其界限（bounds）。他们还表示运行时基准测试并非其发现的核心，而是专注于压缩与质量之间的权衡。该论文已在 arXiv 上更新以反映这些澄清。[OpenReview 链接](https://openreview.net/forum?id=tO3ASKZlok)。** 评论者批评 TurboQuant 作者展示了具有误导性的运行时基准测试，并在受到质疑后淡化其重要性。他们强调了透明度和尊重前人工作的重要性，并警告称，将问题视为无关紧要可能会侵蚀对学术研究的信任。

- 评论者批评 TurboQuant 论文展示了具有误导性的 runtime 基准测试，将 GPU 性能与单进程 CPU 性能进行对比，这可能会夸大所感知的加速效果。他们认为，虽然 GPU 兼容性确实有益，但作者处理批评和疏漏的方式对于维持对研究的信任至关重要。评论者强调，承认并纠正错误比将其斥为无关紧要更重要，特别是在像 Google 这样具有影响力的实验室中。
- 讨论突显了对 TurboQuant 在实际应用中影响的怀疑，特别是在节省 VRAM 方面。评论者指出，虽然 KV cache 量化可以降低成本，但它并不能显著降低大型模型的 VRAM 需求，例如在 5090 GPU 上加载 600M 模型。他们认为，可能受 Google 推动而产生的关于 TurboQuant 的炒作可能被夸大了，因为它并没有从根本上改变大型模型的硬件要求。

- **[TurboQuant 不仅仅适用于 KV：接近 Q4_0 质量的 Qwen3.5-27B，体积缩小约 10%，终于能塞进我的 16GB 5060 Ti 了](https://www.reddit.com/r/Qwen_AI/comments/1s8489c/turboquant_isnt_just_for_kv_qwen3527b_at_nearq4_0/)** (热度: 666): **图片展示了 TurboQuant TQ3_1S 模型在保持 Qwen3.5-27B 模型接近 Q4_0 质量的同时，体积足够紧凑，可以适配 16GB RTX 5060 Ti。TQ3_1S 模型比 Q4_0 缩小了约 10%，大小为 `12.9 GB`，而 Q4_0 为 `14.4 GB`，并且在困惑度 (PPL) 上的性能差距极小，TQ3_1S 为 `7.2570`，而 Q4_0 为 `7.2431`。这证明了 TurboQuant 量化技术（如 Walsh-Hadamard rotation 和 8-centroid quantization）在减小模型体积的同时保持性能的实际应用。** 评论者认为，虽然 TQ3_1S 模型是一个有趣的进展，但它缺乏与更先进的量化方法（如 dynamic quants）的对比，后者可能比过时的 Q4_0 标准提供更好的性能和压缩率。他们还指出，将足够的 KV cache 放入 VRAM 对于获得最佳性能至关重要。

    - No-Refrigerator-1672 强调了不仅要将模型权重，还要将足够的 KV cache 放入 VRAM 以获得最佳性能的重要性。他们认为，如果没有至少 16k 长的 KV cache，性能将被限制在 CPU offload 水平。他们还批评了使用 q4_0 量化的做法，建议像 imatrix 或 unsloth dynamic quants 这样更现代的技术能提供更好的性能和压缩。
    - PaceZealousideal6091 指出，与 q4_0 量化进行对比已经过时，因为该领域已经转向 q3 或 q2 等 dynamic quantization 方法，这些方法提供了更好的压缩和性能。他们承认该实验的学习价值，但强调需要采用更现代的量化技术进行有意义的比较。
    - Additional-Action566 分享了他们在 5090 GPU 上运行 q8 量化的 Qwen 27B 的经验，实现了 262k context size，且剩余 1GB VRAM。他们注意到在达到 170k context 后，吞吐量降至每秒 20 个 token，但仍然觉得性能令人印象深刻。他们提供了 Hugging Face 上的模型链接，并分享了运行该模型的命令行参数。

### 3. DeepSeek 模型更新与问题

- **[DeepSeek 当前状态](https://www.reddit.com/r/DeepSeek/comments/1s7rjw6/deepseek_current_status/)** (热度: 172): ****DeepSeek** 在 3 月 29 日至 30 日经历了 **11 小时的停机**，可能是由于服务器端的静默更新。更新后，该模型表现出具有“搜索 → 分析 → 完善”过程的**交替思维 (interleaved thinking)**，增强了其 Agent 行为。**知识截止日期**不一致，有些对话可以访问到 **2026 年 1 月**的信息，而其他对话则限制在 **2024 年 7 月**，这表明正在进行 A/B testing 或部分推送。编程能力有所提高，特别是在 **SVG 和多步骤脚本**方面，且**俄语**痕迹有所减少。**搜索功能**现在是迭代式的，能够自主完善查询，超越了单次 RAG。应用版本 **1.8.0(190)** 已于 3 月 27 日发布，可能是在为预计 4 月发布的 V4 做准备，而 LTM 和原生图像/视频生成等功能仍待上线。** 一些用户报告了更大的 context window，但幻觉也增多了，性能变差，导致用户不满。其他人对迭代搜索改进的说法表示怀疑，称没有观察到明显变化。一名用户注意到在停机前性能有所提高，但在停机后，模型的可靠性再次下降。

- 一位用户指出 DeepSeek 的 context window 有所增加，但随之而来的是“愚笨和幻觉”的显著上升，这表明模型在准确性和可靠性方面的表现有所下降。这凸显了 AI 模型中常见的权衡：扩展功能有时会导致意想不到的负面后果。
- 另一位用户对 DeepSeek 的迭代查询优化（iterative query refinement）功能表示沮丧，称尽管进行了尝试，但无法让其按预期工作。他们提到该系统本应遵循“search → analyze → refine”的过程，但在执行中似乎失败了，这表明模型的查询处理或用户界面可能存在问题。
- 一位用户报告了 DeepSeek 性能不稳定的情况，指出由于“响应过长”和输出内容荒谬，模型在一段时间内无法使用。他们观察到在一次停机前曾有短暂的好转，但随后性能再次下降。这表明系统后端或模型更新可能存在不稳定性，影响了其可靠性。


- **[Why is DeepSeek so much better at story telling?](https://www.reddit.com/r/DeepSeek/comments/1s892xq/why_is_deepseek_so_much_better_at_story_telling/)** (Activity: 135): **DeepSeek 在故事讲述方面表现出色，归功于它在中国庞大的网文生态系统数据集上进行了训练。该生态包含数百万部连载故事，具有清晰的叙事结构，如悬念（cliffhangers）和节奏环（pacing loops）。这为 LLM 提供了丰富的训练数据，其中可能包括“灰色地带来源”，如抓取的书籍和影子图书馆。这类似于 TikTok 利用强大的视频模式以及 Google 利用结构化知识来增强各自的 AI 能力。** 一位评论者认为，DeepSeek 的有效性可能源于它独立于美国的道德框架，这意味着在其故事讲述能力中蕴含着更广泛的文化视角。

    - **Electronic_Role_5981** 强调，中国庞大的网文生态系统拥有数百万部连载小说，为像 DeepSeek 这样的 LLM 提供了理想的训练数据。这些故事通常具有清晰的结构，如悬念和节奏环，这有利于提升故事讲述能力。此外，使用大规模数据集（可能包括抓取的书籍等“灰色地带来源”）也增强了 DeepSeek 的叙事实力。
    - **Heelerfan98** 和 **WillingnessSilver237** 提到在讲故事方面更倾向于使用 DeepSeek 和 Claude，这表明 DeepSeek 的方法相比其他模型更加轻松自然。这可能暗示了不同的训练方法或数据集侧重点，强调叙事流和创造力，而非严格遵循传统结构。
    - **huyreddit** 将 R1-0528 称为小说翻译的“神级”模型，表明 DeepSeek 在故事讲述方面的能力可能也延伸到了翻译任务中。这暗示该模型的架构或训练数据可能针对处理跨语言的复杂叙事结构进行了优化。

- **[INSANE UPDATE, v3.5?? does not feel like v4 yet](https://www.reddit.com/r/DeepSeek/comments/1s82vh5/insane_update_v35_does_not_feel_like_v4_yet/)** (Activity: 122): **最近的 DeepSeek 更新（被称为 v3.5）显著增强了其能力，特别是在处理速度和思维复杂度方面。用户报告称，该模型现在可以处理大量的研究任务，例如在仅 6 秒内分析 115 页，这表明 tool call 限制和处理效率大幅提升。这次更新似乎是 v4 全面发布的前奏，在演绎逻辑、编程和哲学讨论方面都有所改进。然而，一些用户在使用 web search 功能时遇到了问题，例如陷入循环或无法完成搜索，这些问题在更新前就已存在并持续至今。** 一些用户推测这次更新是为 v4 做准备，可能正在运行“3.2 或 4 lite”版本以测试新功能。另一些人指出，尽管有所改进，但 web search 功能的问题依然存在，如循环错误和搜索不完整。DeepSeek 的免费可用性也被强调为相对于 Gemini 和 CoPilot 等付费替代方案的显著优势。

- B89983ikei 强调了模型准确性的提升，特别是在演绎逻辑和编程任务方面。他们指出，即使面对新问题，模型现在也能“思考得更少但对得更多”，这表明其推理能力有所增强。然而，他们也提到了 DeepSeek 联网搜索功能的问题，该功能有时会陷入死循环或无法完成搜索，这表明更新中可能存在 Bug。
- PoauseOnThatHomie 讨论了相比 Gemini 和 CoPilot 的 Deep Search 等高级付费服务，使用 DeepSeek 的性价比。他们强调 DeepSeek 免费提供了类似的功能，对于希望在不产生额外费用的情况下避开使用限制的用户来说，这是一个更具吸引力的选择。
- lompocus 认为可能正在进行 A-B testing，因为他们在使用模型时遇到了不一致的结果，与其他报告性能提升的人相比，他们收到了“乱码”输出。这表明用户体验存在差异，可能是由于正在测试不同的版本或配置。

# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们将很快发布全新的 AINews。感谢阅读到这里，这是一段美好的旅程。