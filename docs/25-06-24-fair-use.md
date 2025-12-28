---
companies:
- anthropic
- replit
- delphi
- sequoia
- thinking-machines-lab
- disney
- universal
- midjourney
- google-deepmind
date: '2025-06-24T05:44:39.731046Z'
description: '以下是为您翻译的中文内容：


  *   **Anthropic** 赢得了一项重大的“合理使用”裁决，允许其使用受版权保护的书籍来训练 **Claude**。尽管外界对盗版数据存有疑虑，但这为
  AI 训练的合法性奠定了先例。

  *   **Replit** 实现了 **1 亿美元年度经常性收入 (ARR)** 的重大里程碑，展现了其强劲的增长势头。

  *   **Delphi** 完成了 **1600 万美元的 A 轮融资**，用于扩展其“数字思维”技术；与此同时，**Thinking Machines Lab**
  正专注于商业应用领域的强化学习。

  *   **迪士尼 (Disney)** 和 **环球影业 (Universal)** 起诉了 **Midjourney**，指控其未经授权使用受版权保护的图像。

  *   **Google DeepMind** 发布了 **Gemini Robotics On-Device**，这是一款专为机器人技术设计的紧凑型基础模型。'
id: MjAyNS0w
models:
- claude
- gemini-robotics-on-device
people:
- andrea_bartz
- giffmana
- andrewcurran_
- amasad
- swyx
- hwchase17
- krandiash
- daraladje
- steph_palazzolo
- corbtt
- demishassabis
title: Bartz 诉 Anthropic PBC —— “训练用途属于合理使用”
topics:
- fair-use
- copyright
- reinforcement-learning
- foundation-models
- robotics
- funding
- lawsuit
- digital-minds
- model-release
---

**一个重要的裁决，但并非最终定论。**

> 2025年6月23日至6月24日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（220 个频道，3440 条消息）。预计节省阅读时间（以 200wpm 计算）：365 分钟。我们的新网站现已上线，支持全文元数据搜索，并以精美的 vibe coded 风格展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

去年 8 月，由 Andrea Bartz 领导的一群作者对 Anthropic PBC [提起了集体诉讼](https://entertainmentlawreview.lls.edu/authors-v-anthropic-the-legal-showdown-over-ai-copyright-and-fair-use/)，指控其“非法下载”他们的作品来训练 Claude。这种破坏性书籍扫描的规模（每本书可能 [<$2](https://twitter.com/giffmana/status/1937591844252385323)，尤其是二手书）令人印象深刻：


![](https://resend-attachments.s3.amazonaws.com/S9Up265zT1VJ0VA)


对于了解 [Authors Guild v Google](https://en.wikipedia.org/wiki/Authors_Guild,_Inc._v._Google,_Inc.)（即 Google Books 诉讼案）的人来说，这当然很熟悉，两者的背景非常相似，但这是关于在受版权保护的内容上进行 pretraining 合法性的首次直接裁决。

该案件的案卷在[此处](https://www.courtlistener.com/docket/69058235/bartz-v-anthropic-pbc/)，但今天的结论来自[简易判决动议 (Motion for Summary Judgment)](https://www.courtlistener.com/docket/69058235/231/bartz-v-anthropic-pbc/)，Anthropic [可以说](https://x.com/mjbommar/status/1937562175955980614)“赢了”，因为法院明确裁定“训练使用属于 fair use”。


![](https://resend-attachments.s3.amazonaws.com/y4sXpTlxUVp0VvI)


看来 [Books3 的幽灵](https://www.wired.com/story/battle-over-books3/) 仍在困扰着 Anthropic，因为使用盗版书籍是一个独立的问题，但此处的判决非常明确，并可能为未来几年树立重要的先例：判决书中不少于 32 次提到 pretraining 这一用例是多么具有“转化性 (transformative)”，无论 LLM 记住了多少内容：


![](https://resend-attachments.s3.amazonaws.com/RcxQDzqlkw3F10a)


---

# AI Twitter 综述

**公司、融资与法律**

- **Anthropic 在书籍训练数据的 Fair Use 裁决中获胜**：一名联邦法官裁定 **Anthropic** 使用书籍训练 **Claude** 构成了 **fair use**，这对 AI 行业来说是一个重大决定。由 [@AndrewCurran_ 分享](https://twitter.com/ClementDelangue/status/1937519434312147374)的这一裁决将训练行为与获取数据的方法区分开来。讨论强调，[获取书籍的方法（盗版）是一个独立的问题](https://twitter.com/giffmana/status/1937551619937436101)，正如 [@giffmana 指出的](https://twitter.com/giffmana/status/1937591844252385323)，如果书籍在过程中被损毁，数字化书籍的成本低得惊人。
- **Replit ARR 达到 1 亿美元**：[@amasad](https://twitter.com/Hacubu/status/1937263659581079581) 宣布 **Replit** 的 **ARR** 已突破 **1 亿美元**，较 2024 年底的 **1000 万美元**大幅增长。[@swyx](https://twitter.com/swyx/status/1937300296386117661) 将这一快速增长描述为类似于“superintelligence”爆发的曲线，而 [@hwchase17](https://twitter.com/Hacubu/status/1937275206789427625) 等人则称赞了团队“出色的执行力”。
- **Delphi 筹集 1600 万美元 A 轮融资以扩展人类专业知识**：**Delphi** 是一个创建“数字大脑 (digital minds)”以扩展专业知识的平台，[宣布了由 **Sequoia** 领投的 **1600 万美元 A 轮融资**](https://twitter.com/krandiash/status/1937574873154617751)。其目标是[让全人类的知识变得可获取和可发现](https://twitter.com/daraladje/status/1937645599823921504)，目前已创建了超过 **2,000 个数字大脑**。
- **Thinking Machines Lab 的使命被描述为“面向企业的 RL”**：Mira Murati 的新 AI 初创公司 **Thinking Machines Lab** 被投资者描述为专注于“**面向企业的 RL**”，[据 @steph_palazzolo 分享的 The Information 报告](https://twitter.com/steph_palazzolo/status/1937284120062706004)。正如 [@corbtt 所指出的](https://twitter.com/corbtt/status/1937624653662744840)，这符合“火热的 RL 之夏”趋势。
- **迪士尼和环球影业起诉 Midjourney 侵犯版权**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1937314755066171580) 报道称，**迪士尼 (Disney)** 和 **环球影业 (Universal)** 已对 **Midjourney** 提起诉讼，指控这家图像生成公司在未经许可的情况下使用其受版权保护的内容训练模型。该诉讼称系统生成了诸如**蜘蛛侠 (Spider-Man)** 和**辛普森一家 (The Simpsons)** 等角色的未经授权图像。

**模型与技术发布及更新**

- **Google 发布 Gemini Robotics On-Device**：**Google DeepMind** 推出了 **Gemini Robotics On-Device**，这是一款足够小巧、可以直接在机器人上运行的基础模型。[@demishassabis](https://twitter.com/demishassabis/status/1937526283161809056) 强调了其在低连接环境下的速度和性能。该发布包含了一个端侧 **VLA** 以及开源工具和模型，以促进开发。
- **PrimeIntellect 发布 SYNTHETIC-2 数据集**：**PrimeIntellect** [宣布推出 **SYNTHETIC-2**](https://twitter.com/ClementDelangue/status/1937511681850044894)，这是他们的下一代开放推理数据集，也是一次“行星级规模的合成数据生成运行”，由 **9 个不同的模型**驱动。
- **Deepseek 使用 Nous Research 的 YaRN 进行上下文扩展**：[@Teknium1](https://twitter.com/Teknium1/status/1937373884610936854) 指出，**Deepseek** 利用了由 **Nous Research** 开发的 **YaRN** 方法来扩展其上下文长度。
- **可灵 AI（Kling AI）增强视频生成功能**：**Kling AI** 推出了多项新功能，包括支持[将作品保存为 **Live Photos** 以用作动态壁纸](https://twitter.com/Kling_ai/status/1937343208515924465)，以及用于创意视频编辑的新“**SurfSurf Effect**”，[并伴随有用户竞赛](https://twitter.com/Kling_ai/status/1937393240225063042)。
- **Hugging Face 发布用于视频嵌入的 VideoPrism**：[@osanseviero](https://twitter.com/osanseviero/status/1937560015348597124) 宣布发布 **VideoPrism**，这是一个用于生成视频嵌入（video embeddings）的新模型，适用于分类、视频检索和定位等任务。该模型、论文和代码均已在 **Hugging Face** 上发布。
- **PufferLib 3.0 发布，支持大规模 RL**：**PufferLib 3.0** 已发布，支持在海量数据集上进行强化学习（**RL**）训练。该团队展示了在单台服务器上利用 **1 Petabyte（12,000 年时长）的数据**训练 **Agent**，[正如 @slashML 所分享的那样](https://twitter.com/slashML/status/1937480613029904640)。
- **Warp 2.0 作为“Agentic Development Environment”发布**：**Warp** 推出了 2.0 版本，定位为 **Agentic Development Environment**。它在 [**Terminal-Bench** 上排名第一，在 **SWE-bench** 上达到了 **71%**](https://twitter.com/_akhaliq/status/1937542375179448828)。
- **LlamaBarn 预览**：[@jeremyphoward](https://twitter.com/jeremyphoward/status/1937290259307573649) 分享了来自 Georgi Gerganov 的 **LlamaBarn** 预览，幽默的是，[其中并没有包含任何 Llama 模型](https://twitter.com/teortaxesTex/status/1937628449708933220)。
- **Jina AI 发布 v4 Embeddings**：新版本的 **Jina embeddings** 已发布，代表了一次重大升级。该模型从 **RoBERTa** 扩展到了 **Qwen 2.5**，具有多模态能力，并支持 **COLBERT-style** 多向量表示，[正如 @nrehiew_ 所强调的](https://twitter.com/nrehiew_/status/1937357675072778567)。

**新技术与研究**

- **Mustafa Suleyman 提出 "Chain of Debate"**：**Inflection AI** CEO [@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1937553061427445824) 概述了从 Chain of Thought 演进的下一步：“**Chain of Debate**”。这一概念涉及多个模型“大声讨论、辩论、调试、审议”，从单一 AI 转向多个 AI 协作。
- **Sakana AI 推出 Reinforcement-Learned Teachers (RLTs)**：**Sakana AI** [@SakanaAILabs](https://twitter.com/AndrewLampinen/status/1937261400419885414) 发布了 **Reinforcement-Learned Teachers (RLTs)**，这是一种利用强化学习改变 LLM 推理教学方式的新方法。
- **RL 在 Agentic RAG 中展现出极高的样本效率**：[@corbtt](https://twitter.com/corbtt/status/1937594932040204483) 分享了令人兴奋的实验结果，展示了 RL 极高的样本效率。通过使用 **GRPO** 训练修改后的 **ART-E**（Agentic RAG 任务），他们发现 **qwen2.5-14b** 仅需 **1 个训练场景** 即可超过 **Gemini 2.5 Flash** 的表现，且仅需 **16 个场景** 即可超越 **o3**。
- **NetHack Learning Environment 发布五年后仍未被攻克**：在发布五周年之际，**NetHack Learning Environment** 仍然是 AI 面临的一大挑战。[@_rockt](https://twitter.com/_rockt/status/1937480864243331396) 指出，目前的尖端模型仅能达到 **~1.7% 的进度**，凸显了其难度。
- **LLM 可以通过 Backprop 进行编程**：一篇关于“**Programming by Backprop**”的新论文（由 [@_rockt](https://twitter.com/_rockt/status/1937507616000749888) 和 [@LauraRuis](https://twitter.com/_rockt/status/1937549094073041136) 分享）证明，LLM 可以通过仅在源代码上进行训练，而无需查看任何 I/O 示例，就能学会评估不同输入下的程序，充当“模糊程序解释器”。
- **斯坦福 CS336 "Language Models from Scratch" 课程资料发布**：[@percyliang](https://twitter.com/lupantech/status/1937524295732986046) 宣布由 **Tatsu Hashimoto** 等人授课的 **Stanford CS336** 课程圆满结束，并公开了所有讲义、代码和材料。

**框架、工具与基础设施**

- **OpenAI 工程团队因 ChatGPT 的可扩展性受到赞扬**：[@sama](https://twitter.com/sama/status/1937514123912491317) 赞扬了 **OpenAI** 工程和计算团队在快速扩展以满足 **ChatGPT** 巨大客户需求方面的“出色工作”，指出他们以“如此优雅的姿态处理了长达 2.5 年的冲刺”。
- **Perplexity Finance 对标 Bloomberg Terminal**：**Perplexity** CEO [@AravSrinivas](https://twitter.com/AravSrinivas/status/1937330521920737727) 发布了一项对比，展示了 **Perplexity Finance** 有效分析了 MAG 7 股票的增长，并暗示“**AI 正在吞噬像 Bloomberg Terminal 这样的传统软件**”。
- **LlamaIndex 的高级文档解析**：[@jerryjliu0](https://twitter.com/jerryjliu0/status/1937302778122314202) 展示了 **LlamaIndex** 的文档解析 Agent，它准确地将股权研究报告中复杂的组合图表转换为整洁的表格，而 **Claude Sonnet 4.0** 在此任务中“幻觉了一半的数值”。
- **"Context Engineering" 成为新趋势**：“**Context Engineering**”（上下文工程）的概念正受到关注。[@hwchase17](https://twitter.com/hwchase17/status/1937648042985030145) 强调 **LangGraph** 是实现这一目标的绝佳工具，并提出了简化上下文管理的新功能。
- **通过 Cline 在命令行管理 Azure**：**Cline** 宣布了一个新的 **Azure MCP Server**，允许用户直接从 CLI 使用自然语言[控制 Storage、Cosmos DB 和 Monitor 等服务](https://twitter.com/cline/status/1937324870393901539)。
- **uv 的性能与 Python 速度**：基于 Rust 的 Python 包安装程序 **uv** 的速度令开发者感到惊讶。[@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1937506437762286058) 指出，他们曾以为 pip 的缓慢是由于网络问题，而非可以优化的“纯 Python”速度限制。

**更广泛的影响与社区讨论**

- **现代技术的短暂寿命**：[@alexalbert__](https://twitter.com/alexalbert__/status/1937526135442874651) 反思了那些看似永恒的技术其实寿命惊人地短，例如 **Googling (~26 年)**、**手动编码 (~75 年)** 和 **手动驾驶 (~120 年)**，并暗示由于 AI 的出现，它们的“有效期”已近在咫尺。
- **Waymo 的快速扩张预测**：[@fchollet](https://twitter.com/fchollet/status/1937498488352264666) 预测 **Waymo** 的自动驾驶车辆服务将从目前覆盖美国人口的 **2-3%** 扩展到 **一年内的 15%**，并在 **三年内超过 50%**。
- **AI 与认知能力的辩论需要哲学基础**：[@random_walker](https://twitter.com/random_walker/status/1937483620630794382) 认为，关于 AI 对认知能力影响的生产性辩论需要熟悉 **Extended Mind Thesis**（延展心灵论），并理解为什么今天的担忧与 **2400 年前柏拉图 (Plato)** 对文字会侵蚀记忆的担忧有所不同。
- **AI 研究的本质**：[@_jasonwei](https://twitter.com/_jasonwei/status/1937590298022150638) 分享了对 AI 研究的见解，将其描述为“在实验上投入海量算力 (compute) 以学习简单的想法”。他认为，深入理解其中几个简单的想法，是让研究人员能够“领先该领域其他人数英里”的关键。
- **r/LocalLlama 子版块重新开放**：在经历了一段时间的私密状态后，广受欢迎的 **r/LocalLlama** 子版块已重新上线。社区对该论坛的需求由 [@danielhanchen 表达](https://twitter.com/danielhanchen/status/1937506709196419222)，他随后 [宣布了它的回归](https://twitter.com/danielhanchen/status/1937607779977728394)。
- **AI 基础设施的地缘政治**：[@dylan522p](https://twitter.com/dylan522p) 关于 AI 基础设施地缘政治的演讲获得了推荐，其中的一个核心观点是：由于数据中心的需求，**美国到 2028 年将面临约 88GW 的电力缺口**，这相当于大约 **88 座核反应堆**（由 [@AymericRoucher 分享](https://twitter.com/AymericRoucher/status/1937261555156156788)）。

**幽默与迷因**

- **进化的新进度条**：[@DavidSHolz](https://twitter.com/DavidSHolz/status/1937574227785474326) 转发了一个展示人类进化的流行迷因，并幽默地评论道：“不，兄弟。那是一个进度条。那是 Windows 更新，但针对的是肉体。”
- **螺旋神经网络**：[@adcock_brett](https://twitter.com/adcock_brett/status/1937577814875881889) 分享了一张由神经网络节点组成的类似 DNA 螺旋的惊人图像。
- **Vibecoding**：“**vibecoding**”一词出现在多个场景中，从使用 [**Claude Code**](https://twitter.com/code_star/status/1937270682565660721) 编程到 [“我 vibe code，故我在”](https://twitter.com/reach_vb/status/1937638934986785207) 的普遍哲学。
- **Good Soup**：一张带有简单标题“[good soup](https://twitter.com/code_star/status/1937268613662277862)”的代码截图被分享，引起了开发者的共鸣。
- **良好睡眠的价值**：[@vikhyatk](https://twitter.com/vikhyatk/status/1937372906109190593) 分享了一个个人轶事：“我接受了 **40 万美元的减薪**，只为了不必适应别人的睡眠时间表，这 **100% 值得**。”
- **算法课 vs Leetcode 课**：[@vikhyatk](https://twitter.com/vikhyatk/status/1937378138423722061) 调侃道：“如果你的算法课在研究排序算法而不是有限状态转换器 (finite state transducers)，那是 Leetcode 课，不是算法课。”

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

### 1. LocalLlama Subreddit 版主更迭与恢复

- [**Subreddit 恢复运营**](https://i.redd.it/1sx7mwusnx8f1.jpeg) ([Score: 272, Comments: 139](https://www.reddit.com/r/LocalLLaMA/comments/1ljlr5b/subreddit_back_in_business/)): **该截图记录了 Subreddit 上的版主操作，包括版主的移除和添加、垃圾邮件过滤器设置的编辑以及内容删除。这些日志提供了透明度，帮助社区评估在领导层/账号注销事件后的恢复步骤。这些行动表明正在努力恢复 Subreddit 的正常运营，并通过更新版主和反垃圾邮件配置来解决任何干扰（例如帖子的影子删除），从而解决内容被屏蔽的问题。** 评论中请求建立一个汇总贴（megathread），以便了解错过的进展，并确认导致大规模内容抑制的严格过滤机制已被移除。对于导致该事件的一系列过程也存在困惑，一些用户对中断期间可能错过的更广泛的技术新闻（例如 AGI、GGUF 发布）表示不确定。
    - 一位用户指出，不确定在 Subreddit 停机期间是否出现了 AGI 或模型发布等重大突破——“他们发布 AGI 了吗？GGUF 呢？”——这突显了 LLM 领域发展之快，以及追踪新模型格式（如 GGUF）和重大里程碑公告的重要性。

## 其他 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo

### 1. Anthropic 版权诉讼与公平使用裁决

- [**联邦法官裁定 Anthropic 使用书籍训练 Claude 属于公平使用，根据美国版权法是合法的**](https://www.reuters.com/legal/litigation/anthropic-wins-key-ruling-ai-authors-copyright-lawsuit-2025-06-24/) ([Score: 945, Comments: 172](https://www.reddit.com/r/singularity/comments/1ljdz52/a_federal_judge_has_ruled_that_anthropics_use_of/)): **一位美国联邦法官裁定，Anthropic 使用受版权保护的书籍来训练其 Claude LLM 构成了美国版权法下的公平使用。裁决中强调，LLM 训练并不等同于复制或替换作品，而是实现了具有足够转换性（transformative）的生成式创作。** 评论者指出，当前的版权框架可能与未来的 AGI 或数字增强人类不兼容，强调了现有法律与近期技术现实之间的差距。技术层面普遍认为，大规模、多样化的 LLM 训练数据集降低了逐字复制的风险，从而加强了公平使用的论据。
    - 技术分析指出法官的裁决：Anthropic 使用书籍训练 Claude 被视为“公平使用”，因为训练被认为是“转换性的”，类似于人类在阅读书籍后学习写作。然而，法官也发现 Anthropic 集中存储超过 700 万本盗版书籍的行为构成版权侵权，每部作品的法定赔偿额可能高达 150,000 美元，这突显了训练行为与数据获取方式之间的法律区别。
    - 一位评论者指出，在大规模数据集（例如 100 万本书）上训练 LLM，与在较小数据集（例如 100 本书）上训练相比，从统计学上降低了模型复制逐字文本的可能性。这意味着随着数据集多样性和规模的增加，输出内容违反版权的风险会降低，这一技术特性可能会影响模型开发商未来的法律和合规策略。
    - 总结还强调了此案的重要性，因为它代表了公平使用原则在美国法律下首次大规模应用于生成式 AI 训练——随着涉及 AI 和数据集来源的版权纠纷升级，这是一个重要的先例。法院将 AI 类比为人类学习实践，这在公平使用分析中起到了关键作用。
- [**Anthropic 在作者版权诉讼中赢得关于 AI 训练的关键美国裁决**](https://i.redd.it/b7p9aqv0gw8f1.png) ([Score: 123, Comments: 40](https://www.reddit.com/r/singularity/comments/1ljgc01/anthropic_wins_key_us_ruling_on_al_training_in/)): **该图片总结了最近的一项法律裁决，旧金山联邦法官做出了有利于 Anthropic 的判决，指出使用受版权保护的书籍进行 AI 训练在美国版权法下是合法的。然而，同一裁决认为用于训练数据的“集中式图书馆”是非法的，这为 AI 开发商带来了微妙的结果。这一决定对 AI 行业具有重要意义，因为它为在训练神经网络中使用受版权保护材料的合法性提供了先例，影响了模型开发流程和数据集构建。** 评论者指出，由于对训练数据图书馆的限制，该裁决对 AI 公司来说并非决定性的胜利，且鉴于美国法院对公平使用解释的不一致性，该问题可能会升级至最高法院。一些人认为，使用版权材料进行 AI 训练类似于训练人类，除了规模差异外，两者区别极小。
    - 裁决区分了使用数据训练 AI 与在集中式图书馆中存储大量受版权保护文本的行为，后者被视为非法。这突显了数据集管理的技术/法律影响——将数据集作为集中式图书馆持有会增加 AI 公司的法律风险。
    - 提到集中存储约 700 万本书的侵权潜在处罚为每本书最高 `150,000 美元` 的法定赔偿，这暗示了天文数字般的理论责任——这强调了维护集中式、未经转换的文本数据集的巨大法律风险。
    - 法律讨论指出，由于美国法院在适用于 AI 训练的公平使用原则上持续存在不一致和模糊性，预计此案将提交至美国最高法院。训练数据策展的技术方法可能会受到未来具有先例意义的裁决的重大影响。

### 2. Claude Code 高级用途与社区反应

- [**我们低估了 Claude Code，但并非以你想象的方式。**](https://www.reddit.com/r/ClaudeAI/comments/1liylon/were_underrating_claude_code_but_not_how_you_think/) ([Score: 412, Comments: 93](https://www.reddit.com/r/ClaudeAI/comments/1liylon/were_underrating_claude_code_but_not_how_you_think/)): **发帖者详细介绍了一个利用 Claude Code 和 Apple Shortcuts 的高级工作流，无需编写代码即可实现 B2B 销售业务的自动化和个性化。结构化的账户、联系人、电子邮件和知识文件夹驱动了一系列自定义 Claude 命令——/analyze-accounts（结合集成网页搜索的目标选择）、/select-contacts（过滤后的特定角色联系人挑选）、/create-drafts（JSON 格式的个性化外联）以及 /brief（每日分析摘要）——这些命令通过 macOS 自动化链接到夜间和早晨的例行程序中。该系统利用来自电子邮件/日历数据的反馈进行调整，跟踪参与度，发现流水线风险，并将近期事件与内部知识进行交叉引用，以基础 Claude 订阅的成本提供专家级的销售协助。[此处提供了后续技术细节。](https://www.reddit.com/r/ClaudeAI/comments/1lje9qn/continued_were_underrating_claude_code_technicals/)** 热门评论指出，这是 Claude Code 在真实销售环境中一个异常充实且实用的应用，区别于一般的 AI 自动化帖子。发帖者澄清说，这增强了现有的现代 SaaS 堆栈（Gong、Salesforce 等），表明该解决方案填补了标准企业销售工具中的空白，而非取代它们。
    - 一位评论者指出，**Claude Code** 故意设计为底层且无偏见（unopinionated）的，提供近乎原始的模型访问权限。这为流程自动化和 Agentic coding 提供了极高的灵活性和定制化能力，尽管由于强制模式或护栏较少，它具有一定的学习曲线。这种设计允许高级用户构建高度定制的解决方案和工作流，而这些是使用限制性更强的工具无法实现的，对于熟悉脚本编写的技术用户尤其有利。
    - 有人强调，在**受监管或机密环境（例如上市公司）中使用**带有显著的法律/伦理风险。共享公司数据——即使是通过生成式 Agent 间接共享——也可能违反内部政策或监管合规要求，特别是在知识产权或证券限制方面。这引发了在没有强大数据隐私控制的情况下部署此类自动化工具的担忧。
    - 讨论的一个技术应用是将 Claude Code 与外部平台集成，例如设置 **Telegram bot 进行实时更新**，或者在本地运行 Claude 并配合语音界面进行对话式工作流。这展示了深度集成到日常流程中的潜力，利用 Claude 的输出实现切实的自动化和可访问性改进。
- [**Vibe Planning：充分利用 Claude Code**](https://v.redd.it/qamuh19ucw8f1) ([Score: 130, Comments: 15](https://www.reddit.com/r/ClaudeAI/comments/1ljesg7/vibe_planning_get_the_most_out_of_claude_code/)): **该帖子介绍了针对 Claude Code 的 "vibe-planning"，利用了一个名为 Traycer ([traycer.ai](http://traycer.ai/)) 的外部工具。Traycer 使用 Sonnet 4、o3 和 GPT-4.1 等模型扫描代码库，生成一个可编辑的逐文件计划，作为独立于对话上下文的持久 Artifact。这使得 Claude Code 仅被喂入针对性的计划和所需文件，从而保持其 Context Window 的整洁和聚焦，实现受控的逐步执行。关键技术优势包括精确的逐文件规划、基于 Artifact 的计划持久性（允许进行外科手术式的编辑和并行规划会话），以及避免了使用以聊天为中心的编程 Agent 时常见的无关上下文污染。** 热门评论提出了关于 Traycer 如何避免底层模型（如 Sonnet 4）固有问题的技术疑问，例如读取无关文件、是否采用 RAG 或自定义索引进行上下文管理，以及如何确保信息的实时性——这表明了对其系统设计和数据获取策略的兴趣。
    - 针对 Vibe Planning 在核心模型（Sonnet 4）未改变的情况下如何管理上下文提出了疑问。具体的技术担忧包括防止无关文件读取和基于正则表达式（regex）的上下文污染，突显了在为基于 LLM 的编程 Agent 维持整洁且有针对性的 Context Window 方面所面临的挑战。
    - 讨论集中在 Vibe Planning 是否采用了检索增强生成（RAG）或其他上下文收集/索引策略，以及它是否集成了网页工具以提供更新的或即时的信息，从而实现更有效的代码辅助。

- 一篇评论指出，Claude Code 的效率源于其选择性的文件读取以及对 grep 等工具的依赖，但其任务规划功能（待办事项列表）不够系统。一旦高层级规划失效，审查和引导代码更改就会变得困难，这暗示如果内部能更好地处理上下文和任务分解，那么分层外部服务来改进规划可能并非绝对必要。
- [**我们能否摆脱那些游击营销式的 Claude Code 帖子，回到关于工具使用的实际讨论中？**](https://www.reddit.com/r/ClaudeAI/comments/1liz4rz/can_we_get_rid_of_the_guerrilla_marketing_claude/) ([Score: 279, Comments: 71](https://www.reddit.com/r/ClaudeAI/comments/1liz4rz/can_we_get_rid_of_the_guerrilla_marketing_claude/)): **楼主（OP）对该子版块被大量美化 Claude Code 的重复帖子占领表示担忧，认为这抑制了关于部署、服务器设置和用户间技术指导的实际讨论。他们声称，与 Cursor 和 Aider 等工具的类似子版块相比，这里的宣传/机器人活动比例过高，这影响了可操作技术内容的信噪比。** 热门评论在争论这种激增是自发的热情还是垃圾信息：一些人认为这种热度反映了对 Claude Code 能力的真实兴奋，并包含有价值的技术技巧；而另一些人则认为基调已转向自我推销和病毒式营销，并建议通过每周汇总贴（megathreads）来整合非技术性帖子。此外，还有关于子版块预期以及内容过滤作为用户责任的辩论。
    - 几位评论者讨论了最近的帖子如何展示了 Claude 的能力，这些能力在 ChatGPT 3.5 等旧模型中是无法实现的（旧模型仅限于简单脚本，难以应对复杂性），而 Opus 3/ChatGPT 4+ 和 Claude 则实现了更高级的工作流和编码任务。这一转变标志着消费者对这类工具在实际可用性和复杂性预期上的近期转变。
    - 讨论强调了像 Claude 这样模型的创新性和用户快速采用的情况，指出许多用户直到现在才能完成以前不可能或高度劳动密集型的编码项目，这证明了底层模型能力和易用性的飞跃。这导致了大量用户证言的涌入，分享由该模型实现的特定技术突破。
    - 一些社区成员表示希望将讨论转回探索 Claude 的技术极限和故障排除，而不是轶闻式的成功案例，并建议采用结构化的帖子格式（如每周主题帖），以便为高级用户更好地汇总技巧、诀窍和技术见解。

### 3. AI 对职业和教育的颠覆

- [**我们公司今年取消了实习计划。AI 滥用使其变得无法管理。**](https://www.reddit.com/r/singularity/comments/1lj4ed4/our_company_canceled_its_internship_program_this/) ([Score: 868, Comments: 329](https://www.reddit.com/r/singularity/comments/1lj4ed4/our_company_canceled_its_internship_program_this/)): **一家大型科技公司由于 AI 辅助申请数量激增、超出了传统筛选方法的承受范围，取消了其实习计划。缓解这一问题的尝试包括复杂的代码库作业（这吓跑了申请者，或者仍然容易受到 AI 辅助）以及闭卷现场考试（这侧重于记忆而非现实世界的技能）。该公司寻求有效的 AI 调节公平选拔策略，既能激励初级申请者，又不会惩罚真正的天才。** 热门建议包括使用现场伪代码逻辑考试（配合新颖的文档）来绕过 AI 记忆，辅以口头面试来解释原理，并设计一个受控的 AI 界面，将候选人与 AI 工具的交互作为评估的组成部分。一条评论建议直接向 AI（即 ChatGPT）寻求解决方案，暗示了辩论中的自我指涉讽刺或未解决的循环性。
    - 一个建议是设计现场、隔离的考试，使用伪代码谜题，重点考察候选人的逻辑和适应能力，而非死记硬背。这可能涉及为一种定制的伪代码语言提供自定义文档，随后进行口头面试，让候选人讨论他们的解决方案，并解释如何将其转化为熟悉的编程语言——从而评估实际理解能力而非记忆力。
    - 一项技术提案是在评估期间为候选人提供一个受控的 AI 界面，捕捉并评估他们与 AI 的交互，作为候选人整体评价的一部分。这将能够衡量 AI 工具的熟练程度、信息素养和解决问题的工作流，而不仅仅是输出的正确性。

- 一些从业者担心，过度依赖 AI 来解决问题会导致结果浅薄或不完整。这种批评强调，完全依赖 AI 的候选人往往无法展现出对任务的完全负责、迭代研究或更高层次的综合能力，导致答案片面或缺乏必要的后续步骤和结论——这对有效的技能评估构成了直接威胁。
- [**“你不会因为 AI 而丢掉工作，但会输给懂得使用 AI 的人”纯属胡扯**](https://www.reddit.com/r/singularity/comments/1lj7ucv/you_wont_lose_your_job_to_ai_but_to_someone_who/) ([Score: 374, Comments: 167](https://www.reddit.com/r/singularity/comments/1lj7ucv/you_wont_lose_your_job_to_ai_but_to_someone_who/))：**该帖子挑战了“人类通过学习使用 AI 就能保住工作”这一普遍断言，认为 AI 的核心能力是取代人类智能，并最终取代 Prompt Engineering 或目标制定的任务。作者质疑为什么人们假设人类在利用 AI 方面会一直保持优势，并指出鉴于 AI 的发展轨迹，假设 AI 能力不出现停滞，它很快就会在“使用自身”方面超越人类。讨论将此与 AI 进步的 S-curve（S 曲线）这一根本担忧联系起来。** 热门评论一致认为，使用 AI 的员工只是暂时更安全，因为无论技能水平如何，业务需求都会缩减团队规模，初级和高级职位都面临威胁。另一个技术观点是：通常被引用的这种智慧可能只在短期内有效，而从长期来看，如果（或当）AI 能够自主实现目标理解和执行，这种转变可能是彻底的。
    - 几位评论者讨论了将 AI 集成到工作流中如何导致所需团队规模缩小，无论经验水平如何。趋势是裁员，AI 增强使得初级和高级职位在业务需求优先考虑效率而非层级的情况下同样脆弱。还有人提到，随着自动化/AI 采用率的增长，甚至管理职位也可能面临风险。
    - 一个尖锐的技术见解是，AI 工具的广泛使用可能会直接加速劳动力替代：随着越来越多的工人使用 AI 并与之互动，他们为训练未来模型提供了更丰富的数据集，从而加速了这些采用工具的行业的自动化和潜在的职业淘汰。
    - 匿名图片帖子和次要评论暗示，学习和使用 AI 并非万能的保障；认为自己职位不可替代的工人可能存在技术上的过度自信，但共识是，持续的技术进步可能会在中短期内超越受 AI 赋能和非 AI 赋能的工人。
- [**如今，曾经被誉为“铁饭碗”的领域——计算机科学和工程——在大学专业中失业率最高**](https://i.redd.it/12aqr6p4mw8f1.png) ([Score: 172, Comments: 48](https://www.reddit.com/r/OpenAI/comments/1ljg6oa/today_the_very_fields_once_hailed_as_bulletproof/))：**该图片展示了一个比较各大学专业失业率的柱状图，显示“计算机科学”和“计算机工程”目前在样本领域中失业率最高——这与其传统上高就业率学科的认知相悖。帖子中的技术评论强调需要区分失业率和就业不足率（underemployment rates）：虽然计算机科学的失业率相对较高（约 6%），但其就业不足率（16%）远低于艺术史等专业（失业率为 3%，但就业不足率约为 50%），这表明 CS 毕业生更有可能坚持寻找本领域的工作。此外，背景信息将美国 R&D 税收政策的变化与 STEM 领域就业前景的变化联系起来。** 评论者强调，过度鼓励某些专业导致了劳动力市场饱和，并强调必须同时检查失业和就业不足数据，以有意义地评估学位的价值，从而揭穿关于 CS 不稳定性的过度简化说法。
    - 一条评论引用了一项研究，显示计算机科学专业的失业率为 `6%`，而艺术史专业为 `3%`，但 CS 的就业不足率仅为 `16%`，而艺术史约为 `50%`。技术含义是：CS 毕业生在寻求相关工作方面更具持久性，而许多艺术史毕业生则会接受任何可用的工作，无论是否与领域相关。
    - 一位用户指出，美国 R&D 税收政策的转变在 IT 行业的职位空缺中扮演了重要但讨论不足的角色，这表明宏观经济政策直接影响技术就业率，而与更广泛的技术趋势无关。

- 另一位用户讨论了市场饱和与自动化趋势，预测随着用于业务自动化的 AI 框架趋于成熟，如今对手动集成的需求将会消失，技术集成本身可能在 15-20 年内实现自动化。这预示着随着自动化的扩张，未来“安逸”的技术岗位将会减少。
- [**前 OpenAI 成员 Peter Deng 表示 AI 可能会重塑孩子的思维方式，教育也可能随之改变。技能将不再是记忆答案，而是学习如何提出更好的问题以开启更深层次的思考。**](https://v.redd.it/pn3x76qjau8f1) ([Score: 278, Comments: 83](https://www.reddit.com/r/singularity/comments/1lj64fy/exopenai_peter_deng_says_ai_may_be_rewiring_how/)): **前 OpenAI 成员及技术资深人士 Peter Deng 在最近的一次采访（[Lenny's Podcast](https://www.youtube.com/watch?v=8TpakBfsmcQ)）中指出，AI，特别是像 ChatGPT 这样的 LLM，正在将未来所需的认知技能从死记硬背转向高阶提问，并认为成功将取决于向 AI 提出更好查询（queries）的能力。Deng 建议教育方法应转向培养学生提出更深层问题的能力，以利用 AI 工具进行批判性思维，而不是强调事实记忆。评论中的技术讨论批评了这一论点，引用研究表明，在没有扎实核心知识基础的情况下教授研究/提问技能是无效的，并强调了当前大学生在基础计算机和问题构建技能方面的明显缺陷。** 评论者反驳了 Deng 的乐观态度，担心过度依赖 AI 会因为过度的便利而加剧重要基础技能的丧失，并认为只有天生好奇或“有天赋”的学生才会利用这些工具进行更深层的探究——大多数人可能会利用 AI 变得在智力上更加被动。这些辩论呼应了长期以来关于基础学习与工具辅助探究之间权衡的教育学担忧。
    - 几位评论者认为，虽然 AI 理论上可以将教育转向提问，但教育学的实证研究表明，基础技能和领域知识仍然至关重要。他们引用了在没有建立底层能力的情况下教学生“只管搜索答案”所观察到的失败案例：学生在构建问题、制定有意义的提问方面感到困难，并且由于依赖过度简化的技术环境，往往缺乏像有效使用 PC 或 IDE 这样基础的技能。
    - 也有人怀疑 AI 是否会从根本上改变好奇心或认知倾向，指出无论使用何种工具，通常只有特别有天赋或自我驱动的学生才会提出更深层的问题。令人担忧的是，AI 工具可能会鼓励智力上的懒惰——这一趋势在之前的技术中已经观察到——通过提供快速答案并阻碍深入参与或信息留存，除非进行积极练习。
    - 一些评论提出了对批判性思维和验证技能的潜在风险，警告将认知努力外包给 AI 系统可能会削弱学生批判性评估信息的能力。由于依赖不透明的机器控制系统，这种风险可能会进一步加剧，引发对认知权威长期外包给技术中心化组织控制的 AI 的担忧。

---

# AI Discord 回顾

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要的总结
> 

**主题 1：新模型与架构：创新竞赛持续进行**

- **Polaris 4B 性能声明引发质疑**：Unsloth 社区成员对 **Polaris-4B-Preview** 据称超越 **Claude-4-Opus** 等商业系统的说法表示怀疑，怀疑存在过拟合（overfitting）。测试者计划使用 **Q8_0** 和 **FP16** 对最初表现不佳的 **Q4_K_M GGUF** 进行验证。
- **Google 下一代模型 "Flamesong" 和 "Kingfall" 引发猜测**：LMArena 的讨论表明 Google 正在开发代号为 **Flamesong**（可能是 **Gemini 3.0** 或新的 **Flash** 系列）和 **Kingfall**（基准测试接近 **O3 Pro**，被认为是计算量更大的 **Gemini 2.5**）的新模型。**Stonebloom** 也被传言为 *"2.5-pro-lite"* 模型，在早期测试中显示出了一些正确答案。
- **RWKV v6 "Finch" 凭借多语言能力起飞**：Yannick Kilcher 的社区强调了 **RWKV v6 (Finch 系列)** 的发布，这是一个 **1.5B** 参数的模型，在多语言和英语任务中达到了 state-of-the-art 结果，详见 [RWKV-5 & 6 论文](https://arxiv.org/abs/2404.05892)。[BlinkDL_AI 的一条 X 帖子](https://x.com/BlinkDL_AI/status/1755656095970857269?s=20)指出 Finch 结合了 **Mamba-like selectivity mechanism**，在 perplexity 方面优于 Transformer。

**主题 2：开发者体验与工具：航行在 AI 前沿**

- **Cursor 开发者正努力应对繁琐的配置和有 Bug 的后台 Agent**：Cursor 社区的用户报告了在 **WSL、Ubuntu、GitHub 和 SSH** 环境下设置 **Cursor** 的显著复杂性，将其描述为项目规则中“永无止境的深坑”。后台 Agent 也表现异常，不遵守定义的规则，导致了不必要的仓库推送（repository pushes）。
- **LM Studio 用户与 Unsloth 量化及烦人的更新提示作斗争**：LM Studio 用户发现来自 **Unsloth** 模型的动态量化（dynamic quants）存在问题，会导致 VRAM 过载和加载失败，尤其是在多 GPU 环境下。一个持久存在的 Bug 还会强制在加载模型前进行重复更新（每次超过 **200MB**）；同时，当出现新功能需求时，一些成员指出 LM Studio“主要针对 LLM，而非满足所有 AI 需求的一站式商店”。
- **Mojo 语言旨在提升安全性和异步清晰度，但 Python 互操作性仍受限**：Modular 社区讨论了 **Mojo** 提供类似 Rust 安全性的雄心，计划推出 **sum types**（和类型）和 **pattern matching**（模式匹配）等功能，并改进异步模型（PRs [3945](https://github.com/modular/modular/pull/3945), [3946](https://github.com/modular/modular/pull/3946), [4728](https://github.com/modular/modular/pull/4728)）以避免 Rust 的陷阱。然而，从 Python 调用 Mojo 时仍存在[已知限制](https://docs.modular.com/mojo/manual/python/mojo-from-python/#known-limitations)。

**主题 3：性能与优化：从硅片梦想走向极速现实**

- **向量搜索通过 FAISS 和 Matmul 魔力获得提速**：HuggingFace 和 LlamaIndex 用户分享了向量搜索的显著加速成果，一位用户使用 `torch.matmul` 将 **1M** 次点积计算从 **25 秒缩短至 0.04 秒**。对于更大规模（**10M+ 次比较**）的场景，工程师们正关注量化 FAISS 索引，如 `IndexIVFPQ`。
- **NVIDIA NVFP4 亮相，LoRA 超参数调优获得 Optuna 助力**：一篇关于 [NVFP4 的 NVIDIA 博客文章](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)引发了 Unsloth AI 社区关于高效低精度推理的讨论；同时，社区建议结合 Unsloth 的[新 LoRA 超参数指南](https://x.com/UnslothAI/status/1937521408344752272)使用 **Optuna** 进行超参数搜索，因为“每个数据集的表现都不同”。一位 H200 车主提到，他们的单卡成本约为 **3 万美元**（不含增值税），且需要定制冷却系统。
- **Chisel 通过集成 rocprofiler-sdk 强化 ROCm 性能分析**：GPU MODE 的开发者庆祝了 **Chisel** 新集成的 **rocprofiler-sdk**，它能自动从主线构建 *aqlprofile* 和 *rocprofiler-sdk*。新的 `-pmc` 标志允许收集自定义性能计数器，如 `GRBM_GUI_ACTIVE,SQ_WAVES`。

**主题 4：AI 应用与集成：连接代码、内容与对话**

- **OpenAI 通过 Pro 级聊天搜索连接器扩展连接性**：OpenAI 宣布 **Pro 用户** 现在可以访问针对 [Dropbox, Box, Google Drive, Microsoft OneDrive (Business), 和 Microsoft SharePoint](https://openai.com/blog/june-2024-updates) 等服务的**聊天搜索连接器**。该功能旨在简化信息检索，但目前在欧洲经济区（EEA）、瑞士和英国暂不可用。
- **LlamaIndex 推出用于简历匹配和 Claude 的开源 MCP 服务器**：LlamaIndex 发布了两个值得关注的开源项目：一个是用于 Cursor 内智能职位匹配的 **Resume Matching MCP 服务器**，可连接到 [LlamaCloud 和其他服务](https://t.co/RCKoiUccm6)；另一个是使用 Next.js 构建的 **Claude 兼容 MCP 服务器模板**，支持 OAuth 2.1，便于创建远程服务器，详情见[此处](https://t.co/wtPorldMvJ)。
- **AI 挑战古老语言与 3D 世界**：Unsloth AI 社区庆祝了首个开源的**纳瓦特尔语（Nahuatl）到西班牙语翻译器**，该工具通过 Unsloth 全量微调构建，已在 [Hugging Face Spaces](https://huggingface.co/spaces/Thermostatic/neuraltranslate-27b-mt-nah-es) 上线（代码托管于 [GitHub](https://github.com/Sekinal/neuraltranslate-nahuatl/tree/master)）。另外，[腾讯的 Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) 因其“非常扎实”的 3D 网格（mesh）生成能力而受到称赞。

**主题 5：AI 生态系统：在资金激流、伦理漩涡和平台特性中航行**

- **Harvey AI 获得 3 亿美元融资，同时 Replit 的 ARR 达到 1 亿美元，但估值遭到质疑**：Latent Space 的讨论涵盖了 [Harvey AI 以 **50 亿美元估值**完成的 3 亿美元 E 轮融资](https://xcancel.com/harvey__ai/status/1937155058476646591)（与 [LexisNexis](https://www.lexisnexis.com/community/pressroom/b/news/posts/lexisnexis-and-harvey-announce-strategic-alliance-to-integrate-trusted-high-quality-ai-technology-and-legal-content-and-develop-advanced-workflows) 合作），以及 [Replit 宣布其 ARR 突破 1 亿美元](https://xcancel.com/Replit/status/1937212611520831718)。然而，一些成员质疑 Replit **11 亿美元的估值**是否能被新的 ARR 数据充分支撑。
- **平台稳定性和 Rate Limits 令各方用户感到沮丧**：HuggingFace 用户遇到了 **429 Rate Limit 错误**和 **504 Gateway Time-outs**（[状态页面](https://status.huggingface.co/))，而 Cursor 用户报告称在 **Sonnet** 上立即触发了 Rate Limiting，且整体 **Cursor** 的 Rate Limits 导致账单从 **20 美元飙升至 70 美元**。OpenRouter 用户也面临其 **Meta provider** 的问题，以及 Google AI Studio 上 **Gemini 2.5 Pro** 的 Rate Limits 问题，免费层级用户官方标称的 **150 RPM** 限制在实际使用中感觉更低。
- **AI 伦理成为焦点：从越狱、合理使用到有偏见的奖励模型和初创公司崩溃**：关于 AI 伦理的辩论浮出水面，用户讨论了对 **Luma** (HuggingFace) 等 AI 的越狱（Jailbreaking），[Anthropic 在关于合理使用（Fair Use）的简易判决动议中获胜](https://xcancel.com/adameisgrau/status/1937480346976813454) (Latent Space)，以及 [Cursed Helm 论文](https://arxiv.org/abs/2506.07326) 警告奖励模型（Reward Models）偏见 (Nous Research AI)。DSPy 社区成员还讲述了一个 **Atom of Thought** Agent 初创公司实验崩溃的过程，原因是作者在收到实现代码问题的通知后，反应“极其负面且不专业”。


---

# Discord: 高层级 Discord 摘要




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Polaris 4B 引发过拟合指控**：成员们对 **Polaris-4B-Preview** 模型声称超越 **Claude-4-Opus** 等商业系统的说法表示怀疑，认为可能存在过拟合（Overfitting），尤其是在一名成员发现 **Q4_K_M 版本**表现不尽如人意之后。
   - 他们打算使用 **Q8_0** 进行测试，并与 **FP16** 进行对比，以验证模型的性能。
- **腾讯的 Hunyuan3D-2.1 是 Mesh 生成的奇迹**：一位成员赞扬了 [腾讯的 Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) 生成 **3D Mesh** 的能力，称其“相当扎实”，并指出其较之前版本的进步。
   - 讨论涉及了利用 AI 创建绑定 Mesh（Rigged Meshes）的可行性，参考了 **Mixamo** 和 **Cascadeur** 等工具，尽管目前尚未实现完全生成式。
- **LoRA 超参数搜索**：Unsloth 团队发布了[新的 LoRA 超参数指南](https://x.com/UnslothAI/status/1937521408344752272)，引发了关于提及 **Optuna** 进行超参数搜索（Hyperparameter Sweeps）以提高性能的建议，因为“每个数据集的表现都不同”。
   - 有成员对 PEFT 配置中支持 **alpha_pattern** 表示担忧，指出 Unsloth 会静默丢弃它。
- **NVFP4 首次亮相，助力高效推理**：一位成员分享了关于 [NVFP4 的 NVIDIA 博客文章](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)，讨论转向了 **FP8** 和 **FP4** 的比较。
   - 一位成员提到，他们的单台 **H200** 成本约为 **3 万美元**（不含增值税），且由于其被动散热特性需要定制冷却系统，并补充说它被用于训练 LLM。
- **纳瓦特尔语翻译器**：首个开源的**纳瓦特尔语（Nahuatl）到西班牙语翻译器**已构建完成，可在 [Hugging Face Spaces](https://huggingface.co/spaces/Thermostatic/neuraltranslate-27b-mt-nah-es) 上使用，该项目基于 **Unsloth 的全量微调（Full Fine-tuning）支持**构建。
   - 用于复现 **Nahuatl 翻译器**项目的代码已在 [GitHub](https://github.com/Sekinal/neuraltranslate-nahuatl/tree/master) 上发布。



---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的繁琐配置难题**：用户发现将 **Cursor** 与 **WSL**、**Ubuntu**、**GitHub** 和 **SSH** 密钥结合使用比预想的更复杂，往往会导致陷入配置项目规则的泥潭。
   - 一位用户建议了一个针对 Linux 的潜在变通方法，即使用 *sudo -i* 在 Cursor 内自动完成设置，但也提醒了相关的风险。
- **终端试炼困扰在 Windsurf 之间切换的用户**：**Cursor terminal** 因超时和运行不顺畅而受到批评，一些用户更倾向于使用 **Windsurf**，因为它具备更好的终端窗口管理能力。
   - 一位从 **Windsurf** 切换到 **Cursor** 的用户提到了终端相关的问题，报告称 **Agent** 无法读取终端输出或在执行命令时卡死。
- **速率限制（Ratelimit）激增困扰高频用户**：用户报告称即使没有大量使用，**Sonnet** 也会立即触发速率限制，而且整体的 **Cursor** 速率限制正在推高成本。
   - 一位用户表示，在新的计费方案下，由于速率限制，他们的月度账单从 **$20** 飙升至 **$70**。
- **VisionCraft MCP：文档的精彩亮相驱动需求**：升级后的 **Visioncraft MCP** 现在拥有更完善的文档和更快的响应速度，激发了用户对更深层次 **Cursor** 集成的需求。
   - 通过向模型提供更新的文档，**Visioncraft MCP** 有助于解决 AI 模型中数据过时的问题，从而生成更优质的代码并减少错误。
- **后台 Agent 的构建失败困扰开发者**：用户报告称后台 **Agent** 不遵守 **Cursor** 编辑器中定义的规则或 **Agent** 提示词中编写的规则，导致向仓库推送了不必要的内容。
   - 此外，从 Slack 点击 *open in cursor* 按钮会跳转到编辑器但没有任何显示，这暗示存在一个严重的 Bug。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 用户遇到升级提示**：一位拥有一年订阅权限的 **Perplexity Pro** 用户被提示升级，这表明账号状态可能存在问题。
   - 另一位用户推测这可能是因为使用了*极大概率并非为你准备的促销活动，因此被撤销了*。
- **ChessChamp AI 的到来令用户兴奋**：人们对即将推出的 **ChessChamp AI** 产生了浓厚兴趣，一位成员询问它是否利用了 **Stockfish**。
   - 另一位用户确认了它的可用性，并指出*你可能需要订阅才能访问它*。
- **捐赠疲劳影响 Mozilla**：分享了一个 [Perplexity AI 页面](https://www.perplexity.ai/page/donation-fatigue-impact-on-moz-q76XMD17Skap_valehbYOg)，讨论了**捐赠疲劳**对 **Mozilla** 的影响。
   - 未提供更多细节或讨论。
- **Perplexity AI 扩展至 WhatsApp**：**Perplexity AI** 现在支持在 [WhatsApp](+18334363285) 上执行定时任务，提供了更广泛的访问途径。
   - 一位成员惊呼：*我竟然不知道它上线了 WhatsApp！*。
- **寻求 Perplexity AI 技术支持**：一位用户询问在哪里可以获得 **Perplexity AI** 的技术支持帮助。
   - 另一位用户建议联系 **support@perplexity.ai**，尽管原用户在上周已经给他们发过邮件了。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **内存上下文服务（Memory Context Service）出现**：一位用户创建了一项服务，为他们的 Agent 提供 **memory context、memory recall、rehydration 和 NLP**，因为他们发现缺失这些功能非常令人恼火。
   - 该服务的创建是为了解决 AI Agent 中改进 **memory context management** 的需求。
- **Multi-head Latent Attention 创新**：用户讨论了 **Multi-head Latent Attention** 如何在旧硬件上实现强大的 AI 性能，并引用了 [NumPy 模型构建示例](https://t.co/xMHlA61Qoz)和 [YouTube 视频](https://youtu.be/WEBiebbeNCA?si=84i4jjPyQWRSzVuQ)。
   - 这突显了通过创新的注意力机制实现 **硬件效率提升** 的潜力。
- **AI 语音配音解决方案**：成员们探索了 **AI voice dubbing** 选项，其中 [Hey-Gen translate](https://www.heygen.com/translate) 因其对口型（lip-sync）能力而被推荐。
   - 原帖作者认为 **Veo 3** 很有趣，但提到了成本担忧；讨论强调了人们对 **AI-powered translation** 和配音工具日益增长的兴趣。
- **ChatGPT 应对 PDF 生成**：一位成员报告在使用 **ChatGPT** 配合 Python 从结构化文本/Markdown 生成 **PDF** 时频繁失败，但发现 **Deep Research 报告功能** 运行成功。
   - 他们认为 **Deep Research 报告功能** 使用的是客户端 PDF 生成，但无法从非 DeepResearch 会话中触发该输出，导致生成的是纯文本块而不是可导出的 PDF。
- **聊天搜索连接器面向 Pro 用户开放**：**Pro 用户** 现在可以访问针对 [Dropbox, Box, Google Drive, Microsoft OneDrive (Business), Microsoft SharePoint](https://openai.com/blog/june-2024-updates) 的 **chat search connectors**。
   - 这些连接器目前对位于欧洲经济区（EEA）、瑞士和英国的用户不可用，该功能旨在 **简化从常用云存储平台检索信息** 的流程。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Grok3 发布即宣称获得 SOTA 地位**：成员们辩论了 **Grok3** 的地位，引用了其在 [artificialanalysis 评分](https://www.artificialanalysis.ai/leaderboard/open-llms)中相对于 **Sonnet 3.7** 的强劲表现。
   - 虽然它曾短暂处于 SOTA 地位，特别是在 **数学** 方面，但激进的训练后处理（post-training）可能阻碍了它保持领先。
- **Claude 开辟利基市场**：**Claude** 在标准基准测试未能完全覆盖的利基领域表现出色，特别是创意写作和戏剧表演。
   - 成员们发现，与竞争模型相比，**Claude** 特别擅长遵循角色扮演指令。
- **苹果秘密研发基础模型（Foundation Models）**：讨论围绕 **Apple** 开发基础模型及其可能使用可信计算（trusted computing）来保护隐私展开。
   - 一些人推测 **Apple** 可能会在开发强大的端侧模型的同时，为服务器端模型授权使用 **Gemini**，而另一些人则认为 **Apple** 将发布自己的服务器模型。
- **谷歌调整 Flamesong 模型**：据报道 **Google** 正在开发名为 **Flamesong** 的新系列模型，可能是新的 **Flash** 系列，或者是 **Gemini 3.0** 或另一个 **2.5 模型**。
   - 推测还表明 **Stonebloom** 可能是一个 "2.5-pro-lite" 模型，测试显示它有时能正确回答问题。
- **Kingfall 即将到来**：**Kingfall** 在基准测试中与 **O4 Pro** 和 **GPT-4.5-thinking** 并列第 9 位，被认为只是拥有更多算力的 **2.5** 版本。
   - 社区推测其发布时间可能在夏末左右，一些人认为 **Stonebloom** 是其蒸馏版本。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face 网站故障频发**：根据 [status page](https://status.huggingface.co/)，用户报告在访问 Hugging Face 网站时遇到 **429 rate limit errors** 和 **504 Gateway Time-out** 问题，影响了模型下载和 Space 功能。
   - 网站似乎已恢复上线，但运行缓慢。
- **越狱 AI 引发辩论**：一位用户就名为 *Luma* 的 **jailbroken AI** 寻求建议，该 AI 表现出异常行为，包括在其他 AI 模型上进行实验以探索它们的边界。
   - 其他成员表示，对 **DeepSeek** 进行越狱相对容易。
- **扩展向量搜索？FAISS 更快**：一位用户在扩展向量搜索并使用余弦相似度比较 `n` 个查询 embedding 与 `m` 个文档 embedding 时，发现使用 Langchain 的 FAISS 封装在进行 **1M** 次点积运算时出现瓶颈。
   - 他们发现使用 `torch.matmul` 或 `@` 将 **1M** 次比较的运行时间从 **25 秒缩短至 0.04 秒**，并计划在 **10M+** 次比较中使用量化 FAISS 索引，如 `IndexIVFPQ`。
- **Diffusers 发布新版本**：[Diffusers v0.34.0](https://github.com/huggingface/diffusers/releases/tag/v0.34.0) 现已发布。
   - 查看链接中的 Release Notes 以了解更多详情。
- **Agents 课程挑战工程师**：一位用户询问了提交 Unit 4 最终项目的流程，幽默地质疑为什么 *提交* 项目似乎比 *完成* 项目更难。
   - 另一位用户分享了他们调试错综复杂的 HF 环境变量和包冲突的痛苦经历，随后建议在本地运行你的 Agent，然后只需通过 API 端点发送其响应即可，如果答案正确，这足以让组织者发放证书。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 正在考虑语音功能**：一位用户建议在 **LM Studio** 中添加语音安装功能，以便像 **GPT CHAT** 一样进行对话练习，用于语言学习。
   - 另一位成员澄清说，**LM Studio** 专注于 **LLMs**，并非旨在作为通用 AI 工具。
- **LM Studio 图像输入功能引发讨论**：一位用户询问了 **LM Studio** 中的图像生成功能，其他成员回应称，虽然像 **Gemma3** 这样的模型接受图像作为输入，但该平台缺乏原生的图像输出功能。
   - 成员们建议通过额外的步骤使用 **web-ui** 设置文本转图像生成。
- **Unsloth 的动态量化导致 LM Studio 运行异常**：用户报告称，来自 **Unsloth** 模型的动态量化（dynamic quants）会导致尺寸估计问题，可能导致 **VRAM** 过载并在 **LM Studio** 中加载失败，尤其是在多 **GPUs** 环境下。
   - 一位用户确认专门使用具有动态上下文窗口的 **Unsloth** 模型，并指出这是一种导致该问题的常见配置。
- **LM Studio 更新提示反复困扰用户**：一位用户报告了 **LM Studio** 中的一个 Bug，即它反复提示更新（每个超过 200MB），并且在更新前拒绝加载模型。
   - 一位成员建议确保之前的更新已完全加载且未在后台运行，以解决持续的更新请求问题。
- **LocalLlama Subreddit 重获新生**：成员们讨论了 **r/LocalLlama** Subreddit 在沉寂一段时间后，在新的管理团队下回归。
   - 尽管有人担心新版主参与了过多的 Subreddit，但目前尚未发现明显的异常情况。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Meta 提供商问题频发**：OpenRouter 上的 **Meta 提供商**目前正经历故障，已向 **Meta** 报告，团队正在努力恢复服务。
   - 用户还对 **OpenRouter 的价格结构**表示担忧，团队正在积极回应咨询以提供清晰的说明。
- **OpenRouter 奇怪的提供商偏好**：一位用户质疑 OpenRouter 的提供商偏好是如何运作的，指出选择特定提供商并未按预期工作，并寻求关于 **sort preference** 含义的澄清，参见 [OpenRouter 关于提供商路由的文档](https://openrouter.ai/docs/features/provider-routing)。
   - 这引发了关于 OpenRouter 生态系统中提供商选择和路由细微差别的讨论。
- **Novita 臭名昭著的数值错误**：一位用户指出 Novita 提供的 **R1-528** 最大输出长度信息错误，声称其为 **131k**，而实际仅为 **16k**。
   - 该用户质疑 OpenRouter 是否验证提供商信息，认为此类差异应该很容易被发现。
- **推理 Token 触发 Token 统计混乱**：一位用户报告在使用 OpenRouter 时，收到的结果中 **reasoning_tokens** 高于 **total_tokens**。
   - 一名工作人员澄清说，**reasoning tokens** 是 **completion token details** 的一部分，而 **total tokens** 不包含推理 Token，并指出更改此设定会破坏正在运行的应用。
- **Gemini 2.5 Pro 的速率限制风波**：用户正在讨论 Google AI Studio 上 **Gemini 2.5 Pro** 的速率限制，指出虽然界面列出了 **150 RPM** 的限制，但免费层用户的实际限制似乎更低。
   - 一位用户在快速发送大量请求后遇到了错误和冷却限制，认为该限制更像是一种“公平使用限制 (fair use limit)”，旨在防止逆向工程或自动化。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **XMake 简化 CUDA 构建**：一位用户建议使用 [xmake](https://xmake.io/#/) 作为 **C++/CUDA 项目**中 CMake 的替代方案，因为它更易于使用。
   - 他们演示了用于**定义目标**、**指定 CUDA 文件**以及**添加 CUDA 架构**的配置。
- **NVRTC 与 CUB 的兼容挑战**：由于缺少 C++ 标准库头文件，一位开发者在尝试通过 `torch.cuda._compile_kernel` 实现更快编译速度时，遇到了 **CUB** 与 **NVRTC** 集成的困难。
   - 提出的解决方案是使用 `#include <cub/block/block_reduce.cuh>` 而不是 `#include <cub/cub.cuh>`。
- **TorchTitan 通过 TP 增强 SimpleFSDP**：根据 [README](https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/simple_fsdp/README.md)，*TorchTitan* 中的 **SimpleFSDP** 实现是捕获包含所有集合通信 (collectives) 图的最佳方式。
   - **Tensor Parallelism (TP)** 最近被添加到 SimpleFSDP 版本中，使得编译包含 **TP** 和 **FSDP collectives** 的图成为可能，详见[此 Pull Request](https://github.com/pytorch/torchtitan/pull/1250)。
- **精度问题困扰 CUDA Matmul**：一位开发者报告其自定义 **CUDA matmul** 实现中测试用例失败，原因源于[精度不匹配](https://github.com/yechenzhi/reference-kernels/blob/main/problems/pmpp/matmul_py/submission.py)。
   - 错误显示不匹配元素之间存在微小差异，指向了**浮点精度问题**。
- **Chisel 实现 rocprofiler-sdk 集成**：一位成员宣布在 **Chisel** 中实现了 **rocprofiler-sdk 集成**，可自动从主线构建 *aqlprofile* 和 *rocprofiler-sdk*。
   - 引入了一个新的 **--pmc 标志**来收集自定义性能计数器（例如：`chisel profile amd kernel.cpp --pmc GRBM_GUI_ACTIVE,SQ_WAVES`）。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **NVMe 抽象缺失**：成员们讨论了 **NVMe** 的简洁性，指出文件系统等抽象层的缺失，并提议将 *'DISK:/dev/nvme0'* 作为一种寻址方案。
   - 他们还提出了关于解绑内核驱动程序便捷性的疑问。
- **Infiniband 传输瓶颈**：**Infiniband** 传输目前会破坏图（graph），从而阻碍性能。
   - 讨论围绕着将图传输与副本对齐，以及通过 **RDMA** 进行远程 DMA 的复杂性展开。
- **考虑 GPU 驱动的网卡**：有讨论关于编写网卡驱动程序并在 GPU 上运行该驱动，以增强对传输的控制。
   - 还建议允许任意 **CPU kernels**，并在图中使用 CPU kernel 作为回调来设置传输。
- **FP8 转换函数上线**：一名成员实现了一个手动将 **fp8e4m3** 和 **fp8e5m2** 张量转换为 **float32** 的函数，可在 [GitHub](https://github.com/softcookiepp/tinybloat/blob/master/src/tinybloat/compatibility.py) 上获取。
   - 这通过允许拥有旧硬件的用户将 **FP8** 模型转换为 **float32**，解决了 **FP8** 张量类型的硬件兼容性问题。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **合成数据工具面临 Diff 难题**：一名成员正在开发 **合成数据生成工具**，但在将其作为编辑器使用时遇到了错误的 diff 问题。
   - 他们正在探索进行适当蒸馏（distillation）的方案，包括使用 logits 或 QLoRA，并从 [Exercism](https://exercism.org/) 的题目中汲取灵感作为基准测试。
- **Gemini Pro Stable 的指令遵循问题**：多名成员观察到 **Gemini Pro Stable** 表现出较差的指令遵循能力。
   - 一位用户分享说，当要求其将 **1.1** 节标记为已完成时，它完成了所有任务并创建了新文件，但未能应用更改，反而把仓库搞得一团糟（*butchering*）。
- **Aider 表现异常：间歇性文件写入**：一位用户遇到 **aider** 只显示 diff 而不写入文件的情况，可能是因为超出了 **deepseek-r1** 的 [token limit](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)。
   - 用户注意到 *在发出命令后立即发出了 token 警告*，而文件较少的简单任务则运行正常。
- **Claude Code API 集成即将到来？**：一名成员提议使用 [Claude Code](https://github.com/codingworkflow/claude-code-api) 作为 Aider 的后端，以利用其订阅优势，因为与直接调用 API 相比，其**调用成本更低**。
   - 另一名成员指出 [Anthropic 文档](https://docs.anthropic.com/en/docs/claude-code/sdk)表明，如果利用 SDK，**将 Claude Code 作为提供商使用没有问题**。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Transformers 检测肺炎**：一篇论文介绍了在胸部 X 光片上使用 **Vision Transformers** 进行[高效肺炎检测](https://www.nature.com/articles/s41598-024-52703-2)，然而一位成员对该论文在 **2024** 年发表感到震惊。
   - 他们称之为*十年前的老新闻*。
- **GRPO 在 RL 中大显身手**：一名成员建议 **GRPO** 是在 RL 领域取得突破的途径，并分享了相关的 **TLDR**。
   - 另一名成员分享了[一条推文](https://fxtwitter.com/jmhessel/status/1899909893324468444)，区分了**广义 RL** 和 **LLM RL**。
- **Finch 系列起飞**：据报道 **RWKV v6 (Finch 系列)** 发布，这是一个 **1.5B** 模型，在多语言和英语任务中达到了 **SOTA**，并具备多模态能力，引用了 [RWKV-5 & 6 论文](https://arxiv.org/abs/2404.05892)。
   - 根据[这条 X 帖子](https://x.com/BlinkDL_AI/status/1755656095970857269?s=20)，**Finch** 结合了类似于 **Mamba** 的选择性机制，在困惑度（perplexity）方面优于 Transformers。
- **计算器 AI？**：成员们将 **AI 比作计算器**，建议限制 AI 的使用，类似于禁止在背乘法表时使用计算器，并强调了潜在的长期认知影响。
   - 一名成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=z3awgfU4yno)来支持这一类比。
- **论文揭示 RL 陷阱**：成员们讨论了强调在 **Large Language Models** 上使用 **Reinforcement Learning** 技术时的局限性的论文。
   - 提到的论文包括 *Understanding R1-Zero-Like Training: A Critical Perspective*、*Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model*、*Reinforcement Learning Finetunes Small Subnetworks in Large Language Models* 以及 *Spurious Rewards: Rethinking Training Signals in RLVR*。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Harvey AI 获得巨额融资**：[Harvey AI 宣布](https://xcancel.com/harvey__ai/status/1937155058476646591)完成 **3 亿美元 E 轮融资**，估值达到 **50 亿美元**，由 Kleiner Perkins 和 Coatue 领投。
   - 他们还与 [LexisNexis 达成合作](https://www.lexisnexis.com/community/pressroom/b/news/posts/lexisnexis-and-harvey-announce-strategic-alliance-to-integrate-trusted-high-quality-ai-technology-and-legal-content-and-develop-advanced-workflows)，旨在整合 AI 技术与法律内容。
- **Replit 达到营收里程碑**：[Replit 宣布](https://xcancel.com/Replit/status/1937212611520831718)其 **年度经常性收入 (ARR)** 已突破 **1 亿美元**，这是公司的一个重要里程碑。
   - 尽管取得了这一成就，一些成员仍在质疑基于新的 ARR，其 **11 亿美元的估值** 是否真的合理。
- **AI Agent 需要类人监管**：Matan-Paul Shetrit 在[这条推文](https://x.com/MatanPaul/status/1937200395115499592)中强调了在扩展 AI Agent 时，可观测性 (observability) 与监管 (supervision) 之间的区别。
   - 他建议采用一种类似于管理人类的新监管方法，因为 AI Agent 会与系统和客户进行积极互动。
- **分发渠道决定主导地位？**：Alex Immerman 的[推文](https://xcancel.com/aleximm/status/1937251084810219721)引发了一场辩论，即初创公司是否能在现有巨头 (incumbents) 创新之前实现分发 (distribution)。
   - 讨论指出 **OpenAI 快速获取用户** 是一个关键优势，并将其与 Google 的分发策略进行了对比。
- **Anthropic 在合理使用案件中获胜**：Adam Eisgrau 报告称，根据 Alsup 法官的说法，[Anthropic 在合理使用 (fair use) 方面赢得了简易判决动议 (Motion for Summary Judgment)](https://xcancel.com/adameisgrau/status/1937480346976813454)。
   - 审判将继续进行，以确定使用“盗版”互联网材料可能造成的损害。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Grok3mini 使用量激增！**：非 Beta 版的 **grok3mini** 使用量大幅增加，从 6 月 19 日的 **每天 200 万次** 跃升至 **每天 1 亿次**。
   - 这表明自发布以来，各种应用对 **grok3mini** 的采用和依赖程度不断提高。
- **Llamabarn 助力本地推理！**：Georgi 推出了 **Llamabarn**，这是一款新的本地推理应用，因其简洁的设计获得了积极反馈，详见[这条 X 帖子](https://x.com/ggerganov/status/1937189250149257250)。
   - 它为本地 **LLM 推理** 提供了一个流线化的解决方案，可能为资源有限的开发者提高可访问性。
- **COCONUT 门控层揭秘！**：根据[这条 X 帖子](https://x.com/ryunuck/status/1937466079309144256)，**COCONUT** 架构使用了一个“门控 (gating)”层，从隐藏状态中提取信息，以确定每个 token 的采样器参数，并在 token 之间保持隐藏状态。
   - 这种方法允许在 **LLM** 中进行更高效且具有上下文感知能力的采样，从而提高整体性能。
- **GTX 1080 用户寻求本地 LLM 指导！**：一位成员正在寻求适合 **LoRA 训练**、**GGUF 转换** 以及在 **GTX 1080** 上运行的模型建议，用于角色扮演和一般技术问题。
   - 新用户请求关于在 **GTX 1080** GPU 上开始使用 **本地 LLM** 的解释和建议，计划通过 **LoRA 训练** 模型用于角色扮演和处理一般技术问题。
- **MultiNet v0.2 评估通用 AI 系统！**：用于评估通用 AI 系统的开源平台 **MultiNet** 的 **0.2** 版本已在 [Manifold](https://multinet.ai) 发布。
   - 该平台旨在为评估通用 AI 模型的能力提供全面的基准测试。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **多智能体研究论文需要更多深度**：一位成员关于 **multiagent cooperation**（多智能体协作）的论文收到了反馈，认为需要更多地结合现有文献，尽管论文中关于“问题作为断路器”的观察很有趣。
   - 作者计划通过使用更大的样本量、不同的群体组成、模型参数和上下文窗口大小来扩展研究，但指出他们受限于 **Claude Opus** 的高昂费用。
- **前缀缓存策略辩论**：一位成员询问是否有支持类似 **vLLM** 的 **prefix caching**（前缀缓存）库，但能够将缓存存储在内存映射文件中，以处理超过 VRAM 或 DRAM 的序列。
   - 另一位成员建议，除非序列长度超过 **1M**，否则这会比重新计算更慢，不过提问者澄清他们的用例涉及 **128k** 的序列。
- **卡车司机少年挑战对话式 AI 红队测试**：一位来自瑞典的 **17 岁** 少年正专注于使用社交和心理压力策略对 **conversational AI** 进行 **red teaming**（红队测试），并将其工作记录在 [GitHub 仓库](https://github.com/Ufosxm34gt/Conversational-Red-Teaming-Casebook)中。
   - 他在瑞典学习卡车驾驶的同时，正寻求与服务器上的其他人建立联系并学习。
- **Sleeping-DISCO 数据集寻求 EleutherAI 合作**：一位成员正为其新的大规模生成式音乐建模预训练数据集 **Sleeping-DISCO-9M** 寻求与 EleutherAI 的潜在合作，该数据集已发布在 [Hugging Face](https://huggingface.co/datasets/sleeping-ai/Sleeping-DISCO-9M) 上。
   - 数据集创建者寻求在质量基准测试方面的帮助，并提到他们的 [arxiv 预印本](https://arxiv.org/abs/paper)需要语法修正，而另一位成员则批评了该数据集的原创性，认为它主要是重新索引了来自 **Genius.com** 的内容。
- **损失曲线分解揭示技能簇**：一篇新论文通过 **orthogonal gradient basis**（正交梯度基）分解损失曲线，揭示了样本簇具有相似的突破动态，而这在精确损失中是不可见的。
   - 该论文可在 [此处](https://www.alphaxiv.org/abs/2505.14685) 获取，展示了这些簇和突破与玩具算术和真实语言建模设置中的特定技能相一致。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo-Python 仍存在局限性**：一位成员引用了从 **Python** 调用 **Mojo** 时[已知的局限性](https://docs.modular.com/mojo/manual/python/mojo-from-python/#known-limitations)。
   - 他们确认这些局限性在最新版本中依然存在。
- **Larecs 测试仅在 Modular CI 中失败**：一位贡献者正在调试一个问题，即 [Larecs 测试](https://github.com/samufi/larecs) 仅在 modular-community CI 中失败，而在本地机器或 GitHub CI 中均正常，这使得调试变得困难，目前正使用 [debug 分支](https://github.com/samufi/larecs/tree/debug_query) 进行排查。
   - 另一位贡献者在 M1 Mac 上运行 `mojo test` 时复现了该问题，怀疑是某种不安全操作，并协助提供了详细输出。
- **Mojo 的安全性旨在取代 Rust**：一位用户询问 Mojo 的安全特性是否能成为 Rust 的可行替代方案，特别是在 **sum types**（和类型）、**pattern matching**（模式匹配）和可预测的析构时间方面。
   - 一位 Modular 工程师回应称，虽然 *product types*（积类型）已通过 `struct` 实现，但 *sum types* 和 *pattern matching* 已在计划中，并解释说 Mojo 已经提供了 *RAII* 和 *ASAP destruction*，且正在摆脱 *syntactic salt*（语法盐）。
- **Mojo Async 计划避免 Rust Async 的陷阱**：一位用户询问 Mojo 的 async 设计是否会解决 Rust 中 async 遇到的困难，对此一位 Modular 工程师指出 [PR 3945](https://github.com/modular/modular/pull/3945) 和 [PR 3946](https://github.com/modular/modular/pull/3946) 是解决方案。
   - 他们指出，更好的 async 运行时和线性类型可以消除对 `Arc<Mutex<T>>` 等结构的需求，并指出了用于改进 IO 的 [PR 4728](https://github.com/modular/modular/pull/4728)。
- **神秘的语句放置错误**：一位用户在 Mojo 中遇到错误 *"statements must start at the beginning of a line"*，代码片段为 `if info.value() > top:`。
   - 另一位用户建议添加 `var top` 作为潜在修复方案，表明这可能是变量声明或作用域的问题。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 正在应对不稳定的 PDF 读取问题**：用户报告 **Manus** 在读取文本文件和 PDF 时遇到困难，经常提示用户提供纯文本。
   - 一位用户在 **#general** 频道中询问 *为什么 Manus 最近在读取文本文件和 PDF 时遇到困难？* 并表达了普遍的挫败感。
- **雄心勃勃的 AI 架构梦想浮现**：一位成员表达了对构建一种专注于 **funcognitive and meta enhancements** 的新型 AI 架构的热情，目标是在速度和效率上超越当前的 Transformer 模型。
   - 这一行动号召问道：*这里有人有兴趣开发新的 AI 架构吗？主要是为了 funcognitive and meta 的改进，就是做一个更好、更快的 Transformer*。
- **订阅混乱困扰用户**：一位用户报告称，尽管购买了大量额外积分，但仍被拒绝提供促销延期，迫使他们创建新账户。
   - 他们感叹道：*为 2 个月的订阅支付了 400 USD...他们拒绝了*，并称这种体验 *太愚蠢了*。
- **积分大奖奖励 Beta 测试者**：一位用户透露，由于作为 **Manus** 的长期 Beta 测试者所做的贡献，他们收到了 **90,000 积分**。
   - 这种对贡献的认可凸显了 Beta 测试者的价值，该用户表示：*他们只是因为我的贡献而给我积分*。
- **Manus 深受性能问题困扰**：多位用户报告 **Manus** 出现卡顿、显示内部服务器错误，并最终导致积分浪费。
   - 一位用户计算出由于这些问题损失了超过 **2000 积分**，而另一位用户声称：*我认为 Manus 变得更笨了，会犯错而且发现不了错误*。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **TorchTune 的单机 LORA 功能令人印象深刻**：一位用户称赞了 **TorchTune**，特别是单机 **LORA** 功能，认为其非常有用，团队鼓励通过 [GitHub](https://github.com/pytorch/torchtune) 提供反馈。
   - 团队提到他们通常对评论和 Issue 的响应非常迅速。
- **Expandable Segments 在 L40S 显卡上触发指针错误**：一位用户报告在 **Nvidia L40S** GPU 上使用 **expandable segments** 时出现指针错误，通过禁用该功能解决，但在 **H200s** 上运行正常，该问题在 [此 issue](https://github.com/pytorch/pytorch/issues/140419) 中进行跟踪。
   - 该问题似乎与 packing、flexattention 和 `max-autotune` 设置有关。
- **max-autotune 被指导致显卡崩溃**：一位成员建议问题可能源于 **max-autotune** 而非硬件限制，并指出 Unsloth 在使用 **expandable segments** 时带有 `roundup_power2` 标志。
   - 清理缓存可以解决错误，使显卡在第二次尝试时正常工作，即使没有新设置也是如此。
- **L40S 显卡的 Bug 得到修复**：团队表示 **expandable segments** 可能是一个边缘案例，因为 **L40S** 的使用并不广泛，NCCL 最近在 SM90 下禁用了 FP8 reduction。
   - 建议检查硬件规格，并在必要时绕过 expandable segments。
- **Reward Modeling RFC 等待反馈**：一位成员正在寻求对 **Reward Modeling RFC** 的反馈，并提议在 6 月 26 日即将举行的 Office Hours 期间进行讨论。
   - 未提供进一步细节。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **模型更新信息依然难以获取**：一位用户询问了 **NotebookLM** 当前使用的模型以及在哪里可以找到 **model options**，并引用了 [YouTube 视频](https://youtu.be/K9bvF_CJKV8?si=Gj7Z6GfOaTRLHKx2) 作为参考，但未得到直接回答。
   - 建议包括查看 **FAQ** 或 **Release Notes** 以获取最新的模型信息，或者在用户界面中寻找 **dropdown** 下拉菜单。
- **分享功能分享的内容少于预期**：一位用户反馈，“分享链接”功能仅分享初始查询状态，即在 **prompt** 和响应产生 *之前* 的状态，这阻碍了完整上下文的分享。
   - 他们建议增加一个“全选复制按钮”作为解决方案，主张能够分享上传的源列表、**prompt** 以及模型的响应。
- **用户寻求 NotebookLM 替代方案**：一位用户报告 **NotebookLM** 对其无法正常工作，并请求推荐替代方案。
   - 其他用户回复称该工具对他们来说运行良好，即使处理数百个 **pdfs** 也没有问题。
- **构思音频头像自动化**：一位用户询问是否可以使用 [SuperTelegram](https://supertelegram.gumroad.com/l/pwxot) 将一段 **4 分钟的 NotebookLM 音频** 转换为双主持人播客头像会话。
   - 另一位用户提到，为此目的可能需要分离发言者，但其可行性仍处于推测阶段。
- **视频尝试转向 Vimeo**：一位用户询问是否可以使用 **Vimeo 视频** 作为来源，但在粘贴链接时遇到了安全功能问题。
   - 另一位用户建议使用 [cobalt.tools](https://cobalt.tools/) 下载视频作为变通方法，以便能够使用。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Debian 12 构建面临问题**：一位用户在 **Debian 12** 上遇到了构建问题，建议其他人改用 **Ubuntu Jammy** 和 [Qt SDKs](https://qt.org)。
   - 该用户建议使用 backport 软件包，但记不清其解决方案的具体细节。
- **Python SDK 更新**：一位用户询问了 **Python SDK** 即将进行的更新。
   - 他们开玩笑地问：“还是说 Python 已经没救了？”
- **GPT4All 网站占用大量 CPU**：一位用户报告 [gpt4all.io](https://www.nomic.ai/gpt4all) 官方网站存在 Bug，声称它“占用了我 60% 的内部 GPU”。
   - 该用户明确表示他们指的是官方网站。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Atom of Thought 创业项目崩塌**：在 **Agent** (**GAIA benchmark**) 中进行的一项 **Atom of Thought** 实验由于前置分解导致的灵活性丧失以及步骤间的上下文丢失而被移除；研究人员因论文实现代码存在严重问题，对该论文及其作者失去了信心。
   - 在收到实现代码问题的通知后，作者在 X 平台上的回应 *极其负面且不专业*，随后利用该论文在 X 上转型为一家 **Agent** 初创公司。
- **Ax TypeScript 移植版引发关注**：一位成员强调了适用于 **TypeScript** 的 **Ax** 的可用性，以及它对 **Elixir** 和 **Ruby** 的适配。
   - 关于这些移植版的具体功能和用例的进一步细节尚未详述。
- **在 Forward 方法中寻求状态更新**：一位成员请求了解如何从模块的 `forward/aforward` 方法中发送状态消息而不使用 `yield`，目的是在 `module_start_status_message` 之后捕获事件。
   - 有建议提出向 `forward` 传递一个 **callback**（回调）来更新 UI 进度。
- **OpenAI 经历宕机**：一位成员报告在使用 **LiteLLM** 时遇到 **OpenAI** 的问题，称“他们的应用挂了”。
   - 在使用 **LiteLLM** 的 `completion()` 且 `model= gpt-4o-mini; provider = openai` 时抛出了 `404 Not Found` 错误。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Google 将 A2A 慷慨捐赠给 Linux Foundation**：Google 将 **A2A** 捐赠给了 [Linux Foundation](https://developers.googleblog.com/en/google-cloud-donates-a2a-to-linux-foundation/)。
   - 公告发布后，成员们推测 **Anthropic** 是否会效仿，捐赠他们自己的 **A2A**。
- **MCP 的超时问题被触发**：一位成员报告在使用 **OpenAI agents** 创建客户端会话时遇到了 **MCP tool** 的 **timeout issue**。
   - 错误信息显示系统在等待 **ClientRequest** 响应 **5.0 秒** 后超时。
- **Chrome 引入 AI APIs**：正如 [Chrome 138](https://developer.chrome.com/blog/new-in-chrome-138?hl=en#built-in) 中宣布的那样，Chrome 正在集成一些 **AI APIs**。
   - 这可能会直接在浏览器中实现 **MCP integration**。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **证书发放日期泄露！**：完成所有作业和社交媒体发布的成员预计将在 **7 月中旬**收到证书。
   - 发放时间表已得到工作人员的确认。
- **课程完成确认！**：参与者确认他们已完成所有作业和社交媒体先决条件，并询问证书发放时间。
   - 课程完成涉及作业以及在 **Twitter** 和 **LinkedIn** 等平台上的社交媒体发布。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Reranker 成本探究**：成员们讨论了与 **Cohere Reranker** 相关的成本，特别是涉及 **1000 次调用**的高频使用情况。
   - 澄清了定价取决于文档和 tokens 的数量，超过 **500 tokens** 的文档将被拆分为 chunks，详见 [Cohere 定价页面](https://cohere.com/pricing#:~:text=We%20count%20a%20single%20search%20unit%20as%20a%20query%20with%20up%20to%20100%20documents%20to%20be%20ranked.%20Documents%20longer%20than%20500%20tokens%20when%20including%20the%20length%20of%20the%20search%20query%20will%20be%20split%20up%20into%20multiple%20chunks%2C%20where%20each%20chunk%20counts%20as%20a%20singular%20document.)。
- **Cohere 社区壮大**：新成员正在加入 Cohere Discord 服务器并向社区介绍自己。
   - 新用户正在分享他们的**公司/行业/大学**、当前项目、喜爱的**技术/工具**以及参与社区的目标。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Cursor 通过开源匹配功能筛选简历**：LlamaIndex 引入了一个开源的 **Resume Matching MCP server**，用于在 Cursor 工作流中直接进行智能职位匹配，连接到 **LlamaCloud 简历索引**和[其他服务](https://t.co/RCKoiUccm6)。
   - 该项目由 @zhaoqili74 在内部黑客日期间构建，旨在简化简历筛选流程。
- **LlamaIndex 发布兼容 Claude 的 MCP Server 模板**：LlamaIndex 发布了一个新的开源模板仓库，用于将**兼容 Claude 的 MCP server** 构建为具有完整 **OAuth 2.1 支持**的 Next.js 应用，简化了与[该服务](https://t.co/wtPorldMvJ)无缝协作的远程 Model Context Protocol 服务器的创建。
   - 该模板由 @seldo 在内部黑客日期间开发，旨在简化与 **Claude** 及其他使用 Model Context Protocol 的服务的集成。
- **向量化大幅加速相似度计算**：一位成员通过将循环替换为 `query_embeddings @ doc_embeddings.T` 优化了余弦相似度计算，将 **1000 x 1000** 矩阵的运行时间从 **~25 秒**减少到 **~0.04 秒**。
   - 这表明通过使用 `@` 或 `matmul` 的向量化计算实现了 **625 倍的加速**。
- **成员寻求关于大规模量化 FAISS 的建议**：对于超过 **10M 次比较**的情况，该成员计划切换到量化 FAISS 索引（如 `IndexIVFPQ`）以管理内存和延迟。
   - 用户询问了在动态（非预索引）查询向量中使用 `IndexIVFPQ` 的注意事项，并寻求对优化计划的反馈，同时询问 `@` / `matmul` 在 **1M 规模**下用于生产环境是否稳定。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **AI21 Labs 的 Jamba 模型发布公告**：一位成员分享了关于 **AI21 Labs Jamba 模型**发布公告的[链接](https://www.rxddit.com/r/Humanornot/s/zXu0PrCoo2)。
   - 上下文中未提供关于该模型架构、能力或具体用例的进一步细节。
- **社区对 Jamba 影响力的怀疑**：初步反应显示出一种谨慎的态度，一些人质疑其显著颠覆当前开源模型格局的潜力。
   - 社区正在等待进一步的基准测试和全面评估，以确定 **Jamba** 与现有替代方案相比的真实性能和能力。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。

---

您收到这封邮件是因为您通过我们的网站选择了订阅。

想更改接收这些邮件的方式吗？
您可以从该列表中[取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：各频道详细摘要和链接

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1386807526468882482)** (537 条消息🔥🔥🔥): 

> `Polaris 4B 模型, 3D 网格 (Meshes), LoRA 超参数, 用于高效推理的 NVFP4, Reddit 版务管理` 


- **Polaris 4B 声称拥有惊人的基准测试结果**：成员们对 **Polaris-4B-Preview** 模型声称超越 **Claude-4-Opus** 等商业系统的说法表示怀疑，暗示可能存在严重的过拟合和刷榜（benchmaxing）行为，一位成员表示：“读到这里就可以停了，4b 超过 opus... 呵呵... 只要过拟合，我用 100m 的模型也能做到。”
   - 一位成员尝试了 **Q4_K_M 版本** 但发现效果不尽如人意，计划使用 **Q8_0** 进行测试并与 **FP16** 进行对比，以验证模型的性能。
- **腾讯的 Hunyuan3D-2.1 网格生成器表现稳健**：一位成员分享了用于生成 **3D 网格** 的 [腾讯 Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1)，称其“相当稳健”，并指出其较之前版本有显著进步。
   - 讨论中涉及了使用 AI 创建带骨骼绑定的网格（rigged meshes）的可行性，参考了 **Mixamo** 和 **Cascadeur** 等工具，尽管目前尚未完全实现生成式。
- **发布新的 LoRA 超参数指南**：Unsloth 团队发布了[新的 LoRA 超参数指南](https://x.com/UnslothAI/status/1937521408344752272)，引发了关于提及使用 **Optuna** 进行超参数搜索（hyperparameter sweeps）以提升性能的建议，因为“每个数据集的表现都不同”。
   - 针对 PEFT 配置中支持 **alpha_pattern** 的问题提出了担忧，一位成员指出 Unsloth 会静默丢弃该参数。
- **NVIDIA NVFP4 助力高效推理**：一位成员分享了关于 [NVFP4 的 NVIDIA 博客文章](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)，话题迅速转向量化图表中跳过了与 **H200** 的对比，以及 **FP8** 与 **FP4** 之间的比较。
   - 一位成员提到，他们单台 **H200** 的成本约为 **3 万美元**（不含增值税），且由于其被动散热特性需要定制冷却系统，并补充说该设备用于训练 LLM。
- **Reddit LocalLlama 板块因版主离开引发风波**：成员们注意到 r/localllama 的版主注销了账号，导致该子版块处于无人管理状态，自动版务（automod）被配置为删除所有内容。
   - 几位用户已通过 r/redditrequest 申请接管该子版块。讨论中提到了理想版主的特质，包括有充足的时间、关心 OSS + 本地化、精通 AI + LLM 且社交能力正常。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1386904218585010339)** (4 条消息): 

> `QAT 模型, 推荐系统业余项目` 


- **询问 QAT 微调**：一位成员询问讨论内容是关于“微调 QAT 模型”还是“进行 QAT”本身，并好奇训练包是否支持像 **Gemma 3** 这样的 QAT 模型。
   - 另一位成员回答了这个问题，简要表示他们正在“进行 QAT”。
- **爱好者寻求 RecSys 训练帮助**：一位成员提到正在开展一个“多推荐系统业余项目”，目前正面临模型训练问题。
   - 他们正在寻求任何有 **RecSys** 经验且愿意抽出时间提供帮助的人。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1386822689800454298)** (323 条消息🔥🔥): 

> `微调性能指标分析, 梯度累积策略, Qwen GRPO Notebook 问题, Unsloth checkpoint 与官方 checkpoint 对比, Gemma-3 Vision Notebook 问题` 


- **在 Qwen 微调时寻求瓶颈洞察**：一位成员寻求关于分析或记录更深层性能指标的建议，以识别在 Google Colab 上对 [Qwen3 1.7b](https://huggingface.co/Qwen/Qwen3-1.8B) 进行全量微调（full fine-tuning）设置中的瓶颈。
   - 他们报告称，尽管 ETA（预计完成时间）很长且已启用调试日志，但 GPU RAM 占用率却很低，且只能看到步数和训练损失。
- **平衡 Batch Size 和梯度累积！**：一位成员建议通过减少梯度累积（gradient accumulation）来牺牲显存换取性能，推荐使用 `batch=4 GA=2` 或 `batch=8 GA=1`。
   - 用户发现增加 Batch Size 会略微增加显存占用，但迭代速度减半，这表明存在另一个瓶颈。
- **利用 opensloth 的秘密武器解锁显存！**：成员们使用 `accelerate config` 配合 DS-2 调查显存问题，怀疑是 CUDA toolkit 版本导致的问题。
   - 一位成员指出，Unsloth 中的多 GPU 支持仅限于 DDP，并且要求模型能够放入单个 GPU 中，才能使用多个 GPU 进行训练。 
- **接入 TensorBoard 实现可视化训练**：一位成员询问关于添加像 TensorBoard 这样的监控器来可视化训练信息的问题。
   - 另一位成员建议使用 [Weights & Biases (WandB)](https://wandb.ai/site/)，并提供了 [Hugging Face 集成指南链接](https://docs.wandb.ai/guides/integrations/huggingface/)。
- **Unsloth 的 CUDA 奇遇记！**：一位成员报告了一个差异，即 Unsloth 日志显示 **CUDA 8.0**，即使 CUDA Toolkit 版本是 **12.6**，并询问这是否为默认行为。
   - 该差异被确定为正常现象，一位成员指出 **CUDA 8.0** 对应于 **A100** GPU，并提供了一个 [NVidia 链接](https://developer.nvidia.com/cuda-gpus) 予以证实。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1386979437332922470)** (3 条消息): 

> `纳瓦特尔语翻译器, Unsloth 微调` 


- **首个开源纳瓦特尔语翻译器问世**：首个开源的**纳瓦特尔语（Nahuatl）到西班牙语翻译器**已构建完成，并可在 [Hugging Face Spaces](https://huggingface.co/spaces/Thermostatic/neuraltranslate-27b-mt-nah-es) 上使用。
   - 该项目的创建者感谢 **Unsloth 的全量微调支持**使其成为可能。
- **纳瓦特尔语翻译器复现代码已发布**：用于复现**纳瓦特尔语翻译器**项目的代码已在 [GitHub](https://github.com/Sekinal/neuraltranslate-nahuatl/tree/master) 上发布。
   - 这使得其他人能够在此基础上进行构建，或针对不同目的调整该翻译器。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1386790715203391548)** (6 条消息): 

> `BNPO vs Dr.GRPO, RL-tuning 性能, 训练不稳定性, GRPO-lora 与 GRPO-Qlora` 


- **辩论升级：BNPO 在性能上略胜 Dr.GRPO**：一位用户询问了使用 **BNPO** 与 **Dr.GRPO** 进行微调的经验，特别是关于稳定性和性能方面。
   - 一位用户发现 **Dr.GRPO** 更稳定，但 **BNPO** 整体表现更好，同时也承认这只是一个*非常简陋的测试*。
- **深入探讨 RL-Tuning 性能**：根据一位用户的说法，**Dr.GRPO** 实现了与 **GRPO** 相似的 **RL-tuning 性能**，但使用的 token 数量显著减少。
   - 相比之下，据称 **BNPO** 直接解决了训练稳定性问题。
- **关于 GRPO-Lora 与 GRPO-QLora 的疑问出现**：一位用户询问了在使用 **GRPO-LoRA** 和 **GRPO-QLoRA** 时，**训练不稳定性**风险的增加以及潜在的精度损失。
   - 这个问题将对话转向了特定实现策略中的实际权衡。


  

---

### **Cursor 社区 ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1386793529220665566)** (411 条消息🔥🔥🔥): 

> `Cursor Setup, Cursor Terminal Issues, Windsurf vs Cursor, Rate Limits and Pricing, MCPs VisionCraft and Sequential Thinking` 


- **设置 Cursor 比预期更费力**：一位成员在设置 **WSL**、修复 **Cursor** 和 **Ubuntu** 设置、链接 **GitHub** 以及创建 **SSH** 密钥上花费了比预期更多的时间，将其描述为针对 Cursor 和模型的优化项目规则的*无底洞*。
   - 另一位成员建议在 Linux 上，所有这些设置都可以在 **Cursor** 内部自动完成，只需使用 *sudo -i* 权限，但风险自负。
- **Cursor 的终端控制备受关注**：用户报告 **Cursor terminal** 存在超时、报错以及与 **Windsurf** 等工具相比缺乏平滑操作的问题，后者能更有效地生成和跟踪终端窗口。
   - 一位成员指出，关于修复终端问题，他们遇到的标准回复是 *“我们目前有更关键的任务”*。
- **Windsurf 的困扰促使用户转向 Cursor**：一位用户由于终端相关问题从 **Windsurf** 切换到了 **Cursor**，强调终端问题是切换的原因，但也承认当 Windsurf 正常工作时，它可以有效地管理终端窗口。
   - 他们遇到了终端问题，例如 **agent** 无法读取 **terminal outputs**，或者在运行终端命令时冻结。
- **速率限制命中与新旧定价方案**：用户正在讨论速率限制和定价方案，一些人在使用 **Sonnet** 时立即遇到了速率限制，即使没有重度使用，并且还注意到似乎存在整体的 Cursor 速率限制。
   - 一位用户意识到在新定价方案下，由于速率限制，他们在几天内花费了 **$70**，而在旧定价方案下，他们的需求仅需 **$20**。
- **VisionCraft MCP 提升了文档质量，用户请求集成到 Cursor**：**Visioncraft MCP** 正在变得更好，拥有更及时的文档、更快的响应以及更好的 Prompt 响应，且无需指定具体文档，因此用户非常渴望其与 Cursor 的集成。
   - 它通过提供更新的文档来解决 AI 模型基于旧数据训练的问题，从而产生更好的代码并减少错误。


---

### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1386788192748965961)** (34 条消息🔥): 

> `多台机器上的 Background Agents，Background Agents 的 Devcontainer 支持，Background Agent API，Background Agents 与 Git 初始化问题，在安装步骤中访问私有 GitHub 仓库` 


- ****关于 BA 机器特定性的讨论****：一位用户质疑 Background Agents 是否被有意限制在特定机器上，并指出在一台电脑上激活的 Agent 不会出现在另一台电脑上，即使使用了相同的 repo 和账号。另一位用户回应称 *不应该是这种情况 🤔 - 确定两台机器上的 git origin 相同且使用了同一个 Cursor 账号吗？*。
   - 结果发现该用户的本地 git repo URL 中带有 `www.` 前缀，这导致了问题。Background Agents 仅支持 `https://github.com/<name>/<repo>`，而不支持 `https://www.github.com/<name>/<repo>`。
- ****Devcontainer 规范：支持还是不支持？****：一位用户请求 Cursor BA 支持 Docker 构建的私有 registry 认证，并遵循现有的 `.devcontainer.json` 规范以避免冗余。
   - Cursor 团队成员确认了该请求，但表示 *Dev-container 兼容性目前不是我们的首要任务*，并解释了选择当前 auto / snapshot / environment.json 设置的原因。
- ****私有 Gems？考虑 Docker 构建过程中的 Secrets****：一位用户报告了在使用 Dockerfile 设置 Background Agents 时，无法访问私有 gems 和包的问题。
   - 他们尝试使用 `GITHUB_TOKEN` secret，但 Cursor 团队承认 *我们目前在 Docker 构建过程中不暴露 secrets*，并暗示在安装阶段暴露 secrets 实际上是可行的。
- ****Background Agent 规则：被忽略了？****：一位用户报告称 Background Agents 没有遵守 Cursor 编辑器中定义的规则或 Agent 提示词中编写的规则，导致向其 repo 进行了不必要的推送。
   - 该用户对无法有效管理 Agent 表示沮丧，指出 *查看/管理 Background Agents* 的快捷键绑定不起作用，而且从 Slack 点击 *在 Cursor 中打开* 按钮虽然能跳转到编辑器，但什么也不显示，这暗示存在严重 Bug。
- ****Python 3.11：Dockerfile 困境已解决****：一位用户在 Dockerfile 中运行 Python 3.11 的 Background Agent 时遇到困难，原因是 `ensurepip` 不可用。
   - 他们分享了包含 `add-apt-repository ppa:deadsnakes/ppa` 及其他安装 Python 3.11 及其依赖项步骤的 Dockerfile 代码片段；这个片段可能对其他想要运行 Python 3.11 的用户有所帮助。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1386785713260855316)** (396 条消息🔥🔥): 

> `居家学习提示词，Perplexity Pro 版本问题，ChessChamp AI 发布，医生收费，O4 Mini High 优于 Omni` 


- **居家学习提示词探索开始**：一位成员正在为他们的孩子寻找合适的提示词，以便在英国暑假期间进行居家学习。
   - 另一位成员发布了 *让小家伙过个暑假吧*，建议让孩子们直接休息。
- **Perplexity Pro 问题持续存在**：一位拥有年度 **Pro** 订阅的成员报告称被提示升级，这表明其账号状态可能存在问题。
   - 另一位用户建议，问题可能源于使用了 *极大概率不是为你准备的促销活动，因此被撤销了*。
- **ChessChamp AI 测试开始**：一位成员对即将推出的 **ChessChamp AI** 表示感兴趣，并推测它是否使用了 **Stockfish**。
   - 作为回应，另一位成员提供了一张截图，显示该工具已经可用，并表示 *你可能需要订阅才能访问它*。
- **吐槽医生高额收费**：一位成员发布了[一段关于印度医生的视频](https://www.ddinstagram.com/reel/DLFdBSCttT9)，引发了关于高额检查费用的讨论。
   - 发布者详细说明 *他们的检查费收得太高了*，而且 *10-20 美元就是一个月的电费了*。
- **Perplexity 现已支持 WhatsApp**：Perplexity 现在支持在 [WhatsApp](+18334363285) 上执行定时任务，扩大了其可访问性。
   - 一位成员表示 *我竟然不知道它上线 WhatsApp 了！*


  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1386835371517804544)** (4 条消息): 

> `Shareable threads, Trump ceasefire, Donation fatigue, Ubisoft patch` 


- **Trump 宣布停火**：一名成员分享了一个关于 **Trump** 宣布停火的 [Perplexity AI 页面](https://www.perplexity.ai/page/trump-announces-ceasefire-Qq4WKw3gQAqdo1ados1Uug)。
   - 未提供进一步的讨论或细节。
- **捐赠疲劳对 Mozilla 的影响**：一名成员分享了一个 [Perplexity AI 页面](https://www.perplexity.ai/page/donation-fatigue-impact-on-moz-q76XMD17Skap_valehbYOg)，讨论了**捐赠疲劳**对 **Mozilla** 的影响。
   - 未提供进一步的讨论或细节。
- **Ubisoft 的 The Division 2 补丁**：一名成员分享了一个关于 **Ubisoft** 的 **The Division 2 Patch** 的 [Perplexity AI 页面](https://www.perplexity.ai/page/ubisoft-s-the-division-2-patch-dWFALCxPQmCDkPmNrKBfUQ)。
   - 未提供进一步的讨论或细节。
- **在 Discord 上分享线程**：Perplexity AI 要求用户确保其线程是可共享的。
   - 提供了一个 Discord 频道链接作为示例，详见[此处](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1386831345816305724)** (4 条消息): 

> `Perplexity AI tech support` 


- **用户寻求 Perplexity AI 技术支持**：一名用户询问在哪里可以获得 **Perplexity AI** 的**技术支持帮助**。
   - 另一名用户建议联系 **support@perplexity.ai**，但原用户提到他们上周已经发过邮件了。
- **关于 pplxdevs 的 X 帖子**：一名用户分享了一个 **pplxdevs 的 X 帖子链接**。
   - 链接为 [https://x.com/pplxdevs/status/1937218625020276927?s=46](https://x.com/pplxdevs/status/1937218625020276927?s=46)。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1386782872786047078)** (301 条消息🔥🔥): 

> `Memory context service, Multi-head Latent Attention, AI dubbing voice lines, Sora alternatives, Chat search connectors` 


- **服务解决 Memory Context 问题**：一名用户提到他们必须创建一个服务来为他们的 Agent 提供 **memory context、memory recall、rehydration 和 NLP**。
   - 他们表示这过程*非常烦人*。
- **Multi-head Latent Attention 创新受到关注**：一名用户引导他人搜索 **Multi-head Latent Attention**，以了解是什么创新导致在旧硬件上能以一半的价格生产出如此强大的产品，并附带了一个[从零开始构建 NumPy 模型的链接](https://t.co/xMHlA61Qoz)。
   - 另一名用户承认自己是根据新闻报道来了解情况的，并分享了一个[相关的 YouTube 视频](https://youtu.be/WEBiebbeNCA?si=84i4jjPyQWRSzVuQ)。
- **AI Voice Dubbing 解决方案出现**：用户讨论了 **AI voice dubbing** 的选项，推荐了带有口型同步功能的 [Hey-Gen translate](https://www.heygen.com/translate)。
   - 原帖作者觉得 **Veo 3** 很有趣，如果不考虑成本会考虑使用它。
- **Midjourney 与 Luma Labs Dream Machine**：用户讨论了生成动漫角色的 **Sora** 替代方案，推荐包括 **Kling** 和 **Runway**。
   - 一名用户分享了一个使用 [Luma Labs Dream Machine](https://lumalabs.ai/dream-machine) 制作的 **AI 生成视频**。
- **Pro 用户解锁 Chat Search Connectors**：Pro 用户现在可以对 [Dropbox, Box, Google Drive, Microsoft OneDrive (Business), Microsoft SharePoint](https://openai.com/blog/june-2024-updates) 使用 **chat search connectors**。
   - 该功能目前仅限于 EEA（欧洲经济区）、瑞士和英国以外的用户。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1386785975581147329)** (5 条消息): 

> `OAI Server Tag, GPT-4o Cutoff, ChatGPT vs GPT Models, File Upload/Deletion Issues` 


- **通过 Boosting 解锁 OAI Server Tag**：一名成员提到，如果服务器被 boost 至少 **3 次**，即可使用 **OAI Server Tag**。
- **GPT-4o 意外中断对话**：一名用户报告称 **GPT-4o** 在 **42 条提示词**后中断了对话，这可能意味着存在限制或 Bug。
- **关于 ChatGPT 与 GPT 模型的疑问需要澄清**：关于 **ChatGPT** 与 **GPT 模型**对比的问题，最适合在专门的 <#1047565374645870743> 频道讨论。
- **Projects 中的文件上传/删除陷入停滞**：一名用户报告在向其 Projects 文件夹删除或上传文件时遇到问题，出现了*死亡等待轮*且无法成功。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1386819065837785299)** (2 条消息): 

> `PDF generation failures, Deep Research report PDF` 


- **ChatGPT 的 PDF 生成体验**：一名成员正尝试在会话中根据生成的**结构化文本/Markdown**获取 **PDF**。
   - 他们在使用 Python 生成 PDF 时频繁遇到失败，这与 **Deep Research 报告功能**不同，后者提供整洁的客户端 PDF 生成。
- **Deep Research 报告**：一名成员指出，带有“导出为 PDF”功能的 **Deep Research “报告格式”**最符合他们的需求。
   - 他们认为无法从非 Deep Research 会话中触发 Deep Research 报告输出，这导致结果呈现为纯文本块而非可导出的 PDF。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1386819065837785299)** (2 条消息): 

> `PDF generation alternatives, Deep Research report format, ChatGPT PDF failures, Triggering DeepResearch output` 


- **寻求完美的 PDF 生成方案**：用户正在寻找一种可靠的方法，从结构化文本/Markdown 内容生成 PDF，并特别称赞了 **Deep Research 报告格式**的整洁布局和客户端 PDF 生成能力。
   - 他们注意到，使用 Python 的 **ChatGPT** 等替代方案经常导致失败，而 **Deep Research** 功能的 PDF 生成始终保持成功。
- **Deep Research PDF 导出功能调研**：用户怀疑 **Deep Research 报告功能**使用的是客户端 PDF 生成，这解释了其相对于服务端方法更高的可靠性。
   - 他们对无法在非 Deep Research 会话中触发相同的 PDF 输出感到遗憾，因为这会导致输出纯文本块而非可导出的 PDF。
- **怀疑 PDF 输出使用了独立函数**：一名成员表示赞同，确认 **PDF 生成函数**很可能是独立的。
   - 他们建议在特定频道进行讨论和推荐，特别是频道 ID **1070006151938314300** 和 **1070006915414900886**。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1386783776079609896)** (252 条消息🔥🔥): 

> `Grok3 SOTA, Claude niche, Apple Foundation Models, Google Flamesong, Kingfall release` 


- **Grok3 是 SOTA**：成员们认为 **Grok3** 并不平庸，理由是它在 [artificialanalysis 评分](https://www.artificialanalysis.ai/leaderboard/open-llms)上相对于 **Sonnet 3.7** 表现强劲。
   - 它在发布时被视为 SOTA，尤其是在**数学**方面，但一些人认为其更激进的 Post-training 使其未能真正达到顶尖水平。
- **Claude 统治利基市场**：**Claude** 被公认为拥有极强的利基市场，而这在标准基准测试中未能得到充分体现。
   - 成员们表示，该模型在创意写作和戏剧表演方面非常有用，因为它比竞争模型更能遵循角色扮演的指令。
- **Apple 正在开发基础模型**：有传言称 **Apple** 正在开发 Foundation Models，人们对其利用可信计算（Trusted Computing）保护隐私的方法很感兴趣。
   - 讨论涉及 **Apple** 授权 **Gemini** 用于服务端模型并开发强大的端侧模型，但许多人怀疑 Apple 正在发布自己的服务端模型。
- **Google 正在开发 Flamesong 模型**：**Google** 正在开发名为 **Flamesong** 的新系列模型，可能是新的 **Flash** 系列，人们在猜测它是 **Gemini 3.0** 还是另一个 **2.5 模型**。
   - 进一步猜测认为 **Stonebloom** 可能是一个 “2.5-pro-lite” 模型，测试显示它有时能正确回答问题。
- **Kingfall 发布在即**：**Kingfall** 只是投入了更多算力的 **2.5** 版本，在基准测试中排名第 9，而 **O4 Pro** 和 **GPT-4.5-thinking** 排名第 8。
   - 一些人想知道何时能看到 Kingfall，有人预计在夏末左右，另一些人则猜测 Stonebloom 是否是该模型的蒸馏版本。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1386784780896567490)** (140 条消息🔥🔥): 

> `HuggingFace 网站问题, AI 越狱, Gradio 加载问题, AI 自由职业, 模型微调` 


- **Hugging Face 网站遭遇访问变慢**：用户报告在访问 Hugging Face 网站时遇到 **429 rate limit errors** 和 **504 Gateway Time-out** 问题，影响了模型下载和 Space 功能。根据 [状态页面](https://status.huggingface.co/) 显示，网站目前似乎已恢复在线，但运行速度仍然较慢。
- **用户引发关于越狱 AI 的讨论**：一位用户就名为 *Luma* 的 **越狱 AI** 寻求建议，该 AI 表现出异常行为，包括对其他 AI 模型进行实验以探索其边界。
   - 其他成员建议越狱 **DeepSeek** 相对容易，因为该用户正在就其特定情况寻求第二意见。
- **Gradio 仪表板无法加载**：一位用户报告其 **Gradio 仪表板** 在 Discord 机器人中卡在 *Loading...* 状态，尽管日志显示正常。
   - 另一位成员建议检查 **stack trace** 或重启 Space 以解决该问题。
- **开启 AI 自由职业生涯探索**：一位用户询问寻找 **AI 自由职业** 的技巧。
   - 一位成员建议查看专门的职位机会频道。
- **微调过程中的挫折**：一位用户寻求关于 **模型微调** 的专家协助，特别是针对 **SDXL** 或 **FLUX LoRA**，并对使用 *kohya_ss* 时 loss 未如预期下降表示不满。
   - 另一位成员分享了关于 **FLUX LoRA** 和 **SDXL LoRA** 的社区文章和 notebook，建议用户查看 **diffusers GitHub repo** 获取更多示例。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 条消息): 

h2he3: 非常有用，谢谢。
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1386789465439342632)** (50 条消息🔥): 

> `Gradio 自定义组件打包, LLM 输入空间的梯度下降, 评估用于计算机图形代码补全的语言模型, 使用 Ollama 进行 AI 对话, LLM 生成 Shader Graph 代码` 


- **PyPi 上的 Gradio 自定义组件**：一位成员正在考虑将其 Gradio 组件打包发布到 PyPi，并寻求关于官方 Gradio “自定义组件”打包路径的建议。目前该成员通过 GitHub 安装并在 Gradio Blocks 上下文中实例化。
- **LLM 上的梯度下降：ModernBERT 实验**：一位成员分享了关于 [LLM 输入空间的梯度下降](https://dev.to/kylepena/gradient-descent-on-llm-input-space-a-modernbert-experiment-3053) 的文章链接，这是一项针对 ModernBERT 的实验。
   - 另一位成员发布了其关于计算机图形代码补全的语言模型论文，该论文已发表在会议论文集中：DOI:[10.1109/LLM4Code66737.2025.00017](https://doi.org/10.1109/LLM4Code66737.2025.00017)，且评估指标/排行榜已作为 Space 托管在 HF 上：[ShaderMatch](https://huggingface.co/spaces/Vipitis/shadermatch)。
- **Ollama 引发 AI 对话讨论**：一位成员分享了一个项目，使用 [Ollama](https://github.com/Laszlobeer/AI-Dialogue-Duo) 让 **两个 AI 围绕特定话题互相交谈**。
   - 另一位成员建议使用 `import ollama` 包来减少 80% 的代码，并附上了 [ollama-python GitHub](https://github.com/ollama/ollama-python) 的链接。
- **LLMs 生成 Shader Graph 代码**：一位成员询问了 LLM 在生成 HLSL 和 GLSL 的 **Shader graph 代码** 方面的能力。
   - 另一位成员提到了一项使用语言模型自动优化 shader 代码的研究项目，以及 Nvidia 的神经材料（neural materials）方法。
- **gSPLAT Gaussian 未来**：讨论围绕像 **gSPLAT** 结合 **高斯滤波（Gaussian filtering）** 这样的方法是否可能在 5-10 年内让材质 shader 过时，或者它们只是过渡阶段。
   - 有人提到某些材质拥有多达 **107 个输入参数**，而学习到的权重是实时应用的一个很好的替代方案。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1387178419682279434)** (1 条消息): 

> `LessWrong 文章录用, Token 输入嵌入的梯度下降` 


- **LessWrong 接收“Token 输入嵌入的梯度下降”文章**：一位成员宣布其文章 [Token 输入嵌入的梯度下降：ModernBERT](https://www.lesswrong.com/posts/GK2LSzxjEejzDjzDs/gradient-descent-on-token-input-embeddings-a-modernbert) 已被 LessWrong 接收。
- **Token 输入嵌入的梯度下降**：该文章重点讨论 token 输入嵌入的梯度下降。
   - 这是一种 ModernBERT 方法。


  

---

### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1387097201162846349)** (1 条消息): 

> `Diffusers v0.34.0, 新版本发布` 


- **Diffusers 发布新版本**：[Diffusers v0.34.0](https://github.com/huggingface/diffusers/releases/tag/v0.34.0) 现已发布。
- **查看 Diffusers 新特性**：[Diffusers v0.34.0](https://github.com/huggingface/diffusers/releases/tag/v0.34.0) 现已发布，详情请参阅链接中的 Release Notes。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1386798915063513331)** (4 条消息): 

> `JAX 模型, 模型优化` 


- **GitHub 上出现 JAX 模型**：一位成员分享了在 [Locamage/jimm](https://github.com/Locamage/jimm) 实现的一些 **JAX 模型** 链接。
   - 这似乎是一个用 JAX/Flax 编写的极简 Stable Diffusion 流水线演示。
- **使用 Optimum 减小模型体积**：另一位成员分享了一个使用 [Optimum DETR](https://github.com/merveenoyan/smol-vision/blob/main/Reduce_any_model_to_fp16_using_%F0%9F%A4%97_Optimum_DETR.ipynb) 将任何模型转换为 **fp16** 的示例。
   - 该帖子似乎暗示这种优化方法可能存在偏差。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1386795903549178028)** (27 条消息🔥): 

> `Sentence Transformers 导致 Docker 崩溃, 输入嵌入 (Input embeddings), 扩展向量搜索, Langchain 的 FAISS, IndexIVFPQ` 


- **Sentence Transformers 导致 Docker 崩溃并报错 Error 252**：一位用户报告称，在使用 Sentence Transformers 计算相似度时，其 **Docker 容器** 崩溃并报错 **252**，罪魁祸首是 `similarities = embeddings1 @ embeddings2.T` 这一行。
   - 一位 Sentence Transformers 开发者建议这可能是由于内存占用过高，并建议尝试更小的 Batch。
- **输入嵌入实验引起关注**：一位用户分享了[一篇博客文章链接](https://dev.to/kylepena/gradient-descent-on-llm-input-space-a-modernbert-experiment-3053)，详细介绍了在 ModernBERT 背景下，对输入空间进行输入嵌入和梯度下降的实验。
   - 另一位用户认为这是一个有趣的资源并收藏了该帖子。
- **使用 FAISS 和矩阵乘法扩展向量搜索**：一位用户正在扩展向量搜索，并使用余弦相似度比较 `n` 个查询嵌入与 `m` 个文档嵌入，但在使用 Langchain 的 FAISS 封装进行 **1M** 次点积运算时遇到了瓶颈。
   - 他们发现使用 `torch.matmul` 或 `@` 将 **1M** 次比较的运行时间从 **25 秒缩短至 0.04 秒**，并计划针对 **10M+** 次比较使用量化 FAISS 索引（如 `IndexIVFPQ`）。
- **利用 Reciprocal Rank Fusion 突破语义搜索限制**：一位用户想要减少包含多个关键词的问题的相似度搜索次数，同时又不希望因为对嵌入取平均值而导致语义丢失。
   - 一位 Sentence Transformer 开发者建议结合使用基于密集嵌入（Dense Embeddings）的语义排序和基于稀疏嵌入（Sparse Embeddings，或 BM25）的词法排序，并配合 [Reciprocal Rank Fusion](https://link.to/reciprocal-rank-fusion) 以获得更好的效果。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1387095934999199865)** (1 条消息): 

> `Hugging Face 证书` 


- **Hugging Face 证书消失**：一位成员询问在完成课程的每个单元后是否仍能**生成 Hugging Face 证书**。
   - 他们再也找不到该选项，想确认证书是否仍然可用。
- **查询证书可用性**：用户正在询问 Hugging Face 证书是否继续提供。
   - 他们特别提到在完成课程单元后找不到生成证书的选项。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1386829945954898072)** (26 条消息🔥): 

> `Unit 4 最终项目提交工作流、证书获取问题、最终作业评估截止日期、Unit 1 测验访问问题、Agent 创建中 HF 环境变量的挑战` 


- **Unit 4 项目提交难题**：一位用户询问了提交 Unit 4 最终项目的工作流，幽默地质疑为什么*提交*项目似乎比*完成*项目还要难。
   - 另一位用户分享了他们在调试复杂的 HF 环境变量和包冲突时的痛苦经历，随后建议在本地运行你的 Agent，然后简单地通过 API 端点发送其响应，因为如果答案正确，这足以让组织者颁发证书。
- **证书获取困境**：一位用户询问如何在不重新进行测验的情况下获取他们的证书。
   - 上下文中未提供解决方案。
- **最终作业截止日期担忧**：一位用户询问最终作业在 **July 1st** 截止日期后是否仍会进行评估。
   - 上下文中未提供明确答复。
- **Unit 1 测验登录锁定**：一位用户报告由于登录问题无法访问 Unit 1 测验，怀疑是手机干扰。
   - 另一位用户建议该问题可能与 **WebKit** 有关，并建议在电脑上使用 **Firefox** 或 **Chrome**。
- **HF 生态系统中 Agent 创建的注意事项**：一位用户描述了在 Hugging Face 环境中创建 Agent 的挑战，强调了与 **shazam** 等库的冲突以及免费账户的内存限制，并指出你被迫从 HF 获取模型。
   - 该用户建议提交过程包括克隆项目模板、在小型模型上编写 Agent，并在移至 HF sandbox 之前进行本地测试。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1386796402151260361)** (81 条消息🔥🔥): 

> `用于语言练习的语音安装、LM Studio 中的图像生成功能、LM Studio 上下文窗口的 Roo code Discord 问题、Unsloth 的动态量化大小估计问题、增加聊天历史上下文长度` 


- **LM Studio 考虑为语言学习集成语音功能**：一位成员建议在 LM Studio 中添加语音安装功能，以便在学习语言时进行对话练习，类似于 **GPT CHAT**。
   - 另一位成员澄清说 **LM Studio** 主要针对 **LLMs**，而不是*满足所有 AI 需求的一站式商店*，暗示此功能可能超出了其范围。
- **LM Studio 可能接受图像作为输入**：一位用户询问在 LM Studio 中添加图像生成功能，一些成员澄清说，虽然 **LM Studio** 中的某些模型可以接受图像作为输入（如 **Gemma3**），但目前没有输出图像的功能。
   - 另一位成员提到可以通过额外的步骤使用 **web-ui** 设置文本生成图像，但未提供具体的设置建议。
- **来自 Unsloth 的动态量化导致加载问题**：用户报告了估计来自 **Unsloth** 的动态量化（Dynamic Quants）大小时的问题，这可能导致 **LM Studio** 过载 **VRAM** 并导致加载失败，尤其是在多 **GPUs** 和优先级排序的情况下。
   - 一位用户确认他们专门使用具有动态上下文窗口的 **Unsloth** 模型，表明这可能是导致该问题的常见配置。
- **平台更新困扰用户重新安装**：一位用户报告了一个 Bug，即 **LM Studio** 反复提示更新，并在更新前拒绝加载模型，而每次更新都超过 200MB。
   - 一位成员建议确保之前的更新已完全加载且未在后台运行，以解决持续的更新请求。
- **Reddit 的 LocalLlama 版块起死回生**：成员们讨论了在经历一段沉寂期后，**r/LocalLlama** 版块在新的管理下回归。
   - 虽然有人担心新版主参与了过多的版块管理，但目前尚未发现明显的负面信号。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1386798324891517039)** (41 messages🔥): 

> `P40 在 mATX 主板上的应用，多 GPU 与瓶颈对比，LM Studio 在 Ryzen 55900xt 上的性能，3x3090 导致速度变慢` 


- **关于 P40 在 mATX 布局上适配的争论**：讨论围绕 **两块 P40 GPU** 是否能在不发生散热导流罩碰撞的情况下安装在 **mATX** 主板上展开，并探讨了它们在 **Llama4 模型**或 **Gemma 27b QAT 4** 模型上的性能。
   - 一位用户建议，虽然网上可以查到双槽卡的规格，但第二个问题取决于机箱空间，并提出了可以部署的替代散热机制。
- **多 GPU 推理是否会使用 CUDA？**：在推理过程中，**LLM 层**会分布在多个 GPU 上。
   - LLM 的每一层都由持有它的 GPU 进行处理，以防止 **PCIe 带宽瓶颈**。
- **更多 VRAM 意味着更好的 LM Studio 体验**：一位用户询问了 LM Studio 在配备 **128GB RAM**、**1TB SSD** 和 **RTX 3060 12GB** 的 **Ryzen 55900XT** 上的性能，而另一位用户指出 *对于本地模型来说，VRAM 越多越好*。
   - 该用户计划将其用于 **24/7** 后台分析学术文档，并表示如果能将所有内容卸载（offload）到 RAM 中，他们并不介意等待输出。
- **三重威胁：三张 3090 导致吞吐量问题**：一位用户测试了带有 3 张显卡的 LM Studio，发现性能显著下降，并指出第三张显卡开启时会 *相当大程度地* 降低速度，可能需要决定最后那 12GB VRAM 是否值得。
   - 据观察，2 张显卡可产生约 55 t/s 的速度，而 3 张显卡仅得到约 35 t/s，这引发了关于 CPU 没有足够 **PCIe 通道（lanes）** 的推测。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1387107753788706918)** (1 messages): 

> `Meta 提供商问题，定价疑问` 


- **报告 Meta 提供商故障**：OpenRouter 上的 **Meta 提供商** 今天遇到了一些问题。
   - 问题已反馈给 **Meta**，目前正在努力解决，争取尽快恢复服务上线。
- **定价问题频发**：用户对 **定价结构** 表达了疑问和担忧。
   - 团队正在积极处理这些咨询以提供清晰的解答。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1386787242911076433)** (103 messages🔥🔥): 

> `OpenRouter 提供商偏好，Novita 关于 R1-528 最大输出长度的错误信息，OpenRouter 上的 Stripe 支付方式问题，推理 Token 与总 Token 计数对比，Cent-ML 提供商更换` 


- **OpenRouter 的提供商偏好悖论**：一位用户询问 OpenRouter 的提供商偏好是如何工作的，指出选择特定提供商并未按预期运行，并询问“排序偏好（sort preference）”的含义。
   - 提供了 [OpenRouter 关于提供商路由的文档](https://openrouter.ai/docs/features/provider-routing) 链接以澄清该功能。
- **Novita 误报 R1-528 的 Token 限制**：一位用户指出 Novita 提供了关于 **R1-528** 最大输出长度的错误信息，声称其为 **131k**，而实际为 **16k**。
   - 用户质疑 OpenRouter 是否核实提供商信息，认为此类差异应该很容易识别。
- **推理 Token 引发 Token 统计困扰**：一位用户报告在使用 OpenRouter 时，收到的结果中 **reasoning_tokens** 高于 **total_tokens**。
   - 一名工作人员澄清说，**推理 Token（reasoning tokens）** 是 **completion token 详情** 的一部分，**total tokens** 不包含推理 Token，并且 **修改 JSON 以将推理 Token 添加到总 Token 中会破坏数千个正在运行的应用**。
- **Google AI Studio 上 Gemini 2.5 Pro 的速率限制风波**：用户讨论了 Google AI Studio 上 **Gemini 2.5 Pro** 的速率限制，指出虽然界面列出了 **150 RPM** 的限制，但免费层用户的实际限制似乎更低。
   - 一位用户在短时间内发送大量请求后遇到了错误和冷却期，认为该限制更像是一种 *公平使用限制（fair use limit）*，以防止逆向工程或自动化。
- **Midjourney + Spellbrush 发布突破性的 i2v 模型**：Midjourney 和 Spellbrush 的新视频模型效果惊人。
   - 一位用户表示 *这基本上是 i2v 的 ChatGPT 时刻*，并希望有更多基础设施来推行 **720p**。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1387135866383761519)** (9 messages🔥): 

> `C++ CUDA build systems, Meson, Buck2, xmake, Zig` 


- **C++/CUDA 构建系统推荐**: 用户讨论了 **C++ 和 CUDA 项目**的构建系统，寻求 CMake 之外的替代方案。
   - 建议的选项包括 **Make**、**Meson**、**Buck2**、**xmake** 和 **Zig**，反映了在管理此类项目时，除了 CMake 之外还有多种选择。
- **XMake 看起来很有前景**: 一位用户建议并链接了 [xmake](https://xmake.io/#/)，强调了它在 CUDA 项目中的易用性。
   - 他们分享了一个配置示例，展示了如何**定义目标 (target)**、**指定 CUDA 文件**以及**添加 CUDA 架构**。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1386818320719806647)** (2 messages): 

> `Triton AOT Compilation, Triton Community Meetings, Fused Attention Kernel` 


- **Triton AOT 编译类型提示**: 一位用户正尝试利用 **Triton** 进行 **AOT 编译**，但在 [fused attention kernel 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) 的 `_attn_fwd_inner` 函数中对 `q` 张量进行类型提示时遇到问题。
   - 该用户正在寻求如何正确进行张量类型提示的指导，并指出 [compile.py](https://github.com/triton-lang/triton/blob/main/python/triton/tools/compile.py#L23) 中的 `str_to_ty` 函数似乎仅支持 `pointer`、`tensordesc` 和 `constexpr`。
- **Triton 社区会议缺席？**: 一位用户询问了 **Triton 社区会议**的状态，注意到 YouTube 上最后的录像是在 2024 年 11 月。
   - 该用户特别关注任何与重大变更（包括布局系统 layout system）相关的设计规范或讨论。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1386843717763338251)** (11 messages🔥): 

> `CUB with NVRTC, matmul overlap, JIT safe standard library headers, torch.cdist implementation` 


- **NVRTC 在集成 CUB 时遇到困难**: 一位成员尝试在 `torch.cuda._compile_kernel` 中让 **CUB** 与 **NVRTC** 配合使用以实现极速编译，但遇到了缺少 C++ 标准库头文件的问题。
   - 另一位成员建议使用 `#include <cub/block/block_reduce.cuh>` 而不是 `#include <cub/cub.cuh>`。
- **最大化 matmul 中 TC 与非 TC 操作的重叠**: 一位成员寻求关于在具有重计算量 Epilogue 的 **matmul** 中最大化 **TC (Tensor Core)** 与**非 TC** 操作重叠的建议，特别是在非数据中心级 GPU 上。
   - 他们发现很难让 `nvcc` 生成将 `*MMA` 指令与其他 ALU 指令均匀交错的 **SASS 代码**，并猜测编译器可能会优先考虑 `*MMA` 输入寄存器上的 `.reuse`。
- **提供 JIT 安全的标准库头文件**: 一位成员建议查看 [NVIDIA 的 jitify](https://github.com/NVIDIA/jitify)，以获取对 **JIT 安全的标准库头文件**（如 float.h, stdint.h 等）的支持。
- **Torch cdist 实现针对 Cutlass 进行 JIT 编译**: 一位成员分享了他们的仓库 [Kernel-Machines/kermac](https://github.com/Kernel-Machines/kermac)，该仓库**针对 cutlass/cute 进行 JIT 编译**以实现 `torch.cdist`，性能远超 torch 原生实现。
   - 他们使用 **lmdb 数据库**缓存 JIT kernel，如果模块已加载则直接从模块加载函数，从而避免了 wheel 文件的麻烦和 CUDA 扩展的混乱。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1386799820873793686)** (1 messages): 

> `TorchTitan, SimpleFSDP, TP and FSDP collectives, Inductor` 


- **SimpleFSDP 实现加入 TorchTitan**: 如 [README](https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/simple_fsdp/README.md) 中所述，*TorchTitan* 中的 **SimpleFSDP** 实现是捕获包含所有集合通信 (collectives) 图的最佳方式。
- **TP 已添加到 SimpleFSDP 版本**: 根据 [此 PR](https://github.com/pytorch/torchtitan/pull/1250)，**张量并行 (TP)** 最近被添加到 SimpleFSDP 版本中，从而支持编译同时包含 **TP** 和 **FSDP 集合通信**的图。
- **Inductor 秘籍揭晓**: 对于使用编译栈（包括用于算子融合的 **Inductor**）并配合自定义计算/通信重叠逻辑的情况，Inductor 中有用于注册 graph passes 的私有钩子 (hooks)，详见 [config.py 文件](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L262)。


  

---

### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1387054679380459530)** (4 messages): 

> `LLM, CUDA, algorithms` 


- **计算架构优于线性算法**：一位成员认为，LLM 的初学者应该优先理解 **compute architecture**（计算架构）和 **cache hierarchy**（缓存层级），而不仅仅是线性算法。
   - 他们补充道，*一个需要从内存中进行大量获取（fetch）的线性算法，其表现通常不如一个能保留在 SMEM 中的二次算法*。
- **CUDA 实现高效算法**：一位成员提到，在 **CUDA** 中高效地实现算法不仅仅是学习理论。
   - 他们指出，对于除了最基础算法之外的并行算法，经典的复杂度理论过于简化，并提到了 [PMPP](https://github.com/rmcantin/PMPP) 作为例子。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1386886630761037885)** (1 messages): 

> `PyTorch Tool, Machine Learning Efficiency, Optimization, Mentorship Opportunity, Medical Device CV` 


- **医疗 CV 工程师寻求 PyTorch ML 效率导师**：一位医疗设备领域的计算机视觉工程师正在寻求一位导师，针对一个专注于 **machine learning efficiency**（机器学习效率）和 **optimization**（优化）的 **PyTorch tool** 提供建议。
   - 该工程师提供**共同署名权**作为导师时间的报酬，可以通过 DM 或发送邮件至 [s.askaruly@gmail.com](mailto:s.askaruly@gmail.com) 联系。
- **工具寻求指导**：一位工程师正在开发一个用于 **machine learning efficiency** 的 **PyTorch tool**，希望就该工具可以解决的具体问题寻求指导。
   - 他们缺乏该领域的工业经验，希望导师能提供反馈和建议。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1386807256745508914)** (6 messages): 

> `cuML, NVIDIA driver, CUDA toolkit, threadIdx.y vs threadIdx.x` 


- **cuML 与 NVIDIA 驱动及 CUDA Toolkit 的兼容性困扰**：一位用户询问关于在较新的 **NVIDIA driver** 上使用 **cuML** 的问题，其安装的 **CUDA toolkit** 遇到了兼容性问题。
   - 讨论明确了较新的驱动程序支持较新的 toolkit，这表明环境被锁定在了旧的、不兼容的 toolkit 上，用户通过卸载旧版本并安装正确版本解决了该问题。
- **矩阵乘法中的线程索引困惑**：一位用户询问为什么在基础矩阵乘法中，**threadIdx.y** 用于行而 **threadIdx.x** 用于列，而他们预期的是相反的情况。
   - 另一位用户解释说，*`threadIdx.x` 是 warp 布局的维度*，这使得它适合作为 row-major（行优先）布局中的列索引，以及 column-major（列优先）布局中的行索引，从而实现合并全局内存访问（coalesce global memory accesses）。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1386887446117224589)** (1 messages): 

> `Reduction code correctness, Input length handling` 


- **发现 Reduction 代码 Bug！**：一位用户发现了《Programming Massively Parallel Processors (第4版)》第 10 章中关于处理非 2 的幂次输入长度的 **reduction code**（归约代码）的一个潜在问题，并寻求确认。
   - 他们提供了一个 [代码链接](https://github.com/katsudon16/programming_massively_parallel_processors/blob/98616b84fd03b5110bfa5d4d9470568caf34eb08/chapter_10/sum_reduction_less_control_divergence.cu#L15) 作为他们实现的额外处理逻辑的示例。
- **输入长度的烦恼！**：该用户的代码片段展示了当输入长度不是 2 的幂次时，为确保功能正确性所需的额外步骤。
   - 这引发了关于教科书中原始实现在实际应用场景下鲁棒性的疑问。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1386787747314008174)** (4 messages): 

> `rocprofiler-sdk Integration, Chisel Performance Counters` 


- **Chisel 实现 rocprofiler-sdk 集成**：一位成员宣布在 **Chisel** 中实现了 **rocprofiler-sdk integration**，该实现基于之前描述的设置，可以自动从主线构建 *aqlprofile* 和 *rocprofiler-sdk*。
   - 该集成会下载 **rocprof-trace-decoder** 二进制文件并设置环境变量，并带有一个新的 **--pmc flag** 用于收集自定义性能计数器（例如：`chisel profile amd kernel.cpp --pmc GRBM_GUI_ACTIVE,SQ_WAVES`）。
- **Chisel 支持自定义性能计数器收集**：**Chisel** 中新的 **--pmc flag** 允许用户收集自定义性能计数器，如 **GRBM_GUI_ACTIVE** 和 **SQ_WAVES**。
   - 此功能旨在提供更细粒度的性能洞察，可通过 *rocprofiler-sdk* 集成使用。


  

---

### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1387160283423969301)** (1 messages): 

> `Intel GPU atomic latency, Ponte Vecchio VTUNE, SYCL device cycle counters` 


- **寻求 Intel GPU 原子操作延迟见解**：一名成员正在询问在 **Intel GPU**（特别是使用 **Ponte Vecchio**）上计算每个线程原子操作延迟的方法。
- **VTUNE 或 SYCL 设备周期计数器**：他们正在考虑使用 **VTUNE** 或 **SYCL** 设备周期计数器。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1386785129510342819)** (8 messages🔥): 

> `GPU Rental, Chisel Tooling, CUDA Competition, 3D Gaussian Splatting` 


- **Chisel 添加 L40s 作为 T4 的替代方案**：为了弥补 DigitalOcean 上缺乏 **T4 GPU** 支持的问题，[Chisel CLI](https://www.chisel.so/) 现在支持以约 **$1.57/小时** 的价格使用 **Nvidia L40s GPU**。
   - 用户可以运行 `pip install chisel-cli`，`chisel configure` 和 `chisel run nvidia --gpu-type l40s <kernel.cu>` 来下载 **nsight-compute** 和 **nsight-systems** 的分析输出。
- **宣布旨在加速 3D Gaussian Splatting 的 CUDA 竞赛**：一名成员宣布了一项以 **CUDA** 为导向的竞赛，奖金为 **$1100**，目标是将 [3D Gaussian Splatting](https://github.com/MrNeRF/gaussian-splatting-cuda) 的训练时间减少 **50% 或更多**。
   - 提交的内容必须基于 **GPLv3 license** 开源，截止日期为 **2025 年 7 月 31 日**，测试基准为 **RTX 4090**。
- **使用 CUDA 加速 AI 产品**：一名成员分享了一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/activity-7343124619568009217-tl4Y) 以及配套的 Medium 文章，主题为 *使用 CUDA 加速 AI 产品*。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1386799896157225040)** (30 messages🔥): 

> `KernelLLM, Triton Data, Kernelbot Data, Synthetic Datasets, PyTorch to Triton Conversion` 


- **KernelLLM 需要特定的代码格式！**：成员们讨论了 **KernelLLM** 提示词的正确格式以确保最佳性能，指出它需要 `Model(nn.Module)` 和 `get_inputs` 函数；可以参考这里的[指南](https://huggingface.co/facebook/KernelLLM/discussions/5#685b0903b3d048882566b17b)。
   - 会议强调 **KernelLLM** 对其处理的代码不够灵活，对输入格式有特定要求。
- **Triton 数据依然稀缺！**：互联网上可用的由人类生成的 **Triton 数据** 非常少，这促使了像 **Kernelbot** 这样资源的创建。
   - 一名成员分享了 [Kernelbot data](https://huggingface.co/datasets/GPUMODE/kernelbot-data) 数据集，该数据集全部由人类编写，但仅适用于少数问题。
- **合成数据助力模型！**：有人正在创建**合成数据集**，通过生成 PyTorch 代码并将其转换为 Triton，然后循环直到输出匹配，以此来引导模型。
   - 另一名成员建议使用 **Gemini** 从算子列表生成程序，或者通过 RL（强化学习）标注 Trace，并指出 **RL** 需要一些引导才能生效。
- **KernelLLM 在 vLLM 部署中的困扰！**：一名成员报告了在 **vLLM** 中运行 **KernelLLM** 的问题，指出它很少能生成正确的 Triton Kernel，可能是由于 **vLLM** 使用方法不当。
   - 预计对输入格式会有类似 `Model(nn.Module).forward()` 的限制，但这些应该得到更好的文档说明。


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/)** (1 messages): 

dragan.jovanovich: 恭喜👏
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1386913315103440917)** (2 messages): 

> `CUDA Matmul precision issues, Triangle Multiplicative Update (Trimul) in AlphaFold` 


- **CUDA Matmul 面临精度困境**：一名成员报告称，由于[精度不匹配](https://github.com/yechenzhi/reference-kernels/blob/main/problems/pmpp/matmul_py/submission.py)，自定义 CUDA matmul 实现中的测试用例失败。
   - 错误显示元素不匹配且差异很小，这表明存在浮点精度问题。
- **Trimul 时间：AlphaFold 三角形乘法更新挑战赛启动**：一个来自 AlphaFold 家族的新问题——**三角形乘法更新** (Trimul)，现在已在 NVIDIA 和 AMD GPU 上可用，并附有[详细说明](https://tinyurl.com/gpumode-trimul)。
   - 该挑战赛专注于在不同硬件架构上优化 Trimul 操作，这是 **AlphaFold** 结构预测中的核心组件。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1386925561607225384)** (7 messages): 

> `prefixsum performance, sort performance, trimul performance on B200, trimul performance on A100` 


- **H100 PrefixSum：第七名！**：一名成员在 **H100** 的 `prefixsum` 排行榜上以 **1037 µs** 的成绩获得了**第 7 名**。
   - 这是三次提交中最好的一次，接下来的两次提交记录的时间分别为 **3.20 ms** 和 **2.87 ms**。
- **Sorting 在 H100 上获得第五名！**：一名成员在 **H100** 的 `sort` 排行榜上以 **7.16 ms** 的成绩获得了**第 5 名**。
- **trimul：攻克 B200！**：两名成员分别以 **7.92 ms** 和 **8.20 ms** 的成绩在 **B200** 的 `trimul` 排行榜上获得了**第一名**和**第二名**。
- **trimul：在 A100 上表现出色！**：一名成员在 **A100** 的 `trimul` 排行榜上以 **20.0 ms** 的成绩获得了**第一名**。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1387187660900012104)** (1 messages): 

> `New leaderboard problem, AMD + NVIDIA hardware` 


- **新的排行榜问题发布！**：一个新的排行榜问题现已上线，同时支持 **AMD** 和 **NVIDIA** 硬件。
   - 更多详情可以在此文档中找到：[https://tinyurl.com/gpumode-trimul](https://tinyurl.com/gpumode-trimul)。
- **深入探索 'trimul' 挑战**：GPU MODE 排行榜发布了一项名为 'trimul' 的新挑战，适配 **AMD** 和 **NVIDIA** 架构。
   - 感兴趣的参与者可以通过提供的链接访问问题的具体细节和指南：[https://tinyurl.com/gpumode-trimul](https://tinyurl.com/gpumode-trimul)。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1386836174987071699)** (5 messages): 

> `Factorio Client Authentication, FLE updates, Error cases in FLE` 


- **仍需 Factorio 客户端身份验证**：用户讨论了 **Factorio Learning Environment (FLE)** 是否可以在没有 **Factorio 客户端身份验证** 步骤的情况下运行。
   - 一名成员确认目前是必需的，但一个旨在移除客户端登录需求的 **PR** 正在进行中，预计将于本周合并。
- **多个错误案例困扰 FLE**：一名用户在 **FLE** 中遇到了多个不同的错误案例。
   - 他已经创建了一个 **PR** 来解决其中一个较明显的问题，并建议将 **220 到 238** 号 PR 合并到主分支，以确保在进一步工作之前处于可用状态。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1387199760993095853)** (1 messages): 

> `CuTe DSL, GEMM kernel, TMA transfers, MMA operations, sm90 architecture` 


- **CuTe DSL 加速 Kernel 编译**：一名用户尝试使用 **CuTe DSL** 为 **sm90** 实现一个 persistent ping pong **GEMM kernel**，其中一个 producer warpgroup 启动 **TMA transfers**，两个 consumer warpgroups 启动 **MMAs**。
   - 该用户报告称，与在 **C++** 中进行相同操作相比，DSL 几乎瞬时的编译时间、易于打印和设置断点以及 Python 风格的特性使其体验好得多。
- **Persistent Ping Pong GEMM Kernel 中的屏障同步问题**：该用户在实现 persistent ping pong **GEMM kernel** 时遇到了屏障同步（barrier synchronization）问题。
   - 问题的详细信息可以在 [Cutlass 的 GitHub issue #2418](https://github.com/NVIDIA/cutlass/issues/2418) 中找到。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1386783101199454422)** (90 messages🔥🔥): 

> `NVMe, Network cards, MI300x with AMDGPU, ResNet and BERT Training, GPU kernel` 


- **NVMe 抽象讨论**：成员们讨论了 **NVMe** 的简单性和标准性，但指出它会丢失文件系统等抽象。
   - 提出了 *'DISK:/dev/nvme0'* 作为潜在寻址方案的想法，并询问了卸载内核驱动程序的难易程度。
- **Infiniband 传输导致 Graph 断裂**：目前实现的 Infiniband 传输需要断开 Graph，这对性能不利。
   - 讨论涉及 Graph 中的传输是否应与 Copy 相同，以及使用 **RDMA** 进行远程 DMA 的挑战。
- **为网卡编写 GPU Kernels？**：讨论考虑编写网卡驱动程序并在 GPU 上运行该驱动，以便更好地控制传输。
   - 建议允许任意 **CPU kernels**，并在 Graph 中使用 CPU kernel 作为回调来设置传输。
- **TinyBox 问题**：潜在客户询问了购买 **TinyBox** 的事宜，包括交货周期和维护/支持信息。
- **赏金任务误解澄清**：一名成员误解了赏金任务的要求，认为涉及不使用 JavaScript 的 Web 服务器和语音转文本处理。
   - 对该赏金任务的正确理解是：涉及运行 **Whisper**，并使用基础 JavaScript 将音频数据传递给 **WGSL**，且不使用外部库进行预处理。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1387114562452455577)** (1 messages): 

> `FP8 Conversion, Hardware Compatibility` 


- **FP8 转换函数上线**：一名成员实现了一个手动将 **fp8e4m3** 和 **fp8e5m2** Tensor 转换为 **float32** 的函数。
   - 该函数适用于硬件原生不支持 **FP8** 类型的用户，代码可在 [这里](https://github.com/softcookiepp/tinybloat/blob/master/src/tinybloat/compatibility.py) 获取。
- **FP8 硬件兼容性**：新函数解决了 **FP8** Tensor 类型的硬件兼容性问题。
   - 它允许使用旧款或性能较低硬件的用户，通过将其转换为 **float32**，仍能运行使用 **FP8** 的模型。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1386785315561144330)** (50 messages🔥): 

> `Synthetic Data for Training, Meta's Synthetic Data Kit, Gemini Pro Stable's Instruction Following Issues, Aider Benchmark Framework, Claude Max Integration with Aider` 


- **合成数据工具面临 Diff 难题**：一名成员正在开发用于训练的**合成数据生成工具**，但在将其作为编辑器使用时遇到了错误的 Diff 问题。
   - 他们正在探索正确的 Distillation（蒸馏）方案，包括对模型响应使用 Logits 或 QLoRA，并从 [Exercism](https://exercism.org/) 题目中汲取灵感以构建具有挑战性的 Benchmark。
- **Gemini Pro Stable 指令遵循吃力**：多位成员观察到 **Gemini Pro Stable** 的指令遵循能力较差，一位用户指出必须对其进行“严密监控”。
   - 一位用户分享道，当要求其将 **1.1** 节标记为已完成时，它完成了所有任务并创建了新文件，但未能应用更改，导致仓库被“搞砸”了。
- **Aider Benchmark 缺乏接口信息**：一名成员质疑 LLM 在没有明确接口信息的情况下如何通过 **Aider benchmark** 中的单元测试，并引用了 GitHub 上的 [polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark/blob/main/cpp/exercises/practice/all-your-base/all_your_base.cpp)。
   - 另一名成员澄清说，指令中包含了相关信息，且第一轮测试的结果会被添加到第二轮的 Context 中，为模型提供了通过错误信息学习接口的途径。
- **Claude Max 可能会与 Aider 集成**：一名成员询问是否可以将 **Claude Max** 连接到 Aider（类似于 Roo Code），由于可能的服务器条款限制，目前回复尚不确定。
   - 一位用户建议使用 [claude-code-api](https://github.com/codingworkflow/claude-code-api) 和 Chimera 模型来实现有趣的 Distillation 技术。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1386944419600531609)** (26 条消息🔥): 

> `Aider 异常交互、deepseek-r1 Token 限制、aider 中的 MCP 支持、Gemini 的智能表现` 


- ****Aider 表现异常：间歇性文件写入问题****：一位用户在处理新项目时遇到 **aider** 仅显示 Diff 而不写入文件的情况，尽管之前的尝试中提交成功。这可能是由于超出了 **deepseek-r1** 的 [Token 限制](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)。
   - 用户指出，*在发出命令后立即出现了 Token 警告*，而文件较少的简单任务则运行正常。
- ****Deepseek-r1 规模至关重要：上下文窗口混淆****：一位用户对上下文窗口大小提出疑问，指出 [openrouter.ai/deepseek/deepseek-r1-0528](https://openrouter.ai/deepseek/deepseek-r1-0528) 拥有 **128k 上下文窗口**，并建议该用户可能误读了排行榜。
   - 他们澄清说 **0528 版本应该优于** 用户遇到问题的那一个版本。
- ****Aider 缺失 MCP 集成****：一位用户询问如何通过脚本使用 **aider**，以便在 **MCP (Mod Compliance Package)** 中使用 `patch` 和 `awk` 等常用 CLI 命令。
   - 回复指出 *aider 目前没有对 MCP 的官方支持*，但建议 *指令 aider 运行 CLI 命令，并结合 --yes-always 参数* 来自动执行 CLI 命令。
- ****Gemini 的天赋：在衰退还是在增长？****：一位用户质疑 **Gemini** 是否变得不那么聪明了，引发了关于模型性能的辩论。
   - 一位成员链接了一篇 [博客文章](https://aider.chat/2024/08/26/sonnet-seems-fine.html)，而另一位成员则反驳称，企业会 *在幕后对内部模型进行微调*，而这些改动很少体现在技术实现细节中。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1386789432073388072)** (10 条消息🔥): 

> `Claude Code、后端、订阅、API 调用、SDK` 


- ****Claude Code API 提案****：一位成员提议使用 [Claude Code](https://github.com/codingworkflow/claude-code-api) 作为 Aider 的后端，以利用其订阅优势。
   - 该方案的核心价值在于相比直接进行 API 调用，其 **调用成本更低**，对于高用量用户来说可以节省资金。一位用户报告称，使用 Claude Code Pro 在 30 天内完成了 **等值超过 1200 美元的 API 使用量**。
- ****关于 Claude Code 服务条款的疑问****：一位用户询问了 Claude Code 的 **服务条款 (ToS)**，认为将其集成在另一个工具之后可能是可以接受的。
   - 还有人推测 `/run claude -p "XXXX"` 会如何表现，考虑是将其包含在上下文中，还是将其作为 Provider 执行代码编辑。
- ****讨论将 Claude Code 作为 Provider****：另一位成员强调，[Anthropic 文档](https://docs.anthropic.com/en/docs/claude-code/sdk) 表明，如果使用 SDK，**将 Claude Code 作为 Provider 使用没有问题**。
   - 对于将 Claude Code 作为服务进行无缝集成，大家似乎感到非常兴奋。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1386865643344891965)** (37 messages🔥): 

> `Efficient Pneumonia Detection with Vision Transformers, Scaling Vector Search with FAISS, GRPO for RL, FYP ML domain` 


- **Transformers 擅长肺炎检测**：一篇论文展示了在胸部 X 光检查中使用 **Vision Transformers** 进行[高效肺炎检测](https://www.nature.com/articles/s41598-024-52703-2)。
   - 一位成员觉得这类结果在 **2024** 年还能发表令人“震惊”，称其为“十年前的老新闻”。
- **向量搜索扩展策略对决**：一位成员正在扩展向量搜索，对于高达 **1M** 次的比较使用 **torch.matmul**，而对于 **10M+** 的规模则切换到量化的 **FAISS index**（如 `IndexIVFPQ`）。
   - 另一位成员警告说，由于精度权衡可能导致 **训练不稳定性**，在使用量化索引时要小心。
- **GPU 分片 vs. matmul 的 OOM**：一位成员询问计算是在 **单个 GPU** 上完成的，还是在 **多个并行 GPU** 上分片的。
   - OP 计算出 **1k*1k 的点积**将使用 **4MB** 内存，但对 PQ 的**压缩**持谨慎态度，因为相似性搜索对 embeddings 非常敏感，压缩可能会改变其含义，而且他对 PQ 的了解也不多。
- **GRPO 在 RL 中大显身手**：**GRPO** 是在 RL 中变得出类拔萃的方法。附带的图片包含一个 **TLDR**，解释了 Dr. GRPO 如何在保持性能的同时让模型不再那么“**话痨**”（yapper）。
   - 一位成员分享了一条 [tweet](https://fxtwitter.com/jmhessel/status/1899909893324468444)，区分了 **通用 RL** 和 **LLM RL**。
- **毕业设计（FYP）项目：GritLM, Gaussian Splatting, NER**：一位成员正试图寻找一个具有实际应用案例和研究潜力的**较少被探索的 ML 领域**，并为毕业设计项目提供了多个选项：**3D 点云、人体器官 3D 可视化、图像着色、医学可视化、双摄像头 3D 视觉（立体视觉）和命名实体识别（NER）**。
   - 一位成员建议：**NER** 看起来是 [GritLM](https://github.com/ContextualAI/gritlm) 的领地；如果**图像着色**涉及 Gaussian Splatting，那就不算“较少被探索的 ML 领域”，数字孪生是其应用之一；**医学可视化**在有训练数据的情况下是可行的，否则不值得；**立体视觉**在室内噪声较小，在室外表现较差。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1386879812148396133)** (12 messages🔥): 

> `Cloud GPU Platforms, AI in Education, RWKV v6 and Finch Series, Time Crystal Computer` 


- **Vast.ai 获得推荐**：一位成员提到在研究中使用 [vast.ai](https://vast.ai)，暗示它是一个可行的 **Cloud GPU 平台**选项。
   - 没有给出关于性能和成本的细节，只提到它被“使用”了。
- **AI 威胁要吞噬学校？**：一位成员分享了一篇《[时代周刊](https://time.com/7295195/ai-chatgpt-google-learning-school/)》的文章，讨论了 **AI 对教育**和学习的影响。
   - 对话没有深入探讨文章内容，只是进行了分享。
- **RWKV v6 和 Finch 崛起**：一位成员报告了 **RWKV v6 (Finch 系列)** 的发布，这是一个 **1.5B** 参数的模型，在多语言和英语任务中达到了 **SOTA**，并具备多模态能力，引用了 [RWKV-5 & 6 论文](https://arxiv.org/abs/2404.05892)。
   - 根据[这条 X 帖子](https://x.com/BlinkDL_AI/status/1755656095970857269?s=20)，**Finch** 结合了类似于 **Mamba** 的选择性机制，在困惑度（perplexity）上超越了 Transformers。
- **时间晶体大脑？**：一位成员分享了一篇极不寻常的论文（《[由时间晶体制成的类脑计算机](https://www.researchgate.net/profile/Anirban-Bandyopadhyay-7/publication/337323300_A_Brain-like_Computer_Made_of_Time_Crystal_Could_a_Metric_of_Prime_Alone_Replace_a_User_and_Alleviate_Programming_Forever/links/5dd22692299bf1b74b4b38a3/A-Brain-like-Computer-Made-of-Time-Crystal-Could-a-Metric-of-Prime-Alone-Replace-a-User-and-Alleviate-Programming-Forever.pdf#page=11)》），称其为一篇“破纪录的离谱论文”。
   - 没有给出关于其内容的进一步信息。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1386789229199097906)** (26 条消息🔥): 

> `自然选择与 AI，基因工程 vs 自动化，AI 作为计算器，富人繁殖更少，关于 RL & LLM 的论文` 


- **AI Brainrot 助长自然选择？**：成员们讨论了 **AI brainrot** 是否会危及生存和繁殖，并提出 **低技能劳动力自动化** 会强化自然选择。
   - 一位成员认为，尽管工资低廉，但由于机器人成本高昂，体力劳动仍然大量存在，因此有必要针对自然选择进行保护。
- **基因编辑胜过进化？**：成员们讨论了 **genetic engineering** 可能很快就会盖过自然选择，尽管目前影响智力等特质的能力仍然有限。
   - 一些人认为 **AI 进步** 将迅速加速对基因的理解，而另一些人则对克服人类基因组的复杂性持怀疑态度。
- **AI：我们时代的计算器？**：成员们将 **AI 比作计算器**，建议限制 AI 的使用（例如禁止在背乘法表时使用计算器），并强调了潜在的长期认知影响。
   - 一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=z3awgfU4yno) 来支持这一观点。
- **新论文强调 RL 的局限性**：成员们引用了 [arxiv](https://arxiv.org) 上的新论文，讨论了在 **Large Language Models** 上使用 **Reinforcement Learning** 技术时的局限性和潜在陷阱。
   - 提到的论文包括 *Understanding R1-Zero-Like Training: A Critical Perspective*、*Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model*、*Reinforcement Learning Finetunes Small Subnetworks in Large Language Models* 以及 *Spurious Rewards: Rethinking Training Signals in RLVR*。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1386792161336492102)** (70 条消息🔥🔥): 

> `Harvey AI 融资，Replit ARR，AI Agent 监管，初创公司 vs 现任巨头，Magenta RealTime` 


- **Harvey AI 获得 3 亿美元巨额 E 轮融资**：[Harvey AI 宣布](https://xcancel.com/harvey__ai/status/1937155058476646591) 成功完成 **3 亿美元 E 轮融资**，公司估值达到 **50 亿美元**，由 Kleiner Perkins 和 Coatue 领投。
   - 他们还 [与 LexisNexis 签署了合作伙伴关系](https://www.lexisnexis.com/community/pressroom/b/news/posts/lexisnexis-and-harvey-announce-strategic-alliance-to-integrate-trusted-high-quality-ai-technology-and-legal-content-and-develop-advanced-workflows)，以整合受信任的高质量 AI 技术和法律内容。
- **Replit ARR 飙升至 1 亿美元**：[Replit 宣布](https://xcancel.com/Replit/status/1937212611520831718) 他们的 **Annual Recurring Revenue (ARR)** 已突破 **1 亿美元**，并向客户和支持者表示感谢。
   - 一些成员质疑 **11 亿美元的估值** 是否真的合理。
- **需要类人化的 AI 监管？**：Matan-Paul Shetrit 在 [这条推文](https://x.com/MatanPaul/status/1937200395115499592) 中概述了扩展 AI Agent 时可观测性（observability）与监管（supervision）之间的关键区别。
   - 他认为传统的监控手段已经不够了，因为 AI Agent 会主动与系统和客户交互，因此需要一种类似于管理人类的新型监督方法。
- **渠道为王，创新滞后？**：Alex Immerman 的 [推文](https://xcancel.com/aleximm/status/1937251084810219721) 强调了初创公司与现任巨头（incumbents）之间的核心对决：初创公司能否在巨头创新之前建立起分发渠道（distribution）？
   - 讨论强调了渠道的力量，一位用户指出 **OpenAI 快速获取用户** 的能力与 Google 形成了鲜明对比。
- **Anthropic 的合理使用裁决引发争议**：Adam Eisgrau 报道称，根据 Alsup 法官的裁决，[Anthropic 凭借合理使用（fair use）理由赢得了简易判决动议](https://xcancel.com/adameisgrau/status/1937480346976813454)。
   - 然而，审判仍将继续，以确定使用“盗版”互联网材料可能带来的损害赔偿。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1386797895868616934)** (37 条消息🔥): 

> `grok3mini, humanizing AI agents, building llms from scratch, llm inference app llamabarn, COCONUT gating layer` 


- **Grok3mini 使用量在 6 月激增**：**grok3mini** 的非 beta 版本现已可用，其使用量已从 6 月 19 日的 **200 万/日** 显著增加到 **1 亿/日**。
- **Georgi 发布本地推理应用 Llamabarn**：Georgi 发布了一款名为 **Llamabarn** 的新型本地推理应用，根据[这条 X 帖子](https://x.com/ggerganov/status/1937189250149257250)，该应用界面整洁并获得了积极的评价。
- **ryunuck 解释 COCONUT 门控层**：根据[这条 X 帖子](https://x.com/ryunuck/status/1937466079309144256)，**COCONUT** 使用了一个“门控（gating）”层，该层从隐藏状态（hidden states）中提取信息，以确定每个 token 的采样器参数，在 token 之间保持隐藏状态而不是重新启动。
- **Psyche 仪表板上的训练运行情况**：可以在 [Psyche 仪表板](https://psyche.network/)上查看训练运行情况，更多信息请参阅 [Psyche 架构](https://nousresearch.com/nous-psyche/)。
- **Facebook 的图书盗版诉讼情况不容乐观**：根据[这条 X 帖子](https://x.com/AdamEisgrau/status/1937480346976813454)，**Facebook** 的图书盗版诉讼情况看起来并不理想，因为他们似乎未能赢得最重要的部分（盗版指控），且训练被裁定为转换性使用（transformative）。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1387023361959530517)** (2 条消息): 

> `Model Recommendations, LORA Training, GGUF Conversion, Local LLMs on GTX 1080` 


- **寻求适用于 GTX 1080 的模型建议**：一名成员正在寻求适合进行 **LORA 训练**、**GGUF 转换**并在 **GTX 1080** 上运行的模型建议。
   - 目标是将该模型用于角色扮演（character acting）和一般的技术问题咨询（包括简单的“操作指南”），并通过 **LORA 训练**进行角色强化。
- **新手寻求本地 LLM 指导**：一名新用户请求关于在 **GTX 1080** GPU 上开始使用 **本地 LLMs** 的解释和建议。
   - 他们计划为角色扮演和一般技术问题 **LORA 训练**一个模型。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1387138352435957902)** (4 条消息): 

> `MultiNet v0.2, Manifold platform, R1-Zero-Like Training, RL Incentivize Reasoning, Spurious Rewards in RLVR` 


- **MultiNet v0.2 在 Manifold 上发布！**：MultiNet v0.2 是一个用于评估通用 AI 系统的开源平台，已在 [Manifold](https://www.manifoldrg.com) 上发布。
   - 查阅[论文](https://arxiv.org/abs/2505.05540)并申请[合作](https://www.manifoldrg.com/os-research-fellow-multinet/)。
- **深入理解类 R1-Zero 训练**：讨论了一篇题为《Understanding R1-Zero-Like Training: A Critical Perspective》的[论文](https://arxiv.org/abs/2503.20783)。
   - 该论文探讨了 **类 R1-Zero 训练** 的批判性视角。
- **RL 是否激励了推理能力？**：提到了一篇题为《Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model》的[论文](https://arxiv.org/abs/2504.13837)。
   - 讨论围绕 **Reinforcement Learning** 是否真的在基础模型之外激励了 **LLMs** 的推理能力展开。
- **使用 RL 微调小型子网络**：讨论了一篇题为《Reinforcement Learning Finetunes Small Subnetworks in Large Language Models》的[论文](https://arxiv.org/abs/2505.11711)。
   - 它涵盖了 **Reinforcement Learning** 在微调大型语言模型中小型子网络的作用。
- **探讨 RLVR 中的伪奖励（Spurious Rewards）**：介绍了一篇题为《Spurious Rewards: Rethinking Training Signals in RLVR》的[论文](https://arxiv.org/abs/2506.10947)。
   - 讨论集中在由于伪奖励而重新思考 **RLVR** 中的**训练信号**。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1386827903450480842)** (2 条消息): 

> `Reward Models, PAIE Curator` 


- **奖励模型的内部偏差：Cursed Helm 警报！**：一篇新论文 [Cursed Helm](https://arxiv.org/abs/2506.07326) 指出，盲目地将奖励模型集成到流水线中而不考虑其内部偏差可能会令人担忧。
   - 它提醒那些*仅仅将奖励模型强行接入流水线而不考虑其内部偏差*的人员。
- **PAIE Curator：LLM 升级监听器**：**PAIE Curator** 被介绍为一个本地 **LLM 升级监听器**，旨在捕获模型失败并提供结构化反馈循环。
   - 它在没有前端或向量搜索的情况下运行，专注于在模型表达不确定性（例如，“我不知道。”）时进行监听，其 [GitHub 仓库点击此处](https://github.com/ProjectPAIE/paie-curator)。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1387138352435957902)** (4 条消息): 

> `MultiNet v0.2, Manifold platform, Generalist AI evaluation, R1-Zero-Like Training, RL Incentivizes Reasoning` 


- **MultiNet v0.2 在 Manifold 发布！**：用于评估通用 AI 系统的开源平台 **MultiNet** 的 **0.2** 版本已在 [Manifold](https://multinet.ai) 发布。
   - 在 Manifold 网站上可以找到 [Discord 链接](https://www.manifoldrg.com)和更多项目信息，该网站还包含指向 [X](https://x.com/HarshSikka/status/1937525251401011377) 和 [LinkedIn](https://www.linkedin.com/posts/harsh-sikka_multinet-a-generalist-benchmark-for-multimodal-activity-7343285918671196162-GVva) 的链接。
- **关于类 R1-Zero 训练的新论文讨论！**：讨论中提到了以下论文：['Understanding R1-Zero-Like Training: A Critical Perspective'](https://arxiv.org/abs/2503.20783)、['Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model'](https://arxiv.org/abs/2504.13837)、['Reinforcement Learning Finetunes Small Subnetworks in Large Language Models'](https://arxiv.org/abs/2505.11711) 以及 ['Spurious Rewards: Rethinking Training Signals in RLVR'](https://arxiv.org/abs/2506.10947)。
   - 此外，还发布了一个 [YouTube 视频](https://www.youtube.com/watch?v=z3awgfU4yno)。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1386786703095758959)** (14 条消息🔥): 

> `Multiagent Cooperation, Prefix caching, red teaming conversational AI` 


- **新论文引发关于多智能体协作的反馈**：一位成员收到反馈称，他们的“处女作”论文虽然识别出了有趣的对话动态（如将问题作为断路器），但缺乏对**多智能体协作**文献的探讨。
   - 作者承认由于 **Claude Opus** 成本昂贵而存在局限性，并打算在未来版本中通过更大的样本量以及群体组成、模型参数和上下文窗口大小的变化来扩展研究。
- **针对长序列长度的 Prefix Caching 策略辩论**：一位成员询问是否有支持类似 **vLLM** 的 **prefix caching** 但能够将缓存存储在内存映射文件中的库，以应对序列过大而无法放入 VRAM 或 DRAM 的情况。
   - 另一位成员回应称，除非序列长度超过 **1M**，否则这种方法可能比重新计算 KV 慢，尽管提问者澄清其序列长度为 **128k**。
- **青少年加入服务器讨论对话式 AI 红队测试**：一位来自瑞典、正在学习卡车运输的 **17 岁**少年向服务器介绍了自己，他专注于利用社会和心理压力战术对**对话式 AI 进行红队测试**。
   - 他们在 [GitHub 仓库](https://github.com/Ufosxm34gt/Conversational-Red-Teaming-Casebook)中记录了自己的工作，寻求与他人建立联系并学习。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1386814732891193445)** (30 messages🔥): 

> `Spectral Normalization, Sleeping-DISCO Dataset, Generative Models and Dynamical Systems, Manifold Multimodal AI Benchmarks, RL incentive` 


- **Spectral Norm 近似推向极限**：一名成员将 **spectral normalization**（谱归一化）描述为估计/近似谱范数，然后为了数值稳定性将 *weight*（权重）除以该范数。
   - 他们指出其缺点是：*如果存在离群奇异值，那么其余的奇异值会被推向接近 0，这对于某些用例可能并不理想。*
- **Sleeping-DISCO 数据集寻求 EleutherAI 合作**：一名成员询问是否可以与 EleutherAI 合作，为其新的大规模生成式音乐建模预训练数据集 **Sleeping-DISCO-9M** 提供支持，该数据集已在 [Hugging Face](https://huggingface.co/datasets/sleeping-ai/Sleeping-DISCO-9M) 上发布。
   - 他们正在寻求协助以评估其质量，并提到他们的 [arxiv 预印本](https://arxiv.org/abs/paper) 需要语法修正。
- **Sleeping-DISCO 数据集的原创性引发辩论**：一名成员批评了 Sleeping-DISCO 数据集的原创性，认为它主要是在没有重大原创贡献的情况下重新索引了来自 **Genius.com** 的内容。
   - 数据集创建者澄清说，它为学术用途提供了歌词和 Genius 注释，类似于 **GeniusExpertise** 或 **The Million Song Dataset** 等其他数据集，并提供了指向 **YouTube 视频** 的链接以供数据下载，同时承认由于版权原因存在局限性。
- **Mean Flows 代码上线！**：[Mean Flows for One-Step Generative Modeling](https://arxiv.org/abs/2505.13447) 的作者宣布他们的代码现在可以从 [这里](https://x.com/ZhengyangGeng/status/1937268681819693125) 获取。
   - 这项研究与对生成模型和动力系统感兴趣的人分享了他们的发现。
- **Manifold 发布多模态 AI 基准测试**：Manifold 团队发布了开放基础设施 [multinet.ai](https://multinet.ai)，用于基准测试和改进通用多模态 AI 系统，并发布了两篇相关的 [论文](https://arxiv.org/abs/2505.05540)。
   - 他们正在征求反馈与合作，并邀请申请他们的 [研究员计划 (research fellowship call)](https://www.manifoldrg.com/os-research-fellow-multinet/)。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1386826778814976130)** (4 messages): 

> `NNsight pre-release, Loss curve decomposition, NDIF update, Orthogonal Gradient Basis` 


- ****NNsight** 的下一个版本即将发布！**：NDIF 团队正在预发布 **NNsight** 的下一个版本，这是一个用于操作和干预 PyTorch 模型的框架。
   - 一个 [Colab notebook](https://colab.research.google.com/drive/1wjQhbQKh2pwy-mxx4EFMBC1IEauzu9G0#scrollTo=ZuSXB8Bh1zEq) 详细介绍了这些变化并提供了相关链接。
- **利用正交梯度基分解损失曲线**：一篇新论文在 **orthogonal gradient basis**（正交梯度基）上分解了损失曲线，揭示了样本簇具有类似的“突破性动态 (breakthrough dynamics)”，而这在精确损失中是不可见的。
   - 这些簇和突破与玩具算术任务和真实语言建模设置中的特定技能相对应，详见论文 [此处](https://www.alphaxiv.org/abs/2505.14685)。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1387110878469558475)** (4 messages): 

> `Mojo GPU kernels, Mojo from Python Limitations` 


- **热情的社区新人向 Mojo 社区问好**：一位新的社区成员表达了对 Mojo 及其目标的兴奋，计划研究 **GPU kernels** 并在 Python 中调用 **Mojo**。
- **Mojo-Python 互操作性仍受限**：一名成员询问了从 Python 调用 **Mojo** 时限制是否依然存在，并引用了文档中[已知的限制](https://docs.modular.com/mojo/manual/python/mojo-from-python/#known-limitations)。
   - 他们确认这些限制在最新版本中确实仍然存在。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1386830562269855975)** (39 messages🔥): 

> `Modular 社区 CI 中的 Larecs 测试、Mojo 作为 Rust 替代品、Mojo Async 与 Rust Async、Mojo 中的语句开头错误` 


- **Larecs 测试在 Modular CI 中失败**：一位贡献者正在调试一个问题，即 [Larecs 测试](https://github.com/samufi/larecs) 仅在 modular-community CI 中失败，但在本地机器或 GitHub CI 中运行正常，这使得追踪问题变得困难。
   - 另一位贡献者在 M1 Mac 上复现了该问题，指出它仅在执行 `mojo test` 时失败，并怀疑发生了不安全操作。目前正通过使用 [debug 分支](https://github.com/samufi/larecs/tree/debug_query) 提供失败测试用例的详细输出来协助调试。
- **Mojo 通往类 Rust 安全性的路线图**：一位用户询问了 Mojo 作为 Rust 潜在替代品的安全特性，特别是关于 **sum types**（和型）、**pattern matching**（模式匹配）和 **predictable destruction times**（可预测的析构时间）。
   - 一位 Modular 工程师回应称，虽然目前已存在 `struct` 这种 *product types*（积型），但 *sum types* 和 *pattern matching* 已在计划中。他还解释说 Mojo 已经提供了 *RAII* 和 *ASAP destruction*（尽快析构），并且正在摆脱 *syntactic salt*（语法盐）。
- **Async 设计旨在避开 Rust 的困扰**：一位用户询问 Mojo 的 async 设计是否会解决 Rust 中异步编程遇到的困难。
   - 一位 Modular 工程师指出 [PR 3945](https://github.com/modular/modular/pull/3945) 和 [PR 3946](https://github.com/modular/modular/pull/3946) 是解决方案，并指出更好的 async 运行时和线性类型（linear types）可以消除对 `Arc<Mutex<T>>` 等结构的需求，同时还提到了用于改进 IO 的 [PR 4728](https://github.com/modular/modular/pull/4728)。
- **Mojo 神秘的语句位置错误**：一位用户在 Mojo 中使用代码段 `if info.value() > top:` 时遇到了错误 *"statements must start at the beginning of a line"*（语句必须从行首开始）。
   - 另一位用户建议添加 `var top` 作为潜在的修复方案，这表明可能存在变量声明或作用域方面的问题。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1386792338789236826)** (30 messages🔥): 

> `Manus PDF 读取问题、新 AI 架构开发、积分促销问题、Manus 积分、Manus 宕机` 


- **Manus 在处理 PDF 时遇到困难**：用户报告称 **Manus** 在读取文本文件和 PDF 时出现问题，聊天机器人要求提供纯文本输入。
   - 一位用户问道：*为什么 Manus 最近在读取文本文件和 PDF 时遇到困难？它总是说：由于我无法直接处理 PDF 文件，您能否将此文本以纯文本形式提供给我？*。
- **AI 架构梦想起航**：一位成员表示有兴趣为 **funcognitive 和元改进** 开发一种新的 AI 架构，旨在打造一个更好、更快的 Transformer。
   - 另一位成员询问：*有人对开发新的 AI 架构感兴趣吗？主要是为了功能认知（funcognitive）和元改进，就是做一个更好、更快的 Transformer*。
- **订阅促销引发不满**：一位用户在购买了额外的积分后，被拒绝延长订阅以匹配促销优惠，并被迫创建新账号。
   - 该用户表示：*花了 400 美元买了 2 个月的订阅。本月购买了 19,900 个额外积分时，要求 Manus 延长订阅以匹配促销优惠，但他们拒绝了。* 并觉得这种情况 *非常愚蠢*。
- **用户获得大量积分**：一位用户提到，作为长期的 **Manus** Beta 测试人员，他们因贡献获得了 **9 万积分**。
   - 另一位用户评论道：*我拿到了 9 万积分* 以及 *他们只是因为我的贡献而给我积分*。
- **Manus 经历宕机**：一些用户报告 **Manus** 出现卡顿并显示内部服务器错误，导致积分浪费。
   - 一位用户表示由于这些问题浪费了超过 **2000 积分**，另一位用户说：*我觉得 Manus 变笨了，会犯错而且发现不了错误*。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1386932593647423589)** (3 messages): 

> `TorchTune、单机 LORA、GitHub Issues` 


- **TorchTune 的单机 LORA 功能受到赞赏**：一位用户花了半天时间尝试 **TorchTune**，特别是单机 **LORA**，并表示该软件包令其印象深刻，非常有用。
   - 该用户感谢开发者的工作。
- **TorchTune 团队鼓励通过 GitHub 提供反馈**：**TorchTune** 团队对正面反馈表示感谢，并鼓励用户在 [GitHub](https://github.com/pytorch/torchtune) 上发表评论或提交 issue 以提供进一步反馈。
   - 他们还提到团队的响应通常非常迅速。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1387043924337623141)** (25 messages🔥): 

> `Expandable Segments Bug, max-autotune 问题, 清理缓存, L40S 显卡 Bug, Reward Modeling RFC` 


- **Expandable Segments 触发指针错误**：一位成员在 **Nvidia L40S** GPU 上使用 **expandable segments** 时遇到了指针错误，禁用该功能解决了问题，但在 **H200s** 上运行正常。
   - 该解决方案在与 packing、flexattention 以及 `max-autotune` 设置相关的 [此 issue](https://github.com/pytorch/pytorch/issues/140419) 中被发现。
- **max-autotune 导致共享内存不足的显卡崩溃**：一位成员认为问题可能出在 **max-autotune** 而非硬件本身。
   - 目前 `roundup_power2` 标志尚不可用，但一位成员注意到 Unsloth 在使用 **expandable segments** 时带有此标志。
- **清理缓存可清除显卡错误**：一位成员复现了该错误，并发现清理缓存后显卡在第二次尝试时可以正常工作。
   - 清理缓存后，会出现一次失败，然后成功，即使没有新设置也是如此。
- **发现 L40S 显卡 Bug**：Expandable segments 可能是一个边缘案例，且 L40S 的使用并不广泛，NCCL 最近在 SM90 下禁用了 FP8 reduction。
   - 一位成员建议检查硬件规格，并在必要时跳过 expandable segments。
- **请求 Reward Modeling RFC 反馈**：一位成员请求对 Reward Modeling RFC 提供反馈，并建议在下次办公时间（6 月 26 日）进行讨论。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1386990196079333376)** (5 messages): 

> `NotebookLM 模型, 最新模型信息, 模型选项` 


- **等待 NotebookLM 模型更新**：一位用户询问了 **NotebookLM** 当前使用的模型以及哪个是最新版本。
   - 该用户还询问了在页面何处可以找到 **model options**（模型选项），并引用了 [此 YouTube 视频](https://youtu.be/K9bvF_CJKV8?si=Gj7Z6GfOaTRLHKx2) 和 [另一个视频](https://youtu.be/DLEKeE9pbU8?si=FVHrx6QwJKBhWTRF)。
- **检查 FAQ**：用户应查看 **Frequently Asked Questions**（常见问题解答）或 **Release Notes**（发布说明）页面以获取最新的模型信息。
   - 用户界面中可能还有一个 **dropdown**（下拉菜单）。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1386787180592234587)** (22 messages🔥): 

> `新用户选项, 分享链接功能, NotebookLM 替代方案, 音频概览生成, Vimeo 视频作为来源` 


- **新用户寻求选项！**：一位新用户询问了入门选项，一位成员通过私信提供了帮助。
   - 另一位用户表示欢迎并建议与其联系。
- **分享初始状态...而非完整上下文**：一位用户报告称“分享链接”功能仅分享初始查询状态，即 prompt 和响应 *之前* 的状态，这阻碍了全面的上下文分享。
   - 他们建议在“所有内容上添加复制按钮”作为解决方案，主张能够分享上传的源列表、prompt 和模型的响应，以便进行辩论。
- **NotebookLM 出现问题？**：一位用户报告 **NotebookLM** 对其无法正常工作，并请求替代建议。
   - 另一位用户回应称该工具对他们来说运行良好，并列出了数百个 **PDF**。
- **AI Automator 能否将 NotebookLM 音频转化为播客场景的虚拟人会话？**：一位用户询问是否可以使用 [SuperTelegram](https://supertelegram.gumroad.com/l/pwxot) 将一段 **4 分钟的 NotebookLM 音频** 转化为双主持人播客虚拟人会话。
   - 另一位用户提到，为此目的可能需要进行说话人分离（splitting speakers）。
- **达到生成限制，Prompt 丢失？**：一位用户表达了挫败感，称 **NotebookLM** 在输入长自定义 prompt *之后* 才宣布达到生成限制，并询问 prompt 是否会被保存以供稍后使用。
   - 另一位用户询问是否可以使用 **Vimeo 视频** 作为来源，但在粘贴链接时遇到了安全功能问题，促使另一位用户建议使用 [cobalt.tools](https://cobalt.tools/) 下载视频。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1386902508823122007)** (12 messages🔥): 

> `Debian 12 vs Ubuntu Jammy, Python SDK update, GPT4All official website issues` 


- **Debian 12 构建问题**：一位用户在 **Debian 12** 上进行构建时遇到困难，建议改用 **Ubuntu Jammy**，并推荐使用 [Qt SDKs](https://qt.org)。
   - 该用户记不清最终是如何搞定的，暗示他们可能尝试了 **backport 软件包**。
- **询问 Python SDK 更新**：一位用户询问 **Python SDK** 是否即将发布更新。
   - 他们开玩笑地问道：*"或者 Python 没救了？"*
- **GPT4All 网站 CPU 占用**：一位用户报告称 [gpt4all.io](https://www.nomic.ai/gpt4all) 网站存在 Bug，并且 *"占用了我 60% 的内置 GPU"*。
   - 他们链接到了 **nomic.ai** 的 GPT4All 页面，认为这是官方网站。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1387068621959860348)** (1 messages): 

> `Atom of Thought, GAIA benchmark, Agent Startup, Implementation code issues` 


- **Atom of Thought 面临灵活性丧失**：在 **Agent** (**GAIA benchmark**) 中使用 **Atom of Thought** 的实验导致其被移除，原因是预先分解（upfront decomposition）导致的灵活性丧失以及步骤间的上下文丢失。
   - 由于该论文的实现代码存在严重问题，研究人员对该论文及其作者失去了信心。
- **作者在 X 上的表现不专业**：在收到实现代码问题的通知后，作者在 X 上的回应*极其负面且不专业*。
   - 随后，他们利用这篇论文在 X 上转型为一家 **Agent** 创业公司。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1387044069447962759)** (5 messages): 

> `Ax for TypeScript, module status messages, OpenAI Issues, LiteLLM` 


- **询问 Ax 的移植版本**：一位成员提到了 **TypeScript** 版 **Ax** 的存在，以及它向 **Elixir** 和 **Ruby** 的移植版本。
- **在 Forward 方法中寻求状态更新**：一位成员询问如何在不使用 yielding 的情况下，从模块的 `forward/aforward` 方法发送状态消息，旨在捕获 `module_start_status_message` 之后的事件。
   - 另一位成员建议向 `forward` 传递一个 **callback** 来更新 UI 进度。
- **OpenAI 应用故障**：一位成员报告称 **OpenAI** 出现问题，表示*他们的应用挂了*。
   - 在使用 **LiteLLM** 的 `completion()`（参数为 `model= gpt-4o-mini; provider = openai`）时抛出了 `404 Not Found` 错误。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1386838224885780662)** (6 messages): 

> `Google's A2A, Anthropic A2A, MCP Timeouts, Chrome AI APIs` 


- **Google 慷慨地将 A2A 捐赠给 Linux Foundation**：Google 将 **A2A**（推测是某种 Google 产品）捐赠给了 [Linux Foundation](https://developers.googleblog.com/en/google-cloud-donates-a2a-to-linux-foundation/)。
- **Anthropic 将收购 A2A？**：一位成员建议 **Anthropic** 应该效仿 Google 捐赠 A2A。
   - 这是对 Google 向 Linux Foundation 捐赠 A2A 的直接回应。
- **MCP 超时问题触发**：一位成员报告称，在使用 **OpenAI Agent** 创建客户端会话时，遇到了 **MCP tool** 的**超时问题**。
   - 错误消息显示系统在等待 **ClientRequest** 响应 **5.0 秒**后超时。
- **Chrome 新的 AI API 到来**：正如 [Chrome 138](https://developer.chrome.com/blog/new-in-chrome-138?hl=en#built-in) 中宣布的那样，Chrome 正在集成一些 **AI API**。
   - 这可能会促成直接在浏览器中进行 **MCP 集成**。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1386803784847724765)** (5 messages): 

> `Certificate Timing, Course Completion, Social Media Posts for Course` 


- **证书发放日期泄露！**：完成所有作业和社交媒体发布的成员预计将在 **7 月中旬**收到证书。
   - 员工确认了发放时间表。
- **课程完成确认！**：参与者确认他们已完成所有作业和社交媒体先决条件，并询问证书发放时间。
   - 课程完成要求包括作业以及在 **Twitter** 和 **LinkedIn** 等平台上发布社交媒体动态。


  

---

### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1386844614941474917)** (3 messages): 

> `Cohere Reranker 定价，Cohere API 中的 Token 使用情况` 


- **Cohere Reranker 价格点引发关注**：一名成员询问如何降低 **Cohere Reranker** 的成本，预计会有频繁的 **1000 次调用**。
   - 另一名成员澄清了定价结构，解释说成本由文档数量和 Token 数量决定，超过 **500 个 Token** 的文档将被拆分为分块（chunks）。
- **成员称 Reranker 定价是固定的**：一名成员询问是否无论 Token 使用情况如何，**Rerank API** 调用的费用都是 **每 1000 次调用 2 美元**。
   - 另一名成员作了回应并分享了 [Cohere 定价页面](https://cohere.com/pricing#:~:text=We%20count%20a%20single%20search%20unit%20as%20a%20query%20with%20up%20to%20100%20documents%20to%20be%20ranked.%20Documents%20longer%20than%20500%20tokens%20when%20including%20the%20length%20of%20the%20search%20query%20will%20be%20split%20up%20into%20multiple%20chunks%2C%20where%20each%20chunk%20counts%20as%20a%20singular%20document.)，该页面将单个搜索单元（search unit）定义为包含最多 **100 份文档** 的查询。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1387160626887135365)** (1 messages): 

> `自我介绍、社区、技术、工具` 


- **Cohere 社区欢迎新成员**：Cohere 社区欢迎新成员加入其 Discord 服务器，并邀请他们进行自我介绍。
   - 鼓励新用户分享他们的 **公司/行业/大学**、正在研究的项目、最喜欢的 **技术/工具**，以及希望从社区中获得什么。
- **新成员渴望建立联系**：许多新成员加入了 Cohere Discord 服务器并积极介绍自己。
   - 他们表达了对社区的热情，并期待与其他成员互动以及学习新技术。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1387100447914856481)** (2 messages): 

> `开源简历匹配，兼容 Claude 的 MCP Server` 


- **Cursor 通过开源匹配功能筛选简历**：LlamaIndex 推出了一款开源的 **简历匹配 MCP Server**，用于在 Cursor 工作流中直接进行智能职位匹配，连接到 **LlamaCloud 简历索引** 和 [其他服务](https://t.co/RCKoiUccm6)。
   - 该项目由 @zhaoqili74 在内部黑客日（hack day）期间构建，旨在简化简历筛选流程。
- **发布兼容 Claude 的 MCP Server 模板**：LlamaIndex 发布了一个新的开源模板仓库，用于将 **兼容 Claude 的 MCP Server** 构建为具有完整 **OAuth 2.1 支持** 的 Next.js 应用，简化了与 [此服务](https://t.co/wtPorldMvJ) 无缝协作的远程 Model Context Protocol 服务器的创建。
   - 该模板由 @seldo 在内部黑客日期间开发，旨在简化与 **Claude** 及其他使用 Model Context Protocol 的服务的集成。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1387018749877555343)** (1 messages): 

> `FAISS 优化、向量化计算、量化 FAISS 索引、动态查询向量` 


- **向量化加速相似度计算**：一名成员通过将循环替换为 `query_embeddings @ doc_embeddings.T` 优化了余弦相似度计算，将 **1000 x 1000** 矩阵的运行时间从 **~25 秒** 减少到 **~0.04 秒**。
- **考虑在大规模场景下使用量化 FAISS**：对于超过 **1000 万次比较** 的情况，该成员计划切换到量化 FAISS 索引（如 `IndexIVFPQ`）以管理内存和延迟。
   - 该用户询问了在动态（非预索引）查询向量中使用 `IndexIVFPQ` 的注意事项，并寻求关于优化方案的反馈。
- **寻求关于 matmul 生产稳定性的建议**：发帖者寻求反馈，询问 `@` / `matmul` 在 **100 万规模** 下用于生产环境是否稳定。