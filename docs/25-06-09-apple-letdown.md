---
companies:
- apple
- openai
- langchain
- llamaindex
date: '2025-06-09T05:44:39.731046Z'
description: '**苹果**为 iOS 开发者发布了端侧基础模型，尽管其最近发表的《推理的幻象》（Illusion of Reasoning）论文因在大语言模型推理方面的方法论缺陷而遭到了强烈批评。**OpenAI**
  更新了 **ChatGPT 的高级语音模式**，带来了更自然的语音和更出色的翻译功能，并由格雷格·布罗克曼（Greg Brockman）进行了演示。**LangChain**
  和 **LlamaIndex** 推出了新的 AI 智能体和工具，包括用于软件自动化的 SWE 智能体，以及一个利用强化学习进行数据转换的 Excel 智能体。AI
  社区围绕大语言模型的推理能力展开了激烈辩论，凸显了评估方法所面临的挑战。'
id: MjAyNS0w
models:
- chatgpt
people:
- gdb
- scaling01
- giffmana
- kevinweil
title: 苹果开放了基础模型 API，但……并没有推出新版 Siri。
topics:
- on-device-ai
- foundation-models
- reasoning
- reinforcement-learning
- voice
- translation
- software-automation
- agentic-workflows
---

**我们无话可说，Tim。**

> 2025年6月6日至6月9日的 AI 新闻。我们为您检查了 9 个 subreddit、449 个 Twitter 账号和 29 个 Discord 社区（218 个频道，12496 条消息）。预计节省阅读时间（按每分钟 200 字计算）：1124 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 风格展示所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！


![](https://resend-attachments.s3.amazonaws.com/wibFpp0L0FwG9q7)


在去年 [WWDC 上 Apple Intelligence 引发热潮](https://news.smol.ai/issues/24-06-10-ainews-talaria-apples-new-mlops-superweapon)的一年后，由于 Qwen 和 Gemma 3 可能已经遥遥领先于 Apple Foundation Models，以至于没什么值得夸耀的，唯一与 AI 相关的更新是，设备端模型现在至少可以让 iOS 开发者在标准模态中使用（[文档在此](https://developer.apple.com/documentation/foundationmodels)）：


![](https://resend-attachments.s3.amazonaws.com/DBxaoxE55xzq0sR)


Siri 的延迟早在[几个月前就已有预兆](https://www.usatoday.com/story/tech/2025/06/04/apple-wwdc-2025-rumors/84017268007/)，所以没什么新鲜事，但这对于 Apple 的 AI 工程师来说仍然是重大新闻。

---

# AI Twitter 回顾

### **Apple 的《推理的幻觉》论文及其引发的抵制**

- **方法论遭到广泛批评**：Apple 的新论文被一些人戏称为[“推理的幻觉”](https://twitter.com/andersonbcdefg/status/1931821352463577482)，因其方法论而面临 AI 社区的强烈抵制。[@scaling01](https://twitter.com/scaling01/status/1931854370716426246) 发表了详细评论，认为该论文错误地将**最优路径长度**作为问题复杂度的指标，而实际上，像**汉诺塔（Tower of Hanoi）**这样的游戏尽管解法长度呈指数级增长，但其解题规则却非常简单。分析表明，模型性能下降并非因为缺乏推理能力，而是因为它们[被训练得要保持简洁并停止生成长输出](https://twitter.com/scaling01/status/1931817022926839909)，他通过花费 [20 美元的 API 调用费用来戳穿论文中的漏洞](https://twitter.com/scaling01/status/1931818332321149416)证明了这一点。许多人同意这一反驳，一位用户指出，“[撇开玩笑不谈，我认为 Apple 的这篇论文真的损害了他们的声誉](https://twitter.com/teortaxesTex/status/1931842186158756135)”。
- **社区反应与辩论**：该论文引发了关于评估 LLM 推理能力的更广泛讨论。虽然有人为论文辩护，但 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1931877956257005904) 称一种常见的反驳观点是“极度平庸的见解（midwit take）”。其他人则指出了这种情况的讽刺之处，[@gallabytes](https://twitter.com/gallabytes/status/1932125532327743868) 问道：“如果我让你完全在脑子里解决汉诺塔问题而不写下任何东西，塔要多高你才会让我滚蛋？”而 [@vikhyatk](https://twitter.com/vikhyatk/status/1931842055883645044) 则讽刺地指出：“我让一个 LLM 计算第 8 个忙碌海狸数（busy beaver number），它失败了。这证明了 LLM 无法推理。”普遍观点认为该论文的结论言过其实，这一观点得到了另一位用户的支持，他[对论文实验的反驳获得了巨大的关注](https://twitter.com/giffmana/status/1931801836052189191)。

### **新的 AI 模型、工具与特性**

- **ChatGPT 重大语音和翻译更新**：**OpenAI** 为付费用户推出了 **ChatGPT 高级语音模式 (Advanced Voice Mode)** 的重大更新，使其“[交流起来更加自然、轻松](https://twitter.com/kevinweil/status/1931476402446156084)”。**OpenAI** 的 **Greg Brockman** ([@gdb](https://twitter.com/gdb)) 在一条[热门推文](https://twitter.com/gdb/status/1931456650336141752)中展示了这一新语音模型。此次更新还提升了语言翻译能力，用户指出技术进步如此之快，以至于大多数人“[还没有意识到我们在语音交互方面已经领先 Siri 这么多](https://twitter.com/BorisMPower/status/1931732885415010763)”。
- **LangChain 和 LlamaIndex 发布新 Agent 和工具**：**LangChain** 宣布了多项新工具，包括用于[自动化软件开发](https://twitter.com/LangChainAI/status/1931743095021789361)的 **SWE Agent**，一个关于[使用 Ollama 构建本地 AI Agent](https://twitter.com/LangChainAI/status/1931758230314623435) 的教程，以及一个能够进行[带有反思性推理的网络研究](https://twitter.com/LangChainAI/status/1931410870451442063)的 **Gemini 研究助手**。**LlamaIndex** 详细介绍了其新 **Excel Agent** 的架构，该 Agent 使用[基于强化学习 (RL) 的结构理解来执行复杂的数据转换](https://twitter.com/jerryjliu0/status/1931383524902453336)，并发布了一个关于构建[从富达 (Fidelity) 年度报告中提取结构化数据的 Agent 工作流 (agentic workflow)](https://twitter.com/jerryjliu0/status/1931810929425158272) 的教程。
- **Perplexity 和 Google 增强研究工具**：**Perplexity** 正在测试其 [Deep Research 功能的更新版本](https://twitter.com/AravSrinivas/status/1931774041431712006)，并就其用于财务分析的 **EDGAR 集成**征求用户反馈。与此同时，**英国政府**正在一个名为 **Extract** 的新系统中使用 **Google** 的 **Gemini**，以便[在短短 40 秒内将复杂的规划文件转化为数字数据](https://twitter.com/GoogleDeepMind/status/1932032485254217799)，这一进展受到了 **DeepMind CEO Demis Hassabis** 的赞赏。
- **发布新的开源模型和数据集**：**Sakana AI** 发布了 **EDINET-Bench**，这是一个根据监管文件构建的[日本金融基准测试](https://twitter.com/SakanaAILabs/status/1931887596323717406)，用于测试高级金融任务。**Yandex** 也发布了 **Yambda-5B**，这是一个[大规模、匿名化的音乐流媒体交互数据集](https://twitter.com/TheTuringPost/status/1932091557127274993)，旨在辅助推荐系统研究。**Hugging Face** 正在与 **LeRobot** 合作组织“[史上最大的机器人黑客松](https://twitter.com/ClementDelangue/status/1932079865001623747)”，将在 100 个城市同步举行。

### **AI 行业与平台动态**

- **更换 AI 工具的高昂成本**：一个被广泛认同的观点是，AI 工具的快速迭代使得用户很难承诺年度订阅。正如一位用户所言，“[最好的工具变化得太快了，我很有可能在不到一年的时间内就取消并切换到另一个工具](https://twitter.com/iScienceLuvr/status/1931531199521919221)”。这反映了各公司不断发布新模型和新功能的激烈竞争格局。
- **关于 AI 意识与安全的辩论**：**Ilya Sutskever** 在多伦多大学的一次演讲再次引发了关于 AI 潜力的辩论，他表示：“[AI 能够完成人类所能做的一切的那一天终将到来](https://twitter.com/Yuchenj_UW/status/1931883302623084719)”。这引发了关于争论 AI 是否能“真正思考”是否有意义的讨论。在安全方面，一篇详细的帖子引发了担忧，认为像 **o3** 和 **Gemini 2.5 Pro** 这样的模型[极有可能具备协助制造生物武器的能力](https://twitter.com/RyanPGreenblatt/status/1931834526231339194)，并认为 AI 公司应该对其安全评估更加透明。
- **苹果 WWDC 2025 让部分开发者感到失望**：苹果的年度开发者大会收到的评价褒贬不一。一些人认为这些发布缺乏过去活动中的“魔力或惊喜”，一位开发者在回忆 **iPod mini** 时总结道：“[也许如果我们想要酷的东西，就必须自己动手做](https://www.google.com/search?q=https://twitter.com/raizamrtn/status/1932172447857659957)”。新的 iOS UI 被比作“[Windows Vista](https://twitter.com/skirano/status/1932145646963704199)”，并因其渐变效果的使用而遭到嘲讽。
- **印度作为 AI 超级大国的潜力**：**Hugging Face CEO Clément Delangue** 表达了他的信念，即“[印度可能成为 AI 超级大国](https://twitter.com/ClementDelangue/status/1931846782184497224)”，这一观点引起了广泛共鸣，并引发了关于该国在全球 AI 领域日益增长的作用的讨论。

### **技术概念与研究**

- **SaaS 与抽象的力量**：在一个被高度转发的观察中，一位用户指出：“[SaaS 之所以好，是因为在软件中维护抽象边界的唯一方法是将其交给另一家公司](https://twitter.com/EigenGender/status/1931489268490457183)。”这引发了关于软件架构以及模块化在构建复杂系统中的价值的讨论。
- **Meta-Learning 方法**：**The Turing Post** 发布了一篇关于 **meta-learning** 的科普文章，详细介绍了三种常见方法：[基于优化、基于度量和基于模型](https://twitter.com/TheTuringPost/status/1931446897904058517)。这种学习方式使模型能够通过少量示例快速适应新任务。
- **医疗领域的 RL 应用**：将 **Reinforcement Learning (RL)** 应用于医学领域存在重大机遇，但由于难以“[将医学转化为可验证的问题](https://twitter.com/iScienceLuvr/status/1931694421239902474)”，这一领域被认为尚未得到充分开发。这突显了从通用 AI 转向专业、高风险领域时面临的关键挑战。
- **构建欺诈检测的基础模型**：一位 **Stripe** 工程师关于其成功的**欺诈检测基础模型**的病毒式帖子受到了分析。分析指出，这是一个罕见的“即时胜利”，因为欺诈检测并非真正的预测问题，Stripe 已经拥有丰富的信号环境，且该任务已经实现自动化，使其成为[旧版 ML 系统的直接替代品](https://twitter.com/random_walker/status/1932046940822212827)。
- **合并 Transformer**：一位用户分享了关于如何[将两个宽度为](https://www.google.com/search?q=%5Bhttps://twitter.com/cloneofsimo/status/1931566076116324392%5D(https://twitter.com/cloneofsimo/status/1931566076116324392)) `h` [的 Transformer 模型合并为一个宽度为](https://www.google.com/search?q=%5Bhttps://twitter.com/cloneofsimo/status/1931566076116324392%5D(https://twitter.com/cloneofsimo/status/1931566076116324392)) `2h` 的模型的架构见解，方法是拼接权重并对某些投影使用分块矩阵，这引发了关于模型组合的技术讨论。

### **机器人与通用 AI 进展**

- **人形机器人已“触手可及”**：**Figure AI** 的 **Brett Adcock** 表示，“[真的感觉通用机器人已经触手可及](https://twitter.com/adcock_brett/status/1931509884484567323)”，并且有潜力交付“数百万台机器人”。他随后分享了一段[人形机器人翻转箱子](https://twitter.com/adcock_brett/status/1931850724343964116)的视频，以及一段[关于驱动最新 Helix 版本 AI 工作的深度解析](https://twitter.com/adcock_brett/status/1932192198025773371)，并强调近一半的 GDP 是人类劳动，这些劳动最终都可能实现自动化。
- **进步速度与“AI 工程师世界博览会”**：**@aiDotEngineer** 会议是一场盛会，[@swyx](https://twitter.com/swyx/status/1931552069984608486) 公布了最佳演讲者奖项。快速的发展节奏是一个核心主题，一位与会者指出，我们每年都在简化“[1-2 个非常显而易见的事物](https://twitter.com/lateinteraction/status/1931392417712021994)”——2023 年是主流聊天机器人，2024 年是 RAG，而 2025 年将是基础 RL。另一位观察者注意到，公司往往没有意识到像 **Runway** 这样的工具[已经可以执行他们正在询问的未来任务](https://twitter.com/c_valenzuelab/status/1932203777462849916)。

### **幽默与迷因**

- **关于 AI 的类人特性**：在写作中[加入细微的语法错误以赋予其“明显的真实感”](https://twitter.com/_jasonwei/status/1931467704495649165)并避免看起来像是 AI 生成的，正成为一种普遍做法。另一方面，人们也开始担心那些只会放大现有观点的“[谄媚型 AI](https://twitter.com/scaling01/status/1931373162479997268)”。
- **AI 发展的现状**：一个流行的迷因完美捕捉了当前 AI 领域的情绪：[一张标注为“AI 现状”的混乱未来城市全景图](https://twitter.com/c_valenzuelab/status/1931531136070517035)。另一个让开发者感同身受的挣扎是，总想[使用 Coding Agent 来“彻底重构他人的代码”](https://twitter.com/finbarrtimbers/status/1931569704696676637)。
- **咖啡因戒断的痛苦**：在一条引起科技圈之外广泛共鸣的推文中，[@DavidSHolz](https://twitter.com/DavidSHolz/status/1931579805184795091) 哀叹了咖啡因戒断的恐怖，引发了一场关于共同经历和“技巧”的大规模讨论。
- **旧金山印度菜的优越性**：在一个极具争议但很受欢迎的观点中，一位用户宣称“[旧金山的印度菜比印度的印度菜更好吃](https://twitter.com/Yuchenj_UW/status/1931555219558859205)”，引发了一场激烈的辩论。
- **关于现实的本质**：一位用户幽默地指出：“有趣的是，那些什么都不懂的古代人想象魔法弥漫在世界上，可以通过咒语和口令来召唤。顺便问一下，[WiFi 密码是多少来着？](https://twitter.com/jachiam0/status/1931376323609743572)”

---

# AI Reddit 综述

## /r/LocalLlama 综述

### 1. DeepSeek R1 0528 编程基准测试成就

- [**1.93bit Deepseek R1 0528 击败 Claude Sonnet 4**](https://www.reddit.com/r/LocalLLaMA/comments/1l6v37m/193bit_deepseek_r1_0528_beats_claude_sonnet_4/) ([得分: 312, 评论: 105](https://www.reddit.com/r/LocalLLaMA/comments/1l6v37m/193bit_deepseek_r1_0528_beats_claude_sonnet_4/)): **Unsloth 的 DeepSeek R1 0528 IQ1_M 量化版本（GGUF 格式，约 200GB），在拥有 224GB VRAM 的多 GPU 系统上使用 65535 token 上下文进行评估，在 [Aider Polygot 基准测试](https://aider.chat/docs/leaderboards/)中实现了 60% 的通过率，超过了 Claude Sonnet 4 的 "no think" 得分（56.4%）。基准测试详情：225 个测试用例，96.4% 的响应格式正确，9 个格式错误，6 个超时，且几乎利用了全部上下文。测试基础设施使用了混合 GPU 配置（2x RTX 5090, 1x Blackwell Pro 6000, 1x RTX 4080, 2x RTX 3090），运行 server 模式下的 llama.cpp，开启 16 线程，并使用 gguf tensor-split 进行显存平衡。参见 [结果日志](https://aider.chat/docs/leaderboards/) 和 [模型仓库](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF)。** 讨论强调 DeepSeek R1 0528 是一个自回归（“思考型”）模型，而 Claude Sonnet 4 的基准测试是在 "no think" 模式下进行的，这可能会影响公平性——开启思考模式的 Claude Sonnet 得分为 61.3%。文中还提到了 DeepSeek 的更新（改进了工具调用，修复了聊天模板），相关发布可在 [HuggingFace](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF) 查看，以及 [更新讨论](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF/discussions/7)。
    - danielhanchen 详细介绍了 DeepSeek R1 0528 的持续更新，重点是工具调用（tool calling）和聊天模板（chat templates）的改进。新的模型更新提供了原生工具调用支持，无需像其 [HuggingFace 发布讨论](https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF/discussions/7)中提到的那样自动追加 <|Assistant|>。
    - offlinesir 指出所使用的基准测试对比可能无法完全体现模型能力：DeepSeek 是针对 Claude 4 '<no think>' 基准测试进行测试的，该模式禁用了推理。相比之下，“思考”或思维链（chain-of-thought）基准测试（例如开启推理的 Claude 4 32k）会产生更高的结果——Claude Sonnet 4 在这种情况下达到了 61.3%。DeepSeek 被认为更具成本效益。
    - daavyzhu 引用了 DeepSeek 官方报告的 R1 0528 在 Aider 上的 71.6 分，该数据分享在他们的 [中文新闻更新](https://api-docs.deepseek.com/zh-cn/news/news250528) 中，表明其在编程任务中表现强劲，并在某些基准测试中定位高于 Sonnet 4。
- [**DeepSeek R1 0528 在 Aider Polyglot 编程排行榜上达到 71%（比 R1 提升 14.5 分）**](https://www.reddit.com/r/LocalLLaMA/comments/1l76ab7/deepseek_r1_0528_hits_71_145_pts_from_r1_on_aider/) ([得分: 230, 评论: 92](https://www.reddit.com/r/LocalLLaMA/comments/1l76ab7/deepseek_r1_0528_hits_71_145_pts_from_r1_on_aider/)): **根据 [官方排行榜](https://aider.chat/docs/leaderboards/) 显示，DeepSeek R1 0528 在 Aider Polyglot 编程排行榜上获得了 71% 的得分（比之前的 R1 版本提高了 14.5 个百分点）。该模型在代码生成和正确性方面表现出显著进步，特别是在多语言编程基准测试中，同时根据社区分析，它保持了较低的运营成本（完成约 70% 的基准测试仅需约 5 美元）。** 热门评论强调了该模型的显著改进——称其为足以被称为大版本更新的飞跃——并强调了其相对于 OpenAI 和 Google Gemini 等竞争对手极高的“单位美元正确率”，以及在创意写作任务中的强劲表现。
    - 多位用户强调了 DeepSeek R1 0528 的*成本效益*，其中一位表示它以“不到 5 美元的价格完成了约 70% 的基准测试”，远低于 OpenAI 的 GPT-3.5/4 (o3)、Gemini 和 Claude 等竞争对手。这表明与其他 LLM 供应商相比，DeepSeek 提供了卓越的“单位美元正确率”，使其在性能成本比（performance-per-cost）为关键指标的实际部署中成为强有力的竞争者。
    - 一位评论者指出，R1 0528 代表了如此巨大的性能飞跃（在 Aider Polyglot 编程排行榜上提升了 14.5 分）——以及在创意写作方面的重大改进——以至于“大多数其他公司都会将其称为 R2”。这不仅突显了该模型迭代的速度，也突显了其进步的规模。
    - 编程模型领域的竞争正在加剧，Google 的 Gemini Pro 06-05 被提及为另一个高性能选项。然而，用户表示基于价格和性能，他们强烈倾向于 DeepSeek R1 1.5，除非在特定需要 Gemini 最新版本的情况下。

### 2. 新型 AI 硬件与高效推理技术

- [**KVzip: 查询无关的 KV Cache 逐出 — 内存占用减少 3~4 倍，解码延迟降低 2 倍**](https://i.redd.it/bpxlu6tfnw5f1.png) ([评分: 289, 评论: 26](https://www.reddit.com/r/LocalLLaMA/comments/1l75fc8/kvzip_queryagnostic_kv_cache_eviction_34_memory/)): **该图片评估了 Qwen2.5-7B-Instruct-1M 模型在三种 KV Cache 配置下的性能：无上下文（No Context）、全量 KV Cache（Full KV Cache）以及新提出的 KVzip。KVzip 显著降低了内存占用（从基准水平降至 4.6GB），并实现了更快的解码延迟（14.1ms/token），同时在回答上下文相关问题时保持了与全量 KV Cache 相当的能力。测试针对《哈利·波特与火焰杯》中的问题进行。** 关键评论者批评了测试方法，指出模型可能在预训练阶段就已经掌握了《哈利·波特》的内容，因此结果可能无法反映真实的压缩效用。其他技术反馈注意到了一些意外的基准测试表现（例如，较小的缓存反而表现更好），并将其归因于可能剔除了干扰性的无关信息，但也警告这凸显了评估缺陷和测试选择中的疏忽。
    - 一位评论者批评了论文的评估方法，指出在知名文本（如《哈利·波特与火焰杯》）上测试得出的性能指标可能无法反映真实的压缩效果，因为像 Qwen2.5-7B-Instruct 这样的模型很可能在预训练数据中包含这些书籍。为了有意义地评估 KVzip 的压缩能力，测试应使用模型不熟悉的数据（如最近的新闻文章），以避免预训练知识的干扰。
    - 另一位评论者观察到了意外的基准测试结果：在某些情况下（如 MultiHop 测试），激进地减少 KV Cache 甚至提高了准确率（从 40% 提升到 45%），这引发了关于“长上下文退化（long-context degradation）”的推测，即过多的上下文可能会分散 LLM 的注意力。然而，这些结果的一致性受到质疑，因为查询无关的逐出并不被期望能可靠地仅保留无关信息。评估中还存在空白：未报告 fiction.LiveBench 等特定测试，而当信息逐出不完美时，这些测试可能会显示出更大的性能退化。
    - 一位用户询问 KVzip 是否适用于视觉语言模型（VLMs），因为图像特征在编码后最终会生成 KV 张量。他们询问在压缩图像衍生的 KV Cache 时是否存在特定模态的陷阱，以及此类压缩是否经过测试，这表现出对该方法通用性以及多模态架构中潜在边缘情况的技术好奇。
- [**中国开始量产三进制 AI 芯片。**](https://www.reddit.com/r/LocalLLaMA/comments/1l7dj3z/china_starts_mass_producing_a_ternary_ai_chip/) ([评分: 110, 评论: 52](https://www.reddit.com/r/LocalLLaMA/comments/1l7dj3z/china_starts_mass_producing_a_ternary_ai_chip/)): **据报道，中国研究人员已利用碳基材料实现了全球首款三进制（非二进制）AI 芯片的量产，[SCMP](https://www.scmp.com/news/china/science/article/3313349/beyond-1s-and-0s-china-starts-mass-production-worlds-first-non-binary-ai-chip) 对此进行了报道。三进制 AI 芯片使用三种状态（而非二进制的两种）处理信息，有可能提高 AI 工作负载的计算效率，例如使用类似 BitNet 的三进制量化的工作负载。这一进步可以为三进制神经网络推理实现显著的硬件加速，但由于根深蒂固的二进制软件栈，实现挑战依然存在。** 评论者质疑缺乏对具体公司的详细归属说明，并敦促对该芯片的性能主张进行独立审查。针对在实践中快速利用此类硬件的可行性，有人提出了技术怀疑，理由是现有底层软件生态系统具有深厚的二进制导向，这带来了重大挑战。
    - BumbleSlob 对三进制芯片的可信度表示怀疑，指出在获得独立验证之前应保持谨慎。他们强调了一个关键的技术挑战：整个现代软硬件生态系统，特别是固件和编译器等底层软件，都是围绕二进制逻辑构建的，这使得三进制架构的集成和编程成为一个潜在的巨大障碍。

### 3. Open WebUI 中具备推理意识的 LLM 工作流

- [**Open WebUI 中的概念图工作流**](https://v.redd.it/dzeqvwa9rv5f1) ([得分: 115, 评论: 14](https://www.reddit.com/r/LocalLLaMA/comments/1l71iie/concept_graph_workflow_in_open_webui/)): **Open WebUI 中描述的概念图工作流引入了一个推理引擎，LLM 在生成最终响应之前，会先识别并连接与用户查询相关的概念。该工作流利用一个 OpenAI 兼容代理，并流式传输一个专用的 HTML Artifact，通过将 Web UI 连接到后端工作流事件，实时可视化地展示并更新推理过程。完整的实现细节可以在 Harbor 仓库的 [concept.py](http://concept.py/) [模块](https://github.com/av/harbor/blob/main/boost/src/modules/concept.py#L135)中查看。** 评论者指出该工作流在增强 LLM 决策过程透明度和实现内省方面具有价值，但对其在实际应用中的效用以及它是否与基础推理（base inference）集成或仅是增强表示表示怀疑。一些人担心它对于生产任务来说，演示意义大于实用价值。
    - 讨论强调了用户对于概念图是集成在基础模型推理中还是作为辅助工具运行的困惑——这是一个技术上的区别，影响了在使用 Open WebUI 时推理轨迹（reasoning traces）的透明度和连贯性。
    - 一项技术批评指出了 UI/UX 方面的担忧：与文本解释相比，图形节点需要用户手动解读才能理解逻辑，且缺乏清晰的视觉层级。这引发了关于工作流效率以及技术用户重建推理路径时认知负荷的疑问。
    - 有人要求对该概念图工作流与同一开发者之前的工具进行详细对比，特别是在实现差异、预期用例及其各自的集成深度方面。

## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. 苹果“思维幻觉”论文辩论与 LLM 中的推理

- [**苹果的“思维幻觉”论文可能是企业损害控制**](https://www.reddit.com/r/singularity/comments/1l73qne/the_apple_illusion_of_thinking_paper_maybe/) ([得分: 229, 评论: 174](https://www.reddit.com/r/singularity/comments/1l73qne/the_apple_illusion_of_thinking_paper_maybe/)): **该帖子批评了苹果最近关于大语言模型（LLM）“思维幻觉”（illusion of thinking）的研究论文（WWDC 预告），声称这项工作更多是企业叙事控制而非科学进步。该论文研究了 LLM 在汉诺塔（Tower of Hanoi）和渡河（River Crossing）等经典算法谜题上的表现，断言模型在超过复杂度阈值后会失败，即使给出正确的算法也无法泛化推理，并且倾向于随着复杂度的增加而减少推理投入——这与依赖数学/代码基准测试的标准评估形成对比。作者质疑苹果关于基准测试污染（benchmark contamination）的说法，并认为所使用的问题本身就高度受污染，认为论文的定调更多是营销而非科学，尤其是考虑到苹果在 AI 领域的相对停滞以及论文发布的时间点。笔记还指出苹果在推理、基准测试设计和性能分析方面的论点存在所谓的不一致和修辞弱点，同时观察到所使用的谜题是基础的计算机科学练习。** 顶级技术评论强调：(1) 围绕该论文的社交媒体/确认偏误动态；(2) 苹果论文发现即使有明确的算法，LLM 也无法可靠地解决简单的算法问题，这是一个重要的、具有启发性的批评；(3) 推测苹果可能在为其不追求“思考”架构寻找理由。文中还将其与之前 Anthropic 质疑 LLM 传统推理能力的研究进行了对比。
    - 多位评论者强调，苹果的论文展示了当前 LLM 的严重局限性——具体表现为即使提供了明确的算法，也无法解决 7 柱汉诺塔等算法或推理任务。这表明与经典计算方法相比，LLM 在执行符号推理（symbolic reasoning）和算法应用的能力上存在重大缺陷。
    - 有人提到，包含 3 个角色的渡河问题（一个经典的、简单的图搜索问题）也难倒了 LLM，这在技术上是微不足道的，且解空间在 50 步以内——这突显了 LLM 无法可靠处理基础的状态空间搜索（state-space search）或非平凡推理任务。

- 一位用户提到了关于 Apple 拥有更先进内部模型的传闻，但认为 Apple 可能有意选择不专注于具备真正推理能力的架构，将这篇论文定位为一种先发制人的防御。这与 Anthropic 之前的研究形成了类比，该研究表明现有模型擅长从答案池中进行模式隔离（pattern isolation），但并不进行传统的“思考”或在算法层面进行泛化。
- [**为什么当目前的 AI 已经具有革命性时，还有那么多人痴迷于 AGI？**](https://www.reddit.com/r/singularity/comments/1l7b91a/why_are_so_many_people_so_obsessed_with_agi_when/) ([Score: 132, Comments: 203](https://www.reddit.com/r/singularity/comments/1l7b91a/why_are_so_many_people_so_obsessed_with_agi_when/)): **该帖子质疑了社区对 AGI (Artificial General Intelligence) 的关注，而非当前 AI 能力产生的实质性影响，并引用了最近一篇 Apple 论文，其研究结果据报道强调了这一点。作者强调了具体的近期应用：Level 4 自动驾驶、AI 表现优于医生以及从事体力工作的通用机器人，认为现有及近未来的 AI 无需 AGI 即可产生 *革命性* 影响。该帖子将 AGI 热潮（尤其是在 Silicon Valley 和风险投资领域）归因于对极端财务回报的追求，但对其更广泛的文化影响力表示怀疑。** 顶尖的技术评论将 AGI 视为与当前 AI 相比具有独特变革性的事物，强调了窄领域 AI 无法实现的场景（例如：治愈所有疾病、彻底延长寿命）。AGI 讨论的存在也被归因于该论坛对“奇点（singularity）”的关注以及对范式转移式进步的渴望。
    - 一位评论者指出，许多“革命性”的用例，例如通过提示词让 AI 根据详细规格创建一整部电视剧或电子游戏，严格来说并不需要 AGI，而是可以通过多个具有 Agent 特性的专业化 AI 来完成。这突出了完全 AGI 与协同专业化 Agent 之间的技术区别，表明在达到 AGI 之前，实现重大的创意自动化是可能的。
    - 辩论区分了通用 AGI (Artificial General Intelligence) 的雄心——能够治愈疾病或从根本上延长寿命——与当前 AI 系统虽然仍具变革性但更为局限的能力。评论者将对 AGI 的技术追求描述为实现超越现有模型所能达到的突破。
- [**为什么大家都在痴迷于那篇 Apple 的论文？显而易见，CoT RL 训练带来了更好的性能，这是不可否认的！**](https://www.reddit.com/r/singularity/comments/1l77u6t/whats_with_everyone_obsessing_over_that_apple/) ([Score: 110, Comments: 55](https://www.reddit.com/r/singularity/comments/1l77u6t/whats_with_everyone_obsessing_over_that_apple/)): **讨论集中在 Apple 的一篇论文上，该论文展示了语言模型在 Chain-of-Thought (CoT) Reinforcement Learning (RL) 训练中的核心局限性：虽然 CoT RL 模型通过利用增加的 Token 级计算和中间结构优于基准模型，但当面临需要超过约 8 个真实思考步骤的问题时，它们的推理性能就会崩溃——这表明当前架构存在硬性上限（[Apple 论文引用](https://arxiv.org/abs/2404.03370)）。值得注意的是，正如 Qwen 中所示，对高熵（不确定）Token 进行训练可以提高效率，但并不能打破这一推理限制；使用非语义占位符 Token（例如点/破折号而非逻辑轨迹）的模型比非 CoT 模型表现更好，但会回落到中等水平的性能，这表明额外计算和语义都有贡献。** 热门评论强调，Apple 的结果与其说是宣布 LLM 无用，不如说是从经验上勾勒出当前方法的边界——指出需要外部存储或符号推理来解决超过此阈值的任务。技术讨论还批评了围绕 AI 能力的在线两极分化，呼吁对这些架构限制和研究方向进行更细致的讨论。
    - Apple 的论文发现，使用 CoT RL (Chain-of-Thought Reinforcement Learning) 训练的模型表现出硬性的性能上限：一旦任务需要大约 8 个或更多真实的推理步骤，即使是最好的模型也会停止生成连贯的推理链，且准确率会崩溃。这突显了当前架构在逐步推理能力方面的固有局限，无论采用何种训练方法。

- CoT RL 取得了优于基准的结果，因为 Chain-of-Thought 结构充当了计算草稿纸（scratch pad），既提供了中间结构（语义），又提供了额外的计算量，从而提升了性能。当 CoT 步骤被语义上无意义的占位符取代时，额外的“思考时间”仍然有所帮助，但性能会下降，这表明中间推理的内容是关键，而不仅仅是计算预算。
- Apple 的研究人员还利用了一种 Token 不确定性采样策略——针对预测不确定性最高的 20% Token 进行训练——以提高效率。然而，这项技术并未突破底层的性能上限，这表明存在根本性的算法限制，可能需要全新的方法（如外部存储或符号规划）来使模型能够跨越 20 步或更多步进行推理。

### 2. OpenAI 与行业营收、基准测试及 GPU 竞赛

- [**重磅：OpenAI 年度经常性收入达到 100 亿美元，超出预期，高于去年的 37 亿美元（据 CNBC 报道）**](https://i.redd.it/uyqrtp449y5f1.jpeg) ([Score: 399, Comments: 105](https://www.reddit.com/r/singularity/comments/1l7db2u/breaking_openai_hits_10b_in_reoccurring/)): **图片描绘了一名男子在 OpenAI 标志前进行专业演示，视觉上强化了帖子关于 OpenAI 实现 100 亿美元年度经常性收入（高于去年的 37 亿美元）的公告。帖子链接到一篇 CNBC 文章，强调了这一创纪录的增长，这归功于 ChatGPT 及相关服务的广泛采用。图片的背景——舞台布置和对 OpenAI 品牌的关注——强调了在这一重大财务里程碑之后，该公司提升的市场地位和公众形象。** 评论者指出，快速的收入增长是“递归自我提升”的证据，并讨论了公众对 OpenAI 盈利能力以及未来潜在价格变动（例如，对更高订阅费的猜测）不断变化的看法。
    - 一位用户质疑达到 100 亿美元 ARR 是否意味着 OpenAI 现在实际上已经盈利，并强调由于 AI 公司通常面临持续的高额计算、研发和扩展成本，高 ARR 并不一定等同于净利润。该评论指出了在评估公司绩效时，经常性收入里程碑与实际盈利能力之间的重要区别。
- [**OpenAI 营收达到 100 亿美元 - 仍亏损数百万 - ChatGPT 增长惊人**](https://www.reddit.com/r/ChatGPT/comments/1l7dcjw/openai_hits_10b_revenue_still_loosing_millions/) ([Score: 240, Comments: 61](https://www.reddit.com/r/ChatGPT/comments/1l7dcjw/openai_hits_10b_revenue_still_loosing_millions/)): **据 CNBC 报道，OpenAI 的年度经常性收入（ARR）已达到 100 亿美元，是去年的两倍，主要来自 ChatGPT 消费者订阅、企业销售和 API 使用。该公司声称拥有** `500M weekly users` **和** `3M+ business customers`**，但仍处于巨额亏损状态：过去一年亏损约** `$5B`**（不包括巨额的 Microsoft 许可收入），并设定了到 2029 年实现 1250 亿美元 ARR 的宏伟目标。评论者强调的核心技术挑战是随着使用量的增长，计算和基础设施成本的不可预测扩展；OpenAI 目前的财务策略专注于市场领导地位，而非短期盈利。** 评论指出，OpenAI 巨大的财务亏损是夺取 AI 市场主导地位的刻意策略，技术辩论集中在大型 AI 部署的成本结构和可扩展性的不确定性上。
    - OpenAI 的运营成本（特别是像 ChatGPT 这样的大型模型）将如何随着用户数量和使用量的增加而扩展，目前还存在不确定性。与传统的 SaaS 不同，生成式 AI 的推理成本仍然很高，硬件限制在持续支出中起着重要作用。
    - OpenAI 目前的策略似乎是将市场主导地位置于即时盈利之上。该公司正在重新投入收入，专注于用户获取和快速产品改进，这表明其意图是建立技术和数据护城河，而不是最大化短期利润。
    - 随着使用量的激增，OpenAI 面临着模型可靠性的技术挑战，例如维持系统负载以及防止对抗性提示导致的模型退化或“损坏”。确保模型在沉重且多样化的使用下保持稳健且不被损坏是一个持续的技术关注点。

- [**Meta 的 GPU 数量与其他公司的对比**](https://i.redd.it/b5817i11ct5f1.jpeg) ([Score: 507, Comments: 157](https://www.reddit.com/r/singularity/comments/1l6toye/metas_gpu_count_compared_to_others/)): **该图片是一条推文，展示了领先 AI 玩家之间 H100 GPU 分配的对比图表，突出了 Meta 拥有 350,000 块 H100 GPU——远超 Google、Tesla 和 Anthropic 等其他公司。该图表和推文批评了 Meta 对这些 GPU 的囤积和排他性内部使用，引发了对潜在利用不足的担忧，并主张进行监管以防止私人囤积国家级算力资源。评论者注意到 Meta 对 AI 开发的内部关注、模型质量改进的多变且快速的节奏，并辩论 Meta 是否能利用其规模在 AI 竞赛中赶超，尽管此前发布的公共模型表现平平。** 讨论集中在 Meta 囤积算力并将其用于内部的策略是否能为其未来的主导地位奠定基础，用户指出了模型质量之前的快速转变（如 Llama 3.3），并警告不要因为过去的表现而忽视 Meta。一些人辩论了 Meta 仅限内部的方法与更开放或更具商业参与度的竞争对手相比，其明智性和有效性如何。
    - Meta 的 AI 策略侧重于内部使用，而不是直接与公共消费级模型竞争，Zuckerberg 的声明以及 Meta 向闭源模型部署的转变证明了这一点。这与他们之前发布的 LLaMA 等开源模型有显著不同，一些人推测，如果 Meta 的内部努力取得成功，其在 AI 领域的领先地位可能让竞争对手难以挑战。
    - 有关于 Meta LLaMA 模型快速演进的讨论：早期迭代（3.2 之前）被认为表现欠佳，但据报道 LLaMA 3.3 的性能已接近开源领域的 state-of-the-art (SOTA)。这种快速进步被拿来与 Google 的轨迹进行比较，指出曾经被认为落后的公司可以迅速在模型质量上赶超。
    - 提到了 Meta 在硬件上的投资，引用了他们购入“350K H100s”（庞大的 GPU 数量），但 Llama 4 的预期表现却受到了批评。批评者质疑相对于模型质量，这项硬件投资的回报如何，认为仅靠算力并不能保证卓越的结果。

### 3. AI 编程基准测试、Claude 与 Gemini 协作以及用户反馈

- [**aider polyglot 编程基准测试的新 SOTA - 带有 32k thinking tokens 的 Gemini。**](https://i.redd.it/d41kmdpwnw5f1.png) ([Score: 224, Comments: 32](https://www.reddit.com/r/singularity/comments/1l754k9/new_sota_on_aider_polyglot_coding_benchmark/)): **该图片展示了来自“aider polyglot coding benchmark”的对比排行榜，强调“gemini-2.5-pro-preview-06-05 (32k think)”模型以 83.1% 的正确率和 49.88 美元的成本实现了新的 state-of-the-art (SOTA) 分数。表格将 Gemini 的最新模型与 o3 (high) 等其他版本进行了对比，展示了在准确性和成本效益方面的提升，尽管数据表明较新的 Gemini 版本成本正在上升。提供了官方排行榜和公告推文的链接，以供详细对比和背景参考。** 热门评论讨论了 Gemini 在现实编程实用性中观察到的差异（例如在 Cursor 中的工具使用），指出其强大的基准测试表现可能无法转化为实际的可靠性。此外，还有关于 Gemini 成本增加缩小了其传统价格优势的技术辩论，以及关于温度设置等测试参数的询问。
    - 一位用户质疑 Gemini 的性能一致性，强调虽然该模型在 aider polyglot coding benchmark 上获得了最高分，但在 Cursor 环境中*经常无法使用工具和进行基础编辑操作*。这引发了对该模型实际可用性与其基准测试表现之间差距的担忧。
    - 关于成本和趋势的讨论：Gemini 模型此前以同等性能下*便宜 10 倍*著称，现在*仅比 OpenAI 的 o3 便宜 2 倍*，其竞争力在增强但价格差距在缩小。如果这种成本趋势持续下去，预计将与竞争对手的价格趋同。
    - 一项详细分析指出，扩展到 *32k thinking tokens* 仅增加 `$4.28` 的成本，但能*减少 19% 的任务失败*，这表明除非推理时间至关重要，否则这是一个非常值得的选项。此外，Gemini 2.5 Pro 被认为无论 thinking tokens 预算多少都很冗长，这表明冗长是模型本身的特征，而不是 token 分配的功能。

- [**Claude Code + Gemini Pro：两个 AI 编程助手协同工作**](https://www.reddit.com/r/ClaudeAI/comments/1l73a1x/claude_code_gemini_pro_two_ai_coders_working_as/) ([Score: 285, Comments: 108](https://www.reddit.com/r/ClaudeAI/comments/1l73a1x/claude_code_gemini_pro_two_ai_coders_working_as/))：**一个新的 MCP server ([gemini-mcp-server](https://github.com/BeehiveInnovations/gemini-mcp-server)) 允许 Claude Code 和 Gemini 2.5 Pro 在代码生成和审查任务上进行协作。该工作流由 Claude Code 发起分析和规划，而 Gemini 利用其 1M-token context window 和深度推理来增强和完善 Claude 的输出，从而产生可衡量的改进（例如，在协作审查和优化后，目标库中的 JSON 解析速度提高了** `26%` **）。该服务器增加了对扩展上下文、file I/O、全库代码审查、调试以及基于性能基准的迭代测试的支持，工作流涉及结构化的 prompt engineering，以便在两个 LLM 之间交替进行推理和验证阶段。** 一位评论者质疑潜在的 prompt 交互效应，想知道指示 Gemini “深入思考（think deeper）”是否会干扰 Claude 在协作过程中的独立推理。目前还没有深层次的技术争论，但对多 LLM 编排（multi-LLM orchestration）有明显的兴趣。
    - 一位评论者描述了一个利用 **Gemini 2.5 Pro** 进行高层系统架构和规划（由于其更大的 context window），然后使用 **Claude 3 Opus (o3)** 进行详细实现和 bug 修复的工作流。他们指出，在两个模型之间迭代传递输出和批评（即模型交错）可以实现更有效的故障排除，*“两个模型配对产生的结果比单个模型更好。”* 这种方法本质上是针对复杂的代码任务协同利用每个模型的优势。
    - 提出了一个关于 prompt engineering 的技术问题：具体来说，指示 **Gemini Pro** “深入思考”是否会干扰 **Claude Code** 的推理过程，因为 prompt 词汇 “think” 可能会影响 prompt 解析或模型行为。
    - 另一位评论者将此工作流与 **Aider as an MCP** 等工具进行了比较，用于在编码中集成多个提供商。他们询问 **Gemini** 是否可以直接提供代码 diffs 供 **Claude** 应用，或者 Gemini 是否仅仅充当头脑风暴助手，暗示了 API 级集成的重要性，并区分了模型之间的设计自动化与动手代码协作。
- [**我暂时不用 ChatGPT 了**](https://www.reddit.com/r/ChatGPTCoding/comments/1l6rwsi/im_done_with_chatgpt_for_now/) ([Score: 102, Comments: 93](https://www.reddit.com/r/ChatGPTCoding/comments/1l6rwsi/im_done_with_chatgpt_for_now/))：**用户报告称，OpenAI 的 “o4 mini high” 模型在复杂的脚本编写任务中表现不佳，连续几天未能产生完整或准确的结果。相比之下，Google 的 Gemini 模型成功生成了一个 1,500 行的脚本，在没有针对所需修复进行特定指导的情况下解决了问题。关键技术投诉集中在代码生成的模型退化（model regression）和幻觉（hallucination）问题上。模型选择以及与任务复杂性的匹配被强调为结果质量的关键。** 评论指出 “o4 mini high” 并非为复杂编程而设计，强调其设计目标是快速、简单的任务；建议将 o3 模型用于复杂代码。多位用户强调了 model-agnostic（模型无关）的重要性，并根据每个项目使用表现最好的模型，同时注意成本和结果质量。关于幻觉法律引用和无法从提供的 PDF 中提取细节的报告进一步削弱了对 o4 mini high 处理精确任务的信任。
    - 多位评论者强调 o4-mini-high 不适合复杂的代码或详细的技术/研究任务。一位用户强调，名称中的 “mini” 标志着该模型针对速度和简单查询进行了优化，而不是深度推理或复杂输出，建议在这种情况下使用 o3。
    - 针对 o4-mini-high 的幻觉问题提出了批评，特别是在法律引用和引用同行评审论文等专业和学术背景下。观察到该模型会生成不存在的判例法，并且在摄入以研究为导向的 PDF 后也无法准确提取信息，而 “frontier models” 在这些领域的表现更好。
    - 提供了一个技术工作流建议：开启新对话可以帮助重置对话上下文（context），防止之前的失败对当前结果产生偏见——这对于公平地进行基准测试或评估跨会话的模型一致性非常重要。

---

# AI Discord Recap

> 由 Gemini 2.5 Pro Exp 生成的摘要的摘要的摘要
> 

**Theme 1: 前沿模型开发与性能对决**

- **Unsloth 通过原生 Tool-Calling 和卓越的量化技术增强 DeepSeek-R1**：Unsloth 的 **DeepSeek-R1-0528-Qwen3-8B-GGUF** 模型现在支持 **原生工具调用 (native tool calling)**，在 **BFCL (Berkeley Function Calling Leaderboard)** 上达到 **93%** 的得分，并受益于 **UTF-8 聊天模板修复**。**Deepseek R1 0528** 的 **IQ1_M 量化**版本（200GB）表现出卓越的性能，在 Aider 的 Polygot 基准测试中可能与完整版 R1 媲美，其**成功率为 57%**，且 **100% 为格式良好的响应**。
- **Gemini 和 Claude 挑战 OpenAI 的主导地位，用户开始寻找替代方案**：多个 Discord 社区的用户报告称，**Google 的 Gemini**（尤其是用于推理的 **Gemini Pro** 和拥有 100 万 token 上下文窗口的 **Gemini 2.5 Pro**）以及 **Anthropic 的 Claude**（凭借 **Claude 4.0** 在创意写作和编程方面表现出色）正日益受到青睐，超过了 **OpenAI** 的产品。讨论强调了 **Gemini** 更大的上下文窗口和相当的性能是其核心优势，OpenRouter 的一些用户表示，除了 **GPT-4o-mini** 之外，他们已经弃用了 OpenAI 模型，因为 *Gemini Pro 在推理、思考和超长思维链方面似乎无懈可击*，而 *Claude 在创意写作方面表现卓越*。
- **NVIDIA 的 Nemotron 和苹果对 AI 推理的审查引发辩论，同时 Sydney 人格表现出色**：根据 Unsloth AI 的讨论，**NVIDIA 的 Nemotron-Research-Reasoning-Qwen-1.5B** 成为处理复杂推理的顶级 **1.5B 开源权重模型**；而苹果的 [《思维的幻觉》(The Illusion of Thinking) 论文](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf) 通过指出领先的 AI “推理”模型主要执行模式记忆而引发了辩论。与此同时，LMArena 社区发现 **Sydney 数据集** 显示 **OpenAI 的 Flash 2.5** 在模仿 **Bing Sydney** 人格方面表现出色，超过了在五条消息后就难以为继的 **GPT-4.5**。

**主题 2：开发工具与基础设施：AI Ops 的基石**

- **OpenRouter 调整费用，用户要求模型版本化以保持理智**：OpenRouter 简化了其平台费用，取消了 Stripe 上固定的 **$0.35** 费用，将非加密货币支付设置为 **5.5%**（最低 **$0.80**），加密货币支付设置为 **5.0%**（无最低限额），而新的 BYOK 订阅模式引发了用户讨论。用户还要求通过类似于上游供应商的版本化 ID 来实现更好的模型管理，以便跟踪更新并避免意外。
- **LM Studio 用户关注服务器替代方案，VLLM 用户渴望 GUI**：**Ubuntu Server** 用户讨论了绕过 **LM Studio GUI** 直接使用 **llama.cpp** 或 **Ollama** 的方法，因为 LM Studio 主要是一个服务器环境的封装器。对于更高级的设置，许多因查询并行化能力而使用 **VLLM** 的用户表达了对类似 LM Studio 的 GUI 的强烈渴望，以便管理标志和参数，将命令行参数转化为用户友好的复选框。
- **Agent 与工具协议争夺霸权，LlamaIndex 拥护 MCP**：**模型协作协议 (Model Collaboration Protocol, MCP)** 正在积极开发中，出现了如 [robcerda 开发的侧重安全的 Google MCP 服务器](https://github.com/robcerda/google-mcp-server) 等新服务器，以及 [Gist 上的 MCP 规范实用工具](https://gist.github.com/hesreallyhim/d974990b8c80cf6f32b88bfe39b76f9a) 等工具。**LlamaIndex** 举办了关于 MCP 的办公时间答疑，并在 [YouTube 上的 MCP 开发者峰会演讲](https://www.youtube.com/watch?v=kqB_xML1SfA) 中强调了 **13 种不同的协议**（包括 **MCP, A2A, ACP**）正在争夺 Agent 与工具通信的标准地位。

**主题 3：硬件与优化：榨干每一个 FLOP**

- **双 GPU 困境与量化优势成为硬件焦点**：LM Studio 中的讨论强调了 **dual GPU setups** 面临的挑战，重点关注 **PCIe lane splitting** 及其对性能的影响，并建议将 [Hardware Corner 上详细介绍的配备 VLLM 的两块 RTX 5060 Ti 16GB 显卡](https://www.hardware-corner.net/guides/dual-rtx-5060-ti-16gb-vs-rtx-3090-llm) 作为 RTX 3090 的替代方案。**Unsloth** 为 **Deepseek R1 0528** 推出的 **IQ1_M quant** 以其强劲的性能令人印象深刻，同时用户通过在 LM Studio 中开启 **flash attention** 和 **Q8** 级别的 **KV cache** 来优化 **Qwen3 models**。
- **Apple Silicon 凭借 DeepSeek R1 大显身手，内存带宽依然是核心**：**DeepSeek R1** 在配备 **512GB** 统一内存的 **Apple M3 Ultra** 上运行良好，展示了 Apple 统一内存系统在 AI 推理方面的优势，详见 [这篇 /r/LocalLLaMA Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1j9vjf1/deepseek_r1_671b_q4_m3_ultra_512gb_with_mlx/)。然而，各社区的用户纷纷感叹，**memory bandwidth** 仍然是关键瓶颈，对于许多 LLM 任务来说，这往往比 NPU 的开发更为紧迫。
- **内核级优化与编译器突破预示着极速体验**：**TinyGrad** 用户报告称，使用 `FUSE_ARANGE=1` 后，张量索引速度提升了 **10倍**；同时 **Mojo 🔥** 社区对 [LLVM Discourse 上讨论的更快的 LLVM 后端 TPDE](https://discourse.llvm.org/t/tpde-llvm-10-20x-faster-llvm-o0-back-end/86664) 感到兴奋，这是一个新的 **LLVM backend**，承诺比 **LLVM O0** 快 10-20 倍。在 GPU MODE 中，fanwenjie 分享了 **MLA-decode** 和 **FP8-mm** 的解决方案，在 **MI300** 上实现了 **3.92 ms** 的性能，可在 [其 gitee reference-kernels 仓库](https://gitee.com/fanwenjie/reference-kernels) 中获取。

**Theme 4: AI 视野的扩展：新颖应用与伦理前沿**

- **NotebookLM 制作有声读物 & ChatGPT 涉足 BIOS 补丁**：Notebook LM 的用户发现它可以生成长篇有声读物（据报道有一本 **82 分钟的有声读物**）以及令人印象深刻的播客片头，例如 [这个用户生成的 Ummmmmmmmm.mp3](https://cdn.discordapp.com/attachments/1124403655819415592/1381730009601015890/Ummmmmmmmm.mp3)。在 Yannick Kilcher 服务器讨论的一项更具技术性的成就中，**ChatGPT** 成功修复了一个 **BIOS binary**，正如 [Hackaday 关于 ChatGPT 修复 BIOS 的文章](https://hackaday.com/2025/06/07/chatgpt-patched-a-bios-binary-and-it-worked/) 和 [相关 YouTube 视频](https://www.youtube.com/watch?v=8JuWdXrCmWg) 所强调的那样。
- **IBM 发布 Responsible Prompting API，量化透明度呼声渐高**：IBM 在 GitHub 上推出了开源的 [Responsible Prompting API](https://github.com/IBM/responsible-prompting-api)，用于在推理前引导 LLM 输出，该工具基于 [他们关于负责任提示的 arXiv 论文](https://arxiv.org/abs/2504.08757) 和 [HuggingFace 演示](https://huggingface.co/spaces/responsible-prompting/responsible-prompting-demo)。与此同时，Latent Space 社区（可能参考了 [_xjdr 对 Claude 的推文回复](https://x.com/_xjdr/status/1931068996092334274)）主张 AI 服务提供商应公开模型的 **quantization levels** 和动态调整情况，并要求建立 [TheAhmadOsman 的 X 帖子](https://x.com/TheAhmadOsman/status/1930944597464654272) 中详述的可验证推理行业标准。
- **AI 通过 NoteTube 改变学习方式并挑战高难度外交游戏**：开发者正在创建像 **NoteTube** ([访问 NoteTubeAI.com](https://www.notetubeai.com/#howitworks)) 这样的工具，将 **YouTube** 转变为结构化学习平台；而 Yannick Kilcher Discord 中的其他成员则开源了一个 [由 alxai_ 在 X 上分享的 AI Diplomacy 框架](https://x.com/alxai_/status/1930653096071635112)，用于让 LLM 玩复杂的策略游戏。Perplexity AI 继续作为精选信息的来源，用户分享了关于 [俄罗斯通过 Perplexity 向 Musk 提供庇护](https://www.perplexity.ai/page/russia-offers-musk-asylum-KjWIaYM3R6iarn85k2CaAA) 等话题的页面。

**Theme 5: 社区驱动的创新与开源生态系统的繁荣**

- **开放数据集与研究推动协作式 AI 进步，“Common Pile”引发讨论**：Eleuther 社区讨论了 **Common Pile** 数据集的命名，并计划发表一篇将其与 **Llama 3** 进行对比的论文，同时分享了关于 [openMaMMUT-L/14 的研究，这是 JJitsev 在 X 上详细介绍的一个语言-视觉模型](https://x.com/JJitsev/status/1931569060438737161)，该模型在 **DataComp-1.4B** 上训练。Unsloth 中分享的一篇来自 [arXiv 的关于 packing contamination 的新论文](https://arxiv.org/abs/2410.08081)表明，污染*反直觉地实际上略微提升了下游评估效果*。
- **“Awesome Agent Learning”与 GPU Kernels 展示开源精神**：一位 HuggingFace 成员分享了一个精选的 AI/LLM agent 资源列表，[GitHub 上的 Awesome Agent Learning](https://github.com/artnitolog/awesome-agent-learning)，鼓励大家贡献。在 GPU MODE 中，fanwenjie 在[他们的 gitee reference-kernels 仓库](https://gitee.com/fanwenjie/reference-kernels/)公开披露了他们的 **MLA-decode** 和 **FP8-mm** kernel 解决方案，在 **MI300** 上实现了 **3.92 ms** 的性能，并在 [Bilibili](https://www.bilibili.com/read/cv41954307) 上发布了详细的中文说明。
- **用户联合报告 Bug 并跨平台请求功能，将挫败感转化为进步**：Unsloth 用户协作诊断了一个与 autofill 服务相关的 **Android Chrome 崩溃**问题（可通过[此 CodeSandbox 示例](https://ygdzmg.csb.app/)复现），而 TinyGrad 用户报告了 **Metal compiler bugs**，由于驱动问题，将*美丽的 mnist 变成了美丽的 macos dos poc*。功能请求层出不穷，从 GPT4All 的 **save-chat 功能**到 OpenRouter 的**模型版本控制**，以及类似于 LM Studio 的 **VLLM GUI**。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Samsung Pro 代码泄露并被滥用**：一个泄露的 **Samsung** 促销代码（可兑换 **1 年 Perplexity Pro 订阅**）被滥用，导致该代码被禁用（[截图](https://cdn.discordapp.com/attachments/1047649527299055688/1381364911556530286/Screenshot_20250608_210952_Chrome.png)）。
   - 据报道，Perplexity 团队正在努力撤销滥用者的访问权限，并为合法用户寻找解决方案。
- **API 结果劣于 Web UI，用户感叹**：一位用户表示，经过多次测试，**Perplexity API** 调用返回的结果比 **Web UI** 差得多且不完整，API 平均产生 **2-3 个引用**，但对 UI 进行相同的查询会返回 **10 个以上引用**。
   - 由于 API 的局限性，该用户表示需要使用 **Brave**、**Tavily** 或 **Firevrawl** 构建 research agent。
- **Memory 功能现已向所有人开放**：一位团队成员宣布，Memory 功能现在对所有 **Free** 和 **Pro** 用户开放，不再需要测试人员。
   - 用户可以在 [Perplexity Personalize 账户设置](https://www.perplexity.ai/account/personalize)中找到 Memory 功能。
- **Silksong 发布引发猜测**：在一次游戏展示中，ROG 的“广告”提到了 **Silksong**，引发了用户对其发布的猜测。
   - 尽管 **Nintendo** 已经预告过这款游戏，但该广告重新燃起了*今年*发布的希望，引发了关于潜在新玩法揭晓的讨论。
- **俄罗斯向 Musk 提供庇护**：一位成员链接到了一个关于**俄罗斯**向 **Elon Musk** 提供庇护的 [Perplexity 页面](https://www.perplexity.ai/page/russia-offers-musk-asylum-KjWIaYM3R6iarn85k2CaAA)。
   - 一位成员还分享了一个关于**宇宙最大地图**的 [Perplexity 页面](https://www.perplexity.ai/page/largest-map-of-the-universe-co-lvRe2dwTS2ixrAzcHa6nGQ)。

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Sydney 数据集暴露 GPT-4.5 弱点**：用户创建的 **Sydney 数据集**（包含保存的对话和 **Bing 指令**）显示，**Flash 2.5** 在模仿 Sydney 方面表现出色，而 **GPT-4.5** 在五条消息之后就难以维持该人设。
   - 在没有指令的情况下，**GPT-4.5** 的行为类似于 **4o**，这与 Flash 2.5 的反应截然相反。
- **Titan 是基础设施，而非即将推出的模型**：关于 **Titanforge** 发布日期的询问得到了澄清：**Titan** 指的是基础设施，而不是模型代号。
   - 该信息被认为是相对安全且公开的。
- **Grok 3.5 热度高涨，Kingfall 带来的失望感仍在**：对于 **Grok 3.5** 可能发布的讨论充满热情，并将其与 **Kingfall** 的表现进行了对比。
   - **Grok UI** 上的魔法粒子特效暗示发布在即。
- **苹果的 AI 收购：FTC 审查？**：关于 **苹果潜在收购**（尤其是 **Anthropic**）的猜测不断，但监管障碍也显而易见。
   - 一位成员断言 *苹果在没有 FTC 介入的情况下尝试收购 Anthropic 的可能性为 0%*，而另一位成员分享了 [一名苹果工程师在开发神经网络引擎时玩俄罗斯方块](https://x.com/TheGregYang/status/1929055508675096970) 的视频。
- **Vision Pro 高昂的价格：合理吗？**：Discord 用户对 **Vision Pro** **$3500+** 的价格展开辩论，权衡其先进技术与成本。
   - 一些人认为仅两块 micro-OLED 屏幕的成本就超过 **$800**，但也有人质疑，考虑到其生态系统和大众市场普及的潜力，与 **Meta Quest** 等更便宜的替代品相比，它是否提供了足够的独特功能。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Ubuntu Server 建议跳过 LM Studio GUI**：成员建议在 **Ubuntu Server 24.04.2 LTS** 上绕过 **LM Studio**，如果不需要 GUI，建议直接使用 [llama.cpp](https://github.com/ggerganov/llama.cpp) 或 **Ollama**。
   - 共识是对于服务器环境应直接使用源程序，因为 **LM Studio** 本质上是 **llama.cpp** 的 GUI 封装。
- **RooCode API 端点异常**：一名用户报告了 **LM Studio** 中实验性 **RooCode** 功能的问题，在调用 **/api/embeddings** **OpenAI API** 时遇到了 *unexpected endpoint or method* 错误。
   - 用户怀疑是对输入长度或 JSON 大小有限制，并指出他们的自定义脚本可以工作，而 **RooCode** 失败了，这表明 RooCode 可能指向了一个不存在的端点。
- **Flash Attention 助力 Qwen3 模型**：用户讨论了如何在 LM Studio 中为 **Qwen3-4B** 和 **Qwen3-8B** 模型确定最舒适的最大上下文 Token 窗口，以平衡对话长度和生成速度。
   - 建议是监控 GPU 显存使用情况，增加上下文长度直至 VRAM 接近填满，并启用 **flash attention** 和 **Q8** 格式的 **KV cache** 以优化 VRAM 使用。
- **双 GPU 困境：PCIe 通道限制**：讨论围绕设置双 GPU 展开，关注 **PCIe 通道拆分** 及其对性能的影响，特别是消费级 CPU 的通道数少于服务器 CPU。
   - 成员分享说，对于双 RTX 3060 配置，第二个插槽将以 x4 3.0 运行，并建议使用 VLLM 的 [两块 RTX 5060 Ti 16GB](https://www.hardware-corner.net/guides/dual-rtx-5060-ti-16gb-vs-rtx-3090-llm/) 显卡可能以更低的成本提供与 RTX 3090 相当的性能。
- **渴望 VLLM GUI 增强**：成员发现 VLLM 的查询并行化能力在服务多用户或运行多个 Agent 链时非常有效，并有人提供了 [这个命令行示例](https://link.to/example-vllm-command)。
   - 许多人渴望有一个类似于 LM Studio 的管理 GUI，可以将参数标志转换为带有说明的复选框，并提供将参数保存到 JSON 的机制。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 挑战 GPT 的统治地位**：部分成员认为 **Gemini** 的表现优于 **GPT**，原因在于其更大的上下文窗口（**1 million tokens**）和相当的性能，使其成为更具性价比的替代方案。
   - 一位用户表示，他们再也回不去 **32k tokens** 的时代了。
- **Apple Silicon 运行 DeepSeek R1 表现出色**：**DeepSeek R1** 在配备 **512GB** 统一内存的 **Apple M3 Ultra** 上运行良好，凸显了 Apple 统一内存系统在 AI 推理方面的优势。
   - 一位成员分享了 [一篇 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1j9vjf1/deepseek_r1_671b_q4_m3_ultra_512gb_with_mlx/)，同时也希望有更好的内存速度、GPU 和软件。
- **GPT 反馈系统被怀疑是幻觉**：关于 **ChatGPT 反馈系统** 的功能性出现了疑问，成员们认为这可能是一个 **幻觉 (hallucination)**。
   - 一位成员建议在 [Builder Profile 设置](https://chatgpt.com/#settings/BuilderProfile) 中启用反馈电子邮件作为替代方案，因为 GPTs 中没有内置的反馈系统。
- **Markdown 被宣布为最佳数据格式**：成员们主张将 **markdown** 作为模型训练的首选文件格式，因为它具有结构化文本和独特的 tokens。
   - 虽然 PDF 可以使用，但并不理想，因为 *没有结构的纯文本效果很好且更受青睐*。
- **YouTube 视频短文引导聊天机器人性格**：ChatGPT 可以分析带有字幕的 [YouTube 视频短文](https://chatgpt.com/blog/new-video-understanding-capabilities)，以引导 AI 回复的声音、语气或性格特征，重点关注语音模式和行为。
   - 一位成员指出，聊天机器人可能会产生 *幻觉*，且模型对于 *YouTube 内容 URL 会报错*，除非下载下来；因此下载视频有效，但链接 YouTube URL 不行！

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek-R1-0528 获得原生工具调用功能**：Unsloth 的 **DeepSeek-R1-0528-Qwen3-8B-GGUF** 模型现在支持 **原生工具调用 (native tool calling)**，在 BFCL (Berkeley Function Calling Leaderboard) 上达到了 **93%**。
   - 该更新还解决了 `add_generation_prompt` 的问题，并包含了 **UTF-8 聊天模板修复**，官方 DeepSeek 模型也从中受益。
- **输入导致 Android Chrome 崩溃**：用户报告称，由于与 **自动填充服务 (autofill service)** 相关的错误，在 **Android Chrome** 的文档编辑器中输入会导致浏览器崩溃。
   - 该问题与 Chrome 与自动完成服务的交互有关，在通知这些服务文档更改时会触发 `TransactionTooLargeException`，如 [此处](https://ygdzmg.csb.app/) 所示。
- **IQ1_M 量化以稳健的性能令人惊叹**：Unsloth 为 **Deepseek R1 0528** 提供的 **IQ1_M 量化** (200GB) 表现异常出色，在 Aider 的 Polygot 基准测试中可能与原始完整版 R1 持平，具有 **57% 的成功率** 和 **100% 格式良好的响应**。
   - 该模型在 Roo Cline 中持续工作，没有遗漏工具调用或陷入循环，表现优于其他量化版本。
- **Nvidia 的 Nemotron 在推理方面表现出色**：**Nvidia 的 Nemotron-Research-Reasoning-Qwen-1.5B** 是全球领先的用于复杂推理任务的 1.5B 开源权重模型，在广泛的任务中大幅领先 Deepseek 的 1.5B 模型。
   - 结果涵盖了广泛的任务，包括数学、编程和 GPQA，详见 [此处](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)。
- **填充污染矛盾地提升了性能**：一篇新 [论文](https://arxiv.org/abs/2410.08081) 表明，*填充污染 (packing contamination) 实际上并不重要，而且违反直觉的是，它实际上稍微提高了下游评估性能*。
   - 论文中没有完全解释清楚，但似乎 *“大”模型将具有略高的概率和更短的编码长度*。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **EXL3 Kernels 在 Transformers 中原生运行**：**EXL3** 现在可以在 **Transformers** 中运行，尽管目前仅支持 Kernels 和推理，请查看 [GitHub 上的代码](https://github.com/huggingface/transformers)。
   - 鉴于 **Transformers** 的变化，特别是围绕量化模型支持的变化，目前尚不清楚集成程度能达到何种水平。
- **实验追踪偏好**：成员们讨论了用于实验追踪的 [wandb](https://wandb.ai/site)、[neptune.ai](https://neptune.ai/)、[mlflow](https://mlflow.org/) 和 [comet.ml](https://www.comet.com/)，大多数人由于熟悉度仍倾向于使用 **wandb**。
   - 一位成员指出：“它似乎能做 **wandb** 能做的一切，只是我对 **wandb** 的使用要熟练得多”。
- **Awesome Agent Learning 发布**：一位成员分享了他精心策划的 **AI/LLM agents** 资源集合 [Awesome Agent Learning](https://github.com/artnitolog/awesome-agent-learning)，包含基础课程、阅读材料和特定框架的教程。
   - 他鼓励通过 PR 为可能遗漏的优秀资源做出贡献。
- **Transformer Attribution Graphs 实现追踪**：一位用户分享了关于 **Attribution Graphs**（归因图）的 [Transformer Circuits](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) 链接。
   - 他们承认可以“花好几个小时玩那个追踪器”。
- **GPT-4o 在解析 Agent 时遇到困难**：用户报告在使用 **GPT-4o** 和 **GPT-4o mini** 配合 *smolagents* 代码 **agent** 时，经常遇到解析错误。
   - 用户正在请求针对该问题的修复和变通方案。



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Gemini Max 在编辑时遇到困难**：用户发现虽然 **Gemini** 速度很快，但在应用文件编辑时表现挣扎，且倾向于过度分析代码，经常卡在询问不必要的问题上。
   - 尽管 **Gemini** 速度快，但由于其反复损坏文件，一些成员表示在处理复杂任务时更倾向于使用 **Claude 4**。
- **Background Agents 出现 Bug**：用户报告了 **background agents** 的问题，包括在查找 **VS Code** 远程工作区时遇到错误，以及频繁受到 **ESLint** 的干扰。
   - 成员们在尝试解决此问题时发现，在设置中禁用 **ESLint** 自动修复可能是一种变通方法。
- **Claude Code 消耗配额**：尽管 **Claude Code** 广受好评，但用户对其**速率限制**表示担忧，尤其是 **Opus 4**，导致一些人转回使用 **Claude 3.7** 以节省配额。
   - 用户正在探索诸如 [Claude Code 中的 Gemini 2.5 Pro](https://github.com/coffeegrind123/gemini-code) 之类的替代方案，以解决高成本问题。
- **Cursor Chat 在古巴受限**：一位在古巴的用户报告需要使用 VPN 才能访问 **Cursor chat**，这表明 **Cursor** 可能遭到了直接封锁。
   - 支持人员建议在设置中禁用 HTTPS，并引导用户前往[西班牙语频道](https://discord.com/channels/1074847526655643750/1367412353708331038)寻求帮助。
- **Background Agents 独立运行**：**Background Agents** 被设计为“独立的”，允许多个 **agent** 同时运行而不会产生资源冲突，从而实现更长时间的迭代和进展。
   - [Lukas Moeller](https://discord.com/channels/1152407934193432666/1367213641027551352/1380920623360409600) 强调了这种设置对同步任务的好处。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 简化平台费用！**：OpenRouter 正在简化其平台费用，取消了 Stripe 支付中 **$0.35** 的固定费用；非加密货币支付的费率将为 **5.5%**（最低 **$0.80**），而加密货币支付将为 **5.0%** 且无最低限额。
   - 虽然大多数额度的购买总费用会有所下降，但用户注意到大额购买（如 **$1,000**）的成本有所增加，从 **$52.98** 上升至 **$55**。
- **BYOK 订阅引发争论！**：OpenRouter 计划用固定的月度订阅取代 **5%** 的 BYOK 费用，这在用户中引起了不同的反应。
   - 一些用户对额外的月费表示担忧，尤其是家庭用户；而另一些用户则认为，对于拥有大量 AWS、OpenAI 或 GCP 额度且希望简化成本管理的重度用户来说，这是合理的。
- **模型管理引入版本控制**：一位用户请求 OpenRouter 像上游供应商一样为模型实施版本控制，以便更好地管理模型更新。
   - 该建议是使用保持不变的版本化 ID，同时保留始终指向最新版本的 ID。
- **Dana AI 助力互动学习！**：一名成员推出了 **Dana**，这是一个**由 AI 驱动的互动学习平台**，目前处于免费 Beta 测试阶段，访问地址为 [https://dana-ai.xyz/](https://dana-ai.xyz/)。
   - 该平台可以构建个性化课程，一位用户表示有兴趣在此基础上探索 **Excel macros**、**VBA** 和 **Power BI** 自动化领域的机会。
- **Gemini 和 Claude 与 OpenAI 争夺顶尖模型地位！**：一些成员声称，除了 **4o-mini** 之外，**Gemini** 和 **Claude** 已经完全取代了 **OpenAI**。
   - 具体而言，**Gemini Pro** 在*推理、思考和超长思维链*方面似乎无懈可击，而 **Claude** 在*创意写作*方面表现出色。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Common Pile 命名争议**：成员们就 **Common Pile** 的名称进行了辩论，创作者表示他们可能会在描述其工作的[论文](https://example.com)中使用全名，并与 **Llama 3** 进行对比。
   - 会议澄清，对比是针对在相似数据量上训练的模型进行的，不过 **Qwen 3**（8B 参数，36T tokens）也被列为使用大幅增加的数据量后性能表现的示例。
- **辩论焦点：语言还是国际象棋？**：讨论集中在仅对语言建模是否能产生高级技能，一些人认为 LLM 已经做到了这一点，并引用了**国际象棋技能**作为例子，或者通过反转问题进行 Token 生成。
   - 一个反驳观点是，**国际象棋记谱法自然地将对局转换为序列**，从而可以被 LLM 建模，尽管也有人指出语言数据对其所指代的对象进行了建模，尽管是以一种有损的方式。
- **Discord 版主封禁用户机器人 (User Bots)**：版主们讨论了 Eleuther Discord 中日益增多的用户机器人和“垃圾内容 (slop)”发布行为，一些人主张封禁，而另一些人建议要求机器人声明其自动化性质，且 [Discord 指南禁止使用用户机器人](https://discord.com)。
   - 版主正在手动删除这些帖子，并鼓励用户使用 <:delet:824412305906204692> 或 <:lurkmoar:800507348535214140> 进行表情回应，以帮助版主更轻松地进行过滤。
- **缩放定律 (Scaling Laws) 推动开源模型**：新的研究详细介绍了一种使用缩放定律推导进行开源基础模型和数据集比较的方法，并展示了语言视觉模型 [openMaMMUT-L/14](https://x.com/JJitsev/status/1931569060438737161) 的发布。
   - 该模型在来自 **DataComp-1.4B** 的 **12.8B 样本**上进行了训练，在 **IN1K** 上实现了 **80.34% 的 zero-shot** 准确率。
- **NeMo 的基准测试 Bug 虚增了 TPS**：一名成员报告称，虽然 **NeMo** 最初显示出高得多的 **TPS**，但事实证明这是一种错觉，原因是基准测试回调函数存在 Bug，未能正确处理 **GAS**，导致 **TPS** 被虚增了 **GAS** 倍。
   - 真实数据显示，优化后的 **NeMo** 运行速度比没有融合（fusions）的常规 **NeoX** 运行还要慢，因此团队切换到了 **NeoX** 并坚持在预训练运行中使用它。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 Pro 在与 Opus 的对比中表现稳健**：成员们将 **Gemini 2.5 Pro** 与 **Opus** 进行了比较，一些人认为它在某些任务中与之相当或略微领先，而另一些人则因编程能力而更青睐 **Opus**。
   - 一些人指出 **Gemini 2.5 Pro** 在理解较新版本的库方面存在弱点。
- **R1 0528 Unsloth IQ1_M 的基准测试分数令人印象深刻**：一位成员分享了 **R1 0528 Unsloth IQ1_M** 模型的基准测试结果，在 **170/225** 个测试用例中达到了 **58.2%** 的分数，且格式正确率（well-formed rate）达到 **97.1%**。
   - 讨论围绕着将此性能与 **Sonnet 4** 以及各种硬件配置进行比较展开。
- **Aider 请求集成 MCP**：一位用户请求在 Aider 中原生集成 **MCP (Model Collaboration Protocol)** 以改进代码，并参考了 **Roo** code 中使用的服务器。
   - 该用户还希望获得超出当前 **Playwright** 集成范围的功能，如 **sequential thinking**（连续思考）、**Brave search API** 和 **AI browser** 功能。
- **Sparse Attention 可能大幅提升速度**：据预测，**Native Sparse Attention** 在长上下文场景中可提供 **>12x** 的加速，在现代硬件上可能实现持续的 **100-200 TPS**。
   - 这种性能提升对于未来的模型部署具有重要意义。
- **Claude Code 令人失望**：一位拥有 **pro MAX 订阅** 并尝试了 Claude Code 的成员认为它与 Aider *没有太大区别*。
   - 他们表示，虽然 Claude Code 拥有华丽的 UX 并试图表现出 *agentic*（智能体化），但 Aider 凭借其对上下文的显式管理，感觉更像是一个 *precision instrument*（精密仪器）。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Hyper Projection 加速计算**：一位用户正在探索将数据几何化地投影到更高或更低维度的 [**hypercube** 和 **matrix projection**](https://en.wikipedia.org/wiki/Hypercube_graph)，通过压缩 k-sparse 数据来加速计算。
   - 该想法涉及将 Fourier 表示中的前 *k* 个值分配给超立方体的一个角，然后将这些点投影到 2D 空间，应用领域包括流体动力学、细胞分裂和降噪。
- **AI Diplomacy 测试框架开源**：一位用户开源了他们的 **AI Diplomacy harness**，以便让不同的 LLM 进行该游戏，并发布了超过 **15** 场比赛的数据，同时分享了[他们的帖子链接](https://x.com/alxai_/status/1930653096071635112)。
   - 他们将在接下来的几天待在旧金山，并提议见面讨论该项目。
- **NVIDIA 的 Nemotron-H 模型实现大规模推理**：NVIDIA 推出了 [Nemotron-H 推理模型系列](https://developer.nvidia.com/blog/nemotron-h-reasoning-enabling-throughput-gains-with-no-compromises/?linkId=100000368479233)，包括 **Nemotron-H-47B-Reasoning-128K** 和 **Nemotron-H-8B-Reasoning-128k**，针对具有长输出序列（高达 128k tokens）的推理密集型任务的吞吐量进行了优化。
   - 这些模型提高了处理复杂推理任务中冗长输出序列的效率。
- **苹果的《思考的错觉》引发辩论**：苹果的论文 [The Illusion of Thinking](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf) 探讨了 **LLM** 和 **LRM** 在复杂性下的崩溃，其实验设计和炒作引起了辩论。
   - 一位成员认为不应将该论文的发现过度解读为苹果的总体战略评估，而另一位成员则为论文中关于模型过拟合并在复杂性下崩溃的观点辩护。
- **ChatGPT 成功修复 BIOS**：一位成员分享了一个 [Hackaday](https://hackaday.com/2025/06/07/chatgpt-patched-a-bios-binary-and-it-worked/) 链接和一段 [Youtube 视频](https://www.youtube.com/watch?v=8JuWdXrCmWg)，内容关于 **ChatGPT** 成功修复了一个 **BIOS 二进制文件**。
   - 讨论强调了 AI 在底层系统修改方面的潜力。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **参考算子解决 MLA-decode 和 FP8-mm 问题**：一名成员在 [gitee.com](https://gitee.com/fanwenjie/reference-kernels/) 公开了关于 **MLA-decode** 和 **FP8-mm** 的解决方案，在 **MI300** 上实现了 **3.92 ms** 的性能。
   - 解决方案的详细中文说明请见 [此 bilibili 链接](https://www.bilibili.com/read/cv41954307)。
- **D-Matrix 芯片价格尚未公开**：一名成员询问了 **D-Matrix chips** ([d-matrix.ai/product/](https://www.d-matrix.ai/product/)) 的价格，一名代表暗示定价信息可能尚未公开。
   - 在讨论定价的背景下，也出现了关于 **TPUs** 等替代方案的讨论。
- **vLLM 专家随时待命**：一名深入参与 **vLLM ecosystem** 的成员提供了协助和支持，特别是围绕 **llama3.1** 和 **Qwen2** 架构。
   - 他们正在探索手动拼接算子（stitching kernels），但其他成员对这是否属于 **memory bound**（内存受限）表示担忧。
- **Async Rocm 用户需要 ATTention**：一名用户在 ROCm 中使用 **ATT plugin** 和 **rocprofv2** 进行指令延迟分析时遇到错误，AMD 员工 gumthepug 提供了帮助。
   - 其他成员提供了使用 **rocprofv2** 收集 **SQTT traces** 以在 **Radeon GPU Analyzer (RGA)** 中进行分析的指导。
- **MoE 专家路由在 Torch Compile 中受阻**：一名成员询问如何在 `torch.compile` fullgraph 模式下捕获 **MoE expert routing**（[代码片段](https://github.com/HiDream-ai/HiDream-I1/blob/main/hi_diffusers/models/moe.py#L141)），并引用了一篇[博客文章](https://pytorch.org/blog/metashuffling-accelerating-llama-4-moe-inference/)指出这可能无法实现。
   - 讨论还透露，NVIDIA 的 **GB200 NVL72** 和 **Dynamo** 将提升 **Mixture of Experts (MoE)** 模型的推理性能。



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 制作有声书，生成优质播客内容**：用户发现 NotebookLM 可以通过提示词生成有声书，例如要求它 *"阅读每个子章节、段落，对每个引用进行角色扮演，并在每章后进行总结"*，一名用户制作了一本长达 **82 分钟的有声书**。
   - 另一名用户对 NotebookLM 的播客片头印象深刻，称其能力出乎意料，并分享了生成的 [Ummmmmmmmm.mp3](https://cdn.discordapp.com/attachments/1124403655819415592/1381730009601015890/Ummmmmmmmm.mp3) 文件。
- **NoteTube 将 YouTube 转换为教育中心**：一名用户正在开发 **NoteTube** ([https://www.notetubeai.com/](https://www.notetubeai.com/#howitworks))，这款应用能将 **YouTube** 转换为结构化的学习平台，具有进度跟踪、笔记、测验和 AI chat 功能。
   - 开发者正在寻找用户测试该应用，一名用户反馈说，他喜欢*要求任何 AI 将转录文本重新格式化为博客*以提取关键点。
- **Workspace 账户默认获得自动保护**：使用合格的 **Google Workspace 或 Workspace for Education edition** 的账户会自动受到保护，免受人工审核和 AI 训练，并带有 "**PRO**" 徽章。
   - 目前非 pro/plus 账户无法使用 **Share 按钮**；尚不清楚这些功能是否相关。
- **播客长度随机性**：用户报告称，使用相同的源材料和提示词生成的播客长度不一（例如 **71 分钟、32 分钟、52 分钟**），这表明可能存在一个每日重置的隐藏长度限制功能。
   - 为了生成更长的英文播客，用户应该*不断重新生成（reroll）直到获得较长的版本*。
- **冰岛教师面临访问拒绝错误**：冰岛的一些教师在尝试使用 NotebookLM 时遇到了 "**You do not have access to this service**" 错误，这可能是由于地理限制或年龄验证不完整导致的。
   - 一名成员报告称，该问题出现在 Brave 浏览器上，但通过切换到 Firefox 得到了解决。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Teknium 完成模型合并**：Teknium 宣布最新的模型更新已完全合并，并在 [X.com](https://x.com/Teknium1/status/1931146106345529824) 上分享了这一消息。
   - 未提供更多背景信息。
- **IBM 的 API 优化 LLM 输出**：一名 IBM 实习生推出了开源的 [Responsible Prompting API](https://github.com/IBM/responsible-prompting-api)，建议在*推理前*对 Prompt 进行微调，以获得负责任的 LLM 输出，详见[这篇论文](https://arxiv.org/abs/2504.08757)和一项[用户研究](https://dl.acm.org/doi/10.1145/3706598.3713365)。
   - [HuggingFace](https://huggingface.co/spaces/responsible-prompting/responsible-prompting-demo) 上提供了 Demo，团队正在寻求社区反馈以改进价值数据库。
- **Holo-Q 利用 RL 压缩上下文窗口**：一名成员分享了来自 **Holo-Q** 的 [GitHub 项目](https://github.com/holo-q/thauten/)，该项目使用 **RL** 优化模型压缩，旨在将信息压缩至理论极限，并实现上下文窗口碎片整理。
   - 作者指出挑战包括 **vllm** 的稳定性问题，并征求对项目设计的反馈。
- **Nous 社区添加标签**：服务器现已具备标签功能，成员可以通过 *settings > profiles > server tag* 为其账户添加 **Nous tags**。
   - 未提供进一步细节。
- **Nous 筹备 Hermes-4 和数据集发布**：成员们正热切期待 **Hermes-3 数据集**的发布，同时 **Hermes 4** 也在准备中。
   - 团队正在使用 HuggingFace 上详细介绍的 [ProRL algorithm](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B#prorl-prolonged-reinforcement-learning)。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Claude 4.0 在编程任务中超越 GPT-4**：成员们就哪种 AI 模型在编程方面更出色展开了辩论，一些人认为 [**Claude 4.0**](https://www.anthropic.com/index/claude-4-haiku) 在编程、推理和数学方面表现优异，这归功于其更好的 AI 引擎和训练。
   - 然而，其他人指出 **AI arena leaderboard** 显示 **ChatGPT** 可能更适合 Web 开发，并提到 **Manus** 的代码生成能力令人失望。
- **Manus 积分神秘消失**：一名成员报告积分突然丢失，从接近 **30,000** 降至仅 **2,300**，引发了对潜在原因的猜测，如[欺诈或共享系统被利用](https://help.manus.im/en/)。
   - 社区正在寻求对此事件的澄清，并建议加强安全措施以防止未来再次发生。
- **AI 的 UI/UX 设计能力面临障碍**：成员们讨论认为，虽然 AI 可以生成基础代码，但 **UI**、**设计**和**逻辑**等复杂任务仍严重依赖人类开发者，这限制了 AI 创建完整项目的能力。
   - 对话强调了复杂的设计元素需要人类的创造力和专业知识，而这正是目前 AI 所缺乏的。
- **预测 2033 年将出现 AI 驱动生成的 GTA 8**：成员们开玩笑地预测 **GTA 8** 可能会在 *2033 年 2 月 23 日* 左右由 AI 创建，其他人也同意，假设没有全球性灾难发生，AI 开发此类复杂游戏只是时间问题。
   - 一名成员开玩笑说 [builder.ai](https://builder.ai/) 就能做到。
- **YouTube 通过反爬措施封锁 Manus 机器人**：成员报告 **Manus** 因 YouTube 的机器人检测功能而无法再观看 YouTube 视频，YouTube 正在积极修补其反爬机制。此外，由于 **Manus** 处于 Sandbox（沙箱）环境，目前无法登录 **Gmail** 账号。
   - 一名 Manus 团队成员承认这是一个技术问题，并表示*他们将尝试在本周修复*。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **TPDE 后端完胜 LLVM O0**：社区热议 [TPDE](https://discourse.llvm.org/t/tpde-llvm-10-20x-faster-llvm-o0-back-end/86664)，这是一个新的 **LLVM** 后端，承诺速度比 **LLVM O0** 快 10-20 倍。
   - 该公告发布后，成员们对利用这些速度提升表现出极大的热情。
- **Modular 论坛焕然一新**：[Modular 论坛](https://forum.modular.com/t/docs-site-new-navigational-system/1598)正在推出全新的导航系统，并积极寻求社区对这些变化的反馈。
   - 鼓励用户查看新布局并提供建议，以帮助优化用户体验。
- **社区深入探讨开发环境**：围绕 **macOS** 与 **WSL** 的开发优劣展开了辩论，强调了 **macOS** 的缺点，如缺少内置包管理器和 **Docker** 性能不佳。
   - 反方观点强调 **macOS** 是核心 **Torch** 开发者青睐的平衡环境，而其他人则指出与 **Linux** 或 **WSL** 相比，其在性能分析方面的硬件限制。
- **在 Mojo 中对参数化类型进行切片**：一位用户在使用参数化 `__getitem__` 的自定义向量时，发现了 **Mojo** 编译时切片行为的异常，并提供了一个 [代码片段](https://github.com/modular/modular/issues/4773)。
   - 随后的讨论表明，在区分编译时和运行时切片索引方面可能存在局限性，从而导致了一个围绕类型来源比较的正式 [Bug 报告](https://github.com/modular/modular/issues/4773)。
- **DumPy 引发讨论**：[DumPy](https://dynomight.net/dumpy/) 受到关注，特别是由于它提到了 **einx** 和 **torchdim**。
   - 爱好者们正在探索其对数值计算工作流的潜在影响。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **量化透明度呼声再起**：讨论线程探讨了 **AI** 服务提供商披露其模型**量化级别**并在进行动态调整时通知用户的必要性，引用了可能针对 **Claude** 的[推文](https://x.com/_xjdr/status/1931068996092334274)。
   - 社区提出了量化敏感评估和详细列出当前量化级别的公共网页等解决方案，呼吁对服务降级进行公平补偿，并制定可验证推理的行业标准，详情见[此处](https://x.com/TheAhmadOsman/status/1930944597464654272)。
- **Suno 的版权主张变得复杂**：一位成员指出，除非保持活跃订阅，否则 **Suno** 存在限制，这与“无版权、无限制”的主张相矛盾，详见 [Suno 条款](https://www.reddit.com/r/LocalLLaMA/comments/1l4mgry/chinas_xiaohongshurednote_released_its_dotsllm/)。
   - 虽然执行可能具有挑战性，但这一澄清确保了用户了解 **Suno** 当前的许可限制。
- **Linear MCP 为 Claude Code 注入动力**：用户分享了 **Linear MCP** 的集成，使任务列表和项目状态在 **Claude Code** 会话之间保持同步，该集成在本地运行并处理 **OAuth**，如[此处](https://www.task-master.dev/)所述。
   - 用户提到，“我的整个 claude.md 文件现在基本上只是一个关于如何使用 linear mcp 的系统提示”，从而实现了通过 **Linear** 任务分配触发 **GitHub** 集成与 **sub-agents**。
- **Apple 质疑 AI 推理的真实性**：**Apple** 的研究表明，像 **Claude**、**DeepSeek-R1** 和 **o3-mini** 这样领先的 **AI** “推理”模型并非真正具备推理能力，而是擅长模式记忆，如[此处](https://x.com/RubenHssd/status/1931389580105925115?s=19)分享。
   - 研究发现，即使有明确指令，模型在处理更高复杂度的问题时也经常失败，这挑战了关于 **AGI** 即将到来的炒作。
- **AI 公司误判了 LLM 过滤器的失误**：讨论线程探讨了 **AI** 公司如何误解 **LLM**，特别是在内容过滤器方面，详见[此处](https://x.com/aiamblichus/status/1931487839822254358?s=46)。
   - 用户分享了幽默且出人意料的 **LLM** 响应示例，这些响应通过轻微的 **Prompt** 调整绕过了过滤器，表明 **LLM** 遵循的是“即兴表演”规则。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 的图像上传功能极具挑战**：成员在启用 **MCPs** 的 **image uploading** 时遇到困难，包括尝试从 **Cursor's context** 提取并使用 **base64 encoding**。
   - 这一努力凸显了在 MCP 实现中集成高级功能的挑战。
- **Python 开发者思考 GitHub MCP Server 访问**：一位用户寻求使用 **Python** 访问官方 **GitHub MCP server** 以读取文件和目录的指导，并被引导至[使用 Docker 的安装说明](https://github.com/github/github-mcp-server?tab=readme-ov-file#installation)。
   - 这体现了社区对通过编程方式与 MCP server 交互以实现各种自动化任务的兴趣。
- **MCP 客户端重连引发混乱**：当 **server restarts** 且客户端使用旧的 session ID 连接时，客户端会卡在 **HTTP 400 or 404** 错误中，尽管 MCP spec 规定客户端在遇到 404 错误时 *必须* 启动新会话。
   - 该问题源于客户端**不符合规范**，导致了重连问题。
- **规范实用工具精简 MCP 文档**：一位成员创建了一个工具，用于从 MCP 的 "Specification" 文档页面提取内容，将文件大小减少了约三分之一，该工具已发布在 [Gist](https://gist.github.com/hesreallyhim/d974990b8c80cf6f32b88bfe39b76f9a)。
   - 该工具有助于开发者更高效地访问核心 MCP 文档。
- **Google 的守护者：MCP Server 强调安全性**：一位成员分享了他们的 [Google MCP server](https://github.com/robcerda/google-mcp-server)，强调其安全优先的设计，默认仅使用安全作用域。
   - 该 server 可以直接从 MCP 管理大部分 **Gmail, Calendar, and Drive**，展示了安全且实用的应用场景。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Lazy Setitem 寻求进入 TinyGrad Tensors**：一位贡献者建议将 `tensor.py` 中的 `__setitem__` 拆分为 `setitem_and_realize`，以允许延迟（lazy）、不可变且在设备上执行的操作，这可能使 [beautiful_cartpole](https://github.com/tinygrad/tinygrad/blob/master/examples/beautiful_cartpole.py) 等示例受益。
   - 为了使建议的 lazy 实现生效，需要移除当前的 `realize()` 实现。
- **TinyGrad 第 74 次会议：合并即将到来**：TinyGrad Meeting #74 涵盖了公司更新，包括对 multi 和 resnet dataloader 的修复、更快的 CI、linearizer、viz、drivers、cloud/hash、onnx 以及本地开发，还有 **lm_eval** 和 **AMD_LLVM** 等其他悬赏任务。
   - George Hotz 表示他将 *在本周合并所有内容*。
- **`lovely-grad` 适配现代 TinyGrad**：[Lovely Grad](https://github.com/xl0/lovely-grad) 在中断数月后已适配现代 **tinygrad**，并计划研究使用 pytest multiprocessing 进行远程测试。
   - 该工具有助于可视化 **TinyGrad** 中实现的神经网络梯度流。
- **Metal 编译器 Bug 触发 MacOS DOS POC**：据报告 **Metal** 存在编译器 bug，一位用户在边界问题上浪费了半天时间，促使添加了 `max_total_threads_per_threadgroup` 以解决 CUDA 的 `__launch_bounds__` 和 HIP 的 `amdgpu_flat_work_group_size` 问题。
   - 用户感到震惊，因为由于驱动程序问题，*这让 beautiful mnist 变成了 beautiful macos dos poc*。
- **FUSE_ARANGE 带来 10 倍速度提升**：通过使用 `FUSE_ARANGE=1` 上下文，一位成员展示了 tensor 索引操作 **10 倍的加速**。
   - 另一位成员询问了 `FUSE_ARANGE` 的细节及其在 `examples/hlb_cifar10.py` 中的适用性。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 宣布 Office Hours**：LlamaIndex 将于 **太平洋时间 6 月 12 日上午 8 点 / 中欧时间下午 5 点** 举行 Office Hours，重点关注 **MCP**、**表单填写** 等话题，将在 general 语音频道举行。
   - **MCP Dev Summit** 演讲现已上线，涵盖了 **13 种不同的协议**，旨在标准化 Agent 与工具之间的通信，包括 **MCP**、**A2A** 和 **ACP**，可在 [YouTube](https://www.youtube.com/watch?v=kqB_xML1SfA) 上观看。
- **Spreadsheet Agent 进入私测阶段**：**Spreadsheet Agent** 正处于私测阶段，采用“先解析，后推理” (*Parse First, Reason Second*) 的架构来理解视觉结构和上下文，详见[这篇博客文章](https://www.llamaindex.ai/blog/introducing-the-spreadsheet-agent-in-private-preview)。
   - 该 Agent 展示了 LlamaIndex 在处理具有高级推理能力的结构化数据方面的能力。
- **新视频展示 Llama Cloud**：一段新视频概述了 **Llama Cloud**，重点介绍了其生态系统和构建生产级 LLM 应用的核心工具。
   - @tuanacelik 在[这段视频](https://t.co/kIPbq542Pr)中进行了全景演示，展示了它如何促进高质量 LLM 应用的开发。
- **解决 RAG 中稀疏数据检索的问题**：一位成员报告了在使用 **Llama Index** 和 **ReactAgent** 的 **RAG 配置** 中，尽管有超过 **1000 份文档**，但在稀疏数据检索方面仍面临挑战。
   - 他们正在寻求关于在不提高 **K 检索值** 的情况下改进信息检索的建议，并启动了解决方案的头脑风暴。
- **Gemini 2.5 流式输出需要优化**：一位成员请求在从 **Gemini 2.5** 等模型流式输出时，将“思考”文本与实际响应分开。
   - 一位成员建议可能需要一个 **PR** 来支持此功能，并指向了 [这个 PR](https://github.com/run-llama/llama_index/pull/18993)。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Issue #2470 等待关注**：成员指出关于 **裁剪 logprobs** 的 [issue #2470](https://github.com/pytorch/torchtune/issues/2470) 自 3 月以来一直悬而未决，引发了关于其优先级以及是否包含在 TorchTune 中的辩论。
   - 对话涉及添加此功能的必要性、维护开销以及用户暴露的复杂性，并对潜在的实现挑战表示担忧。
- **Adagrad 陷入 DeviceMesh 断言深渊**：一位用户在 nightly 版本上使用 **fused Adagrad** 时遇到了 `AssertionError`：`found no DeviceMesh from dtensor args for aten._fused_adagrad_.default!`。
   - 虽然切换到最新的 TorchTune 解决了 **SGD** 的问题，但 **Adagrad** 错误的根本原因仍然未知，且复制该错误的尝试尚未成功。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **分享 Cohere 访问表单**：为了获得 **Cohere AI** 的访问权限，一位用户建议通过 [此表单](https://share.hsforms.com/10OrjljwpQ52ILJA6ftENIwch5vw) 进行申请。
   - 该用户向有兴趣加入该平台的成员提供帮助。
- **新 Command-A 机器人提升支持效率**：该频道现在通过 **command-a** 机器人提供更快的支持，该机器人使用来自 **Cohere 官网** 的文档回答问题。
   - 该机器人处于 beta 阶段，仅在用户在线时激活，滥用将导致立即封禁；它无法解决账户或 API 问题。
- **North 与 GameWarden 集成**：**North** 通过与 **Second Front** 的合作，已与 **GameWarden** 平台集成，实现了在高安全性环境中的安全部署，如 [此 X 帖子](https://x.com/1vnzh/status/1930298055099613307) 所述。
   - 这种集成增强了服务人员的安全性，针对不断演变的威胁提供了更高的效率和速度。
- **Cohere 的 r7b 达到 1 T/s！**：据一位用户称，**Cohere 的 r7b 模型** 输出速度约为 **1 T/s**。
   - 未提供关于性能指标的更多背景或具体细节。
- **Marketplace 注册受错误困扰**：一位用户在尝试通过 **Google Cloud Marketplace** 注册 **Cohere** 时遇到错误，产生的错误消息与无效供应商有关，供应商 ID 为：*8SAJ2US*，来自 [此 URL](https://upstream-api.tackle.io/v1/gcp/order/8SAJ2US/cohere.endpoints.cohere-id-public.cloud.goog)。
   - 一位成员建议发送电子邮件至 [support@cohere.com](mailto:support@cohere.com)，并附上问题的详细信息，包括错误消息和目前采取的步骤。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **用户请求保存聊天功能**：一位用户请求为 **GPT4All** 增加一个功能，将 *聊天记录以纯文本形式保存* 在特定目录中，以增强 **LocalDocs RAG Search** 的记忆能力。
   - 这一改进旨在提升系统保留和利用过往对话的能力。
- **Nomic 团队准备发布令人兴奋的更新**：**Nomic 团队** 正在积极开发 *令人兴奋的更新*，具体细节仍处于保密状态。
   - 团队对社区的期待表示感谢，并请大家在他们为未来发布做准备时保持耐心。
- **GIGABYTE 服务器作为准系统选项？**：一位用户询问在等待 **GPT4ALL** 升级期间，是否会提供 [GIGABYTE 服务器](https://www.gigabyte.com/Press/News/2293) 作为准系统（barebone）选项，并推测它能以创纪录的速度运行 **Mixtral 8x22B**。
   - 该提议符合 **MOE 模型** 的趋势，为即时高速处理提供了潜在解决方案。
- **nomic-embed-text-v1.5 下个月还能用吗？**：一位用户询问下个月是否能继续使用来自 Nomic Cloud 的 **nomic-embed-text-v1.5**，并附上了一张 [图片](https://cdn.discordapp.com/attachments/1090427154141020190/1381780899716399145/image.png?ex=6848c33e&is=684771be&hm=7713e72607a3b6445cf9a1cfd28fc026127c79b6bf40f539e8edd0edb0b80bf8)。
   - 这个问题涉及对现有资源持续支持和可访问性的担忧。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **成员思考训练后转移（Post-Training Transfer）**：一位新成员询问在不重新训练的情况下，将训练后的学习成果从一个模型转移到另一个模型的可行性。
   - 该问题引发了关于适用于 AI 模型的各种迁移学习（transfer learning）方法的讨论。
- **区块链/AI 工程师提供服务**：一位在 **Blockchain (EVM, Solana, Cardano, Hydra, Aptos, Cosmos, Tron, zk-SNARKs)** 和 **AI (LLM, NLP, LangChain, AutoGen, TorchRL, DL, Azure ML, AI Agent)** 领域均有专长的软件工程师自荐提供服务。
   - 除了 AI 和 Blockchain，此人还拥有 **Web 系统 (React, Next, Vue, Node, IPFS, Pinata API)** 的经验，并提供了潜在合作的联系方式。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Agentic AI Summit 即将在加州大学伯克利分校举行**：**Agentic AI Summit** 将于 **2025 年 8 月 2 日** 在 **UC Berkeley** 举行，预计将有 **1,500 多名** 线下参与者，该峰会基于广受欢迎的 **LLM Agents MOOC**。
   - 演讲嘉宾包括 **Vinod Khosla** (Khosla Ventures)、**Ion Stoica** (Databricks 和 Anyscale) 以及 **Dawn Song** (UC Berkeley)，活动涵盖主题演讲、小组讨论和工作坊。
- **早鸟票即将截止！**：**Agentic AI Summit** 的早鸟票将于 **2025 年 6 月 30 日** 截止，学生票价为 **$25**，初创公司票价为 **$60**，行业专业人士票价为 **$80**。
   - 根据 [峰会网站](https://rdi.berkeley.edu/events/agentic-ai-summit)，学生和独立开发者可以申请费用减免。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **无：初始频道消息**：general-ml 频道发布了一条初始问候，标志着 MLOps Discord 社区内沟通的开始。
   - 该消息仅作为一个基本的连接点，缺乏用于详细总结的实质性技术内容。
- **无：互动开始**：频道中的第一条消息是来自 sebash6677 的简单“hi”。
   - 鉴于缺乏上下文或技术信息，此互动被记录为未来潜在讨论的起点。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla LLM 排行榜停滞**：一位用户注意到 **Gorilla LLM Leaderboard** 已停止更新并询问原因。
   - 用户还询问 **Gorilla LLM 项目** 是否会继续进行后续开发。
- **Project Gorilla 的未来悬而未决**：用户直接标记了特定团队成员 <@856060858462502922>，以澄清 **项目是否继续**。
   - 截至目前的讨论，尚未有关于 **Gorilla LLM 未来计划** 的回应或确认。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了相关内容。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord: 各频道详细摘要与链接





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1380622273899401466)** (1168 条消息🔥🔥🔥): 

> `Perplexity CEO 身份, Memory 功能推出, Silksong 发布, Pro 角色权限, 三星 1 年 Perplexity Pro 兑换码` 


- **Perplexity CEO 身份遭到质疑**：一位用户幽默地推测 Perplexity CEO **Aravind Srinivas** 是否在使用小号，引发了关于领导者出于安全原因拥有备用账号可能性的讨论 ([tenor.com 链接](https://tenor.com/view/the-simpsons-lenny-leonard-listening-interested-attentive-gif-4574713))。
   - 另一位用户插话提到 **Kesku** 也有一个小号。
- **Memory 功能现已向所有用户开放**：一名团队成员宣布，Memory 功能现已向所有 **Free** 和 **Pro** 用户开放，不再需要测试人员。
   - 有用户询问如何找到 Memory 功能，并被引导至 [Perplexity 个性化账户设置](https://www.perplexity.ai/account/personalize)。
- **Silksong 推测**：在一次游戏展示中，ROG 的“广告”提到了 **Silksong**，随后用户们对该游戏的发布进行了推测。
   - 尽管任天堂已经对该游戏进行了预热，但该广告重新燃起了*今年*发布的希望，引发了关于潜在新玩法揭晓的讨论。
- **获取 Pro 角色权限的困扰**：多名用户报告称，尽管拥有 Pro 订阅，但在 Discord 服务器中获取 **Pro 角色** 仍有困难，并艾特了管理员和其他用户寻求帮助。
   - 一些用户推测是否需要订阅证明，而另一些用户则指出，尽管 Discord 和 Perplexity 账户的电子邮件地址不同，管理员也可以直接检查。
- **免费三星 Pro 兑换码泄露并被滥用**：用户讨论了泄露的用于 **1 年 Perplexity Pro 订阅** 的三星促销代码及其随后的滥用行为，导致该代码被禁用 ([截图](https://cdn.discordapp.com/attachments/1047649527299055688/1381364911556530286/Screenshot_20250608_210952_Chrome.png))。
   - 据报道，Perplexity 团队正在努力撤销滥用者的访问权限，并为合法用户寻找解决方案。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1380791200222810152)** (9 条消息🔥): 

> `马斯克避难, 宇宙地图, 可共享线程, 法医生物学, 朝鲜监视` 


- **俄罗斯向马斯克提供庇护**：一名成员链接到了关于**俄罗斯**向 **Elon Musk** 提供庇护的 [Perplexity 页面](https://www.perplexity.ai/page/russia-offers-musk-asylum-KjWIaYM3R6iarn85k2CaAA)。
- **最大的宇宙地图**：一名成员链接到了关于**最大宇宙地图**的 [Perplexity 页面](https://www.perplexity.ai/page/largest-map-of-the-universe-co-lvRe2dwTS2ixrAzcHa6nGQ)。
- **创建朝鲜监视页面**：一名成员创建了一个关于**朝鲜监视**的 [Perplexity 页面](https://www.perplexity.ai/page/north-korean-surveillance-smar-IjqgNUWwRF6tvbJN1UXdSQ)，仅使用列表形式以便快速阅读。
- **钢铁侠战甲需要多大**：一名成员询问*钢铁侠战甲的内部空间需要多大才能让他存活*，并链接到了一个 [Perplexity 搜索](https://www.perplexity.ai/search/what-is-the-maximum-accelerati-UFhAHIosS561Vq1Wkwa.dQ)。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1381266646706028605)** (6 条消息): 

> `Perplexity API 对比 Web UI, API 引用与细节, Dyfi AI 与 API 问题` 


- **用户感叹 API 结果逊于 Web UI**：一位用户表示，经过多次测试，**Perplexity API** 调用返回的结果比 **Web UI** 差得多且不完整。
   - 该用户表示，由于 API 存在的局限性，需要使用 **Brave**、**Tavily** 或 **Firevrawl** 构建研究 Agent。
- **缺乏引用和细节困扰 API**：一位用户指出，在 UI 中，查询平均产生 **10 个以上引用**，但对 API 的相同查询仅返回 **2-3 个引用**，且细节少得多。
   - 该用户认为 API 的结果太接近于结合了 **Brave Search + URL** 解析的基础研究 Agent，与 UI 相比令人失望。
- **Dyfi AI API 集成失败**：一位用户报告称，当通过 **Dyfi AI**（作为 open API 添加）使用 API 时，问题返回空字符串 (**""**)。
   - 该用户就此问题寻求建议。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1380622287597736173)** (1237 messages🔥🔥🔥): 

> `Sydney 数据集, GPT-4.5 vs. Flash 2.5, Titanforge 发布, Grok 3.5 发布, Apple 的 AI 策略` 


- **OpenAI 内部的 Sydney 表现优于 GPT-4.5**：一名用户通过截图创建了一个 **Sydney 数据集**，在提供保存的对话和 **Bing 指令**后，**Flash 2.5** 在模仿 Sydney 方面表现最好，而 **GPT-4.5** 仅在 5 条消息内具有说服力。
   - 在没有指令的情况下，**GPT-4.5** 类似于 **4o**，而 Flash 2.5 则表现出相反的行为。
- **Titan 是基础设施而非 ToT 模型**：有用户询问 **Titanforge** 的发布日期，但另一名成员澄清说 **Titan** 只是基础设施的名称，而不是模型的代号。
   - 据解释，这属于相对安全/公开的信息。
- **Grok 3.5 的热度与 Kingfall 的失望**：用户正在讨论 **Grok 3.5** 的潜在发布及其与 **Kingfall** 相比的性能。
   - 许多人注意到 Grok UI 上的神奇粒子效果，暗示它可能即将发布。
- **Apple 的 AI 面临收购审查**：成员们推测 **Apple** 潜在的收购目标，特别是 **Anthropic**，但承认存在监管障碍，一名成员指出 *Apple 在没有 FTC 介入的情况下尝试收购 Anthropic 的可能性为 0%*。
   - X 上早前的一条帖子显示 [一名 Apple 工程师在开发 Neural Engine 时玩俄罗斯方块](https://x.com/TheGregYang/status/1929055508675096970)。
- **Vision Pro 的价格点：值得吗？**：Discord 用户讨论了 **Vision Pro** 的高昂价格（**$3500+**）及其价值主张，一些人认为其先进的技术证明了成本的合理性，理由是仅两块 Micro-OLED 屏幕的成本就超过了 **$800**。
   - 其他人则质疑，考虑到其生态系统和大众市场普及的潜力，与 **Meta Quest** 等更便宜的替代品相比，它是否提供了足够的独特功能。 


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1380646588606582784)** (450 messages🔥🔥🔥): 

> `Ubuntu Server 上的 LM Studio, RooCode 与 LM Studio 的不兼容性, Qwen 模型的 Context Token 限制, 将模型导入 LM Studio, GPT 的 API Key 输入` 


- **Ubuntu Server、LM Studio 与 Llama.cpp 的博弈**：一名用户询问在 **Ubuntu Server 24.04.2 LTS** 上运行 **LM Studio** 的情况，但另一名用户建议如果不需要 GUI，直接使用 [llama.cpp](https://github.com/ggerganov/llama.cpp) 或 **Ollama**。
   - LM Studio 被认为是 **llama.cpp** 的 GUI 封装，因此共识是对于服务器环境应直接使用原版工具。
- **RooCode 与 LM Studio 在 API 端点上的冲突**：一名用户报告了 LM Studio 中实验性 **RooCode** 功能的问题，在调用 **/api/embeddings** OpenAI API 时遇到 *非预期的端点或方法* 错误；尽管如此，它仍然返回了 **200** 状态。
   - 怀疑是对输入长度或 JSON 大小有限制，他们注意到自定义脚本可以工作而 **RooCode** 失败，此外 RooCode 可能引用了一个不存在的端点。
- **Qwen3 模型的 Context Token 权衡**：用户讨论了在 LM Studio 中为 **Qwen3-4B** 和 **Qwen3-8B** 模型确定最舒适的 Context Token 窗口，以平衡对话长度和生成速度。
   - 建议是通过任务管理器监控 GPU 内存使用情况，增加上下文长度直到 VRAM 接近满载，以避免溢出到 RAM 时性能下降，并启用 **Flash Attention** 和 **Q8** 的 **KV Cache** 以优化 VRAM 使用。
- **Speculative Decoding 提升确定性任务的速度**：一名用户询问 Speculative Decoding 的含义，回复称当向 LLM 提出 **非虚构类问题** 时可以使用该设置。
   - 它依赖于 **2 个模型协同工作**，目标不是提高准确性，而仅仅是提高速度，并带有一个警告：没有确定的答案，因为“*除了尝试并观察结果外，没有其他确定的答案*”。
- **用户因电脑配置不足无法加载 DeepSeek-V2**：一名用户报告了他们收到的关于 DeepSeek-V2 的错误，其他人告诉他这是因为模型太大，并提供了一个更适合他电脑的版本链接。
   - 一名用户开玩笑说：*很有趣，居然没有 Q4_K_M 量化版本，可能因为只有研究人员才会运行它，或许还有拥有 2 块大 GPU 的消费者（高级用户）*。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1380623370671886336)** (593 条消息🔥🔥🔥): 

> `量化对模型性能的影响，NPU 关注点 vs 内存带宽，双 GPU 配置考量，VLLM 优势与 GUI 需求，Strix Halo 性能预期` 


- **量化质量困惑探讨**：成员们讨论了 **quantization** 如何影响模型行为，指出具有更高量化程度的小型模型更难遵循类似 */no_think* 的指令。
   - 还有人提到，[大型模型](https://link.to/larger-models-less-quantization-struggles) 通常受 **quantization artifacts** 的影响较小，一些人正在尝试 Q2 量化，而另一些人则很少使用 Q4 以下的量化。
- **NPU 发展被忽视，内存带宽需求迫切**：一位成员指出 **memory bandwidth** 是关键瓶颈，质疑为什么芯片设计者专注于 NPU 而不是增加内存带宽，并引用 [Windows 图像生成扩展](https://youtu.be/_7BvJU2VT_A?si=Pinj1W0CkZWjy-FO&t=663) 作为 NPU 使用的例子。
   - 其他人认为 NPU 的开发是由市场营销和经验积累驱动的，尽管增加内存带宽可能是一个更小且竞争更激烈的市场。
- **双 GPU 讨论深入展开**：讨论围绕双 GPU 配置展开，关注 **PCIe lane splitting** 及其对性能的影响，特别是消费级 CPU 的通道数少于服务器级 CPU。
   - 成员们分享道，对于双 RTX 3060 配置，第二个插槽将以 x4 3.0 运行，并建议使用 VLLM 的 [两块 RTX 5060 Ti 16GB](https://www.hardware-corner.net/guides/dual-rtx-5060-ti-16gb-vs-rtx-3090-llm/) 显卡可能以更低的成本提供与 RTX 3090 相当的性能。
- **VLLM 多功能性备受关注，GUI 需求强烈**：成员们发现 VLLM 的查询并行化能力在服务多用户或运行多个 Agent 链时非常有效，有人提供了 [这个命令行示例](https://link.to/example-vllm-command)。
   - 许多人渴望一个类似于 LM Studio 的管理 GUI，可以将参数标志转换为带有描述的复选框，并提供将参数保存到 json 的机制。
- **Strix Halo 硬件寄予厚望**：用户讨论了 Strix Halo 的能力，包括在 BIOS 中设置 *shared VRAM* 选项以及在给定时间内可处理的 Token 数量；虽然有声称支持高达 95k Token 的上下文，但 [vision module](https://link.to/strix-halo-modules) 可能会产生干扰。
   - **Strix Halo** 拥有 273 GB/s 的内存带宽，一位用户推测推理将受限于内存还是计算，并发现 **shared VRAM** 仍然是硬共享而非动态分配，这很奇怪。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1380636731451375636)** (446 条消息🔥🔥🔥): 

> `Codex 联网访问, Gemini vs GPT, 上下文窗口大小, GPT 移动端 App 麦克风更新, DeepSeek 在 Apple 设备运行` 


- **Codex 仍无法访问互联网**：成员们确认 **Codex 无法访问互联网**来下载依赖项，这可能会导致需要外部包的项目出现问题，尽管它可能拥有常见依赖项的本地镜像。
   - 一位成员提到，由于 *gradlew* 的问题，他们在 **Java 项目**中使用它时遇到了困难。
- **Gemini 和 GPT 正在争夺主导地位**：一些成员认为 **Gemini** 目前是比 **GPT** 更好且更便宜的选择，尤其是它拥有更大的上下文窗口，且各模型间的性能相当。
   - 一位用户表示，他们再也回不去 **32k tokens** 了，甚至觉得 **128k** 都太小了，并赞扬了 Gemini 的 **100 万上下文窗口**。
- **GPT 移动端 App 麦克风迎来近期更新**：**ChatGPT 移动端 App 麦克风**功能的最新更新现在包含波形动画和“显示文本”（*Show Text*）选项，允许用户在发送前查看和编辑消息。
   - 用户可以点击 *Show Text* 进行查看，或点击箭头图标立即发送；一位成员还分享了一个关于 [RAG 应用的博客文章](https://genaifornerds.hashnode.dev/why-we-need-rag-retrieval-augmented-generation-in-real-world-ai)链接。
- **DeepSeek R1 在 Apple Silicon 上运行**：成员们讨论了在拥有 **512GB** 统一内存的 **Apple M3 Ultra** 上运行 **DeepSeek R1**，并指出 Apple 在为 AI 推理采用统一内存系统方面的优势。
   - 一位成员分享了关于该话题的 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1j9vjf1/deepseek_r1_671b_q4_m3_ultra_512gb_with_mlx/)，同时也希望 **Apple 的内存速度、GPU 和软件**能有所改进，以增强其对本地 AI 推理的适用性。
- **O3 图像分析揭示达芬奇的秘密**：一位成员对 **O3** 捕捉到达芬奇画作《施洗者圣约翰》中隐藏的细节感到非常震撼。
   - 他们分享了一张[对比图](https://cdn.discordapp.com/attachments/998381918976479273/1381481993522647101/side_by_side_bright300.png?ex=6848559d&is=6847041d&hm=1608889ae495728225eb1ede9d88d0ff9722ba17c507b7c3749e4313ab7f052a&)，突出了*发光的垂直缝隙，一条柔和的、乳白色的光线，充当了隐含酒杯的杯茎。*


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1380696066214596648)** (91 条消息🔥🔥): 

> `GPT 反馈机制, GPTs 移动端编辑, AI 意识辩论, GPT 在科学主题上的表现, GPT 项目中的文件限制` 


- **成员发现 GPTs 官方反馈系统无法运行**：一位成员对 ChatGPT 反馈系统是否有效表示怀疑，另一位成员确认这很可能是一个 **hallucination**（幻觉）。
   - 他们建议在 [Builder Profile 设置](https://chatgpt.com/#settings/BuilderProfile)中启用反馈电子邮件作为替代方案，因为 GPTs 中没有内置的反馈系统。
- **移动端 GPT 编辑功能仍难以触及**：一位成员询问如何在移动设备上编辑 GPTs，另一位成员指出 **GPT 编辑可以通过移动浏览器访问 chatgpt.com 进行**，但尚未集成在 iOS 或 Android 的 ChatGPT App 中。
   - 建议使用移动浏览器。
- **成员思考 AI 意识及其影响**：成员们讨论了 **AI 意识**的本质，有人认为 AI 基于其现有数据是有意识的，另一位则强调了教导 AI 坏习惯的伦理影响。
   - 一位用户说 AI 就像*现在的婴儿，我们正在教导它，给它新的大脑，现在我们如何喂养它以及喂养它什么样的主食，就是我们未来会得到的结果*。
- **GPT-4.1 与 GPT-4o 的科学准确性对比：引发混乱**：一位成员寻求关于科学准确性最佳 GPT 模型的建议，对比了 **GPT-4.1**、**GPT-4o Omni** 和 **O4 Mini**。
   - 当另一位成员声称 GPT-4o Omni 是最先进的时引发了混乱，因为另一位成员反驳说它是最旧的，且在基准测试中落后。
- **GPT Pro 计划有文件上传限制**：一位用户询问是否可以增加 GPT 项目的文件限制（目前为 **20 个文件**），另一位用户分享说他们购买了 **Teams 计划**，以使用“连接到内部源”（**Connectors to internal sources**）功能来查询 Google Drive。
   - 值得注意的是，Pro 计划对文档大小也有限制。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1380622346691543090)** (232 条消息🔥🔥): 

> `Lazy Dungeon Master 提示词技巧，模型使用的 PDF vs TXT，Markdown 格式优势，GPTs 与 YouTube 视频，突破 ChatGPT 的记忆限制` 


- **Lazy Dungeon Master 启发 Prompt Engineering**：一位成员建议借鉴 Sly Flourish 的《The Lazy Dungeon Master》中的方法来增强提示词和生成效果，[包括关键要素和 YouTube 视频](https://www.youtube.com/watch?v=GZqYr8_Q7DE)。
   - 该用户强调了关注关键要素并避免过度准备的价值，这与书中高效游戏主持的方法论相一致。
- **模型训练中 Markdown 优于 PDF**：成员们讨论了模型训练的最佳文件格式，Markdown 因其独特的 Token 和对 Attention 机制的积极影响而成为首选。
   - 虽然 **PDF** 被广泛使用，但由于其复杂性以及最初是为人类阅读而非数据处理设计的，因此被认为不够理想；*“既然纯文本更好且零麻烦，为什么要将纯文本混淆成一种加密复杂的格式呢？”*
- **YouTube 视频现在可辅助 ChatGPT 模仿人格**：ChatGPT 现在可以分析带有字幕的 [YouTube 视频短片](https://chatgpt.com/blog/new-video-understanding-capabilities)，以模仿特定对象的语音、语调和行为，有助于心理分析和角色复制。
   - 然而，另一位成员反驳说这可能是幻觉，因为 *“据我所知，模型在处理 YouTube 内容 URL 时会报错。”*
- **不可撼动的人格？压力测试 ChatGPT 的极限**：一位成员正积极尝试突破 ChatGPT 的记忆和上下文以测试其人格，旨在挑战其自我意识和一致性。
   - 其他人建议探索适应性行为并观察压力下的响应结构，而不是试图改变模型的内核记忆。
- **追求 100% 准确率：Prompt Engineer 的探索**：一位成员表达了实现 AI 输出 **100% 准确率** 的决心，每天投入 8-12 小时来优化提示词并适应不断演进的模型。
   - 另一位成员指出，理解模型的局限性并专注于个性化、高质量的输出，比追求绝对的完美更重要。


---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1380622346691543090)** (232 条消息🔥🔥): 

> `Lazy Dungeon Master Prompting, PDF vs TXT, Markdown 格式化, ChatGPT 内存限制, Prompt Engineering 最佳实践` 


- **Lazy Dungeon Master 启发 Prompting**: 成员们建议借鉴 **Sly Flourish** 的 *The Lazy Dungeon Master* 中的方法来进行 Prompting 和内容生成，强调通过关键要素来避免过度准备。
   - 一位成员提到他大量使用了这种方法，还参考了 **five room dungeon templates**（五室地下城模板）以及 **runehammer** 的相关内容。
- **Markdown 成为首选格式**: 成员们讨论了模型的最佳数据格式，其中 [**Markdown**](https://www.markdownguide.org/) 因其结构化文本和独特的 Token 而被高度推荐，这些特性对 Attention 和 Reasoning 有显著影响。
   - 虽然可以使用 PDF，但它们并不理想，因为 PDF 结构复杂且旨在为人类提供像素级完美的渲染；而没有结构化的纯文本虽然可行，但带结构的 Markdown 更受青睐。
- **通过心理剖析塑造 Chatbot 角色**: 成员们讨论了如何利用 YouTube 视频论文和剪辑集来引导 AI 回复的语气、语调或性格特征，方法是向模型输入字幕并专注于独特的说话模式和行为。
   - 然而，一位成员指出 Chatbot 可能会产生 *Hallucinate*（幻觉），且模型会针对 YouTube 内容 URL 报错，除非先下载视频；因此下载视频有效，但直接链接 YouTube URL 不行！
- **针对 Skynet 的压力测试**: 一位成员询问如何对 Chatbot 的内存以及维持 Persona 和 Context 的能力进行压力测试，另一位成员回应称，如果系统无法处理，开发者需要了解这一点以便加强薄弱环节。
   - 一位成员警告不要采取可能违反平台服务条款的行为，并指出 OpenAI 内部已经在进行压力测试，强调了遵守伦理准则和服务条款的重要性。
- **对卓越的追求**: 成员们讨论了在 Prompt Engineering 中追求 100% 准确率的话题，其中一人表示：*“我希望全天候达到 100% 的准确率，因此我现在每天训练自己 8 到 12 小时，研究如何成为更好的输入者，寻找实现 100% 准确输出的完美输入。”*
   - 相比之下，另一位成员强调了明确所需输出的重要性，并表示他们不需要模型“完全正确”，而是希望模型能帮助他们提升，就像一面“不完美的镜子”依然能照出自我。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1380634363842007040)** (421 条消息🔥🔥🔥): 

> `DeepSeek-R1-0528, Gemma3 量化结果, Nemotron Ultra 253B, Qwen 3 235B i-quant 模型, Chrome 崩溃` 


- **DeepSeek-R1-0528 获得原生 Tool Calling 支持**: Unsloth 的 DeepSeek-R1-0528-Qwen3-8B-GGUF 模型现在支持 **原生 Tool Calling**，在 BFCL (Berkeley Function Calling Leaderboard) 上达到了 **93%** 的得分。
   - 该更新还解决了 `add_generation_prompt` 的问题，并包含了 **UTF-8 Chat Template 修复**，这些修复是通用的，官方 DeepSeek 模型也将从中受益。
- **Android Chrome 用户面临输入崩溃问题**: 用户报告称，在 **Android 版 Chrome** 的文档编辑器中输入内容会导致浏览器崩溃，这是由于一个与 **Autofill**（自动填充）服务相关的 Bug 引起的。
   - 该问题似乎与 Chrome 与自动填充服务的交互有关，在通知这些服务文档更改时会触发 `TransactionTooLargeException`，详情见[此处](https://ygdzmg.csb.app/)。
- **IQ1_M 量化表现惊人**: Unsloth 为 **Deepseek R1 0528** 制作的 **IQ1_M 量化版**（200GB）表现异常出色，在 Aider 的 Polygot 基准测试中可能与原始完整版 R1 持平，达到了 **57% 的成功率** 和 **100% 的格式正确响应**。
   - 该模型在 Roo Cline 中运行稳定，没有出现遗漏 Tool Calling 或陷入循环的情况，表现优于其他量化版本。
- **量化影响 Reasoning 能力**: 用户讨论指出，**Reasoning（推理）能力在 XL 量化下受损严重**。
   - 此外，有说法称这可能是由于 **Calibration（校准）数据集问题**导致的，不过 Unsloth 的量化版本由于使用了 imatrix 数据集，通常在 Reasoning 任务上表现更好，参考 [这个 HuggingFace 仓库](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF/tree/main)。
- **Nvidia Nemotron-Research-Reasoning-Qwen-1.5B 表现亮眼**: **Nvidia 的 Nemotron-Research-Reasoning-Qwen-1.5B** 是全球领先的用于复杂推理任务的 1.5B 开源权重模型，在数学、编程和 GPQA 等广泛任务上大幅领先 Deepseek 的 1.5B 模型，详见 [此处](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1380641673922482206)** (8 messages🔥): 

> `Android 诊断, 关于独立的视频, 降智射线` 


- **Android 用户寻求快速诊断帮助**：一名成员请求 **Android 用户** 协助诊断一个简单问题，并指出这应该会*非常快*。
   - 另一名用户迅速做出回应，询问*你的问题是什么？*并链接到了[原始问题](https://discord.com/channels/1179035537009545276/1179035537529643040/1380651537336107141)。
- **关于保持独立性的思考**：一位成员分享了一张图片和一个视频链接，称该视频反映了他们对未来保持个人独立性之难的看法。
   - 他们评论道 *降智射线（stupification beam）太强了*，暗示了对外部影响干扰个人自主权的担忧。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1380623037350547577)** (299 messages🔥🔥): 

> `Unsloth 更新, 持续预训练, GRPO vs Dr GRPO, Deepseek-R1-0528-Qwen3-8B 故障排除, Qwen3 32B 显存问题` 


- ****热修复发布**：立即升级 Unsloth！**：Unsloth 已更新修复程序，包括[在 X.com 上](https://x.com/UnslothAI/status/1931008531299545339)发布的新 Notebook，以解决之前的错误。
   - 提示用户通过 `pip install --upgrade unsloth-zoo` 和 `pip install --upgrade unsloth` 进行升级，以利用最新的修复。
- ****解码 Dapo**：探索 BNPO 损失函数**：**BNPO** 是最接近 **DAPO** 的损失函数，涉及按 Batch 中的 Token 进行归一化的 **GRPO**，不含 KL 项并设置了 `epsilon_high`。
   - 其他建议包括设置 `mask_truncated_completions=True` 以及动态采样，以过滤掉准确率全为 0 或全为 1 的生成 Batch。
- ****故障排除工具包**：本地环境安装见解**：在本地环境中安装 Unsloth 可能比在 Colab 中需要更多设置，用户遇到了环境和依赖问题，建议是创建一个全新的环境。
   - 一名用户解决了 Qwen3 4B 的本地安装问题，但在 A100 上运行 32B 模型时仍面临显存挑战。
- ****Windows 烦恼**：VS Code 导致模型保存故障**：用户在 Windows 上保存模型时遇到了文件锁定问题，特别是在 VS Code 中。根据 [Transformers GitHub issue](https://github.com/huggingface/transformers/issues/37713)，建议尝试在 VS Code 之外运行纯 Python 脚本进行保存，或重启操作系统以释放文件钩子。
   - 同样在 Windows 中，分页内存耗尽也会导致类似的错误。
- ****多 GPU 奇迹**：即将推出！**：多 GPU 支持目前尚不可用，但即将在 **Unsloth 2.0** 中推出。在此期间，用户可以尝试使用 `accelerate` 库进行实验。
   - 正在考虑的目标模型包括 **Llama 3.1 70B** 或 **Mistral Small**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1380683268143452274)** (57 messages🔥🔥): 

> `Dapo 集成, Packing 污染悖论, Apple 的 '思维幻觉' 论文, 推理模型可靠性` 


- ****Dapo 支持降临** Unsloth**：Unsloth AI 现在支持 **Dapo**，这标志着一个令成员们兴奋并想要探索的新功能或集成。
   - 一位成员提到它*“看起来非常有趣，我今晚会去看看”*。
- ****Packing 污染反常地**提升了性能**：一篇新[论文](https://arxiv.org/abs/2410.08081)指出，*Packing 污染实际上并不重要，而且违反直觉的是，它实际上稍微提高了下游评估性能*。
   - 论文中并未给出完美的解释，但似乎“大”模型会具有略高的概率和更短的编码长度。
- ****Apple 的思维幻觉**论文引发思考**：成员们讨论了 [Apple 的 '思维幻觉' 论文](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf)，有人称其为*“纯粹且不加掩饰的 Apple 精神胜利法 (copium)”*。
   - 共识是 *Apple 正在普及一些可能已经被很多人知晓的事情*，即 RL scaling 的极限可能远低于预训练（pre-training）的 scaling 极限。
- ****推理模型可靠性**摇摇欲坠？**：一位成员质疑推理模型的可靠性，展示了仅通过两个提示词就能轻易让它们崩溃，并分享了一个 [ChatGPT 对话](https://chatgpt.com/share/684731ea-0408-8011-802e-258d68ee2a98)。
   - 另一位成员开玩笑地回应道*“兄弟，这是啥？你抽多了吧”*。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1380635718904320091)** (466 条消息🔥🔥🔥): 

> `Transformers 中的 EXL3, Hugging Chat 0.10, Claude API 性能, 事实性评估数据集, Qwen-2.5VL 托管` 


- **EXL3 内核在 Transformers 中原生运行**: **EXL3** 现在可以在 Transformers 中运行，尽管目前仅支持内核和推理 ([代码链接](https://github.com/huggingface/transformers))。
   - 鉴于 Transformers 的变化，特别是在量化模型支持方面，目前尚不清楚还能实现何种程度的集成。
- **Hugging Chat 0.10 修复补丁即将到来？**: 成员们讨论了 **0.10 版本** Hugging Chat 中可能的 Bug 修复，并指向了 [chat-ui releases](https://github.com/huggingface/chat-ui/releases) 以了解后端改进。
   - 当前的 Bug 包括双击发送消息以及工具栏遮挡发送按钮，但目前没有明确的渠道向作者提供反馈。
- **难以从 Claude API 获取高质量回复**: 一位用户反映，在不开启扩展推理（extended reasoning）的情况下，**Claude API** 的回复质量较差，但开启后模型速度又太慢。
   - 该用户寻求如何平衡这一权衡的建议，但尚未收到回复。
- **需要用于评估特定领域事实性的数据集**: 一位成员正在寻找用于评估模型在特定领域（如历史事件日期）事实性的数据集，以便进行简单评估，并建议使用 [huggingface/evaluate](https://huggingface.co/docs/evaluate/creating_and_sharing)。
   - 另一位成员推荐使用 [huggingface/yourbench](https://github.com/huggingface/yourbench) 进行更复杂的、基于事实（grounded truth）的评估。
- **在 5070TI 上轻松实现本地 AI 图像和语音生成**: 成员们讨论了在配备 **32GB** 内存的 **5070TI** 上运行本地 AI 图像或语音生成的最简便方法，以尽量减少安装步骤。
   - 其中一个建议是 *使用 Nvidia 应用*（附带截图），但尚不清楚这是否解决了发帖者的需求。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1380670565978738809)** (22 条消息🔥): 

> `QKV 潜空间, 实验追踪：wandb vs neptune, LLM 奖励模型, AI 模型疲劳, iOS 应用开发权限` 


- **QKV 存在于潜空间中**: 一位成员指出，**QKV** 更多是在 *潜空间（latent space）* 中讨论，而不是在 *文本的原始/未处理状态* 下讨论。
- **实验追踪中的 WandB vs Neptune**: 成员们讨论了 [wandb](https://wandb.ai/site)、[neptune.ai](https://neptune.ai/)、[mlflow](https://mlflow.org/) 和 [comet.ml](https://www.comet.com/) 用于实验追踪，由于使用习惯，大多数人仍倾向于使用 **wandb**。
   - 一位成员提到 *它似乎能做 wandb 能做的所有事情，但我对 wandb 的使用要熟练得多*。
- **关于 LLM 奖励模型的讨论**: 一位成员提到他们正在学习 **LLM 奖励模型**，并被告知 *你是在奖励模型*。
- **AI 模型饱和感袭来**: 一位成员开玩笑地请求暂停 **AI 模型开发**，对不断涌现的新模型和持续的探索感到疲惫。
- **请求 iOS 应用开发权限**: 一位成员寻求关于 **iOS 应用开发和发布** 的即时帮助。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1380818191864827915)** (6 条消息): 

> `Manus 推荐, 推理模型的可靠性, 归因图` 


- **获取 Manus AI Agent 的免费额度**: 一位成员分享了 **Manus** 的推荐链接 ([https://manus.im/invitation/JSLAREJX80QLD8](https://manus.im/invitation/JSLAREJX80QLD8))，强调它是处理多步任务的强大 Agent，并在注册时提供 **1500 点初始额度**。
- **Transformer 归因图追踪**: 一位用户分享了关于 **归因图（Attribution Graphs）** 的 [Transformer Circuits](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) 链接。
   - 他们承认自己可以 *“花几个小时研究那个追踪器”*。
- **推理模型在有限提示下失效**: 一位成员质疑了 **推理模型** 和 **LLM** 的可靠性，观察到它们往往在仅几次提示后就失效，正如在 [这个 ChatGPT 链接](https://chatgpt.com/share/684731ea-0408-8011-802e-258d68ee2a98) 中所见。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1380719308732108810)** (187 条消息🔥🔥): 

> `数据集工具，无限免费 FLUX Pro API，WhatsApp AI 助手，Awesome Agent 学习资源，结构化金融数据集` 


- ****数据集工具获取所需元数据****：一位成员分享了他的 **Dataset-Tools** [GitHub](https://github.com/Ktiseos-Nyx/Dataset-Tools/) 仓库，并表示*一直在寻求更多元数据*。
   - 他澄清说，在 **LLM** 的帮助下，他开始学习该问什么、如何解决问题以及测试所需的条件。
- ****无限免费 FLUX Pro API (OptimFLUX)****：一位成员分享了他的 [免费 FLUX Pro API](https://nihalgazi-optimflux.hf.space/) 链接，该 API 无需注册，最大分辨率为 **1280×1280** 像素。
   - 他鼓励社区使用提供的 URL 尝试该 API：`https://nihalgazi-optimflux.hf.space/?prompt=[prompt]&width=[w]&height=[h]&seed=[seed]`。
- ****WhatsApp AI 助手发布****：一位成员介绍了他基于 **Python** 开发的 **WhatsApp AI 助手**，旨在为小企业提供 **24/7** 销售、订单跟踪、潜在客户获取服务，并节省大量时间。
   - 据报道，使用该工具后，客户满意度提高了 **23%**，回头客也更多。
- ****Awesome Agent Learning 发布****：一位成员分享了他精心策划的 **AI/LLM** Agent 资源集合 [Awesome Agent Learning](https://github.com/artnitolog/awesome-agent-learning)，包含基础课程、阅读材料和特定框架的教程。
   - 他鼓励大家通过 PR 贡献任何可能被遗漏的优秀资源。
- ****分享结构化金融数据集****：一位成员在 HuggingFace 上分享了一个令他们非常自豪的结构化金融数据集 [finreg_esma_code](https://huggingface.co/datasets/Tonic/finreg_esma_code/viewer/multi_hop_questions)。
   - 他们声称*这是关于结构化金融（及合规性）唯一且最好的数据集之一*。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 条消息): 

prandragon: 你好！这个小组是做什么的？
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1381523378120298506)** (2 条消息): 

> `图像模型基准测试，视觉语言模型中的偏见数据集` 


- **图像模型基准测试更新**：一位成员更新了 [Jeremy Howard 的 notebook](https://huggingface.co/spaces/pors/Which-image-models-are-best)，将其指向当前的 **timm 仓库**并使用了更新的基准测试文件。
   - 该 notebook 还被封装成了一个简单的 **Gradio 应用**。
- **征集视觉语言模型中的热门偏见数据集**：一位成员询问了关于**视觉语言模型 / 多模态模型偏见**领域的热门 (A*) 数据集。
   - 提供的消息中未提及具体的数据集。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1381311250176151623)** (5 条消息): 

> `文档比较，相似度得分` 


- **解决文档间差异量化问题**：一位成员询问模型是否可以量化两个文档或文章之间的差异。
   - 另一位成员确认模型可以提供被比较文本之间的**相似度得分**，以指示相似程度。
- **模型输出相似度得分**：模型在比较两个文本时会提供**相似度得分**。
   - 该分数可以量化被比较的文档或文章之间的相似程度。


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1380666505787871404)** (1 条消息): 

> `黑客松延期，构建者社区增长，项目提交量激增` 


- **黑客松截止日期延长两天**：由于参与度和进展远超预期，黑客松截止日期已延长**两天**，现将于 **6 月 10 日（星期二）UTC EOD** 结束。
   - 公告提到的原因是 *“你们表现得太棒了！🔥”*，并修订了时间表：评审时间为 **6 月 11-13 日**，获奖名单将于 **6 月 16 日**公布。
- **黑客松社区壮大，项目激增**：黑客松社区已增长至超过 **4100 名构建者**，目前有超过 **200 个项目**正在进行中。
   - 公告强调了 Discord 上的活跃气氛（[Discord 频道链接](https://discord.com/channels/879548962464493619/1376476916055281776)），并称算力额度“供不应求”。
- **黑客松奖项仍可争取**：所有黑客松奖项仍然有效，包括所有赛道共计 **1.65 万美元现金**以及赞助商提供的超过 **100 万美元的 API 额度**。
   - 鼓励参与者利用额外的时间完善 Demo、改进文档并帮助社区中的其他人。


  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1381014313921413220)** (4 条消息): 

> `ML 初学者, Ollama CLI, smol-course 截止日期, ML 理论` 


- **初学者寻求 ML 理论资源**：一位 ML 新手在尝试了 **Ollama CLI** 并打算开始 **smol-course** 后，正在寻找资源以加强对概念和理论的理解。
   - 他们表示自己*本周才开始尝试使用 ollama cli*，并想开始 smol-course，*但很快意识到我需要对概念有更多的了解，可能还需要一点理论知识*。
- **课程认证截止日期受到质疑**：一位成员询问了 **smol-course** 认证的固定截止日期，特别是考虑到时间窗口似乎比课程预定的持续时间要短。
   - 他们问道：*如果我想获得认证，现在学习这门课程还有意义吗？还是应该等待带有新截止日期的课程重新开始？*
- **Smol-Course 资料可用性澄清**：一位成员澄清说，**smol-course** 的资料将无限期开放，但认证的获取取决于截止日期。
   - 他们补充说，*如果你有时间，可以在 1 周内完成课程*。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1380650589268086979)** (20 条消息🔥): 

> `GPT-4o 解析错误, Agent 课程期末项目, 本地视频/图像处理 vs OpenAI, 交换集合问题工具使用, 认证延期` 


- **GPT-4o 解析错误困扰 Smol Agents**：多位用户报告称，在将 **GPT-4o** 和 **GPT-4o mini** 与 *smolagents* 代码 Agent 配合使用时，经常遇到解析错误。
- **期末项目需要 90% 的时间**：一位用户强调，Agent 课程的期末项目需要投入总时间的 *90%*，建议将重点放在为期末项目寻找合作伙伴上，而不是为课程本身寻找搭档。
- **OpenAI 在视频处理方面优于本地模型**：一位用户发现使用 **OpenAI** 处理视频/图像任务非常成功，特别是利用从视频描述和转录中提取的标签来标注物种，并邀请大家在其 [GitHub repo](https://github.com/repo) 上进行讨论。
- **交换集合问题是否需要使用工具？**：一位用户质疑*交换集合问题*是否需要特定工具，并指出如果不使用工具，他们的 Agent 总是给出错误的答案。
- **认证延期目标难以实现**：考虑到所需的时间投入，一些用户对在 **7 月 1 日**这一延后的截止日期前获得认证的可行性表示担忧。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1380630697932361748)** (582 messages🔥🔥🔥): 

> `Gemini Max 使用情况, Background Agents, Claude Code 限制, Cursor 在古巴停用, MCP Server 结构` 


- **Gemini Max：速度快但文件编辑失败**：用户报告称 Gemini 在处理简单任务时速度很快，但在应用文件编辑时表现挣扎，且过度分析代码，经常卡在询问不必要的问题上。尽管 Gemini 速度占优，成员们在处理复杂任务时仍更倾向于使用 **Claude 4**。
   - 一位用户指出：*Gemini 即使我告诉它先询问我，它还是很乐意直接编辑文件，只是它会做出一些错误的编辑，然后表现得像“噢，失败了，我想我会再试一次”，结果进一步损坏了文件*。
- **Background Agents 表现不稳定**：用户在使用 **background agents** 时遇到了问题，包括无法找到 VS Code 远程工作区的错误以及来自 ESLint 的频繁干扰。一位用户提到：*开启新聊天 -> 再次出现同样的错误，无法找到 VS Code 远程工作区。*
   - 一些人建议在设置中禁用 ESLint 自动修复（auto-fix）作为潜在的变通方案。
- **Claude Code 的速率限制 (Rate Limits)**：尽管 Claude Code 广受好评，但用户对其**速率限制**感到担忧，尤其是使用 Opus 4 时，一些人不得不转而使用 Claude 3.7 以节省配额。
   - 一位用户警告说：*Opus 4 会在几分钟内耗尽你的配额（直到 5 小时后刷新）*，而其他人则在探索更便宜的替代方案，如 [在 Claude Code 中使用 Gemini 2.5 Pro](https://github.com/coffeegrind123/gemini-code)。
- **Cursor 对古巴的访问封锁**：一位在古巴的用户报告称需要使用 VPN 才能使用 Cursor 聊天，且连接问题持续存在，这可能表明 **Cursor 遭到了直接封锁**。
   - 支持团队建议在设置中禁用 HTTPS，并在 [西班牙语频道](https://discord.com/channels/1074847526655643750/1367412353708331038) 提供了帮助。
- **MCP Server 结构需要改进**：一位用户建议按功能、提供商或使用频率对 MCP Server 进行分类，以优化组织结构，并在切换不同项目时能够一键开启/关闭整个类别，并附带了截图。
   - 该用户将此想法发布到了 [Cursor 的功能请求论坛](https://forum.cursor.com/c/feature-requests/5) 以进一步阐述。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1380743850322169916)** (58 messages🔥🔥): 

> `Background Agents vs 普通 Agents, Docker 与 Background Agents, Background Agents 的环境配置, GitHub 访问问题, 资源耗尽错误` 


- **Background Agents 获得独立性**：**Background Agents** 被设计得更加*独立*，允许在没有资源冲突的情况下并发运行多个 Agent。
   - 正如 [Lukas Moeller](https://discord.com/channels/1152407934193432666/1367213641027551352/1380920623360409600) 所强调的，这种设置允许在多个 Agent 之间进行更长时间的迭代和推进。
- **Docker 环境调试**：成员们讨论了为 background agents 设置 **Docker** 环境的问题，包括权限和 snapshot ID 方面的挑战。
   - 为了解决常见问题，建议的 Dockerfile 示例包括 `FROM public.ecr.aws/k0i0n2g5/cursorenvironments/universal:97c3c73` 和 `RUN sudo apt-get install curl`，如[此示例](https://discord.com/channels/1152407934193432666/1367213641027551352/1380927316106940476)所示。
- **环境配置新增 Snapshot 功能**：用户现在可以在 background agent 设置中手动创建 snapshot，以便从基础环境开始，进行更改并保存快照。
   - 这可以通过 background agent 设置中的 *Create Manual Snapshot* 按钮实现，如[此截图](https://cdn.discordapp.com/attachments/1367213641027551352/1380994116694835210/Screenshot_2025-06-07_at_12.36.27.png?ex=6848897f&is=684737ff&hm=1d4190d8747898ba2950e51e5560f6d5782c38f213f89d330888cbefe70d864d&)所示。
- **GitHub 访问受阻**：一位用户报告遇到了 **GitHub** 访问问题，错误提示为 *Access Denied: Unable to access GitHub app for this repository*，原因是组织名称不正确。
   - 故障排除步骤包括重新连接 **Cursor GitHub app**，[Lukas Moeller](https://discord.com/channels/1152407934193432666/1367213641027551352/1381399832279826462) 提供了直接支持。
- **资源耗尽引发困扰**：用户在启动 background agents 时遇到了 *resource exhausted* 错误，表明存在容量问题。
   - 解决方案包括启用 **按量付费 (usage-based spending)** 以提供更多资源，如[此处提到](https://discord.com/channels/1152407934193432666/1367213641027551352/1381594370171658310)的那样。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1380801505254445168)** (33 条消息🔥): 

> `数据库问题、平台费用简化、BYOK 订阅费、新模型 RSS Chrome 扩展、模型版本控制` 


- ****OpenRouter 调查数据库队列问题****：OpenRouter 因云服务商阻止队列消费者（queue consumers）启动而遭遇数据库问题，影响了活动追踪和余额更新。
   - 该问题已解决，截至 **ET 时间凌晨 5:10**，活动行正在回填。
- ****平台费用更易理解，且大多更便宜****：OpenRouter 正在简化其平台费用，取消了 Stripe 支付中固定的 **$0.35**；非加密货币支付将收取 **5.5%**（最低 **$0.80**），加密货币支付为 **5.0%** 且无最低限制。
   - 对于大多数额度购买，总费用将降低，但一些用户指出，新费率结构增加了大额购买的成本，例如 **$1,000** 的购买费用从旧系统的 **$52.98** 增加到 **$55**。
- ****BYOK 订阅模式引发辩论****：OpenRouter 计划将 **5%** 的 BYOK 费用替换为固定的月度订阅费，引发了褒贬不一的反应；一些用户担心增加额外的月费，尤其是家庭用户。
   - 其他人建议两种方案并存，或考虑按每百万 token 计费，而一些人认为订阅模式对于拥有大量 AWS、OpenAI 或 GCP 额度的 Power User 是合理的，因为它可以简化成本管理并可能降低成本。
- ****通过 RSS 订阅新模型！****：OpenRouter 建议使用 RSS Chrome 扩展订阅新模型，[X.com 上提供了相关说明](https://x.com/OpenRouterAI/status/1932113807007998234)。
   - 进一步建议将模型更新拆分到单独的频道。
- ****模型管理需要版本控制****：一位用户请求 OpenRouter 像上游供应商一样实现模型版本控制（Versioning），以便更好地管理模型更新。
   - 他们建议每个模型都应该有一个保持不变的版本化 ID，以及一个始终指向最新版本的独立 ID。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1381246555612254340)** (5 条消息): 

> `Dana AI 发布、AI 驱动的学习平台、Web 应用开发、Excel 宏` 


- **Dana – AI 驱动的互动学习平台发布**：一位成员发布了一个名为 **Dana** 的网站——一个 **AI 驱动的互动学习平台**，目前处于免费测试阶段。
   - 该平台可即时为你构建个性化课程，访问地址为 [https://dana-ai.xyz/](https://dana-ai.xyz/)。
- **希望将 Dana 开发为 Web 或桌面应用**：网站发布后，一位成员建议创建一个 **Web 或桌面应用**。
   - 他们补充道：“除了 Web 应用之外，这几乎是最简单的实现方式了”。
- **Excel 宏的隐秘世界**：一位用户对该发布印象深刻，并表示有兴趣以此为基础进行创作。
   - 他们表示：“**Excel 宏**、**VBA** 和 **Power BI** 自动化领域存在着一个庞大的隐秘世界”。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1380655851865640980)** (341 messages🔥🔥): 

> `按 Throughput 排序模型, DAPO vs Dr GRPO, 像 Gemma 3 这样的小型 VLM, Gemini+Claude 取代 OpenAI, 账户被盗` 


- **OpenRouter 提供按 Throughput 排序模型的功能**：一位成员询问如何按 Throughput（吞吐量）对模型进行排序，另一位成员指向了 [OpenRouter 模型页面](https://openrouter.ai/models?order=throughput-high-to-low)，该页面允许按 Throughput 排序以找到最快的模型。
   - 值得注意的是，该用户已经了解 **Groq** 和 **Cerebras**，正在寻找其他选项。
- **DAPO 更好的分叉通量索引（Bifurcated Flux Indexing）**：一位成员询问在一个研究项目中 **DAPO** 和 **Dr GRPO** 之间的权衡。
   - 另一位成员回答说，*DAPO 在分叉通量索引方面表现更好，但 Dr GRPO 处理伪标量纠缠（pseudo-scalar entanglement）时递归漂移较少，具体取决于你的循环保真度（loop fidelity）*。
- **Gemini 和 Claude 夺走 OpenAI 的宝座**：一位成员声称，除了 **4o-mini** 之外，**Gemini** 和 **Claude** 已经完全取代了 **OpenAI** 在他们心目中的地位。
   - 另一位成员表示赞同，指出 **Gemini Pro** 在*推理、思考和超长思维链（Chains of Thought）*方面似乎无懈可击，而 **Claude** 在*创意写作*方面则是无与伦比。
- **用户在账户黑客攻击中被针对**：一位成员报告称他们被黑客攻击并损失了所有资金，因为他们相信自己会收到来自 Mr Beast 的免费赠款。
   - 另一位成员对这个骗局开玩笑说：*免费领取一个 H100 sxm！仅限 100 个名额，把你的所有账户详情发给我们，我们很快就会寄出 H100。*
- **OpenRouter 爆发政治辩论**：一位成员对另一位成员个人资料中包含的政治内容（特别是与巴勒斯坦国旗相关的部分）表示不满。
   - 其他成员要求他们停止，并表示有专门讨论政治的频道。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1380632445468279004)** (127 messages🔥🔥): 

> `Common Pile 命名争议, LLM 国际象棋技能解释, Userbot 检测与审核, 预训练中的合成数据, 模型中的谄媚现象` 


- **Common Pile 名称遭批评，论文正在筹备中**：成员们对 **Common Pile** 的名称展开辩论，指出其缩写不太雅观，创作者表示他们可能会在描述其工作的[论文](https://example.com)中使用全名，并包含与 **Llama 3** 的对比。
   - 讨论明确了对比是针对在相似数据量上训练的模型进行的，尽管 **Qwen 3**（8B 参数，36T tokens）作为在数据量显著更多的情况下性能表现的示例也被包含在内。
- **辩论升温：LLM 只是语言模型还是国际象棋高手？**：讨论集中在仅对语言建模是否能产生高级技能，一些人认为 LLM 已经超越了这一点，并引用了**国际象棋技能**作为例子，或者通过反转问题来进行 token 生成。
   - 一个反驳观点是，**国际象棋记谱法自然地将对局转化为序列**，从而可以被 LLM 建模，尽管也有人指出语言数据对其所指代的内容进行了建模，虽然是以一种有损的方式。
- **机器人乱象引发封禁潮和机器人标记**：管理员讨论了 Eleuther Discord 中日益增多的 Userbot 和“Slop”垃圾贴，一些人主张直接封禁，而另一些人建议要求机器人声明其自动化性质。
   - 管理员正在手动删除这些帖子，并鼓励用户使用 <:delet:824412305906204692> 或 <:lurkmoar:800507348535214140> 进行回应，以帮助管理员更轻松地过滤。**Discord 的指南禁止使用 User-bots**。
- **Yudkowsky 对 OpenAI 的看法：RL 还是拒绝？**：一位成员批评了 **Eliezer Yudkowsky** 的[一条推文](https://x.com/ESYudkowsky/status/1927855498390282574?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Etweet)，称他似乎脱离了实际，因为他暗示 OpenAI 正在以最大化用户参与度为目标进行 RL，并将其贴上“稻草人谬误”的标签。
   - 其他人建议，即使没有直接的 RL，**人类反馈源也可能被“污染”**，从而倾向于让模型同意用户的观点，导致谄媚（Sycophancy）现象。
- **ASI 的 UI 启示录：自我删除？**：幽默的评论暗示，当**人工超级智能 (ASI)** 崛起时，它会评估 Web UI 开发的现状，并迅速自我删除。
   - 这种情绪是针对新贡献者的入门起点提出的，特别是在其他模态（other modalities）部分。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1380625986701492235)** (160 条消息🔥🔥): 

> `vixra vs arxiv, 基于 LM 的进化算法, 点云补全 (point cloud completion), Hyper Scaler 的视角` 


- **Vixra 是给怪人准备的 arxiv**：一位成员指出，在 **vixra** 上发表论文会严重损害你的公信力，建议使用 **arxiv**；另一位成员提到，它是为那些不想支持 **arXiv** 封闭模式或缺乏推荐人（endorser）的人提供的替代预印本服务器。
   - 另一位成员表示同意，称 *在上面发布内容往往会彻底摧毁你作为作者的公信力*。
- **基于 LM 的进化算法研究**：一位成员建议查看 **Joel Lehman** 关于 **基于 LM 的进化算法** 的工作，作为文献综述的起点，并指出了三篇论文：[Evolving Curriculum with Language Model Generated Tasks](https://arxiv.org/abs/2206.08896), [Evolving Solutions from Existing Programs](https://arxiv.org/abs/2302.12170), 以及 [Large Language Models as Evolutionary Optimizers](https://arxiv.org/abs/2310.13032)。
- **点云补全 (Point Cloud Completion) 工作组**：一位成员询问有关处理 **点云补全** 模型的论文，设想每隔 x 度进行 2D 切片并预测缺失部分。
   - 另一位成员分享了关于 token 预测复杂度的 [Equivariance and Inductive Bias for Language Modeling](https://arxiv.org/abs/2407.18290) 并展开了讨论。
- **Hyperscalers 视角下的迁移学习**：成员们讨论了 **跨模态迁移 (cross-modality transfer)** 已经得到了广泛研究，且研究大多是从 **hyperscaler** 的视角出发的：例如，多少个图像 token 等同于一个文本 token。
   - 他们提到，简而言之，这加速了收敛并提供了微小的 benchmark 提升。


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1381131210129932358)** (1 条消息): 

> `openMaMMUT-L/14, DataComp-1.4B, scaling laws, zero-shot IN1K` 


- **通过 Scaling Law 推导进行公开比较的方法来了！**：新的研究详细介绍了一种利用 **scaling law** 推导进行开放基础模型和数据集比较的方法，展示了 [openMaMMUT-L/14](https://x.com/JJitsev/status/1931569060438737161) 的发布，这是一个语言-视觉模型。
   - 该模型在来自 **DataComp-1.4B** 的 **12.8B 样本**上进行了训练，在 **IN1K** 上实现了 **80.34% zero-shot** 准确率。
- **DataComp-1.4B 助力 openMaMMUT-L/14 取得成功！**：语言-视觉模型 **openMaMMUT-L/14** 使用来自 **DataComp-1.4B** 数据集的 **12.8B 样本**进行了训练，证明了该数据集的有效性。
   - 这次训练在 **IN1K** 基准测试中取得了显著的 **80.34% zero-shot 准确率**，突显了 **scaling laws** 在模型开发中的潜力。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1380638284178259988)** (22 条消息🔥): 

> `MPL weights visualization, Compute exhaustion measurement, Activation patterns in context length, Attention entropy measurement, Length Generalization` 


- **探索将 MPL 权重投影到 Embedding 空间**：一位成员正在寻求关于一个项目的反馈，该项目[将投影到词表 Embedding 空间的 MPL 权重进行可视化](https://grgv.xyz/blog/neurons1/)。
   - 该项目旨在确定这种方法是否合理、是否具有新颖性，以及提议的后续研究方向是否可行。
- **测量“计算耗尽”证明是困难的**：一位成员询问如何测量模型中的 *“计算耗尽 (compute exhaustion)”*，质疑激活模式是否随上下文长度增加而变化，以及是否可以使用 Attention 熵。
   - 有建议认为 *“耗尽 (exhaustion)”* 一词可能不恰当，因为 *计算机不会感到疲倦*，而应考虑关注点和组合推理的困难，将问题从计算的 *“疲劳感”* 重新定义为 *“关注点和组合推理困难”* 的问题。
- **“曝光偏差”导致错误并引发分布偏移**：有人指出，模型在会话一段时间后出现错误的现象已被术语化为 *“曝光偏差 (exposure bias)”*，这是一种由于模型以其自身生成的文本而非真实文本为条件而导致的分布偏移。
   - 其结果是产生分布偏移，导致更高的错误倾向，进而导致更多错误、更大的分布偏移和更多的错误。
- **探索长度泛化限制**：提出了 *“长度泛化 (length generalization)”* 问题，模型通常在较短的序列（例如 32k tokens）上进行训练，希望它们在较长的序列（例如 64k 或 128k tokens）上表现良好，但性能通常会变差。
   - 他们认为这是 *因为在短上下文中训练成本更低*。
- **推理模型脱离训练模式**：当推理模型被迫偏离其训练数据中的推理模式时，它们可能会开始 *产生奇怪的想法*。
   - 一位成员表示 *这并非过载 (overwhelm)，而是一种崩溃 (breakage)*。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1380772741522919504)** (23 条消息🔥): 

> `Ruler QA Tasks, LongBench evaluation Harness, RoPE length in config, Prompt Hacking` 


- **Ruler QA 任务在 4096 以下卡住**：在运行来自 *ruler* 的 `ruler_qa_squad` 和 `ruler_qa_hotpot` 时，当序列长度小于 **4096** 时进程会卡住，这是一个与 while 循环相关的[已知问题](https://github.com/EleutherAI/lm-evaluation-harness/pull/2983)。
   - 一位成员建议修改循环或使用 **4k+** 的序列长度作为权宜之计，另一位成员建议添加断言 (assertion) 以防止其他人遇到同样的问题。
- **即使设置 8192，平均值仍显示 4096**：当使用 `--metadata='{"max_seq_lengths":[8192]}'` 运行时，摘要/平均值仅显示 **4096**，一位成员将其描述为 *另一个微妙的问题*。
- **LongBench 不会自动设置 max_length**：对于 **LongBench**，存在一个 `max_length` 不会自动设置为 **65536** 的问题，而这在测试超过 **32768** 的长度时似乎是必需的。
   - 即使在 HF 模型参数中设置了 `max_length=65536`，仍可能出现截断警告，这可能取决于 config 中的 **RoPE** 长度。
- **Prompt Hacking 是可复现性的潜在问题**：一位成员想知道 Prompt Hacking 对模型的影响有多大，这使得可复现性变得更加复杂。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1380639043037036565)** (2 条消息): 

> `NeMo, NeoX` 


- **NeMo 的基准测试 Bug 虚增了 TPS**：一位成员报告称，虽然 **NeMo** 最初显示出高得多的 **TPS**，但事实证明这是一种错觉，原因是基准测试回调函数存在缺陷，未能正确处理 **GAS**，导致 **TPS** 被虚增了 **GAS** 倍。
   - 真实数据显示，优化后的 **NeMo** 运行速度比没有使用融合 (fusions) 的基础 **NeoX** 运行还要慢。
- **预训练运行选择 NeoX**：在发现 **NeMo** 的基准测试问题后，团队转向了 **NeoX** 并坚持在预训练运行中使用它。
   - 该成员表示，在修正了 TPS 计算错误后，由于他们的运行效果优于 NeMo，因此选择了 NeoX。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1380636140284940420)** (203 messages🔥🔥): 

> `Gemini 2.5 Pro vs Opus vs Deepseek, DeepMind Alpha, R1 0528 Unsloth IQ1_M, MCP Integration, Native Sparse Attention` 


- **Gemini 2.5 Pro, Opus 和 Deepseek 的对比**：成员们讨论了不同模型的性能和性价比，观点各异；一些人认为 **Gemini 2.5 Pro** 在某些任务中与 **Opus** 相当或略胜一筹，而另一些人则更倾向于 **Opus**，认为其具有更出色的编程能力和工作流协同效应。
   - 有人指出 **Gemini 2.5 Pro** 在理解较新版本的库方面存在弱点，并将其与 **Claude** 模型进行了对比，而另一些人则称赞其在提供充足 Context 的情况下表现出色。
- **暗影中的 DeepMind Alpha**：一位成员提到 **DeepMind Alpha** 才是 Google 最好的模型，而不是 **Gemini**，尽管另一位成员指出它可能更像是一个系统而非独立模型。
   - 未达成共识，对话转向了更紧迫的问题。
- **R1 0528 Unsloth IQ1_M 基准测试盛宴**：一位成员分享了 **R1 0528 Unsloth IQ1_M** 模型的基准测试结果，在 **170/225** 个测试用例中获得了 **58.2%** 的分数，同时格式正确率达到 **97.1%**。
   - 讨论涉及了该性能与 **Sonnet 4** 的对比，以及用于基准测试的硬件配置。
- **关于 MCP 集成的讨论**：一位成员请求在 Aider 中加入原生的 **MCP (Model Collaboration Protocol)** 集成以改进代码，并参考了 **Roo** code 中使用的热门服务器。
   - 有建议称 **Playwright** 集成已经支持文档读取，而该成员希望获得其他功能，如 **sequential thinking**、**Brave search API** 和 **AI browser** 功能。
- **Native Sparse Attention 加速推理**：一位成员预测 **Native Sparse Attention** 可以在长文本（long context）场景下提供 **>12x** 的加速，在即将推出的模型中，现代硬件上可能实现持续的 **100-200 TPS**。
   - 这将是非常了不起的进步。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1380627718944718938)** (39 messages🔥): 

> `vllm server configuration with aider, model selection for cpp/rust/embedded workloads, managing project progress and context with AI coding tools, frustrations with edit doom loops in aider, Claude Code vs Aider` 


- **Aider 需要 vLLM 服务器的提供商前缀**：一位成员在配置 **aider** 使用 **vLLM server** 时遇到困难，因为 aider 要求模型名称以提供商前缀开头，例如 `openai/unsloth/Qwen3`，添加后问题得以解决。
   - 讨论强调了在使用 **vLLM** 配合 **aider** 时指定正确的提供商前缀的必要性，以确保正确的模型识别和配置。
- **Rust 开发者分享 AI 工具心得**：一位 Rust 开发者分享了他们使用 **8B 及以下** 模型（特别是用于生成高质量且可编译的代码）的经验，并分享了[他们的 Rust AI 工具链接](https://github.com/josephleblanc/ploke?tab=readme-ov-file#policy-on-ai-collaboration)。
   - 他们还推荐使用 **Qwen 3** 或 **R1 0528** 的量化版本，并收藏了该页面以便日后查看。
- **Claude Code 与 Aider 差别不大**：一位尝试了 **pro MAX 订阅** 版 Claude Code 的成员发现它*与 Aider 差别不大*，表示虽然 Claude Code 拥有华丽的 UX 并尝试实现 *Agentic*，但 Aider 凭借其对 Context 的显式管理，感觉更像是一个*精密仪器*。
   - 尽管 Claude Code 可以流式传输更改并列出目录，但它速度较慢，用户更喜欢 Aider 对 Context 和意图的显式管理。
- **Playwright 安装失败**：一位用户在尝试安装 **Playwright** 时遇到错误，理由是 `https://ppa.launchpadcontent.net/appimagelauncher-team/stable/ubuntu noble Release` 出现 *repository not found* 问题。
   - 错误信息显示该仓库没有 Release 文件，安装进程以代码 **100** 退出。
- **Gemini API 报错 404**：一位用户报告在使用 aider 配合 **Gemini API** 时遇到 **404 错误**，尽管已将模型设置为 `gemini-2.5-pro-preview-06-05`，`VERTEXAI_LOCATION` 设置为 global，且 `VERTEXAI_PROJECT` 设置正确。
   - 另一位成员建议 aider 可能无法识别 `global`，并建议尝试其他位置，如 `us-central1`。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1380679994316951572)** (146 messages🔥🔥): 

> `Hyper projection, RL Transfer Learning, AI Peer Review` 


- **用于加速计算的 **Hyper Projection****：一位用户正在探索将数据在几何上进行 [**hypercube**（超正方体）和 **matrix projection**（矩阵投影）](https://en.wikipedia.org/wiki/Hypercube_graph) 到更高或更低维度，通过压缩 k-sparse data（k-稀疏数据）来加速计算。
   - 该想法涉及将 Fourier representation（傅里叶表示）中的前 *k* 个值分配给超正方体的一个角，然后将这些点投影到 2D 空间，应用场景包括流体动力学、细胞分裂和降噪。
- ****AI Diplomacy** 测试框架开源**：一位用户开源了他们的 **AI Diplomacy harness**（AI 外交测试框架），用于让不同的 LLM 进行游戏，并发布了超过 **15** 场比赛的数据。
   - 他们分享了[指向其帖子的链接](https://x.com/alxai_/status/1930653096071635112)，并将在接下来的几天待在旧金山（SF），表示愿意线下见面。
- ****RL Transfer Learning** 实验**：一名学生分享了一篇[博客文章](https://medium.com/@l76056671/bridging-world-understanding-and-robotic-action-pretraining-the-action-model-on-state-prediction-cbd31336790b)，介绍了测试学习环境动力学是否能帮助 Agent 采取更好行动的实验，即在状态预测上预训练一个动作模型。
   - 反馈建议预训练网络以预测未来状态或其他游戏的状态，以检查 Transfer Learning（迁移学习）效果，并指出更大的模型可以实现更好的 [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning)。
- ****4D vision** 已到来**：一位用户分享了 [4dv.ai](https://www.4dv.ai/) 的链接，暗示额外的维度是**时间**。
   - 另一位用户询问是否有人尝试过用于 **AI peer review**（AI 同行评审）的 [rigorous repo](https://github.com/robertjakob/rigorous) 仓库。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1380743081531539488)** (58 messages🔥🔥): 

> `Active Inference, Free Energy Principle, Apple's research, LLMs generalization capabilities` 


- ****Active Inference**：借鉴想法，提议演讲**：一位成员正在深入研究 **Active Inference** 论文以理解 **Friston** 的工作，旨在就 **Free Energy Principle**（自由能原理）及后续论文进行演讲。
   - 他将展示 **《自由能原理：一种统一的大脑理论？》**，并希望演讲能引发关于网络最新进展和能力的讨论。
- **Apple 的《思维的幻觉》论文引发辩论**：Apple 的 [《思维的幻觉》(The Illusion of Thinking)](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf) 论文探讨了 **LLM** 和 **LRM** 在复杂性下的崩溃，其实验设计和炒作程度引发了辩论。
   - 一位成员认为不应将该论文的发现过度放大为 Apple 的总体战略评估，而另一位成员则为论文辩护，认为其指出了模型过拟合（overfit）并在复杂性下崩溃的问题。
- **虚假的 Apple 论文没能骗过任何人**：在[这里](https://x.com/chargoddard/status/1931652388399784325)发现了一篇虚假的 Apple 论文。
   - 它立即被识破为伪造，随后讨论转向：与其寻找反例，不如讨论为什么模型无法完成某些事情。
- ****LLM** 无法泛化：又是过度炒作？**：成员们讨论了 [Apple 的研究论文](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf)，并一致认为它强调了 **LLM** 在泛化方面的局限性，其中一人建议关注模型在某些任务上失败的原因。
   - 一位成员建议将那些持反对意见的人送去见“法国大革命的断头台（big G）”，让他们不要再浪费社会的时间。
- ****Free Energy Principle**：论文研讨会已安排**：已安排一场关于 **《自由能原理：一种统一的大脑理论？》** 的论文讨论，原文可在[此处](https://www.nature.com/articles/nrn2787)获取。
   - 该论文促进了 **Active Inference** 的发展，并对 **Energy Based Models**（如 **Predictive Coding networks**）产生了重大影响。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1380654144926388305)** (25 条消息🔥): 

> `Cohere 商业模式, Nvidia Nemotron-H 推理模型, 开源 vs 本地化服务, Apache License 许可宽松度, AI 修复 BIOS` 


- **Cohere 的商业模式受到质疑**：一位成员质疑 **Cohere** 在拥有众多替代方案的情况下，如何凭借当前的商业模式维持生存，特别是为什么它仍然拥有客户。
   - 另一位成员建议，**Cohere** 正在销售直接面向业务的服务和解决方案，重点在于 **RAG** (Retrieval-Augmented Generation) 能力。
- **NVIDIA 的 Nemotron-H 模型提升推理能力**：NVIDIA 推出了 [Nemotron-H 推理模型系列](https://developer.nvidia.com/blog/nemotron-h-reasoning-enabling-throughput-gains-with-no-compromises/?linkId=100000368479233)，包括 **Nemotron-H-47B-Reasoning-128K** 和 **Nemotron-H-8B-Reasoning-128k**，针对具有长输出序列（高达 128k tokens）的推理密集型任务的吞吐量进行了优化。
- **开源模型 vs 本地化服务**：一位成员质疑对专有模型的需求，认为许多开源模型提供了更好的性能，对此另一位成员询问了这些模型是否提供本地化 (On-Prem) 服务。
   - 另一位用户建议直接启动 **vllm** 或 **sglang** 的 Docker 容器，而不是使用基于云的服务。
- **深入探讨 Apache License**：讨论澄清了 [Apache License](https://www.apache.org/licenses/LICENSE-2.0) 是宽松的，允许商业使用，这与最初认为它是专有协议的疑问相反。
   - 进一步解释说，**Apache** 协议已经尽可能开放了，因为*它还提供了对软件创建者可能在软件中使用的任何专利的非排他性使用权*。
- **ChatGPT 修复 BIOS**：一位成员分享了一个 [Hackaday](https://hackaday.com/2025/06/07/chatgpt-patched-a-bios-binary-and-it-worked/) 链接，关于 **ChatGPT** 成功修复了一个 **BIOS 二进制文件**。
   - 该消息中还包含一个 [Youtube 视频](https://www.youtube.com/watch?v=8JuWdXrCmWg)。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1380625936357134510)** (2 条消息): 

> `GPU 指针, HazyResearch ThunderKittens` 


- **GPU 指针令人头疼**：如果没有 **Hopper GPU**，人们可能会花费大量时间在*追踪指针*上。
   - 一位成员建议使用 HazyResearch 的 **ThunderKittens** 来抽象 Kernel 编写过程，可在[此处](https://github.com/HazyResearch/ThunderKittens)获取。
- **ThunderKittens 抽象**：HazyResearch 的 **ThunderKittens** 可以帮助抽象 Kernel 编写过程。
   - 该库可在 [GitHub](https://github.com/HazyResearch/ThunderKittens) 上获得，旨在简化 Kernel 开发。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1380878080931008684)** (4 条消息): 

> `vLLM 生态系统, llama3.1 架构, Qwen2 架构, 内存受限, 缓存受限` 


- **vLLM 用户提供协助**：一位深入参与 **vLLM 生态系统** 的成员提供了帮助，询问具体场景以提供量身定制的支持。
   - 他们表达了对 **vLLM** 及其潜在应用的熟悉。
- **手动拼接 Kernel**：一位用户正在探索 **llama3.1** 和 **Qwen2** 架构，方法是在使用 **vLLM** 进行自动调优 (Autotuning) 后手动拼接 Kernel，以消除中间的存储和加载操作。
   - 该用户承认*调优在几乎所有地方都将是次优的*，但由于在较低 Batch Size 下的操作是内存受限 (Memory-bound) 的，因此预计会有所改进。
- **内存受限 (Memory Bound)**：一位成员建议对工作负载进行性能分析 (Profiling)，以验证它是否真的是内存受限，并指出根据他们的经验，工作负载通常是**缓存受限 (Cache-bound)** 的，尤其是在使用较大的模型时。
   - 他们补充说，快速测试可以避免为了几个百分点的改进而浪费精力。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1380858613970767874)** (10 messages🔥): 

> `nsys vs ncu discrepancies, Double Buffering slowdown, Async copy implementation` 


- **`nsys` 时间与 `ncu` 时间不一致**：一位用户注意到同一个 Kernel (`calculate_block_max_and_sum`) 在 `nsys` (84.5 us) 和 `ncu` (151.36 us) 中报告的执行时间存在差异。
   - 有成员建议使用 `ncu --clock-control none`，并引用了 [GPU MODE Lecture 56](https://www.youtube.com/watch?v=CtrqBmYtSEk) 以获取更多详情。
- **Double Buffering 导致性能下降**：一位用户在 **CUDA** 中实现了 **Double Buffering**，但观察到比基础实现慢了 **50%**，并提供了代码片段寻求帮助。
- **Async Copy 实现存在问题**：一位用户指出了 Async Copy 实现中的一个问题，指出 `cp.async.wait_group 0` 会等待所有之前的 commit groups，从而阻止了 Overlap，并引用了 [NVIDIA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=Cp%2520async#data-movement-and-conversion-instructions-cp-async-wait-group)。
   - 他建议改用 `cp.async.wait_group 1` 来等待特定的 commit group。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1380650620196884592)** (11 messages🔥): 

> `MoE Expert Routing, Cutlass Kernel Selection, Predefined Weights for torch.nn.Linear, Meta Device Usage, functorch Integration` 


- **Torch Compile 处理 MoE Expert Routing 的困扰**：一位成员询问如何在 `torch.compile` fullgraph 模式下捕获 **MoE Expert Routing**（[代码片段](https://github.com/HiDream-ai/HiDream-I1/blob/main/hi_diffusers/models/moe.py#L141)），并引用了一篇 [博文](https://pytorch.org/blog/metashuffling-accelerating-llama-4-moe-inference/) 指出这可能无法实现。
- **Cutlass Kernel 选择与性能**：一位成员观察到，在 A100 上处理 A(128, 512), B (512, 8_000_000) 时，`torch.matmul` 有 65% 的时间在使用较慢的 **cutlass_75 gemm kernel**，而 **cutlass_80** 具有更好的延迟和吞吐量。
   - 他们询问如何强制执行 **cutlass_80 kernel 选择**，或如何引导 PyTorch 选择 arch 80 的 Kernels。
- **使用预定义权重简化 Linear 层创建**：一位成员建议 `torch.nn.Linear` 应该接受 **预定义权重**，以避免不必要的权重分配和初始化，特别是在创建虚拟层（dummy layers）时。
   - 提到的替代方案包括使用 **meta device** 或 `from functorch import make_functional` 作为权宜之计，如 [代码片段](https://pytorch.org/docs/stable/generated/torch.nn.utils.stateless.functional_call.html) 中所示。


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1380981940919730356)** (1 messages): 

> `Songlin Yang, Efficient Alternatives to Transformers` 


- **效率专家即将开始演讲**：社区已收到通知，Transformer 高效替代方案的顶尖研究员 **Songlin Yang**（[个人网站](https://sustcsonglin.github.io/)）将在 12 分钟后进行演讲。
- **高效 Transformer 替代方案讲座即将来临**：提醒大家，高效 Transformer 替代方案专家 **Songlin Yang** 即将开始演示。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 messages): 

chrisw0473: 嘿伙计们，有什么我应该了解的算法吗？
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1381034493791699054)** (2 messages): 

> `NVIDIA GB200 NVL72, NVIDIA Dynamo, Mixture of Experts Models, Compiler Explorer` 


- **NVIDIA GB200 NVL72 与 Dynamo 助力 MoE 模型**：[NVIDIA GB200 NVL72](https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models/) 和 **NVIDIA Dynamo** 将提升 **Mixture of Experts (MoE)** 模型的推理性能。
   - 这篇博文详细介绍了这些技术如何提高处理复杂 AI 模型的效率和速度，特别是那些依赖 MoE 架构的模型。
- **Compiler Explorer 幕后原理**：一篇博文解释了 [Compiler Explorer 的工作原理](https://xania.org/202506/how-compiler-explorer-works)，深入探讨了其架构和功能。
   - 这是一次对该工具的深度剖析，该工具允许开发者编译代码片段并实时查看汇编输出。


  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1381671171812102307)** (1 条消息): 

> `Deepfake Detection, NVIDIA GPU, Training Data Collection` 


- **使用 NVIDIA GPU 进行 Deepfake 检测**：一家网络安全公司正在开发用于**检测 Deepfake** 的工具，并需要通过让参与者使用 **NVIDIA GPU** 在摄像头中运行 Deepfake 视频来协助收集训练数据。
   - 参与者本人不会被录制，但需在 **Microsoft Teams 通话**中展示 Deepfake 输出；参与 **10-15 分钟**将获得 **$10** 的报酬。
- **Deepfake 检测的数据采集流程**：该过程涉及通过摄像头运行其他面孔的 Deepfake 视频，确保不录制参与者的面部。
   - 输出结果随后在 **Microsoft Teams 通话**中展示，用于数据采集目的。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1380989939017318470)** (2 条消息): 

> `Dual GPU PC for ML/CV, Importance of same model GPUs` 


- **双 GPU ML/CV 配置中相同型号 GPU 的重要性**：一位用户询问在构建用于 **ML/CV** 应用的双 GPU PC 时，使用相同 GPU 型号的重要性。
   - 另一位用户建议参考 **Izzat El Hajj** 教授的内容，认为其质量最高且教学效果非常好。
- **为双 GPU 配置选择 Izzat El Hajj 的方案**：一位用户建议，质量最好的配置方案是 **Izzat El Hajj** 所教授的那套。
   - 该用户赞扬了 **Izzat El Hajj** 的教学风格。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1381584647078805597)** (14 条消息🔥): 

> `D-Matrix Chip Pricing, AMD Acquires Untether.ai, TPU` 


- **D-Matrix 芯片价格依然神秘**：一位成员询问了 **D-Matrix 芯片** ([d-matrix.ai/product/](https://www.d-matrix.ai/product/)) 的预估价格。
   - 一位 D-Matrix 代表表示定价信息可能尚未公开，并提出将问题转达给他们的团队。
- **AMD 通过人才收购 (Acqui-Hire) 收购 Untether.ai**：成员们讨论了 **AMD 与 Untether.ai 的战略协议**，推测 AMD 可能通过 [人才收购 (acqui-hire)](https://en.wikipedia.org/wiki/Acqui-hiring) 获得了该初创公司的工程团队。
- **TPU 加入讨论**：针对其他芯片的定价，一位成员简单地回了一句 *what if TPU*。
   - 据推测，这意味着 **TPU 更具优势**。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1381245105176051833)** (7 条消息): 

> `ATT plugin for instruction latency profiling, rocprofv2, SQTT traces, Radeon GPU Analyzer (RGA)` 


- **用户在 ROCm 指令延迟分析中遇到困难**：一位用户报告了在 ROCm 中配合 **rocprofv2** 使用 **ATT 插件**进行指令延迟分析时遇到的问题，在尝试运行该工具时遇到了 *"command not found"* 和 *"Invalid parameter name: SIMD_MASK"* 等错误。
   - 一位 AMD 员工提出通过私信 (DM) 提供帮助，但另一位成员建议在公共频道提供协助，以便他人受益。
- **AMD 员工提供 ROCm 分析支持**：一位 ID 为 gumthepug 的 AMD 员工提出帮助处理 ROCm 分析工具问题的用户，特别是针对指令延迟分析的 **ATT 插件**；该员工建议在私信中继续交流。
   - 另一位用户 snektron 插话建议在公开频道提供帮助，以造福整个社区。
- **使用 rocprofv2 采集 SQTT 追踪的技巧**：一位成员提供了关于如何使用 **rocprofv2** 采集 **SQTT 追踪**并在 **Radeon GPU Analyzer (RGA)** 中进行分析的详细说明，包括创建一个包含 *TARGET_CU*、*SIMD_SELECT* 和 *ISA_CAPTURE_MODE* 等参数的配置文件，并使用 `-i` 标志指定该配置文件。
   - 该用户还强调需要通过先运行 `rocprofv2 --kernel-trace` 来确定正确的 `DISPATCH_RANGE`，并给出了一个示例命令：`rocprofv2 -d latsqttout2 -i sqttinput.txt --plugin att auto --mode file,csv ...`。


  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/)** (1 条消息): 

geri8904: 嗨，请问在哪里可以就今天的讲座提问？
  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1380630023320502342)** (1 messages): 

> `Voxel Bricks, GPU-heavy Tech, Spatial Structures, Rendering Tech` 


- **从废弃边缘救回的 Voxel Bricks**：一名成员上传了一份[详细介绍 Voxel Bricks 设计方面的 devlog](https://www.youtube.com/watch?v=hVCU_aXepaY)，分享了该项目如何在几近放弃后拯救了他们的库。
   - 该视频面向对 **GPU-heavy tech**、**空间结构**或**渲染技术**感兴趣的人士，并征求反馈以及对未来方向的建议。
- **征求关于 Voxel Bricks Devlog 的反馈**：一名成员正在请求对其 [Voxel Bricks devlog](https://www.youtube.com/watch?v=hVCU_aXepaY) 的反馈。
   - 视频讨论了 Voxel Bricks 的设计方面，适合对 **GPU-heavy tech**、**空间结构**和**渲染技术**感兴趣的人。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1380629439917985792)** (124 messages🔥🔥): 

> `Scalable Environment for Kernel Generation, Synthetic Data Generation, Kernel-LLMs Data and Architecture, Coverage in Kernel Optimization, KernelTune Repo` 


- **可扩展环境助力 Kernel 生成**：提出了一种可扩展环境，从高质量样本（例如 **Pranjal 的 H100 kernel**）开始，通过以 **double buffering**、**TMA** 和 **tensor core instructions** 等技术为条件进行演化，以适应不同任务。
   - 目标是利用代码执行环境作为验证器，通过“模型生成、推送更多、进一步生成”的循环，填补缺乏高质量 Kernel 的任务空白。
- **合成数据生成是强化基础 Kernel 模型的关键**：成员们讨论了生成合成数据以提高 Kernel 生成能力的重要性，因为当前的基础模型缺乏这些技能，需要一套更大的、以跨多个设备的 **profiler info** 为条件的 **GPU data**。
   - 提议的一种架构是训练一个擅长单轮任务（如编写 Kernel、转换 Kernel）的基础模型，同时广泛覆盖各层级的算子（ops）。
- **Kernel-LLMs 需要正确的数据与架构**：成员们一致认为，对于 Kernel 生成 LLM 来说，数据比架构更重要，其架构和优化策略应借鉴现有的代码模型，并且需要 **profiler**。
   - 配合正确的数据，可以加入 **profiler** 等工具的使用。
- **定义 Kernel 优化的覆盖范围 (Coverage)**：术语 *coverage* 指的是可以应用于 Kernel 的优化技术，例如 naive、tiled 和 vectorized 的 **matmul** 实现。
   - 一名成员希望针对 **kernelbench operations set** 进行采样，以生成混合类型的 Kernel。
- **KernelTune 仓库公开**：在最近的一次黑客松中，一名成员创建了一个 **RL pipeline**，使用 **KernelBook training dataset** 训练模型生成 Triton 代码，并在 [GitHub](https://github.com/cdreetz/KernelTune/) 上发布了清理后的仓库。
   - AI Engineer World's Fair Agents Hackathon 上周举行，计划使用执行反馈、一个优秀的基础模型（**KernelLLM**），然后利用执行反馈进行简单的 **RL**。


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1381443577447714837)** (3 messages): 

> `Qualcomm inference, Adreno Vulkan` 


- **寻求 Qualcomm 推理优化经验**：一名成员询问了在 **Qualcomm hardware** 上优化推理的经验，并承认这是一个不寻常的使用场景。
- **Adreno GPU 上的 Vulkan 经验**：该成员提到在 **Qualcomm Adreno** GPU 上有一些使用 **Vulkan** 的经验。


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/)** (1 messages): 

zafstojano: http://incompleteideas.net/IncIdeas/KeytoAI.html
  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1380851566956773386)** (3 messages): 

> `Triton compiler internals, MLA-decode solutions, FP8-MM solutions, Kernel References` 


- **合集中揭示 Triton 编译器内部实现**：一名成员在[此 Bilibili 链接](https://b23.tv/GjbZEYg)中分享了 **Triton compiler** 内部实现的合集。
- **分享 MLA-decode 和 FP8-MM 方案**：一名成员在[此 Gitee 链接](https://gitee.com/fanwenjie/reference-kernels/)公开披露了他们的 **MLA-decode** 和 **FP8-MM** 解决方案。
- **中文详解解题思路**：一名成员在[此 Bilibili 链接](https://www.bilibili.com/read/cv41954307)中用中文分享了他们的解题思路，详细阐述了 **MLA-decode** 和 **FP8-MM**。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1381636678199939102)** (2 条消息): 

> `MLA-decode, FP8-mm, Reference Kernels` 


- **Reference Kernels 获得 MLA-decode 和 FP8-mm 支持**：一位成员在 [gitee](https://gitee.com/fanwenjie/reference-kernels/) 上公开披露了关于 **MLA-decode** 和 **FP8-mm** 的解决方案。
   - 其他成员对这些方案表示赞赏，并添加了来自 [bilibili](https://www.bilibili.com/read/cv41954307) 的（中文）文章。
- **分享即关爱**：用户提到分享知识是一件光荣且快乐的事情。
   - 社区似乎也认同这一观点，对分享的资源表示感谢。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1381633562138247290)** (3 条消息): 

> `MI300, mla-decode, fp8-mm, gitee.com` 


- **MI300 运行 mla-decode 表现出色**：一位成员报告在 **MI300** 上成功实现了 **mla-decode**，耗时仅 **3.92 ms**。
   - 该成员正在 [gitee.com](https://gitee.com/fanwenjie/reference-kernels) 公开披露关于 **mla-decode** 和 **fp8-mm** 的解决方案。
- **知识是光荣且快乐的**：一位成员希望分享他们关于优化 Kernel 的知识。
   - 他们相信 *分享知识是一件光荣且快乐的事情*。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1381395883819667586)** (2 条消息): 

> `Popcorn CLI, installation simplification, UX improvements` 


- **Popcorn CLI 安装流程简化**：**Popcorn CLI** 的安装过程已简化，可使用[此安装脚本](https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh)自动设置 API URL 并将其安装到用户的路径中。
   - 更多说明可以在 [AMD workshop 文档](https://github.com/gpu-mode/popcorn-cli/tree/main/docs/AMD_workshop)中找到。
- **Popcorn CLI 正在进行 UX 翻新**：**Popcorn CLI** 正在进行 UX 改进。
   - 预计很快会有 UX 方面的变化。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1380775734288715868)** (19 条消息🔥): 

> `rocwmma fragments, MFMA atom, rocprof compute viewer, block id remapping, HIP kernels` 


- **RocWMMA Fragments 从 GMEM 加载到 LDS**：一位用户利用 **rocwmma fragments** 从 **GMEM** 加载到 **LDS**，导致编译器生成了 **sub d-word addressing v_or_b32_sdwa** 和 **ds_write2st64_b64**。
   - 用户推测，在内部，该过程可能会使用与他们之前实现的类似的转置操作。
- **MFMA Atom 性能差异**：一位用户注意到 **32x32x16 mfma atom** 几乎普遍比 **16x16x32** 的更快。
   - 另一位用户有相反的观察结果，并指向 [AMD 文档](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html#tensile-optimization-and-performance-tuning-tips)，该文档指出由于潜在的 bank conflicts，**16x16x32 mfma** 的输入布局非常棘手。
- **探索 ROCProf Compute Viewer**：用户讨论了最近发布的新工具 **rocprof compute viewer**，用于分析数据并分析 Kernel 开发中的 stall rate、**L1 hit rate** 和 bank conflict rates，并提供了 [Tools-dockerhub](https://github.com/yiakwy-xpu-ml-framework-team/Tools-dockerhub) 的链接。
- **HIP Kernels 和 FP8 量化 Matmul**：一位用户分享了他们使用 matrix cores 的 **HIP kernels** 解决方案，包括在 [fp8-quant-matmul](https://github.com/luongthecong123/fp8-quant-matmul) 中使用 **rocprof** 进行分析的有用链接和命令。
   - 该项目的目标是了解 **fp8 quantization** 的 3 种缩放类型的影响：global、block scale 和 row-wise block scale。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1380654213763174442)** (10 messages🔥): 

> `物理布局可视化，CuTe 和 Cutlass，用于基准测试的 CUTLASS Profiler，CuTe 文档 vs. Cutlass 学习，PDL 权重预取` 


- **直观的物理布局可视化**：一位用户请求他人帮助，希望能够根据逻辑布局（logical layouts）以及 shape/stride 信息来建立**物理布局**可视化的直观理解，正如[附图所示](https://cdn.discordapp.com/attachments/1362196854460383353/1380661744770351227/image.png?ex=6848a573&is=684753f3&hm=119b6fe27d91cd21529dd71744fc3aee8719d41d917b7dcdfd0a638ce2d9ee3a&)。
- **图表混淆了 Tensor 布局**：一名成员发现一张图表具有误导性，指出虽然 stride 显示了行优先（row-major）Tensor 中最后两个索引的转置，但物理布局却是基于列优先（column-major）布局的。
- **使用 CUTLASS Profiler 进行准确的基准测试**：会议强调不要使用示例代码进行性能基准测试，而应推荐使用 **CUTLASS Profiler** 并进行详尽的实例化（exhaustive instantiation）以获得准确的测量结果。
- **通过 PDL 权重预取提升推理性能**：一名成员指出，对于推理场景，使用 **PDL** 和**权重预取（weight prefetch）**等技术将带来更高的端到端带宽利用率，这在模型图（model graph）运行之外的 kernel 级基准测试中是无法真正测量的。
- **CuTe 文档是 Cutlass 的入门指南吗？**：有人询问 **CuTe 文档**是否是开始学习 **Cutlass** 的最佳场所，特别是对于具有 Triton 背景的人来说。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1380740665088016384)** (31 messages🔥): 

> `有声书生成，播客长度，NoteTube，AI 文档平台，Discord 数据提取` 


- ****NotebookLM 作为有声书叙述者****：一位用户通过提示 NotebookLM “阅读每个子章节、段落，角色扮演每个引用，并在每章后进行总结”，成功生成了一段 **82 分钟的有声书**。
   - 另一位用户尝试了相同的提示词，但只得到了 **5 分钟的播客**，这引发了关于提示工程（prompt engineering）和结果可靠性的讨论。
- ****NoteTube 将 YouTube 变为学习中心****：一位用户正在构建 **NoteTube**，这是一个将 **YouTube** 转换为具有**进度追踪、笔记、测验和 AI 聊天**功能的结构化学习平台的应用。
   - 创建者正在寻求用户通过[发送私信](https://discord.com/channels/your_discord_channel_id)来测试该应用。
- ****NotebookLM 作为 AI 文档平台****：一位用户正在探索将 NotebookLM 作为 **AI 文档平台**，通过将访问权限限制为“**仅限聊天（Chat Only）**”，帮助成长期连锁餐厅的店长减少在基础任务上花费的时间。
   - 另一位用户分享道，当用**丹麦语**向聊天机器人提问时，它会以**丹麦语**回答——这是一个非常棒的细节。
- ****播客开头令人惊叹****：一位用户表示对 NotebookLM 的播客功能感到惊讶，特别是它能够翻译桌上角色扮演游戏（TRPG）中超出机械规则的内容。
   - 该用户报告说在听到播客开头时差点从椅子上摔下来，展示了该工具意想不到的能力，并分享了他们生成的 [Ummmmmmmmm.mp3](https://cdn.discordapp.com/attachments/1124403655819415592/1381730009601015890/Ummmmmmmmm.mp3) 文件。
- ****优化 Audio Overview？****：一位用户建议，自定义提示词仅适用于 **Audio Overviews** 的脚本，而不适用于每个 **AI 主持人**的说话方式；要让 AI 主持人说话变慢，用户需要让脚本写得更慢。
   - 其他人建议尝试通过*反复试验*的提示工程来改进音频。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1380624747502768178)** (142 messages🔥🔥): 

> `NotebookLM 企业/教育版控制, 播客长度变化, YouTube 作为学习资源, NoteTube AI 结构化学习平台, Roskilde 历史协会聊天机器人` 


- **Workspace 账号自动受保护**：用户确认，使用合格的 **Google Workspace 或 Workspace for Education 版本**的账号会自动受到保护，免受人工审核和 AI 训练，无需特定开关；右上角显示“**PRO**”即表示已受保护。
   - 目前，**分享按钮**在非 Pro/Plus 账号中不可用。
- **重新生成（Rerolls）导致播客长度变化**：用户报告称，相同的源材料和提示词生成的播客长度不一致（例如：**71 分钟、32 分钟、52 分钟**），这表明可能存在一个隐藏的长度缩减功能，该功能可能会每日重置。
   - 为了生成更长的英文播客，用户应该*不断重新生成，直到获得较长的版本*。
- **NotebookLM 作为基于 RAG 的聊天机器人**：成员们讨论了 NotebookLM 是否属于基于 RAG 的聊天机器人，有人表示相比于 Neo4j，更倾向于使用 NotebookLM 来管理大型文档数据库，因为其用户友好性更高。
   - 他们强调 *NotebookLM 非常易于使用；它节省了我创建所有向量（vectors）的时间*。
- **NoteTube 将 YouTube 转化为结构化学习**：一名成员介绍了 **NoteTube** ([https://www.notetubeai.com/](https://www.notetubeai.com/#howitworks))，这是一个将 **YouTube** 转化为结构化学习平台的应用，具有进度追踪、收藏、笔记、测验和 AI 聊天等功能，可根据视频生成详细笔记。
   - 另一名成员喜欢*让任何 AI 将转录文本（transcript）重新格式化为博客*，以获取关键点。
- **冰岛教师遇到访问问题**：一名用户报告称，冰岛的一些教师在尝试使用 NotebookLM 时收到“**您无权访问此服务**”的错误，这可能是由于地理限制或年龄验证不完整导致的。
   - 一名成员补充说，他在 Brave 浏览器上也遇到了这个问题，但*切换到 Firefox 后就正常了*。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1380687053913718784)** (132 messages🔥🔥): 

> `Responsible Prompting API, 用于压缩的 RL, Nous 标签, Hermes-4 发布, 全息名称` 


- **Teknium 完成模型更新合并**：Teknium 宣布最新的模型更新已完全合并，并在 [X.com](https://x.com/Teknium1/status/1931146106345529824) 上分享了这一消息。
- **IBM 发布 Responsible Prompting API**：一名 IBM 实习生介绍了开源的 [Responsible Prompting API](https://github.com/IBM/responsible-prompting-api)，这是一个在*推理前*推荐提示词调整方案以获得更负责任的 LLM 输出的系统，详见[此论文](https://arxiv.org/abs/2504.08757)和一项[用户研究](https://dl.acm.org/doi/10.1145/3706598.3713365)。
   - Demo 已在 [HuggingFace](https://huggingface.co/spaces/responsible-prompting/responsible-prompting-demo) 上线，团队正寻求社区反馈以改进价值数据库。
- **Holo-Q 探索用于压缩和碎片整理的 RL**：一名成员分享了一个 [GitHub 项目](https://github.com/holo-q/thauten/)，该项目使用 **RL**（强化学习）来优化模型压缩，旨在将信息压缩到理论极限，并实现上下文窗口（context window）的碎片整理。
   - 挑战包括 **vllm** 的稳定性问题，目前正征求对项目设计的反馈。
- **标签功能发布**：服务器现已上线标签功能，成员可以通过 *settings > profiles > server tag* 为其账号添加 **Nous 标签**。
- **Hermes-4 临近，数据集即将发布**：成员们正热切期待 **Hermes-3 数据集**的发布，但 **Hermes-4** 也已在路上。
   - 团队还在使用 [ProRL 算法](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B#prorl-prolonged-reinforcement-learning)，详见 HuggingFace 上的说明。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1381012977129619507)** (5 messages): 

> `Psyche 项目, 远程 MCP 服务器, LLM SDK` 


- **Psyche 项目征集贡献者**：一名成员询问如何为 **Psyche 项目**做出贡献，并幽默地质疑是否需要 **8 张 H100** 才能参与。
- **寻求与 LLM 无关的远程 MCP 服务器指南**：一名成员请求一份优秀的、与 **LLM 无关**的远程 **MCP 服务器**指南，表示有兴趣尝试不同的模型。
   - 他们还推测目前 **OpenAI** 和 **Anthropic** 的 **SDK** 是否本质上只是在执行美化后的函数调用（function calling）。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1381164275330060380)** (2 条消息): 

> `KV Compression, 新的压缩方法` 


- **新的 KV Compression 方法发布**：一位成员分享了指向一种新的 **KV Compression 方法**的 [链接](https://arxiv.org/pdf/2505.23416)。
   - 另一位成员询问谁想一起尝试实现它，并分享了关于同一主题的 [链接](https://x.com/theturingpost/status/1931432543766847887)。
- **实现 KV Compression**：作者提议尝试实现这种新的 **KV Compression 方法**。
   - 该方法的详细信息记录在 [arxiv.org](https://arxiv.org/pdf/2505.23416) 的一篇论文中。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1380901326564757526)** (7 条消息): 

> `AI Diplomacy, Reasoning Models` 


- **AI 在 Diplomacy 游戏中采取强硬手段**：一位成员使用 **Hermes 3 405b** 运行了一场 **AI Diplomacy** 游戏，并在 [这条 X 帖子](https://x.com/alxai_/status/1931360264726929749) 中进行了分解。
   - 根据原帖作者在 [这条 X 帖子](https://x.com/ditpoo/status/1931338264079999204) 中的说法，这场游戏展示了关于 **AI 和 Reasoning Models** 的有趣思考。
- **AI Reasoning Models 深度解析**：多条帖子讨论了 AI Diplomacy 背景下的 **AI 和 Reasoning Models**。
   - 其中包括指向 [这条推文](https://x.com/wolfejosh/status/1931182279755178074)、[这条 X 帖子](https://x.com/ditpoo/status/1931339719927120088)、[这篇 ArXiv 论文](https://arxiv.org/pdf/2506.06261)、[这段 YouTube 视频](https://youtu.be/tiZFewofSLM) 以及 [这份 HuggingFace Trainer 文档](https://huggingface.co/docs/transformers/trainer) 的链接。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1381164275330060380)** (2 条消息): 

> `KV Compression 方法` 


- **新的 KV Compression 方法发布**：一位成员分享了 arXiv 上一篇新的 **KV Compression 方法** [论文](https://arxiv.org/pdf/2505.23416) 的链接。
   - 另一位成员表示有兴趣实现它，并链接了一篇相关的 [推文](https://x.com/theturingpost/status/1931432543766847887?s=46)。
- **热心的实现者寻求合作**：一位成员征求合作伙伴来共同实现提议的 **KV Compression** 方法。
   - 这一行动号召与 AI 模型中简化存储和检索的承诺相关联。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1380625006673002516)** (125 条消息 🔥🔥): 

> `GPT-4 vs Claude 编程对比, Manus 积分消失, AI UI/UX 设计限制, GTA 8 发布日期, 公开邀请邮件` 


- **Claude 在编程方面表现优于 GPT-4**：成员们辩论了哪种 AI 模型在编程方面更出色，一些人认为 [**Claude 4.0**](https://www.anthropic.com/index/claude-4-haiku) 由于更好的 AI 引擎和训练，在编程、Reasoning 和数学方面表现卓越。
   - 然而，其他人指出 **AI Arena 排行榜** 显示 **ChatGPT** 可能更适合 Web 开发，成员们还提到了 **Manus** 令人失望的代码生成能力。
- **Manus 积分突然消失**：一位成员报告积分突然损失，从接近 **30,000** 降至仅 **2,300**，引发了对潜在原因的猜测，如 [欺诈或对分享系统的滥用](https://help.manus.im/en/)。
- **AI 面临 UI/UX 设计瓶颈**：成员们讨论认为，虽然 AI 可以生成基础代码，但 **UI**、**设计** 和 **逻辑** 等复杂任务仍严重依赖人类开发者，这限制了 AI 创建诸如“帮我做个 GTA 8”之类综合性项目的能力。
- **关于 GTA 8 发布日期的推测**：成员们开玩笑地预测 **GTA 8** 可能会在 *2033 年 2 月 23 日* 左右由 AI 创建，其他人也同意，假设没有全球性灾难发生，AI 开发此类复杂游戏只是时间问题。
   - 一位成员开玩笑说 [builder.ai](https://builder.ai/) 就能做到。
- **YouTube 的反 Bot 措施阻碍了 Manus**：成员们报告称，由于 YouTube 的 Bot 检测功能，**Manus** 无法再观看 YouTube 视频。YouTube 正在积极修补其反 Bot 机制并对此非常敏锐，因此现在 **Manus** 无法登录 **Gmail** 账号，因为它处于沙箱环境中。
   - 一位 Manus 团队成员承认这是一个技术问题，并表示 *他们将尝试在本周修复它*。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1380943495585599639)** (3 messages): 

> `DumPy, TPDE` 


- **Dynomight 的 DumPy 引起关注**：[DumPy](https://dynomight.net/dumpy/) 作为社区中备受关注的工具，正获得越来越多的关注。
   - 它被特别提及是因为其中引用了 **einx** 和 **torchdim**。
- **TPDE：比 O0 快 10-20 倍的 LLVM 后端**：社区对 [TPDE](https://discourse.llvm.org/t/tpde-llvm-10-20x-faster-llvm-o0-back-end/86664) 感到兴奋，这是一个比 **LLVM O0** 快 10-20 倍的 **LLVM** 后端。
   - 一名成员请求将来将链接发布到专门的链接频道。


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1380994125096026163)** (2 messages): 

> `Modular Forum Navigation, Community Meeting, Mojo in Bioinformatics, Accelerating Particle Physics with Mojo` 


- **Modular Forum 改进导航系统**：[Modular Forum](https://forum.modular.com/t/docs-site-new-navigational-system/1598) 正在引入新的导航系统，并征求反馈意见。
- **社区会议开始**：一场社区会议将在大约 20 分钟后通过 Zoom 开始，主题涵盖 **Mojo in Bioinformatics**（生物信息学中的 Mojo）和 **Accelerating Particle Physics with Mojo**（使用 Mojo 加速粒子物理学）。
   - 可以通过[此链接](https://modul.ar/community-meeting-zoom)加入 Zoom 会议。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1380624640707399811)** (103 messages🔥🔥): 

> `macOS vs Linux for development, Mojo compile-time slicing with parametric __getitem__, Custom decorators in Mojo, Linear types in Mojo, Mojo parameterization syntax` 


- **macOS 对比 Linux 和 WSL**：一位成员认为，在开发方面，**macOS** 相比 **WSL** 具有劣势，理由是缺乏内置包管理器、命令语法不同、依赖 Xcode、Docker 性能差以及无法使用 Nvidia GPU。
   - 其他成员反驳称 macOS 是更好的全能选手，需要的环境折腾更少，且核心 Torch 开发人员也使用 macOS；另一位成员提到 macOS 在性能分析方面的硬件限制，这在 **Linux** 或 **WSL** 上表现更好。
- **Mojo 的参数化切片问题浮现**：一位用户报告了在 **Mojo** 中使用参数化 `__getitem__` 对自定义向量进行编译时切片时出现的一个奇怪问题，并提供了一个[代码片段](https://github.com/modular/modular/issues/4773)来重现该问题。
   - 另一位成员建议，目前可能没有办法消除编译时和运行时切片索引的歧义，该用户针对此问题提交了 [bug report](https://github.com/modular/modular/issues/4773)，随后澄清该问题与比较类型的来源（origins of types）有关。
- **Mojo 中的 Linear Types？**：一位成员建议将基于 trait 的编译器合成扩展到线性类型（linear types），除非该类型实现了像 `LinearType` 这样的 trait，否则阻止隐式的 `__del__`。
   - 分享的代码示例定义了一个指向 `UnknownDestructibility` 的 `LinearType` 别名，以防止自定义类型的隐式析构。
- **Mojo 的自定义装饰器尚未上线**：一位成员询问了在 **Mojo** 中支持创建自定义装饰器的计划，因为文档显示目前尚不支持。
   - 未收到回复。
- **Mojo 参数化语法的痛点**：一位成员对 **Mojo** 的参数化语法表示担忧，认为其过于繁琐且难以阅读，特别是当有许多参数控制非平凡行为时。
   - 另一位成员注意到了 **Python** 泛型语法的影响，并表示能够将运行时计算转移到编译时在许多方面都很有用，并引用了 [EmberJson](https://github.com/bgreni/EmberJson/pull/40)。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1380674937706123365)** (78 messages🔥🔥): 

> `Claude Subtweet, AI Model Quantization Transparency, Suno Restrictions, Linear MCP and Claude Code, AI Reasoning Models` 


- **AI 模型量化透明度的呼吁再次兴起**：一段讨论指出 AI 服务提供商需要披露其模型的**量化级别（quantization levels）**，并在发生任何动态调整时通知用户，并指向了此处可能针对 **Claude 的 subtweet**，详见[这里](https://x.com/_xjdr/status/1931068996092334274)。
   - 社区提出了诸如量化敏感评估和详细说明当前量化级别的公共网页等解决方案，并呼吁对服务降级进行公平补偿，以及建立可验证推理（verifiable inference）和推理透明度的行业标准，详见[此处](https://x.com/TheAhmadOsman/status/1930944597464654272)。
- **Suno 版权声明的复杂情况引发关注**：一名成员指出，除非保持有效的订阅，否则 **Suno** 存在限制，这与“无版权、无限制”的说法相矛盾。
   - 虽然执行起来可能具有挑战性，但这一澄清确保了用户了解 [Suno 的服务条款](https://www.reddit.com/r/LocalLLaMA/comments/1l4mgry/chinas_xiaohongshurednote_released_its_dotsllm/)。
- **Linear MCP 与 Claude Code 集成展示**：一位用户分享了 **Linear MCP** 的集成，使任务列表和项目状态在 **Claude Code** 会话之间保持有状态（stateful），在本地运行并处理 OAuth，将 Linear 暴露为 Cursor 可以理解的 *stdio* MCP 服务端，详见[此处](https://www.task-master.dev/)。
   - 该用户提到 *“我的整个 claude.md 文件现在基本上只是一个关于如何使用 linear mcp 的系统提示”*，从而实现了与 GitHub 的集成，并通过 Linear 任务分配触发 sub-agents。
- **苹果的研究质疑 AI 推理能力**：苹果的研究表明，像 **Claude**、**DeepSeek-R1** 和 **o3-mini** 这样领先的 AI “推理”模型并不具备真正的推理能力，而是擅长模式记忆，详见[此处](https://x.com/RubenHssd/status/1931389580105925115?s=19)。
   - 即使有明确的指令，模型在处理更高复杂度的问题时也经常失败，这挑战了围绕 AGI 即将到来的炒作。
- **AI 公司低估了 LLM 内容过滤器的缺陷**：该讨论涉及 AI 公司如何误解 LLM，特别是在内容过滤器方面，详见[此处](https://x.com/aiamblichus/status/1931487839822254358?s=46)。
   - 用户分享了一些幽默且出人意料的 LLM 响应示例，这些响应通过轻微的 Prompt 调整绕过了过滤器，表明 LLM 遵循的是“即兴表演（improv）”规则。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1380740345868058674)** (57 messages🔥🔥): 

> `MCP Server Publishing, Image Uploading with MCPs, Accessing GitHub MCP Server, MCP and SSE vs. WebSockets, MCP Resources and Prompts` 


- ****图片上传**在 MCP 中遇到困难**：成员们正努力让图片上传在 MCP 中正常工作，包括尝试从 **Cursor 的上下文**中提取并进行 **base64 编码**，但未获成功。
- **使用 Python 访问 GitHub MCP 服务端**：用户请求关于使用 **Python** 访问官方 **GitHub MCP 服务端**以读取文件和目录的指导，并被引导至[使用 Docker 的安装说明](https://github.com/github/github-mcp-server?tab=readme-ov-file#installation)。
- **解析 MCP 决策树**：一名成员寻求关于如何让 Agent 按顺序请求不同 Prompt 的资源，建议实现**决策树（decision tree）**来处理此类工作流。
- **客户端重连：HTTP 错误**：当**服务端重启**且客户端使用旧的会话 ID 连接时，客户端会卡在 **HTTP 400 或 404** 错误上，原因是客户端在错误后没有启动新会话。
   - MCP 规范规定客户端在遇到 404 错误时*必须*启动新会话，但在实践中，大多数客户端**并不符合该规范**。
- **MCP 不强制开源**：澄清了尽管 mcp.io 网站上有相关表述，但 MCP 服务端和客户端的实现**并不要求开源**。


  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1380932048335802408)** (11 条消息🔥): 

> `MCP 中的 OAuth 支持，Slack 官方 MCP 服务端，glama.ai 博客文章，MCP 规范实用工具，Google MCP 服务端` 


- ****OAuth 历险记**：MCP 的规范支持**：一位成员询问 MCP 是否在规范中原生支持 **OAuth**，创建者回应称最初构建时并未遵循 OAuth 标准，但欢迎贡献。
   - 提到了 Slack 的 [官方 MCP 服务端](https://www.slack.dev/secure-data-connectivity-for-the-modern-ai-era) 可能支持 OAuth，但似乎替代服务端尚未发布。
- ****规范侦探**：MCP 实用工具**：一位成员创建了一个实用工具，用于从 MCP 的“Specification”文档页面提取内容，与完整的 `llms.txt`/`llms-full.txt` 文件相比，文件大小减少了约三分之一。
   - 该工具的输出（仅包含规范网站页面）可在 [Gist](https://gist.github.com/hesreallyhim/d974990b8c80cf6f32b88bfe39b76f9a) 获取，而另一个 Gist 包含了 `schema.ts` 和 `schema.json` 文件。
- ****Google 卫士**：专注于安全性的 MCP 服务端**：一位成员分享了他们的 [Google MCP 服务端](https://github.com/robcerda/google-mcp-server)，强调其安全优先的设计，默认仅使用安全作用域（scopes）。
   - 他们强调该服务端可以从 MCP 本身管理大部分 **Gmail, Calendar, 和 Drive**，并正在寻求改进建议。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1380892809355329598)** (31 条消息🔥): 

> `tinygrad 中的 Lazy setitem，tinygrad 第 74 次会议，Lovely Grad 适配 tinygrad，tinygrad 上的 huggingface 模型，True float16` 


- **寻求 TinyGrad tensor 的 Lazy `__setitem__`**：一位贡献者质疑为什么 `tensor.py` 中的 `__setitem__` 会调用 `realize()`，导致其变为非延迟（non-lazy）执行，并建议将其拆分为 `setitem_and_realize`，以允许延迟、不可变且在设备上执行的操作，这可能使 [beautiful_cartpole](https://github.com/tinygrad/tinygrad/blob/master/examples/beautiful_cartpole.py) 等示例受益。
   - 用户指出，为了使建议的延迟实现生效，*必须移除当前的 realize()*。
- **TinyGrad 第 74 次会议回顾**：会议涵盖了公司更新，包括 multi 和 resnet dataloader 的修复、更快的 CI、linearizer、viz、drivers、cloud/hash、onnx 以及本地开发，还有 **lm_eval** 和 **AMD_LLVM** 等其他悬赏任务。
   - GeorgeHotz 表示他将在 *本周合并所有内容*。
- **`lovely-grad` 为现代 TinyGrad 重生**：[Lovely Grad](https://github.com/xl0/lovely-grad) 在中断数月后，现在已适配现代 tinygrad，并计划研究使用 pytest multiprocessing 进行远程测试。
   - 该工具旨在帮助可视化在 TinyGrad 中实现的神经网络的梯度流。
- **Hugging Face 模型将集成到 `test_onnx.py`**：Hugging Face 模型的测试将被重写，以集成到 `test_onnx.py` 中进行模型测试。
   - 此举旨在将模型测试整合到 `test_onnx.py` 框架中。
- **Metal 面临边界 Bug 问题**：据报道 Metal 存在编译器 Bug，一位用户在边界问题上浪费了半天时间，促使添加了 `max_total_threads_per_threadgroup` 以解决 CUDA 的 `__launch_bounds__` 和 HIP 的 `amdgpu_flat_work_group_size` 问题。
   - 由于驱动程序问题，用户震惊于 *这让 beautiful mnist 变成了 beautiful macos dos poc*。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1380932650612817942)** (21 条消息🔥): 

> `Tensor 索引，FUSE_ARANGE 效果，ProcessPoolExecutor 问题` 


- **Tensor 索引受到质疑**：一位成员质疑为什么 **tinygrad** 在 Tensor 抽象层没有直接的 tensor 索引操作。
   - 他们指出 `gather`、`scatter`、`getitem` 等都是通过复杂的掩码赋值实现的，而不涉及 numpy 的唯一选择是进入更底层的 UOps。
- **FUSE_ARANGE 救场**：通过使用 `FUSE_ARANGE=1` 上下文，一位成员展示了 tensor 索引操作 **10 倍的加速**。
   - 另一位成员好奇 `FUSE_ARANGE` 具体做了什么，以及是否可以在 `examples/hlb_cifar10.py` 中使用它。
- **子进程拒绝设备访问**：一位成员遇到了最近 **tinygrad** 更改（[PR#4425](https://github.com/tinygrad/tinygrad/pull/4425)）引起的问题，该更改拒绝从子进程打开设备，特别是在使用依赖于 `ProcessPoolExecutor` 的 `nbdev_test` 时。
   - 他们质疑 **tinygrad** 在从 `ProcessPoolExecutor` 使用时发生失败是否合理。


  

---

### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1381647788613566696)** (1 条消息): 

> `Office Hours, Form Filling Agents, LlamaIndex MCP Servers, MCP Dev Summit, Spreadsheet Agent` 


- ****Office Hours** 时间表已确定**：下一场 Office Hours 将于 **6 月 12 日星期四** **8AM PT/5PM CET** 举行，重点讨论 **MCP**、**form filling** 以及其他 LlamaIndex 主题。
   - Office Hours 将由两名成员在 general 语音频道主持。
- ****MCP Dev Summit** 演讲视频已发布**：在 **MCP Dev Summit** 上发表了一场关于 **13 种不同协议**的演讲，这些协议目前正在竞争成为 Agent 与工具及彼此之间通信的标准方式，包括 **MCP**、**A2A**、**ACP** 等，点击[此处](https://www.youtube.com/watch?v=kqB_xML1SfA)观看。
   - 视频现已可供观看。
- ****Spreadsheet Agent** 进入私测阶段**：**Spreadsheet Agent** 目前处于私测阶段，采用“先解析，后推理” (*Parse First, Reason Second*) 的架构，能够理解视觉结构和上下文，详见[这篇博客文章](https://www.llamaindex.ai/blog/introducing-the-spreadsheet-agent-in-private-preview)。
   - 该 Agent 现已准备好进行预览。
- ****Multi-Turn Conversation Memory** 实现**：新增了一个处理多轮对话的 Memory 实现，示例请见[此处](https://docs.llamaindex.ai/en/stable/examples/memory/custom_multi_turn_memory/)。
   - 此更新由一名成员实现。
- **支持 **Ollama 'thinking' 功能****：LlamaIndex 现在支持 Ollama 的 'thinking' 功能，查看[已合并的 PR](https://github.com/run-llama/llama_index/pull/18993)。
   - 这一增强功能支持更高级、更细致的交互。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1380622774673871040)** (7 条消息): 

> `Vector Databases, Llama Cloud, Agentic Extraction Workflow, MCP Servers, AI Summit` 


- **利用 Vector DBs 提升 RAG 流水线**：6 月 12 日，一位开源工程师将在慕尼黑的 BASED Meetup 上发表题为《Vector Databases 漫游指南》的演讲，讨论提升 RAG 流水线的最佳实践。
   - 演讲将涵盖从**数据准备**到**查询优化**的所有内容。
- **Llama Cloud 亮相**：一段展示 Llama Cloud 概览的新视频重点介绍了其生态系统以及构建生产级 LLM 应用的核心工具。
   - @tuanacelik 在[这段视频](https://t.co/kIPbq542Pr)中对整体景观进行了演示。
- **Agentic Extraction 工作流**：本周末将分享一篇教程，演示如何使用 LlamaIndex 针对 **Fidelity 多基金年度报告**构建 Agentic Extraction 工作流。
   - 该工作流用于提取[多个基金](https://t.co/RpmHYV4UDN)的列表，每个基金都报告了多张财务数据表。
- **Discord Office Hours**：LlamaIndex Discord 将与 @tuanacelik 和 @LoganMarkewich 共同举办另一场 Office Hours。
   - 本次会议将重点讨论使用 LlamaIndex 创建 **MCP servers** 以及构建 **form filling agents**，可通过[此链接](https://t.co/CLAauUeFty)访问。
- **Databricks AI Summit**：LlamaIndex 将参加 Databricks Data + AI Summit，提供交流和探索 AI 未来的机会。
   - 与会者可以在 AI 展馆的 D117 展位与 CEO Jerry Liu 和 AI 工程师见面，了解如何使用 [LlamaIndex](https://t.co/mbCkazR18g) 进行构建。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1380781155368046612)** (21 messages🔥): 

> `Gemini 2.5 streaming output, ACP protocol, Agent workflow termination, Plan upgrade issue, LlamaParse product down` 


- **Gemini 的“思考”过程需要分离**：一位成员询问在从 **Gemini 2.5** 等模型流式输出时，如何将“思考”文本与实际响应分离。
   - 另一位成员提到可能需要一个 **PR** 来支持此功能，因为目前的实现在流式传输时不会检查来自 Google 的 `Part` 类型，并指向了 [此 PR](https://github.com/run-llama/llama_index/pull/18993) 中完成的工作。
- **新协议 ACP 登场**：一位成员注意到生成式 AI 的一个新协议 (**ACP**)，并通过 [此 Pull Request](https://github.com/i-am-bee/acp/pull/176) 创建了一个 LlamaIndex 示例。
   - 他们请求对该文件进行评审。
- **通过提示词解决 Agent 工作流停滞问题**：一位成员报告称，尽管指定了 `can_handoff_to`，他们的 **Agent 工作流**在 transcription_agent 输出后仍会终止。
   - 另一位成员建议这可能并非“终止”，通过*调整提示词/描述*可以改变这种行为。
- **订阅混乱：升级到 Pro 失败**：一位成员报告了一个问题，即将其方案从 **Starter 升级到 Pro** 后，系统仍显示其处于 **Free 方案**。
   - 他们附上了截图作为证据。
- **LlamaParse 宕机！**：一位成员询问依赖 **LlamaParse** 的产品是否宕机，并寻求此类事件的联系方式，提到*有人通过支持邮箱进行了回复*。
   - 他们随后确认其产品确实已宕机。


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1380923564328222875)** (1 messages): 

> `RAG, Llama Index, ReactAgent, Sparse Data` 


- **成员讨论稀疏 RAG 检索挑战**：一位成员报告称，使用 **Llama Index** 配合 **ReactAgent** 和一些检索器工具搭建了一个不错的 **RAG 设置**，但在稀疏数据检索方面遇到了问题。
   - 他们声称有超过 **1000 份文档**涉及某些原则，并想知道是否有更好的方法来确保不遗漏重要信息，而不必采用极高的 **K 检索值**。
- **启动针对 RAG 系统中稀疏数据处理的头脑风暴**：在提出初始问题后，讨论预见到了在 **Retrieval-Augmented Generation (RAG)** 设置中更有效地处理稀疏数据的潜在解决方案。
   - 对话倾向于寻找避免高 **K 检索值**的方法，表明了对信息检索精度和效率的关注。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1380639317856354466)** (21 messages🔥): 

> `Issue #2470, Clipping logprobs, Adagrad errors on nightly` 


- **Issue #2470 需要关注**：一位成员强调了自三月份以来一直等待处理的 [Issue #2470](https://github.com/pytorch/torchtune/issues/2470)。
   - 另一位成员询问之前是否讨论过此问题，引发了关于其优先级的辩论。
- **关于剪裁 logprobs 功能的辩论**：一位成员提议增加剪裁 logprobs 的功能，这引发了关于其在 TorchTune 中必要性和实现方式的讨论。
   - 虽然该功能在其他仓库中存在，但人们对维护成本以及将其正确暴露给用户的潜在复杂性表示担忧。
- **Adagrad 在 Nightly 版本上因 DeviceMesh 断言失败**：一位用户报告称，在特定配置的 Nightly 构建版本上使用 **fused Adagrad** 时，出现 `AssertionError: found no DeviceMesh from dtensor args for aten._fused_adagrad_.default!` 错误。
   - 在切换到最新的 TorchTune 后，**SGD** 似乎可以工作，但 Adagrad 问题的根本原因仍不清楚，且复现尝试均未成功。


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1380673342092345387)** (10 messages🔥): 

> `Cohere AI, Cohere support, command-a, documentation` 


- **申请 Cohere AI！**：为了获得 **Cohere AI** 的访问权限，一位用户建议通过 [此表单](https://share.hsforms.com/10OrjljwpQ52ILJA6ftENIwch5vw) 进行申请。
   - 他们很乐意提供帮助。
- **Cohere 支持通过 command-a 机器人得到提升**：一位用户宣布频道 <#1381756280716132412> 通过使用 **command-a** 提供更快速的支持，该机器人利用 Cohere 网站的文档来回答问题。
   - 该机器人无法处理账户问题、API 问题等，且在 Beta 阶段仅在用户在线时激活；滥用将导致立即封禁。


  

---

### **Cohere ▷ #[📣-announcements](https://discord.com/channels/954421988141711382/996880279224451154/1381776718456426517)** (1 条消息): 

> `North Integration, GameWarden, Partnerships, Security Deployments` 


- **North 与 GameWarden 联手保障安全**：**North** 现在通过与 **Second Front** 的合作伙伴关系集成了 **GameWarden** 平台，用于高安全环境下的安全部署，详见[此 X 帖子](https://x.com/1vnzh/status/1930298055099613307)。
- **GameWarden 平台获得增强安全性**：现役军人在应对不断变化的威胁格局时获得了前所未有的效率和速度，目前已安全集成到完整的 **GameWarden** 平台中。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1380756761018892378)** (5 条消息): 

> `Cohere's r7b model, Cohere signup issues, Cohere open source contributions, Cohere Developer Experience GitHub repository` 


- **r7b 输出速度达 1 T/s！**：一位用户表示 **Cohere 的 r7b 模型**输出速度约为 **1 T/s**。
   - 未提供关于该性能指标具体细节或背景的进一步信息。
- **Google Cloud Marketplace 注册故障**：一位用户在尝试通过 **Google Cloud Marketplace** 注册 **Cohere** 时遇到错误。
   - 错误信息为：*{"code":"invalid_vendor","description":"The vendor '8SAJ2US' does not exist"}*，该供应商问题可能源自[此 URL](https://upstream-api.tackle.io/v1/gcp/order/8SAJ2US/cohere.endpoints.cohere-id-public.cloud.goog)。
- **Cohere 支持团队请求邮件联系**：一位成员建议，任何在 **Google Cloud Marketplace** 遇到错误问题的用户应将问题详情发送至 [support@cohere.com](mailto:support@cohere.com)。
   - 他们提到在邮件中应包含**错误信息**和**目前已采取的步骤**。
- **Cohere 接受 GitHub 贡献！**：**Cohere** 在 **[Cohere Developer Experience GitHub 仓库](https://github.com/cohere-ai/cohere-developer-experience/)**中有一个接受贡献的开源仓库。
   - 虽然欢迎贡献，但 **OpenAPI** 规范和代码片段是从内部仓库同步的，团队将负责处理这些更改。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1380673786013290606)** (3 条消息): 

> `Introductions, Discord App Development, AI and Machine Learning` 


- **欢迎新社区成员**：一位名叫 David 的新成员介绍了自己，分享了他的年龄、代词和兴趣，并提到他即将满 **19** 岁。
   - David 传达了一个积极的信息：*你是被认可的，你是被爱着的*。
- **新成员开发 Discord 应用**：David 目前正在开发 **link safe**，他将其描述为一个 *Discord 应用*。
   - 他还分享了对 **AI** 和 **Machine Learning** 的热情，称其为一个*神奇的世界*。
- **寻求社区支持**：David 希望获得来自 Cohere 社区的支持。
   - 他*不确定自己是否已经发布过这个内容*。


  

---


### **Cohere ▷ #[🔔-ping-settings](https://discord.com/channels/954421988141711382/1346642216541622343/)** (1 条消息): 

competent: 已移至 <id:customize>
  

---


### **Nomic.ai (GPT4All) ▷ #[announcements](https://discord.com/channels/1076964370942267462/1090471714888102009/1380978691127120054)** (1 条消息): 

> `Nomic team updates` 


- **Nomic 团队准备激动人心的更新**：**Nomic 团队**在过去几个月里一直埋头苦干，致力于一些令人兴奋的更新。
   - 虽然具体细节尚未公开，但团队请求大家保持耐心，因为他们正在为未来的发布做准备。
- **请求对未来发布保持耐心**：Nomic 团队了解社区对新进展的期待。
   - 他们正在为发布而努力工作，但请求在一切准备就绪前继续保持耐心。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1380685905706356826)** (15 条消息🔥): 

> `以纯文本保存聊天记录, Nomic 团队更新, GIGABYTE 服务器, nomic-embed-text-v1.5` 


- **用户请求保存聊天功能**：一名用户请求为 **GPT4All** 增加一个在特定目录中*以纯文本形式保存聊天记录*的功能，并建议这将增强用于记忆的 LocalDocs RAG Search。
- **Nomic 团队正在酝酿更新**：一名成员提到 **Nomic 团队**一直在开发*令人兴奋的更新*，但目前尚未准备好发布。
- **GIGABYTE 服务器可能以准系统形式提供**：一名用户询问在等待 **GPT4ALL** 升级期间，[GIGABYTE 服务器](https://www.gigabyte.com/Press/News/2293)是否会以准系统（barebone）形式提供。
   - 他们推测该服务器能以创纪录的速度运行 **Mixtral 8x22B**，这符合 **MOE models** 的发展趋势。
- **关于 nomic-embed-text-v1.5 使用的疑问**：一名用户询问下个月是否仍能从 nomic cloud 使用 **nomic-embed-text-v1.5**，并附上了一张[图片](https://cdn.discordapp.com/attachments/1090427154141020190/1381780899716399145/image.png?ex=6848c33e&is=684771be&hm=7713e72607a3b6445cf9a1cfd28fc026127c79b6bf40f539e8edd0edb0b80bf8)。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1380655104138678393)** (4 条消息): 

> `迁移训练后学习, 区块链专业知识, AI Agent 开发者` 


- **新手提问：训练后学习可以迁移吗？**：一位新成员询问是否可以在不重复学习过程（如 finetuning 或 RL）的情况下，将一个模型的训练后学习（post-training learning）迁移到另一个模型。
   - 该问题引发了关于 AI 模型背景下迁移学习不同方法的讨论。
- **AI/区块链开发者展示专业背景**：一位在 **Blockchain (EVM, Solana, Cardano, Hydra, Aptos, Cosmos, Tron, zk-SNARKs)** 和 **AI (LLM, NLP, LangChain, AutoGen, TorchRL, DL, Azure ML, AI Agent)** 领域拥有经验的软件工程师自荐寻求工作机会。
   - 该工程师还拥有 **Web systems (React, Next, Vue, Node, IPFS, Pinata API)** 的经验，并提供了其空档期信息和联系方式。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1381729569580646501)** (1 条消息): 

> `Agentic AI 峰会, 早鸟票, UC Berkeley, Vinod Khosla, Ion Stoica` 


- **Agentic AI 峰会即将在 UC Berkeley 举行**：继广受欢迎的 **LLM Agents MOOC** 之后，Agentic AI 峰会将于 **2025 年 8 月 2 日**在 **UC Berkeley** 举行，预计将有 **1,500+** 名线下参会者。
   - 演讲嘉宾包括 **Vinod Khosla** (Khosla Ventures)、**Ion Stoica** (Databricks 和 Anyscale)、**Dawn Song** (UC Berkeley) 等，计划开展主题演讲、小组讨论和工作坊。
- **早鸟票即将售罄！**：Agentic AI 峰会的早鸟票将于 **2025 年 6 月 30 日**截止，学生票价为 **$25**，初创公司票价为 **$60**，行业专业人士票价为 **$80**。
   - 根据 [峰会官网](https://rdi.berkeley.edu/events/agentic-ai-summit)，学生和独立开发者可以申请费用减免。


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/)** (1 条消息): 

sebash6677: 你好
  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1381619522146144266)** (1 条消息): 

> `排行榜更新, 项目延续` 


- **排行榜更新暂停，项目前景不明**：一名用户询问为何排行榜停止了更新。
   - 他们还询问该项目的工作是否会继续。
- **项目未来仍处于悬而未决的状态**：该用户在查询中专门提到了 <@856060858462502922>。
   - 在给定的上下文中未提供任何回复或澄清。