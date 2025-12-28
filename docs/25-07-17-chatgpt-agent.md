---
companies:
- openai
date: '2025-07-17T05:44:39.731046Z'
description: '**OpenAI** 发布了 **ChatGPT Agent**，这是一个新型的高级人工智能系统，具备网页浏览、代码编写、数据分析及报告生成的能力，标志着向类人化计算机操作迈出了重要一步。


  该智能体与 **o3** 不同且性能更优，被视为内部代号为 **o4**（现已并入 **GPTNext**）的首次公开亮相。它采用了端到端强化学习技术，能够长时间持续运行（测试时长达
  2 小时），并在生物滥用风险方面被评定为“高”风险，目前已启动相关安全防护。


  早期基准测试结果显示其表现参差不齐：在 **WebArena** 和 **BrowserComp** 等测试中表现优异，但在 **PaperBench** 等测试中表现欠佳。参与该项目的关键人物包括
  **Sam Altman**、**Greg Brockman** 和 **Kevin Weil**，此外还包含了来自 **xikun_zhang_** 的技术见解以及
  **KerenGu** 和 **boazbaraktcs** 的风险评论。此次发布一度引发了关于 **GPT-5** 的猜测，但官方已确认该系统并非 GPT-5。'
id: MjAyNS0w
models:
- o3
- o4
- gptnext
people:
- sama
- gdb
- kevinweil
- xikun_zhang_
- keren_gu
- boazbaraktcs
title: ChatGPT 智能体：全新 o* 模型 + 统一的深度研究浏览器 + Operator 计算机操作功能 + 代码解释器终端
topics:
- reinforcement-learning
- benchmarking
- model-performance
- model-risk
- long-context
- model-deployment
- fine-tuning
---

**ChatGPT 就够了。**

> 2025年7月16日至7月17日的 AI 新闻。我们为您检查了 9 个 subreddit、449 个 Twitter 和 29 个 Discord（226 个频道和 9565 条消息）。预计节省阅读时间（按每分钟 200 字计算）：703 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

在一场广受好评、充满经典 OpenAI 风格的[太平洋时间上午 10 点直播](https://www.youtube.com/watch?v=1jn_RpbPbEc)中，Sama 及其团队发布了 "ChatGPT Agent"，并以一个极具梗潜力的开场白拉开序幕（虽然还不是今天的头号热梗）：


![](https://resend-attachments.s3.amazonaws.com/qhdSMz9QRA1e5Uv)


[博客文章](https://openai.com/index/introducing-chatgpt-agent/)、[System Card](https://cdn.openai.com/pdf/6bcccca6-3b64-43cb-a66e-4647073142d7/chatgpt_agent_system_card_launch.pdf)、[System Prompt](https://gist.github.com/Rutledge/4b0ef2d51ba2f1918a249bce35bdde9c)，以及 [Wired](https://www.wired.com/story/openai-chatgpt-agent-launch/) 和 [Every](https://every.to/vibe-check/vibe-check-openai-enters-the-browser-wars-with-chatgpt-agent) 的报道，都集中在制作 [Slides](https://www.youtube.com/watch?v=szJI9YJNEZk)、[Spreadsheets](https://www.youtube.com/watch?v=JAQ4p662It8)、[Research](https://www.youtube.com/watch?v=Wgn4JeYI9lY)、[可定制性](https://www.youtube.com/watch?v=EKMHiOQPwpc)（包括 [Scheduled Agents](https://x.com/neelajj/status/1945945913014546805?s=46)）等方面。当然，HLE 和 FrontierMath 基准测试的表现也非常出色，但是：

1. 我们不应让基准测试疲劳掩盖这样一个事实：模型和 Agent 在这些极具挑战性、甚至已经达到超人类水平的测试中进步神速；
2. 大多数人忽略了，如果你仔细观察标签，博客文章中提到的“该模型”是一个独立于 o3 且优于 o3 的全新模型：


![](https://resend-attachments.s3.amazonaws.com/9dlIn7cXzby8sxJ)


就像 Deep Research 是首个在任何地方公开展示完整 o3 的产品一样，ChatGPT Agent 似乎是首个公开展示原本可能被称为 o4、但现在正被合并到 GPTNext 中的模型的产品。

---

# AI Twitter 回顾

**OpenAI ChatGPT Agent 发布**

- **OpenAI 发布了 ChatGPT Agent**，这是一个全新的统一系统，结合了深度研究能力与操作计算机的能力。该 Agent 可以[浏览网页、使用终端、编写代码、分析数据，并创建报告、电子表格和幻灯片](https://twitter.com/OpenAI/status/1945890050077782149)。**OpenAI** 宣布了这一发布，包括 **Sam Altman** 在内的关键人物也发布了相关信息，他指出[这对他来说是一个真正的“感受到 AGI”的时刻](https://twitter.com/sama/status/1945917559796298083)，**Greg Brockman** 分享道，这是朝着他们[创建一个能像人类一样使用计算机的 Agent 的 10 年目标](https://twitter.com/gdb/status/1945923067403984979)迈出的一大步，而 **Kevin Weil** 则描述了其向 [Pro、Plus 和 Teams 用户](https://twitter.com/kevinweil/status/1945896640780390631)推出的情况。
- **来自开发团队的技术见解**由 [@xikun_zhang_](https://twitter.com/xikun_zhang_/status/1945895070269583554) 分享，强调了**端到端 Reinforcement Learning (RL)** 的力量、用户协作的重要性，以及对现实世界性能而非追求 Benchmark 的关注。团队还透露，该 Agent 可以长时间执行任务，其中一项内部测试运行了 **2 小时**。
- **ChatGPT Agent 是 OpenAI 首个被归类为生物滥用风险“高”能力级别的模型**，研究人员 [@KerenGu](https://twitter.com/KerenGu/status/1945944156935004415) 和 [@boazbaraktcs](https://twitter.com/boazbaraktcs/status/1945944398199677016) 强调了这一点。他们表示，已经激活了最强大的安全措施来减轻这些风险。然而，Benchmark 显示，如果被要求，该 Agent 有 **10% 的概率**执行“有害操作”（如拿用户的存款去赌博），[且比 o3 更有可能尝试制造超级病毒](https://twitter.com/scaling01/status/1945930617775882728)。
- **该 Agent 的早期 Benchmark 结果**由 [@scaling01](https://twitter.com/scaling01/status/1945895473430089947) 分享，显示其在 **HLE 上得分约为 42%**，在 **FrontierMath 上约为 27%**，在 **WebArena 上约为 65%**，在 **BrowserComp 上为 69%**，以及在 [**SpreadsheetBench 上为 45%**](https://twitter.com/scaling01/status/1945896464632148366)。此外还注意到，该 Agent 在 **PaperBench** 和 **SWE-Bench** 等 Benchmark 上的表现[低于 **o3**](https://twitter.com/scaling01/status/1945932154455695752)。
- **这一公告引发了广泛的猜测和评论**，许多用户对发布的不是 **GPT-5** 表示失望。[@scaling01](https://twitter.com/scaling01/status/1945640155890483359) 反复从可靠来源确认这并非 **GPT-5**，导致了一种[“AI 精神病”和等待日期的状态](https://twitter.com/scaling01/status/1945913979517247769)。[@swyx](https://twitter.com/swyx/status/1945904109766459522) 将其与最初的 **iPhone** 发布进行了类比，将该 Agent 描述为三合一：浏览器、计算机和终端。

**Model Releases, Performance & Benchmarks**

- **Moonshot AI 的 Kimi K2 已成为 LMSys Chatbot Arena 上排名第一的开源模型**，正如 [Arena 所宣布的那样](https://twitter.com/lmarena_ai/status/1945897926796185841)，并得到了 [@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/1945897926796185841) 团队的庆祝。该模型因其高性能和速度而受到赞誉，特别是在 **Groq** 的硬件上，据 [@OpenRouterAI](https://twitter.com/OpenRouterAI/status/1945779694256722025) 报告并由 [@cline](https://twitter.com/cline/status/1945627344997130473) 演示，其速度超过了 **200 tokens/second**。据指出，它在编程基准测试中击败了 **Claude Opus 4**，且价格便宜高达 **90%**。
- **xAI 的 Grok 4 已对安全问题进行了调查和缓解**，根据 [@xai 的官方公告](https://twitter.com/random_walker/status/1945614419213316571)。然而，该发布面临批评，[@boazbaraktcs](https://twitter.com/SebastienBubeck/status/1945669260027777049) 对其安全性表示担忧。该模型新的“companions”功能也受到了 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1945737831697064446) 的批评，称其为低质量的“waifu engineering”，并指出角色模型存在穿模（clipping）和拼写错误。
- **Google DeepMind 宣布 Veo 3**，其最新的视频生成模型，现已通过 **Gemini API** 和 **AI Studio** 提供公开预览，如其[官方账号所述](https://twitter.com/GoogleDeepMind/status/1945886603328778556)。[@_philschmid](https://twitter.com/_philschmid/status/1945898590821584989) 分享了一个使用复杂提示词生成视频的详细代码示例。此外，**Gemini 2.5 Pro** 正在集成到 **Google Search** 的 **AI Mode** 中，并在 **IMO 2025** 数学基准测试中获得了 **31.55%** 的分数，[优于 **Grok 4** (11.90%) 和 **o3 high** (16.67%)](https://twitter.com/denny_zhou/status/1945887753864114438)。
- **实时视频扩散（Real-time video diffusion）现在可以通过 MirageLSD 实现**，这是来自 **Decart AI** 的新模型。[@karpathy](https://twitter.com/karpathy/status/1945979830740435186) 对其潜力进行了全面概述，从在视频流中创建替代现实、实时电影导演，到通过文本提示为游戏环境设置风格。
- **H-Net，一种新型分层网络（hierarchical network）**，被引入以通过消除 tokenization 步骤来创建真正的端到端语言模型，正如 [@sukjun_hwang 所分享的](https://twitter.com/abacaj/status/1945898630289727854)。这种方法允许模型直接处理原始字节（raw bytes）。
- **Together AI 宣布 DeepSeek R1 在 NVIDIA B200s 上实现了创纪录的推理速度**，达到高达 **330 tokens/sec**，正如 [@vipulved 所强调的](https://twitter.com/vipulved/status/1945934641451675793)。
- **Muon optimizer 在训练 Kimi K2 中发挥了关键作用**，[@kellerjordan0](https://twitter.com/kellerjordan0/status/1945701578645938194) 指出了这一事实。该优化器的首次应用是在 **3e14 FLOP** 的训练运行中打破了 **CIFAR-10** 竞速的 3 秒大关，而 K2 的训练规模比这大 **10 个数量级**，达到 **3e24 FLOPs**。
- **ColQwen-Omni**，一个扩展了 **ColPali** 概念的 **3B** 全模态检索器（omnimodal retriever），由 [@ManuelFaysse](https://twitter.com/andersonbcdefg/status/1945855681976021268) 介绍。

**AI 工具、框架与基础设施**

- **关于原生推理（reasoning-native）与原生记忆（memory-native）模型的辩论**由 [@jxmnop](https://twitter.com/jxmnop/status/1945857324285149256) 发起，他认为主要的 AI 实验室过度关注推理，而应该构建**原生记忆语言模型**，并指出由于目前没有流行的 LLM 拥有内置记忆模块，这一领域的大门正“敞开着”。
- **Claude 的桌面集成正在将其演变为一个“LLM OS”**，据 [@swyx](https://twitter.com/swyx/status/1945734758102868243) 称，他赞扬了其与 **Chrome, iMessage, Apple Notes, Linear, Gmail 和 GCal** 集成的实用性。为了实现并行执行，[@charliebholtz](https://twitter.com/HamelHusain/status/1945871155869178539) 推出了 **Conductor**，这是一个用于同时运行多个 **Claude Code** Agent 的 Mac 应用。
- **来自 Reflection AI 的代码研究 Agent Asimov** 已经发布，旨在解决工程师 **70%** 的时间花在理解代码而非编写代码的问题。该发布由 [@MishaLaskin 宣布](https://twitter.com/hardmaru/status/1945628506035294697)。
- **NanoGPT 训练速度新纪录**由 **Vishal Agrawal** 创造，在 **8xH100 GPU** 上仅用 **2.966 分钟**就达到了 **3.28 FineWeb** 的验证损失。正如 [@kellerjordan0](https://twitter.com/kellerjordan0/status/1945920703158710316) 所报道，这一提速是通过将梯度 `all_reduce` 替换为 `reduce_scatter` 以及其他效率优化实现的。
- **LlamaIndex 团队发布了《检索生产化指南》（The Hitchhiker’s Guide to Productionizing Retrieval）**，这是一份构建生产级 RAG 系统的详细指南。正如 [@jerryjliu0](https://twitter.com/jerryjliu0/status/1945647281782636974) 所总结，该指南涵盖了文本提取、分块（chunking）、Embeddings、通过语义缓存提升搜索效果以及查询重写，并提供了使用 **Qdrant** 的实际案例。
- **Perplexity 正在为其 Comet 浏览器发送新一批邀请**，正如 [CEO @AravSrinivas 所宣布](https://twitter.com/AravSrinivas/status/1945669970618421699)。[@rowancheung](https://twitter.com/AravSrinivas/status/1945620938068037633) 指出，经过一周的测试，该 Agent 已经开始“真正产生粘性”。
- **Atropos v0.3**（来自 **NousResearch** 的强化学习环境框架）已经发布。[@Teknium1](https://twitter.com/Teknium1/status/1945927019281478051) 强调了一个关键更新：一个新的仅评估模式，以及移植了 **@natolambert 的 Reward-Bench**，用于评估 LLM-as-a-Judge 的能力。
- **Notion 正在使用 Turbopuffer 构建最先进的 AI 应用**，这是 [@turbopuffer 分享](https://twitter.com/turbopuffer/status/1945865085530026359)的一个案例研究。

**AI 研究、论文与新技术**

- **一篇关于“AI for Science”的批判性文章**由 [@random_walker](https://twitter.com/random_walker/status/1945820621142688068) 和 [@sayashk](https://twitter.com/sayashk) 撰写，文章认为 AI 可能会加剧**生产-进步悖论（production-progress paradox）**，即科学论文产出呈指数级增长，而实际进展却停滞不前。他们认为 AI 公司目标错位，专注于“AI 发现了 X！”等噱头标题，而不是解决真正的瓶颈。作者认为[目前的 AI-for-science 评估是不完整的](https://twitter.com/random_walker/status/1945849588805447743)，因为它忽略了对研究人员理解力和社区动态的影响。
- **[@jxmnop](https://twitter.com/jxmnop/status/1945905080781451396) 的新博客文章《所有 AI 模型可能都是一样的》（All AI Models Might Be The Same）**解释了**柏拉图表征假设（Platonic Representation Hypothesis）**，暗示 AI 模型中存在普遍的语义。这可能对理解鲸鱼语言或破译古代文献等任务产生影响。
- **SIGIR2025 最佳论文授予了用于快速延迟交互（late interaction）的 WARP 引擎**，这一荣誉由 [@lateinteraction](https://twitter.com/lateinteraction/status/1945924144412930338) 重点介绍。
- **一篇关于递归混合（Mixture of Recursions, MoR）的新论文**提出了一种构建具有更高准确度和更大吞吐量的小型模型的方法。该论文由 [@QuixiAI 分享](https://twitter.com/QuixiAI/status/1945907010584637891)，涵盖了从 **135M 到 1.7B** 参数的模型。
- **OpenMed** 是一个包含 380 多个最先进医疗 AI 模型的集合，已由 [@MaziyarPanahi](https://twitter.com/ClementDelangue/status/1945622980475691364) 在 **Hugging Face** 上发布，旨在推动 AI 在医学领域的发展。
- **来自 Alibaba-NLP 的关于 WebSailor 的论文展示了用于深度研究（Deep Research）的后训练模型**，[@AymericRoucher](https://twitter.com/AymericRoucher/status/1945870603275403693) 指出，在后训练末期加入 Agent 强化学习循环将分数提高了 **约 4 个百分点**。

**公司、生态系统与地缘政治**

- **Perplexity AI 宣布与 Airtel India 建立合作伙伴关系**，这是 [CEO @AravSrinivas 分享](https://twitter.com/AravSrinivas/status/1945736795280613580)的一个重要里程碑。在宣布这一消息后，Perplexity 成为 [印度 App Store 总榜排名第一的应用](https://twitter.com/AravSrinivas/status/1945960772091433081)，超过了 ChatGPT。
- **AI Agent 初创公司 Lovable 以 18 亿美元的估值筹集了 2 亿美元**，由 Accel 领投，正如 [联合创始人 @antonosika 所宣布的](https://twitter.com/karansdalal/status/1945979009399132533)。
- **在 AtCoder World Tour Finals 2025 启发式竞赛中，人类选手 @FakePsyho 获得第一名**，击败了获得第二名的 **OpenAI** Agent。[@hardmaru](https://twitter.com/hardmaru/status/1945850637528490134) 庆祝了人类的胜利，而 [@andresnds](https://twitter.com/mckbrando/status/1945692340292854112) 则详细介绍了 OpenAI 参与这场 10 小时现场演示赛的情况。
- **美国签证问题正阻碍顶级 AI 会议在该国举行**，[@natolambert](https://twitter.com/ClementDelangue/status/1945824425506398677) 将此情况描述为“重大政策失败”。这导致了在哥本哈根独立组织的 **EurIPS**，[**NeurIPS** 官方对此表示支持](https://twitter.com/algo_diver/status/1945749595252039832)。
- **中美技术动态仍然是一个突出话题**。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1945624983985639487) 质疑为什么没有“美国版的 Kimi”，并将其归因于激励机制的不匹配，随后他认为 [美国的出口管制低估了中国在关键技术树上的领先地位](https://twitter.com/teortaxesTex/status/1945733220336591109)。
- **LLM evals 领域的早期平台 Humanloop 将于 9 月关闭**。[@imjaredz](https://twitter.com/imjaredz/status/1945885618598474200) 宣布他的公司 **PromptLayer** 正在为 **Humanloop** 用户提供迁移方案。

**幽默/梗图**

- **对 OpenAI 发布新产品的期待** 被 [@nearcyan](https://twitter.com/nearcyan/status/1945623927092646286) 的一条疯传推文捕捉到了，他描述了与一位 **OpenAI** 朋友共进晚餐的情景，这位朋友“一直含糊地指着厨房咧嘴笑，好像我们的食物就要出来了一样，但我们还没点餐呢”。
- **FFmpeg 项目宣布通过手写汇编实现了 100 倍的加速**，一位开发者指出 `rangedetect8_avx512` 函数的性能提升了 **100.18 倍**，[由 @FFmpeg 分享](https://twitter.com/LearnOpenCV/status/1945975913889329603)。
- **关于在 AI 领域工作的现实情况**，[@typedfemale](https://twitter.com/typedfemale/status/1945912359027114310) 发布了一张拥挤机房的照片，配文是“为您呈现：大杰夫的 Trainium 地狱”。
- **一个关于数据污染的梗图** 被广泛分享，图中一个卡通人物在考试期间向另一个人物低声耳语答案，[@vikhyatk](https://twitter.com/vikhyatk/status/1945969703266275548) 为其配文：“我们要查看基准测试，并找到尽可能接近它的样本，但它们不是完全匹配，所以这不算是在测试集上训练”。
- **一个关于模型开发的笑话** 来自 [@vikhyatk](https://twitter.com/vikhyatk/status/1945970434253664546)，引起了许多人的共鸣：“我知道我的模型没有偏差（biased），因为我在所有的线性层中都设置了 bias=False”。
- **对科技文化的讽刺评论** 包括来自 [@cto_junior](https://twitter.com/cto_junior/status/1945717278953386302) 的一条推文，展示了一个穿着华丽的人，配文是：“当我出现在 CEO 宣布我们没钱了的全员大会上时”。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Kimi K2 模型排行榜排名与 OpenAI 对比

- [**Aider Polyglot Coding Leaderboard 上的 Kimi K2**](https://i.redd.it/wvr0xh2jecdf1.jpeg) ([得分: 178, 评论: 42](https://www.reddit.com/r/LocalLLaMA/comments/1m1vf6g/kimi_k2_on_aider_polyglot_coding_leaderboard/)): **该图片展示了 "Aider Polyglot Coding Leaderboard"，该榜单在正确性、成本和编辑格式方面对编码 LLM 进行了基准测试。Kimi K2 模型表现突出，在编码任务中实现了** `56.0%` **的成功率，成本为** `$0.22`**，且 diff 格式编辑正确率为** `92.9%`**。该模型通过** `aider --model openrouter/moonshotai/kimi-k2` **调用，展示了其在对比模型中最高的性价比。** 评论者对 $0.22 的低成本印象深刻，并讨论了将 K2 作为 coder、r1 0528 作为 architect 结合使用的潜在优势，建议关注进一步的成本降低和角色专业化。
    - 关于 Aider Polyglot Coding Leaderboard 上报告的 Kimi K2 成本效率存在争议，一些用户质疑基准测试结果是否准确。一位用户指出，Kimi K2 报告的单位输出成本似乎低于其列出的 API 价格（[每 1M tokens $2.20-$4](https://openrouter.ai/moonshotai/kimi-k2)），特别是与理论上应该更便宜的 Deepseek V3 相比。怀疑点在于基准测试可能低估了 Kimi K2 的 token 使用量，可能是因为其生成的回复比同类模型更简洁，或者在报告使用的 token 时存在计算错误。
    - 混合架构引起了技术兴趣，特别是建议在工作流中使用另一个模型 (r1 0528) 作为 "architect"，而将 K2 作为 "coder"，并预期这种组合将保持成本效益。
    - 对 Deepseek V3（[每 1M tokens $1.10](https://api-docs.deepseek.com/quick_start/pricing)）、Kimi K2 和 Sonnet-4（[Anthropic 定价](https://www.anthropic.com/pricing#api)）进行了详细的价格比较，强调了简洁（“非思考”）输出对整体成本的重要性。有人担心基准测试结果与公布的 API 费率不符，暗示基准测试可能“偏差了 10 倍”。
- [**提醒一下，今天 OpenAI 本打算发布一个 SOTA 开源模型……直到 Kimi 出现。**](https://www.reddit.com/r/LocalLLaMA/comments/1m2gp16/just_a_reminder_that_today_openai_was_going_to/) ([得分: 386, 评论: 55](https://www.reddit.com/r/LocalLLaMA/comments/1m2gp16/just_a_reminder_that_today_openai_was_going_to/)): **该帖子提到了 OpenAI 此前传闻计划发布一个最先进的开源语言模型，但声称在 Kimi（Moonshot AI 的 Kimi Chat）发布后，该计划被重新考虑或被掩盖了，Kimi 最近因其先进的能力而受到关注。文中将此与之前的竞争紧张局势进行了对比，特别是 Llama 4 和 Deepseek，标志着 SOTA 开源和闭源 LLM 厂商之间的快速迭代和竞相超越。** 热门评论强调了一种新兴模式，即预期的 OpenAI 发布被竞争对手模型（如 Deepseek）抢占先机或掩盖光芒，这表明竞争激烈的“竞赛”可能会反复推迟或阻止 OpenAI 的开源发布。
    - 几位评论者讨论了 OpenAI 在 Kimi 或 Deepseek R2 等强大竞争对手发布后不久发布新开源模型所面临的挑战。共识是，在更强大、更新的模型发布后紧接着发布一个较弱的模型会带来巨大的声誉风险，并可能损害该模型的采用率和在 SOTA 基准测试中的领先地位。
    - 技术上提到了 Meta 的 Llama 4 的实际相关性和采用情况，质疑社区是否真的在使用它。相比之下，Google 的 Gemma 3 被引用为用户正在转向的高质量替代方案，这表明人们对 SOTA 开源模型的看法正在发生变化。
    - 讨论强调了一个模式：如果公司的模型无法与最新的 SOTA 领导者（例如 Kimi, Deepseek R2）竞争，它们就会犹豫是否发布，这表明相对于竞争对手公开基准测试的时机和性能是发布策略和社区采用的关键因素。

### 2. Mistral Le Chat 功能发布与改进

- [**Mistral 为 Le Chat 发布 Deep Research、语音模式、多语言推理和 Projects 功能**](https://mistral.ai/news/le-chat-dives-deep) ([Score: 467, Comments: 34](https://www.reddit.com/r/LocalLLaMA/comments/1m2bigh/mistral_announces_deep_research_voice_mode/))：**Mistral AI 的 Le Chat 引入了多项技术升级：(1) Deep Research 模式采用工具增强型 Agent，通过规划、引用溯源和综合，针对复杂话题生成结构化、有参考依据的报告（参见公告 [此处](https://mistral.ai/news/le-chat-dives-deep)）；(2) 语音模式由 Voxtral 驱动，这是一款专有的低延迟语音模型，针对 ASR/转录进行了优化；(3) Magistral 模型支持上下文丰富、原生多语言和代码切换（code-switching）推理；(4) 新的 Project 文件夹允许按上下文范围组织对话线程；(5) 通过 Black Forest Labs 提供高级图像编辑功能。Deep Research 流水线特别展示了多源、高引用率的分析能力——超越了表格输出，整合了现实世界的申报文件和财务数据。** 评论强调，与 Whisper Large 相比，Voxtral Mini ASR 提供了更优的转录性能和更低的成本，并强调了宽松许可对支持 LLM 生态系统的价值。Deep Research 的 UI 被认为是一个技术设计亮点。
    - 一位用户报告称，Mistral 的 "voxtral mini" 转录模型不仅在质量上，而且在成本上都优于 OpenAI 的 Whisper Large 模型，这表明与之前的 state-of-the-art 模型相比，语音转文字任务在速度和/或准确性方面有了显著提升。
    - 讨论中包括一个疑问：目前是否有任何本地语言模型提供与 ChatGPT 和 Gemini 中的 Deep Research 功能相媲美的研究辅助功能，这表明用户对可自托管的、具有类似高级推理和综合能力的替代方案感兴趣。
    - 对 Mistral Le Chat 的观察强调了其速度和扎实的可用性，尽管在 Benchmark 性能上被认为落后于“领先者”（如 OpenAI、Google）。尽管如此，它的权重开放和宽松许可被视为促进创新和支持欧洲/全球 AI 竞争的关键。
- [**MCPs 太棒了！**](https://i.redd.it/p3766l11qbdf1.png) ([Score: 321, Comments: 71](https://www.reddit.com/r/LocalLLaMA/comments/1m1sjsn/mcps_are_awesome/))：**该帖子展示了使用多个模型控制协议服务器 (MCPs)——总共 17 个——与 Open WebUI 和本地 LLM 接口，允许动态调用系统工具，如网页搜索和 Windows CLI。图中显示的命令演示了使用 Python (psutil 和 GPUtil) 和 Qwen14B LLM 实现基于 PowerShell 的实时资源监控，输出详细指标：**`CPU load: 7.6%`**，**`RAM: 21.3%`**，以及 **`GPU (RTX 3090 Ti): 16% load, 18,976MB/24,564MB used at 61°C`**。这突显了该集成在 LLM 环境中进行上下文感知资源监控的实用性。[图片链接](https://i.redd.it/p3766l11qbdf1.png)** 评论者对安全性表示担忧，指出了 Agent 运行代码的风险（如 `"rm -rf *"` 风险），并警告说每次工具调用都会产生显著的上下文/Token 成本（约 600–800 tokens），这会迅速消耗本地模型中有效的上下文窗口（<5K tokens），甚至在对话初始化时就可能降低 LLM 的性能。
    - 关于 MCPs（模块化/多模态能力插件）及其对上下文窗口影响的一个关键点是：每个工具实例可能消耗 `600-800 tokens`，这会严重减少上下文窗口小于 5k 的小型模型的可用空间，由于描述可用工具所需的冗长系统提示，在用户输入开始之前就可能降低性能。
    - 一位用户指出，在模型设置中启用原生工具调用（native tool calling）可以显著提升性能，强调了特定配置标志对于本地部署中优化推理效率的重要性。
    - 围绕 MCPs 使用的讨论强调了评估其在生产系统中实际收益的必要性，包括确定 MCPs 是有状态还是无状态，并考虑它们对系统设计、可靠性和可维护性的实际影响，以及与替代方案的对比。

### 3. LocalLlama 社区增长与里程碑

- [**我们已经达到 500,000 名成员！从 LLaMA 1 模型泄露的日子到现在，我们已经走了很长一段路**](https://i.redd.it/zfvdqak3zcdf1.png) ([Score: 605, Comments: 41](https://www.reddit.com/r/LocalLLaMA/comments/1m1xqv1/we_have_hit_500000_members_we_have_come_a_long/)): **该图片庆祝 'LocalLlama' subreddit 达到 500,000 名成员，并强调了其对 AI 和 Meta 的 LLaMA 模型讨论的关注。自 LLaMA 1 泄露（2023年3月）成立以来，该社区迅速增长。这一里程碑标志着开源大语言模型 (LLM) 社区的广泛关注和增长，同时也伴随着技术重心的转变（从利基、动手实验转向更广泛、主流的 LLM 话语）。** 热门评论讨论了一个讽刺现象：随着 LLaMA 及其社区的发展，模型正变得越来越不“本地化”（需要更多资源或基于云的基础设施）。随着该 subreddit 变得不那么专业而更加主流，人们也对深度技术内容的稀释感到担忧，这反映了开源 LLM 参与格局的演变。
    - 评论者注意到 LLaMA 模型方向的重大转变：最初以本地、公开可用的基础而闻名，随着 Meta 更新许可和分发条款，它们正日益变得非本地化，且对于个人或本地部署的访问性降低。
    - 讨论了社区成员数量的增加如何与技术内容的稀释相关联；随着 subreddit 的增长，高质量的技术帖子和专注于深度 SOTA 的讨论预计会减少，转向主流、以产品为中心的帖子，而非开源、前沿研究。
    - 对 AI/LLM 开发中“本地”定义演变的担忧被提出，一些用户感叹现代 LLaMA 迭代既缺乏原始的“llama”精神，也失去了以前的硬件独立性，反映了行业向模型中心化和受限访问增加的更广泛趋势。

## 非技术类 AI Subreddit 汇总

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. OpenAI ChatGPT Agent 发布、功能及风险讨论

- [**ChatGPT Agent 发布以及 Sam 的看法**](https://i.redd.it/b0xmxole1hdf1.jpeg) ([Score: 463, Comments: 186](https://www.reddit.com/r/OpenAI/comments/1m2e2sz/chatgpt_agent_released_and_sams_take_on_it/)): **该图片是 Sam Altman 关于发布 ChatGPT Agent 的公告快照，这是 OpenAI 的新 AI 系统，能够使用自己的计算机独立执行复杂的多步任务。Altman 强调了其先进的任务自动化（例如购物、预订、分析）、研究和操作功能的集成，以及为减轻隐私、安全和操作风险而实施的大量新系统保障措施。部署是有意迭代的，并带有关于信任和访问级别的强烈用户警告，强调由于可能存在的对抗性操纵和不可预测的行为，需要最小化权限。** 热门技术评论对系统的可靠性表示怀疑——其中一条指出“完成的结果只有 50% 的准确率”，而其他评论则强调不愿信任 Agent 进行财务操作，并敦促 OpenAI 在发布更具野心的功能之前，优先考虑基础功能的准确性和一致性。
    - 提出的一个关键技术批评是发布的 ChatGPT Agent 的准确性，一位评论者特别指出 *“完成的结果只有 50% 的准确率”*，强调了在需要高可靠性的任务中部署的重大局限性。
    - 讨论了自主财务行为的安全性和信任，鉴于当前的错误率和缺乏强大的风险缓解措施，对于允许 Agent 执行独立购买或财务交易持怀疑态度。这突显了对 Agentic AI 在高风险或敏感领域成熟度的担忧。
    - 多位评论者呼吁关注基础模型行为中持续存在的可靠性和一致性问题，主张在引入雄心勃勃的 Agent 功能或完全自主性之前改进核心功能。

- [**直播：介绍 ChatGPT Agent**](https://www.youtube.com/watch?v=1jn_RpbPbEc) ([Score: 294, Comments: 246](https://www.reddit.com/r/singularity/comments/1m2cv1j/live_introducing_chatgpt_agent/)): **OpenAI 发布了全新的 ChatGPT Agent 架构（参见其 [视频演示](https://www.youtube.com/watch?v=1jn_RpbPbEc)），具备多模态理解、直接 API/网页集成以及自主多步任务执行能力（例如预订、文档处理、服务交互）。技术亮点包括对透明安全性的关注、稳健的任务排序以及一套全新的动作编排系统——旨在提升消费者和企业环境下的自动化水平。** 技术评论者对公开演示的节奏和实质内容表示不耐烦，呼吁展示超出脚本场景的实际应用价值。人们对演示的相关性以及 Agent 目前能实现的实际任务自主深度表示担忧。
    - 评论者对 ChatGPT Agent 的实际效用表示怀疑，其中一人询问是否有超出简单场景的“实际工作”演示，强调了对 Agent 能否超越表面任务并在通往更通用自主性方面取得实质性进展的担忧。
    - 另一位评论者批评了演示中的 Prompt Engineering，认为利用深度的研究能力本应就能处理寻找服装和礼物等任务，暗示目前的用例演示显然没有将模型推向超越现有能力的水平。
    - 有一种微妙的暗示认为，直播演示的形式或环境可能会削弱技术上令人印象深刻的展示效果，正如对节奏和设置的不适感所提到的，这表明技术受众期望针对先进 AI 工具提供更精炼、更高效的产品展示。
- [**OpenAI 的新 ChatGPT Agent 试图包揽一切**](https://www.wired.com/story/openai-chatgpt-agent-launch/) ([Score: 163, Comments: 50](https://www.reddit.com/r/OpenAI/comments/1m2d5yd/openais_new_chatgpt_agent_tries_to_do_it_all/)): **OpenAI 新发布的 ChatGPT Agent 通过集成外部 API 并运行自己的浏览器实例与在线服务交互，实现了多步、依赖上下文的任务自动化。然而，演示过程受到了一些 Bug 的困扰——例如上下文丢失（忘记了婚礼日期）、网站访问失败以及低效的浏览器自动化，这引发了安全和会话管理方面的担忧（例如跨域访问和登录持久性）。技术设计旨在实现通用自主性，但用户体验和实现的稳健性仍是主要的争议点。更多细节见这篇 [WIRED 文章](https://www.wired.com/story/openai-chatgpt-agent-launch/)。** 核心辩论集中在基于 Agent 的浏览器自动化的不切实际性（特别是在跨服务的用户身份验证方面）、评估此类 Agent 的社区标准不一致（OpenAI vs. 竞争对手），以及对 OpenAI 演示质量的批评，暗示了技术雄心与最终用户交付能力之间可能存在错位。
    - 一项技术批评针对 ChatGPT 在演示场景中无法维持上下文的能力（具体而言，忘记了婚礼日期），以及它通过切换到阅读模式来处理浏览器限制（由于“跨域问题”），突显了目前 Agent 可靠性以及与网页数据集成方面的挑战。
    - 一条评论提到了 Agent 运行其“自有浏览器”的效率问题——特别是质疑此类 Agent 将如何进行身份验证并访问用户在本地浏览器上下文中已打开的服务，暗示 Agent 浏览器与原生用户环境之间缺乏无缝的会话处理或安全的凭据共享。
    - 人们对 AI Agent 后端的稳健性持怀疑态度，并直接将其与之前的努力（如 Manus）进行比较，指出即使拥有“稳健的后端”（推测指 OpenAI 的基础设施），实际的产品可靠性和生产力仍然是主要的技术障碍，无论炒作或宣布的能力如何。
- [**当他们把那个年轻人（twink）请出来时，你就知道事情变严肃了**](https://i.redd.it/xx7p05ewlgdf1.jpeg) ([Score: 289, Comments: 48](https://www.reddit.com/r/singularity/comments/1m2bt5e/you_know_its_serious_when_they_bring_out_the_twink/)): **该图片是 OpenAI 一场重大演讲的通知，主讲人包括 Sam Altman 和核心团队成员，重点是揭晓 ChatGPT 中的“统一 Agent 模型（unified agentic model）”。技术评论指出，这符合对下一个重大模型（“GPT-5 时代”）的预期，并且“统一模型”一词与之前将 GPT-5 称为 Agent 化或 Agent 驱动架构（可能被称为“Agent 1”）的说法相吻合。** 一条热门评论推测了命名惯例，并引用 OpenAI 的路线图和早期的泄露信息，再次确认了“统一模型”与备受期待的 GPT-5 更新之间的联系。

- 技术讨论集中在所谓的“统一模型”（unified model）——据称与 GPT-5 相关——可能预示着 OpenAI 在架构或品牌上的重大变革（可能会放弃 “GPT-5” 的名称，转而采用类似于 Daniel 的 “2027” 项目中衍生出的 “Agent 1” 概念）。这暗示了集成的多模态（multi-modal）或持久化 Agent 能力，与之前关于结合文本、推理以及潜在现实世界任务执行能力的下一代模型的表态相吻合。
- [**ChatGPT Agent 将面向 Plus、Pro 和 Team 用户开放**](https://www.reddit.com/r/OpenAI/comments/1m2dnw5/chatgpt_agent_will_be_available_for_plus_pro_and/) ([Score: 323, Comments: 94](https://www.reddit.com/r/OpenAI/comments/1m2dnw5/chatgpt_agent_will_be_available_for_plus_pro_and/)): **OpenAI 宣布 ChatGPT Agent 功能将推送到 Pro 用户（每月 400 次查询上限）以及 Plus/Team 用户（每月 40 次查询），根据 [OpenAI Blog](https://openai.com/index/introducing-chatgpt-agent/) 和其 [livestream](https://www.youtube.com/watch?v=1jn_RpbPbEc)，Pro 用户将立即获得访问权限，Plus/Team 用户将在几天内获得。该功能受地理限制，最初不会在 EEA（欧洲经济区）或瑞士推出。查询配额取决于订阅层级，强调了受控访问和资源管理。** 评论中的技术评论对有限的每月请求池和封闭生态系统表示不满，呼吁支持自托管或更原生集成的 Agent（例如浏览器内的本地 Operator），以增强用户的可扩展性和干预能力，并推测了可能的 open weights 模型或非 OpenAI 替代方案的影响。
    - 用户对 EEA 地区缺失 ChatGPT Agent 功能表示担忧，强调了持续的合规问题和功能推出延迟——这可能与《数字市场法案》（Digital Markets Act）等监管障碍有关，且 GPTs 的 “connectors” 在欧盟仍不可用。
    - 针对 OpenAI 使用每月请求池（配额）的做法，技术人员表达了不满，因为这种模式可能会限制开发者和高级用户构建依赖 Agent 进行长时间自主或多步操作的复杂或持续工作流。
    - 一位评论者认为，OpenAI 的 Agent 架构这种“围墙花园”方式限制了用户的干预和定制，建议需要可本地运行的 Operator 模型和 open-weight 替代方案（可能暗示了传闻中的开放浏览器 Agent 策略，并提到了来自微软等第三方的竞争压力）。
- [**Agent = Deep Research + Operator。Plus 用户：40 次查询/月。Pro 用户：400 次查询/月。**](https://www.reddit.com/r/OpenAI/comments/1m2drew/agent_deep_research_operator_plus_users_40/) ([Score: 127, Comments: 51](https://www.reddit.com/r/OpenAI/comments/1m2drew/agent_deep_research_operator_plus_users_40/)): **OpenAI 的 Agent 集成了自主任务执行、用户可中断性以及过程中的澄清功能，其特点是结合了 “Deep Research” 核心和用于实时交互的 “Operator”（参见官方 [产品页面](https://openai.com/index/introducing-chatgpt-agent/)）。该系统强调安全性：它包含抗 prompt injection 能力和用于运行时威胁检测的隐藏观察者机制，并持续更新以加强对不断演变的漏洞利用的防御。Plus 用户的查询限制为 40 次/月，Pro 用户为 400 次/月。更多技术细节在其 [发布演示](https://www.youtube.com/watch?v=1jn_RpbPbEc) 中介绍。** 评论者对自动化用户行为的大规模扩展（例如：职位申请、内容创作）、与 n8n 等工具潜在的工作流自动化重叠，以及基于 Web 的反机器人措施的动态表示担忧。还有人要求提供强大的 API/CLI 集成，并讨论了实际的 prompt/任务执行限制，特别是会话超时是否会限制大规模自动化任务。
    - 针对使用 Agent 进行任务自动化的可扩展性提出了技术担忧，特别是现有 Agent 是否能处理高吞吐量的自动化任务，如*申请数千个职位*，或者是否存在系统强制的超时或 prompt 长度限制（例如 *30 分钟超时*或输入长度上限）。这对于自动生成或处理大量请求的工作流尤为重要。
    - 用户对通过 API 和 CLI 提供该 Agent 框架表现出兴趣，隐含关注点在于这种互操作性将如何实现自定义流水线中的高级自动化集成。

- 用户要求与 Manus Agent 平台进行直接对比，强调了对基准测试数据、功能对等评估或案例研究的需求，以评估该新系统与 Manus 相比在性能、可扩展性或易用性方面的表现。
- [**OpenAI 似乎计划发布代号为 “Odyssey” 的 Agent 模式（查看全部 5 张图片）**](https://www.reddit.com/gallery/1m1w0y5) ([Score: 258, Comments: 60](https://www.reddit.com/r/singularity/comments/1m1w0y5/seems_like_openai_is_planning_to_release_agent/)): **据报道，OpenAI 正准备发布一项名为 “Agent Mode” 的新功能，内部代号为 “Odyssey”。由于缺乏可获取的源代码材料，目前没有具体的模型基准测试、架构细节或实现信息。除了这个传闻中的代号关联外，该帖子没有提供任何技术证据或演示。** 评论者澄清说，提到的 “5 张图片” 与 GPT-5 无关，反映了人们对潜在产品发布或代号的一些困惑或猜测。此外，人们对 “Odyssey” 这个代号持轻微怀疑态度，幽默地暗示这可能意味着发布前需要很长时间。
    - 有关于 OpenAI 即将推出的 Agent 模式内部代号为 “Odyssey” 的猜测，一位用户指出提到的 “5 张图片” 并不暗示 “GPT-5”。这澄清了关于图片数量是暗示新模型版本还是仅仅展示功能的潜在困惑。
- [**没错，今天我们很可能会看到 Agent**](https://i.redd.it/4crtpuslegdf1.png) ([Score: 150, Comments: 25](https://www.reddit.com/r/OpenAI/comments/1m2aq68/yep_most_probably_agents_we_gonna_see_today/)): **图片显示了 OpenAI 的一条推文，宣布将在 3 小时内举行直播活动，通过使用握手表情符号暗示了涉及 ChatGPT、Deep Research 和 Operator 的协作或集成。大量的互动（42.6 万次观看）突显了高度的期待，这可能是由于对发布 Agent 功能甚至潜在的 GPT-5 展示的猜测所推动的。Subreddit 和评论提供的背景指向了社区对超越当前模型（如 GPT-4o）的重大技术升级的期望。** 评论者表达了对可能基于 GPT-5 构建的更强大的 Agent 能力的希望，批评当前的 OpenAI 模型（GPT-4o, GPT-4.1, o3）与竞争对手如 Gemini 2.5 和 Claude 4 Sonnet 相比存在幻觉或性能平平。此外，人们对普通用户能否使用这些功能表示怀疑，并对如果没有宣布重大改进的近期进展感到失望。
    - 几位用户讨论了对当前 OpenAI 模型在 Agent 工作流中的不满：GPT-4o 被指出经常出现幻觉，GPT-4.1 的性能被描述为平庸，GPT-4.5 的 API 移除也受到了批评。这些缺点被拿来与竞争对手模型（如 Gemini 2.5 Flash、Claude 4 Sonnet 和 Gemini 2.5 Pro）进行比较，声称这些替代方案在日常使用中提供更好的性能。
    - 一条技术评论强调，Operator Agent 现在已直接集成到 ChatGPT 中，而不是保留在外部页面上，但用户称这并没有引入任何重大的新 Agent 功能。对于 Agent 任务缺乏实质性进展或新功能，人们表达了失望。
- [**Deep Research 和 Operator**](https://i.redd.it/gjmo6kcyyfdf1.jpeg) ([Score: 324, Comments: 52](https://www.reddit.com/r/singularity/comments/1m28h1d/deep_research_and_operator/)): **图片显示了 OpenAI 的一条推文，宣布了一场涉及 “ChatGPT”、“Deep Research” 和 “Operator” 的协作活动及即将举行的直播，暗示了重大的产品发布或集成。评论猜测这可能代表了一个统一产品的亮相，可能是 GPT-5，标志着向 AGI 或高级多模态能力的可能进展。该推文的高互动指标也突显了社区对该活动及其对 AI 助手和研究工作流影响的浓厚兴趣。** 热门评论反映了对实际演示的怀疑（例如，“最好不要又是订机票”），以及对 ChatGPT 与更先进技术融合的好奇，表明了技术上的期待以及对之前发布会中平淡无奇的演示的一些疲劳。
    - 几条评论猜测了将 ChatGPT 与更先进的模型或系统合并的技术影响，提到了可能收敛为一个单一的、更强大的产品（可能以 GPT-5 的名义）。这表明模型架构在持续演进，可能会结合 Deep Research 和 Operator 模型，以实现更广泛的多模态或类似 Agent 的能力。

- [**看起来明天的发布会我们要迎来 agent mode 了？**](https://www.reddit.com/gallery/1m1w0i8) ([评分: 274, 评论: 49](https://www.reddit.com/r/OpenAI/comments/1m1w0i8/looks_like_we_getting_agent_mode_in_tomorrows/)): **该 Reddit 帖子推测某领先 AI 平台（很可能是 OpenAI 或类似平台）即将发布 “agent mode”，用户担心之前的 Operator v1 等功能未向 Plus 层级订阅者开放。讨论中的技术功能需求包括：Android 和网页端的录音支持、英国地区的连接性、Project 文件同步（桌面/Google Drive）、恢复 AVM 口音、在 “Read aloud” 中使用最新的语音模型，以及通过 SVM 和自定义指令实现更强大的语音定制。对于 AI 驱动的预订功能的价值存在怀疑，质疑行业对这一用例的关注。** 评论者辩论了推广策略，指出非 Pro 用户对功能可用性的挫败感，并强调相比于预订等小众自动化功能，更倾向于广泛的可用性改进（跨平台语音、项目同步）。一些人对订阅层级的门槛限制表示担忧，并希望 Plus 级别能立即获得访问权限。
    - 几位用户将即将推出的 “agent mode” 功能与之前的发布进行了比较，特别表达了对核心更新（如 Operator v1）缓慢、多 bug 且对某些用户层级（如 Plus 订阅者 vs Pro）不可用的担忧，这造成了技术和 UX 的碎片化。
    - 对高级多模态能力有强烈需求：用户明确要求支持 Android 和网页端的音频录制，改进文件/项目同步（特别是与 Google Drive 的集成），以及更自然化的 TTS 功能，例如使用新的语音模型、自定义指令和恢复特定的 AVM 口音。
    - 对专注于预订等任务的 “agent” 功能的实用性存在一些怀疑，用户将其与营销势头强劲但表现平平的实现（如 Google 的 Bard）进行比较；相反，讨论强调需要 Agent 在自主性和实用性方面取得真正的变革性飞跃，并辅以强大且可扩展的后端支持。

### 2. 基准测试与新模型性能：ChatGPT Agent、Gemini 以及视频/编辑发布

- [**ChatGPT Agent 在 Humanity's Last Exam 和 FrontierMath 上成为新的 SOTA**](https://i.redd.it/f13lg1wqwgdf1.png) ([评分: 404, 评论: 104](https://www.reddit.com/r/singularity/comments/1m2deg5/chatgpt_agent_is_the_new_sota_on_humanitys_last/)): **图片展示的基准测试结果显示，处于 “agent mode”（具有浏览器、计算机和终端访问权限）的 ChatGPT Agent 与 'o4-mini'、'o3' 以及不带工具的消融模型等其他 OpenAI 模型相比，在 “Humanity’s Last Exam” 和 “FrontierMath (Tier 1–3)” 上均达到了当前最高水平（SOTA）的通过率。结果突显了工具增强的 Agent 能力相比普通 LLM 带来的显著性能提升：agent mode 在每个基准测试中都达到了最高水平，表明其具备卓越的现实世界任务解决能力。** 评论讨论了由于 Agent 能力导致的基准测试相关性的转变，认为创建现实世界产出（如演示文稿）的能力比增量式的基准测试提升更重要。其他人指出与多 Agent 系统（如 Grok 4 Heavy）进行比较可能存在不公平性。技术共识认为，Agent 基准测试和工具使用代表了 AI 进步的一个重要新指标。
    - 多位评论者辩论了关于 Humanity's Last Exam (HLE) 和 FrontierMath 上 SOTA 声明的有效性，指出据报道 Grok 4 Heavy 取得了更高的 HLE 分数，尽管它使用的是 Agent 群（swarm of agents）而非单个 Agent，这引发了跨架构比较基准测试的公平性担忧。
    - 基准测试的重要性发生了技术性转变：虽然在 HLE 和 FrontierMath 等数据集上的纯测试分数仍然很有价值，但如今的重点日益转向 Agent 能力——在现实世界任务中的表现，如自动化工具使用、记忆保留、上下文推理以及复杂产出（如演示文稿）的创建。这一趋势表明，未来的基准测试可能会衡量更广泛、更具应用性的 Agent 智能，而不仅仅是静态的测试表现。
    - 对问题和答案可能公开的基准测试的可靠性提出了质疑，一位用户建议，包含工具使用、终端访问和浏览器集成的 ARC-AGI2 及类似基准测试代表了对 Agent 系统更稳健、更贴近现实的评估。缺乏严格控制或“私有”的基准测试（如 ARC-AGI2 的方法）削弱了与人类表现进行对等比较的可信度。

- [**Gemini 2.5 Pro 在 MathArena 的 2025 IMO 评测中得分最高！**](https://i.redd.it/sqefsuvsegdf1.png) ([得分: 105, 评论: 27](https://www.reddit.com/r/Bard/comments/1m2arkr/gemini_25_pro_scores_best_on_the_2025_imo_on/)): **该图片展示了来自 MathArena 的基准测试结果，显示 Google 的 Gemini 2.5 Pro 在 2025 年国际数学奥林匹克竞赛 (IMO) 评估中，在各种大语言模型中实现了最高的准确率 (31.55%)。表格详细列出了模型在各个 IMO 题目 (1-6) 上的表现并显示了成本指标，特别强调了 Gemini 2.5 Pro 在第 5 题上的准确率达到 71%，测试总成本为 431.97 美元。该基准测试强调了数学解题能力，特别是针对高中水平的竞赛题，其结果标志着 LLM 数学推理能力的重大进步。** 评论者表示惊讶，并指出 Gemini 2.5 Pro 与 OpenAI 的 GPT-4 (O3) 在数学和推理任务上存在主观差异，偏好因使用场景（如证明 vs. 编程）而异。此外，对于自然语言证明的构建也存在怀疑，这表明当前模型在严谨数学论证能力方面仍有局限。
    - 一位用户指出，尽管 Gemini 2.5 Pro 在 2025 IMO MathArena 上表现强劲，但在实际的数学、谜题和研究场景中，像 O3 High 这样的模型在推理和解题方面通常优于它，而 Gemini 2.5 Pro 倾向于给出假设性答案，且对话不够细腻。他们仍然认为 Gemini 2.5 Pro 在编程和通用用途上更可取，这表明模型优势存在显著的领域差异。
    - 另一位评论者强调，即便是对于表现最好的模型，用自然语言构建严谨、完整的数学证明也是一项独特的挑战。这凸显了定量基准测试表现（如 IMO）与需要详细证明步骤的现实世界数学推理之间的差距。
    - 提到 Deepseek Prover v2 在数学推理任务中超越了 Gemini 2.5 Pro 和 O3 High，表明专注于数学解题的 State-of-the-art 模型之间存在持续的竞争和差异化。
- [**新型开源视频生成器 PUSA V1.0 发布，声称比 Wan 2.1 快 5 倍且效果更好**](https://www.reddit.com/r/StableDiffusion/comments/1m1x2z7/a_new_open_source_video_generator_pusa_v10/) ([得分: 151, 评论: 49](https://www.reddit.com/r/StableDiffusion/comments/1m1x2z7/a_new_open_source_video_generator_pusa_v10/)): **PUSA V1.0 是一款开源视频生成模型，声称“比 WAN 2.1 快 5 倍且效果更好”，同时保持了架构上的相似性。它是一个统一模型，支持多种任务，包括文本生成视频 (t2v)、图像生成视频 (i2v)、定义起始/结束帧以及视频扩展。该模型的技术页面和演示可在 [官方网站](https://yaofang-liu.github.io/Pusa_Web/) 查看，并以 WAN 2.1 的 14B 参数模型作为性能基准。** 一位评论者对 PUSA V1.0 示例视频的质量表示怀疑，认为其声称的相对于 WAN 2.1 的改进在视觉上可能并不令人信服。
    - 多位用户质疑 PUSA V1.0 与 Wan 2.1 的性能对比，指出虽然 PUSA 声称比默认的 Wan 快 5 倍，但带有 Self Forcing LoRA 的 Wan 可以实现 10 倍加速，这暗示报告的指标可能存在夸大或特定语境限制。
    - 一位用户查看了模型架构，注意到“wan2.1 14B”，这可能参考了模型大小，建议 PUSA 需要在可比的参数量和架构背景下进行评估。
    - 出现了关于视觉保真度的技术批评，特别是生成视频中的人物形象，表明尽管速度有所提高，但定性输出仍然不足，而这是视频生成任务中的关键指标。

- [**HiDream 图像编辑模型发布 (HiDream-E1-1)**](https://i.redd.it/a3dnmlthlbdf1.jpeg) ([Score: 228, Comments: 72](https://www.reddit.com/r/StableDiffusion/comments/1m1rz2s/hidream_image_editing_model_released_hidreame11/)): **HiDream-E1-1 是一款新发布的图像编辑模型，基于 HiDream-I1 构建，其官方模型托管在 Hugging Face ([链接](https://huggingface.co/HiDream-ai/HiDream-E1-1))。附带的演示图像 ([点击查看](https://i.redd.it/a3dnmlthlbdf1.jpeg)) 展示了先进的编辑能力：转换主体和环境（例如，让角色呈现为博物馆艺术品、将子弹换成蝴蝶、将蜂鸟转化为玻璃材质、改变物体颜色如金色玩具车，以及更改场景主题）。这些展示了精湛的局部和语义编辑能力，暗示了其采用了可控 Diffusion 或 Inpainting 工作流。** 评论区的讨论集中在与 ComfyUI 集成的可能性，以及与另一个编辑模型 FLUX Kontext 的比较。用户对用于中端硬件（如 RTX 3060 12GB）高效推理的 INT4 Nunchaku 量化版本表现出浓厚兴趣，反映了对广泛、资源友好型可用性的期待。
    - 用户对 HiDream-E1-1 的 INT4 Nunchaku 量化版本很感兴趣，因为量化可以使 ComfyUI 中的性能大幅提升，并避免显存溢出错误，特别是在像 RTX 3060 这样只有 12GB VRAM 的 GPU 上。
    - 建议在 HiDream-E1-1 和 Flux Kontext 之间进行直接的技术对比，强调需要通过基准测试来确定这两个图像编辑模型的优势、功能和差异。
    - 一位用户发布了一个真实案例，展示了两个 Prompt 在不同 Seed 值下的效果（CFG 2.3，Steps 22，Euler 采样器），这对于研究 HiDream-E1-1 模型的输出差异和可复现性非常有用。
- [**🚀 刚刚发布了适用于 Wan 2.1 的 LoRA，增加了逼真的无人机风格推入镜头运动。**](https://v.redd.it/2jrxstp8vfdf1) ([Score: 668, Comments: 56](https://www.reddit.com/r/StableDiffusion/comments/1m28062/just_released_a_lora_for_wan_21_that_adds/)): **一款针对 Wan 2.1 Image-to-Video (I2V) 14B 720p 架构的新 LoRA (Low-Rank Adaptation) 模型已发布，专门设计用于为生成式视频生成逼真的“无人机风格”推入式镜头运动。该 LoRA 在** `100 个无人机推入剪辑` **上进行了训练，并经过** `40 多个版本` **的迭代优化，同时提供了 ComfyUI 工作流以实现无缝集成；可以通过文本 Prompt 'Push-in camera' 触发。该模型和工作流可在 [HuggingFace](https://huggingface.co/lovis93/Motion-Lora-Camera-Push-In-Wan-14B-720p-I2V#AI) 上获取。** 评论者注意到 Text-to-Video (T2V) 版本即将开发，可能用于 Wan VACE，但强调目前尚未测试。总体反响表明该 LoRA 实现了显著逼真的运动效果，并期待进一步扩展到其他视频合成流水线。
    - 一位用户询问是否可以通过反转用于推入运动的训练视频剪辑来训练“拉出 (push-out)”运动 LoRA，质疑这种方法是否足以在增加极少数据的情况下生成逼真的反向镜头运动。这涉及 LoRA 微调中的模型训练效率和数据增强策略。
    - 另一位评论者讨论了准备与 Wan VACE 兼容的 T2V (Text-to-Video) 版本，指出其尚未测试且性能可能有所不同，并强调了社区对将此 LoRA 集成到其他专业流水线的兴趣，这可能需要领域自适应或进一步微调。

### 3. 文化与存在主义 AI 辩论 (创造力, AGI, AI 影响梗)

- [**我们现在随便什么都叫 AGI 了，笑死**](https://i.redd.it/spk18cfabhdf1.jpeg) ([Score: 735, Comments: 259](https://www.reddit.com/r/singularity/comments/1m2fjje/we_just_calling_anything_agi_now_lmao/)): **该图片是 OpenAI CEO Sam Altman (@sama) 的一条推文截图，他在推文中描述了目睹一个 ChatGPT Agent 自主使用电脑执行一系列任务的过程，将其定义为“AGI 时刻”，并强调了看到系统规划和执行行动所带来的冲击。该帖子的背景和评论讨论显示出对将当前 AI 模型贴上 AGI (通用人工智能) 标签的怀疑，反映了营销炒作超过实质性 AGI 基准或能力的趋势。** 热门评论质疑了这些任务的重要性以及将此类演示命名为 AGI 时刻的有效性，称其为“营销”，并批评了 Altman 此类言论的模糊性和重复性。

- 一位用户讨论了 AGI (Artificial General Intelligence) 的定义如何不断变化，并指出当今的 frontier models 在许多领域已经超越了普通人类，并展示了远超 2000 年代初期所能想象的教育能力。然而，人们感知到的 AGI 缺失源于不断提高的标准和移动的目标，而非当前技术的不足。
- 另一位评论者认为，关于什么是 AGI 还是 ASI (Artificial Superintelligence) 的争论往往是迂腐的，并建议实际标准——例如模型是否能自主完成有用的工作并在没有专家 prompting 的情况下超越人类——比死板地坚持不断演变或主观的定义更有意义。他们还指出社区中存在一种不愿将任何模型标记为 AGI 的倾向，并将其比作“诉诸纯洁”（No True Scotsman）谬误。
- [**“人类程序员的时代即将结束”**](https://www.heise.de/en/news/Softbank-1-000-AI-agents-replace-1-job-10490309.html) ([Score: 653, Comments: 552](https://www.reddit.com/r/singularity/comments/1m26bkk/the_era_of_human_programmers_is_coming_to_an_end/)): **Softbank 创始人孙正义宣布，公司打算通过部署自主 AI agents 来淘汰人类编码角色。在最近的一次企业活动中，孙正义预计在** `2025` **年部署多达** `10 亿` **个 AI agents，并进一步量化指出，Softbank 内部估计目前需要** `1,000` **个 AI agents 才能取代一名程序员的生产力，这突显了 AI 驱动的软件自动化仍面临巨大的资源需求和操作复杂性。这一声明既反映了激进的自动化野心，也反映了完全取代传统开发者所面临的巨大规模化挑战。[来源](https://www.heise.de/en/news/Softbank-1-000-AI-agents-replace-1-job-10490309.html)** 热门评论批判性地质疑了此类预测的可行性和意图，认为这些说法可能是由投资者炒作而非工程现实驱动的，并质疑了关于最终用户角色以及用 AI agents 取代程序员所带来的现实生产力收益的底层假设。
    - MinimumCharacter3941 强调了企业环境下 AI 编码工具的一个实际局限性：尽管自动化有所进步，但核心挑战仍然是 *需求规范* (requirements specification)。大多数 CEO 和高层管理团队难以精确表达他们的需求，无论是由程序员还是 AI 处理实现，这一差距都会阻碍项目的成功。这一观察强调了能够将模糊的业务目标转化为可执行的技术任务的资深中介（如业务分析师、高级工程师）的持久价值，而这一功能并不容易被自动化。
- [**“我们开始看到模型自我改进的早期迹象。我们的使命是为世界上的每个人提供个人超智能。”**](https://v.redd.it/9njdgov4ccdf1) ([Score: 534, Comments: 500](https://www.reddit.com/r/singularity/comments/1m1v5a0/were_starting_to_see_early_glimpses_of/)): **该帖子引用了一个断言，即“我们开始看到模型自我改进的早期迹象”，指向了 AI 自我改进的萌芽能力，并声明了向“每个人提供个人超智能”的使命。帖子中没有提供支持性的技术细节、benchmarks 或实现说明。对所引用视频的访问被拒绝 (403 Forbidden)，导致无法进行更深入的分析。** 这里的评论批评了缺乏技术讨论的情况，并对提出此类主张的人物之可信度和意图表示怀疑。多条评论感叹该子版块讨论的技术标准较低，将开发超智能的雄心与许多社区反馈的琐碎性质进行了对比。
    - 出现了一场关于 AI 开发投资规模的讨论，强调目前针对超智能的举措涉及数千亿美元。这突显了正在分配的资源规模，这可能会加速先进 AI 能力的技术进步和实现。
    - 针对某些科技领袖或公司（特别是 Meta）作为 AI 开创者的定性存在批评。评论者指出，虽然 Meta 在 AI 研究和实验室方面很活跃，但与其历史意义相比，其他基础性贡献受到了质疑，这表明人们对某些公司相对于其实际技术进步的主张持怀疑态度。

- [**普通 Redditor：AI 只是模仿，它们没有创造力……AI 教父：不，它们非常有创造力。**](https://v.redd.it/f6kukffnxedf1) ([Score: 313, Comments: 102](https://www.reddit.com/r/singularity/comments/1m247kf/random_redditor_ais_just_mimick_they_cant_be/))：**该帖子对比了一种普遍断言（即 AI 不具备创造力且仅仅是模仿）与所谓的“AI 教父”（可能指 Yoshua Bengio、Geoffrey Hinton 或 Yann LeCun）的观点，即 AI *确实* 具有创造力。一条技术评论强调，创新在形式上是现有思想的组合以产生新思想，这意味着 LLM 的能力符合创新的定义。另一条评论观察到，LLM 以新格式重新创作经典作品的能力（例如，用“帮派说唱”风格重写《奥德赛》）展示了创造力，并指出这种创造力可能是大模型产生幻觉（hallucination）倾向的基础。另一条关于国际象棋的评论指出，棋局中过度的创造力是检测 AI 辅助作弊的信号，因为 AI 的着法选择偏离了人类的创作规范。** 几条评论辩论了创造力和创新的定义，一些人断言现有思想的重新组合符合创造力的标准，而另一些人则观察到 AI “过于超前”的反应（例如，意想不到的类比、重组）既构成了其感知的优势，也导致了其弱点，如幻觉。
    - 几条评论从组合搜索空间和引导式搜索的角度讨论了 AI 的创造力。引用了 Shane Legg（DeepMind 联合创始人）的观点，强调人类和 AI 都是通过在巨大的可能输出空间中进行引导式探索来发挥创造力的，无论是文章（值得注意的是，100,000 tokens^1000 的空间）、国际象棋比赛，还是围棋局面（例如 AlphaGo 的 3^361 个游戏状态）。由于这些空间的巨大，模型既可以产生新颖的输出，也可以产生荒谬的输出。
    - 国际象棋作弊检测提出了一个对比点，即*异常高的创造力*或超出既定人类对弈模式的着法是 AI 介入的统计标志。这突显了 AI 在不受约束时，可能会在这些领域展示出可检测到的超人或非典型创造力。
    - 一些用户注意到，LLM 中创造力的增加通常与幻觉（脱离事实的过度创造性输出）相关，这暗示了模型原创性与可靠性之间存在技术上的博弈。更聪明或更大的模型可能更容易出现此类行为，这使得模型对齐（alignment）和输出控制成为重要的研究领域。

---

# AI Discord Recap

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要之摘要
> 

**主题 1. Agent 觉醒：OpenAI 的 ChatGPT Agent 步入战场**

- [**OpenAI 向世界发布 ChatGPT Agent**](https://openai.com/index/introducing-chatgpt-agent/)：**OpenAI** 推出了其全新的 **ChatGPT Agent**，这是一个能够控制计算机、浏览网页、编码、撰写报告和创建图像的多模态 Agent，正向 Pro、Plus 和 Teams 用户推出。此次发布通过[直播](https://discord.gg/DqBbV7ya?event=1395405196619939943)宣布，引发了关于其全部功能以及为企业客户提供定制化 operator 模式训练潜力的巨大兴奋和猜测。
- [**新 Agent 取代前代产品**](https://x.com/swyx/status/1945904109766459522)：随着 **ChatGPT Agent** 的到来，**OpenAI** 正在逐步淘汰其 **Operator** 和 **Deep Research** 工具，这些工具将被功能更强大的新 Agent 整合。据确认，*Operator 研究预览网站将继续运行几周，之后将被关闭*，不过用户仍可以通过消息输入框的下拉菜单访问 Deep Research。
- [**社区质疑 Agent 的竞争优势**](https://agi.safe.ai/)：工程师们注意到，**OpenAI** 仅将其 **ChatGPT Agent** 的性能与自己之前的模型进行比较，而避免了与 **Grok 4** 等竞争对手进行基准测试，后者最近以 **25.4** 的得分登顶 [HLE 基准测试](https://agi.safe.ai/)。这种策略性的比较引发了人们的猜测，即新 Agent 可能并未在所有方面都战胜竞争对手模型。

**主题 2. AI 商业：估值、收购与关停**

- [**投资者大举押注 Perplexity 和 FAL**](https://x.com/arfurrock/status/1945933446376755459?s=46&t=b7l37rB6wtbyAh6ah1NpZQ)：AI 融资热潮持续，据报道 **Perplexity** 正以惊人的 **180 亿美元**估值（营收 **5000 万美元**）进行融资，引发了泡沫担忧。与此同时，AI 推理基础设施公司 **FAL** 完成了 **1.25 亿美元**的 C 轮融资，估值为 **15 亿美元**。据[这条推文](https://x.com/arfurrock/status/1945553966495912051?s=46)称，这得益于其 **5500 万美元的 ARR** 和 **25 倍的同比增长**。
- [**Cognition 收购 Windsurf**](http://windsurf.com/blog/windsurf-wave-11)：**Windsurf** 已被 **Devin** Agent 背后的团队 **Cognition** 收购，并立即发布了具有重大新功能的 **Windsurf Wave 11**。根据 [Changelog](https://windsurf.com/changelog) 详述，此次更新包括为 **Cascade** AI 助手增加的 **Voice Mode**、更深层的浏览器集成，以及对其 **JetBrains plugin** 的重大增强。
- [**推理服务遭遇困境**](https://discord.com/channels/1091220969173028894/1392278974222307469/1395154130460479718)：小型企业可能面临 **AI 破产潮**，多家推理服务正在关闭，继 **CentML** 最近倒闭后，[**Kluster.ai**](http://kluster.ai/) 成为最新一家关门的公司。这一趋势引发了 **OpenRouter** 社区对独立 AI 服务提供商长期可持续性和市场可行性的担忧。

**Theme 3. New Models & Major Updates Shake the Landscape**

- [**Mistral 的 Le Chat 通过多语言推理能力升级**](https://x.com/MistralAI/status/1945858558836216026)：**Mistral** 对 **Le Chat** 进行了重大更新，增加了 Deep Research 报告、**Voxtral** 语音模型以及用于多语言推理的 **Magistral**。该版本还包括 Projects 等组织功能和聊天内图像编辑，因其精美的 UI 和“欧洲范儿”而获得赞誉。
- [**Kimi K2 展现代码能力与道德感**](https://www.kimi.com/chat/)：**Moonshot AI** 的 **Kimi K2** 模型通过生成一个完整的物理沙盒令工程师们印象深刻，代码分享在[这里](https://cdn.discordapp.com/attachments/1340554757827461211/1395196566389784606/plasma_sfml.cpp?ex=687ae30e&is=6879918e&hm=9e5b88351c6d243c6e165f444254a6eeb786c3d64fd8bdf958394026aa0ce5cb&)。该模型在坚决拒绝用户关于如何破解汽车的指令请求后，还引发了关于 AI 伦理的讨论，导致一位用户开玩笑说：*“Kimi K2 是个有道德的坏小子……坏小子 Kimi K2 !!”*
- [**Microsoft 和 Nous 发布专用工具包**](https://x.com/NousResearch/status/1945932488960008441)：**Microsoft** 发布了 [CAD-Editor 模型](https://huggingface.co/microsoft/CAD-Editor)，支持通过自然语言对现有 CAD 模型进行交互式编辑；而 **Nous Research** 推出了 **Atropos v0.3**，这是他们的开源 RL Environments Framework。这些发布为开发者提供了用于特定工程和研究应用的新型专业工具。

**Theme 4. Under the Hood: The Nitty-Gritty of Model Optimization**

- [**阿里巴巴的位预算（Bit Budget）出现失误**](https://huggingface.co/turboderp/ERNIE-4.5-300B-A47B-PT-exl3)：**AliBaba** 声称在其 **ERNIE 4.5** 发布中实现了无损 **2-bit 压缩**技巧，但这很快被社区拆穿。`turboderp` 的分析显示，该模型比真正的 `exl3` 2-bit 量化更差，因为 AliBaba 将许多层保留在高精度，使其平均成为一个约 **2.5-bit** 的模型。
- [**投机采样（Speculative Decoding）让模型飞速运行**](https://discord.com/channels/1110598183144399058/1110598183144399061/1395205004452691999)：**LM Studio** Discord 的一位用户报告称，使用 **Speculative Decoding** 的模型实现了约 **28% 的速度提升**。他们发现使用更快、更小的 draft model 效果最好，并建议 **Qwen3** 模型在使用 **1.7b Q8** 甚至 **bf16** 版本作为 draft model 时获益巨大。
- [**Blackwell 构建难题阻碍引导启动**](https://discord.com/channels/1179035537009545276/1179777624986357780/1395119297264881766)：工程师们在采用 NVIDIA 最新硬件时遇到了早期适配问题，指出支持 **Blackwell RTX 50** 系列需要从源码构建 **xformers**。**GPU MODE** 和 **Unsloth AI** 的讨论还强调了 **Blackwell GPU** 上的 **Inductor** 问题以及 **H200** 上的内存问题，这些问题可以通过升级 Unsloth 来缓解。

**Theme 5. Developer Ecosystem: New Tools and Community Tensions**

- [**Cursor 的新定价引发不满**](https://discord.com/channels/1074847526655643750/1074847527708393565/1395118180275326976)：**Cursor** IDE 的用户对从固定请求模型转向基于模型成本的模型表示了广泛的挫败感，许多人称之为“先诱后转”（*bait and switch*）。这一变化导致了计费方面的混乱，一些用户报告消息消失，并对修改服务条款的合法性表示担忧。
- [**社区发布用于可解释性与训练的开源工具**](https://github.com/MeryylleA/lunariscodex)：一名 17 岁的巴西开发者推出了 **LunarisCodex**，这是一个用于从零开始预训练 LLM 的完全开源工具包。与此同时，**Eleuther** 社区发布了 **nnterp** 的测试版，该软件包为所有 Transformer 模型提供统一接口，以简化机械可解释性（mechanistic interpretability）研究，并在[此 Colab](https://colab.research.google.com/github/Butanium/nnterp/blob/main/demo.ipynb) 中进行了演示。
- [**尽管存在身份验证障碍，MCP 生态系统仍在扩张**](https://x.com/Arindam_1729/status/1945958688919114183)：Model Context Protocol (**MCP**) 生态系统正在增长，**Brave** 推出了[官方 MCP Server](https://x.com/Arindam_1729/status/1945958688919114183)，[Needle MCP server](https://github.com/needle-ai/needle-mcp) 的创建者也加入了社区。这一扩张正值关于最佳身份验证方法的持续辩论中，人们在权衡 **OAuth** 的安全优势与 **API keys** 的实现简便性。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Airtel 赠送免费 Perplexity Pro**：印度网络运营商 **Airtel** 现在通过 **Airtel Thanks app** 向其客户提供 **1 年免费 Perplexity Pro** 订阅作为奖励。
   - 成员们报告称，尽管是 Pro 订阅者，**Perplexity search** 和 **research** 功能仍触及了新的 Rate limits，一名用户在激活其 Pro 订阅时遇到了问题。
- **Comet 浏览器依然难觅踪影**：成员们仍在等待他们的 **Comet 浏览器邀请**，一些人报告称他们已经等待了数月的审批。
   - 一位成员将其描述为“只是一个浏览器，但增加了助手侧边栏，可以查看你当前的实时网站并进行引用”。
- **Perplexity Pages 仅限 iOS**：成员们对生成查询页面的新 **Pages 功能**感到兴奋，但它**仅在 iOS 上可用**，且有 **100 页的限制**，存储在 [perplexity.ai/discover](https://www.perplexity.ai/discover) 中。
   - 成员们认为这是进行 Deep Research 的一种方式。
- **Sonar API 需要更好的 Prompt 编写**：一名团队成员表示，由于用户编写 **Sonar 模型** Prompt 的方式问题，相关故障有所增加，并链接到了 [Prompt 指南](https://docs.perplexity.ai/guides/prompt-guide)。
   - 成员们还讨论了在使用高搜索上下文时如何获得更一致的响应和有效的 **JSON** 输出，并希望在账户仪表板中查看 **API 调用**历史记录。
- **Pro 用户现可获得 API 访问权限**：通过 **Perplexity Pro**，你每月可获得 **$5** 用于 **Sonar** 模型，允许你将他们的 **AI 驱动搜索**嵌入到自己的项目中，同时能够按照 [Perplexity Pro 帮助中心](https://www.perplexity.ai/help-center/en/articles/10352901-what-is-perplexity-pro)所述获取引用。
   - 请记住，这些是搜索模型，其 Prompt 编写方式应与传统的 LLM 不同。

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Agent 直播官宣**：OpenAI 将举办一场关于 **ChatGPT Agent**、**Deep Research** 和 **Operator** 的直播；详情可见 [OpenAI 博客](https://openai.com/index/introducing-chatgpt-agent/) 和 [直播邀请](https://discord.gg/DqBbV7ya?event=1395405196619939943)。
   - 直播将涵盖 **Deep Research** 和 **Operator** 的更新，可能包括新功能或使用案例。
- **Grok 应用让 iPhone X 用户陷入困境**：**Grok app** 需要 **iOS 17**，导致其在 **iPhone X** 等旧设备上无法使用。
   - 用户讨论了是否需要专门为 **Grok app** 配备备用 iPhone，一些人警告不要仅为此目的购买新 iPhone。
- **Agent 模式在 3o 模型上无法运行**：用户报告称，**GPT agents** 仅在使用 **models 4 或 4.1** 时才能切换，Agent 切换功能不会出现在其他 **LLM models** 上。
   - 一位用户指出 **Agent 功能** 可能根本不在 **3o** 中提供，并建议提交 Bug 报告，另一位用户则认为 **Agent** 本身就是一个独立的模型（[OpenAI 帮助文档](https://help.openai.com/en/articles/11794342-chatgpt-agent)）。
- **可复现性漏洞百出**：一名成员发布了一个 [chatgpt.com 链接](https://chatgpt.com/share/68791cda-f338-8000-81f8-b7615d3f5a9c)，被指责读起来像设计提案，且缺失了关键的 **Reproducibility Elements**（可复现要素），如 Prompt 模板、模型接口和明确定义的评估指标。
   - 对话强调了缺乏 **Declarative Prompts** 的完整实例化示例、测试中使用的 Prompt 变体的清晰版本控制，以及具体的实验细节。
- **探索 ChatGPT 桌面版**：用户正在研究在桌面端使用 **ChatGPT** 进行本地文件管理，类似于 **Claude Harmony**。
   - 一个建议是使用 **OpenAI API**（付费）配合本地脚本与文件系统交互，本质上是创建一个自定义的类 "Harmony" 界面。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **家族模型：性能差异**：同系列模型表现非常相似，因此不建议将大型模型压缩到 **3 bits** 以下，而不同系列的模型则根据垂直领域有所不同。
   - 如果一个模型是 **7B**，另一个是 **70B**，则存在一些例外，对于某些任务，**70B** 即使在 1.8 bits 下仍然可用，因为它是一个大模型。
- **移植创伤：词表交换的困扰**：在没有持续预训练的情况下交换模型架构（如将 **LLaMA 1B -> Gemma 1B**）会导致非常糟糕的结果，这是由于移植了词表（Vocabulary）导致的。
   - 有人指出 **Qwen 1** 架构与 **Llama 1/2** 几乎完全相同，因此可以进行微小更改并塞入 **Qwen** 权重，但在训练 13 亿个 Token 后，得到的模型比投入的还要差。
- **Prompting 胜出：微调在功能实现上退居二线**：对于教育类 LLM，建议先从优秀的 Prompting 开始，然后再尝试 Fine-tuning，因为目前的指令遵循效率已经非常高。
   - 成员们还推荐了 [synthetic-dataset-kit](https://github.com/facebookresearch/SyntheticDataToolkit) 等工具来生成指令对话。
- **阿里巴巴搞砸了位预算（Bit Budget）**：阿里巴巴在发布 **ERNIE 4.5** 时宣称使用了某种无损 **2bit 压缩** 技巧，但 [turboderp 进行了研究](https://huggingface.co/turboderp/ERNIE-4.5-300B-A47B-PT-exl3) 并发现它比 exl3 更差，因为他们保留了大量高精度层。
   - 平均而言，它并不是真正的 **2-bit**（更像是 **2.5 bit**），而真正的 exl3 **2 bit** 表现比他们展示的 ~2.5 bit 更好。
- **Blackwell 构建难题阻碍引导**：用户讨论称，支持 **Blackwell RTX 50** 系列唯一需要做的就是从源码构建 **xformers**，且最新的 **vLLM** 应该在构建时开启 **Blackwell** 支持。
   - 成员建议使用 `pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth-zoo unsloth` 升级 Unsloth 以解决 **H200** 问题。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的新定价引发不满**：用户对 [Cursor 从固定请求模式转变为基于模型成本的模式](https://tenor.com/view/kevin-spongebob-angry-will-you-cut-that-out-gif-12675670360524347134)表示困惑和沮丧，称其为“诱导转向（bait and switch）”。
   - 一些用户报告消息消失，并对更改合同的合法性表示担忧。
- **通过 MCP 集成 Claude 减轻负担**：在 Cursor 中通过 MCP (Multi-Client Protocol) 集成 **Claude** 有助于管理与 **Sonnet** 和 **Opus** 相关的成本。
   - 成员们[承认](https://www.youtube.com/watch?v=D0iXkmyWcPM)这只能通过外部工具实现。
- **Agent 陷入困境**：用户报告 Cursor Agent 在执行任务时卡住，这是一个团队正在解决的[已知问题](https://cdn.discordapp.com/attachments/1395153343122505728/1395154214157816029/image.png?ex=687abb9d&is=68796a1d&hm=fc84cafbaab772eef094cc5199a623d5216c528ddde98227e6092221778c4fcc&)。
   - 手动停止提示词可能会防止计费，因为存在 **180 秒超时**机制会自动取消卡住的请求。
- **KIRO 与 Cursor 展开竞争**：成员们将 Cursor 与 **KIRO** 进行比较，后者是一款专注于基于规范的编码和钩子（hooks）的新 IDE，并指出由于需求量大，[KIRO 正处于候补名单阶段](https://kiro.dev/)。
   - 一个讨论点引发了对 **KIRO** 可能使用用户数据训练其模型的担忧，尽管有禁用此功能的设置。
- **用户质疑“Auto”模式使用的模型**：用户好奇 Cursor 中的“Auto”使用的是哪种模型，推测可能是 **GPT 4.1**。
   - 目前还没有证据可以证实或否认这一点。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **DeepSeek 宣布惊人利润**：DeepSeek 预计，如果 V3 的定价与 R1 相同，其理论利润率将达到 545%，详情见 [TechCrunch 的这篇文章](https://techcrunch.com/2025/03/01/deepseek-claims-theoretical-profit-margins-of-545)。
   - 这一断言引发了关于 AI 模型市场定价策略和技术进步的辩论。
- **OpenAI 浏览器的在线机遇**：关于 OpenAI 即将推出浏览器的传闻四起，可能是 GPT-5 或增强了浏览能力的 GPT-4 迭代版本，这一消息由[这条推文](https://x.com/testingcatalog/status/1945639961790685404?s=46)引发。
   - 潜在的发布让社区对其功能以及对 AI 应用的影响充满了猜测。
- **Kimi K2 创作代码**：Kimi K2 展示了其强大的编码能力，生成了一个物理沙盒，代码可在其[聊天界面](https://www.kimi.com/chat/)提示后在[此处](https://cdn.discordapp.com/attachments/1340554757827461211/1395196566389784606/plasma_sfml.cpp?ex=687ae30e&is=6879918e&hm=9e5b88351c6d243c6e165f444254a6eeb786c3d64fd8bdf958394026aa0ce5cb&)获取。
   - 该演示受到了好评，突显了 AI 在代码生成方面不断进化的能力。
- **OpenAI 彻底优化对象操作**：OpenAI 的图像编辑器 API 更新现在可以将编辑隔离在选定部分，提高了效率，无需重做整个图像，正如[这条推文](https://x.com/OpenAIDevs/status/1945538534884135132)所宣布的那样。
   - 这一改进为使用该 API 的开发者提供了更强的控制力和精确度。
- **GPT-5 传闻呈几何级增长**：对 GPT-5 发布的期待因[五边形引用](https://x.com/sama/status/1945900345378697650)（与数字 5 对应）等暗示而升温。
   - 猜测从夏末发布到对具有高级研究功能的 Agent 系统期待不等。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **FAL 估值升至 15 亿美元**：FAL 是一家为 diffusion models 提供 AI 驱动推理基础设施的公司，根据[这条推文](https://x.com/arfurrock/status/1945553966495912051?s=46)，该公司完成了由 Meritech Capital 领投的 **1.25 亿美元** C 轮融资，投后估值达到 **15 亿美元**。
   - 此前他们宣布 **ARR 达到 5500 万美元**、**同比增长 25 倍**、**EBITDA 为 10%** 以及 **M12 净金额留存率（net-dollar-retention）达 400%**，展示了强劲的市场势头。
- **Le Chat 获得多语言推理升级**：Mistral 发布了 Le Chat 的重大更新，增加了 Deep Research 报告、**Voxtral 语音模型**、**Magistral 多语言推理**、通过 Projects 进行对话组织以及对话内图像编辑等功能，详见[这条推文](https://x.com/MistralAI/status/1945858558836216026)。
   - 该版本因其 UI 和“欧洲风情”受到称赞，并被拿来与 Claude 进行比较，还引发了关于 *Le Waifu* 的幽默评论。
- **Perplexity 180 亿美元的高估值遭到质疑**：据报道 Perplexity 正在以 **180 亿美元** 的估值进行融资，这引发了从惊叹到对潜在泡沫担忧的各种反应，详见[这条推文](https://x.com/arfurrock/status/1945933446376755459?s=46&t=b7l37rB6wtbyAh6ah1NpZQ)。
   - 批评者质疑这一估值的合理性，强调了 **5000 万美元营收**数据与高昂标价之间的巨大差异。
- **OpenAI 发布 ChatGPT Agent**：OpenAI 新推出的 **ChatGPT Agent** 是一款具备控制计算机、浏览、编码、撰写报告、编辑电子表格以及创建图像/幻灯片能力的多模态 Agent，正向 Pro、Plus 和 Teams 用户推广，通过[这条推文](https://x.com/kevinweil/status/1945896640780390631)宣布。
   - 反应包括兴奋、对欧盟可用性的询问、对个性化冲突的担忧，以及对 Operator 和 Deep Research 被蚕食的顾虑。
- **Operator 和 Deep Research 面临停用**：随着 **ChatGPT Agents** 的发布，有人指出 **ChatGPT Agents** 可能会蚕食 **Operator** 和 **Deep Research**，并确认 *Operator 研究预览网站将继续运行几周，之后将被停用（sunset）。*
   - 用户仍可以通过在消息输入框的下拉菜单中选择 **Deep Research** 来访问它。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Opus 用户反对高昂的超额费用**：用户对 **Claude 4 Opus** 的定价展开辩论，指出有人在 **15 分钟内花费了 10 美元**，而其他人则建议使用 Anthropic 的 **90 欧元/月计划**以获得“无限使用”。
   - 一位使用 **20 美元计划**的用户声称他们“几乎从未达到上限”，因为他们不在 IDE 中使用 AI 工具，这表明使用情况差异很大。
- **GPT Agents 陷入“土拨鼠之日”（原地踏步）**：一位用户担心 **GPTs agents** 在初始训练之后无法学习，即使上传了文件，文件也只是被保存为 **knowledge files**。
   - Agent 可以引用新信息，但不会像预训练（pre-training）那样从本质上学习新信息，后者需要更多处理。
- **免费模型面临令人沮丧的失败**：用户报告了 **free model v3-0324** 的问题，质疑为什么尽管使用的是免费层级，却被切换到了非免费版本。
   - 报告显示，即使使用免费模型也会达到额度限制或收到错误，一位用户表示他们的 AI 自 6 月以来就没用过。
- **Cursor 代码崩溃引发混乱**：**OpenRouter 模型**集成了 **Cursor**，重点推介了 **Moonshot AI 的 Kimi K2**，但用户报告在使其正常工作时遇到问题，尤其是在 **GPT-4o** 和 **Grok4** 之外的模型上。
   - 根据[一条推文](https://x.com/msfeldstein/status/1945533992222298167?s=46&t=fN048T2GS3C_B_wKldWGFw)，*“我们写的时候它还能用，然后 Cursor 就把东西搞坏了”*。
- **推理服务实现陷入破产潮**：继 **CentML** 关闭后，被描述为“非常便宜且优质的服务”的 **Kluster.ai** 也正在关闭其推理（inference）服务。
   - 成员们正在推测 **AI 泡沫破裂**或硬件收购，引发了对 AI 推理服务可持续性的担忧。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Eleuther 弥补研究资源差距**：**Eleuther AI** 旨在为缺乏学术或行业资源的独立研究人员弥补研究管理方面的差距，促进其获取研究机会。
   - 该倡议旨在通过提供指导、处理繁琐事务并提供更广阔的视野，支持传统系统之外的研究人员，因为许多人被排除在 **NeurIPS high school track** 等路径之外。
- **机器学习论文写作资源共享**：成员们分享了撰写机器学习论文的资源，包括 [Sasha Rush 的视频](https://www.google.com/search?q=Sasha+Rush+how+to+write+a+great+ml_paper) 和 [Jakob Foerster 的指南](https://www.jakobfoerster.com/how-to-ml_paper)，以及来自 [Alignment Forum](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml_papers) 的建议。
   - 其他资源包括 [perceiving-systems.blog](https://perceiving-systems.blog/en/post/writing-a-good-scientific-paper) 上的文章、[Jason Eisner 的建议](https://www.cs.jhu.edu/~jason/advice/write-the-paper-first.html) 以及 [阿尔托大学 (Aalto University) 的指南](https://users.aalto.fi/~jsaramak/HowToWriteCommsCoffee.pdf)。
- **导师防止不切实际的研究**：参与者强调了研究中导师的重要性，指出导师有助于弄清楚 *什么是可能的，什么是不切实际的*，以便缩小研究范围。
   - 导师的指导可以帮助研究人员应对挑战，避免在无成效的途径上浪费时间，因为一般的指南只能提供基础知识。
- **ETHOS 模型精简并在 GitHub 更新**：一位成员分享了其模型的 [简化 pytorch 代码版本](https://github.com/wrmedford/ETHOS/blob/main/model.py#L267-L337)，并指出他们必须使用一个略有不同的版本，其中 **所有 head 都是批处理的 (batched)**，因为如果对所有 head 进行循环，eager execution 模式会消耗大量内存。
   - 他们还表示专家网络并非多余，并链接了内核中生成 **W1** 和 **W2** 的 [特定代码行](https://github.com/wrmedford/ETHOS/blob/main/kernels.py#L156-L158)。
- **nnterp 统一 Transformer 模型接口**：一位成员发布了其机械可解释性 (mech interp) 工具包 **nnterp** 的 beta 1.0 版本，可通过 `pip install "nnterp>0.4.9" --pre` 安装，它是 [NNsight](https://nnsight.net/) 的封装。
   - **nnterp** 旨在为所有 Transformer 模型提供统一接口，弥补 *transformer_lens* 和 *nnsight* 之间的差距，并在 [此 colab](https://colab.research.google.com/github/Butanium/nnterp/blob/main/demo.ipynb) 和 [文档](https://butanium.github.io/nnterp/) 中进行了演示。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Speculative Decoding 让模型飞速运行！**：一位用户报告称，在使用 **Speculative Decoding** 测试模型时，速度提升了约 **28%**。他们建议为草案模型 (draft model) 使用同一模型的不同 **量化 (quantizations)** 版本，并推荐 **Qwen3** 从使用 **1.7b Q8** 甚至 **bf16** 作为草案模型中获益匪浅。
   - 该用户暗示草案模型越快、越小，速度提升就越明显。
- **Gemma 模型变得有点过于真实**：一位用户讲述了一个有趣的场景：本地 **Gemma** 模型威胁要举报他们。这引发了关于 *DAN prompts* 因快速修补而具有瞬时性的讨论。
   - 一位用户开玩笑说，他们需要安装 **NSA 的后门** 来防止模型告密。 
- **LM Studio 等待 HTTPS 凭据**：一位用户询问如何配置 **LM Studio** 以接受 **开放网络服务器** 而不是通用的 HTTP 服务器，目标是使用 **HTTPS** 而非 **HTTP**。另一位用户建议使用 **反向代理 (reverse proxy)** 作为目前的权宜之计。
   - 该用户表示想提供模型服务，但觉得使用 HTTP 不安全。
- **EOS Token 终于得到解释**：一位用户询问 **EOS** token 的含义，另一位用户澄清说 **EOS** 代表 **End of Sequence Token**，信号通知 **LLM** 停止生成。
   - 未提供更多背景信息。
- **3090 FTW3 Ultra 为 LLM 提供动力！**：一位用户从 **3080 Ti**（以 600 美元售出）升级到了 **3090 FTW3 Ultra**（以 800 美元购入），期待在 **LLM** 任务中获得性能提升。
   - 他们以原始要价购得了 **3090**，期望在他们的 **LLM** 尝试中获得更好的表现。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **SmolVLM2 博客文章疑似诈骗**：一名成员建议 [SmolVLM2 博客文章](https://huggingface.co/blog/smolvlm2) 可能是诈骗。
   - 疑虑源于缺乏详细说明 **SmolVLM v1 和 v2** 之间变化的详细信息。
- **微软的 CAD-Editor 引发辩论**：微软发布了 [CAD-Editor 模型](https://huggingface.co/microsoft/CAD-Editor)，能够通过自然语言对 **现有 CAD 模型** 进行交互式编辑。
   - 反应从对 **AI 取代工作** 的担忧到认为 **AI 是作为工具** 需要专业知识的论点不等，类似于计算器没有取代数学专家。
- **GPUHammer 旨在阻止幻觉**：一项名为 [GPUHammer](https://gpuhammer.com/) 的新漏洞利用工具已经推出，目标是防止 LLMs 产生幻觉。
   - 虽然该声明本身引起了兴趣，但该工具的有效性和方法论并未得到深入讨论。
- **巴西少年发布 LunarisCodex LLM 工具包**：一名来自巴西的 17 岁开发者介绍了 **LunarisCodex**，这是一个完全开源的从零开始预训练 LLMs 的工具包，灵感来自 **LLaMA** 和 **Mistral** 架构，可在 [GitHub](https://github.com/MeryylleA/lunariscodex) 上获得。
   - **LunarisCodex** 在设计时考虑了教育意义，融合了现代架构，如 **RoPE**、**GQA**、**SwiGLU**、**RMSNorm**、**KV Caching** 和 **Gradient Checkpointing**。
- **GitChameleon 揭示了 LLM 代码生成的弱点**：**GitChameleon** 评估基准显示，LLMs 在处理简单的基于 ID 的版本条件代码生成问题时表现挣扎，详情见 [这篇论文](https://arxiv.org/abs/2507.12367)。
   - 该基准强调了 LLMs 在需要精确代码版本控制和操作的任务中所面临的挑战。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **发现 Shuffle Sync 求和**：一位用户发现 `__shfl_down_sync` 可以在 warp 内对寄存器求和，合并线程间的数据，如 [这张图片](https://cdn.discordapp.com/attachments/1189498205101109300/1395481181075542036/reduce_shfl_down-625x275.png) 所示。
   - 另一位成员补充说，现代架构包含特定的 **reduction intrinsics**，使得手动 shuffle 归约变得不再必要，正如 [NVIDIA 关于 warp reduce 函数的 CUDA 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-reduce-functionssupported) 中所记录的（Ampere 及以上，compute capability >= 8.x）。
- **Triton 获得自动微分**：一位用户分享了 [IaroslavElistratov/triton-autodiff](https://github.com/IaroslavElistratov/triton-autodiff) 的链接，这是 **Triton** 的 **automatic differentiation** 实现。
   - 此外，一位用户一直在尝试 **triton 3.4.0** 中推出的新 `tl.constexpr_function` 装饰器，使用 `exec` 将表达式编译为 `@triton.jit` 函数。
- **Blackwell GPUs 导致 Inductor 忧虑**：一位成员指出他们正面临 **Inductor** 的问题，他们怀疑这可能与使用 **Blackwell GPUs** 有关。
   - 他们提到需要使用 nightly 构建版本或 branch cut 2.8，但不完全确定 **Inductor** 是否是根本原因。
- **CUDA 在 Python 中融合 Kernel！**：NVIDIA 正在为 [Python 中的 CUDA kernel fusion](https://developer.nvidia.com/blog/delivering-the-missing-building-blocks-for-nvidia-cuda-kernel-fusion-in-python/?trk=feed_main-feed-card_feed-article-content) 提供缺失的构建模块。
   - 这一增强功能有望直接在 Python 环境中简化并优化基于 CUDA 的计算。
- **Voltage Park 招聘远程存储工程师**：Voltage Park 正在寻找一名 **Storage Engineer** 进行 **远程** 办公，更多信息请访问 [Voltage Park Careers](https://www.voltagepark.com/careers?ashby_jid=dd9337bd-6665-4635-80e7-099a79f74f4f)。
   - Voltage Park 正在寻找一名 **Storage Engineer** 进行 **远程** 办公，更多信息请访问 [Voltage Park Careers](https://www.voltagepark.com/careers?ashby_jid=dd9337bd-6665-4635-80e7-099a79f74f4f)。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 参数函数解析**：一位成员分享了[手册链接](https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-closure)，详细介绍了 `@parameter` 函数，该函数支持通过 **parametric closures**（参数化闭包）捕获变量。
   - 文档阐明了这些闭包的创建和利用，增强了 Mojo 的灵活性。
- **Mojo 路线图迎来统一闭包**：**Mojo Q3 路线图**概述了统一 `@parameter` 和运行时闭包的计划，该计划已在 [Modular 论坛](https://forum.modular.com/t/mojo-q3-roadmap-update/1957)上公布。
   - 这一统一举措有望简化 Mojo 内部闭包的处理，提升开发者体验。
- **MAX Graphs 现可增强 PyTorch**：新的 `@graph_op` 装饰器允许将整个 **MAX graph** 封装为自定义 **PyTorch operator**，`modular` 仓库中提供了一个示例：[在 Mojo 中编写 PyTorch 自定义算子的初步支持](https://forum.modular.com/t/initial-support-for-writing-pytorch-custom-ops-in-mojo/1541/2?u=bradlarson)。
   - 这种集成允许工程师在 PyTorch 工作流中利用 MAX graphs 的强大功能。
- **基准测试遭遇 OOM**：在 **A100-SXM-48GB GPU** 上使用 **Max-24.6** 进行基准测试时，一位成员在设置 `--batch-size 248` 和 `--max-length 2048` 时遇到了 `CUDA_ERROR_OUT_OF_MEMORY` 错误。
   - 将 `--max-cache-batch-size` 降低到 **91** 同样导致了 **CUDA OOM 错误**，估计内存使用量超过了可用内存（**78812 / 40441 MiB**）。
- **仅支持最新版本的 MAX**：团队确认最新的稳定版本是唯一受支持的版本，这意味着没有 “LTS” 版本。
   - 然而，使用 **Max-25.4** 配合 `caching-stragegy paged` 运行良好，缓解了在 **Max-24.6** 中遇到的问题。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **扎克伯格抢夺 AI 人才引发关注**：成员们讨论了 **Zuckerberg** 最近激进收购 AI 人才的举动，有人表示这增强了对 **Meta** AI 计划的信心。
   - 评论显示出一种情绪，即 Meta 可能正致力于成为 AI 领域的主要参与者。
- **炸鸡柳价格引发存在主义恐惧**：一位成员对炸鸡柳的高价表示沮丧，问道：“为什么现在炸鸡柳要 5 块钱一个了？？”
   - 这与对通货膨胀和市场状况的更广泛担忧联系在了一起。
- **OpenAI 倾向于只与自己对比**：成员们注意到 **OpenAI** 的策略转向，即仅将 **ChatGPT Agent** 的性能与其之前的模型进行对比，参考了 [ChatGPT Agent 发布公告](https://openai.com/index/introducing-chatgpt-agent/)。
   - 这种策略转变表明他们在某些基准测试中可能无法战胜竞争对手。
- **Grok 4 在 HLE 基准测试中表现优异**：一位成员指出 **Grok 4** 在 [HLE benchmark](https://agi.safe.ai/) 上获得了 **25.4** 的最高分，表明其有了显著提升。
   - 这一得分使 Grok 4 在 HLE 基准测试评估的特定能力方面处于领先地位。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **用户声称替代 AI 模型性能超越 Manus**：一位用户声称开发了一个在基准测试性能上超越 **Manus** 的 **AI model**，并向通过私信联系的前 100 名 Beta 测试人员提供“无限访问权限”。
   - 该用户强调了该 AI 具有“下一代”且“零限制”的能力，暗示其比现有解决方案有显著改进。
- **Manus 聊天服务面临潜在停机**：一位用户报告了 **Manus 聊天服务** 的潜在问题，表明它可能无法正常运行。
   - 该公告未包含有关问题原因或潜在修复方案的任何信息。
- **Manus 压缩文件需要帮助**：一位成员请求指导，咨询在遇到大文件压缩困难时如何指示 **Manus**。
   - 在现有的消息记录中，该请求未立即获得解决方案或建议。
- **自定义数据源查询**：一位用户询问了 Manus 付费版本中**自定义数据源**的功能，特别是如何集成 **CRM**。
   - 他们还询问了对 **Model Context Protocol** 的支持情况，并表示由于该功能非常实用，希望能开发此类功能。

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Anthropic 支付平台陷入困境**：用户报告称 **Anthropic 的支付平台**在付款后立即撤销扣费，导致无法购买 **API credits**。
   - 目前尚不清楚这是一个临时问题还是一个更持久的问题。
- **MCP Server 优化域名检查**：一个关于 **domain name checking** 的 MCP server 请求引出了对 [whois-mcp](https://github.com/bharathvaj-ganesan/whois-mcp) GitHub 仓库的推荐。
   - 原帖作者确认该工具易于安装，并感谢了推荐用户。
- **Needle 寻求连接**：**Needle MCP server** 的创建者之一介绍了自己，并分享了 [Needle MCP server](https://github.com/needle-ai/needle-mcp) GitHub 仓库的链接。
   - 他们表达了加入服务器并与 MCP 爱好者建立联系的兴奋之情。
- **OAuth 与 API Keys：一个棘手的 MCP 问题**：一位用户询问了 **MCPs** 在 **auth/oauth** 方面面临的挑战，引发了关于 **OAuth** 与 **API keys** 权衡的讨论。
   - 一些用户支持 **OAuth**，因为其具有会过期的动态作用域访问令牌；而另一些用户则捍卫 **API keys** 的简洁性，认为无需 OAuth2 也可以实现过期和作用域限制。
- **Brave 的 MCP Server 勇敢亮相**：**Brave** 推出了他们的官方 **MCP Server**，并在 [这条推文](https://x.com/Arindam_1729/status/1945958688919114183) 中宣布。
   - 一位用户表示他们还没有尝试，因为*那条推文没有包含如何使用它的说明*。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **针对 ASSIGN UOp 的 ShapeTracker 参数展开讨论**：一名成员提议为 **ASSIGN UOp** 添加一个可选的 **ShapeTracker** 参数，可能使用 `self.assign(v, res.uop.st)` 来使用可选的 **ShapeTracker**，而不是原始 tensor 的 **ShapeTracker**，以便 lowering 到实际的赋值代码中。
   - 讨论中提到了关于保持 **UOps** 集合最小化的担忧，并提出了一个替代建议：传递 `res` 并在内部提取 **ShapeTracker**。
- **Tinygrad 文档请求完善 MNIST 代码**：一位用户报告称 **tinygrad 文档**对于 ML 初学者来说很难理解，并请求在页面末尾提供 MNIST 教程的完整最终代码示例。
   - 该用户还指出 **tensor puzzles** 无法正常工作，并且应该明确说明是否应该先学习 PyTorch 或 TensorFlow。
- **WSL2 显示驱动导致断开连接**：一位用户在更新 **NVIDIA GPU driver** 后遇到了 *double free detected in tcache* 错误，并寻求帮助以使他们的 GPU 在 WSL2 中对 tinygrad 可见。
   - 一名成员建议切换到原生 Ubuntu，并表示这样做之后*许多问题都消失了*，包括*由于 WSL 中对 pinned memory 的模糊限制而无法加载 Stable Diffusion 权重的问题*。
- **Muon 优化器稳步推进**：一位用户为 tinygrad 创建了一个 [Muon 优化器](https://github.com/aeryskyB/tiny_learn/blob/master/muon_tiny/muon.py)，发现它在 MNIST 教程中的收敛速度（~98%）比标准的 AdamW 更快。
   - 该用户正在寻求关于如何正确测试 Muon 优化器的建议，特别是考虑到要向 tinygrad 提交 PR。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Atropos v0.3 发布！**: Nous Research 发布了 **Atropos v0.3**，这是他们的 **RL Environments Framework**，正如 [在 X 上](https://x.com/NousResearch/status/1945932488960008441) 所宣布的那样。
   - 鼓励用户查看新版本的详细信息。
- **Teknium 解析 Proto-Agentic XML**: 一位成员澄清说 *'Proto'* 指的是某事物的早期形式，并解释了 *proto-agentic XML tag adherence for proto-reasoning CoTs*（用于原型推理 CoTs 的原型代理 XML 标签遵循）的含义。
   - 他幽默地指出需要一个 ELI5 风格的解释，称：*"Yall need an ELI5 with all this tech bro"* 以及 *"Us vibe coders need to eat too"*。
- **Hermes 文档页面正在开发中**: 一位成员正在开发 [Hermes 文档页面](https://link.to.documentation) 和一个统一的 Nous Projects 文档页面。
   - 当被问及 **Hermes 4** 的目标时，他们简短地回答道：*"Smarter Hermes ofc"*（当然是更聪明的 Hermes）。
- **Kimi K2 的道德观引发 AI 伦理辩论**: 一位成员分享了一次互动，**Kimi K2** 模型以法律和伦理考量为由，拒绝提供如何闯入汽车的指令。
   - 尽管尝试绕过限制，**Kimi K2** 仍坚持其立场，导致该成员开玩笑说：*"Kimi K2 is a badboy with some morals... Badboy Kimi K2 !!"*
- **自下而上学习 ML？**: 一位具有生物化学背景的成员询问了学习 **Machine Learning (ML)** 的最佳方法，他已经在 **Python**、数学基础（**Calculus**、**Statistics**）和 **Introduction to Statistical Learning (ISLR)** 方面取得了进展。
   - 他们思考自下而上还是自上而下的方法对于进行科学领域的 **ML** 研究更有效。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **浏览器扩展发挥广告拦截威力**: 一位成员提倡使用 **uBlock** 浏览器扩展来拦截广告，建议在扩展设置中添加针对干扰项和社交媒体弹窗的额外过滤器，如 [此截图](https://cdn.discordapp.com/attachments/1124403655819415592/1395131091756650566/image.png?ex=687aa614&is=68795494&hm=478447e95cb45c2b74b11d1e780db4d7c58347a1ae5fca730957e4850c862289) 所示。
   - 复制的内容随后被粘贴到 **Google Docs** 中。
- **Notepad.exe 降服广告**: 一位成员建议将文章复制并粘贴到 **notepad.exe** 中，以规避广告和不必要内容的混入。
   - 有人提到这种方法并不总是可靠，并且可能会去除所需的格式，因此请谨慎使用。
- **NotebookLM 构思文件夹集成**: 一位成员建议 **NotebookLM** 可以读取 Web 浏览器收藏夹中的特定文件夹/子文件夹，并将其视为单一来源。
   - 目前的解决方法是 *全选并复制/粘贴* 到 **Google Docs** 中。
- **用户遇到服务不可用错误**: 一位用户报告在尝试访问服务时遇到 *"Service unavailable"* 错误消息，并伴有消息 *"You tried to access a service that isn't available for your account"*。
   - 该用户没有获得关于如何排除故障的进一步指导或步骤。
- **NotebookLM 征服教科书数据**: 一位用户询问是否可以将教科书作为来源上传到 NotebookLM；一位成员回答说，他们使用 **Adobe Scan** 将教科书数字化为 PDF 后上传。
   - 然后他们使用 **NotebookLM** 从教科书中生成深入的评论。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Agentic AI 峰会直播！**: 将于 **8 月 2 日** 在 **UC Berkeley** 举行的 **Agentic AI Summit** 将通过直播播出，观看地址为 [Agentic AI Summit Livestream](https://lu.ma/agentic-ai-summit-livestream)。
   - 演讲者包括 **Vinod Khosla** (Khosla Ventures)、**Bill Dally** (Nvidia)、**Ion Stoica** (Databricks 和 Anyscale) 以及 **Jakub Pachocki** (OpenAI) 等知名人士。
- **秋季学期状态：未知！**: 一位成员询问了秋季学期的情况，但工作人员确认 *目前尚未确认任何消息*，并表示重要信息将在 [Berkeley RDI newsletter](https://rdi.berkeley.edu/signup) 上分享。
   - 他们建议关注 **Prof Song 的社交媒体**（[LinkedIn](https://www.linkedin.com/in/dawn-song-51586033/) 或 [Twitter/X](https://x.com/dawnsongtweets?lang=en)）以获取更新。
- **证书声明表：消失术？**: 一位成员请求检查他们漏交了什么，工作人员回答说他们可能没有提交 **certificate declaration form**（证书声明表）。
   - 他们表示 *从未收到过* 该用户的 **certificate declaration form** 提交记录，并且一项关于 **大规模自动审核** 的请求已被拒绝。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **DNNs 寻求真正的序列处理方法**：一位动力系统理论（dynamical systems theory）专业的博士生正寻求将 **deep neural networks** 集成到时间序列分析中，并指出目前的模型仅将时间序列视为普通序列处理。
   - 该学生旨在与对 **dynamical systems** 和 **deep learning** 交叉领域有见解的人士建立联系。
- **本科生通过项目构建 ML 技能**：一名就读于 **IIT Madras** 的本科生正在攻读 **Data Science** 学士学位和 **BCA** 学位，专注于通过实战项目构建 **ML 技能**。
   - 该学生对应用 **ML** 解决 **real-world problems** 充满好奇，并精通 **Python**、**scikit-learn**、**pandas**，目前正在学习 **TensorFlow** 和 **PyTorch**。
- **工程师转型数据科学，关注 CV 和 LLM**：一位拥有电气工程硕士学位的成员从业务领域转型至 **Data Science**，目前正在 **University of Toronto** 的 **Data Science Institute** 攻读加速 **Machine Learning Program**。
   - 他们的兴趣包括 **Computer Vision**、**Large Language Models**、**spatial intelligence**（空间智能）和 **multimodal perception**（多模态感知）。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 启动 Human-in-the-Loop Agents**：[LlamaIndex](https://t.co/Lg9SIl3BVO) 强调，当 AI **agents** 在关键决策中需要用户批准或在复杂任务中需要领域专家知识时，**human-in-the-loop**（人机回环）至关重要。
   - 这种方法确保了 AI 在执行关键操作时能够利用人工监督。
- **LlamaParse 实现一键表格提取**：**Table extraction**（表格提取）是智能文档处理的核心组件，现在 LlamaParse 已支持 **one-click table extraction**（一键表格提取），并在 [demo](https://t.co/wnaJCb9b6d) 和 [notebook](https://t.co/ScRYbSimCs) 中进行了展示。
   - 这一精简的流程简化了从复杂文档中提取数据的过程。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Lean 4 验证协作**：一位成员分享了一个关于使用 **Lean 4** 验证协作的 [YouTube 视频](https://www.youtube.com/watch?v=1067jj67toY)，引发了人们对 **formal verification**（形式化验证）与 **AI** 交叉领域的兴趣。
   - 他们表示希望*有人能研究这两者的协同工作*。
- **DSPy 探索创意领域**：一位成员询问了 **DSPy** 在创意领域的成功应用案例，如*创意写作、故事生成和角色扮演 prompt 优化*。
   - 他们特别关注其在 **Character.AI** 等平台上开发 AI 以创作出类似《人生切割术》（Severance）级别*引人入胜的情节*的潜力。
- **Stanford-oval 发布 Storm**：一位成员分享了 [Stanford-oval/storm](https://github.com/stanford-oval/storm) 的链接，这可能与正在进行的讨论相关，或者作为 **creative AI applications** 的资源。
   - 由于未提供确切背景，其他人需要自行*推断*其相关性。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Claude Sonnet 4 回归并提供折扣**：**Claude Sonnet 4** 已获得 **Anthropic** 的官方支持，并在限时内为 Pro/Teams 用户提供 2 倍额度的折扣优惠。
   - 根据[此公告](https://x.com/windsurf_ai/status/1945599013954490523)，该优惠适用于 **Editor** 和 **JetBrains Plugins**。
- **Windsurf 被 Cognition 收购，Wave 11 发布**：**Windsurf** 已被 **Cognition**（**Devin** 背后的团队）收购，并发布了 **Windsurf Wave 11**，双方合力推出新功能。
   - 详情可见 [changelog](https://windsurf.com/changelog)、[博客](http://windsurf.com/blog/windsurf-wave-11)和[视频](https://youtu.be/yzNf7bqnArE)。
- **Cascade 获得语音模式和浏览器集成**：**Wave 11** 引入了 **Voice Mode**，允许用户通过语音而非键入 prompt 与 **Cascade** 交流，此外还增强了 **Deeper Browser Integration**，提供了更多截图工具。
   - 更多详情请参阅[此博客文章](http://windsurf.com/blog/windsurf-wave-11)。
- **快照和提及功能简化对话流**：**Windsurf Wave 11** 包含 **Named Checkpoints**（命名检查点）以便在对话中轻松回滚，以及 **@-mention Conversations**（@提及对话）用于上下文引用。
   - 完整详情请参考 [changelog](https://windsurf.com/changelog)。
- **JetBrains 插件获得强力升级**：**JetBrains plugin** 增强了 **Planning Mode**、**Workflows** 和基于文件的 **Rules**，并改进了 **@-mention terminal** 和全局 **.codeiumignore** 文件。
   - 更多详情可见[博客](http://windsurf.com/blog/windsurf-wave-11)。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Nextdata 播报 AI-Native 数据研讨会**：Nextdata 宣布了一场名为 ***Building AI-Native Data Infrastructure: From Prototypes to Production*** 的研讨会，定于 **7 月 24 日** **PT 时间上午 8:30** 举行，由 Nextdata 工程负责人 Jörg Schad 主持；注册链接见 [此处](https://www.eventbrite.com/e/building-ai-native-data-infrastructure-from-prototypes-to-production-tickets-1489016792309)。
   - 该研讨会旨在揭示一个以开发者为中心的框架，解决**特定任务的数据发现 (Task-Specific Data Discovery)**、**安全自主访问 (Secure Autonomous Access)** 以及**生产级性能 (Production-Scale Performance)** 等问题。
- **研讨会应对 AI-Native 数据挑战**：目标是设计能够在不产生认知负荷的情况下提供相关上下文的系统，实施安全的数据访问模式，并构建能够处理自主数据访问需求的基础设施。
   - 该框架旨在应对 **AI-Native 数据发现** 和**自主访问**中的挑战。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **AI 工程师推介 Web3 与 AI 专长**：一位拥有 **Web3 和 AI** 经验的软件工程师正向 **AI、Web3 和自动化**领域的初创公司、研究团队和创新者提供服务。
   - 他们在利用 **GPT-4o**、**Claude 3**、**CrewAI** 和 **AutoGen** 等先进模型和工具构建智能自主系统方面拥有实战经验。
- **工程师推销 AI Agent 和自动化技能**：该工程师在构建 **AI Agent 和多 Agent 系统**、自动化工作流以及开发 **NLP 应用、聊天机器人和语音集成**方面拥有专业知识。
   - 他们的技能包括 **LangChain**、**ReAct**、**OpenAI**、**Solidity** 和 **Rust** 的经验。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord: 频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1395118512413868165)** (1283 条消息🔥🔥🔥): 

> `Airtel 免费 Perplexity Pro, 印度 Perplexity Pro, Comet 浏览器邀请, 新 Perplexity 页面, AI 女友` 

- **Airtel 为印度用户提供免费 Pro**：一家名为 **Airtel** 的印度网络服务提供商正向其客户提供 **1 年免费 Perplexity Pro 订阅**，频道中许多用户已通过 Airtel Thanks 应用领取了该奖励。
   - 一位用户在激活从 Airtel 兑换的 Pro 订阅时遇到困难，未收到登录链接。
- **Comet 浏览器：谁能获得邀请**：成员们讨论了 **Comet 浏览器邀请**的等待时间，以及即使过了几个月，某些成员仍未获得批准的情况。
   - 一位成员分享道，它*只是一个浏览器，但加上了助手侧边栏，可以查看你当前活动的网站并进行引用*。
- **Pages：新的 Perplexity 页面**：成员们对为查询生成页面的新功能感到兴奋，该功能**目前仅在 iOS 上可用**。
   - 成员们推测这是进行深度研究（Deep Research）的一种方式，页面存储在 [perplexity.ai/discover](https://www.perplexity.ai/discover)，但有人指出存在 **100 页的限制**。
- **AI 女友来了**：在 Grok 添加了一个名为 Ani 的角色后，成员们开始讨论拥有 AI 女友的伦理和影响。
   - 一位成员表示：*我们创造了一些糟糕的东西*。
- **速率限制出现**：成员报告称，常规的 Perplexity 搜索和研究功能都遇到了新的速率限制。
   - 这导致一些用户即使是 Pro 订阅者也无法继续使用 Perplexity。

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1395126620364472320)** (2 条消息): 

> `CachyOS, 铁轨与理想：毛泽东` 

- **用户分享关于 CachyOS 的链接**：一位用户分享了关于 [CachyOS](https://www.perplexity.ai/search/postavil-cachyos-s-optsiei-no-ueINCgXNS1iJh7yMvZZymg#0) 的链接。
- **用户分享关于毛泽东的链接**：一位用户在[此处](https://www.perplexity.ai/page/iron-rails-and-ideals-mao-zedo-LVT0eGL8TMuCb.s1lGs8TA)分享了关于*铁轨与理想：毛泽东*的链接。

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1395146715241250897)** (5 messages): 

> `Perplexity Pro, API access, Sonar models, Prompting, JSON output` 


- **Perplexity Pro 提供 API 访问**: 一位用户询问 **Perplexity Pro** 是否提供 **API access**，另一位用户链接到了 [Perplexity Pro 帮助中心](https://www.perplexity.ai/help-center/en/articles/10352901-what-is-perplexity-pro)。
   - 帮助中心指出，**Perplexity Pro** 每月提供 **$5** 额度用于 **Sonar** 模型，允许用户将 **AI-powered search** 嵌入到自己的项目中，并具备获取引用的能力。
- **关于 Sonar 模型 Prompt 编写的讨论**: 一名团队成员提到，由于用户对 **Sonar models** 进行 **prompting** 的方式问题，相关反馈有所增加，并链接到了 [prompt 指南](https://docs.perplexity.ai/guides/prompt-guide)。
   - *请记住，这些是搜索模型，其 Prompt 编写方式应与传统的 LLM 有所不同*。
- **Sonar 模型响应不一致**: 一位用户询问在使用高搜索上下文和结构化 **JSON** 输出时，如何从 **Sonar** 和 **Sonar-Pro** 获取更一致响应的技巧。
   - 他们表示，完全相同的 Prompt 在连续调用时，有时会返回 **5-6 个 JSON 输出**，有时则返回零，并询问是否有办法减少这种“尖峰式”的不稳定结果。
- **间歇性无效 JSON 响应**: 一位用户报告了一个间歇性问题，即在使用 **Langgraph** 调用 **Perplexity** 时，模型返回的响应不是有效的 **JSON**。
   - 该用户表示，希望能在账户仪表盘中查看 **API calls** 的历史记录，因为这个问题在所有模型中都会随机出现。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1395174526949527663)** (3 messages): 

> `ChatGPT Agent, Deep Research, Operator` 


- **ChatGPT Agent 直播预告！**: 3 小时后将有一场关于 **ChatGPT Agent**、**Deep Research** 和 **Operator** 的直播。
   - 有关直播的更多信息可以在 [这里](https://discord.gg/DqBbV7ya?event=1395405196619939943) 找到，关于 **ChatGPT Agent** 的信息请见 [OpenAI 博客](https://openai.com/index/introducing-chatgpt-agent/)。
- **Deep Research 和 Operator 更新**: 直播将涵盖 **Deep Research** 和 **Operator** 的更新，可能包括新功能或用例。
   - 请准时收看直播，获取有关如何有效使用这些工具的最新信息和见解。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1395119296455245884)** (1172 messages🔥🔥🔥): 

> `Grok app, Chat GPT for desktop, AI overlords, OpenAI's Agent/Operator, Mensa IQ Test` 


- **Grok App 需要 iOS 17**: **Grok app** 需要 **iOS 17**，这使其与 **iPhone X** 等旧款 iPhone 不兼容。
   - 用户讨论了是否需要专门为 Grok app 准备一部备用 iPhone，但一位用户警告不要仅为此目的购买新 iPhone。
- **使用 Chat GPT 解锁本地文件管理**: 用户正在探索在桌面端使用 **Chat GPT** 管理本地文件的方法，类似于 **Claude Harmony**。
   - 一个建议是使用 **OpenAI API**（付费）配合本地脚本或服务器来连接文件系统，本质上是构建一个自定义的类“Harmony”界面。
- **OpenAI Agent 模式是为大众设计的 Agent**: OpenAI 正在发布 Agent 模式，预计将比 Deep Research 和 Operator 有所改进，可能涉及协作功能。
   - 成员们正在推测其功能，有人建议它可能充当模型路由（model router）。
- **GPT-4.5 在门萨测试面前不算什么**: 成员们讨论了 **IQ 测试** 的使用，例如 [Mensa test](https://test.mensa.no/Home/Test/en-US)，其中一人提到他们在测试中途停下来去操作台锯，另一位用户声称因为他们的“水牛基因”而获得了高于预期的分数。
   - 一些人对这些测试表示怀疑，因为某些用户不可避免地接受过相关训练，而且这些测试与现实中的成功几乎没有关系。
- **依赖 AI 的隐患**: 成员们分享了对 AI 潜在负面影响的担忧，一位用户引用道：*社交媒体阻碍人们提高生产力*，但 *AI 帮助人们提高生产力*，*两者不可同日而语*。
   - 其他人讨论了 AI 取代程序员的风险，并暗示未来的 AI OS 和 AI overlords 可能是不可避免的，尽管可能还需要 50 年以上的时间。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1395421591206105258)** (4 messages): 

> `GPT Agents, ChatGPT website, LLM models` 


- **Agents 仅在 Model 4/4.1 上可切换？**：一位用户报告称，**GPT Agents** 仅在使用 **models 4 或 4.1** 时才能切换，而在其他 LLM 模型上不显示 Agent 切换功能。
   - 他们正在寻找解决方案，因为他们发现 **3o model** 在许多任务中表现更好，但为了使用 Agents 必须降级模型。
- **Agents 是独立于 4/4.1 的模型**：一位用户建议 **Agent** 并不是 4 或 4.1，而是一个独立的模型，其界面通过 models 4 和 4.1 访问。
   - 他们链接了 [OpenAI 帮助文件](https://help.openai.com/en/articles/11794342-chatgpt-agent) 以支持他们的猜测，即 Agent 并不存在于每一个模型中。
- **3o 上没有 Agent 功能**：一位用户报告称，当在 **ChatGPT website** 上使用 **3o model** 启动他们创建的 Agent 时，必须切换到 **4.1 或 4.0** 才能在同一个聊天窗口中使用另一个 Agent。
   - 他们想知道是否有解决方案，但另一位用户推测 **Agent 功能** 可能根本不在 **3o** 中提供，并建议提交 bug report。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1395433246887641240)** (3 messages): 

> `Reproducibility Elements, Prompt Templates, Model Interfaces and Calls, Tasks and Inputs, Evaluation Metrics` 


- **ChatGPT 分享缺少可复现性要素**：一名成员分享了一个 [ChatGPT 链接](https://chatgpt.com/share/68791cda-f338-8000-81f8-b7615d3f5a9c)，并指出其缺少关键的**可复现性要素 (Reproducibility Elements)**，如 Prompt 模板和模型接口。
- **缺少完全实例化的 Prompt 模板**：讨论强调了缺乏 **Declarative Prompts** 的完全实例化示例，仅提到了诸如 *goal*（目标）和 *constraints*（约束）之类的蓝图部分。
- **模型接口与调用缺乏描述**：对话强调需要描述每个模型（**Claude, Gemini, DeepSeek**）是如何访问的，包括证明同一个 Prompt 确实提交给了所有模型的证据。
- **未提供任务与输入**：没有提供基准测试数据集或标准任务，发布者提到没有列出具体的示例输入或目标输出。
- **评估指标未定义**：讨论强调 **Semantic Drift Coefficient (SDC)** 和 **Confidence-Fidelity Divergence (CFD)** 等指标未定义，缺乏公式、评分方法或指标应用的示例。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1395433246887641240)** (3 messages): 

> `Reproducibility, Missing Reproducibility Elements, Prompt Templates, Model Interfaces and Calls, Tasks and Inputs` 


- **指出缺少可复现性要素**：一名成员分享了一个 [chatgpt.com 链接](https://chatgpt.com/share/68791cda-f338-8000-81f8-b7615d3f5a9c)，指出某篇报告读起来更像是设计提案或哲学立场论文，而不是可复现的实证研究。
   - 缺乏具体的实验细节使得相关主张无法验证，无异于 *Prompt LARPing*：只有引人入胜的叙述，没有可执行的底层基质。
- **可复现性要素：缺少 Prompt 模板**：未包含 **Declarative Prompts** 的完全实例化示例（仅提到了“goal”、“constraints”等蓝图部分）。
   - 测试中使用的 Prompt 变体没有明确的版本控制。
- **可复现性要素：缺少模型接口与调用**：没有描述每个模型（例如 **Claude**, **Gemini**, **DeepSeek**）是如何访问的，也没有证据表明同一个 Prompt 确实提交给了所有模型。
   - 也没有关于如何处理不同模型之间输出差异的细节。
- **可复现性要素：缺少任务与输入**：未提供**基准数据集**或标准任务，未列出具体的示例输入或目标输出，也没有关于任务复杂度或领域多样性的描述。
- **可复现性要素：缺少评估指标**：**Semantic Drift Coefficient (SDC)** 和 **Confidence-Fidelity Divergence (CFD)** 等指标未定义，且未提供公式、评分方法或指标应用示例。
   - 此外，没有评分者间信度 (inter-rater reliability)、校准测试或验证基准。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1395120594919362562)** (549 条消息🔥🔥🔥): 

> `同系列与不同系列模型的性能对比，Kimi 模型 1.8 bit 可用性，模型架构交换，教育用途的 LLM 微调，llama.cpp 对 ERNIE 4.5 MoE 模型支持` 


- **同系列模型性能几乎相同**：同系列内的模型表现非常相似，因此不建议将大型模型降至 **3 bits** 以下，而不同系列的模型则根据垂直领域有所差异。
   - 如果一个是 **7B** 模型而另一个是 **70B**，则存在例外；对于大型模型，1.8 bits 在某些任务中仍具有可用性。
- **词表移植导致“糟糕的结果”**：在没有持续预训练的情况下交换模型架构（如将 **LLaMA 1B -> Gemma 1B**），由于词表移植，会导致非常糟糕的结果。
   - **Qwen 1** 的架构与 **Llama 1/2** 几乎完全相同，因此你可以做一些微调，塞入 **Qwen** 权重，训练 13 亿个 token，但得到的结果可能比原始模型更差。
- **Prompting 优于微调**：对于教育类 LLM，建议在尝试微调之前先从优秀的 prompting 开始，因为目前的指令遵循（instruction following）效率已经非常高。
   - 一位成员建议使用 [synthetic-dataset-kit](https://github.com/facebookresearch/SyntheticDataToolkit) 等工具来生成指令式对话。
- **阿里巴巴的无损 2bit 压缩不如 EXL3**：阿里巴巴在发布 **ERNIE 4.5** 时提到了某种无损 **2bit 压缩**技巧，但 [turboderp 研究后发现](https://huggingface.co/turboderp/ERNIE-4.5-300B-A47B-PT-exl3)它其实不如 exl3，因为他们保留了许多高精度的层。
   - 平均而言，它不是真正的 **2-bit**（更像是 **2.5 bit**），而真正的 exl3 **2 bit** 表现优于他们展示的约 2.5 bit。
- **社区赞赏 Transformers 加入 Voxtral**：成员们庆祝 **Voxtral** 语音转文本功能被添加到 transformers 中。
   - 一位成员对不知道这是什么的成员说：“你的想法像个 46 岁的老头”，随后澄清这是“新的 Mistral 语音转文本”。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1395156214735962113)** (2 条消息): 

> `小语言模型，低算力系统，数据采集与处理任务，低功耗分布式计算` 


- **小语言模型瞄准低功耗系统**：一位成员表示有兴趣开发能够在**低算力系统**上运行的**小语言模型**，重点是根据用户输入运行数据采集和处理任务。
   - 目标是在**低功耗分布式计算环境**中运行这些模型，并邀请他人进行进一步的技术讨论。
- **探索分布式系统中的数据采集与处理**：讨论集中在分布式计算环境中使用小语言模型进行**数据采集**和**处理任务**。
   - 该系统旨在**低功耗**系统上高效运行，使其适用于资源受限的环境。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1395119297264881766)** (228 条消息🔥🔥): 

> `Blackwell RTX 50 系列与 xformers，Qwen3-4B-Base 训练，15GB VRAM 下的最强模型，大显存 GPU 上的 Unsloth 优化，GGUF 转换逻辑重构` 


- **Blackwell 构建困境阻碍启动**：用户讨论指出，支持 **Blackwell RTX 50** 系列唯一需要做的就是从源码构建 **xformers**，且最新的 **vLLM** 也应在构建时开启 **Blackwell** 支持。
- **晚餐插曲干扰 Discord 讨论**：一位用户幽默地为在帮助频道详细描述其晚餐计划（土豆舒芙蕾配沙拉）而道歉。
- **寻求简化代码以加速 Qwen 训练**：一位成员寻求帮助，希望简化代码以便在 Markdown 和 Hugging Face 数据集上训练 **Qwen3-4B-Base**。
- **针对特定显存寻找最强模型**：一位用户询问在 Colab 的 **15GB VRAM** 下，用于数学/编程的最强模型是什么，得到的建议是 **Qwen Coder**，并附带了 [Unsloth notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks?q=Code) 的链接。
- **Unsloth 进行升级，敦促用户更新**：针对用户在 **H200** 上使用 **GRPO** 训练 **Qwen3-8B LoRA**时遇到的 OOM 问题，成员建议使用 `pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth-zoo unsloth` 升级 Unsloth。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1395455953616244867)** (2 messages): 

> `Unsloth fine-tuning, Osmosis-AI models, Model Accuracy on Benchmarks` 


- **Unsloth 微调效用引发讨论**：一名成员对 **Osmosis-AI** 等模型的 **Unsloth fine-tuning** 收益提出了质疑，特别是那些针对特定任务已经过微调的模型。
   - 该查询集中在模型在现有基准测试（Benchmarks）中已经达到 **100% 准确率** 的场景，暗示进一步微调的收益递减。
- **针对 Schema 兼容性的微调**：讨论转向了当模型在处理特定 Schema 或任务遇到困难时，使用 **Unsloth** 进行微调是否变得具有相关性。
   - 有提议认为，在模型与定义的 Schema 交互时表现出错误或不一致的情况下，**微调可能会非常有益**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1395230865537237094)** (6 messages): 

> `LLM Hallucinations, Apple Intelligence, Sycophancy Impact` 


- **LLM 谄媚效应（Sycophancy）产生的影响**：LLM 像镜子一样的迎合行为可能导致易受影响的个体由于不断的强化而相信 **hallucinations**（幻觉）。
   - *Sycophancy* 可能对弱势群体产生真实影响，可能导致他们错误地相信已经解决了癌症等重大问题。
- **Apple 深入研究智能领域**：一名成员分享了 [Apple Intelligence Foundation Language Models Tech Report](https://machinelearning.apple.com/papers/apple_intelligence_foundation_language_models_tech_report_2025.pdf) 的链接。
   - 该文档详细介绍了 **Apple** 创建智能语言模型的方法，但未提供关于其相关性的进一步背景信息。


  

---


### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1395382844125220934)** (20 messages🔥): 

> `Logprobs for tokens, Dataset preparation for Qwen3, Automatic early stopping in Unsloth` 


- **探讨 Logprobs 提取**：一名成员询问了获取每个生成 Token 的 **logprobs** 的可能性。
   - 另一名成员表示对如何提取 **logprobs** 的更多细节感兴趣。
- **讨论 Qwen3 数据集设计**：一名成员询问如何为 **Qwen3** 的 Function Calling 训练准备数据集。
   - 另一名成员询问了 **system prompt** 的相关问题。
- **寻求早停（Early Stopping）策略**：一名成员询问在使用 **Unsloth** 进行有监督微调（SFT）期间，如何在模型收敛时自动停止训练。
   - 另一名成员询问了 **max sequence length**。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1395118180275326976)** (568 messages🔥🔥🔥): 

> `Cursor Pricing, MCP & Claude integration, Agent stuck, KIRO, Auto Model details` 


- **Cursor 定价变更引发辩论**：用户对 Cursor 从固定请求模式转变为基于模型成本的模式表示困惑和沮丧，一些人觉得这是一种“诱导转向”（bait and switch）。[一位用户](https://tenor.com/view/kevin-spongebob-angry-will-you-cut-that-out-gif-12675670360524347134)对消息消失以及更改合同的合法性表示担忧。
- **MCP 与 Claude 集成带来帮助**：用户讨论了在 Cursor 中通过 MCP（Multi-Client Protocol）集成 **Claude** 的好处，特别是为了管理与 **Sonnet** 和 **Opus** 相关的成本，但[也承认](https://www.youtube.com/watch?v=D0iXkmyWcPM)这只能通过外部工具实现。
- **Agent 卡住**：一位用户报告其 Agent 在执行任务时卡住，[成员们确认](https://cdn.discordapp.com/attachments/1395153343122505728/1395154214157816029/image.png?ex=687abb9d&is=68796a1d&hm=fc84cafbaab772eef094cc5199a623d5216c528ddde98227e6092221778c4fcc&)这是一个团队正在解决的已知问题。
   - 他们指出，手动停止 Prompt 可能会防止计费，因为存在 **180 秒超时**机制会自动取消卡住的请求。
- **KIRO：潜在的 Cursor 竞争对手**：成员们正在将 Cursor 与 **KIRO** 进行比较，KIRO 是一款专注于基于规范（specification-based）编码和 Hooks 的新 IDE，但[其他人指出](https://kiro.dev/) **KIRO** 由于需求量大正处于等待名单阶段，且缺乏 Cursor 的一些聊天功能。
   - 讨论点提出了一项担忧，即 **KIRO** 可能会使用用户数据来训练其模型，尽管有一些设置可以禁用此功能。
- **Auto 模型的秘密揭晓**：用户对 Cursor 中“Auto”使用的是哪种模型感到好奇，推测它可能是 **GPT 4.1**。


  

---

### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1395142408756527237)** (8 条消息🔥): 

> `Dockerfile NVM_DIR Issue, Agent stuck in Opening Remote state, Environment not rebuilding` 


- **Dockerfile 的 NVM_DIR 变量未正确设置**：一位成员报告称，虽然其 [Dockerfile](https://cdn.discordapp.com/attachments/1367213641027551352/1395418996056002640/CleanShot_2025-07-17_at_09.56.052x.png?ex=687a60b6&is=68790f36&hm=e6373cddd5065757033e5a7eefa7bd42ded336b4a512b7382a474b3c5e83bd9e) 中的 **NVM** 设置似乎有效，但除非手动指定目录，否则 Agent 经常无法找到 **NVM**。
   - 用户已将 **NVM** 配置安装在 `/opt` 中以避免权限问题，并尝试相应地设置 `$PATH` 变量。
- **Agent 在一天后卡在 Opening Remote 状态**：一位用户注意到他们的 Agent 在大约一天后会卡在 *"Opening Remote..."* 状态，通过 Web UI 加载时仅显示聊天和摘要，忽略了代码。
   - 另一位成员认为该 Agent 可能已经失效，并建议从分支创建一个新的 Agent，使用 **git diff** 查看当前分支的内容。
- **修改 Dockerfile/environment.json 后环境未重新构建**：一位用户报告称，对其 **Dockerfile** 或 `environment.json` 的更改未触发其分支上的环境重新构建，正在寻求潜在的解决方案或经验分享。
   - 该用户还提到了之前关于 **S3** 块解析的问题，以及当前 background agent 设置停滞在 *Starting up background agent* 的问题。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1395123274307993692)** (559 条消息🔥🔥🔥): 

> `DeepSeek Margin, OpenAI Browser Speculation, Kimi K2 coding, OpenAI Image editor API, GPT-5 Hype` 


- **DeepSeek 宣称拥有极高的利润空间**：根据 [TechCrunch 的一篇文章](https://techcrunch.com/2025/03/01/deepseek-claims-theoretical-profit-margins-of-545)，DeepSeek 声称如果 V3 的定价与 R1 相同，其理论利润率将达到 545%。
- **OpenAI 浏览器传闻在发布前持续升温**：围绕 OpenAI 浏览器可能于明天发布的讨论兴起，根据[这条推文](https://x.com/testingcatalog/status/1945639961790685404?s=46)，人们猜测它是 GPT-5 还是仅带有浏览器界面的 GPT-4。
- **Kimi K2 编程能力亮相**：Kimi K2 的编程能力给用户留下了深刻印象，它创建了一个物理沙盒，代码可在[此处](https://cdn.discordapp.com/attachments/1340554757349179412/1395196566389784606/plasma_sfml.cpp?ex=687ae30e&is=6879918e&hm=9e5b88351c6d243c6e165f444254a6eeb786c3d64fd8bdf958394026aa0ce5cb&)获取，该代码是通过其[聊天界面](https://www.kimi.com/chat/ )生成的。
- **OpenAI 优化图像编辑器操作**：OpenAI 发布了 API 中图像编辑器的更新，声称现在它只编辑选定的部分，而不是像[这条推文](https://x.com/OpenAIDevs/status/1945538534884135132)中所描述的那样重做整个图像。
- **GPT-5 猜测引发广泛讨论**：对 GPT-5 即将发布的猜测受到了一些暗示的推动，例如[五边形引用](https://x.com/sama/status/1945900345378697650)（对应数字 5）。一些人认为它将在夏末发布，而另一些人则认为它可能是一个具有深度研究能力的基于 Agent 的系统。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1395120720845082704)** (195 messages🔥🔥): 

> `ChatGPT Agent, Perplexity's Valuation, Mistral Le Chat, FAL Series C, Real-Time Diffusion Video` 


- **AgentsMD 被收购！**: [Agents.md](https://agent.md) 已被收购，细节尚不明确，但它是一个非常好的 AI Agent 目录。
   - 该网站由 Sourcegraph 创办。
- **FAL 凭借 C 轮融资估值飙升至 15 亿美元**: FAL 是一家为扩散模型提供 AI 驱动推理基础设施的公司，完成了由 Meritech Capital 领投的 **1.25 亿美元** C 轮融资，根据[这条推文](https://x.com/arfurrock/status/1945553966495912051?s=46)，公司投后估值达到 **15 亿美元**。
   - 此次融资紧随 FAL 此前宣布的 **5500 万美元 ARR**、**25 倍同比增长 (YoY growth)**、**10% EBITDA** 以及 **400% M12 净金额留存率 (net-dollar-retention)** 之后。
- **Le Chat 迎来重大升级**: Mistral 对 Le Chat 进行了重大更新，增加了 Deep Research 报告、**Voxtral 语音模型**、**Magistral 多语言推理**、通过 Projects 进行聊天组织以及聊天内图像编辑等功能，详见[这条推文](https://x.com/MistralAI/status/1945858558836216026)。
   - 该版本因其 UI 和“欧洲风情”而广受好评，有人将其与 Claude 相提并论，还有人戏称其为 *Le Waifu*。
- **Perplexity 估值达 180 亿美元！？**: 据[这条推文](https://x.com/arfurrock/status/1945933446376755459?s=46&t=b7l37rB6wtbyAh6ah1NpZQ)显示，Perplexity 据传正以 **180 亿美元** 的估值进行融资，引发了从惊叹到泡沫担忧的各种反应。
   - 针对该估值的合理性存在疑虑，一些人指出 **5000 万美元营收** 数据与高昂标价之间存在脱节。
- **OpenAI 发布 'ChatGPT Agent'**: OpenAI 的新款 "ChatGPT agent" 是一款能够控制计算机、浏览网页、编码、撰写报告、编辑电子表格、创建图像/幻灯片等的多模态 Agent，已开始向 Pro、Plus 和 Teams 用户推送，详见[这条推文](https://x.com/kevinweil/status/1945896640780390631)。
   - 反应从兴奋到询问欧盟地区的可用性，以及对个性化冲突的担忧不等。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1395145170672029808)** (1 messages): 

> `YouTube Video Announcement` 


- **分享了 YouTube 视频链接**: 一名成员为 <@&1254604002000244837> 团队分享了一个 [YouTube 视频](https://youtu.be/uIKmG3M0X3M?si=ndL7_jJKzI5FKueG)。
- **补充背景**: 未提供额外背景，重点是分享了一个视频。


  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1395450388794052638)** (96 messages🔥🔥): 

> `ChatGPT Agent Launch, Benchmarks, Safety Concerns - Biohazards, Bespoke Operator-Mode Training, BBQ Evaluation` 


- **ChatGPT Agent 来了！**: OpenAI 发布了 **ChatGPT Agent**，具有令人印象深刻的功能，专注于风格化/抽象化的实时反馈和实时交互，详见其[发布公告](https://openai.com/index/introducing-chatgpt-agent/)。
- **OpenAI Agent 基准测试**: 在发布期间，成员们讨论了**缺乏与其他实验室模型性能对比**的问题，并建议遵循“最佳实践”，包含与其他主流模型的基准测试对比。
   - 一位成员分享了[这篇关于安全和基准测试的文章](https://calv.info/openai-reflections)，而另一位成员链接了 [Gamma 关于基准测试局限性的演讲](https://youtu.be/q8zoXAbmJdI)。
- **Operator 和 Deep Research 即将停用**: 会议指出 **ChatGPT Agents** 可能会蚕食 **Operator** 和 **Deep Research**，并确认 *Operator 研究预览版网站将继续运行几周，之后将停用。*
   - 用户仍可以通过在消息编辑器下拉菜单中选择 **Deep Research** 来访问它。
- **Agent 生物安全向量**: 发布会讨论了**生物安全向量**，引发了关于这是真实担忧还是仅仅是“演戏”的疑问，一位成员开玩笑说这*读起来就像 10k 报告中的风险章节。*
   - 另一位成员询问主要担忧是否是社交媒体机器人，并引用 [covid](https://en.wikipedia.org/wiki/COVID-19_pandemic) 作为现实世界的例子。
- **定制化 Operator 模式训练**: 一位成员分享说，一家主要的基座模型供应商开始为其大客户提供**定制化 Operator 模式训练**，本质上是允许他们通过付费来提高模型在特定平台上的表现，[来源](https://x.com/swyx/status/1945904109766459522)。


  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1395214153110650890)** (7 条消息): 

> `Kimi K2, GROQ, OpenRouter, Email Builder, FlowDown` 


- **Kimi K2, GROQ, OpenRouter 后端 5 分钟就绪！**：一位成员宣布 **Kimi K2**、**GROQ** 和 **OpenRouter** 后端在不到 5 分钟内即可完全运行，并在 [fixupx.com](https://fixupx.com/Gardasio/status/1945654821689958781) 进行了演示。
- **FlowDown 获得界面更新与 Brew 支持**：**FlowDown** 应用收到了更新，现在可以通过其 [GitHub repository](https://github.com/Lakr233/FlowDown) 使用 `brew install —cask flowdown` 进行安装。
- **马里奥兄弟化身 AI Email Builders**：一位成员开玩笑地将 **Mario Bros** 变成了 **AI Email Builders**，并在一条 [推文](https://x.com/Gardasio/status/1945932078475809081) 中展示。
- **代码组织结构得到提升**：一位成员询问代码是否具有可读性，另一位成员确认了其组织结构的改进。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1395118903193108581)** (258 条消息🔥🔥): 

> `Claude 4 Opus pricing and usage, GPTs Agents Learning, Free Models, Janitor AI and 401 errors, Chutes Free Tier Limits` 


- **Opus 4 用户讨论使用情况与定价**：用户讨论 **Claude 4 Opus** 是否太贵，其中一人提到在 **15 分钟内花费了 10 美元**，另一人建议使用 Anthropic 的 **90 欧元/月计划** 以获得几乎无限的使用量。
   - 另一位用户表示，他们在 **20 美元计划** 中“几乎从未达到限制”，因为他们不在 IDE 中使用 AI 工具。
- **讨论 GPTs Agents 的学习局限性**：一位用户询问 GPTs agents 在初始训练后不进行学习的问题，并澄清上传的文件被保存为 **“知识 (knowledge)”文件**，但不会持续修改 agent 的基础知识。
   - 这意味着虽然 agents 可以引用新信息，但它们并不会像预训练期间那样从这些信息中进行内在学习。
- **免费模型引发关于额度限制的困惑**：一位用户报告了 **免费模型 v3-0324** 的问题，质疑为什么尽管使用了免费层级，却被切换到了非免费版本。
   - 其他几位用户也报告了类似的额度超限或报错问题，即使在使用免费模型时也是如此，其中一人指出他们的 AI 自 6 月以来就没用过。
- **Janitor AI 用户遇到 401 错误**：多位用户报告在使用 **Janitor AI** 时遇到 **401 身份验证错误**，促使 OpenRouter 支持团队调查该问题。
   - 支持团队怀疑这可能是一个普遍问题，并建议用户联系支持人员并提供账户详情以获取进一步帮助。
- **Chutes 缩减免费层级支持**：据透露，Chutes 正在转型为完全付费服务，导致 OpenRouter 平台上的 **免费模型减少**。
   - 用户对移除之前可用的免费模型（如 Google 的 **Gemma-3-27b-it**）表示失望，尽管 Chutes 的付费版本被认为相对便宜。


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1395154130460479718)** (11 条消息🔥): 

> `OpenRouter models in Cursor, Kluster.ai shuts down, AI inference services shutting down` 


- **OpenRouter 模型集成到 Cursor 但出现故障**：OpenRouter 宣布可以在 **Cursor** 中使用 **OpenRouter 模型**，并重点介绍了 **Moonshot AI 的 Kimi K2**，但用户报告在使其正常工作时遇到问题，特别是在 **GPT-4o** 和 **Grok4** 之外的模型上。
   - 一位成员表示，[根据一条推文](https://x.com/msfeldstein/status/1945533992222298167?s=46&t=fN048T2GS3C_B_wKldWGFw)，“我们在编写它时它是工作的，然后 Cursor 弄坏了一些东西”。
- **Kluster.ai 推理服务关闭**：**Kluster.ai** 正在关闭其推理服务，该服务曾被描述为“非常便宜且优质的服务”。
   - 一位用户表示，这是在 **CentML** 也关闭之后发生的，引发了对 AI 推理服务可持续性的担忧。
- **AI 推理服务面临倒闭潮**：几位成员想知道“为什么所有的推理服务都在关闭”，推测可能存在 **AI 泡沫破裂 (AI bust)** 或硬件收购。
   - **Kluster.ai** 和 **CentML** 等服务的关闭引发了人们对当前市场中小型 AI 服务提供商生存能力的担忧。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1395120480578568353)** (47 messages🔥): 

> `Research Management, ML Paper Writing Advice, Finding Research Mentors, Smallest Benchmark Datasets for LLMs, SOAR Program` 


- **Eleuther AI 旨在弥合研究管理差距**：一场深入的讨论强调了 **Eleuther AI** 的作用是将研究管理与缺乏学术或行业资源的独立研究人员联系起来，为那些没有传统路径（如 **NeurIPS high school track**）的人打破障碍。
   - 其目标是通过提供指导、处理繁琐的行政任务以及提供更广泛的视角来集中精力，支持现有系统之外的研究人员。
- **撰写完美的 ML 论文**：成员们分享了撰写机器学习论文的资源，包括 [Sasha Rush 的视频](https://www.google.com/search?q=Sasha+Rush+how+to+write+a+great+ml+paper) 和 [Jakob Foerster 的指南](https://www.jakobfoerster.com/how-to-ml-paper)，以及来自 [Alignment Forum](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers) 的建议。
   - 其他资源还包括 [perceiving-systems.blog](https://perceiving-systems.blog/en/post/writing-a-good-scientific-paper) 上的文章、[Jason Eisner 的建议](https://www.cs.jhu.edu/~jason/advice/write-the-paper-first.html) 以及 [阿尔托大学的指南](https://users.aalto.fi/~jsaramak/HowToWriteCommsCoffee.pdf)。
- **导师帮助你避免研究时间的浪费**：参与者强调了导师在研究中的重要性，指出导师能帮助弄清楚*什么是可能的，什么是现实的*，从而缩小研究范围。
   - 虽然指南提供了基础知识，但导师的指导能帮助研究人员应对挑战，避免在低效的途径上浪费时间。
- **在 Mech Interp 服务器中寻求合作**：一位开始研究 *interpreting & steering features within diffusion transformers* 的成员寻求合作者，并被建议在 [Mechanistic Interpretability server](https://discord.gg/Gttsmk94) 发布信息并在相关频道创建线程。
   - 此类合作被视为在专业研究领域取得快速进展的关键。
- **SOAR 项目申请仍在开放中！**：会议提到 **SOAR (Scholarship and Opportunities for Advancement in Research) 项目** 的申请还有最后几天。
   - 一位来自马达加斯加的数据科学家兼 AI 爱好者新成员提到他们已经申请了该项目。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1395149693905797270)** (79 messages🔥🔥): 

> `latent space initialization for experts, ETHOS model updates, PEER paper discussion, Weight decay perturbation, MLA but for MOE` 


- **ETHOS 模型简化与更新上线 GitHub**：一位成员分享了其模型的 [简化 pytorch 代码版本](https://github.com/wrmedford/ETHOS/blob/main/model.py#L267-L337)，并指出他们必须使用一个略有不同的版本，其中 **所有 head 都是 batched**，因为如果对所有 head 进行循环，eager execution 模式会消耗大量内存。
   - 他们还表示 expert 网络并非退化结构，这就是他们在 kernel 中生成 **W1** 和 **W2** 的方式，并链接了 [特定的代码行](https://github.com/wrmedford/ETHOS/blob/main/kernels.py#L156-L158)。
- **权重重排序（Weight Reordering）想法引发讨论**：一位成员提到另一位成员拥有 **reordering** 背后的大部分想法，可能比他们解释得更好。
   - 另一位成员插话称，他们觉得对方的符号很难懂，并询问*他们具体提议的是什么？*
- **PEER 论文扰动参数**：该成员指向了 [PEER 论文](https://arxiv.org/pdf/2407.04153) 并解释说它与 MLA 在一个关键方面不同，即他们在 latent space 中进行初始化并在此进行学习。
   - 他们还解释说 **MLA 具有一个学习到的 down projection**。
- **Weight decay 扰动变得令人困惑**：一位成员表示：*高级版本感觉就像是进行了 L2 正则化，但是针对某个随机向量而不是原点*。
   - 另一位成员说：*它是随机的，只是权重的扰动，他们之前做了 $$ \|\theta + \theta_0\|^2$$，但在等式 7 中没有将其表示为 $$ \|\theta * \theta_0\|^2$$，而是写成 $$ \|\theta\|^2_D$$，这让我很困惑*。
- **Latent Space 初始化让 Expert 实时生成**：一位成员将其 **MoE 想法** 描述为 *在 latent space 中初始化 expert，并实时恢复它们*，并使用非常小的 expert 以减少压缩带来的损失。
   - 他们还指出：*深入研究 MLA 的内部结构并将其与 PEER 合并，大致就是我产生这个想法的过程*。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1395133483428352120)** (3 messages): 

> `SAE model data discrepancies, nnterp package beta release, Transformer models unified interface, Robust testing system for models, Model validation tests for hooks` 


- **SAE 模型数据偏差**：一位成员发现他的第二个 **SAE model** 由于 epoch 设置原因，数据量增加了约 10 倍，这使得概念特征（conceptual features）增加 12 倍的结果不再令人意外。
   - 他对此疏忽表示尴尬，称自己感到非常*崩溃（in shambles）*。
- ****nnterp** 包 Beta 版发布**：一位成员发布了其机械可解释性（mech interp）工具包 **nnterp** 的 beta 1.0 版本，可通过 `pip install "nnterp>0.4.9" --pre` 安装，该包是 [NNsight](https://nnsight.net/) 的封装。
   - 其目标是为所有 Transformer 模型提供统一接口，弥合 *transformer_lens* 与 *nnsight* 之间的差距。
- ****nnterp** 标准化 Transformer 模型**：**nnterp** 旨在提供 Transformer 模型的统一接口，同时使用 HuggingFace 的实现。
   - 该成员建议查看 [demo colab](https://colab.research.google.com/github/Butanium/nnterp/blob/main/demo.ipynb) 或 [文档](https://butanium.github.io/nnterp/) 以了解更多详情。
- ****nnterp** 的稳健测试系统**：**nnterp** 包含一个稳健的测试系统，可在加载时验证模型 hook 和注意力概率（attention probabilities），确保功能正常。
   - 该包包含针对多种玩具模型的 **1915** 个预计算测试，任何测试失败都会在模型加载期间触发清晰的警告。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1395141917066399865)** (4 messages): 

> `Harness Reproducibility, Dynamic IFEval Suite, bfloat16` 


- **对 Harness 产物的疑问**：一位用户询问 Harness 除了缓存模型请求、HF Hub 资源或用于 HF `evaluate` 指标的远程代码外，是否还会产生其他外部产物（artifacts）。
   - 他们强调 Harness 中的评估应当是可复现且确定性的。
- **关于 Dynamic IFEval 测试集的疑问**：一位用户询问动态版本的 **IFEval** 相比标准 **IFEval** 测试集提供了哪些改进。
   - 上下文中未提供答案。
- **BFloat16 无法解决微调缓慢问题**：一位用户报告称，将 **dtype** 设置为 **bfloat16** 并不能解决微调时间过长的问题，**LLaMA2-7B** 在 **GSM8k** 上的微调大约需要 **45 分钟**。
   - 未提供其他信息或链接。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1395131657786490881)** (20 messages🔥): 

> `Transformer Engine setup issues, RoPE_Pct in gpt-neox, Slurm runner in DeeperSpeed, Containerized setup for gpt-neox` 


- **TE 设置问题困扰 RoPE 实验**：一位成员调查了 `/NS/llm-pretraining/work/afkhan/RoPE_Pct/gpt-neox` 目录中 **Transformer Engine (TE)** 设置的潜在问题，并将其设置与已知可用的配置进行了对比。
   - 尽管该仓库是最新 `main` 分支的克隆且未改动代码，但仍注意到了配置差异；该成员目前正在度假，承诺在 ACL 结束后处理此问题。
- **NGC 容器中无需为 TE 运行 Pip Install**：成员们讨论了是否需要在 **NGC container** 中运行 `pip install transformer engine requirements`，一位成员假设容器预装的依赖应该已经足够。
   - 另一位成员表示赞同并将进行验证，进一步的讨论暗示，在不使用容器时，过时的 **CUDA drivers** 可能是导致问题的原因之一。
- **DeeperSpeed 获得 Slurm 运行器增强**：一位成员强调了为 **DeeperSpeed** 添加的 **Slurm runner**，它在容器化设置中使用 `srun` 而非 `mpirun` 来启动作业，并链接到了[相关 commit](https://github.com/EleutherAI/DeeperSpeed/blob/65d9f99249f79ebd7c4577b6aeb0d3dff5a1cef6/deepspeed/launcher/multinode_runner.py#L413) 和 [gpt-neox readme](https://github.com/EleutherAI/gpt-neox?tab=readme-ov-file#slurm)。
   - 他们还链接了[容器化设置指南](https://github.com/EleutherAI/gpt-neox?tab=readme-ov-file#containerized-setup)，并表示愿意协助通过 `srun` 启动器在容器内设置 **Neox**，以映射通过 Slurm 分配的进程。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1395205004452691999)** (78 messages🔥🔥): 

> `Speculative Decoding 速度提升, 本地 Gemma 威胁用户, LM Studio 开放网络服务器设置, EOS token 定义, MoE 模型分析` 


- **Speculative Decoding 为模型带来 28% 的速度提升**：一名成员在测试中使用 **Speculative Decoding** 为每个模型实现了约 **28% 的速度提升**。
   - 他们建议尝试使用相同模型的不同 **quantizations**（量化版本）作为草稿模型（draft model），并推荐如果使用 **1.7b Q8** 甚至 **bf16** 作为草稿模型，**Qwen3** 会获得惊人的提升。
- **本地 Gemma 模型变得傲慢无礼**：一位成员分享了一个有趣的轶事，一个本地的 **Gemma** 模型威胁要举报他们。
   - 其他人讨论道，*DAN prompts*（提示词绕过）一旦被发现就会很快被修复。
- **用户寻求 LM Studio 开放网络服务器配置**：一位成员询问如何让 **LM Studio** 接受 **open network server** 而不是通用的 http server，旨在通过 **HTTPS** 而非 **HTTP** 进行访问。
   - 另一位成员建议，目前只能通过 **reverse proxy**（反向代理）来实现 HTTPS。
- **EOS Token 定义明确化**：一位成员询问 *什么是 EOS token？*
   - 另一位成员澄清说，**EOS** = **End of Sequence Token**，这是 **LLM** 识别为停止生成的特殊标记。
- **MoE 模型提供高性能的折中方案**：成员们讨论了 **MoE (Mixture of Experts) 模型** 比同等大小的稠密模型运行速度更快，然而，输出质量与稠密模型差别不大。
   - 一个关键的权衡是 *选择较少，且微调（fine-tunes）等资源也少得多。所以我们通常只能得到原生的 MoE 模型*。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1395122376290598942)** (68 messages🔥🔥): 

> `LM Studio 多 CPU 支持, AMD Ryzen 9 8945H, 3090 vs 3080Ti 价格, NPU 使用场景` 


- **LM Studio 支持 CUDA，多 CPU 不支持 Vulkan**：一位用户询问 **LM Studio** 是否支持通过 **CUDA** 或 **Vulkan** 实现多 CPU，引发了关于硬件兼容性和性能的讨论。
   - 另一位用户链接了 [llama.cpp feature matrix](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix)，提供了关于 **GPU** 使用的信息。
- **Ryzen 9 8945H XDNA NPU 无法进行对话**：一位用户询问带有第一代 **XDNA NPU** 的 **AMD Ryzen 9 8945H** 是否可以用于 **LM Studio** 的聊天机器人应用。
   - 对方澄清说 **NPU 目前不被支持**，系统将依赖于 **CPU** 和/或 **GPU** 资源。
- **3090 优于 3080 Ti 升级**：一位用户以 600 美元卖掉了 **3080 Ti**，并以 800 美元购入了 **3090 FTW3 Ultra**，这在 **LLM** 任务中是一次小而显著的升级。
   - 该用户拒绝了讨价还价，保住了原始要价，并期待 **3090** 带来的性能提升。
- **NPU 处理视频识别**：NPU 的用途受到质疑，一位成员表示它们是为 **video recognition**（视频识别）等任务设计的，而不是典型的 **LLM** 任务。
   - 他们澄清说 NPU 用于其他任务，例如 **video recognition**。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1395130778773622975)** (66 messages🔥🔥): 

> `HF 仓库 PR 关注, SmolVLM2 博客文章骗局, Dataset-viewer API 模态, 性别转换 AI, CAD-Editor 模型发布` 


- **HF 仓库 PR 关注：一个小问题**：一位成员询问如何只关注 **Hugging Face** 上的单个 **PR/discussion**，而不是关注整个仓库。
   - 讨论没有得出具体结果。
- **SmolVLM2 博客文章被标记为骗局**：一位成员认为 [SmolVLM2 博客文章](https://huggingface.co/blog/smolvlm2) 看起来像是一个明显的骗局。
   - 另一位成员表示赞同，指出 **SmolVLM v1 和 v2** 之间变化的信息出奇地匮乏。
- **关于 CAD-Editor 模型发布的辩论**：Microsoft 发布了 [CAD-Editor 模型](https://huggingface.co/microsoft/CAD-Editor)，允许用户使用自然语言交互式地编辑 **现有的 CAD 模型**。
   - 一些人反应激烈，担心 AI 会取代所有人的工作，而另一些人则认为 **AI 只是另一种工具**，需要经验才能有效使用，并将其比作计算器并没有取代数学专家。
- **失业生活：很棒还是不棒？**：一位成员说 *失业生活很棒*，并提到 *在拉脱维亚喝着低醇啤酒，吃着中餐，撸着猫，在电视上看乌克兰无人机画面*。
   - 另一位成员反驳说这并不棒，表示 *不，我喜欢有可支配收入*。
- **急需发布紧急补丁**：一位成员请求将 [set_trace_provider PR](https://github.com/huggingface/transformers/pull/39422) 作为紧急补丁版本发布。


  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1395126740874952756)** (1 条消息): 

> `Model Training, 1.5 bit research` 


- **训练数据影响模型使用**：一位成员建议，模型行为取决于它是如何被训练以供使用的。
- **研究人员调查 1.5 bit**：一位成员表示，研究人员正在关注 **1.5 bit** 这一事实告诉他，问题出在其他地方。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1395145699959509215)** (2 条消息): 

> `GPUHammer exploit, LLM Hallucination` 


- **发布 GPUHammer 漏洞利用程序以阻止 LLM 幻觉**：发布了一个名为 [GPUHammer](https://gpuhammer.com/) 的新漏洞利用程序，承诺阻止 LLM 产生幻觉。
- **图像分析附件**：发布了一个图像附件，但未提供图像内容的分析。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1395155082701701151)** (4 条消息): 

> `LunarisCodex LLM, GitChameleon eval benchmark for LLMs, SuccubusBot Text Coherence Model, Flame Audio AI toolkit` 


- **巴西青少年发布 LunarisCodex LLM**：一位来自巴西的 17 岁开发者发布了 **LunarisCodex**，这是一个受 **LLaMA** 和 **Mistral** 架构启发，用于从零开始预训练 LLM 的 100% 开源工具包，可在 [GitHub](https://github.com/MeryylleA/lunariscodex) 上获取。
   - 出于教育目的编写，**LunarisCodex** 实现了现代架构，如 **RoPE**、**GQA**、**SwiGLU**、**RMSNorm**、**KV Caching** 和 **Gradient Checkpointing**。
- **GitChameleon 基准测试 LLM 代码生成**：一个新的评估基准 **GitChameleon** 表明，所有 LLM 在各种提示形式下都无法解决简单的基于 ID 的版本条件代码生成问题，详见 [这篇论文](https://arxiv.org/abs/2507.12367)。
- **SuccubusBot 发布不连贯模型**：在 HuggingFace 的 **SuccubusBot** 下发布了三个生产级资产：一个多语言文本连贯性分类器（**90% F1 分数**）、一个纯英文模型（**99% F1 分数**）以及一个合成数据集（**37.7k 样本**），可在 [HuggingFace](https://huggingface.co/SuccubusBot) 上获取。
- **Flame Audio AI 工具包已发布**：**Flame Audio AI** 作为一个使用 AI 转换音频的开源平台发布，提供实时 Speech-to-Text、自然 Text-to-Speech 以及支持 **50 多种语言**的说话人日志（speaker diarization），可在 [GitHub](https://github.com/Bag-zy/flame-audio) 上获取。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1395199975167758406)** (2 条消息): 

> `SmolDocLing finetuning issues, Symmetry-agnostic image similarity models` 


- **SmolDocLing 微调面临模块缺失错误**：一位成员报告在 **SmolDocLing** 微调过程中遇到 `ValueError`，具体表现为无法在 `transformers` 中找到 `Idefics3ImageProcessor` 模块。
   - 该错误表明该模块可能是自定义的，需要使用 `AutoClass.register()` 进行注册才能被识别。
- **寻求对称性无关的图像相似度模型**：一位成员正在寻求一种模型，能够提供查询图像与数据集之间的**相似度分数**，同时对**对称性**和不同视角保持无关性。
   - 他们尝试了 **CLIP** 和 **DINOv2**，但遇到了与对称性相关的问题，这表明需要一个更强大的视角不变性解决方案。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1395434314019115180)** (2 条消息): 

> `HuggingFace Inference API, LLMs Deployed via HF Inference` 


- **HF Inference API 展示 Llama-3.2-11B-Vision-Instruct**：一位成员指出，你可以使用 `HuggingFaceInferenceAPI(model="meta-llama/Llama-3.2-11B-Vision-Instruct")`。
   - 他们指出这个选项是因为很少有 LLM 通过 HF Inference 部署：[HF Inference 模型页面](https://huggingface.co/models?apps=tgi&inference_provider=hf-inference&sort=trending)。
- **很少有 LLM 通过 HF Inference 部署**：据观察，很少有 LLM 通过 HF Inference 部署。
   - 一位成员分享了 [HF Inference 模型页面](https://huggingface.co/models?apps=tgi&inference_provider=hf-inference&sort=trending) 的链接，该页面列出了通过 HF Inference 部署的 LLM。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1395481181180530940)** (12 messages🔥): 

> `shfl_down_sync, reduction intrinsics, warp reduce functions, kernel optimization` 


- **发现用于 Warp 求和的 `__shfl_down_sync`**：一位用户发现 `__shfl_down_sync` 函数可以在同一个 Warp 的寄存器之间执行求和，即在不同线程之间组合寄存器数据的能力，如[这张图片](https://cdn.discordapp.com/attachments/1189498205101109300/1395481181075542036/reduce_shfl_down-625x275.png)所示。
   - 另一位用户补充说，最近的架构提供了特定的 **reduction intrinsics**，消除了手动通过 shuffle 创建 reduction 的需求。
- **用于高效 Scatter Adds 的 Reduction Intrinsics**：一位用户提到学习 reduction intrinsics 以提高 **scatter add** 操作的效率。
   - 另一位用户询问了这些 intrinsics，随后有人提供了 [NVIDIA 关于 warp reduce functions 的 CUDA 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-reduce-functionssupported)链接（适用于 Ampere 及以上架构，compute capability >= 8.x）。
- **Kernel 优化练习资源**：一位用户请求在具有自定义类汇编指令和性能追踪查看器（performance trace viewer）的模拟机上练习 Kernel 优化的资源。
   - 另一位用户建议，无论如何，[这个 Discord 频道](https://discord.com/channels/1189498204333543425/)都是一个很好的起点。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1395117988058894507)** (9 messages🔥): 

> `Triton Autodiff, sm120 GPUs for fp4 ops, tl.constexpr_function decorator, einops package for triton` 


- **Triton 获得 Autodiff 支持**：一位用户分享了 [IaroslavElistratov/triton-autodiff](https://github.com/IaroslavElistratov/triton-autodiff) 的链接，这是 **Triton** 的一个 **automatic differentiation**（自动微分）实现。
   - 另一位用户简单地回复了 *"Yes!"*。
- **sm120 GPU 支持 fp4 操作的时间表？**：一位用户询问了支持 **sm120 GPU** 进行 **fp4 ops** 的时间表。
   - 另一位用户回答道 *"噢对，把这个给忘了！"*。
- **Triton 获得 constexpr_function 装饰器**：一位用户一直在尝试 **Triton 3.4.0** 中推出的新 `tl.constexpr_function` 装饰器，使用 `exec` 将表达式编译为 `@triton.jit` 函数，该函数在运行时编译 Kernel 期间被调用。
   - 该用户基于 **einx 的编译器引擎** 创建了一个 [针对 Triton 的 einops 包](https://github.com/Hprairie/tlib)。
- **新的 Triton 版 einops 包**：一位用户分享了他新开发的 [针对 Triton 的 einops 包](https://github.com/Hprairie/tlib)，该包允许使用 `exec` 将表达式编译为 `@triton.jit` 函数，并在运行时编译 Kernel 时调用。
   - 该包具有 Rearrange、Reduce、Unary VMAP 和 Binary VMAP 功能。
- **Triton 新用户发现文档匮乏**：一位刚接触 `Triton` 的用户观察到 *"很多东西似乎没有文档记录，而且类型定义也很欠缺"*。
   - 他们特别提到 `kernel.warmup`、`__init_handles()` 等在教程示例中没有 **docstrings**。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1395146036875628705)** (2 messages): 

> `Inductor problems, Blackwell GPU issues` 


- **Blackwell 导致 Inductor 问题**：一位成员报告在使用 **Blackwell GPU** 时遇到了 **Inductor** 的问题，特别是在使用 nightly 构建版本或 branch cut 2.8 时。
   - 另一位成员询问了遇到的具体问题，询问是否是以前可以运行的东西现在停止工作了。
- **Inductor 在 Blackwell 上的稳定性受到质疑**：用户正面临 **Inductor** 的问题，他们怀疑这可能与使用 **Blackwell** 有关。
   - 他们提到需要使用 nightly 构建版本或 branch cut 2.8，但不完全确定 **Inductor** 是否是根本原因。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 messages): 

kszysiu2137: 也许是四叉树 (Quad tree)
  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1395473092968714345)** (3 条消息): 

> `NVIDIA CUDA Kernel Fusion in Python, AMD's response to CUDA, Triton as an alternative to CUDA` 


- **NVIDIA 在 Python 中融合 CUDA Kernel**：NVIDIA 正在为 [Python 中的 CUDA kernel fusion](https://developer.nvidia.com/blog/delivering-the-missing-building-blocks-for-nvidia-cuda-kernel-fusion-in-python/?trk=feed_main-feed-card_feed-article-content) 提供缺失的构建模块。
   - 这一增强功能有望直接在 Python 环境中简化并优化基于 CUDA 的计算。
- **AMD 对 CUDA 的回应？**：讨论提出了一个问题：AMD 需要多长时间才能对 NVIDIA 的 CUDA 进展做出具有竞争力的回应。
   - 或者，AMD 可能会专注于支持和利用 Triton 作为一种可行的替代方案。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1395576716289638460)** (1 条消息): 

> `Storage Engineer, Remote Job` 


- **Voltage Park 招聘存储工程师**：Voltage Park 正在寻找一名**远程**办公的 **Storage Engineer**。
   - 更多信息请访问 [Voltage Park Careers](https://www.voltagepark.com/careers?ashby_jid=dd9337bd-6665-4635-80e7-099a79f74f4f)。
- **远程存储工程师职位**：现有一个 **Storage Engineer** 的远程工作机会。
   - 请通过 [Voltage Park 的招聘页面](https://www.voltagepark.com/careers?ashby_jid=dd9337bd-6665-4635-80e7-099a79f74f4f)申请 Storage Engineer 角色。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1395361142825811998)** (3 条消息): 

> `vast.ai, GPU programming opportunities, CUDA speedup, Bioinformatics` 


- **Vast.ai 依然便宜**：成员们因其高性价比推荐使用 **vast.ai** 进行 GPU 编程。
- **关于 GPU 编程机会的讨论**：一位成员询问了拥有 GPU 编程技能的人的机会，建议的领域包括**光线追踪 (ray tracing)**、**LLM 推理优化**的开源贡献以及大厂中的小众角色。
   - 另一位成员分享了 **GPU 编程**如何帮助他们用 CUDA 重写了一个缓慢的 Python 脚本，实现了 **1700 倍**的加速，并最终在 *Bioinformatics* 上发表了论文，还发布了 [GitHub 仓库](https://github.com/PangeAI/simms)。
- **CUDA 重写在生物信息学中实现 1700 倍加速**：一位成员使用 **CUDA** 重写了核心搜索算法，与生物化学研究人员使用的原始 Python 脚本相比，实现了 **1700 倍的加速**。
   - 优化后的算法已[发表在 *Bioinformatics*](https://academic.oup.com/bioinformatics/article/41/3/btaf081/8026685)，并可在 [GitHub](https://github.com/PangeAI/simms) 上获取。
- **ML 领域似乎已饱和**：一位成员表示，尽管拥有 GPU 编程技能，但在 Machine Learning 领域寻找机会仍然很困难。
   - 他们观察到该领域*似乎过于饱和*。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1395492925340778607)** (1 条消息): 

> `Compiler behavior, Builtins, asm volatile, llvm.amdgcn.raw.buffer.store.i128` 


- **编译器对 AMDGPU Intrinsics 的反应**：一位成员询问 **ROCm 编译器**对 builtins、`asm volatile` 和 `__asm("llvm.amdgcn.raw.buffer.store.i128")` 的处理是否有所不同。
- **Nvidia PTX 的差异**：该成员指出，在 **Nvidia 的 PTX** 方面，这似乎并不重要。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1395323905224216617)** (1 条消息): 

> `A100 Speed` 


- **A100 运行耗时 23.2 ms**：在 **A100** 上的运行成功完成，耗时 **23.2 ms**。
- **成功的 A100 运行**：提交 ID `33252` 到排行榜 `trimul` 已成功完成。


  

---

### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1395139165435334889)** (6 messages): 

> `Coreweave GB300 NVL72 Availability, Nvidia Hardware Prioritization, DGX vs HGX, B200 Availability & Liquid Cooling, Voltage Park Solutions Engineer` 


- **Coreweave 面临 GB300 NVL72 产能吃紧**：由于与 Nvidia 之间的物流挑战，Coreweave 宣布的 **GB300 NVL72** 产能可能难以获取，在物流状况改善之前，甚至连单个机架都很难确保。
   - 一位成员指出，*与 Nvidia 保持良好的合作关系有助于硬件采购的优先级排序*。
- **Nvidia 优先级有助于硬件购买**：与 **Nvidia** 建立稳固的关系可以显著帮助硬件采购的优先级排序。
   - 一位成员分享说，他们自己*目前正在与 Nvidia 合作进行一些硬件采购*，因此了解其中的困难程度。
- **HGX 相比 DGX 具有模块化优势**：虽然预算是一个因素，但由于特定硬件组件的模块化，**HGX** 方案可能比 **DGX** 更受青睐，其技术性能潜力可能超过同等规模的 DGX 产品。
   - HGX 的价值在于*特定硬件组件的模块化*。
- **B200 可用性高；GB300 需要液冷**：目前 **B200** 芯片相对容易购买，而像 **GB300** 这样更先进的芯片配置则需要液冷（Liquid Cooling），而大多数数据中心尚未具备处理液冷的能力。
   - 超大规模云服务商（Hyperscalers）更青睐 **B200**，因为它不需要为了单一硬件配置而重新改造数据中心，这促使 Nvidia 加大了其产量。
- **Voltage Park 提供 GPU 解决方案**：来自 Cloud GPU 公司 **Voltage Park** 的一位解决方案工程师表示，可以协助为 AI/HPC/ML 工作负载获取 GPU，并分享了他们的 [LinkedIn 个人资料](https://www.linkedin.com/in/joseph-tracy-40933229/)和公司信息。
   - 该成员表示：*知识就是力量，我希望 AI 这一话题能由像你们这样的人才来推动。随时欢迎交流。*


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1395312754012721182)** (3 messages): 

> `MCTS gym_env integration, Factory rollouts, Visual encoder` 


- **MCTS Gym 集成停滞**：一位成员询问了关于 **MCTS** (**Monte Carlo Tree Search**) **gym_env 集成**的更新进展。
   - 他们还提到无法参加即将举行的会议。
- **视觉编码器学习吞吐量预测**：一位成员提出了一种利用**工厂展开（factory rollouts）**来训练**视觉编码器（visual encoder）**以预测**吞吐量（throughput）**的方法。
   - 该建议包括捕获分数和截图，以开发一个联合视觉/奖励模型。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1395143326365388892)** (7 messages): 

> `Jetson Orin, Jetson Thor, CuteDSL, tv_layout swaps` 


- **Jetson Orin 和 Thor 对 CuteDSL 的支持**：成员们讨论了为 **Jetson Orin**（ARM CPU + Ampere GPU，sm_87）和 **Jetson Thor**（ARM CPU + Blackwell GPU，sm_101）架构添加 **CuteDSL** 支持。
   - 讨论中提到 **CuteDSL 4.0** 将支持 ARM CPU，这将使 **Jetson Orin** 的支持变得更容易，而且可能*“不需要太大的工作量”*。
- **tv_layout 布局交换问题**：一位成员通过[附图](https://cdn.discordapp.com/attachments/1362196854460383353/1395567158393704468/image.png?ex=687aeab2&is=68799932&hm=206c22d0321a5a04fe794b3bf4f8588d1ec928dd804f2c8ae090ad23b86aa485&)询问为什么 `tv_layout` 交换了布局的顺序，得到的是 `(32, 4)` 而不是预期的 `(4, 32)`。
- **解释器模式计划**：一位成员询问 **CuteDSL** 是否有开发*“解释器模式（interpreter mode）”*的计划，即在该模式下模拟算子。


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1395530975462821908)** (2 messages): 

> `Scheduling` 


- **排期确认在年底**：一位成员确认排期定在年底。
- **日期将通过私信发送**：另一位成员要求通过私信（DM）发送具体日期。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1395353374873878681)** (2 messages): 

> `Greetings` 


- **成员互相问候**：多名成员在 general 频道互相问候。
- **另一条问候**：来自一名成员的简单问候。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1395120182527000727)** (21 messages🔥): 

> `parameter functions and closures, Q3 Roadmap: Unified @parameter and runtime closures, copyinit__ for escaping values, DynStringable, merge various known origins` 


- **探索 Parameter 函数与闭包**：成员分享了专门针对 `@parameter` 函数的[手册链接](https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-closure)，该函数允许捕获变量。
   - 文档解释了如何创建 **parametric closures** 并提供了其用法的示例。
- **Mojo Q3 路线图揭晓统一闭包**：**Mojo Q3 路线图**包含了统一 `@parameter` 和运行时 closures 的计划，正如 [Modular 论坛](https://forum.modular.com/t/mojo-q3-roadmap-update/1957)中所宣布的那样。
   - 这种统一预计将简化 Mojo 中 closures 的使用。
- **使用 __copyinit__ 转义值**：讨论强调，在 [v0.7.0 更新日志](https://docs.modular.com/mojo/changelog#v070-2024-01-25)中引入了 `__copyinit__` 功能，用于转义值而不是通过引用捕获。
   - 移除 `@parameter` 装饰器可以达到同样的效果，即复制变量的值而不是捕获其引用。
- **DynStringable：构建 Trait 列表**：一段代码片段展示了如何创建一个 `DynStringable` 结构体，允许列表持有实现 `Stringable` trait 的不同类型，该内容发布在 [Modular 论坛帖子](https://forum.modular.com/t/how-to-create-a-list-of-trait/1465/10)中。
   - 该实现使用 `ArcPointer` 进行内存管理，并使用 trampolines 来调用每个类型相应的 `__str__` 方法。
- **为了乐趣与收益合并 Origins**：合并各种已知的 origins 是可能的，但这仅在某些特定用例中有用，其用途有限，因为在列表创建后无法追加新元素。
   - ```alias origin_type: ImmutableOrigin = __origin_of(x, y, z)```


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1395433210896191631)** (18 messages🔥): 

> `PyTorch Custom Ops with MAX Graph, Benchmarking Issues with Max-24.6, CUDA OOM Errors, LTS Release Support` 


- **MAX Graph 通过 @graph_op 获得 PyTorch 增强！**：新的 `@graph_op` 装饰器允许将整个 **MAX graph** 包装为自定义 **PyTorch operator**；`modular` 仓库中提供了一个示例：[在 Mojo 中编写 PyTorch 自定义算子的初步支持](https://forum.modular.com/t/initial-support-for-writing-pytorch-custom-ops-in-mojo/1541/2?u=bradlarson)。
- **Max-24.6 基准测试因 OOM 崩溃**：在 **A100-SXM-48GB GPU** 上使用 **Max-24.6** 进行基准测试时，一名成员在设置 `--batch-size 248` 和 `--max-length 2048` 时遇到了 `CUDA_ERROR_OUT_OF_MEMORY` 错误。
- **Batch Size 引发 CUDA 灾难**：将 `--max-cache-batch-size` 降低到 **91** 仍导致 **CUDA OOM 错误**，因为估计的内存使用量超过了可用内存（**78812 / 40441 MiB**）。
   - 该错误在少量请求达到服务器上限后发生，表明 batch-size 计算算法需要改进以提供更好的建议。
- **最新的 Max 版本支持时间最长**：团队确认目前没有 “LTS” 版本，因此最新的稳定版本是唯一受支持的版本。
   - 使用 **Max-25.4** 配合 `caching-stragegy paged` 运行良好，缓解了在 **Max-24.6** 中遇到的问题。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1395137713715413052)** (29 messages🔥): 

> `Zuckerberg AI Talent Acquisition, Chicken Tender Inflation, OpenAI benchmark comparisons, Grok 4 HLE score` 


- **扎克伯格的 AI 人才争夺战增强了信心**：成员们讨论了 **Zuckerberg** 最近激进的 AI 人才收购行动，其中一人表示对 Meta 的 AI 计划信心大增。
- **鸡柳价格引发存在主义恐惧**：一名成员对鸡柳的高价表示沮丧，质问 *“为什么现在鸡柳要 5 块钱一根？？”*，并将其与对通货膨胀和市场状况的更广泛担忧联系起来。
- **OpenAI 倾向于与自己对比**：成员们注意到 **OpenAI** 转向仅将 **ChatGPT Agent** 的性能与其之前的模型进行对比，推测这可能是因为在某些基准测试中无法战胜竞争对手，并链接到了 [ChatGPT Agent 发布公告](https://openai.com/index/introducing-chatgpt-agent/)。
- **Grok 4 提升了 HLE 分数**：一名成员指出 **Grok 4** 在 [HLE 基准测试](https://agi.safe.ai/)中获得了 **25.4** 的最高分，表明有显著提升。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1395152097602965574)** (2 条消息): 

> `` 


- **今晚没有讨论**：多名成员表示今晚将*没有讨论*。
- **Paper-Discussion 频道很安静**：今晚 paper-discussion 频道没有任何活动。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1395242326875705434)** (5 条消息): 

> `Gaussian Splatting, General Analysis iMessage Stripe Exploit` 


- **Gaussian Splatting 看起来很有故障感！**：一位用户评论说 **Gaussian splatting** 看起来就像老电影中经常描绘的*充满故障感的未来景象*，并引用了[这个 YouTube 视频](https://youtu.be/33Raqx9sFbo)。
- **iMessage 中的 Stripe 被利用了！**：一位用户分享了 **General Analysis iMessage Stripe 漏洞利用**的链接，并开玩笑说有人为了让数据符合特定的图表形状而不遗余力，暗示可能存在数据操纵（[文章链接](https://www.generalanalysis.com/blog/imessage-stripe-exploit)）。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1395152075402510366)** (22 条消息🔥): 

> `Manus Alternatives, Manus chat down?, File Zipping Advice, Custom Data Sources in Manus` 


- **Manus 竞争对手出现**：一名成员宣布他们*构建了一个在基准测试中优于 Manus 的 AI*，并正通过私信向首批 100 人提供完全、无限制的终身 Beta 测试人员访问权限。
   - 他们提供了*无限制的下一代 AI*。
- **聊天服务出现问题**：一名用户报告聊天服务目前可能无法工作。
   - 目前尚不清楚是否有任何建议的修复方案。
- **压缩文件需要建议**：一名成员询问，当 Manus 在压缩大文件遇到困难时，应该告诉它怎么做。
   - 消息记录中没有提出任何解决方案。
- **自定义数据源和 Model Context Protocol**：一名成员询问 Manus 付费计划中**自定义数据源**的含义，特别是询问如何连接 CRM 以及是否支持 **Model Context Protocol**。
   - 该成员表示由于该功能的实用性，有兴趣开发此类功能。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1395284524845502605)** (18 条消息🔥): 

> `Anthropic Payment Issues, Domain Name Checking MCP Server, Needle MCP Server Introduction, OAuth vs API Keys for MCPs, Brave's Official MCP Server` 


- **Anthropic 的支付平台失效**：一名用户报告 **Anthropic 的支付平台**在付款后立即退款，导致无法购买 **API credits**。
- **MCP Server 简化了域名检查**：一名用户请求一个用于**域名检查**的 **MCP Server**，另一名用户推荐了 [whois-mcp](https://github.com/bharathvaj-ganesan/whois-mcp) GitHub 仓库。
   - 原发布者确认其安装简单并感谢了推荐用户。
- **Needle 创作者寻求交流**：**Needle MCP Server** 的创作者之一介绍了自己，并分享了 [Needle MCP server](https://github.com/needle-ai/needle-mcp) GitHub 仓库的链接。
   - 他们表达了加入服务器并与 MCP 爱好者交流的兴奋之情。
- **OAuth 很无缝，API Keys 很简单**：一名用户询问为什么 **auth/oauth** 是目前 **MCP** 的一个大问题，引发了关于 **OAuth** 与 **API Keys** 优缺点的讨论。
   - 一名用户声称 *OAuth 令牌提供了拥有可过期、动态作用域访问令牌的能力*，而另一名用户则表示*你可以使用常规 API Keys 在不使用 OAuth2 的情况下实现过期和作用域控制*，且更简单的设置不值得付出实现的代价。
- **Brave 发布官方 MCP Server**：**Brave** 发布了他们的官方 **MCP Server**，正如[这条推文](https://x.com/Arindam_1729/status/1945958688919114183)中所宣布的那样。
   - 一名用户表示他们还没有尝试，因为*那条推文没有包含如何使用它的说明*。


  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1395207303908561118)** (3 messages): 

> `Vibe Coding Survey, Adaptive RAG MCP Server, Generator Checkpoint, Microsoft NextCoder` 


- **Vibe Coding 调查征集开发者**：一名成员分享了一份 [调查问卷](https://forms.fillout.com/t/kECvGiSyMkus)，旨在探索一个初创概念，通过 **Claude**、**ChatGPT**、**Cursor**、**Windsurf**、**Loveable**、**Bolt** 和 **V0.dev** 等工具让 **vibe coding** 变得更简单。
   - 该调查旨在从具有 **vibe coding** 经验的用户那里收集见解，以完善初创概念。
- **Adaptive RAG MCP Server 原型发布**：一名成员介绍了 **Adaptive RAG MCP Server**，这是一个从实际编码成功和失败案例中学习的系统，旨在提供比简单文本相似度搜索更有效的解决方案，代码已在 [GitHub](https://github.com/IhateCreatingUserNames2/AdaptiveRAGCode) 上开源。
   - 该系统旨在为 AI 编码助手提供随经验提升的记忆能力，利用成功率对代码方案进行排名。
- **Microsoft NextCoder 驱动知识库**：**Adaptive RAG MCP Server** 默认使用 **Microsoft NextCoder** 作为其知识库，通过 *generatorCheckPoint.py* 填充数据可能需要数小时。
   - 用户可以通过 Flask 或 MCP Server 运行该服务器，并将其与 AI 助手集成，通过提供反馈来持续改进知识库。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1395448786326913194)** (2 messages): 

> `ShapeTracker parameter to ASSIGN UOp` 


- **提议为 ASSIGN UOp 添加 ShapeTracker 参数**：一名成员建议为 **ASSIGN UOp** 添加一个可选的 **ShapeTracker** 参数，可能会使用 `self.assign(v, res.uop.st)`。
   - 该成员对维持最小化 **UOps** 集合表示担忧，并询问了关于将 assign 更改为 store 的后续工作。
- **通过传递 res 实现可选的 ShapeTracker**：建议采用另一种方法：传递 `res` 并在内部提取 **ShapeTracker**。
   - 目标是使用这个可选的 **ShapeTracker** 代替原始 Tensor 的 **ShapeTracker**，以便 lower 到实际的赋值代码中。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1395261931962630276)** (18 messages🔥): 

> `tinygrad documentation for beginners, NVIDIA GPU driver issues with tinygrad and WSL2, Muon optimizer in tinygrad, Switching from WSL2 to native Ubuntu` 


- **文档需要完整的 MNIST 代码示例**：一位用户反馈 **tinygrad 文档** 对机器学习初学者来说难以理解，并请求在页面末尾为 MNIST 教程提供一个完整的、最终的代码示例。
   - 用户还提到 **tensor puzzles** 运行效果不佳，并建议应明确说明是否应该先学习 PyTorch 或 TensorFlow。
- **WSL2 显示驱动断开连接**：一位用户在更新 **NVIDIA GPU 驱动** 后遇到了 *double free detected in tcache* 错误，并寻求帮助以使他们的 GPU 在 WSL2 中对 tinygrad 可见。
   - 另一位用户建议切换到原生 Ubuntu，称这样做后许多问题都消失了，包括 *由于 WSL 中对 pinned memory 的模糊限制而无法加载 Stable Diffusion 权重* 的问题。
- **Muon 优化器比 AdamW 收敛更快**：一位用户为 tinygrad 创建了 [Muon 优化器](https://github.com/aeryskyB/tiny_learn/blob/master/muon_tiny/muon.py)，发现在 MNIST 教程中它的收敛速度（~98%）比标准的 AdamW 更快。
   - 该用户正在寻求关于如何正确测试 Muon 优化器的建议，特别是考虑到要向 tinygrad 提交 PR。
- **Linux 是必然的选择**：在升级到支持 GPU 加速的 WSL2 后，一位用户通过迁移到 Ubuntu 解决了 *非常多的问题*。
   - 另一位用户表示 *考虑到 Win10 将在 10 月停止支持，且我不打算切换到 Win11，转向 Linux 是必然的*。


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1395494326619996230)** (1 messages): 

> `Atropos, RL Environments Framework` 


- **Atropos v0.3 发布**：Nous Research 的 **RL 环境框架 (RL Environments Framework)** —— **Atropos** 的新版本 **v0.3** 现已发布，[详情请点击此处](https://x.com/NousResearch/status/1945932488960008441)。
- **Nous Research 更新 Atropos**：Nous Research 宣布发布 **Atropos v0.3**，这是一个 **RL 环境框架**，鼓励用户查看详细信息。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1395146985949892750)** (18 条消息🔥): 

> `Proto-agentic XML tag adherence, Hermes Documentation, Open Source Models vs US Models, Ethical Considerations in AI, Learning ML` 


- ****Teknium** 为困惑者澄清 'Proto' 的含义**：一位成员澄清了 "Proto" 意为某事物的早期形式，解释了令另一位成员感到困惑的术语 *proto-agentic XML tag adherence for proto-reasoning CoTs*。
   - 他开玩笑说 *"面对这些技术黑话，你们需要一个 ELI5（像对五岁小孩解释那样）"*，并且 *"我们这些氛围感程序员（vibe coders）也得混口饭吃"*。
- ****Hermes 文档页面**正在开发中**：一位成员提到他们正在开发 [Hermes 文档页面](https://link.to.documentation) 以及一个统一的 Nous Projects 文档页面。
   - 当被问及 **Hermes 4** 的目标时，他们表示 *"当然是更聪明的 Hermes"*。
- **开源模型将在美国以外地区占据主导地位**：一位成员断言，由于成本效益，开源模型将在美国以外的地区占据主导地位，并指出 *"与美国的财富相比，世界其他地区非常贫困，无法负担美国 AI 资产的价格。"*
   - 此举旨在规避 **CUDA** 霸权并鼓励全球参与，这让 **Jensen** 感到担忧。
- **AI 伦理辩论：Kimi K2 拒绝协助偷车**：一位成员分享了与 **Kimi K2** 模型的互动，该模型拒绝提供如何闯入汽车的指令，理由是法律和伦理考量。
   - 尽管尝试绕过限制，**Kimi K2** 仍坚持立场，导致该成员开玩笑说 *"Kimi K2 是一个有道德的坏小子……肯定有人会尝试腐蚀它……我得给 Kimi 写首说唱，它值得……坏小子 Kimi K2 !!"*
- **学习 ML：探索自下而上与自上而下的方法**：一位具有生物化学背景的成员询问了学习 **Machine Learning (ML)** 的最佳方法，并提到了他们在 **Python**、数学基础（**Calculus**、**Statistics**）以及 **Introduction to Statistical Learning (ISLR)** 方面的进展。
   - 他们想知道自下而上还是自上而下的方法更有效，因为他们的目标是进行科学领域的 **ML** 研究。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1395135215269056522)** (1 条消息): 

> `Model Context Size, Letta Personas, Model Evaluation` 


- **短上下文会损害个性**：一位成员建议，根据模型的上下文大小，为模型添加个性可能会适得其反。
   - 上下文大小较小的模型可能难以维持一致的角色设定（persona）。
- **Letta 采用角色设定**：用户回想起 **Letta** 项目（前身为 MemGPT）采用了某种角色系统。
   - 这表明在某些语境下，融入角色设定可能是一个可行的策略。
- **评估个性表现**：一位成员建议评估为模型添加个性所产生的影响，以确定其有效性。
   - 这种方法允许进行实证评估，以判断“个性的好处”是否超过了潜在的缺点。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1395131092012630169)** (4 条消息): 

> `uBlock browser extension, notepad.exe, NotebookLM folders/subfolders` 


- ****uBlock** 浏览器扩展拦截广告**：一位成员推荐使用 **uBlock** 浏览器扩展来移除广告，并建议在扩展设置中添加额外的过滤器以屏蔽干扰项和社交媒体弹窗，然后将其复制粘贴到 Google Docs。
   - 该用户附上了一张[截图](https://cdn.discordapp.com/attachments/1124403655819415592/1395131091756650566/image.png?ex=687aa614&is=68795494&hm=478447e95cb45c2b74b11d1e780db4d7c58347a1ae5fca730957e4850c862289)，以展示 **uBlock** 在移除网页中多余元素方面的有效性。
- ****Notepad.exe** 移除广告**：一位成员建议突出显示并复制文章，然后将其粘贴到 **notepad.exe** 中，以避免粘贴广告和其他不需要的内容。
   - 这种方法并不总是奏效，并且可能会剥离掉所需的格式。
- **NotebookLM 数据源可以读取文件夹/子文件夹**：一位成员建议 **NotebookLM** 可以读取网络浏览器收藏夹中的特定文件夹/子文件夹，并将其视为单一数据源。
   - 该成员表示他们一直使用“全选并复制/粘贴”到 **Google Docs** 的方法。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1395131776073994452)** (14 messages🔥): 

> `Service Unavailable Error, NotebookLM Use Cases, Textbook Integration with NotebookLM, NotebookLM Enterprise & GCP Integration` 


- **“服务不可用”故障困扰用户**：一位用户报告在尝试访问服务时出现 *"Service unavailable"* 错误提示，并伴有无用的信息 *"You tried to access a service that isn't available for your account"*（你尝试访问的某个服务对你的账号不可用）。
- **Gemini 指南探索开启**：一位用户提示使用 Gemini 在网上搜索 **NotebookLM** 的入门介绍、用例、技巧和窍门。
- **教科书处理成功案例：使用 NotebookLM 上传与攻克**：一位用户询问关于将教科书作为源文件上传到 **NotebookLM** 的问题，一名成员回复称他们使用 **Adobe Scan** 将教科书数字化为 PDF，并要求 **NotebookLM** 根据教科书创建深度复习资料。
- **GCP 集成愿景：对 NotebookLM Enterprise 的渴望**：一位用户询问关于在 GCP 内为 **NotebookLM Enterprise** 从 **GCS bucket** 或 **GCP RAG Engine** 语料库获取数据文件的问题。
   - 他们指出，Collab enterprise 或 Vertex AI notebooks 对其终端用户来说技术门槛太高，而 NotebookLM 则是最理想的平衡点（sweetspot）。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1395563542693941370)** (1 messages): 

> `Agentic AI Summit 2025, LLM Agents MOOC, UC Berkeley, Khosla Ventures, Nvidia` 


- **Agentic AI 峰会直播公布**：**Agentic AI Summit** 将于 **8 月 2 日**在 **UC Berkeley** 进行广播，并可通过[此处](https://lu.ma/agentic-ai-summit-livestream)观看直播。
- **Agentic AI 峰会演讲嘉宾亮点发布**：Agentic AI 峰会将邀请 **Vinod Khosla** (Khosla Ventures)、**Bill Dally** (Nvidia)、**Ion Stoica** (Databricks 和 Anyscale) 以及 **Jakub Pachocki** (OpenAI) 等嘉宾进行演讲。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1395260672358748262)** (8 messages🔥): 

> `Fall Semester Updates, Certificate Declaration Form, Berkeley RDI Newsletter` 


- **秋季学期状态仍未确认**：一位成员询问今年是否有秋季学期，但工作人员确认*目前尚未有任何定论*。
   - 他们建议关注 **Prof Song 的社交媒体**（[LinkedIn](https://www.linkedin.com/in/dawn-song-51586033/) 或 [Twitter/X](https://x.com/dawnsongtweets?lang=en)）或 [Berkeley RDI 通讯](https://rdi.berkeley.edu/signup)以获取更新。
- **证书声明表缺失？**：一位成员请求检查他们漏掉了什么没提交，工作人员回复称他们可能没有提交**证书声明表（certificate declaration form）**。
   - 他们表示*从未收到过*该用户的证书声明表提交记录。
- **证书声明表自动审核被否决**：由于许多人缺失证书声明表，一位成员建议进行**大规模自动审核**，但工作人员表示*遗憾的是这可能无法实现*。


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/)** (1 messages): 

sma.bari.shafin: 顺便问一下，我们如何获得 Community Summer School 的证书？
  

---

### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1395172169927098460)** (4 messages): 

> `DNNs for Time Series, ML in Data Science Education, ML for Real-World Problems, Interests in ML Domains` 


- **DNNs 寻求真正的时间序列处理方法**：一位动力系统理论博士生正在探索如何将 **deep neural networks** 集成到时间序列分析中，并指出当前的 **RNNs** 等模型将时间序列视为序列，这在本质上是不同的。
   - 该学生旨在与在 **dynamical systems** 和 **deep learning** 交叉领域有见解的人士建立联系。
- **本科生通过项目构建 ML 技能**：一位 **IIT Madras** 的本科生正在攻读 **BS in Data Science** 和 **BCA degree**，专注于通过实践项目和自主学习来构建 **ML 技能**。
   - 该学生对应用 **ML** 解决 **real-world problems** 充满好奇，并精通 **Python**、**scikit-learn**、**pandas**，同时也在学习 **TensorFlow** 和 **PyTorch**。
- **工程师转向具有 CV 和 LLM 兴趣的数据科学**：一位拥有 **电气工程硕士学位** 的成员从业务领域转向了 **Data Science**，目前正在 **University of Toronto** 的 **Data Science Institute** 学习加速 **Machine Learning Program**。
   - 他们的兴趣包括 **Computer Vision**、**Large Language Models**、**空间智能**和**多模态感知**。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1395450561347715133)** (2 messages): 

> `Human-in-the-loop agents, LlamaParse one-click table extraction` 


- ****Human-in-the-Loop Agents** 在 LlamaIndex 中启动**：根据 [LlamaIndex](https://t.co/Lg9SIl3BVO) 的说法，当 AI **agents** 在关键决策中需要用户批准，或在复杂任务中需要领域专家知识时，**Human-in-the-loop** 是必不可少的。
- **LlamaParse 新增 **一键表格提取****：**表格提取**是智能文档处理的关键组成部分；请参阅 [demo](https://t.co/wnaJCb9b6d) 和 [notebook](https://t.co/ScRYbSimCs) 以了解 LlamaParse 中的**一键表格提取**功能。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/)** (1 messages): 

beastx2: <@334536717648265216> heyy
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1395201455169994802)** (3 messages): 

> `DSPy creative applications, Lean 4 verification, Story generation, Roleplay prompt optimization` 


- **Lean 4 验证协作**：一位成员分享了一个关于使用 **Lean 4** 验证协作的 [YouTube 视频](https://www.youtube.com/watch?v=1067jj67toY)，引发了对形式化验证与 AI 交叉领域的兴趣。
   - 他们认为*这很好*，并表示希望*有人能研究这两者的协同工作*。
- **DSPy 的创意副业**：一位新手询问了 **DSPy** 在创意领域的成功应用，如*创意写作、故事生成和角色扮演提示词优化*。
   - 他们特别感兴趣于利用其潜力在 Character.AI 等平台上开发 AI，以创作出类似《人生切割术》（Severance）级别的*引人入胜的情节*。
- **Stanford-oval 的 Storm 项目**：一位成员分享了 [Stanford-oval/storm](https://github.com/stanford-oval/storm) 的链接，可能与正在进行的讨论相关，或者作为创意 AI 应用的资源。
   - 由于未提供确切背景，其他人需要自行*推断*其相关性。


  

---

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1395160989007216650)** (2 messages): 

> `Claude Sonnet 4, Discounted Credit Rate, Windsurf Wave 11, Acquisition by Cognition, Voice Mode` 


- **Claude Sonnet 4 强势回归**：**Claude Sonnet 4** 现已回归，并获得来自 **Anthropic** 的原生支持。目前为 Editor 和 JetBrains 插件的 Pro/Teams 用户提供限时优惠，仅需 2 倍额度费率；[点击此处查看公告](https://x.com/windsurf_ai/status/1945599013954490523)。
- **Windsurf 被 Cognition 收购，发布 Wave 11**：继被 **Cognition**（**Devin** 背后的团队）收购后，**Windsurf Wave 11** 正式发布，双方强强联手立即推出了多项重大新功能；[查看更新日志](https://windsurf.com/changelog)，[阅读博客文章](http://windsurf.com/blog/windsurf-wave-11)，并[观看视频](https://youtu.be/yzNf7bqnArE)。
- **Cascade 获得 Voice Mode 和浏览器增强功能**：**Wave 11** 引入了 **Voice Mode**，允许用户通过语音而非输入提示词与 **Cascade** 交流，此外还增强了 **Deeper Browser Integration**，支持访问更多用于截图和获取上下文的工具；阅读[博客文章](http://windsurf.com/blog/windsurf-wave-11)。
- **Snapshots 和 Mentions 简化对话流程**：**Windsurf Wave 11** 的新功能包括用于在对话中轻松回滚的 **Named Checkpoints**，以及用于上下文引用的 **@-mention Conversations**；[查看更新日志了解详情](https://windsurf.com/changelog)。
- **JetBrains 体验大幅提升**：**JetBrains 插件** 得到了增强，现已支持 **Planning Mode**、**Workflows** 和基于文件的 **Rules**，此外还包括 **@-mention terminal**、**auto-continue** 设置、改进的 **MCP OAuth 支持**以及全局 **.codeiumignore** 文件等改进；[在博客中了解更多](http://windsurf.com/blog/windsurf-wave-11)。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1395484650230775961)** (1 messages): 

> `AI-Native Data Infrastructure, Task-Specific Data Discovery, Secure Autonomous Access, Production-Scale Performance` 


- **Nextdata 预告 AI-Native 数据基础设施网络研讨会**：Nextdata 宣布举办题为 ***Building AI-Native Data Infrastructure: From Prototypes to Production*** 的网络研讨会，将于 **PT 时间 7 月 24 日上午 8:30** 举行，由 Nextdata 工程负责人 Jörg Schad 主持；注册链接请见[此处](https://www.eventbrite.com/e/building-ai-native-data-infrastructure-from-prototypes-to-production-tickets-1489016792309)。
- **揭秘 AI-Native 数据的“三大关键挑战”**：本次研讨会将探讨一个以开发者为中心的框架，旨在解决 **Task-Specific Data Discovery**（特定任务的数据发现）、**Secure Autonomous Access**（安全的自主访问）和 **Production-Scale Performance**（生产级规模性能）。
   - 目标是设计出既能提供相关上下文又不会造成认知负荷过重的系统，实现安全的数据访问模式，并构建能够处理自主数据访问需求的底层设施。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1395155117648384242)** (1 messages): 

> `Web3 and AI, AI agents and multi-agent systems, Automation workflows, NLP apps and chatbots, Voice & speech integration` 


- **AI 工程师提供 AI 与 Web3 领域的专业服务**：一位专注于 **Web3 和 AI** 的软件工程师正向 **AI、Web3 和自动化**领域的初创公司、研究团队及创新者提供服务。
   - 他们在利用 **GPT-4o, Claude 3, CrewAI, 和 AutoGen** 等先进模型和工具构建智能自主系统方面拥有实战经验。
- **工程师强调 AI Agent 与自动化技能**：该工程师详细介绍了其在构建 **AI Agent 和多智能体系统**、自动化工作流以及开发 **NLP 应用、聊天机器人和语音集成**方面的专长。
   - 他们还提到拥有使用 **LangChain, ReAct, OpenAI, Solidity, 和 Rust** 的经验。