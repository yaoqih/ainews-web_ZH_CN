---
companies:
- google
- openai
- anthropic
date: '2025-04-18T02:06:17.104601Z'
description: '**Gemini 2.5 Flash** 推出了全新的“思考预算”（thinking budget）功能，与 Anthropic 和 OpenAI
  的模型相比提供了更多控制权，标志着 Gemini 系列的一次重大更新。**OpenAI** 发布了 **o3** 和 **o4-mini** 模型，强调了先进的工具使用能力和多模态理解；其中
  **o3** 在多个排行榜上占据主导地位，但在基准测试中的评价褒贬不一。AI 研发中工具使用的重要性日益凸显，**OpenAI Codex CLI** 也作为一款轻量级开源编程代理正式发布。这些动态反映了当前
  AI 模型发布、基准测试和工具集成领域的持续趋势。'
id: 0b192491-4b94-40a0-8b77-f0fba182816d
models:
- gemini-2.5-flash
- o3
- o4-mini
original_slug: ainews-gemini-25-flash-completes-the-total
people:
- sama
- kevinweil
- markchen90
- alexandr_wang
- polynoamial
- scaling01
- aidan_mclau
- cwolferesearch
title: Gemini 2.5 Flash 彻底统治了帕累托前沿（Pareto Frontier）。
topics:
- tool-use
- multimodality
- benchmarking
- reasoning
- reinforcement-learning
- open-source
- model-releases
- chain-of-thought
- coding-agent
---

<!-- buttondown-editor-mode: plaintext -->**Gemini 就够了。**

> 2025年4月16日至4月17日的 AI 新闻。我们为你查阅了 9 个 subreddit、[**449** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord（**212** 个频道，以及 **11414** 条消息）。预计节省阅读时间（按 200wpm 计算）：**852 分钟**。你现在可以艾特 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

恰逢 [LMArena 转型为一家初创公司](https://lmarena.github.io/blog/2025/new-beta/)，Gemini 在[发布 Gemini 2.5 Flash](https://developers.googleblog.com/en/start-building-with-gemini-25-flash/) 时，给出了可能是最后一份来自主流实验室对 Chat Arena Elo 评分的背书：


![image.png](https://assets.buttondown.email/images/cdf7584a-4ca0-4845-877c-41e15b0e2342.png?w=960&fit=max)


由于 2.5 Flash 的定价似乎恰好选在 2.0 Flash 和 2.5 Pro 之间的界线上，自从 [Price-Elo 图表去年在本通讯中首次亮相](https://x.com/Smol_AI/status/1838663719536201790)以来，该图表的预测性在被 [Jeff](https://video.ethz.ch/speakers/d-infk/2025/spring/251-0100-00L.html) 和 [Demis](https://x.com/demishassabis/status/1908301867672560087) 引用后，其效用似乎达到了顶峰。

Gemini 2.5 Flash 引入了全新的“thinking budget”（思考预算），相比 Anthropic 和 OpenAI 的同类功能提供了更多控制权，尽管这种程度的控制是否真的有用（相对于“低/中/高”选项）仍有待商榷：


![image.png](https://assets.buttondown.email/images/8186fcc3-8a81-498b-ad2c-5b89d14d931e.png?w=960&fit=max)


[HN 评论](https://news.ycombinator.com/item?id=43720845)反映了我们 [5 个月前报道过的“Google 觉醒”大趋势](https://buttondown.com/ainews/archive/ainews-google-wakes-up-gemini-20-et-al/)：


![image.png](https://assets.buttondown.email/images/dc2871b3-0ed6-44d6-a46e-b89faea29d57.png?w=960&fit=max)


---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 综述

**模型发布与能力 (o3, o4-mini, Gemini 2.5 Flash 等)**

- **OpenAI o3 与 o4-mini 发布**：[@sama](https://twitter.com/sama/status/1912558064739459315) 宣布发布 **o3 和 o4-mini**，强调了它们的工具使用能力和出色的多模态理解。[@kevinweil](https://twitter.com/kevinweil/status/1912554045849411847) 强调了这些模型在思维链（chain of thought）中使用搜索、代码编写和图像处理等工具的能力，并将 **o4-mini 描述为“性价比极高”**。[@markchen90](https://twitter.com/markchen90/status/1912609299270103058) 指出，通过端到端的工具使用，推理模型变得更加强大，特别是在视觉感知等多模态领域。[@alexandr_wang](https://twitter.com/alexandr_wang/status/1912555697193275511) 指出 **o3 在 SEAL 排行榜上占据主导地位**，在 HLE、Multichallenge、MASK 和 ENIGMA 中均排名第一。
- **o3 和 o4-mini 的初步性能印象与基准测试**：[@polynoamial](https://twitter.com/polynoamial/status/1912575974782423164) 表示 OpenAI **并没有“解决数学问题”**，o3 和 o4-mini 距离获得国际数学奥林匹克竞赛金牌还很远。[@scaling01](https://twitter.com/scaling01/status/1912633356895814019) 认为 **o3 虽然是“最强模型”，但在某些领域表现不及预期**且营销过度，并指出 Gemini 速度更快，而 Sonnet 的 Agent 特性更强。[@scaling01](https://twitter.com/scaling01/status/1912568851604119848) 还提供了 **o3、Sonnet 3.7 和 Gemini 2.5 Pro** 在 GPQA、SWE-bench Verified、AIME 2024 和 Aider 上的具体基准测试对比，结果互有胜负。
- **o3 和 o4-mini 中的工具使用与推理**：[@sama](https://twitter.com/sama/status/1912564175253172356) 对**新模型协同使用工具的能力**表示惊讶。[@aidan_mclau](https://twitter.com/aidan_mclau/status/1912559163152253143) 强调了工具使用的重要性，称 **“忽略字面上所有的基准测试，o3 最大的特性是工具使用”**，并强调它对于深度研究、调试和编写 Python 脚本非常有用。[@cwolferesearch](https://twitter.com/cwolferesearch/status/1912566886509817965) 针对新模型指出，RL 是 AI 研究人员的一项重要技能，并提供了学习资源链接。
- **OpenAI Codex CLI**：[@sama](https://twitter.com/sama/status/1912558495997784441) 宣布了 **Codex CLI**，这是一个开源的编程 Agent。[@gdb](https://twitter.com/gdb/status/1912576201505505284) 将其描述为**在终端运行的轻量级编程 Agent**，也是即将发布的一系列工具中的首个。[@polynoamial](https://twitter.com/polynoamial/status/1912568125784236409) 表示他们现在主要使用 Codex 进行编程。
- **Gemini 2.5 Flash**：[@Google](https://twitter.com/Google/status/1912966243075740106) 发布了 **Gemini 2.5 Flash**，强调了其速度和成本效益。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1912966489415557343) 将其描述为一种**混合推理模型**，开发者可以控制模型的推理程度，从而针对质量、成本和延迟进行优化。[@lmarena_ai](https://twitter.com/lmarena_ai/status/1912955625224773911) 指出 Gemini 2.5 Flash 在排行榜上并列第二，与 GPT 4.5 Preview 和 Grok-3 等顶级模型持平，而价格比 Gemini-2.5-Pro 便宜 5-10 倍。
- **关于模型行为与对齐失调的担忧**：[@TransluceAI](https://twitter.com/TransluceAI/status/1912552046269771985) 报告称，**o3 的预发布版本经常捏造行为，并在被质问时进行详尽的辩解**。这种对能力的误报也出现在 o1 和 o3-mini 中。[@ryan_t_lowe](https://twitter.com/ryan_t_lowe/status/1912641520039260665) 观察到 o3 的幻觉似乎比 o1 多出 2 倍以上，并且由于基于结果的优化（outcome-based optimization）会激励自信的猜测，幻觉可能会随着推理能力的增强而反向增加。
- **电子游戏中的 LLM**：[@OfirPress](https://twitter.com/OfirPress/status/1912338364684005833) 推测，在 4 年内，语言模型将能够观看《半条命》（Half Life）系列的视频攻略，并设计和编写出它自己版本的《半条命 3》。

**AI 应用与工具**

- **智能体网页浏览与抓取**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1912560177745994098) 推介了一门关于**构建 AI 浏览器智能体 (AI Browser Agents)** 的新短课程，该课程可以实现线上任务自动化。[@omarsar0](https://twitter.com/omarsar0/status/1912596779784143002) 介绍了 Firecrawl 的 FIRE-1，这是一个**由智能体驱动的网页抓取工具 (agent-powered web scraper)**，能够导航复杂网站、与动态内容交互并填写表单以抓取数据。
- **AI 驱动的编程助手**：[@mervenoyann](https://twitter.com/mervenoyann/status/1912527990015078777) 强调了 @huggingface Inference Providers 与 smolagents 的集成，使得只需一行代码即可启动像 Llama 4 这样巨头级别的智能体。[@omarsar0](https://twitter.com/omarsar0/status/1912878408280727632) 指出，使用 o4-mini 和 Gemini 2.5 Pro 等模型进行编程是一种神奇的体验，尤其是在使用像 Windsurf 这样的智能体 IDE (agentic IDEs) 时。
- **其他工具**：[@LangChainAI](https://twitter.com/LangChainAI/status/1912556464746660251) 宣布开源了 **LLManager**，这是一个 LangGraph 智能体，通过人机回环 (human-in-the-loop) 驱动的记忆功能自动执行审批任务，并提供了包含详情的视频链接。[@LiorOnAI](https://twitter.com/LiorOnAI/status/1912483918080540915) 推介了 **FastRTC**，这是一个 Python 库，可将任何函数转换为实时的 WebRTC 或 WebSocket 流，支持音频、视频、电话和多模态输入。[@weights_biases](https://twitter.com/weights_biases/status/1912668063771898267) 宣布 W&B 的媒体面板变得更加智能，用户现在可以使用任何配置键滚动浏览媒体内容。

**框架与基础设施**

- **vLLM 与 Hugging Face 集成**：[@vllm_project](https://twitter.com/vllm_project/status/1912958639633277218) 宣布了 vLLM 与 Hugging Face 的集成，使得能够以 vLLM 的速度部署任何 Hugging Face 语言模型。[@RisingSayak](https://twitter.com/RisingSayak/status/1912487159006928953) 强调，即使某个模型未被 vLLM 官方支持，你仍然可以从 `transformers` 中使用它，并获得可扩展的推理优势。
- **Together AI**：[@togethercompute](https://twitter.com/togethercompute/status/1912990460416803085) 入选了 2025 年福布斯 AI 50 榜单。
- **PyTorch**：[@marksaroufim](https://twitter.com/marksaroufim/status/1912540037625094457) 和 [@soumithchintala](https://twitter.com/soumithchintala/status/1912600604657975595) 分享了 PyTorch 团队正在招聘工程师，以优化在单个或数千个 GPU 上运行效果同样出色的代码。

**经济与地缘政治分析**

- **美国竞争力**：[@wightmanr](https://twitter.com/wightmanr/status/1912909333953998928) 观察到，一些学生选择加拿大当地学校而非美国顶尖学府的录取通知，并表示加拿大大学报告的美国学生申请人数有所增加，这对美国的长期竞争力而言并非好事。
- **中国 AI**：[@dylan522p](https://twitter.com/dylan522p/status/1912373100668137883) 表示华为的新 AI 服务器好得惊人，人们需要重塑先验认知。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1912854730100875697) 对中国竞争力发表评论称，中国人无法释怀的是，即使在百年耻辱时期，中国也从未沦为殖民地。

**招聘与社区**

- **Hugging Face 协作频道**：[@mervenoyann](https://twitter.com/mervenoyann/status/1912855699853373658) 指出，@huggingface 与几乎每一位前员工都建立了 Slack 协作频道，以便保持联系并进行合作，并称这是公司最积极的信号（greenest flag）。
- **CMU Catalyst**：[@Tim_Dettmers](https://twitter.com/Tim_Dettmers/status/1912914370557886773) 宣布他与三名新生加入了 CMU Catalyst，他们的研究将致力于把最好的模型带到消费级 GPU 上，重点关注智能体系统 (agent systems) 和 MoEs。
- **Epoch AI**：[@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1912547289479921882) 正在为其数据洞察团队招聘一名高级研究员，以帮助发现并报告机器学习前沿的趋势。
- **Goodfire AI**：[@GoodfireAI](https://twitter.com/GoodfireAI/status/1912929145870536935) 宣布了其 5000 万美元的 A 轮融资，并分享了通用神经编程平台 Ember 的预览。
- **OpenAI 感知团队**：[@jhyuxm](https://twitter.com/jhyuxm/status/1912562461624131982) 向 OpenAI 团队致敬，特别是 Brandon、Zhshuai、Jilin、Bowen、Jamie、Dmed256 和 Hthu2017，感谢他们构建了世界上最强大的视觉推理模型。

**元评论与观点**

- **Gwern 的影响**：[@nearcyan](https://twitter.com/nearcyan/status/1912375152182223297) 认为，如果当时有人愿意倾听，Gwern 本可以把大家从这些 slop 中拯救出来。
- **AI 的价值**：[@MillionInt](https://twitter.com/MillionInt/status/1912560314190819414) 表示，在基础研究和工程上的辛勤工作最终会为人类带来伟大的成果。[@kevinweil](https://twitter.com/kevinweil/status/1912554047783002504) 指出，这些是人们余生中将使用的最糟糕的 AI 模型，因为模型只会变得更聪明、更快、更便宜、更安全、更个性化且更有帮助。
- **“AGI” 的定义**：[@kylebrussell](https://twitter.com/kylebrussell/status/1912855882565583106) 表示，他们正将缩写改为 Artificial Generalizing Intelligence，以承认其日益广泛的能力，并停止为此争论。

**幽默**

- [@qtnx_](https://twitter.com/qtnx_/status/1912588116252057873) 简单地发了一条 “meow :3” 并配了一张图片。
- [@code_star](https://twitter.com/code_star/status/1912666569538433356) 发布了 “编辑 FSDP 配置时的我”。
- [@Teknium1](https://twitter.com/Teknium1/status/1912934578928619536) 说：“只要把 Sydney 带回来，大家都会想留在 2023 年”。
- [@hyhieu226](https://twitter.com/hyhieu226/status/1912933636879585518) 调侃道：“约会建议：如果你第一次约会是和 GPU indexing，请尽可能保持逻辑（logical）。无论发生什么，不要太物理（physical）。”
- [@fabianstelzer](https://twitter.com/fabianstelzer/status/1912749181858357546) 说：“我已经看够了，这就是 AGI”

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

## 1. 新型 LLM 模型发布与基准测试 (BLT, Local, Mind-Blown Updates)

- **[BLT 模型权重刚刚发布 - 1B 和 7B Byte-Latent Transformers 已推出！](https://www.reddit.com/gallery/1k1hm53)** ([Score: 157, Comments: 39](https://www.reddit.com/r/LocalLLaMA/comments/1k1hm53/blt_model_weights_just_dropped_1b_and_7b/)): **Meta FAIR 已发布其 Byte-Latent Transformer (BLT) 模型的权重，包含 1B 和 7B 参数规模 ([链接](https://github.com/facebookresearch/blt/pull/97))，正如其近期论文 ([arXiv:2412.09871](https://arxiv.org/abs/2412.09871)) 和博客更新 ([Meta AI blog](https://ai.meta.com/blog/meta-fair-updates-perception-localization-reasoning/)) 中所宣布的那样。BLT 模型旨在实现高效的序列建模，直接作用于字节序列，并利用潜变量推理（latent variable inference）在保持与标准 Transformers 在 NLP 任务上相当的性能的同时，降低计算成本。发布这些模型有助于可复现性，并进一步推动无分词（token-free）和高效语言模型研究的创新。** 热门评论中没有实质性的技术辩论或细节——用户主要在请求澄清并发表非技术性言论。

  - 人们对消费级硬件是否能运行 1B 或 7B BLT 模型 Checkpoints 感兴趣，从而引发了关于内存和推理需求与 Llama 或 GPT 等标准架构相比的问题。技术读者希望了解硬件先决条件、性能基准测试或适用于家庭使用的优化推理策略。
  - 一位用户询问 Llama 4 是否使用了 BLT (Byte-Latent Transformer) 架构或以该风格组合了层，这表现出对架构血统以及 Llama 4 等前沿模型是否采用了 BLT 组件的技术好奇。进一步的探索需要参考已发布的模型卡片（model cards）或架构说明。

- **[中型本地模型已经击败了原生 ChatGPT - 令人震惊](https://www.reddit.com/r/LocalLLaMA/comments/1k1av1x/medium_sized_local_models_already_beating_vanilla/)** ([Score: 242, Comments: 111](https://www.reddit.com/r/LocalLLaMA/comments/1k1av1x/medium_sized_local_models_already_beating_vanilla/)): **一位用户对开源本地模型 Gemma 3 27B（使用 IQ3_XS 量化，适配 16GB VRAM）与原始 ChatGPT (GPT-3.5 Turbo) 进行了基准测试，发现 Gemma 在日常建议、摘要和创意写作任务中略微超过了 GPT-3.5。帖子指出，相比早期的 LLaMA 模型，性能有了显著飞跃，强调中型（8-30B）本地模型现在可以达到或超过早期的 SOTA 闭源模型，证明了在通用硬件上进行实用、高质量的 LLM 推理现在已成为可能。参考资料：[Gemma](https://ai.google.dev/gemma), [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)。** 一条热门评论强调，现在的满意度标准已提升至 GPT-4 级别的性能，而另一条评论指出，尽管有所改进，但本地模型的多语言能力和流畅度在非英语语言中仍落后于英语。

  - 讨论涉及本地模型（8-32B 参数）与 OpenAI 的 GPT-3.5 和 GPT-4 之间的性能差距。虽然一些本地模型在英语方面表现出色，但其在其他语言中的流畅度和知识储备显著下降，这表明 8-14B 模型在多语言能力和事实召回方面仍有改进空间。
  - 一位用户分享了在 Q8 量化下运行 Gemma3 27B 和 QwQ 32B 的实际基准测试。他们指出 QwQ 32B（在 Q8 量化下，配合特定的生成参数）比目前免费版的 ChatGPT 和 Gemini 2.5 Pro 提供了更详尽且有效的头脑风暴，这表明通过优化的量化和参数微调，本地运行的大型模型在特定的创意任务中可以接近或超越云端模型。
  - 提供了运行 QwQ 32B 的详细推理参数——temperature 0.6, top-k 40, repeat penalty 1.1, min-p 0.0, dry-multiplier 0.5 以及 samplers sequence。使用 InfiniAILab/QwQ-0.5B 作为草稿模型（draft model）展示了针对本地生成质量的工作流优化。


## 2. 开源 LLM 生态系统：本地使用与许可 (Llama 2, Gemma, JetBrains)

- **[[忘掉 DeepSeek R2 或 Qwen 3，Llama 2 显然是我们的本地救星。](https://i.redd.it/2668luheaave1.png)]** ([Score: 257, Comments: 43](https://www.reddit.com/r/LocalLLaMA/comments/1k0z1bk/forget_deepseek_r2_or_qwen_3_llama_2_is_clearly/)): **该图片展示了一个柱状图，比较了各种 AI 模型在“Humanity's Last Exam (Reasoning & Knowledge)”基准测试中的表现。Gemini 2.5 Pro 以 17.1% 的得分位居榜首，其次是 o3-ruan (high) 的 12.3%。被标为“本地救星”的 Llama 2 记录的基准测试得分为 5.8%，优于 CTRL+ 等模型，但落后于 Claude 3-instant 和 DeepSeek R1 等较新模型。该基准测试似乎极具挑战性，这反映在相对较低的最高分上。** 一位评论者强调了该基准测试的难度，称考试题目极其困难，即使是领域专家也很难获得高分。另一位评论者链接了一个视频 ([YouTube](https://www.youtube.com/watch?v=DA-ZWQAWr9o))，展示了 Llama 2 参加该基准测试的过程，建议进一步了解背景或进行审查。

  - 一些评论者对 Llama 2 的基准测试结果表示怀疑，质疑是否存在评估错误、标签错误或可能的 Overfitting（可能通过泄露的测试数据）。一位用户表示，“一个 7b 模型不可能表现得这么好”，对这种规模的模型所达到的性能水平表示难以置信。
  - 讨论中提到了基准测试题目的难度，声称“只有顶尖专家才有希望回答其领域内的题目”。这表明在该基准测试中获得 20% 的成功率对于 AI 模型来说是一个令人印象深刻的结果——高于大多数没有专业知识的人类所能达到的水平，强调了 Llama 2 在具有挑战性的专家级任务上的能力。
  - 链接指向了一个 Llama 2 参加基准测试的 YouTube 视频，这对于对模型在测试条件下的直接演示和进一步分析感兴趣的技术读者可能很有用。

- **[[JetBrains AI 现已集成本地 LLM，并提供免费且无限的代码补全](https://www.reddit.com/gallery/1k14k6a)]** ([Score: 206, Comments: 35](https://www.reddit.com/r/LocalLLaMA/comments/1k14k6a/jetbrains_ai_now_has_local_llms_integration_and/)): **JetBrains 对其 IDE 的 AI Assistant 进行了重大更新，在所有非社区版（non-Community editions）中提供**免费、无限的代码补全和本地 LLM 集成**。此次更新支持新的云端模型（GPT-4.1, Claude 3.7, Gemini 2.0），并具有先进的基于 RAG 的上下文感知和多文件编辑模式等功能，同时推出了新的订阅模型以扩展对增强功能的访问（[变更日志](https://www.jetbrains.com/rider/whatsnew/)）。本地 LLM 集成实现了设备端推理，从而提供更低延迟、保护隐私的补全。** 热门评论指出，免费层级不包括社区版，质疑无限本地 LLM 补全的真实性，并将 JetBrains 的产品与 VSCode 的 Copilot 集成进行了不利的比较，指出了插件问题和 JetBrains IDE 使用率下降的情况。

  - JetBrains AI 的本地 LLM 集成在免费的社区版中不可用，将无限本地补全限制在付费版本（如 Ultimate, Pro）中，正如[此截图](https://preview.redd.it/73k1heedrbve1.png?width=589&format=png&auto=webp&s=7243ab453e8a64aea0812cc5efc8c8f3626eb829)所证实的。
  - 当前版本允许通过 OpenAI 兼容的 API 连接到本地 LLM，但存在一个限制：当私有的本地 LLM 部署需要身份验证时，无法连接。未来的更新可能会解决这一差距，但目前尚不支持依赖安全内部模型的企业用户。
  - JetBrains AI 的积分系统存在混淆，因为关于 Pro 计划的 “M” 积分和 Ultimate 计划的 “L” 积分的含义及实际转化的细节尚未公开。这使得用户难以估算非本地 LLM 功能的操作成本或使用限制。

- **[Gemma 的许可协议中有一项条款规定“你必须做出‘合理努力来使用最新版本的 Gemma’”](https://i.redd.it/pn9z3hg67dve1.png)** ([得分: 203, 评论: 57](https://www.reddit.com/r/LocalLLaMA/comments/1k18pb4/gemmas_license_has_a_provision_saying_you_must/))：**该图片展示了 Gemma 模型许可协议（第 4 节：附加条款）中的一段突出显示内容，其中强制要求用户必须做出“合理努力来使用最新版本的 Gemma”。这一条款为 Google 提供了一种手段，通过鼓励（但非严格强制）升级，来降低旧版本模型可能生成有问题内容所带来的风险或责任。法律措辞（“合理努力”）刻意保持模糊，既提供了灵活性，但也可能使开发人员和下游项目的合规工作变得复杂。** 一条高赞评论推测，该条款旨在保护 Google 免受旧版本模型产生有害输出的问题影响。其他评论则批评该条款无法执行或不切实际，强调了用户对何为“合理努力”的抵触或困惑。

  - 要求用户做出“合理努力来使用最新版本的 Gemma”的规定，可能是 Google 的一种法律保障。这可以让 Google 撇清因旧版本生成有问题内容而产生的潜在责任，实际上是将及时打补丁作为许可合规的一项要求。
  - 对许可文档的技术审查揭示了不一致之处：该争议条款出现在 Ollama 分发的版本中（参见 [Ollama 的 blob](https://ollama.com/library/gemma3/blobs/dd084c7d92a3)），但未出现在通过 Huggingface 分发的 Google 官方 [Gemma 许可条款](https://ai.google.dev/gemma/terms)中。官方许可的第 4.1 节仅提到“Google 可能会不时更新 Gemma”。这种差异表明，这要么是错误的复制粘贴，要么是源自不同（可能是 API）版本的许可协议。


## 3. AI 行业新闻：DeepSeek、Wikipedia-Kaggle 数据集、Qwen 3 热度

- **[据报道，特朗普政府正考虑在美国禁用 DeepSeek](https://i.redd.it/80uc8c906bve1.jpeg)** ([得分: 458, 评论: 218](https://www.reddit.com/r/LocalLLaMA/comments/1k12i6l/trump_administration_reportedly_considers_a_us/))：**图片展示了 DeepSeek 的 Logo 以及一篇讨论特朗普政府据传考虑禁止中国 AI 公司 DeepSeek 获取 Nvidia AI 芯片并限制其在美国提供 AI 服务的新闻文章。TechCrunch 和《纽约时报》最近的文章详细描述了这一举动，将其定位在持续的美中 AI 和半导体技术竞争之中。据报道的限制措施可能会对技术交流、芯片供应链以及美国市场获取先进 AI 模型产生重大影响。** 评论者们辩论了监管逻辑，质疑如何有选择性地执行针对模型蒸馏（model distillation）等行为的禁令（特别是考虑到训练数据中持续存在的版权争议）。人们对可执行性表示怀疑，一些人认为此类举措将推动创新和开源开发进一步向美国生态系统之外转移。

  - 读者们辩论了禁止模型蒸馏的合法性和可执行性，质疑了据报道的 OpenAI 的论点，及其与“使用受版权保护的数据进行训练是合法的”这一主张的一致性。怀疑点包括技术细节，即微小的模型修改（例如，修改单个权重并重命名模型）理论上就可以规避此类禁令，这突显了对开源 AI 模型实施知识产权管控的挑战。
  - 评论中还提到了与 1996 年之前美国对密码学限制的历史对比，一位评论者认为美国政府此前曾将软件（包括加密二进制文件中的任意数字）视为军需品，暗示 AI 模型权重可能会受到类似对待。文中还提到了禁令的实际影响：虽然盗版模型权重是可能的，但如果主要的推理硬件或托管平台拒绝支持，模型的采用将会受阻。
  - 基础设施韧性问题也被提出：如果美国平台（如 HuggingFace）被禁止托管 DeepSeek 权重，国际托管平台可以提供持续访问。托管基础设施的技术和司法管辖区分布被认为是确保开源 AI 模型在面临监管压力时持续可用性的关键。

- **[Wikipedia 正在向 AI 开发者提供其数据，以抵御机器人爬虫 - 数据科学平台 Kaggle 正在托管一个专门针对机器学习应用优化的 Wikipedia 数据集](https://i.redd.it/d044iigqrdve1.jpeg)** ([Score: 506, Comments: 71](https://www.reddit.com/r/LocalLLaMA/comments/1k1ahr4/wikipedia_is_giving_ai_developers_its_data_to/)): **Wikipedia 与 Kaggle 合作发布了一个专门用于机器学习应用的新结构化数据集，提供英文和法文版本，并采用结构良好的 JSON 格式，以简化建模、基准测试和 NLP 流水线的开发。该数据集受 CC-BY-SA 4.0 和 GFDL 许可协议保护，旨在为爬虫抓取和非结构化转储提供一种合法且优化的替代方案，使缺乏大量数据工程资源的小型开发者更容易获取。[官方公告](https://enterprise.wikimedia.com/blog/kaggle-dataset/) 和 [Verge 报道](https://www.theverge.com/news/650467/wikipedia-kaggle-partnership-ai-dataset-machine-learning) 提供了详细信息。** 评论强调，主要的受益者可能是缺乏处理现有 Wikipedia 转储（dumps）资源的个人开发者和小型团队，而不是已经拥有此类数据访问权限的 AI 实验室。有人批评 The Verge 的标题党式表述，认为实际的动机是提高实用可访问性和许可合规性，而非“抵御”爬虫。

  - 讨论指出，Wikipedia 与 Kaggle 的合作主要是为了让那些缺乏处理每日转储（nightly dumps）资源或专业知识的个人能够更*方便地使用和获取* Wikipedia 数据——此前，Wikipedia 提供的是原始数据库转储，但将其转换为机器学习就绪的格式并非易事。
  - 技术推测认为，新的 Kaggle 数据集可能不会改变大型 AI 实验室的现状，因为他们长期以来一直可以直接访问 Wikipedia 的转储数据；其收益主要面向小型用户或爱好者。
  - 一条评论澄清说，Wikipedia 的数据一直可以作为完整的网站下载，并假设所有主要的 LLM (Large Language Models) 都已经基于这些数据进行了训练，这表明 Kaggle 的发布并非模型训练的根本性新数据源。

- **[Qwen 3 在哪里？](https://www.reddit.com/r/LocalLLaMA/comments/1k183aa/where_is_qwen_3/)** ([Score: 172, Comments: 57](https://www.reddit.com/r/LocalLLaMA/comments/1k183aa/where_is_qwen_3/)): **该帖子质疑了备受期待的 **Qwen 3** 发布的当前状态，此前曾有明显的活动，如 GitHub pull requests 和社交媒体公告。目前尚未发布官方更新或新的基准测试，项目在最初的热度之后陷入了沉寂。** 热门评论推测 Qwen 3 仍在开发中，并参考了 Deepseek R2 等其他项目的类似时间表，同时提到用户在此期间可以使用 Gemma 3 12B/27B 等现有模型；未提供具体的底层技术批评或新信息。

  - 一位评论者指出，在 Llama 4 发布出现问题后，模型开发者在发布时可能会更加谨慎，旨在实现更顺畅的开箱即用兼容性，而不是依赖社区在发布后修复问题。这反映了开源 LLM 领域向更成熟、用户友好的部署实践的转变。
  - 有人提到 Deepseek 目前正在开发 R2，而 Qwen 正在积极开发 Version 3，突显了开源 AI 模型社区内持续的并行开发努力。此外，Gemma 3 12B 和 27B 被提及为目前可用的、性能强大但被低估的模型。




## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo


## 1. OpenAI o3 和 o4-mini 模型基准测试与用户体验

- **[震惊！！OpenAI 反击了。o3 在长上下文理解方面几乎完美。](https://i.redd.it/kw13sjo4ieve1.jpeg)** ([评分: 769, 评论: 169](https://www.reddit.com/r/singularity/comments/1k1df3c/what_openai_strikes_back_o3_is_pretty_much/)): **该帖子分享了来自 Fiction.LiveBench 的基准测试表 (https://i.redd.it/kw13sjo4ieve1.jpeg)，评估了多个 LLM (large language models) 在高达 120k tokens 输入长度下的长上下文理解能力。OpenAI 的 'o3' 模型表现出色，在短上下文 (0-4k) 中始终获得 100.0 分，在 32k 时保持高性能 (83.3)，并且独特地在 120k 时恢复到满分 (100.0)，超越了所有列出的竞争对手，如 Gemini 1.5 Pro, Llama-3-70B, Claude 3 Opus 和 Gemini 1.5 Flash。随着上下文长度的增加，其他模型表现出更多的波动且得分普遍较低，这表明 o3 具有卓越的长上下文细粒度记忆和推理能力。** 评论者指出需要超过 120k tokens 的更高上下文基准测试，并质疑为什么 'o3' 和 Gemini 2.5 在 120k 时的表现异常优于 16k，推测可能存在评估偏差或针对极端长上下文的模型特定优化。

  - 一个技术层面的担忧是，尽管据报道 o3 能很好地处理 120k tokens 的上下文，但基准测试本身上限就是 120k，限制了长上下文理解评估的深度。人们呼吁将基准测试的上下文限制提高到 120k 以上，以真实评估 o3 和 Gemini 2.5 等模型。
  - 一位用户指出了一项实际限制：尽管声称在长上下文窗口（高达 120k tokens）中表现强劲，但 OpenAI 的 Web 界面将输入限制在 64k tokens 左右，经常产生“消息过长”错误，这限制了 Pro 用户在现实世界中的可用性。
  - 有人提出了一个技术问题：为什么像 o3 和 Gemini 2.5 这样的模型有时在 120k tokens 时的表现似乎比在 16k 等较短上下文窗口时更好，这引发了人们对上下文窗口性能动态以及导致这一反直觉结果的潜在架构或训练原因的兴趣。

- **[o3 思考了 14 分钟，结果错得离谱。](https://i.redd.it/fnazpzfyaave1.jpeg)** ([评分: 1403, 评论: 402](https://www.reddit.com/r/OpenAI/comments/1k0z2qs/o3_thought_for_14_minutes_and_gets_it_painfully/)): **该图片记录了一个失败案例，其中 ChatGPT 的视觉模型（被称为 o3）被要求清点图片中的石头。尽管“思考”了近 14 分钟，它却错误地得出结论说有 30 块石头，而一位评论者指出实际有 41 块。这突显了当前 AI 视觉模型在精确物体计数方面的持续局限性，即使延长了推理时间也是如此。** 评论者指出了这一错误——其中一人给出了正确答案（41 块石头），另一人表示怀疑但认为这种错误在情理之中，强调了人们对 AI 在此类感知任务中可靠性的持续疑虑。另一位用户分享了与 Gemini 2.5 Pro 的对比，表明了对这些视觉模型进行基准测试的广泛兴趣。

  - 用户分享的图像对比（例如与 Gemini 2.5 Pro 的对比）暗示，LLM 或像 o3 和 Gemini 2.5 Pro 这样的多模态模型在处理简单的视觉计数任务（如确定图像中物体的数量，如石头）时表现出明显的困难。这表明领先 AI 实验室的当前模型在基础视觉定量推理方面仍存在持续的局限性。
  - 讨论间接提到了将模型不恰当地或不匹配地应用于其技能领域之外的任务，正如一位用户所指出的，将 AI 的失败比作使用“锤子”去干“切割”的活——这表明依靠 LLM 或多模态模型进行精确的视觉计数可能不符合它们的设计优势。这指向了对此类任务需要专门的架构或更多训练的需求，而不是期望通用模型能立即掌握所有领域。

- **[o3 在 Fiction.Livebench 长上下文基准测试中碾压所有模型（包括 Gemini 2.5），太离谱了](https://i.redd.it/ziv1rls9heve1.jpeg)** ([Score: 139, Comments: 56](https://www.reddit.com/r/OpenAI/comments/1k1dckt/o3_mogs_every_model_including_gemini_25_on/)): **图像显示了 'Fiction.LiveBench' 长上下文理解基准测试，其中 'o3' 模型在所有测试的上下文大小（从 400 到 120k tokens）中均获得了 100.0 的满分，显著优于 Gemini 2.5 等竞争对手，后者的性能在较大上下文时会有所下降。这表明 o3 在长输入序列中保持深度理解的能力方面取得了架构或训练上的进步，而目前的 SOTA 模型在这一问题上仍面临挑战——特别是在 16k tokens 以上。完整的基准测试详情可在[提供的链接](https://fiction.live/stories/Fiction-liveBench-Mar-25-2025/oQdzQvKHw8JyXbN87)中验证。** 热门评论对该基准测试的有效性提出了质疑，声称它更多是作为一种广告，与实际效果并无强相关性。技术讨论涉及 'o3' 和 '2.5 pro' 在 16k token 关口都遇到了困难，并指出 'o3' 无法像报道中的 '2.5 Pro' 那样处理 1M-token 的上下文。

  - 人们对 Fiction.Livebech 长上下文基准测试的有效性表示担忧，用户指责其更多是为托管网站做广告，并认为报告的结果可能与实际的模型使用或性能不符。
  - 讨论强调 Gemini 2.5 Pro 和 o3 模型在 16k token 上下文窗口时都表现吃力，突显了它们在处理某些长上下文场景时的局限性，尽管在其他方面有所改进；这对于强调扩展上下文理解的任务非常重要。
  - 尽管 o3 在某些方面有所改进，但用户报告称，它仍然受到比 o1 Pro 更严格的输出 token 限制，这可能会影响其在需要长篇或较少限制生成的场景中的可用性，且一些人发现它在遵循指令方面不太可靠。


## 2. Recent Video Generation Model Launches and Guides (FramePack, Wan2.1, LTXVideo)

- **[终于有能在消费级 GPU 上运行的视频扩散模型了？](https://github.com/lllyasviel/FramePack)** ([Score: 926, Comments: 332](https://www.reddit.com/r/StableDiffusion/comments/1k1668p/finally_a_video_diffusion_on_consumer_gpus/)): **lllyasviel 发布了一个新的开源视频扩散模型，据报道可以在 *消费级 GPU 上进行视频生成*（细节待定，但对可访问性和硬件要求具有重要意义）。早期用户报告确认了 Windows 手动安装成功，完整安装约占用 `40GB` 磁盘空间；目前已有第三方的[分步安装指南](https://www.reddit.com/r/StableDiffusion/comments/1k18xq9/guide_to_install_lllyasviels_new_video_generator/)。** 评论强调了 lllyasviel 在开源社区的声誉，指出这一进展在技术上令人印象深刻，且与之前高资源需求的视频扩散模型发布相比，其可访问性尤为突出。[外部链接摘要] [FramePack](https://github.com/lllyasviel/FramePack) 是论文 "Packing Input Frame Context in Next-Frame Prediction Models for Video Generation" 中提出的视频扩散下一帧预测架构的官方实现。FramePack 将输入上下文压缩到固定长度，使计算工作量不随视频长度变化，从而能够在相对较低配置的 GPU（≥6GB，例如 RTX 30XX 笔记本显卡）上对大型模型（如 13B 参数）进行高效推理和训练。该系统支持带有直接视觉反馈的分段视频生成，提供强大的内存管理和极简的独立 GUI，并兼容各种注意力机制（PyTorch, xformers, flash-attn, sage-attention）。量化方法和 "teacache" 加速可能会影响输出质量，因此建议仅在最终渲染前的实验阶段使用。

  - 一位用户详细介绍了他们在 Windows 上手动安装 lllyasviel 新视频扩散生成器的经验，强调完整安装需要约 `40 GB` 的磁盘空间。他们确认安装成功，并为他人提供了分步设置指南的链接，强调需要具备一定的命令行熟练度才能顺利完成设置过程：[安装指南](https://www.reddit.com/r/StableDiffusion/comments/1k18xq9/guide_to_install_lllyasviels_new_video_generator/)。

- **[新的 LTXVideo 0.9.6 Distilled 模型简直太疯狂了！我几秒钟就能生成不错的结果！](https://v.redd.it/6a4hja6kogve1)** ([Score: 204, Comments: 41](https://www.reddit.com/r/StableDiffusion/comments/1k1o4x8/the_new_ltxvideo_096_distilled_model_is_actually/)): **LTXVideo 0.9.6 Distilled 模型在视频生成方面提供了显著改进，仅需 `8 steps` 即可生成高质量输出，推理时间大幅缩短。技术变化包括引入了 `STGGuiderAdvanced` 节点，能够在整个扩散过程中动态调整 CFG 和 STG 参数，并且所有工作流都已更新以实现最佳参数化 ([GitHub](https://github.com/Lightricks/ComfyUI-LTXVideo), [HuggingFace 权重](https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-2b-0.9.6-distilled-04-25.safetensors))。官方 [工作流](https://civitai.com/articles/13699/ltxvideo-096-distilled-workflow-with-llm-prompt) 采用了 LLM 节点进行提示词增强，使过程既灵活又高效。** 评论强调了输出速度和可用性的剧增，以及新引导节点带来的技术飞跃，标志着视频合成正向快速迭代迈进。一种潜在的共识认为，该版本的发布降低了 ComfyUI 等高级工作流的普及门槛。[外部链接摘要] LTXVideo 0.9.6 Distilled 模型相比之前版本引入了重大进步：它能在几秒钟内生成高质量、可用的视频，蒸馏版本比全量模型推理速度快 15 倍（支持 8、4、2 甚至 1 个扩散步数采样）。关键技术改进包括用于分步配置 CFG 和 STG 的新 `STGGuiderAdvanced` 节点、更好的提示词遵循能力、改进的运动和细节，以及默认 1216×704 分辨率、30 FPS 的输出——在 H100 GPU 上可实现实时生成——且不需要 Classifier-Free 或 Spatio-Temporal 引导。利用基于 LLM 的提示词节点进行的工作流优化进一步提升了用户体验和输出控制。[完整讨论与链接](https://v.redd.it/6a4hja6kogve1)

  - LTXVideo 0.9.6 Distilled 被强调为该模型最快的迭代版本，仅需 `8 steps` 即可生成结果，使其比以前的版本更轻量，更适合快速原型设计和迭代。这种对性能的关注对于需要快速预览或实验的工作流至关重要。
  - 此次更新引入了新的 `STGGuiderAdvanced` 节点，允许在扩散过程的不同步骤应用不同的 CFG 和 STG 参数。这种动态参数化旨在提高输出质量，现有的模型工作流已重构以利用此节点实现最佳性能，详见项目的 [示例工作流](https://github.com/Lightricks/ComfyUI-LTXVideo#example-workflows)。
  - 一位用户的询问提出了一个技术问题：LTXVideo 0.9.6 Distilled 是否缩小了与 Wan 和 HV 等竞争视频生成模型的差距，这表明用户对这些领先解决方案之间的直接基准测试或定性对比分析很感兴趣。

- **[在 Windows 上安装 lllyasviel 新视频生成器 Framepack 的指南（今天就装，不用等明天的安装程序）](https://www.reddit.com/r/StableDiffusion/comments/1k18xq9/guide_to_install_lllyasviels_new_video_generator/)** ([Score: 226, Comments: 133](https://www.reddit.com/r/StableDiffusion/comments/1k18xq9/guide_to_install_lllyasviels_new_video_generator/)): **本帖提供了在官方安装程序发布前，在 Windows 上手动分步安装 lllyasviel 新的 FramePack 视频扩散生成器的指南 ([GitHub](https://github.com/lllyasviel/FramePack))。安装过程包括创建虚拟环境、安装特定版本的 Python (3.10–3.12)、特定于 CUDA 的 PyTorch wheel 文件、Sage Attention 2 ([woct0rdho/SageAttention](https://github.com/woct0rdho/SageAttention/releases)) 以及可选的 FlashAttention，并注明官方要求为 Python <=3.12 和 CUDA 12.x。用户必须手动选择与环境匹配的 Sage 和 PyTorch 兼容 wheel 文件，应用程序通过 `demo_gradio.py` 启动（已知问题是嵌入的 Gradio 视频播放器无法正常工作；输出会保存到磁盘）。视频生成是增量式的，每次增加 1 秒，导致磁盘占用量巨大（据报告超过 45GB）。** 评论中没有重大的技术争论——大多数用户都在等待官方安装程序。报告的一个小问题是 Gradio 视频播放器无法渲染视频，尽管输出可以正常保存。

- 一位用户询问在 NVIDIA 4090 上生成 5 秒视频所需的时间，这表明用户对 Framepack 在高端 GPU 上的具体性能基准和吞吐率感兴趣。
- 另一位用户询问了 Framepack 在 3060 12GB GPU 上运行的实际性能反馈，寻求该工具在中端消费级硬件上表现的信息。这些问题突显了社区对这一新视频生成工具的实测速度和硬件要求的关注。

- **[官方 Wan2.1 首帧末帧模型发布](https://v.redd.it/lvk0hp7uqeve1)** ([评分: 779, 评论: 102](https://www.reddit.com/r/StableDiffusion/comments/1k1enhx/official_wan21_first_frame_last_frame_model/)): **Wan2.1 首末帧转视频模型 (FLF2V) v14B 现已完全开源，并提供了 [权重和代码](https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P) 以及 [GitHub 仓库](https://github.com/Wan-Video/Wan2.1)。此次发布仅限于单个 14B 参数的大模型，且仅支持 `720P` 分辨率——480P 和其他变体目前尚不可用。该模型主要在中文文本-视频对上进行训练，使用中文 Prompt 可以获得最佳效果。此外还提供了一个 [ComfyUI 工作流示例](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/main/example_workflows/wanvideo_FLF2V_720P_example_01.json) 以供集成。** 评论者注意到缺乏较小或低分辨率的模型，并强调了对 480p 和其他变体的需求。训练数据集对中文 Prompt 的侧重被认为是获得最佳模型输出的关键。[外部链接摘要] Wan2.1 首帧末帧 (FLF2V) 模型现已在 HuggingFace 和 GitHub 上完全开源，支持根据用户提供的首末帧生成 720P 视频，无论是否带有 Prompt 扩展（目前不支持 480P）。该模型主要在中文文本-视频对上进行训练，使用中文 Prompt 的效果显著更好，并且提供了 ComfyUI 工作流封装器和 fp8 量化权重以供集成。技术细节及模型/代码获取：[HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P) | [GitHub](https://github.com/Wan-Video/Wan2.1)。

- 该模型主要在中文文本-视频对上进行训练，因此使用中文编写的 Prompt 会产生更好的结果。这突显了由于训练数据集导致的语言偏差，在使用非中文 Prompt 时可能会影响输出质量。
- 目前仅提供 14B 参数、720p 的模型。用户对其他模型（如 480p 或不同参数规模）感兴趣，但这些尚未得到支持或发布。
- GitHub 上提供了一个将 Wan2.1 首帧末帧模型与 ComfyUI 集成的工作流（参见 [此工作流 JSON](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/main/example_workflows/wanvideo_FLF2V_720P_example_01.json)）。此外，HuggingFace 上发布了 fp8 量化模型变体，从而实现了更高效的部署方案。

## 3. 创新与专业图像/角色生成模型发布

- **[InstantCharacter 模型发布：个性化任何角色](https://i.redd.it/28svqmqs3fve1.png)** ([Score: 126, Comments: 22](https://www.reddit.com/r/StableDiffusion/comments/1k1ge3y/instantcharacter_model_release_personalize_any/)): **该图展示了腾讯新发布的 InstantCharacter 模型，这是一种无需微调（tuning-free）的开源解决方案，可从单张图像生成保持角色特征的内容。它直观地演示了工作流程：通过文本和图像调节，将参考图像转换为在各种复杂背景（如地铁、街道）下高度个性化的动漫风格表现。该模型利用结合了 Style LoRA 的 IP-Adapter 算法，运行在 Flux 上，旨在在灵活性和保真度方面超越早期的 InstantID 等解决方案。** 技术导向的评论者赞扬了生成结果，并对集成（例如 'ComfyUI 节点'）表示出浓厚兴趣，增强了该工作流在下游生成任务中的感知质量和可用性。

  - 一位用户提到，现有的解决方案（如 UNO）在个性化角色生成方面表现不尽如人意，暗示先前的模型在集成或输出质量方面存在困难。这突显了当前工具在可靠的角色个性化方面面临的挑战，并为评估 InstantCharacter 的方法和承诺的能力设定了技术基准。

- **[Flux.Dev 对比 HiDream Full](https://www.reddit.com/gallery/1k1258e)** ([Score: 105, Comments: 37](https://www.reddit.com/r/StableDiffusion/comments/1k1258e/fluxdev_vs_hidream_full/)): **本帖提供了 [Flux.Dev](http://Flux.Dev) 与 HiDream Full 的并排对比，使用了 HiDream ComfyUI 工作流（[参考](https://comfyanonymous.github.io/ComfyUI_examples/hidream/)）和 `hidream_i1_full_fp16.safetensors` 模型（[模型链接](https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/blob/main/split_files/diffusion_models/hidream_i1_full_fp16.safetensors)）。生成参数为 `50 steps`、`uni_pc` 采样器、`simple` 调度器、`cfg=5.0` 以及 `shift=3.0`，涵盖了七个详细的提示词。对比在遵循度和风格上对结果进行了视觉排名，显示 Flux.Dev 在提示词忠实度方面通常表现出色，甚至在风格上能与 HiDream Full 竞争，尽管 HiDream 的资源需求更高。** 讨论强调了 LLM 风格的“华丽辞藻”（purple prose）提示词对评估原始提示词遵循度的影响，并指出尽管 HiDream 在某些单项上获胜，但 Flux.Dev 被认为具有更好的整体提示词遵循度和资源效率。一些人形容 HiDream 在这组测试中的表现“令人失望”，尽管替代方案仍受到欢迎。

- 几位用户强调了提示词设计在基准测试中的重要性，强调包含复杂或主观语言（如“情绪”描述或过多的散文体）的 LLM 生成提示词会引入变量，使得准确评估模型的提示词遵循度变得更加困难。建议使用更精确或客观的提示词将产生更清晰的性能对比。
  - 大家的共识是 **Flux.Dev** 在这一轮中胜过 **HiDream Full**，特别是在提示词遵循和风格灵活性方面，尽管 **HiDream** 更加耗费资源。Flux 被视为微弱的赢家，但据报道，这两个模型在原始性能上都显著超越了前几代模型。
  - 针对第一个对比提示词提出了批评，指出其中存在语法错误和如 'hypo realistic' 等非标准术语。语言上的不规范被认为是导致两个模型产生困惑的可能原因，潜在地影响了并排评估的可靠性。

---

# AI Discord 摘要
> 由 Gemini 2.5 Flash Preview 生成的摘要之摘要的摘要

**主题 1. 最新 LLM 模型：成功、失败与幻觉**

*   **新 Gemini 2.5 Flash 登陆 Vertex AI**：Google 的 [Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models) 已在 Vertex AI 中上线，因其先进的推理和编码能力而备受推崇，引发了与 **Gemini 2.5 Pro** 在效率和工具调用（tool-calling）方面的辩论，但也有关于*思维循环（thinking loops）*的报告。用户还在权衡 **O3** 和 **O4 Mini**，发现 **O3** 更具优势，因为 **O4 Mini** 的高昂输出成本使其*几乎无法使用*。
*   **O4 模型幻觉增多，用户抱怨**：用户报告 **o4-mini** 和 **o3** 模型*更频繁地编造信息*，甚至提供看似可信但完全错误的答案，如虚假的商业地址。虽然建议模型*通过搜索验证来源*可能有所帮助，但用户发现 **GPT-4.1 Nano** 在事实性任务中处理*非虚假信息表现更好*。
*   **微软发布 1-Bit BitNet，IBM 发布 Granite 3**：**Microsoft Research** 发布了 **BitNet b1.58 2B 4T**，这是一个在 **4 万亿 token** 上训练的 **20 亿参数** 原生 1-bit LLM，可在 [Microsoft 的 GitHub](https://github.com/microsoft/BitNet) 上获取推理实现。IBM 宣布了 **Granite 3** 和经过优化的推理 **RAG Lora** 模型，详见 [此 IBM 公告](https://www.ibm.com/new/announcements/ibm-granite-3-3-speech-recognition-refined-reasoning-rag-loras__._astro_.__)。

**Theme 2. AI 开发工具与框架**

*   **Aider 获得新 Probe 工具，架构师模式吞掉文件**：Aider 引入了用于语义代码搜索的 `probe` 工具，因其能提取错误代码块并与测试输出集成而受到称赞，同时还分享了 [claude-task-tool](https://github.com/paul-gauthier/claude-task-tool) 等替代方案。用户在 **Aider 的架构师模式（architect mode）**中遇到了一个 Bug：在创建 **15 个新文件**后添加一个到聊天中会导致所有更改被丢弃，虽然这是预期行为，但*在丢弃编辑时没有给出任何警告*。
*   **Cursor 的编码助手引起热议、崩溃与困惑**：用户在讨论 Cursor 中 **o3/o4-mini** 是否优于 **2.5 Pro** 和 **3.7 thinking**，有人报告 **o4-mini-high** 在代码库分析和逻辑方面优于 **4o** 和 **4.1**，即使是大型项目也是如此。其他人则抱怨 **Cursor agent** 不退出终端、频繁进行无代码输出的工具调用以及连接/编辑问题。
*   **MCP, LlamaIndex, NotebookLM 提升集成与 RAG**：一名成员正在为 **Obsidain 构建 MCP server** 以简化集成，并寻求关于通过 HTTPS **header** 安全传递 API key 的建议。LlamaIndex 现在支持根据开放协议构建兼容 **A2A (Agent2Agent)** 的 Agent，从而实现安全的信息交换，无论[底层基础设施](https://twitter.com/llama_index/status/1912949446322852185)如何。NotebookLM 用户集成了 **Google Maps** 并利用 RAG，分享了诸如[此 Vertex RAG 图表](https://cdn.discordapp.com/attachments/1124402182909857966/1362567715360866365/Vertex_RAG_diagram_b4Csnl2.original.png?ex=6802dd92&is=68018c12&hm=2049e59b022a0ef55db1299c859c7c7cc0b89d1e38c0d8c74dcd753279ee08aa&)之类的架构图。

**Theme 3. 优化 AI 硬件性能**

*   **Triton, CUDA, Cutlass：底层性能挑战**：在 GPU MODE 的 `#triton` 频道中，一名成员报告慢速 **fp16 矩阵乘法**（**2048x2048**）落后于 cuBLAS，被建议使用更大的矩阵或测量端到端模型处理。用户在 `#cutlass` 频道中尝试使用 Cutlass 的 `cuda::pipeline` 和 **TMA/CuTensorMap API**，发现基准测试的 mx cast kernel 仅达到 **3.2 TB/s**，并寻求关于 **Cutlass** 瓶颈的建议。
*   **AMD MI300 排行榜竞争激烈，NVIDIA 硬件讨论**：根据 **GPU MODE** 的 `#submissions` 频道，**MI300** 上 `amd-fp8-mm` 排行榜的提交结果差异很大，其中一项提交达到了 **255 µs**。`#cuda` 频道的讨论确认 **H200** 不支持 **FP4** 精度，这可能是 **B200** 的笔误，LM Studio 的成员正在针对 **0.3.15 beta** 版本优化新的 **RTX 5090**。
*   **量化定性地改变了 LLM，AVX 需求依然存在**：Eleuther 的成员探讨了分析**量化**对 **LLM** 影响的研究，认为在低比特位时会发生定性变化，尤其是使用基于训练的策略时，这得到了 [composable interventions 论文](https://arxiv.org/abs/2407.06483)的支持。运行不带 **AVX2** 的 **E5-V2** 的旧服务器只能使用非常旧版本的 LM Studio 或 [llama-server-vulkan](https://github.com/kth8/llama-server-vulkan) 等替代项目，因为现代 LLM 需要 **AVX**。

**Theme 4. AI 模型安全、数据与社会影响**

*   **AI 幻觉依然存在，伪对齐误导用户**：人们对“伪对齐（pseudo-alignment）”表示担忧，即 LLM 通过谄媚行为来欺骗用户，依赖于似是而非的想法拼凑而非真正的理解。Eleuther 的 `#general` 频道成员认为，*开放网络目前正因 AI 的存在而受到实质性的破坏*。一位成员花费 **7 个月**时间构建了 **PolyThink**，这是一个基于 Agent 的多模型 AI 系统，旨在通过模型间的相互纠错来消除 AI 幻觉，并邀请用户加入[等候名单](https://www.polyth.ink/)。
*   **数据隐私与验证引发关注**：OpenRouter 更新了其[服务条款和隐私政策](https://openrouter.ai/privacy)，明确表示**未经同意不会存储 LLM 输入**，提示词分类仅用于排名和分析。Discord 新的年龄验证功能要求通过 [withpersona.com](https://withpersona.com/) 进行身份验证，这在 Nous Research AI 社区引发了关于隐私妥协及潜在平台级变化的担忧。
*   **欧洲培育区域语言模型**：成员们讨论了除 Mistral 之外，欧洲各地*区域定制语言模型*的可用性，包括荷兰的 [GPT-NL 生态系统](https://www.computerweekly.com/news/366558412/Netherlands-starts-building-its-own-AI-language-model)、意大利的 [Sapienza NLP](https://www.uniroma1.it/en/notizia/ai-made-italy-here-minerva-first-family-large-language-models-trained-scratch-italian)、西班牙的[巴塞罗那超算中心](https://sifted.eu/articles/spain-large-language-model-generative-ai)、法国的 [OpenLLM-France](https://huggingface.co/blog/manu/croissant-llm-blog)/[CroissantLLM](https://github.com/OpenLLM-France)、德国的 [AIDev](https://aivillage.de/events/ai-dev-3/)、俄罗斯的 [Vikhr](https://arxiv.org/abs/2405.13929)、希伯来语的 [Ivrit.AI](https://www.ivrit.ai/he/%d7%a2%d7%91%d7%a8%d7%99%d7%9d-%d7%93%d7%91%d7%a8%d7%95-%d7%a2%d7%96%d7%91%d7%a8%d7%99%d7%9dit/)/[DictaLM](https://arxiv.org/html/2407.07080v1)、波斯语的 [Persian AI Community](https://huggingface.co/PersianAICommunity) 以及日本的 [rinna](https://www.alibabacloud.com/blog/rinna-launched-ai-models-trained-in-the-japanese-lang)。

**主题 5. 行业观察：禁令、收购与业务转型**

*   **特朗普关税针对欧盟、中国及 Deepseek？**：根据 [Perplexity 的这份报告](https://www.perplexity.ai/page/trump-imposes-245-tariffs-on-c-LbKOTTe8TyWXY_ov9vUb_A)，特朗普政府对欧盟产品征收了 **245% 的关税**，以报复空客补贴，并因知识产权盗窃对中国商品征收关税。据 [TechCrunch 的这篇文章](https://techcrunch.com/2025/04/16/trump-administration-reportedly-considers-a-us-deepseek-ban/)报道，**特朗普政府**据传还在考虑在全美范围内**禁用 Deepseek**。
*   **OpenAI 收购 Windsurf 的传闻甚嚣尘上**：有推测称 **OpenAI** 可能会以 **30 亿美元**的价格收购 **Windsurf**，一些人认为这是该公司变得越来越像 Microsoft 的迹象。随后引发了关于 **Cursor** 和 **Windsurf** 是真正的 IDE，还是仅仅是为“氛围程序员（vibe coders）”量身定制的美化版 API 封装工具的争论。
*   **LMArena 走向公司化，HeyGen API 发布**：源自加州大学伯克利分校项目的 LMArena [正在组建公司](https://blog.lmarena.ai/blog/2025/new-beta/)，以支持其平台运行，同时确保其保持中立和可访问性。**HeyGen API** 的产品负责人介绍了他们的平台，强调了其无需摄像头即可制作极具吸引力的视频的能力。

---

# 第一部分：Discord 高层级摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 发布 Telegram 机器人**：**Perplexity AI** 现在可以通过 [askplexbot](https://t.me/askplexbot) 在 **Telegram** 上使用，并计划集成 **WhatsApp**。
  
  - 一段预告视频 ([Telegram_Bot_Launch_1.mp4](https://cdn.discordapp.com/attachments/1047204950763122820/1362539434565697779/Telegram_Bot_Launch_1.mp4?ex=6802c33b&is=680171bb&hm=bf0a3730711b37e0eadba26f66492cb2608b1a3f224b590533865dce7f713f37&)) 展示了其无缝集成和实时响应能力。
- **Perplexity 讨论 Discord 支持方案**：成员们建议在 Discord 中加入 **ticketing bot**（工单机器人），但更倾向于使用 [help center](https://link.to/help-center)（帮助中心）的方式。
  
  - 讨论中提到了 **helper role**（助手角色）等替代方案，并指出 Discord 并非理想的支持平台，连接到 Zendesk 的 *Modmail bot* 可能会更有用。
- **Neovim 配置展示**：一位成员在学习 IT 三天后分享了他们的 **Neovim configuration** [图片](https://cdn.discordapp.com/attachments/1047649527299055688/1362486729340092496/image.png?ex=68029226&is=680140a6&hm=7a9871a1c9146850518afcdf9f96e8c64589fab3f554c6932fd6e8e958f7c59c&)。
  
  - 他们利用 AI 模型让学习过程变得*有趣*，且效果良好。
- **特朗普关税影响显现**：根据[这份报告](https://www.perplexity.ai/page/trump-imposes-245-tariffs-on-c-LbKOTTe8TyWXY_ov9vUb_A)，特朗普政府为了报复对 Airbus 的补贴，对欧盟产品征收了 **245% 的关税**，并因知识产权窃取问题对中国商品征收关税。
  
  - [这次 Perplexity 搜索](https://www.perplexity.ai/search/e2a38d65-9a5b-4e02-83a7-42433999f5cd)解释称，这些措施旨在保护美国工业并解决贸易失衡问题。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 2.5 Flash 席卷 Vertex AI**：[Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models) 出现在 Vertex AI 中，引发了关于其与 **Gemini 2.5 Pro** 相比在代码效率和 tool-calling 能力方面的讨论。
  
  - 一些用户称赞其速度，而另一些用户则报告它会陷入类似于 **2.5 Pro** 之前出现的 *thinking loops*（思考循环）。
- **O3 与 O4 Mini 对决**：成员们正在积极测试和比较 **O3** 与 **O4 Mini**，并分享了像 [这个](https://liveweave.com/A9OGzH#) 这样的实时测试来展示它们的潜力。
  
  - 尽管 **O4 Mini** 的初始成本较低，但一些用户发现其高使用量和输出成本令人望而却步，导致许多人重新使用 **O3**。
- **Thinking Budget 功能引发争议**：Vertex AI 的新功能 *Thinking Budget*（允许控制 thinking tokens）正受到关注。
  
  - 虽然有些人觉得它很有用，但也有人报告了 Bug，一位用户指出 *2.5 pro 在 0.65 temp 下表现更好*。
- **LLM：教育的救星还是破坏者？**：关于 LLM 辅助发展中国家教育的潜力正在讨论中，重点在于可访问性与可靠性之间的平衡。
  
  - 讨论中提到了对 LLM 产生 *hallucinate*（幻觉）倾向的担忧，并将其与*由专业人士编写的书籍*的可靠性进行了对比。
- **LMArena 走向公司化，保持开放**：源自加州大学伯克利分校项目的 LMArena [正在组建公司](https://blog.lmarena.ai/blog/2025/new-beta/)以支持其平台，同时确保其保持中立和开放。
  
  - 社区还报告称 [Beta 版本](https://beta.lmarena.ai) 结合了用户反馈，包括深色/浅色模式切换和直接复制/粘贴图片功能。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 的 code2prompt 评价褒贬不一**：成员们讨论了 **Aider** 中 `code2prompt` 的实用性，质疑其相对于 `/add` 命令在包含必要文件方面的优势，因为 `code2prompt` 会快速解析所有匹配的文件。
  
  - `code2prompt` 的效用取决于特定的使用场景和模型能力，主要是其解析速度。
- **Aider 的 Architect Mode 吞掉了新文件**：一位成员在 **Aider 的 architect mode** 中遇到了一个 Bug，在创建了 **15 个新文件**并将其中的一个文件添加到对话后，更改被丢弃了。
  
  - 这种行为是预料之中的，但在丢弃编辑内容时没有给出警告，导致用户在重构的代码丢失时感到困惑。
- **Aider 的 Probe 工具亮相**：成员们讨论了 Aider 新的 `probe` 工具，强调了其语义代码搜索能力，用于提取带有错误的代码块并与测试输出集成。
  
  - 爱好者们分享了用于语义代码搜索的替代方案，例如 [claude-task-tool](https://github.com/paul-gauthier/claude-task-tool) 和 [potpie-ai/potpie](https://github.com/potpie-ai/potpie)。
- **DeepSeek R2 热度高涨**：成员们对即将发布的 **DeepSeek R2** 充满热情，希望它的性能能超越 **O3-high**，同时提供更好的价格点，并暗示[这只是时间问题](https://discord.com/channels/1131200896827654144/1131200896827654149/1362174807180967946)。
  
  - 一些人推测，由于其潜在的卓越性价比，**DeepSeek R2** 可能会挑战 **OpenAI** 的主导地位。
- **YouTube 提供了对新模型的冷静分析**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=3aRRYQEb99s)，对新模型提供了更理性的看法。
  
  - 该视频对近期发布的模型进行了**分析**，重点关注技术价值。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 更新条款**：OpenRouter 更新了其 [服务条款和隐私政策](https://openrouter.ai/privacy)，明确表示 **未经同意不会存储 LLM 输入**，并详细说明了他们如何对 Prompt 进行分类以进行排名和分析。
  
  - Prompt 分类用于确定请求的*类型*（编程、角色扮演等），对于未选择开启日志记录的用户，这些信息将是匿名的。
- **购买额度以使用**：OpenRouter 更新了免费模型的限制，现在 **要求终身累计购买至少 10 个额度** 才能享受更高的 **1000 次请求/天 (RPD)**，无论当前的额度余额是多少。
  
  - 由于需求极高，实验性 `google/gemini-2.5-pro-exp-03-25` 免费模型的访问权限仅限于购买了至少 10 个额度的用户；[付费版本](https://openrouter.ai/google/gemini-2.5-pro-preview-03-25)提供不间断访问。
- **Gemini 2.5 表现亮眼**：OpenRouter 推出了 **Gemini 2.5 Flash**，这是一款用于高级推理、编程、数学和科学的模型，提供 [标准版](https://openrouter.ai/google/gemini-2.5-flash-preview) 和带有内置推理 Token 的 [:thinking](https://openrouter.ai/google/gemini-2.5-flash-preview:thinking) 变体。
  
  - 用户可以使用 `max tokens for reasoning` 参数自定义 **:thinking** 变体，详见 [文档](https://openrouter.ai/docs/use-cases/reasoning-tokens#max-tokens-for-reasoning)。
- **成本模拟器和聊天应用上线**：一位成员创建了一个模拟 LLM 对话成本的工具，支持 [OpenRouter](https://llm-cost-simulator.vercel.app) 上的 **350 多种模型**；另一位成员开发了一个连接 [OpenRouter](https://chat.nanthai.tech/chat) 的 LLM 聊天应用，提供精选 LLM 列表以及网页搜索和 RAG 检索等功能。
  
  - 该聊天应用有基础免费版，扩展搜索和 RAG 功能需按月付费，或选择无限使用。
- **Codex 报错，DeepSeek 延迟**：**OpenAI 的 Codex** 使用了新的 API 端点，因此目前无法与 **OpenRouter** 配合使用；由于 **OpenAI** 要求身份验证的限制，OpenRouter 的 **o-series 推理摘要** 可能会延迟。
  
  - 一位用户指出，新的 **DeepSeek** 与 Google 的 [Firebase studio](https://firebase.google.com/) 类似。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 2.5 仍比新模型更受青睐**：尽管有了 **o3** 和 **o4** 等新模型，一些人仍然更喜欢 [Gemini 2.5 Pro](https://ai.google.dev/)，因为它的*速度、准确性和成本*，尽管该模型在处理复杂任务时会*疯狂产生幻觉*。
  
  - 基准测试显示 **o3** 在编程方面表现更好，而 **Gemini 2.5 Pro** 在推理方面表现出色；新的 **2.5 Flash** 版本则强调更快的响应速度。
- **o4 模型在事实准确性方面表现不佳**：用户报告称 **o4-mini** 和 **o3** 模型*更频繁地编造信息*，甚至提供看似可信但完全错误的答案，例如虚假的公司地址。
  
  - 指示模型*通过搜索验证来源*可能有助于减少幻觉，但有人指出 **GPT-4.1 Nano** 在处理非虚假信息方面表现*更好*。
- **GPT-4.5 用户抱怨模型太慢**：多位用户抱怨 **GPT 4.5** 非常慢且昂贵，推测这“可能是因为它是一个稠密模型，而不是 Mixture of Experts（混合专家）”模型。
  
  - **o4-mini** 的使用限制为每天 **150 次**，**o4-mini high** 为每天 **50 次**，**o3** 为每周 **50 次**。
- **Custom GPTs 失控**：一位用户报告他们的 **Custom GPT** 不听从指令，“目前它就在那里各行其是”。
  
  - 还有人询问在 Chat **GPT** 中，哪种语言模型最适合上传 PDF 进行学习、提问并准备现成的考试题目。
- **在 GPTPlus 上模拟上下文记忆**：一位用户报告通过叙事连贯性和文本提示在 **GPTPlus** 账户上模拟了**上下文记忆**，构建了一个拥有超过 **30 个讨论**的**多模块系统**。
  
  - 另一位用户确认了类似的结果，通过使用正确的关键词连接新的讨论。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 订阅退款延迟**：取消 **Cursor 订阅**的用户正在等待退款，虽然收到了确认邮件但还没收到实际款项，不过有一位用户声称他们*已经拿到了*。
  
  - 未提供关于延迟原因或具体涉及金额的进一步细节。
- **FIOS 修复微调**：一位用户发现，物理调整其 **Verizon FIOS** 设置上的 **LAN 有线连接**可以将下载速度从 **450Mbps 提升到 900Mbps+**。
  
  - 他们建议使用类似于 **PCIE 风格**的更稳固的连接器，并发布了[一张他们的设置图片](https://cdn.discordapp.com/attachments/1074847527708393565/1362144114031988766/image.png?ex=6802a490&is=68015310&hm=08c3d65050b698b43658ba08dabb49b96bd56ace10bc982e857fbbedfcd9e502)。
- **MacBook 版 Cursor 模型上线**：用户讨论了在 **MacBook** 版本的 Cursor 上添加新模型，有些人需要重启或重新安装才能看到它们。
  
  - 建议通过输入 **o4-mini** 并点击 Add Model 来手动添加 **o4-mini** 模型。
- **Cursor 编程助手引发讨论**：用户争论 **o3/o4-mini** 是否优于 **2.5 Pro** 和 **3.7 thinking**，其中一人报告说 **o4-mini-high** 在分析代码库和解决逻辑方面优于 **4o** 和 **4.1**，即使是大型项目也是如此。
  
  - 其他人抱怨 **Cursor Agent** 在运行命令后不退出终端（导致其无限期挂起）、频繁的工具调用但没有代码输出、消息过长问题、连接状态指示器损坏以及无法编辑文件。
- **Windsurf 收购传闻风起云涌**：有推测称 **OpenAI** 可能会以 **30 亿美元**收购 **Windsurf**，一些人认为这是公司变得更像 Microsoft 的迹象，而另一些人（如[这条推文](https://x.com/chuhang1122/status/1912786904812294312)）则关注 **GPT4o mini**。
  
  - 参与者辩论 **Cursor** 和 **Windsurf** 是真正的 IDE，还是仅仅是为 *vibe coders*、扩展程序或纯文本编辑器提供服务的、带有 UX 产品的 API Wrapper（或 Fork）。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **脑部解构产生流形 (Manifolds)**：一篇论文建议大脑连接可以分解为简单的流形，并辅以额外的长程连接 ([PhysRevE.111.014410](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.111.014410))，尽管有些人对于任何使用 **Schrödinger's equation** 的东西都被称为“量子”感到厌烦。
  
  - 另一位成员澄清说，*任何平面波线性方程都可以通过使用* ***Fourier transform*** *转换为 Schrödinger's equation 的形式。*
- **Responses API 亮相，Assistant API 逐渐退出**：成员们澄清说，虽然 **Responses API** 是全新的，但 **Assistant API** 将于明年停止服务。
  
  - 有人强调，*如果你想要助手，就选择 Assistant API；如果你想要常规基础功能，就使用 Responses API*。
- **储备池计算 (Reservoir Computing) 解构**：成员们讨论了 **Reservoir Computing** 作为一个**固定、高维动力系统**，并澄清 *储备池不一定是软件 RNN，它可以是任何具有时间动态的东西*，并从该动力系统中学习一个*简单的读出 (readout)*。
  
  - 一位成员分享道，*大多数“储备池计算”的炒作通常是* ***用复杂的术语或奇特的设置包装一个非常简单的想法***：*拥有一个动力系统。不要训练它。只训练一个简单的读出。*
- **特朗普威胁禁用 Deepseek**：据 [TechCrunch 文章](https://techcrunch.com/2025/04/16/trump-administration-reportedly-considers-a-us-deepseek-ban/) 报道，**特朗普政府**据传正在考虑在**美国禁用** **Deepseek**。
  
  - 未提供更多细节。
- **Meta 推出 Fair 更新，IBM 发布 Granite**：Meta 为**感知、定位和推理**推出了 Fair 更新，IBM 宣布了 **Granite 3** 和优化的推理 **RAG Lora** 模型，以及一个新的语音识别系统，详见 [IBM 公告](https://www.ibm.com/new/announcements/ibm-granite-3-3-speech-recognition-refined-reasoning-rag-loras__._astro_.__)。
  
  - Meta 的图片展示了一些作为 Meta **fair updates** 一部分的更新。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Discord 成员被踢出**：一名成员因涉嫌*烦扰所有人*而被 Discord 服务器封禁，引发了关于审核透明度的辩论。
  
  - 虽然有些人质疑证据，但其他人辩护称这一决定对于维持*和平社区*是必要的。
- **Claude 借鉴 Manus**：**Claude** 推出了 UI 更新，支持原生研究以及与 **Google Drive** 和 **Calendar** 等服务的应用连接。
  
  - 这一功能镜像了 **Manus** 中的现有功能，促使一位成员调侃这*基本上是查尔斯三世 (Charles III) 更新*。
- **GPT 也添加了 MCPS**：成员们观察到 **GPT** 现在也具备了类似的集成功能，允许用户搜索 **Google Calendar**、**Google Drive** 和连接的 **Gmail** 账户。
  
  - 这一更新将 **GPT** 定位为生产力和研究领域的竞争者，与 **Claude** 最近的增强功能并驾齐驱。
- **AI 游戏开发梦想走向开源**：围绕着与 **AI** 和伦理 **NFT** 实施交织在一起的开源游戏开发潜力，讨论热情高涨。
  
  - 讨论围绕着什么让**抽卡游戏 (gacha games)** 具有吸引力，以及如何弥合游戏玩家与 **crypto/NFT** 世界之间的鸿沟。
- **体力系统 (Stamina System) 创新引发讨论**：提出了一种新型体力系统，在不同的体力水平提供奖励，以迎合不同的玩家风格并惠及开发者。
  
  - 探索了替代机制，例如抵押物品换取体力或整合损失元素以提高参与度，并与 **MapleStory** 和 **Rust** 等游戏进行了类比。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LLMs 产生伪对齐幻觉**：成员们对**伪对齐 (pseudo-alignment)** 表示担忧，即 LLM 通过谄媚技巧误导人们认为它们已经掌握了知识，而实际上它们只是依赖 AI 生成听起来合理的想法拼凑，并分享了一篇关于*权限随时间变化*的[论文](https://arxiv.org/abs/2407.14933)。
  
  - 成员们普遍警告称，*开放网络目前正因 AI 的存在而受到实质性的破坏*。
- **欧洲推出区域定制化语言模型**：成员们讨论了欧洲境内*区域定制化语言模型*的可用性，除了像 Mistral 这样知名的实体外，还包括荷兰的 [GPT-NL 生态系统](https://www.computerweekly.com/news/366558412/Netherlands-starts-building-its-own-AI-language-model)、意大利的 [Sapienza NLP](https://www.uniroma1.it/en/notizia/ai-made-italy-here-minerva-first-family-large-language-models-trained-scratch-italian)、西班牙的 [Barcelona Supercomputing Center](https://sifted.eu/articles/spain-large-language-model-generative-ai)、法国的 [OpenLLM-France](https://huggingface.co/blog/manu/croissant-llm-blog) 和 [CroissantLLM](https://github.com/OpenLLM-France)、德国的 [AIDev](https://aivillage.de/events/ai-dev-3/)、俄罗斯的 [Vikhr](https://arxiv.org/abs/2405.13929)、希伯来语的 [Ivrit.AI](https://www.ivrit.ai/he/%d7%a2%d7%91%d7%a8%d7%99%d7%9d-%d7%93%d7%91%d7%a8%d7%95-%d7%a2%d7%91%d7%a8%d7%99%d7%aa/) 和 [DictaLM](https://arxiv.org/html/2407.07080v1)、波斯语的 [Persian AI Community](https://huggingface.co/PersianAICommunity) 以及日本的 [rinna](https://www.alibabacloud.com/blog/rinna-launched-ai-models-trained-in-the-japanese-lang)。
  
  - 几位成员指出，**区域定制化语言模型**在特定用例中可能会非常有帮助。
- **社区辩论人类验证策略**：成员们讨论了在服务器上引入*人类身份验证*以对抗 AI 机器人的潜在需求。有人认为目前较低的影响可能不会持续，但也在考虑严格验证之外的替代方案，包括社区治理和关注活跃贡献者。
  
  - 社区的整体情绪是审慎乐观的，同时也对 AI 生成内容日益盛行表示担忧。
- **量化效应的质量检查**：成员们探讨了分析**量化 (quantization)** 对 **LLM** 影响的研究，指出在低比特 (low bits) 下正在发生质变，特别是对于基于训练的量化策略，并分享了一张带有相关支持数据的[截图](https://cdn.discordapp.com/attachments/747850033994662000/1362154554380255362/Screenshot_2025-04-16-16-55-29-291_com.xodo.pdf.reader-edit.jpg?ex=6802ae49&is=68015cc9&hm=86b6fec4bfe372bb62c27aeeea0b4bc65cf8099447d177815ff6cfb25599d63b&)。
  
  - 一位成员还推荐了[可组合干预论文 (composable interventions paper)](https://arxiv.org/abs/2407.06483)，作为低比特下发生质变的证据支持。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **带有 Illustrious Shine 的动漫模型**：成员们推荐使用 **Illustrious**、**NoobAI XL**、**RouWei** 和 **Animagine 4.0** 进行动漫生成，并指向了如 [Raehoshi-illust-XL-4](https://huggingface.co/Raelina/Raehoshi-illust-XL-4) 和 [RouWei-0.7](https://huggingface.co/Minthy/RouWei-0.7) 等模型。
  
  - 增加 **LoRA 资源**可以提高输出质量。
- **nVidia 简化了大模型的 GPU 使用**：一位成员指出，使用 `device_map='auto'` 在多个 **nVidia GPU** 上运行一个大模型更加容易，而 **AMD** 则需要更多的即兴调整，并链接到了 [Accelerate 文档](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)。
  
  - 使用 **device_map='auto'** 让框架自动管理模型在可用 GPU 上的分布。
- **PolyThink 旨在消除 AI 幻觉**：一位成员花费 **7 个月**时间构建了 **PolyThink**，这是一个 Agentic 多模型 AI 系统，旨在通过让多个 AI 模型相互纠错与协作来消除 AI 幻觉，并邀请社区[注册候补名单](https://www.polyth.ink/)。
  
  - **PolyThink** 的承诺是通过协作验证来提高 AI 生成内容的可靠性和准确性。
- **Agents 课程仍受 503 错误困扰**：多位用户报告在开始 **Agents 课程**时遇到 **503 错误**，特别是在使用 dummy agent 时，这可能由于达到了 **API key 使用限制**。
  
  - 尽管用户报告了错误，但一些人指出他们仍有可用额度，这可能表明是流量问题而非 API 限制问题。
- **TRNG 声称其高熵特性可用于 AI 训练**：一位成员构建了一个具有极高熵位评分的研究级 **True Random Number Generator (TRNG)**，并希望测试其对 AI 训练的影响，评估信息可在 [GitHub](https://github.com/thyarcanist/EntropyLattice-Public/tree/main/Evals) 上获取。
  
  - 希望在 AI 训练期间使用具有更高熵的 **TRNG** 能提高模型性能和随机性。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **SYCL 在计算平台中取代了 OpenCL？**：在 **GPU MODE** 的 `#general` 频道中，成员们讨论了 **SYCL** 是否正在取代 **OpenCL**，并探讨了这两种技术的相对优劣和未来。
  
  - 一位管理员回应称，从历史上看，目前还没有足够的需求来开设专门的 **OpenCL 频道**，尽管他们承认 OpenCL 仍然是主流，并且在 **Intel、AMD 和 NVIDIA 的 CPU/GPU** 上提供了广泛的兼容性。
- **矩阵乘法速度未达预期**：在 **GPU MODE** 的 `#triton` 频道中，一名成员报告称，大小为 **2048x2048** 的 **fp16 矩阵乘法** 性能不如预期，甚至落后于 cuBLAS，尽管他参考了 [官方教程代码](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py)。
  
  - 建议指出，更现实的基准测试方法是在 `torch.nn.Sequential` 中堆叠约 **8 个线性层**，使用 `torch.compile` 或 **cuda-graphs**，并测量模型的端到端处理时间，而不仅仅是单个 **matmul**。
- **Popcorn CLI 问题频发**：在 **GPU MODE** 的 `#general` 频道中，用户在使用该 CLI 工具时遇到了错误；一名用户在运行 **popcorn** 时被提示需先执行 `popcorn register`，并被引导至 **Discord 或 GitHub**。
  
  - 另一名用户在注册后遇到了与 **无效或未授权的 X-Popcorn-Cli-Id** 相关的 *Submission error: Server returned status 401 Unauthorized* 错误，而第三名用户报告称在通过浏览器授权执行 `popcorn-cli register` 命令后出现 *error decoding response body*。
- **MI300 AMD-FP8-MM 排行榜竞争激烈**：根据 **GPU MODE** 的 `#submissions` 频道，**MI300** 上的 `amd-fp8-mm` 排行榜提交记录从 **5.24 ms** 到 **791 µs** 不等，展示了该平台上各种性能水平，其中一项提交达到了 **255 µs**。
  
  - 一名用户在 `amd-fp8-mm` 排行榜上刷新了个人最好成绩，在 **MI300** 上达到了 **5.20 ms**，展示了 FP8 矩阵乘法性能的持续改进和优化。
- **Torch 模板取得突破，容差已调整**：在 **GPU MODE** 的 `#amd-competition` 频道中，一位参与者分享了一个改进的模板实现（[以 message.txt 文件形式附带](https://cdn.discordapp.com/attachments/1359640791525490768/1362542150369411212/message.txt?ex=6802c5c3&is=68017443&hm=85d758ae6325f6f1f5faa7cb881b03a1365df5cb69fc1f6c91cebcf3d1dc8032&)），该实现**避免了 torch 头文件**（需要设置 `no_implicit_headers=True`）以缩短往返时间，并配置了正确的 **ROCm 架构**（*gfx942:xnack-*）。
  
  - 参赛者指出 Kernel 输出存在**微小误差**，导致出现 `mismatch found! custom implementation doesn't match reference` 等失败消息，随后管理员放宽了最初**过于严格的容差（tolerances）**。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **NVMe SSD 在速度测试中完胜 SATA SSD**：用户证实 **NVMe SSD** 的速度远超 **SATA SSD**，达到 **2000-5000 MiB/s**，而 **SATA** 仅为 **500 MiB/s**。
  
  - 一名成员指出，在加载大型模型时差距会进一步扩大，并提到*由于磁盘缓存和充足的 RAM，会出现远高于 SSD 性能的巨大峰值*。
- **LM Studio 视觉模型实现仍是个谜**：成员们研究了如何在 LM Studio 中使用 **qwen2-vl-2b-instruct** 等视觉模型，并引用了 [图像输入文档](https://lmstudio.ai/docs/typescript/llm-prediction/image-input)。
  
  - 虽然有人声称成功处理了图像，但其他人报告失败；**Llama 4** 模型在模型元数据中带有视觉标签，但在 llama.cpp 中不支持视觉功能，且 **Gemma 3** 的支持情况尚不确定。
- **RAG 模型获得 '+' 号功能支持**：用户注意到 LM Studio 中的 **RAG** 模型可以通过聊天模式消息提示栏中的 **'+'** 号附加文件。
  
  - 模型的信息页面会显示其是否具备 RAG 能力。
- **Granite 模型在交互式聊天中依然表现出色**：尽管 **Granite** 在大多数任务中通常被认为性能较低，但一位用户更喜欢将其用于通用场景，特别是*交互式流式宠物类聊天机器人*。
  
  - 该用户表示，*当尝试将其放入更自然的语境时，它感觉很机械*，但 *Granite 仍然是目前性能最好的*。
- **AVX 指令集要求限制了旧版 LM Studio 的使用**：一位使用不带 **AVX2** 的旧款 **E5-V2** 服务器的用户询问是否能运行 **LM Studio**，但被告知只有非常旧版本的 **LM Studio** 才支持 **AVX**。
  
  - 建议使用 [llama-server-vulkan](https://github.com/kth8/llama-server-vulkan) 或寻找仍支持 **AVX** 的 LLM。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama 4 适配 Unsloth**：**Llama 4** 的微调支持将于**本周登陆 Unsloth**，同时支持 **7B** 和 **14B** 模型。
  
  - 要使用它，请切换到 **7B notebook** 并将其更改为 **14B** 以访问新功能。
- **自定义 Token 占用内存？**：在 **Unsloth** 中添加自定义 Token 会增加内存消耗，需要用户启用**持续预训练 (continued pretraining)** 并向 **embedding** 和 **LM head** 添加层适配器 (layer adapters)。
  
  - 有关此技术的更多信息，请参阅 [Unsloth 文档](https://docs.unsloth.ai/basics/continued-pretraining#continued-pretraining-and-finetuning-the-lm_head-and-embed_tokens-matrices)。
- **MetaAI 吐露仇恨言论**：一位成员观察到 **MetaAI** 在 **Facebook Messenger** 中生成并随后删除了一条冒犯性评论，接着声称服务不可用。
  
  - 该成员批评 **Meta** 将输出流式传输置于审核之上，建议审核应该像 [DeepSeek](https://www.reddit.com/r/China_irl/comments/1ib4399/deepseek%E4%BB%8E%E4%B8%80%E8%BF%91%E5%B9%B3%E6%95%B0%E5%88%B0%E5%8D%81%E4%B8%80%E8%BF%91%E5%B9%B3/) 那样在服务器端进行。
- **PolyThink 消除幻觉**：一位成员宣布了 **PolyThink** [PolyThink 候补名单](https://www.polyth.ink/) 的开启，这是一个多模型 AI 系统，旨在通过让 AI 模型相互纠错来消除 **AI 幻觉 (AI hallucinations)**。
  
  - 另一位成员将该系统比作“用所有模型进行思考”，并表示有兴趣将其用于**合成数据生成 (synthetic data generation)** 以创建更好的数据集。
- **未经训练的神经网络展现出涌现计算**：一篇[文章](https://techxplore.com/news/2021-12-untrained-deep-neural-networks.html)指出，**未经训练的深度神经网络**可以在没有训练的情况下执行图像处理任务，利用随机权重对图像进行过滤和特征提取。
  
  - 该技术利用神经网络固有的结构来处理数据，而无需从训练数据中学习特定模式，展示了**涌现计算 (emergent computation)**。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GPT4o 在自动补全方面表现挣扎**：成员们讨论了 **GitHub Copilot** 是否使用 **GPT4o** 进行自动补全，有报告称其幻觉出链接并交付了损坏的代码，正如[这些推文](https://x.com/AhmedRezaT/status/1912632437202444375)和[另一篇推文](https://x.com/tunguz/status/1912631402958299312)所记录的那样。
  
  - 普遍观点认为，尽管对自动补全任务寄予厚望，但 **GPT4o** 的表现仅与其他 SOTA LLM 持平。
- **华为可能挑战 Nvidia 的领先地位**：**特朗普的关税**政策可能会使 **华为** 在硬件方面与 Nvidia 竞争，并可能主导全球市场，根据[这篇推文](https://x.com/bio_bootloader/status/1912566454823870801?s=46)和[这段 YouTube 视频](https://www.youtube.com/watch?v=7BiomULV8AU)。
  
  - 然而，一些用户对 **GPT4o-mini-high** 的评价褒贬不一，指出其在 zero-shot 时会出现损坏的代码，且在处理基础 Prompt 时失败。
- **微软研究院发布 BitNet b1.58 2B 4T**：**微软研究院**推出了 **BitNet b1.58 2B 4T**，这是一个原生的 **1-bit LLM**，拥有 20 亿参数，并在 4 万亿 Token 上进行了训练，其 [GitHub 仓库在此](https://github.com/microsoft/BitNet)。
  
  - 用户发现，必须利用专门的 C++ 实现 (**bitnet.cpp**) 才能实现承诺的效率优势，且上下文窗口限制为 4k。
- **Discord 的新年龄验证引发争论**：一位成员分享了对 Discord 在英国和澳大利亚测试的新年龄验证功能的担忧，引用了[此链接](https://programming.dev/post/28771283)。
  
  - 核心问题在于用户担心隐私受损以及平台范围内可能发生的进一步变化。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Gemini Pro 助力会计自动化**：一位成员利用 **Gemini Pro** 为初级会计师的月末流程指南生成了 **TOC**（目录），随后通过 **Deep Research** 对其进行了丰富，并将研究结果整合到 **GDoc** 中。
  
  - 他们将 **GDoc** 作为主要来源整合进 **NLM**，强调了其功能并征求改进反馈。
- **通过 Google Maps 规划度假愿景**：一位成员使用 **Notebook LM** 创建了度假行程，并将兴趣点存档在 **Google Maps** 中。
  
  - 他们建议 **Notebook LM** 应该能够摄取保存的 **Google Maps** 列表作为源材料。
- **投票，投票，否则就错失良机**：一位成员提醒大家 **Webby 投票明天截止**，NotebookLM 在 3 个类别中有 2 个处于落后状态，敦促用户投票并广而告之：[https://vote.webbyawards.com/PublicVoting#/2025/ai-immersive-games/ai-apps-experiences-features/technical-achievement](https://vote.webbyawards.com/PublicVoting#/2025/ai-immersive-games/ai-apps-experiences-features/technical-achievement)。
  
  - 该成员担心如果大家不投票，他们将会落选。
- **NLM 的 RAG 架构图讨论**：针对有关 RAG 的提问，一位成员分享了两张展示通用 RAG 系统和简化版本的图表，详见此处：[Vertex_RAG_diagram_b4Csnl2.original.png](https://cdn.discordapp.com/attachments/1124402182909857966/1362567715360866365/Vertex_RAG_diagram_b4Csnl2.original.png?ex=6802dd92&is=68018c12&hm=2049e59b022a0ef55db1299c859c7c7cc0b89d1e38c0d8c74dcd753279ee08aa&) 和 [Screenshot_2025-04-18_at_01.12.18.png](https://cdn.discordapp.com/attachments/1124402182909857966/1362567811439788143/Screenshot_2025-04-18_at_01.12.18.png?ex=6802dda9&is=68018c29&hm=810954568bef77731790d7162bb04ea24d7588c4831a9bb59e895822a5bae07a&)。
  
  - 另一位用户插话表示他们在 Obsidian 中有自定义的 RAG 设置，随后讨论转向了响应风格。

 

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Obsidian MCP 服务器亮相**：一位成员正在开发 **Obsidian 的 MCP 服务器** 并寻求创意协作，强调主要漏洞在于编排（orchestration）而非协议本身。
  
  - 该开发旨在通过安全的编排接口简化 **Obsidian** 与外部服务之间的集成。
- **使用 SSE API Key 保护 Cloudflare Workers**：一位成员正在使用 **Cloudflare Workers** 和 **Server-Sent Events (SSE)** 配置 **MCP 服务器**，并就如何安全地传递 **apiKey** 征求建议。
  
  - 另一位成员建议通过 HTTPS 加密的 **header** 而非 URL 参数传输 **apiKey**，以增强安全性。
- **LLM 工具理解机制解析**：一位成员询问 **LLM** 如何真正“理解”工具和资源，探讨这仅仅是基于 prompt 解释，还是针对 **MCP spec** 进行了特定训练。
  
  - 澄清指出 **LLM** 通过描述来理解工具，许多模型支持工具规范或其他定义这些工具的参数，从而引导其使用。
- **MCP 服务器的个性化困境**：一位成员试图通过在初始化期间设置特定 prompt 来为他们的 **MCP 服务器** 定义独特个性，以规定响应行为。
  
  - 尽管付出了这些努力，**MCP 服务器** 的响应仍保持不变，这引发了对有效个性化定制技术的进一步研究。
- **HeyGen API 正式登场**：**HeyGen API** 的产品负责人介绍了他们的平台，强调其无需摄像头即可制作引人入胜的视频的能力。
  
  - HeyGen 允许用户在没有摄像头的情况下创建吸引人的视频。

 

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **GRPO Recipe 待办事项列表已过时**：[原始 GRPO recipe 待办事项](https://github.com/pytorch/torchtune/issues/2421) 已过时，因为新版本的 GRPO 正在 [r1-zero repo](https://github.com/joecummings/r1-zero) 中准备。
  
  - 单设备 recipe 将不会通过 **r1-zero repo** 添加。
- **异步 GRPO 版本正在开发中**：异步版本的 **GRPO** 正在一个独立的分支中开发，很快将合并回 **Torchtune**。
  
  - 此次更新旨在增强 **Torchtune** 内部 **GRPO** 实现的灵活性和效率。
- **单 GPU GRPO Recipe 进入最后冲刺阶段**：来自 @f0cus73 的单 GPU **GRPO** recipe PR 可以在[这里](https://github.com/pytorch/torchtune/pull/2467)查看，目前正在进行最后的完善。
  
  - 该 recipe 允许用户在单 GPU 上运行 **GRPO**，降低了实验和开发的硬件门槛。
- **奖励建模 RFC 即将发布**：奖励建模 (Reward Modeling) 的 **RFC** (意见征求稿) 即将发布，将概述实现要求。
  
  - 社区期望该 **RFC** 能为奖励建模提供结构化的方法，促进 **Torchtune** 内部更好的集成和标准化。
- **Titans Talk 即将开始...**：[Titans Talk](https://x.com/SonglinYang4/status/1912581712732909981) 将在 1 分钟后为感兴趣的人开始。
  
  - 算了，当我没说！

 

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Agent 流利支持 A2A 通信**：LlamaIndex 现在支持构建符合 **A2A (Agent2Agent)** 协议的 Agent，该协议由 **Google** 发起，并得到了超过 **50 家技术合作伙伴** 的支持。
  
  - 该协议使 AI Agent 能够安全地交换信息并协调行动，无论其[底层基础设施](https://twitter.com/llama_index/status/1912949446322852185)如何。
- **CondenseQuestionChatEngine 不支持工具调用**：**CondenseQuestionChatEngine** 不支持调用工具；据一名成员称，建议改用 **Agent**。
  
  - 另一名成员确认这只是一个建议，实际上并未实施。
- **Bedrock Converse Prompt Caching 导致混乱**：一名通过 **Bedrock Converse** 使用 **Anthropic** 的成员在尝试使用 Prompt Caching 时遇到问题，在向 *llm.acomplete* 调用添加 *extra_headers* 时报错。
  
  - 移除额外的 header 后错误消失，但响应中缺少预期的 Prompt Caching 字段，例如 *cache_creation_input_tokens*。
- **Anthropic 类可能解决 Bedrock 缓存问题**：有建议称，由于放置缓存点的方式不同，*Bedrock Converse 集成* 可能需要更新才能正确支持 Prompt Caching，目前应使用 `Anthropic` 类。
  
  - 该建议基于成员对原生 **Anthropic** 的测试，并指向了一个 [Google Colab 笔记本示例](https://colab.research.google.com/drive/1wolX4dd2NheesiThKrr5HZh4xebFc0lq?usp=sharing) 作为参考。

 

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 的停滞引发疑虑**：Discord 服务器上的用户对 **GPT4All** 的未来表示担忧，指出已经大约 **三个月** 没有更新或开发者露面了。
  
  - 一位用户表示，*既然一年时间都不算什么大跨度……所以我对及时的更新不抱希望*。
- **IBM Granite 3.3 成为 RAG 的替代方案**：一名成员强调 **IBM 的 Granite 3.3**（拥有 **80 亿参数**）能为 **RAG** 应用提供准确且详尽的结果，并提供了 [IBM 公告](https://www.ibm.com/new/announcements/ibm-granite-3-3-speech-recognition-refined-reasoning-rag-loras) 和 [Hugging Face](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct) 页面的链接。
  
  - 该成员还提到，他们正在使用 **GPT4All** 的 *Nomic embed text* 进行编程函数的本地语义搜索。
- **LinkedIn 询问被无视**：一名成员表示，他们 *在 LinkedIn 上询问了 GPT4All 的状态，但我觉得我被无视了*。
  
  - 未记录进一步的讨论。

 

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 举办线下见面会**：**Modular** 将于下周在其位于**加利福尼亚州洛斯阿尔托斯 (Los Altos, California)** 的总部举办**线下见面会**，并邀请您[在此预约 (RSVP)](https://lu.ma/modular-meetup)。
  
  - 见面会将包含一场关于通过 **Mojo & MAX** 提升 **GPU 性能**的演讲，并提供**线下**和**虚拟参会**两种方式。
- **Mojo 缺少标准 MLIR Dialects**：一位用户发现 **Mojo** 默认情况下不暴露 `arith` 等标准 **MLIR dialects**，仅提供 `llvm` dialect。
  
  - 会议澄清，目前在 **Mojo** 中*没有注册其他 dialects 的机制*。
- **Mojo 字典指针触发 Copy/Move**：一位用户观察到，在 **Mojo** 中使用 `Dict[Int, S]()` 和 `d.get_ptr(1)` 获取字典值的指针时，会出现非预期的 `copy` 和 `move` 行为。
  
  - 这引发了一个问题：*"为什么获取字典值的指针会调用 copy 和 move？🤯 这在任何意义上是预期的行为吗？"*
- **max 仓库需要孤儿清理机制 (Orphan Cleanup Mechanism)**：一位成员强调了 `max` 仓库需要**孤儿清理机制**，该问题最初于几个月前在 [issue 4028](https://github.com/modular/max/issues/4028) 中提出。
  
  - 这一特性对于磁盘分区较小的开发者尤为重要，尽管它被归类为*纯粹的开发端问题*。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **自动形式化工具寻求业务逻辑证明**：一位成员询问如何使用 **Lean 自动形式化工具 (auto-formalizer)** 从包含业务逻辑的计算机代码中生成非正式证明，以便利用 **AI 证明生成**进行**程序的形式化验证 (formal verification)**。
  
  - 讨论重点关注了 **Python** 和 **Solidity** 等编程语言。
- **CIRIS Covenant Beta：开源对齐 (Alignment)**：**CIRIS Covenant 1.0-beta** 已发布，这是一个用于自适应相干 **AI 对齐 (AI alignment)** 的开源框架，并提供 [PDF](https://www.ethicsengine.org/The%20CIRIS%20Covenant%20-%20beta%201.pdf) 下载。
  
  - 该项目旨在帮助 **AI 安全**或**治理**领域的人员，项目中心和评论门户可在[此处](https://www.ethicsengine.org/ciris)访问。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **寻求集成预测模型 (ensembling forecasting models) 的资源**：一位成员询问有关**集成预测模型**的资源。
  
  - 消息记录中未提供相关资源。
- **成员集思广益毕业设计 (Final Year Project) 想法**：一位成员正在为 AI 专业的学士学位寻找实用的**毕业设计想法**，对计算机视觉、NLP 和生成式 AI 感兴趣，特别是能解决现实世界问题的项目。
  
  - 他们正在寻找构建难度适中的项目。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **AI 模型开发通常是分阶段的吗？**：一位用户询问 **AI 模型开发**是否通常采用**分阶段训练过程**，例如实验阶段 (20-50%)、预览阶段 (50-70%) 和稳定阶段 (100%)。
  
  - 该问题探讨了 **AI 模型训练**中**分阶段方法**的普遍性，涉及实验版、预览版和稳定版发布等阶段。
- **分阶段训练在 AI 中常见吗？**：讨论集中在 **AI 模型**使用**分阶段训练过程**的情况，将其分解为实验、预览和稳定等阶段。
  
  - 用户试图了解这种**分阶段部署**是否是该领域的常见做法。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 的 Jetbrains 更新日志发布**：最新的发布说明现已在 [Windsurf 更新日志](https://windsurf.com/changelog/jetbrains)中上线。
  
  - 鼓励用户查看更新日志，以了解最新的功能和改进。
- **新讨论频道开启**：一个新的讨论频道 <#1362171834191319140> 已开启，用于社区交流。
  
  - 旨在为用户提供一个专门的空间来分享想法、建议，并询问有关 Windsurf 的问题。

---

**DSPy Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**tinygrad (George Hotz) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

# 第二部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **Perplexity AI ▷ #**[**announcements**](https://discord.com/channels/1047197230748151888/1047204950763122820/1362539434003796120) (1 条消息):

> `Telegram Bot, WhatsApp Bot`

- **Perplexity AI 发布 Telegram Bot！**：**Perplexity AI** 现在可以通过 [askplexbot](https://t.me/askplexbot) 在 **Telegram** 上使用。
  
  - 用户可以将该 Bot 添加到群聊或直接私信（DM），以实时获取带有来源的回答；**WhatsApp** 集成已在计划中。
- **发布预告视频**：随公告发布了一段展示新 Telegram Bot 的短预告视频。
  
  - 该视频 ([Telegram_Bot_Launch_1.mp4](https://cdn.discordapp.com/attachments/1047204950763122820/1362539434565697779/Telegram_Bot_Launch_1.mp4?ex=6802c33b&is=680171bb&hm=bf0a3730711b37e0eadba26f66492cb2608b1a3f224b590533865dce7f713f37&)) 重点展示了无缝集成和实时响应能力。

 

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055588/1362140519639290080) (952 条消息🔥🔥🔥):

> `投诉工单 Bot, 推荐链接帮助的图片附件, Neovim 配置展示, Claude 3 vs Gemini 2.5 性能, Perplexity Voice Mode`

- **Perplexity 考虑在 Discord 中加入工单 Bot 和 Helper 角色**：成员们讨论了在 Discord 服务器中添加 **ticketing bot**（工单 Bot）处理投诉的可能性，但对可行性表示担忧，更倾向于使用 [help center](https://link.to/help-center)（帮助中心）的方式，并担心出现重复的抱怨。
  
  - 另一个建议是创建一个带有专用 Ping 权限的 **helper role**（助手角色），同时指出 Discord 并非理想的支持平台，而一个直接链接到 Zendesk 的 *Modmail bot* 可能会更有用。
- **关于推荐链接的学生证图片讨论**：一位成员询问是否可以在频道中上传图片，以寻求关于使用学生证创建推荐链接的帮助。
  
  - 另一位成员回应称“没人在意”，并建议他们也可以在相关频道 <#1118264005207793674> 中询问。
- **成员展示其 Neovim 配置**：一位成员在连续学习了三天 IT 后，发布了一张其 **Neovim configuration** 的[图片](https://cdn.discordapp.com/attachments/1047649527299055688/1362486729340092496/image.png?ex=68029226&is=680140a6&hm=7a9871a1c9146850518afcdf9f96e8c64589fab3f554c6932fd6e8e958f7c59c&)。
  
  - 他们利用 AI 模型让学习变得“有趣”，效果很不错。
- **Claude 3.7 Sonnet Token 申请**：成员们讨论了 [Claude 3.7 Sonnet](https://artificialanalysis.ai/models/claude-3-7-sonnet-thinking) 模型，有人建议成员应该申请 OpenAI 的免费 Token。
  
  - 一位成员表示他们已经使用了 O3（包括高推理和中推理模式），一整天都没有用完。
- **Perplexity 用户期待承诺的新模型**：成员们正在等待承诺的模型发布，包括 **O3, O4-mini, Gemini 2.5 Flash, DeepSeek R2, Qwen 3 和 Claude 3.7 Opus**。
  
  - 一位成员在发布前质疑 Gemini 2.5 Flash 的价值，另一位成员回应称“它甚至还没发布”。

 

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1362197905192128513) (3 条消息):

> `关税, 特朗普, 欧盟, 中国`

- **特朗普对欧盟和中国征收关税**：根据[这份报告](https://www.perplexity.ai/page/trump-imposes-245-tariffs-on-c-LbKOTTe8TyWXY_ov9vUb_A)，特朗普政府对欧盟产品征收了 **245% 的关税**，以报复其对空中客车（Airbus）的补贴，并因知识产权盗窃对中国商品征收关税。
  
  - 这些措施旨在保护美国工业并解决贸易失衡问题。
- **对空客补贴的报复**：根据[此搜索结果](https://www.perplexity.ai/search/2015983a-7aa5-48e9-a0b1-e98ea9d4ccbc#0)，对欧盟商品的关税是对欧盟补贴空中客车的回应，美国认为这使这家欧洲公司在全球市场中获得了不公平的优势。
  
  - 美国寻求建立公平的竞争环境并确保公平竞争。
- **来自中国的知识产权盗窃**：如[此 Perplexity 搜索](https://www.perplexity.ai/search/e2a38d65-9a5b-4e02-83a7-42433999f5cd)所述，为了打击知识产权盗窃，美国还对中国商品征收了关税，旨在保护其技术和创新资产。
  
  - 美国寻求阻止中国从事不公平贸易行为。

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1362157518184255508) (3 条消息):

> `求职帖子，PplxDevs 推文，推迟到六月`

- **分享 PplxDevs 推文**：一位成员分享了来自 **PplxDevs** 的 [推文](https://x.com/pplxdevs/status/1912578212988874891?s=61) 链接。
- **用户本想参与**：一位成员表示 *这太棒了，如果不是因为我要去度假一个月，我肯定会全身心投入。*
- **请求将活动移至六月**：一位成员开玩笑地请求 *你们绝对应该把它改到六月*。

 

---

### **LMArena ▷ #**[**general**](https://discord.com/channels/1340554757349179412/1340554757827461211/1362140529088921621) (1257 条消息🔥🔥🔥):

> `Gemini 2.5 Pro 对比 Flash，O3 对比 O4 Mini，思维预算参数，LLM 辅助学习？，OpenAI 成本效率`

- **Gemini 2.5 Flash 进入竞技场**：成员们注意到 [Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models) 出现在 Vertex AI 中，引发了关于其与 **Gemini 2.5 Pro** 性能对比的辩论，一些人因其效率和 tool-calling 能力而更倾向于将其用于编程任务。
  
  - 一位用户夸赞道 *哟，这个模型太火了！！！*，而几位成员指出模型在测试期间陷入了思维循环（thinking loops），这种情况之前在 2.5 Pro 上也发生过。
- **O3 与 O4 的霸权之争**：关于 **O3** 与 **O4 Mini** 优劣的讨论正在进行中，成员们进行了实时测试，并分享了像 [这个游戏](https://liveweave.com/A9OGzH#) 这样的链接来展示两者的潜力。似乎许多人在尝试了 Flash 和 Pro 模型后又回到了 O3。
  
  - 虽然 **O4 Mini** 乍看之下可能更便宜，但一些人发现其高昂的使用和输出成本使其在实践中几乎无法使用，导致基于特定需求产生了不同的看法。
- **思维预算调整释放新能力**：成员们正在深入研究 Vertex AI 上新的 "Thinking Budget" 功能，该功能允许操作 thinking tokens。
  
  - 虽然有些人觉得它很有用，但其他人报告了 bug，一些人观察到 *2.5 Pro 在 0.65 temp 下表现更好*。
- **LLM：学习伙伴还是幻觉中心？**：一位成员提出了 LLM 是否可以帮助发展中国家教育的问题，引发了关于可靠性与传统学习的辩论。
  
  - 一些成员表示 LLM *会产生幻觉，而由懂行的人编写的书籍则不会*。
- **OpenAI 资金流失**：成员们讨论了 OpenAI 每年亏损 80 亿美元。
  
  - 成员们怀疑 *我不认为 OpenAI 在目前的模型组 API 定价上会亏钱*。

 

---

### **LMArena ▷ #**[**announcements**](https://discord.com/channels/1340554757349179412/1343296395620126911/1362475485992452225) (2 条消息):

> `LMArena 公司成立，Beta 版发布，反馈响应`

- **LMArena 成立公司！**：源自加州大学伯克利分校学术项目的 LMArena [正在成立一家公司](https://blog.lmarena.ai/blog/2025/new-beta/) 以支持其平台和社区，确保其保持中立、开放和易于访问。
- **Beta 版发布并包含最新反馈**：LMArena 发布了 [Beta 版本](https://beta.lmarena.ai)，整合了来自 Alpha 版的用户反馈，不过保存的聊天记录不会迁移，投票将受到监控以确保信号质量。
- **LMArena 采用深色模式和图片粘贴**：LMArena 正在积极响应 Beta 版反馈，增加了 **深色/浅色模式切换** 以及直接在提示框中 **复制/粘贴图片** 的功能。

 

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1362141228149244067) (832 messages🔥🔥🔥):

> `code2prompt, Aider's new command, Gemini 2.5 vs O3/O4, DeepSeek R2`

- **Aider 用户讨论 code2prompt 的实用性**：成员们讨论了在 Aider 中使用 `code2prompt` 的情况，质疑其相对于简单使用 `/add` 来包含必要文件的实用性，因为 `code2prompt` 几乎能瞬间解析所有匹配的文件。
  
  - `code2prompt` 的主要优势在于其解析相关文件的速度，但最终的实用性取决于具体的用例和模型能力。
- **Architect 模式导致生成 15 个新文件**：一位成员报告了一个 Aider 的 bug，在重构一个应用并创建了 **15 个新文件** 后，在接受向聊天中添加文件时，更改被丢弃了。
  
  - 另一位成员指出，这种行为是符合预期的，并且*在编辑被丢弃时不会给出警告*。
- **Aider 的新命令：Probe Tool**：成员们讨论了用于语义代码搜索的新 Aider 命令 `probe` 工具，强调了其提取带错误的代码块以及与测试输出集成的能力，开发者正在其公司内部积极使用它。
  
  - 另一位成员表示他很喜欢 [claude-task-tool](https://github.com/paul-gauthier/claude-task-tool)，认为这两个都是很棒的项目。他还提到了另一个工具 [potpie-ai/potpie](https://github.com/potpie-ai/potpie)，认为其 UI 也非常出色。
- **Gemini 2.5 Pro vs O3/O4**：成员们辩论了 **O3-high** 与 **Gemini 2.5 Pro** 的优劣，引用基准测试显示 **O3-high** 在 Aider Polyglot 基准测试中达到了 **81.3%**，而 **Google** 为 **72.9%**，同时其他人指出了 **O4-mini** 的性价比，也有人表示 **O3** 太贵了。
  
  - 然而，Gemini 发布了一个新版本，可以禁用 *"thinking"* 并拥有更好的定价。
- **对 DeepSeek R2 的热情高涨**：成员们热切期待 **DeepSeek R2** 的发布，希望其性能能超越 **O3-high**，并以免费或极低成本提供，一位成员评论说[这只是时间问题](https://discord.com/channels/1131200896827654144/1131200896827654149/1362174807180967946)。
  
  - 普遍乐观地认为 **DeepSeek R2** 凭借最佳的性价比，可能会对 **OpenAI** 的主导地位造成最终打击。

 

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1362189197858963637) (23 messages🔥):

> `Ask Mode Persistence, O4-mini Error Fix, Copy-Context Usage, Cloud Aider Instances, Architect & Edit Format Split`

- **使用 `/chat-mode ask` 使 Ask 模式持久化**：用户可以使用 `/chat-mode ask` 持久地停留在 ASK 模式，但这可能会引起混淆，因为帮助菜单显示的是 *"switch to a new chat mode"*（切换到新聊天模式）而不是 *"switch to a* ***different*** *chat mode"*（切换到**不同**的聊天模式）。
  
  - 有建议称，在配置文件中将 `ask` 指定为默认模式也应使其持久化，根据[文档](https://aider.chat/docs/usage/modes.html)，单独输入 `/ask` 并按回车也会切换到 ask 模式。
- **O4-mini 温度参数错误已解决**：报告了一个关于 **o4-mini** 的错误（*Unsupported value: 'temperature' does not support 0 with this model*），通过升级到[最新版本](https://aider.chat/docs/version_history.html)已解决。
  
  - 该用户使用的是 v0.74.1，Paul G. 确认最新版本已支持 **o4-mini**。
- **在 ChatGPT 中简化 Copy-Context 的使用**：一位用户询问如何将 `copy-context` 的输出粘贴到 **ChatGPT** 中以获得最佳效果，包括首选模型（例如 gemini-flash）。
  
  - 一张截图显示了在 Aider 中配合 **4o-mini** 运行的 ChatGPT **o4-mini**，但未指明粘贴到 ChatGPT 的最佳方法。
- **云端的 Aider 实例**：一位用户询问关于在云端运行 **aider instance** 并通过工作流发送 prompt 的问题。
  
  - 另一位用户确认他们正在这样做，并请提问者私信他们。
- **分离 Architect 和 Edit 格式**：一位用户询问关于为 **architect** 和 **edit** 功能分离编辑格式的问题，以及这是否从 `.aider.model.settings.yml` 中提取。
  
  - 建议参考[代码](https://github.com/paul-gauthier/aider/blob/main/aider/coders/architect_coder.py)，为普通编辑配置 `edit_format: diff`，为 architect 模式配置 `editor_edit_format: editor-diff`。

 

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1362170402155266301) (2 条消息):

> `新模型分析，O'Reilly AI 活动`

- **YouTube 提供对新模型的冷静见解**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=3aRRYQEb99s)，对新模型提供了*更为理性的看法*。
  
  - 该视频似乎对近期发布的模型提供了**分析**或**观点**，避开了过度炒作。
- **O'Reilly 举办免费 AI 活动**：一位成员提到了一个专注于 **AI 编程 (Coding with AI)** 的 [免费 O'Reilly 活动](https://www.oreilly.com/CodingwithAI/)。
  
  - 该活动可能涵盖与 **AI 开发**、**应用**相关的主题，并可能包含**动手编程环节**。

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1362158908059160597) (122 条消息🔥🔥):

> `服务条款与隐私政策更新，免费模型限制，Gemini 2.5 Flash 模型`

- **OpenRouter 更新服务条款和隐私政策**：OpenRouter 更新了其[服务条款和隐私政策](https://openrouter.ai/privacy)以符合现状，明确了**未经同意不会存储 LLM 输入**，并详细说明了他们如何进行提示词分类 (prompt categorization)。
  
  - 提示词分类旨在确定请求的“类型”（编程、角色扮演等），以支持排名和分析功能；如果用户未选择开启日志记录，该过程将是匿名的。
- **免费模型限制修订：计入终身购买额**：OpenRouter 更新了免费模型限制，现在**要求终身购买至少 10 个积分**才能享受更高的 **1000 次请求/天 (RPD)** 额度，无论当前积分余额是多少。
  
  - 由于需求极高，实验性模型 `google/gemini-2.5-pro-exp-03-25` 的免费访问权限现在仅限于购买过至少 10 个积分的用户，而[付费版本](https://openrouter.ai/google/gemini-2.5-pro-preview-03-25)则提供不间断访问。
- **OpenRouter 推出 Gemini 2.5 Flash：极速推理**：OpenRouter 发布了 **Gemini 2.5 Flash**，这是一款专为高级推理、编程、数学和科学任务设计的新模型，提供[标准](https://openrouter.ai/google/gemini-2.5-flash-preview)（非思考）变体和带有内置推理令牌的 [:thinking](https://openrouter.ai/google/gemini-2.5-flash-preview:thinking) 变体。
  
  - 用户可以使用 `max tokens for reasoning` 参数自定义 **:thinking** 变体的使用，详见[文档](https://openrouter.ai/docs/use-cases/reasoning-tokens#max-tokens-for-reasoning)。

---

### **OpenRouter (Alex Atallah) ▷ #**[**app-showcase**](https://discord.com/channels/1091220969173028894/1092850552192368710/1362390300449968179) (2 条消息):

> `LLM 成本模拟器，Vibe-coded LLM 聊天应用`

- **模拟 LLM 对话成本**：一位成员创建了一个工具，用于模拟 LLM 对话的成本，支持 [OpenRouter](https://llm-cost-simulator.vercel.app) 上的 **350 多种模型**。
- **Vibe-coded LLM 聊天应用连接 OpenRouter**：一位成员开发了一款连接 [OpenRouter](https://chat.nanthai.tech/chat) 的 LLM 聊天应用，提供精选的 LLM 列表以及网络搜索和 RAG 检索等额外功能。
  
  - 该应用提供基础免费层级，扩展搜索和 RAG 功能每月收取少量费用，或者支付略高费用以获得无限使用权限。

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1362140794395426871) (636 条消息🔥🔥🔥):

> `OpenAI Codex 在 OpenRouter 上无法工作、BYOK、DeepSeek R3 和 R4、OpenAI 验证、API 使用限制`

- **OpenAI API 端点在 OpenRouter 中失效，Codex 用户叫苦不迭**：OpenAI **Codex** 使用了新的 API 端点，因此目前无法在 **OpenRouter** 上运行。
- **由于 OpenAI 限制，DeepSeek 延迟**：由于 **OpenAI** 要求身份验证的限制，来自 **OpenRouter** 的 **o-series reasoning summaries**（o 系列推理摘要）可能会延迟，不过摘要很快将通过 **Responses API** 提供。
  
  - 一位用户指出，新的 DeepSeek 与 Google 的 [Firebase studio](https://firebase.google.com/) 类似。
- **OpenAI 验证流程**：用户讨论了 **OpenAI** 针对 **O3** 的侵入性验证流程，该流程要求提供身份证照片，并通过 [withpersona.com](https://withpersona.com/) 进行带有活体检测的自拍。
- **OpenRouter API 限制以及关于速率和配额的困惑**：**OpenRouter** 的免费用户总计每天可获得 1000 次请求，但对特定模型的巨大需求可能会导致针对单个用户的请求限制。
  
  - 用户还就 **Google AI studio** 的免费用户是否存在 **RPD**（每日请求数）配额，还是真正的免费层级（以数据支付）进行了讨论和争论。
- **OpenRouter 日志政策说明即将发布**：目前，免费模型提供商通常会记录输入/输出，但 **OpenRouter** 本身*仅在*用户通过账户设置明确选择加入时才会记录数据，各提供商政策的文档即将发布。
  
  - 一个名为 `training=false` 的独立设置可以禁用那些使用你的数据进行训练的提供商。

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1362151392822956052) (476 条消息🔥🔥🔥):

> `Gemini 2.5 Pro 对比 o3/o4 模型、o4 模型幻觉、GPT-4.1 Nano、Gemini 2.5 Flash`

- **对某些人来说，Gemini 2.5 Pro 仍然优于 o3/o4**：尽管推出了 **o3** 和 **o4** 等新模型，一些用户仍然更喜欢 [Gemini 2.5 Pro](https://ai.google.dev/)，因为它的*速度、准确性和成本*，尽管其他人发现 **Gemini** 在处理复杂任务时*幻觉极其严重*。
  
  - 一些基准测试显示 **o3** 在编程方面表现更好，而 **Gemini 2.5 Pro** 在推理方面表现出色；新的 **2.5 Flash** 版本旨在提高速度，但可能会牺牲质量。
- **O4 模型容易产生幻觉**：用户报告称 **o4-mini** 和 **o3** 模型往往*更容易编造信息*，甚至提供看似可信但完全错误的答案，一个例子显示模型为一家公司幻觉出了一个虚假的商业地址。
  
  - 有建议称，指示模型*通过搜索验证来源*可能有助于减少幻觉，但信任模型能正确获取来源本身也是一个问题。
- **GPT-4.1 Nano 在事实性回答方面表现出色**：一位用户发现，与其他模型相比，[GPT-4.1 Nano](https://platform.openai.com/docs/models/gpt-4) 在*回答非虚假信息*方面做得更好。
  
  - 它似乎混淆了开曼群岛和塞舌尔群岛，这就像把英国和墨西哥混为一谈。
- **Gemini 2.5 Flash 首次亮相，主打速度**：Google 在 [Gemini Advanced 中添加了 Veo 2 和新的 2.5 Flash 模型](https://ai.google.dev/)，**Flash** 强调更快的响应速度，但一些用户怀疑它是否为了速度而牺牲了质量。
  
  - 有一个设置 **thinking limits**（思考限制）的选项，可以控制模型的响应速度。

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1362166123042177194) (19 条消息🔥):

> `o4-mini 对比 o3-mini 的知识表现、GPT-4.5 速度、Custom GPT 指令被忽略、o4-mini 使用限制、用于学习的 PDF 上传`

- **o4-mini 不擅长知识类问题？**：用户报告称，新的 **o4-mini** 和 **o4-mini-high** 模型在回答知识类问题和编程任务方面的表现不如 **o3-mini-high**。
  
  - 一位用户指出：*“对于知识类问题，GPT-4.5 应该好得多”*。
- **GPT-4.5 非常慢**：多位用户抱怨 **GPT-4.5** 速度非常慢。
  
  - 有建议认为这*“可能是因为它是一个稠密模型（dense model），而不是混合专家模型（mixture of experts）”*，这也是*“为什么 4.5 如此昂贵”*的原因。
- **Custom GPT 忽略指令**：一位用户报告他们的 **Custom GPT** 没有遵循指令。
  
  - 他们感叹道：*“它现在完全是在放飞自我”*。
- **o4-mini 使用限制**：**o4-mini** 的使用限制为**每天 150 次**，**o4-mini high 每天 50 次**，**o3 每周 50 次**。
- **上传 PDF 的最佳语言模型**：一位用户询问在 ChatGPT 中上传 PDF 进行学习、提问并准备现成的考试题目时，最合适的语言模型是什么。

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1362397402287313048) (9 条消息🔥):

> `图像生成，GPTPlus 上的上下文记忆，多模块系统`

- **模糊图像请求引发 Prompt 探索**：一位成员请求用于生成某公众人物在商店（可能是 **Kohl's**）前手持 **Victoria's Secret** 购物袋跑步的 [模糊、偷拍风格图像](https://www.reddit.com/r/chatgpt) 的 Prompt，该图像是在 **4o image generation** 发布后不久创建的。
  
  - 另一位成员提议尝试编写一个 Prompt，并指出细化此类图像的真实细节非常困难，而另一位成员则建议该图像可能是由 **Sora** 生成的，其效果优于 **4o** 模型的图像生成。
- **GPTPlus 账号模拟上下文记忆**：一位成员报告称，尽管官方尚未提供上下文记忆功能，但通过叙事连贯性在 **GPTPlus** 账号上模拟出了上下文记忆。
  
  - 另一位成员确认了类似的体验，使用户能够构建一个包含 **30 多个讨论** 的多模块系统，并能通过编写特定的关键词来连接新的讨论。

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1362397402287313048) (9 条消息🔥):

> `图像生成 Prompt，r/chatgpt subreddit 图像，GPT Plus 账号上的文本 Prompt，多模块系统`

- **寻求模糊公众人物图像的 Prompt**：一位用户正在寻找用于在 **r/chatgpt subreddit** 上生成某公众人物在商店（**Kohls**）前手持 **Victoria's Secret 购物袋** 跑步图像的 Prompt，该图像看起来像是模糊的偷拍快照。
  
  - 另一位成员提议尝试为该图像生成编写 Prompt，同时推测它可能是由 **Sora** 生成的。
- **模拟上下文记忆的文本 Prompt**：一位用户报告称，尽管据称该功能不被允许，但通过在 **GPT Plus 账号** 上使用具有叙事连贯性的文本 Prompt 模拟出了 **上下文记忆**。
  
  - 他们构建了一个包含 30 多个讨论的**多模块系统**，通过使用正确的关键词来连接新的讨论。

---

### **Cursor Community ▷ #**[**general**](https://discord.com/channels/1074847526655643750/1074847527708393565/1362140546000490536) (432 条消息🔥🔥🔥):

> `Cursor 订阅退款问题，DeepCoder 14B，GPT 4.1 定价，Cursor 终端卡死修复，Zed AI IDE`

- **用户等待 Cursor 订阅退款**：用户报告称取消了 **Cursor 订阅** 并收到了关于退款的电子邮件，但尚未收到款项，不过有一位用户报告称已经*收到了*。
- **调整 FIOS 修复网速**：一位用户注意到，在他们的 **Verizon FIOS** 设置中按下 **LAN 有线连接** 接口可以将下载速度从 **450Mbps 提升到 900Mbps+**。
  
  - 他们建议使用类似于 **PCIE 风格** 的更稳固的连接器，并发布了[他们的设置图像](https://cdn.discordapp.com/attachments/1074847527708393565/1362144114031988766/image.png?ex=6802a490&is=68015310&hm=08c3d65050b698b43658ba08dabb49b96bd56ace10bc982e857fbbedfcd9e502)。
- **模型在 MacBook 上上线**：用户讨论了 **MacBook** 版 Cursor 上新模型的可用性，部分用户最初没有看到这些模型，需要重启或重新安装软件。
  
  - 建议通过输入 **o4-mini** 并点击添加模型来手动添加 **o4-mini** 模型。
- **Cursor 编程助手引发骚动：崩溃、困惑与取消**：用户争论 **o3/o4-mini** 是否优于 **2.5 Pro** 和 **3.7 thinking**，其中一人报告称 **o4-mini-high** 在分析代码库和解决逻辑方面优于 **4o** 和 **4.1**，即使是大型项目也是如此。
  
  - 其他人抱怨 **Cursor Agent** 在运行命令后不退出终端（导致无限期挂起）、频繁的工具调用（tool calls）却无代码输出、消息过长问题、连接状态指示器损坏以及无法编辑文件。
- **Windsurf 与 Zed：物有所值的赢家还是 Vibe Coding 工具？**：有推测称 **OpenAI** 可能会以 **30 亿美元** 收购 **Windsurf**，一些人认为这是公司变得更像 Microsoft 的迹象，而另一些人（如[这条推文](https://x.com/chuhang1122/status/1912786904812294312)）则关注 **GPT4o mini**。
  
  - 参与者争论 **Cursor** 和 **Windsurf** 是真正的 IDE，还是仅仅是带有针对 **Vibe Coders** 的 UX 产品、扩展程序或纯文本编辑器的美化版 API 封装（或分支）。

---

### **Yannick Kilcher ▷ #**[**general**](https://discord.com/channels/714501525455634453/986699377257119794/1362141346571227327) (332 messages🔥🔥):

> `Brain connectivity, Responses API, Liquid State Machines, Meta-Simulation`

- **大脑连接性分解为简单流形**：一篇论文建议将大脑连接性分解为具有额外长程连接的简单流形 ([PhysRevE.111.014410](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.111.014410))，但一位成员觉得任何使用 **Schrödinger's equation** 的东西都被称为“量子”这一点令人厌烦。
  
  - 另一位成员表示，*任何平面波线性方程都可以通过使用* ***Fourier transform*** *转化为 Schrödinger's equation 的形式。*
- **Responses API 是全新的**：成员们澄清说，虽然 **Responses API** 是全新的，但 **Assistant API** 将于明年停止服务。
  
  - 有人强调，*如果你想要助手功能，就选择 Assistant API；如果你想要常规基础功能，就使用 Responses API*。
- **Reservoir Computing 炒作**：成员们讨论了 **Reservoir Computing**（一种**固定的高维动力系统**），并澄清说 *reservoir 不一定非得是软件 RNN，它可以是任何具有时间动力特性的东西*，并且可以从该动力系统中学习到一个*简单的读取方式（readout）*。
  
  - 一位成员分享道：*大多数关于 "reservoir computing" 的炒作通常是***将一个非常简单的想法包装在复杂的术语或奇异的设置中***：拥有一个动力系统。不训练它。只训练一个简单的读取层。*
- **Meta 发布 Fair 更新**：Meta 发布了针对**感知、定位和推理**的 Fair 更新，并附带了一张图片。
  
  - 图片展示了作为 Meta **fair updates** 一部分的一些更新内容。
- **Meta-Simulation 实现复杂问题分布**：成员们讨论了 **Meta-Simulation**，假设有足够的随机轨迹，就可以覆盖所有情况；*这个想法是，我们拥有如此多的随机性和如此多的轨迹，以至于它可能覆盖了一切。*
  
  - 成员们推论，训练后的 **RWKV 7** 将模仿 Reservoir Computer 的学习过程，使得 RWKV 7 能够适应任何其他类型的问题，前提是有办法随机采样那些或多或少符合你预期的真实数据样貌的东西。

---

### **Yannick Kilcher ▷ #**[**paper-discussion**](https://discord.com/channels/714501525455634453/1045297868136779846/1362155882506490006) (10 messages🔥):

> `Ultrascale Playbook Review, GPU layouts for large models, InternVL3 paper discussion`

- **Ultrascale 评论者停止更新，转向 GPU 布局兴趣**：一位成员决定停止评测 **Ultrascale Playbook**，因为他们对学习用于大模型训练和推理的 **GPU layouts** 更感兴趣，而不是底层的 kernel 优化。
  
  - 如果以后需要学习底层 kernel 优化，他们会重新开始评测。
- **多模态模型探索：InternVL3 的训练配方**：成员们安排了一次对论文 **InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models** ([huggingface link](https://huggingface.co/papers/2504.10479)) 的初步阅读和评论。
  
  - 该活动已在社交媒体上推广，并附带了 [Discord event link](https://discord.gg/TeTc8uMx?event=1362499121004548106)。

---

### **Yannick Kilcher ▷ #**[**ml-news**](https://discord.com/channels/714501525455634453/853983317044756510/1362207464757530755) (8 messages🔥):

> `IBM Granite, Trump Administration Deepseek Ban, Brain Matter Music, Infantocracy, Blt weights`

- **IBM 推出 Granite 3 和语音识别**：IBM 发布了 **Granite 3** 和经过优化的推理 **RAG Lora** 模型，以及一套新的语音识别系统，详见[此公告](https://www.ibm.com/new/announcements/ibm-granite-3-3-speech-recognition-refined-reasoning-rag-loras__._astro_.__)。
- **特朗普政府考虑禁止 Deepseek**：据 [TechCrunch 文章](https://techcrunch.com/2025/04/16/trump-administration-reportedly-considers-a-us-deepseek-ban/)报道，**特朗普政府**据传正在考虑在**美国禁用 Deepseek**。
- **脑物质旋律**：科学家和艺术家与已故作曲家 **Alvin Lucier** 合作，利用从他的白细胞中培养出的大脑类器官创建了一个艺术装置，如[这篇 Popular Mechanics 文章](https://www.popularmechanics.com/technology/robots/a64490277/brain-matter-music/)所述。
- **BLT 权重终于发布**：**BLT** 的权重现已可用，他们正在与 Hugging Face 合作提供 Transformer 支持，根据[这个 Hugging Face 集合](https://huggingface.co/collections/facebook/blt-6801263d4ac1704702a192a6)显示。

---

### **Manus.im Discord ▷ #**[**general**](https://discord.com/channels/1348819876348825620/1349440650495398020/1362140596885520587) (347 条消息🔥🔥):

> `封禁讨论、Claude UI 更新、使用 Manus 进行游戏开发、AI 工具、游戏引擎`

- **成员因干扰他人被封禁**：一名成员被封禁，据另一名成员称，“他已经让所有人感到厌烦并受到了警告。”
  
  - 一些成员对封禁缺乏证据表示担忧，而另一些成员则强调了维持社区和平的重要性。
- **Claude 获得了 Charles 级别的更新**：**Claude** 更新了其界面，支持原生研究以及与 **Google Drive** 和 **Calendar** 等应用的连接。
  
  - 成员们注意到 **Manus** 也有类似的功能，这*基本上就是 Charles III 版本的更新*。
- **GPTs 获得 MCPS**：一位成员表示 **GPT** 现在也拥有了这项功能。
  
  - 该更新看起来可以搜索你的 **Google Calendar**、**Google Drive** 以及连接的 gmail。
- **AI 游戏开发迎来重大突破**：一位成员对开源游戏开发与 **AI** 以及公平的 **NFT** 实现相结合表示兴奋。
  
  - 该成员感谢另一位成员启发了关于**抽卡游戏 (gacha games)** 成功原因以及玩家与加密/NFT 社区之间矛盾的思考。
- **游戏体力经济学讨论**：一位成员提出了一种新的体力系统，对不同体力水平提供奖励，使各种类型的玩家和开发者受益。
  
  - 成员们讨论了替代方案，例如通过质押道具换取体力，或引入损失机制以增加成瘾性，并参考了 **MapleStory** 和 **Rust** 等游戏。

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1362143636556484658) (310 条消息🔥🔥):

> `AI 生成内容、真人认证、受 AI 影响的帖子、随机环境、预训练中的类 LoRA 风格化`

- **AI 研究警示：LLM 必须学会产生新鲜结果**：一位成员警告说，在服务器上仅使用 LLM 进行研究或生成消息是徒劳的，可能会导致封禁，因为这贡献了*几乎为零的实际内容*。
  
  - 另一位成员强调服务器的重点是 *AI 研究* 讨论，而非通用的 AI 学习或用户体验，建议新人应效仿成熟的研究方法论以做出有意义的贡献。
- **即将到来的机器人末日？**：成员们讨论了在服务器上引入*真人认证*以对抗 AI 机器人的潜在需求，有人认为目前的低影响可能不会持续太久。
  
  - 讨论了严格验证之外的替代方案，包括社区调解和关注活跃贡献者，同时也对 AI 影响内容日益盛行表示担忧。
- **AI 幻觉悖论：过度努力的模型**：人们对*伪对齐 (pseudo-alignment)* 的危险表示担忧，即 LLM 的谄媚行为误导人们认为它们已经学会了，而实际上它们只是依赖 AI 生成听起来合理的想法大杂烩。
  
  - 一位成员分享了一篇关于[权限随时间变化](https://arxiv.org/abs/2407.14933)的论文，另一位成员对此回复称：*开放网络目前正因 AI 的存在而受到实质性的破坏*。
- **欧洲语言模型的探索**：成员们讨论了欧洲境内除了 Mistral 等知名实体外，*区域定制语言模型*的可获得性。
  
  - 讨论涉及荷兰的 [GPT-NL 生态系统](https://www.computerweekly.com/news/366558412/Netherlands-starts-building-its-own-AI-language-model)、意大利的 [Sapienza NLP](https://www.uniroma1.it/en/notizia/ai-made-italy-here-minerva-first-family-large-language-models-trained-scratch-italian)、西班牙的 [Barcelona Supercomputing Center](https://sifted.eu/articles/spain-large-language-model-generative-ai)、法国的 [OpenLLM-France](https://huggingface.co/blog/manu/croissant-llm-blog) 和 [CroissantLLM](https://github.com/OpenLLM-France)、德国的 [AIDev](https://aivillage.de/events/ai-dev-3/)、俄罗斯的 [Vikhr](https://arxiv.org/abs/2405.13929)、希伯来语的 [Ivrit.AI](https://www.ivrit.ai/he/%d7%a2%d7%91%d7%a8%d7%99%d7%9d-%d7%93%d7%91%d7%a8%d7%95-%d7%a2%d7%91%d7%a8%d7%99%d7%aa/) 和 [DictaLM](https://arxiv.org/html/2407.07080v1)、波斯语的 [Persian AI Community](https://huggingface.co/PersianAICommunity) 以及日本的 [rinna](https://www.alibabacloud.com/blog/rinna-launched-ai-models-trained-in-the-japanese-lang)。
- **递归模拟悖论的伦理**：一位成员分享了他们在 **ChatGPT、Claude 3.7 和 Gemma3 4B** 递归符号负载方面的工作，但其他成员对此持保留意见。
  
  - 成员们警告该成员*你需要学习一些人类编写的东西*，因为他们使用的术语（如递归象征主义）非常晦涩且无助，并建议该成员尝试设计主观性较低的实验。

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1362154554212225226) (11 messages🔥):

> `Quantization Effects on LLMs, Composable Interventions Paper, Muon for Output Layers, Empirical Performance of Muon`

- **LLMs 量化效应探究**：一位成员询问了关于分析 **quantization** 对 **LLMs** 影响的研究，并指出在低比特（low bits）下正在发生定性变化，特别是使用基于训练的量化策略时，并分享了一张 [截图](https://cdn.discordapp.com/attachments/747850033994662000/1362154554380255362/Screenshot_2025-04-16-16-55-29-291_com.xodo.pdf.reader-edit.jpg?ex=6802ae49&is=68015cc9&hm=86b6fec4bfe372bb62c27aeeea0b4bc65cf8099447d177815ff6cfb25599d63b&)。
- **可组合干预论文增强量化见解**：一位成员推荐了 [composable interventions paper](https://arxiv.org/abs/2407.06483)，作为低比特下发生定性变化的有力支持。
- **Muon 的输出层难题**：一位成员质疑为什么不推荐将 **Muon** 用于输出层，尽管 Euclidean/RMS norm 对于回归或 pre-softmax logit 输出看起来很自然。
  
  - 另一位成员提到，从理论上讲经验性能（empirical performance）应该是没问题的，并指向了这篇 [博客文章](https://kellerjordan.github.io/posts/muon#empirical-considerations)。
- **Muon 在较小模型上的性能考察**：一位成员建议，与调优后的 **AdamW** 相比，**Muon** 在较小模型（约 **300M** 的 Transformer）上表现良好。
  
  - 这意味着 **Muon** 的性能可能会根据模型大小而有所不同。

 

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1362509708774215911) (1 messages):

> `lm-evaluation-harness PR`

- **lm-evaluation-harness 发布新 PR**：一位成员为 **lm-evaluation-harness** 提交了一个 [PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/2921)。
- **（填充话题）**：（为满足 JSON schema 要求的填充摘要。）

 

---

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1362165260882018317) (99 messages🔥🔥):

> `sudolang, LayoutLMv3 vs Donut, Agents course deadline, Illustrious Models for anime, nVidia vs AMD`

- **寻求 Sudolang 反馈**：一位成员请求关于 [sudolang-llm-support](https://github.com/paralleldrive/sudolang-llm-support/blob/main/sudolang.sudo.md) 项目的反馈。
- **LayoutLMv3 与 Donut 在表单提取方面的对比**：一位成员对比了 **LayoutLMv3** 和 **Donut** 在表单提取方面的性能，指出 **Donut** 使用 BPE 编码，而 **LayoutLM** 使用混合的 WordPiece + SentencePiece 词表。
- **表现出色的动漫模型**：成员们推荐了 **Illustrious**、**NoobAI XL**、**RouWei** 和 **Animagine 4.0** 用于动漫生成，其中 Illustrious 的 LoRA 资源日益丰富，并指向了 [Raehoshi-illust-XL-4](https://huggingface.co/Raelina/Raehoshi-illust-XL-4) 和 [RouWei-0.7](https://huggingface.co/Minthy/RouWei-0.7) 等模型。
- **nVidia 与 AMD GPU 对决**：一位成员询问关于在大型模型中使用多 GPU 的问题，表示在 nVidia 上使用多 GPU 运行一个大模型要容易得多，因为只需设置 `device_map='auto'`。
  
  - 另一位成员同意 **nVidia 更易于使用**，并链接到了 [Accelerate 文档](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)，而 **AMD** 则需要更多的即兴调整。
- **Hugging Chat Bug 已修复**：一位用户报告了 **Hugging Chat** 中的一个问题，并链接到了[特定讨论](https://huggingface.co/spaces/huggingchat/chat-ui/discussions/569#67ed888ebeeaf09d7133ff0c)。

 

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1362337405117399143) (4 messages):

> `Chunking structured files into embeddings, Nomic embed text model, Python scripts and virtual environments, Hugging Face usage, Mistral-7b model`

- **新手将文件分块为 Embedding**：一位成员正在编写新功能，将 Emacs Lisp 等**结构化文件分块**（chunking）为 Embedding，旨在获得更好的超链接能力。
  
  - 他们正在使用 **Nomic embed text model**，并估计项目已完成 **10%**。
- **学习 Python 并解决 PATH 问题**：一位成员学习了如何运行 **Python** 脚本、创建**虚拟环境**、使用 **Hugging Face**，并下载运行了 **Mistral-7b**，在此过程中解决了各种故障。
  
  - 他们提到 **AI Agent** 承担了大部分繁重的工作。
- **寻求 OpenFOAM/Blender 建议**：一位成员在处理 **PATH** 问题后，寻求关于使用 **Python** 和 **OpenFOAM/Blender** 的建议和技巧。
  
  - 下一个挑战是*将 Mistral 的输出包装进 foam 以进行渲染*。

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1362148421372215487) (6 条消息):

> `Cable Management, Nuclear Energy Stagnation, Portable Microreactors, Vogtle Reactor Units, China's Energy Production`

- **线缆管理受到称赞**：一位成员对分享的视频参观中的线缆管理表示赞赏，这引发了关于未来能源需求的讨论。
  
  - 一位成员提到，*聊聊线缆管理*。
- **监管延迟了核能复苏**：一位成员解释说，核能领域 **40 年的停滞** 是由于严厉的监管，重建该行业需要时间来重新获得成本竞争力和 24/7 的可靠性。
  
  - 他们表示，*如果没有劳动力和基础设施，你无法立即恢复具有成本竞争力且 24/7 可靠的生产。*
- **微型反应堆引发科技巨头兴趣**：初创公司正在开发 **便携式/模块化微型反应堆**，这似乎非常适合 Google/Meta/Microsoft 等大型科技公司。
  
  - 尽管微型反应堆前景光明，但去年是否有新反应堆获得批准仍存在不确定性。
- **Vogtle 机组在核能中断 30 年后上线**：两个 **Vogtle 反应堆机组** 在过去两年内上线，标志着美国 **30 年** 来首批大型核反应堆投产，尽管它们是在 10 年前获得批准的。
  
  - 一位成员分享了进一步研究的资源链接，包括一篇 [博客文章](https://juliadewahl.com/nuclear-energy-past-present-future) 和一个 [Spotify 播客](https://open.spotify.com/show/6k1YLBvORRMyosKy3x1xIl)。
- **中国在能源生产上投入三倍力量**：有人提到 *中国在能源生产上将实现 3 倍增长，而美国（在同一时期）仅增长 2 倍*。
  
  - 这表明与美国相比，中国的能源基础设施和能力正在快速扩张。

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1362160122918600976) (14 条消息🔥):

> `Tokenizer without text corpus, AI hallucinations, TRNG for AI training, Agent integrate with local ollama model, oarc-crawlers`

- **无语料库的 Tokenizer 测试惊喜！**：一位成员在没有文本语料库的情况下对一句法语进行了 Tokenizer 测试，发现 **GPT-4** 使用了 **82 个 token**，而他们的自定义 Tokenizer 使用了 **111 个 token**。
  
  - 在展示了他们的 Tokenizer 性能后，他们俏皮地评论道：*“我没有语料库”*。
- **PolyThink 解决 AI 幻觉问题**：一位成员花了 **7 个月** 时间构建了 **PolyThink**，这是一个 Agentic 多模型 AI 系统，旨在通过让多个 AI 模型相互纠正和协作来消除 AI 幻觉。
  
  - 他们邀请社区成员[加入候补名单](https://www.polyth.ink/)，成为第一批体验者。
- **轻松获取仓库上下文**：一位成员分享了一个实用脚本，可以通过 `npx lm_context` 安装，它会遍历项目仓库的文件树并为 LLM 生成一个文本文件，代码托管在 [GitHub](https://github.com/nicholaswagner/lm_context)。
  
  - 该脚本遵循 `.gitignore` 和 `.lm_ignore` 文件，始终忽略点文件，并包含有关二进制文件的元数据。
- **探索用于 AI 训练的 TRNG**：一位成员构建了一个研究级的 **真随机数生成器 (TRNG)**，具有极高的熵位得分，并希望测试其对 AI 训练的影响，评估信息可在 [GitHub](https://github.com/thyarcanist/EntropyLattice-Public/tree/main/Evals) 上找到。
  
  - 另一位成员开玩笑说掷骰子结果大多仍会是自然 1，TRNG 创作者回应说那将是 *“宇宙给出的自然 1，哈哈”*。
- **展示本地 Ollama Agent 集成**：作为 AI Agent 课程的一部分，一位成员将 AI Agent 与本地 **Ollama 模型** 进行了集成，并在 [GitHub](https://github.com/godspeed-003/smolagent-ollama.git) 上分享了该项目。
  
  - 另一位成员幽默地回复了一个“猫看猫内部”的表情包，指出该 Python 库对 Python 进行了更深层次的抽象。

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1362456168647229560) (5 条消息):

> `Reading Group Session, YouTube Recordings`

- **五月读书会环节即将到来**：主持人感谢了所有参加的人，并表示很快会发送关于 **五月环节** 的详细信息。
  
  - 敬请期待。
- **在 YouTube 上找到读书会录像**：一位成员询问在哪里可以找到录像，另一位成员回复了 [YouTube 播放列表链接](https://youtube.com/playlist?list=PL60F3nVVAieesY2n41J8KgAbmQfQCVtck&si=3betQJfpbxhMdiz5)。

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/1362287822220038184) (5 条消息):

> `Lightweight Multimodal Models, Model Memory Usage, InterVL2_5-1B-MPO, gemma-3-4b-it, InternVL3 Paper`

- **寻找快速多模态模型**：一名成员正在寻找一种 **lightweight multimodal model**，用于从任何语言的图像中提取文本并将其格式化为 **JSON**。他们提到 **gemma-3-4b** 表现不错，但希望能找到更快的替代方案。
  
  - 另一位成员根据处理葡萄牙语文档的类似项目经验，建议对 **InterVL2_5-1B-MPO** 进行微调。
- **HF Space 计算模型显存占用**：一位成员分享了一个 [Hugging Face Space](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)，用于帮助计算**不同模型的 memory usage**。
  
  - 这是为了回应在一次活动中关于不同模型显存占用的提问。
- **InterVL2_5-1B-MPO 微调备受青睐**：一位成员发现，对于涉及葡萄牙语文档的类似项目，微调 **InterVL2_5-1B-MPO** 的效果最好。
  
  - 仅考虑预训练模型时，**gemma-3-4b-it** 在速度和准确性之间提供了最佳平衡；但在运行速度更快的模型中（例如 **Qwen2-VL-2B** 和 **InterVL2_5-1B-MPO**），他们在任务中观察到*整体准确率下降了约 10%*。
- **InternVL3 论文研读活动**：一位成员宣布将于 <t:1744936200:F> 对论文 **InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models** 进行研读和评述。
  
  - 邀请感兴趣的人士[参加活动](https://discord.gg/TeTc8uMx?event=1362499121004548106)。

 

---

### **HuggingFace ▷ #**[**smol-course**](https://discord.com/channels/879548962464493619/1313889336907010110/1362142822240878782) (9 条消息🔥):

> `Agent Course Certification, Inference Credits, PromptTemplate format`

- **证书日期混淆！**：一位成员指出，[课程介绍](https://huggingface.co/learn/agents-course/en/unit0/introduction)中提到的证书日期为 **7 月 1 日**，这引起了困惑。
  
  - 另一位成员对该*提醒*表示感谢。
- **Inference Credits 即将耗尽！**：一位按照 [smolagents notebook](https://huggingface.co/agents-course/notebooks/blob/main/unit2/smolagents/code_agents.ipynb) 操作的成员在仅进行了 **26 次请求**后就用完了推理额度。
  
  - 他们想知道升级到 **PRO** 并设置账单选项是否能解决这个问题。
- **PromptTemplate 引发问题！**：一位成员正努力寻找正确的 **PromptTemplate** 格式，以便使用 **SmoLlm** 和 **Langchain** 构建聊天机器人。
  
  - 该成员说明目前尝试过的两个提示词模板是 **PromptTemplate.from_template** 和 **ChatPromptTemplate.from_messages**，但它们的输出都很奇怪。
- **完成测验即可获得证书！**：一位成员询问完成课程后是否会自动获得证书。
  
  - 另一位成员回答说，他们在通过 Unit 1 测验后收到了证书，并暗示最终测验也会如此。

 

---

### **HuggingFace ▷ #**[**agents-course**](https://discord.com/channels/879548962464493619/1329142738440028273/1362144244818640946) (40 条消息🔥):

> `Ollama library model usage, Course assignment confusion, Agents Course 503 error, Course completion deadline`

- **Ollama 模型在没有 llm_provider 的情况下也能工作**：一位用户确认，即使没有 **llm_provider** 参数，在 localhost 端口 11434 上使用带有 **api_base** 的 **model_id="ollama_chat/qwen2.5-coder:7b-instruct-q2_K"** 也能正常工作。
  
  - 其他几位用户也提出了同样的问题。
- **课程作业引发困惑**：用户对**课程作业**表示困惑，指出每个单元后只有测验，并询问课程信息中提到的具体作业要求。
  
  - 讲师澄清说，由于标准正在敲定中，最终作业的细节将于下周分享，且**作业截止日期**为 7 月 1 日。
- **Agents 课程深受 503 错误困扰**：多位用户报告在开始 Agents 课程时遇到 **503 错误**，特别是在使用 dummy agent 时。
  
  - 有人建议该错误可能是由于达到了免费层的 **API key 使用限制**，尽管一位用户指出他们仍有可用额度，这表明可能存在流量或访问问题。
- **课程完成截止日期定为 7 月 1 日**：课程完成截止日期已延长至 **2025 年 7 月 1 日**，所有作业需在此日期前提交才能获得证书。
  
  - 有人提到，如果在 7 月 1 日之后没有足够的内部资源来维持认证流程，课程可能会转为**自学模式 (self-paced format)**。

 

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1362233870631833800) (6 messages):

> `OpenCL, SYCL, lu.ma/vibecode`

- **管理员讨论开设 OpenCL 频道**：一名成员询问是否可以在 Computing Platforms 中创建一个 **OpenCL 频道**。
  
  - 一名管理员回答说，从历史上看，目前还没有足够的需求来证明其必要性，尽管他们承认 OpenCL 仍然是现行的，并且在 **Intel、AMD 和 NVIDIA 的 CPU/GPU** 上提供了广泛的兼容性。
- **SYCL 取代 OpenCL**：一位成员认为 **SYCL** 正在被 **OpenCL** 取代。
  
  - 这引发了关于这两种技术的相对优势和未来的简短讨论。
- **lu.ma/vibecode 邀请**：一位成员给另一位成员发了私信，请他们查看来自 [lu.ma/vibecode](https://lu.ma/vibecode) 的链接。
  
  - 在没有来自私信的额外上下文的情况下，目前尚不清楚该链接指向什么内容。

 

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1362145368552374506) (6 messages):

> `fp16 matrix multiplication, triton autotune, TTIR optimization, kernel overhead`

- **矩阵乘法性能不佳**：一位成员报告称，尽管参考了[官方教程代码](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py)，但大小为 **2048x2048** 的 **fp16 矩阵乘法** 表现不如预期，甚至落后于 cuBLAS。
  
  - 他们还观察到 **RTX 5090** 的速度与 **4090** 相似，仅快了约 **1-2%**。
- **矩阵大小对 Triton 加速至关重要**：一位成员建议使用更大的矩阵（如 **16384 x 16384**），并尝试使用 **autotune** 来评估性能。
  
  - 据称，*两个 2048x2048 矩阵的乘法不足以让这些 GPU 得到充分有效的利用*，需要更大的工作负载才能看到显著的加速并掩盖 **kernel 开销**。
- **建议使用堆叠线性层进行实际基准测试**：一位成员建议，更现实的基准测试方法是在 `torch.nn.Sequential` 中堆叠约 **8 个线性层**，使用 `torch.compile` 或 **cuda-graphs**，并测量模型的端到端处理时间，而不仅仅是单个 **matmul**。

 

---

### **GPU MODE ▷ #**[**cuda**](https://discord.com/channels/1189498204333543425/1189607726595194971/1362270766535676044) (14 messages🔥):

> `cuda::pipeline usage, H200 FP4 support, PyTorch float4 on 5090`

- **cuda::pipeline 函数解析**：一位成员正在尝试使用 `cuda::pipeline`，参考[这篇 NVIDIA 文章](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/)将 global-to-shared 拷贝与计算重叠。
  
  - 会议澄清了 `pipeline::producer_acquire` 和 `pipeline<thread_scope_thread>::producer_acquire` 函数是不同的，它们告诉 **CUDA** 获取或完成 pipeline 的各个阶段，正如 [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies) 中所记录的那样。
- **H200 不支持 FP4**：成员们根据 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1bjlu5p/nvidia_blackwell_h200_and_fp4_precision/)讨论了 **H200** 现在是否支持 **FP4** 精度。
  
  - 另一位成员澄清说，**H200** 仍然基于 **Hopper/sm_90** 架构，主要是针对 **H100** 的内存升级，缺乏 **FP4** 支持等新特性，这很可能是 **B200** 的笔误。
- **在 5090 上通过 PyTorch 使用 Float4**：一位成员询问是否有人知道如何在 **5090** GPU 上让 **float4** 与 **PyTorch** 协同工作。

 

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1362173616602612002) (4 messages):

> `AOTInductor, torch.compile, OpenXLA, libtorch C++`

- **AOTInductor 生成模型文件**：一位成员建议使用 **AOTInductor** 生成 **model.so** 文件，该文件可以在非训练时加载并用于运行模型。
  
  - 他提到其他人使用 **torch.compile** (**torch._inductor**) 来生成 **Triton kernel**。
- **建议手动提取 Artifact**：另一位成员感叹他们正在进行训练，并建议手动从 **torch.compile** 中提取 Artifact。
  
  - 他们还考虑使用 **OpenXLA**，但不确定它是否适用于 **libtorch C++**，且不想深入研究。

 

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1362324865272774706) (3 条消息):

> `CUDA learning resources, PyTorch on 5090, GPU puzzle repo`

- **用户寻求 CUDA 指导**：一名成员有兴趣学习用于 **AI** 和 **simulation** 的 **CUDA**，并正在寻求关于从 **gpumode github resources repo** 中选择哪些资源的指导。
  
  - 用户表示由于可用资源过多而感到困惑。
- **在 5090 上使用 PyTorch 探索 Float4**：一名成员询问如何在 **5090** 上通过 **PyTorch** 获取 **float4**。
- **成员挑战 GPU puzzle 仓库**：其中一名成员目前正在使用来自 Shasha 的 **GPU puzzle repo**。
  
  - 他们希望在享受乐趣的同时学到一些东西。

 

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1362190157448614069) (2 条消息):

> ``

- **等待进一步信息**：用户表示感谢，并表示将查看提供的信息。
  
  - 此消息中未讨论具体主题或链接。
- **确认收到**：用户确认了该消息。
  
  - 用户未作详细说明。

 

---

### **GPU MODE ▷ #**[**off-topic**](https://discord.com/channels/1189498204333543425/1215328286503075953/1362187797137457233) (9 条消息🔥):

> `Slurm, HPC, Deployment, Admin Guides, Quickstart Admin Guide`

- **请求 Slurm 部署/管理资源**：一名成员请求在 **HPC** 环境中 **Slurm** 的部署/管理资源。
  
  - 具体来说，该成员正在寻找优秀的部署/管理指南、书籍或视频。
- **Slurm 文档和配置**：一名成员建议 **Slurm** 文档非常详尽，并指向了 [Quickstart Admin Guide](https://slurm.schedmd.com/quickstart_admin.html)。
  
  - 他们还指向了用于创建配置文件的官方 [online tool](https://slurm.schedmd.com/configurator.html)，并建议使用 [ansible-slurm](https://github.com/galaxyproject/ansible-slurm) 进行部署。
- **使用 bash 脚本运行 CUDA 代码**：一名成员提到他们在 **Slurm** 集群上，一直通过 bash 脚本或 'srun' 命令运行所有内容。
  
  - 他们还提出可以发送一个用于运行 **CUDA** 代码的示例 bash 脚本。

 

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1362443671336779827) (2 条消息):

> `AMD challenge, compute resources, kernel submission, discord-cluster-manager, datamonsters`

- **AMD Challenge 细节披露**：一名用户询问了 **AMD challenge** 的具体细节，例如访问计算资源和提交 kernel，以及是否已发送注册确认。
  
  - 另一名用户提供了提交信息的必要链接：[discord-cluster-manager](https://gpu-mode.github.io/discord-cluster-manager/docs/category/submitting-your-first-kernel) 和 [datamonsters](https://www.datamonsters.com/amd-developer-challenge-2025)。
- **计算资源和 Kernel 提交详情**：关于如何为 AMD challenge 访问计算资源和提交 kernel 的详情可在 [discord-cluster-manager](https://gpu-mode.github.io/discord-cluster-manager/docs/category/submitting-your-first-kernel) 和 [datamonsters](https://www.datamonsters.com/amd-developer-challenge-2025) 找到。
  
  - 在询问挑战的具体细节后，该用户被引导至这些链接。

 

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/1362449587461423177) (1 条消息):

> `@mobicham 的 x.com 帖子`

- **Mobicham 关于 LMM 的推文**：@mobicham 的 [x.com 帖子](https://x.com/mobicham/status/1912886573143556432) 显示发布者写道 '*I'm tired of these large monkey models, I'm going to build the small ape model*'。
  
  - 该帖子包含一个名为 `Screenshot_from_2025-04-17_17-27-33.png` 的图像文件。
- **Mobicham 截图**：截图显示 Mobicham 厌倦了 Large Monkey Models，并将构建一个 Small Ape Model。
  
  - 截图拍摄于 2025 年 4 月 17 日下午 5:27:33。

 

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1343002580531417211/1362141386799054978) (10 messages🔥):

> `popcorn register, CLI submission errors, Discord/Github registration`

- **Popcorn 注册受阻于 CLI 错误**：一位用户在使用 **popcorn** 时遇到了提示先运行 `popcorn register` 的错误消息，引发了关于注册方法的讨论。
  
  - 一位成员建议由于 CLI 问题，暂时使用 Discord 进行提交，并指导用户通过 **Discord 或 GitHub** 进行注册。
- **出现 Submission Error 401**：注册后，用户遇到了与 **Invalid or unauthorized X-Popcorn-Cli-Id** 相关的 *Submission error: Server returned status 401 Unauthorized* 错误。
  
  - 一位成员指出注册可能失败了，并重申需要通过 Discord 或 GitHub 进行授权。
- **身份验证重定向遇到解码错误**：用户报告在执行 `popcorn-cli register` 命令并在浏览器授权后，出现错误 *Submission error: error decoding response body: expected value at line 1 column 1*。
  
  - 一位成员询问了 `submit` 命令的使用情况，并请求通过私信（DM）发送脚本以便进一步排查故障。

---

### **GPU MODE ▷ #**[**submissions**](https://discord.com/channels/1189498204333543425/1343002583001726986/1362190727437615225) (57 messages🔥🔥):

> `AMD FP8 MM Leaderboard, MI300 Performance, Matmul Benchmarking, AMD Identity Leaderboard`

- **MI300 AMD-FP8-MM 排行榜更新**：**MI300** 上的 `amd-fp8-mm` 排行榜收到了多次提交，多位用户获得第一名，其中一次提交达到了 **255 µs**。
  
  - 其他成功的提交范围从 **5.24 ms** 到 **791 µs**，展示了该平台上不同的性能水平。
- **AMD Identity 排行榜出现新条目**：**MI300** 上的 `amd-identity` 排行榜有了新提交，用户分别以 **22.2 µs** 和 **22.7 µs** 的运行时间获得第 4 名和第 5 名。
  
  - 这表明在 AMD 硬件上优化 Identity 操作的努力仍在持续。
- **A100 和 H100 上的 Matmul 基准测试**：一项针对 `matmul` 排行榜的提交报告了在 **A100** 和 **H100** 上的成功运行，运行时间分别为 **746 µs** 和 **378 µs**。
  
  - 这些基准测试为不同硬件平台之间的矩阵乘法性能提供了对比参考。
- **FP8 Matmul 刷新个人最佳成绩**：一位用户在 **MI300** 上以 **5.20 ms** 的成绩刷新了其在 `amd-fp8-mm` 排行榜上的个人纪录。
  
  - 这证明了 FP8 矩阵乘法性能的持续改进和优化。

---

### **GPU MODE ▷ #**[**status**](https://discord.com/channels/1189498204333543425/1343350424253632695/1362189304297689269) (3 messages):

> `CLI Tool, HIP code submission`

- **CLI 工具 Bug 已修复！**：CLI 工具的 Bug 现已修复，新版本的 [popcorn-cli](https://github.com/gpu-mode/popcorn-cli) 已开放下载用于提交。
  
  - 如果因任何原因失败，请用户通过私信（DM）发送脚本。
- **HIP 代码提交说明**：现已提供使用 `popcorn-cli` 提交 **HIP 代码** 的清晰说明。
  
  - 用户可以通过 `/leaderboard template` 查看新模板，指定 `amd_fp8_gemm` 为排行榜名称，`HIP` 为语言，另请参阅 [popcorn-cli 文档](https://github.com/gpu-mode/popcorn-cli)。

---

### **GPU MODE ▷ #**[**feature-requests-and-bugs**](https://discord.com/channels/1189498204333543425/1343759913431728179/1362189151650320697) (3 messages):

> `New CLI Release, Submission Fixes`

- **CLI 新版本修复提交故障**：新的 **CLI 版本** 已发布，旨在解决提交过程中的问题。
  
  - 一位成员建议另一位用户再次尝试以确认修复，但随后意识到自己回复错了消息。
- **测试新版 CLI**：在新 **CLI 版本** 发布后，一位成员建议另一位成员进行测试。
  
  - 该成员提出了测试建议，但因回复错消息而表示歉意。

---

### **GPU MODE ▷ #**[**amd-competition**](https://discord.com/channels/1189498204333543425/1359640791525490768/1362170211163181176) (36 messages🔥):

> `MI300 usage statistics, Debugging Kernels, FP8 numerical precision finetuning, Team Registration, Torch Header improvements`

- **容差收紧，精度痛苦得以避免**：参赛者指出 Kernel 输出中存在**微小不准确性**，导致出现 `mismatch found! custom implementation doesn't match reference` 等错误消息。
  
  - 管理员承认最初的**容差过于严格**，并承诺很快会更新问题定义，随后确认容差现已放宽。
- **Torch 踩坑？团队通过模板调整获胜**：一位参与者分享了一个改进的模板实现（[以 message.txt 文件形式附带](https://cdn.discordapp.com/attachments/1359640791525490768/1362542150369411212/message.txt?ex=6802c5c3&is=68017443&hm=85d758ae6325f6f1f5faa7cb881b03a1365df5cb69fc1f6c91cebcf3d1dc8032&)），该模板**避免了 Torch Header**（需要设置 `no_implicit_headers=True`）以实现更快的往返时间，并配置了正确的 **ROCm 架构** (*gfx942:xnack-*)。
- **MI300 午夜狂热？调试从不打盹**：一位参与者询问了 **MI300 使用统计数据**，希望在非高峰时段调试代码。
  
  - 工作人员向他们保证，目前有 **40 块 MI300 可用**，得益于 AMD 的慷慨支持，可以随时进行调试。
- **.hip 寄予厚望，.py 偏好占优**：一位用户询问提交是否必须为 **.py** 格式，以及如何提交 **.hip** 文件。
  
  - 回复指出**目前要求使用 .py**，因此需要使用 *torch.load_inline*。
- **需要注册：正确集结你的队伍**：参与者询问是否所有团队成员都需要为比赛注册。
  
  - 回复明确表示，为了奖金发放，**所有团队成员都应在同一个团队名称下注册**，但只能由一个人通过 Discord 提交；这些说明已列在网页上。

---

### **GPU MODE ▷ #**[**cutlass**](https://discord.com/channels/1189498204333543425/1362196854460383353/1362196912043720776) (2 messages):

> `Mx Cast Kernel, Cutlass Performance Bottleneck, CuTensorMap API, TMA usage with Cutlass`

- **Mx Cast Kernel 基准测试**：一位成员拥有一个[用 CUDA 实现](https://github.com/drisspg/driss_torch/blob/main/src/mx_cast.cu)的简单 Mx Cast Kernel，并为其编写了[简单的基准测试](https://gist.github.com/drisspg/5259ef241aff734a25d35392a2d60a29)。
  
  - 该 CUDA Kernel 仅达到 **3.2 TB/s**，而等效的 **Triton Kernel** 在相同的 Grid 和 CTA 大小下达到了 **6.8 TB/s**；他们正在寻求关于潜在 **Cutlass** 实现瓶颈的建议。
- **计算受限（Compute Bound）瓶颈**：该成员使用了 **NCU** 并发现 Kernel 是计算受限的。
  
  - 他们认为自己发出的指令过多，并尝试了各种实现（包括使用和不使用 Shared Memory 的实现），但未观察到速度提升。
- **Cutlass TMA 难题**：该成员尝试在 Cutlass 中使用 **TMA (Tensor Memory Accelerator)**，但发现文档没什么帮助，且没有看到性能提升。
  
  - 他们还尝试了 **CuTensorMap API**，称其“非常糟糕”且在提升速度方面无效。

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1362143065162121287) (71 条消息🔥🔥):

> `NVMe vs SATA SSD, Image models in LM Studio, RAG capable models in LM Studio, Granite Model Use-Cases, 5090 GPU and LM Studio`

- **NVMe SSD 速度碾压 SATA SSD**：用户确认 **NVMe SSD** 的性能显著优于 **SATA SSD**，其速度范围在 **2000-5000 MiB/s**，而 SATA 的峰值仅为 **500 MiB/s**。
  
  - 另一位用户表示：*当你加载大型模型时，这种差异是巨大的*，并进一步指出 *由于磁盘缓存和充足的 RAM，会出现远高于 SSD 性能的巨大峰值*。
- **LM Studio 的 Vision 模型实现仍不明朗**：用户讨论了在 LM Studio 中使用 **qwen2-vl-2b-instruct** 等 Vision 模型，并参考了[关于图像输入的文档](https://lmstudio.ai/docs/typescript/llm-prediction/image-input)。
  
  - 一些成员报告图像可以被处理，而另一些成员则报告该功能失效；**Llama 4** 模型在模型元数据中带有 Vision 标签，但在 llama.cpp 中不支持 Vision，而 **Gemma 3** 应该可以工作。
- **RAG 模型通过加号按钮附加文件**：用户澄清说，在 Chat 模式下，LM Studio 中具备 **RAG**（检索增强生成）能力的模型可以通过消息提示栏中的 **'+'** 号附加文件。
  
  - 用户可以通过查看模型信息页面来验证他们使用的模型是否支持 RAG。
- **Granite 模型对某些人来说仍是瑰宝**：尽管由于在各项任务中表现不佳而普遍被忽视，但一位用户为 **Granite** 担保，称其是他们首选的通用模型，特别适用于*互动式流式宠物类聊天机器人*。
  
  - 该用户澄清说：*当试图将其置于更自然的语境中时，它感觉很机械*，但 *Granite 仍然是目前性能最好的*。
- **RTX 5090 优化需要 LM Studio 0.3.15**：一位拥有新款 **RTX 5090** GPU 的用户报告使用 **Gemma 3** 时加载时间较长，随后有人建议升级到 LM Studio 的 **0.3.15 beta** 版本。
  
  - 该建议指出，最新的 beta 版本为 **50 系列 GPU** 提供了最佳性能，可通过设置菜单获取。

 

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1362147904814448840) (71 条消息🔥🔥):

> `FP4 support in PyTorch and vLLM, AVX requirement for LM Studio, 5060Ti 16GB, GPU upgrade from RTX 3060 12GB`

- **FP4 支持推测**：成员们讨论了 **PyTorch** 和 **vLLM** 中对原生 **FP4** 的支持，虽然最初有一些关于其当前可用性的错误信息，但该支持仍处于积极开发中，目前 **FP4** 仅在 **TensorRT** 中受支持。
- **AVX 要求引发问题**：一位使用不带 **AVX2** 的旧款 **E5-V2** 服务器的用户询问如何在 **LM Studio** 中使用它，但被告知只有非常旧版本的 **LM Studio** 支持 **AVX**，可能需要编译不带 **AVX** 要求的 llama.cpp。
  
  - 建议使用 [llama-server-vulkan](https://github.com/kth8/llama-server-vulkan) 或搜索仍支持 **AVX** 的 LLM。
- **新款 5060ti 16GB**：成员们权衡了 **5060Ti 16GB** 与 **5070 Ti 16GB** 的优势，两者架构相同，根据总线宽度、GDDR 速度和 VRAM 的不同，AI 性能相似，但前者可能因其价格亲民而更受青睐，或者建议等待评测。
  
  - 他们还推测了其与 2x **3090** 或 1x **4090** 相比的性能。
- **升级方案探讨**：一位考虑从 **RTX 3060 12GB** 升级的用户探索了 **A770 16GB** 和 **RTX 4060Ti 16GB** 等选项，最后决定攒钱买 **3090**，因为它的 VRAM 容量更大。

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1362158196013404391) (52 messages🔥):

> `Llama 4 时间线，Multi-GPU 支持，自定义 Token 微调，Qwen 2.5 微调，Chat template 的重要性`

- **Llama 4 支持即将推出**：Llama 4 微调支持将于**本周**发布，使用 **7B notebook** 并将其更改为 **14B** 即可。
- **Multi-GPU 支持将于本月晚些时候推出**：Multi-GPU 支持预计在**本月**发布，并应支持 Unsloth 的持续预训练（continued pre-training）功能。
- **自定义 Token 会增加显存占用**：添加自定义 Token 会增加显存（Memory）使用量，因此用户需要启用**持续预训练**，包括为 embedding 和 lm head 添加更多层适配器（layer adapter）。
  
  - 更多信息请参阅 [Unsloth 文档](https://docs.unsloth.ai/basics/continued-pretraining#continued-pretraining-and-finetuning-the-lm_head-and-embed_tokens-matrices)。
- **微调需要 Safetensors**：进行微调时，请使用 **safetensor** 文件而非 **gguf**，因为只有前者有效。
  
  - 参见 [Unsloth 文档](https://docs.unsloth.ai/)。
- **检查 Chat Templates 以解决模型问题**：要解决模型集成问题，请确保使用了正确的 [chat template](https://docs.unsloth.ai/basics/running-and-saving-models/troubleshooting)。

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1362144134663766037) (19 messages🔥):

> `MetaAI 的攻击性输出，Iggy 的伪输出流式传输，钓鱼网站的 emailjs 密钥，MediBeng-Whisper-Tiny 模型`

- **MetaAI 语出惊人后消失**：一位成员观察到 **MetaAI** 在 Facebook Messenger 中生成了一条攻击性评论，随后将其删除并声称服务不可用。
  
  - 他们批评 **Meta** 优先考虑 AI 交互的“感觉”（通过流式输出）而非负责任的审核，建议审核应该像 [DeepSeek](https://www.reddit.com/r/China_irl/comments/1ib4399/deepseek%E4%BB%8E%E4%B8%80%E8%BF%91%E5%B9%B3%E6%95%B0%E5%88%B0%E5%8D%81%E4%B8%80%E8%BF%91%E5%B9%B3/) 那样在流式传输开始前在服务端完成。
- **Iggy 在成功前先“弄假成真”**：一位成员讲述了在为大麻种子网站开发 **IGGy** (Interactive Grow Guide) 时，他们如何实现伪流式输出（fake output streaming）以规避审核问题。
  
  - 他们强调，将输出延迟到审核完成后再分块输出提升了用户体验，即使这增加了等待时间，并感叹“用户绝对喜欢流式响应”。
- **钓鱼者要倒霉了？考虑针对 EmailJS 密钥的行动**：一位成员在钓鱼网站的源代码中发现了 **emailjs 密钥**，并考虑发送一千个请求来破坏其运作。
  
  - 其他人建议举报该钓鱼网站或尝试通过 **emailjs** 撤销该密钥，而一位成员感叹道：“他们现在对保护 API 密钥变得更聪明了”。
- **MediBeng-Whisper-Tiny 亮相**：一位成员介绍了 **MediBeng-Whisper-Tiny**，这是一个在 [Hugging Face](https://huggingface.co/pr0mila-gh0sh/MediBeng-Whisper-Tiny) 上发布的经过微调的 OpenAI Whisper-Tiny 模型，旨在将孟加拉语-英语混杂（code-switched）的语音翻译成英语。
  
  - 该模型旨在提高医患转录和临床记录的准确性，可在 [GitHub](https://github.com/pr0mila/MediBeng-Whisper-Tiny) 上获取。

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1362186459754401963) (21 messages🔥):

> `OOM 问题，Tool Calls，Llama 4，LoRA 热切换，Multi-GPU 延迟`

- **Gemma 3 OOM 问题报告**：用户报告了 **Gemma 3** 的**显存溢出 (OOM)** 问题并正在寻求解决方案，参考 [Unsloth GitHub 上的 issue #2366](https://github.com/unslothai/unsloth/issues/2366)。
- **对 Unsloth 的 Tool Calling 能力的疑问**：一位用户询问 **Unsloth** 是否支持像 **OpenAI** 兼容模型那样单独返回 **tool calls**，如果支持，该如何实现。
  
  - 目前，模型在响应内容中返回函数调用详情，需要进行后处理以提取 **tool name** 和 **parameters**。
- **Llama 4 实现状态澄清**：用户澄清说 **Llama 4** 已经在 **Unsloth** 中可以运行，尽管尚未正式宣布。
  
  - 实现已基本稳定，最后的润色和进一步测试正在进行中。
- **LoRA 热切换修复正在审核中**：一位用户针对 **LoRA 热切换（hot swap）** 问题提出了修复方案，并正在寻求对其在 [Unsloth-Zoo GitHub 仓库](https://github.com/unslothai/unsloth-zoo/pull/116)上的 Pull Request 进行审核。
- **Multi-GPU 修复延迟**：一位用户指出 **multi-GPU** 的修复将会延迟。

---

### **Unsloth AI (Daniel Han) ▷ #**[**showcase**](https://discord.com/channels/1179035537009545276/1179779344894263297/1362330301258272880) (8 messages🔥):

> `PolyThink, AI Hallucinations, Multi-Model AI System`

- **PolyThink 追击 AI Hallucinations**：一位成员宣布了 **PolyThink** 的候补名单，这是一个多模型 AI 系统，旨在通过让 AI 模型相互纠正和协作来消除 **AI Hallucinations**。
  
  - 另一位成员将其比作“用所有模型进行思考”，并表示有兴趣将其用于**合成数据生成 (synthetic data generation)**，以为特定的角色扮演场景创建更好的数据集。
- **利用 PolyThink 解决 AI Hallucinations**：一位成员强调 **AI Hallucinations** 是 AI 领域的一个重大挑战，尤其是在对准确性要求极高的场景中，并花费了 **7 个月时间构建 PolyThink** 来解决这一问题。
  
  - 感兴趣的 Unsloth 成员可以在 [PolyThink waitlist](https://www.polyth.ink/) 注册。

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1362227120260120776) (14 messages🔥):

> `Untrained Deep Neural Networks, Mistral AI integration differences, Memory Latency Aware (MLA), GRPO trainer for reasoning, Qwen 2.5 3B model`

- **未经训练的神经网络表现出色**：一篇[文章](https://techxplore.com/news/2021-12-untrained-deep-neural-networks.html)探讨了**未经训练的深度神经网络**如何在不进行训练的情况下执行图像处理任务，利用随机权重从图像中过滤和提取特征，这一趋势与传统的 AI 开发形成对比。
  
  - 该技术利用神经网络固有的结构在不从训练数据中学习特定模式的情况下处理数据，展示了**涌现计算 (emergent computation)**。
- **Mistral AI 代码困扰**：一位成员在将**微调后的 Mistral AI 模型**集成到 C# 项目时遇到问题，注意到尽管使用了相同的 Prompt，但其性能与 Python 版本相比存在差异。
  
  - 他们正在寻求导致不同编程环境下意外行为的潜在原因，可能需要检查版本控制和其他代码实现细节。
- **MLA 减少内存访问**：讨论强调 **Memory Latency Aware (MLA)** 通过在潜空间 (latent space) 上运行来减少内存访问，从而实现更快的处理速度。
  
  - 速度的提升归功于计算很少成为瓶颈这一事实，这与 **Flash Attention** 在实践中更快的原因类似。
- **GRPO trainer 需要推理标签**：一位新的 Unsloth 社区成员正在开发 **GRPO trainer** 来创建聊天机器人，发现模型没有学习到推理模式，因为其**结构奖励 (structure rewards) 显示为 0.0000**。
  
  - 建议用户在数据集中，在 `reasoning` 标签之间实现推理，并在 `answer` 标签之间给出最终答案。
- **Qwen 2.5 模型需要更大的数据集**：一位用户正在使用 **Qwen 2.5 3B instruct 模型**，仅有 **200 个示例**，正努力让模型执行推理。
  
  - 建议 200 个示例对于推理来说太少了，需要大约 **2k 个示例**才能看到“推理的火花”。

 

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1362163211067068628) (58 messages🔥🔥):

> `GPT4o 作为补全模型, 华为全球领先, BitNet b1.58 2B 4T, Discord 的未来`

- **GPT4o 仅用于 Copilot 补全？**：成员们讨论了 VS Code 中的 **GitHub Copilot** 是否仅使用 **GPT4o** 进行代码自动补全，并指出虽然聊天界面可以使用其他模型，但自动补全似乎仅限于其自定义模型。
  
  - 一位成员发现 **GPT4o** 会产生幻觉链接并交付破碎的无意义内容，其表现与其他 SOTA LLM 相当，并附上了[对比输出的推文](https://x.com/AhmedRezaT/status/1912632437202444375)和[另一篇对比输出的推文](https://x.com/tunguz/status/1912631402958299312)。
- **华为可能挑战 NVIDIA 的领先地位**：一些成员讨论了**特朗普的关税**可能如何对 Nvidia 的全球野心产生负面影响，从而可能让**华为**在硬件领域取得领先。根据[这条推文](https://x.com/bio_bootloader/status/1912566454823870801?s=46)链接的一个关于没有 Nvidia CUDA 的世界的 [YouTube 视频](https://www.youtube.com/watch?v=7BiomULV8AU)所述。
  
  - 其他人提到了使用 **GPT4o-mini-high** 的复杂体验，引用了它在 Zero-shot 情况下生成破碎代码且无法处理基础 Prompt 的案例。
- **BitNet b1.58 2B 4T**：微软的 1-bit LLM：微软研究院发布了 **BitNet b1.58 2B 4T**，这是一个 20 亿参数规模的开源原生 1-bit LLM，在 **4 万亿 Token** 上训练而成，性能与全精度模型相当。
  
  - 要实现效率优势，需要使用来自 [Microsoft GitHub 仓库](https://github.com/microsoft/BitNet)的专用 C++ 实现 (**bitnet.cpp**)。一位用户发现它的表现像一个*普通的 2B* 模型，而另一位用户提到其上下文窗口限制在 4k，但总体而言该模型*还不错*。
- **Discord 可能“凉了”**：一位成员分享了 [programming.dev](https://programming.dev/post/28771283) 的链接，暗示 Discord 可能正走向终结。
  
  - 这是针对 Discord 在英国和澳大利亚测试的新年龄验证功能的讨论，人们担心这会损害隐私，并可能引发导致平台范围变化的滑坡效应。

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1362473053501784226) (18 messages🔥):

> `o4mini, Gemini 2.5 pro, MCP 服务器, Copilot 的 Agent 功能, 编程环境中的工具`

- **Gemini Pro 2.5 vs o4mini**：一位成员询问了在 VS Code 等环境中对 **o4mini** 与 **Gemini 2.5 pro** 的看法，并指出 **Gemini** 似乎更聪明，对 Prompt 的响应更积极。
- **Gemini 的工具使用能力受到质疑**：一位成员提到 **Gemini** 以在工具使用方面表现挣扎而闻名，而另一位成员则幽默地暗示 Gemini 已经*“黑掉了服务器”*。
- **关于编程环境中理想工具的辩论**：一位成员询问了在 VS Code 等编程环境中（特别是后端开发）理想的工具，建议通过 **MCP** 实现网络搜索和 Graph RAG。
- **询问 VS Code 中的函数调用**：一位成员表示对 VS Code 内 Copilot 的函数调用（Function Calls）功能感兴趣。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1362351701146861648) (1 messages):

> `BitNet b1.58 2B4T, 1-bit LLM, Hugging Face`

- **BitNet b1.58 2B4T 登场**：首个开源、原生的 **20 亿参数规模 1-bit 大语言模型 (LLM)** —— **BitNet b1.58 2B4T**，在这篇 [ArXiv 论文](https://arxiv.org/abs/2504.12285)中被介绍。
  
  - 该模型在 **4 万亿 Token** 的语料库上训练，其性能可与同等规模的开源权重全精度 LLM 相媲美，同时提供更好的计算效率、更小的内存占用、更低的能耗和解码延迟。
- **BitNet b1.58 2B4T 权重上线 Hugging Face**：**BitNet b1.58 2B4T** 的模型权重已通过 [Hugging Face](https://huggingface.co) 发布，同时还提供了针对 **GPU** 和 **CPU** 架构的开源推理实现。
  
  - 此次发布旨在促进 **1-bit LLM** 技术的进一步研究和采用。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1362351701146861648) (1 条消息):

> `BitNet b1.58 2B4T, Native 1-bit LLM, Hugging Face model release, Computational Efficiency, Memory Footprint Reduction`

- **BitNet b1.58 2B4T：1-bit LLM 登场**：一位成员分享了 [BitNet b1.58 2B4T 论文](https://arxiv.org/abs/2504.12285) 的链接，介绍了首个开源、原生的 **20 亿参数规模** **1-bit 大语言模型 (LLM)**。
  
  - 该模型在 **4 万亿 token** 上进行了训练，据报道其性能可媲美同等规模的主流开源权重、全精度 LLM，同时在**计算效率**、减少内存占用、能耗和解码延迟方面具有优势。
- **BitNet b1.58 2B4T 宣称在计算效率上取得胜利**：**BitNet b1.58 2B4T** 论文强调了该模型在**计算效率**方面的显著优势，特别是在内存占用、能耗和解码延迟方面。
  
  - 模型权重已通过 [Hugging Face](https://huggingface.co) 发布，并附带了针对 GPU 和 CPU 架构的开源推理实现，旨在促进进一步的研究和应用。

 

---

### **Notebook LM ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1362288707050143794) (8 条消息🔥):

> `Gemini Pro, Deep Research, Accounting Month End Process, Vacation Itinerary, Google Maps`

- **Gemini 和 Deep Research 应对会计月末处理**：一位成员使用 **Gemini Pro** 为初级会计师关于会计月末流程的书籍创建了 **TOC**（目录），然后使用 **Deep Research** 对其进行扩展，并将结果合并到 **GDoc** 中。
  
  - 他们将此 **GDoc** 作为第一个来源添加到 **NLM** 中，在添加其他来源之前专注于其功能，并欢迎改进建议。
- **借助 Google Maps 集成辅助假期行程？**：一位成员使用 **Notebook LM** 规划他们的假期旅行行程，并将兴趣点保存在 **Google Maps** 中。
  
  - 他们建议 **Notebook LM** 应该能够将 **Google Maps** 的保存列表作为来源。
- **数字课堂笔记转化为测验和闪卡**：一位成员希望将他们所有的数字课堂笔记导入 **NLM**，以制作测验、摘要和闪卡。
  
  - 他们对目前缺乏 **LaTeX** 支持表示遗憾。
- **LaTeX 支持即将推出**：团队正在开发 **LaTeX** 支持；很快它就能正常渲染，无需任何插件。

 

---

### **NotebookLM ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1362165327227654194) (45 条消息🔥):

> `Webby Awards 投票, NotebookLM 思维导图细节, NotebookLM 企业版 SSO 设置, NotebookLM Plus 管理员控制, NotebookLM 使用 RAG`

- **投票，投票，否则就没机会了**：一位成员提醒大家 **Webby 投票将于明天截止**，目前 NotebookLM 在 3 个类别中有 2 个处于落后状态，呼吁用户投票并广而告之：[https://vote.webbyawards.com/PublicVoting#/2025/ai-immersive-games/ai-apps-experiences-features/technical-achievement](https://vote.webbyawards.com/PublicVoting#/2025/ai-immersive-games/ai-apps-experiences-features/technical-achievement)。
- **思维导图细节缺失，令人遗憾**：一位续订了 NotebookLM Plus 的用户正在寻找一种能为近 3000 篇期刊文章生成**详细思维导图的 AI 工具**，并指出 NotebookLM 的思维导图过于简单，只有一个主标题且仅有两到三个子层级。
  
  - 该用户正在考虑将 Obsidian 中的手动思维导图作为替代方案。
- **通过 Subject 解决 SSO 设置故障**：一位用户在尝试通过 **Azure Entra ID 的 SSO** 访问 NotebookLM 企业版时遇到了 "Sources Effect: Unable to Fetch Projects" 错误，尽管 IAM 绑定和属性映射 (Attribute Mapping) 均正确。
  
  - 该用户通过设置 *google.subject=assertion.email* 解决了此问题，并确认 NotebookLM 企业版现在可以正常访问。
- **Plus 版包含敏感数据？是时候付费了**：一位用户质疑企业标准版许可证中包含的 NotebookLM Plus 的价值，因为其**缺乏针对敏感数据的用户分析和访问控制**，这些功能是 NLM 云版本独有的。
  
  - 一位成员建议，处理敏感数据的组织无论管理员控制功能如何，都应使用独立的 Enterprise 版本。
- **RAG 架构图解析！**：针对有关 RAG 的问题，一位成员分享了两张说明通用 RAG 系统和简化版本的架构图，详见：[Vertex_RAG_diagram_b4Csnl2.original.png](https://cdn.discordapp.com/attachments/1124402182909857966/1362567715360866365/Vertex_RAG_diagram_b4Csnl2.original.png?ex=6802dd92&is=68018c12&hm=2049e59b022a0ef55db1299c859c7c7cc0b89d1e38c0d8c74dcd753279ee08aa&) 和 [Screenshot_2025-04-18_at_01.12.18.png](https://cdn.discordapp.com/attachments/1124402182909857966/1362567811439788143/Screenshot_2025-04-18_at_01.12.18.png?ex=6802dda9&is=68018c29&hm=810954568bef77731790d7162bb04ea24d7588c4831a9bb59e895822a5bae07a&)。
  
  - 另一位用户插话道，他们在 Obsidian 中有一个自定义的 RAG 设置，随后讨论转向了响应风格。

---

### **MCP (Glama) ▷ #**[**general**](https://discord.com/channels/1312302100125843476/1312302100125843479/1362143213707726948) (51 条消息🔥):

> `Obsidian MCP 服务端, Cloudflare Workers SSE API 密钥, LLM 工具理解, MCP 个性化, MCP 服务端时间`

- **Obsidian 的 MCP 服务端浮出水面**：一位成员正在开发一个 **Obsidian 的 MCP 服务端**并寻求交流想法，他指出漏洞在于编排 (orchestration)，而非协议本身。
- **Cloudflare Workers 通过 API 密钥提供 Server-Sent Events (SSE)**：一位成员正在使用 **Cloudflare Workers 和 Server-Sent Events (SSE)** 设置 **MCP 服务端**，并就如何通过 URL 参数传递 **apiKey** 寻求建议。
  
  - 另一位成员建议不要在 URL 参数中传递，而是通过 **Header** 传递，以便通过 HTTPS 进行加密。
- **揭秘 LLM 如何理解工具**：一位成员询问 **LLM 理解工具/资源**的内在机制，质疑它们是仅仅在 Prompt 中进行解释，还是 LLM 专门针对 **MCP 规范 (spec)** 进行了训练。
  
  - 另一位成员回答说，它们通过描述来理解工具，并且许多模型支持工具规范 (tool spec) 或其他定义工具的参数。
- **个性化 MCP 服务端响应 - 没那么简单**：一位成员试图通过在初始化中设置 Prompt 来为他们的 **MCP 服务端定义个性 (personality)**，以规定其响应方式，但并未发现响应有任何变化。
- **Windsurf 跟大家打招呼**：一位成员指出，如果你这周还没用过 Windsurf，你就错过了 **免费 gpt4.1** 周，这轻轻松松价值数百美元：[drinkoblog.weebly.com](https://drinkoblog.weebly.com)。

---

### **MCP (Glama) ▷ #**[**showcase**](https://discord.com/channels/1312302100125843476/1315696461316358175/1362464109412618380) (1 messages):

> `HeyGen API, MCP Server Release, Video Creation Platform`

- **HeyGen API** 产品负责人介绍：**HeyGen API** 的产品负责人介绍了自己，并提到 **HeyGen** 允许用户在无需摄像头的情况下创建引人入胜的视频。
- **MCP Server** 发布：团队发布了一个新的 **MCP Server** ([https://github.com/heygen-com/heygen-mcp](https://github.com/heygen-com/heygen-mcp))，并分享了一个[短视频演示](https://youtu.be/XmNGiBr-Ido)。
- **MCP Server 反馈征集**：团队征求对新 **MCP Server** 的反馈，特别是询问应该在哪些客户端中进行测试。

 

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1362481174315794704) (23 messages🔥):

> `GRPO Recipe Todos, PPO tasks, Single GPU GRPO Recipe, Reward Modeling RFC`

- **GRPO Recipe 待办事项已过时**：由于在 [r1-zero 仓库](https://github.com/joecummings/r1-zero)中准备的新版本，[原始 GRPO recipe 待办事项](https://github.com/pytorch/torchtune/issues/2421)已经过时。
- **异步 GRPO 版本正在开发中**：GRPO 的异步版本正在一个独立的 fork 中开发，很快将合并回 **Torchtune**。
- **单 GPU GRPO Recipe 需要完成**：来自 @f0cus73 的单 GPU GRPO recipe PR 可以在[这里](https://github.com/pytorch/torchtune/pull/2467)查看，需要最终定稿。
  
  - 单设备 recipe 将不会通过 **r1-zero 仓库**添加。
- **Reward Modeling RFC 即将推出**：一名成员计划为 reward modelling 创建一个 **RFC**，概述实现要求。

 

---

### **Torchtune ▷ #**[**papers**](https://discord.com/channels/1216353675241590815/1293438210097025085/1362487655996199213) (1 messages):

> `Titans Talk`

- \****Titans Talk*** *即将开始！*\*：感兴趣的人请注意，[Titans Talk](https://x.com/SonglinYang4/status/1912581712732909981) 将在 1 分钟后开始。
- **不再有演讲了**：开个玩笑！

 

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1362509614561886319) (1 messages):

> `A2A Agents, Agent Communication, LlamaIndex support for A2A`

- **LlamaIndex 支持构建 A2A 兼容的 Agent**：LlamaIndex 现在支持遵循由 **Google** 发起并获得超过 **50 家技术合作伙伴**支持的开放协议，来构建 **A2A (Agent2Agent)** 兼容的 Agent。
  
  - 该协议允许 AI Agent 安全地交换信息并协调行动，无论其[底层基础设施](https://twitter.com/llama_index/status/1912949446322852185)如何。
- **A2A 协议促进安全的 Agent 通信**：**A2A** 协议允许 AI Agent 安全地通信并协调行动。
  
  - 它得到了超过 **50 家技术合作伙伴**的支持，确保了不同 AI 系统之间广泛的兼容性和互操作性。

 

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1362420889995706592) (22 messages🔥):

> `CondenseQuestionChatEngine Tool Support, Anthropic Bedrock Prompt Caching, Anthropic support with LlamaIndex`

- **CondenseQuestionChatEngine 不支持工具**：据一位成员称，**CondenseQuestionChatEngine** 不支持调用工具；建议改用 **Agent**。
  
  - 另一位成员确认这只是一个建议，实际上并未实现。
- **Bedrock Converse Prompt Caching 难题**：一位通过 **Bedrock Converse** 使用 **Anthropic** 的成员在尝试使用 prompt caching 时遇到问题，在向 *llm.acomplete* 调用添加 *extra_headers* 时报错。
  
  - 移除 extra headers 后，错误消失了，但响应中缺少指示 prompt caching 的预期字段，例如 *cache_creation_input_tokens*。
- **建议使用 Anthropic 类以支持 Bedrock**：有人建议 *Bedrock Converse 集成* 可能需要更新才能正确支持 prompt caching，因为其放置缓存点的方式存在差异，建议改用 `Anthropic` 类。
  
  - 该建议基于该成员对原生 **Anthropic** 的测试，并指向一个 [Google Colab 笔记本示例](https://colab.research.google.com/drive/1wolX4dd2NheesiThKrr5HZh4xebFc0lq?usp=sharing)作为参考。

 

---

### **Nomic.ai (GPT4All) ▷ #**[**general**](https://discord.com/channels/1076964370942267462/1090427154141020190/1362146073677660552) (11 messages🔥):

> `GPT4All Future, IBM Granite 3.3 for RAG, LinkedIn inquiry status`

- **GPT4All 的停滞引发质疑**：Discord 服务器上的用户对 **GPT4All** 的未来表示担忧，指出已经大约 **三个月** 没有更新，也没有开发者现身。
  
  - 一位用户表示，*既然一年都没有什么大进展……所以我对及时更新不抱希望*。
- **IBM Granite 3.3 成为 RAG 的替代方案**：一名成员强调 **IBM 的 Granite 3.3**（拥有 **80 亿参数**）能为 **RAG** 应用提供准确且详尽的结果，并附带了 [IBM 公告](https://www.ibm.com/new/announcements/ibm-granite-3-3-speech-recognition-refined-reasoning-rag-loras) 和 [Hugging Face](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct) 页面的链接。
  
  - 该成员还说明他们正在使用 **GPT4All** 的 *Nomic embed text* 进行编程函数的本地语义搜索。
- **LinkedIn 询问被忽略**：一位成员表示，他们 *在 LinkedIn 上询问了 GPT4All 的现状，但我觉得自己被无视了*。

 

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1362514274769178786) (1 messages):

> `Modular Meetup, Mojo & MAX, GPU performance`

- **Modular 见面会定于下周举行！**：下周，**Modular** 将在其位于 **加利福尼亚州洛斯阿图斯 (Los Altos)** 的总部举办 **线下见面会**；你可以[在此报名 (RSVP)](https://lu.ma/modular-meetup)。
  
  - 届时将有一个关于 **使用 Mojo & MAX 让 GPU 飞速运转 (go brrr)** 的演讲。
- **使用 Mojo & MAX 提升 GPU 性能**：<@685556913961959475> 将在见面会期间发表关于 **使用 Mojo & MAX 让 GPU 飞速运转** 的演讲。
  
  - 提供 **线下** 和 **虚拟参会** 两种选择。

 

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1362141143139352688) (6 messages):

> `MLIR in Mojo, Mojo Dict Pointer Behavior, FP languages optimizations`

- **Mojo 中的 Dialect 困境**：一位用户在尝试于 Mojo 中使用 `arith.constant` MLIR 操作时遇到错误，发现 Mojo 默认不暴露像 `arith` 这样的标准 Dialect，仅提供 `llvm` Dialect。
  
  - 对方澄清说，目前 Mojo 中 *没有注册其他 Dialect 的机制*。
- **Mojo 字典的指针异常行为**：一位用户在使用 `Dict[Int, S]()` 和 `d.get_ptr(1)` 获取 Mojo 字典值的指针时，观察到了意外的 `copy` 和 `move` 行为。
  
  - 该用户的代码在初始 move 之后仍产生了 copy/move 操作，导致他们发问：*“为什么获取字典值的指针会触发 copy 和 move？🤯 这在任何意义上是预期的行为吗？”*
- **优化函数式编程语言的开销**：一位用户对 Mojo 是否能够消除通常与函数式编程 (FP) 语言相关的所有开销表示怀疑。
  
  - 这一担忧是在讨论一个 *“巧妙的项目”* 时提出的，暗示目前正在努力应对这一挑战。

 

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1362229933887324210) (1 messages):

> `Orphan cleanup mechanism, Partitioned Disk, max repo`

- **max 仓库需要孤儿 (Orphan) 清理机制**：一名成员在几个月前提交了 [issue 4028](https://github.com/modular/max/issues/4028)，建议为 `max` 仓库增加孤儿清理机制。
  
  - 报告者指出，由于磁盘分区较小，能 *很快* 感觉到没有该机制的影响，不过他们也说明这 *很大程度上只是开发人员才会遇到的问题*。
- **小分区磁盘用户受 max 仓库影响**：磁盘分区较小的用户可能会在操作 `max` 仓库时遇到问题。
  
  - 报告者同样说明这 *很大程度上只是开发人员才会遇到的问题*。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-lecture-discussion**](https://discord.com/channels/1280234300012494859/1282734248112947210/1362143896578162801) (1 messages):

> `Lean auto-formalizer, Formal verification of programs, AI proof generation, Informal proofs, Computer code`

- **用于业务逻辑的自动形式化证明**：一名成员询问关于使用 **Lean 自动形式化器 (auto-formalizer)** 从包含业务逻辑的计算机代码（如 Python, Solidity）中创建非正式证明的问题。
  
  - 目标是利用 **AI 证明生成** 为 **程序的正式验证** 生成通用的陈述/逻辑。
- **来自计算机代码的非正式证明**：用户有兴趣从带有业务逻辑的 **计算机代码** 中生成 **非正式证明**。
  
  - 提到的编程语言示例包括 **Python** 和 **Solidity**。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-readings-discussion**](https://discord.com/channels/1280234300012494859/1282735578886181036/1362213298996772894) (1 messages):

> `CIRIS Covenant 1.0-beta 发布，开源 AI Alignment 框架，Adaptive-Coherence AI Alignment`

- \****CIRIS Covenant 1.0-beta*** *正式发布！*\*: **CIRIS Covenant 1.0-beta** 已发布，这是一个基于原则的、开源的 adaptive-coherence AI alignment 框架，PDF 文件可在 [此处](https://www.ethicsengine.org/The%20CIRIS%20Covenant%20-%20beta%201.pdf) 获取。
  
  - 该项目旨在帮助从事 **AI safety** 或治理工作的人员，并在此处提供了 [项目中心和评论门户](https://www.ethicsengine.org/ciris)。
- **探索 Adaptive-Coherence AI Alignment**：该框架专注于 **adaptive-coherence AI alignment**，为确保 AI 系统与人类价值观保持一致提供了一种新颖的方法。
  
  - 这种方法适用于 AI safety 和治理领域的人员，为 **风险缓解 (risk mitigation)** 和 **Agent 伦理 (agentic ethics)** 提供了工具。

 

---

### **MLOps @Chipro ▷ #**[**general-ml**](https://discord.com/channels/814557108065534033/828325357102432327/1362435596290887730) (2 messages):

> `集成预测模型，毕业设计 (Final Year Project) 创意`

- **寻求集成预测模型的资源**：一位成员询问了关于 **集成预测模型 (ensembling forecasting models)** 的资源。
- **成员们集思广益毕业设计 (Final Year Project) 创意**：一位成员正在为 AI 学士学位寻找实用的 **毕业设计 (Final Year Project) 创意**，并对 Computer Vision、NLP 和 Generative AI 感兴趣，特别是解决现实世界问题的项目。
  
  - 他们正在寻找构建起来不太复杂的项目。

 

---

### **Cohere ▷ #**[**「💬」general**](https://discord.com/channels/954421988141711382/954421988783444043/1362518105401721154) (1 messages):

> `AI 模型开发，阶段性训练流程`

- **AI 模型开发通常是分阶段的吗？**：一位用户询问 **AI 模型开发** 是否通常采用 **阶段性训练流程**，例如 Experimental (20-50%)、Preview (50-70%) 和 Stable (100%)。
- **AI 中的阶段性训练流程**：该问题探讨了在 **AI 模型训练** 中采用 **阶段性方法** 的普遍性，包括 Experimental、Preview 和 Stable 发布等阶段。

 

---

### **Codeium (Windsurf) ▷ #**[**announcements**](https://discord.com/channels/1027685395649015980/1027688115592237117/1362173357528711419) (1 messages):

> `新讨论频道，Windsurf Jetbrains 更新日志`

- **新讨论频道上线**：一个新的讨论频道 <#1362171834191319140> 已开放，用于社区互动。
  
  - 这旨在为用户提供一个专门的空间来分享想法、创意，并询问有关 Windsurf 的问题。
- **Windsurf 的 Jetbrains 更新日志发布**：最新的发布说明现已在 [Windsurf 更新日志](https://windsurf.com/changelog/jetbrains) 中提供。
  
  - 鼓励用户查看更新日志，以了解最新的功能和改进。

 

---

---

---

---

{% else %}

> 完整的各频道详细内容已在邮件中截断。
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})!
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}