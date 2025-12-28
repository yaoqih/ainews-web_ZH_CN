---
companies:
- google-deepmind
- hugging-face
- anthropic
date: '2025-02-06T02:00:20.087119Z'
description: '以下是为您翻译的内容：


  **Google DeepMind** 正式发布了 **Gemini 2.0** 系列模型，包括 **Flash**、**Flash-Lite** 和 **Pro
  Experimental**。其中，**Gemini 2.0 Flash** 的性能超越了 **Gemini 1.5 Pro**，且价格便宜了 **12 倍**，同时支持**多模态输入**和
  **100 万 token 的上下文窗口**。**Andrej Karpathy** 发布了一段时长 **3 小时 31 分钟**的视频，深入解析了**大语言模型**，涵盖了**预训练**、**微调**和**强化学习**，并以
  **GPT-2** 和 **Llama 3.1** 为例进行讲解。**Jay Alammar**、**Maarten Gr** 和**吴恩达 (Andrew Ng)**
  推出了一门关于 **Transformer 架构**的免费课程，重点讲解**分词器 (tokenizers)**、**嵌入 (embeddings)** 和**混合专家模型
  (MoE)**。**DeepSeek-R1** 在 **Hugging Face** 上的下载量已达到 **120 万次**，并附有一份详尽的 **36 页技术报告**。**Anthropic**
  将其“越狱挑战”的奖励提高至 **1 万美元**和 **2 万美元**；同时，**BlueRaven** 浏览器扩展程序进行了更新，可隐藏 Twitter 指标以实现无偏见的社交互动。'
id: 043b46cf-2ff6-40c4-ac85-344be87f269d
models:
- gemini-2.0-flash
- gemini-2.0-flash-lite
- gemini-2.0-pro-experimental
- gemini-1.5-pro
- deepseek-r1
- gpt-2
- llama-3-1
original_slug: ainews-gemini-20-flash-ga-with-new-flash-lite-20
people:
- andrej-karpathy
- jayalammar
- maartengr
- andrewyng
- nearcyan
title: Gemini 2.0 Flash 正式发布（GA），同步推出全新的 Flash Lite、2.0 Pro 以及 Flash Thinking。
topics:
- multimodality
- context-windows
- cost-efficiency
- pretraining
- fine-tuning
- reinforcement-learning
- transformer
- tokenization
- embeddings
- mixture-of-experts
---

<!-- buttondown-editor-mode: plaintext -->**[REDACTED] is all you need.**

> 2025年2月4日至2月5日的 AI 新闻。我们为您检查了 7 个 Reddit 社区、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**210** 个频道，**5481** 条消息）。为您节省了预计 **571 分钟** 的阅读时间（以 200wpm 计算）。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

Gemini 2.0 自 12 月以来就已经“发布”了（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-google-wakes-up-gemini-20-et-al/)），但现在我们可以正式将 Gemini 2.0 Flash 的价格视为“真实”价格，并将其放入[我们的帕累托前沿图表 (Pareto frontier chart)](https://x.com/swyx/status/1887263009921507656/photo/1) 中：


![image.png](https://assets.buttondown.email/images/113c16be-8968-4d08-905e-eab4134a4c63.png?w=960&fit=max)


我们承认，像这样单纯的智能图表意义正变得越来越小，并且可能会在今年彻底失效，因为它们无法准确描述这些发布版本的**多模态输入和输出**能力，也无法体现 [coding 能力](https://x.com/OfficialLoganK/status/1887269355919917182)或 **100-200 万的超长上下文**，正如 [Sundar Pichai 所展示的](https://x.com/sundarpichai/status/1887169871697350775)：


![image.png](https://assets.buttondown.email/images/1433e7df-251b-4a3d-914b-a2ca9ab9601d.png?w=960&fit=max)


特别值得注意的是新推出的 "Flash Lite" 的性价比，以及 Gemini 2.0 Flash 相对于 1.5 Flash 极微小的价格上涨。

有趣的是，OpenAI “压制 (mogging)” Google 发布节奏的竞争态势似乎停留在 2024 年了。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

- **Google DeepMind 发布 Gemini 2.0 模型，包括 Flash、Flash-Lite 和 Pro Experimental**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1887172464863506547) 宣布 [Gemini 2.0 Flash](https://twitter.com/Google/status/1887170927751729385)、[Flash-Lite](https://twitter.com/Google/status/1887170932302659835) 和 [Pro Experimental](https://twitter.com/Google/status/1887170934211059830) 模型正式全面可用。[@_philschmid](https://twitter.com/_philschmid/status/1887171112850874569) 总结了这次更新，指出 **Gemini 2.0 Flash 的性能超越了 Gemini 1.5 Pro**，同时价格**便宜了 12 倍**。新模型提供了**多模态输入**、**100 万 token 上下文窗口**以及极高的**成本效益**。

- **Andrej Karpathy 发布“深入探讨 ChatGPT 等 LLM”视频**：[@karpathy](https://twitter.com/karpathy/status/1887211193099825254) 发布了一个 **3 小时 31 分钟的 YouTube 视频**，全面概述了 **Large Language Models (LLMs)**，涵盖了 **pretraining**（预训练）、**supervised fine-tuning**（监督微调）和 **reinforcement learning**（强化学习）等阶段。他讨论了 **data**、**tokenization**、**Transformer 内部机制**等主题，并以 **GPT-2 训练**和 **Llama 3.1 基础推理**为例进行了讲解。

- **“Transformer LLM 工作原理”免费课程**：[@JayAlammar](https://twitter.com/JayAlammar/status/1887189786672202233) 和 [@MaartenGr](https://twitter.com/MaartenGr/status/1887192134937190624) 与 [@AndrewYNg](https://twitter.com/AndrewYNg/status/1887184924165492940) 合作，推出了一门免费课程，深入探讨 **Transformer 架构**，包括 **tokenizers**、**embeddings** 和 **mixture-of-expert 模型**等主题。该课程旨在帮助学习者理解现代 LLM 的内部运作机制。

- **DeepSeek-R1 下载量突破 120 万次**：[@omarsar0](https://twitter.com/omarsar0/status/1887259405579649411) 强调，自 1 月 20 日发布以来，**DeepSeek-R1** 在 Hugging Face 上的下载量已达 **120 万次**。他还利用 Deep Research 对 DeepSeek-R1 进行了**技术深度分析**，生成了一份 **36 页的报告**。

- **Anthropic 提高越狱挑战奖励**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1887227067156386027) 宣布，目前**还没有人完全越狱其系统**，因此他们将首位通过所有八个关卡的人员奖励提高至 **$10K**，而通过所有八个关卡并实现通用越狱的奖励提高至 **$20K**。详细信息请参阅其[公告](https://t.co/As1zPIQGOx)。[@nearcyan](https://twitter.com/nearcyan/status/1887217858251530340) 幽默地推出了 **PopTarts: Claude 口味**，作为给渗透测试人员的创意奖励。

- **BlueRaven 扩展隐藏 Twitter 指标**：[@nearcyan](https://twitter.com/nearcyan/status/1887067301662609690) 发布了 **BlueRaven** 的更新，该扩展允许用户在**隐藏所有指标的情况下浏览 Twitter**。这挑战了用户在**不受流行度指标影响的情况下进行互动**。其[源代码](https://twitter.com/nearcyan/status/1887067398085476410)已公开，并支持 Firefox 和 Chromium。

- **Chain-of-Associated-Thoughts (CoAT) 框架发布**：[@omarsar0](https://twitter.com/omarsar0/status/1887187689247752370) 讨论了一个通过结合 **Monte Carlo Tree Search** 与动态知识整合来增强 LLM 推理能力的新框架。该方法旨在提高复杂推理任务中**全面且准确的回答**。更多细节见[论文](https://t.co/FLI7w1Hwld)。

- **关于主题大纲合成的 STROM 论文**：[@_philschmid](https://twitter.com/_philschmid/status/1887085743131984029) 重点介绍了一篇题为 **“Synthesis of Topic Outlines through Retrieval and Multi-perspective” (STROM)** 的论文，该论文提出了一种多问题、迭代式的研究方法。它类似于 **Gemini Deep Research** 和 **OpenAI Deep Research**。感兴趣的人可以查阅其[论文](https://t.co/SRnejL5MAy)和 [GitHub 仓库](https://t.co/zxHYzXYifL)。

- **关于 AI 影响与工具的讨论**：[@omarsar0](https://twitter.com/omarsar0/status/1887189887557730794) 分享了关于 AI 如何使个人能够同时在多个领域表现出色的见解，强调了学习以及 **ChatGPT** 和 **Claude** 等工具的重要性。[@abacaj](https://twitter.com/abacaj/status/1887206493700645300) 分享了一个 Gist，展示了如何在 **GRPO 训练**期间运行 **gsm8k** 评估，并扩展了 GRPOTrainer 以进行自定义评估。

- **Nearcyan 的幽默沉思与测试帖子**：[@nearcyan](https://twitter.com/nearcyan/status/1887214356234182890) 在 Twitter 上发布了测试帖子，观察他的推文是如何被**降权 (deboosted)** 的。他思考了在不知道互动指标的情况下使用 Twitter 的体验。此外，他还幽默地表达了在当今环境下作为一名 **iOS 开发者**的感受 ([推文](https://twitter.com/nearcyan/status/1887267444571709798))。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. DeepSeek VL2 Small 发布及 R1 的基准测试成功**

- **DeepSeek 刚刚发布了 DeepSeek VL2 Small 的官方 Demo - 它在 OCR、文本提取和对话用例方面非常强大 (Hugging Face Space)** ([评分: 615, 评论: 37](https://reddit.com/r/LocalLLaMA/comments/1ii82yg/deepseek_just_released_an_official_demo_for/))：**DeepSeek VL2 Small** 的 Demo（一个 **16B MoE** 模型）已在 [Hugging Face](https://huggingface.co/spaces/deepseek-ai/deepseek-vl2-small) 上发布，展示了其在 **OCR**、文本提取和对话应用中的能力。**Vaibhav Srivastav** 和 **Zizheng Pan** 在 X 上宣布了这一发布，强调了它在各种视觉语言任务中的实用性。
  - **发布时间线与性能**：**DeepSeek VL2 Small** 模型大约在两个月前上传到 **Hugging Face**，预计本月将发布一个推理模型。评论者指出该模型在其尺寸下表现良好，尽管有些人更倾向于使用 **florence-2-large-ft** 来处理特定的视觉任务。
  - **无障碍与集成**：讨论包括视觉语言模型在浏览网站方面的实用性，特别是在无障碍功能实现良好的情况下。建议用户在将其集成到系统之前，先在一些文档上尝试该模型。
  - **模型可用性与工具**：人们对 **DeepSeek V3 Lite** 和 **gguf** 格式很感兴趣，并建议使用 **llama.cpp** 中的 *convert_hf_to_gguf.py* 进行转换。**Environmental-Metal9** 提供的 Demo 链接在[这里](https://huggingface.co/collections/deepseek-ai/deepseek-vl2-675c22accc456d3beb4613ab)，尽管有人反映 Demo 目前无法运行。

- **[2B 模型击败 72B 模型](https://i.redd.it/nxx7b0kblbhe1.jpeg)** ([Score: 164, Comments: 57](https://reddit.com/r/LocalLLaMA/comments/1ii9lab/2b_model_beats_72b_model/)): **DeepSeek R1-V** 项目证明了一个 **2B 参数模型**在视觉语言任务中可以超越 **72B 参数模型**，实现了卓越的有效性和分布外 (out-of-distribution) 鲁棒性。该模型在特定的分布外评估中仅通过 **100 个训练步数**就达到了 99% 和 81% 的准确率，成本仅为 **$2.62**，并在 **8 台 A100 GPU** 上运行了 **30 分钟**。该项目已完全开源，可在 [此处](https://github.com/Deep-Agent/R1-V) 获取。
  - 一些评论者对 **DeepSeek R1-V** 模型的成就表示怀疑，认为结果可能具有误导性，或者过于针对某些特定的基准测试。**Admirable-Star7088** 和 **Everlier** 幽默地指出，较小的模型在特定任务中可以超越较大的模型，但强调较大的模型通常更具通用性。
  - **Real-Technician831** 和 **iam_wizard** 讨论了在特定任务中使用较小模型的实际意义，指出这种方法对于范围较窄的商业应用来说计算效率更高。他们认为这样的结果并不令人惊讶，因为针对特定任务微调较小模型是一种已知的策略。
  - 讨论中还提到了另一个模型 **phi-CTNL**，据 **gentlecucumber** 分享的 [arXiv](https://arxiv.org/abs/2309.08632) 链接显示，该模型在各种基准测试中也击败了更大的模型。这增加了关于特定基准测试性能与通用能力之间关系的讨论。


- **[DeepSeek R1 在泛化基准测试中与 o1 并列第一。](https://i.redd.it/7na44xs3gdhe1.png)** ([Score: 162, Comments: 23](https://reddit.com/r/LocalLLaMA/comments/1iiij1d/deepseek_r1_ties_o1_for_first_place_on_the/)): **DeepSeek R1** 和 **o1** 在 **Generalization Benchmark** 中并列第一，平均排名均为 **1.80**。该基准测试在 **810 个案例**中测试了 AI 模型，并指出 **Qwen QwQ** 在 **280 个案例**中失败。
  - **Generalization Benchmark** 测试 AI 模型从示例中推断特定主题的能力，其中 **o3-mini** 排名第四。有关该基准测试的更多详细信息可以在 [GitHub](https://github.com/lechmazur/generalization) 上找到。
  - **Phi 4** 排名靠前，超越了 **Mistral Large 2**、**Llama 3.3 70b** 和 **Qwen 2.5 72b**，并因其适合自托管 (self-hosting) 的合理尺寸而受到称赞。**Qwen QwQ** 得分更高，但在生成正确的输出格式方面存在问题。
  - **o3-mini-high** 被指出缺失，但其对 **Livebench** 结果的影响非常重要。此外，**Gemini 1.5 Pro** 和 **Gemini 1.5 Flash** 之间存在 **0.99 的相关性**，表明两者性能相似。


**主题 2. Google 关于武器和监控用途的 AI 政策转变**

- **[Google 解除使用其 AI 开发武器和进行监控的禁令](https://www.wired.com/story/google-responsible-ai-principles/)** ([Score: 497, Comments: 126](https://reddit.com/r/LocalLLaMA/comments/1ii3qvv/google_lifts_a_ban_on_using_its_ai_for_weapons/)): **Google** 更新了其 AI 政策，取消了此前禁止将其 AI 技术用于**武器和监控**的规定。这一政策变动标志着 Google 在 AI 伦理应用立场上的重大转变。
  - 用户对 **Google 的 AI 政策转变**表示极大担忧，将其等同于道德滑坡，并质疑将 AI 用于武器和监控的伦理影响。许多评论讽刺地引用了 Google 以前的座右铭“Don't be evil”，暗示其背叛了核心价值观。
  - 讨论强调了政策变动的**政治和国际影响**，提到了 Google 参与以色列军事行动的情况，并将其与中国等其他国家的监控行为进行了比较。对隐私侵蚀以及 AI 在全球冲突中被滥用的担忧十分普遍。
  - 几条评论批评了政策变动背后的**公司动机**，认为股东利益往往高于伦理考量。“做正确的事 (do the right thing)”这一说法被批评为含糊不清且可能带有私利，将公司利益置于社会利益之上。


**主题 3. Gemma 3 发布与社区反应**

- **[Gemma 3 即将来临！](https://i.redd.it/q2q4555s4ehe1.jpeg)** ([Score: 403, Comments: 42](https://reddit.com/r/LocalLLaMA/comments/1iilrym/gemma_3_on_the_way/)): **Omar Sanseviero** 在推文中预告了关于 "Gemma" 的更新，引发了 **r/LocalLLama** 社区的关注。随附的截图强调了 "Gemini" 的功能，包括 "2.0 Flash"、"2.0 Flash Thinking Experimental" 和 "Gemini Advanced"，这表明目前正在进行的是 Gemini 而非 Gemma 3 的活跃开发。
  - 评论者表达了对**更大上下文长度（Context Sizes）**的强烈渴望，提到 **64k** 和 **128k** 是未来模型（如 **Gemma 3**）的首选目标。一些用户认为目前的 **8k** 上下文长度不足。
  - 一些用户强调了对 **Gemma 2** 的成功和偏爱，特别称赞了 **9b simpo** 模型在媒体知识方面的能力，并对具备增强功能甚至 **AGI** 能力的 **Gemma 3** 表示期待。
  - 讨论还反映了 Reddit 的**社区参与**方面，将其比作 2010 年代初期研究人员和开发人员与用户直接互动的场景，说明了该平台在促进 AI 进展讨论中的作用。


## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. Nvidia 的 CUDA 战略：AI 进化的催化剂**

- **黄仁勋（Jensen Huang）是否承认过 Nvidia 只是在 AI 领域碰巧走运？** ([Score: 153, Comments: 92](https://reddit.com/r/OpenAI/comments/1ii3a2h/has_jensen_huang_ever_acknowledged_that_nvidia/)): **Nvidia** 最初的目标是增强图形渲染，但无意中开发出了对训练神经网络至关重要的技术。这种偶然的成功极大地促使**黄仁勋**成为了历史上最富有的人之一。
  - 评论者的共识是，**Nvidia 在 AI 领域的成功**并非源于运气，而是战略远见和长期投资，特别是从 **2006/2007** 年开始开发的 **CUDA**。这一战略举措使 Nvidia 能够建立一个强大的开发者生态系统，这对于他们在 AI 和科学计算领域的统治地位至关重要。
  - **CUDA 的早期采用**最初遭到了质疑，用户回忆起 10-20 年前人们对它的看法。尽管如此， Nvidia 坚持了下来，使他们能够在 **2011** 年 **AlexNet** 论文发表后利用 AI 热潮，该论文使用了 Nvidia GPU 并突显了其战略优势。
  - Nvidia 的投资从 AI 扩展到各种市场，包括**加密货币**、**生物技术**和**自主系统**。评论者指出 Nvidia 参与了**光线追踪（Ray Tracing）**、**机器人技术**甚至军事技术等多种应用，展示了他们致力于在各行业扩大其 GPU 使用的决心。


**主题 2. 字节跳动（ByteDance）和 Google 推进 AI 前沿**

- **[字节跳动新的多模态 AI 研究] (https://v.redd.it/4ns98irddbhe1)** ([Score: 251, Comments: 26](https://reddit.com/r/OpenAI/comments/1ii8t6w/new_bytedance_multimodal_ai_research/)): 帖子提到了**字节跳动（ByteDance）**在**多模态 AI（Multi-modal AI）**方面的新研究，尽管文中未提供具体细节。帖子包含一段视频，此处未作分析。
  - **视听匹配**：讨论强调了**多模态 AI** 将任何音频与视觉效果匹配的能力，例如将美国口音与**爱因斯坦（Einstein）**的形象结合，这种刻意的错配是为了展示该技术的潜力。
  - **来源和内容的真实性**：在 [omnihuman-lab.github.io](https://omnihuman-lab.github.io/) 提供了源代码链接，用户批评了 AI 生成内容与历史表现之间的不匹配，指出 AI 的刻画让爱因斯坦看起来像个“神经典型（Neurotypical）”人。
  - **音频来源**：演示中使用的音频导致了关于口音的困惑，经确认为源自一场 **TEDx 演讲**（[来源](https://singjupost.com/jaak-panksepp-the-science-of-emotions-at-tedxrainier-transcript/?singlepage=1)）。

- **[Google 声称实现了全球最强 AI，并免费提供给用户！](https://www.reddit.com/gallery/1iihuln)** ([Score: 203, Comments: 56](https://reddit.com/r/OpenAI/comments/1iihuln/google_claims_to_achieve_worlds_best_ai_giving_to/))：**Google** 声称开发出了**全球最强 AI**，并正向用户**免费**提供。帖子中未提供更多细节，包括该 AI 的具体功能或应用。
  - 关于 **Google** 的 **Gemini AI** 的讨论呈现出褒贬不一的观点，一些用户对其能力表示怀疑，特别是在害虫防治建议等领域；而另一些用户则认为它在编程任务和使用 **AI Studio** 方面表现出色。**Gemini 2.0** 与前代产品相比显示出一定的局限性，导致一些用户在特定功能上回退到旧版本。
  - 用户对**编程模型**的性能展开了辩论，一些模型如 **o1** 和 **o3-mini** 因能高效生成大量代码而受到称赞，而其他模型在超过 100 行代码时就显得力不从心。有评论强调了这些模型先进的推理能力，突出了它们对编程任务的影响。
  - **Sonnet** 模型在 **lmsys webdev arena** 性能讨论中脱颖而出，用户注意到尽管其体积较小，但表现优于其他模型。关于是否应将 Sonnet 等模型与专注于推理扩展（inference scaling）的模型进行比较存在争议，一些人将竞争优势归功于新的方法论。


**主题 3. 辩论 AI 开源：审视 DeepSeek 及更多**

- **[DeepSeek 就其“完全开源”的说法进行更正](https://i.redd.it/3l8ucogi7ahe1.png)** ([Score: 126, Comments: 36](https://reddit.com/r/OpenAI/comments/1ii5ls8/deepseek_corrects_itself_regarding_their_fully/))：**DeepSeek** 就其 **DeepSeek-R1** 发布了“术语更正”，澄清虽然其代码和模型是在 **MIT License** 下发布的，但它们并不像之前声称的那样是“完全开源”的。宣布这一更正的推文获得了显著关注，拥有 **186 个赞**、**39 次转发**和 **324 条回复**，浏览量达 **27,000 次**。
  - 许多评论者认为，虽然 **MIT License** 通常与开源相关联，但 **DeepSeek** 的说法具有误导性，因为他们没有发布源代码或训练数据。这种区别在判断某物是否真正开源时至关重要，因为获取源代码是一个基本要求。
  - **coder543** 强调 **Llama** 模型和类似项目也不是完全开源的，因为它们缺乏详细的数据集描述和训练代码。这突出了 AI 社区中一个更广泛的问题，即模型发布时带有权重，但没有足够的细节或资源来复制训练过程。
  - 讨论强调了在应用于 AI 模型和软件时，对“开源”一词的误解或误用。一些用户澄清，除非提供了实际的源代码，否则在 **MIT** 下授权并不等同于开源，这说明了许可与实际开放性之间的区别。


---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Thinking 提供的摘要之摘要

**主题 1. Gemini 2.0 模型系列：性能与集成**

- **Flash 2.0 快速上线 Windsurf 和 Perplexity**：[Gemini 2.0 Flash 已在 Windsurf 上线](https://x.com/windsurf_ai/status/1887235006374035966) 以及 [Perplexity AI](https://x.com/aravsrinivas/status/1887174442171969936?s=61)，因其在编程查询中的速度和效率而备受推崇，在 Windsurf 上仅消耗 **0.25 用户提示词额度**。虽然速度受到称赞，但用户注意到其工具调用（tool calling）能力有限，且与 Claude 和 DeepSeek 等模型相比，其可靠性仍处于审查之中。
- **Pro Experimental 基准测试挑战 Claude 3.5 Sonnet**：[Gemini 2.0 Pro Experimental](https://x.com/lmarena_ai/status/1887180371219132898) 在编程和复杂提示词方面的基准测试与 **Claude 3.5 Sonnet** 相当，并在 lmarena.ai 的排行榜上夺得 **第 1 名**。然而，用户观察到 API 响应存在不一致性，且尽管宣传有 **200 万 token 上下文**，但与 Gemini 1.5 Pro 相比，长上下文能力可能有所下降。
- **GitHub Copilot 为开发者引入 Gemini 2.0 Flash**：[GitHub 宣布为所有 Copilot 用户集成 Gemini 2.0 Flash](https://x.com/github/status/1887208954704355350)，使其可以在代码编辑器的模型选择器和 GitHub 上的 Copilot Chat 中使用。此举标志着 Gemini 在 Microsoft 生态系统中的重大胜利，使其在开发者工具集成方面领先于竞争对手。

**主题 2. 编程 IDE 和 AI 助手：功能对比与用户反馈**

- **Cursor IDE 通过 MCP Server 集成获得强力增强**：[Cursor IDE](https://codeium.com/windsurf/download-next) 现在支持 **MCP server 集成**，使用户能够直接在 IDE 中利用 Perplexity 和其他工具，正如 [YouTube 教程](https://www.youtube.com/watch?v=MAicJ6KKccU)中演示的那样。这一增强功能允许复杂的流水线和定制化的 AI 辅助，并可以通过提供的 [GitHub 仓库](https://github.com/daniel-lxs/mcp-starter)轻松设置。
- **JetBrains 的 Codeium 插件面临用户稳定性困扰**：用户报告 **Codeium JetBrains 插件**存在严重的稳定性问题，理由是频繁的无响应和需要重启，这促使一些用户转回使用 Copilot。用户的诉求 *“请给 Jetbrains 插件一些关爱”* 突显了社区对更可靠插件体验的需求。
- **Windsurf Next Beta 旨在超越 Cursor，但额度问题令人痛苦**：[Windsurf Next Beta](https://codeium.com/blog/windsurf-next) 发布以预览创新功能，但用户正挣扎于额度分配，特别是 **flex credits**，导致工作流中断。与 Cursor 的对比突显了 Cursor 在第三方工具和扩展灵活性方面的优势，表明 Windsurf 可以通过采用类似功能来提升其价值。

**Theme 3. 高级模型训练与优化技术**

- **Unsloth 发布动态 4-bit 量化以提升精度**：[Unsloth 推出了动态 4-bit 量化（Dynamic 4-bit Quantization）](https://unsloth.ai/blog/dynamic-4bit)，通过选择性地量化参数，在保持 VRAM 效率的同时提高模型精度。与标准量化技术相比，该方法增强了 DeepSeek 和 Llama 等模型的性能，为模型压缩提供了一种细致入微的方法。
- **Ladder-Residual 架构在 Torchtune 上大幅提升 Llama 70B 性能**：当在 **Torchtune** 中使用时，[Ladder-residual 修改](https://x.com/zhang_muru/status/1886870194443968529)在具有张量并行（tensor parallelism）的多 GPU 设置上，将 **70B Llama** 模型的速度提升了约 **30%**。这项由 TogetherCompute 开发的增强功能标志着分布式模型训练效率的重大进步。
- **Harmonic Loss 挑战神经网络中的交叉熵**：一篇新论文引入了 [harmonic loss](https://arxiv.org/abs/2502.01628) 作为标准交叉熵损失（cross-entropy loss）的替代方案，声称在神经网络和 LLM 中具有更好的可解释性和更快的收敛速度。虽然一些人对其新颖性表示怀疑，但其他人看到了其改变优化目标和改善模型训练动态的潜力。

**Theme 4. AI 开发中的开源与社区**

- **Mistral AI 品牌重塑，加倍投入开源**：[Mistral AI 推出了重新设计的网站](https://mistral.ai/en)，强调了他们对开源模型和为企业部署提供可定制 AI 解决方案的承诺。品牌重塑信号了对透明度和社区参与的关注，巩固了他们作为开源 AI 领先贡献者的地位。
- **GPT4All v3.9.0 发布，带来 LocalDocs 和模型扩展**：[GPT4All v3.9.0 发布](https://discord.com/channels/1076964370942267462/1090471714888102009/1336489836286312521)，具有 **LocalDocs** 功能、错误修复以及对 **OLMoE** 和 **Granite MoE** 等新模型的支持。此更新增强了该开源本地 LLM 平台的可用性和通用性。
- **Stability.ai 任命首席社区官以提升参与度**：Stability.ai 任命 Maxfield 为其新任首席社区官（Chief Community Guy），承认 *“Stability 最近的参与度一直不尽如人意”* 并承诺加强互动。Maxfield 计划实施功能请求板，并增加 Stability 研究人员的透明度，以使开发更好地符合社区需求。

**Theme 5. 推理模型基准测试与性能分析**

- **DeepSeek R1 Nitro 在 OpenRouter 上表现出极速运行时间**：[DeepSeek R1 Nitro 在 OpenRouter 上实现了 97% 的请求完成率](https://x.com/OpenRouterAI/status/1887212200647139731)，为使用 API 的用户展示了改进的运行时间和速度。OpenRouter 鼓励用户*尝试使用*以获得更强的性能。
- **据称 DeepSeek R1 可与 OpenAI 的推理能力相媲美**：讨论强调 [DeepSeek R1](https://discord.com/channels/1053877538025386074/1149866623109439599/1336427090840391680) 是 OpenAI O1 推理模型的强力开源竞争对手，在提供开放权重的同事具备可比拟的能力。成员们注意到它在本地执行的可访问性以及在推理任务中的出色表现。
- **Flux 在 Hugging Face L40S 上的图像生成速度超越 Emu3**：[Flux 使用 flash-attention 在 Huggingface L40S 上仅用 30 秒就生成了一张 1024x1024 的图像](https://discord.com/channels/1189498204333543425/1189498205101109300/1336501897229766676)，显著快于 Emu3 生成较小的 720x720 图像所需的约 600 秒。尽管参数量相似，这种速度差异引发了人们对 Emu3 相对于单模态模型效率的质疑。


---

# PART 1: Discord 高层摘要




## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 管理代码错误**：用户正在利用 **Aider** 管理大型项目中的错误，并使用 `/run` 命令重构代码以自主解决问题，同时根据 [Linting and Testing 文档](https://aider.chat/docs/usage/lint-test.html#linting) 添加文件进行诊断和解决。
   - 社区讨论了使用 **Aider** 进行自动代码修改、简化工作流程和自动化编码任务。
- **Claude 在编程方面击败 R1**：对 **O3 Mini**、**R1** 和 **Claude** 模型的比较显示了不同的编程任务成功率，根据[这条推文](https://x.com/scaling01/status/1887083317838618840)，一些用户认为在特定场景下 **Claude** 略胜 **R1**。
   - 用户对模型准确性的局限性表示沮丧，同时考虑将 **DeepClaude** 等工具与 **OpenRouter** 集成的可能性。
- **LLM 在 Rust 语言上表现挣扎**：社区承认，尽管取得了进展，LLM 在处理复杂任务时仍然会遇到困难，尤其是在 **Rust** 等语言中，难以应对深层推理和多步解决方案。
   - 虽然 LLM 在简单任务上表现出色，但在处理更复杂的问题并获得满意结果方面挑战依然存在。
- **Aider 提交乱码内容**：用户报告称，使用来自 Together.ai 的推理模型时，生成的提交消息充满了 `<think>` 标记，根据[文档](https://aider.chat/docs/config/reasoning.html)，需要进行配置更改。
   - 讨论包括建议使用 `--weak-model something-else` 来避免这些标记，这表明问题源于 **Aider** 与不同 API 提供商之间的交互。
- **Gemini 2.0 现已上线 LMSYS**：**Gemini 2.0** 现已在 [lmarena.ai](https://lmarena.ai) 上可用，以便进行更广泛的比较。
   - 社区可能会评估将其集成到现有工作流程中。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **动态量化提升准确率**：**Unsloth** 推出了 **Dynamic 4-bit Quantization**（动态 4 位量化），通过选择性量化参数，在保持 VRAM 效率的同时提高准确率，详见其 [博客文章](https://unsloth.ai/blog/dynamic-4bit)。
   - 与标准量化技术相比，该方法旨在增强 **DeepSeek** 和 **Llama** 等模型的性能。
- **期待集成 GRPO 以增强训练**：**Unsloth** 正在积极集成 **GRPO training**，以简化并增强模型微调，有望实现更高效的训练过程，详见 [此 GitHub issue](https://github.com/unslothai/unsloth/issues/1561)。
   - 社区对 **GRPO support** 的期待非常高，尽管大家也承认可能还有一些 *kinks*（小瑕疵）需要解决，这表明正式实现可能还需要一些时间。
- **揭示 DeepSeek 在 Oobagooba 上的挑战**：用户在 **Oobagooba** 本地运行 **DeepSeek** 模型时遇到了问题，通常是由于模型权重配置错误导致的。根据 [Unsloth Documentation](https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-r1-on-your-own-loc)，成员建议使用标志 *--enforce-eager* 来防止模型加载失败。
   - 优化建议包括确保使用 *--enforce-eager* 标志，以防止模型加载失败。
- **CPT 模型显示出令人印象深刻的困惑度得分**：**CPT with Unsloth** 模型在 **Perplexity** (PPL) 方面表现出重大改进，基础模型得分从 200 左右降至 80 左右，受到了热烈欢迎。
   - 成员们指出 **DeepSeek model** 已经很老了，表示需要更新版本和更有趣的数据集，特别是针对该模型的 **math versions**（数学版本）。
- **LLM 模型重组的困扰**：一位成员报告称，在尝试在 **PyTorch** 神经网络中重组 **LLM model** 时错误地导入了一个层，导致输出乱码，并寻求优化模型效率的帮助。
   - 他们寻求关于如何理解模型不同部分的效率改进点，以修复输出乱码问题的建议。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Gemini 2.0 Flash 在 Windsurf 亮相**：**Gemini 2.0 Flash** 现已在 Windsurf 上线，每次工具调用仅消耗 **0.25 user prompt credits** 和 **0.25 flow action credits**，因其在编程查询中的速度而备受关注。
   - 根据 [Windsurf's announcement](https://x.com/windsurf_ai/status/1887235006374035966)，尽管它效率很高，但用户观察到其工具调用能力有限。
- **Windsurf Next Beta 发布**：**Windsurf Next Beta** 版本已开放下载 [此处](https://codeium.com/windsurf/download-next)，允许用户测试软件开发中 AI 的创新功能和改进。
   - 正如 [Windsurf Next Launch](https://codeium.com/blog/windsurf-next) 博客文章中所详述，它至少需要 **OS X Yosemite**、**Ubuntu 20.04** 或 **Windows 10 (64-bit)**。
- **用户报告 Codeium 插件使用困难**：多位用户提到了 **Codeium** JetBrains 插件的问题，描述其频繁无响应且需要重启才能维持功能，导致一些用户转回使用 **Copilot**。
   - 一位用户恳求道：*“Please give the Jetbrains plugin some love”*（请给 Jetbrains 插件多一点关爱），以增强其稳定性，突显了社区对可靠工具的需求。
- **Windsurf 额度分配令用户恼火**：用户在 **Windsurf** 中面临额度分配问题，特别是 flex credits，这导致功能受限，一名用户已多次联系支持部门。
   - 讨论强调了用户对系统可靠性的沮丧，这影响了关键的工作流程。
- **Windsurf 落后于 Cursor**：用户指出 Cursor 相对于 Windsurf 的优势在于其安装第三方工具和扩展的灵活性。
   - 他们建议 Windsurf 可以通过允许类似的功能来提升其价值，特别是在 IDE 内移动第三方应用程序方面。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stability 迎来社区负责人**：Stability.ai 引入了 Maxfield 作为新的首席社区负责人 (Chief Community Guy)，强调了他自 2022 年 11 月以来对 **Stable Diffusion** 的参与，并承认 *Stability 最近的社区参与度一直不尽如人意*。
   - Maxfield 计划通过功能需求板（用于收集社区建议）以及增加 Stability 研究人员和创作者的透明度来提升参与度，并表示：*如果我们不构建你们想要的东西，那么所有这些算力的意义何在？*
- **扩散模型嵌套探索**：**general-chat** 频道的讨论显示了对嵌套 AI 架构的兴趣，即一个扩散模型在另一个模型的潜空间 (latent space) 内运行，尽管兼容的 **VAE** 是必不可少的。
   - 用户寻找探索这一概念的论文，但分享的链接很少。
- **模型训练被证明很棘手**：用户报告在训练 **LoRA** 等模型时遇到挑战，指出默认设置通常优于复杂的调整，并引用了 [NeuralNotW0rk/LoRAW](https://github.com/NeuralNotW0rk/LoRAW)。
   - 架构演进的复杂性让一些用户渴望更精简、更易用的工具，以便在潜空间中有效地工作。
- **未来 AI 模型引发关注**：社区对未来的多模态模型进行了推测，对融合文本和图像生成能力的工具表现出热情，例如 [PurpleSmartAI](https://purplesmart.ai/) 之类的项目。
   - 人们有兴趣开发新模型，通过直观的界面增强视频游戏开发等创意用途，并围绕该概念举办了黑客松 [Multimodal AI Agents - Hackathon · Luma](https://lu.ma/fyu8iqnk)。
- **用户对抗 Discord 垃圾信息**：**general-chat** 频道发生了一起垃圾信息事件，用户迅速举报了该消息并主张采取禁言措施。
   - 社区通过标记无关的推广帖子，展示了维护频道纯净的集体努力。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 新增 MCP Server 集成**：Cursor IDE 现在支持 **MCP server 集成**，使用户能够通过命令利用 Perplexity 提供协助，并使用提供的 [GitHub 仓库](https://github.com/daniel-lxs/mcp-starter) 进行轻松设置。
   - 一位用户在[这段 Youtube 视频](https://www.youtube.com/watch?v=MAicJ6KKccU)中展示了如何通过 MCP 打造具有增强功能的“加强版 Cursor”。
- **Gemini 2.0 Pro 编程能力受到质疑**：根据 [lmarena.ai](https://x.com/lmarena_ai/status/1887180371219132898) 的数据，用户批评 **Gemini 2.0 Pro 模型**虽然在数据分析方面表现良好，但在处理编程任务时却很吃力。
   - 基准测试显示，尽管 **Gemini 2.0 Pro** 在处理随机任务时表现尚可，但在编程任务上仍落后于 **O3-mini high** 和 **Sonnet**；可以在 [HuggingFace](https://huggingface.co/blog/prithivMLmods/o3-mini-vs-deepseek-r1) 上查看对比。
- **程序员使用语音听写**：讨论者探索了语音听写工具，引用了 **Andrej Karpathy** 使用 **Whisper** 技术的编程听写方法，尽管 Windows 内置听写功能的准确性仍有待提高。
   - 为编程定制语音接口引发了兴趣，目标是提高速度和准确性。
- **移动端 Cursor 引发讨论**：一位用户提议开发 **Cursor 的 iPhone 应用**，以便随时随地进行编程和提示词编写 (prompting)，但共识表明目前的框架可能无法证明这种开发投入的合理性。
   - 社区权衡了开发移动版 Cursor 的实用性，指出其优势可能无法抵消开发成本。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 的 UI 更改引起用户不满**：用户对 **Perplexity AI** 最近的 UI 更改表示不满，特别是焦点模式（focus modes）的移除和性能变慢。
   - 一些用户在访问 **Claude 3.5 Sonnet** 等模型时遇到困难，在 Pro Search 模式下会自动激活 **R1** 或 **o3-mini**。
- **Gemini 2.0 Flash 登场**：**Gemini 2.0 Flash** 已向所有 **Pro** 用户发布，正如 [Aravind Srinivas 的推文](https://x.com/aravsrinivas/status/1887174442171969936?s=61)所指出的，这是自早期版本以来 Gemini 模型首次回归 Perplexity。
   - 用户对其上下文限制、与之前模型相比的能力以及在当前应用界面中的可用性感到好奇。
- **模型访问限制引发困惑**：**Pro** 用户报告称模型访问不一致，尽管订阅了服务，但仍有人无法使用 **Gemini** 或访问所需模型，并建议通过 [Perplexity 状态页面](https://status.perplexity.com/)进行排查。
   - 用户体验存在差异，一些人认为这些限制没有必要，而另一些人仍在适应跨平台的新功能。
- **Sonar Reasoning Pro 基于 DeepSeek R1**：一名成员澄清说 **Sonar Reasoning Pro** 运行在 **DeepSeek R1** 之上，这已在其[网站](https://sonar.perplexity.ai/)上得到确认。
   - *这一发现对一些不了解底层模型的成员来说是新鲜事。*
- **在脱离联邦提议中提议美国“铁穹”**：一名成员分享了一段[视频](https://www.youtube.com/embed/Nd5zGnxX_kU)，讨论了特朗普关于**美国铁穹（US Iron Dome）**的提议，以及包括**加州脱离联邦提议**在内的持续政治进展。
   - 讨论考虑了此类军事战略对国家安全和地方治理的影响。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **谐波损失（Harmonic Loss）引发乐观情绪**：一篇介绍 **harmonic loss** 作为神经网络标准**交叉熵损失（cross-entropy loss）**替代方案的论文出现，正如 [Twitter](https://x.com/dbaek__/status/1886781418115862544) 上讨论的那样，该论文声称其提高了可解释性并加快了收敛速度。
   - 虽然一些人对其新颖性表示怀疑，但其他人指出它具有改变优化目标的潜力；这引发了关于模型训练期间稳定性和激活交互的讨论。
- **VideoJAM 生成动作**：Hila Chefer 介绍了 **VideoJAM 框架**，旨在通过直接解决视频生成器在没有额外数据的情况下在动作表征方面面临的挑战来增强动作生成，正如 [Twitter](https://fixupx.com/hila_chefer/status/1886791105380851917) 和[项目网站](https://hila-chefer.github.io/videojam-paper.github.io/)上所讨论的那样。
   - 它旨在直接解决视频生成器在动作表征方面面临的挑战，而无需额外数据，并有望改善视频内容中动作生成的动态性。
- **GAS 提升 TPS 优于 Checkpointing**：不使用**激活检查点（activation checkpointing）**并使用 **GAS** 进行训练可显著提高 TPS，对比显示 Batch Size 为 8 且使用 GAS 的 TPS 为 **242K**，而 Batch Size 为 48 且使用 Checkpointing 的 TPS 为 **202K**。
   - 尽管有效 Batch Size 不同，但使用 GAS 的较小 Batch Size 显示出更快的收敛速度，虽然较低的 **HFU/MFU** 可能会引起关注，但如果 TPS 得到改善，则不会被优先考虑。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek R1 Nitro 提速**：**DeepSeek R1 Nitro** 展示了更好的运行时间和速度，请求完成率达到 **97%**。
   - 根据 [OpenRouterAI](https://x.com/OpenRouterAI/status/1887212200647139731) 的消息，现在鼓励用户*尝试使用*。
- **OpenRouter 恢复在线**：用户报告了 **API 问题**和**速率限制错误**，引发了对服务可靠性的担忧，但在撤销最近的一项更改后，服务立即恢复。
   - Toven 确认了停机时间并宣布了修复方案，向用户保证服务功能已恢复。
- **Anthropic API 遭遇速率限制**：用户在使用 API 时遇到速率限制错误，特别是 **Anthropic**，其限制为**每分钟 2000 万输入 token**。
   - Louisgv 提到正在联系 Anthropic，寻求提高速率限制的可能性以解决这些限制。
- **Gemini 2.0：处于磨合期？**：Xiaoqianwx 发起了关于对 **Gemini 2.0** 预期的讨论，以及是否需要更强大的模型来有效竞争。
   - 社区对其表现普遍感到失望，并正在积极讨论该模型的优缺点。
- **OpenRouter 即将推出价格控制功能**：用户询问了关于 API 使用的潜在**价格控制**，特别是针对不同供应商之间的成本差异。
   - Toven 引入了一个新的 `max_price` 参数用于控制 API 调用的支出上限，该功能目前已上线，但尚未提供完整文档。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Gemini 2.0 携 Flash 和 Experimental Pro 亮相**：基准测试显示 **Gemini 2.0 Pro Experimental** 的性能与 **Claude 3.5 Sonnet** 相当，尽管注意到 API 响应存在不一致性；同时 **Gemini 2.0 Flash** 已集成到 **GitHub Copilot** 中供所有用户使用，作为开发者的新工具，使 **Gemini** 在微软生态系统中领先竞争对手获得显著关注。
   - 一些用户认为，虽然 **Flash** 模型性能超越了 **Gemini 1.5 Pro**，但长上下文能力似乎有所减弱，并且他们认为 AI 模型的命名惯例缺乏创意和清晰度（[DeepMind 推文](https://x.com/GoogleDeepMind/status/1887172464863506547), [GitHub 推文](https://x.com/github/status/1887208954704355350)）。
- **Mistral 品牌重塑，专注于开源**：Mistral 推出了重新设计的网站（[Mistral AI](https://mistral.ai/en)），展示了旨在定制 AI 解决方案的开源模型，强调透明度和企业级部署选项，并采用了新的猫咪 logo。
   - 该公司在保持领先独立 AI 实验室形象的同时，采用了俏皮的设计风格。
- **软银的 AGI 梦之队**：讨论围绕公司是否有必要探索各种途径，在**两年内**向**软银**交付 **AGI**，因为该公司预期收入将达到 1000 亿美元。
   - 社区正在思考这是否是一个现实的时间表。
- **DeepSeek R1 在审视中亮相**：2025 年 1 月 20 日，DeepSeek 发布了其开源权重推理模型 **DeepSeek-R1**，引发了围绕其在 [这篇 Gradient Updates 文章](https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1) 中公布的训练成本真实性的辩论。
   - 该模型的架构与 **DeepSeek v3** 相似，引发了关于其性能和定价的讨论。
- **Karpathy 转向 Vibe Coding**：Andrej Karpathy 引入了 **'vibe coding'** 的概念，拥抱像 Cursor Composer 这样的 LLM 并绕过传统编码，他表示自己*几乎不再阅读 diff 了*。
   - 他补充道：*“当我收到错误消息时，我只是不加评论地直接复制粘贴进去，”* 正如在 [这条推文](https://x.com/sighswoon/status/1886826813219070290?s=46) 中所见。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 面临低 VRAM 困扰**：用户分享了在旧 CPU 和 GPU 上运行 **LM Studio** 的挑战，特别是像 **RX 580** 这样显存较低的显卡，导致了性能限制。
   - 一些用户建议在不带 **AVX** 支持的情况下编译 **llama.cpp**，以增强在旧系统上的性能。
- **推荐 Qwen 2.5 用于编程**：**Qwen 2.5 model** 获得了需要编程任务支持的用户的推荐，特别是对于那些具有特定硬件配置的用户。
   - 用户根据本地安装的性能和可用性表达了对模型的偏好。
- **Vulkan 支持效果参差不齐**：在 **llama.cpp** 中启用 **Vulkan** 支持以提高 GPU 利用率需要特定的构建配置，这引发了关于 **LM Studio** 细节的讨论。
   - 分享的资源强调了使用 **Vulkan** 进行编译的设置要求。
- **GPT-Researcher 困扰 LM Studio**：将 **GPT-Researcher** 与 **LM Studio** 集成的用户报告称，在模型加载和 embedding 请求时遇到错误。
   - 具体而言，**404 error** 表示未加载任何模型，导致集成尝试中断。
- **GPU 价格飙升**：对于 *eBay* 和 *Mercari* 等平台上 **GPU** 价格的担忧日益增加，由于需求旺盛，这些硬件现在被视为“增值资产”。
   - 包括 **Jetson board** 在内的组件价格膨胀正受到黄牛的影响。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **DeepSeek 表现优于下滑中的 ChatGPT**：用户报告称 **DeepSeek** 提供的答案更好，且具有比 **ChatGPT** 更具吸引力的 **chain of thought methodology**（思维链方法论），但由于高流量，访问受到限制。
   - 用户对 **DeepSeek** 的思考过程表现出浓厚兴趣，强调其思维链方法论比传统的 AI 回复更引人入胜。
- **TinyRAG 简化 RAG 系统**：[TinyRAG](https://github.com/wannaphong/tinyrag) 项目使用 **llama-cpp-python** 和 **sqlite-vec** 进行排序、查询和生成 LLM 答案，从而简化了 **RAG** 的实现。
   - 该计划为开发者和研究人员提供了一种部署检索增强生成（retrieval-augmented generation）系统的简化方法。
- **距离学习（Distance-Based Learning）论文发表**：一篇新论文《神经网络中的距离学习》（*Distance-Based Learning in Neural Networks*，[arXiv](https://arxiv.org/abs/2502.02103)）介绍了一个几何框架和 **OffsetL2 architecture**。
   - 该研究强调了**基于距离的表示**（distance-based representations）对模型性能的影响，并将其与基于强度的方法进行了对比。
- **Agents 课程下周开启**：**Agents Course** 将于下周一启动，届时将开设用于更新、提问和项目展示的新频道。
   - 随着第一单元目录的预览，热度不断上升；但一些成员对缺乏课程所需的编程基础 **Python** 技能表示担忧。
- **HuggingFace 需要更新 NLP 课程**：成员们要求 **Hugging Face** 更新 **NLP** 课程，因为现有课程缺乏对在当今 **NLP** 框架中至关重要的 **LLM** 的覆盖。
   - 这一差距促使人们建议提供更全面的培训材料，以应对该领域的新兴趋势。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **NURBS 挑战模拟中的网格**：NURBS（非均匀有理 B 样条）提供适用于动态模拟的参数化表示，与传统网格日益低效的情况形成对比，而[现代程序化着色器](https://developer.nvidia.com/blog/high-fidelity-3d-mesh-generation-at-scale-with-meshtron/)有助于解决纹理问题。
   - 成员们注意到行业标准正向动态模型以及 **NURBS** 和 **SubDs** 等先进技术转变，摆脱了静态网格方法在动态应用中的局限性。
- **Gemini 2.0 更新 Flash 和 Lite 版本**：Google 在 Gemini API 和 Google AI Studio 中发布了[更新后的 **Gemini 2.0 Flash**](https://blog.google/technology/google-deepmind/gemini-model-updates-february-2025/)，强调与之前的 **Flash Thinking** 等版本相比具有更低的延迟和更强的性能。
   - 关于新 **Flash Lite** 模型的反馈表明其在返回结构化输出方面存在问题，用户报告在生成有效的 **JSON** 响应时遇到困难。
- **工程师制造爆火的 ChatGPT 哨兵枪**：在一段展示 AI 控制的电动哨兵枪视频走红后，OpenAI 切断了工程师 [sts_3d](https://linktr.ee/sts_3d) 的 API 访问权限，引发了对 AI 武器化的担忧。
   - 该工程师项目的快速进展凸显了与不断演进的 **AI applications** 相关的潜在风险。
- **研究人员攻克廉价 AI 推理模型**：研究人员开发了 **s1 reasoning model**，以不到 **$50** 的云计算额度实现了与 **OpenAI's models** 类似的能力，标志着成本的大幅降低 [[TechCrunch 文章](https://techcrunch.com/2025/02/05/researchers-created-an-open-rival-to-openais-o1-reasoning-model-for-under-50/)]。
   - 该模型利用蒸馏方法，从 **Google's Gemini 2.0** 中提取推理能力，从而展示了 **AI technologies** 更加普及的趋势。
- **Harmonic Loss 论文评价褒贬不一**：[Harmonic Loss 论文](https://arxiv.org/abs/2502.01628) 引入了一种更快的收敛模型，但该模型尚未表现出显著的性能提升，引发了对其实用性的争论。
   - 虽然有些人认为这篇论文“粗糙”，但其简洁性被认为很有价值，尤其是在 [其 GitHub 仓库](https://github.com/KindXiaoming/grow-crystals) 中可以获得额外的见解。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DeepSeek 数据实践受调查**：[一段 YouTube 视频](https://youtube.com/shorts/I_bGa-xIHkk?feature=shared)引发了对 **DeepSeek** 可能将数据发送到中国的担忧，理由是其服务器设在中国。
   - 讨论强调了 AI 开发中数据治理和监管标准的重要性，用户指出了 **data residency**（数据驻留）的影响。
- **ChatGPT 的推理怪癖**：用户观察到 **ChatGPT 4o** 表现出不可预测的行为，例如尽管收到的是英文提示词，却使用多种语言提供推理过程。
   - 这些报告引发了关于模型当前局限性的讨论，以及完善 AI 生成输出的一致性和清晰度的必要性。
- **Gemini 2.0 Token 上下文令人印象深刻**：**Gemini 2.0** 提供的 **2 million token context** 和免费 API 访问激起了开发者的兴趣，他们渴望探索其广阔的能力。
   - 虽然一些用户承认 **Gemini 2.0** 促进了自动化的重要性，但也有人评论说该 AI 过于啰嗦，导致阅读量过大。
- **用户编写修辞提示词**：一名成员详细介绍了一个提示词，用于生成关于为什么 **Coca-Cola** 配 **hot dog** 最好吃的说服性论点，并结合了 *Antimetabole* 和 *Chiasmus* 等高级修辞技巧。
   - 提示词结构包括论证理由、提供示例和应对反驳的部分，旨在达成连贯且有影响力的结论。
- **精灵图表 (Sprite Sheet) 模板咨询**：一位用户寻求关于优化提示词模板的建议，以生成一致的**卡通风格精灵图表**，重点在于角色和动画帧布局。
   - 尽管指定了角色设计和尺寸，但图像并未按预期对齐，因此该用户请求进行优化。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepSeek 在推理能力上媲美 OpenAI**：讨论强调了 **DeepSeek** R1 据称在完全开源且可高效运行的情况下，能与 OpenAI 的 O1 推理模型相媲美。
   - 成员们注意到像 **Gemini** 这样较新的模型在执行数学任务方面的惊人能力，以及品牌命名的复杂性如何困扰用户；参见 [Gemini 2.0 现已向所有人开放](https://blog.google/technology/google-deepmind/gemini-model-updates-february-2025)。
- **AI 与 Crypto 的联系？**：一位成员推测对 AI 的抵制是否与 2020-21 年 **NFT** 和 **crypto** 争议的余波有关。
   - 他们引用了 [为什么大家突然对 AI 感到愤怒](https://rentry.org/vwa65v85) 及其对 AI 认知的启示，将其与过去的技术炒作联系起来。
- **DeepResearch 获得好评**：用户对 OpenAI 的 **DeepResearch** 功能充满热情，称赞其性能和高效检索冷门信息的能力，如[这条推文](https://x.com/marionawfal/status/1886613146942759108?s=46)所示。
   - 成员们讨论了利用知识图谱丰富结果，以增强事实核查和研究的准确性。
- **Liger Kernel 获得 GRPO Chunked Loss**：最近的一个 [pull request](https://github.com/linkedin/Liger-Kernel/pull/553) 为 Liger Kernel 增加了 **GRPO chunked loss**，解决了 issue #548。
   - 开发者可以运行 **make test**、**make checkstyle** 和 **make test-convergence** 来测试正确性和代码风格。
- **基础设施团队被低估了？**：成员们注意到，由于需要致谢 **hardware infrastructure** 团队，许多**预训练论文**都有大量作者。
   - 有人强调，**infra 团队**忍受挑战是为了让**研究科学家**能够专注于工作而不受干扰——*基础设施人员受苦，是为了让研究科学家不必受苦。*

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 编译器转为闭源**：据一名团队成员称，受管理快速变化的需要驱动，**Mojo 编译器**转为了闭源。
   - 编译器爱好者渴望了解其内部工作原理，特别是 **MLIR** 中的自定义 lowering passes，但根据[这段视频](https://www.youtube.com/watch?v=XYzp5rzlXqM)，必须等到 2026 年底。
- **Mojo 计划在 2025 年第四季度开源**：据一名团队成员称，Modular 目标是在明年第四季度开源 Mojo 编译器，尽管有人希望能更早发布。
   - 目前没有计划在整个编译器开源之前发布 MLIR 中的单个 dialects 或 passes，这让*编译器极客们*的希望落空。
- **Mojo 标准库面临设计选择**：围绕 **Mojo 标准库**是否应该演变为具有 Web 服务器和 JSON 解析等功能的通用库展开了辩论。
   - 有人对支持广泛用例的复杂性表示担忧，这提高了向 `stdlib` 贡献新功能的门槛，正如[这个 GitHub 仓库](https://github.com/mojicians/awesome-mojo)所整理的那样。
- **Async 函数引发讨论**：Mojo 中 **async 函数**的处理正在讨论中，有人提议采用新语法以提高清晰度并实现性能优化，如[这份提案](https://github.com/modular/mojo/pull/3946#issuecomment-2601176112)所示。
   - 参与者对维护独立的 async 和 sync 库的复杂性，以及对不同版本功能的可用性影响表示担忧。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **法律 AI 自动化草拟**：AI 现在被用于自动化草拟重复性的法律文件，利用以往案例的模板作为来源，使流程更加高效，成员们发现 [AI 非常有用](https://discord.com/channels/1124402182171672732/1124403655819415592/1336451112470839347)。
   - 一位律师报告称 AI 可靠并提供清晰的来源，特别是对于类似案件或大规模诉讼。
- **虚拟形象提升合同审查**：成员们正在尝试在合同审查中使用虚拟形象，使修订分析更具吸引力，正如 [YouTube 视频](https://youtu.be/1G4W6XWl2WE?t=5) 中展示的那样。
   - 增加虚拟形象旨在使产品差异化，并有效地支持客户团队。
- **NotebookLM Plus 激活问题出现**：根据 [Google Support](https://support.google.com/a/answer/6043385?hl=en&co=DASHER._Family%3DBusiness&oco=0)，Google Workspace 管理员在激活 **NotebookLM Plus** 时面临问题，需要 Business Standard 或更高级别的许可证才能访问高级功能。
   - 已分享相关资源以帮助管理员启用和管理用户访问，重点在于了解具体要求和所需的许可证。
- **电子表格集成仍面临挑战**：正如 [general 频道](https://discord.com/channels/1124402182171672732/1124402182909857966/1336452719602434172) 中所述，用户对 **NotebookLM** 分析电子表格中表格数据的有效性表示担忧，建议 **Gemini** 可能更适合复杂的数据任务。
   - 讨论围绕上传电子表格的最佳实践以及数据识别能力的局限性展开。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 在显存管理上胜过 Unsloth**：用户报告称，在 12GB 的 4070 显卡上，**Torchtune** 处理微调时没有出现 **Unsloth** 中常见的 CUDA 显存问题。
   - 除非使用过大的 batch sizes，否则该工具可以避免遇到同样的显存问题。
- **Ladder-Residual 极大提升 Llama 速度**：根据 [@zhang_muru 在 TogetherCompute 的工作](https://x.com/zhang_muru/status/1886870194443968529)，**Ladder-residual** 修改在多 GPU 张量并行下使 **70B Llama** 模型加速了 **约 30%**。
   - 这一增强功能由 @MayankMish98 共同作者完成，并由 @ben_athi 指导，标志着分布式模型训练的显著进步。
- **Kolo 开启 Torchtune 集成**：Kolo Docker 工具现在提供对 **Torchtune** 的官方支持，为新手简化了本地模型训练和测试，[项目链接](https://github.com/MaxHastings/Kolo)。
   - 由 MaxHastings 创建的 Kolo Docker 工具旨在促进在单一环境中使用一系列工具进行 **LLM 训练** 和 **测试**。
- **为 Torchtune 定制的 Tune Lab UI**：一位成员正在开发 **Tune Lab**，这是一个用于 **Torchtune** 的 FastAPI 和 Next.js 界面，使用现代 UI 组件来增强用户体验，[Tune Lab 仓库](https://github.com/theosis-ai/tune-lab)。
   - 该项目旨在集成预构建和自定义脚本，邀请用户参与其开发。
- **GRPO 为训练带来巨大提升**：据成员报告，**GRPO** 实现取得了显著成功，将 GSM8k 上的训练性能从 10% 提升到 40%。参见 [相关 issue](https://github.com/pytorch/torchtune/issues/2340)
   - 该实现涉及解决与死锁和显存问题相关的调试挑战，并计划重构代码供社区使用。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 计划推出 SWE Agent**：据 [一条推文](https://x.com/harambe_musk/status/1886779961790345657?s=46) 称，OpenAI 计划在第一季度末或第二季度中期发布新的 **SWE Agent**，由面向企业的 **O3** 和 **O3 Pro** 提供支持。
   - 该 Agent 预计将对软件行业产生重大影响，据称可以与中级工程师竞争，这一消息在 [一场直播](https://www.youtube.com/live/Gv7torZn5lM?si=cGtkvCCtfrj3vkcO) 中被发现。
- **OmniHuman 生成头像视频**：据 [一条推文](https://x.com/unseenvie/status/1886672598576325011?s=46) 称，新的 **OmniHuman** 视频研究项目可以从单张图像和音频生成逼真的头像视频，且不受长宽比限制。
   - 该项目被誉为一项突破，其细节水平令观众感到 *震惊*。
- **Figure AI 与 OpenAI 分道扬镳**：据 [一条推文](https://x.com/adcock_brett/status/1886860098980733197) 称，Figure AI 在报道称取得突破后，退出了与 OpenAI 的合作协议，转而专注于内部 AI 技术。
   - 据 [TechCrunch](https://techcrunch.com/2025/02/04/figure-drops-openai-in-favor-of-in-house-models/) 报道，创始人暗示将在 30 天内展示 *从未有人在人形机器人上见过的东西*。
- **Gemini 2.0 Flash 正式发布 (GA)**：据 [一条推文](https://x.com/sundarpichai/status/1887169871697350775) 称，Google 宣布 **Gemini 2.0 Flash** 现已正式发布，使开发者能够创建生产级应用程序。
   - 据 [一条推文](https://x.com/arankomatsuzaki/status/1887211023423431134?s=46) 称，该模型支持 **200 万 token** 的上下文，引发了关于其相对于 Pro 版本性能的讨论。
- **Mistral AI 重塑平台品牌**：据 [其官网](https://mistral.ai/en) 显示，Mistral AI 的网站进行了重大品牌重塑，推广其可定制、可移植且企业级的 AI 平台。
   - 他们强调了自己作为开源 AI 主要贡献者的角色，以及致力于提供引人入胜的用户体验的承诺。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All v3.9.0 发布**：**GPT4All v3.9.0** 已发布，具有 **LocalDocs** 功能，并增强了对 **OLMoE** 和 **Granite MoE** 等新模型的支持。
   - 新版本还修复了在使用推理模型时后续消息出现的错误，并增强了 Windows ARM 支持。
- **推理增强生成 (ReAG) 亮相**：[ReAG](https://github.com/superagent-ai/reag) 直接将原始文档输入语言模型，与传统方法相比，有助于实现更具上下文感知能力的响应。
   - 这种方法通过避免过度简化的语义匹配来提高准确性和相关性。
- **GPT4All 作为自托管服务器**：用户讨论了在桌面端自托管 **GPT4All** 以实现移动端连接，这可以通过 Python 主机实现。
   - 虽然可行，但支持可能有限，并且可能需要非常规的设置。
- **NSFW 内容寻找本地模型**：成员们讨论了用于 NSFW 故事的本地可用 LLM，发现 *wizardlm* 和 *wizardvicuna* 效果欠佳。
   - 像 *obadooga* 和 *writing-roleplay-20k-context-nemo* 这样的替代方案在生成 NSFW 内容方面可能提供更好的性能。
- **UI 滚动 Bug 出现**：一位用户报告了一个 UI Bug，即如果文本超过可见区域，提示窗口的内容无法滚动，从而导致可访问性问题。
   - 此前在 GitHub 上也报告过类似问题，表明这是一个更广泛的问题。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **ChatGPT Pro 引发团队兴趣**：成员们有兴趣购买 **ChatGPT Pro** 订阅，可能通过多个账号分摊成本供团队使用。
   - 兴趣集中在将 **ChatGPT Pro** 用于开发，但对账号拆分和适当的使用策略提出了担忧。
- **Excel MCP 构想涌现**：围绕创建一个用于读取和操作 **Excel 文件** 的 **MCP** 展开了热烈讨论，并就使用 **Python** 还是 **TypeScript** 进行了辩论。
   - 讨论强调了自动化数据处理任务的潜力，但每种语言的可行性是争论的焦点。
- **Playwright 胜过 Puppeteer**：分享的经验表明 **Playwright** 与 **MCP** 配合良好，而 **Puppeteer** 需要本地修改，且此 [GitHub 实现](https://github.com/isaacphi/servers/blob/evaboot/src/puppeteer/index.ts) 尚未达到生产就绪状态。
   - 用户比较了这两种工具在自动化项目中的实现难度，更倾向于 **Playwright** 更简单的集成方式。
- **Home Assistant 添加 MCP 客户端/服务器支持**：**Home Assistant** 发布了对 **MCP 客户端/服务器的支持**，扩展了集成能力，很高兴看到自动化生态系统的进一步融合。
   - 这一集成有望增强自动化工作流，允许用户在其家庭自动化设置中利用 **MCP**。
- **PulseMCP 展示使用案例**：PulseMCP 发布了一个[新展示页](https://www.pulsemcp.com/use-cases)，包含实用的 **MCP** 服务器和客户端组合，并附有详细说明、截图和视频。
   - 示例包括使用 **Gemini voice** 管理 **Notion**，以及利用 **Claude** 将 **Figma 设计** 转换为代码，展示了 **MCP** 应用的多样性。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Flux 在图像生成方面碾压 Emu3**：在 Huggingface L40S 上，**Flux** 使用 `flash-attention` 和 W8A16 量化，在 **30 秒** 内生成了一张 1024x1024 的图像，使 **Emu3** 生成 720x720 图像所需的约 600 秒相形见绌。
   - 尽管参数量相当（Emu3 为 8B，Flux 为 12B），但速度差异引发了关于 **Emu3** 与单模态模型相比效率如何的疑问。
- **OmniHuman 制作逼真的人物视频**：[OmniHuman 项目](https://omnihuman-lab.github.io/) 仅凭一张图像即可生成高质量的人物视频内容，突显了其在多媒体应用中的潜力。
   - 其独特的框架通过混合训练策略实现了端到端的多模态条件人物视频生成，大大提升了生成视频的质量。
- **FlowLLM 启动材料发现**：[FlowLLM](https://arxiv.org/abs/2410.23405) 是一种新型生成模型，它将大语言模型（LLM）与黎曼流匹配（Riemannian flow matching）相结合，用于设计新型晶体材料，显著提高了生成速率。
   - 该方法在材料生成速度上超越了现有方法，基于 LLM 输出开发稳定材料的效率提高了三倍以上。
- **Modal 招聘 ML 性能工程师**：[Modal](https://modal.com/) 是一个 **serverless 计算平台**，为 **Suno** 和 **Liger Kernel 团队** 等用户提供灵活、自动扩展的计算基础设施。
   - Modal 正在招聘 **ML 性能工程师** 以增强 GPU 性能，并为 **vLLM** 等上游库做出贡献，[职位描述点击此处](https://jobs.ashbyhq.com/modal/af17da5e-23ca-4802-854d-5f0546e1ed32)。
- **Torchao 在 Torch Compile 中遇到困难**：一位用户报告称，结合使用 **Torchao** 和 **torch.compile** 似乎会导致 bug，暗示存在兼容性问题。
   - 另一位成员建议该 bug 与[此 GitHub issue](https://github.com/pytorch/pytorch/issues/141548) 一致，涉及 `nn.Module` 无法在设备间转移的问题。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Deepseek 论坛将探索工作流**：@aicampai 正在举办一场关于 **Deepseek** 的虚拟论坛，重点介绍其功能以及如何集成到开发者和工程师的工作流中，详情见[此处](https://t.co/Gh29EHBJMf)。
   - 该论坛旨在提供关于 **Deepseek** 技术及其应用的实战学习体验。
- **新教程指导构建首个 RAG 应用**：@Pavan_Belagatti 发布了一个视频教程，指导用户使用 @llama_index 构建他们的第一个 **Retrieval Augmented Generation (RAG)** 应用程序，链接见[此处](https://t.co/LXlRztHcM4)。
   - 这是为了回应新用户寻求 **RAG** 应用开发的实践见解。
- **Gemini 2.0 发布并支持 LlamaIndex**：@google 宣布 **Gemini 2.0** 已全面上市，且 @llama_index 提供零日支持，详见其[发布博客文章](https://t.co/6oBbYpcFAU)。
   - 用户可以通过 `pip install llama-index-llms-gemini` 安装最新的集成包来体验出色的基准测试结果，更多信息可通过 [Google AI Studio](https://aistudio.google.com/prompts/new_chat?model=gemini-2.0-flash) 获取。
- **LlamaIndex LLM 类缺少超时设置**：一位用户观察到默认的 **LlamaIndex LLM** 类缺乏内置的超时（timeout）功能，而 **OpenAI** 的模型中存在该功能，链接见[此处](https://github.com/run-llama/llama_index/blob/7391f302e18542c68b9cf5025afb510af4a52324/llama-index-integrations/llms/llama-index-llms-azure-inference/llama_index/llms/azure_inference/base.py#L224)。
   - 另一位用户建议，超时设置可能包含在 client kwargs 中。
- **解决 Qwen-2.5 的 Function Calling 问题**：一位用户报告了在使用 **Qwen-2.5** 进行 **Function Calling** 时出现 `ValueError`，建议使用命令行参数并切换到类 **OpenAI** 的实现，文档见[此处](https://qwen.readthedocs.io/en/latest/framework/function_call.html)。
   - 为了顺利在 **Qwen-2.5** 中使用 **Function Calling**，另一位用户转而实现 LlamaIndex 的 `OpenAILike` 类。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC 证书发放延迟**：一名成员报告称 12 月份申请的证书发放延迟，课程工作人员目前正在努力加快分发进度。
   - 课程工作人员希望在未来一两周内解决这些问题。
- **测验难以查找**：成员们询问了 **Quiz 1** 和 **Quiz 2** 的可用性，课程工作人员确认 **Quiz 2** 尚未发布，并提供了 [Quiz 1 的链接](https://forms.gle/c6Zz5kGPUzkNTQiq9)。
   - 成员们可以在周五之后完成 **Quiz 1**，目前没有截止日期。
- **第一课视频现已配备专业字幕**：一名成员确认 [YouTube 链接](https://www.youtube.com/live/g0Dwtf3BH-0) 指向的是第一课的修正版本，标题为“CS 194/294-280 (Advanced LLM Agents) - Lecture 1, Xinyun Chen”。
   - 编辑后的录像包含了**专业字幕**以提高可访问性。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **用户思考 Embed v3 迁移**：一位用户询问如何将现有的 float 类型向量从 **embed v3** 迁移到 **embed v3 light**，并询问是否可以删除多余维度，或者是否需要完全重新生成数据库。
   - 缺乏直接回应突显了此类迁移过程的复杂性和疑虑。
- **渴望 Cohere 的 Moderation Model**：一名成员表达了希望 Cohere 提供 **moderation model**（审核模型）的愿望，以减少对美国服务的依赖。
   - 这一需求凸显了对满足区域要求的本地化 AI 解决方案的渴望。
- **探讨聊天功能定价**：一位用户询问了聊天功能的**月费付费选项**，表明其主要兴趣在于聊天功能而非产品开发。
   - 另一位成员指出存在需要付费使用的 **production API**。
- **对话记忆困扰开发者**：一名成员分享了他们的挫败感，即 AI 的响应在请求之间缺乏上下文关联，并寻求关于如何使用 **Java 代码** 实现**对话记忆（conversational memory）**的指导。
   - 另一位成员确认已创建了与该问题相关的支持工单，并提供了[工单链接](https://discord.com/channels/954421988141711382/1336548080677294191)，加强了社区支持。
- **社区澄清行为准则**：一名成员发出严厉提醒，称未来的违规行为可能导致**封禁（ban）**，强调必须遵守社区准则。
   - 作为回应，另一名成员为过去的行为道歉，表达了遵守社区预期的承诺。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 0.10.1 面临错误**：在将 **tinygrad** 升级到版本 **0.10.1** 时，用户报告测试失败并出现 *NotImplementedError*，原因是未知的重定位类型 4，这表示版本 **19.1.6** 不支持外部函数调用。
   - 这些问题可能与影响编译过程的 *Nix 特有行为* 有关。
- **编译器标志引发关注**：由于设置了 `NIX_ENFORCE_NO_NATIVE`，编译器警告关于跳过非纯净标志 **-march=native** 的问题引起了讨论。
   - 一名成员澄清说，移除 **-march=native** 通常适用于用户机器软件，而 **tinygrad** 使用 **clang** 作为 kernel 的 JIT 编译器，从而在 **tinygrad** 上下文中减轻了该标志的必要性。
- **调试变得更容易**：一位贡献者宣布 **PR #8902** 将改进 **tinygrad** 的调试功能，使复杂问题的解决更加可控。
   - 预期项目持续的改进将有助于缓解观察到的问题。
- **基础操作和 Kernel 实现受到询问**：一名成员询问了 **tinygrad** 中 **base operations** 的数量，寻求澄清该框架的基础元素。
   - 随后有人请求提供与 **tinygrad** 相关的 **kernel implementations** 源码，表明了对理解底层代码库的兴趣。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **API Endpoint 需要公开**：向 **leaderboard** [添加新模型说明](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/CONTRIBUTING.md)中规定，虽然可能需要 **authentication**，但 API endpoint 应该对 **公众** 开放。
   - 可访问性旨在确保 **API endpoint** 更广泛的可用性。
- **Raft 方法在 Llama 3.1 7B 上是否足够？**：一名成员询问 **1000 个用户的数据** 是否足够用于 **Llama 3.1 7B** 的 **Raft method** 训练，以及在应用 **RAG** 之前是否应加入 **synthetic data**。
   - 有人担心 **1000 个用户的数据** 可能无法为有效的模型训练提供足够的样性，建议可能需要 **synthetic data** 来填补空白并提高训练效果。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Chain of Agents 在 DSPy 中亮相**：一位用户介绍了一个以 **DSPy 方式** 实现的 **Chain of Agents** 示例，详情可见[这篇文章](http://x.com/i/article/1887191253370216450)。
   - 讨论还引用了关于 **Chain of Agents** 的原始研究论文，可在此处[访问](https://openreview.net/pdf?id=LuCLf4BJsr)。
- **社区寻求 DSPy Chain of Agents 的 Git 仓库**：一位用户询问讨论中的 **DSPy** 版 **Chain of Agents** 示例是否有可用的 **Git repository**。
   - 这一请求表明社区对 **Chain of Agents** 概念的实际动手实现有着浓厚兴趣。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1336428281410621441)** (827 messages🔥🔥🔥): 

> `Aider 与深度学习模型, 模型性能对比, AI 工具集成, 使用 AI 进行项目管理, LLM 在复杂任务中的局限性`

- **Aider 利用**：用户讨论了以各种方式使用 Aider，包括管理大型项目中的错误以及将其用于代码重构。
   - 工作流包括添加文件并执行类似 /run 的命令来诊断问题，同时允许 Aider 自主工作。
- **模型对比**：对话强调了 O3 Mini、R1 和 Claude 等模型之间的性能差异，用户注意到在编程任务中的成功率各不相同。
   - 一些用户建议 Claude 在特定场景下优于 R1，许多人对模型准确性的局限性表示沮丧。
- **AI 工具集成**：围绕集成多种工具展开了讨论，例如将 DeepClaude 与 OpenRouter 结合使用，以及在编程环境中有效利用 LLM。
   - 用户认为 Aider 等工具在自动化编程任务和简化工作流方面的潜力非常有益。
- **LLM 的局限性**：小组承认，尽管有所进步，LLM 在处理复杂任务时仍然吃力，特别是在 Rust 等语言中。
   - 用户指出，虽然 LLM 在简单任务上表现良好，但在深度推理和多步解决方案方面仍面临重大挑战。
- **Perplexity 与 R1 集成**：几位用户讨论了利用 Perplexity 结合 R1 进行研究任务，强调了两者的协同潜力。
   - 有人指出，Perplexity 可能提供卓越的搜索功能，而 R1 则为收集的数据提供先进的处理能力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/scaling01/status/1887083317838618840">Lisan al Gaib (@scaling01) 的推文</a>：AWS 提到了 Claude 3.5 Opus，别这样玩弄我的感情 👀</li><li><a href="https://aider.chat/docs/usage/lint-test.html#linting">Linting 和测试</a>：自动修复 Linting 和测试错误。</li><li><a href="https://www.twitch.tv/ThePrimeagen">ThePrimeagen - Twitch</a>：CEO @ TheStartup™（价值数十亿），困在 Vim 中却向往着 Emacs</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://tenor.com/view/excited-gif-14981788540580833784">兴奋的 GIF - Excited - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-updates-february-2025/">Gemini 2.0 现已向所有人开放</a>：我们宣布了 Gemini 2.0 Flash 的新更新，并推出了 Gemini 2.0 Flash-Lite 和 Gemini 2.0 Pro Experimental。</li><li><a href="https://status.deepseek.com/">DeepSeek 服务状态</a>：未找到描述</li><li><a href="https://ai.google.dev/gemini-api/docs/models/experimental-models">未找到标题</a>：未找到描述</li><li><a href="https://status.deepseek.com">DeepSeek 服务状态</a>：未找到描述</li><li><a href="https://tenor.com/view/octacat-github-animation-smile-gif-12444790955960484344">Octacat GitHub GIF - Octacat GitHub 动画 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/get-to-work-work-simpsons-smithers-trendizisst-gif-15496843">开始工作 GIF - Get To Work Work Simpsons - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.litellm.ai/">LiteLLM</a>：LiteLLM 处理超过 100 个 LLM 的负载均衡、回退和支出跟踪，全部采用 OpenAI 格式</li><li><a href="https://tenor.com/view/oh-really-oh-really-fo-real-for-real-gif-14262438">噢真的吗 GIF - Oh Really Oh Really - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/let-him-cook-cook-cookin-big-wok-gif-141999319614473134">让他发挥 (Let Him Cook) GIF - Let him cook Cook Cookin - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://aistudio.google.com/changelog">未找到标题</a>：未找到描述</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html">高级模型设置</a>：为 LLM 配置高级设置。</li><li><a href="https://deepclaude.com/docs">DeepClaude</a>：未找到描述</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI 兼容 API</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://www.reddit.com/r/singularity/comments/1iibgfv/google_launch_gemini_20_flash_gemini_20_flashlite/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/sigoden/llm-functions">GitHub - sigoden/llm-functions：使用纯 Bash/JavaScript/Python 函数轻松创建 LLM 工具和 Agent。</a>：使用纯 Bash/JavaScript/Python 函数轻松创建 LLM 工具和 Agent。 - sigoden/llm-functions</li><li><a href="https://www.youtube.com/watch?v=pb6GtL0WFT8">自动化 AI 实战 💪 | Aider &amp; DeepSeek v3 现场编程直播 🧠</a>：在这个实验中，DeepSeek v3（通过 Aider）负责在极少人工干预的情况下构建一个项目。AI 正在开发一个摘要生成应用，该应用可以...</li><li><a href="https://github.com/getAsterisk/deepclaude/issues/13">Aider 的基准测试明确表示不涉及使用 R1 思维 Token（并表示使用它们效果更差）· Issue #13 · getAsterisk/deepclaude</a>：嘿 DeepClaude 的伙计们，我有点困惑为什么你们要突出引用 Aider 的 R1+Sonnet 基准测试结果。关于这些结果的博客文章和 Twitter 帖子明确指出...</li><li><a href="https://github.com/Aider-AI/aider/issues/2052">SDK 不太好用 · Issue #2052 · Aider-AI/aider</a>：嗨，我真的很喜欢你们的工具——我正在使用它，觉得它很棒。但是，当我尝试用 Python 封装它时，并没有预想的那么容易。虽然文档展示了如何使用 coder.r...</li><li><a href="https://github.com/jj-vcs/jj">GitHub - jj-vcs/jj：一个既简单又强大的 Git 兼容 VCS</a>：一个既简单又强大的 Git 兼容 VCS - jj-vcs/jj</li><li><a href="https://github.com/Aider-AI/aider/issues/3139#issue-2832352562">Aider 使用随机字符串作为文件名创建文件 · Issue #3139 · Aider-AI/aider</a>：问题：使用 o3-mini 进行提示，它一直在使用非常奇怪的文件名，比如 2. 用于嵌入工作器模块化集成的新文件 新文件（空文件）────────────────────────────── 我认为...</li><li><a href="https://github.com/Aider-AI/aider/issues/2879">Bug：创建的文件名仅包含文件扩展名而没有文件名。· Issue #2879 · Aider-AI/aider</a>：它建议...</li>

建议了正确的文件名，但随后会生成名为 php 而不是 install.php 的文件，或者名为 sql 而不是 migration.sql 的文件</li><li><a href="https://github.co">GitHub · Build and ship software on a single, collaborative platform</a>：加入全球应用最广泛的 AI 驱动开发者平台，数百万开发者、企业和最大的开源社区在此构建推动人类进步的软件。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1336429273753452625)** (53 messages🔥): 

> `Aider 配置问题、OpenRouter 兼容性、使用多个模型、Aider 的 Git 提交问题、使用 Aider 运行命令` 


- **Aider 执行的配置问题**：用户讨论了 Aider 从工作区根目录执行命令的行为，建议使用 `--subtree-only` 或将配置文件移动到特定分支以潜在地解决问题。
   - *Aider 默认从工作区根级别执行，* 这引发了在根目录 package.json 中创建脚本的考虑。
- **向 Aider 添加模型**：一位用户寻求关于通过 **ollama** 在本地运行 **mistral-small** 的说明，面临 API 身份验证挑战，该问题在检查 API key 后得到解决。
   - 对话强调了模型设置及其在不同配置下的正确配置可能存在的困惑。
- **Git 提交信息中充满 <think> 标记**：一位用户报告在使用来自 Together.ai 的推理模型时，其提交信息中出现了 `<think>` 标记，表明需要更改配置。
   - 建议包括使用 `--weak-model something-else` 来避免这些标记，并查阅 Aider 关于模型设置的文档。
- **Aider 与不同 API 提供商之间的交互**：讨论了各种语言模型与 Aider 的兼容性，包括提到使用不同提供商时可能导致 `<think>` 标记的不一致性。
   - 用户建议使用非推理的弱模型（weak model）以获得更好的结果，因为推理模型带有使用限制。
- **Aider 通用可用性查询**：提出了关于 Aider 代表用户运行命令的能力，以及暂存（staging）更改而非提交（committing）更改的可能性，反映了用户对灵活命令执行的普遍需求。
   - 社区反馈指出了潜在的工作流，并分享了资源以进一步了解其功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/config/reasoning.html">Reasoning models</a>：如何配置来自次要提供商的推理模型设置。</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html#model-settings">Advanced model settings</a>：为 LLM 配置高级设置。</li><li><a href="https://aider.chat/docs/usage/lint-test.html">Linting and testing</a>：自动修复 linting 和测试错误。</li><li><a href="https://github.com/Aider-AI/aider/commit/7b557c0586ce87a115d1f97aee84fe2d775806ac#diff-53d56dc7c26de36f68c39203231afe4a5fedad002697dc314297e64d2e544292R88">refactor: Change default temperature to None and remove debug dump · Aider-AI/aider@7b557c0</a>：未找到描述
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 messages): 

epicureus: lmsys 上的 gemini 2.0 https://lmarena.ai
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1336427372760662026)** (537 messages🔥🔥🔥): 

> `动态量化、本地使用 DeepSeek、GRPO 训练、模型对比、层量化策略`

- **理解动态量化 (Dynamic Quantization)**：Unsloth 引入了动态 4-bit 量化 (Dynamic 4-bit Quantization)，通过选择性地量化某些参数，在压缩模型的同时保持准确性。
   - 与标准量化技术相比，这种动态方法能够提升性能，特别是对于 DeepSeek 和 Llama 等模型。
- **在有限硬件上运行 DeepSeek**：用户评估了在显存 (VRAM) 有限的硬件（如 GTX 960 和 GTX 1660 Super）上运行各种版本 DeepSeek 的可行性。
   - 虽然小型蒸馏模型 (distilled models) 可以在本地运行，但性能受硬件规格限制较大，导致响应时间较慢。
- **即将推出的 GRPO 集成**：Unsloth 正在致力于为其模型集成 GRPO 训练功能，预计将增强微调 (fine-tuning) 流程。
   - 与现有方法相比，这种集成有望简化训练过程并提高效率。
- **比较不同模型版本**：讨论围绕模型蒸馏版本与其 R1 对应版本之间的差异展开，其中 R1 版本提供了更好的性能。
   - 对话强调了在修改现有模型时，需要大量的训练数据来保持准确性。
- **层和张量量化策略**：社区对确定哪些层需要量化以保持模型准确性的策略表现出兴趣。
   - 实施系统化的层量化方法可以实现对模型性能更精确的控制。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Marktechpost/status/1886874013303235064">来自 Marktechpost AI Research News ⚡ (@Marktechpost) 的推文</a>：使用 Unsloth 微调 Llama 3.2 3B Instruct 以生成 Python 代码：综合指南（包含 Colab Notebook）。在本教程中，我们将演练如何设置并对 Llama 3 执行微调...</li><li><a href="https://www.downloadmoreram.com/">DownloadMoreRAM.com - CloudRAM 2.0</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2411.07191">Large Language Models 中的超级权重</a>：最近的研究显示了一个令人惊讶的结果：一小部分大语言模型 (LLM) 参数离群值对模型的质量具有不成比例的重要性。LLM 包含数十亿个参数...</li><li><a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - 动态 4-bit 量化</a>：Unsloth 的动态 4-bit 量化选择性地避免量化某些参数。这在保持与 BnB 4bit 相似的 VRAM 使用量的同时，大大提高了准确性。</li><li><a href="https://huggingface.co/collections/unsloth/llama-32-66f46afde4ca573864321a22">Llama 3.2 - unsloth 集合</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">运行 DeepSeek-R1 动态 1.58-bit</a>：DeepSeek R-1 是最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit 动态 GGUF 版本。</li><li><a href="https://huggingface.co/unsloth/phi-4-GGUF">unsloth/phi-4-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF">unsloth/SmolLM2-135M-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF">unsloth/DeepSeek-R1-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/MaxHastings/Kolo/blob/main/scripts/train.py">Kolo/scripts/train.py at main · MaxHastings/Kolo</a>：使用现有最佳工具在本地微调和测试 LLM 的一站式商店。 - MaxHastings/Kolo</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF">unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://deepnewz.com/ai/stanford-s-s1-32b-model-outperforms-openai-s-o1-preview-27-on-aime24-math-using-bc4ff754">斯坦福大学的 s1-32B 模型在 AIME24 数学问题上使用 1,000 个多样化问题，表现优于 OpenAI 的 o1-Preview 27% | DeepNewz</a>：斯坦福大学的研究人员引入了一种名为简单测试时缩放 (s1) 的新方法，该方法增强了语言模型的推理性能。s1-32B 模型在 1,000 个多样化数据集上进行了微调...</li><li><a href="https://huggingface.co/blog/1_58_llm_extreme_quantization">将 LLM 微调至 1.58bit：极致量化变得简单</a>：未找到描述</li><li><a href="https://github.com/unslothai/llama.cpp">GitHub - unslothai/llama.cpp: C/C++ 中的 LLM 推理</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 unslothai/llama.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1336497280315953163)** (17 条消息🔥): 

> `日期的 Regex、上传 Jupyter Notebooks、LLM 模型分解、LLM 博客平台、GRPO 支持` 


- **日期格式的 Regex 挑战**：讨论了如何为特定的日期短语（如 **'quarta-feira da proxima semana'** 或 **'esse sabado'**）创建 Regex，重点关注多样化的语言输入。
   - 有人指出，Regex 在处理一致的日期格式（如 **dd/mm/yyyy** 或 **yyyy-mm-dd**）时效果更好。
- **Jupyter Notebook 上传困惑**：一名成员询问了在聊天中上传 **Jupyter Notebooks** 的可能性。
   - 未提供明确答复，暗示上传功能存在限制或约束。
- **LLM 模型权重重组问题**：一位用户在尝试在 PyTorch 神经网络中重组 **LLM 模型** 时，因错误导入层而感到沮丧，导致输出乱码。
   - 他们寻求帮助，以了解模型各部分的效率可以在何处得到提高。
- **选择 LLM 见解的博客平台**：对话中包含了关于 LLM 博客平台的建议，提到了 **Substack** 和 **Ghost**。
   - 另一位成员推荐 **GitHub Pages** 作为一个可靠的替代方案，以确保内容的持久性和可分享性。
- **对 GRPO 支持的期待与预期的瑕疵**：大家对即将发布的 **GRPO 支持** 表示兴奋，同时也承认存在一些现有问题。
   - 一位成员评论说，仍有一些 **kinks**（小瑕疵）需要解决，这表明实现可能需要一些时间。



**提及的链接**：<a href="https://colab.research.google.com/drive/1zBxmzMMHl9N1FMhkwpQ38Qi2341C0Uh9?usp=sharing">Google Colab</a>：未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1336460657930932375)** (94 条消息🔥🔥): 

> `Unsloth 模型使用指南、Oobagooba 中的 DeepSeek、训练配置建议、LLM 中的动态量化、使用 vLLM 和 SGLang 进行模型推理` 


- **Unsloth 模型使用指南**：成员们讨论了如何执行特定命令以使用 vLLM 和 SGLang 运行模型，强调了正确配置参数的必要性。
   - 特别是提到运行命令 'vllm serve casperhansen/mistral-small-24b-instruct-2501-awq -tp 2 --port 2222 --enforce-eager' 非常有效。
- **Oobagooba 中的 DeepSeek**：用户分享了在 Oobagooba 本地运行 DeepSeek 模型的经验，指出如果没有正确的模型权重配置，经常会出现挑战。
   - 建议了一些优化和变通方法，包括确保使用 '--enforce-eager' 标志以防止模型加载期间失败。
- **训练配置建议**：一位成员详细介绍了他们用于微调的训练配置，指出训练损失（training loss）不稳定，并征求改进建议。
   - 建议包括对最后几个 checkpoint 进行基准测试，并考虑调整，例如使用验证数据集。
- **LLM 中的动态量化**：新用户对如何有效利用动态量化表示困惑，特别是 Unsloth 的动态 4bit 模型。
   - 他们询问了当前硬件与这些模型的兼容性，以及是否需要转换 safetensor 文件才能运行。
- **使用 vLLM 和 SGLang 进行模型推理**：成员们讨论了在 vLLM 和 SGLang 中运行模型的偏好，强调了高效推理设置的重要性。
   - 几位用户指出，与之前的慢速方法相比，最新的配置可以提高模型推理的性能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit">unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-r1-on-your-own-loc">Unsloth 文档</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-r1-on-your-own-local-device">教程：如何在本地设备上运行 DeepSeek-R1 | Unsloth 文档</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/1561">[修复] 更多微调支持 · Issue #1561 · unslothai/unsloth</a>: 支持 Gemma 等模型的序列分类 Flex Attention；可变序列长度和自动取消填充/填充；Tool Calling 重构并合并 xformers, SDPA, flash-attn, flex-attention</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: 微调 Llama 3.3, DeepSeek-R1, Mistral, Phi-4 &amp; Gemma 2 LLM，速度提升 2-5 倍，显存占用减少 70%</a>: 微调 Llama 3.3, DeepSeek-R1, Mistral, Phi-4 &amp; Gemma 2 LLM，速度提升 2-5 倍，显存占用减少 70% - unslothai/unsloth</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama">教程：如何微调 Llama-3 并在 Ollama 中使用 | Unsloth 文档</a>: 在 Ollama 上本地运行自定义个人助手（类似 ChatGPT）的初学者指南</li><li><a href="https://gist.github.com/fullstackwebdev/5aa69712a30a93bff3b2daebaeb6776f">unsloth_tool_success2.py</a>: GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://gist.github.com/fullstackwebdev/d8c8d46d042828ffeedb0ac2b701b31d">tool_train.py</a>: GitHub Gist：即时分享代码、笔记和代码片段。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1336457490811846747)** (7 条消息): 

> `使用 Unsloth 进行 CPT，DeepSeek 模型版本，Qwen 的数学版本，让 AI 触手可及` 


- **CPT 模型表现出色**：**CPT with Unsloth** 模型在 **Perplexity** (PPL) 方面表现出显著改进，基础模型分数为 179.76 和 258.56，而用户模型分数则为 72.63 和 83.96。
   - *我真的很喜欢它！* 表达了对模型性能的兴奋。
- **DeepSeek 模型被认为过旧**：一名成员指出 **DeepSeek 模型** 非常陈旧，表明需要更新版本。
   - 随后讨论了该模型缺失的数学版本以及对有趣数据集的需求。
- **请求 Qwen 2.5 的数学版本**：一名成员询问是否可以将 **Qwen 2.5** 模型的 **数学版本** 添加到现有资源中。
   - 他们强调了为未来实现提供多样化且有趣的数据集的重要性。
- **数学模型的资源链接**：分享了一个关于 Hugging Face 上可用 **数学模型** 的资源链接，特别是针对 Qwen 变体。
   - 鼓励成员探索已经可用的 **数学版本**，并特别要求向下滚动查看更多详情。
- **关注 AI 的普及性**：成员们表达了一个共同目标，即让社区中的每个人都能更轻松地使用 **AI**。
   - 这种情绪强调了为增强 AI 技术的理解和可用性而进行的协作努力。



**提及的链接**：<a href="https://huggingface.co/unsloth?search_models=math">unsloth (Unsloth AI)</a>：未找到描述

  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1336794325979238410)** (2 条消息): 

> `Gemini 2.0 Flash, Windsurf Next Beta` 


- **Gemini 2.0 Flash 已在 Windsurf 上线**：在 Windsurf 上引入 **Gemini 2.0 Flash**，每条消息消耗 **0.25 用户提示词额度**，每次工具调用消耗 **0.25 flow 操作额度**。尽管其工具调用能力有限，但因其在编程咨询方面的 **极速** 和 **高效** 而受到关注。
   - 访问 [Gemini 2.0 Flash 公告](https://x.com/windsurf_ai/status/1887235006374035966) 了解更多详情。
- **Windsurf Next Beta 现已推出**：通过 [此处](https://codeium.com/windsurf/download-next) 下载并与稳定版并行安装，即可抢先体验 **Windsurf Next**。其目标是让用户探索软件开发 AI 中的创新功能和改进，尽管可能存在一些不完善之处。
   - 最低配置要求包括 Mac 的 **OS X Yosemite**、Linux 的 **Ubuntu 20.04** 以及 **Windows 10 (64-bit)**，更多详情请见 [此处](https://codeium.com/blog/windsurf-next)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/windsurf_ai/status/1887235006374035966">来自 Windsurf (@windsurf_ai) 的推文</a>：Gemini 2.0 Flash 现已在 Windsurf 中可用！根据我们的测试，Flash：⚡ 极速 💪 高效 - 仅消耗 0.25X 额度 🧠 工具调用能力有限，但非常适合咨询关于...</li><li><a href="https://codeium.com/windsurf/download-next">感谢下载 Windsurf 编辑器</a>：未来的编辑器，就在今天。Windsurf 编辑器是首款由 AI Agent 驱动的 IDE，让开发者保持专注。现已支持 Mac、Windows 和 Linux。</li><li><a href="https://codeium.com/blog/windsurf-next">Windsurf Next 发布</a>：介绍 Windsurf Next，这是我们 Windsurf 的可选预发布版本。
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1336433867405398046)** (34 messages🔥): 

> `额度与工具使用、Qodo 疑虑、Runic 开源框架、Codeium 插件问题、Windsurf 与 Codeium 对比` 


- **关于工具使用额度消耗的担忧**：一位用户对 **Claude** 在工具功能上浪费额度表示沮丧，因为它反复尝试错误的命令。另一位成员确认工具错误不应扣除额度，但过度使用 token 的问题依然存在。
   - *“流量很大，而且像疯了一样吞噬额度”* 是讨论效率低下问题的用户中的一种显著情绪。
- **对 Qodo 合法性的怀疑**：一位用户质疑 **Qodo**（原 Codium）是否可能是骗局，而其他用户则表示不愿使用它。这种信任缺失暗示了社区对新 AI 工具的担忧。
   - 回复建议在不清楚新平台的合法性和可靠性之前，对其采取谨慎态度。
- **Runic 框架介绍**：一位成员宣布推出 **Runic**，这是一个完全由 Python 生成的开源框架，旨在通过长期记忆 (LTM) 和检索增强生成 (RAG) 来增强 LLM。鼓励用户通过 `pip install --pre runic` 进行测试。
   - 正如其 [GitHub 页面](https://github.com/livingstonlarus/runic)所强调的，欢迎通过讨论区对测试功能提供反馈。
- **用户在使用 Codeium JetBrains 插件时遇到的问题**：多位用户报告了 **Codeium** JetBrains 插件的使用困扰，理由是反应迟钝且需要频繁重启才能正常工作。一位用户由于这些持续存在的问题已切换回 **Copilot**。
   - 为了提高稳定性和性能，一位成员敦促道：*“请给 Jetbrains 插件多一点关爱”*。
- **关于 Windsurf 与 Codeium 的讨论**：随着 **Windsurf** 的发布，一些用户注意到 **Codeium** JetBrains 插件的关注度和更新有所下降。讨论反映了用户对于转型以及对两个平台持续支持的复杂心情。
   - 这种情绪表明，随着用户社区偏好的转移，用户渴望在所有工具中保持功能的可用性。



**提及的链接**：<a href="https://github.com/livingstonlarus/runic">GitHub - livingstonlarus/runic: An open-source framework that enhances Large Language Models (LLMs) with Long-Term Memory (LTM) and Retrieval-Augmented Generation (RAG). Ideal for AI coding assistants and other applications, it enables LLMs to retain context, adapt over time, and access up to date information, ensuring more intelligent and context-aware interactions.</a>：一个开源框架，通过长期记忆 (LTM) 和检索增强生成 (RAG) 增强大语言模型 (LLM)。非常适合 AI 编程助手和其他应用，它使 LLM 能够保留上下文、随时间适应并访问最新信息，确保更智能和上下文感知的交互。

  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1336433344816087121)** (476 messages🔥🔥🔥): 

> `Windsurf 额度问题、Gemini 模型讨论、Windsurf 与 Cursor 功能对比、用户体验反馈、Windsurf 改进建议` 


- **Windsurf 额度问题**：用户在 Windsurf 的额度分配和使用方面遇到了问题，特别是 flex 额度，导致在关键工作期间功能无法使用。
   - 一位用户提到多次就额度问题联系支持部门，表现出对系统可靠性的沮丧。
- **Gemini 模型讨论**：Gemini 2.0 Flash 模型因其处理文件的速度而受到关注，但也伴随着对其可靠性和性能的担忧。
   - 用户表示有兴趣测试新模型，同时将其能力与 Claude 和 DeepSeek 等现有模型进行对比。
- **Windsurf 与 Cursor 功能对比**：讨论强调了 Cursor 相对于 Windsurf 的优势，特别是在安装第三方工具和扩展的灵活性方面。
   - 用户建议 Windsurf 可以从允许类似功能中受益，特别是在 IDE 内移动第三方应用方面。
- **用户体验反馈**：几位用户对 Windsurf 中 Cascade 的界面和功能表示担忧，指出有改进的必要。
   - 反馈建议更直观的 UI 和更好的大型代码库处理能力可以提升用户体验。
- **Windsurf 改进建议**：用户提出了诸如基于编码标准的自动 commit 消息以及更好的额度结转政策等功能，以增强整体可用性。
   - 这些建议旨在使 Windsurf 在与 Cursor 等现有 IDE 的竞争中更具优势。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/simonw/status/1887173498944335981?s=46">来自 Simon Willison (@simonw) 的推文</a>：“所以今天，我们推出了 2.0 Flash-Lite，这是一款比 1.5 Flash 质量更好、速度和成本却相同的模型。” 我认为价格是每百万 input tokens 7.5美分，每百万 output tokens 30美分...</li><li><a href="https://x.com/GoogleDeepMind/status/1887172472010653763">来自 Google DeepMind (@GoogleDeepMind) 的推文</a>：2.0 Pro Experimental 是我们迄今为止在 coding 和复杂 prompts 方面表现最好的模型，并根据您的反馈进行了改进。🤝 它对 world-knowledge 有更好的理解，并配备了我们迄今为止最大的 context window ...</li><li><a href="https://x.com/kevinhou22/status/1886827501004931511">来自 Kevin Hou (@kevinhou22) 的推文</a>：我们热爱文档！📖 我正在努力改进/添加更多 @ docs 快捷方式到 @windsurf_ai，告诉我你想要什么，我会尽可能多地添加... 🧵 另外向 @mintlify 致敬，感谢他们自动托管所有文档...</li><li><a href="https://hackathon.elevenlabs.io/">ElevenLabs 全球黑客松</a>：一个周末。六个城市及线上。AI agents。</li><li><a href="https://docs.codeium.com/windsurf/advanced">Windsurf - 高级</a>：未找到描述</li><li><a href="https://docs.codeium.com/windsurf/cascade">Windsurf - Cascade</a>：未找到描述</li><li><a href="https://codeium.com/changelog/windsurf-next">Windsurf Next 更新日志 | Windsurf 编辑器和 Codeium 扩展</a>：Windsurf Next 扩展的最新更新和变化。</li><li><a href="https://status.codeium.com">Codeium 状态</a>：未找到描述</li><li><a href="https://codeium.canny.io/feature-requests">功能请求 | Codeium</a>：向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://youtu.be/UocbxPjuyn4?list=TLPQMDMwMjIwMjWMu7hIVlgCAA">OpenAI o3-mini vs DeepSeek R1 (在 Cursor vs Windsurf 中)</a>：在快速发展的 AI 领域，最近有两款模型引起了广泛关注：OpenAI 的 o3-mini 和 DeepSeek 的 R1。两者都旨在增强...</li><li><a href="https://github.com/ZarK/ai-rules">GitHub - ZarK/ai-rules</a>：通过在 GitHub 上创建账号来为 ZarK/ai-rules 的开发做出贡献。</li><li><a href="https://codeium.canny.io/feature-requests/p/auto-commit-message">自动 commit message | 功能请求 | Codeium</a>：从已提交的文件上下文中生成 Commit Messages</li><li><a href="https://codeium.canny.io/feature-requests/p/roll-over-of-pro-credits">Pro Credits 滚动额度 | 功能请求 | Codeium</a>：未使用的 Premium 用户 Prompt Credits 和 Premium Flow Action Credits 将滚动到下个月</li><li><a href="https://www.reddit.com/r/Codeium/comments/1ihn6gp/submit_your_docs_suggestions_to_head_of_product/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1336798894218281130)** (1 条消息): 

> `介绍信息、社区参与计划、功能请求板、研究人员进度分享` 


- **Maxfield 介绍自己为首席社区官 (Chief Community Guy)**：Maxfield 是 Stability 新任首席社区官，他介绍了自己，并分享了自 2022 年 11 月以来长期参与 **Stable Diffusion** 的经历。
   - 他强调了改善社区参与的必要性，并表示致力于听取成员的反馈。
- **提升社区参与度的新举措**：Maxfield 宣布了两项旨在加强社区参与的举措，并指出 **“Stability 最近的参与度一直不尽如人意。”**
   - 他详细说明了实施功能请求板的计划，并增加 Stability 研究人员和创作者的透明度。
- **功能请求板即将推出**：其中一项举措包括一个**功能请求板**，社区成员可以在上面建议并投票选出理想的模型、工具或工作流。
   - Maxfield 断言：**“如果我们不开发你们想要的东西，那么所有这些算力的意义何在？”**
- **聚焦研究人员项目**：Maxfield 表示，社区成员很快将看到更多旨在展示 Stability 优秀研究人员和创作者进展的举措。
   - 他指出：**“这里的人们正在做一些非常疯狂的东西，我们不应该是唯一看到它的人。”**


  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1336438752431570974)** (399 messages🔥🔥): 

> `Latency and Model Compatibility, Training and Fine-Tuning Challenges, Upcoming AI Models and Architectures, Community Tools for AI, Spamming in Discord Channels` 


- **探索 Latent Space 兼容性**：讨论强调了在一个 Diffusion 模型的 Latent Space 内运行另一个模型的潜力，揭示了引人入胜的嵌套架构。
   - 参与者提到，为了使其正常运行，使用兼容的 VAE 至关重要，而其他人则对探索这一概念的底层论文表示感兴趣。
- **模型训练中的挑战**：用户分享了微调 LoRA 等模型的经验，指出默认设置通常优于复杂的调整。
   - 架构演进的复杂性使得一些用户渴望有更精简且用户友好的工具，以便在 Latent Space 中高效工作。
- **AI 模型即将到来的创新**：社区对未来的 Multimodal 模型进行了推测，对融合文本和图像生成能力的工具表示期待。
   - 显然，人们对开发能够通过直观界面增强创意用途（如视频游戏开发）的新模型有着浓厚兴趣。
- **用户创建的社区参与工具**：一位用户正在开发一种用于在 Latent Space 内编辑 Tensor 文件的新工具，强调了现有工具需要更好的用户体验。
   - 讨论了创建一个用于游戏设计和故事板的精简 3D GUI 的潜力，突显了用户在推进 AI 应用方面的主动性。
- **管理与垃圾信息问题**：聊天中出现了垃圾信息事件，促使用户举报该消息并要求管理人员采取行动。
   - 社区展示了在无关推广帖子中维护频道完整性的集体努力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/creep-hands-adventure-time-deer-remove-the-gloves-gif-15274634">Creep Hands GIF - Creep Hands Adventure Time - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://lu.ma/fyu8iqnk">Multimodal AI Agents - Hackathon · Luma</a>：Gen AI AgentsCreatorsCorner，与 Google Deepmind, Weights &amp; Biases, Together.ai, Stytch, Senso, LlamaIndex 等合作，充满热情地……</li><li><a href="https://github.com/NeuralNotW0rk/LoRAW">GitHub - NeuralNotW0rk/LoRAW: 用于 stable-audio-tools 的灵活 LoRA 实现</a>：用于 stable-audio-tools 的灵活 LoRA 实现 - NeuralNotW0rk/LoRAW</li><li><a href="https://purplesmart.ai/">扩展 AI 创造力的边界 - PurpleSmartAI</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1336440201634910351)** (365 messages🔥🔥): 

> `Cursor IDE Features, MCP Server Integration, Gemini 2.0 Pro Model, Voice Dictation in Coding, Mobile IDE Usability`

- **Cursor IDE 引入 MCP Server 支持**：用户讨论了 Cursor 中新实现的 MCP Server 集成，该功能增强了扩展性，包括通过命令使用 Perplexity 提供协助。
   - 该集成允许通过提供的 GitHub 仓库轻松设置，并让用户可以选择直接提问或与工具进行对话。
- **Gemini 2.0 Pro 面临批评**：一些用户对新的 Gemini 2.0 Pro 模型的性能表示失望，称其虽然在数据分析方面表现出色，但在处理编程任务时却很吃力。
   - 基准测试表明，虽然 Gemini 2.0 Pro 在随机任务中表现尚可，但在编程方面仍落后于 O3-mini high 和 Sonnet。
- **编程语音听写投入使用**：讨论者分享了关于高效语音听写工具的见解，并提到了 Andrej Karpathy 使用 Whisper 技术听写代码的方法。
   - 提到了 Windows 内置的听写功能，尽管用户指出其准确性仍有待提高，这引发了对为编程目的定制语音接口的兴趣。
- **Cursor IDE 的移动端可用性受到质疑**：一位用户建议为 Cursor 开发 iPhone 应用，以便随时随地进行编程和 Prompt 编写，而其他用户则反思了这一功能的实用性和需求。
   - 尽管该想法得到了一些支持，但普遍共识认为，目前的框架可能不足以支撑将 Cursor 转化为移动应用的投入。
- **Cursor 的用户体验增强**：几位用户在尝试了 Cursor 的各种功能（特别是与使用 MCP Server 相关的功能）后，报告了成功的集成和改进。
   - 对话强调了用户对功能请求和故障排除的高度参与，反映了对持续改进的积极兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/thekitze/status/1886809191395836008">来自 kitze 🚀 (@thekitze) 的推文</a>：Cursor > Windsurf（目前而言），我会继续在两者之间切换并进一步报告。</li><li><a href="https://x.com/lmarena_ai/status/1887180371219132898">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：新闻：@GoogleDeepMind Gemini-2.0 系列（Pro, Flash, 和 Flash-lite）现已在 Arena 上线！- Gemini-2.0-Pro 在所有类别中排名第一 - Gemini-2.0-Flash 排名第三，现已向开发者广泛开放...</li><li><a href="https://pypi.org/project/mcp-perplexity/">未找到标题</a>：未找到描述</li><li><a href="https://aistudio.google.com/">Google AI Studio</a>：Google AI Studio 是开始使用 Gemini（我们下一代多模态生成式 AI 模型系列）进行构建的最快方式。</li><li><a href="https://tenor.com/view/todd-tar-scare-scared-flashing-light-gif-25292274">Todd Tar GIF - Todd Tar Scare - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://glama.ai/mcp/servers">开源 MCP 服务器</a>：企业级安全性、隐私性，具备 Agent、MCP、Prompt 模板等功能。</li><li><a href="https://github.com/grapeot/devin.cursorrules/tree/multi-agent">GitHub - grapeot/devin.cursorrules (multi-agent 分支)</a>：让 Cursor/Windsurf 达到 Devin 90% 能力的魔法。通过在 GitHub 上创建账号来为 grapeot/devin.cursorrules 的开发做出贡献。</li><li><a href="https://github.com/danie">Danie - 概览</a>：Danie 有 10 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://www.youtube.com/watch?v=MAicJ6KKccU">20 分钟内通过 MCP 构建增强版 Cursor</a>：Cursor 在没有大肆宣传的情况下悄悄发布了 MCP 支持，所以很容易错过。但它实际上是一个杀手级功能，可以加速你的 Composer Agent...</li><li><a href="https://github.com/daniel-lxs/mcp-starter/releases/">发布版本 · daniel-lxs/mcp-starter</a>：通过在 GitHub 上创建账号来为 daniel-lxs/mcp-starter 的开发做出贡献。</li><li><a href="https://huggingface.co/blog/prithivMLmods/o3-mini-vs-deepseek-r1">o3-mini vs Deepseek-R1</a>：未找到描述</li><li><a href="https://github.com/PatrickJS/awesome-cursorrules?tab=readme-ov-file">GitHub - PatrickJS/awesome-cursorrules: 📄 精选的 .cursorrules 文件列表</a>：📄 精选的 .cursorrules 文件列表。通过在 GitHub 上创建账号来为 PatrickJS/awesome-cursorrules 的开发做出贡献。</li><li><a href="https://github.com/daniel-lxs/mcp-starter/releases/tag/v0.1.0">发布版本 v0.1.0 · daniel-lxs/mcp-starter</a>：初始发布</li><li><a href="https://github.com/daniel-lxs/mcp-perplexity">GitHub - daniel-lxs/mcp-perplexity</a>：通过在 GitHub 上创建账号来为 daniel-lxs/mcp-perplexity 的开发做出贡献。</li><li><a href="https://svelte-llm.khromov.se/">svelte-llm - 适用于 LLM 格式的 Svelte 5 和 SvelteKit 开发者文档</a>：未找到描述</li><li><a href="https://github.com/daniel-lxs/mcp-server-perplexity">GitHub - daniel-lxs/mcp-perplexity</a>：通过在 GitHub 上创建账号来为 daniel-lxs/mcp-perplexity 的开发做出贡献。</li><li><a href="https://github.com/daniel-lxs/mcp-starter">GitHub - daniel-lxs/mcp-starter</a>：通过在 GitHub 上创建账号来为 daniel-lxs/mcp-starter 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1336426140494991421)** (312 条消息🔥🔥): 

> `Perplexity AI 的变更、Gemini 2.0 Flash 发布、模型访问问题、Pro 订阅困惑、反馈与用户体验` 


- **用户对 Perplexity AI 的 UI 变更感到沮丧**：许多用户对 Perplexity AI 最近的变更表示不满，特别是 Focus 模式的移除和性能下降，将其比作“平台劣化”（enshitification）。
   - 部分用户在访问 Claude 3.5 Sonnet 等模型时遇到困难，在 Pro Search 模式下会自动激活 R1 或 o3-mini。
- **Gemini 2.0 Flash 正式发布**：宣布 Gemini 2.0 Flash 面向所有 Pro 用户开放，这是自早期版本以来首次将 Gemini 模型引入 Perplexity。
   - 用户渴望了解其与之前模型在 context limits 和功能方面的差异，并注意到它在当前应用界面中的功能和可用性。
- **对模型访问和限制的困惑**：Pro 用户报告了模型访问的不一致性，尽管拥有 Pro 订阅，一些人仍无法使用 Gemini 或访问所需的模型。
   - 用户体验存在差异，一些用户认为这些限制是不必要的，而另一些用户仍在适应跨平台的新功能。
- **对移除功能的决策不满**：移除传统上用于特定搜索的 Focus 模式等功能引发了投诉，认为这些行为类似于流行科技产品中考虑不周的变更。
   - 用户讨论强调了功能上的退步，这令人失望，因为他们之前依赖这些功能进行学术研究。
- **用户对 Perplexity 的体验和期望**：一般性讨论展示了多样的用户体验，从对 context limits 的沮丧到在原生应用或 API 中成功利用功能的用户。
   - 用户分享了他们对集成 API 的担忧和准备工作，而其他人则提到了他们获得的性能速度和输出，并寻求改进。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/perplexity_ai/status/1887186081029738526?s=61">来自 Perplexity (@perplexity_ai) 的推文</a>：本周日，你的问题可能价值 1,000,000 美元。</li><li><a href="https://x.com/aravsrinivas/status/1887174442171969936?s=61">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：我们将向所有 Perplexity Pro 用户开放 Gemini 2.0 Flash。这是我们首次将 Gemini 模型引入 Perplexity。Flash 2.0 是一款令人惊叹的 multimodal 且极具成本效益的模型。更新...</li><li><a href="https://hika.fyi/">Hika AI - 免费 AI 知识搜索</a>：提供高级见解和交互式探索的免费 AI 搜索。</li><li><a href="https://tenor.com/view/dr-evil-gold-member-one-million-dollars-gif-5706036">Dr Evil Gold Member GIF - Dr Evil Gold Member One Million Dollars - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/shy-dog-dog-shy-dog-shoes-martian-shy-gif-10611534617383883284">Shy Dog Shy Dog Shoes GIF - 害羞的狗 穿鞋的狗 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://status.perplexity.com/">Perplexity - 状态</a>：Perplexity 运行状态
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1336483653697536134)** (11 messages🔥): 

> `美国 Iron Dome 提案, 加州独立提案, 小行星与生命, 量子力学与意识, 电力类型` 


- **特朗普提议建立美国 Iron Dome**：一位成员分享了一段有趣的[视频](https://www.youtube.com/embed/Nd5zGnxX_kU)，讨论了在包括加州独立提案在内的持续政治发展背景下，特朗普关于建立美国 Iron Dome 的提议。
   - 讨论指向了此类军事战略对国家安全和地方治理的潜在影响。
- **加州寻求独立**：对话强调了正在进行的**加州**独立运动，并与美国 Iron Dome 等提议一同被提及。
   - 社区成员对该决定的可行性和后果表达了不同看法。
- **携带生命种子的多颗小行星**：参与者讨论了某些小行星可能携带**生命种子**的说法，认为持续的太空探索对于理解我们的起源至关重要。
   - 这引发了关于天体生物学和外星生命形式潜力的发人深省的对话。
- **量子力学入门引发讨论**：对 [量子力学](https://www.perplexity.ai/search/is-quantum-computing-more-sens-Df_s24s_ToGnlkT0HbX9FQ) 的持续探索转向了哲学话题，特别是我们对时间的理解。
   - 成员们就**量子理论**与**意识**的交集发表了各自的观点，形成了一场引人入胜的对话。
- **探索不同类型的电力**：围绕各种类型的**电力**展开了讨论，并分享了详细阐述该主题的资源链接，包括 [电力常识](https://www.perplexity.ai/search/what-types-of-electricity-gene-CNUr_pczQBqvfUS8jW2UcA)。
   - 成员们强调了在当今技术驱动的世界中理解电力系统的重要性。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1336451379161600080)** (5 messages): 

> `Sonar Reasoning Pro, Perplexity API 成本管理, Perplexity API 中的图像上传` 


- **Sonar Reasoning Pro 使用 DeepSeek R1**：一位成员澄清说，**Sonar Reasoning Pro** 运行在 **DeepSeek R1** 之上，正如其 [官方网站](https://sonar.perplexity.ai/) 所确认的那样。
   - *这让一些此前不了解底层模型的成员恍然大悟。*
- **提出成本管理问题**：一位用户询问如何设置每月费用的硬限制（hard limit），以及发票是自动发送还是需要手动添加。
   - 这一担忧反映了用户对管理使用 **Perplexity API** 相关费用的兴趣日益增加。
- **探索使用 Perplexity API 上传图像**：一位新用户在成功尝试了包含图像上传的 webUI 后，表达了对使用 **Perplexity API** 的兴趣。
   - 他们提出了关于目前图像上传限制的问题，并推测了使用其他工具的潜在**变通解决方案（workaround solutions）**。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1336439002718273598)** (114 条消息🔥🔥): 

> `关于 ML 理论与凸优化（Convex Optimization）、Harmonic Loss 与 Cross-Entropy Loss、机器学习背景与协作、Diffusion Models 见解以及 ML 统计背景挑战的讨论` 


- **ML 理论与优化的困境**：成员们讨论了**凸优化（Convex Optimization）**在实践中的挑战，指出由于数据和环境的粗糙性，它在现实场景中往往效果不佳。
   - 有人将凸优化的理想化性质比作在平坦路面上建造一辆完美的自行车，而现实世界的应用则面临着严重的道路颠簸。
- **引入 Harmonic Loss 作为替代方案**：分享了一篇新论文，介绍 **Harmonic Loss** 作为神经网络标准 Cross-Entropy Loss 的优选替代方案，声称其具有提高可解释性和加快收敛速度等优点。
   - 社区成员对其新颖性表示怀疑，指出其与传统方法有相似之处，而其他人则强调了它改变优化目标的潜力。
- **对 ML 项目协作的兴趣**：一位成员表达了在 **LLM Agent** 相关研究项目上进行协作的愿望，寻求与其他研究人员和 ML 爱好者建立联系，以获得头脑风暴和发表论文的机会。
   - 他们强调目标是研究该领域的新颖问题，并鼓励任何对团队合作或讨论感兴趣的人与其联系。
- **关于 Diffusion Models 的思考**：讨论了 **Diffusion Models** 与其他 ML 子学科相比在理论强度上的表现，并就该领域理论与实际应用之间的平衡发表了看法。
   - 成员们指出，虽然理论工作可能看起来很抽象，但它引领了图像生成领域的重大进步，并具有实际相关性。
- **ML 中的统计学基础**：许多人表示希望拥有更扎实的**统计学（Statistics）**基础知识作为理解 ML 的基石，并回顾了他们过去侧重于编程和离散数学的课程经历。
   - 对话强调了 ML 从业者在统计素养方面存在的差距，并强调了统计学在理解研究和方法论中的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/dbaek__/status/1886781418115862544">来自 David D. Baek (@dbaek__) 的推文</a>: 1/9 🚨 新论文发布：你需要的不是 Cross-Entropy Loss！🚨我们引入了 Harmonic Loss 作为训练神经网络和 LLM 时标准 CE Loss 的替代方案！Harmonic Loss 实现了 🛠️si...</li><li><a href="https://x.com/giffmana/status/1886897912740761674">来自 Lucas Beyer (bl16) (@giffmana)</a>: 我简要浏览了 Harmonic Loss 论文。太长不看版：不要使用带有 Softmax 的点积，而是使用归一化的 1/d**n 的欧几里得距离。我挺希望这能奏效。我曾尝试过偏向欧几里得距离的方法...</li><li><a href="https://www.alignment.org/blog/backdoors-as-an-analogy-for-deceptive-alignment/">后门作为欺骗性对齐的类比</a>: ARC 发布了一篇关于后门防御、可学习性和混淆的论文，我们在其中研究了 ML 模型中后门的正式概念。我们的部分动机是后门之间的类比...</li><li><a href="https://arxiv.org/abs/2402.03300">DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a>: 由于数学推理的复杂性和结构化性质，它对语言模型构成了重大挑战。在本文中，我们介绍了 DeepSeekMath 7B，它在 DeepSeek-Co 的基础上继续预训练...</li><li><a href="https://arxiv.org/abs/2405.20304">Group Robust Preference Optimization in Reward-free RLHF</a>: 使大语言模型（LLM）适应特定任务通常涉及通过人类反馈强化学习（RLHF）对偏好数据进行微调。虽然这些数据通常来自不同的...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1336448627844845650)** (210 条消息🔥🔥): 

> `Harmonic loss 作为 CE loss 的替代方案，用于运动生成的 VideoJAM 框架，神经网络中的激活函数，各种优化器技术的评估，改进的 ReLU 方法` 


- **探索 Harmonic Loss 优于 Cross-Entropy**: 一位成员强调了在神经网络中使用 Harmonic Loss 代替传统 **cross-entropy loss** 的潜在好处，声称它能带来更好的可解释性和更快的收敛速度。
   - 讨论涉及了模型训练期间激活的稳定性以及与各种学习参数之间的相互作用。
- **引入 VideoJAM 以增强运动生成**: Hila Chefer 介绍了 **VideoJAM 框架**，强调了它如何在不需要额外数据的情况下，直接解决视频生成器在运动表示方面面临的挑战。
   - 该框架有望改善视频内容中运动生成的动态效果。
- **研究稀疏激活函数**: 成员们讨论了 **ReLU squared** 的性能，并探索了替代方案，如 **GeGLU** 以及基于 L2 或软阈值方法的潜在新函数。
   - 讨论中考虑了这些激活函数在训练期间如何影响模型容量和稀疏性。
- **优化器与梯度稳定性**: 针对各种优化器（包括 AdamW 以及重新设计的二阶方法如 **pSGD 和 Shampoo**）在管理梯度峰值和提高学习稳定性方面的效率进行了辩论。
   - 参与者鼓励探索新兴优化器，以增强性能并缓解与现有技术相关的稳定性问题。
- **改进版 ReLU 实现的性能评估**: 成员们分享了不同 ReLU 函数实现的经验，指出 **ReLU squared** 在某些语境下（特别是 speedrun 场景）表现出性能优势。
   - 对话还包括关于应用 leaky ReLU 的建议，以及在应用 cross-entropy loss 之前保持非负输出的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fixupx.com/ChrSzegedy/status/1886881600367161679">来自 Christian Szegedy (@ChrSzegedy) 的推文</a>: 我非常感兴趣看到调和加权注意力（harmonically weighted attention）将如何工作。那里可能存在真正的潜力。引用 David D. Baek (@dbaek__) 1/9 🚨 新论文预警：Cross-Entropy Loss...</li><li><a href="https://fixupx.com/hila_chefer/status/1886791105380851917">来自 Hila Chefer (@hila_chefer) 的推文</a>: VideoJAM 是我们来自 @AIatMeta 的用于改进运动生成的新框架。我们展示了视频生成器在运动方面挣扎的原因是训练目标偏向外观而非动态。VideoJAM di...</li><li><a href="https://arxiv.org/abs/2411.13010">通过积分推导激活函数</a>: 我们的工作提出了一种设计激活函数的新方法，重点关注其梯度并通过积分推导相应的激活函数。我们引入了 Expanded Int...</li><li><a href="https://hila-chefer.github.io/videojam-paper.github.io/">VideoJAM</a>: VideoJAM: 用于增强视频模型运动生成的联合外观-运动表示</li><li><a href="https://arxiv.org/abs/2502.01612">自我改进的 Transformer 克服了从易到难以及长度泛化的挑战</a>: 大型语言模型（LLM）通常在长度泛化和解决超出其训练分布的复杂问题实例方面面临困难。我们提出了一种自我改进方法，模型迭代地...</li><li><a href="https://www.physicalintelligence.company/blog/pi0">我们的第一个通用策略</a>: Physical Intelligence 正在将通用 AI 带入物理世界。</li><li><a href="https://github.com/Physical-Intelligence/openpi">GitHub - Physical-Intelligence/openpi</a>: 通过在 GitHub 上创建账户，为 Physical-Intelligence/openpi 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1336438248750448761)** (2 messages): 

> `有效 Batch Size 策略，训练中的 Weight Decay 应用` 


- **GAS 相比 Activation Checkpointing 提升了 TPS**：不使用 **activation checkpointing** 而使用 **GAS** 进行训练显著提高了 TPS。对比显示，Batch Size 为 8 且使用 GAS 时为 **242K TPS**，而 Batch Size 为 48 且使用 Checkpointing 时为 **202K TPS**。
   - 尽管有效 Batch Size 不同，使用 GAS 的较小 Batch Size 显示出更快的收敛速度。虽然观察到 **较低的 HFU/MFU** 可能是一个令人担忧的问题，但如果 TPS 得到提升，则不会被优先考虑。
- **OLMo2 的 Weight Decay 方法**：有人提问关于不将 **weight decay** 应用于模型的特定部分（特别是 **embedding layer**），这是根据 OLMo2 论文的方法论提出的。
   - 这引发了关于训练期间选择性 Weight Decay 的影响及其对模型性能潜在影响的讨论。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1336775685405937716)** (2 messages): 

> `DeepSeek R1 Nitro，宕机事件` 


- **DeepSeek R1 Nitro 拥有极速的运行时间**：根据 [OpenRouterAI](https://x.com/OpenRouterAI/status/1887212200647139731) 的数据，**DeepSeek R1 Nitro** 的运行时间和速度显著提高，目前 **97% 的请求** 都能完整完成并带有 finish reason。
   - 鼓励用户 **尝试使用** 以获得最佳性能！
- **小故障导致短暂宕机**：一名成员承认了一个导致宕机的 **小故障**，但报告称在回滚后，服务现在应该已经重新上线。
   - 对造成的不便表示歉意，并向用户保证一切已恢复正常运行。



**提到的链接**：<a href="https://x.com/OpenRouterAI/status/1887212200647139731">来自 OpenRouter (@OpenRouterAI) 的推文</a>：我们的 DeepSeek R1 Nitro 端点在运行时间和速度上有显著提升。看到 97% 的请求现在可以 *完整* 完成并带有 finish reason。快来试试吧！👇

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1336426693677551686)** (298 messages🔥🔥): 

> `OpenRouter 宕机，API 错误和速率限制，Gemini 2.0 更新，供应商路由和定价，社区支持和故障排除` 


- **OpenRouter 经历宕机**：用户注意到了 API 问题和速率限制错误，引发了对持续服务可靠性的担忧，Toven 确认可能会有宕机。
   - 宕机问题很快得到解决，Toven 宣布在撤销最近的更改后，服务立即恢复。
- **速率限制影响 API 调用**：一些用户（如 Tusharmath）在使用 API 时遇到了速率限制错误，特别是 Anthropic，其限制为每分钟 2000 万个输入 tokens。
   - Louisgv 提到正在联系 Anthropic，寻求潜在的速率限制提升，以解决用户遇到的限制问题。
- **对 Gemini 2.0 性能的关注**：Xiaoqianwx 讨论了对 Gemini 2.0 的预期，认为它可能需要更强大的模型才能有效竞争，同时指出其 Benchmark 结果平平。
   - 社区对性能表示失望，并讨论了该模型的相对优缺点。
- **供应商路由和定价问题**：用户询问了关于 API 使用的潜在价格控制，特别是针对不同供应商之间成本差异的问题。
   - Toven 引入了一个新的 max_price 参数，用于控制 API 调用的支出限制，该参数目前已上线但尚未完全记录在文档中。
- **社区故障排除和支持**：成员们利用该频道分享错误信息并寻求帮助，展示了解决问题的协作精神。
   - Toven 鼓励用户如果他们的支持工单（help tickets）没有得到及时处理，可以多次联系，并强调了支持服务的可用性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://livebench.ai/#/">LiveBench</a>: 未找到描述</li><li><a href="https://x.com/OpenAIDevs/status/1886917557896036724">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>: @FeltSteam @UserMac29056 @TheXeophon @chatgpt21 API 中的 chatgpt-4o-latest 现已更新，与上周 ChatGPT 中的 GPT-4o 更新保持一致。抱歉延迟了！https://help.openai.c...</li><li><a href="https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens">管理您的个人访问令牌 - GitHub Docs</a>: 未找到描述</li><li><a href="https://models.inference.ai.azure.com",">未找到标题</a>: 未找到描述</li><li><a href="https://openrouter.ai/settings/integrations).">OpenRouter</a>: LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/docs/features/provider-routing">提供商路由 — OpenRouter | 文档</a>: 将请求路由至最佳提供商</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/tokenizer_config.json">tokenizer_config.json · deepseek-ai/DeepSeek-R1 at main</a>: 未找到描述</li><li><a href="https://ai.google.dev/gemini-api/docs/caching?lang=python">未找到标题</a>: 未找到描述</li><li><a href="https://openrouter.ai/docs/api-reference/list-endpoints-for-a-model">列出模型的端点 — OpenRouter | 文档</a>: 未找到描述</li><li><a href="https://openrouter.ai/credits,">OpenRouter</a>: LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/settings/keys.">OpenRouter</a>: LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/docs/features/provider-routing#json-schema-for-provider-preferences">提供商路由 — OpenRouter | 文档</a>: 将请求路由至最佳提供商</li><li><a href="https://ca.finance.yahoo.com/news/deepseek-users-could-face-million-104352160.html">根据新法律，美国的 DeepSeek 用户可能面临百万美元罚款和监禁</a>: 这款极受欢迎的中国 AI 应用引发了安全、隐私和伦理方面的担忧</li><li><a href="https://openrouter.ai/gryphe/mythomax-l2-13b/providers">MythoMax 13B – 提供商状态</a>: 查看提供商状态并对 MythoMax 13B 发起负载均衡请求 - 这是 Llama 2 13B 性能最高且最受欢迎的微调版本之一，具有丰富的描述和角色扮演能力。#merge</li><li><a href="https://huggingface.co/docs/transformers/en/chat_templating">聊天模板 (Chat Templates)</a>: 未找到描述</li><li><a href="https://openrouter.ai/google/gemini-2.0-flash-001">Gemini Flash 2.0 - API、提供商、统计数据</a>: 与 Gemini Flash 1 相比，Gemini Flash 2.0 的首个 Token 时间 (TTFT) 显著加快。通过 API 运行 Gemini Flash 2.0</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/pricing">未找到标题</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1iih0mm/is_it_susy_baka_that_cusor_and_windsurf_dont_show/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://github.com/ShivamB25/Research-Analysist">GitHub - ShivamB25/Research-Analysist: 自动化网页抓取和报告生成过程</a>: 自动化网页抓取和报告生成过程 - ShivamB25/Research-Analysist</li><li><a href="https://github.com/OpenRouterTeam/ai-sdk-provider">GitHub - OpenRouterTeam/ai-sdk-provider: 适用于 Vercel AI SDK 的 OpenRouter 提供商，通过 OpenRouter 聊天和补全 API 支持数百种 AI 模型。</a>: 适用于 Vercel AI SDK 的 OpenRouter 提供商，通过 OpenRouter 聊天和补全 API 支持数百种 AI 模型。 - OpenRouterTeam/ai-sdk-provider
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1336439608459661342)** (156 条消息🔥🔥): 

> `Gemini 2.0 更新，Mistral 的新产品，AI 基准测试性能，GitHub Copilot 更新`

- **Gemini 2.0 Pro Experimental 性能**：最近对 **Gemini 2.0 Pro Experimental** 的基准测试显示，它在各项任务中的表现与 **Claude 3.5 Sonnet** 相当，但在 API 响应方面存在一些不一致性。
   - 虽然 **Flash** 模型优于之前的 **Gemini 1.5 Pro**，但长上下文（long context）能力似乎有所减弱，这引发了人们对该模型有效性的质疑。
- **Mistral 的新网站和功能**：Mistral 推出了重新设计的网站，展示了其旨在定制 AI 解决方案的开源模型，强调透明度和企业级部署选项。
   - 该公司通过新的猫咪 Logo 进行了品牌重塑，在领先的独立 AI 实验室形象与俏皮的设计风格之间取得了平衡。
- **AI 基准测试洞察**：讨论强调 **Gemini 2.0 Flash** 模型在内部基准测试中表现普遍良好，但在与竞争对手的直接对比中结果参差不齐。
   - 关于该模型上下文长度能力的持续评论引发了用户对潜在性能权衡的担忧。
- **GitHub Copilot 使用 Gemini 2.0**：GitHub 宣布 **Gemini 2.0 Flash** 现已面向所有 Copilot 用户开放，作为开发者的新工具集成到其模型选择器和 Copilot Chat 功能中。
   - 这一集成标志着 Gemini 的一个重要里程碑，因为它在其他竞争对手发布之前就在 Microsoft 的生态系统中获得了关注。
- **对模型命名和更新的批评**：用户对 AI 模型的命名惯例表示沮丧，认为其缺乏创意和清晰度，并以最新模型为例。
   - 普遍情绪认为，尽管 AI 取得了进步，但像 Google 这样的公司在向用户有效传达更新方面仍面临困难。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/terryyuezhuo/status/1887212831780770218">来自 Terry Yue Zhuo (@terryyuezhuo) 的推文</a>：BigCodeBench-HardTLDR 上的新 Gemini 模型（由于 API 敏感度，部分结果为空，所有安全设置均设为 NONE）：- Gemini-2.0-Pro-Exp-02-05：~~Claude 3.5 Sonnet le...</li><li><a href="https://x.com/GoogleDeepMind/status/1887172464863506547">来自 Google DeepMind (@GoogleDeepMind) 的推文</a>：Gemini 2.0 现已向所有人开放。✨⚡ 开始在 @Google AI Studio、@GoogleCloud 的 #VertexAI 和 @GeminiApp 中使用更新后的 2.0 Flash。我们还推出了：🔵 2.0 Pro Experimental，它在...</li><li><a href="https://x.com/legit_rumors/status/1887168398276092364">来自 ʟᴇɢɪᴛ (@legit_rumors) 的推文</a>：Vertex 平台拥有真正的 Gemini 2.0 Pro。网页版 / App / AI Studio 中的 2.0 Pro 仍在推出中——如果他们仍打算今天发布，可能还需要几个小时。</li><li><a href="https://x.com/giffmana/status/1886897912740761674">来自 Lucas Beyer (bl16) (@giffmana) 的推文</a>：我简要看了下 Harmonic Loss 论文。TL;DR：不使用带 Softmax 的点积，而是使用归一化的 1/d**n 欧几里得距离。我挺希望这行得通。我曾多次尝试偏向欧几里得距离...</li><li><a href="https://x.com/legit_rumors/status/1887141216677933305">来自 ʟᴇɢɪᴛ (@legit_rumors) 的推文</a>：Gemini 2.0 Pro Experimental 已发布</li><li><a href="https://x.com/TheXeophon/status/1887206775759229113">来自 Xeophon (@TheXeophon) 的推文</a>：这是对比。引用 Paul Calcraft (@paul_cal) @GoogleDeepMind @Google @googlecloud @GeminiApp：与竞争对手模型的对比在哪里？</li><li><a href="https://x.com/kellerjordan0/status/1886887139855810777">来自 Keller Jordan (@kellerjordan0) 的推文</a>：不幸的是，在 2025 年很难相信“主张”。更容易相信的是“激励”。所以这里有一个激励：我将向第一个使用这种方法改进...的人支付 3,000 美元的赏金。</li><li><a href="https://fxtwitter.com/arankomatsuzaki/status/1887211023423431134">来自 Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：关于 Gemini 2.0 更新的几点注意事项：- 在他们的基准测试中，Flash 和 Pro 之间的整体性能差距似乎非常小。-> Flash 非常出色。Pro 在长尾知识方面表现优异，这对于...非常重要。</li><li><a href="https://x.com/btibor91/status/1886880680077906376?s=61">来自 Tibor Blaho (@btibor91) 的推文</a>：OpenAI 网站现已根据新的设计指南进行了更新。引用 nic (@nicdunz)：OpenAI 官方网站已更新，采用了新的设计指南和其他内容。</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-updates-february-2025/">Gemini 2.0 现已向所有人开放</a>：我们宣布了 Gemini 2.0 Flash 的新更新，并推出了 Gemini 2.0 Flash-Lite 和 Gemini 2.0 Pro Experimental。</li><li><a href="https://fortune.com/2025/02/05/prior-labs-9-million-euro-preseed-funding-tabular-data-ai/">AI 一直在处理电子表格和表格方面表现挣扎。一家德国初创公司的突破可能会改变这一现状</a>：Prior Labs 刚刚获得了 930 万美元的“种子前轮”融资，用于构建其针对“表格数据”的基础模型。</li><li><a href="https://x.com/github/status/1887208954704355350">来自 GitHub (@github) 的推文</a>：🎁 今日上线：面向 *所有* GitHub Copilot 用户的 Gemini 2.0 Flash！在 @code 的模型选择器和 GitHub 的 Copilot Chat 中即可找到。https://gh.io/copilot-chat-gemini</li><li><a href="https://x.com/TheXeophon/status/1887171298868019708">来自 Xeophon (@TheXeophon) 的推文</a>：我要对负责“gemini-2.0-flash-lite-preview-02-05”这个名字的人动手（在《我的世界》里）。引用 ʟᴇɢɪᴛ (@legit_rumors)：Vertex 平台拥有真正的 Gemini 2.0 Pro。网页版 / App /... 中的 2.0 Pro...</li><li><a href="https://x.com/OpenAI/status/1887143439097352219">来自 OpenAI (@OpenAI) 的推文</a>：Deep Research 更新 🌍 现已向 100% 的 Pro 用户推出，包括英国、欧盟、挪威、冰岛、列支敦士登和瑞士。</li><li><a href="https://tenor.com/view/waltergotme-gif-18867690">Waltergotme GIF - Waltergotme - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://youtu.be/k3d_xeVxEOE">焕然一新。</a>：访问 https://openai.com/ 了解更多。</li><li><a href="https://mistral.ai/en">Mistral AI | 掌控前沿 AI</a>：通过从平台到界面的完整 AI 解决方案掌控未来，拥有可在任何地方部署的开放、可定制模型。</li><li><a href="https://securities.miraeasset.com/">̷ - ۷ι  Ʈ</a>：未找到描述
</li>
</ul>

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1336467121609113661)** (19 条消息🔥): 

> `Sama 招聘机器人工程师，Softbank AGI 截止日期，Krutrim 许可争议，Anthropic Jailbreak 挑战，社区对 AI 发展的看法` 


- **Sama 招聘机器人工程师引发关注**：成员们对 Sama 决定招聘**机器人工程师**表示怀疑，质疑他们在现有承诺下是否有能力应对如此**拥挤的硬件问题**。
   - 共识倾向于认为，在管理现有挑战的同时交付稳健的机器人解决方案具有极高的复杂性。
- **Softbank 逼近的 AGI 截止日期给公司带来压力**：有人担心公司必须在**两年**内为 **Softbank** 探索实现 **AGI** 的所有途径。
   - 从 AGI 项目中产生 **1000 亿美元**收益的预期加剧了行业的紧迫感。
- **Krutrim 许可争议浮出水面**：一名成员强调了围绕 Krutrim 的**许可问题**，指出它被指控公然抄袭一个开源项目且未进行适当署名。
   - 社区讨论中的引用指向了可能违反许可协议的行为，引发了伦理担忧。
- **Anthropic 为 Jailbreakers 提高筹码**：Anthropic 宣布为成功通过所有八个层级 Jailbreak 其系统的任何人提供 **1 万美元奖励**，并表示目前**还没有人完全成功**。
   - 该挑战促进了关于 AI 安全性的讨论，因为他们的目标是通过 Constitutional Classifiers 来解决漏洞。
- **社区抨击“安全演戏”**：一位用户批评那些寻求众包专业知识却不给贡献者报酬的 AI 公司，称之为**安全演戏 (Security Theater)**。
   - 这种情绪反映了社区内部对 AI 发展中的激励措施和动机的广泛不满。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/AnthropicAI/status/1887227067156386027">来自 Anthropic (@AnthropicAI) 的推文</a>: 目前还没有人完全 Jailbreak 我们的系统，所以我们要加大筹码。我们现在向第一个通过所有八个层级的人提供 1 万美元，向第一个通过所有八个层级且...</li><li><a href="https://fxtwitter.com/tokenbender/status/1887068921276362854">来自 tokenbender (@tokenbender) 的推文</a>: @teortaxesTex @ClementDelangue 他们声称 12B instruct 是经过预训练然后微调的，但 Mistral 12B 与这个模型之间的区别仅在于最外层，其他一切都是...</li><li><a href="https://x.com/tokenbender/status/1887173989245538484">来自 tokenbender (@tokenbender) 的推文</a>: 我在评论里喷了多少次 Krutrim 了？显然还不够。总有一天（也许已经发生了）他们会认为没人会发现而违反许可协议。Krutrim 并不...</li><li><a href="https://fxtwitter.com/elder_plinius/status/1887225319582466125">来自 Pliny the Liberator 🐉 (@elder_plinius) 的推文</a>: 我不想仅仅为了让你们囤积众包提示词并构建复杂的“安全演戏”表演来安抚那些愚蠢到相信...的人，而提供我世界级的专业知识。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1336426853032005643)** (68 条消息🔥🔥): 

> `DeepSeek R1 发布，OpenAI 的 Sora 工具，Nvidia Digits 关注度，GitHub Pages 证书问题，AI 模型性能讨论`

- **DeepSeek R1 发布引发争议**：2025年1月20日，DeepSeek 发布了其**权重开放的推理模型 DeepSeek-R1**，有人认为该模型可能**虚报**了其训练成本，引发了相当大的争议。
   - 该模型的架构与 **DeepSeek v3** 保持相似，引发了社区对其实际性能和定价的讨论。
- **OpenAI 的 Sora 工具在好莱坞遇冷**：OpenAI 新推出的 **Sora 电影制作工具** 尚未与好莱坞制片厂达成协议，这表明该行业可能存在抵制情绪。
   - 根据 [Bloomberg 的一篇文章](https://www.bloomberg.com/news/articles/2025-02-05/openai-s-sora-filmmaking-tool-meets-resistance-in-hollywood)，这可能反映了传统电影制作行业对新型 AI 工具更广泛的犹豫态度。
- **Nvidia 报告称对 Digits 的兴趣增加**：Nvidia 的一位代表提到，研究界对 **Digits** 的兴趣日益增加，尤其是与他们之前发布的 **Blackwell** 相比。
   - 这标志着一个积极的转变，可能会让资金紧张的大学和研究人员更容易获得 **Digits**。
- **GitHub Pages 的证书问题**：用户报告了 **GitHub Pages** 上持续存在的证书传播问题，评论指出 SSL 证书传播可能需要 **20 分钟到 24 小时**。
   - 一位用户表达了挫败感，因为在此期间多个人在访问其网站时遇到了 **403 错误**。
- **关于 AI 模型性能的讨论**：社区成员反思了 AI 模型响应的一致性，并对 **Deep Research** 等 AI 工具重复的**追问**提出了批评。
   - 人们希望 AI 系统的增强能够实现更好的上下文理解和更深层次的追问。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/Lucas_Shaw/status/1886953516054536569">来自 Lucas Shaw (@Lucas_Shaw) 的推文</a>：在过去的一年里，OpenAI 与好莱坞制片厂会面，展示其新的视频工具 Sora。目前还没有一家公司达成使用协议。与 @shiringhaffary 和 @tgbuckley 一起探讨原因...</li><li><a href="https://x.com/btibor91/status/1887102803987866007">来自 Tibor Blaho (@btibor91) 的推文</a>：好发现 - 这是另一个关于新 OpenAI Enterprise Sales Agent 演示的视频，引用 Cryptonymics🍎🚀 (@cryptonymics)：我们实际上被展示了下一个 Agent，只是我们没看到。观看...</li><li><a href="https://x.com/kalomaze/status/1887076709125824821">来自 kalomaze (@kalomaze) 的推文</a>：呃……</li><li><a href="https://arxiv.org/abs/2412.10302">DeepSeek-VL2：用于高级多模态理解的混合专家视觉语言模型</a>：我们推出了 DeepSeek-VL2，这是一个先进的大型混合专家 (MoE) 视觉语言模型系列，通过两项关键的重大升级，在上一代 DeepSeek-VL 的基础上进行了显著改进。对于...</li><li><a href="https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1">训练 DeepSeek-R1 投入了什么？</a>：本期 Gradient Updates 探讨了 DeepSeek-R1 的架构、训练成本和定价，展示了它如何以 30 倍低的成本与 OpenAI 的 o1 竞争。</li><li><a href="https://huggingface.co/spaces/deepseek-ai/deepseek-vl2-small">与 DeepSeek-VL2-small 聊天 - deepseek-ai 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://www.rlhfbook.com">(进行中) 一点点来自人类反馈的强化学习</a>：来自人类反馈的强化学习 (RLHF) 书籍</li><li><a href="https://www.pi.website/blog/openpi">开源 π0</a>：Physical Intelligence 正在将通用 AI 引入物理世界。</li><li><a href="https://interconnects.ai">Interconnects | Nathan Lambert | Substack</a>：来自前沿 AI 实验室内部的 AI 最前沿，去除炒作。高层思维与技术思维的边界。每周三早晨由领先的工程师、研究人员和投资者阅读...</li><li><a href="https://www.interconnects.ai">Interconnects | Nathan Lambert | Substack</a>：来自前沿 AI 实验室内部的 AI 最前沿，去除炒作。高层思维与技术思维的边界。每周三早晨由领先的工程师、研究人员和投资者阅读...</li><li><a href="https://youtu.be/1rokgVN9Sb8?si=rIrFlKg_2lgjNEA5">谷歌总监对 Lex Fridman 关于 DeepSeek、中国、OpenAI、NVIDIA、xAI、Stargate 的第 459 集节目做出反应</a>：Svicpodcast.com：用于提问、书籍推荐、视频、时事通讯等。patreon.com/svic 关注我们：Threads: www.threads.net/@svicpodcast Twitter: https://x.com/svi...</li><li><a href="https://youtu.be/k3d_xeVxEOE?si=eVIhUSXDlg2iu_h2~~">焕然一新。</a>：访问 https://openai.com/ 了解更多。</li><li><a href="https://youtu.be/BR_HSUUQDjA?si=hpBcvK6eskCCOhfK">初学者的钓鱼指南</a>：使用 ChatGPT 钓比目鱼
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1336469899702439978)** (5 条消息): 

> `Vibe Coding, AI 中的涂黑诗 (Blackout Poetry), Deep Research 中的澄清问题` 


- **Karpathy 的 'Vibe Coding' 成为焦点**: Andrej Karpathy 提出了一个名为 **'vibe coding'** 的新概念，他拥抱了像 Cursor Composer 这样的 LLM 的能力，通常会绕过传统的编码实践。
   - 他幽默地指出，他现在很少阅读 diff 了，并表示：*'当我收到错误消息时，我只是不加评论地直接复制粘贴进去。'*
- **AI 领域意想不到的艺术天赋**: 一位成员对 AI 领域的专业人士能够创作 **涂黑诗 (blackout poetry)** 表示惊讶，并注意到其美感。
   - 这一见解强调了科技社区中涌现出的多样化创意表达。
- **关于研究中追问问题的幽默**: 一位成员开玩笑说在 Deep Research 中 **澄清问题 (clarifying questions)** 的必要性，暗示了引发这种行为的特定交互。
   - 该评论是针对目前关于追问问题批评的讨论而发表的，突显了社区对该话题的幽默参与。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/sighswoon/status/1886826813219070290?s=46">sigh swoon (@sighswoon) 的推文</a>: 引用 Andrej Karpathy (@karpathy) 的话：有一种我称之为 "vibe coding" 的新编码方式，你完全沉浸在氛围中，拥抱指数级增长，甚至忘记了代码的存在。它...</li><li><a href="https://x.com/jam3scampbell/status/1886635547566723451?s=46">James Campbell (@jam3scampbell) 的推文</a>: 请不要告诉我这种交互就是 Deep Research 总是先问澄清问题的原因 😭
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1336852994783973439)** (3 条消息): 

> `对 RL 数据集的怀疑, 强化学习的民主化` 


- **模型开发者对非模型厂商数据的怀疑**: 有人对非模型厂商发布的 **RL 数据集价值** 提出质疑，推测如果没有成熟模型开发者的背书，其 **价值较低**。
   - 人们对模型厂商严格的 **过滤和验证** 流程表示担忧，认为未经核实的数据集可能无法获得认可。
- **关于 RL 民主化的辩论**: 一位同事对 **RL 的民主化** 表现出热情，质疑究竟谁能从中受益，以及模型流水线的哪些部分是可访问的。
   - 这引发了关于非传统组织如何做出有意义贡献的反思，但对 RL 领域的真实影响仍持怀疑态度。


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1336427257215975504)** (14 条消息🔥): 

> `JAX 的使用, Nvidia 之道, Amazon 运营, Grok 发布, Nvidia 公司文化` 


- **巨头对 JAX 的使用**: 一位成员指出 **Google** 和 **xAI** 都在使用 **JAX**，表明它是行业中的重要工具。
   - 另一位成员开玩笑说 Elon 需要开始发布 **Grok**，这可能暗示了 AI 领域的竞争。
- **《Nvidia 之道》(The Nvidia Way) 值得读吗？**: 一位成员询问《Nvidia 之道》是否值得一读，另一位成员确认他们很喜欢这本有声书。
   - 他们强调了 Nvidia 的方法是多么 **硬核 (hardcore)**，并注意到了令人印象深刻的公司文化。
- **《逆向工作法》(Working Backwards) 增强了对 Amazon 的欣赏**: 一位成员分享了阅读《逆向工作法》的心得，解释说这让他对 Amazon **高强度的运营** 有了新的理解。
   - 他们反思了该公司文化和运营中惊人的强度。


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1336694623221059634)** (2 条消息): 

> `SnailBot 新闻, RL 世界的回归` 


- **SnailBot 新闻更新**: 频道发布了关于 **SnailBot News** 的公告，预示着即将到来的更新和对社区的提醒。
   - 最新消息的细节备受期待，在成员中引起了兴奋。
- **RL 世界回归前的短暂间歇**: 一位成员提到要从 **RL 世界** 中休息一下，但以轻松的语气暗示它很快就会回归。
   - 该评论暗示了一个受欢迎的回归以及向更多互动活动的过渡。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1336433044634210314)** (208 条消息🔥🔥): 

> `LM Studio 使用, 模型兼容性, 硬件要求, Vulkan 支持, GPT-Researcher 集成`

- **在有限硬件上运行 LM Studio 的挑战**：用户讨论了在旧 CPU 和 GPU 上运行 LM Studio 的情况，指出了性能限制，特别是对于像 RX 580 这样低显存（VRAM）的显卡。
   - 一些用户建议在编译 llama.cpp 时禁用 AVX 支持，以提高在过时系统上的性能。
- **编程模型推荐**：Qwen 2.5 模型被推荐给具有特定硬件配置的用户，特别是用于编程任务。
   - 用户根据本地安装的性能和易用性分享了他们对不同模型的偏好。
- **LM Studio 中的 Vulkan 支持**：讨论围绕在 llama.cpp 中启用 Vulkan 支持以实现更好的 GPU 利用率展开，这需要特定的构建配置。
   - 用户提供了有关编译 Vulkan 支持的资源链接，强调了正确设置的必要性。
- **将 GPT-Researcher 与 LM Studio 集成**：一些用户尝试将 GPT-Researcher 与 LM Studio 配合使用，但遇到了与模型加载和 Embedding 请求相关的错误。
   - 集成挑战包括一个 404 错误，表明未加载任何模型，从而阻碍了操作。
- **图像和视频模型支持**：用户询问了 LM Studio 支持图像和视频模型的能力，特别提到了视觉（Vision）模型的可用性。
   - 提到了像 Qwen 2-VL 这样的模型支持基础图像识别任务，但不支持生成内容。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://t3.chat/chat">T3 Chat - 最快的 AI 聊天机器人</a>: 未找到描述</li><li><a href="https://imgur.com/a/WnPhj6Y">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行模因、娱乐 GIF、励志故事、病毒视频等来振奋精神...</li><li><a href="https://model.lmstudio.ai/download/lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF">在 LM Studio 中下载并运行 lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF</a>: 在你的 LM Studio 中本地使用 lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF</li><li><a href="https://block.github.io/goose/">codename goose | codename goose</a>: 你的开源 AI Agent，无缝自动化工程任务。</li><li><a href="https://lmstudio.ai/docs/basics/import-model">导入模型 | LM Studio 文档</a>: 使用你在 LM Studio 之外下载的模型文件</li><li><a href="https://huggingface.co/lmstudio-community/MiniCPM-o-2_6-GGUF">lmstudio-community/MiniCPM-o-2_6-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.org/spaces/ggml-org/gguf-my-repo">GGUF My Repo - 由 ggml-org 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://tenor.com/view/richard-stalman-richard-stalman-saint-ignucius-gnu-gif-13909134">Richard Stalman Richard GIF - Richard Stalman Richard Stalman - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-updates-february-2025/">Gemini 2.0 现已向所有人开放</a>: 我们宣布了 Gemini 2.0 Flash 的新更新，并推出了 Gemini 2.0 Flash-Lite 和 Gemini 2.0 Pro Experimental。</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#vulkan">llama.cpp/docs/build.md at master · ggerganov/llama.cpp</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://llamacoder.together.ai/">Llama Coder – AI 代码生成器</a>: 使用 Llama 3.1 405B 生成你的下一个应用</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ii82yg/deepseek_just_released_an_official_demo_for/">Reddit - 深入探索任何事物</a>: 未找到描述</li><li><a href="https://github.com/kth8/llama-server-vulkan">GitHub - kth8/llama-server-vulkan: 使用 Vulkan 运行 llama.cpp 服务器</a>: 使用 Vulkan 运行 llama.cpp 服务器。通过在 GitHub 上创建账户为 kth8/llama-server-vulkan 的开发做出贡献。</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/11678">功能请求：添加对 deepseek vl2 的支持 · Issue #11678 · ggerganov/llama.cpp</a>: 前提条件 我正在运行最新代码。如果可能请注明版本。我仔细遵守了 README.md。我使用了与我的问题相关的关键词进行搜索，以确保我正在创建...</li><li><a href="https://github.com/ggerganov/llama.cpp/releases">发布版本 · ggerganov/llama.cpp</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/ggerganov/llama.cpp/">GitHub - ggerganov/llama.cpp: C/C++ 中的 LLM 推理</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户为 ggerganov/llama.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1336427050566942752)** (39 条消息🔥): 

> `3070 与 8700K 的性能表现，M4 Max 的能力，GPU 价格与供货情况，推理的 PCIe 配置，模型的 RAM 与 VRAM 需求` 


- **较小的模型在 3070 上运行吃力**：用户报告称，在 **3070** 和超频至 **4.8GHz** 的 **8700K** 配置下，只有较小的模型能正常运行，且经常出现错误信息。
   - 一位用户询问了适用于该配置的无审查（uncensored）模型的建议。
- **M4 Max 功耗可达 140W**：MacBook Pro 上的 **M4 Max** 能够达到 **140W** 的满载功耗，展示了其性能实力。
   - 此类规格引发了关于笔记本电脑散热效率的讨论，尤其是在对比不同型号时。
- **二手市场 GPU 价格上涨**：用户对 eBay 和 Mercari 等平台上的 **GPU** 价格表示担忧，指出由于需求旺盛，它们现在已成为**增值资产**。
   - 讨论内容包括受黄牛影响而价格虚高的组件，包括 **Jetson board**。
- **优化推理的内存配置**：一位用户推测在 **2:1 模式**下运行 **9950X** 的效果，即以延迟为代价换取更好的带宽，用于 LLM 推理。
   - 然而，也有人对 **9950X** 的稳定性和 UCLK 速度表示担忧。
- **大型模型的最低 VRAM 要求**：据指出，**24GB VRAM 是最低要求**，才能流畅运行 Q4KM **32B 模型**，尤其是在复杂配置下。
   - 用户被引导至一个根据系统规格计算模型需求的资源。



**提到的链接**：<a href="https://www.canirunthisllm.net/">Can Your Computer Run This LLM?</a>：未找到描述

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1336426992622637188)** (176 条消息🔥🔥): 

> `AI for Math 贡献, DeepSeek 模型性能, AI 艺术风格迁移, Hugging Face Spaces 更新, LLM 基准测试` 


- **探索 AI for Math 的贡献**：一位成员询问了有关 AI for Math 的正在进行的、开放贡献的项目，引发了关于社区帮助的讨论。
   - 另一位成员分享了一个相关 Discord 线程的链接，以提供见解和潜在的项目更新。
- **DeepSeek 受欢迎程度超过 ChatGPT**：一位用户对 ChatGPT 性能的感知下降表示沮丧，并指出 DeepSeek 提供了更好的回答，但由于高流量面临访问问题。
   - 其他人分享了他们对 DeepSeek 思考过程的兴趣，强调其 Chain of Thought 方法论比传统的 AI 回答更具吸引力。
- **利用 LoRA 进行 AI 艺术增强**：一位刚接触 AI 艺术创作的用户询问了如何使用 LoRA 在其艺术风格上训练模型，并表示他们获知 LoRA 可以提供帮助。
   - 资深成员推荐了像 onetrainer 和 kohya-ss 这样的工具来创建 LoRA，并建议将其与 ControlNet 结合使用以获得更好的效果。
- **Hugging Face 界面更新与反馈**：收集了关于 Hugging Face 平台用户界面更改的反馈，包括缩略图功能的修改方式。
   - 用户对改进的感受褒贬不一，一些人认为导航更方便了，而另一些人则觉得新的审美不那么吸引人。
- **LLM 与 Token 澄清的挑战**：一位成员对 LLM 组合结构的内部运作以及 DeepSeek 等模型中 Token 生成的本质表示困惑。
   - 讨论包括了 Token 效率与推理深度之间的区别，强调了理解这些 AI 机制所涉及的复杂性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/learn">Hugging Face - Learn</a>：未发现描述</li><li><a href="https://huggingface.co/spaces/Pendrokar/xVASynth-TTS">xVASynth TTS - Pendrokar 创建的 Hugging Face Space</a>：未发现描述</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/">Open LLM Leaderboard - open-llm-leaderboard 创建的 Hugging Face Space</a>：未发现描述</li><li><a href="https://huggingface.co/posts/victor/435864388294574">@victor 在 Hugging Face 上：“大家好，我们对 https://hf.co/spaces 页面进行了全新的更新！Smart…”</a>：未发现描述</li><li><a href="https://huggingface.co/spaces/m-ric/open_Deep-Research/blob/main/app.py">app.py · m-ric/open_Deep-Research at main</a>：未发现描述</li><li><a href="https://www.youtube.com/watch?v=-zVgWpVXb64">Sneakers (1992): My Voice Is My Passport</a>：Sneakers。导演：Phil Alden Robinson。环球影业，1992 年。这段短片旨在为 WNYC Radio 的“On The Medi...”条目提供插图。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1336435919724478464)** (6 messages): 

> `Modified ESN Simulation, New Paper on Arxiv, Securade.ai HUB, TinyRAG System` 


- **交互式改进型 ESN 模拟上线**：[Modified ESN simulation](https://starsnatched.github.io) 允许用户通过点击神经元来进行刺激和传播，完整的 Python 代码即将发布。
   - 其预期的应用场景是使其能够在没有任何预训练数据的情况下进行学习，并可能嵌入到机器人中。
- **神经网络设计的新见解**：一篇题为 *Distance-Based Learning in Neural Networks* 的新论文已在 [arXiv](https://arxiv.org/abs/2502.02103) 发表，详细介绍了一种新颖的几何框架并引入了 OffsetL2 架构。
   - 该研究强调了**基于距离的表示（distance-based representations）**对模型性能的影响，并与基于强度的法进行了对比。
- **Securade.ai HUB 增强 CCTV 功能**：[Securade.ai HUB](https://github.com/securade/hub) 是一个基于生成式 AI 的边缘平台，可将现有的 CCTV 摄像头转换为智能系统。
   - 它承诺通过利用生成式 AI 技术，为 Computer Vision 提供一种创新的方法。
- **TinyRAG 简化 RAG 系统**：[TinyRAG](https://github.com/wannaphong/tinyrag) 项目是一个简单的 RAG（检索增强生成）系统，利用 llama-cpp-python 和 sqlite-vec 进行排序、查询并提供 LLM 回答。
   - 该计划旨在为开发者和研究人员简化 RAG 系统的实现流程。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.02103">Neural Networks Learn Distance Metrics</a>: 神经网络可能自然地倾向于基于距离的表示，其中较小的激活表示与学习到的原型更接近。这与基于强度的法形成对比，后者...</li><li><a href="https://starsnatched.github.io">Neural Network Simulation</a>: 未找到描述</li><li><a href="https://github.com/securade/hub">GitHub - securade/hub: Securade.ai HUB - 一个基于生成式 AI 的 Computer Vision 边缘平台，可连接到现有 CCTV 摄像头并使其智能化。</a>: Securade.ai HUB - 一个基于生成式 AI 的 Computer Vision 边缘平台，可连接到现有 CCTV 摄像头并使其智能化。 - securade/hub</li><li><a href="https://github.com/wannaphong/tinyrag">GitHub - wannaphong/tinyrag: 简单的 RAG 系统</a>: 简单的 RAG 系统。通过在 GitHub 上创建账号为 wannaphong/tinyrag 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/Tonic/MiniF2F">Tonic/MiniF2F · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1336438447648407604)** (4 messages): 

> `Event Timing Confirmation, Description Approval, Upcoming Event Excitement` 


- **活动时间确认**：一名成员请求确认活动详情，特别是**时间**和**描述**，并引导他人查看 [活动链接](https://discord.gg/hugging-face-879548962464493619?event=1336438204416659516)。
   - *Looking forward to the pres!* 表达了对活动演示的期待。
- **描述确认无误**：另一名成员肯定地确认了活动的**描述**是**完美的**。
   - 这表明即将举行的聚会的所有细节都已准备就绪。
- **对周日的期待**：成员们表达了对**周日**见面的兴奋之情，并称呼彼此为**大佬（legends）**。
   - 这暗示了友好的氛围以及对活动聚会的期待。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1336706100447150152)** (1 messages): 

> `Image Classification Models, ResNet50 Fine-tuning, Publishing Sector Insights` 


- **探索出版行业的图像分类模型**：一位新成员正在开发一个项目，将图像分为 **31 个类别**，如摄影、绘画和图表。
   - 他们正在寻求合适模型的见解，表示对 **ResNet50** 和 Fine-tuning 方法感兴趣。
- **寻求关于 ResNet50 的详细指导**：该成员专门询问了 **ResNet50** 作为其图像分类任务潜在模型的情况。
   - 他们表示希望就该模型的 **Fine-tuning 过程**进行深入交流。


  

---

### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1336429556453871747)** (3 条消息): 

> `Office Hours 公告，Gradio 贡献视频` 


- **加入我们的 Office Hours！**：发布了关于明天举行的 **Office Hours** 的公告，旨在讨论 **最近的发布版本** 和未来的计划，邀请所有人参加并提问。
   - 活动链接见 [此处](https://discord.com/events/879548962464493619/1336129915053015211)。
- **Office Hours 总结**：感谢大家参与今天的 **Office Hours**，并强调了与会者的积极互动。
   - 一位成员分享了一个关于 [如何为 Gradio 贡献代码的 YouTube 视频](https://www.youtube.com/watch?v=YTjwTe5Yurs)，该视频演示了向开源项目进行首次贡献的过程。



**提到的链接**：<a href="https://www.youtube.com/watch?v=YTjwTe5Yurs">如何进行你的第一次开源贡献（以 Gradio 为例）</a>：我们最常被问到的问题之一是：“我该如何开始为开源软件做贡献？”我们录制了一个修复真实 bug 的演示视频...

  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1336587877915557991)** (2 条消息): 

> `更新的 NLP 课程，当前 NLP 课程的局限性` 


- **呼吁 Hugging Face 更新 NLP 课程**：一位成员询问 Hugging Face 是否计划创建一个更新的 NLP 课程，并强调鉴于当前课程缺乏对 **LLM** 的覆盖，更新课程将非常有意义。
   - *这种情绪反映了对不断演变的 NLP 领域的关注，以及对教育资源紧跟技术进步的需求。*
- **当前 NLP 课程的局限性**：对话强调了 **现有的 NLP 课程** 不包含大语言模型，而这在当今的 NLP 框架中至关重要。
   - *这一差距促使人们建议提供更全面的培训材料，以应对该领域的新兴趋势。*


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1336461577502720060)** (15 messages🔥): 

> `Agents Course Registration, Python Coding Skills for Course, Python Learning Resources, Tools for 2D Plane to Python Code, Finetuning Models for AI Agents` 


- **Agents Course 注册确认**：一位成员询问了在报名 Agents Course 后收到注册确认邮件的时间范围，但目前尚未有关于具体时间的回复分享。
   - 这凸显了新参与者对注册后缺乏即时沟通的潜在担忧。
- **Python 编程技能的重要性**：一位参与者表示担心自己缺乏充分利用课程内容所需的 Python 基础技能，并寻求如何准备的建议。
   - 多位成员回应并推荐了 Python 资源和学习材料，以帮助弥补知识差距。
- **推荐的 Python 学习资源**：建议包括《Automate the Boring Stuff with Python》一书以及名为“Python Tutorial for Beginners with VS Code”的 YouTube 教程。
   - 这些资源旨在为准备参加课程的初学者提供 Python 编程的实用入门。
- **将图形转换为 Python 代码**：一位成员提议开发将 2D 图形或平面转换为 Python 代码的工具，以便在项目中进行实际应用。
   - 讨论了创建适合 Finetuning 像 *deepseek-r1* 这样的小型模型的数据集，以增强 AI Agents 的开发。
- **对即将发布的课程感到兴奋**：课程确认将于下周一发布，并伴随有为更新、提问和展示项目而创建的新频道。
   - 第一单元目录的预览引发了参与者的热烈期待。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://automatetheboringstuff.com/">Automate the Boring Stuff with Python</a>：未找到描述</li><li><a href="https://huggingface.co/collections/m-ric/">Could be useful one day - a m-ric Collection</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2308.11432">A Survey on Large Language Model based Autonomous Agents</a>：自主 Agent 长期以来一直是学术界和工业界的研究重点。该领域的早期研究通常侧重于在有限知识内训练 Agent...</li><li><a href="https://youtu.be/yp6yBTcdcII?si=WkOyUaghbvviWnFj">FreeCAD Part Scripting in Python Episode 025</a>：描述：为照明控制器设计外壳。软件工具链包括：Notepad++、FreeCAD、CURA，以及控制 MakerGear 的 Repetier-Server...</li><li><a href="https://www.youtube.com/watch?v=6i3e-j3wSf0">Python Tutorial for Beginners with VS Code 🐍</a>：初学者 Web 开发路线图（免费！）：https://bit.ly/DaveGrayWebDevRoadmap。在这个针对 VS Code 初学者的 Python 教程中，你将学习为什么要...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1336721841636511855)** (1 messages): 

> `HuggingFace Repo Testing, Hardware for Inference` 


- **提供 HuggingFace 模型测试帮助**：一位成员表示，一旦模型在 HuggingFace 仓库中格式化完成，他们随时准备协助进行 Inference。
   - 他们提到拥有必要的 **hardware** 来加载模型进行 **testing**。
- **对技术测试支持的需求**：该成员表现出在模型可用后提供帮助的热情，强调了他们处理 Inference 任务的能力。
   - 这体现了社区协作以及在开发新模型过程中对彻底 **testing** 流程的渴望。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1336428617848324197)** (144 messages🔥🔥): 

> `NURBS vs Meshes, AI Reasoning Models, Perspective and Transformation, Topology in 3D Modeling, Dynamic vs Static Use Cases`

- **NURBS 相比传统 meshes 具有优势**：NURBS 提供参数化且紧凑的表示，适用于精确和动态的 simulations，而 meshes 由于渲染复杂性，正日益被视为低效。
   - 虽然 NURBS 传统上在 texturing 方面面临挑战，但现代 procedural shaders 已经缓解了这些问题，使其在许多应用中变得可行。
- **新兴的经济型 AI reasoning models**：研究人员开发了 s1 reasoning model，以不到 **$50** 的云计算额度实现了与 OpenAI 模型相似的能力，标志着成本的大幅降低。
   - 该模型采用 distillation 方法，从 Google 的 Gemini 2.0 中提取推理能力，从而展示了 AI 技术向更易获取方向发展的趋势。
- **计算建模中透视的重要性**：讨论强调了对高阶代数（如 Projective Geometric Algebra (PGA) 和 Conformal Geometric Algebra (CGA)）的需求，以处理复杂的几何关系和透视。
   - 对话揭示了准确定义 perspective transformations 对于现代建模技术至关重要，特别是在动态环境中。
- **3D mesh topology 的挑战**：目前正在努力改进适用于各种应用的 mesh topologies，因为高保真 meshes 在电子游戏和电影中仍然必不可少，但在 simulations 中效率较低。
   - 向动态模型和先进技术（如 NURBS 和 SubDs）的转变，展示了在传统 mesh 方法面临局限性时行业标准的转变。
- **AI 中的静态与动态方法**：参与者指出，在 AI 和 3D modeling 等领域存在一个根本性挑战，即静态设计难以满足动态应用的需求。
   - 这一辩论强调了在计算模型和 AI agents 中演进方法论的必要性，以便在不断变化的环境中有效运行。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/gan_chuang/status/1886990694327238824?t=dJRlcmQ8WO4BCE6JG-fypw&s=19">来自 Chuang Gan (@gan_chuang) 的推文</a>：LLM 可以执行自回归搜索！LLM 可以执行自回归搜索！LLM 可以执行自回归搜索！介绍 Satori，一个通过深度思考提升 LLM 推理能力的新框架...</li><li><a href="https://x.com/dbaek__/status/1886781418115862544">来自 David D. Baek (@dbaek__) 的推文</a>：1/9 🚨 新论文预警：交叉熵损失（Cross-Entropy Loss）并非你所需要的！ 🚨我们引入了调和损失（harmonic loss）作为训练神经网络和 LLM 时标准 CE 损失的替代方案！调和损失实现了...</li><li><a href="https://archaeologymag.com/2025/01/prize-offered-to-decipher-indus-valley-script/">悬赏 100 万美元破译具有 5300 年历史的印度河谷文字</a>：泰米尔纳德邦政府为任何能够解码印度河谷文明神秘文字的人提供 100 万美元的奖励</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1cgdyjc/vae_as_image_compression/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://techcrunch.com/2025/02/05/researchers-created-an-open-rival-to-openais-o1-reasoning-model-for-under-50/">研究人员以低于 50 美元的成本创建了 OpenAI o1 “推理”模型的开源对手 | TechCrunch</a>：据报道，斯坦福大学和华盛顿大学的 AI 研究人员能够利用不到 50 美元的云计算额度训练出一个 AI “推理”模型</li><li><a href="https://github.com/huggingface/smolagents/tree/gaia-submission-r1/examples/open_deep_research">huggingface/smolagents 的 gaia-submission-r1 分支下的 examples/open_deep_research</a>：🤗 smolagents：一个极简的 Agent 库。Agent 编写 Python 代码来调用工具并编排其他 Agent。- huggingface/smolagents</li><li><a href="https://github.com/bairesearch/ATOR">GitHub - bairesearch/ATOR: Axis Transformation Object Recognition</a>：轴向变换对象识别。通过在 GitHub 上创建账号来为 bairesearch/ATOR 的开发做出贡献。</li><li><a href="https://g.co/gemini/share/792af573036a">‎Gemini - 构建一个基于 LLM 的摘要生成器
</a>：由 Gemini Advanced 创建</li><li><a href="https://www.youtube.com/watch?v=l-9ALe3U-Fg">ChatGPT 是由 1 亿个这种东西组成的 [感知器 (The Perceptron)]</a>：访问 https://drinkag1.com/welchlabs 订阅并首单立减 20 美元！感谢 AG1 赞助今天的视频。虚数...</li><li><a href="https://github.com/bairesearch/ATORpt">GitHub - bairesearch/ATORpt: Axis Transformation Object Recognition (ATOR) for PyTorch - 实验性实现，包括感受野特征/多边形检测、并行处理的几何哈希、端到端神经模型。通过 ViT 对归一化快照（变换后的补丁）进行分类</a>：用于 PyTorch 的轴向变换对象识别 (ATOR) - 实验性实现，包括感受野特征/多边形检测、并行处理的几何哈希、端到端神经模型...</li><li><a href="https://github.com/bairesearch/ATOR/wiki">首页</a>：轴向变换对象识别。通过在 GitHub 上创建账号来为 bairesearch/ATOR 的开发做出贡献。</li><li><a href="https://patentscope.wipo.int/search/en/WO2011088497">WIPO - 搜索国际和国家专利库</a>：未找到描述</li><li><a href="https://developer.nvidia.com/blog/high-fidelity-3d-mesh-generation-at-scale-with-meshtron/">使用 Meshtron 进行大规模高保真 3D 网格生成 | NVIDIA 技术博客</a>：网格（Mesh）是 3D 资产中最重要且应用最广泛的表示形式之一。它们是电影、设计和游戏行业的默认标准，并且几乎得到了原生支持...
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1336430324946829312)** (32 条消息🔥): 

> `Harmonic Loss Paper, AI Peer Review Improvements, Error Bars in AI Research, VideoJAM Analysis, Jailbreaking AI Systems` 


- **Harmonic Loss 论文受到关注**：[Harmonic Loss](https://arxiv.org/abs/2502.01628) 论文提出了一个收敛速度更快的模型，但缺乏明显的性能提升，导致对其实用性的评价褒贬不一。
   - 一位成员建议，虽然这篇论文可能有点“粗糙 (jank)”，但其简洁性值得一读，尤其是配套的 GitHub 提供了更多有价值的见解。
- **讨论 AI 同行评审流程的改进**：有人提议在同行评审中使用 A/B 测试，以提供更一致和有效的反馈，特别是在 AI 相关的出版物中。
   - 这一概念可以利用高质量的评审员基准来减少偏见并改善评审结果。
- **AI 研究中误差棒 (Error Bars) 的价值**：讨论强调了在 AI 研究论文中包含误差棒和其他统计指标的重要性，以增强清晰度和科学严谨性。
   - 成员们指出，省略这些统计工具的情况很常见，但它们对于全面的实验分析至关重要。
- **VideoJAM 论文评审已排期**：计划对 [VideoJAM 论文](https://hila-chefer.github.io/videojam-paper.github.io/) 进行讨论，该论文承诺探索突破性的视频分析方法。
   - 由于该论文是新作，受众的见解和批评对于评估其在该领域的贡献将非常有价值。
- **关于 AI 越狱能力的混合反馈**：针对正在进行的越狱尝试的评论显示，根据最近的挑战，这些系统比之前预期的更具韧性。
   - 一条推文指出，在越狱挑战开始 48 小时后，还没有人通过第 4 关，但许多人成功通过了第 3 关。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/janleike/status/1887186567200129263">Jan Leike (@janleike) 的推文</a>: 我们的越狱挑战已经进行了大约 48 小时，还没有人通过第 4 关，但我们看到更多的人通过了第 3 关</li><li><a href="https://x.com/_clashluke/status/1887144940993454311?s=46">Lucas Nestler (@_clashluke) 的推文</a>: 我创建了一个高效的 L2+HarMax 实现，并发现了一些奇怪的事情：1) 使用这个 `dist = 2 * F.linear(y, weight) - y.norm(dim=-1, keepdim=True) - weight.norm(dim=-1, keepdim=True).T ` 而不是...</li><li><a href="https://arxiv.org/abs/2502.01628">Harmonic Loss 训练可解释的 AI 模型</a>: 在本文中，我们引入了 **harmonic loss** 作为训练神经网络和大型语言模型 (LLMs) 时标准交叉熵损失的替代方案。Harmonic loss 能够提高可解释性...</li><li><a href="https://hila-chefer.github.io/videojam-paper.github.io/">VideoJAM</a>: VideoJAM: 用于增强视频模型运动生成的联合外观-运动表示</li><li><a href="https://www.independent.co.uk/tech/deepseek-ban-map-countries-ai-china-b2691924.html">DeepSeek 在世界上哪些地方被禁，原因何在？</a>: 美国可能成为第二个全面禁止这款中国 AI 应用的国家</li><li><a href="https://github.com/KindXiaoming/grow-crystals">GitHub - KindXiaoming/grow-crystals: 使用 harmonic loss 获得类晶体表示</a>: Getting crystal-like representations with harmonic loss - KindXiaoming/grow-crystals</li><li><a href="http://dx.doi.org/10.47852/bonviewJDSIS52023415">
		医疗领域类人可扩展智能的案例
							| Journal of Data Science and Intelligent Systems
			</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1336513173745959084)** (17 messages🔥): 

> `OpenAI 哨兵枪事件, Gemini 模型更新, Flash thinking 对比 Pro thinking, Gemini 2.0 Flash 性能, AI 模型排行榜` 


- **OpenAI 关停爆火的 ChatGPT 驱动哨兵枪**：在一段展示由 ChatGPT 指令控制的动力哨兵枪视频走红后，OpenAI 切断了一名工程师的 API 访问权限，引发了对 AI 驱动武器的担忧。
   - 这位名为 [sts_3d](https://linktr.ee/sts_3d) 的工程师此前曾展示过其他项目，表现出向制造哨兵枪的快速演进。
- **Gemini 模型更新发布**：Google 宣布更新后的 **Gemini 2.0 Flash** 现已在 Gemini API 和 Google AI Studio 中全面开放（GA），具有低延迟和增强的性能。
   - 像 **Flash Thinking** 这样的早期迭代版本结合了速度与推理能力，旨在提升各种用例的功能性。
- **关于 Flash thinking 与 Pro thinking 的辩论**：关于 **Flash thinking** 和 **Pro thinking** 优劣的讨论浮出水面，一些用户在对比性能时表达了对更新模型的偏好。
   - 一位用户报告称，**Flash thinking** 回答问题的效果通常比之前的 **1.5 Pro** 版本更好。
- **对 Gemini 2.0 Flash Lite 性能的担忧**：关于 **Flash Lite** 的反馈表明其在返回结构化输出方面存在困难，经常导致无效的 JSON 响应。
   - 观众继续探索各种模型，一位用户分享了他们使用 Google 新推出模型的经验。
- **排行榜缺失新模型**：有报告称，昨天新增的 Google 模型尚未反映在模型排行榜（leaderboard）上。
   - 随着用户寻求准确评估性能，模型的可见性和对比情况引起了关注。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/osanseviero/status/1887247587776069957">来自 Omar Sanseviero (@osanseviero) 的推文</a>：嘿 r/LocalLLaMA 👋 我们正在憋大招 🫡 Gemma 蓄势待发</li><li><a href="https://openrouter.ai/rankings">LLM 排名 | OpenRouter</a>：根据各应用的使用情况对语言模型进行排名和分析</li><li><a href="https://openrouter.ai/google/gemini-2.0-flash-001">Gemini Flash 2.0 - API, 提供商, 统计数据</a>：与 Gemini Flash 1.5 相比，Gemini Flash 2.0 提供了显著更快的首个 token 响应时间 (TTFT)。通过 API 运行 Gemini Flash 2.0</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-updates-february-2025/">Gemini 2.0 现已向所有人开放</a>：我们宣布了 Gemini 2.0 Flash 的新更新，并推出了 Gemini 2.0 Flash-Lite 和 Gemini 2.0 Pro Experimental。</li><li><a href="https://openrouter.ai/google/gemini-2.0-flash-lite-preview-02-05:free/api">Google: Gemini Flash Lite 2.0 Preview (免费) – 通过 API 运行</a>：Google: Gemini Flash Lite 2.0 Preview (免费) 的示例代码和 API - 与 Gemini Flash 1.5 相比，Gemini Flash Lite 2.0 提供了显著更快的首个 token 响应时间 (TTFT)</li><li><a href="https://arstechnica.com/ai/2025/01/viral-chatgpt-powered-sentry-gun-gets-shut-down-by-openai/">爆火的 ChatGPT 驱动哨兵枪被 OpenAI 关停</a>：但实际的自主 AI 武器系统要可怕得多。</li><li><a href="https://developers.googleblog.com/en/gemini-2-family-expands/">Gemini 2.0: Flash, Flash-Lite 和 Pro</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1336489406248259595)** (3 messages): 

> `WhatsApp ChatGPT 功能, Deep research 更新, YouTube 视频 'Refreshed'` 


- **WhatsApp ChatGPT 新增功能**：WhatsApp 上的 ChatGPT 现在允许用户在提问时**上传图片**并发送**语音消息**，增强了交互性。
   - 此外，未来的更新将允许用户链接他们的 ChatGPT 账户（Free, Plus, Pro）以获得更好的使用体验。
- **Deep research 已完成对 Pro 用户的推送**：**Deep research** 功能已全面推送到英国、欧盟及多个北欧国家等地区的 **100%** Pro 用户。
   - 随着挪威、冰岛、列支敦士登和瑞士的用户获得访问权限，这次更新标志着这些地区用户体验的重大提升。
- **标题为 'Refreshed' 的 YouTube 视频**：分享了一个新的 [YouTube 视频 'Refreshed'](https://youtu.be/k3d_xeVxEOE?si=5eK68F8GkrErDoXN)，宣传 OpenAI 的最新更新。
   - 观众可以在 [OpenAI 官网](https://openai.com/)查看更多详情。



**提到的链接**：<a href="https://youtu.be/k3d_xeVxEOE?si=5eK68F8GkrErDoXN">Refreshed.</a>：访问 https://openai.com/ 查看更多。

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1336429902332694662)** (92 条消息🔥🔥): 

> `DeepSeek 隐私担忧、Midjourney 对比 Flux、ChatGPT 与推理、Gemini 2.0 特性、Deep Research 可用性` 


- **DeepSeek 引发隐私红旗**：重点讨论了关于 **DeepSeek** 将数据发送至中国的担忧，并引用了一段讨论其数据实践的 [YouTube 视频](https://youtube.com/shorts/I_bGa-xIHkk?feature=shared)。
   - 一位用户指出，这可能是由于**服务器设在中国**，导致不可避免的信息传输。
- **Midjourney 与 Flux 的辩论**：一位用户认为 **Midjourney** 相比 **Flux** 具有更优越的美学表现，部分用户对这一断言提出了挑战。
   - 双方就两个平台生成的艺术作品的有效性展开了争论，展示了 Midjourney 的强大追随者群体。
- **ChatGPT 表现出混合推理**：用户报告 **ChatGPT 4o** 有时会在输入为英文的情况下使用多种语言提供推理过程，凸显了意外行为。
   - 用户对其回复过于冗长感到沮丧，特别提到一位用户在生成请求的文章时遇到了延迟。
- **Gemini 2.0 拥有令人印象深刻的特性**：Gemini 2.0 提供 **200 万 token 上下文**并可通过 API 免费访问，引起了渴望进行实验的开发者的兴趣。
   - 用户注意到自动化与 Gemini 2.0 结合的重要性，尽管一些人表示 AI 冗长的阐述导致阅读量过大。
- **Deep Research 发布消息**：Deep Research 很快将面向**英国、欧盟**及其他地区的 **Pro 用户**开放，预计 Plus 用户也将很快获得访问权限。
   - 用户对延迟表示沮丧，敦促及时更新，因为许多人正焦急等待更深层的研究功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2412.14205">Large-scale Group Brainstorming using Conversational Swarm Intelligence (CSI) versus Traditional Chat</a>：对话式群体智能 (CSI) 是一种由 AI 辅助的方法，旨在使潜在无限规模的网络化人类群体能够进行实时对话式审议和优先级排序。...</li><li><a href="https://youtu.be/l2AsXMs1igs?si=TaSoVOndHHHC9ykR">How large groups can use AI to organize without leadership</a>：使用对话式群体智能 (CSI) 进行大规模群体头脑风暴与传统聊天的对比 ArXiv: https://arxiv.org/abs/2412.14205 Bytez: https://by...</li><li><a href="https://youtube.com/shorts/I_bGa-xIHkk?feature=shared">Is DeepSeek Lying to you? #shorts #wireshark #deepseek #privacy #cybersecurity</a>：#shorts #wireshark #deepseek #privacy #cybersecurity
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1336495155989053544)** (7 条消息): 

> `测试新功能、o3mini 灵感` 


- **对新测试的初步印象**：*rjkmelb* 提到现在下结论还为时过早，但指出**初步印象**看起来非常不错。
   - 另一位成员 *.pythagoras* 认可了这一积极反馈。
- **围绕 o3mini 引用的困惑**：*niebj1749* 推测某个功能正在调用 **o3mini**，从而引发了关于来源的提问。
   - *sohamkoley_21468* 识别出可能来自 *GPT 4o Mini* 的灵感，这导致了成员之间的一些困惑。
- **关于 o3mini 的澄清**：*mustafa.h* 对关于 o3mini 的陈述表示困惑，要求澄清其含义。
   - 这引发了社区成员的简短讨论，大家仍在思考其相关性。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1336533009305964547)** (3 条消息): 

> `Statistics Analysis Prompt Design, Rhetorical Argument Structure in Writing, Sprite Sheet Generation for Animation, Character Design in Sprite Sheets` 


- **设计统计分析 Prompt**：一位用户寻求为统计分析师创建一个通用的 Prompt，通过修改针对特定问题的基础 Prompt，使其能够适应各种统计方法，如 Logistic Regression 和 Poisson Distribution。
   - 该用户的目标是让助手帮助他们解决问题，同时作为一名经济学学生，将结果与自己的理解进行对比。
- **利用修辞构建说服性论点**：一位用户分享了一个详细的 Prompt，用于生成关于“为什么可口可乐配热狗更好”的说服性论点，其中融入了高级修辞技巧，如 Antimetabole 和 Chiasmus。
   - 该结构包括论证理由、提供示例和应对反论点的章节，旨在打造一个连贯且具有影响力的最终论点。
- **创建卡通 Sprite Sheets**：一位用户询问如何创建一个 Prompt 模板来生成具有一致卡通风格的 Sprite Sheets，重点在于角色和动画帧布局，而非像素艺术（Pixel Art）。
   - 当前的模板包含角色设计、各种动作的帧数以及特定尺寸等细节，但导致生成的图像未能按预期对齐。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1336533009305964547)** (3 条消息): 

> `Statistics Analysis Techniques, Rhetoric Argument Construction, Sprite Sheet Generation` 


- **寻求统计分析领域的专业知识**：一位用户正寻求为统计分析师创建一个 Prompt，使用 Logistic Regression 和 Poisson Distribution 等方法来解决经济学中的特定问题。
   - 他们旨在构建一个可以针对不同分析进行定制的基础 Prompt，并提到自己具备统计学知识，但寻求验证方面的协助。
- **构建修辞论点**：一名成员请求建立一个结构，用于开发关于**为什么可口可乐配热狗更好**的说服性论点，并详细说明了要使用的特定修辞技巧。
   - 概述的结构强调了各种修辞手法，如 Antimetabole 和 Chiasmus，以增强论点的有效性和流畅度。
- **创建卡通 Sprite Sheets**：一位用户分享了他们的 Prompt 模板，用于生成具有特定动画帧网格布局的**海盗主题小丑鱼卡通 Sprite Sheet**。
   - 他们表达了对最终生成随机排列的图像而非理想的有序 Sprite Sheet 的担忧，并就其方法寻求反馈。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1336427090840391680)** (96 messages🔥🔥): 

> `分布式训练中的数据可用性、Hermes Reasoner 见解、OpenAI 与 DeepSeek 模型性能对比、AI 抵制与加密货币的关系、来自 OpenAI 的 DeepResearch` 


- **探索分布式训练的数据可用性**：一位成员讨论了 **Celestia** 和 **Mina** 在改进分布式训练语境下 **data availability**（数据可用性）的潜在用途，并质疑其与传统模型相比的效率。
   - 他们幽默地表示这只是在“大声思考”，并承认其具有投机性质。
- **Hermes Reasoner 大显身手**：有一条轻松的评论提到 **Hermes Reasoner** 如何“假装在纸上对齐数字”，引起了关注和乐趣。
   - 另一位成员加入讨论，表达了对不同文化如何解读数学原理的着迷。
- **OpenAI 与 DeepSeek 模型对比**：讨论强调了 **DeepSeek** R1 据称在推理能力上可与 OpenAI 的 O1 模型媲美，同时完全开源，方便用户更有效地运行。
   - 成员们注意到像 **Gemini** 这样较新的模型在执行数学任务方面的出色能力，以及品牌命名的复杂性如何困扰用户。
- **AI 抵制与加密货币曝光度相关**：一位成员推测，对 AI 的抵制是否部分归因于 2020-21 年 **NFT** 和 **crypto** 争议留下的负面残余。
   - 他们引用了一篇写得很好的文章，讨论了这一观点及其影响，表明公众对 AI 的看法可能与过去的技术炒作相互关联。
- **OpenAI DeepResearch 获得积极反响**：用户对 OpenAI 的 **DeepResearch** 功能表示热烈欢迎，注意到其强大的性能和高效检索冷门信息的能力。
   - 成员们讨论了通过知识图谱进一步增强结果，以提高事实核查和研究的准确性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/marionawfal/status/1886613146942759108?s=46">Mario Nawfal (@MarioNawfal) 的推文</a>：研究：CHATGPT 的偏见是真实的——且倾向于左翼。一项新研究证实了许多人的怀疑——ChatGPT 偏向左倾观点，经常回避或限制保守派观点。研究人员发现 A...</li><li><a href="https://x.com/teknium1/status/1886825277260792050?s=46">Teknium (e/λ) (@Teknium1) 的推文</a>：带有推理能力的 Hermes，1+1&lt;think&gt;好吧，我需要算出 1 加 1 等于多少。嗯，让我们从最基础的开始。我记得在数学课上，这是一个简单的加法问题...</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">运行 DeepSeek-R1 Dynamic 1.58-bit</a>：DeepSeek R-1 是最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。运行 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-updates-february-2025">Gemini 2.0 现已向所有人开放</a>：我们宣布了 Gemini 2.0 Flash 的新更新，并推出了 Gemini 2.0 Flash-Lite 和 Gemini 2.0 Pro Experimental。</li><li><a href="https://rentry.org/vwa65v85">为什么大家突然对 AI 感到愤怒</a>：AI 抵制：又一场技术炒作后的宿醉（加密货币是罪魁祸首吗？）（OpenAI Deep research 演示提示词：写一篇关于为什么可能会出现对 AI 的抵制，以及它是否与 NFT/crypto 的曝光度有关的论文...</li><li><a href="https://www.youtube.com/watch?v=QdEuh2UVbu0">DeepSeek R1 理论概述 | GRPO + RL + SFT</a>：这是 DeepSeek R1 论文的概述。我本周阅读了这篇论文，对其方法非常着迷，但要跟上它的逻辑有点困难...</li><li><a href="https://www.youtube.com/watch?v=hRSzhn_lDd8">DeepSeek R1 Zero 实战训练秘籍！</a>：0:00 - 2:24 论文概述 2:24 - 7:41 代码演练 1 7:41 - 15:33 带有直观示例的 GRPO 全面解释 15:33 - 18:40 代码演练 2 18:40 - 27:...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1336720144931754016)** (1 messages): 

> `O3-mini 提示词编写` 


- **为 O3-mini 编写有效的提示词**：一位成员正在寻求帮助，希望为 **O3-mini** 编写一个能鼓励模型进行更深层次思考的提示词。
   - 他们目前正在尝试中，但尚未成功，询问是否有人有见解或信息可以分享。
- **需要更具深度的交互**：讨论强调了创建能够激发 O3-mini 进行更 **深思熟虑的交互** 的提示词的重要性。
   - 成员们表达了对更丰富的提示词的渴望，以引导出更具参与感的对话。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1336485349802905632)** (3 messages): 

> `Pretraining papers, Acknowledgment of authors, Hardware infrastructure team` 


- **预训练论文作者人数激增**：一名成员评论了预训练论文中作者数量众多的现象，强调了为 **Hardware Infrastructure** 团队署名的必要性。
   - *Infra 人员的辛苦付出是为了让研究科学家们免受困扰*。
- **基础设施团队在研究中的作用**：另一次提到强调了 **Hardware Infrastructure** 团队在促进研究方面起着关键作用，这证明了众多作者署名的合理性。
   - 这种协作确保了研究科学家可以专注于他们的工作，而无需承担技术负担。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1336648909929316382)** (2 messages): 

> `Liger-Kernel PR #553, Deep Dive into LLMs` 


- **Liger-Kernel 引入 GRPO 分块损失 (Chunked Loss)**：最近的一个 [pull request](https://github.com/linkedin/Liger-Kernel/pull/553) 为 Liger-Kernel 增加了 **GRPO 分块损失**，解决了 issue #548。
   - 开发者提到运行 **make test**、**make checkstyle** 和 **make test-convergence** 来测试正确性和代码风格。
- **通过 YouTube 的 Deep Dive 探索 LLM**：一段名为《深入探讨类 ChatGPT 的 LLM》的 [YouTube 视频](https://m.youtube.com/watch?v=7xTGNNLPyMI) 探索了 ChatGPT 背后的 AI 技术及其训练过程。
   - 这个面向普通观众的演示涵盖了 **Large Language Models (LLMs)** 的**全过程训练**，并深入讨论了相关产品。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://m.youtube.com/watch?v=7xTGNNLPyMI&pp=ygUgRGVlcCBkaXZlIGludG8gbGxtcyBsaWtlIGNoYXRncHQ%3D">Deep Dive into LLMs like ChatGPT</a>：这是一个面向普通观众的深度探讨，介绍了驱动 ChatGPT 及相关产品的 Large Language Model (LLM) AI 技术。它涵盖了完整的训练过程...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/553">Grpo loss by kashif · Pull Request #553 · linkedin/Liger-Kernel</a>：摘要：添加了 GRPO 分块损失，修复了 issue #548。测试已完成：硬件类型：运行 make test 以确保正确性，运行 make checkstyle 以确保代码风格，运行 make test-convergence 以确保...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1336485349802905632)** (3 messages): 

> `Pretraining Papers Authorship, Importance of Hardware Infra Team` 


- **预训练论文及其作者名单膨胀**：成员们注意到，许多**预训练论文**拥有大量作者，因为必须为 **Hardware Infrastructure** 团队署名。
   - *一堆预训练论文有吨级的作者，因为你必须给硬件 Infra 团队记功。*
- **研究科学家受益于 Infra 团队的努力**：会议强调，**Infra 团队**忍受挑战是为了让**研究科学家**能够专注于工作而不受干扰。
   - *Infra 人员受苦，研究科学家就不必受苦。*


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1336514292597391432)** (7 messages): 

> `Closed Source Compiler, Open Sourcing Timeline, MLIR Dialects and Passes, Function Level Lowering` 


- **编译器转为闭源，社区表示好奇**：一位成员对编译器转向闭源表示理解，强调了在快速变化中管理社区贡献的挑战。
   - *编译器极客*们渴望了解其内部运作机制，特别是 MLIR 中的自定义 lowering passes。
- **编译器开源目标时间线**：一名团队成员表示，编译器预计将在明年第四季度开源，并希望能够提前发布。
   - 在一次社区会议中，他们确认了在 2026 年底前开源的承诺，讨论从[会议视频](https://www.youtube.com/watch?v=XYzp5rzlXqM)的 **17:09** 处开始。
- **无计划提前发布 MLIR passes**：一位成员询问是否会在编译器最终开源之前发布 MLIR 中的单个 dialects 或 passes。
   - 遗憾的是，官方确认在编译器开源之前没有此类计划。
- **用于并行化的函数级 Lowering**：人们对将函数级 lowering 到 LLVM 作为实现并行化的策略很感兴趣，这被认为对 MLIR 生态系统大有裨益。
   - 然而，目前没有计划在最终开源版本之前发布此类功能。



**提到的链接**：<a href="https://www.youtube.com/watch?v=XYzp5rzlXqM),">Modular milestones: GPUs, 2024 reflections, and the road ahead 🚀</a>：在这次特别的社区会议中，我们回顾了 2024 年的进展，并分享了关于以下内容的更新：🧑‍🚀 MAX 24.6，包含 MAX GPU！🔥 我们对 M 的整体方法...

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1336470194452959306)** (87 messages🔥🔥): 

> `Mojo Standard Library, Function Overloading in Mojo, Async Function Handling, Script Struct Implementation, Buffer Handling in APIs` 


- **关于 Mojo 标准库功能的讨论**：用户讨论了 Mojo 标准库是否会变得更加通用（包含 Web 服务器和 JSON 解析等功能），还是应该依赖社区贡献。
   - 考虑到支持各种用例的复杂性，人们对向 stdlib 添加新功能的门槛过高表示担忧。
- **函数重载与静态装饰器**：讨论了函数重载允许为同一个函数名提供不同的签名，从而提高 API 的灵活性。
   - 有人询问是否可以在重载函数之间共享 docstrings，但目前尚不支持。
- **Mojo 中的 Async 函数处理**：参与者探讨了处理 async 函数的提案，建议使用新语法以提高清晰度和性能优化。
   - 人们对维护不同的 async 和 sync 库的复杂性，以及对不同版本功能的可用性影响表示担忧。
- **用于 HTML API 的 Script Struct 实现**：一位用户试图通过使用单行代码通过结构化的 Script 对象更新 DOM 元素来简化其 API，但遇到了不可变性问题。
   - 替代方案涉及将其拆分为多行，成功展示了方法调用中可变性的挑战。
- **Buffer 管理与优化讨论**：对话包括关于管理 API 的 buffer 大小的见解，在处理输入数据时平衡性能与复杂性。
   - 参与者考虑了通用解析器的策略，以及在各种场景中更复杂的结构与性能之间的权衡。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/mojicians/awesome-mojo">GitHub - mojicians/awesome-mojo: A curated list of awesome Mojo 🔥 frameworks, libraries, software and resources</a>：精选的优秀 Mojo 🔥 框架、库、软件和资源列表 - mojicians/awesome-mojo</li><li><a href="https://github.com/modular/mojo/pull/3946#issuecomment-2601176112>),">[proposal] Provided Effect Handlers by owenhilyard · Pull Request #3946 · modular/mojo</a>：该提案包含一种 effect system 的替代方案，我认为它更适合在系统语言中抽象 async、raises 和类似的 function colors，因为在系统语言中上下文可能不...
</li>
</ul>

</div>
  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1336451112470839347)** (10 条消息🔥): 

> `法律实践中的 AI、案例研究辅助、证词摘要、合同审查实验、文档起草自动化` 


- **AI 彻底改变法律文档起草**：一位成员利用 AI 研究案例，并为类似案件或大规模诉讼起草重复性法律文档，并指出其可靠性和清晰的来源引用。
   - 将模板作为来源使用，使 AI 能够适配特定案例，从而使起草过程更加高效且流程化。
- **利用虚拟形象（Avatars）增强合同审查**：另一位成员分享了一项实验，涉及一段关于在合同审查中使用虚拟形象的 YouTube 演示，旨在使修订分析（redlining analysis）更具吸引力。
   - 虚拟形象的集成旨在增强产品的吸引力，并有效地协助客户团队。
- **AI 工具突出叙事弱点**：一位成员表示很高兴能通过 AI 讨论找到一种监测叙事强度的通用方法。
   - 他们注意到主持人的缺乏重点如何能揭示叙事中的弱点，从而指出需要改进的领域。
- **用户渴望创意 AI 功能**：一位成员建议，AI 工具的一个潜在改进是引入滑块来微调创意水平，类似于其他 AI 服务中的功能。
   - 这一功能可以增强用户在法律及其他背景下对 AI 生成内容的控制。
- **证词摘要增强法律工作流**：成员们正在确定 AI 在法律领域的各种应用，特别强调了其在创建证词摘要方面的效用。
   - 大家一致认为，这些工具可能比仅依赖笔记本电脑的传统方法更有效。



**提到的链接**：<a href="https://youtu.be/1G4W6XWl2WE?t=5">演示虚拟形象如何作为数字劳动力增加价值以扩展法律助理团队</a>：我们在该合同审查应用中添加了虚拟形象，使修订分析更具吸引力并使产品更具差异化。虚拟形象由 www.simli.com 提供。

  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1336452719602434172)** (84 条消息🔥🔥): 

> `NotebookLM 访问问题、NotebookLM Plus 激活、上传文件和来源、音频概览功能、电子表格集成` 


- **报告 NotebookLM 访问问题**：用户表示访问 NotebookLM 存在困难，特别是在不支持的地区或使用特定功能（如生成播客和上传 PDF）时。
   - 一位用户建议利用 VPN 来尝试绕过地理位置限制。
- **NotebookLM Plus 的激活与功能**：Google Workspace 管理员需要确保其组织至少拥有 Business Standard 许可证才能激活 NotebookLM Plus 功能。
   - 分享了相关资源以帮助管理员激活这些附加服务并有效地管理用户访问。
- **上传和使用来源的挑战**：多位用户报告了上传 PDF 和 CSV 等文件的问题，对该工具在处理股票图表和财务数据等详细内容时的表现感到沮丧。
   - 讨论强调了需要更好的格式化，并理解如何准备来源材料以获得有效结果。
- **音频概览功能的不一致性**：用户注意到音频概览的交互模式并非总能出现，引发了对其功能和潜在 Bug 的疑问。
   - 建议可能需要删除已生成的音频文件才能访问自定义选项。
- **电子表格使用和数据分析限制**：用户对 NotebookLM 在分析电子表格表格数据方面的有效性表示担忧，建议对于更复杂的数据任务使用 Gemini。
   - 用户讨论了上传电子表格的最佳实践以及在数据识别方面面临的限制。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.google/technology/google-labs/notebooklm-new-features-december-2024/">NotebookLM 获得新外观、音频交互功能和高级版本</a>：NotebookLM 正在引入新功能以及名为 NotebookLM Plus 的高级版本。</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en">升级到 NotebookLM Plus - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://support.google.com/a/answer/181865#zippy=%2Cturn-services-on-or-off-for-users">为用户开启或关闭附加 Google 服务 - Google Workspace 管理员帮助</a>：未找到描述</li><li><a href="https://support.google.com/a/answer/6043385?hl=en&co=DASHER._Family%3DBusiness&oco=0">比较 Google Workspace 版本 - 商务版 - Google Workspace 管理员帮助</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1336442164531695669)** (54 条消息🔥): 

> `Torchtune vs Unsloth 性能对比，Kolo Docker 工具，Torchtune 的 FastAPI 和 Next.js 界面，GRPO 实现，Torchtune 中的自定义脚本集成` 


- **Torchtune 性能远超 Unsloth**：一位成员称赞 **Torchtune** 的表现优于 **Unsloth**，并指出在使用 Unsloth 进行微调时，12GB 显存的 4070 显卡会出现 CUDA 内存问题。
   - 他们指出 **Torchtune** 可以无缝处理微调，而不会遇到同样的内存问题，除非 batch size 设置得过大。
- **Kolo 集成 Torchtune**：Kolo Docker 工具现在正式支持 **Torchtune**，旨在让初学者能够轻松地在本地训练和测试模型。
   - 作者分享了项目链接，展示了其配合多种工具进行 LLM 训练和测试的预期用途。
- **Torchtune 的新界面正在开发中**：一位成员正在为 **Torchtune** 开发名为 **Tune Lab** 的 FastAPI 和 Next.js 界面，使用现代 UI 组件来提升用户体验。
   - 他们讨论了预置脚本和自定义脚本的集成，并引导用户为该项目做出贡献。
- **GRPO 实现取得成功**：一位成员报告了其 **GRPO** 实现的重大成就，成功将 GSM8k 上的训练准确率从 10% 提升到了 40%。
   - 他们详细说明了调试过程中的挑战，包括死锁和内存问题，并计划清理代码以供社区贡献。
- **Tune Lab 中的自定义脚本上传**：讨论了一个潜在功能，即允许用户直接将自己的微调脚本上传到 **Tune Lab** 的 UI 中。
   - 这将涉及将用户设计的 recipes 添加到 API 中，并将脚本接口集成到应用程序的设计中。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#use-your-model-in-the-wild">使用 torchtune 的端到端工作流 — torchtune 主文档</a>：未找到描述</li><li><a href="https://github.com/MaxHastings/Kolo">GitHub - MaxHastings/Kolo: 使用现有最佳工具在本地微调和测试 LLM 的一站式商店。</a>：使用现有最佳工具在本地微调和测试 LLM 的一站式商店。 - MaxHastings/Kolo</li><li><a href="https://github.com/theosis-ai/tune-lab">GitHub - theosis-ai/tune-lab: torchtune 的 UI</a>：torchtune 的 UI。通过在 GitHub 上创建账号为 theosis-ai/tune-lab 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/issues/2340">功能请求：GRPO 支持 · Issue #2340 · pytorch/torchtune</a>：正如大家现在可能已经知道的那样，DeepSeek-R1 及其 GRPO 训练非常成功，我们是否应该考虑将 GRPO 引入 torchtune？
</li>
</ul>

</div>

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1336440980798181507)** (37 messages🔥): 

> `Ladder-residual 架构, 分布式生成问题, FSDP 同步挑战, Full DPO 分布式 PR 检查, 生成性能优化` 


- **Ladder-residual 提升 Llama 性能**：由 @zhang_muru 引入的新 Ladder-residual 修改，在多 GPU 上运行张量并行（Tensor Parallelism）时，将 **70B Llama** 模型的速度提升了 **~30%**。
   - 该工作在 [@togethercompute](https://twitter.com/togethercompute) 完成，由 @MayankMish98 共同作者，@ben_athi 指导。
- **generate 函数中令人头疼的调试**：一名成员报告了在多设备设置下运行 **generate 函数** 时可能出现的死锁问题，怀疑该函数仅针对单设备场景进行了优化。
   - 讨论的解决方案包括忽略 stop tokens 或实现同步阶段，以防止不同 rank 之间的执行不匹配。
- **生成过程中 FSDP 的挑战**：另一位成员指出，由于 all-gather 过程较慢，使用 **FSDP** 进行模型生成效率较低，并建议需要更好的 API 来支持切换并行策略。
   - 他们提到，移除 stop tokens 有助于缓解生成过程中的某些问题，使处理更加顺畅。
- **修复 Full DPO PR 的 GitHub Checks**：Full DPO Distributed PR 的 GitHub checks 出现问题，包括 `ValueError` 和 OOM 错误，引发了关于避免在没有 GPU 的 CPU runner 上运行测试的讨论。
   - 建议包括添加装饰器以在少于两个 GPU 的机器上跳过测试，旨在建立更可靠的 CI/CD 流程。
- **性能优化发现**：测试表明，在前向传播（forward passes）期间禁用 FSDP re-sharding 可以提升性能，但代价是增加峰值显存占用。
   - 讨论强调了在保持分布式设置中生成速度的同时，需要解决潜在的内存效率低下问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/">GitHub · 在单一协作平台上构建和交付软件</a>：加入全球应用最广泛、AI 驱动的开发者平台，数百万开发者、企业和最大的开源社区在此构建推动人类进步的软件。</li><li><a href="https://x.com/zhang_muru/status/1886870194443968529">Muru Zhang (@zhang_muru) 的推文</a>：在多 GPU 上运行模型但经常发现速度不理想？我们引入了 Ladder-residual，这是一种感知并行的架构修改，使具有张量并行的 70B Llama ...</li><li><a href="https://github.com/pytorch/torchtune/pull/2275/commits/fb228c6fb1a0c27795999b7811a55deedbd6bab4).">更好地共同构建软件</a>：GitHub 是人们构建软件的地方。超过 1.5 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/SalmanMohammadi/torch-redistribute/blob/main/fsdp_unwrap_2.py">torch-redistribute/fsdp_unwrap_2.py (main 分支) · SalmanMohammadi/torch-redistribute</a>：通过在 GitHub 上创建账号为 SalmanMohammadi/torch-redistribute 的开发做出贡献。</li><li><a href="https://github.com/sam-pi/torchtune/blob/add-feature-full-dpo/tests/recipes/test_full_dpo_distributed.py#L72.">torchtune/tests/recipes/test_full_dpo_distributed.py (add-feature-full-dpo 分支) · sam-pi/torchtune</a>：PyTorch 原生微调库。通过在 GitHub 上创建账号为 sam-pi/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/a226a58b8c36db5afa123f0885c5337d1ebc91f6/tests/recipes/test_full_finetune_distributed.py#L75">torchtune/tests/recipes/test_full_finetune_distributed.py · pytorch/torchtune</a>：PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/2275">Full DPO Distributed by sam-pi · Pull Request #2275 · pytorch/torchtune</a>：上下文改编自 #1966 的出色工作。此 PR 的目的是什么？是否添加了新功能？请链接此 PR 解决的任何问题：涉及 #2082。更新日志...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1336433357931937864)** (75 messages🔥🔥): 

> `OpenAI SWE Agent, OmniHuman 视频生成, Figure 独立于 OpenAI, Gemini 2.0 Flash 发布, Mistral AI 品牌重塑`

- **OpenAI SWE Agent 预计即将发布**：一项公告披露，OpenAI 计划在第一季度末或第二季度中期发布全新的 **SWE Agent**，由面向企业的 **O3** 和 **O3 Pro** 提供动力。
   - 该 Agent 预计将对软件行业产生重大影响，据称其能力足以与中级工程师竞争。
- **OmniHuman 视频生成引发关注**：一个新的视频研究项目 **OmniHuman** 声称可以仅凭单张图像和音频生成逼真的数字人视频，且没有长宽比限制。
   - 该项目获得了极大关注，被描述为一项突破，其细节之丰富令观众感到“目瞪口呆”。
- **Figure 宣布脱离 OpenAI 独立发展**：在报道称取得重大突破后，Figure AI 决定退出与 OpenAI 的合作协议，转而专注于内部 AI 技术。
   - 创始人暗示将在未来 30 天内展示“从未在人形机器人上见过的东西”，引发了社区的好奇。
- **Gemini 2.0 Flash 正式发布 (GA)**：Google 宣布 **Gemini 2.0 Flash** 现已正式商用，允许开发者通过 AI Studio 或 Vertex AI 构建生产级应用。
   - 该模型支持 **200 万 token** 的上下文，引发了关于其相对于 Pro 版本性能表现的讨论。
- **Mistral AI 品牌重塑公告**：Mistral AI 的网站进行了重大的品牌重塑，推广其可定制、便携且企业级的 AI 平台。
   - 他们强调了自己作为开源 AI 领先贡献者的角色，并致力于提供引人入胜的用户体验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/arankomatsuzaki/status/1887211023423431134?s=46">来自 Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：关于 Gemini 2.0 更新的几点说明：- 在他们的基准测试中，Flash 和 Pro 之间的整体性能差距似乎非常小。-&gt; Flash 非常出色。Pro 在长尾知识方面表现优异，这对于 ... 很重要。</li><li><a href="https://x.com/unseenvie/status/1886672598576325011?s=46">来自 Jianwen Jiang (@unseenvie) 的推文</a>：很高兴向大家展示我们最新的研究成果 OmniHuman。只需一张图像和一段音频，它就能生成极其逼真的各种宽高比和身体比例的数字人视频，这与现有方法的局限性不同...</li><li><a href="https://x.com/altryne/status/1886994096973341126">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：我认为人们只是看了看，留下了深刻印象然后就走开了... 但不... 这是，他们真的突破了某种现实屏障。我不记得有很长一段时间像这样无言以对了 ...</li><li><a href="https://x.com/physical_int/status/1886822689157079077">来自 Physical Intelligence (@physical_int) 的推文</a>：许多人向我们要 π₀ 的代码和权重，我们很高兴地宣布，我们正在新的 openpi 仓库中发布 π₀ 和预训练的 checkpoints！我们在一些公开机器人上测试了该模型，并且...</li><li><a href="https://x.com/kimmonismus/status/1887140760337744193?s=46">来自 Chubby♨️ (@kimmonismus) 的推文</a>：在最近的东京演讲中发现了 OpenAI Sales Associate Agent（放大看，图片来自直播）</li><li><a href="https://x.com/harambe_musk/status/1886779961790345657?s=46">来自 harambe_musk🍌 (@harambe_musk) 的推文</a>：OpenAI 计划在第一季度末或第二季度中期发布由 o3 和 o3 pro 驱动的企业级 SWE agent。预计这将撼动软件行业，因为它显然足够聪明，可以与 ... 竞争。</li><li><a href="https://x.com/willdepue/status/1802921157198549465">来自 will depue (@willdepue) 的推文</a>：兄弟，CTO 刚辞职，现在他在 X 上发帖，安息吧 Figure。引用 Brett Adcock (@adcock_brett) 的话：自创立 Figure 以来，我一直对实现人形机器人的规模化制造感兴趣...</li><li><a href="https://x.com/hyperbolic_labs/status/1887229114769359013?s=46">来自 Hyperbolic (@hyperbolic_labs) 的推文</a>：我们很荣幸 Andrej Karpathy @karpathy 将 Hyperbolic 认可为他最喜欢的与 LLM 基础模型交互的平台。在他最新的关于大语言模型（LLMs）的深度解析视频中，他...</li><li><a href="https://x.com/deedydas/status/1886990427422908504">来自 Deedy (@deedydas) 的推文</a>：Wojciech Zaremba 发表的 OpenAI 新论文显示，如果推理模型较小或在无意义的事情上花费太多时间，你可以通过要求它们“少思考”来破解（jailbreak）它们。这里有如此多...</li><li><a href="https://x.com/lmarena_ai/status/1887180371219132898?s=46">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：新闻：@GoogleDeepMind Gemini-2.0 系列（Pro, Flash 和 Flash-lite）现已在 Arena 上线！- Gemini-2.0-Pro 在所有类别中均排名第一 - Gemini-2.0-Flash 排名第三，现已向开发者广泛开放...</li><li><a href="https://x.com/RemiCadene/status/1886823939856589296">来自 Remi Cadene (@RemiCadene) 的推文</a>：⭐ @LeRobotHF 上首个可用的基础模型 ⭐ Pi0 是最先进的 Vision Language Action 模型。它接收自然语言指令作为输入，并直接输出自主行为。它曾...</li><li><a href="https://x.com/karpathy/status/1887211193099825254?s=46">来自 Andrej Karpathy (@karpathy) 的推文</a>：YouTube 上的新 3 小时 31 分钟视频：“深度解析像 ChatGPT 这样的 LLMs”。这是一个面向普通观众的深度解析，涵盖了驱动 ChatGPT 及相关产品的 LLM AI 技术。它...</li><li><a href="https://www.youtube.com/watch?v=eW7rUtYHD9U">Bob McGrew：AI Agents 以及通往 AGI 之路</a>：根据 OpenAI 前首席研究官 Bob McGrew 的说法，推理和 test-time compute 将解锁更可靠、更强大的 AI agents——以及一条清晰的 ...</li><li><a href="https://x.com/sundarpichai/status/1887169871697350775">来自 Sundar Pichai (@sundarpichai) 的推文</a>：1/ Gemini 2.0 的新更新来了！Gemini 2.0 Flash 现已正式发布（GA），开发者现在可以构建生产级应用。在 AI Studio 或 Vertex AI 中即可找到它。</li><li><a href="https://x.com/adcock_brett/status/1886860098980733197">来自 Brett Adcock (@adcock_brett) 的推文</a>：今天，我决定终止与 OpenAI 的合作协议。Figure 在完全端到端的机器人 AI 方面取得了重大突破，完全由内部构建。我们很高兴在接下来的...中向大家展示。</li><li><a href="https://www.youtube.com/live/Gv7torZn5lM?si=cGtkvCCtfrj3vkcO">直播：OpenAI 创始人 Sam Altman 在东京演讲</a>：观看 OpenAI CEO Sam Altman 在日本东京举行的“通过 AI 转型业务”活动中的现场演讲，软银 CEO 孙正义和 Arm Hold... 也将出席。</li><li><a href="https://youtu.be/k3d_xeVxEOE?si=J58PWRMh5foGFquA">焕然一新。</a></li>

: 查看 https://openai.com/ 了解更多。</li><li><a href="https://youtu.be/7xTGNNLPyMI?si=0kcjG0Xt4J-6hs4n">深入探讨 ChatGPT 等 LLM</a>：这是面向普通观众的深度解析，探讨了驱动 ChatGPT 及其相关产品的 Large Language Model (LLM) AI 技术。它涵盖了完整的训练...</li><li><a href="https://mistral.ai/en">Mistral AI | 触手可及的前沿 AI</a>：通过从平台到界面的完整 AI 解决方案掌控未来，拥有可在任何地方部署的开放、可定制模型。</li><li><a href="https://techcrunch.com/2025/02/04/figure-drops-openai-in-favor-of-in-house-models/">Figure 放弃 OpenAI 转而采用内部模型 | TechCrunch</a>：Figure AI 是一家致力于将通用人形机器人投入商业和住宅使用的机器人公司，周二在 X 上宣布其
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[announcements](https://discord.com/channels/1076964370942267462/1090471714888102009/1336489836286312521)** (1 messages): 

> `GPT4All v3.9.0 Release, LocalDocs Fix, DeepSeek-R1 Update, Windows ARM Improvements, New Model Support` 


- **GPT4All v3.9.0 正式亮相！**：**GPT4All v3.9.0** 已正式发布，包含重大修复和新功能。
   - 显著改进包括 **LocalDocs** 功能以及对新模型的增强支持。
- **LocalDocs 现在无错误运行！**：**LocalDocs** 已修复，以防止在使用推理模型时后续消息中出现错误。
   - 此增强功能优化了用户体验并确保了更流畅的交互。
- **DeepSeek-R1 优化了聊天显示**：**DeepSeek-R1** 的推理输出不再会使聊天名称或 'think' 标签内的后续问题变得杂乱。
   - 此修复增强了 AI 交互过程中的整体清晰度和连贯性。
- **Windows ARM 获得性能提升**：解决了特定 SoC 上的**图形伪影**问题，并修复了 Windows ARM 用户向 **LocalDocs** 添加 PDF 集合时发生的崩溃。
   - 这些改进为在 Windows ARM 架构上运行的用户提供了更顺畅的体验。
- **欢迎新模型 OLMoE 和 Granite MoE！**：此版本引入了对新 **OLMoE** 和 **Granite MoE** 模型的支持，扩展了 GPT4All 的功能。
   - 这一新增功能为用户的 AI 需求和应用提供了更广泛的选择。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1336439673794465903)** (62 messages🔥🔥): 

> `ReAG - Reasoning Augmented Generation, 自托管 GPT4All, 用于 NSFW 内容的本地模型, 用户界面 Bug, 数据湖担忧` 


- **ReAG 为 RAG 提供了一种新方法**：[ReAG - Reasoning Augmented Generation](https://github.com/superagent-ai/reag) 是一种新方法，它直接将原始文档输入给语言模型，从而产生比传统方法更具上下文感知能力的响应。
   - 这种统一的方法有望通过避免过度简化的语义匹配来提高准确性和相关性。
- **将 GPT4All 作为服务器进行自托管**：一位用户探讨了在桌面端设置 GPT4All 以实现移动端连接的可能性，并建议可以通过 Python 宿主来实现。
   - 虽然技术上可行，但其他人警告说支持可能有限，且需要非常规的设置。
- **用于 NSFW 用途的本地模型**：讨论包括寻找可用于 NSFW 故事的本地 LLM，一些成员认为 *wizardlm* 和 *wizardvicuna* 等建议并非最优。
   - 有人推荐了 *obadooga* 和 *writing-roleplay-20k-context-nemo* 等潜在替代方案，以获得更好的性能。
- **关于滚动的 UI Bug 报告**：一位用户报告了一个 UI Bug，即如果文本超过可视区域，提示词窗口的内容无法滚动，从而导致了访问性问题。
   - 另一位成员指出 GitHub 上也报告了类似问题，引发了大家对该问题在其他用户中普遍性的好奇。
- **对数据湖内容的担忧**：成员们对不当话题的盛行表示不安，特别是与数据湖内容中的儿童相关的内容。
   - 这引发了关于特定条目以及所引用内容整体适当性的讨论。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GGUF">TheBloke/Wizard-Vicuna-7B-Uncensored-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/superagent-ai/reag">GitHub - superagent-ai/reag: Reasoning Augmented Generation</a>: 推理增强生成。通过在 GitHub 上创建账号来为 superagent-ai/reag 的开发做出贡献。</li><li><a href="https://github.com/tani/markdown-it-mathjax3">GitHub - tani/markdown-it-mathjax3: 使用 MathJax 插件为 Markdown-it 添加数学公式支持</a>: 使用 MathJax 插件为 Markdown-it 添加数学公式支持 - tani/markdown-it-mathjax3
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1336483711516020819)** (50 messages🔥): 

> `ChatGPT Pro 订阅, MCP Excel 文件操作, Playwright/Puppeteer 自动化, GitHub MCP 使用, Home Assistant MCP 客户端/服务器支持` 


- **对 ChatGPT Pro 订阅的兴趣**：多位成员表达了购买 **ChatGPT Pro** 订阅的兴趣，特别是考虑到潜在的团队使用。
   - *你是否打算在多个账号之间分摊费用？*
- **关于 MCP 读取 Excel 文件的讨论**：有一场关于创建能够读取和操作 **Excel 文件** 的 **MCP** 可行性的对话，成员们对开发此类工具表现出极大的热情。
   - 对于数据处理任务，使用 **Python** 还是 **TypeScript** 的建议引发了辩论。
- **Playwright 和 Puppeteer 自动化查询**：成员们讨论了使用较新的 **Playwright** 和 **Puppeteer** MCP 的经验，其中一位分享说 **Playwright** 运行良好，而 **Puppeteer** 需要进行本地修改。
   - 大家对特定插件表现出兴趣，并分享了一个目前仍处于非生产就绪状态的 GitHub 实现链接。
- **关于 GitHub MCP 使用的见解**：成员们分享了使用来自 **Anthropic** 的 **GitHub MCP** 的经验，强调了它在处理 **README 文件** 问答任务时的实用性。
   - 讨论内容包括获取原始文件等功能以及 **Claude** 上的速率限制挑战，并提出了多种使用策略。
- **发布新的 Home Assistant MCP 支持**：一位成员宣布发布了具有 **MCP 客户端/服务器支持** 的 **Home Assistant**，标志着其功能的扩展。
   - *太棒了！很高兴看到在自动化生态系统中的进一步集成。*



**提及的链接**: <a href="https://github.com/isaacphi/servers/blob/evaboot/src/puppeteer/index.ts">servers/src/puppeteer/index.ts at evaboot · isaacphi/servers</a>: Model Context Protocol 服务器。通过在 GitHub 上创建账号来为 isaacphi/servers 的开发做出贡献。

  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1336489597995188285)** (6 messages): 

> `Sage Smithery 集成，Claude 的 MCP Tools 支持，PulseMCP Use Cases 发布` 


- **Sage 集成 Smithery 以提供无缝体验**：Sage 宣布将 **Smithery** 原生引入其应用，并于今晚发布！
   - 此次集成旨在增强用户交互并简化应用程序内的工作流程。
- **Claude 现在支持数百种工具！**：**Sage for Claude** 的推出通过 Model Context Protocol 带来了对数百种工具的支持，可在 **iOS 和 Mac** 上使用。
   - 这包括对**本地和托管 MCP server** 的一键安装，以及许多易用性改进（Quality-of-life changes），例如使用上箭头键编辑最后一条消息。
- **PulseMCP 发布 Use Cases 展示**：PulseMCP 推出了一个新功能，重点展示实用的 MCP server 和 client 组合，并配有详细说明、截图和视频。
   - 值得关注的用例包括使用 **Gemini 语音管理 Notion**，以及使用 **Claude** 将 **Figma 设计稿转换为代码**。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.pulsemcp.com/use-cases">Community use-cases of MCP in-action | PulseMCP</a>：探索社区使用 Model Context Protocol (MCP) 的各种方式。</li><li><a href="https://x.com/tadasayy/status/1887253558749471034">来自 Tadas Antanavicius (@tadasayy) 的推文</a>：🎉 宣布在 PulseMCP 上发布 Use Cases（关注 @pulsemcp 以获取最新动态）！自 @Anthropic 发布 MCP 以来，已经诞生了大量优秀的 MCP server 和 client，我们构建了一个资源库来突出显示...</li><li><a href="https://github.com/SecretiveShell/Awesome-llms-txt/">GitHub - SecretiveShell/Awesome-llms-txt: 此列表包含托管在各个网站上的 llms.txt 文件索引。</a>：此列表包含托管在各个网站上的 llms.txt 文件索引。 - SecretiveShell/Awesome-llms-txt
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1336501897229766676)** (8 messages🔥): 

> `Huggingface L40S 性能对比，Janus-Pro-7B 结果，EvaByte 架构，自回归图像生成，图像建模中的 Byte transformers` 


- **Flux 1024x1024 图像生成性能远超 Emu3**：在 Huggingface L40S 实例上，生成 720x720 图像时 **Emu3** 耗时约 600 秒，而 **Flux** 使用 `flash-attention` 和 W8A16 量化生成 1024x1024 图像仅需 **30 秒**。
   - 尽管参数量相当（Emu3 为 8B，Flux 为 12B），但巨大的速度差异引发了关于 Emu3 与单模态模型相比效率如何的疑问。
- **Janus-Pro-7B 的速度提升被糟糕的输出质量抵消**：在 Huggingface spaces 测试 **Janus-Pro-7B** 后，用户注意到其速度快得多，可与 **DiTs** 媲美，但对生成的图像质量表示失望。
   - 一位成员分享了输出示例附件，强调了尽管处理速度有所提高，但质量问题依然显著。
- **介绍 EvaByte：高效图像生成的游戏规则改变者**：新推出的 **EvaByte** 是一个 6.5B 的字节级（byte-level）语言模型，采用创新架构，旨在提高自回归图像生成的效率和性能。
   - EvaByte 在 1.5T bytes 数据上进行训练，解码速度比传统系统快达 **2 倍**，将影响自回归图像生成技术的潜在发展。
- **自回归视觉模型的并行生成策略提案**：一项提议的方法论强调了利用视觉 token 依赖关系如何实现自回归模型中的并行化生成，从而大大加快创建过程。
   - 建议将此技术与 EvaByte 架构无缝结合，相比传统的 next-token 预测方法，可能实现 **12 倍的速度提升**。
- **对用于图像建模的 Byte Transformers 的好奇**：一位成员询问是否出现了专门使用 byte transformers 进行图像建模或其他非文本模态建模的论文或模型。
   - 该问题强调了探索 byte transformers 在生成式建模各个领域通用性的兴趣。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2412.15119">Parallelized Autoregressive Visual Generation</a>：自回归模型已成为视觉生成的一种强大方法，但由于其顺序的 token-by-token 预测过程，推理速度较慢。在本文中，我们提出...</li><li><a href="https://hkunlp.github.io/blog/2025/evabyte/">EvaByte: Efficient Byte-level Language Models at Scale | HKU NLP Group </a>：未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1336722325030305862)** (2 条消息): 

> `tl.gather 函数, Triton 安装, 从源码安装` 


- **Triton 安装后 tl.gather 函数的问题**：@webstorms 报告了在安装 **Triton** 版本 **3.1.0**、**3.2.0** 以及 **nightly-build** 后，调用 **tl.gather 函数** 遇到困难。
   - 他们对安装过程表示困惑，认为自己可能在某些简单的步骤上出了错。
- **建议从源码安装**：一位成员建议 **tl.gather 函数** 可能尚未随 **Triton 3.2** 发布，建议从 [源码](https://github.com/triton-lang/triton?tab=readme-ov-file#install-from-source) 安装。
   - 他们提供了一个指向 **[Triton GitHub 仓库](https://github.com/triton-lang/triton)** 的链接以获取进一步帮助。



**提到的链接**：<a href="https://github.com/triton-lang/triton?tab=readme-ov-file#install-from-source)">GitHub - triton-lang/triton: Triton 语言和编译器的开发仓库</a>：Triton 语言和编译器的开发仓库 - triton-lang/triton

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1336575092389445666)** (5 条消息): 

> `GPU 失效 (Invalidations), 微基准测试技术, WGMMA 布局, AI 计算效率` 


- **GPU 失效概览**：讨论表明，重型类型的 **fences** 可能会导致完整的 L1 失效操作，应尽量避免。
   - *NVIDIA 可能不会跟踪由流填充的特定行以进行失效处理*，这使得该场景对性能而言并不理想。
- **通过微基准测试获取 GPU 性能洞察**：建议根据 **wgmma layouts** 的经验，运行 **microbenchmarks** 以获取对 GPU 行为的深入了解。
   - 大家认识到可能缺乏详细文档，因此可能需要直接咨询硬件工程师。
- **减少 AI 计算使用的努力**：最近的讨论强调了通过各种方法（包括 **FlashAttention** 等）提高 AI 效率并减少计算量的关注点。
   - 这描绘了旨在优化 AI 应用中 **计算资源 (compute resources)** 的持续创新。



**提到的链接**：<a href="https://hazyresearch.stanford.edu/blog/2024-05-12-tk">GPUs Go Brrr</a>：如何让 GPU 变快？

  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1336569144195153940)** (5 条消息): 

> `BlockMask 对 .state_dict() 的支持, Flex Attention, Torch 保存与加载` 


- **关于 BlockMask 和 .state_dict() 支持的查询**：一位成员询问了 **flex attention 的 BlockMask** 支持 `.state_dict()` 的可能性，并建议这可能是一个很好的第一个 PR。
   - 提到了 *感谢对 .to() 的支持*，对之前的贡献表示感谢。
- **关于序列化 (pickling) BlockMask 的问题**：另一位成员询问是否可以简单地 *pickle BlockMask*，对最初的想法表示怀疑。
   - 这引发了关于处理 **BlockMask** 灵活性的讨论。
- **BlockMask 集成的提议议题**：一位成员提议，一个更好的入门议题是将 **BlockMask** 添加到 **safe globals** 中，用于仅权重 (weights-only) 操作。
   - 他们表示有信心支持这个 PR，并表示会为其盖章（批准）。
- **BlockMask 功能示例代码**：分享了一个代码片段，演示了使用 `torch` 实现 **BlockMask** 及其保存/加载功能。
   - 它展示了创建 block mask、将其保存到文件并加载回来以供进一步使用的过程。


  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1336451447449194558)** (3 条消息): 

> `OmniHuman framework, FlowLLM for material discovery, Video generation from images, Generative models in research` 


- **OmniHuman 框架生成逼真的真人视频**：[OmniHuman 项目](https://omnihuman-lab.github.io/) 引入了一个端到端的多模态条件真人视频生成框架，该框架可以从单张图像和运动信号（如音频和视频）中创建视频。
   - 该方法通过使用混合训练策略改进了以往的方法，大大提升了生成视频的质量。
- **FlowLLM 推进材料发现**：[FlowLLM](https://arxiv.org/abs/2410.23405) 是一种新型生成模型，它将 Large Language Models (LLM) 与 Riemannian flow matching 相结合，用于设计新型晶体材料，显著提高了生成速率。
   - 这种方法在材料生成速度上超越了现有方法，在基于 LLM 输出开发稳定材料方面的效率提高了三倍以上。
- **将图像转化为视频**：OmniHuman 独特的框架可以仅凭单张图像生成高质量的真人视频内容，彰显了其在多媒体应用方面的潜力。
   - 这一创新在处理弱信号输入方面表现出色，为视频生成技术树立了新标准。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://omnihuman-lab.github.io/">OmniHuman-1 Project</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2410.23405">FlowLLM: Flow Matching for Material Generation with Large Language Models as Base Distributions</a>：材料发现是一个关键的研究领域，具有彻底改变包括碳捕获、可再生能源和电子产品在内的各个领域的潜力。然而，化学空间的巨大规模...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1336752745071251456)** (2 条消息) : 

> `Part-Time AI Software & Hardware Optimization Engineer, Modal Serverless Computing, GPU Performance Engineering` 


- **兼职 AI 工程师职位开放**：我们的客户正在招聘 **兼职 AI 软件与硬件优化工程师**，职位面向欧盟和亚洲地区远程办公。您可以[在此查看完整的职位描述并申请](https://livit.teamtailor.com/jobs/5511494-part-time-ai-software-hardware-optimization-engineer-remote-flexible/9d602719-430f-450f-9a9d-54e92bcaee81)。
- **Modal 助力高性能计算**：[Modal](https://modal.com/) 是一个 **Serverless 计算平台**，为 **Suno** 和 **Liger Kernel** 团队等用户提供灵活、自动扩展的计算基础设施。**GPU Glossary** 被强调为该团队产出的显著技术成果之一。
   - *“我在这里完成了我人生中一些最出色的技术工作……”* 这句话进一步印证了 Modal 积极的团队文化。
- **Modal 招聘 ML 性能工程师**：Modal 正在招聘 **ML 性能工程师**，以增强 GPU 性能并为 **vLLM** 等上游库做出贡献。感兴趣的候选人可以[在此查看职位描述](https://jobs.ashbyhq.com/modal/af17da5e-23ca-4802-854d-5f0546e1ed32)。
   - 鼓励感兴趣的人士通过私信（DM）获取更多信息。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://jobs.ashbyhq.com/modal/af17da5e-23ca-4802-854d-5f0546e1ed32))">Jobs</a>：未找到描述</li><li><a href="https://livit.teamtailor.com/jobs/5511494-part-time-ai-software-hardware-optimization-engineer-remote-flexible/9d602719-430f-450f-9a9d-54e92bcaee81">Part Time AI Software & Hardware Optimization Engineer (Remote/Flexible) - Livit</a>：我们正在寻找一名 AI 软件与硬件优化工程师，负责分析、调整和优化我们现有的基于 CUDA 的 AI 模型，使其在不同的硬件架构上高效运行...
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1336783696161804380)** (4 messages): 

> `Torchao 与 torch.compile 的兼容性，PyTorch issue 讨论，GitHub 上的社区参与` 


- **Torchao 可能与 torch.compile 不兼容**：一位用户报告说，同时使用 **Torchao** 和 **torch.compile** 似乎会导致一个 Bug，暗示存在兼容性问题。
   - _“mega oof”_ 表达了遇到此问题的用户的沮丧心情。
- **链接到 PyTorch issue #141548**：另一位成员建议该 Bug 与[这个 GitHub issue](https://github.com/pytorch/pytorch/issues/141548)一致，该 issue 涉及 `nn.Module` 无法在设备间传输的问题。
   - 该 issue 报告提到了已编译模块和 tensor subclasses 的复杂情况，引发了社区的进一步关注。
- **社区鼓励在 GitHub 上发表评论**：一位成员鼓励该用户在 GitHub issue 下发表评论，以提高 **PyTorch** 团队对该问题的关注度。
   - _“或许你可以去那个 issue 下评论以提高可见度”_ 反映了社区积极解决问题的态度。



**提到的链接**：<a href="https://github.com/pytorch/pytorch/issues/141548">Compiled `nn.Module` with tensor subclass can&#39;t be moved to another device · Issue #141548 · pytorch/pytorch</a>：🐛 描述 Bug。导入 torch aten = torch.ops.aten。类 Subclass(torch.Tensor)：def __new__(cls, data)：return torch.Tensor._make_wrapper_subclass(cls, data.shape, dtype=data.dtype, device=data.....

  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1336475748843786431)** (3 messages): 

> `游戏中的 AI，通用机器人模型，AI 驱动的传真服务` 


- **AI 利用了 Trackmania 中的一个破坏游戏平衡的 Bug**：一段名为 [AI exploits a gamebreaking bug in Trackmania](https://m.youtube.com/watch?v=NUl6QikjR04) 的 YouTube 视频展示了如何通过强化学习（reinforcement learning）在游戏中训练 AI，使其掌握极难的 noseboost 技术。
   - 创作者旨在展示 AI 在克服游戏重大挑战方面的能力。
- **通用机器人模型 π0 发布**：该公司宣布发布其通用机器人模型 [π0](https://github.com/Physical-Intelligence/openpi) 的代码和权重，该模型可以针对各种机器人任务进行微调。
   - 此次发布旨在支持对机器人策略的实验，这可能会从根本上改变我们处理人工智能的方式。
- **创新的 Fax-KI 服务上线**：simple-fax.de 推出了一项名为 **Fax-KI** 的新服务，它将传统的传真机转变为能够通过传真回复查询的智能工具。
   - 用户可以将问题或任务发送到专用传真号码，Fax-KI 会进行分析并回复定制化的答案，增强了传统通信方式的功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.physicalintelligence.company/blog/openpi">开源 π0</a>：Physical Intelligence 正在将通用 AI 引入物理世界。</li><li><a href="https://m.youtube.com/watch?v=NUl6QikjR04">AI exploits a gamebreaking bug in Trackmania</a>：我通过强化学习在 Trackmania 中训练了一个 AI，并尝试让它学习这款游戏中最难的技术：noseboost。为了支持我的工作...</li><li><a href="https://simple-fax.de/fax-ki">Faxgeräte können jetzt auch KI</a>：人工智能目前风靡一时，simple-fax.de 将这项创新技术直接带到了您的传真机上。Fax-KI 将您的传真机变成一个智能工具，不仅...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1336543274885713942)** (1 messages): 

> `Granite 3 模型，Llama 3 模型，PR #558` 


- **引入 Granite 3 模型**：提交了一个 PR 以增加对 **Granite 3.0** 和 **3.1** 模型的支持，并指出它们与 **Llama 3** 模型相似，这意味着这是一个*易于处理的添加*。
   - 该 PR 可以在[这里](https://github.com/linkedin/Liger-Kernel/pull/558)找到。
- **相同但不等效的模型**：尽管 **GraniteMLP** 和 **LlamaMLP** 是相同的，但用 **LigerSwiGLUMLP** 替换 **GraniteMLP** 并没有产生 logit 等效性（logit-equivalence），这凸显了一个有趣的差异。
   - PR 中记录了关于 loss 和参数等效性的问题，指出了模型适配（model patching）中细微的挑战。



**提到的链接**：<a href="https://github.com/linkedin/Liger-Kernel/pull/558">Support Granite 3.0 and 3.1 models by JamesKunstle · Pull Request #558 · linkedin/Liger-Kernel</a>：Granite 3.(0,1) 模型是 Llama 架构模型，在不同位置有一些不同的缩放项。此提交为仅解码器的 Granite 3 模型（非多模态）添加了 Granite 模型适配...

  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1336490449858461717)** (10 messages🔥): 

> `CompositeDataset PR, gsm-symbolic cross-checks, laptop repair, requirements-dev updates, generator issues in gsm-symbolic` 


- **CompositeDataset PR 优化 gsm-symbolic**：`CompositeDataset` 的第一个版本已在 [PR](https://link.to.pr) 中提交，显著改进了各种 `gsm-symbolic` 生成器。
   - 这些增强功能侧重于修复大量问题，为更好的性能铺平道路。
- **Gsm-symbolic 正在进行交叉检查**：目前正在使用 **sonnet** 对 `gsm-symbolic` 进行交叉检查；对于 **difficulty=1.0**，大多数情况看起来令人满意。
   - 计划解决剩余的差异以确保整体一致性。
- **笔记本电脑维修归来**：一位成员提到他们的笔记本电脑坏了并进行了维修，现在已经重新上线。
   - 他们的回归表明他们已准备好重新参与正在进行的讨论。
- **更新 requirements-dev.txt**：由于添加了许多依赖项，有人对更新 **requirements-dev.txt** 的必要性提出了担忧。
   - 另一位成员澄清说，最重要的依赖项都列在 **pyproject.toml** 中，建议通过 `pip install -e .` 进行安装。
- **Gsm-symbolic 中的生成器问题**：目前，100 个 `gsm-symbolic` 生成器中有 **16 个** 已损坏，无法生成具有整数结果的正确问题。
   - 有推测认为可能需要一种**全新的方法**来解决涉及这些特定失败生成器的问题。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1336471854184923248)** (3 messages): 

> `Deepseek exploration, Building RAG applications, Gemini 2.0 launch, Gemini integration` 


- **在虚拟论坛中探索 Deepseek**：加入 @aicampai 参加关于 `Deepseek` 的虚拟论坛，重点关注其功能以及如何将其集成到开发者和工程师的工作流中。该活动旨在提供有关该技术及其应用的手动学习体验。
   - 在[此处](https://t.co/Gh29EHBJMf)查看论坛详情。
- **构建 RAG 应用教程**：@Pavan_Belagatti 的新视频教程指导用户使用 @llama_index 构建他们的第一个检索增强生成 (RAG) 应用程序。本教程旨在响应越来越多寻求实用开发见解的新用户。
   - 在[此处](https://t.co/LXlRztHcM4)观看教程。
- **Gemini 2.0 现已发布**：@google 宣布 `Gemini 2.0` 现已正式发布，同时 @llama_index 也提供了零日支持。用户可以通过 `pip install llama-index-llms-gemini` 安装最新的集成包，以体验其令人印象深刻的基准测试结果。
   - 在[发布博客文章](https://t.co/6oBbYpcFAU)中了解有关 `Gemini` 更新和功能的更多信息。
- **Gemini Flash 更新详解**：`Gemini 2.0 Flash` 的更新版本包括增强的性能、低延迟以及处理复杂推理的能力。用户可以通过 Google AI Studio 和 Vertex AI 中的 `Gemini API` 发现创建和协作的新方式。
   - 有关更多信息，请访问 [Google AI Studio](https://aistudio.google.com/prompts/new_chat?model=gemini-2.0-flash) 获取最新更新。



**提到的链接**：<a href="https://t.co/6oBbYpcFAU">Gemini 2.0 现已向所有人开放</a>：我们宣布了 `Gemini 2.0 Flash` 的新更新，并推出了 `Gemini 2.0 Flash-Lite` 和 `Gemini 2.0 Pro Experimental`。

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1336433710282702998)** (20 messages🔥): 

> `LlamaIndex 中的 Timeout 实现，使用 Qwen-2.5 进行 Function calling，AgentWorkflow 中的文本流式传输，结合 vLLM 使用 OpenAILike，Tool call 流式传输的限制` 


- **LlamaIndex 模型中的 Timeout 实现**：一位用户注意到默认的 LlamaIndex LLM 类缺少内置的 timeout 功能，而 OpenAI 的模型中则包含该功能。另一位用户建议 timeout 可能由 client kwargs 组成，并指向一个 [GitHub 链接](https://github.com/run-llama/llama_index/blob/7391f302e18542c68b9cf5025afb510af4a52324/llama-index-integrations/llms/llama-index-llms-azure-inference/llama_index/llms/azure_inference/base.py#L224) 以获取更多细节。
- **在 Qwen-2.5 中实现 Function Calling**：一位用户遇到了关于 Qwen-2.5 模型不支持 Function Calling API 的 `ValueError`。提供的指导指出，使用命令行参数并切换到 OpenAI-like 实现可以解决这些 Function Calling 处理问题。
   - 另一位成员分享了 [Qwen function calling 文档](https://qwen.readthedocs.io/en/latest/framework/function_call.html)，提供了有关实现的见解。
- **AgentWorkflow 中的文本流式传输**：一位正在构建 AgentWorkflow 的用户在向客户端流式传输文本时遇到问题，因为在流式传输激活时发送的消息 delta 始终为空。有见解分享称，当 LLM 正在编写 tool call 时，delta 将显示为空，检查 `event.tool_calls` 可能有助于理解。
   - 强调了流式传输 tool call 输出的限制，并提出了基于最新 tool call 更新前端的潜在变通方法。
- **为 Qwen-2.5 使用 OpenAILike**：为了在 Qwen-2.5 中顺利使用 Function Calling，一位用户转而实现 LlamaIndex 的 `OpenAILike` 类。提供了安装和设置说明，以成功管理 Function Calling 的参数。
- **Tool Call 流式传输的限制**：一位成员指出，由于 OpenAI 以完成的 JSON 形式流式传输数据的方式，流式传输 tool call 输出存在困难。他们对是否能添加此功能表示怀疑，并建议采用流式传输最新 tool call 版本的替代方案。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://qwen.readthedocs.io/en/latest/framework/function_call.html">Function Calling - Qwen</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/7391f302e18542c68b9cf5025afb510af4a52324/llama-index-integrations/llms/llama-index-llms-azure-inference/llama_index/llms/azure_inference/base.py#L224">llama_index/llama-index-integrations/llms/llama-index-llms-azure-inference/llama_index/llms/azure_inference/base.py at 7391f302e18542c68b9cf5025afb510af4a52324 · run-llama/llama_index</a>: LlamaIndex 是领先的框架，用于在您的数据上构建由 LLM 驱动的 Agent。 - run-llama/llama_index
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1336428327644303443)** (14 messages🔥): 

> `MOOC 证书延迟，Quiz 1 和 Quiz 2 的可用性，Quiz 的技术问题，证书申请流程` 


- **MOOC 证书延迟**：一位成员对未收到 12 月份申请的证书表示沮丧，课程工作人员指出他们正在积极努力加快处理进度。
   - 另一位成员承认了延迟并感谢工作人员的争取，询问证书发放的预期时间表。
- **Quiz 1 和 Quiz 2 的可用性**：几位成员寻求有关 **Quiz 1** 和 **Quiz 2** 可用性的信息，课程工作人员确认 Quiz 2 尚未发布，并提供了 Quiz 1 的链接。
   - 一位成员得到保证，他们可以在周五之后完成 Quiz 1，因为目前没有截止日期。
- **Quiz 的技术问题**：课程工作人员沟通了导致 Quiz 和证书处理延迟的意外技术问题。
   - 他们希望在未来一两周内解决这些问题。



**提及的链接**：<a href="https://forms.gle/c6Zz5kGPUzkNTQiq9">Quiz 1 - 推理时技术 w/ Xinyun Chen (1/27)</a>：说明：每个测验都基于完成情况，但我们鼓励您为了自己的学习而尽力而为！这些测验是检查您是否理解课程内容的绝佳方式...

  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1336653090845360168)** (2 条消息): 

> `Lecture 1 录制，专业字幕` 


- **Lecture 1 视频修复确认**：一名成员询问提供的 [YouTube 链接](https://www.youtube.com/live/g0Dwtf3BH-0) 是否为第一讲的修复版本。
   - 另一名成员确认这确实是带有**专业字幕 (professional captioning)** 的编辑后录像。
- **编辑录像详情**：标题为 'CS 194/294-280 (Advanced LLM Agents) - Lecture 1, Xinyun Chen' 的课程确认已配备专业字幕。
   - 这确保了观众能获得更好的可访问性和理解度。



**提到的链接**：<a href="https://www.youtube.com/live/g0Dwtf3BH-0">CS 194/294-280 (Advanced LLM Agents) - Lecture 1, Xinyun Chen</a>: 未找到描述

  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1336468512239259772)** (10 条消息🔥): 

> `迁移到 Embed v3 Light，Cohere Moderation Model，Chat 功能费用，Cohere Free API` 


- **寻求 Embed v3 迁移建议**：一位用户询问如何将现有的 float 生成结果从 **embed v3** 迁移到 **embed v3 light**，特别是是否可以直接删除多余维度，还是需要重新生成整个数据库。
   - *未提供直接回复*，但该问题突显了关于迁移过程的常见疑虑。
- **希望 Cohere 提供 Moderation Model**：一名成员表示希望 Cohere 提供 **moderation model**，以减少对美国服务的依赖。
   - 这种情绪反映了 AI 领域对更多本地化解决方案的需求。
- **Chat 功能付费订阅咨询**：一位用户询问是否有针对聊天功能的**按月付费选项**，并指出其主要兴趣在于聊天功能而非产品开发。
   - 另一名成员告知他们存在需要付费的 **production API**。
- **估算 Chat 交互成本**：一位用户表示不确定如何估算 **chat interactions** 的成本，强调他们有兴趣将该服务用于个人研究而非产品开发。
   - 一条评论指出，在做出任何决定之前，可以使用 **free API** 来测试功能。
- **直播活动公告**：一名成员分享了一个**直播活动**的链接，同时提供了 Discord 活动链接和 Google Meet 链接。
   - 这传递了持续的社区参与和直接互动的机会。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1336544063712923658)** (2 条消息): 

> `Conversational Memory，Java API 使用，支持工单` 


- **用户寻求 Conversational Memory 方面的帮助**：一名成员对 AI 的回复在不同请求之间没有关联感到沮丧，并寻求关于使用 **conversational memory** 的指导。
   - 他们提到使用 **Java code** 连接 API，并且目前处于 [试用免费定价计划](https://discord.com/channels/954421988141711382/1336548080677294191)。
- **支持工单创建确认**：一名成员感谢另一名成员创建了与其问题相关的支持工单。
   - 他们提供了 [工单链接](https://discord.com/channels/954421988141711382/1336548080677294191) 以供参考。


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1336708245468876902)** (2 条消息): 

> `规则执行，道歉与确认` 


- **关于规则执行的严厉提醒**：一名成员强调，未来的违规行为可能会导致**封禁 (ban)**，重申了遵守社区规则的重要性。
   - 这一声明清楚地突显了维护社区秩序的承诺。
- **对违反规则的道歉**：另一名成员为其之前的行为道歉，承认了遵守规则的重要性。
   - 这一回应显示了纠正错误并符合社区预期的意愿。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1336646493347516456)** (10 条消息🔥): 

> `tinygrad 0.10.1 问题、NixOS 特性、编译器标志与警告、调试改进` 


- **tinygrad 0.10.1 运行出错**：在将 **tinygrad** 升级到 **0.10.1** 时，用户遇到了因未知的重定位类型 4（relocation type 4）导致测试失败并抛出 *NotImplementedError* 的情况。
   - *类型 4 表示调用了不支持的外部函数*；当前使用的版本是 **19.1.6**。
- **CLANG 是否为默认后端？**：有人询问在未指定的情况下运行测试时，**CLANG** 是否为默认后端。
   - 结论是该问题可能源于影响编译的 *Nix 特定行为*。
- **关于编译器警告**：**stderr** 中的警告引起了关注，特别是关于由于设置了 `NIX_ENFORCE_NO_NATIVE` 而跳过非纯标志（impure flag）**-march=native** 的警告。
   - 这些更改可能会改变编译标志，从而影响在旧 CPU 架构上的功能。
- **澄清 clang 用于 JIT 编译的用途**：一名成员解释说，移除 **-march=native** 通常是针对用户机器上的软件，而 **tinygrad** 使用 **clang** 作为 kernel 的 **JIT** 编译器，这简化了此类需求。
   - 这表明移除此类标志不应适用于 **tinygrad** 的上下文。
- **改进 tinygrad 的调试**：一位贡献者提到 **PR #8902** 将使问题的调试变得更易于管理和操作。
   - 这表明项目正在进行持续改进，旨在解决观察到的复杂问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/ARM-software/abi-aa/blob/main/aaelf64/aaelf64.rst">abi-aa/aaelf64/aaelf64.rst at main · ARM-software/abi-aa</a>：Arm® 架构的应用二进制接口 - ARM-software/abi-aa</li><li><a href="https://www.scs.stanford.edu/~zyedidia/arm64/index.html">A64</a>：未找到描述
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1336624892942090291)** (1 条消息): 

> `tinygrad 基础操作、kernel 实现` 


- **询问 tinygrad 基础操作数量**：一名成员询问 **tinygrad** 中存在多少个**基础操作**（base operations）。
   - 这一询问突显了对框架基础元素进行明确说明的需求。
- **寻找 kernel 实现源码**：有人请求获取与 **tinygrad** 相关的 **kernel 实现**位置信息。
   - 这表明了对理解底层代码库以实现更好使用和开发的兴趣。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1336526577768599636)** (3 条消息): 

> `API 模型要求、模型排行榜、身份验证机制` 


- **将模型添加为 API 端点的指南**：一名成员指出了将新模型添加到排行榜的[说明](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/CONTRIBUTING.md)，其中包括要求和设置细节。
   - 值得注意的是，虽然可能需要**身份验证**，但 API 端点应对公众开放。
- **身份验证与可访问性讨论**：一名成员提到模型的 API 端点需要**身份验证**、计费、注册或令牌。
   - 然而，预期是最终将向**公众**开放访问，以确保更广泛的可用性。



**提到的链接**：<a href="https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/CONTRIBUTING.md">gorilla/berkeley-function-call-leaderboard/CONTRIBUTING.md at main · ShishirPatil/gorilla</a>: Gorilla: 用于函数调用 (Tool Calls) 的 LLM 训练与评估 - ShishirPatil/gorilla

  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1336533785147342868)** (1 条消息): 

> `Raft 方法、Llama 3.1 7B、训练合成数据、微调、RAG 实现` 


- **在有限数据下使用 Raft 方法**：一名成员询问大约 **1000 名用户的数据**是否足以使用 **Raft 方法**对 **Llama 3.1 7B** 进行训练。
   - 他们还询问在应用 **RAG**（Retrieval-Augmented Generation）之前，是否需要加入**合成数据**来增强训练。
- **关于数据量的担忧**：讨论强调了关于 **1000 名用户的数据**是否能为有效的模型训练提供足够多样性的担忧。
   - 一名成员建议，**合成数据**对于填补空白和改善训练结果可能是必要的。


  

---

### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1336787564102684753)** (2 条消息): 

> `Chain of Agents, DSPy Way` 


- **在 DSPy 中引入 Chain of Agents**: 一位用户分享了一个 **DSPy Way** 的 **Chain of Agents** 示例，并链接到了[这篇文章](http://x.com/i/article/1887191253370216450)。
   - 他们还引用了关于该主题的原始论文，可以在[这里](https://openreview.net/pdf?id=LuCLf4BJsr)查看。
- **请求 Git repository**: 另一位用户询问是否有与所讨论的 Chain of Agents 示例相关的 **Git repository**。
   - 这一询问突显了社区对该概念实际实现的兴趣。



**提到的链接**: <a href="https://x.com/JuiceSharp/status/1887209289649168467">来自 Sergii Guslystyi (@JuiceSharp) 的推文</a>: http://x.com/i/article/1887191253370216450

  

---


---


{% else %}


> 完整的逐频道详情已针对电子邮件进行了删减。 
> 
> 如果您想查看完整详情，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}