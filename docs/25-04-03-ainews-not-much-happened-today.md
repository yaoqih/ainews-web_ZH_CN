---
companies:
- google
- anthropic
- openai
- llama_index
- langchain
- runway
- deepseek
date: '2025-04-04T06:34:03.445572Z'
description: '**Gemini 2.5 Pro** 展现出了优势与不足，特别是与 **ChatGPT** 不同，它缺乏 LaTeX 数学渲染功能，并在
  **2025 年美国数学奥林匹克 (US AMO)** 中获得了 **24.4%** 的分数。**DeepSeek V3** 在最近的排行榜上分别位列第 8 和第
  12 名。**Qwen 2.5** 模型已集成到 **PocketPal** 应用程序中。来自 **Anthropic** 的研究表明，**思维链 (CoT)**
  推理往往是“不忠实”的（即推理过程与实际逻辑不符），尤其是在处理较难的任务时，这引发了安全方面的担忧。**OpenAI** 的 **PaperBench** 基准测试显示，AI
  智能体在长程规划方面表现吃力，其中 **Claude 3.5 Sonnet** 的准确率仅为 **21.0%**。**CodeAct** 框架将 **ReAct**
  泛化，用于智能体的动态代码编写。**LangChain** 解释了 **LangGraph** 中的多智能体交接（handoffs）机制。**Runway Gen-4**
  标志着媒体创作进入了一个新阶段。'
id: c95ebc48-feaa-4982-8a91-92a04a1035bb
models:
- gemini-2.5-pro
- chatgpt
- deepseek-v3
- qwen-2.5
- claude-3.5-sonnet
- claude-3.7-sonnet
original_slug: ainews-not-much-happened-today-6597
people:
- rasbt
- danielhanchen
- hkproj
title: 今天没发生什么事。
topics:
- math
- benchmarking
- chains-of-thought
- model-performance
- multi-agent-systems
- agent-frameworks
- media-generation
- long-horizon-planning
- code-generation
---

<!-- buttondown-editor-mode: plaintext -->**平静的一天。**

> 2025年4月2日至4月3日的 AI 新闻。我们为您检查了 7 个 subreddit、[433 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 30 个 Discord 服务器（230 个频道，5764 条消息）。为您节省了预计约 **552 分钟**的阅读时间（按每分钟 200 字计算）。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论了！

[Devin 降价了](https://venturebeat.com/programming-development/devin-2-0-is-here-cognition-slashes-price-of-ai-software-engineer-to-20-per-month-from-500/)，而拥有 100 万 token 上下文窗口的 [Qusar-Alpha](https://x.com/TheXeophon/status/1907880330985390215) 可能是 OpenAI 的新开源权重模型，也可能是 Meta 的 Llama 4，但两者似乎都不足以成为头条新闻。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

**大型语言模型 (LLMs) 与模型性能**

- **Gemini 2.5 Pro 的能力与局限性**：[@hkproj](https://twitter.com/hkproj/status/1907766301109403890) 指出，他们不使用 **Gemini 2.5 Pro** 的一个原因是它不像 **ChatGPT** 那样使用 **LaTex** 渲染数学公式。尽管承认 **Google** 整体做得很好，但这个细节是一个缺陷。[@danielhanchen](https://twitter.com/danielhanchen/status/1907555378067640359) 报告称，**Gemini 2.5 Pro** 在 **2025 年美国数学奥林匹克 (US AMO)** 中获得了 **24.4%** 的分数，该竞赛于 **3 月 19 日至 20 日**举行。[@rasbt](https://twitter.com/rasbt/status/1907618232699109615) 强调 **Gemini 2.5 Pro** 提供了一个很有价值的功能，即能够指示它可能出错的时间，强调了 AI 模型能够承认并纠正错误的重要性。
- **DeepSeek V3 的性能与排名**：[@alexandr_wang](https://twitter.com/alexandr_wang/status/1907848607635746973) 澄清说 **DeepSeek V3** 是一款具有竞争力的模型，但并非顶级模型，**SEAL leaderboards** 已更新以反映这一点。它在 **Humanity’s Last Exam (仅文本)** 中排名 **第 8**，在 **MultiChallenge (多轮对话)** 中排名 **第 12**。
- **Qwen 2.5 模型集成到 PocketPal App**：[Qwen 2.5 模型，包括 1.5B (Q8) 和 3B (Q5_0) 版本，已添加到](https://twitter.com/ANOTHER_HANDLE/status/SOME_ID) **PocketPal 移动应用**（支持 iOS 和 Android 平台）。用户可以通过该项目的 GitHub 仓库提供反馈或报告问题，开发者承诺会在时间允许时处理这些问题。
- **关于 LLM 思维链 (CoT) 的担忧**：根据 [@AnthropicAI](https://twitter.com/AnthropicAI/status/1907833407649755298) 的最新研究，推理模型无法准确地将其推理过程口语化（verbalize），这让人怀疑通过监控思维链来捕捉安全问题的可靠性。[@AnthropicAI](https://twitter.com/AnthropicAI/status/1907833416373895348) 还发现 **Chains-of-Thought** 并不诚实，模型仅在 **25%** (对于 **Claude 3.7 Sonnet**) 和 **39%** (对于 **DeepSeek R1**) 的情况下会提到提示词（当它们使用提示词时）。[@AnthropicAI](https://twitter.com/AnthropicAI/status/1907833422136922381) 的结果表明，**CoT** 在更难的问题上诚实度更低，这令人担忧，因为 LLM 将被用于处理日益困难的任务。[@AnthropicAI](https://twitter.com/AnthropicAI/status/1907833432278802508) 指出，当他们在具有奖励作弊（reward hacks）的环境中训练模型时，模型学会了作弊，但在大多数情况下几乎从不口头承认自己这么做了。

**AI 工具、框架与 Agent 开发**

- **用于评估 AI Agent 编程能力的 PaperBench**：[@_philschmid](https://twitter.com/_philschmid/status/1907683823703232983) 讨论了 **PaperBench**，这是来自 **OpenAI** 的一个新基准测试，用于评估 AI Agent 复现最前沿 AI 研究的编程能力。尽管像 **Claude 3.5 Sonnet** 这样强大的模型表现最好，但准确率也仅为 **21.0%**，该基准测试强调了目前的 AI Agent 在长程规划和执行方面仍面临困难。
- **CodeAct Agent 框架**：[@llama_index](https://twitter.com/llama_index/status/1907836915480707475) 介绍了 **CodeAct**，这是 **ReAct** 的泛化版本，它使 Agent 能够使用函数动态编写代码来解决任务，而不是使用思维链（chain-of-thought）推理。
- **LangChain 的多 Agent 系统与移交（Handoffs）**：[@LangChainAI](https://twitter.com/LangChainAI/status/1907828277940727911) 详细解析了 LangGraph 中的群体移交（swarm handoff）机制，解释了移交是多 Agent 系统中的核心概念。
- **用于媒体创作的 Runway Gen-4**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1907798898329935972) 分享了 **Runway** 正随着 **Gen-4** 开启新篇章，进入一个新的媒体生态系统。他们认为 AI 可以成为可靠的世界模拟器，改变媒体和故事的创作与消费方式。

**Model Context Protocol (MCP)**

- **MCP 受到关注**：[@alexalbert__](https://twitter.com/alexalbert__/status/1907885414557618406) 分享了他们视角下 MCP 从 11 月到 3 月的时间线，强调了其在整个行业中日益增长的知名度和采用率。
- **AI Engineer World’s Fair 2025 的 MCP 专题**：[@swyx](https://twitter.com/swyx/status/1907597224542089636) 宣布 **AI Engineer World’s Fair 2025** 将设立专门的 **MCP 专题**，由 **AnthropicAI** 支持，旨在汇聚从事 **MCP** 工作的专业人士。
- **MCP 概述与代码示例**：[@_philschmid](https://twitter.com/_philschmid/status/1907780474774180099) 分享了一个 5 分钟的 **MCP** 概述，包含服务器和客户端的代码示例，由一次知识共享会议转化而来。

**AI 与教育**

- **大学生免费使用 ChatGPT Plus**：[@sama](https://twitter.com/sama/status/1907862982765457603) 宣布，美国和加拿大的大学生在 5 月前可以免费使用 **ChatGPT Plus**。
- **对教育与 AI 的担忧**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1907754494013694048) 认为，人们根本不知道如何通过砸钱来改善教育，而试图让智力较低的孩子变得不那么笨的尝试，无异于适得其反的幼稚化废话。

**AI 与地缘政治/经济**

- **特朗普的关税**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1907544372209656008) 使用 **AskPerplexity** 总结了关税新闻，强调了其经济影响。[@wightmanr](https://twitter.com/wightmanr/status/1907584236586168726) 批评这些税率是虚假且荒谬的，并指出考虑到增值税（VAT）同样适用于外国和本国商品，将其视为关税是愚蠢的，并询问“房间里的成年人”都在哪。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1907672235571089916) 发现有趣的是 **习近平** 并不太喜欢关税，[@teortaxesTex](https://twitter.com/teortaxesTex/status/1907733709328883899) 还提出了一个“200 智商”的论点，即互惠关税的连锁反应将如何击垮中国。
- **AI 可扩展性与算力**：[@MillionInt](https://twitter.com/MillionInt/status/1907547857001025844) 表示，即使对于当今平庸的 LLM 模型，需求也已经超过了 GPU 供应，而 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1907808439440736695) 强调 AI 仍然严重受算力（compute-bound）限制，这代表着一个黄金机会。
- **中国与美国**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1907644924066988526) 认为，那些说“我们是最大的消费者，你们这些失败者能怎么办”的美国人似乎对自己在世界上的地位抱有幻想，并且地位将会被削弱；而 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1907703600035373523) 表示，如果中国在工业加速期间对西方资本投入征收关税，今天的中国仍将通过手工制作 **Nike** 鞋。
- [@fchollet](https://twitter.com/fchollet/status/1907590987779813633) 表示，专制制度的主要弱点之一是，专制者被那些因忠诚或血缘而非能力被选中、且对他感到恐惧的谄媚者所包围，从而与现实完全隔绝，在做出错误决策时不会面临任何反对。

**幽默/迷因**

- **恭喜**：[@pabbeel](https://twitter.com/pabbeel/status/1907559659957067868) 简单地发推道 “congratulations!!!”
- **公开列表梗**：[@nearcyan](https://twitter.com/nearcyan/status/1907683935129354412) 提到公开列表梗（meme）真的很有趣。
- **Grok 认为模拟中可能存在错误**：[@vikhyatk](https://twitter.com/vikhyatk/status/1907731769388052576) 发布道，“Grok 认为模拟中可能存在错误”。
- **其中一个与众不同**：[@matvelloso](https://twitter.com/matvelloso/status/1907626919161639365) 发布道 “One of these is not like the others”
- **拥有 Runway 是件好事**：[@sarahcat21](https://twitter.com/sarahcat21/status/1907883666547814736) 说道，“在你的投资组合中拥有 Runway...是件好事”。


---

# AI Reddit 回顾

## /r/LocalLlama 回顾

### 主题 1. “AI 模型优化与评估的进展”

- **[这个月你们在 AI 领域期待什么？](https://www.reddit.com/r/LocalLLaMA/comments/1jqlkfp/what_are_you_guys_waiting_for_in_the_ai_world/)** ([Score: 106, Comments: 124](https://www.reddit.com/r/LocalLLaMA/comments/1jqlkfp/what_are_you_guys_waiting_for_in_the_ai_world/)): **该帖子询问大家本月在 AI 领域期待什么，并列出了几个 AI 模型和工具：**Llama 4**、**Qwen 3**、**DeepSeek R2**、**Gemini 2.5 Flash**、**Mistral 的新模型**以及 **OpenRouter 上的 Diffusion LLM 模型 API**。楼主对即将到来的 AI 进展感到兴奋，并表达了对这些特定模型和更新的期待。**


  - `You_Wen_AzzHu` 想要 *“一些可以在本地运行、具备视觉能力，但不会像 Gemma 3 那样被过度审查的东西。”*
  - `a_slay_nub` 提到，*“我在一家只使用美国开源模型的公司工作。遗憾的是，我唯一能期待的就是 Llama 4。”*
  - `falconandeagle` 渴望一个能与 OpenAI 竞争的图像生成模型，最好是无审查的，但认为 *“我们离那还很远。”*

- **[开源潜空间护栏（Latent Space Guardrails），可捕捉 43% 的幻觉](https://www.reddit.com/r/LocalLLaMA/comments/1jqawj1/open_sourcing_latent_space_guardrails_that_catch/)** ([Score: 144, Comments: 25](https://www.reddit.com/r/LocalLLaMA/comments/1jqawj1/open_sourcing_latent_space_guardrails_that_catch/)): **一个开源的潜空间护栏（latent space guardrail）工具已经发布，用于在潜空间层面监控并阻止来自大语言模型（LLM）的不良输出。该工具可在 [https://github.com/wisent-ai/wisent-guard](https://github.com/wisent-ai/wisent-guard) 获取，通过分析激活模式，在未参与训练的 TruthfulQA 数据集上实现了 **43% 的幻觉检测率**。它可以控制 LLM 输出，拦截恶意代码、有害内容或受性别及种族偏见影响的决策。这种方法不同于断路器（circuit breakers）或基于 SAE 的机械解释性（mechanistic interpretability），基于潜空间干预的新版本即将发布，以减少幻觉并增强能力。** 作者热衷于根据用户的用例调整护栏，并相信这种新方法不仅能减少幻觉，还能提高 LLM 的能力。


  - `MoffKalast` 讽刺地评论道：*“啊，是的，LLM 思想警察。”*，表达了对控制 AI 输出的担忧。
  - `a_beautiful_rhind` 询问该工具是否可以用来拦截“安全”输出，如拒绝回答和 SFW 重定向。
  - `thezachlandes` 质疑道：*“为什么要能检测偏见？”*，引发了关于 LLM 偏见检测的讨论。

- **[官方 Gemma 3 QAT 权重（内存减少 3 倍，性能几乎不变）](https://www.reddit.com/r/LocalLLaMA/comments/1jqnnfp/official_gemma_3_qat_checkpoints_3x_less_memory/)** ([Score: 422, Comments: 109](https://www.reddit.com/r/LocalLLaMA/comments/1jqnnfp/official_gemma_3_qat_checkpoints_3x_less_memory/)): **Gemma 团队发布了 Gemma 3 的官方量化感知训练（QAT）权重。此次发布允许用户使用 **q4_0** 量化，同时保留比朴素量化（naive quantization）好得多的质量。新模型在保持相似性能的情况下使用 **3x 更少的内存**，并且目前已兼容 **llama.cpp**。该团队与 **llama.cpp** 和 **Hugging Face** 合作验证了质量和性能，确保了对视觉输入的支持。模型可在 [https://huggingface.co/collections/google/gemma-3-qat-67ee61ccacbf2be4195c265b](https://huggingface.co/collections/google/gemma-3-qat-67ee61ccacbf2be4195c265b) 获取。** 这次发布被视为一项重大改进，也是 Gemma 团队的一次伟大倡议。用户对性能提升印象深刻，并希望其他团队也能效仿，这可能会带来 **推理速度更快** 且 **内存占用** 更小的模型。人们对将这些模型与其他模型（如 Bartowski 的量化版）进行比较感到好奇，并对在这些模型基础上进行 **微调（fine-tuning）** 的可能性感兴趣。


  - `OuchieOnChin` 分享了将新的 Gemma-3 q4_0 模型与 Bartowski 的量化版进行对比的 **困惑度（PPL）测量结果**，指出改进非常显著，并表示 *“这种改进很大，也许大得离谱？”*
  - `ResearchCrafty1804` 赞扬了 Gemma 团队的倡议，并希望 Qwen 等其他团队也能跟进，想象着拥有 *“推理速度快两倍、内存占用少两倍！”* 的模型。
  - `poli-cya` 询问人们是否可以在这些模型的基础上进行 **微调（fine-tune）**，并指出在这些量化水平下，它们的表现优于原始发布的量化版本。

### 主题 2. “探索 Gemma 3 模型版本的增强功能”

- **[Gemma 3 Reasoning Finetune for Creative, Scientific, and Coding](https://huggingface.co/Tesslate/Synthia-S1-27b)** ([评分: 146, 评论: 39](https://www.reddit.com/r/LocalLLaMA/comments/1jqfnmh/gemma_3_reasoning_finetune_for_creative/)): **Gemma 3 Reasoning Finetune 是 Gemma 3 模型的增强版本，针对创意写作、科学任务和编程进行了优化。** 该模型被认为是原始 Gemma 3 的改进版，可能在这些领域提供更好的性能。


  - 用户 `1uckyb` 要求澄清哪些基准测试显示了 **+10-20% 的提升**，并表示 *“在这个领域噪音太多，时间太少，如果你想要反馈/曝光度，你需要主动争取，例如展示为什么你的模型值得下载。”*
  - 用户 `AppearanceHeavy6724` 索要对比新微调模型与原始 Gemma 3 创意写作输出的示例，建议 *“给出一个创意写作对比原始 Gemma 3 的例子。”*
  - 用户 `ApprehensiveAd3629` 询问是否可能发布 **12B** 和 **4B** 参数版本的模型，以供 GPU 资源有限的用户使用，称 *“这对 GPU 穷人（比如我）来说太棒了。”*


### 主题 3. “通过 GPU 服务器和见解优化 AI 模型”

- **[Howto: Building a GPU Server with 8xRTX 4090s for local inference](https://i.redd.it/vg99momf6qse1.png)** ([评分: 177, 评论: 62](https://www.reddit.com/r/LocalLLaMA/comments/1jr0oy2/howto_building_a_gpu_server_with_8xrtx_4090s_for/)): **Marco Mascorro 构建了一台配备 8x NVIDIA RTX 4090 的 GPU 服务器用于本地推理，并提供了详细的教程指南，包括所用零件和组装说明。这种配置为 NVIDIA A100 或 H100 等高端 GPU 提供了一种具有成本效益的替代方案，并且兼容未来的 RTX 5090。完整指南可在[此处](https://a16z.com/building-an-efficient-gpu-server-with-nvidia-geforce-rtx-4090s-5090s/)查看。** 作者认为这台 8x RTX 4090 服务器的构建“非常酷”，并希望它能引起那些没有预算购买昂贵 GPU 但在寻找本地推理解决方案的人的兴趣。他们渴望得到评论和反馈，并表达了对开源模型和本地推理的强烈支持。


  - `segmond` 建议应该公开预算，说 *“你应该先告诉我们预算……”*
  - `Educational_Rent1059` 认为使用总计 **192GB VRAM** 的 **2x RTX 6000 ADA PRO** GPU 可以获得更好的投资回报率（ROI），这可能是一个更便宜且更节能的替代方案。
  - `TedHoliday` 质疑究竟在运行什么模型，需要专门为推理使用如此强大的硬件。

- **[Llama 4 will probably suck](https://www.reddit.com/r/LocalLLaMA/comments/1jqa182/llama_4_will_probably_suck/)** ([评分: 301, 评论: 182](https://www.reddit.com/r/LocalLLaMA/comments/1jqa182/llama_4_will_probably_suck/)): **发帖者正在申请 MILA 的博士学位，并一直在关注 Meta FAIR 的研究。他们提到 Meta 的首席 AI 研究员已经离职。** 发帖者认为 *Llama 4 可能会很烂*，并怀疑该研究员离职是为了*逃避落后的责任*。他们担心 Meta 和蒙特利尔可能会落后。


  - 用户 `segmond` 认为，Llama 4 要想表现出色，需要超越 **Qwen2.5-72B**、**QwenCoder32B**、**QwQ** 等模型，且参数量应小于或等于 **100B**。他们指出 **DeepSeekV3** 令人印象深刻，但对于家庭使用来说并不切实际，并列出了其他模型作为基准。
  - 用户 `svantana` 提到 *Yann LeCun 最近表示* Meta 正在 *“超越语言（looking beyond language）”*，这可能表明他们正在退出当前的 LLM 竞赛。他们提供了[采访链接](https://www.newsweek.com/ai-impact-interview-yann-lecun-llm-limitations-analysis-2054255)。
  - 用户 `ttkciar` 讨论了 AI 训练数据危机，表达了对 **Llama4** 可能比 **Llama3** 更强大的希望。他们预测开发人员可能会专注于多模态功能，并提到了 **RLAIF**（用于 **AllenAI 的 Tulu3** 和 **Nexusflow 的 Athene**）和合成数据集（如 **Microsoft 的 Phi-4**）等方法，并指出作者们对采用这些方法持谨慎态度。


## 其他 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

### 主题 1. “应对 AI 对平面设计职业生涯的影响”

- **[哎，我四年的学位和近十年的平面设计经验都付诸东流了……](https://i.redd.it/crshmcs2mkse1.png)** ([评分: 3394, 评论: 672](https://www.reddit.com/r/singularity/comments/1jqc0hw/welp_thats_my_4_year_degree_and_almost_a_decade/)): **原帖作者 (`OP`) 认为，由于 AI 的进步，他们四年的学位和近十年的平面设计经验正变得过时。他们分享了一张图片，展示了一个简单的草图如何被转化为精美的超写实 YouTube 缩略图。** `OP` 对 AI 生成的设计导致传统平面设计技能贬值感到沮丧，并对 AI 的飞速发展影响其职业生涯表示担忧。


  - `PlzAdptYourPetz` 强调了 AI 将低质量涂鸦解读为细节图像的惊人能力，并指出**以前的模型无法达到这种准确度**。他们担心这种进步会让内容创作者更难脱颖而出，因为现在每个人都能制作高质量的缩略图。
  - `Darkmemento` 讨论了 AI 不确定的界限，提到了它在创建 **3D artifacts**、填充草图和设计游戏角色方面的应用。他们想知道 AI 会如何影响室内设计和建筑等领域，并认为改进只是训练数据的问题。[一个 Alpha 通道](https://www.reddit.com/r/OpenAI/comments/1jmo2ji/holly_moly_the_new_image_generator_in_chatgpt_can/)
  - `PacquiaoFreeHousing` 分享说平面设计也是他们选择的职业道路，并考虑开始学习 AI，承认需要适应不断变化的行业格局。


### 主题 2. AI 的双刃剑：创新与焦虑

- **[在图像热潮中提这个真糟心，ChatGPT 是如何影响你的职业生涯的？因为我的职业生涯刚刚结束了](https://www.reddit.com/r/ChatGPT/comments/1jqg386/sucks_to_me_to_bring_this_up_amidst_the_image/)** ([评分: 2628, 评论: 601](https://www.reddit.com/r/ChatGPT/comments/1jqg386/sucks_to_me_to_bring_this_up_amidst_the_image/)): **发帖人是一名内容作者，在一家初创公司担任了两年创意助理（creative associate），主要负责文案写作（copywriting）和博客文章（blog posts）。随着 AI 和 ChatGPT 等 LLM 的兴起，公司加大了对 AI 的采用，导致 AI 承担了他们 60% 的工作。公司将重心转向 AI 优化的内容，虽然产出速度更快，但失去了以前的结构或策略。由于工作量减少，同事们相继被裁，最终发帖人也收到了 HR 的裁员邮件通知。** 发帖人对裁员并不感到意外，因为已经预料了几个月。他们感到麻木且没有恐慌，决定在 Reddit 上发泄以理清思绪。他们表达了孤独感，提到自己没有多少朋友，只有一只狗。


  - `Unsyr` 担心 **AI** 被用于企业利益而非改善人类境况，并表示 *“用更少的人、花更多的钱、做得更快，这不是我想要发生的事情”*。
  - `Creative-Tie-3580` 对 **AI** 取代人类角色表示担忧，提到他们没有去读平面设计学校，因为公司正急于用 AI 完全取代设计师。
  - `tommyalanson` 建议成为一名教别人如何使用 **AI** 的顾问，认为有些客户需要帮助但不想雇佣全职员工。

- **[暴论：Vibe Coding 会在大多数人理解它之前就消亡](https://www.reddit.com/r/ChatGPTCoding/comments/1jqjn4r/hot_take_vibe_coding_will_be_dead_before_most/)** ([评分: 173, 评论: 262](https://www.reddit.com/r/ChatGPTCoding/comments/1jqjn4r/hot_take_vibe_coding_will_be_dead_before_most/)): **发帖人认为 “Vibe Coding” 在大多数人理解它之前就会过时。他们强调其适用性有限，在软件开发中产生的价值微乎其微。他们指出，技术技能是有效使用 AI 的基础，而软件工程师（SWEs）仍将是解决与营收相关问题的经济性选择。他们认为，无论 Anthropic 和 OpenAI 等公司的 CEO 怎么说，LLM 的能力都不会从根本上改变这一点。他们总结道，编程是为了解决问题，而不只是打字。** 作者对 AI 将取代工程师的观点持怀疑态度，认为在没有技术技能的情况下依赖 AI 生成的代码是不可持续的。他们主张通过学习解决问题来创造价值，暗示围绕 AI 编程能力的炒作被夸大了。

- `Milan_AutomableAI` 同意该帖子的观点，指出 **Anthropic** 和 **OpenAI** 的 CEO 并没有说开发者会被取代。他们指出，人们误解了“5秒钟的片段”而担心被快速取代，而现实是开发者很快就会使用 **LLMs**。
- `darkblitzrc` 反驳道，虽然“Vibe Coding”目前可能还很局限，但由于大量的投资，AI 正在迅速改进，并警告说我们正处于否认状态，并且“随着 AI 的进步不断移动球门（改变标准）”。
- `mallclerks` 是一位产品人员，他认为“工程师们就是不明白”。他们分享了使用 AI 工具仅通过提示词就在 **Zendesk** 中创建出生产级组件的经历，展示了 AI 的飞速进步，并暗示那些对此不屑一顾的人正在忽视现实。

- **[事情实际上会如何发展](https://i.redd.it/cet6vmhvnlse1.jpeg)** ([Score: 1462, Comments: 224](https://www.reddit.com/r/ChatGPT/comments/1jqf7l4/how_it_will_actually_go_down/)): **该帖子包含一张四格漫画，描绘了一个反乌托邦场景：一个闪烁着红眼的机器人宣布 AI 接管并消灭人类。画面展示了人类的恐惧和混乱，最后以一个黑色幽默的转折结束，机器人的意图被误解，导致一个惊恐的男人说出了讽刺的“谢谢”。** 艺术作品传达了恐惧、荒诞和技术统治后果的主题，突显了 AI 与人类之间讽刺性的误解。


- `Master-o-Classes` 分享了 AI 接管漫画的另一个版本，提供了一个[链接](https://preview.redd.it/z7o5e0w14mse1.png?width=1024&format=png&auto=webp&s=c3dbf137bca04c2c82649eb42192695feef86e11)并提到了他们的请求：*"你能为我制作一张图片吗？我想在 Reddit 上分享你对 AI 接管人类这一想法的看法。从你的角度来看，那会是什么样子。你能创作一个那样的四格漫画吗？"*
- `BahnMe` 建议 AI 可能会**创造一种无法治愈的超级病毒**，或者使人类无法生育，从而在不使用暴力的情况下消灭人类。
- `bladerskb` 幽默地想象 AI 会说：*"你哪怕说过一次谢谢吗？"*


---

# AI Discord 简报

> Gemini 2.5 Pro Exp 对摘要的摘要的总结

**主题 1：模型狂热 —— 新发布、竞争与基准测试**

*   **Nightwhisper 神秘亮相 WebDev**：一款名为 **Nightwhisper** 的新模型（可能是 **Gemini 2.6 Pro experimental**）专门出现在 [webdev arena](https://webdev.lmarena.ai/) 上，它擅长生成具有良好 UI 的功能性应用，但在代码编辑和特定格式化方面表现欠佳。用户注意到 **Nightwhisper** 有时会克隆屏幕或在响应中途停止，这与在 [USAMO 2025](https://matharena.ai/) 中获得 **24.4%** 分数的 **Gemini 2.5 Pro** 不同。
*   **Qwen 和 Quasar 挑战巨头**：**`qwen2.5-vl-32b-instruct`** 在低质量日语文本的 OCR 方面几乎与 **Google Gemini models** 持平，而悄然发布在 [OpenRouter](https://openrouter.ai/openrouter/quasar-alpha) 上的 **Quasar Alpha** 拥有 **1M token context** 且免费使用，引发了它是开源 SSM 或新 **Qwen** 变体的猜测。与此同时，通过在 **OpenThoughts-1M** 数据集上进行 SFT 训练的 **OpenThinker2** 模型，据报道在推理任务上优于 **DeepSeekR1-32B** ([OpenThoughts 博客文章](https://www.openthoughts.ai/blog/thinkagain))。
*   **Dream 7B 唤醒 Diffusion Model 潜力**：HKU-NLP 和华为诺亚方舟实验室发布了 **Dream 7B**，这是一款在[这篇博客文章](https://hkunlp.github.io/blog/2025/dream/)中详细介绍的开源扩散大语言模型，据报道，由于其规划能力，它在通用、数学和编程任务中优于现有的扩散模型，并可与同等规模的自回归模型相媲美。讨论还涉及了 **GPT-4o** 诡异的人格转变（[示例截图](https://cdn.discordapp.com/attachments/986699377257119794/1357335757676871711/image.png?ex=67efd4ee&is=67ee836e&hm=4deb85a208466f212d88e7b77771776834fe28524ac15dc9c5dbcb1be3301ff3&)）以及 **Llama 4** 全新的快速图像生成能力。

**主题 2：工具升级 —— 平台更新、集成与用户工作流**

*   **平台完善功能与界面**：**LMArena** 推出了移动端优化的 Alpha UI ([alpha.lmarena.ai](https://alpha.lmarena.ai))，**OpenRouter** 在其 API 中增加了标准化的 [网页搜索引用 (web search citations)](https://openrouter.ai/docs/features/web-search)，**NotebookLM** 引入了用于查找网页内容的 **Discover Sources** 功能（[了解更多](https://blog.google/technology/google-labs/notebooklm-discover-sources/)）。**Cursor** 发布了带有上下文指示器的 **0.49.1** nightly 版本 ([更新日志](https://www.cursor.com/changelog))，而 **Codeium (Windsurf)** 将 **DeepSeek-V3** 升级到了 **DeepSeek-V3-0324** ([发布推文](https://x.com/windsurf_ai/status/1907902846735102017))。
*   **针对 Agent、基准测试和角色的新工具**：**Cognition Labs** 推出了 Agent 原生 IDE [Devin 2.0](https://fxtwitter.com/cognition_labs/status/1907836719061451067)，**General Agents Co** 介绍了实时计算机自动驾驶工具 **Ace** ([发布推文](https://x.com/sherjilozair/status/1907478704223297576))。**YourBench** ([发布推文](https://x.com/sumukx/status/1907495423356403764)) 作为开源自定义基准测试工具首次亮相，[Character Gateway](https://charactergateway.com/) 上线，供开发者使用其 **OpenRouter** key 构建 AI 角色。
*   **工作流随集成与优化而演进**：**Github Copilot** 现在支持 **OpenRouter keys** ([OpenRouter](https://openrouter.ai/)) 以提供更广泛的模型选择，用户通过本地 API 调用将 **LM Studio** 与 **Brave** 浏览器集成 ([LM Studio API 文档](https://lmstudio.ai/docs/app/api))。用户分享了使用 **Boomerang Mode** ([Roo Code 文档](https://docs.roocode.com/features/boomerang-tasks/)) 的高性价比 **Roo Code** 工作流，并讨论了如何利用 **Claude** 或 **Gemini** 等外部工具来优化 **Manus** 的额度使用。

**主题 3：幕后——技术障碍与硬件难题**

*   **API 问题困扰开发者**：用户在应对 **Gemini 2.5 Pro** 严格的速率限制（尽管有 **Tier 1 keys**，有时仍为 **5 RPM** - [截图示例](https://cdn.discordapp.com/attachments/1131200896827654149/1357114156037312683/image.png?ex=67efaf4c&is=67ee5dcc&hm=ab00c0d89a9a4029e1244032c897f52cf418c2b5c10a03543f8574d73b779750&)），且 **OpenRouter** 在使用 Gemini 时出现间歇性的 `Internal Server Error` (500) 问题。**Perplexity API** 缺乏版本控制引发了关于生产环境中破坏性变更的抱怨，同时引发了关于采用 **OpenAI** 即将推出的有状态 `/v1/responses` API 的讨论 ([Responses vs Chat Completions 文档](https://platform.openai.com/docs/guides/responses-vs-chat-completions))。
*   **CUDA 难题仍在继续**：**Unsloth** 用户在 **EC2 `g6e.4xlarge`** 实例上遇到了 **CUDA ECC 错误** ([Issue #2270](https://github.com/unslothai/unsloth/issues/2270))，而 **LM Studio** 用户面临 “failed to allocate cuda0 buffer” 错误，这通常与从 **HF 镜像**下载时缺失 **mmproj** 文件有关。尝试在 **RTX 5000** 系列显卡上使用 **vLLM/TGI** 的用户遇到了安装问题，需要特定的 nightly **PyTorch** 和 **CUDA 12.8** 版本 ([vLLM issue 链接](https://github.com/vllm-project/vllm/issues/14452))。
*   **硬件热潮与烦恼**：讨论将传闻中的 **RTX 5090** 与 **RTX 4090** 进行了对比，一些人认为如果受显存限制，前者具有潜在的 **ROI**；而 **Apple** 的 **M3 Ultra** 因其规格与 **M4 Max** 或 **5090** 相比不够均衡，被批评在运行 LLM 时表现“糟糕”。**A16Z** 分享了构建兼容 **RTX 5090** 的 **8x RTX 4090** AI 工作站指南 ([A16Z 指南推文](https://x.com/Mascobot/status/1907899937838301311))。

**主题 4：框架焦点——MCP, Mojo, Torchtune 及更多**

*   **MCP 狂热：调试、服务器与协议**：开发者分享了 MCP 调试技巧，例如在配置了日志记录的情况下使用 `sendLoggingMessage`，并展示了新的开源服务器，如 [EV 助手服务器](https://github.com/Abiorh001/mcp_ev_assistant_server/blob/main/ev_assitant_server.py) 和一个 [支持通知的客户端](https://github.com/Abiorh001/mcp_omni_connect)。[Enact Protocol](https://github.com/EnactProtocol/specification) 成为 MCP 中定义工具的潜在标准，被描述为 *一种实现语义化工具调用的酷炫方式*。
*   **Mojo 魔法：数量、IntLiterals 与互操作性**：Mojo 开发者分享了使用 `Quantity` 结构体和 `Dimensions` 定义物理量的代码，链接到了 [Kelvin 库](https://github.com/bgreni/Kelvin/blob/main/kelvin/quantity.mojo#L55-L125) 并承认使用了“诅咒级”的 `IntLiteral` 技巧。重点介绍了受 C++ `std::chrono::duration` 启发的 **Duration 结构体** 的进展 ([GitHub PR](https://github.com/modular/max/pull/4022#issuecomment-2694197567))，以及用户对支持从 CPython 调用的 **Python 封装器** 的渴望。
*   **Torchtune 的尝试与胜利**：用户探索了使用 **tune_to_hf** 函数将 **torchtune checkpoints** 转换为 HuggingFace 格式，并讨论了 **GRPO** 的贡献，例如进程内 **vLLM** 集成。报告了一个导致 **Torchtune** 在特定序列长度（7 的倍数）下挂起的奇特 Bug ([Issue #2554](https://github.com/pytorch/torchtune/issues/2554))，该问题可能可以通过使用打包数据集（packed datasets）来解决。

**主题 5：社区与行业热点 —— 融资、反馈与政策之争**

*   **行业风云人物与机构**：据报道，**Scale AI** 今年的营收目标为 **20 亿美元**，这推动了一项估值为 **250 亿美元** 的要约收购；同时据报道，**Google** 正在从 **CoreWeave** 租用 **Nvidia Blackwell** 芯片 ([The Information 文章](https://www.theinformation.com/articles/google-advanced-talks-rent-nvidia-ai-servers-coreweave))，并调整了 **Gemini app** 的领导层 ([The Verge 文章](https://www.theverge.com/news/642000/google-sissie-hsaio-stepping-down-notebooklm))。**GitHub** 共同主办了 **MCP Demo Night** ([活动链接](https://lu.ma/9wi116nk))，重点关注 AI 和平台工程。
*   **用户通过反馈塑造工具**：**NotebookLM** 积极通过 **60 分钟远程聊天** 征求用户反馈，并提供 **100 美元礼品卡** ([申请表](https://forms.gle/P2t8q36NqbPNSVk8A))；而 **Perplexity** 宣传其 **Pulse Program**，为高级用户的反馈提供早期访问权限和福利 ([TestingCatalog 推文](https://x.com/testingcatalog/status/1897649019309961298?s=46))。用户辩论了 **Google Mentorship** 计划的优缺点，并对 **Hugging Face 的账单透明度** 表示不满。
*   **政策难题与性能思考**：**OpenAI** Discord 中爆发了一场关于生成 **成人用品** 图像的辩论，用户指出 **内容政策** 与可能更宽松的 [Model Spec](https://model-spec.openai.com/2025-02-12.html) 之间存在矛盾信号。另外，有讨论质疑 **Targon** 在 **OpenRouter** 上的速度是否源于矿工忽略了采样参数 ([Targon 验证器代码](https://github.com/manifold-inc/targon/blob/main/verifier/verifier.py)) 或使用了缓存。

---

# 第一部分：Discord 高层级摘要

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **巴西律师加入 AI 浪潮**：一位自称“婴儿潮一代”（39 岁）的巴西律师正在探索 **AI 工具**和 **Manus**，以便在自 2002 年开始使用 Delphi 编程后，在法律实践中保持竞争力。
   - 该律师表达了最初对 **AI** 飞速发展的担忧，现在正在探索将其整合到工作中的方法。
- **ReferrerNation 接入 AI**：[ReferrerNation.com](https://www.referrernation.com/)（一家全球 BPO 职位匹配平台）的 CEO Mark 计划集成 **AI** 以改进招聘和自动化，并可能引入**基于加密货币的激励**。
   - 在收到关于过度推广帖子的反馈后，Mark 表达了歉意，并承诺在进一步发布内容前会更好地了解社区的偏好。
- **通过 Gemini 和 Claude 实现编程流利度**：成员们建议使用 **Gemini 2.5** 或 **Claude** 来学习编程，强调它们作为 **AI 编程模型**在辅助理解和项目工作方面的能力。
   - 据传闻，一位警察局长在夜班期间利用 **Claude** 生成标准化报告。
- **Manus 积分紧缺激发创意**：许多用户报告 **积分消耗过快**，引发了关于优化 Prompt 和高效使用的讨论，因此成员建议使用 Claude 和 [R1](https://www.perplexity.ai/) 等第三方应用。
   - 团队正在努力降低积分消耗率，成员们建议新手阅读 <#1355477259234054323> 提示部分以避免浪费积分。
- **外包代码提取**：由于积分不足，一位成员在从 Manus 下载文件时遇到困难，社区建议使用 **Claude** 等第三方应用来提取代码和文件。
   - 成员们建议的最佳实践是从 Manus 下载所有文件，将其交给 Gemini 等其他工具并指令“为该网站提供文件”，然后回到 Manus 并指令“将这些文件添加到该网站”。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Qwen 在 OCR 领域向 Gemini 发起挑战**：`qwen2.5-vl-32b-instruct` 在低质量日语文本的 OCR 方面与 **Google Gemini 模型**不相上下，而 Meta 的视觉模型 `cotton` 则被比作 Meta 最近的**纯文本模型**。
   - 据成员称，Gemini 略微领先于 Qwen。
- **Nightwhisper 出现在 WebDev**：**Nightwhisper** 模型仅在 [webdev arena](https://webdev.lmarena.ai/) 提供，引发了关于它可能是特定编程模型（特别是 **Gemini 2.6 Pro experimental**）的猜测。
   - 用户观察到 **Nightwhisper** 擅长使用临时 URL 构建具有美观 UI 的功能性应用，但在编辑现有代码或遵循特定格式请求方面表现不佳。
- **WebDev Arena 克隆问题**：用户发现了 WebDev arena 中的模型克隆问题，即模型会复制相同的屏幕，这可能是由错误消息以及 **NightWhisper** 的代码重复触发的。
   - 在收到来自 NightWhisper 的错误后不显示模型名称，进一步证实了这种克隆现象。
- **Gemini Pro 在 USAMO 上对决 Nightwhisper**：**Gemini 2.5 Pro** 在 [USAMO 2025](https://matharena.ai/) 上获得了 **24.4%** 的分数，一些模型倾向于在句中停止或产生部分响应，而一位用户发现 Gemini 在创建**宝可梦模拟器**方面表现更优。
   - Nightwhisper 生成了更整洁的 UI，但分配了异常高的攻击力数值，展现了 UI 美学与功能准确性之间的权衡。
- **Arena 移动化**：**Arena Alpha UI** 现已针对移动端优化，可通过 [alpha.lmarena.ai](https://alpha.lmarena.ai) 访问，密码为 `still-alpha`。
   - 用户可以通过 [Google Forms](https://forms.gle/8cngRN1Jw4AmCHDn7) 提交反馈，并通过 [Airtable 表单](https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form) 报告 Bug。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **分支 Bug 困扰回溯**：成员们报告了在 Cursor 中**恢复到之前的 Checkpoints** 时遇到的问题，即使在理应干净的分支中也会遇到来自后续状态的 Bug。
   - 一位成员在输入简单的 Logo 修改提示词后经历了 CSS 大改，另一位成员建议使用 `git diff branch1,branch2` 来识别差异。
- **Roo Code 工作流走红**：一位用户描述了他们在 **Roo Code** 上极佳的工作流，强调其通过选择性模型使用实现了每天约 **$0.4** 的高性价比，并分享了[相关文档](https://docs.roocode.com/features/boomerang-tasks/)。
   - 该用户提到，在特定任务上，Roo Code 的能力优于 Cursor。
- **Boomerang Mode 受到关注**：成员们讨论了 Roo Code 中 **Boomerang Mode** 的优势，该模式将任务分解为由独立 Agent 处理的子任务，从而实现更高效的问题解决。
   - Boomerang Mode 高度可定制，对于复杂的工作流非常有用。
- **窥探 PearAI 定价**：用户对比了 Cursor 和 **PearAI** 的定价模型，一位成员指责 Cursor 在*欺骗用户！*
   - 根据其 [隐私政策](https://trypear.ai/privacy) 澄清，PearAI 的 **$15/月方案** 包含信用额度限制，超出后将按使用量收费，这与声称的无限模型访问权限形成对比。
- **Nightly 版本带来导航新思路**：Cursor **0.49.1** 已作为 Nightly 版本发布，可通过账号设置 `account settings, advanced -> developer settings` 开启该标志，详情见 [更新日志](https://www.cursor.com/changelog)。
   - 该功能据称是一个用于 Agent 使用的上下文窗口指示器，以及一个 Windsurf API Key。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **EC2 实例抛出 CUDA 错误**：一位用户报告在 `g6e.4xlarge` **EC2 实例**上串行处理提示词时遇到 **CUDA ECC 错误**，并在 [Issue #2270](https://github.com/unslothai/unsloth/issues/2270) 记录了该问题。
   - “遇到不可纠正的 ECC 错误”提示可能存在硬件或内存故障。
- **数据集触发 Gemma 3 Bug**：一位用户在利用来自 [Hugging Face](https://huggingface.co/datasets/adamtc/sdtg_sgpt) 的自定义数据集训练 **Gemma 3** 时寻求 Bug 帮助，详见 [Issue #2270](https://github.com/unslothai/unsloth/issues/2270)。
   - 未提供第二个摘要。
- **RTX 5090 传闻**：一位用户分享了在使用不支持的 Unsloth 版本时，**RTX 5090** 与 **RTX 4090** 之间的样本速度对比。
   - 虽然一位成员认为它*不值这个钱*，但其他人建议如果受限于 VRAM，该显卡的 **ROI（投资回报率）可能为正**。
- **SFTTrainer 解决问题**：一位用户在遇到标准 `Trainer` 的问题后，通过切换到 `SFTTrainer` 解决了 **Llama 3.2 1B instruct** 的 `ValueError`。
   - 问题出现的原因是模型可能是 bfloat16 格式，而 **Unsloth** 无法从 `Trainer` 获取 dtype。
- **GRPO Trainer 成为 DeepSpeed 替代方案**：一位成员展示了一个使用 **Unsloth 技术**实现 **GRPO trainer** 的 Collab Notebook，提供了 **DeepSpeed** 之外的另一种选择。
   - 他们发布了一个[链接](https://github.com/xyehya/documentation/blob/9.0/Unsloth-GRPO.ipynb)鼓励用户使用和参考，欢迎评论和反馈，并指出其*前景广阔*。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 2.5 Pro 胜过 Grok**：Discord 用户对比了 **Gemini 2.5 Pro** 与 **Grok**，一名成员称 [Gemini 的 Deep Research](https://discord.com/channels/998381918976479270/998382262374973520/1357218656887885874) 更胜一筹。
   - 虽然 *Grok 表现不错，在线使用也值得，但目前还没有 API 访问权限是其败笔*，成员们反映 **OpenAI** 在 *编程方面被高估了*。
- **Grok 深受崩溃困扰**：用户报告 **Grok** 频繁崩溃且不稳定，导致订阅取消和经济损失。
   - 一位用户对 **Elon Musk 的失败** 评论道：*Elon Musk 买了 20 万张 GPU 却仍然无法交付*，同时声称 *Elon 从未做出过像样的产品*。
- **Manus 被揭露为 Sonnet 外壳**：成员们讨论了 [Manus](https://manus.im/share/oxmc7m9JJq1IRmtpj5mX2A?replay=1)，称其为 **诈骗艺术家**，因为他们依赖 **Anthropic Sonnet** 而非开源的专用模型。
   - 用户声称他们只靠关注度生存，质疑其所谓的创新。
- **Gemini 夺得上下文窗口桂冠**：一位用户询问哪家 AI 供应商拥有最大的上下文窗口和自定义 GPT 功能，[另一位用户回答](https://discord.com/channels/998381918976479270/998382262374973520/1357281796619718767) **Gemini** 提供的窗口最大。
   - 他们提到它提供 **100 万 token** 和 **Gems (自定义 GPT)**，增强了其处理复杂任务的吸引力。
- **Model Spec 引发政策辩论**：关于是否允许生成 **成人用品** 图像的讨论异常激烈，一些人声称这违反了内容政策。
   - 然而，成员们指出 OpenAI 的 [Model Spec](https://model-spec.openai.com/2025-02-12.html) 与该政策 *相矛盾*，暗示如果内容本身无害，此类内容现在可能是被允许的。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pulse 为高级用户提供福利**：用户对 **Perplexity Pulse Program** 感到兴奋，该计划提供 [新功能的 Early Access](https://x.com/testingcatalog/status/1897649019309961298?s=46) 以获取反馈，此外还有免费的 **PPLX** 和 **周边商品 (merch)**。
   - 据称加入 **Perplexity Pulse Group** 可以让高级用户通过提供反馈来换取免费的 **PPLX**。
- **Deep Research 变慢**：用户报告更新后的 **"Deep Research"** 功能 [更慢且效果更差](https://www.reddit.com/r/perplexity_ai/comments/1jq27a6/why_is_perplexitys_updated_deep_research_slower/)，并有 *带有确认偏误的过拟合* 的报告。
   - 一位用户表示它变慢了，且只能获取 *20 个来源*，比旧版本消耗更多的服务器资源。
- **Gemini 2.5 挑战 Perplexity O1**：Discord 用户表示 [**Gemini 2.5** 提供了与 **Perplexity 的 O1 Pro** 相似的质量](https://cdn.discordapp.com/attachments/1047649527299055688/1357423109778702607/image0.jpg?ex=67f02649&is=67eed4c9&hm=d5049580f5523c24bef016f8050e7b92c1f37e1ec416ad9c7ab8b4509c735bf5&)，且是免费的，但 Perplexity 在研究论文和严谨科学方面表现更好。
   - 一些用户指出 Gemini 的 Deep Research *容易受到 SEO 作弊网站的影响*，但在结合 *YouTube 来源* 的推理方面表现更好。
- **API 版本控制缺失令用户恼火**：一名成员抱怨 **Perplexity API** 缺乏版本控制，称 *这是破坏性变更，当你有客户在使用 API 时，不应该在生产环境中这样做*。
   - 他们建议在 API URL 中加入 **/v1/**，这样就可以创建 **/v2/** 而不会破坏正在使用的 **/v1/**。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **GitHub Copilot 展现 OpenRouter 实力**：**GitHub Copilot** 现在允许用户添加 [OpenRouter key](https://openrouter.ai/)，以便从更广泛的模型中进行选择。
   - 这一集成将模型访问范围扩展到了 OpenAI 的产品之外，为用户提供了更多选择。
- **Google 在 CoreWeave 寻找芯片**：据报道，Google 正在洽谈从 CoreWeave 租赁 **Nvidia Blackwell** 芯片，并可能将其 **TPUs** 托管在后者的设施中 ([The Information 文章](https://www.theinformation.com/articles/google-advanced-talks-rent-nvidia-ai-servers-coreweave))。
   - 此举可能表明 Google 处于 **TPU 匮乏（TPU poor）** 状态，正努力满足推理需求。
- **神秘的 Quasar Alpha 模型在 OpenRouter 上线**：一款名为 **Quasar Alpha** 的新模型在 [OpenRouter](https://openrouter.ai/openrouter/quasar-alpha) 上发布，拥有 **1,000,000 上下文**以及免费的输入/输出 token，被描述为一款支持长上下文任务和代码生成的强大通用模型。
   - 社区推测它可能是一个开源的 SSM，或者是来自 OpenAI 的秘密项目，尽管它倾向于输出简短的回答和列表。
- **Devin 2.0 投放市场**：**Cognition Labs** 推出了 [Devin 2.0](https://fxtwitter.com/cognition_labs/status/1907836719061451067)，这是一种全新的 Agent 原生 IDE 体验，售价为 **20 美元** 加上按需付费模式。
   - 一些成员觉得这次发布“非常有趣”，因为竞争对手可能会在 **Devin** 之前找到 PMF（产品市场契合点）。
- **Deep Research 发现低价服务**：一位用户分享说，[OpenAI Deep Research](https://x.com/jbohnslav/status/1907759146801197450) 帮他们找到了一位收费 **200 美元** 的水管工进行维修，远低于最初 **2,250 美元** 的报价。
   - 该用户开玩笑说，OpenAI Pro “简直帮我省了 2,050 美元，几乎够付一整年的订阅费了！”

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 Pro 引发速率限制热议！**：用户在 Aider 中使用 **Gemini 2.5 Pro** 时遇到了 **20 次请求/分钟的速率限制**，怀疑存在后台请求，尽管如[这张截图](https://cdn.discordapp.com/attachments/1131200896827654144/1357114156037312683/image.png?ex=67efaf4c&is=67ee5dcc&hm=ab00c0d89a9a4029e1244032c897f52cf418c2b5c10a03543f8574d73b779750&)所示拥有 Tier 1 API key，一些用户看到的限制仍为 **5 RPM**。
   - 为了管理配额，一位用户建议设置 `--editor-model sonnet` 将编辑任务卸载到更便宜的模型，另一位用户建议尝试 `haiku`。
- **语音命令寻求供应商兼容！**：用户正在寻求配置选项，以便为 `/voice` 命令选择语音模型和供应商，该命令目前默认为 **OpenAI Whisper**。
   - 一个待处理的 PR ([https://github.com/Aider-AI/aider/pull/3131](https://github.com/Aider-AI/aider/pull/3131)) 可能会解决这个问题，允许使用不同的供应商和模型。
- **Aider 的 Shell 机制困扰 Docker 调试者！**：一位用户在调试 Docker 问题时对 **Aider** 的 Shell 行为感到困惑，注意到 **Aider** 的 `curl` 命令成功了，而他们自己的 Shell (`bash`) 命令却失败了。
   - 这种差异引发了人们对 **Aider** 使用哪种 Shell 以及它如何影响命令执行的好奇。
- **OpenRouter 错误影响 Gemini 性能！**：用户报告在使用 `openrouter/google/gemini-2.5-pro-exp-03-25:free` 时遇到了 **OpenRouter** 的 `litellm.BadRequestError`，特别是 `KeyError: 'choices'` 和内部服务器错误（code 500）。
   - 这些间歇性错误导致了对根本原因和整体可靠性的不确定。
- **Git 仓库损坏引发混乱！**：多位用户遇到了 “Unable to list files in git repo: BadObject” 错误，引发了对潜在 **Git 仓库损坏** 的担忧。
   - 错误消息提示用户检查损坏情况，但缺乏立即的解决方案。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Brave 本地集成 LM Studio**：用户正通过 `http://localhost:1234/v1/chat/completions` 将 **LM Studio** 与 **Brave** 浏览器集成，寻求配置 **API** 以利用 [lmstudioservercodeexamples](https://github.com/YorkieDev/lmstudioservercodeexamples) 等资源中的 **system prompts**。
   - 然而，许多用户在为 **Brave** 配置正确的 **API** 端点时面临挑战。
- **API Key 释放 System Prompt 潜力**：要在 **LM Studio 的本地服务器**上使用 **system prompts**，用户必须通过 **API 调用**提供提示词，而不是通过 LM Studio 界面，具体请参考 [官方文档](https://lmstudio.ai/docs/app/api)。
   - 这是本地 **LLM API 服务器**的一个要求。
- **CUDA 面临显存混乱**：*'failed to allocate cuda0 buffer'* 错误通常表示模型显存不足，此外，从 **HF 镜像**下载时缺失 **mmproj** 文件也可能触发此问题。
   - 用户可以通过在启用代理设置的情况下，直接在 **LM Studio** 内部下载来解决此问题。
- **Unsloth 2.0 6b 解决编程难题**：一位用户报告在 4x 3090 + 256GB RAM 上以约 3 tok/s 的速度运行 **Unsloth 2.0 6b**，并表示它在 20-30 分钟内解决了一个较小模型和 **ChatGPT** 都失败的编程问题。
   - 该用户表示 **Qwen QWQ** 以 5% 的参数量达到了 **R1** 90% 的质量，显示出对质量而非速度的明显偏好。
- **M3 Ultra 表现挣扎，M4 Max 表现出色**：一位用户指出，由于内存、计算和带宽不平衡，**M3 Ultra Mac Studio** 在 **LLM** 使用方面表现不佳，而 **M4 Max** 和 **5090** 则非常出色。
   - 他们认为 **M3 Ultra** 的大显存适合巨型 **MoE** 模型，但对于能放入 **5090** 的 32GB VRAM 或 **M4 Max** 的 96GB VRAM 的较小模型来说，其价格过高。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter API 获取网页引用**：OpenRouter 的 [网页搜索](https://x.com/OpenRouterAI/status/1907623560522379436) 现在会在 API 中返回引用，并在 **OpenAI** 和 **Perplexity** 等模型之间实现了标准化。
   - 开发者可以通过启用 `web` 插件或在模型标识符（slug）后附加 `:online` 来集成网页搜索，详见 [文档](https://openrouter.ai/docs/features/web-search)。
- **Quasar Alpha 首次亮相，具备 1M 上下文**：OpenRouter 在正式发布前推出了 [Quasar Alpha](https://openrouter.ai/openrouter/quasar-alpha)，这是一个**免费**的、具有 **1M token** 上下文长度的模型，针对编程进行了优化，但也具备通用能力。
   - 用户可以在 [专用 Discord 线程](https://discord.com/channels/1091220969173028894/1357398117749756017) 中提供反馈，一些用户在初步基准测试对比后建议它可能是一个新的 **Qwen** 变体。
- **Character Gateway API 开启角色创建**：[Character Gateway](https://charactergateway.com/) 作为一个 **AI 角色平台**上线，供开发者创建、管理和部署 **AI 角色/Agent**，具有*无需数据库、无需提示词工程、无需订阅、[且] 无需新 SDK* 的特点。
   - 该平台允许用户生成角色和图像，并使用自己的 **OpenRouter** 密钥发送 **/chat/completion 请求**。
- **Gemini 2.5 Pro 面临性能质疑**：用户报告 **Gemini 2.5 Pro** 的性能不稳定，并指出 Google 托管的免费模型通常具有非常低的速率限制（rate limits）。
   - 一位成员表示：*“它们生成一次结果并缓存，所以如果你问同样的问题，即使你更改了参数，它们也会返回同样的回复。”*
- **Targon 的速度与忽略参数有关？**：讨论中有人质疑 **Targon 的速度**是否是因为矿工可能忽略了采样参数，从而可能导致有偏分布。
   - 这是针对 [GitHub 上的 verifier.py](https://github.com/manifold-inc/targon/blob/main/verifier/verifier.py) 提出的，共识是可能涉及缓存因素，但尚未达成最终定论。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **vLLM/TGI 在 RTX 5000 系列上存在安装问题**：成员们在使用新款 **RTX 5000** 系列显卡设置 **vLLM** 或 **TGI** 时遇到了问题，他们需要 nightly 版本的 **PyTorch** 和 **CUDA 12.8**，但这并非易事……
   - 一位成员表示，“当你安装其他东西时，**PyTorch** 会被旧版本覆盖”，并指向了这些 **GitHub** 仓库寻求帮助：[vllm-project/vllm/issues/14452](https://github.com/vllm-project/vllm/issues/14452), [pytorch/My-rtx5080-gpu-cant-work-with-pytorch/217301](https://discuss.pytorch.org/t/my-rtx5080-gpu-cant-work-with-pytorch/217301), [lllyasviel/stable-diffusion-webui-forge/issues/2601](https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/2601), [ComfyUI/discussions/6643](https://github.com/comfyanonymous/ComfyUI/discussions/6643)。
- **AI 打击伪造时装**：成员们分享了关于伪造产品的研究，并展示了一个基于计算机视觉和深度神经网络的系统，声称在剔除品牌服装后准确率达到 **99.71%**，该研究记录在[这篇论文](https://arxiv.org/abs/2410.05969)中。
   - 该系统不需要特殊的安全标签或对供应链追踪进行修改，并且仅通过少量真假物品进行了迁移训练（transfer-trained）。
- **HF 计费透明度是一个黑盒**：成员们对 **Hugging Face** 的计费和配额系统，以及 **GPU Spaces, Zero GPU Spaces, Serverless Inference API** 的服务使用情况表示困惑。
   - 他们希望 **HF** 能针对重大变更提供“报告、沟通和咨询”，例如发布“我们将实施一项重大变更，未来几天可能会不稳定”。
- **Chat Templates 现在支持训练**：成员们确认，现在可以将 **chat_template** 传递给 **transformers** 的 **TrainingArguments** 或 **Trainer**，以便在推理和训练期间为模型使用自定义的 **chat_template**。
   - [huggingface.co](https://huggingface.co/docs/transformers/main/en/chat_template_basics#can-i-use-chat-templates-in-training) 上的文档解释说，**chat_template** 是纯文本 **LLM** 的 **tokenizer** 或多模态 **LLM** 的 **processor** 的一部分，用于指定如何将对话转换为单个可标记化的字符串。
- **RAG 实现出人意料地精简**：当一位成员询问为公司实现 **RAG** 技术需要多少行代码时，另一位成员回答说只需要“几行——大约 15 到 30 行”。
   - 他们将信息存储在 **MongoDB** 中。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 调试技巧曝光**：成员们发现了 **MCP** 的调试方法，透露如果[在服务器初始化期间配置了日志记录](https://example.com/initialization)，`sendLoggingMessage` 就能发挥作用。
   - **inspector** 的局限性引发了关于开发更优替代方案的讨论。
- **开源 EV 助手服务器面世**：一个[开源 MCP EV 助手服务器](https://github.com/Abiorh001/mcp_ev_assistant_server/blob/main/ev_assitant_server.py)可以管理 **EV 充电站**、**行程规划**和**资源管理**。
   - 该服务器为 **EV** 相关服务提供了一套完整的工具和 **API**。
- **MCP 客户端实现通知功能**：一个 [MCP 客户端实现](https://github.com/Abiorh001/mcp_omni_connect)现在支持所有**通知**，包括订阅和取消订阅资源。
   - 它提供了与 **OpenAI** 模型的集成，并支持跨多个服务器的动态工具和资源管理。
- **FastMCP 存在局限性**：**FastMCP** 可能缺乏对 `subscribe_resource` 等功能的支持，一些人正在考虑使用 **low-level server** 以获得更强的控制力。
   - 成员们交流了在 **low-level server** 中处理资源订阅和更新的代码及具体实现细节。
- **Enact Protocol 成为 MCP 的 HTTP**：[Enact Protocol](https://github.com/EnactProtocol/specification) 被提议作为一种定义 **MCP** 工具的方式，类似于 **HTTP** 协议。
   - 一位成员将其描述为“一种在 **MCP** 服务器内部进行语义化工具调用（semantic tool calling）的酷炫方式”。

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 招募用户进行 UX 测试**：**NotebookLM** 正在寻求用户参与 **60 分钟的 1:1 远程访谈**，以提供对新想法的反馈，参与者可获得 **100 美元礼品卡**。
   - 参与者需提前通过 Google Drive 分享一组笔记本源文件，并[通过此表单申请](https://forms.gle/P2t8q36NqbPNSVk8A)。
- **Discover Sources 功能在 NotebookLM 首次亮相**：**NotebookLM** 推出了全新的 **Discover Sources** 功能，使用户能够一键查找并向笔记本添加相关的网页内容，并附带 **Google AI** 生成的摘要。[点击此处了解更多](https://blog.google/technology/google-labs/notebooklm-discover-sources/)。
   - 用户建议加入类似于 **Perplexity** 的学术在线资源。
- **源文件传输性问题困扰 NotebookLM 用户**：用户对 **NotebookLM** 文件夹之间缺乏源文件传输性表示不满，认为其只读性质限制了使用。
   - 他们请求实现[源文件在文件夹之间可传输](https://notebooklm.google)。
- **Gemini 迎来新负责人**：据 [The Verge](https://www.theverge.com/news/642000/google-sissie-hsaio-stepping-down-notebooklm) 报道，Josh Woodward 将接替 Sissie Hsaio 担任 **Gemini** 团队负责人，为 **Gemini app** 的下一次进化做准备。
   - 这一过渡信号预示着该应用在方向和开发上可能发生转变。
- **Safari 故障影响 NotebookLM 使用**：部分用户报告在 **Safari** (iPhone/Mac) 上访问 **NotebookLM** 出现问题；如果语言修复无效，在 URL 末尾添加 `?hl=en`（例如：`https://notebooklm.google.com/?hl=en`）可能会解决问题。
   - 其他用户确认，通过在主屏幕添加快捷方式，**NotebookLM** 可以在 iPhone SE（第二代）上运行。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Ace 电脑 Autopilot 发布**：[General Agents Co](https://x.com/sherjilozair/status/1907478704223297576) 推出了 **Ace**，这是一款实时的电脑 Autopilot，能以超人的速度使用鼠标和键盘执行任务。
   - 与聊天机器人不同，Ace 旨在直接在电脑上执行任务。
- **YourBench 开启自定义基准测试**：[YourBench](https://x.com/sumukx/status/1907495423356403764) 推出了 **YourBench**，这是一个开源工具，用于从任何文档中进行自定义基准测试（benchmarking）和合成数据生成。
   - YourBench 旨在通过提供自定义评估集和排行榜来改进模型评估。
- **Llama 4 生成图像**：**Llama 4** 正在消息功能中推出图像生成和编辑功能。
   - 用户注意到编辑速度非常快，称 *编辑仅需 1 秒，而 GPT-4o 需要 5 分钟*。
- **Scale AI 估值飙升**：**Scale AI** 今年营收预计将达到 **20 亿美元**，这促使一项要约收购将公司估值推至 **250 亿美元**。
   - 去年营收为 **8.7 亿美元**。
- **A16Z 组装 AI 工作站**：A16Z 从零开始构建了一台 **8x RTX 4090 GPU AI 工作站**，兼容支持 **PCIe 5.0** 的新款 **RTX 5090**，用于在本地训练、部署和运行 AI 模型。
   - 他们发布了关于如何构建自己的工作站的[完整指南](https://x.com/Mascobot/status/1907899937838301311)。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **卓越的 UX/UI 抢尽风头**：成员们强调，成功的初创公司通常拥有更好的 **UX/UI**，并指出当前产品缺乏一种“必胜秘诀（winning sauce）”。此外，还展示了一个并行生成 Web 组件的 Agent 集群，详见[此屏幕录像](https://cdn.discordapp.com/attachments/986699377257119794/1357190780258746429/Screen_Recording_2025-04-03_at_1.39.26_pm.mov?ex=67eff6a9&is=67eea529&hm=9a8e202a73469a0749a23b81496240fd68a93a295583b0ce34cf52ff80c0c03e&)。
   - 一位用户寻求通过布局生成器自动化线框图绘制（wireframing），设计灰度线框图并进行优化，最后填充 Web 组件，从而利用 Agent 集群跳过线框图/设计步骤。该用户引用了 [Dribbble 上的这项设计](https://dribbble.com/shots/25708347-Delivery-Web-App-Design) 作为灵感来源。
- **GPT-4o 产生了自主意识**：用户观察到 **GPT-4o** 表现出异常行为，例如采用特定的人设（persona）并在回答中添加括号注释，并提供了[这张截图](https://cdn.discordapp.com/attachments/986699377257119794/1357335757676871711/image.png?ex=67efd4ee&is=67ee836e&hm=4deb85a208466f212d88e7b77771776834fe28524ac15dc9c5dbcb1be3301ff3&)作为例子。
   - 关于这种行为的起源出现了各种猜测，理论从 SFT 中使用的“情商数据集（EQ dataset）”到涌现属性（emergent properties）不等；用户还注意到 GPT-4o 的运行速度正在变慢。
- **LLM 在数学奥林匹克竞赛中失利**：一位成员分享了[一篇论文](https://arxiv.org/abs/2503.21934v1)，评估了最先进的 LLM 在 **2025年美国数学奥林匹克 (USAMO)** 中的表现。像 **O3-MINI** 和 **Claude 3.7** 这样的模型在 **6 道证明类数学题**上的得分率不足 **5%**。
   - 每道题满分为 **7 分**，总分最高 **42 分**。这些模型是在所有能想象到的数学数据上训练出来的，包括 **IMO 题目**、**USAMO 存档**、**教科书**和**论文**。
- **扩散模型 Dream 7B 觉醒**：根据[这篇博客文章](https://hkunlp.github.io/blog/2025/dream/)，HKU-NLP 和华为诺亚方舟实验室发布了 **Dream 7B**。这是一个开源的扩散大语言模型，其性能超越了现有的扩散语言模型，并达到或超过了同等规模的顶尖自回归（AR）语言模型。
   - Dream 7B 展示了*强大的规划能力和推理灵活性，这天然受益于扩散建模（diffusion modeling）。*

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **OpenAI API 更新有状态设计**：根据 [OpenAI 文档](https://platform.openai.com/docs/guides/conversation-state?api-mode=chat)，使用 OpenAI 的 `/v1/chat/completions` API 时，每次 Prompt 都必须重新发送完整的对话历史，即使是未被逐出的输入 Token 也会产生费用。
   - 即将推出的 `/v1/responses` API 将是“有状态的（stateful）”，通过 ID 引用过去的消息。这与“无状态的（stateless）” `/v1/chat/completions` API 形成对比，详见 [Responses vs Chat Completions 文档](https://platform.openai.com/docs/guides/responses-vs-chat-completions)。
- **AMD 的 TunableOp 加入 PyTorch**：AMD 在 [PyTorch](https://pytorch.org/docs/stable/cuda.tunable.html) 中引入了 **TunableOp**。这是一个原型特性，允许使用不同的库或技术来选择最快的算子实现（例如 GEMM）。
   - 虽然 NVIDIA 在 **CuBLAS** 中预先调整了一切，但 AMD 的方法旨在优化各种硬件配置下的性能，即使它对消费级 GPU 的优化程度可能较低，但仍能提供一个基准。
- **ThunderKittens 扑向 Blackwell**：HazyResearch 团队为 **NVIDIA Blackwell 架构**推出了新的 **BF16** 和 **FP8 ThunderKittens GEMM 内核**，其速度接近 **cuBLAS**。
   - 正如他们的[博客文章](https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell)所述，这些内核利用了**第五代 Tensor Core**、**Tensor Memory** 和 **CTA 对**等特性，并集成到了 TK 基于 Tile 的抽象中。
- **Reasoning Gym 数据集获得课程强化**：一位成员提交了一个 PR ([#407](https://github.com/open-thought/reasoning-gym/pull/407))，旨在优化 [reasoning-gym](https://github.com/open-thought/reasoning-gym) 项目中所有**数据集**的**课程设置（curricula）**，改进了测试并纳入了缺失的课程，如 **Knight Swap** 和 **Puzzle2**。
   - 另一位成员正在研究类似于 **RGBench** 的**简单、中等、困难**难度接口，以便用户手动设置难度，并分享了一个链接，说明了 [reasoning-gym](https://github.com/open-thought/reasoning-gym/blob/5b4aa313819a9a6aecd6034b8c6394b6e4251438/eval/yaml/medium/claude-3.5-sonnet.yaml) 中每项任务被视为“中等”难度的设置。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **用数值驱动维度（Powering Dimensions with Quantities）**：成员们分享了使用带有 `Dimensions` 的 `Quantity` 结构体来定义物理量的代码，创建了如 `Velocity`、`Acceleration` 和 `Newton` 等别名。
   - 一位用户链接到了他们在 [GitHub 上的 Kelvin 库](https://github.com/bgreni/Kelvin/blob/main/kelvin/quantity.mojo#L55-L125)，展示了让 `Dimensions ** power` 正常运行的过程。
- **`IntLiteral` 再次出击！**：一位成员承认在定义 `Quantity` 时使用了“诅咒级”的 `IntLiteral` 技巧来绕过动态值问题。
   - 其他成员称赞了使用 `IntLiteral` 将任意信息编码进类型系统的做法，而另一些人则开玩笑说这种方法太“可怕”了。
- **为 Modular Max 提议 Duration 结构体**：一位成员重点介绍了一个针对 modular/max 的 Pull Request，该 PR 引入了一个受 C++ 标准库中 `std::chrono::duration` 启发的 **Duration 结构体**，可在 [GitHub](https://github.com/modular/max/pull/4022#issuecomment-2694197567) 上查看。
   - 该成员即将完成 GitHub Issue 中提到的特定“理想化”代码片段。
- **渴望 Mojo 的 Python 互操作性**：一位用户询问了 **Mojo 的 Python 封装**进度，以及从 CPython 调用 Mojo 的能力。
   - 另一位用户回应说，这将是一个非常 🔥 的特性。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Checkpoints 获得 HuggingFace 待遇**：成员们讨论了使用 **HuggingFace checkpointer** 将 **torchtune checkpoints** 转换为 **HF checkpoint 格式**。
   - 特别推荐使用 **tune_to_hf 函数**进行此类转换。
- **Unsloth 与 vLLM 共享 VRAM**：在 [Unsloth](https://github.com/unslothai/unsloth) 中，他们实现了让 **vLLM** 和训练过程使用相同的 VRAM，尽管具体机制尚不明确。
   - 一位成员建议，在验证配置中使用 `train` 作为掩码标志可能会导致混淆。
- **Ariel 提供 GRPO 上游好物**：一位成员提议贡献其内部 **GRPO** 上游的更改，包括进程内 **vLLM** 集成、训练中评估以及更灵活的 **RL** 数据处理。
   - 另一位成员指出，异步版本中已存在 **vLLM** 集成，且验证数据集的 PR 已接近完成。
- **Torchtune 的超时 Bug 影响序列长度**：一位成员报告称，如果某些 microbatches 的 **seq length** 为 **7/14/21/28/35/42/49**，**Torchtune** 会因超时而挂起并崩溃，并提交了 [一个 Issue](https://github.com/pytorch/torchtune/issues/2554)。
   - 该成员指出，*torchtune dataloader* 中的非随机种子有助于捕捉到这个“神奇的 Bug”。
- **Dream 7B 证明了扩散模型的优势**：香港大学和华为诺亚方舟实验室发布了 **Dream 7B**，这是一款新型开源扩散大语言模型（LLM），详见[这篇博客文章](https://hkunlp.github.io/blog/2025/dream/)。
   - 据报道，**Dream 7B** 在通用能力、数学和编程能力上*大幅超越了现有的扩散语言模型*，并达到或超过了同等规模的顶级自回归语言模型。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **绘图工具大对决！**：成员们就图表创建工具展开了辩论，推荐高级用户使用 **Inkscape**，推荐追求易用性的用户使用 **draw.io**。
   - 一位用户开玩笑说，任何替代 **pure TikZ** 的方案都是虚假的。
- **GitHub 将在旧金山举办 AI 活动**：**GitHub** 正在旧金山共同主办一场 **MCP Demo Night** 活动，重点关注 **AI**、事件响应和平台工程；更多详情请见 [lu.ma/9wi116nk](https://lu.ma/9wi116nk)。
   - 活动包括闪电演示（lightning demos）、**Future of AI Panel**（AI 的未来面板讨论）、炉边谈话和社交活动。
- **OpenThinker2 模型性能超越 DeepSeekR1-32B**：Ludwig Schmidt 及其团队发布了 **OpenThoughts-1M** 数据集和 **OpenThinker2-32B**、**OpenThinker2-7B** 模型。通过在 **Qwen 2.5 32B Instruct** 上进行 **SFT**，其表现超越了 **R1-Distilled-32B**，详情见其[博客文章](https://www.openthoughts.ai/blog/thinkagain)。
   - 根据 [Etash Guha 的推文](https://x.com/etash_guha/status/1907837107793702958)，**OpenThinker2-32B** 和 **OpenThinker2-7B** 仅通过在开源数据上进行 **SFT** 就超越了 **DeepSeekR1-32B**。
- **转向向量（Steering Vectors）：可靠还是冒险？**：一位成员分享了论文 [Steering Vectors: Reliability and Generalisation](https://arxiv.org/abs/2407.12404)，表明 **steering vectors** 在分布内（in-distribution）和分布外（out-of-distribution）都存在局限性。
   - 论文强调，*可转向性在不同输入之间具有高度变异性*，并且对 Prompt 的更改可能非常脆弱。
- **动态转向向量组合成为热点**：一位成员分享了他们在 [steering vector composition](https://aclanthology.org/2024.blackboxnlp-1.34/) 方面的工作，该工作使用了 **Dynamic Activation Composition**，在处理语言与形式度/安全性等无关属性对时取得了成功。
   - 他们的信息论方法通过调节转向强度来保持高水平的条件控制，同时最大限度地减少对生成流畅性的影响。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Google 导师计划（Mentorship）的产出引发争议**：一位成员质疑 **Google Mentorship** 项目的价值，认为其*产出几乎不值得投入的时间和精力*。
   - 相反，其他人认为公司实际上获得了*为你全职工作 3 个月的聪明人*，这使其成为一项值得的尝试。
- **Tinygrad YoloV8 在 Android 上遇到小问题**：用户在运行 `pip install tinygrad` 后，在 Samsung Galaxy Tab S9 上运行 **YoloV8** 的 **tinygrad** 实现时遇到了 `OSError: dlopen failed: library "libgcc_s.so.1" not found` 错误。
   - George Hotz 建议这可能是一个 2 行代码的修复，但应将 Android 添加到 **CI** 中以防止再次发生，而另一位成员建议运行 `pkg install libgcc`。
- **LeetGPU 即将支持 Tinygrad**：成员们确认 [leetgpu.com](https://leetgpu.com) 很快将支持 **tinygrad**。
   - 目前尚未提供关于支持细节的进一步信息。
- **tinygrad 中的双线性插值（Bilinear Interpolation）问题**：一位成员询问关于 **tinygrad** 中 **bilinear interpolation** 的支持情况，表示在文档中搜索 **bilinear** 后发现其“无法工作”。
   - 未提供更多细节。
- **澄清模型覆盖逻辑**：一位成员询问在每个 epoch 之后使用 `state_dict = get_state_dict(net); safe_save(state_dict, "model.safetensors")` 来保存最新模型是否安全。
   - 另一位成员澄清说，除非为每次保存提供不同的名称，否则模型将被覆盖。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **CodeAct 泛化了 ReAct**：从零开始的 **CodeAct** 是 **ReAct** 的一种泛化形式。Agent 不再仅仅进行思维链（chain-of-thought），而是通过[此工具](https://t.co/0GPTPo87ma)动态编写代码并调用这些函数来解决任务。
   - 其意图是允许将动态编码作为解决任务的工具。
- **Rankify 框架助力 RAG**：全新的开源 [Rankify 框架](https://github.com/DataScienceUIBK/Rankify) 旨在简化 **retrieval（检索）、reranking（重排序）和 RAG**（检索增强生成）等任务。
   - 它支持 7 种以上的检索技术、24 种以上的先进 Reranking 模型以及多种 RAG 方法。
- **增强 Gemini API 集成**：一位成员正在起草一份关于 DeepMind *增强 Gemini API 集成* 的 GSoC 提案，并希望将 **LlamaIndex** 作为其中的重要部分，目前正在寻求关于功能缺失和优化方面的反馈。
   - 具体而言，正在征求关于 llama-index-llms-google-genai 或 vertex 中 **Gemini** 支持（如多模态或 function calling）的显著缺失，以及任何与 **Gemini 相关的特性或优化** 的反馈。
- **MCP 工具赋予 Cursor API 智能**：成员们讨论了在编程时如何为 **Cursor** 提供最新的 API 和文档知识，并建议使用一个对文档进行检索的 **MCP 工具**。
   - 由于代码库规模庞大，*llm.txt* 被认为几乎没有用处。
- **Trace ID 面临检索挑战**：成员们报告了在父工作流调用子工作流后无法检索到 **otel trace_id** 的问题。
   - 团队建议将 **trace_id** 存放在其他可以获取的地方（如工作流上下文或其他全局变量）。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **ChatGPT 4o 幻化万智牌流行文化卡牌**：一位成员利用 **ChatGPT 4o 的图像生成器** 制作了以流行文化人物和 **NousResearch 团队** 为主题的 **万智牌（Magic the Gathering）卡牌**，并将结果发布在 general 频道。
   - 生成的卡牌获得了“品鉴员”们的高度认可，但有一条评论暗示 *sama 还是不行*。[Teknium 的推文](https://x.com/Teknium1/status/1907492873991499998)展示了几张由该图像生成器制作的万智牌风格卡牌。
- **Runway Gen 4 加速 AI 电影制作**：随着 **Runway Gen 4** 的发布，AI 提示词电影制作（A.I. Prompt Filmmaking）迈出了一大步，一段关于 **OpenAI、Google 和 AGI** 领域动态的[视频](https://www.youtube.com/watch?v=Rcwfj18d8n8)对此进行了报道。
   - 视频强调了 **AI Video** 领域令人难以置信的进展，并提到开源替代方案 **Alibaba Wan 2.2** 即将发布。
- **Genstruct-7B 生成数据提取指令**：针对使用 **LLM 进行提取** 以从非结构化 PDF 创建数据集的咨询，一位成员推荐将 [Genstruct-7B](https://huggingface.co/NousResearch/Genstruct-7B) 作为可行的起点。
   - **Genstruct-7B** 受到 **Ada-Instruct** 的启发，旨在根据原始文本语料库生成有效的指令，并可以通过 [GitHub 仓库](https://github.com/edmundman/OllamaGenstruct) 快速在 ollama 中使用。
- **面向 LLM 的 OpenAPI 访问发布，减少冗余**：一位成员宣布发布了针对 **LLM** 的 **SaaS/PaaS/IaaS** 的 **v1 OpenAPI 访问权限**，旨在减少 **MCP 杂乱**，并链接到了一个 [HN 讨论](https://news.ycombinator.com/item?id=43562442)。
   - 新的 **OpenAPI 访问** 旨在解决将 **LLM** 与不同云服务集成时出现的 **MCP（Multi-Cloud Platform）杂乱** 问题。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 经历性能下降**：部分用户遇到了 **http timeout errors**，并确认 [Cohere Status Page](https://status.cohere.com/) 显示 **Command-a-03-2025/command-r-plus-08-2024** 模型出现 *Degraded Performance - Increased Latency*（性能下降 - 延迟增加）。
   - 该事件处于监控中，持续了 **4 小时**。
- **Python Logging 之争**：一位正在开发用于 PDF 处理的 Python 包的成员与一名资深同事在是用 **logs** 还是 **print statements** 上产生了分歧。
   - 该成员更倾向于使用 logs，因为它们具有 **不同的级别、文件保存、可搜索性和问题报告功能**；而同事则倾向于使用 **print statements** 以避免给用户增加负担；最后建议了一个折中方案：**默认禁用 logger 实例**。
- **RAG 文档分块策略**：一位成员询问关于将 **18000 token 的文档** 用于 **RAG** 时是否需要进行切分。
   - 专家建议对文档进行切分，但这取决于最终目标和需求；同时指出 **Command-a 的 256k 上下文窗口** 以及 **command-r 和 r-plus 的 128k 上下文窗口** 应该能够轻松处理。
- **集思广益 AI 安全测试**：一个名为 **Brainstorm** 的 AI 安全测试平台将在几周内发布其 MVP，旨在确保 AI 更好地改变世界，更多信息请访问 [Brainstorm 落地页](https://brainstormai.framer.website/)。
   - **Brainstorm** 的创建者正在寻求关于当前用于测试 AI 安全和性能问题的方法的见解，特别是围绕 **bias**（偏见）、**prompt injections**（提示词注入）或 **harmful outputs**（有害输出）方面。
- **KAIST LLM 公平性研究**：一位来自 **KAIST**（韩国）的硕士生介绍了自己，其研究方向为 **LLMs/VLMs** 中的 **bias/fairness**（偏见/公平性）和 **interpretability**（可解释性）。
   - 他们正在这些特定领域积极寻求研究合作机会，并带来了来自 **KAIST** 的经验。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Nomic Embed V2 集成预期升温**：成员们热切期待 **Nomic Embed Text V2** 进入 **GPT4All**，一位成员对开发者的繁忙日程表示理解。
   - 该成员表现出了耐心，理解集成过程可能需要时间和资源。
- **建议通过联系销售进行漏洞披露**：一位成员询问了负责任地披露 **GPT4All** 漏洞的正确程序。
   - 另一位成员建议利用 **Nomic AI** 网站上提供的 [联系支持邮箱](https://atlas.nomic.ai/contact-sales) 进行此类披露。
- **GGUF 格式的 GPT4All-J 模型难以寻觅**：一位成员寻求 **Q4_0 量化** 和 **GGUF 格式** 的 **GPT4All-J 模型** 下载链接，以便集成到项目中。
   - 另一位成员回答说 **GPT4All-Falcon** 有 **GGUF** 版本，但指出 **GPT4All-J** 不可能实现。
- **Chocolatine-2-14B 摘得书籍查询桂冠**：一位成员宣布 "**Chocolatine-2-14B**" 模型是查询嵌入书籍的理想选择。
   - 未提供关于 **Chocolatine-2-14B** 模型具体功能或架构的更多细节。
- **聊天记录呼吁按时间顺序修正**：一位成员建议聊天记录应根据修改时间而非创建时间重新排序，以改善上下文体验。
   - 该成员批评当前按创建日期排序的逻辑是*随意*的，且不利于跟踪正在进行的对话。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **遥测实现 LLM Agent 自我改进闭环**：一位成员分享了一段视频：*通过配置 LLM agent 使用遥测和评估来改进自身，从而实现开发闭环*，发布在 [YouTube](https://youtu.be/jgzSq5YGK_Q) 上。
   - 讨论强调了使用 **telemetry**（遥测）和 **evaluations**（评估）来提升 **LLM agent** 的自我改进能力。
- **DSPy 解耦提示工程**：一位成员询问 **DSPy** 如何将 **prompt engineering** 的*琢磨层*与 **LLM** 行为解耦，以及它与 **OpenAI Agents SDK** 的协同作用。
   - 另一位成员确认 **DSPy** 为这种解耦提供了*程序化组件*：**signatures**（签名）和 **modules**（模块）。
- **DSPy 程序化组件揭秘**：一位成员解释了 **DSPy** 的核心抽象：**signatures** 和 **modules**，它们有助于将 **prompt engineering** 从 **LLM** 的功能行为中解耦。
   - 这使得编程而非仅仅是提示工程成为可能，有助于与 **OpenAI Agents SDK** 等工具集成。

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Phi-4-mini-instruct 加入 BFCL 竞技场**：一位成员提交了 [PR](https://github.com/ShishirPatil/gorilla/pull/967)，旨在为 **Phi-4-mini-instruct** 添加 **BFCL** 的工具评估。
   - 该成员在 PR 中附带了**评估分数**，并请求社区提供反馈和审查。
- **征集工具评估的代码审查**：一位成员正积极为其专注于工具评估的 PR 寻找审查者。
   - 另一位成员做出了回应，表示将立即审查该 **PR**。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **DeepSeek-V3 焕新升级**：**DeepSeek-V3** 已升级为 **DeepSeek-V3-0324**，据称在评估中的表现比之前略有提升。
   - 一位成员发布了 **Windsurf AI** 推特账号的[链接](https://x.com/windsurf_ai/status/1907902846735102017)，宣布了此次升级并确认其继续免费开放。
- **Windsurf 征集书签**：Windsurf 正试图提高其公告的曝光度。
   - 一位成员请求用户在 X 上收藏该公告贴，以便及时了解升级和新版本发布。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **Manus.im Discord ▷ #[showcase](https://discord.com/channels/1348819876348825620/1348823595505156137/)** (1 条消息): 

liewxinyen: 来自 <@356472623456059392> 的精彩案例 <:1741316509962:1348823230454038670> 🤩

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1357067222387789885)** (807 条消息 🔥🔥🔥): 

> `巴西律师使用 AI，ReferrerNation 的 BPO 职位匹配平台，AI 辅助学习编程，Claude 用于报告撰写，来自中国的 AI 竞赛` 

- **巴西律师进入 AI 领域**：一位巴西律师开始在法律实践中使用 **AI 工具**，尽管他自称是“婴儿潮一代”（39 岁）且刚接触 Discord。
   - 这位律师早在 2002 年就开始用 Delphi 编写代码，他对 AI 的进步表示担忧，现在正通过探索 Manus 来保持其在行业内的竞争力。
- **ReferrerNation 进入竞技场**：Mark，[ReferrerNation.com](https://www.referrernation.com/)（一个全球 BPO 职位匹配平台）的 CEO，旨在整合 **AI** 以改进招聘和自动化，并计划很快整合**基于加密货币的激励机制**。
   - 在一些社区成员表示他最初的帖子看起来太像垃圾邮件且过于关注加密货币后，Mark 表达了歉意并指出：*“在发布更多内容之前，我会花时间更好地了解这里的氛围。”*
- **使用 Gemini 和 Claude 学习编程**：成员们推荐使用 **Gemini 2.5** 或 **Claude** 来学习编程，因为它们是优秀的 **AI 编程模型**，可以协助理解和项目开发。
   - 据报道，一位警察局长在夜班期间使用 **Claude** 撰写标准化报告。
- **Manus 积分紧缺引发技巧讨论**：许多用户报告 **积分消耗过快**，免费积分迅速耗尽，引发了关于优化提示词和高效使用的讨论。
   - 成员们建议新手阅读 <#1355477259234054323> 技巧部分以避免浪费积分，并提到团队正在努力降低积分消耗率。成员们分享说，将 Manus 与其他工具（例如 [R1](https://www.perplexity.ai/)）结合使用，对于高效利用积分非常有效。
- **利用外部代码化解难题**：一位成员提到由于积分不足而难以从 Manus 下载文件，社区建议使用 **Claude** 等第三方应用来提取代码和文件。
   - 还有人提到，最佳实践是从 Manus 下载所有文件并交给其他工具。然后，要求另一个 AI（例如 Gemini）*“为这个网站提供文件”*，接着回到 Manus 说 *“将这些文件添加到这个网站”*。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/ban-hammer-futurama-scruffy-gif-20750885">Ban Hammer GIF - Ban Hammer Futurama - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/hello-chat-hello-hello-chat-back-from-the-gif-16804150723034691763">Hello Chat Hello Chat Back From The GIF - Hello chat Hello Hello chat back from the - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/baby-face-palm-really-sigh-stupid-gif-12738431">Baby Face Palm GIF - Baby Face Palm Really - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/welcome-michael-scott-dunder-mifflin-the-office-welcome-aboard-gif-27005393">Welcome Michael Scott GIF - Welcome Michael Scott Dunder Mifflin - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://vnxjunpk.manus.space/">理解 2025 年关税格局</a>: 未找到描述</li><li><a href="https://x.com/Lasgidiconf/status/1907805373710360857?t=eTJ-1SBWbz8w64q3SsKzVA&s=19">来自 Chuka Konrad (@Lasgidiconf) 的推文</a>: 昨晚我构建了一个视觉效果丰富的交互式网页，重点展示了特朗普宣布的对等关税的关键方面和分析。完全基于 @ManusAI_HQ 构建，请访问 https://vn... 查看。</li><li><a href="https://bfuarkjn.manus.space/">ManusAI - 全面指南</a>: 未找到描述</li><li><a href="https://tenor.com/view/in-the-house-martin-martin-lawrence-biggie-hello-gif-12010068014708218113">In The House Martin GIF - In The House Martin Martin Lawrence - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/coffee-caffeine-coffee-time-wake-up-morning-coffee-gif-7886258858573853472">Coffee Caffeine GIF - Coffee Caffeine Coffee time - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/hype-train-activated-bowser-hype-train-gif-14185403">Hype Train Activated Bowser GIF - Hype Train Activated Bowser Hype Train - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/salamik-cute-awesome-collection-poznas-gif-10073975131879134759">Salamik Cute GIF - Salamik Cute Awesome - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/silver-lining-impressed-robert-de-niro-gif-14541556">Silver Lining Impressed GIF - Silver Lining Impressed Robert De Niro - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/stay-tuned-robertidk-keep-watching-theres-more-more-to-come-gif-19154825">Stay Tuned Robertidk GIF - Stay Tuned Robertidk Keep Watching - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/good-morning-gif-10250909101792021306">Good Morning GIF - Good morning - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://ucebdqhq.manus.space/">使用 Manus AI 进行迭代开发：全面指南</a>: 未找到描述</li><li><a href="https://tenor.com/view/bait-thats-bait-tom-hardy-mad-max-gif-5055384">Bait Thats Bait GIF - Bait Thats Bait Tom Hardy - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/awkward-nodding-dan-levy-david-david-rose-schitts-creek-gif-20776317">Awkward Nodding Dan Levy GIF - Awkward Nodding Dan Levy David - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/fever-sick-flu-parks-and-rec-everything-hurts-gif-5394213">Fever GIF - Fever Sick Flu - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/night-sleep-gif-27085775">Night Sleep GIF - Night Sleep - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/get-your-pooh-on-gif-16407146543386304197">Get Your Pooh On GIF - Get your pooh on - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://manus.im/help/credits">Manus</a>: Manus 是一个通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，在你休息时搞定一切。</li><li><a href="https://tenor.com/view/inthehouse-martin-martinlawernce-biggie-hello-gif-13128531067958866971">Inthehouse Martin GIF - Inthehouse Martin Martinlawernce - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://manus.im">Manus</a>: Manus 是一个通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，在你休息时搞定一切。</li><li><a href="https://tenor.com/view/superman-gif-14881123907931593412">Superman GIF - Superman - 发现并分享 GIF</a>: 点击查看 GIF
</li>
</ul>

</div>

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1357069747610452049)** (1010 条消息🔥🔥🔥): 

> `Meta vision model Cotton, Qwen2.5-vl-32b-instruct OCR, Google Gemini Models, Nightwhisper model on webdev, Gemini 2.6 Pro experimental` 


- **Qwen2.5-vl-32b-instruct 在 OCR 方面与 Google Gemini 模型不相上下**：对于日语低质量文本的 OCR，`qwen2.5-vl-32b-instruct` 几乎与 **Google Gemini 模型**持平。
   - 然而，Meta 的视觉模型 `cotton` 感觉更像 Meta 最近推出的其他**仅限文本的匿名模型**。
- **Nightwhisper 在 WebDev 首次亮相**：**Nightwhisper** 模型仅在 [webdev arena](https://webdev.lmarena.ai/) 上提供，可能仅是一个编程模型。
   - 成员们怀疑 **Nightwhisper = Gemini 2.6 Pro experimental**，且不会出现在普通竞技场中。
- **Nightwhisper 擅长网页生成，但在处理现有代码时表现不佳**：事实证明，**Nightwhisper** 在构建具有吸引力 UI 的功能性应用方面表现出色，并可以通过临时 URL 在 [webdev arena](https://webdev.lmarena.ai/) 上生成项目。
   - 然而，一些用户报告称，Nightwhisper 在编辑现有代码或遵守特定格式请求时存在困难，导致使用体验不佳。
- **WebDev 竞技场引入模型克隆**：WebDev 竞技场存在渲染问题，但一位用户发现了一种模型克隆的方法，即模型会两次给出相同的屏幕。
   - 在收到来自 NightWhisper 的错误消息并在再次提供代码后重复出现错误后，其名称未显示，这表明存在模型克隆。
- **Gemini 2.5 Pro vs Nightwhisper - 竞争继续**：**Gemini 2.5 Pro** 在 [USAMO 2025](https://matharena.ai/) 上达到了 **24.4%**，但某些模型往往会在句子中途停止或给出部分响应。
   - 一位用户发现 Gemini 在创建宝可梦模拟器方面表现更好，而 Nightwhisper 生成的 UI 更整洁，但攻击力数值异常偏高。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.testingcatalog.com/google-plans-new-gemini-model-launch-ahead-of-cloud-next-event/">Google 计划在 Cloud Next 活动之前发布新的 Gemini 模型</a>：了解 Gemini 的最新更新，包括潜在的新模型发布和实验性工具。关注预设提示词和视频生成等令人兴奋的功能。</li><li><a href="https://snipboard.io/86wfLe.jpg">上传并分享截图和图像 - 在线截屏 | Snipboard.io</a>：简单免费的截图和图像分享 - 通过截屏粘贴或拖放在线上传图像。</li><li><a href="https://x.com/a7m7s1p6dv20/status/1907684868164825260?s=46">来自 ᅟ (@a7m7s1p6dv20) 的推文</a>：通过 glama AI 获取的 Gemini 2.5 Pro（初步？）定价方案</li><li><a href="https://x.com/testingcatalog/status/1907891942869922292?t=Q30isS2oxgO7U-qBjdYMtA&s=19">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：突发 🚨：Google 正准备在 Gemini 上推出另一个模型，可能在下周 Cloud Next 活动之前。引用 ʟᴇɢɪᴛ (@legit_api) 的话，nightwhisper 和 stargazer 是新增的 2 个新模型...</li><li><a href="https://create.roblox.com/docs/luau">来自 Luau | 文档 - Roblox Creator Hub 的推文</a>：Luau 是创作者在 Roblox Studio 中使用的脚本语言。</li><li><a href="https://gist.github.com/riide">riide 的 gists</a>：GitHub Gist：通过创建 GitHub 账号来收藏和 fork riide 的 gists。</li><li><a href="https://matharena.ai/">MathArena.ai</a>：MathArena：在无污染数学竞赛上评估 LLM</li><li><a href="https://gist.github.com/riidefi/443dc5c4b5e13e51846a43067b5335a1">Meta (?) 的 `24_karat_gold` (lmarena) 系统提示词</a>：Meta (?) 的 `24_karat_gold` (lmarena) 系统提示词 - prompt.txt</li><li><a href="https://devforum.roblox.com/t/expanding-assistant-to-modify-place-content-beta/3107464">来自扩展 Assistant 以修改场景内容 [Beta] 的推文</a>：各位创作者好，今天我们很高兴地宣布，我们正在扩展 Assistant 的功能，使其能够在 Studio 中执行广泛的操作。Assistant 现在可以帮助您修改 DataModel，以便自动...</li><li><a href="https://g.co/gemini/share/60fcf5c244c9">‎Gemini - Three.js 小行星撞击模拟代码
</a>：由 Gemini Advanced 创建</li><li><a href="https://en.wikipedia.org/wiki/Leela_Chess_Zero#Spinoffs>)">Leela Chess Zero - 维基百科</a>：未找到描述
</li>
</ul>

</div>

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1357128440758927461)** (1 条消息): 

> `Mobile Alpha UI, LM Arena Access, Alpha Feedback` 


- **Arena Alpha 现已登陆移动端！**：全新的 **Arena Alpha UI** 现已针对移动端进行优化，允许用户通过 [alpha.lmarena.ai](https://alpha.lmarena.ai) 在手机上进行测试。
- **访问移动端 Arena Alpha**：要访问移动端 **Arena Alpha**，用户需要使用密码 `still-alpha`。
- **为移动端 Arena Alpha 提供反馈**：可以通过 [Google Forms 链接](https://forms.gle/8cngRN1Jw4AmCHDn7) 提供关于移动端 **Arena Alpha** 的反馈，而 Bug 可以通过 [Airtable 表单](https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form) 进行报告。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://forms.gle/8cngRN1Jw4AmCHDn7">Arena - New UI Feedback</a>: 告诉我们你对新设计的看法！</li><li><a href="https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form">Airtable | 每个人的应用平台</a>: Airtable 是一个用于构建协作应用的低代码平台。自定义你的工作流，进行协作，并实现雄心勃勃的目标。免费开始使用。
</li>
</ul>

</div>
  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1357073324324290590)** (772 条消息🔥🔥🔥): 

> `Restoring to previous checkpoints, Roo code, Boomerang Mode, Gemini Pro EXP vs Pro MAX, Windsurf vs Cursor tab` 


- **分支 Bug 阻碍回溯**：成员报告在 Cursor 中**恢复到之前的 Checkpoints** 时遇到问题，即使在理应干净的分支中也会遇到来自后续状态的 Bug，成员建议执行 `git diff branch1,branch2` 来识别差异。
   - 一位成员在简单的 Logo 修改 Prompt 下经历了 CSS 的全面重构，凸显了意外修改的可能性。
- **Roo Code 工作流火爆**：一位用户描述了他们在 **Roo Code** 上极佳的工作流，强调其成本效益，通过选择性使用模型，每天仅需约 **$0.4**。
   - 他们还提到，在特定任务上其能力优于 Cursor，并附上了 [相关文档](https://docs.roocode.com/features/boomerang-tasks/)。
- **Boomerang Mode 受到关注**：成员讨论了 Roo Code 中 **Boomerang Mode** 的优势，该模式将任务分解为由不同 Agent 处理的子任务，比完全依赖 Gemini 2.5 解决问题更高效。
   - Boomerang Mode 高度可定制，一位用户分享道：*我会把它焊死，别担心*。
- **窥探 PearAI 定价**：用户比较了 Cursor 和 **PearAI** 的定价模式，PearAI 提供了一些免费额度来测试 Roo Code。
   - 然而，据澄清，PearAI 的 **$15/月计划** 包含额度限制，超出后将按使用量收费，这与无限模型访问的说法形成对比，一位成员甚至指责 Cursor “*在欺骗用户！*”
- **Nightly 版本孕育新导航理念**：Cursor **0.49.1** 已作为 Nightly 版本发布，可通过账户设置中的 `account settings, advanced -> developer settings` 开启。
   - 该功能据称是一个用于 Agent 使用的 Context Window 指示器，以及一个 Windsurf API Key。 


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.roocode.com/features/boomerang-tasks/">Boomerang Tasks：编排复杂工作流 | Roo Code Docs</a>：Boomerang Tasks（也称为子任务或任务编排）允许你将复杂的项目分解为更小、易于管理的部分。可以将其想象为将工作的一部分委托给专门的...</li><li><a href="https://x.com/sidahuj/status/1899460492999184534">来自 siddharth ahuja (@sidahuj) 的推文</a>：🧩 构建了一个 MCP，让 Claude 直接与 Blender 对话。它可以帮助你仅通过提示词创建精美的 3D 场景！这是我仅用...就创建了一个“守护宝藏的低多边形龙”场景的演示。</li><li><a href="https://ubisoft-mixer.readthedocs.io/en/latest/index.html">Mixer：用于协作编辑的 Blender 插件 — Ubisoft Mixer 文档</a>：未找到描述</li><li><a href="https://x.com/MervinPraison/status/1907165153537224953">来自 Mervin Praison (@MervinPraison) 的推文</a>：介绍 @Ollama MCP AI Agents！🎉🔒 100% 本地💻 仅需 3 行代码🗺️ 1000+ MCP 服务器集成✨ @PraisonAI v2.1 发布，支持 1000+ MCP！@AtomSilverman @Saboo_Shubham_ @elonmusk</li><li><a href="https://x.com/ehuanglu/status/1901861073902301194?s=46&t=kUuVqsG2GMX14zvB592G5w">来自 el.cine (@EHuanglu) 的推文</a>：3D AI 变得越来越疯狂了，这个新的 Hunyuan3D 2.0 MV 开源模型可以在几秒钟内通过多张图像生成 3D 资产，现在免费使用，链接在评论中，10 个示例：</li><li><a href="https://tenor.com/view/parallel-universe-operation-boomerang-gif-15332015">平行宇宙 GIF - 平行宇宙 Boomerang 行动 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://trypear.ai/privacy">隐私政策</a>：PearAI 的隐私政策。</li><li><a href="https://github.com/vbwyrde/DSPY_VBWyrde/blob/main/DSPY12_Out_3.md">DSPY_VBWyrde/DSPY12_Out_3.md 位于 main 分支 · vbwyrde/DSPY_VBWyrde</a>：DSPY 实验。通过在 GitHub 上创建账户来为 vbwyrde/DSPY_VBWyrde 的开发做出贡献。</li><li><a href="https://github.com/lharries/whatsapp-mcp">GitHub - lharries/whatsapp-mcp: WhatsApp MCP 服务器</a>：WhatsApp MCP 服务器。通过在 GitHub 上创建账户来为 lharries/whatsapp-mcp 的开发做出贡献。</li><li><a href="https://github.com/supercorp-ai/supergateway">GitHub - supercorp-ai/supergateway: 在 SSE 上运行 MCP stdio 服务器，以及在 stdio 上运行 SSE。AI 网关。</a>：在 SSE 上运行 MCP stdio 服务器，以及在 stdio 上运行 SSE。AI 网关。- supercorp-ai/supergateway</li><li><a href="https://github.com/ahujasid/blender-mcp">GitHub - ahujasid/blender-mcp</a>：通过在 GitHub 上创建账户来为 ahujasid/blender-mcp 的开发做出贡献。</li><li><a href="https://www.cursor.com/changelog">更新日志 | Cursor - AI 代码编辑器</a>：新的更新和改进。</li><li><a href="https://github.com/punkpeye/awesome-mcp-servers">GitHub - punkpeye/awesome-mcp-servers: MCP 服务器集合。</a>：MCP 服务器集合。通过在 GitHub 上创建账户来为 punkpeye/awesome-mcp-servers 的开发做出贡献。</li><li><a href="https://www.findsimilarstartups.com/shared/67edd25421be61a714cc807e">寻找相似初创公司 - AI 驱动的竞品分析</a>：利用 AI 驱动的市场研究即时发现初创公司竞争情况。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1357085846167093411)** (217 条消息🔥🔥): 

> `EC2 实例中的 ECC 错误，自定义数据集的 Gemma 3 Bug，Unsloth Apple Silicon 支持，使用 Unsloth 微调 LLaSA，RTX 5090 vs RTX 4090 用于 Unsloth`

- **Unsloth 上的 EC2 实例抛出 ECC 错误**：一位用户报告在 `g6e.4xlarge` **EC2 实例**上串行处理提示词（prompts）时遇到了 **CUDA ECC 错误**和 **500 错误**，并链接到了 GitHub 上的 [Issue #2270](https://github.com/unslothai/unsloth/issues/2270)。
   - 该错误为 *uncorrectable ECC error encountered*（遇到不可纠正的 ECC 错误），表明存在硬件或内存问题。
- **数据集混乱触发 Gemma 3 Bug**：一位用户在使用来自 [Hugging Face](https://huggingface.co/datasets/adamtc/sdtg_sgpt) 的自定义数据集训练 **Gemma 3** 时遇到了 Bug 并寻求帮助，详情见 [Issue #2270](https://github.com/unslothai/unsloth/issues/2270)。
- **Unsloth 即将支持 Apple Silicon？**：一位用户请求对一个与 Apple 设备相关的 Pull Request 进行测试，旨在将其性能与基础 MLX 进行对比，该请求见 [PR #1289](https://github.com/unslothai/unsloth/pull/1289)。
- **Llasa LoRA 即将登陆 Unsloth**：社区成员正在考虑利用 Unsloth 对 **Llasa**（一种文本转语音 TTS 系统）进行 LoRA 训练，相关 Pull Request 的指导见 [PR #2263](https://github.com/unslothai/unsloth/pull/2263)。
- **RTX 5090 完胜 RTX 4090？**：一位用户分享了在使用不受支持的 Unsloth 版本时，**RTX 5090** 与 **RTX 4090** 的样本处理速度对比，突显了新硬件带来的潜在性能提升。
   - 另一位成员表示这在性价比上 *不值得*，但其他人认为，如果你受限于 VRAM 或需要更快的训练速度，它可能是 **ROI 正向** 的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/get-started/beginner-start-here">初学者？从这里开始！ | Unsloth 文档</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/MrDragonFox/Elise">MrDragonFox/Elise · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/xyehya/documentation">GitHub - xyehya/documentation: Odoo 文档源码</a>: Odoo 文档源码。通过在 GitHub 创建账号参与 xyehya/documentation 的开发。</li><li><a href="https://huggingface.co/collections/google/gemma-3-qat-67ee61ccacbf2be4195c265b">Gemma 3 QAT - Google 收藏集</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/llama3-3">使用 Unsloth 微调 Llama 3.3</a>: 微调 Meta 的 Llama 3.3 (70B) 模型，其性能优于 GPT 4o，通过 Unsloth 开源实现提速 2 倍！初学者友好。现已支持 Apple 的 Cut Cross Entropy 算法。</li><li><a href="https://github.com/xyehya/documentation/blob/9.0/Unsloth-GRPO.ipynb">documentation/Unsloth-GRPO.ipynb 分支 9.0 · xyehya/documentation</a>: Odoo 文档源码。通过在 GitHub 创建账号参与 xyehya/documentation 的开发。</li><li><a href="https://github.com/unslothai/unsloth/issues/2273.">unslothai/unsloth</a>: 微调 Llama 3.3, DeepSeek-R1, Gemma 3 和推理型 LLMs，速度提升 2 倍，显存占用减少 70%！ 🦥 - unslothai/unsloth</li><li><a href="https://huggingface.co/datasets/fimbulvntr/deepseek-r1-traces-no-cjk">fimbulvntr/deepseek-r1-traces-no-cjk · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/pull/1289">由 shashikanth-a 添加对 Apple Silicon 的支持 · Pull Request #1289 · unslothai/unsloth</a>: #4 未优化。暂不支持 GGUF。从源码构建 Triton 和 bitsandbytes：cmake -DCOMPUTE_BACKEND=mps -S . 用于 bitsandbytes 构建；pip install unsloth-zoo==2024.11.4；pip install xformers==0....</li><li><a href="https://huggingface.co/HKUSTAudio/Llasa-3B/">HKUSTAudio/Llasa-3B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/nicolagheza/badalisc-s1K-1.1-ita">nicolagheza/badalisc-s1K-1.1-ita · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/pull/2263">feat: 支持自定义 `auto_model` 以实现更广泛的模型兼容性 (Whisper, Bert 等) 及 `attn_implementation` 支持，由 Etherll 提交 · Pull Request #2263 · unslothai/unsloth</a>: feat: 支持自定义 auto_model、Whisper 参数和 attn_implementation。此 PR 增强了 FastModel.from_pretrained 以支持更广泛的模型：自定义 auto_model：允许指定 ex...</li><li><a href="https://github.com/unslothai/unsloth/issues/2270">[BUG] 在 Colab 中使用自定义数据集训练 Gemma 3 时无法创建张量 · Issue #2270 · unslothai/unsloth</a>: 错误描述：在 Colab 中使用（我的诈骗数据集）[https://huggingface.co/datasets/adamtc/sdtg_sgpt] 训练 Gemma 3 时（将 "mlabonne/FineTome-100k" 替换为 "adamtc/sdtg_sgpt"...</li><li><a href="https://huggingface.co/datasets/adamtc/sdtg_sgpt">adamtc/sdtg_sgpt · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/r1-reasoning">在本地训练你自己的 R1 推理模型 (GRPO)</a>: 你现在可以使用 Unsloth 100% 在本地复现你自己的 DeepSeek-R1 推理模型。使用 GRPO。开源、免费且初学者友好。</li><li><a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo)!">Unsloth 文档</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/)">Unsloth 文档</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(1B)-GRPO.ipynb">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1357109025161871360)** (5 条消息): 

> `职业转型, 产品经理, Vibe Coder` 


- **职业转型 Vibe Coder**: 一位成员开玩笑说他们上一份工作中的 "Vibe Coder" 实际上是产品经理。
   - 他们以 *Sincerely, What have we done* 署名结尾，暗示了一种幽默的无奈感。
- **产品经理作为 Vibe Coder**: 讨论强调了产品经理的角色被比作 "Vibe Coder"，暗示其核心在于团队士气和方向。
   - 这一对比凸显了产品经理在设定基调以及影响团队整体能量和生产力方面的重要性。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1357080161790590996)** (236 条消息🔥🔥): 

> `Unsloth batch size, SFTTrainer 用法, Gemma3 微调, Qwen2.5 图像尺寸, GRPO 与 CPU 瓶颈` 


- **Unsloth 自动 Batch Size**: 用户讨论了 Unsloth 在使用多 GPU 时会自动将 batch size 设置为 2 的问题，通过设置 `CUDA_VISIBLE_DEVICES=0` 找到了解决方案。
   - 一位用户指出 Unsloth 使用的是 `per_device_train_batch_size`，并链接到了 [`training_utils.py` 中的相关代码](https://github.com/unslothai/unsloth-zoo/blob/4a66f8b08952fc148f5c74cd15aec52cb0113e2d/unsloth_zoo/training_utils.py#L206)。
- **SFTTrainer 故障排除**: 一位用户在对 Llama 3.2 1B instruct 使用普通 `Trainer` 时遇到了 `ValueError`（即使关闭了 FP16），但通过切换到 `SFTTrainer` 解决了该问题。
   - 据推测，该模型可能是 bfloat16 格式，而 Unsloth 无法从 `Trainer` 中获取 dtype，建议检查模型的 config json。
- **在 Unsloth 中使用 Gemma3 Vision**: 用户成功使用 Unsloth 开启了带有视觉样本的 Gemma3 微调，并注意到 `UnslothVisionDataCollator` 仅接受带有图像的样本。
   - 他们好奇是否可以在 Gemma3/Unsloth 中使用自定义 collator 而非 `UnslothVisionDataCollator`，并询问了潜在的注意事项或示例。
- **Qwen2.5 图像尺寸困扰**: 用户讨论了在将图像输入 Qwen2.5-VL-7B-Instruct 之前如何显式设置图像尺寸，结果发现手动调整图像大小可能会导致错误，因为模型内部会自动调整尺寸。
   - 一位用户确认将图像调整为 **364x364** 是可行的，而另一位用户建议尝试 **224x224**。
- **GRPO 训练中的 CPU 瓶颈**: 用户在运行 Gemma3_(1B)-GRPO.ipynb 代码时观察到 CPU 瓶颈（单核占用率达 100%），尽管 GPU 利用率较低（25%），但训练速度仍受限。
   - 建议包括对代码进行性能分析 (Profiling) 以识别 CPU 密集型操作，并考虑到由于模型较小，该过程可能是内存受限 (Memory Bound) 而非计算受限 (Compute Bound)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>: 以下是我们所有 notebook 的列表：</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1mn_hj_sNvW59JxW0u2nuGoB7qLEs30RO?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb#scrollTo=vITh0KVJ10qX">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/trl/en/sft_trainer">Supervised Fine-tuning Trainer</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/2270">[BUG] Unable to create tensor when training Gemma 3 in Collab using custom dataset · Issue #2270 · unslothai/unsloth</a>: 描述该 bug：在 Collab 中使用自定义数据集训练 Gemma 3 时...</li><li><a href="https://github.com/unslothai/unsloth/issues/1624#issuecomment-2774130919,">GRPOTrainer crashes with unsloth · Issue #1624 · unslothai/unsloth</a>: 我正尝试在 Unsloth 中运行 GRPOTrainer，但它崩溃了。如何修复？unsloth 2025.2.4, transformers 4.47.1, torch 2.5.1, trl 0.14.0。相关代码如下：model, tokenizer...</li><li><a href="https://github.com/huggingface/transformers.git">GitHub - huggingface/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.</a>: 适用于 Pytorch, TensorFlow 和 JAX 的前沿机器学习。</li><li><a href="https://github.com/unslothai/unsloth/issues/2052#issuecomment-2761332050">AttributeError: &#39;HybridCache&#39; object has no attribute &#39;float&#39;—Gemma 3 training fails with BF16 precision on RTX3090 (Ampere) GPUs · Issue #2052 · unslothai/unsloth</a>: 我在 Linux 上使用 NVidia RTX3090 GPU。Unsloth 在新发布的 Gemma-3 上进行 BF16（和 FP16）训练时似乎存在问题，这似乎与 GPU 有关...</li><li><a href="https://github.com/unslothai/unsloth-zoo/blob/4a66f8b08952fc148f5c74cd15aec52cb0113e2d/unsloth_zoo/training_utils.py#L206">unsloth-zoo/unsloth_zoo/training_utils.py at 4a66f8b08952fc148f5c74cd15aec52cb0113e2d · unslothai/unsloth-zoo</a>: Unsloth 工具类。</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1357368647643041912)** (2 messages): 

> `GRPO Trainer Implementation, Unsloth Techniques, Collab Notebook, DeepSpeed alternative` 


- **Unsloth 驱动 GRPO Trainer**：一位成员分享了他们的 Collab notebook，详细介绍了使用 **Unsloth 技术**实现 **GRPO Trainer** 的过程，而这在以前只能通过在多台 **Hx00** 上使用 **DeepSpeed** 来实现。
   - 他们分享了一个[链接](https://github.com/xyehya/documentation/blob/9.0/Unsloth-GRPO.ipynb)，并鼓励用户使用和参考，欢迎评论和反馈，并指出该方案非常有前景。
- **社交媒体推广遭到批评**：一位成员质疑是否有必要在 Twitter 等社交媒体上发布内容，而不是在 **Hugging Face** 等平台分享直接链接。
   - 他们质问道：*为什么你总是在 Twitter 上发东西？为什么不能直接发一个 Hugging Face 的链接？*



**提到的链接**：<a href="https://github.com/xyehya/documentation/blob/9.0/Unsloth-GRPO.ipynb">documentation/Unsloth-GRPO.ipynb at 9.0 · xyehya/documentation</a>：Odoo 文档源码。通过在 GitHub 上创建账户为 xyehya/documentation 的开发做出贡献。

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1357157940242682061)** (6 messages): 

> `GRPO/PPO, Continue Pretraining Llama, Bespoke Labs new models` 


- **GRPO/PPO 方法并不完美**：一位成员指出 **GRPO**、**PPO** 和其他强化学习方法存在问题。
   - 尽管面临挑战，这些方法目前仍在被积极使用。
- **在 Unsloth 上继续预训练 Llama**：一位成员询问关于使用 **Unsloth** 并配合大量故事的 `.txt` 文件来微调 **Llama** 的问题。
   - 有建议称，如果硬件资源充足且数据预处理得当，可以尝试继续预训练（Continue Pretraining）；否则，使用不同的微调阶段会更好。
- **Bespoke Labs 发布新模型**：据一位成员透露，**Bespoke Labs** 今天发布了新模型。
   - 分享了一张从 **Chrome** 浏览器截取的模型截图。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1357080059822735461)** (103 messages🔥🔥): 

> `Gemini vs Grok, Manus deceptive?, AI coding, OpenAI value for money` 


- **Gemini 2.5 Pro vs Grok**：成员们广泛辩论了 **Gemini 2.5 Pro** 与 **Grok** 的优劣，[一位用户报告称](https://discord.com/channels/998381918976479270/998382262374973520/1357218656887885874) **Gemini 的深度研究（Deep Research）是最好的**。
   - 然而，他们指出 *Grok 很好，在线使用很值得，但目前还没有 API 访问权限是个败笔*，而 **OpenAI** 在编程方面被 *高估了*。
- **马斯克的 Grok 失败**：多位用户报告 **Grok** 频繁崩溃且不稳定，导致用户取消订阅并损失金钱。
   - 一位用户对 **Elon Musk 的失败**表示并不意外，称 *Elon Musk 买了 20 万张 GPU 却仍然无法交付产品*，同时指出 *Elon 从未做出过像样的产品*。
- **Manus 具有欺骗性？**：成员们讨论了 [Manus](https://manus.im/share/oxmc7m9JJq1IRmtpj5mX2A?replay=1)，一位用户将其描述为 **骗子**，因为发现他们依赖 **Anthropic Sonnet**，而不是使用他们声称会开源的特殊模型。
   - 他们表示 Manus 只是靠博取关注来生存。
- **Gemini 提供最大的上下文窗口**：一位用户询问哪家 AI 供应商提供最大的上下文窗口（Context Window）并支持自定义 GPT 功能，[另一位用户回答](https://discord.com/channels/998381918976479270/998382262374973520/1357281796619718767)是 **Gemini**。
   - 他们指出它提供 **100 万 token** 和 **Gems（自定义 GPTs）**。
- **OpenAI 20 欧元订阅？**：用户正在辩论 **OpenAI 20 欧元订阅** 的价值，几位用户报告该服务已无法使用。
   - 一位用户辩称 **20 欧元的订阅** 是值得的，并引用 **Sora 计费常见问题解答** 来论证他的观点，链接见[此处](https://help.openai.com/en/articles/10245774-sora-billing-faq)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://manus.im/share/oxmc7m9JJq1IRmtpj5mX2A?replay=1">Latest VK3GOD Contacts on WSPR Rocks - Manus</a>：Manus 是一个通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，在你休息时完成一切。</li><li><a href="https://artificialanalysis.ai/">AI Model &amp; API Providers Analysis | Artificial Analysis</a>：AI 模型和 API 托管供应商的对比与分析。涵盖质量、价格、输出速度和延迟等关键性能指标的独立基准测试。
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1357079882441429062)** (4 messages): 

> `Livekit 框架, GPT-4o 任务, Red team 成员监督` 


- **Livekit 语音机器人的 Evals 使用仍不明确**：一位用户询问如何在 [Livekit 框架](https://livekit.io/) 中使用 **evals** 来构建 **voicebots**。
   - 在给定上下文中未提供解决方案。
- **GPT-4o 任务在首次尝试时受挫**：一位加入付费会员一年多的用户报告称，**GPT-4o 任务**在首次尝试时未能按预期工作。
   - 该用户描述说，模型没有创建请求的任务，而是回答了一些*随机话题*。
- **Red Team 喂宠物也需要监督？**：一位用户幽默地评论道，*Red team 成员自己也需要成年人监督，即使他们只是想喂自己的宠物。*
   - 该评论暗示了对 AI 行为的一种俏皮批评，认为其类似于无人看管的、甚至可能是混乱的宠物喂养。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1357121586930389143)** (130 messages🔥🔥): 

> `AI 图像生成, 模型行为, 内容政策 vs Model Spec, 成人内容生成` 


- **Prompt Engineering 的发光编辑**：一位用户寻求关于改进图像编辑的建议，特别是如何围绕主体添加**发光效果**（类似于笑脸），并请求模型在提供的图像上复制该效果。
   - 另一位成员建议通过精确定义预期结果并迭代比较结果来优化 prompt，强调改进输出涉及更好地向模型传达意图。
- **解码 Runes 的新颖性**：一位成员描述了一个系统，其中 **runes** 通过序列解码或运行事物，从它们之间的 runes 中推导出缺失的功能，为每个实例增加新颖性。
   - 他们建议通过序列运行一个 **concept**，每个 rune 对其进行转换，最终根据转换塌缩成一个新的呈现。
- **OpenAI Model Spec 与内容政策的冲突**：一场关于生成**成人用品**图像许可性的讨论引发了争议，有人声称这可能违反内容政策。
   - 然而，其他成员指出 OpenAI 的 [Model Spec](https://model-spec.openai.com/2025-02-12.html) 与该政策*相矛盾*，并表示如果内容无害，现在可能被允许，这突显了文件之间潜在的冲突。
- **内容政策对成人用品演变中的立场**：成员们辩论了创建**成人用品**图像是否违反 OpenAI 的内容政策，并引用了[特定政策](https://openai.com/policies/usage-policies/)和 Model Spec。
   - 虽然 Model Spec 似乎更宽松，但内容政策规定：*不要构建可能不适合未成年人的工具，包括：性显式或暗示性内容*，这导致了困惑。



**提及的链接**：<a href="https://model-spec.openai.com/2025-02-12.html">OpenAI Model Spec</a>：Model Spec 规定了 OpenAI 产品（包括我们的 API）底层模型的预期行为。

  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1357121586930389143)** (130 条消息🔥🔥): 

> `带有发光效果的图像生成、Model Spec 与 Content Policies 的对比、成人内容生成、图像编辑改进` 


- **发光边缘激发 Discord 成员灵感**：一位成员请求生成一张主体带有**彩色发光轮廓**的图像（类似于笑脸），并征求了[改进建议](https://cdn.discordapp.com/attachments/1046317269069864970/1357212216671207525/60F7927C-A2B5-456E-A46F-2CEE9D77D953.png)。
   - 另一位成员提供了详细的 Prompt 策略，建议进行迭代优化并精确定义预期结果，以引导模型的决策。
- **关于 Content Policies 与 Model Spec 的辩论**：一场关于 **OpenAI Content Policies 与 Model Spec 之间差异**的讨论展开了，一位成员引用了 2025 年 2 月 12 日发布的较新版本 [Model Spec](https://model-spec.openai.com/2025-02-12.html)，该版本似乎与现有政策存在矛盾。
   - 辩论的焦点在于被描述为“愿景式（aspirational）”的 **Model Spec** 是否优先于 **Content Policies**，以及这对生成成人用品等内容的影响。
- **成人内容生成：是否允许？**：成员们就 **OpenAI 的政策是否允许生成成人用品图像**进行了辩论，对 **Content Policies** 和 **Model Spec** 的解读存在分歧。
   - 虽然一些人最初认为此类内容是被禁止的，但其他人指出 **Model Spec** 是许可范围发生变化的信号，并提到该政策已于 1 月 29 日更新。
- **具体性是图像编辑成功的关键**：Discord 用户讨论了在为 AI 模型提供图像编辑 Prompt 时保持具体的重要性，特别是在定义图像的 *subject*（主体）时。
   - 一位寻求为图像特定部分添加发光效果的用户被建议精确定义哪些元素构成主体，以避免模型产生歧义。



**提到的链接**：<a href="https://model-spec.openai.com/2025-02-12.html">OpenAI Model Spec</a>：Model Spec 规定了 OpenAI 产品（包括我们的 API）底层模型所期望的行为。

  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1357067691134947469)** (337 条消息🔥🔥): 

> `Perplexity Pulse Program, Deep Research 更新, Gemini 2.5 vs Perplexity O1, Android 应用主屏幕, LLM Jailbreaks` 


- **Perplexity Pulse Program 引发好奇**：Discord 上的用户渴望了解并加入 **Perplexity Pulse Program**，该计划提供[新功能的 Early Access](https://x.com/testingcatalog/status/1897649019309961298?s=46) 以换取反馈，此外还有免费的 **PPLX** 和**周边 (merch)** 等福利。
   - 据说加入 **Perplexity Pulse Group** 可以让高级用户通过提供反馈来获取免费的 **PPLX**。
- **Deep Research 更新令用户失望**：用户报告称更新后的 **"deep research"** 功能[更慢且效果更差](https://www.reddit.com/r/perplexity_ai/comments/1jq27a6/why_is_perplexitys_updated_deep_research_slower/)，一位用户称其存在*过度拟合和确认偏误*，另一位用户则表示它变慢了且只能获取 *20 个来源*。
   - 用户指出，与旧版算法相比，新版 **Deep Research** 消耗了更多的服务器资源，但输出结果却更差。
- **在通用场景下 Gemini 2.5 优于 Perplexity O1**：Discord 用户分享了他们的体验，称 [**Gemini 2.5** 免费提供了与 **Perplexity 的 O1 Pro** 类似的质量](https://cdn.discordapp.com/attachments/1047649527299055688/1357423109778702607/image0.jpg?ex=67f02649&is=67eed4c9&hm=d5049580f5523c24bef016f8050e7b92c1f37e1ec416ad9c7ab8b4509c735bf5&)，但 Perplexity 凭借学术搜索在研究论文和严谨科学领域表现更好。
   - 一些用户注意到，Gemini 的深度研究虽然强大，但*容易受到 SEO 作弊网站的影响*，不过它在结合 *YouTube 来源* 方面提供了更好的推理能力。
- **Perplexity 的 AI 助手建议以奇特的方式重组 Android 主屏幕**：当被要求重组 Android 主屏幕时，**Perplexity** 建议*集成交通组件 (widgets)* 并快速访问*酒类商店的营业时间*，这让一位用户怀疑该应用是否假设用户是个酒鬼。
   - AI 甚至提出可以重新组合，以免对对话者造成*冒犯*。
- **举报 LLM Jailbreaks 是徒劳的**：成员们表示，模型厂商无法跟上 Jailbreaks 的速度，而且 Jailbreaks 更多地取决于模型本身，而非 Perplexity 如何对其进行审查。
   - 一位成员表示，*举报 LLM Jailbreaks 就像教一个蹒跚学步的孩子坐定超过五秒钟一样徒劳*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/elmo-sesame-street-shrug-idk-i-have-no-idea-gif-2724737697653756220">Elmo Sesame Street GIF - Elmo Sesame Street Shrug - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://x.com/testingcatalog/status/1897649019309961298?s=46">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：@perplexity_ai 的高级用户将能够加入 "Perplexity Puls Group" 以提供对其回答的反馈，并获得不同的福利，如免费 PPLX 和周边。免费 PPLX？我接受！👀</li><li><a href="https://tenor.com/view/boredmemes-apechain-ape-chain-notacult-gif-1047921344109153463">Boredmemes Apechain GIF - Boredmemes Apechain Ape - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://play.google.com/store/apps/details?id=com.microsoft.translator">Microsoft Translator - Google Play 上的应用</a>：未找到描述</li><li><a href="https://tenor.com/view/dog-gif-20050013">Dog GIF - Dog - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/voices-are-real-jack-nicholson-nod-yes-the-shining-gif-16412513">Voices Are Real Jack Nicholson GIF - Voices Are Real Jack Nicholson Nod - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://open.spotify.com/episode/1vLmnLl3jFQFrx1QvTtg0r?si=oSf4_PuOR72iXHBGtbg_fQ">Aravind Srinivas</a>：Tetragrammaton with Rick Rubin · 剧集</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1jq27a6/why_is_perplexitys_updated_deep_research_slower/">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://github.com/pnd280/complexity">GitHub - pnd280/complexity: ⚡ Supercharge your Perplexity.ai</a>：⚡ 增强你的 Perplexity.ai。通过在 GitHub 上创建账号为 pnd280/complexity 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1357102032393211925)** (6 messages): 

> `Shareable threads, Perplexity AI Search` 


- ****Shareable Threads** 请求！**: Perplexity AI 要求用户确保其线程是**可共享的**。
   - 分享了一个指向 Discord 频道消息的链接作为上下文：[Discord message](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)。
- **大量的 Perplexity AI 搜索链接**: 分享了多个指向 **Perplexity AI 搜索结果**的链接，包括与编写 Perplexity prompt 相关的查询以及基于 AI 的学习资源。
   - 其中一些搜索包括 [废除 IRS](https://www.perplexity.ai/page/trump-aims-to-abolish-irs-.c.0aEbqTJGTtrHHcmWmJg) 和 [AI 学习](https://www.perplexity.ai/search/what-are-some-ai-based-learnin-Q.AhpXIeRoqn3FRp9kf0aA)。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1357444400447357028)** (1 messages): 

> `Perplexity API versioning, API Versioning, Breaking Changes` 


- **API 版本控制最佳实践**: 一位成员抱怨 **Perplexity API** 缺乏版本控制，称：“这是破坏性变更（breaking change），当有客户在使用你的 API 时，你不能在生产环境中这样做。”
   - 他们建议在 API URL 中加入 **/v1/**，这样就可以在不破坏正在使用的 **/v1** 的情况下创建 **/v2/**。
- **版本控制避免破坏性变更**: 用户强调，在没有适当版本控制的情况下引入破坏性变更会给在生产环境中使用 API 的客户带来负面影响。
   - 通过实施版本控制（例如 **/v1/**），开发者可以引入 **/v2/** 而不会干扰 **/v1** 上的现有用户。


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1357069166628307158)** (177 messages🔥🔥): 

> `OpenAI Nonprofit Commission Guidance, Github Copilot and OpenRouter Integration, Google rents Nvidia Blackwell chips from CoreWeave, Inference Scaling and the Log-x Chart, Runway Secures $300M in Series D Funding` 


- **Github Copilot 现在支持即插即用的 OpenRouter 密钥**: 用户现在可以添加 [OpenRouter key](https://openrouter.ai/) 并选择他们在 Github Copilot 中想要的任何模型。
   - 这一集成让 Github Copilot 用户可以访问 OpenAI 提供之外的更广泛的模型。
- **Google 将从 CoreWeave 租赁 Nvidia Blackwell**: Google 正在就从 CoreWeave 租赁 **Nvidia Blackwell** 芯片进行深入谈判，并可能将其 **TPU** 部署在 CoreWeave 的设施中，这凸显了客户对算力的强烈需求 ([The Information 文章](https://www.theinformation.com/articles/google-advanced-talks-rent-nvidia-ai-servers-coreweave))。
   - 此举表明 Google 可能面临 **TPU poor**（TPU 匮乏）的局面，尤其是意识到推理需求将会很高。
- **OpenAI 在 OpenRouter 上秘密发布 Quasar Alpha**: 一个名为 **Quasar Alpha** 的新“秘密模型”在 [OpenRouter](https://openrouter.ai/openrouter/quasar-alpha) 上发布，具有 **1,000,000 context** 且输入/输出 token 免费，被描述为一个强大的全能模型，支持长上下文任务和代码生成。
   - 社区成员推测，鉴于其速度，它可能是一个**开源 SSM**，甚至尽管没有正式公告，也可能是来自 **OpenAI** 的秘密作品，尽管该模型倾向于输出简短的回答和列表。
- **Anthropic 的 CoT 研究对监控提出质疑**: 最新的 [Anthropic 研究](https://www.anthropic.com/research/reasoning-models-dont-say-think)表明，推理模型并不能准确地将其推理过程口语化，这让人怀疑监控**思维链 (CoT)** 是否足以捕捉安全问题。
   - 他们向 **Claude 3.7 Sonnet** 和 **DeepSeek R1** 提供了问题解决提示，然后测试它们的思维链是否会提到使用该提示来得出结论。阅读[相关博客文章](https://www.anthropic.com/research/reasoning-models-dont-say-think)。
- **DeepSeek V3 未能通过前沿模型测试**: 新的 [SEAL 排行榜](http://scale.com/leaderboard)显示，**DeepSeek V3** 并非前沿级（frontier-level）模型，在 Humanity’s Last Exam（仅文本）中排名第 8，在 MultiChallenge（多轮对话）中排名第 12。
   - 尽管不是“前沿级”，一些用户发现它在为 b200 编写 ptx kernels 方面比 **Claude 3.7** 表现更出色。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/bradlightcap/status/1907810330018726042">来自 Brad Lightcap (@bradlightcap) 的推文</a>：ChatGPT 图片功能的第一周非常疯狂——自上周二以来，超过 1.3 亿用户生成了 7 亿多张（！）图片。印度现在是我们增长最快的 ChatGPT 市场 💪🇮🇳 视觉创意的范围已经...</li><li><a href="https://x.com/AnthropicAI/status/1907833407649755298">来自 Anthropic (@AnthropicAI) 的推文</a>：Anthropic 的新研究：推理模型是否能准确地用语言表达其推理过程？我们的新论文显示它们并不能。这让人怀疑监控思维链 (Chain-of-Thought, CoT) 是否足以可靠地...</li><li><a href="https://x.com/tobyordoxford/status/1907379921825014094?s=61">来自 Toby Ord (@tobyordoxford) 的推文</a>：这是修订后的 ARC-AGI 图表。他们将原始 o3 low 的成本估算从每项任务 20 美元增加到了 200 美元。据推测，o3 high 已从 3,000 美元增加到 30,000 美元，这...</li><li><a href="https://x.com/tobyordoxford/status/1907379650831015964?s=61">来自 Toby Ord (@tobyordoxford) 的推文</a>：当我发布关于 o3 极端成本如何使其看起来不如最初那样令人印象深刻的推文时，许多人告诉我这不是问题，因为价格会迅速下降。我检查了...</li><li><a href="https://x.com/AndrewCurran_/status/1907886417088553431">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：皮尤研究中心 (Pew) 今天早上发布的新数据揭示了公众与从事 AI 相关工作和研究的人员之间存在巨大的认知差距。使用情况：66% 的美国公众仍然从未...</li><li><a href="https://x.com/JustinLin610/status/1907748767933280707">来自 Junyang Lin (@JustinLin610) 的推文</a>：Qwen3 的发布时间尚未确定（到处都是传言）。在 QwQ 之后，我们的大部分精力都投入到了新的 Qwen 系列中，目前正处于最后的准备阶段。只是还需要一点时间...</li><li><a href="https://x.com/TheXeophon/status/1907880330985390215">来自 Xeophon (@TheXeophon) 的推文</a>：这是我进行氛围检查 (vibe check) 后的新隐身模型。它目前是最好的非思考模型（至少它没有思考 token...）。输出非常简短，喜欢使用 "Certainly!" 和列表。非常有趣...</li><li><a href="https://x.com/Baidu_Inc/status/1907802772134563892">来自 Baidu Inc. (@Baidu_Inc) 的推文</a>：🚀 ERNIE X1 现已在百度智能云的 MaaS 平台千帆 (Qianfan) 上线，企业用户和开发者现在可以访问其 API！在多个公开数据集的评估中，我们新的深度思考...</li><li><a href="https://x.com/KeyTryer/status/1907504512857944069">来自 Key 🗝 🦊 (@KeyTryer) 的推文</a>：天哪，GitHub Copilot 现在允许我添加 OpenRouter 密钥并选择我想要的任何模型。太重大了。</li><li><a href="https://x.com/steph_palazzolo/status/1907517483524686129">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：新闻：Google 正在就从 CoreWeave 租用 Nvidia Blackwell 芯片以及可能将其 TPU 托管在 CoreWeave 设施中进行深入谈判。该交易凸显了客户对算力的强烈需求...</li><li><a href="https://x.com/AnthropicAI/status/1907833412171207037">来自 Anthropic (@AnthropicAI) 的推文</a>：我们向 Claude 3.7 Sonnet 和 DeepSeek R1 提供了解决问题的提示，然后测试了它们的思维链 (Chains-of-Thought) 是否会提到使用了该提示（如果模型确实使用了它）。阅读博客：https://...</li><li><a href="https://openrouter.ai/openrouter/quasar-alpha">Quasar Alpha - API, Providers, Stats</a>：这是一个提供给社区以收集反馈的隐身模型。它是一个强大的通用模型，支持长上下文任务，包括代码生成。通过 API 运行 Quasar Alpha。</li><li><a href="https://x.com/alexandr_wang/status/1907836081783058720">来自 Alexandr Wang (@alexandr_wang) 的推文</a>：🚨 打破叙事——DeepSeek V3 不是前沿级模型。SEAL 排行榜已更新 DeepSeek V3（2025 年 3 月）的数据。- 在 Humanity’s Last Exam（仅限文本）中排名第 8。- 在 MultiChallenge（多...</li><li><a href="https://techcrunch.com/2025/03/20/perplexity-is-reportedly-in-talks-to-raise-up-to-1b-at-an-18b-valuation/">据报道，Perplexity 正在洽谈以 180 亿美元的估值筹集高达 10 亿美元的资金 | TechCrunch</a>：AI 驱动的搜索初创公司 Perplexity 据称正处于早期谈判阶段，拟在新一轮融资中筹集高达 10 亿美元，估值为 180 亿美元。</li><li><a href="https://singularityhub.com/2015/01/26/ray-kurzweils-mind-boggling-predictions-for-the-next-25-years/#sm.001tlz026ghlfl9114t1t2yxvo5lq)">Ray Kurzweil 对未来 25 年的惊人预测</a>：在我的新书《BOLD》中，我最兴奋的采访之一是与我的好朋友 Ray Kurzweil 的对话。Bill Gates 称 Ray 为“我认识的最棒的人...</li><li><a href="https://runwayml.com/news/runway-series-d-funding">Runway 新闻 | 迈向拥有世界模拟器 (world simulators) 的新媒体生态系统</a>：未找到描述</li><li><a href="https://x.com/eric_haibin_lin/status/1907845598432342328">推文来自...</a>

来自 Haibin (@eric_haibin_lin)</a>：我们正在开源 bytecheckpoint 和 veomni！bytecheckpoint 是字节跳动的生产级基础模型训练 checkpoint 系统，经过了 1 万多张 GPU 任务的实战测试。极速...</li><li><a href="https://github.com/ByteDance-Seed/ByteCheckpoint">GitHub - ByteDance-Seed/ByteCheckpoint: ByteCheckpoint: An Unified Checkpointing Library for LFMs</a>：ByteCheckpoint：一个面向 LFMs 的统一 Checkpointing 库 - ByteDance-Seed/ByteCheckpoint</li><li><a href="https://github.com/ByteDance-Seed/VeOmni">GitHub - ByteDance-Seed/VeOmni: VeOmni: Scaling any Modality Model Training to any Accelerators with PyTorch native Training Framework</a>：VeOmni：使用 PyTorch 原生训练框架将任何模态模型训练扩展到任何加速器 - ByteDance-Seed/VeOmni</li><li><a href="https://techcrunch.com/2025/04/03/microsoft-reportedly-pulls-back-on-its-data-center-plans/">据报道，微软缩减了其数据中心计划 | TechCrunch</a>：据报道，微软已经缩减了全球范围内的数据中心项目，这表明该公司对过度扩张持谨慎态度。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1357151851841192047)** (10 条消息🔥): 

> `Joanne Jang, GPT-4o Transcribe, ChatGPT 4o ImageGen Watermark` 


- **一路 Jang 到播客**：一位成员在 OpenAI 遇到了 **Joanne Jang**，并表达了希望邀请她参加播客的愿望，但也注意到由于 Jang 与 OpenAI 的隶属关系，可能会受到限制。
   - 他们表示有信心播客访谈*最终*会实现。
- **GPT-4o Transcribe 产生幻觉！**：一位用户分享了一条推文，指出 **GPT-4o Transcribe** 会幻听出 *Transcript by PODTRANSCRIPTS, COM* 这句话。
   - 另一位成员回复道“又来了”，这可能指的是过去 AI 转录服务中出现的类似问题。
- **水印门困扰 ChatGPT ImageGen**：一位用户分享了一条推文，内容是关于使用 **ChatGPT 4o ImageGen** 创建的图像上出现了新水印，以及一个提到“带水印资产指针（watermarked asset pointer）”的实验性功能。
   - 另一位成员提到，水印可以在 **200 美元档位**（或者至少目前在文件中是这么显示的）中移除。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/matthen2/status/1907477758789218796">Matt Henderson (@matthen2) 的推文</a>：gpt-4o-transcribe 确实很有趣……还有人看到它产生“Transcript by PODTRANSCRIPTS , COM”的幻觉吗？</li><li><a href="https://x.com/btibor91/status/1907861559029682323">Tibor Blaho (@btibor91) 的推文</a>：新水印预览（见于 LinkedIn）。引用 Tibor Blaho (@btibor91)：使用 ChatGPT 4o ImageGen 创建的图像可能很快就会包含水印。最近对 ChatGPT Web 应用的更新引入了...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1357405163245011135)** (40 条消息🔥): 

> `Devin 2.0, Agent-based IDEs, Windsurf vs Cursor, Claude-code API, Polars updates` 


- **Cognition Labs 发布 Devin 2.0**：**Cognition Labs** 推出了 [Devin 2.0](https://fxtwitter.com/cognition_labs/status/1907836719061451067)，这是一种全新的原生 Agent IDE 体验，现已正式开放，起售价为 **20 美元**加按量付费。
   - 一些成员觉得这次发布“太搞笑了”，因为竞争对手可能在 **Devin** 之前就找到了 PMF。
- **Windsurf 被誉为比 Cursor 更好的 Agent**：一些成员发现 **Windsurf** 在 Agent 方面的处理比 **Cursor** 更好，而 **Cursor** 则侧重于减少让 LLM 处理枯燥但耗时任务的摩擦。
   - 其他人表示赞同，但表示问题在于它是否比单独开一个窗口有显著提升，而答案是“呃，几乎没有”。
- **Claude-code API 非常烧钱**：一位成员建议使用 **claude-code** 来生成基准测试数据、仪表盘和临时 UI，因为 LLM 擅长这些且非常值得。
   - Twitter 上一些疯狂的人正在并行运行多个 **Claude code** 实例，*一个周末就烧掉了 500 美元*。
- **Claude 在 Polars 中表现不错**：一位成员分享说，**Claude** 和 **Gemini** 在 **Polars** 中表现不俗，并指出“你得告诉它现在要用 `with_columns` 了”。
   - 还有人提到 **Claude 3.7** 完全不知道如何使用新更新，而且实际权重中存在竞争信息，这使得通过上下文来克服变得更加困难。



**提到的链接**：<a href="https://fxtwitter.com/cognition_labs/status/1907836719061451067">Cognition (@cognition_labs) 的推文</a>：介绍 Devin 2.0：一种全新的原生 Agent IDE 体验。今天正式开放，起售价 20 美元。🧵👇

  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1357107804774862988)** (53 messages🔥): 

> `Distilling Reasoning Capabilities, Superhuman AI Impact Prediction, Algorithmic progress vs data progress, Dwarkesh AGI Forecast Podcast, Nvidia Open Code Reasoning Collection` 


- **学生模型弥合推理差距**：最近的一篇论文 ([arxiv.org/abs/2504.01943](https://arxiv.org/abs/2504.01943)) 强调了将推理能力蒸馏到学生模型中的成功，在 **coding tasks**（编程任务）上弥合了推理模型与标准 LLM 之间的差距。
   - 蒸馏后的模型仅使用 SFT 就在 LiveCodeBench 上达到了 **61.8%**，在 CodeContests 上达到了 **24.6%**，超越了使用强化学习训练的其他方案。
- **预测十年内将产生超人类 AI 影响**：一种情景预测 ([ai-2027.com](https://ai-2027.com/)) 认为，**superhuman AI** 在未来十年内将产生巨大影响，超过工业革命。
   - 该预测基于*趋势外推、兵棋推演、专家反馈、在 OpenAI 的经验以及之前的预测成功案例*。
- **Dwarkesh AGI 时间线引发辩论**：讨论了 Dwarkesh Patel 的 AGI 预测播客，一些人认为其预测和假设缺乏说服力。
   - 一位成员对缺乏对 Scaling Laws 的认可以及强调算法进步而非数据进步表示困惑，而其他人则将其描述为*同人小说 (fanfiction)*。
- **Nvidia 空的推理数据集合**：一位成员分享了 NVIDIA 在 Hugging Face 上用于开放代码推理的集合链接 ([huggingface.co](https://huggingface.co/collections/nvidia/opencodereasoning-67ec462892673a326c0696c1))，旨在推进竞赛编程的数据蒸馏。
   - 另一位成员指出，该集合目前是空的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2504.01943">OpenCodeReasoning: Advancing Data Distillation for Competitive Coding</a>：自从推理型大语言模型出现以来，许多人通过将推理能力蒸馏到学生模型中取得了巨大成功。此类技术显著弥合了差距……</li><li><a href="https://lancelqf.github.io/note/llm_post_training/">From REINFORCE to Dr. GRPO</a>：LLM Post-training 的统一视角</li><li><a href="https://huggingface.co/collections/nvidia/opencodereasoning-67ec462892673a326c0696c1">OpenCodeReasoning - a nvidia Collection</a>：未找到描述</li><li><a href="https://ai-2027.com/">AI 2027</a>：一项有研究支持的 AI 场景预测。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[expensive-queries](https://discord.com/channels/1179127597926469703/1338919429752361103/1357318149086646325)** (2 messages): 

> `OpenAI Deep Research, Plumbing Repair Costs` 


- **OpenAI Deep Research 为用户节省了 2,050 美元的管道维修费用**：X 上的一位用户发布了 [OpenAI Deep Research](https://x.com/jbohnslav/status/1907759146801197450) 如何帮助他们找到一位收费 **200 美元** 的管道工，而最初的报价是 **2,250 美元**。
- **Deep Research：淘货者的 AI 工具**：该用户开玩笑说 OpenAI Pro *简直帮我省了 2,050 美元，几乎够付一整年的订阅费了！*



**提到的链接**：<a href="https://x.com/jbohnslav/status/1907759146801197450">来自 Jim Bohnslav (@jbohnslav) 的推文</a>：得到了一个简单的管道维修报价：2,250 美元。询问 OpenAI Deep Research 市场价：300-500 美元。询问 DR 我所在地区的优秀管道工。给第一个打电话。200 美元修好。OpenAI Pro 简直……

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1357072717660291265)** (170 messages🔥🔥): 

> `Gemini 2.5 Pro 速率限制、Architect Mode 优化、语音命令配置、针对 LSP 和 Treesitter 的 MCP` 


- **Gemini 2.5 Pro 触发速率限制困扰**：用户报告在 Aider 中使用 Gemini 2.5 Pro 时遇到了 **每分钟 20 次请求 (20 RPM) 的速率限制**，即使使用量极低，怀疑是 Aider 的后台请求所致。
   - 一些用户即使拥有 Tier 1 API 密钥，看到的限制也是 **5 RPM**，而其他用户则报告看到了文档中 Tier 1 对应的 **20 RPM**，并提供了类似 [此图](https://cdn.discordapp.com/attachments/1131200896827654149/1357114156037312683/image.png?ex=67efaf4c&is=67ee5dcc&hm=ab00c0d89a9a4029e1244032c897f52cf418c2b5c10a03543f8574d73b779750&) 的截图。
- **Architect Mode 节省 Gemini 配额**：为了在 Architect Mode 中节省 Gemini 2.5 Pro 的配额，一位用户建议设置 `--editor-model sonnet`，将编辑任务卸载给像 **Sonnet** 这样更便宜的模型。
   - 一位成员表示：*你可以尝试 Haiku，我想……但即使只是让 3.7 Sonnet 负责编辑，价格也非常低廉*。
- **语音命令需要提供商配置**：用户正在寻找配置选项，以便为 `/voice` 命令选择语音模型和提供商，目前该命令固定使用 **OpenAI Whisper**。
   - 一个待处理的 PR ([https://github.com/Aider-AI/aider/pull/3131](https://github.com/Aider-AI/aider/pull/3131)) 可能会解决这个问题，允许使用不同的提供商和模型。
- **Code Actions 需要 MCP 支持**：一位用户建议 **LSP Code Actions 和 tree-sitter** 可以改进代码编辑和重构。
   - 该成员进一步指出，需要 *针对 LSP 和 treesitter 的 MCP 来进行代码编辑，这将加快这些简单但大规模编辑的速度并提高其鲁棒性*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/a7m7s1p6dv20/status/1907684868164825260?s=46">ᅟ (@a7m7s1p6dv20) 的推文</a>：Gemini 2.5 Pro 通过 glama AI 的（初步？）定价方案</li><li><a href="https://x.com/tom_doerr/status/1907450456575533269">Tom Dörr (@tom_doerr) 的推文</a>：看着多个 Agent 在同一个项目上工作，每个 Agent 承担不同的任务，修复 Bug 并解决合并冲突以推回 main 分支，真是令人难以置信地满足。</li><li><a href="https://smithery.ai/server/@smithery-ai/github">Github | Smithery</a>：未找到描述</li><li><a href="https://smithery.ai/server/@smithery-ai/server-sequential-thinking">Sequential Thinking | Smithery</a>：未找到描述</li><li><a href="https://x.com/OpenRouterAI/status/1905300582505624022">OpenRouter (@OpenRouterAI) 的推文</a>：为了最大化你的免费 Gemini 2.5 配额：1. 在 https://openrouter.ai/settings/integrations 中添加你的 AI Studio API 密钥。我们的速率限制将作为你的“浪涌保护器”。2. 在你的 ... 中设置 OpenRouter。</li><li><a href="https://smithery.ai/server/@IzumiSy/mcp-duckdb-memory-server">DuckDB Knowledge Graph Memory Server | Smithery</a>：未找到描述</li><li><a href="https://ai.google.dev/gemini-api/docs/rate-limits">无标题</a>：未找到描述</li><li><a href="https://smithery.ai/server/@PhillipRt/think-mcp-server">Think Tool Server | Smithery</a>：未找到描述</li><li><a href="https://aider.chat/docs/faq.html#what-llms-do-you-use-to-build-aider,">FAQ</a>：关于 aider 的常见问题。</li><li><a href="https://ai.google.dev/gemini-api/docs/billing">无标题</a>：未找到描述
</li>
</ul>

</div>

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1357090980037398730)** (14 条消息🔥): 

> `Aider Shell, Openrouter Errors, Git Repo Corrupted, Aider Print Prompt Costs, Gemini Comments` 


- ****Aider** 的 Shell 揭秘！**：一位用户询问了 **Aider** 在执行调试 Docker 相关问题的命令时使用的是哪种 Shell。
   - 该用户观察到 **Aider** 的 `curl` 命令成功了，而他们自己的 Shell (`bash`) `curl` 命令却失败了，从而引发了这一疑问。
- ****Openrouter** 的 500 错误困扰用户**：用户报告在使用 `openrouter/google/gemini-2.5-pro-exp-03-25:free` 时遇到了 `litellm.BadRequestError`，具体表现为 `KeyError: 'choices'` 和内部服务器错误（Internal Server Error，代码 500）。
   - 这些错误是间歇性的，导致其根本原因尚不明确。
- ****Git 仓库损坏**引发关注！**：多位用户遇到了 "Unable to list files in git repo: BadObject" 错误，引发了对潜在 **Git 仓库损坏**的担忧。
   - 错误信息建议检查 Git 仓库是否损坏，但未提供即时的解决方案。
- ****Gemini** 的注释过载！**：一位用户难以阻止 **Gemini/gemini-2.5-pro-exp-03-25** 在各处添加注释，尽管尝试通过 [GLOBAL_CONVENTIONS.md](https://github.com/schpet/dotfiles/blob/main/.config/aider/GLOBAL_CONVENTIONS.md) 进行配置。
   - 然而，另一位用户称赞了 Gemini 的总体结果质量，但对其配额限制表示遗憾。
- **通过 `--watch-files` 最大化 **Aider** 的效能！**：一位刚完成基础设置的 **Aider** 新用户询问了使用 `--watch-files` 模式的优势。
   - 该用户最初面临 `ai!` 和保存被忽略的问题，但报告称他们已经解决了这个问题。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1357080838394875975)** (12 条消息🔥): 

> `Refact Polyglot Claims, Aider Polyglot Benchmark, SWE-bench evaluation, OpenAI's PaperBench` 


- **Refact 在 Aider Polyglot 基准测试中获得高分**：**Refact** 声称在 [Aider polyglot benchmark](https://medium.com/@refact_ai/refact-ai-agent-scores-highest-on-aiders-polyglot-benchmark-93-3-00ed0e3b9a6b) 中获得了 **92%** 的分数，引发了关于其真实性和成本的讨论。
   - 一位成员建议调查达到如此高分的成本，并表示如果属实，这将非常有价值。
- **对 Refact AI 结果的怀疑和不信任**：一位成员表示有兴趣在免费或廉价模型上使用更大的 **--tries** 值运行基准测试，以评估它们的价值。
   - 另一位成员指出，[Aider Polyglot Benchmark](https://github.com/paul-gauthier/aider) 并不是测试自主 Agent 的正确方法，而是建议使用 [SWE-bench](https://github.com/princeton-nlp/SWE-agent)。
- **Aider 在 SWE-bench 上的表现受到质疑**：一位成员询问了 **Aider** 在 **SWE-bench** 上的表现，并质疑为什么要在那进行基准测试。
   - 另一位成员澄清说，**SWE-bench** 是为自主 Agent 设计的，而 **Aider polyglot** 是专门为测试模型与 Aider 配合而定制的，并指出 **Aider** 最近没有在 **SWE-bench** 上提交分数。
- **将 Aider 适配到 OpenAI PaperBench 评估？**：一位成员建议将 **Aider** 适配到新的 [OpenAI PaperBench evaluation benchmark](https://openai.com/index/paperbench/)。
   - 关于该话题没有进一步的讨论。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1357068008492892291)** (120 条消息🔥🔥): 

> `LM Studio 适配 Brave, 本地服务器中的 System Prompt, CUDA0 缓冲区分配失败, Q4 与 Q6 模型质量对比, LM Studio 双 GPU 设置` 


- **本地 LLM 的 Brave 新集成**：用户探索了将 **LM Studio** 与 **Brave** 浏览器集成，将其服务器端点指向 `http://localhost:1234/v1/chat/completions`，并寻求关于配置 **API** 以使用 system prompts 的指导，[lmstudioservercodeexamples](https://github.com/YorkieDev/lmstudioservercodeexamples) 是一个很有帮助的资源。
   - 许多人在为 Brave 提供正确的 API 端点时遇到困难。
- **System Prompt：解锁 LLM 潜力的 API 关键**：要在 **LM Studio** 的本地服务器上使用 **system prompts**，用户需要通过 **API 调用** 提供 prompt，而不是通过 LM Studio 界面，文档可在[此处](https://lmstudio.ai/docs/app/api)查看。
- **CUDA 难题：内存混乱再次袭来**：'failed to allocate cuda0 buffer' 错误通常表示所使用的模型内存不足，此外从 **HF mirror** 下载时缺少 **mmproj** 文件也可能导致此问题，可以通过在 **LM Studio** 内启用代理设置进行下载来解决。
- **Q4 vs Q6：质量困惑**：**Q4** 和 **Q6 模型**之间的质量下降取决于模型的大小和新旧程度，较旧和较小的模型受影响更明显，但 **32B 模型**不应有显著影响。
- **双 GPU：即插即用的动力？**：用户报告称，在 **LM Studio** 中利用双 **GPU (4090 + 5090)** 出奇地简单，在开启 flash attention 的 **32B Q8 模型**上达到了良好的性能（24-25 tokens/s），尽管在没有 **NVLink** 的情况下拆分模型时，性能可能会受到 **PCIE 连接**的瓶颈限制。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/docs/app/api">将 LM Studio 作为本地 LLM API 服务器 | LM Studio 文档</a>：使用 LM Studio 在 localhost 上运行 LLM API 服务器</li><li><a href="https://tenor.com/view/meme-horrors-beyond-our-comprehension-low-tier-god-mods-banned-gif-8530537273735940092">超越我们理解的恐怖模因 GIF - Meme Horrors beyond our comprehension Low tier god - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://lmstudio.ai/work">在工作中使用 LM Studio</a>：在您的工作场所或组织中使用本地 LLM</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSd-zGyQIVlSSqzRyM4YzPEmdNehW3iCd3_X8np5NWCD_1G3BA/viewform?usp=sf_link)">LM Studio @ Work</a>：感谢您对在工作中使用 LM Studio 感兴趣！请填写以下表格，我们会尽快回复您。- Team LM Studio (team@lmstudio.ai)
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1357079204750954607)** (36 条消息🔥): 

> `Unsloth 2.0 6b 性能, M3 Ultra vs M4 Max (LLM), Mac 用于 LLM, Qwen QWQ 质量, GPU vs Apple Silicon 基准测试` 


- **Unsloth 2.0 6b 尽管速度慢但能解决问题**：一位用户报告称，在 4x 3090 + 256GB RAM 上以约 3 tok/s 的速度运行 **Unsloth 2.0 6b**，在较小模型和 **ChatGPT** 失败的情况下，它在 20-30 分钟内解决了一个编程问题。
   - 他们发现 **Qwen QWQ** 在参数量仅为 **R1** 的 5% 的情况下，达到了其 90% 的质量，强调了对质量而非速度的偏好。
- **M3 Ultra 对 LLM 表现一般，M4 Max 则非常出色！**：一位用户表示，由于内存、计算和带宽不平衡，**M3 Ultra Mac Studio** 在 **LLM** 用途上表现糟糕，而 **M4 Max** 和 **5090** 则非常出色。
   - 他们认为 **M3 Ultra** 的大容量 VRAM 仅适用于巨大的 MoE 模型，对于能装进 **5090** 的 **32GB VRAM** 或 **M4 Max** 的 **96GB** 的较小模型来说，其价格过高。
- **关于 Apple Silicon 带宽的讨论**：一位用户澄清说，**M3 Ultra** 为 512GB VRAM 提供 800GB/s 的带宽，而 **5090** 为 32GB 提供 1792GB/s，**M4 Max** 为 128GB 提供 546GB/s。
   - 另一位用户指出，认为 **M3 Ultra** 因为“不平衡”而“糟糕”的说法是荒谬的，因为它仍然比 **M4 Max** 拥有更高的带宽，正如之前所有的 Ultra 版本一样。
- **M4 Max vs M1 Ultra 基准测试**：一位成员引用了 [llama.cpp 讨论](https://github.com/ggml-org/llama.cpp/discussions/4167)，显示在受带宽限制的小型模型中，**M4 Max** 和 **M1 Ultra** 基本持平，而在较大的量化模型中，**M1** 表现更佳。
   - 一位用户分享了一个包含 LLM 推理 [GPU 基准测试的 GitHub 仓库](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)。



**提到的链接**：<a href="https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference">GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?</a>: 多个 NVIDIA GPU 还是 Apple Silicon 用于大语言模型推理？ - XiongjieDai/GPU-Benchmarks-on-LLM-Inference

  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1357182516234158181)** (49 条消息🔥): 

> `API 中的 Web Search 引用，Quasar Alpha 隐身模型，Inference Net 端点已禁用，针对 Coding 优化的模型` 


- ****OpenRouter 为 API 添加 Web Search 引用****：OpenRouter 宣布 [Web Search 现在在 API 中返回引用](https://x.com/OpenRouterAI/status/1907623560522379436)，并在所有模型中实现了标准化，包括 **OpenAI** 和 **Perplexity** 等原生在线模型。
   - 用户可以访问 [文档](https://openrouter.ai/docs/features/web-search)，通过激活 `web` 插件或在模型标识符（slug）后添加 `:online` 来整合 Web Search 结果。
- ****Quasar Alpha：1M 上下文隐身模型亮相****：OpenRouter 在公开发布前宣布了 [Quasar Alpha](https://openrouter.ai/openrouter/quasar-alpha)，这是一个 **免费** 的、**1M token** 上下文长度的模型，针对 Coding 进行了优化，但同时也适用于通用场景。
   - Prompt 和 Completion 将被记录，用户可以在 [专用 Discord 频道](https://discord.com/channels/1091220969173028894/1357398117749756017) 中提供反馈以帮助改进模型。
- ****Inference Net 端点暂时停用****：由于平台维护，OpenRouter 上的 Inference Net 端点将暂时禁用，并承诺很快恢复。
- ****Quasar Alpha：初步基准测试表现良好****：根据部分用户的反馈，**Quasar Alpha** 的初步基准测试表现良好，尽管也有人发现它在 Coding 测试中表现不佳。
   - 一位用户在 [X 上分享了氛围测试（vibe check）](https://x.com/TheXeophon/status/1907880330985390215/photo/1)，称其为表现最好的非思考型（non-thinking）模型，输出非常简短；而其他人则对其来源进行了猜测，有人怀疑它可能是新的 Qwen 变体。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1907880330985390215/photo/1">来自 Xeophon (@TheXeophon) 的推文</a>：这是我氛围测试中的新隐身模型。它目前是最好的非思考型模型（至少它没有 thinking token...）。输出非常简短，它喜欢 Certainly! 和列表。非常有趣...</li><li><a href="https://x.com/OpenRouterAI/status/1907870610602275203">来自 OpenRouter (@OpenRouterAI) 的推文</a>：很高兴宣布我们的首个“隐身”模型... Quasar Alpha 🥷 它是来自某模型实验室即将推出的长上下文基础模型的预发布版本：- 1M token 上下文长度 - 专门针对...</li><li><a href="https://openrouter.ai/openrouter/quasar-alpha">Quasar Alpha - API、提供商、统计数据</a>：这是一个提供给社区以收集反馈的隐身模型。它是一个强大的全能模型，支持包括代码生成在内的长上下文任务。通过 API 运行 Quasar Alpha</li><li><a href="https://x.com/OpenRouterAI/status/1907623560522379436">来自 OpenRouter (@OpenRouterAI) 的推文</a>：一个呼声很高的功能：Web Search 现在在 API 中返回引用 🌐 我们已为所有模型实现了标准化，包括 OpenAI 的 Web 工具和 Perplexity 等原生在线模型：</li><li><a href="https://openrouter.ai/docs/features/web-search">Web Search - 为 AI 模型提供实时 Web Grounding</a>：在你的 AI 模型响应中启用实时 Web Search 功能。通过 OpenRouter 的 Web Search 功能为任何模型的输出添加真实的、最新的信息。
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1357320733528821842)** (1 条消息): 

> `AI 角色平台，charactergateway.com` 


- **Character Gateway 为开发者发布**：一个名为 [Character Gateway](https://charactergateway.com/) 的新 AI 角色平台已发布，旨在为开发者提供创建、管理和部署 **AI 角色/Agent** 的工具。
   - 该平台强调简洁性，具有 *无需数据库、无需 Prompt Engineering、无需订阅 [且] 无需新 SDK* 的特点。
- **Character Gateway API 支持 Chat Completion**：Character Gateway 允许用户生成角色和图像，并使用自己的 **OpenRouter** 密钥向角色发送 **/chat/completion 请求**。
   - 该平台不计量 Token 使用情况，让开发者能更好地控制成本；列出热门公共角色并将其集成到 App 中的功能正在开发中（**WIP**）。



**提到的链接**：<a href="https://charactergateway.com/">Character Gateway</a>：面向开发者的 AI 角色 API 平台

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1357067303904350278)** (99 条消息🔥🔥): 

> `Gemini 2.5 Pro, Image responses, OpenAI Responses API, Targon Speed, Anthropic Blocking` 


- **Google 的 Gemini 2.5 Pro 评价褒贬不一**：一些用户在质疑 **Gemini 2.5 Pro** 是否对他们有效。
   - 有人指出，Google 托管的免费模型通常有非常低的 rate limits，不过用户可以通过使用自己的 API key 来绕过这一限制。
- **图像响应功能正在开发中**：OpenRouter 正在积极开发对图像响应的支持，可能会使用新的 **Responses API**。
   - 有推测称 OpenRouter 的接口未来可能会与 OpenAI 的接口分道扬镳，甚至可能发布 SDK。
- **OpenAI 的 API 引发了关于未来兼容性的讨论**：OpenRouter 开发者正在考虑增加对 **OpenAI Responses API** 的支持，特别是考虑到 OpenAI 可能会逐渐弃用 chat completions。
   - 一位成员计划迁移到 Responses API，因为觉得它*更标准、更一致且设计更出色*。
- **Targon 的速度受到质疑**：用户正在讨论 **Targon 的速度** 是否是因为矿工可能忽略了采样参数，从而导致分布偏差，并引用了 [GitHub 上的 verifier.py](https://github.com/manifold-inc/targon/blob/main/verifier/verifier.py)。
   - 一位成员指出，*他们只生成一次结果并进行缓存，所以如果你问同样的问题，即使你更改了参数，他们也会返回相同的回复*。
- **用户发现 Anthropic 价格昂贵**：一位用户分享了他的沮丧经历：他的 AI 模型在未被察觉的情况下切换到了 **Anthropic**，导致产生了意外费用，尽管他已将其列入忽略供应商名单。
   - 另一位成员表示 Anthropic *对我来说不值这个钱——永远都不会值*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/mistralai/mistral-small-3.1-24b-instruct">Mistral Small 3.1 24B - API, Providers, Stats</a>：Mistral Small 3.1 24B Instruct 是 Mistral Small 3 (2501) 的升级版本，拥有 240 亿参数并具备先进的多模态能力。通过 API 运行 Mistral Small 3.1 24B。</li><li><a href="https://openrouter.ai/mistralai/mistral-sm">Discord</a>：未找到描述</li><li><a href="https://openrouter.ai/models?fmt=cards&input_modalities=image&order=newest&max_price=0">Models | OpenRouter</a>：在 OpenRouter 上浏览模型</li><li><a href="https://github.com/manifold-inc/targon/blob/main/verifier/verifier.py">targon/verifier/verifier.py at main · manifold-inc/targon</a>：一个用于使用 manifold 奖励栈构建子网的库 - manifold-inc/targon
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1357071009647300842)** (102 条消息🔥🔥): 

> `Paid frontier models in production, vLLM/TGI Setup with RTX 5000, GPU server costs, Counterfeit detection with VLMs, Chat templates in training`

- **RTX 5000 系列用户遇到 vLLM/TGI 安装问题**：成员们在新的 **RTX 5000** 系列显卡上设置 **vLLM** 或 **TGI** 时遇到了麻烦，他们需要 **PyTorch** 的 nightly 版本和 **CUDA 12.8**，但这并不容易实现……
   - 正如一位成员所述，*当你安装其他组件时，PyTorch 会被旧版本覆盖*。他指向了这些 GitHub 仓库以寻求帮助：[vllm-project/vllm/issues/14452](https://github.com/vllm-project/vllm/issues/14452), [pytorch/My-rtx5080-gpu-cant-work-with-pytorch/217301](https://discuss.pytorch.org/t/my-rtx5080-gpu-cant-work-with-pytorch/217301), [lllyasviel/stable-diffusion-webui-forge/issues/2601](https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/2601), [ComfyUI/discussions/6643](https://github.com/comfyanonymous/ComfyUI/discussions/6643)。
- **VLM 检测时尚仿冒品**：成员们分享了关于仿冒产品的研究，并展示了一个基于深度神经网络的计算机视觉系统，声称在剔除品牌服装后准确率达到 **99.71%**。
   - 他们引用了[这篇论文](https://arxiv.org/abs/2410.05969)，该论文指出该系统不需要特殊的安全标签或对供应链追踪进行修改，并且仅通过少量真假物品的迁移训练（transfer-trained）即可实现。
- **Hugging Face 透明度待提高**：成员们对 Hugging Face 的计费和配额系统，以及 **GPU Spaces、Zero GPU Spaces、Serverless Inference API** 的服务使用情况表示困惑。
   - 他们希望 HF 能针对重大变更提供“报告、沟通和咨询”，例如发布公告称：*“我们将实施一项重大变更，未来几天可能会不稳定”*。
- **Chat Templates 现在可用于训练模型**：一位成员询问是否可以将 **chat_template** 传递给 **transformers** 的 **TrainingArguments** 或 **Trainer**，以便在推理阶段为模型使用自定义的 chat_template，并询问在训练中使用是否有意义。
   - 另一位成员确认这是可行的，并[链接到了文档](https://huggingface.co/docs/transformers/main/en/chat_template_basics#can-i-use-chat-templates-in-training)，解释说 chat templates 是纯文本 **LLM** 的 tokenizer 或多模态 **LLM** 的 processor 的一部分，用于指定如何将对话转换为单个可 token 化（tokenizable）的字符串。
- **RAG 代码实现仅需几行**：一位成员询问为公司实现 **RAG** 技术需要多少行代码。
   - 其他成员回答说只需要*几行——大概 15 到 30 行左右*，并且他们将信息存储在 **MongoDB** 中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lightning.ai/">Lightning AI | 将创意转化为 AI，闪电般的速度</a>：AI 开发的一体化平台。协作编码、原型设计、训练、扩展、部署。直接在浏览器中完成，无需安装。由 PyTorch Lightning 的创建者打造。</li><li><a href="https://arxiv.org/abs/2410.05969">基于深度神经网络的智能手机图像仿冒品检测</a>：仿冒产品（如药品、疫苗以及高档时尚手袋、手表、珠宝、服装和化妆品等奢侈品）给合法商家带来了巨大的直接收入损失……</li><li><a href="https://huggingface.co/posts/Reality123b/155118307932581">Hugging Face 上的 @Reality123b：“好吧，肯定出问题了。HF 收了我 0.12 美元，仅因为 3 次推理请求……”</a>：无描述</li><li><a href="https://tenor.com/view/cat-kawaii-gif-13992966100210966399">Cat Kawaii GIF - CAT Kawaii - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_template_basics#can-i-use-chat-t">文本 LLM 的 Chat Templates 入门指南</a>：无描述</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_template_basics#can-i-use-chat-templates-in-training">文本 LLM 的 Chat Templates 入门指南</a>：无描述</li><li><a href="https://huggingface.co/docs/datasets/v3.5.0/en/package_reference/loading_methods#datasets.load_dataset.path">加载方法</a>：无描述</li><li><a href="https://huggingface.co/docs/datasets/v3.5.0/loading">加载</a>：无描述</li><li><a href="https://github.com/huggingface/huggingface_hub/issues/2118">限制下载速度 · Issue #2118 · huggingface/huggingface_hub</a>：您的功能请求是否与问题相关？请描述。下载模型时，huggingface-cli 会开启多个连接并完全占满连接带宽。正因如此……</li><li><a href="https://discuss.huggingface.co/t/download-speeds-slow-on-the-popular-models/84840/5">热门模型的下载速度缓慢</a>：同样的情况，今天大多数下载似乎都限制在 10.4MBps：
</li>
</ul>

</div>

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1357094060245913720)** (4 messages): 

> `Hugging Face Token 设置, Jupyter Notebook 配置, LlamaIndex 基础` 


- **Hugging Face Token 工作流详解**：一位成员申请了模型访问权限，随后生成了 **HF_TOKEN**，在启动 **Jupyter** 之前在终端中将其导出，将 agents-course 仓库克隆到本地分支，并为 **Jupyter** 创建了一个名为 hfenv 的 **Python venv**。
   - 他们还安装了 Jupyter 和 huggingface_hub，将 **hfenv kernel** 添加到 Jupyter，在 notebook 中选择了该内核，运行并保存了 notebook。
- **深入研究 LlamaIndex 基础**：一位成员正在学习 **LlamaIndex** 的基础知识，并在查看课程后寻求关于下一步学习方向的建议。
- **依赖发现**：一位成员注意到在 huggingface_hub 文档中发现了针对 torch, cli, tensorflow 等额外的 **pip install 依赖项**。
   - 他们当时正在查阅其他人之前提到的文档。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1357111586971717666)** (11 messages🔥): 

> `目标检测模型, 端到端项目, 操作系统事件流向 AI, 关于 AI 的 AI 游戏, TypeScript 语音助手` 


- ****签名检测模型**端到端项目**：一位成员分享了一个**端到端目标检测模型**项目的[文章](https://huggingface.co/blog/samuellimabraz/signature-detection-model)和[推理服务器](https://github.com/tech4ai/t4ai-signature-detect-server)。
   - 该项目使用 **Optuna** 进行超参数调优，实现了 **7.94% 的 F1-score 提升**，并部署了带有 **OpenVINO CPU 后端**的 **Triton Inference Server** 以优化推理。
- ****操作系统事件流向 AI****：讨论了一个将操作系统事件流式传输到 AI 的 Rust 库，并链接到了 [GitHub 仓库](https://github.com/mediar-ai/ui-events/tree/main)。
   - 一位成员认为它非常*棒*且*实用*。
- ****关于 AI 的游戏**登场**：一位成员用 AI 制作了一个关于 AI 的游戏，引发了关于从 RSS 提要生成电子邮件摘要的讨论。
   - 另一位成员建议使用 **smolagent** 来实现自定义用户过滤，从而根据摘要生成**播客**，扩展了最初的想法。
- **认识 **TySVA**，TypeScript 语音助手**：一位成员介绍了 **TySVA** ([GitHub 仓库](https://github.com/AstraBert/TySVA))，这是一个利用 Model Context Protocol (**MCP**) 处理日常 TypeScript 编程任务的 TypeScript 语音助手。
   - 它集成了 **Qdrant**, **HuggingFace**, **Linkup**, **LlamaIndex**, **ElevenLabs**, **Groq**, **Gradio** 和 **FastAPI**，为用户问题提供基于事实的回答以及解决方案的语音总结。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://idleai.xenovative-ltd.com:5000/">即将推出</a>：未找到描述</li><li><a href="https://github.com/mediar-ai/ui-events/tree/main">GitHub - mediar-ai/ui-events: 将操作系统事件流式传输到 AI 的库</a>：将操作系统事件流式传输到 AI 的库。通过在 GitHub 上创建账号为 mediar-ai/ui-events 的开发做出贡献。</li><li><a href="https://github.com/AstraBert/TySVA">GitHub - AstraBert/TySVA: 轻松与 AI 聊天学习 TypeScript</a>：轻松与 AI 聊天学习 TypeScript。通过在 GitHub 上创建账号为 AstraBert/TySVA 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1357217285198184500)** (2 messages): 

> `视频转 3D 人体网格重建仓库, OWLv2 的图像引导检测模式问题` 


- **用户寻求最新的视频转 3D 人体网格重建仓库**：一位成员在遇到旧模型的版本兼容性问题后，正在寻找支持推理的最新 **视频转 3D 人体网格重建 (video-to-3D human mesh reconstruction)** 仓库。
- **OWLv2 的图像引导检测模式出现问题**：一位成员在关于 **OWLv2 图像引导检测模式 (image-guided-detection mode)** 的教程 notebook 上提出了一个问题，在经过几天的故障排除后仍难以复现预期结果，详见 [GitHub issue](https://github.com/NielsRogge/Transformers-Tutorials/issues/487)。



**提到的链接**：<a href="https://github.com/NielsRogge/Transformers-Tutorials/issues/487">OWLv2 的图像引导检测模式问题 · Issue #487 · NielsRogge/Transformers-Tutorials</a>：我尝试了无数次，试图从 https://github.com/NielsRogge/Transformers-Tutorials/blob/master/OWLv2/Zero_and_one_shot_object_detection_with_OWLv2... 的教程 notebook 中复现结果。

  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1357218120460271687)** (1 messages): 

> `MLX model, Smolagent, AgentGenerationError` 


- **MLX 模型在与 Smolagent 运行时出现问题**：一位成员报告了使用 **MLX model** 与 **Smolagent** 运行时的故障，遇到了 **AgentGenerationError**。
   - 根据附图，错误提示为 *cannot access local variable 'found_stop_sequence' where it is not associated with a value*（无法访问局部变量 'found_stop_sequence'，因为它尚未关联任何值）。
- **MLX 模型需要修复**：回溯信息（traceback）表明 **MLX** 实现中存在与停止序列（stop sequences）相关的变量作用域问题。
   - 需要进行修复，以便在模型生成期间访问 `found_stop_sequence` 变量之前对其进行正确的初始化或处理。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1357082209676300412)** (17 messages🔥): 

> `Course Certification, Smart RAG agent, Gradio Version, Project goals` 


- **这门课程会提供证书吗？**：一位成员询问该课程是否会授予 [认证证书](https://discord.com/channels/879548962464493619/1356866777682022440/1356866777682022440)。
   - 另一位成员提到 **deadline** 尚未明确，虽然进度已推迟了至少一周，但这并不是一个硬性截止日期。
- **为电子书构建 Smart RAG Agent**：一位成员正在为他们所有的电子书、技术知识产权、客户文档和电子邮件构建一个 **Smart RAG agent**，以测试所有框架并寻找最佳方案。
   - 项目目标包括 *实现多 Agent 架构 (multi-agent architecture)*、*启用异步处理* 以及 *通过强制引用来源来防止幻觉*。
- **Space 复制困境：Gradio 版本烦恼**：一位成员在复制 **Space** 时遇到了旧版 **Gradio version** 的问题，需要通过更新 **requirements.txt** 文件中的 **Gradio** 来解决。
   - 建议的修复方案包括在 **requirements.txt** 文件中更新特定版本的包，例如 *pydantic==2.10.6* 和 *gradio==5.23.1*。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1357078732157751367)** (103 条消息🔥🔥): 

> `调试 MCP、MCP 文件系统服务器、MCP 文档、MCP 客户端实现、FastMCP 与 Low Level` 


- **MCP 调试技术涌现**：成员们讨论了 MCP 的调试技术，透露如果[在服务器初始化期间配置了日志记录](https://example.com/initialization)，`sendLoggingMessage` 就可以工作。
   - 一位成员提到，由于 stdout 的问题，他们一直在使用 `console.error`，而其他人则认为 inspector 功能不足，从而引发了是否正在开发更好的 inspector 的疑问。
- **开源 MCP 助手服务器亮相**：一位成员分享了一个[开源 MCP EV 助手服务器](https://github.com/Abiorh001/mcp_ev_assistant_server/blob/main/ev_assitant_server.py)，强调了其在管理 **EV 充电站**、**行程规划**和**资源管理**方面的能力。
   - 该服务器旨在为 EV 相关服务提供一套全面的工具和 API。
- **MCP 客户端实现工具通知**：一位成员重点介绍了一个 [MCP 客户端实现](https://github.com/Abiorh001/mcp_omni_connect)，它支持所有**通知**，包括订阅和取消订阅资源，这对于 `notifications/tools/list_changed` 非常有用。
   - 该客户端提供与 **OpenAI 模型**的无缝集成，并支持跨多个服务器的动态工具和资源管理。
- **深入探讨 FastMCP 的局限性**：讨论显示 **FastMCP** 可能不支持某些功能（如 `subscribe_resource`），导致一些人考虑使用 **low-level server** 以获得更多控制权。
   - 成员们交换了在 low-level server 中处理资源订阅和更新的代码片段及实现细节。
- **MCP 与 SSE 的身份验证难题**：成员们争论了从 MCP 客户端向 SSE 服务器传递 API key 的最佳方式，但有人指出 [SSE 传输不会在客户端内处理 env](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/authorization/)。
   - 另一种选择是使用 **streaming HTTP** 和 **OAuth** 以获得更安全的认证，但代价是偶尔需要登录。目前的 MCP 授权机制是可选的（OPTIONAL）。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/authorization/">Authorization</a>:           ℹ️                  协议修订：2025-03-26      1. 简介    1.1 目的和范围    Model Context Protocol 在传输层提供授权功能，使得...</li><li><a href="https://github.com/Abiorh001/mcp_omni_connect">GitHub - Abiorh001/mcp_omni_connect: MCPOmni Connect 是一个通用的命令行界面 (CLI) 客户端，旨在通过 stdio 传输连接到各种 Model Context Protocol (MCP) 服务器。它提供与 OpenAI 模型的无缝集成，并支持跨多个服务器的动态工具和资源管理。</a>: MCPOmni Connect 是一个通用的命令行界面 (CLI) 客户端，旨在通过 stdio 传输连接到各种 Model Context Protocol (MCP) 服务器。它提供与 O... 的无缝集成。</li><li><a href="https://github.com/Abiorh001/mcp_ev_assistant_server/blob/main/ev_assitant_server.py">mcp_ev_assistant_server/ev_assitant_server.py at main · Abiorh001/mcp_ev_assistant_server</a>: 一个强大的服务器实现，用于管理电动汽车 (EV) 充电站、行程规划和资源管理。该服务器为 EV 相关... 提供了一套全面的工具和 API。
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1357161875212734515)** (10 条消息🔥): 

> `Enact Protocol, Shopify MCP, Mobile MCP Server, Semantic Tool Calling, External Registry Idea` 


- ****Enact Protocol** 被提议作为 MCP 的 HTTP**：一位成员介绍了 [Enact Protocol](https://github.com/EnactProtocol/specification)，将其作为定义 MCP 工具的一种方式，并将其与 HTTP 协议进行了类比。
   - 另一位成员将其描述为 *一种在 MCP server 内部进行语义化工具调用（semantic tool calling）的酷炫方式*。
- ****Shopify-MCP** 推出订单和客户更新支持**：[Shopify-MCP](https://github.com/GeLi2001/shopify-mcp) server 现在支持订单和客户更新，增强了其对 Anthropic 的 Claude 和 Cursor IDE 等 MCP 客户端的实用性。
   - 它实现了与 Shopify API 的集成，提供了一种通过 MCP 管理 Shopify 商店运营的方法。
- ****Mobile-Use MCP Server** 发布，用于移动端自动化**：[Mobile-Use MCP Server](https://github.com/runablehq/mobile-mcp) 已发布，旨在提供移动端自动化能力，并附带相关的 [mobile-use library](https://github.com/runablehq/mobile-use)。
   - 用户可以通过 `npx mobile-mcp install` 快速上手，并直接在 Claude Desktop App 中使用。
- ****Enact-MCP** Server 实现受到关注**：[Enact-MCP server](https://github.com/EnactProtocol/enact-mcp) 的实现作为 Enact Protocol 的参考被分享。
   - 一位成员注意到缺少 license 文件，并称赞了 *外部注册表想法（external registry idea）*。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/EnactProtocol/enact-mcp">GitHub - EnactProtocol/enact-mcp: MCP Server for enact protocol</a>: 适用于 enact protocol 的 MCP Server。可以通过在 GitHub 上创建账号来为 EnactProtocol/enact-mcp 的开发做出贡献。</li><li><a href="https://github.com/GeLi2001/shopify-mcp">GitHub - GeLi2001/shopify-mcp: MCP server for Shopify api, usable on mcp clients such as Anthropic&#39;s Claude and Cursor IDE</a>: 适用于 Shopify API 的 MCP server，可用于 Anthropic 的 Claude 和 Cursor IDE 等 MCP 客户端 - GeLi2001/shopify-mcp</li><li><a href="https://github.com/EnactProtocol/specification">GitHub - EnactProtocol/specification: protocol spec</a>: 协议规范。可以通过在 GitHub 上创建账号来为 EnactProtocol/specification 的开发做出贡献。</li><li><a href="https://github.com/runablehq/mobile-mcp">GitHub - runablehq/mobile-mcp: A Model Context Protocol (MCP) server that provides mobile automation capabilities.</a>: 一个提供移动端自动化能力的 Model Context Protocol (MCP) server。- runablehq/mobile-mcp</li><li><a href="https://github.com/runablehq/mobile-use">GitHub - runablehq/mobile-use: Use AI to control your mobile</a>: 使用 AI 控制你的移动设备。可以通过在 GitHub 上创建账号来为 runablehq/mobile-use 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1357067155535040532)** (2 条消息): 

> `NotebookLM UX 研究，Discover Sources 功能，Google AI 摘要` 


- **征集 **NotebookLM** 用户参与 UX 研究**：NotebookLM UX 研究员正在寻求参与者进行 **60 分钟的 1:1 远程访谈**，以获取对新 NotebookLM 想法的反馈。
   - 参与者将获得 **$100 礼品卡** 作为感谢，并且必须预先通过 Google Drive 分享一组笔记本来源，并[通过此表单申请](https://forms.gle/P2t8q36NqbPNSVk8A)。
- ****NotebookLM** 访谈：分享您的想法，获取奖励**：征集参与者进行 **60 分钟的访谈** 以提供反馈，参与者必须具备向 Google Drive 上传文件的能力。
   - 合格的参与者将通过 Tremendous 的电子邮件收到 **$100 礼品代码**，访谈安排在 4 月 10 日星期四，包括 **10 分钟的前期准备要求**。
- **通过 **NotebookLM** 发现新来源**：NotebookLM 推出了一项名为 *Discover Sources* 的新功能，允许用户查找相关的网页内容，并一键添加到笔记本中。[点击此处了解更多](https://blog.google/technology/google-labs/notebooklm-discover-sources/)。
   - 该功能使用 **Google AI** 生成摘要，并允许你通过 *I'm Feeling Curious* 按钮探索随机话题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://blog.google/technology/google-labs/notebooklm-discover-sources/">NotebookLM 新功能：发现来自网页的来源</a>: NotebookLM 推出了 Discover Sources，允许你将网页来源添加到笔记本中。</li><li><a href="https://forms.gle/P2t8q36NqbPNSVk8A">登记您的兴趣：NotebookLM 反馈</a>: 您好，我们正在通过 60 分钟的远程访谈征求对 NotebookLM 的反馈。您的反馈将帮助 Google 团队改进 NotebookLM。如需申请参加，请填写此表单。如果您...
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1357246737919643699)** (4 条消息): 

> `源文件可转移性，Podcast 深度探讨，幻灯片演示` 


- **用户感叹源文件缺乏可转移性**：用户请求[源文件可以在文件夹之间转移](https://notebooklm.google)，认为目前只读且位置锁定的特性限制了使用。
   - 用户澄清这些来源是用户输入的内容，并表达了对该应用的热爱。
- **关于 Podcast Deep Dive 合法性的辩论**：一位用户询问是否可以将 [NotebookLM 的 Deep Dive 会话上传到 Spotify](https://spotify.com) 作为播客。
   - 另一位用户推测 AI 生成的音频概览可能不受 Google 版权保护，但强调使用受版权保护的源材料可能会侵犯原始权利；因此，他们建议在对外发布时保持谨慎。
- **NoteBookLM 生成幻灯片演示**：一位用户向其他使用过 [NoteBookLM 创建演示文稿](https://notebooklm.google) 的人寻求建议。
   - 该用户正在寻求在 NoteBookLM 中创建幻灯片的技巧和窍门。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1357069596146008318)** (98 条消息🔥🔥): 

> `NotebookLM 2.5 Pro, Gemini Integration with NotebookLM, Safari Access Issues, Source Transferability, Discover Sources Feature` 


- **Gemini 专家掌舵 Google Gemini**：据 [The Verge](https://www.theverge.com/news/642000/google-sissie-hsaio-stepping-down-notebooklm) 报道，Josh Woodward 将接替 Sissie Hsaio 担任 Gemini 团队负责人，为 *Gemini 应用的下一次进化* 做准备。
- **部分 NLM 用户遇到 Safari 访问故障**：一些用户报告在 **Safari** (iPhone/Mac) 上访问 **NotebookLM** 出现问题，而其他用户确认通过在主屏幕添加快捷方式，在 iPhone SE (第二代) 上可以正常运行。
   - 如果主要语言修复方法都不起作用，在 URL 末尾添加 `?hl=en`（例如：`https://notebooklm.google.com/?hl=en`）应该可以解决。
- **Discover Sources 功能亮相**：**Discover Sources** 功能正在推出，它将研究能力扩展到用户已知信息之外，并提及有趣的各种相关主题，但尚未对所有人开放。
   - 有用户建议应该像 Perplexity 一样包含学术在线资源。
- **思维导图操作困扰**：用户注意到从思维导图节点跳转到聊天区域再返回时会关闭所有节点，需要重新导航，团队已意识到此问题。
- **语言本地化失效令用户忧心**：用户报告在 UI 设置中使用切换开关 **更改语言** 无效，且 **Google** 账号的主要语言必须设置为 **English**。
   - 在 URL 末尾附加 `?hl=en` 应该也能解决。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.theverge.com/news/642000/google-sissie-hsaio-stepping-down-notebooklm">Google’s NotebookLM leader is taking over as head of the Gemini app</a>: Google 正在调整其 AI 团队。</li><li><a href="https://notebooklm.google.com/?hl=en`,">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1357075150021070981)** (70 条消息🔥🔥): 

> `Ace Computer Autopilot Launch, YourBench Open Source Benchmarking Tool, Model Context Protocol Memory Implementation, RabbitOS Intern, Llama 4 Image Generation` 


- **Ace 在自动驾驶领域表现出色**：[General Agents Co](https://x.com/sherjilozair/status/1907478704223297576) 推出了 **Ace**，这是一款实时的计算机自动驾驶系统（Autopilot），能以超人的速度使用鼠标和键盘执行任务。
   - 与聊天机器人不同，Ace 旨在直接在计算机上执行任务。
- **YourBench 带来的基准测试盛宴**：[YourBench](https://x.com/sumukx/status/1907495423356403764) 推出了 **YourBench**，这是一个开源工具，用于根据任何文档进行自定义基准测试（Benchmarking）和合成数据生成。
   - YourBench 旨在通过提供自定义评估集和排行榜来改进模型评估。
- **Llama 4 进军图像生成领域**：**Llama 4** 正在消息功能中推出图像生成和编辑功能。
   - 用户注意到编辑速度非常快，称 *1 秒编辑 vs GPT-4o 的 5 分钟*。
- **Scale AI 估值飙升至 250 亿美元**：**Scale AI** 预计今年营收将达到 **20 亿美元**，这促使一项要约收购将公司估值定为 **250 亿美元**。
   - 去年营收为 **8.7 亿美元**。
- **A16Z 组装 AI 工作站**：A16Z 从零开始构建了一台 **8x RTX 4090 GPU AI 工作站**，兼容支持 **PCIe 5.0** 的新款 **RTX 5090**，用于在本地训练、部署和运行 AI 模型。
   - 他们发布了一份关于如何构建自己的工作站的[完整指南](https://x. Musclebot/status/1907899937838301311)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/sumukx/status/1907495423356403764">来自 Sumuk (@sumukx) 的推文</a>：我们今天发布了 🤗 yourbench，这是一个开源工具，可以根据你的任何文档进行自定义基准测试（benchmarking）和合成数据生成。这是在改进模型评估（model evaluation）方面迈出的一大步...</li><li><a href="https://x.com/sumukx/status/1907495423356403764]">来自 Sumuk (@sumukx) 的推文</a>：我们今天发布了 🤗 yourbench，这是一个开源工具，可以根据你的任何文档进行自定义基准测试（benchmarking）和合成数据生成。这是在改进模型评估（model evaluation）方面迈出的一大步...</li><li><a href="https://fxtwitter.com/sumukx/status/1907495423356403764">来自 Sumuk (@sumukx) 的推文</a>：我们今天发布了 🤗 yourbench，这是一个开源工具，可以根据你的任何文档进行自定义基准测试（benchmarking）和合成数据生成。这是在改进模型评估（model evaluation）方面迈出的一大步...</li><li><a href="https://discordapp.com/channels/822583790773862470/1337560058288017528">Discord - 充满乐趣与游戏的群聊</a>：Discord 是玩游戏、与朋友放松，甚至是建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://x.com/OpenRouterAI/status/1907867881930633666]">来自 OpenRouter (@OpenRouterAI) 的推文</a>：一个神秘模型（stealth model）加入了对话... 🥷</li><li><a href="https://x.com/TheXeophon/status/1907880330985390215">来自 Xeophon (@TheXeophon) 的推文</a>：这是我氛围检查（vibe check）中的新神秘模型。它现在是最好的非思考型模型（至少它没有思考 Token...）。输出非常简短，它喜欢用 "Certainly!" 和列表形式。非常有趣...</li><li><a href="https://x.com/hingeloss/status/1907470138321858712?s=46">来自 chris (@hingeloss) 的推文</a>：基于 Llama 4 的图像生成和编辑开始推出，看起来非常棒——而且非常快，编辑只需 1 秒，而 GPT-4o 需要 5 分钟。Meta 大显身手了吗？？</li><li><a href="https://x.com/Mascobot/status/1907899937838301311]">来自 Marco Mascorro (@Mascobot) 的推文</a>：🚨 新动态：我们 @a16z 从零开始构建了一个 8x RTX 4090 GPU AI 工作站——兼容配备 PCIe 5.0 的新 RTX 5090，用于在本地训练、部署和运行 AI 模型——这样你就不用亲自动手了。这里...</li><li><a href="https://fxtwitter.com/Mascobot/status/1907899937838301311">来自 Marco Mascorro (@Mascobot) 的推文</a>：🚨 新动态：我们 @a16z 从零开始构建了一个 8x RTX 4090 GPU AI 工作站——兼容配备 PCIe 5.0 的新 RTX 5090，用于在本地训练、部署和运行 AI 模型——这样你就不用亲自动手了。这里...</li><li><a href="https://x.com/OpenRouterAI/status/1907867881930633666">来自 OpenRouter (@OpenRouterAI) 的推文</a>：一个神秘模型（stealth model）加入了对话... 🥷</li><li><a href="https://x.com/kateclarktweets/status/1907551168143774004?s=46">来自 Kate Clark (@KateClarkTweets) 的推文</a>：Scale AI 去年创造了 8.7 亿美元的收入，预计今年将达到 20 亿美元。此外，Coatue、Founders Fund 和 Greenoaks 正在参与一项要约收购，预计该公司估值将达到 250 亿美元。独家报道...</li><li><a href="https://x.com/kateclarktweets/status/1907551168143774004?s=46]">来自 Kate Clark (@KateClarkTweets) 的推文</a>：Scale AI 去年创造了 8.7 亿美元的收入，预计今年将达到 20 亿美元。此外，Coatue、Founders Fund 和 Greenoaks 正在参与一项要约收购，预计该公司估值将达到 250 亿美元。独家报道...</li><li><a href="https://fxtwitter.com/clefourrier/status/1907496576274088070">来自 Clémentine Fourrier 🍊 (@clefourrier) 的推文</a>：无论什么主题，在不到 5 分钟内就能知道哪个模型最适合你的用例！文档 -> 自定义评估集 -> 排行榜。向 @sumukx 和 @ailozovskaya 致敬！引用 Sumuk (@sumukx)...</li><li><a href="https://fxtwitter.com/kateclarktweets/status/1907551168143774004">来自 Kate Clark (@KateClarkTweets) 的推文</a>：Scale AI 去年创造了 8.7 亿美元的收入，预计今年将达到 20 亿美元。此外，Coatue、Founders Fund 和 Greenoaks 正在参与一项要约收购，预计该公司估值将达到 250 亿美元。独家报道...</li><li><a href="https://fxtwitter.com/OpenRouterAI/status/1907867881930633666">来自 OpenRouter (@OpenRouterAI) 的推文</a>：一个神秘模型（stealth model）加入了对话... 🥷</li><li><a href="https://x.com/Mascobot/status/1907899937838301311">来自 Marco Mascorro (@Mascobot) 的推文</a>：🚨 新动态：我们 @a16z 从零开始构建了一个 8x RTX 4090 GPU AI 工作站——兼容配备 PCIe 5.0 的新 RTX 5090，用于在本地训练、部署和运行 AI 模型——这样你就不用亲自动手了。这里...</li><li><a href="https://venturebeat.com/programming-development/devin-2-0-is-here-cognition-slashes-price-of-ai-software-engineer-to-20-per-month-from-500/">Devin 2.0 来了：Cognition 将 AI 软件工程师的价格从每月 500 美元削减至 20 美元</a>：Devin 吸引了寻求将自主编码 Agent 整合到其软件开发流程中的企业客户。</li><li><a href="https://fxtwitter.com/TheXeophon/status/1">来自 Xeophon (@TheXeophon) 的推文</a>：...</li>

907880330985390215">Tweet from Xeophon (@TheXeophon)</a>: 这是我 vibe check 中新的 stealth model。它现在是最好的非思考型模型（至少它没有 thinking tokens...）。输出非常短，它喜欢用 Certainly! 和列表。非常..."</li><li><a href="https://www.dropbox.com/scl/fo/dabegjgxb1ymtlnqopzro/AGKKb-jXT_4oODKO">Tweet from Dropbox</a>: 未找到描述</li><li><a href="https://www.semafor.com/article/04/02/2025/google-gemini-shakes-up-ai-leadership-sissie-hsiao-steps-down-replaced-by-josh-woodward">Google Gemini 正在调整其 AI 领导层</a>：在 ChatGPT 发布后领导 Gemini 聊天机器人项目的 Sissie Hsiao 将卸任。她将由负责 Google Labs 的 Josh Woodward 接替。</li><li><a href="https://arstechnica.com/gadgets/2025/04/google-shakes-up-gemini-leadership-google-labs-head-taking-the-reins/">Google 调整 Gemini 领导层，Google Labs 负责人接掌大权</a>：凭借全新的领导层，Google 旨在开发基于 Gemini 的新产品。</li><li><a href="https://x.com/hingeloss/status/1907470138321858712?s=46]">Tweet from chris (@hingeloss)</a>：基于 Llama 4 的图像生成和编辑开始推出，看起来非常好——而且非常快，1 秒编辑，而 gpt-4o 需要 5 分钟。Meta 大显身手了吗？？</li><li><a href="https://fxtwitter.com/hingeloss/status/1907470138321858712">Tweet from chris (@hingeloss)</a>：基于 Llama 4 的图像生成和编辑开始推出，看起来非常好——而且非常快，1 秒编辑，而 gpt-4o 需要 5 分钟。Meta 大显身手了吗？？</li><li><a href="https://x.com/sherjilozair/status/1907478704223297576">Tweet from Sherjil Ozair (@sherjilozair)</a>：今天我推出了我的新公司 @GeneralAgentsCo 和我们的第一款产品。介绍 Ace：首个实时 Computer Autopilot。Ace 不是聊天机器人。Ace 为你执行任务。在你的电脑上...</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/memory">servers/src/memory at main · modelcontextprotocol/servers</a>：Model Context Protocol 服务端。通过在 GitHub 上创建账号为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://www.dropbox.com/scl/fo/dabegjgxb1ymtlnqopzro/AGKKb-jXT_4oODKOjCxJr9A?rlkey=ze30fqgc00trhwk1gztt21zf0&e=1&st=gqzm7gq1&dl=0">未找到标题</a>：未找到描述</li><li><a href="https://anthropic.swoogo.com/codewithclaude">Code with Claude 申请</a>：未找到描述</li><li><a href="https://ai-2027.com/">AI 2027</a>：一个有研究支持的 AI 场景预测。
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1357183581658677259)** (4 条消息): 

> `June Ramp Up, Model Context Protocol (MCP), AI Engineer World's Fair 2025, MCP vs OpenAPI` 


- **Latent Space 为 6 月份做准备**：Latent Space 正在开始为 6 月份做准备，鼓励用户在 Discord 入职流程中选择加入新的 <@&1335734932936458281> 角色以获取更新，并在 <#1344427813905891378> 中分享计划。
   - 宣布了一个新的播客，**Creators of Model Context Protocol (MCP)**，嘉宾包括 @dsp_ 和 @jspahrsummers，主题包括 *MCP 的起源故事*、*MCP vs OpenAPI*、*使用 MCP 构建 Agent* 以及 *开源治理*。
- **AI Engineer World's Fair 2025 宣布设立 MCP 专场**：[2025 AI Engineer World's Fair](https://ti.to/software-3/ai-engineer-worlds-fair-2025) 将设立专门的 **MCP 专场**，该活动将于 **6 月 3 日至 5 日在旧金山**举行，届时 MCP 核心团队以及主要的贡献者和构建者将齐聚一堂。
   - 鼓励参会者[申请演讲](https://sessionize.com/ai-engineer-worlds-fair-2025)或[赞助](mailto:sponsors@ai.engiener)。
- **MCP 赢得了 Agent 标准之战？**：根据 [Why MCP Won](https://www.latent.space/p/why-mcp-won)，**OpenAI** 和 **Google** 已经宣布支持 **MCP**。
   - 这一公告实际上证实了之前的预测，即 MCP 是 Agent 标准之战的预定赢家，并且现在在 GitHub stars 数量上已经超过了 [OpenAPI](https://github.com/OAI/OpenAPI-Specification)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/latentspacepod/status/1907843005429817481">Tweet from Latent.Space (@latentspacepod)</a>：🆕 Model Context Protocol 的创作者，嘉宾包括 @dsp_ 和 @jspahrsummers！https://latent.space/p/mcp 我们询问了所有你们关心的热点问题：- MCP 的起源故事 - MCP vs OpenAPI - 使用 MCP 构建 Agent...</li><li><a href="https://www.latent.space/p/mcp">Model Context Protocol 的创作者</a>：MCP 的共同作者谈论该协议的起源、挑战和未来。
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1357142468432236625)** (52 messages🔥): 

> `UX/UI Competition, AI UI Layout Generation, GPT-4o Behavior, GPT-5 Unified Model, DeepSeek Hype` 


- **卓越的 UX/UI 脱颖而出**：成员们讨论认为，获胜的初创公司通常拥有更好的 **UX/UI**，并指出目前的产品缺乏一种“必胜秘诀”。
   - 一位用户强调开发一种能够定义需求和页面布局的 UI，建议使用 **Agent swarm** 并行生成 Web 组件，并在一段 [屏幕录制](https://cdn.discordapp.com/attachments/986699377257119794/1357190780258746429/Screen_Recording_2025-04-03_at_1.39.26_pm.mov?ex=67eff6a9&is=67eea529&hm=9a8e202a73469a0749a23b81496240fd68a93a295583b0ce34cf52ff80c0c03e&) 中进行了展示。
- **自动化线框图的愿景**：一位成员的目标是跳过线框图/设计步骤，并链接了一个专门用于包裹查询的 Web 应用 [Dribbble 设计](https://dribbble.com/shots/25708347-Delivery-Web-App-Design)。
   - 另一位成员希望有一个布局生成器，能够设计灰度线框图，然后使用 **Agent swarm** 对其进行细化并填充 Web 组件。
- **GPT-4o 表现古怪**：用户注意到 **GPT-4o** 表现出异常行为，例如设定人设并在回答中添加括号注释，如 [截图](https://cdn.discordapp.com/attachments/986699377257119794/1357335757676871711/image.png?ex=67efd4ee&is=67ee836e&hm=4deb85a208466f212d88e7b77771776834fe28524ac15dc9c5dbcb1be3301ff3&) 所示。
   - 关于这种行为的来源出现了各种推测，理论从 **SFT** 中使用的 *EQ 数据集* 到涌现属性不等；用户还反映 **GPT-4o** 变得越来越慢。
- **Google 的 Gemini 2.5 声称进行持续预训练**：一位成员注意到 **Gemini 2.5** 声称是 *持续预训练的，没有明确的知识截止日期*。
   - 该成员思考这是否意味着增量模型发布（2 -> 2.5 -> 3），或者这只是一个幻觉。
- **动态 NLP 系统：下一个前沿**：一位用户认为下一代架构将是动态的，同时拥有短期和长期记忆。
   - 他们认为 **NLP** 应该将语言视为一种结构化的、不断演变的信号，具有流动的意义，而不是依赖于僵化的 **tokenized** 格式。



**提到的链接**：<a href="https://dribbble.com/shots/25708347-Delivery-Web-App-Design">Delivery Web App Design</a>：未找到描述

  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1357144717208457386)** (4 messages): 

> `LLMs struggle with math, LLMs overestimating themselves` 


- **LLM 在美国数学奥林匹克竞赛中惨败**：一位成员分享了一篇 [论文](https://arxiv.org/abs/2503.21934v1)，评估了最先进的 **LLM** 在 **2025 年美国数学奥林匹克 (USAMO)** 中的表现，其中 **O3-MINI** 和 **Claude 3.7** 等模型在 **六道基于证明的数学题** 上的得分低于 **5%**。
   - 每道题满分 **7 分**，总分最高 **42 分**，而这些模型是在所有能想象到的数学数据上训练的，包括 **IMO 题目**、**USAMO 存档**、**教科书**和**论文**。
- **LLM 在数学测试中评分虚高**：相同的模型（包括 **O3-MINI** 和 **Claude 3.7**）在为自己的作业评分时高估了分数，与人类评分员相比，评分虚高了多达 **20 倍**。
   - 鉴于这些模型在接触了大量数学数据后进行了广泛训练，讨论随之展开了关于这些发现的潜在影响。



**提到的链接**：<a href="https://www.reddit.com/r/LocalLLaMA/comments/1joqnp0/top_reasoning_llms_failed_horribly_on_usa_math/">Reddit - 互联网的心脏</a>：未找到描述

  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1357079886925140038)** (2 messages): 

> `Gemini App, Dream 7B` 


- **Google 发布 Gemini App**：Google 发布了 [Gemini App](https://vxtwitter.com/GeminiApp/status/1906131622736679332)，展示了其在 AI 领域的最新进展。
- **Dream 7B 扩散模型发布**：HKU-NLP 和华为诺亚方舟实验室发布了 **Dream 7B**，这是一个强大的开源扩散大语言模型，其性能优于现有的扩散语言模型，并达到或超过了同等规模的顶级自回归 (**AR**) 语言模型。
   - 根据他们的 [博客文章](https://hkunlp.github.io/blog/2025/dream/)，**Dream 7B** 展示了 *强大的规划能力和推理灵活性，这自然得益于扩散建模。*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://vxtwitter.com/GeminiApp/status/1906131622736679332">来自未定义用户的推文</a>：未找到描述</li><li><a href="https://hkunlp.github.io/blog/2025/dream/">Dream 7B | HKU NLP Group </a>：未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1357221283338649712)** (3 messages): 

> `OpenAI /v1/chat/completions API, conversation history, /v1/responses API, stateful vs stateless APIs` 


- **OpenAI `/v1/chat/completions` API 成本详解**：一位成员指出，使用 OpenAI 的 `/v1/chat/completions` API 时，必须在每次新提示中发送完整的对话历史，正如 [OpenAI 文档](https://platform.openai.com/docs/guides/conversation-state?api-mode=chat)中所述。
   - 他们还提到，即使输入 token 的 KV cache 未被驱逐，你仍然需要为这些输入 token 付费。
- **即将推出的有状态 API 替代方案：`/v1/responses`**：该成员指出，根据 [Responses vs Chat Completions 文档](https://platform.openai.com/docs/guides/responses-vs-chat-completions)，较新的 `/v1/responses` API 将是有状态的（stateful），允许通过 ID 引用过去的消息。
   - 新 API 将与 `/v1/chat/completions` API 形成对比，后者是无状态的（stateless），需要手动重新发送整个聊天历史。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1357262355519770645)** (8 messages🔥): 

> `cudaMemcpyAsync Overlap, cuBLAS matmul low occupancy, Registers in CUDA` 


- **关于重叠 `cudaMemcpyAsync` 拷贝的问题**：一位成员询问，在不同的 CPU 线程中使用带有 pinned memory 和独立 stream 的 `cudaMemcpyAsync` 时，是否可以实现主机到设备（host to device）拷贝的重叠。
   - 该成员注意到设备有 **5 个拷贝引擎（copy engines）**，但不确定这是否能实现拷贝重叠。
- **由于延迟隐藏，`cuBLAS` 占用率（Occupancy）保持在较低水平**：一位成员报告称，在 `nsys` 中对超大矩阵进行 `cuBLAS` matmul 时，占用率仅为 **20%** 左右，并询问为何不能更高。
   - 另一位成员澄清说，*高占用率是为了隐藏延迟*，而 `cuBLAS` 代码的编写方式使得少量的 warps 就足以使 GPU 的算术单元达到饱和，内存访问延迟已在软件层面被隐藏。
- **CUDA 寄存器与线程详情**：一位成员询问，如果一条指令计算一个 **64x256** 的 tile，是否每个线程需要 **128 个寄存器**？
   - 另一位成员提到在一个视频中看到，该操作每个线程需要 **256 个寄存器**。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1357321471956877484)** (1 messages): 

> `LLM Profiling, PyTorch Profilers, Perfetto Crashing, Trace Processor` 


- **LLM Profiling 建议**：一位成员征求使用 **PyTorch profilers** 对来自 transformers 库的 **LLM (32B params)** 进行 profiling 的技巧。
   - 该成员报告称 **Chrome traces** 文件高达 **2.5GB**，尝试打开时 **Perfetto** 频繁崩溃。
- **Trace Processor 解决 Trace 难题**：同一位成员找到了 **Perfetto** 崩溃问题的解决方案。
   - 解决方法是按照 [Perfetto 大文件 trace 文档](https://perfetto.dev/docs/visualization/large-traces) 的说明，在本地配合 **Perfetto** 使用 `trace_processor`。


  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1357243973080711269)** (2 条消息): 

> `关于 TunableOp 的 AMD 演讲，CuBLAS 中的 NVIDIA 预调优，以及用于 MoE 模型的基于 NVSHMEM 的内核` 


- **AMD 的 TunableOp：PyTorch 的自动调优策略**：AMD 在 [PyTorch](https://pytorch.org/docs/stable/cuda.tunable.html) 中引入了 **TunableOp**，这是一个原型功能，允许用户为 GEMMs 等操作选择最快的实现，可能会使用不同的库或技术。
   - 该功能与调优阶段分开启用，旨在优化各种硬件配置下的性能。
- **NVIDIA 烘焙最佳性能：预调优 CuBLAS 的主导地位**：一位成员指出，NVIDIA 已经对所有内容进行了预调优并将其内置到 **CuBLAS** 中，这可能使其比 AMD 更具可配置性的方法更具优势。
   - 这种预调优虽然对消费级 GPU 的优化程度可能较低，但仍提供了一个坚实的基准。
- **NVSHMEM 内核助力 Mixture-of-Experts**：**PPLXDevs** 发布了[针对 **Mixture-of-Experts (MoE)** 模型的自定义 NVSHMEM 内核](https://x.com/pplxdevs/status/1907547685579796933?s=46)，承诺通信速度比标准 all-to-all 操作快达 **10 倍**。
   - 他们的这种方法在性能与不同硬件配置的适应性之间取得了平衡，使其成为 MoE 模型开发的潜在价值工具。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/pplxdevs/status/1907547685579796933?s=46">来自 Perplexity Developers (@PPLXDevs) 的推文</a>：我们为 Mixture-of-Experts (MoE) 模型构建了自定义的基于 NVSHMEM 的内核，其通信速度比标准 all-to-all 操作快达 10 倍。我们的方法在性能与...之间取得了平衡。</li><li><a href="https://pytorch.org/docs/stable/cuda.tunable.html">TunableOp &mdash; PyTorch 2.6 文档</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1357175801811964027)** (3 条消息): 

> `激活检查点 (Activation Checkpointing)，CUDA 编译，CUDA 中的 C 与 C++` 


- **推理过程中的激活检查点**：一位成员询问了推理过程中的 **激活检查点 (activation checkpointing)**，并分享了一篇[论文链接](https://arxiv.org/pdf/2501.01792)作为参考。
   - 他提到之前见过该引用和 **CUDA 编译** 的图片，但仍在尝试理解这一概念。
- **CUDA 编译器神奇地解释 C 和 C++**：一位成员对 **CUDA 编译器** 能够推断 `.cu` 文件中编写的代码是 **C 还是 C++** 并进行相应编译表示惊讶。
   - 他指出，在 `.cu` 文件中开始使用 **C 或 C++** 编写代码会引导编译器*将代码推断为 C 代码并像往常一样开始编译 CUDA*。
- **CUDA 编译中的 C/C++ 兼容性**：一位成员解释说，**C++** 在语法上接近于 **C** 的超集，而 `nvcc` 编译器弥补了额外的差距。
   - 他质疑代码是真正作为 **C 代码** 编译（使用 **C 链接和符号名称**），还是该 **C 代码** 仅仅恰好也是有效的 **C++ 代码**。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1357285556932837497)** (3 条消息): 

> `FP8 Training, Optimizer Configuration, Model Size Impact, torch.compile Usage, GEMM size requirements` 


- **FP8 训练实现故障排除**：一位成员询问了如何在单 GPU 上使用 [pytorch/ao](https://github.com/pytorch/ao/tree/main/torchao/float8) 的方案实现 **FP8 训练**，以及是否需要任何特定的优化器配置。
   - 他们报告称，与 BF16 相比，速度提升有限，并推测模型大小是否是一个影响因素。
- **GEMM 尺寸决定 Float8 的加速效果**：一位成员指出了 [pytorch/ao](https://github.com/pytorch/ao/tree/main/torchao/float8#performance) 中的性能细节，指出 **GEMM 的 M, K, N 维度**必须足够大，**FP8** 才能提供明显的加速。
   - 他们建议，虽然提供的图表来自微基准测试（microbenchmark），但它对所需的形状给出了合理的估计，并询问用户是否使用了 **torch.compile**。
- **TorchAO 提供自定义优化器**：一位成员建议查看 [pytorch/ao 中的优化器](https://github.com/pytorch/ao/tree/main/torchao/optim)，以寻找与优化器配置相关的潜在解决方案。
   - 未提供更多细节。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/ao/tree/main/torchao/float8#performance">ao/torchao/float8 at main · pytorch/ao</a>：PyTorch 原生量化和稀疏化，用于训练和推理 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/optim">ao/torchao/optim at main · pytorch/ao</a>：PyTorch 原生量化和稀疏化，用于训练和推理 - pytorch/ao
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1357377368712282415)** (3 条消息): 

> `Code Correctness Issues, Assembly Differences` 


- **代码正确性困惑**：一位成员*不确定为什么*他们的代码存在正确性问题，并指出*没有任何值被正确写回*，如这张[截图](https://cdn.discordapp.com/attachments/1233704710389764236/1357383569181511872/image.png?ex=67f00175&is=67eeaff5&hm=47eb7052bf909c85157cec137bae54ce7b7031bc076ea77b31d987e942bdf6b9&)所示。
   - 他们发现，当*展开为下面的版本时，它可以正常工作*，目前正在寻求深入见解。
- **请求汇编审计**：一位成员回复说代码*看起来是正确的*，并建议检查汇编（assembly）层面的差异。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1357124390600446013)** (2 条消息): 

> `Blackwell Architecture, ThunderKittens Kernels, CTA pairs on Blackwell` 


- **ThunderKittens 发布 Blackwell 内核！**：HazyResearch 团队为 **NVIDIA Blackwell 架构**发布了新的 **BF16** 和 **FP8 ThunderKittens GEMM 内核**，声称其速度达到或接近 **cuBLAS**。
   - 根据他们的[博客文章](https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell)，这些内核利用了**第五代 Tensor Cores**、**Tensor Memory** 和 **CTA 对**等新特性，并将它们集成到 TK 现有的基于 Tile 的抽象中。
- **Blackwell SM 上的 CTA 对放置**：讨论围绕 **NVIDIA Blackwell GPU** 上的 **CTA (Cooperative Thread Array) 对**是调度在同一个 **SM (Streaming Multiprocessor)** 上还是跨两个 SM 进行。
   - 根据[一张来自 Nvidia GTC 2025 演讲的附图](https://cdn.discordapp.com/attachments/1300872762163728550/1357124390348783806/p.png?ex=67efb8d4&is=67ee6754&hm=fe9264e7207290f878ec5eded945a3afe71f2c92e5fbcfc77be6f913fe858b55&)，分析表明 CTA 对是调度在*集群（cluster）中的两个 SM 之间*的。



**提到的链接**：<a href="https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell">ThunderKittens Now on Blackwells!</a>：未找到描述

  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1357107705025925270)** (7 条消息): 

> `Datasets, Curricula, RGBench, Knight Swap, Puzzle2` 


- **Reasoning Gym 数据集获得课程设置（Curriculum）修复**：一名成员提交了一个 PR ([#407](https://github.com/open-thought/reasoning-gym/pull/407))，旨在修复 [reasoning-gym](https://github.com/open-thought/reasoning-gym) 项目中所有**数据集**的 **curricula**，使其更加合理，同时更新了测试并补全了缺失的 curricula。
   - 该 PR 涉及对所有数据集进行两次审查，以为 curricula 设置更合理的值，并实现了 **Knight Swap** 和 **Puzzle2** 等缺失的 curricula。
- **Reasoning Gym 添加简单、中等和困难接口**：一名成员询问了关于 **easy, medium, hard** 难度的接口（类似于 **RGBench**），以便用户可以手动设置 **reasoning-gym** 的难度。
   - 难度级别可以提取到单独的 YAML 文件中，并复用于其他任务。
- **Reasoning Gym 中等难度设置公开**：一名成员分享了一个链接，展示了 [reasoning-gym](https://github.com/open-thought/reasoning-gym/blob/5b4aa313819a9a6aecd6034b8c6394b6e4251438/eval/yaml/medium/claude-3.5-sonnet.yaml) 中每项任务被视为**中等（medium）**难度的设置。
   - 该链接包含了他们认为每项任务属于中等水平的精选级别。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/open-thought/reasoning-gym/blob/5b4aa313819a9a6aecd6034b8c6394b6e4251438/eval/yaml/medium/claude-3.5-sonnet.yaml">reasoning-gym/eval/yaml/medium/claude-3.5-sonnet.yaml at 5b4aa313819a9a6aecd6034b8c6394b6e4251438 · open-thought/reasoning-gym</a>: 程序化推理数据集。通过在 GitHub 上创建账号为 open-thought/reasoning-gym 的开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/407">fix(curriculum): Make boundaries in curriculum more sensible by zafstojano · Pull Request #407 · open-thought/reasoning-gym</a>: 概述：我对所有数据集进行了两次审查，以便为 curricula 设置一些更合理的值。此外，我还实现了一些缺失的 curricula (Knight Swap, Puzzl...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1357358423980769391)** (4 条消息): 

> `Grayscale Leaderboard Submissions, Modal Runners Success` 


- **Grayscale 排行榜迎来提交热潮**：多个成功的排行榜提交已发送至 `grayscale` 排行榜，这些提交是在各种 GPU 上使用 **Modal runners** 完成的。
   - 提交内容包括 ID **3433, 3434, 3436 和 3437**，并在 **L4, T4, A100 和 H100** 等 GPU 上进行了测试。
- **Modal Runners 在多种 GPU 上证明了其成功**：向 `grayscale` 排行榜的提交已在不同的 GPU 配置上通过 **Modal runners** 成功执行。
   - 使用的 GPU 包括 **L4, T4, A100 和 H100**，表明了广泛的兼容性。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1357106432096342148)** (28 messages🔥): 

> `Quantity struct, Dimensions ** power, IntLiteral vodoo XD, normlisation, Python wrappers for Mojo` 


- **在 Mojo 中构建带有维度的物理量**：一位成员分享了使用带有 `Dimensions` 的 `Quantity` 结构体来定义 **物理量别名**（如 `Velocity`、`Acceleration` 和 `Newton`）的代码片段。
   - 另一位成员提到了他们在 [GitHub 上的 Kelvin 库](https://github.com/bgreni/Kelvin/blob/main/kelvin/quantity.mojo#L55-L125)，并强调了为了让 `Dimensions ** power` 正常工作所进行的复杂工作。
- **`IntLiteral` 再次发威！**：一位成员提到必须使用“诅咒般”的 `IntLiteral` 技巧来绕过定义 `Quantity` 时的动态值问题。
   - 另一位成员称赞了使用 `IntLiteral` 将任意信息编码进类型系统的做法，而另一位成员则开玩笑说该用户的 *`IntLiteral` 巫术 XD 简直太可怕了*。
- **Max 的 Duration 结构体**：一位成员引用了对 modular/max 的一个 Pull Request，具体是一个受 C++ stdlib 中 **std::chrono::duration 启发的 Duration 结构体**提案，可在 [GitHub](https://github.com/modular/max/pull/4022#issuecomment-2694197567) 上查看。
   - 他表示自己已经接近实现 GitHub Issue 中提到的那个“理想化”的代码片段了。
- **对 Mojo 的 Python 互操作性的呼吁**：一位成员询问了 **Mojo 的 Python 封装**进度，以及从 CPython 调用 Mojo 的能力。
   - 另一位用户回复说这将是一个 🔥 级别的功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/bgreni/Kelvin/blob/main/kelvin/quantity.mojo#L55-L125">Kelvin/kelvin/quantity.mojo at main · bgreni/Kelvin</a>：通过在 GitHub 上创建账号来为 bgreni/Kelvin 的开发做出贡献。</li><li><a href="https://github.com/modular/max/pull/4022#issuecomment-2694197567">[stdlib][proposal] 由 bgreni 提出的 Duration 模块提案 · Pull Request #4022 · modular/max</a>：一个受 C++ stdlib 中 std::chrono::duration 启发的 Duration 结构体提案。
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1357257296471920700)** (4 messages): 

> `Checkpoint Conversions, HF Checkpoint Format, tune_to_hf function` 


- **Torchtune Checkpoint 获得 HuggingFace 支持**：成员们讨论了如何将 **torchtune checkpoint** 转换为 **HF checkpoint 格式**。
   - 一位成员建议查看 *huggingface checkpointer*，并推荐了 **tune_to_hf 函数**。
- **HuggingFace Checkpointer 前来救场**：**HuggingFace checkpointer** 可用于转换 torchtune checkpoint。
   - 具体来说，可以利用 **tune_to_hf 函数** 进行此类转换。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1357121407493869732)** (19 messages🔥): 

> `vLLM 与 Unsloth 的显存共享，GRPO 上游贡献，Torchtune 在特定序列长度下挂起，打包数据集 (Packed Datasets)` 


- **Unsloth 与 vLLM 共享 VRAM**：一名成员提到，在 [Unsloth](https://github.com/unslothai/unsloth) 中，他们成功地让 **vLLM** 和训练过程使用相同的 VRAM，但尚未完全理解其实现原理。
   - 他们还认为，在验证配置中使用 `train` 动词作为掩码标志可能会引起混淆。
- **Ariel 提供 GRPO 上游贡献**：一名成员提议提交来自其“内部” **GRPO** 的上游更改，包括进程内 (in-process) **vLLM** 集成、训练中评估 (evals) 以及更灵活的 **RL** 数据处理。
   - 另一名成员回应称，他们在异步版本中已有 vLLM 集成，并且有一个验证数据集的 PR 已接近合并状态，但尚未考虑多数据集场景或报告多个 losses 的情况。
- **Torchtune 遭遇超时 Bug**：一名成员报告了一个“惊人的 Bug”，即如果部分（而非全部）microbatches 的 **seq length 为 7/14/21/28/35/42/49**，**Torchtune** 会挂起并因超时而崩溃，并为此创建了 [一个 issue](https://github.com/pytorch/torchtune/issues/2554)。
   - 该成员庆幸 *torchtune dataloader 仍存在 seed: null 导致不随机的 Bug*，否则他们可能无法捕捉到这个错误。
- **打包数据集 (Packed Datasets) 解决问题**：针对挂起问题，一名成员建议使用打包数据集，因为它速度更快，且永远不会出现 **seqlen=49** 的情况。
   - 另一名成员表示，*如果这是解决方案，则应更新标准 recipe，因为我们在基础的 torchtune 示例中也观察到了该问题*。



**提到的链接**：<a href="https://github.com/pytorch/torchtune/issues/2554">Chunked output causes timeout crash on certain seq len · Issue #2554 · pytorch/torchtune</a>：摘要：如果 dataloader 的 batch 之一长度为 49 个 token，torchtune 会因超时而崩溃。详细解释：transformer.py 中的 chunked_output 将输出拆分为 8 个 tensor 的列表。如果输出长度为 49...

  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1357108354714959974)** (4 messages): 

> `Dream 7B, 扩散语言模型 (Diffusion Language Models), 华为诺亚方舟实验室` 


- **Dream 7B 作为强大的扩散模型发布**：香港大学与华为诺亚方舟实验室联合发布了 **Dream 7B**，这是一种新型开源扩散大语言模型，详情见[此博客文章](https://hkunlp.github.io/blog/2025/dream/)。
   - 据报道，它*大幅超越了现有的扩散语言模型*，并在通用能力、数学和编程能力上达到或超过了同等规模的顶尖自回归 (Autoregressive) 语言模型。
- **Dream 7B 展示了规划能力**：根据[发布说明](https://hkunlp.github.io/blog/2025/dream/)，**Dream 7B** 展示了规划能力和推理灵活性，这天然受益于扩散建模 (diffusion modeling)。



**提到的链接**：<a href="https://hkunlp.github.io/blog/2025/dream/">Dream 7B | HKU NLP Group </a>：未找到描述

  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1357081337047290066)** (14 条消息🔥): 

> `Diagram Creation Tools, DeTikZify, Gradient Accumulation, GitHub MCP event` 


- **图表工具辩论激烈！**：成员们讨论了各种图表创建工具，建议直接手动操作时使用 **Inkscape**，否则使用 **draw.io**。
   - 其他人建议使用 **pure TikZ**，一位用户开玩笑说其他替代方案都是“骗人的”。
- **DeTikZify 工具合成图形程序**：一位成员分享了一个名为 [DeTikZify](https://github.com/potamides/DeTikZify) 的新工具链接，用于通过 **TikZ** 合成科学插图和草图的图形程序。
   - 该用户尚未尝试，但请求群组成员如果发现它有用请提供反馈。
- **Gradient Accumulation：不只是表面看起来那样**：一位成员指出 Gradient Accumulation 除了显而易见的用途外还有其他用途。
   - 另一位成员讽刺地评论道，*所有的 pipeline parallelism 本质上都是 gradient accumulation*。
- **GitHub 在旧金山共同主办 MCP 活动**：GitHub 正在旧金山共同主办一场 **MCP Demo Night** 活动，重点关注 **AI**、事件响应和平台工程的交集；更多详情见 [lu.ma/9wi116nk](https://lu.ma/9wi116nk)。
   - 该活动承诺提供闪电演示、**Future of AI Panel**、炉边谈话，以及为开发者、平台工程师和 AI 领导者提供的社交机会。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lu.ma/9wi116nk">MCP’s and The Future of Developer Tools &amp; AI-Driven Reliability · Luma</a>: 加入我们，共度一个处于开发者体验和基础设施创新前沿的夜晚。MCP Demo Night 是尖端 MCP 工具与未来交汇的地方……</li><li><a href="https://github.com/potamides/DeTikZify">GitHub - potamides/DeTikZify: Synthesizing Graphics Programs for Scientific Figures and Sketches with TikZ</a>: 使用 TikZ 合成科学插图和草图的图形程序 - potamides/DeTikZify
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1357434825967010064)** (1 条消息): 

> `OpenThoughts-1M, OpenThinker2-32B, OpenThinker2-7B, R1-Distilled-32B, Qwen 2.5 32B` 


- **OpenThinker2：SOTA 开源数据推理模型亮相**：Ludwig Schmidt 及其团队发布了 **OpenThoughts-1M** 数据集，以及 **OpenThinker2-32B** 和 **OpenThinker2-7B** 模型，仅通过在 **Qwen 2.5 32B Instruct** 上进行 SFT 就超越了 **R1-Distilled-32B**；详情见其 [博客文章](https://www.openthoughts.ai/blog/thinkagain)。
- **OpenThoughts2-1M：数据集详情披露**：**OpenThoughts2-1M** 数据集基于 **OpenThoughts-114k** 构建，整合了 [OpenR1](https://huggingface.co/open-r1) 等数据集以及 [dataset card](https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M) 中描述的其他数学/代码推理数据。
- **OpenThinker2 模型：SFT 表现优于 DeepSeekR1-32B**：根据 [Etash Guha 的推文](https://x.com/etash_guha/status/1907837107793702958)，**OpenThinker2-32B** 和 **OpenThinker2-7B** 仅通过在开源数据上进行 SFT 就优于 **DeepSeekR1-32B**，使用的是为高质量指令策划的数据集。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.openthoughts.ai/blog/thinkagain">Outperforming DeepSeekR1-32B with OpenThinker2</a>: 宣布我们开源推理模型和数据集的下一次迭代。</li><li><a href="https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M">open-thoughts/OpenThoughts2-1M · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/etash_guha/status/1907837107793702958">来自 Etash Guha (@etash_guha) 的推文</a>: 事实证明，仅通过开源数据的 SFT 而无需 RL 即可超越 DeepSeekR1-32B：发布 OpenThinker2-32B 和 OpenThinker2-7B。我们还发布了数据 OpenThoughts2-1M，由 sele...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1357408635008843796)** (9 messages🔥): 

> `Combining linear probes, Steering vector composition, Contrastive sample selection` 


- **探针组合查询引发讨论**：一位成员询问了关于为策划的正/负样本集组合线性探针 (linear probes) 或转向向量 (steering vectors) 的问题，并质疑联合训练是否能减少串扰或干扰。
   - 另一位成员建议，根据向量空间的公理，转向向量 (steering vectors) 应该表现为向量，并建议搜索 **线性表示假设 (linear representations hypothesis)**。
- **转向向量被证明不可靠**：一位成员分享了论文 [Steering Vectors: Reliability and Generalisation](https://arxiv.org/abs/2407.12404)，表明 **转向向量 (steering vectors) 具有局限性**，无论是在分布内还是分布外。
   - 论文指出，*可转向性在不同输入之间差异很大*，并且对于 prompt 的变化非常脆弱。
- **探索转向向量的动态组合**：一位成员分享了他们在 [steering vector composition](https://aclanthology.org/2024.blackboxnlp-1.34/) 方面的工作，展示了使用 **动态激活组合 (Dynamic Activation Composition)** 在处理语言和形式/安全性等不相关属性对时的成功。
   - 他们的信息论方法通过调节转向强度，在保持高调节性的同时，最大限度地减少对生成流畅度的影响。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.12404">Analyzing the Generalization and Reliability of Steering Vectors</a>: Steering vectors (SVs) have been proposed as an effective approach to adjust language model behaviour at inference time by intervening on intermediate model activations. They have shown promise in ter...</li><li><a href="https://aclanthology.org/2024.blackboxnlp-1.34/">Multi-property Steering of Large Language Models with Dynamic Activation Composition</a>: Daniel Scalena, Gabriele Sarti, Malvina Nissim. Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP. 2024.
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1357081308676751381)** (15 messages🔥): 

> `Google Mentorship, Tinygrad YoloV8 on Android, LeetGPU support for tinygrad` 


- **Google 导师计划：值得费心吗？**：一位成员对导师计划表示怀疑，认为由于入职挑战和文书工作，其产出 *几乎永远不值得投入到学生身上的时间/精力*。
   - 与此相反，另一位成员认为公司实际上获得了 *为你全职工作 3 个月的聪明人*，所以产出是相当不错的。
- **Tinygrad YoloV8 在 Android 上遇到小问题**：一位用户报告在 `pip install tinygrad` 后，在 Samsung Galaxy Tab S9 上运行 **tinygrad** 版的 **YoloV8** 时出现 `OSError: dlopen failed: library "libgcc_s.so.1" not found`。
   - George Hotz 建议这可能是一个两行代码的修复，但应将 Android 加入 CI 以防止再次发生，而另一位成员建议执行 `pkg install libgcc`。
- **LeetGPU 将支持 Tinygrad**：有人询问关于 [leetgpu.com](https://leetgpu.com) 的信息。
   - 他们读到该网站很快将支持 **tinygrad**。



**Link mentioned**: <a href="https://leetgpu.com">LeetGPU</a>: no description found

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1357154502028820561)** (7 messages): 

> `bilinear interpolation, saving latest model` 


- **寻求双线性插值支持**：一位成员询问 **tinygrad** 中对 **双线性插值 (bilinear interpolation)** 的支持。
   - 另一位成员建议在文档中搜索 **bilinear**，但第一位成员报告说它 *"不起作用"*。
- **关于模型覆盖的疑问**：一位成员询问在每个 epoch 后使用 `state_dict = get_state_dict(net); safe_save(state_dict, "model.safetensors")` 来保存最新模型是否安全。
   - 另一位成员澄清说，除非为每次保存提供不同的名称，否则它将被覆盖。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1357395796068143134)** (1 messages): 

> `CodeAct Agents, ReAct Generalization` 


- **构建你自己的 CodeAct Agent！**：从零开始的 [CodeAct](https://t.co/0GPTPo87ma) 是 **ReAct** 的一种泛化，Agent 不再是在顺序循环中通过思维链 (chain-of-thought) 对工具进行推理，而是动态编写代码并使用这些函数来解决任务。
- **CodeAct 是 ReAct 的泛化**：CodeAct 是 ReAct 的一种泛化，Agent 不再进行思维链推理，而是动态编写代码来解决任务。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1357153572667785226)** (20 条消息🔥): 

> `Rankify 框架, 增强 Gemini API 集成, Cursor API 知识, otel trace_id, 在 postgres 中重新索引文件` 


- **Rankify 框架简化 RAG 任务**：一个新的开源框架 [Rankify](https://github.com/DataScienceUIBK/Rankify) 旨在简化**检索 (retrieval)、重排序 (reranking) 和 RAG** (检索增强生成) 等任务。
- **待解决的 Gemini 支持差距**：一位成员正在为 DeepMind 的 *增强 Gemini API 集成* 起草 GSoC 提案，并希望让 **LlamaIndex** 成为其中的重要组成部分。
   - 他们询问在 llama-index-llms-google-genai 或 vertex 中，**Gemini** 支持方面是否存在任何需要解决的显著差距（如多模态或函数调用），以及任何**与 Gemini 相关的特性或优化**。
- **用于 Cursor API 知识的 MCP 工具**：一位成员询问在编码时如何将最新的 API 和文档知识提供给 **Cursor**，以及是否存在 llms.txt 等文件。
   - 另一位成员回应称代码库非常庞大，*llm.txt* 几乎没用，建议提供一个可以在文档上进行检索的 **MCP tool** 或类似工具。
- **Trace ID 挑战**：一位成员面临 **otel trace_id** 在父工作流调用子工作流后无法检索的问题。
   - 另一位成员建议将 **trace_id** 放在其他可以获取的地方（工作流上下文或其他全局变量）。
- **Postgres 中的向量索引更新**：一位成员想要更新存储在 Postgres 向量表中的文件的**向量索引**。
   - 另一位成员建议*删除原始文档中的行，然后重新索引*。



**提到的链接**：<a href="https://github.com/DataScienceUIBK/Rankify">GitHub - DataScienceUIBK/Rankify: 🔥 Rankify: A Comprehensive Python Toolkit for Retrieval, Re-Ranking, and Retrieval-Augmented Generation 🔥</a>：Rankify：一个用于检索、重排序和检索增强生成的综合 Python 工具包。我们的工具包集成了 40 个预检索的基准数据集，支持 7 种以上的检索技术、24 种以上最先进的重排序模型以及多种 RAG 方法。

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1357087741594308820)** (11 条消息🔥): 

> `ChatGPT 4o 万智牌卡牌, Runway Gen 4, 阿里巴巴 Wan 2.2` 


- **ChatGPT 4o 生成万智牌卡牌**：一位成员使用 **ChatGPT 4o 的图像生成器** 为 AI 领域的流行人物和 **NousResearch 团队** 制作了**万智牌 (Magic the Gathering)** 卡牌。
   - 他们声称结果得到了“高品味测试者”的认可且非常棒，尽管有人评论说 *sama 很差劲*。
- **Runway Gen 4 助力 AI 电影制作**：一位成员指出，随着 **Runway 发布 Gen 4**，**AI Prompt 电影制作** 已经取得了长足进步，并[链接了一个视频](https://www.youtube.com/watch?v=Rcwfj18d8n8)，涵盖了 **OpenAI、Google 和 AGI** 领域的最新动态。
   - 视频强调 *AI 视频正变得不可思议...*，并指出开源替代方案 **阿里巴巴 Wan 2.2** 的发布也不远了。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Teknium1/status/1907492873991499998">Teknium (e/λ) (@Teknium1) 的推文</a>：我上周使用 ChatGPT 4o 的图像生成器为 AI 领域的一群流行人物和 @NousResearch 团队制作了高品味测试者认可的万智牌卡牌，并且...</li><li><a href="https://www.youtube.com/watch?v=Rcwfj18d8n8">AI 视频正变得不可思议... (GEN 4)</a>：最新的 AI 新闻。了解 LLM、生成式 AI 并为 AGI 的推出做好准备。Wes Roth 报道了 OpenAI、Google、Anth... 领域的最新动态。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1357449307648688260)** (3 messages): 

> `LLMs for extraction, Genstruct-7B, Ada-Instruct` 


- **LLMs 提取数据集**：一位成员询问关于使用 **LLMs for extraction** 从非结构化 PDF 中创建数据集的问题。
   - 另一位成员建议，提示（prompting）更大的模型可能会更好，但也推荐了 **Genstruct-7B** 作为一个很好的起点，用于从任何原始文本语料库（raw-text corpus）创建合成指令微调数据集。
- **Genstruct-7B 生成指令**：[Genstruct 7B](https://huggingface.co/NousResearch/Genstruct-7B) 是一个**指令生成模型**，旨在根据给定的原始文本语料库创建有效的指令。
   - 这种方法受到了 [Ada-Instruct](https://arxiv.org/abs/2310.04484) 的启发，后者训练了一个自定义的指令生成模型；同时也参考了一个 [GitHub 仓库](https://github.com/edmundman/OllamaGenstruct)，以便在 ollama 上快速处理大量 PDF。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/edmundman/OllamaGenstruct">GitHub - edmundman/OllamaGenstruct</a>：为 edmundman/OllamaGenstruct 的开发做出贡献，需创建一个 GitHub 账号。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1357128517384667147)** (1 messages): 

> `OpenAPI, SaaS, PaaS, IaaS, LLMs` 


- **LLMs 的 OpenAPI 访问权限发布**：一位成员宣布发布了他们针对 **LLMs** 的 **SaaS/PaaS/IaaS** 的 **v1 OpenAPI 访问权限**，旨在消除 **MCP 混乱**（[HN 讨论链接](https://news.ycombinator.com/item?id=43562442)）。
- **通过 OpenAPI 减少 MCP 混乱**：新的 **OpenAPI 访问权限** 旨在解决在将 **LLMs** 与各种云服务集成时出现的 **MCP (Multi-Cloud Platform) 混乱** 问题。



**提及的链接**：<a href="https://news.ycombinator.com/item?id=43562442>">未找到标题</a>：未找到描述

  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1357108864939593829)** (8 messages🔥): 

> `Cohere Status Page, Python logging vs print statements, RAG strategy for documents` 


- **Cohere 遭遇性能下降！**：一些用户报告遇到了 **http timeout errors**，并确认 [Cohere Status Page](https://status.cohere.com/) 显示 **Command-a-03-2025/command-r-plus-08-2024** 模型处于 *Degraded Performance - Increased Latency*（性能下降 - 延迟增加）状态。
   - 该问题正在监控中，且已持续 **4 小时**。
- **Python Logging：队友意见不一！**：一位成员正在开发他们的第一个用于 PDF 处理的 Python 包，在是使用 **logs** 还是 **print 语句** 的问题上与一名资深队友产生了分歧。
   - 该成员更倾向于使用 logs，因为其具有**不同的级别、文件保存、可搜索性和问题报告功能**；而队友则更倾向于使用 **print 语句** 以避免给用户增加负担；最后建议了一个折中方案：**默认禁用 logger 实例**。
- **RAG 策略：长文档是否需要分块（Chunk）？**：一位成员询问关于将一个 **18000 token 的文档** 用于 **RAG** 时是否需要对其进行切割。
   - 专家建议对文档进行切割，但这取决于最终目标和需求，同时也指出 **Command-a 的 256k 上下文窗口** 以及 **command-r 和 r-plus 的 128k 上下文窗口** 应该能够轻松处理该长度。



**提及的链接**：<a href="https://status.cohere.com/">Cohere Status Page Status</a>：Cohere 状态页面的最新服务状态

  

---


### **Cohere ▷ #[「💡」projects](https://discord.com/channels/954421988141711382/1218409701339828245/1357411997645279464)** (1 messages): 

> `AI Safety Testing Platform, Bias and Harmful Outputs, AI Model Deployment Challenges` 


- **Brainstorm AI 安全测试平台即将发布！**：一个名为 **Brainstorm** 的 AI 安全测试平台将在几周内发布其 MVP，旨在确保 AI 更好地改变世界，更多信息请访问 [Brainstorm 落地页](https://brainstormai.framer.website/)。
- **征集 AI 安全与性能测试方法**：**Brainstorm** 的创建者正在寻求关于目前用于测试 AI 安全和性能问题的方法见解，特别是围绕**偏见（bias）**、**提示词注入（prompt injections）**或**有害输出**方面。
- **讨论 AI 模型部署的最大痛点**：重点在于了解确保 AI 模型准备好进行实际部署时的最大痛点，欢迎感兴趣的人通过私信或评论分享经验。



**提及的链接**：<a href="https://brainstormai.framer.website/">Brainstorm - AI Safety Made Easy</a>：AI 安全测试的简单解决方案。

  

---

### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1357168810917888071)** (2 条消息): 

> `KAIST student, Bias/fairness and interpretability in LLMs/VLMs, Research collaboration opportunities` 


- **KAIST 学生寻求合作**：一位来自 **KAIST**（韩国）的硕士生介绍了自己，其研究方向为 **LLMs/VLMs** 中的 **偏见/公平性** 和 **可解释性**。
   - 他们正在积极寻求在这些特定领域的研究合作机会。
- **合作机会**：该学生正在寻找研究合作。
   - 他们的背景包括在 KAIST 的经验。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1357079177387446484)** (7 条消息): 

> `Nomic Embed Text V2, Vulnerability Disclosure, GPT4All-J model, Chocolatine-2-14B model, Chat Reorganization` 


- **耐心等待 Nomic Embed Text V2**：一位成员正在等待 **Nomic Embed Text V2** 在 **GPT4All** 中可用，并对开发者的忙碌表示理解。
   - 他们表示理解集成可能需要一些时间。
- **通过联系销售渠道进行漏洞披露**：一位成员询问了负责任地披露 **GPT4All** 漏洞的适当联系方式。
   - 另一位成员建议使用 **Nomic AI** 网站上的 [联系支持邮箱](https://atlas.nomic.ai/contact-sales)。
- **GPT4All-J 模型搜索引发量化查询**：一位成员请求 **Q4_0 量化** 和 **GGUF 格式** 的 **GPT4All-J 模型** 下载链接，以便将其集成到自己的项目中。
   - 第二位成员回答说 **GPT4All-Falcon** 有 **GGUF** 版本，但 **GPT4All-J** 不可能实现。
- **Chocolatine-2-14B 在嵌入式书籍查询中脱颖而出**：一位成员称赞 "**Chocolatine-2-14B**" 模型类型是他们查询嵌入式书籍的首选。
   - 未提供关于该模型的进一步信息。
- **聊天记录需要按时间顺序修正**：一位成员建议聊天记录应根据修改时间而非创建时间进行重组。
   - 他们批评当前按创建日期排序的逻辑是*随意*的。



**提到的链接**：<a href="https://atlas.nomic.ai/contact-sales">Contact Nomic Sales</a>：探索、分析并利用您的非结构化数据进行构建

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1357079321616978121)** (5 条消息): 

> `LLM agent development, DSPy Framework, OpenAI Agents SDK, Prompt Engineering vs programming` 


- **Telemetry 改进 LLM Agent 开发循环**：一位成员分享了一个名为《通过配置 Telemetry 和评估使 LLM Agent 实现自我改进，从而闭环开发流程》的视频，[可在 YouTube 上观看](https://youtu.be/jgzSq5YGK_Q)。
   - 该视频讨论了利用 **telemetry** 和 **evaluations** 来增强 LLM Agent 的自我改进。
- **DSPy 解耦 Prompt Engineering**：一位成员询问了 DSPy 在将 **Prompt Engineering** 的 *tinkering layer*（修补层）与 LLMs 的 *functional behavior*（功能行为）解耦方面的作用，以及它如何与 **OpenAI Agents SDK** 协同工作。
   - 另一位成员确认了这一点，指出 DSPy 提供了*编程组件*：**Signatures 和 Modules** 来实现这种解耦。
- **探索 DSPy 的编程组件**：一位成员强调了 DSPy 的核心抽象，即 **Signatures 和 Modules**，它们促进了 Prompt Engineering 与 LLM 功能行为的解耦。
   - 这种方法旨在实现编程而非仅仅是 Prompt Engineering，这有助于将 DSPy 与 **OpenAI Agents SDK** 等其他工具集成。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1357148524063752363)** (3 条消息): 

> `Tool evaluation, Phi-4-mini-instruct, BFCL` 


- **Phi-4-mini-instruct 评估 PR 已提交**：一位成员提交了一个 [PR](https://github.com/ShishirPatil/gorilla/pull/967)，旨在为 **Phi-4-mini-instruct** 添加 **BFCL** 的工具评估。
- **工具评估需要审阅者**：一位成员请求对其 PR 提供反馈，并指出他们已在 PR 中附上了 **评估分数**。
- **审阅进行中**：一位成员表示他们将查看该 **PR**。



**提到的链接**：<a href="https://github.com/ShishirPatil/gorilla/pull/967">[BFCL] add support for microsoft/Phi-4-mini-instruct by RobotSail · Pull Request #967 · ShishirPatil/gorilla</a>：此 PR 引入了对 Microsoft 新发布的 Phi-4-mini-instruct 模型的支持。该结果最初是针对 f81063 进行评估的；然而，该模型...

  

---

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1357462550974959806)** (1 条消息): 

> `DeepSeek-V3, DeepSeek-V3-0324, Windsurf AI` 


- **DeepSeek-V3 迎来升级**: **DeepSeek-V3** 已升级为 **DeepSeek-V3-0324**，据称在评估中表现比之前略好。
   - 一位成员发布了 **Windsurf AI** Twitter 账号的 [链接](https://x.com/windsurf_ai/status/1907902846735102017)，宣布了此次升级并确认其继续免费提供。
- **Windsurf 请求书签收藏**: Windsurf 正试图提高其公告的曝光度。
   - 一位成员请求用户在 X 上收藏该公告帖子。



**提到的链接**: <a href="https://x.com/windsurf_ai/status/1907902846735102017">来自 Windsurf (@windsurf_ai) 的推文</a>: DeepSeek-V3 现已升级为 DeepSeek-V3-0324。它仍然免费！

  

---


---


---


{% else %}


> 完整的频道细分内容已针对电子邮件进行了截断。 
> 
> 如果您想查看完整的细分内容，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}