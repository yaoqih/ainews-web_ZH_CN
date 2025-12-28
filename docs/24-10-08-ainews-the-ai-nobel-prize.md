---
companies:
- openai
- anthropic
- reka-ai
- zep
date: '2024-10-09T01:33:48.218940Z'
description: '**杰弗里·辛顿 (Geoff Hinton)** 和 **约翰·霍普菲尔德 (John Hopfield)** 因在**人工神经网络**方面的研究贡献荣获**诺贝尔物理学奖**。颁奖词长达
  **14 页**，详细阐述了他们的成就。


  **Zep** 发布了其针对 AI 智能体的低延迟记忆层社区版，重点强调了利用知识图谱构建记忆。在 OpenAI 的开发者大会（DevDay）上，官方推出了实时语音
  API、视觉模型微调以及提示词缓存（Prompt Caching）等新功能，其中重复使用的 Token 可享受 **50% 的折扣**。


  **Anthropic 的 Claude 3.5 Sonnet** 被公认为目前最顶尖的模型。**Reka AI Labs** 更新了其 **Reka Flash**
  模型，增强了多模态和函数调用能力。**GOT (通用 OCR Transformer)** 在 OCR 基准测试中达到了 **98.79% 的准确率**。


  关于开源 AI 模型的讨论强调了其在促进竞争和去中心化方面的作用。软件开发领域的洞察包括单点登录 (SSO) 的重要性、严谨的测试以及 AI 辅助编程工作流。伦理和社会议题则涵盖了对税收政策的批评以及法国首位人工智能部长的任命。'
id: f285ff85-b5c0-4008-ab38-4fd516cc1561
models:
- claude-3.5-sonnet
- reka-flash
- got
original_slug: ainews-the-ai-nobel-prize
people:
- geoff-hinton
- john-hopfield
- philschmid
- alexalbert
- mervenoyann
- clementdelangue
- svpino
- bindureddy
- ylecun
- rohanpaul_ai
title: AI 诺贝尔奖 或 人工智能诺贝尔奖
topics:
- artificial-neural-networks
- nobel-prize
- knowledge-graphs
- memory-layers
- real-time-voice-api
- vision
- fine-tuning
- prompt-caching
- multimodality
- function-calling
- ocr
- open-source
- single-sign-on
- software-testing
- ai-assisted-coding
- ai-ethics
---

<!-- buttondown-editor-mode: plaintext -->**Artificial Neural Networks 是你成为物理学家所需要的一切。**

> 2024年10月7日至10月8日的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discord（**226** 个频道和 **2556** 条消息）。预计节省阅读时间（以 200wpm 计算）：**277 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论了！

我们可以讨论 [新的 Differential Transformer 论文](https://news.ycombinator.com/item?id=41776324)，或者新的 [AdderLM 论文](https://reddit.com//r/LocalLLaMA/comments/1fy9apg/addition_is_all_you_need_for_energyefficient/)，但别开玩笑了，今天的大新闻是 Geoff Hinton 和 John Hopfield 获得诺贝尔物理学奖。


![image.png](https://assets.buttondown.email/images/de8fd1dd-d9c3-4bdc-842a-44274fd84c74.png?w=960&fit=max)


这份 [14 页的引文](https://www.nobelprize.org/uploads/2024/09/advanced-physicsprize2024.pdf) 涵盖了他们的代表作，而来自 [AI 圈的梗图](https://x.com/DrJimFan/status/1843681423443800315) 以及职业物理学家的反应一直……很有趣。

https://youtu.be/dR1ncz-Lozc?feature=shared

当然，Hopfield 对 [物理学奖项并不陌生](https://pni.princeton.edu/sites/g/files/toruqf321/files/documents/John%20Hopfield%20Now%20What%203_0.pdf)。

---

**[由 Zep 赞助]**：Zep 是一个为 AI Agent 和助手设计的低延迟记忆层。他们持续更新用户交互的内部图谱，以提供快速、确定性的事实检索。他们刚刚发布了新的社区版；[去 GitHub 看看吧！](https://shortclick.link/uu8gwd)

> Swyx 评论：在 AI Engineer 大会上，将 Knowledge Graphs 用于 Memory 是 [最热门的话题之一](https://www.youtube.com/watch?v=knDDGYHnnSI) —— 其他流行框架也在推出“长期记忆”支持，但这是一个不绑定于 LangChain、Autogen 等的开源解决方案。[Readme 包含了一个非常棒的 FAQ](https://github.com/getzep/zep#why-use-zep-for-long-term-memory)，我们很乐意看到这一点。Memory 层在 2024 年似乎和 Vector 数据库在 2023 年一样火爆。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。


**AI 与语言模型**

- OpenAI 的 DevDay 介绍了实时语音 API、视觉模型微调和节省成本的 prompt caching 等新功能。[@_philschmid](https://twitter.com/_philschmid/status/1843317930471403990) 指出重复使用的 token 可享受 50% 的折扣。

- Anthropic 的 Claude 3.5 Sonnet 模型被公认为目前最强的模型。[@alexalbert__](https://twitter.com/alexalbert__/status/1843322457903841341) 在播客节目中分享了这一见解。

- Reka AI Labs 宣布了其 Reka Flash 模型的更新，包括改进的多模态能力和 function calling 支持。[@RekaAILabs](https://twitter.com/RekaAILabs/status/1843298155682820566) 详细介绍了在图像、视频和音频模态方面的增强。

- GOT (Generic OCR Transformer) 模型因其 OCR 能力受到赞誉。[@mervenoyann](https://twitter.com/mervenoyann/status/1843278355749065084) 分享称，该模型在基准测试数据集上达到了 98.79% 的准确率。

- 关于开源 AI 模型的讨论仍在继续，[@ClementDelangue](https://twitter.com/ClementDelangue/status/1843289934989500874) 认为开源创造了良性竞争，并能对抗 AI 领域的权力集中。

**软件开发与工程**

- [@svpino](https://twitter.com/svpino/status/1843261247925461304) 详细解释了单点登录 (SSO) 的工作原理，强调了其在现代身份验证系统中的重要性。

- [@svpino](https://twitter.com/svpino/status/1843340889827201101) 强调了软件开发中全面测试的重要性，并指出未测试的代码本质上是无法运行的代码。

- [@bindureddy](https://twitter.com/bindureddy/status/1843410752067252678) 建议允许应聘者在面试中使用 AI 工具是一种机智的表现，而非作弊。

- [@bindureddy](https://twitter.com/bindureddy/status/1843372716612890773) 报告了一个内部里程碑，他们的 AI 工程师现在可以查看 stack traces、解决问题，并在不同程度的人工干预下提交 pull requests。

**AI 伦理与社会影响**

- [@ylecun](https://twitter.com/ylecun/status/1843400142910820476) 批评了特朗普的税收计划，声称该计划将降低前 5% 人群的税收，同时增加其他所有人的税收。

- 法国任命全球首位 AI 部长被 [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1843357058823008479) 视为历史性举措。

- [@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1843423602265485460) 分享了关于文明脆弱性的思考，以及在技术压力面前维护标准和缓和冲突的重要性。

**AI 研究与开发**

- [@_philschmid](https://twitter.com/_philschmid/status/1843327203662045432) 分享的视觉指南解释了 Mixture of Experts (MoE) 架构，突出了其在参数使用方面的效率。

- [@OfirPress](https://twitter.com/OfirPress/status/1843294092924796989) 宣布了一个名为 SWE-bench Multimodal 的新基准测试，包含 617 个带有图像的任务，旨在现实场景中挑战 AI Agent。

- [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1843358705649070415) 分享了关于 Inverse Painting 的研究，该技术可以为任何艺术品生成绘画过程的延时摄影视频。

**AI 工具与应用**

- [@mickeyxfriedman](https://twitter.com/mickeyxfriedman/status/1843319816062738468) 宣布 FlairAI 现在支持通过结合针对品牌美学和产品训练的模型，生成品牌风格一致的视频广告。

- [@_akhaliq](https://twitter.com/_akhaliq/status/1843363506697187626) 分享了关于 openai-gradio 的信息，这是一个用于轻松创建由 OpenAI API 驱动的 Web 应用的 Python 包。

- [@jerryjliu0](https://twitter.com/jerryjliu0/status/1843410331290480981) 讨论了在幻灯片中使用 contextual retrieval 以获得更好的 chunking 策略，从而提高问答能力。

**迷因与幽默**

- [@ylecun](https://twitter.com/ylecun/status/1843398583120269606) 开玩笑说，周期性的空调故障能防止 AGI 长期失控。

- [@karpathy](https://twitter.com/karpathy/status/1843324726107832727) 幽默地将 Sydney（可能指 Bing 的聊天机器人）称为“AI 界的哈兰贝 (Harambe)”。

- [@lateinteraction](https://twitter.com/lateinteraction/status/1843364387240980541) 针对 Python 的无 GIL 模式开了一个双关语玩笑，说他们可以为此写两个 threads，但不能并行。


---

# AI Reddit 综述

## /r/LocalLlama 综述

**主题 1. 节能 AI：基于加法的算法声称可降低 95% 的能耗**

- **[A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)** ([Score: 73, Comments: 7](https://reddit.com//r/LocalLLaMA/comments/1fya0dx/a_visual_guide_to_mixture_of_experts_moe/)): **混合专家模型 (MoE)** 是一种高效的模型架构，它使用多个专门的神经网络（**experts**）和一个**门控网络 (gating network)** 将输入路由到最合适的专家。这种方法允许模型在增加参数量的同时保持计算效率，因为每个输入仅激活一部分专家。MoE 架构已成功应用于多个领域，包括 **Google 的 Switch Transformer** 和 **Microsoft 的 Turing-NLG** 等 **语言模型 (language models)**，展示了与传统稠密模型相比更优的性能和可扩展性。
- **Addition is All You Need for Energy-Efficient Language Models: Reduce energy costs by 95% using integer adders instead of floating-point multipliers.** ([Score: 318, Comments: 65](https://reddit.com//r/LocalLLaMA/comments/1fy9apg/addition_is_all_you_need_for_energyefficient/)): 研究人员提出了一种名为 **AdderLM** 的新方法，在语言模型中用**整数加法 (integer additions)** 替换**浮点乘法 (floating-point multiplications)**，有望将**能耗**降低高达 **95%**。该方法在 [arXiv 论文](https://arxiv.org/html/2410.00907) 中有详细介绍，在保持与传统模型相当的性能的同时，显著降低了 AI 系统的计算成本和功率需求。
  - **AdderLM** 的实现面临挑战，因为大公司并未在传统 Transformer 边界之外开发模型。**Jamba-1.5** 模型在长上下文方面表现出色，但缺乏广泛采用，且运行需要 **80GB+ VRAM**。
  - 用户对 **Jamba 模型** 的性能展开了辩论，一些人认为 **398B 模型** 的表现与其体量不符，而另一些人则称赞 **1.5 版本** 处理长上下文的能力。本地部署缺乏简便的量化方案仍是一个问题。
  - 论文糟糕的语法引发了担忧，但用加法替换乘法的概念引起了读者的兴趣。一些人推测，如果这种方法在 **llama.cpp** 等工具中实现，可能会带来**以 CPU 为中心的解决方案**，并可能挑战 **Nvidia 的垄断**。


**Theme 2. Zamba 2: New Mamba-based Models Outperform Larger Competitors**


- **Zamba 2 2.7B & 1.2B Instruct - Mamba 2 based & Apache 2.0 licensed - beats Gemma 2 2.6B & Mistral 7B Instruct-v0.1** ([Score: 125, Comments: 30](https://reddit.com//r/LocalLLaMA/comments/1fyc34z/zamba_2_27b_12b_instruct_mamba_2_based_apache_20/)): **Zamba 2** 是一款基于 **Mamba 2** 的模型，拥有 **2.7B** 和 **1.2B** 参数版本。如提供的图像所示，它在基准测试中超越了 **Gemma 2 2.6B** 和 **Mistral 7B Instruct-v0.1**。这些模型在 **Hugging Face** 上以 **Apache 2.0 许可**发布，可通过 [Zamba2-2.7B-instruct](https://huggingface.co/Zyphra/Zamba2-2.7B-instruct) 和 [Zamba2-1.2B-instruct](https://huggingface.co/Zyphra/Zamba2-1.2B-instruct) 获取，不过对 llama.cpp 的支持尚在进行中。


- **Where do you actually rank LLaMA 3.2 405B among the big boys?** ([Score: 56, Comments: 58](https://reddit.com//r/LocalLLaMA/comments/1fy711x/where_do_you_actually_rank_llama_32_405b_among/)): 该帖子比较了多个领先的**大语言模型**的性能，包括 **LLaMA 3.1 405B**、**Gemini 1.5 Pro**、**GPT-4**、**Claude 3.5 Sonnet**、**Grok 2**、**Mistral Large 2**、**Qwen 110B**、**Deepseek 2.5** 和 **Command R+**。作者试图了解 **LLaMA 3.1 405B** 在性能和能力方面在这些“大厂模型”中处于什么位置。
  - **Claude 3.5 Sonnet** 和 **GPT-4** 变体在推理和性能方面始终排名靠前，**Claude 3.5 Sonnet** 经常位列前三。用户对 **GPT-4o** 的评价褒贬不一，有些人认为它非常出色，而另一些人则形容它“过度调优 (overcooked)”或使用起来令人沮丧。
  - **LLaMA 3.1 405B** 通常排在前 5 名，部分用户将其排在 **Mistral Large 2** 之上。它被指出“运行难度极高”，但在长上下文任务和通用用途中表现良好。
  - **Gemini 1.5 Pro** 最近的更新显著提升了其性能，用户现在将其与顶级模型并列。它在长上下文任务中表现优异，能有效处理高达 **100k tokens**，使其在法律文档和其他大规模文本处理中特别有用。


**Theme 3. Open WebUI 0.3.31: New Features Rivaling Commercial AI Providers**

- **试试我这款支持本地模型的开源浏览器助手。** ([Score: 64, Comments: 21](https://reddit.com//r/LocalLLaMA/comments/1fy4f3e/try_my_opensource_browser_assistant_that_works/)): 该帖子介绍了一款支持 **local LLM models** 的 **open-source browser assistant**，提供预定义提示词和自定义选项。该扩展支持包括 **YouTube**、**Reddit**、**Slack**、**Gmail**、**X**、**Telegram** 和 **GitHub** 在内的多种网站，并 **100% locally** 运行，页面数据通过默认运行在 **port 8080** 的后台进程直接发送到选定的助手。该扩展适用于 **Firefox** 和 **Chrome**，并提供了 **GitHub** 仓库和浏览器扩展商店的链接。
  - 该扩展 **100% locally** 运行，无需遥测或账号。它支持各种 AI 模型的 **custom endpoints**，并能与本地运行的 **Open WebUI** 配合使用。
  - 用户对 **YouTube transcription** 功能表现出兴趣，该功能每 30 秒提取一次时间戳。开发者澄清，目前设置的 **minimum supported Firefox version** 为 129。
  - 关于与 **LM Studio** 兼容性的讨论揭示了局限性，因为该扩展只能在浏览器内运行。开发者建议在处理基于 Web 的任务时使用 **Open WebUI**，而将 **LM Studio** 用于其他用途。

- **[Open WebUI 0.3.31 新增了类似 Claude 的 ‘Artifacts’、类似 OpenAI 的实时代码迭代，以及将完整文档放入上下文（而非分块/嵌入）的选项。](https://github.com/open-webui/open-webui/releases)** ([Score: 484, Comments: 80](https://reddit.com//r/LocalLLaMA/comments/1fyaij3/open_webui_0331_adds_claudelike_artifacts/)): **Open WebUI 0.3.31** 引入了多项新功能，包括用于在可调节窗口中实时渲染 HTML、CSS 和 JS 的 **Claude-like 'Artifacts'**，用于聊天分支导航的 **Svelte Flow interface**，以及允许将整个文档加载到上下文中而无需分块的 **"full document retrieval" mode**。此次更新还在 Artifacts 中增加了支持实时更新的 **editable code blocks**，以及针对 **LLM** 响应的 **ask/explain feature**，使 **Open WebUI** 的功能更接近商业 AI 提供商。
  - **Open WebUI 0.3.31** 引入了在可调节窗口中对 HTML、CSS 和 JS 的 **live rendering**，用户认为这比 "**1000x better than chatgpt UI**"。更新还包括在 UI 中运行 **Python code** 的能力。
  - 一位用户通过使用 **L3.1 8B zero-shot** 生成一个 **landing page for a cat library** 展示了新功能。提示词 "Build me a landing page for a cat library" 生成了一个虽然基础但功能齐全的设计。
  - 用户对此次更新表示兴奋，并询问了 **version 0.4** 中的后续功能。一个 [public milestone](https://github.com/open-webui/open-webui/milestone/4) 暗示了进一步的改进，尽管有些功能比预期更早发布。


**Theme 4. AntiSlop Sampler: 减少 LLM 输出中的重复语言**

- **Prompt 编写倦怠？你如何应对？** ([Score: 79, Comments: 87](https://reddit.com//r/LocalLLaMA/comments/1fykk44/promptwriting_burnout_how_do_you_cope/)): **Prompt-writing burnout** 被描述为一个耗费精力的循环：编写、提炼和测试提示词，作者估计他们已经写了相当于 **"a thousand pages"** 的内容。发帖者在提示词上的成功率波动不定，导致频繁的修改，偶尔甚至需要完全重来。为了应对这种疲劳，他们发现休息、散步以及在 AI 建议下玩 **Helldivers** 和 **Valheim** 等电子游戏可以缓解压力，但他们仍在寻求社区的其他策略。

- **[AntiSlop Sampler 获得 OpenAI 兼容 API。可在 Open-WebUI 中试用（详情见评论）](https://v.redd.it/5lywrxcxfgtd1)** ([Score: 120, Comments: 46](https://reddit.com//r/LocalLLaMA/comments/1fyr1ch/antislop_sampler_gets_an_openaicompatible_api_try/)): **AntiSlop Sampler** 是一款用于减少 AI 生成文本中重复语言的工具，现在已拥有 **OpenAI 兼容 API**。此次更新允许用户将 AntiSlop Sampler 集成到支持 OpenAI API 的应用程序中，通过减少冗余和重复，潜在地提高 AI 生成内容的质量。新功能可以在 **Open-WebUI** 中进行测试，原帖评论中提供了更多详细信息。
  - 用户对 **AntiSlop Sampler 的实现**表现出浓厚兴趣，并讨论了其**多语言能力**以及与 **llama.cpp** 和 **ExllamaV2** 等其他后端的潜在集成。开发者提供了一个 [GitHub 链接](https://github.com/sam-paech/antislop-sampler/blob/main/calculate_over_represented_words.ipynb) 用于计算 slop 短语。
  - 项目创建者分享了在 **Open-WebUI** 中运行 AntiSlop Sampler 的**详细设置指南**，包括安装步骤和配置设置。用户可以在 [JSON 文件](https://github.com/sam-paech/antislop-sampler/blob/main/slop_phrase_prob_adjustments.json) 中调整 **slop 短语概率**，以自定义工具的行为。
  - 一些用户在测试该工具时报告了褒贬不一的结果，并对生成文本的**连贯性损失**表示担忧。开发者针对这些问题进行了回应，建议调整**强度参数 (strength parameter)**，并提供了基准模型与 AntiSlop 增强模型之间的 [基准测试对比 (benchmark comparisons)](https://eqbench.com/results/creative-writing-v2/)。


**主题 5. 优化 AI Agent：利用 DSPy 和 Argilla 改进搜索和提示词**

- **使用 DSPy 和 Argilla 为搜索 Agent 优化提示词用法** ([Score: 108, Comments: 2](https://reddit.com//r/LocalLLaMA/comments/1fy2eqy/optimizing_prompt_usage_for_search_agent_with/)): 该帖子描述了如何使用 **DSPy**、**Langchain 工具**和 **Argilla** 优化 **ArXiv Agent**，以提高其搜索和回答科学论文问题的能力。作者使用 **DSPy 的 AvatarOptimizer** 来增强 **ArXiv API** 的提示词结构化，从而实现更高效、更准确的信息提取，并使用 **Argilla 的 UI** 进行详细的响应审查以评估改进效果。优化后的 Agent 对问题的理解能力更强，从 ArXiv 提取的信息也更具相关性，示例 Notebook 可在 [GitHub](https://github.com/argilla-io/argilla-cookbook/blob/main/dspy_agent_arxiv_tools_prompt_optimization.ipynb) 获取。

- **试试我的开源浏览器助手，它支持本地模型。** ([Score: 64, Comments: 21](https://reddit.com//r/LocalLLaMA/comments/1fy4f3e/try_my_opensource_browser_assistant_that_works/)): 这款开源浏览器助手 **Taaabs** 可与**本地 LLM** 配合使用，并为包括 **YouTube**、**Reddit**、**Slack**、**Gmail** 和 **GitHub** 在内的各种网站提供预定义提示词及自定义选项。该扩展程序 **100% 本地运行**，通过后台进程将页面数据直接发送到选定的助手，默认情况下 **OpenWebUI** 运行在 **8080 端口**，并支持用于图像分析的**视觉模式 (vision mode)**。用户可以从 [GitHub 仓库](https://github.com/taaabs/taaabs) 安装 Taaabs，或通过提供的链接下载 **Firefox** 和 **Chrome** 浏览器版本。
  - 用户对 **Taaabs** 表现出极大的热情，并提出了关于**数据隐私**、**Firefox 兼容性**和 **YouTube 转录**的问题。开发者确认了 **100% 本地处理**、无需账号，且每 30 秒提供一次**提炼后的转录文本**。
  - 该扩展在 **AI 模型选择**方面具有灵活性，包括预定义的聊天机器人和自定义端点。用户可以使用 **Open WebUI** 设置本地实例，或使用 **Groq** 等外部 API 以优先考虑速度。
  - 一些用户遇到了 **LM Studio** 集成和**新标签页覆盖**功能的问题。开发者解决了这些疑虑，承诺在下次更新中移除新标签页功能，并澄清 **LM Studio** 作为一个独立应用，无法直接与浏览器扩展程序兼容。

## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Model Releases and Improvements**

- **Salesforce 的“微型巨人” xLAM-1b 模型在 function calling 方面超越 GPT 3.5**：Salesforce 发布了 xLAM-1b，这是一个拥有 10 亿参数的模型，在 [**function calling 中实现了 70% 的准确率，超越了 GPT 3.5**](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/)。尽管其体积相对较小，但被称为“function calling 巨人”。

- **具备 function calling 能力的 Phi-3 Mini (6月版)**：Rubra AI 在 6 月发布了更新的 Phi-3 Mini 模型，[**具备 function calling 能力**](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/)。它与 Mistral-7b v3 具有竞争力，并且表现优于基础版 Phi-3 Mini。

- **Microsoft/OpenAI 攻克多数据中心分布式训练**：据分析师 Dylan Patel 称，[Microsoft 和 OpenAI 已经实现了多数据中心分布式训练](https://www.reddit.com/r/singularity/comments/1fydbil/microsoftopenai_have_cracked_multidatacenter/)，这可能使更大规模的模型训练变得更加高效。

**AI Research and Techniques**

- **Inverse Painting 生成绘画过程的延时摄影视频**：一种名为 [Inverse Painting 的新技术可以生成延时摄影视频](https://www.reddit.com/r/singularity/comments/1fybddi/inverse_painting_can_generate_timelapse_videos_of/)，展示任何艺术作品的绘画过程，并能从多种绘画技巧中学习。

- **MonST3R 在运动场景下估算几何结构**：研究人员开发了 [MonST3R，一种在存在运动的场景中估算 3D 几何结构的方法](https://www.reddit.com/r/singularity/comments/1fyax9h/monst3r_a_simple_approach_for_estimating_geometry/)，这可以改进从视频中进行的 3D 重建。

- **新的 LLM 采样方法可能减少幻觉**：工程师们正在评估一种[基于熵（entropy）的新型 LLM 采样方法](https://www.reddit.com/r/singularity/comments/1fyacda/engineers_are_evaluating_a_new_sampling_method/)，该方法可以减少幻觉，并允许类似于 OpenAI O1 模型的动态推理时计算（inference-time compute）。

**AI Capabilities and Impact**

- **AI 图像正在占据 Google 搜索结果**：一则帖子显示 [AI 生成的图像越来越多地出现在 Google 图片搜索结果中](https://www.reddit.com/r/singularity/comments/1fyf93x/ai_images_taking_over_google/)，突显了 AI 内容在网络上的日益普及。

- **Max Tegmark 预测 AI 将飞速进步**：AI 研究员 Max Tegmark 表示，[未来 2 年内将出现重大的 AI 突破](https://www.reddit.com/r/singularity/comments/1fyngp8/max_tegmark_says_crazy_things_will_happen_due_to/)，这使得长期规划变得困难，并可能“令我们大受震撼”。

- **与历史相比，变革速度正在加快**：一则帖子将[当今的技术变革速度与历史时期进行了对比](https://www.reddit.com/r/singularity/comments/1fyq1sk/mind_blown/)，认为与前几个世纪相比，变革正在迅速加速。

**AI Image Generation Techniques**

- **用于生成写实照片的文件路径提示词**：用户发现[在提示词中包含 Windows 文件路径](https://www.reddit.com/r/StableDiffusion/comments/1fy2riz/cusersyour_prompt_herepicturesphotos_also_works/)（例如 "C:\Users\name\Pictures\Photos\"）可以生成看起来更真实的 AI 照片。

- **从草图生成图像、3D 和视频**：一个演示展示了在 ComfyUI 中使用 AI [从单个草图输入生成图像、3D 模型和视频](https://www.reddit.com/r/StableDiffusion/comments/1fyh67m/generate_image_3d_and_video_from_a_single_sketch/)。

- **90 年代亚洲摄影风格**：一位用户分享了[模仿 90 年代亚洲摄影风格的 AI 生成图像](https://www.reddit.com/r/StableDiffusion/comments/1fytgnf/90s_asian_look_photography/)，展示了复制特定审美时期的能力。


---

# AI Discord Recap

> 由 O1-preview 为我们提供的总结的总结摘要

**主题 1. 前沿 AI 模型发布与探索**

- **Nvidia 凭借 Llama-3.1-Nemotron-51B 加倍投入**：Nvidia 发布了 [Llama-3.1-Nemotron-51B](https://x.com/NVIDIAAIDev/status/1838263496049570053)，这是一款经过 NAS 优化的模型，在保持准确性的同时，在单张 H100 GPU 上实现了 **2倍吞吐量**。用户可以通过 [Nvidia AI](http://ai.nvidia.com) 的 API 体验该模型，或从 [Hugging Face](https://huggingface.co/nvidia/NVLM-D-72B) 下载。
- **Meta 通过 CoTracker 2.1 追踪 7 万个点**：Meta 发布了 [CoTracker 2.1](https://x.com/NielsRogge/status/1842958590396772599)，通过在单张 GPU 上联合追踪 **70,000 个点**，增强了视频运动预测能力。详细介绍这些进展的配套论文可在此处[查阅](https://huggingface.co/papers/2307.07635)。
- **Google 合并高达 64B 参数的模型**：一名 Google 实习生的研究探讨了[大规模模型合并](https://arxiv.org/abs/2410.03617)，将语言模型合并至 **64B 参数**。该研究解决了合并大型模型时关于性能和泛化的问题，在社区中引发了兴奋和质疑。

**主题 2. 诺贝尔奖争议：AI 与物理学的交汇**

- **Hinton 和 Hopfield 斩获诺贝尔奖，物理学界反应强烈**：**2024 年诺贝尔物理学奖**授予了 Geoffrey Hinton 和 John J. Hopfield，以表彰他们在人工神经网络方面的工作，这引发了争论。批评者认为，该奖项优先考虑 AI 而非传统的物理学成就，可能会削弱该奖项的威望。
- **物理学家质疑诺贝尔奖对 AI 的关注**：物理论坛的成员表达了沮丧，认为将物理学奖授予 AI 工作忽略了更值得获奖的物理学研究。一些人将其视为炒作掩盖了有影响力的科学的信号。
- **诺贝尔级别讨论 AI 伦理**：瑞典皇家科学院将重点转向包含 AI 伦理和安全，表明对 AI 影响的更广泛考量。这一举动反映了社会对 AI 与传统科学交汇的关注。

**主题 3. 微调热潮与优化障碍**

- **Unsloth Studio 旨在简化微调**：人们对 **Unsloth Studio** 的发布充满期待，预计它将简化 Windows 上的微调流程，无需像 Docker 这样复杂的设置。用户对目前的困难表示沮丧，并希望获得无缝的安装程序体验。
- **Aider 用户要求控制自动提交 (Auto-Commits)**：开发者要求 **Aider** 在提交代码更改前进行确认，而不是自动提交。成本估算的透明度和界面中更好的标签也是寻求更多控制权的用户关注的热点话题。
- **LM Studio 0.3.4 通过 MLX 提升 Mac 性能**：**LM Studio 0.3.4** 的发布引入了适用于 Apple Silicon Mac 的 [MLX 引擎](https://github.com/lmstudio-ai/mlx-engine)，带来了 **10-50%** 的速度提升。用户注意到效率有所提高，尤其是在运行较大模型时。

**主题 4. GPU 闲谈：硬件难题与启示**

- **GPU 对决：Tesla P40 对阵 RTX 4060 Ti 引发辩论**：成员们权衡了拥有 **24GB VRAM** 的 **Tesla P40** 与拥有 **16GB VRAM** 的 **RTX 4060 Ti** 的优缺点。虽然 P40 提供更多显存，但与 4060 Ti 相比，其性能较慢且推理能力有限。
- **NVIDIA 与 AMD：讨论性能差异**：用户一致认为将 **RTX 3060** 与 **RX 6600** 混合使用会导致效率低下，主张坚持使用 NVIDIA GPU 以获得更好的速度和兼容性。双 3060 可能会增加 VRAM，但不会显著提升处理速度。
- **HBM 和 SRAM 扩展性受到审视**：对 **HBM** 的成本效益出现了质疑，讨论强调它占据了像 **H100** 这样设备成本的很大一部分。还注意到 **SRAM 扩展**未能跟上逻辑扩展的步伐，指出了潜在的设计疏忽。

**主题 5. AI 工具与 API：用户的成功与考验**

- **Cohere API 以简洁性吸引开发者**：新用户称赞 **Cohere API** 易于使用，能够以极少的代码实现多工具 Agent 设置。**深色模式 (Dark Mode)** 的引入也让用户感到兴奋，提升了开发者体验。
- **OpenRouter 通过 Prompt Caching 节省成本**：**OpenRouter** 上的 **OpenAI prompt caching** 可节省高达 **50%** 的推理成本。用户可以在[活动页面](https://openrouter.ai/activity)审计他们的节省情况，该功能目前支持八款 OpenAI 模型。
- **Anthropic 的 Message Batches API 提供批量处理功能**：Anthropic 推出了 [Message Batches API](https://x.com/AnthropicAI/status/1843695536614060201)，允许在 **24 小时**内异步处理多达 **10,000 个查询**。虽然一些用户欣赏其成本效益，但也有人对响应延迟表示担忧。


---

# 第 1 部分：Discord 高层摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Nvidia 发布 Llama-3.1-Nemotron-51B**：Nvidia 推出了 [Llama-3.1-Nemotron-51B](https://x.com/NVIDIAAIDev/status/1838263496049570053)，这是一个经过 NAS 优化的模型，在保持准确性的同时，在单个 H100 GPU 上实现了 **2 倍吞吐量**。
   - 用户可以通过 [Nvidia AI](http://ai.nvidia.com) 的 API 体验该模型，或从 [Hugging Face](https://huggingface.co/nvidia/NVLM-D-72B) 下载。
- **Meta 增强视频运动预测**：Meta 发布了 [CoTracker 2.1](https://x.com/NielsRogge/status/1842958590396772599)，能够在单个 GPU 上跟踪 **7 万个点**，提升了运动预测能力。
   - 随附论文详细介绍了这些进展，可以在[这里](https://huggingface.co/papers/2307.07635)找到。
- **Hugging Face Accelerate 1.0 特性**：Hugging Face 发布了 [Accelerate 1.0](https://x.com/TheZachMueller/status/1843320011139813644)，引入了旨在优化模型训练过程的新功能。
   - 用户可以通过访问[发布博客](https://huggingface.co/blog/accelerate-v1)了解更多详情。
- **LLM 受限于训练范围**：成员们强调，像 GPT-2 和 GPT-3 这样的 LLM 受限于其训练分布，限制了它们解决陌生问题的能力。
   - 虽然它们可以辅助各种任务，但缺乏真正的理解和独立的输出过滤。
- **Tokenizer 准确性的重要性**：讨论确认了使用特定于模型的正确 Tokenizer 的必要性，因为不匹配的 Tokenizer 会导致无效的结果。
   - 由于许多模型共享 Tokenization 方法，效率得以提高，这使其成为开发者关注的关键点。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 0.3.4 提升 Mac 性能**：**LM Studio 0.3.4** 的发布引入了 [MLX engine](https://github.com/lmstudio-ai/mlx-engine)，用于改进 Apple Silicon Mac 上的端侧 LLM，支持同时执行模型和结构化 JSON 响应。
   - 用户报告称，使用 MLX 时，大型模型的速度提升了 **10-20%**，小型模型提升高达 **50%**，这使其与旧版本区别开来。
- **自动更新困扰用户**：用户对 **0.3.4** 版本无法通过自动更新获取表示沮丧，必须从网站手动下载，这导致了现有工作流中的 Bug。
   - 这种非预期的聊天记录迁移导致了褒贬不一的体验，突显了用户面临的过渡困难。
- **关于 GPU VRAM 优势的辩论**：在关于 **VRAM 选项**的持续讨论中，成员们评估了拥有 **24GB** 的 **Tesla P40** 与拥有 **16GB** 的 **RTX 4060 Ti** 的优劣，强调了 P40 的显存优势，但也指出了其性能较慢。
   - 考虑到与更通用的 4060 Ti 相比，P40 的推理应用有限，人们对此表示担忧。
- **性能差异：NVIDIA vs AMD**：小组一致认为，将 **RTX 3060** 与 **RX 6600** 串联使用会导致效率低下，主张使用专门的 NVIDIA 配置以获得最佳速度。
   - 一位成员指出，双 **3060** 可以增加 VRAM，但可能无法有效提高处理速度。
- **用户体验揭示硬件限制**：在围绕 **Stable Diffusion** 的讨论中，用户注意到不同模型在 **VRAM** 使用方面的巨大限制，并指出了对处理速度的影响。
   - 人们对在当前硬件配置上高效运行新模型的可行性表示担忧，特别是在对比**高端 GPU** 时。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Studio 发布备受期待**：用户正热切期待 **Unsloth Studio** 的发布，该工具承诺简化 Windows 上的 Fine-tuning 流程，同时跳过像 Docker 这样复杂的设置。
   - 针对 Docker 和 GPU 驱动设置的*挫败感频现*，这让人们对通过安装程序获得流畅体验寄予厚望。
- **探索用于内容审核的 LLM 微调**：有人提议针对**内容审核（content moderation）**微调 LLM，目标是一个包含 **50k** 条短文本的数据集。
   - 建议指出 **Llama Guard** 和 **Gemma Shield** 是实现有效分类的潜在工具。
- **深度解析模型合并策略**：参与者讨论了一篇关于**大规模模型合并（model merging at scale）**的新论文，强调了跨各种模型大小和配置的方法论。
   - 鉴于之前排行榜中凸显的问题，人们对合并大型模型的实用性持怀疑态度。
- **推理方法的性能疑问**：用户询问 **vLLM** 的 **inference** 在消费级硬件上是否能与 Unsloth 有效竞争。
   - 社区讨论中出现了一种需求，即需要权衡设置成本与性能收益。
- **重点推荐用于模型训练的 Colab 资源**：一位成员分享了一个旨在辅助 **ShareGPT** 和 **Llama** 训练的 **Colab notebook** 链接，并获得了积极反馈。
   - 该资源有助于缓解之前的一些挫败感，旨在为用户简化训练流程。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 提示确认 Commit**：用户希望 Aider 在编码后提示确认 Commit，而不是自动 Commit，并关注在界面中清晰标注预估成本。
   - 许多人认为禁用自动 Commit 可以增强对代码更改的控制，而成本管理仍然是一个关键话题。
- **Embeddings 驱动语义搜索**：讨论显示 **Embeddings** 在**语义搜索（semantic search）**中起着关键作用，帮助 LLM 根据向量表示检索相关文档。
   - 跨平台保持一致的 **Embeddings** 对于防止文档检索中的相关性丢失至关重要。
- **Python 3.13 掀起热潮**：**Python 3.13** 已经发布，具有[更好的 REPL](https://docs.python.org/3.13/whatsnew/3.13.html#whatsnew313-better-interactive-interpreter) 和对**移动平台**的支持，标志着更广泛的可访问性努力。
   - 该版本还引入了一个[实验性 JIT 编译器](https://docs.python.org/3.13/whatsnew/3.13.html#an-experimental-just-in-time-jit-compiler)，这可能会显著优化性能。
- **使用 NotebookLM 进行 AI 播客制作**：一位成员详细介绍了他们使用 **Google NotebookLM** 创建关于 SmartPoi 项目节目的经验，并分享了他们的[概览剧集](https://www.circusscientist.com/2024/10/07/smartpoi-ai-podcast-episode-1/)。
   - 尽管存在一些内容混淆，AI 生成的播客非常有说服力，甚至让家属相信它是真实的。
- **引入 Message Batches API**：Anthropic 引入的 [Message Batches API](https://x.com/AnthropicAI/status/1843695536614060201) 被称赞为异步处理大型查询的高性价比解决方案。
   - 虽然一些人对响应延迟表示担忧，但其他人看到了其在更高效生成训练数据方面的潜力。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI 诺贝尔奖得主引发争议**：物理学界就将**诺贝尔物理学奖**授予 **Hinton 和 Hopfield** 的 AI 工作是否合适展开了激烈辩论，引发了人们对炒作掩盖具有影响力的研究的担忧。
   - 成员们认为，重大认可应优先考虑传统的物理学成就，而*为神经网络颁奖可能会稀释该奖项的声望*。
- **Normalized Transformer 的令人兴奋的进展**：新的 **nGPT** 架构引入了归一化向量的超球体表示（hypersphere representation），声称通过增强表示学习，训练效率提升了 **20倍**。
   - 这种方法通过在每一层保持单位范数向量来优化训练动态，从而可能简化学习过程。
- **模型合并（Model Merging）性能审查**：Google 一项关于 **model merging** 的新研究探讨了大规模模型的性能影响，检查了高达 **64B 参数** 的可扩展性问题。
   - 主要发现解决了关于 held-in 性能的常见问题，提高了对超出常规边界合并模型时性能不一致性的认识。
- **生成式奖励模型（Generative Reward Models）受到关注**：研究强调了 **Generative Reward Models** 的重要性，它结合了人类和 AI 反馈以增强 LLM 训练性能。
   - *关于实现的讨论强调了 AI 系统决策中推理的必要性*，以实现有效的训练后性能。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **用于文档分类的 AI 吸引用户**：成员们讨论了 AI 有效分类文档的潜力，尽管对当前能力持怀疑态度，有时仍更倾向于手动整理。
   - 他们提出了几种可以处理大型文件集的工具，引发了关于如何高效管理海量数据集的有趣辩论。
- **云端成本 vs 本地 AI 分析**：对 AI 成本的担忧浮现，特别是对 **18,478 个文件** 进行云端分析的费用估计将达到约 **$12,000**。
   - 成员们权衡了云端解决方案的服务器开销与本地硬件相关的成本，辩论了数据分析的最佳路径。
- **AVM 和多模态 AI 能力令工程师兴奋**：围绕 AVM 的讨论突出了多模态 AI 技术的令人兴奋的融合，指出它可能显著改变用户交互。
   - 成员们表达了对可能增强 AVM 工具功能的即将推出的特性的期待。
- **Prompt 排行榜引发辩论**：Prompt 排行榜的可能性引发了关于如何客观地为 Prompt 有效性评分的幽默讨论。
   - 出现了关于在不同输出中保持 Prompt 评估一致性的可行性和方法的问题。
- **Gemini Advanced Prompt 的成功**：一位成员报告了使用为 Gemini Advanced 精心设计的 Prompt 取得的持续成功，在不同的交互中生成了高质量的响应。
   - 他们被提醒注意社区准则，强调了在讨论其他 AI 时遵守规定的必要性。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenAI Prompt Caching 发布取得成效**：上周，**OpenAI prompt caching** 正式发布，显著降低了推理成本，降幅最高可达 **50%**。它可与 8 款 OpenAI 模型无缝协作，并集成了 **Anthropic** 和 **DeepSeek** 等供应商。
   - 用户可以在 **openrouter.ai/activity** 页面审计其缓存带来的**节省情况**，具体收益细节可通过 `/generation` API 查看。
- **重复生成问题干扰流程**：用户报告在 **OpenRouter** 中每个请求会出现两次生成，引发了关于潜在设置问题和超时管理的讨论。建议增加超时时间以获得更好的性能。
   - 虽然部分用户将其归因于个人配置，但集体反馈表明需要进一步排查故障。
- **Anthropic API 审核之战**：一位用户在 **Claude 3.5 Sonnet** 的审核机制上遇到挑战，发现使用 `:beta` 端点可能会缓解某些强制审核问题。标准端点执行强制审核，而 beta 选项允许自我审核。
   - 这引发了关于在不同条件下使用 **Anthropic** API 时最佳实践的重要疑问。
- **高效供应商选择的见解**：成员们交流了如何有效地将请求路由到特定供应商（特别是 **Anthropic**）的策略，以减轻速率限制错误。默认的负载均衡选项和手动供应商固定（provider pinning）被强调为可行的替代方案。
   - 这引发了关于进一步优化请求处理以防止中断的咨询。
- **429 错误频发引发关注**：在使用 **Sonnet** 时频繁出现 **429 错误** 的担忧促使了关于资源耗尽的讨论，并建议避免将流量导向 **Anthropic** 的回退（fallback）选项。用户强调了保持稳定 API 访问的必要性。
   - 这涉及到在高流量场景下对鲁棒的错误处理和速率管理策略的需求。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **GPU 对决：RX 6900 XT vs RTX 4070**：用户讨论了 **GPU 性能**，对比了 **RX 6900 XT** 和 **RTX 4070**，并指出 AMD 显卡可能因 CUDA 依赖项而表现落后。
   - **VRAM** 被证明至关重要，大多数人推荐 Nvidia 显卡，因为在图像生成过程中效率更高且显存问题更少。
- **利用 Inpainting 技术为图像定型**：围绕使用 **ipadapter** 和 **ControlNet** 等 **inpainting** 技术为图像应用特定风格展开了讨论。
   - 成员们敦促分享图像，以便在不改变原始元素的情况下获得关于**风格迁移（style transfers）**的更好反馈。
- **ControlNet 模型备受关注**：一位用户对 **ControlNet 模型** 的咨询引出了一个分享的 [GitHub 链接](https://github.com/lllyasviel/ControlNet)，提供了相关见解和示例。
   - 该分享资源强调了对扩散模型的控制，通过视觉辅助工具使其更容易理解。
- **新手对 Automatic1111 UI 的困惑**：新用户在聊天中涌入大量关于 **Automatic1111 UI** 的咨询，寻求安装支持和优化配置。
   - 建议包括探索 **Forge WebUI**，作为解决常见 **Automatic1111** 问题的潜在方案。
- **社区集结协助图像生成**：成员们积极寻求关于使用 **Stable Diffusion** 进行**图像生成**各方面的帮助，讨论工作流优化。
   - 社区非常强调**社区支持**，特别是针对本地连接问题等故障排查挑战。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere API 吸引新用户**：一位新成员对 **Cohere API** 赞不绝口，强调了其在用极少代码设置多工具 Agent 方面的**简洁性**。
   - 在将 AI 集成到团队工作流中时，*开发者体验是一个重要因素*。
- **深色模式引发热议**：用户对 **Cohere 的新深色模式** 表现出极大热情，频道内讨论非常活跃。
   - 这一功能的引入是一个广受欢迎的改进，许多人指出它提升了用户体验。
- **数据保留问题引发关注**：用户询问如何限制 **Cohere 存储用户 Prompt**，引发了关于**数据保留设置**的讨论。
   - 一位成员提供了一个详细说明如何退出的链接，强调了数据隐私的重要性。
- **使用大量样本进行微调**：一位成员分享称，他们在微调中使用了 **67,349 个样本**，由于 API 限制，将其拆分为每批 **96** 个。
   - *“不确定这是否是正确的方法”* 反映了他们对该流程的不确定性。
- **Rerank API 数据处理遇到困难**：一位用户注意到，在使用 Python SDK 时，Rerank API 未能按预期返回文档，特别是在使用 'return_documents: True' 参数时。
   - 通过 Thunder Client 进行的测试表明 SDK 可能存在 Bug，目前正在进一步调查。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **语音模式故障**：成员们反映了对**高级语音模式**的沮丧；在 iOS 上重新安装 App 解决了问题，但在 Mac OS 上无效。
   - 一位成员提到该模式有时间限制，较短的响应让人感到效率低下。
- **Hinton 和 Hopfield 荣获诺贝尔奖！**：John J. Hopfield 和 Geoffrey E. Hinton 因在机器学习领域的开创性工作获得了 **2024 年诺贝尔物理学奖**。
   - 讨论中出现了对**机器学习与物理学**交叉点的质疑，反映了对认可 AI 贡献的怀疑态度。
- **Anthropic 推出高性价比 API**：Anthropic 推出了 **Message Batches API**，允许在 **24 小时**内进行多达 **10,000 次查询**的异步处理。
   - 一位成员指出其与 **OpenAI 的 batching** 相似，暗示了竞争格局的日益激烈。
- **Salesforce 的生成式 UX 起航**：Salesforce 推出了 **Generative Lightning UX**，旨在根据用户需求动态定制企业应用布局。
   - 目前处于试点阶段，Salesforce 正在积极寻求用户反馈，以迎接预期的 **2025 年发布**。
- **Weights & Biases 揭秘 Cursor 使用技巧**：Weights & Biases 的一次 **Cursor tips & tricks** 会议强调了在团队间分享有效使用策略的重要性。
   - 随后启动了一个跟进线程，以对这些实用技巧进行更深入的讨论。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **知识图谱增强 LLM 能力**：最近的一场演示重点介绍了与 LLM 集成的**知识图谱**，展示了其潜在优势，让与会者对实际应用充满期待。
   - 讨论集中在增强 Transformer 以兼容这些图谱而不进行扁平化处理，强调了保留结构化数据的必要性。
- **OpenAI 推出 o1 推理系统**：OpenAI 发布了他们的新推理系统 [o1](https://openai.com/o1/)，该系统基于 [Q*](https://www.interconnects.ai/p/q-star) 等模型，并承诺具备在线搜索能力。
   - 尽管前景广阔，但它目前仍是一个原型，其**推理缩放定律（inference scaling laws）**表明处理成本很高。
- **Diff Transformer 改进注意力机制**：[Diff Transformer](https://arxiv.org/abs/2410.05258) 采用差分注意力机制，在减少噪声的同时增强对相关上下文的关注，提升了长上下文建模的性能。
   - 这种方法在防止幻觉方面特别有效，在特定应用中表现优于传统模型。
- **Google 关于大规模模型合并的见解**：Google 的研究调查了大规模模型合并，对高达 **64B 参数**的语言模型进行了实验，并通过 [arXiv](https://arxiv.org/abs/2410.03617) 分享了发现。
   - 该研究对合并大型模型所带来的性能提升的泛化性和持久性提出了疑问。
- **对免费文本转视频模型的关注**：一位用户询问是否有**免费的文本转视频模型**（动画或其他类型），并提到 *animate2diff* 可能是一个可用资源。
   - 社区表达了收集更多关于此话题见解的愿望，并寻求其他成员的贡献。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **推理优化之旅开启**：一位新用户表达了希望使用 **Triton** 和基于 **CUDA** 的优化来**开启他们的推理优化之旅**，这反映了人们对高级引擎优化的兴趣日益增长。
   - *对于新手来说，利用社区知识在该领域成功探索至关重要。*
- **对 HBM 有效性的怀疑**：**HBM** 仍然是 **H100** 等设备的重要成本因素，引发了关于其效用以及与 **LPDDR5** 相比的能效讨论。
   - *社区正在评估其收益是否与其成本相符，特别是在功耗方面。*
- **SRAM Scaling 问题显现**：社区成员指出 **SRAM scaling** 未能跟上逻辑缩放（logic scaling）的步伐，这让来自 Graphcore 等公司的贡献者感到惊讶。
   - *有人对追溯到 **2015** 年的设计疏忽表示担忧。*
- **探索 DataLoaders 的 GPU 加速**：一场热烈的讨论确定了 **DataLoaders** 可以在 GPU 上加速，但多进程（multiprocessing）方面的挑战似乎阻碍了性能。
   - *减少对多进程的依赖可能会提高 GPU 效率。*
- **INT8 Mixed Precision 带来性能提升**：**INT8 mixed precision** 训练在 **4090 GPU** 上实现了 **1.7 倍的加速**，有可能在不进行权衡的情况下与 **A100** 的性能相媲美。
   - *鼓励进行进一步的实验以验证这些结果。*

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 黑客松启动**：为迎接 #SFTechWeek，**有史以来第二次 LlamaIndex 黑客松**将于本周五开始，为创新者提供超过 **$12,000** 的现金奖励。
   - 参与者可以[在此](https://t.co/GG7XRnQg5k)报名并获取关于构建复杂多 **Agent** 系统的见解。
- **LlamaParse Premium 脱颖而出**：**LlamaParse premium** 被定位为一款强大的文档解析器，专为上下文增强的 **LLM** 应用量身定制，擅长处理复杂文档。
   - 该[链接](https://t.co/Zd0pWD3wj2)详细介绍了其处理交错扫描文档和多表 Excel 表格的能力。
- **Oracle 集成新功能**：一项重大更新显示 **Oracle** 增加了 **四项新集成**：data loader、text splitter、embeddings 和 vector search。
   - 这些工具的文档强调了它们的功能，特别是 [data loader 的功能](https://t.co/kGud3qKVgO)。
- **Docstore 支持 Chunks 和完整文档**：成员们确认 **docstore** 能够同时容纳 **chunks** 和完整文档，因为它们在同一个类下运行。
   - *cheesyfishes* 强调了它的适应性，证明其对各种存储需求都有利。
- **Contextual Retrieval 与元数据增强**：关于来自 Anthropic 的 **contextual retrieval** 的见解出现，强调了 **metadata** 和 **chunk enrichment** 对增强模型交互的重要性。
   - 讨论指出，利用 **prompt caching** 来增强未来的可扩展性具有潜力。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 进入 TIOBE 前 50 名！**：[2024 年 10 月 TIOBE 指数](https://www.tiobe.com/tiobe-index/)显示 **Mojo** 已攀升至前 50 种编程语言之列，突显了其作为一种快速且安全语言的吸引力。
   - 成员们注意到 Mojo 在一年内迅速崛起，吸引了原本关注 Python 等更成熟语言的注意力。
- **Mojo 关键字需要更清晰**：针对重新评估 **Mojo** 的 **'inout'** 和 **'borrowed'** 等关键字以增强引用子系统清晰度的讨论浮出水面，这与一个 [GitHub 提案](https://github.com/modularml/mojo/issues/3623) 相关。
   - 参与者一致认为，更清晰的关键字约定可以显著帮助初学者掌握该语言。
- **WebAssembly 与 JavaScript 之争**：关于 **WebAssembly** 是否可以取代 JavaScript 进行 DOM 访问引发了辩论，社区意见不一，重点强调了改进垃圾回收（Garbage Collection）的需求。
   - 讨论揭示了人们对使用 WebAssembly 效率的持续关注，并指出了当前执行模型中潜在的缺点。
- **Max 推理引擎求助！**：一位用户报告了在 Intel NUC 上使用 **max inference engine** 时遇到的问题，特别是在通过 **TorchScript** 和 **ONNX** 使用时，直到他们切换到早于 **2.4** 的版本才解决。
   - 这一解决方案鼓励更多用户检查其版本兼容性，以防止类似问题的发生。
- **图编译时间受到质疑**：针对多个张量操作导致的长达 **400-500 ms** 的**图编译（graph compilation）**时间，社区表达了担忧。
   - 讨论建议创建可重用的操作（如通用的 reshape），作为简化图创建过程的一种方法。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **诺贝尔奖授予神经网络领域**：**2024 年诺贝尔物理学奖**授予了 **John J. Hopfield** 和 **Geoffrey E. Hinton**，以表彰他们在**人工神经网络**方面的基础性工作。这一认可强调了他们对机器学习的卓越贡献。
   - 社区对这一荣誉表达了“温馨”的感受。
- **OpenAI 获得独立算力**：据 CFO **Sarah Friar** 称，由于 Microsoft 响应速度较慢，OpenAI 正在通过与 Microsoft 竞争对手的数据中心协议来确保**其自身的算力容量**。鉴于 Microsoft 的信任问题，此举被视为“激进但并不令人意外”。
   - 讨论的一种替代策略包括这些协议对 OpenAI 在竞争市场中自主权的影响。
- **8B 模型在文本任务上优于 11B**：据报道，**8B** 模型在**纯文本**任务中比主要为图像设计的 **11B Vision** 对应模型更有效。用户指出，“所有的增加都是为了处理图像”，这表明在文本性能上存在权衡。
   - 社区对这种性能差异将如何影响未来的模型开发感到好奇。
- **AI 可解释性的重要性日益增加**：一篇博客文章强调了随着大型语言模型（LLMs）从单一任务表现演变为复杂的**系统级生产力**，**可解释性**的重要性正在不断提升。这种对**可审计推理**的需求在围绕 AI 问责制的讨论中持续升温。
   - 随着模型变得越来越复杂，建立透明度对于培养用户对 AI 应用的信任和理解至关重要。
- **采样见解与行业认知**：参与者讨论认为，许多大公司将**采样（sampling）**方法视为黑盒，主要关注 **beam/nucleus** 技术，而对其他替代方案的探索不足。这引起了**贝叶斯主义者**对当前所用采样方法质量的担忧。
   - 呼吁采用更好的采样技术，并对主流方法之外的领域进行更广泛的探索。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Discord 体验问题引发挫败感**：成员们对被移出 Discord 表示沮丧，质疑这是否是一场 *psyop*，而其他人则强调了不同设备上的性能差异。
   - 这些问题引发了关于潜在解决方案以及支持团队需要改进沟通的讨论。
- **周边商品与推荐奖励推测升温**：一位新成员询问了关于推荐奖励相关周边商品的公告，但聊天中并未详述目前的活动。
   - 关于潜在奖励的猜测仍然是成员们关注但尚不明确的话题。
- **中国强大的声激光器震撼发布**：一段令人兴奋的视频透露，**中国**研制出了**世界上最强大的声激光器**，展示了令人印象深刻的技术。
   - 你可以在这段[视频](https://www.youtube.com/embed/LbtAFX7Pg6M)中观看实况，该视频引发了大量关于声学技术进步的讨论。
- **Cerebras IPO 正面交锋 Nvidia**：围绕 **Cerebras** 在 IPO 过程中可能遇到的**挑战**展开了讨论，特别是与 **Nvidia** 的竞争。
   - 更多详细见解请参阅这篇揭示这一重大行业事件的文章，阅读更多请点击[这里](https://www.perplexity.ai/page/cerebras-ipo-challenges-nvidia-LmwxVQHLRa.VXzSMV4Ubkw)。
- **速率限制提升请求引发紧迫感**：一位成员紧急寻求关于申请提升 Rate Limit（速率限制）的指导，并提到多次给支持团队发送邮件均未收到回复。
   - 关于是否联系了正确的支持邮箱的澄清表明，沟通流程中可能存在疏漏。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **创建“创建工具的工具”**：一位成员强调需要**创建工具的工具**，以提升未来开发的**效率**。
   - 这类工具代表了增强自动化和社区参与度的一个日益增长的趋势。
- **助手开发助手**：成员们探索了开发**可以创建其他助手的助手**的巨大潜力。
   - 这种 **meta-development**（元开发）的概念有望显著提升生产力。
- **自定义 LM 与 Adapter 的抉择**：围绕何时选择**自定义 Adapter** 而非**自定义 LM**，讨论指出需要更清晰的文档说明。
   - 成员们建议审查现有的 [语言模型文档](https://dspy-docs.vercel.app/docs/building-blocks/language_models) 以进行改进。
- **自定义 LM 客户端逐步淘汰**：**DSPy 2.5** 已弃用除 `dspy.LM` 之外的所有自定义 LM 客户端，这些客户端也将在 **DSPy 2.6** 中逐步淘汰；鼓励用户进行迁移。
   - 可以在此 [Notebook](https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb) 中找到有用的迁移指导。
- **LM 配置困惑**：出现了一个关于 `lm_kwargs` 未在 **MIPROv2 optimizer** 中填充的问题，引发了对预期行为的质疑。
   - 一位成员确认 `lm.kwargs` 应该包含 kwargs，除非 **predictor** 进行了明确的相反配置。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open-Interpreter 保持 Tool Calling 的一致性**：一位成员询问 **Open-Interpreter** 如何确保准确的 Tool Calling，得知这在很大程度上归功于与 LLM 配合使用的 System Message。
   - *Mikebirdtech* 澄清说，虽然它不是严格确定性的，但 System Message 支持了可靠的性能。
- **探索 Structured Output 的潜力**：讨论了用于自定义 Tool Calling 的 **Structured Output**（结构化输出），因为过去的实验暗示了巨大的未开发潜力。
   - 大家普遍认为，来自 **Ollama** 和 **llamacpp** 等工具的增强功能可能会使此类开发变得可行。
- **Mozilla AI 演讲即将开启**：**Mikebirdtech** 提醒大家下周将有来自 **Mozilla AI** 关于开源倡议的演讲，敦促大家通过 Discord 活动中的链接参加。
   - 现场气氛热烈，凸显了该演讲对 AI 爱好者的潜在相关性和吸引力。



---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **线下讲座出勤受限**：由于教室容量限制，只有 **Berkeley 学生**可以线下参加讲座，其他人只能远程参与。
   - 这一决定引发了关于 **Berkeley MOOC** 准入门槛和社区参与度的讨论。
- **关于 AI Agents 使用 Autogen 的辩论**：成员们就生产环境中使用 **Autogen** 还是使用原始 API 调用来实现其初创公司中的 AI agents 展开了辩论。
   - 这一对话强调了针对实际应用优化 **Autogen** 的重要性。
- **使用 Redis 构建框架**：一位用户分享了关于开发自己的框架并使用 **Redis** 连接 worker 的见解，旨在简化操作。
   - 该方法的目标是**减少抽象层级**并提高对复杂用例的控制力。
- **Omar 令人兴奋的 DSPy 讲座**：一位成员对 **Omar** 即将举行的 **DSPy** 讲座表示期待，认为这是社区中的一件大事。
   - 他们致力于为 **DSPy** 开发做出贡献，展现了对提升该框架能力的浓厚兴趣。
- **对 DSPy 的贡献正在进行中**：该成员计划积极为 **DSPy** 做出贡献，强化了对其开发的承诺。
   - 这种参与说明了人们对增强 **DSPy** 工具和功能的兴趣日益增长。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 网站导航问题被指出**：一位成员担心用户除非点击一个小按钮，否则很难在 **tinygrad** 网站上找到特定页面，指出了可能的导航缺陷。
   - 经过进一步思考，他们确认点击该按钮确实会将用户引导至目标页面。
- **Swift 编译悬赏挑战**：一位用户正在挑战来自 **exo** 的悬赏，旨在将 tinygrad 编译为 Swift，并分享了 [GitHub issue](https://github.com/exo-explore/exo/issues/238) 链接作为参考。
   - 他们希望在保留 exo 的 Python 根基的同时，寻求管理员关于实现这一目标的建议。
- **开发出 Tensor.sum() 的变通方案**：使用 **qazalin 的额外缓冲区计数 PR** 创建了一个变通方案，以解决 **Tensor.sum()** 因缓冲区过多而导致的错误。
   - 该方法被指出**效率非常低**，需要迭代地添加和拆分操作以避免问题。
- **改进的范数（Norm）计算方法**：一个新脚本通过迭代计算**范数**并对其求平方来处理梯度，以优化内存使用。
   - 该方法涉及创建 **norm1_squared** 和 **norm2_squared** 组，增强了稳定性，但牺牲了一些效率。
- **George Hotz 强调文档价值**：George Hotz 强调了阅读问题文档的重要性，引导用户有效地利用现有资源。
   - 该建议旨在提高用户的清晰度，减少围绕 tinygrad 功能的困惑。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **旅行计划存疑**：一位成员表示有兴趣参加活动，但不确定届时是否能够成行。
   - 这种担忧反映了涉及旅行时，行程安排和承诺的复杂性。
- **ChatPromptTemplate 的利用**：一位用户详细介绍了他们在聊天应用中使用 `ChatPromptTemplate` 生成消息的方法，包括示例提示词设置。
   - 该实现展示了如何构建 `example_prompt` 和 `example_selector` 以增强聊天交互。
- **消息中的引号转义导致 JSON 问题**：多位用户报告称其 `messages` 对象中的双引号被编码为 `&quot;`，导致 JSON 格式无效。
   - 他们寻求关于防止这种转义问题的指导，以确保在聊天中传输有效的 JSON。
- **集成 FewShotChatMessagePromptTemplate**：一位用户演示了如何使用指定的示例选择器和提示词来实现 `FewShotChatMessagePromptTemplate`。
   - 该设置旨在增强上下文并改善聊天交互过程中的响应。

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **BF16 训练问题需要关注**：调整**学习率** (LR) 对于正确的 **BF16** 训练至关重要，因为 **BF16 权重**在微小变化下可能无法正确更新，这可能导致性能不佳。建议实施 **BF16 混合精度训练**来解决此问题，尽管额外的 **FP32 梯度**会增加内存负担。
   - *另一位成员*强调，如果没有适当的学习率调整，**BF16** 训练可能会导致严重的效率低下。
- **理解 1B 模型中的 BF16 效应**：讨论中提到 **BF16** 在 **1B 模型**中的影响更为显著，这可能是由于较少的参数对更新的响应较小。一位成员指出，**BF16 权重更新下溢 (underflow)** 可以追溯到 `weight` 与 `weight_delta` 之间的关系。
   - 提议*通过 BF16 混合精度训练的结果进行验证*，作为澄清这些观察结果的一种方式。
- **实验随机舍入 (Stochastic Rounding)**：人们对在优化器权重更新中引入**随机舍入**产生了兴趣，旨在评估其对 **Torchtune** 的潜在影响。一位成员表示准备运行实验，并仔细权衡收益与复杂性。
   - 团队旨在探索这种方法的实际影响，同时保持对任何由此产生的复杂性的认识。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Hinton 获诺贝尔奖的前瞻性**：50 年后，授予 **Geoffrey Hinton** **诺贝尔奖**可能会像 1949 年授予 **Moniz** 脑白质切断术奖项一样被评价，这反映了与当今机器学习进展的显著脱节。
   - 论述指出，*Hinton 对现代技术的理解与当前领域现状严重脱节*。
- **大规模模型合并见解**：来自 Google 的**新研究**讨论了针对高达 **640 亿参数**语言模型的**模型合并**方法，强调了影响性能和泛化能力的因素。
   - 该研究在 [tweet](https://x.com/prateeky2806/status/1843643582432854171) 中被引用，其发现引发了关于在更大架构中合并有效性的关键询问。
- **围绕 Autoarena 工具的好奇心**：一位用户介绍了 **Autoarena** 工具（访问地址 [autoarena.app](https://www.autoarena.app)），强调了其针对技术用户的潜在功能。
   - 该工具引发了兴趣，导致了对其在该领域可能应用的推测。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**OpenAccess AI Collective (axolotl) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道划分的详细摘要和链接

{% if medium == 'web' %}

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1293291064202891265)** (1 条消息): 

> - `Nvidia models`
> - `Meta's VLMs`
> - `Open Source Hugging Face Accelerate 1.0`
> - `Video language models`
> - `ColPali multimodal retrieval`

- **Nvidia 发布高效 Llama-3.1-Nemotron-51B**：Nvidia 推出了 [Llama-3.1-Nemotron-51B](https://x.com/NVIDIAAIDev/status/1838263496049570053)，这是一款经过 NAS 优化的模型，在保持准确性的同时，在单张 H100 GPU 上实现了 **2 倍吞吐量**。
   - 用户可以通过 [Nvidia AI](http://ai.nvidia.com) 的 API 体验该模型，或从 [Hugging Face](https://huggingface.co/nvidia/NVLM-D-72B) 下载。
- **Meta 的 CoTracker 2.1 提升视频运动预测水平**：Meta 发布了 [CoTracker 2.1](https://x.com/NielsRogge/status/1842958590396772599)，这是其模型的增强版本，能够在单张 GPU 上联合追踪 **7 万个点**。
   - 相应的论文可以在[这里](https://huggingface.co/papers/2307.07635)找到，详细介绍了视频运动预测方面的进展。
- **Hugging Face Accelerate 1.0 开启新可能**：Hugging Face 宣布发布 [Accelerate 1.0](https://x.com/TheZachMueller/status/1843320011139813644)，其中包含许多用于优化模型训练的新功能。
   - 欲了解更多详情，建议用户阅读[公告博客](https://huggingface.co/blog/accelerate-v1)。
- **视频转文本模型终于到来**：Hugging Face 为视频语言模型推出了一项新任务，实现了 [video-text-to-text](https://x.com/mervenoyann/status/1843235751666016418)（视频文本转文本）功能。
   - 新功能附带了 transformers 库中提供的详尽文档。
- **ColPali 彻底改变多模态文档检索**：ColPali 是一种创新的多模态文档检索方法，可与 [Qdrant](https://danielvanstrien.xyz/posts/post-with-code/colpali-qdrant/2024-10-02_using_colpali_with_qdrant.html) 无缝集成以实现高效索引。
   - 尽管对其实用性存在一些质疑，但它提供了一种高效索引和搜索 ColPali embedding 的简单方法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/NVIDIAAIDev/status/1838263496049570053),">来自 NVIDIA AI Developer (@NVIDIAAIDev) 的推文</a>：👀 体验高效的 NVIDIA Llama-3.1-Nemotron-51B —— 一款经过 NAS 优化的模型，在保持准确性的同时实现了 2 倍吞吐量，可在单张 H100 GPU 上运行。✨ 试用 Llama-3.1-Nemotron-51B N...</li><li><a href="https://x.com/NielsRogge/status/1842958590396772599)">来自 Niels Rogge (@NielsRogge) 的推文</a>：Meta 在 @huggingface 上发布了 CoTracker 2.1，这是其基于 Transformer 的视频运动预测模型的改进版本！能够在单张 GPU 上联合追踪 7 万个点。论文（附链接...</li><li><a href="https://x.com/triswarkentin/status/1841823657108373838)">来自 Tris Warkentin (@triswarkentin) 的推文</a>：Gemma 2 变得更好了！🚀 新的日语微调 2B 模型以及 15 万美元的 Kaggle 竞赛，旨在为每种语言构建 Gemma 模型。很高兴 @sundarpichai 在这里分享这一激动人心的时刻！阅读更多...</li><li><a href="https://x.com/TheZachMueller/status/1843320011139813644)">来自 Zach Mueller (@TheZachMueller) 的推文</a>：这一天终于到来了，@huggingface Accelerate 1.0 现已发布！有大量的新功能值得探索，未来还有更多。我将快速分享我最喜欢的部分 🧵 想要回顾一下，请访问...</li><li><a href="https://x.com/mervenoyann/status/1843235751666016418)">来自 merve (@mervenoyann) 的推文</a>：你的 LLM 无法理解视频和图像？太遗憾了 😔 幸运的是，我们为视频语言模型发布了一项新任务 🤗 在 @huggingface /models 的左侧选项卡中查找 video-text-to-text ⏯️ 它还附带...</li><li><a href="https://x.com/AdinaYakup/status/1843318863380750581)">来自 Adina Yakup (@AdinaYakup) 的推文</a>：这是来自 @huggingface 中文社区的排行榜和竞技场集合 🔥🏆🇨🇳 https://huggingface.co/collections/zh-ai-community/leaderboards-and-arenas-664b6913bfd9b93ba4ac242...</li><li><a href="https://x.com/flngr/status/1842358136239210866)">来自 Julian Bilcke (@flngr) 的推文</a>：现在的样子（我是服务器的唯一用户，所以很流畅 😂）</li><li><a href="https://x.com/vanstriendaniel/status/1841515562557702330),">来自 Daniel van Strien (@vanstriendaniel) 的推文</a>：ColPali 是一种令人兴奋的多模态文档检索新方法，但有人怀疑它在现有向量数据库中的实际用途。事实证明，使用 @qdrant_engine 来索引和搜索...非常容易</li><li><a href="https://x.com/IAMJBDEL/status/1841627341195510256),">来自 JB Delbrouck (@IAMJBDEL) 的推文</a>：Paper Central 是一个新的 🤗 Hugging Face Space，旨在提供最新研究论文的最新信息。它是第一个将所有关键来源汇集在一起的门户...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1292925458819715102)** (566 条消息🔥🔥🔥): 

> - `LLMs 性能与局限性`
> - `Tokenization 的重要性`
> - `AI 进展与研究`
> - `机器学习中的连接组（Connectome）复制`
> - `GPU 使用与兼容性` 


- **LLMs 受限于训练分布**：讨论强调语言模型在其训练分布范围内运行，这意味着它们无法在训练范围之外产生突破性发现或解决新颖问题。
   - 参与者指出，虽然像 GPT-2 和 GPT-3 这样的 LLMs 可以协助完成任务，但它们缺乏真正的理解能力和独立过滤输出的 Agency。
- **正确 Tokenizers 的重要性**：参与者一致认为，必须使用特定于模型系列的正确 Tokenizer，因为不匹配的 Tokenizers 不会产生有效的结果。
   - 有人指出，虽然学习不同的 Tokenizers 可能不是必不可少的，但了解许多模型共享 Tokenization 可以提高效率。
- **AI 研究与可能的应用**：成员们对正在进行的 AI 研究以及开源模型在各个领域的潜在应用表示好奇。
   - 有建议在 ML 中复制连接组（Connectome）结构，并指出之前对简单生物的研究可以作为潜在的起点。
- **查询 GPU 兼容性与性能**：频道探讨了不同 OS 环境下 GPU 设置的兼容性，特别是与 AI 模型使用和性能相关的方面。
   - 参与者询问了如何在 Windows 和 Linux 设置中高效地使用 CPU 进行模型推理，重点在于能效和输出效果。
- **模型升级与可行性**：用户讨论了将当前 LLM 模型升级到新版本的可行性，权衡了性能提升与现有设置之间的关系。
   - 讨论强调了 LLM 技术的持续发展，敦促用户考虑在本地和云端环境中增强能力的进展。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.mov-axbx.com/wopr/wopr_concept.html">来自 Building WOPR 的推文：一台 7x4090 AI 服务器</a>: 未找到描述</li><li><a href="https://tenor.com/view/bugs-bunny-looney-tunes-cartoons-gif-25067683">Bugs Bunny Looney Tunes GIF - Bugs Bunny Looney Tunes Cartoons - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/spaces/Vipitis/shadermatch">ShaderMatch - Vipitis 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://tenor.com/view/hehe-hee-smile-steve-harvey-gif-7550012">Hehe Hee GIF - Hehe Hee Smile - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://flywire.ai/">FlyWire</a>: 未找到描述</li><li><a href="https://ollama.com/unclemusclez/unsloth-llama3.2/tags">标签 · unclemusclez/unsloth-llama3.2</a>: 使用 Unsloth 的 Llama 3.2</li><li><a href="https://github.com/TuragaLab/flyvis">GitHub - TuragaLab/flyvis: 一个受连接组约束的果蝇视觉系统深度机制网络 (DMN) 模型</a>: 一个受连接组约束的果蝇视觉系统深度机制网络 (DMN) 模型 - TuragaLab/flyvis
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1293034102957805721)** (7 messages): 

> - `Jailbreaking LLMs`
> - `Alpaca Dataset and Fine-tuning`
> - `Model Merging at Scale`
> - `Google's Research Contributions` 


- **探索 LLMs 的 Jailbreaking 技术**：一位成员正在学习如何对 **LLMs** 进行 **jailbreak**，这是 AI 领域一个日益增长的研究和实验方向。
   - 随着社区寻求理解模型的局限性，对 **jailbreaking** 的关注变得越来越普遍。
- **使用 Alpaca 数据集和稳定模型进行练习**：一位成员使用 **Alpaca dataset** 进行微调，但表示在 **Qwen 2 1.5 instruct** 模型上的进展并不顺利。
   - 他们目前面临挑战，并正在寻求同行关于改进实现的建议。
- **Google 关于大规模 Model Merging 的见解**：来自 **Google** 的关于 **model merging at scale** 的新工作解决了关于大型模型（最高达 **64B parameters**）性能的重要问题。
   - 该研究探讨了模型大小和合并方法等不同因素如何影响 **generalization** 和 held-in 性能，并附带了[完整论文](https://arxiv.org/abs/2410.03617)的链接。
- **关于提高 Model Merging 研究可见度的讨论**：一位社区成员表示有兴趣提高 **model merging** 工作的可见度，并建议在讨论环节进行潜在的演示。
   - 另一位成员提议进行一次 **reading group talk**（类似于他们之前的演示），以分享其研究见解。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/p">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/prateeky2806/status/1843643582432854171">来自 Prateek Yadav (@prateeky2806) 的推文</a>：有没有想过 model merging 是否在大规模下有效？对于更大的模型，收益是否会消失？也许你考虑过将 model merging 用于大模型的 post-training，但不确定它是否能泛化...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/)** (1 messages): 

pelolisu: Diffusers Logo
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1292950197084618844)** (2 messages): 

> - `Extending Character Set in TrOCR` 


- **扩展 TrOCR 字符集的挑战**：一位成员询问了扩展 **TrOCR** 模型的 **character set/dictionary** 的难度，并寻求实现建议。
   - 聊天中尚未针对这一特定查询提供任何回复，该问题仍处于开放讨论状态。
- **请求回复**：该成员请求如果有人回复关于 **TrOCR** 字符集扩展的查询，请对其进行标记。
   - 这突显了团队成员之间协作和分享见解的需求。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1293288401306517524)** (1 messages): 

> - `T5 model ONNX files`
> - `Conversion methods to ONNX`
> - `ONNX export with torch` 


- **检查 T5 模型的 ONNX 文件夹**：一位成员建议在 Hugging Face 的 T5 模型页面中查看位于 **onnx** 文件夹下的所需 **ONNX** 文件。
   - 如果缺少必要文件，他们建议将 **onnx** 文件夹下载到本地结构中。
- **探索转换为 ONNX 的方法**：讨论中包含了一个 **Hugging Face blog** 的链接，该博客概述了将 Transformers 模型转换为 **ONNX** 的三种方法。
   - 无论是使用底层的 `torch` API 还是 **optimum** 的高层 API，每种方法都能完成相同的导出任务。
- **使用 torch.onnx 导出**：对于底层转换，该成员描述了如何使用 `torch.onnx` 将模型 checkpoints 转换为 **ONNX** 图，并强调了特定参数的需求。
   - 他们提供了一个代码片段，演示了如何使用 **transformers** 和 **torch** 库加载模型和 tokenizer。



**提及的链接**：<a href="https://huggingface.co/blog/convert-transformers-to-onnx">使用 Hugging Face Optimum 将 Transformers 转换为 ONNX</a>：未找到描述

  

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1293034761023000669)** (6 条消息): 

> - `Image Model Identification`
> - `Diffusion Models` 


- **识别图像生成模型**：一位用户询问是使用哪种模型创建了一张特定的低分辨率图像。回复建议可能是 **Flux** 或带有 griffin **LORA** 的 pony 模型，但指出该图像的通用性质影响了识别。
   - *需要强调的是*，如果没有更清晰的分辨率，识别模型仍然具有挑战性。
- **关于 Diffusers 库的困惑**：对话澄清了虽然 *diffusers* 是一个可以加载各种模型的库，但它并不直接指定图像的创建方式。用户提到 **Stable Diffusion XL** 和 **Flux** 都是与 diffusers 兼容的 **Diffusion Models** 类型。
   - 一位用户评论说，许多图像（如讨论中的那张）也可能是使用缺乏特定 **LORAs** 或角色知识的付费模型创建的。


  

---



### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1293278529198358582)** (2 条消息): 

> - `LM Studio 0.3.4 release`
> - `New features in LM Studio`
> - `Bug fixes in LM Studio` 


- **LM Studio 0.3.4 发布，支持 Apple MLX**：**LM Studio 0.3.4** 推出了 [MLX engine](https://github.com/lmstudio-ai/mlx-engine)，用于在 **Apple Silicon Macs** 上高效运行端侧 **LLMs**，并支持从 **Hugging Face** 下载模型。
   - 该更新允许同时运行多个模型，并强制执行结构化 **JSON** 响应，使开发者更容易处理各种 **LLMs**。
- **新工具简化模型管理**：新的键盘快捷键，如用于搜索模型的 `Cmd+Shift+M` 和用于管理 **LM Runtimes** 的 `Cmd+Shift+R`，提升了 **LM Studio** 的用户体验。
   - 更新还包括一个通过 UI 设置结构化输出的功能，简化了配置模型响应的过程。
- **关键错误修复提升稳定性**：该更新解决了关键错误，包括修复了长时间使用后的**黑屏**问题，记录在 [issue #98](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/98) 中。
   - 其他修复确保了本地服务器的额外端口可以正常工作，并解决了 **Obsidian** 中 **embedding API** 的问题，提升了整体功能性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/98))">Issues · lmstudio-ai/lmstudio-bug-tracker</a>：LM Studio 桌面应用程序的错误追踪 - Issues · lmstudio-ai/lmstudio-bug-tracker</li><li><a href="https://github.com/lmstudio-ai/lms/issues/80))">Issues · lmstudio-ai/lms</a>：LM Studio CLI。通过在 GitHub 上创建账号为 lmstudio-ai/lms 的开发做出贡献。</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/142))">Issues · lmstudio-ai/lmstudio-bug-tracker</a>：LM Studio 桌面应用程序的错误追踪 - Issues · lmstudio-ai/lmstudio-bug-tracker</li><li><a href="https://github.com/lmstudio-ai/mlx-engine">GitHub - lmstudio-ai/mlx-engine: Apple MLX engine for LM Studio</a>：用于 LM Studio 的 Apple MLX 引擎。通过在 GitHub 上创建账号为 lmstudio-ai/mlx-engine 的开发做出贡献。</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.4">LM Studio 0.3.4 ships with Apple MLX</a>：在 Apple Silicon Macs 上使用 MLX 进行超快速且高效的端侧 LLM 推理。</li><li><a href="https://lmstudio.ai/download">Download LM Studio - Mac, Linux, Windows</a>：发现、下载并运行本地 LLMs
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1292961918109356143)** (227 条消息🔥🔥): 

> - `LM Studio 更新`
> - `MLX Engine 介绍`
> - `模型性能对比`
> - `LM Studio 相关问题`
> - `用户使用 LLM 模型的经验` 


- **LM Studio 版本更新引发困惑**：用户报告了 LM Studio 自动更新的问题，特别是 0.3.4 版本尚无法通过自动更新功能获取，只能从官网下载。
   - 新版本带来了诸如对话迁移之类的变化，但部分用户在现有工作流中遇到了 Bug。
- **为 Mac 引入 MLX Engine**：MLX Engine 是专为 Apple Silicon Macs 设计的新型推理引擎，为受支持的模型提供了显著的性能提升。
   - 用户注意到，支持 MLX 的模型在大模型上表现出约 10-20% 的速度提升，在小模型上提升高达 50%。
- **性能对比凸显模型能力**：参与者讨论了他们在不同模型上的使用体验，对比了 Llama 3.2、Gemma 2 以及其他模型在兼容硬件上的潜力和性能。
   - Llama 3.1 被认为表现良好，而 Gemma 2 在使用 MLX 时没有表现出显著差异。
- **用户体验揭示了局限性**：一些用户分享了使用 LM Studio 的困难，包括更新限制以及在过渡到新版本时遇到的历史对话问题。
   - 用户还对自己的硬件高效运行新模型的能力表示不满，特别是与高端 GPU 相比时。
- **功耗讨论**：用户对比了推理过程中 GPU 的功耗，注意到在执行类似任务时，RTX 4090 的功耗明显高于 M3 Max。
   - 这些讨论强调了不同硬件配置之间效率和性能预期的差异。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/bunnycore/Llama-3.2-3B-Mix-IQ4_XS-GGUF">bunnycore/Llama-3.2-3B-Mix-IQ4_XS-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/rombodawg/Rombos-LLM-V2.5-Qwen-14b">rombodawg/Rombos-LLM-V2.5-Qwen-14b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Llama-3.2-11B-Vision">meta-llama/Llama-3.2-11B-Vision · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/intel/s/EOe2ECMtPp">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/ml-explore/mlx">GitHub - ml-explore/mlx: MLX: An array framework for Apple silicon</a>: MLX：适用于 Apple Silicon 的数组框架。通过在 GitHub 上创建账号来为 ml-explore/mlx 的开发做出贡献。</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/9643">Llama-3.2 11B Vision Support · Issue #9643 · ggerganov/llama.cpp</a>: 现在能以任何方式运行吗？
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1292943318031925290)** (81 条消息🔥🔥): 

> - `Linux Resource Usage` (Linux 资源占用)
> - `GPU VRAM Options` (GPU VRAM 选项)
> - `Multi-GPU Configurations` (多 GPU 配置)
> - `Performance of AMD vs NVIDIA` (AMD 与 NVIDIA 的性能对比)
> - `Stable Diffusion Model Efficiency` (Stable Diffusion 模型效率)


- **Linux 在低资源下表现更好**：一位成员表示，**Linux** 应该能更有效地利用较少资源，这更多是个人偏好而非技术能力。
   - 建议通过增加 **RAM** 而非更换 **OS** 来提升性能。
- **比较 VRAM 选项：Tesla vs 4060 Ti**：在考虑增加 **VRAM** 时，成员们讨论了是选择 **24GB** 的 **Tesla P40** 还是 **16GB** 的 **RTX 4060 Ti**，并强调了 **P40** 的显存优势。
   - 指出了 **P40** 性能较慢且仅限于 **inference** 的使用场景，并与 **4060 Ti** 进行了对比。
- **多 GPU 设置的挑战**：关于 **multi-GPU setups** 的讨论强调了正确配置以管理散热和功耗的必要性，其中 **4090** 的体积和 **PCI-e lanes** 是主要考虑因素。
   - 一位成员考虑使用 **A6000** 以获得更好的 **VRAM** 利用率，同时确保大模型的扩展性。
- **NVIDIA 与 AMD 之间的性能差异**：成员们一致认为，将 **RTX 3060** 与 **RX 6600** 组合并不理想，因为效率低下，更倾向于坚持使用 NVIDIA 以获得速度。
   - 组合 GPU 可能无法达到预期效果，正如一位成员指出，使用双 **3060** 虽然能有效提供更多 **VRAM**，但处理速度相近。
- **应对 Stable Diffusion 的性能问题**：一位成员分享了使用 **Stable Diffusion** 的经验以及不同模型对 **VRAM** 的限制，指出较大的模型可能会影响处理速度。
   - 他们暗示了利用 AI 执行特定编码任务的可能性，并分享了关于 **QR codes** 被有效使用但性能结果各异的见解。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://server-konfigurieren.de/product/GPU-Rack-Server/4he-supermicro-4029gp-trt2-xeon-scalable-gpu-server-2">
		Server kaufen bei LS Computersysteme - Die Serverspezialisten
	</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/15vhogy/nvidia_tesla_k80/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.ebay.com/itm/125006475381">AMD Radeon Instinct MI60 32GB HBM2 Graphics Accelerator Mining Card 80MH @ 160W  | eBay</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1292928002975338627)** (246 条消息🔥🔥): 

> - `Unsloth Studio Release` (Unsloth Studio 发布)
> - `Fine-Tuning Models` (模型微调)
> - `Model Merging Research` (模型合并研究)
> - `Performance of LLMs` (LLM 性能)
> - `RAG and Fine-Tuning` (RAG 与微调)


- **Unsloth Studio 备受期待的发布**：用户热切期待 **Unsloth Studio** 的发布，希望它能简化 Windows 上的微调流程，无需复杂的安装。
   - 一位用户表达了对设置 Docker 和 GPU 驱动程序的沮丧，希望能通过可执行安装程序获得无缝体验。
- **微调技术讨论**：对**微调后的模型再次进行微调 (fine-tune a fine-tune)** 是可以接受的，尤其是在为指令模型使用相同的 chat template 时。
   - 然而，将其扩展到多层微调可能会变得过度，讨论中分享了关于潜在陷阱的警告。
- **模型合并研究**：分享了一篇关于**大规模模型合并 (model merging at scale)** 的新论文，讨论了不同模型大小和配置下的方法论及评估。
   - 参与者对合并大型模型的实际收益表示怀疑，并提到了过去对 leaderboard 排行榜的担忧。
- **推理方法之间的性能比较**：有人提出疑问，在消费级硬件上 **vLLM 的推理**是否明显快于 Unsloth。
   - 用户正在权衡设置成本与潜在的性能提升，这表明需要对效率有更清晰的了解。
- **DPO 微调的挑战**：一位用户在尝试对 Llama 3.1 进行 8k 上下文长度的 DPO 微调时遇到了 **OOM** 问题，引发了对 VRAM 消耗的担忧。
   - 讨论强调了 SFT 和 DPO 方法之间不同的资源需求，并呼吁社区支持。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing#scrollTo=vITh0KVJ10qX">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/chat-templates">Chat Templates | Unsloth 文档</a>: 未找到描述</li><li><a href="https://www.all-hands.dev/blog/evaluation-of-llms-as-coding-agents-on-swe-bench-at-30x-speed>">All Hands AI</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2405.09673">LoRA Learns Less and Forgets Less</a>: Low-Rank Adaptation (LoRA) 是一种广泛使用的针对大型语言模型的高效参数微调方法。LoRA 通过仅对选定的权重矩阵训练低秩扰动来节省内存。在...</li><li><a href="https://tenor.com/view/wow-gif-20411229">Wow GIF - Wow - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/ohearn-sad-ohearn-mike-ohearn-sad-mike-sad-gif-13532193191719643333">Ohearn Sad Mike Ohearn Sad GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://arxiv.org/abs/2410.03617">What Matters for Model Merging at Scale?</a>: 模型合并旨在将多个专家模型组合成一个能力更强的单一模型，具有降低存储和推理成本、提高泛化能力以及支持去中心化等优点...</li><li><a href="https://x.com/prateeky2806/status/1843643582432854171">Prateek Yadav (@prateeky2806) 的推文</a>: 有没有想过模型合并在大规模下是否有效？也许对于更大的模型，收益会减弱？也许你考虑过使用模型合并来对大型模型进行后期训练，但不确定它是否能生成...</li><li><a href="https://ollama.com/unclemusclez/unsloth-llama3.2">unclemusclez/unsloth-llama3.2</a>: 带有 Unsloth 的 Llama 3.2</li><li><a href="https://github.com/chigkim/Ollama-MMLU-Pro">GitHub - chigkim/Ollama-MMLU-Pro</a>: 通过创建账号为 chigkim/Ollama-MMLU-Pro 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/macadeliccc/opus_samantha?">macadeliccc/opus_samantha · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1292928175441186887)** (60 条消息🔥🔥): 

> - `Unsloth 在 Windows 上的功能`
> - `为内容审核微调 LLM`
> - `Colab 中的 ShareGPT 格式`
> - `补全任务的 Prompt 设计`
> - `模型训练的 Colab 资源` 


- **Unsloth 与 Windows 的兼容性**：一位用户询问 **Unsloth** 是否可以在不使用 WSL 的情况下在 **Windows** 上运行。
   - 关于此话题未提供直接回复。
- **为内容审核微调 LLM**：一位成员提议针对涉及短文本的**内容审核（Content Moderation）**任务微调 LLM，数据集包含 **50k** 条条目。
   - 建议包括可能使用 **Llama Guard** 和 **Gemma Shield** 来管理分类。
- **数据集格式说明**：讨论了模型训练所需的数据集格式，提到了 **ShareGPT** 和 **HF 的通用格式**。
   - 成员们强调了为了兼容多轮对话（Multiturn）格式而对数据集进行归一化（Normalize）的必要性。
- **代码合并任务的 Prompt 设计**：用户展示了一个旨在合并代码片段的 Prompt 模板，并寻求关于使用 **Instruct** 模型的建议。
   - 出现了关于使用特殊 Token 和 Prompt 结构以获得最佳性能的问题。
- **分享 Colab 资源**：一位成员分享了一个 **Colab notebook** 链接，以帮助进行 **ShareGPT** 和 **Llama** 训练。
   - 该资源受到了积极评价，一位用户对早些时候的挫败感表示了歉意。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnq">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1292925400661622815)** (157 条消息🔥🔥): 

> - `Aider 的功能`
> - `Embeddings 与语义搜索`
> - `Message Batches API`
> - `免费 LLM 选项`
> - `Aider 中的成本估算` 


- **Aider 的编辑选项与偏好**：用户表示需要 Aider 在编码后提示确认 commit，而不是自动 commit，一些人主张完全禁用自动 commit。
   - 同时也出现了对预估成本可见性处理的关注，建议使用更清晰的标签来表明这些数字是估算值。
- **理解 AI 系统中的 Embeddings**：讨论集中在如何利用 Embeddings 进行语义搜索，允许 LLM 通过匹配向量表示来检索相关文档，而 LLM 仅处理实际的文本输入。
   - 用户认识到在不同系统间保持 Embeddings 一致性的重要性，以防止文档检索的相关性丢失。
- **探索 Message Batches API**：讨论了 Anthropic 推出的 Message Batches API，强调了其在异步处理大量查询时的成本效益和更低的价格。
   - 虽然有人质疑延迟响应的实用性，但其他人认识到它在生成训练数据等应用中的潜力。
- **用于编码的免费 LLM**：用户寻求适用于编码任务的可靠免费 LLM 推荐，Llama 3.1 和 Hermes 模型因其能力被提及。
   - 对负担能力的担忧促使用户比较各种免费和付费选项，强调了避免 GPT-4o 等高级模型相关成本的愿望。
- **Aider 中的成本管理**：用户讨论了在 Aider 中追踪项目成本的挑战，强调希望获得每个项目更详细的成本明细，而不仅仅是会话估算。
   - 虽然一些人看重成本显示功能，但其他人建议进行改进，以明确成本仅为估算值。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/install/install.html">安装 aider</a>：aider 是你终端里的 AI 配对编程工具</li><li><a href="https://x.com/AnthropicAI/status/1843695536614060201">来自 Anthropic (@AnthropicAI) 的推文</a>：推出 Message Batches API——一种异步处理海量查询的经济高效方式。你可以一次提交多达 10,000 个查询的批处理。每个批处理将在 24 小时内处理完毕...</li><li><a href="https://aider.chat/docs/config/dotenv.html">使用 .env 配置</a>：使用 .env 文件为 aider 存储 LLM API 密钥。</li><li><a href="https://aider.chat/docs/usage/commands.html#entering-multi-line-chat-messages">聊天内命令</a>：使用 /add、/model 等聊天内命令控制 aider。</li><li><a href="https://x.com/hwchase17/status/1843677417405378910?s=46">来自 Harrison Chase (@hwchase17) 的推文</a>：🚀我们正在 LangGraph 中推出“长期记忆”支持。其核心，长期记忆“仅仅”是一个持久的文档存储，允许你对记忆进行 *put*、*get* 和 *search*...</li><li><a href="https://aider.chat/docs/usage/tips.html">技巧</a>：使用 aider 进行 AI 配对编程的技巧。</li><li><a href="https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-to-the-chat">常见问题解答</a>：关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/config/options.html#--auto-commits">选项参考</a>：关于 aider 所有设置的详细信息。</li><li><a href="https://alexgarcia.xyz/blog/2024/sqlite-vec-hybrid-search/index.html">使用 SQLite 进行混合全文搜索和向量搜索</a>：将 SQLite 内置的 FTS5 全文搜索扩展与 sqlite-vec 向量搜索扩展结合，实现混合搜索！
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1292924705866780682)** (101 条消息🔥🔥): 

> - `Aider Confusion on API Key Usage` (Aider 在 API Key 使用上的困惑)
> - `Command Line Options for Aider` (Aider 的命令行选项)
> - `Context Management in Aider` (Aider 中的上下文管理)
> - `Deepseek Model Usage` (Deepseek 模型使用)
> - `Feedback and Feature Requests` (反馈与功能需求)


- **Aider Confusion on API Key Usage**: 用户澄清了 Aider 在查询 API 时，`.env` 文件中的值优先级高于 YAML 配置，这导致了关于 API Key 的潜在困惑。
   - 这种情况尤其发生在 `.env` 文件和配置文件包含重叠的键名时，因此需要仔细管理环境变量。
- **Command Line Options for Aider**: 讨论了诸如 `--no-suggest-shell-commands` 和指定 `claude-3.5-sonnet:beta` 等模型命令，以增强 Aider 的功能。
   - 推荐使用别名 (Aliases) 和 wrapper scripts 作为简化命令使用的方法，并优先考虑 API Key 的命令行参数。
- **Context Management in Aider**: 用户寻求在 Aider 的上下文中选择性包含大文件部分内容的方法，建议使用外部脚本或命令来提取必要的代码片段。
   - 鼓励用户利用 `/run <command line>` 执行来有效管理上下文，但提醒要注意可能对 LLM 造成的困惑。
- **Deepseek Model Usage**: 一位新用户询问如何在 Aider 中使用 **deepseek/deepseek-coder** 模型，并表示有兴趣将其与 **gpt4o** 结合使用，以增强 architect mode 的功能。
   - 用户讨论了将 Aider 作为混合了自动化的聊天界面的概念化方案，其中模型在定义的角色内运行以增强编码任务。
- **Feedback and Feature Requests**: 邀请用户分享与 Aider 功能相关的反馈或功能需求，鼓励对该应用进行建设性的探索。
   - 强调了与 Aider 进行深思熟虑交互的必要性，因为用户需要处理其命令和配置的复杂性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/install/install.html">Installing aider</a>: aider 是你终端里的 AI 结对编程 (pair programming) 工具</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: 你可以通过命令行或 Python 对 aider 进行脚本化操作。</li><li><a href="https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-to-the-chat">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://aider.chat/">Home</a>: aider 是你终端里的 AI 结对编程 (pair programming) 工具</li><li><a href="https://aider.chat/docs/usage/modes.html#architect-mode-and-the-editor-model">Chat modes</a>: 使用 chat, ask 和 help 聊天模式。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1292937808343994410)** (4 条消息): 

> - `Python 3.13 发布`
> - `Google NotebookLM 播客` 


- **Python 3.13 带来重大更新**：今天标志着 **Python 3.13** 的正式发布，其特色包括具有改进错误消息的 [更好的 REPL](https://docs.python.org/3.13/whatsnew/3.13.html#whatsnew313-better-interactive-interpreter)、[无 GIL 运行](https://docs.python.org/3.13/whatsnew/3.13.html#free-threaded-cpython) 选项，以及引入了 [实验性 JIT 编译器](https://docs.python.org/3.13/whatsnew/3.13.html#an-experimental-just-in-time-jit-compiler)。此外，得益于 Beeware 项目，**iOS 和 Android** 现在已成为 [Tier 3 支持平台](https://docs.python.org/3.13/whatsnew/3.13.html#support-for-mobile-platforms)。
   - 一位成员强调了这些更新的影响，指出对 **移动平台** 的支持表明了扩大可访问性的严肃承诺。
- **使用 Google NotebookLM 创建 AI 播客**：一位成员分享了他们使用 **Google NotebookLM** 创建以 SmartPoi 项目为主题的播客节目的经验，将其描述为想法的融合且格式良好。他们提供了 [概览剧集](https://www.circusscientist.com/2024/10/07/smartpoi-ai-podcast-episode-1/) 的链接，以及如何开始使用 NotebookLM 的指南。
   - 他们提到，尽管内容中存在一些混淆，但他们的家人确信这是一个真实的播客，展示了 AI 生成音频的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://simonwillison.net/2024/Oct/7/whats-new-in-python-313/">What’s New In Python 3.13</a>: 今天是 Python 3.13 的发布日。重大的标志性功能包括具有改进错误提示的 [更好的 REPL](https://docs.python.org/3.13/whatsnew/3.13.html#whatsnew313-better-interactive-interpreter) ...</li><li><a href="https://www.circusscientist.com/2024/10/07/smartpoi-ai-podcast-episode-1/">SmartPoi AI Podcast Episode 1 - Circus Scientist</a>: 概览剧集：本集由 AI 生成！操作方法：我使用了 Google NotebookLM，并上传了所有与 SmartPoi 和 Magic Poi 项目相关的博客文章和网页。查看...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1293184008607371284)** (36 条消息🔥): 

> - `Nobel Prize in Physics` (诺贝尔物理学奖)
> - `AI research and recognition` (AI 研究与认可)
> - `Hinton and Hopfield's contributions` (Hinton 与 Hopfield 的贡献)
> - `Physics community reactions` (物理学界的反应)
> - `Model merging research` (模型合并研究)

- **关于 AI 研究获得诺贝尔奖的争议**：物理学界的许多人正在讨论将**诺贝尔物理学奖**授予 **Hinton** 和 **Hopfield** 以表彰其 AI 工作的恰当性，称其过于牵强，或者反映了在评估严谨科学时价值取向的退化。
   - 一位成员评论说，这可能会助长“炒作”性工作而非有影响力的研究，并认为重大的认可应集中在传统的物理学成就上。
- **对 Hinton 获得认可的复杂情绪**：虽然一些人承认由于 Hinton 在 AI 发展中的历史地位，授予他奖项存在**合理的逻辑联系**，但另一些人认为奖项的分配过于宽泛，偏离了基础物理学。
   - 有观点认为，如果神经网络没有被如此**过度炒作**，该奖项本可以表彰更具影响力的物理学贡献。
- **诺贝尔奖活动中讨论的 AI 伦理**：参与者注意到**瑞典皇家科学院**正在将其重点转向包括 **AI 伦理与安全**在内的讨论，反映了更广泛的社会关注。
   - 这种对话的转变可能表明，该奖项希望涵盖与传统科学交叉的新兴领域。
- **围绕模型合并研究的热情**：来自 Google 的一项关于 **model merging**（模型合并）的新研究揭示了合并大规模模型的影响和性能，展示了关于其可扩展性的有趣发现。
   - 该研究解决了常见问题，并探讨了各种因素对高达 **64B 参数**模型在留存性能（held-in performance）和泛化能力方面的影响。
- **物理学界的反应**：**r/physics** 社区对诺贝尔奖的授予表达了沮丧，将其贴上**高度互联网化现象（Very Online phenomenon）**的标签，并质疑什么才构成“真正的物理学家”。
   - 一些人担心该奖项可能会损害其声望，认为它强调了应用而非传统的卓越物理学研究。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/prateeky2806/status/1843643582432854171">来自 Prateek Yadav (@prateeky2806) 的推文</a>：有没有想过模型合并在大规模情况下是否有效？对于更大的模型，收益是否会消失？也许你考虑过将模型合并用于大型模型的后期训练，但不确定它是否能泛化...</li><li><a href="https://www.openread.academy/en/paper/reading?corpusId=784288">具有涌现集体计算能力的神经网络与物理系统。</a>：OpenRead 阅读与笔记</li><li><a href="https://www.reddit.com/r/Physics/comments/1fyx6yd/yeah_physics/">Reddit - 深入探讨</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1292954537472163972)** (188 条消息🔥🔥): 

> - `Normalized Transformer (nGPT)`
> - `MuP and Initialization`
> - `Diff Transformer`
> - `Gradient Descent Behavior`
> - `Generative Reward Models`

- **nGPT 提出超球面表示**：该论文介绍了一种名为 **nGPT** 的新型 Transformer 架构，它将所有向量在超球面上进行归一化，声称训练效率提升高达 **20x**。
   - 该架构专注于表示学习，每一层在修改其输出的同时保持单位范数向量，从而可能提高学习速度。
- **对 MuP 有效性的担忧**：讨论指出 **MuP** 在 Transformer 之外的网络中失效，其理论假设并不成立，特别是在梯度与参数之间的对齐方面。
   - 批评意见包括其在偏置和权重初始化缩放时的表现，导致对其预期收益的误导。
- **Diff Transformer 增强相关上下文**：**Diff Transformer** 通过减去两个 softmax 注意力图来减少对无关上下文的关注，在语言建模任务中表现出更好的性能。
   - 它强调稀疏注意力模式，并在长上下文建模、缓解幻觉和 In-context Learning 方面提供了显著优势。
- **理解 Gradient Descent 中的 Scaling Laws**：一个讨论串探讨了在 Gradient Descent 中观察到的幂律行为，并得到了 **Francis Bach** 分析优化真实行为的博客文章的支持。
   - 对话反思了这些 Scaling Laws 在训练过程中如何体现的数学见解，并将其与传统的优化文献进行了对比。
- **Generative Reward Models 及其重要性**：研究社区分享了关于 **Generative Reward Models** 的见解，这些模型利用人类和 AI 反馈来提高 LLM 训练性能。
   - 关于实现和实现有效的 Post-training 性能的讨论，强调了 AI 系统决策中 Reasoning 的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.synthlabs.ai/research/generative-reward-models">Generative Reward Models that Unify RLHF and RLAIF Approaches</a>：一个统一了 RLHF 和 RLAIF 的新型框架，旨在更好地使 LLM 与人类偏好对齐，性能优于传统方法高达 45%。</li><li><a href="https://arxiv.org/abs/2410.02703">Selective Attention Improves Transformer</a>：注意力上下文中的冗余元素会降低性能。我们引入了 Selective Attention，这是一种对标准注意力机制的简单、无参数改进，它减少了对不...</li><li><a href="https://arxiv.org/abs/2410.01623">Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?</a>：低秩训练已成为减少大语言模型（LLMs）训练内存使用的一种极具前景的方法。之前的方法要么依赖于分解权重矩阵（例如 LoRA），要么...</li><li><a href="http://elm.baulab.info/">Erasing Conceptual Knowledge from Language Models</a>：通过低秩模型编辑，在处理无害性、无缝性和特异性的同时，从语言模型中擦除概念性知识。</li><li><a href="https://arxiv.org/abs/2410.01131">nGPT: Normalized Transformer with Representation Learning on the Hypersphere</a>：我们提出了一种新型神经网络架构，即在超球面上进行表示学习的归一化 Transformer (nGPT)。在 nGPT 中，构成嵌入、MLP、注意力矩阵的所有向量...</li><li><a href="https://arxiv.org/abs/2410.05258">Differential Transformer</a>：Transformer 往往会向无关上下文分配过多的注意力。在这项工作中，我们引入了 Diff Transformer，它在消除噪声的同时放大了对相关上下文的注意力。具体而言...</li><li><a href="https://x.com/taha_yssne/status/1843468224232599645">Tweet from Taha Yassine (@taha_yssne)</a>：我喜欢可视化，所以我尝试复现了这一个。在这里，我还展示了超过特定熵阈值的 Token 的第二和第三候选者。不过 Top-p 可能会更有意义。根据 gpt...</li><li><a href="https://arxiv.org/abs/2310.17813">A Spectral Condition for Feature Learning</a>：训练更大规模神经网络的需求推动了对大网络宽度下初始化和训练的研究。一个关键挑战是如何扩展训练，使网络的内部表示...</li><li><a href="https://nickcdryan.com/2024/05/24/adaptive-skip-connections-improve-training/)">Adaptive skip connections improve training</a>：摘要：对残差连接的跳跃（恒等）部分应用单一的、线性的、可学习的权重，可以略微提高训练性能。它还揭示了有趣的训练动态：d…</li><li><a href="https://x.com/main_horse/status/1841807705935372348">Tweet from main (@main_horse)</a>：@papers_anon 他们的代码大约在 4 小时前发布。我导入了它并运行了一些全秩（fft）和 fira。我注意到 FFT 的运行开始时比 fira 好，但很快就因为不稳定而崩溃...</li><li><a href="https://www.wolfram.com/llm-benchmarking-project/">Wolfram LLM Benchmarking Project</a>：Wolfram 对 LLM 性能持续跟踪的结果。该基准测试基于 Wolfram Language 代码生成任务。</li><li><a href="https://x.com/yaroslavvb/status/1843758350171099468">Tweet from Yaroslav Bulatov (@yaroslavvb)</a>：很高兴看到 Francis Bach 在研究梯度下降的真实行为。这与优化文献传统研究的“假设”行为形成了对比。引用...</li><li><a href="https://math.stackexchange.com/a/4981650/998)">Showing $\sum_{i=1}^k i^{-2}(1-i^{-2})^t\approx \frac{\sqrt{\pi }}{2 \sqrt{t}}$ for large $k$</a>：对于较大的 $k$，我观察到以下情况：
$$f(t)=\sum_{i=1}^k i^{-2}(1-i^{-2})^t \approx \frac{\sqrt{\pi }}{2 \sqrt{t}}$$

证明这一点最简单的方法是什么？
Notebook</li><li><a href="https://github.com/microsoft/unilm/tree/master/Diff-Transformer">unilm/Diff-Transformer at master · microsoft/unilm</a>：跨任务、语言和模态的大规模自监督预训练 - microsoft/unilm</li><li><a href="https://www.semanticscholar.org/reader/26e6c380381634082fb1a75ccdd08536ff50d30c">[PDF] Position: LLM Unlearning Benchmarks are Weak Measures of Progress | Semantic Scholar</a>：一个利用人工智能方法提供高度相关结果和新型过滤工具的学术搜索引擎。</li><li><a href="https://www.dropbox.com/scl/fi/bgic6wij5sbqwbaa0iiyp/video1662236635.mp4?rlkey=9avtjodyn1495yc1euc9hhq6e&e=2&st=q9y0yb2x&dl=0">no title found</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages): 

zackt1234: https://discord.com/channels/729741769192767510/1214931475850469426/1292977027254583397

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1292933522545311784)** (138 messages🔥🔥): 

> - `Document Categorization AI`
> - `AI Tools for File Management`
> - `Cloud vs Local AI Costs`
> - `AVM and Multi-modal AI`
> - `AI Subscriptions Comparison` 


- **文档分类 AI 引起关注**：成员们讨论了 AI 通过阅读内容对数据进行分类的潜力，并提出了能够处理其庞大文件集的工具建议。
   - 一些人对目前的能力表示怀疑，认为手动整理文件或使用多个专业工具可能会更容易。
- **云端与本地 AI 成本讨论**：使用 AI 工具分析海量数据的成本是核心关注点，据估计，使用 API 服务分析 18,478 个文件可能耗资约 **$12,000**。
   - 围绕本地 AI 与云端服务的可行性产生了疑问，用户在权衡服务器成本与本地硬件支出。
- **对 AVM 和多模态 AI 的兴趣**：关于 AVM 的讨论强调了其在多模态交互方面的潜力，并强调了未来增强功能可能如何彻底改变用户体验。
   - 成员们注意到各种 AI 技术的融合，并对 AVM 类工具的未来能力表示兴奋。
- **AI 订阅建议**：用户争论是订阅 ChatGPT 还是 Claude，指出 ChatGPT 的 **o1-preview** 存在局限性，而 Claude 对现有用户已被证明非常有用。
   - 共识建议订阅应侧重于获取工具的完整版本，而非受限的预览版。
- **Nvidia 在 AI 硬件领域的统治地位**：关于 Nvidia 的讨论强调了该公司如何成为 AI 硬件的核心，特别是在由于其 CUDA 工具包而在训练和运行大型模型方面。
   - 成员们指出了 AMD GPU 在 AI 应用中面临的挑战，并对旧硬件的高成本和有限支持表示沮丧。



**提及链接**：<a href="https://topai.tools/s/automated-file-organization">70 Best Automated File Organization AI tools - 2024</a>：发现 70 个最佳的付费和免费 AI 自动化文件整理工具，并了解它们的功能和定价。寻找最佳的 AI 自动化文件整理工具。

  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1292975938786558077)** (23 messages🔥): 

> - `Learning Styles in Dog Training`
> - `Prompt Leaderboard Discussion`
> - `Curiosity About Prompt Creation`
> - `Gemini Advanced Prompt Success` 


- **驯犬中的学习风格**：成员们讨论了驯犬与理解 AI 模型之间的类比，强调理解解剖结构可能对某些人有帮助，而其他人可能觉得没必要。
   - *有人指出，不同的个体有独特的学习方式，识别这些多样化的方法至关重要。*
- **关于 Prompt 排行榜的辩论**：提出了是否可以存在 Prompt 排行榜的话题，成员们质疑如何客观地对 Prompt 进行评分。
   - *讨论包括了对评估 Prompt 并进行哈希处理以产生一致输出的可行性的思考。*
- **Yoshijpn 对 Prompt 的见解**：一位成员表达了对具有特定文档的有限 Prompt 集的渴望，并注意到在 4096 字符限制下潜在 Prompt 的巨大数量。
   - *他们评论说，可能的 Prompt 总数远超人类的存储能力，推测其数量达 5 x 10^9982。*
- **寻求 Prompt Engineer 协助**：一位成员为一个项目请求 Prompt Engineer 的协助，并表示愿意支付报酬。
   - *这突显了社区内对熟练 Prompt Engineer 的持续需求。*
- **Gemini Advanced Prompt 的成功案例**：一位成员分享了他们为 Gemini Advanced 创建 Prompt 的成功经验，该 Prompt 在多个对话中都能持续生成高质量的输出。
   - *然而，他们被提醒应遵守社区准则，避免在频道中讨论其他 AI。*


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1292975938786558077)** (23 messages🔥): 

> - `Learning Styles in Training`（训练中的学习风格）
> - `Prompt Engineering Queries`（Prompt Engineering 咨询）
> - `Interest in Prompt Leaderboards`（对 Prompt 排行榜的兴趣）
> - `Limitations of Prompt Length`（Prompt 长度的限制）
> - `Collaboration for Prompt Creation`（Prompt 创作的协作）


- **训练中的学习风格各异**：成员们讨论了学习偏好的差异，将其比作训犬，有些人可能需要深入了解解剖学知识，而有些人则不需要也能训练得很好。
   - 一位成员表示，相比于技术原理，他更喜欢与模型输出进行交互。
- **Prompt 排行榜引发关注**：一位成员幽默地提出了建立 Prompt 排行榜的想法，并询问如何对这些 Prompt 进行评分。
   - 另一位成员对社区对这一想法的潜在反应表示好奇。
- **探索 Prompt 创作参数**：讨论了创建标准化 Prompt 集的前提条件，包括长度限制和输出一致性。
   - 成员们对生成唯一 Prompt 的复杂性表现出兴趣，并对可能存在的 Prompt 总数进行了推测。
- **Prompt Engineering 协作**：一位成员为某个项目寻求资深 Prompt Engineer 的帮助，并表示愿意为高质量的工作支付报酬。
   - 另一位成员分享了他们在 Gemini Advanced 模型上编写高质量 Prompt 以实现一致输出的成功经验。
- **AI 巨头的粉丝**：一位成员幽默地自荐为 Sam Altman，并表达了对 Dario Amodei 的钦佩。
   - 这一俏皮的评论突显了讨论中活跃且多样化的个性。


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1293272888820105238)** (2 messages): 

> - `OpenAI prompt caching`（OpenAI Prompt Caching）
> - `Prompt caching audits`（Prompt Caching 审计）
> - `Cost savings with caching`（通过缓存节省成本）
> - `Updates on model endpoints`（模型端点更新）
> - `Anthropic beta endpoints`（Anthropic Beta 端点）


- **OpenAI Prompt Caching 上线**：上周，**OpenAI prompt caching** 正式上线，能够显著降低推理成本，最高可节省 **50%**。
   - 该功能已为 8 个 OpenAI 模型自动开启，并支持 **Anthropic** 和 **DeepSeek** 等提供商，后续将支持更多。
- **审计你的缓存节省情况**：用户现在可以直接在 **openrouter.ai/activity 页面**审计通过 **prompt caching** 节省的费用。
   - 该功能也可以通过 `/generation` API 访问，以追踪每次生成节省了多少费用。
- **提供缓存使用洞察**：响应体中的 `cache_discount` 字段揭示了该响应在**缓存使用**上节省了多少，有助于用户决策。
   - 然而，像 **Anthropic** 这样的提供商可能会在缓存写入（cache writes）上显示负折扣，从而影响整体成本节省。
- **模型端点更新已推出**：免费模型端点现在将在模型页面上显示准确的端点上下文长度（context length），以提高用户清晰度。
   - 征求关于 **Anthropic `:beta` 自监管端点**的反馈，因为这些端点计划很快结束 Beta 阶段。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/prompt-caching#openai">Prompt Caching | OpenRouter</a>: 优化 LLM 成本高达 90%</li><li><a href="https://openrouter.ai/docs/prompt-caching#inspecting-cache-usage">Prompt Caching | OpenRouter</a>: 优化 LLM 成本高达 90%
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1292973032712966164)** (112 条消息🔥🔥): 

> - `OpenRouter 性能问题`
> - `Anthropic API 使用`
> - `Prompt Caching 详情`
> - `Model Provider 选择`
> - `Rate Limits` 


- **OpenRouter 遇到重复生成问题**：用户报告每个请求会出现两次生成。一名成员怀疑这可能与他们的设置有关，而另一名成员建议增加超时时间以更好地处理。
- **Anthropic 3.5 Sonnet 内容审核的挑战**：一名成员遇到了 Claude 3.5 Sonnet 的审核问题，意识到使用 `:beta` 端点可以避免其中一些问题。常规端点强制执行审核，而 `:beta` 变体允许自我审核。
- **关于 Prompt Caching 机制的见解**：讨论围绕 OpenRouter 文档中详细介绍的 Prompt Caching 展开。成员们注意到它为几个提供商自动执行缓存，但 Anthropic 需要手动激活，且输入 Token 的成本增加 25%。
- **请求的提供商选择策略**：用户查询引导了解如何将请求路由到特定提供商（如 Anthropic）以避免 Rate Limit 错误。默认的负载均衡行为和手动固定（pinning）提供商被强调为可选方案。
- **Google Vertex 的 Rate Limit 担忧**：一名用户报告在使用 Sonnet 时频繁出现 429 错误，表明资源耗尽。建议禁用回退（fallback）选项，将请求重定向到 Anthropic。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://cloud.google.com/vertex-ai/generative-ai/docs/quotas#error-code-429">未找到标题</a>：未找到描述</li><li><a href="https://openrouter.ai/docs/provider-routing#disabling-fallbacks">Provider Routing | OpenRouter</a>：在多个提供商之间路由请求</li><li><a href="https://openrouter.ai/docs/prompt-caching#anthropic-claude">Prompt Caching | OpenRouter</a>：优化 LLM 成本高达 90%</li><li><a href="https://openrouter.ai/docs/prompt-caching">Prompt Caching | OpenRouter</a>：优化 LLM 成本高达 90%</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>：设置模型使用限制</li><li><a href="https://openrouter.ai/docs/requests#uploading-base64-encoded-images">Requests | OpenRouter</a>：处理传入和传出请求
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1292924687504117760)** (95 条消息🔥🔥): 

> - `Stable Diffusion WebUI 设置`
> - `GPU 性能对比`
> - `图像生成与修改技巧`
> - `ControlNet 模型`
> - `通用帮助与资源共享` 


- **为 Stable Diffusion 选择合适的 GPU**：在对比 **RX 6900 XT** 和 **RTX 4070** 时，用户讨论了性能影响，指出像 **6900 XT** 这样的 AMD 显卡由于 CUDA 依赖可能会较慢。
   - 强调了 **VRAM** 的重要性，建议倾向于 Nvidia 以减少内存问题并提高效率，特别是在生成图像方面。
- **特定风格的 Inpainting 技术**：用户分享了关于如何在不改变原始元素的情况下对现有图像应用特定风格的查询，考虑了 **ipadapter** 和 **ControlNet** 等方法。
   - 社区成员建议发布图像，以便更好地协助实现所需的风格迁移，而不会产生不必要的更改。
- **ControlNet 模型解释**：一名用户询问了 **ControlNet 模型**，促使另一名成员分享了一个 [GitHub 链接](https://github.com/lllyasviel/ControlNet) 以获取详细解释和示例。
   - 分享的链接强调了对扩散模型的控制，并提供了视觉示例以便更好地理解。
- **为新用户探索 Auto1111 UI**：新用户提出了关于 **Automatic1111** UI 的问题，以及在哪里可以找到设置问题的支持，寻求最佳配置指导。
   - 聊天中包含了对 **Forge WebUI** 的建议，认为它是解决 Automatic1111 面临的一些常见问题的潜在替代方案。
- **图像生成的社区支持**：成员们就使用 **Stable Diffusion** 进行图像生成的各个方面寻求帮助，讨论了工作流优化并分享了见解。
   - 对话涵盖了社区支持对于解决 Web UI 本地连接等故障排除问题的重要性。



**提到的链接**：<a href="https://github.com/lllyasviel/ControlNet">GitHub - lllyasviel/ControlNet: Let us control diffusion models!</a>：让我们控制扩散模型！通过在 GitHub 上创建账户，为 lllyasviel/ControlNet 的开发做出贡献。

  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1292936288953176064)** (19 messages🔥): 

> - `Cohere API Performance`
> - `Cohere Dark Mode`
> - `Data Retention Settings`
> - `AI Club Collaboration`
> - `Cohere Outage` 


- **Cohere API 给新用户留下深刻印象**：一位新成员对 **Cohere API** 表示赞赏，强调了其只需极少代码即可设置多工具 **Agent** 的**简便性**。
   - 在评估将 AI 集成到团队工作流时，**开发者体验是一个重要因素**。
- **对 Cohere 新推出的 Dark Mode 感到兴奋**：一位用户兴奋地宣布 **Cohere 现在支持 Dark Mode**。
   - 这一发现引起了频道内其他用户的热烈讨论。
- **限制 Cohere 的数据存储**：一位用户询问如何限制 **Cohere 存储用户 Prompt**，并询问是否有此类设置。
   - 另一位成员确认用户可以通过其 **Dashboard** 选择退出（opt out），并提供了数据保留设置的链接。
- **学生俱乐部合作潜力**：一位来自印度 SRM 的学生寻求与 **Cohere** 合作，在校园内启动一个 AI 和数据科学俱乐部。
   - 回复引导他们访问 **Cohere** 的社区资源以获取潜在支持。
- **Cohere 遭遇停机**：用户报告遇到 **503 错误**，表明 **Cohere 服务中断**。
   - 该问题在频道中被记录，因为它影响了对平台的访问。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://dashboard.cohere.com/data-retention">Login | Cohere</a>: 登录以通过易于使用的 API 访问高级 LLM 和 NLP 工具。</li><li><a href="https://cohere.com/research">Cohere For AI (C4AI)</a>: Cohere For AI (C4AI) 是 Cohere 的非营利研究实验室，致力于解决复杂的机器学习问题。 
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1292924974105100381)** (25 messages🔥): 

> - `Fine Tuning Cohere API`
> - `Commercial Use of Cohere APIs`
> - `Changing Frequency and Presence Penalties`
> - `Data Usage and Privacy Controls`
> - `Crafting Effective Prompts` 


- **使用 Batch Processing 进行 Fine Tuning**：一位成员提到使用了总计 **67,349 个示例**进行 **Fine Tuning**，由于限制，将其拆分为每组 **96** 个的批次提交给 API。
   - 他们的感受是*不确定这是否是正确的方法*。
- **Cohere API 的商业用途**：一位成员询问关于将 **Cohere API** 用于商业目的的问题，另一位成员确认其目标客户是**企业市场**。
   - 他们被引导至 [FAQs](https://docs.cohere.com/docs/cohere-faqs#billing-pricing-licensing-account-management) 以获取更多信息。
- **调整 Frequency Penalty 和 Presence Penalty**：一位用户在更改 **Frequency Penalty** 和 **Presence Penalty** 时需要帮助，被引导至 **Dashboard** 中的**高级选项**以查找设置。
   - 频道中分享了一个 Python 示例，演示了如何在 **co.chat** 函数中直接添加这些惩罚参数。
- **数据隐私和使用政策**：一位成员询问关于限制 Cohere 存储用户 **Prompt** 的问题；另一位成员强调了为客户提供的**数据退出政策（data opt-out policy）**。
   - 通过一个链接提供了更多关于 Cohere 如何维护数据控制以及客户可用的数据隐私选项的细节。
- **编写 System Role Prompt 的指南**：针对关于 **System Role** 语言结构的查询，确认其遵循**标准 Markdown 方法**。
   - 成员们被引导至[文档](https://docs.cohere.com/v2/docs/crafting-effective-prompts)以更清晰地理解如何编写有效的 **Prompt**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/v2/docs/crafting-effective-prompts">Crafting Effective Prompts — Cohere</a>: 此页面描述了进行 Prompt Engineering 的不同有效方法。</li><li><a href="https://docs.cohere.com/reference/chat">Chat — Cohere</a>: 根据提供的对话生成模型的响应消息。要了解有关 Chat API 功能的更多信息，请参阅我们的 [Text Generation 指南](https://docs.cohere.com/v2/docs/chat-api...</li><li><a href="https://cohere.com/data-usage-policy">Enterprise Data Commitments</a>: Cohere 维持严格的控制以保护企业数据并尊重企业客户对其数据的权利。 </li><li><a href="https://docs.cohere.com/docs/cohere-faqs#billing-pricing-licensing-account-management">Cohere FAQs — Cohere</a>: Cohere 是一个使用 LLM 的强大平台。此页面涵盖了与功能、定价、故障排除等相关的常见问题。
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1292982440389771264)** (22 messages🔥): 

> - `Rerank API 处理半结构化数据`
> - `Python SDK 问题`
> - `API v1 和 v2 的区别`
> - `Janitor AI 代理链接`
> - `文档中的高级设置` 


- **Rerank API 在处理半结构化数据时遇到困难**：一位用户注意到在使用 Python SDK 时，Rerank API 在请求中使用 'return_documents: True' 参数却不返回文档的问题。
   - 他们通过 Thunder Client 直接测试 API 成功，表明 SDK 中可能存在 Bug。
- **Python SDK 面临安装问题**：一位成员表示，由于 v5.11 的安装问题，他们正在使用 Python SDK v5.10.0，尽管他们之前的代码在 API v1 中可以运行。
   - 该成员计划稍后单独测试 SDK 以缩小问题范围。
- **API v1 和 v2 的使用说明**：讨论了关于 /v1/chat/completions 端点的链接，一位成员要求澄清如何在 Janitor AI 中使用它。
   - 有建议称，虽然没有明确提到，但 SDK 仍然可以使用代理。
- **寻找 Janitor AI 的代理链接**：一位用户请求 Janitor AI 的特定代理链接格式（如 /v1/chat/completions），并表示很难找到。
   - 成员们确认虽然没有明确提供，但可以使用代理，且 SDK 中已包含该功能。
- **需要高级设置文档**：一位成员强调了在文档中添加高级设置的重要性，以便更好地引导用户。
   - 提出这一建议是为了增强用户体验和功能性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cohere.com/docs/overview#example-with-semi-structured-data)">Rerank 概览 — Cohere</a>：此页面描述了 Cohere 的 ReRank 模型如何工作。</li><li><a href="https://docs.cohere.com/v1/reference/chat">Chat (v1 API) — Cohere</a>：根据用户消息生成文本响应。要了解如何使用 Chat API 和 RAG，请参考我们的 [文本生成指南](https://docs.cohere.com/docs/chat-api)。</li><li><a href="https://docs.cohere.com/reference/rerank">Rerank — Cohere</a>：此端点接收一个查询和一个文本列表，并生成一个有序数组，每个文本都被分配一个相关性分数。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1292971042037305387)** (9 messages🔥): 

> - `Companion Discord 机器人`
> - `审核工具`
> - `识别合适的模型`
> - `Hugging Face 资源` 


- **Companion 机器人介绍**：一位成员介绍了由 Cohere 驱动的 **Companion** Discord 机器人，旨在为服务器社区提供动态人格建模和丰富的交互。该机器人不仅提供审核功能，还能进化以提供真实的对话体验，详见 [GitHub 仓库](https://github.com/rapmd73/Companion)。
   - 另一位成员对该项目表示兴奋，称其非常出色并感谢分享的信息。
- **关于审核功能的讨论**：成员们讨论了 Companion 机器人作为服务器环境审核工具的潜在价值。有人评论说，它可以作为现有审核机器人的“调味剂”，增强用户参与度。
   - 这次交流认可了拥有有效工具以促进社区内尊重沟通的重要性。
- **寻找合适的模型**：一位用户询问如何为他们的项目找到合适的模型，指出了社区内的特定需求。另一位成员推荐 **Hugging Face** 作为寻找适用于不同应用的各种模型的宝贵资源。
   - 讨论强调了用户在寻求增强项目时可以获得的资源可及性。



**提到的链接**：<a href="https://github.com/rapmd73/Companion">GitHub - rapmd73/Companion: 一个以有趣且异想天开的方式利用 AI 的 Discord 聊天机器人。同时也提供一些审核工具。</a>：一个以有趣且异想天开的方式利用 AI 的 Discord 聊天机器人。同时也提供一些审核工具。 - GitHub - rapmd73/Companion: A discord chat bot utilizing AI in a fun and whimsical way. Provid...

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1292933795963473962)** (58 messages🔥🔥): 

> - `对高级语音模式的挫败感`
> - `Hinton 和 Hopfield 获得诺贝尔奖`
> - `Anthropic Message Batches API`
> - `Salesforce Generative Lightning UX`
> - `Cursor 提示与技巧`

- **高级语音模式的挫败感**：一位成员表达了无法访问 **advanced voice mode** 的沮丧，并分享说在 iOS 上重新安装应用解决了他们的问题，但在 Mac OS 上却不行。
   - 另一位成员指出 **advanced voice mode** 有时间限制，较短的回复使其感觉效果不佳。
- **Hinton 和 Hopfield 获得诺贝尔奖**：瑞典皇家科学院宣布将 **2024 Nobel Prize in Physics** 授予 **John J. Hopfield** 和 **Geoffrey E. Hinton**，以表彰他们在实现 machine learning 方面的基础性工作。
   - 评论反映了对 **machine learning 和 physics** 重叠部分的怀疑，并提到了关于 AI 基础贡献认可的争议。
- **Anthropic 推出 Message Batches API**：Anthropic 推出了 **Message Batches API**，允许以更低的成本提交多达 **10,000 个查询**，并在 **24 小时** 内异步处理。
   - 一位成员将其与 **OpenAI 的 batching** 进行了比较，指出根据快速查阅文档发现两者非常相似。
- **Salesforce Generative Lightning UX 发布**：Salesforce 发布了 **Generative Lightning UX**，它可以为企业级应用动态创建布局，旨在通过定制信息需求来提升用户体验。
   - 该计划目前处于试点阶段，鼓励用户提供反馈，以便为明年的发布改进产品。
- **Weights & Biases 分享 Cursor 技巧**：一位成员分享了在 Weights & Biases 举行的 **Cursor tips & tricks** 会议上的见解，强调了在团队内讨论使用策略的价值。
   - 提供了一个后续线程，用于更详细地探索分享的技巧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/NobelPrize/status/1843589140455272810">来自诺贝尔奖 (@NobelPrize) 的推文</a>：【快讯】瑞典皇家科学院决定将 2024 年 #NobelPrize 物理学奖授予 John J. Hopfield 和 Geoffrey E. Hinton，“以表彰他们在……方面的基础性发现和发明。”</li><li><a href="https://x.com/alexrkonrad/status/1843638797768286691?s=46">来自 Alex Konrad (@alexrkonrad) 的推文</a>：独家：助力 Airtable、Brex、Notion 和 Stripe 构建 AI 产品的 Braintrust 已在 a16z 领投的 A 轮融资中筹集了 3600 万美元。这家成立一年的初创公司提供 LLM 评估和监控服务……</li><li><a href="https://x.com/anthropicai/status/1843695536614060201?s=46">来自 Anthropic (@AnthropicAI) 的推文</a>：推出 Message Batches API——一种异步处理大量查询的经济高效方式。您可以一次提交多达 10,000 个查询的批处理。每个批处理将在 24 小时内处理……</li><li><a href="https://x.com/schmidhuberai/status/1735313711240253567?s=46">来自 Jürgen Schmidhuber (@SchmidhuberAI) 的推文</a>：3 位图灵奖得主如何重新发表了他们未能归功于原作者的关键方法和思想。在 https://people.idsia.ch/~juergen/ai-priority-dispute... 下有十多个具体的 AI 优先权争议。</li><li><a href="https://www.wheresyoured.at/subprimeai/">《次贷 AI 危机》</a>：我在本通讯中写的任何内容都不是为了散布怀疑或“仇恨”，而是对我们现状以及当前道路可能走向的冷静评估。我相信人工……</li><li><a href="https://x.com/clarashih/status/1843501862764372083?s=46">来自 Clara Shih (@clarashih) 的推文</a>：上周 @OpenAI 推出了 ChatGPT Canvas，这是一个显示文本、代码和可视化输出的界面。在企业中，我们依赖更结构化、可信的 UX 元素——记录详情、列表……</li><li><a href="https://x.com/ricklamers/status/1843616108752056500?s=46">来自 Rick Lamers (@RickLamers) 的推文</a>：恭喜 John J. Hopfield 和 Geoffrey E. Hinton 🙇‍♂️</li><li><a href="https://www.reddit.com/r/midjourney/comments/1fxy">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://x.com/altryne/status/1843738554352185542?s=46&t=2qGo-Hp_MDNyh14F888CkQ">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：今天我和 @weights_biases 的同事们开了一个“Cursor 技巧与窍门”会议，我想我会在这个推文中分享我们“发现”并相互分享的内容 🧵 如果你还没……</li><li><a href="https://www.reddit.com/r/midjourney/comments/1fxy4q6/comment/lqqud6l/?utm_source=share&utm_medium=mweb3x&utm_name=mweb3xcss&utm_term=1&utm_content=share_button">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://x.com/tsarnick/status/1843616586550390803?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Tsarathustra (@tsarnick) 的推文</a>：Geoffrey Hinton 表示他对被授予诺贝尔物理学奖感到“目瞪口呆”，他认为 AI 将在智力上超过人类，所以我们应该担心它“变得……”</li><li><a href="https://x.com/strubell/status/1843349791029567912?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Emma Strubell (@strubell) 的推文</a>：我不怎么上这个网站了，但我回来是因为 @Google 的首席科学家在这里选择花时间试图诋毁我们 5 年多前发表的论文中的一个数字……</li><li><a href="https://x.com/altryne/status/1843738554352185542">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：今天我和 @weights_biases 的同事们开了一个“Cursor 技巧与窍门”会议，我想我会在这个推文中分享我们“发现”并相互分享的内容 🧵 如果你还没……</li><li><a href="https://x.com/schmidhuberai/status/1735313711240253567?s=46&t=Ht1CveN3LQ3w0Dd6ESCvhQ">来自 Jürgen Schmidhuber (@SchmidhuberAI) 的推文</a>：3 位图灵奖得主如何重新发表了他们未能归功于原作者的关键方法和思想。在 https://people.idsia.ch/~juergen/ai-priority-dispute... 下有十多个具体的 AI 优先权争议。
</li>
</ul>

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1292928243497959434)** (39 条消息🔥): 

> - `AI 中的 Knowledge Graphs`
> - `Hermes 模型数据集`
> - `竞赛免费算力`
> - `LLM 评估服务`
> - `使用 LLM 进行原型设计` 


- **Knowledge Graphs 令人惊叹**：最近的一个演示展示了一个与 LLM 无缝集成的 **knowledge graph**，其潜在优势让与会者感到惊讶。
   - 成员们表达了对实际实现的渴望，并讨论了如何增强 Transformer 以处理这些图结构而不进行扁平化处理。
- **讨论 Hermes 模型数据集**：一位成员询问 Nous Research 是否会发布用于 **Hermes 2** 的数据集，确认该数据集为 **openhermes 2.5 dataset**。
   - 此外，还提到 Hermes 2 Pro 的 function calling 数据集最近也已发布。
- **对测试用免费算力的兴趣**：一位成员提议为潜在的竞赛提供 **free compute**（免费算力），引起了频道内其他人的兴趣。
   - 然而，对于是否有人会接受这些资源的提议，目前还存在不确定性。
- **LLM 评估框架**：推荐使用名为 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 的服务对语言模型进行评估。
   - 它旨在促进 few-shot 评估，吸引了那些对各种模型性能指标感兴趣的人。
- **使用图结构进行 LLM 原型设计**：一位成员表示计划使用 LLM 进行原型设计，特别是关于微调现有模型。
   - 关于使用 Transformer 处理无序图的微调技术产生了一些疑问。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b:free/providers">Nous: Hermes 3 405B Instruct (free) – Provider Status</a>：查看提供商状态并向 Nous: Hermes 3 405B Instruct (free) 发送负载均衡请求 - Hermes 3 是一个通用语言模型，相比 Hermes 2 有许多改进，包括先进的 agentic 能力...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness</li><li><a href="https://forcemultiplier.vercel.app/demo">Rubrics by ForceMultiplier.AI - Advanced 3D Force GraphRAG</a>：通过 Rubrics 体验数据分析的未来，这是一个革命性的 3D Force GraphRAG 系统。在沉浸式环境中可视化、分析和理解复杂的数据结构。</li><li><a href="https://neo4j.com/docs/">Neo4j documentation - Neo4j Documentation</a>：Neo4j 文档</li><li><a href="https://networkx.org/documentation/stable/reference/index.html">Reference &#8212; NetworkX 3.3 documentation</a>：无描述</li><li><a href="https://ggc-discrete-math.github.io/graph_theory.html">
   Discrete Math
  </a>：无描述</li><li><a href="https://research.facebook.com/publications/pytorch-biggraph-a-large-scale-graph-embedding-system/">PyTorch-BigGraph: A Large-scale Graph Embedding System - Meta Research</a>：我们介绍了 PyTorch-BigGraph (PBG)，这是一个嵌入系统，它对传统的多关系嵌入系统进行了多项改进，使其能够扩展到具有数十亿个节点的图...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1292987806918115359)** (4 条消息): 

> - `Nous 掩码注意力`
> - `创建评估数据集`
> - `LLM Judge 经验`
> - `Llama 文件位置`
> - `Llama-stack` 


- **Nous 处理打包样本的 attention masking**：一位成员询问 **Nous** 如何处理打包样本中的 **masking attention**，并指出原生的 **Llama** 实现会在较大的因果掩码下失效。
   - 尽管在训练方案中增加了样本打包，该成员报告称，由于失去了批处理推理，效率受到了显著影响。
- **在评估数据集创建方面的合作**：一位成员表示有兴趣在创建和使用 **evals** 方面进行合作，并提到他们正在进行一些令人兴奋的工作。
   - 他们专门寻求任何有使用 **LLMs as Judges** 处理特定评估数据集经验的人的建议。
- **下载后的 Llama 文件位置**：一位用户询问从 **llama-stack** 下载模型后，**Meta** 将 **Llama** 文件隐藏在何处。
   - 另一位成员回答说他们从未听说过 **llama-stack**，使得该问题悬而未决。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1293179616001003601)** (6 messages): 

> - `Diff Transformer`
> - `Model Merging at Scale`
> - `Text to Video Models` 


- **介绍 Diff Transformer**：*Diff Transformer* 通过差异化注意力机制（differential attention mechanism）增强对相关上下文的关注，同时消除噪声，为 Transformer 对无关上下文过度分配注意力的问题提供了解决方案。实验结果表明，它显著提升了长上下文建模（long-context modeling）和缓解幻觉（hallucination mitigation）方面的性能，详见[这篇论文](https://arxiv.org/abs/2410.05258)。
   - 你可以在 GitHub 上探索 [Diff Transformer 代码](https://github.com/microsoft/unilm/tree/master/Diff-Transformer)以进行实际应用。
- **谷歌关于模型合并的实习研究**：一名实习生与 Google 合作研究了大规模模型合并（model merging），探讨了其对参数量高达 **64B** 的语言模型的影响。他们分享了关于合并方法和模型规模如何影响性能与泛化能力的见解，详见其在 [arXiv 上的实习工作](https://arxiv.org/abs/2410.03617)。
   - 讨论中提出了关于合并更大模型时收益持久性的疑问，这是未来模型训练策略的一个重要考虑因素。
- **关于免费文本生成视频模型的咨询**：一位用户询问了是否有可用的免费文本生成视频（text-to-video）模型，包括动画和非动画类型。有人建议尝试 *animate2diff*，社区中的其他人可能也有更多见解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/prateeky2806/status/1843643582432854171">Prateek Yadav (@prateeky2806) 的推文</a>：有没有想过模型合并在大规模下是否有效？对于更大的模型，收益是否会消失？也许你考虑过在大型模型的后训练（post-training）中使用模型合并，但不确定它是否能泛化...</li><li><a href="https://arxiv.org/abs/2410.05258">Differential Transformer</a>：Transformer 往往会过度分配注意力给无关的上下文。在这项工作中，我们引入了 Diff Transformer，它在消除噪声的同时增强了对相关上下文的关注。具体来说，t...</li><li><a href="https://github.com/microsoft/unilm/tree/master/Diff-Transformer">microsoft/unilm 的 Diff-Transformer 分支</a>：跨任务、语言和模态的大规模自监督预训练 - microsoft/unilm
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1292938440782123149)** (3 messages): 

> - `OpenAI o1 model`
> - `Open O1 project`
> - `Large-scale model merging` 


- **OpenAI 发布全新推理系统 o1**：OpenAI 发布了他们的新推理系统 [o1](https://openai.com/o1/)，它基于之前的 [Q*](https://www.interconnects.ai/p/q-star) 等模型构建，并承诺为挑战性任务提供在线搜索能力。
   - 尽管潜力巨大，但 o1 目前仍是一个原型，其**推理缩放定律（inference scaling laws）**证实了处理过程的高昂成本。
- **Open O1 旨在使 AI 力量民主化**：新启动的 Open O1 网站寻求利用先进的开源替代方案来匹配 OpenAI 私有 o1 模型的能力。
   - 他们的使命包括在代码生成和数学问题解决方面实现**类 o1 的性能**。
- **探索大规模模型合并**：一个令人兴奋的实习项目探索了大规模模型合并，讨论了模型规模和合并方法对性能的影响。
   - 这项工作质疑了**模型合并**对于更大的语言模型（高达 **64B 参数**）是否依然有效，研究结果可以在[研究论文](https://arxiv.org/abs/2410.03617)中查阅。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/prateeky2806/status/1843643582432854171">Prateek Yadav (@prateeky2806) 的推文</a>：有没有想过模型合并在大规模下是否有效？对于更大的模型，收益是否会消失？也许你考虑过在大型模型的后训练中使用模型合并，但不确定它是否能泛化...</li><li><a href="https://opensource-o1.github.io/">Open-Source O1</a>：暂无描述</li><li><a href="https://www.interconnects.ai/p/reverse-engineering-openai-o1">逆向工程 OpenAI 的 o1 </a>：将测试时计算（test-time compute）产品化向我们展示了 AI 的未来。探索已深入到语言模型训练中。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1293179616001003601)** (6 messages): 

> - `Diff Transformer`
> - `Model Merging at Scale`
> - `Text to Video Models` 


- **Diff Transformer 通过消除噪声实现更清晰的注意力**：[Diff Transformer](https://arxiv.org/abs/2410.05258) 引入了一种微分注意力机制 (differential attention mechanism)，在抑制噪声的同时增强对相关上下文的关注，从而产生更稀疏的注意力模式。
   - 实验结果显示，它在语言建模以及**长上下文建模 (long-context modeling)**和**幻觉缓解 (hallucination mitigation)**等实际应用中表现出色，优于传统的 Transformer。
- **Google 对大规模模型合并的探索**：Google 的一项新工作研究了在大规模情况下模型合并的有效性，实验对象包括参数量高达 **64B** 的语言模型。
   - 该研究探讨了关于泛化和 held-in 性能的问题，并通过帖子中链接的 [arXiv 论文](https://arxiv.org/abs/2410.03617) 展示了研究结果。
- **关于免费文本生成视频模型的咨询**：一名成员询问目前是否有任何**免费的文本生成视频模型 (free text to video models)**可用，无论是动画还是非动画形式。
   - 作为回应，有人提到了 *animate2diff*，并建议其他成员可能对此话题有更深入的见解。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/prateeky2806/status/1843643582432854171">来自 Prateek Yadav (@prateeky2806) 的推文</a>：有没有想过模型合并在大规模下是否有效？也许对于更大的模型，收益会逐渐消失？也许你考虑过将模型合并用于大型模型的训练后处理，但不确定它是否能泛化...</li><li><a href="https://arxiv.org/abs/2410.05258">Differential Transformer</a>：Transformer 倾向于向无关上下文过度分配注意力。在这项工作中，我们引入了 Diff Transformer，它在消除噪声的同时放大对相关上下文的注意力。具体来说，它...</li><li><a href="https://github.com/microsoft/unilm/tree/master/Diff-Transformer">microsoft/unilm 仓库中的 Diff-Transformer</a>：跨任务、语言和模态的大规模自监督预训练 - microsoft/unilm
</li>
</ul>

</div>
  

---



### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1293064889325191178)** (7 messages): 

> - `Inference optimisation`
> - `HBM concerns`
> - `SRAM scaling issues`
> - `DGX-1 performance comparison`
> - `3D stacking solutions` 


- **开启推理优化之旅**：一位新用户表达了**开始推理优化之旅**的愿望，并寻求关于 **Triton** 和**基于 CUDA 的优化**的帮助。
   - 这突显了社区成员对高级优化的兴趣日益增长，这对新手可能大有裨益。
- **对 HBM 有效性的担忧**：讨论揭示了对 **HBM** 在 **H100** 等设备上性价比的怀疑，强调其成本仍占总成本的很大比例。
   - 此外，其功耗并不比 **LPDDR5** 显著降低，引发了对其效用的质疑。
- **SRAM 缩放问题**：参与者指出了 **SRAM 缩放 (SRAM scaling)** 的问题，指出其未能跟上逻辑缩放的步伐，这对像 Graphcore 这样的公司来说是始料未及的。
   - 对话指出在 **2015** 年左右的设计阶段缺乏对这一问题的预见。
- **NVIDIA DGX-1 系统对比**：一位用户对比了 **GP100** 和 **GV100** GPU 之间的算力能力，在 **FP32 Compute** 和**内存带宽 (Memory Bandwidth)** 指标上有显著差异。
   - 这突显了硬件效率在当前讨论中的持续相关性。
- **通过 3D 堆叠的缓解策略**：展望未来，针对内存架构问题提出的一种解决方案涉及 **3D 堆叠 (3D stacking)**，类似于 **MI300X** 中使用的方法。
   - 该策略旨在通过对 SMs 使用领先工艺，同时将 SRAM 和 I/O 卸载到旧工艺晶圆上来优化性能。



**Link mentioned**: <a href="https://www.cudahandbook.com/2017/10/dont-move-the-data/">不要移动数据！ | </a>：未找到描述

  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1293179473021374487)** (3 messages): 

> - `TMA Descriptor Initialization`
> - `Batch Matrix Multiplication`
> - `Compilation Issues with tl.dot` 


- **TMA Descriptor Initialization 开销问题**: 当前使用主机端 **TMA descriptor initialization** 的实现在处理 4k x 4k 等较小矩阵时会产生约 **80%** 的开销，而对于 32k x 32k 矩阵，开销约为 **10%**。
   - 为定义的 **BLOCK_SIZES** 预缓存描述符能略微提升性能，但仅限于权重，且在 nightly builds 中设备端描述符初始化存在问题。
- **BMM 实现挑战**: 一位成员在对形状为 `[B_M, B_K]` 的 2D 输入 `a` 和形状为 `[B_M, B_K, B_N]` 的 `modified_weight` 执行 BMM 时，遇到了 **tl.dot** 的断言错误。
   - 使用 **tl.broadcast_to** 的变通方法会将所需的 shared memory 增加 **16x**，这被认为是次优的。
- **tl.dot 的编译问题**: 另一位成员指出 BMM 代码仅在 **interpret mode** 下工作，并引用了 GitHub 上关于 **tl.dot** 处理 3D 形状时出现编译错误的相关 issue。
   - 该问题在 [GitHub Issue #4867](https://github.com/triton-lang/triton/issues/4867) 中跟踪，明确了使用 3D tensor 形状时的编译错误。



**提到的链接**: <a href="https://github.com/triton-lang/triton/issues/4867">tl.dot with 3D shapes compilation error · Issue #4867 · triton-lang/triton</a>: This code import triton import triton.language as tl from torch import Tensor @triton.jit def get_three_d_weights( currents, # [B_M, B_N] weight_block, # [B_K, B_N] BLOCK_SIZE_K: tl.constexpr, ): p...

  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1292977997359353876)** (19 messages🔥): 

> - `GPU Acceleration for DataLoaders`
> - `DALI Challenges`
> - `CUDA Operations and Performance`
> - `PyTorch Conference Insights`
> - `SPDL for Efficient Data Loading` 


- **DataLoaders 的 GPU 加速是可能的**: 讨论强调 **DataLoaders** 可以在 GPU 上加速，一些人认为由于 *multiprocessing* 导致的设计限制损害了性能。
   - 一位成员指出，“主要问题……是它使用了多进程”，建议减少多进程以优化 GPU 利用率。
- **对 DALI 的复杂情感**: 一位成员对 **DALI** 的使用体验表示沮丧，称其“上手极具挑战性”，感觉是在浪费时间。
   - 这种困难感得到了他人的共鸣，指出其学习曲线陡峭且在实现过程中存在障碍。
- **探索用于处理的 CUDA 操作**: 讨论了使用 **CUDA operations** 增强预处理效率的潜力，并建议使用 *torch.cuda.graph()* 来减少 kernel launch 开销。 
   - 一位成员指出，注册自定义 CUDA 操作会有所帮助，并强调 **Inductor** 无法像处理内置 torch 操作那样有效地对自定义代码进行推理。
- **来自 PyTorch Conference 的见解**: 一位成员分享了来自 **PyTorch Conference** 的见解，提到了使用 CUDA 操作进行预处理任务带来的吞吐量提升。
   - 分享了关于会议的详细信息，包括日期和注册链接，邀请更多人参加 *sessions*。
- **用于高效数据加载实现的 SPDL**: SPDL 框架被强调为一种创新方法，通过*按阶段并行化流水线*而非基于进程的加载来实现高效的数据加载。
   - 一位成员讨论了这如何带来更高的吞吐量，并承诺即将推出改进 PyTorch 数据加载体验的功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://facebookresearch.github.io/spdl/main/migration/paradigm_shift.html">Paradigm Shift - SPDL 0.0.6 documentation</a>: 无描述</li><li><a href="https://pytorch2024.sched.com/event/1fHn5">PyTorch Conference 2024: Blobs to Clips: Efficient End-to-End Vid...</a>: 在 PyTorch Conference 2024 查看更多关于此活动的信息
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 messages): 

vayuda: 搭载 M 系列芯片的 Mac 是否使用 ARM SVE 指令？
  

---

### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1293224606408249345)** (4 messages): 

> - `CUDA architecture details`
> - `Persistent threads`
> - `Occupancy and register usage` 


- **理解 CUDA 中的 4 个处理块**：一位成员强调，CUDA 中的执行单元是按“每 1/4 SM”分配的，为了获得最佳性能，必须在这 4 个处理块之间平衡工作负载。
   - *Persistent threads* 可能会使调度复杂化，特别是如果像“生产者（producers）”和“消费者（consumers）”这样的线程块处理不当，可能会导致死锁。
- **处理器对 Occupancy 的影响**：一位成员解释说，一个 A100/H100 SM 支持多达 64 个 warps，而划分为 4 个处理器会影响 occupancy，因为它会以 4 的倍数减少可用 warps。
   - 例如，如果一个 kernel 需要 200 个寄存器，它实际上可以运行 8 个 warps 而不是预期的 10 个，从而使 *cudaLaunchCooperativeKernel* 的使用变得复杂。
- **制定 Kernel 启动参数策略**：为了确保 kernel 成功启动，可以指定 *launch_bounds* 功能来分配所需的 warps，但如果优化不当，可能会导致资源浪费。
   - 使用此类 bounds 可以帮助防止死锁，并确保 CUDA 编译器妥善管理寄存器限制，以匹配可用的处理单元。


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1293122895060668458)** (3 messages): 

> - `dtype Clarification`
> - `Quantized Training`
> - `Mixed-Precision Training`
> - `INT8 Speedup on 4090 GPU` 


- **澄清 dtype 类型**：一位成员讨论了两种类型的 **dtype**：tensor/storage dtype 和 compute dtype，强调使用 **quantized/low-bit** tensor dtype 来节省内存，并使用低比特 compute dtype 来加快计算速度。
   - 例如，虽然 **8-bit Adam** 需要 8-bit 的优化器状态（optim state）以提高内存效率，但为了保证精度，计算是在 **FP32** 中处理的。
- **关于 INT8 量化训练的见解**：**INT8 quantized training** 使用 **INT8** 作为 tensor dtype，而 compute dtype 保持为 **BF16**，旨在节省内存；然而，在内存节省和精度方面的结果令人失望。
   - 相比之下，**INT8 mixed-precision training** 采用 BF16 作为 tensor dtype，INT8 作为 compute dtype，在速度和精度方面取得了可喜的结果。
- **INT8 混合精度带来的显著加速**：一位成员惊讶地发现，**INT8 mixed precision** 在 **4090 GPU** 上可以实现 **1.7x speedup** 且无需任何权衡，使其性能可与 **A100** 媲美。
   - 他们强调需要进行更多实验来确认这些发现，同时感谢了一位成员的贡献。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1293045204642959450)** (5 messages): 

> - `ViT Sparsity Experiment`
> - `WeightNormSparsifier`
> - `Model Inference Time` 


- **ViT 稀疏化实验显示速度变慢**：一位在预训练 **ViT** 上进行稀疏化实验的成员注意到，将 **WeightNormSparsifier** 应用于 MLP 和 attention 线性层时，推理时间有所增加，正如 [torchao/sparsity readme](https://github.com/torchao/sparsity) 中提到的那样。
   - 他们询问这种变慢是否属于典型情况，并寻求关于其实现中可能存在的疏忽的反馈。
- **关于 ViT 变慢预期的讨论**：另一位成员对 **ViT** 出现变慢表示怀疑，对最初的观察提出了挑战，并建议审查共享的代码。
   - 他们表示，仅仅因为使用 **WeightNormSparsifier** 就导致性能下降是出乎意料的。
- **需要调用稀疏化方法来提高速度**：一位成员澄清说，仅使用 **WeightNormSparsifier** 导致的预期变慢是因为其掩码（masking）应用，这本身并不会加速模型。
   - 他们建议该成员应调用 `sparsify_` 以利用 sparse kernels 并可能提升性能。


  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1292927279617278003)** (5 messages): 

> - `Raspberry Pi compatibility` (Raspberry Pi 兼容性)
> - `Nobel Prize in Physics` (诺贝尔物理学奖)
> - `Hinton's Nobel Prize win` (Hinton 获得诺贝尔奖)


- **Raspberry Pi 可能支持新软件**：一名成员提到他们还没有为 **Raspberry Pi** 进行编译，但由于之前显示过兼容性，因此有可能运行成功。
   - *他们幽默地补充说，感觉自己像个在 Discord 里摸索的老爷爷，只是随便点点按钮看看会发生什么。*
- **ptrblock 获得诺贝尔物理学奖**：一条突发公告强调 **ptrblock** 因“对物理学的基础性贡献”获得了 **诺贝尔物理学奖**，该消息分享于一条 [推文](https://x.com/jxmnop/status/1843648364459770191) 中。
   - *另一名成员开玩笑说，ptrblock 应该获得诺贝尔和平奖，因为他在调试 PyTorch 代码时保持了他们的心态平和。*
- **Hinton 被公认为首位“纯计算机科学”诺贝尔奖得主**：聊天中指出 **Hinton** 是首位获得“纯计算机科学”诺贝尔奖的得主，这标志着一个重要的里程碑。
   - 这一认可引发了关于计算机科学在重量级奖项中的地位和认可的讨论。



**提到的链接**：<a href="https://x.com/jxmnop/status/1843648364459770191">来自 jack morris @ COLM (@jxmnop) 的推文</a>：突发：诺贝尔物理学奖授予 ptrblock，以表彰其“对物理学的基础性贡献”。

  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1293287425782579230)** (2 messages): 

> - `Raspberry Pi 5`
> - `External GPU gaming` (外接 GPU 游戏)
> - `Amdgpu Linux kernel patch` (Amdgpu Linux 内核补丁)
> - `GLmark2 performance` (GLmark2 性能)


- **Raspberry Pi 5 展现出色的外接 GPU 支持**：在汉诺威创客嘉年华 (Maker Faire Hanover) 看到 [Pineboards](https://www.tomshardware.com/raspberry-pi/raspberry-pi-5-and-external-amd-gpu-used-to-play-4k-open-source-kart-racing-game-pineboards-demos-supertuxkart-using-hat-upcity-lite-board) 演示使用外接 GPU 进行 4K 游戏后，一名成员分享了他们的 GPU 测试平台计划。
   - *测试的热情显而易见*，他们正准备记录 **amdgpu** Linux 内核补丁在 Raspberry Pi 上的进展。
- **记录 GPU 补丁工作**：一名成员提供了一个 [直播](https://www.youtube.com/watch?v=EAlrCFJZlnI) 演示，关于如何为 Raspberry Pi 5 设置外接 GPU 补丁以增强游戏体验。
   - 他们打算详细说明实现过程以及实现“完全”外接 GPU 支持的剩余任务，引发了社区的期待。
- **使用 GLmark2 进行基准测试**：该成员展示了在 Raspberry Pi 5 上利用 AMD RX 460 外接 GPU 运行 **GLmark2** 的情况，强调了性能的提升。
   - 一张记录该设置的照片展示了在 Raspberry Pi 平台上进行 **外接 GPU 游戏** 的潜力。



**提到的链接**：<a href="https://www.jeffgeerling.com/blog/2024/use-external-gpu-on-raspberry-pi-5-4k-gaming">在 Raspberry Pi 5 上使用外接 GPU 进行 4K 游戏 | Jeff Geerling</a>：未找到描述

  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1292973263902871592)** (1 messages): 

> - `ORT Min JS examples` (ORT Min JS 示例)
> - `WebGPU backend` (WebGPU 后端)


- **寻找使用 WebGPU 的 ORT Min JS 开源示例**：一名成员正在寻找任何使用 **WebGPU** 后端的 **ORT Min JS** [开源示例](https://link.to.examples)。
   - 他们特别强调需要实际的实现参考。
- **请求额外资源**：同一名成员还对与 **WebGPU** 和 **ORT Min JS** 集成相关的更广泛资源感兴趣。
   - 他们希望从社区收集一系列有用的链接和指南。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1293068899566358561)** (3 messages): 

> - `BFloat16 conversion` (BFloat16 转换)
> - `Model performance on Mac` (Mac 上的模型性能)
> - `GPU integer shifts` (GPU 整数位移)


- **T5 模型的 BFloat16 转换**：一名用户成功将 T5 系列模型转换为 **BFloat16**，并正在使用 **MLX** 运行它。
   - 尽管存在兼容性担忧，这一转变旨在优化模型性能。
- **Mac 上的 BFloat16 模拟**：有人担心 Mac 不原生支持 **BFloat16**，从而导致模拟运行，这可能会影响性能。
   - 有建议称，如果数值范围不是关键因素，使用 **float16** 可能是 **更好的选择**。
- **质疑浮点转换对性能的影响**：一名成员对往返 **float** 转换导致的性能缓慢表示质疑，指出这仅仅是 **16 位位移**。
   - 他们询问 **GPU 是否没有向量化整数位移**，暗示了潜在的优化空间。


  

---

### **GPU MODE ▷ #[avx](https://discord.com/channels/1189498204333543425/1291829797563011227/1292982635768971314)** (2 messages): 

> - `vpternlogd instruction`
> - `AVX-512 ISA`
> - `Logic design`
> - `Amiga programming` 


- **发现 vpternlogd：三元逻辑的奇迹**：分享了一篇关于 **vpternlogd** 指令的有趣帖子，强调了它利用三个输入执行复杂**位布尔逻辑**的能力。
   - 该操作可以使用 **512-bit registers**，使其成为希望简化逻辑操作的 SIMD CPU 程序员的强大工具。
- **对逻辑设计概念的怀旧反思**：一位成员回想起了逻辑设计课程中的**最小项（minterms）**和**最大项（maxterms）**，并将其与编程的复杂性进行了类比。
   - 他们幽默地提到，这看起来像是 **Amiga 芯片设计者为软件开发者编写了文档**，将怀旧情怀与技术讨论融合在一起。



**提到的链接**：<a href="https://arnaud-carre.github.io/2024-10-06-vpternlogd/">AVX Bitwise ternary logic instruction busted!</a>：现代 AVX 指令如何与 1985 年的 blitter 芯片共享相似设计，作者 Arnaud Carré

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1293001786583679057)** (5 messages): 

> - `LlamaIndex Hackathon`
> - `LlamaParse premium`
> - `Oracle integrations`
> - `LlamaIndex Workflows Tutorial` 


- **LlamaIndex Hackathon 正式启动**：不要错过本周五开始的**第二届 LlamaIndex hackathon**（属于 #SFTechWeek），奖金超过 **$12,000** 美元！
   - 点击[此处](https://t.co/GG7XRnQg5k)报名，学习如何为了乐趣和利润构建复杂的**多 Agent 系统**。
- **LlamaParse Premium 大放异彩**：**LlamaParse premium** 被誉为上下文增强型 LLM 应用的最佳文档解析器，能够处理复杂的文档，如幻灯片和多表 Excel 工作表。
   - 它能高效处理交错的扫描文档以及其他包含大量文本和视觉内容的文档类型，详见此链接：[更多信息](https://t.co/Zd0pWD3wj2)。
- **新的 Oracle 集成上线**：来自 **Oracle** 的重大更新包括**四个新集成**：数据加载器（data loader）、文本分割器（text splitter）、embeddings 和向量搜索（vector search）。
   - 有关这些工具功能的详细更新可以在相应文档中找到，包括对 [data loader](https://t.co/kGud3qKVgO) 的支持。
- **全面的 LlamaIndex Workflows 教程**：查看关于 **LlamaIndex Workflows** 的详细教程，该教程将其与 LangGraph 进行了对比，并指导用户如何入门。
   - 教程还包括构建 AI 研究 Agent 的技巧和调试策略，可在此处访问：[此处](https://t.co/uVJwXeY3lP)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://t.co/kGud3qKVgO">未找到标题</a>：未找到描述</li><li><a href="https://t.co/3nDETnSWJe">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1292943597846532119)** (49 messages🔥): 

> - `Docstore Functionality`
> - `Contextual Retrieval from Anthropic`
> - `Ingestion Pipeline for Qdrant`
> - `DuckDB Vector Store Limitations`
> - `RAG Pipeline Query Handling` 


- **Docstore 可以同时处理 Chunks 和完整文档**：成员们澄清说，**docstore** 可以同时存储 Chunks 和完整文档，因为它们在底层实际上是相同的类。
   - *cheesyfishes* 强调其多功能性允许它无缝地容纳这两种类型。
- **Contextual Retrieval 见解**：*cheesyfishes* 指出，来自 Anthropic 的 Contextual Retrieval 主要涉及 **Metadata 和 Chunk 富化**，并强调了其与现有模型的相似之处。
   - 讨论的一个显著方面是利用 **Prompt caching** 来实现可扩展性，这表明检索机制的方法在持续演进。
- **Qdrant Ingestion Pipeline 用法**：*tharak_* 询问了关于使用 Ingestion Pipeline 直接对处理后的文档进行索引的问题，旨在提高摘要提取和 Embedding 生成等阶段的效率。
   - 讨论反映了通过将索引过程直接集成在 Ingestion 阶段来简化流程，从而增强整体系统性能。
- **DuckDB Vector Store 功能**：关于 **DuckDB Vector Store** 的功能存在一些困惑，特别是关于 add 与 delete 方法，重点在于缺乏 **Upsert** 函数。
   - *whitefang_jr* 解释说 add 方法需要一个 Node 列表，并强调了明确 **Node** 和 **Document** 标识符之间区别的必要性。
- **处理 RAG 管道中的 'And' 查询**：成员们讨论了分解包含 'and' 一词的查询策略，例如将术语分开以进行独立的上下文检索。
   - *tharak_* 建议使用 **Entity recognition** 来区分实体和操作符，以增强 RAG 管道中的查询处理。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs]">未找到标题</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/portkey.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://errors.pydantic.dev/2.9/u/class-not-fully-defined">正在重定向...</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/#nodes">Documents / Nodes - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.cloud.llamaindex.ai/API/add-files-to-pipeline-api-v-1-pipelines-pipeline-id-files-put#:~:text=of%20the%20file-,custom_metadata,-object">Add Files To Pipeline | LlamaCloud Documentation</a>：向管道添加文件。</li><li><a href="https://docs.cloud.llamaindex.ai/API/import-pipeline-metadata-api-v-1-pipelines-pipeline-id-metadata-put">Import Pipeline Metadata | LlamaCloud Documentation</a>：为管道导入 Metadata。</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/storing/docstores/">Document Stores - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/#document-management">Ingestion Pipeline - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/d5b7511a3c51937abf7b21402b826e28de58aabd/llama-index-integrations/vector_stores/llama-index-vector-stores-duckdb/llama_index/vector_stores/duckdb/base.py#L287C21-L287C40">llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-duckdb/llama_index/vector_stores/duckdb/base.py at d5b7511a3c51937abf7b21402b826e28de58aabd · run-llama/llama_index</a>：LlamaIndex 是适用于你的 LLM 应用程序的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1292990449606201375)** (13 条消息🔥): 

> - `TIOBE Index`
> - `Mojo Programming Language`
> - `WebAssembly`
> - `Rust Frontend Frameworks`
> - `Data Attributes in DOM` 


- **TIOBE Index 显示 Mojo 正在崛起**：[2024 年 10 月的 TIOBE index](https://www.tiobe.com/tiobe-index/) 强调了 **Mojo** 已进入前 50 名语言，并突显了对快速、安全且易于学习的编程语言的需求。
   - 鉴于其在一年内的快速上升，Mojo 的表现优于几种成熟的语言，使其成为编程领域中一个极具前景的新竞争者。
- **对 Mojo 未来的期待**：社区成员对 **Mojo** 表达了热情，特别是关于它在 dotAI 闪电演讲中的潜在角色，以及它相比 Python 的吸引力。
   - *Mojo 被视为一种快速的替代方案*，可能会吸引那些以前只对 Python 感兴趣的人。
- **WebAssembly vs JavaScript 之争**：**关于 WebAssembly** 及其访问 DOM 能力的讨论引发了关于它是否可以取代 JavaScript 的辩论。
   - 虽然意见不一，但一位成员指出，为了通过 JavaScript 与 DOM 进行更顺畅的交互，需要解决垃圾回收（Garbage Collection）问题。
- **Rust 前端框架见解**：关于 **Rust frontend frameworks** 如何运作的问题随之而来，并将其与管理 DOM 交互的 JavaScript 胶水代码进行了比较。
   - 这引发了对 Rust 等语言如何通过 JavaScript 进行前端开发接口对接机制的关注。
- **DOM 中 Data Attributes 的利用**：一位用户强调了 DOM 中的一个特性，即数据可以直接作为以 `data-myattribute` 开头的属性存储在元素中。
   - 这种功能提供了一种在 Web 开发中使用自定义数据增强元素的有效方式。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.tiobe.com/tiobe-index/">TIOBE Index - TIOBE</a>: 未找到描述</li><li><a href="https://github.com/modularml/mojo/issues/3623)">Issues · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1292942279161675796)** (19 messages🔥): 

> - `Mojo 关键字重新评估`
> - `ECS 实现挑战`
> - `对 Mojo 提案的反馈`
> - `为初学者展示关键字`
> - `游戏开发讨论` 


- **Mojo 关键字重新评估讨论**：成员们正在讨论是否需要重新考虑 Mojo 中的核心关键字，如 **'inout'** 和 **'borrowed'**，这与一个 [GitHub proposal](https://github.com/modularml/mojo/issues/3623) 相关。
   - 讨论表明，人们希望在 Mojo 引用子系统中建立更清晰的参数约定。
- **在 Mojo 中实现 ECS 的挑战**：一位成员表示在 Mojo 字典中需要一种从类型到索引的映射，以便进行 **ECS** (Entity Component System) 访问。讨论强调 **Mojo** 缺乏对类型内省（introspecting types）的高级支持，使得这项任务变得复杂。
   - 另一位用户建议使用 **Variant**，并指出实现这种映射可能会非常混乱。
- **对 Mojo 关键字想法的反馈**：一位成员针对关于 Mojo 关键字列表的提案提交了实质性反馈，强调了为开发者提供易于获取的概览的重要性。其他人也对这种列表表示支持，强调其对初学者快速学习语言的效用。
   - 有观点认为，将列表限制在一页之内可以显著提升新用户的学习体验。
- **探索 Mojo 中的访问控制术语**：在 **Mojo proposal** 的背景下，一位用户主张采用 **access control**（访问控制）术语，以帮助不太熟悉 **Rust/C++** 的程序员理解。这一反馈符合改善编码环境用户体验的目标。
   - 建议使用此类术语来简化所有用户对代码结构和权限的推理。
- **游戏开发社区参与**：针对 ECS 的讨论，一位成员提议联系 **Mojo community** 中对游戏开发感兴趣的人。他们鼓励正在开发游戏相关项目的成员之间进行协作和想法分享。
   - 尽管有些成员希望避开 ECS，但社区内的联网和知识共享机会受到了热烈欢迎。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.python.org/3/reference/lexical_analysis.html#keywords">2. Lexical analysis</a>：Python 程序由解析器读取。解析器的输入是由词法分析器生成的令牌流。本章描述了词法分析器如何将文件分解为令牌。Python...</li><li><a href="https://github.com/modularml/mojo/issues/3623">[Discuss] Resyntaxing argument conventions and References · Issue #3623 · modularml/mojo</a>：Mojo 引用子系统的设计正趋于完善。为了敲定主要点，重新评估 Mojo 中的几个早期决策有助于使设计更加...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1292942271267868683)** (13 messages🔥): 

> - `Max 推理引擎问题`
> - `自定义算子教程`
> - `PyTorch 版本兼容性`
> - `图编译时间` 


- **Max 推理引擎故障排除**：一位用户报告了在他们的 Intel NUC 上使用 **max inference engine** 时遇到的问题，在 **TorchScript** 和 **ONNX** 路径上都遇到了错误。
   - 经过讨论，通过切换到官方 PyTorch 频道并使用 2.4 以下的版本找到了解决方案。
- **自定义算子（Custom Ops）正在经历变更**：有人询问在最近的更新后 **custom ops tutorial** 的位置，其中包括创建自定义 Gelu 算子的示例。
   - 据悉 **custom ops 正在重构中**，这影响了相关文档的可用性。
- **讨论图编译挑战**：针对在执行多个张量操作时 **graph compilation**（图编译）耗时的问题提出了担忧，估计成本约为 **400-500 ms**。
   - 建议创建一个通用的 reshape 操作以便重复使用，从而减少图创建过程中的开销。
- **PyTorch 版本冲突已解决**：一位用户澄清了他们通过 conda-forge 频道安装的 **PyTorch** 导致了与 max 推理引擎的兼容性问题。
   - 切换到官方频道并满足版本限制后解决了这些问题。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1293163567088144394)** (5 messages): 

> - `2024 Nobel Prize in Physics`
> - `OpenAI's Compute Capacity`
> - `Microsoft Competition` 


- **神经网络荣获诺贝尔奖**：瑞典皇家科学院宣布将 **2024年诺贝尔物理学奖** 授予 **John J. Hopfield** 和 **Geoffrey E. Hinton**，以表彰他们在利用 **人工神经网络** 实现机器学习方面的开创性工作。这一认可凸显了他们对该领域做出重大贡献的 **基础性发现和发明**。
   - 社区对这一享有盛誉的认可表示：“*非常温馨 (very wholesome)*”。
- **OpenAI 确保独立算力**：由于担心 **Microsoft 的响应时间**，OpenAI 开始 **确保自身的算力容量**，并与 Microsoft 的竞争对手签署了数据中心协议。首席财务官 (CFO) **Sarah Friar** 指出，Microsoft 在提供必要算力方面的 **行动不够迅速**。
   - 该公告引发了关于竞争格局的评论，包括有人认为考虑到 Microsoft 的信任问题，此举虽然 **劲爆但并不令人意外**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/NobelPrize/status/1843589140455272810">来自诺贝尔奖 (@NobelPrize) 的推文</a>: 突发新闻：瑞典皇家科学院决定将 2024 年 #NobelPrize 物理学奖授予 John J. Hopfield 和 Geoffrey E. Hinton，以表彰他们在基础性发现和发明方面的贡献...</li><li><a href="https://x.com/anissagardizy8/status/1843647826859044945">来自 Anissa Gardizy (@anissagardizy8) 的推文</a>: 独家：OpenAI 开始确保自身的算力容量，与 MSFT 的竞争对手签署数据中心协议。CFO Sarah Friar 表示 MSFT 在向公司提供算力方面行动不够迅速...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1292962395446313070)** (14 messages🔥): 

> - `Llama 3.2 11B vs 8B performance`
> - `Vision integration in text models`
> - `Research on PRMs/Verifiers`
> - `State-of-the-art audio models` 


- **8B 模型在文本任务上优于 11B**：讨论表明，与主要为处理图像而设计的 **11B Vision** 模型相比，**8B** 模型在 **纯文本** 任务中可能表现更好。
   - *所有的增加部分都是为了处理图像*，这暗示了在文本性能上存在权衡。
- **视觉增强可能提升文本模型**：一位参与者指出，如果 **Text Backbone** 在训练期间保持非冻结状态，添加 **Vision Backbone** 可能会提升纯文本性能。
   - 关于这种融合中 **3B 参数** 的分配产生了疑问，凸显了集成方法的不确定性。
- **关于 LRMs 和规划性能的辩论**：一个分享的帖子讨论了像 **o1** 这样的 **LRMs** 在增强规划任务性能方面的有效性，尽管每次调用的成本更高。
   - 值得注意的是，据报道准确率有所提高，例如在进行 Back Prompting 后，在硬积木世界 (hard blocks world) 实例上的准确率从 **24% 提升到了 98%**。
- **询问 SOTA 音频模型**：一位成员提出了关于 **State-of-the-art (SOTA) 音频模型** 使用体验的问题，寻求社区的见解。
   - 另一位成员建议联系 Twitter 上的 **reach_vb** 以获取该领域的专家建议。
- **关于 PRMs 的研究有限**：一位用户指出关于 **PRMs** 的论文非常稀缺，这与关于 **LLMs as judges** 的丰富资源形成了鲜明对比。
   - 面对海量的 LLM 文献，该用户幽默地评论道：“*选一个你喜欢的毒药吧 (Choose your poison)*”。



**提到的链接**: <a href="https://x.com/rao2z/status/1843307760311533768">来自 Subbarao Kambhampati (@rao2z) 的推文</a>: LRM-Modulo? 我们对 o1 的初步实验表明，虽然像它这样的 LRMs 确实似乎提升了规划问题的性能底线，但它们距离鲁棒性还很远。一个想法是将 LRMs 视为...

  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1293220387286683751)** (14 条消息🔥): 

> - `质疑声明`
> - `AI 研究中的能源消耗`
> - `内部争议`
> - `Emma 与 Jeff 的历史纠葛` 


- **Jeff Dean 尖锐的帖子引发关注**：Jeff Dean 最近对 AI 论文中能源排放声明的批评引发了辩论，一些人认为这可能过于激进，从而引发了对其论点有效性的讨论。
   - 有人担心发布这些批评是否为行业过度使用能源提供了 *casus belli*（开战理由），并质疑了这在问责制方面的含义。
- **对 Emma 对 Jeff 的反应感到惊讶**：众人对 Emma 对 Jeff 言论的反应表示惊讶，暗示她觉得 Jeff 试图抹黑她的工作，尽管她通常表现得很理智。
   - 一些讨论强调了两人之间紧张关系的潜在历史背景，暗示之前的互动可能影响了当前的看法。
- **内部数据引发怀疑**：提到了 Jeff Dean 可以访问内部指标，微妙地暗示这使他在论证中占据优势，但也引发了关于透明度的质疑。
   - 这导致了对基于专有数据与外部评估所做声明的合法性的复杂感受。
- **对外交辞令式回应的抵制**：试图以外交辞令回应 Jeff 声明的努力遭到了负面对待，包括一封匿名邮件对接受 Jeff 观点的行为表示难以置信。
   - 这种抵制标志着围绕该讨论的极化观点，反映了社区内更深层次的分歧。
- **关于能源研究的总体情绪**：关于 AI 研究中能源消耗的总体情绪依然暗淡，一位成员指出这是一个 *“令人沮丧的研究领域”*，但承认需要继续探索。
   - 这种看法说明了在没有充分解决方案或共识的情况下，应对 AI 环境影响所面临的挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/JeffDean/status/1843493504347189746">Jeff Dean (@🏡) (@JeffDean) 的推文</a>: 我只是在看时间线：2019 年 - Strubell 等人的论文 (https://arxiv.org/abs/1906.02243) 在评估 Evolved Transformer 神经架构搜索的排放成本时出现了诚实的错误...</li><li><a href="https://x.com">GitHub - FixTweet/FxTwitter 的推文</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等 - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1292950218794340467)** (6 条消息): 

> - `玩具功能`
> - `采样技术`
> - `AI 可解释性` 


- **玩具功能令人印象深刻但仍需调整**：一位用户评论说新玩具（toys）令人印象深刻，尽管他们不确定适配度是否完美。
   - 他们计划继续试用并分享经验，以评估其性能。
- **采样见解与行业认知**：讨论强调，大公司的许多人将 **sampling**（采样）视为黑盒，主要关注 **beam/nucleus** 方法而缺乏适当的探索。
   - 这引发了关于需要更好的采样技术的评论，以及 **Bayesians**（贝叶斯主义者）如何特别关注这一话题。
- **AI 模型的可解释性受到关注**：一篇分享的博客文章强调了在 LLM 及其推理能力的背景下，**explainability**（可解释性）日益增长的重要性。
   - 该文章讨论了产品形态从**单个任务**向更复杂的**系统级生产力**的演变，强调了对可审计推理的需求。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/giffmana/status/1843359235792613807">Lucas Beyer (bl16) (@giffmana) 的推文</a>: @_xjdr @waterruupto @doomslide 此外，你会惊讶于大公司的许多人只将采样视为黑盒，并认为 beam/nucleus 就是全部。大多数成熟的代码库对此并无帮助...</li><li><a href="https://www.normalcomputing.com/blog-posts/explainable-language-models-existing-and-novel-approaches">Normal Computing</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1292936556629459017)** (31 messages🔥): 

> - `Discord 体验问题`
> - `推荐奖励周边公告`
> - `Perplexity 中的图像生成`
> - `Perplexity 网页端 vs 移动端性能`
> - `Perplexity 盈利担忧` 


- **成员面临 Discord 移除问题**：一位用户对被移出 Discord 表示沮丧，质疑这是否是一种 *psyop*，其他人也纷纷提到性能问题。
   - 几位用户指出，不同设备和平台上的体验差异很大。
- **咨询推荐奖励的周边商品**：一位新人询问有关与推荐（referrals）挂钩的周边商品公告，寻求对当前优惠活动的澄清。
   - 聊天中未提及任何公告，导致对潜在奖励的进一步猜测。
- **关于图像生成能力的疑问**：一名成员询问有关使用 Perplexity 生成图像的信息，特别是针对 Pro 用户，但发现应用中缺乏该功能。
   - 相关讨论指出，功能在不同平台受限导致了使用困难。
- **Discord 的性能引发关注**：多位用户报告在浏览器上使用 Perplexity 时性能缓慢，并表示在移动端运行更顺畅。
   - 对桌面端易用性挑战的挫败感增加，引发了关于潜在修复方案的询问。
- **Perplexity 的盈利能力受到质疑**：一位用户质疑 Perplexity 在提供学生折扣和实惠计划的情况下如何产生利润。
   - *担忧集中在* 这种定价模式对于 Perplexity 未来的财务健康是否具有可持续性。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1293081748250562603)** (6 messages): 

> - `中国的声激光`
> - `5MVA 设计`
> - `小型电路设计`
> - `Cerebras IPO 挑战`
> - `生成描述` 


- **中国发布世界上最强大的声激光**：最近的一段视频强调 **中国** 已经开发出 **世界上最强大的声激光**，展示了其创新技术。
   - 您可以在[这里](https://www.youtube.com/embed/LbtAFX7Pg6M)观看视频。
- **设计 5MVA Dat**：一位用户寻求关于如何有效 **设计 5MVA Dat** 的指导，引发了关于适当方法的讨论。
   - 更多详情，请查看[此处](https://www.perplexity.ai/search/if-i-want-to-design-a-5mva-dat-XoIDVRYTQO.oJosPfZlo4Q)的对话。
- **创建一个小型电路**：有人询问如何 **创建一个小型电路**，促使用户分享他们的设计和方法。
   - 讨论可以在[这里](https://www.perplexity.ai/search/come-creo-un-piccolo-circuito-3jChpBLgS56Tm0Xj_jXUWA)找到。
- **Cerebras 在 IPO 中面临来自 Nvidia 的挑战**：一篇文章讨论了 Cerebras 在其 IPO 中可能面临的 **挑战**，特别是与 **Nvidia** 的竞争。
   - 在[这里](https://www.perplexity.ai/page/cerebras-ipo-challenges-nvidia-LmwxVQHLRa.VXzSMV4Ubkw)阅读更多相关内容。
- **寻求有用的描述生成**：两位用户询问了 **生成有用描述** 的方法，表明了对实用内容创建工具的需求。
   - 相关讨论可以从[这里](https://www.perplexity.ai/search/generate-a-useful-description-gIIt3d80RJShcR4yalXahw)进入。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1292995293549432893)** (2 messages): 

> - `Rate Limit 提升请求`
> - `支持响应问题` 


- **请求提高 Rate Limit**：一名成员询问如何请求增加 Rate Limit 使用额度，并提到他们已多次向支持部门发送邮件但未收到回复。
   - 他们表达了请求的紧迫性，寻求有人协助升级处理此问题。
- **关于支持联系方式的澄清**：另一名成员询问发帖人是否向正确的支持地址发送了邮件，特别是 api@perplexity.ai 或通用支持邮箱。
   - 这表明沟通中可能存在疏忽，提示需要更清晰的寻求支持说明。


  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1293285248981008424)** (1 messages): 

> - `工具创建`
> - `Assistant 开发` 


- **创建用于创建工具的工具**：一名成员强调了在未来开发中创建 **工具的工具 (tools that create tools)** 的重要性。
   - 这种方法旨在提高社区内的 **效率** 并促进创新。
- **创建 Assistant 的 Assistant**：讨论围绕开发 **可以创建其他 Assistant 的 Assistant** 展开，呼应了自动化领域日益增长的趋势。
   - 这种 **元开发 (meta-development)** 可能会在生产力和功能方面带来重大进步。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1292962561888882719)** (13 条消息🔥): 

> - `Custom LM vs Custom Adapter`
> - `LM Clients 的弃用`
> - `LM 配置与 Adapters`
> - `Optimizer 与 Adapter 问题`
> - `DSPy 社区交流` 


- **澄清 Custom LM 和 Adapter 的用法**：成员们讨论了记录选择 **custom Adapter** 而非 **custom LM** 原因的价值，特别指出了在使用不同 LM 配置时选择正确模型的挑战。
   - 建议评估现有的 [language models 文档](https://dspy-docs.vercel.app/docs/building-blocks/language_models) 在此问题上的清晰度。
- **Custom LM Clients 的弃用通知**：**DSPy 2.5** 已弃用除 `dspy.LM` 之外的所有自定义 LM 客户端，这些客户端将在 **DSPy 2.6** 中逐步淘汰。建议迁移到 `dspy.LM`，以提高与 **Adapters** 等新功能的一致性。
   - 分享了一个迁移 [notebook](https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb) 链接，指导用户过渡到更新后的标准。
- **理解 Adapters 中的 LM 配置**：有人提出了关于 `lm_kwargs` 在 **MIPROv2 optimizer** 中未被填充的问题，导致了对这是否为预期行为的困惑。
   - 另一位成员澄清，除非向 **predictor** 传递了特定配置，否则 `lm.kwargs` 将包含来自 LM 的 kwargs。
- **来自 DSPy 社区的参与**：成员们进行了自我介绍并分享了见解，其中一位表示他们将尝试使用 optimizers 并提供反馈。
   - 进行了友好的交流，确认了对 **DSPy** 社区正在进行的努力和贡献感到兴奋。



**提到的链接**：<a href="https://dspy-docs.vercel.app/docs/deep-dive/language_model_clients/custom-lm-client">Creating a Custom Local Model (LM) Client | DSPy</a>: ---

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1292946157890310195)** (13 条消息🔥): 

> - `Open-Interpreter Tool Calling`
> - `工具的 Structured Output`
> - `Mozilla AI 演讲公告` 


- **Open-Interpreter Tool Calling 详解**：一位成员询问 **Open-Interpreter** 如何保持确定性且准确的 tool calling，得到的解释是，由于 LLMs 的存在，它并非纯粹的确定性，但由于 system message，它在很大程度上是一致的。
   - *Mikebirdtech* 确认虽然它使用了 LLMs，但 system message 有助于实现一定程度的一致性。
- **探索自定义工具的 Structured Output**：另一位成员建议探索用于自定义 tool calling 的 **structured output**，强调了其未被开发的潜力和过去的实验。
   - 大家达成共识，认为来自 **Ollama** 和 **llamacpp** 等工具的进一步研究和改进支持将促进其实现。
- **即将举行的 Mozilla AI 演讲提醒**：**Mikebirdtech** 兴奋地提醒成员们，即将由 **Mozilla AI** 的代表进行一场专注于开源倡议的演讲，并表示这将非常有趣。
   - 活动定于下周举行，鼓励成员们不要错过，并提供了 Discord 活动链接。


  

---



### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1292950587461206077)** (6 条消息): 

> - `线下参加讲座`
> - `使用 Autogen 的 AI agent 初创公司`
> - `使用 Redis 构建框架` 


- **线下讲座出席限制**：由于教室容量限制，仅邀请 **Berkeley 学生** 线下参加讲座。
   - 一位用户询问是否可以线下参加，但被告知了出席限制。
- **关于在生产环境中使用 Autogen 的辩论**：一位成员询问其他 AI agent 初创公司是在生产环境中使用 **Autogen**，还是更倾向于使用带有原始 API 调用（raw API calls）的扁平化 Python 文件。
   - 对话围绕在实际应用中优化 AI agents 的实现展开。
- **使用 Redis 开发自定义框架**：一位成员提到正在使用 **Redis** 构建自己的框架，连接节点（workers）以增强功能。
   - 他们澄清这种方法旨在**减少抽象**，并实现对复杂用例更好的控制。
- **AI 框架中的状态控制**：用户强调开发框架是为了**更好的控制**，并通过管理状态（memory）来解决更复杂的用例。
   - 这反映了关于 AI 应用架构和效率的持续讨论。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1292973040207921182)** (1 messages): 

> - `Omar 的 DSPy 讲座`
> - `对 DSPy 的贡献` 


- **Omar 令人兴奋的 DSPy 讲座**：一位成员对即将到来的由 **Omar** 主讲的 **DSPy** 讲座表示热烈期待，强调了其重要性。
   - *“对这门讲座感到兴奋”* 体现了对 **DSPy** 框架近期发展的浓厚兴趣。
- **正在对 DSPy 做出贡献**：同一位成员正积极使用 **DSPy**，并旨在为其开发和改进做出贡献。
   - 他们的投入展示了增强 **DSPy** 工具能力的强烈兴趣。


  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1293046050155790399)** (5 messages): 

> - `tinygrad 网站导航`
> - `Exo 悬赏挑战`
> - `tinygrad 文档` 


- **用户对 tinygrad 网站导航的担忧**：一位成员指出，除非点击一个小按钮，否则大多数用户可能不会进入某个特定页面，这表明 **tinygrad** 网站可能存在导航问题。
   - 他们随后重新考虑，确认如果用户点击该按钮，确实会被引导至该页面。
- **关于 exo 悬赏挑战的讨论**：一位用户正在尝试来自 **exo** 的悬赏任务，旨在将 **tinygrad** 编译为 **Swift**，并分享了 [GitHub issue](https://github.com/exo-explore/exo/issues/238) 链接作为参考。
   - 他们表达了希望保持 **exo** 的 **Python** 基础，同时寻求管理员或贡献者在解决该问题上的指导。
- **George Hotz 强调文档的重要性**：George Hotz 建议其他人 *“阅读问题文档”*，强调了查阅现有资源的重要性。
   - 这一指导强调了一个共同的主题，即引导用户查阅详尽的文档以获取清晰解答。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/exo-explore/exo/issues/238">[悬赏 - $1000] 将 tinygrad 编译为 swift · Issue #238 · exo-explore/exo</a>：如果可能，我希望保持 exo 100% 使用 python。希望在 tinygrad 中编译 swift 推理代码。此处的交付成果是 tinygrad 中一个已合并的 PR，以及在 exo 中演示如何实现此功能的简短示例...</li><li><a href="https://docs.tinygrad.org/">tinygrad 文档 - tinygrad docs</a>：未找到描述
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1293011160903188613)** (1 messages): 

> - `Buffer 计数变通方案`
> - `Tensor 操作中的低效问题` 


- **Tensor.sum() 问题的变通方案**：利用 **qazalin 的额外 buffer 计数 PR** 创建了一个变通方案，以绕过由 **Tensor.sum()** 引起的错误，该错误在一次性加载过多 **buffer** 时会出现问题。
   - 尽管在当前版本中被认为 **非常低效**，但 *“迭代地添加和拆分操作”* 对于绕过该问题是必要的。
- **范数计算调整**：该脚本通过计算 **norms**（范数）、对其求平方并迭代求和来处理梯度，以便更好地管理内存。
   - 这种方法涉及创建 **norm1_squared** 和 **norm2_squared** 组，以牺牲效率为代价增强了稳定性。


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1292932406038233149)** (3 messages): 

> - `出行担忧`
> - `ChatPromptTemplate 用法`
> - `无效的 JSON 格式` 


- **出行计划存疑**：一位成员表示有兴趣参加活动，但不确定届时是否能够出行。
   - 这种担忧凸显了涉及出行时在日程安排和承诺方面的挑战。
- **使用 ChatPromptTemplate 进行消息传递**：一位成员分享了他们使用 `ChatPromptTemplate` 配合 **few-shot prompt** 设置在聊天应用中生成消息的实现。
   - 他们详细说明了如何为聊天模型构建 `example_prompt` 和 `example_selector`。
- **双引号转义导致的 JSON 问题**：另一位成员报告了其 `messages` 对象的一个问题，即双引号被替换为 `&quot;`，导致 **JSON** 格式无效。
   - 他们寻求关于如何防止这种转义发生的建议，以便向聊天接口发送有效的 **JSON**。


  

---

### **LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1293187678250074213)** (1 messages): 

> - `Escaping Quotes in Messages` (消息中的引号转义)
> - `ChatPromptTemplate Usage` (ChatPromptTemplate 用法)
> - `FewShotChatMessagePromptTemplate` 


- **消息中的引号转义导致 JSON 问题**：一位用户报告称，他们的 `messages` 对象中所有双引号都被转义为 `&quot;`，导致发送到聊天时 JSON 格式无效。
   - 他们询问如何禁用这种转义以保持原始格式。
- **有效使用 ChatPromptTemplate**：用户分享了他们使用 `ChatPromptTemplate.from_messages()` 的实现，以指定模板格式来格式化聊天消息。
   - 这种方法包括 Human 和 AI 消息配置，以便更好地进行模板管理。
- **集成 FewShotChatMessagePromptTemplate**：用户演示了如何使用示例选择器 (example selector) 和示例提示词 (example prompt) 设置 `FewShotChatMessagePromptTemplate`。
   - 该集成旨在增强输入处理并改善聊天的上下文响应。


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/)** (1 messages): 

gustaf_81960_10487: <@387269332142391298> 更新你的证书！
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1292926102632923236)** (4 messages): 

> - `BF16 Training Challenges` (BF16 训练挑战)
> - `Learning Rate Adjustments` (学习率调整)
> - `Stochastic Rounding in Optimizers` (优化器中的随机舍入)


- **BF16 训练问题值得关注**：一位成员强调，调整 **learning rate** (LR) 的必要性与全 **BF16** 训练有关，因为如果变化太小，**BF16 权重**可能无法正确更新。
   - 该成员建议采用 **BF16 混合精度训练**来缓解此问题，尽管由于额外的 **FP32 梯度**和**优化器状态**会导致更高的显存占用。
- **理解 1B 模型中的 BF16 效应**：有人提出疑问，为什么 **BF16** 的影响在 **1B 模型**中显得更显著，并猜测参数较少可能意味着更新的内容较少。
   - 另一位成员指出，**BF16 权重更新下溢**受 `weight` 和 `weight_delta` 之间关系的影响，并建议针对 **BF16 混合精度训练**结果进行验证。
- **实验随机舍入 (Stochastic Rounding)**：一位成员表示有兴趣在优化器的权重更新中加入 **stochastic rounding**，以评估其对 **Torchtune** 的实用性。
   - 他们表示愿意进行实验，权衡实验收益与潜在的复杂性。


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1293231079615893525)** (2 messages): 

> - `Geoffrey Hinton Nobel Comparison` (Geoffrey Hinton 诺贝尔奖对比)
> - `Model Merging at Scale` (大规模模型合并)


- **对 Geoffrey Hinton 未来获得诺贝尔奖的看法**：50 年后，授予 **Geoffrey Hinton** 诺贝尔奖将被视为与今天看待 1949 年授予 **Moniz** 脑白质切除术（因其在*精神病治疗价值*）诺贝尔奖的观点类似。
   - *Hinton 深刻地误解了现代机器学习*，这表明他与当前的进展脱节。
- **关于大规模模型合并 (Model Merging) 的见解**：来自 Google 实习生的**新研究**讨论了涉及高达 **640 亿参数**模型的大规模模型合并场景。
   - 该研究探讨了**模型大小**、合并方法和专家数量如何影响**域内性能 (held-in performance) 和泛化能力**，研究结果详见其 [arXiv 论文](https://arxiv.org/abs/2410.03617)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/prateeky2806/status/1843643582432854171">来自 Prateek Yadav (@prateeky2806) 的推文</a>：有没有想过模型合并在大规模下是否有效？也许对于更大的模型，收益会消失？也许你考虑过将模型合并用于大型模型的训练后处理，但不确定它是否能泛化...</li><li><a href="https://x.com/JJitsev/status/1843612156170051591">来自 Jenia Jitsev 🏳️‍🌈 🇺🇦 (@JJitsev) 的推文</a>：从现在起 50 年后，授予 Geoffrey Hinton 诺贝尔奖将被视为与今天看待脑白质切除术诺贝尔奖一样，Moniz 在 1949 年因“发现白质切断术的治疗价值”而获奖...
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1292931402517446677)** (2 条消息): 

> - `Model Merging at Scale`
> - `Autoarena Tool` 


- **探索大规模模型合并 (Exploring Model Merging at Scale)**：Google 的最新研究调查了大规模 **model merging**，研究了参数量高达 **64B parameters** 的语言模型组合，以及各种因素如何影响性能和泛化能力。
   - 正如这个 [thread](https://x.com/prateeky2806/status/1843643582432854171) 中提到的，该研究探讨了关于合并更大模型的有效性及其影响的重要问题。
- **介绍 Autoarena 工具**：一位用户重点介绍了一个名为 **Autoarena** 的有趣工具，访问地址为 [autoarena.app](https://www.autoarena.app/)，该工具似乎为用户提供了实用的功能。
   - 该工具引发了好奇，表明它可能为其预期应用提供创新的解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/prateeky2806/status/1843643582432854171">来自 Prateek Yadav (@prateeky2806) 的推文</a>: 有没有想过模型合并在大规模下是否有效？也许对于更大的模型，其收益会逐渐消失？也许你考虑过将模型合并用于大型模型的 post-training，但不确定它是否能泛化...</li><li><a href="https://www.autoarena.app/">AutoArena</a>: 未找到描述
</li>
</ul>

</div>
  

---



---



---



---



---



---



---



---



{% else %}


> 完整的逐频道细分内容已针对电子邮件进行截断。
> 
> 如果你想查看完整的细分内容，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}